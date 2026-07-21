// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Expander.h
 * @brief Downward expander with configurable ratio, hysteresis, and sidechain.
 *
 * A generalization of the noise gate: instead of fully closing (infinite ratio),
 * the expander applies a configurable ratio below the threshold. Includes an
 * integrated envelope detector to prevent low-frequency intermodulation
 * distortion, a true-stereo sidechain high-pass filter, and an external
 * sidechain input (processBlock(audio, sidechain) keys the detector from the
 * sidechain signal; a sidechain shorter than the audio block falls back to the
 * internal key so the caller's buffer is never over-read).
 *
 * Threading model:
 * - prepare() / reset() / getState() / setState(): setup or UI threads only,
 *   never concurrently with processBlock().
 * - All parameter setters are RT-safe atomic publications from any thread;
 *   the audio thread picks them up at the next block.
 * - getGateState() / getCurrentGainDb() are approximate cross-thread metering
 *   reads (unsynchronized).
 * - processBlock() before prepare() is a pass-through (it used to hard-mute:
 *   the smoothing coefficients were still zero, freezing the gain at 0).
 *
 * Channels beyond the 16-channel detector cap still receive the (scalar)
 * expander gain; only the per-channel sidechain HPF state is capped.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h,
 *               StateBlob.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class Expander
 * @brief Downward expander with ratio control, hysteresis, and sidechain.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Expander
{
public:
    ~Expander() = default; // non-virtual: leaf class (no virtual dispatch)

    enum class State { Closed, Open, Hold };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Initializes the expander with a specific sample rate.
     * @param sampleRate The sample rate in Hz. Non-finite or non-positive
     *                   rates are ignored (previous state is kept).
     */
    void prepare(double sampleRate) noexcept
    {
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0) return;
        sampleRate_ = sampleRate;
        updateCoefficients();
        // Re-derive the hold counter for the (possibly new) rate: it used to
        // be computed only inside setHold() with the rate of that moment, so
        // the default 50 ms hold was silently ZERO until setHold was called,
        // and a re-prepare at a new rate kept stale hold lengths.
        setHold(holdMs_.load(std::memory_order_relaxed));
        reset();
        prepared_ = true;
    }

    /**
     * @brief Initializes the expander using an AudioSpec struct.
     * @param spec Framework audio specification object (invalid specs are ignored).
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid()) return;
        prepare(spec.sampleRate);
    }

    /**
     * @brief Processes an audio block in place (internal detector key).
     * @param buffer View of the audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        cacheParams();

        if (cachedScHpfEnabled_)
            processBlockInternal<true, false>(buffer, nullptr);
        else
            processBlockInternal<false, false>(buffer, nullptr);
    }

    /**
     * @brief Processes an audio block in place, keyed by an external sidechain.
     *
     * The detector (and its optional HPF) runs on the sidechain channels; the
     * resulting gain is applied to the audio buffer. If the sidechain is empty
     * or shorter than the audio block, the internal key is used instead.
     *
     * @param buffer    Audio to process in place.
     * @param sidechain Detector key signal (not modified).
     */
    void processBlock(AudioBufferView<T> buffer, AudioBufferView<T> sidechain) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        cacheParams();

        if (sidechain.getNumChannels() <= 0 ||
            sidechain.getNumSamples() < buffer.getNumSamples())
        {
            if (cachedScHpfEnabled_)
                processBlockInternal<true, false>(buffer, nullptr);
            else
                processBlockInternal<false, false>(buffer, nullptr);
            return;
        }

        if (cachedScHpfEnabled_)
            processBlockInternal<true, true>(buffer, &sidechain);
        else
            processBlockInternal<false, true>(buffer, &sidechain);
    }

    // -- Parameters ----------------------------------------------------------
    // All setters ignore non-finite values (NaN attack/release/ratio used to
    // poison the gain smoother permanently through max()'s first-argument
    // NaN pass-through).

    /** @brief Sets the threshold in decibels. */
    void setThreshold(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        threshold_.store(dB, std::memory_order_relaxed);
    }

    /** @brief Sets the expansion ratio (e.g., 4.0 for 4:1). */
    void setRatio(T ratio) noexcept
    {
        if (!std::isfinite(ratio)) return;
        ratio_.store(std::max(ratio, T(1)), std::memory_order_relaxed);
    }

    /** @brief Sets the hysteresis gap in decibels to prevent chattering. */
    void setHysteresis(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        hysteresis_.store(std::max(dB, T(0)), std::memory_order_relaxed);
    }

    /** @brief Sets the maximum gain reduction limit in decibels (<= 0). */
    void setRange(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        const T r = std::min(dB, T(0));
        rangeDb_.store(r, std::memory_order_relaxed);
        rangeLinear_.store(decibelsToGain(r), std::memory_order_relaxed);
    }

    /** @brief Sets the attack time in milliseconds. */
    void setAttack(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /** @brief Sets the hold time in milliseconds before release begins. */
    void setHold(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        holdMs_.store(std::max(ms, T(0)), std::memory_order_relaxed);
        if (sampleRate_ > 0.0)
        {
            // Capped in double before the int cast (a huge finite hold time
            // would otherwise be a UB float-to-int conversion).
            const double h = std::min(
                sampleRate_ * static_cast<double>(holdMs_.load(std::memory_order_relaxed)) / 1000.0,
                1.0e9);
            holdSamples_.store(static_cast<int>(h), std::memory_order_relaxed);
        }
    }

    /** @brief Sets the release time in milliseconds. */
    void setRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /**
     * @brief Enables and configures the sidechain high-pass filter.
     *
     * Invalid cutoffs (non-finite or <= 0) keep the previous frequency and
     * only apply the toggle; the design clamp is [1, 0.45 * sampleRate]. A
     * negative cutoff used to flip the one-pole into an unstable growth
     * filter, blowing up the detector and pinning the expander open.
     *
     * @param enabled  True to engage the HPF.
     * @param cutoffHz Cutoff frequency in Hz.
     */
    void setSidechainHPF(bool enabled, double cutoffHz = 80.0) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        if (std::isfinite(cutoffHz) && cutoffHz > 0.0)
            scHpfFreqHz_.store(static_cast<T>(cutoffHz), std::memory_order_relaxed);
        updateCoefficients();
    }

    // -- Queries -------------------------------------------------------------

    /** @return The current state of the expander's logic gate (approximate
     *  cross-thread metering read).
     *  (Renamed from getState(), which now follows the framework-wide preset
     *  serialization convention.) */
    [[nodiscard]] State getGateState() const noexcept { return state_; }

    /** @return The current actual gain being applied, in decibels (approximate
     *  cross-thread metering read). */
    [[nodiscard]] T getCurrentGainDb() const noexcept { return gainToDecibels(gateGain_); }

    /** @brief Resets all internal DSP states to prevent clicks on playback start. */
    void reset() noexcept
    {
        state_ = State::Closed;
        gateGain_ = rangeLinear_.load(std::memory_order_relaxed);
        envelope_ = T(0);
        holdCounter_ = 0;

        std::fill(scHpfState_.begin(), scHpfState_.end(), T(0));
        std::fill(scHpfPrev_.begin(), scHpfPrev_.end(), T(0));
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("XPND"), 1);
        // static_cast<float>: the blob stores float, and StateWriter's
        // overload set is ambiguous for a double argument.
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        w.write("ratio", static_cast<float>(ratio_.load(std::memory_order_relaxed)));
        w.write("hysteresis", static_cast<float>(hysteresis_.load(std::memory_order_relaxed)));
        w.write("attack", static_cast<float>(attackMs_.load(std::memory_order_relaxed)));
        w.write("hold", static_cast<float>(holdMs_.load(std::memory_order_relaxed)));
        w.write("release", static_cast<float>(releaseMs_.load(std::memory_order_relaxed)));
        w.write("rangeDb", static_cast<float>(rangeDb_.load(std::memory_order_relaxed)));
        w.write("scHpf", scHpfEnabled_.load(std::memory_order_relaxed));
        w.write("scHpfFreq", static_cast<float>(scHpfFreqHz_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("XPND")) return false;
        setThreshold(static_cast<T>(r.read("threshold", -40.0f)));
        setRatio(static_cast<T>(r.read("ratio", 4.0f)));
        setHysteresis(static_cast<T>(r.read("hysteresis", 4.0f)));
        setAttack(static_cast<T>(r.read("attack", 0.5f)));
        setHold(static_cast<T>(r.read("hold", 50.0f)));
        setRelease(static_cast<T>(r.read("release", 100.0f)));
        setRange(static_cast<T>(r.read("rangeDb", -80.0f)));
        setSidechainHPF(r.read("scHpf", false),
                        static_cast<double>(r.read("scHpfFreq", 80.0f)));
        return true;
    }

protected:
    void cacheParams() noexcept
    {
        cachedThreshold_    = threshold_.load(std::memory_order_relaxed);
        cachedRatio_        = ratio_.load(std::memory_order_relaxed);
        cachedHysteresis_   = hysteresis_.load(std::memory_order_relaxed);
        cachedRangeLinear_  = rangeLinear_.load(std::memory_order_relaxed);
        cachedAttackCoeff_  = attackCoeff_.load(std::memory_order_relaxed);
        cachedReleaseCoeff_ = releaseCoeff_.load(std::memory_order_relaxed);
        cachedHoldSamples_  = holdSamples_.load(std::memory_order_relaxed);
        cachedScHpfEnabled_ = scHpfEnabled_.load(std::memory_order_relaxed);
        cachedDetAttCoeff_  = detAttCoeff_.load(std::memory_order_relaxed);
        cachedDetRelCoeff_  = detRelCoeff_.load(std::memory_order_relaxed);
        cachedScHpfCoeff_   = scHpfCoeff_.load(std::memory_order_relaxed);
        cachedScHpfA0_      = scHpfA0_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Internal processing loop templated on HPF/sidechain to avoid branching.
     * @tparam UseHPF Compile-time flag to include HPF processing.
     * @tparam UseSc  Compile-time flag: detector keyed by the external sidechain.
     * @param buffer    View of the audio buffer (gain target).
     * @param sidechain Detector key when UseSc (guaranteed >= buffer length).
     */
    template <bool UseHPF, bool UseSc>
    void processBlockInternal(AudioBufferView<T> buffer,
                              const AudioBufferView<T>* sidechain) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        // Detector channel count is capped by the per-channel HPF state
        // arrays; the scalar gain below is applied to ALL audio channels.
        const int detCh = UseSc ? std::min(sidechain->getNumChannels(), MAX_CHANNELS)
                                : std::min(nCh, MAX_CHANNELS);

        for (int i = 0; i < nS; ++i)
        {
            T sc = T(0);

            // 1. Calculate Sidechain Signal
            for (int ch = 0; ch < detCh; ++ch)
            {
                T input = UseSc ? sidechain->getChannel(ch)[i]
                                : buffer.getChannel(ch)[i];
                if constexpr (UseHPF)
                {
                    // One-pole HPF on the raw bipolar signal per channel,
                    // normalized for unity gain at Nyquist (the unnormalized
                    // form boosted the detected HF slightly).
                    T output = cachedScHpfA0_ * (input - scHpfPrev_[ch]) + cachedScHpfCoeff_ * scHpfState_[ch];
                    scHpfPrev_[ch] = input;
                    scHpfState_[ch] = output;
                    input = output;
                }
                sc = std::max(sc, std::abs(input));
            }

            // 2. Smoothed-peak envelope detector (prevents LF intermodulation).
            // Coefficients are derived from the sample rate in
            // updateCoefficients() - fixed per-sample constants made the
            // detector 4x faster at 192 kHz than at 44.1 kHz.
            T envCoeff = sc > envelope_ ? cachedDetAttCoeff_ : cachedDetRelCoeff_;
            envelope_ += envCoeff * (sc - envelope_);

            // 3. Gain Calculation (Converting envelope to dB only once)
            T levelDb = gainToDecibels(envelope_ + T(1e-9));

            updateStateMachine(levelDb);
            T gain = computeGain(levelDb);

            // 4. Apply Gain
            for (int ch = 0; ch < nCh; ++ch)
            {
                buffer.getChannel(ch)[i] *= gain;
            }
        }
    }

    void updateStateMachine(T levelDb) noexcept
    {
        T openThresh  = cachedThreshold_;
        T closeThresh = cachedThreshold_ - cachedHysteresis_;

        switch (state_)
        {
            case State::Closed:
                if (levelDb > openThresh) state_ = State::Open;
                break;
            case State::Open:
                if (levelDb < closeThresh)
                {
                    state_ = State::Hold;
                    holdCounter_ = cachedHoldSamples_;
                }
                break;
            case State::Hold:
                if (levelDb > openThresh)
                    state_ = State::Open;
                else if (--holdCounter_ <= 0)
                    state_ = State::Closed;
                break;
        }
    }

    [[nodiscard]] T computeGain(T levelDb) noexcept
    {
        T targetGain;

        if (state_ == State::Open || state_ == State::Hold)
        {
            targetGain = T(1);
        }
        else
        {
            T underDb = cachedThreshold_ - levelDb;
            T reductionDb = underDb * (T(1) - T(1) / cachedRatio_);

            // Optimization note: replace decibelsToGain with fast_exp10 if available
            T expandedGain = decibelsToGain(-reductionDb);
            targetGain = std::max(expandedGain, cachedRangeLinear_);
        }

        // Apply smoothing to the linear gain
        T coeff = (targetGain > gateGain_) ? cachedAttackCoeff_ : cachedReleaseCoeff_;
        gateGain_ += coeff * (targetGain - gateGain_);

        return gateGain_;
    }

    void updateCoefficients() noexcept
    {
        if (!(sampleRate_ > 0.0)) return;
        T fs = static_cast<T>(sampleRate_);
        attackCoeff_.store(T(1) - std::exp(T(-1) / (fs * attackMs_.load(std::memory_order_relaxed) / T(1000))),
                           std::memory_order_relaxed);
        releaseCoeff_.store(T(1) - std::exp(T(-1) / (fs * releaseMs_.load(std::memory_order_relaxed) / T(1000))),
                            std::memory_order_relaxed);

        // Detector ballistics (~0.5 ms attack, ~5 ms release), sample-rate aware.
        detAttCoeff_.store(T(1) - std::exp(T(-1) / (fs * T(0.0005))), std::memory_order_relaxed);
        detRelCoeff_.store(T(1) - std::exp(T(-1) / (fs * T(0.005))),  std::memory_order_relaxed);

        // Recompute the sidechain HPF for the (possibly new) sample rate.
        // Design clamp [1, 0.45 * fs] keeps the one-pole stable; the atomic
        // stores make this publication safe from any thread (the coefficient
        // used to be a plain member written cross-thread).
        const double scFreq = std::clamp(
            static_cast<double>(scHpfFreqHz_.load(std::memory_order_relaxed)),
            1.0, sampleRate_ * 0.45);
        const T c = static_cast<T>(std::exp(-std::numbers::pi * 2.0 * scFreq / sampleRate_));
        scHpfCoeff_.store(c, std::memory_order_relaxed);
        scHpfA0_.store((T(1) + c) / T(2), std::memory_order_relaxed);
    }

    double sampleRate_ = 48000.0;
    bool prepared_ = false;

    std::atomic<T> threshold_ { T(-40) };
    std::atomic<T> ratio_ { T(4) };
    std::atomic<T> hysteresis_ { T(4) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> holdMs_ { T(50) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<T> rangeDb_ { T(-80) };
    std::atomic<T> rangeLinear_ { T(0.0001) };

    std::atomic<bool> scHpfEnabled_ { false };
    std::atomic<T> scHpfFreqHz_ { T(80) };
    std::atomic<T> scHpfCoeff_ { T(0.995) };
    std::atomic<T> scHpfA0_ { T(0.9975) };

    // Per-channel sidechain HPF state (16 = detector channel cap).
    static constexpr int MAX_CHANNELS = 16;
    std::array<T, MAX_CHANNELS> scHpfState_{};
    std::array<T, MAX_CHANNELS> scHpfPrev_{};

    std::atomic<T> attackCoeff_ { T(0) };
    std::atomic<T> releaseCoeff_ { T(0) };
    std::atomic<T> detAttCoeff_ { T(0.01) };
    std::atomic<T> detRelCoeff_ { T(0.001) };
    std::atomic<int> holdSamples_ { 0 };

    // Cached per-block
    T cachedThreshold_ = T(-40);
    T cachedRatio_ = T(4);
    T cachedHysteresis_ = T(4);
    T cachedRangeLinear_ = T(0.0001);
    T cachedAttackCoeff_ = T(0);
    T cachedReleaseCoeff_ = T(0);
    T cachedDetAttCoeff_ = T(0.01);
    T cachedDetRelCoeff_ = T(0.001);
    T cachedScHpfCoeff_ = T(0.995);
    T cachedScHpfA0_ = T(0.9975);
    int cachedHoldSamples_ = 0;
    bool cachedScHpfEnabled_ = false;

    State state_ = State::Closed;
    T gateGain_ = T(0);
    T envelope_ = T(0);
    int holdCounter_ = 0;
};

} // namespace dspark
