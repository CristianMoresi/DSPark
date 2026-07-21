// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file NoiseGate.h
 * @brief Noise gate with hysteresis, hold time, and duck mode.
 *
 * A professional noise gate that attenuates audio below a threshold.
 * Uses a state machine (Open/Hold/Close) with hysteresis to prevent
 * chattering. Includes an internal peak envelope detector for distortion-free
 * low-frequency tracking. Thresholds are precomputed in the linear domain, so
 * the per-sample path is branch-light and free of log/pow calls (the envelope
 * recursion itself is serial and does not vectorize).
 *
 * Features:
 * - State machine: Open -> Hold -> Close (with hysteresis)
 * - True envelope peak detection (prevents audio-rate chatter)
 * - Open/close threshold hysteresis
 * - Configurable hold time before closing
 * - Range: -inf to 0 dB (allows partial attenuation)
 * - Sidechain: internal or external with optional HPF
 * - Duck mode (attenuate when above threshold, for ducking music under voice)
 * - Stereo linked detection
 * - Smooth attack/release transitions
 * - Adaptive hold (holds at least one estimated signal period)
 *
 * Threading: prepare() belongs to the setup thread; processBlock(),
 * processSample() and reset() belong to the audio thread. All setters are
 * lock-free atomic publications, safe from any thread; they are consumed at
 * the start of the next block (or the next processSample() call). Non-finite
 * setter arguments are ignored.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h,
 *               StateBlob.h.
 *
 * @code
 *   dspark::NoiseGate<float> gate;
 *   gate.prepare(48000.0);
 *   gate.setThreshold(-40.0f);  // open at -40 dB
 *   gate.setHysteresis(4.0f);   // close at -44 dB
 *   gate.setAttack(0.5f);       // 0.5 ms attack
 *   gate.setHold(50.0f);        // 50 ms hold
 *   gate.setRelease(100.0f);    // 100 ms release
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = gate.processSample(input[i]);
 * @endcode
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
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class NoiseGate
 * @brief High-performance noise gate with state machine, hysteresis, and zero-allocation processing.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class NoiseGate
{
public:
    ~NoiseGate() = default; // Removed virtual to prevent vptr injection in header-only DSP

    /** @brief Gate state machine phases. */
    enum class State
    {
        Closed, ///< Gate is fully closed (applying range attenuation).
        Open,   ///< Gate is fully open (passing audio).
        Hold    ///< Gate is open but counting down hold time before closing.
    };

    /** @brief Operating mode for the gate processing. */
    enum class GateMode
    {
        Amplitude,  ///< Standard amplitude gain reduction (default).
        Frequency   ///< Gatelope-style: narrows bandpass dynamically instead of direct gain reduction.
    };

    /**
     * @brief Prepares the noise gate for processing.
     *
     * Release-safe: a non-positive or non-finite sample rate makes this call
     * a no-op (the previous state and rate are kept).
     *
     * @param sampleRate Operating sample rate in Hz (must be > 0).
     */
    void prepare(double sampleRate) noexcept
    {
        if (!(sampleRate > 0.0)) return; // NaN-safe validity gate
        sampleRate_ = sampleRate;
        syncParams();
        reset();
    }

    /**
     * @brief Prepares from AudioSpec (unified API).
     * @param spec AudioSpec containing format details.
     */
    void prepare(const AudioSpec& spec) noexcept { prepare(spec.sampleRate); }

    /**
     * @brief Processes an AudioBufferView in-place.
     *
     * Detection is linked across ALL channels (peak), so no channel is left
     * ungated. In Frequency mode, channels beyond kMaxChannels fall back to
     * amplitude gating (per-channel filter state is kept for the first two).
     *
     * @param buffer Audio buffer (mono or multi-channel).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        syncParamsIfDirty();

        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh <= 0 || nS <= 0) return;

        const bool freqMode = (cachedGateMode_ == GateMode::Frequency);

        for (int i = 0; i < nS; ++i)
        {
            // Stereo / N-channel linked detection: peak across ALL channels.
            // The sidechain HPF runs per channel BEFORE the peak link; channels
            // beyond the HPF state array are detected unfiltered (sharing a
            // filter between channels would corrupt its history).
            T level = T(0);
            T det0 = T(0);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T s = buffer.getChannel(ch)[i];
                if (cachedScHpfEnabled_ && ch < kMaxScChannels)
                    s = applyScHpf(s, ch);
                if (ch == 0) det0 = s;
                level = std::max(level, std::abs(s));
            }

            // Zero-crossing runs on the same (filtered) signal the detector
            // sees, exactly like the per-sample path.
            if (cachedAdaptiveHold_) updateZeroCrossing(det0);

            T envelope = computeEnvelopeFollower(level);
            updateStateMachine(envelope);
            T gain = getCurrentGain();

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* d = buffer.getChannel(ch);
                // Frequency mode keeps per-channel filter state only for the first
                // kMaxChannels; any extra channels fall back to amplitude gating.
                d[i] = (freqMode && ch < kMaxChannels) ? applyFrequencyGate(d[i], ch)
                                                       : d[i] * gain;
            }
        }
    }

    /**
     * @brief Processes audio with an external sidechain signal.
     *
     * Detection (including the optional HPF, the adaptive-hold zero-crossing
     * tracker and Frequency mode) behaves exactly like the internal-key path;
     * only the level source changes to the sidechain buffer.
     *
     * @param audio Audio buffer to gate (modified in-place).
     * @param sidechain External sidechain signal (read-only).
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        DenormalGuard guard;
        syncParamsIfDirty();

        const int nCh = audio.getNumChannels();
        const int nS  = audio.getNumSamples();
        const int scCh = sidechain.getNumChannels();
        if (nCh <= 0 || nS <= 0) return;

        // NOTE: no std::assume_aligned - view pointers are not guaranteed 32-byte
        // aligned (sub-views / driver buffers), and assuming so is UB. __restrict
        // still conveys no-aliasing to the compiler for the mono fast path.
        if (nCh == 1 && scCh == 1)
        {
            T* __restrict outL = audio.getChannel(0);
            const T* __restrict scL = sidechain.getChannel(0);
            for (int i = 0; i < nS; ++i)
                outL[i] = processSampleInternal(outL[i], scL[i], 0);
        }
        else
        {
            const bool freqMode = (cachedGateMode_ == GateMode::Frequency);

            for (int i = 0; i < nS; ++i)
            {
                T scMax = T(0);
                T det0 = T(0);
                for (int c = 0; c < scCh; ++c)
                {
                    T s = sidechain.getChannel(c)[i];
                    if (cachedScHpfEnabled_ && c < kMaxScChannels)
                        s = applyScHpf(s, c);
                    if (c == 0) det0 = s;
                    T a = std::abs(s);
                    if (a > scMax) scMax = a;
                }

                if (cachedAdaptiveHold_) updateZeroCrossing(det0);

                T envelope = computeEnvelopeFollower(scMax);
                updateStateMachine(envelope);
                T gain = getCurrentGain();

                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* d = audio.getChannel(ch);
                    d[i] = (freqMode && ch < kMaxChannels) ? applyFrequencyGate(d[i], ch)
                                                           : d[i] * gain;
                }
            }
        }
    }

    // -- Parameters (Thread-Safe Setters) -----------------------------------------

    /**
     * @brief Sets the opening threshold.
     * @param dB Threshold in decibels. Non-finite values are ignored.
     */
    void setThreshold(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        threshold_.store(dB, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the hysteresis amount (gap between open and close thresholds).
     * @param dB Hysteresis in decibels (floored to 0). Non-finite values are ignored.
     */
    void setHysteresis(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        hysteresis_.store(std::max(T(0), dB), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets attack time in milliseconds. Non-finite values are ignored. */
    void setAttack(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        attackMs_.store(std::max(T(0.01), ms), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets hold time in milliseconds. Non-finite values are ignored. */
    void setHold(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        holdMs_.store(std::max(T(0), ms), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets release time in milliseconds. Non-finite values are ignored. */
    void setRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::max(T(0.01), ms), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets maximum attenuation range.
     * @param dB Range in decibels (capped at 0). Non-finite values are ignored.
     */
    void setRange(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        rangeDb_.store(std::min(T(0), dB), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Toggles ducking mode (invert gate logic). */
    void setDuckMode(bool enabled) noexcept
    {
        duckMode_.store(enabled, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Selects the processing mode (Amplitude or Frequency; wild enum values clamp). */
    void setGateMode(GateMode mode) noexcept
    {
        const int m = std::clamp(static_cast<int>(mode), 0, static_cast<int>(GateMode::Frequency));
        gateMode_.store(static_cast<GateMode>(m), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Toggles adaptive hold based on zero-crossing rate. */
    void setAdaptiveHold(bool enabled) noexcept
    {
        adaptiveHold_.store(enabled, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Configures sidechain High-Pass filter.
     *
     * An invalid cutoff (non-finite or non-positive) keeps the previous
     * frequency and only applies the toggle: a negative cutoff would make the
     * one-pole coefficient exceed 1 (an unstable, runaway filter).
     *
     * @param enabled Filter active state.
     * @param cutoffHz Filter cutoff in Hz (clamped to [1, 0.45 * fs]).
     */
    void setSidechainHPF(bool enabled, double cutoffHz = 80.0) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        if (std::isfinite(cutoffHz) && cutoffHz > 0.0)
            scHpfFreq_.store(static_cast<T>(cutoffHz), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_release);
    }

    // -- Single Sample Processing -------------------------------------------------

    /**
     * @brief Processes a single mono sample.
     *
     * Bit-identical to processBlock() for mono streams.
     *
     * @note No DenormalGuard here (per-sample hot path); per-sample callers
     *       are expected to guard their own processing loop.
     *
     * @param input Source sample.
     * @return Gated output sample.
     */
    [[nodiscard]] T processSample(T input) noexcept
    {
        syncParamsIfDirty();
        return processSampleInternal(input, input, 0);
    }

    /**
     * @brief Processes a mono sample using an external sidechain.
     * @param input Source sample.
     * @param sidechain Key/Sidechain sample.
     * @return Gated output sample.
     */
    [[nodiscard]] T processSampleWithSidechain(T input, T sidechain) noexcept
    {
        syncParamsIfDirty();
        return processSampleInternal(input, sidechain, 0);
    }

    /**
     * @brief Resets DSP state (clears filters, sets state to Closed).
     */
    void reset() noexcept
    {
        state_ = State::Closed;
        gateGain_ = cachedRangeLinear_;
        envelopeState_ = T(0);
        holdCounter_ = 0;
        scHpfState_.fill(T(0));
        scHpfPrev_.fill(T(0));
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            freqLpState_[ch] = T(0);
            freqHpState_[ch] = T(0);
            freqHpPrev_[ch] = T(0);
            freqLpFreq_[ch] = T(20000);
            freqHpFreq_[ch] = T(20);
        }
        zeroCrossCount_ = 0;
        zeroCrossSamples_ = 0;
        prevSign_ = false;
        estimatedPeriod_ = 0;
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("GATE"), 1);
        w.write("threshold", threshold_.load(std::memory_order_relaxed));
        w.write("hysteresis", hysteresis_.load(std::memory_order_relaxed));
        w.write("attack", attackMs_.load(std::memory_order_relaxed));
        w.write("hold", holdMs_.load(std::memory_order_relaxed));
        w.write("release", releaseMs_.load(std::memory_order_relaxed));
        w.write("range", rangeDb_.load(std::memory_order_relaxed));
        w.write("duck", duckMode_.load(std::memory_order_relaxed));
        w.write("gateMode", static_cast<int32_t>(gateMode_.load(std::memory_order_relaxed)));
        w.write("adaptiveHold", adaptiveHold_.load(std::memory_order_relaxed));
        w.write("scHpf", scHpfEnabled_.load(std::memory_order_relaxed));
        w.write("scHpfFreq", scHpfFreq_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("GATE")) return false;
        setThreshold(static_cast<T>(r.read("threshold", -40.0f)));
        setHysteresis(static_cast<T>(r.read("hysteresis", 4.0f)));
        setAttack(static_cast<T>(r.read("attack", 0.5f)));
        setHold(static_cast<T>(r.read("hold", 50.0f)));
        setRelease(static_cast<T>(r.read("release", 100.0f)));
        setRange(static_cast<T>(r.read("range", -80.0f)));
        setDuckMode(r.read("duck", false));
        setGateMode(static_cast<GateMode>(r.read("gateMode", 0)));
        setAdaptiveHold(r.read("adaptiveHold", false));
        setSidechainHPF(r.read("scHpf", false),
                        static_cast<double>(r.read("scHpfFreq", 80.0f)));
        return true;
    }

protected:
    static constexpr int kMaxChannels = 2;

    void syncParamsIfDirty() noexcept
    {
        // Plain load first: the exchange RMW is only paid when a publication
        // is actually pending (this runs per sample in the per-sample path).
        if (paramsDirty_.load(std::memory_order_acquire)
            && paramsDirty_.exchange(false, std::memory_order_acquire))
            syncParams();
    }

    void syncParams() noexcept
    {
        T fs = static_cast<T>(sampleRate_);
        if (!(fs > T(0))) return; // NaN-safe (prepare() already gates the rate)

        // Cache logical switches to prevent atomic loads in hot path
        cachedDuck_ = duckMode_.load(std::memory_order_relaxed);
        cachedGateMode_ = gateMode_.load(std::memory_order_relaxed);
        cachedAdaptiveHold_ = adaptiveHold_.load(std::memory_order_relaxed);
        cachedScHpfEnabled_ = scHpfEnabled_.load(std::memory_order_relaxed);

        // Compute thresholds in LINEAR domain to eliminate log10 in hot path
        T thDb = threshold_.load(std::memory_order_relaxed);
        T hystDb = hysteresis_.load(std::memory_order_relaxed);
        cachedThresholdLinear_ = decibelsToGain(thDb);
        cachedCloseThresholdLinear_ = decibelsToGain(thDb - hystDb);
        cachedRangeLinear_ = decibelsToGain(rangeDb_.load(std::memory_order_relaxed));

        T attMs = std::max(attackMs_.load(std::memory_order_relaxed), T(0.01));
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));
        attackCoeff_  = T(1) - std::exp(T(-1) / (fs * attMs / T(1000)));
        releaseCoeff_ = T(1) - std::exp(T(-1) / (fs * relMs / T(1000)));

        // Fixed ~2ms release for the detector envelope (prevents bass chattering)
        detectorReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.002)));

        // Parameter smoothing coefficient for Frequency Mode (prevents hardcoded 0.001 dependency)
        freqSmoothCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.02))); // 20ms smoothing

        // Cap the sample count in double before the cast (a cast of an
        // out-of-int-range double is undefined behaviour).
        T hMs = std::max(holdMs_.load(std::memory_order_relaxed), T(0));
        holdSamples_ = static_cast<int>(std::min(fs * static_cast<double>(hMs) / 1000.0, 1.0e9));

        // The setter rejects invalid cutoffs; the clamp keeps the coefficient
        // in the stable range even against a hostile in-memory value.
        const double scFreq = std::clamp(
            static_cast<double>(scHpfFreq_.load(std::memory_order_relaxed)),
            1.0, sampleRate_ * 0.45);
        scHpfCoeff_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0 * scFreq / static_cast<double>(fs)));
        scHpfA0_ = (T(1) + scHpfCoeff_) / T(2); // Normalization to prevent high-frequency boost

        cachedNyquist_ = static_cast<T>(fs * 0.5);
        cachedFsInvPi2_ = static_cast<T>(std::numbers::pi * 2.0 / fs);
    }

    /**
     * @brief Updates envelope state to prevent intra-cycle chattering.
     */
    [[nodiscard]] T computeEnvelopeFollower(T rawLevel) noexcept
    {
        if (rawLevel > envelopeState_)
            envelopeState_ = rawLevel; // Instant attack for precise triggering
        else
            envelopeState_ += detectorReleaseCoeff_ * (rawLevel - envelopeState_);

        return envelopeState_;
    }

    void updateStateMachine(T envelopeLinear) noexcept
    {
        bool above = envelopeLinear > cachedThresholdLinear_;
        bool below = envelopeLinear < cachedCloseThresholdLinear_;

        if (cachedDuck_)
            std::swap(above, below);

        int effectiveHoldSamples = holdSamples_;
        if (cachedAdaptiveHold_ && estimatedPeriod_ > effectiveHoldSamples)
            effectiveHoldSamples = estimatedPeriod_;

        switch (state_)
        {
            case State::Closed:
                if (above) state_ = State::Open;
                break;

            case State::Open:
                if (below)
                {
                    state_ = State::Hold;
                    holdCounter_ = effectiveHoldSamples;
                }
                break;

            case State::Hold:
                if (above)
                {
                    state_ = State::Open;
                }
                else
                {
                    --holdCounter_;
                    if (holdCounter_ <= 0)
                        state_ = State::Closed;
                }
                break;
        }
    }

    void updateZeroCrossing(T sample) noexcept
    {
        bool sign = sample >= T(0);
        if (sign != prevSign_)
            ++zeroCrossCount_;
        prevSign_ = sign;

        ++zeroCrossSamples_;
        if (zeroCrossSamples_ >= kZeroCrossWindow)
        {
            if (zeroCrossCount_ > 0)
                estimatedPeriod_ = (kZeroCrossWindow * 2) / zeroCrossCount_;
            else
                estimatedPeriod_ = 0;

            zeroCrossCount_ = 0;
            zeroCrossSamples_ = 0;
        }
    }

    [[nodiscard]] inline T getCurrentGain() noexcept
    {
        T targetGain = (state_ == State::Open || state_ == State::Hold) ? T(1) : cachedRangeLinear_;

        // Fast path for established states to avoid unnecessary math
        if (targetGain == gateGain_) return gateGain_;

        T coeff = (targetGain > gateGain_) ? attackCoeff_ : releaseCoeff_;
        gateGain_ += coeff * (targetGain - gateGain_);
        return gateGain_;
    }

    [[nodiscard]] T applyFrequencyGate(T input, int ch) noexcept
    {
        T gateOpenness = gateGain_;

        T targetLp = T(20) + (cachedNyquist_ - T(20)) * gateOpenness;
        T targetHp = T(20) + (cachedNyquist_ * T(0.4)) * (T(1) - gateOpenness);

        freqLpFreq_[ch] += freqSmoothCoeff_ * (targetLp - freqLpFreq_[ch]);
        freqHpFreq_[ch] += freqSmoothCoeff_ * (targetHp - freqHpFreq_[ch]);

        // Bilinear/Euler approximation to avoid std::exp() in the audio inner loop
        T lpCoeff = std::min(T(1), freqLpFreq_[ch] * cachedFsInvPi2_);
        freqLpState_[ch] += lpCoeff * (input - freqLpState_[ch]);

        T hpCoeff = T(1) - std::min(T(1), freqHpFreq_[ch] * cachedFsInvPi2_);
        T hpOut = hpCoeff * (freqHpState_[ch] + freqLpState_[ch] - freqHpPrev_[ch]);

        freqHpPrev_[ch] = freqLpState_[ch];
        freqHpState_[ch] = hpOut;

        return hpOut;
    }

    /** @brief Per-channel one-pole sidechain high-pass (unity gain at Nyquist). */
    [[nodiscard]] T applyScHpf(T input, int ch) noexcept
    {
        T output = scHpfA0_ * (input - scHpfPrev_[ch]) + scHpfCoeff_ * scHpfState_[ch];
        scHpfPrev_[ch] = input;
        scHpfState_[ch] = output;
        return output;
    }

    [[nodiscard]] T processSampleInternal(T input, T sidechain, int ch) noexcept
    {
        if (cachedScHpfEnabled_ && ch < kMaxScChannels)
            sidechain = applyScHpf(sidechain, ch);

        T rawLevel = std::abs(sidechain);

        if (cachedAdaptiveHold_)
            updateZeroCrossing(sidechain);

        T envelope = computeEnvelopeFollower(rawLevel);
        updateStateMachine(envelope);

        if (cachedGateMode_ == GateMode::Frequency)
        {
            (void)getCurrentGain();
            return applyFrequencyGate(input, ch);
        }

        return input * getCurrentGain();
    }

    double sampleRate_ = 48000.0;

    std::atomic<T> threshold_ { T(-40) };
    std::atomic<T> hysteresis_ { T(4) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> holdMs_ { T(50) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<T> rangeDb_ { T(-80) };
    std::atomic<bool> duckMode_ { false };
    std::atomic<GateMode> gateMode_ { GateMode::Amplitude };
    std::atomic<bool> adaptiveHold_ { false };
    std::atomic<bool> paramsDirty_ { true };

    std::atomic<bool> scHpfEnabled_ { false };
    std::atomic<T> scHpfFreq_ { T(80) };

    // Cached internal variables
    T cachedThresholdLinear_ = T(0.01);
    T cachedCloseThresholdLinear_ = T(0.0063);
    T cachedRangeLinear_ = T(0.0001);
    bool cachedDuck_ = false;
    GateMode cachedGateMode_ = GateMode::Amplitude;
    bool cachedAdaptiveHold_ = false;
    bool cachedScHpfEnabled_ = false;
    T cachedNyquist_ = T(24000);
    T cachedFsInvPi2_ = T(0);

    static constexpr int kMaxScChannels = 16;
    T scHpfCoeff_ = T(0.995);
    T scHpfA0_ = T(0.9975);
    std::array<T, kMaxScChannels> scHpfState_ {};
    std::array<T, kMaxScChannels> scHpfPrev_ {};

    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    int holdSamples_ = 0;

    // Fast Envelope Follower State
    T envelopeState_ = T(0);
    T detectorReleaseCoeff_ = T(0);

    State state_ = State::Closed;
    T gateGain_ = T(0);
    int holdCounter_ = 0;

    T freqSmoothCoeff_ = T(0);
    std::array<T, kMaxChannels> freqLpState_ {};
    std::array<T, kMaxChannels> freqHpState_ {};
    std::array<T, kMaxChannels> freqHpPrev_ {};
    std::array<T, kMaxChannels> freqLpFreq_ {};
    std::array<T, kMaxChannels> freqHpFreq_ {};

    static constexpr int kZeroCrossWindow = 2048;
    int zeroCrossCount_ = 0;
    int zeroCrossSamples_ = 0;
    bool prevSign_ = false;
    int estimatedPeriod_ = 0;
};

} // namespace dspark
