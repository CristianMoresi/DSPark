// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file NoiseGate.h
 * @brief Noise gate with hysteresis, hold time, and duck mode.
 *
 * A professional noise gate that attenuates audio below a threshold.
 * Uses a state machine (Open/Hold/Close) with hysteresis to prevent
 * chattering. Supports partial gating via the range parameter and
 * a duck mode that inverts the behaviour.
 *
 * Features:
 * - State machine: Open → Hold → Close (with hysteresis)
 * - Open/close threshold hysteresis (prevents chatter)
 * - Configurable hold time before closing
 * - Range: -inf to 0 dB (allows partial attenuation)
 * - Sidechain: internal or external with optional HPF
 * - Duck mode (attenuate when above threshold — for ducking music under voice)
 * - Stereo linked detection
 * - Smooth attack/release transitions
 *
 * Dependencies: DspMath.h.
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

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class NoiseGate
 * @brief Noise gate with state machine, hysteresis, and duck mode.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class NoiseGate
{
public:
    virtual ~NoiseGate() = default;
    /** @brief Gate state. */
    enum class State
    {
        Closed, ///< Gate is closed (attenuating).
        Open,   ///< Gate is open (passing audio).
        Hold    ///< Gate is in hold phase before closing.
    };

    /** @brief Gate mode. */
    enum class GateMode
    {
        Amplitude,  ///< Standard amplitude gating (default).
        Frequency   ///< Frequency-narrowing gate (Gatelope-style): narrows bandpass instead of reducing gain.
    };

    /**
     * @brief Prepares the noise gate.
     * @param sampleRate Sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        // Sync cached values (including cachedRangeLinear_) before reset() —
        // reset() seeds gateGain_ from cachedRangeLinear_, so it must be set
        // to the user-configured range, not the default.
        syncParams();
        paramsDirty_.store(false, std::memory_order_relaxed);
        reset();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    /**
     * @brief Processes an AudioBufferView in-place (unified API).
     * @param buffer Audio buffer (mono or stereo).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        syncParamsIfDirty();

        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh >= 2)
            processStereoInternal(buffer.getChannel(0), buffer.getChannel(1), nS);
        else if (nCh == 1)
            processInternal(buffer.getChannel(0), nS);
    }

    /**
     * @brief Processes audio with an external sidechain signal.
     *
     * Level detection reads from the sidechain buffer; gating is applied
     * to the audio buffer. Uses linked detection (max across sidechain channels).
     *
     * @param audio     Audio buffer to gate (modified in-place).
     * @param sidechain External sidechain signal (read-only).
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        DenormalGuard guard;
        syncParamsIfDirty();
        const int nCh = audio.getNumChannels();
        const int nS  = audio.getNumSamples();
        const int scCh = sidechain.getNumChannels();

        for (int i = 0; i < nS; ++i)
        {
            // Linked detection from sidechain (max across channels)
            T scMax = T(0);
            for (int c = 0; c < scCh; ++c)
            {
                T a = std::abs(sidechain.getChannel(c)[i]);
                if (a > scMax) scMax = a;
            }

            T levelDb = gainToDecibels(scMax);
            updateStateMachine(levelDb);
            T gain = getCurrentGain();

            for (int ch = 0; ch < nCh; ++ch)
                audio.getChannel(ch)[i] *= gain;
        }
    }

    // -- Parameters ---------------------------------------------------------------

    /**
     * @brief Sets the opening threshold.
     * @param dB Threshold in dB (e.g., -40 dB).
     */
    void setThreshold(T dB) noexcept
    {
        threshold_.store(dB, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the hysteresis amount.
     * @param dB Hysteresis in dB (default: 4 dB).
     */
    void setHysteresis(T dB) noexcept
    {
        hysteresis_.store(std::max(dB, T(0)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Sets the attack time in milliseconds. */
    void setAttack(T ms) noexcept
    {
        attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Sets the hold time in milliseconds. */
    void setHold(T ms) noexcept
    {
        holdMs_.store(std::max(ms, T(0)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Sets the release time in milliseconds. */
    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the attenuation range when the gate is closed.
     * @param dB Range in dB (e.g., -80 for near-silence).
     */
    void setRange(T dB) noexcept
    {
        rangeDb_.store(std::min(dB, T(0)), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /** @brief Enables duck mode (inverted gate). */
    void setDuckMode(bool enabled) noexcept
    {
        duckMode_.store(enabled, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the gate mode.
     *
     * - **Amplitude** (default): Standard gain reduction.
     * - **Frequency** (Gatelope-style): Gate narrows bandpass instead of
     *   reducing amplitude. Treble closes before bass. More transparent.
     */
    void setGateMode(GateMode mode) noexcept
    {
        gateMode_.store(mode, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Enables zero-crossing adaptive hold.
     *
     * When enabled, hold time automatically extends to at least one
     * estimated fundamental period (based on zero-crossing rate).
     * Prevents cutting off in the middle of a waveform cycle.
     */
    void setAdaptiveHold(bool enabled) noexcept
    {
        adaptiveHold_.store(enabled, std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Enables the sidechain high-pass filter.
     * @param enabled True to enable.
     * @param cutoffHz Cutoff frequency in Hz.
     */
    void setSidechainHPF(bool enabled, double cutoffHz = 80.0) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        scHpfFreq_.store(static_cast<T>(cutoffHz), std::memory_order_relaxed);
        paramsDirty_.store(true, std::memory_order_relaxed);
    }

    // -- Processing ---------------------------------------------------------------

    /**
     * @brief Processes a single mono sample.
     *
     * Checks the parameter-dirty flag on each call; a single atomic-exchange
     * has no measurable cost compared to recomputing envelope coefficients
     * or (previously) silently ignoring parameter changes from non-audio
     * threads. For tight per-sample loops over long buffers, prefer the
     * `process(T*, int)` block overload, which syncs once per block.
     *
     * @param input Input sample.
     * @return Gated output sample.
     */
    [[nodiscard]] T processSample(T input) noexcept
    {
        syncParamsIfDirty();
        return processSampleInternal(input, input);
    }

    /**
     * @brief Processes a mono sample with external sidechain.
     * @param input Input sample.
     * @param sidechain External sidechain signal.
     * @return Gated output sample.
     */
    [[nodiscard]] T processSampleWithSidechain(T input, T sidechain) noexcept
    {
        syncParamsIfDirty();
        return processSampleInternal(input, sidechain);
    }

    /**
     * @brief Processes stereo samples in-place with linked detection.
     * @param left Left channel sample (modified in-place).
     * @param right Right channel sample (modified in-place).
     */
    void processStereo(T& left, T& right) noexcept
    {
        syncParamsIfDirty();
        processStereoInternalSample(left, right);
    }

    /**
     * @brief Processes mono buffer in-place.
     * @param data Audio samples.
     * @param numSamples Number of samples.
     */
    void process(T* data, int numSamples) noexcept
    {
        syncParamsIfDirty();
        processInternal(data, numSamples);
    }

    /**
     * @brief Processes stereo buffers in-place.
     * @param left Left channel buffer.
     * @param right Right channel buffer.
     * @param numSamples Number of samples.
     */
    void processStereo(T* left, T* right, int numSamples) noexcept
    {
        syncParamsIfDirty();
        processStereoInternal(left, right, numSamples);
    }

    // -- State queries ------------------------------------------------------------

    /** @brief Returns the current gate state. */
    [[nodiscard]] State getState() const noexcept { return state_; }

    /** @brief Returns the current gate gain (0 to 1). */
    [[nodiscard]] T getGainLinear() const noexcept { return gateGain_; }

    /** @brief Returns the current gate gain in dB. */
    [[nodiscard]] T getGainDb() const noexcept { return gainToDecibels(gateGain_); }

    /**
     * @brief Resets the gate to closed state.
     */
    void reset() noexcept
    {
        state_ = State::Closed;
        gateGain_ = cachedRangeLinear_;
        holdCounter_ = 0;
        scHpfState_ = T(0);
        scHpfPrev_ = T(0);
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            freqLpState_[ch] = T(0);
            freqHpState_[ch] = T(0);
            freqLpFreq_[ch] = T(20000);
            freqHpFreq_[ch] = T(20);
        }
        zeroCrossCount_ = 0;
        zeroCrossSamples_ = 0;
        prevSign_ = false;
        estimatedPeriod_ = 0;
    }

protected:
    static constexpr int kMaxChannels = 2;

    /// Sync atomic params only if a setter flagged them dirty. A6 fix:
    /// per-sample entry points no longer silently ignore parameter changes
    /// made from non-audio threads, while the common fast path (nothing
    /// changed) costs a single relaxed atomic exchange.
    void syncParamsIfDirty() noexcept
    {
        if (paramsDirty_.exchange(false, std::memory_order_relaxed))
            syncParams();
    }

    /// Internal mono block: assumes syncParamsIfDirty() has already run.
    void processInternal(T* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = processSampleInternal(data[i], data[i]);
    }

    /// Internal stereo-sample: assumes syncParamsIfDirty() has already run.
    void processStereoInternalSample(T& left, T& right) noexcept
    {
        // Linked detection
        T level = std::max(std::abs(left), std::abs(right));
        T levelDb = gainToDecibels(level);

        updateStateMachine(levelDb);

        T gain = getCurrentGain();
        left  *= gain;
        right *= gain;
    }

    /// Internal stereo block: assumes syncParamsIfDirty() has already run.
    void processStereoInternal(T* left, T* right, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            processStereoInternalSample(left[i], right[i]);
    }

    /// Sync atomic params to audio-thread cached values
    void syncParams() noexcept
    {
        cachedThreshold_ = threshold_.load(std::memory_order_relaxed);
        cachedHysteresis_ = hysteresis_.load(std::memory_order_relaxed);
        cachedDuck_ = duckMode_.load(std::memory_order_relaxed);
        cachedGateMode_ = gateMode_.load(std::memory_order_relaxed);
        cachedAdaptiveHold_ = adaptiveHold_.load(std::memory_order_relaxed);
        cachedRangeLinear_ = decibelsToGain(rangeDb_.load(std::memory_order_relaxed));

        T fs = static_cast<T>(sampleRate_);
        if (fs > T(0))
        {
            T attMs = std::max(attackMs_.load(std::memory_order_relaxed), T(0.01));
            T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));
            attackCoeff_  = T(1) - std::exp(T(-1) / (fs * attMs / T(1000)));
            releaseCoeff_ = T(1) - std::exp(T(-1) / (fs * relMs / T(1000)));

            T hMs = std::max(holdMs_.load(std::memory_order_relaxed), T(0));
            holdSamples_ = static_cast<int>(sampleRate_ * static_cast<double>(hMs) / 1000.0);

            T scFreq = scHpfFreq_.load(std::memory_order_relaxed);
            scHpfCoeff_ = static_cast<T>(
                std::exp(-std::numbers::pi * 2.0 * static_cast<double>(scFreq) / sampleRate_));
        }
    }

    void updateStateMachine(T levelDb) noexcept
    {
        T openThresh  = cachedThreshold_;
        T closeThresh = cachedThreshold_ - cachedHysteresis_;

        bool above = levelDb > openThresh;
        bool below = levelDb < closeThresh;

        if (cachedDuck_)
            std::swap(above, below);

        // Adaptive hold: extend hold to at least one estimated fundamental period
        int effectiveHoldSamples = holdSamples_;
        if (cachedAdaptiveHold_ && estimatedPeriod_ > effectiveHoldSamples)
            effectiveHoldSamples = estimatedPeriod_;

        switch (state_)
        {
            case State::Closed:
                if (above)
                    state_ = State::Open;
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

    /// Track zero-crossings for adaptive hold period estimation
    void updateZeroCrossing(T sample) noexcept
    {
        bool sign = sample >= T(0);
        if (sign != prevSign_)
            ++zeroCrossCount_;
        prevSign_ = sign;

        ++zeroCrossSamples_;
        if (zeroCrossSamples_ >= kZeroCrossWindow)
        {
            // Estimate period from zero-crossing rate
            // 2 zero-crossings per period → period = window / (crossings/2)
            if (zeroCrossCount_ > 0)
                estimatedPeriod_ = (kZeroCrossWindow * 2) / zeroCrossCount_;
            else
                estimatedPeriod_ = 0;
            zeroCrossCount_ = 0;
            zeroCrossSamples_ = 0;
        }
    }

    [[nodiscard]] T getCurrentGain() noexcept
    {
        T targetGain = (state_ == State::Open || state_ == State::Hold)
                       ? T(1) : cachedRangeLinear_;

        T coeff = (targetGain > gateGain_) ? attackCoeff_ : releaseCoeff_;
        gateGain_ += coeff * (targetGain - gateGain_);

        return gateGain_;
    }

    /// Apply frequency-narrowing gate to a single sample on one channel
    [[nodiscard]] T applyFrequencyGate(T input, int ch) noexcept
    {
        T nyquist = static_cast<T>(sampleRate_ * 0.5);
        T gateOpenness = gateGain_; // 1=open, ~0=closed

        // Target frequencies based on gate state
        T targetLp = T(20) + (nyquist - T(20)) * gateOpenness;
        T targetHp = T(20) + (nyquist * T(0.4)) * (T(1) - gateOpenness);

        // Smooth frequency tracking
        freqLpFreq_[ch] += T(0.001) * (targetLp - freqLpFreq_[ch]);
        freqHpFreq_[ch] += T(0.001) * (targetHp - freqHpFreq_[ch]);

        // One-pole LP
        T lpCoeff = T(1) - std::exp(T(-1) * twoPi<T> * freqLpFreq_[ch] / static_cast<T>(sampleRate_));
        freqLpState_[ch] += lpCoeff * (input - freqLpState_[ch]);

        // One-pole HP
        T hpCoeff = std::exp(T(-1) * twoPi<T> * freqHpFreq_[ch] / static_cast<T>(sampleRate_));
        T hpOut = hpCoeff * (freqHpState_[ch] + freqLpState_[ch] - freqHpPrev_[ch]);
        freqHpPrev_[ch] = freqLpState_[ch];
        freqHpState_[ch] = hpOut;

        return hpOut;
    }

    [[nodiscard]] T processSampleInternal(T input, T sidechain, int ch = 0) noexcept
    {
        // Sidechain HPF
        if (scHpfEnabled_.load(std::memory_order_relaxed))
        {
            T output = sidechain - scHpfPrev_ + scHpfCoeff_ * scHpfState_;
            scHpfPrev_ = sidechain;
            scHpfState_ = output;
            sidechain = output;
        }

        T level = std::abs(sidechain);
        T levelDb = gainToDecibels(level);

        if (cachedAdaptiveHold_)
            updateZeroCrossing(sidechain);

        updateStateMachine(levelDb);

        if (cachedGateMode_ == GateMode::Frequency)
        {
            // Frequency mode: compute gain for tracking, but apply frequency narrowing
            (void)getCurrentGain(); // advance gateGain_ state
            return applyFrequencyGate(input, ch);
        }

        return input * getCurrentGain();
    }

    void updateCoefficients() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        T fs = static_cast<T>(sampleRate_);
        T attMs = std::max(attackMs_.load(std::memory_order_relaxed), T(0.01));
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));
        attackCoeff_ = T(1) - std::exp(T(-1) / (fs * attMs / T(1000)));
        releaseCoeff_ = T(1) - std::exp(T(-1) / (fs * relMs / T(1000)));
    }

    double sampleRate_ = 48000.0;

    // Atomic parameters
    std::atomic<T> threshold_ { T(-40) };
    std::atomic<T> hysteresis_ { T(4) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> holdMs_ { T(50) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<T> rangeDb_ { T(-80) };
    std::atomic<bool> duckMode_ { false };
    std::atomic<GateMode> gateMode_ { GateMode::Amplitude };
    std::atomic<bool> adaptiveHold_ { false };

    // A6: dirty flag set by every setter, consumed by syncParamsIfDirty().
    // Starts true so the first call to any process* API initialises cached
    // values from the atomic defaults (prepare() also calls syncParams()).
    std::atomic<bool> paramsDirty_ { true };

    // Sidechain HPF
    std::atomic<bool> scHpfEnabled_ { false };
    std::atomic<T> scHpfFreq_ { T(80) };
    T scHpfCoeff_ = T(0.995);
    T scHpfState_ = T(0);
    T scHpfPrev_ = T(0);

    // Audio-thread cached params
    T cachedThreshold_ = T(-40);
    T cachedHysteresis_ = T(4);
    T cachedRangeLinear_ = T(0.0001);
    bool cachedDuck_ = false;
    GateMode cachedGateMode_ = GateMode::Amplitude;
    bool cachedAdaptiveHold_ = false;

    // Coefficients
    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    int holdSamples_ = 0;

    // State
    State state_ = State::Closed;
    T gateGain_ = T(0);
    int holdCounter_ = 0;

    // Frequency-narrowing gate state
    std::array<T, kMaxChannels> freqLpState_ {};
    std::array<T, kMaxChannels> freqHpState_ {};
    std::array<T, kMaxChannels> freqHpPrev_ {};
    std::array<T, kMaxChannels> freqLpFreq_ {};
    std::array<T, kMaxChannels> freqHpFreq_ {};

    // Zero-crossing adaptive hold
    static constexpr int kZeroCrossWindow = 2048;
    int zeroCrossCount_ = 0;
    int zeroCrossSamples_ = 0;
    bool prevSign_ = false;
    int estimatedPeriod_ = 0;
};

} // namespace dspark
