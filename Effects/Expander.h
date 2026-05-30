// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Expander.h
 * @brief Downward expander with configurable ratio, hysteresis, and sidechain.
 *
 * A generalization of the noise gate: instead of fully closing (infinite ratio),
 * the expander applies a configurable ratio below the threshold. Includes an 
 * integrated envelope detector to prevent low-frequency intermodulation distortion,
 * and a true-stereo sidechain high-pass filter.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>
#include <array>

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
     * @param sampleRate The sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = std::max(sampleRate, 1.0);
        updateCoefficients();
        reset();
    }

    /**
     * @brief Initializes the expander using an AudioSpec struct.
     * @param spec Framework audio specification object.
     */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate);
    }

    /**
     * @brief Processes an audio block in place.
     * @param buffer View of the audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        cacheParams();

        if (cachedScHpfEnabled_)
            processBlockInternal<true>(buffer);
        else
            processBlockInternal<false>(buffer);
    }

    // -- Parameters ----------------------------------------------------------

    /** @brief Sets the threshold in decibels. */
    void setThreshold(T dB) noexcept { threshold_.store(dB, std::memory_order_relaxed); }
    
    /** @brief Sets the expansion ratio (e.g., 4.0 for 4:1). */
    void setRatio(T ratio) noexcept { ratio_.store(std::max(ratio, T(1)), std::memory_order_relaxed); }
    
    /** @brief Sets the hysteresis gap in decibels to prevent chattering. */
    void setHysteresis(T dB) noexcept { hysteresis_.store(std::max(dB, T(0)), std::memory_order_relaxed); }
    
    /** @brief Sets the maximum gain reduction limit in decibels. */
    void setRange(T dB) noexcept
    {
        rangeDb_ = std::min(dB, T(0));
        rangeLinear_.store(decibelsToGain(rangeDb_), std::memory_order_relaxed);
    }

    /** @brief Sets the attack time in milliseconds. */
    void setAttack(T ms) noexcept
    {
        attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /** @brief Sets the hold time in milliseconds before release begins. */
    void setHold(T ms) noexcept
    {
        holdMs_.store(std::max(ms, T(0)), std::memory_order_relaxed);
        if (sampleRate_ > 0.0) {
            holdSamples_.store(static_cast<int>(sampleRate_ * static_cast<double>(holdMs_.load(std::memory_order_relaxed)) / 1000.0),
                               std::memory_order_relaxed);
        }
    }

    /** @brief Sets the release time in milliseconds. */
    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /** 
     * @brief Enables and configures the sidechain high-pass filter. 
     * @param enabled True to engage the HPF.
     * @param cutoffHz Cutoff frequency in Hz.
     */
    void setSidechainHPF(bool enabled, double cutoffHz = 80.0) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        if (sampleRate_ > 0.0) {
            scHpfCoeff_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0 * cutoffHz / sampleRate_));
        }
    }

    // -- Queries -------------------------------------------------------------

    /** @return The current state of the expander's logic gate. */
    [[nodiscard]] State getState() const noexcept { return state_; }
    
    /** @return The current actual gain being applied, in decibels. */
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
    }

    /**
     * @brief Internal processing loop templated on HPF state to avoid branching.
     * @tparam UseHPF Compile-time flag to include HPF processing.
     * @param buffer View of the audio buffer.
     */
    template <bool UseHPF>
    void processBlockInternal(AudioBufferView<T> buffer) noexcept
    {
        // Clamp to the per-channel HPF state arrays (MAX_CHANNELS): an
        // AudioBufferView is not capped at 16 channels, so an unclamped count would
        // index scHpfState_/scHpfPrev_ out of bounds for >16-channel buffers.
        const int nCh = std::min(buffer.getNumChannels(), MAX_CHANNELS);
        const int nS  = buffer.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            T sc = T(0);

            // 1. Calculate Sidechain Signal
            for (int ch = 0; ch < nCh; ++ch)
            {
                T input = buffer.getChannel(ch)[i];
                if constexpr (UseHPF)
                {
                    // Filter must be applied to raw bipolar signal per channel
                    T output = input - scHpfPrev_[ch] + scHpfCoeff_ * scHpfState_[ch];
                    scHpfPrev_[ch] = input;
                    scHpfState_[ch] = output;
                    input = output;
                }
                sc = std::max(sc, std::abs(input));
            }

            // 2. Fast Envelope Detector (RMS-ish integration to prevent IMD)
            // Using a fixed fast integration time (~2ms) for the detector
            T envCoeff = sc > envelope_ ? T(0.01) : T(0.001); 
            envelope_ += envCoeff * (sc - envelope_);

            // 3. Gain Calculation (Converting envelope to dB only once)
            // Optimization note: replace gainToDecibels with fast_log10 approximation if available in DspMath.h
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
        if (sampleRate_ <= 0.0) return;
        T fs = static_cast<T>(sampleRate_);
        attackCoeff_.store(T(1) - std::exp(T(-1) / (fs * attackMs_.load(std::memory_order_relaxed) / T(1000))),
                           std::memory_order_relaxed);
        releaseCoeff_.store(T(1) - std::exp(T(-1) / (fs * releaseMs_.load(std::memory_order_relaxed) / T(1000))),
                            std::memory_order_relaxed);
    }

    double sampleRate_ = 48000.0;

    std::atomic<T> threshold_ { T(-40) };
    std::atomic<T> ratio_ { T(4) };
    std::atomic<T> hysteresis_ { T(4) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> holdMs_ { T(50) };
    std::atomic<T> releaseMs_ { T(100) };
    T rangeDb_ = T(-80);
    std::atomic<T> rangeLinear_ { T(0.0001) };

    std::atomic<bool> scHpfEnabled_ { false };
    T scHpfCoeff_ = T(0.995);
    
    // Per-channel sidechain HPF state (16 = AudioBufferView channel cap).
    static constexpr int MAX_CHANNELS = 16;
    std::array<T, MAX_CHANNELS> scHpfState_{};
    std::array<T, MAX_CHANNELS> scHpfPrev_{};

    std::atomic<T> attackCoeff_ { T(0) };
    std::atomic<T> releaseCoeff_ { T(0) };
    std::atomic<int> holdSamples_ { 0 };

    // Cached per-block
    T cachedThreshold_ = T(-40);
    T cachedRatio_ = T(4);
    T cachedHysteresis_ = T(4);
    T cachedRangeLinear_ = T(0.0001);
    T cachedAttackCoeff_ = T(0);
    T cachedReleaseCoeff_ = T(0);
    int cachedHoldSamples_ = 0;
    bool cachedScHpfEnabled_ = false;

    State state_ = State::Closed;
    T gateGain_ = T(0);
    T envelope_ = T(0);
    int holdCounter_ = 0;
};

} // namespace dspark
