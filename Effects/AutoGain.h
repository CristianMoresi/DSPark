// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file AutoGain.h
 * @brief Automatic gain compensation for honest A/B comparison.
 *
 * Measures the input level before processing and adjusts the output level
 * after processing to match. This eliminates the loudness bias that makes
 * louder signals sound "better", enabling honest A/B testing.
 *
 * Usage pattern (sandwich):
 * @code
 *   autoGain.pushReference(buffer);   // measure input level
 *   myEffect.processBlock(buffer);    // apply your processing
 *   autoGain.compensate(buffer);      // adjust output to match input level
 * @endcode
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SimdOps.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <type_traits>

namespace dspark {

/**
 * @class AutoGain
 * @brief Block-adaptive automatic gain compensation with SIMD-friendly linear interpolation.
 *
 * @tparam T Sample type (float or double). Must be lock-free atomic compatible if modified concurrently.
 */
template <FloatType T>
class AutoGain
{
    // Ensure the atomic type won't trigger a hidden mutex lock in the audio thread
    static_assert(std::atomic<T>::is_always_lock_free, 
        "AutoGain requires a lock-free float type for thread safety in the audio path.");

public:
    /**
     * @brief Prepares the auto-gain processor.
     * @param spec Audio environment specification containing sample rate and channels.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        reset();
    }

    /**
     * @brief Snapshots the input level. Must be called BEFORE processing.
     * @param buffer Input audio (read-only measurement).
     */
    void pushReference(AudioBufferView<T> buffer) noexcept
    {
        refLevelDb_ = measureRmsDb(buffer);
    }

    /**
     * @brief Measures output level and applies smoothed gain compensation. 
     * Must be called AFTER processing.
     * @param buffer Processed audio (modified in-place).
     */
    void compensate(AudioBufferView<T> buffer) noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();

        if (numSamples == 0 || numCh == 0) return;

        T outLevelDb = measureRmsDb(buffer);
        T targetDb = refLevelDb_ - outLevelDb;

        // Safety: Prevent NaN propagation if both levels are -Inf
        if (std::isnan(targetDb)) 
            targetDb = T(0);

        // Clamp to safety limits
        const T maxComp = maxCompensation_.load(std::memory_order_relaxed);
        targetDb = std::clamp(targetDb, -maxComp, maxComp);

        // Silence bypass (-90 dB threshold)
        if (refLevelDb_ < SILENCE_THRESH_DB && outLevelDb < SILENCE_THRESH_DB)
            targetDb = T(0);

        // Calculate analytical end-state of the one-pole filter for the current block size:
        // alpha = exp(-N / (Fs * tau))
        const T alpha = static_cast<T>(std::exp(-static_cast<double>(numSamples)
                                  / (sampleRate_ * static_cast<double>(smoothTimeSecs_))));
        const T endCompensationDb = targetDb + (compensationDb_ - targetDb) * alpha;

        // Convert dB to linear gain for interpolation
        const T startGain = decibelsToGain(compensationDb_);
        const T endGain = decibelsToGain(endCompensationDb);
        const T gainStep = (endGain - startGain) / static_cast<T>(numSamples);

        // Apply linearly interpolated gain.
        // This loop structure guarantees no loop-carried dependencies, enabling strict SIMD vectorization.
        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            for (int i = 0; i < numSamples; ++i)
            {
                data[i] *= (startGain + static_cast<T>(i) * gainStep);
            }
        }

        // Update internal state for the next block
        compensationDb_ = endCompensationDb;
    }

    /**
     * @brief Hard resets the internal state to avoid feedback loops or stale measurements.
     */
    void reset() noexcept
    {
        refLevelDb_ = SILENCE_THRESH_DB;
        compensationDb_ = T(0);
    }

    /** 
     * @brief Returns the current internal compensation in dB. Useful for UI metering. 
     * @return Current gain offset in decibels.
     */
    [[nodiscard]] T getCompensationDb() const noexcept { return compensationDb_; }

    /**
     * @brief Thread-safe assignment of the maximum allowed compensation limit.
     * @param dB Max gain change in decibels (absolute value used symmetrically).
     */
    void setMaxCompensation(T dB) noexcept 
    { 
        maxCompensation_.store(std::abs(dB), std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the smoothing time constant.
     * @param ms Smoothing time in milliseconds.
     */
    void setSmoothingTime(T ms) noexcept
    {
        smoothTimeSecs_ = std::max<T>(ms * T(0.001), T(0.001)); // Prevent division by zero
    }

private:
    /**
     * @brief Calculates the global RMS level across all active channels.
     * @param buffer Read-only audio view.
     * @return RMS level in decibels.
     */
    [[nodiscard]] T measureRmsDb(AudioBufferView<T> buffer) const noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();

        T sumSq = T(0);
        const int totalSamples = numSamples * numCh;

        for (int ch = 0; ch < numCh; ++ch)
            sumSq += simd::sumOfSquares(buffer.getChannel(ch), numSamples);

        // Apply a strict epsilon floor (approx -150dB) to avoid std::sqrt(0) and gainToDecibels(0) -> -Inf
        T meanSq = std::max<T>(sumSq / static_cast<T>(totalSamples), T(1e-15));
        return gainToDecibels(std::sqrt(meanSq));
    }

    static constexpr T SILENCE_THRESH_DB = T(-90);   ///< Threshold below which audio is considered dead silence.

    double sampleRate_ = 44100.0;                    ///< Current system sample rate.
    int numChannels_ = 2;                            ///< Expected number of processing channels.

    T smoothTimeSecs_ = T(0.100);                    ///< Smoothing time constant in seconds.
    T refLevelDb_ = SILENCE_THRESH_DB;               ///< Snapshot of input level.
    T compensationDb_ = T(0);                        ///< Current applied compensation state.

    std::atomic<T> maxCompensation_{ T(12) };        ///< Lock-free UI bound for max +/- dB change.
};

} // namespace dspark