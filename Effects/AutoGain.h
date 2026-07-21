// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

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
 * Threading: prepare() belongs to the setup thread; pushReference(),
 * compensate() and reset() belong to the audio thread. Setters are lock-free
 * atomic publications, safe from any thread, consumed at the next
 * compensate(). Non-finite setter arguments are ignored.
 * getCompensationDb() may be read from any thread for metering (approximate:
 * the value is written unsynchronised by the audio thread).
 *
 * Dependencies: DspMath.h, SimdOps.h, AudioSpec.h, AudioBuffer.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SimdOps.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class AutoGain
 * @brief Block-adaptive automatic gain compensation with SIMD-friendly linear interpolation.
 *
 * Levels are measured across the channels passed to prepare(); channels
 * beyond those are neither measured nor compensated (pass-through), and both
 * calls are no-ops before prepare().
 *
 * Note on silence: when BOTH the reference and the output are below -90 dB
 * the compensation target is held at 0 dB (nothing meaningful to match).
 * If only the reference is silent while the effect produces output (e.g. a
 * reverb tail), the match pulls the output down, bounded by the max
 * compensation clamp.
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
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification containing sample rate and channels.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        reset();
    }

    /**
     * @brief Snapshots the input level. Must be called BEFORE processing.
     *
     * An empty view (or one with no prepared channels) keeps the previous
     * reference instead of degrading it.
     *
     * @param buffer Input audio (read-only measurement).
     */
    void pushReference(AudioBufferView<T> buffer) noexcept
    {
        if (std::min(buffer.getNumChannels(), numChannels_) <= 0 ||
            buffer.getNumSamples() <= 0)
            return;
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
        const T smoothSecs = smoothTimeSecs_.load(std::memory_order_relaxed);
        const T alpha = static_cast<T>(std::exp(-static_cast<double>(numSamples)
                                  / (sampleRate_ * static_cast<double>(smoothSecs))));
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
     * Non-finite values are ignored (a NaN would silently disable the clamp).
     */
    void setMaxCompensation(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        maxCompensation_.store(std::abs(dB), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the smoothing time constant.
     * @param ms Smoothing time in milliseconds, floored at 1 ms. Non-finite
     * values are ignored (a NaN would poison the compensation state
     * permanently).
     */
    void setSmoothingTime(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        smoothTimeSecs_.store(std::max<T>(ms * T(0.001), T(0.001)),
                              std::memory_order_relaxed);
    }

    /** @return The maximum allowed compensation in dB (positive). */
    [[nodiscard]] T getMaxCompensation() const noexcept
    {
        return maxCompensation_.load(std::memory_order_relaxed);
    }

    /** @return The smoothing time constant in milliseconds. */
    [[nodiscard]] T getSmoothingTime() const noexcept
    {
        return smoothTimeSecs_.load(std::memory_order_relaxed) * T(1000);
    }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float (setState reads float back); the explicit
        // casts also keep this overload resolvable when T is double.
        StateWriter w(stateId("AGAN"), 1);
        w.write("maxComp", static_cast<float>(maxCompensation_.load(std::memory_order_relaxed)));
        w.write("smoothMs", static_cast<float>(getSmoothingTime()));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("AGAN")) return false;
        setMaxCompensation(static_cast<T>(r.read("maxComp", 12.0f)));
        setSmoothingTime(static_cast<T>(r.read("smoothMs", 100.0f)));
        return true;
    }

private:
    /**
     * @brief Calculates the global RMS level across all active channels.
     * @param buffer Read-only audio view.
     * @return RMS level in decibels (silence floor when the view is empty).
     */
    [[nodiscard]] T measureRmsDb(AudioBufferView<T> buffer) const noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();
        if (numCh <= 0 || numSamples <= 0) return SILENCE_THRESH_DB;

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
    int numChannels_ = 0;                            ///< Expected number of processing channels.

    std::atomic<T> smoothTimeSecs_{ T(0.100) };      ///< Smoothing time constant in seconds.
    T refLevelDb_ = SILENCE_THRESH_DB;               ///< Snapshot of input level.
    T compensationDb_ = T(0);                        ///< Current applied compensation state.

    std::atomic<T> maxCompensation_{ T(12) };        ///< Lock-free UI bound for max +/- dB change.
};

} // namespace dspark
