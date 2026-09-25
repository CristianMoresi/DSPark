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
 * Levels are loudness, not raw RMS: both sides are K-weighted (ITU-R
 * BS.1770, the LUFS filter; setWeighting(Flat) selects plain RMS) and
 * integrated with the same 400 ms time constant (the BS.1770 momentary
 * window). Raw per-block RMS over-corrected low-end changes (a sub-bass rise
 * the ear barely hears moved the match by several dB) and, with small blocks,
 * made the match wobble whenever the processed block's waveform differed
 * from the reference's (latency, phase shifts): each block's ratio was an
 * independent noisy estimate. Integrating both sides from the same start
 * keeps the ratio exact for stationary material from the first block on.
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
 * getCompensationDb() may be read from any thread for metering: it loads an
 * atomic word the audio thread publishes once per compensate(), so the read is
 * synchronised and may be up to one block behind.
 *
 * Dependencies: DspMath.h, Biquad.h, AudioSpec.h, AudioBuffer.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
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
    /** @brief Level measurement used for the match. */
    enum class Weighting
    {
        KWeighted, ///< ITU-R BS.1770 K-weighting (loudness, LUFS filter). Default.
        Flat       ///< Unweighted mean square (plain RMS).
    };

    /**
     * @brief Prepares the auto-gain processor.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state. Allocates the per-channel weighting state.
     *
     * @param spec Audio environment specification containing sample rate and channels.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        shelf_ = BiquadCoeffs::makeKWeightingShelf(sampleRate_);
        highPass_ = BiquadCoeffs::makeKWeightingHighPass(sampleRate_);
        refFilters_.assign(static_cast<size_t>(numChannels_), WeightingState {});
        outFilters_.assign(static_cast<size_t>(numChannels_), WeightingState {});
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
        integrate(refMeanSquare_, buffer, refFilters_);
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

        integrate(outMeanSquare_, buffer, outFilters_);
        const T refLevelDb = meanSquareToDb(refMeanSquare_);
        const T outLevelDb = meanSquareToDb(outMeanSquare_);
        T targetDb = refLevelDb - outLevelDb;

        // Clamp to safety limits
        const T maxComp = maxCompensation_.load(std::memory_order_relaxed);
        targetDb = std::clamp(targetDb, -maxComp, maxComp);

        // Silence bypass (-90 dB threshold)
        if (refLevelDb < SILENCE_THRESH_DB && outLevelDb < SILENCE_THRESH_DB)
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
        // Publish it for cross-thread metering. The working member stays
        // audio-thread-private: getCompensationDb() used to read it directly,
        // which is a plain word written by the audio thread and read by
        // another -- a data race, not merely an approximate number.
        publishedCompensationDb_.store(compensationDb_, std::memory_order_relaxed);
    }

    /**
     * @brief Hard resets the internal state to avoid feedback loops or stale measurements.
     */
    void reset() noexcept
    {
        refMeanSquare_ = 0.0;
        outMeanSquare_ = 0.0;
        for (auto& f : refFilters_) f = {};
        for (auto& f : outFilters_) f = {};
        compensationDb_ = T(0);
        publishedCompensationDb_.store(T(0), std::memory_order_relaxed);
    }

    /** @brief Selects the level measurement (K-weighted loudness by default). RT-safe. */
    void setWeighting(Weighting w) noexcept
    {
        const int v = std::clamp(static_cast<int>(w), 0, static_cast<int>(Weighting::Flat));
        weighting_.store(static_cast<Weighting>(v), std::memory_order_relaxed);
    }

    /** @return The level measurement in use. */
    [[nodiscard]] Weighting getWeighting() const noexcept
    {
        return weighting_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current internal compensation in dB. Useful for UI metering.
     *
     * Safe from any thread: it loads a published atomic word, not the audio
     * thread's working state. The value is the one the last completed
     * compensate() left, so it may be up to one block old.
     *
     * @return Current gain offset in decibels.
     */
    [[nodiscard]] T getCompensationDb() const noexcept
    {
        return publishedCompensationDb_.load(std::memory_order_relaxed);
    }

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
        w.write("weighting", static_cast<int32_t>(getWeighting()));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("AGAN")) return false;
        setMaxCompensation(static_cast<T>(r.read("maxComp", 12.0f)));
        setSmoothingTime(static_cast<T>(r.read("smoothMs", 100.0f)));
        setWeighting(static_cast<Weighting>(r.read("weighting", 0)));
        return true;
    }

private:
    /** Two-biquad K-weighting state (double, like LoudnessMeter's). */
    struct WeightingState { double s1a = 0.0, s2a = 0.0, s1b = 0.0, s2b = 0.0; };

    static inline double tdf2(double x, const BiquadCoeffs& c, double& s1, double& s2) noexcept
    {
        const double y = c.b0 * x + s1;
        s1 = c.b1 * x - c.a1 * y + s2;
        s2 = c.b2 * x - c.a2 * y;
        return y;
    }

    /**
     * @brief Folds one block into a level: the (weighted) mean square over
     * the prepared channels, integrated with the 400 ms one-pole evaluated
     * for the block length. A non-finite block leaves the level unchanged and
     * clears that side's filter state (it would otherwise stay poisoned).
     */
    void integrate(double& meanSquare, AudioBufferView<T> buffer,
                   std::vector<WeightingState>& filters) noexcept
    {
        const int numCh = std::min({ buffer.getNumChannels(), numChannels_,
                                     static_cast<int>(filters.size()) });
        const int numSamples = buffer.getNumSamples();
        if (numCh <= 0 || numSamples <= 0) return;

        const bool weighted = weighting_.load(std::memory_order_relaxed) == Weighting::KWeighted;
        double sumSq = 0.0;
        for (int ch = 0; ch < numCh; ++ch)
        {
            const T* x = buffer.getChannel(ch);
            WeightingState& f = filters[static_cast<size_t>(ch)];
            if (weighted)
            {
                for (int i = 0; i < numSamples; ++i)
                {
                    const double y = tdf2(tdf2(static_cast<double>(x[i]), shelf_, f.s1a, f.s2a),
                                          highPass_, f.s1b, f.s2b);
                    sumSq += y * y;
                }
            }
            else
            {
                for (int i = 0; i < numSamples; ++i)
                    sumSq += static_cast<double>(x[i]) * static_cast<double>(x[i]);
            }
        }

        const double blockMs = sumSq / static_cast<double>(numSamples * numCh);
        if (!std::isfinite(blockMs))
        {
            for (auto& state : filters) state = {};
            return;
        }
        const double a = std::exp(-static_cast<double>(numSamples) / (sampleRate_ * kIntegrationSeconds));
        meanSquare = blockMs + (meanSquare - blockMs) * a;
    }

    /** Mean square to dB, floored like the old -150 dB epsilon. */
    [[nodiscard]] static T meanSquareToDb(double meanSquare) noexcept
    {
        return static_cast<T>(10.0 * std::log10(std::max(meanSquare, 1e-15)));
    }

    static constexpr double kIntegrationSeconds = 0.4; ///< BS.1770 momentary window.
    static constexpr T SILENCE_THRESH_DB = T(-90);   ///< Threshold below which audio is considered dead silence.

    double sampleRate_ = 44100.0;                    ///< Current system sample rate.
    int numChannels_ = 0;                            ///< Expected number of processing channels.

    std::atomic<T> smoothTimeSecs_{ T(0.100) };      ///< Smoothing time constant in seconds.
    std::atomic<Weighting> weighting_{ Weighting::KWeighted };
    double refMeanSquare_ = 0.0;                     ///< Integrated reference level.
    double outMeanSquare_ = 0.0;                     ///< Integrated processed level.
    BiquadCoeffs shelf_ = BiquadCoeffs::makeKWeightingShelf(48000.0);
    BiquadCoeffs highPass_ = BiquadCoeffs::makeKWeightingHighPass(48000.0);
    std::vector<WeightingState> refFilters_;
    std::vector<WeightingState> outFilters_;
    T compensationDb_ = T(0);                        ///< Audio-thread working state.
    std::atomic<T> publishedCompensationDb_{ T(0) }; ///< Cross-thread metering readout.

    std::atomic<T> maxCompensation_{ T(12) };        ///< Lock-free UI bound for max +/- dB change.
};

} // namespace dspark
