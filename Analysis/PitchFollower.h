// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PitchFollower.h
 * @brief Musical pitch tracker: PitchDetector + the logic that makes it usable.
 *
 * A raw pitch detector is not directly usable as a modulation source: it
 * reports octave errors on transients, garbage with high confidence during
 * noise bursts, and jumps that would make any controlled parameter zipper.
 * PitchFollower wraps Analysis/PitchDetector.h with the control logic every
 * pitch-driven effect needs:
 *
 * - **Confidence gating** - readings below the confidence threshold (or
 *   outside the configured frequency range) never move the output; the last
 *   reliable pitch is held through consonants and silence.
 * - **Octave-jump correction** - a reading ~1-2 octaves away from the tracked
 *   pitch is folded back to the closest octave of the current target before
 *   being considered (the classic YIN 2x/0.5x failure on transients).
 * - **Jump confirmation** - a genuinely large interval must persist for three
 *   consecutive readings before the target moves (one bad frame never jerks
 *   the output).
 * - **Glide in semitone domain** - the public value slews toward the target
 *   at a constant rate expressed in milliseconds per octave, so the movement
 *   is musically uniform across the whole range (Hz-domain smoothing is
 *   faster at high pitches and sluggish at low ones).
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with processing).
 * - processBlock() / pushSamples() / reset(): audio thread (stream owner).
 * - setRange() / setConfidence() / setGlide(): any thread (relaxed atomics;
 *   non-finite values are ignored).
 * - getSmoothedHz() / getRawHz() / getConfidence() / isTracking() and the
 *   parameter getters: any thread, lock-free.
 *
 * @code
 *   follower.prepare(spec);
 *   follower.setRange(70.0f, 800.0f);          // vocal range
 *   follower.setGlide(60.0f);                  // ms per octave
 *   // per block:
 *   follower.processBlock(buffer);             // internal mono sum
 *   eq.setFrequency(follower.getSmoothedHz() * 0.9f);  // low-cut under tonic
 * @endcode
 *
 * Dependencies: PitchDetector.h, AudioSpec.h, AudioBuffer.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "PitchDetector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class PitchFollower
 * @brief Gated, octave-safe, semitone-smoothed pitch tracking source.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PitchFollower
{
public:
    // -- Lifecycle -------------------------------------------------------------

    /**
     * @brief Prepares the follower and the wrapped detector.
     *
     * An invalid spec (non-finite or non-positive fields) is ignored,
     * keeping the previous state.
     *
     * @param spec       Audio environment specification.
     * @param windowSize Detector analysis window (default 2048 - good down to
     *                   ~50 Hz at 48 kHz; updates every windowSize/4 samples).
     */
    void prepare(const AudioSpec& spec, int windowSize = 2048)
    {
        if (!spec.isValid()) return;
        sampleRate_ = spec.sampleRate;
        detector_.prepare(spec.sampleRate, windowSize, windowSize / 4);
        monoScratch_.assign(static_cast<size_t>(std::max(spec.maxBlockSize, 1)), T(0));
        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /**
     * @brief Forgets the tracked pitch (parameters are kept).
     * Allocation-free; call from the stream owner (it touches the same
     * plain tracking state update() writes).
     */
    void reset() noexcept
    {
        hasTarget_ = false;
        targetSt_ = 0.0;
        currentSt_ = 0.0;
        pendingSt_ = 0.0;
        pendingCount_ = 0;
        samplesSinceValid_ = std::numeric_limits<int64_t>::max() / 2;
        smoothedHz_.store(T(0), std::memory_order_relaxed);
        tracking_.store(false, std::memory_order_relaxed);
    }

    // -- Parameters (thread-safe) ------------------------------------------------

    /**
     * @brief Accepted pitch range in Hz (default 60 to 1200).
     * Non-finite values are ignored (the old max() passed NaN through and
     * closed the validity gate forever).
     */
    void setRange(T minHz, T maxHz) noexcept
    {
        if (!std::isfinite(minHz) || !std::isfinite(maxHz)) return;
        minHz = std::max(minHz, T(10));
        maxHz = std::max(maxHz, minHz);
        minHz_.store(minHz, std::memory_order_relaxed);
        maxHz_.store(maxHz, std::memory_order_relaxed);
    }

    /** @brief Confidence threshold [0, 1] below which readings are ignored (default 0.85). */
    void setConfidence(T threshold) noexcept
    {
        if (!std::isfinite(threshold)) return;
        confidence_.store(std::clamp(threshold, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Glide time in milliseconds per octave (default 60).
     * 0 disables smoothing (the output snaps to each accepted reading).
     * Non-finite values are ignored.
     */
    void setGlide(T msPerOctave) noexcept
    {
        if (!std::isfinite(msPerOctave)) return;
        glideMs_.store(std::max(msPerOctave, T(0)), std::memory_order_relaxed);
    }

    /** @brief Returns the lower bound of the accepted pitch range in Hz. */
    [[nodiscard]] T getMinHz() const noexcept { return minHz_.load(std::memory_order_relaxed); }

    /** @brief Returns the upper bound of the accepted pitch range in Hz. */
    [[nodiscard]] T getMaxHz() const noexcept { return maxHz_.load(std::memory_order_relaxed); }

    /** @brief Returns the confidence gating threshold [0, 1]. */
    [[nodiscard]] T getConfidenceThreshold() const noexcept { return confidence_.load(std::memory_order_relaxed); }

    /** @brief Returns the glide time in milliseconds per octave. */
    [[nodiscard]] T getGlide() const noexcept { return glideMs_.load(std::memory_order_relaxed); }

    // -- Readout (lock-free, any thread) ------------------------------------------

    /** @return Smoothed pitch in Hz; 0 until the first reliable reading. */
    [[nodiscard]] T getSmoothedHz() const noexcept
    {
        return smoothedHz_.load(std::memory_order_relaxed);
    }

    /** @return Last raw detector reading in Hz (ungated - may be garbage). */
    [[nodiscard]] T getRawHz() const noexcept { return detector_.getFrequencyHz(); }

    /** @return Detector confidence of the last raw reading [0, 1]. */
    [[nodiscard]] T getConfidence() const noexcept { return detector_.getConfidence(); }

    /** @return True while reliable readings arrived within the last 250 ms. */
    [[nodiscard]] bool isTracking() const noexcept
    {
        return tracking_.load(std::memory_order_relaxed);
    }

    // -- Processing ----------------------------------------------------------------

    /**
     * @brief Feeds a block (any channel count); channels are averaged to mono.
     * @param buffer Read-only audio block.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        if (nCh <= 0 || nS <= 0) return;

        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, static_cast<int>(monoScratch_.size()));
            if (nCh == 1)
            {
                const T* src = buffer.getChannel(0) + i;
                std::copy(src, src + chunk, monoScratch_.begin());
            }
            else
            {
                const T invCh = T(1) / static_cast<T>(nCh);
                std::fill(monoScratch_.begin(), monoScratch_.begin() + chunk, T(0));
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* src = buffer.getChannel(ch) + i;
                    for (int k = 0; k < chunk; ++k)
                        monoScratch_[static_cast<size_t>(k)] += src[k] * invCh;
                }
            }
            pushSamples({ monoScratch_.data(), static_cast<size_t>(chunk) });
            i += chunk;
        }
    }

    /**
     * @brief Feeds mono samples directly (alternative to processBlock).
     * @param samples Mono audio span.
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed) || samples.empty()) return;
        detector_.pushSamples(samples);
        update(static_cast<int>(samples.size()));
    }

private:
    /** @brief Gating, octave correction, jump confirmation and glide. */
    void update(int numSamples) noexcept
    {
        const T raw  = detector_.getFrequencyHz();
        const T conf = detector_.getConfidence();
        const bool valid = conf >= confidence_.load(std::memory_order_relaxed)
                        && raw >= minHz_.load(std::memory_order_relaxed)
                        && raw <= maxHz_.load(std::memory_order_relaxed);

        if (valid)
        {
            double st = 12.0 * std::log2(static_cast<double>(raw) / 440.0);

            if (!hasTarget_)
            {
                targetSt_ = currentSt_ = st;   // first lock: no sweep-in
                hasTarget_ = true;
            }
            else
            {
                double dev = st - targetSt_;
                if (std::abs(dev) > 7.0)
                {
                    // Octave folding: a 2x/0.5x detector error lands within
                    // ~1.5 st of the target after removing whole octaves.
                    for (const double oct : { -24.0, -12.0, 12.0, 24.0 })
                    {
                        if (std::abs(st + oct - targetSt_) < 1.5)
                        {
                            st += oct;
                            dev = st - targetSt_;
                            break;
                        }
                    }
                }

                if (std::abs(dev) > 7.0)
                {
                    // A real large interval: require three consistent readings.
                    if (pendingCount_ > 0 && std::abs(st - pendingSt_) < 1.0)
                    {
                        if (++pendingCount_ >= 3)
                        {
                            targetSt_ = st;
                            pendingCount_ = 0;
                        }
                    }
                    else
                    {
                        pendingSt_ = st;
                        pendingCount_ = 1;
                    }
                }
                else
                {
                    targetSt_ = st;
                    pendingCount_ = 0;
                }
            }
            samplesSinceValid_ = 0;
        }
        else
        {
            samplesSinceValid_ += numSamples;   // freeze: target stays put
        }

        if (hasTarget_)
        {
            const double glideMs = static_cast<double>(glideMs_.load(std::memory_order_relaxed));
            if (glideMs < 0.5)
            {
                currentSt_ = targetSt_;
            }
            else
            {
                // Constant musical speed: 12 semitones in glideMs milliseconds.
                const double maxStep = 12.0 * static_cast<double>(numSamples)
                                     / (glideMs * 0.001 * sampleRate_);
                currentSt_ += std::clamp(targetSt_ - currentSt_, -maxStep, maxStep);
            }
            smoothedHz_.store(static_cast<T>(440.0 * std::exp2(currentSt_ / 12.0)),
                              std::memory_order_relaxed);
        }

        tracking_.store(hasTarget_
                        && samplesSinceValid_ < static_cast<int64_t>(0.25 * sampleRate_),
                        std::memory_order_relaxed);
    }

    // -- Members --------------------------------------------------------------------
    double sampleRate_ = 48000.0;
    std::atomic<bool> prepared_{ false };

    PitchDetector<T> detector_;
    std::vector<T> monoScratch_;

    bool hasTarget_ = false;
    double targetSt_ = 0.0;     ///< Gated target, semitones relative to A440.
    double currentSt_ = 0.0;    ///< Glided value, semitones relative to A440.
    double pendingSt_ = 0.0;
    int pendingCount_ = 0;
    int64_t samplesSinceValid_ = 0;

    std::atomic<T> minHz_ { T(60) };
    std::atomic<T> maxHz_ { T(1200) };
    std::atomic<T> confidence_ { T(0.85) };
    std::atomic<T> glideMs_ { T(60) };

    std::atomic<T> smoothedHz_ { T(0) };
    std::atomic<bool> tracking_ { false };
};

} // namespace dspark
