// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Crossfade.h
 * @brief Crossfade between two audio signals with selectable curve and artifact-free parameter smoothing.
 *
 * Provides crossfade curves for audio transitions. Designed for zero-allocation
 * real-time contexts: position changes glide instead of stepping, and the
 * curve is evaluated at every sample of a glide, so an equal-power fade stays
 * equal power while it moves.
 *
 * Curves:
 * - **Linear:** Constant sum (A * (1-t) + B * t). Fast, but exhibits a -3 dB power dip at the centre for uncorrelated signals.
 * - **EqualPower:** Constant energy (A^2 + B^2 = 1) with the sine/cosine law. Unlike the square-root
 *   law it has a finite slope at both ends, so a fade starts and ends without a step-like onset
 *   (the square root reaches -20 dB after 1% of the fade, the sine after 6.4%). Perfect for uncorrelated audio.
 * - **SCurve:** Smoothstep S-curve. Provides a perceptually smoother transition speed, though it shares the -3 dB centre power dip of the Linear curve.
 *
 * Glide: after prepare(), the applied position moves by at most one full
 * sweep per 20 ms, whatever the block size. Without prepare(), a change ramps
 * across the next block of the block call (and applies at once in the
 * single-sample call), which suits offline and per-sample callers. Either way
 * the first processing call after construction, prepare() or reset() starts
 * settled on the current position and curve.
 *
 * Dependencies: Core/AudioSpec.h, Core/DspMath.h.
 *
 * Threading: prepare() belongs to the setup thread; reset() and the
 * processing calls to the audio thread. setCurve()/setPosition() are safe
 * from any thread (atomics, consumed by the processing calls); non-finite
 * positions are ignored. getGainA() and getGainB() are safe from any thread:
 * they load atomic words the processing call publishes, so they may be up to
 * one call behind. gainsFor() is a pure function.
 */

#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>

namespace dspark {

/**
 * @class Crossfade
 * @brief Artifact-free, SIMD-friendly crossfader for two audio signals.
 *
 * Marked as final to explicitly prohibit inheritance and avoid vtable overhead,
 * adhering to the framework's zero virtual dispatch policy in DSP nodes.
 *
 * @tparam T Sample type (float or double). Requires std::is_floating_point_v<T>.
 */
template <FloatType T>
class Crossfade final
{
public:
    /** @brief Defines the amplitude response of the crossfade transition. */
    enum class Curve
    {
        Linear,     ///< Linear interpolation. Constant amplitude, drops power at center.
        EqualPower, ///< Sine/cosine law. Constant power, no volume drop, smooth at both ends.
        SCurve      ///< Smoothstep interpolation. Slower progression at extremes.
    };

    Crossfade() = default;
    ~Crossfade() = default;

    /**
     * @brief Enables the time-based glide (optional).
     *
     * Once prepared, position changes move the applied position by at most one
     * full sweep per 20 ms in both processing calls, independently of the
     * block size: a per-block ramp lands in 0.7 ms with 32-sample blocks and
     * clicks. Also lands the applied position on the requested one (reset()).
     * Invalid rates (non-positive or non-finite) are ignored.
     *
     * @param sampleRate Sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) return;
        maxPosStep_ = static_cast<T>(1.0 / (sampleRate * kMinGlideSeconds));
        reset();
    }

    /**
     * @brief Enables the time-based glide from an audio spec (see prepare(double)).
     * @param spec Audio environment; only the sample rate is used.
     */
    void prepare(const AudioSpec& spec) noexcept { prepare(spec.sampleRate); }

    /**
     * @brief Lands the applied position and curve on the requested ones, with
     *        no glide (for example at transport start). Settings changed
     *        before the next processing call are applied at once too.
     */
    void reset() noexcept
    {
        settle();
        needsSettle_ = true;
    }

    /**
     * @brief Sets the crossfade curve type.
     * Thread-safe. Can be called from the GUI thread. A curve change is
     * blended from the old law to the new one across the next block.
     * @param curve The desired crossfade curve. Out-of-range values (a wild
     *              cast) are clamped into the enum range.
     */
    void setCurve(Curve curve) noexcept
    {
        curve = static_cast<Curve>(std::clamp(static_cast<int>(curve), 0,
                                              static_cast<int>(Curve::SCurve)));
        curve_.store(curve, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target crossfade blend position.
     *
     * Thread-safe. The processing calls glide to it (see the file overview).
     *
     * @param position Target blend: 0.0 = 100% A, 1.0 = 100% B. Automatically
     *                 clamped [0, 1]; non-finite values are ignored.
     */
    void setPosition(T position) noexcept
    {
        if (!std::isfinite(position)) return;
        position_.store(std::clamp(position, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Retrieves the last requested position.
     * @return Current requested blend position [0, 1].
     */
    [[nodiscard]] T getPosition() const noexcept
    {
        return position_.load(std::memory_order_relaxed);
    }

    /**
     * @brief The gains a curve applies at a position.
     *
     * The exact law the processing calls use, so a caller can predict or
     * normalize a blend without running one.
     *
     * @param curve    Crossfade curve.
     * @param position Blend position, clamped to [0, 1] (NaN reads as 0).
     * @param gainA    Receives the gain of signal A.
     * @param gainB    Receives the gain of signal B.
     */
    static void gainsFor(Curve curve, T position, T& gainA, T& gainB) noexcept
    {
        // min/max with this argument order also resolves NaN to 0.
        position = std::min(T(1), std::max(T(0), position));
        fillGains(curve, &position, &gainA, &gainB, 1);
    }

    /**
     * @brief Crossfades two individual samples.
     *
     * Unprepared, the gains follow the position at once, which suits
     * per-sample automation through setPosition(). Prepared, the applied
     * position glides at the same rate as in the block call.
     * Must only be called from the audio thread.
     *
     * @param a Input sample A (Dry/Left).
     * @param b Input sample B (Wet/Right).
     * @return Blended output sample.
     */
    [[nodiscard]] inline T process(T a, T b) noexcept
    {
        settleIfFresh();
        const T target = position_.load(std::memory_order_relaxed);
        const Curve curve = curve_.load(std::memory_order_relaxed);
        if (target != curPos_ || curve != lastCurve_)
        {
            curPos_ = maxPosStep_ > T(0) ? moveTowards(curPos_, target, maxPosStep_) : target;
            lastCurve_ = curve;
            gainsFor(curve, curPos_, gainA_, gainB_);
            publishGains();
        }
        return a * gainA_ + b * gainB_;
    }

    /**
     * @brief Crossfades two audio buffers into an output buffer with automatic parameter smoothing.
     *
     * While the applied position is settled this is a vectorized static gain
     * loop. During a glide (see the file overview) the curve is evaluated at
     * every sample; a curve change is blended from the old law to the new one
     * across the block.
     *
     * @param inputA Pointer to the first input buffer array. Must not be null.
     * @param inputB Pointer to the second input buffer array. Must not be null.
     * @param output Pointer to the output buffer array. Must not be null. May
     *               alias either input.
     * @param numSamples Number of samples to process. Must be > 0.
     */
    void process(const T* inputA, const T* inputB, T* output, int numSamples) noexcept
    {
        assert(inputA != nullptr && inputB != nullptr && output != nullptr);
        assert(numSamples > 0);
        if (numSamples <= 0) return;

        settleIfFresh();
        const T target = position_.load(std::memory_order_relaxed);
        const Curve curve = curve_.load(std::memory_order_relaxed);

        if (target == curPos_ && curve == lastCurve_)
        {
            // Hot path: settled. Ideal for SIMD autovectorization.
            const T gA = gainA_;
            const T gB = gainB_;
            for (int i = 0; i < numSamples; ++i)
                output[i] = inputA[i] * gA + inputB[i] * gB;
            return;
        }

        const T startPos = curPos_;
        const Curve oldCurve = lastCurve_;
        const bool prepared = maxPosStep_ > T(0);
        const T step = prepared ? maxPosStep_
                                : std::abs(target - startPos) / static_cast<T>(numSamples);
        const T invSamples = T(1) / static_cast<T>(numSamples);

        T pos[kChunk], gA[kChunk], gB[kChunk], oldA[kChunk], oldB[kChunk];
        for (int start = 0; start < numSamples; start += kChunk)
        {
            const int n = std::min(kChunk, numSamples - start);
            for (int i = 0; i < n; ++i)
                pos[i] = moveTowards(startPos, target, step * static_cast<T>(start + i + 1));
            fillGains(curve, pos, gA, gB, n);

            if (curve != oldCurve)
            {
                fillGains(oldCurve, pos, oldA, oldB, n);
                for (int i = 0; i < n; ++i)
                {
                    const T w = static_cast<T>(start + i + 1) * invSamples;
                    gA[i] = oldA[i] + w * (gA[i] - oldA[i]);
                    gB[i] = oldB[i] + w * (gB[i] - oldB[i]);
                }
            }

            for (int i = 0; i < n; ++i)
                output[start + i] = inputA[start + i] * gA[i] + inputB[start + i] * gB[i];
        }

        // The block ramp of an unprepared crossfader always lands on the target.
        curPos_ = prepared ? moveTowards(startPos, target, step * static_cast<T>(numSamples))
                           : target;
        lastCurve_ = curve;
        gainsFor(curve, curPos_, gainA_, gainB_);
        publishGains();
    }

    /**
     * @brief Processes crossfading using a per-sample automation buffer.
     *
     * The positions are applied as given (no glide). The next call of the
     * other processing methods glides from the last automated position to
     * the requested one.
     *
     * @param inputA Pointer to the first input buffer.
     * @param inputB Pointer to the second input buffer.
     * @param positions Array of target positions [0, 1] per sample. Values
     *                  are clamped; NaN reads as 0 (100% A).
     * @param output Pointer to the output buffer.
     * @param numSamples Number of samples to process.
     */
    void processAutomated(const T* inputA, const T* inputB,
                          const T* positions, T* output, int numSamples) noexcept
    {
        assert(inputA && inputB && positions && output);
        assert(numSamples > 0);
        if (numSamples <= 0) return;

        const Curve curve = curve_.load(std::memory_order_relaxed);

        T pos[kChunk], gA[kChunk], gB[kChunk];
        for (int start = 0; start < numSamples; start += kChunk)
        {
            const int n = std::min(kChunk, numSamples - start);
            // min/max with this argument order also resolves a NaN in the
            // automation buffer to 0 (100% A) instead of poisoning the output.
            for (int i = 0; i < n; ++i)
                pos[i] = std::min(T(1), std::max(T(0), positions[start + i]));
            fillGains(curve, pos, gA, gB, n);
            for (int i = 0; i < n; ++i)
                output[start + i] = inputA[start + i] * gA[i] + inputB[start + i] * gB[i];
        }

        // Update the audio-thread state to the end of the automation block.
        // The atomic position_ is NOT written back: that would race with the
        // GUI thread.
        needsSettle_ = false;
        curPos_ = std::min(T(1), std::max(T(0), positions[numSamples - 1]));
        lastCurve_ = curve;
        gainsFor(curve, curPos_, gainA_, gainB_);
        publishGains();
    }

    /** @brief Gets the current internal gain multiplier for signal A.
     *  Safe from any thread: loads a published atomic word, so it may be up to
     *  one processing call behind. */
    [[nodiscard]] T getGainA() const noexcept
    {
        return publishedGainA_.load(std::memory_order_relaxed);
    }

    /** @brief Gets the current internal gain multiplier for signal B.
     *  Safe from any thread: loads a published atomic word, so it may be up to
     *  one processing call behind. */
    [[nodiscard]] T getGainB() const noexcept
    {
        return publishedGainB_.load(std::memory_order_relaxed);
    }

private:
    static constexpr int kChunk = 64;
    static constexpr double kMinGlideSeconds = 0.02;

    /**
     * @brief sin((pi/2) t) and cos((pi/2) t) for t in [0, 1].
     *
     * Each half is evaluated on [0, pi/4] by its Taylor series (to x^9 and
     * x^10: error 1.8e-9 in double, float-rounding-limited in float), and the
     * upper half by the mirror sin(pi/2 - x) = cos(x). The ends are exact
     * (0 and 1), no range reduction is needed and the loop vectorizes.
     */
    static inline void equalPowerGains(T t, T& gA, T& gB) noexcept
    {
        const bool upper = t > T(0.5);
        const T u  = upper ? T(1) - t : t;
        const T x  = halfPi<T> * u;
        const T x2 = x * x;
        const T s = x * (T(1) + x2 * (T(-1.0 / 6.0) + x2 * (T(1.0 / 120.0)
                  + x2 * (T(-1.0 / 5040.0) + x2 * T(1.0 / 362880.0)))));
        const T c = T(1) + x2 * (T(-0.5) + x2 * (T(1.0 / 24.0) + x2 * (T(-1.0 / 720.0)
                  + x2 * (T(1.0 / 40320.0) + x2 * T(-1.0 / 3628800.0)))));
        gB = upper ? c : s;
        gA = upper ? s : c;
    }

    /**
     * @brief Gains of one curve for n positions already in [0, 1] (the curve
     *        switch sits outside the sample loops).
     */
    static void fillGains(Curve curve, const T* pos, T* gA, T* gB, int n) noexcept
    {
        switch (curve)
        {
            case Curve::EqualPower:
                for (int i = 0; i < n; ++i)
                    equalPowerGains(pos[i], gA[i], gB[i]);
                break;

            case Curve::SCurve:
                for (int i = 0; i < n; ++i)
                {
                    const T t = pos[i] * pos[i] * (T(3) - T(2) * pos[i]);
                    gA[i] = T(1) - t;
                    gB[i] = t;
                }
                break;

            case Curve::Linear:
            default:
                // default: unreachable through the clamped setter; keeps the
                // outputs defined if the enum ever grows without a case.
                for (int i = 0; i < n; ++i)
                {
                    gA[i] = T(1) - pos[i];
                    gB[i] = pos[i];
                }
                break;
        }
    }

    /** @brief Lands the applied state on the requested position and curve. */
    void settle() noexcept
    {
        curPos_    = position_.load(std::memory_order_relaxed);
        lastCurve_ = curve_.load(std::memory_order_relaxed);
        gainsFor(lastCurve_, curPos_, gainA_, gainB_);
        publishGains();
    }

    /** @brief The first processing call after construction or reset() starts settled. */
    void settleIfFresh() noexcept
    {
        if (!needsSettle_) return;
        needsSettle_ = false;
        settle();
    }

    /** @brief Publishes the working gain pair for cross-thread metering. */
    void publishGains() noexcept
    {
        publishedGainA_.store(gainA_, std::memory_order_relaxed);
        publishedGainB_.store(gainB_, std::memory_order_relaxed);
    }

    // Communication: GUI -> Audio Thread
    std::atomic<Curve> curve_ { Curve::EqualPower };
    std::atomic<T> position_ { T(0) };

    // Audio-thread state: the applied (gliding) position, its curve and gains.
    // The working pair stays audio-thread-private; the getters read the
    // published copies.
    T curPos_ = T(0);
    Curve lastCurve_ = Curve::EqualPower;
    T gainA_ = T(1);
    T gainB_ = T(0);
    T maxPosStep_ = T(0);   ///< Glide rate per sample; 0 = unprepared (block ramp).
    bool needsSettle_ = true;   ///< Next processing call starts settled.
    std::atomic<T> publishedGainA_ { T(1) };
    std::atomic<T> publishedGainB_ { T(0) };
};

} // namespace dspark
