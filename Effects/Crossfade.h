// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Crossfade.h
 * @brief Crossfade between two audio signals with selectable curve and artifact-free parameter smoothing.
 *
 * Provides highly optimized crossfade curves for audio transitions. Designed for zero-allocation
 * real-time contexts, featuring automatic block-level parameter smoothing to prevent zipper noise.
 *
 * Curves:
 * - **Linear:** Constant sum (A * (1-t) + B * t). Fast, but exhibits a -3 dB power dip at the centre for uncorrelated signals.
 * - **EqualPower:** Constant energy (A^2 + B^2 = 1) via the hardware sqrt instruction. Perfect for uncorrelated audio.
 * - **SCurve:** Smoothstep S-curve. Provides a perceptually smoother transition speed, though it shares the -3 dB centre power dip of the Linear curve.
 *
 * Dependencies: Core/DspMath.h.
 *
 * Threading: setCurve()/setPosition() are safe from any thread (atomics,
 * consumed by the processing calls on the audio thread); non-finite positions
 * are ignored. The processing calls and the gain getters belong to the audio
 * thread (getters read unsynchronized audio-thread state: metering only).
 */

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
        EqualPower, ///< Equal power interpolation (hardware SQRT). Constant power, no volume drop.
        SCurve      ///< Smoothstep interpolation. Slower progression at extremes.
    };

    Crossfade() = default;
    ~Crossfade() = default;

    /**
     * @brief Sets the crossfade curve type.
     * Thread-safe. Can be called from the GUI thread.
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
     * Thread-safe. Updates are smoothed automatically over the next processed audio block
     * to eliminate zipper noise and clicks.
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
     * @brief Crossfades between two individual samples.
     *
     * @warning This method updates internal gain states immediately. It does NOT provide 
     * parameter smoothing. Use the block-based `process` for artifact-free parameter changes.
     * Must only be called from the audio thread.
     *
     * @param a Input sample A (Dry/Left).
     * @param b Input sample B (Wet/Right).
     * @return Blended output sample.
     */
    [[nodiscard]] inline T process(T a, T b) noexcept
    {
        refreshGainsFromAtomics();
        return a * gainA_ + b * gainB_;
    }

    /**
     * @brief Crossfades two audio buffers into an output buffer with automatic parameter smoothing.
     *
     * If the parameter has changed since the last block, this method automatically applies a linear 
     * ramp to the gains across the block to prevent zipper noise. Otherwise, it executes a highly 
     * optimized, autovectorization-friendly static gain loop.
     *
     * @param inputA Pointer to the first input buffer array. Must not be null.
     * @param inputB Pointer to the second input buffer array. Must not be null.
     * @param output Pointer to the output buffer array. Must not be null.
     * @param numSamples Number of samples to process. Must be > 0.
     */
    void process(const T* inputA, const T* inputB, T* output, int numSamples) noexcept
    {
        assert(inputA != nullptr && inputB != nullptr && output != nullptr);
        assert(numSamples > 0);

        const T oldGainA = gainA_;
        const T oldGainB = gainB_;

        refreshGainsFromAtomics();

        if (oldGainA == gainA_ && oldGainB == gainB_)
        {
            // Hot-path: No parameter change. Ideal for SIMD autovectorization.
            const T gA = gainA_;
            const T gB = gainB_;
            for (int i = 0; i < numSamples; ++i)
                output[i] = inputA[i] * gA + inputB[i] * gB;
        }
        else
        {
            // Parameter change detected: Apply linear block smoothing (De-zippering)
            const T invSamples = T(1) / static_cast<T>(numSamples);
            const T stepA = (gainA_ - oldGainA) * invSamples;
            const T stepB = (gainB_ - oldGainB) * invSamples;

            T currentA = oldGainA;
            T currentB = oldGainB;

            for (int i = 0; i < numSamples; ++i)
            {
                currentA += stepA;
                currentB += stepB;
                output[i] = inputA[i] * currentA + inputB[i] * currentB;
            }
        }
    }

    /**
     * @brief Processes crossfading using a per-sample automation buffer.
     *
     * Extremely CPU intensive if used with heavy curves. EqualPower uses hardware SQRT 
     * to maintain real-time viability under per-sample automation.
     *
     * @param inputA Pointer to the first input buffer.
     * @param inputB Pointer to the second input buffer.
     * @param positions Array of target positions [0, 1] per sample.
     * @param output Pointer to the output buffer.
     * @param numSamples Number of samples to process.
     */
    void processAutomated(const T* inputA, const T* inputB,
                          const T* positions, T* output, int numSamples) noexcept
    {
        assert(inputA && inputB && positions && output);
        assert(numSamples > 0);

        Curve curv = curve_.load(std::memory_order_relaxed);
        T gA = gainA_, gB = gainB_;

        for (int i = 0; i < numSamples; ++i)
        {
            // min/max with this argument order also resolves a NaN in the
            // automation buffer to 0 (100% A) instead of poisoning the output.
            T pos = std::min(T(1), std::max(T(0), positions[i]));
            computeGains(curv, pos, gA, gB);
            output[i] = inputA[i] * gA + inputB[i] * gB;
        }

        // Update internal state to match the end of the automation block.
        // NOTE: We DO NOT write back to the atomic position_ to avoid Data Races
        // with the GUI thread. We only update the audio-thread local cache.
        lastPos_ = std::min(T(1), std::max(T(0), positions[numSamples - 1]));
        lastCurve_ = curv;
        gainA_ = gA;
        gainB_ = gB;
    }

    /** @brief Gets the current internal gain multiplier for signal A. */
    [[nodiscard]] T getGainA() const noexcept { return gainA_; }

    /** @brief Gets the current internal gain multiplier for signal B. */
    [[nodiscard]] T getGainB() const noexcept { return gainB_; }

private:
    /**
     * @brief Computes raw gain values based on the requested curve and position.
     */
    static inline void computeGains(Curve curve, T pos, T& gA, T& gB) noexcept
    {
        switch (curve)
        {
            case Curve::Linear:
                gA = T(1) - pos;
                gB = pos;
                break;

            case Curve::EqualPower:
                // SIMD Optimization: Replaced std::cos/sin with hardware-accelerated std::sqrt.
                // sqrt(1-x)^2 + sqrt(x)^2 = (1-x) + x = 1.0 (Constant Power).
                // Avoids heavy trigonometric penalties while maintaining identical energetic response.
                gA = std::sqrt(T(1) - pos);
                gB = std::sqrt(pos);
                break;

            case Curve::SCurve:
            {
                T t = pos * pos * (T(3) - T(2) * pos);
                gA = T(1) - t;
                gB = t;
                break;
            }

            default:
                // Unreachable through the clamped setter; keeps the out
                // parameters defined if the enum ever grows without a case.
                gA = T(1) - pos;
                gB = pos;
                break;
        }
    }

    /**
     * @brief Syncs GUI-driven atomic changes into the audio thread cache.
     */
    inline void refreshGainsFromAtomics() noexcept
    {
        T pos = position_.load(std::memory_order_relaxed);
        Curve curv = curve_.load(std::memory_order_relaxed);
        
        if (pos != lastPos_ || curv != lastCurve_)
        {
            computeGains(curv, pos, gainA_, gainB_);
            lastPos_   = pos;
            lastCurve_ = curv;
        }
    }

    // Communication: GUI -> Audio Thread
    std::atomic<Curve> curve_ { Curve::EqualPower };
    std::atomic<T> position_ { T(0) };
    
    // Audio Thread Local State (Cache)
    T gainA_ = T(1);
    T gainB_ = T(0);
    T lastPos_ = T(-1);  // Sentinel to force initial compute
    Curve lastCurve_ = Curve::EqualPower;
};

} // namespace dspark