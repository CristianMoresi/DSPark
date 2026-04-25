// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Crossfade.h
 * @brief Crossfade between two audio signals with selectable curve.
 *
 * Provides three crossfade curves for smooth transitions between audio sources.
 * Useful for preset morphing, scene transitions, A/B comparison, and layer mixing.
 *
 * Curves:
 * - **Linear:** Constant sum (A * (1-t) + B * t). Simple but has a -6 dB dip at centre.
 * - **EqualPower:** Constant power (A * cos(t*pi/2) + B * sin(t*pi/2)). No level dip.
 * - **SCurve:** Smooth S-curve (smoothstep). Starts and ends slowly, fast in the middle.
 *
 * Dependencies: DspMath.h.
 *
 * @code
 *   dspark::Crossfade<float> xfade;
 *   xfade.setCurve(dspark::Crossfade<float>::Curve::EqualPower);
 *   xfade.setPosition(0.5f);  // 50% blend
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = xfade.process(inputA[i], inputB[i]);
 * @endcode
 */

#include "../Core/DspMath.h"

#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Crossfade
 * @brief Crossfades between two signals with configurable curve.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Crossfade
{
public:
    virtual ~Crossfade() = default;
    /** @brief Crossfade curve type. */
    enum class Curve
    {
        Linear,     ///< Linear interpolation (constant sum).
        EqualPower, ///< Equal-power crossfade (constant energy).
        SCurve      ///< Smooth S-curve (smoothstep).
    };

    /**
     * @brief Sets the crossfade curve.
     * @param curve Curve type.
     */
    void setCurve(Curve curve) noexcept { curve_.store(curve, std::memory_order_relaxed); }

    /**
     * @brief Sets the crossfade position.
     *
     * Thread-safe: only publishes the new position atomically. The gain pair
     * (gainA_, gainB_) is recomputed lazily by the audio-thread hot path on
     * next use — avoids torn-pair reads that would break the constant-power
     * invariant (gainA² + gainB² = 1) during concurrent updates.
     *
     * @param position Blend position: 0.0 = 100% A, 1.0 = 100% B.
     */
    void setPosition(T position) noexcept
    {
        position_.store(std::clamp(position, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current position.
     */
    [[nodiscard]] T getPosition() const noexcept { return position_.load(std::memory_order_relaxed); }

    /**
     * @brief Crossfades between two samples.
     *
     * NOTE: no longer `const` — the call refreshes the cached gain pair when
     * the atomic position changes, which is a state mutation. Only callable
     * from a single (audio) thread.
     *
     * @param a First input sample (dry / scene A).
     * @param b Second input sample (wet / scene B).
     * @return Blended output.
     */
    [[nodiscard]] T process(T a, T b) noexcept
    {
        refreshGainsFromAtomics();
        return a * gainA_ + b * gainB_;
    }

    /**
     * @brief Crossfades two buffers into an output buffer.
     *
     * @param inputA First input buffer.
     * @param inputB Second input buffer.
     * @param output Output buffer.
     * @param numSamples Number of samples.
     */
    void process(const T* inputA, const T* inputB, T* output,
                 int numSamples) noexcept
    {
        refreshGainsFromAtomics();
        const T gA = gainA_;
        const T gB = gainB_;
        for (int i = 0; i < numSamples; ++i)
            output[i] = inputA[i] * gA + inputB[i] * gB;
    }

    /**
     * @brief Crossfades with per-sample position automation.
     *
     * @param inputA First input buffer.
     * @param inputB Second input buffer.
     * @param positions Per-sample position values (0 to 1).
     * @param output Output buffer.
     * @param numSamples Number of samples.
     */
    void processAutomated(const T* inputA, const T* inputB,
                          const T* positions, T* output,
                          int numSamples) noexcept
    {
        Curve curv = curve_.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            T pos = std::clamp(positions[i], T(0), T(1));
            // Recompute locally — no shared state writes.
            T gA, gB;
            computeGains(curv, pos, gA, gB);
            output[i] = inputA[i] * gA + inputB[i] * gB;
        }
        // Publish the last position so subsequent static-position calls see a
        // consistent value with the automation's endpoint.
        position_.store(std::clamp(positions[numSamples - 1], T(0), T(1)),
                        std::memory_order_relaxed);
        lastPos_    = position_.load(std::memory_order_relaxed);
        lastCurve_  = curv;
        computeGains(curv, lastPos_, gainA_, gainB_);
    }

    /**
     * @brief Returns the current gain for signal A.
     */
    [[nodiscard]] T getGainA() const noexcept { return gainA_; }

    /**
     * @brief Returns the current gain for signal B.
     */
    [[nodiscard]] T getGainB() const noexcept { return gainB_; }

protected:
    static void computeGains(Curve curve, T pos, T& gA, T& gB) noexcept
    {
        switch (curve)
        {
            case Curve::Linear:
                gA = T(1) - pos;
                gB = pos;
                break;

            case Curve::EqualPower:
            {
                constexpr T halfPi = pi<T> / T(2);
                gA = std::cos(pos * halfPi);
                gB = std::sin(pos * halfPi);
                break;
            }

            case Curve::SCurve:
            {
                T t = pos * pos * (T(3) - T(2) * pos);
                gA = T(1) - t;
                gB = t;
                break;
            }
        }
    }

    /// Audio-thread-only: refreshes gainA_/gainB_ if atomic position or curve
    /// has changed since the last call. Cheap branch for the common no-change
    /// case; cos/sin only on actual automation.
    void refreshGainsFromAtomics() noexcept
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

    std::atomic<Curve> curve_ { Curve::EqualPower };
    std::atomic<T> position_ { T(0) };
    // Audio-thread-owned cache: written only by refreshGainsFromAtomics() /
    // processAutomated(), never by the GUI setters.
    T gainA_ = T(1);
    T gainB_ = T(0);
    T lastPos_ = T(-1);  // sentinel: first process() call always recomputes
    Curve lastCurve_ = Curve::EqualPower;
};

} // namespace dspark
