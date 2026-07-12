// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file SmoothedValue.h
 * @brief Click-free parameter smoother with selectable curves.
 *
 * Wraps a parameter value in a per-sample or per-block ramp (exponential,
 * linear, adaptive chase, or disabled) so that host automation and UI edits
 * never step audibly. The workhorse behind the de-zippered setters of the
 * effect classes.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <cmath>
#include <algorithm>
#include <span>

namespace dspark {

/**
 * @class SmoothedValue
 * @brief Zero-allocation parameter smoother for real-time audio.
 *
 * Provides click-free parameter transitions using various curves, processed
 * per-sample or per-block in the audio thread. A default-constructed smoother
 * is functional at 44.1 kHz with a 20 ms ramp; prepare() adapts it to the
 * real rate.
 *
 * Timing semantics of `rampTimeMs` per curve:
 * - **Exponential**: one-pole time constant (63% arrival). The value snaps
 *   exactly onto the target once within a relative epsilon, about 16 time
 *   constants after a unit step, and isSmoothing() then reports false.
 * - **Linear**: constant velocity sized to traverse a UNIT step in
 *   rampTimeMs; smaller steps arrive proportionally sooner, exactly on
 *   target.
 * - **Chase**: adaptive (not clocked): gentle right after a target change,
 *   accelerating as it settles.
 *
 * The recursion state is double regardless of T (the framework rule for
 * recursive state): in float the one-pole stalls hundreds of epsilons short
 * of the target when the per-sample increment rounds to zero, which would
 * leave isSmoothing() true forever (and callers that gate work on it, like
 * Gain's bulk path or CrossoverFilter's coefficient updates, stuck on their
 * slow path).
 *
 * @note **Thread Safety:** This class is not inherently thread-safe to avoid
 * atomic memory barriers in the DSP hot-path. Parameter updates via
 * `setTargetValue()` should occur synchronously within the audio thread
 * (e.g., processing an event queue at the start of a block).
 *
 * @tparam T Value type (float or double, constrained by FloatType).
 */
template <FloatType T>
class SmoothedValue
{
public:
    enum class SmoothingType
    {
        Exponential,  ///< One-pole IIR. Natural, musical interpolation.
        Linear,       ///< Rate-limited ramp. Constant velocity, exact target arrival.
        Disabled,     ///< Instant snapping, no smoothing applied.
        Chase         ///< Adaptive speed (Airwindows style). Gentle start after a jump, accelerating settle.
    };

    /** @brief Prevents accidental copying of stateful DSP objects. */
    SmoothedValue(const SmoothedValue&) = delete;
    SmoothedValue& operator=(const SmoothedValue&) = delete;

    SmoothedValue() noexcept { prepare(sampleRate_, rampTimeMs_); }

    /**
     * @brief Precalculates internal coefficients based on sample rate and timing.
     * @param sampleRate Sample rate in Hz (must be > 0).
     * @param rampTimeMs Smoothing duration in milliseconds (must be > 0).
     *                   See the class notes for the per-curve semantics.
     */
    void prepare(double sampleRate, double rampTimeMs = 20.0) noexcept
    {
        sampleRate_ = std::max(1.0, sampleRate);
        rampTimeMs_ = std::max(0.1, rampTimeMs);

        // Exponential: 1-pole coefficient
        const double tau = rampTimeMs_ / 1000.0;
        expCoeff_ = std::exp(-1.0 / (sampleRate_ * tau));

        // Linear: Rate of change per sample (Rate Limiter)
        linearRate_ = 1000.0 / (rampTimeMs_ * sampleRate_);

        // Chase: Sample-rate correction ratio relative to 44.1kHz base
        const double srRatio = 44100.0 / sampleRate_;
        chaseMultDecay_ = std::pow(0.9999, srRatio);
        chaseAddDecay_ = 0.01 * srRatio;
    }

    /** @brief Sets the smoothing algorithm. */
    void setSmoothingType(SmoothingType type) noexcept { type_ = type; }

    /** @brief Returns the current smoothing algorithm. */
    [[nodiscard]] SmoothingType getSmoothingType() const noexcept { return type_; }

    /**
     * @brief Updates the target value. Safe to call continuously (e.g., from host automation).
     * @param newTarget The destination value. NaN is ignored (it would poison
     *                  the recursion and the arrival checks permanently).
     */
    void setTargetValue(T newTarget) noexcept
    {
        if (newTarget != newTarget) return; // NaN guard
        if (newTarget != target_)
        {
            target_ = newTarget;
            if (type_ == SmoothingType::Chase)
                chaseSpeed_ = 2500.0; // Reset chase velocity on target change
        }
    }

    /**
     * @brief Calculates and returns the next smoothed value.
     * @return The updated current value.
     */
    [[nodiscard]] T getNextValue() noexcept
    {
        const double target = static_cast<double>(target_);
        if (current_ == target) return target_;

        switch (type_)
        {
            case SmoothingType::Exponential:
            {
                current_ = target + expCoeff_ * (current_ - target);
                break;
            }
            case SmoothingType::Linear:
            {
                if (current_ < target)
                    current_ = std::min(current_ + linearRate_, target);
                else
                    current_ = std::max(current_ - linearRate_, target);
                break;
            }
            case SmoothingType::Disabled:
            {
                current_ = target;
                break;
            }
            case SmoothingType::Chase:
            {
                chaseSpeed_ = std::max(350.0, std::min(2500.0, chaseSpeed_ * chaseMultDecay_ - chaseAddDecay_));
                current_ = (current_ * chaseSpeed_ + target) / (chaseSpeed_ + 1.0);
                break;
            }
        }

        // Exact arrival: snap once within a relative epsilon of the target.
        // Relative because with large magnitudes (e.g. a frequency of 10000)
        // a purely absolute threshold would be finer than the float ulp.
        const double eps = kEpsilon * std::max(1.0, std::abs(target));
        if (std::abs(current_ - target) < eps)
            current_ = target;

        return static_cast<T>(current_);
    }

    /**
     * @brief Computes a block of smoothed values into an output buffer.
     *
     * Recommended over per-sample getNextValue() calls: the curve branch is
     * resolved once and the recursion state stays in registers. (The
     * recursion itself is serial, so the loop does not vectorise; a settled
     * smoother short-circuits to a plain fill.) Sample-exact match with
     * getNextValue().
     *
     * @param buffer Span representing the output buffer to fill.
     */
    void processBlock(std::span<T> buffer) noexcept
    {
        const double target = static_cast<double>(target_);
        const size_t n = buffer.size();
        if (current_ == target)
        {
            std::fill(buffer.begin(), buffer.end(), target_);
            return;
        }

        // Work on locals: the compiler cannot prove the span does not alias
        // the members, so member accesses would be re-loaded each iteration.
        // The per-sample arrival snap matches getNextValue(); once settled,
        // the rest of the block is a plain fill.
        double current = current_;
        const double eps = kEpsilon * std::max(1.0, std::abs(target));
        size_t i = 0;

        switch (type_)
        {
            case SmoothingType::Exponential:
            {
                const double coeff = expCoeff_;
                for (; i < n; ++i)
                {
                    current = target + coeff * (current - target);
                    if (std::abs(current - target) < eps) { current = target; break; }
                    buffer[i] = static_cast<T>(current);
                }
                break;
            }

            case SmoothingType::Linear:
            {
                const double rate = linearRate_;
                for (; i < n; ++i)
                {
                    if (current < target)      current = std::min(current + rate, target);
                    else if (current > target) current = std::max(current - rate, target);
                    if (std::abs(current - target) < eps) { current = target; break; }
                    buffer[i] = static_cast<T>(current);
                }
                break;
            }

            case SmoothingType::Disabled:
                current = target;
                break;

            case SmoothingType::Chase:
            {
                double speed = chaseSpeed_;
                const double multDecay = chaseMultDecay_;
                const double addDecay  = chaseAddDecay_;
                for (; i < n; ++i)
                {
                    speed = std::max(350.0, std::min(2500.0, speed * multDecay - addDecay));
                    current = (current * speed + target) / (speed + 1.0);
                    if (std::abs(current - target) < eps) { current = target; break; }
                    buffer[i] = static_cast<T>(current);
                }
                chaseSpeed_ = speed;
                break;
            }
        }

        // Settled (or Disabled): the remainder of the block is the target.
        for (; i < n; ++i)
            buffer[i] = target_;

        current_ = current;
    }

    /** @brief Returns the current internal value without advancing the state. */
    [[nodiscard]] T getCurrentValue() const noexcept { return static_cast<T>(current_); }

    /** @brief Returns the set target value. */
    [[nodiscard]] T getTargetValue() const noexcept { return target_; }

    /** @brief Evaluates if the smoother is actively transitioning. */
    [[nodiscard]] bool isSmoothing() const noexcept { return current_ != static_cast<double>(target_); }

    /** @brief Forces the current value to instantly match the target, bypassing time. */
    void skip() noexcept { current_ = static_cast<double>(target_); chaseSpeed_ = 350.0; }

    /** @brief Hard-resets both current and target states to a specific value. */
    void reset(T value = T(0)) noexcept
    {
        target_ = value;
        current_ = static_cast<double>(value);
        chaseSpeed_ = 350.0;
    }

    /** @brief Updates sample rate and/or ramp time, recalculating internal steps. */
    void setRampTime(double sampleRate, double rampTimeMs) noexcept
    {
        prepare(sampleRate, rampTimeMs);
    }

private:
    double current_{ 0.0 };
    T target_{ T(0) };
    SmoothingType type_{ SmoothingType::Exponential };

    // DSP coefficients (double: recursive state and its drivers, see @class)
    double expCoeff_{ 0.0 };
    double linearRate_{ 0.0 };

    // Chase state & coefficients
    double chaseSpeed_{ 350.0 };
    double chaseMultDecay_{ 0.9999 };
    double chaseAddDecay_{ 0.01 };

    double sampleRate_{ 44100.0 };
    double rampTimeMs_{ 20.0 };

    static constexpr double kEpsilon = 1e-7; ///< Relative arrival threshold
};

} // namespace dspark
