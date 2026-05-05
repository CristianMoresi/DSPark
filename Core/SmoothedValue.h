// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include "DspMath.h"

#include <cmath>
#include <algorithm>
#include <span>

namespace dspark {

/**
 * @class SmoothedValue
 * @brief Zero-allocation, SIMD-friendly parameter smoother for real-time audio.
 *
 * Provides click-free parameter transitions using various curves. This class is
 * designed to be processed per-sample or per-block in the audio thread.
 * * @note **Thread Safety:** This class is not inherently thread-safe to avoid 
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
        Chase         ///< Adaptive speed (Airwindows style). Fast attack, smooth settle.
    };

    /** @brief Prevents accidental copying of stateful DSP objects. */
    SmoothedValue(const SmoothedValue&) = delete;
    SmoothedValue& operator=(const SmoothedValue&) = delete;
    SmoothedValue() = default;

    /**
     * @brief Precalculates internal coefficients based on sample rate and timing.
     * @param sampleRate Sample rate in Hz (must be > 0).
     * @param rampTimeMs Smoothing duration in milliseconds (must be > 0).
     */
    void prepare(double sampleRate, double rampTimeMs = 20.0) noexcept
    {
        sampleRate_ = std::max(1.0, sampleRate);
        rampTimeMs_ = std::max(0.1, rampTimeMs);

        // Exponential: 1-pole coefficient
        double tau = rampTimeMs_ / 1000.0;
        expCoeff_ = static_cast<T>(std::exp(-1.0 / (sampleRate_ * tau)));

        // Linear: Rate of change per sample (Rate Limiter)
        linearRate_ = static_cast<T>(1000.0 / (rampTimeMs_ * sampleRate_));

        // Chase: Sample-rate correction ratio relative to 44.1kHz base
        double srRatio = 44100.0 / sampleRate_;
        chaseMultDecay_ = static_cast<T>(std::pow(0.9999, srRatio));
        chaseAddDecay_ = static_cast<T>(0.01 * srRatio);
    }

    /** @brief Sets the smoothing algorithm. */
    void setSmoothingType(SmoothingType type) noexcept { type_ = type; }

    /** @brief Returns the current smoothing algorithm. */
    [[nodiscard]] SmoothingType getSmoothingType() const noexcept { return type_; }

    /**
     * @brief Updates the target value. Safe to call continuously (e.g., from host automation).
     * @param newTarget The destination value.
     */
    void setTargetValue(T newTarget) noexcept
    {
        if (newTarget != target_)
        {
            target_ = newTarget;
            if (type_ == SmoothingType::Chase)
                chaseSpeed_ = T(2500); // Reset chase velocity on target change
        }
    }

    /**
     * @brief Calculates and returns the next smoothed value.
     * @return The updated current value.
     */
    [[nodiscard]] T getNextValue() noexcept
    {
        if (current_ == target_) return current_;

        switch (type_)
        {
            case SmoothingType::Exponential:
            {
                current_ = target_ + expCoeff_ * (current_ - target_);
                break;
            }
            case SmoothingType::Linear:
            {
                if (current_ < target_)
                    current_ = std::min(current_ + linearRate_, target_);
                else
                    current_ = std::max(current_ - linearRate_, target_);
                break;
            }
            case SmoothingType::Disabled:
            {
                current_ = target_;
                break;
            }
            case SmoothingType::Chase:
            {
                chaseSpeed_ = std::max(T(350), std::min(T(2500), chaseSpeed_ * chaseMultDecay_ - chaseAddDecay_));
                current_ = (current_ * chaseSpeed_ + target_) / (chaseSpeed_ + T(1));
                break;
            }
        }

        // Anti-denormal protection & exact arrival
        if (std::abs(current_ - target_) < epsilon_)
            current_ = target_;

        return current_;
    }

    /**
     * @brief Computes a block of smoothed values into an output buffer.
     * * Highly recommended over getNextValue() for SIMD auto-vectorization
     * and reducing branch overhead.
     * * @param buffer Span representing the output buffer to fill.
     */
    void processBlock(std::span<T> buffer) noexcept
    {
        if (current_ == target_)
        {
            std::fill(buffer.begin(), buffer.end(), target_);
            return;
        }

        // Branch pulled OUTSIDE the loop for CPU instruction cache / SIMD efficiency
        switch (type_)
        {
            case SmoothingType::Exponential:
                for (auto& sample : buffer)
                {
                    current_ = target_ + expCoeff_ * (current_ - target_);
                    sample = current_;
                }
                break;

            case SmoothingType::Linear:
                for (auto& sample : buffer)
                {
                    if (current_ < target_)      current_ = std::min(current_ + linearRate_, target_);
                    else if (current_ > target_) current_ = std::max(current_ - linearRate_, target_);
                    sample = current_;
                }
                break;

            case SmoothingType::Disabled:
                current_ = target_;
                std::fill(buffer.begin(), buffer.end(), target_);
                break;

            case SmoothingType::Chase:
                for (auto& sample : buffer)
                {
                    chaseSpeed_ = std::max(T(350), std::min(T(2500), chaseSpeed_ * chaseMultDecay_ - chaseAddDecay_));
                    current_ = (current_ * chaseSpeed_ + target_) / (chaseSpeed_ + T(1));
                    sample = current_;
                }
                break;
        }

        if (std::abs(current_ - target_) < epsilon_)
            current_ = target_;
    }

    /** @brief Returns the current internal value without advancing the state. */
    [[nodiscard]] T getCurrentValue() const noexcept { return current_; }

    /** @brief Returns the set target value. */
    [[nodiscard]] T getTargetValue() const noexcept { return target_; }

    /** @brief Evaluates if the smoother is actively transitioning. */
    [[nodiscard]] bool isSmoothing() const noexcept { return current_ != target_; }

    /** @brief Forces the current value to instantly match the target, bypassing time. */
    void skip() noexcept { current_ = target_; chaseSpeed_ = T(350); }

    /** @brief Hard-resets both current and target states to a specific value. */
    void reset(T value = T(0)) noexcept
    {
        current_ = target_ = value;
        chaseSpeed_ = T(350);
    }

    /** @brief Updates sample rate and/or ramp time, recalculating internal steps. */
    void setRampTime(double sampleRate, double rampTimeMs) noexcept
    {
        prepare(sampleRate, rampTimeMs);
    }

private:
    T current_{ T(0) };
    T target_{ T(0) };
    SmoothingType type_{ SmoothingType::Exponential };

    // DSP Coefficients
    T expCoeff_{ T(0) };
    T linearRate_{ T(0) };
    
    // Chase state & coefficients
    T chaseSpeed_{ T(350) };
    T chaseMultDecay_{ T(0.9999) };
    T chaseAddDecay_{ T(0.01) };

    double sampleRate_{ 44100.0 };
    double rampTimeMs_{ 20.0 };

    static constexpr T epsilon_ = T(1e-7); ///< Threshold for anti-denormal snapping
};

} // namespace dspark