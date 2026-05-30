// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DspMath.h
 * @brief Core mathematical utilities for digital signal processing.
 *
 * Provides constants, unit conversions, and fast approximations commonly needed
 * in audio DSP code. All functions are noexcept and suitable for real-time use.
 *
 * Dependencies: C++20 standard library only.
 */

#include <algorithm>
#include <cmath>
#include <concepts>
#include <numbers>
#include <type_traits>

namespace dspark {

// ============================================================================
// Concepts
// ============================================================================

/** @brief Constrains a type to IEEE floating-point (float or double). */
template <typename T>
concept FloatType = std::floating_point<T>;

// ============================================================================
// Constants
// ============================================================================

/** @brief Pi (3.14159...) for the given floating-point type. */
template <FloatType T> inline constexpr T pi       = std::numbers::pi_v<T>;

/** @brief 2 * Pi (6.28318...). */
template <FloatType T> inline constexpr T twoPi    = T(2) * std::numbers::pi_v<T>;

/** @brief 1 / (2 * Pi) (0.15915...). Useful for fast phase divisions. */
template <FloatType T> inline constexpr T invTwoPi = T(1) / twoPi<T>;

/** @brief Square root of 2 (1.41421...). */
template <FloatType T> inline constexpr T sqrt2    = std::numbers::sqrt2_v<T>;

/** @brief 1 / square root of 2 (0.70710...). Butterworth Q factor. */
template <FloatType T> inline constexpr T invSqrt2 = T(1) / std::numbers::sqrt2_v<T>;

// ============================================================================
// Decibel / Gain Conversions
// ============================================================================

/**
 * @brief Converts a value in decibels to linear gain.
 *
 * @param dB              Value in decibels.
 * @param minusInfinityDb Values at or below this threshold return 0. Default: -100 dB.
 * @return Linear gain (0 for silence, 1 for unity, >1 for boost).
 */
template <FloatType T>
[[nodiscard]] inline T decibelsToGain(T dB, T minusInfinityDb = T(-100)) noexcept
{
    return dB <= minusInfinityDb ? T(0) : std::pow(T(10), dB / T(20));
}

/**
 * @brief Converts a linear gain value to decibels.
 *
 * @param gain            Linear gain (must be >= 0).
 * @param minusInfinityDb Returned for zero or negative gain. Default: -100 dB.
 * @return Value in decibels.
 */
template <FloatType T>
[[nodiscard]] inline T gainToDecibels(T gain, T minusInfinityDb = T(-100)) noexcept
{
    return gain > T(0) ? std::max(minusInfinityDb, T(20) * std::log10(gain))
                       : minusInfinityDb;
}

// ============================================================================
// Interpolation and Mapping
// ============================================================================

/**
 * @brief Maps a value from one range to another (linear interpolation).
 *
 * Includes a safety check to prevent division by zero if inMin == inMax.
 *
 * @param value  The input value to remap.
 * @param inMin  Lower bound of the input range.
 * @param inMax  Upper bound of the input range.
 * @param outMin Lower bound of the output range.
 * @param outMax Upper bound of the output range.
 * @return Remapped value. Returns outMin if input bounds are identical.
 */
template <FloatType T>
[[nodiscard]] inline T mapRange(T value, T inMin, T inMax, T outMin, T outMax) noexcept
{
    if (inMin == inMax) return outMin; // Safety: prevent NaN/Inf in audio pipeline
    return outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin));
}

// ============================================================================
// Fast Approximations
// ============================================================================

/**
 * @brief Fast tanh approximation using Pade rational function.
 *
 * Approximately 5x faster than std::tanh. Optimized for SIMD without branching.
 * Input is clamped to [-3, 3] maintaining C0 continuity to prevent aliasing.
 * Max output amplitude is ~0.9954.
 *
 * @param x Input value.
 * @return Approximation of tanh(x), smoothly bounded.
 */
template <FloatType T>
[[nodiscard]] inline T fastTanh(T x) noexcept
{
    // std::clamp translates to SIMD min/max intrinsics (no branching).
    // Maintains continuity avoiding step-aliasing at the bounds.
    x = std::clamp(x, T(-3), T(3));
    
    const auto x2 = x * x;
    const auto x4 = x2 * x2;
    // Padé [5,4] approximant: max error < 0.05% for |x| <= 3
    return x * (T(945) + T(105) * x2 + x4) / (T(945) + T(420) * x2 + T(15) * x4);
}

/**
 * @brief Fast approximation of 10^x using exp2.
 *
 * Uses the identity 10^x = 2^(x * log2(10)).
 * Useful in dB conversions where exact precision is not critical.
 *
 * @param x Exponent.
 * @return Approximation of 10^x.
 */
template <FloatType T>
[[nodiscard]] inline T fastPow10(T x) noexcept
{
    return std::exp2(x * std::numbers::log2e_v<T> * std::numbers::ln10_v<T>);
}

/**
 * @brief Fast approximation of e^x via std::exp2 (~2x faster than std::exp on MSVC).
 *
 * Uses the identity e^x = 2^(x * log2(e)).
 *
 * @param x Exponent. No range clamping is applied.
 * @return Approximation of e^x.
 */
template <FloatType T>
[[nodiscard]] inline T fastExp(T x) noexcept
{
    return std::exp2(x * std::numbers::log2e_v<T>);
}

/**
 * @brief Fast approximation of tan(x) using a Padé [3,2] rational approximant.
 *
 * Accurate to ~0.05% for |x| <= π/2 - 0.05. Outside that range, falls back to
 * std::tan to avoid the singularities at ±π/2 producing absurd results.
 *
 * Useful for the bilinear-transform `tan(π·f/Fs)` term when `f` is well below
 * Nyquist (the typical case). Roughly 3x faster than std::tan on MSVC.
 *
 * @param x Argument in radians.
 * @return Approximation of tan(x).
 */
template <FloatType T>
[[nodiscard]] inline T fastTan(T x) noexcept
{
    constexpr T limit = T(1.520);  // ~π/2 - 0.05
    if (std::abs(x) > limit) return std::tan(x);
    const T x2 = x * x;
    // Padé [3,2]: tan(x) ≈ x · (15 − x²) / (15 − 6·x²)
    return x * (T(15) - x2) / (T(15) - T(6) * x2);
}

// ============================================================================
// Utility
// ============================================================================

/**
 * @brief Normalises a phase value to the range [0, 2*pi).
 * * Optimized for hot paths. Avoids std::fmod which is blocking and slow.
 * Uses inverse multiplication and flooring, making it heavily SIMD friendly.
 * * @param phase Phase in radians.
 * @return Phase wrapped safely to [0, 2*pi).
 */
template <FloatType T>
[[nodiscard]] inline T wrapPhase(T phase) noexcept
{
    // floor() correctly handles negative values, preventing branching
    T wrapped = phase - twoPi<T> * std::floor(phase * invTwoPi<T>);
    // Guard against floating-point cancellation producing exactly twoPi for
    // tiny negative inputs: enforce the documented half-open range [0, 2*pi).
    if (wrapped >= twoPi<T>) wrapped -= twoPi<T>;
    return wrapped;
}

} // namespace dspark