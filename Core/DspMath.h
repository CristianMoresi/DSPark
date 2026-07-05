// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

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

/** @brief Pi / 2 (1.57079...). Quarter period; sin/cos phase offset. */
template <FloatType T> inline constexpr T halfPi   = std::numbers::pi_v<T> / T(2);

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
 * @note The clamp at |x| = 3 is C0 but not C1: the derivative has a small
 * discontinuity there (output is already ~0.9954, so the resulting kink is
 * inaudible in practice). Use std::tanh where exact C1 behaviour matters.
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
    // Pade [5,4] approximant: max error < 0.05% for |x| <= 3
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
    // log2(10) folded into one constant: rounding it once is more accurate
    // than multiplying log2(e) * ln(10) at runtime, and it saves a multiply
    // (compilers may not reassociate x * a * b without fast-math).
    constexpr T kLog2Of10 =
        static_cast<T>(std::numbers::ln10_v<long double> / std::numbers::ln2_v<long double>);
    return std::exp2(x * kLog2Of10);
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
 * @brief Fast approximation of tan(x) using a Pade [5,4] rational approximant.
 *
 * Accurate to better than 0.01% for |x| <= 1.45 and ~0.1% up to the 1.52
 * fallback limit. Outside that range, falls back to std::tan to avoid the
 * singularities at +-pi/2 producing absurd results.
 *
 * Useful for the bilinear-transform `tan(pi*f/Fs)` term: with this accuracy a
 * 20 kHz cutoff at 44.1 kHz lands within a fraction of a cent of the target.
 * Roughly 3x faster than std::tan on MSVC.
 *
 * @param x Argument in radians.
 * @return Approximation of tan(x).
 */
template <FloatType T>
[[nodiscard]] inline T fastTan(T x) noexcept
{
    constexpr T limit = T(1.520);  // ~pi/2 - 0.05
    if (std::abs(x) > limit) return std::tan(x);
    const T x2 = x * x;
    const T x4 = x2 * x2;
    // Pade [5,4]: tan(x) ~= x * (945 - 105*x^2 + x^4) / (945 - 420*x^2 + 15*x^4)
    return x * (T(945) - T(105) * x2 + x4) / (T(945) - T(420) * x2 + T(15) * x4);
}

/**
 * @brief Fast sine approximation (degree-9 odd minimax polynomial).
 *
 * Maximum error ~4e-6 in float (over 100 dB below the signal) and below 1e-7
 * in double: inaudible even for audio-rate synthesis in either precision.
 * About 3-6x faster than std::sin depending on platform. The input is
 * range-reduced internally (two-term Cody-Waite), so any finite argument
 * within a few thousand periods of zero stays accurate.
 *
 * @param x Argument in radians.
 * @return Approximation of sin(x).
 */
template <FloatType T>
[[nodiscard]] inline T fastSin(T x) noexcept
{
    // Range-reduce to [-pi, pi] with a two-term Cody-Waite split of 2*pi so the
    // subtraction stays accurate in float even for arguments several periods out.
    const T k = std::floor(x * invTwoPi<T> + T(0.5));
    x -= k * T(6.28125);                  // high part (exactly representable)
    x -= k * T(0.0019353071795864769253); // low part of 2*pi

    // Fold into [-pi/2, pi/2] where the polynomial converges fast:
    // sin(x) = sin(pi - x) for x > pi/2 (and the odd mirror for x < -pi/2).
    if (x > halfPi<T>)       x = pi<T> - x;
    else if (x < -halfPi<T>) x = -pi<T> - x;

    const T x2 = x * x;
    // Minimax coefficients for sin on [-pi/2, pi/2], max abs error ~6e-8.
    return x * (T(0.9999999995)
         + x2 * (T(-0.1666666580)
         + x2 * (T(0.0083333075)
         + x2 * (T(-0.0001984090)
         + x2 *  T(0.0000027526)))));
}

/**
 * @brief Fast cosine approximation. See fastSin() for accuracy notes
 * (float error is ~7e-6 here: half an ulp more from the pi/2 offset).
 * @param x Argument in radians.
 * @return Approximation of cos(x).
 */
template <FloatType T>
[[nodiscard]] inline T fastCos(T x) noexcept
{
    return fastSin(x + halfPi<T>);
}

/**
 * @brief Fast natural logarithm approximation.
 *
 * Splits the input into exponent and mantissa via std::frexp, then evaluates
 * the classic 4-term atanh series (Cephes style) for ln of the mantissa,
 * centred on [sqrt(0.5), sqrt(2)). Relative error is below 2e-7: ample for
 * envelope/dB work. Roughly 2-4x faster than std::log.
 *
 * @param x Input value. Must be > 0 (no guard: matches std::log contract).
 * @return Approximation of ln(x).
 */
template <FloatType T>
[[nodiscard]] inline T fastLog(T x) noexcept
{
    int e = 0;
    T m = std::frexp(x, &e);             // x = m * 2^e, m in [0.5, 1)
    // Normalise mantissa into [sqrt(0.5), sqrt(2)) so the series is centred.
    if (m < T(0.70710678118654752440)) { m *= T(2); --e; }

    // Classic atanh form (Cephes): ln(m) = 2*atanh(s), s = (m-1)/(m+1).
    // |s| <= 0.1716 so the odd series converges below 1e-7 with 4 terms.
    const T s  = (m - T(1)) / (m + T(1));
    const T s2 = s * s;
    const T lnm = T(2) * s * (T(1)
                + s2 * (T(1.0 / 3.0)
                + s2 * (T(1.0 / 5.0)
                + s2 *  T(1.0 / 7.0))));

    return lnm + static_cast<T>(e) * std::numbers::ln2_v<T>; // + e * ln2
}

// ============================================================================
// Utility
// ============================================================================

/**
 * @brief Normalises a phase value to the range [0, 2*pi).
 *
 * Optimized for hot paths: avoids std::fmod (slow, hostile to vectorisation)
 * in favour of inverse multiplication and flooring.
 *
 * @param phase Phase in radians.
 * @return Phase wrapped safely to [0, 2*pi).
 */
template <FloatType T>
[[nodiscard]] inline T wrapPhase(T phase) noexcept
{
    // floor() correctly handles negative values, preventing branching
    T wrapped = phase - twoPi<T> * std::floor(phase * invTwoPi<T>);
    // Rounding in the k * twoPi product can land wrapped just outside
    // [0, 2*pi) on either side: large phases where the product overshoots
    // the input leave a slightly negative result, and tiny negative phases
    // can round the sum up to exactly twoPi. Enforce the documented
    // half-open range. Order matters: the += branch can itself round up to
    // exactly twoPi, which the second branch then folds to 0.
    if (wrapped < T(0))      wrapped += twoPi<T>;
    if (wrapped >= twoPi<T>) wrapped -= twoPi<T>;
    return wrapped;
}

} // namespace dspark