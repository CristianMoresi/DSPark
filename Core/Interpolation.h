// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Interpolation.h
 * @brief Sample-accurate, high-performance interpolation algorithms for audio.
 *
 * Provides highly optimized interpolation methods. Designed for SIMD auto-vectorization
 * and strict real-time constraints (no modulo operators, no floating-point divisions).
 * * | Method    | Points | Quality       | CPU Cost | Use case                     |
 * |-----------|--------|---------------|----------|------------------------------|
 * | Linear    | 2      | Low           | Ultra-Low| LFOs, crossfades             |
 * | Hermite   | 4      | Good+         | Low      | Modulated delays (default)   |
 * | Lagrange  | 4      | High          | Low      | Precision resampling         |
 * | Allpass   | 2      | Frequency-dep | Low      | Static fractional delays     |
 *
 * Hermite is the recommended default for modulated delay lines (chorus,
 * vibrato, reverb modulation): it is stateless, so fast `frac` changes never
 * destabilise anything — unlike the recursive Allpass interpolator.
 *
 * @note For maximum SIMD performance, prefer the overloads that take raw samples
 * (y0, y1, y2, y3) directly, allowing your container class to handle buffer wrapping.
 */

#include "DspMath.h"
#include <cassert>

namespace dspark {

/**
 * @brief Linear interpolation between two adjacent samples.
 *
 * @param y0 Sample at index n.
 * @param y1 Sample at index n+1.
 * @param frac Fractional position (0.0 to 1.0).
 * @return Interpolated sample.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLinear(T y0, T y1, T frac) noexcept {
    return y0 + frac * (y1 - y0);
}

/**
 * @brief Linear interpolation reading directly from a buffer.
 *
 * @param buffer Pointer to the start of the audio buffer.
 * @param length Size of the buffer in samples.
 * @param position Absolute read position (0 <= position < length).
 * @return Interpolated sample.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLinear(const T* buffer, int length, T position) noexcept {
    assert(position >= T(0) && position < static_cast<T>(length));
    
    int idx0 = static_cast<int>(position);
    T frac = position - static_cast<T>(idx0);
    
    int idx1 = idx0 + 1;
    if (idx1 >= length) idx1 = 0;

    return interpolateLinear(buffer[idx0], buffer[idx1], frac);
}

/**
 * @brief 4-point, 3rd-order Hermite interpolation (Optimized x-form).
 *
 * Mathematically identical to Catmull-Rom but heavily optimized for DSP.
 * Reduces multiplication operations compared to standard polynomial evaluation.
 *
 * @param y0 Sample at index n-1.
 * @param y1 Sample at index n.
 * @param y2 Sample at index n+1.
 * @param y3 Sample at index n+2.
 * @param frac Fractional position between y1 and y2 (0.0 to 1.0).
 * @return Interpolated sample.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateHermite(T y0, T y1, T y2, T y3, T frac) noexcept {
    T c = (y2 - y0) * T(0.5);
    T v = y1 - y2;
    T w = c + v;
    T a = w + v + (y3 - y1) * T(0.5);
    T b = w + a;
    return (((a * frac) - b) * frac + c) * frac + y1;
}

/**
 * @brief 4-point Hermite interpolation reading directly from a buffer.
 *
 * @param buffer Pointer to the start of the audio buffer.
 * @param length Size of the buffer in samples.
 * @param position Absolute read position (0 <= position < length).
 * @return Interpolated sample.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateHermite(const T* buffer, int length, T position) noexcept {
    assert(position >= T(0) && position < static_cast<T>(length));

    int idx1 = static_cast<int>(position);
    T frac = position - static_cast<T>(idx1);

    int idx0 = (idx1 > 0) ? idx1 - 1 : length - 1;
    int idx2 = idx1 + 1; if (idx2 >= length) idx2 -= length;
    int idx3 = idx2 + 1; if (idx3 >= length) idx3 -= length;

    return interpolateHermite(buffer[idx0], buffer[idx1], buffer[idx2], buffer[idx3], frac);
}

/**
 * @brief Alias for backward compatibility. In professional DSP, Catmull-Rom 
 * is implemented via the optimized Hermite formulation.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateCubic(const T* buffer, int length, T position) noexcept {
    return interpolateHermite(buffer, length, position);
}

/**
 * @brief 4-point Lagrange interpolation from discrete samples.
 *
 * Optimized to avoid floating-point divisions in the hot path.
 *
 * @param y0 Sample at index n-1.
 * @param y1 Sample at index n.
 * @param y2 Sample at index n+1.
 * @param y3 Sample at index n+2.
 * @param frac Fractional position between y1 and y2 (0.0 to 1.0).
 * @return Interpolated sample.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLagrange(T y0, T y1, T y2, T y3, T frac) noexcept {
    T d = frac;
    T dm1 = d - T(1);
    T dm2 = d - T(2);
    T dp1 = d + T(1);

    // Precomputed division multipliers: 1/6 ~ 0.16666667, 1/2 = 0.5
    static constexpr T inv6 = T(1.0 / 6.0);
    static constexpr T inv2 = T(0.5);

    T l0 = -(d * dm1 * dm2) * inv6;
    T l1 =  (dp1 * dm1 * dm2) * inv2;
    T l2 = -(dp1 * d * dm2) * inv2;
    T l3 =  (dp1 * d * dm1) * inv6;

    return l0 * y0 + l1 * y1 + l2 * y2 + l3 * y3;
}

/**
 * @brief 4-point Lagrange interpolation reading directly from a buffer.
 *
 * @param buffer Pointer to the start of the audio buffer.
 * @param length Size of the buffer in samples.
 * @param position Absolute read position (0 <= position < length).
 * @return Interpolated sample.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLagrange(const T* buffer, int length, T position) noexcept {
    assert(position >= T(0) && position < static_cast<T>(length));

    int idx1 = static_cast<int>(position);
    T frac = position - static_cast<T>(idx1);

    int idx0 = (idx1 > 0) ? idx1 - 1 : length - 1;
    int idx2 = idx1 + 1; if (idx2 >= length) idx2 -= length;
    int idx3 = idx2 + 1; if (idx3 >= length) idx3 -= length;

    return interpolateLagrange(buffer[idx0], buffer[idx1], buffer[idx2], buffer[idx3], frac);
}

/**
 * @brief Allpass interpolation (1-pole Thiran) for fractional delay.
 *
 * Achieves perfectly flat magnitude response at the cost of non-linear phase.
 * * @warning Allpass filters are IIR (recursive). Do NOT modulate the 'frac' 
 * parameter quickly (e.g., audio-rate LFOs), as it will cause state explosion 
 * and loud clicks. Best used for static or slowly changing delays.
 *
 * @param currentSample The current input sample (x[n]).
 * @param previousSample The previous input sample (x[n-1]).
 * @param frac Fractional delay (0.0 to 1.0). For stability, optimal range is > 0.5.
 * @param state Allpass filter state (y[n-1]). Maintained by the caller.
 * @return Interpolated sample value (y[n]).
 */
template <FloatType T>
[[nodiscard]] inline T interpolateAllpass(T currentSample, T previousSample,
                                          T frac, T& state) noexcept
{
    // Ensure coefficient calculation avoids division by zero if frac approaches -1.
    // Clamp frac for safety to avoid unstable poles.
    T safeFrac = (frac < T(0.001)) ? T(0.001) : frac;
    T coeff = (T(1) - safeFrac) / (T(1) + safeFrac);
    
    T output = coeff * (currentSample - state) + previousSample;
    
    // Denormalization protection (flush-to-zero) for IIR state
    if (std::abs(output) < T(1e-15)) output = T(0);
    
    state = output;
    return output;
}

} // namespace dspark