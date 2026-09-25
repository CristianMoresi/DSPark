// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Interpolation.h
 * @brief Fractional-position interpolation for delay lines, tables and buffers.
 *
 * Free-function interpolators used across the framework (RingBuffer's
 * interpolated reads, modulated delay lines). All of them are small scalar
 * helpers that inline into the caller's loop; none allocates or locks.
 * SincInterpolator is the high-fidelity reader for resampling a stream
 * (pitch shifting, varispeed); it owns a kernel table built on construction.
 *
 * | Method    | Points | Quality       | CPU Cost  | Use case                     |
 * |-----------|--------|---------------|-----------|------------------------------|
 * | Linear    | 2      | Low           | Ultra-Low | LFOs, crossfades             |
 * | Hermite   | 4      | Good+         | Low       | Modulated delays (default)   |
 * | Lagrange  | 4      | High          | Low       | Precision fractional reads   |
 * | Allpass   | 2      | Frequency-dep | Low       | Static fractional delays     |
 * | Sinc      | 32     | Transparent   | Medium    | Resampling a stream          |
 *
 * Polynomial accuracy: Linear reconstructs degree-1 signals exactly, Hermite
 * (Catmull-Rom) degree 2, Lagrange degree 3. Hermite is the recommended
 * default for modulated delay lines (chorus, vibrato, reverb modulation): it
 * is stateless and C1-continuous across segments, so fast `frac` changes
 * never destabilise anything, unlike the recursive Allpass interpolator.
 *
 * Two overload families:
 * - Raw-sample overloads (y0..y3, frac): no bounds logic at all, the fastest
 *   path. The container handles buffer wrapping (RingBuffer uses these).
 * - (buffer, length, position) overloads: treat the buffer as CIRCULAR;
 *   neighbours needed beyond either end wrap around to the other end. Meant
 *   for ring buffers and periodic tables. On a plain linear buffer, positions
 *   within one sample of the edges (4-point methods) blend in samples from
 *   the opposite end.
 *
 * Threading: all functions are pure and re-entrant; the Allpass state lives
 * in the caller. Real-time safe: no allocation, no locks, no modulo; the only
 * floating-point division is the Allpass coefficient (one per call).
 * SincInterpolator allocates its table in the constructor (setup thread);
 * its reads are const, allocation-free and safe from any thread.
 *
 * Dependencies: DspMath.h (FloatType concept), SimdOps.h (dot products).
 */

#include "DspMath.h"
#include "SimdOps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

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
 * @brief Linear interpolation reading directly from a circular buffer.
 *
 * The buffer is treated as circular: reading at position length - 0.5
 * interpolates between the last and the first sample.
 *
 * @param buffer Pointer to the start of the audio buffer.
 * @param length Size of the buffer in samples (must be > 0).
 * @param position Absolute read position.
 * @return Interpolated sample.
 *
 * @pre 0 <= position < length. Violations assert in debug builds; release
 * builds clamp to the nearest valid sample (NaN reads position 0).
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLinear(const T* buffer, int length, T position) noexcept {
    assert(length > 0 && position >= T(0) && position < static_cast<T>(length));
    if (length <= 0) return T(0);

    // An out-of-range position (including NaN) would make the int cast below
    // undefined and the buffer reads out of bounds: clamp before casting.
    int idx0;
    T frac;
    if (!(position >= T(0))) { idx0 = 0; frac = T(0); }
    else if (position >= static_cast<T>(length)) { idx0 = length - 1; frac = T(0); }
    else {
        idx0 = static_cast<int>(position);
        frac = position - static_cast<T>(idx0);
    }

    int idx1 = idx0 + 1;
    if (idx1 >= length) idx1 = 0;

    return interpolateLinear(buffer[idx0], buffer[idx1], frac);
}

/**
 * @brief 4-point, 3rd-order Hermite interpolation (optimized x-form).
 *
 * Mathematically identical to Catmull-Rom but evaluated with fewer
 * multiplications than the standard polynomial form.
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
 * @brief 4-point Hermite interpolation reading directly from a circular buffer.
 *
 * Interpolates between buffer[floor(position)] and the next sample; the two
 * outer neighbours wrap around the buffer ends (circular contract, see @file
 * notes).
 *
 * @param buffer Pointer to the start of the audio buffer.
 * @param length Size of the buffer in samples (must be > 0).
 * @param position Absolute read position.
 * @return Interpolated sample.
 *
 * @pre 0 <= position < length. Violations assert in debug builds; release
 * builds clamp to the nearest valid sample (NaN reads position 0).
 */
template <FloatType T>
[[nodiscard]] inline T interpolateHermite(const T* buffer, int length, T position) noexcept {
    assert(length > 0 && position >= T(0) && position < static_cast<T>(length));
    if (length <= 0) return T(0);

    // Same release-safe clamp as the linear overload (undefined int cast /
    // out-of-bounds reads otherwise).
    int idx1;
    T frac;
    if (!(position >= T(0))) { idx1 = 0; frac = T(0); }
    else if (position >= static_cast<T>(length)) { idx1 = length - 1; frac = T(0); }
    else {
        idx1 = static_cast<int>(position);
        frac = position - static_cast<T>(idx1);
    }

    int idx0 = (idx1 > 0) ? idx1 - 1 : length - 1;
    int idx2 = idx1 + 1; if (idx2 >= length) idx2 -= length;
    int idx3 = idx2 + 1; if (idx3 >= length) idx3 -= length;

    return interpolateHermite(buffer[idx0], buffer[idx1], buffer[idx2], buffer[idx3], frac);
}

/**
 * @brief Alias of interpolateHermite (Catmull-Rom evaluated in Hermite form).
 * Kept for backward compatibility.
 */
template <FloatType T>
[[nodiscard]] inline T interpolateCubic(const T* buffer, int length, T position) noexcept {
    return interpolateHermite(buffer, length, position);
}

/**
 * @brief 4-point Lagrange interpolation from discrete samples.
 *
 * Reconstructs polynomials up to degree 3 exactly (the divisions of the
 * Lagrange basis are precomputed constants).
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
 * @brief 4-point Lagrange interpolation reading directly from a circular buffer.
 *
 * Interpolates between buffer[floor(position)] and the next sample; the two
 * outer neighbours wrap around the buffer ends (circular contract, see @file
 * notes).
 *
 * @param buffer Pointer to the start of the audio buffer.
 * @param length Size of the buffer in samples (must be > 0).
 * @param position Absolute read position.
 * @return Interpolated sample.
 *
 * @pre 0 <= position < length. Violations assert in debug builds; release
 * builds clamp to the nearest valid sample (NaN reads position 0).
 */
template <FloatType T>
[[nodiscard]] inline T interpolateLagrange(const T* buffer, int length, T position) noexcept {
    assert(length > 0 && position >= T(0) && position < static_cast<T>(length));
    if (length <= 0) return T(0);

    // Same release-safe clamp as the linear overload (undefined int cast /
    // out-of-bounds reads otherwise).
    int idx1;
    T frac;
    if (!(position >= T(0))) { idx1 = 0; frac = T(0); }
    else if (position >= static_cast<T>(length)) { idx1 = length - 1; frac = T(0); }
    else {
        idx1 = static_cast<int>(position);
        frac = position - static_cast<T>(idx1);
    }

    int idx0 = (idx1 > 0) ? idx1 - 1 : length - 1;
    int idx2 = idx1 + 1; if (idx2 >= length) idx2 -= length;
    int idx3 = idx2 + 1; if (idx3 >= length) idx3 -= length;

    return interpolateLagrange(buffer[idx0], buffer[idx1], buffer[idx2], buffer[idx3], frac);
}

/**
 * @brief Allpass interpolation (first-order Thiran) for fractional delay.
 *
 * Perfectly flat magnitude response at every frequency, unlike polynomial
 * interpolators (which low-pass); the cost is a frequency-dependent phase
 * response and a recursive state.
 *
 * @warning Allpass interpolation is IIR (recursive). Do NOT modulate `frac`
 * quickly (e.g. audio-rate LFOs): the state carries history filtered with
 * the previous coefficient and produces loud clicks. Best for static or
 * slowly changing delays; use interpolateHermite for modulated delay lines.
 *
 * The filter is stable for any frac > 0, but as frac approaches 0 the pole
 * approaches z = -1 and the interpolator rings for thousands of samples near
 * Nyquist. Canonical usage keeps the fractional delay in [0.5, 1.5): when the
 * fractional part f of the total delay is below 0.5, read one integer sample
 * earlier and pass frac = f + 1 instead.
 *
 * @param currentSample The current input sample (x[n]).
 * @param previousSample The previous input sample (x[n-1]).
 * @param frac Fractional delay in samples, relative to x[n]. Stable for any
 * value > 0; see the [0.5, 1.5) mapping above for clean low-frac behaviour.
 * @param state Allpass filter state (y[n-1]). Maintained by the caller;
 * initialise to 0 and keep one state per delay tap per channel.
 * @return Interpolated sample value (y[n]).
 */
template <FloatType T>
[[nodiscard]] inline T interpolateAllpass(T currentSample, T previousSample,
                                          T frac, T& state) noexcept
{
    // frac == 0 would place the pole exactly at z = -1 (a marginally stable
    // resonator at Nyquist) and frac == -1 would divide by zero: clamp away
    // from both. The clamp changes the delivered delay only for out-of-range
    // requests.
    T safeFrac = (frac < T(0.001)) ? T(0.001) : frac;
    T coeff = (T(1) - safeFrac) / (T(1) + safeFrac);

    T output = coeff * (currentSample - state) + previousSample;

    // Cut the recursive tail below ~-300 dBFS so a decaying state can never
    // reach the denormal range (matters on targets without FTZ/DAZ).
    if (std::abs(output) < T(1e-15)) output = T(0);

    state = output;
    return output;
}

/**
 * @class SincInterpolator
 * @brief 32-tap Kaiser-windowed sinc reader for transparent fractional reads.
 *
 * The reader for resampling a stream at a fixed or varying rate. Its
 * worst-case error against the ideal fractional delay, over every fractional
 * position, stays below -96 dB up to 15 kHz at 44.1 and 48 kHz (and 18 kHz
 * at 48 kHz). A 4-point cubic (Hermite / Catmull-Rom) reader reaches -24 dB
 * at 10 kHz and -12 dB at 15 kHz, and because its error changes with the
 * fractional position, a moving read point turns it into modulation noise.
 *
 * The kernel is a full-band sinc (cutoff at Nyquist) under a Kaiser window
 * (beta 10), tabulated at 256 fractional phases, each normalized to unit DC
 * gain, and linearly interpolated between phases. Integer positions are an
 * exact identity. Near Nyquist the 32 taps cannot hold the band: at 20 kHz
 * the error is -50 dB at 48 kHz and -17 dB at 44.1 kHz.
 *
 * Reading position intPos + frac needs the samples intPos - 15 through
 * intPos + 16.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SincInterpolator
{
public:
    static constexpr int kTaps   = 32;          ///< Kernel length.
    static constexpr int kBefore = kTaps / 2 - 1; ///< Samples needed before intPos.
    static constexpr int kAfter  = kTaps / 2;   ///< Samples needed after intPos.
    static constexpr int kPhases = 256;         ///< Tabulated fractional phases.

    /** @brief Builds the kernel table (allocates: construct on a setup thread). */
    SincInterpolator()
        : table_(static_cast<size_t>((kPhases + 1) * kTaps))
    {
        constexpr double kBeta = 10.0;
        constexpr double kPi = 3.14159265358979323846;
        const double invI0Beta = 1.0 / besselI0(kBeta);
        const double halfSpan = static_cast<double>(kTaps) / 2.0;

        for (int p = 0; p <= kPhases; ++p)
        {
            T* row = table_.data() + static_cast<size_t>(p) * kTaps;
            const double phase = static_cast<double>(p) / kPhases;
            if (p == 0 || p == kPhases)
            {
                // Integer positions: an exact identity (sin(pi * k) rounds
                // to ~1e-16, not 0, so the sinc is not evaluated here).
                std::fill(row, row + kTaps, T(0));
                row[p == 0 ? kBefore : kBefore + 1] = T(1);
                continue;
            }
            double h[kTaps];
            double sum = 0.0;
            for (int t = 0; t < kTaps; ++t)
            {
                const double x = static_cast<double>(t - kBefore) - phase;
                const double r = x / halfSpan;
                const double window = (r * r < 1.0)
                    ? besselI0(kBeta * std::sqrt(1.0 - r * r)) * invI0Beta
                    : 0.0;
                h[t] = std::sin(kPi * x) / (kPi * x) * window;
                sum += h[t];
            }
            for (int t = 0; t < kTaps; ++t)
                row[t] = static_cast<T>(h[t] / sum);   // unit DC gain
        }
    }

    /**
     * @brief Interpolates kTaps consecutive samples at x[kBefore] + frac.
     * @param x    Samples intPos - kBefore through intPos + kAfter.
     * @param frac Fractional position in [0, 1).
     * @return Interpolated value.
     */
    [[nodiscard]] T read(const T* x, double frac) const noexcept
    {
        const double p = frac * static_cast<double>(kPhases);
        const int ip = std::clamp(static_cast<int>(p), 0, kPhases - 1);
        const T t = static_cast<T>(p - static_cast<double>(ip));
        const T* h0 = table_.data() + static_cast<size_t>(ip) * kTaps;
        const T* h1 = h0 + kTaps;   // the table holds kPhases + 1 rows
        const T a = simd::dotProductT(h0, x, kTaps);
        const T b = simd::dotProductT(h1, x, kTaps);
        return a + t * (b - a);
    }

    /**
     * @brief Reads a power-of-two ring buffer at intPos + frac.
     * @param ring   Ring storage.
     * @param mask   Ring size minus one (the size must be a power of two).
     * @param intPos Integer read position (masked, so any value works).
     * @param frac   Fractional position in [0, 1).
     * @return Interpolated value.
     */
    [[nodiscard]] T readRing(const T* ring, int64_t mask, int64_t intPos, double frac) const noexcept
    {
        const int64_t first = intPos - kBefore;
        const auto start = static_cast<size_t>(first & mask);
        if (start + static_cast<size_t>(kTaps) <= static_cast<size_t>(mask) + 1)
            return read(ring + start, frac);   // contiguous: no gather
        T window[kTaps];
        for (int t = 0; t < kTaps; ++t)
            window[t] = ring[static_cast<size_t>((first + t) & mask)];
        return read(window, frac);
    }

private:
    /** @brief Modified Bessel I0 (power series, converges for Kaiser betas). */
    [[nodiscard]] static double besselI0(double x) noexcept
    {
        double sum = 1.0, term = 1.0;
        const double halfX = x / 2.0;
        for (int k = 1; k < 60; ++k)
        {
            const double f = halfX / static_cast<double>(k);
            term *= f * f;
            sum += term;
            if (term < sum * 1e-17) break;
        }
        return sum;
    }

    std::vector<T> table_;   ///< (kPhases + 1) rows of kTaps coefficients.
};

} // namespace dspark
