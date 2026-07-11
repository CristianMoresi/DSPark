// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file WindowFunctions.h
 * @brief Standard window functions for spectral analysis and FIR filter design.
 *
 * Window functions taper the edges of an audio frame to reduce spectral leakage
 * when using the FFT. Each window has different trade-offs between main lobe width
 * (frequency resolution) and side lobe level (leakage suppression).
 *
 * All windows are computed internally in double precision and cast to T on
 * store: window generation is setup-time work, so float builds get the full
 * float accuracy for free (this matters for high-attenuation designs such as
 * Kaiser beta 10+, where a float Bessel series would eat into the stopband).
 *
 * Quick guide for choosing a window:
 *
 * | Window         | Main lobe | Side lobes | Best for                                 |
 * |----------------|-----------|------------|------------------------------------------|
 * | Hann           | Medium    | -31 dB     | General-purpose spectral analysis        |
 * | Hamming        | Medium    | -42 dB     | Speech analysis, FIR design              |
 * | Blackman       | Wide      | -58 dB     | High dynamic range analysis              |
 * | BlackmanHarris | Wide      | -92 dB     | Precision measurement                    |
 * | Kaiser         | Variable  | Variable   | Configurable -- FIR design               |
 * | FlatTop        | Very wide | -93 dB     | Amplitude-accurate measurement           |
 * | Rectangular    | Narrowest | -13 dB     | Transient analysis (no windowing)        |
 * | Triangular     | Medium    | -26 dB     | Simple overlap-add applications          |
 *
 * Dependencies: C++20 standard library only.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace dspark {

/**
 * @struct WindowFunctions
 * @brief Static utility generating window functions for DSP analysis/synthesis.
 *
 * All functions write `size` values into the output array. Valid `size` must be > 0.
 * A size of 1 produces a unit impulse (1.0).
 *
 * The `periodic` flag selects the divisor of the phase term: `periodic = true`
 * (the default) generates DFT-periodic windows whose overlap-add properties
 * hold exactly (use for STFT/WOLA processing); `false` generates symmetric
 * windows that start and end on the same value (use for FIR filter design,
 * where symmetry gives exact linear phase).
 *
 * @tparam T Processing type (float or double).
 */
template <typename T>
struct WindowFunctions
{
    // Prevent instantiation (static-only utility class)
    WindowFunctions() = delete;

    /**
     * @brief Rectangular window (no windowing -- all ones).
     * @param output Destination buffer.
     * @param size Number of samples.
     */
    static void rectangular(T* output, int size) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        std::fill(output, output + size, T(1));
    }

    /**
     * @brief Triangular (Bartlett) window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void triangular(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const double halfN = static_cast<double>(N) / 2.0;

        for (int i = 0; i < size; ++i)
            output[i] = static_cast<T>(1.0 - std::abs((static_cast<double>(i) - halfN) / halfN));
    }

    /**
     * @brief Hann (raised cosine) window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void hann(T* output, int size, bool periodic = true) noexcept
    {
        static constexpr double kCoeffs[] = { 0.5, -0.5 };
        cosineSum(output, size, periodic, kCoeffs);
    }

    /**
     * @brief Hamming window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void hamming(T* output, int size, bool periodic = true) noexcept
    {
        static constexpr double kCoeffs[] = { 0.54, -0.46 };
        cosineSum(output, size, periodic, kCoeffs);
    }

    /**
     * @brief Blackman window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void blackman(T* output, int size, bool periodic = true) noexcept
    {
        static constexpr double kCoeffs[] = { 0.42, -0.5, 0.08 };
        cosineSum(output, size, periodic, kCoeffs);
    }

    /**
     * @brief Blackman-Harris window (4-term, -92 dB side lobes).
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void blackmanHarris(T* output, int size, bool periodic = true) noexcept
    {
        // Harris 1978, 4-term minimum side-lobe coefficients.
        static constexpr double kCoeffs[] = { 0.35875, -0.48829, 0.14128, -0.01168 };
        cosineSum(output, size, periodic, kCoeffs);
    }

    /**
     * @brief Flat-top window (amplitude-accurate: scallop loss < 0.01 dB).
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void flatTop(T* output, int size, bool periodic = true) noexcept
    {
        // ISO 18431-2 / MATLAB flattopwin coefficients.
        static constexpr double kCoeffs[] = { 0.21557895, -0.41663158, 0.277263158,
                                              -0.083578947, 0.006947368 };
        cosineSum(output, size, periodic, kCoeffs);
    }

    /**
     * @brief Kaiser window with configurable shape parameter beta.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param beta Shape parameter (typically 0.0 to 14.0; accurate to ~30).
     * @param periodic True for WOLA (default), false for symmetric FIR design.
     */
    static void kaiser(T* output, int size, T beta, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const double b = static_cast<double>(beta);
        const double invDenominator = 1.0 / besselI0(b);

        for (int i = 0; i < size; ++i)
        {
            const double x = 2.0 * static_cast<double>(i) / static_cast<double>(N) - 1.0;
            // Protected against rounding making (1 - x*x) strictly negative.
            const double arg = b * std::sqrt(std::max(0.0, 1.0 - x * x));
            output[i] = static_cast<T>(besselI0(arg) * invDenominator);
        }
    }

    // -- Utility methods -------------------------------------------------------

    /**
     * @brief Applies a window to a signal buffer in-place.
     *
     * Real-time safe. No alignment requirements: the element-wise loop
     * auto-vectorizes with unaligned loads (it is not a reduction).
     *
     * @param signal Signal buffer to window (modified in-place).
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     */
    static void apply(T* signal, const T* window, int size) noexcept
    {
        assert(signal != nullptr && window != nullptr);

        // No alignment assumption: std::assume_aligned<32> on a pointer that is
        // not actually 32-byte aligned is undefined behaviour, and these buffers
        // come from arbitrary callers (sub-views, vectors -> typically 16-byte).
        for (int i = 0; i < size; ++i)
            signal[i] *= window[i];
    }

    /**
     * @brief Computes the coherent gain of a window (mean of its samples).
     *
     * Useful for recovering the true amplitude of a peak in a spectral bin
     * after applying a windowing function. Accumulates in double so large
     * windows do not lose calibration accuracy in float.
     *
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     * @return Coherent gain (0.0 to 1.0).
     */
    [[nodiscard]] static T coherentGain(const T* window, int size) noexcept
    {
        assert(window != nullptr);
        if (size <= 0) return T(0);

        double sum = 0.0;
        for (int i = 0; i < size; ++i)
            sum += static_cast<double>(window[i]);

        return static_cast<T>(sum / static_cast<double>(size));
    }

    /**
     * @brief Computes the energy gain (RMS) of a window.
     *
     * Used for scaling the noise floor accurately when calculating Power
     * Spectral Density (PSD). Accumulates in double.
     *
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     * @return Energy gain (RMS multiplier).
     */
    [[nodiscard]] static T energyGain(const T* window, int size) noexcept
    {
        assert(window != nullptr);
        if (size <= 0) return T(0);

        double sumSq = 0.0;
        for (int i = 0; i < size; ++i)
        {
            const double w = static_cast<double>(window[i]);
            sumSq += w * w;
        }

        return static_cast<T>(std::sqrt(sumSq / static_cast<double>(size)));
    }

private:
    /**
     * @brief Shared generator for cosine-sum windows: w[i] = sum_t a[t]*cos(t*2*pi*i/N).
     *
     * The alternating signs of the textbook definitions are baked into the
     * coefficient arrays. Computed in double, stored as T.
     */
    template <std::size_t NumTerms>
    static void cosineSum(T* output, int size, bool periodic,
                          const double (&coeffs)[NumTerms]) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const double twoPiInvN = 2.0 * std::numbers::pi_v<double> / static_cast<double>(N);

        for (int i = 0; i < size; ++i)
        {
            const double x = twoPiInvN * static_cast<double>(i);
            double acc = coeffs[0];
            for (std::size_t t = 1; t < NumTerms; ++t)
                acc += coeffs[t] * std::cos(static_cast<double>(t) * x);
            output[i] = static_cast<T>(acc);
        }
    }

    /**
     * @brief Modified Bessel function of the first kind, order 0 (I0).
     *
     * Power series with relative-convergence cutoff at 1e-12; the 50-term cap
     * keeps it accurate to full double precision for arguments up to ~30
     * (covers every practical Kaiser beta).
     */
    [[nodiscard]] static double besselI0(double x) noexcept
    {
        double sum = 1.0;
        double term = 1.0;
        const double halfX = x / 2.0;

        for (int k = 1; k < 50; ++k)
        {
            const double f = halfX / static_cast<double>(k);
            term *= f * f;
            sum += term;
            if (term < sum * 1e-12) break; // converged
        }

        return sum;
    }
};

} // namespace dspark
