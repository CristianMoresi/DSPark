// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WindowFunctions.h
 * @brief Standard window functions for spectral analysis and FIR filter design.
 *
 * Window functions taper the edges of an audio frame to reduce spectral leakage
 * when using the FFT. Each window has different trade-offs between main lobe width
 * (frequency resolution) and side lobe level (leakage suppression).
 *
 * Quick guide for choosing a window:
 *
 * | Window         | Main lobe | Side lobes | Best for                                 |
 * |----------------|-----------|------------|------------------------------------------|
 * | Hann           | Medium    | -31 dB     | General-purpose spectral analysis        |
 * | Hamming        | Medium    | -42 dB     | Speech analysis, FIR design              |
 * | Blackman       | Wide      | -58 dB     | High dynamic range analysis              |
 * | BlackmanHarris | Wide      | -92 dB     | Precision measurement                    |
 * | Kaiser         | Variable  | Variable   | Configurable — FIR design                |
 * | FlatTop        | Very wide | -93 dB     | Amplitude-accurate measurement           |
 * | Rectangular    | Narrowest | -13 dB     | Transient analysis (no windowing)        |
 * | Triangular     | Medium    | -26 dB     | Simple overlap-add applications          |
 *
 * Dependencies: C++20 standard library only.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>

namespace dspark {

/**
 * @struct WindowFunctions
 * @brief Static utility generating window functions for DSP analysis/synthesis.
 *
 * All functions write `size` values into the output array. Valid `size` must be > 0.
 * A size of 1 produces a unit impulse (1.0).
 *
 * @tparam T Processing type (float or double).
 */
template <typename T>
struct WindowFunctions
{
    // Prevent instantiation (Static-only utility class)
    WindowFunctions() = delete;

    /**
     * @brief Rectangular window (no windowing — all ones).
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
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void triangular(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T halfN = static_cast<T>(N) / T(2);

        for (int i = 0; i < size; ++i)
            output[i] = T(1) - std::abs((static_cast<T>(i) - halfN) / halfN);
    }

    /**
     * @brief Hann (raised cosine) window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void hann(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T invN = T(1) / static_cast<T>(N);
        const T twoPi = T(2) * std::numbers::pi_v<T>;

        for (int i = 0; i < size; ++i)
            output[i] = T(0.5) - T(0.5) * std::cos(twoPi * static_cast<T>(i) * invN);
    }

    /**
     * @brief Hamming window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void hamming(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T invN = T(1) / static_cast<T>(N);
        const T twoPi = T(2) * std::numbers::pi_v<T>;

        for (int i = 0; i < size; ++i)
            output[i] = T(0.54) - T(0.46) * std::cos(twoPi * static_cast<T>(i) * invN);
    }

    /**
     * @brief Blackman window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void blackman(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T invN = T(1) / static_cast<T>(N);
        const T twoPi = T(2) * std::numbers::pi_v<T>;

        constexpr T a0 = T(0.42);
        constexpr T a1 = T(0.5);
        constexpr T a2 = T(0.08);

        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) * invN;
            output[i] = a0 - a1 * std::cos(twoPi * x) + a2 * std::cos(T(2) * twoPi * x);
        }
    }

    /**
     * @brief Blackman-Harris window (4-term).
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void blackmanHarris(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T invN = T(1) / static_cast<T>(N);
        const T twoPi = T(2) * std::numbers::pi_v<T>;

        constexpr T a0 = T(0.35875);
        constexpr T a1 = T(0.48829);
        constexpr T a2 = T(0.14128);
        constexpr T a3 = T(0.01168);

        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) * invN;
            output[i] = a0 - a1 * std::cos(twoPi * x)
                           + a2 * std::cos(T(2) * twoPi * x)
                           - a3 * std::cos(T(3) * twoPi * x);
        }
    }

    /**
     * @brief Flat-top window.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void flatTop(T* output, int size, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T invN = T(1) / static_cast<T>(N);
        const T twoPi = T(2) * std::numbers::pi_v<T>;

        constexpr T a0 = T(0.21557895);
        constexpr T a1 = T(0.41663158);
        constexpr T a2 = T(0.277263158);
        constexpr T a3 = T(0.083578947);
        constexpr T a4 = T(0.006947368);

        for (int i = 0; i < size; ++i)
        {
            T x = static_cast<T>(i) * invN;
            output[i] = a0 - a1 * std::cos(twoPi * x)
                           + a2 * std::cos(T(2) * twoPi * x)
                           - a3 * std::cos(T(3) * twoPi * x)
                           + a4 * std::cos(T(4) * twoPi * x);
        }
    }

    /**
     * @brief Kaiser window with configurable shape parameter beta.
     * @param output Destination buffer.
     * @param size Number of samples.
     * @param beta Shape parameter (typically 0.0 to 14.0).
     * @param periodic True for WOLA (default), false for symmetric DFT analysis.
     */
    static void kaiser(T* output, int size, T beta, bool periodic = true) noexcept
    {
        assert(output != nullptr);
        if (size <= 0) return;
        if (size == 1) { output[0] = T(1); return; }

        const int N = periodic ? size : size - 1;
        const T denominator = bessel_I0(beta);

        for (int i = 0; i < size; ++i)
        {
            T x = T(2) * static_cast<T>(i) / static_cast<T>(N) - T(1);
            // Protected against numerical inaccuracy making (1 - x*x) strictly negative
            T arg = beta * std::sqrt(std::max(T(0), T(1) - x * x));
            output[i] = bessel_I0(arg) / denominator;
        }
    }

    // -- Utility methods -------------------------------------------------------

    /**
     * @brief Applies a window to a signal buffer in-place.
     *
     * @note C++20 `std::assume_aligned<32>` is used to guarantee SIMD
     * vectorization (AVX/AVX2). 
     * @warning The pointers `signal` and `window` MUST be strictly 32-byte 
     * aligned. Passing unaligned pointers will result in Undefined Behavior (UB).
     *
     * @param signal Signal buffer to window (modified in-place).
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     */
    static void apply(T* signal, const T* window, int size) noexcept
    {
        assert(signal != nullptr && window != nullptr);
        
        // CRITICAL: Validate 32-byte alignment in Debug to prevent silent UB
        assert(reinterpret_cast<std::uintptr_t>(signal) % 32 == 0 && "Signal buffer is not 32-byte aligned!");
        assert(reinterpret_cast<std::uintptr_t>(window) % 32 == 0 && "Window buffer is not 32-byte aligned!");

        // C++20 hint to force SIMD vectorization
        auto* __restrict alignedSignal = std::assume_aligned<32>(signal);
        const auto* __restrict alignedWindow = std::assume_aligned<32>(window);

        for (int i = 0; i < size; ++i)
            alignedSignal[i] *= alignedWindow[i];
    }

    /**
     * @brief Computes the coherent gain of a window.
     * 
     * Useful for recovering the true amplitude of a peak in a spectral bin 
     * after applying a windowing function.
     *
     * @warning The pointer `window` MUST be strictly 32-byte aligned for SIMD optimization.
     * 
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     * @return Coherent gain (0.0 to 1.0).
     */
    [[nodiscard]] static T coherentGain(const T* window, int size) noexcept
    {
        assert(window != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(window) % 32 == 0 && "Window buffer is not 32-byte aligned!");
        if (size <= 0) return T(0);

        const auto* __restrict alignedWindow = std::assume_aligned<32>(window);
        T sum = T(0);
        
        for (int i = 0; i < size; ++i)
            sum += alignedWindow[i];
            
        return sum / static_cast<T>(size);
    }

    /**
     * @brief Computes the energy gain (RMS) of a window.
     * 
     * Used for scaling the noise floor accurately when calculating Power Spectral Density (PSD).
     * 
     * @warning The pointer `window` MUST be strictly 32-byte aligned for SIMD optimization.
     *
     * @param window Pre-computed window values.
     * @param size   Number of samples.
     * @return Energy gain (RMS multiplier).
     */
    [[nodiscard]] static T energyGain(const T* window, int size) noexcept
    {
        assert(window != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(window) % 32 == 0 && "Window buffer is not 32-byte aligned!");
        if (size <= 0) return T(0);

        const auto* __restrict alignedWindow = std::assume_aligned<32>(window);
        T sumSq = T(0);
        
        for (int i = 0; i < size; ++i)
            sumSq += alignedWindow[i] * alignedWindow[i];
            
        return std::sqrt(sumSq / static_cast<T>(size));
    }

private:
    /**
     * @brief Modified Bessel function of the first kind, order 0 (I0).
     */
    [[nodiscard]] static T bessel_I0(T x) noexcept
    {
        T sum = T(1);
        T term = T(1);
        const T halfX = x / T(2);

        for (int k = 1; k < 25; ++k)
        {
            term *= (halfX / static_cast<T>(k));
            term *= (halfX / static_cast<T>(k));
            sum += term;
            if (term < sum * T(1e-12)) break; // Converged
        }

        return sum;
    }
};

} // namespace dspark