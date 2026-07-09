// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Biquad.h
 * @brief Biquad filter with Transposed Direct Form II, Audio EQ Cookbook coefficients, and thread-safe updates.
 *
 * Provides two classes:
 *
 * - **BiquadCoeffs<T>**: Coefficient set with static factory methods for all
 * standard filter types. Formulas from Robert Bristow-Johnson's "Audio EQ Cookbook".
 * Memory aligned to 32 bytes for SIMD operations.
 *
 * - **Biquad<T, MaxChannels>**: Per-channel filter state using Transposed Direct
 * Form II (TDF-II). Features a lock-free shadow buffering system to allow safe
 * coefficient updates from the GUI thread without tearing or blocking the audio thread.
 *
 * Dependencies: C++20 standard library (<algorithm>, <array>, <atomic>, <cassert>, <cmath>, <numbers>, <span>).
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <atomic>
#include <cassert>

#include "AudioBuffer.h"
#include "DenormalGuard.h"

namespace dspark {

// ============================================================================
// BiquadCoeffs -- Coefficient storage + factory methods
// ============================================================================

/**
 * @struct BiquadCoeffs
 * @brief Stores normalised biquad coefficients (b0, b1, b2, a1, a2).
 *
 * Coefficients are pre-normalised by a0 in every factory method, so the
 * filter processing loop never needs to divide by a0. Memory is aligned
 * to 32 bytes to facilitate explicit SIMD vectorisation and CPU cache optimal loads.
 *
 * All factory methods clamp frequency, Q and slope to safe ranges, so any
 * finite parameter combination yields a stable filter. NaN parameters are
 * not sanitised (they propagate into the coefficients), and the sample rate
 * is trusted as-is: the framework contract is a valid AudioSpec upstream.
 *
 * @tparam T Coefficient type (float or double).
 */
template <typename T>
struct alignas(32) BiquadCoeffs
{
    T b0 = T(1), b1 = T(0), b2 = T(0);
    T a1 = T(0), a2 = T(0);

    // -- Factory methods (Audio EQ Cookbook) ----------------------------------

    /**
     * @brief Low-pass filter.
     * @param sampleRate Sample rate in Hz.
     * @param freq       Centre frequency in Hz.
     * @param Q          Quality factor (default: 1/sqrt(2) = Butterworth).
     */
    [[nodiscard]] static BiquadCoeffs makeLowPass(double sampleRate, double freq, double Q = 0.7071067811865476) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        Q = std::max(Q, 0.001);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double alpha = sinw0 / (2.0 * Q);

        const double a0 = 1.0 + alpha;
        return normalise(a0, {
            T((1.0 - cosw0) / 2.0),
            T( 1.0 - cosw0),
            T((1.0 - cosw0) / 2.0),
            T(-2.0 * cosw0),
            T( 1.0 - alpha)
        });
    }

    /**
     * @brief High-pass filter.
     * @param sampleRate Sample rate in Hz.
     * @param freq       Centre frequency in Hz.
     * @param Q          Quality factor (default: Butterworth).
     */
    [[nodiscard]] static BiquadCoeffs makeHighPass(double sampleRate, double freq, double Q = 0.7071067811865476) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        Q = std::max(Q, 0.001);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double alpha = sinw0 / (2.0 * Q);

        const double a0 = 1.0 + alpha;
        return normalise(a0, {
            T( (1.0 + cosw0) / 2.0),
            T(-(1.0 + cosw0)),
            T( (1.0 + cosw0) / 2.0),
            T(-2.0 * cosw0),
            T( 1.0 - alpha)
        });
    }

    /**
     * @brief Band-pass filter (constant 0 dB peak gain).
     *
     * Implements the Audio EQ Cookbook BPF variant with `b0 = alpha`, whose
     * peak gain at the centre frequency is exactly 0 dB regardless of Q,
     * the most useful variant for mixing and crossover work.
     *
     * @param sampleRate Sample rate in Hz.
     * @param freq       Centre frequency in Hz.
     * @param Q          Quality factor (default: Butterworth).
     */
    [[nodiscard]] static BiquadCoeffs makeBandPass(double sampleRate, double freq, double Q = 0.7071067811865476) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        Q = std::max(Q, 0.001);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double alpha = sinw0 / (2.0 * Q);

        const double a0 = 1.0 + alpha;
        return normalise(a0, {
            T(alpha),
            T(0.0),
            T(-alpha),
            T(-2.0 * cosw0),
            T( 1.0 - alpha)
        });
    }

    /**
     * @brief Peak (parametric EQ) filter.
     *
     * Uses the correct Audio EQ Cookbook formula where A = 10^(dBgain/40),
     * giving the expected gain in dB at the centre frequency.
     *
     * @param sampleRate Sample rate in Hz.
     * @param freq       Centre frequency in Hz.
     * @param Q          Quality factor.
     * @param gainDb     Gain in decibels (positive = boost, negative = cut).
     */
    [[nodiscard]] static BiquadCoeffs makePeak(double sampleRate, double freq, double Q, double gainDb) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        Q = std::max(Q, 0.001);
        const double A     = std::pow(10.0, gainDb / 40.0); // dB/40 = sqrt of linear gain
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double alpha = sinw0 / (2.0 * Q);

        const double a0 = 1.0 + alpha / A;
        return normalise(a0, {
            T(1.0 + alpha * A),
            T(-2.0 * cosw0),
            T(1.0 - alpha * A),
            T(-2.0 * cosw0),
            T(1.0 - alpha / A)
        });
    }

    /**
     * @brief Peaking filter with prescribed Nyquist gain (Orfanidis design).
     *
     * The bilinear (cookbook) peaking filter cramps near Nyquist: high-
     * frequency bells get narrower and their response is pinned at fs/2,
     * deviating audibly from the analog prototype above ~fs/6. This design
     * (Orfanidis, JAES 45(6), 1997) prescribes the digital gain at Nyquist
     * to equal the ANALOG prototype's gain there, matching the analog bell
     * shape across the band: the standard "de-cramped" EQ used by
     * state-of-the-art digital equalizers.
     *
     * Falls back to identity for |gain| < 0.01 dB. At low frequencies it
     * converges to the cookbook response (as it should).
     *
     * @param sampleRate Sample rate in Hz.
     * @param freq       Center frequency in Hz.
     * @param Q          Quality factor (bandwidth = freq / Q).
     * @param gainDb     Peak gain in decibels.
     */
    [[nodiscard]] static BiquadCoeffs makePeakMatched(double sampleRate, double freq,
                                                      double Q, double gainDb) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.495));
        Q = std::max(Q, 0.001);
        if (std::abs(gainDb) < 0.01)
            return { T(1), T(0), T(0), T(0), T(0) };

        const double G0 = 1.0;                                  // reference gain
        const double G = std::pow(10.0, gainDb / 20.0);         // peak gain
        const double GB = std::pow(10.0, gainDb / 40.0);        // bandwidth gain (half-dB)

        const double w0 = 2.0 * std::numbers::pi * freq / sampleRate;
        const double Dw = w0 / Q;

        // Analog prototype gain at the physical Nyquist frequency:
        // |Ha(jW)|^2 = (G0^2 (W^2-W0^2)^2 + G^2 Dw^2 W^2) /
        //              (    (W^2-W0^2)^2 +      Dw^2 W^2),  W in rad/s.
        const double W0 = 2.0 * std::numbers::pi * freq;
        const double DW = W0 / Q;
        const double Wn = std::numbers::pi * sampleRate;        // 2*pi*fs/2
        const double d2 = (Wn * Wn - W0 * W0) * (Wn * Wn - W0 * W0);
        const double G1 = std::sqrt((G0 * G0 * d2 + G * G * DW * DW * Wn * Wn)
                                    / (d2 + DW * DW * Wn * Wn));

        // Orfanidis closed-form coefficients.
        const double G2 = G * G, G02 = G0 * G0, GB2 = GB * GB, G12 = G1 * G1;
        const double F   = std::abs(G2 - GB2);
        const double G00 = std::abs(G2 - G02);
        const double F00 = std::abs(GB2 - G02);
        const double F01 = std::abs(GB2 - G0 * G1);
        const double F11 = std::abs(GB2 - G12);
        const double G01 = std::abs(G2 - G0 * G1);
        const double G11 = std::abs(G2 - G12);

        const double t0 = std::tan(w0 / 2.0);
        const double W2 = std::sqrt(G11 / G00) * t0 * t0;
        const double DWd = (1.0 + std::sqrt(F00 / F11) * W2) * std::tan(Dw / 2.0);

        const double C = F11 * DWd * DWd - 2.0 * W2 * (F01 - std::sqrt(F00 * F11));
        const double D = 2.0 * W2 * (G01 - std::sqrt(G00 * G11));
        const double A = std::sqrt(std::max((C + D) / std::max(F, 1e-30), 0.0));
        const double B = std::sqrt(std::max((G2 * C + GB2 * D) / std::max(F, 1e-30), 0.0));

        const double a0 = 1.0 + W2 + A;
        return normalise(a0, {
            T(G1 + G0 * W2 + B),
            T(-2.0 * (G1 - G0 * W2)),
            T(G1 + G0 * W2 - B),
            T(-2.0 * (1.0 - W2)),
            T(1.0 + W2 - A)
        });
    }

    /**
     * @brief Low-shelf filter.
     * @param sampleRate Sample rate in Hz.
     * @param freq       Transition frequency in Hz.
     * @param gainDb     Shelf gain in decibels.
     * @param slope      Shelf slope (default: 1.0 for standard 6 dB/oct transition).
     */
    [[nodiscard]] static BiquadCoeffs makeLowShelf(double sampleRate, double freq, double gainDb, double slope = 1.0) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        // Shelf slope S must stay in (0, 1]: for S > 1 the radicand below goes
        // negative and std::sqrt yields NaN coefficients that poison the audio.
        slope = std::clamp(slope, 0.0001, 1.0);
        const double A     = std::pow(10.0, gainDb / 40.0);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double radicand = (A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0;
        const double alpha = sinw0 / 2.0 * std::sqrt(std::max(0.0, radicand));
        const double twoSqrtAAlpha = 2.0 * std::sqrt(A) * alpha;

        const double a0 = (A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha;
        return normalise(a0, {
            T(A * ((A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha)),
            T(2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0)),
            T(A * ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha)),
            T(-2.0 * ((A - 1.0) + (A + 1.0) * cosw0)),
            T((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha)
        });
    }

    /**
     * @brief High-shelf filter.
     * @param sampleRate Sample rate in Hz.
     * @param freq       Transition frequency in Hz.
     * @param gainDb     Shelf gain in decibels.
     * @param slope      Shelf slope (default: 1.0).
     */
    [[nodiscard]] static BiquadCoeffs makeHighShelf(double sampleRate, double freq, double gainDb, double slope = 1.0) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        // Shelf slope S must stay in (0, 1]: for S > 1 the radicand below goes
        // negative and std::sqrt yields NaN coefficients that poison the audio.
        slope = std::clamp(slope, 0.0001, 1.0);
        const double A     = std::pow(10.0, gainDb / 40.0);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double radicand = (A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0;
        const double alpha = sinw0 / 2.0 * std::sqrt(std::max(0.0, radicand));
        const double twoSqrtAAlpha = 2.0 * std::sqrt(A) * alpha;

        const double a0 = (A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha;
        return normalise(a0, {
            T(A * ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha)),
            T(-2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0)),
            T(A * ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha)),
            T(2.0 * ((A - 1.0) - (A + 1.0) * cosw0)),
            T((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha)
        });
    }

    /**
     * @brief Notch (band-reject) filter.
     * @param sampleRate Sample rate in Hz.
     * @param freq       Centre frequency in Hz.
     * @param Q          Quality factor (default: Butterworth).
     */
    [[nodiscard]] static BiquadCoeffs makeNotch(double sampleRate, double freq, double Q = 0.7071067811865476) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        Q = std::max(Q, 0.001);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double alpha = sinw0 / (2.0 * Q);

        const double a0 = 1.0 + alpha;
        return normalise(a0, {
            T(1.0),
            T(-2.0 * cosw0),
            T(1.0),
            T(-2.0 * cosw0),
            T(1.0 - alpha)
        });
    }

    /**
     * @brief All-pass filter.
     * @param sampleRate Sample rate in Hz.
     * @param freq       Centre frequency in Hz.
     * @param Q          Quality factor (default: Butterworth).
     */
    [[nodiscard]] static BiquadCoeffs makeAllPass(double sampleRate, double freq, double Q = 0.7071067811865476) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        Q = std::max(Q, 0.001);
        const double w0    = 2.0 * std::numbers::pi * freq / sampleRate;
        const double cosw0 = std::cos(w0);
        const double sinw0 = std::sin(w0);
        const double alpha = sinw0 / (2.0 * Q);

        const double a0 = 1.0 + alpha;
        return normalise(a0, {
            T(1.0 - alpha),
            T(-2.0 * cosw0),
            T(1.0 + alpha),
            T(-2.0 * cosw0),
            T(1.0 - alpha)
        });
    }

    /**
     * @brief Creates a DC-blocking high-pass filter.
     *
     * A very low frequency high-pass (default 5 Hz) used to remove DC offset
     * introduced by nonlinear processing (saturation, waveshaping, etc.).
     *
     * @param sampleRate Sample rate in Hz.
     * @param freq       Cut-off frequency in Hz (default: 5 Hz).
     */
    [[nodiscard]] static BiquadCoeffs makeDcBlocker(double sampleRate, double freq = 5.0) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        return makeHighPass(sampleRate, freq, 0.7071067811865476);
    }

    // -- First-order filter factory methods ------------------------------------

    /**
     * @brief First-order (6 dB/oct) low-pass filter.
     *
     * Uses bilinear-transformed RC filter. Coefficients are already normalised
     * (a0 = 1 implicit), so no normalise() call is needed.
     *
     * @param sampleRate Sample rate in Hz.
     * @param frequency  Cut-off frequency in Hz.
     */
    [[nodiscard]] static BiquadCoeffs makeFirstOrderLowPass(double sampleRate, double frequency) noexcept
    {
        frequency = std::clamp(frequency, 1.0, std::max(1.0, sampleRate * 0.499));
        double w = std::tan(std::numbers::pi * frequency / sampleRate);
        double n = 1.0 / (1.0 + w);
        BiquadCoeffs c;
        c.b0 = static_cast<T>(w * n);
        c.b1 = static_cast<T>(w * n);
        c.b2 = T(0);
        c.a1 = static_cast<T>((w - 1.0) * n);
        c.a2 = T(0);
        return c;
    }

    /**
     * @brief First-order (6 dB/oct) high-pass filter.
     *
     * Uses bilinear-transformed CR filter. Coefficients are already normalised
     * (a0 = 1 implicit), so no normalise() call is needed.
     *
     * @param sampleRate Sample rate in Hz.
     * @param frequency  Cut-off frequency in Hz.
     */
    [[nodiscard]] static BiquadCoeffs makeFirstOrderHighPass(double sampleRate, double frequency) noexcept
    {
        frequency = std::clamp(frequency, 1.0, std::max(1.0, sampleRate * 0.499));
        double w = std::tan(std::numbers::pi * frequency / sampleRate);
        double n = 1.0 / (1.0 + w);
        BiquadCoeffs c;
        c.b0 = static_cast<T>(n);
        c.b1 = static_cast<T>(-n);
        c.b2 = T(0);
        c.a1 = static_cast<T>((w - 1.0) * n);
        c.a2 = T(0);
        return c;
    }

    /**
     * @brief Creates a first-order tilt filter.
     *
     * Tilts the spectrum around a pivot frequency: +gainDb above the pivot,
     * -gainDb below. A single-knob tonal balance control found in mastering
     * EQs and channel strips (SSL, Neve, Tonelux Tilt).
     *
     * @param sampleRate Sample rate in Hz.
     * @param pivotFreq  Pivot frequency in Hz (typically 600-3000 Hz).
     * @param gainDb     Tilt amount in dB (positive = bright, negative = dark).
     */
    [[nodiscard]] static BiquadCoeffs makeTilt(double sampleRate, double pivotFreq, double gainDb) noexcept
    {
        pivotFreq = std::clamp(pivotFreq, 1.0, std::max(1.0, sampleRate * 0.499));
        double g = std::pow(10.0, gainDb / 20.0);
        double sqrtG = std::sqrt(g);
        double c = std::tan(std::numbers::pi * pivotFreq / sampleRate);

        // First-order tilt shelf via bilinear transform:
        //   DC gain = 1/sqrt(g), pivot gain = 1, Nyquist gain = sqrt(g)
        //   Total swing = gainDb (half below pivot, half above).
        double norm = 1.0 / (1.0 + sqrtG * c);

        BiquadCoeffs coeffs;
        coeffs.b0 = static_cast<T>((sqrtG + c) * norm);
        coeffs.b1 = static_cast<T>((c - sqrtG) * norm);
        coeffs.b2 = T(0);
        coeffs.a1 = static_cast<T>((sqrtG * c - 1.0) * norm);
        coeffs.a2 = T(0);
        return coeffs;
    }

    // -- Frequency response analysis -------------------------------------------

    /**
     * @brief Evaluates magnitude response |H(f)| at a single frequency.
     *
     * Evaluates the transfer function H(z) = B(z)/A(z) at z = e^(j*2*pi*f/fs).
     * Essential for drawing EQ curves and filter response plots.
     *
     * @param frequency  Frequency to evaluate in Hz.
     * @param sampleRate Sample rate in Hz.
     * @return Magnitude (linear scale, 1.0 = unity gain).
     */
    [[nodiscard]] T getMagnitude(double frequency, double sampleRate) const noexcept
    {
        double w = 2.0 * std::numbers::pi * frequency / sampleRate;
        double cosW  = std::cos(w);
        double cos2W = std::cos(2.0 * w);
        double sinW  = std::sin(w);
        double sin2W = std::sin(2.0 * w);

        double nRe = static_cast<double>(b0) + static_cast<double>(b1) * cosW + static_cast<double>(b2) * cos2W;
        double nIm = -static_cast<double>(b1) * sinW - static_cast<double>(b2) * sin2W;
        double dRe = 1.0 + static_cast<double>(a1) * cosW + static_cast<double>(a2) * cos2W;
        double dIm = -static_cast<double>(a1) * sinW - static_cast<double>(a2) * sin2W;

        double numMag2 = nRe * nRe + nIm * nIm;
        double denMag2 = dRe * dRe + dIm * dIm;

        return (denMag2 > 1e-30) ? static_cast<T>(std::sqrt(numMag2 / denMag2)) : T(0);
    }

    /**
     * @brief Computes magnitude responses for a batch of frequencies.
     *
     * Uses C++20 std::span for bounds-safe array access. Efficient batch
     * evaluation for drawing frequency response curves.
     *
     * @param frequencies Span of input frequencies in Hz.
     * @param magnitudes  Span where the evaluated magnitudes will be stored.
     * Must be at least as large as the frequencies span.
     * @param sampleRate  Sample rate in Hz.
     */
    void getMagnitudeForFrequencyArray(std::span<const T> frequencies,
                                       std::span<T> magnitudes,
                                       double sampleRate) const noexcept
    {
        assert(magnitudes.size() >= frequencies.size());
        for (size_t i = 0; i < frequencies.size(); ++i)
            magnitudes[i] = getMagnitude(static_cast<double>(frequencies[i]), sampleRate);
    }

private:
    /** @brief Normalises all coefficients by a0 (divides b* and a* by a0).
     * Division is performed in double precision to avoid premature truncation
     * when T is float; coefficients are only cast to T after the division. */
    [[nodiscard]] static BiquadCoeffs normalise(double a0, BiquadCoeffs raw) noexcept
    {
        double invA0 = 1.0 / a0;
        raw.b0 = static_cast<T>(static_cast<double>(raw.b0) * invA0);
        raw.b1 = static_cast<T>(static_cast<double>(raw.b1) * invA0);
        raw.b2 = static_cast<T>(static_cast<double>(raw.b2) * invA0);
        raw.a1 = static_cast<T>(static_cast<double>(raw.a1) * invA0);
        raw.a2 = static_cast<T>(static_cast<double>(raw.a2) * invA0);
        return raw;
    }
};

// ============================================================================
// Biquad -- Filter processor with per-channel state
// ============================================================================

/**
 * @class Biquad
 * @brief Biquad filter using Transposed Direct Form II (TDF-II) with thread-safe updates.
 *
 * Implements a lock-free shadow buffering system (seqlock) to prevent torn
 * reads when coefficients are updated by the UI thread concurrently with the
 * audio thread. Per-channel states are stored compactly so adjacent channels
 * share cache lines during block processing.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of independent filter channels.
 */
template <typename T, int MaxChannels = 8>
class alignas(32) Biquad
{
public:
    Biquad() noexcept = default;

    // std::atomic<bool> makes the default copy/move ops vanish. Provide manual
    // move semantics so this class can live inside std::array / std::vector
    // (e.g. CrossoverFilter::AllPassChain). Copying is left deleted on
    // purpose: it would imply two threads observing the same staged coefficient
    // dirty flag, which is semantically ambiguous and not needed in practice.
    // Moves only happen during setup (single-threaded relocation into a
    // container), so the seqlock counter is simply reset to an even value.
    Biquad(Biquad&& other) noexcept
        : activeCoeffs_(other.activeCoeffs_),
          stagedCoeffs_(other.stagedCoeffs_),
          coeffsDirty_(other.coeffsDirty_.load(std::memory_order_relaxed)),
          coeffsSeq_(0),
          state_(other.state_)
    {}

    Biquad& operator=(Biquad&& other) noexcept
    {
        if (this == &other) return *this;
        activeCoeffs_ = other.activeCoeffs_;
        stagedCoeffs_ = other.stagedCoeffs_;
        coeffsDirty_.store(other.coeffsDirty_.load(std::memory_order_relaxed),
                           std::memory_order_relaxed);
        coeffsSeq_.store(0, std::memory_order_relaxed);
        state_ = other.state_;
        return *this;
    }

    Biquad(const Biquad&)            = delete;
    Biquad& operator=(const Biquad&) = delete;

    /**
     * @brief Sets the filter coefficients asynchronously.
     *
     * Safely updates coefficients without locking. The audio thread will
     * pick up the new coefficients safely on the next process call to avoid
     * torn reads and filter blow-ups.
     *
     * @param c New coefficient set.
     */
    void setCoeffs(const BiquadCoeffs<T>& c) noexcept
    {
        // Seqlock publish (single producer). An odd sequence number marks a
        // write in progress; the audio thread retries its read while odd or if
        // the counter moved, so a concurrent update can never be observed as a
        // torn coefficient set (mixing a1 of one filter with a2 of another).
        //
        // Standards note: the reader's struct copy races with this write in the
        // strict C++ memory model (classic seqlock caveat; the Linux kernel and
        // mainstream audio frameworks rely on the same pattern). The retry loop
        // discards any value read during a write, and the acquire/release fences
        // order the accesses on every supported compiler/architecture.
        coeffsSeq_.fetch_add(1, std::memory_order_acq_rel);   // -> odd
        stagedCoeffs_ = c;
        coeffsSeq_.fetch_add(1, std::memory_order_release);   // -> even
        coeffsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Promotes any pending staged coefficients to active.
     *
     * Real-time safe. Both processBlock() and processSample() invoke this
     * automatically (the per-sample call is gated by a relaxed-load fast
     * path so it is essentially free when there is no pending update).
     *
     * Exposed publicly for the rare case where you have just called
     * setCoeffs() on this same thread and want the change reflected
     * immediately, e.g. for an offline / introspection getCoeffs() right
     * after pushing new ones.
     *
     * @return true if the active coefficients changed in this call.
     */
    bool applyPendingCoeffs() noexcept
    {
        if (coeffsDirty_.exchange(false, std::memory_order_acquire))
        {
            // Seqlock read: retry on a torn/in-progress publish. The writer's
            // critical section is a 5-float copy, so this converges in at most
            // a couple of iterations even under a busy GUI thread.
            BiquadCoeffs<T> tmp;
            unsigned s0, s1;
            do {
                s0  = coeffsSeq_.load(std::memory_order_acquire);
                tmp = stagedCoeffs_;
                // The fence keeps the copy above from sinking below the
                // re-read of the counter. A plain acquire load only orders
                // LATER accesses; without the fence, both the compiler and
                // weakly-ordered CPUs (ARM) may complete the copy after s1
                // is read, and a torn copy would pass the s0 == s1 check.
                std::atomic_thread_fence(std::memory_order_acquire);
                s1  = coeffsSeq_.load(std::memory_order_relaxed);
            } while ((s0 & 1u) != 0u || s0 != s1);
            activeCoeffs_ = tmp;
            return true;
        }
        return false;
    }

    /**
     * @brief Returns the active coefficient set currently in use by the DSP thread.
     *
     * Intended for the thread that owns processing (or single-threaded /
     * offline use). A GUI thread reading this concurrently with a promotion
     * may observe a mid-update set; for drawing response curves, keep your
     * own copy of the coefficients you computed.
     */
    [[nodiscard]] const BiquadCoeffs<T>& getCoeffs() const noexcept { return activeCoeffs_; }

    /** @brief Resets all per-channel filter states to zero to avoid ringing/clicks. */
    void reset() noexcept
    {
        for (auto& s : state_)
            s = {};
    }

    /**
     * @brief Processes a single sample for a specific channel.
     *
     * Transposed Direct Form II implementation. Self-sufficient: any
     * setCoeffs() update from another thread is picked up here on the very
     * next sample with virtually zero overhead; the fast-path is a relaxed
     * load (a plain MOV on x86) compiled into a single branchless check.
     *
     * No external sequencing is required. The caller is free to mix
     * processSample() and processBlock() calls in any order, and concurrent
     * setCoeffs() from the GUI thread always becomes audible deterministically
     * within at most a couple of samples (single sample on x86/ARM).
     *
     * @pre channel must be in [0, MaxChannels). Enforced by assert in debug
     * builds; out-of-range access in release builds is undefined behaviour.
     *
     * @param input   Input sample.
     * @param channel Channel index (0-based).
     * @return Filtered output sample.
     */
    T processSample(T input, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels && "Channel index out of bounds");

        // Lock-free fast path: relaxed load is free on every modern CPU.
        // Branch is marked unlikely so the compiler keeps the hot DSP path
        // straight-line; the dirty flag is only true on the first sample
        // of a block where setCoeffs() landed since last process call.
        if (coeffsDirty_.load(std::memory_order_relaxed)) [[unlikely]]
            applyPendingCoeffs();

        auto& s = state_[channel];

        const T output = activeCoeffs_.b0 * input + s.z1;
        s.z1 = activeCoeffs_.b1 * input - activeCoeffs_.a1 * output + s.z2;
        s.z2 = activeCoeffs_.b2 * input - activeCoeffs_.a2 * output;

        return output;
    }

    /**
     * @brief Processes a full audio buffer in-place.
     *
     * Thread-safe block processing. Absorbs asynchronous coefficient updates
     * at the start of the block to ensure atomic parameter changes.
     *
     * The inner loop runs on a local copy of the coefficients: with no
     * per-sample dirty-flag check in sight, the compiler keeps b0..a2 and the
     * filter state in registers for the whole block (roughly 1.5-2x faster
     * than routing every sample through processSample()).
     *
     * Channels beyond MaxChannels are left untouched (pass-through): only
     * the first min(numChannels, MaxChannels) channels are filtered.
     *
     * @param buffer Audio buffer to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;

        // Pick up any pending lock-free coefficient update from the GUI thread.
        applyPendingCoeffs();

        const int numChannels = std::min(buffer.getNumChannels(), MaxChannels);
        const int numSamples  = buffer.getNumSamples();

        // Block-local coefficient copy: stable for the whole block by design
        // (updates land at the next block boundary), and register-friendly.
        const BiquadCoeffs<T> c = activeCoeffs_;

        for (int ch = 0; ch < numChannels; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& s = state_[ch];
            T z1 = s.z1;
            T z2 = s.z2;

            for (int i = 0; i < numSamples; ++i)
            {
                const T input  = data[i];
                const T output = c.b0 * input + z1;
                z1 = c.b1 * input - c.a1 * output + z2;
                z2 = c.b2 * input - c.a2 * output;
                data[i] = output;
            }

            s.z1 = z1;
            s.z2 = z2;
        }
    }

private:
    // Kept compact on purpose (no over-alignment): the states are only ever
    // touched by the processing thread, sequentially per channel, so packing
    // adjacent channels into the same cache line is strictly better than
    // padding each one out to its own.
    struct State
    {
        T z1 = T(0);
        T z2 = T(0);
    };

    BiquadCoeffs<T> activeCoeffs_ {};
    BiquadCoeffs<T> stagedCoeffs_ {};
    std::atomic<bool> coeffsDirty_{false};
    std::atomic<unsigned> coeffsSeq_{0}; ///< Seqlock counter for tear-free staged publish.

    std::array<State, MaxChannels> state_ {};
};

} // namespace dspark