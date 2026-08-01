// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Biquad.h
 * @brief Biquad filter with Transposed Direct Form II, Audio EQ Cookbook coefficients, and thread-safe updates.
 *
 * Provides two classes:
 *
 * - **BiquadCoeffs**: Coefficient set with static factory methods for all
 * standard filter types. Formulas from Robert Bristow-Johnson's "Audio EQ Cookbook".
 * Memory aligned to 32 bytes for SIMD operations.
 *
 * - **Biquad<T, MaxChannels>**: Per-channel filter state using Transposed Direct
 * Form II (TDF-II). Features a lock-free shadow buffering system to allow safe
 * coefficient updates from the GUI thread without tearing or blocking the audio thread.
 *
 * Numerical design (why the filter core is double regardless of the sample type):
 * a biquad whose corner sits far below the sample rate parks its poles a hair
 * inside the unit circle - 1 - |z| is about pi*fc/(Q*fs), i.e. 5.8e-5 at 10 Hz /
 * 768 kHz - so the denominator evaluated at DC, 1 + a1 + a2 = (2 - 2*cos w0)/a0,
 * is a cancellation of terms of size 2 landing on 6.7e-9, far under the float
 * resolution of the terms it is made of (1.2e-7). Realised in float that sum is
 * no longer resolvable (measured 5.96e-08 against a true 2.68e-08 at 20 Hz /
 * 768 kHz, Q = 2.5628), the recursion stops being contractive and its rounding
 * error is integrated instead of decaying, which turns a high-pass into a DC
 * SOURCE and, steep enough, into an oscillator. Measured on the float core this
 * replaces, on a DC-free 1 kHz tone at -6 dBFS through FilterEngine: a 10 Hz
 * 12 dB/oct corner published -1.46e-01 of sustained DC at 384 kHz with the peak
 * inflated from 0.5068 to 0.6838 (+2.60 dB), a 20 Hz corner +6.62e-03 at
 * 768 kHz (peak 0.5130 -> 0.5821, +1.10 dB), and the realised DC gain - the sum
 * of the impulse response, which must be zero for a high-pass - reached +0.577
 * at 20 Hz / 768 kHz and -0.962 at 50 Hz / 768 kHz, i.e. the "high-pass" passed
 * DC at unity. The impulse response says it plainly: 20 Hz at 768 kHz, peak |h|
 * per half second, 24 dB/oct went 1.0 -> 4.1e-04 -> 4.8e-06 and then STOPPED
 * decaying (2.7e-06, 1.6e-06, 8.6e-07: a pole on the circle), and 48 dB/oct went
 * 1.0 -> 1.8e+19 -> 4.2e+37 -> non-finite - an unstable filter, reached from
 * setHighPass(20, 0.707, 48) at a rate a host hits by oversampling 48 kHz by 16.
 * In double the same cancellation carries an absolute error of about 2e-16, so
 * the poles land within 1e-8 (relative) of the design; the same points now decay
 * 1.0 -> 3.8e-04 -> 5.3e-15 -> 9.7e-26 and 1.0 -> 6.7e-04 -> 5.4e-10 -> 2.0e-15.
 * Scalar double throughput matches float on x86-64 and ARM64, so coefficients,
 * history and recursion are all double and only the buffer I/O is T;
 * processSampleCore() exists so a cascade can keep the intermediate signal in
 * the core precision instead of re-quantising it to T between stages.
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
 * @brief Stores normalised biquad coefficients (b0, b1, b2, a1, a2), always double.
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
 * There is no float instantiation and no coefficient type parameter: a corner
 * far below the sample rate is exact as a DESIGN but cannot be REALISED in
 * float (see the @file header for the measurements), and a coefficient set that
 * cannot be realised is not a coefficient set. One precision, one rule, and the
 * magnitude response drawn from these numbers is the response the audio path
 * actually applies. For DC removal specifically, use DCBlocker, which is built
 * for it.
 */
struct alignas(32) BiquadCoeffs
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;

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
        return normalise(a0,
            (1.0 - cosw0) / 2.0,
             1.0 - cosw0,
            (1.0 - cosw0) / 2.0,
            -2.0 * cosw0,
             1.0 - alpha);
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
        return normalise(a0,
             (1.0 + cosw0) / 2.0,
            -(1.0 + cosw0),
             (1.0 + cosw0) / 2.0,
            -2.0 * cosw0,
              1.0 - alpha);
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
        return normalise(a0,
            alpha,
            0.0,
            -alpha,
            -2.0 * cosw0,
             1.0 - alpha);
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
        return normalise(a0,
            1.0 + alpha * A,
            -2.0 * cosw0,
            1.0 - alpha * A,
            -2.0 * cosw0,
            1.0 - alpha / A);
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
            return { 1.0, 0.0, 0.0, 0.0, 0.0 };

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
        return normalise(a0,
            G1 + G0 * W2 + B,
            -2.0 * (G1 - G0 * W2),
            G1 + G0 * W2 - B,
            -2.0 * (1.0 - W2),
            1.0 + W2 - A);
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
        return normalise(a0,
            A * ((A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha),
            2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0),
            A * ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha),
            -2.0 * ((A - 1.0) + (A + 1.0) * cosw0),
            (A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
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
        return normalise(a0,
            A * ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha),
            -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0),
            A * ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha),
            2.0 * ((A - 1.0) - (A + 1.0) * cosw0),
            (A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha);
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
        return normalise(a0,
            1.0,
            -2.0 * cosw0,
            1.0,
            -2.0 * cosw0,
            1.0 - alpha);
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
        return normalise(a0,
            1.0 - alpha,
            -2.0 * cosw0,
            1.0 + alpha,
            -2.0 * cosw0,
            1.0 - alpha);
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
        const double w = std::tan(std::numbers::pi * frequency / sampleRate);
        const double n = 1.0 / (1.0 + w);
        BiquadCoeffs c;
        c.b0 = w * n;
        c.b1 = w * n;
        c.b2 = 0.0;
        c.a1 = (w - 1.0) * n;
        c.a2 = 0.0;
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
        const double w = std::tan(std::numbers::pi * frequency / sampleRate);
        const double n = 1.0 / (1.0 + w);
        BiquadCoeffs c;
        c.b0 = n;
        c.b1 = -n;
        c.b2 = 0.0;
        c.a1 = (w - 1.0) * n;
        c.a2 = 0.0;
        return c;
    }

    /**
     * @brief Creates a first-order tilt filter.
     *
     * Tilts the spectrum around a pivot frequency: -gainDb/2 at DC, unity at
     * the pivot, +gainDb/2 at Nyquist (total span = gainDb). A single-knob
     * tonal balance control found in mastering EQs and channel strips
     * (SSL, Neve, Tonelux Tilt).
     *
     * @param sampleRate Sample rate in Hz.
     * @param pivotFreq  Pivot frequency in Hz (typically 600-3000 Hz).
     * @param gainDb     Tilt amount in dB (positive = bright, negative = dark).
     */
    [[nodiscard]] static BiquadCoeffs makeTilt(double sampleRate, double pivotFreq, double gainDb) noexcept
    {
        pivotFreq = std::clamp(pivotFreq, 1.0, std::max(1.0, sampleRate * 0.499));
        const double g = std::pow(10.0, gainDb / 20.0);
        const double sqrtG = std::sqrt(g);
        const double c = std::tan(std::numbers::pi * pivotFreq / sampleRate);

        // First-order tilt shelf via bilinear transform:
        //   DC gain = 1/sqrt(g), pivot gain = 1, Nyquist gain = sqrt(g)
        //   Total swing = gainDb (half below pivot, half above).
        const double norm = 1.0 / (1.0 + sqrtG * c);

        BiquadCoeffs coeffs;
        coeffs.b0 = (sqrtG + c) * norm;
        coeffs.b1 = (c - sqrtG) * norm;
        coeffs.b2 = 0.0;
        coeffs.a1 = (sqrtG * c - 1.0) * norm;
        coeffs.a2 = 0.0;
        return coeffs;
    }

    // -- Frequency response analysis -------------------------------------------

    /**
     * @brief Evaluates magnitude response |H(f)| at a single frequency.
     *
     * Evaluates the transfer function H(z) = B(z)/A(z) at z = e^(j*2*pi*f/fs).
     * Essential for drawing EQ curves and filter response plots. These are the
     * same numbers the recursion runs on, so the drawn curve is the realised
     * response, not a re-derivation of it.
     *
     * @param frequency  Frequency to evaluate in Hz.
     * @param sampleRate Sample rate in Hz.
     * @return Magnitude (linear scale, 1.0 = unity gain).
     */
    [[nodiscard]] double getMagnitude(double frequency, double sampleRate) const noexcept
    {
        const double w = 2.0 * std::numbers::pi * frequency / sampleRate;
        const double cosW  = std::cos(w);
        const double cos2W = std::cos(2.0 * w);
        const double sinW  = std::sin(w);
        const double sin2W = std::sin(2.0 * w);

        const double nRe = b0 + b1 * cosW + b2 * cos2W;
        const double nIm = -b1 * sinW - b2 * sin2W;
        const double dRe = 1.0 + a1 * cosW + a2 * cos2W;
        const double dIm = -a1 * sinW - a2 * sin2W;

        const double numMag2 = nRe * nRe + nIm * nIm;
        const double denMag2 = dRe * dRe + dIm * dIm;

        return (denMag2 > 1e-30) ? std::sqrt(numMag2 / denMag2) : 0.0;
    }

    /**
     * @brief Computes magnitude responses for a batch of frequencies.
     *
     * Uses C++20 std::span for bounds-safe array access. Efficient batch
     * evaluation for drawing frequency response curves. Templated on the array
     * element type so a UI can keep its curve buffers in float.
     *
     * @param frequencies Span of input frequencies in Hz.
     * @param magnitudes  Span where the evaluated magnitudes will be stored.
     * Must be at least as large as the frequencies span.
     * @param sampleRate  Sample rate in Hz.
     */
    template <typename U>
    void getMagnitudeForFrequencyArray(std::span<const U> frequencies,
                                       std::span<U> magnitudes,
                                       double sampleRate) const noexcept
    {
        assert(magnitudes.size() >= frequencies.size());
        for (size_t i = 0; i < frequencies.size(); ++i)
            magnitudes[i] = static_cast<U>(getMagnitude(static_cast<double>(frequencies[i]), sampleRate));
    }

private:
    /** @brief Normalises a raw coefficient set by a0 (divides b* and a* by a0).
     * Takes the raw values one by one rather than a BiquadCoeffs by value: the
     * struct is over-aligned, and passing it through a register-class boundary
     * is exactly what GCC's -Wpsabi note is about. */
    [[nodiscard]] static BiquadCoeffs normalise(double a0, double b0r, double b1r, double b2r,
                                                double a1r, double a2r) noexcept
    {
        const double invA0 = 1.0 / a0;
        return { b0r * invA0, b1r * invA0, b2r * invA0, a1r * invA0, a2r * invA0 };
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
 * audio thread. The shared staging slot is made of std::atomic<double> words
 * (relaxed inside the seq-counter critical section, with a writer release /
 * reader acquire fence pair), so the handoff is data-race free by the C++
 * memory model, not merely tear-free in practice; the audio thread promotes
 * into its private, plain activeCoeffs_, so the per-sample recursion is
 * untouched. Per-channel states are stored compactly so adjacent channels
 * share cache lines during block processing.
 *
 * Threading (ADR-013 SPSC model):
 * - setCoeffs(): control thread (ONE non-audio writer; canonical seqlock
 *   publish).
 * - processSample() / processSampleCore() / processBlock() /
 *   applyPendingCoeffs(): audio thread (single stream owner).
 * - reset(): stream owner only (writes the plain per-channel states).
 * - getCoeffs(): processing thread or offline only (returns a reference to
 *   the audio-thread-private active set -- see its own note).
 * - Move construction/assignment: setup-time only (single-threaded
 *   relocation).
 *
 * The filter core (coefficients, history and recursion) is always double
 * precision, independent of the sample type: see the @file header for the
 * measurements behind that decision. The realised response is therefore
 * rate-independent down to the corners the designs already supported on paper.
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
          coeffsDirty_(other.coeffsDirty_.load(std::memory_order_relaxed)),
          coeffsSeq_(0),
          state_(other.state_)
    {
        copyStagedRelaxed(other); // Word-wise: the staging words are atomics.
    }

    Biquad& operator=(Biquad&& other) noexcept
    {
        if (this == &other) return *this;
        activeCoeffs_ = other.activeCoeffs_;
        copyStagedRelaxed(other); // Word-wise: the staging words are atomics.
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
    void setCoeffs(const BiquadCoeffs& c) noexcept
    {
        // Seqlock publish (single producer). An odd sequence number marks a
        // write in progress; the audio thread retries its read while odd or if
        // the counter moved, so a concurrent update can never be observed as a
        // torn coefficient set (mixing a1 of one filter with a2 of another).
        //
        // Data-race freedom: the staging slot is 5 std::atomic<double> words
        // stored relaxed inside the critical section, so every cross-thread
        // word access is defined by the C++ memory model (the previous plain
        // struct copy was formal UB -- the same defect class CI TSan flagged in
        // FIRFilter, D-M002-C3). The release fence below pairs with the
        // reader's acquire fence ([atomics.fences]/2): a reader that observes
        // any mid-publish word must also observe the odd counter and retry
        // (Boehm, "Can Seqlocks Get Along With Programming Language Memory
        // Models?", MSPC 2012). Control-thread only: zero audio-path cost.
        coeffsSeq_.fetch_add(1, std::memory_order_acq_rel);   // -> odd
        std::atomic_thread_fence(std::memory_order_release);
        stagedCoeffs_.b0.store(c.b0, std::memory_order_relaxed);
        stagedCoeffs_.b1.store(c.b1, std::memory_order_relaxed);
        stagedCoeffs_.b2.store(c.b2, std::memory_order_relaxed);
        stagedCoeffs_.a1.store(c.a1, std::memory_order_relaxed);
        stagedCoeffs_.a2.store(c.a2, std::memory_order_relaxed);
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
            // critical section is a 5-double copy, so this converges in at most
            // a couple of iterations even under a busy GUI thread.
            BiquadCoeffs tmp;
            unsigned s0, s1;
            do {
                s0  = coeffsSeq_.load(std::memory_order_acquire);
                // Relaxed word loads: each access is race-free because the
                // staging words are std::atomic<double>.
                tmp.b0 = stagedCoeffs_.b0.load(std::memory_order_relaxed);
                tmp.b1 = stagedCoeffs_.b1.load(std::memory_order_relaxed);
                tmp.b2 = stagedCoeffs_.b2.load(std::memory_order_relaxed);
                tmp.a1 = stagedCoeffs_.a1.load(std::memory_order_relaxed);
                tmp.a2 = stagedCoeffs_.a2.load(std::memory_order_relaxed);
                // The fence keeps the copy above from sinking below the
                // re-read of the counter. A plain acquire load only orders
                // LATER accesses; without the fence, both the compiler and
                // weakly-ordered CPUs (ARM) may complete the copy after s1
                // is read, and a torn copy would pass the s0 == s1 check.
                // It also pairs with the writer's release fence, so a copy
                // that read any mid-publish word cannot pass validation.
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
    [[nodiscard]] const BiquadCoeffs& getCoeffs() const noexcept { return activeCoeffs_; }

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
        return static_cast<T>(processSampleCore(static_cast<double>(input), channel));
    }

    /**
     * @brief One recursion step in the core's own precision (double).
     *
     * Same filter, same state, no conversion at the boundary: a cascade calls
     * this to keep the intermediate signal in the core precision instead of
     * re-quantising it to T between stages (FilterEngine does). Identical
     * contract to processSample() otherwise, including the lock-free pickup of
     * pending coefficients.
     *
     * @pre channel must be in [0, MaxChannels).
     *
     * @param input   Input sample.
     * @param channel Channel index (0-based).
     * @return Filtered output sample.
     */
    double processSampleCore(double input, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels && "Channel index out of bounds");

        // Lock-free fast path: relaxed load is free on every modern CPU.
        // Branch is marked unlikely so the compiler keeps the hot DSP path
        // straight-line; the dirty flag is only true on the first sample
        // of a block where setCoeffs() landed since last process call.
        if (coeffsDirty_.load(std::memory_order_relaxed)) [[unlikely]]
            applyPendingCoeffs();

        auto& s = state_[channel];

        const double output = activeCoeffs_.b0 * input + s.z1;
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
        const BiquadCoeffs c = activeCoeffs_;

        for (int ch = 0; ch < numChannels; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& s = state_[ch];
            double z1 = s.z1;
            double z2 = s.z2;

            for (int i = 0; i < numSamples; ++i)
            {
                const double input  = static_cast<double>(data[i]);
                const double output = c.b0 * input + z1;
                z1 = c.b1 * input - c.a1 * output + z2;
                z2 = c.b2 * input - c.a2 * output;
                data[i] = static_cast<T>(output);
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
        double z1 = 0.0;
        double z2 = 0.0;
    };

    // Shared staging slot for the seqlock publish: written word-by-word by the
    // control thread inside the odd/even critical section and read word-by-word
    // by the audio thread's retry loop. Every word is std::atomic<double>
    // (relaxed) so each cross-thread access is defined by the C++ memory model
    // (no non-atomic concurrent read/write, i.e. no data-race UB); the seq
    // counter plus the writer release / reader acquire fence pair order and
    // tear-protect the 5-word set. std::atomic<double> is lock-free and one
    // machine word on every supported target, so the publish stays lock- and
    // allocation-free. Defaults mirror BiquadCoeffs{} (identity filter).
    struct StagedCoeffs
    {
        std::atomic<double> b0{1.0}, b1{0.0}, b2{0.0};
        std::atomic<double> a1{0.0}, a2{0.0};
    };

    /** @brief Word-wise relaxed copy of the staging slot (setup-time moves only). */
    void copyStagedRelaxed(const Biquad& other) noexcept
    {
        stagedCoeffs_.b0.store(other.stagedCoeffs_.b0.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        stagedCoeffs_.b1.store(other.stagedCoeffs_.b1.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        stagedCoeffs_.b2.store(other.stagedCoeffs_.b2.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        stagedCoeffs_.a1.store(other.stagedCoeffs_.a1.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
        stagedCoeffs_.a2.store(other.stagedCoeffs_.a2.load(std::memory_order_relaxed),
                               std::memory_order_relaxed);
    }

    BiquadCoeffs activeCoeffs_ {};
    StagedCoeffs stagedCoeffs_ {};
    std::atomic<bool> coeffsDirty_{false};
    std::atomic<unsigned> coeffsSeq_{0}; ///< Seqlock counter for tear-free staged publish.

    std::array<State, MaxChannels> state_ {};
};

} // namespace dspark
