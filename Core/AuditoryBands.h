// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file AuditoryBands.h
 * @brief An auditory front end: bands uniform on the Bark scale over a power
 *        spectrum, smoothing that follows the ear's resolution, specific and
 *        total loudness, and the level of the dominant component of a stereo
 *        pair.
 *
 * The formulas are the published ones:
 *
 * - Bark scale after Traunmuller (1990): z = 26.81 f / (1960 + f) - 0.53,
 *   with his corrections below 2 and above 20.1 Bark.
 * - Critical bandwidth after Zwicker and Terhardt (1980):
 *   25 + 75 (1 + 1.4 (f / 1 kHz)^2)^0.69 Hz.
 * - Threshold in quiet after Terhardt (1979):
 *   3.64 (f / 1 kHz)^-0.8 - 6.5 exp(-0.6 (f / 1 kHz - 3.3)^2)
 *   + 0.001 (f / 1 kHz)^4 dB SPL.
 * - Specific loudness as a power law over the threshold (Zwicker):
 *   N' = (P + P_th)^a - P_th^a, with the exponent a rising from 0.23 where
 *   the threshold is at its lowest toward 0.30 where it is high (the
 *   loudness of the low end grows faster with level, as the equal-loudness
 *   contours show by drawing closer at high levels).
 *
 * AuditoryBands maps the bins of a power spectrum to bands uniform in Bark:
 * each bin feeds the two bands around it with triangular weights in Bark
 * (the weights of a bin sum to one), so a band's power is the power of a
 * band one spacing wide; where a band is narrower than a bin it reads the
 * power density at its centre by linear interpolation between bins, times
 * its width. The level of a full-scale sine is the reference: its power is
 * referred to setReferenceSpl() (100 dB SPL at 0 dBFS by default) for the
 * loudness law.
 *
 * dominantPower() is the level of the dominant component of a stereo pair,
 * wherever it sits in the image: the larger eigenvalue of the 2 x 2
 * covariance of the two channels.
 *
 * Threading: prepare() on the setup thread (allocates); the rest on the
 * owner's thread (no state is shared).
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dspark {

/** @brief The published formulas of the auditory scale (all in double). */
struct AuditoryScale
{
    /** @return The Bark value of a frequency (Traunmuller 1990). */
    [[nodiscard]] static double hzToBark(double hz) noexcept
    {
        const double f = std::max(hz, 0.0);
        double z = 26.81 * f / (1960.0 + f) - 0.53;
        if (z < 2.0) z += 0.15 * (2.0 - z);
        else if (z > 20.1) z += 0.22 * (z - 20.1);
        return z;
    }

    /** @return The frequency of a Bark value (the inverse of hzToBark()). */
    [[nodiscard]] static double barkToHz(double bark) noexcept
    {
        double z = bark;
        if (z < 2.0) z = (z - 0.3) / 0.85;                 // undo z + 0.15 (2 - z)
        else if (z > 20.1) z = (z + 0.22 * 20.1) / 1.22;   // undo z + 0.22 (z - 20.1)
        z = std::clamp(z, -0.52, 26.0);
        return 1960.0 * (z + 0.53) / (26.28 - z);
    }

    /** @return The critical bandwidth at a frequency, in hertz (Zwicker and Terhardt 1980). */
    [[nodiscard]] static double criticalBandwidth(double hz) noexcept
    {
        const double k = std::max(hz, 0.0) / 1000.0;
        return 25.0 + 75.0 * std::pow(1.0 + 1.4 * k * k, 0.69);
    }

    /** @return The threshold in quiet at a frequency, in dB SPL (Terhardt 1979), capped at 120. */
    [[nodiscard]] static double thresholdInQuietDb(double hz) noexcept
    {
        const double k = std::clamp(hz, 10.0, 24000.0) / 1000.0;
        const double t = 3.64 * std::pow(k, -0.8) - 6.5 * std::exp(-0.6 * (k - 3.3) * (k - 3.3)) + 1e-3 * k * k * k * k;
        return std::min(t, 120.0);
    }

    /**
     * @return The exponent of the specific loudness at a frequency: 0.23
     *         where the threshold in quiet is lowest, rising linearly with
     *         the threshold to 0.30 for 40 dB above that minimum.
     */
    [[nodiscard]] static double loudnessExponent(double hz) noexcept
    {
        constexpr double kLowest = -5.0;   // the threshold's minimum, near 3.3 kHz
        return 0.23 + 0.07 * std::clamp((thresholdInQuietDb(hz) - kLowest) / 40.0, 0.0, 1.0);
    }

    /**
     * @return The power of the dominant component of a stereo pair: the
     *         larger eigenvalue of the covariance [ll lr; lr rr].
     */
    [[nodiscard]] static double dominantPower(double ll, double rr, double lr) noexcept
    {
        return dominantPairPower(ll, rr, lr);
    }

    /**
     * @return The dominant power of a pair read in a band of frequencies,
     *         whose cross term is complex (the mean of one spectrum times
     *         the conjugate of the other): a channel that is the other one
     *         delayed reads as one source, whatever the delay turns the
     *         phase to in the band.
     */
    [[nodiscard]] static double dominantPower(double ll, double rr, double lrReal, double lrImag) noexcept
    {
        return dominantPairPower(ll, rr, std::sqrt(lrReal * lrReal + lrImag * lrImag));
    }
};

/**
 * @class AuditoryBands
 * @brief Bands uniform in Bark over a power spectrum, with the loudness law.
 *
 * @tparam T Sample type of the spectra (float or double).
 */
template <FloatType T>
class AuditoryBands
{
public:
    /**
     * @brief Lays the bands out.
     * @param sampleRate Rate of the spectrum (invalid values are ignored).
     * @param fftSize    Size of the transform (the spectrum has fftSize / 2 + 1 bins).
     * @param numBands   Number of bands, 2 to 512.
     * @param loHz       Centre of the lowest band.
     * @param hiHz       Centre of the highest band (at most 0.49 of the rate).
     */
    void prepare(double sampleRate, int fftSize, int numBands = 72, double loHz = 20.0, double hiHz = 20000.0)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0) || fftSize < 8) return;
        fs_ = sampleRate;
        numBins_ = fftSize / 2 + 1;
        binHz_ = fs_ / static_cast<double>(fftSize);
        n_ = std::clamp(numBands, 2, 512);
        const double hi = std::clamp(hiHz, 40.0, 0.49 * fs_);
        const double lo = std::clamp(loHz, 10.0, 0.5 * hi);
        z0_ = AuditoryScale::hzToBark(lo);
        dz_ = (AuditoryScale::hzToBark(hi) - z0_) / static_cast<double>(n_ - 1);
        centre_.resize(static_cast<size_t>(n_));
        bark_.resize(static_cast<size_t>(n_));
        exponent_.resize(static_cast<size_t>(n_));
        threshold_.resize(static_cast<size_t>(n_));
        narrow_.resize(static_cast<size_t>(n_));
        widthBins_.resize(static_cast<size_t>(n_));
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            bark_[i] = z0_ + dz_ * b;
            centre_[i] = AuditoryScale::barkToHz(bark_[i]);
            exponent_[i] = AuditoryScale::loudnessExponent(centre_[i]);
            // A band one spacing wide, in bins.
            widthBins_[i] = (AuditoryScale::barkToHz(bark_[i] + 0.5 * dz_) - AuditoryScale::barkToHz(bark_[i] - 0.5 * dz_)) / binHz_;
            narrow_[i] = widthBins_[i] < 2.0;
        }
        // Each bin's two bands and their triangular weights.
        binBand_.assign(static_cast<size_t>(numBins_), -2);
        binWeight_.assign(static_cast<size_t>(numBins_), 0.0);
        for (int k = 1; k < numBins_; ++k)
        {
            const double pos = (AuditoryScale::hzToBark(binHz_ * k) - z0_) / dz_;
            if (pos < -1.0 || pos > static_cast<double>(n_)) continue;
            const int lower = static_cast<int>(std::floor(pos));
            binBand_[static_cast<size_t>(k)] = lower;
            binWeight_[static_cast<size_t>(k)] = 1.0 - (pos - lower);   // to `lower`; the rest to lower + 1
        }
        setReferenceSpl(refSpl_);
    }

    /** @brief The SPL of a full-scale sine for the loudness law (60 to 130 dB). */
    void setReferenceSpl(double dbSpl) noexcept
    {
        if (!std::isfinite(dbSpl)) return;
        refSpl_ = std::clamp(dbSpl, 60.0, 130.0);
        for (int b = 0; b < n_; ++b)
            threshold_[static_cast<size_t>(b)] = std::pow(10.0, (AuditoryScale::thresholdInQuietDb(centre_[static_cast<size_t>(b)]) - refSpl_) / 10.0);
    }

    [[nodiscard]] int getNumBands() const noexcept { return n_; }
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }
    /** @return The centre of a band in hertz. */
    [[nodiscard]] double getCentre(int band) const noexcept { return centre_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The centre of a band in Bark. */
    [[nodiscard]] double getBark(int band) const noexcept { return bark_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The spacing of the bands in Bark. */
    [[nodiscard]] double getSpacing() const noexcept { return dz_; }
    /** @return The loudness exponent of a band. */
    [[nodiscard]] double getExponent(int band) const noexcept { return exponent_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The threshold in quiet of a band, as a power relative to a full-scale sine. */
    [[nodiscard]] double getThreshold(int band) const noexcept { return threshold_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }

    /**
     * @brief Band powers from a power spectrum.
     * @param binPower numBins powers (|X[k]|^2 scaled so a full-scale sine
     *                 reads 1 in its bin, or any consistent scale).
     * @param out      getNumBands() powers.
     */
    void bandPowers(const T* binPower, T* out) const noexcept
    {
        for (int b = 0; b < n_; ++b) out[b] = T(0);
        for (int k = 1; k < numBins_; ++k)
        {
            const int lower = binBand_[static_cast<size_t>(k)];
            if (lower < -1) continue;
            const double p = static_cast<double>(binPower[k]);
            const double w = binWeight_[static_cast<size_t>(k)];
            if (lower >= 0 && lower < n_ && !narrow_[static_cast<size_t>(lower)]) out[lower] += static_cast<T>(w * p);
            if (lower + 1 >= 0 && lower + 1 < n_ && !narrow_[static_cast<size_t>(lower + 1)]) out[lower + 1] += static_cast<T>((1.0 - w) * p);
        }
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            if (!narrow_[i]) continue;
            const double pos = std::clamp(centre_[i] / binHz_, 0.0, static_cast<double>(numBins_ - 1));
            const int k = std::min(static_cast<int>(pos), numBins_ - 2);
            const double f = pos - k;
            out[b] = static_cast<T>(widthBins_[i] * ((1.0 - f) * static_cast<double>(binPower[k]) + f * static_cast<double>(binPower[k + 1])));
        }
    }

    /**
     * @brief Smooths values across bands in both directions with a one-pole
     *        (forward, then backward): a spread like the ear's masking.
     * @param v     The values, in place.
     * @param n     How many.
     * @param coeff Pole, 0 (none) to just under 1.
     */
    static void smoothAcross(T* v, int n, T coeff) noexcept
    {
        if (n < 2 || !(coeff > T(0))) return;
        const T c = std::min(coeff, T(0.999));
        for (int b = 1; b < n; ++b) v[b] += c * (v[b - 1] - v[b]);
        for (int b = n - 2; b >= 0; --b) v[b] += c * (v[b + 1] - v[b]);
    }

    /**
     * @brief Specific loudness of band powers: (P + P_th)^a - P_th^a, with
     *        the band's exponent and threshold.
     */
    void specificLoudness(const T* bandPower, T* out) const noexcept
    {
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double th = threshold_[i], a = exponent_[i];
            out[b] = static_cast<T>(std::pow(std::max(0.0, static_cast<double>(bandPower[b])) + th, a) - std::pow(th, a));
        }
    }

    /** @return The total loudness: the specific loudness integrated over Bark (trapezoids). */
    [[nodiscard]] T totalLoudness(const T* specific) const noexcept
    {
        double s = 0.0;
        for (int b = 1; b < n_; ++b) s += 0.5 * (static_cast<double>(specific[b - 1]) + static_cast<double>(specific[b]));
        return static_cast<T>(s * dz_);
    }

private:
    double fs_ = 48000.0, binHz_ = 23.4375, z0_ = 0.0, dz_ = 0.3, refSpl_ = 100.0;
    int numBins_ = 0, n_ = 0;
    std::vector<double> centre_, bark_, exponent_, threshold_, widthBins_;
    std::vector<bool> narrow_;
    std::vector<int> binBand_;
    std::vector<double> binWeight_;
};

} // namespace dspark
