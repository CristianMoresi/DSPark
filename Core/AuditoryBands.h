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
 * A third of a critical band is about 30 Hz wide under 500 Hz, whatever the
 * frequency: three bands from 20 to 100 Hz. prepareDetailed() lays the bands
 * out with no band wider than a fraction of an octave: under the frequency
 * where the two widths meet the bands follow the octave, and over it the
 * Bark scale as before. prepareLike() takes the bands of another instance
 * over a spectrum of its own (another transform, another rate), so two
 * analyses of different resolution can feed one set of bands.
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
#include <array>
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
        thresholdLoudness_.resize(static_cast<size_t>(n_));
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
        uniform_ = true;
        widthBark_.assign(static_cast<size_t>(n_), dz_);
        widthBark_.front() = widthBark_.back() = 0.5 * dz_;
        setReferenceSpl(refSpl_);
    }

    /**
     * @brief Lays the bands out with more of them in the low end.
     *
     * Starts from the layout of prepare() (barkBands bands uniform in Bark
     * from loHz to hiHz) and lets no band be wider than 1 / bandsPerOctave
     * of an octave: the density of bands along frequency is the larger of
     * the two, so the bands follow the octave in the low end and the Bark
     * scale from where a band of that scale is the narrower one. The number
     * of bands follows from it (getNumBands()).
     *
     * @param sampleRate     Rate of the spectrum (invalid values are ignored).
     * @param fftSize        Size of the transform.
     * @param barkBands      Bands of the Bark layout it starts from, 2 to 512.
     * @param loHz           Centre of the lowest band.
     * @param hiHz           Centre of the highest band (at most 0.49 of the rate).
     * @param bandsPerOctave Bands per octave at least; 0 or less is prepare().
     */
    void prepareDetailed(double sampleRate, int fftSize, int barkBands, double loHz, double hiHz, double bandsPerOctave)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0) || fftSize < 8) return;
        if (!std::isfinite(bandsPerOctave) || !(bandsPerOctave > 0.0))
        {
            prepare(sampleRate, fftSize, barkBands, loHz, hiHz);
            return;
        }
        fs_ = sampleRate;
        const int base = std::clamp(barkBands, 2, 512);
        const double hi = std::clamp(hiHz, 40.0, 0.49 * fs_);
        const double lo = std::clamp(loHz, 10.0, 0.5 * hi);
        z0_ = AuditoryScale::hzToBark(lo);
        dz_ = (AuditoryScale::hzToBark(hi) - z0_) / static_cast<double>(base - 1);
        // Bands counted along the logarithm of frequency, with the density of
        // the Bark layout or of the octave, whichever is the larger.
        constexpr int kSteps = 4096;
        std::vector<double> logHz(kSteps + 1), count(kSteps + 1);
        const double span = std::log(hi / lo);
        const double perOctave = bandsPerOctave / 0.6931471805599453;
        double before = 0.0;
        for (int i = 0; i <= kSteps; ++i)
        {
            const double x = std::log(lo) + span * static_cast<double>(i) / kSteps;
            const double hz = std::exp(x), step = 1e-3 * hz;
            const double perBark = (AuditoryScale::hzToBark(hz + step) - AuditoryScale::hzToBark(hz - step)) / (2.0 * step) * hz / dz_;
            const double density = std::max(perBark, perOctave);
            logHz[static_cast<size_t>(i)] = x;
            count[static_cast<size_t>(i)] = i == 0 ? 0.0
                : count[static_cast<size_t>(i - 1)] + 0.5 * (density + before) * span / kSteps;
            before = density;
        }
        const double total = count[kSteps];
        n_ = std::clamp(static_cast<int>(std::lround(total)) + 1, 2, 512);
        resizeBands();
        size_t at = 0;
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double want = total * static_cast<double>(b) / static_cast<double>(n_ - 1);
            while (at + 2 <= static_cast<size_t>(kSteps) && count[at + 1] < want) ++at;
            const double rise = count[at + 1] - count[at];
            const double part = rise > 0.0 ? std::clamp((want - count[at]) / rise, 0.0, 1.0) : 0.0;
            centre_[i] = b == 0 ? lo : (b == n_ - 1 ? hi : std::exp(logHz[at] + part * (logHz[at + 1] - logHz[at])));
            bark_[i] = AuditoryScale::hzToBark(centre_[i]);
            exponent_[i] = AuditoryScale::loudnessExponent(centre_[i]);
        }
        mapBins(fftSize);
        setReferenceSpl(refSpl_);
    }

    /**
     * @brief Takes the bands of another instance over a spectrum of its own.
     *
     * The centres, the thresholds and the exponents are those of the other
     * instance; the bins are those of this rate and this transform. A band
     * over half this rate reads nothing that means anything.
     *
     * @param other      The instance whose bands are taken (already prepared).
     * @param sampleRate Rate of this spectrum (invalid values are ignored).
     * @param fftSize    Size of this transform.
     */
    void prepareLike(const AuditoryBands& other, double sampleRate, int fftSize)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0) || fftSize < 8 || other.n_ < 2) return;
        fs_ = sampleRate;
        n_ = other.n_;
        z0_ = other.z0_;
        dz_ = other.dz_;
        refSpl_ = other.refSpl_;
        resizeBands();
        centre_ = other.centre_;
        bark_ = other.bark_;
        exponent_ = other.exponent_;
        mapBins(fftSize);
        setReferenceSpl(refSpl_);
    }

    /** @brief The SPL of a full-scale sine for the loudness law (60 to 130 dB). */
    void setReferenceSpl(double dbSpl) noexcept
    {
        if (!std::isfinite(dbSpl)) return;
        refSpl_ = std::clamp(dbSpl, 60.0, 130.0);
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            threshold_[i] = std::pow(10.0, (AuditoryScale::thresholdInQuietDb(centre_[i]) - refSpl_) / 10.0);
            thresholdLoudness_[i] = std::pow(threshold_[i], exponent_[i]);
        }
    }

    [[nodiscard]] int getNumBands() const noexcept { return n_; }
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }
    /** @return The centre of a band in hertz. */
    [[nodiscard]] double getCentre(int band) const noexcept { return centre_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The centre of a band in Bark. */
    [[nodiscard]] double getBark(int band) const noexcept { return bark_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The spacing of the bands in Bark (of the Bark layout: the widest a band is). */
    [[nodiscard]] double getSpacing() const noexcept { return dz_; }
    /**
     * @return The stretch of the Bark axis a band stands for: from half way
     *         to the band below to half way to the band above (the first
     *         and the last take the half toward their only neighbour), which
     *         is its weight in a sum over the bands that stands for an
     *         integral over Bark.
     */
    [[nodiscard]] double getWidthBark(int band) const noexcept { return widthBark_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return Whether the bands are uniform in Bark (prepare(), not prepareDetailed()). */
    [[nodiscard]] bool isUniform() const noexcept { return uniform_; }
    /** @return The loudness exponent of a band. */
    [[nodiscard]] double getExponent(int band) const noexcept { return exponent_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The threshold in quiet of a band, as a power relative to a full-scale sine. */
    [[nodiscard]] double getThreshold(int band) const noexcept { return threshold_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }
    /** @return The threshold to the exponent of a band: what specificLoudness() takes off. */
    [[nodiscard]] double getThresholdLoudness(int band) const noexcept { return thresholdLoudness_[static_cast<size_t>(std::clamp(band, 0, n_ - 1))]; }

    /**
     * @brief Band powers from a power spectrum.
     * @param binPower numBins powers (|X[k]|^2 scaled so a full-scale sine
     *                 reads 1 in its bin, or any consistent scale).
     * @param out      getNumBands() powers.
     */
    void bandPowers(const T* binPower, T* out) const noexcept
    {
        bandPowersInterleaved<1>(binPower, out);
    }

    /**
     * @brief The band powers of several spectra at once: each what
     *        bandPowers() gives for its own.
     * @tparam Lanes How many spectra (1 to 8).
     * @param binPower numBins groups of `Lanes` powers, one of each spectrum.
     * @param out      getNumBands() groups of `Lanes` powers, the same way.
     */
    template <int Lanes>
    void bandPowersInterleaved(const T* binPower, T* out) const noexcept
    {
        static_assert(Lanes >= 1 && Lanes <= 8);
        constexpr auto lanes = static_cast<size_t>(Lanes);
        std::fill_n(out, n_ * Lanes, T(0));
        // The bins that feed the same two bands lie together: their sums are
        // carried in registers from the first to the last of them.
        for (int k = 1; k < numBins_;)
        {
            const int lower = binBand_[static_cast<size_t>(k)];
            if (lower < -1)
            {
                ++k;
                continue;
            }
            const bool intoLower = lower >= 0 && lower < n_ && !narrow_[static_cast<size_t>(lower)];
            const bool intoUpper = lower + 1 < n_ && !narrow_[static_cast<size_t>(lower + 1)];
            std::array<T, lanes> below {}, above {};
            if (intoLower) std::copy_n(out + Lanes * lower, Lanes, below.data());
            if (intoUpper) std::copy_n(out + Lanes * (lower + 1), Lanes, above.data());
            for (; k < numBins_ && binBand_[static_cast<size_t>(k)] == lower; ++k)
            {
                const double w = binWeight_[static_cast<size_t>(k)], rest = 1.0 - w;
                const T* power = binPower + Lanes * k;
                for (size_t j = 0; j < lanes; ++j)
                {
                    const double p = static_cast<double>(power[j]);
                    below[j] += static_cast<T>(w * p);
                    above[j] += static_cast<T>(rest * p);
                }
            }
            if (intoLower) std::copy_n(below.data(), Lanes, out + Lanes * lower);
            if (intoUpper) std::copy_n(above.data(), Lanes, out + Lanes * (lower + 1));
        }
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            if (!narrow_[i]) continue;
            const double pos = std::clamp(centre_[i] / binHz_, 0.0, static_cast<double>(numBins_ - 1));
            const int k = std::min(static_cast<int>(pos), numBins_ - 2);
            const double f = pos - k;
            const T* at = binPower + Lanes * k;
            for (int j = 0; j < Lanes; ++j)
                out[Lanes * b + j] = static_cast<T>(widthBins_[i] * ((1.0 - f) * static_cast<double>(at[j]) + f * static_cast<double>(at[j + Lanes])));
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
            out[b] = static_cast<T>(std::pow(std::max(0.0, static_cast<double>(bandPower[b])) + th, a) - thresholdLoudness_[i]);
        }
    }

    /** @return The total loudness: the specific loudness integrated over Bark (trapezoids). */
    [[nodiscard]] T totalLoudness(const T* specific) const noexcept
    {
        double s = 0.0;
        if (!uniform_)
        {
            for (int b = 1; b < n_; ++b)
                s += 0.5 * (static_cast<double>(specific[b - 1]) + static_cast<double>(specific[b]))
                   * (bark_[static_cast<size_t>(b)] - bark_[static_cast<size_t>(b - 1)]);
            return static_cast<T>(s);
        }
        for (int b = 1; b < n_; ++b) s += 0.5 * (static_cast<double>(specific[b - 1]) + static_cast<double>(specific[b]));
        return static_cast<T>(s * dz_);
    }

private:
    void resizeBands()
    {
        centre_.resize(static_cast<size_t>(n_));
        bark_.resize(static_cast<size_t>(n_));
        exponent_.resize(static_cast<size_t>(n_));
        threshold_.resize(static_cast<size_t>(n_));
        thresholdLoudness_.resize(static_cast<size_t>(n_));
        narrow_.resize(static_cast<size_t>(n_));
        widthBins_.resize(static_cast<size_t>(n_));
        widthBark_.resize(static_cast<size_t>(n_));
    }

    /**
     * @brief The bins of a transform over bands at any centres: each bin
     *        feeds the two bands around it, with weights that are linear in
     *        Bark between their centres.
     */
    void mapBins(int fftSize)
    {
        uniform_ = false;
        numBins_ = fftSize / 2 + 1;
        binHz_ = fs_ / static_cast<double>(fftSize);
        const auto last = static_cast<size_t>(n_ - 1);
        for (int b = 0; b < n_; ++b)
        {
            const auto i = static_cast<size_t>(b);
            const double below = b > 0 ? 0.5 * (bark_[i - 1] + bark_[i]) : bark_[i] - 0.5 * (bark_[i + 1] - bark_[i]);
            const double above = b < n_ - 1 ? 0.5 * (bark_[i] + bark_[i + 1]) : bark_[i] + 0.5 * (bark_[i] - bark_[i - 1]);
            // A band one spacing wide, in bins; its share of the Bark axis
            // stops at the first and at the last centre.
            widthBins_[i] = (AuditoryScale::barkToHz(above) - AuditoryScale::barkToHz(below)) / binHz_;
            narrow_[i] = widthBins_[i] < 2.0;
            widthBark_[i] = std::min(above, bark_[last]) - std::max(below, bark_[0]);
        }
        binBand_.assign(static_cast<size_t>(numBins_), -2);
        binWeight_.assign(static_cast<size_t>(numBins_), 0.0);
        int lower = -1;   // the band at or under the bin
        for (int k = 1; k < numBins_; ++k)
        {
            const double z = AuditoryScale::hzToBark(binHz_ * k);
            while (lower + 1 < n_ && bark_[static_cast<size_t>(lower + 1)] <= z) ++lower;
            double part = 0.0;   // of the way from `lower` to the band above it
            if (lower < 0)
            {
                part = 1.0 - (bark_[0] - z) / (bark_[1] - bark_[0]);
                if (part < 0.0) continue;
            }
            else if (lower >= n_ - 1)
            {
                part = (z - bark_[last]) / (bark_[last] - bark_[last - 1]);
                if (part > 1.0) continue;
            }
            else
            {
                part = (z - bark_[static_cast<size_t>(lower)]) / (bark_[static_cast<size_t>(lower + 1)] - bark_[static_cast<size_t>(lower)]);
            }
            binBand_[static_cast<size_t>(k)] = lower;
            binWeight_[static_cast<size_t>(k)] = 1.0 - part;   // to `lower`; the rest to lower + 1
        }
    }

    double fs_ = 48000.0, binHz_ = 23.4375, z0_ = 0.0, dz_ = 0.3, refSpl_ = 100.0;
    int numBins_ = 0, n_ = 0;
    std::vector<double> centre_, bark_, exponent_, threshold_, thresholdLoudness_, widthBins_, widthBark_;
    std::vector<bool> narrow_;
    bool uniform_ = true;
    std::vector<int> binBand_;
    std::vector<double> binWeight_;
};

} // namespace dspark
