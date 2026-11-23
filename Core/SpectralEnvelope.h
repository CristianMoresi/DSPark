// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralEnvelope.h
 * @brief The envelope of a spectrum by a smoothing whose width grows with
 *        frequency, and the resonances as what stands over it.
 *
 * A one-pole runs across the bins forward and then backward (no shift: the
 * two passes are symmetric), and its coefficient at bin k follows a width
 * proportional to the frequency: about `widthOctaves` of an octave, so the
 * envelope keeps the shape of the spectrum at the ear's scale while a line
 * or a narrow peak stands out of it. In the log domain the envelope is a
 * smoothing of decibels (the soft view: a resonance is a ratio); in the
 * linear domain it smooths magnitudes (the hard view: a resonance is an
 * excess of amplitude).
 *
 * Threading: prepare() on the setup thread (allocates); compute() on the
 * owner's thread.
 *
 * SpectralFocus, in this file too, smooths the complex spectrum instead of
 * its power: the analysis window then shortens as frequency rises.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace dspark {

/**
 * @class SpectralEnvelope
 * @brief Envelope of a spectrum, smoothing proportional to frequency.
 *
 * @tparam T Sample type of the spectra (float or double).
 */
template <FloatType T>
class SpectralEnvelope
{
public:
    /**
     * @brief Sets the bins and the width.
     * @param numBins      Bins of the spectra (fftSize / 2 + 1), at least 2.
     * @param widthOctaves Width of the smoothing, 0.05 to 4 octaves.
     * @param minBins      Narrowest smoothing, in bins (1 by default; a
     *                     spectrum padded q times more keeps the same
     *                     width in hertz with q).
     */
    void prepare(int numBins, double widthOctaves = 1.0 / 3.0, double minBins = 1.0)
    {
        n_ = std::max(2, numBins);
        minBins_ = std::isfinite(minBins) ? std::max(1.0, minBins) : 1.0;
        coeff_.assign(static_cast<size_t>(n_), T(0));
        setWidth(widthOctaves);
    }

    /** @brief Width of the smoothing in octaves (0.05 to 4; non-finite values are ignored). */
    void setWidth(double widthOctaves) noexcept
    {
        if (!std::isfinite(widthOctaves)) return;
        width_ = std::clamp(widthOctaves, 0.05, 4.0);
        // A one-pole run both ways has an equivalent width of about
        // 2 c / (1 - c) bins; the width wanted at bin k is k (2^(w/2) -
        // 2^(-w/2)) bins, at least minBins_.
        // Two passes of a smoothing 1 / sqrt(2) as wide have the same
        // equivalent width as one, with far less of a tail.
        const double span = (std::pow(2.0, 0.5 * width_) - std::pow(2.0, -0.5 * width_)) / std::sqrt(static_cast<double>(passes_));
        for (int k = 0; k < n_; ++k)
        {
            double bins = span * k;
            if (taperExponent_ > 0.0 && taperBin_ > 0.0 && k > taperBin_) bins *= std::pow(taperBin_ / k, taperExponent_);
            bins = std::max(minBins_, bins);
            coeff_[static_cast<size_t>(k)] = static_cast<T>(bins / (bins + 2.0));
        }
    }

    /**
     * @brief Passes of the smoothing: 1 (the default) or 2. Two passes keep
     *        the equivalent width and lose the long tails of one: the
     *        envelope follows a wide rise of the spectrum closely while a
     *        narrow line still stands over it.
     */
    void setPasses(int passes) noexcept
    {
        passes_ = passes >= 2 ? 2 : 1;
        setWidth(width_);
    }

    /**
     * @brief Narrows the smoothing above a corner: its width in octaves
     *        falls as (corner / frequency) to the exponent (0, the default,
     *        keeps it constant; 1 makes it constant in hertz above the
     *        corner; at most 2). Non-finite values are ignored.
     * @param cornerBin The corner, in bins.
     * @param exponent  0 to 2.
     */
    void setTaper(double cornerBin, double exponent) noexcept
    {
        if (!std::isfinite(cornerBin) || !std::isfinite(exponent)) return;
        taperBin_ = std::max(0.0, cornerBin);
        taperExponent_ = std::clamp(exponent, 0.0, 2.0);
        setWidth(width_);
    }

    [[nodiscard]] int getPasses() const noexcept { return passes_; }

    [[nodiscard]] double getWidth() const noexcept { return width_; }
    [[nodiscard]] int getNumBins() const noexcept { return n_; }

    /**
     * @brief The envelope of a spectrum.
     * @param in  numBins values (magnitudes, or decibels when logDomain).
     * @param out numBins values (may be `in`).
     */
    void compute(const T* in, T* out) const noexcept
    {
        if (out != in) std::copy(in, in + n_, out);
        computeInterleaved<1>(out);
    }

    /**
     * @brief The envelopes of several spectra at once, in place: each what
     *        compute() gives for its own. The recursion waits on itself at
     *        every bin, so a few run in the time of one.
     * @tparam Lanes How many spectra (1 to 8).
     * @param values numBins groups of `Lanes` values, one of each spectrum.
     */
    template <int Lanes>
    void computeInterleaved(T* values) const noexcept
    {
        static_assert(Lanes >= 1 && Lanes <= 8);
        if (n_ < 2) return;
        const T* coeff = coeff_.data();
        for (int pass = 0; pass < passes_; ++pass)
        {
            std::array<T, static_cast<size_t>(Lanes)> y;
            std::copy_n(values, Lanes, y.data());
            for (int k = 1; k < n_; ++k)
            {
                T* v = values + Lanes * k;
                for (int j = 0; j < Lanes; ++j)
                {
                    const auto u = static_cast<size_t>(j);
                    y[u] = v[j] + coeff[k] * (y[u] - v[j]);
                    v[j] = y[u];
                }
            }
            for (int k = n_ - 2; k >= 0; --k)
            {
                T* v = values + Lanes * k;
                for (int j = 0; j < Lanes; ++j)
                {
                    const auto u = static_cast<size_t>(j);
                    y[u] = v[j] + coeff[k] * (y[u] - v[j]);
                    v[j] = y[u];
                }
            }
        }
    }

    /**
     * @brief The resonances, never negative: given decibels, the spectrum
     *        over its envelope in decibels (soft); given magnitudes, the
     *        excess of amplitude over it (hard).
     * @param in  numBins values.
     * @param out numBins values of the excess (must not be `in`).
     */
    void resonances(const T* in, T* out) const noexcept
    {
        compute(in, out);
        for (int k = 0; k < n_; ++k) out[k] = std::max(T(0), in[k] - out[k]);
    }

    /**
     * @brief The spectrum less its envelope, with its sign (what stands over
     *        the envelope is positive, what lies under it negative).
     * @param in  numBins values.
     * @param out numBins values (must not be `in`).
     */
    void excess(const T* in, T* out) const noexcept
    {
        compute(in, out);
        for (int k = 0; k < n_; ++k) out[k] = in[k] - out[k];
    }

private:
    int n_ = 0, passes_ = 1;
    double width_ = 1.0 / 3.0, minBins_ = 1.0, taperBin_ = 0.0, taperExponent_ = 0.0;
    std::vector<T> coeff_;
};

/**
 * @class SpectralFocus
 * @brief A frame's spectrum read through a window that shortens as frequency
 *        rises, from the one transform of the whole window.
 *
 * Smoothing a complex spectrum across frequency is the same as multiplying
 * the frame by a weight in time: the wider the smoothing, the narrower the
 * weight. With the time origin moved to the window's peak, a smoothing whose
 * width grows with frequency (as SpectralEnvelope's, a one-pole run both
 * ways) weighs the low bins by nearly the whole window and the high ones by
 * a short stretch around its peak: the low end keeps its resolution in
 * frequency and the top gains it in time, where the ear has it. The weight of
 * a smoothing of coefficient c is (1 - c)^2 / (1 - 2 c cos(theta) + c^2) at
 * the angle theta = 2 pi (t - peak) / transform.
 *
 * power() gives the power of each bin scaled so that a noise, or a tone
 * summed over its lobe, reads what it reads through the whole window (each
 * bin by the energy of the window over the energy of its own shorter one).
 *
 * Threading: prepare() on the setup thread (allocates); smooth() and power()
 * on the owner's thread, with no allocation.
 *
 * @tparam T Sample type of the powers (float or double).
 */
template <FloatType T>
class SpectralFocus
{
public:
    /**
     * @brief Sets the window, the transform and the width.
     * @param window       The analysis window the frames are made with.
     * @param windowSize   Its length, at least 2.
     * @param transform    The transform's size (windowSize or more, for a
     *                     zero-padded frame that starts at the window's start).
     * @param widthOctaves Width of the smoothing, 0.05 to 4 octaves.
     * @param minBins      Narrowest smoothing, in bins (1 by default).
     */
    void prepare(const double* window, int windowSize, int transform, double widthOctaves = 1.0 / 3.0, double minBins = 1.0)
    {
        n_ = std::max(2, windowSize);
        m_ = std::max(n_, transform);
        bins_ = m_ / 2 + 1;
        int peak = 0;
        double energy = 0.0;
        for (int i = 0; i < n_; ++i)
        {
            if (window[i] > window[peak]) peak = i;
            energy += window[i] * window[i];
        }
        peak_ = peak;
        const double width = std::clamp(std::isfinite(widthOctaves) ? widthOctaves : 1.0 / 3.0, 0.05, 4.0);
        const double least = std::isfinite(minBins) ? std::max(1.0, minBins) : 1.0;
        const double span = std::pow(2.0, 0.5 * width) - std::pow(2.0, -0.5 * width);
        coeff_.resize(static_cast<size_t>(bins_));
        turnRe_.resize(static_cast<size_t>(bins_));
        turnIm_.resize(static_cast<size_t>(bins_));
        gain_.resize(static_cast<size_t>(bins_));
        re_.assign(static_cast<size_t>(bins_), 0.0);
        im_.assign(static_cast<size_t>(bins_), 0.0);
        const double step = 2.0 * 3.141592653589793 / static_cast<double>(m_);
        for (int k = 0; k < bins_; ++k)
        {
            const double bins = std::max(least, span * k);
            coeff_[static_cast<size_t>(k)] = bins / (bins + 2.0);
            // The origin of time moved to the window's peak: a turn of the phase by bin.
            const double angle = step * static_cast<double>((static_cast<long long>(k) * peak) % m_);
            turnRe_[static_cast<size_t>(k)] = std::cos(angle);
            turnIm_[static_cast<size_t>(k)] = std::sin(angle);
        }
        // The energy of the whole window over that of each bin's own: found at
        // a few coefficients (it moves slowly with the bin) and read between them.
        std::vector<double> cosine(static_cast<size_t>(n_));
        for (int i = 0; i < n_; ++i) cosine[static_cast<size_t>(i)] = std::cos(step * static_cast<double>(i - peak));
        const auto gainOf = [&](double c) {
            double own = 0.0;
            const double top = (1.0 - c) * (1.0 - c), flat = 1.0 + c * c;
            for (int i = 0; i < n_; ++i)
            {
                const double w = window[i] * top / (flat - 2.0 * c * cosine[static_cast<size_t>(i)]);
                own += w * w;
            }
            return own > 0.0 ? energy / own : 1.0;
        };
        // The knots sit evenly in the logarithm of the smoothing's width, where
        // the logarithm of the gain is close to a line.
        constexpr int kKnots = 48;
        const auto widthOf = [](double c) { return 2.0 * c / (1.0 - c); };   // the smoothing's width in bins
        const double wLow = std::log(widthOf(coeff_.front())), wHigh = std::log(widthOf(coeff_.back()));
        std::array<double, kKnots> knot {};
        for (int j = 0; j < kKnots; ++j)
        {
            const double bins = std::exp(wLow + (wHigh - wLow) * j / (kKnots - 1.0));
            knot[static_cast<size_t>(j)] = std::log(gainOf(bins / (bins + 2.0)));
        }
        for (int k = 0; k < bins_; ++k)
        {
            const double x = wHigh > wLow ? (std::log(widthOf(coeff_[static_cast<size_t>(k)])) - wLow) / (wHigh - wLow) * (kKnots - 1.0) : 0.0;
            const int j = std::clamp(static_cast<int>(x), 0, kKnots - 2);
            const double t = x - j;
            gain_[static_cast<size_t>(k)] = std::exp(knot[static_cast<size_t>(j)] + t * (knot[static_cast<size_t>(j + 1)] - knot[static_cast<size_t>(j)]));
        }
    }

    [[nodiscard]] int getNumBins() const noexcept { return bins_; }
    /** @return The scale of a bin's power (see power()). */
    [[nodiscard]] double getGain(int bin) const noexcept { return gain_[static_cast<size_t>(std::clamp(bin, 0, bins_ - 1))]; }

    /**
     * @brief The smoothed complex spectrum of a frame.
     * @param spectrum The frame's transform, interleaved (real, imaginary), getNumBins() pairs.
     * @param re       getNumBins() values: the real parts.
     * @param im       getNumBins() values: the imaginary parts.
     */
    void smooth(const double* spectrum, double* re, double* im) const noexcept
    {
        for (int k = 0; k < bins_; ++k)
        {
            const auto i = static_cast<size_t>(k);
            const double xr = spectrum[2 * k], xi = spectrum[2 * k + 1];
            re[k] = xr * turnRe_[i] - xi * turnIm_[i];
            im[k] = xr * turnIm_[i] + xi * turnRe_[i];
        }
        for (int k = 1; k < bins_; ++k)
        {
            const double c = coeff_[static_cast<size_t>(k)];
            re[k] += c * (re[k - 1] - re[k]);
            im[k] += c * (im[k - 1] - im[k]);
        }
        for (int k = bins_ - 2; k >= 0; --k)
        {
            const double c = coeff_[static_cast<size_t>(k)];
            re[k] += c * (re[k + 1] - re[k]);
            im[k] += c * (im[k + 1] - im[k]);
        }
    }

    /**
     * @brief The power of each bin through its own window, scaled to read as
     *        through the whole one.
     * @param spectrum The frame's transform, interleaved, getNumBins() pairs.
     * @param out      getNumBins() powers.
     */
    void power(const double* spectrum, T* out) noexcept
    {
        smooth(spectrum, re_.data(), im_.data());
        for (int k = 0; k < bins_; ++k)
        {
            const auto i = static_cast<size_t>(k);
            out[k] = static_cast<T>(gain_[i] * (re_[i] * re_[i] + im_[i] * im_[i]));
        }
    }

private:
    int n_ = 0, m_ = 0, bins_ = 0, peak_ = 0;
    std::vector<double> coeff_, turnRe_, turnIm_, gain_, re_, im_;
};

} // namespace dspark
