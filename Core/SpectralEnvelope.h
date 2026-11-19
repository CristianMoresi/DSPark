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
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
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
     */
    void prepare(int numBins, double widthOctaves = 1.0 / 3.0)
    {
        n_ = std::max(2, numBins);
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
        // 2^(-w/2)) bins, at least one.
        const double span = std::pow(2.0, 0.5 * width_) - std::pow(2.0, -0.5 * width_);
        for (int k = 0; k < n_; ++k)
        {
            const double bins = std::max(1.0, span * k);
            coeff_[static_cast<size_t>(k)] = static_cast<T>(bins / (bins + 2.0));
        }
    }

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
        for (int k = 1; k < n_; ++k) out[k] += coeff_[static_cast<size_t>(k)] * (out[k - 1] - out[k]);
        for (int k = n_ - 2; k >= 0; --k) out[k] += coeff_[static_cast<size_t>(k)] * (out[k + 1] - out[k]);
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

private:
    int n_ = 0;
    double width_ = 1.0 / 3.0;
    std::vector<T> coeff_;
};

} // namespace dspark
