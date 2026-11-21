// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file GraphicBank.h
 * @brief A graphic equalizer whose bands do not interact: the gains asked of
 *        the bands become the gains of overlapping sections by least squares,
 *        so the summed response is the one asked for.
 *
 * Bells wide enough to join into a smooth curve overlap: set every band to
 * +6 dB and the middle of the bank reads far more, because each band adds
 * the skirts of its neighbours. The bank solves for the section gains
 * instead: on a grid of the band centres and the points half way between
 * them (where the target is the mean of its neighbours, a triangular
 * objective between centres; the centres weigh ten times more), it takes the
 * response in decibels of each section at a gain of one decibel and finds,
 * by weighted, regularized least squares,
 * the section gains whose sum meets the target; the matrix of that solution
 * is computed once in prepare(), so a change of the gains costs one product
 * by it. The sections are bells, the first a low shelf and the last a high
 * shelf, of the published cookbook form; a bell's coefficients follow its
 * gain in closed form, without a trigonometric function (the numerator
 * scales with the gain and the denominator is corrected with it), and every
 * update is ramped linearly over the samples the caller gives. A refining
 * step reads the real summed response on the grid and corrects the gains
 * for what the decibels of a bell are not (a bell's shape widens a little
 * with its gain), and is kept only when it brings the response nearer: a
 * shape the sections cannot follow (a zigzag from band to band, finer than
 * their width) keeps the regularized first solution, its section gains
 * bounded (30 dB). A caller whose gains move a little at a time can refine
 * one call in several and ask the others to carry what the last refinement
 * added (kCarryRefinement): the response then does not step back and forth
 * between the refined solution and the plain one.
 *
 * One channel runs on a LaneCascade (the cascade on the lanes of a vector,
 * latency Lanes - 1 samples); two or more run on a BiquadCascade, which
 * already fills a vector with a pair of channels and has no latency (on two
 * channels it costs about 0.55 of the lanes).
 *
 * Threading: prepare() on the setup thread (allocates); the rest on the
 * processing thread (owner-managed).
 *
 * Dependencies: AudioBuffer.h, AuditoryBands.h, Biquad.h, BiquadCascade.h,
 * DspMath.h, LaneCascade.h.
 */

#include "AudioBuffer.h"
#include "AuditoryBands.h"
#include "Biquad.h"
#include "BiquadCascade.h"
#include "DspMath.h"
#include "LaneCascade.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace dspark {

/**
 * @class GraphicBank
 * @brief A bank of overlapping sections driven by least squares.
 *
 * @tparam T     Sample type (float or double).
 * @tparam Lanes Lanes of the cascade (latency Lanes - 1).
 */
template <FloatType T, int Lanes = 8>
class GraphicBank
{
public:
    static constexpr int kMaxBands = 256;
    /** @brief setGains(): no refining step; the correction of the last refining call is added instead. */
    static constexpr int kCarryRefinement = -1;

    /**
     * @brief Lays the bank out (allocates).
     * @param sampleRate Rate (invalid values are ignored).
     * @param centresHz  Band centres, rising (clamped under 0.49 of the rate).
     * @param bands      Number of bands, 2 to 256.
     * @param channels   Channels to run.
     * @param qScale     The bells' Q is the centre over the critical bandwidth
     *                   there, times this (0.25 to 8).
     */
    void prepare(double sampleRate, const double* centresHz, int bands, int channels, double qScale = 1.0)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0) || centresHz == nullptr || bands < 2) return;
        fs_ = sampleRate;
        nb_ = std::min(bands, kMaxBands);
        const double qs = std::clamp(qScale, 0.25, 8.0);
        cw_.resize(static_cast<size_t>(nb_));
        alpha_.resize(static_cast<size_t>(nb_));
        centre_.resize(static_cast<size_t>(nb_));
        for (int s = 0; s < nb_; ++s)
        {
            const double f = std::clamp(centresHz[s], 10.0, 0.49 * fs_);
            centre_[static_cast<size_t>(s)] = f;
            const double w = 2.0 * 3.141592653589793 * f / fs_;
            cw_[static_cast<size_t>(s)] = std::cos(w);
            const double q = s == 0 || s == nb_ - 1 ? 0.0 : qs * f / AuditoryScale::criticalBandwidth(f);
            alpha_[static_cast<size_t>(s)] = (s == 0 || s == nb_ - 1) ? std::sin(w) * 0.7071067811865476
                                                                      : std::sin(w) / (2.0 * q);
        }
        // The grid: the centres and the points half way between them (in log).
        grid_ = 2 * nb_ - 1;
        z1_.resize(static_cast<size_t>(grid_));
        for (int g = 0; g < grid_; ++g)
        {
            const double f = g % 2 == 0 ? centre_[static_cast<size_t>(g / 2)]
                                        : std::sqrt(centre_[static_cast<size_t>(g / 2)] * centre_[static_cast<size_t>(g / 2 + 1)]);
            z1_[static_cast<size_t>(g)] = std::polar(1.0, -2.0 * 3.141592653589793 * f / fs_);
        }
        // Response in dB of each section at +1 dB on the grid, and the
        // regularized least-squares solution P = (A'A + l I)^-1 A'.
        std::vector<double> a(static_cast<size_t>(grid_ * nb_));
        for (int s = 0; s < nb_; ++s)
        {
            const BiquadCoeffs c = sectionAt(s, 1.0);
            for (int g = 0; g < grid_; ++g) a[static_cast<size_t>(g * nb_ + s)] = dbAt(c, g);
        }
        std::vector<double> m(static_cast<size_t>(nb_ * nb_), 0.0);
        for (int i = 0; i < nb_; ++i)
            for (int j = 0; j < nb_; ++j)
            {
                double acc = 0.0;
                for (int g = 0; g < grid_; ++g) acc += weightOf(g) * a[static_cast<size_t>(g * nb_ + i)] * a[static_cast<size_t>(g * nb_ + j)];
                m[static_cast<size_t>(i * nb_ + j)] = acc;
            }
        double trace = 0.0;
        for (int i = 0; i < nb_; ++i) trace += m[static_cast<size_t>(i * nb_ + i)];
        const double lambda = kRegularization * trace / nb_;
        for (int i = 0; i < nb_; ++i) m[static_cast<size_t>(i * nb_ + i)] += lambda;
        // Cholesky: m = L L'.
        for (int j = 0; j < nb_; ++j)
        {
            double d = m[static_cast<size_t>(j * nb_ + j)];
            for (int k = 0; k < j; ++k) d -= m[static_cast<size_t>(j * nb_ + k)] * m[static_cast<size_t>(j * nb_ + k)];
            const double l = std::sqrt(std::max(d, 1e-30));
            m[static_cast<size_t>(j * nb_ + j)] = l;
            for (int i = j + 1; i < nb_; ++i)
            {
                double v = m[static_cast<size_t>(i * nb_ + j)];
                for (int k = 0; k < j; ++k) v -= m[static_cast<size_t>(i * nb_ + k)] * m[static_cast<size_t>(j * nb_ + k)];
                m[static_cast<size_t>(i * nb_ + j)] = v / l;
            }
        }
        p_.assign(static_cast<size_t>(nb_ * grid_), 0.0);
        std::vector<double> col(static_cast<size_t>(nb_));
        for (int g = 0; g < grid_; ++g)
        {
            for (int i = 0; i < nb_; ++i) col[static_cast<size_t>(i)] = weightOf(g) * a[static_cast<size_t>(g * nb_ + i)];
            for (int i = 0; i < nb_; ++i)   // forward: L y = col
            {
                double v = col[static_cast<size_t>(i)];
                for (int k = 0; k < i; ++k) v -= m[static_cast<size_t>(i * nb_ + k)] * col[static_cast<size_t>(k)];
                col[static_cast<size_t>(i)] = v / m[static_cast<size_t>(i * nb_ + i)];
            }
            for (int i = nb_ - 1; i >= 0; --i)   // back: L' x = y
            {
                double v = col[static_cast<size_t>(i)];
                for (int k = i + 1; k < nb_; ++k) v -= m[static_cast<size_t>(k * nb_ + i)] * col[static_cast<size_t>(k)];
                col[static_cast<size_t>(i)] = v / m[static_cast<size_t>(i * nb_ + i)];
            }
            for (int i = 0; i < nb_; ++i) p_[static_cast<size_t>(i * grid_ + g)] = col[static_cast<size_t>(i)];
        }
        target_.assign(static_cast<size_t>(grid_), 0.0);
        response_.assign(static_cast<size_t>(grid_), 0.0);
        gains_.assign(static_cast<size_t>(nb_), 0.0);
        candidate_.assign(static_cast<size_t>(nb_), 0.0);
        correction_.assign(static_cast<size_t>(nb_), 0.0);
        useLanes_ = channels <= 1;
        if (useLanes_)
        {
            lanes_.prepare(nb_, 1);
            for (int s = 0; s < nb_; ++s) lanes_.setCoeffs(s, sectionAt(s, 0.0));
        }
        else
        {
            serial_.setNumSections(nb_);
            for (int s = 0; s < nb_; ++s) serial_.setCoeffs(s, sectionAt(s, 0.0));
            serial_.reset();
        }
    }

    [[nodiscard]] int getNumBands() const noexcept { return nb_; }
    [[nodiscard]] double getCentre(int band) const noexcept { return centre_[static_cast<size_t>(std::clamp(band, 0, nb_ - 1))]; }
    /** @return The latency: Lanes - 1 samples on one channel, none on more. */
    [[nodiscard]] int getLatency() const noexcept { return useLanes_ ? Lanes - 1 : 0; }
    /** @return A section's gain in dB (not the band's: see the file comment). */
    [[nodiscard]] double getSectionGain(int section) const noexcept { return gains_[static_cast<size_t>(std::clamp(section, 0, nb_ - 1))]; }

    /**
     * @brief The gains asked of the bands.
     * @param gainsDb     getNumBands() gains in dB (clamped to +-24).
     * @param rampSamples Samples over which the sections glide to them.
     * @param refine      Refining steps (0 to 3) on the real response, or
     *                    kCarryRefinement.
     */
    void setGains(const T* gainsDb, int rampSamples, int refine = 1) noexcept
    {
        if (nb_ < 2) return;
        for (int g = 0; g < grid_; ++g)
        {
            const double v = g % 2 == 0 ? static_cast<double>(gainsDb[g / 2])
                                        : 0.5 * (static_cast<double>(gainsDb[g / 2]) + static_cast<double>(gainsDb[g / 2 + 1]));
            target_[static_cast<size_t>(g)] = std::isfinite(v) ? std::clamp(v, -24.0, 24.0) : 0.0;
        }
        solve(target_.data(), gains_.data(), false);
        for (auto& g : gains_) g = std::clamp(g, -kMaxSectionDb, kMaxSectionDb);
        if (refine == kCarryRefinement)
        {
            for (int s = 0; s < nb_; ++s)
                gains_[static_cast<size_t>(s)] = std::clamp(gains_[static_cast<size_t>(s)] + correction_[static_cast<size_t>(s)], -kMaxSectionDb, kMaxSectionDb);
        }
        else
        {
            correction_ = gains_;   // the plain solution, until the refinement is known
        }
        // Refining: a step on what the real response misses, kept only if it
        // brings the response nearer (a shape the sections cannot follow,
        // such as a zigzag from band to band, keeps the first solution).
        const int steps = std::clamp(refine, 0, 3);
        if (steps > 0)
        {
            double err = missing(gains_.data());
            for (int r = 0; r < steps; ++r)
            {
                candidate_ = gains_;
                solve(response_.data(), candidate_.data(), true);
                for (auto& g : candidate_) g = std::clamp(g, -kMaxSectionDb, kMaxSectionDb);
                const double next = missing(candidate_.data());
                if (!(next < err)) break;
                gains_ = candidate_;
                err = next;
            }
        }
        // What the refinement added to the plain solution (nothing without it).
        if (refine != kCarryRefinement)
            for (int s = 0; s < nb_; ++s)
                correction_[static_cast<size_t>(s)] = gains_[static_cast<size_t>(s)] - correction_[static_cast<size_t>(s)];
        if (useLanes_)
        {
            for (int s = 0; s < nb_; ++s)
                lanes_.setTargetCoeffs(s, sectionAt(s, gains_[static_cast<size_t>(s)]));
            lanes_.startRamp(rampSamples);
        }
        else
        {
            for (int s = 0; s < nb_; ++s)
                serial_.rampCoeffs(s, sectionAt(s, gains_[static_cast<size_t>(s)]), rampSamples);
        }
    }

    /** @return The bank's response at a frequency, in dB, with the gains last set. */
    [[nodiscard]] double responseDb(double hz) const noexcept
    {
        const std::complex<double> z = std::polar(1.0, -2.0 * 3.141592653589793 * hz / fs_);
        double db = 0.0;
        for (int s = 0; s < nb_; ++s) db += dbOf(sectionAt(s, gains_[static_cast<size_t>(s)]), z);
        return db;
    }

    /** @brief Clears the history (the gains stay). */
    void reset() noexcept { if (useLanes_) lanes_.reset(); else serial_.reset(); }

    /** @brief Runs a block in place. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (useLanes_) lanes_.processBlock(buffer);
        else serial_.processBlock(buffer);
        // A sample that is not a number would stay in the sections for good:
        // it leaves as silence and the sections start again from rest.
        bool broken = false;
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            T* d = buffer.getChannel(ch);
            for (int i = 0; i < buffer.getNumSamples(); ++i)
                if (!std::isfinite(d[i])) { d[i] = T(0); broken = true; }
        }
        if (broken) reset();
    }

private:
    /** @brief Weight of a grid point in the fit: the centres carry the gains asked, the points between them follow. */
    [[nodiscard]] static double weightOf(int g) noexcept { return g % 2 == 0 ? 1.0 : kMidWeight; }
    static constexpr double kMidWeight = 0.1;
    static constexpr double kRegularization = 1e-2;   ///< Tikhonov weight, against the mean of A'WA's diagonal.
    static constexpr double kMaxSectionDb = 30.0;     ///< Largest gain of a section.

    /** @brief A section's coefficients at a gain, in closed form from its precomputed terms. */
    [[nodiscard]] BiquadCoeffs sectionAt(int s, double gainDb) const noexcept
    {
        const double A = std::pow(10.0, gainDb / 40.0);
        const double cw = cw_[static_cast<size_t>(s)], al = alpha_[static_cast<size_t>(s)];
        BiquadCoeffs c;
        if (s == 0 || s == nb_ - 1)
        {
            const double t = 2.0 * std::sqrt(A) * al;
            const double sign = s == 0 ? 1.0 : -1.0;   // low shelf / high shelf
            const double a0 = (A + 1.0) + sign * (A - 1.0) * cw + t;
            c.b0 = A * ((A + 1.0) - sign * (A - 1.0) * cw + t) / a0;
            c.b1 = sign * 2.0 * A * ((A - 1.0) - sign * (A + 1.0) * cw) / a0;
            c.b2 = A * ((A + 1.0) - sign * (A - 1.0) * cw - t) / a0;
            c.a1 = -sign * 2.0 * ((A - 1.0) + sign * (A + 1.0) * cw) / a0;
            c.a2 = ((A + 1.0) + sign * (A - 1.0) * cw - t) / a0;
            return c;
        }
        const double ia = A / (A + al);   // 1 / a0, a0 = 1 + alpha / A
        c.b0 = (1.0 + al * A) * ia;
        c.b1 = -2.0 * cw * ia;
        c.b2 = (1.0 - al * A) * ia;
        c.a1 = c.b1;
        c.a2 = (A - al) / (A + al);
        return c;
    }

    [[nodiscard]] static double dbOf(const BiquadCoeffs& c, std::complex<double> z1) noexcept
    {
        const std::complex<double> z2 = z1 * z1;
        const std::complex<double> num = c.b0 + c.b1 * z1 + c.b2 * z2;
        const std::complex<double> den = 1.0 + c.a1 * z1 + c.a2 * z2;
        return 10.0 * std::log10(std::max(std::norm(num), 1e-30) / std::max(std::norm(den), 1e-30));
    }

    [[nodiscard]] double dbAt(const BiquadCoeffs& c, int g) const noexcept { return dbOf(c, z1_[static_cast<size_t>(g)]); }

    /**
     * @brief The weighted squared error of the real response of a set of
     *        section gains on the grid; leaves what it misses in response_.
     */
    double missing(const double* gains) noexcept
    {
        for (int g = 0; g < grid_; ++g) response_[static_cast<size_t>(g)] = 0.0;
        for (int s = 0; s < nb_; ++s)
        {
            const BiquadCoeffs c = sectionAt(s, gains[s]);
            for (int g = 0; g < grid_; ++g) response_[static_cast<size_t>(g)] += dbAt(c, g);
        }
        double err = 0.0;
        for (int g = 0; g < grid_; ++g)
        {
            const double d = target_[static_cast<size_t>(g)] - response_[static_cast<size_t>(g)];
            response_[static_cast<size_t>(g)] = d;
            err += weightOf(g) * d * d;
        }
        return err;
    }

    /** @brief x (+)= P v. */
    void solve(const double* v, double* x, bool add) const noexcept
    {
        for (int s = 0; s < nb_; ++s)
        {
            double acc = 0.0;
            const double* row = p_.data() + static_cast<size_t>(s * grid_);
            for (int g = 0; g < grid_; ++g) acc += row[g] * v[g];
            x[s] = add ? x[s] + acc : acc;
        }
    }

    double fs_ = 48000.0;
    int nb_ = 0, grid_ = 0;
    std::vector<double> cw_, alpha_, centre_, p_, target_, response_, gains_, candidate_, correction_;
    std::vector<std::complex<double>> z1_;
    bool useLanes_ = true;
    LaneCascade<T, Lanes> lanes_;
    BiquadCascade<T, kMaxBands, 8> serial_;
};

} // namespace dspark
