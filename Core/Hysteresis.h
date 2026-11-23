// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Hysteresis.h
 * @brief Jiles-Atherton magnetic hysteresis with an implicit NR solver.
 *
 * The physical core shared by tape and transformer modelling: magnetization
 * M responds to the applied field H with saturation, memory (remanence) and
 * rate-dependent loop losses, following the Jiles-Atherton model
 * (Jiles & Atherton 1986; audio-rate treatment after Chowdhury, DAFx-19):
 *
 *   M_an(Q) = Ms * L(Q),   L(x) = coth(x) - 1/x,   Q = (H + alpha*M) / a
 *
 *   phi1  = (1-c) * dltM * (M_an - M) / [ (1-c)*dlt*k - alpha*(M_an - M) ]
 *           with dlt = sign(dH/dt), dltM the reversal gate
 *
 *   dM/dt = [ phi1 + c*(Ms/a)*L'(Q) ] / [ 1 - c*alpha*(Ms/a)*L'(Q) ] * dH/dt
 *
 * The gate dltM zeroes the irreversible term whenever magnetization would
 * move unphysically against the drive direction at reversal points.
 *
 * Integration uses the alpha transform (a damped trapezoid) for BOTH the field
 * derivative and the magnetization integral:
 *
 *   dH_n = (1+a) fs (H_n - H_{n-1}) - a dH_{n-1}
 *   M_n  = M_{n-1} + T/(1+a) * (W_n + a W_{n-1}),   a = 0.9
 *
 * The pair is exact on the linear part (the derivative and the integral
 * cancel), and the derivative's pole sits at z = -a instead of on the unit
 * circle. With the plain trapezoid (a = 1) any transient left a permanent
 * alternating dH/dt that erased the loop's remanence (a 0.001*a field tick
 * erased all of it within 50 ms; now 1.5%), kept a -7.5 dB error after any
 * abrupt onset, and at 1x in TapeMachine could pin the core silent. Solved
 * per sample by Newton-Raphson on M_n, seeded by an explicit Euler predictor
 * (or, for a field with a period of its own, by the same phase of the last
 * two periods: setPredictorPeriod()); converges in 1-3 iterations at audio
 * rates. All internal state is double
 * regardless of T: the model mixes quantities spanning ~9 orders of
 * magnitude.
 *
 * Verified by the test suite on first principles: odd symmetry of the B-H
 * loop, monotone loop area vs the loss parameter k, remanence after field
 * removal, and loop collapse as the reversible fraction c -> 1.
 *
 * Mono processor: hold one instance per channel (the framework pattern for
 * stateful per-channel cores, like Hilbert). Threading is owner-managed:
 * the consuming effect calls setParameters()/processSample() from its own
 * audio thread (the framework's dirty-flag pattern), so no atomics live here.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace dspark {

/**
 * @class Hysteresis
 * @brief Per-channel Jiles-Atherton hysteresis processor (field in, M out).
 *
 * @tparam T Sample type of the audio interface (internal math is double).
 */
template <FloatType T>
class Hysteresis
{
public:
    /** @brief Prepares the integrator for the given sample rate. A
     *  non-positive or non-finite rate is ignored (NaN used to pass the old
     *  `<= 0` test and poison the integrator). */
    void prepare(double sampleRate)
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) return;
        invFs2_ = 0.5 / sampleRate;
        fs2_ = 2.0 * sampleRate;
        reset();
    }

    /** @brief Clears magnetization and integrator state. RT-safe. */
    void reset() noexcept
    {
        m_ = 0.0;
        hPrev_ = 0.0;
        hDotPrev_ = 0.0;
        wPrev_ = 0.0;
        past_.fill(0.0);
        pastAt_ = 0;
        pastCount_ = 0;
    }

    /**
     * @brief Sets the Jiles-Atherton parameters.
     *
     * Defaults follow the audio-tape calibration from the DAFx literature.
     *
     * @param ms    Saturation magnetization (A/m).            [3.5e5]
     * @param a     Anhysteretic shape parameter (A/m).        [2.2e4]
     * @param alpha Inter-domain coupling.                     [1.6e-3]
     * @param k     Coercivity / loop-loss parameter (A/m).    [2.7e4]
     * @param c     Reversible magnetization fraction [0, 1).  [1.7e-1]
     *
     * A call with any non-finite argument is ignored. Range clamping alone
     * does not reject NaN, which would otherwise poison the integrator.
     */
    void setParameters(double ms, double a, double alpha, double k, double c) noexcept
    {
        if (!std::isfinite(ms) || !std::isfinite(a) || !std::isfinite(alpha)
            || !std::isfinite(k) || !std::isfinite(c))
            return;
        ms_    = std::max(ms, 1.0);
        a_     = std::max(a, 1.0);
        alpha_ = std::max(alpha, 0.0);
        k_     = std::max(k, 1.0);
        c_     = std::clamp(c, 0.0, 0.999);
        chi_   = ms_ / a_;
    }

    /**
     * @brief Where the solver stops: a Newton step under this share of Ms.
     *
     * At the default (1e-9) the model is evaluated once more at the point
     * reached. A looser stop takes that last evaluation from the slope of the
     * step just made, and from the second step on stops as well when what is
     * left, read from the rate the steps shrink at (a cube over a square, as
     * Newton's), is under it. A non-finite or non-positive value is ignored.
     *
     * @param relative The share of Ms, 1e-12 to 1e-3.
     */
    void setTolerance(double relative) noexcept
    {
        if (!(relative > 0.0) || !std::isfinite(relative)) return;
        tolerance_ = std::clamp(relative, 1e-12, 1e-3);
        bySlope_ = tolerance_ > 1e-9;
    }

    /**
     * @brief The period of the field in samples, for the predictor (0, the
     *        default, for none). With a field that repeats itself every
     *        `period` samples beside a slower programme (a bias carrier),
     *        the Newton steps start from dM/dt at the same phase of the
     *        last two periods, carried on in a line, in place of the last
     *        one: fewer steps, the same answer within the tolerance.
     * @param period 0, or 1 to kMaxPeriod samples (others are ignored).
     */
    void setPredictorPeriod(int period) noexcept
    {
        if (period < 0 || period > kMaxPeriod || period == period_) return;
        period_ = period;
        pastCount_ = 0;
    }
    /** @return The period the predictor follows (0: none). */
    [[nodiscard]] int getPredictorPeriod() const noexcept { return period_; }

    /** @brief Saturation magnetization Ms (A/m). */
    [[nodiscard]] double getSaturation() const noexcept { return ms_; }

    /**
     * @brief Small-signal susceptibility dM/dH around the demagnetized state.
     *
     * The reversible-branch slope c*chi_an/(1 - c*alpha*chi_an), chi_an = Ms/(3a).
     * Used by consumers (TapeMachine) for exact small-signal gain makeup.
     */
    [[nodiscard]] double getSmallSignalSusceptibility() const noexcept
    {
        const double chiAn = ms_ / (3.0 * a_);
        return c_ * chiAn / (1.0 - c_ * alpha_ * chiAn);
    }

    /**
     * @brief Processes one sample of applied field H, returns magnetization M.
     * @param fieldH Applied magnetic field (A/m scale of the JA parameters).
     */
    [[nodiscard]] T processSample(T fieldH) noexcept
    {
        // Non-finite guard: a single NaN/Inf field would poison m_/hPrev_/
        // hDotPrev_/wPrev_ permanently (the recursive JA state never clears it,
        // and inf-inf on the next sample becomes a sticky NaN). Ignore the bad
        // sample and hold the last magnetization, so a transient upstream
        // glitch cannot kill the core for the rest of the stream.
        if (!std::isfinite(fieldH)) return static_cast<T>(m_);

        const double h = static_cast<double>(fieldH);
        const double hDot = (1.0 + kAlpha) * 0.5 * fs2_ * (h - hPrev_) - kAlpha * hDotPrev_;

        // Explicit predictor, then the matching alpha-transform corrector
        // with Newton steps.
        constexpr double kGain = 1.0 / (1.0 + kAlpha);
        const double step = 2.0 * invFs2_ * kGain;          // T / (1 + a)
        const double rhs = m_ + step * kAlpha * wPrev_;
        double m = m_ + 2.0 * invFs2_ * wPrev_;
        if (period_ > 0 && pastCount_ >= 2 * period_)
        {
            // W at this phase a period and two periods ago, carried on: the
            // corrector's own equation then gives the M it implies.
            const double one = past_[static_cast<std::size_t>((pastAt_ + kRing - period_) & (kRing - 1))];
            const double two = past_[static_cast<std::size_t>((pastAt_ + kRing - 2 * period_) & (kRing - 1))];
            m = rhs + step * (2.0 * one - two);
        }

        double w = 0.0, dwdm = 0.0;
        bool settled = false;
        const double stop = tolerance_ * ms_;
        double before = 0.0;
        for (int it = 0; it < 4; ++it)
        {
            evaluate(m, h, hDot, w, dwdm);
            const double g  = m - rhs - step * w;
            const double gp = 1.0 - step * dwdm;
            const double dm = g / gp;
            m -= dm;
            const double size = std::abs(dm);
            // Loose: the steps shrinking as Newton's do (each under half the
            // last), what is left is about step^3 / last^2.
            const bool shrunk = bySlope_ && it > 0 && size < 0.5 * before && size * size * size < stop * before * before;
            before = size;
            if (size < stop || shrunk)
            {
                if (bySlope_)
                {
                    w -= dwdm * dm;
                    settled = true;
                }
                break;
            }
        }
        if (!settled || std::abs(m) > ms_)
        {
            // Physical clamp: |M| cannot exceed the saturation magnetization.
            m = std::clamp(m, -ms_, ms_);
            evaluate(m, h, hDot, w, dwdm);
        }
        wPrev_ = w;
        if (period_ > 0)
        {
            past_[static_cast<std::size_t>(pastAt_)] = w;
            pastAt_ = (pastAt_ + 1) & (kRing - 1);
            if (pastCount_ < 2 * period_) ++pastCount_;
        }
        hPrev_ = h;
        hDotPrev_ = hDot;
        m_ = m;
        return static_cast<T>(m);
    }

private:
    /** @brief Langevin function and derivatives, numerically safe near 0. */
    static void langevin(double x, double& l, double& lp, double& lpp) noexcept
    {
        const double ax = std::abs(x);
        if (ax < 1e-3)
        {
            // Taylor: L = x/3 - x^3/45, L' = 1/3 - x^2/15, L'' = -2x/15
            const double x2 = x * x;
            l   = x * (1.0 / 3.0 - x2 / 45.0);
            lp  = 1.0 / 3.0 - x2 / 15.0;
            lpp = -2.0 * x / 15.0;
        }
        else if (ax > 30.0)
        {
            // csch^2 underflows to 0 well before this point.
            l   = (x > 0.0 ? 1.0 : -1.0) - 1.0 / x;
            lp  = 1.0 / (x * x);
            lpp = -2.0 / (x * x * x);
        }
        else
        {
            // With u = e^(2x) - 1: coth = 1 + 2/u and csch^2 = 4 (u + 1)/u^2,
            // one exponential and one division (expm1 keeps u whole for a
            // small x, where e^(2x) - 1 would lose its digits).
            const double u = std::expm1(2.0 * x);
            const double r = 1.0 / u;
            const double coth = 1.0 + 2.0 * r;
            const double csch2 = 4.0 * (u + 1.0) * r * r;
            const double ix = 1.0 / x;
            l   = coth - ix;
            lp  = ix * ix - csch2;
            lpp = 2.0 * (csch2 * coth - ix * ix * ix);
        }
    }

    /** @brief Evaluates W = dM/dt and dW/dM at (m, h, hDot). */
    void evaluate(double m, double h, double hDot, double& w, double& dwdm) const noexcept
    {
        if (hDot == 0.0)
        {
            w = 0.0;
            dwdm = 0.0;
            return;
        }

        const double q = (h + alpha_ * m) / a_;
        double l = 0.0, lp = 0.0, lpp = 0.0;
        langevin(q, l, lp, lpp);

        const double mAn = ms_ * l;
        const double dM  = mAn - m;
        const double delta = (hDot > 0.0) ? 1.0 : -1.0;
        const double deltaM = (dM * delta > 0.0) ? 1.0 : 0.0;   // gate reversal

        // Guard the JA denominator singularity (can cross zero at hard drive).
        const double oneMc = 1.0 - c_;
        double d1 = oneMc * delta * k_ - alpha_ * dM;
        const double d1Min = 0.01 * oneMc * k_;
        if (std::abs(d1) < d1Min)
            d1 = (d1 >= 0.0) ? d1Min : -d1Min;

        const double invD1 = 1.0 / d1;
        const double phi1 = oneMc * deltaM * dM * invD1;
        const double cChiLp = c_ * chi_ * lp;

        const double num = phi1 + cChiLp;
        // Guard the reversible-branch denominator too: physical JA parameter
        // sets satisfy c*alpha*Ms/(3a) < 1 so den stays near 1, but an extreme
        // user set makes it sweep through zero as Q moves, spiking dM/dt (and
        // wPrev_, which has no physical clamp) by orders of magnitude for a
        // sample. The M clamp recovers afterwards; this bounds the spike.
        double den = 1.0 - alpha_ * cChiLp;
        if (std::abs(den) < 0.01)
            den = (den >= 0.0) ? 0.01 : -0.01;
        const double invDen = 1.0 / den;
        w = hDot * num * invDen;

        // Analytic dW/dM for the Newton step.
        const double alphaOverA = alpha_ / a_;
        const double dmAn  = ms_ * lp * alphaOverA - 1.0;          // d(M_an - M)/dM
        const double dPhi1 = oneMc * deltaM * dmAn * (oneMc * delta * k_) * (invD1 * invD1);
        const double dLp   = lpp * alphaOverA;
        const double dNum  = dPhi1 + c_ * chi_ * dLp;
        const double dDen  = -alpha_ * c_ * chi_ * dLp;
        dwdm = hDot * (dNum * den - num * dDen) * (invDen * invDen);
    }

    /// Damping of the alpha transform: the derivative's homogeneous mode
    /// decays by 0.9 per sample (it was undamped at 1).
    static constexpr double kAlpha = 0.9;

public:
    static constexpr int kMaxPeriod = 8; ///< The longest period the predictor follows.

private:
    static constexpr int kRing = 16;     ///< dM/dt kept: two of the longest periods.

    // -- Members ----------------------------------------------------------------
    double ms_ = 3.5e5, a_ = 2.2e4, alpha_ = 1.6e-3, k_ = 2.7e4, c_ = 1.7e-1;
    double chi_ = 3.5e5 / 2.2e4;
    double tolerance_ = 1e-9;   ///< setTolerance().
    bool bySlope_ = false;      ///< The last evaluation comes from the slope.

    double invFs2_ = 0.5 / 48000.0;
    double fs2_ = 2.0 * 48000.0;

    double m_ = 0.0;
    double hPrev_ = 0.0;
    double hDotPrev_ = 0.0;
    double wPrev_ = 0.0;
    std::array<double, kRing> past_ {}; ///< dM/dt of the last samples, for the predictor.
    int pastAt_ = 0, pastCount_ = 0, period_ = 0;
};

} // namespace dspark
