// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

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
 *   M_an(Q)  = Ms · L(Q),  L(x) = coth(x) − 1/x,  Q = (H + α·M)/a
 *
 *             (1−c)·δ_M·(M_an − M)
 *   φ1      = ─────────────────────────,   δ = sign(dH/dt)
 *             (1−c)·δ·k − α·(M_an − M)
 *
 *   dM      φ1 + c·(Ms/a)·L'(Q)
 *   ──  =   ───────────────────────── · dH/dt
 *   dt      1 − c·α·(Ms/a)·L'(Q)
 *
 * δ_M gates the irreversible term so magnetization never moves unphysically
 * against the drive direction at reversal points.
 *
 * Integration is trapezoidal (matching the WDF/bilinear convention used
 * throughout the framework) solved per sample with a Newton-Raphson
 * iteration on M_n, seeded by an explicit Euler predictor — converges in
 * 1-3 iterations at audio rates. All internal state is double regardless of
 * T: the model mixes quantities spanning ~9 orders of magnitude.
 *
 * Verified by the test suite on first principles: odd symmetry of the B-H
 * loop, monotone loop area vs the loss parameter k, remanence after field
 * removal, and loop collapse as the reversible fraction c → 1.
 *
 * Mono processor: hold one instance per channel (the framework pattern for
 * stateful per-channel cores, like Hilbert).
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>

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
    /** @brief Prepares the integrator for the given sample rate. */
    void prepare(double sampleRate)
    {
        if (sampleRate <= 0.0) return;
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
     */
    void setParameters(double ms, double a, double alpha, double k, double c) noexcept
    {
        ms_    = std::max(ms, 1.0);
        a_     = std::max(a, 1.0);
        alpha_ = std::max(alpha, 0.0);
        k_     = std::max(k, 1.0);
        c_     = std::clamp(c, 0.0, 0.999);
        chi_   = ms_ / a_;
    }

    /** @brief Saturation magnetization Ms (A/m). */
    [[nodiscard]] double getSaturation() const noexcept { return ms_; }

    /**
     * @brief Small-signal susceptibility dM/dH around the demagnetized state.
     *
     * The reversible-branch slope c·χ_an/(1 − c·α·χ_an) with χ_an = Ms/(3a).
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
        const double h = static_cast<double>(fieldH);
        const double hDot = fs2_ * (h - hPrev_) - hDotPrev_;

        // Explicit predictor, then trapezoidal corrector with Newton steps.
        double m = m_ + 2.0 * invFs2_ * wPrev_;
        const double rhs = m_ + invFs2_ * wPrev_;

        double w = 0.0, dwdm = 0.0;
        for (int it = 0; it < 4; ++it)
        {
            evaluate(m, h, hDot, w, dwdm);
            const double g  = m - rhs - invFs2_ * w;
            const double gp = 1.0 - invFs2_ * dwdm;
            const double dm = g / gp;
            m -= dm;
            if (std::abs(dm) < 1e-9 * ms_)
                break;
        }
        // Physical clamp: |M| cannot exceed the saturation magnetization.
        m = std::clamp(m, -ms_, ms_);

        evaluate(m, h, hDot, w, dwdm);
        wPrev_ = w;
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
            // Taylor: L = x/3 − x³/45, L' = 1/3 − x²/15, L'' = −2x/15
            const double x2 = x * x;
            l   = x * (1.0 / 3.0 - x2 / 45.0);
            lp  = 1.0 / 3.0 - x2 / 15.0;
            lpp = -2.0 * x / 15.0;
        }
        else if (ax > 30.0)
        {
            // csch² underflows to 0 well before this point.
            l   = (x > 0.0 ? 1.0 : -1.0) - 1.0 / x;
            lp  = 1.0 / (x * x);
            lpp = -2.0 / (x * x * x);
        }
        else
        {
            const double e = std::exp(x);
            const double ie = 1.0 / e;
            const double sh = 0.5 * (e - ie);
            const double ch = 0.5 * (e + ie);
            const double coth = ch / sh;
            const double csch2 = 1.0 / (sh * sh);
            const double ix = 1.0 / x;
            l   = coth - ix;
            lp  = ix * ix - csch2;
            lpp = 2.0 * (csch2 * coth - ix * ix * ix);
        }
    }

    /** @brief Evaluates W = dM/dt and ∂W/∂M at (m, h, hDot). */
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

        const double phi1 = oneMc * deltaM * dM / d1;
        const double cChiLp = c_ * chi_ * lp;

        const double num = phi1 + cChiLp;
        const double den = 1.0 - alpha_ * cChiLp;
        w = hDot * num / den;

        // Analytic ∂W/∂M for the Newton step.
        const double alphaOverA = alpha_ / a_;
        const double dmAn  = ms_ * lp * alphaOverA - 1.0;          // d(M_an − M)/dM
        const double dPhi1 = oneMc * deltaM * dmAn * (oneMc * delta * k_) / (d1 * d1);
        const double dLp   = lpp * alphaOverA;
        const double dNum  = dPhi1 + c_ * chi_ * dLp;
        const double dDen  = -alpha_ * c_ * chi_ * dLp;
        dwdm = hDot * (dNum * den - num * dDen) / (den * den);
    }

    // -- Members ----------------------------------------------------------------
    double ms_ = 3.5e5, a_ = 2.2e4, alpha_ = 1.6e-3, k_ = 2.7e4, c_ = 1.7e-1;
    double chi_ = 3.5e5 / 2.2e4;

    double invFs2_ = 0.5 / 48000.0;
    double fs2_ = 2.0 * 48000.0;

    double m_ = 0.0;
    double hPrev_ = 0.0;
    double hDotPrev_ = 0.0;
    double wPrev_ = 0.0;
};

} // namespace dspark
