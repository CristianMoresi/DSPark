// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LadderZdf.h
 * @brief Shared zero-delay-feedback ladder step, without parameter policy.
 *
 * Internal arithmetic used by LadderFilter and by owners that change the
 * cutoff every sample (synthesizer voices). Coefficients are supplied by the
 * owner from the prewarped integrator gain g = tan(pi * fc / fs) and the
 * resonance. No allocation or shared state; calls modifying the same state
 * belong to one processing thread.
 */

#include "../DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

namespace dspark::detail
{
/** @brief Output tap mix of the four-stage cascade. */
enum class LadderTap
{
    LP6,
    LP12,
    LP18,
    LP24,
    BP12,
    HP24
};

/** @brief Coefficients derived from g and the resonance. */
template <FloatType T> struct LadderCoeffs
{
    T G, G2, G3, G4, ig, k, invFbDen;
};

/**
 * @brief Derives the ladder coefficients.
 * @param g   Prewarped integrator gain, tan(pi * fc / fs).
 * @param res Resonance 0..1; 1 is the self-oscillation boundary (k = 4).
 */
template <FloatType T> [[nodiscard]] inline LadderCoeffs<T> makeLadderCoeffs(T g, T res) noexcept
{
    LadderCoeffs<T> c;
    c.G   = g / (T(1) + g);
    c.G2  = c.G * c.G;
    c.G3  = c.G2 * c.G;
    c.G4  = c.G3 * c.G;
    c.ig  = T(1) / (T(1) + g);
    c.k   = res * T(4);                         // k = 4 -> self-oscillation
    c.invFbDen = T(1) / (T(1) + c.k * c.G4);    // zero-delay loop denominator
    return c;
}

/**
 * @brief Advances the four TPT one-pole stages and the resolved feedback loop
 *        by one sample.
 * @param input Current input sample.
 * @param z     Integrator states, updated in place.
 * @param c     Coefficients from makeLadderCoeffs().
 * @param drive Feedback saturation: values above 1 pass the estimated
 *              feedback through a bounded tanh, at or below 1 it is linear.
 */
template <LadderTap Tap, FloatType T>
[[nodiscard]] inline T ladderStep(T input, std::array<T, 4>& z, const LadderCoeffs<T>& c,
                                  T drive) noexcept
{
    // Estimate the LP24 output from the integrator states (zero-delay logic)
    const T Sest = c.G3 * c.ig * z[0]
                 + c.G2 * c.ig * z[1]
                 + c.G  * c.ig * z[2]
                 +        c.ig * z[3];

    // Saturate the estimated feedback (cheap ZDF nonlinearity: the state
    // contribution is shaped, the direct G4*u term stays linear).
    // fastTanh clamps its input, so the saturated loop stays bounded.
    T fbSignal = Sest;
    if (drive > T(1))
        fbSignal = fastTanh(fbSignal * drive) / drive;

    // Resolve zero-delay loop (denominator precomputed with the coefficients)
    const T u = (input - c.k * fbSignal) * c.invFbDen;

    // Four cascaded TPT one-pole stages. Tap outputs carry no state
    // between samples, so they stay in registers ({} avoids a spurious
    // uninitialised-variable warning; the stores are dead after unroll).
    std::array<T, 4> y {};
    T x = u;
    for (int i = 0; i < 4; ++i)
    {
        const T v = (x - z[i]) * c.G;
        y[i] = v + z[i];
        z[i] = y[i] + v;
        x = y[i];
    }

    // Output tap mix (resolved at compile time per tap instantiation)
    if constexpr (Tap == LadderTap::LP6)       return y[0];
    else if constexpr (Tap == LadderTap::LP12) return y[1];
    else if constexpr (Tap == LadderTap::LP18) return y[2];
    else if constexpr (Tap == LadderTap::LP24) return y[3];
    else if constexpr (Tap == LadderTap::BP12) return y[0] - y[2];
    else
    {
        // HP24: apply the binomial (1-L)^4 to `u` (the ladder's true input,
        // post feedback), NOT to `input`. With resonance, input = u*(1+k) at
        // DC, so the input-based form leaked DC with gain k/(1+k) - e.g. 67%
        // of the DC passed straight through a "high-pass" at resonance 0.5.
        // With u the DC gain is exactly 0 and the resonant peak is preserved.
        return u - T(4)*y[0] + T(6)*y[1] - T(4)*y[2] + y[3];
    }
}

// ============================================================================
// Per-stage saturating ladder (implicit solve)
// ============================================================================

/**
 * @brief Largest prewarped integrator gain of the per-stage model: tan(0.45 pi),
 *        a cutoff of 0.45 * fs.
 *
 * Above it the exact trapezoidal solution of the saturating stages stops
 * being an audio signal: at a cutoff of 0.499 * fs (gain 280) its states grew
 * to 90 and its output to 8.6 times full scale on full-scale noise, while at
 * 0.45 * fs they stay below 6 and 1.6.
 */
inline constexpr double kLadderStagesMaxGain = 6.313751514675041;

/** @brief Coefficients of ladderStepStages(), derived by makeLadderStageCoeffs(). */
struct LadderStageCoeffs
{
    double g, k, d, gOverD, tol;
};

/**
 * @brief Derives the coefficients of the per-stage saturating ladder.
 * @param g     Prewarped integrator gain, tan(pi * fc / fs); clamped to
 *              kLadderStagesMaxGain.
 * @param res   Resonance; 1 is the self-oscillation boundary (k = 4).
 * @param drive Level scale of the stage nonlinearity (> 0): a stage
 *              saturates as tanh(drive * v) / drive.
 */
template <FloatType T>
[[nodiscard]] inline LadderStageCoeffs makeLadderStageCoeffs(T g, T res, T drive) noexcept
{
    LadderStageCoeffs c;
    c.g = std::min(static_cast<double>(g), kLadderStagesMaxGain);
    c.k = 4.0 * static_cast<double>(res);
    c.d = static_cast<double>(drive);
    c.gOverD = c.g / c.d;
    // Newton stops once the largest stage update is below this many knee
    // widths (1 / drive). Newton converges quadratically, so the error left
    // is about 10 * tol^2 knee widths (measured 9.8e-8 at 1e-4, 1.6e-11 at
    // 1e-7): below the sample type's resolution.
    c.tol = std::is_same_v<T, float> ? 1e-4 : 1e-7;
    return c;
}

/**
 * @brief tanh(a) by its [9/8] continued-fraction convergent, clamped to +-1
 *        where the convergent reaches 1 (|a| = 6.297).
 *
 * Absolute error below 6.8e-6 everywhere; the clamp leaves a slope step of
 * 2.5e-5. The numerator and denominator are also used by ladderTanhRatio().
 */
[[nodiscard]] inline double ladderTanh(double a) noexcept
{
    constexpr double kClamp = 6.297019181717114;
    if (a >  kClamp) return  1.0;
    if (a < -kClamp) return -1.0;
    const double a2 = a * a;
    const double num = 34459425.0 + a2 * (4729725.0 + a2 * (135135.0 + a2 * (990.0 + a2)));
    const double den = 34459425.0 + a2 * (16216200.0 + a2 * (945945.0 + a2 * (13860.0 + a2 * 45.0)));
    return a * num / den;
}

/** @brief ladderTanh(a) / a, finite at a = 0 (where it is 1). */
[[nodiscard]] inline double ladderTanhRatio(double a) noexcept
{
    constexpr double kClamp = 6.297019181717114;
    const double m = a < 0.0 ? -a : a;
    if (m > kClamp) return 1.0 / m;
    const double a2 = a * a;
    const double num = 34459425.0 + a2 * (4729725.0 + a2 * (135135.0 + a2 * (990.0 + a2)));
    const double den = 34459425.0 + a2 * (16216200.0 + a2 * (945945.0 + a2 * (13860.0 + a2 * 45.0)));
    return num / den;
}

/** @brief Newton step cap of ladderStepStages() (the worst case measured is 10). */
inline constexpr int kLadderStagesMaxIterations = 16;

/**
 * @brief Advances the per-stage saturating ladder by one sample.
 *
 * Solves the trapezoidal (TPT) discretization of the transistor-ladder model
 * (Huovilainen 2004; D'Angelo and Valimaki 2013)
 *
 *     dy_i/dt = wc * [phi(y_{i-1}) - phi(y_i)],  y_0 = x - k * y_4,
 *     phi(v)  = tanh(drive * v) / drive,
 *
 * where every stage saturates and the feedback loop is delay-free. The four
 * implicit stage equations y_i = s_i + g * (phi(y_{i-1}) - phi(y_i)) are
 * solved jointly, as in Zavalishin's TPT treatment of nonlinear zero-delay
 * loops:
 *
 * 1. A fixed-pivot linear solve (each phi replaced by v * phi(s)/s at the
 *    integrator states) gives a stable starting point at any gain.
 * 2. Newton iterations on the four equations. The Jacobian is bidiagonal
 *    plus the feedback corner, so each step is an O(4) elimination, and the
 *    closing pivot is always >= 1 (no singularity at any setting). Each step
 *    is clipped to one knee width (1 / drive): undamped Newton cycles
 *    between the two saturation branches once g exceeds about 1.
 *
 * The solve runs in double whatever T is. It stops when the largest update
 * falls below c.tol knee widths, after at most kLadderStagesMaxIterations
 * steps: measured with the float tolerance over noise, sine and saw inputs at
 * 44.1 to 96 kHz, cutoff 0.004 to 0.45 * fs, resonance 0 to 1.15 and drive
 * 0.3 to 10, a mean of 1.8 Newton steps after the start, 4.5 at the worst
 * setting, and at most 10 (the cap is never reached).
 * In the linear limit (drive -> 0) it reduces to ladderStep() without
 * drive. Output taps are those of ladderStep().
 *
 * @param input Current input sample.
 * @param z     Integrator states, updated in place.
 * @param c     Coefficients from makeLadderStageCoeffs().
 */
template <LadderTap Tap, FloatType T>
[[nodiscard]] inline T ladderStepStages(T input, std::array<T, 4>& z, const LadderStageCoeffs& c) noexcept
{
    const double x = static_cast<double>(input);
    const double g = c.g, k = c.k, d = c.d, gd = c.gOverD;
    const double s0 = static_cast<double>(z[0]), s1 = static_cast<double>(z[1]);
    const double s2 = static_cast<double>(z[2]), s3 = static_cast<double>(z[3]);

    // 1) Fixed-pivot start: phi(v) ~ v * r_i with r_i = tanh(d s)/(d s) at
    //    the states. Each stage is then linear, y_i = a_i * y_{i-1} + b_i,
    //    and the loop closes in one division.
    const double r0 = ladderTanhRatio(d * (x - k * s3));
    const double r1 = ladderTanhRatio(d * s0), r2 = ladderTanhRatio(d * s1);
    const double r3 = ladderTanhRatio(d * s2), r4 = ladderTanhRatio(d * s3);
    const double i1 = 1.0 / (1.0 + g * r1), i2 = 1.0 / (1.0 + g * r2);
    const double i3 = 1.0 / (1.0 + g * r3), i4 = 1.0 / (1.0 + g * r4);
    const double a1 = g * r0 * i1, a2 = g * r1 * i2, a3 = g * r2 * i3, a4 = g * r3 * i4;
    const double b1 = s0 * i1, b2 = s1 * i2, b3 = s2 * i3, b4 = s3 * i4;
    const double A = a1 * a2 * a3 * a4;
    const double B = ((b1 * a2 + b2) * a3 + b3) * a4 + b4;
    double u = (x - k * B) / (1.0 + k * A);
    double y0 = a1 * u + b1;
    double y1 = a2 * y0 + b2;
    double y2 = a3 * y1 + b3;
    double y3 = a4 * y2 + b4;

    // 2) Clipped Newton steps on r_i = y_i - s_i - g (phi_{i-1} - phi_i).
    for (int it = 0; it < kLadderStagesMaxIterations; ++it)
    {
        u = x - k * y3;
        const double t0 = ladderTanh(d * u),  t1 = ladderTanh(d * y0);
        const double t2 = ladderTanh(d * y1), t3 = ladderTanh(d * y2);
        const double t4 = ladderTanh(d * y3);
        const double p0 = 1.0 - t0 * t0, p1 = 1.0 - t1 * t1, p2 = 1.0 - t2 * t2;
        const double p3 = 1.0 - t3 * t3, p4 = 1.0 - t4 * t4;
        // Diagonal 1 + g phi'(y_i); sub-diagonal -g phi'(y_{i-1}); corner
        // (row 1, column 4) g k phi'(u). The reciprocals are independent.
        const double q1 = 1.0 / (1.0 + g * p1), q2 = 1.0 / (1.0 + g * p2);
        const double q3 = 1.0 / (1.0 + g * p3), q4 = 1.0 / (1.0 + g * p4);
        const double e1 = y0 - s0 - gd * (t0 - t1);
        const double e2 = y1 - s1 - gd * (t1 - t2);
        const double e3 = y2 - s2 - gd * (t2 - t3);
        const double e4 = y3 - s3 - gd * (t3 - t4);
        // Forward elimination: delta_i = P_i + Q_i * delta_4.
        const double P1 = -e1 * q1,                       Q1 = -g * k * p0 * q1;
        const double P2 = (g * p1 * P1 - e2) * q2,        Q2 = g * p1 * Q1 * q2;
        const double P3 = (g * p2 * P2 - e3) * q3,        Q3 = g * p2 * Q2 * q3;
        const double P4 = (g * p3 * P3 - e4) * q4,        Q4 = g * p3 * Q3 * q4;
        const double d4 = P4 / (1.0 - Q4);   // Q4 <= 0: the pivot is >= 1
        const double d1 = P1 + Q1 * d4, d2 = P2 + Q2 * d4, d3 = P3 + Q3 * d4;
        double m = std::max(std::max(std::abs(d1), std::abs(d2)),
                            std::max(std::abs(d3), std::abs(d4))) * d;
        const double scale = m > 1.0 ? 1.0 / m : 1.0;
        y0 += scale * d1;
        y1 += scale * d2;
        y2 += scale * d3;
        y3 += scale * d4;
        if (m < c.tol) break;   // (also false for NaN: the cap ends that case)
    }
    u = x - k * y3;

    z[0] = static_cast<T>(2.0 * y0 - s0);
    z[1] = static_cast<T>(2.0 * y1 - s1);
    z[2] = static_cast<T>(2.0 * y2 - s2);
    z[3] = static_cast<T>(2.0 * y3 - s3);

    double out;
    if constexpr (Tap == LadderTap::LP6)       out = y0;
    else if constexpr (Tap == LadderTap::LP12) out = y1;
    else if constexpr (Tap == LadderTap::LP18) out = y2;
    else if constexpr (Tap == LadderTap::LP24) out = y3;
    else if constexpr (Tap == LadderTap::BP12) out = y0 - y2;
    else                                       out = u - 4.0 * y0 + 6.0 * y1 - 4.0 * y2 + y3;
    return static_cast<T>(out);
}
} // namespace dspark::detail
