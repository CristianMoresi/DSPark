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
    /// Linear (drive-free) prediction: G = g / (1 + g), the state weights
    /// of the loop estimate and 1 / (1 + k G^4).
    double G, w0, w1, w2, w3, invLoop;
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
    const double ig = 1.0 / (1.0 + c.g);
    c.G = c.g * ig;
    c.w3 = ig;
    c.w2 = c.G * ig;
    c.w1 = c.G * c.w2;
    c.w0 = c.G * c.w1;
    c.invLoop = 1.0 / (1.0 + c.k * (c.G * c.G) * (c.G * c.G));
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
 * @brief Integrator gain below which ladderStepStages() falls back to the
 *        integrator states instead of the fixed-pivot solve (cutoff about
 *        0.047 * fs) when the linear prediction is not used.
 *
 * At a low gain each stage moves little in one sample, so the states are
 * already a close start and the fixed-pivot solve (five tanh ratios and five
 * divisions, about one Newton step of work) does not pay for itself. Above
 * it the fixed-pivot start saves more than it costs.
 */
inline constexpr double kLadderStagesStateStartGain = 0.15;

/**
 * @brief Largest stage level of the linear prediction, in knee widths, at
 *        which ladderStepStages() starts Newton from that prediction (above
 *        kLadderStagesStateStartGain).
 *
 * The prediction is the drive-free ladder solved exactly (a dozen operations,
 * no division): close to the root wherever the stages are not deep in
 * saturation, and often closer than the fixed-pivot solve, which costs about
 * one Newton step. With the float tolerance over noise, sines and a saw at
 * 44.1 to 96 kHz, cutoff 0.004 to 0.45 * fs, resonance 0 to 1.15 and drive
 * 0.3 to 10, the mean cost falls from 2.7 to 2.2 steps (a fixed-pivot start
 * counted as one), and from 2.9 to 2.0 for sines at a 15 kHz cutoff with
 * drive 3. Beyond the limit the fixed-pivot start takes over.
 */
inline constexpr double kLadderStagesLinearStartLimit = 2.0;

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
 * 1. Start. Below kLadderStagesStateStartGain the states themselves. Above
 *    it, the drive-free ladder solved exactly (the linear prediction, no
 *    division) while every stage of it stays within
 *    kLadderStagesLinearStartLimit knee widths, otherwise a fixed-pivot
 *    linear solve (each phi replaced by v * phi(s)/s at the integrator
 *    states), a stable starting point at any gain.
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
 * 0.3 to 10, a mean of 2.2 steps counting a fixed-pivot start as one (2.7
 * with the fixed-pivot start at every gain above the state-start limit),
 * 4.1 Newton steps at the worst setting, and at most 7 (the cap is never
 * reached).
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

    // 1) Start. Below kLadderStagesStateStartGain the states. Above it the
    //    drive-free ladder solved exactly, while every stage of it stays
    //    within kLadderStagesLinearStartLimit knee widths; otherwise a
    //    fixed-pivot solve: phi(v) ~ v * r_i with r_i = tanh(d s)/(d s) at
    //    the states. Each stage is then linear, y_i = a_i * y_{i-1} + b_i,
    //    and the loop closes in one division.
    double u = 0.0, y0 = s0, y1 = s1, y2 = s2, y3 = s3;
    bool pivot = false;
    if (g >= kLadderStagesStateStartGain)
    {
        // Linear prediction in closed form (every stage from u and the
        // states at once): y_i = G y_{i-1} + (1 - G) s_i.
        const double G = c.G, H = 1.0 - G;
        const double h0 = H * s0, h1 = H * s1, h2 = H * s2, h3 = H * s3;
        u = (x - k * ((c.w0 * s0 + c.w1 * s1) + (c.w2 * s2 + c.w3 * s3))) * c.invLoop;
        const double G2 = G * G;
        const double l0 = G * u + h0;
        const double l1 = G2 * u + (G * h0 + h1);
        const double l2 = (G2 * G) * u + (G2 * h0 + (G * h1 + h2));
        const double l3 = (G2 * G2) * u + ((G2 * G) * h0 + (G2 * h1 + (G * h2 + h3)));
        const double reach = std::max(std::max(std::abs(u), std::abs(l0)),
                                      std::max(std::max(std::abs(l1), std::abs(l2)), std::abs(l3))) * d;
        pivot = !(reach < kLadderStagesLinearStartLimit);
        y0 = l0; y1 = l1; y2 = l2; y3 = l3;
    }
    if (pivot)
    {
        const double r0 = ladderTanhRatio(d * (x - k * s3));
        const double r1 = ladderTanhRatio(d * s0), r2 = ladderTanhRatio(d * s1);
        const double r3 = ladderTanhRatio(d * s2), r4 = ladderTanhRatio(d * s3);
        const double i1 = 1.0 / (1.0 + g * r1), i2 = 1.0 / (1.0 + g * r2);
        const double i3 = 1.0 / (1.0 + g * r3), i4 = 1.0 / (1.0 + g * r4);
        const double a1 = g * r0 * i1, a2 = g * r1 * i2, a3 = g * r2 * i3, a4 = g * r3 * i4;
        const double b1 = s0 * i1, b2 = s1 * i2, b3 = s2 * i3, b4 = s3 * i4;
        const double A = a1 * a2 * a3 * a4;
        const double B = ((b1 * a2 + b2) * a3 + b3) * a4 + b4;
        u = (x - k * B) / (1.0 + k * A);
        y0 = a1 * u + b1;
        y1 = a2 * y0 + b2;
        y2 = a3 * y1 + b3;
        y3 = a4 * y2 + b4;
    }

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

// ============================================================================
// Input-saturating ladder (sub-stepped, delay-free loop)
// ============================================================================

#if defined(_MSC_VER) && !defined(__clang__)
    #define DSPARK_LADDER_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_LADDER_INLINE inline __attribute__((always_inline))
#else
    #define DSPARK_LADDER_INLINE inline
#endif

/** @brief Sub-steps per sample of the input-saturating ladder. */
inline constexpr int kLadderInputSubsteps = 4;

/** @brief Where ladderTanh()'s rational function reaches 1, and that squared. */
inline constexpr double kLadderTanhLimit = 6.297019181717114;
inline constexpr double kLadderTanhLimit2 = kLadderTanhLimit * kLadderTanhLimit;

/**
 * @brief The two polynomials of ladderTanh() in a^2 (tanh(a) = a num / den),
 *        each summed as two halves so that both are ready four operations
 *        after a^2.
 */
inline void ladderTanhParts(double a2, double& num, double& den) noexcept
{
    const double a4 = a2 * a2;
    const double a6 = a4 * a2;
    num = ((34459425.0 + 4729725.0 * a2) + 135135.0 * a4) + a6 * (990.0 + a2);
    den = ((34459425.0 + 16216200.0 * a2) + 945945.0 * a4) + a6 * (13860.0 + 45.0 * a2);
}

/** @brief Coefficients of ladderStepInput(), derived by makeLadderInputCoeffs(). */
struct LadderInputCoeffs
{
    double G = 0.0, H = 1.0;       ///< One stage per sub-step: y = G in + H s, then s = 2 y - s.
    double d = 1.0, invD = 1.0;    ///< Level scale of the saturator and its reciprocal.
    double kd = 0.0;               ///< k d.
    double c = 0.0;                ///< Loop gain past the saturator, k G^4.
    double onePlusC = 1.0;
    double kappa = 0.0;            ///< k times what one unit of saturator output adds to the next feedback sum.
    double cS[4] = {};             ///< Feedback sum of the states: the last stage's output with no input.
    double cAhead[4] = {};         ///< k d times that sum one sub-step on, from the states now.
    double pre = 0.0, post = 0.0;  ///< Level corrections of the two rate changes.
};

/**
 * @brief State of ladderStepInput(), in double whatever the sample type is.
 *
 * The integrators run at the sub-step rate, where a stage moves by a small
 * fraction of its input per step, so they are kept at the precision of the
 * arithmetic; a block and the same samples one by one then give the same
 * output.
 */
struct LadderInputState
{
    double q[4] = {};         ///< Integrator states.
    double ahead = 0.0;       ///< cAhead . q: k d times the feedback sum q reaches a sub-step on with no input.
    double loopStates = 0.0;  ///< That term as the last sub-step formed it, from the states it started from,
    double loopDrive = 0.0;   ///< and kappa times its saturator output: together, k d times the feedback sum of q.
    double input = 0.0;       ///< Previous input after its level correction.
    double output = 0.0;      ///< Previous output.
};

/** @brief Recomputes the carried sums after the states or the coefficients changed. */
inline void ladderInputRefresh(LadderInputState& h, const LadderInputCoeffs& c) noexcept
{
    h.ahead = ((c.cAhead[0] * h.q[0] + c.cAhead[1] * h.q[1]) + c.cAhead[2] * h.q[2]) + c.cAhead[3] * h.q[3];
    h.loopStates = c.kd * ((c.cS[0] * h.q[0] + c.cS[1] * h.q[1]) + (c.cS[2] * h.q[2] + c.cS[3] * h.q[3]));
    h.loopDrive = 0.0;
}

/**
 * @brief Derives the coefficients of the input-saturating ladder.
 *
 * The integrators are prewarped at the sub-step rate, so the cutoff is exact
 * and the response below it is that of a ladder at
 * (4 fs / pi) tan(pi fc / (4 fs)): within 0.2% of fc at 0.1 fs and 4.4% at
 * 0.45 fs.
 *
 * @param cutoffRatio Cutoff as a fraction of the sample rate; held within
 *                    (0, 0.499].
 * @param res         Resonance; 1 is the self-oscillation boundary (k = 4).
 * @param drive       Level scale of the saturator (> 0): the loop input
 *                    saturates as tanh(drive * v) / drive.
 */
template <FloatType T>
[[nodiscard]] inline LadderInputCoeffs makeLadderInputCoeffs(double cutoffRatio, T res, T drive) noexcept
{
    LadderInputCoeffs c;
    const double ratio = cutoffRatio > 1e-6 ? std::min(cutoffRatio, 0.499) : 1e-6;
    const double g = std::tan(pi<double> * ratio / kLadderInputSubsteps);
    c.G = g / (1.0 + g);
    c.H = 1.0 - c.G;
    const double G2 = c.G * c.G;
    const double power[5] = { 1.0, c.G, G2, G2 * c.G, G2 * G2 };
    const double k = 4.0 * static_cast<double>(res);
    c.d = static_cast<double>(drive);
    c.invD = 1.0 / c.d;
    c.kd = k * c.d;
    c.c = k * power[4];
    c.onePlusC = 1.0 + c.c;
    // One sub-step of the four trapezoidal stages fed w: y_i = G^(i+1) w +
    // sum_{j <= i} G^(i-j) H s_j, and each state moves to 2 y_i - s_i, so the
    // states map as s' = a s + b w.
    double a[4][4] = {}, b[4] = {};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
            a[i][j] = j < i ? 2.0 * power[i - j] * c.H : (j == i ? 1.0 - 2.0 * c.G : 0.0);
        b[i] = 2.0 * power[i + 1];
        c.cS[i] = c.H * power[3 - i];
    }
    // The feedback sum one sub-step on is therefore (cS a) s + (cS b) w.
    double perUnit = 0.0;
    for (int j = 0; j < 4; ++j)
    {
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) sum += c.cS[i] * a[i][j];
        c.cAhead[j] = c.kd * sum;
        perUnit += c.cS[j] * b[j];
    }
    c.kappa = k * perUnit;
    // The input is interpolated linearly over the sub-steps and the output is
    // their mean; they lose 5.0 and 2.5 dB at 0.417 fs. A one-pole correction
    // on each side, y = (1 + b) x - b y[-1], restores them within 0.25 dB up
    // to there.
    c.pre = 0.30;
    c.post = 0.14;
    return c;
}

/** @brief One channel's registers while ladderStepInput() runs a sample. */
struct LadderInputLane
{
    double q0 = 0.0, q1 = 0.0, q2 = 0.0, q3 = 0.0;
    double ahead = 0.0, kl = 0.0, kw = 0.0, R = 0.0;
    double x = 0.0, xs = 0.0, dx = 0.0, acc = 0.0;

    /** @brief Loads the state and the first sub-step's loop drive. */
    template <FloatType T>
    DSPARK_LADDER_INLINE void begin(T input, const LadderInputState& h, const LadderInputCoeffs& c) noexcept
    {
        const double xPrev = h.input;
        x = (1.0 + c.pre) * static_cast<double>(input) - c.pre * xPrev;
        q0 = h.q[0];
        q1 = h.q[1];
        q2 = h.q[2];
        q3 = h.q[3];
        dx = (x - xPrev) * (1.0 / kLadderInputSubsteps);
        xs = xPrev + dx;
        ahead = h.ahead;
        // The same two terms a sub-step hands the next one, so the last
        // saturator output of a sample reaches the first loop drive of the
        // next through one subtraction as well.
        R = (c.d * xs - h.loopStates) - h.loopDrive;
        acc = 0.0;
    }

    /** @brief One sub-step: the loop input, the stages and the taps. */
    template <LadderTap Tap>
    DSPARK_LADDER_INLINE void substep(const LadderInputCoeffs& c) noexcept
    {
        // Saturated loop input, one Newton step from the explicit value:
        // with tanh(R) = R nu / de, t / (1 + c (1 - t^2)) is
        // R nu de / ((1 + c) de^2 - c R^2 nu^2). It is divided out twice,
        // plain for the stages and scaled by kappa for the next loop drive,
        // which then follows the division by one subtraction. Past the limit
        // of the rational tanh the saturator is at 1 and the Newton step
        // leaves it there (1 - t^2 = 0): a branch, with nothing to divide. A
        // NaN drive takes it too, as -1.
        kl = ahead;
        double wk;
        const double r2 = R * R;
        if (r2 < kLadderTanhLimit2) [[likely]]
        {
            double nu, de;
            ladderTanhParts(r2, nu, de);
            const double nd = nu * de;
            const double den = c.onePlusC * (de * de) - (c.c * r2) * (nu * nu);
            wk = (R * nd) / den;
            kw = ((c.kappa * R) * nd) / den;
        }
        else
        {
            wk = R > 0.0 ? 1.0 : -1.0;
            kw = R > 0.0 ? c.kappa : -c.kappa;
        }
        xs += dx;
        R = (c.d * xs - kl) - kw;
        // The four stages in turn; each state moves to twice its output minus
        // itself.
        const double w = wk * c.invD;
        const double y0 = c.G * w + c.H * q0;
        const double y1 = c.G * y0 + c.H * q1;
        const double y2 = c.G * y1 + c.H * q2;
        const double y3 = c.G * y2 + c.H * q3;
        if constexpr (Tap == LadderTap::LP6)       acc += y0;
        else if constexpr (Tap == LadderTap::LP12) acc += y1;
        else if constexpr (Tap == LadderTap::LP18) acc += y2;
        else if constexpr (Tap == LadderTap::LP24) acc += y3;
        else if constexpr (Tap == LadderTap::BP12) acc += y0 - y2;
        else
        {
            // HP24: the binomial (1 - L)^4 on the cascade's input, the
            // saturated loop input (see ladderStep()).
            acc += ((w - 4.0 * y0) + 6.0 * y1) + (y3 - 4.0 * y2);
        }
        q0 = 2.0 * y0 - q0;
        q1 = 2.0 * y1 - q1;
        q2 = 2.0 * y2 - q2;
        q3 = 2.0 * y3 - q3;
        // For the sub-step after the next; the last stage's state joins last.
        ahead = ((c.cAhead[0] * q0 + c.cAhead[1] * q1) + c.cAhead[2] * q2) + c.cAhead[3] * q3;
    }

    /** @brief Stores the state and returns the level-corrected mean of the sub-steps. */
    template <FloatType T>
    [[nodiscard]] DSPARK_LADDER_INLINE T finish(LadderInputState& h, const LadderInputCoeffs& c) const noexcept
    {
        h.q[0] = q0;
        h.q[1] = q1;
        h.q[2] = q2;
        h.q[3] = q3;
        h.ahead = ahead;
        h.loopStates = kl;
        h.loopDrive = kw;
        h.input = x;
        const double y = (1.0 + c.post) * (acc * (1.0 / kLadderInputSubsteps)) - c.post * h.output;
        h.output = y;
        return static_cast<T>(y);
    }
};

/**
 * @brief Advances the input-saturating ladder by one sample.
 *
 * Linear trapezoidal stages with one saturator on the loop input,
 *
 *     dy_i/dt = wc * (y_{i-1} - y_i),  y_0 = phi(x - k * y_4),
 *     phi(v)  = tanh(drive * v) / drive,
 *
 * run in kLadderInputSubsteps sub-steps per sample: the input is
 * interpolated linearly across them and the output is their mean, each with
 * its level correction. The loop is closed without delay in every sub-step.
 * With the cascade written y_4 = G^4 w + S (S from the states), the loop
 * input u solves u + k G^4 phi(u) = x - k S. k G^4 stays below 0.036 at
 * every setting (the sub-steps keep G small), so one Newton step from the
 * explicit value r = x - k S leaves the square of that:
 * phi(u) = t / (1 + k G^4 (1 - t^2)), t = tanh(drive r). The arithmetic is
 * double whatever T is; the tanh is ladderTanh()'s rational function.
 *
 * What a sample costs is the path from one loop drive to the next, so that
 * path is kept to the saturator and one subtraction: the feedback sum a
 * sub-step leaves behind is formed as cAhead . s, from the states it started
 * from, plus kappa times its saturator output, and the stages themselves
 * advance off that path. The function is always inlined: a block loop then
 * keeps the state in registers and loads the coefficients once.
 *
 * @param input Current input sample.
 * @param h     State of the channel, updated in place. Its carried sums must
 *              match its integrators and c (ladderInputRefresh() after
 *              either changed outside this function).
 * @param c     Coefficients from makeLadderInputCoeffs().
 */
template <LadderTap Tap, FloatType T>
[[nodiscard]] DSPARK_LADDER_INLINE T ladderStepInput(T input, LadderInputState& h, const LadderInputCoeffs& c) noexcept
{
    LadderInputLane lane;
    lane.begin(input, h, c);
    for (int j = 0; j < kLadderInputSubsteps; ++j) lane.substep<Tap>(c);
    return lane.finish<T>(h, c);
}

/**
 * @brief ladderStepInput() on two channels at once, sub-step by sub-step.
 *
 * The two solves are independent, so taking their sub-steps in turn lets
 * their dependency chains overlap. Each channel's arithmetic is that of
 * ladderStepInput(): the outputs are the same.
 *
 * @param a  First channel's sample, replaced by its output.
 * @param b  Second channel's sample, replaced by its output.
 * @param ha First channel's state.
 * @param hb Second channel's state.
 * @param c  Coefficients from makeLadderInputCoeffs().
 */
template <LadderTap Tap, FloatType T>
DSPARK_LADDER_INLINE void ladderStepInputPair(T& a, T& b, LadderInputState& ha, LadderInputState& hb,
                                              const LadderInputCoeffs& c) noexcept
{
    LadderInputLane la, lb;
    la.begin(a, ha, c);
    lb.begin(b, hb, c);
    for (int j = 0; j < kLadderInputSubsteps; ++j)
    {
        la.substep<Tap>(c);
        lb.substep<Tap>(c);
    }
    a = la.finish<T>(ha, c);
    b = lb.finish<T>(hb, c);
}

#undef DSPARK_LADDER_INLINE
} // namespace dspark::detail
