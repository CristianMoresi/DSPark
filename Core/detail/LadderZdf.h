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

#include <array>

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
} // namespace dspark::detail
