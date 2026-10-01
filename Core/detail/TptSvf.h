// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TptSvf.h
 * @brief Shared trapezoidal state-variable integrator, without parameter policy.
 *
 * Internal arithmetic used by StateVariableFilter and modulated filter banks.
 * Coefficients are supplied by the owner. No allocation or shared state; calls
 * modifying the same state belong to one processing thread.
 */
#include "../DspMath.h"

namespace dspark::detail
{
/** @brief Equivalent-current memories of the two trapezoidal integrators. */
template <FloatType T> struct TptSvfState
{
    T ic1eq = T(0);
    T ic2eq = T(0);
};

/** @brief Lowpass and constant-skirt-gain bandpass outputs. */
template <FloatType T> struct TptSvfOutput
{
    T lowpass;
    T bandpass;
};

/**
 * @brief Advances the expanded Simper TPT recurrence by one sample.
 * @param input Current input sample.
 * @param state Integrator memories, updated in place.
 * @param a1 1 / (1 + g * (g + k)), where g = tan(pi * f / fs) and k = 1 / Q.
 * @param a2 g * a1.
 * @param a3 g * a2.
 * @note Keeps the operation order of StateVariableFilter, including its state
 *       update. See https://cytomic.com/files/dsp/SvfLinearTrapOptimised2.pdf .
 */
template <FloatType T>
[[nodiscard]] inline TptSvfOutput<T> tptSvfStep(T input, TptSvfState<T> &state, T a1, T a2,
                                                T a3) noexcept
{
    const T v3 = input - state.ic2eq;
    const T v1 = a1 * state.ic1eq + a2 * v3;
    const T v2 = state.ic2eq + a2 * state.ic1eq + a3 * v3;
    state.ic1eq = T(2) * v1 - state.ic1eq;
    state.ic2eq = T(2) * v2 - state.ic2eq;
    return {v2, v1};
}
} // namespace dspark::detail
