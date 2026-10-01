// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file RationalKnee.h
 * @brief Shared rational soft-knee curve and its antiderivative.
 *
 * Internal mathematical transfer functions, not physical-device models. The
 * stateless functions require finite arguments, positive ceiling, and a knee
 * strictly between zero and the ceiling. No allocation or shared state.
 */
#include "../DspMath.h"
#include <cmath>

namespace dspark::detail
{
/** @brief Positive magnitude of the original Clipper rational soft knee. */
template <FloatType T>
[[nodiscard]] inline T rationalKneeMagnitude(T magnitude, T knee, T ceiling) noexcept
{
    if (magnitude <= knee)
        return magnitude;
    const T excess = magnitude - knee;
    const T range = ceiling - knee;
    return knee + (range * excess) / (excess + range);
}

/** @brief Unit-ceiling curve with independent positive and negative knees. */
[[nodiscard]] inline double rationalKneeShape(double input, double positiveKnee,
                                              double negativeKnee) noexcept
{
    const double knee = input >= 0.0 ? positiveKnee : negativeKnee;
    return std::copysign(rationalKneeMagnitude(std::abs(input), knee, 1.0), input);
}

/** @brief Antiderivative of the unit-ceiling curve, with F(0) = 0. */
[[nodiscard]] inline double rationalKneeIntegral(double input, double positiveKnee,
                                                 double negativeKnee) noexcept
{
    const double magnitude = std::abs(input);
    const double knee = input >= 0.0 ? positiveKnee : negativeKnee;
    if (magnitude <= knee)
        return 0.5 * magnitude * magnitude;
    const double range = 1.0 - knee;
    const double excess = magnitude - knee;
    return 0.5 * knee * knee + excess - range * range * std::log((excess + range) / range);
}

} // namespace dspark::detail
