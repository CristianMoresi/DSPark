// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file ClipperShape.h
 * @brief Shared stateless clipping curves, without gain, filtering or resampling.
 *
 * Internal kernels shared by clipping processors and worker implementations. A
 * positive finite ceiling is required. These are mathematical transfer curves,
 * not measured physical models. Threading: pure functions with no shared state.
 */
#include "../DspMath.h"
#include "RationalKnee.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace dspark::detail
{
enum class ClipperCurve
{
    Hard,
    Tanh,
    Sine,
    GoldenRatio,
    SymmetricKnee,
    AsymmetricKnee
};

/** @brief Exact small-signal slope of the implemented stateless curve. */
template <ClipperCurve Curve, FloatType T>
[[nodiscard]] constexpr T clipperSmallSignalSlope() noexcept
{
    if constexpr (Curve == ClipperCurve::Sine)
        return fastSinPolynomial<T>[0];
    else
        return T(1);
}

/** @brief Exact linear-region endpoints of a rational clipping curve. */
template <ClipperCurve Curve, FloatType T>
[[nodiscard]] inline std::array<T, 2> clipperLinearLimits(T ceiling) noexcept
{
    static_assert(Curve == ClipperCurve::GoldenRatio || Curve == ClipperCurve::SymmetricKnee ||
                  Curve == ClipperCurve::AsymmetricKnee);
    if constexpr (Curve == ClipperCurve::SymmetricKnee)
        return {-T(0.9) * ceiling, T(0.9) * ceiling};
    else if constexpr (Curve == ClipperCurve::AsymmetricKnee)
        return {-T(0.78) * ceiling, T(0.96) * ceiling};
    else
    {
        const T limit = ceiling / T(1.6180339887498948482);
        return {-limit, limit};
    }
}

/** @brief Shared transfer functions; no state or implicit antialiasing. */
template <ClipperCurve Curve, FloatType T>
[[nodiscard]] inline T clipperShape(T sample, T ceiling) noexcept
{
    if constexpr (Curve == ClipperCurve::Hard)
        return std::clamp(sample, -ceiling, ceiling);
    else if constexpr (Curve == ClipperCurve::Tanh)
        return ceiling * std::tanh(sample / ceiling);
    else if constexpr (Curve == ClipperCurve::Sine)
    {
        constexpr T halfPi = static_cast<T>(std::numbers::pi * 0.5);
        return ceiling * fastSin(std::clamp(sample / ceiling, -halfPi, halfPi));
    }
    else if constexpr (Curve == ClipperCurve::GoldenRatio)
    {
        constexpr T phi = static_cast<T>(1.6180339887498948482);
        const T threshold = ceiling / phi;
        const T absSample = std::abs(sample);
        if (absSample <= threshold)
            return sample;
        const T sign = std::copysign(T(1), sample);
        return sign * rationalKneeMagnitude(absSample, threshold, ceiling);
    }
    else
    {
        const auto limits = clipperLinearLimits<Curve>(ceiling);
        const T threshold = sample >= T(0) ? limits[1] : -limits[0];
        return std::copysign(rationalKneeMagnitude(std::abs(sample), threshold, ceiling), sample);
    }
}
} // namespace dspark::detail
