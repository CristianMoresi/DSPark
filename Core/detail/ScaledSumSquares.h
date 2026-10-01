// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/** @file ScaledSumSquares.h
 * @brief Shared overflow/underflow-safe finite-sample RMS accumulator.
 * Threading: one stream owner. Fixed storage, no allocations.
 * Dependencies: C++20 standard library only.
 */
#include <algorithm>
#include <cmath>

namespace dspark::detail {

/** @brief Scaled sum of squares; caller validates finite input and counts values. */
class ScaledSumSquares
{
public:
    void reset() noexcept { scale_ = sum_ = 0; }
    void add(double value) noexcept
    {
        const double magnitude = std::abs(value);
        if (magnitude > scale_)
        {
            const double ratio = scale_ / magnitude;
            sum_ = 1 + sum_ * ratio * ratio;
            scale_ = magnitude;
        }
        else if (magnitude > 0)
        {
            const double ratio = magnitude / scale_;
            sum_ += ratio * ratio;
        }
    }
    [[nodiscard]] double peak() const noexcept { return scale_; }
    [[nodiscard]] double rms(double count) const noexcept
    {
        return count > 0 ? scale_ * std::sqrt(std::min(1.0, sum_ / count)) : 0;
    }
private:
    double scale_ = 0, sum_ = 0;
};
} // namespace dspark::detail
