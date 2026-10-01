// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file BoxAverage.h
 * @brief Internal moving-average arithmetic shared by live and offline gain computers.
 * Threading: the caller owns the running sum and sample history.
 */
namespace dspark::detail
{
[[nodiscard]] inline double advanceBoxAverage(double incoming, double outgoing, double &sum,
                                              double inverseLength) noexcept
{
    sum += incoming - outgoing;
    return sum * inverseLength;
}
} // namespace dspark::detail
