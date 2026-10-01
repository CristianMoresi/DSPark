// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DcBlock.h
 * @brief Shared one-pole DC-blocking state and difference-form recurrence.
 *
 * Internal double-precision arithmetic, shared with DCBlocker. The caller owns
 * coefficient selection, denormal handling and reset. No allocations or shared
 * state; calls modifying the same state belong to one processing thread.
 */
namespace dspark::detail
{
/** @brief Previous input and output of the one-pole DC blocker. */
struct DcBlockState
{
    double x1 = 0.0;
    double y1 = 0.0;
};

/** @brief Applies (1 - z^-1) / (1 - pole * z^-1), with a pole in [0, 1). */
[[nodiscard]] inline double dcBlockStep(double pole, DcBlockState &state, double input) noexcept
{
    const double output = (input - state.x1) + pole * state.y1;
    state.x1 = input;
    state.y1 = output;
    return output;
}
} // namespace dspark::detail
