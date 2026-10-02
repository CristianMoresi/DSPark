// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file ContinuousStereoColor.h
 * @brief Source-rate rational color using continuous interval integration.
 *
 * A FIR bank reconstructs a degree-seven polynomial on each sample interval.
 * The shared clipping kernel integrates its eight shaped moments, then a second
 * FIR bank projects them through the bandlimited output kernel. There is no
 * upsampled audio stream. Both curves and their knee isolation are shared with
 * the oversampled path. The fixed 0.45 Fs to 0.50 Fs transition is intentional.
 *
 * Projection also applies H_source/H_base for the source's .9995 DC pole;
 * the matching base-rate recurrence is evaluated once per sample. This avoids
 * approximating that transfer with a short Taylor FIR near base-rate Nyquist.
 * Fixed coefficients are shared read-only; setup allocates private FIR states.
 * process/reset are allocation-free, with one owner per stream.
 */
#include "../FIRFilter.h"
#include "ContinuousClip.h"
#include "DcBlock.h"
#include "StereoColorCoefficients.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace dspark::detail
{
template <ClipperCurve C> class ContinuousStereoColor final
{
    static_assert(C == ClipperCurve::AsymmetricKnee || C == ClipperCurve::SymmetricKnee);
    static constexpr int count = stereo_color_coefficients::moments, block = 256;
    static constexpr int powers = stereo_color_coefficients::powers;
    std::array<FIRFilter<double>, powers> input_;
    std::array<FIRFilter<double>, count> output_;
    continuous_clip::Interval<C, powers - 1, false, false, count> integral_{.08};
    std::array<std::array<double, block>, powers> scratch_{};
    DcBlockState dc_{};
    double pole_;

  public:
    [[nodiscard]] static constexpr int latency() noexcept
    {
        return stereo_color_coefficients::latency;
    }
    // Dynamic payload; the owning wrapper accounts for sizeof(*this).
    [[nodiscard]] static constexpr std::size_t allocationBound() noexcept
    {
        // Five vector requests per FIR. Include their implementation-specific
        // alignment bookkeeping (MSVC adds 39 bytes to large vector requests),
        // as well as the scalar payload; offline budgets count both.
        return (powers * 257 + count * 258) * (sizeof(std::atomic<double>) + 4 * sizeof(double)) +
               (powers + count) * (sizeof(int) + 5 * 64);
    }
    explicit ContinuousStereoColor(int sourceFactor) : pole_(std::pow(.9995, sourceFactor))
    {
        assert(sourceFactor == 1 || sourceFactor == 2 || sourceFactor == 4 || sourceFactor == 8);
        const int sourceIndex = sourceFactor == 8 ? 3 : sourceFactor == 4 ? 2 : sourceFactor == 2 ? 1 : 0;
        for (int k = 0; k < powers; ++k)
        {
            input_[k].prepare(257, 1);
            input_[k].setCoefficients(stereo_color_coefficients::reconstruction[k]);
        }
        for (int k = 0; k < count; ++k)
        {
            output_[k].prepare(258, 1);
            output_[k].setCoefficients(stereo_color_coefficients::projection[sourceIndex][k]);
        }
    }
    void reset() noexcept
    {
        dc_ = {};
        for (auto &filter : input_) filter.reset();
        for (auto &filter : output_) filter.reset();
    }
    void process(AudioBufferView<double> data) noexcept
    {
        for (int offset = 0; offset < data.getNumSamples(); offset += block)
        {
            const int size = std::min(block, data.getNumSamples() - offset);
            auto *samples = data.getChannel(0) + offset;
            for (int k = 0; k < powers; ++k)
            {
                auto *channel = scratch_[k].data();
                std::copy_n(samples, size, channel);
                input_[k].processBlock({&channel, 1, size});
            }
            for (int i = 0; i < size; ++i)
            {
                std::array<double, powers> power{};
                for (int k = 0; k < powers; ++k) power[k] = scratch_[k][i];
                const auto moments = integral_.polynomial(power);
                for (int k = 0; k < count; ++k) scratch_[k][i] = moments[k];
            }
            for (int k = 0; k < count; ++k)
            {
                auto *channel = scratch_[k].data();
                output_[k].processBlock({&channel, 1, size});
            }
            for (int i = 0; i < size; ++i)
            {
                double result = 0;
                for (int k = 0; k < count; ++k) result += scratch_[k][i];
                samples[i] = dcBlockStep(pole_, dc_, result);
            }
        }
    }
};
} // namespace dspark::detail
