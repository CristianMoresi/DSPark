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
 *
 * Both banks run as one pass each over tap-interleaved coefficients
 * (simd::firCorrelateBank / firCorrelateBankSum): the eight reconstruction
 * filters read each input sample once, and the eight projection filters read
 * the moments interleaved, so every load is a contiguous vector. The
 * reconstruction is bit-identical to eight separate FIRs; the projection sums
 * each bank as one multiply-add chain (relative difference to a tree-ordered
 * dot product about 1e-16). Fixed coefficients are shared read-only constant
 * tables; the histories live in the object. process/reset are
 * allocation-free, with one owner per stream.
 */
#include "../AudioBuffer.h"
#include "../SimdOps.h"
#include "ContinuousClip.h"
#include "DcBlock.h"
#include "StereoColorCoefficients.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace dspark::detail
{
namespace stereo_color_banks
{
namespace sc = stereo_color_coefficients;
inline constexpr int reconstructionTaps = 257, projectionTaps = 258;
// Tap-interleaved, time-reversed (correlation order: tap 0 meets the oldest
// input of the window), as a FIR stores its coefficients.
inline constexpr auto reconstruction = [] {
    std::array<double, std::size_t(reconstructionTaps) * sc::powers> out{};
    for (int j = 0; j < reconstructionTaps; ++j)
        for (int k = 0; k < sc::powers; ++k)
            out[std::size_t(j) * sc::powers + std::size_t(k)] =
                sc::reconstruction[std::size_t(k)][std::size_t(reconstructionTaps - 1 - j)];
    return out;
}();
inline constexpr auto projection = [] {
    std::array<std::array<double, std::size_t(projectionTaps) * sc::moments>, 4> out{};
    for (int s = 0; s < 4; ++s)
        for (int j = 0; j < projectionTaps; ++j)
            for (int k = 0; k < sc::moments; ++k)
                out[std::size_t(s)][std::size_t(j) * sc::moments + std::size_t(k)] =
                    sc::projection[std::size_t(s)][std::size_t(k)][std::size_t(projectionTaps - 1 - j)];
    return out;
}();
} // namespace stereo_color_banks

template <ClipperCurve C> class ContinuousStereoColor final
{
    static_assert(C == ClipperCurve::AsymmetricKnee || C == ClipperCurve::SymmetricKnee);
    static constexpr int count = stereo_color_coefficients::moments, block = 256;
    static constexpr int powers = stereo_color_coefficients::powers;
    static constexpr int inputHistory = stereo_color_banks::reconstructionTaps - 1;
    static constexpr int momentHistory = stereo_color_banks::projectionTaps - 1;
    continuous_clip::Interval<C, powers - 1, false, false, count> integral_{.08};
    // Input window: inputHistory past samples, then one block.
    alignas(64) std::array<double, inputHistory + block> input_{};
    // Moment window, interleaved by moment: momentHistory past frames, then one block.
    alignas(64) std::array<double, (momentHistory + block) * count> moments_{};
    alignas(64) std::array<std::array<double, block>, powers> scratch_{};
    alignas(64) std::array<double, block> projected_{};
    const double *projection_;
    DcBlockState dc_{};
    double pole_;

  public:
    [[nodiscard]] static constexpr int latency() noexcept
    {
        return stereo_color_coefficients::latency;
    }
    // Dynamic payload; the owning wrapper accounts for sizeof(*this). The
    // coefficient tables are shared constants and the histories live in the
    // object, so nothing else is allocated.
    [[nodiscard]] static constexpr std::size_t allocationBound() noexcept
    {
        return 0;
    }
    explicit ContinuousStereoColor(int sourceFactor) : pole_(std::pow(.9995, sourceFactor))
    {
        assert(sourceFactor == 1 || sourceFactor == 2 || sourceFactor == 4 || sourceFactor == 8);
        const int sourceIndex = sourceFactor == 8 ? 3 : sourceFactor == 4 ? 2 : sourceFactor == 2 ? 1 : 0;
        projection_ = stereo_color_banks::projection[std::size_t(sourceIndex)].data();
    }
    ContinuousStereoColor(const ContinuousStereoColor &) = delete;
    ContinuousStereoColor &operator=(const ContinuousStereoColor &) = delete;
    void reset() noexcept
    {
        dc_ = {};
        input_.fill(0);
        moments_.fill(0);
    }
    void process(AudioBufferView<double> data) noexcept
    {
        for (int offset = 0; offset < data.getNumSamples(); offset += block)
        {
            const int size = std::min(block, data.getNumSamples() - offset);
            auto *samples = data.getChannel(0) + offset;
            std::copy_n(samples, size, input_.data() + inputHistory);
            std::array<double *, powers> powerOut{};
            for (int k = 0; k < powers; ++k) powerOut[std::size_t(k)] = scratch_[std::size_t(k)].data();
            simd::firCorrelateBank<double, powers>(input_.data(), stereo_color_banks::reconstruction.data(),
                                                   stereo_color_banks::reconstructionTaps, powerOut.data(), size);
            std::copy_n(input_.data() + size, inputHistory, input_.data());
            double *frame = moments_.data() + std::size_t(momentHistory) * count;
            for (int i = 0; i < size; ++i)
            {
                std::array<double, powers> power{};
                for (int k = 0; k < powers; ++k) power[std::size_t(k)] = scratch_[std::size_t(k)][std::size_t(i)];
                const auto moments = integral_.polynomial(power);
                std::copy_n(moments.data(), count, frame + std::size_t(i) * count);
            }
            simd::firCorrelateBankSum<double, count>(moments_.data(), projection_,
                                                     stereo_color_banks::projectionTaps, projected_.data(), size);
            std::copy_n(moments_.data() + std::size_t(size) * count, std::size_t(momentHistory) * count,
                        moments_.data());
            for (int i = 0; i < size; ++i)
                samples[i] = dcBlockStep(pole_, dc_, projected_[std::size_t(i)]);
        }
    }
};
} // namespace dspark::detail
