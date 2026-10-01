// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file RetimedDcBlock.h
 * @brief Internal source-clock DC transfer using shared Core recurrence and FIRs.
 *
 * Setup allocates. Processing/reset are allocation-free and stream-owned.
 * Power-of-two working/source factors must lie in [1,16]. Factors below the
 * source clock use a finite half-angle FIR correction, not just a retuned pole.
 * The stereo path requires a working factor of at least 2 for that correction.
 */
#include "../FIRFilter.h"
#include "ContinuousClip.h"
#include "DcBlock.h"
#include <array>
#include <cassert>
#include <cmath>
#include <utility>
#include <vector>
namespace dspark::detail
{
class RetimedDcBlock
{
    using Series = std::vector<long double>;
    std::array<DcBlockState, 16> states_{};
    FIRFilter<double> filter_;
    int stride_ = 1, write_ = 0, latency_ = 0;
    double pole_ = .9995;
    static Series multiply(const Series &a, const Series &b)
    {
        Series result(a.size());
        for (std::size_t i = 0; i < a.size(); ++i)
            for (std::size_t j = 0; i + j < a.size(); ++j)
                result[i + j] += a[i] * b[j];
        return result;
    }
    static Series halfAngle(const Series &h)
    {
        auto square = multiply(h, h);
        Series root(h.size()), product(h.size()), result(h.size());
        product[0] = 1;
        for (std::size_t i = 1; i < h.size(); ++i)
            product[i] = 4 * square[i - 1] - (i >= 2 ? 4 * square[i - 2] : 0);
        root[0] = 1;
        for (std::size_t i = 1; i < h.size(); ++i)
        {
            long double sum = 0;
            for (std::size_t j = 1; j < i; ++j)
                sum += root[j] * root[i - j];
            root[i] = (product[i] - sum) / 2;
        }
        root[0] += 1;
        for (std::size_t i = 0; i < h.size(); ++i)
        {
            long double value = h[i];
            for (std::size_t j = 1; j <= i; ++j)
                value -= root[j] * result[i - j];
            result[i] = value / root[0];
        }
        return result;
    }
    static std::vector<double> ratioTaps(const Series &h, double pole)
    {
        const int order = static_cast<int>(h.size()) - 1, center = order + 1;
        Series basis(static_cast<std::size_t>(2 * center + 1)), symmetric(basis.size());
        basis[center] = 1;
        for (int k = 0; k <= order; ++k)
        {
            for (std::size_t i = 0; i < basis.size(); ++i)
                symmetric[i] += h[k] * basis[i];
            if (k == order)
                break;
            Series next(basis.size());
            for (int j = center - k; j <= center + k; ++j)
            {
                next[j - 1] -= .25L * basis[j];
                next[j] += .5L * basis[j];
                next[j + 1] -= .25L * basis[j];
            }
            basis = std::move(next);
        }
        std::vector<double> taps(basis.size());
        taps[center] = (1 + pole) / 2;
        const long double b = (1 - pole) / 2;
        for (std::size_t i = 1; i + 1 < taps.size(); ++i)
        {
            taps[i - 1] += static_cast<double>(b * symmetric[i] / 2);
            taps[i + 1] -= static_cast<double>(b * symmetric[i] / 2);
        }
        return taps;
    }

  public:
    [[nodiscard]] static int filterLength(int factor, int sourceFactor) noexcept
    {
        if (factor >= sourceFactor)
            return 0;
        int stages = 0;
        for (int ratio = sourceFactor / factor; ratio > 1; ratio /= 2)
            ++stages;
        return 1 + 2 * stages * (factor <= 2 ? 25 : 13);
    }
    // Cumulative coefficients/temporary series/FIR requests, with allocation
    // alignment and debug-vector overhead. Factors have the constructor's bounds.
    [[nodiscard]] static std::size_t allocationBound(int factor, int sourceFactor) noexcept
    {
        const int length = filterLength(factor, sourceFactor);
        if (length == 0)
            return 0;
        const std::size_t order = factor <= 2 ? 24 : 12, n = order + 1, taps = 2 * n + 1;
        const auto stages = static_cast<std::size_t>((length - 1) / (2 * n));
        const auto series =
            (stages + 1 + 4 * (stages - 1)) * n * sizeof(long double) + stages * sizeof(Series);
        const auto kernels =
            stages * (order + 2) * taps * sizeof(long double) +
            (stages * taps + 1 + stages + n * stages * (stages + 1)) * sizeof(double);
        const auto fir =
            static_cast<std::size_t>(length) * (sizeof(std::atomic<double>) + 4 * sizeof(double)) +
            sizeof(int);
        return series + kernels + fir + (stages * order + 9 * stages + 4) * 64;
    }
    static std::vector<double> coefficients(int factor, int sourceFactor)
    {
        if (factor >= sourceFactor)
            return {1.};
        const int ratio = sourceFactor / factor, order = factor <= 2 ? 24 : 12;
        int stages = 0;
        while ((1 << stages) < ratio)
            ++stages;
        std::vector<Series> series(static_cast<std::size_t>(stages),
                                   Series(static_cast<std::size_t>(order + 1)));
        for (int k = 0; k <= order; ++k)
            series[0][k] = continuous_clip::choose(2 * k + 2, k + 1) / (2 * std::pow(4.L, k + 1));
        for (int i = 1; i < stages; ++i)
            series[i] = halfAngle(series[i - 1]);
        std::vector<double> result{1.};
        for (int i = stages - 1; i >= 0; --i)
        {
            const double pole = std::pow(.9995, ratio / (1 << (i + 1)));
            auto taps = ratioTaps(series[i], pole);
            std::vector<double> combined(result.size() + taps.size() - 1);
            for (std::size_t j = 0; j < result.size(); ++j)
                for (std::size_t k = 0; k < taps.size(); ++k)
                    combined[j + k] += result[j] * taps[k];
            result = std::move(combined);
        }
        return result;
    }
    RetimedDcBlock(int factor, int sourceFactor)
    {
        assert(factor >= 1 && factor <= 16 && (factor & (factor - 1)) == 0);
        assert(sourceFactor >= 1 && sourceFactor <= 16 && (sourceFactor & (sourceFactor - 1)) == 0);
        if (factor >= sourceFactor)
        {
            stride_ = factor / sourceFactor;
            return;
        }
        pole_ = std::pow(.9995, sourceFactor / factor);
        const auto taps = coefficients(factor, sourceFactor);
        latency_ = static_cast<int>(taps.size() / 2);
        filter_.prepare(static_cast<int>(taps.size()), 1);
        filter_.setCoefficients(taps);
    }
    [[nodiscard]] int latency() const noexcept
    {
        return latency_;
    }
    void reset() noexcept
    {
        states_ = {};
        write_ = 0;
        filter_.reset();
    }
    double process(double x) noexcept
    {
        const double y = dcBlockStep(pole_, states_[write_], x);
        write_ = (write_ + 1) % stride_;
        return latency_ ? filter_.processSample(y, 0) : y;
    }
};
} // namespace dspark::detail
