// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineConvolutionWindow.h
 * @brief Exact three-leaf Toeplitz convolution on one requested output leaf.
 * Uses Core FFTReal and complex multiplication; no stream state or PCM history.
 * Threading: one offline worker owns each instance. Setup reserves all storage.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "../FFT.h"
#include "../SimdOps.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

namespace dspark::detail
{
class OfflineConvolutionWindow final
{
  public:
    // For input x[0,3B), return y[i] = sum_j h[B+i-j]*x[j], i in [0,B).
    // Only lags [-2B+1,2B-1] occur. A 4B circular embedding represents every
    // one of them without collision. Other output positions are discarded.
    template <class Impulse>
    OfflineConvolutionWindow(OfflineSession &job, int block, Impulse impulse) : block_(block)
    {
        if (block < 128 || block > 65536 || (block & (block - 1)))
            offlineFail(OfflineStatus::InvalidInput);
        const auto size = 4 * static_cast<std::uint64_t>(block);
        time_ = job.allocateScratch<double>(size + 2);
        spectrum_ = job.allocateScratch<double>(size + 2);
        kernel_ = job.allocateScratch<double>(size + 2);
        // The Core real FFT requests <4N doubles. Include cumulative Stockham
        // plan-vector growth and implementation-specific allocation bookkeeping.
        job.charge(offlineBytes(4 * size, sizeof(double)) + 2048);
        fft_.emplace(static_cast<std::size_t>(size));
        for (int i = 0; i < 4 * block; ++i)
        {
            const int lag = i < 2 * block ? i : i - 4 * block;
            const double value = i == 2 * block ? 0 : impulse(lag);
            if (!std::isfinite(value))
                offlineFail(OfflineStatus::NonFiniteInput);
            time_[i] = value;
        }
        fft_->forward(time_.get(), kernel_.get());
        for (std::uint64_t i = 0; i < size + 2; ++i)
            if (!std::isfinite(kernel_[i]))
                offlineFail(OfflineStatus::NumericalFailure);
    }
    OfflineConvolutionWindow(const OfflineConvolutionWindow &) = delete;
    OfflineConvolutionWindow &operator=(const OfflineConvolutionWindow &) = delete;

    // Fill all 3B finite, normalized input values before each process call.
    [[nodiscard]] double *input() noexcept
    {
        return time_.get();
    }
    [[nodiscard]] const double *process() noexcept
    {
        const int size = 4 * block_;
        std::fill_n(time_.get() + 3 * block_, block_ + 2, 0.);
        fft_->forward(time_.get(), spectrum_.get());
        std::fill_n(time_.get(), size + 2, 0.);
        simd::complexMulAccum(time_.get(), spectrum_.get(), kernel_.get(), size / 2 + 1);
        fft_->inverse(time_.get(), time_.get());
        return time_.get() + block_;
    }

  private:
    int block_;
    OfflineScratchArray<double> time_, spectrum_, kernel_;
    std::optional<FFTReal<double>> fft_;
};
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
