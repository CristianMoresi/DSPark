// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineBoxSmoother.h
 * @brief Internal finite four-box kernel shared by source-aligned gain plans.
 * Threading: worker-owned setup and cursors; a prepared kernel is immutable.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "BoxAverage.h"
#include <algorithm>
#include <array>
#include <memory>

namespace dspark::detail
{
class OfflineBoxSmoother final
{
  public:
    void prepare(OfflineSession &job, std::int64_t span, bool active)
    {
        if (span < 1 || span > 1000000)
            offlineFail(OfflineStatus::InvalidInput);
        span_ = span;
        double normalization = 1;
        for (int i = 0; i < 4; ++i)
        {
            lengths_[i] = static_cast<int>(span_ / 4 + (i < span_ % 4) + 1);
            normalization *= lengths_[i];
        }
        if (active)
        {
            weights_ = job.allocate<double>(static_cast<std::uint64_t>(span_ + 1));
            // Coefficients of four finite geometric series, by inclusion-exclusion.
            // Integer arithmetic preserves exact nonnegative tap counts; the
            // setup bound above keeps all cubic products and sums within int64.
            for (std::int64_t k = 0; k <= span_; ++k)
            {
                std::int64_t ways = 0;
                for (unsigned mask = 0; mask < 16; ++mask)
                {
                    std::int64_t remaining = k;
                    int sign = 1;
                    for (int i = 0; i < 4; ++i)
                        if (mask & (1u << i))
                        {
                            remaining -= lengths_[i];
                            sign = -sign;
                        }
                    if (remaining >= 0)
                        ways += sign * ((remaining + 1) * (remaining + 2) * (remaining + 3) / 6);
                }
                weights_[static_cast<std::size_t>(k)] = static_cast<double>(ways) / normalization;
            }
        }
    }
    [[nodiscard]] std::int64_t span() const noexcept
    {
        return span_;
    }
    [[nodiscard]] const double *weights() const noexcept
    {
        return weights_.get();
    }
    [[nodiscard]] std::size_t retainedBytes() const noexcept
    {
        return weights_ ? static_cast<std::size_t>(span_ + 1) * sizeof(double) : 0;
    }
    class Cursor final
    {
      public:
        Cursor(const OfflineBoxSmoother &filter, OfflineSession &job) : filter_(filter)
        {
            if (!filter.weights_)
                return;
            storage_ = job.allocate<double>(static_cast<std::uint64_t>(filter.span_ + 4));
            int offset = 0;
            for (int i = 0; i < 4; ++i)
            {
                boxes_[i] = storage_.get() + offset;
                inverse_[i] = 1.0 / filter.lengths_[i];
                offset += filter.lengths_[i];
            }
        }
        void reset() noexcept
        {
            if (storage_)
                std::fill_n(storage_.get(), static_cast<std::size_t>(filter_.span_ + 4), 0.0);
            sums_.fill(0);
            indices_.fill(0);
            zeroRun_ = 0;
            boxesCleared_ = false;
        }
        [[nodiscard]] double advance(double raw, std::int64_t frame) noexcept
        {
            double value = raw;
            for (int i = 0; i < 4; ++i)
            {
                const double incoming = value;
                value = advanceBoxAverage(incoming, boxes_[i][indices_[i]], sums_[i], inverse_[i]);
                boxes_[i][indices_[i]] = incoming;
                if (++indices_[i] == filter_.lengths_[i])
                    indices_[i] = 0;
            }
            zeroRun_ = raw == 0 ? std::min(zeroRun_ + 1, static_cast<int>(filter_.span_ + 1)) : 0;
            if (raw != 0)
                boxesCleared_ = false;
            if (zeroRun_ > filter_.span_)
            {
                sums_.fill(0);
                value = 0;
                // All mathematical history entries are zero. Clear roundoff
                // once, so long silent spans stay constant-work exact identity.
                if (!boxesCleared_)
                {
                    std::fill_n(storage_.get(), static_cast<std::size_t>(filter_.span_ + 4), 0.0);
                    boxesCleared_ = true;
                }
            }
            if ((frame & 4095) == 4095)
            {
                for (int i = 0; i < 4; ++i)
                {
                    sums_[i] = 0;
                    for (int k = 0; k < filter_.lengths_[i]; ++k)
                        sums_[i] += boxes_[i][k];
                }
                value = sums_[3] * inverse_[3];
            }
            return value;
        }

      private:
        const OfflineBoxSmoother &filter_;
        std::unique_ptr<double[]> storage_;
        std::array<double *, 4> boxes_{};
        std::array<double, 4> sums_{}, inverse_{};
        std::array<int, 4> indices_{};
        int zeroRun_ = 0;
        bool boxesCleared_ = false;
    };

  private:
    std::int64_t span_ = 1;
    std::array<int, 4> lengths_{{1, 1, 1, 1}};
    std::unique_ptr<double[]> weights_;
};
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
