// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflinePunchEnvelope.h
 * @brief Internal sparse, source-aligned transient boost weights.
 * Threading: worker-owned setup and cursors; completed curves are immutable.
 */
#include "OfflineBoxSmoother.h"
#if DSPARK_HAS_OFFLINE
#include "../WindowFunctions.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace dspark::detail
{
class OfflinePunchEnvelope final
{
  public:
    void prepare(OfflineSession &job, std::size_t count, double rate)
    {
        if (!std::isfinite(rate) || rate < 8000 || rate > 384000)
            offlineFail(OfflineStatus::InvalidInput);
        count_ = count;
        capacity_ = std::max(std::size_t(1), count);
        events_ = job.allocate<Event>(capacity_);
        const auto frames = [rate](double seconds) {
            return static_cast<int>(std::llround(seconds * rate));
        };
        span_ = frames(.002);
        rise_ = frames(.005) - span_;
        fall_ = frames(.045) - span_;
        minimumHold_ = frames(.015);
        maximumHold_ = frames(.050);
        life_ = maximumHold_ + span_ + fall_;
        smoother_.prepare(job, span_, count != 0);
        if (count)
        {
            riseWindow_ = job.allocate<double>(static_cast<std::uint64_t>(2 * rise_ + 1));
            fallWindow_ = job.allocate<double>(static_cast<std::uint64_t>(2 * fall_ + 1));
            WindowFunctions<double>::hann(riseWindow_.get(), 2 * rise_ + 1, false);
            WindowFunctions<double>::hann(fallWindow_.get(), 2 * fall_ + 1, false);
        }
    }
    void set(std::size_t index, std::int64_t onset, std::int64_t duration)
    {
        if (index >= count_ || onset < 0 || duration <= 0 ||
            onset > std::numeric_limits<std::int64_t>::max() - life_ - span_ ||
            (index && onset <= events_[index - 1].onset))
            offlineFail(OfflineStatus::NumericalFailure);
        events_[index] = {
            onset, static_cast<int>(std::clamp(duration, static_cast<std::int64_t>(minimumHold_),
                                               static_cast<std::int64_t>(maximumHold_)))};
    }
    [[nodiscard]] bool isValid() const noexcept
    {
        return events_ != nullptr;
    }
    [[nodiscard]] std::size_t retainedBytes() const noexcept
    {
        return isValid()
                   ? capacity_ * sizeof(Event) + smoother_.retainedBytes() +
                         (count_
                              ? static_cast<std::size_t>(2 * (rise_ + fall_) + 2) * sizeof(double)
                              : 0)
                   : 0;
    }
    [[nodiscard]] double weightAt(std::int64_t frame) const noexcept
    {
        if (!isValid() || !count_)
            return 0;
        double weight = 0;
        for (int k = 0; k <= span_; ++k)
            weight += smoother_.weights()[k] * rawAt(frame + k, firstAt(frame + k));
        return std::clamp(weight, 0.0, 1.0);
    }
    class Cursor final
    {
      public:
        Cursor(const OfflinePunchEnvelope &curve, OfflineSession &job)
            : curve_(curve), smoother_(curve.smoother_, job)
        {
        }
        [[nodiscard]] double operator()(std::int64_t frame) noexcept
        {
            if (!curve_.isValid() || !curve_.count_)
                return 0;
            // Match the renderer's fixed source-clock blocks after any rewind.
            if (frame != next_ || (frame & 4095) == 0)
            {
                smoother_.reset();
                first_ = curve_.firstAt(frame);
                for (int k = 0; k < curve_.span_; ++k)
                    (void)advance(frame + k);
            }
            const double result = advance(frame + curve_.span_);
            next_ = frame + 1;
            return std::clamp(result, 0.0, 1.0);
        }

      private:
        double advance(std::int64_t frame) noexcept
        {
            while (first_ < curve_.count_ && curve_.events_[first_].onset < frame - curve_.life_)
                ++first_;
            return smoother_.advance(curve_.rawAt(frame, first_), frame);
        }
        const OfflinePunchEnvelope &curve_;
        OfflineBoxSmoother::Cursor smoother_;
        std::int64_t next_ = -1;
        std::size_t first_ = 0;
    };

  private:
    struct Event
    {
        std::int64_t onset = 0;
        int hold = 0;
    };
    [[nodiscard]] std::size_t firstAt(std::int64_t frame) const noexcept
    {
        std::size_t lo = 0, hi = count_;
        while (lo < hi)
        {
            const auto mid = lo + (hi - lo) / 2;
            if (events_[mid].onset < frame - life_)
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo;
    }
    [[nodiscard]] double rawAt(std::int64_t frame, std::size_t first) const noexcept
    {
        double weight = 0;
        for (auto i = first; i < count_; ++i)
        {
            const auto age = frame - events_[i].onset;
            if (age < -rise_)
                break;
            if (age < 0)
                weight = std::max(weight, riseWindow_[static_cast<std::size_t>(age + rise_)]);
            else if (age <= events_[i].hold + span_)
                return 1;
            else if (const auto elapsed = age - events_[i].hold - span_; elapsed < fall_)
                weight = std::max(weight, fallWindow_[static_cast<std::size_t>(fall_ + elapsed)]);
        }
        return weight;
    }
    std::unique_ptr<Event[]> events_;
    std::unique_ptr<double[]> riseWindow_, fallWindow_;
    OfflineBoxSmoother smoother_;
    std::size_t count_ = 0, capacity_ = 0;
    int span_ = 1, rise_ = 1, fall_ = 1, minimumHold_ = 1, maximumHold_ = 1, life_ = 3;
};
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
