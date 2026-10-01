// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineAttenuation.h
 * @brief Internal source-aligned minimum hold, release and four-box envelope.
 * Threading: construction is worker-owned; completed curves are immutable.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "OfflineBoxSmoother.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

namespace dspark::detail
{

// Extend each ordered required hold by L frames, then take the minimum gain
// and exponential release. At source frame p, four box averages read that raw
// envelope over [p,p+L]. Every term is <= required[p], so the average retains
// the peak bound. The four-box cascade smooths both attacks and changes of winner.
// Amplitude deficits make release prefixes independent of absolute gain.
// Optional local releases change the common decay clock at each extended hold
// end. All surviving tails share that clock; these are not independent tails
// with different slopes, for which a single maximum prefix would be invalid.
class OfflineAttenuation final
{
  public:
    struct Hold
    {
        std::int64_t begin = 0, end = 0;
        double deficit = 0, releasePrefix = 0;
    };
    void prepare(OfflineSession &job, std::size_t count, std::int64_t attackFrames,
                 double releaseFrames, bool localRelease = false)
    {
        if (attackFrames < 1 || attackFrames > 1000000 || !std::isfinite(releaseFrames) ||
            releaseFrames <= 0)
            offlineFail(OfflineStatus::InvalidInput);
        capacity_ = std::max(std::size_t(1), count);
        holds_ = job.allocate<Hold>(capacity_);
        releases_ = localRelease ? job.allocate<double>(capacity_) : nullptr;
        count_ = count;
        attack_ = attackFrames;
        release_ = releaseFrames;
        coefficient_ = std::exp(-1 / release_);
        smoother_.prepare(job, attack_, count_ != 0);
        maximumDeficit_ = 0;
    }
    void set(std::size_t index, std::int64_t begin, std::int64_t end, double gain,
             double localReleaseFrames = 0)
    {
        if (index >= count_ || begin < 0 || end <= begin ||
            end > std::numeric_limits<std::int64_t>::max() - attack_ || !std::isfinite(gain) ||
            gain <= 0 || gain > 1 ||
            (index && (begin < holds_[index - 1].begin || end < holds_[index - 1].end)) ||
            (releases_ && (!std::isfinite(localReleaseFrames) || localReleaseFrames <= 0)))
            offlineFail(OfflineStatus::NumericalFailure);
        if (releases_)
            releases_[index] = localReleaseFrames;
        auto &hold = holds_[index];
        hold = {begin, end, 1 - gain, 1 - gain};
        maximumDeficit_ = std::max(maximumDeficit_, hold.deficit);
        if (index)
        {
            const auto &previous = holds_[index - 1];
            hold.releasePrefix =
                std::max(hold.deficit, previous.releasePrefix * decay(end - previous.end, index - 1));
        }
    }
    [[nodiscard]] bool isValid() const noexcept
    {
        return holds_ != nullptr;
    }
    [[nodiscard]] std::size_t retainedBytes() const noexcept
    {
        return isValid() ? capacity_ * (sizeof(Hold) + (releases_ ? sizeof(double) : 0)) +
                               smoother_.retainedBytes() : 0;
    }
    // Keep event holds and release constants; move the smoothed attack earlier.
    // Offline peak calibration can require more anticipation to avoid ringing
    // exceeding the target after baseband projection.
    [[nodiscard]] OfflineAttenuation withAttack(OfflineSession &job,
                                                 std::int64_t attackFrames) const
    {
        if (!isValid())
            offlineFail(OfflineStatus::InvalidInput);
        OfflineAttenuation result;
        result.prepare(job, count_, attackFrames, release_, releases_ != nullptr);
        for (std::size_t i = 0; i < count_; ++i)
            result.set(i, holds_[i].begin, holds_[i].end, 1 - holds_[i].deficit,
                       releases_ ? releases_[i] : 0);
        return result;
    }
    // Random access uses the finite box support; streaming rendering uses the
    // cursor below, with constant amortized work and O(lookahead) scratch.
    [[nodiscard]] double gainAt(std::int64_t frame) const noexcept
    {
        if (!isValid() || count_ == 0)
            return 1;
        double sum = 0;
        for (std::int64_t k = 0; k <= attack_; ++k)
            sum += smoother_.weights()[static_cast<std::size_t>(k)] * rawAt(frame + k);
        return gain(sum);
    }
    class Cursor final
    {
      public:
        Cursor(const OfflineAttenuation &curve, OfflineSession &job)
            : curve_(curve), smoother_(curve.smoother_, job)
        {
        }
        [[nodiscard]] double operator()(std::int64_t frame) noexcept
        {
            if (!curve_.isValid() || curve_.count_ == 0)
                return 1;
            // Fixed source-clock anchors make a cached block identical whether
            // reached sequentially or after a rewind for another channel/map.
            if (frame != next_ || (frame & 4095) == 0)
            {
                smoother_.reset();
                tail_ = 0;
                ended_ = 0;
                tailEnd_ = std::numeric_limits<std::size_t>::max();
                for (std::int64_t k = 0; k < curve_.attack_; ++k)
                    advance(frame + k);
            }
            const double deficit = advance(frame + curve_.attack_);
            next_ = frame + 1;
            return curve_.gain(deficit);
        }

      private:
        double advance(std::int64_t frame) noexcept
        {
            while (ended_ < curve_.count_ && curve_.holds_[ended_].end + curve_.attack_ <= frame)
                ++ended_;
            if (ended_ != tailEnd_ || (frame & 4095) == 0)
            {
                tail_ =
                    ended_
                        ? curve_.holds_[ended_ - 1].releasePrefix *
                              curve_.decay(frame - curve_.holds_[ended_ - 1].end - curve_.attack_,
                                           ended_ - 1)
                        : 0;
                coefficient_ = curve_.releases_ && ended_
                                   ? std::exp(-1 / curve_.releases_[ended_ - 1])
                                   : curve_.coefficient_;
            }
            else
                tail_ *= coefficient_;
            tailEnd_ = ended_;
            const double raw = curve_.active(frame, ended_, tail_);
            return smoother_.advance(raw, frame);
        }
        const OfflineAttenuation &curve_;
        OfflineBoxSmoother::Cursor smoother_;
        std::int64_t next_ = -1;
        std::size_t ended_ = 0, tailEnd_ = std::numeric_limits<std::size_t>::max();
        double tail_ = 0, coefficient_ = 0;
    };

  private:
    [[nodiscard]] double decay(std::int64_t frames, std::size_t hold) const noexcept
    {
        const double time = static_cast<double>(frames) / (releases_ ? releases_[hold] : release_);
        return time >= 36 ? 0 : std::exp(-time);
    }
    [[nodiscard]] double gain(double deficit) const noexcept
    {
        return deficit <= 2.32e-16 ? 1 : 1 - std::min(maximumDeficit_, deficit);
    }
    [[nodiscard]] double active(std::int64_t frame, std::size_t ended, double tail) const noexcept
    {
        double deficit = tail;
        for (auto i = ended; i < count_ && holds_[i].begin <= frame; ++i)
            deficit = std::max(deficit, holds_[i].deficit);
        return deficit <= 2.32e-16 ? 0 : deficit;
    }
    [[nodiscard]] double rawAt(std::int64_t frame) const noexcept
    {
        std::size_t lo = 0, hi = count_;
        while (lo < hi)
        {
            const auto mid = lo + (hi - lo) / 2;
            if (holds_[mid].end + attack_ <= frame)
                lo = mid + 1;
            else
                hi = mid;
        }
        return active(
            frame, lo,
            lo ? holds_[lo - 1].releasePrefix * decay(frame - holds_[lo - 1].end - attack_, lo - 1)
               : 0);
    }
    std::unique_ptr<Hold[]> holds_;
    std::unique_ptr<double[]> releases_;
    std::size_t count_ = 0, capacity_ = 0;
    std::int64_t attack_ = 1;
    OfflineBoxSmoother smoother_;
    double release_ = 1, coefficient_ = 0, maximumDeficit_ = 0;
};
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
