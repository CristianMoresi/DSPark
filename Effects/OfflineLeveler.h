// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflineLeveler.h
 * @brief Automatic, upward macro RMS leveling over a complete mono/stereo source.
 *
 * OFFLINE ONLY. This is not a causal compressor or a real-time callback processor.
 * A complete-source energy map selects the strongest sustained 3 s input window
 * as reference. Slow, stereo-linked gain raises quieter active passages, with a
 * 24 dB maximum correction by default. Its gain plan has no downward correction,
 * makeup gain, or peak limiter. Floating-point output may exceed 0 dBFS;
 * render reports sample and true peak, including the interpolator tail.
 *
 * Exclusions preserve PCM exactly; transition feathers lie outside their bounds.
 * The scalar-plan energy bound covers 3 s windows on the 100 ms analysis grid,
 * using actual final-bin duration. Bandlimited interpolation can change those
 * energies and individual sample magnitudes. Analysis keeps O(duration/100 ms)
 * storage; rendering uses bounded PCM blocks and compact interpolation maps.
 * run(input,output,...) additionally stages one owning output for atomic delivery.
 *
 * Threading: all work is synchronous and offline, on a worker owned by the caller.
 * This stateless facade can serve independent jobs concurrently. Each job owns its
 * source/sink; options remain immutable for the call. A completed Plan is read-only
 * and can be shared while alive and unmoved. Nothing is implicitly published to an
 * audio thread. No processSample/processBlock real-time interface is provided.
 *
 * Dependencies: OfflineEnergyAnalyzer.h, OfflineProcessing.h, TruePeakDetector.h,
 * AudioBuffer.h and DspMath.h. Decibel math and true-peak interpolation are shared
 * with existing DSPark processors; no alternate loudness or limiter is embedded.
 */

#include "../Analysis/OfflineEnergyAnalyzer.h"
#include "../Core/OfflineProcessing.h"

#if DSPARK_HAS_OFFLINE

#include "../Core/detail/OfflineGain.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace dspark
{

/** @brief Complete-file automatic upward leveling, with reusable immutable plans. */
template <FloatType T> class OfflineLeveler final
{
  public:
    /** @brief Principal amount plus optional expert controls and protected regions. */
    struct Options
    {
        double amount = 0;          ///< 0..1, fraction of the measured level difference in dB.
        double speed = 0.5;         ///< 0..1: 6..2 s context, 1.2..0.2 s transitions.
        double maximumBoostDb = 24; ///< Finite 0..24 dB cap; no downward correction.
        std::span<const OfflineRegion> exclusions; ///< Copied and canonicalized into the plan.
    };

    /** @brief Why the computed plan changed, limited, or left the source alone. */
    enum class Reason : std::uint8_t
    {
        Leveled,
        BoostLimited,
        RepresentabilityLimited,
        ZeroAmount,
        BoostDisabled,
        NoActivity,
        AllExcluded,
        NoCorrection
    };

    /** @brief Plan range and final render peaks. Silence peaks are -infinity dB. */
    struct Report
    {
        Reason reason = Reason::NoCorrection;
        double referenceRmsDb = -std::numeric_limits<double>::infinity();
        double minimumGainDb = 0;
        double maximumGainDb = 0;
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        std::size_t controlPoints = 0;
        std::size_t boundaries = 0;
        std::size_t limitedPoints = 0;
        std::size_t representabilityLimitedBins = 0; ///< Correction reduced to keep PCM finite.
        bool peaksMeasured = false;
        OfflineGainRenderInfo renderInfo;
    };

    /** @brief Immutable source-bound linear-amplitude automation, with int64 lookup. */
    class Plan final
    {
      public:
        Plan() = default;
        Plan(Plan &&) noexcept = default;
        Plan &operator=(Plan &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept
        {
            return gains_ != nullptr;
        }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept
        {
            return spec_;
        }
        [[nodiscard]] Report getReport() const noexcept
        {
            return report_;
        }
        [[nodiscard]] std::size_t controlPoints() const noexcept
        {
            return isValid() ? count_ : 0;
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid() ? count_ * sizeof(double) + exclusions_.retainedBytes() : 0;
        }
        /** @brief Borrowed read-only regions; invalidated by destruction/move of the plan. */
        [[nodiscard]] std::span<const OfflineRegion> exclusions() const noexcept
        {
            return isValid() ? exclusions_.view() : std::span<const OfflineRegion>();
        }
        /** @brief Gain at a source-relative frame. Outside the source or on an invalid plan: 1. */
        [[nodiscard]] double gainAt(std::int64_t frame) const noexcept
        {
            if (!isValid() || frame < 0 || frame >= spec_.frames)
                return 1;
            const auto index = static_cast<std::size_t>(frame / hop_);
            const double fraction = static_cast<double>(frame % hop_) / static_cast<double>(hop_);
            const double gain = std::lerp(gains_[index], gains_[index + 1], fraction);
            return exclusions_.apply(frame, gain, featherFrames_);
        }

      private:
        friend class OfflineLeveler;
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_;
        Report report_;
        std::int64_t hop_ = 1;
        std::size_t count_ = 0;
        double featherFrames_ = 1, inputPeak_ = 0;
        std::unique_ptr<double[]> gains_;
        detail::OfflineExclusions exclusions_;
    };

    /** @brief Completed plan or explicit failure, never an old/stale plan. */
    struct PlanResult
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Plan plan;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept
        {
            return offlineSucceeded(status);
        }
    };

    /** @brief Render result, measurements, and the operation's requested dynamic payload. */
    struct Result
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Report report;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept
        {
            return offlineSucceeded(status);
        }
    };

    /** @brief Builds/rebuilds a plan from reusable energy features without reading PCM. */
    [[nodiscard]] PlanResult makePlan(const typename OfflineEnergyAnalyzer<T>::Analysis &analysis,
                                      const Options &options = {},
                                      const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            if (!analysis.isValid())
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            Plan plan;
            plan.spec_ = analysis.getSpec();
            detail::offlineValidateSpec(plan.spec_, 2);
            plan.fingerprint_ = analysis.fingerprint();
            plan.hop_ = analysis.binFrames();
            plan.inputPeak_ = analysis.samplePeak();
            const auto bins = analysis.bins();
            const auto n = bins.size();
            const auto total = static_cast<std::int64_t>(n + 1) * 3;
            job.checkpoint(OfflinePhase::Plan, 0, total);
            plan.count_ = n + 1;
            plan.gains_ = job.allocate<double>(n + 1);
            std::fill_n(plan.gains_.get(), n + 1, 1.0);
            plan.exclusions_.assign(job, options.exclusions, plan.spec_.frames);
            plan.report_.controlPoints = n + 1;
            auto powers = job.allocate<double>(n);
            double scale = 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                powers[i] = bins[i].rms;
                scale = std::max(scale, bins[i].rms);
            }
            const auto quantile = (n - 1) / 10 * 9 + (n - 1) % 10 * 9 / 10;
            std::nth_element(powers.get(), powers.get() + quantile, powers.get() + n);
            const double floorDb = std::max(-65.0, gainToDecibels(powers[quantile], -6400.0) - 35);
            bool active = false;
            for (const auto bin : bins)
                active = active || gainToDecibels(bin.rms, -6400.0) > floorDb;

            const bool allExcluded = plan.exclusions_.covers(plan.spec_.frames);
            if (!active || options.amount == 0 || options.maximumBoostDb == 0 || allExcluded)
            {
                plan.report_.reason = allExcluded           ? Reason::AllExcluded
                                      : !active             ? Reason::NoActivity
                                      : options.amount == 0 ? Reason::ZeroAmount
                                                            : Reason::BoostDisabled;
                job.checkpoint(OfflinePhase::Plan, total, total);
                result.status = OfflineStatus::NoChange;
                result.memoryBytes = job.bytes();
                result.plan = std::move(plan);
                return result;
            }
            for (std::size_t i = 0; i < n; ++i)
            {
                const double ratio = bins[i].rms / scale;
                powers[i] = ratio * ratio;
            }
            const double binsPerSecond = plan.spec_.sampleRate / static_cast<double>(plan.hop_);
            const auto macro = std::min(
                n,
                std::max<std::size_t>(1, static_cast<std::size_t>(std::round(3 * binsPerSecond))));
            const auto context =
                std::max<std::size_t>(2, static_cast<std::size_t>(std::round(binsPerSecond)));
            // Local sums avoid subtracting two large cumulative sums after a loud
            // section. At 100 ms resolution all analysis windows have bounded size.
            const auto meanPower = [&](std::size_t from, std::size_t to) {
                double energy = 0, frames = 0;
                for (auto j = from; j < to; ++j)
                {
                    energy += powers[j] * static_cast<double>(bins[j].frames);
                    frames += static_cast<double>(bins[j].frames);
                }
                return energy / frames;
            };
            const auto meanDb = [&](std::size_t from, std::size_t to) {
                double localScale = 0, energy = 0, frames = 0;
                for (auto j = from; j < to; ++j)
                    localScale = std::max(localScale, bins[j].rms);
                if (localScale == 0)
                    return -6400.0;
                for (auto j = from; j < to; ++j)
                {
                    const double r = bins[j].rms / localScale;
                    energy += r * r * static_cast<double>(bins[j].frames);
                    frames += static_cast<double>(bins[j].frames);
                }
                return gainToDecibels(localScale * std::sqrt(std::min(1.0, energy / frames)),
                                      -6400.0);
            };
            double targetPower = 0;
            for (std::size_t i = 0; i <= n - macro; ++i)
                targetPower = std::max(targetPower, meanPower(i, i + macro));
            const double targetDb =
                gainToDecibels(scale * std::sqrt(std::min(1.0, targetPower)), -6400.0);
            plan.report_.referenceRmsDb = targetDb;
            auto novelty = job.allocate<double>(n + 1);
            auto edges = job.allocate<std::size_t>(n + 2);
            std::size_t edgeCount = 1;
            if (n >= 2 * context)
                for (auto i = context; i <= n - context; ++i)
                {
                    const auto half = context / 2;
                    const double left = meanDb(i - context, i), right = meanDb(i, i + context);
                    if (left > floorDb + 6 && right > floorDb + 6 && std::abs(right - left) >= 3 &&
                        std::abs(meanDb(i - context, i - half) - meanDb(i - half, i)) < 1.5 &&
                        std::abs(meanDb(i, i + half) - meanDb(i + half, i + context)) < 1.5)
                        novelty[i] = std::abs(right - left);
                }
            if (n >= 2 * context)
                for (auto i = context; i <= n - context; ++i)
                {
                    if (novelty[i] == 0)
                        continue;
                    bool maximum = true;
                    for (auto j = std::max(context, i - context);
                         j <= std::min(n - context, i + context); ++j)
                        if (novelty[j] > novelty[i] || (novelty[j] == novelty[i] && j < i))
                        {
                            maximum = false;
                            break;
                        }
                    if (maximum && i - edges[edgeCount - 1] >= context * 2)
                        edges[edgeCount++] = i;
                }
            edges[edgeCount++] = n;
            plan.report_.boundaries = edgeCount - 2;
            const auto radius = std::max<std::size_t>(
                1, static_cast<std::size_t>(
                       std::round(3 * std::pow(1.0 / 3, options.speed) * binsPerSecond)));
            const auto smoothing = std::max<std::size_t>(
                1, static_cast<std::size_t>(
                       std::round(0.6 * std::pow(1.0 / 6, options.speed) * binsPerSecond)));
            plan.featherFrames_ =
                std::max(plan.spec_.sampleRate * 0.1,
                         static_cast<double>(smoothing) * static_cast<double>(plan.hop_));
            auto desired = job.allocate<double>(n + 1);
            std::size_t region = 0;
            for (std::size_t i = 0; i <= n; ++i)
            {
                if ((i & 1023) == 0)
                    job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i), total);
                while (region + 2 < edgeCount && i >= edges[region + 1])
                    ++region;
                auto from = std::max(edges[region], i > radius ? i - radius : 0);
                auto to = std::min(edges[region + 1], i + radius);
                if (to <= from)
                {
                    from = std::min(i, n - 1);
                    to = from + 1;
                }
                const double correction = std::max(0.0, targetDb - meanDb(from, to));
                double activity = std::clamp(
                    (gainToDecibels(bins[std::min(i, n - 1)].rms, -6400.0) - floorDb) / 6, 0.0,
                    1.0);
                activity = activity * activity * (3 - 2 * activity);
                if (correction > options.maximumBoostDb && activity > 0.99)
                    ++plan.report_.limitedPoints;
                desired[i] =
                    options.amount * std::min(options.maximumBoostDb, correction) * activity;
            }
            const auto smooth = [n, smoothing](const double *values, std::size_t i) {
                double sum = 0, weights = 0;
                const auto radiusSigned = static_cast<std::int64_t>(smoothing);
                for (auto j = -radiusSigned; j <= radiusSigned; ++j)
                {
                    const double w =
                        static_cast<double>(smoothing + 1) - static_cast<double>(std::abs(j));
                    const auto index = static_cast<std::size_t>(
                        std::clamp(static_cast<std::int64_t>(i) + j, std::int64_t(0),
                                   static_cast<std::int64_t>(n)));
                    sum += w * values[index];
                    weights += w;
                }
                return sum / weights;
            };
            for (std::size_t i = 0; i <= n; ++i)
                plan.gains_[i] = decibelsToGain(std::min(desired[i], smooth(desired.get(), i)));
            const double maxSample = static_cast<double>(std::numeric_limits<T>::max());
            const double maxGain = decibelsToGain(options.maximumBoostDb);
            for (std::size_t i = 0; i < n; ++i)
                if (bins[i].peak > maxSample / maxGain)
                {
                    const double bound =
                        std::max(1.0, std::nextafter(maxSample / bins[i].peak, 0.0));
                    if (std::max(plan.gains_[i], plan.gains_[i + 1]) > bound)
                        ++plan.report_.representabilityLimitedBins;
                    plan.gains_[i] = std::min(plan.gains_[i], bound);
                    plan.gains_[i + 1] = std::min(plan.gains_[i + 1], bound);
                }
            // Bound each grid window using the maximum endpoint gain of each bin.
            // Scaling only the extra gain preserves source dynamics and monotonic
            // validity of all previously bounded windows.
            for (std::size_t start = 0; start <= n - macro; ++start)
            {
                if ((start & 1023) == 0)
                    job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(n + 1 + start),
                                   total);
                const auto end = start + macro;
                double a = 0, b = 0, c = 0, frames = 0;
                for (auto j = start; j < end; ++j)
                {
                    const double energy = powers[j] * static_cast<double>(bins[j].frames);
                    const double extra = std::max(plan.gains_[j], plan.gains_[j + 1]) - 1;
                    a += energy * extra * extra;
                    b += 2 * energy * extra;
                    c += energy;
                    frames += static_cast<double>(bins[j].frames);
                }
                const double budget = std::max(0.0, targetPower * frames - c);
                if (a + b > budget && a + b > 0)
                {
                    const double factor = std::clamp(
                        a == 0 ? budget / b : 2 * budget / (b + std::sqrt(b * b + 4 * a * budget)),
                        0.0, 1.0);
                    for (auto j = start; j <= end; ++j)
                        plan.gains_[j] = 1 + (plan.gains_[j] - 1) * factor;
                }
            }
            auto lower = job.allocate<double>(n + 1);
            for (std::size_t i = 0; i <= n; ++i)
                lower[i] = std::min(
                    {plan.gains_[i], plan.gains_[i ? i - 1 : 0], plan.gains_[std::min(n, i + 1)]});
            double minimum = std::numeric_limits<double>::infinity(), maximum = 1;
            for (std::size_t i = 0; i <= n; ++i)
            {
                if ((i & 1023) == 0)
                    job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(2 * (n + 1) + i),
                                   total);
                plan.gains_[i] = std::max(1.0, std::min(plan.gains_[i], smooth(lower.get(), i)));
                minimum = std::min(minimum, plan.gains_[i]);
                maximum = std::max(maximum, plan.gains_[i]);
            }
            if (!plan.exclusions_.view().empty())
                minimum = 1;
            plan.report_.minimumGainDb = gainToDecibels(minimum);
            plan.report_.maximumGainDb = gainToDecibels(maximum);
            plan.report_.reason = plan.report_.representabilityLimitedBins
                                      ? Reason::RepresentabilityLimited
                                  : maximum == 1               ? Reason::NoCorrection
                                  : plan.report_.limitedPoints ? Reason::BoostLimited
                                                               : Reason::Leveled;
            job.checkpoint(OfflinePhase::Plan, total, total);
            result.status = maximum == 1 ? OfflineStatus::NoChange : OfflineStatus::Success;
            result.memoryBytes = job.bytes();
            result.plan = std::move(plan);
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Complete analysis plus planning; only the compact gain plan is retained. */
    [[nodiscard]] PlanResult analyze(OfflineAudioSource<T> &source, const Options &options = {},
                                     const OfflineJobOptions &job = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            auto energy = OfflineEnergyAnalyzer<T>().analyze(source, job);
            if (!energy.succeeded())
            {
                result.status = energy.status;
                return result;
            }
            result = makePlan(energy.analysis, options,
                              detail::offlineRemaining(job, energy.analysis.retainedBytes()));
            result.memoryBytes =
                std::max(energy.memoryBytes, result.memoryBytes + energy.analysis.retainedBytes());
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /**
     * @brief Verifies a source-bound plan, then renders transactionally to a sink.
     *
     * Complete-source product maps serve calibration, measurement and provisional PCM.
     * Every repeated PCM block read is checked against its preflight fingerprint.
     * Output measurement and provisional writes share a pass; failures abort the sink.
     * Progress cancellation is checked before commit; a successful commit is final.
     */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink,
                                const OfflineJobOptions &options = {}) const
    {
        auto result = detail::offlineRenderGain<Result>(
            source, plan.spec_, plan.fingerprint_, plan.inputPeak_, plan.isValid(),
            plan.report_.maximumGainDb != 0, plan.report_, sink, options,
            [&plan](detail::OfflineSession &) {
                return [&plan](std::int64_t frame) { return plan.gainAt(frame); };
            }, {&plan.exclusions_, plan.featherFrames_, false});
        if (result.succeeded() && result.report.renderInfo.representabilityLimited)
        {
            result.report.reason = Reason::RepresentabilityLimited;
            const auto adjusted = [&](double db) {
                return gainToDecibels(1 + result.report.renderInfo.deltaScale *
                                             (decibelsToGain(db) - 1));
            };
            result.report.minimumGainDb = adjusted(result.report.minimumGainDb);
            result.report.maximumGainDb = adjusted(result.report.maximumGainDb);
        }
        return result;
    }

    /**
     * @brief Convenience complete-file owning render; input and output may be the same buffer.
     *
     * Output changes only on success. Allocates one candidate output; use the
     * source/plan/sink interface to stage very long renders without resident PCM.
     */
    template <int MaxChannels>
    [[nodiscard]] Result run(const AudioBuffer<T, MaxChannels> &input,
                             AudioBuffer<T, MaxChannels> &output, double sampleRate,
                             const Options &options = {}, const OfflineJobOptions &job = {}) const
    {
        return detail::offlineRun(*this, input, output, sampleRate, options, job);
    }

  private:
    static void validateOptions(const Options &options)
    {
        if (!std::isfinite(options.amount) || (options.amount < 0) || (options.amount > 1) ||
            !std::isfinite(options.speed) || (options.speed < 0) || (options.speed > 1) ||
            !std::isfinite(options.maximumBoostDb) || (options.maximumBoostDb < 0) ||
            (options.maximumBoostDb > 24))
            detail::offlineFail(OfflineStatus::InvalidInput);
    }
};

} // namespace dspark

#endif // DSPARK_HAS_OFFLINE
