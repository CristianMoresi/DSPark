// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflinePunch.h
 * @brief Automatic complete-source, stereo-linked transient boost.
 *
 * OFFLINE ONLY. Attack events receive an equal boost in dB, independent of
 * their strength and tempo. The source-aligned envelope rises during 5 ms
 * before the onset, holds for the event duration clamped to 15..50 ms, and
 * falls over 45 ms. Overlaps use the maximum weight, never summed boosts.
 * A positive four-box kernel smooths cosine flanks and their intersections;
 * its 2 ms support is included in those timings, without delaying audio.
 *
 * Reuses complete-source transient analysis, Hann windows, shared gain
 * transport/exclusions and TruePeakDetector. There is no output normalization,
 * limiter, audio oversampling, or PCM-sized gain map. Exclusions preserve PCM
 * exactly, with 5 ms feathers outside their bounds. Boost is reduced only when
 * the floating-point sample type could overflow; the report exposes that cap.
 * Bandlimited multiplication can produce interpolation tails outside the control
 * envelope; exact PCM is guaranteed for zero amount and explicit exclusions.
 *
 * Threading: synchronous worker operations, not audio callbacks. The facade
 * is stateless; a completed Plan is immutable and shareable while alive and
 * unmoved. Source, sink and options remain caller-owned.
 */
#include "../Analysis/OfflineTransientAnalyzer.h"
#if DSPARK_HAS_OFFLINE
#include "../Core/detail/OfflineGain.h"
#include "../Core/detail/OfflinePunchEnvelope.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace dspark
{
/** @brief Offline transient emphasis with a positive boost-dB control. */
template <FloatType T> class OfflinePunch final
{
  public:
    struct Options
    {
        double boostDb = 0; ///< Equal transient boost, 0..12 dB; zero is exact identity.
        std::span<const OfflineRegion> exclusions{}; ///< Exact PCM inside these source regions.
    };
    enum class Reason : std::uint8_t
    {
        Boosted,
        RepresentabilityLimited,
        ZeroAmount,
        NoSignal,
        NoTransients,
        AllExcluded
    };
    struct Report
    {
        Reason reason = Reason::NoTransients;
        double requestedBoostDb = 0, effectiveBoostDb = 0;
        double inputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        std::size_t transients = 0;
        bool peaksMeasured = false;
        OfflineGainRenderInfo renderInfo;
    };
    /** @brief Immutable sparse weight plan bound to a complete source fingerprint. */
    class Plan final
    {
      public:
        Plan() = default;
        Plan(Plan &&) noexcept = default;
        Plan &operator=(Plan &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept
        {
            return curve_.isValid();
        }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept
        {
            return spec_;
        }
        [[nodiscard]] Report getReport() const noexcept
        {
            return report_;
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid() ? curve_.retainedBytes() + exclusions_.retainedBytes() : 0;
        }
        /** @brief Borrowed regions, valid while the plan remains alive and unmoved. */
        [[nodiscard]] std::span<const OfflineRegion> exclusions() const noexcept
        {
            return isValid() ? exclusions_.view() : std::span<const OfflineRegion>();
        }
        [[nodiscard]] double gainAt(std::int64_t frame) const noexcept
        {
            if (!isValid() || frame < 0 || frame >= spec_.frames)
                return 1;
            return gain(frame, curve_.weightAt(frame));
        }

      private:
        friend class OfflinePunch;
        [[nodiscard]] double gain(std::int64_t frame, double weight) const noexcept
        {
            return exclusions_.apply(frame, decibelsToGain(report_.effectiveBoostDb * weight),
                                     featherFrames_);
        }
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_;
        Report report_;
        double inputPeak_ = 0, featherFrames_ = 1;
        detail::OfflinePunchEnvelope curve_;
        detail::OfflineExclusions exclusions_;
    };
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
    /** @brief Reuses attack events, then verifies PCM and bounds representable boost. */
    [[nodiscard]] PlanResult makePlan(
        OfflineAudioSource<T> &source,
        const typename OfflineTransientAnalyzer<T>::Analysis &analysis, const Options &options = {},
        const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            if (!analysis.isValid() || (options.boostDb > 0 && !analysis.getOptions().attacks))
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            Plan plan;
            plan.spec_ = analysis.energy().getSpec();
            if (source.getSpec() != plan.spec_)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            plan.fingerprint_ = analysis.energy().fingerprint();
            plan.inputPeak_ = analysis.energy().samplePeak();
            plan.featherFrames_ = std::max(1.0, std::round(.005 * plan.spec_.sampleRate));
            auto &report = plan.report_;
            report.requestedBoostDb = options.boostDb;
            report.inputSamplePeakDb =
                gainToDecibels(plan.inputPeak_, -std::numeric_limits<double>::infinity());
            job.checkpoint(OfflinePhase::Plan, 0, plan.spec_.frames);
            plan.exclusions_.assign(job, options.exclusions, plan.spec_.frames);
            const bool allExcluded = plan.exclusions_.covers(plan.spec_.frames);
            const auto attacks = analysis.attacks();
            const bool active =
                options.boostDb > 0 && plan.inputPeak_ > 0 && !attacks.empty() && !allExcluded;
            plan.curve_.prepare(job, active ? attacks.size() : 0, plan.spec_.sampleRate);
            if (!active)
                report.reason = allExcluded            ? Reason::AllExcluded
                                : options.boostDb == 0 ? Reason::ZeroAmount
                                : plan.inputPeak_ == 0 ? Reason::NoSignal
                                                       : Reason::NoTransients;
            else
            {
                report.transients = attacks.size();
                report.effectiveBoostDb = options.boostDb;
                report.reason = Reason::Boosted;
                for (std::size_t i = 0; i < attacks.size(); ++i)
                {
                    if ((i & 1023) == 0)
                        job.checkpoint(OfflinePhase::Plan, 0, plan.spec_.frames);
                    plan.curve_.set(i, attacks[i].begin, attacks[i].end - attacks[i].begin);
                }
                detail::OfflineBlock<T> scratch(job, plan.spec_, jobOptions.blockFrames);
                detail::OfflinePunchEnvelope::Cursor weights(plan.curve_, job);
                constexpr double maximum = static_cast<double>(std::numeric_limits<T>::max());
                detail::offlineScanMagnitude(
                    source, plan.spec_, plan.fingerprint_, plan.inputPeak_, scratch, job,
                    [&](std::int64_t frame, double magnitude) {
                        const double weight = weights(frame);
                        const double gain = plan.gain(frame, weight);
                        if (magnitude > maximum / gain)
                        {
                            const double mask =
                                plan.exclusions_.apply(frame, 2, plan.featherFrames_) - 1;
                            const double cap = 20 / std::log(10.) *
                                               std::log1p((maximum / magnitude - 1) / mask) /
                                               weight;
                            // Reserve roundoff before evaluating exp and the feather again.
                            report.effectiveBoostDb =
                                std::max(0.0, cap - 64 * std::numeric_limits<double>::epsilon() *
                                                        std::max(1.0, cap));
                            report.reason = Reason::RepresentabilityLimited;
                        }
                    });
            }
            job.checkpoint(OfflinePhase::Plan, plan.spec_.frames, plan.spec_.frames);
            result.status =
                report.effectiveBoostDb > 0 ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
            result.plan = std::move(plan);
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }
    /** @brief Complete-source analysis and planning; retains only sparse automation. */
    [[nodiscard]] PlanResult analyze(OfflineAudioSource<T> &source, const Options &options = {},
                                     const OfflineJobOptions &job = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            typename OfflineTransientAnalyzer<T>::Options features;
            features.attacks = options.boostDb > 0;
            features.pulses = false;
            auto analyzed = OfflineTransientAnalyzer<T>().analyze(source, features, job);
            if (!analyzed.succeeded())
            {
                result.status = analyzed.status;
                return result;
            }
            result = makePlan(source, analyzed.analysis, options,
                              detail::offlineRemaining(job, analyzed.analysis.retainedBytes()));
            result.memoryBytes = std::max(analyzed.memoryBytes,
                                          result.memoryBytes + analyzed.analysis.retainedBytes());
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }
    /** @brief Verifies the source, then measures provisional PCM before committing the sink. */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink, const OfflineJobOptions &job = {}) const
    {
        auto result = detail::offlineRenderGain<Result>(
            source, plan.spec_, plan.fingerprint_, plan.inputPeak_, plan.isValid(),
            plan.report_.effectiveBoostDb > 0, plan.report_, sink, job,
            [&](detail::OfflineSession &session) {
                return
                    [&plan, weights = detail::OfflinePunchEnvelope::Cursor(plan.curve_, session)](
                        std::int64_t frame) mutable { return plan.gain(frame, weights(frame)); };
            }, {&plan.exclusions_, plan.featherFrames_, false});
        if (result.succeeded() && result.report.renderInfo.representabilityLimited)
        {
            result.report.reason = Reason::RepresentabilityLimited;
            result.report.effectiveBoostDb = gainToDecibels(
                1 + result.report.renderInfo.deltaScale *
                        (decibelsToGain(result.report.effectiveBoostDb) - 1));
        }
        return result;
    }
    /** @brief Owning convenience render; aliases are safe and failures preserve output. */
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
        if (!std::isfinite(options.boostDb) || (options.boostDb < 0) || (options.boostDb > 12))
            detail::offlineFail(OfflineStatus::InvalidInput);
    }
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
