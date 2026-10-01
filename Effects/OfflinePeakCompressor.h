// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflinePeakCompressor.h
 * @brief Automatic source-aligned peak reduction with event-local hold.
 *
 * OFFLINE ONLY. Complete-source peak and sustained level select an absolute
 * sample ceiling. Each transient retains constant linked gain for its own
 * interval. Four moving averages reach that gain before the interval starts;
 * exponential amplitude recovery starts at its end. Overlaps use minimum gain.
 * Additional threshold excursions cover peaks outside detected events. No hard
 * clipping, makeup gain, audio delay, or oversampling is applied.
 *
 * Reuses OfflineTransientAnalyzer, EnvelopeFollower, DspMath and the verified
 * gain renderer shared with OfflineLeveler. Analysis and planning use bounded
 * source passes and sparse holds, never a complete PCM or per-sample gain array.
 * run() additionally stages an owning output. Exclusions preserve source PCM
 * exactly and can prevent the requested whole-file peak reduction.
 * Rendering calibrates the bandlimited processing delta. If its peak constraints
 * are infeasible, anticipation can extend up to 10 ms, retaining event holds and
 * the release constant. Report exposes actual anticipation and attempt count.
 *
 * Threading: synchronous worker operations, not audio callbacks. The stateless
 * facade serves independent jobs; a completed Plan is immutable and shareable
 * while alive and unmoved. Source, sink and options remain caller-owned.
 */

#include "../Analysis/OfflineTransientAnalyzer.h"
#if DSPARK_HAS_OFFLINE
#include "../Analysis/EnvelopeFollower.h"
#include "../Core/detail/OfflineAttenuation.h"
#include "../Core/detail/OfflineGain.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace dspark
{

/** @brief Complete-file peak compressor with a positive reduction-dB control. */
template <FloatType T> class OfflinePeakCompressor final
{
  public:
    struct Options
    {
        double reductionDb = 0; ///< Requested whole-file sample-peak reduction, 0..18 dB.
        double lookaheadMs = 2; ///< Requested anticipation, 0.1..10 ms; render may extend to 10 ms.
        double releaseMs = 80;  ///< Amplitude recovery time constant, 60..100 ms.
        std::span<const OfflineRegion> exclusions{}; ///< Exact PCM; feathering is outside regions.
    };
    enum class Reason : std::uint8_t
    {
        Compressed,
        CrestLimited,
        ResolutionLimited,
        ProtectedPeaks,
        ZeroAmount,
        NoSignal,
        AllExcluded,
        NoCorrection
    };
    struct Report
    {
        Reason reason = Reason::NoCorrection;
        double requestedReductionDb = 0, effectiveReductionDb = 0, achievedReductionDb = 0;
        double availableHeadroomDb = 0;
        double maximumReductionDb = 0; ///< Initial control depth; renderInfo holds calibrated bounds.
        double lookaheadMs = 0; ///< Actual anticipation after successful render calibration.
        double releaseMs = 0;
        double requestedLookaheadMs = 0; ///< Requested anticipation before render calibration.
        int renderAttempts = 0; ///< Complete-file calibration attempts; zero in a new plan.
        double inputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        std::size_t transientHolds = 0, additionalHolds = 0;
        bool peaksMeasured = false;
        OfflineGainRenderInfo renderInfo;
    };

    /** @brief Immutable source-bound sparse gain plan; positions are int64 source frames. */
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
            return exclusions_.apply(frame, curve_.gainAt(frame),
                                     static_cast<double>(attackFrames_));
        }

      private:
        friend class OfflinePeakCompressor;
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_;
        Report report_;
        double inputPeak_ = 0;
        std::int64_t attackFrames_ = 1;
        detail::OfflineAttenuation curve_;
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

    /**
     * @brief Reuses a transient map and measures the actual stage PCM.
     * Three bounded scans measure sustained headroom, count holds and fill them.
     * All scans verify the analysis fingerprint. A positive amount requires an
     * attack map; zero amount accepts energy-only analysis and skips these scans.
     */
    [[nodiscard]] PlanResult makePlan(
        OfflineAudioSource<T> &source,
        const typename OfflineTransientAnalyzer<T>::Analysis &analysis, const Options &options = {},
        const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            if (!analysis.isValid() || (options.reductionDb > 0 && !analysis.getOptions().attacks))
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            Plan plan;
            plan.spec_ = analysis.energy().getSpec();
            if (source.getSpec() != plan.spec_)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            plan.fingerprint_ = analysis.energy().fingerprint();
            plan.inputPeak_ = analysis.energy().samplePeak();
            plan.attackFrames_ =
                std::max(std::int64_t(1), static_cast<std::int64_t>(std::llround(
                                              plan.spec_.sampleRate * options.lookaheadMs * .001)));
            auto &report = plan.report_;
            report.requestedReductionDb = options.reductionDb;
            report.requestedLookaheadMs = options.lookaheadMs;
            report.lookaheadMs =
                1000 * static_cast<double>(plan.attackFrames_) / plan.spec_.sampleRate;
            report.releaseMs = options.releaseMs;
            report.inputSamplePeakDb =
                gainToDecibels(plan.inputPeak_, -std::numeric_limits<double>::infinity());
            job.checkpoint(OfflinePhase::Plan, 0, plan.spec_.frames);
            plan.exclusions_.assign(job, options.exclusions, plan.spec_.frames);
            const double releaseFrames = options.releaseMs * .001 * plan.spec_.sampleRate;
            const bool allExcluded = plan.exclusions_.covers(plan.spec_.frames);
            if (options.reductionDb == 0 || plan.inputPeak_ == 0 || allExcluded)
            {
                report.reason = allExcluded                ? Reason::AllExcluded
                                : options.reductionDb == 0 ? Reason::ZeroAmount
                                                           : Reason::NoSignal;
                plan.curve_.prepare(job, 0, plan.attackFrames_, releaseFrames);
            }
            else
            {
                detail::OfflineBlock<T> scratch(job, plan.spec_, jobOptions.blockFrames);
                EnvelopeFollower<double, 1> sustain;
                sustain.prepare({plan.spec_.sampleRate, 1, 1});
                sustain.setAttack(60);
                sustain.setRelease(300);
                double loudestSustain = 0;
                scan(source, plan, scratch, job, [&](std::int64_t, double magnitude) {
                    loudestSustain = std::max(loudestSustain, sustain.processSample(magnitude));
                });
                report.availableHeadroomDb =
                    loudestSustain > 0 ? std::max(0.0, -gainToDecibels(loudestSustain)) : 0;
                report.effectiveReductionDb =
                    std::min(options.reductionDb, report.availableHeadroomDb);
                const bool crestLimited = report.availableHeadroomDb < options.reductionDb;
                const double ceiling = detail::offlineRepresentablePeakGain<T>(
                    plan.inputPeak_, decibelsToGain(-report.effectiveReductionDb));
                const double representableReduction = -gainToDecibels(ceiling);
                const bool resolutionLimited =
                    report.effectiveReductionDb - representableReduction > .001;
                report.effectiveReductionDb = std::max(0.0, representableReduction);
                if (report.effectiveReductionDb < .001)
                {
                    report.effectiveReductionDb = 0;
                    report.reason =
                        resolutionLimited ? Reason::ResolutionLimited : Reason::NoCorrection;
                    plan.curve_.prepare(job, 0, plan.attackFrames_, releaseFrames);
                }
                else
                {
                    std::size_t count = 0;
                    const auto countHold = [&](std::int64_t, std::int64_t, double, bool) {
                        if (count == std::numeric_limits<std::size_t>::max())
                            detail::offlineFail(OfflineStatus::MemoryLimit);
                        ++count;
                    };
                    buildHolds(source, plan, analysis.attacks(), ceiling, scratch, job, countHold);
                    plan.curve_.prepare(job, count, plan.attackFrames_, releaseFrames);
                    std::size_t index = 0;
                    const auto addHold = [&](std::int64_t begin, std::int64_t end, double peak,
                                             bool mapped) {
                        const double gain = ceiling / peak;
                        plan.curve_.set(index++, begin, end, gain);
                        report.maximumReductionDb =
                            std::max(report.maximumReductionDb, -gainToDecibels(gain));
                        if (mapped)
                            ++report.transientHolds;
                        else
                            ++report.additionalHolds;
                    };
                    buildHolds(source, plan, analysis.attacks(), ceiling, scratch, job, addHold);
                    if (index != count)
                        detail::offlineFail(OfflineStatus::SourceMismatch);
                    report.reason = resolutionLimited ? Reason::ResolutionLimited
                                    : crestLimited    ? Reason::CrestLimited
                                                      : Reason::Compressed;
                }
            }
            job.checkpoint(OfflinePhase::Plan, plan.spec_.frames, plan.spec_.frames);
            result.status =
                report.maximumReductionDb == 0 ? OfflineStatus::NoChange : OfflineStatus::Success;
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
            features.attacks = options.reductionDb > 0;
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

    /** @brief Verifies and renders transactionally; reports the actual final peak reduction. */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink, const OfflineJobOptions &job = {}) const
    {
        const auto attempt = [&](const detail::OfflineAttenuation &curve,
                                 std::int64_t attackFrames, const OfflineJobOptions &options) {
            auto report = plan.report_;
            report.lookaheadMs = 1000 * static_cast<double>(attackFrames) / plan.spec_.sampleRate;
            return detail::offlineRenderGain<Result>(
                source, plan.spec_, plan.fingerprint_, plan.inputPeak_, plan.isValid(),
                plan.report_.maximumReductionDb != 0, report, sink, options,
                [&plan, &curve](detail::OfflineSession &session) {
                    return [&plan, cursor = detail::OfflineAttenuation::Cursor(curve, session)](
                               std::int64_t frame) mutable {
                        return plan.exclusions_.apply(frame, cursor(frame),
                                                      static_cast<double>(plan.attackFrames_));
                    };
                },
                {&plan.exclusions_, static_cast<double>(plan.attackFrames_), true});
        };
        auto result = attempt(plan.curve_, plan.attackFrames_, job);
        int attempts = 1;
        auto memoryBytes = result.memoryBytes;
        // A fixed processing-delta scale can be infeasible: increasing it can
        // reduce the transient but deepen an earlier negative ringing peak.
        // Extend only the source-aligned attack, keeping the holds and release.
        // Failed target attempts have not begun the sink transaction.
        if (plan.isValid() && !result.report.renderInfo.targetFeasible)
        {
            const auto maximum =
                static_cast<std::int64_t>(std::llround(.010 * plan.spec_.sampleRate));
            auto attack = plan.attackFrames_;
            try
            {
                while (attack < maximum && !result.report.renderInfo.targetFeasible)
                {
                    attack = std::min(maximum, attack * 2);
                    detail::OfflineSession session(job);
                    auto curve = plan.curve_.withAttack(session, attack);
                    result = attempt(curve, attack, detail::offlineRemaining(job, session.bytes()));
                    memoryBytes = std::max(memoryBytes, result.memoryBytes + session.bytes());
                    ++attempts;
                }
            }
            catch (...)
            {
                result.status = detail::offlineExceptionStatus();
            }
        }
        result.memoryBytes = memoryBytes;
        result.report.renderAttempts = attempts;
        if (result.succeeded() && plan.inputPeak_ > 0)
        {
            result.report.achievedReductionDb =
                std::max(0.0, result.report.inputSamplePeakDb - result.report.outputSamplePeakDb);
            if (!plan.exclusions_.view().empty() &&
                result.report.achievedReductionDb + .0001 < result.report.effectiveReductionDb)
                result.report.reason = Reason::ProtectedPeaks;
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
        if (!std::isfinite(options.reductionDb) || (options.reductionDb < 0) ||
            (options.reductionDb > 18) || !std::isfinite(options.lookaheadMs) ||
            (options.lookaheadMs < .1) || (options.lookaheadMs > 10) ||
            !std::isfinite(options.releaseMs) || (options.releaseMs < 60) ||
            (options.releaseMs > 100))
            detail::offlineFail(OfflineStatus::InvalidInput);
    }
    template <class Consume>
    static void scan(OfflineAudioSource<T> &source, const Plan &plan,
                     detail::OfflineBlock<T> &scratch, const detail::OfflineSession &job,
                     Consume consume)
    {
        detail::offlineScanMagnitude(source, plan.spec_, plan.fingerprint_, plan.inputPeak_,
                                     scratch, job, [&](std::int64_t frame, double magnitude) {
                                         consume(frame, magnitude / plan.inputPeak_);
                                     });
    }
    template <class Emit>
    static void buildHolds(OfflineAudioSource<T> &source, const Plan &plan,
                           std::span<const typename OfflineTransientAnalyzer<T>::Event> events,
                           double ceiling, detail::OfflineBlock<T> &scratch,
                           const detail::OfflineSession &job, Emit emit)
    {
        std::size_t event = 0;
        double mappedPeak = 0, extraPeak = 0;
        std::int64_t extraBegin = 0, extraEnd = 0;
        // A quiet gap separates otherwise undetected excursions. This avoids
        // releasing on individual periods of a stationary low-frequency tone.
        const auto gap = static_cast<std::int64_t>(std::ceil(plan.spec_.sampleRate * .03));
        const auto flushExtra = [&] {
            if (extraPeak > 0)
                emit(extraBegin, extraEnd, extraPeak, false);
            extraPeak = 0;
        };
        scan(source, plan, scratch, job, [&](std::int64_t frame, double magnitude) {
            while (event < events.size() && frame >= events[event].end)
            {
                if (mappedPeak > ceiling)
                    emit(events[event].begin, events[event].end, mappedPeak, true);
                mappedPeak = 0;
                ++event;
            }
            if (event < events.size() && frame >= events[event].begin)
            {
                flushExtra();
                mappedPeak = std::max(mappedPeak, magnitude);
            }
            else
            {
                if (extraPeak > 0 && frame - extraEnd >= gap)
                    flushExtra();
                if (magnitude > ceiling)
                {
                    if (extraPeak == 0)
                        extraBegin = frame;
                    extraEnd = frame + 1;
                    extraPeak = std::max(extraPeak, magnitude);
                }
            }
        });
        if (event < events.size() && mappedPeak > ceiling)
            emit(events[event].begin, events[event].end, mappedPeak, true);
        flushExtra();
    }
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
