// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineClip.h
 * @brief Shared complete-source planning, calibration and transactional clipping.
 * Threading: worker only; returned plans are immutable and independently reusable.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "../../Analysis/OfflineEnergyAnalyzer.h"
#include "../TruePeakDetector.h"
#include "OfflineClipProjection.h"
#include "OfflineGainSource.h"
#include "OfflineWorker.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>

namespace dspark
{
/** @brief Soft curves shared with Clipper, including its sine approximation. */
enum class OfflineSoftClipCurve
{
    Sine,
    Tanh,
    GoldenRatio
};

namespace detail
{
struct OfflineClipOptions
{
    double reductionDb = 0; ///< Positive sample-peak reduction, 0..12 dB; zero is exact identity.
    int oversamplingFactor = 4;                ///< Explicit local factor: 1, 2, 4, 8 or 16.
    std::span<const OfflineRegion> exclusions; ///< Copied source-frame regions; PCM stays exact.
};
struct OfflineSoftClipOptions : OfflineClipOptions
{
    OfflineSoftClipCurve curve = OfflineSoftClipCurve::Sine;
};

struct OfflineClipMeasurement
{
    double peak = 0, truePeak = 0;
    bool representable = true;
};
struct OfflineClipCalibration
{
    double ceiling = 1, peak = 0, errorDb = std::numeric_limits<double>::infinity();
    int attempts = 0;
    bool converged = false, bracketed = false;
};

inline double offlineClipPeakErrorDb(double peak, double target) noexcept
{
    return peak > 0 ? (20 / std::log(10.)) * std::log1p((peak - target) / target)
                    : -std::numeric_limits<double>::infinity();
}

// A fixed absolute dB tolerance can accept unity or even gain for a small
// reduction request. Bound the error by half the representable reduction too.
inline bool offlineClipTargetMet(double peak, double target, double errorDb) noexcept
{
    if (target == 1)
        return peak == 1;
    const double toleranceDb = std::min(.005, -(10 / std::log(10.)) * std::log1p(target - 1));
    return peak < 1 && std::abs(errorDb) <= toleranceDb;
}

// Search the actual rounded output peak. The peak need not be monotone in the
// ceiling: maintain a measured sign bracket and reset all DSP for every trial.
// Neither a final limiter nor an output gain adjustment participates.
template <class Render>
OfflineClipCalibration offlineCalibrateClip(double initialCeiling, double target, Render render)
{
    constexpr int maximumPasses = 24;
    OfflineClipCalibration best;
    const auto trial = [&](double ceiling)
    {
        const auto measured = render(ceiling);
        ++best.attempts;
        if (!(measured.peak >= 0) || !std::isfinite(measured.peak))
            offlineFail(OfflineStatus::NumericalFailure);
        const double error = offlineClipPeakErrorDb(measured.peak, target);
        if (measured.representable && std::abs(error) < std::abs(best.errorDb))
        {
            best.ceiling = ceiling;
            best.peak = measured.peak;
            best.errorDb = error;
        }
        best.converged = offlineClipTargetMet(best.peak, target, best.errorDb);
        return measured.peak;
    };
    double lo = initialCeiling, hi = initialCeiling;
    double fl = trial(initialCeiling), fh = fl;
    if (best.converged)
        return best;
    if (fl < target)
    {
        while (best.attempts < maximumPasses && fh < target && hi < 1e12)
        {
            lo = hi;
            fl = fh;
            hi *= 2;
            fh = trial(hi);
            if (best.converged)
                return best;
        }
    }
    else
    {
        // The caller implements the exact zero-curve endpoint, including its
        // exclusion mask. An arbitrary tiny positive ceiling is both more
        // expensive and less reliable as the lower calibration bracket.
        lo = 0;
        fl = trial(lo);
        if (best.converged)
            return best;
    }
    best.bracketed = fl < target && fh > target;
    if (!best.bracketed)
        return best;
    double lowWeight = 1, highWeight = 1;
    int previousSide = 0;
    bool flatEndpoint = false;
    while (best.attempts < maximumPasses)
    {
        // Illinois weighting prevents a nearly flat unity endpoint from
        // pinning false position. Keep the actual measured sign bracket.
        const double lowResidual = lowWeight * (target - fl);
        const double highResidual = highWeight * (fh - target);
        const double fraction = flatEndpoint
                                    ? .5
                                    : std::clamp(lowResidual / (lowResidual + highResidual), .1, .9);
        const double ceiling = lo + fraction * (hi - lo);
        if (!(ceiling > lo && ceiling < hi))
            break;
        const double peak = trial(ceiling);
        if (best.converged)
            return best;
        if (peak > target)
        {
            // Rounded PCM can have an exactly flat region. Bisect it rather
            // than spending repeated complete renders on the same peak.
            flatEndpoint = peak == fh;
            hi = ceiling;
            fh = peak;
            lowWeight = previousSide == 1 ? lowWeight * .5 : 1;
            highWeight = 1;
            previousSide = 1;
        }
        else
        {
            flatEndpoint = peak == fl;
            lo = ceiling;
            fl = peak;
            highWeight = previousSide == -1 ? highWeight * .5 : 1;
            lowWeight = 1;
            previousSide = -1;
        }
    }
    return best;
}

template <FloatType T, bool Soft> class OfflineClipProcessor
{
  public:
    using Curve = OfflineSoftClipCurve;
    using Options = std::conditional_t<Soft, OfflineSoftClipOptions, OfflineClipOptions>;
    enum class Reason
    {
        Ready,
        Processed,
        ZeroAmount,
        NoSignal,
        AllExcluded,
        RepresentabilityLimited,
        ConstraintLimited,
        CalibrationFailed
    };
    struct Report
    {
        Reason reason = Reason::ZeroAmount;
        double requestedReductionDb = 0;
        double effectiveReductionDb = 0; ///< After rounding the target toward the input PCM peak.
        double achievedReductionDb = 0;
        double targetErrorDb = 0; ///< Final output peak relative to the effective target.
        double inputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        double normalizedCeiling = 1; ///< Curve ceiling divided by the complete-source peak.
        int oversamplingFactor = 4;
        int calibrationPasses = 0;
        int compensatedLatencyFrames = 0; ///< Internal delay removed from the result.
        bool targetMet = false;           ///< Set only after final rounded PCM verification.
        bool representabilityLimited = false;
    };
    /** @brief Immutable format/fingerprint/control plan, with owned exclusion regions. */
    class Plan final
    {
      public:
        Plan() = default;
        Plan(Plan &&) noexcept = default;
        Plan &operator=(Plan &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept
        {
            return state_ != nullptr;
        }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept
        {
            return state_ ? state_->spec : OfflineAudioSpec{};
        }
        [[nodiscard]] Report getReport() const noexcept
        {
            return state_ ? state_->report : Report{};
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return state_ ? sizeof(State) + state_->exclusions.retainedBytes() : 0;
        }
        /** @brief Borrowed regions, valid while the plan remains alive and unmoved. */
        [[nodiscard]] std::span<const OfflineRegion> exclusions() const noexcept
        {
            return state_ ? state_->exclusions.view() : std::span<const OfflineRegion>{};
        }

      private:
        friend class OfflineClipProcessor;
        struct State
        {
            OfflineAudioSpec spec;
            OfflineFingerprint fingerprint;
            OfflineExclusions exclusions;
            Report report;
            Curve curve = Curve::Sine;
            double inputPeak = 0, targetGain = 1, featherFrames = 1;
            bool active = false;
        };
        std::unique_ptr<State> state_;
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

    /** @brief Reuses complete-source energy analysis; no PCM or filtered audio is retained. */
    [[nodiscard]] PlanResult makePlan(OfflineAudioSource<T> &source,
                                      const typename OfflineEnergyAnalyzer<T>::Analysis &analysis,
                                      const Options &options = {},
                                      const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            if (!analysis.isValid())
                offlineFail(OfflineStatus::InvalidInput);
            OfflineSession job(jobOptions);
            if (source.getSpec() != analysis.getSpec())
                offlineFail(OfflineStatus::SourceMismatch);
            job.checkpoint(OfflinePhase::Plan, 0, analysis.getSpec().frames);
            job.charge(sizeof(typename Plan::State));
            Plan plan;
            plan.state_ = std::make_unique<typename Plan::State>();
            auto &state = *plan.state_;
            state.spec = analysis.getSpec();
            state.fingerprint = analysis.fingerprint();
            state.inputPeak = analysis.samplePeak();
            state.featherFrames = std::max(1., std::round(.005 * state.spec.sampleRate));
            if constexpr (Soft)
                state.curve = options.curve;
            state.exclusions.assign(job, options.exclusions, state.spec.frames);
            const bool allExcluded = state.exclusions.covers(state.spec.frames);
            auto &report = state.report;
            report.requestedReductionDb = options.reductionDb;
            report.oversamplingFactor = options.oversamplingFactor;
            report.inputSamplePeakDb =
                gainToDecibels(state.inputPeak, -std::numeric_limits<double>::infinity());
            if (state.inputPeak > 0)
            {
                state.targetGain = offlineRepresentablePeakGain<T>(
                    state.inputPeak, decibelsToGain(-options.reductionDb));
                report.effectiveReductionDb = -gainToDecibels(state.targetGain);
                report.representabilityLimited =
                    (state.targetGain == 1 && options.reductionDb > 0) ||
                    options.reductionDb - report.effectiveReductionDb > 1e-6;
            }
            state.active = options.reductionDb > 0 && state.inputPeak > 0 && state.targetGain < 1 &&
                           !allExcluded;
            report.reason = options.reductionDb == 0 ? Reason::ZeroAmount
                            : state.inputPeak == 0   ? Reason::NoSignal
                            : allExcluded            ? Reason::AllExcluded
                            : state.targetGain == 1  ? Reason::RepresentabilityLimited
                                                     : Reason::Ready;
            job.checkpoint(OfflinePhase::Plan, state.spec.frames, state.spec.frames);
            if (source.getSpec() != state.spec)
                offlineFail(OfflineStatus::SourceMismatch);
            result.status = state.active ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
            result.plan = std::move(plan);
        }
        catch (...)
        {
            result.status = offlineExceptionStatus();
        }
        return result;
    }
    /** @brief Analyzes the complete source and makes a reusable worker plan. */
    [[nodiscard]] PlanResult analyze(OfflineAudioSource<T> &source, const Options &options = {},
                                     const OfflineJobOptions &job = {}) const
    {
        PlanResult result;
        try
        {
            validateOptions(options);
            auto analyzed = OfflineEnergyAnalyzer<T>().analyze(source, job);
            if (!analyzed.succeeded())
            {
                result.status = analyzed.status;
                return result;
            }
            result = makePlan(source, analyzed.analysis, options,
                              offlineRemaining(job, analyzed.analysis.retainedBytes()));
            result.memoryBytes = std::max(analyzed.memoryBytes,
                                          result.memoryBytes + analyzed.analysis.retainedBytes());
        }
        catch (...)
        {
            result.status = offlineExceptionStatus();
        }
        return result;
    }
    /** @brief Calibrates actual PCM, then verifies a provisional render before committing it. */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink, const OfflineJobOptions &job = {}) const
    {
        if constexpr (!Soft)
            return renderCurve<ClipperCurve::Hard>(source, plan, sink, job);
        else
        {
            const auto curve = plan.isValid() ? plan.state_->curve : Curve::Sine;
            switch (curve)
            {
            case Curve::Sine:
                return renderCurve<ClipperCurve::Sine>(source, plan, sink, job);
            case Curve::Tanh:
                return renderCurve<ClipperCurve::Tanh>(source, plan, sink, job);
            case Curve::GoldenRatio:
                return renderCurve<ClipperCurve::GoldenRatio>(source, plan, sink, job);
            }
            Result result;
            result.status = OfflineStatus::InvalidInput;
            return result;
        }
    }
    /** @brief Owning render; aliasing is safe and failure leaves output unchanged. */
    template <int MaxChannels>
    [[nodiscard]] Result run(const AudioBuffer<T, MaxChannels> &input,
                             AudioBuffer<T, MaxChannels> &output, double sampleRate,
                             const Options &options = {}, const OfflineJobOptions &job = {}) const
    {
        return offlineRun(*this, input, output, sampleRate, options, job);
    }

  private:
    static void validateOptions(const Options &options)
    {
        const int factor = options.oversamplingFactor;
        if (!std::isfinite(options.reductionDb) || (options.reductionDb < 0) ||
            (options.reductionDb > 12) ||
            (factor != 1 && factor != 2 && factor != 4 && factor != 8 && factor != 16))
            offlineFail(OfflineStatus::InvalidInput);
        if constexpr (Soft)
            if (options.curve != Curve::Sine && options.curve != Curve::Tanh &&
                options.curve != Curve::GoldenRatio)
                offlineFail(OfflineStatus::InvalidInput);
    }
    template <ClipperCurve C> static double initialCeiling(double target)
    {
        double low = 0, high = 1;
        while (clipperShape<C>(1., high) < target && high < 1e12)
            high *= 2;
        for (int i = 0; i < 64; ++i)
        {
            const double middle = (low + high) / 2;
            if (clipperShape<C>(1., middle) > target)
                high = middle;
            else
                low = middle;
        }
        return (low + high) / 2;
    }

    template <ClipperCurve C>
    static Result renderCurve(OfflineAudioSource<T> &source, const Plan &plan,
                              OfflineAudioSink<T> &sink, const OfflineJobOptions &options)
    {
        Result result;
        OfflineSinkTransaction<T> transaction;
        try
        {
            if (!plan.isValid())
                offlineFail(OfflineStatus::InvalidInput);
            const auto &state = *plan.state_;
            const auto spec = state.spec;
            if (source.getSpec() != spec)
                offlineFail(OfflineStatus::SourceMismatch);
            result.report = state.report;
            auto &report = result.report;
            OfflineSession job(options);
            job.checkpoint(OfflinePhase::Verify, 0, spec.frames);
            const int blockLimit = std::min(4096, options.blockFrames);
            OfflineBlock<T> input(job, spec, blockLimit), output(job, spec, blockLimit);
            const double normalizer = state.inputPeak > 0 ? state.inputPeak : 1;
            const double maximum = static_cast<double>(std::numeric_limits<T>::max());
            // Reuse the existing verified PCM cache. Unity control performs no
            // gain processing; only its source validation and bounded reads are used.
            struct UnityControl
            {
                double operator()(std::int64_t) const noexcept
                {
                    return 1;
                }
            };
            using Cache = OfflineGainSource<T, UnityControl>;
            struct Reader
            {
                Cache *cache;
                int channel;
                void operator()(std::int64_t first, int count, double *values)
                {
                    cache->readAudio(channel, first, count, values);
                }
            };
            using Projection = offline_clip::BoundedClip<C, Reader>;
            std::optional<Cache> cached;
            std::optional<offline_clip::TailPolicy<C>> policy;
            std::array<offline_clip::TailCertificate, 2> sourceStats;
            bool sourceStatsReady = false;
            if (state.active)
            {
                cached.emplace(source, spec, state.fingerprint, state.inputPeak, job, options,
                               UnityControl{});
                const int order = report.oversamplingFactor <= 2 ? 18 : 10;
                // All temporary inverseBoxPower vector requests, before construction.
                job.charge(
                    static_cast<std::size_t>(6 * (order + 1) + (order + 3) * (2 * order + 1)) *
                    sizeof(double));
                policy.emplace(report.oversamplingFactor);
            }
            // Bilateral finite-source projection is evaluated at original indices;
            // no causal source-frame delay is inserted and later removed.
            report.compensatedLatencyFrames = 0;
            result.memoryBytes = job.bytes();
            auto passOptions = offlineRemaining(options, job.bytes());
            struct ProgressContext
            {
                const OfflineJobOptions *options;
                std::int64_t frames;
            } progress{&options, spec.frames};
            if (options.progress)
            {
                passOptions.progressContext = &progress;
                passOptions.progress = [](void *context, OfflineProgress)
                {
                    const auto &p = *static_cast<ProgressContext *>(context);
                    // Internal high-rate/padded clocks are not source positions.
                    return p.options->progress(p.options->progressContext,
                                               {OfflinePhase::Verify, 0, p.frames});
                };
            }
            // Keep allocation accounting and reusable storage alive across
            // calibration. Geometry changes return their scratch to the same
            // pool; only genuinely new upstream payload consumes the budget.
            OfflineSession passJob(passOptions);
            OfflineMemoryPool scratch(passJob, false);
            passJob.useScratchResource(scratch);
            std::array<std::optional<Projection>, 2> projections;
            const auto pass = [&](double ceiling, bool publish)
            {
                const auto phase = publish ? OfflinePhase::Render : OfflinePhase::Verify;
                job.checkpoint(OfflinePhase::Verify, 0, spec.frames);
                if (cached && ceiling > 0)
                {
                    cached->reset();
                    std::array<offline_clip::TailCertificate, 2> shapeStats;
                    for (std::size_t leaf = 0; leaf < cached->leaves(); ++leaf)
                    {
                        const auto first = static_cast<std::int64_t>(leaf) * cached->block();
                        const int count = static_cast<int>(
                            std::min<std::int64_t>(cached->block(), spec.frames - first));
                        for (int c = 0; c < spec.channels; ++c)
                        {
                            const auto *raw = cached->audio(leaf, c);
                            for (int i = 0; i < count; ++i)
                            {
                                const double x = static_cast<double>(raw[i]) / normalizer;
                                if (!sourceStatsReady)
                                    sourceStats[c].append(x);
                                shapeStats[c].append(clipperShape<C>(x, ceiling));
                            }
                        }
                        job.checkpoint(OfflinePhase::Verify, first + count, spec.frames);
                    }
                    sourceStatsReady = true;
                    std::array<offline_clip::TailPlan, 2> tails;
                    std::int64_t padding = 64;
                    for (int c = 0; c < spec.channels; ++c)
                    {
                        tails[c] = policy->choose(ceiling, sourceStats[c], shapeStats[c]);
                        padding = std::max(padding, tails[c].padding);
                    }
                    // Balance duration-sized moments against fixed FFT/cached PCM
                    // scratch. The actual allocations still enforce the job budget.
                    int grid = 128;
                    long double best = std::numeric_limits<long double>::infinity();
                    const long double extent =
                        static_cast<long double>(spec.frames) + 2.L * padding;
                    for (int candidate = 128;
                         candidate <= 16384 && candidate * report.oversamplingFactor <= 65536;
                         candidate *= 2)
                    {
                        const long double cost =
                            (1000.L * report.oversamplingFactor - 32) * candidate +
                            3456.L * extent / candidate;
                        if (cost < best)
                        {
                            best = cost;
                            grid = candidate;
                        }
                    }
                    for (int c = 0; c < spec.channels; ++c)
                    {
                        cached->reset();
                        if (projections[c] && projections[c]->hasGeometry(padding, grid))
                            projections[c]->reset(ceiling, tails[c].exterior);
                        else
                            projections[c].emplace(passJob, Reader{&*cached, c}, spec.frames,
                                                   padding, report.oversamplingFactor, ceiling,
                                                   tails[c].exterior, grid);
                        projections[c]->prepareProjection();
                    }
                }
                result.memoryBytes = std::max(result.memoryBytes, job.bytes() + passJob.bytes());
                // Prepare every owned DSP allocation before opening the destination.
                if (publish)
                {
                    transaction.sink = &sink;
                    offlineCallSink([&] { return sink.begin(spec); });
                }
                job.checkpoint(phase, 0, spec.frames);
                if (cached)
                    cached->reset();
                OfflineClipMeasurement measured;
                OfflineFingerprint fingerprint;
                TruePeakDetector<double, 2> detector;
                std::int64_t written = 0;
                const auto consume = [&](std::int64_t first, int count,
                                         const std::array<const double *, 2> &filtered)
                {
                    if (first != written)
                        offlineFail(OfflineStatus::NumericalFailure);
                    for (int offset = 0; offset < count;)
                    {
                        const auto at = first + offset;
                        int length = std::min(input.frames(), count - offset);
                        std::array<const T *, 2> raw{};
                        if (cached)
                        {
                            const auto leaf = static_cast<std::size_t>(at / cached->block());
                            const int local = static_cast<int>(at % cached->block());
                            length = std::min(length, cached->block() - local);
                            for (int c = 0; c < spec.channels; ++c)
                                raw[c] = cached->audio(leaf, c) + local;
                        }
                        else
                        {
                            auto block = input.view(length);
                            offlineRead(source, spec, at, block);
                            for (int c = 0; c < spec.channels; ++c)
                                raw[c] = block.getChannel(c);
                        }
                        auto out = output.view(length);
                        for (int f = 0; f < length; ++f)
                        {
                            const double mask =
                                state.active
                                    ? 1 - state.exclusions.apply(at + f, 0, state.featherFrames)
                                    : 0;
                            for (int c = 0; c < spec.channels; ++c)
                            {
                                const T rawValue = raw[c][f];
                                if (!std::isfinite(rawValue))
                                    offlineFail(OfflineStatus::NonFiniteInput);
                                if (std::abs(static_cast<double>(rawValue)) > state.inputPeak)
                                    offlineFail(OfflineStatus::SourceMismatch);
                                offlineHash(fingerprint, rawValue);
                                double normalized = static_cast<double>(rawValue) / normalizer;
                                T value = rawValue;
                                if (mask != 0)
                                {
                                    const double shaped =
                                        ceiling == 0 ? 0 : filtered[c][offset + f];
                                    normalized = std::lerp(normalized, shaped, mask);
                                    if (!std::isfinite(normalized))
                                        offlineFail(OfflineStatus::NumericalFailure);
                                    if (std::abs(normalized) > maximum / normalizer)
                                    {
                                        measured.representable = false;
                                        if (publish)
                                            offlineFail(OfflineStatus::NumericalFailure);
                                    }
                                    else
                                    {
                                        value = static_cast<T>(normalized * normalizer);
                                        if (!std::isfinite(value))
                                            offlineFail(OfflineStatus::NumericalFailure);
                                        normalized = static_cast<double>(value) / normalizer;
                                    }
                                }
                                measured.peak = std::max(measured.peak, std::abs(normalized));
                                if (publish)
                                {
                                    out.getChannel(c)[f] = value;
                                    measured.truePeak = std::max(
                                        measured.truePeak, detector.processSample(normalized, c));
                                }
                            }
                        }
                        if (publish)
                            offlineCallSink(
                                [&]
                                {
                                    std::array<const T *, 2> pointers{};
                                    for (int c = 0; c < spec.channels; ++c)
                                        pointers[c] = out.getChannel(c);
                                    return sink.write(at, {pointers.data(), spec.channels, length});
                                });
                        offset += length;
                        written += length;
                        job.checkpoint(phase, written, spec.frames);
                    }
                };
                if (cached && ceiling > 0)
                {
                    for (auto leaf = projections[0]->firstLeaf(); leaf < projections[0]->endLeaf();
                         ++leaf)
                    {
                        std::array<const double *, 2> filtered{};
                        int count = 0;
                        for (int c = 0; c < spec.channels; ++c)
                        {
                            const auto data = projections[c]->get(leaf);
                            filtered[c] = data.data();
                            if (c && count != static_cast<int>(data.size()))
                                offlineFail(OfflineStatus::NumericalFailure);
                            count = static_cast<int>(data.size());
                        }
                        consume(projections[0]->leafOrigin(leaf), count, filtered);
                    }
                }
                else
                    while (written < spec.frames)
                        consume(written,
                                static_cast<int>(
                                    std::min<std::int64_t>(input.frames(), spec.frames - written)),
                                {});
                if (written != spec.frames || fingerprint != state.fingerprint ||
                    source.getSpec() != spec)
                    offlineFail(OfflineStatus::SourceMismatch);
                if (publish)
                    for (int c = 0; c < spec.channels; ++c)
                        measured.truePeak = std::max(measured.truePeak, detector.getTailPeak(c));
                return measured;
            };
            OfflineClipCalibration calibrated;
            if (state.active)
            {
                calibrated =
                    offlineCalibrateClip(initialCeiling<C>(state.targetGain), state.targetGain,
                                         [&](double ceiling) { return pass(ceiling, false); });
                report.calibrationPasses = calibrated.attempts;
                report.normalizedCeiling = calibrated.ceiling;
                report.targetErrorDb = calibrated.errorDb;
                if (!calibrated.converged)
                {
                    const bool constrained =
                        !calibrated.bracketed && !state.exclusions.view().empty();
                    report.reason =
                        constrained ? Reason::ConstraintLimited : Reason::CalibrationFailed;
                    offlineFail(constrained ? OfflineStatus::TargetUnreachable
                                            : OfflineStatus::NumericalFailure);
                }
            }
            const auto final = pass(calibrated.ceiling, true);
            if (state.active && (!final.representable || final.peak != calibrated.peak))
                offlineFail(OfflineStatus::NumericalFailure);
            constexpr double silence = -std::numeric_limits<double>::infinity();
            report.outputSamplePeakDb = final.peak > 0 ? gainToDecibels(final.peak, silence) +
                                                             gainToDecibels(normalizer, silence)
                                                       : silence;
            report.outputTruePeakDb = final.truePeak > 0 ? gainToDecibels(final.truePeak, silence) +
                                                               gainToDecibels(normalizer, silence)
                                                         : silence;
            report.achievedReductionDb =
                state.inputPeak > 0 && final.peak > 0 ? -gainToDecibels(final.peak) : 0;
            report.targetErrorDb = offlineClipPeakErrorDb(final.peak, state.targetGain);
            report.targetMet = state.inputPeak > 0 &&
                               offlineClipTargetMet(final.peak, state.targetGain,
                                                    report.targetErrorDb);
            if (state.active && !report.targetMet)
                offlineFail(OfflineStatus::NumericalFailure);
            if (state.active)
                report.reason = report.representabilityLimited ? Reason::RepresentabilityLimited
                                                               : Reason::Processed;
            job.checkpoint(OfflinePhase::Render, spec.frames, spec.frames);
            if (source.getSpec() != spec)
                offlineFail(OfflineStatus::SourceMismatch);
            offlineCallSink([&] { return sink.commit(); });
            transaction.sink = nullptr;
            result.status = state.active ? OfflineStatus::Success : OfflineStatus::NoChange;
        }
        catch (...)
        {
            result.status = offlineExceptionStatus();
        }
        return result;
    }
};
} // namespace detail
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
