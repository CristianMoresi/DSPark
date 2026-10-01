// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineStereoGenerator.h
 * @brief Complete-source stereo generation and optional host-owned delta caching.
 *
 * OFFLINE ONLY. Reuses StereoGenerator, shared source validation, exclusions,
 * transactions and TruePeakDetector. No additional energy/tempo/transient map is
 * needed. The output has two channels and exactly the source frame count. Mono
 * duplication must be explicitly enabled. Width zero and protected PCM remain
 * exact; 5 ms feathers lie outside protected regions. No limiter or gain trim.
 *
 * Modulation starts at the source timelineOrigin; read offsets remain relative
 * to zero. Complete-source rendering starts with empty filter history, flushes
 * zeros, strips alignment delay and truncates the tail to the source duration.
 * A trimmed source does not reconstruct earlier filter history automatically.
 *
 * Threading: synchronous worker operations, never audio callbacks. The facade
 * is stateless. Completed plans and cache certificates are immutable while alive
 * and unmoved. Source, sink, cache PCM and job options remain caller-owned.
 */
#include "../Analysis/OfflineEnergyAnalyzer.h"
#if DSPARK_HAS_OFFLINE
#include "../Core/TruePeakDetector.h"
#include "../Core/detail/OfflineStereo.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace dspark
{
/** @brief Offline processed-copy stereo generation with a one-width control. */
template <FloatType T> class OfflineStereoGenerator final
{
  public:
    using GenerationOptions = StereoGenerator<double>::Options;
    struct Options
    {
        float width = 0; ///< 0..1; zero returns exact original PCM, optionally duplicated.
        int oversamplingFactor = 4; ///< Local color factor: 2, 4, 8 or 16.
        float lowCutHz = 0;         ///< Generated delta only; off or 20..5000 Hz.
        bool duplicateMono = false; ///< Explicit permission to turn mono into two channels.
        std::span<const OfflineRegion> exclusions{}; ///< Copied source-relative regions.
    };
    enum class Reason : std::uint8_t
    {
        Ready,
        Generated,
        ZeroWidth,
        NoSignal,
        AllExcluded
    };
    struct Report
    {
        Reason reason = Reason::ZeroWidth;
        float width = 0;
        GenerationOptions generation;
        double inputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        int compensatedLatencyFrames = 0; ///< Alignment delay already removed from output/cache.
        bool duplicatedMono = false;
        bool usedDeltaCache = false;
        bool peaksMeasured = false;
    };

    /** @brief Owned controls/exclusions bound to a complete source fingerprint. */
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
        [[nodiscard]] OfflineAudioSpec getOutputSpec() const noexcept
        {
            return state_ ? state_->outputSpec : OfflineAudioSpec{};
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
        friend class OfflineStereoGenerator;
        struct State
        {
            OfflineAudioSpec spec, outputSpec;
            OfflineFingerprint fingerprint;
            detail::OfflineExclusions exclusions;
            Report report;
            double inputPeak = 0, featherFrames = 1;
            bool active = false;
        };
        std::unique_ptr<State> state_;
    };

    /** @brief Certificate for caller-owned, mono-double canonical delta storage.
     * Width and exclusions are deliberately absent from the cache key: they apply
     * at composition. Source PCM/format/clock and all generation settings are
     * bound. renderCached verifies both complete PCM streams before publication.
     * This object owns metadata only; it never keeps pointers to host audio.
     */
    class DeltaCache final
    {
      public:
        DeltaCache() = default;
        DeltaCache(DeltaCache &&) noexcept = default;
        DeltaCache &operator=(DeltaCache &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept
        {
            return state_ != nullptr;
        }
        [[nodiscard]] OfflineAudioSpec getSourceSpec() const noexcept
        {
            return state_ ? state_->sourceSpec : OfflineAudioSpec{};
        }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept
        {
            return state_ ? state_->deltaSpec : OfflineAudioSpec{};
        }
        [[nodiscard]] GenerationOptions getGenerationOptions() const noexcept
        {
            return state_ ? state_->generation : GenerationOptions{};
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return state_ ? sizeof(State) : 0;
        }

      private:
        friend class OfflineStereoGenerator;
        struct State
        {
            OfflineAudioSpec sourceSpec, deltaSpec;
            OfflineFingerprint sourceFingerprint, deltaFingerprint;
            GenerationOptions generation;
            double inputPeak = 0, deltaPeak = 0;
            int latency = 0;
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
    struct CacheResult
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        DeltaCache cache;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept
        {
            return offlineSucceeded(status);
        }
    };

    /** @brief Scans the complete source once for provenance/peak; no feature map is built. */
    [[nodiscard]] PlanResult analyze(OfflineAudioSource<T> &source, const Options &options = {},
                                     const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            const auto spec = source.getSpec();
            validate(spec, options);
            detail::OfflineSession job(jobOptions);
            job.checkpoint(OfflinePhase::Analyze, 0, spec.frames);
            detail::OfflineBlock<T> block(job, spec, std::min(4096, jobOptions.blockFrames));
            const auto summary = detail::offlineScanSource(
                source, spec, block, job, OfflinePhase::Analyze, [](std::int64_t, double) {});
            result.plan = createPlan(source, spec, summary.fingerprint, summary.peak, options, job);
            result.status =
                result.plan.state_->active ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Reuses a complete energy analysis; render still verifies source PCM. */
    [[nodiscard]] PlanResult makePlan(OfflineAudioSource<T> &source,
                                      const typename OfflineEnergyAnalyzer<T>::Analysis &analysis,
                                      const Options &options = {},
                                      const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            if (!analysis.isValid())
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            result.plan = createPlan(source, analysis.getSpec(), analysis.fingerprint(),
                                     analysis.samplePeak(), options, job);
            result.status =
                result.plan.state_->active ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Changes controls/exclusions without another scan; render verifies the source. */
    [[nodiscard]] PlanResult replan(OfflineAudioSource<T> &source, const Plan &previous,
                                    const Options &options,
                                    const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            if (!previous.isValid())
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            const auto &old = *previous.state_;
            result.plan =
                createPlan(source, old.spec, old.fingerprint, old.inputPeak, options, job);
            result.status =
                result.plan.state_->active ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Generates and composes stereo, verifying rounded output before commit. */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink,
                                const OfflineJobOptions &jobOptions = {}) const
    {
        return renderImpl(source, plan, sink, nullptr, nullptr, jobOptions);
    }

    /** @brief Writes a canonical unit-width delta to a transactional host sink.
     * Exclusions and requested width do not affect this cache. Audio storage is
     * host-owned and excluded from the worker budget, like any sink callback.
     * A certificate is returned only after the delta sink commits successfully.
     */
    [[nodiscard]] CacheResult buildDeltaCache(OfflineAudioSource<T> &source, const Plan &plan,
                                              OfflineAudioSink<double> &sink,
                                              const OfflineJobOptions &jobOptions = {}) const
    {
        CacheResult result;
        detail::OfflineSinkTransaction<double> transaction;
        try
        {
            if (!plan.isValid())
                detail::offlineFail(OfflineStatus::InvalidInput);
            const auto &state = *plan.state_;
            if (source.getSpec() != state.spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            detail::OfflineSession job(jobOptions);
            job.checkpoint(OfflinePhase::Verify, 0, state.spec.frames);
            job.charge(sizeof(typename DeltaCache::State));
            DeltaCache candidate;
            candidate.state_ = std::make_unique<typename DeltaCache::State>();
            auto &cache = *candidate.state_;
            cache.sourceSpec = state.spec;
            cache.deltaSpec = state.spec;
            cache.deltaSpec.channels = 1;
            cache.sourceFingerprint = state.fingerprint;
            cache.generation = state.report.generation;
            cache.inputPeak = state.inputPeak;
            detail::OfflineStereoStream<T> stream(source, state.spec, state.fingerprint,
                                                  state.inputPeak, cache.generation,
                                                  state.inputPeak > 0, job, jobOptions.blockFrames);
            cache.latency = stream.latency();
            result.memoryBytes = job.bytes();
            transaction.sink = &sink;
            detail::offlineCallSink([&] { return sink.begin(cache.deltaSpec); });
            stream.run([&](std::int64_t first, AudioBufferView<T>, std::span<const double> delta) {
                for (const double sample : delta)
                {
                    if (!std::isfinite(sample))
                        detail::offlineFail(OfflineStatus::NumericalFailure);
                    detail::offlineHash(cache.deltaFingerprint, sample);
                    cache.deltaPeak = std::max(cache.deltaPeak, std::abs(sample));
                }
                const double *channels[]{delta.data()};
                detail::offlineCallSink([&] {
                    return sink.write(first, {channels, 1, static_cast<int>(delta.size())});
                });
            });
            job.checkpoint(OfflinePhase::Render, state.spec.frames, state.spec.frames);
            if (source.getSpec() != state.spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            detail::offlineCallSink([&] { return sink.commit(); });
            transaction.sink = nullptr;
            result.status = cache.deltaPeak > 0 ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.cache = std::move(candidate);
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Composes verified source/delta PCM with new width/exclusions in O(frames).
     * Both streams are verified even for neutral jobs. A changed source, clock,
     * generation setting or cache content rejects the entire provisional output.
     */
    [[nodiscard]] Result renderCached(OfflineAudioSource<T> &source, const Plan &plan,
                                      OfflineAudioSource<double> &delta, const DeltaCache &cache,
                                      OfflineAudioSink<T> &sink,
                                      const OfflineJobOptions &jobOptions = {}) const
    {
        return renderImpl(source, plan, sink, &delta, &cache, jobOptions);
    }

    /** @brief Owning render; failure preserves output, including in-place and mono duplication. */
    template <int MaxChannels>
    [[nodiscard]] Result run(const AudioBuffer<T, MaxChannels> &input,
                             AudioBuffer<T, MaxChannels> &output, double sampleRate,
                             const Options &options = {}, const OfflineJobOptions &job = {}) const
    {
        return detail::offlineRun(*this, input, output, sampleRate, options, job);
    }

  private:
    static void validate(OfflineAudioSpec spec, const Options &options)
    {
        detail::offlineValidateSpec(spec, 2);
        if (spec.channels == 1 && !options.duplicateMono)
            detail::offlineFail(OfflineStatus::UnsupportedLayout);
        if (!std::isfinite(options.width) || (options.width < 0) || (options.width > 1) ||
            StereoGenerator<double>::getPrepareMemoryBound(
                {spec.sampleRate, 256, 2}, {options.oversamplingFactor, options.lowCutHz}) == 0)
            detail::offlineFail(OfflineStatus::InvalidInput);
    }

    static Plan createPlan(OfflineAudioSource<T> &source, OfflineAudioSpec spec,
                           OfflineFingerprint fingerprint, double peak, const Options &options,
                           detail::OfflineSession &job)
    {
        validate(spec, options);
        if (source.getSpec() != spec)
            detail::offlineFail(OfflineStatus::SourceMismatch);
        job.checkpoint(OfflinePhase::Plan, 0, spec.frames);
        job.charge(sizeof(typename Plan::State));
        Plan plan;
        plan.state_ = std::make_unique<typename Plan::State>();
        auto &state = *plan.state_;
        state.spec = spec;
        state.outputSpec = spec;
        state.outputSpec.channels = 2;
        state.fingerprint = fingerprint;
        state.inputPeak = peak;
        state.featherFrames = std::max(1., std::round(.005 * spec.sampleRate));
        state.exclusions.assign(job, options.exclusions, spec.frames);
        auto &report = state.report;
        report.width = options.width;
        report.generation = {options.oversamplingFactor, options.lowCutHz};
        report.inputSamplePeakDb = gainToDecibels(peak, -std::numeric_limits<double>::infinity());
        report.duplicatedMono = spec.channels == 1;
        const bool excluded = state.exclusions.covers(spec.frames);
        state.active = options.width > 0 && peak > 0 && !excluded;
        report.reason = options.width == 0 ? Reason::ZeroWidth
                        : peak == 0        ? Reason::NoSignal
                        : excluded         ? Reason::AllExcluded
                                           : Reason::Ready;
        job.checkpoint(OfflinePhase::Plan, spec.frames, spec.frames);
        if (source.getSpec() != spec)
            detail::offlineFail(OfflineStatus::SourceMismatch);
        return plan;
    }

    static Result renderImpl(OfflineAudioSource<T> &source, const Plan &plan,
                             OfflineAudioSink<T> &sink, OfflineAudioSource<double> *deltaSource,
                             const DeltaCache *certificate, const OfflineJobOptions &options)
    {
        Result result;
        detail::OfflineSinkTransaction<T> transaction;
        try
        {
            if (!plan.isValid() || (certificate && !certificate->isValid()))
                detail::offlineFail(OfflineStatus::InvalidInput);
            const auto &state = *plan.state_;
            const auto &spec = state.spec;
            if (source.getSpec() != spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            const auto *cache = certificate ? certificate->state_.get() : nullptr;
            if (cache &&
                (cache->sourceSpec != spec || cache->sourceFingerprint != state.fingerprint ||
                 cache->inputPeak != state.inputPeak ||
                 cache->generation != state.report.generation ||
                 deltaSource->getSpec() != cache->deltaSpec))
                detail::offlineFail(OfflineStatus::SourceMismatch);
            result.report = state.report;
            auto &report = result.report;
            report.usedDeltaCache = cache != nullptr;
            detail::OfflineSession job(options);
            job.checkpoint(OfflinePhase::Verify, 0, spec.frames);
            const int limit = std::min(256, options.blockFrames);
            detail::OfflineBlock<T> output(job, state.outputSpec, limit);
            std::optional<detail::OfflineStereoStream<T>> stream;
            std::optional<detail::OfflineBlock<T>> cacheOriginal;
            std::optional<detail::OfflineBlock<double>> cacheDelta;
            if (cache)
            {
                cacheOriginal.emplace(job, spec, limit);
                cacheDelta.emplace(job, cache->deltaSpec, limit);
                report.compensatedLatencyFrames = cache->latency;
            }
            else
            {
                stream.emplace(source, spec, state.fingerprint, state.inputPeak, report.generation,
                               state.active, job, limit);
                report.compensatedLatencyFrames = stream->latency();
            }
            result.memoryBytes = job.bytes();
            TruePeakDetector<double, 2> detector;
            const double scale = state.inputPeak > 0 ? state.inputPeak : 1;
            double peak = 0, truePeak = 0;
            transaction.sink = &sink;
            detail::offlineCallSink([&] { return sink.begin(state.outputSpec); });
            const auto compose = [&](std::int64_t first, AudioBufferView<T> raw,
                                     std::span<const double> delta) {
                const int count = raw.getNumSamples();
                auto out = output.view(count);
                for (int i = 0; i < count; ++i)
                {
                    const double mask =
                        state.active ? 1 - state.exclusions.apply(first + i, 0, state.featherFrames)
                                     : 0;
                    const double change = mask == 0 ? 0 : delta[i] * report.width * mask;
                    for (int c = 0; c < 2; ++c)
                    {
                        const T original = raw.getChannel(c % spec.channels)[i];
                        T value = original;
                        if (change != 0)
                        {
                            const double processed =
                                static_cast<double>(original) + (c == 0 ? change : -change);
                            if (!std::isfinite(processed) ||
                                std::abs(processed) > std::numeric_limits<T>::max())
                                detail::offlineFail(OfflineStatus::NumericalFailure);
                            value = static_cast<T>(processed);
                        }
                        out.getChannel(c)[i] = value;
                        const double normalized = static_cast<double>(value) / scale;
                        peak = std::max(peak, std::abs(normalized));
                        truePeak = std::max(truePeak, detector.processSample(normalized, c));
                    }
                }
                detail::offlineCallSink([&] { return sink.write(first, out); });
            };
            if (stream)
                stream->run(compose);
            else
            {
                OfflineFingerprint originalFingerprint, deltaFingerprint;
                job.checkpoint(OfflinePhase::Render, 0, spec.frames);
                for (std::int64_t first = 0; first < spec.frames;)
                {
                    const int count = static_cast<int>(
                        std::min<std::int64_t>(cacheOriginal->frames(), spec.frames - first));
                    auto original = cacheOriginal->view(count);
                    auto delta = cacheDelta->view(count);
                    detail::offlineRead(source, spec, first, original);
                    detail::offlineRead(*deltaSource, cache->deltaSpec, first, delta);
                    for (int i = 0; i < count; ++i)
                    {
                        for (int c = 0; c < spec.channels; ++c)
                        {
                            const T sample = original.getChannel(c)[i];
                            if (!std::isfinite(sample))
                                detail::offlineFail(OfflineStatus::NonFiniteInput);
                            if (std::abs(static_cast<double>(sample)) > state.inputPeak)
                                detail::offlineFail(OfflineStatus::SourceMismatch);
                            detail::offlineHash(originalFingerprint, sample);
                        }
                        const double sample = delta.getChannel(0)[i];
                        if (!std::isfinite(sample) || std::abs(sample) > cache->deltaPeak)
                            detail::offlineFail(OfflineStatus::SourceMismatch);
                        detail::offlineHash(deltaFingerprint, sample);
                    }
                    compose(first, original,
                            {delta.getChannel(0), static_cast<std::size_t>(count)});
                    first += count;
                    job.checkpoint(OfflinePhase::Render, first, spec.frames);
                }
                if (originalFingerprint != state.fingerprint ||
                    deltaFingerprint != cache->deltaFingerprint || source.getSpec() != spec ||
                    deltaSource->getSpec() != cache->deltaSpec)
                    detail::offlineFail(OfflineStatus::SourceMismatch);
            }
            for (int i = 0; i < TruePeakDetector<double, 2>::getTaps() - 1; ++i)
                for (int c = 0; c < 2; ++c)
                    truePeak = std::max(truePeak, detector.processSample(0, c));
            if (!std::isfinite(peak) || !std::isfinite(truePeak))
                detail::offlineFail(OfflineStatus::NumericalFailure);
            constexpr double silence = -std::numeric_limits<double>::infinity();
            report.outputSamplePeakDb =
                peak > 0 ? gainToDecibels(peak, silence) + gainToDecibels(scale, silence) : silence;
            report.outputTruePeakDb =
                truePeak > 0 ? gainToDecibels(truePeak, silence) + gainToDecibels(scale, silence)
                             : silence;
            report.peaksMeasured = true;
            if (state.active)
                report.reason = Reason::Generated;
            job.checkpoint(OfflinePhase::Render, spec.frames, spec.frames);
            if (source.getSpec() != spec || (cache && deltaSource->getSpec() != cache->deltaSpec))
                detail::offlineFail(OfflineStatus::SourceMismatch);
            detail::offlineCallSink([&] { return sink.commit(); });
            transaction.sink = nullptr;
            result.status = state.active ? OfflineStatus::Success : OfflineStatus::NoChange;
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
