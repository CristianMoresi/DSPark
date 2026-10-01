// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflineTempoAnalyzer.h
 * @brief Source-bound tempo and beat intervals from shared complete-file features.
 *
 * OFFLINE ONLY. Reuses BeatTracker's conditioning, metrical ranking and beat-grid
 * engine. Retained OfflineTransientAnalyzer features avoid another source scan
 * or FFT. This analyzer never owns audio. Beat positions are interpolated
 * spectral-feature centers, not exact acoustic attack boundaries; the separate
 * pulse map remains the source of event-local processing positions.
 *
 * Threading: synchronous worker calls. Completed Analysis is immutable and
 * move-only; borrowed spans remain valid while its owner is alive and unmoved.
 * The complete engine's allocations are charged before requesting storage.
 * Cooperative progress uses total=0 during iterative tempo/grid work.
 */
#include "OfflineTransientAnalyzer.h"
#if DSPARK_HAS_OFFLINE
#include "detail/OfflineBeatEngine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace dspark
{

/** @brief Complete-source tempo map for automatic offline processors. */
template <FloatType T> class OfflineTempoAnalyzer final
{
  public:
    static constexpr std::uint32_t algorithmRevision = 2;
    struct Options
    {
        double minimumBpm = 40, maximumBpm = 240; ///< Search range within 20..480 BPM.
        double minimumConfidence = 0.25; ///< Explicit ambiguity gate in [0,1].
    };
    struct Beat
    {
        std::int64_t frame = 0; ///< Source-relative grid position, not timelineOrigin + frame.
        double nextIntervalBpm = 0; ///< Three-interval median tempo; zero on the last point.
        bool reliable = false; ///< Global/local confidence, consistent interval and pulse anchors.
    };
    enum class Reason : std::uint8_t { Tracked, Uncertain, NoRhythm, NoSignal };
    struct Report
    {
        Reason reason = Reason::NoRhythm;
        double tempoBpm = 0, secondaryTempoBpm = 0, confidence = 0;
        std::size_t reliableIntervals = 0;
        std::size_t localWindows = 0, locallyConfirmedIntervals = 0;
    };
    class Analysis final
    {
      public:
        Analysis() = default;
        Analysis(Analysis &&) noexcept = default;
        Analysis &operator=(Analysis &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept { return beats_ != nullptr; }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept { return spec_; }
        [[nodiscard]] OfflineFingerprint fingerprint() const noexcept { return fingerprint_; }
        [[nodiscard]] Options getOptions() const noexcept { return options_; }
        [[nodiscard]] Report getReport() const noexcept { return report_; }
        [[nodiscard]] std::span<const Beat> beats() const noexcept
        {
            return isValid() ? std::span<const Beat>(beats_.get(), count_)
                             : std::span<const Beat>();
        }
        /** @brief Reliable local tempo, or zero outside an anchored interval. */
        [[nodiscard]] double tempoAt(std::int64_t frame) const noexcept
        {
            const auto map = beats();
            if (map.size() < 2 || frame < map.front().frame || frame >= map.back().frame)
                return 0;
            auto next = std::upper_bound(map.begin(), map.end(), frame,
                                        [](std::int64_t p, const Beat &b) { return p < b.frame; });
            const auto &beat = *(next - 1);
            return beat.reliable ? beat.nextIntervalBpm : 0;
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid() ? std::max(std::size_t(1), count_) * sizeof(Beat) : 0;
        }

      private:
        friend class OfflineTempoAnalyzer;
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_;
        Options options_;
        Report report_;
        std::unique_ptr<Beat[]> beats_;
        std::size_t count_ = 0;
    };
    struct Result
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Analysis analysis;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept { return offlineSucceeded(status); }
    };

    /** @brief Reuses immutable features; retainFeatures and pulses must be enabled. */
    [[nodiscard]] Result
    analyze(const typename OfflineTransientAnalyzer<T>::Analysis &features, Options options = {},
            const OfflineJobOptions &jobOptions = {}) const
    {
        Result result;
        try
        {
            validate(options);
            detail::OfflineSession job(jobOptions);
            if (!features.isValid() || !features.getOptions().retainFeatures ||
                !features.getOptions().pulses)
                detail::offlineFail(OfflineStatus::InvalidInput);
            Analysis candidate;
            candidate.spec_ = features.energy().getSpec();
            candidate.fingerprint_ = features.energy().fingerprint();
            candidate.options_ = options;
            const auto frames = features.features();
            const auto tracked = detail::OfflineBeatEngine<T>::analyze(
                candidate.spec_.sampleRate, features.hopFrames(), frames.size(),
                [&](std::size_t i) {
                    typename OnsetDetector<T>::OdfFrame frame;
                    frame.value = frames[i].attack;
                    frame.registers = frames[i].registers;
                    frame.referenceSample = features.featureCenter(i);
                    return frame;
                },
                options.minimumBpm, options.maximumBpm, 25, job);
            candidate.report_.tempoBpm = static_cast<double>(tracked.tempoBpm);
            candidate.report_.secondaryTempoBpm = static_cast<double>(tracked.secondaryTempoBpm);
            candidate.report_.confidence = static_cast<double>(tracked.confidence);
            for (const auto frame : tracked.beats())
                if (frame >= 0 && frame < candidate.spec_.frames)
                    ++candidate.count_;
            candidate.beats_ = job.template allocate<Beat>(std::max(std::size_t(1), candidate.count_));
            std::size_t index = 0;
            for (const auto frame : tracked.beats())
                if (frame >= 0 && frame < candidate.spec_.frames)
                    candidate.beats_[index++].frame = frame;
            for (std::size_t i = 0; i + 1 < candidate.count_; ++i)
            {
                auto &beat = candidate.beats_[i];
                const auto next = candidate.beats_[i + 1].frame;
                if (next <= beat.frame)
                    detail::offlineFail(OfflineStatus::NumericalFailure);
                const double interval = static_cast<double>(next - beat.frame);
                std::array<double, 3> local{interval, interval, interval};
                if (i)
                    local[0] = static_cast<double>(beat.frame - candidate.beats_[i - 1].frame);
                if (i + 2 < candidate.count_)
                    local[2] = static_cast<double>(candidate.beats_[i + 2].frame - next);
                std::sort(local.begin(), local.end());
                const double quarter = local[1];
                beat.nextIntervalBpm = 60 * candidate.spec_.sampleRate / quarter;
                const double tolerance = std::min(0.05 * candidate.spec_.sampleRate, 0.125 * interval);
                // The search range constrains nominal metrical candidates,
                // not individual intervals of the delivered expressive grid.
                // Local timing must not spuriously lose confidence at a range
                // endpoint, or be clamped to a tempo the grid did not produce.
                beat.reliable = candidate.report_.confidence >= options.minimumConfidence &&
                                std::abs(interval - quarter) <= 2 * tolerance &&
                                anchored(features.pulses(), beat.frame, tolerance) &&
                                anchored(features.pulses(), next, tolerance);
                candidate.report_.reliableIntervals += beat.reliable ? 1 : 0;
            }
            confirmLocalIntervals(features, candidate, job);
            candidate.report_.reason =
                features.energy().samplePeak() == 0 ? Reason::NoSignal
                : candidate.count_ < 2              ? Reason::NoRhythm
                : candidate.report_.reliableIntervals ? Reason::Tracked : Reason::Uncertain;
            job.checkpoint(OfflinePhase::Analyze, candidate.spec_.frames, candidate.spec_.frames);
            result.memoryBytes = job.bytes();
            result.analysis = std::move(candidate);
            result.status = OfflineStatus::Success;
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Builds shared features then tempo; two bounded source scans, no retained PCM. */
    [[nodiscard]] Result analyze(OfflineAudioSource<T> &source, Options options = {},
                                 const OfflineJobOptions &jobOptions = {}) const
    {
        Result result;
        try
        {
            validate(options);
            auto features = OfflineTransientAnalyzer<T>().analyze(source, {false, true, true},
                                                                 jobOptions);
            if (!features.succeeded())
                detail::offlineFail(features.status);
            result = analyze(features.analysis, options,
                             detail::offlineRemaining(jobOptions, features.memoryBytes));
            if (!result.succeeded())
                return result;
            if (source.getSpec() != result.analysis.getSpec())
                detail::offlineFail(OfflineStatus::SourceMismatch);
            result.memoryBytes += features.memoryBytes;
        }
        catch (...)
        {
            result = Result{};
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

  private:
    static void confirmLocalIntervals(
        const typename OfflineTransientAnalyzer<T>::Analysis &features, Analysis &candidate,
        detail::OfflineSession &job)
    {
        if (candidate.count_ < 2 || candidate.report_.confidence >= candidate.options_.minimumConfidence)
            return;
        const auto frames = features.features();
        const auto window = static_cast<std::size_t>(std::ceil(
            std::max(8., 240 / candidate.options_.minimumBpm) * candidate.spec_.sampleRate /
            features.hopFrames()));
        if (frames.size() <= window)
            return;
        detail::OfflineBeatMemory workspace(job);
        for (std::size_t begin = 0;;)
        {
            const auto local = detail::OfflineBeatEngine<T>::analyze(
                candidate.spec_.sampleRate, features.hopFrames(), window,
                [&](std::size_t i) {
                    typename OnsetDetector<T>::OdfFrame frame;
                    frame.value = frames[begin + i].attack;
                    frame.registers = frames[begin + i].registers;
                    frame.referenceSample = features.featureCenter(begin + i);
                    return frame;
                }, candidate.options_.minimumBpm, candidate.options_.maximumBpm, 25, job, &workspace);
            ++candidate.report_.localWindows;
            const auto grid = local.beats();
            if (local.confidence >= candidate.options_.minimumConfidence && grid.size() >= 2)
            {
                const auto first = std::lower_bound(candidate.beats_.get(),
                    candidate.beats_.get() + candidate.count_, grid.front(),
                    [](const Beat &beat, auto p) { return beat.frame < p; });
                auto index = static_cast<std::size_t>(first - candidate.beats_.get());
                if (index)
                    --index;
                for (; index + 1 < candidate.count_ && candidate.beats_[index].frame <= grid.back(); ++index)
                {
                    auto &beat = candidate.beats_[index];
                    if (beat.reliable)
                        continue;
                    const auto end = candidate.beats_[index + 1].frame;
                    const double interval = static_cast<double>(end - beat.frame);
                    const double quarter = 60 * candidate.spec_.sampleRate / beat.nextIntervalBpm;
                    const double tolerance = std::min(.05 * candidate.spec_.sampleRate, .125 * interval);
                    if (std::abs(interval - quarter) > 2 * tolerance ||
                        !anchored(features.pulses(), beat.frame, tolerance) ||
                        !anchored(features.pulses(), end, tolerance))
                        continue;
                    auto position = std::lower_bound(grid.begin(), grid.end(), beat.frame);
                    if (position == grid.end() || (position != grid.begin() &&
                        beat.frame - *(position - 1) < *position - beat.frame))
                        --position;
                    if (position + 1 == grid.end() ||
                        std::abs(static_cast<double>(*position - beat.frame)) > tolerance ||
                        std::abs(static_cast<double>(*(position + 1) - end)) > tolerance)
                        continue;
                    beat.reliable = true;
                    ++candidate.report_.reliableIntervals;
                    ++candidate.report_.locallyConfirmedIntervals;
                }
            }
            if (begin + window == frames.size())
                break;
            begin = std::min(begin + window / 2, frames.size() - window);
        }
    }
    static void validate(Options options)
    {
        if (!std::isfinite(options.minimumBpm) || !std::isfinite(options.maximumBpm) ||
            (options.minimumBpm < 20) || (options.maximumBpm > 480) ||
            options.minimumBpm > options.maximumBpm || !std::isfinite(options.minimumConfidence) ||
            (options.minimumConfidence < 0) || (options.minimumConfidence > 1))
            detail::offlineFail(OfflineStatus::InvalidInput);
    }
    [[nodiscard]] static bool
    anchored(std::span<const typename OfflineTransientAnalyzer<T>::Event> pulses,
             std::int64_t frame, double tolerance) noexcept
    {
        const auto next = std::lower_bound(pulses.begin(), pulses.end(), frame,
                                          [](const auto &event, auto p) { return event.begin < p; });
        return (next != pulses.end() && static_cast<double>(next->begin - frame) <= tolerance) ||
               (next != pulses.begin() && static_cast<double>(frame - (next - 1)->begin) <= tolerance);
    }
};

} // namespace dspark
#endif
