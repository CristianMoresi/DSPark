// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflineBeatCompressor.h
 * @brief Automatic pulse leveling with a source-bound local tempo release.
 *
 * OFFLINE ONLY. Each detected pulse's first 30 ms is measured across channels.
 * Peaks above the upper median are reduced by at most reductionDb. A 4 ms
 * preroll and the shared 2 ms finite attack anticipate each pulse; its measured
 * body is held. The common exponential amplitude-recovery clock follows local
 * reliable tempo, changing at hold ends. Unknown tempo uses 250 ms explicitly.
 * Overlaps use minimum gain, never multiplied attenuation. No makeup, limiting,
 * whole-file trim, audio delay or oversampling is applied.
 *
 * Reuses shared transient/tempo analysis, OfflineAttenuation and the bandlimited
 * gain renderer. The reduction cap bounds the control, not the ratio of each
 * rendered sample to its input: bandlimited multiplication has interpolation
 * tails. Exact PCM is preserved for zero amount and explicit exclusions.
 *
 * Threading: synchronous worker calls only. The facade is stateless. Completed
 * plans own immutable sparse events and are shareable while alive and unmoved.
 */
#include "../Analysis/OfflineTempoAnalyzer.h"
#if DSPARK_HAS_OFFLINE
#include "../Core/detail/OfflineAttenuation.h"
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
/** @brief Complete-source pulse compression with one positive reduction-dB control. */
template <FloatType T> class OfflineBeatCompressor final
{
  public:
    enum class Note : std::uint8_t { Eighth, Quarter, Half, Whole };
    enum class TempoSource : std::uint8_t { Local, Manual, Fallback };
    struct Options
    {
        double reductionDb = 0; ///< Maximum event attenuation, 0..18 dB.
        Note releaseNote = Note::Quarter;
        double manualBpm = 0; ///< Zero selects analysis; an override accepts 20..480 BPM.
        typename OfflineTempoAnalyzer<T>::Options tempo; ///< Used when constructing a tempo map.
        std::span<const OfflineRegion> exclusions{};
    };
    struct Pulse
    {
        std::int64_t begin = 0, end = 0; ///< Original source event, before anticipation.
        double peakDb = -std::numeric_limits<double>::infinity();
        double reductionDb = 0, releaseMs = 250;
        TempoSource tempoSource = TempoSource::Fallback;
    };
    enum class Reason : std::uint8_t
    {
        Compressed, EvenPulses, ZeroAmount, NoSignal, NoPulses, AllExcluded
    };
    struct Report
    {
        Reason reason = Reason::NoPulses;
        double requestedReductionDb = 0, maximumReductionDb = 0, maximumExcessDb = 0;
        double referencePeakDb = -std::numeric_limits<double>::infinity();
        double tempoBpm = 0, tempoConfidence = 0;
        double inputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        std::size_t pulses = 0, reducedPulses = 0, fallbackPulses = 0;
        bool peaksMeasured = false;
        OfflineGainRenderInfo renderInfo;
    };
    class Plan final
    {
      public:
        Plan() = default;
        Plan(Plan &&) noexcept = default;
        Plan &operator=(Plan &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept { return curve_.isValid(); }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept { return spec_; }
        [[nodiscard]] Report getReport() const noexcept { return report_; }
        [[nodiscard]] std::span<const Pulse> pulses() const noexcept
        {
            return isValid() ? std::span<const Pulse>(pulses_.get(), report_.pulses)
                             : std::span<const Pulse>();
        }
        [[nodiscard]] std::span<const OfflineRegion> exclusions() const noexcept
        {
            return isValid() ? exclusions_.view() : std::span<const OfflineRegion>();
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid() ? curve_.retainedBytes() + exclusions_.retainedBytes() +
                                  report_.pulses * sizeof(Pulse) : 0;
        }
        [[nodiscard]] double gainAt(std::int64_t frame) const noexcept
        {
            return !isValid() || frame < 0 || frame >= spec_.frames
                       ? 1 : exclusions_.apply(frame, curve_.gainAt(frame), featherFrames_);
        }
      private:
        friend class OfflineBeatCompressor;
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_;
        Report report_;
        double inputPeak_ = 0, featherFrames_ = 1;
        detail::OfflineAttenuation curve_;
        detail::OfflineExclusions exclusions_;
        std::unique_ptr<Pulse[]> pulses_;
    };
    struct PlanResult
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Plan plan;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept { return offlineSucceeded(status); }
    };
    struct Result
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Report report;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept { return offlineSucceeded(status); }
    };

    /**
     * @brief Reuses source-bound pulse and tempo maps; one verified PCM scan.
     * A valid tempo map must match the source fingerprint even with manual BPM.
     * An empty tempo map is accepted only with manual BPM or a zero amount.
     */
    [[nodiscard]] PlanResult makePlan(
        OfflineAudioSource<T> &source,
        const typename OfflineTransientAnalyzer<T>::Analysis &analysis,
        const typename OfflineTempoAnalyzer<T>::Analysis &tempo, const Options &options = {},
        const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            validate(options);
            if (!analysis.isValid() || (options.reductionDb > 0 && !analysis.getOptions().pulses) ||
                (options.reductionDb > 0 && options.manualBpm == 0 && !tempo.isValid()))
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            Plan plan;
            plan.spec_ = analysis.energy().getSpec();
            plan.fingerprint_ = analysis.energy().fingerprint();
            if (source.getSpec() != plan.spec_ ||
                (tempo.isValid() && (tempo.getSpec() != plan.spec_ ||
                                    tempo.fingerprint() != plan.fingerprint_)))
                detail::offlineFail(OfflineStatus::SourceMismatch);
            plan.inputPeak_ = analysis.energy().samplePeak();
            const auto attack = std::max(std::int64_t(1),
                static_cast<std::int64_t>(std::llround(.002 * plan.spec_.sampleRate)));
            const auto preroll = static_cast<std::int64_t>(std::llround(.004 * plan.spec_.sampleRate));
            plan.featherFrames_ = static_cast<double>(attack + preroll);
            auto &report = plan.report_;
            report.requestedReductionDb = options.reductionDb;
            report.inputSamplePeakDb = gainToDecibels(plan.inputPeak_, silence);
            report.tempoBpm = options.manualBpm > 0 ? options.manualBpm : tempo.getReport().tempoBpm;
            report.tempoConfidence = tempo.isValid() ? tempo.getReport().confidence : 0;
            job.checkpoint(OfflinePhase::Plan, 0, plan.spec_.frames);
            plan.exclusions_.assign(job, options.exclusions, plan.spec_.frames);
            const auto events = analysis.pulses();
            const bool excluded = plan.exclusions_.covers(plan.spec_.frames);
            const bool active = options.reductionDb > 0 && plan.inputPeak_ > 0 &&
                                !events.empty() && !excluded;
            plan.curve_.prepare(job, active ? events.size() : 0, attack,
                                .25 * plan.spec_.sampleRate, active);
            if (!active)
                report.reason = excluded ? Reason::AllExcluded
                              : options.reductionDb == 0 ? Reason::ZeroAmount
                              : plan.inputPeak_ == 0 ? Reason::NoSignal : Reason::NoPulses;
            else
            {
                report.pulses = events.size();
                plan.pulses_ = job.template allocate<Pulse>(events.size());
                auto peaks = job.template allocate<double>(events.size());
                detail::OfflineBlock<T> scratch(job, plan.spec_, jobOptions.blockFrames);
                const auto window = std::max(std::int64_t(1),
                    static_cast<std::int64_t>(std::llround(.030 * plan.spec_.sampleRate)));
                std::size_t first = 0, next = 0;
                detail::offlineScanMagnitude(source, plan.spec_, plan.fingerprint_, plan.inputPeak_,
                    scratch, job, [&](std::int64_t frame, double magnitude) {
                        while (next < events.size() && events[next].begin <= frame)
                            ++next;
                        while (first < next && frame - events[first].begin >= window)
                            ++first;
                        for (auto i = first; i < next; ++i)
                            peaks[i] = std::max(peaks[i], magnitude);
                    });
                for (std::size_t i = 0; i < events.size(); ++i)
                    plan.pulses_[i].peakDb = gainToDecibels(peaks[i], silence);
                std::nth_element(peaks.get(), peaks.get() + events.size() / 2,
                                  peaks.get() + events.size());
                report.referencePeakDb = gainToDecibels(peaks[events.size() / 2], silence);
                for (std::size_t i = 0; i < events.size(); ++i)
                {
                    if ((i & 1023) == 0)
                        job.checkpoint(OfflinePhase::Plan, 0, plan.spec_.frames);
                    auto &pulse = plan.pulses_[i];
                    pulse.begin = events[i].begin;
                    pulse.end = events[i].end;
                    const double excess = pulse.peakDb > report.referencePeakDb
                                              ? pulse.peakDb - report.referencePeakDb : 0;
                    pulse.reductionDb = std::min(options.reductionDb, excess);
                    double bpm = options.manualBpm;
                    if (bpm > 0)
                        pulse.tempoSource = TempoSource::Manual;
                    else
                    {
                        bpm = pulseTempo(tempo, pulse.begin);
                        if (bpm > 0)
                            pulse.tempoSource = TempoSource::Local;
                    }
                    pulse.releaseMs = bpm > 0 ? noteBeats(options.releaseNote) * 60000 / bpm : 250;
                    report.fallbackPulses += bpm > 0 ? 0 : 1;
                    report.reducedPulses += pulse.reductionDb > 0 ? 1 : 0;
                    report.maximumExcessDb = std::max(report.maximumExcessDb, excess);
                    report.maximumReductionDb = std::max(report.maximumReductionDb, pulse.reductionDb);
                    plan.curve_.set(i, std::max(std::int64_t(0), pulse.begin - preroll), pulse.end,
                                    decibelsToGain(-pulse.reductionDb),
                                    pulse.releaseMs * .001 * plan.spec_.sampleRate);
                }
                report.reason = report.reducedPulses ? Reason::Compressed : Reason::EvenPulses;
            }
            job.checkpoint(OfflinePhase::Plan, plan.spec_.frames, plan.spec_.frames);
            result.status = report.maximumReductionDb > 0 ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
            result.plan = std::move(plan);
        }
        catch (...) { result.status = detail::offlineExceptionStatus(); }
        return result;
    }

    /** @brief Reuses retained features to build tempo; manual BPM skips the tempo engine. */
    [[nodiscard]] PlanResult makePlan(
        OfflineAudioSource<T> &source,
        const typename OfflineTransientAnalyzer<T>::Analysis &analysis, const Options &options = {},
        const OfflineJobOptions &job = {}) const
    {
        PlanResult result;
        try
        {
            validate(options);
            if (options.manualBpm > 0 || options.reductionDb == 0)
                return makePlan(source, analysis, {}, options, job);
            auto tempo = OfflineTempoAnalyzer<T>().analyze(analysis, options.tempo, job);
            if (!tempo.succeeded())
                detail::offlineFail(tempo.status);
            result = makePlan(source, analysis, tempo.analysis, options,
                              detail::offlineRemaining(job, tempo.memoryBytes));
            result.memoryBytes += tempo.memoryBytes;
        }
        catch (...) { result.status = detail::offlineExceptionStatus(); }
        return result;
    }

    /** @brief Complete-source automatic analysis and planning, with no retained audio. */
    [[nodiscard]] PlanResult analyze(OfflineAudioSource<T> &source, const Options &options = {},
                                     const OfflineJobOptions &job = {}) const
    {
        PlanResult result;
        try
        {
            validate(options);
            auto analyzed = OfflineTransientAnalyzer<T>().analyze(
                source, {false, options.reductionDb > 0,
                         options.reductionDb > 0 && options.manualBpm == 0}, job);
            if (!analyzed.succeeded())
                detail::offlineFail(analyzed.status);
            result = makePlan(source, analyzed.analysis, options,
                              detail::offlineRemaining(job, analyzed.memoryBytes));
            result.memoryBytes += analyzed.memoryBytes;
        }
        catch (...) { result.status = detail::offlineExceptionStatus(); }
        return result;
    }

    /** @brief Verified transactional render through the shared bandlimited gain operator. */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink, const OfflineJobOptions &job = {}) const
    {
        detail::OfflineGainRenderSettings settings{&plan.exclusions_, plan.featherFrames_, false};
        if (plan.isValid() && !plan.pulses().empty())
        {
            const auto window = std::max(std::int64_t(1),
                static_cast<std::int64_t>(std::llround(.030 * plan.spec_.sampleRate)));
            const std::array<const Pulse *, 2> edge{&plan.pulses().front(), &plan.pulses().back()};
            for (int side = 0; side < 2; ++side)
            {
                const auto &pulse = *edge[side];
                const bool touches = side == 0
                    ? pulse.begin <= plan.featherFrames_ : pulse.end == plan.spec_.frames;
                if (pulse.reductionDb > 0 && touches)
                {
                    const auto end = pulse.begin + std::min(window, plan.spec_.frames - pulse.begin);
                    settings.boundaryTargets[side] = {
                        {pulse.begin, end}, decibelsToGain(pulse.peakDb -
                            plan.report_.inputSamplePeakDb - pulse.reductionDb)};
                }
            }
        }
        return detail::offlineRenderGain<Result>(
            source, plan.spec_, plan.fingerprint_, plan.inputPeak_, plan.isValid(),
            plan.report_.maximumReductionDb > 0, plan.report_, sink, job,
            [&plan](detail::OfflineSession &session) {
                return [&plan, cursor = detail::OfflineAttenuation::Cursor(plan.curve_, session)]
                    (std::int64_t frame) mutable {
                        return plan.exclusions_.apply(frame, cursor(frame), plan.featherFrames_);
                    };
            }, settings);
    }
    /** @brief Owning convenience operation; aliases are safe and failure preserves output. */
    template <int MaxChannels>
    [[nodiscard]] Result run(const AudioBuffer<T, MaxChannels> &input,
                             AudioBuffer<T, MaxChannels> &output, double sampleRate,
                             const Options &options = {}, const OfflineJobOptions &job = {}) const
    {
        return detail::offlineRun(*this, input, output, sampleRate, options, job);
    }
  private:
    static constexpr double silence = -std::numeric_limits<double>::infinity();
    static double noteBeats(Note note) noexcept
    {
        switch (note)
        {
            case Note::Eighth: return .5;
            case Note::Quarter: return 1;
            case Note::Half: return 2;
            case Note::Whole: return 4;
        }
        return 0;
    }
    static void validate(const Options &options)
    {
        if (!std::isfinite(options.reductionDb) || (options.reductionDb < 0) || (options.reductionDb > 18) ||
            !std::isfinite(options.manualBpm) || (options.manualBpm != 0 &&
                ((options.manualBpm < 20) || (options.manualBpm > 480))) || noteBeats(options.releaseNote) == 0 ||
            !std::isfinite(options.tempo.minimumBpm) || !std::isfinite(options.tempo.maximumBpm) ||
            (options.tempo.minimumBpm < 20) || (options.tempo.maximumBpm > 480) ||
            options.tempo.minimumBpm > options.tempo.maximumBpm ||
            !std::isfinite(options.tempo.minimumConfidence) || (options.tempo.minimumConfidence < 0) ||
            (options.tempo.minimumConfidence > 1))
            detail::offlineFail(OfflineStatus::InvalidInput);
    }
    static double pulseTempo(const typename OfflineTempoAnalyzer<T>::Analysis &tempo,
                             std::int64_t frame) noexcept
    {
        const double local = tempo.tempoAt(frame);
        if (local > 0)
            return local;
        const auto beats = tempo.beats();
        if (beats.size() < 2)
            return 0;
        const auto next = std::lower_bound(beats.begin(), beats.end(), frame,
            [](const auto &beat, auto p) { return beat.frame < p; });
        const double tolerance = .05 * tempo.getSpec().sampleRate;
        const auto withinTolerance = [&](auto beat) {
            return std::abs(static_cast<double>(beat->frame - frame)) <= tolerance;
        };
        if (next != beats.end() && withinTolerance(next) && next->reliable)
            return next->nextIntervalBpm;
        if (next != beats.begin())
        {
            const auto previous = next - 1;
            if (previous != beats.begin() && withinTolerance(previous) && (previous - 1)->reliable)
                return (previous - 1)->nextIntervalBpm;
            if (next != beats.end() && withinTolerance(next) && previous->reliable)
                return previous->nextIntervalBpm;
        }
        // A boundary pulse may precede the first or follow the last grid point
        // by one measured quarter note. Never extrapolate through an interior gap.
        const auto &edge = frame < beats.front().frame ? beats.front() : beats[beats.size() - 2];
        const auto boundary = frame < beats.front().frame ? beats.front().frame : beats.back().frame;
        if ((frame < beats.front().frame || frame >= beats.back().frame) && edge.reliable)
        {
            const double distance = std::abs(static_cast<double>(frame - boundary));
            const double period = 60 * tempo.getSpec().sampleRate / edge.nextIntervalBpm;
            if (distance <= tolerance || std::abs(distance - period) <= tolerance)
                return edge.nextIntervalBpm;
        }
        return 0;
    }
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
