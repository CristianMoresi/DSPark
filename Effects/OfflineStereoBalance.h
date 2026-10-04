// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineStereoBalance.h
 * @brief Complete-source side-energy leveling and an optional windowed ratio
 * guard.
 *
 * OFFLINE ONLY. Supply the actual stereo PCM after generation at the chosen
 * width. The generator itself performs no balance analysis. This processor
 * changes side only, using the shared bandlimited gain renderer; original mid
 * is preserved up to output rounding. There is no limiter, master trim or genre
 * preset.
 *
 * Energy analysis uses 10 ms cells. Leveling follows a 410 ms centered energy
 * context and two centered 101-cell averages of log gain at ordinary audio
 * rates. The guard checks forward 40 ms windows on that grid, including short
 * final windows. Its bound concerns actual output window energies, not
 * individual samples, spectral bands or perceived width. Silent side windows
 * are gated at the larger of -140 dBFS power and -60 dB relative to source mean
 * M+S power.
 *
 * Controls and source provenance are immutable in a completed Plan. Rendering
 * verifies actual rounded stereo PCM before publication, refining local guard
 * attenuation if necessary. An unreachable bound returns TargetUnreachable and
 * aborts output; absence of mid cannot be repaired by side processing.
 * Threading: synchronous worker calls, never real-time audio callbacks.
 */
#include "../Analysis/OfflineEnergyAnalyzer.h"
#if DSPARK_HAS_OFFLINE
#include "../Core/detail/BoxAverage.h"
#include "../Core/detail/OfflineGain.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <utility>

namespace dspark
{
/** @brief Optional offline balance of an already generated stereo signal. */
template <FloatType T> class OfflineStereoBalance final
{
  public:
    struct Options
    {
        double levelingAmount = 0; ///< 0..1 of the measured log-gain correction.
        bool guard = false;        ///< Verify the actual windowed side/mid energy ceiling.
        std::optional<double> targetSideShare; ///< Energy fraction .001..0.49;
                                               ///< absent: measured, clamped.
        double guardMarginDb = 3;              ///< 0..12 dB above the target side/mid power ratio.
        double maximumBoostDb = 12;            ///< 0..24 dB leveler boost cap.
        double maximumCutDb = 24;              ///< 0..60 dB leveler attenuation cap.
        double maximumGuardCutDb = 60;         ///< 0..120 dB additional guard attenuation cap.
        std::span<const OfflineRegion> exclusions{}; ///< Exact source-relative protected PCM.
    };
    enum class Reason : std::uint8_t
    {
        Balanced,
        Disabled,
        Silence,
        NoMid,  ///< Mid power is at or below the analysis gate.
        NoSide, ///< Side power is at or below the analysis gate.
        GainLimited,
        GuardUnreachable,
        AllExcluded
    };
    struct Report
    {
        Reason reason = Reason::Disabled;
        double targetSideShare = std::numeric_limits<double>::quiet_NaN();
        double inputSideShare = std::numeric_limits<double>::quiet_NaN();
        double outputSideShare = std::numeric_limits<double>::quiet_NaN();
        double guardCeilingRatio = 0;
        double maximumWindowExcessDb = -std::numeric_limits<double>::infinity();
        double minimumGainDb = 0, maximumGainDb = 0; ///< Planned bounds; rendered bounds after render.
        double outputSamplePeakDb = -std::numeric_limits<double>::infinity();
        double outputTruePeakDb = -std::numeric_limits<double>::infinity();
        std::size_t limitedPoints = 0, guardWindows = 0, violatedWindows = 0;
        std::int64_t analysisHopFrames = 0, guardWindowFrames = 0;
        int guardCalibrationPasses = 0;
        bool peaksMeasured = false, guardMeasured = false, guardTargetMet = false;
        OfflineGainRenderInfo renderInfo;
    };

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
            return state_ ? sizeof(State) + 2 * state_->count * sizeof(double) +
                                state_->exclusions.retainedBytes()
                          : 0;
        }
        [[nodiscard]] std::span<const OfflineRegion> exclusions() const noexcept
        {
            return state_ ? state_->exclusions.view() : std::span<const OfflineRegion>{};
        }
        /** @brief Planned scalar gain; rendering may add reported guard
         * attenuation. */
        [[nodiscard]] double gainAt(std::int64_t frame) const noexcept
        {
            return state_ ? control(*state_, state_->guard.get(), frame) : 1;
        }

      private:
        friend class OfflineStereoBalance;
        struct State
        {
            OfflineAudioSpec spec;
            OfflineFingerprint fingerprint, sideFingerprint;
            Report report;
            Options options;
            detail::OfflineExclusions exclusions;
            std::unique_ptr<double[]> level, guard;
            std::size_t count = 0, window = 1;
            std::int64_t hop = 1;
            double peak = 0, sidePeak = 0, gate = 0, feather = 1;
            bool changed = false;
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

    /** @brief Replans controls from an immutable complete-source mid/side
     * analysis. */
    [[nodiscard]] PlanResult
    makePlan(const typename OfflineEnergyAnalyzer<T>::MidSideAnalysis &analysis,
             const Options &options = {}, const OfflineJobOptions &jobOptions = {}) const
    {
        PlanResult result;
        try
        {
            validate(options);
            if (!analysis.isValid())
                detail::offlineFail(OfflineStatus::InvalidInput);
            detail::OfflineSession job(jobOptions);
            job.charge(sizeof(State));
            Plan plan;
            plan.state_ = std::make_unique<State>();
            auto &s = *plan.state_;
            s.spec = analysis.getSpec();
            s.fingerprint = analysis.fingerprint();
            s.sideFingerprint = analysis.sideFingerprint();
            s.peak = analysis.samplePeak();
            s.sidePeak = analysis.sidePeak();
            s.hop = analysis.binFrames();
            s.options = options;
            s.options.exclusions = {};
            s.exclusions.assign(job, options.exclusions, s.spec.frames);
            s.feather = std::max(1., s.spec.sampleRate * .005);
            const auto bins = analysis.bins();
            const auto n = bins.size();
            job.checkpoint(OfflinePhase::Plan, 0, static_cast<std::int64_t>(n));
            s.count = n + 1;
            s.level = job.allocate<double>(s.count);
            s.guard = job.allocate<double>(s.count);
            const double perSecond = s.spec.sampleRate / static_cast<double>(s.hop);
            s.window =
                std::max<std::size_t>(1, static_cast<std::size_t>(std::round(.04 * perSecond)));
            s.report.analysisHopFrames = s.hop;
            s.report.guardWindowFrames = s.hop * static_cast<std::int64_t>(s.window);
            s.report.inputSideShare = share(analysis.midRms(), analysis.sideRms());
            s.report.targetSideShare = options.targetSideShare.value_or(
                std::isfinite(s.report.inputSideShare)
                    ? std::clamp(s.report.inputSideShare, .001, .49)
                    : std::numeric_limits<double>::quiet_NaN());
            const double ratio = std::isfinite(s.report.targetSideShare)
                                     ? s.report.targetSideShare / (1 - s.report.targetSideShare)
                                     : 0;
            s.report.guardCeilingRatio = ratio * std::pow(10., options.guardMarginDb / 10);
            const double scale = s.peak > 0 ? s.peak : 1;
            const double mr = analysis.midRms() / scale, sr = analysis.sideRms() / scale;
            const double floor = std::min(1., 1e-7 / scale);
            s.gate = std::max(floor * floor, (mr * mr + sr * sr) * 1e-6);
            auto mid = job.allocate<double>(n), side = job.allocate<double>(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                if ((i & 1023) == 0)
                    job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i),
                                   static_cast<std::int64_t>(n));
                const double m = bins[i].midRms / scale, v = bins[i].sideRms / scale;
                mid[i] = m * m;
                side[i] = v * v;
            }
            const auto mean = [&](const double *power, std::size_t begin, std::size_t end)
            {
                double energy = 0, frames = 0;
                for (auto i = begin; i < end; ++i)
                {
                    energy += power[i] * static_cast<double>(bins[i].frames);
                    frames += static_cast<double>(bins[i].frames);
                }
                return energy / frames;
            };
            const auto radius =
                std::max<std::size_t>(1, static_cast<std::size_t>(std::round(.2 * perSecond)));
            for (std::size_t i = 0; i < n; ++i)
            {
                if ((i & 1023) == 0)
                    job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i),
                                   static_cast<std::int64_t>(n));
                const auto begin = i > radius ? i - radius : 0;
                const auto end = std::min(n, i + radius + 1);
                const double m = mean(mid.get(), begin, end), v = mean(side.get(), begin, end);
                if (options.levelingAmount > 0 && m > s.gate && v > s.gate)
                {
                    const double wanted = .5 * (std::log(ratio) + std::log(m) - std::log(v));
                    const double bounded = std::clamp(wanted, -options.maximumCutDb * logDb,
                                                      options.maximumBoostDb * logDb);
                    s.report.limitedPoints += wanted != bounded;
                    s.level[i] = options.levelingAmount * bounded;
                }
            }
            auto scratch = job.allocate<double>(s.count);
            const auto smoothRadius =
                std::max<std::size_t>(1, static_cast<std::size_t>(std::round(.5 * perSecond)));
            average(s.level.get(), scratch.get(), n, smoothRadius, job);
            average(scratch.get(), s.level.get(), n, smoothRadius, job);
            s.level[n] = s.level[n - 1];
            if (options.guard)
            {
                for (std::size_t i = 0; i < n; ++i)
                {
                    if ((i & 1023) == 0)
                        job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i),
                                       static_cast<std::int64_t>(n));
                    const auto end = std::min(n, i + s.window);
                    double m = 0, v = 0, frames = 0,
                           maximum = -std::numeric_limits<double>::infinity();
                    for (auto j = i; j < end; ++j)
                    {
                        const double count = static_cast<double>(bins[j].frames);
                        m += mid[j] * count;
                        v += side[j] * count;
                        frames += count;
                        maximum = std::max({maximum, s.level[j], s.level[j + 1]});
                    }
                    if (v > s.gate * frames)
                    {
                        const double safe = m > 0 ? .5 * (std::log(s.report.guardCeilingRatio) +
                                                          std::log(m) - std::log(v)) -
                                                        maximum
                                                  : -std::numeric_limits<double>::infinity();
                        scratch[i] = std::clamp(safe, -options.maximumGuardCutDb * logDb, 0.);
                    }
                    else
                        scratch[i] = 0;
                }
                for (std::size_t i = 0; i <= n; ++i)
                    for (auto j = i > s.window ? i - s.window : 0; j <= std::min(i, n - 1); ++j)
                        s.guard[i] = std::min(s.guard[i], scratch[j]);
                limitSlopes(s, s.guard.get());
            }
            s.report.reason = s.peak == 0              ? Reason::Silence
                              : mr * mr <= s.gate      ? Reason::NoMid
                              : sr * sr <= s.gate      ? Reason::NoSide
                              : s.report.limitedPoints ? Reason::GainLimited
                                                       : Reason::Balanced;
            if ((!options.guard && options.levelingAmount == 0) ||
                s.exclusions.covers(s.spec.frames))
            {
                std::fill_n(s.level.get(), s.count, 0.);
                std::fill_n(s.guard.get(), s.count, 0.);
                s.report.reason =
                    s.exclusions.covers(s.spec.frames) ? Reason::AllExcluded : Reason::Disabled;
            }
            // Unity belongs to the source control range when PCM is protected.
            // Otherwise initialize from a real control, not an unused 0 dB value.
            s.report.minimumGainDb = s.report.maximumGainDb =
                s.exclusions.view().empty() ? (s.level[0] + s.guard[0]) / logDb : 0;
            for (std::size_t i = 0; i < s.count; ++i)
            {
                const double db = (s.level[i] + s.guard[i]) / logDb;
                s.report.minimumGainDb = std::min(s.report.minimumGainDb, db);
                s.report.maximumGainDb = std::max(s.report.maximumGainDb, db);
                s.changed = s.changed || db != 0;
            }
            job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(n),
                           static_cast<std::int64_t>(n));
            result.status = s.changed ? OfflineStatus::Success : OfflineStatus::NoChange;
            result.memoryBytes = job.bytes();
            result.plan = std::move(plan);
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Analyzes the supplied post-generation stereo source and builds its
     * gain plan. */
    [[nodiscard]] PlanResult analyze(OfflineAudioSource<T> &source, const Options &options = {},
                                     const OfflineJobOptions &job = {}) const
    {
        PlanResult result;
        try
        {
            validate(options);
            auto energy = OfflineEnergyAnalyzer<T>().analyzeMidSide(source, job);
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

    /** @brief Verifies, optionally calibrates the guard, and publishes
     * transactionally. */
    [[nodiscard]] Result render(OfflineAudioSource<T> &source, const Plan &plan,
                                OfflineAudioSink<T> &sink,
                                const OfflineJobOptions &options = {}) const
    {
        Result result;
        try
        {
            if (!plan.state_)
                detail::offlineFail(OfflineStatus::InvalidInput);
            const auto &s = *plan.state_;
            result.report = s.report;
            if (source.getSpec() != s.spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            detail::OfflineSession job(options);
            auto guards = job.allocate<double>(s.count);
            auto corrections = job.allocate<double>(s.count);
            auto measured = job.allocate<Energy>(s.count - 1);
            std::copy_n(s.guard.get(), s.count, guards.get());
            SideSource side(source, s, job, options.blockFrames);
            StereoSink output(source, s, job, options.blockFrames, measured.get());
            bool publish = !s.options.guard;
            for (int pass = 0; pass < 17; ++pass)
            {
                output.destination = publish ? &sink : nullptr;
                output.guardRequired = publish && s.options.guard;
                auto innerJob = detail::offlineRemaining(options, job.bytes());
                auto rendered = detail::offlineRenderGain<Result>(
                    side, side.getSpec(), s.sideFingerprint, s.sidePeak, true, s.changed,
                    result.report, output, innerJob, [&](detail::OfflineSession &)
                    { return [&](std::int64_t frame) { return control(s, guards.get(), frame); }; },
                    {&s.exclusions, s.feather, false});
                result.memoryBytes =
                    std::max(result.memoryBytes, job.bytes() + rendered.memoryBytes);
                result.report.renderInfo = rendered.report.renderInfo;
                result.report.minimumGainDb =
                    gainToDecibels(rendered.report.renderInfo.minimumControlGain);
                result.report.maximumGainDb =
                    gainToDecibels(rendered.report.renderInfo.maximumControlGain);
                output.copyReport(result.report);
                if (!rendered.succeeded())
                {
                    result.status = output.failure != OfflineStatus::Success ? output.failure
                                    : side.failure != OfflineStatus::Success ? side.failure
                                                                             : rendered.status;
                    return result;
                }
                if (publish)
                {
                    result.status = rendered.status;
                    return result;
                }
                ++result.report.guardCalibrationPasses;
                if (result.report.guardTargetMet)
                {
                    publish = true;
                    continue;
                }
                if (pass == 15 || !refine(s, measured.get(), guards.get(), corrections.get()))
                    break;
            }
            if (result.report.reason != Reason::NoMid)
                result.report.reason = Reason::GuardUnreachable;
            result.status = OfflineStatus::TargetUnreachable;
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }

    /** @brief Owning convenience render; output changes only after successful
     * verification. */
    template <int MaxChannels>
    [[nodiscard]] Result run(const AudioBuffer<T, MaxChannels> &input,
                             AudioBuffer<T, MaxChannels> &output, double sampleRate,
                             const Options &options = {}, const OfflineJobOptions &job = {}) const
    {
        return detail::offlineRun(*this, input, output, sampleRate, options, job);
    }

  private:
    using State = typename Plan::State;
    static constexpr double logDb = .1151292546497022842;
    struct Energy
    {
        double mid = 0, side = 0;
        std::int64_t frames = 0;
    };
    static double share(double midRms, double sideRms) noexcept
    {
        const double scale = std::max(midRms, sideRms);
        if (scale == 0)
            return std::numeric_limits<double>::quiet_NaN();
        const double m = midRms / scale, s = sideRms / scale;
        return s * s / (m * m + s * s);
    }
    static void validate(const Options &options)
    {
        const auto range = [](double x, double low, double high)
        { return std::isfinite(x) && x >= low && x <= high; };
        if (!range(options.levelingAmount, 0, 1) || !range(options.guardMarginDb, 0, 12) ||
            !range(options.maximumBoostDb, 0, 24) || !range(options.maximumCutDb, 0, 60) ||
            !range(options.maximumGuardCutDb, 0, 120) ||
            (options.targetSideShare && !range(*options.targetSideShare, .001, .49)))
            detail::offlineFail(OfflineStatus::InvalidInput);
    }
    static void average(const double *input, double *output, std::size_t count, std::size_t radius,
                        const detail::OfflineSession &job)
    {
        std::size_t begin = 0, end = 0;
        double sum = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto first = i > radius ? i - radius : 0, last = std::min(count, i + radius + 1);
            while (end < last)
                sum = detail::advanceBoxAverage(input[end++], 0., sum, 1.);
            while (begin < first)
                sum = detail::advanceBoxAverage(0., input[begin++], sum, 1.);
            if ((i & 1023) == 0)
            {
                job.checkpoint(OfflinePhase::Plan, static_cast<std::int64_t>(i),
                               static_cast<std::int64_t>(count));
                sum = 0;
                for (auto j = begin; j < end; ++j)
                    sum += input[j];
            }
            output[i] = sum / static_cast<double>(end - begin);
        }
    }
    static void limitSlopes(const State &s, double *guard) noexcept
    {
        const double step = static_cast<double>(s.hop) / s.spec.sampleRate;
        for (std::size_t i = 1; i < s.count; ++i)
            guard[i] = std::min(guard[i], guard[i - 1] + 20 * logDb * step);
        for (auto i = s.count - 1; i > 0; --i)
            guard[i - 1] = std::min(guard[i - 1], guard[i] + 60 * logDb * step);
    }
    static double control(const State &s, const double *guard, std::int64_t frame) noexcept
    {
        if (frame < 0 || frame >= s.spec.frames)
            return 1;
        const auto i = static_cast<std::size_t>(frame / s.hop);
        const double t = static_cast<double>(frame % s.hop) / static_cast<double>(s.hop);
        const double logGain = std::lerp(s.level[i] + guard[i], s.level[i + 1] + guard[i + 1], t);
        return s.exclusions.apply(frame, std::exp(logGain), s.feather);
    }
    static Energy window(const State &s, const Energy *bins, std::size_t first) noexcept
    {
        Energy result;
        for (auto j = first; j < std::min(s.count - 1, first + s.window); ++j)
        {
            result.mid += bins[j].mid;
            result.side += bins[j].side;
            result.frames += bins[j].frames;
        }
        return result;
    }
    static double excess(const State &s, const Energy &e) noexcept
    {
        if (e.side <= s.gate * static_cast<double>(e.frames))
            return -std::numeric_limits<double>::infinity();
        return e.mid > 0
                   ? 10 / std::log(10.) *
                         (std::log(e.side) - std::log(e.mid) - std::log(s.report.guardCeilingRatio))
                   : std::numeric_limits<double>::infinity();
    }
    static bool refine(const State &s, const Energy *bins, double *guards,
                       double *corrections) noexcept
    {
        std::fill_n(corrections, s.count, 0.);
        for (std::size_t i = 0; i + 1 < s.count; ++i)
        {
            const double over = excess(s, window(s, bins, i));
            if (over <= 1e-9)
                continue;
            const double correction =
                std::isfinite(over) ? (over + .01) * logDb : s.options.maximumGuardCutDb * logDb;
            for (auto j = i; j <= std::min(s.count - 1, i + s.window); ++j)
                corrections[j] = std::max(corrections[j], correction);
        }
        bool changed = false;
        for (std::size_t j = 0; j < s.count; ++j)
        {
            const double next =
                std::max(-s.options.maximumGuardCutDb * logDb, guards[j] - corrections[j]);
            changed = changed || next < guards[j];
            guards[j] = next;
        }
        limitSlopes(s, guards);
        return changed;
    }

    class SideSource final : public OfflineAudioSource<double>
    {
      public:
        OfflineStatus failure = OfflineStatus::Success;
        SideSource(OfflineAudioSource<T> &source, const State &state, detail::OfflineSession &job,
                   int block)
            : source_(source), s_(state), scratch_(job, state.spec, std::min(4096, block))
        {
        }
        OfflineAudioSpec getSpec() const noexcept override
        {
            auto spec = source_.getSpec();
            spec.channels = 1;
            return spec;
        }
        bool read(std::int64_t first, AudioBufferView<double> output) override
        {
            try
            {
                for (int offset = 0; offset < output.getNumSamples();)
                {
                    const int count = std::min(scratch_.frames(), output.getNumSamples() - offset);
                    auto input = scratch_.view(count);
                    detail::offlineRead(source_, s_.spec, first + offset, input);
                    for (int i = 0; i < count; ++i)
                    {
                        const double l = input.getChannel(0)[i], r = input.getChannel(1)[i];
                        if (!std::isfinite(l) || !std::isfinite(r))
                            detail::offlineFail(OfflineStatus::NonFiniteInput);
                        output.getChannel(0)[offset + i] = std::midpoint(l, -r);
                    }
                    offset += count;
                }
                return true;
            }
            catch (...)
            {
                failure = detail::offlineExceptionStatus();
                return false;
            }
        }

      private:
        OfflineAudioSource<T> &source_;
        const State &s_;
        detail::OfflineBlock<T> scratch_;
    };

    class StereoSink final : public OfflineAudioSink<double>
    {
      public:
        OfflineAudioSink<T> *destination = nullptr;
        OfflineStatus failure = OfflineStatus::Success;
        bool guardRequired = false;
        StereoSink(OfflineAudioSource<T> &source, const State &state, detail::OfflineSession &job,
                   int block, Energy *bins)
            : source_(source), s_(state), scratch_(job, state.spec, std::min(4096, block)),
              bins_(bins)
        {
        }
        bool begin(const OfflineAudioSpec &) override
        {
            next_ = inBin_ = 0;
            fingerprint_ = {};
            mid_.reset();
            side_.reset();
            totalMid_.reset();
            totalSide_.reset();
            detector_.reset();
            peak_ = truePeak_ = 0;
            failure = OfflineStatus::Success;
            return destination ? destination->begin(s_.spec) : true;
        }
        bool write(std::int64_t first, AudioBufferView<const double> side) override
        {
            try
            {
                if (first != next_)
                    detail::offlineFail(OfflineStatus::InvalidInput);
                for (int offset = 0; offset < side.getNumSamples();)
                {
                    const int count = std::min(scratch_.frames(), side.getNumSamples() - offset);
                    auto output = scratch_.view(count);
                    detail::offlineRead(source_, s_.spec, first + offset, output);
                    const double scale = s_.peak > 0 ? s_.peak : 1;
                    for (int i = 0; i < count; ++i)
                    {
                        const double l = output.getChannel(0)[i], r = output.getChannel(1)[i];
                        if (!std::isfinite(l) || !std::isfinite(r))
                            detail::offlineFail(OfflineStatus::NonFiniteInput);
                        detail::offlineHash(fingerprint_, output.getChannel(0)[i]);
                        detail::offlineHash(fingerprint_, output.getChannel(1)[i]);
                        const double delta = side.getChannel(0)[offset + i] - std::midpoint(l, -r);
                        if (delta != 0)
                        {
                            const double a = l + delta, b = r - delta;
                            const double maximum =
                                static_cast<double>(std::numeric_limits<T>::max());
                            if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a) > maximum ||
                                std::abs(b) > maximum)
                                detail::offlineFail(OfflineStatus::NumericalFailure);
                            output.getChannel(0)[i] = static_cast<T>(a);
                            output.getChannel(1)[i] = static_cast<T>(b);
                        }
                        const double a = output.getChannel(0)[i], b = output.getChannel(1)[i];
                        const double m = std::midpoint(a, b) / scale,
                                     v = std::midpoint(a, -b) / scale;
                        mid_.add(m);
                        side_.add(v);
                        totalMid_.add(m);
                        totalSide_.add(v);
                        peak_ = std::max({peak_, std::abs(a), std::abs(b)});
                        truePeak_ = std::max({truePeak_, detector_.processSample(a / scale, 0),
                                              detector_.processSample(b / scale, 1)});
                        ++inBin_;
                        ++next_;
                        if (inBin_ == s_.hop || next_ == s_.spec.frames)
                        {
                            const double frames = static_cast<double>(inBin_);
                            const double mr = mid_.rms(frames), sr = side_.rms(frames);
                            bins_[static_cast<std::size_t>((next_ - 1) / s_.hop)] = {
                                mr * mr * frames, sr * sr * frames, inBin_};
                            inBin_ = 0;
                            mid_.reset();
                            side_.reset();
                        }
                    }
                    if (destination)
                    {
                        std::array<const T *, 2> channels{output.getChannel(0),
                                                          output.getChannel(1)};
                        detail::offlineCallSink(
                            [&]
                            {
                                return destination->write(first + offset,
                                                          {channels.data(), 2, count});
                            });
                    }
                    offset += count;
                }
                return true;
            }
            catch (...)
            {
                failure = detail::offlineExceptionStatus();
                return false;
            }
        }
        bool commit() override
        {
            if (next_ != s_.spec.frames || fingerprint_ != s_.fingerprint ||
                source_.getSpec() != s_.spec)
            {
                failure = OfflineStatus::SourceMismatch;
                return false;
            }
            if (guardRequired)
                for (std::size_t i = 0; i + 1 < s_.count; ++i)
                    if (excess(s_, window(s_, bins_, i)) > 1e-9)
                    {
                        failure = OfflineStatus::TargetUnreachable;
                        return false;
                    }
            return destination ? destination->commit() : true;
        }
        void abort() noexcept override
        {
            if (destination)
                destination->abort();
        }
        void copyReport(Report &report)
        {
            report.peaksMeasured = next_ == s_.spec.frames;
            if (!report.peaksMeasured)
                return;
            const double silence = -std::numeric_limits<double>::infinity();
            for (int c = 0; c < 2; ++c)
                truePeak_ = std::max(truePeak_, detector_.getTailPeak(c));
            report.outputSamplePeakDb = gainToDecibels(peak_, silence);
            report.outputTruePeakDb = truePeak_ > 0
                                          ? gainToDecibels(truePeak_, silence) +
                                                gainToDecibels(s_.peak > 0 ? s_.peak : 1, silence)
                                          : silence;
            report.outputSideShare = share(totalMid_.rms(static_cast<double>(next_)),
                                           totalSide_.rms(static_cast<double>(next_)));
            report.guardMeasured = s_.options.guard;
            report.guardWindows = report.violatedWindows = 0;
            report.maximumWindowExcessDb = silence;
            if (s_.options.guard)
                for (std::size_t i = 0; i + 1 < s_.count; ++i)
                {
                    const auto e = window(s_, bins_, i);
                    if (e.side <= s_.gate * static_cast<double>(e.frames))
                        continue;
                    ++report.guardWindows;
                    const double db = excess(s_, e);
                    report.maximumWindowExcessDb = std::max(report.maximumWindowExcessDb, db);
                    report.violatedWindows += db > 1e-9;
                }
            report.guardTargetMet = s_.options.guard && report.violatedWindows == 0;
        }

      private:
        OfflineAudioSource<T> &source_;
        const State &s_;
        detail::OfflineBlock<T> scratch_;
        Energy *bins_;
        OfflineFingerprint fingerprint_;
        std::int64_t next_ = 0, inBin_ = 0;
        detail::ScaledSumSquares mid_, side_, totalMid_, totalSide_;
        TruePeakDetector<double, 2> detector_;
        double peak_ = 0, truePeak_ = 0;
    };
};
} // namespace dspark
#endif // DSPARK_HAS_OFFLINE
