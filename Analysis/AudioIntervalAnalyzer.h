// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/** @file AudioIntervalAnalyzer.h
 * @brief Exact sample-interval selection with bounded streaming measurement.
 * Reuses LoudnessMeter, its Core true-peak detector and ScaledSumSquares.
 * No PCM is retained. All calls require one owner; publish Result copies to
 * other threads. No method allocates. See docs/interval-analysis.md for scope,
 * validity, complete-window coverage and continuous-observation semantics.
 * Dependencies: LoudnessMeter.h, Core/detail/ScaledSumSquares.h.
 */

#include "LoudnessMeter.h"
#include "../Core/detail/ScaledSumSquares.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dspark {

/** @brief One requested mono/stereo interval per instance; feed several instances
 * the same blocks to measure overlapping intervals in one source pass.
 * The region is always measured as an independent finite fragment. Optional
 * continuous loudness observations are separate, explicitly scoped results;
 * they never replace or masquerade as the region's integrated loudness.
 */
template <FloatType T>
class AudioIntervalAnalyzer final
{
public:
    /** @brief Absolute, nonnegative frame coordinates [begin,end), per channel. */
    struct Interval
    {
        std::int64_t begin = 0, end = 0;
        bool operator==(const Interval&) const noexcept = default;
    };
    enum class Context { Independent, IncludeContinuous };
    enum class Status { Pending, Complete, Truncated, InvalidArgument,
                        DiscontinuousInput, NonFiniteInput };
    enum class Validity { NotRequested, Valid, Silence, Empty, TooShort,
                          BelowGate, Incomplete, InvalidInput, NumericalError };

    /** @brief A numeric value with its validity and actual window/gate support.
     * Invalid/unavailable values are NaN. Silence has an explicit status; meter
     * floors remain -100 LUFS/dBTP, not a claim of physical negative infinity.
     * IIR state may depend on earlier samples within Loudness::input.
     */
    struct Measurement
    {
        double value = std::numeric_limits<double>::quiet_NaN();
        Validity validity = Validity::NotRequested;
        Interval support;
        [[nodiscard]] bool hasValue() const noexcept
        { return validity == Validity::Valid || validity == Validity::Silence; }
    };

    /** @brief Readouts at the last complete 100 ms hop within input.
     * committedEnd is a source-frame position, not the caller's block boundary.
     * pendingFrames are included in sample statistics but not loudness windows.
     * integrated/range support bounds the complete windows supplied to gating;
     * rejected windows inside that span do not contribute to the final value.
     */
    struct Loudness
    {
        Interval input;
        std::int64_t committedEnd = 0;
        int pendingFrames = 0;
        int hopFrames = 0;
        std::uint64_t integratedBlocks = 0, rangeBlocks = 0;
        Measurement momentary, shortTerm, integrated, range;
    };

    /** @brief Immutable-by-convention result, safe to copy to another thread. */
    struct Result
    {
        Status status = Status::Pending;
        Context context = Context::Independent;
        Interval requested, measured;
        double sampleRate = 0;
        int channels = 0;
        Measurement rms, samplePeak, truePeakDbtp;
        Loudness region, continuous;
        [[nodiscard]] bool complete() const noexcept { return status == Status::Complete; }
    };

    AudioIntervalAnalyzer() noexcept { result_.status = Status::InvalidArgument; }

    /** @brief Starts a fresh measurement, resetting all filters, windows and gates.
     * streamOrigin is the first frame that processBlock must receive. It must
     * precede or equal requested.begin. Independent mode can start directly at
     * requested.begin; continuous mode observes exactly [streamOrigin,end).
     * Only mono/stereo are accepted. False leaves an explicit invalid result.
     */
    [[nodiscard]] bool prepare(const AudioSpec& spec, Interval requested,
                               Context context = Context::Independent,
                               std::int64_t streamOrigin = 0) noexcept
    {
        result_ = {};
        result_.requested = requested;
        result_.measured = { requested.begin, requested.begin };
        result_.context = context;
        result_.sampleRate = spec.sampleRate;
        result_.channels = spec.numChannels;
        energy_.reset();
        contextSilent_ = true;
        sealed_ = false;
        nextFrame_ = origin_ = streamOrigin;
        maxBlockSize_ = spec.maxBlockSize;
        if (!spec.isValid() || spec.numChannels > 2 || streamOrigin < 0 ||
            requested.begin < streamOrigin || requested.end < requested.begin ||
            (context != Context::Independent && context != Context::IncludeContinuous))
        {
            result_.status = Status::InvalidArgument;
            return false;
        }
        regionMeter_.prepare(spec);
        if (context == Context::IncludeContinuous) continuousMeter_.prepare(spec);
        if (requested.begin == requested.end) result_.status = Status::Complete;
        return true;
    }

    /** @brief Consumes a contiguous block at firstFrame; clips both region edges.
     * No samples at or after requested.end enter any measurement. Blocks after
     * completion are ignored. Gaps/overlaps, layout/size errors and non-finite
     * samples in a measured scope fail explicitly. Ignored audio is not scanned.
     */
    [[nodiscard]] bool processBlock(AudioBufferView<const T> block,
                                    std::int64_t firstFrame) noexcept
    {
        if (result_.status == Status::Complete) return true;
        if (sealed_ || result_.status != Status::Pending) return false;
        const int count = block.getNumSamples();
        if (count < 0 || count > maxBlockSize_ || block.getNumChannels() != result_.channels ||
            firstFrame < 0 || firstFrame > std::numeric_limits<std::int64_t>::max() - count)
            return fail(Status::InvalidArgument);
        if (firstFrame != nextFrame_) return fail(Status::DiscontinuousInput);
        if (count == 0) return true;
        for (int ch = 0; ch < result_.channels; ++ch)
            if (block.getChannel(ch) == nullptr) return fail(Status::InvalidArgument);
        const auto end = std::min(firstFrame + count, result_.requested.end);
        const auto begin = std::max(firstFrame, result_.requested.begin);
        const bool observe = result_.context == Context::IncludeContinuous;
        const auto usedBegin = observe ? firstFrame : begin;
        // Validate the relevant portion before mutating a meter on this block.
        for (auto frame = usedBegin; frame < end; ++frame)
            for (int ch = 0; ch < result_.channels; ++ch)
                if (!std::isfinite(block.getChannel(ch)[frame - firstFrame]))
                    return fail(Status::NonFiniteInput);
        if (observe)
        {
            feed(continuousMeter_, block, 0, static_cast<int>(end - firstFrame));
            for (auto frame = firstFrame; frame < end; ++frame)
                for (int ch = 0; ch < result_.channels; ++ch)
                    contextSilent_ &= block.getChannel(ch)[frame - firstFrame] == T(0);
        }
        if (begin < end)
        {
            feed(regionMeter_, block, static_cast<int>(begin - firstFrame),
                 static_cast<int>(end - begin));
            for (auto frame = begin; frame < end; ++frame)
                for (int ch = 0; ch < result_.channels; ++ch)
                    energy_.add(static_cast<double>(block.getChannel(ch)[frame - firstFrame]));
            result_.measured.end = end;
        }
        nextFrame_ = firstFrame + count;
        if (end == result_.requested.end) result_.status = Status::Complete;
        return true;
    }

    /** @brief Seals the result, including the finite region's true-peak tail.
     * Call at EOF or after completing the interval. Idempotent. Early EOF yields
     * Truncated and unavailable measurements, with the actual measured interval.
     * The tail is evaluated on copied Core state, never as extra loudness audio.
     */
    [[nodiscard]] Result finish() noexcept
    {
        if (sealed_) return result_;
        sealed_ = true;
        if (result_.status == Status::Pending) result_.status = Status::Truncated;
        if (!result_.complete())
        {
            const auto validity = result_.status == Status::Truncated
                ? Validity::Incomplete : Validity::InvalidInput;
            result_.rms.validity = result_.samplePeak.validity = result_.truePeakDbtp.validity = validity;
            markLoudness(result_.region, validity);
            if (result_.context == Context::IncludeContinuous) markLoudness(result_.continuous, validity);
            return result_;
        }
        const auto frames = result_.measured.end - result_.measured.begin;
        const auto validity = frames == 0 ? Validity::Empty :
            energy_.peak() == 0 ? Validity::Silence : Validity::Valid;
        result_.rms = { frames ? energy_.rms(static_cast<double>(frames) * result_.channels) :
            std::numeric_limits<double>::quiet_NaN(), validity, result_.measured };
        result_.samplePeak = { frames ? energy_.peak() :
            std::numeric_limits<double>::quiet_NaN(), validity, result_.measured };
        regionMeter_.finalizeTruePeak();
        result_.truePeakDbtp = { frames ? static_cast<double>(regionMeter_.getTruePeakDb()) :
            std::numeric_limits<double>::quiet_NaN(), validity, result_.measured };
        if (!regionMeter_.isMeasurementValid())
            result_.truePeakDbtp = { std::numeric_limits<double>::quiet_NaN(),
                Validity::NumericalError, result_.measured };
        result_.region = loudness(regionMeter_, result_.measured, energy_.peak() == 0);
        if (result_.context == Context::IncludeContinuous)
            result_.continuous = loudness(continuousMeter_,
                { origin_, frames ? result_.measured.end : origin_ }, contextSilent_);
        return result_;
    }

private:
    bool fail(Status status) noexcept { result_.status = status; return false; }
    void feed(LoudnessMeter<T>& meter, AudioBufferView<const T> block, int offset, int count) noexcept
    {
        if (result_.channels == 1) meter.process(block.getChannel(0) + offset, count);
        else meter.process(block.getChannel(0) + offset, block.getChannel(1) + offset, count);
    }
    static void markLoudness(Loudness& reading, Validity validity) noexcept
    {
        reading.momentary.validity = reading.shortTerm.validity =
            reading.integrated.validity = reading.range.validity = validity;
    }
    static Loudness loudness(const LoudnessMeter<T>& meter, Interval input, bool silent) noexcept
    {
        Loudness out;
        out.input = input;
        const auto info = meter.getMeasurementInfo();
        out.committedEnd = input.begin + info.committedFrames;
        out.pendingFrames = info.pendingFrames;
        out.hopFrames = info.hopFrames;
        out.integratedBlocks = info.integratedBlocks;
        out.rangeBlocks = info.rangeBlocks;
        if (input.begin == input.end) { markLoudness(out, Validity::Empty); return out; }
        if (!meter.isMeasurementValid()) { markLoudness(out, Validity::NumericalError); return out; }
        const auto metric = [&](double value, int hops, bool gated, std::uint64_t blocks,
                                bool cumulative) -> Measurement
        {
            const auto length = static_cast<std::int64_t>(hops) * info.hopFrames;
            if (info.committedFrames < length)
                return { std::numeric_limits<double>::quiet_NaN(), Validity::TooShort, input };
            const Interval support { cumulative ? input.begin : out.committedEnd - length,
                                     out.committedEnd };
            if (gated && blocks == 0 && !silent)
                return { std::numeric_limits<double>::quiet_NaN(), Validity::BelowGate, support };
            return { value, silent ? Validity::Silence : Validity::Valid, support };
        };
        out.momentary = metric(meter.getMomentaryLUFS(), 4, false, 0, false);
        out.shortTerm = metric(meter.getShortTermLUFS(), 30, false, 0, false);
        out.integrated = metric(meter.getIntegratedLUFS(), 4, true, info.integratedBlocks, true);
        out.range = metric(meter.getLoudnessRange(), 30, true, info.rangeBlocks, true);
        return out;
    }

    LoudnessMeter<T> regionMeter_, continuousMeter_;
    detail::ScaledSumSquares energy_;
    Result result_;
    std::int64_t origin_ = 0, nextFrame_ = 0;
    int maxBlockSize_ = 0;
    bool contextSilent_ = true, sealed_ = false;
};
} // namespace dspark
