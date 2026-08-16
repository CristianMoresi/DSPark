// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LoopFinder.h
 * @brief Bounded offline search for perceptually continuous audio loops.
 *
 * find() scores candidate loop endpoints over explicit inclusive ranges
 * with a waveform/derivative/spectral seam cost and pre-counted work
 * bounds; renderLoop() writes one seamless cycle using the normalized
 * equal-power overlap. Failure is always explicit: a non-success result
 * carries the neutral payload and renders nothing.
 *
 * Threading: offline, owner-managed. One caller owns an instance for the
 * duration of a find(); renderLoop() is a stateless static call. Neither
 * belongs on the audio thread and there are no cross-thread setters.
 *
 * Dependencies: AudioBuffer.h, DspMath.h, FFT.h, Crossfade.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Effects/Crossfade.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class LoopFinder
 * @brief Finds and renders bounded, deterministic loop seams.
 * @tparam T Sample type (float or double).
 *
 * find() is an offline operation: one caller owns an instance for the complete
 * search and bounded scratch allocation is permitted. renderLoop() performs no
 * allocation and is transactional; invalid or non-finite input leaves the
 * destination unchanged. Source samples supplied to either operation must be
 * finite.
 */
template <FloatType T>
class LoopFinder final
{
public:
    enum class Status : std::uint8_t {
        Success,
        EmptyInput,
        InvalidArgument,
        InputTooShort,
        InputTooLong,
        NonFiniteInput,
        InsufficientActivity,
        CandidateLimitExceeded,
        NoAcceptableLoop,
        InvalidOutput,
        NumericalFailure
    };

    struct Range {
        std::int64_t first = 0;
        std::int64_t last = -1;
    };

    struct Settings {
        int comparisonLength = 1024;
        int crossfadeLength = 256;
        int coarseStride = 64;
        int refinementRadius = 63;
        std::uint32_t maxCoarsePairs = 131072;
        std::uint32_t maxRefinementPairs = 262144;
        T maximumCost = T(0.20);
        T minimumActivityRms = T(1.0e-5);
    };

    struct Result {
        Status status = Status::InvalidArgument;
        std::int64_t start = 0;
        std::int64_t end = 0;
        int crossfadeLength = 0;
        T seamCost = T(1);

        [[nodiscard]] explicit operator bool() const noexcept
        {
            return status == Status::Success;
        }

        [[nodiscard]] std::int64_t renderedLength() const noexcept
        {
            if (status != Status::Success || start < 0 || end < start
                || crossfadeLength < 2)
                return 0;

            // Safe because 0 <= start <= end <= INT64_MAX.
            const std::int64_t sourceLength = end - start;
            const std::int64_t fade = static_cast<std::int64_t>(crossfadeLength);
            if (fade > sourceLength / 2)
                return 0;
            return sourceLength - fade;
        }
    };

    /**
     * @brief Searches explicit inclusive endpoint ranges for the lowest-cost
     *        supported loop seam.
     *
     * The complete source is checked for finiteness. Coarse and refinement
     * coordinate counts are conservatively bounded before any candidate is
     * scored; scratch is independent of the number of candidates.
     */
    [[nodiscard]] Result find(AudioBufferView<const T> audio,
                              Range startRange,
                              Range endRange,
                              std::int64_t minLoopLength,
                              std::int64_t maxLoopLength,
                              const Settings& settings = {}) const
    {
        const int channels = audio.getNumChannels();
        const int sampleCount = audio.getNumSamples();
        if (channels == 0 || sampleCount == 0)
            return failure(Status::EmptyInput);

        if (channels < 1 || channels > kMaxChannels)
            return failure(Status::InvalidArgument);
        for (int ch = 0; ch < channels; ++ch)
            if (audio.getChannel(ch) == nullptr)
                return failure(Status::InvalidArgument);

        // Every comparison below is parenthesised on purpose. Left bare in a
        // template, MSVC reads `endRange.last < ... || endRange.last >` as a
        // template argument list on a member name it has not resolved yet and
        // rejects the whole predicate.
        if (!validSettings(settings)
            || (startRange.first < 0) || (startRange.last < startRange.first)
            || (startRange.last >= static_cast<std::int64_t>(sampleCount))
            || (endRange.first < 0) || (endRange.last < endRange.first)
            || (endRange.last > static_cast<std::int64_t>(sampleCount))
            || (minLoopLength < 1) || (maxLoopLength < minLoopLength)
            || (maxLoopLength > static_cast<std::int64_t>(sampleCount)))
            return failure(Status::InvalidArgument);

        const std::int64_t window = settings.comparisonLength;
        const std::int64_t fade = settings.crossfadeLength;
        if (window == std::numeric_limits<std::int64_t>::max()
            || fade > std::numeric_limits<std::int64_t>::max() / 2)
            return failure(Status::InvalidArgument);
        const std::int64_t minimumPeriod =
            std::max(window + 1, fade * 2);

        if (sampleCount > kMaxInputSamples)
            return failure(Status::InputTooLong);

        long double sourceScale = 0.0L;
        for (int ch = 0; ch < channels; ++ch)
        {
            const T* const samples = audio.getChannel(ch);
            for (int n = 0; n < sampleCount; ++n)
            {
                if (!std::isfinite(samples[n]))
                    return failure(Status::NonFiniteInput);
                sourceScale = std::max(sourceScale,
                    std::abs(static_cast<long double>(samples[n])));
            }
        }

        if (sampleCount < minimumPeriod)
            return failure(Status::InputTooShort);

        bool sourceHasActivity = false;
        long double sourceVarianceSum = 0.0L;
        int sourceActiveChannels = 0;
        if (sourceScale > 0.0L)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                const T* const samples = audio.getChannel(ch);
                long double mean = 0.0L;
                for (int n = 0; n < sampleCount; ++n)
                    mean += static_cast<long double>(samples[n]) / sourceScale;
                mean /= static_cast<long double>(sampleCount);

                long double variance = 0.0L;
                for (int n = 0; n < sampleCount; ++n)
                {
                    const long double centered =
                        static_cast<long double>(samples[n]) / sourceScale - mean;
                    variance += centered * centered;
                }
                variance /= static_cast<long double>(sampleCount);
                variance = std::clamp(variance, 0.0L, 1.0L);
                const long double rms = sourceScale * std::sqrt(variance);
                if (!std::isfinite(variance) || !std::isfinite(rms))
                    return failure(Status::NumericalFailure);
                if (rms >= static_cast<long double>(settings.minimumActivityRms))
                {
                    sourceHasActivity = true;
                    sourceVarianceSum += variance;
                    ++sourceActiveChannels;
                }
            }
        }
        if (!sourceHasActivity)
            return failure(Status::InsufficientActivity);

        const long double sourceRms = sourceScale * std::sqrt(std::min(
            1.0L, sourceVarianceSum
                / static_cast<long double>(sourceActiveChannels)));
        if (!std::isfinite(sourceRms))
            return failure(Status::NumericalFailure);

        const std::uint64_t startCount = latticeCount(startRange, settings.coarseStride);
        const std::uint64_t endCount = latticeCount(endRange, settings.coarseStride);
        std::uint64_t coarseUpper = 0;
        if (!checkedMultiply(startCount, endCount, coarseUpper))
            return failure(Status::CandidateLimitExceeded);

        const std::uint64_t seedUpper = std::min<std::uint64_t>(8u, coarseUpper);
        const std::uint64_t radiusSpan =
            static_cast<std::uint64_t>(settings.refinementRadius) * 2u + 1u;
        const std::uint64_t startSpan = std::min(
            static_cast<std::uint64_t>(startRange.last - startRange.first) + 1u,
            radiusSpan);
        const std::uint64_t endSpan = std::min(
            static_cast<std::uint64_t>(endRange.last - endRange.first) + 1u,
            radiusSpan);
        std::uint64_t refinementUpper = 0;
        std::uint64_t upperRectangle = 0;
        if (!checkedMultiply(startSpan, endSpan, upperRectangle)
            || !checkedMultiply(seedUpper, upperRectangle, refinementUpper)
            || coarseUpper > settings.maxCoarsePairs
            || refinementUpper > settings.maxRefinementPairs)
            return failure(Status::CandidateLimitExceeded);

        ScoringScratch scratch(settings.comparisonLength);
        std::array<Candidate, 8> seeds {};
        int seedCount = 0;
        Candidate best {};
        bool hasBest = false;

        auto geometricallyValid = [&](std::int64_t start,
                                      std::int64_t end) noexcept {
            if (start < 0 || end <= start
                || end > static_cast<std::int64_t>(sampleCount))
                return false;
            const std::int64_t period = end - start;
            return period >= minimumPeriod
                && period >= minLoopLength && period <= maxLoopLength;
        };

        auto consider = [&](std::int64_t start, std::int64_t end,
                            bool keepSeed) {
            if (!geometricallyValid(start, end)) return ScoreState::Inactive;
            Candidate candidate { start, end, T(1) };
            const ScoreState state = scoreCandidate(audio, start, end,
                                                     settings, scratch,
                                                     candidate.cost);
            if (state != ScoreState::Scored) return state;
            if (!hasBest || better(candidate, best))
            {
                best = candidate;
                hasBest = true;
            }
            if (keepSeed) insertSeed(seeds, seedCount, candidate);
            return ScoreState::Scored;
        };

        for (std::uint64_t si = 0; si < startCount; ++si)
        {
            const std::int64_t start = startRange.first
                + static_cast<std::int64_t>(si)
                    * static_cast<std::int64_t>(settings.coarseStride);
            for (std::uint64_t ei = 0; ei < endCount; ++ei)
            {
                const std::int64_t end = endRange.first
                    + static_cast<std::int64_t>(ei)
                        * static_cast<std::int64_t>(settings.coarseStride);
                if (consider(start, end, true) == ScoreState::NumericalFailure)
                    return failure(Status::NumericalFailure);
            }
        }

        std::uint64_t exactRefinementAttempts = 0;
        for (int seed = 0; seed < seedCount; ++seed)
        {
            const std::int64_t lowStart = std::max(startRange.first,
                seeds[static_cast<std::size_t>(seed)].start
                    - settings.refinementRadius);
            const std::int64_t highStart = std::min(startRange.last,
                seeds[static_cast<std::size_t>(seed)].start
                    + settings.refinementRadius);
            const std::int64_t lowEnd = std::max(endRange.first,
                seeds[static_cast<std::size_t>(seed)].end
                    - settings.refinementRadius);
            const std::int64_t highEnd = std::min(endRange.last,
                seeds[static_cast<std::size_t>(seed)].end
                    + settings.refinementRadius);
            const std::uint64_t widthStart =
                static_cast<std::uint64_t>(highStart - lowStart) + 1u;
            const std::uint64_t widthEnd =
                static_cast<std::uint64_t>(highEnd - lowEnd) + 1u;
            std::uint64_t rectangle = 0;
            if (!checkedMultiply(widthStart, widthEnd, rectangle)
                || exactRefinementAttempts > refinementUpper
                || rectangle > refinementUpper - exactRefinementAttempts)
                return failure(Status::NumericalFailure);
            exactRefinementAttempts += rectangle;
        }
        if (exactRefinementAttempts > settings.maxRefinementPairs)
            return failure(Status::CandidateLimitExceeded);

        for (int seed = 0; seed < seedCount; ++seed)
        {
            const std::int64_t lowStart = std::max(startRange.first,
                seeds[static_cast<std::size_t>(seed)].start
                    - settings.refinementRadius);
            const std::int64_t highStart = std::min(startRange.last,
                seeds[static_cast<std::size_t>(seed)].start
                    + settings.refinementRadius);
            const std::int64_t lowEnd = std::max(endRange.first,
                seeds[static_cast<std::size_t>(seed)].end
                    - settings.refinementRadius);
            const std::int64_t highEnd = std::min(endRange.last,
                seeds[static_cast<std::size_t>(seed)].end
                    + settings.refinementRadius);
            for (std::int64_t start = lowStart; start <= highStart; ++start)
            {
                for (std::int64_t end = lowEnd; end <= highEnd; ++end)
                {
                    if (consider(start, end, false)
                        == ScoreState::NumericalFailure)
                        return failure(Status::NumericalFailure);
                }
            }
        }

        if (!hasBest || best.cost > settings.maximumCost)
            return failure(Status::NoAcceptableLoop);
        return { Status::Success, best.start, best.end,
                 settings.crossfadeLength, best.cost };
    }

    /**
     * @brief Renders one normalized equal-power loop cycle without allocation.
     * @return Success, InvalidArgument, InvalidOutput or NumericalFailure.
     */
    [[nodiscard]] static Status renderLoop(AudioBufferView<const T> audio,
                                           const Result& loop,
                                           AudioBufferView<T> output) noexcept
    {
        const int channels = audio.getNumChannels();
        const int sampleCount = audio.getNumSamples();
        // Parenthesised for the same reason as the range check in find():
        // bare, `loop.end < ... || loop.end >` reads as a template argument
        // list to MSVC.
        if (loop.status != Status::Success || (channels < 1)
            || (channels > kMaxChannels) || (sampleCount <= 0)
            || (loop.start < 0) || (loop.end <= loop.start)
            || (loop.end > static_cast<std::int64_t>(sampleCount))
            || (loop.crossfadeLength < 2)
            || !std::isfinite(loop.seamCost) || (loop.seamCost < T(0))
            || (loop.seamCost > T(1)))
            return Status::InvalidArgument;
        for (int ch = 0; ch < channels; ++ch)
            if (audio.getChannel(ch) == nullptr)
                return Status::InvalidArgument;

        const std::int64_t period = loop.end - loop.start;
        const std::int64_t fade = loop.crossfadeLength;
        if (fade > period / 2)
            return Status::InvalidArgument;
        const std::int64_t rendered = period - fade;
        if (rendered <= 0 || rendered > std::numeric_limits<int>::max())
            return Status::InvalidArgument;

        if (output.getNumChannels() != channels
            || output.getNumSamples() != static_cast<int>(rendered))
            return Status::InvalidOutput;
        for (int ch = 0; ch < channels; ++ch)
            if (output.getChannel(ch) == nullptr)
                return Status::InvalidOutput;

        const std::size_t sourceBytes = static_cast<std::size_t>(sampleCount)
            * sizeof(T);
        const std::size_t outputBytes = static_cast<std::size_t>(rendered)
            * sizeof(T);
        for (int outChannel = 0; outChannel < channels; ++outChannel)
        {
            const T* const destination = output.getChannel(outChannel);
            for (int sourceChannel = 0; sourceChannel < channels; ++sourceChannel)
                if (rangesOverlap(audio.getChannel(sourceChannel), sourceBytes,
                                  destination, outputBytes))
                    return Status::InvalidOutput;
            for (int other = 0; other < outChannel; ++other)
                if (rangesOverlap(output.getChannel(other), outputBytes,
                                  destination, outputBytes))
                    return Status::InvalidOutput;
        }

        for (int ch = 0; ch < channels; ++ch)
        {
            const T* const source = audio.getChannel(ch);
            for (int n = 0; n < sampleCount; ++n)
                if (!std::isfinite(source[n]))
                    return Status::NumericalFailure;
            if (!dryRenderChannel(source, loop))
                return Status::NumericalFailure;
        }

        for (int ch = 0; ch < channels; ++ch)
        {
            const T* const source = audio.getChannel(ch);
            T* const destination = output.getChannel(ch);
            std::copy_n(source + static_cast<std::ptrdiff_t>(loop.start),
                        static_cast<std::size_t>(rendered), destination);
            Crossfade<T> crossfade;
            crossfade.setCurve(Crossfade<T>::Curve::EqualPower);
            for (int i = 0; i < loop.crossfadeLength; ++i)
                destination[i] = renderOverlapSample(source, loop, i, crossfade);
        }
        return Status::Success;
    }

private:
    static constexpr int kMaxChannels = 16;
    static constexpr int kMaxInputSamples = 1 << 26;

    enum class ScoreState : std::uint8_t { Inactive, Scored, NumericalFailure };

    struct Candidate {
        std::int64_t start = 0;
        std::int64_t end = 0;
        T cost = T(1);
    };

    struct ScoringScratch {
        explicit ScoringScratch(int windowSize)
            : fft(static_cast<std::size_t>(windowSize)),
              window(static_cast<std::size_t>(windowSize)),
              timeA(static_cast<std::size_t>(windowSize)),
              timeB(static_cast<std::size_t>(windowSize)),
              frequencyA(static_cast<std::size_t>(windowSize + 2)),
              frequencyB(static_cast<std::size_t>(windowSize + 2))
        {
            for (int i = 0; i < windowSize; ++i)
            {
                window[static_cast<std::size_t>(i)] = T(0.5)
                    - T(0.5) * std::cos(T(2) * std::numbers::pi_v<T>
                        * static_cast<T>(i) / static_cast<T>(windowSize));
            }
        }

        FFTReal<T> fft;
        std::vector<T> window;
        std::vector<T> timeA;
        std::vector<T> timeB;
        std::vector<T> frequencyA;
        std::vector<T> frequencyB;
        std::array<std::pair<long double, long double>, kMaxChannels> channelCosts {};
    };

    [[nodiscard]] static Result failure(Status status) noexcept
    {
        return { status, 0, 0, 0, T(1) };
    }

    [[nodiscard]] static bool validSettings(const Settings& settings) noexcept
    {
        const int window = settings.comparisonLength;
        return window >= 64 && window <= 4096
            && (window & (window - 1)) == 0
            && settings.crossfadeLength >= 2
            && settings.crossfadeLength <= window
            && settings.coarseStride >= 1 && settings.coarseStride <= 4096
            && settings.refinementRadius >= 0
            && settings.refinementRadius <= 4096
            && settings.maxCoarsePairs >= 1
            && settings.maxCoarsePairs <= 1048576u
            && settings.maxRefinementPairs >= 1
            && settings.maxRefinementPairs <= 1048576u
            && std::isfinite(settings.maximumCost)
            && settings.maximumCost >= T(0) && settings.maximumCost <= T(1)
            && std::isfinite(settings.minimumActivityRms)
            && settings.minimumActivityRms > T(0)
            && settings.minimumActivityRms <= T(1);
    }

    [[nodiscard]] static std::uint64_t latticeCount(Range range,
                                                     int stride) noexcept
    {
        return static_cast<std::uint64_t>(range.last - range.first)
            / static_cast<std::uint64_t>(stride) + 1u;
    }

    [[nodiscard]] static bool checkedMultiply(std::uint64_t a,
                                               std::uint64_t b,
                                               std::uint64_t& result) noexcept
    {
        if (a != 0u && b > std::numeric_limits<std::uint64_t>::max() / a)
            return false;
        result = a * b;
        return true;
    }

    [[nodiscard]] static bool better(const Candidate& lhs,
                                     const Candidate& rhs) noexcept
    {
        return lhs.cost < rhs.cost
            || (lhs.cost == rhs.cost
                && (lhs.start < rhs.start
                    || (lhs.start == rhs.start && lhs.end < rhs.end)));
    }

    static void insertSeed(std::array<Candidate, 8>& seeds,
                           int& count,
                           Candidate candidate) noexcept
    {
        int position = 0;
        while (position < count
               && !better(candidate, seeds[static_cast<std::size_t>(position)]))
            ++position;
        if (position >= 8) return;
        const int newCount = std::min(8, count + 1);
        for (int i = newCount - 1; i > position; --i)
            seeds[static_cast<std::size_t>(i)] =
                seeds[static_cast<std::size_t>(i - 1)];
        seeds[static_cast<std::size_t>(position)] = candidate;
        count = newCount;
    }

    [[nodiscard]] static ScoreState scoreCandidate(
        AudioBufferView<const T> audio,
        std::int64_t start,
        std::int64_t end,
        const Settings& settings,
        ScoringScratch& scratch,
        T& outputCost) noexcept
    {
        const int channels = audio.getNumChannels();
        const int window = settings.comparisonLength;
        long double scale = 0.0L;
        for (int ch = 0; ch < channels; ++ch)
        {
            const T* const source = audio.getChannel(ch);
            for (int i = 0; i < window; ++i)
            {
                scale = std::max(scale, std::abs(static_cast<long double>(
                    source[end - window + i])));
                scale = std::max(scale, std::abs(static_cast<long double>(
                    source[start + i])));
            }
        }
        if (scale == 0.0L) return ScoreState::Inactive;

        std::array<long double, kMaxChannels> meanA {};
        std::array<long double, kMaxChannels> meanB {};
        std::array<long double, kMaxChannels> varianceA {};
        std::array<long double, kMaxChannels> varianceB {};
        std::array<bool, kMaxChannels> included {};
        int includedCount = 0;
        long double aggregateVarianceA = 0.0L;
        long double aggregateVarianceB = 0.0L;

        for (int ch = 0; ch < channels; ++ch)
        {
            const T* const source = audio.getChannel(ch);
            for (int i = 0; i < window; ++i)
            {
                meanA[static_cast<std::size_t>(ch)] +=
                    static_cast<long double>(source[end - window + i]) / scale;
                meanB[static_cast<std::size_t>(ch)] +=
                    static_cast<long double>(source[start + i]) / scale;
            }
            meanA[static_cast<std::size_t>(ch)] /= window;
            meanB[static_cast<std::size_t>(ch)] /= window;
            for (int i = 0; i < window; ++i)
            {
                const long double a =
                    static_cast<long double>(source[end - window + i]) / scale
                    - meanA[static_cast<std::size_t>(ch)];
                const long double b =
                    static_cast<long double>(source[start + i]) / scale
                    - meanB[static_cast<std::size_t>(ch)];
                varianceA[static_cast<std::size_t>(ch)] += a * a;
                varianceB[static_cast<std::size_t>(ch)] += b * b;
            }
            varianceA[static_cast<std::size_t>(ch)] = std::clamp(
                varianceA[static_cast<std::size_t>(ch)] / window, 0.0L, 1.0L);
            varianceB[static_cast<std::size_t>(ch)] = std::clamp(
                varianceB[static_cast<std::size_t>(ch)] / window, 0.0L, 1.0L);
            const long double rmsA = scale * std::sqrt(
                varianceA[static_cast<std::size_t>(ch)]);
            const long double rmsB = scale * std::sqrt(
                varianceB[static_cast<std::size_t>(ch)]);
            if (!std::isfinite(rmsA) || !std::isfinite(rmsB))
                return ScoreState::NumericalFailure;
            if (std::max(rmsA, rmsB)
                >= static_cast<long double>(settings.minimumActivityRms))
            {
                included[static_cast<std::size_t>(ch)] = true;
                ++includedCount;
                aggregateVarianceA += varianceA[static_cast<std::size_t>(ch)];
                aggregateVarianceB += varianceB[static_cast<std::size_t>(ch)];
            }
        }
        if (includedCount == 0) return ScoreState::Inactive;
        const long double aggregateRmsA = scale * std::sqrt(std::min(1.0L,
            aggregateVarianceA / static_cast<long double>(includedCount)));
        const long double aggregateRmsB = scale * std::sqrt(std::min(1.0L,
            aggregateVarianceB / static_cast<long double>(includedCount)));
        if (!std::isfinite(aggregateRmsA) || !std::isfinite(aggregateRmsB))
            return ScoreState::NumericalFailure;
        if (aggregateRmsA < static_cast<long double>(settings.minimumActivityRms)
            || aggregateRmsB
                < static_cast<long double>(settings.minimumActivityRms))
            return ScoreState::Inactive;

        int costCount = 0;
        long double maximumChannelCost = 0.0L;
        for (int ch = 0; ch < channels; ++ch)
        {
            if (!included[static_cast<std::size_t>(ch)]) continue;
            const T* const source = audio.getChannel(ch);
            long double d0Numerator = 0.0L;
            long double d0Denominator = 0.0L;
            long double d1Numerator = 0.0L;
            long double d1Denominator = 0.0L;
            long double previousA = 0.0L;
            long double previousB = 0.0L;
            for (int i = 0; i < window; ++i)
            {
                const long double a =
                    static_cast<long double>(source[end - window + i]) / scale;
                const long double b =
                    static_cast<long double>(source[start + i]) / scale;
                const long double difference = a - b;
                d0Numerator += difference * difference;
                d0Denominator += a * a + b * b;
                if (i > 0)
                {
                    const long double derivativeA = a - previousA;
                    const long double derivativeB = b - previousB;
                    const long double derivativeDifference =
                        derivativeA - derivativeB;
                    d1Numerator += derivativeDifference * derivativeDifference;
                    d1Denominator += derivativeA * derivativeA
                        + derivativeB * derivativeB;
                }
                previousA = a;
                previousB = b;
                scratch.timeA[static_cast<std::size_t>(i)] = static_cast<T>(
                    (a - meanA[static_cast<std::size_t>(ch)])
                    * scratch.window[static_cast<std::size_t>(i)]);
                scratch.timeB[static_cast<std::size_t>(i)] = static_cast<T>(
                    (b - meanB[static_cast<std::size_t>(ch)])
                    * scratch.window[static_cast<std::size_t>(i)]);
            }

            const long double d0 = d0Denominator == 0.0L ? 0.0L
                : std::clamp(0.5L * d0Numerator / d0Denominator, 0.0L, 1.0L);
            const long double d1 = d1Denominator == 0.0L ? 0.0L
                : std::clamp(0.5L * d1Numerator / d1Denominator, 0.0L, 1.0L);

            scratch.fft.forward(scratch.timeA.data(), scratch.frequencyA.data());
            scratch.fft.forward(scratch.timeB.data(), scratch.frequencyB.data());
            const int bins = window / 2 + 1;
            long double totalA = 0.0L;
            long double totalB = 0.0L;
            for (int bin = 0; bin < bins; ++bin)
            {
                const long double weight =
                    (bin == 0 || bin == bins - 1) ? 1.0L : 2.0L;
                const long double ar = scratch.frequencyA[2 * bin];
                const long double ai = scratch.frequencyA[2 * bin + 1];
                const long double br = scratch.frequencyB[2 * bin];
                const long double bi = scratch.frequencyB[2 * bin + 1];
                totalA += weight * (ar * ar + ai * ai);
                totalB += weight * (br * br + bi * bi);
            }

            long double djs = 0.0L;
            if ((totalA == 0.0L) != (totalB == 0.0L))
            {
                djs = 1.0L;
            }
            else if (totalA != 0.0L)
            {
                long double divergenceA = 0.0L;
                long double divergenceB = 0.0L;
                for (int bin = 0; bin < bins; ++bin)
                {
                    const long double weight =
                        (bin == 0 || bin == bins - 1) ? 1.0L : 2.0L;
                    const long double ar = scratch.frequencyA[2 * bin];
                    const long double ai = scratch.frequencyA[2 * bin + 1];
                    const long double br = scratch.frequencyB[2 * bin];
                    const long double bi = scratch.frequencyB[2 * bin + 1];
                    const long double probabilityA =
                        weight * (ar * ar + ai * ai) / totalA;
                    const long double probabilityB =
                        weight * (br * br + bi * bi) / totalB;
                    const long double midpoint =
                        (probabilityA + probabilityB) * 0.5L;
                    if (probabilityA > 0.0L)
                        divergenceA += probabilityA
                            * std::log(probabilityA / midpoint);
                    if (probabilityB > 0.0L)
                        divergenceB += probabilityB
                            * std::log(probabilityB / midpoint);
                }
                djs = std::clamp((divergenceA + divergenceB)
                    * 0.5L / std::numbers::ln2_v<long double>, 0.0L, 1.0L);
            }

            const long double channelCost = std::clamp(
                0.45L * d0 + 0.20L * d1 + 0.35L * djs, 0.0L, 1.0L);
            const long double channelWeight =
                varianceA[static_cast<std::size_t>(ch)]
                + varianceB[static_cast<std::size_t>(ch)];
            if (!std::isfinite(d0) || !std::isfinite(d1)
                || !std::isfinite(djs) || !std::isfinite(channelCost)
                || !std::isfinite(channelWeight))
                return ScoreState::NumericalFailure;
            scratch.channelCosts[static_cast<std::size_t>(costCount++)] =
                { channelWeight, channelCost };
            maximumChannelCost = std::max(maximumChannelCost, channelCost);
        }

        std::sort(scratch.channelCosts.begin(),
                  scratch.channelCosts.begin() + costCount,
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.first < rhs.first
                          || (lhs.first == rhs.first && lhs.second < rhs.second);
                  });
        long double weightedCost = 0.0L;
        long double totalWeight = 0.0L;
        for (int i = 0; i < costCount; ++i)
        {
            weightedCost += scratch.channelCosts[static_cast<std::size_t>(i)].first
                * scratch.channelCosts[static_cast<std::size_t>(i)].second;
            totalWeight += scratch.channelCosts[static_cast<std::size_t>(i)].first;
        }
        if (!(totalWeight > 0.0L) || !std::isfinite(weightedCost)
            || !std::isfinite(totalWeight))
            return ScoreState::NumericalFailure;
        const long double cost = std::clamp(
            0.75L * weightedCost / totalWeight
                + 0.25L * maximumChannelCost,
            0.0L, 1.0L);
        if (!std::isfinite(cost)) return ScoreState::NumericalFailure;
        outputCost = static_cast<T>(cost);
        return std::isfinite(outputCost)
            ? ScoreState::Scored : ScoreState::NumericalFailure;
    }

    [[nodiscard]] static bool rangesOverlap(const void* first,
                                             std::size_t firstBytes,
                                             const void* second,
                                             std::size_t secondBytes) noexcept
    {
        const auto firstBegin = reinterpret_cast<std::uintptr_t>(first);
        const auto secondBegin = reinterpret_cast<std::uintptr_t>(second);
        if (firstBytes > std::numeric_limits<std::uintptr_t>::max() - firstBegin
            || secondBytes
                > std::numeric_limits<std::uintptr_t>::max() - secondBegin)
            return true;
        const auto firstEnd = firstBegin + firstBytes;
        const auto secondEnd = secondBegin + secondBytes;
        return firstBegin < secondEnd && secondBegin < firstEnd;
    }

    [[nodiscard]] static T renderOverlapSample(const T* source,
                                                const Result& loop,
                                                int index,
                                                Crossfade<T>& crossfade) noexcept
    {
        const T tail = source[loop.end - loop.crossfadeLength + index];
        const T head = source[loop.start + index];
        if (index == 0) return tail;
        if (index == loop.crossfadeLength - 1) return head;
        const T scale = std::max(std::abs(tail), std::abs(head));
        if (scale == T(0)) return T(0);
        const T position = static_cast<T>(index)
            / static_cast<T>(loop.crossfadeLength - 1);
        crossfade.setPosition(position);
        const T mixed = crossfade.process(tail / scale, head / scale);
        const T denominator = crossfade.getGainA() + crossfade.getGainB();
        const T normalized = std::clamp(mixed / denominator, T(-1), T(1));
        return scale * normalized;
    }

    [[nodiscard]] static bool dryRenderChannel(const T* source,
                                                const Result& loop) noexcept
    {
        Crossfade<T> crossfade;
        crossfade.setCurve(Crossfade<T>::Curve::EqualPower);
        for (int i = 0; i < loop.crossfadeLength; ++i)
            if (!std::isfinite(renderOverlapSample(source, loop, i, crossfade)))
                return false;
        return true;
    }
};

} // namespace dspark
