// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LoudnessNormalizer.h
 * @brief Transparent offline LUFS normalization under a true-peak ceiling.
 *
 * `normalize()` measures the complete programme, chooses one constant gain and
 * multiplies every sample of every channel by that same value exactly once.
 * A constant gain preserves the waveform, crest factor and spectral balance;
 * there is no limiter, envelope, lookahead, release or iterative processing.
 * When the requested loudness and the true-peak ceiling conflict, the ceiling
 * wins and the result reports the signed loudness miss through `outLUFS`, the
 * requested and applied gains, and `ceilingLimited`.
 *
 * Integrated loudness follows the first-two-channel semantics of
 * `LoudnessMeter`: mono uses channel 0 and wider buffers use channels 0 and 1.
 * The ITU-R BS.1770 Annex 2 true-peak ceiling is different: it is measured over
 * every runtime channel and the selected gain is applied to every runtime
 * channel. Each channel is followed by `TruePeakDetector::getTaps()-1` virtual
 * zeros during offline measurement, so an interpolation peak whose causal FIR
 * support completes after the final programme sample is included without
 * extending or mutating the caller's buffer. Consequently neither a channel
 * above index 15 nor an end-of-programme peak can escape the ceiling.
 *
 * The operation validates dimensions and sample rate, then scans every sample
 * before measurement or mutation. Empty input, an invalid rate, non-finite
 * input, silence and a non-representable numerical plan return an explicit
 * status. Every rejected path leaves the complete caller buffer unchanged;
 * non-finite rejection preserves NaN payloads and signs bit for bit.
 *
 * OFFLINE ONLY. The algorithm makes full-programme validation, measurement,
 * gain and verification passes. It uses zero bytes of programme-length scratch
 * and performs no allocation. Its fixed measurement state is
 * O(`MaxChannels * Annex2Taps`) for true-peak histories plus one constant-size
 * loudness meter, independent of programme duration.
 *
 * Dependencies: LoudnessMeter.h, AudioBuffer.h, DspMath.h, DenormalGuard.h,
 * TruePeakDetector.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/TruePeakDetector.h"
#include "LoudnessMeter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dspark {

/**
 * @class LoudnessNormalizer
 * @brief Offline waveform-preserving LUFS normalizer with a hard true-peak ceiling.
 *
 * Threading: offline and single-threaded. The caller owns the buffer exclusively
 * for the duration of normalize(). The two parameter controls are lock-free
 * atomic words and may be changed between runs; changing either while a run is
 * in progress is not supported.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LoudnessNormalizer final
{
public:
    /** @brief Completion or conservative no-mutation outcome. */
    enum class Status : std::uint8_t
    {
        Success,
        NoMeasurableLoudness,
        EmptyInput,
        InvalidSampleRate,
        NonFiniteInput,
        NumericalFailure
    };

    /** @brief Measurements and the constant-gain decision for one run. */
    struct Result
    {
        Status status = Status::EmptyInput;
        T measuredLUFS = T(-100);     ///< Integrated loudness of the input.
        T requestedGainDb = T(0);     ///< Gain needed to reach the target.
        T appliedGainDb = T(0);       ///< One gain applied to every sample.
        T outLUFS = T(-100);          ///< Re-measured output loudness.
        T outTruePeakDb = T(-100);    ///< All-channel output true peak.
        bool targetReached = false;   ///< Output is within 0.1 LU of target.
        bool ceilingLimited = false;  ///< The ceiling selected the lower gain.
    };

    // -- Parameters (control thread, between runs) ----------------------------

    /** @brief Sets the target integrated loudness (default -23 LUFS). */
    void setTargetLUFS(T target) noexcept
    {
        if (std::isfinite(target))
            targetLUFS_.store(target, std::memory_order_relaxed);
    }

    /** @return The configured target integrated loudness. */
    [[nodiscard]] T getTargetLUFS() const noexcept
    {
        return targetLUFS_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Sets the true-peak ceiling (default -1 dBTP).
     *
     * Finite values are clamped to [-40, 0] dBTP. Non-finite requests leave
     * the current control value unchanged.
     */
    void setTruePeakCeilingDb(T ceilingDb) noexcept
    {
        if (std::isfinite(ceilingDb))
            ceilingDb_.store(std::clamp(ceilingDb, T(-40), T(0)),
                             std::memory_order_relaxed);
    }

    /** @return The configured true-peak ceiling. */
    [[nodiscard]] T getTruePeakCeilingDb() const noexcept
    {
        return ceilingDb_.load(std::memory_order_relaxed);
    }

    // -- Offline processing ---------------------------------------------------

    /**
     * @brief Normalizes `audio` in place with one validated constant gain.
     * @param audio      Complete owning buffer; every runtime channel is bound.
     * @param sampleRate Finite positive sample rate in Hz.
     * @return Explicit status, input/output measurements and gain decision.
     *
     * Loudness uses channel 0 for mono and channels 0/1 for wider buffers.
     * True peak and gain application cover every runtime channel up to the
     * buffer's compile-time `MaxChannels`. Every failure status is decided
     * before the first write and leaves `audio` bitwise unchanged.
     */
    template <int MaxChannels>
    [[nodiscard]] Result normalize(AudioBuffer<T, MaxChannels>& audio,
                                   double sampleRate)
    {
        DenormalGuard guard;

        Result out;
        const int numChannels = audio.getNumChannels();
        const int numSamples = audio.getNumSamples();
        if (numChannels <= 0 || numSamples <= 0)
        {
            out.status = Status::EmptyInput;
            return out;
        }
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
        {
            out.status = Status::InvalidSampleRate;
            return out;
        }

        T maxAbsInput = T(0);
        for (int channel = 0; channel < numChannels; ++channel)
        {
            const T* samples = audio.getChannel(channel);
            for (int i = 0; i < numSamples; ++i)
            {
                const T sample = samples[i];
                if (!std::isfinite(sample))
                {
                    out.status = Status::NonFiniteInput;
                    return out;
                }
                maxAbsInput = std::max(maxAbsInput, std::abs(sample));
            }
        }

        const T measured = measureIntegrated(audio, sampleRate);
        if (!meter_.isMeasurementValid())
        {
            out.status = Status::NumericalFailure;
            return out;
        }
        const T inputTruePeak = measureTruePeakLinear(audio);
        if (!std::isfinite(measured) || !std::isfinite(inputTruePeak)
            || inputTruePeak < T(0))
        {
            out.status = Status::NumericalFailure;
            return out;
        }

        if (measured <= kMeterFloorLUFS || !(inputTruePeak > T(0)))
        {
            out.measuredLUFS = measured;
            out.status = Status::NoMeasurableLoudness;
            out.outLUFS = measured;
            out.outTruePeakDb = linearToDecibels(inputTruePeak, T(-100));
            return out;
        }

        const T target = targetLUFS_.load(std::memory_order_relaxed);
        const T ceilingDb = ceilingDb_.load(std::memory_order_relaxed);
        const T requestedGainDb = target - measured;
        const T requestedLinear = decibelsToLinear(requestedGainDb);
        const T ceilingLinear = decibelsToLinear(ceilingDb);
        const T ceilingSafeLinear = static_cast<T>(
            static_cast<double>(ceilingLinear) * (1.0 - kCeilingAimMargin)
            / static_cast<double>(inputTruePeak));
        const T appliedLinear = std::min(requestedLinear, ceilingSafeLinear);
        const T appliedGainDb = linearToDecibels(appliedLinear, T(0));

        if (!std::isfinite(requestedGainDb) || !std::isfinite(requestedLinear)
            || !std::isfinite(ceilingLinear)
            || !std::isfinite(ceilingSafeLinear)
            || !std::isfinite(appliedLinear) || !(appliedLinear > T(0))
            || !std::isfinite(appliedGainDb)
            || static_cast<long double>(maxAbsInput)
               > static_cast<long double>(std::numeric_limits<T>::max())
                 / static_cast<long double>(appliedLinear))
        {
            out.status = Status::NumericalFailure;
            return out;
        }

        out.measuredLUFS = measured;
        out.requestedGainDb = requestedGainDb;
        out.appliedGainDb = appliedGainDb;
        out.ceilingLimited = ceilingSafeLinear < requestedLinear;

        // This is the only caller-buffer mutation in the operation.
        for (int channel = 0; channel < numChannels; ++channel)
        {
            T* samples = audio.getChannel(channel);
            for (int i = 0; i < numSamples; ++i)
                samples[i] = static_cast<T>(samples[i] * appliedLinear);
        }

        out.outLUFS = measureIntegrated(audio, sampleRate);
        out.outTruePeakDb = linearToDecibels(measureTruePeakLinear(audio), T(-100));
        out.targetReached = std::abs(out.outLUFS - target) <= kTargetToleranceLU;
        out.status = Status::Success;
        return out;
    }

private:
    static constexpr T kMeterFloorLUFS = T(-99);
    static constexpr T kTargetToleranceLU = T(0.1);
    static constexpr double kCeilingAimMargin = 1.0e-4;

    [[nodiscard]] static T decibelsToLinear(T decibels) noexcept
    {
        return static_cast<T>(
            std::pow(10.0, static_cast<double>(decibels) / 20.0));
    }

    [[nodiscard]] static T linearToDecibels(T gain, T zeroSentinel) noexcept
    {
        return gain > T(0)
             ? static_cast<T>(20.0 * std::log10(static_cast<double>(gain)))
             : zeroSentinel;
    }

    template <int MaxChannels>
    [[nodiscard]] T measureIntegrated(const AudioBuffer<T, MaxChannels>& audio,
                                      double sampleRate) noexcept
    {
        const int numChannels = audio.getNumChannels();
        meter_.prepare(sampleRate, std::min(numChannels, 2));
        meter_.reset();
        if (numChannels >= 2)
            meter_.process(audio.getChannel(0), audio.getChannel(1),
                           audio.getNumSamples());
        else
            meter_.process(audio.getChannel(0), audio.getNumSamples());
        return meter_.getIntegratedLUFS();
    }

    template <int MaxChannels>
    [[nodiscard]] static T measureTruePeakLinear(
        const AudioBuffer<T, MaxChannels>& audio) noexcept
    {
        TruePeakDetector<T, MaxChannels> detector;
        detector.reset();
        T peak = T(0);
        for (int channel = 0; channel < audio.getNumChannels(); ++channel)
        {
            const T* samples = audio.getChannel(channel);
            for (int i = 0; i < audio.getNumSamples(); ++i)
                peak = std::max(peak, detector.processSample(samples[i], channel));
            for (int i = 0; i < TruePeakDetector<T, MaxChannels>::getTaps() - 1; ++i)
                peak = std::max(peak, detector.processSample(T(0), channel));
        }
        return peak;
    }

    std::atomic<T> targetLUFS_ { T(-23) };
    std::atomic<T> ceilingDb_ { T(-1) };
    LoudnessMeter<T> meter_;
};

} // namespace dspark
