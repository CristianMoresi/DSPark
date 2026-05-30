// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Resampler.h
 * @brief High-quality sample rate converter for audio signals.
 *
 * Converts audio between different sample rates (e.g., 44100 ↔ 48000 ↔ 96000 Hz)
 * with configurable quality. Uses windowed-sinc interpolation — the same method
 * used in professional DAWs and mastering tools.
 *
 * Optimized for CPU cache and SIMD autovectorization using double-buffered delay lines.
 * Zero allocations in the audio thread when properly initialized.
 *
 * Two modes:
 *
 * - **Offline (batch)**: Convert an entire buffer at once. Best for file processing.
 * - **Streaming**: Process chunks incrementally. Best for real-time applications
 * where audio arrives in blocks.
 *
 * Dependencies: AudioBuffer.h, DspMath.h, WindowFunctions.h.
 */

#include "AudioBuffer.h"
#include "DspMath.h"
#include "WindowFunctions.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class Resampler
 * @brief Windowed-sinc sample rate converter optimized for real-time DSP.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class Resampler
{
public:
    enum class Quality
    {
        Draft,   ///< 8-point sinc, fast (~60 dB stop-band)
        Normal,  ///< 32-point sinc, balanced (~90 dB stop-band)
        High,    ///< 64-point sinc, high quality (~120 dB stop-band)
        Ultra    ///< 128-point sinc, maximum quality (~140 dB stop-band)
    };

    /**
     * @brief Prepares the resampler for a given rate conversion.
     *
     * Allocates internal buffers. MUST be called outside the audio thread.
     *
     * @param sourceRate      Source sample rate in Hz.
     * @param targetRate      Target sample rate in Hz.
     * @param quality         Interpolation quality (default: Normal).
     */
    void prepare(double sourceRate, double targetRate,
                 Quality quality = Quality::Normal)
    {
        // Guard against non-positive rates (would make ratio_ = inf/NaN and
        // poison every downstream division).
        sourceRate_ = std::max(sourceRate, 1.0);
        targetRate_ = std::max(targetRate, 1.0);
        ratio_ = targetRate_ / sourceRate_;

        sincPoints_ = qualityToSincPoints(quality);

        buildSincTable();
        reset();
    }

    /**
     * @brief Prepares the resampler using AudioSpec (unified API).
     *
     * Pre-allocates multi-channel states. MUST be called outside the audio thread.
     *
     * @param spec       Audio environment (sampleRate, numChannels used).
     * @param targetRate Target sample rate in Hz.
     * @param quality    Interpolation quality (default: Normal).
     */
    void prepare(const AudioSpec& spec, double targetRate,
                 Quality quality = Quality::Normal)
    {
        prepare(spec.sampleRate, targetRate, quality);
        ensureChannelStates(spec.numChannels);
    }

    /** * @brief Resets the internal state (delay lines, all channels) to zero. 
     * Thread-safe to call from audio thread if no allocation is triggered.
     */
    void reset() noexcept
    {
        history_.assign(static_cast<size_t>(sincPoints_ * 2), T(0));
        writePos_ = 0;
        fractionalPos_ = 0.0;

        for (auto& cs : channelStates_)
        {
            std::fill(cs.history.begin(), cs.history.end(), T(0));
            cs.writePos = 0;
            cs.fractionalPos = 0.0;
        }
    }

    /**
     * @brief Resamples an entire buffer (offline batch processing).
     *
     * @param input       Source audio samples.
     * @param inputLength Number of input samples.
     * @return Vector of resampled output samples.
     */
    [[nodiscard]] std::vector<T> process(const T* input, int inputLength)
    {
        auto outputLength = static_cast<int>(
            std::ceil(static_cast<double>(inputLength) * ratio_));

        std::vector<T> output(static_cast<size_t>(outputLength));
        const int halfSinc = sincPoints_ / 2;

        for (int outIdx = 0; outIdx < outputLength; ++outIdx)
        {
            // Calculate absolute source position to avoid float drift accumulation
            double srcPos = static_cast<double>(outIdx) / ratio_;
            if (srcPos >= inputLength) break;

            int intPos = static_cast<int>(srcPos);
            double frac = srcPos - static_cast<double>(intPos);

            output[static_cast<size_t>(outIdx)] = interpolateOffline(input, inputLength, intPos, frac, halfSinc);
        }

        return output;
    }

    /**
     * @brief Resamples a block of audio in single-channel streaming mode.
     *
     * @param input       Input audio samples.
     * @param inputLength Number of input samples.
     * @param output      Output buffer (must hold at least getMaxOutputSamples()).
     * @return Number of output samples produced.
     */
    int processBlock(const T* input, int inputLength, T* output) noexcept
    {
        int outIdx = 0;

        for (int i = 0; i < inputLength; ++i)
        {
            pushSampleSingle(input[i]);

            while (fractionalPos_ < 1.0)
            {
                output[outIdx++] = interpolateFromHistory(history_.data(), writePos_, fractionalPos_);
                fractionalPos_ += 1.0 / ratio_;
            }
            fractionalPos_ -= 1.0;
        }

        return outIdx;
    }

    /**
     * @brief Resamples multi-channel audio using AudioBufferView (streaming).
     *
     * Processes each channel sequentially with its independent state.
     *
     * @param input  Input audio buffer view.
     * @param output Output audio buffer view (pre-allocated).
     * @return Number of output samples produced per channel.
     */
    int processBlock(AudioBufferView<T> input, AudioBufferView<T> output) noexcept
    {
        int numCh = std::min(input.getNumChannels(), output.getNumChannels());
        int inLen = input.getNumSamples();

        assert(static_cast<int>(channelStates_.size()) >= numCh && 
               "Resampler channels not allocated! Call prepare(spec...) first.");

        int outCount = 0;
        for (int ch = 0; ch < numCh; ++ch)
        {
            outCount = processChannel(input.getChannel(ch), inLen,
                                      output.getChannel(ch),
                                      channelStates_[static_cast<size_t>(ch)]);
        }

        return outCount;
    }

    /**
     * @brief Returns the maximum number of output samples for a given input length.
     * @param inputLength Number of input samples.
     * @return Maximum possible output samples.
     */
    [[nodiscard]] int getMaxOutputSamples(int inputLength) const noexcept
    {
        return static_cast<int>(
            std::ceil(static_cast<double>(inputLength) * ratio_)) + 2;
    }

    /** @brief Returns the conversion ratio (targetRate / sourceRate). */
    [[nodiscard]] double getRatio() const noexcept { return ratio_; }

    /** @brief Returns the latency in output samples introduced by the sinc filter. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return static_cast<int>(std::round(static_cast<double>(sincPoints_ / 2) * ratio_));
    }

private:
    static constexpr int kOversample = 256;

    static int qualityToSincPoints(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Draft:  return 8;
            case Quality::Normal: return 32;
            case Quality::High:   return 64;
            case Quality::Ultra:  return 128;
        }
        return 32;
    }

    void buildSincTable()
    {
        sincTable_.resize(static_cast<size_t>(kOversample * sincPoints_));

        std::vector<T> kaiserWin(static_cast<size_t>(sincPoints_));
        WindowFunctions<T>::kaiser(kaiserWin.data(), sincPoints_, T(10), false);

        const int halfSinc = sincPoints_ / 2;
        constexpr double kPi = std::numbers::pi;

        // Apply 0.95 margin on downsampling to prevent transition-band aliasing.
        double cutoff = (ratio_ < 1.0) ? (ratio_ * 0.95) : 1.0;

        for (int phase = 0; phase < kOversample; ++phase)
        {
            double frac = static_cast<double>(phase) / static_cast<double>(kOversample);
            const int base = phase * sincPoints_;
            double sum = 0.0;

            for (int tap = 0; tap < sincPoints_; ++tap)
            {
                double x = (static_cast<double>(tap - halfSinc) + frac) * cutoff;
                T sincVal;

                if (std::abs(x) < 1e-10)
                    sincVal = static_cast<T>(cutoff);
                else
                    sincVal = static_cast<T>(cutoff * std::sin(kPi * x) / (kPi * x));

                sincVal *= kaiserWin[static_cast<size_t>(tap)];
                sincTable_[static_cast<size_t>(base + tap)] = sincVal;
                sum += static_cast<double>(sincVal);
            }

            // Normalise each polyphase branch to unity DC gain so the passband
            // is flat (otherwise the windowed sinc leaves a small gain ripple).
            if (std::abs(sum) > 1e-12)
            {
                const T inv = static_cast<T>(1.0 / sum);
                for (int tap = 0; tap < sincPoints_; ++tap)
                    sincTable_[static_cast<size_t>(base + tap)] *= inv;
            }
        }
    }

    /**
     * @brief Offline interpolation with boundary checks.
     */
    T interpolateOffline(const T* data, int length, int intPos, double frac, int halfSinc) const noexcept
    {
        double exactPhase = frac * static_cast<double>(kOversample);
        int p1 = static_cast<int>(exactPhase);
        double pf = exactPhase - static_cast<double>(p1);

        int p0 = std::max(p1 - 1, 0);
        int p2 = std::min(p1 + 1, kOversample - 1);
        int p3 = std::min(p1 + 2, kOversample - 1);
        p1 = std::clamp(p1, 0, kOversample - 1);

        size_t offset0 = static_cast<size_t>(p0 * sincPoints_);
        size_t offset1 = static_cast<size_t>(p1 * sincPoints_);
        size_t offset2 = static_cast<size_t>(p2 * sincPoints_);
        size_t offset3 = static_cast<size_t>(p3 * sincPoints_);

        T t = static_cast<T>(pf);
        T result = T(0);

        for (int tap = 0; tap < sincPoints_; ++tap)
        {
            int srcIdx = intPos + tap - halfSinc;
            T sample = (srcIdx >= 0 && srcIdx < length) ? data[srcIdx] : T(0);

            T y0 = sincTable_[offset0 + tap];
            T y1 = sincTable_[offset1 + tap];
            T y2 = sincTable_[offset2 + tap];
            T y3 = sincTable_[offset3 + tap];

            T a1 = T(0.5) * (y2 - y0);
            T a2 = y0 - T(2.5) * y1 + T(2) * y2 - T(0.5) * y3;
            T a3 = T(0.5) * (y3 - y0) + T(1.5) * (y1 - y2);

            T sincVal = y1 + t * (a1 + t * (a2 + t * a3));
            result += sample * sincVal;
        }

        return result;
    }

    /**
     * @brief Pushes a sample into the global single-channel history buffer.
     * Uses double-buffering trick to eliminate modulo in read path.
     */
    void pushSampleSingle(T sample) noexcept
    {
        history_[static_cast<size_t>(writePos_)] = sample;
        history_[static_cast<size_t>(writePos_ + sincPoints_)] = sample;
        writePos_++;
        if (writePos_ >= sincPoints_) writePos_ = 0;
    }

    /**
     * @brief High-performance contiguous memory interpolator.
     * Relies on the double-buffered history allowing linear SIMD loading.
     */
    T interpolateFromHistory(const T* historyPtr, int readPos, double frac) const noexcept
    {
        double exactPhase = frac * static_cast<double>(kOversample);
        int p1 = static_cast<int>(exactPhase);
        double pf = exactPhase - static_cast<double>(p1);

        int p0 = std::max(p1 - 1, 0);
        int p2 = std::min(p1 + 1, kOversample - 1);
        int p3 = std::min(p1 + 2, kOversample - 1);
        p1 = std::clamp(p1, 0, kOversample - 1);

        size_t offset0 = static_cast<size_t>(p0 * sincPoints_);
        size_t offset1 = static_cast<size_t>(p1 * sincPoints_);
        size_t offset2 = static_cast<size_t>(p2 * sincPoints_);
        size_t offset3 = static_cast<size_t>(p3 * sincPoints_);

        T t = static_cast<T>(pf);
        T result = T(0);

        // Linear memory read from 'readPos'. No modulo required.
        const T* h_ptr = &historyPtr[readPos];

        for (int tap = 0; tap < sincPoints_; ++tap)
        {
            T y0 = sincTable_[offset0 + tap];
            T y1 = sincTable_[offset1 + tap];
            T y2 = sincTable_[offset2 + tap];
            T y3 = sincTable_[offset3 + tap];

            T a1 = T(0.5) * (y2 - y0);
            T a2 = y0 - T(2.5) * y1 + T(2) * y2 - T(0.5) * y3;
            T a3 = T(0.5) * (y3 - y0) + T(1.5) * (y1 - y2);

            T sincVal = y1 + t * (a1 + t * (a2 + t * a3));
            result += h_ptr[tap] * sincVal;
        }

        return result;
    }

    struct ChannelState
    {
        std::vector<T> history;
        int writePos = 0;
        double fractionalPos = 0.0;
    };

    void ensureChannelStates(int numChannels)
    {
        if (static_cast<int>(channelStates_.size()) < numChannels)
        {
            channelStates_.resize(static_cast<size_t>(numChannels));
            for (auto& cs : channelStates_)
            {
                if (cs.history.empty())
                {
                    // Allocate double size to support modulo-free SIMD convolution
                    cs.history.assign(static_cast<size_t>(sincPoints_ * 2), T(0));
                    cs.writePos = 0;
                    cs.fractionalPos = 0.0;
                }
            }
        }
    }

    int processChannel(const T* input, int inputLength, T* output,
                       ChannelState& state) noexcept
    {
        int outIdx = 0;

        for (int i = 0; i < inputLength; ++i)
        {
            state.history[static_cast<size_t>(state.writePos)] = input[i];
            state.history[static_cast<size_t>(state.writePos + sincPoints_)] = input[i];
            state.writePos++;
            
            if (state.writePos >= sincPoints_) 
                state.writePos = 0;

            while (state.fractionalPos < 1.0)
            {
                output[outIdx++] = interpolateFromHistory(state.history.data(), state.writePos, state.fractionalPos);
                state.fractionalPos += 1.0 / ratio_;
            }

            state.fractionalPos -= 1.0;
        }

        return outIdx;
    }

    double sourceRate_ = 44100.0;
    double targetRate_ = 48000.0;
    double ratio_ = 1.0;
    double fractionalPos_ = 0.0;

    int sincPoints_ = 32;

    std::vector<T> sincTable_;
    std::vector<T> history_;
    int writePos_ = 0;

    std::vector<ChannelState> channelStates_;
};

} // namespace dspark