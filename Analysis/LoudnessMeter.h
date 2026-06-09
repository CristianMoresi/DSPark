// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file LoudnessMeter.h
 * @brief Thread-safe EBU R128 / ITU-R BS.1770 loudness meter.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/TruePeakDetector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <array>

namespace dspark {

/**
 * @class LoudnessMeter
 * @brief Real-time safe EBU R128 loudness meter.
 *
 * Implements Momentary (400ms), Short-Term (3s), and Integrated (gated) measurements.
 * Utilizes a constant-memory histogram for infinite integrated loudness tracking
 * without memory allocation or O(N) CPU scaling, ensuring strict RT compliance.
 *
 * All readout methods (`getMomentaryLUFS`, etc.) are lock-free and thread-safe
 * to be called from GUI threads while the audio thread is processing.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LoudnessMeter
{
public:
    /**
     * @brief Prepares the meter and pre-calculates filter coefficients.
     * @param sampleRate Sample rate in Hz (must be > 0).
     * @param numChannels Number of channels (currently up to 2 supported).
     */
    void prepare(double sampleRate, int numChannels = 2)
    {
        if (sampleRate <= 0.0) return;
        
        sampleRate_ = sampleRate;
        numChannels_ = std::clamp(numChannels, 1, kMaxChannels);

        computeKWeighting(sampleRate);

        // 100 ms block length
        blockSamples_ = static_cast<int>(sampleRate * 0.1);
        
        reset();
    }

    /** @brief Unified API preparation. */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Processes an interleaved or non-interleaved buffer (Read-only).
     * @param buffer AudioBufferView to analyze.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh >= 2)
            process(buffer.getChannel(0), buffer.getChannel(1), nS);
        else if (nCh == 1)
            process(buffer.getChannel(0), nS);
    }

    /**
     * @brief Processes a mono block of samples.
     * @param data Pointer to mono audio data.
     * @param numSamples Number of samples to process.
     */
    void process(const T* data, int numSamples) noexcept
    {
        T tpMax = truePeakMax_.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            tpMax = std::max(tpMax, truePeak_.processSample(data[i], 0));

            double filtered = applyKWeighting(static_cast<double>(data[i]), 0);
            currentBlockPower_ += filtered * filtered;

            if (++currentBlockSamples_ >= blockSamples_)
                commitBlock();
        }
        truePeakMax_.store(tpMax, std::memory_order_relaxed);
    }

    /**
     * @brief Processes a stereo block of samples.
     * @param left Pointer to left channel data.
     * @param right Pointer to right channel data.
     * @param numSamples Number of samples per channel.
     */
    void process(const T* left, const T* right, int numSamples) noexcept
    {
        T tpMax = truePeakMax_.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            tpMax = std::max(tpMax, truePeak_.processSample(left[i], 0));
            tpMax = std::max(tpMax, truePeak_.processSample(right[i], 1));

            double filtL = applyKWeighting(static_cast<double>(left[i]), 0);
            double filtR = applyKWeighting(static_cast<double>(right[i]), 1);

            currentBlockPower_ += (filtL * filtL + filtR * filtR);

            if (++currentBlockSamples_ >= blockSamples_)
                commitBlock();
        }
        truePeakMax_.store(tpMax, std::memory_order_relaxed);
    }

    /**
     * @brief Reads the Momentary loudness (400 ms window).
     * @return Loudness in LUFS (minimum -100.0).
     */
    [[nodiscard]] T getMomentaryLUFS() const noexcept
    {
        return calculateLUFSFromBlocks(4);
    }

    /**
     * @brief Reads the Short-term loudness (3 second window).
     * @return Loudness in LUFS (minimum -100.0).
     */
    [[nodiscard]] T getShortTermLUFS() const noexcept
    {
        return calculateLUFSFromBlocks(30);
    }

    /**
     * @brief Computes the Integrated loudness using standard two-pass gating.
     * @note Executed in O(1) time thanks to the histogram implementation. Safe for RT thread.
     * @return Loudness in LUFS (minimum -100.0).
     */
    [[nodiscard]] T getIntegratedLUFS() const noexcept
    {
        // Pass 1: Absolute Gate (-70 LUFS)
        double sumPowerUngated = 0.0;
        uint32_t countUngated = 0;

        for (int i = 0; i < kNumBins; ++i)
        {
            uint32_t binCount = histogram_[i].load(std::memory_order_relaxed);
            if (binCount > 0)
            {
                double binPower = lufsToPower(kMinHistogramLUFS + static_cast<double>(i) * kBinWidth);
                sumPowerUngated += binPower * binCount;
                countUngated += binCount;
            }
        }

        if (countUngated == 0) return T(-100);

        double meanPowerUngated = sumPowerUngated / countUngated;
        double ungatedLUFS = powerToLUFS(meanPowerUngated);

        // Pass 2: Relative Gate (-10 LU below ungated mean)
        double relativeGateLUFS = ungatedLUFS - 10.0;
        int relativeGateBin = static_cast<int>(std::floor((relativeGateLUFS - kMinHistogramLUFS) / kBinWidth));
        relativeGateBin = std::clamp(relativeGateBin, 0, kNumBins - 1);

        double sumPowerGated = 0.0;
        uint32_t countGated = 0;

        for (int i = relativeGateBin; i < kNumBins; ++i)
        {
            uint32_t binCount = histogram_[i].load(std::memory_order_relaxed);
            if (binCount > 0)
            {
                double binPower = lufsToPower(kMinHistogramLUFS + static_cast<double>(i) * kBinWidth);
                sumPowerGated += binPower * binCount;
                countGated += binCount;
            }
        }

        if (countGated == 0) return T(-100);

        return static_cast<T>(powerToLUFS(sumPowerGated / countGated));
    }

    /**
     * @brief Returns the maximum true peak observed since the last reset.
     * @return True peak in dBTP (ITU-R BS.1770-4, 4x oversampled estimate).
     */
    [[nodiscard]] T getTruePeakDb() const noexcept
    {
        return gainToDecibels(truePeakMax_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Computes the EBU R128 Loudness Range (EBU Tech 3342).
     *
     * Short-term (3 s) loudness sampled once per second, gated at -70 LUFS
     * absolute and -20 LU relative; LRA is the spread between the 10th and
     * 95th percentiles. O(bins) and real-time safe like the integrated gate.
     *
     * @return Loudness range in LU (0 if not enough material yet).
     */
    [[nodiscard]] T getLoudnessRange() const noexcept
    {
        // Pass 1: relative gate from the mean of absolute-gated ST values.
        double sumPower = 0.0;
        uint32_t count = 0;
        for (int i = 0; i < kNumBins; ++i)
        {
            const uint32_t c = lraHistogram_[i].load(std::memory_order_relaxed);
            if (c > 0)
            {
                sumPower += lufsToPower(kMinHistogramLUFS + i * kBinWidth) * c;
                count += c;
            }
        }
        if (count == 0) return T(0);

        const double relGateLUFS = powerToLUFS(sumPower / count) - 20.0;
        int gateBin = static_cast<int>(std::floor((relGateLUFS - kMinHistogramLUFS) / kBinWidth));
        gateBin = std::clamp(gateBin, 0, kNumBins - 1);

        // Pass 2: 10th / 95th percentiles above the relative gate.
        uint32_t gatedCount = 0;
        for (int i = gateBin; i < kNumBins; ++i)
            gatedCount += lraHistogram_[i].load(std::memory_order_relaxed);
        if (gatedCount == 0) return T(0);

        const auto target10 = static_cast<uint32_t>(0.10 * gatedCount);
        const auto target95 = static_cast<uint32_t>(0.95 * gatedCount);

        double p10 = kMinHistogramLUFS, p95 = kMaxHistogramLUFS;
        uint32_t running = 0;
        bool have10 = false;
        for (int i = gateBin; i < kNumBins; ++i)
        {
            running += lraHistogram_[i].load(std::memory_order_relaxed);
            if (!have10 && running >= target10)
            {
                p10 = kMinHistogramLUFS + i * kBinWidth;
                have10 = true;
            }
            if (running >= target95)
            {
                p95 = kMinHistogramLUFS + i * kBinWidth;
                break;
            }
        }
        return static_cast<T>(std::max(0.0, p95 - p10));
    }

    /**
     * @brief Clears all measurements and resets filters.
     * Lock-free, but may tear slightly if audio thread is simultaneously writing.
     */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            preState_[ch] = {};
            rlbState_[ch] = {};
        }

        for (auto& power : blockPowers_)
            power.store(0.0, std::memory_order_relaxed);

        for (auto& bin : histogram_)
            bin.store(0, std::memory_order_relaxed);
        for (auto& bin : lraHistogram_)
            bin.store(0, std::memory_order_relaxed);

        truePeak_.reset();
        truePeakMax_.store(T(0), std::memory_order_relaxed);

        blockWritePos_.store(0, std::memory_order_relaxed);
        totalCommittedBlocks_ = 0;
        currentBlockPower_ = 0.0;
        currentBlockSamples_ = 0;
    }

private:
    static constexpr int kMaxChannels = 2; // Expandable to 8 with proper spatial weighting
    
    // Histogram specs: -70 LUFS to +30 LUFS with 0.1 resolution = 1000 bins
    static constexpr double kMinHistogramLUFS = -70.0;
    static constexpr double kMaxHistogramLUFS = 30.0;
    static constexpr double kBinWidth = 0.1;
    static constexpr int kNumBins = 1000;

    // Filters state (Forced to double to prevent DF2T precision issues)
    struct BiquadState { double z1 = 0.0, z2 = 0.0; };
    struct BiquadCoeff { double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0; };

    void computeKWeighting(double sr)
    {
        // Exact equations derived from ITU-R BS.1770-4
        double fc = 1681.97, G = 3.99984, Q = 0.70710678;
        double A = std::pow(10.0, G / 40.0);
        double w0 = 2.0 * std::numbers::pi * fc / sr;
        double alpha = std::sin(w0) / (2.0 * Q);

        double a0 = (A + 1.0) - (A - 1.0) * std::cos(w0) + 2.0 * std::sqrt(A) * alpha;
        pre_.b0 = (A * ((A + 1.0) + (A - 1.0) * std::cos(w0) + 2.0 * std::sqrt(A) * alpha)) / a0;
        pre_.b1 = (-2.0 * A * ((A - 1.0) + (A + 1.0) * std::cos(w0))) / a0;
        pre_.b2 = (A * ((A + 1.0) + (A - 1.0) * std::cos(w0) - 2.0 * std::sqrt(A) * alpha)) / a0;
        pre_.a1 = (2.0 * ((A - 1.0) - (A + 1.0) * std::cos(w0))) / a0;
        pre_.a2 = (((A + 1.0) - (A - 1.0) * std::cos(w0) - 2.0 * std::sqrt(A) * alpha)) / a0;

        double fc2 = 38.13547, Q2 = 0.500327;
        double w02 = 2.0 * std::numbers::pi * fc2 / sr;
        double alpha2 = std::sin(w02) / (2.0 * Q2);

        double a02 = 1.0 + alpha2;
        rlb_.b0 = (1.0 + std::cos(w02)) / 2.0 / a02;
        rlb_.b1 = -(1.0 + std::cos(w02)) / a02;
        rlb_.b2 = (1.0 + std::cos(w02)) / 2.0 / a02;
        rlb_.a1 = -2.0 * std::cos(w02) / a02;
        rlb_.a2 = (1.0 - alpha2) / a02;
    }

    double applyBiquad(double input, const BiquadCoeff& c, BiquadState& s) noexcept
    {
        // Adding 1e-18 protects against denormal numbers (Flush-To-Zero fallback)
        double output = c.b0 * input + s.z1;
        s.z1 = c.b1 * input - c.a1 * output + s.z2;
        s.z2 = c.b2 * input - c.a2 * output + 1e-18; 
        return output;
    }

    double applyKWeighting(double input, int channel) noexcept
    {
        double x = applyBiquad(input, pre_, preState_[channel]);
        return applyBiquad(x, rlb_, rlbState_[channel]);
    }

    void commitBlock() noexcept
    {
        double meanPower = currentBlockPower_ / currentBlockSamples_;

        // Update sliding window ring buffer
        int currentPos = blockWritePos_.load(std::memory_order_relaxed);
        blockPowers_[currentPos].store(meanPower, std::memory_order_relaxed);

        int nextPos = (currentPos + 1) % 30; // 30 blocks = 3s max window
        blockWritePos_.store(nextPos, std::memory_order_release);
        ++totalCommittedBlocks_;

        // BS.1770-4 integrated gating uses 400 ms blocks with 75% overlap.
        // With a 100 ms sub-block hop, that is EXACTLY the mean of the last
        // four sub-blocks committed at every 100 ms step. (Gating raw 100 ms
        // blocks has higher variance and fails the EBU conformance vectors.)
        if (totalCommittedBlocks_ >= 4)
        {
            double gating400 = meanPower;
            for (int back = 1; back < 4; ++back)
            {
                int idx = (currentPos - back + 30) % 30;
                gating400 += blockPowers_[idx].load(std::memory_order_relaxed);
            }
            gating400 *= 0.25;

            const double lufs = powerToLUFS(gating400);
            if (lufs >= kMinHistogramLUFS)
            {
                int binIndex = static_cast<int>(std::round((lufs - kMinHistogramLUFS) / kBinWidth));
                binIndex = std::clamp(binIndex, 0, kNumBins - 1);
                histogram_[binIndex].fetch_add(1, std::memory_order_relaxed);
            }
        }

        // EBU Tech 3342 loudness range: short-term (3 s) values sampled every
        // 1 s (10 sub-blocks), absolute-gated at -70 LUFS into a histogram.
        if (totalCommittedBlocks_ >= 30 && (totalCommittedBlocks_ % 10) == 0)
        {
            const double stLufs = static_cast<double>(calculateLUFSFromBlocks(30));
            if (stLufs >= kMinHistogramLUFS)
            {
                int binIndex = static_cast<int>(std::round((stLufs - kMinHistogramLUFS) / kBinWidth));
                binIndex = std::clamp(binIndex, 0, kNumBins - 1);
                lraHistogram_[binIndex].fetch_add(1, std::memory_order_relaxed);
            }
        }

        currentBlockPower_ = 0.0;
        currentBlockSamples_ = 0;
    }

    T calculateLUFSFromBlocks(int numBlocks) const noexcept
    {
        double sum = 0.0;
        int currentPos = blockWritePos_.load(std::memory_order_acquire);
        
        for (int i = 0; i < numBlocks; ++i)
        {
            int idx = (currentPos - 1 - i + 30) % 30;
            sum += blockPowers_[idx].load(std::memory_order_relaxed);
        }

        return static_cast<T>(powerToLUFS(sum / numBlocks));
    }

    [[nodiscard]] static double powerToLUFS(double meanPower) noexcept
    {
        if (meanPower <= 1e-10) return -100.0; // Prevent log10(0)
        return -0.691 + 10.0 * std::log10(meanPower);
    }

    [[nodiscard]] static double lufsToPower(double lufs) noexcept
    {
        return std::pow(10.0, (lufs + 0.691) / 10.0);
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 2;

    BiquadCoeff pre_, rlb_;
    BiquadState preState_[kMaxChannels], rlbState_[kMaxChannels];

    int blockSamples_ = 4800;
    double currentBlockPower_ = 0.0;
    int currentBlockSamples_ = 0;

    // Concurrency-safe components
    std::array<std::atomic<double>, 30> blockPowers_;
    std::atomic<int> blockWritePos_{0};
    int64_t totalCommittedBlocks_ = 0;

    // O(1) Histogram for Integrated Loudness
    std::array<std::atomic<uint32_t>, kNumBins> histogram_;

    // EBU Tech 3342 loudness-range histogram (short-term values, 1 s hop)
    std::array<std::atomic<uint32_t>, kNumBins> lraHistogram_;

    // ITU-R BS.1770-4 true-peak (dBTP) tracking
    TruePeakDetector<T, kMaxChannels> truePeak_;
    std::atomic<T> truePeakMax_ { T(0) };
};

} // namespace dspark