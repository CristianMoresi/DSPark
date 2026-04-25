// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DCBlocker.h
 * @brief Removes DC offset from audio signals with configurable filter order.
 *
 * Order 1 uses a lightweight 1-pole high-pass filter (2 multiplies, 2 additions
 * per sample). Orders 2–10 use cascaded Butterworth biquad high-pass stages
 * with predefined Q values for maximally-flat passband response.
 *
 * Transfer function (order 1): H(z) = (1 - z^-1) / (1 - R * z^-1)
 * where R controls the cutoff frequency (~5 Hz at R=0.995, 48 kHz).
 *
 * Higher orders provide steeper roll-off below the cutoff, useful for
 * aggressive DC removal without affecting the audible spectrum.
 *
 * Dependencies: DspMath.h, Biquad.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::DCBlocker<float> dc;
 *   dc.prepare(48000.0);
 *
 *   // Higher-order Butterworth DC blocking:
 *   dc.setOrder(6);         // 6th-order (36 dB/oct roll-off)
 *   dc.setCutoff(10.0f);    // 10 Hz cutoff
 *   dc.processBlock(buffer);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/Biquad.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class DCBlocker
 * @brief DC blocking filter with configurable Butterworth order (1–10).
 *
 * Order 1 = efficient 1-pole filter. Orders 2–10 = cascaded biquad HPFs
 * with Butterworth Q values for maximally-flat response.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DCBlocker
{
public:
    virtual ~DCBlocker() = default;

    static constexpr int kMaxBiquadStages = 5;
    static constexpr int kMaxChannels = 16;

    /**
     * @brief Prepares the DC blocker.
     *
     * @param sampleRate Sample rate in Hz.
     * @param numChannels Number of audio channels (default: 2).
     * @param cutoffHz    Cutoff frequency in Hz (default: 5 Hz).
     */
    void prepare(double sampleRate, int numChannels = 2,
                 double cutoffHz = 5.0)
    {
        sampleRate_ = sampleRate;
        numChannels_ = numChannels;
        cutoffHz_.store(static_cast<T>(cutoffHz), std::memory_order_relaxed);

        // 1-pole coefficient for order 1
        R_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0
                                     * cutoffHz / sampleRate));

        xPrev_.assign(static_cast<size_t>(numChannels), T(0));
        yPrev_.assign(static_cast<size_t>(numChannels), T(0));

        // Prepare biquad stages
        updateBiquadCoeffs(static_cast<float>(cutoffHz));
        for (auto& stage : biquadStages_)
            stage.reset();

        lastCutoff_ = static_cast<T>(cutoffHz);
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Sets the filter order (1–10).
     *
     * - Order 1: efficient 1-pole filter (6 dB/oct).
     * - Order 2–10: cascaded Butterworth biquad HPFs (12*N dB/oct for even orders).
     *   Odd orders are rounded down to the nearest even order.
     *
     * Call before prepare(), or from the audio thread.
     *
     * @param order Filter order (1–10).
     */
    void setOrder(int order) noexcept
    {
        order_ = std::clamp(order, 1, 10);
    }

    /** @brief Returns the current filter order. */
    [[nodiscard]] int getOrder() const noexcept { return order_; }

    /**
     * @brief Sets the cutoff frequency (thread-safe, picked up by next processBlock).
     * @param hz Cutoff frequency in Hz.
     */
    void setCutoff(T hz) noexcept
    {
        cutoffHz_.store(std::max(hz, T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the cutoff frequency with immediate coefficient update.
     *
     * Legacy API — updates coefficients immediately. For thread-safe
     * runtime changes, use setCutoff(T hz) instead.
     *
     * @param sampleRate Sample rate in Hz.
     * @param cutoffHz   Cutoff frequency in Hz.
     */
    void setCutoff(double sampleRate, double cutoffHz) noexcept
    {
        sampleRate_ = sampleRate;
        cutoffHz_.store(static_cast<T>(cutoffHz), std::memory_order_relaxed);
        R_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0
                                     * cutoffHz / sampleRate));
        updateBiquadCoeffs(static_cast<float>(cutoffHz));
        lastCutoff_ = static_cast<T>(cutoffHz);
    }

    /**
     * @brief Processes an AudioBufferView in-place.
     *
     * Reads the atomic cutoff value and updates coefficients before processing.
     *
     * @param buffer Audio buffer.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        T cutoff = cutoffHz_.load(std::memory_order_relaxed);

        if (order_ <= 1)
        {
            // Recompute R_ only when cutoff actually changed. exp() in the
            // steady-state path is pure waste otherwise.
            if (cutoff != lastCutoff_)
            {
                R_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0
                                             * static_cast<double>(cutoff) / sampleRate_));
                lastCutoff_ = cutoff;
            }
            const int nCh = std::min(buffer.getNumChannels(), numChannels_);
            const int nS = buffer.getNumSamples();
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    data[i] = processSample1Pole(ch, data[i]);
            }
        }
        else
        {
            // Same optimisation for the biquad cascade: only rebuild
            // coefficients when cutoff changed.
            if (cutoff != lastCutoff_)
            {
                updateBiquadCoeffs(static_cast<float>(cutoff));
                lastCutoff_ = cutoff;
            }
            const int nCh = std::min(buffer.getNumChannels(),
                                     std::min(numChannels_, kMaxChannels));
            const int nS = buffer.getNumSamples();
            int numStages = std::clamp(order_ / 2, 1, kMaxBiquadStages);

            for (int i = 0; i < nS; ++i)
            {
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T sample = buffer.getChannel(ch)[i];
                    for (int s = 0; s < numStages; ++s)
                        sample = biquadStages_[s].processSample(sample, ch);
                    buffer.getChannel(ch)[i] = sample;
                }
            }
        }
    }

    /**
     * @brief Processes a single sample for a given channel.
     *
     * Uses whichever coefficients are currently set. For order > 1,
     * call setCutoff(sr, hz) or processBlock(AudioBufferView) first
     * to ensure coefficients are up to date.
     *
     * @param channel Channel index.
     * @param input Input sample.
     * @return Sample with DC offset removed.
     */
    [[nodiscard]] T processSample(int channel, T input) noexcept
    {
        if (order_ <= 1)
            return processSample1Pole(channel, input);

        T sample = input;
        int numStages = std::clamp(order_ / 2, 1, kMaxBiquadStages);
        for (int s = 0; s < numStages; ++s)
            sample = biquadStages_[s].processSample(sample, channel);
        return sample;
    }

    /**
     * @brief Processes a block of samples for one channel in-place.
     *
     * @param channel Channel index.
     * @param data Audio samples (modified in-place).
     * @param numSamples Number of samples.
     */
    void processBlock(int channel, T* data, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = processSample(channel, data[i]);
    }

    /**
     * @brief Processes all channels from separate buffers.
     *
     * @param channelData Array of pointers to each channel's data.
     * @param numChannels Number of channels.
     * @param numSamples Number of samples per channel.
     */
    void process(T** channelData, int numChannels, int numSamples) noexcept
    {
        for (int ch = 0; ch < numChannels; ++ch)
            processBlock(ch, channelData[ch], numSamples);
    }

    /**
     * @brief Resets the filter state to zero.
     */
    void reset() noexcept
    {
        std::fill(xPrev_.begin(), xPrev_.end(), T(0));
        std::fill(yPrev_.begin(), yPrev_.end(), T(0));
        for (auto& stage : biquadStages_)
            stage.reset();
    }

protected:
    /** @brief 1-pole DC blocking filter (order 1 path). */
    [[nodiscard]] T processSample1Pole(int channel, T input) noexcept
    {
        auto ch = static_cast<size_t>(channel);
        T output = input - xPrev_[ch] + R_ * yPrev_[ch];
        xPrev_[ch] = input;
        yPrev_[ch] = output;
        return output;
    }

    /**
     * @brief Updates biquad HPF coefficients from Butterworth Q table.
     *
     * Q values from Butterworth polynomial for maximally-flat passband:
     * - Order 2: 1 stage, Q = {0.7071}
     * - Order 4: 2 stages, Q = {0.5412, 1.3066}
     * - Order 6: 3 stages, Q = {0.5177, 0.7071, 1.9319}
     * - Order 8: 4 stages, Q = {0.5098, 0.6013, 0.8999, 2.5628}
     * - Order 10: 5 stages, Q = {0.5062, 0.5612, 0.7071, 1.1013, 3.1962}
     */
    void updateBiquadCoeffs(float cutoff) noexcept
    {
        static constexpr float qTable[6][kMaxBiquadStages] = {
            {},                                                    // index 0 (unused)
            { 0.7071f },                                           // order 2
            { 0.5412f, 1.3066f },                                  // order 4
            { 0.5177f, 0.7071f, 1.9319f },                         // order 6
            { 0.5098f, 0.6013f, 0.8999f, 2.5628f },                // order 8
            { 0.5062f, 0.5612f, 0.7071f, 1.1013f, 3.1962f }       // order 10
        };

        int tableIdx = std::clamp(order_ / 2, 1, 5);
        for (int s = 0; s < tableIdx; ++s)
        {
            auto c = BiquadCoeffs<T>::makeHighPass(
                sampleRate_, static_cast<double>(cutoff),
                static_cast<double>(qTable[tableIdx][s]));
            biquadStages_[s].setCoeffs(c);
        }
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 2;
    int order_ = 1;
    T R_ = T(0.995);
    std::atomic<T> cutoffHz_ { T(5) };
    T lastCutoff_ = T(-1); ///< Cached cutoff used to skip coefficient recompute (sentinel -1 forces first update)

    // 1-pole state (order 1)
    std::vector<T> xPrev_;
    std::vector<T> yPrev_;

    // Biquad cascade state (order 2+)
    std::array<Biquad<T, kMaxChannels>, kMaxBiquadStages> biquadStages_ {};
};

} // namespace dspark
