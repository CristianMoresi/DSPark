// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file StereoWidth.h
 * @brief Stereo width control via Mid/Side processing.
 *
 * Adjusts the stereo image width from mono to extra-wide using M/S encoding.
 * Includes an optional bass-mono feature that collapses low frequencies to
 * mono below a configurable cutoff — essential for mastering and vinyl/club
 * compatibility.
 *
 * Width values:
 * - 0.0 = Mono (only mid signal)
 * - 1.0 = Original stereo image (unchanged)
 * - 2.0 = Extra wide (side signal boosted 2x)
 *
 * Dependencies: DspMath.h.
 *
 * @code
 *   dspark::StereoWidth<float> width;
 *   width.prepare(48000.0);
 *   width.setWidth(1.5f);           // wider than original
 *   width.setBassMono(true, 120.0); // mono below 120 Hz
 *
 *   // In audio callback:
 *   for (int i = 0; i < numSamples; ++i)
 *       width.processSample(left[i], right[i]);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class StereoWidth
 * @brief Stereo image width control with optional bass mono.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class StereoWidth
{
public:
    virtual ~StereoWidth() = default;
    /**
     * @brief Prepares the processor.
     * @param sampleRate Sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        updateBassMonoCoeff();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    /**
     * @brief Processes an AudioBufferView in-place (unified API, stereo).
     * @param buffer Audio buffer (must have >= 2 channels).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumChannels() >= 2)
            process(buffer.getChannel(0), buffer.getChannel(1), buffer.getNumSamples());
    }

    /**
     * @brief Sets the stereo width.
     *
     * @param width Width factor: 0=mono, 1=original, 2=extra wide.
     */
    void setWidth(T width) noexcept
    {
        width_.store(std::max(T(0), width), std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current width setting.
     */
    [[nodiscard]] T getWidth() const noexcept { return width_.load(std::memory_order_relaxed); }

    /**
     * @brief Enables or disables bass-mono and sets cutoff.
     *
     * @param enabled True to enable bass mono.
     * @param cutoffHz Frequency below which signal is collapsed to mono (default: 100 Hz).
     *
     * @note Must be called from a single thread (typically GUI). The
     *       coefficient is published with release/acquire ordering so the
     *       audio thread sees a consistent value, but two concurrent
     *       writers would race on the cutoff write.
     */
    void setBassMono(bool enabled, double cutoffHz = 100.0) noexcept
    {
        bassMonoCutoff_ = cutoffHz;
        // Write the coefficient **before** the enable flag so the audio
        // thread cannot observe `enabled=true` with a stale coefficient.
        updateBassMonoCoeff();
        bassMonoEnabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Processes one stereo sample pair in-place.
     *
     * @param left Left channel sample (modified in-place).
     * @param right Right channel sample (modified in-place).
     */
    void processSample(T& left, T& right) noexcept
    {
        // Encode to M/S
        T mid  = (left + right) * T(0.5);
        T side = (left - right) * T(0.5);

        // Apply width to side signal
        side *= width_.load(std::memory_order_relaxed);

        // Bass mono: filter the side signal to remove lows.
        // Acquire ordering pairs with the release store in setBassMono
        // so that `bassMonoCoeff_` is visible when the flag reads true.
        if (bassMonoEnabled_.load(std::memory_order_acquire))
        {
            // 1-pole highpass on side signal
            T filteredSide = side - bassMonoState_;
            bassMonoState_ += bassMonoCoeff_ * filteredSide;
            side = filteredSide;
        }

        // Decode back to L/R
        left  = mid + side;
        right = mid - side;
    }

    /**
     * @brief Processes stereo buffers in-place.
     *
     * @param left Left channel buffer.
     * @param right Right channel buffer.
     * @param numSamples Number of samples per channel.
     */
    void process(T* left, T* right, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            processSample(left[i], right[i]);
    }

    /**
     * @brief Resets internal state.
     */
    void reset() noexcept
    {
        bassMonoState_ = T(0);
    }

protected:
    void updateBassMonoCoeff() noexcept
    {
        if (sampleRate_ > 0.0)
        {
            // 1-pole LP coefficient: c = 1 - exp(-2pi * fc / fs)
            bassMonoCoeff_ = static_cast<T>(
                1.0 - std::exp(-std::numbers::pi * 2.0
                               * bassMonoCutoff_ / sampleRate_));
        }
    }

    double sampleRate_ = 48000.0;
    std::atomic<T> width_ { T(1) };

    // Bass mono
    std::atomic<bool> bassMonoEnabled_ { false };
    double bassMonoCutoff_ = 100.0;
    T bassMonoCoeff_ = T(0);
    T bassMonoState_ = T(0);
};

} // namespace dspark
