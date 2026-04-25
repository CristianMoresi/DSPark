// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Tremolo.h
 * @brief Amplitude modulation (tremolo) with configurable LFO.
 *
 * Modulates the signal amplitude using an internal LFO. Supports sine,
 * triangle, and square wave shapes. Optional stereo mode offsets the LFO
 * phase between L/R channels for an auto-pan effect.
 *
 * Dependencies: Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::Tremolo<float> tremolo;
 *   tremolo.prepare(spec);
 *   tremolo.setRate(4.0f);    // 4 Hz
 *   tremolo.setDepth(0.8f);   // 80% depth
 *
 *   // In audio callback:
 *   tremolo.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Phasor.h"
#include "../Core/Smoothers.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Tremolo
 * @brief LFO-driven amplitude modulation with stereo auto-pan option.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Tremolo
{
public:
    enum class Shape
    {
        Sine,      ///< Classic smooth tremolo
        Triangle,  ///< Sharper modulation
        Square     ///< On/off gating effect
    };

    /**
     * @brief Prepares the tremolo processor.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        T rateVal = rate_.load(std::memory_order_relaxed);
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            phasors_[ch].prepare(sampleRate_);
            phasors_[ch].setFrequency(rateVal);
        }

        // M8: depth smoother prevents zipper on automation. ~5 ms ramp is
        // short enough to track fast gestures but long enough to mask steps.
        depthSmoother_.reset(sampleRate_, kDepthRampMs,
                             static_cast<float>(depth_.load(std::memory_order_relaxed)));

        // Stereo: offset R channel by 0.5 (180 degrees)
        if (stereo_.load(std::memory_order_relaxed) && numChannels_ >= 2)
            phasors_[1].setPhase(T(0.5));
    }

    /**
     * @brief Processes audio in-place (applies tremolo modulation).
     * @param buffer Audio data to modulate.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        // M8: FTZ/DAZ on denormals for the multiplications (keeps CPU
        // from degrading when the input tail goes to near-zero).
        DenormalGuard guard;

        int numCh = std::min(buffer.getNumChannels(), numChannels_);
        int numSamples = buffer.getNumSamples();

        depthSmoother_.setTargetValue(static_cast<float>(
            depth_.load(std::memory_order_relaxed)));
        bool stereoVal = stereo_.load(std::memory_order_relaxed);
        auto shapeVal = shape_.load(std::memory_order_relaxed);

        for (int i = 0; i < numSamples; ++i)
        {
            T depthVal = static_cast<T>(depthSmoother_.getNextValue());

            // Advance phasors once per sample frame
            T phases[kMaxChannels];
            int numPhasors = stereoVal ? std::min(numCh, kMaxChannels) : 1;
            for (int p = 0; p < numPhasors; ++p)
                phases[p] = phasors_[p].advance();

            for (int ch = 0; ch < numCh; ++ch)
            {
                int phasorIdx = (stereoVal && ch < kMaxChannels) ? ch : 0;
                T mod = computeShape(phases[phasorIdx], shapeVal);
                T gain = T(1) - depthVal * (T(1) - mod) * T(0.5);
                buffer.getChannel(ch)[i] *= gain;
            }
        }
    }

    /** @brief Resets internal LFO state. */
    void reset() noexcept
    {
        for (auto& p : phasors_)
            p.reset();
    }

    /**
     * @brief Sets the LFO rate.
     * @param hz Modulation frequency (0.1 – 20 Hz typical).
     */
    void setRate(T hz) noexcept
    {
        rate_.store(hz, std::memory_order_relaxed);
        for (auto& p : phasors_)
            p.setFrequency(hz);
    }

    /**
     * @brief Sets the modulation depth.
     * @param depth 0.0 = no modulation, 1.0 = full modulation.
     */
    void setDepth(T depth) noexcept
    {
        depth_.store(std::clamp(depth, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Sets the LFO waveform shape. */
    void setShape(Shape shape) noexcept { shape_.store(shape, std::memory_order_relaxed); }

    /**
     * @brief Enables/disables stereo mode (auto-pan).
     *
     * When enabled, the right channel LFO is 180° out of phase with the left,
     * creating a panning effect.
     *
     * @param enabled True for stereo tremolo / auto-pan.
     */
    void setStereo(bool enabled) noexcept
    {
        stereo_.store(enabled, std::memory_order_relaxed);
        if (enabled && numChannels_ >= 2)
            phasors_[1].setPhase(T(0.5));
    }

    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDepth() const noexcept { return depth_.load(std::memory_order_relaxed); }
    [[nodiscard]] Shape getShape() const noexcept { return shape_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isStereo() const noexcept { return stereo_.load(std::memory_order_relaxed); }

private:
    /// Generates a shaped LFO value from phase [0, 1) to [-1, 1].
    [[nodiscard]] T computeShape(T phase, Shape shapeVal) const noexcept
    {
        switch (shapeVal)
        {
            case Shape::Sine:
                return std::sin(phase * T(2) * static_cast<T>(std::numbers::pi));

            case Shape::Triangle:
            {
                // 0→0.25: 0→1, 0.25→0.75: 1→-1, 0.75→1: -1→0
                T t = phase * T(4);
                if (t < T(1)) return t;
                if (t < T(3)) return T(2) - t;
                return t - T(4);
            }

            case Shape::Square:
                return (phase < T(0.5)) ? T(1) : T(-1);
        }

        return T(0);
    }

    static constexpr int kMaxChannels = 2;
    static constexpr float kDepthRampMs = 5.0f;

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;
    std::atomic<T> rate_ { T(4) };
    std::atomic<T> depth_ { T(0.5) };
    std::atomic<Shape> shape_ { Shape::Sine };
    std::atomic<bool> stereo_ { false };

    Smoothers::LinearSmoother depthSmoother_;
    Phasor<T> phasors_[kMaxChannels]{};
};

} // namespace dspark
