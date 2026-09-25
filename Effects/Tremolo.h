// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Tremolo.h
 * @brief Amplitude modulation (tremolo) with configurable LFO and analog-style shaping.
 *
 * Implements a highly optimized amplitude modulator. Features zero-allocation
 * processing, thread-safe parameter handling, and click-free analog-style
 * waveforms (smoothed square wave). Shape and stereo-mode changes crossfade
 * the modulation over 5 ms, and depth changes ramp over 5 ms, so no setting
 * steps the gain.
 * Optional stereo mode creates a 180-degree out-of-phase LFO on the right channel
 * for wide auto-pan effects.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h, Smoothers.h,
 *               DenormalGuard.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Phasor.h"
#include "../Core/Smoothers.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class Tremolo
 * @brief LFO-driven amplitude modulation with stereo auto-pan option.
 *
 * Designed with a template-dispatch architecture to ensure zero branching
 * inside the inner audio loops, maximizing L1 cache hits and SIMD autovectorization.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Tremolo
{
public:
    enum class Shape
    {
        Sine,     ///< Classic smooth tremolo (opto-isolator style)
        Triangle, ///< Linear modulation, sharper peaks
        Square    ///< Slew-rate limited gating effect (analog-style, click-free)
    };

    /**
     * @brief Prepares the tremolo processor and allocates internal states.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification (sample rate and max channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        T initialRate = rate_.load(std::memory_order_relaxed);
        currentRate_ = initialRate;
        lfo_.prepare(sampleRate_);
        lfo_.setFrequency(initialRate);

        depthSmoother_.reset(sampleRate_, kDepthRampMs,
                             static_cast<float>(depth_.load(std::memory_order_relaxed)));

        // Start settled on the current shape and stereo mode (no fade).
        activeShape_ = shape_.load(std::memory_order_relaxed);
        offsetR_ = stereo_.load(std::memory_order_relaxed) ? T(0.5) : T(0);
        fadeLength_ = std::max(1, static_cast<int>(sampleRate_ * (kFadeMs / 1000.0)));
        fadeRemaining_ = 0;
    }

    /**
     * @brief Processes an audio block in-place with SIMD-friendly dispatch.
     * @param buffer Audio buffer view to modulate.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();

        if (numCh == 0 || numSamples == 0)
            return;

        DenormalGuard guard;

        // Thread-safe parameter polling (Audio Thread owns the state)
        updateInternalState();

        // A shape or stereo-mode change crossfades from the old modulation
        // to the new one; both used to switch at once, stepping the gain
        // (up to the full depth when the right LFO jumped half a cycle).
        int start = 0;
        if (fadeRemaining_ > 0)
        {
            start = std::min(numSamples, fadeRemaining_);
            processFade(buffer, numCh, start);
        }
        if (start == numSamples) return;

        // Template dispatching eliminates branching inside the hot path
        switch (activeShape_)
        {
            case Shape::Sine:     processBlockShape<Shape::Sine>(buffer, numCh, start, numSamples); break;
            case Shape::Triangle: processBlockShape<Shape::Triangle>(buffer, numCh, start, numSamples); break;
            case Shape::Square:   processBlockShape<Shape::Square>(buffer, numCh, start, numSamples); break;
        }
    }

    /**
     * @brief Hard-resets the internal LFO phase to zero.
     * @note Will cause an audible click if called while audio is actively passing.
     */
    void reset() noexcept
    {
        lfo_.reset();
        fadeRemaining_ = 0;
    }

    /**
     * @brief Sets the LFO rate. Thread-safe (can be called from UI thread).
     * @param hz Modulation frequency in Hz (floored to 0; audio-rate AM is
     *           legitimate). Non-finite values are ignored.
     */
    void setRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        rate_.store(std::max(T(0), hz), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the modulation depth. Thread-safe.
     * @param depth Range [0.0 (bypass), 1.0 (full amplitude cut)].
     *              Non-finite values are ignored.
     */
    void setDepth(T depth) noexcept
    {
        if (!std::isfinite(depth)) return;
        depth_.store(std::clamp(depth, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the LFO waveform shape. Thread-safe.
     * @param shape Waveform type (Sine, Triangle, Square; wild enum values clamp).
     */
    void setShape(Shape shape) noexcept
    {
        const int s = std::clamp(static_cast<int>(shape), 0,
                                 static_cast<int>(Shape::Square));
        shape_.store(static_cast<Shape>(s), std::memory_order_relaxed);
    }

    /**
     * @brief Enables auto-pan by offsetting the right channel LFO phase by 180 degrees.
     * @param enabled True for stereo mode, false for synchronized mono modulation.
     */
    void setStereo(bool enabled) noexcept { stereo_.store(enabled, std::memory_order_relaxed); }

    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDepth() const noexcept { return depth_.load(std::memory_order_relaxed); }
    [[nodiscard]] Shape getShape() const noexcept { return shape_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isStereo() const noexcept { return stereo_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TREM"), 1);
        w.write("rate", rate_.load(std::memory_order_relaxed));
        w.write("depth", depth_.load(std::memory_order_relaxed));
        w.write("shape", static_cast<int32_t>(shape_.load(std::memory_order_relaxed)));
        w.write("stereo", stereo_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TREM")) return false;
        setRate(static_cast<T>(r.read("rate", 4.0f)));
        setDepth(static_cast<T>(r.read("depth", 0.5f)));
        setShape(static_cast<Shape>(r.read("shape", 0)));
        setStereo(r.read("stereo", false));
        return true;
    }

private:
    static constexpr float kDepthRampMs = 5.0f;
    static constexpr double kFadeMs = 5.0;   ///< Shape / stereo-mode crossfade.
    static constexpr T kSquareSlewTime = T(0.02); ///< 2% phase transition to prevent clicks

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;

    std::atomic<T> rate_{ T(4) };
    std::atomic<T> depth_{ T(0.5) };
    std::atomic<Shape> shape_{ Shape::Sine };
    std::atomic<bool> stereo_{ false };

    // Internal state owned by Audio Thread
    T currentRate_ = T(4);
    Shape activeShape_ = Shape::Sine;
    T offsetR_ = T(0);                  ///< Right LFO phase offset: 0.5 in stereo mode.
    Shape fadeShape_ = Shape::Sine;     ///< Shape being faded out.
    T fadeOffsetR_ = T(0);              ///< Right offset being faded out.
    int fadeLength_ = 1;
    int fadeRemaining_ = 0;

    Smoothers::LinearSmoother depthSmoother_;
    Phasor<T> lfo_;   ///< The right channel reads it at offsetR_.

    /**
     * @brief Polls atomics and updates local phasor state safely.
     */
    inline void updateInternalState() noexcept
    {
        T targetRate = rate_.load(std::memory_order_relaxed);
        if (targetRate != currentRate_)
        {
            currentRate_ = targetRate;
            lfo_.setFrequency(currentRate_);
        }

        const Shape shape = shape_.load(std::memory_order_relaxed);
        const T offset = stereo_.load(std::memory_order_relaxed) ? T(0.5) : T(0);
        if (shape != activeShape_ || offset != offsetR_)
        {
            // A change during a fade restarts it from the latest settled law.
            fadeShape_ = activeShape_;
            fadeOffsetR_ = offsetR_;
            activeShape_ = shape;
            offsetR_ = offset;
            fadeRemaining_ = fadeLength_;
        }

        depthSmoother_.setTargetValue(static_cast<float>(depth_.load(std::memory_order_relaxed)));
    }

    /** @brief Wraps a phase in [0, 2) back into [0, 1). */
    [[nodiscard]] static inline T wrapPhase(T phase) noexcept
    {
        return phase >= T(1) ? phase - T(1) : phase;
    }

    /** @brief Runtime-dispatched shape (used only while a fade runs). */
    [[nodiscard]] inline T evalShape(Shape shape, T phase) const noexcept
    {
        switch (shape)
        {
            case Shape::Triangle: return computeShape<Shape::Triangle>(phase);
            case Shape::Square:   return computeShape<Shape::Square>(phase);
            case Shape::Sine:
            default:              return computeShape<Shape::Sine>(phase);
        }
    }

    /**
     * @brief Processes the first n samples of the block while a shape or
     *        stereo-mode crossfade runs (the modulation blends linearly,
     *        which blends the gains linearly).
     */
    void processFade(AudioBufferView<T>& buffer, int numCh, int n) noexcept
    {
        T* const channelL = buffer.getChannel(0);
        T* const channelR = (numCh > 1) ? buffer.getChannel(1) : nullptr;
        const T invLength = T(1) / static_cast<T>(fadeLength_);

        for (int i = 0; i < n; ++i)
        {
            const T depthVal = static_cast<T>(depthSmoother_.getNextValue());
            const T phaseL = lfo_.advance();
            const T w = static_cast<T>(fadeLength_ - fadeRemaining_ + 1) * invLength;
            --fadeRemaining_;

            const T oldL = evalShape(fadeShape_, phaseL);
            const T modL = oldL + w * (evalShape(activeShape_, phaseL) - oldL);
            const T gainL = T(1) - depthVal * (T(1) - modL) * T(0.5);
            channelL[i] *= gainL;

            if (channelR != nullptr)
            {
                const T oldR = evalShape(fadeShape_, wrapPhase(phaseL + fadeOffsetR_));
                const T modR = oldR + w * (evalShape(activeShape_, wrapPhase(phaseL + offsetR_)) - oldR);
                channelR[i] *= T(1) - depthVal * (T(1) - modR) * T(0.5);
            }

            for (int ch = 2; ch < numCh; ++ch)
                buffer.getChannel(ch)[i] *= gainL;
        }
    }

    /**
     * @brief Computes a specific LFO shape. Template forces inlining and avoids branching.
     */
    template <Shape S>
    [[nodiscard]] inline T computeShape(T phase) const noexcept
    {
        if constexpr (S == Shape::Sine)
        {
            // fastSin: > 100 dB accurate - far beyond audibility for an LFO.
            return fastSin(phase * twoPi<T>);
        }
        else if constexpr (S == Shape::Triangle)
        {
            T t = phase * T(4);
            if (t < T(1)) return t;
            if (t < T(3)) return T(2) - t;
            return t - T(4);
        }
        else if constexpr (S == Shape::Square)
        {
            // Trapezoidal anti-aliased square wave (click-free)
            if (phase < kSquareSlewTime) return T(-1) + (phase / kSquareSlewTime) * T(2);
            if (phase < T(0.5)) return T(1);
            if (phase < T(0.5) + kSquareSlewTime) return T(1) - ((phase - T(0.5)) / kSquareSlewTime) * T(2);
            return T(-1);
        }
    }

    /**
     * @brief Processes the block for a specific waveform shape.
     */
    template <Shape S>
    void processBlockShape(AudioBufferView<T>& buffer, int numCh, int start, int end) noexcept
    {
        T* const channelL = buffer.getChannel(0);
        T* const channelR = (numCh > 1) ? buffer.getChannel(1) : nullptr;
        const bool stereo = offsetR_ != T(0);

        for (int i = start; i < end; ++i)
        {
            T depthVal = static_cast<T>(depthSmoother_.getNextValue());
            T phaseL = lfo_.advance();

            T modL = computeShape<S>(phaseL);
            T gainL = T(1) - depthVal * (T(1) - modL) * T(0.5);

            channelL[i] *= gainL;

            if (channelR != nullptr)
            {
                if (stereo)
                {
                    T modR = computeShape<S>(wrapPhase(phaseL + offsetR_));
                    T gainR = T(1) - depthVal * (T(1) - modR) * T(0.5);
                    channelR[i] *= gainR;
                }
                else
                {
                    channelR[i] *= gainL;   // mono modulation: same gain
                }
            }

            // Fallback for multi-channel beyond Stereo (surround routed as mono modulation)
            for (int ch = 2; ch < numCh; ++ch)
            {
                buffer.getChannel(ch)[i] *= gainL;
            }
        }
    }
};

} // namespace dspark
