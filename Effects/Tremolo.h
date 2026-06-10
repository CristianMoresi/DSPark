// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Tremolo.h
 * @brief Amplitude modulation (tremolo) with configurable LFO and analog-style shaping.
 *
 * Implements a highly optimized, SIMD-friendly amplitude modulator. Features
 * zero-allocation processing, thread-safe parameter handling, and click-free
 * analog-style waveforms (smoothed square wave). 
 * Optional stereo mode creates a 180-degree out-of-phase LFO on the right channel 
 * for wide auto-pan effects.
 *
 * Dependencies: Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h, Smoothers.h
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
#include <numbers>

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
     * @param spec Audio environment specification (sample rate and max channels).
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        T initialRate = rate_.load(std::memory_order_relaxed);
        currentRate_ = initialRate;

        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            phasors_[ch].prepare(sampleRate_);
            phasors_[ch].setFrequency(initialRate);
        }

        depthSmoother_.reset(sampleRate_, kDepthRampMs,
                             static_cast<float>(depth_.load(std::memory_order_relaxed)));

        isStereoActive_ = stereo_.load(std::memory_order_relaxed);
        if (isStereoActive_ && numChannels_ >= 2)
            phasors_[1].setPhase(T(0.5));
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

        Shape currentShape = shape_.load(std::memory_order_relaxed);

        // Template dispatching eliminates branching inside the hot path
        switch (currentShape)
        {
            case Shape::Sine:     processBlockShape<Shape::Sine>(buffer, numCh, numSamples); break;
            case Shape::Triangle: processBlockShape<Shape::Triangle>(buffer, numCh, numSamples); break;
            case Shape::Square:   processBlockShape<Shape::Square>(buffer, numCh, numSamples); break;
        }
    }

    /**
     * @brief Hard-resets the internal LFO phase to zero.
     * @note Will cause an audible click if called while audio is actively passing.
     */
    void reset() noexcept
    {
        phasors_[0].reset();
        phasors_[1].reset();
        if (isStereoActive_)
            phasors_[1].setPhase(T(0.5));
    }

    /**
     * @brief Sets the LFO rate. Thread-safe (can be called from UI thread).
     * @param hz Modulation frequency in Hz.
     */
    void setRate(T hz) noexcept { rate_.store(hz, std::memory_order_relaxed); }

    /**
     * @brief Sets the modulation depth. Thread-safe.
     * @param depth Range [0.0 (bypass), 1.0 (full amplitude cut)].
     */
    void setDepth(T depth) noexcept { depth_.store(std::clamp(depth, T(0), T(1)), std::memory_order_relaxed); }

    /**
     * @brief Sets the LFO waveform shape. Thread-safe.
     * @param shape Waveform type (Sine, Triangle, Square).
     */
    void setShape(Shape shape) noexcept { shape_.store(shape, std::memory_order_relaxed); }

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
    static constexpr int kMaxChannels = 2;
    static constexpr float kDepthRampMs = 5.0f;
    static constexpr T kSquareSlewTime = T(0.02); ///< 2% phase transition to prevent clicks

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;

    std::atomic<T> rate_{ T(4) };
    std::atomic<T> depth_{ T(0.5) };
    std::atomic<Shape> shape_{ Shape::Sine };
    std::atomic<bool> stereo_{ false };

    // Internal state owned by Audio Thread
    T currentRate_ = T(4);
    bool isStereoActive_ = false;

    Smoothers::LinearSmoother depthSmoother_;
    Phasor<T> phasors_[kMaxChannels]{};

    /** 
     * @brief Polls atomics and updates local phasor state safely. 
     */
    inline void updateInternalState() noexcept
    {
        T targetRate = rate_.load(std::memory_order_relaxed);
        if (targetRate != currentRate_)
        {
            currentRate_ = targetRate;
            phasors_[0].setFrequency(currentRate_);
            phasors_[1].setFrequency(currentRate_);
        }

        bool targetStereo = stereo_.load(std::memory_order_relaxed);
        if (targetStereo != isStereoActive_)
        {
            isStereoActive_ = targetStereo;
            if (isStereoActive_) {
                // Wrap phase to keep synchronization logic intact
                T newPhase = std::fmod(phasors_[0].getPhase() + T(0.5), T(1));
                phasors_[1].setPhase(newPhase);
            } else {
                phasors_[1].setPhase(phasors_[0].getPhase());
            }
        }

        depthSmoother_.setTargetValue(static_cast<float>(depth_.load(std::memory_order_relaxed)));
    }

    /**
     * @brief Computes a specific LFO shape. Template forces inlining and avoids branching.
     */
    template <Shape S>
    [[nodiscard]] inline T computeShape(T phase) const noexcept
    {
        if constexpr (S == Shape::Sine)
        {
            // fastSin: > 100 dB accurate — far beyond audibility for an LFO.
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
    void processBlockShape(AudioBufferView<T>& buffer, int numCh, int numSamples) noexcept
    {
        T* const channelL = buffer.getChannel(0);
        T* const channelR = (numCh > 1) ? buffer.getChannel(1) : nullptr;

        for (int i = 0; i < numSamples; ++i)
        {
            T depthVal = static_cast<T>(depthSmoother_.getNextValue());
            T phaseL = phasors_[0].advance();
            
            T modL = computeShape<S>(phaseL);
            T gainL = T(1) - depthVal * (T(1) - modL) * T(0.5);

            channelL[i] *= gainL;

            if (channelR != nullptr)
            {
                if (isStereoActive_)
                {
                    T phaseR = phasors_[1].advance();
                    T modR = computeShape<S>(phaseR);
                    T gainR = T(1) - depthVal * (T(1) - modR) * T(0.5);
                    channelR[i] *= gainR;
                }
                else
                {
                    // If not stereo, Phasor 1 is kept in sync but we reuse GainL to save CPU
                    (void)phasors_[1].advance();
                    channelR[i] *= gainL;
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
