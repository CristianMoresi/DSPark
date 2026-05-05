// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Gain.h
 * @brief Smoothed gain processor with fade, mute, and click-free phase inversion.
 *
 * Applies gain to audio with exponential click-free ramping. Ensures zero artifacts
 * during parameter changes. Thread-safe design ensures control methods can be called
 * from the UI thread while processing runs lock-free in the audio thread.
 *
 * Dependencies: DspMath.h, SmoothedValue.h.
 */

#include "../Core/DspMath.h"
#include "../Core/SmoothedValue.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/SimdOps.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class Gain
 * @brief Professional click-free gain processor.
 *
 * Features:
 * - Thread-safe lock-free parameter updates.
 * - Exponential smoothing for perceptually uniform transitions.
 * - Smooth cross-zero transitions for phase inversion (no clicks).
 * - SIMD-accelerated bulk processing.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Gain
{
public:
    Gain()
    {
        gainSmooth_.reset(T(1));
    }

    ~Gain() = default;

    /**
     * @brief Prepares the gain processor for playback.
     * @param sampleRate Sample rate in Hz.
     * @param rampTimeMs Smoothing time in milliseconds (default: 10 ms).
     */
    void prepare(double sampleRate, double rampTimeMs = 10.0)
    {
        sampleRate_ = sampleRate;
        rampTimeMs_.store(rampTimeMs, std::memory_order_relaxed);
        
        gainSmooth_.prepare(sampleRate, rampTimeMs);
        forceSynchronize();
    }

    /** 
     * @brief Prepares from AudioSpec, preserving existing ramp time. 
     */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, rampTimeMs_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Sets the target gain in decibels. (Thread-safe, callable from UI).
     * @param dB Gain in dB (0 = unity).
     */
    void setGainDb(T dB) noexcept
    {
        targetGainLinear_.store(decibelsToGain(dB), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target gain as a linear multiplier. (Thread-safe, callable from UI).
     * @param linear Gain multiplier (>= 0).
     */
    void setGainLinear(T linear) noexcept
    {
        targetGainLinear_.store(std::max(T(0), linear), std::memory_order_relaxed);
    }

    /** @brief Returns current target gain in dB. */
    [[nodiscard]] T getGainDb() const noexcept
    {
        return gainToDecibels(targetGainLinear_.load(std::memory_order_relaxed));
    }

    /** @brief Returns current target linear gain. */
    [[nodiscard]] T getGainLinear() const noexcept
    {
        return targetGainLinear_.load(std::memory_order_relaxed);
    }

    /** @brief Returns the current internal smoothed value. */
    [[nodiscard]] T getCurrentGain() const noexcept
    {
        return gainSmooth_.getCurrentValue();
    }

    /**
     * @brief Enables or disables mute smoothly. (Thread-safe, callable from UI).
     */
    void setMuted(bool muted) noexcept
    {
        muted_.store(muted, std::memory_order_relaxed);
    }

    [[nodiscard]] bool isMuted() const noexcept { return muted_.load(std::memory_order_relaxed); }

    /**
     * @brief Enables smooth phase inversion. (Thread-safe, callable from UI).
     * Reverses polarity by ramping smoothly through zero, avoiding clicks.
     */
    void setInverted(bool inverted) noexcept 
    { 
        inverted_.store(inverted, std::memory_order_relaxed); 
    }

    [[nodiscard]] bool isInverted() const noexcept { return inverted_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets the smoothing ramp time. (Thread-safe).
     */
    void setRampTime(double rampTimeMs) noexcept
    {
        rampTimeMs_.store(rampTimeMs, std::memory_order_relaxed);
        rampTimeChanged_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Processes an AudioBufferView in-place. (Audio Thread only).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        updateInternalState();

        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();

        if (nCh == 1)
        {
            process(buffer.getChannel(0), nS);
        }
        else
        {
            T* channels[16];
            int useCh = nCh < 16 ? nCh : 16;
            for (int c = 0; c < useCh; ++c)
                channels[c] = buffer.getChannel(c);
            process(channels, useCh, nS);
        }
    }

    /**
     * @brief Processes an interleaved or single-channel buffer in-place.
     */
    void process(T* data, int numSamples) noexcept
    {
        updateInternalState();
        int i = 0;

        // Per-sample ramp (branchless in the hot path)
        for (; i < numSamples && gainSmooth_.isSmoothing(); ++i)
        {
            data[i] *= gainSmooth_.getNextValue();
        }

        // SIMD bulk path
        if (i < numSamples)
        {
            simd::applyGain(data + i, gainSmooth_.getCurrentValue(), numSamples - i);
        }
    }

    /**
     * @brief Processes separate channel buffers in-place.
     */
    void process(T** channelData, int numChannels, int numSamples) noexcept
    {
        updateInternalState();
        int i = 0;

        for (; i < numSamples && gainSmooth_.isSmoothing(); ++i)
        {
            const T g = gainSmooth_.getNextValue();
            for (int ch = 0; ch < numChannels; ++ch)
                channelData[ch][i] *= g;
        }

        if (i < numSamples)
        {
            const T g = gainSmooth_.getCurrentValue();
            const int remaining = numSamples - i;
            for (int ch = 0; ch < numChannels; ++ch)
                simd::applyGain(channelData[ch] + i, g, remaining);
        }
    }

    /**
     * @brief Processes input to output (not in-place).
     */
    void process(const T* input, T* output, int numSamples) noexcept
    {
        updateInternalState();
        int i = 0;

        for (; i < numSamples && gainSmooth_.isSmoothing(); ++i)
        {
            output[i] = input[i] * gainSmooth_.getNextValue();
        }

        if (i < numSamples)
        {
            const T g = gainSmooth_.getCurrentValue();
            const int remaining = numSamples - i;
            
            // Allow out-of-place SIMD by copying then applying in-place SIMD gain
            std::copy_n(input + i, remaining, output + i);
            simd::applyGain(output + i, g, remaining);
        }
    }

    /** @brief Skips smoothing — immediately sets current gain to target. */
    void skipRamp() noexcept
    {
        forceSynchronize();
    }

    /** @brief Resets gain internal state to match targets immediately. */
    void reset() noexcept
    {
        forceSynchronize();
    }

protected:
    /**
     * @brief Synchronizes atomic UI variables with audio thread state.
     * Must be called at the start of any audio processing block.
     */
    inline void updateInternalState() noexcept
    {
        if (rampTimeChanged_.load(std::memory_order_relaxed))
        {
            gainSmooth_.setRampTime(sampleRate_, rampTimeMs_.load(std::memory_order_relaxed));
            rampTimeChanged_.store(false, std::memory_order_relaxed);
        }

        const T baseTarget = muted_.load(std::memory_order_relaxed) ? T(0) 
                             : targetGainLinear_.load(std::memory_order_relaxed);
        
        const T finalTarget = inverted_.load(std::memory_order_relaxed) ? -baseTarget : baseTarget;

        if (finalTarget != currentTarget_)
        {
            gainSmooth_.setTargetValue(finalTarget);
            currentTarget_ = finalTarget;
        }
    }

    /** @brief Forces instantaneous synchronization of state without ramping. */
    void forceSynchronize() noexcept
    {
        rampTimeChanged_.store(false, std::memory_order_relaxed);
        
        const T baseTarget = muted_.load(std::memory_order_relaxed) ? T(0) 
                             : targetGainLinear_.load(std::memory_order_relaxed);
        const T finalTarget = inverted_.load(std::memory_order_relaxed) ? -baseTarget : baseTarget;
        
        currentTarget_ = finalTarget;
        gainSmooth_.reset(currentTarget_);
    }

    double sampleRate_ = 48000.0;
    
    // Audio Thread State
    T currentTarget_ { T(1) };
    SmoothedValue<T> gainSmooth_;

    // Atomic Communication (UI -> Audio Thread)
    std::atomic<T> targetGainLinear_ { T(1) };
    std::atomic<double> rampTimeMs_ { 10.0 };
    std::atomic<bool> rampTimeChanged_ { false };
    std::atomic<bool> muted_ { false };
    std::atomic<bool> inverted_ { false };
};

} // namespace dspark
