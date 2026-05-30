// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DryWetMixer.h
 * @brief Real-time safe dry/wet mixer with parameter smoothing and mix laws.
 *
 * Captures a copy of the dry (unprocessed) signal before an effect runs,
 * then blends it with the wet (processed) signal. Includes block-level 
 * parameter smoothing to prevent zipper noise during DAW automation, and 
 * supports both Linear and Equal Power mixing laws.
 *
 * @note If your effect introduces algorithmic latency (e.g., FIR filters), 
 * ensure the dry signal is delayed (Latency Compensation) before calling pushDry() 
 * to prevent phase cancellation (comb filtering).
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h.
 *
 * @code
 * dspark::DryWetMixer<float> mixer;
 *
 * void prepare(const dspark::AudioSpec& spec) {
 * mixer.prepare(spec);
 * mixer.setMixRule(dspark::DryWetMixer<float>::MixRule::EqualPower);
 * }
 *
 * void process(dspark::AudioBufferView<float> buffer) {
 * mixer.pushDry(buffer);
 * applyEffect(buffer);       // Modifies buffer in-place (now wet)
 * mixer.mixWet(buffer, 0.5f); // Smoothly transitions to 50% wet
 * }
 * @endcode
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"

#include <algorithm>
#include <cmath>

// DSPARK_RESTRICT is normally provided by SimdOps.h (pulled in via AudioBuffer.h).
// Guard against redefinition so this header stays self-contained if included alone.
#ifndef DSPARK_RESTRICT
  #if defined(__clang__) || defined(__GNUC__)
    #define DSPARK_RESTRICT __restrict__
  #elif defined(_MSC_VER)
    #define DSPARK_RESTRICT __restrict
  #else
    #define DSPARK_RESTRICT
  #endif
#endif

namespace dspark {

/**
 * @class DryWetMixer
 * @brief Pre-allocated, SIMD-friendly dry/wet blender for real-time audio.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels (compile-time bound).
 */
template <typename T, int MaxChannels = 16>
class DryWetMixer
{
public:
    /** @brief Defines the mathematical curve used for mixing. */
    enum class MixRule {
        /** Linear crossfade. Best for highly correlated signals (EQ, saturation). */
        Linear,
        /** Constant power crossfade. Best for uncorrelated signals (Reverb, Delay). */
        EqualPower 
    };

    /**
     * @brief Allocates the internal dry buffer for the given audio spec.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        dryBuffer_.resize(spec.numChannels, spec.maxBlockSize);
        reset();
    }

    /** @brief Resets the internal buffer and smoothing states to zero. */
    void reset() noexcept
    {
        dryBuffer_.clear();
        if (delayHist_.getNumSamples() > 0) delayHist_.clear();
        histPos_ = 0;
        capturedSamples_ = 0;  // No valid dry snapshot until the next pushDry().
        currentMix_ = T(-1); // Forces immediate jump on first use
    }

    /**
     * @brief Sets the mathematical curve to use during the mix phase.
     * @param rule The chosen MixRule (Linear or EqualPower).
     */
    void setMixRule(MixRule rule) noexcept
    {
        mixRule_ = rule;
    }

    /**
     * @brief Delays the captured dry signal to compensate for an effect's
     *        internal latency (e.g. an oversampler's FIR group delay).
     *
     * When the wet path introduces @p samples of delay, the dry copy must be
     * delayed by the same amount; otherwise blending dry+wet (mix < 1, Delta or
     * adaptive modes) produces comb filtering. Pass 0 to disable (default).
     *
     * @note **NOT Real-Time Safe.** Allocates the internal delay history.
     *       Call from the setup thread (typically right after prepare()).
     * @param samples Latency in samples at this mixer's sample rate (>= 0).
     */
    void setLatencyCompensation(int samples)
    {
        samples = std::max(0, samples);
        if (samples == latencySamples_) return;
        latencySamples_ = samples;
        delayHist_.resize(dryBuffer_.getNumChannels(), samples);
        histPos_ = 0;
    }

    /** @brief Returns the configured dry-path latency compensation in samples. */
    [[nodiscard]] int getLatencyCompensation() const noexcept { return latencySamples_; }

    /**
     * @brief Captures a snapshot of the dry (unprocessed) signal.
     *
     * @param input The unprocessed audio buffer. Accepts const or mutable views 
     * implicitly, provided the View template converts appropriately.
     */
    void pushDry(const AudioBufferView<const T>& input) noexcept
    {
        if (dryBuffer_.getNumSamples() == 0) return; // Prevent ops if unprepared

        const int chCount  = std::min(input.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(input.getNumSamples(), dryBuffer_.getNumSamples());

        if (latencySamples_ <= 0)
        {
            for (int ch = 0; ch < chCount; ++ch)
            {
                const T* DSPARK_RESTRICT src = input.getChannel(ch);
                T* DSPARK_RESTRICT dst       = dryBuffer_.getChannel(ch);
                std::copy_n(src, nSamples, dst);
            }
        }
        else
        {
            // Latency-compensated capture: route the dry through a per-channel
            // circular delay of exactly latencySamples_ so it stays time-aligned
            // with a wet path that incurs the same delay.
            const int D = latencySamples_;
            for (int ch = 0; ch < chCount; ++ch)
            {
                const T* DSPARK_RESTRICT src = input.getChannel(ch);
                T* DSPARK_RESTRICT dst       = dryBuffer_.getChannel(ch);
                T* DSPARK_RESTRICT hist      = delayHist_.getChannel(ch);
                int pos = histPos_;
                for (int i = 0; i < nSamples; ++i)
                {
                    dst[i]    = hist[pos];   // sample written D steps ago
                    hist[pos] = src[i];
                    if (++pos == D) pos = 0;
                }
            }
            histPos_ = (histPos_ + nSamples) % D;
        }

        capturedSamples_ = nSamples;
    }

    /**
     * @brief Blends the stored dry signal with the current (wet) buffer in-place.
     *
     * Automatically applies sample-accurate linear interpolation to the mix 
     * proportion to prevent zipper noise when the parameter changes.
     *
     * @param wetBuffer     The processed buffer (modified in-place).
     * @param targetMix     Target mix amount: 0.0 = fully dry, 1.0 = fully wet.
     */
    void mixWet(AudioBufferView<T> wetBuffer, T targetMix) noexcept
    {
        // Handle invalid floating point inputs (NaN) safely
        if (std::isnan(targetMix)) targetMix = T(0);
        targetMix = std::clamp(targetMix, T(0), T(1));

        const int chCount  = std::min(wetBuffer.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(wetBuffer.getNumSamples(), capturedSamples_);

        if (nSamples == 0 || chCount == 0) return;

        // Initialize smoothing state on first run
        if (currentMix_ < T(0)) currentMix_ = targetMix;

        const bool needsSmoothing = std::abs(currentMix_ - targetMix) > T(1e-5);
        const T mixStep = needsSmoothing ? (targetMix - currentMix_) / T(nSamples) : T(0);

        for (int ch = 0; ch < chCount; ++ch)
        {
            T* DSPARK_RESTRICT wetData       = wetBuffer.getChannel(ch);
            const T* DSPARK_RESTRICT dryData = dryBuffer_.getChannel(ch);
            
            // Local copy of mix value so every channel starts at the same smoothed origin
            T mixVal = currentMix_; 

            if (mixRule_ == MixRule::EqualPower)
            {
                for (int i = 0; i < nSamples; ++i)
                {
                    // Fast constant-power approximation using square root
                    const T w = std::sqrt(mixVal);
                    const T d = std::sqrt(T(1) - mixVal);
                    wetData[i] = dryData[i] * d + wetData[i] * w;
                    
                    if (needsSmoothing) mixVal += mixStep;
                }
            }
            else // MixRule::Linear
            {
                for (int i = 0; i < nSamples; ++i)
                {
                    const T w = mixVal;
                    const T d = T(1) - w;
                    wetData[i] = dryData[i] * d + wetData[i] * w;
                    
                    if (needsSmoothing) mixVal += mixStep;
                }
            }
        }

        // Update the persistent state after the block is processed
        if (needsSmoothing) currentMix_ = targetMix;
    }

    /**
     * @brief Retrieves a read-only pointer to the captured dry channel data.
     * @param ch Channel index (0-based).
     * @return Pointer to dry samples, or nullptr if out of bounds.
     */
    [[nodiscard]] const T* getDryChannel(int ch) const noexcept
    {
        if (ch < 0 || ch >= dryBuffer_.getNumChannels()) return nullptr;
        return dryBuffer_.getChannel(ch);
    }

    /** @brief Returns the internal capacity of channels in the dry buffer. */
    [[nodiscard]] int getDryNumChannels() const noexcept { return dryBuffer_.getNumChannels(); }

    /** @brief Returns the number of samples valid from the last pushDry() call. */
    [[nodiscard]] int getDryCapturedSamples() const noexcept { return capturedSamples_; }

private:
    AudioBuffer<T, MaxChannels> dryBuffer_;
    int capturedSamples_ = 0;

    // Optional dry-path latency compensation (circular delay). Disabled (0) by default.
    AudioBuffer<T, MaxChannels> delayHist_;
    int latencySamples_ = 0;
    int histPos_        = 0;

    MixRule mixRule_ = MixRule::Linear;
    T currentMix_    = T(-1); // Internal state for parameter smoothing
};

} // namespace dspark