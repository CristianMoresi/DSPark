// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Clipper.h
 * @brief Multi-mode clipping processor with oversampling, multi-stage architecture, and time-aware slew limiting.
 *
 * This class provides a high-performance, SIMD-friendly implementation of audio clipping.
 * It is designed for zero-allocation in the audio thread and is entirely lock-free.
 *
 * Four clipping modes are supported:
 * - **Hard**: Digital clamp. Transparent until the ceiling is hit, then applies a brickwall limit.
 * - **Soft**: Hyperbolic tangent (tanh). Smooth mathematical saturation with no hard discontinuities.
 * - **Analog**: Sine-based waveshaping. Models the soft clipping characteristics of magnetic/transformer saturation.
 * - **GoldenRatio**: True soft-knee using golden ratio thresholds. Operates linearly up to `ceiling/phi`, 
 *   then transitions using a smooth rational asymptotic curve to the ceiling.
 *
 * @note Thread Safety: `prepare()` allocates memory (if oversampling is enabled) and MUST NOT be called 
 * concurrently with `processBlock()`. Parameter setters (`setMode`, `setInputGain`, etc.) use `std::atomic` 
 * and are perfectly safe to call from the GUI thread during playback.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h, DryWetMixer.h, Oversampling.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/DryWetMixer.h"
#include "../Core/Oversampling.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <memory>
#include <numbers>

namespace dspark {

/**
 * @class Clipper
 * @brief Real-time audio clipper with analog modeling and anti-aliasing features.
 *
 * @tparam T Sample type (must satisfy dspark::FloatType, typically float or double).
 */
template <FloatType T>
class Clipper
{
public:
    /** @brief Defines the harmonic waveshaping algorithm used for clipping. */
    enum class Mode
    {
        Hard,         ///< Brickwall digital clipping. High odd harmonics.
        Soft,         ///< Tanh soft clipping. Even/odd blend, tape-like.
        Analog,       ///< Sine-based soft clipping. Transformer-like saturation.
        GoldenRatio   ///< Mathematical soft-knee using phi. Extremely transparent until heavy drive.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the clipper for processing, allocating any necessary internal buffers.
     *
     * If oversampling is configured > 1x, this method allocates the oversampling filters.
     * Call it from the main/setup thread before playback begins (never concurrently
     * with processBlock), and again whenever the sample rate changes.
     *
     * @param spec The current audio environment specifications (sample rate, max block size).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        int osFactor = osFactor_.load(std::memory_order_relaxed);
        
        if (osFactor > 1) {
            oversampler_ = std::make_unique<Oversampling<T>>(
                osFactor, Oversampling<T>::Quality::High);
            oversampler_->prepare(spec);
        } else {
            oversampler_.reset();
        }

        // Align the dry copy with the latent (oversampled) wet path so that a
        // dry/wet blend below 100% does not comb-filter.
        mixer_.setLatencyCompensation(oversampler_ ? oversampler_->getLatency() : 0);

        for (int ch = 0; ch < kMaxChannels; ++ch)
            slewPrev_[ch] = T(0);
    }

    /**
     * @brief Resets the internal state of the clipper.
     * 
     * Clears history buffers, slew limiter states, and oversampling memory.
     * Useful when stopping/starting transport or flushing tails.
     */
    void reset() noexcept
    {
        mixer_.reset();
        if (oversampler_) oversampler_->reset();
        for (int ch = 0; ch < kMaxChannels; ++ch)
            slewPrev_[ch] = T(0);
        gainReductionDb_.store(T(0), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------

    /**
     * @brief Processes an audio buffer in-place through the clipping algorithm.
     * 
     * Handles dry/wet mixing, upsampling, processing, and downsampling transparently.
     *
     * @param buffer View of the audio data to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) return;

        T mixVal = mix_.load(std::memory_order_relaxed);
        mixer_.pushDry(buffer);

        if (oversampler_ && oversampler_->getFactor() > 1)
        {
            auto upView = oversampler_->upsample(buffer);
            processInternal(upView, spec_.sampleRate * oversampler_->getFactor());
            oversampler_->downsample(buffer);
        }
        else
        {
            processInternal(buffer, spec_.sampleRate);
        }

        mixer_.mixWet(buffer, mixVal);
    }

    // -- Parameters (Thread-Safe Setters/Getters) -------------------------------

    /**
     * @brief Sets the clipping algorithm.
     * @param mode The desired waveshaping Mode.
     */
    void setMode(Mode mode) noexcept 
    { 
        mode_.store(mode, std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the absolute maximum output level.
     * @param dB Ceiling level in decibels Full Scale (dBFS). Clamped between -60 and 0.
     */
    void setCeiling(T dB) noexcept 
    { 
        ceilingDb_.store(std::clamp(dB, T(-60), T(0)), std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the input drive/gain before clipping.
     * @param dB Gain applied to the input signal in dB. Clamped between 0 and 48.
     */
    void setInputGain(T dB) noexcept 
    { 
        inputGainDb_.store(std::clamp(dB, T(0), T(48)), std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the number of cascaded clipping stages.
     * 
     * Multi-stage clipping distributes the input gain logarithmically across multiple
     * clipping algorithms. This alters the harmonic profile, making heavy distortion
     * sound smoother compared to a single aggressive stage.
     * 
     * @param count Number of stages. Clamped between 1 and 4.
     */
    void setStages(int count) noexcept 
    { 
        stages_.store(std::clamp(count, 1, kMaxStages), std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the dry/wet ratio of the processor.
     * @param amount Linear ratio where 0.0 is entirely dry and 1.0 is fully processed.
     */
    void setMix(T amount) noexcept 
    { 
        mix_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed); 
    }
    
    /**
     * @brief Enables slew limiting to reduce aliasing at clipping discontinuities.
     * 
     * Limits how fast the signal can change over time. By defining this in milliseconds,
     * the effect remains consistent regardless of the sample rate or oversampling factor.
     * 
     * @param ms Maximum allowed rise time in milliseconds to reach full scale. 0 = off.
     */
    void setSlewLimit(T ms) noexcept 
    { 
        slewLimitMs_.store(std::max(ms, T(0)), std::memory_order_relaxed); 
    }
    
    /**
     * @brief Sets the oversampling multiplier to mitigate aliasing.
     * 
     * Values will be snapped to the nearest power of two. 
     * @note Changing this requires a subsequent call to `prepare()` to allocate filters.
     * 
     * @param factor Oversampling ratio (1 = off, 2, 4, 8, 16).
     */
    void setOversampling(int factor) noexcept
    {
        factor = std::bit_ceil(static_cast<unsigned int>(std::max(1, factor)));
        osFactor_.store(std::min(factor, 16), std::memory_order_relaxed);
    }

    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getCeiling() const noexcept { return ceilingDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getInputGain() const noexcept { return inputGainDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getStages() const noexcept { return stages_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getOversampling() const noexcept { return osFactor_.load(std::memory_order_relaxed); }
    
    /** @brief Returns latency in samples introduced by oversampling filters. */
    [[nodiscard]] int getLatency() const noexcept { return oversampler_ ? oversampler_->getLatency() : 0; }

    /** @brief Retrieves the maximum gain reduction applied during the last block (for UI metering). */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReductionDb_.load(std::memory_order_relaxed); }

protected:
    static constexpr int kMaxStages = 4;
    static constexpr int kMaxChannels = 16;
    
    /// Mathematical Golden Ratio used for the GoldenRatio soft-knee transition.
    static constexpr T kPhi = static_cast<T>(1.6180339887498948482);

    /**
     * @brief Core DSP routing. Resolves atomics and branches to the SIMD-friendly template.
     */
    void processInternal(AudioBufferView<T>& buffer, double currentSampleRate) noexcept
    {
        Mode modeVal   = mode_.load(std::memory_order_relaxed);
        T ceilDb       = ceilingDb_.load(std::memory_order_relaxed);
        T gainDb       = inputGainDb_.load(std::memory_order_relaxed);
        int numStages  = stages_.load(std::memory_order_relaxed);
        T slewMs       = slewLimitMs_.load(std::memory_order_relaxed);

        T ceiling      = dbToLinear(ceilDb);
        T totalGainLin = dbToLinear(gainDb);
        
        // Split gain logarithmically across cascaded stages
        T stageGain    = (numStages > 1) 
                           ? std::pow(totalGainLin, T(1) / static_cast<T>(numStages)) 
                           : totalGainLin;

        // Calculate maximum allowed delta per sample based on the physical sample rate
        T maxSlewDelta = T(0);
        if (slewMs > T(0)) {
            maxSlewDelta = (ceiling / (slewMs * T(0.001))) / static_cast<T>(currentSampleRate);
        }

        // Branchless inner loop routing: The switch is resolved outside the sample loop
        switch (modeVal)
        {
            case Mode::Hard:        dispatchClipping<Mode::Hard>(buffer, totalGainLin, stageGain, numStages, ceiling, maxSlewDelta); break;
            case Mode::Soft:        dispatchClipping<Mode::Soft>(buffer, totalGainLin, stageGain, numStages, ceiling, maxSlewDelta); break;
            case Mode::Analog:      dispatchClipping<Mode::Analog>(buffer, totalGainLin, stageGain, numStages, ceiling, maxSlewDelta); break;
            case Mode::GoldenRatio: dispatchClipping<Mode::GoldenRatio>(buffer, totalGainLin, stageGain, numStages, ceiling, maxSlewDelta); break;
        }
    }

    /**
     * @brief Templated processing loop. Highly optimized for auto-vectorization.
     */
    template <Mode M>
    void dispatchClipping(AudioBufferView<T>& buffer, T totalGainLin, T stageGain, int numStages, T ceiling, T maxSlewDelta) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();
        
        T peakIn  = T(0);
        T peakOut = T(0);

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                T sample = data[i];
                T driven = sample * totalGainLin;

                // Compiler will unroll this loop for small bounded N
                for (int s = 0; s < numStages; ++s)
                {
                    sample *= stageGain;
                    sample = processSample<M>(sample, ceiling);
                }

                // Slew Limiter processing (branch highly predictable if static)
                if (maxSlewDelta > T(0))
                {
                    T delta = sample - slewPrev_[ch];
                    if (std::abs(delta) > maxSlewDelta)
                        sample = slewPrev_[ch] + std::copysign(maxSlewDelta, delta);
                }
                
                slewPrev_[ch] = sample;
                
                peakIn  = std::max(peakIn, std::abs(driven));
                peakOut = std::max(peakOut, std::abs(sample));
                data[i] = sample;
            }
        }

        // Safe gain reduction calculation
        if (peakIn > T(1e-6)) {
            auto ratio = std::min(peakOut / peakIn, T(1));
            gainReductionDb_.store(gainToDecibels(ratio, T(-100)), std::memory_order_relaxed);
        } else {
            gainReductionDb_.store(T(0), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Compile-time resolution of the waveshaping math.
     */
    template <Mode M>
    [[nodiscard]] static inline T processSample(T sample, T ceiling) noexcept
    {
        if constexpr (M == Mode::Hard)
        {
            return std::clamp(sample, -ceiling, ceiling);
        }
        else if constexpr (M == Mode::Soft)
        {
            return ceiling * std::tanh(sample / ceiling);
        }
        else if constexpr (M == Mode::Analog)
        {
            // Unity-slope sine shaper: sin(x/ceiling) has derivative 1 at the
            // origin, so quiet material passes at exactly 0 dB like the other
            // modes (the old pre-scaled form had slope pi/2 — a +3.9 dB jump
            // when switching modes). The knee spans |x| in [~ceiling*0.5,
            // ceiling*pi/2] and lands exactly on the ceiling.
            constexpr T halfPi = static_cast<T>(std::numbers::pi * 0.5);
            return ceiling * fastSin(std::clamp(sample / ceiling, -halfPi, halfPi));
        }
        else if constexpr (M == Mode::GoldenRatio)
        {
            // True mathematical Golden Ratio soft-knee
            T threshold = ceiling / kPhi;
            T absSample = std::abs(sample);
            
            if (absSample <= threshold) 
                return sample;

            // Rational asymptotic curve to the ceiling
            T sign = std::copysign(T(1), sample);
            T excess = absSample - threshold;
            T range = ceiling - threshold;
            
            return sign * (threshold + (range * excess) / (excess + range));
        }
        
        return sample;
    }

    // Mathematical utility helpers
    [[nodiscard]] static T dbToLinear(T dB) noexcept { return std::pow(T(10), dB / T(20)); }
    [[nodiscard]] static T gainToDecibels(T linear, T minusInfinityDb) noexcept 
    { 
        return linear > T(1e-5) ? T(20) * std::log10(linear) : minusInfinityDb; 
    }

    AudioSpec spec_ {};
    DryWetMixer<T> mixer_;
    std::unique_ptr<Oversampling<T>> oversampler_;

    // Lock-free parameter states
    std::atomic<Mode> mode_ { Mode::Hard };
    std::atomic<T> ceilingDb_ { T(0) };
    std::atomic<T> inputGainDb_ { T(0) };
    std::atomic<int> stages_ { 1 };
    std::atomic<T> mix_ { T(1) };
    std::atomic<T> slewLimitMs_ { T(0) }; 
    std::atomic<int> osFactor_ { 1 }; // Default to 1 (off)

    // History states per channel
    T slewPrev_[kMaxChannels] {};
    
    // Metering state
    std::atomic<T> gainReductionDb_ { T(0) };
};

} // namespace dspark