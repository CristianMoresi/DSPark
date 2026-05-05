// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Phaser.h
 * @brief Classic analog-modeled phaser effect using LFO-modulated allpass filter stages.
 *
 * Creates the sweeping "jet" sound by passing the signal through a series of
 * first-order allpass filters whose cutoff frequencies are modulated by an LFO. 
 * Includes an analog-style saturation stage in the feedback loop to emulate
 * classic hardware behavior and prevent harsh digital clipping.
 *
 * Three levels of API complexity are provided:
 * - **Level 1 (Simple):** `setRate()`, `setDepth()`, `setMix()`
 * - **Level 2 (Intermediate):** `setStages()`, `setFeedback()`, `setFrequencyRange()`
 * - **Level 3 (Expert):** `setLfoWaveform()`, `setCenterFrequency()`
 *
 * Dependencies: Biquad.h, Oscillator.h, DryWetMixer.h, AudioSpec.h, AudioBuffer.h, DspMath.h.
 *
 * @code
 *   dspark::Phaser<float> phaser;
 *   phaser.prepare(spec);
 *   phaser.setRate(0.5f);     // 0.5 Hz sweep
 *   phaser.setFeedback(0.7f); // High resonance
 *   phaser.processBlock(buffer);
 * @endcode
 */

#include "../Core/Oscillator.h"
#include "../Core/DryWetMixer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DspMath.h" // Must provide fast_exp and fast_tan

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Phaser
 * @brief Zero-latency, highly optimized allpass-based phaser.
 *
 * Employs parameter smoothing to prevent zipper noise and utilizes 
 * fast math approximations for real-time performance.
 *
 * @tparam T Sample type (must satisfy FloatType concept, e.g., float or double).
 */
template <FloatType T>
class Phaser
{
public:
    virtual ~Phaser() = default;

    static constexpr int kMaxStages = 12;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the phaser and allocates internal state blocks.
     * @param spec Hardware audio specifications (sample rate, block size).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        lfo_.prepare(spec.sampleRate);
        lfo_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfo_.setWaveform(lfoWaveform_);

        // Calculate smoothing coefficient (approx 50ms response time)
        smoothCoef_ = T(1) - std::exp(static_cast<T>(-2.0 * std::numbers::pi * 20.0) / static_cast<T>(spec_.sampleRate));

        reset();
    }

    /**
     * @brief Processes an audio block in-place.
     * @param buffer The AudioBufferView containing channels to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        // Target parameters loaded once per block
        const T targetDepth = depth_.load(std::memory_order_relaxed);
        const T targetMix   = mix_.load(std::memory_order_relaxed);
        const T targetFb    = feedback_.load(std::memory_order_relaxed);
        const int stagesVal = numStages_.load(std::memory_order_relaxed);
        const T minF        = minFreq_.load(std::memory_order_relaxed);
        const T maxF        = maxFreq_.load(std::memory_order_relaxed);

        mixer_.pushDry(buffer);

        const T logMin = std::log(minF);
        const T logMax = std::log(maxF);
        const T fsInv  = T(1) / static_cast<T>(spec_.sampleRate);
        constexpr T kPi = static_cast<T>(std::numbers::pi);

        for (int i = 0; i < nS; ++i)
        {
            // Parameter smoothing (1-pole lowpass) to prevent zipper noise
            currentDepth_ += smoothCoef_ * (targetDepth - currentDepth_);
            currentFb_    += smoothCoef_ * (targetFb - currentFb_);

            // LFO calculates log-scaled frequency
            T lfoVal = lfo_.getNextSample(); 
            T modAmount = (lfoVal + T(1)) * T(0.5) * currentDepth_; 
            T logFreq = logMin + modAmount * (logMax - logMin);
            
            // Convert log-frequency back to linear Hz (fastExp ≈ 2× std::exp)
            T cutoff = fastExp(logFreq);

            // Clamp to Nyquist boundary
            T nyquist = static_cast<T>(spec_.sampleRate) * T(0.499);
            cutoff = std::min(cutoff, nyquist);

            // Bilinear-transform coefficient: tan(π·f/Fs).
            // fastTan is safe here — `cutoff` is always < Nyquist so the
            // argument is bounded to π/2 by the clamp above.
            T tanVal = fastTan(kPi * cutoff * fsInv);
            T c = (tanVal - T(1)) / (tanVal + T(1));

            // Process active channels
            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = buffer.getChannel(ch)[i];

                // Analog-modeled feedback with soft clipping to prevent digital blowups
                T fbSignal = std::tanh(fbState_[ch] * currentFb_);
                sample += fbSignal;

                // Series first-order allpass chain
                for (int s = 0; s < stagesVal; ++s)
                {
                    auto& st = stages_[s];
                    T y = c * sample + st.xPrev[ch] - c * st.yPrev[ch];
                    st.xPrev[ch] = sample;
                    st.yPrev[ch] = y;
                    sample = y;
                }

                fbState_[ch] = sample; // 1-sample delay for next iteration
                buffer.getChannel(ch)[i] = sample;
            }
        }

        // Mix parameter is smoothed internally by DryWetMixer if implemented correctly,
        // otherwise apply similar smoothing logic here.
        mixer_.mixWet(buffer, targetMix); 
    }

    /** 
     * @brief Clears internal DSP state and history buffers. 
     * Call this when transport stops or playback jumps.
     */
    void reset() noexcept
    {
        for (auto& stage : stages_)
        {
            stage.xPrev.fill(T(0));
            stage.yPrev.fill(T(0));
        }
        for (auto& fb : fbState_) fb = T(0);

        currentDepth_ = depth_.load(std::memory_order_relaxed);
        currentFb_    = feedback_.load(std::memory_order_relaxed);

        lfo_.reset();
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the speed of the phaser sweep.
     * @param hz LFO frequency in Hertz. Range: [0.01, 20.0].
     */
    void setRate(T hz) noexcept
    {
        hz = std::clamp(hz, T(0.01), T(20));
        rate_.store(hz, std::memory_order_relaxed);
        lfo_.setFrequency(hz);
    }

    /**
     * @brief Sets how wide the frequency sweep is.
     * @param amount Range: [0.0 (static), 1.0 (full spectrum)].
     */
    void setDepth(T amount) noexcept
    {
        depth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Balances the unprocessed and processed signal.
     * @param dryWet Range: [0.0 (fully dry), 1.0 (fully wet)].
     */
    void setMix(T dryWet) noexcept 
    { 
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); 
    }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Configures the intensity/color of the phaser via filter stages.
     * @param count Number of allpass stages. Range: [1, 12]. Evens (2,4,6) are standard.
     */
    void setStages(int count) noexcept
    {
        numStages_.store(std::clamp(count, 1, kMaxStages), std::memory_order_relaxed);
    }

    /**
     * @brief Injects phase-shifted signal back into the input for resonance.
     * @param amount Feedback gain. Range: [-0.99, 0.99]. Negative values alter the vocal formant shape.
     */
    void setFeedback(T amount) noexcept
    {
        feedback_.store(std::clamp(amount, T(-0.99), T(0.99)), std::memory_order_relaxed);
    }

    /**
     * @brief Defines the midpoint of the logarithmic frequency sweep.
     * @param hz Center frequency in Hertz.
     */
    void setCenterFrequency(T hz) noexcept
    {
        T curMin = minFreq_.load(std::memory_order_relaxed);
        T curMax = maxFreq_.load(std::memory_order_relaxed);
        T halfRange = std::sqrt(curMax / (curMin > T(0) ? curMin : T(1))); 
        
        minFreq_.store(std::max(hz / halfRange, T(20)), std::memory_order_relaxed);
        maxFreq_.store(std::min(hz * halfRange, static_cast<T>(spec_.sampleRate) * T(0.499)), std::memory_order_relaxed);
    }

    /**
     * @brief Explicitly defines the sweep boundaries.
     * @param minHz Lower limit of the allpass cutoff sweep.
     * @param maxHz Upper limit of the allpass cutoff sweep.
     */
    void setFrequencyRange(T minHz, T maxHz) noexcept
    {
        T mn = std::max(minHz, T(20));
        minFreq_.store(mn, std::memory_order_relaxed);
        maxFreq_.store(std::max(maxHz, mn + T(1)), std::memory_order_relaxed);
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Changes the geometric shape of the LFO modulation.
     * @param wf Target waveform (e.g., Sine, Triangle).
     */
    void setLfoWaveform(typename Oscillator<T>::Waveform wf) noexcept
    {
        lfoWaveform_ = wf;
        lfo_.setWaveform(wf);
    }

    /** @brief Retrieves the active stage count. @return Number of stages. */
    [[nodiscard]] int getStages() const noexcept { return numStages_.load(std::memory_order_relaxed); }

    /** @brief Retrieves the current sweep rate. @return Rate in Hz. */
    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }

protected:
    static constexpr int kMaxChannels = 16;

    AudioSpec spec_ {};

    // Parameters
    std::atomic<T> rate_     { T(0.5) };
    std::atomic<T> depth_    { T(0.8) };
    std::atomic<T> mix_      { T(0.5) };
    std::atomic<T> feedback_ { T(0) };
    std::atomic<T> minFreq_  { T(200) };
    std::atomic<T> maxFreq_  { T(6000) };
    std::atomic<int> numStages_ { 4 };
    typename Oscillator<T>::Waveform lfoWaveform_ = Oscillator<T>::Waveform::Sine;

    // Smoothed state parameters
    T currentDepth_ { T(0.8) };
    T currentFb_    { T(0) };
    T smoothCoef_   { T(0.01) }; 

    struct FirstOrderAllpass
    {
        std::array<T, kMaxChannels> xPrev {};
        std::array<T, kMaxChannels> yPrev {};
    };

    // Processing state
    std::array<FirstOrderAllpass, kMaxStages> stages_ {};
    std::array<T, kMaxChannels> fbState_ {};
    Oscillator<T> lfo_;
    DryWetMixer<T> mixer_;
};

} // namespace dspark
