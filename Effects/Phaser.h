// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Phaser.h
 * @brief Classic phaser effect using LFO-modulated allpass filter stages.
 *
 * Creates the sweeping "jet" sound by passing the signal through a series of
 * allpass filters whose cutoff frequencies are modulated by an LFO. The allpass
 * filters shift the phase of different frequencies by different amounts; when
 * mixed with the dry signal, this creates moving notches in the spectrum.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `phaser.setRate(0.5f); phaser.setDepth(0.7f);`
 * - **Level 2 (intermediate):** Number of stages, feedback, frequency range.
 * - **Level 3 (expert):** Direct center frequency, LFO waveform.
 *
 * Dependencies: Biquad.h, Oscillator.h, DryWetMixer.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::Phaser<float> phaser;
 *   phaser.prepare(spec);
 *   phaser.setRate(0.5f);     // Slow sweep
 *   phaser.setDepth(0.8f);    // Deep effect
 *   phaser.setMix(0.5f);      // 50/50
 *   phaser.processBlock(buffer);
 *
 *   // Deeper phasing:
 *   phaser.setStages(6);          // 6 allpass stages
 *   phaser.setFeedback(0.7f);     // Resonant peaks
 * @endcode
 */

#include "../Core/Oscillator.h"
#include "../Core/DryWetMixer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Phaser
 * @brief Allpass-based phaser with configurable stages, feedback, and LFO.
 *
 * The LFO sweeps the allpass cutoff frequencies logarithmically between
 * minFreq and maxFreq. More stages = deeper phasing (more notches).
 * Feedback adds resonant peaks at the notch frequencies.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Phaser
{
public:
    virtual ~Phaser() = default;

    static constexpr int kMaxStages = 12;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the phaser for processing.
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        lfo_.prepare(spec.sampleRate);
        lfo_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfo_.setWaveform(lfoWaveform_);

        for (auto& stage : stages_)
        {
            stage.xPrev.fill(T(0));
            stage.yPrev.fill(T(0));
        }
        for (auto& fb : fbState_)
            fb = T(0);
    }

    /**
     * @brief Processes audio through the phaser.
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        T depthVal    = depth_.load(std::memory_order_relaxed);
        T mixVal      = mix_.load(std::memory_order_relaxed);
        T fbVal       = feedback_.load(std::memory_order_relaxed);
        int stagesVal = numStages_.load(std::memory_order_relaxed);
        T minF        = minFreq_.load(std::memory_order_relaxed);
        T maxF        = maxFreq_.load(std::memory_order_relaxed);

        mixer_.pushDry(buffer);

        const T logMin = std::log(minF);
        const T logMax = std::log(maxF);
        const T fsInv = T(1) / static_cast<T>(spec_.sampleRate);
        constexpr T kPi = static_cast<T>(std::numbers::pi);

        for (int i = 0; i < nS; ++i)
        {
            // LFO modulates the allpass cutoff frequency (log scale)
            T lfoVal = lfo_.getNextSample(); // [-1, 1]
            T modAmount = (lfoVal + T(1)) * T(0.5) * depthVal; // [0, depth]
            T logFreq = logMin + modAmount * (logMax - logMin);
            T cutoff = std::exp(logFreq);

            // Clamp to Nyquist
            T nyquist = static_cast<T>(spec_.sampleRate) * T(0.499);
            cutoff = std::min(cutoff, nyquist);

            // First-order allpass coefficient derived from the bilinear
            // transform:  c = (tan(π·fc/fs) − 1) / (tan(π·fc/fs) + 1)
            // Maps fc ∈ (0, fs/2) to c ∈ (−1, 1), with c = 0 at fc = fs/4.
            // Transfer function: H(z) = (c + z⁻¹) / (1 + c·z⁻¹).
            //
            // This replaces the previous biquad all-pass (a9 optimisation):
            // 2 mul + 2 add per sample/ch instead of 5 mul + 4 add, matching
            // the classic analog phaser topology (MXR Phase 90, Small Stone)
            // which chains single-pole stages rather than second-order ones.
            T tanVal = std::tan(kPi * cutoff * fsInv);
            T c = (tanVal - T(1)) / (tanVal + T(1));

            // Process each channel
            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = buffer.getChannel(ch)[i];

                // Add feedback
                sample += fbState_[ch] * fbVal;

                // First-order allpass chain: y[n] = c·x[n] + x[n-1] - c·y[n-1]
                for (int s = 0; s < stagesVal; ++s)
                {
                    auto& st = stages_[s];
                    T y = c * sample + st.xPrev[ch] - c * st.yPrev[ch];
                    st.xPrev[ch] = sample;
                    st.yPrev[ch] = y;
                    sample = y;
                }

                fbState_[ch] = sample;
                buffer.getChannel(ch)[i] = sample;
            }
        }

        mixer_.mixWet(buffer, mixVal);
    }

    /** @brief Resets all internal state. */
    void reset() noexcept
    {
        for (auto& stage : stages_)
        {
            stage.xPrev.fill(T(0));
            stage.yPrev.fill(T(0));
        }
        for (auto& fb : fbState_)
            fb = T(0);
        lfo_.reset();
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the LFO sweep rate.
     * @param hz LFO frequency in Hz (typical: 0.1 - 5 Hz).
     */
    void setRate(T hz) noexcept
    {
        hz = std::clamp(hz, T(0.01), T(20));
        rate_.store(hz, std::memory_order_relaxed);
        lfo_.setFrequency(hz);
    }

    /**
     * @brief Sets the modulation depth.
     * @param amount 0.0 = no sweep, 1.0 = full range sweep.
     */
    void setDepth(T amount) noexcept
    {
        depth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dry/wet mix.
     * @param dryWet 0.0 = dry, 1.0 = fully wet.
     */
    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Sets the number of allpass stages.
     *
     * More stages = more notches = deeper phasing effect.
     * Must be even for classic phaser sound (2, 4, 6, 8, 10, 12).
     *
     * @param count Number of stages (1 to 12).
     */
    void setStages(int count) noexcept
    {
        numStages_.store(std::clamp(count, 1, kMaxStages), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the feedback amount.
     *
     * Feedback adds resonant peaks at the notch frequencies,
     * making the phasing more pronounced and vocal.
     *
     * @param amount Feedback gain (0.0 to 0.99).
     */
    void setFeedback(T amount) noexcept
    {
        feedback_.store(std::clamp(amount, T(-0.99), T(0.99)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the center frequency for the allpass sweep.
     *
     * The LFO sweeps logarithmically around this frequency.
     * Default is geometric mean of minFreq and maxFreq.
     *
     * @param hz Center frequency in Hz.
     */
    void setCenterFrequency(T hz) noexcept
    {
        T curMin = minFreq_.load(std::memory_order_relaxed);
        T curMax = maxFreq_.load(std::memory_order_relaxed);
        T halfRange = std::sqrt(curMax / curMin);
        minFreq_.store(std::max(hz / halfRange, T(20)), std::memory_order_relaxed);
        maxFreq_.store(std::min(hz * halfRange, static_cast<T>(spec_.sampleRate) * T(0.499)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the frequency range for the allpass sweep.
     *
     * The allpass cutoff frequencies sweep between these limits.
     *
     * @param minHz Lower frequency bound.
     * @param maxHz Upper frequency bound.
     */
    void setFrequencyRange(T minHz, T maxHz) noexcept
    {
        T mn = std::max(minHz, T(20));
        minFreq_.store(mn, std::memory_order_relaxed);
        maxFreq_.store(std::max(maxHz, mn + T(1)), std::memory_order_relaxed);
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Sets the LFO waveform.
     *
     * Sine = smooth classic sweep. Triangle = more linear sweep.
     *
     * @param wf LFO waveform type.
     */
    void setLfoWaveform(typename Oscillator<T>::Waveform wf) noexcept
    {
        lfoWaveform_ = wf;
        lfo_.setWaveform(wf);
    }

    /** @brief Returns the number of active stages. */
    [[nodiscard]] int getStages() const noexcept { return numStages_.load(std::memory_order_relaxed); }

    /** @brief Returns the current rate in Hz. */
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

    // First-order allpass stage (per-channel state, shared coefficient computed per sample).
    // Direct-form with two state variables per channel: xPrev and yPrev.
    struct FirstOrderAllpass
    {
        std::array<T, kMaxChannels> xPrev {};
        std::array<T, kMaxChannels> yPrev {};
    };

    // Processing
    std::array<FirstOrderAllpass, kMaxStages> stages_ {};
    std::array<T, kMaxChannels> fbState_ {};
    Oscillator<T> lfo_;
    DryWetMixer<T> mixer_;
};

} // namespace dspark
