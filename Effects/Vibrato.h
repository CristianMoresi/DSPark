// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Vibrato.h
 * @brief Pitch modulation via LFO-driven variable delay.
 *
 * Modulates pitch by varying a short delay line with an LFO. Uses cubic
 * interpolated reads from RingBuffer for artifact-free sub-sample delays.
 *
 * Dependencies: Phasor.h, RingBuffer.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::Vibrato<float> vibrato;
 *   vibrato.prepare(spec);
 *   vibrato.setRate(5.0f);           // 5 Hz
 *   vibrato.setDepth(0.5f);          // 0.5 semitones
 *
 *   // In audio callback:
 *   vibrato.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Phasor.h"
#include "../Core/RingBuffer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Vibrato
 * @brief Pitch vibrato using modulated delay line.
 *
 * The modulation depth is specified in semitones (0–2 typical). The maximum
 * delay is computed from depth so that the instantaneous pitch deviation
 * matches the desired range.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Vibrato
{
public:
    /**
     * @brief Prepares the vibrato processor.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        // Max delay: enough for 2 semitones at lowest rate (0.1 Hz)
        // deviation_samples = depth_semitones * sampleRate / (rate * 2π * 12)
        // At 2 semitones, 0.1 Hz: ~1326 samples at 48kHz. Round up generously.
        int maxDelay = static_cast<int>(sampleRate_ * 0.1) + 64;

        T rate = rate_.load(std::memory_order_relaxed);
        for (int ch = 0; ch < numChannels_ && ch < kMaxChannels; ++ch)
        {
            delays_[ch].prepare(maxDelay);
            phasors_[ch].prepare(sampleRate_);
            phasors_[ch].setFrequency(rate);
            modPhasors_[ch].prepare(sampleRate_);
            modPhasors_[ch].setFrequency(modRate_.load(std::memory_order_relaxed));
        }
    }

    /**
     * @brief Processes audio in-place (applies vibrato).
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(),
                             std::min(numChannels_, kMaxChannels));
        int numSamples = buffer.getNumSamples();

        T rate     = rate_.load(std::memory_order_relaxed);
        T depth    = depthSemitones_.load(std::memory_order_relaxed);
        T modRate  = modRate_.load(std::memory_order_relaxed);
        T modDepth = modDepth_.load(std::memory_order_relaxed);

        // Update primary LFO rate
        for (int ch = 0; ch < numCh; ++ch)
            phasors_[ch].setFrequency(rate);

        // Update deviation from depth
        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi);
        T effectiveRate = std::max(rate, T(0.01));

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& delay = delays_[ch];
            auto& phasor = phasors_[ch];
            auto& modPhasor = modPhasors_[ch];

            modPhasor.setFrequency(modRate);

            for (int i = 0; i < numSamples; ++i)
            {
                delay.push(data[i]);

                // FM-on-LFO: secondary oscillator modulates primary rate
                T modPhase = modPhasor.advance();
                T fmMod = std::sin(modPhase * kTwoPi) * modDepth;
                T instantRate = effectiveRate * (T(1) + fmMod);

                // Auto depth coupling: scale deviation inversely with rate change
                T ratioSqrt = std::sqrt(std::max(instantRate, T(0.01)) / effectiveRate);
                T adjustedDepth = depth / ratioSqrt;

                // Convert depth (semitones) to peak delay-line deviation in samples.
                //
                // A semitone is a factor of 2^(1/12). A phase-modulated signal
                // x(t − d(t)) where d(t) = D·sin(ωt) produces an instantaneous
                // pitch shift of (1 − ω·D·cos(ωt)). Setting the peak shift equal
                // to semitones·ln(2)/12 gives:
                //     D = (semitones · ln(2) · fs) / (ω · 12)
                // Without the ln(2) factor the actual peak deviation would be
                // 1/ln(2) ≈ 1.4427× larger than requested — i.e. setDepth(1)
                // would produce ~1.44 semitones instead of 1.
                constexpr T kLn2 = static_cast<T>(std::numbers::ln2_v<double>);
                T deviation = adjustedDepth * kLn2 * static_cast<T>(sampleRate_) /
                              (kTwoPi * std::max(instantRate, T(0.01)) * T(12));
                T centre = deviation + T(4);

                T phase = phasor.advance();
                T lfo = std::sin(phase * kTwoPi);

                T delaySamples = std::max(centre + lfo * deviation, T(1));
                data[i] = delay.readInterpolated(delaySamples);
            }
        }
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            delays_[ch].reset();
            phasors_[ch].reset();
            modPhasors_[ch].reset();
        }
    }

    /**
     * @brief Sets the LFO rate.
     * @param hz Vibrato frequency (0.1 – 14 Hz typical).
     */
    void setRate(T hz) noexcept
    {
        rate_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the vibrato depth in semitones.
     * @param semitones 0.0 – 2.0 typical.
     */
    void setDepth(T semitones) noexcept
    {
        depthSemitones_.store(std::clamp(semitones, T(0), T(4)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the FM modulation rate (secondary LFO).
     *
     * A secondary oscillator modulates the primary LFO's rate, creating
     * non-repetitive, complex pitch modulation patterns.
     *
     * @param hz Secondary LFO rate (0 = off, typical: 0.1–2 Hz).
     */
    void setModRate(T hz) noexcept
    {
        modRate_.store(std::max(hz, T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the FM modulation depth.
     * @param amount 0 = off, 0.5 = moderate FM, 1 = full FM.
     */
    void setModDepth(T amount) noexcept
    {
        modDepth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDepth() const noexcept { return depthSemitones_.load(std::memory_order_relaxed); }

private:
    static constexpr int kMaxChannels = 2;

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;

    // Atomic parameters
    std::atomic<T> rate_ { T(5) };
    std::atomic<T> depthSemitones_ { T(0.5) };
    std::atomic<T> modRate_ { T(0) };
    std::atomic<T> modDepth_ { T(0) };

    RingBuffer<T> delays_[kMaxChannels]{};
    Phasor<T> phasors_[kMaxChannels]{};
    Phasor<T> modPhasors_[kMaxChannels]{};
};

} // namespace dspark
