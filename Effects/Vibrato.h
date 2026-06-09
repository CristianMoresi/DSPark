// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Vibrato.h
 * @brief Pitch modulation via LFO-driven variable delay.
 *
 * Modulates pitch by varying a short delay line with a primary LFO. Features
 * FM modulation on the primary LFO for complex, non-static pitch variations, 
 * and parameter smoothing to prevent zipper noise during automation.
 *
 * Dependencies: Phasor.h, RingBuffer.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::Vibrato<float> vibrato;
 *   vibrato.prepare(spec);
 *   vibrato.setRate(5.0f);          // 5 Hz
 *   vibrato.setDepth(0.5f);         // 0.5 semitones
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
#include <vector>

namespace dspark {

/**
 * @class Vibrato
 * @brief Professional-grade pitch vibrato with LFO FM and parameter smoothing.
 *
 * The modulation depth is specified in semitones. A secondary oscillator (FM)
 * can modulate the primary LFO rate. Internally applies block-based parameter 
 * smoothing to ensure artifact-free automation.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Vibrato
{
public:
    /**
     * @brief Allocates delay lines and resets state.
     * @param spec Audio environment specification. Defines num channels and sample rate.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        invSampleRate_ = T(1.0) / static_cast<T>(sampleRate_);

        // Ensure max delay covers worst-case scenario: 4 semitones at 0.1 Hz
        // At 48kHz, this is ~2650 samples.
        constexpr T minAllowedHz = T(0.1);
        constexpr T maxAllowedSemitones = T(4.0);
        constexpr T kLn2 = static_cast<T>(std::numbers::ln2_v<double>);
        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi);
        
        int maxDelaySamples = static_cast<int>(
            (maxAllowedSemitones * kLn2 * sampleRate_) / (kTwoPi * minAllowedHz * T(12))
        ) + 128; // 128 samples of padding for safety

        delays_.resize(numChannels_);
        phasors_.resize(numChannels_);
        modPhasors_.resize(numChannels_);

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            delays_[ch].prepare(maxDelaySamples);
            phasors_[ch].prepare(sampleRate_);
            modPhasors_[ch].prepare(sampleRate_);
        }

        // Initialize smoothing state to prevent startup jumps
        currentRate_ = rate_.load(std::memory_order_relaxed);
        currentDepth_ = depthSemitones_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Processes audio in-place, applying vibrato per channel.
     * @param buffer Audio data (must match channels passed in prepare).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();
        if (numSamples == 0 || numCh == 0) return;

        // Fetch targets
        const T targetRate = rate_.load(std::memory_order_relaxed);
        const T targetDepth = depthSemitones_.load(std::memory_order_relaxed);
        const T modRate = modRate_.load(std::memory_order_relaxed);
        const T modDepth = modDepth_.load(std::memory_order_relaxed);

        // Parameter smoothing increments (linear ramp over the block)
        const T rateInc = (targetRate - currentRate_) / static_cast<T>(numSamples);
        const T depthInc = (targetDepth - currentDepth_) / static_cast<T>(numSamples);

        constexpr T kLn2 = static_cast<T>(std::numbers::ln2_v<double>);
        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi);
        const T deviationScaler = (kLn2 * static_cast<T>(sampleRate_)) / (kTwoPi * T(12));

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& delay = delays_[ch];
            auto& phasor = phasors_[ch];
            auto& modPhasor = modPhasors_[ch];

            modPhasor.setFrequency(modRate);
            
            // Local state for smoothing
            T smoothRate = currentRate_;
            T smoothDepth = currentDepth_;

            for (int i = 0; i < numSamples; ++i)
            {
                smoothRate += rateInc;
                smoothDepth += depthInc;

                delay.push(data[i]);

                T effectiveRate = std::max(smoothRate, T(0.01));

                // Secondary LFO (FM)
                T fmMod = T(0);
                if (modDepth > T(0)) {
                    T modPhase = modPhasor.advance();
                    fmMod = fastSin(modPhase * kTwoPi) * modDepth;
                }

                // Floor at 0.1 Hz — the SAME minimum the delay-line sizing in
                // prepare() assumes. A lower floor (0.01) let deep FM request
                // deviations ~100x the allocated buffer (wrapped garbage audio).
                T instantRate = std::max(effectiveRate * (T(1) + fmMod), T(0.1));

                // Advance primary phasor manually using the instantaneous FM rate
                phasor.setFrequency(instantRate);
                T phase = phasor.advance(); 

                // Inverse square coupling to maintain perceived depth during FM sweeps.
                // Calculating sqrt per-sample is heavy. In a highly optimized scenario, 
                // this could be approximated or moved to a block-rate calculation if FM is slow.
                T ratioSqrt = std::sqrt(instantRate / effectiveRate);
                T adjustedDepth = smoothDepth / ratioSqrt;

                // Final Deviation calculation
                T deviation = (adjustedDepth * deviationScaler) / instantRate;
                T centre = deviation + T(4.0); // Offset to prevent delay dropping below 0

                T lfo = fastSin(phase * kTwoPi);
                T delaySamples = std::clamp(centre + lfo * deviation, T(1.0),
                                            static_cast<T>(delay.getCapacity() - 4));

                data[i] = delay.readInterpolated(delaySamples);
            }
        }

        // Update current state for the next block
        currentRate_ = targetRate;
        currentDepth_ = targetDepth;
    }

    /** @brief Clears delay line memory and resets LFO phases. */
    void reset() noexcept
    {
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            delays_[ch].reset();
            phasors_[ch].reset();
            modPhasors_[ch].reset();
        }
    }

    /**
     * @brief Sets the primary LFO rate. Parameter is smoothed internally.
     * @param hz Vibrato frequency (0.1 – 14 Hz typical).
     */
    void setRate(T hz) noexcept
    {
        rate_.store(std::max(hz, T(0.1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the vibrato pitch depth. Parameter is smoothed internally.
     * @param semitones Modulation depth (0.0 – 4.0 typical).
     */
    void setDepth(T semitones) noexcept
    {
        depthSemitones_.store(std::clamp(semitones, T(0), T(4)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the rate of the secondary FM oscillator.
     * @param hz Secondary LFO rate in Hz. Set to 0 to disable FM.
     */
    void setModRate(T hz) noexcept
    {
        modRate_.store(std::max(hz, T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the intensity of the FM modulation on the primary LFO.
     * @param amount 0.0 (off) to 1.0 (full modulation).
     */
    void setModDepth(T amount) noexcept
    {
        modDepth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDepth() const noexcept { return depthSemitones_.load(std::memory_order_relaxed); }

private:
    double sampleRate_ = 44100.0;
    T invSampleRate_ = T(1.0 / 44100.0);
    int numChannels_ = 0;

    // Atomic targets for UI Thread
    std::atomic<T> rate_ { T(5) };
    std::atomic<T> depthSemitones_ { T(0.5) };
    std::atomic<T> modRate_ { T(0) };
    std::atomic<T> modDepth_ { T(0) };

    // Smoothed state for Audio Thread
    T currentRate_ { T(5) };
    T currentDepth_ { T(0.5) };

    // Dynamic allocation via STL, safe because it only happens in prepare()
    std::vector<RingBuffer<T>> delays_;
    std::vector<Phasor<T>> phasors_;
    std::vector<Phasor<T>> modPhasors_;
};

} // namespace dspark
