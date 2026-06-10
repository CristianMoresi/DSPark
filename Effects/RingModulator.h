// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file RingModulator.h
 * @brief Ring modulation — multiplies the signal by an oscillator carrier.
 *
 * Produces sum and difference frequencies by multiplying the input signal
 * with a sine wave carrier. Classic effect for metallic, bell-like, or
 * robotic tones. Includes a dry/wet mix control and parameter smoothing
 * to prevent zipper noise during real-time automation.
 *
 * Dependencies: Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::RingModulator<float> ring;
 *   ring.prepare(spec);
 *   ring.setFrequency(440.0f);   // carrier at 440 Hz
 *   ring.setMix(1.0f);           // 100% wet
 *
 *   // In audio callback:
 *   ring.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Phasor.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class RingModulator
 * @brief Signal × carrier ring modulation with mix control and zero-latency smoothing.
 *
 * Uses a single shared carrier oscillator for all channels ensuring phase
 * coherence. Optimized for SIMD vectorization and CPU cache locality via
 * internal chunk-based processing.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class RingModulator
{
public:
    /** @brief Modulation mathematical mode. */
    enum class Mode
    {
        Classic,        ///< Standard multiplication (sum & difference frequencies).
        GeometricMean   ///< Geometric-mean mode: sqrt(|in|*|carrier|) * sign — richer, more musical.
    };

    /**
     * @brief Prepares the modulator for audio processing.
     * @param spec Audio specification including sample rate and channels.
     */
    void prepare(const AudioSpec& spec)
    {
        phasor_.prepare(spec.sampleRate);
        
        T initialFreq = frequency_.load(std::memory_order_relaxed);
        T initialMix = mix_.load(std::memory_order_relaxed);
        
        phasor_.setFrequency(initialFreq);
        currentFreq_ = initialFreq;
        currentMix_ = initialMix;
        numChannels_ = spec.numChannels;
        sampleRate_ = static_cast<T>(spec.sampleRate);
    }

    /**
     * @brief Processes a block of audio in-place.
     * @param buffer View of the audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();
        if (numSamples <= 0 || numCh <= 0) return;

        // Fetch targets for smoothing
        const T targetFreq = frequency_.load(std::memory_order_relaxed);
        const T targetMix = mix_.load(std::memory_order_relaxed);
        const T soarVal = soar_.load(std::memory_order_relaxed);
        const auto modeVal = mode_.load(std::memory_order_relaxed);

        // Calculate smoothing steps
        const T invSamples = T(1) / static_cast<T>(numSamples);
        const T freqStep = (targetFreq - currentFreq_) * invSamples;
        const T mixStep = (targetMix - currentMix_) * invSamples;

        // Process in L1-cache friendly chunks to allow outer channel loop
        constexpr int CHUNK_SIZE = 64;
        std::array<T, CHUNK_SIZE> carrierChunk;
        std::array<T, CHUNK_SIZE> mixChunk;

        const T twoPi = T(2) * static_cast<T>(std::numbers::pi);

        for (int start = 0; start < numSamples; start += CHUNK_SIZE)
        {
            const int chunkLen = std::min(CHUNK_SIZE, numSamples - start);

            // 1. Precalculate shared carrier and mix block
            for (int i = 0; i < chunkLen; ++i)
            {
                currentFreq_ += freqStep;
                currentMix_ += mixStep;
                
                phasor_.setFrequency(currentFreq_);
                T phase = phasor_.advance();

                // fastSin: error > 100 dB below the carrier — inaudible even
                // though the carrier itself is audible in ring modulation.
                carrierChunk[i] = fastSin(phase * twoPi);
                mixChunk[i] = currentMix_;
            }

            // 2. Process channels with hoisted branches and SIMD-friendly loops
            if (modeVal == Mode::GeometricMean)
            {
                for (int ch = 0; ch < numCh; ++ch)
                {
                    T* data = buffer.getChannel(ch) + start;

                    for (int i = 0; i < chunkLen; ++i)
                    {
                        const T dry = data[i];
                        const T carrier = carrierChunk[i];
                        
                        const T absIn = std::abs(dry);
                        const T absCarrier = std::abs(carrier);
                        
                        // Scaled soarVal prevents DC offset when input is zero
                        const T gm = std::sqrt(absIn * absCarrier + soarVal * absIn);
                        
                        // Ultra-fast sign evaluation without branching
                        const T wet = gm * std::copysign(T(1), dry * carrier);
                        
                        data[i] = dry + (wet - dry) * mixChunk[i];
                    }
                }
            }
            else // Mode::Classic
            {
                for (int ch = 0; ch < numCh; ++ch)
                {
                    T* data = buffer.getChannel(ch) + start;

                    for (int i = 0; i < chunkLen; ++i)
                    {
                        const T dry = data[i];
                        const T wet = dry * carrierChunk[i];
                        data[i] = dry + (wet - dry) * mixChunk[i];
                    }
                }
            }
        }
    }

    /** @brief Resets the internal phase of the carrier oscillator. */
    void reset() noexcept { phasor_.reset(); }

    /**
     * @brief Sets the target carrier frequency. Modulated smoothly.
     * @param hz Frequency in Hertz.
     */
    void setFrequency(T hz) noexcept
    {
        frequency_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target dry/wet mix. Modulated smoothly.
     * @param mix Mix value from 0.0 (100% dry) to 1.0 (100% wet).
     */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the modulation mode.
     * @param m The chosen structural mode (Classic or GeometricMean).
     */
    void setMode(Mode m) noexcept 
    { 
        mode_.store(m, std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the soar threshold for Geometric Mean mode.
     *
     * Prevents cross-zero dropouts. The value is internally scaled by the 
     * input amplitude to prevent static DC offsets during silence.
     *
     * @param amount 0 = strict geometric mean, 0.01 = subtle, 0.1 = strong.
     */
    void setSoar(T amount) noexcept
    {
        soar_.store(std::max(amount, T(0)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("RING"), 1);
        w.write("frequency", frequency_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("soar", soar_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("RING")) return false;
        setFrequency(static_cast<T>(r.read("frequency", 440.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setMode(static_cast<Mode>(r.read("mode", 0)));
        setSoar(static_cast<T>(r.read("soar", 0.0f)));
        return true;
    }

private:
    int numChannels_ = 2;
    T sampleRate_ = T(48000);

    // Atomic targets for lock-free UI/Thread communication
    std::atomic<T> frequency_ { T(440) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<Mode> mode_ { Mode::Classic };
    std::atomic<T> soar_ { T(0) };

    // DSP State (Internal audio-thread only, no atomics required)
    T currentFreq_ { T(440) };
    T currentMix_ { T(1) };
    Phasor<T> phasor_;
};

} // namespace dspark
