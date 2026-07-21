// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file RingModulator.h
 * @brief Ring modulation - multiplies the signal by an oscillator carrier.
 *
 * Produces sum and difference frequencies by multiplying the input signal
 * with a sine wave carrier. Classic effect for metallic, bell-like, or
 * robotic tones. Includes a dry/wet mix control and parameter smoothing
 * to prevent zipper noise during real-time automation.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Phasor.h, DspMath.h, AudioSpec.h, AudioBuffer.h, StateBlob.h.
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
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class RingModulator
 * @brief Signal x carrier ring modulation with mix control and zero-latency smoothing.
 *
 * Uses a single shared carrier oscillator for all channels ensuring phase
 * coherence. The shared carrier is precomputed serially chunk by chunk (the
 * phase accumulator is recursive); the per-channel apply loops are
 * branch-free and auto-vectorizable.
 *
 * Channels beyond those passed to prepare() are left untouched
 * (pass-through), as is the whole buffer before prepare().
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
        GeometricMean   ///< Geometric-mean mode: sqrt(|in|*|carrier|) * sign - richer, more musical.
    };

    /**
     * @brief Prepares the modulator for audio processing.
     *
     * Settles the parameter smoothing state on the published targets. An
     * invalid spec (non-positive or non-finite fields) is a no-op that keeps
     * the previous state.
     *
     * @param spec Audio specification including sample rate and channels.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        phasor_.prepare(spec.sampleRate);

        const T initialFreq = frequency_.load(std::memory_order_relaxed);
        const T initialMix = mix_.load(std::memory_order_relaxed);

        phasor_.setFrequency(initialFreq);
        currentFreq_ = initialFreq;
        currentMix_ = initialMix;
        numChannels_ = spec.numChannels;
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

                // fastSin: error > 100 dB below the carrier - inaudible even
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

        // Land exactly on the published targets: the accumulated ramp ends
        // within rounding of them, and the next block must start settled
        // (matches the framework's smoothing convention).
        currentFreq_ = targetFreq;
        currentMix_ = targetMix;
    }

    /** @brief Resets the internal phase of the carrier oscillator. */
    void reset() noexcept { phasor_.reset(); }

    /**
     * @brief Sets the target carrier frequency. Smoothed internally.
     * @param hz Frequency in Hertz. Negative values are legal (identical
     * spectrum, inverted carrier phase); there is no upper clamp (carriers
     * beyond Nyquist alias by design). Non-finite values are ignored.
     */
    void setFrequency(T hz) noexcept
    {
        if (!std::isfinite(hz)) return; // NaN/Inf would poison the smoothed ramp
        frequency_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target dry/wet mix. Smoothed internally.
     * @param mix Mix value from 0.0 (100% dry) to 1.0 (100% wet), clamped.
     * Non-finite values are ignored.
     */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the modulation mode.
     *
     * Applied per block without crossfade (an instantaneous timbre change).
     * Out-of-range values are clamped so the getter stays honest.
     *
     * @param m The chosen structural mode (Classic or GeometricMean).
     */
    void setMode(Mode m) noexcept
    {
        const int v = std::clamp(static_cast<int>(m),
                                 static_cast<int>(Mode::Classic),
                                 static_cast<int>(Mode::GeometricMean));
        mode_.store(static_cast<Mode>(v), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the soar threshold for Geometric Mean mode.
     *
     * Prevents cross-zero dropouts. The value is internally scaled by the
     * input amplitude to prevent static DC offsets during silence. The floor
     * it puts under the carrier's zero crossings turns the dropout into a
     * small step there (the audible trade-off of filling the notch). Applied
     * per block (not smoothed).
     *
     * @param amount 0 = strict geometric mean, 0.01 = subtle, 0.1 = strong.
     * Floored at 0; non-finite values are ignored.
     */
    void setSoar(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        soar_.store(std::max(amount, T(0)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSoar() const noexcept { return soar_.load(std::memory_order_relaxed); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float (setState reads float back); the explicit
        // casts also keep this overload resolvable when T is double.
        StateWriter w(stateId("RING"), 1);
        w.write("frequency", static_cast<float>(frequency_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("soar", static_cast<float>(soar_.load(std::memory_order_relaxed)));
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
    int numChannels_ = 0;

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
