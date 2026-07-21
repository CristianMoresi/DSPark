// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file NoiseGenerator.h
 * @brief Audio noise generator conforming to the AudioProcessor contract.
 *
 * Generates white, pink, or brown noise using AnalogRandom internally.
 * Follows the standard prepare/processBlock/reset pattern. This acts as a
 * signal generator, completely overwriting the input buffer (channels beyond
 * the prepared count are left untouched).
 *
 * Threading: prepare() belongs to the setup thread (allocates). processBlock()
 * and reset() belong to the audio thread. Setters are lock-free atomic
 * publications, safe from any thread; a type change is applied at the next
 * processBlock(). Non-finite gains are ignored (setLevel(-inf) is the
 * documented silence and maps to gain 0).
 *
 * Dependencies: AnalogRandom.h, DspMath.h, AudioSpec.h, AudioBuffer.h,
 *               StateBlob.h.
 *
 * @code
 *   dspark::NoiseGenerator<float> noise;
 *   noise.prepare(spec);
 *   noise.setType(dspark::NoiseGenerator<float>::Type::Pink);
 *   noise.setLevel(-12.0f);
 *
 *   // In the audio callback, fills the buffer with noise:
 *   noise.processBlock(buffer);
 * @endcode
 */

#include "../Core/AnalogRandom.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class NoiseGenerator
 * @brief Generates decorellated noise (white, pink, brown) across multiple channels.
 *
 * Ensures cache-friendly memory layout and lock-free thread safety. Implements
 * block-level linear gain smoothing to prevent audio artifacts (zipper noise)
 * upon parameter changes.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class NoiseGenerator
{
public:
    enum class Type
    {
        White,
        Pink,
        Brown
    };

    /**
     * @brief Prepares the noise generator and allocates internal state.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification containing sample rate and channel count.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        int targetChannels = spec.numChannels;

        generators_.clear();
        generators_.reserve(static_cast<size_t>(targetChannels));

        for (int ch = 0; ch < targetChannels; ++ch)
        {
            AnalogRandom::Generator<T> gen;
            gen.prepare(sampleRate_);
            gen.setRange(T(-1), T(1));
            // A new value EVERY sample (rate == fs): full-bandwidth noise. A lower
            // rate would sample-and-hold the source, giving a non-flat/imaged
            // spectrum instead of true white/pink/brown.
            gen.setRateHz(static_cast<T>(sampleRate_));
            applyNoiseType(gen);

            // Decorrelate output across channels
            gen.reseed(static_cast<uint64_t>(ch + 1) * 0x9E3779B97F4A7C15ULL);
            generators_.push_back(std::move(gen));
        }

        // Sync smoothing state to prevent ramp-up from zero on first block
        currentGain_ = gain_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Fills the buffer with generated noise.
     *
     * Overwrites all existing data in the buffer. Applies smoothed gain changes
     * transparently. Real-time safe (lock-free, zero allocations).
     *
     * @param buffer Audio buffer view to fill.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (generators_.empty())
            return;

        if (typeDirty_.exchange(false, std::memory_order_acquire))
        {
            for (auto& gen : generators_)
                applyNoiseType(gen);
        }

        const int numCh = std::min<int>(buffer.getNumChannels(), static_cast<int>(generators_.size()));
        const int numSamples = buffer.getNumSamples();

        if (numSamples <= 0 || numCh <= 0)
            return;

        // Block-level gain smoothing setup
        const T targetGain = gain_.load(std::memory_order_relaxed);
        const T startGain = currentGain_;
        const bool needsSmoothing = std::abs(targetGain - startGain) > T(1e-6);
        const T gainStep = needsSmoothing ? ((targetGain - startGain) / static_cast<T>(numSamples)) : T(0);

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& gen = generators_[static_cast<size_t>(ch)];

            if (needsSmoothing)
            {
                T g = startGain;
                for (int i = 0; i < numSamples; ++i)
                {
                    g += gainStep;
                    data[i] = gen.getNextSample() * g;
                }
            }
            else
            {
                for (int i = 0; i < numSamples; ++i)
                {
                    data[i] = gen.getNextSample() * targetGain;
                }
            }
        }

        // Advance global smoothed state for the next block
        currentGain_ = targetGain;
    }

    /**
     * @brief Resets the internal PRNG and filter states.
     *
     * Restores the per-channel deterministic seeds too, so an offline render
     * after reset() reproduces the exact same noise sequence (the generator's
     * own reset() only clears filter state; the PRNG would otherwise continue
     * from wherever it was).
     */
    void reset() noexcept
    {
        int ch = 0;
        for (auto& gen : generators_)
        {
            gen.reset();
            gen.reseed(static_cast<uint64_t>(++ch) * 0x9E3779B97F4A7C15ULL);
        }

        currentGain_ = gain_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Sets the noise spectrum type.
     *
     * Thread-safe and lock-free. The update is deferred to the audio thread
     * during the next processBlock() call to avoid mutating filter states mid-read.
     *
     * @param type Noise color (White, Pink, or Brown).
     */
    void setType(Type type) noexcept
    {
        // Wild enum values (a corrupted blob, a stray cast) clamp into range.
        const int t = std::clamp(static_cast<int>(type), 0,
                                 static_cast<int>(Type::Brown));
        type_.store(static_cast<Type>(t), std::memory_order_relaxed);
        typeDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the output level in decibels. Thread-safe.
     * @param levelDb Level in dB (0 = unity, -inf = silence). NaN and +inf
     *                are ignored; -inf maps to gain 0 as documented.
     */
    void setLevel(T levelDb) noexcept
    {
        if (std::isnan(levelDb) || levelDb > T(1e6)) return; // rejects +inf too
        setGain(decibelsToGain(levelDb)); // -inf (<= -100 dB) maps to 0
    }

    /**
     * @brief Sets the output level as linear gain. Thread-safe.
     * @param gain Linear gain coefficient (1.0 = unity).
     *             Non-finite values are ignored.
     */
    void setGain(T gain) noexcept
    {
        if (!std::isfinite(gain)) return;
        gain_.store(gain, std::memory_order_relaxed);
    }

    /**
     * @brief Retrieves the currently requested noise type.
     * @return The active NoiseGenerator::Type.
     */
    [[nodiscard]] Type getType() const noexcept
    {
        return type_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Retrieves the current target gain in linear scale.
     * @return The linear gain coefficient.
     */
    [[nodiscard]] T getGain() const noexcept
    {
        return gain_.load(std::memory_order_relaxed);
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("NOIS"), 1);
        w.write("gain", gain_.load(std::memory_order_relaxed));
        w.write("type", static_cast<int32_t>(type_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("NOIS")) return false;
        setGain(static_cast<T>(r.read("gain", 1.0f)));
        setType(static_cast<Type>(r.read("type", 0)));
        return true;
    }

private:
    void applyNoiseType(AnalogRandom::Generator<T>& gen) const noexcept
    {
        switch (type_.load(std::memory_order_relaxed))
        {
            case Type::White:
                gen.setNoiseType(AnalogRandom::NoiseType::White);
                break;
            case Type::Pink:
                gen.setNoiseType(AnalogRandom::NoiseType::Pink);
                break;
            case Type::Brown:
                gen.setNoiseType(AnalogRandom::NoiseType::Brown);
                break;
        }
    }

    double sampleRate_ = 44100.0;

    // Core parameters (Thread-safe)
    std::atomic<T> gain_ { T(1) };
    std::atomic<Type> type_ { Type::White };
    std::atomic<bool> typeDirty_ { false };

    // Internal state (Audio thread only)
    T currentGain_ { T(1) };

    // Value semantics for memory locality. Avoids pointer chasing.
    std::vector<AnalogRandom::Generator<T>> generators_;
};

} // namespace dspark
