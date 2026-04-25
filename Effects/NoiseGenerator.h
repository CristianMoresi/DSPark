// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file NoiseGenerator.h
 * @brief Audio noise generator conforming to the AudioProcessor contract.
 *
 * Generates white, pink, or brown noise using AnalogRandom internally.
 * Follows the standard prepare/processBlock/reset pattern, so it can be
 * used in a ProcessorChain as a signal generator.
 *
 * Dependencies: AnalogRandom.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::NoiseGenerator<float> noise;
 *   noise.prepare(spec);
 *   noise.setType(dspark::NoiseGenerator<float>::Type::Pink);
 *   noise.setLevel(-12.0f);  // -12 dB
 *
 *   // In audio callback — fills the buffer with noise:
 *   noise.processBlock(buffer);
 * @endcode
 */

#include "../Core/AnalogRandom.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <atomic>
#include <memory>

namespace dspark {

/**
 * @class NoiseGenerator
 * @brief Generates noise (white, pink, brown) as an AudioProcessor.
 *
 * Output fills the buffer — this is a generator, not an effect.
 * Each channel uses an independent AnalogRandom generator to avoid
 * mono-summing artifacts.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
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
     * @brief Prepares the noise generator.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        generators_.clear();
        generators_.reserve(static_cast<size_t>(numChannels_));

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            auto gen = std::make_unique<AnalogRandom::Generator<T>>();
            gen->prepare(sampleRate_);
            gen->setRange(T(-1), T(1));
            gen->setRateHz(static_cast<T>(sampleRate_ * 0.5));
            applyNoiseType(*gen);

            // Different seed per channel to avoid correlated output
            gen->reseed(static_cast<uint64_t>(ch + 1) * 0x9E3779B97F4A7C15ULL);
            generators_.push_back(std::move(gen));
        }
    }

    /**
     * @brief Fills the buffer with noise (generator — overwrites content).
     * @param buffer Audio buffer to fill with noise.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        // M7b fix: setType() never touches the generator state directly. If a
        // type change is pending, apply it here on the audio thread before the
        // inner loop — no torn reads on the internal filter state.
        if (typeDirty_.exchange(false, std::memory_order_acquire))
        {
            for (auto& gen : generators_)
                applyNoiseType(*gen);
        }

        int numCh = std::min(buffer.getNumChannels(), numChannels_);
        int numSamples = buffer.getNumSamples();

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& gen = *generators_[static_cast<size_t>(ch)];

            T g = gain_.load(std::memory_order_relaxed);
            for (int i = 0; i < numSamples; ++i)
                data[i] = gen.getNextSample() * g;
        }
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        for (auto& gen : generators_)
            gen->reset();
    }

    /**
     * @brief Sets the noise type.
     *
     * Thread-safe: only publishes the request — the audio thread picks it up
     * at the top of the next processBlock and reconfigures the generators
     * there, so the filter state is never mutated mid-read.
     *
     * @param type White, Pink, or Brown.
     */
    void setType(Type type) noexcept
    {
        type_.store(type, std::memory_order_relaxed);
        typeDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the output level in decibels.
     * @param levelDb Level in dB (0 = unity, -inf = silence).
     */
    void setLevel(T levelDb) noexcept
    {
        gain_.store(decibelsToGain(levelDb), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the output level as linear gain.
     * @param gain Linear gain (1.0 = unity).
     */
    void setGain(T gain) noexcept
    {
        gain_.store(gain, std::memory_order_relaxed);
    }

    /** @brief Returns the current noise type. */
    [[nodiscard]] Type getType() const noexcept { return type_.load(std::memory_order_relaxed); }

    /** @brief Returns the current gain in linear. */
    [[nodiscard]] T getGain() const noexcept { return gain_.load(std::memory_order_relaxed); }

private:
    void applyNoiseType(AnalogRandom::Generator<T>& gen) noexcept
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
    int numChannels_ = 2;
    std::atomic<T> gain_ { T(1) };
    std::atomic<Type> type_ { Type::White };
    std::atomic<bool> typeDirty_ { false };   // set by setType, drained by processBlock

    std::vector<std::unique_ptr<AnalogRandom::Generator<T>>> generators_;
};

} // namespace dspark
