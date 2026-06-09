// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>

namespace dspark {

/**
 * @class Dither
 * @brief TPDF dithering processor with optional 1st-order noise shaping.
 *
 * Converts high-resolution audio to a lower bit-depth by adding a Triangular 
 * Probability Density Function (TPDF) noise before quantisation. This eliminates 
 * correlated quantisation distortion (truncation distortion), trading it for a 
 * constant, uncorrelated noise floor.
 *
 * The optional 1st-order noise shaper applies an error-feedback loop to shift 
 * the quantisation noise energy into higher frequencies where the human ear is 
 * less sensitive, while keeping the dither noise itself perfectly flat.
 *
 * @note **Thread Safety:** A single instance is NOT safe to be called concurrently 
 * from multiple threads due to the PRNG state mutation. Use one instance per 
 * processing thread.
 *
 * @tparam T Sample type (`float` or `double`).
 */
template <typename T>
class Dither
{
public:
    /**
     * @brief Constructs a dithering processor.
     *
     * Automatically seeds the internal PRNG with a globally unique seed to ensure 
     * uncorrelated noise between multiple instances (e.g., Left and Right channels).
     *
     * @param targetBits Target bit depth (min 8). Max is 24 for `float`, 32 for `double`.
     * @param noiseShaping Enable 1st-order noise shaping.
     */
    explicit Dither(int targetBits = 16, bool noiseShaping = false) noexcept
        : noiseShaping_(noiseShaping)
    {
        static std::atomic<uint32_t> seedGenerator{ 0x193A6B54u };
        // Increment by golden ratio to ensure diverse initial states across instances
        rngState_ = seedGenerator.fetch_add(0x9E3779B9u, std::memory_order_relaxed);
        // xorshift32 has a fixed point at 0: if the rolling seed ever lands
        // there, the dither would fall silent for that instance.
        if (rngState_ == 0) rngState_ = 1;

        setTargetBitDepth(targetBits);
    }

    /**
     * @brief Sets the target bit depth and recalculates scaling factors.
     * @param bits Target bit depth. Automatically clamped based on the precision of `T`.
     */
    void setTargetBitDepth(int bits) noexcept
    {
        // float (32-bit IEEE 754) only has 24 bits of mantissa precision.
        constexpr int maxBits = (sizeof(T) == 4) ? 24 : 32;
        targetBits_ = std::clamp(bits, 8, maxBits);

        // Calculate levels: 2^(N-1)
        int64_t levels = int64_t(1) << (targetBits_ - 1);
        
        quantScale_ = static_cast<T>(levels);
        quantStep_ = T(1) / quantScale_;
    }

    /** @brief Enables or disables 1st-order noise shaping. */
    void setNoiseShaping(bool enabled) noexcept { noiseShaping_ = enabled; }

    /** @brief Resets the noise shaping state. Call this when playback stops or flushes. */
    void reset() noexcept
    {
        std::fill(errorState_.begin(), errorState_.end(), T(0));
    }

    /**
     * @brief Applies TPDF dither and quantises a single sample.
     * @param input Input sample in [-1.0, 1.0].
     * @param channel Channel index for independent noise shaping history (max 16).
     * @return Quantised sample.
     */
    [[nodiscard]] T processSample(T input, int channel = 0) noexcept
    {
        if (noiseShaping_ && channel < kMaxChannels)
            return processSampleInternal<true>(input, channel);
        else
            return processSampleInternal<false>(input, channel);
    }

    /**
     * @brief Applies dithering to an entire audio buffer in-place.
     * * Optimised to evaluate the noise shaping branch outside the sample loop,
     * allowing the compiler to pipeline and potentially vectorise the processing.
     *
     * @param data Pointer to audio data.
     * @param numSamples Number of samples to process.
     * @param channel Channel index.
     */
    void processBlock(T* data, int numSamples, int channel = 0) noexcept
    {
        if (noiseShaping_ && channel < kMaxChannels)
        {
            for (int i = 0; i < numSamples; ++i)
                data[i] = processSampleInternal<true>(data[i], channel);
        }
        else
        {
            for (int i = 0; i < numSamples; ++i)
                data[i] = processSampleInternal<false>(data[i], channel);
        }
    }

    [[nodiscard]] int getTargetBitDepth() const noexcept { return targetBits_; }
    [[nodiscard]] T getQuantisationStep() const noexcept { return quantStep_; }

private:
    static constexpr int kMaxChannels = 16;

    /**
     * @brief Internal processing template to resolve branching at compile time.
     */
    template <bool ApplyNoiseShaping>
    [[nodiscard]] inline T processSampleInternal(T input, int channel) noexcept
    {
        // 2 LSB Peak-to-Peak TPDF noise
        T noise = (nextRandom() + nextRandom()) * quantStep_;
        
        T preQuantise = input;

        if constexpr (ApplyNoiseShaping)
        {
            preQuantise -= errorState_[static_cast<size_t>(channel)];
        }

        // Add dither to the signal (w[n] = v[n] + d[n])
        T dithered = preQuantise + noise;

        // Quantise: multiply by scale, round, multiply by step (no slow divisions)
        T quantised = std::round(dithered * quantScale_) * quantStep_;
        quantised = std::clamp(quantised, T(-1), T(1));

        if constexpr (ApplyNoiseShaping)
        {
            // Error is STRICTLY the quantisation error: e[n] = y[n] - w[n]
            // This ensures dither remains flat while quantisation noise is shaped.
            errorState_[static_cast<size_t>(channel)] = quantised - dithered;
        }

        return quantised;
    }

    /**
     * @brief Fast Xorshift32 PRNG.
     * @return Uniform random value in [-0.5, 0.5).
     */
    [[nodiscard]] inline T nextRandom() noexcept
    {
        rngState_ ^= rngState_ << 13;
        rngState_ ^= rngState_ >> 17;
        rngState_ ^= rngState_ << 5;

        constexpr T scale = T(1) / static_cast<T>(std::numeric_limits<uint32_t>::max());
        return static_cast<T>(rngState_) * scale - T(0.5);
    }

    int targetBits_ = 16;
    T quantStep_ = T(1) / T(32768);
    T quantScale_ = T(32768);
    bool noiseShaping_ = false;
    uint32_t rngState_;
    std::array<T, kMaxChannels> errorState_{};
};

} // namespace dspark