// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Dither.h
 * @brief TPDF dither and quantisation to a target bit depth.
 *
 * Converts high-resolution audio to a lower bit depth by adding triangular
 * (TPDF) dither before quantisation, with optional first-order noise shaping
 * of the total requantisation error. Output samples lie exactly on the
 * integer grid of the target depth, matching the WAV/MP3 writers.
 *
 * Dependencies: none (STL only).
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace dspark {

/**
 * @class Dither
 * @brief TPDF dithering processor with optional 1st-order noise shaping.
 *
 * Adds Triangular Probability Density Function (TPDF) noise of 2 LSB
 * peak-to-peak before quantisation. This replaces correlated quantisation
 * distortion (truncation harmonics, noise modulation) with a constant,
 * signal-independent noise floor.
 *
 * The optional noise shaper feeds back the total requantisation error
 * (dither + quantiser) through a one-sample delay, first-order highpass
 * shaping the entire noise floor: about 15 dB less noise at 1 kHz (44.1 kHz
 * rate) in exchange for a gentle rise towards Nyquist where hearing is least
 * sensitive.
 *
 * Quantised output lies exactly on the integer level grid of the target
 * depth: levels in [-2^(bits-1), 2^(bits-1) - 1], so positive full scale is
 * one step below 1.0, exactly like the int16/int24 file writers. Feeding the
 * output to WavFile round-trips bit-exactly.
 *
 * Threading is owner-managed: call setters and processing from the thread
 * that owns the stream (the PRNG and error state are plain members by
 * design). Use one instance per processing thread.
 *
 * @tparam T Sample type (`float` or `double`).
 */
template <typename T>
class Dither
{
    static_assert(std::is_floating_point_v<T>, "Dither requires a floating-point sample type");

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

        // Levels per unit: 2^(N-1). The representable grid is asymmetric,
        // [-levels, levels - 1], mirroring the signed-integer formats.
        int64_t levels = int64_t(1) << (targetBits_ - 1);

        quantScale_ = static_cast<T>(levels);
        quantStep_  = T(1) / quantScale_;
        loLevel_    = -quantScale_;
        hiLevel_    = quantScale_ - T(1);
        // Feedback cap: a legitimate shaping error never exceeds 1.5 LSB
        // (TPDF within (-1, 1] LSB + quantiser within +-0.5 LSB); see
        // processSampleInternal for why the cap is needed at all.
        errorCap_   = T(2) * quantStep_;
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
     * @param channel Channel index for independent noise shaping history
     *                (0 to 15; out-of-range indices dither without shaping).
     * @return Quantised sample on the target grid, in [-1, 1 - step].
     */
    [[nodiscard]] T processSample(T input, int channel = 0) noexcept
    {
        if (noiseShaping_ && channel >= 0 && channel < kMaxChannels)
            return processSampleInternal<true>(input, channel);
        else
            return processSampleInternal<false>(input, channel);
    }

    /**
     * @brief Applies dithering to an entire audio buffer in-place.
     *
     * The noise shaping branch is resolved once outside the sample loop.
     * (The PRNG recurrence is serial, so this loop does not vectorise.)
     *
     * @param data Pointer to audio data.
     * @param numSamples Number of samples to process.
     * @param channel Channel index.
     */
    void processBlock(T* data, int numSamples, int channel = 0) noexcept
    {
        if (noiseShaping_ && channel >= 0 && channel < kMaxChannels)
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
        // 2 LSB peak-to-peak TPDF noise
        const T noise = (nextRandom() + nextRandom()) * quantStep_;

        T preQuantise = input;

        if constexpr (ApplyNoiseShaping)
        {
            preQuantise -= errorState_[static_cast<size_t>(channel)];
        }

        // Add dither to the signal (w[n] = v[n] + d[n])
        const T dithered = preQuantise + noise;

        // Quantise on the integer level grid and clamp to the representable
        // range of the target depth. Ties are broken by the dither, so the
        // rounding mode of nearbyint (default round-to-nearest) is fine and
        // compiles to a single instruction on SSE4.1/NEON.
        const T level = std::clamp(std::nearbyint(dithered * quantScale_), loLevel_, hiLevel_);
        const T quantised = level * quantStep_;

        if constexpr (ApplyNoiseShaping)
        {
            // Feed back the TOTAL requantisation error e[n] = y[n] - v[n]
            // (dither + quantiser), the classic error-feedback structure
            // with the dither inside the loop: the whole noise floor is
            // shaped by (1 - z^-1), not just the quantiser error.
            // The cap bounds the loop when the level clamp engages (input at
            // or beyond positive full scale): without it the clip error
            // accumulates sample after sample and pins the output at full
            // scale long after the overload has passed.
            const T err = quantised - preQuantise;
            errorState_[static_cast<size_t>(channel)] = std::clamp(err, -errorCap_, errorCap_);
        }

        return quantised;
    }

    /**
     * @brief Fast Xorshift32 PRNG.
     * @return Uniform random value in (-0.5, 0.5].
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
    T loLevel_ = T(-32768);
    T hiLevel_ = T(32767);
    T errorCap_ = T(2) / T(32768);
    bool noiseShaping_ = false;
    uint32_t rngState_;
    std::array<T, kMaxChannels> errorState_{};
};

} // namespace dspark
