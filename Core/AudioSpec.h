// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file AudioSpec.h
 * @brief Describes the audio processing environment (sample rate, block size, channels).
 *
 * Every DSP processor receives an AudioSpec in its `prepare()` method to configure
 * internal resources (buffers, filter coefficients, smoothing rates, etc.).
 *
 * Dependencies: none.
 *
 * @code
 * dspark::AudioSpec spec { .sampleRate = 48000.0, .maxBlockSize = 512, .numChannels = 2 };
 * if (spec.isValid())
 *     mySaturator.prepare(spec);
 * @endcode
 */

namespace dspark {

/**
 * @struct AudioSpec
 * @brief Describes the audio environment for a DSP processor.
 *
 * Passed to `prepare()` before processing begins.
 *
 * Processors should typically check `if (newSpec == currentSpec)` to avoid
 * redundant allocations or state resets when the host triggers multiple
 * prepare calls.
 */
struct AudioSpec
{
    /**
     * @brief Sample rate in Hz.
     *
     * Initialized to 0.0 by default to enforce explicit initialization.
     * Processing with an uninitialized sample rate will cause immediate,
     * observable failures (NaNs) instead of silent tuning bugs.
     */
    double sampleRate = 0.0;

    /**
     * @brief Maximum number of samples per processing block.
     *
     * Processors use this to pre-allocate internal buffers. The actual block
     * size passed to `process()` may vary but will never exceed this value.
     */
    int maxBlockSize = 0;

    /**
     * @brief Number of audio channels (e.g., 1 = mono, 2 = stereo).
     */
    int numChannels = 0;

    /**
     * @brief Checks if the specification contains valid, processable parameters.
     *
     * Use this in assertions at the start of your processor's `prepare()`
     * method. A NaN sample rate fails the check (every comparison against
     * NaN is false), so corrupted specs are rejected too.
     *
     * @return true if all parameters are strictly positive (> 0), false otherwise.
     */
    [[nodiscard]] constexpr bool isValid() const noexcept
    {
        return sampleRate > 0.0 && maxBlockSize > 0 && numChannels > 0;
    }

    /**
     * @brief Default member-wise equality (C++20).
     *
     * Allows processors to diff the incoming specification against their
     * cached one to skip unnecessary re-allocations or coefficient
     * recalculations. The comparison is exact on purpose: hosts hand over
     * literal configuration values, not computed ones.
     */
    constexpr bool operator==(const AudioSpec&) const noexcept = default;
};

} // namespace dspark
