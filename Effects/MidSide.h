// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file MidSide.h
 * @brief Mid/Side stereo encoding and decoding for real-time audio.
 *
 * Provides lossless conversion between Left/Right and Mid/Side representations.
 * Useful for stereo processing where independent control over the centre image
 * (mid) and stereo width (side) is needed.
 *
 * Dependencies: AudioBuffer.h only.
 *
 * The transform is its own inverse (encode == decode conceptually), but separate 
 * functions are provided for clarity at the call site.
 */

#include "../Core/AudioBuffer.h"

#include <cassert>

// Compiler-specific restrict macro to guarantee auto-vectorization in hot paths
#if defined(__clang__) || defined(__GNUC__)
    #define DSPARK_RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define DSPARK_RESTRICT __restrict
#else
    #define DSPARK_RESTRICT
#endif

namespace dspark {

/**
 * @struct MidSide
 * @brief Static utility for Mid/Side stereo encoding and decoding.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
struct MidSide
{
    /**
     * @brief Encodes a single sample frame from Left/Right to Mid/Side in-place.
     *
     * Useful for interleaving M/S processing inside a custom DSP loop to 
     * maintain L1/L2 cache locality and avoid multi-pass buffer traversals.
     *
     * @param left_mid Reference to the left sample (becomes mid).
     * @param right_side Reference to the right sample (becomes side).
     */
    static constexpr void encodeSample(T& left_mid, T& right_side) noexcept
    {
        const T l = left_mid;
        const T r = right_side;
        left_mid   = (l + r) * T(0.5);
        right_side = (l - r) * T(0.5);
    }

    /**
     * @brief Decodes a single sample frame from Mid/Side to Left/Right in-place.
     *
     * @param mid_left Reference to the mid sample (becomes left).
     * @param side_right Reference to the side sample (becomes right).
     */
    static constexpr void decodeSample(T& mid_left, T& side_right) noexcept
    {
        const T m = mid_left;
        const T s = side_right;
        mid_left   = m + s;
        side_right = m - s;
    }

    /**
     * @brief Encodes an entire stereo buffer from Left/Right to Mid/Side.
     *
     * M = (L + R) * 0.5
     * S = (L - R) * 0.5
     *
     * The 0.5 scaling preserves peak amplitude (unity round-trip gain) and avoids 
     * clipping when summing highly correlated signals.
     *
     * @param buffer Stereo buffer (2 channels). Modified in-place.
     */
    static void encode(AudioBufferView<T>& buffer) noexcept
    {
        assert(buffer.getNumChannels() >= 2);
        
        const int n = buffer.getNumSamples();
        if (n <= 0) return; // Early exit for empty buffers

        T* DSPARK_RESTRICT left  = buffer.getChannel(0);
        T* DSPARK_RESTRICT right = buffer.getChannel(1);

        for (int i = 0; i < n; ++i)
        {
            const T l = left[i];
            const T r = right[i];
            left[i]  = (l + r) * T(0.5); // Mid
            right[i] = (l - r) * T(0.5); // Side
        }
    }

    /**
     * @brief Decodes an entire stereo buffer from Mid/Side back to Left/Right.
     *
     * L = M + S
     * R = M - S
     *
     * @param buffer Stereo buffer (2 channels, containing M/S). Modified in-place.
     */
    static void decode(AudioBufferView<T>& buffer) noexcept
    {
        assert(buffer.getNumChannels() >= 2);

        const int n = buffer.getNumSamples();
        if (n <= 0) return; // Early exit

        T* DSPARK_RESTRICT mid  = buffer.getChannel(0);
        T* DSPARK_RESTRICT side = buffer.getChannel(1);

        for (int i = 0; i < n; ++i)
        {
            const T m = mid[i];
            const T s = side[i];
            mid[i]  = m + s; // Left
            side[i] = m - s; // Right
        }
    }
};

} // namespace dspark