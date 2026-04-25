// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file RingBuffer.h
 * @brief Circular buffer with interpolated reads for audio delay lines.
 *
 * Provides a power-of-two circular buffer optimised for audio use. Supports
 * integer and fractional-sample reads with multiple interpolation methods.
 * This is the foundation for delay lines, comb filters, lookahead limiters,
 * chorus effects, and any processor that needs past-sample access.
 *
 * Features:
 * - Power-of-two capacity (bitwise wrap — no modulo)
 * - Single-sample push / read interface
 * - Integer and fractional delay reads
 * - Multiple interpolation modes (linear, cubic, Hermite, Lagrange)
 * - Block push for efficient buffer filling
 *
 * Dependencies: DspMath.h, Interpolation.h.
 *
 * @code
 *   dspark::RingBuffer<float> ring;
 *   ring.prepare(65536);  // 64k samples max delay
 *
 *   // In audio callback:
 *   ring.push(inputSample);
 *   float delayed = ring.readInterpolated(delayInSamples);  // fractional OK
 * @endcode
 */

#include "DspMath.h"
#include "Interpolation.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace dspark {

/**
 * @class RingBuffer
 * @brief Power-of-two circular buffer with interpolated read access.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class RingBuffer
{
public:
    /**
     * @brief Interpolation method for fractional-sample reads.
     */
    enum class InterpMethod
    {
        Linear,   ///< 2-point linear (fast, lower quality)
        Cubic,    ///< 4-point Catmull-Rom (good balance)
        Hermite,  ///< 4-point Hermite (smooth transients)
        Lagrange  ///< 4-point Lagrange (highest accuracy)
    };

    /**
     * @brief Allocates the ring buffer with the given capacity.
     *
     * Capacity is rounded up to the next power of two.
     *
     * @param maxSamples Maximum number of samples to store.
     */
    void prepare(int maxSamples)
    {
        assert(maxSamples > 0);

        // Round up to next power of two
        capacity_ = 1;
        while (capacity_ < maxSamples)
            capacity_ <<= 1;

        mask_ = capacity_ - 1;
        buffer_.assign(static_cast<size_t>(capacity_), T(0));
        writePos_ = 0;
    }

    /**
     * @brief Clears the buffer to zero without deallocating.
     */
    void reset() noexcept
    {
        std::fill(buffer_.begin(), buffer_.end(), T(0));
        writePos_ = 0;
    }

    /**
     * @brief Pushes a single sample into the buffer.
     * @param sample The sample to write.
     */
    void push(T sample) noexcept
    {
        buffer_[static_cast<size_t>(writePos_)] = sample;
        writePos_ = (writePos_ + 1) & mask_;
    }

    /**
     * @brief Pushes a block of samples into the buffer.
     * @param samples Source buffer.
     * @param numSamples Number of samples to push.
     */
    void pushBlock(const T* samples, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            push(samples[i]);
    }

    /**
     * @brief Reads a sample at an integer delay (in samples).
     *
     * A delay of 0 returns the most recently pushed sample.
     * A delay of 1 returns the sample before that, etc.
     *
     * @param delaySamples Delay in samples (0 to capacity-1).
     * @return The delayed sample.
     *
     * @note In debug builds an assertion fires if `delaySamples >= capacity_`.
     *       The bit-mask wrap would otherwise silently return a stale sample
     *       from the "other side" of the ring, which is rarely what the
     *       caller intended (M1).
     */
    [[nodiscard]] T read(int delaySamples) const noexcept
    {
        assert(delaySamples >= 0 && delaySamples < capacity_
               && "RingBuffer::read: delaySamples out of range (0..capacity-1)");
        int idx = (writePos_ - 1 - delaySamples) & mask_;
        return buffer_[static_cast<size_t>(idx)];
    }

    /**
     * @brief Reads a sample at a fractional delay with interpolation.
     *
     * @param delaySamples Fractional delay in samples (e.g., 10.7).
     * @param method Interpolation method (default: Cubic).
     * @return The interpolated sample.
     *
     * @note 4-point interpolators (Cubic/Hermite/Lagrange) read samples at
     *       `intDelay - 1 ... intDelay + 2`. For `delaySamples < 1` this
     *       reaches into a "future" sample which does not exist yet and is
     *       returned as whatever stale value sat there before — usually
     *       zero in a fresh buffer, but potentially old audio after reuse.
     *       Callers that need very short delays must either use Linear
     *       interpolation or guarantee `delaySamples >= 1`. Asserted in
     *       debug (M2).
     */
    [[nodiscard]] T readInterpolated(T delaySamples,
                                     InterpMethod method = InterpMethod::Cubic) const noexcept
    {
        int intDelay = static_cast<int>(delaySamples);
        T frac = delaySamples - static_cast<T>(intDelay);
        assert(delaySamples >= T(0)
               && static_cast<int>(delaySamples) + 2 < capacity_
               && "RingBuffer::readInterpolated: delaySamples out of range");
        assert((method == InterpMethod::Linear || delaySamples >= T(1))
               && "RingBuffer::readInterpolated: 4-point interpolators need delaySamples >= 1");

        switch (method)
        {
            case InterpMethod::Linear:
            {
                T s0 = read(intDelay);
                T s1 = read(intDelay + 1);
                return s0 + frac * (s1 - s0);
            }

            case InterpMethod::Cubic:
            {
                T s[4];
                s[0] = read(intDelay - 1);
                s[1] = read(intDelay);
                s[2] = read(intDelay + 1);
                s[3] = read(intDelay + 2);
                return interpolateCubic(s, 4, T(1) + frac);
            }

            case InterpMethod::Hermite:
            {
                T s[4];
                s[0] = read(intDelay - 1);
                s[1] = read(intDelay);
                s[2] = read(intDelay + 1);
                s[3] = read(intDelay + 2);
                return interpolateHermite(s, 4, T(1) + frac);
            }

            case InterpMethod::Lagrange:
            {
                T s[4];
                s[0] = read(intDelay - 1);
                s[1] = read(intDelay);
                s[2] = read(intDelay + 1);
                s[3] = read(intDelay + 2);
                return interpolateLagrange(s, 4, T(1) + frac);
            }
        }

        return read(intDelay); // fallback
    }

    /**
     * @brief Returns the buffer capacity (power of two).
     */
    [[nodiscard]] int getCapacity() const noexcept { return capacity_; }

    /**
     * @brief Returns direct access to the internal buffer.
     *
     * Use with care — the write position is not exposed. Intended for
     * advanced use cases like FFT analysis of the buffer contents.
     */
    [[nodiscard]] const T* data() const noexcept { return buffer_.data(); }

    /**
     * @brief Returns the current write position.
     */
    [[nodiscard]] int getWritePosition() const noexcept { return writePos_; }

private:
    std::vector<T> buffer_;
    int capacity_ = 0;
    int mask_ = 0;
    int writePos_ = 0;
};

} // namespace dspark
