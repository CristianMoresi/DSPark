// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file RingBuffer.h
 * @brief Circular buffer with interpolated reads for audio delay lines.
 *
 * Provides a power-of-two circular buffer optimized for strict real-time audio use.
 * Completely lock-free and allocation-free in the processing path. Data storage 
 * is aligned to 32 bytes to guarantee safe SIMD auto-vectorization.
 *
 * Features:
 * - 32-byte memory alignment (AVX ready)
 * - Zero branching in hot paths via compile-time template selection
 * - Safe pre-initialization state to prevent segfaults
 * - Multiple interpolation modes (Linear, Cubic, Hermite, Lagrange)
 *
 * Dependencies: DspMath.h, Interpolation.h.
 */

#include "DspMath.h"
#include "Interpolation.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace dspark {

/**
 * @brief Interpolation method for fractional-sample reads.
 */
enum class InterpMethod
{
    Linear,   ///< 2-point linear (fast, lower quality)
    Cubic,    ///< 4-point Catmull-Rom (good balance)
    Hermite,  ///< 4-point Hermite (smooth transients, best for modulated delay)
    Lagrange  ///< 4-point Lagrange (highest accuracy, static delay)
};

/**
 * @class RingBuffer
 * @brief Power-of-two circular buffer with compile-time interpolated read access.
 *
 * @tparam T Sample type (float or double). Requires FloatType concept.
 */
template <FloatType T>
class RingBuffer
{
public:
    /**
     * @brief Constructs an empty ring buffer. No allocation happens until
     * prepare() is called — honouring the framework's "allocate only in
     * prepare()" contract. push()/read() are safe no-ops before prepare().
     */
    RingBuffer() noexcept = default;

    /**
     * @brief Allocates the ring buffer with the given capacity.
     *
     * Capacity is rounded up to the next power of two. Memory is aligned to 
     * 32 bytes for SIMD safety. Not thread-safe against concurrent process() calls.
     *
     * @param maxSamples Maximum number of samples to store.
     */
    void prepare(int maxSamples)
    {
        assert(maxSamples > 0);

        capacity_ = 1;
        while (capacity_ < maxSamples)
            capacity_ <<= 1;

        mask_ = capacity_ - 1;

        // Ensure 32-byte alignment for SIMD operations. std::aligned_alloc
        // additionally requires the size to be an integral multiple of the
        // alignment (UB / nullptr otherwise on POSIX), so round bytes up to 32.
        size_t bytes = static_cast<size_t>(capacity_) * sizeof(T);
        bytes = (bytes + 31u) & ~static_cast<size_t>(31u);

        // Custom deleter for aligned allocation
        buffer_.reset(static_cast<T*>(
            #if defined(_MSC_VER)
                _aligned_malloc(bytes, 32)
            #else
                std::aligned_alloc(32, bytes)
            #endif
        ));

        // On allocation failure leave the buffer in a safe unprepared state.
        if (!buffer_) { capacity_ = 0; mask_ = 0; writePos_ = 0; return; }

        reset();
    }

    /**
     * @brief Clears the buffer to zero without deallocating.
     * Safe to call from the audio thread.
     */
    void reset() noexcept
    {
        if (buffer_)
            std::fill_n(buffer_.get(), capacity_, T(0));
        writePos_ = 0;
    }

    /**
     * @brief Pushes a single sample into the buffer.
     * @param sample The sample to write.
     */
    inline void push(T sample) noexcept
    {
        if (capacity_ == 0) [[unlikely]] return; // not prepared
        buffer_[writePos_] = sample;
        writePos_ = (writePos_ + 1) & mask_;
    }

    /**
     * @brief Pushes a block of samples into the buffer.
     *
     * Copies in at most two contiguous memcpy spans (pre-wrap / post-wrap)
     * with a single write-position update — several times faster than a
     * per-sample push loop for block-based delays.
     *
     * @param samples Source buffer.
     * @param numSamples Number of samples to push.
     */
    void pushBlock(const T* samples, int numSamples) noexcept
    {
        if (capacity_ == 0 || numSamples <= 0) [[unlikely]] return;

        if (numSamples >= capacity_)
        {
            // Only the trailing capacity_ samples survive a full overwrite.
            samples += numSamples - capacity_;
            std::memcpy(buffer_.get(), samples, static_cast<size_t>(capacity_) * sizeof(T));
            writePos_ = 0;
            return;
        }

        const int firstSpan = std::min(numSamples, capacity_ - writePos_);
        std::memcpy(buffer_.get() + writePos_, samples,
                    static_cast<size_t>(firstSpan) * sizeof(T));

        const int remaining = numSamples - firstSpan;
        if (remaining > 0)
            std::memcpy(buffer_.get(), samples + firstSpan,
                        static_cast<size_t>(remaining) * sizeof(T));

        writePos_ = (writePos_ + numSamples) & mask_;
    }

    /**
     * @brief Reads a sample at an integer delay.
     *
     * A delay of 0 returns the most recently pushed sample.
     *
     * @param delaySamples Delay in samples (0 to capacity-1).
     * @return The delayed sample.
     */
    [[nodiscard]] inline T read(int delaySamples) const noexcept
    {
        if (capacity_ == 0) [[unlikely]] return T(0); // not prepared
        assert(delaySamples >= 0 && delaySamples < capacity_
               && "RingBuffer::read: delay out of range");

        int idx = (writePos_ - 1 - delaySamples) & mask_;
        return buffer_[idx];
    }

    /**
     * @brief Reads a sample at a fractional delay using the specified interpolation.
     *
     * Uses compile-time template selection to eliminate branching in the audio thread.
     * * @tparam Method Interpolation method to use.
     * @param delaySamples Fractional delay in samples.
     * @return The interpolated sample.
     *
     * @note 4-point interpolators require delaySamples >= 1.0 to avoid reading 
     * future/unwritten samples. Modulating below 1.0 will cause audio artifacts.
     */
    template <InterpMethod Method = InterpMethod::Cubic>
    [[nodiscard]] inline T readInterpolated(T delaySamples) const noexcept
    {
        int intDelay = static_cast<int>(delaySamples);
        T frac = delaySamples - static_cast<T>(intDelay);
        
        assert(delaySamples >= T(0) && intDelay + 2 < capacity_);

        if constexpr (Method == InterpMethod::Linear)
        {
            T s0 = read(intDelay);
            T s1 = read(intDelay + 1);
            return s0 + frac * (s1 - s0);
        }
        else
        {
            assert(delaySamples >= T(1) && "4-point interpolators require delay >= 1.0");

            T s[4];
            // Optimized contiguous logical read
            s[0] = read(intDelay - 1);
            s[1] = read(intDelay);
            s[2] = read(intDelay + 1);
            s[3] = read(intDelay + 2);

            // Interpolate within the s[1]..s[2] interval, so an integer delay
            // (frac == 0) returns exactly read(intDelay) == s[1]. The 4-point
            // (y0,y1,y2,y3,frac) overloads do precisely that. (Calling the
            // buffer+position overloads with `frac` would misread it as an
            // absolute index and interpolate s[0]..s[1] with a wrapped neighbour,
            // i.e. a 1-sample delay error.)
            if constexpr (Method == InterpMethod::Cubic || Method == InterpMethod::Hermite)
                return interpolateHermite(s[0], s[1], s[2], s[3], frac);
            else if constexpr (Method == InterpMethod::Lagrange)
                return interpolateLagrange(s[0], s[1], s[2], s[3], frac);
        }
    }

    [[nodiscard]] int getCapacity() const noexcept { return capacity_; }
    [[nodiscard]] const T* data() const noexcept { return buffer_.get(); }
    [[nodiscard]] int getWritePosition() const noexcept { return writePos_; }

private:
    struct AlignedDeleter {
        void operator()(T* ptr) const {
            #if defined(_MSC_VER)
                _aligned_free(ptr);
            #else
                std::free(ptr);
            #endif
        }
    };

    std::unique_ptr<T[], AlignedDeleter> buffer_;
    int capacity_ = 0;
    int mask_ = 0;
    int writePos_ = 0;
};

} // namespace dspark