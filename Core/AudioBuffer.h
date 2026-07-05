// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file AudioBuffer.h
 * @brief Owning audio buffer and non-owning view for real-time DSP processing.
 *
 * Provides two complementary classes:
 *
 * - **AudioBufferView<T>**: A lightweight, non-owning view over channel pointers.
 * Processors receive this type. It is trivially copyable and allows implicit
 * conversion from mutable (T) to constant (const T) views.
 *
 * - **AudioBuffer<T, MaxChannels>**: An owning buffer with a single contiguous,
 * 32-byte aligned memory allocation. Ensures SIMD padding is correctly zeroed 
 * to avoid denormal propagation during vectorized over-reads.
 *
 * Dependencies: SimdOps.h and the C++20 standard library.
 */

#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

namespace dspark {

// ============================================================================
// AudioBufferView -- Non-owning view
// ============================================================================

/**
 * @class AudioBufferView
 * @brief Non-owning view over audio channel data.
 *
 * @tparam T Sample type (typically `float`, `double`, `const float`, or `const double`).
 * @tparam MaxViewChannels Compile-time limit for the internal pointer array.
 */
template <typename T, int MaxViewChannels = 16>
class AudioBufferView
{
public:
    /** @brief Creates an empty view (0 channels, 0 samples). */
    AudioBufferView() noexcept = default;

    /**
     * @brief Constructs a view from an array of channel pointers safely.
     * @tparam U Type of the source pointers (must be convertible to T*).
     * @param channelPtrs Array of pointers, one per channel.
     * @param numChannels Number of active channels.
     * @param numSamples  Number of samples per channel.
     */
    template <typename U>
    AudioBufferView(U* const* channelPtrs, int numChannels, int numSamples) noexcept
    {
        static_assert(std::is_convertible_v<U*, T*>, "Pointer type U* must be convertible to T*");
        assert(numChannels >= 0 && numChannels <= MaxViewChannels);

        // Release-safe clamp: an out-of-range channel count must never write
        // past the fixed pointer array (assert-only protection vanishes in
        // release builds).
        numChannels_ = std::clamp(numChannels, 0, MaxViewChannels);
        numSamples_  = numSamples >= 0 ? numSamples : 0;

        for (int ch = 0; ch < numChannels_; ++ch)
            channels_[ch] = channelPtrs[ch];
    }

    /**
     * @brief Converting constructor allowing mutable to const view conversions.
     *
     * Enables passing AudioBufferView<float> to functions expecting
     * AudioBufferView<const float>.
     *
     * @tparam U Source sample type.
     * @param other The view to convert from.
     */
    template <typename U>
    AudioBufferView(const AudioBufferView<U, MaxViewChannels>& other) noexcept
        : numChannels_(other.getNumChannels()), numSamples_(other.getNumSamples())
    {
        static_assert(std::is_convertible_v<U*, T*>, "Cannot convert view (e.g., const to non-const)");
        
        for (int ch = 0; ch < numChannels_; ++ch)
            channels_[ch] = other.getChannel(ch);
    }

    // -- Accessors -----------------------------------------------------------

    /** @brief Returns a pointer to the sample data for the given channel. */
    [[nodiscard]] T* getChannel(int ch) const noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channels_[ch];
    }

    /** @brief Returns the number of channels in this view. */
    [[nodiscard]] int getNumChannels() const noexcept { return numChannels_; }

    /** @brief Returns the number of samples per channel. */
    [[nodiscard]] int getNumSamples() const noexcept { return numSamples_; }

    // -- Sub-views -----------------------------------------------------------

    /**
     * @brief Returns a zero-copy sub-view starting at an offset.
     * @param startSample Offset from the beginning of each channel.
     * @param length      Number of samples in the sub-view.
     * @return A new AudioBufferView referencing the sub-range.
     */
    [[nodiscard]] AudioBufferView getSubView(int startSample, int length) const noexcept
    {
        assert(startSample >= 0);
        assert(length >= 0);
        assert(startSample + length <= numSamples_);

        AudioBufferView sub;
        sub.numChannels_ = numChannels_;
        sub.numSamples_  = length;
        for (int ch = 0; ch < numChannels_; ++ch)
            sub.channels_[ch] = channels_[ch] + startSample;

        return sub;
    }

    // -- Operations ----------------------------------------------------------

    /** @brief Fills the valid sample range of all channels with zeros. */
    void clear() const noexcept
    {
        // A hard error, not a silent no-op: matching applyGain/copyFrom so a
        // clear() on a read-only view can never be mistaken for a real one.
        static_assert(!std::is_const_v<T>, "Cannot clear a const view");
        const auto bytes = static_cast<std::size_t>(numSamples_) * sizeof(T);
        for (int ch = 0; ch < numChannels_; ++ch)
            std::memset(channels_[ch], 0, bytes);
    }

    /**
     * @brief Safely copies samples from a source view into this view.
     *
     * For same-type trivially-copyable samples the copy goes through
     * std::memmove, which is guaranteed safe even when the ranges overlap.
     * Channel count and length are trimmed to the smaller of the two views.
     *
     * @tparam U Source sample type.
     * @tparam M Source view channel capacity (any capacity is accepted).
     * @param src Source view to copy from.
     */
    template <typename U, int M>
    void copyFrom(const AudioBufferView<U, M>& src) const noexcept
    {
        static_assert(!std::is_const_v<T>, "Cannot copy into a const view");

        const int chCount  = std::min(numChannels_, src.getNumChannels());
        const int nSamples = std::min(numSamples_, src.getNumSamples());

        for (int ch = 0; ch < chCount; ++ch)
        {
            T* dst     = channels_[ch];
            const U* s = src.getChannel(ch);

            if constexpr (std::is_same_v<std::remove_const_t<U>, T>)
            {
                std::memmove(dst, s, static_cast<std::size_t>(nSamples) * sizeof(T));
            }
            else
            {
                for (int i = 0; i < nSamples; ++i)
                    dst[i] = static_cast<T>(s[i]);
            }
        }
    }

    /**
     * @brief Adds samples from a source view into this view with optional gain.
     *
     * Channel count and length are trimmed to the smaller of the two views.
     * Unlike copyFrom(), the source must NOT overlap this view: the mix goes
     * through restrict-qualified SIMD kernels (see SimdOps.h).
     *
     * @tparam U Source sample type.
     * @tparam M Source view channel capacity (any capacity is accepted).
     * @param src  Source view.
     * @param gain Scaling factor.
     */
    template <typename U, int M>
    void addFrom(const AudioBufferView<U, M>& src, T gain = T(1)) const noexcept
    {
        static_assert(!std::is_const_v<T>, "Cannot add into a const view");

        const int chCount  = std::min(numChannels_, src.getNumChannels());
        const int nSamples = std::min(numSamples_, src.getNumSamples());

        for (int ch = 0; ch < chCount; ++ch)
        {
            T* dst     = channels_[ch];
            const U* s = src.getChannel(ch);

            if constexpr (std::is_same_v<std::remove_const_t<U>, T>)
            {
                simd::addWithGain(dst, s, gain, nSamples);
            }
            else
            {
                for (int i = 0; i < nSamples; ++i)
                    dst[i] += static_cast<T>(s[i]) * gain;
            }
        }
    }

    /**
     * @brief Multiplies all samples in all valid channels by a gain factor.
     * @param gain Scaling factor.
     */
    void applyGain(T gain) const noexcept
    {
        static_assert(!std::is_const_v<T>, "Cannot apply gain to a const view");
        for (int ch = 0; ch < numChannels_; ++ch)
            simd::applyGain(channels_[ch], gain, numSamples_);
    }

    /**
     * @brief Returns the peak absolute sample value across all channels.
     * @return Peak magnitude (>= 0).
     */
    [[nodiscard]] std::remove_const_t<T> getPeakLevel() const noexcept
    {
        // remove_const keeps the accumulator (and the returned value)
        // assignable when T is a const sample type: this read-only query
        // must work on read-only views.
        using Value = std::remove_const_t<T>;
        Value peak = Value(0);
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            const Value chPeak = simd::peakLevel(channels_[ch], numSamples_);
            if (chPeak > peak) peak = chPeak;
        }
        return peak;
    }

private:
    std::array<T*, MaxViewChannels> channels_ {};
    int numChannels_ = 0;
    int numSamples_  = 0;
};

// The view is part of the hot-path calling convention: it must stay cheap to
// copy by value. Guard the promise made in the file header.
static_assert(std::is_trivially_copyable_v<AudioBufferView<float>>,
              "AudioBufferView must remain trivially copyable");
static_assert(std::is_trivially_copyable_v<AudioBufferView<const float>>,
              "AudioBufferView must remain trivially copyable");

// ============================================================================
// AudioBuffer -- Owning, SIMD-aligned buffer
// ============================================================================

/**
 * @class AudioBuffer
 * @brief Owning audio buffer with contiguous, 32-byte aligned storage.
 *
 * @tparam T Sample type (float or double).
 * @tparam MaxChannels Maximum supported channels.
 */
template <typename T, int MaxChannels = 16>
class AudioBuffer
{
    static constexpr std::size_t kAlignment = 32;
    static_assert(std::has_single_bit(kAlignment), "Alignment must be a power of two");
    static_assert(std::is_same_v<T, std::remove_cv_t<T>> && std::is_trivially_copyable_v<T>,
                  "AudioBuffer requires a non-const, trivially copyable sample type");

public:
    AudioBuffer() = default;

    ~AudioBuffer() { deallocate(); }

    AudioBuffer(const AudioBuffer&) = delete;
    AudioBuffer& operator=(const AudioBuffer&) = delete;

    /** @brief C++20 idiomatic move constructor. */
    AudioBuffer(AudioBuffer&& other) noexcept
        : rawData_(std::exchange(other.rawData_, nullptr))
        , channelPtrs_(std::exchange(other.channelPtrs_, {}))
        , numChannels_(std::exchange(other.numChannels_, 0))
        , numSamples_(std::exchange(other.numSamples_, 0))
        , allocatedBytes_(std::exchange(other.allocatedBytes_, 0))
        , strideBytes_(std::exchange(other.strideBytes_, 0))
    {}

    /** @brief C++20 idiomatic move assignment. */
    AudioBuffer& operator=(AudioBuffer&& other) noexcept
    {
        if (this != &other)
        {
            deallocate();
            rawData_        = std::exchange(other.rawData_, nullptr);
            channelPtrs_    = std::exchange(other.channelPtrs_, {});
            numChannels_    = std::exchange(other.numChannels_, 0);
            numSamples_     = std::exchange(other.numSamples_, 0);
            allocatedBytes_ = std::exchange(other.allocatedBytes_, 0);
            strideBytes_    = std::exchange(other.strideBytes_, 0);
        }
        return *this;
    }

    // -- Allocation ----------------------------------------------------------

    /**
     * @brief Allocates the buffer for the given dimensions.
     *
     * Existing contents are NOT preserved: the buffer (including its SIMD
     * padding) is zeroed after every call. Memory is only re-allocated when
     * the request exceeds the current capacity; shrinking reuses it. May
     * throw std::bad_alloc; the buffer stays coherent and reusable after.
     *
     * @param numChannels Number of active channels (must be <= MaxChannels).
     * @param numSamples  Number of audio samples per channel.
     */
    void resize(int numChannels, int numSamples)
    {
        assert(numChannels >= 0 && numChannels <= MaxChannels);
        assert(numSamples >= 0);

        // Release-safe clamps: out-of-range requests must never overflow the
        // fixed channel-pointer array or produce a negative allocation size.
        numChannels = std::clamp(numChannels, 0, MaxChannels);
        if (numSamples < 0) numSamples = 0;

        const auto samplesBytes = static_cast<std::size_t>(numSamples) * sizeof(T);
        const auto stride       = alignUp(samplesBytes, kAlignment);
        const auto totalBytes   = stride * static_cast<std::size_t>(numChannels);

        // Smart re-allocation: reuse memory if requested size fits in capacity
        if (totalBytes > allocatedBytes_)
        {
            deallocate();
            rawData_ = static_cast<T*>(::operator new(totalBytes, std::align_val_t(kAlignment)));
            allocatedBytes_ = totalBytes;
        }

        numChannels_ = numChannels;
        numSamples_  = numSamples;
        strideBytes_ = stride;

        auto* base = reinterpret_cast<char*>(rawData_);
        
        // Map active channels
        for (int ch = 0; ch < numChannels; ++ch)
            channelPtrs_[ch] = reinterpret_cast<T*>(base + stride * static_cast<std::size_t>(ch));
            
        // Nullify unused channels for safety
        for (int ch = numChannels; ch < MaxChannels; ++ch)
            channelPtrs_[ch] = nullptr;

        clear(); // Guarantees padding zeroing for SIMD safety
    }

    // -- View creation -------------------------------------------------------

    /** @brief Returns a non-owning mutable view of this buffer. */
    [[nodiscard]] AudioBufferView<T> toView() noexcept
    {
        return { channelPtrs_.data(), numChannels_, numSamples_ };
    }

    /** @brief Returns a non-owning const view of this buffer. */
    [[nodiscard]] AudioBufferView<const T> toView() const noexcept
    {
        return { channelPtrs_.data(), numChannels_, numSamples_ };
    }

    // -- Accessors -----------------------------------------------------------

    /** @brief Returns a pointer to the sample data. */
    [[nodiscard]] T* getChannel(int ch) noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channelPtrs_[ch];
    }

    /** @brief Const overload. */
    [[nodiscard]] const T* getChannel(int ch) const noexcept
    {
        assert(ch >= 0 && ch < numChannels_);
        return channelPtrs_[ch];
    }

    /** @brief Returns the number of active channels. */
    [[nodiscard]] int getNumChannels() const noexcept { return numChannels_; }

    /** @brief Returns the number of samples per channel. */
    [[nodiscard]] int getNumSamples() const noexcept { return numSamples_; }

    // -- Operations ----------------------------------------------------------

    /**
     * @brief Clears all channels, including SIMD alignment padding.
     * Ensures any SIMD over-read will consume pure zeros (no denormals).
     */
    void clear() noexcept
    {
        // Degenerate sizes (0 samples or 0 channels) own no storage at all;
        // memset on a null pointer is undefined even with a zero length.
        if (rawData_ == nullptr) return;
        for (int ch = 0; ch < numChannels_; ++ch)
            std::memset(channelPtrs_[ch], 0, strideBytes_);
    }

private:
    [[nodiscard]] static constexpr std::size_t alignUp(std::size_t value, std::size_t alignment) noexcept
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    void deallocate() noexcept
    {
        if (rawData_)
        {
            ::operator delete(rawData_, std::align_val_t(kAlignment));
            rawData_ = nullptr;
        }
        // Keep the capacity coherent with the (now absent) allocation. If
        // resize()'s operator new throws after this call, a stale capacity
        // would make a later, smaller resize() skip its allocation and build
        // channel pointers over a null base.
        allocatedBytes_ = 0;
    }

    T* rawData_        = nullptr;
    std::array<T*, MaxChannels> channelPtrs_    {};
    int                         numChannels_    = 0;
    int                         numSamples_     = 0;
    std::size_t                 allocatedBytes_ = 0;
    std::size_t                 strideBytes_    = 0;
};

} // namespace dspark