// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

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
 * Dependencies: C++20 standard library (<array>, <bit>, <cstddef>, <cstring>, <new>, <utility>).
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
// AudioBufferView — Non-owning view
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
        : numChannels_(numChannels), numSamples_(numSamples)
    {
        static_assert(std::is_convertible_v<U*, T*>, "Pointer type U* must be convertible to T*");
        assert(numChannels <= MaxViewChannels);
        
        for (int ch = 0; ch < numChannels; ++ch)
            channels_[ch] = channelPtrs[ch];
    }

    /**
     * @brief Converting constructor allowing mutable to const view conversions.
     * * Enables passing AudioBufferView<float> to functions expecting AudioBufferView<const float>.
     * * @tparam U Source sample type.
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
        if constexpr (!std::is_const_v<T>)
        {
            const auto bytes = static_cast<std::size_t>(numSamples_) * sizeof(T);
            for (int ch = 0; ch < numChannels_; ++ch)
                std::memset(channels_[ch], 0, bytes);
        }
    }

    /**
     * @brief Safely copies samples from a source view into this view.
     * Utilizes std::copy_n to prevent UB in case of memory aliasing.
     * * @tparam U Source sample type.
     * @param src Source view to copy from.
     */
    template <typename U>
    void copyFrom(const AudioBufferView<U>& src) const noexcept
    {
        static_assert(!std::is_const_v<T>, "Cannot copy into a const view");
        
        const int chCount  = std::min(numChannels_, src.getNumChannels());
        const int nSamples = std::min(numSamples_, src.getNumSamples());

        for (int ch = 0; ch < chCount; ++ch)
        {
            T* dst     = channels_[ch];
            const U* s = src.getChannel(ch);

            // std::copy_n handles trivial types by forwarding to memmove safely 
            // if ranges overlap, preventing UB associated with raw memcpy.
            std::copy_n(s, nSamples, dst);
        }
    }

    /**
     * @brief Adds samples from a source view into this view with optional gain.
     * @tparam U Source sample type.
     * @param src  Source view.
     * @param gain Scaling factor.
     */
    template <typename U>
    void addFrom(const AudioBufferView<U>& src, T gain = T(1)) const noexcept
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
    [[nodiscard]] T getPeakLevel() const noexcept
    {
        T peak = T(0);
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            const T chPeak = simd::peakLevel(channels_[ch], numSamples_);
            if (chPeak > peak) peak = chPeak;
        }
        return peak;
    }

private:
    // Marked mutable to allow const views to correctly expose const pointers
    mutable std::array<T*, MaxViewChannels> channels_ {};
    int numChannels_ = 0;
    int numSamples_  = 0;
};

// ============================================================================
// AudioBuffer — Owning, SIMD-aligned buffer
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
     * @param numChannels Number of active channels (must be <= MaxChannels).
     * @param numSamples  Number of audio samples per channel.
     */
    void resize(int numChannels, int numSamples)
    {
        assert(numChannels >= 0 && numChannels <= MaxChannels);
        assert(numSamples >= 0);

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
    }

    T* rawData_        = nullptr;
    std::array<T*, MaxChannels> channelPtrs_    {};
    int                         numChannels_    = 0;
    int                         numSamples_     = 0;
    std::size_t                 allocatedBytes_ = 0;
    std::size_t                 strideBytes_    = 0;
};

} // namespace dspark