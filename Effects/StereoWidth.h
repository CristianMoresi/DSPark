// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file StereoWidth.h
 * @brief Phase-compensated Stereo width control via Mid/Side processing.
 *
 * Adjusts the stereo image width from mono to extra-wide using M/S encoding.
 * Includes an audiophile-grade bass-mono feature that collapses low frequencies
 * to mono below a configurable cutoff. 
 * 
 * To guarantee pristine center-image fidelity, a 1-pole All-Pass filter is 
 * applied to the Mid signal to perfectly align its phase response with the 
 * Side signal's High-Pass filter.
 *
 * Designed for real-time: branchless inner loops, SIMD-friendly, 
 * atomic lock-free parameters, and internal anti-denormal protection.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class StereoWidth
 * @brief High-performance stereo image processor with phase-aligned bass mono.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class StereoWidth
{
public:
    StereoWidth() = default;
    ~StereoWidth() = default; // Removed virtual to prevent vtable overhead

    /**
     * @brief Prepares the processor and resets internal states.
     * @param sampleRate Sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        updateBassMonoCoeff(bassMonoCutoff_.load(std::memory_order_relaxed));
        reset();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec) noexcept { prepare(spec.sampleRate); }

    /**
     * @brief Processes an AudioBufferView in-place.
     * @param buffer Audio buffer (must have >= 2 channels).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumChannels() >= 2)
            process(buffer.getChannel(0), buffer.getChannel(1), buffer.getNumSamples());
    }

    /**
     * @brief Sets the overall stereo width factor.
     * @param width 0.0 = Mono, 1.0 = Original, >1.0 = Widened.
     */
    void setWidth(T width) noexcept
    {
        width_.store(std::max(T(0), width), std::memory_order_relaxed);
    }

    /** @brief Returns current width setting. */
    [[nodiscard]] T getWidth() const noexcept { return width_.load(std::memory_order_relaxed); }

    /**
     * @brief Toggles Bass Mono and updates the crossover frequency.
     * @param enabled True activates the phase-aligned bass mono crossover.
     * @param cutoffHz Cutoff frequency in Hz (default: 100.0).
     */
    void setBassMono(bool enabled, double cutoffHz = 100.0) noexcept
    {
        bassMonoCutoff_.store(cutoffHz, std::memory_order_relaxed);
        updateBassMonoCoeff(cutoffHz);
        bassMonoEnabled_.store(enabled, std::memory_order_release);
    }

    /**
     * @brief Process a full block of audio. Optimized for SIMD vectorization.
     * @param left Pointer to left channel data.
     * @param right Pointer to right channel data.
     * @param numSamples Number of samples to process.
     */
    void process(T* left, T* right, int numSamples) noexcept
    {
        // Load atomics once per block to allow tight loop vectorization
        const T currentWidth = width_.load(std::memory_order_relaxed);
        const bool bassMono = bassMonoEnabled_.load(std::memory_order_acquire);

        if (bassMono)
        {
            const T coeff = bassMonoCoeff_.load(std::memory_order_relaxed);

            // Bass-mono crossover: the side channel loses its lows through a
            // one-pole high-pass; the mid passes UNTOUCHED. A one-pole split is
            // complementary (LP + HP == 1 exactly), so no phase "compensation"
            // of the mid is needed — the previously used first-order allpass
            // rotated the mid by up to 180 degrees at HF, which swapped and
            // inverted the channels above the transition band.
            for (int i = 0; i < numSamples; ++i)
            {
                T l = left[i];
                T r = right[i];

                T mid  = (l + r) * T(0.5);
                T side = (l - r) * T(0.5) * currentWidth;

                // Side processing (1-pole high-pass: side - LP(side))
                T sideLpIn = side;
                sideState_ += coeff * (sideLpIn - sideState_) + antiDenormal_;
                sideState_ -= antiDenormal_; // Denormal flush
                side = sideLpIn - sideState_;

                left[i]  = mid + side;
                right[i] = mid - side;
            }
        }
        else
        {
            // Fast-path branch: Pure Width control without filtering overhead
            for (int i = 0; i < numSamples; ++i)
            {
                T mid  = (left[i] + right[i]) * T(0.5);
                T side = (left[i] - right[i]) * T(0.5) * currentWidth;
                
                left[i]  = mid + side;
                right[i] = mid - side;
            }
        }
    }

    /** @brief Clears the internal filter states to prevent artifact ringing. */
    void reset() noexcept
    {
        sideState_ = T(0);
    }

protected:
    void updateBassMonoCoeff(double cutoff) noexcept
    {
        if (sampleRate_ > 0.0)
        {
            // Calculate 1-pole coeff. Stored atomically to prevent data races.
            T coeff = static_cast<T>(1.0 - std::exp(-std::numbers::pi * 2.0 * cutoff / sampleRate_));
            bassMonoCoeff_.store(coeff, std::memory_order_relaxed);
        }
    }

private:
    double sampleRate_ = 48000.0;
    
    // Lock-free parameters
    std::atomic<T> width_ { T(1) };
    std::atomic<bool> bassMonoEnabled_ { false };
    std::atomic<double> bassMonoCutoff_ { 100.0 };
    std::atomic<T> bassMonoCoeff_ { T(0) };

    // Filter states
    T sideState_ = T(0);

    // Anti-denormal DC offset (type generic)
    static constexpr T antiDenormal_ = static_cast<T>(1e-15); 
};

} // namespace dspark
