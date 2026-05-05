// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file PitchDetector.h
 * @brief Real-time monophonic pitch detection using the YIN algorithm.
 *
 * Implements the YIN autocorrelation method (de Cheveigné & Kawahara, 2002).
 * Refactored for DSPark: Uses mirrored-buffer technique for 100% contiguous
 * memory reads, enabling flawless auto-vectorization (SIMD) on hot paths.
 * Fully lock-free and thread-safe for UI/Audio thread communication.
 */

#include <atomic>
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class PitchDetector
 * @brief Thread-safe, SIMD-optimized YIN pitch detector.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class PitchDetector
{
public:
    /**
     * @brief Prepares the detector and allocates internal structures.
     * * Must be called before audio processing begins. Zero allocations
     * happen after this point.
     *
     * @param sampleRate The system sample rate in Hz.
     * @param windowSize Analysis window size. Must be even (default: 2048).
     * @param hopSize    Number of samples between detections (overlap). Lower is smoother.
     */
    void prepare(double sampleRate, int windowSize = 2048, int hopSize = 512)
    {
        sampleRate_ = sampleRate;
        windowSize_ = windowSize;
        halfWindow_ = windowSize / 2;
        hopSize_    = std::clamp(hopSize, 1, windowSize);

        // Mirrored buffer technique: size is 2x windowSize.
        // Guarantees continuous memory layout for SIMD without modulo operations.
        buffer_.assign(static_cast<size_t>(windowSize_ * 2), T(0));
        yinBuffer_.assign(static_cast<size_t>(halfWindow_), T(0));

        writePos_ = 0;
        samplesSinceLastDetect_ = 0;

        frequency_.store(T(0), std::memory_order_relaxed);
        confidence_.store(T(0), std::memory_order_relaxed);
    }

    /**
     * @brief Pushes audio samples into the analysis buffer.
     * * Automatically triggers pitch detection when the hop size is reached.
     * 100% safe to run in the real-time audio thread (no locks, no allocations).
     *
     * @param samples Span of input audio data (mono).
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        for (const T sample : samples)
        {
            // Write into mirrored buffer
            buffer_[static_cast<size_t>(writePos_)] = sample;
            buffer_[static_cast<size_t>(writePos_ + windowSize_)] = sample;

            writePos_++;
            if (writePos_ >= windowSize_)
            {
                writePos_ = 0;
            }

            samplesSinceLastDetect_++;
            if (samplesSinceLastDetect_ >= hopSize_)
            {
                detect();
                samplesSinceLastDetect_ = 0;
            }
        }
    }

    /** @brief Returns the detected frequency in Hz safely from any thread. */
    [[nodiscard]] T getFrequencyHz() const noexcept 
    { 
        return frequency_.load(std::memory_order_relaxed); 
    }

    /** @brief Returns the detection confidence [0.0 - 1.0] safely from any thread. */
    [[nodiscard]] T getConfidence() const noexcept 
    { 
        return confidence_.load(std::memory_order_relaxed); 
    }

    /** @brief Returns nearest MIDI note (69 = A4), or -1 if unvoiced. */
    [[nodiscard]] int getMidiNote() const noexcept
    {
        const T freq = getFrequencyHz();
        if (freq <= T(0)) return -1;
        return static_cast<int>(std::round(T(69) + T(12) * std::log2(freq / T(440))));
    }

    /** @brief Returns cent offset from the nearest MIDI note [-50, +50]. */
    [[nodiscard]] T getCentsOffset() const noexcept
    {
        const T freq = getFrequencyHz();
        if (freq <= T(0)) return T(0);
        T midiExact = T(69) + T(12) * std::log2(freq / T(440));
        return (midiExact - std::round(midiExact)) * T(100);
    }

    /** @brief Sets sensitivity threshold (0.01 - 0.5). Lower is stricter. */
    void setThreshold(T threshold) noexcept
    {
        threshold_ = std::clamp(threshold, T(0.01), T(0.5));
    }

    /** @brief Resets state buffers. Not thread-safe with pushSamples(). */
    void reset() noexcept
    {
        std::fill(buffer_.begin(), buffer_.end(), T(0));
        writePos_ = 0;
        samplesSinceLastDetect_ = 0;
        frequency_.store(T(0), std::memory_order_relaxed);
        confidence_.store(T(0), std::memory_order_relaxed);
    }

private:
    void detect() noexcept
    {
        // Obtain a perfectly contiguous block of memory representing the current window.
        // writePos_ points to the oldest sample in the mirrored buffer.
        const T* currentWindow = &buffer_[static_cast<size_t>(writePos_)];

        // Silence check / Energy calculation on contiguous memory
        T energy = T(0);
        for (int i = 0; i < windowSize_; ++i) {
            energy += currentWindow[i] * currentWindow[i];
        }

        if (energy < T(1e-10))
        {
            frequency_.store(T(0), std::memory_order_relaxed);
            confidence_.store(T(0), std::memory_order_relaxed);
            return;
        }

        // CMND Computation - now highly SIMD vectorizable
        yinBuffer_[0] = T(1);
        T runningSum = T(0);

        for (int tau = 1; tau < halfWindow_; ++tau)
        {
            T sum = T(0);
            
            // Inner hot-loop: No modulo, pure contiguous memory arrays
            for (int j = 0; j < halfWindow_; ++j)
            {
                T diff = currentWindow[j] - currentWindow[j + tau];
                sum += diff * diff;
            }

            runningSum += sum;
            yinBuffer_[static_cast<size_t>(tau)] = 
                (runningSum > T(0)) ? sum * static_cast<T>(tau) / runningSum : T(0);
        }

        // Search for dip below threshold
        int tauEstimate = -1;
        for (int tau = 2; tau < halfWindow_; ++tau)
        {
            if (yinBuffer_[static_cast<size_t>(tau)] < threshold_)
            {
                while (tau + 1 < halfWindow_ && 
                       yinBuffer_[static_cast<size_t>(tau + 1)] < yinBuffer_[static_cast<size_t>(tau)])
                {
                    ++tau;
                }
                tauEstimate = tau;
                break;
            }
        }

        if (tauEstimate < 0)
        {
            frequency_.store(T(0), std::memory_order_relaxed);
            confidence_.store(T(0), std::memory_order_relaxed);
            return;
        }

        // Sub-sample precision
        T betterTau = parabolicInterp(tauEstimate);
        T finalConfidence = std::clamp(T(1) - yinBuffer_[static_cast<size_t>(tauEstimate)], T(0), T(1));

        frequency_.store(static_cast<T>(sampleRate_) / betterTau, std::memory_order_relaxed);
        confidence_.store(finalConfidence, std::memory_order_relaxed);
    }

    [[nodiscard]] T parabolicInterp(int tau) const noexcept
    {
        if (tau < 1 || tau >= halfWindow_ - 1)
            return static_cast<T>(tau);

        T s0 = yinBuffer_[static_cast<size_t>(tau - 1)];
        T s1 = yinBuffer_[static_cast<size_t>(tau)];
        T s2 = yinBuffer_[static_cast<size_t>(tau + 1)];

        T denom = s0 - T(2) * s1 + s2;
        
        // Prevent Divide by Zero on flat local minimums
        if (std::abs(denom) < T(1e-12))
            return static_cast<T>(tau);

        T adjustment = (s0 - s2) / (T(2) * denom);
        return static_cast<T>(tau) + adjustment;
    }

    double sampleRate_ = 44100.0;
    int windowSize_ = 2048;
    int halfWindow_ = 1024;
    int hopSize_ = 512;
    int writePos_ = 0;
    int samplesSinceLastDetect_ = 0;
    T threshold_ = T(0.10);

    // Thread-safe outputs
    std::atomic<T> frequency_{T(0)};
    std::atomic<T> confidence_{T(0)};

    // Mirrored buffer for 100% SIMD-friendly contiguous memory
    std::vector<T> buffer_;     // Size: 2 * windowSize_
    std::vector<T> yinBuffer_;  // Size: halfWindow_
};

} // namespace dspark