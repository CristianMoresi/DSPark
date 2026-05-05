// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Goertzel.h
 * @brief Goertzel algorithm for efficient single-frequency detection.
 *
 * The Goertzel algorithm computes the magnitude and phase at a specific 
 * frequency in O(N) time. It acts as an extremely narrow bandpass filter 
 * evaluated over a specific block size (N).
 * * @note Because it uses a rectangular window inherently, if the target frequency 
 * does not align exactly with a harmonic of the window (fs/N), it will suffer 
 * from spectral leakage. For precise offline analysis, consider pre-windowing 
 * your input data (e.g., Hann window).
 *
 * Dependencies: DspMath.h, <cassert>
 */

#include "../Core/DspMath.h"

#include <cmath>
#include <numbers>
#include <cassert>

namespace dspark {

/**
 * @class Goertzel
 * @brief Single-frequency magnitude detector using the Goertzel algorithm.
 *
 * Designed for continuous streaming or block-based real-time analysis. 
 * State is unified: you can mix `processBlock` and `pushSample` safely.
 *
 * @tparam T Sample type (float or double). Requires dspark::FloatType concept.
 */
template <FloatType T>
class Goertzel
{
public:
    /**
     * @brief Prepares the detector for a specific frequency.
     *
     * @param sampleRate Sample rate in Hz. Must be > 0.
     * @param targetFreqHz The frequency to detect. Must be < sampleRate / 2.
     * @param blockSize Number of samples per analysis window. Must be > 0.
     */
    void prepare(double sampleRate, double targetFreqHz, int blockSize) noexcept
    {
        assert(sampleRate > 0.0 && "Sample rate must be positive");
        assert(blockSize > 0 && "Block size must be strictly positive");
        assert(targetFreqHz >= 0.0 && targetFreqHz <= (sampleRate * 0.5) && "Frequency must be within Nyquist limit");

        sampleRate_ = sampleRate;
        targetFreq_ = targetFreqHz;
        blockSize_ = blockSize;

        // Exact target frequency omega calculation (Generalized Goertzel)
        double omega = 2.0 * std::numbers::pi * targetFreq_ / sampleRate_;
        
        coeff_ = static_cast<T>(2.0 * std::cos(omega));
        cosOmega_ = static_cast<T>(std::cos(omega));
        sinOmega_ = static_cast<T>(std::sin(omega));

        // Magnitude normalization factor: 2/N for standard waves, 1/N for DC
        normalisationFactor_ = (targetFreqHz > 0.001) 
                                ? static_cast<T>(2.0 / blockSize_) 
                                : static_cast<T>(1.0 / blockSize_);

        reset();
    }

    /**
     * @brief Processes a block of audio samples, updating the internal state.
     * * This method accumulates samples. To know if a full analysis block has 
     * been completed, check `isReady()` after processing.
     *
     * @param data Pointer to the audio samples.
     * @param numSamples Number of samples in the buffer.
     */
    void processBlock(const T* data, int numSamples) noexcept
    {
        if (numSamples <= 0) return;

        // Bring state to local variables for compiler optimization (register allocation)
        T s1 = s1_;
        T s2 = s2_;
        const T coeff = coeff_;
        
        for (int i = 0; i < numSamples; ++i)
        {
            T s0 = data[i] + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;

            if (++sampleCount_ >= blockSize_)
            {
                computeResult(s1, s2);
                s1 = T(0);
                s2 = T(0);
                sampleCount_ = 0;
            }
        }

        // Save local state back to member variables
        s1_ = s1;
        s2_ = s2;
    }

    /**
     * @brief Feeds a single sample into the running Goertzel computation.
     *
     * @param sample Input audio sample.
     * @return True if a new magnitude calculation was just completed (N samples reached).
     */
    bool pushSample(T sample) noexcept
    {
        T s0 = sample + coeff_ * s1_ - s2_;
        s2_ = s1_;
        s1_ = s0;

        if (++sampleCount_ >= blockSize_)
        {
            computeResult(s1_, s2_);
            s1_ = T(0);
            s2_ = T(0);
            sampleCount_ = 0;
            return true;
        }
        return false;
    }

    /**
     * @brief Manually forces the computation of the result before N samples are reached.
     * * @warning Evaluating before reaching the configured blockSize will result in 
     * inaccurate magnitude and phase due to incomplete integration and incorrect scaling.
     */
    void forceCompute() noexcept
    {
        computeResult(s1_, s2_);
        s1_ = T(0);
        s2_ = T(0);
        sampleCount_ = 0;
    }

    // -- Results ------------------------------------------------------------------

    /**
     * @brief Checks if a new result has been computed.
     * * Calling this will return true only once after an analysis block completes,
     * resetting the flag internally.
     * * @return True if new data is available.
     */
    [[nodiscard]] bool checkNewResultAvailable() noexcept
    {
        if (hasNewResult_)
        {
            hasNewResult_ = false;
            return true;
        }
        return false;
    }

    /**
     * @brief Returns the magnitude at the target frequency (linear scale).
     * @return Magnitude (Amplitude of the detected frequency).
     */
    [[nodiscard]] T getMagnitude() const noexcept
    {
        return std::sqrt(real_ * real_ + imag_ * imag_);
    }

    /**
     * @brief Returns the power at the target frequency.
     * @return Power (Magnitude squared).
     */
    [[nodiscard]] T getPower() const noexcept
    {
        return real_ * real_ + imag_ * imag_;
    }

    /**
     * @brief Returns the magnitude in decibels.
     * @return Magnitude in dB (returns lowest bound if magnitude is 0).
     */
    [[nodiscard]] T getMagnitudeDb() const noexcept
    {
        // Assuming dspark::gainToDecibels handles 0.0 appropriately
        return gainToDecibels(getMagnitude());
    }

    /**
     * @brief Returns the phase angle at the target frequency.
     * @return Phase in radians in the range [-pi, pi].
     */
    [[nodiscard]] T getPhase() const noexcept
    {
        return std::atan2(imag_, real_);
    }

    /**
     * @brief Returns the currently configured target frequency.
     * @return Frequency in Hz.
     */
    [[nodiscard]] double getTargetFrequency() const noexcept { return targetFreq_; }

    /**
     * @brief Resets the internal IIR state and counters to zero.
     * * Retains configured sample rate, frequency, and block size.
     */
    void reset() noexcept
    {
        s1_ = T(0);
        s2_ = T(0);
        sampleCount_ = 0;
        real_ = T(0);
        imag_ = T(0);
        hasNewResult_ = false;
    }

private:
    /**
     * @brief Computes real and imaginary parts and applies correct normalization.
     */
    void computeResult(T s1, T s2) noexcept
    {
        real_ = (s1 - s2 * cosOmega_) * normalisationFactor_;
        imag_ = (s2 * sinOmega_) * normalisationFactor_;
        hasNewResult_ = true;
    }

    double sampleRate_ = 48000.0;
    double targetFreq_ = 440.0;
    int blockSize_ = 2048;

    T coeff_ = T(0);
    T cosOmega_ = T(0);
    T sinOmega_ = T(0);
    T normalisationFactor_ = T(0);

    // Streaming state
    T s1_ = T(0);
    T s2_ = T(0);
    int sampleCount_ = 0;

    // Results
    T real_ = T(0);
    T imag_ = T(0);
    bool hasNewResult_ = false;
};

} // namespace dspark