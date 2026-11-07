// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Goertzel.h
 * @brief Goertzel algorithm for efficient single-frequency detection.
 *
 * The Goertzel algorithm computes the magnitude and phase at a specific
 * frequency in O(N) time. It acts as an extremely narrow bandpass filter
 * evaluated over a specific block size (N).
 *
 * @note Because it uses a rectangular window inherently, if the target
 * frequency does not align exactly with a harmonic of the window (fs/N), it
 * will suffer from spectral leakage. For precise offline analysis, consider
 * pre-windowing your input data (e.g., Hann window).
 *
 * SELECTIVITY IS A RATIO, NOT A COUNT. The block length N is a required
 * argument here precisely because only the caller knows the requirement, and
 * that requirement is always a frequency: the analysis bin width is fs/N and
 * the block spans N/fs seconds, so a block size chosen at one sample rate does
 * NOT carry to another. To hold a requirement constant, scale N with fs:
 *
 *     N >= fs / (wanted bin width in Hz),   rounded up as you prefer.
 *
 * Measured on the classic musical case -- separating one semitone from its
 * neighbour at C4 (261.63 Hz target, 277.18 Hz interferer) -- with N held at
 * 2048 samples: 9.4 dB rejection at 44.1 kHz, 7.5 dB at 48 kHz, 1.6 dB at
 * 88.2 kHz, 1.3 dB at 96 kHz, 0.4 dB at 176.4 kHz, 0.2 dB at 192 kHz. The two
 * semitones are simply no longer distinguishable above 48 kHz. Scaling N with
 * the rate instead (2048 at 48 kHz, 4096 at 96 kHz, 8192 at 192 kHz) holds the
 * rejection at 7.5-9.4 dB across that whole range. ChordDetector applies
 * exactly this rule when it sizes the window of its note bank.
 *
 * Threading: owner-managed. All methods (prepare, push/process, readouts,
 * reset) are meant to be called from the thread that owns the stream; no
 * internal synchronization is provided. Use one instance per thread.
 *
 * Dependencies: DspMath.h.
 */

#include "../Core/DspMath.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Goertzel
 * @brief Single-frequency magnitude detector using the Goertzel algorithm.
 *
 * Designed for continuous streaming or block-based real-time analysis.
 * State is unified: you can mix `processBlock` and `pushSample` safely.
 * Before prepare() the detector is inert and reports zero magnitude.
 *
 * The resonator runs in double whatever T is: near DC its pole pair sits on
 * the unit circle and a float state misreads low tones over long windows
 * (-3.96 dB at 27.5 Hz with a 65536-sample window at 192 kHz).
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
     * Release-safe: a non-finite or non-positive sample rate, a non-finite
     * target frequency or a non-positive block size are ignored (no-op
     * keeping the previous configuration); the target frequency is clamped
     * to [0, sampleRate/2].
     *
     * @param sampleRate Sample rate in Hz. Must be > 0.
     * @param targetFreqHz The frequency to detect, in [0, sampleRate/2].
     * @param blockSize Number of samples per analysis window. Must be > 0.
     *                  There is deliberately no default: the useful quantity
     *                  is the bin width fs/blockSize, so the count that meets
     *                  a requirement at one sample rate does not meet it at
     *                  another (see the selectivity note in the file header).
     *                  At exact DC and Nyquist the recursion sits on a double
     *                  pole and its state grows O(N^2); the double core keeps
     *                  that exact for any practical window.
     */
    void prepare(double sampleRate, double targetFreqHz, int blockSize) noexcept
    {
        assert(sampleRate > 0.0 && "Sample rate must be positive");
        assert(blockSize > 0 && "Block size must be strictly positive");
        assert(targetFreqHz >= 0.0 && targetFreqHz <= (sampleRate * 0.5) && "Frequency must be within Nyquist limit");

        if (!std::isfinite(sampleRate) || sampleRate <= 0.0
            || !std::isfinite(targetFreqHz) || blockSize <= 0)
            return;

        sampleRate_ = sampleRate;
        targetFreq_ = std::clamp(targetFreqHz, 0.0, sampleRate * 0.5);
        blockSize_ = blockSize;

        // Exact target frequency omega calculation (Generalized Goertzel)
        double omega = 2.0 * std::numbers::pi * targetFreq_ / sampleRate_;

        coeff_ = 2.0 * std::cos(omega);
        cosOmega_ = std::cos(omega);
        sinOmega_ = std::sin(omega);

        // Magnitude normalization: 2/N for interior bins, 1/N for DC and for
        // exact Nyquist (neither has a mirrored bin; the old 2/N at Nyquist
        // read a full-scale alternating signal 2x / +6 dB high).
        const bool isDc      = targetFreq_ <= 0.001;
        const bool isNyquist = targetFreq_ >= sampleRate_ * 0.5 * (1.0 - 1e-12);
        normalisationFactor_ = (isDc || isNyquist) ? 1.0 / blockSize_ : 2.0 / blockSize_;

        reset();
    }

    /**
     * @brief Processes a block of audio samples, updating the internal state.
     *
     * This method accumulates samples. To know if a full analysis block has
     * been completed, check `checkNewResultAvailable()` after processing.
     *
     * @param data Pointer to the audio samples.
     * @param numSamples Number of samples in the buffer.
     */
    void processBlock(const T* data, int numSamples) noexcept
    {
        if (data == nullptr || numSamples <= 0) return;

        // Bring state to local variables for compiler optimization (register allocation)
        double s1 = s1_;
        double s2 = s2_;
        const double coeff = coeff_;

        for (int i = 0; i < numSamples; ++i)
        {
            double s0 = static_cast<double>(data[i]) + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;

            if (++sampleCount_ >= blockSize_)
            {
                computeResult(s1, s2);
                s1 = 0.0;
                s2 = 0.0;
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
        double s0 = static_cast<double>(sample) + coeff_ * s1_ - s2_;
        s2_ = s1_;
        s1_ = s0;

        if (++sampleCount_ >= blockSize_)
        {
            computeResult(s1_, s2_);
            s1_ = 0.0;
            s2_ = 0.0;
            sampleCount_ = 0;
            return true;
        }
        return false;
    }

    /**
     * @brief Manually forces the computation of the result before N samples are reached.
     *
     * @warning Evaluating before reaching the configured blockSize will result in
     * inaccurate magnitude and phase due to incomplete integration and incorrect scaling.
     */
    void forceCompute() noexcept
    {
        computeResult(s1_, s2_);
        s1_ = 0.0;
        s2_ = 0.0;
        sampleCount_ = 0;
    }

    // -- Results ------------------------------------------------------------------

    /**
     * @brief Checks if a new result has been computed.
     *
     * Calling this will return true only once after an analysis block completes,
     * resetting the flag internally.
     *
     * @return True if new data is available.
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
     * @return Magnitude in dB, floored at -100 dB for zero/silence
     *         (dspark::gainToDecibels floor).
     */
    [[nodiscard]] T getMagnitudeDb() const noexcept
    {
        return gainToDecibels(getMagnitude());
    }

    /**
     * @brief Returns the phase angle at the target frequency.
     *
     * The phase is referenced to the analysis block boundary: it is
     * reproducible between blocks when the target frequency is an exact
     * multiple of sampleRate/blockSize (bin-exact tones).
     *
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

    /** @brief Returns the configured sample rate in Hz. */
    [[nodiscard]] double getSampleRate() const noexcept { return sampleRate_; }

    /** @brief Returns the configured analysis block size in samples. */
    [[nodiscard]] int getBlockSize() const noexcept { return blockSize_; }

    /**
     * @brief Resets the internal IIR state and counters to zero.
     *
     * Retains configured sample rate, frequency, and block size.
     */
    void reset() noexcept
    {
        s1_ = 0.0;
        s2_ = 0.0;
        sampleCount_ = 0;
        real_ = T(0);
        imag_ = T(0);
        hasNewResult_ = false;
    }

private:
    /**
     * @brief Computes real and imaginary parts and applies correct normalization.
     */
    void computeResult(double s1, double s2) noexcept
    {
        real_ = static_cast<T>((s1 - s2 * cosOmega_) * normalisationFactor_);
        imag_ = static_cast<T>((s2 * sinOmega_) * normalisationFactor_);
        hasNewResult_ = true;
    }

    double sampleRate_ = 48000.0;
    double targetFreq_ = 440.0;
    int blockSize_ = 2048;

    // Coefficients and resonator state in double (see the class comment).
    double coeff_ = 0.0;
    double cosOmega_ = 0.0;
    double sinOmega_ = 0.0;
    double normalisationFactor_ = 0.0;

    // Streaming state
    double s1_ = 0.0;
    double s2_ = 0.0;
    int sampleCount_ = 0;

    // Results
    T real_ = T(0);
    T imag_ = T(0);
    bool hasNewResult_ = false;
};

} // namespace dspark
