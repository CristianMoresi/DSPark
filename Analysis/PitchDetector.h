// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PitchDetector.h
 * @brief Real-time monophonic pitch detection using the YIN algorithm.
 *
 * Implements the YIN autocorrelation method (de Cheveigne & Kawahara, 2002)
 * with the difference function computed as a frequency-domain
 * cross-correlation (YIN-FFT, 3 FFTs per detection). A mirrored input
 * buffer keeps every analysis window fully contiguous in memory.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with pushSamples()).
 * - pushSamples() / reset(): audio thread (stream owner); reset() is not
 *   thread-safe with pushSamples().
 * - getFrequencyHz() / getConfidence() / getMidiNote() / getCentsOffset():
 *   any thread, lock-free. Frequency and confidence are published as two
 *   independent atomics, so a reader may pair a fresh frequency with the
 *   previous confidence for one detection (benign for tracking/metering).
 * - setThreshold(): control thread (single-word relaxed atomic; non-finite
 *   values are ignored).
 *
 * Dependencies: DspMath.h, FFT.h.
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class PitchDetector
 * @brief Thread-safe YIN pitch detector with lock-free readout.
 *
 * A non-finite stretch in the input signal simply reads as "unvoiced"
 * (frequency 0, confidence 0) and flushes out of the analysis window on its
 * own: the pipeline holds no recursive state, so the detector self-recovers
 * once clean samples refill the window.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PitchDetector
{
public:
    /**
     * @brief Prepares the detector and allocates internal structures.
     *
     * Must be called before audio processing begins. Zero allocations
     * happen after this point. Release-safe: a non-finite or non-positive
     * sample rate is ignored (no-op keeping the previous configuration);
     * an explicit windowSize is clamped to [64, 1 << 20].
     *
     * @param sampleRate The system sample rate in Hz.
     * @param windowSize Analysis window in samples. Values <= 0 (the default)
     *                   select the AUTOMATIC window: the smallest power of two
     *                   in [512, 16384] spanning at least 2048/48000 s
     *                   (~42.7 ms) at sampleRate -- 2048 at 44.1/48 kHz, 4096
     *                   at 88.2/96 kHz, 8192 at 176.4/192 kHz, 16384 at
     *                   384 kHz, 512 at 8 kHz. Read it back with
     *                   getWindowSize().
     * @param hopSize    Samples between detections. Values <= 0 (the default)
     *                   select windowSize/4, which is 512 at the 44.1/48 kHz
     *                   window and keeps the detection RATE at ~one per
     *                   10.7 ms at every sample rate. Explicit positive values
     *                   are clamped to [1, windowSize].
     *
     * THE WINDOW IS A REGISTER, NOT A COUNT. YIN searches lags tau in
     * [2, windowSize/2), so the lowest fundamental the detector can reach is
     *
     *     fmin = fs / (windowSize/2 - 1)      (approximately 2*fs/windowSize)
     *
     * -- a property of the RATIO fs/windowSize alone. Measured, identical
     * across rates at equal ratio: fs/windowSize = 23.44 Hz recovers down to
     * A1 55 Hz (48 kHz/2048, 96 kHz/4096, 192 kHz/8192 all agree to within
     * 0.01 cents); fs/windowSize = 5.86 Hz reaches C1 32.7 Hz. Holding the
     * COUNT fixed instead moves the floor with the rate: an explicit 2048
     * window gives fmin 46.9 Hz at 48 kHz but 93.8 Hz at 96 kHz and 187.7 Hz
     * at 192 kHz, so E1/A1/E2 -- and at 176.4/192 kHz A2 as well -- are simply
     * never reported. Out-of-register notes read as unvoiced (0 Hz,
     * confidence 0) rather than as a wrong pitch: measured across
     * 44.1..192 kHz on band-limited sawtooths from E1 to A5, every reading was
     * either correct to within 0.1 cents or absent, never confidently wrong.
     * The automatic window keeps fmin at 43.1 Hz (44.1 kHz family) / 46.9 Hz
     * (48 kHz family) from 8 kHz all the way to 384 kHz, where it reaches the
     * 16384 clamp. ABOVE 384 kHz the ceiling binds and the floor climbs again
     * (measured 93.8 Hz at 768 kHz) -- documented, not fixed; pass an explicit
     * window there. For a lower floor at any rate pass an explicit window:
     * fmin scales as fs/windowSize, so doubling the window halves it.
     */
    void prepare(double sampleRate, int windowSize = 0, int hopSize = 0)
    {
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
            return;

        fft_.reset(); // gate OFF: pushSamples() is a no-op while rebuilding

        sampleRate_ = sampleRate;
        if (windowSize <= 0)
        {
            // Automatic: hold the analysis TIME SPAN constant across sample
            // rates (the span 2048 samples cover at 48 kHz). The reachable
            // register is fs/(windowSize/2 - 1), so a constant span pins the
            // lowest detectable fundamental instead of letting it climb with
            // the rate.
            const double target = sampleRate_ * (kAutoSpanRef / kAutoSpanRate);
            int n = kAutoMinWindow;
            while (n < kAutoMaxWindow && static_cast<double>(n) < target) n <<= 1;
            windowSize_ = n;
        }
        else
        {
            windowSize_ = std::clamp(windowSize, 64, 1 << 20);
        }
        halfWindow_ = windowSize_ / 2;
        hopSize_    = (hopSize <= 0) ? std::max(1, windowSize_ / 4)
                                     : std::clamp(hopSize, 1, windowSize_);

        // Mirrored buffer technique: size is 2x windowSize.
        // Guarantees continuous memory layout without modulo operations.
        buffer_.assign(static_cast<size_t>(windowSize_) * 2, T(0));
        yinBuffer_.assign(static_cast<size_t>(halfWindow_), T(0));

        // YIN-FFT resources: the difference function is computed via one
        // cross-correlation in the frequency domain (3 FFTs) instead of the
        // O(windowSize^2) direct form - ~20x faster at the default window.
        fftSize_ = 1;
        while (fftSize_ < windowSize_ * 2) fftSize_ <<= 1;
        fftTime_.assign(static_cast<size_t>(fftSize_), T(0));
        specHalf_.assign(static_cast<size_t>(fftSize_) + 2, T(0));
        specFull_.assign(static_cast<size_t>(fftSize_) + 2, T(0));
        corrTime_.assign(static_cast<size_t>(fftSize_), T(0));
        prefixSq_.assign(static_cast<size_t>(windowSize_) + 1, T(0));

        writePos_ = 0;
        samplesSinceLastDetect_ = 0;

        frequency_.store(T(0), std::memory_order_relaxed);
        confidence_.store(T(0), std::memory_order_relaxed);

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize_)); // gate ON, last
    }

    /**
     * @brief Pushes audio samples into the analysis buffer.
     *
     * Automatically triggers pitch detection when the hop size is reached.
     * Lock-free and allocation-free. No-op before prepare().
     *
     * @note The difference function runs as a frequency-domain
     * cross-correlation (YIN-FFT): O(N log N) per detection - roughly 20x
     * faster than the direct O(windowSize^2) form at the 2048-sample window
     * the automatic policy picks at 44.1/48 kHz, and generally fine on the
     * audio thread. Under the automatic policy the detection RATE is
     * rate-invariant (hop = window/4, about one detection per 10.7 ms), but
     * CPU per second of audio is NOT: each detection costs O(N log N) and N
     * follows the rate, so the per-second cost grows by a measured ~2.2x at
     * 96 kHz and ~4.8x at 192 kHz relative to 48 kHz (35.2 / 76.7 / 169.7 ms
     * of CPU per 10 s of audio at 48/96/192 kHz; g++ -O2, one core --
     * absolute numbers vary by machine, the growth tracks N log N). Against
     * the old fixed-2048 default at the SAME rate the automatic window costs
     * roughly 20% more (169.7 vs 139.3 ms at 192 kHz). prepare()-time heap grows
     * the same way: ~160 KB at 44.1/48 kHz, ~320 KB at 88.2/96 kHz, ~640 KB
     * at 176.4/192 kHz (float); the audio path stays allocation-free. Budget
     * from these figures, not from the rate alone. For very low-latency
     * callbacks you can still feed an SPSC queue (Core/SpscQueue.h) and
     * detect on a worker.
     *
     * @param samples Span of input audio data (mono).
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (fft_ == nullptr)
            return;

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

    /**
     * @brief Sets the sensitivity threshold (clamped to 0.01 - 0.5).
     * Lower is stricter. Non-finite values are ignored.
     */
    void setThreshold(T threshold) noexcept
    {
        if (!std::isfinite(threshold)) return;
        threshold_.store(std::clamp(threshold, T(0.01), T(0.5)), std::memory_order_relaxed);
    }

    /** @brief Returns the sensitivity threshold. */
    [[nodiscard]] T getThreshold() const noexcept
    {
        return threshold_.load(std::memory_order_relaxed);
    }

    /**
     * @return The analysis window (samples) in effect: the automatic choice if
     *         prepare() was called with windowSize <= 0, the clamped explicit
     *         request otherwise (member default 2048 before the first
     *         successful prepare()). The lowest reachable fundamental is
     *         sampleRate / (getWindowSize()/2 - 1).
     */
    [[nodiscard]] int getWindowSize() const noexcept { return windowSize_; }

    /** @return Samples between detections in effect (windowSize/4 by
     *          default), i.e. one detection every getHopSize()/sampleRate s. */
    [[nodiscard]] int getHopSize() const noexcept { return hopSize_; }

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

        // Silence check / Energy calculation on contiguous memory. The sum
        // spans the whole window, so it doubles as the non-finite gate: with
        // any NaN/Inf sample present the old code filled the CMND with zeros
        // ((NaN > 0) is false) and published fs/2 at confidence 1.0 - a fake
        // detection with maximum confidence. Report unvoiced instead; the
        // bad samples flush out of the window on their own.
        T energy = T(0);
        for (int i = 0; i < windowSize_; ++i) {
            energy += currentWindow[i] * currentWindow[i];
        }

        if (energy < T(1e-10) || !std::isfinite(energy))
        {
            frequency_.store(T(0), std::memory_order_relaxed);
            confidence_.store(T(0), std::memory_order_relaxed);
            return;
        }

        // YIN-FFT difference function:
        //   d(tau) = E1 + E2(tau) - 2*r(tau)
        // with E1 = sum of x[0..W)^2 (constant), E2(tau) the energy of the
        // shifted window (prefix sums), and r(tau) the cross-correlation of
        // the first half against the full window - computed with 3 FFTs.
        const int W = halfWindow_;

        // (a) prefix sums of squared samples over the full window
        prefixSq_[0] = T(0);
        for (int i = 0; i < windowSize_; ++i)
            prefixSq_[static_cast<size_t>(i + 1)] =
                prefixSq_[static_cast<size_t>(i)] + currentWindow[i] * currentWindow[i];
        const T e1 = prefixSq_[static_cast<size_t>(W)];

        // (b) r(tau) via FFT cross-correlation: IFFT(conj(FFT(first half)) * FFT(window))
        std::fill(fftTime_.begin(), fftTime_.end(), T(0));
        std::copy(currentWindow, currentWindow + W, fftTime_.begin());
        fft_->forward(fftTime_.data(), specHalf_.data());

        std::fill(fftTime_.begin(), fftTime_.end(), T(0));
        std::copy(currentWindow, currentWindow + windowSize_, fftTime_.begin());
        fft_->forward(fftTime_.data(), specFull_.data());

        const int numBins = fftSize_ / 2 + 1;
        for (int k = 0; k < numBins; ++k)
        {
            const T aRe = specHalf_[static_cast<size_t>(2 * k)];
            const T aIm = specHalf_[static_cast<size_t>(2 * k + 1)];
            const T bRe = specFull_[static_cast<size_t>(2 * k)];
            const T bIm = specFull_[static_cast<size_t>(2 * k + 1)];
            // conj(A) * B
            specFull_[static_cast<size_t>(2 * k)]     = aRe * bRe + aIm * bIm;
            specFull_[static_cast<size_t>(2 * k + 1)] = aRe * bIm - aIm * bRe;
        }
        fft_->inverse(specFull_.data(), corrTime_.data());

        // (c) CMND from the closed-form difference function
        const T threshold = threshold_.load(std::memory_order_relaxed);
        yinBuffer_[0] = T(1);
        T runningSum = T(0);

        for (int tau = 1; tau < W; ++tau)
        {
            const T e2 = prefixSq_[static_cast<size_t>(tau + W)] - prefixSq_[static_cast<size_t>(tau)];
            T d = e1 + e2 - T(2) * corrTime_[static_cast<size_t>(tau)];
            if (d < T(0)) d = T(0); // guard tiny negative round-off

            runningSum += d;
            yinBuffer_[static_cast<size_t>(tau)] =
                (runningSum > T(0)) ? d * static_cast<T>(tau) / runningSum : T(0);
        }

        // Search for dip below threshold
        int tauEstimate = -1;
        for (int tau = 2; tau < halfWindow_; ++tau)
        {
            if (yinBuffer_[static_cast<size_t>(tau)] < threshold)
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

        // Sub-sample precision. The dip search guarantees a local minimum
        // (left neighbour above, right neighbour not below), so the
        // parabolic adjustment is bounded to +-0.5 by construction.
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

    /// Automatic-window policy: the span 2048 samples cover at 48 kHz,
    /// resolved inside [512, 16384] (covers 8 kHz .. 384 kHz without clamping).
    static constexpr double kAutoSpanRef = 2048.0;
    static constexpr double kAutoSpanRate = 48000.0;
    static constexpr int kAutoMinWindow = 512;
    static constexpr int kAutoMaxWindow = 16384;

    double sampleRate_ = 44100.0;
    int windowSize_ = 2048;
    int halfWindow_ = 1024;
    int hopSize_ = 512;
    int writePos_ = 0;
    int samplesSinceLastDetect_ = 0;
    std::atomic<T> threshold_{ T(0.10) };

    // Thread-safe outputs
    std::atomic<T> frequency_{T(0)};
    std::atomic<T> confidence_{T(0)};

    // Mirrored buffer keeps every analysis window contiguous
    std::vector<T> buffer_;     // Size: 2 * windowSize_
    std::vector<T> yinBuffer_;  // Size: halfWindow_

    // YIN-FFT resources (cross-correlation difference function)
    int fftSize_ = 4096;
    std::unique_ptr<FFTReal<T>> fft_; // doubles as the "prepared" gate
    std::vector<T> fftTime_;    // Size: fftSize_
    std::vector<T> specHalf_;   // Size: fftSize_ + 2
    std::vector<T> specFull_;   // Size: fftSize_ + 2
    std::vector<T> corrTime_;   // Size: fftSize_
    std::vector<T> prefixSq_;   // Size: windowSize_ + 1
};

} // namespace dspark
