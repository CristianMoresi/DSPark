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
 * The period is chosen as in the frame stage of the probabilistic-threshold
 * method of Mauch & Dixon (2014): instead of one fixed threshold, a Beta
 * distribution of thresholds votes, each picking the first dip of the
 * normalised difference function below it, and the most probable dip wins.
 * A dip is reported when the prior accepts it with probability 1% or more,
 * which at the default setting voices dips up to 0.30 where the single
 * threshold stopped at 0.10; above 0.10 the window must also be steady (its
 * halves within a 0.6 energy ratio), so note starts and ends read as before.
 *
 * Measured at 48 kHz on harmonic tones of 70 to 1000 Hz (random spectral
 * tilt) in white noise, readings within 50 cents of the fundamental: 87.4%
 * of 2000 tones at 5 dB SNR, where the single threshold read every frame
 * unvoiced, and 92.0% of 400 at 10 dB (88.5% before), with octave errors
 * down from 6.8% to 0; a clean-room port of the method's frame stage scores
 * 87.3% and 92.0% on the same tones. At 0 dB SNR the detector reads
 * unvoiced rather than guess (that frame stage is right 30% of the time
 * there and wrong 70%). Noisy readings report their honest confidence, about
 * 0.76 at 5 dB, so a confidence gate keeps ignoring them. Steady clean tones
 * read exactly as before; on clean material the changes are fewer octave
 * errors and a few windows across a legato note change that now read the
 * period they share, at reduced confidence.
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
        rawDiff_.assign(static_cast<size_t>(halfWindow_), T(0));

        // YIN-FFT resources: the difference function is computed via one
        // cross-correlation in the frequency domain (3 FFTs) instead of the
        // O(windowSize^2) direct form - ~20x faster at the default window.
        // A frame of windowSize is enough: the first half against the whole
        // window at lags below windowSize/2 reaches at most sample
        // windowSize - 2, so nothing wraps around.
        fftSize_ = 1;
        while (fftSize_ < windowSize_) fftSize_ <<= 1;
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
     * follows the rate, so the per-second cost grows by a measured ~2.1x at
     * 96 kHz and ~4.3x at 192 kHz relative to 48 kHz (7.1 / 14.9 / 30.5 ms
     * of CPU per 10 s of audio at 48/96/192 kHz; float, automatic window,
     * g++ 13 -O2, min of 15 runs pinned to six cores with taskset -c 0-5 --
     * absolute numbers vary by machine, the growth tracks N log N). Against
     * the old fixed-2048 default at the SAME rate the automatic window costs
     * roughly 5% more (30.5 vs 28.9 ms at 192 kHz). prepare()-time heap grows
     * the same way: ~100 KB at 44.1/48 kHz, ~190 KB at 88.2/96 kHz, ~390 KB
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

    /**
     * @brief Returns the detection confidence [0.0 - 1.0] safely from any thread.
     *
     * One minus the normalised difference at the chosen dip: above
     * 1 - getThreshold() (0.9 by default) for the readings single-threshold
     * YIN also made, lower for noisier ones down to 1 - the voicing ceiling
     * (0.70 by default; about 0.76 for a tone at 5 dB SNR). Gate on it when
     * only clean readings should count (PitchFollower does, at 0.85).
     */
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
     * @brief Sets the sensitivity threshold (clamped to 0.01 - 0.5; default 0.1).
     *
     * The mean of the Beta(2, b) threshold distribution (Mauch & Dixon's
     * prior; the default is their Beta(2, 18)). Lower is stricter: the voicing ceiling,
     * the deepest normalised difference that still reads voiced, follows it
     * at about three times the mean (0.033 at 0.01, 0.16 at 0.05, 0.30 at
     * 0.1, 0.43 at 0.15) and is capped at 0.5 from a mean of 0.2 up.
     * Non-finite values are ignored.
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

        // Silence check on the window energy, the last prefix sum. It spans
        // the whole window, so it doubles as the non-finite gate: with any
        // NaN/Inf sample present the old code filled the CMND with zeros
        // ((NaN > 0) is false) and published fs/2 at confidence 1.0 - a fake
        // detection with maximum confidence. Report unvoiced instead; the
        // bad samples flush out of the window on their own.
        const T energy = prefixSq_[static_cast<size_t>(windowSize_)];
        if (energy < T(1e-10) || !std::isfinite(energy))
        {
            frequency_.store(T(0), std::memory_order_relaxed);
            confidence_.store(T(0), std::memory_order_relaxed);
            return;
        }
        const T e1 = prefixSq_[static_cast<size_t>(W)];

        // (b) r(tau) via FFT cross-correlation: IFFT(conj(FFT(first half)) * FFT(window))
        std::copy(currentWindow, currentWindow + W, fftTime_.begin());
        std::fill(fftTime_.begin() + W, fftTime_.end(), T(0));
        fft_->forward(fftTime_.data(), specHalf_.data());

        std::copy(currentWindow, currentWindow + windowSize_, fftTime_.begin());
        fft_->forward(fftTime_.data(), specFull_.data());   // tail beyond the window is still zero

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

        // (c) CMND from the closed-form difference function. While the
        // running sum is zero the window has no energy at those lags, which
        // is 0/0 and no evidence of a period, so it is the neutral 1.
        const T threshold = threshold_.load(std::memory_order_relaxed);
        yinBuffer_[0] = T(1);
        T runningSum = T(0);

        for (int tau = 1; tau < W; ++tau)
        {
            const T e2 = prefixSq_[static_cast<size_t>(tau + W)] - prefixSq_[static_cast<size_t>(tau)];
            T d = e1 + e2 - T(2) * corrTime_[static_cast<size_t>(tau)];
            if (d < T(0)) d = T(0); // guard tiny negative round-off

            rawDiff_[static_cast<size_t>(tau)] = d;
            runningSum += d;
            yinBuffer_[static_cast<size_t>(tau)] =
                (runningSum > T(0)) ? d * static_cast<T>(tau) / runningSum : T(1);
        }

        // (d) Candidate selection over a distribution of thresholds (the
        // frame stage of Mauch & Dixon 2014). A threshold s picks the first
        // local minimum of the CMND below s, so scanning the minima in lag
        // order, a minimum that is lower than every earlier one is picked by
        // exactly the thresholds in (its value, the previous record]: its
        // probability is the prior mass of that interval. Thresholds below
        // the deepest minimum pick nothing and hand kAbsoluteMinWeight of
        // their mass to the deepest minimum. The most probable candidate
        // wins; ties go to the shorter lag, as the first-dip rule does.
        //
        // A dip is valued at the vertex of the parabola through its three
        // samples: a period that falls between two lags (a high note, a
        // sweep) has a shallow sample at its true minimum, and valued at the
        // samples an exact-integer multiple of it outvoted it (measured: a
        // 3.3 kHz sawtooth at 44.1 kHz read an octave low at confidence
        // 0.997). At a strict minimum the vertex lies within half a lag and
        // its value is at least c1 - |c0 - c2| / 8; that bound, in the sample
        // type and loosened by kScreenSlack (far above its rounding error),
        // rejects nearly every lag before the exact test in double. The
        // screen is branch-free on purpose: on noise the minima fall at
        // random lags.
        if (threshold != cachedThreshold_) updatePrior(threshold);
        int bestTau = -1, lastTau = -1;
        double bestMass = -1.0, lastMass = 0.0, recordValue = 1.0, recordSurvival = 0.0;
        T screen = T(1) + kScreenSlack;
        const T* c = yinBuffer_.data();
        for (int tau = 2; tau + 1 < W; ++tau)
        {
            const T c0 = c[tau - 1], c1 = c[tau], c2 = c[tau + 1];
            if (!((c1 < c0) & (c1 <= c2) & (c1 - std::abs(c0 - c2) * T(0.125) < screen)))
                continue;
            const double d0 = static_cast<double>(c0), d1 = static_cast<double>(c1),
                         d2 = static_cast<double>(c2);
            const double v = std::max(0.0, d1 - (d0 - d2) * (d0 - d2) / (8.0 * (d0 - 2.0 * d1 + d2)));
            if (!(v < recordValue)) continue;
            if (lastTau >= 0 && lastMass > bestMass) { bestMass = lastMass; bestTau = lastTau; }
            const double s = priorSurvival(v);
            lastMass = s - recordSurvival;
            lastTau = tau;
            recordValue = v;
            recordSurvival = s;
            screen = static_cast<T>(v) + kScreenSlack;

            // Every later candidate shares at most the prior mass below this
            // record, plus the absolute-minimum share of it. Once the leader
            // holds more, and the share that goes to the deepest minimum
            // cannot swap this record with the earlier best, no later lag
            // changes the choice: stop (a clean dip near 0 ends the scan at
            // once, as the first-dip rule did). The result is the full scan's.
            const double below = 1.0 - s;
            const bool settled = (lastMass > bestMass)
                              || (bestMass > lastMass + kAbsoluteMinWeight * below);
            if (settled && std::max(bestMass, lastMass) > (1.0 + kAbsoluteMinWeight) * below)
                break;
        }
        if (lastTau >= 0)
        {
            lastMass += kAbsoluteMinWeight * (1.0 - recordSurvival);
            if (lastMass > bestMass) { bestMass = lastMass; bestTau = lastTau; }
        }

        // Voicing: the chosen dip must be one the threshold prior accepts
        // with probability at least kVoicedSurvival, a ceiling on its CMND
        // value (0.30 at the default prior mean of 0.1). A periodic signal in
        // white noise dips to about the noise share of the power (0.24 at
        // 5 dB SNR); noise without a period has no dip near the ceiling
        // (white noise bottoms out near 0.85). A dip above the threshold
        // itself also needs a steady window, its two halves within
        // kMinHalfEnergyRatio of each other in energy: across a note's start
        // or end the dip is shallow for another reason, and the period read
        // there is biased (measured 9 cents high a hop after a 220 Hz note
        // stopped), so those windows stay unvoiced as under the single
        // threshold.
        const int tauEstimate = bestTau;
        const double dip = (tauEstimate < 0) ? 1.0
                         : static_cast<double>(yinBuffer_[static_cast<size_t>(tauEstimate)]);
        const double firstHalf = static_cast<double>(e1);
        const double secondHalf = static_cast<double>(energy) - firstHalf;
        const bool steady = std::min(firstHalf, secondHalf)
                          >= kMinHalfEnergyRatio * std::max(firstHalf, secondHalf);
        if (tauEstimate < 0 || !(dip < voicingCeiling_)
            || (!(dip < static_cast<double>(threshold)) && !steady))
        {
            frequency_.store(T(0), std::memory_order_relaxed);
            confidence_.store(T(0), std::memory_order_relaxed);
            return;
        }

        // Sub-sample precision from the raw difference function (see
        // parabolicInterp()); the CMND only selects the dip.
        T betterTau = parabolicInterp(tauEstimate);
        T finalConfidence = std::clamp(T(1) - yinBuffer_[static_cast<size_t>(tauEstimate)], T(0), T(1));

        frequency_.store(static_cast<T>(sampleRate_) / betterTau, std::memory_order_relaxed);
        confidence_.store(finalConfidence, std::memory_order_relaxed);
    }

    /**
     * @brief Sub-sample period estimate around the selected dip.
     *
     * The parabola is fitted to the RAW difference function d(tau), not to
     * the CMND the dip was selected on: the CMND's tau / running-sum weight
     * tilts the dip and biased the vertex, by up to 1 cent at 1760 Hz (48
     * kHz) for a pure tone. Near its minimum a sinusoid's d(tau) is a raised
     * cosine, 1 - cos(w (tau - tau0)) with w = 2 pi / tau0, and a three-point
     * parabola on it returns tan(w delta) / (2 tan(w / 2)) instead of the
     * offset delta, so that bias is inverted exactly. What remains comes from
     * harmonics sharpening the dip: 2.6 cents at worst over the starting phase
     * for a 1/h harmonic tone at 1760 Hz, against 0.01 cent for a pure one.
     */
    [[nodiscard]] T parabolicInterp(int tau) const noexcept
    {
        if (tau < 2 || tau >= halfWindow_ - 1)
            return static_cast<T>(tau);

        // Centre on the raw function's own lowest sample next to the CMND dip.
        const int centre = tau;
        if (rawDiff_[static_cast<size_t>(centre - 1)] < rawDiff_[static_cast<size_t>(tau)]) tau = centre - 1;
        if (rawDiff_[static_cast<size_t>(centre + 1)] < rawDiff_[static_cast<size_t>(tau)]) tau = centre + 1;
        if (tau < 2 || tau >= halfWindow_ - 1)
            return static_cast<T>(tau);

        const double s0 = static_cast<double>(rawDiff_[static_cast<size_t>(tau - 1)]);
        const double s1 = static_cast<double>(rawDiff_[static_cast<size_t>(tau)]);
        const double s2 = static_cast<double>(rawDiff_[static_cast<size_t>(tau + 1)]);

        const double denom = s0 - 2.0 * s1 + s2;

        // Flat or inverted curvature (no usable dip on the raw function).
        if (!(denom > 1e-30))
            return static_cast<T>(tau);

        const double vertex = std::clamp((s0 - s2) / (2.0 * denom), -0.5, 0.5);
        const double w = 6.283185307179586 / static_cast<double>(tau);
        const double offset = std::atan(2.0 * vertex * std::tan(0.5 * w)) / w;
        return static_cast<T>(static_cast<double>(tau) + offset);
    }

    /**
     * @brief Probability that a threshold drawn from the prior exceeds x.
     *
     * The prior is Beta(2, b) on [0, 1], whose survival function has the
     * closed form (1 - x)^b (1 + b x) for any real b > 0.
     */
    [[nodiscard]] double priorSurvival(double x) const noexcept
    {
        if (!(x > 0.0)) return 1.0;
        if (!(x < 1.0)) return 0.0;
        return std::pow(1.0 - x, priorB_) * (1.0 + priorB_ * x);
    }

    /**
     * @brief Rebuilds the threshold prior for a new mean (audio thread, on a
     *        threshold change only): Beta(2, b) with mean 2 / (2 + b), and the
     *        CMND ceiling x where its survival falls to kVoicedSurvival, never
     *        above kMaxVoicingCeiling.
     */
    void updatePrior(T threshold) noexcept
    {
        cachedThreshold_ = threshold;
        const double mean = static_cast<double>(threshold);
        priorB_ = 2.0 * (1.0 - mean) / mean;
        double lo = 0.0, hi = 1.0;   // survival falls monotonically from 1 to 0
        for (int i = 0; i < 48; ++i)
        {
            const double mid = 0.5 * (lo + hi);
            if (priorSurvival(mid) > kVoicedSurvival) lo = mid; else hi = mid;
        }
        voicingCeiling_ = std::min(lo, kMaxVoicingCeiling);
    }

    /// Margin of the sample-type screen over the exact record (CMND units).
    static constexpr T kScreenSlack = T(1.0e-4);
    /// Weight of the deepest minimum when a threshold picks nothing (Mauch &
    /// Dixon's 0.01).
    static constexpr double kAbsoluteMinWeight = 0.01;
    /// Prior probability of acceptance below which the chosen dip reads
    /// unvoiced: the voicing ceiling (0.30 at the default prior mean 0.1).
    static constexpr double kVoicedSurvival = 1.0e-2;
    /// Least energy ratio of the window halves for a dip above the threshold
    /// to read voiced (a steady tone measures 0.72 or more, clean or noisy).
    static constexpr double kMinHalfEnergyRatio = 0.6;
    /// Highest voicing ceiling: a dip at half the mean difference or above
    /// (noise power equal to the signal's) never reads voiced.
    static constexpr double kMaxVoicingCeiling = 0.5;

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

    // Threshold prior as last built by updatePrior() (audio thread only).
    T cachedThreshold_ = T(-1);
    double priorB_ = 18.0;
    double voicingCeiling_ = 0.0;

    // Thread-safe outputs
    std::atomic<T> frequency_{T(0)};
    std::atomic<T> confidence_{T(0)};

    // Mirrored buffer keeps every analysis window contiguous
    std::vector<T> buffer_;     // Size: 2 * windowSize_
    std::vector<T> yinBuffer_;  // Size: halfWindow_
    std::vector<T> rawDiff_;    // Size: halfWindow_ (difference function d(tau))

    // YIN-FFT resources (cross-correlation difference function)
    int fftSize_ = 2048;
    std::unique_ptr<FFTReal<T>> fft_; // doubles as the "prepared" gate
    std::vector<T> fftTime_;    // Size: fftSize_
    std::vector<T> specHalf_;   // Size: fftSize_ + 2
    std::vector<T> specFull_;   // Size: fftSize_ + 2
    std::vector<T> corrTime_;   // Size: fftSize_
    std::vector<T> prefixSq_;   // Size: windowSize_ + 1
};

} // namespace dspark
