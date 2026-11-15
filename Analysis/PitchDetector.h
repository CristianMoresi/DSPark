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
 * A dip is the whole stretch of lags below the threshold, read at its
 * deepest minimum, so the small minima noise leaves on a broad dip's walls
 * do not split its vote (chooseDip()). A dip is reported when the prior
 * accepts it with probability 1% or more, which at the default setting
 * voices dips up to 0.30 where the single threshold stopped at 0.10; above
 * 0.10 the window must also be steady (its halves within a 0.6 energy
 * ratio), so note starts and ends read as before.
 *
 * Three more looks keep a noisy window from reading wrong or not at all. A
 * choice above the threshold yields to the first separate dip about as deep
 * (in noise the dips at a period and at its multiples are equally deep, and
 * the deepest is as often a multiple). A window that reads unvoiced and
 * holds power below the lowest note it can read is looked at again without
 * that band (clearHistory()): rumble and the bottom of pink or brown noise
 * fill the difference function at the lags where low notes dip. And a steady
 * window that still reads unvoiced is read once more with its noise floor
 * removed (setNoiseSuppression(), on by default). None of them runs on a
 * window that reads voiced at once, so a clean tone costs what it did.
 *
 * Measured at 48 kHz frame by frame on sustained harmonic notes (40
 * partials, spectral tilt 3 to 12 dB/octave, gaps in between) in continuous
 * noise: windows within 50 cents of the note, at the defaults, against the
 * frame stage of that method followed by its HMM decoded causally (a
 * clean-room port, in brackets). Notes of 70 to 1000 Hz in white noise at
 * 10, 5 and 0 dB SNR: 100, 99.3 and 97.5% (89.4, 32.7, 7.0); in pink noise
 * 99.6, 98.7 and 85.3% (98.4, 80.5, 13.4). Notes of 47 to 67 Hz in white
 * noise at 5 and 0 dB: 99.5 and 95.7% (47.8, 14.0); in pink noise at 10, 5
 * and 0 dB: 97.1, 87.6 and 68.3% (83.5, 56.5, 17.1); in brown noise at 5
 * dB: 87.8% (85.7). Windows read wrong: under 1% from 70 Hz up, 1.4 to 4.3%
 * below it in pink and brown noise (the port 0 to 16.3%). Gaps between the
 * notes read as a note in 0.2% of their windows in brown noise, none in
 * white or pink. With setNoiseSuppression(false) the 0 dB rows read 0.4 to
 * 12.4% and the 5 dB ones 74.9 to 94.4%. Noisy readings report their honest
 * confidence, about 0.76 at 5 dB and 0.5 at 0 dB, so a confidence gate keeps
 * ignoring them. Steady clean tones read exactly as before. Across a legato
 * note change two notes can share a longer
 * period (four periods of one, five of the other); a window whose dip lies
 * there reads unvoiced, not that period: on clean tones, sweeps and stepped
 * melodies at three rates the wrong readings fall from 184 to 46 of 66090,
 * with every correct reading kept.
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
 *   values are ignored). setNoiseSuppression(): control thread (relaxed
 *   atomic, read once per detection).
 *
 * Dependencies: DspMath.h, FFT.h, SimdOps.h.
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/SimdOps.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
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

        // Mirrored buffer technique: the last kHistoryWindows windows of
        // input, written twice, so the analysis window (the newest of them)
        // and the whole history are both contiguous without modulo
        // operations. The history is read by the looks a noisy window gets
        // (the persistence tests and the noise suppression), for periods
        // whose multiples the window cannot hold.
        historySize_ = windowSize_ * kHistoryWindows;
        buffer_.assign(static_cast<size_t>(historySize_) * 2, T(0));
        clearBuffer_.assign(static_cast<size_t>(historySize_) * 2, T(0));
        clearPending_ = 0;
        designRumbleFilter();
        yinBuffer_.assign(static_cast<size_t>(halfWindow_), T(0));
        rawDiff_.assign(static_cast<size_t>(halfWindow_), T(0));
        yinClear_.assign(static_cast<size_t>(halfWindow_), T(0));
        rawClear_.assign(static_cast<size_t>(halfWindow_), T(0));

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

        // Noise floor suppression (setNoiseSuppression()): allocated whether
        // or not it is on, so switching it is real-time safe.
        cleanTime_.assign(static_cast<size_t>(windowSize_) * 2, T(0));
        cleanSpec_.assign(static_cast<size_t>(windowSize_) * 2 + 2, T(0));
        cleanPower_.assign(static_cast<size_t>(windowSize_) + 1, T(0));
        cleanFloor_.assign(static_cast<size_t>((windowSize_ + kFloorBins) / kFloorBins), T(0));
        cleanScratch_.assign(static_cast<size_t>(kFloorBins), T(0));
        yinClean_.assign(static_cast<size_t>(halfWindow_), T(0));
        rawClean_.assign(static_cast<size_t>(halfWindow_), T(0));
        persistSq_.assign(static_cast<size_t>(historySize_) + 1, 0.0);
        const int multipleReach = static_cast<int>(std::ceil(kMultipleSearch * historySize_)) + 2;
        multipleScratch_.assign(static_cast<size_t>(2 * multipleReach + 1), 0.0);
        dipRecords_.assign(static_cast<size_t>(halfWindow_ / 2 + 2), DipRecord {});
        dipGroups_.assign(dipRecords_.size(), DipGroup {});
        survivalTable_.assign(static_cast<size_t>(kSurvivalSteps) + 1, 0.0);
        survivalSlope_.assign(static_cast<size_t>(kSurvivalSteps) + 1, 0.0);
        cachedThreshold_ = T(-1);   // rebuild the prior tables at the first detection
        recent_.fill(0.0);
        fftClean_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(windowSize_) * 2);

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
     * the same way: ~300 KB at 44.1/48 kHz, ~600 KB at 88.2/96 kHz, ~1190 KB
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
            buffer_[static_cast<size_t>(writePos_ + historySize_)] = sample;
            // One more sample the rumble-free copy has not seen (the count
            // saturates once the history no longer holds them all).
            if (clearPending_ <= historySize_) ++clearPending_;

            writePos_++;
            if (writePos_ >= historySize_)
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
     * @brief Reads a tone through broadband noise (on by default).
     *
     * On: a steady window that still reads unvoiced, and whose own dip is
     * not at the floor noise alone reaches, is read once more with its
     * noise floor removed by spectral subtraction (the floor estimated per
     * band from the window itself, so white, pink and other smooth noises
     * alike). A dip of the cleaned window below the threshold locates the
     * period; the period itself is then read from the input's own
     * difference function next to that dip, since the subtraction shifts a
     * partial's peak by a few hertz (which read tones near 100 Hz up to a
     * semitone sharp). Where the cleaned window does not dip that far (low
     * notes, whose partials crowd the floor estimate, and pink noise), the
     * input's own dip is the candidate. The period is refined from the
     * correlation peaks at its multiples over the last two windows of input
     * (refineByMultiples()), and the reading is kept only when the input
     * repeats at those multiples about as strongly as at the period itself
     * (a tone does, a narrow band of noise that the cleaning made look
     * periodic does not) and two of the three detections before it found
     * the same period (a narrow band of noise wanders). The reading's
     * confidence stays the input's own dip at that period, about 0.5 at
     * 0 dB SNR, so a confidence gate keeps ignoring it. A reading through
     * the noise therefore starts one or two detections after the tone.
     *
     * The file description gives the frame-by-frame figures. With it on, a
     * note of 70 to 1000 Hz at 0 dB SNR reads right in 97.5% of its windows
     * in white noise and 85.3% in pink (0.4 and 11.0% with it off), a note
     * of 47 to 67 Hz in 95.7 and 68.3% (2.6 and 12.4%), and notes of 70 to
     * 1000 Hz at 2 dB SNR with 50 cents of vibrato in 96.5% (12.2%).
     *
     * Noise alone reads the same on or off, within 0.3% of windows over
     * 20 s: white and pink none, brown 0.4% (off 0.1%), and a band of noise
     * what the first looks read of it (Q 2 at 60 Hz 10.4%, at 120 Hz 4.7%;
     * Q 5 at 60 Hz 25.1%, at 300 Hz 10.1%; Q 10 from 60 Hz to 1 kHz 33 to
     * 47%). The read costs more than a detection, and only where it runs:
     * about 145 ns per sample on a tone at 0 dB SNR and 100 on pink or brown
     * noise alone, against 18 and 38 with it off. White noise alone, whose
     * dip stays at the noise floor, costs 17 either way, and a window that
     * reads voiced at once is not read again (15 to 18). Off, the output is
     * that of the first looks. Control thread; relaxed atomic.
     */
    void setNoiseSuppression(bool on) noexcept
    {
        noiseSuppression_.store(on, std::memory_order_relaxed);
    }

    /** @brief Whether noise floor suppression is on (setNoiseSuppression()). */
    [[nodiscard]] bool getNoiseSuppression() const noexcept
    {
        return noiseSuppression_.load(std::memory_order_relaxed);
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
        std::fill(clearBuffer_.begin(), clearBuffer_.end(), T(0));
        for (RumbleSection& section : rumble_) section.s1 = section.s2 = 0.0;
        clearPending_ = 0;
        recent_.fill(0.0);
        writePos_ = 0;
        samplesSinceLastDetect_ = 0;
        frequency_.store(T(0), std::memory_order_relaxed);
        confidence_.store(T(0), std::memory_order_relaxed);
    }

private:
    void detect() noexcept
    {
        // Obtain a perfectly contiguous block of memory representing the current window.
        // writePos_ points to the oldest sample of the history in the mirrored
        // buffer; the window is its newest windowSize_ samples.
        const T* history = &buffer_[static_cast<size_t>(writePos_)];
        const T* currentWindow = history + (historySize_ - windowSize_);

        T energy = T(0), e1 = T(0);
        if (!differenceFunction(currentWindow, yinBuffer_.data(), rawDiff_.data(), energy, e1))
        {
            rememberCandidate(0.0);
            publishUnvoiced();
            return;
        }

        // Share of the window's power in the three lowest bins of its
        // spectrum, which reach up to the lowest note the window can hold.
        const double lowShare = lowBandPower_ / static_cast<double>(energy);
        const double subShare = subBandPower_ / static_cast<double>(energy);

        const T threshold = threshold_.load(std::memory_order_relaxed);
        if (threshold != cachedThreshold_) updatePrior(threshold);
        WindowReading reading = readWindow(currentWindow, energy, e1, threshold);
        double secondCandidate = 0.0;   // period the second look could not confirm
        // Slow wander alone dips at the long lags: brown noise read as a
        // note in 0.75% of its windows. Where the window is mostly below its
        // lowest note (kSubRumbleShare of its power in the two lowest bins),
        // a dip above the threshold must also repeat at its multiples, as a
        // tone does (periodPersists()); 0.11% of those windows then read.
        if (reading.voiced && reading.dip >= static_cast<double>(threshold) && subShare >= kSubRumbleShare)
        {
            historySumsReady_ = false;
            double contrast = 0.0;
            const double period = static_cast<double>(parabolicInterp(rawDiff_.data(), reading.tau));
            reading.voiced = periodPersists(history, period, contrast);
        }

        // A window that reads unvoiced is looked at once more without what
        // lies below the lowest note it can hold. Rumble, hum's subharmonics
        // and the bottom of pink or brown noise carry no period the detector
        // could report, yet they fill the difference function at the long
        // lags where low notes dip: 47 to 67 Hz tones in brown noise at 5 dB
        // SNR read in 57% of their windows, and in 89% without that band.
        // The look is taken only when the high-pass removes kRumbleShare of
        // the window's power, and it stands only when it reads a note or dips
        // deeper than the first (the filter also takes a little of a
        // fundamental next to its corner): a clean tone or one in white noise
        // is read as before. When it stands, everything after this point (the
        // noise suppression included) works on the rumble-free
        // history. The filter itself is only run when those lowest bins hold
        // half that share or more, so a window with nothing down there (a
        // consonant, hiss, a mid tone in white noise) costs nothing extra.
        if (!reading.voiced && lowShare >= 0.5 * kRumbleShare)
        {
            const T* clear = clearHistory();
            const T* clearWindow = clear + (historySize_ - windowSize_);
            const double kept = static_cast<double>(simd::sumOfSquaresT(clearWindow, windowSize_));
            T clearEnergy = T(0), clearFirst = T(0);
            if (kept < (1.0 - kRumbleShare) * static_cast<double>(energy)
                && differenceFunction(clearWindow, yinClear_.data(), rawClear_.data(), clearEnergy, clearFirst))
            {
                // readWindow() and the code below read yinBuffer_ and rawDiff_.
                yinBuffer_.swap(yinClear_);
                rawDiff_.swap(rawClear_);
                WindowReading second = readWindow(clearWindow, clearEnergy, clearFirst, threshold);
                // What the filter leaves of brown noise is a band just above
                // its corner, and a band of noise dips like a low note. A
                // tone repeats at every multiple of its period and holds it
                // from one detection to the next; the band does neither. So
                // a dip above the threshold must pass periodPersists() and
                // agreesWithRecent() here, and a period that does not is
                // remembered as this detection's candidate for the next.
                if (second.voiced && second.dip >= static_cast<double>(threshold))
                {
                    historySumsReady_ = false;
                    double contrast = 0.0;
                    const double period = static_cast<double>(parabolicInterp(rawDiff_.data(), second.tau));
                    secondCandidate = period;
                    second.voiced = periodPersists(clear, period, contrast) && agreesWithRecent(period);
                }
                if (second.voiced || second.dip < reading.dip)
                {
                    reading = second;
                    history = clear;
                    currentWindow = clearWindow;
                }
                else
                {
                    yinBuffer_.swap(yinClear_);
                    rawDiff_.swap(rawClear_);
                }
            }
        }

        const int tauEstimate = reading.tau;
        const double dip = reading.dip;
        const bool steady = reading.steady;
        if (reading.voiced)
        {
            // Sub-sample precision from the raw difference function (see
            // parabolicInterp()); the CMND only selects the dip.
            const T period = parabolicInterp(rawDiff_.data(), tauEstimate);
            rememberCandidate(static_cast<double>(period));
            publish(period, std::clamp(T(1) - yinBuffer_[static_cast<size_t>(tauEstimate)], T(0), T(1)));
            return;
        }

        // Noise floor suppression (on by default): a steady window that reads
        // unvoiced is read again. Its noise removed, a dip of the cleaned
        // window below the threshold locates the period, which is then taken
        // from the input's own difference function next to it (the
        // subtraction shifts the partials' peaks by a few hertz); when the
        // cleaning finds none, the input's own dip is the candidate. Either
        // is refined over its multiples (refineByMultiples()) and kept only
        // when the input repeats at its multiples (periodPersists()) and the
        // period agrees with recent detections (agreesWithRecent()). The
        // confidence is the input's own dip there: a reading taken through
        // the noise stays a low-confidence one (about 0.5 at 0 dB SNR).
        double candidate = 0.0;
        if (noiseSuppression_.load(std::memory_order_relaxed) && steady && dip < kFallbackCeiling)
        {
            historySumsReady_ = false;
            const T* clean = suppressNoise(currentWindow);
            T cleanEnergy = T(0), cleanFirst = T(0);
            if (differenceFunction(clean, yinClean_.data(), rawClean_.data(), cleanEnergy, cleanFirst))
            {
                int unused[kMaxPartials];
                int numUnused = 0;
                const int tau = chooseDip(yinClean_.data(), unused, numUnused);
                if (tau >= 0 && yinClean_[static_cast<size_t>(tau)] < threshold)
                {
                    const int lag = inputDipNear(tau);
                    const double period = refineByMultiples(history,
                        static_cast<double>(parabolicInterp(rawDiff_.data(), lag)));
                    double contrast = 0.0;
                    if (periodPersists(history, period, contrast))
                    {
                        candidate = period;
                        if (agreesWithRecent(period))
                        {
                            rememberCandidate(period);
                            publish(static_cast<T>(period),
                                    std::clamp(T(1) - yinBuffer_[static_cast<size_t>(lag)], T(0), T(1)));
                            return;
                        }
                    }
                }
            }
            // The input's own dip, where noise keeps the cleaned window from
            // dipping below the threshold: low notes, whose partials crowd
            // the floor estimate, and pink noise. Not a dip at the floor that
            // noise alone reaches (kFallbackCeiling), nor a periodicity as
            // weak as low-passed noise shows or as strong as a narrow band of
            // noise makes (kMinFallbackContrast, kMaxFallbackContrast).
            const double own = (tauEstimate >= 0)
                ? static_cast<double>(parabolicInterp(rawDiff_.data(), tauEstimate)) : 0.0;
            if (tauEstimate >= 0 && dip < kFallbackCeiling
                && !(candidate > 0.0 && std::abs(own - candidate) < kSameCandidate * candidate))
            {
                const double period = refineByMultiples(history, own);
                double contrast = 0.0;
                if (periodPersists(history, period, contrast)
                    && contrast >= kMinFallbackContrast && contrast <= kMaxFallbackContrast)
                {
                    if (candidate == 0.0) candidate = period;
                    if (agreesWithRecent(period))
                    {
                        rememberCandidate(period);
                        publish(static_cast<T>(period),
                                std::clamp(T(1) - yinBuffer_[static_cast<size_t>(tauEstimate)], T(0), T(1)));
                        return;
                    }
                }
            }
        }
        rememberCandidate(candidate > 0.0 ? candidate : secondCandidate);
        publishUnvoiced();
    }

    /// What one look at a window found (readWindow()).
    struct WindowReading
    {
        int tau = -1;         ///< chosen lag, or -1
        double dip = 1.0;     ///< CMND value there
        bool steady = false;  ///< the window's halves hold comparable energy
        bool voiced = false;  ///< the dip reads as a note
    };

    /**
     * @brief Chooses the dip of the CMND in yinBuffer_ and decides voicing.
     *
     * Voicing: the chosen dip must be one the threshold prior accepts with
     * probability at least kVoicedSurvival, a ceiling on its CMND value
     * (0.30 at the default prior mean of 0.1). A periodic signal in white
     * noise dips to about the noise share of the power (0.24 at 5 dB SNR);
     * noise without a period has no dip near the ceiling (white noise
     * bottoms out near 0.85). A dip above the threshold itself also needs a
     * steady window, its two halves within kMinHalfEnergyRatio of each other
     * in energy: across a note's start or end the dip is shallow for another
     * reason, and the period read there is biased (measured 9 cents high a
     * hop after a 220 Hz note stopped), so those windows stay unvoiced as
     * under the single threshold. Nor may it be a period two notes of the
     * window share (sharesItsPeriod()).
     * @param window    The window the difference function was taken from.
     * @param energy    Its energy.
     * @param e1        The energy of its first half.
     * @param threshold The sensitivity threshold in effect.
     */
    [[nodiscard]] WindowReading readWindow(const T* window, T energy, T e1, T threshold) noexcept
    {
        int partial[kMaxPartials];
        int numPartial = 0;
        WindowReading reading;
        reading.tau = chooseDip(yinBuffer_.data(), partial, numPartial);
        reading.dip = (reading.tau < 0) ? 1.0
                    : static_cast<double>(yinBuffer_[static_cast<size_t>(reading.tau)]);
        const double firstHalf = static_cast<double>(e1);
        const double secondHalf = static_cast<double>(energy) - firstHalf;
        reading.steady = std::min(firstHalf, secondHalf)
                       >= kMinHalfEnergyRatio * std::max(firstHalf, secondHalf);
        reading.voiced = reading.tau >= 0 && reading.dip < voicingCeiling_
            && (reading.dip < static_cast<double>(threshold)
                || (reading.steady && !sharesItsPeriod(window, reading.tau, partial, numPartial)));
        return reading;
    }

    /**
     * @brief The input history through a high-pass below the lowest note the
     *        window can hold, oldest sample first (the layout of buffer_).
     *
     * A sixth-order Butterworth high-pass at kRumbleCorner of the lowest
     * frequency the window reads, run in double (its corner sits near a
     * thousandth of the sample rate). It runs here, over the samples that
     * arrived since the last call, so a stream that reads voiced at once
     * never pays for it; when more arrived than the history holds, the
     * filter starts over at the oldest sample kept and the first window of
     * the history carries its settling (the analysis window is the last).
     * A sample that is not finite restarts it as well.
     */
    [[nodiscard]] const T* clearHistory() noexcept
    {
        const T* raw = &buffer_[static_cast<size_t>(writePos_)];
        if (clearPending_ > historySize_)
        {
            for (RumbleSection& section : rumble_) section.s1 = section.s2 = 0.0;
            clearPending_ = historySize_;
        }
        const double limit = static_cast<double>(std::numeric_limits<T>::max());
        for (int k = historySize_ - clearPending_; k < historySize_; ++k)
        {
            double v = static_cast<double>(raw[k]);
            for (RumbleSection& section : rumble_)
            {
                const double y = section.b0 * v + section.s1;
                section.s1 = section.b1 * v - section.a1 * y + section.s2;
                section.s2 = section.b2 * v - section.a2 * y;
                v = y;
            }
            if (!(std::abs(v) <= limit))
            {
                for (RumbleSection& section : rumble_) section.s1 = section.s2 = 0.0;
                v = 0.0;
            }
            int position = writePos_ + k;
            if (position >= historySize_) position -= historySize_;
            clearBuffer_[static_cast<size_t>(position)] = static_cast<T>(v);
            clearBuffer_[static_cast<size_t>(position + historySize_)] = static_cast<T>(v);
        }
        clearPending_ = 0;
        // A state that has decayed to nothing is zero (no subnormal tail).
        for (RumbleSection& section : rumble_)
            if (std::abs(section.s1) < 1.0e-30 && std::abs(section.s2) < 1.0e-30)
                section.s1 = section.s2 = 0.0;
        return &clearBuffer_[static_cast<size_t>(writePos_)];
    }

    /** @brief Designs the high-pass of clearHistory() for the current window. */
    void designRumbleFilter() noexcept
    {
        // The lowest frequency the window reads: a period of halfWindow_ - 1.
        const double lowest = sampleRate_ / static_cast<double>(std::max(2, halfWindow_ - 1));
        const double corner = std::min(kRumbleCorner * lowest, 0.4 * sampleRate_);
        const double w0 = 6.283185307179586 * corner / sampleRate_;
        const double c = std::cos(w0), s = std::sin(w0);
        for (int k = 0; k < kRumbleSections; ++k)
        {
            // Butterworth: the k-th pole pair of an order 2 * kRumbleSections.
            const double q = 1.0 / (2.0 * std::cos(3.141592653589793 * (2.0 * k + 1.0)
                                                   / (4.0 * kRumbleSections)));
            const double alpha = s / (2.0 * q), a0 = 1.0 + alpha;
            RumbleSection& section = rumble_[static_cast<size_t>(k)];
            section.b0 = 0.5 * (1.0 + c) / a0;
            section.b1 = -2.0 * section.b0;
            section.b2 = section.b0;
            section.a1 = -2.0 * c / a0;
            section.a2 = (1.0 - alpha) / a0;
            section.s1 = section.s2 = 0.0;
        }
    }

    /**
     * @brief Records this detection's period candidate (0 for none) for
     *        agreesWithRecent().
     */
    void rememberCandidate(double period) noexcept
    {
        for (size_t i = recent_.size() - 1; i > 0; --i) recent_[i] = recent_[i - 1];
        recent_[0] = period;
    }

    /**
     * @brief True when at least kAgreementsNeeded of the last
     *        kRecentCandidates detections found a period within
     *        kAgreementCents of `period` per detection of age.
     *
     * A tone holds its period from one detection to the next; a narrow band
     * of noise wanders. A reading taken through the noise must be confirmed
     * by the detections before it: over 20 s of band-limited noise at 60 to
     * 1000 Hz and Q 2 to 10, the second read then voices at most 0.1% of
     * windows more than the first reading alone (up to 5% without), while
     * a steady tone in noise is confirmed one or two hops after it is first
     * found, and 50 cents of vibrato at 5.5 Hz still agrees.
     */
    [[nodiscard]] bool agreesWithRecent(double period) const noexcept
    {
        int agree = 0;
        for (int i = 0; i < kRecentCandidates; ++i)
        {
            const double p = recent_[static_cast<size_t>(i)];
            if (p > 0.0 && std::abs(1200.0 * std::log2(period / p)) <= kAgreementCents * (i + 1)) ++agree;
        }
        return agree >= kAgreementsNeeded;
    }

    void publish(T period, T confidence) noexcept
    {
        frequency_.store(static_cast<T>(sampleRate_) / period, std::memory_order_relaxed);
        confidence_.store(confidence, std::memory_order_relaxed);
    }

    void publishUnvoiced() noexcept
    {
        frequency_.store(T(0), std::memory_order_relaxed);
        confidence_.store(T(0), std::memory_order_relaxed);
    }

    /**
     * @brief YIN's difference function and its cumulative mean normalised
     *        form (CMND) for one window, by FFT cross-correlation.
     *
     *   d(tau) = E1 + E2(tau) - 2 r(tau)
     *
     * with E1 the energy of x[0..W) (constant), E2(tau) that of the shifted
     * window (prefix sums), and r(tau) the cross-correlation of the first half
     * against the full window, computed with 3 FFTs.
     *
     * @return False for a silent or non-finite window (nothing to read).
     */
    bool differenceFunction(const T* window, T* yin, T* raw, T& energy, T& e1) noexcept
    {
        const int W = halfWindow_;

        // (a) prefix sums of squared samples over the full window
        prefixSq_[0] = T(0);
        for (int i = 0; i < windowSize_; ++i)
            prefixSq_[static_cast<size_t>(i + 1)] =
                prefixSq_[static_cast<size_t>(i)] + window[i] * window[i];

        // Silence check on the window energy, the last prefix sum. It spans
        // the whole window, so it doubles as the non-finite gate: with any
        // NaN/Inf sample present the CMND would be all zeros ((NaN > 0) is
        // false) and read fs/2 at confidence 1.0 - a fake detection with
        // maximum confidence. Report unvoiced instead; the bad samples flush
        // out of the window on their own.
        energy = prefixSq_[static_cast<size_t>(windowSize_)];
        if (energy < T(1e-10) || !std::isfinite(energy)) return false;
        e1 = prefixSq_[static_cast<size_t>(W)];

        // (b) r(tau) via FFT cross-correlation: IFFT(conj(FFT(first half)) * FFT(window))
        std::copy(window, window + W, fftTime_.begin());
        std::fill(fftTime_.begin() + W, fftTime_.end(), T(0));
        fft_->forward(fftTime_.data(), specHalf_.data());

        std::copy(window, window + windowSize_, fftTime_.begin());
        fft_->forward(fftTime_.data(), specFull_.data());   // tail beyond the window is still zero

        // Power of the window in its three lowest bins (detect() screens
        // for rumble with it): bin spacing is half the lowest frequency the
        // window reads, so they reach up to that note at any window size.
        {
            double low = 0.0;
            const int lowBins = std::min(3, fftSize_ / 2 + 1);
            for (int k = 0; k < lowBins; ++k)
            {
                const double re = static_cast<double>(specFull_[static_cast<size_t>(2 * k)]);
                const double im = static_cast<double>(specFull_[static_cast<size_t>(2 * k + 1)]);
                low += (k == 0 ? 1.0 : 2.0) * (re * re + im * im);
            }
            lowBandPower_ = low / static_cast<double>(fftSize_);
            // The two lowest alone lie below every note the window reads.
            double sub = 0.0;
            for (int k = 0; k < std::min(2, lowBins); ++k)
            {
                const double re = static_cast<double>(specFull_[static_cast<size_t>(2 * k)]);
                const double im = static_cast<double>(specFull_[static_cast<size_t>(2 * k + 1)]);
                sub += (k == 0 ? 1.0 : 2.0) * (re * re + im * im);
            }
            subBandPower_ = sub / static_cast<double>(fftSize_);
        }

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
        yin[0] = T(1);
        T runningSum = T(0);
        for (int tau = 1; tau < W; ++tau)
        {
            const T e2 = prefixSq_[static_cast<size_t>(tau + W)] - prefixSq_[static_cast<size_t>(tau)];
            T d = e1 + e2 - T(2) * corrTime_[static_cast<size_t>(tau)];
            if (d < T(0)) d = T(0); // guard tiny negative round-off

            raw[tau] = d;
            runningSum += d;
            yin[tau] = (runningSum > T(0)) ? d * static_cast<T>(tau) / runningSum : T(1);
        }
        return true;
    }

    /**
     * @brief Candidate selection over a distribution of thresholds (the frame
     *        stage of Mauch & Dixon 2014); returns the chosen lag, or -1.
     *
     * A threshold s picks the first dip of the CMND below s, and a dip is an
     * excursion: the stretch of lags where the CMND stays below s, read at
     * its deepest minimum. Scanning the minima in lag order, a minimum lower
     * than every earlier one (a record) is the first one below the
     * thresholds in (its value, the previous record]; the highest CMND
     * sample between two consecutive records (their barrier) decides whether
     * a threshold sees them as one excursion. So a record keeps the
     * thresholds up to its barrier, and those above it pass on to the next,
     * deeper record (and on, while every barrier on the way stays below the
     * threshold). Each record's probability is the prior mass of the
     * thresholds it ends up with. Thresholds below the deepest minimum pick
     * nothing and hand kAbsoluteMinWeight of their mass to the deepest
     * minimum. The most probable candidate wins; ties go to the shorter lag.
     * The records below kPartialLevel go to `partial` (at most
     * kMaxPartials), for sharesItsPeriod().
     *
     * Reading each threshold's first minimum instead splits a broad dip
     * across the small minima that noise leaves on its walls: a 47 to 67 Hz
     * tone at 10 dB SNR has dozens of them on the way down to its period,
     * each with a sliver of the mass, and one on the wall won, reading the
     * tone 50 to 140 cents sharp at confidence 0.81 to 0.89 (31.5% of 200
     * tones). Separate dips stay apart: between a period and its multiple
     * the CMND rises far above any threshold the prior weighs.
     *
     * A dip is valued at the vertex of the parabola through its three
     * samples: a period that falls between two lags (a high note, a sweep)
     * has a shallow sample at its true minimum, and valued at the samples an
     * exact-integer multiple of it outvoted it (measured: a 3.3 kHz sawtooth
     * at 44.1 kHz read an octave low at confidence 0.997). At a strict
     * minimum the vertex lies within half a lag and its value is at least
     * c1 - |c0 - c2| / 8; that bound, in the sample type and loosened by
     * kScreenSlack (far above its rounding error), rejects nearly every lag
     * before the exact test in double. The screen is branch-free on purpose:
     * on noise the minima fall at random lags.
     *
     * A choice above the threshold is a reading through noise, and there the
     * dips at a period and at its multiples are equally deep but for the
     * noise, so the deepest is as often the second or third multiple as the
     * period: 17.5% of the windows of 70 to 1000 Hz tones at 5 dB SNR read a
     * multiple of their period. Such a choice yields to the first earlier
     * record within kMultipleMargin of it that is a dip of its own, with the
     * CMND rising kMultipleBarrier above both in between (a minimum on the
     * wall of the chosen dip is not). That leaves 0.2% of those windows
     * wrong; a tone whose odd partials hold under a tenth of its power, which
     * also repeats at half its period, then reads an octave up in noise.
     */
    [[nodiscard]] int chooseDip(const T* c, int* partial, int& numPartial) noexcept
    {
        const int W = halfWindow_;
        DipRecord* rec = dipRecords_.data();
        const int capacity = static_cast<int>(dipRecords_.size());
        DipGroup* groups = dipGroups_.data();
        int numGroups = 0;
        double passing = 0.0;   // prior mass that passes on to the newest record
        int count = 0, leader = -1;
        double leaderMass = -1.0, recordValue = 1.0, recordSurvival = 0.0;
        T screen = T(1) + kScreenSlack;
        T barrier = T(0);       // highest CMND sample since the last record
        T stopLevel = std::numeric_limits<T>::infinity();   // settled once the barrier reaches it
        numPartial = 0;
        for (int tau = 2; tau + 1 < W; ++tau)
        {
            const T c0 = c[tau - 1], c1 = c[tau], c2 = c[tau + 1];
            barrier = std::max(barrier, c0);
            if (barrier >= stopLevel) break;
            if (!((c1 < c0) & (c1 <= c2) & (c1 - std::abs(c0 - c2) * T(0.125) < screen)))
                continue;
            const double d0 = static_cast<double>(c0), d1 = static_cast<double>(c1),
                         d2 = static_cast<double>(c2);
            const double v = std::max(0.0, d1 - (d0 - d2) * (d0 - d2) / (8.0 * (d0 - 2.0 * d1 + d2)));
            if (!(v < recordValue) || count == capacity) continue;
            const double s = priorSurvival(v);
            DipRecord& r = rec[count];
            r.lag = tau;
            r.value = v;
            r.survival = s;
            r.mass = s - recordSurvival;   // its own thresholds, (v, previous record]
            if (count > 0)
            {
                // The previous record's barrier is now known: every threshold
                // above it that reached that record passes on to this one.
                // Thresholds in (rec[i].value, rec[i - 1].value] first meet
                // record i and reach this one when they clear every barrier
                // from record i on. The records with the same highest such
                // barrier form a group on a stack, so each one is merged once
                // and the passing mass is a running total.
                DipRecord& prev = rec[count - 1];
                prev.barrier = static_cast<double>(barrier);
                prev.barrierSurvival = priorSurvival(prev.barrier);
                int first = count - 1;
                while (numGroups > 0 && groups[numGroups - 1].gate <= prev.barrier)
                {
                    --numGroups;
                    passing -= groups[numGroups].mass;
                    first = groups[numGroups].first;
                }
                // Records first..count-1 now share this barrier; those whose
                // upper bound (the record before them, 1 for the first) lies
                // above it pass the thresholds in between. The bounds fall
                // with the index: find the first one at or below it.
                int lo = first, hi = count;
                while (lo < hi)
                {
                    const int mid = (lo + hi) / 2;
                    const double upper = (mid == 0) ? 1.0 : rec[mid - 1].value;
                    if (upper > prev.barrier) lo = mid + 1; else hi = mid;
                }
                const auto upperSum = [&](int i) noexcept { return (i <= 1) ? 0.0 : rec[i - 1].prefix; };
                DipGroup& g = groups[numGroups++];
                g.first = first;
                g.gate = prev.barrier;
                g.mass = static_cast<double>(lo - first) * prev.barrierSurvival
                       - (upperSum(lo) - upperSum(first));
                passing += g.mass;
                prev.mass -= passing;
                r.mass += passing;
                if (prev.mass > leaderMass) { leaderMass = prev.mass; leader = count - 1; }
            }
            r.prefix = (count == 0) ? 0.0 : rec[count - 1].prefix + rec[count - 1].survival;
            ++count;
            recordValue = v;
            recordSurvival = s;
            if (v < kPartialLevel && numPartial < kMaxPartials) partial[numPartial++] = tau;
            screen = static_cast<T>(v) + kScreenSlack;
            barrier = c1;
            stopLevel = settledAbove(r.mass, s, leaderMass);
        }
        if (count == 0) return -1;
        const DipRecord& last = rec[count - 1];
        const double lastMass = last.mass + kAbsoluteMinWeight * (1.0 - last.survival);
        int chosen = (lastMass > leaderMass) ? count - 1 : leader;
        // A choice above the threshold yields to the first separate dip
        // about as deep (see above).
        const double value = rec[chosen].value;
        if (value > static_cast<double>(cachedThreshold_))
        {
            const double limit = value + kMultipleMargin;
            for (int i = 0; i < chosen; ++i)
                if (rec[i].value <= limit && rec[i].barrier >= limit + kMultipleBarrier)
                {
                    chosen = i;
                    break;
                }
        }
        return rec[chosen].lag;
    }

    /**
     * @brief The barrier above the newest record at which the choice of
     *        chooseDip() is settled (infinity while it cannot be).
     *
     * Later lags can only add records deeper than the newest one, which
     * gather at most the prior mass below it (plus the deepest minimum's
     * share) and the thresholds above the newest record's barrier, which
     * pass on to them: no more than the survival at that barrier. Once the
     * leader keeps more than any later record could gather, no later lag
     * changes the choice and the scan stops with the full scan's result (a
     * clean dip ends it a few lags past its bottom).
     * @param mass        The newest record's mass so far.
     * @param survival    Prior survival at its value.
     * @param leaderMass  The best mass among the earlier records (-1 if none).
     */
    [[nodiscard]] T settledAbove(double mass, double survival, double leaderMass) const noexcept
    {
        const double below = 1.0 - survival;
        const double future = (1.0 + kAbsoluteMinWeight) * below;
        const bool earlierLeads = leaderMass >= mass + kAbsoluteMinWeight * below;
        double target = earlierLeads ? leaderMass - future
                                     : std::min(mass - leaderMass, 0.5 * (mass - future));
        target -= kSettleMargin;
        if (!(target > 0.0)) return std::numeric_limits<T>::infinity();
        if (earlierLeads && mass <= target) return T(-1);
        // The first node of the survival table at or below the target (the
        // table holds the exact survival at its nodes, and it falls
        // monotonically).
        int lo = 0, hi = kSurvivalSteps;
        while (lo < hi)
        {
            const int mid = (lo + hi) / 2;
            if (survivalTable_[static_cast<size_t>(mid)] > target) lo = mid + 1; else hi = mid;
        }
        return static_cast<T>(static_cast<double>(lo) / static_cast<double>(kSurvivalSteps));
    }

    /**
     * @brief The window with its broadband noise floor removed (spectral
     *        subtraction), for the reading through noise.
     *
     * The window, zero-padded to twice its length, is transformed; the noise
     * floor of each band of kFloorBins bins is the median power there over
     * ln 2 (the median of a noise bin's exponential power distribution),
     * interpolated across bands, so it follows coloured noise; each bin keeps
     * max(0, 1 - kOverSubtraction * floor / power) of itself. A harmonic's
     * bins stand far above the floor (its power is concentrated, the noise's
     * spread over the band) and pass almost untouched; noise-only bins go to
     * zero. Model-free: no training, no assumption on the source beyond a
     * floor that is smooth across a band.
     */
    const T* suppressNoise(const T* window) noexcept
    {
        const int n = windowSize_;
        std::copy(window, window + n, cleanTime_.begin());
        std::fill(cleanTime_.begin() + n, cleanTime_.end(), T(0));
        fftClean_->forward(cleanTime_.data(), cleanSpec_.data());
        const int bins = n + 1;
        for (int k = 0; k < bins; ++k)
        {
            const T re = cleanSpec_[static_cast<size_t>(2 * k)], im = cleanSpec_[static_cast<size_t>(2 * k + 1)];
            cleanPower_[static_cast<size_t>(k)] = re * re + im * im;
        }
        const int bands = (bins + kFloorBins - 1) / kFloorBins;
        for (int b = 0; b < bands; ++b)
        {
            const int lo = b * kFloorBins, hi = std::min(bins, lo + kFloorBins);
            std::copy(cleanPower_.begin() + lo, cleanPower_.begin() + hi, cleanScratch_.begin());
            const auto mid = cleanScratch_.begin() + (hi - lo) / 2;
            std::nth_element(cleanScratch_.begin(), mid, cleanScratch_.begin() + (hi - lo));
            cleanFloor_[static_cast<size_t>(b)] = *mid * T(1.4426950408889634);   // / ln 2
        }
        for (int k = 0; k < bins; ++k)
        {
            // The band floors sit at the band centres; linear in between.
            const double u = (static_cast<double>(k) + 0.5) / kFloorBins - 0.5;
            const int b0 = std::clamp(static_cast<int>(std::floor(u)), 0, bands - 1);
            const int b1 = std::min(b0 + 1, bands - 1);
            const T frac = static_cast<T>(std::clamp(u - static_cast<double>(b0), 0.0, 1.0));
            const T floorK = cleanFloor_[static_cast<size_t>(b0)]
                           + frac * (cleanFloor_[static_cast<size_t>(b1)] - cleanFloor_[static_cast<size_t>(b0)]);
            const T power = cleanPower_[static_cast<size_t>(k)];
            const T gain = power > T(0) ? std::max(T(0), T(1) - T(kOverSubtraction) * floorK / power) : T(0);
            cleanSpec_[static_cast<size_t>(2 * k)] *= gain;
            cleanSpec_[static_cast<size_t>(2 * k + 1)] *= gain;
        }
        fftClean_->inverse(cleanSpec_.data(), cleanTime_.data());
        return cleanTime_.data();
    }

    /**
     * @brief True when the dip at `tau` is a period two different signals in
     *        the window share rather than the period of one.
     *
     * Across a note change the two notes can share a longer period (4 periods
     * of one, 5 of the other, a major third apart): both halves of the window
     * repeat at that lag, so its dip is moderately deep, while each note's
     * own period, a divisor of it, repeats in one half only. A steady tone
     * whose period is `tau` repeats the same way in both halves at every lag,
     * so a divisor's difference is about equal in them. The test looks at the
     * earlier minima deep enough to be a partial period (kPartialLevel) that
     * divide `tau`, and compares the normalised difference of the two halves
     * of the window at each: one half repeating where the other does not
     * (kStationaryRatio) marks the shared period. Direct sums over the
     * window, a few per detection, only for dips above the threshold.
     */
    [[nodiscard]] bool sharesItsPeriod(const T* x, int tau, const int* partial, int count) const noexcept
    {
        const int W = halfWindow_;
        const int H = W / 2;
        for (int r = 0; r < count; ++r)
        {
            const int lag = partial[r];
            if (lag >= tau) break;
            const int k = static_cast<int>(std::lround(static_cast<double>(tau) / static_cast<double>(lag)));
            if (k < 2 || std::abs(static_cast<double>(tau - k * lag)) > 0.02 * static_cast<double>(k * lag) + 1.0)
                continue;
            double part[2] = { 0.0, 0.0 };
            for (int h = 0; h < 2; ++h)
            {
                double diff = 0.0, power = 0.0;
                for (int j = h * H; j < (h + 1) * H; ++j)
                {
                    const double a = static_cast<double>(x[j]), b = static_cast<double>(x[j + lag]);
                    diff += (a - b) * (a - b);
                    power += a * a + b * b;
                }
                part[h] = power > 0.0 ? diff / power : 1.0;
            }
            if (std::min(part[0], part[1]) < kStationaryRatio * std::max(part[0], part[1]))
                return true;
        }
        return false;
    }

    /**
     * @brief The lowest sample of the input's own difference function within
     *        kRefineSpan of `tau` (the lag the cleaned window dipped at).
     */
    [[nodiscard]] int inputDipNear(int tau) const noexcept
    {
        const T* raw = rawDiff_.data();
        const int lo = std::max(2, static_cast<int>(static_cast<double>(tau) * (1.0 - kRefineSpan)));
        const int hi = std::min(halfWindow_ - 2, static_cast<int>(static_cast<double>(tau) * (1.0 + kRefineSpan)) + 1);
        int best = std::clamp(tau, lo, hi);
        for (int t = lo; t <= hi; ++t)
            if (raw[t] < raw[best]) best = t;
        return best;
    }

    /**
     * @brief Prefix sums of the squared history (persistSq_), computed on
     *        first use in a detection and shared by refineByMultiples() and
     *        periodPersists().
     * @param history The input history, kHistoryWindows windows, oldest first.
     */
    void historyEnergy(const T* history) noexcept
    {
        if (historySumsReady_) return;
        historySumsReady_ = true;
        // Four samples at a time with one carry, so the running sum is not a
        // chain of dependent additions (historySize_ is a multiple of 4).
        double* sq = persistSq_.data();
        sq[0] = 0.0;
        double carry = 0.0;
        for (int i = 0; i < historySize_; i += 4)
        {
            const double a = static_cast<double>(history[i]) * static_cast<double>(history[i]);
            const double b = a + static_cast<double>(history[i + 1]) * static_cast<double>(history[i + 1]);
            const double c = b + static_cast<double>(history[i + 2]) * static_cast<double>(history[i + 2]);
            const double d = c + static_cast<double>(history[i + 3]) * static_cast<double>(history[i + 3]);
            sq[i + 1] = carry + a;
            sq[i + 2] = carry + b;
            sq[i + 3] = carry + c;
            sq[i + 4] = carry + d;
            carry += d;
        }
    }

    /**
     * @brief Normalised correlation of the history with itself at `lag`, over
     *        its first `span` samples (historyEnergy() must have run).
     */
    [[nodiscard]] double historyCorrelation(const T* x, int lag, int span) const noexcept
    {
        const double* sq = persistSq_.data();
        const double e = sq[span] + sq[lag + span] - sq[lag];
        const double r = static_cast<double>(simd::dotProductT(x, x + lag, span));
        return e > 0.0 ? 2.0 * r / e : 0.0;
    }

    /**
     * @brief A period refined from the correlation peaks at its multiples
     *        over the history.
     *
     * Through strong noise the analysis window's own dip wanders: pink
     * noise, strong near a low note, tilts the difference function around
     * the period and shifted 47 to 67 Hz tones at 0 dB SNR 50 to 135 cents
     * (one window in five). The history holds two to eight periods of
     * every period the detector reads; the peak of the correlation near k
     * periods gives the period to 1/k of its own error, and the noise's
     * tilt is flatter there. Each multiple k from 2 to kRefineMultiples that
     * leaves an eighth of a window to correlate over is searched near k times
     * the estimate so far (kMultipleSearch of it at k = 2, narrowing as the
     * estimate firms up; every few lags first, then every lag around the
     * best; parabolic peak), and the estimate becomes the mean of the
     * window's period and each multiple's, weighted by k squared. Over the
     * same tones (one window each, the input's own dip refined in double),
     * the period is then within 50 cents in 94.5% of windows where the
     * window alone gave 80.5%; higher multiples added no accuracy.
     * @param x      The input history, kHistoryWindows windows, oldest first.
     * @param period The period found in the analysis window, in samples.
     */
    [[nodiscard]] double refineByMultiples(const T* x, double period) noexcept
    {
        if (!(period >= 2.0)) return period;
        historyEnergy(x);
        const int n = historySize_;
        double weighted = period, weight = 1.0, estimate = period;
        double* values = multipleScratch_.data();
        const int capacity = static_cast<int>(multipleScratch_.size());
        for (int k = 2; k <= kRefineMultiples; ++k)
        {
            const int centre = static_cast<int>(std::lround(static_cast<double>(k) * estimate));
            const double search = static_cast<double>(k) * estimate * kMultipleSearch
                                / static_cast<double>(std::max(1, k - 1));
            const int reach = std::max(2, static_cast<int>(std::ceil(search)));
            const int span = n - (centre + reach) - 1;
            const int count = 2 * reach + 1;
            if (span < windowSize_ / 8 || count > capacity) break;
            // Every step-th lag first, then every lag within a step of the
            // best of them: the peak near k periods is several lags wide.
            const int step = std::max(1, reach / kSearchStepDivisor);
            int best = -1;
            const auto evaluate = [&](int j) noexcept {
                values[j] = historyCorrelation(x, centre - reach + j, span);
                if (best < 0 || values[j] > values[best]) best = j;
            };
            for (int j = 0; j < count; j += step) evaluate(j);
            if (step > 1)
            {
                const int coarse = best;
                for (int j = std::max(0, coarse - step + 1); j <= std::min(count - 1, coarse + step - 1); ++j)
                    if (j % step != 0) evaluate(j);
            }
            double peak = static_cast<double>(centre - reach + best);
            if (best > 0 && best + 1 < count)
            {
                const double a = values[best - 1], b = values[best], c = values[best + 1];
                const double curvature = a - 2.0 * b + c;
                if (curvature < -1e-30) peak += std::clamp(0.5 * (a - c) / curvature, -0.5, 0.5);
            }
            const double w = static_cast<double>(k) * static_cast<double>(k);
            weighted += w * peak / static_cast<double>(k);
            weight += w;
            estimate = weighted / weight;
        }
        return estimate;
    }

    /**
     * @brief True when the input repeats at the multiples of `period` about
     *        as strongly as at `period` itself, over the last kHistoryWindows
     *        windows of input (the analysis window and the one before it).
     *
     * A tone in noise repeats equally at every multiple of its period; a
     * narrow band of noise that the cleaned window made look periodic
     * decorrelates at each one (by exp(-pi / Q) per period at the band's
     * centre). The test compares the periodic part of the window's
     * correlation at 2, 3 .. kMaxMultiple periods with that at one period:
     * each is the correlation's peak there (the best of the three lags
     * around the multiple) above the mean of the lags half a period either
     * side, so a smooth noise correlation, such as pink noise's, cancels.
     * The mean of the multiples must reach kPersistRatio of the first. All
     * the multiples that leave an eighth of a window to correlate over are
     * used, over one common span. The history holds at least three periods
     * of every period the detector reads, so the lowest notes are tested too,
     * and the longer span and the extra multiples make a narrow band of noise
     * less likely to pass than over the window alone.
     * @param history  The input history, kHistoryWindows windows, oldest first.
     * @param period   The period under test, in samples.
     * @param contrast Receives the periodic part of the correlation at one
     *                 period (0 when the test cannot run).
     */
    [[nodiscard]] bool periodPersists(const T* history, double period, double& contrast) noexcept
    {
        contrast = 0.0;
        historyEnergy(history);
        if (!(period >= 2.0)) return false;
        const auto reach = [&](int k) noexcept {
            return static_cast<int>(std::lround((static_cast<double>(k) + 0.5) * period)) + 2;
        };
        const int minSpan = windowSize_ / 8;
        const int n = historySize_;
        int kmax = kMaxMultiple;
        while (kmax >= 2 && n - reach(kmax) < minSpan) --kmax;
        if (kmax < 2) return false;
        const int span = n - reach(kmax);
        const auto corr = [&](int lag) noexcept { return historyCorrelation(history, lag, span); };
        const auto periodic = [&](int k) noexcept {
            const int c = static_cast<int>(std::lround(static_cast<double>(k) * period));
            const int h = static_cast<int>(std::lround(0.5 * period));
            const double peak = std::max({ corr(c - 1), corr(c), corr(c + 1) });
            return peak - 0.5 * (corr(c - h) + corr(c + h));
        };
        const double first = periodic(1);
        contrast = first;
        if (!(first > 0.0)) return false;
        double sum = 0.0;
        for (int k = 2; k <= kmax; ++k) sum += periodic(k);
        return sum >= kPersistRatio * first * static_cast<double>(kmax - 1);
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
    [[nodiscard]] T parabolicInterp(const T* raw, int tau) const noexcept
    {
        if (tau < 2 || tau >= halfWindow_ - 1)
            return static_cast<T>(tau);

        // Centre on the raw function's own lowest sample next to the CMND dip.
        const int centre = tau;
        if (raw[centre - 1] < raw[tau]) tau = centre - 1;
        if (raw[centre + 1] < raw[tau]) tau = centre + 1;
        if (tau < 2 || tau >= halfWindow_ - 1)
            return static_cast<T>(tau);

        const double s0 = static_cast<double>(raw[tau - 1]);
        const double s1 = static_cast<double>(raw[tau]);
        const double s2 = static_cast<double>(raw[tau + 1]);

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
    [[nodiscard]] double exactSurvival(double x) const noexcept
    {
        if (!(x > 0.0)) return 1.0;
        if (!(x < 1.0)) return 0.0;
        return std::pow(1.0 - x, priorB_) * (1.0 + priorB_ * x);
    }

    /**
     * @brief exactSurvival() from a table built by updatePrior(): cubic
     *        Hermite interpolation on kSurvivalSteps intervals with the exact
     *        derivative, -b (b + 1) x (1 - x)^(b - 1), at each node.
     *
     * The error is below 1e-9 at the default prior and 1e-5 at the
     * strictest. chooseDip() evaluates it at every record and barrier, where
     * a pow each cost a noisy window about a microsecond.
     */
    [[nodiscard]] double priorSurvival(double x) const noexcept
    {
        if (!(x > 0.0)) return 1.0;
        if (!(x < 1.0)) return 0.0;
        const double u = x * static_cast<double>(kSurvivalSteps);
        const int i = std::min(static_cast<int>(u), kSurvivalSteps - 1);
        const double t = u - static_cast<double>(i), h = 1.0 / static_cast<double>(kSurvivalSteps);
        const double* s = survivalTable_.data();
        const double* d = survivalSlope_.data();
        const double t2 = t * t, t3 = t2 * t;
        return (2.0 * t3 - 3.0 * t2 + 1.0) * s[i] + (t3 - 2.0 * t2 + t) * h * d[i]
             + (3.0 * t2 - 2.0 * t3) * s[i + 1] + (t3 - t2) * h * d[i + 1];
    }

    /**
     * @brief Rebuilds the threshold prior for a new mean (audio thread, on a
     *        threshold change only): Beta(2, b) with mean 2 / (2 + b), its
     *        survival table, and the CMND ceiling x where its survival falls
     *        to kVoicedSurvival, never above kMaxVoicingCeiling.
     */
    void updatePrior(T threshold) noexcept
    {
        cachedThreshold_ = threshold;
        const double mean = static_cast<double>(threshold);
        priorB_ = 2.0 * (1.0 - mean) / mean;
        for (int i = 0; i <= kSurvivalSteps; ++i)
        {
            const double x = static_cast<double>(i) / static_cast<double>(kSurvivalSteps);
            survivalTable_[static_cast<size_t>(i)] = (i == 0) ? 1.0 : exactSurvival(x);
            survivalSlope_[static_cast<size_t>(i)] = (i == kSurvivalSteps) ? 0.0
                : -priorB_ * (priorB_ + 1.0) * x * std::pow(1.0 - x, priorB_ - 1.0);
        }
        double lo = 0.0, hi = 1.0;   // survival falls monotonically from 1 to 0
        for (int i = 0; i < 48; ++i)
        {
            const double mid = 0.5 * (lo + hi);
            if (exactSurvival(mid) > kVoicedSurvival) lo = mid; else hi = mid;
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
    /// Deepest normalised difference of an earlier minimum that the
    /// shared-period test (sharesItsPeriod()) considers a partial period.
    static constexpr double kPartialLevel = 0.75;
    /// Ratio of the window halves' differences at a partial period below
    /// which one half repeats there and the other does not.
    static constexpr double kStationaryRatio = 0.25;
    /// Partial periods a detection keeps for that test.
    static constexpr int kMaxPartials = 16;
    /// Span, as a fraction of the lag, around a cleaned dip in which the
    /// input's own difference function is searched for the period.
    static constexpr double kRefineSpan = 0.06;
    /// Most multiples of a period that periodPersists() correlates at.
    static constexpr int kMaxMultiple = 8;
    /// Least mean of the periodic correlation at the multiples, relative to
    /// that at one period, for a reading taken through the noise (a narrow
    /// band of Q 5 measures about 0.2, a tone about 1).
    static constexpr double kPersistRatio = 0.6;
    /// Bins per band of the noise floor estimate (375 Hz at the automatic window).
    static constexpr int kFloorBins = 32;
    /// Over-subtraction of the noise floor: bins below twice the floor go to zero.
    static constexpr double kOverSubtraction = 2.0;
    /// Windows of input history kept for periodPersists().
    static constexpr int kHistoryWindows = 2;
    /// Search span, as a fraction of k times the period, for the correlation
    /// peak near each multiple in refineByMultiples().
    static constexpr double kMultipleSearch = 0.03;
    /// Highest multiple refineByMultiples() reads (more add CPU, not accuracy).
    static constexpr int kRefineMultiples = 4;
    /// Its coarse search steps over the span / this many lags each side.
    static constexpr int kSearchStepDivisor = 16;
    /// Deepest input dip a fallback reading may have (white noise bottoms out
    /// near 0.85; a tone at 0 dB SNR dips to about 0.5).
    static constexpr double kFallbackCeiling = 0.8;
    /// Relative distance within which the input's own dip is the candidate
    /// the cleaned window already found (and is not read twice).
    static constexpr double kSameCandidate = 0.02;
    /// Periodic correlation contrast at one period a fallback reading needs:
    /// a tone at 0 dB SNR measures about 0.9 (0.35 to 1.3); low-passed noise
    /// alone mostly below 0.25, a narrow band of noise at Q 5 to 10 up to 1.8.
    static constexpr double kMinFallbackContrast = 0.35;
    static constexpr double kMaxFallbackContrast = 1.3;
    /// Detections remembered for agreesWithRecent().
    static constexpr int kRecentCandidates = 3;
    /// Of them, how many must agree with a reading taken through the noise.
    static constexpr int kAgreementsNeeded = 2;
    /// Agreement tolerance for that test, in cents per detection of age (a
    /// 50-cent vibrato at 5.5 Hz moves up to 18.5 cents per detection).
    static constexpr double kAgreementCents = 25.0;
    /// Intervals of the prior survival table (priorSurvival()).
    static constexpr int kSurvivalSteps = 1024;
    /// Margin of the settling test in chooseDip() (prior mass).
    static constexpr double kSettleMargin = 1.0e-9;
    /// How much deeper a choice above the threshold must be than an earlier
    /// separate dip to keep it (in noise the dips at a period and at its
    /// multiples differ by a few hundredths), and how far the CMND must rise
    /// above both in between for them to be separate dips.
    static constexpr double kMultipleMargin = 0.15;
    static constexpr double kMultipleBarrier = 0.25;
    /// Second look without rumble (clearHistory()): sections of its
    /// Butterworth high-pass (sixth order), the corner as a fraction of the
    /// lowest frequency the window reads, and the share of the window's power
    /// the filter must remove for the look to be taken.
    static constexpr int kRumbleSections = 3;
    static constexpr double kRumbleCorner = 0.9;
    static constexpr double kRumbleShare = 0.02;
    /// Share of the window's power in its two lowest bins (below every note
    /// it reads) from which a dip above the threshold must also pass
    /// periodPersists() at the first look.
    static constexpr double kSubRumbleShare = 0.10;

    /// Automatic-window policy: the span 2048 samples cover at 48 kHz,
    /// resolved inside [512, 16384] (covers 8 kHz .. 384 kHz without clamping).
    static constexpr double kAutoSpanRef = 2048.0;
    static constexpr double kAutoSpanRate = 48000.0;
    static constexpr int kAutoMinWindow = 512;
    static constexpr int kAutoMaxWindow = 16384;

    double sampleRate_ = 44100.0;
    int windowSize_ = 2048;
    int historySize_ = 4096;
    int halfWindow_ = 1024;
    int hopSize_ = 512;
    int writePos_ = 0;
    int samplesSinceLastDetect_ = 0;
    std::atomic<T> threshold_{ T(0.10) };

    // Threshold prior as last built by updatePrior() (audio thread only).
    T cachedThreshold_ = T(-1);
    double priorB_ = 18.0;
    double voicingCeiling_ = 0.0;
    std::vector<double> survivalTable_;   // Size: kSurvivalSteps + 1 (priorSurvival())
    std::vector<double> survivalSlope_;   // Its derivative at the same nodes

    // Thread-safe outputs
    std::atomic<T> frequency_{T(0)};
    std::atomic<T> confidence_{T(0)};

    // Mirrored buffer keeps every analysis window contiguous
    std::vector<T> buffer_;     // Size: 2 * historySize_
    std::vector<T> yinBuffer_;  // Size: halfWindow_
    std::vector<T> rawDiff_;    // Size: halfWindow_ (difference function d(tau))

    // The history without rumble (clearHistory())
    struct RumbleSection
    {
        double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;  ///< high-pass section
        double s1 = 0.0, s2 = 0.0;                                ///< its state
    };
    std::array<RumbleSection, kRumbleSections> rumble_ {};
    std::vector<T> clearBuffer_;   // Size: 2 * historySize_, mirrored like buffer_
    int clearPending_ = 0;         // Newest input samples not yet filtered into it
    double lowBandPower_ = 0.0;    // Power in the three lowest bins of the last window transformed
    double subBandPower_ = 0.0;    // Power in the two lowest of them
    std::vector<T> yinClear_;      // Size: halfWindow_ (CMND of the rumble-free window)
    std::vector<T> rawClear_;      // Size: halfWindow_ (its difference function)

    // YIN-FFT resources (cross-correlation difference function)
    int fftSize_ = 2048;
    std::unique_ptr<FFTReal<T>> fft_; // doubles as the "prepared" gate
    std::vector<T> fftTime_;    // Size: fftSize_
    std::vector<T> specHalf_;   // Size: fftSize_ + 2
    std::vector<T> specFull_;   // Size: fftSize_ + 2
    std::vector<T> corrTime_;   // Size: fftSize_
    std::vector<T> prefixSq_;   // Size: windowSize_ + 1

    // Noise floor suppression (see suppressNoise())
    std::atomic<bool> noiseSuppression_{ true };
    std::unique_ptr<FFTReal<T>> fftClean_;   // 2 * windowSize_
    std::vector<T> cleanTime_;    // Size: 2 * windowSize_ (the cleaned window first)
    std::vector<T> cleanSpec_;    // Size: 2 * windowSize_ + 2
    std::vector<T> cleanPower_;   // Size: windowSize_ + 1
    std::vector<T> cleanFloor_;   // One floor per band of kFloorBins bins
    std::vector<T> cleanScratch_; // Size: kFloorBins
    std::vector<T> yinClean_;     // Size: halfWindow_
    std::vector<T> rawClean_;     // Size: halfWindow_
    std::vector<double> persistSq_; // Size: historySize_ + 1 (historyEnergy())
    bool historySumsReady_ = false;       // persistSq_ holds this detection's sums
    std::vector<double> multipleScratch_; // Correlations around one multiple (refineByMultiples())

    /// One record of chooseDip()'s scan: a minimum deeper than every earlier one.
    struct DipRecord
    {
        int lag = 0;
        double value = 0.0, survival = 0.0;          ///< vertex value, prior survival there
        double barrier = 0.0, barrierSurvival = 0.0; ///< highest CMND sample before the next record
        double mass = 0.0;                           ///< prior mass of the thresholds it holds
        double prefix = 0.0;                         ///< sum of the earlier records' survivals
    };
    std::vector<DipRecord> dipRecords_;   // Size: halfWindow_ / 2 + 2 (chooseDip())
    /// Records of chooseDip() that share the highest barrier up to the newest one.
    struct DipGroup
    {
        int first = 0;        ///< first record of the group (the next group's first ends it)
        double gate = 0.0;    ///< the barrier they share
        double mass = 0.0;    ///< the prior mass they pass on
    };
    std::vector<DipGroup> dipGroups_;     // Same size as dipRecords_ (chooseDip())
    std::array<double, kRecentCandidates> recent_ {}; // Last detections' periods (agreesWithRecent())
};

} // namespace dspark
