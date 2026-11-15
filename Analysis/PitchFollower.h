// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PitchFollower.h
 * @brief Musical pitch tracker: PitchDetector + the logic that makes it usable.
 *
 * A raw pitch detector is not directly usable as a modulation source: it
 * reports octave errors on transients, garbage with high confidence during
 * noise bursts, and jumps that would make any controlled parameter zipper.
 * PitchFollower wraps Analysis/PitchDetector.h with the control logic every
 * pitch-driven effect needs:
 *
 * - **Confidence gating** - readings below the confidence threshold (or
 *   outside the configured frequency range) never move the output; the last
 *   reliable pitch is held through consonants and silence.
 * - **Octave-jump correction** - a single reading ~1-2 octaves away from
 *   the tracked pitch is folded back to the closest octave of the current
 *   target (the classic YIN 2x/0.5x failure on transients).
 * - **Jump confirmation** - a large interval, an octave leap among them,
 *   must persist for two consecutive readings before the target moves (one
 *   bad frame never jerks the output). A reading is one detector detection,
 *   whatever the host's block size, so the block size changes neither what
 *   is rejected nor how long a real interval takes to arrive. Folding every
 *   octave-away reading instead never let a melody's octave leap through,
 *   and locked onto a wrong octave the detector read at a note's start:
 *   measured on legato melodies with an octave leap in one step of eight,
 *   21.9% of voiced frames an octave off (0.7% now), and 28.6% on notes
 *   with a fundamental 15 dB below its partials (0.7% now).
 * - **Glide in semitone domain** - the public value slews toward the target
 *   at a constant rate expressed in milliseconds per octave, so the movement
 *   is musically uniform across the whole range (Hz-domain smoothing is
 *   faster at high pitches and sluggish at low ones).
 * - **Voicing** - isVoiced() tells whether the latest detection passed the
 *   gates; isTracking() holds for 250 ms after the last one that did.
 * - **Offline form** - track() reads a whole signal and decodes the pitch
 *   track over the complete sequence of detections, so it also uses what
 *   comes after each one (see its description for the measured gain).
 *
 * Measured against an HMM note tracker (the second stage of Mauch & Dixon
 * 2014, decoded causally by its forward filter) on the same material,
 * 48 kHz, legato melodies of 80 to 700 Hz with 25 ms glides,
 * readings within 50 cents of voiced frames (RPA), voicing false alarms
 * (VFA) and the time from a note change to a reading within 50 cents,
 * with setGlide(0), setConfidence(0.5) and isVoiced(): clean 93.7% / 1.9%
 * / 22 ms against 93.5% / 5.1% / 24 ms; 100 cents of vibrato 94.6% against
 * 94.1%; 5 dB SNR white noise 93.8% against 71.0%; with
 * setNoiseSuppression() and setConfidence(0.45), 0 dB 86.2% where the HMM
 * reads nothing. The default gate (0.85) holds the last clean pitch
 * through noisy passages, and the default glide trades latency for a
 * smooth control signal.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with processing).
 * - processBlock() / pushSamples() / reset(): audio thread (stream owner).
 * - setRange() / setConfidence() / setGlide() / setNoiseSuppression():
 *   control thread (ONE non-audio writer; independent single-word relaxed
 *   atomics -- a torn min/max pair from concurrent setRange() calls is
 *   prevented by the single-writer contract; non-finite values are
 *   ignored).
 * - getSmoothedHz() / getRawHz() / getConfidence() / isTracking() /
 *   isVoiced() and the parameter getters: any thread, lock-free.
 *
 * @code
 *   follower.prepare(spec);
 *   follower.setRange(70.0f, 800.0f);          // vocal range
 *   follower.setGlide(60.0f);                  // ms per octave
 *   // per block:
 *   follower.processBlock(buffer);             // internal mono sum
 *   eq.setFrequency(follower.getSmoothedHz() * 0.9f);  // low-cut under tonic
 * @endcode
 *
 * Dependencies: PitchDetector.h, AudioSpec.h, AudioBuffer.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "PitchDetector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class PitchFollower
 * @brief Gated, octave-safe, semitone-smoothed pitch tracking source.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PitchFollower
{
public:
    // -- Lifecycle -------------------------------------------------------------

    /**
     * @brief Creates a follower with the detector's noise suppression off.
     *
     * The follower gates readings at a confidence of 0.85 by default, which
     * a reading taken through noise never reaches (about 0.5 at 0 dB SNR),
     * so the detector's second reading would be paid for and discarded. Turn
     * it on together with a lower confidence (setNoiseSuppression(),
     * setConfidence()) to follow a note through noise.
     */
    PitchFollower() noexcept { detector_.setNoiseSuppression(false); }

    /**
     * @brief Prepares the follower and the wrapped detector.
     *
     * An invalid spec (non-finite or non-positive fields) is ignored,
     * keeping the previous state.
     *
     * @param spec       Audio environment specification.
     * @param windowSize Detector analysis window. Values <= 0 (the default)
     *                   select PitchDetector's AUTOMATIC window: 2048 at
     *                   44.1/48 kHz, 4096 at 88.2/96 kHz, 8192 at
     *                   176.4/192 kHz. Explicit positive values are forwarded
     *                   as given (with a hop of windowSize/4).
     *
     * The tracked register is the detector's: the lowest reachable fundamental
     * is sampleRate/(windowSize/2 - 1), so it depends on the RATIO, never on
     * the count. The automatic window is good down to ~47 Hz at every rate
     * from 44.1 to 192 kHz. A fixed 2048-sample window would instead stop at
     * ~94 Hz at 96 kHz and ~188 Hz at 192 kHz: measured, a 55 Hz or 82 Hz
     * sawtooth then never leaves getSmoothedHz() at 0 and isTracking() stays
     * false, whatever setRange() says -- the range gate cannot widen a
     * register the detector cannot reach. For a lower floor pass an explicit
     * window (doubling it halves the floor and doubles the analysis latency).
     */
    void prepare(const AudioSpec& spec, int windowSize = 0)
    {
        if (!spec.isValid()) return;
        sampleRate_ = spec.sampleRate;
        detector_.prepare(spec.sampleRate, windowSize,
                          (windowSize > 0) ? windowSize / 4 : 0);
        monoScratch_.assign(static_cast<size_t>(std::max(spec.maxBlockSize, 1)), T(0));
        hopPhase_ = 0;
        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /**
     * @brief Forgets the tracked pitch (parameters are kept).
     * Allocation-free; call from the stream owner (it touches the same
     * plain tracking state update() writes).
     */
    void reset() noexcept
    {
        hasTarget_ = false;
        targetSt_ = 0.0;
        currentSt_ = 0.0;
        pendingSt_ = 0.0;
        pendingCount_ = 0;
        samplesSinceValid_ = std::numeric_limits<int64_t>::max() / 2;
        smoothedHz_.store(T(0), std::memory_order_relaxed);
        tracking_.store(false, std::memory_order_relaxed);
        voiced_.store(false, std::memory_order_relaxed);
    }

    // -- Parameters (thread-safe) ------------------------------------------------

    /**
     * @brief Accepted pitch range in Hz (default 60 to 1200).
     * Non-finite values are ignored (the old max() passed NaN through and
     * closed the validity gate forever).
     */
    void setRange(T minHz, T maxHz) noexcept
    {
        if (!std::isfinite(minHz) || !std::isfinite(maxHz)) return;
        minHz = std::max(minHz, T(10));
        maxHz = std::max(maxHz, minHz);
        minHz_.store(minHz, std::memory_order_relaxed);
        maxHz_.store(maxHz, std::memory_order_relaxed);
    }

    /**
     * @brief Confidence threshold [0, 1] below which readings are ignored (default 0.85).
     *
     * The default accepts the clean readings only. The detector reads a tone
     * in noise at its honest, lower confidence (about 0.76 at 5 dB SNR):
     * a gate of 0.7 follows such a tone (measured: a 220 Hz tone at 5 dB SNR
     * tracked 100% of the time, within 15 cents), where 0.85 holds the last
     * clean pitch.
     */
    void setConfidence(T threshold) noexcept
    {
        if (!std::isfinite(threshold)) return;
        confidence_.store(std::clamp(threshold, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Glide time in milliseconds per octave (default 60).
     * 0 disables smoothing (the output snaps to each accepted reading).
     * Non-finite values are ignored.
     */
    void setGlide(T msPerOctave) noexcept
    {
        if (!std::isfinite(msPerOctave)) return;
        glideMs_.store(std::max(msPerOctave, T(0)), std::memory_order_relaxed);
    }

    /** @brief Returns the lower bound of the accepted pitch range in Hz. */
    [[nodiscard]] T getMinHz() const noexcept { return minHz_.load(std::memory_order_relaxed); }

    /** @brief Returns the upper bound of the accepted pitch range in Hz. */
    [[nodiscard]] T getMaxHz() const noexcept { return maxHz_.load(std::memory_order_relaxed); }

    /** @brief Returns the confidence gating threshold [0, 1]. */
    [[nodiscard]] T getConfidenceThreshold() const noexcept { return confidence_.load(std::memory_order_relaxed); }

    /** @brief Returns the glide time in milliseconds per octave. */
    [[nodiscard]] T getGlide() const noexcept { return glideMs_.load(std::memory_order_relaxed); }

    /** @return The wrapped detector's analysis window in samples (the
     *          automatic choice unless prepare() was given an explicit one).
     *          The lowest trackable fundamental is
     *          sampleRate/(getWindowSize()/2 - 1). */
    [[nodiscard]] int getWindowSize() const noexcept { return detector_.getWindowSize(); }

    /**
     * @brief Reads tones through broadband noise (opt-in, default off): the
     *        wrapped detector's PitchDetector::setNoiseSuppression().
     *
     * Those readings carry the honest, low confidence of a reading taken
     * through the noise (about 0.5 at 0 dB SNR), so the confidence gate
     * (setConfidence()) must be lowered for the follower to use them.
     * Control thread; relaxed atomic.
     */
    void setNoiseSuppression(bool on) noexcept { detector_.setNoiseSuppression(on); }

    /** @brief Whether the detector's noise suppression is on. */
    [[nodiscard]] bool getNoiseSuppression() const noexcept { return detector_.getNoiseSuppression(); }

    // -- Readout (lock-free, any thread) ------------------------------------------

    /** @return Smoothed pitch in Hz; 0 until the first reliable reading. */
    [[nodiscard]] T getSmoothedHz() const noexcept
    {
        return smoothedHz_.load(std::memory_order_relaxed);
    }

    /** @return Last raw detector reading in Hz (ungated - may be garbage). */
    [[nodiscard]] T getRawHz() const noexcept { return detector_.getFrequencyHz(); }

    /** @return Detector confidence of the last raw reading [0, 1]. */
    [[nodiscard]] T getConfidence() const noexcept { return detector_.getConfidence(); }

    /** @return True while reliable readings arrived within the last 250 ms. */
    [[nodiscard]] bool isTracking() const noexcept
    {
        return tracking_.load(std::memory_order_relaxed);
    }

    /**
     * @return True when the latest detection passed the confidence and range
     *         gates: a per-detection voicing flag, without isTracking()'s
     *         250 ms hold.
     */
    [[nodiscard]] bool isVoiced() const noexcept
    {
        return voiced_.load(std::memory_order_relaxed);
    }

    // -- Processing ----------------------------------------------------------------

    /**
     * @brief Feeds a block (any channel count); channels are averaged to mono.
     * @param buffer Read-only audio block.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        if (nCh <= 0 || nS <= 0) return;

        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, static_cast<int>(monoScratch_.size()));
            if (nCh == 1)
            {
                const T* src = buffer.getChannel(0) + i;
                std::copy(src, src + chunk, monoScratch_.begin());
            }
            else
            {
                const T invCh = T(1) / static_cast<T>(nCh);
                std::fill(monoScratch_.begin(), monoScratch_.begin() + chunk, T(0));
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* src = buffer.getChannel(ch) + i;
                    for (int k = 0; k < chunk; ++k)
                        monoScratch_[static_cast<size_t>(k)] += src[k] * invCh;
                }
            }
            pushSamples({ monoScratch_.data(), static_cast<size_t>(chunk) });
            i += chunk;
        }
    }

    /**
     * @brief Feeds mono samples directly (alternative to processBlock).
     * @param samples Mono audio span.
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed) || samples.empty()) return;
        // Stop at every detection, so each one is judged exactly once: called
        // per block instead, a small block judged one reading several times
        // and a large one skipped readings.
        const int hop = detector_.getHopSize();
        std::size_t done = 0;
        while (done < samples.size())
        {
            const auto chunk = std::min(samples.size() - done,
                                        static_cast<std::size_t>(hop - hopPhase_));
            detector_.pushSamples(samples.subspan(done, chunk));
            done += chunk;
            hopPhase_ += static_cast<int>(chunk);
            const bool freshReading = hopPhase_ >= hop;
            if (freshReading) hopPhase_ = 0;
            update(static_cast<int>(chunk), freshReading);
        }
    }

    // -- Offline tracking ------------------------------------------------------------

    /** @brief One detection of an offline pitch track (see track()). */
    struct TrackPoint
    {
        double seconds = 0.0;  ///< Centre of the detection's analysis window.
        T frequencyHz = T(0);  ///< Pitch; on an unvoiced point, that of the nearest voiced one (0 if none).
        T confidence = T(0);   ///< The detector's confidence in its own reading there (0: it read nothing).
        bool voiced = false;   ///< Whether the point belongs to a note.
    };

    /** @brief Settings of track(). */
    struct TrackSettings
    {
        T minHz = T(60);               ///< Lowest pitch reported.
        T maxHz = T(1200);             ///< Highest pitch reported.
        bool noiseSuppression = true;  ///< PitchDetector::setNoiseSuppression().
        int windowSize = 0;            ///< Analysis window, as prepare(); 0 = automatic.
    };

    /**
     * @brief Offline: reads a whole mono signal and decodes its pitch track
     *        over the complete sequence of detections.
     *
     * The follower judges every detection with the past alone, so a note
     * change costs it a detection of confirmation, an octave error can only
     * be folded towards what came before, and a detection the detector
     * declines stays empty. With the whole sequence known, every detection
     * offers its own reading, the octaves of it, the readings of the three
     * detections on each side and the line between them, and the track is the
     * cheapest path through those and "unvoiced" (a Viterbi decode): a pitch
     * change costs its size up to six semitones, another octave of a reading
     * or a neighbour's pitch a fixed amount, and entering or leaving a note
     * more than filling a gap of a few detections. The detection before a
     * note's first reading joins the note: the detector does not read a
     * window until the note fills it.
     *
     * Measured at 48 kHz on the material of the class description, against
     * the follower at its best settings there and against the HMM note
     * tracker decoded over the whole file (Viterbi). Frames within 50 cents
     * (RPA) / overall accuracy (voicing and pitch right) / voicing false
     * alarms: clean legato 95.2 / 95.3 / 2.4% (follower 93.7 / 94.2 / 1.9,
     * HMM 94.2 / 94.3 / 5.1); 100 cents of vibrato 95.7 / 95.7 / 4.0
     * (94.6 / 94.8 / 2.9, HMM 95.0 / 94.7 / 5.6); 5 dB SNR white noise
     * 96.7 / 96.3 / 0.0 (93.8 / 94.1 / 0.0, HMM 95.1 / 94.8 / 4.7); 5 dB
     * pink noise 96.3 / 95.9 / 0.0 (93.3 / 92.9 / 0.0, HMM 95.1 / 95.1 /
     * 4.3); 0 dB white noise 92.3 / 91.6 / 1.6 (86.2 / 87.3 / 0.0, HMM
     * none); tones with a fundamental 15 dB below their partials 100 / 99.5
     * / 0.6 (97.8 / 98.0 / 0.6, HMM 99.8 / 98.4 / 6.9). Octave errors: at
     * most 0.2% of voiced frames (follower up to 0.9%).
     *
     * Not real-time: it allocates (the result, one detector and about 200
     * bytes per detection for the decode) and runs on the calling thread.
     * Any thread; it shares no state with a follower instance.
     *
     * @param mono       Mono samples of the whole signal.
     * @param sampleRate Sample rate in Hz.
     * @param settings   Range, noise suppression and analysis window.
     * @return One point per detection whose window lies inside the signal
     *         (one every quarter of the window), in time order. Empty for an
     *         invalid rate or range, or a signal shorter than the window.
     */
    [[nodiscard]] static std::vector<TrackPoint> track(std::span<const T> mono, double sampleRate,
                                                       const TrackSettings& settings)
    {
        std::vector<TrackPoint> points;
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0) return points;
        const double minHz = static_cast<double>(settings.minHz), maxHz = static_cast<double>(settings.maxHz);
        if (!(std::isfinite(minHz) && std::isfinite(maxHz) && minHz > 0.0 && maxHz >= minHz)) return points;

        PitchDetector<T> detector;
        detector.prepare(sampleRate, settings.windowSize, (settings.windowSize > 0) ? settings.windowSize / 4 : 0);
        detector.setNoiseSuppression(settings.noiseSuppression);
        const std::size_t window = static_cast<std::size_t>(std::max(detector.getWindowSize(), 1));
        const std::size_t hop = static_cast<std::size_t>(std::max(detector.getHopSize(), 1));
        if (mono.size() < window) return points;

        // One reading per detection: the detector fires at the end of each hop.
        const std::size_t hops = mono.size() / hop;
        std::vector<TrackReading> readings;
        readings.reserve(hops);
        points.reserve(hops);
        for (std::size_t k = 0; k < hops; ++k)
        {
            detector.pushSamples(mono.subspan(k * hop, hop));
            const std::size_t end = (k + 1) * hop;
            if (end < window) continue;   // the window still reaches before the signal
            const double hz = static_cast<double>(detector.getFrequencyHz());
            TrackReading reading;
            TrackPoint point;
            point.seconds = (static_cast<double>(end) - 0.5 * static_cast<double>(window)) / sampleRate;
            if (hz >= minHz && hz <= maxHz)
            {
                reading.voiced = true;
                reading.semitones = 12.0 * std::log2(hz / 440.0);
                reading.confidence = static_cast<double>(detector.getConfidence());
                point.confidence = detector.getConfidence();
            }
            readings.push_back(reading);
            points.push_back(point);
        }
        decodeTrack(readings, 12.0 * std::log2(minHz / 440.0), 12.0 * std::log2(maxHz / 440.0), points);
        return points;
    }

    /** @brief track() with the default settings (60 to 1200 Hz, noise
     *         suppression on, automatic window).
     *  @param mono       Mono samples of the whole signal.
     *  @param sampleRate Sample rate in Hz.
     *  @return The decoded track (see the other overload). */
    [[nodiscard]] static std::vector<TrackPoint> track(std::span<const T> mono, double sampleRate)
    {
        return track(mono, sampleRate, TrackSettings{});
    }

private:
    // -- Offline decode (see track()) -------------------------------------------------

    /// One detection as the decode reads it.
    struct TrackReading
    {
        double semitones = 0.0;   ///< Pitch re 440 Hz.
        double confidence = 0.0;
        bool voiced = false;
    };

    static constexpr int kTrackReach = 3;                             ///< Detections a pitch is borrowed from, each side.
    static constexpr int kTrackCandidates = 4 + 2 * kTrackReach;      ///< Reading, two octaves, neighbours, the line between.
    static constexpr double kTrackOwn = 4.0;            ///< A reading, times one minus its confidence.
    static constexpr double kTrackOctave = 1.0;         ///< Another octave of the reading.
    static constexpr double kTrackBorrowVoiced = 3.0;   ///< A neighbour's pitch where the detector read another.
    static constexpr double kTrackBorrowSilent = 0.5;   ///< A neighbour's pitch where it read nothing.
    static constexpr double kTrackBorrowStep = 0.5;     ///< ... per detection of distance beyond the first.
    static constexpr double kTrackUnvoicedBase = 1.0;   ///< Calling a detection with a reading unvoiced,
    static constexpr double kTrackUnvoicedConf = 4.0;   ///< ... plus this times its confidence.
    static constexpr double kTrackEdge = 0.5;           ///< Leaving the detection before a first reading unvoiced.
    static constexpr double kTrackSwitch = 4.0;         ///< Entering or leaving a note.
    static constexpr double kTrackPerSemitone = 1.0;    ///< A pitch change, per semitone,
    static constexpr double kTrackJumpCap = 6.0;        ///< ... up to this many.
    static constexpr double kTrackSame = 0.25;          ///< Candidates this close (semitones) are one.
    static constexpr double kTrackLine = 2.0;           ///< Neighbours this close are joined by a line.

    /// The candidates of one detection and where each one's best path came from.
    struct TrackFrame
    {
        double semitones[kTrackCandidates] = {};
        double cost[kTrackCandidates] = {};
        std::uint8_t back[kTrackCandidates + 1] = {};   ///< State 0 is unvoiced, i + 1 candidate i.
        std::uint8_t count = 0;
        double unvoicedCost = 0.0;

        void add(double st, double c) noexcept
        {
            for (int i = 0; i < count; ++i)
                if (std::abs(semitones[i] - st) < kTrackSame)
                {
                    cost[i] = std::min(cost[i], c);
                    return;
                }
            if (count < kTrackCandidates)
            {
                semitones[count] = st;
                cost[count] = c;
                ++count;
            }
        }
    };

    /** Decodes the readings over the whole sequence and writes the pitch and
     *  the voicing of every point (see track()). */
    static void decodeTrack(const std::vector<TrackReading>& readings, double minSemitones, double maxSemitones,
                            std::vector<TrackPoint>& points)
    {
        const std::ptrdiff_t total = static_cast<std::ptrdiff_t>(readings.size());
        if (total == 0) return;
        const auto at = [&](std::ptrdiff_t k) -> const TrackReading& { return readings[static_cast<std::size_t>(k)]; };

        std::vector<TrackFrame> frames(readings.size());
        for (std::ptrdiff_t k = 0; k < total; ++k)
        {
            TrackFrame& frame = frames[static_cast<std::size_t>(k)];
            const TrackReading& own = at(k);
            if (own.voiced)
            {
                const double cost = kTrackOwn * (1.0 - own.confidence);
                frame.add(own.semitones, cost);
                if (own.semitones - 12.0 >= minSemitones) frame.add(own.semitones - 12.0, cost + kTrackOctave);
                if (own.semitones + 12.0 <= maxSemitones) frame.add(own.semitones + 12.0, cost + kTrackOctave);
                frame.unvoicedCost = kTrackUnvoicedBase + kTrackUnvoicedConf * own.confidence;
            }
            else if (k + 1 < total && at(k + 1).voiced)
            {
                frame.unvoicedCost = kTrackEdge;
            }
            const double borrow = own.voiced ? kTrackBorrowVoiced : kTrackBorrowSilent;
            std::ptrdiff_t left = -1, right = -1;
            for (std::ptrdiff_t d = 1; d <= kTrackReach; ++d)
            {
                const double distant = borrow + kTrackBorrowStep * static_cast<double>(d - 1);
                if (k - d >= 0 && at(k - d).voiced)
                {
                    if (left < 0) left = k - d;
                    frame.add(at(k - d).semitones, distant);
                }
                if (k + d < total && at(k + d).voiced)
                {
                    if (right < 0) right = k + d;
                    // The detection before a first reading takes it at no cost.
                    frame.add(at(k + d).semitones, (!own.voiced && d == 1) ? 0.0 : distant);
                }
            }
            if (left >= 0 && right >= 0 && std::abs(at(left).semitones - at(right).semitones) < kTrackLine)
            {
                const double w = static_cast<double>(k - left) / static_cast<double>(right - left);
                frame.add(at(left).semitones + (at(right).semitones - at(left).semitones) * w,
                          borrow + kTrackBorrowStep * static_cast<double>(std::min(k - left, right - k) - 1));
            }
        }

        // Viterbi over "unvoiced" and each detection's candidates.
        double best[kTrackCandidates + 1] = {}, next[kTrackCandidates + 1] = {};
        best[0] = frames[0].unvoicedCost;
        for (int i = 0; i < frames[0].count; ++i) best[i + 1] = frames[0].cost[i];
        for (std::size_t k = 1; k < frames.size(); ++k)
        {
            const TrackFrame& before = frames[k - 1];
            TrackFrame& frame = frames[k];
            {
                double low = best[0];
                int from = 0;
                for (int i = 0; i < before.count; ++i)
                {
                    const double v = best[i + 1] + kTrackSwitch;
                    if (v < low) { low = v; from = i + 1; }
                }
                next[0] = low + frame.unvoicedCost;
                frame.back[0] = static_cast<std::uint8_t>(from);
            }
            for (int j = 0; j < frame.count; ++j)
            {
                double low = best[0] + kTrackSwitch;
                int from = 0;
                for (int i = 0; i < before.count; ++i)
                {
                    const double v = best[i + 1]
                        + kTrackPerSemitone * std::min(std::abs(frame.semitones[j] - before.semitones[i]), kTrackJumpCap);
                    if (v < low) { low = v; from = i + 1; }
                }
                next[j + 1] = low + frame.cost[j];
                frame.back[j + 1] = static_cast<std::uint8_t>(from);
            }
            std::copy(next, next + frame.count + 1, best);
        }

        // Walk the cheapest path back; unvoiced points keep the pitch of the
        // nearest voiced one (the earlier one when two are as near).
        int state = 0;
        for (int i = 1; i <= frames.back().count; ++i)
            if (best[i] < best[state]) state = i;
        for (std::size_t k = frames.size(); k-- > 0;)
        {
            TrackPoint& point = points[k];
            point.voiced = state > 0;
            point.frequencyHz = state > 0
                ? static_cast<T>(440.0 * std::pow(2.0, frames[k].semitones[state - 1] / 12.0)) : T(0);
            state = frames[k].back[state];
        }
        constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
        std::size_t voicedAt = kNone;
        for (std::size_t k = 0; k < points.size(); ++k)
        {
            if (points[k].voiced) voicedAt = k;
            else if (voicedAt != kNone) points[k].frequencyHz = points[voicedAt].frequencyHz;
        }
        voicedAt = kNone;
        std::size_t earlier = kNone;   // the voiced point before the current unvoiced run
        for (std::size_t k = points.size(); k-- > 0;)
        {
            if (points[k].voiced) { voicedAt = k; earlier = kNone; continue; }
            if (voicedAt == kNone) continue;
            if (earlier == kNone)
            {
                std::size_t j = k;
                while (j > 0 && !points[j - 1].voiced) --j;
                earlier = (j > 0) ? j - 1 : kNone - 1;   // kNone - 1: no voiced point before the run
            }
            const bool none = earlier == kNone - 1;
            if (none || voicedAt - k < k - earlier) points[k].frequencyHz = points[voicedAt].frequencyHz;
        }
    }

    /** @brief Gating, octave correction and jump confirmation on a fresh
     *  detection; the glide and the tracking state on every call. */
    void update(int numSamples, bool freshReading) noexcept
    {
        const T raw  = detector_.getFrequencyHz();
        const T conf = detector_.getConfidence();
        const bool valid = conf >= confidence_.load(std::memory_order_relaxed)
                        && raw >= minHz_.load(std::memory_order_relaxed)
                        && raw <= maxHz_.load(std::memory_order_relaxed);

        if (freshReading) voiced_.store(valid, std::memory_order_relaxed);
        if (!freshReading)
        {
            samplesSinceValid_ += numSamples;
        }
        else if (valid)
        {
            double st = 12.0 * std::log2(static_cast<double>(raw) / 440.0);

            if (!hasTarget_)
            {
                targetSt_ = currentSt_ = st;   // first lock: no sweep-in
                hasTarget_ = true;
            }
            else if (std::abs(st - targetSt_) <= 7.0)
            {
                targetSt_ = st;
                pendingCount_ = 0;
            }
            else
            {
                // A large interval, an octave leap among them, is taken once
                // kIntervalReadings consecutive readings agree on it; one odd
                // reading never moves the target. Meanwhile a reading whole
                // octaves away from the target (the classic 2x / 0.5x
                // detector error) still updates it, folded back to the
                // target's octave.
                if (pendingCount_ > 0 && std::abs(st - pendingSt_) < 1.0)
                {
                    ++pendingCount_;
                }
                else
                {
                    pendingSt_ = st;
                    pendingCount_ = 1;
                }
                if (pendingCount_ >= kIntervalReadings)
                {
                    targetSt_ = st;
                    pendingCount_ = 0;
                }
                else
                {
                    for (const double oct : { -24.0, -12.0, 12.0, 24.0 })
                    {
                        if (std::abs(st + oct - targetSt_) < 1.5)
                        {
                            targetSt_ = st + oct;
                            break;
                        }
                    }
                }
            }
            samplesSinceValid_ = 0;
        }
        else
        {
            samplesSinceValid_ += numSamples;   // freeze: target stays put
        }

        if (hasTarget_)
        {
            const double glideMs = static_cast<double>(glideMs_.load(std::memory_order_relaxed));
            if (glideMs < 0.5)
            {
                currentSt_ = targetSt_;
            }
            else
            {
                // Constant musical speed: 12 semitones in glideMs milliseconds.
                const double maxStep = 12.0 * static_cast<double>(numSamples)
                                     / (glideMs * 0.001 * sampleRate_);
                currentSt_ += std::clamp(targetSt_ - currentSt_, -maxStep, maxStep);
            }
            smoothedHz_.store(static_cast<T>(440.0 * std::exp2(currentSt_ / 12.0)),
                              std::memory_order_relaxed);
        }

        tracking_.store(hasTarget_
                        && samplesSinceValid_ < static_cast<int64_t>(0.25 * sampleRate_),
                        std::memory_order_relaxed);
    }

    // -- Members --------------------------------------------------------------------
    /// Consecutive readings that confirm a large interval or an octave leap.
    static constexpr int kIntervalReadings = 2;
    double sampleRate_ = 48000.0;
    std::atomic<bool> prepared_{ false };

    PitchDetector<T> detector_;
    std::vector<T> monoScratch_;

    bool hasTarget_ = false;
    double targetSt_ = 0.0;     ///< Gated target, semitones relative to A440.
    double currentSt_ = 0.0;    ///< Glided value, semitones relative to A440.
    double pendingSt_ = 0.0;
    int pendingCount_ = 0;
    int hopPhase_ = 0;          ///< Samples since the detector's last detection.
    int64_t samplesSinceValid_ = 0;

    std::atomic<T> minHz_ { T(60) };
    std::atomic<T> maxHz_ { T(1200) };
    std::atomic<T> confidence_ { T(0.85) };
    std::atomic<T> glideMs_ { T(60) };

    std::atomic<T> smoothedHz_ { T(0) };
    std::atomic<bool> tracking_ { false };
    std::atomic<bool> voiced_ { false };
};

} // namespace dspark
