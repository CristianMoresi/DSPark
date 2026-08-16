// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PitchCorrector.h
 * @brief Real-time monophonic pitch correction (scale-aware retune).
 *
 * PitchCorrector composes three existing processors into the classic
 * detect-quantize-shift chain:
 *
 *   input -> PitchDetector (YIN, channel 0) -> nearest in-scale note
 *         -> retune smoother -> PitchShifter (phase vocoder) -> output
 *
 * On every control interval the detected fundamental is converted to a
 * fractional MIDI note, quantized to the nearest note of the configured scale,
 * and the difference in semitones becomes the correction the internal shifter
 * applies. A one-pole glide whose time constant is the retune speed carries
 * the correction toward that value: 0 ms snaps hard to the target, larger
 * values let the output start at the sung pitch and drift to the note - the
 * familiar range from robotic quantization to transparent intonation help.
 *
 * Scale semantics: setScale() takes a 12-bit harmony::NoteSet whose bit k means
 * "k semitones above the root belongs to the scale" (the layout of the
 * harmony::allScales masks, e.g. harmony::allScales[0].mask for the major
 * scale) plus a root pitch class (0 = C .. 11 = B). The mask is rotated to the
 * root with harmony::scaleAtRoot(). The default is the chromatic scale at root
 * C: snap to the nearest equal-tempered semitone. An empty mask (after masking
 * to the low 12 bits) disables correction - there is nothing to snap to - and
 * the signal passes through the vocoder unshifted.
 *
 * Quantization is memoryless and deterministic: the nearest in-scale note wins,
 * and an exact tie between two candidates resolves to the LOWER note. There is
 * no hysteresis, so an input sustained exactly on the boundary between two
 * targets can alternate between them at the control rate. Landing there takes
 * a pitch held to within the detector's own error, a couple of cents at worst,
 * of the exact midpoint between two scale notes; it is a corner case, and it is
 * documented rather than hidden.
 *
 * Unvoiced input holds the correction instead of releasing it. A momentary
 * unvoiced report - a consonant, a soft passage, a vibrato extreme leaving the
 * detector's confidence gate - would otherwise pull the correction toward zero
 * and back within a few tens of milliseconds, which is an audible pitch dip in
 * the middle of a held note. Holding costs nothing on material that has no
 * pitch of its own, and the correction is replaced as soon as a fundamental is
 * reported again. The correction returns to zero only on reset(), or when the
 * scale is emptied.
 *
 * Two gates decide what may ESTABLISH a held correction, because a policy that
 * holds a value must be careful about which value it holds:
 *
 * - Warm-up. Reports are ignored until one full analysis window has been fed
 *   since prepare() or reset(). The detector's first window is three quarters
 *   zeros, and on any non-silent input it yields the shortest lag its search
 *   allows - half the sample rate, at full confidence. Held, that would
 *   transpose a track that never contains a note at all, which is the opposite
 *   of what the hold is for. One window is 43 ms at 44.1/48 kHz, well inside
 *   the signal latency, so nothing audible waits for it.
 * - Register. A reported fundamental above a quarter of the sample rate is
 *   refused: not even its second harmonic fits below Nyquist, so it is not a
 *   pitch this effect could act on, and every musical fundamental is far below
 *   that at every supported rate.
 *
 * Timing: the correction updates every 64 samples on an absolute grid, the
 * detector reports once per 512 samples at the 44.1/48 kHz analysis window, and
 * the shifter's engine adopts a new target once per analysis hop (about 10.7 ms
 * at 48 kHz with the 2048-sample frame used here) and slews it at up to
 * 0.5 semitones per hop so target changes stay click-free. That slew, not the
 * control path, sets how fast a hard snap (retune speed 0) can land: one hop
 * per half semitone of CHANGE, plus up to one hop of adoption delay.
 *
 * Both sizes that matter follow from ONE property of the mask you configure -
 * its widest gap, the largest number of semitones between two adjacent notes
 * of the scale:
 *
 *   largest correction magnitude  =  half the widest gap
 *   largest change between two consecutive corrections  =  the widest gap
 *
 * The magnitude is half because a pitch is never further than half a gap from
 * the nearer of the two notes bounding it; the change is the whole gap because
 * a voice crossing the middle of that gap swings from one bound to the other
 * in one decision, as does a key change under a held note. So the chromatic
 * default gives 0.5 and 1 semitone, a diatonic scale 1 and 2, harmonic minor
 * and the pentatonics 1.5 and 3, scales like Hirajoshi 2 and 4, and a one-note
 * mask - which setScale() accepts - 6 and 12.
 *
 * Settling follows the change, measured from the control change to the output
 * staying within 10 cents of the new note, worst case over a full
 * analysis-hop period of control phases, both directions, at 48 / 44.1 kHz:
 *
 *   change 0.5 semitones   12.3 / 14.3 ms
 *   change 1               23.3 / 25.6 ms
 *   change 2               44.3 / 48.6 ms
 *   change 3               66.0 / 72.3 ms
 *   change 4               88.8 / 96.2 ms
 *
 * Up to one semitone that fits inside the pipeline's own latency, so the
 * transition is complete before the audio it governs is heard. Above it the
 * traversal is audible by its excess: these are traversals and not errors -
 * the pitch moves continuously, without a click, and lands within 10 cents.
 * The table is the worst of 32 control phases per row, both directions, read
 * by a heterodyne fundamental tracker averaging over one period of the target.
 * Against the engine's slew law (half a semitone per analysis hop, the hop
 * dilating with the active ratio) the rows above one semitone agree to within
 * 4%; the two smallest rows read up to 20% above it, which is the estimator
 * window rather than the class - at one slew hop the tracker's own averaging
 * is a fifth of the quantity being measured. Either way the correction layer
 * adds nothing to the rate limit it inherits. A nonzero retune speed reaches
 * within 10 cents of a 100-cent target after about 2.5 time constants on top
 * of that floor; below roughly 8 ms of retune speed the floor is all there is,
 * and asking for a faster glide changes nothing.
 *
 * Accuracy: the correction is computed from the detected fundamental, so the
 * corrected pitch inherits the detector's own error and little else. Measured
 * from E2 to C6 at 44.1 and 48 kHz against an autocorrelation probe that shares
 * no code with the detector, the steady-state output sits within 2 cents of the
 * target note; the worst case is at the top of that range, where a period is
 * short enough that locating it costs the most in relative terms.
 *
 * Sample rate and the analysis frame: the frame is chosen from the rate so its
 * analysis SPAN stays about 43 ms - the smallest power of two spanning at least
 * what 2048 samples cover at 48 kHz, so 2048 at 44.1/48 kHz, 4096 at
 * 88.2/96 kHz, 8192 at 176.4/192 kHz. This is the policy PitchDetector already
 * applies to its own window, and it is here for the same reason: everything the
 * frame decides is a property of time, not of samples. A frame long enough to
 * resolve the harmonics of a bass voice is long enough in MILLISECONDS; the
 * settling times above are milliseconds; so a frame pinned in samples would
 * halve its own span every time the rate doubled and take the low register with
 * it. At a 21 ms span - which is what 2048 samples give at 96 kHz - the
 * harmonic-to-residual ratio of a corrected F2 falls to 3.5-7.3 dB, from
 * 42-48 dB at 43 ms. Holding the span keeps every millisecond figure in this
 * file true at every supported rate, at the cost of latency growing with the
 * rate in samples while staying constant in time. Pass an explicit frame to
 * prepare() to override the policy; what that trade costs is measured below.
 *
 * At the automatic frame, both ends of that trade were measured before the span
 * was fixed. Every decibel below is read by one stated instrument - a four-term
 * Blackman-Harris DFT over a single un-padded 0.68 s segment, summing the
 * window's main lobe around each harmonic of the MEASURED fundamental against
 * everything else up to 20 kHz - which reads about 89 dB on a signal
 * constructed to have no residual at all, and returns planted ratios of 15 to
 * 40 dB within 0.4 dB. That headroom is what makes these figures reproducible:
 * a probe reports 1/M = 1/R + 1/F for a true ratio R and its own leakage floor
 * F, so one whose floor sat near 20 dB would report its own aperture here and
 * not the effect. Halving the span - a 21 ms span, whether by asking for 1024
 * at 48 kHz or by leaving 2048 pinned at 96 kHz - halves every settling time
 * and drops the harmonic-to-residual ratio of a corrected vowel from 42-48 dB
 * to 3.5-7.3 dB at F2 (87.31 Hz) and from 38-41 dB to 4.0-4.4 dB at A2
 * (110 Hz), while A3 (220 Hz) moves only from 47-51 dB to 40-42 dB: 34 to
 * 45 dB of harmonic structure at the low notes against 5 to 11 dB an octave
 * up, paid by exactly the voices this effect is pointed at. The low notes read
 * below A3 even at the full span, by 3 to 13 dB, and that is the resolution of
 * a 43 ms window rather than damage: at a fixed span 220 Hz gets about two and
 * a half times as many bins per harmonic spacing as 87 Hz. Doubling the span
 * to 85 ms keeps the low register but a one-semitone change no longer settles
 * inside the pipeline's latency (42 ms at 48 kHz). The 43 ms span is the only
 * one in that family which satisfies both ends.
 *
 * Latency: exactly the internal shifter's twice the frame - 4096 samples at
 * 44.1/48 kHz (85.3 ms at 48 kHz, 92.9 ms at 44.1 kHz), and about the same
 * time at every other rate - reported by getLatency(). The correction
 * DECISION additionally trails the input by the detector's analysis span (about
 * 43 ms plus one hop). Because that is shorter than the signal latency, a
 * freshly attacked note leaves the corrector already corrected; the decision
 * delay shows up instead as the response lag to a pitch change inside a note,
 * and as roughly 30 ms of the previous note's tail carrying the new note's
 * correction.
 *
 * Determinism: the control grid is absolute and advances only in whole
 * intervals, so the correction trajectory is a function of the absolute sample
 * position alone. Together with the shifter's own block-size independence, the
 * rendered output is bit-for-bit identical however the host chops the stream
 * into blocks.
 *
 * Monophonic assumption: the detector tracks ONE fundamental, taken from
 * channel 0. Chords, dense reverb or heavy unison make it report whichever
 * periodicity wins (or nothing at all), and the correction follows that
 * estimate; this effect is for solo voices and monophonic instruments. All
 * processed channels receive the same correction through the stereo-linked
 * shifter, which preserves inter-channel phase relations exactly.
 *
 * Artifact bounds at extreme settings: the correction is produced by a phase
 * vocoder, so the shifter's own character applies. Large corrections - remote
 * scales, or an octave error on noisy input - carry the usual resampling timbre
 * shift unless formant preservation is on, and a hard snap onto a heavily
 * detuned source is the deliberately audible robotic effect. The correction is
 * clamped to the shifter's +/-12 semitone range; a one-note scale can
 * legitimately ask for up to 6 semitones.
 *
 * One bound is worth stating in numbers, because it is the case a singer
 * reaches by accident: a vibrato that straddles the midpoint between two scale
 * notes makes the quantizer alternate, and the hard snap then EXAGGERATES the
 * wobble instead of removing it. On a 6 Hz vibrato at 48 kHz, an input swinging
 * 100 cents across that midpoint comes out swinging up to 260 cents, and one
 * staying inside a cell comes out at up to 130 cents peak to peak from a
 * 70-cent input, because the decision itself lags by about 55 ms. Both figures
 * are the worst over two source spectra, read by a heterodyne tracker whose
 * averaging window is one period of the target note; a wider window smooths the
 * alternation and reports less, so these are upper readings rather than
 * typical ones. A retune speed of a few tens of milliseconds is what removes
 * vibrato; a hard snap chases it.
 *
 * Formant preservation: setFormantPreserve(true) enables the shifter's cepstral
 * envelope pre-warp, keeping the vocal tract signature in place while the pitch
 * moves (the anti-chipmunk correction). It costs two extra FFTs per analysis
 * frame and is off by default.
 *
 * Threading: setScale(), setRetuneSpeedMs(), setFormantPreserve() and their
 * getters are lock-free single-word atomics and may be called from any thread,
 * with one non-audio writer as everywhere in DSPark; the scale mask and its
 * root travel packed in one word, so a torn mask/root pair cannot exist.
 * prepare() belongs to the setup thread (it allocates). processBlock(),
 * reset() and getLatency() belong to the stream owner. The inner detector and
 * shifter are private members whose control entry points are driven only by
 * the stream owner inside processBlock() and reset(); their own control-to-
 * audio hand-offs therefore collapse to same-thread sequential code, with this
 * class's relaxed words as the only cross-thread channel.
 *
 * Dependencies: Analysis/PitchDetector.h, Effects/PitchShifter.h,
 * Music/HarmonyConstants.h, Core/AudioBuffer.h, Core/AudioSpec.h,
 * Core/DenormalGuard.h, Core/DspMath.h.
 */

#include "../Analysis/PitchDetector.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Music/HarmonyConstants.h"
#include "PitchShifter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <span>

namespace dspark {

/**
 * @class PitchCorrector
 * @brief Scale-aware monophonic retune over the framework's YIN detector and
 *        phase-vocoder shifter.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PitchCorrector final
{
public:
    // The published words are read on the audio thread every control interval,
    // and a word that is not lock-free would take a mutex inside the callback.
    static_assert(std::atomic<T>::is_always_lock_free,
                  "audio-thread stores must not lock");
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "audio-thread stores must not lock");

    // -- Lifecycle -------------------------------------------------------------

    /**
     * @brief Allocates the detector and shifter state (setup thread).
     *
     * Invalid specifications (per AudioSpec::isValid()) are ignored and the
     * previous state is kept; an unprepared instance passes audio through
     * untouched. The detector uses its own automatic window policy, which holds
     * its analysis span constant across rates; the shifter's frame follows the
     * policy documented above.
     *
     * @param spec    Audio environment specification.
     * @param fftSize Analysis frame for the internal shifter. Values <= 0 (the
     *                default) select the AUTOMATIC frame: the smallest power of
     *                two in [512, 32768] spanning at least 2048/48000 s
     *                (42.7 ms) at this rate - 2048 at 44.1/48 kHz, 4096 at
     *                88.2/96 kHz, 8192 at 176.4/192 kHz. Read it back with
     *                getFrameSize(). An explicit request is rounded up to a
     *                power of two and clamped to [256, 1 << 20]; going below
     *                the automatic value buys settling time and costs the low
     *                register, in the proportions the frame note above states.
     */
    void prepare(const AudioSpec& spec, int fftSize = 0)
    {
        if (!spec.isValid()) return;

        prepared_ = false;

        const int frame = fftSize > 0 ? sanitizeFrame(fftSize)
                                      : automaticFrame(spec.sampleRate);

        detector_.prepare(spec.sampleRate);
        shifter_.prepare(spec, frame);

        sampleRate_ = spec.sampleRate;
        frameSize_ = frame;
        latency_ = shifter_.getLatency();

        prepared_ = true;
        reset();
    }

    /**
     * @brief Clears all signal state and keeps the parameters (stream owner).
     *
     * The applied correction restarts at zero: a fresh stream has no note
     * history worth continuing to correct.
     */
    void reset() noexcept
    {
        if (!prepared_) return;

        detector_.reset();
        target_ = 0.0;
        correction_ = 0.0;
        publishedCorrection_ = 0.0;
        shifter_.setSemitones(T(0));
        shifter_.reset();
        controlPhase_ = 0;
        smoothingForMs_ = -1.0;   // rebuild the glide coefficient on next block
        warmupRemaining_ = detector_.getWindowSize();
    }

    // -- Parameters (any thread) -------------------------------------------------

    /**
     * @brief Selects the scale the output snaps to.
     *
     * @param scaleBitmask 12-bit harmony::NoteSet: bit k is the scale degree k
     *                     semitones above the root (the harmony::allScales
     *                     layout). Bits above the low 12 are ignored, and an
     *                     empty mask disables correction.
     * @param rootPitchClass Root as semitones above C; any integer folds into
     *                       0..11.
     */
    void setScale(std::uint16_t scaleBitmask, int rootPitchClass) noexcept
    {
        const auto mask = static_cast<std::uint32_t>(scaleBitmask & 0x0FFFu);
        const auto root = static_cast<std::uint32_t>((rootPitchClass % 12 + 12) % 12);
        const auto absolute = static_cast<std::uint32_t>(harmony::scaleAtRoot(
            static_cast<harmony::NoteSet>(mask), static_cast<int>(root)));
        // One packed word: the audio thread must never pair a fresh mask with a
        // stale root, so the mask, the root and the pre-rotated absolute
        // pitch-class set travel together.
        scaleWord_.store(absolute | (root << 12) | (mask << 16),
                         std::memory_order_relaxed);
    }

    /**
     * @brief Sets the retune speed: the time constant, in milliseconds, of the
     *        glide from the sung pitch to the target note.
     *
     * 0 selects the hard snap (the full correction at once, quantized
     * character). Values are clamped to [0, 10000]; non-finite values are
     * ignored.
     */
    void setRetuneSpeedMs(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        retuneSpeedMs_.store(std::clamp(ms, T(0), T(kMaxRetuneMs)),
                             std::memory_order_relaxed);
    }

    /** @brief Enables the shifter's cepstral formant preservation (the
     *  anti-chipmunk envelope pre-warp). Default off. */
    void setFormantPreserve(bool on) noexcept
    {
        formantPreserve_.store(on, std::memory_order_relaxed);
    }

    /** @return The scale mask as passed to setScale() (root-relative). */
    [[nodiscard]] std::uint16_t getScaleMask() const noexcept
    {
        return static_cast<std::uint16_t>(
            (scaleWord_.load(std::memory_order_relaxed) >> 16) & 0x0FFFu);
    }

    /** @return The root pitch class in effect (0 = C .. 11 = B). */
    [[nodiscard]] int getRootPitchClass() const noexcept
    {
        return static_cast<int>(
            (scaleWord_.load(std::memory_order_relaxed) >> 12) & 0x0Fu);
    }

    /** @return The retune speed in milliseconds (0 = hard snap). */
    [[nodiscard]] T getRetuneSpeedMs() const noexcept
    {
        return retuneSpeedMs_.load(std::memory_order_relaxed);
    }

    /** @return Whether formant preservation is requested. */
    [[nodiscard]] bool getFormantPreserve() const noexcept
    {
        return formantPreserve_.load(std::memory_order_relaxed);
    }

    /** @brief Reports the signal latency in samples: twice the analysis frame,
     *  which is 4096 samples at 44.1/48 kHz and about 85 ms at every supported
     *  rate, since the frame follows the rate. Zero before prepare(). */
    [[nodiscard]] int getLatency() const noexcept
    {
        return prepared_ ? latency_ : 0;
    }

    /** @brief The analysis frame in effect, in samples: the automatic choice
     *  for this rate, or the rounded explicit request. Zero before prepare(). */
    [[nodiscard]] int getFrameSize() const noexcept
    {
        return prepared_ ? frameSize_ : 0;
    }

    // -- Processing --------------------------------------------------------------

    /**
     * @brief Processes audio in-place. Pass-through until prepare() succeeds.
     *
     * Channels beyond the prepared count are left untouched, as in the shifter.
     * Detection reads channel 0; every processed channel receives the same
     * correction.
     *
     * @param buffer Audio block.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        const int numSamples = buffer.getNumSamples();
        if (numSamples <= 0) return;

        DenormalGuard guard;

        // One load per block for each independent word. The scale word is
        // internally consistent by packing; the retune speed and the formant
        // flag stand alone.
        const std::uint32_t absoluteMask =
            scaleWord_.load(std::memory_order_relaxed) & 0x0FFFu;

        const double retuneMs = static_cast<double>(
            retuneSpeedMs_.load(std::memory_order_relaxed));
        if (retuneMs != smoothingForMs_) rebuildSmoothing(retuneMs);

        const bool formant = formantPreserve_.load(std::memory_order_relaxed);
        if (formant != publishedFormant_)
        {
            // Forwarded by the stream owner, which is the shifter's only
            // control writer here (see the file documentation).
            shifter_.setFormantPreserve(formant);
            publishedFormant_ = formant;
        }

        const T* const detectionInput =
            buffer.getNumChannels() > 0 ? buffer.getChannel(0) : nullptr;

        int offset = 0;
        while (offset < numSamples)
        {
            // The grid is absolute (controlPhase_ persists across blocks): the
            // correction advances once, at the START of each whole interval,
            // and every sample of that interval is then rendered with it in
            // force. The shifter's engine adopts a new value at its own
            // analysis hops, which do not land on this grid once the ratio
            // leaves unity, so publishing at the start of the interval that
            // contains a hop - rather than at whichever chunk boundary the
            // host block size happens to create - is what makes the rendered
            // output identical however the stream is chopped into blocks.
            if (controlPhase_ == 0) advanceCorrection(absoluteMask);

            const int chunk =
                std::min(numSamples - offset, kControlInterval - controlPhase_);

            if (detectionInput != nullptr)
                detector_.pushSamples(std::span<const T>(
                    detectionInput + offset, static_cast<std::size_t>(chunk)));

            if (warmupRemaining_ > 0)
                warmupRemaining_ = std::max(0, warmupRemaining_ - chunk);

            shifter_.processBlock(buffer.getSubView(offset, chunk));

            controlPhase_ += chunk;
            if (controlPhase_ >= kControlInterval) controlPhase_ = 0;
            offset += chunk;
        }
    }

private:
    /// The automatic frame holds an analysis SPAN, not a sample count: the
    /// smallest power of two spanning at least the time 2048 samples cover at
    /// 48 kHz. Both ends of the trade this frame settles are properties of
    /// TIME, so a fixed sample count only satisfies them at one rate.
    static constexpr double kAutoSpanRef = 2048.0;
    static constexpr double kAutoSpanRate = 48000.0;
    static constexpr int kAutoMinFrame = 512;
    static constexpr int kAutoMaxFrame = 32768;

    /** @brief The automatic frame for a rate: constant analysis span. */
    [[nodiscard]] static int automaticFrame(double sampleRate) noexcept
    {
        const double target = sampleRate * (kAutoSpanRef / kAutoSpanRate);
        int frame = kAutoMinFrame;
        while (frame < kAutoMaxFrame && static_cast<double>(frame) < target)
            frame <<= 1;
        return frame;
    }

    /** @brief Rounds an explicit request up to a power of two in the range the
     *  shifter accepts. The shifter IGNORES a malformed frame and keeps its
     *  previous state, which here would mean a silently unprepared stream, so
     *  the value is made valid before it is handed over rather than after. */
    [[nodiscard]] static int sanitizeFrame(int requested) noexcept
    {
        constexpr int kMax = 1 << 20;
        int frame = 256;
        while (frame < requested && frame < kMax) frame <<= 1;
        return frame;
    }

    /// Correction update interval in samples (1.3 ms at 48 kHz): fine enough
    /// that the shifter's roughly 512-sample hops always adopt a fresh value,
    /// cheap enough to disappear next to the vocoder.
    static constexpr int kControlInterval = 64;

    /// Landing radius of the glide, in semitones (0.1 cent): inside it the
    /// smoother lands exactly, so a settled correction stops republishing and
    /// the trajectory cannot crawl through denormals.
    static constexpr double kLandingSemitones = 0.001;

    /// Longest accepted glide time constant, in milliseconds.
    static constexpr double kMaxRetuneMs = 10000.0;

    /** @brief Rebuilds the one-pole glide coefficient for a new retune speed.
     *
     *  Called from processBlock() only when the published speed changed, so the
     *  transcendental is paid once per parameter change rather than per block.
     */
    void rebuildSmoothing(double retuneMs) noexcept
    {
        smoothingForMs_ = retuneMs;
        if (retuneMs <= 0.0 || sampleRate_ <= 0.0)
        {
            smoothing_ = 1.0;   // hard snap
            return;
        }
        const double tauSamples = retuneMs * 0.001 * sampleRate_;
        smoothing_ = 1.0 - std::exp(-static_cast<double>(kControlInterval) / tauSamples);
    }

    /**
     * @brief Advances the correction by one whole control interval and hands
     *        the shifter the new value (audio thread).
     */
    void advanceCorrection(std::uint32_t absoluteMask) noexcept
    {
        if (absoluteMask == 0u)
        {
            target_ = 0.0;   // no scale to snap to: correction disabled
        }
        else if (warmupRemaining_ == 0)
        {
            // Two gates before a report may establish a held correction: the
            // analysis window must be full (its first, mostly-zero window
            // reports the shortest lag the search allows, at full confidence),
            // and the fundamental must be one this effect could act on - above
            // a quarter of the sample rate not even its second harmonic fits
            // below Nyquist. Unvoiced, or refused: keep the previous target.
            const double frequency = static_cast<double>(detector_.getFrequencyHz());
            if (frequency > 0.0 && frequency <= 0.25 * sampleRate_)
            {
                const double midi = 69.0 + 12.0 * std::log2(frequency / 440.0);
                if (std::isfinite(midi))
                    target_ = std::clamp(nearestInScale(midi, absoluteMask) - midi,
                                         -12.0, 12.0);
            }
        }

        if (smoothing_ >= 1.0)
        {
            correction_ = target_;
        }
        else
        {
            correction_ += (target_ - correction_) * smoothing_;
            if (std::abs(target_ - correction_) < kLandingSemitones)
                correction_ = target_;   // exact landing
        }

        if (correction_ != publishedCorrection_)
        {
            shifter_.setSemitones(static_cast<T>(correction_));
            publishedCorrection_ = correction_;
        }
    }

    /**
     * @brief Nearest MIDI note whose pitch class belongs to `absoluteMask`.
     *
     * Scans the two octaves around the estimate, which always contains the
     * nearest member of a non-empty pitch-class set (the farthest possible
     * target, a one-note scale, sits 6 semitones away). The scan ascends and
     * compares strictly, so an exact tie resolves to the lower note.
     */
    [[nodiscard]] static double nearestInScale(double midi,
                                               std::uint32_t absoluteMask) noexcept
    {
        const int base = static_cast<int>(std::floor(midi));
        int bestNote = base;
        double bestDistance = 1.0e9;
        for (int note = base - 12; note <= base + 13; ++note)
        {
            const int pitchClass = ((note % 12) + 12) % 12;
            if ((absoluteMask & (1u << pitchClass)) == 0u) continue;
            const double distance = std::abs(static_cast<double>(note) - midi);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestNote = note;
            }
        }
        return static_cast<double>(bestNote);
    }

    // -- Members -----------------------------------------------------------------

    PitchDetector<T> detector_;   ///< YIN front-end (reused unchanged).
    PitchShifter<T> shifter_;     ///< Phase-vocoder back-end (reused unchanged).

    double sampleRate_ = 48000.0;
    int frameSize_ = 0;
    int latency_ = 0;
    bool prepared_ = false;

    // Stream-owner state (audio thread only).
    double target_ = 0.0;                ///< Correction the scale asks for.
    double correction_ = 0.0;            ///< Smoother output, in semitones.
    double publishedCorrection_ = 0.0;   ///< Last value handed to the shifter.
    double smoothing_ = 1.0;             ///< Glide coefficient per interval.
    double smoothingForMs_ = -1.0;       ///< Retune speed it was built for.
    bool publishedFormant_ = false;      ///< Last formant flag handed over.
    int controlPhase_ = 0;               ///< Position inside the control grid.
    int warmupRemaining_ = 0;            ///< Detector samples still to ignore.

    /// Packed scale word: bits 0..11 the absolute pitch-class set (pre-rotated),
    /// 12..15 the root, 16..27 the mask as given (for the getter).
    std::atomic<std::uint32_t> scaleWord_ { 0x0FFF0FFFu };
    std::atomic<T> retuneSpeedMs_ { T(0) };
    std::atomic<bool> formantPreserve_ { false };
};

} // namespace dspark
