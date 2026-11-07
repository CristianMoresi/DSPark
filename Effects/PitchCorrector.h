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
 *   since prepare() or reset(). The detector's first windows are mostly zeros,
 *   so any period it reads there rests on a fraction of a window. Held, such a
 *   reading would stay for the whole track, which is the opposite of what the
 *   hold is for. One window is 43 ms at 44.1/48 kHz, well inside the signal
 *   latency, so nothing audible waits for it.
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
 *   change 0.5 semitones   12.5 / 14.5 ms
 *   change 1               23.3 / 25.6 ms
 *   change 2               44.6 / 49.0 ms
 *   change 3               66.4 / 72.7 ms
 *   change 4               89.2 / 96.6 ms
 *
 * Up to one semitone that fits inside the pipeline's own latency, so the
 * transition is complete before the audio it governs is heard. Above it the
 * traversal is audible by its excess: these are traversals and not errors -
 * the pitch moves continuously, without a click, and lands within 10 cents.
 * The table is the worst of 32 control phases per row, both directions, read
 * by a heterodyne fundamental tracker averaging over one period of the target.
 * Against the engine's slew law (half a semitone per analysis hop, the hop
 * dilating with the active ratio) the rows above one semitone agree to within
 * 5.1%; the two smallest rows read up to 20% above it, which is the estimator
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
 * harmonic-to-residual ratio of a corrected F2 falls below about 7.5 dB, from
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
 * to below about 7.5 dB at F2 (87.31 Hz) and from 38-41 dB to 4.0-4.4 dB at A2
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
 * time at every other rate - reported by getLatency(); the low-latency engine
 * (below) runs at a quarter of that. The correction
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
 * Low-latency engine: setLowLatency(true) replaces the phase vocoder with a
 * pitch-synchronous splice in the time domain. A read head trails the input
 * by half the detector's window (1024 samples at 44.1/48 kHz, 21.3 ms at
 * 48 kHz; the same span at every rate) and
 * advances at the correction ratio through the input history (32-tap sinc;
 * the stretched kernel of Core/Interpolation.h above 2^(1/8)). Every fragment
 * (a sixteenth of the detector window, 128 samples at 48 kHz) the head may
 * jump by a whole number of periods, at least a fragment, under a
 * raised-cosine crossfade over the next fragment: the two reads are then the
 * same waveform, so the splice is inaudible on voiced sound. The jumps keep
 * the head within half a jump of the centre, so the mean delay is the
 * reported latency; while a correction is applied the instantaneous delay
 * wanders within that band: half a period either way for voices below the
 * fragment rate (375 Hz at 48 kHz), less than a fragment above it. At zero
 * correction, and at the first sound after silence (where the head
 * re-centres for free), the engine is an exact delay of getLatency() samples.
 * Unvoiced sound has no period to align on: the head returns to the centre
 * with a power-complementary fade once it strays a fragment from it.
 *
 * The detector's estimate describes its own integration window, which trails
 * the read head by several hundred samples (and up to a hop more), so on its
 * own it would make the correction chase a vibrato about 13 ms late. Once per
 * fragment the engine therefore refines the period AT the read head, by
 * normalized correlation of two windows placed symmetrically about it, over
 * lags within 3 percent of the detector's period, and both the correction
 * and the jumps use that value.
 * A correction change takes effect at the next fragment, with no slew: a 1 to
 * 4 semitone hard snap lands within 10 cents in 4 to 5 ms (the vocoder table
 * above does not apply). The read head plays the input back at the correction
 * ratio, so the formants move with the pitch, as they do in the vocoder
 * without formant preservation.
 *
 * With setFormantPreserve(true) the low-latency engine renders by
 * pitch-synchronous overlap-add instead: grains of the input, one input
 * period apart at their source, are placed one corrected period apart and
 * crossfaded with a raised cosine between their centres, so each output
 * period is one input period at its own rate and the spectral envelope stays
 * in place. Same latency, the same exact delay at zero correction and in
 * silence. A G3 vowel corrected five semitones up keeps its envelope centroid
 * within 8 percent and its spectral centroid within 1 percent (the read head:
 * +20 and +25 percent; the vocoder with its envelope pre-warp: +7.5 and 0).
 *
 * Measured at 48 kHz on synthetic voices (the file's instruments, see the
 * tests), against the default engine and a reference time-domain corrector
 * (2x upsampling, cubic reader, autocorrelation periods, latency 1024
 * reported, 1055 measured):
 *
 *                               low-latency   default   reference
 *   latency, samples                1024        4096       1024 (1055)
 *   +-40 cent vibrato on A3, rms     1.3         36.2        5.5  cents
 *     with formant preservation      4.0
 *   natural voice (jitter, breath)   2.3          4.5        3.2  cents
 *   stepped melody, worst error     0.00         0.14       0.09  cents
 *   harmonic/residual, F2..A4      87-90        39-57      32-35  dB
 *     with formant preservation     59-69
 *   CPU, mono                         42           47         54  ns/sample
 *     with formant preservation       49
 *   (the last column is an unoptimized port, so its CPU is indicative)
 *
 * (Harmonic/residual: a sustained vowel 30 cents sharp corrected onto its
 * note; the instrument's own floor is 89 dB.)
 *
 * Threading: setScale(), setRetuneSpeedMs(), setFormantPreserve(),
 * setLowLatency() and their getters are lock-free single-word atomics and may
 * be called from any thread, with one non-audio writer as everywhere in
 * DSPark; the scale mask and its root travel packed in one word, so a torn
 * mask/root pair cannot exist. prepare() belongs to the setup thread (it
 * allocates), and so do getState()/setState(), which allocate.
 * processBlock(), reset() and getLatency() belong to the stream owner (the
 * latency is a relaxed word, so a host may poll it). The inner detector and
 * shifter are private members whose control entry points are driven only by
 * the stream owner inside processBlock() and reset(); their own control-to-
 * audio hand-offs therefore collapse to same-thread sequential code, with this
 * class's relaxed words as the only cross-thread channel.
 *
 * Dependencies: Analysis/PitchDetector.h, Effects/PitchShifter.h,
 * Music/HarmonyConstants.h, Core/AudioBuffer.h, Core/AudioSpec.h,
 * Core/DenormalGuard.h, Core/DspMath.h, Core/Interpolation.h,
 * Core/StateBlob.h.
 */

#include "../Analysis/PitchDetector.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/Interpolation.h"
#include "../Core/StateBlob.h"
#include "../Music/HarmonyConstants.h"
#include "PitchShifter.h"

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
     * Invalid specifications and non-finite sample rates are ignored and the
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
     *                The low-latency engine sizes itself from the detector's
     *                window instead, so this frame does not change its latency.
     */
    void prepare(const AudioSpec& spec, int fftSize = 0)
    {
        if (!std::isfinite(spec.sampleRate) || !spec.isValid()) return;

        prepared_ = false;

        const int frame = fftSize > 0 ? sanitizeFrame(fftSize)
                                      : automaticFrame(spec.sampleRate);

        detector_.prepare(spec.sampleRate);
        // The corrector keeps the Standard engine: its retune dynamics and the
        // 2 * frame latency are tuned to it, and it moves the shift on every
        // control tick, where Studio's strike anchoring buys nothing.
        shifter_.setQuality(PitchShifter<T>::Quality::Standard);
        shifter_.prepare(spec, frame);

        sampleRate_ = spec.sampleRate;
        frameSize_ = frame;
        spectralLatency_ = shifter_.getLatency();
        prepareSplice(spec);

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

        // The engine is chosen here, so a mode change takes effect as a
        // restart of the stream (the two latencies differ).
        lowActive_ = lowLatency_.load(std::memory_order_relaxed);
        lowFormant_ = lowActive_ && formantPreserve_.load(std::memory_order_relaxed);
        latency_.store(lowActive_ ? spliceDelay_ : spectralLatency_,
                       std::memory_order_relaxed);
        resetSplice();
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

    /** @brief Enables formant preservation (default off): the shifter's
     *  cepstral envelope pre-warp in the default engine, pitch-synchronous
     *  overlap-add in the low-latency one. In the low-latency engine a change
     *  restarts the stream at the next processBlock(), as reset() does. */
    void setFormantPreserve(bool on) noexcept
    {
        formantPreserve_.store(on, std::memory_order_relaxed);
    }

    /**
     * @brief Selects the low-latency engine (default off).
     *
     * On: the pitch-synchronous splice engine described in the file overview,
     * with a latency of half the detector's analysis window (1024 samples at
     * 44.1/48 kHz, 21.3 ms at 48 kHz) instead of the phase vocoder's twice the
     * frame. Formant preservation selects its pitch-synchronous overlap-add
     * rendering (see the file overview). A change of either flag takes
     * effect at the next processBlock() as a restart of the stream, as reset()
     * does: re-read getLatency() after it.
     */
    void setLowLatency(bool on) noexcept
    {
        lowLatency_.store(on, std::memory_order_relaxed);
    }

    /** @return Whether the low-latency engine is requested. */
    [[nodiscard]] bool getLowLatency() const noexcept
    {
        return lowLatency_.load(std::memory_order_relaxed);
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

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("PCOR"), 1);
        w.write("scaleMask", static_cast<int32_t>(getScaleMask()));
        w.write("root", static_cast<int32_t>(getRootPitchClass()));
        w.write("retuneMs", static_cast<float>(getRetuneSpeedMs()));
        w.write("formant", getFormantPreserve());
        w.write("lowLatency", getLowLatency());
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids).
     *  A missing field restores its default: chromatic from C, hard snap,
     *  formants off, low latency off. */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("PCOR")) return false;
        setScale(static_cast<std::uint16_t>(r.read("scaleMask", int32_t(0x0FFF)) & 0x0FFF),
                 r.read("root", int32_t(0)));
        setRetuneSpeedMs(static_cast<T>(r.read("retuneMs", 0.0f)));
        setFormantPreserve(r.read("formant", false));
        setLowLatency(r.read("lowLatency", false));
        return true;
    }

    /** @brief Reports the signal latency in samples. Default engine: twice the
     *  analysis frame, which is 4096 samples at 44.1/48 kHz and about 85 ms at
     *  every supported rate, since the frame follows the rate. Low-latency
     *  engine: half the detector's window, 1024 samples at 44.1/48 kHz (about
     *  21 ms at every rate). Zero before prepare(). */
    [[nodiscard]] int getLatency() const noexcept
    {
        return prepared_ ? latency_.load(std::memory_order_relaxed) : 0;
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

        // A change of engine restarts the stream: the latencies differ, so no
        // crossfade can join the two.
        if (lowLatency_.load(std::memory_order_relaxed) != lowActive_) reset();
        // So does a change of the low-latency engine's formant mode: its grain
        // engine and the read head keep different state.
        if (lowActive_ && formantPreserve_.load(std::memory_order_relaxed) != lowFormant_) reset();

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
            if (controlPhase_ == 0)
            {
                // The low-latency engine refines the pitch at its read head
                // once per fragment, before the correction is taken from it.
                if (lowActive_ && splicePhase_ == 0) updateLocalPitch();
                advanceCorrection(absoluteMask);
                // Splice decisions sit on the same absolute grid and read the
                // detector BEFORE this interval is pushed, so the host's block
                // size cannot move them.
                if (lowActive_ && !lowFormant_ && splicePhase_ == 0) decideSplice();
            }

            const int chunk =
                std::min(numSamples - offset, kControlInterval - controlPhase_);

            if (detectionInput != nullptr)
                detector_.pushSamples(std::span<const T>(
                    detectionInput + offset, static_cast<std::size_t>(chunk)));

            if (warmupRemaining_ > 0)
                warmupRemaining_ = std::max(0, warmupRemaining_ - chunk);

            if (lowActive_)
            {
                if (lowFormant_) grainChunk(buffer, offset, chunk);
                else spliceChunk(buffer, offset, chunk);
            }
            else
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
            // analysis window must be full (a mostly-zero window is too little
            // evidence for a value that will be held), and the fundamental
            // must be one this effect could act on - above
            // a quarter of the sample rate not even its second harmonic fits
            // below Nyquist. Unvoiced, or refused: keep the previous target.
            const double frequency = lowActive_
                ? localHz_ : static_cast<double>(detector_.getFrequencyHz());
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
            if (lowActive_)
            {
                spliceRatio_ = std::exp2(correction_ / 12.0);
                // Rates above 2^(1/8) decimate enough to need the stretched
                // kernel; below it the full-band sinc folds only content above
                // 20 kHz back (at 44.1 kHz and up).
                spliceStep_ = spliceRatio_ > kStretchedFrom
                    ? StretchedSincReader<T>::stepFor(spliceRatio_) : -1;
            }
            else
            {
                shifter_.setSemitones(static_cast<T>(correction_));
            }
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

    // -- Low-latency engine (pitch-synchronous splice) ---------------------------

    /// Rate above which the stretched (decimating) kernel replaces the sinc.
    static constexpr double kStretchedFrom = 1.0905077326652577;   // 2^(1/8)
    /// Below this magnitude a sample counts as silence for the free re-centre
    /// (-120 dBFS: a jump across it cannot be heard).
    static constexpr double kSilence = 1.0e-6;

    /** @brief The shared readers (built on first use: prepare() calls them). */
    static const SincInterpolator<T>& sincTable()
    {
        static const SincInterpolator<T> table;
        return table;
    }
    static const StretchedSincReader<T>& stretchedTable()
    {
        static const StretchedSincReader<T> table;
        return table;
    }

    /** @brief Sizes the splice engine from the detector's window (setup thread). */
    void prepareSplice(const AudioSpec& spec)
    {
        sinc_ = &sincTable();
        stretched_ = &stretchedTable();

        // The detector's window ends at the write head, so its centre is half
        // a window back: a read head at that delay plays exactly the audio the
        // current estimate describes. The delay must also hold a jump of one
        // full period either way of it, and the detector reports periods up to
        // half its window, so half a window is both the alignment and the
        // room the jumps need.
        const int window = detector_.getWindowSize();
        spliceDelay_ = window / 2;
        spliceFragment_ = std::max(kControlInterval, window / 16);
        // The head never trails by more than the centre, half the longest
        // jump, two fragments of drift and a kernel's reach: under a window.
        int size = 1;
        while (size < 2 * window) size <<= 1;
        ringMask_ = static_cast<std::int64_t>(size - 1);
        spliceChannels_ = std::max(1, spec.numChannels);
        ring_.assign(static_cast<std::size_t>(spliceChannels_),
                     std::vector<T>(static_cast<std::size_t>(size), T(0)));

        // Crossfade laws over one fragment, sampled at the midpoints so the
        // two halves of a fade are mirror images: raised cosine (amplitude
        // complementary) for period-aligned jumps, whose two reads are the same
        // waveform, and sine/cosine (power complementary) for jumps through
        // unvoiced sound, whose two reads are uncorrelated.
        const auto f = static_cast<std::size_t>(spliceFragment_);
        fadeAmp_.assign(f, T(0));
        fadeOldPow_.assign(f, T(0));
        fadeNewPow_.assign(f, T(0));
        for (std::size_t k = 0; k < f; ++k)
        {
            const double x = (static_cast<double>(k) + 0.5) / static_cast<double>(f);
            fadeAmp_[k] = static_cast<T>(0.5 - 0.5 * std::cos(pi<double> * x));
            fadeOldPow_[k] = static_cast<T>(std::cos(0.5 * pi<double> * x));
            fadeNewPow_[k] = static_cast<T>(std::sin(0.5 * pi<double> * x));
        }
        // Closest a read may come to the write head: the widest kernel's reach
        // (the stretched kernel at the +12 semitone clamp) plus one sample.
        minReadDelay_ = static_cast<double>(stretched_->reach(
            StretchedSincReader<T>::stepFor(2.0)) + 2);

        // Local period search: two windows of at most kLocalSpan around the
        // longest lag the detector reports, plus the search margin.
        const int longest = window / 2 + localMargin(static_cast<double>(window / 2)) + 2;
        localScratch_.assign(static_cast<std::size_t>(2 * longest + kLocalSpan + 8), T(0));
        localEnergy_.assign(localScratch_.size() + 1, 0.0);
    }

    /// Longest window of the local period search, in samples.
    static constexpr int kLocalSpan = 256;

    /// Lags searched either side of the detector's period: 3 percent (half
    /// a semitone) plus one sample, enough for a vibrato's excursion between
    /// two detector reports.
    [[nodiscard]] static int localMargin(double period) noexcept
    {
        return static_cast<int>(std::ceil(0.03 * period)) + 1;
    }

    /**
     * @brief The period AT the read head (audio thread, once per fragment).
     *
     * The detector's estimate describes its own integration window, which
     * trails the head by about 400 samples plus up to a hop at 48 kHz, so on
     * its own it would make the correction chase a vibrato about 13 ms late.
     * This refines it where the output is: the normalized correlation of two
     * windows placed symmetrically about the head, `lag` apart, over lags
     * within localMargin() of the detector's period, peak refined by a
     * parabola. Channel 0, as the detector. Returns `coarse` when the search
     * finds no clear peak.
     */
    [[nodiscard]] double localPeriod(double coarse) noexcept
    {
        const int margin = localMargin(coarse);
        const int lagLo = std::max(2, static_cast<int>(std::floor(coarse)) - margin);
        const int lagHi = static_cast<int>(std::ceil(coarse)) + margin;
        const int span = std::clamp(static_cast<int>(coarse), 64, kLocalSpan);
        // Centre on the head, but never so close to the write head that the
        // later window would need samples not yet written.
        std::int64_t centre = headInt_;
        const std::int64_t newest = writeCount_ - 1;
        const std::int64_t reachAhead = lagHi / 2 + span / 2 + 2;
        if (centre + reachAhead > newest) centre = newest - reachAhead;
        const std::int64_t first = centre - lagHi / 2 - span / 2 - 1;
        const int count = static_cast<int>(std::min<std::int64_t>(
            { newest - first + 1, static_cast<std::int64_t>(lagHi + span + 4),
              static_cast<std::int64_t>(localScratch_.size()) }));
        const T* ring = ring_.front().data();
        for (int i = 0; i < count; ++i)
            localScratch_[static_cast<std::size_t>(i)] =
                ring[static_cast<std::size_t>((first + i) & ringMask_)];
        localEnergy_[0] = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double v = static_cast<double>(localScratch_[static_cast<std::size_t>(i)]);
            localEnergy_[static_cast<std::size_t>(i) + 1] = localEnergy_[static_cast<std::size_t>(i)] + v * v;
        }

        auto ncc = [&](int lag) noexcept {
            const std::int64_t a = centre - (lag + 1) / 2 - span / 2 - first;
            const std::int64_t b = a + lag;
            if (a < 0 || b + span > count) return -2.0;
            const double ab = static_cast<double>(simd::dotProductT(
                localScratch_.data() + a, localScratch_.data() + b, span));
            const double ea = localEnergy_[static_cast<std::size_t>(a + span)] - localEnergy_[static_cast<std::size_t>(a)];
            const double eb = localEnergy_[static_cast<std::size_t>(b + span)] - localEnergy_[static_cast<std::size_t>(b)];
            const double den = std::sqrt(ea * eb);
            return den > 1e-20 ? ab / den : -2.0;
        };

        int best = -1;
        double bestValue = 0.5;   // below this there is no clear periodicity
        for (int lag = lagLo; lag <= lagHi; ++lag)
        {
            const double v = ncc(lag);
            if (v > bestValue) { bestValue = v; best = lag; }
        }
        if (best <= lagLo || best >= lagHi) return coarse;   // edge: not a peak
        const double ym = ncc(best - 1), yp = ncc(best + 1);
        const double den = ym - 2.0 * bestValue + yp;
        const double shift = den < 0.0 ? 0.5 * (ym - yp) / den : 0.0;
        return static_cast<double>(best) + std::clamp(shift, -0.5, 0.5);
    }

    /** @brief Refreshes localHz_ at a fragment boundary (audio thread). */
    void updateLocalPitch() noexcept
    {
        const double frequency = static_cast<double>(detector_.getFrequencyHz());
        if (!(frequency > 0.0 && frequency <= 0.25 * sampleRate_) || warmupRemaining_ > 0)
        {
            localHz_ = frequency;   // unvoiced, refused or warming up: as reported
            return;
        }
        localHz_ = sampleRate_ / localPeriod(sampleRate_ / frequency);
    }

    /** @brief Clears the splice engine's signal state (stream owner). */
    void resetSplice() noexcept
    {
        for (auto& r : ring_) std::fill(r.begin(), r.end(), T(0));
        writeCount_ = 0;
        headInt_ = -static_cast<std::int64_t>(spliceDelay_);
        headFrac_ = 0.0;
        nextInt_ = headInt_;
        nextFrac_ = 0.0;
        fading_ = false;
        fadeEqualPower_ = false;
        holdUnity_ = false;
        splicePhase_ = 0;
        spliceRatio_ = 1.0;
        spliceStep_ = -1;
        localHz_ = 0.0;
        lastLoud_ = std::numeric_limits<std::int64_t>::min() / 2;
        markPrev_ = markNext_ = -static_cast<double>(spliceDelay_);
        offsetCur_ = offsetNext_ = 0.0;
    }

    /**
     * @brief The decision at a fragment boundary (audio thread): whether the
     *        read head jumps for the next fragment, and where to.
     *
     * Runs before the fragment's first sample is written, so it sees exactly
     * the input before the boundary whatever the host's block size.
     */
    void decideSplice() noexcept
    {
        if (fading_)
        {
            headInt_ = nextInt_;   // the faded-in head is now the only one
            headFrac_ = nextFrac_;
            fading_ = false;
        }
        holdUnity_ = false;

        const auto now = writeCount_;
        const double delay = static_cast<double>(now - headInt_) - headFrac_;
        const double centre = static_cast<double>(spliceDelay_);
        const auto centreInt = now - static_cast<std::int64_t>(spliceDelay_);
        const auto reach = static_cast<std::int64_t>(minReadDelay_);

        // Everything both heads could read is silence: re-centre for free, and
        // read this fragment at unity so the next sound starts exactly at the
        // reported latency.
        if (lastLoud_ < std::min(headInt_, centreInt) - reach)
        {
            headInt_ = centreInt;
            headFrac_ = 0.0;
            holdUnity_ = true;
            return;
        }

        double jump = 0.0;
        bool equalPower = false;
        const double frequency = localHz_;
        const bool voiced = warmupRemaining_ == 0 && frequency > 0.0
                         && frequency <= 0.25 * sampleRate_;
        if (voiced)
        {
            // A jump of whole periods splices the waveform onto itself; it must
            // span at least a fragment, so it outruns the drift of a fragment
            // at any rate up to the +12 semitone clamp. The head is kept within
            // half a jump of the centre, so the mean delay is the latency.
            const double period = sampleRate_ / frequency;
            const double span = period * std::ceil(static_cast<double>(spliceFragment_) / period);
            if (delay < centre - 0.5 * span)      jump = -span;   // back: more delay
            else if (delay > centre + 0.5 * span) jump = span;    // forward
        }
        else if (std::abs(delay - centre) > static_cast<double>(spliceFragment_))
        {
            // No period to align on: return to the centre with a
            // power-complementary fade.
            jump = delay - centre;
            equalPower = true;
        }

        // Safety net for geometry the band rule cannot hold (a fragment
        // drifting further than a jump): never let a read reach the write head.
        const double drift = static_cast<double>(spliceFragment_) * std::max(0.0, spliceRatio_ - 1.0);
        if (jump == 0.0 && delay - 2.0 * drift < minReadDelay_)
        {
            jump = delay - centre;
            equalPower = true;
        }

        if (jump != 0.0)
        {
            const double target = static_cast<double>(headInt_) + headFrac_ + jump;
            nextInt_ = static_cast<std::int64_t>(std::floor(target));
            nextFrac_ = target - static_cast<double>(nextInt_);
            fading_ = true;
            fadeEqualPower_ = equalPower;
        }
    }

    /** @brief Writes a chunk of input into the rings; a non-finite sample
     *  reads as silence (audio thread). */
    void writeRings(AudioBufferView<T> buffer, int offset, int chunk, int nCh) noexcept
    {
        for (int ch = 0; ch < spliceChannels_; ++ch)
        {
            T* ring = ring_[static_cast<std::size_t>(ch)].data();
            const T* in = ch < nCh ? buffer.getChannel(ch) + offset : nullptr;
            for (int k = 0; k < chunk; ++k)
            {
                T x = in != nullptr ? in[k] : T(0);
                if (!std::isfinite(x)) x = T(0);
                ring[static_cast<std::size_t>((writeCount_ + k) & ringMask_)] = x;
                if (std::abs(static_cast<double>(x)) > kSilence) lastLoud_ = writeCount_ + k;
            }
        }
    }

    /**
     * @brief Chooses the next grain when the output reaches a grain centre
     *        (grain mode, audio thread).
     *
     * The next mark lies one corrected period on, the input period divided
     * by the correction ratio. Its source centre is the current one moved by
     * the whole number of input periods that brings it closest to the mark,
     * so the two grains hold the same waveform and the source never drifts
     * more than half a period from the output. Without a period (unvoiced,
     * silent, warming up) the next grain sits a fragment on at offset 0: the
     * exact delay, reached through one fade.
     */
    void advanceGrain() noexcept
    {
        markPrev_ = markNext_;
        offsetCur_ = offsetNext_;
        const double frequency = localHz_;
        const bool voiced = !holdUnity_ && warmupRemaining_ == 0 && frequency > 0.0
                         && frequency <= 0.25 * sampleRate_;
        if (!voiced)
        {
            markNext_ = markPrev_ + static_cast<double>(spliceFragment_);
            offsetNext_ = 0.0;
            return;
        }
        const double period = sampleRate_ / frequency;
        markNext_ = markPrev_ + period / spliceRatio_;
        const double source = markPrev_ + offsetCur_;
        const double periods = std::round((markNext_ - source) / period);
        offsetNext_ = source + periods * period - markNext_;
    }

    /**
     * @brief Renders one chunk of the low-latency engine's formant-preserving
     *        mode in place (audio thread): pitch-synchronous overlap-add.
     *
     * The output is a chain of grains, each a stretch of the input around a
     * source centre, placed with its centre on an output mark (see
     * advanceGrain()). Between two marks the output fades from the first
     * grain to the second with a raised cosine, which makes every grain a
     * raised-cosine-windowed stretch of two corrected periods and their sum
     * exact. Every output period is then one period of the input at its
     * original rate: the pitch follows the marks and the spectral envelope,
     * the formants, stays where it was. Reads are 32-tap sinc at the grains'
     * fractional offsets; the reported latency and the free re-centre in
     * silence are those of the read-head engine.
     */
    void grainChunk(AudioBufferView<T> buffer, int offset, int chunk) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), spliceChannels_);
        writeRings(buffer, offset, chunk, nCh);

        // Everything a grain could read is silence (offsets stay within half
        // a period, under half the delay): re-centre for free.
        const auto now = writeCount_;
        holdUnity_ = lastLoud_ < now - static_cast<std::int64_t>(spliceDelay_ + spliceDelay_ / 2)
                               - static_cast<std::int64_t>(minReadDelay_);
        if (holdUnity_) { offsetCur_ = 0.0; offsetNext_ = 0.0; }

        std::int64_t pos1[kControlInterval], pos2[kControlInterval];
        double frac1[kControlInterval], frac2[kControlInterval];
        T fade[kControlInterval];
        bool cross = false;
        for (int k = 0; k < chunk; ++k)
        {
            const double t = static_cast<double>(now + k - spliceDelay_);
            while (t >= markNext_) advanceGrain();
            const double span = markNext_ - markPrev_;
            const double u = span > 0.0 ? std::clamp((t - markPrev_) / span, 0.0, 1.0) : 1.0;
            fade[k] = static_cast<T>(0.5 - 0.5 * fastCos(pi<double> * u));
            const double p1 = t + offsetCur_, p2 = t + offsetNext_;
            const double f1 = std::floor(p1), f2 = std::floor(p2);
            pos1[k] = static_cast<std::int64_t>(f1); frac1[k] = p1 - f1;
            pos2[k] = static_cast<std::int64_t>(f2); frac2[k] = p2 - f2;
            cross = cross || offsetCur_ != offsetNext_;
        }
        headInt_ = now + chunk - static_cast<std::int64_t>(spliceDelay_)
                 + static_cast<std::int64_t>(std::lround(offsetCur_));

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* ring = ring_[static_cast<std::size_t>(ch)].data();
            T* out = buffer.getChannel(ch) + offset;
            for (int k = 0; k < chunk; ++k)
            {
                const T a = sinc_->readRing(ring, ringMask_, pos1[k], frac1[k]);
                if (!cross || (pos1[k] == pos2[k] && frac1[k] == frac2[k]))
                {
                    out[k] = a;
                    continue;
                }
                const T b = sinc_->readRing(ring, ringMask_, pos2[k], frac2[k]);
                out[k] = a + (b - a) * fade[k];
            }
        }

        writeCount_ += chunk;
        splicePhase_ += chunk;
        if (splicePhase_ >= spliceFragment_) splicePhase_ = 0;
    }

    /** @brief One read of a ring at an integer position plus a fraction. */
    T readRing(const T* ring, std::int64_t pos, double frac) const noexcept
    {
        return spliceStep_ < 0
            ? sinc_->readRing(ring, ringMask_, pos, frac)
            : stretched_->readRing(ring, ringMask_, pos, frac, spliceStep_);
    }

    /**
     * @brief Renders one chunk of the low-latency engine in place (audio
     *        thread). A chunk never crosses a control interval, and so never a
     *        fragment.
     */
    void spliceChunk(AudioBufferView<T> buffer, int offset, int chunk) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), spliceChannels_);

        // 1. Write the input.
        writeRings(buffer, offset, chunk, nCh);

        // 2. The head trajectories, shared by every channel (stereo-linked).
        const double ratio = holdUnity_ ? 1.0 : spliceRatio_;
        std::int64_t pos1[kControlInterval], pos2[kControlInterval];
        double frac1[kControlInterval], frac2[kControlInterval];
        {
            std::int64_t i1 = headInt_, i2 = nextInt_;
            double f1 = headFrac_, f2 = nextFrac_;
            for (int k = 0; k < chunk; ++k)
            {
                pos1[k] = i1; frac1[k] = f1;
                pos2[k] = i2; frac2[k] = f2;
                f1 += ratio;
                f2 += ratio;
                const double c1 = std::floor(f1);
                const double c2 = std::floor(f2);
                i1 += static_cast<std::int64_t>(c1); f1 -= c1;
                i2 += static_cast<std::int64_t>(c2); f2 -= c2;
            }
            headInt_ = i1; headFrac_ = f1;
            if (fading_) { nextInt_ = i2; nextFrac_ = f2; }
        }

        // 3. Read every channel.
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* ring = ring_[static_cast<std::size_t>(ch)].data();
            T* out = buffer.getChannel(ch) + offset;
            if (!fading_)
            {
                for (int k = 0; k < chunk; ++k) out[k] = readRing(ring, pos1[k], frac1[k]);
                continue;
            }
            const auto at = static_cast<std::size_t>(splicePhase_);
            for (int k = 0; k < chunk; ++k)
            {
                const T a = readRing(ring, pos1[k], frac1[k]);
                const T b = readRing(ring, pos2[k], frac2[k]);
                const std::size_t j = at + static_cast<std::size_t>(k);
                out[k] = fadeEqualPower_ ? a * fadeOldPow_[j] + b * fadeNewPow_[j]
                                         : a + (b - a) * fadeAmp_[j];
            }
        }

        writeCount_ += chunk;
        splicePhase_ += chunk;
        if (splicePhase_ >= spliceFragment_) splicePhase_ = 0;
    }

    // -- Members -----------------------------------------------------------------

    PitchDetector<T> detector_;   ///< YIN front-end (reused unchanged).
    PitchShifter<T> shifter_;     ///< Phase-vocoder back-end (reused unchanged).

    double sampleRate_ = 48000.0;
    int frameSize_ = 0;
    int spectralLatency_ = 0;              ///< The phase vocoder's latency.
    std::atomic<int> latency_ { 0 };       ///< Latency of the running engine.
    bool prepared_ = false;

    // Low-latency engine (stream owner, sized by prepare()).
    const SincInterpolator<T>* sinc_ = nullptr;
    const StretchedSincReader<T>* stretched_ = nullptr;
    std::vector<std::vector<T>> ring_;     ///< Input history per channel.
    std::vector<T> fadeAmp_, fadeOldPow_, fadeNewPow_;
    std::int64_t ringMask_ = 0;
    int spliceChannels_ = 1;
    int spliceDelay_ = 0;                  ///< Centre delay = reported latency.
    int spliceFragment_ = 64;              ///< Decision and crossfade length.
    double minReadDelay_ = 34.0;
    std::int64_t writeCount_ = 0;          ///< Samples written since reset().
    std::int64_t headInt_ = 0;             ///< Read head, integer part.
    double headFrac_ = 0.0;
    std::int64_t nextInt_ = 0;             ///< Head faded in during a jump.
    double nextFrac_ = 0.0;
    std::int64_t lastLoud_ = 0;            ///< Last sample above kSilence.
    std::vector<T> localScratch_;          ///< Contiguous copy for the search.
    std::vector<double> localEnergy_;      ///< Its prefix sums of squares.
    double localHz_ = 0.0;                 ///< Pitch at the head (0: unvoiced).
    double spliceRatio_ = 1.0;             ///< Read rate from the correction.
    int spliceStep_ = -1;                  ///< Stretched kernel step, -1 = sinc.
    int splicePhase_ = 0;                  ///< Position inside the fragment.
    bool fading_ = false;
    bool fadeEqualPower_ = false;
    bool holdUnity_ = false;               ///< Silent fragment: read at unity.
    bool lowActive_ = false;               ///< Engine in use (stream owner).
    bool lowFormant_ = false;              ///< Its formant-preserving grain mode in use.
    double markPrev_ = 0.0;                ///< Grain mode: output time of the current grain centre.
    double markNext_ = 0.0;                ///< Output time of the next grain centre.
    double offsetCur_ = 0.0;               ///< Source minus output time, current grain.
    double offsetNext_ = 0.0;              ///< The same for the next grain.
    std::atomic<bool> lowLatency_ { false };

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
