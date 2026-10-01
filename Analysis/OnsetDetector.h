// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file OnsetDetector.h
 * @brief Causal SuperFlux onset detector with a Boeck-2012 adaptive peak picker.
 *
 * Implements the SuperFlux onset-detection function (Boeck & Widmer, DAFx-13):
 * a log-frequency triangular filterbank magnitude spectrogram, a frequency
 * maximum filter that suppresses vibrato/tremolo false positives, and a
 * half-wave-rectified spectral flux to the mu-th previous frame. The onset
 * strength envelope (ODF) is peak-picked with the online-capable rule of
 * Boeck, Krebs & Schedl (ISMIR 2012). Two simpler ODFs are also provided:
 * plain SpectralFlux and a rectified ComplexDomain function (Dixon, DAFx-06).
 *
 * The STFT front-end is built directly on FFTReal + WindowFunctions with a
 * mirrored analysis ring (the same technique as PitchDetector), which gives
 * exact, block-size-independent control over frame timing -- a precondition
 * for the deterministic latency contract below. It is the shared onset
 * front-end consumed by BeatTracker.
 *
 * Frame length is a TIME requirement, not a sample count. The ODF is computed
 * over a log-frequency filterbank whose declared spacing is a quarter tone, and
 * the STFT bin width fs/fftSize decides the frequency above which that spacing
 * is actually delivered: f_qt = (fs/fftSize) / (2^(1/24) - 1), about 34 times
 * the bin width. A CONSTANT sample count therefore moves the detector's usable
 * low register with the sample rate (measured: 800 Hz at 48 kHz/2048, 3199 Hz
 * at 192 kHz/2048, and soft bass onsets around E1..B2 with 10 ms attacks fall
 * from 7/8 recalled at 48 kHz to 5/8 at 96 kHz at a fixed 2048). The default
 * frame is therefore AUTOMATIC and holds the span constant instead; see
 * prepare(). The ODF magnitude scale is frame-invariant as well (band
 * magnitudes are scaled by 2048/fftSize before the log compression), so the
 * peak-pick delta selects the same sensitivity at every rate -- neither the
 * resolution nor the threshold's meaning moves with the session rate. Recall
 * on that corpus under the defaults is 7 of 8 at EVERY rate: one soft F#1
 * stays below the default delta everywhere. That is the detector's soft-bass
 * sensitivity limit, the same at all rates -- raise it with setThreshold()
 * (smaller delta), at the usual false-positive cost.
 *
 * Latency (ONE definition). Frame N (Hann; automatic by default), hop =
 * round(fs/200) (221 samples at 44.1 kHz, 240 at 48 kHz; ~5 ms). The detector
 * reports every onset at a
 * single fixed causal reporting latency
 *
 *     L = fftSize + hop        (getLatencySamples()).
 *
 * With the automatic frame this is 2048 + 221 = 2269 samples at 44.1 kHz
 * (~51.4 ms), 2048 + 240 = 2288 at 48 kHz (~47.7 ms), 4096 + 480 = 4576 at
 * 96 kHz (~47.7 ms) and 8192 + 960 = 9152 at 192 kHz (~47.7 ms) -- one
 * latency in TIME across the range, not one in samples.
 * The onsetDetected() latch asserts exactly L samples after the onset's
 * reference sample (the analysis-frame centre that localises the transient).
 * Detection completes internally earlier (~fftSize/2 + hop); events are held
 * and released at the fixed offset L so a caller sees one deterministic,
 * block-size-independent latency regardless of where in a frame the onset
 * falls. The adaptive threshold's pre_avg look-back (100 ms) is a
 * backward-looking warm-up, NOT part of L: it is history the moving mean needs
 * before its first valid decision and adds zero per-onset reporting latency.
 *
 * Picker defaults: pre_max = post_max = 30 ms, pre_avg = 100 ms,
 * post_avg = 70 ms (offline only), combination width = 30 ms. In the causal
 * streaming path post_avg = 0 and post_max is one hop -- the single-frame
 * confirmation that a candidate is a maximum, which is exactly the +hop term
 * of L. detectOffline() uses the symmetric (post_* > 0) picker.
 *
 * Threading:
 * - prepare(): setup thread (allocates; not concurrent with the audio path).
 * - processBlock() / pushSamples() / reset(): audio thread (stream owner);
 *   reset() is not concurrent with pushSamples().
 * - onsetDetected() / getOnsetStrength() / getLastOnsetSample() /
 *   getLatencySamples(): any thread, lock-free (published as atomics; the
 *   words are independent, so a reader overlapping a release may pair a
 *   fresh reference sample with the previous strength -- benign for
 *   triggering/metering).
 * - getLastOdfFrame(): STREAM OWNER ONLY -- plain words, no publication. It
 *   is the envelope readout for the component that is itself driving
 *   pushSamples() and therefore knows when a frame boundary passed; any other
 *   thread must use the atomic onset readouts above.
 * - setMethod() / setThreshold() / setAdaptiveWhitening(): control thread
 *   (independent single-word relaxed atomics; non-finite thresholds are
 *   ignored).
 * - getMethod() / getThreshold() / getAdaptiveWhitening(): any thread,
 *   lock-free (the same relaxed atomics the setters write).
 * - detectOffline() / detectOfflineOnsets() and the incremental session
 *   beginOffline() / pushOffline() / finishOffline() / finishOfflineOnsets():
 *   offline (allocates); not an audio-thread call. A session drives the same
 *   analysis state as the streaming path, so it must not be interleaved with
 *   pushSamples() or run while the audio path is running.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O, so it is unaffected by DSPARK_NO_FILE_IO.
 *
 * Dependencies: DspMath.h, detail/OnsetFeatures.h, AudioBuffer.h, AudioSpec.h.
 */

#include "../Core/DspMath.h"
#include "detail/OnsetFeatures.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class OnsetDetector
 * @brief Causal SuperFlux onset detector with lock-free readout.
 *
 * Role: analysis readout. It consumes const audio (AudioBufferView<const T> or
 * a raw span) and never mutates it. All heap use happens in prepare(); the
 * audio path (processBlock/pushSamples) allocates nothing, takes no lock and
 * throws nothing.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class OnsetDetector final
{
public:
    /** @brief Onset-detection function family. Default SuperFlux. */
    enum class Method { SpectralFlux, ComplexDomain, SuperFlux };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Allocates all state and configures the STFT front-end.
     *
     * Not real-time safe (allocates). Any previous stream state is cleared.
     * An invalid spec (non-finite or non-positive sample rate) is ignored,
     * preserving the previous configuration.
     *
     * @param spec    Audio environment (only sampleRate is used; the detector
     *                is mono -- feed channel 0, or mix down before pushing).
     * @param fftSize Analysis frame size. Values <= 0 (the default) select the
     *                AUTOMATIC frame: the smallest power of two in
     *                [512, 16384] spanning at least 2048/48000 s (~42.7 ms) at
     *                spec.sampleRate -- 2048 at 44.1/48 kHz, 4096 at
     *                88.2/96 kHz, 8192 at 176.4/192 kHz, 16384 at 384 kHz,
     *                512 at 8 kHz. Above 384 kHz the 16384 ceiling binds and
     *                the bin width widens again (documented, not fixed: pass
     *                an explicit frame there). Explicit positive values are rounded up to
     *                a power of two in [64, 1<<16] and honoured as given, with
     *                the reduced validity described below. Read the resolved
     *                value back with getFftSize().
     * @param hop     Hop in samples; hop <= 0 selects round(fs/200) (~5 ms).
     *                Clamped to [1, fftSize].
     *
     * FRAME LENGTH IS A TIME REQUIREMENT. Everything the frame decides is a
     * function of the RATIO fs/fftSize, never of the count alone:
     *
     * - STFT bin width          = fs/fftSize   (23.44 Hz at 48 kHz/2048)
     * - quarter-tone floor f_qt = (fs/fftSize) / (2^(1/24) - 1), the frequency
     *   above which the filterbank's declared quarter-tone spacing is really
     *   delivered (measured band counts: 135 bands and f_qt 800 Hz at
     *   48 kHz/2048; 111 bands and 1600 Hz at 96 kHz/2048; 88 bands and
     *   3199 Hz at 192 kHz/2048 -- the same 2048 samples, four times the floor)
     * - reporting latency L/fs  = (fftSize + hop)/fs seconds
     *
     * With a CONSTANT sample count the low register therefore degrades as the
     * rate rises: measured on soft bass onsets (E1..B2, 10 ms attacks) the
     * fixed 2048 frame recalls 7/8 at 44.1/48 kHz but 5/8 at 96 kHz, while
     * the automatic frame holds the 7/8 reference recall there (the missed
     * F#1 sits below the default delta at every rate; see the file header).
     * Percussive clicks, mid-register notes
     * and the vibrato/tremolo false-positive guard were measured unaffected at
     * every rate from 44.1 to 192 kHz, so this is a low-register loss, not a
     * general one. Explicit frames stay available for callers who want the
     * shorter one: at 192 kHz an explicit 2048 buys ~10.7 ms of frame span
     * (L ~= 15.7 ms) at the cost of the register above.
     *
     * ODF SCALE. The SuperFlux band magnitudes are scaled by 2048/fftSize
     * before the log10(x + 1) compression, so the onset-strength scale --
     * and with it the meaning of setThreshold()'s delta -- is the same at
     * every frame length, and therefore at every rate under the automatic
     * frame. Without this, |X| grows linearly with the frame and quiet-onset
     * sensitivity roughly doubles per rate-family doubling against a fixed
     * delta. The factor is exactly 1 at the 2048-sample reference where the
     * default delta was tuned, so 44.1/48 kHz default behaviour (and any
     * explicit-2048 caller at any rate) is bit-identical to previous
     * releases. Explicit frames OTHER than 2048 now read the ODF on the
     * reference scale too -- an intentional behaviour change: one delta means
     * one sensitivity, at every frame length. Under adaptive whitening the
     * per-bin peak division removes the growth wherever the running peak
     * exceeds the whitening floor (1e-4), so the whitened path is not scaled
     * again. Caveat: the floor is an absolute magnitude, so bins whose peak
     * is held AT the floor keep the raw frame-scaled magnitude -- very quiet
     * whitened material therefore retains a residual rate dependence
     * (borderline events can appear at high rates that a 48 kHz session does
     * not report).
     *
     * CPU AND MEMORY. The hop is TIME-fixed (round(fs/200), ~200 frames per
     * second at every rate) while the automatic frame follows the rate, so
     * CPU per second of audio is NOT rate-invariant under the automatic
     * frame: measured ~2x at 88.2/96 kHz and ~3.8x at 176.4/192 kHz versus
     * the old fixed-2048 default at the same rate (50.4 / 95.6 / 189.6 ms of
     * CPU per 10 s of audio at 48/96/192 kHz automatic, vs 50.4 ms at
     * 192 kHz with an explicit 2048; g++ -O2, one core -- absolute numbers
     * vary by machine, the growth tracks the frame size). prepare()-time
     * heap grows the same way: ~125 KB at 44.1/48 kHz, ~223 KB at
     * 88.2/96 kHz, ~420 KB at 176.4/192 kHz (float instantiation). The audio
     * path stays allocation-free at every size. Budget from these figures;
     * an explicit 2048 restores the old cost at the cost of the register
     * above.
     */
    void prepare(const AudioSpec& spec, int fftSize = 0, int hop = 0)
    {
        if (!(spec.sampleRate > 0.0) || !std::isfinite(spec.sampleRate))
            return;

        prepared_ = false; // gate OFF: the audio path is a no-op while rebuilding

        sampleRate_ = spec.sampleRate;

        if (fftSize <= 0)
        {
            // Automatic: hold the analysis TIME SPAN constant across sample
            // rates (the span 2048 samples cover at 48 kHz). Bin width and the
            // filterbank's quarter-tone floor are both fs/fftSize, so a
            // constant span pins them -- and with them the usable low register
            // -- instead of letting them widen as fs rises.
            fftSize_ = detail::OnsetSpectrum<T>::defaultFrameSize(sampleRate_);
        }
        else
        {
            int fs = std::clamp(fftSize, kMinFft, kMaxFft);
            int pow2 = kMinFft;
            while (pow2 < fs) pow2 <<= 1;
            fftSize_ = pow2;
        }

        if (hop <= 0)
            hop_ = std::max(1, static_cast<int>(std::lround(sampleRate_ / 200.0)));
        else
            hop_ = hop;
        hop_ = std::clamp(hop_, 1, fftSize_);

        latencySamples_.store(static_cast<int64_t>(fftSize_) + hop_,
                              std::memory_order_relaxed);

        // Analysis-window group-delay compensation: half-wave-rectified spectral
        // flux peaks on the rising flank of a transient, ~kLocalizationLead*N
        // before the windowed-energy peak. Adding it back centres the reported
        // onset on the transient (calibrated on band-limited clicks; softer
        // onsets localise slightly late but stay within the acceptance window).
        localizationOffset_ = static_cast<int>(std::lround(kLocalizationLead
                                                           * static_cast<double>(fftSize_)));
        // Warm-up: suppress onsets until the analysis ring is fully primed so
        // the silence->first-input ramp cannot fire a spurious onset.
        primeFrames_ = fftSize_ / hop_ + 2;

        ring_.assign(static_cast<size_t>(fftSize_) * 2, T(0));
        spectrum_.prepare(fftSize_);
        novelty_.prepare(sampleRate_, fftSize_);

        // Peak-picker windows in frames (derived from the ms defaults above).
        preMaxFrames_  = msToFrames(30.0);
        postMaxFrames_ = msToFrames(30.0);   // offline; causal uses 1
        preAvgFrames_  = msToFrames(100.0);
        postAvgFrames_ = msToFrames(70.0);   // offline; causal uses 0
        waitFrames_    = msToFrames(30.0);

        // Causal ODF history: enough for the pre_avg look-back plus the
        // single-frame causal confirmation.
        odfHistLen_ = std::max(preAvgFrames_, preMaxFrames_) + 4;
        odfHist_.assign(static_cast<size_t>(odfHistLen_), T(0));

        // Pending-onset ring: onsets are held from detection until their
        // release sample (reference + L). At most ceil(L/hop)+2 can be
        // in flight; size generously to a power-of-two-ish bound.
        const int maxPending = (fftSize_ + hop_) / hop_ + 4;
        pending_.assign(static_cast<size_t>(std::max(8, maxPending)),
                        PendingOnset{});
        pendingHead_ = 0;
        pendingCount_ = 0;

        resetState();

        method_.store(Method::SuperFlux, std::memory_order_relaxed);
        // A conservative default delta, tuned against a synthetic corpus of
        // clicks, soft onsets and noise beds. Callers override per material.
        threshold_.store(kDefaultDelta, std::memory_order_relaxed);
        whitening_.store(false, std::memory_order_relaxed);

        prepared_ = true; // gate ON
    }

    /** @brief Selects the ODF family. Lock-free. */
    void setMethod(Method m) noexcept { method_.store(m, std::memory_order_relaxed); }

    /**
     * @brief Sets the adaptive peak-pick delta (margin above the moving mean).
     * Non-finite values are ignored; negative values are clamped to 0.
     * The delta is read against the frame-invariant ODF scale (see
     * prepare()), so one value selects the same sensitivity at every rate
     * and frame length.
     */
    void setThreshold(T deltaAboveMean) noexcept
    {
        if (!std::isfinite(deltaAboveMean)) return;
        threshold_.store(std::max(T(0), deltaAboveMean), std::memory_order_relaxed);
    }

    /** @brief Enables Stowell-Plumbley adaptive whitening (default off). */
    void setAdaptiveWhitening(bool on) noexcept
    {
        whitening_.store(on, std::memory_order_relaxed);
    }

    /** @brief ODF family in force. */
    [[nodiscard]] Method getMethod() const noexcept
    {
        return method_.load(std::memory_order_relaxed);
    }

    /** @brief Peak-pick delta in force, after setThreshold()'s clamping. */
    [[nodiscard]] T getThreshold() const noexcept
    {
        return threshold_.load(std::memory_order_relaxed);
    }

    /** @brief True when adaptive whitening is on. */
    [[nodiscard]] bool getAdaptiveWhitening() const noexcept
    {
        return whitening_.load(std::memory_order_relaxed);
    }

    // -- Audio path (causal, RT-safe) ---------------------------------------

    /**
     * @brief Feeds a mono block; reads channel 0 only. Const, never mutated.
     *
     * Lock-free and allocation-free. Safe no-op before prepare().
     */
    void processBlock(AudioBufferView<const T> in) noexcept
    {
        if (!prepared_ || in.getNumChannels() < 1) return;
        const T* ch0 = in.getChannel(0);
        pushSamples(std::span<const T>(ch0, static_cast<size_t>(in.getNumSamples())));
    }

    /**
     * @brief Feeds a mono stream of samples. Lock-free, allocation-free.
     *
     * Onsets fire at the fixed reporting latency L = fftSize + hop after their
     * reference sample; onsetDetected() latches per processing call (see the
     * file header). Safe no-op before prepare().
     */
    void pushSamples(std::span<const T> samples) noexcept
    {
        if (!prepared_) return;

        bool firedThisCall = false;

        for (const T s : samples)
        {
            const T x = std::isfinite(s) ? s : T(0);

            ring_[static_cast<size_t>(writePos_)] = x;
            ring_[static_cast<size_t>(writePos_ + fftSize_)] = x;
            if (++writePos_ >= fftSize_) writePos_ = 0;

            ++totalSamples_;

            // Release any pending onset whose reporting sample has arrived.
            while (pendingCount_ > 0)
            {
                const PendingOnset& p = pending_[static_cast<size_t>(pendingHead_)];
                if (totalSamples_ - 1 >= p.reportSample)
                {
                    lastOnsetSample_.store(p.referenceSample, std::memory_order_relaxed);
                    onsetStrength_.store(p.strength, std::memory_order_relaxed);
                    firedThisCall = true;
                    pendingHead_ = (pendingHead_ + 1) % static_cast<int>(pending_.size());
                    --pendingCount_;
                }
                else break;
            }

            if (++hopCounter_ >= hop_)
            {
                hopCounter_ = 0;
                analyzeFrame();
            }
        }

        onsetLatched_.store(firedThisCall, std::memory_order_relaxed);
    }

    // -- Readout (lock-free) -------------------------------------------------

    /** @brief True if an onset was reported during the most recent call. */
    [[nodiscard]] bool onsetDetected() const noexcept
    {
        return onsetLatched_.load(std::memory_order_relaxed);
    }

    /** @brief Onset strength (ODF value, frame-invariant scale; see
     *         prepare()) of the most recent reported onset. */
    [[nodiscard]] T getOnsetStrength() const noexcept
    {
        return onsetStrength_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reference sample index (frame centre) of the most recent onset.
     * The latch fires exactly getLatencySamples() samples after this index.
     */
    [[nodiscard]] int64_t getLastOnsetSample() const noexcept
    {
        return lastOnsetSample_.load(std::memory_order_relaxed);
    }

    /** @brief The single causal reporting latency, L = fftSize + hop (samples). */
    [[nodiscard]] int getLatencySamples() const noexcept
    {
        return static_cast<int>(latencySamples_.load(std::memory_order_relaxed));
    }

    /** @brief Analysis frame in samples actually in effect: the automatic
     *         choice when prepare() got fftSize <= 0, the rounded explicit
     *         request otherwise. Divide by the sample rate for the span in
     *         seconds; fs/getFftSize() is the STFT bin width. */
    [[nodiscard]] int getFftSize() const noexcept { return fftSize_; }

    /** @brief Hop in samples in effect (round(fs/200) unless overridden). */
    [[nodiscard]] int getHopSize() const noexcept { return hop_; }

    /** @brief Number of log-frequency filterbank bands built for the resolved
     *         frame; a direct readout of the analysis resolution in force. */
    [[nodiscard]] int getNumBands() const noexcept { return novelty_.getNumBands(); }

    // -- Onset-strength envelope (stream owner only) -------------------------

    /**
     * @brief One frame of the onset-strength envelope (the ODF before the
     *        peak picker).
     *
     * The peak picker answers "was there an onset"; a periodicity analysis
     * needs the continuous strength curve the picker thresholds, because the
     * pulse it looks for is carried by the shape between onsets as much as by
     * the events that clear the threshold. This is that curve, one frame at a
     * time.
     */
    /// Register groups the SuperFlux bands are split into for the
    /// per-register readout: below 200 Hz (kick, bass), 200-800 Hz, 800 Hz to
    /// 3.2 kHz, and above (hats, consonants). Two octaves each above the first.
    static constexpr int kNumRegisters = detail::OnsetNovelty<T>::kNumRegisters;

    struct OdfFrame
    {
        T value = T(0);              ///< ODF value of the most recent frame.
        int64_t referenceSample = 0; ///< Sample index the frame localises to.
        int64_t frameIndex = 0;      ///< Frames computed since the last reset.
        /// The same flux restricted to each register group (mean over the
        /// group's bands; SuperFlux only, zero for the other methods). A
        /// consumer that wants every register to count - a beat tracker, for
        /// which a kick is as much evidence as a dense hi-hat - normalises
        /// these separately instead of reading `value`, where the upper
        /// registers own most of the bands and so most of the mean.
        std::array<T, kNumRegisters> registers {};
    };

    /**
     * @brief The most recent analysis frame's onset-strength value.
     *
     * STREAM OWNER ONLY -- this is not a cross-thread readout. It hands back
     * three plain words that describe one frame, and it is meant for the
     * component that is itself feeding pushSamples(): that caller knows
     * exactly when a frame boundary passed (every getHopSize() samples from
     * the last reset), so it can read the frame it just caused without any
     * publication at all. Reading it from another thread would race the
     * writer word by word and could pair a fresh value with a stale reference
     * sample, which is the one thing a beat grid cannot survive. Other
     * threads use onsetDetected() / getOnsetStrength() / getLastOnsetSample(),
     * which are published atomically for exactly that purpose.
     *
     * `frameIndex` counts from 1 for the first frame after a reset and is 0
     * before any frame has been computed. Frames below getWarmupFrames() are
     * computed over a partly-empty analysis ring and their values are not
     * meaningful; see that method.
     *
     * Reflects the streaming path (processBlock/pushSamples). detectOffline()
     * runs its own envelope internally and leaves this cleared.
     */
    [[nodiscard]] OdfFrame getLastOdfFrame() const noexcept
    {
        return OdfFrame { lastOdfValue_, lastOdfRef_, frameIndex_, lastRegisters_ };
    }

    /**
     * @brief Frames at the start of a stream whose ODF value is warm-up, not
     *        signal.
     *
     * The analysis ring starts empty, so the first frames measure the step
     * from silence into the first input as well as the input itself, and part
     * of the flux they report is an artefact of that step. The detector
     * suppresses its own onsets over this span, which is what the number is
     * for.
     *
     * Whether an envelope reader should discard the same span is its own
     * decision and is not obviously yes. Measured on a beat grid built from
     * this envelope, discarding it costs a real beat whenever the material
     * starts at sample 0 -- F 0.9919 against 1.0000, one beat missing at the
     * head of every such signal -- while the hazard it guards against did not
     * appear even on a full-level tone starting at sample 0 with no beat
     * there, because a consumer that removes a local baseline has already
     * removed the step. That consumer therefore keeps the frames.
     *
     * Equal to fftSize/hop + 2: the frames needed to fill the ring, plus two
     * so the flux to the previous frame is itself computed from two full
     * frames.
     */
    [[nodiscard]] int getWarmupFrames() const noexcept { return primeFrames_; }

    /**
     * @brief How far behind the newest input sample a frame's reference sample
     *        sits, in samples.
     *
     * The envelope has its own delay and it is NOT getLatencySamples(). That
     * one is the ONSET latch delay: detected events are deliberately held and
     * released at a fixed offset so a caller sees one block-size-independent
     * latency. An envelope reader takes each frame as it is computed and does
     * not wait for that release, so what it pays is only the distance from the
     * frame's reference sample to the input sample that completed the frame --
     * half the analysis frame, less the group-delay compensation already
     * folded into the reference. Always smaller than getLatencySamples(); a
     * consumer that reports positions in the caller's timeline uses the
     * reference sample directly and needs this only to state its own delay.
     */
    [[nodiscard]] int getEnvelopeLatencySamples() const noexcept
    {
        return fftSize_ / 2 - localizationOffset_;
    }

    // -- Offline -------------------------------------------------------------

    /** @brief One offline onset: where it is and how strong it was. */
    struct Onset
    {
        int64_t sample = 0; ///< Reference sample (frame centre).
        T strength = T(0);  ///< ODF value at the peak, the scale getOnsetStrength() reports.
    };

    /**
     * @brief Offline detection over a whole mono buffer (channel 0).
     *
     * Runs the same ODF with the symmetric (post_max/post_avg > 0) picker for
     * slightly higher F, and returns onset sample positions (frame-centre
     * references, ascending). Allocates -- not an audio-thread call. Resets
     * the streaming state on entry. Identical to beginOffline(), one
     * pushOffline() of the whole buffer, finishOffline().
     */
    std::vector<int64_t> detectOffline(AudioBufferView<const T> whole)
    {
        return positionsOf(detectOfflineOnsets(whole));
    }

    /** @brief detectOffline() with each onset's strength alongside its position. */
    std::vector<Onset> detectOfflineOnsets(AudioBufferView<const T> whole)
    {
        if (!prepared_ || whole.getNumChannels() < 1) return {};
        const int n = std::max(0, whole.getNumSamples());
        beginOffline(n);
        pushOffline(std::span<const T>(whole.getChannel(0), static_cast<size_t>(n)));
        return finishOfflineOnsets();
    }

    /**
     * @brief Opens an incremental offline analysis.
     *
     * For material that arrives in pieces -- a file decoded block by block, a
     * recording too long to hold -- without concatenating it first. Feed it
     * with pushOffline() in blocks of any size and close it with
     * finishOffline(); the result is bit-identical to detectOffline() over the
     * concatenation, whatever the blocking. Only the onset-strength envelope
     * is kept between calls (one value and one position per hop), not the
     * audio.
     *
     * Resets the streaming state and latches the method and whitening in force
     * for the whole session; the threshold is read when the session finishes.
     * @param expectedSamples Optional length hint, used only to reserve the
     *        envelope up front.
     */
    void beginOffline(int64_t expectedSamples = 0)
    {
        offOdf_.clear();
        offRef_.clear();
        offlineOpen_ = false;
        if (!prepared_) return;
        if (expectedSamples > 0)
        {
            const size_t frames = static_cast<size_t>(expectedSamples / std::max(1, hop_) + 2);
            offOdf_.reserve(frames);
            offRef_.reserve(frames);
        }
        resetState();
        offMethod_ = method_.load(std::memory_order_relaxed);
        offWhiten_ = whitening_.load(std::memory_order_relaxed);
        offTotal_ = 0;
        offHop_ = 0;
        offlineOpen_ = true;
    }

    /** @brief Feeds the next piece of an offline session. No-op outside one. */
    void pushOffline(std::span<const T> samples)
    {
        if (!offlineOpen_) return;
        for (const T s : samples)
        {
            const T v = std::isfinite(s) ? s : T(0);
            ring_[static_cast<size_t>(writePos_)] = v;
            ring_[static_cast<size_t>(writePos_ + fftSize_)] = v;
            if (++writePos_ >= fftSize_) writePos_ = 0;
            ++offTotal_;
            if (++offHop_ >= hop_)
            {
                offHop_ = 0;
                offOdf_.push_back(computeOdf(offMethod_, offWhiten_));
                offRef_.push_back(referenceSample(offTotal_));
            }
        }
    }

    /** @brief Closes the session and returns onset positions, as detectOffline().
     *         Empty when no session is open. */
    std::vector<int64_t> finishOffline()
    {
        return positionsOf(finishOfflineOnsets());
    }

    /** @brief Closes the session and returns onsets with their strengths. */
    std::vector<Onset> finishOfflineOnsets()
    {
        std::vector<Onset> out;
        if (!offlineOpen_) return out;
        offlineOpen_ = false;

        // Symmetric peak-pick over the whole envelope.
        const T delta = threshold_.load(std::memory_order_relaxed);
        const std::vector<T>& odf = offOdf_;
        detail::OnsetPeakPicker<T> picker;
        const typename detail::OnsetPeakPicker<T>::Windows windows {
            static_cast<size_t>(preMaxFrames_), static_cast<size_t>(postMaxFrames_),
            static_cast<size_t>(preAvgFrames_), static_cast<size_t>(postAvgFrames_),
            static_cast<size_t>(waitFrames_)
        };
        for (size_t f = static_cast<size_t>(primeFrames_); f < odf.size(); ++f)
        {
            if (picker.accept(f, odf.size(), [&](size_t i) { return odf[i]; }, windows, delta))
                out.push_back(Onset { offRef_[f], odf[f] });
        }

        offOdf_.clear();
        offRef_.clear();
        resetState();
        return out;
    }

    /** @brief Clears all streaming state and abandons an open offline
     *         session. Not concurrent with pushSamples(). */
    void reset() noexcept
    {
        offlineOpen_ = false;
        resetState();
    }

private:
    [[nodiscard]] static std::vector<int64_t> positionsOf(const std::vector<Onset>& onsets)
    {
        std::vector<int64_t> out;
        out.reserve(onsets.size());
        for (const Onset& o : onsets) out.push_back(o.sample);
        return out;
    }

    // -- Constants -----------------------------------------------------------
    static constexpr int kMinFft = 64;
    static constexpr int kMaxFft = 1 << 16;
    /// Automatic-frame policy: the span 2048 samples cover at 48 kHz, resolved
    /// inside [512, 16384] (covers 8 kHz .. 384 kHz without hitting a clamp).
    static constexpr int64_t kBig = int64_t(1) << 60;
    static constexpr T kDefaultDelta = T(0.03);
    static constexpr double kLocalizationLead = 0.34; ///< Flux-to-energy lead (fraction of N).

    struct PendingOnset
    {
        int64_t referenceSample = 0; ///< Frame-centre reference (localisation).
        int64_t reportSample = 0;    ///< reference + L (latch release point).
        T strength = T(0);
    };

    // -- Frame timing --------------------------------------------------------

    /** @brief Reference sample (frame centre) of the frame that fires when
     *  totalSamples == firePos. The window covers [firePos-fftSize, firePos-1]. */
    [[nodiscard]] int64_t referenceSample(int64_t firePos) const noexcept
    {
        return firePos - static_cast<int64_t>(fftSize_) / 2
             + static_cast<int64_t>(localizationOffset_);
    }

    [[nodiscard]] int msToFrames(double ms) const noexcept
    {
        return std::max(1, static_cast<int>(std::lround(ms * 0.001 * sampleRate_
                                                        / static_cast<double>(hop_))));
    }

    // -- STFT + ODF ----------------------------------------------------------

    /** @brief Evaluates the shared spectral and novelty kernels for one frame. */
    T computeOdf(Method method, bool whiten) noexcept
    {
        spectrum_.compute(&ring_[static_cast<size_t>(writePos_)]);
        using FeatureMethod = typename detail::OnsetNovelty<T>::Method;
        const auto selected = method == Method::SpectralFlux ? FeatureMethod::SpectralFlux
                            : method == Method::ComplexDomain ? FeatureMethod::ComplexDomain
                            : method == Method::SuperFlux ? FeatureMethod::SuperFlux
                            : static_cast<FeatureMethod>(-1);
        const auto result = novelty_.process(spectrum_.magnitudes(), spectrum_.phases(),
            selected, whiten);
        curRegisters_ = result.registers;
        return result.value;
    }

    /** @brief One causal analysis frame: ODF + online peak-pick + scheduling. */
    void analyzeFrame() noexcept
    {
        const Method m = method_.load(std::memory_order_relaxed);
        const bool whiten = whitening_.load(std::memory_order_relaxed);
        const T value = computeOdf(m, whiten);

        // Push into the causal ODF history ring.
        odfHist_[static_cast<size_t>(odfWrite_)] = value;
        odfWrite_ = (odfWrite_ + 1) % odfHistLen_;
        ++frameIndex_;

        // Envelope readout for the stream owner (see getLastOdfFrame()). The
        // frame fired at totalSamples_, so it localises exactly where an onset
        // decided from it would: one definition of frame time, not two.
        lastOdfValue_ = value;
        lastOdfRef_ = referenceSample(totalSamples_);
        lastRegisters_ = curRegisters_;

        // Causal peak-pick with a single-frame confirmation: we decide whether
        // the PREVIOUS frame was a maximum now that we have the current one.
        // This one-hop confirmation is exactly the +hop term of L.
        if (frameIndex_ < 2) { lastConfirmOdf_ = value; return; }

        const T prev = odfAt(1);      // previous frame's ODF (candidate)
        const T curr = value;         // current frame (confirmation)

        // (1) prev is a causal local max over [prev-preMax, prev+1].
        bool isMax = (prev >= curr);
        if (isMax)
        {
            for (int j = 2; j <= preMaxFrames_ + 1 && j < frameIndex_; ++j)
            {
                if (odfAt(j) > prev) { isMax = false; break; }
            }
        }

        if (isMax)
        {
            // (2) prev exceeds the backward moving mean + delta.
            T sum = T(0); int cnt = 0;
            for (int j = 1; j <= preAvgFrames_ && j < frameIndex_; ++j)
            {
                sum += odfAt(j); ++cnt;
            }
            const T mean = (cnt > 0) ? sum / static_cast<T>(cnt) : T(0);
            const T delta = threshold_.load(std::memory_order_relaxed);

            // (3) combination width since the last onset, and past warm-up.
            const int64_t candFrame = frameIndex_ - 1;
            if (candFrame >= primeFrames_ && prev >= mean + delta
                && candFrame - lastOnsetFrame_ > waitFrames_)
            {
                // The candidate is the previous frame; its fire position was
                // totalSamples_ - hop (one hop before the current frame's fire).
                const int64_t candFirePos = totalSamples_ - hop_;
                const int64_t ref = referenceSample(candFirePos);
                scheduleOnset(ref, prev);
                lastOnsetFrame_ = candFrame;
            }
        }
    }

    /** @brief ODF value j frames back (j>=1) from the most recent write. */
    [[nodiscard]] T odfAt(int j) const noexcept
    {
        int idx = odfWrite_ - 1 - j;
        idx %= odfHistLen_;
        if (idx < 0) idx += odfHistLen_;
        return odfHist_[static_cast<size_t>(idx)];
    }

    void scheduleOnset(int64_t referenceSample, T strength) noexcept
    {
        if (pendingCount_ >= static_cast<int>(pending_.size())) return; // saturate
        const int tail = (pendingHead_ + pendingCount_) % static_cast<int>(pending_.size());
        PendingOnset& p = pending_[static_cast<size_t>(tail)];
        p.referenceSample = referenceSample;
        p.reportSample = referenceSample + latencySamples_.load(std::memory_order_relaxed);
        p.strength = strength;
        ++pendingCount_;
    }

    void resetState() noexcept
    {
        std::fill(ring_.begin(), ring_.end(), T(0));
        novelty_.reset();
        std::fill(odfHist_.begin(), odfHist_.end(), T(0));

        writePos_ = 0;
        hopCounter_ = 0;
        totalSamples_ = 0;
        frameIndex_ = 0;
        odfWrite_ = 0;
        lastOdfValue_ = T(0);
        lastOdfRef_ = 0;
        lastRegisters_.fill(T(0));
        curRegisters_.fill(T(0));
        lastConfirmOdf_ = T(0);
        lastOnsetFrame_ = -kBig;
        pendingHead_ = 0;
        pendingCount_ = 0;

        onsetLatched_.store(false, std::memory_order_relaxed);
        onsetStrength_.store(T(0), std::memory_order_relaxed);
        lastOnsetSample_.store(-1, std::memory_order_relaxed);
    }

    // -- Members -------------------------------------------------------------
    double sampleRate_ = 44100.0;
    int fftSize_ = 2048;
    int hop_ = 221;
    int localizationOffset_ = 0;
    int primeFrames_ = 12;
    bool prepared_ = false;
    detail::OnsetSpectrum<T> spectrum_;
    detail::OnsetNovelty<T> novelty_;
    std::vector<T> ring_; // Mirrored chronological input, size 2*fftSize.

    // Peak-picker windows (frames).
    int preMaxFrames_ = 6, postMaxFrames_ = 6;
    int preAvgFrames_ = 20, postAvgFrames_ = 14;
    int waitFrames_ = 6;

    // Causal ODF history ring.
    std::vector<T> odfHist_;
    int odfHistLen_ = 32;
    int odfWrite_ = 0;
    int64_t frameIndex_ = 0;
    T lastOdfValue_ = T(0);   ///< Envelope readout (stream owner only).
    int64_t lastOdfRef_ = 0;  ///< Reference sample of lastOdfValue_'s frame.
    std::array<T, kNumRegisters> lastRegisters_ {};  ///< Per-register readout.
    std::array<T, kNumRegisters> curRegisters_ {};   ///< This frame's registers.
    T lastConfirmOdf_ = T(0);
    int64_t lastOnsetFrame_ = -kBig;

    // Streaming counters.
    int writePos_ = 0;
    int hopCounter_ = 0;
    int64_t totalSamples_ = 0;

    // Pending-onset ring (held from detection to report point).
    std::vector<PendingOnset> pending_;
    int pendingHead_ = 0;
    int pendingCount_ = 0;

    // Atomic parameters / readouts.
    std::atomic<Method> method_ { Method::SuperFlux };
    std::atomic<T> threshold_ { kDefaultDelta };
    std::atomic<bool> whitening_ { false };
    std::atomic<bool> onsetLatched_ { false };
    std::atomic<T> onsetStrength_ { T(0) };
    std::atomic<int64_t> lastOnsetSample_ { -1 };
    std::atomic<int64_t> latencySamples_ { 2269 };

    // Offline session (beginOffline() .. finishOffline()).
    std::vector<T> offOdf_;
    std::vector<int64_t> offRef_;
    int64_t offTotal_ = 0;
    int offHop_ = 0;
    Method offMethod_ = Method::SuperFlux;
    bool offWhiten_ = false;
    bool offlineOpen_ = false;
};

} // namespace dspark
