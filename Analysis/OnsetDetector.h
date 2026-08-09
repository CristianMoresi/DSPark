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
 * Boeck, Krebs & Schedl (ISMIR 2012) -- the same recipe used by
 * librosa.util.peak_pick. Two simpler ODFs are also provided:
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
 * - detectOffline(): offline convenience (allocates); not an audio-thread call.
 *
 * Embedded/wasm: compiles under -fno-exceptions -fno-rtti (no throw on any
 * path); no file I/O, so it is unaffected by DSPARK_NO_FILE_IO.
 *
 * Dependencies: DspMath.h, FFT.h, WindowFunctions.h, AudioBuffer.h, AudioSpec.h.
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"

#include <algorithm>
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

        fft_.reset(); // gate OFF: the audio path is a no-op while rebuilding

        sampleRate_ = spec.sampleRate;

        if (fftSize <= 0)
        {
            // Automatic: hold the analysis TIME SPAN constant across sample
            // rates (the span 2048 samples cover at 48 kHz). Bin width and the
            // filterbank's quarter-tone floor are both fs/fftSize, so a
            // constant span pins them -- and with them the usable low register
            // -- instead of letting them widen as fs rises.
            const double target = sampleRate_ * (kAutoSpanRef / kAutoSpanRate);
            int n = kAutoMinFft;
            while (n < kAutoMaxFft && static_cast<double>(n) < target) n <<= 1;
            fftSize_ = n;
        }
        else
        {
            int fs = std::clamp(fftSize, kMinFft, kMaxFft);
            int pow2 = kMinFft;
            while (pow2 < fs) pow2 <<= 1;
            fftSize_ = pow2;
        }
        numBins_ = fftSize_ / 2 + 1;

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
        // ODF scale invariance: cancel the linear growth of |X| with the
        // frame length so the peak-pick delta means the same thing at every
        // rate. Exactly 1.0 at the 2048-sample reference; an exact power of
        // two at every other frame (fftSize is a power of two).
        odfScale_ = static_cast<T>(kOdfRefFrame / static_cast<double>(fftSize_));
        // Warm-up: suppress onsets until the analysis ring is fully primed so
        // the silence->first-input ramp cannot fire a spurious onset.
        primeFrames_ = fftSize_ / hop_ + 2;

        // Analysis window (periodic Hann) and its ring (mirrored for a
        // contiguous most-recent-fftSize window without a modulo).
        window_.assign(static_cast<size_t>(fftSize_), T(0));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        ring_.assign(static_cast<size_t>(fftSize_) * 2, T(0));

        fft_time_.assign(static_cast<size_t>(fftSize_), T(0));
        fft_spec_.assign(static_cast<size_t>(fftSize_) + 2, T(0));
        mag_.assign(static_cast<size_t>(numBins_), T(0));
        phase_.assign(static_cast<size_t>(numBins_), T(0));
        prevPhase_.assign(static_cast<size_t>(numBins_), T(0));
        prevPhase2_.assign(static_cast<size_t>(numBins_), T(0));
        prevMag_.assign(static_cast<size_t>(numBins_), T(0));
        whitenPeak_.assign(static_cast<size_t>(numBins_), T(0));

        buildFilterBank();

        // Band history for the SuperFlux flux-to-mu-th-previous-frame. mu is
        // one hop by construction (adjacent frames at 200 fps), so two frames
        // of history are enough; keep a small ring keyed on frame index.
        muFrames_ = 1;
        bandCur_.assign(static_cast<size_t>(numBands_), T(0));
        bandPrev_.assign(static_cast<size_t>(numBands_), T(0));
        bandMaxPrev_.assign(static_cast<size_t>(numBands_), T(0));

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

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize_)); // gate ON
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

    // -- Audio path (causal, RT-safe) ---------------------------------------

    /**
     * @brief Feeds a mono block; reads channel 0 only. Const, never mutated.
     *
     * Lock-free and allocation-free. Safe no-op before prepare().
     */
    void processBlock(AudioBufferView<const T> in) noexcept
    {
        if (fft_ == nullptr || in.getNumChannels() < 1) return;
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
        if (fft_ == nullptr) return;

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
    [[nodiscard]] int getNumBands() const noexcept { return numBands_; }

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
    struct OdfFrame
    {
        T value = T(0);              ///< ODF value of the most recent frame.
        int64_t referenceSample = 0; ///< Sample index the frame localises to.
        int64_t frameIndex = 0;      ///< Frames computed since the last reset.
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
        return OdfFrame { lastOdfValue_, lastOdfRef_, frameIndex_ };
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

    // -- Offline convenience -------------------------------------------------

    /**
     * @brief Offline detection over a whole mono buffer (channel 0).
     *
     * Runs the same ODF with the symmetric (post_max/post_avg > 0) picker for
     * slightly higher F, and returns onset sample positions (frame-centre
     * references, ascending). Allocates -- not an audio-thread call. Resets
     * the streaming state on entry.
     */
    std::vector<int64_t> detectOffline(AudioBufferView<const T> whole)
    {
        std::vector<int64_t> out;
        if (fft_ == nullptr || whole.getNumChannels() < 1) return out;

        const int n = whole.getNumSamples();
        const T* x = whole.getChannel(0);

        // Build the full ODF envelope (offline: allocation allowed).
        std::vector<T> odf;
        std::vector<int64_t> odfRef; // reference sample per frame
        odf.reserve(static_cast<size_t>(n / std::max(1, hop_) + 2));
        odfRef.reserve(odf.capacity());

        resetState();
        const Method m = method_.load(std::memory_order_relaxed);
        const bool whiten = whitening_.load(std::memory_order_relaxed);

        int64_t total = 0;
        int hopc = 0;
        for (int i = 0; i < n; ++i)
        {
            const T v = std::isfinite(x[i]) ? x[i] : T(0);
            ring_[static_cast<size_t>(writePos_)] = v;
            ring_[static_cast<size_t>(writePos_ + fftSize_)] = v;
            if (++writePos_ >= fftSize_) writePos_ = 0;
            ++total;
            if (++hopc >= hop_)
            {
                hopc = 0;
                const T value = computeOdf(m, whiten);
                odf.push_back(value);
                odfRef.push_back(referenceSample(total));
            }
        }

        // Symmetric peak-pick over the whole envelope.
        const T delta = threshold_.load(std::memory_order_relaxed);
        const int nf = static_cast<int>(odf.size());
        int64_t lastFrame = -kBig;
        for (int f = 0; f < nf; ++f)
        {
            if (f < primeFrames_) continue; // warm-up guard (ring priming)
            bool isMax = true;
            for (int j = f - preMaxFrames_; j <= f + postMaxFrames_; ++j)
            {
                if (j < 0 || j >= nf) continue;
                if (odf[static_cast<size_t>(j)] > odf[static_cast<size_t>(f)]) { isMax = false; break; }
            }
            if (!isMax) continue;

            T sum = T(0); int cnt = 0;
            for (int j = f - preAvgFrames_; j <= f + postAvgFrames_; ++j)
            {
                if (j < 0 || j >= nf) continue;
                sum += odf[static_cast<size_t>(j)]; ++cnt;
            }
            const T mean = (cnt > 0) ? sum / static_cast<T>(cnt) : T(0);
            if (odf[static_cast<size_t>(f)] < mean + delta) continue;
            if (f - lastFrame <= waitFrames_) continue;

            out.push_back(odfRef[static_cast<size_t>(f)]);
            lastFrame = f;
        }

        resetState();
        return out;
    }

    /** @brief Clears all streaming state. Not concurrent with pushSamples(). */
    void reset() noexcept { resetState(); }

private:
    // -- Constants -----------------------------------------------------------
    static constexpr int kMinFft = 64;
    static constexpr int kMaxFft = 1 << 16;
    /// Automatic-frame policy: the span 2048 samples cover at 48 kHz, resolved
    /// inside [512, 16384] (covers 8 kHz .. 384 kHz without hitting a clamp).
    static constexpr double kAutoSpanRef = 2048.0;
    static constexpr double kAutoSpanRate = 48000.0;
    static constexpr int kAutoMinFft = 512;
    static constexpr int kAutoMaxFft = 16384;
    static constexpr int64_t kBig = int64_t(1) << 60;
    static constexpr T kDefaultDelta = T(0.03);
    /// ODF magnitude reference frame. Un-normalised |X| grows linearly with
    /// the frame length, so the SuperFlux filterbank accumulation is scaled
    /// by kOdfRefFrame / fftSize before the log10(x + 1) compression: the
    /// growth cancels and a given peak-pick delta selects the same
    /// quiet-onset sensitivity at every frame length (and therefore at every
    /// rate under the automatic frame). 2048 is the frame kDefaultDelta was
    /// tuned at, so the factor is exactly 1 at the 44.1/48 kHz reference --
    /// and because fftSize is always a power of two the factor is an exact
    /// power of two everywhere: the scaling introduces no rounding at all.
    /// The compensation must sit on the magnitude (inside the log), not on
    /// the delta: log10(x + 1) is not affine, so no post-log delta rescale
    /// could keep the reference behaviour unchanged.
    static constexpr double kOdfRefFrame = 2048.0;
    static constexpr double kLocalizationLead = 0.34; ///< Flux-to-energy lead (fraction of N).
    static constexpr double kFMin = 27.5;      ///< Filterbank low edge (Hz).
    static constexpr double kFMaxHz = 16000.0; ///< Filterbank high edge (Hz).
    static constexpr int kBandsPerOctave = 24; ///< Quarter-tone resolution.

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

    /** @brief Windows the current most-recent-fftSize ring window, FFTs it,
     *  and fills mag_/phase_. */
    void computeSpectrum() noexcept
    {
        const T* w = window_.data();
        const T* r = &ring_[static_cast<size_t>(writePos_)]; // oldest..newest, contiguous
        for (int k = 0; k < fftSize_; ++k)
            fft_time_[static_cast<size_t>(k)] = r[k] * w[k];

        fft_->forward(fft_time_.data(), fft_spec_.data());

        for (int k = 0; k < numBins_; ++k)
        {
            const T re = fft_spec_[static_cast<size_t>(2 * k)];
            const T im = fft_spec_[static_cast<size_t>(2 * k + 1)];
            mag_[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
            phase_[static_cast<size_t>(k)] = std::atan2(im, re);
        }
    }

    /** @brief Applies per-bin adaptive whitening (Stowell-Plumbley) to mag_. */
    void applyWhitening() noexcept
    {
        for (int k = 0; k < numBins_; ++k)
        {
            T& pk = whitenPeak_[static_cast<size_t>(k)];
            const T decayed = pk * kWhitenDecay;
            const T m = mag_[static_cast<size_t>(k)];
            pk = std::max({ m, kWhitenFloor, decayed });
            mag_[static_cast<size_t>(k)] = m / pk;
        }
    }

    /** @brief Computes the ODF value for the current frame and rotates the
     *  per-frame history buffers. Assumes computeSpectrum() was called. */
    T computeOdf(Method m, bool whiten) noexcept
    {
        computeSpectrum();
        if (whiten) applyWhitening();

        T odf = T(0);
        switch (m)
        {
            case Method::SpectralFlux:
            {
                for (int k = 0; k < numBins_; ++k)
                {
                    const T d = mag_[static_cast<size_t>(k)] - prevMag_[static_cast<size_t>(k)];
                    if (d > T(0)) odf += d;
                }
                odf /= static_cast<T>(numBins_);
                break;
            }
            case Method::ComplexDomain:
            {
                // Rectified complex-domain deviation (Dixon 2006): phase-predict
                // each bin, sum |X - Xhat| where magnitude increased.
                for (int k = 0; k < numBins_; ++k)
                {
                    const T target = princArg(T(2) * prevPhase_[static_cast<size_t>(k)]
                                              - prevPhase2_[static_cast<size_t>(k)]);
                    const T pm = prevMag_[static_cast<size_t>(k)];
                    const T cm = mag_[static_cast<size_t>(k)];
                    const T re = cm * std::cos(phase_[static_cast<size_t>(k)])
                               - pm * std::cos(target);
                    const T im = cm * std::sin(phase_[static_cast<size_t>(k)])
                               - pm * std::sin(target);
                    if (cm >= pm) odf += std::sqrt(re * re + im * im);
                }
                odf /= static_cast<T>(numBins_);
                break;
            }
            case Method::SuperFlux:
            {
                // Log-filtered magnitude bands, flux to the mu-th previous
                // frame after a frequency maximum filter on the reference.
                // The frame-invariant magnitude scale (kOdfRefFrame/fftSize)
                // applies to the raw spectrum only: adaptive whitening
                // already divides each bin by its running peak, which
                // carries the same linear-in-N growth, so above kWhitenFloor
                // the whitened spectrum is dimensionless and scaling it
                // again would INVERT the rate dependence instead of removing
                // it. Below the floor the divisor is the absolute constant
                // kWhitenFloor, so those bins keep the linear-in-N growth --
                // the residual rate dependence documented at prepare().
                filterLogBands(bandCur_, whiten ? T(1) : odfScale_);
                for (int b = 0; b < numBands_; ++b)
                {
                    const T d = bandCur_[static_cast<size_t>(b)]
                              - bandMaxPrev_[static_cast<size_t>(b)];
                    if (d > T(0)) odf += d;
                }
                odf /= static_cast<T>(numBands_);
                // Rotate: previous <- current, and rebuild the max-filtered
                // reference from the (new) previous frame.
                bandPrev_ = bandCur_;
                maxFilterFreq(bandPrev_, bandMaxPrev_);
                break;
            }
        }

        // Rotate per-bin history (prevPhase2_ <- prevPhase_ <- phase_) and the
        // previous magnitude, used by SpectralFlux/ComplexDomain next frame.
        rotatePhaseHistory();
        std::copy(mag_.begin(), mag_.end(), prevMag_.begin());

        return odf;
    }

    /** @brief prevPhase2_ <- prevPhase_ <- phase_ (correct 2-frame ring). */
    void rotatePhaseHistory() noexcept
    {
        std::copy(prevPhase_.begin(), prevPhase_.end(), prevPhase2_.begin());
        std::copy(phase_.begin(), phase_.end(), prevPhase_.begin());
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

    // -- Filterbank ----------------------------------------------------------

    /** @brief Builds the log-frequency triangular filterbank (quarter-tone,
     *  peak-normalised, not area-normalised). Filter count depends on
     *  fftSize/fs; ~138 at 44.1 kHz / 2048. */
    void buildFilterBank()
    {
        fbStart_.clear();
        fbWeights_.clear();
        fbOffset_.clear();

        const double binHz = sampleRate_ / static_cast<double>(fftSize_);
        const double fMax = std::min(kFMaxHz, sampleRate_ * 0.5 * 0.999);

        // Quarter-tone centre bins, strictly increasing and unique.
        std::vector<int> centres;
        for (int i = 0; ; ++i)
        {
            const double f = kFMin * std::pow(2.0, static_cast<double>(i)
                                              / static_cast<double>(kBandsPerOctave));
            if (f > fMax) break;
            int bin = static_cast<int>(std::lround(f / binHz));
            bin = std::clamp(bin, 0, numBins_ - 1);
            if (centres.empty() || bin > centres.back())
                centres.push_back(bin);
        }

        // Triangular filters over consecutive triples (b[j-1], b[j], b[j+1]).
        numBands_ = 0;
        for (size_t j = 1; j + 1 < centres.size(); ++j)
        {
            const int lo = centres[j - 1];
            const int ce = centres[j];
            const int hi = centres[j + 1];
            if (!(lo < ce && ce < hi)) continue;

            fbStart_.push_back(lo);
            fbOffset_.push_back(static_cast<int>(fbWeights_.size()));
            for (int k = lo; k <= hi; ++k)
            {
                T wv;
                if (k <= ce)
                    wv = static_cast<T>(static_cast<double>(k - lo)
                                        / static_cast<double>(ce - lo));
                else
                    wv = static_cast<T>(static_cast<double>(hi - k)
                                        / static_cast<double>(hi - ce));
                fbWeights_.push_back(wv);
            }
            ++numBands_;
        }
        fbCount_.clear();
        for (int b = 0; b < numBands_; ++b)
        {
            const int off = fbOffset_[static_cast<size_t>(b)];
            const int nextOff = (b + 1 < numBands_)
                                ? fbOffset_[static_cast<size_t>(b + 1)]
                                : static_cast<int>(fbWeights_.size());
            fbCount_.push_back(nextOff - off);
        }
        if (numBands_ < 1) numBands_ = 1; // degenerate guard (tiny fftSize)
    }

    /** @brief Applies the filterbank to mag_ and takes log10(scale*x + 1)
     *  per band. @p scale is the frame-invariance factor kOdfRefFrame /
     *  fftSize (see the constant), or exactly 1 under adaptive whitening;
     *  scaling the accumulated band is identical to scaling the spectrum
     *  before the filterbank (the filterbank is linear) and touches the
     *  other ODF families not at all. */
    void filterLogBands(std::vector<T>& out, T scale) noexcept
    {
        for (int b = 0; b < numBands_ && b < static_cast<int>(fbStart_.size()); ++b)
        {
            const int start = fbStart_[static_cast<size_t>(b)];
            const int off = fbOffset_[static_cast<size_t>(b)];
            const int cnt = fbCount_[static_cast<size_t>(b)];
            T acc = T(0);
            for (int i = 0; i < cnt; ++i)
            {
                const int k = start + i;
                if (k >= 0 && k < numBins_)
                    acc += mag_[static_cast<size_t>(k)]
                         * fbWeights_[static_cast<size_t>(off + i)];
            }
            out[static_cast<size_t>(b)] = std::log10(acc * scale + T(1));
        }
    }

    /** @brief 3-neighbour frequency maximum filter (SuperFlux vibrato guard). */
    void maxFilterFreq(const std::vector<T>& in, std::vector<T>& out) const noexcept
    {
        for (int b = 0; b < numBands_; ++b)
        {
            T mx = in[static_cast<size_t>(b)];
            if (b > 0) mx = std::max(mx, in[static_cast<size_t>(b - 1)]);
            if (b + 1 < numBands_) mx = std::max(mx, in[static_cast<size_t>(b + 1)]);
            out[static_cast<size_t>(b)] = mx;
        }
    }

    static T princArg(T x) noexcept
    {
        // Wrap to (-pi, pi].
        const T twoPiT = twoPi<T>;
        T y = x - twoPiT * std::floor(x / twoPiT + T(0.5));
        return y;
    }

    void resetState() noexcept
    {
        std::fill(ring_.begin(), ring_.end(), T(0));
        std::fill(prevMag_.begin(), prevMag_.end(), T(0));
        std::fill(prevPhase_.begin(), prevPhase_.end(), T(0));
        std::fill(prevPhase2_.begin(), prevPhase2_.end(), T(0));
        std::fill(whitenPeak_.begin(), whitenPeak_.end(), T(0));
        std::fill(bandPrev_.begin(), bandPrev_.end(), T(0));
        std::fill(bandMaxPrev_.begin(), bandMaxPrev_.end(), T(0));
        std::fill(odfHist_.begin(), odfHist_.end(), T(0));

        writePos_ = 0;
        hopCounter_ = 0;
        totalSamples_ = 0;
        frameIndex_ = 0;
        odfWrite_ = 0;
        lastOdfValue_ = T(0);
        lastOdfRef_ = 0;
        lastConfirmOdf_ = T(0);
        lastOnsetFrame_ = -kBig;
        pendingHead_ = 0;
        pendingCount_ = 0;

        onsetLatched_.store(false, std::memory_order_relaxed);
        onsetStrength_.store(T(0), std::memory_order_relaxed);
        lastOnsetSample_.store(-1, std::memory_order_relaxed);
    }

    // -- Whitening constants -------------------------------------------------
    static constexpr T kWhitenDecay = T(0.9995);
    static constexpr T kWhitenFloor = T(1e-4);

    // -- Members -------------------------------------------------------------
    double sampleRate_ = 44100.0;
    int fftSize_ = 2048;
    int numBins_ = 1025;
    int hop_ = 221;
    int numBands_ = 1;
    int muFrames_ = 1;
    int localizationOffset_ = 0;
    int primeFrames_ = 12;
    T odfScale_ = T(1); ///< kOdfRefFrame / fftSize (ODF frame invariance).

    std::unique_ptr<FFTReal<T>> fft_; // doubles as the "prepared" gate

    std::vector<T> window_;
    std::vector<T> ring_;      // size 2*fftSize
    std::vector<T> fft_time_;  // size fftSize
    std::vector<T> fft_spec_;  // size fftSize+2
    std::vector<T> mag_;       // numBins
    std::vector<T> phase_;     // numBins
    std::vector<T> prevPhase_, prevPhase2_, prevMag_, whitenPeak_;

    // Filterbank (CSR-style: start bin, flat weights, per-band offset/count).
    std::vector<int> fbStart_;
    std::vector<T> fbWeights_;
    std::vector<int> fbOffset_;
    std::vector<int> fbCount_;

    std::vector<T> bandCur_, bandPrev_, bandMaxPrev_;

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
};

} // namespace dspark
