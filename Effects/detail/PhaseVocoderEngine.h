// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PhaseVocoderEngine.h
 * @brief Shared phase-vocoder analysis/synthesis engine (internal detail).
 *
 * The identity-phase-locked phase vocoder used by Effects/PitchShifter.h
 * (engine + resample-back) and by the time-stretching effects (engine only).
 * This is an INTERNAL header in namespace dspark::detail: its API is a
 * contract with the owning effect classes, not with library users, and it is
 * not part of the umbrella header.
 *
 * What it owns, per hop (streaming, zero allocation after prepare()):
 *
 *   input ring -> analysis hop Ra (variable, fractional-accumulator exact)
 *     -> FFT -> peak picking & phase propagation (reference channel)
 *     -> per-channel rigid phase rotation per region
 *     -> optional spectral post-stages for resample-back owners
 *        (cepstral formant pre-warp, above-Nyquist/ratio anti-alias taper)
 *     -> IFFT -> overlap-add at fixed synthesis hop Rs = N/4 (exact COLA)
 *     -> OLA accumulator ring, read by the owner
 *
 * Techniques (implemented from the primary papers, no third-party code):
 *
 * - **Identity phase locking** (Laroche & Dolson 1999): spectral peaks are
 *   detected each frame and every bin in a peak's region of influence is
 *   rotated by the *same* phase increment as its peak, preserving vertical
 *   phase coherence. Switching it off (params.phaseLock) leaves the plain
 *   phase vocoder, where every bin propagates its own instantaneous
 *   frequency independently - the control condition phase locking is meant
 *   to beat, kept in the engine so an owner can expose the comparison.
 * - **Harmonic/percussive split** (opt-in, params.percussiveSplit; enabled
 *   at prepare()): median filtering of the magnitude spectrogram along time
 *   enhances sustained partials, median filtering along frequency enhances
 *   broadband strikes (Fitzgerald, "Harmonic/percussive separation using
 *   median filtering", DAFx 2010). The two medians drive complementary
 *   Wiener-form masks; the harmonic share is phase-propagated as above and
 *   the percussive share keeps its analysis phase, so it overlap-adds at the
 *   synthesis hop instead of being phase-propagated (Driedger, Mueller &
 *   Ewert, "Improving time-scale modification of music signals using
 *   harmonic-percussive separation", IEEE Signal Processing Letters 21(1),
 *   2014). Both paths share one analysis frame, so the split adds no
 *   transform - but it is NOT cheap: the two medians measure 4.85x to 5.24x
 *   the cost of the whole rest of the engine (stereo, 2048-sample frame,
 *   48 kHz, 512-sample blocks, 42.7 s of audio, best of five runs), because
 *   a median runs per bin per frame while the transform runs once per frame.
 * - **Onset detection**, selected by the owner at prepare(). Two detectors
 *   live here because the two owners want different things from one, and
 *   neither answer is right for both:
 *   - the *broadband frame-energy* test (the default): a frame fires when
 *     its total magnitude-squared rises 6 dB over an envelope that decays
 *     at 0.7 per frame. It is deliberately blunt, and an owner whose
 *     rendering is already published stays on it because its firings are
 *     part of that rendering;
 *   - *half-wave-rectified spectral flux* over a log-frequency triangular
 *     filterbank (quarter-tone spacing, 27.5 Hz to 16 kHz), compressed as
 *     log10(x + 1), differenced against a three-neighbour frequency maximum
 *     filter of the previous frame so that vibrato does not read as an
 *     attack (Boeck & Widmer, "Maximum filter vibrato suppression for onset
 *     detection", DAFx-13). A frame fires when that flux rises a fixed
 *     amount above the running median of the recent flux history.
 *   The energy test cannot see a strike over a sustained bed at all: on a
 *   strike over a 110 Hz harmonic bed the weakest onset frame carries
 *   0.13 dB more energy than the median non-onset frame, against the 6 dB
 *   that test needs, where the flux carries 14.46 dB. An owner that drives a
 *   hop schedule from onsets therefore asks for the flux detector; an owner
 *   that only resets phase on them, and whose users already have renders on
 *   disk, does not, because a more sensitive detector resets phase more
 *   often and that is audible on material with strikes over sustain.
 * - **Transient phase reset**: a firing frame re-initialises synthesis
 *   phases to the analysis phases. Peaks with no history (new partials) are
 *   reset individually even without a global onset.
 * - **Transient-locked analysis hop** (opt-in at prepare(); only an owner
 *   that reads the synthesis stream directly may ask for it, see prepare()):
 *   across a detected attack the analysis
 *   hop is forced to Ra = Rs for ceil(fftSize / Rs) frames - one whole
 *   window - so every frame whose window contains the strike places it at
 *   the same offset and the positional spread fftSize * |ratio - 1| that
 *   doubles an attack is identically zero. The input the lock does not
 *   consume is carried as a debt and repaid by widening later hops, capped
 *   at Rs/2 per hop. Three rules keep the debt self-limiting rather than
 *   cumulative, and each is load-bearing: a firing inside an engaged lock is
 *   ignored (one strike, one lock), a lock does not arm until the previous
 *   lock's debt is fully repaid, and no lock arms on the first frame of a
 *   stream. With them max|debt| = N * Rs * |1 - ratio| / ratio exactly, and
 *   independently of how dense the strikes are; without the arming gate the
 *   debt accumulates across strikes instead of being cleared between them.
 * - **Exact fractional analysis hop**: the analysis hop carries a fractional
 *   accumulator so the average stretch is exactly Rs / (Rs / ratio) = ratio
 *   for arbitrary ratios, with no cumulative drift.
 * - Channels share the reference channel's peak/phase decisions (rigid
 *   per-region rotation), which preserves inter-channel phase differences
 *   exactly.
 *
 * Ownership split: the ENGINE owns the analysis input rings, the spectral
 * state and the OLA accumulator rings; the OWNER owns the output stage (the
 * fractional resampling reader in PitchShifter, the plain hop reader in a
 * time-stretcher), drives the chunk loop, and reads the OLA ring through the
 * accessors below.
 *
 * Threading (SPSC model, see docs/threading.md; the role split below holds
 * from construction on, not only after prepare()):
 * - prepare(): setup thread only (allocates; never concurrent with any other
 *   call).
 * - publishParams(): CONTROL thread only (ONE non-audio writer; canonical
 *   seqlock publish of the multi-word parameter set).
 * - pushInput() / commitInput() / samplesToNextHop() / activeRatio() /
 *   olaData() / olaMask() / olaSize() / writeHead(): AUDIO thread only (the
 *   single stream owner).
 * - reset(): stream owner only (writes the plain signal state).
 *
 * Cross-thread publication (the one and only handoff in this class):
 * publishParams() moves the multi-word set {targetSemitones,
 * transientPreserve, formantPreserve, phaseLock, percussiveSplit} from the
 * control thread to the audio
 * thread through a canonical seqlock: counter to odd (acq_rel), a RELEASE
 * FENCE, relaxed stores of the three std::atomic data words, counter to even
 * (release), dirty flag (release). The release fence is mandatory: an
 * acquire/release counter alone does not order the relaxed word stores
 * against it, so without it a reader could observe a mid-publish word while
 * the counter still looked even ([atomics.fences]/2; Boehm, "Can Seqlocks
 * Get Along With Programming Language Memory Models?", MSPC 2012). The
 * audio-thread reader (adoptParamsIfDirty) is BOUNDED: at most
 * kSeqlockMaxAttempts validation attempts, each of which tests the counter
 * for oddness BEFORE copying, copies the words into locals, issues the
 * pairing ACQUIRE FENCE, and accepts only if the counter re-reads unchanged.
 * The acquire fence keeps the word copy from sinking below the counter
 * re-read (an acquire LOAD orders only later accesses) and pairs with the
 * writer's release fence, so a copy that read any mid-publish word cannot
 * validate. Commit-on-validate: the locals are committed to the
 * audio-thread-private active copies only after validation, so a give-up
 * can never leave a torn set live. On give-up the reader adopts nothing,
 * keeps the set in use, and re-arms the dirty flag (release) so the update
 * lands on a later hop; the audio thread's worst case here is
 * kSeqlockMaxAttempts copies of three words - this class's own instruction
 * count - never the time the control thread holds the counter odd.
 * Bounding cannot make a torn set adoptable: the accept predicate is the
 * same as the unbounded form's, so the bound can only DEFER an adoption.
 *
 * Origin rule (see docs/threading.md): the staged channel above carries
 * CONTROL -> AUDIO publications only. Everything the audio thread computes
 * for its own use -
 * the glided active ratio, the fractional hop accumulator, the spectral
 * state - is stream-owner state written directly as plain members; the audio
 * thread never publishes to itself through the staged channel, so the
 * adoption above is cold by construction (it runs only when the control
 * thread actually published).
 *
 * Every buffer the audio path touches, including the harmonic/percussive
 * median history, is sized and allocated in prepare(); the hop path only
 * writes into what is already there. The median windows are chosen from the
 * physical spans below and therefore scale with the sample rate, so their
 * behaviour is the same at 44.1 kHz and at 96 kHz.
 *
 * Dependencies: Core/FFT.h, Core/WindowFunctions.h, Core/DspMath.h.
 */

#include "../../Core/DspMath.h"
#include "../../Core/FFT.h"
#include "../../Core/WindowFunctions.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {
namespace detail {

/**
 * @class PhaseVocoderEngine
 * @brief Identity-phase-locked STFT analysis/synthesis core (internal).
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PhaseVocoderEngine
{
public:
    // The staged parameter words are single machine words on every supported
    // target; the audio path must never take a lock to read them.
    static_assert(std::atomic<double>::is_always_lock_free,
                  "PhaseVocoderEngine requires lock-free std::atomic<double>");
    static_assert(std::atomic<unsigned>::is_always_lock_free,
                  "PhaseVocoderEngine requires lock-free std::atomic<unsigned>");
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "PhaseVocoderEngine requires lock-free std::atomic<bool>");

    /** @brief The control-published parameter set (adopted as ONE unit). */
    struct Params
    {
        double targetSemitones  = 0.0;    ///< Stretch/shift target, semitones, clamped +-12.
        bool   transientPreserve = true;  ///< Phase reset on detected transients.
        bool   formantPreserve   = false; ///< Cepstral envelope pre-warp (resample-back owners).
        bool   phaseLock         = true;  ///< Identity phase locking; false = plain vocoder.
        bool   percussiveSplit   = false; ///< Median-filter harmonic/percussive split.
    };

    // -- Lifecycle (setup thread) ---------------------------------------------

    /**
     * @brief Allocates all rings and spectral state.
     *
     * Invalid arguments (non-positive/non-finite rate, channel count < 1, or
     * an fftSize that is not a power of two in [256, 1 << 20]) are rejected:
     * nothing is touched and false is returned.
     *
     * @param sampleRate           Sample rate in Hz.
     * @param numChannels          Channel count (>= 1).
     * @param fftSize              STFT frame size, power of two in [256, 1<<20].
     * @param resampleCompensation True for owners that resample the synthesis
     *                             stream back by `ratio` (PitchShifter): the
     *                             engine then tapers bins that would land
     *                             above Nyquist after resampling, and the
     *                             formant pre-warp targets env(k*ratio).
     * @param separationSupport    True to allocate the harmonic/percussive
     *                             median history, which is what lets an owner
     *                             turn params.percussiveSplit on and off while
     *                             streaming without ever allocating again.
     *                             False leaves those buffers empty and the
     *                             split permanently off.
     * @param transientLockedHop   True to force the analysis hop to Rs across a
     *                             detected attack (see the file doc). Only an
     *                             owner that reads the synthesis stream
     *                             directly may ask for it. An owner that
     *                             resamples the synthesis stream back advances
     *                             its reader by `ratio` per input sample and so
     *                             assumes Ra = Rs/ratio every hop; a locked hop
     *                             breaks that assumption for the length of the
     *                             lock and the reader runs past the write head
     *                             into ring content no frame has written yet.
     *                             Measured on a click train, which is 84%
     *                             silence - 48 kHz, 10 s at 120 BPM, a
     *                             2048-sample frame, over -12 to +12
     *                             semitones: locking the hop under a
     *                             resample-back reader raises the output RMS by
     *                             0.8 to 5.9 dB, which is energy appearing in
     *                             the silence. How far it rises is a property
     *                             of that bed - a sparser one has more silence
     *                             to fill - so the range is a reading and not
     *                             a bound. The lock also buys such an owner
     *                             nothing, because resampling puts the strike
     *                             back where it started whatever the analysis
     *                             hop did.
     * @param spectralFluxDetector True to detect onsets by log-filterbank
     *                             spectral flux, false to keep the broadband
     *                             frame-energy test (see the file doc). It is
     *                             the owner's choice and not the engine's,
     *                             and it defaults to the energy test, because
     *                             the detector decides where phase is reset
     *                             and so it is part of an owner's rendering:
     *                             an owner whose output has already been
     *                             released cannot change detector without
     *                             changing renders its users already have.
     *                             Measured on strikes over a sustained bed,
     *                             switching an owner from the energy test to
     *                             the flux detector moves strike
     *                             concentration by up to -91.6% and adds up
     *                             to 16.5 dB of pre-echo. Only an owner that
     *                             needs onsets its schedule can act on -
     *                             which the energy test cannot supply over
     *                             sustained material - should ask for it, and
     *                             its own acceptance measurements then cover
     *                             it. False also leaves the filterbank
     *                             unallocated and the flux front end
     *                             permanently off.
     * @return true if the engine (re)allocated and is ready for reset().
     */
    bool prepare(double sampleRate, int numChannels, int fftSize,
                 bool resampleCompensation, bool separationSupport = false,
                 bool transientLockedHop = false,
                 bool spectralFluxDetector = false)
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate) || numChannels < 1
            || (fftSize & (fftSize - 1)) != 0 || fftSize < 256 || fftSize > (1 << 20))
            return false;

        prepared_ = false;

        sampleRate_  = sampleRate;
        numChannels_ = numChannels;
        fftSize_     = fftSize;
        numBins_     = fftSize_ / 2 + 1;
        synthHop_    = fftSize_ / 4;
        ringMask_    = fftSize_ - 1;
        resampleCompensation_ = resampleCompensation;

        accumSize_ = fftSize_ * 4;          // power of two: frame + read margin
        accumMask_ = accumSize_ - 1;

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize_));

        window_.resize(static_cast<size_t>(fftSize_));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        for (auto& w : window_) w = std::sqrt(w);   // sqrt-Hann analysis+synthesis

        const int nCh = numChannels_;
        inputRing_.assign(static_cast<size_t>(nCh), {});
        accum_.assign(static_cast<size_t>(nCh), {});
        for (int ch = 0; ch < nCh; ++ch)
        {
            inputRing_[static_cast<size_t>(ch)].assign(static_cast<size_t>(fftSize_), T(0));
            accum_[static_cast<size_t>(ch)].assign(static_cast<size_t>(accumSize_), T(0));
        }

        fftIn_.resize(static_cast<size_t>(fftSize_));
        spec_.resize(static_cast<size_t>(fftSize_ + 2));
        fftResult_.resize(static_cast<size_t>(fftSize_));
        prevAnalysis_.resize(static_cast<size_t>(fftSize_ + 2));
        prevSynth_.resize(static_cast<size_t>(fftSize_ + 2));

        // Formant preservation scratch (cepstral envelope).
        cepsTime_.resize(static_cast<size_t>(fftSize_));
        cepsSpec_.resize(static_cast<size_t>(fftSize_ + 2));
        envLog_.resize(static_cast<size_t>(numBins_));
        formantGain_.resize(static_cast<size_t>(numBins_));
        mag_.resize(static_cast<size_t>(numBins_));
        prevMag_.resize(static_cast<size_t>(numBins_));
        rotRe_.resize(static_cast<size_t>(numBins_));
        rotIm_.resize(static_cast<size_t>(numBins_));
        peakBin_.resize(static_cast<size_t>(numBins_ / 2 + 2));

        // Onset detection front end, sized only for the owner that asked for
        // it. Everything the flux detector needs per frame is allocated
        // here: the filterbank tables, the two band frames it differences,
        // and the flux history the running median is taken over. An owner on
        // the frame-energy test allocates none of it and runs the same
        // detector, over the same magnitudes, as it did before this
        // selection existed.
        fluxDetectorEnabled_ = spectralFluxDetector;
        if (fluxDetectorEnabled_)
        {
            buildOnsetFilterBank();
            bandCur_.assign(static_cast<size_t>(numBands_), T(0));
            bandPrev_.assign(static_cast<size_t>(numBands_), T(0));
            bandMaxPrev_.assign(static_cast<size_t>(numBands_), T(0));
            fluxHist_.assign(static_cast<size_t>(kFluxWindow), 0.0);
            fluxScratch_.assign(static_cast<size_t>(kFluxWindow), 0.0);
            // Un-normalised |X| grows linearly with the frame length, so the
            // band accumulation is referred to a 2048-sample frame before
            // the log compression: the growth cancels and one firing
            // threshold means the same sensitivity at every frame size.
            // fftSize is a power of two, so the factor is exact and
            // introduces no rounding.
            odfScale_ = static_cast<T>(kOdfRefFrame / static_cast<double>(fftSize_));
        }
        else
        {
            numBands_ = 0;
            fbStart_.clear();
            fbOffset_.clear();
            fbCount_.clear();
            fbWeights_.clear();
            bandCur_.clear();
            bandPrev_.clear();
            bandMaxPrev_.clear();
            fluxHist_.clear();
            fluxScratch_.clear();
            odfScale_ = T(1);
        }
        // One lock spans every analysis frame whose window can contain the
        // strike; that count is a property of the window geometry, not of
        // any signal. An owner that did not ask for the locked hop gets a
        // lock length of zero, which is the schedule with no lock in it.
        lockEnabled_ = transientLockedHop;
        lockFrames_ = lockEnabled_ ? (fftSize_ + synthHop_ - 1) / synthHop_ : 0;

        // Harmonic/percussive median filtering. Both window lengths are
        // chosen from a PHYSICAL span, so they follow the sample rate and the
        // frame size instead of being fixed bin/frame counts:
        //   - along time, 0.2 s of history, the span over which a musical
        //     partial is steady but a strike is not;
        //   - along frequency, 500 Hz, wide enough to bridge the harmonic
        //     comb of any pitched note in range and narrow enough to leave a
        //     broadband strike untouched.
        // Both are forced odd so the median is a sample of the window rather
        // than an average of two, and both are clamped so extreme frame sizes
        // cannot ask for a window longer than the data.
        separationEnabled_ = separationSupport;
        if (separationEnabled_)
        {
            const double hopSeconds = static_cast<double>(synthHop_) / sampleRate_;
            const double binHz = sampleRate_ / static_cast<double>(fftSize_);
            timeMedian_ = makeOdd(static_cast<int>(std::lround(0.2 / hopSeconds)), 3, 31);
            freqMedian_ = makeOdd(static_cast<int>(std::lround(500.0 / binHz)), 3, 63);
            freqMedian_ = std::min(freqMedian_, makeOdd(numBins_, 3, 63));

            magHist_.assign(static_cast<size_t>(timeMedian_) * static_cast<size_t>(numBins_), T(0));
            medianScratch_.resize(static_cast<size_t>(std::max(timeMedian_, freqMedian_)));
            maskHarm_.resize(static_cast<size_t>(numBins_));
            maskPerc_.resize(static_cast<size_t>(numBins_));
            specRaw_.resize(static_cast<size_t>(fftSize_ + 2));
        }
        else
        {
            timeMedian_ = 0;
            freqMedian_ = 0;
            magHist_.clear();
            medianScratch_.clear();
            maskHarm_.clear();
            maskPerc_.clear();
            specRaw_.clear();
        }

        prepared_ = true;
        return true;
    }

    /** @brief Clears all signal state and re-adopts the latest published
     *  parameters (jump, no glide). Stream owner only. */
    void reset() noexcept
    {
        if (!prepared_) return;
        for (auto& r : inputRing_) std::fill(r.begin(), r.end(), T(0));
        for (auto& a : accum_)     std::fill(a.begin(), a.end(), T(0));
        std::fill(prevAnalysis_.begin(), prevAnalysis_.end(), T(0));
        std::fill(prevSynth_.begin(), prevSynth_.end(), T(0));
        std::fill(prevMag_.begin(), prevMag_.end(), T(0));
        std::fill(magHist_.begin(), magHist_.end(), T(0));
        histPos_ = 0;
        histFilled_ = 0;

        std::fill(bandPrev_.begin(), bandPrev_.end(), T(0));
        std::fill(bandMaxPrev_.begin(), bandMaxPrev_.end(), T(0));
        std::fill(fluxHist_.begin(), fluxHist_.end(), 0.0);
        fluxPos_ = 0;
        fluxFilled_ = 0;
        onsetEnv_ = 0.0;
        lockLeft_ = 0;
        debt_ = 0.0;

        inputPos_ = 0;
        writeHead_ = static_cast<int64_t>(accumSize_);       // keep indices positive

        adoptParamsIfDirty();
        stActive_ = std::clamp(targetSemitones_, -12.0, 12.0);
        ratioActive_ = std::exp2(stActive_ / 12.0);
        hopCarry_ = 0.0;
        inputSinceHop_ = 0;
        analysisHop_ = computeAnalysisHop();
        firstFrame_ = true;
    }

    // -- Parameters (control thread) ------------------------------------------

    /**
     * @brief Publishes a whole parameter set (control thread only).
     *
     * Canonical seqlock publish; see the file doc for the fence derivation.
     * The set is adopted as one unit at the next analysis hop (or at
     * reset()); the active shift then glides toward the target at up to
     * 0.5 semitones per hop. May be called before prepare().
     */
    void publishParams(const Params& p) noexcept
    {
        seq_.fetch_add(1, std::memory_order_acq_rel);   // -> odd
        // Release fence: pairs with the reader's acquire fence via
        // fence-fence synchronization ([atomics.fences]/2). Without it the
        // relaxed data stores below are not ordered against the counter and
        // a torn copy could pass validation on weakly ordered targets.
        std::atomic_thread_fence(std::memory_order_release);
        stgSemitones_.store(p.targetSemitones, std::memory_order_relaxed);
        stgTransient_.store(p.transientPreserve, std::memory_order_relaxed);
        stgFormant_.store(p.formantPreserve, std::memory_order_relaxed);
        stgPhaseLock_.store(p.phaseLock, std::memory_order_relaxed);
        stgSplit_.store(p.percussiveSplit, std::memory_order_relaxed);
        seq_.fetch_add(1, std::memory_order_release);   // -> even
        dirty_.store(true, std::memory_order_release);
    }

    // -- Streaming (audio thread, stream owner) -------------------------------

    /** @return Samples the owner may push before the next analysis hop fires. */
    [[nodiscard]] int samplesToNextHop() const noexcept
    {
        return analysisHop_ - inputSinceHop_;
    }

    /** @brief Writes `count` samples (count <= samplesToNextHop()) into
     *  channel `ch`'s analysis ring. Positions advance in commitInput(). */
    void pushInput(int ch, const T* src, int count) noexcept
    {
        auto& ring = inputRing_[static_cast<size_t>(ch)];
        int wp = inputPos_;
        for (int k = 0; k < count; ++k)
        {
            ring[static_cast<size_t>(wp)] = src[k];
            wp = (wp + 1) & ringMask_;
        }
    }

    /**
     * @brief Advances the shared input position by `count` samples and runs
     * one analysis->synthesis hop when the analysis boundary is reached.
     *
     * @param count             Samples pushed to every channel since the last
     *                          commit (count <= samplesToNextHop()).
     * @param numActiveChannels Channels to synthesize this hop (<= prepared).
     */
    void commitInput(int count, int numActiveChannels) noexcept
    {
        inputPos_ = (inputPos_ + count) & ringMask_;
        inputSinceHop_ += count;
        if (inputSinceHop_ >= analysisHop_)
        {
            inputSinceHop_ = 0;
            processHop(numActiveChannels);
        }
    }

    /** @return The glided active stretch ratio (updates once per hop). */
    [[nodiscard]] double activeRatio() const noexcept { return ratioActive_; }

    /** @return The currently ADOPTED parameter set (audio thread / stream
     *  owner only: these are the audio-thread-private active copies, not the
     *  staged words). Changes only at hop boundaries or reset(). */
    [[nodiscard]] Params activeParams() const noexcept
    {
        Params p;
        p.targetSemitones   = targetSemitones_;
        p.transientPreserve = transientOn_;
        p.formantPreserve   = formantOn_;
        p.phaseLock         = phaseLockOn_;
        p.percussiveSplit   = splitOn_;
        return p;
    }

    /** @return Channel `ch`'s OLA accumulator ring (owner read access). */
    [[nodiscard]] const T* olaData(int ch) const noexcept
    {
        return accum_[static_cast<size_t>(ch)].data();
    }

    /** @return Index mask of the OLA accumulator ring (size - 1). */
    [[nodiscard]] int64_t olaMask() const noexcept
    {
        return static_cast<int64_t>(accumMask_);
    }

    /** @return OLA accumulator ring size in samples (4 * fftSize). */
    [[nodiscard]] int olaSize() const noexcept { return accumSize_; }

    /** @return Absolute write head of the OLA ring (advances Rs per hop). */
    [[nodiscard]] int64_t writeHead() const noexcept { return writeHead_; }

    /** @return Prepared frame size (0 before a successful prepare()). */
    [[nodiscard]] int fftSize() const noexcept { return prepared_ ? fftSize_ : 0; }

    /** @return Synthesis hop Rs = fftSize / 4. */
    [[nodiscard]] int synthHop() const noexcept { return synthHop_; }

private:
    static constexpr double kTwoPi = 2.0 * std::numbers::pi;

    /// Attempt bound for the audio-thread seqlock read in adoptParamsIfDirty().
    /// Three is enough that attempt 1 succeeds whenever the reader does not
    /// collide with a publish (the overwhelming majority of calls) and
    /// attempts 2-3 absorb one unlucky collision; past that, deferring the
    /// update by one hop costs less than spending more of the block budget.
    static constexpr int kSeqlockMaxAttempts = 3;

    /// How far the frequency median must dominate the time median before a
    /// bin is treated as a strike rather than a partial. Two is the value the
    /// separation literature settles on: it is well past the point where the
    /// two medians are merely comparable, and short of demanding that a bin
    /// be broadband to the exclusion of everything else.
    static constexpr double kSeparationFactor = 2.0;

    /// Onset filterbank: quarter-tone log-frequency triangles from 27.5 Hz
    /// (the lowest note of a piano) to 16 kHz, the same construction the
    /// framework's standalone onset detector uses, so the two agree on what
    /// an onset is.
    static constexpr double kOnsetFMin = 27.5;
    static constexpr double kOnsetFMax = 16000.0;
    static constexpr int kOnsetBandsPerOctave = 24;
    /// Frame length the band magnitudes are referred to before the log
    /// compression, so that one firing threshold means the same sensitivity
    /// at every frame size and every sample rate.
    static constexpr double kOdfRefFrame = 2048.0;
    /// Frames of flux history the firing threshold's running median is taken
    /// over. Twelve frames is three windows at 75% overlap: long enough that
    /// one strike cannot move the median, short enough to follow a passage
    /// that changes density.
    static constexpr int kFluxWindow = 12;
    /// How far above the running median a frame's flux must rise to fire, in
    /// the units of the compressed band difference above. It is an amount
    /// rather than a multiple: on sustained material the median flux is
    /// almost zero, and any multiple of almost zero is reached by the
    /// numerical wobble of a steady partial - measured, a multiplicative
    /// form fires 148 times on a bed with no onset in it at all, where this
    /// form fires none.
    static constexpr double kFluxDelta = 0.02;

    /** @brief Adopts the staged parameter set, or defers (bounded seqlock read).
     *
     *  At most kSeqlockMaxAttempts validation attempts. On give-up the
     *  previously adopted set stays in force and the dirty flag is re-armed,
     *  so the audio thread's worst case here is three copies of three words -
     *  this class's own instruction count - and never the time a control
     *  thread happens to hold the sequence counter odd. */
    void adoptParamsIfDirty() noexcept
    {
        if (!dirty_.exchange(false, std::memory_order_acquire)) return;

        // Bounded seqlock read (kSeqlockMaxAttempts): never adopts a torn set
        // (the accept test is the unbounded form's) and on give-up keeps the
        // previously adopted private copy.
        bool adopted = false;
        double st = 0.0;
        bool tr = true, fo = false, pl = true, sp = false;
        for (int attempt = 0; attempt < kSeqlockMaxAttempts; ++attempt)
        {
            const unsigned s0 = seq_.load(std::memory_order_acquire);
            if ((s0 & 1u) != 0u) continue;   // writer mid-publish: do not copy
            st = stgSemitones_.load(std::memory_order_relaxed);
            tr = stgTransient_.load(std::memory_order_relaxed);
            fo = stgFormant_.load(std::memory_order_relaxed);
            pl = stgPhaseLock_.load(std::memory_order_relaxed);
            sp = stgSplit_.load(std::memory_order_relaxed);
            // The fence orders the copy above BEFORE the re-read below and
            // pairs with the writer's release fence ([atomics.fences]/2); a
            // plain load-acquire orders only LATER accesses, so without it
            // the copy could sink below the re-read on weakly ordered CPUs.
            std::atomic_thread_fence(std::memory_order_acquire);
            if (s0 == seq_.load(std::memory_order_relaxed)) { adopted = true; break; }
        }
        if (!adopted)
        {
            // Adopt nothing: the active copies do not move. Re-arm and pick
            // the update up at a later hop.
            dirty_.store(true, std::memory_order_release);
            return;
        }
        // Commit-on-validate: the active copies are written only here, after
        // the copy validated, so a give-up can never leave a torn set live.
        targetSemitones_ = st;
        transientOn_     = tr;
        formantOn_       = fo;
        phaseLockOn_     = pl;
        splitOn_         = sp && separationEnabled_;
    }

    /** @brief Wraps a phase difference into (-pi, pi]. */
    [[nodiscard]] static double princArg(double x) noexcept
    {
        return x - kTwoPi * std::round(x / kTwoPi);
    }

    /**
     * @brief Next analysis hop from the active ratio, with exact fractional
     * carry, the transient lock and the debt repayment.
     *
     * While a lock is engaged the analysis hop equals the synthesis hop, so
     * the strike sits at the same offset in every frame that sees it. The
     * input those frames did not consume - Rs/ratio - Rs each - accumulates
     * as a debt, which unlocked hops repay by up to Rs/2 apiece. Rs/2 is the
     * smallest binary fraction that clears one lock's debt inside one
     * unlocked hop over the +-10% band this device is built for
     * (k >= N * (1 - ratio) / ratio = 0.444 there), and a quarter of a hop
     * does not.
     *
     * The cost of the smaller cap is not visible in the timing of the strikes
     * that do lock - those read the same to the sample either way - but in
     * how many strikes lock at all. Measured at 16 strikes per second: a
     * quarter-hop cap leaves 103 of 479 strikes with no lock, because the
     * previous lock's debt is still outstanding when the next strike arrives,
     * where a half-hop cap locks 478 of them. A whole hop measures
     * identically to a half, because past that point what binds is the
     * existence of the unlocked hop and not its width.
     */
    [[nodiscard]] int computeAnalysisHop() noexcept
    {
        double ideal;
        if (lockLeft_ > 0)
        {
            --lockLeft_;
            debt_ += static_cast<double>(synthHop_) / ratioActive_
                   - static_cast<double>(synthHop_);
            ideal = static_cast<double>(synthHop_) + hopCarry_;
        }
        else
        {
            const double cap = 0.5 * static_cast<double>(synthHop_);
            const double repay = std::clamp(debt_, -cap, cap);
            debt_ -= repay;
            ideal = static_cast<double>(synthHop_) / ratioActive_ + repay + hopCarry_;
        }
        int hop = static_cast<int>(ideal);
        hop = std::clamp(hop, 1, fftSize_);
        hopCarry_ = ideal - static_cast<double>(hop);
        return hop;
    }

    /**
     * @brief Arms the transient lock for this frame, or refuses to.
     *
     * Three refusals. A firing inside an engaged lock is ignored, so one
     * strike buys one lock and a burst cannot chain them. A lock does not arm
     * while any debt is outstanding, which is what keeps the debt from
     * accumulating across strikes - every lock starts from zero, so its
     * maximum is a property of the ratio alone and not of the tempo - and it
     * is the refusal that carries the timing: at 16 strikes per second the
     * worst per-onset displacement is 1.71 ms with it and 4.40 ms without.
     * And no lock arms on the very first frame, which always reports an onset
     * by construction and would otherwise open every stream, stationary
     * material included, with a lock nothing asked for.
     */
    void armTransientLock(bool transient) noexcept
    {
        if (!lockEnabled_) return;
        if (!transient) return;
        if (firstFrame_) return;
        if (lockLeft_ > 0) return;
        if (std::abs(debt_) >= 1.0) return;
        lockLeft_ = lockFrames_;
    }

    /** @brief One analysis->synthesis frame for all channels. */
    void processHop(int nCh) noexcept
    {
        // Adopt pending control publications at the hop boundary (cold by
        // construction: it runs only when the control thread published; the
        // audio thread never publishes to itself through the staged channel).
        adoptParamsIfDirty();

        // --- glide the active ratio toward the target (max 0.5 st per hop) ----
        const double stTarget = std::clamp(targetSemitones_, -12.0, 12.0);
        stActive_ += std::clamp(stTarget - stActive_, -0.5, 0.5);
        ratioActive_ = std::exp2(stActive_ / 12.0);

        // --- reference-channel analysis ---------------------------------------
        analyzeChannel(0);

        const bool transient = detectTransient();
        armTransientLock(transient);
        buildRotations(transient);
        // The magnitude history is kept whenever the owner allocated it, not
        // only while the split is engaged: turning the split on mid-stream
        // must find a full window behind it, not an empty one.
        if (separationEnabled_)
        {
            pushMagnitudeHistory();
            if (splitOn_) updateSeparationMasks();
        }

        // Snapshot the reference analysis spectrum for the next frame's
        // heterodyned phase increments (after decisions, before modification).
        std::copy(spec_.begin(), spec_.end(), prevAnalysis_.begin());
        std::copy(mag_.begin(), mag_.end(), prevMag_.begin());

        // --- synthesis: reference channel first, then the rest -----------------
        synthesizeChannel(0, true);
        for (int ch = 1; ch < nCh; ++ch)
        {
            analyzeChannel(ch);
            synthesizeChannel(ch, false);
        }

        analysisHop_ = computeAnalysisHop();
        writeHead_ += synthHop_;
        firstFrame_ = false;
    }

    /** @brief Windows the input ring into fftIn_ and fills spec_ for `ch`. */
    void analyzeChannel(int ch) noexcept
    {
        const auto& ring = inputRing_[static_cast<size_t>(ch)];
        const int readPos = inputPos_;   // oldest sample (ring size == fftSize)
        for (int k = 0; k < fftSize_; ++k)
        {
            const int idx = (readPos + k) & ringMask_;
            fftIn_[static_cast<size_t>(k)] = ring[static_cast<size_t>(idx)]
                                           * window_[static_cast<size_t>(k)];
        }
        fft_->forward(fftIn_.data(), spec_.data());

        if (ch == 0)
        {
            for (int k = 0; k < numBins_; ++k)
            {
                const T re = spec_[static_cast<size_t>(2 * k)];
                const T im = spec_[static_cast<size_t>(2 * k + 1)];
                mag_[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
            }
        }
    }

    /**
     * @brief The onset test the owner selected at prepare().
     *
     * The first frame always reports true whichever detector is running:
     * there is no history to compare against, and a stream's first frame has
     * no previous synthesis phase to propagate from either.
     */
    [[nodiscard]] bool detectTransient() noexcept
    {
        return fluxDetectorEnabled_ ? detectTransientByFlux()
                                    : detectTransientByEnergy();
    }

    /**
     * @brief Broadband frame-energy onset test: a 6 dB rise over an envelope
     * that decays at 0.7 per frame.
     *
     * Blunt by construction. It cannot see an attack that does not lift the
     * whole frame's energy, which is why an owner driving a hop schedule off
     * onsets asks for the flux detector instead. It is kept, and kept exact,
     * because the firings of a published effect are part of what that effect
     * sounds like: changing them changes renders that already exist.
     */
    [[nodiscard]] bool detectTransientByEnergy() noexcept
    {
        double energy = 0.0;
        for (int k = 0; k < numBins_; ++k)
        {
            const double m = static_cast<double>(mag_[static_cast<size_t>(k)]);
            energy += m * m;
        }
        const double prevEnv = onsetEnv_;
        onsetEnv_ = std::max(energy, onsetEnv_ * 0.7);
        if (!transientOn_)
            return firstFrame_;
        return firstFrame_ || (energy > 4.0 * prevEnv && energy > 1e-12);
    }

    /**
     * @brief Half-wave-rectified spectral flux over the log-frequency
     * filterbank, fired against the running median of its own history.
     *
     * The flux is taken against a three-neighbour frequency maximum filter
     * of the previous band frame rather than against the band frame itself,
     * so a partial that wobbles by up to one band - vibrato, a bent note -
     * contributes nothing, while a strike, which lights up bands that were
     * dark in every neighbour, contributes everything. The log compression
     * before the difference is what makes one threshold work across levels:
     * without it the flux of a loud passage swamps the flux of a quiet one
     * and no single multiple of the median fits both.
     *
     * The history is advanced on every frame, whether or not transient
     * handling is engaged, so that switching the parameter on mid-stream
     * finds a full window behind it rather than an empty one.
     */
    [[nodiscard]] bool detectTransientByFlux() noexcept
    {
        const double flux = computeSpectralFlux();

        // Median of the trailing window, taken before this frame is pushed:
        // a frame is compared with the material that preceded it, never with
        // itself.
        bool rise = false;
        if (fluxFilled_ >= kFluxWindow)
        {
            std::copy(fluxHist_.begin(), fluxHist_.end(), fluxScratch_.begin());
            const auto mid = fluxScratch_.begin() + kFluxWindow / 2;
            std::nth_element(fluxScratch_.begin(), mid, fluxScratch_.end());
            rise = (flux - *mid) >= kFluxDelta;
        }

        fluxHist_[static_cast<size_t>(fluxPos_)] = flux;
        fluxPos_ = (fluxPos_ + 1) % kFluxWindow;
        if (fluxFilled_ < kFluxWindow) ++fluxFilled_;

        if (!transientOn_)
            return firstFrame_;
        return firstFrame_ || rise;
    }

    /** @brief One frame of log-filterbank half-wave-rectified spectral flux,
     *  and the band-history rotation that goes with it. */
    [[nodiscard]] double computeSpectralFlux() noexcept
    {
        filterLogBands();

        double flux = 0.0;
        for (int b = 0; b < numBands_; ++b)
        {
            const double d = static_cast<double>(bandCur_[static_cast<size_t>(b)])
                           - static_cast<double>(bandMaxPrev_[static_cast<size_t>(b)]);
            if (d > 0.0) flux += d;
        }
        flux /= static_cast<double>(std::max(1, numBands_));

        // Rotate: previous <- current, and rebuild the maximum-filtered
        // reference the next frame will be differenced against.
        bandPrev_ = bandCur_;
        maxFilterBands();
        return flux;
    }

    /** @brief Applies the filterbank to mag_ and compresses each band as
     *  log10(scale * x + 1). Scaling the accumulated band is identical to
     *  scaling the spectrum first, the filterbank being linear. */
    void filterLogBands() noexcept
    {
        for (int b = 0; b < numBands_; ++b)
        {
            const int start = fbStart_[static_cast<size_t>(b)];
            const int off   = fbOffset_[static_cast<size_t>(b)];
            const int cnt   = fbCount_[static_cast<size_t>(b)];
            T acc = T(0);
            for (int i = 0; i < cnt; ++i)
            {
                const int k = start + i;
                if (k >= 0 && k < numBins_)
                    acc += mag_[static_cast<size_t>(k)]
                         * fbWeights_[static_cast<size_t>(off + i)];
            }
            bandCur_[static_cast<size_t>(b)] = std::log10(acc * odfScale_ + T(1));
        }
    }

    /** @brief Three-neighbour maximum filter of bandPrev_ into bandMaxPrev_. */
    void maxFilterBands() noexcept
    {
        for (int b = 0; b < numBands_; ++b)
        {
            T mx = bandPrev_[static_cast<size_t>(b)];
            if (b > 0) mx = std::max(mx, bandPrev_[static_cast<size_t>(b - 1)]);
            if (b + 1 < numBands_) mx = std::max(mx, bandPrev_[static_cast<size_t>(b + 1)]);
            bandMaxPrev_[static_cast<size_t>(b)] = mx;
        }
    }

    /**
     * @brief Builds the quarter-tone log-frequency triangular filterbank.
     *
     * Centres are the quarter-tone grid from 27.5 Hz up, rounded to bins and
     * de-duplicated, so at small frame sizes the low end thins out to
     * whatever the resolution actually supports instead of pretending to a
     * spacing it cannot deliver. Each filter is the triangle over a
     * consecutive triple of centres, peak-normalised.
     */
    void buildOnsetFilterBank()
    {
        fbStart_.clear();
        fbOffset_.clear();
        fbCount_.clear();
        fbWeights_.clear();
        numBands_ = 0;

        const double binHz = sampleRate_ / static_cast<double>(fftSize_);
        const double fMax = std::min(kOnsetFMax, sampleRate_ * 0.5 * 0.999);

        std::vector<int> centres;
        for (int i = 0; ; ++i)
        {
            const double f = kOnsetFMin * std::pow(2.0, static_cast<double>(i)
                                 / static_cast<double>(kOnsetBandsPerOctave));
            if (f > fMax) break;
            int bin = static_cast<int>(std::lround(f / binHz));
            bin = std::clamp(bin, 0, numBins_ - 1);
            if (centres.empty() || bin > centres.back())
                centres.push_back(bin);
        }

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
                const T wv = (k <= ce)
                    ? static_cast<T>(static_cast<double>(k - lo)
                                     / static_cast<double>(ce - lo))
                    : static_cast<T>(static_cast<double>(hi - k)
                                     / static_cast<double>(hi - ce));
                fbWeights_.push_back(wv);
            }
            ++numBands_;
        }
        for (int b = 0; b < numBands_; ++b)
        {
            const int off = fbOffset_[static_cast<size_t>(b)];
            const int nextOff = (b + 1 < numBands_)
                                ? fbOffset_[static_cast<size_t>(b + 1)]
                                : static_cast<int>(fbWeights_.size());
            fbCount_.push_back(nextOff - off);
        }
        // A frame size too small to carry a single triangle leaves the flux
        // at zero, which never fires: the detector goes quiet rather than
        // reading out of an empty table.
        if (numBands_ < 1) numBands_ = 0;
    }

    /**
     * @brief Peak picking + phase propagation on the reference channel.
     *
     * Fills rotRe_/rotIm_ with one rigid rotation phasor per bin (constant
     * across each peak's region of influence). On transients the rotation is
     * identity, which resets synthesis phases to analysis phases.
     */
    void buildRotations(bool transient) noexcept
    {
        if (transient)
        {
            std::fill(rotRe_.begin(), rotRe_.end(), T(1));
            std::fill(rotIm_.begin(), rotIm_.end(), T(0));
            return;
        }

        if (!phaseLockOn_)
        {
            buildPerBinRotations();
            return;
        }

        // --- find spectral peaks (local maxima over +-2 bins, above floor) -----
        T maxMag = T(0);
        for (int k = 0; k < numBins_; ++k)
            maxMag = std::max(maxMag, mag_[static_cast<size_t>(k)]);

        numPeaks_ = 0;
        if (maxMag > T(1e-9))
        {
            const T floorMag = maxMag * T(1e-4);   // -80 dB relative floor
            // Candidates stop at numBins-3 so the +-2-bin neighbour tests
            // stay inside mag_ (the old bound of numBins-2 read mag_[k+2]
            // one float past the end whenever the bin below Nyquist was a
            // candidate above the floor - broadband material at small frame
            // sizes reached it routinely).
            const int last = numBins_ - 3;
            for (int k = 2; k <= last; ++k)
            {
                const T m = mag_[static_cast<size_t>(k)];
                if (m < floorMag) continue;
                if (m > mag_[static_cast<size_t>(k - 1)] && m >= mag_[static_cast<size_t>(k + 1)]
                    && m > mag_[static_cast<size_t>(k - 2)] && m >= mag_[static_cast<size_t>(k + 2)])
                {
                    peakBin_[static_cast<size_t>(numPeaks_++)] = k;
                    k += 2;   // a neighbour cannot also be a peak
                }
            }
        }

        if (numPeaks_ == 0)
        {
            std::fill(rotRe_.begin(), rotRe_.end(), T(1));
            std::fill(rotIm_.begin(), rotIm_.end(), T(0));
            return;
        }

        // --- per-peak heterodyned phase propagation ----------------------------
        // analysisHop_ still holds the hop just consumed (it is recomputed
        // after this call), which is exactly the Ra of this frame interval.
        const double Ra = static_cast<double>(analysisHop_);
        const double Rs = static_cast<double>(synthHop_);
        const double binW = kTwoPi / static_cast<double>(fftSize_);

        int regionStart = 0;
        for (int p = 0; p < numPeaks_; ++p)
        {
            const int bin = peakBin_[static_cast<size_t>(p)];
            const double re  = static_cast<double>(spec_[static_cast<size_t>(2 * bin)]);
            const double im  = static_cast<double>(spec_[static_cast<size_t>(2 * bin + 1)]);
            const double pre = static_cast<double>(prevAnalysis_[static_cast<size_t>(2 * bin)]);
            const double pim = static_cast<double>(prevAnalysis_[static_cast<size_t>(2 * bin + 1)]);

            double rotR = 1.0, rotI = 0.0;
            // A peak with no spectral history is a fresh partial: keep its
            // analysis phase instead of propagating from noise.
            if (prevMag_[static_cast<size_t>(bin)] > T(0.1) * mag_[static_cast<size_t>(bin)])
            {
                // Measured inter-frame phase advance, inherently wrapped.
                const double deltaPhi = std::atan2(im * pre - re * pim, re * pre + im * pim);
                const double omegaK = binW * static_cast<double>(bin);
                const double omegaInst = omegaK + princArg(deltaPhi - omegaK * Ra) / Ra;

                const double psiPrev = std::atan2(
                    static_cast<double>(prevSynth_[static_cast<size_t>(2 * bin + 1)]),
                    static_cast<double>(prevSynth_[static_cast<size_t>(2 * bin)]));
                const double phi = std::atan2(im, re);
                const double theta = psiPrev + Rs * omegaInst - phi;
                rotR = std::cos(theta);
                rotI = std::sin(theta);
            }

            // Region of influence: up to the magnitude valley before the next peak.
            int regionEnd = numBins_;   // exclusive
            if (p + 1 < numPeaks_)
            {
                const int nextBin = peakBin_[static_cast<size_t>(p + 1)];
                int valley = bin + 1;
                T valleyMag = mag_[static_cast<size_t>(valley)];
                for (int k = bin + 2; k < nextBin; ++k)
                {
                    if (mag_[static_cast<size_t>(k)] < valleyMag)
                    {
                        valleyMag = mag_[static_cast<size_t>(k)];
                        valley = k;
                    }
                }
                regionEnd = valley + 1;
            }

            for (int k = regionStart; k < regionEnd; ++k)
            {
                rotRe_[static_cast<size_t>(k)] = static_cast<T>(rotR);
                rotIm_[static_cast<size_t>(k)] = static_cast<T>(rotI);
            }
            regionStart = regionEnd;
        }

        // DC and Nyquist are real-valued in the packed spectrum: never rotate.
        rotRe_[0] = T(1);                                     rotIm_[0] = T(0);
        rotRe_[static_cast<size_t>(numBins_ - 1)] = T(1);     rotIm_[static_cast<size_t>(numBins_ - 1)] = T(0);
    }

    /**
     * @brief Plain-vocoder rotations: one independent phase increment per bin.
     *
     * The classic phase vocoder, with no coupling between neighbouring bins.
     * It is exact for a single stationary sinusoid per bin and drifts apart
     * across the bins of one partial as soon as the partial moves, which is
     * the vertical incoherence that makes a plain vocoder sound hollow. Kept
     * as the control condition for the locked path above.
     */
    void buildPerBinRotations() noexcept
    {
        const double Ra = static_cast<double>(analysisHop_);
        const double Rs = static_cast<double>(synthHop_);
        const double binW = kTwoPi / static_cast<double>(fftSize_);

        // DC and Nyquist are real-valued in the packed spectrum: never rotate.
        rotRe_[0] = T(1);                                   rotIm_[0] = T(0);
        rotRe_[static_cast<size_t>(numBins_ - 1)] = T(1);
        rotIm_[static_cast<size_t>(numBins_ - 1)] = T(0);

        for (int k = 1; k < numBins_ - 1; ++k)
        {
            double rotR = 1.0, rotI = 0.0;
            // A bin with no magnitude history carries no phase to propagate.
            if (prevMag_[static_cast<size_t>(k)] > T(0.1) * mag_[static_cast<size_t>(k)])
            {
                const double re  = static_cast<double>(spec_[static_cast<size_t>(2 * k)]);
                const double im  = static_cast<double>(spec_[static_cast<size_t>(2 * k + 1)]);
                const double pre = static_cast<double>(prevAnalysis_[static_cast<size_t>(2 * k)]);
                const double pim = static_cast<double>(prevAnalysis_[static_cast<size_t>(2 * k + 1)]);

                const double deltaPhi = std::atan2(im * pre - re * pim, re * pre + im * pim);
                const double omegaK = binW * static_cast<double>(k);
                const double omegaInst = omegaK + princArg(deltaPhi - omegaK * Ra) / Ra;

                const double psiPrev = std::atan2(
                    static_cast<double>(prevSynth_[static_cast<size_t>(2 * k + 1)]),
                    static_cast<double>(prevSynth_[static_cast<size_t>(2 * k)]));
                const double phi = std::atan2(im, re);
                const double theta = psiPrev + Rs * omegaInst - phi;
                rotR = std::cos(theta);
                rotI = std::sin(theta);
            }
            rotRe_[static_cast<size_t>(k)] = static_cast<T>(rotR);
            rotIm_[static_cast<size_t>(k)] = static_cast<T>(rotI);
        }
    }

    /** @brief Largest odd value <= `v`, clamped into [lo, hi] and kept odd. */
    [[nodiscard]] static int makeOdd(int v, int lo, int hi) noexcept
    {
        v = std::clamp(v, lo, hi);
        if ((v & 1) == 0) --v;
        return std::max(v, lo | 1);
    }

    /** @brief Rolls the current magnitudes into the trailing history ring. */
    void pushMagnitudeHistory() noexcept
    {
        T* slot = magHist_.data() + static_cast<size_t>(histPos_) * static_cast<size_t>(numBins_);
        std::copy(mag_.begin(), mag_.end(), slot);
        histPos_ = (histPos_ + 1) % timeMedian_;
        if (histFilled_ < timeMedian_) ++histFilled_;
    }

    /**
     * @brief Complementary harmonic/percussive masks for the current frame.
     *
     * A partial that is steady in time survives a median taken ALONG TIME at
     * its own bin and is wiped out by one taken ALONG FREQUENCY across it; a
     * broadband strike is the other way round. Comparing the two medians
     * therefore separates the two kinds of content without any model of
     * either. The masks are complementary and binary: every bin goes down
     * exactly one of the two paths, so the split neither loses nor invents
     * energy.
     *
     * The time median is TRAILING (the current frame and the ones before it)
     * rather than centred: a centred window would need frames that have not
     * been analysed yet, which is latency this engine does not have to spend.
     * The cost is that the mask reacts to a strike one window late on its
     * decay side, not on its onset.
     */
    void updateSeparationMasks() noexcept
    {
        const int nHist = histFilled_;
        const int half = freqMedian_ / 2;

        for (int k = 0; k < numBins_; ++k)
        {
            // Median along time at this bin, over the frames seen so far.
            for (int f = 0; f < nHist; ++f)
                medianScratch_[static_cast<size_t>(f)] =
                    magHist_[static_cast<size_t>(f) * static_cast<size_t>(numBins_)
                             + static_cast<size_t>(k)];
            auto mid = medianScratch_.begin() + nHist / 2;
            std::nth_element(medianScratch_.begin(), mid,
                             medianScratch_.begin() + nHist);
            const double h = static_cast<double>(*mid);

            // Median along frequency in this frame, clamped at both edges so
            // the window never reads outside the spectrum.
            const int lo = std::max(0, k - half);
            const int hi = std::min(numBins_ - 1, k + half);
            const int n = hi - lo + 1;
            for (int j = 0; j < n; ++j)
                medianScratch_[static_cast<size_t>(j)] = mag_[static_cast<size_t>(lo + j)];
            auto midF = medianScratch_.begin() + n / 2;
            std::nth_element(medianScratch_.begin(), midF, medianScratch_.begin() + n);
            const double p = static_cast<double>(*midF);

            // A bin goes down the percussive path only when the frequency
            // median dominates the time median by the separation factor.
            // Everything else, including everything ambiguous, stays
            // harmonic: the phase-propagated path is the one that is right
            // for most material, so the split must earn each bin it takes
            // rather than divide every bin in proportion. A proportional
            // (Wiener) mask measured 1.5 dB of log-spectral distance on a
            // purely harmonic bed, where the correct answer is that nothing
            // is percussive at all.
            const bool percussive = (p > kSeparationFactor * h);
            maskHarm_[static_cast<size_t>(k)] = percussive ? T(0) : T(1);
            maskPerc_[static_cast<size_t>(k)] = percussive ? T(1) : T(0);
        }
    }

    /** @brief Applies rotations + spectral post-stages to spec_, IFFTs and
     *  overlap-adds. */
    void synthesizeChannel(int ch, bool isReference) noexcept
    {
        // The percussive path re-uses this channel's UNROTATED spectrum, so
        // it has to be kept before the rotation overwrites it.
        if (splitOn_)
            std::copy(spec_.begin(), spec_.end(), specRaw_.begin());

        // Rigid per-region rotation preserves intra-region (and inter-channel)
        // relative phases exactly.
        for (int k = 1; k < numBins_ - 1; ++k)
        {
            const T re = spec_[static_cast<size_t>(2 * k)];
            const T im = spec_[static_cast<size_t>(2 * k + 1)];
            const T rr = rotRe_[static_cast<size_t>(k)];
            const T ri = rotIm_[static_cast<size_t>(k)];
            spec_[static_cast<size_t>(2 * k)]     = re * rr - im * ri;
            spec_[static_cast<size_t>(2 * k + 1)] = re * ri + im * rr;
        }

        // Formant preservation: pre-warp synthesis magnitudes by
        // env(k*ratio)/env(k) so the owner's output resampler puts the
        // spectral envelope back where the input had it. The gain table is
        // computed once on the reference channel and shared, keeping the
        // stereo image coherent.
        if (formantOn_ && std::abs(ratioActive_ - 1.0) > 1e-6)
        {
            if (isReference)
                computeFormantGains();
            for (int k = 1; k < numBins_ - 1; ++k)
            {
                const T g = formantGain_[static_cast<size_t>(k)];
                spec_[static_cast<size_t>(2 * k)]     *= g;
                spec_[static_cast<size_t>(2 * k + 1)] *= g;
            }
        }

        // Anti-alias for upward shifts (resample-back owners): the owner's
        // resampler multiplies frequencies by `ratio`, so taper bins that
        // would land above Nyquist.
        if (resampleCompensation_ && ratioActive_ > 1.0)
        {
            const int cut = static_cast<int>(static_cast<double>(fftSize_ / 2) / ratioActive_);
            const int taperStart = std::max(1, cut - 4);
            for (int k = taperStart; k < numBins_; ++k)
            {
                T g = T(0);
                if (k <= cut)
                    g = static_cast<T>(cut - k + 1) / static_cast<T>(cut - taperStart + 1);
                spec_[static_cast<size_t>(2 * k)]     *= g;
                spec_[static_cast<size_t>(2 * k + 1)] *= g;
            }
        }

        if (isReference)
            std::copy(spec_.begin(), spec_.end(), prevSynth_.begin());

        // Harmonic/percussive recombination. The phase recursion above is fed
        // the fully propagated spectrum (the line just past), never the mix:
        // the percussive share deliberately breaks the phase recursion, and
        // letting it back in would poison the next frame's propagation.
        if (splitOn_)
        {
            for (int k = 0; k < numBins_; ++k)
            {
                const T mh = maskHarm_[static_cast<size_t>(k)];
                const T mp = maskPerc_[static_cast<size_t>(k)];
                const auto re = static_cast<size_t>(2 * k);
                const auto im = static_cast<size_t>(2 * k + 1);
                spec_[re] = spec_[re] * mh + specRaw_[re] * mp;
                spec_[im] = spec_[im] * mh + specRaw_[im] * mp;
            }
        }

        fft_->inverse(spec_.data(), fftResult_.data());

        synthesizeTail(ch);
    }

    /**
     * @brief Cepstral envelope of the current synthesis spectrum and the
     * env(k*ratio)/env(k) pre-warp gains.
     *
     * The log magnitude is mirrored to an even sequence, transformed, low-
     * quefrency liftered (~1 ms keeps formant structure, drops the harmonic
     * comb) and transformed back into a smooth log envelope.
     */
    void computeFormantGains() noexcept
    {
        // Even-symmetric log magnitude.
        for (int k = 0; k < numBins_; ++k)
        {
            const T re = spec_[static_cast<size_t>(2 * k)];
            const T im = spec_[static_cast<size_t>(2 * k + 1)];
            cepsTime_[static_cast<size_t>(k)] =
                std::log(std::sqrt(re * re + im * im) + T(1e-9));
        }
        for (int k = numBins_; k < fftSize_; ++k)
            cepsTime_[static_cast<size_t>(k)] =
                cepsTime_[static_cast<size_t>(fftSize_ - k)];

        fft_->forward(cepsTime_.data(), cepsSpec_.data());

        // Lifter: keep quefrencies below ~1 ms (the smooth envelope), zero
        // the rest (the harmonic comb). Cepstral index is quefrency in
        // samples: 1 ms = 0.001 * fs.
        const int qCut = std::max(8, static_cast<int>(0.001 * sampleRate_));
        const int keep = std::min(qCut, numBins_ - 1);
        for (int k = keep + 1; k < numBins_; ++k)
        {
            cepsSpec_[static_cast<size_t>(2 * k)]     = T(0);
            cepsSpec_[static_cast<size_t>(2 * k + 1)] = T(0);
        }

        fft_->inverse(cepsSpec_.data(), cepsTime_.data());
        for (int k = 0; k < numBins_; ++k)
            envLog_[static_cast<size_t>(k)] = cepsTime_[static_cast<size_t>(k)];

        // Gains with linear interpolation at k*ratio, clamped to +-40 dB.
        for (int k = 0; k < numBins_; ++k)
        {
            const double pos = std::min(static_cast<double>(k) * ratioActive_,
                                        static_cast<double>(numBins_ - 1));
            const auto i0 = static_cast<int>(pos);
            const auto frac = static_cast<T>(pos - i0);
            const T target = envLog_[static_cast<size_t>(i0)]
                + (envLog_[static_cast<size_t>(std::min(i0 + 1, numBins_ - 1))]
                   - envLog_[static_cast<size_t>(i0)]) * frac;
            const T delta = std::clamp(target - envLog_[static_cast<size_t>(k)],
                                       T(-4.6), T(4.6));
            formantGain_[static_cast<size_t>(k)] = std::exp(delta);
        }
    }

    /** @brief Overlap-add of the already-inverted frame (split for clarity). */
    void synthesizeTail(int ch) noexcept
    {
        // Overlap-add at the fixed synthesis hop. sqrt-Hann analysis+synthesis
        // with Rs = N/4 overlap-adds to a constant 2.0, hence the 0.5 norm.
        auto& acc = accum_[static_cast<size_t>(ch)];
        const auto m = static_cast<int64_t>(accumMask_);

        // The tail region [w + N - Rs, w + N) is entered for the first time by
        // this frame: clear the stale ring content before accumulating.
        for (int k = fftSize_ - synthHop_; k < fftSize_; ++k)
            acc[static_cast<size_t>((writeHead_ + k) & m)] = T(0);

        constexpr T kNorm = T(0.5);
        for (int k = 0; k < fftSize_; ++k)
        {
            const auto idx = static_cast<size_t>((writeHead_ + k) & m);
            acc[idx] += fftResult_[static_cast<size_t>(k)]
                      * window_[static_cast<size_t>(k)] * kNorm;
        }
    }

    // -- Members -----------------------------------------------------------------
    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    bool prepared_ = false;
    bool resampleCompensation_ = false;

    int fftSize_ = 2048;
    int numBins_ = 1025;
    int synthHop_ = 512;
    int ringMask_ = 2047;
    int accumSize_ = 8192;
    int accumMask_ = 8191;

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_;

    std::vector<std::vector<T>> inputRing_;   ///< Per-channel analysis ring (size N).
    std::vector<std::vector<T>> accum_;       ///< Per-channel synthesis OLA ring.

    std::vector<T> fftIn_, spec_, fftResult_;
    std::vector<T> prevAnalysis_, prevSynth_; ///< Reference-channel spectra (re,im).
    std::vector<T> mag_, prevMag_;            ///< Reference-channel magnitudes.
    std::vector<T> cepsTime_, cepsSpec_;      ///< Formant cepstrum scratch.
    std::vector<T> envLog_, formantGain_;     ///< Envelope + pre-warp gains.
    std::vector<T> rotRe_, rotIm_;            ///< Per-bin rigid rotation phasor.
    std::vector<int> peakBin_;
    int numPeaks_ = 0;

    // Onset detection. The frame-energy test needs one scalar; the flux
    // detector needs the log-frequency filterbank tables, the two band
    // frames the flux is taken between, and the flux history the firing
    // threshold's running median is taken over. Which of the two runs is the
    // owner's choice, fixed in prepare(), and only the selected one is
    // allocated for.
    bool fluxDetectorEnabled_ = false;        ///< Owner asked for spectral flux.
    double onsetEnv_ = 0.0;                   ///< Frame-energy test's tracked envelope.
    std::vector<int> fbStart_, fbOffset_, fbCount_;
    std::vector<T> fbWeights_;
    int numBands_ = 0;
    T odfScale_ = T(1);
    std::vector<T> bandCur_, bandPrev_, bandMaxPrev_;
    std::vector<double> fluxHist_;            ///< Trailing flux window (ring).
    std::vector<double> fluxScratch_;         ///< Selection scratch for the median.
    int fluxPos_ = 0;                         ///< Next history slot to overwrite.
    int fluxFilled_ = 0;                      ///< Slots written since reset().

    // Transient-locked hop.
    bool lockEnabled_ = false;                ///< Owner asked for the locked hop.
    int lockFrames_ = 0;                      ///< Lock length, ceil(fftSize / Rs).
    int lockLeft_ = 0;                        ///< Locked frames still to come.
    double debt_ = 0.0;                       ///< Input the lock has not consumed.

    // Harmonic/percussive split (allocated only when the owner asked for it).
    bool separationEnabled_ = false;
    int timeMedian_ = 0;                      ///< Median length along time, frames (odd).
    int freqMedian_ = 0;                      ///< Median length along frequency, bins (odd).
    std::vector<T> magHist_;                  ///< timeMedian_ x numBins_ magnitude ring.
    std::vector<T> medianScratch_;            ///< Selection scratch for one median.
    std::vector<T> maskHarm_, maskPerc_;      ///< Complementary per-bin masks.
    std::vector<T> specRaw_;                  ///< Analysis spectrum before rotation.
    int histPos_ = 0;                         ///< Next history slot to overwrite.
    int histFilled_ = 0;                      ///< Slots written since reset().

    int inputPos_ = 0;
    int64_t writeHead_ = 0;

    // Audio-thread-private active state: the stream owner writes these as
    // plain members and never publishes them through the staged channel.
    double stActive_ = 0.0;
    double ratioActive_ = 1.0;
    double hopCarry_ = 0.0;
    int analysisHop_ = 512;
    int inputSinceHop_ = 0;
    bool firstFrame_ = true;

    // Audio-thread-private active copies of the published parameter set
    // (committed only by a validated bounded read; defaults == Params{}).
    double targetSemitones_ = 0.0;
    bool transientOn_ = true;
    bool formantOn_ = false;
    bool phaseLockOn_ = true;
    bool splitOn_ = false;

    // Control -> audio staged parameter channel (canonical seqlock; the ONLY
    // cross-thread handoff in this class).
    std::atomic<double> stgSemitones_ { 0.0 };
    std::atomic<bool> stgTransient_ { true };
    std::atomic<bool> stgFormant_ { false };
    std::atomic<bool> stgPhaseLock_ { true };
    std::atomic<bool> stgSplit_ { false };
    std::atomic<unsigned> seq_ { 0 };        ///< Seqlock counter (odd = writing).
    std::atomic<bool> dirty_ { false };      ///< Pending-update flag (fast path).
};

} // namespace detail
} // namespace dspark
