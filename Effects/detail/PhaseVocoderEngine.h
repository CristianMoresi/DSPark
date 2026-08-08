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
 *   phase coherence.
 * - **Transient phase reset**: onsets (frame energy rising more than 6 dB
 *   over the tracked envelope) re-initialise synthesis phases to the
 *   analysis phases. Peaks with no history (new partials) are reset
 *   individually even without a global onset.
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
 * Threading (SPSC model, docs/threading.md; ADR-013/014/015 apply from birth):
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
 * transientPreserve, formantPreserve} from the control thread to the audio
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
 * count - never the time the control thread holds the counter odd (ADR-014).
 * Bounding cannot make a torn set adoptable: the accept predicate is the
 * same as the unbounded form's, so the bound can only DEFER an adoption.
 *
 * Origin rule (ADR-015): the staged channel above carries CONTROL -> AUDIO
 * publications only. Everything the audio thread computes for its own use -
 * the glided active ratio, the fractional hop accumulator, the spectral
 * state - is stream-owner state written directly as plain members; the audio
 * thread never publishes to itself through the staged channel, so the
 * adoption above is cold by construction (it runs only when the control
 * thread actually published).
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
     * @return true if the engine (re)allocated and is ready for reset().
     */
    bool prepare(double sampleRate, int numChannels, int fftSize,
                 bool resampleCompensation)
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

        inputPos_ = 0;
        writeHead_ = static_cast<int64_t>(accumSize_);       // keep indices positive

        adoptParamsIfDirty();
        stActive_ = std::clamp(targetSemitones_, -12.0, 12.0);
        ratioActive_ = std::exp2(stActive_ / 12.0);
        hopCarry_ = 0.0;
        inputSinceHop_ = 0;
        analysisHop_ = computeAnalysisHop();
        onsetEnv_ = 0.0;
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
        bool tr = true, fo = false;
        for (int attempt = 0; attempt < kSeqlockMaxAttempts; ++attempt)
        {
            const unsigned s0 = seq_.load(std::memory_order_acquire);
            if ((s0 & 1u) != 0u) continue;   // writer mid-publish: do not copy
            st = stgSemitones_.load(std::memory_order_relaxed);
            tr = stgTransient_.load(std::memory_order_relaxed);
            fo = stgFormant_.load(std::memory_order_relaxed);
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
    }

    /** @brief Wraps a phase difference into (-pi, pi]. */
    [[nodiscard]] static double princArg(double x) noexcept
    {
        return x - kTwoPi * std::round(x / kTwoPi);
    }

    /** @brief Next analysis hop from the active ratio, with exact fractional carry. */
    [[nodiscard]] int computeAnalysisHop() noexcept
    {
        const double ideal = static_cast<double>(synthHop_) / ratioActive_ + hopCarry_;
        int hop = static_cast<int>(ideal);
        hop = std::clamp(hop, 1, fftSize_);
        hopCarry_ = ideal - static_cast<double>(hop);
        return hop;
    }

    /** @brief One analysis->synthesis frame for all channels. */
    void processHop(int nCh) noexcept
    {
        // Adopt pending control publications at the hop boundary (cold: only
        // runs when the control thread actually published; ADR-015).
        adoptParamsIfDirty();

        // --- glide the active ratio toward the target (max 0.5 st per hop) ----
        const double stTarget = std::clamp(targetSemitones_, -12.0, 12.0);
        stActive_ += std::clamp(stTarget - stActive_, -0.5, 0.5);
        ratioActive_ = std::exp2(stActive_ / 12.0);

        // --- reference-channel analysis ---------------------------------------
        analyzeChannel(0);

        const bool transient = detectTransient();
        buildRotations(transient);

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

    /** @brief Frame-energy onset detector (6 dB rise over tracked envelope). */
    [[nodiscard]] bool detectTransient() noexcept
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

    /** @brief Applies rotations + spectral post-stages to spec_, IFFTs and
     *  overlap-adds. */
    void synthesizeChannel(int ch, bool isReference) noexcept
    {
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

    int inputPos_ = 0;
    int64_t writeHead_ = 0;

    // Audio-thread-private active state (stream-owner writes only; ADR-015).
    double stActive_ = 0.0;
    double ratioActive_ = 1.0;
    double hopCarry_ = 0.0;
    int analysisHop_ = 512;
    int inputSinceHop_ = 0;
    double onsetEnv_ = 0.0;
    bool firstFrame_ = true;

    // Audio-thread-private active copies of the published parameter set
    // (committed only by a validated bounded read; defaults == Params{}).
    double targetSemitones_ = 0.0;
    bool transientOn_ = true;
    bool formantOn_ = false;

    // Control -> audio staged parameter channel (canonical seqlock; the ONLY
    // cross-thread handoff in this class).
    std::atomic<double> stgSemitones_ { 0.0 };
    std::atomic<bool> stgTransient_ { true };
    std::atomic<bool> stgFormant_ { false };
    std::atomic<unsigned> seq_ { 0 };        ///< Seqlock counter (odd = writing).
    std::atomic<bool> dirty_ { false };      ///< Pending-update flag (fast path).
};

} // namespace detail
} // namespace dspark
