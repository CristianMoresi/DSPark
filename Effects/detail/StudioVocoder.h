// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file StudioVocoder.h
 * @brief Phase-gradient vocoder with strike-anchored time map (internal detail).
 *
 * The engine behind `Quality::Studio` in Effects/TimeStretch.h and
 * Effects/PitchShifter.h. It is an INTERNAL header in namespace dspark::detail:
 * its API is a contract with those two owners, not with library users. It
 * exposes the same streaming surface as detail::PhaseVocoderEngine
 * (pushInput / commitInput / samplesToNextHop / olaData / writeHead), so an
 * owner drives either engine with the same loop.
 *
 * What it does per frame (streaming, zero allocation after prepare()):
 *
 *   input ring -> analysis frame centred on a(c), the time map
 *     -> FFT (and a second FFT 16 samples later, for the frequency estimate)
 *     -> heap-ordered phase propagation on the channel-summed reference
 *     -> the same rigid per-bin rotation applied to every channel
 *     -> optional formant pre-warp and anti-alias taper (resample owners)
 *     -> IFFT -> overlap-add at the fixed synthesis hop Rs = N/4
 *
 * Techniques, each implemented from its paper:
 *
 * - **Phase-gradient heap integration** (Prusa & Holighaus, "Phase Vocoder
 *   Done Right", EUSIPCO 2017). Bins are visited in decreasing order of
 *   magnitude across the previous and the current frame. A bin reached from
 *   the previous frame is advanced along time by its instantaneous
 *   frequency; a bin reached from a louder neighbour in the current frame
 *   inherits that neighbour's rotation, which keeps the analysis frame's
 *   vertical phase structure. On a stationary partial that is identity phase
 *   locking without a peak picker; on an attack, whose energy is new, the
 *   frequency direction wins by itself and the strike keeps its shape with
 *   no detector involved.
 * - **One reference for every channel.** Magnitudes are summed in power over
 *   the channels and phase increments are taken from cross-spectra summed
 *   over the channels, so a source panned hard to either side steers the
 *   propagation as well as a centred one, and the common rotation preserves
 *   every inter-channel phase difference exactly.
 * - **Unambiguous instantaneous frequency.** A second transform, 16 samples
 *   after the first, measures each bin's frequency with a +-1500 Hz
 *   unambiguous range at 48 kHz; the hop-length phase difference, which is
 *   precise but wraps every sampleRate / Ra Hz, is then unwrapped around it.
 *   Long analysis hops (slow-downs to 0.5x at long frames) no longer alias a
 *   moving partial by a multiple of that wrap.
 * - **Strike-anchored time map.** A log-frequency spectral-flux detector
 *   (Boeck & Widmer, DAFx-13) runs `lookahead()` samples ahead of the
 *   analysis and locates each onset to the sample on a 32-sample energy
 *   envelope. The time map from stream to input is then bent so that it has
 *   slope 1 across the strike's window - every frame that sees the strike
 *   sees it at the offset the timeline puts it at - and rejoins the nominal
 *   line within about one frame on either side. The strike lands where the
 *   stretched timeline puts it, with the input's own attack.
 * - **Unrotated strike bins** (the phase reset of Duxbury, Davies and
 *   Sandler, AES 112th Convention, 2002, restricted to the bins a strike rises into). Inside
 *   an isolated strike's lock the bins rising by 3 dB or more take no
 *   rotation: the lock has slope 1, so they are a delayed copy of the input.
 *   The heap would give them one rigid but non-zero rotation, which adds
 *   the strike's Hilbert transform and its 1/t tail ahead of it.
 *
 * Coordinates. Input samples are counted from reset() (the first pushed
 * sample is input 0). The synthesis stream is indexed by writeHead(), which
 * advances Rs per frame and marks the end of the completed overlap-add. On the
 * nominal timeline, stream position `streamPositionOf(x)` carries input `x`.
 *
 * Threading: as detail::PhaseVocoderEngine - publishParams() from ONE control
 * thread through a bounded seqlock; everything else on the stream owner's
 * thread; prepare() on the setup thread.
 *
 * Dependencies: Core/FFT.h, Core/DspMath.h.
 */

#include "../../Core/DspMath.h"
#include "../../Core/FFT.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {
namespace detail {

/**
 * @class StudioVocoder
 * @brief Streaming phase-gradient vocoder with a strike-anchored time map.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class StudioVocoder
{
public:
    static_assert(std::atomic<double>::is_always_lock_free,
                  "StudioVocoder requires lock-free std::atomic<double>");
    static_assert(std::atomic<unsigned>::is_always_lock_free,
                  "StudioVocoder requires lock-free std::atomic<unsigned>");
    static_assert(std::atomic<bool>::is_always_lock_free,
                  "StudioVocoder requires lock-free std::atomic<bool>");

    /** @brief The control-published parameter set (adopted as ONE unit). */
    struct Params
    {
        double targetSemitones   = 0.0;    ///< Stretch ratio as semitones, clamped +-12.
        bool   transientPreserve = true;   ///< Anchor the time map on detected strikes.
        bool   formantPreserve   = false;  ///< Cepstral pre-warp (resample owners).
    };

    // -- Lifecycle (setup thread) ---------------------------------------------

    /**
     * @brief Allocates every buffer the stream will use.
     *
     * @param sampleRate           Sample rate in Hz.
     * @param numChannels          Channel count (>= 1).
     * @param fftSize              Frame size, power of two in [256, 1 << 20].
     * @param resampleCompensation True for owners that resample the stream by
     *                             the ratio afterwards (pitch shifting): enables
     *                             the anti-alias taper and the formant pre-warp.
     * @return false (nothing touched) on invalid arguments.
     */
    bool prepare(double sampleRate, int numChannels, int fftSize, bool resampleCompensation)
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate) || numChannels < 1
            || (fftSize & (fftSize - 1)) != 0 || fftSize < 256 || fftSize > (1 << 20))
            return false;

        prepared_ = false;
        sampleRate_ = sampleRate;
        numChannels_ = numChannels;
        N_ = fftSize;
        bins_ = N_ / 2 + 1;
        Rs_ = N_ / 4;
        resample_ = resampleCompensation;

        // Onset detector: a ~21 ms frame and an eighth-frame hop at every rate.
        No_ = 256;
        while (No_ < 8192 && static_cast<double>(No_) * 1.5 < 0.0213 * sampleRate_) No_ <<= 1;
        hopO_ = No_ / 8;
        // The detector confirms an onset one hop after its frame and locates
        // it at most No + 3 hops behind the newest input; the analysis stays
        // this far behind so that no frame has seen a strike before it is
        // planned for.
        D_ = No_ + No_ / 2;
        delta_ = std::min(16, std::max(1, N_ / 64));

        int ring = 1;
        while (ring < 2 * N_ + 2 * D_ + No_) ring <<= 1;
        ringSize_ = ring;
        ringMask_ = ring - 1;
        accumSize_ = 4 * N_;
        accumMask_ = accumSize_ - 1;

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(N_));
        fftO_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(No_));

        window_.resize(static_cast<size_t>(N_));
        for (int k = 0; k < N_; ++k)
            window_[static_cast<size_t>(k)] = static_cast<T>(std::sqrt(
                0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * k / N_)));
        windowO_.resize(static_cast<size_t>(No_));
        for (int k = 0; k < No_; ++k)
            windowO_[static_cast<size_t>(k)] = static_cast<T>(
                0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * k / No_));

        const auto nb = static_cast<size_t>(bins_);
        const auto spec = static_cast<size_t>(N_ + 2);
        ring_.assign(static_cast<size_t>(numChannels_), std::vector<T>(static_cast<size_t>(ringSize_), T(0)));
        accum_.assign(static_cast<size_t>(numChannels_), std::vector<T>(static_cast<size_t>(accumSize_), T(0)));
        spec_.assign(static_cast<size_t>(numChannels_), std::vector<T>(spec, T(0)));
        prevSpec_.assign(static_cast<size_t>(numChannels_), std::vector<T>(spec, T(0)));
        specD_.resize(spec);
        frame_.resize(static_cast<size_t>(N_));
        mag_.resize(nb);
        prevMag_.resize(nb);
        crossRe_.resize(nb); crossIm_.resize(nb);
        dRe_.resize(nb); dIm_.resize(nb);
        theta_.resize(nb);
        tinc_.resize(nb);
        omPrev_.resize(nb);
        done_.resize(nb);
        rotRe_.resize(nb); rotIm_.resize(nb);
        gain_.resize(nb);
        heapKey_.resize(2 * nb + 2);
        heapCode_.resize(2 * nb + 2);
        cepsTime_.resize(static_cast<size_t>(N_));
        cepsSpec_.resize(spec);
        envLog_.resize(nb);

        // Onset filterbank: quarter-tone triangles, 27.5 Hz to 16 kHz.
        buildOnsetBank();
        frameO_.resize(static_cast<size_t>(No_));
        specO_.resize(static_cast<size_t>(No_ + 2));
        powO_.resize(static_cast<size_t>(No_ / 2 + 1));
        bandCur_.assign(static_cast<size_t>(numBands_), T(0));
        bandPrevMax_.assign(static_cast<size_t>(numBands_), T(0));
        bandPrev_.assign(static_cast<size_t>(numBands_), T(0));

        prepared_ = true;
        return true;
    }

    /** @brief Clears all signal state and adopts the latest parameters
     *  (jump, no glide). Stream owner only. */
    void reset() noexcept
    {
        if (!prepared_) return;
        for (auto& r : ring_) std::fill(r.begin(), r.end(), T(0));
        for (auto& a : accum_) std::fill(a.begin(), a.end(), T(0));
        for (auto& s : prevSpec_) std::fill(s.begin(), s.end(), T(0));
        std::fill(prevMag_.begin(), prevMag_.end(), T(0));
        std::fill(theta_.begin(), theta_.end(), 0.0);
        for (int k = 0; k < bins_; ++k)
            omPrev_[static_cast<size_t>(k)] = kTwoPi * k / N_;

        adoptParamsIfDirty();
        stActive_ = std::clamp(targetSemitones_, -12.0, 12.0);
        ratio_ = std::exp2(stActive_ / 12.0);

        inCount_ = 0;
        frameStart_ = static_cast<int64_t>(accumSize_);
        firstFrame_ = true;
        // Frame 0 covers input [Rs - N, Rs): it becomes ready after Rs samples
        // plus the lookahead, like the first hop of a classic vocoder.
        aNext_ = static_cast<double>(Rs_) - 0.5 * N_;
        aNom_ = aNext_;
        aCur_ = aNext_;
        cCur_ = static_cast<double>(frameStart_) + 0.5 * N_ - Rs_;
        origin_ = aNext_;
        originStream_ = static_cast<double>(frameStart_) + 0.5 * N_;
        originRatio_ = ratio_;
        prevAnalysisStart_ = 0;
        steer_ = false;

        lockHead_ = lockCount_ = 0;
        rejoinE_ = 0.0; rejoinC_ = 0.0; rejoinJ_ = 1.0;
        lastLockS_ = -1e18; lastLockT_ = -1e18;

        // Detector.
        nextFrameO_ = 0;
        std::fill(bandPrev_.begin(), bandPrev_.end(), T(0));
        std::fill(bandPrevMax_.begin(), bandPrevMax_.end(), T(0));
        fluxFill_ = 0; fluxPos_ = 0;
        fluxPrev_ = 0.0; fluxPrev2_ = 0.0; candPrev_ = false; candFrame_ = -1;
        lastOnsetFrame_ = -1000;
        onsetRing_.fill(-1e18);
        onsetPos_ = 0;
    }

    // -- Parameters (control thread) ------------------------------------------

    /** @brief Publishes a whole parameter set (one control thread only). */
    void publishParams(const Params& p) noexcept
    {
        seq_.fetch_add(1, std::memory_order_acq_rel);
        std::atomic_thread_fence(std::memory_order_release);
        stgSemitones_.store(p.targetSemitones, std::memory_order_relaxed);
        stgTransient_.store(p.transientPreserve, std::memory_order_relaxed);
        stgFormant_.store(p.formantPreserve, std::memory_order_relaxed);
        seq_.fetch_add(1, std::memory_order_release);
        dirty_.store(true, std::memory_order_release);
    }

    // -- Streaming (stream owner) -----------------------------------------------

    /** @return Input samples to push before the next frame can run (0 = the
     *  next commitInput() runs it). */
    [[nodiscard]] int samplesToNextHop() const noexcept
    {
        const int64_t need = analysisStart() + N_ + D_ - inCount_;
        return static_cast<int>(std::max<int64_t>(0, need));
    }

    /** @brief Writes `count` (<= samplesToNextHop(), or any count when it is
     *  0... see commitInput) samples of channel `ch` at the input head. */
    void pushInput(int ch, const T* src, int count) noexcept
    {
        auto& r = ring_[static_cast<size_t>(ch)];
        int64_t p = inCount_;
        for (int k = 0; k < count; ++k, ++p)
            r[static_cast<size_t>(p & ringMask_)] = src[k];
    }

    /**
     * @brief Advances the input head by `count` and runs at most one frame.
     *
     * The onset detector consumes the new input first, so a strike it finds
     * is planned before the frame that would reach it is analysed.
     */
    void commitInput(int count, int numActiveChannels) noexcept
    {
        inCount_ += count;
        runDetector(numActiveChannels);
        if (inCount_ >= analysisStart() + N_ + D_)
            processFrame(numActiveChannels);
    }

    /** @brief Owner-supplied nominal analysis centre for the NEXT frame (a
     *  resampling owner keeps its reader and the timeline in step with it). */
    void steerTimeline(double nominalCentre) noexcept
    {
        steerValue_ = nominalCentre;
        steer_ = true;
    }

    /**
     * @brief Caps how far a strike anchor may move the analysis ahead of the
     *        nominal timeline, in input samples (owner with a fixed latency
     *        budget; unlimited by default).
     *
     * A lock of half-width h runs h (1/r - 1) ahead of the timeline as it
     * starts below ratio 1, and h (1 - 1/r) ahead as it ends above it; the
     * cap narrows the lock instead of letting a frame need input the owner's
     * latency does not cover.
     */
    void setAnchorLeadLimit(double samples) noexcept { leadLimit_ = std::max(0.0, samples); }

    /** @return The glided active ratio (stretch factor, stream per input). */
    [[nodiscard]] double activeRatio() const noexcept { return ratio_; }

    /** @return Nominal analysis centre (input coordinate) of the next frame. */
    [[nodiscard]] double nextNominalCentre() const noexcept { return aNom_; }

    /** @return Stream centre of the next frame. */
    [[nodiscard]] double nextStreamCentre() const noexcept
    {
        return static_cast<double>(frameStart_) + 0.5 * N_;
    }

    /** @return Stream position of input `x` on the timeline in force at reset. */
    [[nodiscard]] double streamPositionOf(double x) const noexcept
    {
        return originStream_ + originRatio_ * (x - origin_);
    }

    [[nodiscard]] const T* olaData(int ch) const noexcept { return accum_[static_cast<size_t>(ch)].data(); }
    [[nodiscard]] int64_t olaMask() const noexcept { return static_cast<int64_t>(accumMask_); }
    [[nodiscard]] int olaSize() const noexcept { return accumSize_; }
    /** @return End of the completed overlap-add (advances Rs per frame). */
    [[nodiscard]] int64_t writeHead() const noexcept { return frameStart_; }
    [[nodiscard]] int fftSize() const noexcept { return prepared_ ? N_ : 0; }
    [[nodiscard]] int synthHop() const noexcept { return Rs_; }
    /** @return How far (samples) the analysis trails the newest input. */
    [[nodiscard]] int lookahead() const noexcept { return D_; }
    /** @return Input samples committed since reset(). */
    [[nodiscard]] int64_t inputCount() const noexcept { return inCount_; }

private:
    static constexpr double kTwoPi = 2.0 * std::numbers::pi;
    static constexpr int kSeqlockMaxAttempts = 3;
    static constexpr int kMaxLocks = 32;
    /// Bins further than this below the frame's loudest are left on their own
    /// time propagation; nothing audible depends on how they are ordered.
    static constexpr double kHeapFloorDb = 100.0;
    /// Amplitude rise over the previous frame (3 dB) that marks a bin as one
    /// a strike is entering, inside the strike's lock.
    static constexpr T kRiseRatio = static_cast<T>(1.4142135623730951);
    /// Narrowest lock, in frames, whose strike bins are copied unrotated.
    static constexpr double kResetCover = 0.375;
    /// Onset threshold above the running median of the flux history.
    static constexpr double kFluxDelta = 0.025;
    static constexpr int kFluxWindow = 12;

    struct Lock { double s, target, h; };

    [[nodiscard]] static double princArg(double x) noexcept
    {
        return x - kTwoPi * std::round(x / kTwoPi);
    }

    [[nodiscard]] int64_t analysisStart() const noexcept
    {
        return static_cast<int64_t>(std::llround(aNext_)) - N_ / 2;
    }

    /** @brief Adopts the staged parameter set, or defers (bounded seqlock read).
     *  After kSeqlockMaxAttempts torn or mid-publish reads the previously
     *  adopted set stays in force and the dirty flag is re-armed, so the
     *  audio thread never waits on the control thread. */
    void adoptParamsIfDirty() noexcept
    {
        if (!dirty_.exchange(false, std::memory_order_acquire)) return;
        for (int attempt = 0; attempt < kSeqlockMaxAttempts; ++attempt)
        {
            const unsigned s0 = seq_.load(std::memory_order_acquire);
            if ((s0 & 1u) != 0u) continue;   // writer mid-publish: do not copy
            const double st = stgSemitones_.load(std::memory_order_relaxed);
            const bool tr = stgTransient_.load(std::memory_order_relaxed);
            const bool fo = stgFormant_.load(std::memory_order_relaxed);
            // Orders the copy above before the re-read below and pairs with
            // the writer's release fence ([atomics.fences]/2).
            std::atomic_thread_fence(std::memory_order_acquire);
            if (s0 == seq_.load(std::memory_order_relaxed))
            {
                targetSemitones_ = st; transientOn_ = tr; formantOn_ = fo;
                return;
            }
        }
        dirty_.store(true, std::memory_order_release);
    }

    // -- Frame ------------------------------------------------------------------

    void analyse(int ch, int64_t start, T* out) noexcept
    {
        const auto& r = ring_[static_cast<size_t>(ch)];
        for (int k = 0; k < N_; ++k)
            frame_[static_cast<size_t>(k)] = r[static_cast<size_t>((start + k) & ringMask_)]
                                           * window_[static_cast<size_t>(k)];
        fft_->forward(frame_.data(), out);
    }

    void processFrame(int nCh) noexcept
    {
        nCh = std::clamp(nCh, 1, numChannels_);
        const int64_t A = analysisStart();
        const double Ra = firstFrame_ ? static_cast<double>(Rs_)
                                      : static_cast<double>(A - prevAnalysisStart_);
        const double binW = kTwoPi / N_;
        const double Rs = static_cast<double>(Rs_);

        // --- analysis: magnitudes, time and delta cross-spectra ----------------
        std::fill(mag_.begin(), mag_.end(), T(0));
        std::fill(crossRe_.begin(), crossRe_.end(), 0.0);
        std::fill(crossIm_.begin(), crossIm_.end(), 0.0);
        std::fill(dRe_.begin(), dRe_.end(), 0.0);
        std::fill(dIm_.begin(), dIm_.end(), 0.0);
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* X = spec_[static_cast<size_t>(ch)].data();
            const T* P = prevSpec_[static_cast<size_t>(ch)].data();
            analyse(ch, A, X);
            analyse(ch, A + delta_, specD_.data());
            for (int k = 0; k < bins_; ++k)
            {
                const double xr = X[2 * k], xi = X[2 * k + 1];
                const double pr = P[2 * k], pi = P[2 * k + 1];
                const double dr = specD_[static_cast<size_t>(2 * k)];
                const double di = specD_[static_cast<size_t>(2 * k + 1)];
                mag_[static_cast<size_t>(k)] += static_cast<T>(xr * xr + xi * xi);
                crossRe_[static_cast<size_t>(k)] += xr * pr + xi * pi;
                crossIm_[static_cast<size_t>(k)] += xi * pr - xr * pi;
                dRe_[static_cast<size_t>(k)] += dr * xr + di * xi;
                dIm_[static_cast<size_t>(k)] += di * xr - dr * xi;
            }
        }
        T maxMag = T(0);
        for (int k = 0; k < bins_; ++k)
        {
            auto& m = mag_[static_cast<size_t>(k)];
            m = std::sqrt(m);
            maxMag = std::max(maxMag, m);
        }

        // --- time-direction candidates -------------------------------------------
        const double dl = static_cast<double>(delta_);
        for (int k = 0; k < bins_; ++k)
        {
            const double wk = binW * k;
            const double omNow = wk + princArg(std::atan2(dIm_[static_cast<size_t>(k)],
                                                          dRe_[static_cast<size_t>(k)]) - wk * dl) / dl;
            double inc = 0.0;
            if (!firstFrame_)
            {
                const double dphi = std::atan2(crossIm_[static_cast<size_t>(k)],
                                               crossRe_[static_cast<size_t>(k)]);
                const double prior = 0.5 * (omNow + omPrev_[static_cast<size_t>(k)]);
                const double om = Ra >= 1.0 ? prior + princArg(dphi - prior * Ra) / Ra : prior;
                inc = princArg(theta_[static_cast<size_t>(k)] + Rs * om - dphi);
            }
            tinc_[static_cast<size_t>(k)] = inc;
            omPrev_[static_cast<size_t>(k)] = omNow;
        }

        // --- heap-ordered propagation ----------------------------------------------
        if (firstFrame_)
            std::fill(theta_.begin(), theta_.end(), 0.0);
        else
            propagate(maxMag);

        // Inside a strike's lock the map has slope 1, so the frame is placed
        // exactly where a plain delayed copy of the input would put it. The
        // bins the strike is rising into take no rotation at all: the heap
        // hands them one rotation, rigid but not zero, and a constant phase
        // turn of an impulse is cos(phi) times the impulse plus sin(phi)
        // times its Hilbert transform, whose 1/t tail reaches before the
        // strike - measured as pre-echo 30 dB down and up to 15 ms long at
        // ratio 1.25. Zero rotation copies the strike itself; the bins that
        // are not rising keep their propagated phases, so a partial that
        // sustains through the strike stays continuous. Two cases are left
        // to the heap. A lock narrower than the window leaves frames outside
        // it that still see the strike, and a zero-rotated copy makes their
        // misplaced copies coherent too (at 960 strikes per minute, 4096
        // frame and ratio 0.8 the median strike moved from 0.08 to 7.9 ms).
        // And a strike with another one within a frame of it would share
        // frames with it, and the copy would carry that one at this lock's
        // offset rather than at its own place.
        if (transientOn_ && !firstFrame_ && lockCount_ > 0)
        {
            const Lock& L = locks_[static_cast<size_t>(lockHead_)];
            const double c = static_cast<double>(frameStart_) + 0.5 * N_;
            bool alone = L.h >= kResetCover * N_;
            for (const double o : onsetRing_)
                if (std::abs(o - L.s) > 1.0 && std::abs(o - L.s) < static_cast<double>(N_))
                    alone = false;
            if (alone && c >= L.target - L.h && c <= L.target + L.h)
                for (int k = 0; k < bins_; ++k)
                    if (mag_[static_cast<size_t>(k)] > kRiseRatio * prevMag_[static_cast<size_t>(k)])
                        theta_[static_cast<size_t>(k)] = 0.0;
        }
        theta_[0] = 0.0;
        theta_[static_cast<size_t>(bins_ - 1)] = 0.0;
        for (int k = 0; k < bins_; ++k)
        {
            rotRe_[static_cast<size_t>(k)] = static_cast<T>(std::cos(theta_[static_cast<size_t>(k)]));
            rotIm_[static_cast<size_t>(k)] = static_cast<T>(std::sin(theta_[static_cast<size_t>(k)]));
        }

        // --- magnitude post-stages for resampling owners --------------------------
        const bool formant = resample_ && formantOn_ && std::abs(ratio_ - 1.0) > 1e-6;
        const bool taper = resample_ && ratio_ > 1.0;
        if (formant) computeFormantGains();
        else std::fill(gain_.begin(), gain_.end(), T(1));
        if (taper)
        {
            const int cut = static_cast<int>(static_cast<double>(N_ / 2) / ratio_);
            const int taperStart = std::max(1, cut - 4);
            for (int k = taperStart; k < bins_; ++k)
                gain_[static_cast<size_t>(k)] *= (k <= cut)
                    ? static_cast<T>(cut - k + 1) / static_cast<T>(cut - taperStart + 1) : T(0);
        }

        // --- synthesis ---------------------------------------------------------------
        const T norm = static_cast<T>(Rs / (0.5 * N_));
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* X = spec_[static_cast<size_t>(ch)].data();
            std::copy(X, X + N_ + 2, prevSpec_[static_cast<size_t>(ch)].data());
            for (int k = 0; k < bins_; ++k)
            {
                const T re = X[2 * k], im = X[2 * k + 1];
                const T rr = rotRe_[static_cast<size_t>(k)] * gain_[static_cast<size_t>(k)];
                const T ri = rotIm_[static_cast<size_t>(k)] * gain_[static_cast<size_t>(k)];
                X[2 * k] = re * rr - im * ri;
                X[2 * k + 1] = re * ri + im * rr;
            }
            X[1] = T(0);
            X[N_ + 1] = T(0);
            fft_->inverse(X, frame_.data());
            auto& acc = accum_[static_cast<size_t>(ch)];
            for (int k = N_ - Rs_; k < N_; ++k)
                acc[static_cast<size_t>((frameStart_ + k) & accumMask_)] = T(0);
            for (int k = 0; k < N_; ++k)
                acc[static_cast<size_t>((frameStart_ + k) & accumMask_)]
                    += frame_[static_cast<size_t>(k)] * window_[static_cast<size_t>(k)] * norm;
        }
        for (int ch = nCh; ch < numChannels_; ++ch)
            std::fill(prevSpec_[static_cast<size_t>(ch)].begin(),
                      prevSpec_[static_cast<size_t>(ch)].end(), T(0));
        std::copy(mag_.begin(), mag_.end(), prevMag_.begin());

        prevAnalysisStart_ = A;
        firstFrame_ = false;
        aCur_ = aNext_;
        cCur_ = static_cast<double>(frameStart_) + 0.5 * N_;
        frameStart_ += Rs_;
        planNext();
    }

    /**
     * @brief Heap integration over the previous and current frame.
     *
     * Codes: 2k = bin k of the previous frame, 2k+1 = bin k of this frame.
     */
    void propagate(T maxMag) noexcept
    {
        const T tol = maxMag * static_cast<T>(std::pow(10.0, -kHeapFloorDb / 20.0));
        int todo = 0;
        for (int k = 0; k < bins_; ++k)
        {
            const bool quiet = mag_[static_cast<size_t>(k)] < tol;
            done_[static_cast<size_t>(k)] = quiet ? 1 : 0;
            theta_[static_cast<size_t>(k)] = tinc_[static_cast<size_t>(k)];
            if (!quiet) ++todo;
        }
        heapSize_ = 0;
        for (int k = 0; k < bins_; ++k)
            if (prevMag_[static_cast<size_t>(k)] > tol)
                heapPush(prevMag_[static_cast<size_t>(k)], 2 * k);

        while (todo > 0)
        {
            if (heapSize_ == 0)
            {
                int best = -1; T bm = T(-1);
                for (int k = 0; k < bins_; ++k)
                    if (!done_[static_cast<size_t>(k)] && mag_[static_cast<size_t>(k)] > bm)
                    { bm = mag_[static_cast<size_t>(k)]; best = k; }
                done_[static_cast<size_t>(best)] = 1; --todo;
                heapPush(mag_[static_cast<size_t>(best)], 2 * best + 1);
                continue;
            }
            const int code = heapPop();
            const int k = code >> 1;
            if ((code & 1) == 0)
            {
                if (!done_[static_cast<size_t>(k)])
                {
                    done_[static_cast<size_t>(k)] = 1; --todo;   // theta = tinc already
                    heapPush(mag_[static_cast<size_t>(k)], 2 * k + 1);
                }
            }
            else
            {
                const double th = theta_[static_cast<size_t>(k)];
                if (k + 1 < bins_ && !done_[static_cast<size_t>(k + 1)])
                {
                    done_[static_cast<size_t>(k + 1)] = 1; --todo;
                    theta_[static_cast<size_t>(k + 1)] = th;
                    heapPush(mag_[static_cast<size_t>(k + 1)], 2 * (k + 1) + 1);
                }
                if (k > 0 && !done_[static_cast<size_t>(k - 1)])
                {
                    done_[static_cast<size_t>(k - 1)] = 1; --todo;
                    theta_[static_cast<size_t>(k - 1)] = th;
                    heapPush(mag_[static_cast<size_t>(k - 1)], 2 * (k - 1) + 1);
                }
            }
        }
    }

    void heapPush(T key, int code) noexcept
    {
        int i = heapSize_++;
        while (i > 0)
        {
            const int p = (i - 1) >> 1;
            if (heapKey_[static_cast<size_t>(p)] >= key) break;
            heapKey_[static_cast<size_t>(i)] = heapKey_[static_cast<size_t>(p)];
            heapCode_[static_cast<size_t>(i)] = heapCode_[static_cast<size_t>(p)];
            i = p;
        }
        heapKey_[static_cast<size_t>(i)] = key;
        heapCode_[static_cast<size_t>(i)] = code;
    }

    int heapPop() noexcept
    {
        const int top = heapCode_[0];
        const T key = heapKey_[static_cast<size_t>(heapSize_ - 1)];
        const int code = heapCode_[static_cast<size_t>(heapSize_ - 1)];
        --heapSize_;
        int i = 0;
        for (;;)
        {
            int c = 2 * i + 1;
            if (c >= heapSize_) break;
            if (c + 1 < heapSize_ && heapKey_[static_cast<size_t>(c + 1)] > heapKey_[static_cast<size_t>(c)]) ++c;
            if (heapKey_[static_cast<size_t>(c)] <= key) break;
            heapKey_[static_cast<size_t>(i)] = heapKey_[static_cast<size_t>(c)];
            heapCode_[static_cast<size_t>(i)] = heapCode_[static_cast<size_t>(c)];
            i = c;
        }
        if (heapSize_ > 0)
        {
            heapKey_[static_cast<size_t>(i)] = key;
            heapCode_[static_cast<size_t>(i)] = code;
        }
        return top;
    }

    /** @brief Cepstral envelope of the reference magnitude and the
     *  env(k * ratio) / env(k) pre-warp gains. */
    void computeFormantGains() noexcept
    {
        for (int k = 0; k < bins_; ++k)
            cepsTime_[static_cast<size_t>(k)] = std::log(mag_[static_cast<size_t>(k)] + T(1e-9));
        for (int k = bins_; k < N_; ++k)
            cepsTime_[static_cast<size_t>(k)] = cepsTime_[static_cast<size_t>(N_ - k)];
        fft_->forward(cepsTime_.data(), cepsSpec_.data());
        const int keep = std::min(std::max(8, static_cast<int>(0.001 * sampleRate_)), bins_ - 1);
        for (int k = keep + 1; k < bins_; ++k)
        {
            cepsSpec_[static_cast<size_t>(2 * k)] = T(0);
            cepsSpec_[static_cast<size_t>(2 * k + 1)] = T(0);
        }
        fft_->inverse(cepsSpec_.data(), cepsTime_.data());
        for (int k = 0; k < bins_; ++k)
            envLog_[static_cast<size_t>(k)] = cepsTime_[static_cast<size_t>(k)];
        for (int k = 0; k < bins_; ++k)
        {
            const double pos = std::min(static_cast<double>(k) * ratio_, static_cast<double>(bins_ - 1));
            const auto i0 = static_cast<int>(pos);
            const auto fr = static_cast<T>(pos - i0);
            const int i1 = std::min(i0 + 1, bins_ - 1);
            const T target = envLog_[static_cast<size_t>(i0)]
                           + (envLog_[static_cast<size_t>(i1)] - envLog_[static_cast<size_t>(i0)]) * fr;
            gain_[static_cast<size_t>(k)] = std::exp(std::clamp(
                target - envLog_[static_cast<size_t>(k)], T(-4.6), T(4.6)));
        }
    }

    // -- Time map -------------------------------------------------------------

    /** @brief Chooses the analysis centre of the next frame. */
    void planNext() noexcept
    {
        adoptParamsIfDirty();
        const double stTarget = std::clamp(targetSemitones_, -12.0, 12.0);
        stActive_ += std::clamp(stTarget - stActive_, -0.5, 0.5);
        ratio_ = std::exp2(stActive_ / 12.0);
        const double r = ratio_;

        const double c = static_cast<double>(frameStart_) + 0.5 * N_;
        if (steer_) { aNom_ = steerValue_; steer_ = false; }
        else aNom_ += static_cast<double>(Rs_) / r;

        // Retire locks the stream has passed; the map then rejoins the
        // nominal line from where the lock left it.
        while (lockCount_ > 0)
        {
            const Lock& L = locks_[static_cast<size_t>(lockHead_)];
            if (c < L.target + L.h) break;
            rejoinC_ = L.target + L.h;
            rejoinE_ = (L.s + L.h) - (aNom_ + (rejoinC_ - c) / r);
            rejoinJ_ = std::max(static_cast<double>(N_), 2.0 * L.h * std::abs(r - 1.0));
            lockHead_ = (lockHead_ + 1) % kMaxLocks;
            --lockCount_;
        }

        double a;
        if (lockCount_ > 0)
        {
            const Lock& L = locks_[static_cast<size_t>(lockHead_)];
            if (c >= L.target - L.h)
                a = L.s + (c - L.target);
            else
            {
                const double span = (L.target - L.h) - cCur_;
                const double f = span > 0.0 ? (c - cCur_) / span : 1.0;
                a = aCur_ + ((L.s - L.h) - aCur_) * std::clamp(f, 0.0, 1.0);
            }
        }
        else
        {
            const double f = std::clamp(1.0 - (c - rejoinC_) / rejoinJ_, 0.0, 1.0);
            a = aNom_ + rejoinE_ * f;
        }
        aNext_ = std::max(a, aCur_);
    }

    /** @brief Plans a lock on a strike at input `s` (called by the detector). */
    void addOnset(double s) noexcept
    {
        onsetRing_[static_cast<size_t>(onsetPos_)] = s;
        onsetPos_ = (onsetPos_ + 1) % kOnsetRing;
        if (!transientOn_ || lockCount_ >= kMaxLocks) return;
        const double r = ratio_;
        const double cNext = static_cast<double>(frameStart_) + 0.5 * N_;
        const double target = cNext + (s - aNom_) * r;
        double h = 0.5 * N_;
        h = std::min(h, s - aCur_ - 1.0);                 // the map never runs backwards
        h = std::min(h, target - cNext);                        // the lock starts at or after the next frame
        if (r < 1.0) h = std::min(h, leadLimit_ / (1.0 / r - 1.0));
        else if (r > 1.0) h = std::min(h, leadLimit_ / (1.0 - 1.0 / r));
        double prevS = lastLockS_, prevT = lastLockT_, prevH = 0.0;
        if (lockCount_ > 0)
        {
            const Lock& P = locks_[static_cast<size_t>((lockHead_ + lockCount_ - 1) % kMaxLocks)];
            prevS = P.s; prevT = P.target; prevH = P.h;
        }
        h = std::min(h, (s - prevS) - prevH - static_cast<double>(Rs_));
        h = std::min(h, (target - prevT) - prevH - static_cast<double>(Rs_));
        if (!(h >= 0.5 * Rs_)) return;
        locks_[static_cast<size_t>((lockHead_ + lockCount_) % kMaxLocks)] = Lock { s, target, h };
        ++lockCount_;
        lastLockS_ = s; lastLockT_ = target;
        // Re-plan the next frame now that it may be on the approach.
        const double keepNom = aNom_;
        replanNextFrame(keepNom);
    }

    void replanNextFrame(double nom) noexcept
    {
        const double c = static_cast<double>(frameStart_) + 0.5 * N_;
        const Lock& L = locks_[static_cast<size_t>(lockHead_)];
        double a;
        if (c >= L.target - L.h)
            a = L.s + (c - L.target);
        else
        {
            const double span = (L.target - L.h) - cCur_;
            const double f = span > 0.0 ? (c - cCur_) / span : 1.0;
            a = aCur_ + ((L.s - L.h) - aCur_) * std::clamp(f, 0.0, 1.0);
        }
        (void) nom;
        // Never ask for input the ring no longer holds or the frame already
        // passed; never run backwards.
        aNext_ = std::max(a, aCur_);
    }

    // -- Onset detector -----------------------------------------------------------

    void buildOnsetBank()
    {
        fbStart_.clear(); fbCount_.clear(); fbOffset_.clear(); fbW_.clear();
        const double binHz = sampleRate_ / No_;
        const int nb = No_ / 2 + 1;
        const double fMax = std::min(16000.0, 0.5 * sampleRate_ * 0.999);
        std::vector<int> centres;
        for (int i = 0; ; ++i)
        {
            const double f = 27.5 * std::pow(2.0, i / 24.0);
            if (f > fMax) break;
            const int b = std::clamp(static_cast<int>(std::lround(f / binHz)), 0, nb - 1);
            if (centres.empty() || b > centres.back()) centres.push_back(b);
        }
        for (size_t j = 1; j + 1 < centres.size(); ++j)
        {
            const int lo = centres[j - 1], ce = centres[j], hi = centres[j + 1];
            fbStart_.push_back(lo);
            fbOffset_.push_back(static_cast<int>(fbW_.size()));
            for (int k = lo; k <= hi; ++k)
                fbW_.push_back(static_cast<T>(k <= ce ? double(k - lo) / (ce - lo)
                                                       : double(hi - k) / (hi - ce)));
            fbCount_.push_back(hi - lo + 1);
        }
        numBands_ = static_cast<int>(fbStart_.size());
        odfScale_ = static_cast<T>(2048.0 / No_);
    }

    /** @brief Runs every detector frame the new input completes. */
    void runDetector(int nCh) noexcept
    {
        nCh = std::clamp(nCh, 1, numChannels_);
        while (nextFrameO_ * hopO_ + No_ <= inCount_)
        {
            const int64_t start = nextFrameO_ * hopO_;
            std::fill(powO_.begin(), powO_.end(), T(0));
            for (int ch = 0; ch < nCh; ++ch)
            {
                const auto& r = ring_[static_cast<size_t>(ch)];
                for (int k = 0; k < No_; ++k)
                    frameO_[static_cast<size_t>(k)] = r[static_cast<size_t>((start + k) & ringMask_)]
                                                    * windowO_[static_cast<size_t>(k)];
                fftO_->forward(frameO_.data(), specO_.data());
                for (int k = 0; k <= No_ / 2; ++k)
                {
                    const T re = specO_[static_cast<size_t>(2 * k)], im = specO_[static_cast<size_t>(2 * k + 1)];
                    powO_[static_cast<size_t>(k)] += re * re + im * im;
                }
            }
            const T inv = T(1) / static_cast<T>(nCh);
            double flux = 0.0;
            for (int b = 0; b < numBands_; ++b)
            {
                T acc = T(0);
                const int s0 = fbStart_[static_cast<size_t>(b)], off = fbOffset_[static_cast<size_t>(b)];
                for (int i = 0; i < fbCount_[static_cast<size_t>(b)]; ++i)
                    acc += std::sqrt(powO_[static_cast<size_t>(s0 + i)] * inv) * fbW_[static_cast<size_t>(off + i)];
                const T v = std::log10(acc * odfScale_ + T(1));
                bandCur_[static_cast<size_t>(b)] = v;
                const double d = static_cast<double>(v - bandPrevMax_[static_cast<size_t>(b)]);
                if (d > 0.0) flux += d;
            }
            flux /= std::max(1, numBands_);
            for (int b = 0; b < numBands_; ++b)
            {
                T mx = bandCur_[static_cast<size_t>(b)];
                if (b > 0) mx = std::max(mx, bandCur_[static_cast<size_t>(b - 1)]);
                if (b + 1 < numBands_) mx = std::max(mx, bandCur_[static_cast<size_t>(b + 1)]);
                bandPrevMax_[static_cast<size_t>(b)] = mx;
            }
            if (nextFrameO_ == 0) flux = 0.0;   // no previous frame to rise from

            // The previous frame is an onset if it rose past the median of the
            // frames before it and is a local maximum of the flux.
            if (candPrev_ && fluxPrev_ >= flux && candFrame_ - lastOnsetFrame_ > 4)
            {
                lastOnsetFrame_ = candFrame_;
                const double s = locateOnset(candFrame_, nCh);
                if (s >= 0.0) addOnset(s);
            }

            double med = 0.0;
            if (fluxFill_ > 3)
            {
                const int n = std::min(fluxFill_, kFluxWindow);
                for (int i = 0; i < n; ++i) medScratch_[static_cast<size_t>(i)] = fluxHist_[static_cast<size_t>(i)];
                std::nth_element(medScratch_.begin(), medScratch_.begin() + n / 2, medScratch_.begin() + n);
                med = medScratch_[static_cast<size_t>(n / 2)];
                if ((n & 1) == 0)
                {
                    const double lo = *std::max_element(medScratch_.begin(), medScratch_.begin() + n / 2);
                    med = 0.5 * (med + lo);
                }
            }
            candPrev_ = (flux - med > kFluxDelta) && flux >= fluxPrev_;
            candFrame_ = nextFrameO_;
            fluxHist_[static_cast<size_t>(fluxPos_)] = flux;
            fluxPos_ = (fluxPos_ + 1) % kFluxWindow;
            ++fluxFill_;
            fluxPrev_ = flux;
            ++nextFrameO_;
        }
    }

    /** @brief Sample-accurate onset inside detector frame `i`: the first
     *  crossing of a quarter of the local peak of a 32-sample RMS envelope. */
    [[nodiscard]] double locateOnset(int64_t i, int nCh) const noexcept
    {
        const int64_t lo = std::max<int64_t>({ 16, i * hopO_ - 2 * hopO_, inCount_ - ringSize_ + 64 });
        const int64_t hi = std::min<int64_t>(i * hopO_ + (3 * No_) / 4 + hopO_, inCount_ - 16);
        if (hi - lo < 8) return -1.0;
        auto pw = [&](int64_t n) {
            double e = 0.0;
            for (int ch = 0; ch < nCh; ++ch)
            {
                const double v = ring_[static_cast<size_t>(ch)][static_cast<size_t>(n & ringMask_)];
                e += v * v;
            }
            return e;
        };
        // Running 32-sample window centred on n: [n - 16, n + 16).
        double run = 0.0;
        for (int64_t n = lo - 16; n < lo + 16; ++n) run += pw(n);
        double peak = -1.0; int64_t pk = lo;
        // First pass: peak.
        {
            double rr = run;
            for (int64_t n = lo; n < hi; ++n)
            {
                if (rr > peak) { peak = rr; pk = n; }
                rr += pw(n + 16) - pw(n - 16);
            }
        }
        if (peak <= 0.0) return -1.0;
        // Walk back from the peak to the first sample at or below a quarter
        // of it in amplitude (1/16 in power).
        double rr = 0.0;
        for (int64_t n = pk - 16; n < pk + 16; ++n) rr += pw(n);
        int64_t j = pk;
        while (j > lo && rr > peak / 16.0)
        {
            rr += pw(j - 17) - pw(j + 15);
            --j;
        }
        return static_cast<double>(j);
    }

    // -- Members ------------------------------------------------------------------
    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    bool prepared_ = false;
    bool resample_ = false;
    int N_ = 4096, bins_ = 2049, Rs_ = 1024;
    int D_ = 1536, delta_ = 16;
    int ringSize_ = 0, ringMask_ = 0, accumSize_ = 0, accumMask_ = 0;

    std::unique_ptr<FFTReal<T>> fft_, fftO_;
    std::vector<T> window_, windowO_;
    std::vector<std::vector<T>> ring_, accum_, spec_, prevSpec_;
    std::vector<T> specD_, frame_, mag_, prevMag_, rotRe_, rotIm_, gain_;
    std::vector<double> crossRe_, crossIm_, dRe_, dIm_, theta_, tinc_, omPrev_;
    std::vector<unsigned char> done_;
    std::vector<T> heapKey_;
    std::vector<int> heapCode_;
    int heapSize_ = 0;
    std::vector<T> cepsTime_, cepsSpec_, envLog_;

    // Stream state.
    int64_t inCount_ = 0;
    int64_t frameStart_ = 0;
    int64_t prevAnalysisStart_ = 0;
    bool firstFrame_ = true;
    double stActive_ = 0.0, ratio_ = 1.0;
    double aNext_ = 0.0, aNom_ = 0.0, aCur_ = 0.0, cCur_ = 0.0;
    double origin_ = 0.0, originStream_ = 0.0, originRatio_ = 1.0;
    bool steer_ = false;
    double steerValue_ = 0.0;
    double leadLimit_ = 1e18;

    // Time-map locks.
    std::array<Lock, kMaxLocks> locks_ {};
    int lockHead_ = 0, lockCount_ = 0;
    double rejoinE_ = 0.0, rejoinC_ = 0.0, rejoinJ_ = 1.0;
    double lastLockS_ = -1e18, lastLockT_ = -1e18;

    // Onset detector.
    int No_ = 1024, hopO_ = 128, numBands_ = 0;
    T odfScale_ = T(2);
    std::vector<int> fbStart_, fbCount_, fbOffset_;
    std::vector<T> fbW_, frameO_, specO_, powO_, bandCur_, bandPrev_, bandPrevMax_;
    std::array<double, kFluxWindow> fluxHist_ {};
    std::array<double, kFluxWindow> medScratch_ {};
    int fluxFill_ = 0, fluxPos_ = 0;
    int64_t nextFrameO_ = 0, candFrame_ = -1, lastOnsetFrame_ = -1000;
    static constexpr int kOnsetRing = 8;
    std::array<double, kOnsetRing> onsetRing_ {};   ///< Recent onsets, locked or not.
    int onsetPos_ = 0;
    double fluxPrev_ = 0.0, fluxPrev2_ = 0.0;
    bool candPrev_ = false;

    // Parameters.
    double targetSemitones_ = 0.0;
    bool transientOn_ = true, formantOn_ = false;
    std::atomic<double> stgSemitones_ { 0.0 };
    std::atomic<bool> stgTransient_ { true };
    std::atomic<bool> stgFormant_ { false };
    std::atomic<unsigned> seq_ { 0 };
    std::atomic<bool> dirty_ { false };
};

} // namespace detail
} // namespace dspark
