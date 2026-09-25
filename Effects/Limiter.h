// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Limiter.h
 * @brief Brickwall lookahead limiter with optional ISP (true-peak) detection
 *        and adaptive release for mastering.
 *
 * A peak limiter that prevents audio from exceeding a configurable ceiling.
 * The gain computer is the modern lookahead design: the gain each sample
 * requires (ceiling / peak) enters an exact sliding-window minimum spanning
 * the lookahead, a one-pole release lets the gain recover, and two cascaded
 * moving averages turn every reduction into an S-shaped ramp that completes
 * exactly when the peak reaches the (delayed) output. Because the averaged
 * window only ever contains gains at or below each peak's requirement, the
 * ceiling is met by construction: transients are turned down smoothly
 * instead of being clipped. Optional ISP (Inter-Sample Peak) detection feeds
 * the sidechain with the ITU-R BS.1770 4x oversampled true-peak estimate and
 * shortens the ramp so the reduction covers every sample that estimate is
 * formed from. A final clamp at the ceiling remains as a rounding backstop
 * (and covers live ceiling or lookahead changes); true-peak levels are
 * reduced but not mathematically bounded.
 *
 * @note This class is strictly real-time safe. It performs zero allocations
 *       in the audio thread. The maximum lookahead time dictates the memory
 *       allocated during prepare().
 *
 * Features:
 * - Brickwall limiting (sample peaks never exceed the ceiling)
 * - ISP true-peak detection (4x oversampled FIR sidechain)
 * - Sliding-minimum gain hold + S-curve attack (artifact-free transients)
 * - CPU-optimized adaptive release (avoids std::exp in hot paths)
 * - Real-time safe parameter updates (lock-free atomics)
 *
 * Threading: prepare() belongs to the setup thread (allocates; never call it
 * concurrently with processing). processBlock(), processSample() and reset()
 * belong to the audio thread. All setters are lock-free atomic publications,
 * safe from any thread; changes are consumed at the next block (or the next
 * channel-0 sample). Non-finite setter arguments are ignored. getLatency()
 * derives from the published lookahead parameter, so a host reads the correct
 * value immediately after setLookahead(). getGainReductionDb() is metering:
 * a cross-thread read of a published atomic word (up to one block behind).
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, RingBuffer.h,
 *               SmoothedValue.h, DenormalGuard.h, TruePeakDetector.h,
 *               StateBlob.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/RingBuffer.h"
#include "../Core/SmoothedValue.h"
#include "../Core/DenormalGuard.h"
#include "../Core/TruePeakDetector.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Keeps the gain computer's rare paths (once per hold block, once per 2^16
// frames) out of the per-sample loop, so the hot part inlines. Guarded the same
// way as in Core/Biquad.h.
#ifndef DSPARK_NOINLINE
  #if defined(_MSC_VER)
    #define DSPARK_NOINLINE __declspec(noinline)
  #elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_NOINLINE __attribute__((noinline))
  #else
    #define DSPARK_NOINLINE
  #endif
#endif

namespace dspark {

/**
 * @class Limiter
 * @brief High-performance brickwall lookahead limiter.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Limiter
{
public:
    ~Limiter() = default; // non-virtual: leaf class (no virtual dispatch)

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Allocates memory and prepares the limiter for processing.
     *
     * Invalid arguments are release-safe: a non-positive or non-finite sample
     * rate makes this call a no-op (the previous state, if any, is kept).
     * Configured parameters (ceiling, release, lookahead, toggles) survive
     * re-preparation.
     *
     * @warning Must be called from the main/setup thread, NEVER from the audio thread.
     *
     * @param sampleRate Sample rate in Hz (must be > 0).
     * @param numChannels Number of channels to process (clamped to [1, 16]).
     * @param initialLookaheadMs Optional lookahead override in ms; any
     *        non-positive or non-finite value (the default) keeps the current
     *        lookahead parameter.
     */
    void prepare(double sampleRate, int numChannels = 2, double initialLookaheadMs = -1.0)
    {
        if (!(sampleRate > 0.0)) return; // NaN-safe validity gate: keep previous state
        prepared_ = false;               // basic guarantee while (re)allocating

        sampleRate_ = sampleRate;
        // The true-peak detector state is a fixed kMaxChannels array; clamp so
        // detection can never index it out of bounds for high channel counts.
        numChannels_ = std::clamp(numChannels, 1, kMaxChannels);

        // Caching inverse sample rate for fast math in hot paths
        invSampleRate_ = T(1) / static_cast<T>(sampleRate_);

        // Allocate for maximum possible lookahead to prevent RT-allocations
        // later. The sample count is capped in double before the cast (a cast
        // of an out-of-int-range double is undefined behaviour).
        const double maxLaExact = sampleRate_ * kMaxLookaheadMs / 1000.0;
        const int maxLookaheadSamples = static_cast<int>(std::min(maxLaExact, 1.0e7)) + 1;

        // Use the clamped count: a degenerate numChannels (<= 0, or negative cast
        // to size_t, or > kMaxChannels) must neither under-allocate (which would
        // make processBlock index delayLines_ out of bounds) nor over-allocate.
        delayLines_.resize(static_cast<size_t>(numChannels_));
        for (auto& dl : delayLines_)
            dl.prepare(maxLookaheadSamples * 2 + kChunk); // lookahead glide + one processBlock() chunk

        // Gain-computer histories: power-of-two rings covering the longest
        // hold window (L + 1) plus the moving-average lookback.
        int hist = 4;
        while (hist < maxLookaheadSamples + 4) hist <<= 1;
        histSize_ = hist;
        histMask_ = hist - 1;
        eHist_.assign(static_cast<size_t>(hist), 1.0);
        avgHist_.assign(static_cast<size_t>(hist), 1.0);
        rHist_.assign(static_cast<size_t>(hist), 1.0);
        sufMin_.assign(static_cast<size_t>(hist) + 1, 1.0);

        if (std::isfinite(initialLookaheadMs) && initialLookaheadMs > 0.0)
            lookaheadMs_.store(std::clamp(static_cast<T>(initialLookaheadMs),
                                          T(0.5), static_cast<T>(kMaxLookaheadMs)),
                               std::memory_order_relaxed);
        (void)lookaheadDirty_.exchange(false, std::memory_order_acquire);
        applyLookaheadTarget();
        lookaheadCurrent_ = static_cast<T>(lookaheadSamples_);

        // Duration cap for the adaptive-release counter (2 seconds).
        maxLimitSamples_ = static_cast<int>(std::min(sampleRate_ * 2.0, 1.0e9));

        // Ceiling smoother setup
        const T ceilDb = ceilingDb_.load(std::memory_order_relaxed);
        lastCeilingDb_ = ceilDb;
        const T ceilLinear = decibelsToGain(ceilDb);
        ceilingSmooth_.prepare(sampleRate, 30.0);
        ceilingSmooth_.reset(ceilLinear);

        lastReleaseMs_ = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
        updateReleaseCoefficient();

        reset();
        prepared_ = true;
    }

    /**
     * @brief Prepares from AudioSpec (unified API).
     *
     * Preserves the configured lookahead parameter (it does NOT reset it to
     * the constructor default), so a host re-activation keeps the latency the
     * user dialled in.
     */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Processes an AudioBufferView in-place.
     *
     * RT-Safe: Yes. Lock-free and allocation-free.
     *
     * The gain envelope is linked: the loudest channel drives the reduction
     * applied to all channels. Channels beyond the prepared count pass
     * through untouched (and undelayed).
     *
     * @param buffer Audio buffer view.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();

        if (lookaheadDirty_.exchange(false, std::memory_order_acquire))
            applyLookaheadTarget();
        syncParameters();

        const bool isp          = truePeakEnabled_.load(std::memory_order_relaxed);
        const bool adaptive     = adaptiveRelease_.load(std::memory_order_relaxed);
        const bool safetyClip   = safetyClipEnabled_.load(std::memory_order_relaxed);
        const T relMs           = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
        const T lookTarget      = static_cast<T>(lookaheadSamples_);

        T* chData[kMaxChannels] = {};
        for (int ch = 0; ch < nCh; ++ch)
            chData[ch] = buffer.getChannel(ch);

        // Three passes per chunk: detection (the true-peak FIR, throughput
        // bound), the serial gain computer (latency bound) and the gain
        // application. Kept apart, the out-of-order core no longer stalls the
        // FIR work behind the gain recursion. The per-sample arithmetic is
        // unchanged, so processSample() stays bit-identical for mono.
        for (int start = 0; start < nS; start += kChunk)
        {
            const int n = std::min(kChunk, nS - start);

            // Pass 1: linked peak detection (loudest channel per frame).
            std::fill(chunkPeak_, chunkPeak_ + n, T(0));
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* x = chData[ch] + start;
                for (int i = 0; i < n; ++i)
                {
                    const T chPeak = isp ? truePeak_.processSample(x[i], ch) : std::abs(x[i]);
                    chunkPeak_[i] = std::max(chunkPeak_[i], chPeak); // NaN is ignored
                }
            }

            // Pass 2: ceiling smoother, lookahead glide and gain computer.
            for (int i = 0; i < n; ++i)
            {
                const T ceiling = ceilingSmooth_.getNextValue();
                chunkCeiling_[i] = ceiling;

                // Live-change smoothing of the lookahead read offset: at most
                // one sample of change per sample, so a setLookahead() during
                // playback glides (brief micro pitch-shift) instead of clicking.
                if (lookaheadCurrent_ < lookTarget)      lookaheadCurrent_ += T(1);
                else if (lookaheadCurrent_ > lookTarget) lookaheadCurrent_ -= T(1);
                chunkLook_[i] = static_cast<int>(lookaheadCurrent_);

                advanceGain(chunkPeak_[i], ceiling, adaptive, relMs);
                chunkGain_[i] = currentGain_;
            }

            // Pass 3: delay, apply the gain, optional safety clip.
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* x = chData[ch] + start;
                auto& line = delayLines_[static_cast<size_t>(ch)];
                line.pushBlock(x, n); // frame i now sits (n - 1 - i) samples back
                for (int i = 0; i < n; ++i)
                {
                    const T ceiling = chunkCeiling_[i];
                    T out = line.read(chunkLook_[i] + (n - 1 - i)) * chunkGain_[i];

                    // Rounding backstop: the gain computer already meets the
                    // ceiling; the clamp only trims float rounding and the
                    // brief transition after a live ceiling or lookahead change.
                    out = std::clamp(out, -ceiling, ceiling);

                    if (safetyClip)
                    {
                        const T clipCeil = std::min(kSafetyClipCeiling, ceiling);
                        if (std::abs(out) > clipCeil)
                            out = applySafetyClipper(out, clipCeil);
                    }

                    x[i] = out;
                }
            }
        }

        // One relaxed store per block, outside the per-sample loop.
        publishedGain_.store(currentGain_, std::memory_order_relaxed);
    }

    /**
     * @brief Processes a single sample of one channel.
     *
     * The shared state (ceiling smoother, lookahead glide and gain computer)
     * advances on channel 0, so call channel 0 first within each sample
     * frame. For mono streams (channel 0 only) this path is bit-identical to
     * processBlock(). Peaks of the other channels enter the gain computer one
     * frame late; for properly linked multi-channel limiting prefer
     * processBlock().
     *
     * @note No DenormalGuard here (per-sample hot path); per-sample callers
     *       are expected to guard their own processing loop.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (!prepared_) return input;
        // Release-safe channel bound (delayLines_ is sized to numChannels_):
        // out-of-range channels are an exact pass-through, no state touched.
        if (channel < 0 || channel >= numChannels_) return input;

        const bool isp        = truePeakEnabled_.load(std::memory_order_relaxed);
        const bool adaptive   = adaptiveRelease_.load(std::memory_order_relaxed);
        const bool safetyClip = safetyClipEnabled_.load(std::memory_order_relaxed);

        if (channel == 0)
        {
            if (lookaheadDirty_.exchange(false, std::memory_order_acquire))
                applyLookaheadTarget();
            syncParameters();
            sampleCeiling_ = ceilingSmooth_.getNextValue();

            const T lookTarget = static_cast<T>(lookaheadSamples_);
            if (lookaheadCurrent_ < lookTarget)      lookaheadCurrent_ += T(1);
            else if (lookaheadCurrent_ > lookTarget) lookaheadCurrent_ -= T(1);
            sampleLookNow_ = static_cast<int>(lookaheadCurrent_);
        }
        const T ceiling = sampleCeiling_;

        delayLines_[channel].push(input);
        const T chPeak = isp ? truePeak_.processSample(input, channel) : std::abs(input);

        // The gain computer advances on channel 0. Peaks of the other channels
        // reach it on the next frame (they are processed after channel 0), so
        // they are one sample late: prefer processBlock() for linked
        // multi-channel limiting. Mono is bit-identical to processBlock().
        if (channel == 0)
        {
            const T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
            advanceGain(std::max(chPeak, pendingPeak_), ceiling, adaptive, relMs);
            pendingPeak_ = T(0);
            publishedGain_.store(currentGain_, std::memory_order_relaxed);
        }
        else if (chPeak > pendingPeak_)
        {
            pendingPeak_ = chPeak;
        }

        T out = delayLines_[channel].read(sampleLookNow_) * currentGain_;
        out = std::clamp(out, -ceiling, ceiling); // exact brickwall contract

        if (safetyClip)
        {
            const T clipCeil = std::min(kSafetyClipCeiling, ceiling);
            if (std::abs(out) > clipCeil)
                out = applySafetyClipper(out, clipCeil);
        }
        return out;
    }

    /** @brief Resets the internal state (delays, gain reduction). RT-Safe. */
    void reset() noexcept
    {
        for (auto& dl : delayLines_) dl.reset();
        truePeak_.reset();
        currentGain_ = T(1);
        publishedGain_.store(T(1), std::memory_order_relaxed);
        limitingDuration_ = 0;
        pendingPeak_ = T(0);
        resetGainComputer();
        lookaheadCurrent_ = static_cast<T>(lookaheadSamples_);
        ceilingSmooth_.skip();
        sampleCeiling_ = ceilingSmooth_.getCurrentValue();
        sampleLookNow_ = lookaheadSamples_;
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the absolute output ceiling.
     * @param dB Ceiling in dBFS (e.g., -1.0 for streaming). RT-Safe.
     *           Non-finite values are ignored.
     */
    void setCeiling(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        ceilingDb_.store(dB, std::memory_order_relaxed);
    }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Sets the base release time.
     * @param ms Release time in milliseconds (floored to 1 ms). RT-Safe.
     *           Non-finite values are ignored.
     */
    void setRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::max(T(1), ms), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the lookahead time dynamically.
     *
     * @note RT-Safe (atomic publication from any thread). It only adjusts read
     *       pointers up to the max memory allocated during prepare(); the read
     *       offset glides to the new value at one sample per sample. Non-finite
     *       values are ignored. getLatency() reflects the new value immediately.
     *
     * @param ms Lookahead in milliseconds (clamped to 0.5 to 10.0 ms).
     */
    void setLookahead(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        lookaheadMs_.store(std::clamp(ms, T(0.5), static_cast<T>(kMaxLookaheadMs)),
                           std::memory_order_relaxed);
        lookaheadDirty_.store(true, std::memory_order_release);
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /** @brief Enables 4x oversampled ISP true-peak detection. RT-Safe.
     *  The attack ramp shortens by the estimator's support (11 samples) so the
     *  reduction covers every sample a true-peak reading is formed from. */
    void setTruePeak(bool enabled) noexcept
    {
        truePeakEnabled_.store(enabled, std::memory_order_relaxed);
        lookaheadDirty_.store(true, std::memory_order_release);
    }

    /** @brief Enables program-dependent adaptive release. RT-Safe. */
    void setAdaptiveRelease(bool enabled) noexcept { adaptiveRelease_.store(enabled, std::memory_order_relaxed); }

    /**
     * @brief Enables the post-limiter soft-knee safety clipper. RT-Safe.
     *
     * Softens the region above -0.3 dBFS up to the ceiling; it only has an
     * effect when the ceiling is set above -0.3 dBFS (below that, the
     * brickwall clamp already keeps the output under the clipper threshold).
     */
    void setSafetyClip(bool enabled) noexcept { safetyClipEnabled_.store(enabled, std::memory_order_relaxed); }

    // Getters
    [[nodiscard]] bool isTruePeakEnabled() const noexcept { return truePeakEnabled_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isAdaptiveReleaseEnabled() const noexcept { return adaptiveRelease_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isSafetyClipEnabled() const noexcept { return safetyClipEnabled_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLookahead() const noexcept { return lookaheadMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Reports the processing latency (the lookahead) in samples.
     *
     * Derived from the published lookahead parameter, so it is correct
     * immediately after setLookahead() from any thread (the audio thread may
     * consume the change one block later; the glide covers the transition).
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return lookaheadSamplesFor(lookaheadMs_.load(std::memory_order_relaxed));
    }

    /** @brief Current gain reduction in dB (metering).
     *
     *  Safe from any thread: it loads a published atomic word, so off the audio
     *  thread it may be up to one block behind. It used to read `currentGain_`
     *  directly -- a plain word the audio thread writes every sample, read from
     *  another thread, which is a data race and not merely an approximate
     *  number. */
    [[nodiscard]] T getGainReductionDb() const noexcept
    {
        return gainToDecibels(publishedGain_.load(std::memory_order_relaxed));
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("LIMT"), 1);
        w.write("ceiling", static_cast<float>(ceilingDb_.load(std::memory_order_relaxed)));
        w.write("release", static_cast<float>(releaseMs_.load(std::memory_order_relaxed)));
        w.write("lookahead", static_cast<float>(lookaheadMs_.load(std::memory_order_relaxed)));
        w.write("truePeak", truePeakEnabled_.load(std::memory_order_relaxed));
        w.write("adaptive", adaptiveRelease_.load(std::memory_order_relaxed));
        w.write("safetyClip", safetyClipEnabled_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("LIMT")) return false;
        setCeiling(static_cast<T>(r.read("ceiling", -0.3f)));
        setRelease(static_cast<T>(r.read("release", 100.0f)));
        setLookahead(static_cast<T>(r.read("lookahead", 2.0f)));
        setTruePeak(r.read("truePeak", false));
        setAdaptiveRelease(r.read("adaptive", false));
        setSafetyClip(r.read("safetyClip", false));
        return true;
    }

protected:
    static constexpr int kMaxChannels = 16;
    static constexpr double kMaxLookaheadMs = 10.0;
    static constexpr T kSafetyClipCeiling = T(0.96605); ///< -0.3 dBFS

    // Shared ITU-R BS.1770-4 true-peak detector (Core/TruePeakDetector.h).
    TruePeakDetector<T, kMaxChannels> truePeak_;

    /** Lookahead in samples for a given ms parameter (shared by the audio-side
     *  recompute and getLatency() so both always agree exactly). The product is
     *  capped in double before the int cast (out-of-range casts are UB). */
    [[nodiscard]] int lookaheadSamplesFor(T ms) const noexcept
    {
        const double clampedMs = std::clamp(static_cast<double>(ms), 0.5, kMaxLookaheadMs);
        const double exact = sampleRate_ * clampedMs / 1000.0;
        return std::max(1, static_cast<int>(std::min(exact, 1.0e7)));
    }

    /** Consumes the published lookahead parameter (audio/setup thread). */
    inline void applyLookaheadTarget() noexcept
    {
        lookaheadSamples_ = lookaheadSamplesFor(lookaheadMs_.load(std::memory_order_relaxed));
        configureGainComputer();
    }

    // ---- Gain computer ------------------------------------------------------
    //
    // r[n] = min(1, ceiling / peak[n]) is the gain sample n requires. With a
    // lookahead of L samples the audio sample x[p] leaves the delay line at
    // p + L, and the applied gain there is the double moving average of the
    // released envelope e over [p + L - span, p + L]. e never exceeds the
    // sliding minimum h[n] = min(r[n - L .. n]), so every e in that window is
    // at or below r[p] whenever span <= L: the ceiling holds by construction.
    // In ISP mode a true-peak reading at n is formed from x[n - 11 .. n], so
    // the span shrinks by 11 samples to cover the earliest of them.

    /** Sizes the hold window and the two moving averages for the current
     *  lookahead / ISP mode and rebuilds their running sums from history. */
    void configureGainComputer() noexcept
    {
        if (histSize_ == 0) return;
        const int L = std::clamp(lookaheadSamples_, 1, histSize_ - 3);
        const int isp = truePeakEnabled_.load(std::memory_order_relaxed)
                      ? TruePeakDetector<T, kMaxChannels>::getTaps() - 1 : 0;
        const int span = std::max(0, L - isp);          // (a - 1) + (b - 1)
        holdLen_ = L + 1;
        boxA_ = span / 2 + 1;
        boxB_ = span - (boxA_ - 1) + 1;
        invBoxA_ = 1.0 / static_cast<double>(boxA_);
        invBoxB_ = 1.0 / static_cast<double>(boxB_);

        // Rebuild the running sums and the hold from the stored history
        // (O(L), only on a configuration change), so the new windows are
        // exact from the next frame on.
        resumBoxes();
        restartHoldBlock(frame_ - holdLen_);
    }

    /** Recomputes both moving-average sums exactly from their histories. */
    DSPARK_NOINLINE void resumBoxes() noexcept
    {
        sum1_ = 0.0;
        for (int k = 0; k < boxA_; ++k)
            sum1_ += eHist_[static_cast<size_t>((frame_ - 1 - k) & histMask_)];
        sum2_ = 0.0;
        for (int k = 0; k < boxB_; ++k)
            sum2_ += avgHist_[static_cast<size_t>((frame_ - 1 - k) & histMask_)];
    }

    /** Sliding minimum (van Herk / Gil-Werman, streaming form): the frames are
     *  cut into blocks of holdLen_. Once a block is complete its suffix minima
     *  are stored; the window ending at a frame of the next block is then the
     *  minimum of one stored suffix and the running prefix of that block.
     *  Branch-free per frame, one backward pass per block (amortised O(1)).
     *  This stores the suffix minima of the block starting at firstFrame and
     *  starts a new block at the following frame. */
    DSPARK_NOINLINE void restartHoldBlock(int64_t firstFrame) noexcept
    {
        double m = 1.0;
        sufMin_[static_cast<size_t>(holdLen_)] = 1.0; // empty suffix
        for (int j = holdLen_ - 1; j >= 0; --j)
        {
            m = std::min(m, rHist_[static_cast<size_t>((firstFrame + j) & histMask_)]);
            sufMin_[static_cast<size_t>(j)] = m;
        }
        blockPos_ = 0;
        prefixMin_ = 1.0;
    }

    /** Clears the gain computer to unity gain (history included). */
    void resetGainComputer() noexcept
    {
        std::fill(eHist_.begin(), eHist_.end(), 1.0);
        std::fill(avgHist_.begin(), avgHist_.end(), 1.0);
        std::fill(rHist_.begin(), rHist_.end(), 1.0);
        std::fill(sufMin_.begin(), sufMin_.end(), 1.0);
        blockPos_ = 0;
        prefixMin_ = 1.0;
        frame_ = 0;
        envelope_ = 1.0;
        sum1_ = static_cast<double>(boxA_);
        sum2_ = static_cast<double>(boxB_);
        sinceResum_ = 0;
        currentGain_ = T(1);
    }

    /** One frame of the gain computer; leaves the gain to apply in currentGain_. */
    inline void advanceGain(T peak, T ceiling, bool adaptive, T relMs) noexcept
    {
        // min() instead of a branch on peak > ceiling (data-dependent, so it
        // mispredicts constantly under limiting); peak == 0 gives +inf -> 1.
        const double required = std::min(1.0, static_cast<double>(ceiling / peak));

        // Exact minimum of r over the last holdLen_ frames (see restartHoldBlock()).
        rHist_[static_cast<size_t>(frame_ & histMask_)] = required;
        prefixMin_ = std::min(prefixMin_, required);
        const double held = std::min(sufMin_[static_cast<size_t>(blockPos_ + 1)], prefixMin_);
        if (++blockPos_ == holdLen_)
            restartHoldBlock(frame_ - (holdLen_ - 1));

        // Release: the envelope follows reductions at once (the moving
        // averages below shape the attack) and recovers with the one-pole.
        // min() selects between the two without a data-dependent branch.
        const bool attacking = held < envelope_;
        double coeff = static_cast<double>(releaseCoeff_);
        if (adaptive)
        {
            // Program-dependent release: up to 3x slower after sustained
            // limiting. 1 / (1 + fs * tau) avoids std::exp per sample.
            T baseFactor = T(1);
            if (limitingDuration_ > 0)
            {
                const T durationMs = static_cast<T>(limitingDuration_) * T(1000) * invSampleRate_;
                baseFactor = T(1) + std::min(durationMs / T(100), T(2));
            }
            coeff = 1.0 / (1.0 + sampleRate_ * static_cast<double>(relMs * baseFactor) / 1000.0);
        }
        envelope_ = std::min(held, envelope_ + coeff * (held - envelope_));
        limitingDuration_ = attacking ? std::min(limitingDuration_ + 1, maxLimitSamples_)
                                      : (envelope_ > 0.999 ? 0 : limitingDuration_);

        // Two cascaded moving averages (lengths boxA_, boxB_): an S-shaped,
        // click-free ramp spanning exactly the window the hold covers.
        const size_t w = static_cast<size_t>(frame_ & histMask_);
        eHist_[w] = envelope_;
        sum1_ += envelope_ - eHist_[static_cast<size_t>((frame_ - boxA_) & histMask_)];
        const double avg1 = sum1_ * invBoxA_;
        avgHist_[w] = avg1;
        sum2_ += avg1 - avgHist_[static_cast<size_t>((frame_ - boxB_) & histMask_)];
        ++frame_;

        // Periodic exact re-summation keeps the running sums drift-free.
        if (++sinceResum_ >= kResumPeriod)
        {
            sinceResum_ = 0;
            resumBoxes();
        }

        currentGain_ = static_cast<T>(std::min(sum2_ * invBoxB_, 1.0));
    }

    // Fast-path synchronization for atomic variables. The linear ceiling is
    // only recomputed when the dB parameter actually changed (skips the pow).
    inline void syncParameters() noexcept
    {
        const T ceilDb = ceilingDb_.load(std::memory_order_relaxed);
        if (ceilDb != lastCeilingDb_)
        {
            lastCeilingDb_ = ceilDb;
            ceilingSmooth_.setTargetValue(decibelsToGain(ceilDb));
        }

        const T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
        if (relMs != lastReleaseMs_)
        {
            lastReleaseMs_ = relMs;
            updateReleaseCoefficient();
        }
    }

    inline void updateReleaseCoefficient() noexcept
    {
        if (sampleRate_ > 0)
            releaseCoeff_ = T(1) - std::exp(T(-1) / (static_cast<T>(sampleRate_) * lastReleaseMs_ / T(1000)));
    }

    /** Symmetric soft-knee clipper for the region above the clip threshold,
     *  with a hard backstop at 1.05x the threshold. */
    [[nodiscard]] inline T applySafetyClipper(T out, T clipCeil) const noexcept
    {
        T sign = (out >= T(0)) ? T(1) : T(-1);
        T excess = std::abs(out) - clipCeil;
        T blend = T(1) / (T(1) + excess * T(10));
        out = sign * (clipCeil * blend + std::abs(out) * (T(1) - blend));

        const T hardCeil = clipCeil * T(1.05);
        if (std::abs(out) > hardCeil)
            out = std::clamp(out, -hardCeil, hardCeil);

        return out;
    }

    bool prepared_ = false;
    double sampleRate_ = 48000.0;
    T invSampleRate_ = T(1.0 / 48000.0);
    int numChannels_ = 2;
    int lookaheadSamples_ = 96;      ///< Audio-side lookahead (consumed from lookaheadMs_).
    int maxLimitSamples_ = 96000;    ///< Adaptive-release duration cap (2 s).

    std::atomic<T> ceilingDb_ { T(-0.3) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<T> lookaheadMs_ { T(2) };
    std::atomic<bool> lookaheadDirty_ { false };
    std::atomic<bool> truePeakEnabled_ { false };
    std::atomic<bool> adaptiveRelease_ { false };
    std::atomic<bool> safetyClipEnabled_ { false };

    SmoothedValue<T> ceilingSmooth_;
    T lastCeilingDb_ = T(-0.3);      ///< Change detector for the ceiling pow skip.

    T releaseCoeff_ = T(0);
    T lastReleaseMs_ = T(-1);

    T currentGain_ = T(1);
    /// Cross-thread metering readout of currentGain_: published once per
    /// processBlock() and once per processSample() call. processSample() is the
    /// per-sample entry point, so there the store does run once per sample --
    /// what it never does is sit inside processBlock()'s own sample loop.
    std::atomic<T> publishedGain_ { T(1) };
    int limitingDuration_ = 0;
    T lookaheadCurrent_ = T(96); ///< Smoothed read offset (glides on live changes).

    // Gain computer (see advanceGain()): sliding minimum, released envelope
    // and the two moving averages, all in double (recursive state).
    static constexpr int kResumPeriod = 1 << 16;
    int histSize_ = 0;
    int histMask_ = 0;
    std::vector<double> eHist_;      ///< Released envelope history.
    std::vector<double> avgHist_;    ///< First moving-average output history.
    std::vector<double> rHist_;      ///< Required-gain history.
    std::vector<double> sufMin_;     ///< Suffix minima of the last complete hold block (+1 empty).
    int blockPos_ = 0;               ///< Position inside the current hold block.
    double prefixMin_ = 1.0;         ///< Running minimum of the current hold block.
    int64_t frame_ = 0;
    double envelope_ = 1.0;
    double sum1_ = 1.0;
    double sum2_ = 1.0;
    int boxA_ = 1;
    int boxB_ = 1;
    double invBoxA_ = 1.0;
    double invBoxB_ = 1.0;
    int holdLen_ = 97;
    int sinceResum_ = 0;
    T pendingPeak_ = T(0);           ///< Other channels' peaks for the next processSample() frame.

    // processBlock() chunk scratch (see its three passes).
    static constexpr int kChunk = 128;
    T chunkPeak_[kChunk] = {};
    T chunkCeiling_[kChunk] = {};
    T chunkGain_[kChunk] = {};
    int chunkLook_[kChunk] = {};

    // Per-sample path caches (advanced on channel 0, reused by later channels).
    T sampleCeiling_ = T(0.96605);
    int sampleLookNow_ = 96;

    std::vector<RingBuffer<T>> delayLines_;
};

} // namespace dspark
