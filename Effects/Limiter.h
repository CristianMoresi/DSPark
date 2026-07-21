// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Limiter.h
 * @brief Brickwall lookahead limiter with optional ISP (true-peak) detection
 *        and adaptive release for mastering.
 *
 * A peak limiter that prevents audio from exceeding a configurable ceiling.
 * It uses a lookahead delay line with a peak hold over the lookahead window
 * and a smoothed attack envelope, so transients are turned down transparently
 * instead of being clipped. Optional ISP (Inter-Sample Peak) detection feeds
 * the sidechain with the ITU-R BS.1770 4x oversampled true-peak estimate,
 * which greatly reduces inter-sample overs. The exact brickwall guarantee is
 * in the sample domain (a final clamp at the ceiling); true-peak levels after
 * that clamp are reduced but not mathematically bounded.
 *
 * @note This class is strictly real-time safe. It performs zero allocations
 *       in the audio thread. The maximum lookahead time dictates the memory
 *       allocated during prepare().
 *
 * Features:
 * - Brickwall limiting (sample peaks never exceed the ceiling)
 * - ISP true-peak detection (4x oversampled FIR sidechain)
 * - Lookahead peak hold + smoothed attack curve (artifact-free transients)
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
 * an unsynchronized cross-thread read of the live envelope (approximate).
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
            dl.prepare(maxLookaheadSamples * 2); // Double size for safety margin

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

        for (int i = 0; i < nS; ++i)
        {
            T ceiling = ceilingSmooth_.getNextValue();
            T peak = T(0);

            // Live-change smoothing of the lookahead read offset: at most one
            // sample of change per sample, so a setLookahead() during playback
            // glides (brief micro pitch-shift) instead of clicking.
            if (lookaheadCurrent_ < lookTarget)      lookaheadCurrent_ += T(1);
            else if (lookaheadCurrent_ > lookTarget) lookaheadCurrent_ -= T(1);
            const int lookNow = static_cast<int>(lookaheadCurrent_);

            // Phase 1: Peak detection and delay line push
            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = chData[ch][i];
                delayLines_[ch].push(sample);

                T chPeak = isp ? truePeak_.processSample(sample, ch) : std::abs(sample);
                if (chPeak > peak) peak = chPeak;
            }

            // Phase 2: Peak-hold over the lookahead window, then gain envelope.
            // The gain reduction for a transient must persist until that transient
            // reaches the (delayed) output lookaheadSamples_ later; otherwise an
            // isolated peak would be output after the envelope has already released
            // and could exceed the ceiling. Holding the detected peak for the
            // look-ahead duration guarantees the brickwall without an instantaneous
            // (clicky) attack; the one-pole attack still reaches ~99% over the window.
            if (peak >= heldPeak_)      { heldPeak_ = peak; peakHoldCounter_ = lookaheadSamples_; }
            else if (peakHoldCounter_ > 0) { --peakHoldCounter_; }
            else                          { heldPeak_ = peak; }

            T targetGain = (heldPeak_ > ceiling) ? ceiling / heldPeak_ : T(1);
            smoothGain(targetGain, adaptive, relMs);

            // Phase 3: Apply gain, guarantee the ceiling, optional safety clip
            for (int ch = 0; ch < nCh; ++ch)
            {
                T out = delayLines_[ch].read(lookNow) * currentGain_;

                // Hard ceiling guarantee: the smoothed attack converges to
                // ~99% of the target inside the lookahead window, leaving up
                // to ~0.09 dB of residual overshoot. Clamping that residual is
                // inaudible (it only ever trims the last 1%) and makes the
                // brickwall contract exact.
                out = std::clamp(out, -ceiling, ceiling);

                if (safetyClip)
                {
                    T clipCeil = std::min(kSafetyClipCeiling, ceiling);
                    if (std::abs(out) > clipCeil)
                        out = applySafetyClipper(out, clipCeil);
                }

                chData[ch][i] = out;
            }
        }
    }

    /**
     * @brief Processes a single sample of one channel.
     *
     * The shared state (ceiling smoother, lookahead glide, peak hold decay and
     * gain envelope) advances on channel 0, so call channel 0 first within
     * each sample frame. For mono streams (channel 0 only) this path is
     * bit-identical to processBlock(). Any channel's peak can raise the
     * shared peak hold, but only channel 0 detects into the envelope; for
     * properly linked multi-channel limiting prefer processBlock().
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

        // Shared peak hold: any channel may raise it; the per-frame decay and
        // the gain envelope advance on channel 0 only.
        if (chPeak >= heldPeak_)      { heldPeak_ = chPeak; peakHoldCounter_ = lookaheadSamples_; }
        else if (channel == 0)
        {
            if (peakHoldCounter_ > 0) --peakHoldCounter_;
            else                      heldPeak_ = chPeak;
        }

        if (channel == 0)
        {
            const T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
            const T targetGain = (heldPeak_ > ceiling) ? ceiling / heldPeak_ : T(1);
            smoothGain(targetGain, adaptive, relMs);
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
        limitingDuration_ = 0;
        heldPeak_ = T(0);
        peakHoldCounter_ = 0;
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

    /** @brief Enables 4x oversampled ISP true-peak detection. RT-Safe. */
    void setTruePeak(bool enabled) noexcept { truePeakEnabled_.store(enabled, std::memory_order_relaxed); }

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

    /** @brief Current gain reduction in dB (metering; unsynchronized read). */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainToDecibels(currentGain_); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("LIMT"), 1);
        w.write("ceiling", ceilingDb_.load(std::memory_order_relaxed));
        w.write("release", releaseMs_.load(std::memory_order_relaxed));
        w.write("lookahead", lookaheadMs_.load(std::memory_order_relaxed));
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

        // Calculate an attack coefficient that guarantees reaching 99% of target gain
        // exactly within the lookahead window to prevent transient clipping.
        // Formula: alpha = 1 - exp(-ln(100) / samples)
        attackCoeff_ = T(1) - std::exp(T(-4.60517) / static_cast<T>(lookaheadSamples_));
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

    inline void smoothGain(T targetGain, bool adaptive, T relMs) noexcept
    {
        if (targetGain < currentGain_)
        {
            // Smoothed attack to prevent discontinuous clicks on transients
            currentGain_ += attackCoeff_ * (targetGain - currentGain_);

            // Limit duration to max 2 seconds to prevent integer overflow
            if (limitingDuration_ < maxLimitSamples_) limitingDuration_++;
        }
        else
        {
            T coeff;
            if (adaptive)
            {
                T baseFactor = T(1);
                if (limitingDuration_ > 0)
                {
                    T durationMs = static_cast<T>(limitingDuration_) * T(1000) * invSampleRate_;
                    baseFactor = T(1) + std::min(durationMs / T(100), T(2));
                }
                T adaptedRelease = relMs * baseFactor;

                // Fast path for exp() approximation: 1 / (1 + fs * tau_seconds)
                // Avoids brutal CPU spike of std::exp() in the audio thread loop
                coeff = T(1) / (T(1) + (static_cast<T>(sampleRate_) * adaptedRelease / T(1000)));
            }
            else
            {
                coeff = releaseCoeff_;
            }

            currentGain_ += coeff * (targetGain - currentGain_);
            if (currentGain_ > T(1)) currentGain_ = T(1);
            if (currentGain_ > T(0.999)) limitingDuration_ = 0;
        }
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
    T attackCoeff_ = T(1);
    T lastReleaseMs_ = T(-1);

    T currentGain_ = T(1);
    int limitingDuration_ = 0;
    T lookaheadCurrent_ = T(96); ///< Smoothed read offset (glides on live changes).

    // Look-ahead peak hold (brickwall guarantee): holds the detected peak for
    // lookaheadSamples_ so the gain stays reduced until the peak is output.
    T heldPeak_ = T(0);
    int peakHoldCounter_ = 0;

    // Per-sample path caches (advanced on channel 0, reused by later channels).
    T sampleCeiling_ = T(0.96605);
    int sampleLookNow_ = 96;

    std::vector<RingBuffer<T>> delayLines_;
};

} // namespace dspark
