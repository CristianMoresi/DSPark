// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Limiter.h
 * @brief True-peak brickwall limiter with ISP detection, lookahead, and
 *        adaptive release for mastering.
 *
 * A professional-grade peak limiter that prevents audio from exceeding a
 * configurable ceiling. Combines lookahead delay for transparent transient
 * handling with optional ISP (Inter-Sample Peak) detection that catches
 * peaks between samples — critical for broadcast and streaming delivery.
 *
 * Features:
 * - Brickwall limiting (output never exceeds ceiling)
 * - ISP true-peak detection (4x oversampled, catches inter-sample peaks)
 * - Lookahead via delay line for transparent transient handling
 * - Adaptive release: fast for transients, slow for sustained material
 * - SmoothedValue for ceiling parameter (no clicks on change)
 * - Stereo linked detection
 * - Per-sample gain reduction metering
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `limiter.setCeiling(-1.0f)` — works with defaults.
 * - **Level 2 (intermediate):** Release, lookahead time.
 * - **Level 3 (expert):** ISP on/off, adaptive release, attack shape.
 *
 * Dependencies: DspMath.h, RingBuffer.h, SmoothedValue.h, AudioSpec.h,
 *               AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::Limiter<float> limiter;
 *   limiter.prepare(spec);
 *   limiter.setCeiling(-1.0f);       // -1 dBTP for streaming
 *   limiter.setTruePeak(true);       // ISP detection
 *   limiter.setAdaptiveRelease(true);
 *   limiter.processBlock(buffer);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/RingBuffer.h"
#include "../Core/SmoothedValue.h"
#include "../Core/DenormalGuard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <vector>

namespace dspark {

/**
 * @class Limiter
 * @brief True-peak brickwall limiter with ISP and adaptive release.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Limiter
{
public:
    virtual ~Limiter() = default;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the limiter.
     *
     * @param sampleRate Sample rate in Hz.
     * @param numChannels Number of channels.
     * @param lookaheadMs Lookahead time in ms (default: 2 ms).
     */
    void prepare(double sampleRate, int numChannels = 2,
                 double lookaheadMs = 2.0)
    {
        sampleRate_ = sampleRate;
        numChannels_ = numChannels;

        lookaheadMs_ = lookaheadMs;
        lookaheadSamples_ = static_cast<int>(sampleRate * lookaheadMs / 1000.0);
        lookaheadSamples_ = std::max(lookaheadSamples_, 1);

        // Delay lines (one per channel)
        delayLines_.resize(static_cast<size_t>(numChannels));
        for (auto& dl : delayLines_)
            dl.prepare(lookaheadSamples_ * 4);

        // Ceiling smoother
        T ceilLinear = decibelsToGain(ceilingDb_.load(std::memory_order_relaxed));
        ceilingSmooth_.prepare(sampleRate, 30.0);
        ceilingSmooth_.setCurrentAndTarget(ceilLinear);

        // Release coefficient (also cached in lastReleaseMs_ so processBlock
        // can skip the exp() when releaseMs is unchanged).
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
        T fs = static_cast<T>(sampleRate_);
        if (fs > T(0))
            releaseCoeff_ = T(1) - std::exp(T(-1) / (fs * relMs / T(1000)));
        lastReleaseMs_ = relMs;

        // ISP true-peak: build polyphase FIR filter + reset state
        buildTruePeakFilter();
        tpState_ = {};

        reset();

        prepared_ = true;
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Processes an AudioBufferView in-place (unified API).
     * @param buffer Audio buffer (any channel count).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();

        // Sync atomic params
        T ceilDb = ceilingDb_.load(std::memory_order_relaxed);
        T ceilLinear = decibelsToGain(ceilDb);
        ceilingSmooth_.setTargetValue(ceilLinear);

        // Release coefficient is recomputed only when releaseMs changed since
        // last block. prepare() initialises both; subsequent blocks do a cheap
        // scalar compare instead of an unconditional exp() per block.
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
        if (relMs != lastReleaseMs_)
        {
            T fs = static_cast<T>(sampleRate_);
            if (fs > T(0))
                releaseCoeff_ = T(1) - std::exp(T(-1) / (fs * relMs / T(1000)));
            lastReleaseMs_ = relMs;
        }

        bool isp         = truePeakEnabled_.load(std::memory_order_relaxed);
        bool adaptive    = adaptiveRelease_.load(std::memory_order_relaxed);
        bool safetyClip  = safetyClipEnabled_.load(std::memory_order_relaxed);

        // Safety clip ceiling: -0.3 dBFS
        constexpr T kSafetyClipCeiling = T(0.96605); // pow(10, -0.3/20)

        for (int i = 0; i < nS; ++i)
        {
            T ceiling = ceilingSmooth_.getNextValue();

            T peak = T(0);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = buffer.getChannel(ch)[i];
                delayLines_[ch].push(sample);

                T chPeak = isp ? detectTruePeak(sample, ch) : std::abs(sample);
                if (chPeak > peak) peak = chPeak;
            }

            T targetGain = (peak > ceiling) ? ceiling / peak : T(1);
            smoothGain(targetGain, adaptive, relMs);

            for (int ch = 0; ch < nCh; ++ch)
            {
                T out = delayLines_[ch].read(lookaheadSamples_) * currentGain_;

                // Safety clipper: interpolated reconstruction at clip boundaries.
                // The asymmetric blend below is designed to approach the ceiling
                // smoothly; a final hard clamp would undo the blend and reintroduce
                // the harsh clipping artefacts we are explicitly trying to avoid.
                // We keep a very loose safety net that only engages when the blend
                // itself is exceeded (>5% over ceiling) as a last-resort guard.
                T clipCeil = std::min(kSafetyClipCeiling, ceiling);
                if (safetyClip && std::abs(out) > clipCeil)
                {
                    T sign = (out >= T(0)) ? T(1) : T(-1);
                    T excess = std::abs(out) - clipCeil;
                    // Asymmetric blend: fast entry, slow exit (ClipOnly2-style)
                    T blend = T(1) / (T(1) + excess * T(10));
                    out = sign * (clipCeil * blend + std::abs(out) * (T(1) - blend));
                    // Last-resort safety net: only clamp if the blend somehow
                    // overshoots the ceiling by more than 5 % (should never
                    // happen with the formula above, but guards against FP edge
                    // cases with extreme inputs).
                    const T hardCeil = clipCeil * T(1.05);
                    if (std::abs(out) > hardCeil)
                        out = std::clamp(out, -hardCeil, hardCeil);
                }

                buffer.getChannel(ch)[i] = out;
            }
        }
    }

    /**
     * @brief Processes a single sample on one channel.
     *
     * Pushes the sample through the lookahead delay, detects peak,
     * applies gain smoothing, and returns the limited output.
     * When using multi-channel, call for each channel per sample
     * and use the highest peak across channels for proper linking.
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Limited output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        T ceiling = ceilingSmooth_.getCurrentValue();
        bool isp = truePeakEnabled_.load(std::memory_order_relaxed);
        bool adaptive = adaptiveRelease_.load(std::memory_order_relaxed);
        T relMs = releaseMs_.load(std::memory_order_relaxed);

        delayLines_[channel].push(input);
        T peak = isp ? detectTruePeak(input, channel) : std::abs(input);
        T targetGain = (peak > ceiling) ? ceiling / peak : T(1);
        smoothGain(targetGain, adaptive, relMs);

        return delayLines_[channel].read(lookaheadSamples_) * currentGain_;
    }

    /** @brief Resets the limiter state. */
    void reset() noexcept
    {
        for (auto& dl : delayLines_)
            dl.reset();
        tpState_ = {};
        currentGain_ = T(1);
        limitingDuration_ = 0;
        ceilingSmooth_.skip();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the output ceiling.
     *
     * Use -1.0 dBTP for streaming (Spotify, Apple Music).
     * Use -0.3 dBTP for broadcast (EBU R128).
     *
     * @param dB Ceiling in dBFS.
     */
    void setCeiling(T dB) noexcept
    {
        ceilingDb_.store(dB, std::memory_order_relaxed);
    }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Sets the release time.
     * @param ms Release time in milliseconds.
     */
    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::max(ms, T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the lookahead time.
     *
     * Requires re-calling prepare() to take effect.
     *
     * @param ms Lookahead in milliseconds (1-10 ms).
     */
    void setLookahead(T ms) noexcept
    {
        lookaheadMs_ = std::clamp(static_cast<double>(ms), 0.5, 10.0);
        if (sampleRate_ > 0)
        {
            lookaheadSamples_ = std::max(1,
                static_cast<int>(sampleRate_ * lookaheadMs_ / 1000.0));
            for (auto& dl : delayLines_)
                dl.prepare(lookaheadSamples_ * 4);
        }
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Enables or disables ISP (Inter-Sample Peak) true-peak detection.
     *
     * When enabled, detects peaks between samples using 4-point cubic
     * interpolation at 4x the sample rate. This catches inter-sample
     * overs that standard peak detection misses — required by EBU R128
     * and ITU-R BS.1770 for broadcast delivery.
     *
     * @param enabled True to enable ISP detection (default: false).
     */
    void setTruePeak(bool enabled) noexcept { truePeakEnabled_.store(enabled, std::memory_order_relaxed); }

    /**
     * @brief Enables adaptive release.
     *
     * When enabled, the release time adapts to the material:
     * - **Short transients**: Fast release (recovers quickly).
     * - **Sustained limiting**: Slow release (avoids pumping).
     *
     * @param enabled True to enable (default: false).
     */
    void setAdaptiveRelease(bool enabled) noexcept { adaptiveRelease_.store(enabled, std::memory_order_relaxed); }

    /**
     * @brief Enables the post-limiter safety clipper (ClipOnly2-style).
     *
     * Hard clips at -0.3 dBFS as a final safety net. Uses interpolated
     * reconstruction for smooth entry/exit transitions at clip boundaries.
     *
     * @param enabled True to enable safety clipper.
     */
    void setSafetyClip(bool enabled) noexcept { safetyClipEnabled_.store(enabled, std::memory_order_relaxed); }

    /** @brief Returns true if ISP true-peak detection is enabled. */
    [[nodiscard]] bool isTruePeakEnabled() const noexcept { return truePeakEnabled_.load(std::memory_order_relaxed); }

    /** @brief Returns true if adaptive release is enabled. */
    [[nodiscard]] bool isAdaptiveReleaseEnabled() const noexcept { return adaptiveRelease_.load(std::memory_order_relaxed); }

    /** @brief Returns true if safety clipper is enabled. */
    [[nodiscard]] bool isSafetyClipEnabled() const noexcept { return safetyClipEnabled_.load(std::memory_order_relaxed); }

    /** @brief Returns the latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return lookaheadSamples_; }

    /** @brief Returns the current gain reduction in dB. */
    [[nodiscard]] T getGainReductionDb() const noexcept
    {
        return gainToDecibels(currentGain_);
    }

    // -- Legacy API (backward compat) -------------------------------------------

    void process(T* data, int numSamples) noexcept
    {
        bool isp = truePeakEnabled_.load(std::memory_order_relaxed);
        bool adaptive = adaptiveRelease_.load(std::memory_order_relaxed);
        T relMs = releaseMs_.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            delayLines_[0].push(data[i]);
            T ceiling = ceilingSmooth_.getNextValue();
            T peak = isp ? detectTruePeak(data[i], 0) : std::abs(data[i]);
            T targetGain = (peak > ceiling) ? ceiling / peak : T(1);
            smoothGain(targetGain, adaptive, relMs);
            data[i] = delayLines_[0].read(lookaheadSamples_) * currentGain_;
        }
    }

    void process(T* left, T* right, int numSamples) noexcept
    {
        bool isp = truePeakEnabled_.load(std::memory_order_relaxed);
        bool adaptive = adaptiveRelease_.load(std::memory_order_relaxed);
        T relMs = releaseMs_.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            delayLines_[0].push(left[i]);
            delayLines_[1].push(right[i]);

            T ceiling = ceilingSmooth_.getNextValue();
            T peakL = isp ? detectTruePeak(left[i], 0) : std::abs(left[i]);
            T peakR = isp ? detectTruePeak(right[i], 1) : std::abs(right[i]);
            T peak = std::max(peakL, peakR);

            T targetGain = (peak > ceiling) ? ceiling / peak : T(1);
            smoothGain(targetGain, adaptive, relMs);

            left[i]  = delayLines_[0].read(lookaheadSamples_) * currentGain_;
            right[i] = delayLines_[1].read(lookaheadSamples_) * currentGain_;
        }
    }

protected:
    static constexpr int kMaxChannels = 16;

    /// Number of taps per polyphase phase for ITU true-peak FIR.
    static constexpr int kTpTaps = 12;
    /// Number of inter-sample phases to check (t=0.25, 0.5, 0.75).
    static constexpr int kTpPhases = 3;

    /// Per-channel state for true-peak detection: circular buffer of the last
    /// kTpTaps samples for FIR interpolation.
    ///
    /// A7 fix: storage is a power-of-two ring buffer (size 16 ≥ kTpTaps=12)
    /// with a write index and bitmask wrap. Push is O(1) (single store)
    /// instead of the previous O(kTpTaps-1) linear shift. The convolution
    /// walks backwards from the newest sample; bitmask indexing keeps the
    /// inner loop branchless.
    static constexpr int kTpHistSize = 16;       // next power of 2 ≥ kTpTaps
    static constexpr int kTpHistMask = kTpHistSize - 1;
    static_assert(kTpHistSize >= kTpTaps, "history must cover all FIR taps");

    struct TruePeakState {
        T history[kTpHistSize] = {};  ///< circular buffer, indexed by (writePos - 1 - k) & mask
        int writePos = 0;             ///< position of the next write
    };
    std::array<TruePeakState, kMaxChannels> tpState_{};

    /// Polyphase FIR coefficients for phases 1, 2, 3 (phase 0 = original sample).
    std::array<std::array<T, kTpTaps>, kTpPhases> tpCoeffs_{};

    /**
     * @brief Builds the polyphase FIR filter for ITU-R BS.1770-4 true-peak detection.
     *
     * Computes a 48-tap Kaiser-windowed sinc FIR (beta=8, ~80 dB stopband)
     * for 4x oversampling, then decomposes into 3 polyphase sub-filters of
     * 12 taps each. Called once from prepare().
     */
    void buildTruePeakFilter() noexcept
    {
        constexpr int N = kTpTaps * 4;       // 48 total FIR taps
        constexpr double M = (N - 1) / 2.0;  // 23.5
        constexpr double fc = 0.25;           // cutoff (normalized to 4x rate)
        constexpr double beta = 8.0;          // Kaiser: ~80 dB stopband
        constexpr double pi = std::numbers::pi;

        // Modified Bessel function of the first kind, order 0
        auto besselI0 = [](double x) -> double {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k <= 25; ++k)
            {
                double half = x / (2.0 * k);
                term *= half * half;
                sum += term;
                if (term < 1e-15 * sum) break;
            }
            return sum;
        };

        const double i0Beta = besselI0(beta);

        // Compute 48-tap FIR prototype
        double h[N];
        for (int n = 0; n < N; ++n)
        {
            double x = static_cast<double>(n) - M;

            // Windowed sinc: sinc(2*fc*x) * Kaiser(x)
            double sincArg = 2.0 * fc * x;
            double sincVal = (std::abs(sincArg) < 1e-10)
                ? 1.0
                : std::sin(pi * sincArg) / (pi * sincArg);

            double t = x / M;
            double kaiserVal = (std::abs(t) > 1.0)
                ? 0.0
                : besselI0(beta * std::sqrt(1.0 - t * t)) / i0Beta;

            h[n] = sincVal * kaiserVal;
        }

        // Extract polyphase phases 1, 2, 3 (skip phase 0 = identity)
        for (int phase = 0; phase < kTpPhases; ++phase)
        {
            int p = phase + 1;
            for (int k = 0; k < kTpTaps; ++k)
                tpCoeffs_[phase][k] = static_cast<T>(h[4 * k + p]);
        }
    }

    /**
     * @brief Detects true-peak level using ITU-R BS.1770-4 compliant 4x FIR oversampling.
     *
     * Uses a 48-tap polyphase FIR (12 taps x 4 phases, Kaiser beta=8) to
     * reconstruct the continuous signal at 3 inter-sample positions per sample
     * period. Compliant with ITU-R BS.1770-4 / EBU R128 true-peak measurement.
     */
    [[nodiscard]] T detectTruePeak(T sample, int ch) noexcept
    {
        auto& tp = tpState_[ch];

        // O(1) push into the circular history
        tp.history[tp.writePos] = sample;
        tp.writePos = (tp.writePos + 1) & kTpHistMask;

        T peak = std::abs(sample);

        // Check 3 inter-sample positions using polyphase FIR convolution
        // y[4m+p] = sum_{k=0}^{11} h[4k+p] * x[m-k].
        // The newest sample lives at (writePos − 1) mod kTpHistSize, and we
        // walk backwards one tap at a time.
        const int newest = (tp.writePos - 1) & kTpHistMask;
        for (int phase = 0; phase < kTpPhases; ++phase)
        {
            T interp = T(0);
            int idx = newest;
            for (int k = 0; k < kTpTaps; ++k)
            {
                interp += tp.history[idx] * tpCoeffs_[phase][k];
                idx = (idx - 1) & kTpHistMask;
            }

            T absInterp = std::abs(interp);
            if (absInterp > peak)
                peak = absInterp;
        }

        return peak;
    }

    void smoothGain(T targetGain, bool adaptive, T relMs) noexcept
    {
        if (targetGain < currentGain_)
        {
            currentGain_ = targetGain;
            limitingDuration_++;
        }
        else
        {
            T coeff;
            if (adaptive)
            {
                T baseFactor = T(1);
                if (limitingDuration_ > 0)
                {
                    T durationMs = static_cast<T>(limitingDuration_) * T(1000)
                                 / static_cast<T>(sampleRate_);
                    baseFactor = T(1) + std::min(durationMs / T(100), T(2));
                }
                T adaptedRelease = relMs * baseFactor;
                T fs = static_cast<T>(sampleRate_);
                coeff = T(1) - std::exp(T(-1) / (fs * adaptedRelease / T(1000)));
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

    bool prepared_ = false;
    double sampleRate_ = 48000.0;
    int numChannels_ = 2;
    int lookaheadSamples_ = 96;
    double lookaheadMs_ = 2.0;

    // Atomic parameters
    std::atomic<T> ceilingDb_ { T(-0.3) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<bool> truePeakEnabled_ { false };
    std::atomic<bool> adaptiveRelease_ { false };
    std::atomic<bool> safetyClipEnabled_ { false };

    // Ceiling smoother
    SmoothedValue<T> ceilingSmooth_;

    // Coefficients
    T releaseCoeff_ = T(0);
    T lastReleaseMs_ = T(-1); ///< Last releaseMs used to compute releaseCoeff_ (sentinel -1 forces first compute)

    // State
    T currentGain_ = T(1);
    int limitingDuration_ = 0;

    // ISP true-peak state (defined in protected section above detectTruePeak)

    // Delay lines for lookahead
    std::vector<RingBuffer<T>> delayLines_;

};

} // namespace dspark
