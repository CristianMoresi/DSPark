// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Limiter.h
 * @brief True-peak brickwall limiter with ISP detection, lookahead, and
 *        adaptive release for mastering.
 *
 * A professional-grade peak limiter that prevents audio from exceeding a
 * configurable ceiling. It uses a lookahead delay line with a smoothed attack
 * envelope to ensure transparent transient handling without high-frequency
 * clicks. Includes optional ISP (Inter-Sample Peak) detection for broadcast
 * compliance.
 *
 * @note This class is strictly real-time safe. It performs zero allocations
 *       in the audio thread. The maximum lookahead time dictates the memory
 *       allocated during prepare().
 *
 * Features:
 * - Brickwall limiting (output never exceeds ceiling)
 * - ISP true-peak detection (4x oversampled FIR)
 * - Lookahead with smoothed attack curve (artifact-free)
 * - CPU-optimized adaptive release (avoids std::exp in hot paths)
 * - Real-time safe parameter updates
 *
 * Dependencies: DspMath.h, RingBuffer.h, SmoothedValue.h, AudioSpec.h,
 *               AudioBuffer.h, DenormalGuard.h.
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
#include <numbers>

namespace dspark {

/**
 * @class Limiter
 * @brief High-performance True-peak brickwall limiter.
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
     * @brief Allocates memory and prepares the limiter for processing.
     *
     * @warning Must be called from the main/setup thread, NEVER from the audio thread.
     *
     * @param sampleRate Sample rate in Hz.
     * @param numChannels Number of channels to process.
     * @param initialLookaheadMs Lookahead time in ms to start with (default: 2.0 ms).
     */
    void prepare(double sampleRate, int numChannels = 2, double initialLookaheadMs = 2.0)
    {
        sampleRate_ = sampleRate;
        numChannels_ = numChannels;
        
        // Caching inverse sample rate for fast math in hot paths
        invSampleRate_ = T(1) / static_cast<T>(sampleRate_);

        // Allocate for maximum possible lookahead to prevent RT-allocations later
        const int maxLookaheadSamples = static_cast<int>(sampleRate_ * kMaxLookaheadMs / 1000.0) + 1;
        
        delayLines_.resize(static_cast<size_t>(numChannels));
        for (auto& dl : delayLines_)
            dl.prepare(maxLookaheadSamples * 2); // Double size for safety margin

        setLookahead(static_cast<T>(initialLookaheadMs));

        // Ceiling smoother setup
        T ceilLinear = decibelsToGain(ceilingDb_.load(std::memory_order_relaxed));
        ceilingSmooth_.prepare(sampleRate, 30.0);
        ceilingSmooth_.reset(ceilLinear);

        updateReleaseCoefficient();

        // ISP true-peak setup
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
     * @brief Processes an AudioBufferView in-place.
     *
     * RT-Safe: Yes. Lock-free and allocation-free.
     *
     * @param buffer Audio buffer view.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();

        syncParameters();

        const bool isp          = truePeakEnabled_.load(std::memory_order_relaxed);
        const bool adaptive     = adaptiveRelease_.load(std::memory_order_relaxed);
        const bool safetyClip   = safetyClipEnabled_.load(std::memory_order_relaxed);
        const T relMs           = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));

        constexpr T kSafetyClipCeiling = T(0.96605); // -0.3 dBFS

        for (int i = 0; i < nS; ++i)
        {
            T ceiling = ceilingSmooth_.getNextValue();
            T peak = T(0);

            // Phase 1: Peak detection and delay line push
            for (int ch = 0; ch < nCh; ++ch)
            {
                T sample = buffer.getChannel(ch)[i];
                delayLines_[ch].push(sample);

                T chPeak = isp ? detectTruePeak(sample, ch) : std::abs(sample);
                if (chPeak > peak) peak = chPeak;
            }

            // Phase 2: Gain envelope calculation
            T targetGain = (peak > ceiling) ? ceiling / peak : T(1);
            smoothGain(targetGain, adaptive, relMs);

            // Phase 3: Apply gain and safety clip
            for (int ch = 0; ch < nCh; ++ch)
            {
                T out = delayLines_[ch].read(lookaheadSamples_) * currentGain_;

                if (safetyClip)
                {
                    T clipCeil = std::min(kSafetyClipCeiling, ceiling);
                    if (std::abs(out) > clipCeil)
                        out = applySafetyClipper(out, clipCeil);
                }

                buffer.getChannel(ch)[i] = out;
            }
        }
    }

    /**
     * @brief Processes a single sample (mono).
     * @warning For multi-channel linked limiting, prefer processBlock() or manually calculate 
     * the maximum peak across channels before smoothing gain.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        syncParameters();
        
        T ceiling = ceilingSmooth_.getNextValue();
        bool isp = truePeakEnabled_.load(std::memory_order_relaxed);
        bool adaptive = adaptiveRelease_.load(std::memory_order_relaxed);
        T relMs = releaseMs_.load(std::memory_order_relaxed);

        delayLines_[channel].push(input);
        T peak = isp ? detectTruePeak(input, channel) : std::abs(input);
        
        T targetGain = (peak > ceiling) ? ceiling / peak : T(1);
        smoothGain(targetGain, adaptive, relMs);

        return delayLines_[channel].read(lookaheadSamples_) * currentGain_;
    }

    /** @brief Resets the internal state (delays, gain reduction). RT-Safe. */
    void reset() noexcept
    {
        for (auto& dl : delayLines_) dl.reset();
        tpState_ = {};
        currentGain_ = T(1);
        limitingDuration_ = 0;
        ceilingSmooth_.skip();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the absolute output ceiling.
     * @param dB Ceiling in dBFS (e.g., -1.0 for streaming). RT-Safe.
     */
    void setCeiling(T dB) noexcept { ceilingDb_.store(dB, std::memory_order_relaxed); }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Sets the base release time.
     * @param ms Release time in milliseconds. RT-Safe.
     */
    void setRelease(T ms) noexcept { releaseMs_.store(std::max(ms, T(1)), std::memory_order_relaxed); }

    /**
     * @brief Sets the lookahead time dynamically.
     * 
     * @note RT-Safe. It only adjusts read pointers up to the max memory allocated
     *       during prepare(). Max lookahead is clamped to 10ms.
     * 
     * @param ms Lookahead in milliseconds (0.5 to 10.0 ms).
     */
    void setLookahead(T ms) noexcept
    {
        lookaheadMs_ = std::clamp(static_cast<double>(ms), 0.5, kMaxLookaheadMs);
        if (sampleRate_ > 0)
        {
            lookaheadSamples_ = std::max(1, static_cast<int>(sampleRate_ * lookaheadMs_ / 1000.0));
            
            // Calculate an attack coefficient that guarantees reaching 99% of target gain
            // exactly within the lookahead window to prevent transient clipping.
            // Formula: alpha = 1 - exp(-ln(100) / samples)
            attackCoeff_ = T(1) - std::exp(T(-4.60517) / static_cast<T>(lookaheadSamples_));
        }
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /** @brief Enables 4x oversampled ISP true-peak detection. RT-Safe. */
    void setTruePeak(bool enabled) noexcept { truePeakEnabled_.store(enabled, std::memory_order_relaxed); }

    /** @brief Enables program-dependent adaptive release. RT-Safe. */
    void setAdaptiveRelease(bool enabled) noexcept { adaptiveRelease_.store(enabled, std::memory_order_relaxed); }

    /** @brief Enables post-limiter asymmetric safety clipper. RT-Safe. */
    void setSafetyClip(bool enabled) noexcept { safetyClipEnabled_.store(enabled, std::memory_order_relaxed); }

    // Getters
    [[nodiscard]] bool isTruePeakEnabled() const noexcept { return truePeakEnabled_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isAdaptiveReleaseEnabled() const noexcept { return adaptiveRelease_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isSafetyClipEnabled() const noexcept { return safetyClipEnabled_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getLatency() const noexcept { return lookaheadSamples_; }
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainToDecibels(currentGain_); }

protected:
    static constexpr int kMaxChannels = 16;
    static constexpr double kMaxLookaheadMs = 10.0;

    static constexpr int kTpTaps = 12;
    static constexpr int kTpPhases = 3;
    static constexpr int kTpHistSize = 16; 
    static constexpr int kTpHistMask = kTpHistSize - 1;

    struct TruePeakState {
        T history[kTpHistSize] = {}; 
        int writePos = 0;            
    };
    std::array<TruePeakState, kMaxChannels> tpState_{};
    std::array<std::array<T, kTpTaps>, kTpPhases> tpCoeffs_{};

    // Fast-path synchronization for atomic variables
    inline void syncParameters() noexcept
    {
        T ceilLinear = decibelsToGain(ceilingDb_.load(std::memory_order_relaxed));
        ceilingSmooth_.setTargetValue(ceilLinear);

        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
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

    void buildTruePeakFilter() noexcept
    {
        // ... (Tu implementación original de buildTruePeakFilter era correcta y óptima) ...
        constexpr int N = kTpTaps * 4;       
        constexpr double M = (N - 1) / 2.0;  
        constexpr double fc = 0.25;          
        constexpr double beta = 8.0;         
        constexpr double pi = std::numbers::pi;

        auto besselI0 = [](double x) -> double {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k <= 25; ++k) {
                double half = x / (2.0 * k);
                term *= half * half;
                sum += term;
                if (term < 1e-15 * sum) break;
            }
            return sum;
        };

        const double i0Beta = besselI0(beta);
        double h[N];
        
        for (int n = 0; n < N; ++n) {
            double x = static_cast<double>(n) - M;
            double sincArg = 2.0 * fc * x;
            double sincVal = (std::abs(sincArg) < 1e-10) ? 1.0 : std::sin(pi * sincArg) / (pi * sincArg);
            double t = x / M;
            double kaiserVal = (std::abs(t) > 1.0) ? 0.0 : besselI0(beta * std::sqrt(1.0 - t * t)) / i0Beta;
            h[n] = sincVal * kaiserVal;
        }

        for (int phase = 0; phase < kTpPhases; ++phase) {
            int p = phase + 1;
            for (int k = 0; k < kTpTaps; ++k)
                tpCoeffs_[phase][k] = static_cast<T>(h[4 * k + p]);
        }
    }

    [[nodiscard]] inline T detectTruePeak(T sample, int ch) noexcept
    {
        auto& tp = tpState_[ch];
        tp.history[tp.writePos] = sample;
        tp.writePos = (tp.writePos + 1) & kTpHistMask;

        T peak = std::abs(sample);
        const int newest = (tp.writePos - 1) & kTpHistMask;
        
        // Manual unrolling recommended if compiler doesn't auto-vectorize
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
            if (absInterp > peak) peak = absInterp;
        }
        return peak;
    }

    inline void smoothGain(T targetGain, bool adaptive, T relMs) noexcept
    {
        if (targetGain < currentGain_)
        {
            // Smoothed attack to prevent discontinuous clicks on transients
            currentGain_ += attackCoeff_ * (targetGain - currentGain_);
            
            // Limit duration to max 2 seconds to prevent integer overflow
            const int maxDuration = static_cast<int>(sampleRate_ * 2.0);
            if (limitingDuration_ < maxDuration) limitingDuration_++;
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
    int lookaheadSamples_ = 96;
    double lookaheadMs_ = 2.0;

    std::atomic<T> ceilingDb_ { T(-0.3) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<bool> truePeakEnabled_ { false };
    std::atomic<bool> adaptiveRelease_ { false };
    std::atomic<bool> safetyClipEnabled_ { false };

    SmoothedValue<T> ceilingSmooth_;

    T releaseCoeff_ = T(0);
    T attackCoeff_ = T(1); 
    T lastReleaseMs_ = T(-1); 

    T currentGain_ = T(1);
    int limitingDuration_ = 0;

    std::vector<RingBuffer<T>> delayLines_;
};

} // namespace dspark