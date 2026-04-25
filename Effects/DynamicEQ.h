// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DynamicEQ.h
 * @brief Dynamic parametric equalizer with per-band level detection.
 *
 * Each band independently detects level, then applies dynamic gain above
 * and/or below its threshold. Dual above/below threshold control per band
 * enables any combination: downward compression (cut above), upward boost
 * (boost below), expansion (cut below), or dynamic boost (boost above).
 *
 * Optional oversampling (1x/2x/4x) and lookahead (0–10 ms) for transparent
 * processing of fast transients.
 *
 * Architecture per band:
 * ```
 *   Input → [Sidechain BP filter] → Envelope → Gain Computer →
 *   → Dynamic Peak EQ (makePeak with computed gain) → Output
 * ```
 *
 * Dependencies: Biquad.h, DspMath.h, AudioSpec.h, AudioBuffer.h,
 *               Oversampling.h, RingBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::DynamicEQ<float> deq;
 *   deq.prepare(spec);
 *   deq.setNumBands(4);
 *
 *   // Band 0: cut above -10 dB at 3 kHz (de-esser style)
 *   dspark::DynamicEQ<float>::BandConfig cfg;
 *   cfg.frequency = 3000.0f;
 *   cfg.threshold = -10.0f;
 *   cfg.aboveRatio = 4.0f;
 *   cfg.aboveBoost = false;  // cut
 *   deq.setBand(0, cfg);
 *
 *   deq.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/Oversampling.h"
#include "../Core/RingBuffer.h"
#include "../Core/DenormalGuard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class DynamicEQ
 * @brief Dynamic parametric EQ with dual above/below threshold per band.
 *
 * @tparam T        Sample type (float or double).
 * @tparam MaxBands Maximum number of bands (compile-time, default 8).
 */
template <FloatType T, int MaxBands = 8>
class DynamicEQ
{
public:
    /**
     * @brief Full configuration for a single dynamic EQ band.
     */
    struct BandConfig
    {
        T frequency      = T(1000);
        T q              = T(1.0);
        T threshold      = T(-20);
        bool enabled     = true;

        // Above threshold: what happens when band level exceeds threshold
        T aboveRatio     = T(1);      ///< 1 = no action, >1 = compress/expand
        T aboveAttackMs  = T(5);
        T aboveReleaseMs = T(50);
        T aboveRangeDb   = T(12);     ///< Max gain applied above
        bool aboveBoost  = false;     ///< false = cut, true = boost

        // Below threshold: what happens when band level is below threshold
        T belowRatio     = T(1);      ///< 1 = no action, >1 = expand/gate
        T belowAttackMs  = T(10);
        T belowReleaseMs = T(100);
        T belowRangeDb   = T(12);     ///< Max gain applied below
        bool belowBoost  = false;     ///< false = cut, true = boost
    };

    // -- Lifecycle -----------------------------------------------------------

    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        sampleRate_ = spec.sampleRate;

        // Oversampling
        if (oversamplingFactor_ > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(oversamplingFactor_);
            oversampler_->prepare(spec);
        }

        // Lookahead ring buffers
        int maxLaSamples = static_cast<int>(sampleRate_ * 0.01) + 1;
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].prepare(maxLaSamples);
        updateLookahead();

        // 1 ms one-pole coefficient-smoother. The per-sample gainDb is
        // continuous, but bandFilter_ coefficients refresh every
        // kCoefRefreshInterval samples (C5 optimisation). The 1 ms smoother
        // keeps the filter-response path continuous and removes zipper
        // artefacts that would otherwise appear between refreshes.
        if (sampleRate_ > 0)
            coefSmoothCoeff_ = static_cast<T>(
                1.0 - std::exp(-1.0 / (sampleRate_ * 0.001)));

        // Reset per-band state
        for (int b = 0; b < MaxBands; ++b)
        {
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                bandEnvelope_[b][ch] = T(0);
                aboveEnv_[b][ch] = T(0);
                belowEnv_[b][ch] = T(0);
            }
            bandDetector_[b].reset();
            bandFilter_[b].reset();
            smoothedLinkedGainDb_[b] = T(0);
            updateBandCoefficients(b);
        }
    }

    /**
     * @brief Processes audio in-place (self-sidechain).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlockImpl(buffer, buffer);
    }

    /**
     * @brief Processes audio with external sidechain for detection.
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        processBlockImpl(audio, sidechain);
    }

    // -- Configuration -------------------------------------------------------

    void setBand(int band, const BandConfig& config) noexcept
    {
        if (band < 0 || band >= MaxBands) return;
        configs_[band] = config;
        if (sampleRate_ > 0)
            updateBandCoefficients(band);
    }

    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 1, MaxBands), std::memory_order_relaxed);
    }

    /**
     * @brief Sets oversampling factor (1, 2, or 4).
     *
     * Call prepare() after changing.
     */
    void setOversampling(int factor) noexcept
    {
        oversamplingFactor_ = std::clamp(factor, 1, 4);
        // Round to power of 2
        if (oversamplingFactor_ == 3) oversamplingFactor_ = 4;
    }

    /**
     * @brief Sets lookahead time in milliseconds (0–10 ms).
     *
     * Adds latency but allows more transparent dynamic processing.
     */
    void setLookahead(T ms) noexcept
    {
        lookaheadMs_ = std::clamp(ms, T(0), T(10));
        updateLookahead();
    }

    // -- Queries -------------------------------------------------------------

    [[nodiscard]] T getBandGainDb(int band) const noexcept
    {
        if (band < 0 || band >= MaxBands) return T(0);
        return bandGainDb_[band];
    }

    [[nodiscard]] int getLatency() const noexcept
    {
        int la = lookaheadSamples_;
        if (oversampler_) la += oversampler_->getLatency();
        return la;
    }

    [[nodiscard]] int getNumBands() const noexcept { return numBands_.load(std::memory_order_relaxed); }

    void reset() noexcept
    {
        for (int b = 0; b < MaxBands; ++b)
        {
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                bandEnvelope_[b][ch] = T(0);
                aboveEnv_[b][ch] = T(0);
                belowEnv_[b][ch] = T(0);
            }
            bandDetector_[b].reset();
            bandFilter_[b].reset();
            bandGainDb_[b] = T(0);
            smoothedLinkedGainDb_[b] = T(0);
        }
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].reset();
    }

private:
    static constexpr int kMaxChannels = 16;

    // -- Core processing -----------------------------------------------------

    void processBlockImpl(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        DenormalGuard guard;
        const int nCh  = std::min(audio.getNumChannels(), kMaxChannels);
        const int scCh = sidechain.getNumChannels();
        const int nS   = audio.getNumSamples();
        const int nb   = numBands_.load(std::memory_order_relaxed);
        const int laSamples = lookaheadSamples_.load(std::memory_order_relaxed);

        // Coefficient refresh interval: the dynamic peak-EQ biquad rebuilds
        // its coefficients every kCoefRefreshInterval samples instead of per
        // sample. At 48 kHz, 16 samples = 333 µs — far below perceptual
        // thresholds — and the 1 ms smoother on smoothedLinkedGainDb_ keeps
        // the coefficient path continuous. Net cost: ~16× fewer makePeak()
        // calls (trig + exp) in the hot loop.
        constexpr int kCoefRefreshInterval = 16;
        const T coefSmooth = coefSmoothCoeff_;

        for (int i = 0; i < nS; ++i)
        {
            // Stage 1: per-band detection, per-channel envelope, stereo-linked
            // smoothed gainDb, and coefficient refresh at the chosen interval.
            //
            // Stereo-linking the gainDb (max magnitude across channels) is the
            // pro-industry default (Waves F6, FabFilter Pro-Q 3, TDR Nova, Ozone
            // Dynamic EQ): it preserves sensitivity to whichever channel has
            // more signal while keeping the filter response identical across
            // channels — which preserves the stereo image.
            for (int b = 0; b < nb; ++b)
            {
                if (!configs_[b].enabled) continue;

                T linkedMag = T(0);
                T signedLinked = T(0);

                for (int ch = 0; ch < nCh; ++ch)
                {
                    int sc = std::min(ch, scCh - 1);
                    T scSample = sidechain.getChannel(sc)[i];

                    // Detect: bandpass filter on sidechain → envelope
                    T detected = bandDetector_[b].processSample(scSample, ch);
                    T absDet = std::abs(detected);

                    // Envelope follower with per-band attack/release
                    T& env = bandEnvelope_[b][ch];
                    T envCoeff = (absDet > env) ? bandAttackCoeff_[b] : bandReleaseCoeff_[b];
                    env += envCoeff * (absDet - env);

                    T levelDb = gainToDecibels(env);

                    // Per-channel dynamic gain
                    T dynGainDb = computeBandGain(b, levelDb);

                    // Link via max magnitude across channels
                    T mag = std::abs(dynGainDb);
                    if (mag > linkedMag)
                    {
                        linkedMag = mag;
                        signedLinked = dynGainDb;
                    }
                }

                // 1 ms one-pole smoother on the linked gainDb → fed to coefs
                T& smoothed = smoothedLinkedGainDb_[b];
                smoothed += coefSmooth * (signedLinked - smoothed);

                // Metering uses the smoothed, linked value
                bandGainDb_[b] = smoothed;

                // Refresh peak-EQ coefficients only every N samples
                if ((i & (kCoefRefreshInterval - 1)) == 0)
                {
                    if (std::abs(smoothed) > T(0.01))
                    {
                        auto coeffs = BiquadCoeffs<T>::makePeak(
                            sampleRate_,
                            static_cast<double>(configs_[b].frequency),
                            static_cast<double>(configs_[b].q),
                            static_cast<double>(smoothed));
                        bandFilter_[b].setCoeffs(coeffs);
                    }
                    else
                    {
                        // Near-unity: flat biquad (bypass)
                        bandFilter_[b].setCoeffs(BiquadCoeffs<T>{});
                    }
                }
            }

            // Stage 2: apply (now stationary within the refresh interval)
            // filter per-channel. Filter state itself remains per-channel.
            for (int ch = 0; ch < nCh; ++ch)
            {
                T audioSample;
                if (laSamples > 0)
                {
                    lookaheadBuf_[ch].push(audio.getChannel(ch)[i]);
                    audioSample = lookaheadBuf_[ch].read(laSamples);
                }
                else
                {
                    audioSample = audio.getChannel(ch)[i];
                }

                T output = audioSample;
                for (int b = 0; b < nb; ++b)
                {
                    if (!configs_[b].enabled) continue;
                    output = bandFilter_[b].processSample(output, ch);
                }

                audio.getChannel(ch)[i] = output;
            }
        }
    }

    /**
     * @brief Computes the dynamic gain (dB) for a band based on detected level.
     *
     * Dual above/below threshold control:
     * - Above: overDb * (1 - 1/ratio), clamped to rangeDb, with sign from boost flag.
     * - Below: underDb * (1 - 1/ratio), clamped to rangeDb, with sign from boost flag.
     */
    [[nodiscard]] T computeBandGain(int b, T levelDb) const noexcept
    {
        const auto& cfg = configs_[b];
        T gainDb = T(0);

        if (levelDb > cfg.threshold)
        {
            // Above threshold
            if (cfg.aboveRatio > T(1.001))
            {
                T overDb = levelDb - cfg.threshold;
                T amount = overDb * (T(1) - T(1) / cfg.aboveRatio);
                amount = std::min(amount, cfg.aboveRangeDb);
                gainDb += cfg.aboveBoost ? amount : -amount;
            }
        }
        else
        {
            // Below threshold
            if (cfg.belowRatio > T(1.001))
            {
                T underDb = cfg.threshold - levelDb;
                T amount = underDb * (T(1) - T(1) / cfg.belowRatio);
                amount = std::min(amount, cfg.belowRangeDb);
                gainDb += cfg.belowBoost ? amount : -amount;
            }
        }

        return gainDb;
    }

    // -- Coefficient helpers -------------------------------------------------

    void updateBandCoefficients(int b) noexcept
    {
        if (sampleRate_ <= 0) return;
        T fs = static_cast<T>(sampleRate_);

        // Bandpass detector coefficients
        auto bpCoeffs = BiquadCoeffs<T>::makeBandPass(
            sampleRate_,
            static_cast<double>(configs_[b].frequency),
            static_cast<double>(configs_[b].q));
        bandDetector_[b].setCoeffs(bpCoeffs);

        // Average of above/below attack/release for the envelope
        T atkMs = (configs_[b].aboveAttackMs + configs_[b].belowAttackMs) / T(2);
        T relMs = (configs_[b].aboveReleaseMs + configs_[b].belowReleaseMs) / T(2);

        bandAttackCoeff_[b]  = T(1) - std::exp(T(-1) / (fs * std::max(atkMs, T(0.01)) / T(1000)));
        bandReleaseCoeff_[b] = T(1) - std::exp(T(-1) / (fs * std::max(relMs, T(0.01)) / T(1000)));
    }

    void updateLookahead() noexcept
    {
        if (sampleRate_ > 0)
            lookaheadSamples_.store(static_cast<int>(
                static_cast<T>(sampleRate_) * lookaheadMs_ / T(1000)),
                std::memory_order_relaxed);
    }

    // -- Members -------------------------------------------------------------

    AudioSpec spec_ {};
    double sampleRate_ = 0;
    std::atomic<int> numBands_ { 0 };

    std::array<BandConfig, MaxBands> configs_ {};

    // Per-band detection and filtering
    std::array<Biquad<T>, MaxBands> bandDetector_ {};
    std::array<Biquad<T>, MaxBands> bandFilter_ {};
    std::array<std::array<T, kMaxChannels>, MaxBands> bandEnvelope_ {};
    std::array<std::array<T, kMaxChannels>, MaxBands> aboveEnv_ {};
    std::array<std::array<T, kMaxChannels>, MaxBands> belowEnv_ {};
    std::array<T, MaxBands> bandGainDb_ {};
    std::array<T, MaxBands> bandAttackCoeff_ {};
    std::array<T, MaxBands> bandReleaseCoeff_ {};
    std::array<T, MaxBands> smoothedLinkedGainDb_ {};   ///< 1 ms-smoothed linked gainDb (C5: fed to refreshed coeffs)
    T coefSmoothCoeff_ = T(0);                         ///< 1 ms one-pole coefficient for the smoother above

    // Oversampling
    int oversamplingFactor_ = 1;
    std::unique_ptr<Oversampling<T>> oversampler_;

    // Lookahead
    T lookaheadMs_ = T(0);
    std::atomic<int> lookaheadSamples_ { 0 };
    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuf_ {};
};

} // namespace dspark
