// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DynamicEQ.h
 * @brief Dynamic parametric equalizer with per-band level detection and true VCA ballistics.
 *
 * Implements a high-performance dynamic EQ. Unlike standard envelope followers,
 * this implementation applies attack/release ballistics directly to the gain reduction
 * signal, accurately modeling analog VCA behavior and allowing true independent
 * times for Above and Below threshold processing.
 *
 * Architecture per band:
 * ```
 *   Sidechain → [BP Filter] → RMS/Peak Detect → Gain Computer → Ballistics Smoother → [Peak EQ]
 * ```
 *
 * @note Thread-safe parameter updates via `std::atomic` ensuring zero-locking
 *       in the real-time audio thread.
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
#include <type_traits>

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
     * @note Must remain trivially copyable to allow lock-free atomic updates.
     */
    struct BandConfig
    {
        T frequency      = T(1000);
        T q              = T(1.0);
        T threshold      = T(-20);
        bool enabled     = true;

        T aboveRatio     = T(1);
        T aboveAttackMs  = T(5);
        T aboveReleaseMs = T(50);
        T aboveRangeDb   = T(12);
        bool aboveBoost  = false;

        T belowRatio     = T(1);
        T belowAttackMs  = T(10);
        T belowReleaseMs = T(100);
        T belowRangeDb   = T(12);
        bool belowBoost  = false;
    };

    static_assert(std::is_trivially_copyable_v<BandConfig>, "BandConfig must be trivially copyable for std::atomic");

    DynamicEQ()
    {
        for (int i = 0; i < MaxBands; ++i) {
            configs_[i].store(BandConfig{}, std::memory_order_relaxed);
            paramsDirty_[i].store(true, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Initializes the dynamic EQ, allocating ring buffers and oversamplers.
     * @param spec Audio specification containing sample rate and block size.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        sampleRate_ = spec.sampleRate;

        if (oversamplingFactor_ > 1)
        {
            // Two oversamplers: the audio-path one owns the buffer that we
            // both process and downsample back; the sidechain one only acts
            // as a level-detection upsampler so its buffer is read-only.
            oversampler_   = std::make_unique<Oversampling<T>>(oversamplingFactor_);
            oversamplerSc_ = std::make_unique<Oversampling<T>>(oversamplingFactor_);
            oversampler_  ->prepare(spec);
            oversamplerSc_->prepare(spec);
        }
        else
        {
            oversampler_.reset();
            oversamplerSc_.reset();
        }

        int maxLaSamples = static_cast<int>(sampleRate_ * oversamplingFactor_ * 0.01) + 1;
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].prepare(maxLaSamples);

        updateLookahead();
        reset();
        isPrepared_ = true;
    }

    /**
     * @brief Processes audio in-place using self-sidechain.
     * @param buffer In/Out audio buffer.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlock(buffer, buffer);
    }

    /**
     * @brief Processes audio with an external sidechain.
     * @param audio In/Out audio buffer.
     * @param sidechain Reference signal used for level detection.
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        if (!isPrepared_) return;
        DenormalGuard guard;

        if (oversamplingFactor_ > 1 && oversampler_ && oversamplerSc_)
        {
            // Up-sample audio and sidechain through their dedicated oversamplers.
            // Each Oversampling instance owns one internal high-rate buffer, so
            // we cannot share one between the two streams; processing them
            // separately keeps each stream's polyphase filter state consistent.
            auto upAudio = oversampler_->upsample(audio);
            auto upSc    = oversamplerSc_->upsample(sidechain);

            processCore(upAudio, upSc, sampleRate_ * oversamplingFactor_);

            oversampler_->downsample(audio);
        }
        else
        {
            // Standard processing path
            processCore(audio, sidechain, sampleRate_);
        }
    }

    /**
     * @brief Thread-safe configuration update for a specific band.
     */
    void setBand(int band, const BandConfig& config) noexcept
    {
        if (band < 0 || band >= MaxBands) return;
        configs_[band].store(config, std::memory_order_release);
        paramsDirty_[band].store(true, std::memory_order_release);
    }

    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 1, MaxBands), std::memory_order_relaxed);
    }

    void setOversampling(int factor) noexcept
    {
        oversamplingFactor_ = std::clamp(factor, 1, 4);
        if (oversamplingFactor_ == 3) oversamplingFactor_ = 4;
        isPrepared_ = false; // Forces user to call prepare()
    }

    void setLookahead(T ms) noexcept
    {
        lookaheadMs_ = std::clamp(ms, T(0), T(10));
        updateLookahead();
    }

    [[nodiscard]] T getBandGainDb(int band) const noexcept
    {
        if (band < 0 || band >= MaxBands) return T(0);
        return meterGainDb_[band].load(std::memory_order_relaxed);
    }

    void reset() noexcept
    {
        for (int b = 0; b < MaxBands; ++b)
        {
            bandDetector_[b].reset();
            bandFilter_[b].reset();
            currentGainDb_[b] = T(0);
            meterGainDb_[b].store(T(0), std::memory_order_relaxed);
            paramsDirty_[b].store(true, std::memory_order_relaxed);
        }
        for (int ch = 0; ch < kMaxChannels; ++ch)
            lookaheadBuf_[ch].reset();
    }

private:
    static constexpr int kMaxChannels = 16;
    static constexpr T kMinLevelDb = T(-100.0);
    static constexpr T kMinEnvelope = T(1e-12); // Prevents NaN in log10

    struct BandState
    {
        BandConfig cfg;
        T aboveAtkCoeff, aboveRelCoeff;
        T belowAtkCoeff, belowRelCoeff;
    };

    void processCore(AudioBufferView<T>& audio, AudioBufferView<T>& sidechain, double currentFs) noexcept
    {
        const int nCh  = std::min(audio.getNumChannels(), kMaxChannels);
        const int scCh = sidechain.getNumChannels();
        const int nS   = audio.getNumSamples();
        const int nb   = numBands_.load(std::memory_order_relaxed);
        const int laSamples = lookaheadSamples_.load(std::memory_order_relaxed);

        // 1. Thread-Safe State Update
        for (int b = 0; b < nb; ++b)
        {
            if (paramsDirty_[b].exchange(false, std::memory_order_acquire))
                updateBandInternalState(b, currentFs);
        }

        // 2. Audio Processing Loop
        for (int i = 0; i < nS; ++i)
        {
            for (int b = 0; b < nb; ++b)
            {
                if (!states_[b].cfg.enabled) continue;
                
                T maxLevelDb = kMinLevelDb;
                
                // Sidechain Detection (Stereo Linked by Max Peak)
                for (int ch = 0; ch < nCh; ++ch)
                {
                    int sc = std::min(ch, scCh - 1);
                    T scSample = sidechain.getChannel(sc)[i];
                    
                    T detected = std::abs(bandDetector_[b].processSample(scSample, ch));
                    T levelDb = gainToDecibels(std::max(detected, kMinEnvelope));
                    
                    if (levelDb > maxLevelDb) maxLevelDb = levelDb;
                }

                // Gain Computer
                T targetGainDb = computeTargetGain(states_[b].cfg, maxLevelDb);

                // Gain Ballistics (Attack/Release applied to the Gain itself)
                T& currentGain = currentGainDb_[b];
                T diff = targetGainDb - currentGain;
                
                T coeff;
                if (maxLevelDb > states_[b].cfg.threshold) {
                    coeff = (std::abs(targetGainDb) > std::abs(currentGain)) 
                            ? states_[b].aboveAtkCoeff : states_[b].aboveRelCoeff;
                } else {
                    coeff = (std::abs(targetGainDb) > std::abs(currentGain)) 
                            ? states_[b].belowAtkCoeff : states_[b].belowRelCoeff;
                }
                
                currentGain += coeff * diff;

                // Update filter coefficients dynamically (Optimized for sample-by-sample)
                // Note: For extreme CPU optimization, this can be moved to a block-based loop
                // using AudioBuffer chunks, but kept per-sample here to ensure zero phase clicks.
                if (std::abs(currentGain) > T(0.01)) {
                    bandFilter_[b].setCoeffs(BiquadCoeffs<T>::makePeak(
                        currentFs,
                        states_[b].cfg.frequency,
                        states_[b].cfg.q,
                        currentGain));
                } else {
                    bandFilter_[b].setCoeffs(BiquadCoeffs<T>{}); // Bypass
                }

                if ((i & 63) == 0) // Sub-sample metering update
                    meterGainDb_[b].store(currentGain, std::memory_order_relaxed);
            }

            // Apply Filters
            for (int ch = 0; ch < nCh; ++ch)
            {
                T audioSample = audio.getChannel(ch)[i];
                
                if (laSamples > 0) {
                    lookaheadBuf_[ch].push(audioSample);
                    audioSample = lookaheadBuf_[ch].read(laSamples);
                }

                for (int b = 0; b < nb; ++b) {
                    if (states_[b].cfg.enabled) {
                        audioSample = bandFilter_[b].processSample(audioSample, ch);
                    }
                }
                audio.getChannel(ch)[i] = audioSample;
            }
        }
    }

    [[nodiscard]] T computeTargetGain(const BandConfig& cfg, T levelDb) const noexcept
    {
        T gainDb = T(0);

        if (levelDb > cfg.threshold)
        {
            if (cfg.aboveRatio > T(1.001)) {
                T overDb = levelDb - cfg.threshold;
                T amount = std::min(overDb * (T(1) - T(1) / cfg.aboveRatio), cfg.aboveRangeDb);
                gainDb += cfg.aboveBoost ? amount : -amount;
            }
        }
        else
        {
            if (cfg.belowRatio > T(1.001)) {
                T underDb = cfg.threshold - levelDb;
                T amount = std::min(underDb * (T(1) - T(1) / cfg.belowRatio), cfg.belowRangeDb);
                gainDb += cfg.belowBoost ? amount : -amount;
            }
        }
        return gainDb;
    }

    void updateBandInternalState(int b, double fs) noexcept
    {
        auto cfg = configs_[b].load(std::memory_order_acquire);
        states_[b].cfg = cfg;

        bandDetector_[b].setCoeffs(BiquadCoeffs<T>::makeBandPass(fs, cfg.frequency, cfg.q));

        auto calcCoeff = [fs](T ms) {
            return T(1) - std::exp(T(-1) / (fs * std::max(ms, T(0.01)) / T(1000)));
        };

        states_[b].aboveAtkCoeff = calcCoeff(cfg.aboveAttackMs);
        states_[b].aboveRelCoeff = calcCoeff(cfg.aboveReleaseMs);
        states_[b].belowAtkCoeff = calcCoeff(cfg.belowAttackMs);
        states_[b].belowRelCoeff = calcCoeff(cfg.belowReleaseMs);
    }

    void updateLookahead() noexcept
    {
        if (sampleRate_ > 0) {
            int samples = static_cast<int>(sampleRate_ * oversamplingFactor_ * lookaheadMs_ / T(1000));
            lookaheadSamples_.store(samples, std::memory_order_relaxed);
        }
    }

    // -- State & Mem ---------------------------------------------------------
    bool isPrepared_ = false;
    AudioSpec spec_ {};
    double sampleRate_ = 0;
    
    std::atomic<int> numBands_ { 0 };
    std::array<std::atomic<BandConfig>, MaxBands> configs_ {};
    std::array<std::atomic<bool>, MaxBands> paramsDirty_ {};
    std::array<BandState, MaxBands> states_ {};

    std::array<Biquad<T>, MaxBands> bandDetector_ {};
    std::array<Biquad<T>, MaxBands> bandFilter_ {};
    std::array<T, MaxBands> currentGainDb_ {};
    std::array<std::atomic<T>, MaxBands> meterGainDb_ {};

    int oversamplingFactor_ = 1;
    std::unique_ptr<Oversampling<T>> oversampler_;     // audio path
    std::unique_ptr<Oversampling<T>> oversamplerSc_;   // sidechain path

    T lookaheadMs_ = T(0);
    std::atomic<int> lookaheadSamples_ { 0 };
    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuf_ {};
};

} // namespace dspark
