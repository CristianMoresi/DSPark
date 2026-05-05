// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file LevelFollower.h
 * @brief Envelope follower for real-time peak and RMS level metering.
 *
 * Provides thread-safe, lock-free metering tracking for peak and RMS levels.
 * Optimized with branchless SIMD-friendly DSP paths and subnormal protection.
 * RMS is calculated using a symmetric moving window, while Peak tracking
 * uses asymmetric attack/release.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h.
 *
 * @code
 * dspark::LevelFollower<float> meter;
 * meter.prepare(spec);
 * meter.setAttackMs(1.0f);
 * meter.setReleaseMs(100.0f);
 * meter.setRmsWindowMs(300.0f);
 *
 * // In process() (Audio Thread):
 * meter.process(buffer.toConstView());
 *
 * // In UI Thread (Safe, lock-free reading):
 * float peakDb = meter.getPeakLevelDb(0);
 * float rmsDb  = meter.getRmsLevelDb(0);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <array>
#include <cmath>
#include <atomic>
#include <algorithm>

namespace dspark {

/**
 * @class LevelFollower
 * @brief Per-channel peak and RMS envelope follower with lock-free readout.
 *
 * Uses one-pole smoothing. Avoids branch prediction failures in the hot path
 * and protects against CPU denormalization spikes. Thread-safe for UI metering.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels supported.
 */
template <typename T, int MaxChannels = 16>
class LevelFollower
{
public:
    /**
     * @brief Prepares the follower for the given audio environment.
     * @param spec Audio specification (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_  = spec.sampleRate;
        numChannels_ = std::clamp(spec.numChannels, 1, MaxChannels);
        updateCoefficients();
        reset();
    }

    /**
     * @brief Sets the attack time for peak metering.
     * @param ms Attack time in milliseconds (must be > 0).
     */
    void setAttackMs(float ms) noexcept
    {
        attackMs_ = std::max(T(0.001), static_cast<T>(ms));
        updateCoefficients();
    }

    /**
     * @brief Sets the release time for peak metering.
     * @param ms Release time in milliseconds (must be > 0).
     */
    void setReleaseMs(float ms) noexcept
    {
        releaseMs_ = std::max(T(0.001), static_cast<T>(ms));
        updateCoefficients();
    }

    /**
     * @brief Sets the integration window for true RMS metering.
     * @param ms Window time in milliseconds (must be > 0).
     */
    void setRmsWindowMs(float ms) noexcept
    {
        rmsWindowMs_ = std::max(T(0.001), static_cast<T>(ms));
        updateCoefficients();
    }

    /** @brief Resets all envelope states to zero safely. */
    void reset() noexcept
    {
        for (auto& s : state_)
        {
            s.peak.store(T(0), std::memory_order_relaxed);
            s.rmsAccum.store(T(0), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Processes a block of audio and updates level tracking.
     * @param buffer Read-only audio buffer view.
     */
    void process(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS  = buffer.getNumSamples();
        
        // Anti-denormal offset to prevent extreme CPU spikes on decay
        static constexpr T kAntiDenormal = std::is_same_v<T, float> ? T(1e-15) : T(1e-30);

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* data = buffer.getChannel(ch);
            auto& s = state_[ch];

            // Local copies to avoid atomic operations per sample
            T localPeak = s.peak.load(std::memory_order_relaxed);
            T localRms  = s.rmsAccum.load(std::memory_order_relaxed);

            for (int i = 0; i < nS; ++i)
            {
                const T absSample = std::abs(data[i]);

                // Branchless Peak selector for SIMD vectorization
                const T isAttack  = static_cast<T>(absSample > localPeak);
                const T peakCoeff = isAttack * attackCoeff_ + (T(1) - isAttack) * releaseCoeff_;
                
                localPeak = absSample + peakCoeff * (localPeak - absSample) + kAntiDenormal;

                // Symmetric one-pole lowpass for True RMS integration
                const T squared = data[i] * data[i];
                localRms = squared + rmsCoeff_ * (localRms - squared) + kAntiDenormal;
            }

            // Publish final states for the UI thread once per block
            s.peak.store(localPeak, std::memory_order_relaxed);
            s.rmsAccum.store(localRms, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Returns the current peak level for the given channel safely.
     * @param channel Channel index (0-based).
     * @return Peak level (linear, >= 0).
     */
    [[nodiscard]] T getPeakLevel(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(0);
        return state_[channel].peak.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current RMS level for the given channel safely.
     * @param channel Channel index (0-based).
     * @return RMS level (linear, >= 0).
     */
    [[nodiscard]] T getRmsLevel(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(0);
        const T squaredRms = state_[channel].rmsAccum.load(std::memory_order_relaxed);
        return std::sqrt(std::max(squaredRms, T(0)));
    }

    /**
     * @brief Returns the current peak level in decibels.
     * @param channel Channel index (0-based).
     * @return Peak level in dB.
     */
    [[nodiscard]] T getPeakLevelDb(int channel) const noexcept
    {
        return gainToDecibels(getPeakLevel(channel));
    }

    /**
     * @brief Returns the current RMS level in decibels (optimized).
     * @param channel Channel index (0-based).
     * @return RMS level in dB.
     */
    [[nodiscard]] T getRmsLevelDb(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(-100);
        const T squaredRms = state_[channel].rmsAccum.load(std::memory_order_relaxed);
        
        // Fast path: avoid std::sqrt by using 10*log10(x^2) instead of 20*log10(x)
        if (squaredRms <= T(1e-10)) return T(-100); 
        return T(10) * std::log10(squaredRms);
    }

private:
    void updateCoefficients() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        const auto fs = static_cast<T>(sampleRate_);
        
        attackCoeff_  = std::exp(T(-1) / (fs * attackMs_  / T(1000)));
        releaseCoeff_ = std::exp(T(-1) / (fs * releaseMs_ / T(1000)));
        rmsCoeff_     = std::exp(T(-1) / (fs * rmsWindowMs_ / T(1000)));
    }

    struct ChannelState
    {
        // Padded and atomic to prevent false sharing and data races
        alignas(32) std::atomic<T> peak{ T(0) };
        std::atomic<T> rmsAccum{ T(0) };
    };

    double sampleRate_  = 44100.0;
    int    numChannels_ = 0;
    
    T attackMs_    = T(1);
    T releaseMs_   = T(100);
    T rmsWindowMs_ = T(300);

    T attackCoeff_  = T(0);
    T releaseCoeff_ = T(0);
    T rmsCoeff_     = T(0);

    std::array<ChannelState, MaxChannels> state_{};
};

} // namespace dspark