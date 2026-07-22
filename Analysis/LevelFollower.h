// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LevelFollower.h
 * @brief Envelope follower for real-time peak and RMS level metering.
 *
 * Tracks per-channel peak and RMS levels with lock-free readout for UI
 * metering. Peak uses an asymmetric attack/release one-pole; RMS uses a
 * symmetric one-pole exponential integrator over the squared signal (a
 * VU-style exponential window whose time constant equals the configured
 * window time - not a rectangular moving average). A small constant offset
 * keeps both recursions out of the subnormal range on decay tails; its
 * equilibrium floor sits far below the -100 dB readout floor.
 *
 * Threading:
 * - prepare() / reset(): setup thread (not concurrent with process()).
 * - setAttackMs() / setReleaseMs() / setRmsWindowMs(): any thread.
 *   Parameters and coefficients are published atomically; process() picks
 *   them up at the next block. A block may observe one changed coefficient
 *   before another (benign for metering). Non-finite values are ignored.
 * - process(): audio thread (stream owner).
 * - getPeakLevel() / getRmsLevel() / *Db(): any thread (relaxed atomic
 *   reads; values are approximate while a block is in flight - metering).
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
 * meter.process(buffer.toView());
 *
 * // In UI Thread (Safe, lock-free reading):
 * float peakDb = meter.getPeakLevelDb(0);
 * float rmsDb  = meter.getRmsLevelDb(0);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <type_traits>

namespace dspark {

/**
 * @class LevelFollower
 * @brief Per-channel peak and RMS envelope follower with lock-free readout.
 *
 * One-pole envelope followers with a branchless attack/release selector in
 * the hot loop (avoids branch mispredictions on noisy program material; the
 * per-sample recursion itself is inherently serial). Channels beyond the
 * prepared count (or beyond MaxChannels) are not metered. A non-finite
 * input sample would stick in the recursions forever, so the published
 * state is sanitized once per block: the meter recovers on the next block
 * instead of reporting NaN until reset().
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels supported.
 */
template <FloatType T, int MaxChannels = 16>
class LevelFollower
{
public:
    /**
     * @brief Prepares the follower for the given audio environment.
     *
     * Recomputes coefficients for the new sample rate and resets all
     * envelope states. An invalid spec (non-positive or non-finite fields)
     * is ignored, keeping the previous state.
     *
     * @param spec Audio specification (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid())
            return;

        sampleRate_  = spec.sampleRate;
        numChannels_ = std::min(spec.numChannels, MaxChannels);
        updateCoefficients();
        reset();
    }

    /**
     * @brief Sets the attack time for peak metering.
     * @param ms Attack time in milliseconds (floored to 0.001; non-finite
     *           values are ignored).
     */
    void setAttackMs(float ms) noexcept
    {
        if (!std::isfinite(ms))
            return;
        attackMs_.store(std::max(T(0.001), static_cast<T>(ms)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /**
     * @brief Sets the release time for peak metering.
     * @param ms Release time in milliseconds (floored to 0.001; non-finite
     *           values are ignored).
     */
    void setReleaseMs(float ms) noexcept
    {
        if (!std::isfinite(ms))
            return;
        releaseMs_.store(std::max(T(0.001), static_cast<T>(ms)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /**
     * @brief Sets the integration time constant for RMS metering.
     *
     * The RMS detector is a one-pole exponential integrator over the squared
     * signal; this is its time constant (63% settling on a level step), not
     * the length of a rectangular window.
     *
     * @param ms Time constant in milliseconds (floored to 0.001; non-finite
     *           values are ignored).
     */
    void setRmsWindowMs(float ms) noexcept
    {
        if (!std::isfinite(ms))
            return;
        rmsWindowMs_.store(std::max(T(0.001), static_cast<T>(ms)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /** @brief Returns the peak attack time in milliseconds. */
    [[nodiscard]] T getAttackMs() const noexcept
    {
        return attackMs_.load(std::memory_order_relaxed);
    }

    /** @brief Returns the peak release time in milliseconds. */
    [[nodiscard]] T getReleaseMs() const noexcept
    {
        return releaseMs_.load(std::memory_order_relaxed);
    }

    /** @brief Returns the RMS integration time constant in milliseconds. */
    [[nodiscard]] T getRmsWindowMs() const noexcept
    {
        return rmsWindowMs_.load(std::memory_order_relaxed);
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
     *
     * No-op before prepare(). Channels beyond the prepared count are not
     * metered.
     *
     * @param buffer Read-only audio buffer view.
     */
    void process(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS  = buffer.getNumSamples();

        // Constant offset that keeps the decay recursions out of the
        // subnormal range (CPU spikes). Its equilibrium floor is
        // kAntiDenormal / (1 - coeff), far below the -100 dB readout floor.
        static constexpr T kAntiDenormal = std::is_same_v<T, float> ? T(1e-15) : T(1e-30);

        // One relaxed load per block; the per-sample loop runs on locals.
        const T attackCoeff  = attackCoeff_.load(std::memory_order_relaxed);
        const T releaseCoeff = releaseCoeff_.load(std::memory_order_relaxed);
        const T rmsCoeff     = rmsCoeff_.load(std::memory_order_relaxed);

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

                // Branchless attack/release selector (avoids per-sample
                // branch mispredictions; the recursion itself is serial)
                const T isAttack  = static_cast<T>(absSample > localPeak);
                const T peakCoeff = isAttack * attackCoeff + (T(1) - isAttack) * releaseCoeff;

                localPeak = absSample + peakCoeff * (localPeak - absSample) + kAntiDenormal;

                // Symmetric one-pole lowpass over x^2 (exponential RMS)
                const T squared = data[i] * data[i];
                localRms = squared + rmsCoeff * (localRms - squared) + kAntiDenormal;
            }

            // A non-finite input would otherwise stick in the recursions
            // forever (the one-pole never drains a NaN); a meter must report
            // the signal as it is now, so sanitize at publish time and start
            // clean on the next block.
            if (!std::isfinite(localPeak)) localPeak = T(0);
            if (!std::isfinite(localRms))  localRms  = T(0);

            // Publish final states for the UI thread once per block
            s.peak.store(localPeak, std::memory_order_relaxed);
            s.rmsAccum.store(localRms, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Returns the current peak level for the given channel safely.
     * @param channel Channel index (0-based; out-of-range returns 0).
     * @return Peak level (linear, >= 0).
     */
    [[nodiscard]] T getPeakLevel(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return T(0);
        return state_[channel].peak.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the current RMS level for the given channel safely.
     * @param channel Channel index (0-based; out-of-range returns 0).
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
     * @return Peak level in dB (floored at -100).
     */
    [[nodiscard]] T getPeakLevelDb(int channel) const noexcept
    {
        return gainToDecibels(getPeakLevel(channel));
    }

    /**
     * @brief Returns the current RMS level in decibels (optimized).
     * @param channel Channel index (0-based).
     * @return RMS level in dB (floored at -100).
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
        if (!(sampleRate_ > 0.0))
            return;
        const auto fs = static_cast<T>(sampleRate_);

        const T attackMs  = attackMs_.load(std::memory_order_relaxed);
        const T releaseMs = releaseMs_.load(std::memory_order_relaxed);
        const T rmsMs     = rmsWindowMs_.load(std::memory_order_relaxed);

        attackCoeff_.store(std::exp(T(-1) / (fs * attackMs / T(1000))), std::memory_order_relaxed);
        releaseCoeff_.store(std::exp(T(-1) / (fs * releaseMs / T(1000))), std::memory_order_relaxed);
        rmsCoeff_.store(std::exp(T(-1) / (fs * rmsMs / T(1000))), std::memory_order_relaxed);
    }

    struct ChannelState
    {
        // Written once per block by the audio thread, read at UI rate:
        // contention is negligible, so no cache-line padding is needed.
        std::atomic<T> peak{ T(0) };
        std::atomic<T> rmsAccum{ T(0) };
    };

    double sampleRate_  = 44100.0;
    int    numChannels_ = 0;

    std::atomic<T> attackMs_    { T(1) };
    std::atomic<T> releaseMs_   { T(100) };
    std::atomic<T> rmsWindowMs_ { T(300) };

    std::atomic<T> attackCoeff_  { T(0) };
    std::atomic<T> releaseCoeff_ { T(0) };
    std::atomic<T> rmsCoeff_     { T(0) };

    std::array<ChannelState, MaxChannels> state_{};
};

} // namespace dspark
