// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include "DspMath.h"
#include "AudioSpec.h"

namespace dspark {

/**
 * @class Phasor
 * @brief Phase accumulator generating a [0, 1) ramp for oscillators and modulation.
 *
 * This class provides a highly optimized, SIMD-friendly phase generator. It uses
 * branch-predicted integer wrapping to avoid expensive math library calls in the hot path.
 * It is fully suited for real-time PolyBLEP/MinBLEP oscillators by supporting 
 * fractional phase offsets during hard synchronization.
 *
 * @note This class is not internally thread-safe. Parameter updates (e.g., setFrequency)
 * should be synchronized or smoothed by the caller prior to block processing.
 *
 * @tparam T Sample type (must satisfy FloatType concept).
 */
template <FloatType T>
class Phasor
{
public:
    /**
     * @brief Prepares the phasor for a given sample rate.
     * @param sampleRate Sample rate in Hz. Must be > 0.
     */
    void prepare(double sampleRate) noexcept
    {
        if (sampleRate <= 0.0) return; // Prevent Infinity/NaN propagation
        
        sampleRate_ = sampleRate;
        invSampleRate_ = 1.0 / sampleRate;
        updateIncrement();
    }

    /** * @brief Prepares from AudioSpec (unified API). 
     * @param spec Framework audio specification struct.
     */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    /**
     * @brief Sets the baseline oscillation frequency.
     * @param frequencyHz Frequency in Hz. Negative values produce a descending ramp.
     */
    void setFrequency(T frequencyHz) noexcept
    {
        frequency_ = frequencyHz;
        updateIncrement();
    }

    /**
     * @brief Returns the baseline frequency.
     * @return Current frequency in Hz.
     */
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

    /**
     * @brief Advances the phase by one sample and returns the current phase.
     * * Uses ultra-fast conditional wrapping suitable for standard audio rates.
     * @return Current phase value in [0, 1).
     */
    [[nodiscard]] T advance() noexcept
    {
        const T current = phase_;
        // wrapPhase has a fast in-range exit, so the common case (|increment| < 1)
        // costs one compare — but unlike an unbounded while-loop it can never hang
        // on a pathological frequency (>> sample rate).
        phase_ = wrapPhase(phase_ + increment_);
        return current;
    }

    /**
     * @brief Advances the phase with per-sample frequency modulation.
     *
     * Uses bitwise/truncation wrapping to safely handle arbitrarily large FM 
     * indices without unbounded loops.
     *
     * @param fmHz Instantaneous frequency deviation in Hz.
     * @return Current phase value in [0, 1).
     */
    [[nodiscard]] T advanceWithFM(T fmHz) noexcept
    {
        const T current = phase_;
        
        const T modulatedIncrement = static_cast<T>(
            static_cast<double>(frequency_ + fmHz) * invSampleRate_);

        phase_ = wrapPhase(phase_ + modulatedIncrement);

        return current;
    }

    /**
     * @brief Returns the current phase without advancing the accumulator.
     * @return Phase value in [0, 1).
     */
    [[nodiscard]] T getPhase() const noexcept { return phase_; }

    /**
     * @brief Forcibly overwrites the current phase.
     * @param newPhase Any phase value (will be safely wrapped to [0, 1)).
     */
    void setPhase(T newPhase) noexcept
    {
        phase_ = wrapPhase(newPhase);
    }

    /**
     * @brief Hard syncs the oscillator, optionally with a sub-sample offset.
     *
     * For analog modeling, the `fractionalOffset` is crucial for anti-aliasing
     * techniques (like PolyBLEP) to account for inter-sample sync events.
     *
     * @param fractionalOffset Sub-sample phase offset (default: 0).
     */
    void hardSync(T fractionalOffset = T(0)) noexcept 
    { 
        phase_ = wrapPhase(fractionalOffset); 
    }

    /**
     * @brief Soft syncs the oscillator conditionally.
     *
     * Standard implementation: resets the phase only if the current phase 
     * is beyond a specified threshold (typically the second half of the cycle).
     *
     * @param threshold Phase threshold required to allow synchronization.
     */
    void softSync(T threshold = T(0.5)) noexcept
    {
        if (phase_ >= threshold)
            phase_ = T(0);
    }

    /**
     * @brief Resets the phasor to a specific initial state (default 0).
     * @param startPhase Initial phase value.
     */
    void reset(T startPhase = T(0)) noexcept
    {
        phase_ = wrapPhase(startPhase);
    }

    /**
     * @brief Returns the calculated phase increment per sample.
     * @return Increment value.
     */
    [[nodiscard]] T getIncrement() const noexcept { return increment_; }

private:
    /**
     * @brief Pre-calculates the phase increment per sample.
     */
    void updateIncrement() noexcept
    {
        increment_ = static_cast<T>(static_cast<double>(frequency_) * invSampleRate_);
    }

    /**
     * @brief Fast fallback for phase wrapping with arbitrary values (e.g., extreme FM).
     * Replaces std::floor with integer casting.
     */
    static constexpr T wrapPhase(T p) noexcept
    {
        if (p >= T(0) && p < T(1)) return p; // Fast exit
        const T wrapped = p - static_cast<int>(p);
        return wrapped < T(0) ? wrapped + T(1) : wrapped;
    }

    double sampleRate_ = 48000.0;
    double invSampleRate_ = 1.0 / 48000.0;

    T frequency_ = T(0);
    T phase_ = T(0);
    T increment_ = T(0);
};

} // namespace dspark