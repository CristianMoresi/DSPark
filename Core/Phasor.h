// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Phasor.h
 * @brief Phase accumulator producing a [0, 1) ramp for oscillators and LFOs.
 *
 * Building block for table oscillators, LFOs and modulation carriers
 * (WavetableOscillator, Vibrato, Tremolo, RingModulator). The accumulator
 * and increment live in double internally (framework rule: recursive state
 * in double) so ultra-slow LFO rates neither stall nor drift - a float
 * accumulator freezes completely below ~3e-3 Hz at 96 kHz (the per-sample
 * increment rounds to zero against the phase's ulp) and runs slow LFOs
 * measurably fast (+0.125% at 0.1 Hz). The public API stays in T.
 *
 * Threading: owner-managed. Not internally thread-safe: call setters and
 * advance() from the owning (audio) thread, or synchronise externally.
 *
 * Dependencies: DspMath.h (FloatType), AudioSpec.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"

#include <cmath>
#include <limits>

namespace dspark {

/**
 * @class Phasor
 * @brief Phase accumulator generating a [0, 1) ramp for oscillators and modulation.
 *
 * Scalar recursive accumulator with a fast in-range wrap (one compare in the
 * common case) that also handles arbitrarily large excursions (extreme FM,
 * frequencies far above the sample rate) without unbounded loops. Supports
 * fractional phase offsets on hard sync for PolyBLEP/MinBLEP-style
 * anti-aliased oscillators.
 *
 * @note This class is not internally thread-safe. Parameter updates (e.g.,
 * setFrequency) should be synchronized or smoothed by the caller prior to
 * block processing.
 *
 * @tparam T Sample type (must satisfy FloatType concept).
 */
template <FloatType T>
class Phasor
{
public:
    /**
     * @brief Prepares the phasor for a given sample rate.
     * @param sampleRate Sample rate in Hz. Must be > 0 (invalid values,
     * including NaN, are ignored).
     */
    void prepare(double sampleRate) noexcept
    {
        if (!(sampleRate > 0.0)) return; // Reject non-positive and NaN

        sampleRate_ = sampleRate;
        invSampleRate_ = 1.0 / sampleRate;
        updateIncrement();
    }

    /**
     * @brief Prepares from AudioSpec (unified API).
     * @param spec Framework audio specification struct.
     */
    void prepare(const AudioSpec& spec) noexcept { prepare(spec.sampleRate); }

    /**
     * @brief Sets the baseline oscillation frequency.
     * @param frequencyHz Frequency in Hz. Negative values produce a
     * descending ramp. NaN is ignored (keeps the previous frequency).
     */
    void setFrequency(T frequencyHz) noexcept
    {
        if (frequencyHz != frequencyHz) return; // NaN would poison the accumulator
        frequency_ = frequencyHz;
        updateIncrement();
    }

    /**
     * @brief Returns the baseline frequency.
     * @return Current frequency in Hz.
     */
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

    /**
     * @brief Returns the phase for the current sample, then advances the
     * accumulator (post-increment semantics).
     * @return Current phase value in [0, 1).
     */
    [[nodiscard]] T advance() noexcept
    {
        const T current = currentPhase();
        // wrapUnit has a fast in-range exit, so the common case (|increment| < 1)
        // costs one compare - but unlike an unbounded while-loop it can never
        // hang on a pathological frequency (>> sample rate).
        phase_ = wrapUnit(phase_ + increment_);
        return current;
    }

    /**
     * @brief Advances the phase with per-sample frequency modulation.
     *
     * The instantaneous increment is computed in double, and the wrap safely
     * handles arbitrarily large FM indices without unbounded loops. A NaN
     * excursion cannot poison the accumulator permanently (the wrap resets
     * it to 0), but the caller should keep fmHz finite.
     *
     * @param fmHz Instantaneous frequency deviation in Hz.
     * @return Current phase value in [0, 1).
     */
    [[nodiscard]] T advanceWithFM(T fmHz) noexcept
    {
        const T current = currentPhase();

        const double modulatedIncrement =
            (static_cast<double>(frequency_) + static_cast<double>(fmHz)) * invSampleRate_;

        phase_ = wrapUnit(phase_ + modulatedIncrement);

        return current;
    }

    /**
     * @brief Returns the current phase without advancing the accumulator.
     * @return Phase value in [0, 1).
     */
    [[nodiscard]] T getPhase() const noexcept { return currentPhase(); }

    /**
     * @brief Forcibly overwrites the current phase.
     * @param newPhase Any phase value (safely wrapped to [0, 1); NaN resets to 0).
     */
    void setPhase(T newPhase) noexcept
    {
        phase_ = wrapUnit(static_cast<double>(newPhase));
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
        phase_ = wrapUnit(static_cast<double>(fractionalOffset));
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
        if (phase_ >= static_cast<double>(threshold))
            phase_ = 0.0;
    }

    /**
     * @brief Resets the phasor to a specific initial state (default 0).
     * @param startPhase Initial phase value (wrapped to [0, 1)).
     */
    void reset(T startPhase = T(0)) noexcept
    {
        phase_ = wrapUnit(static_cast<double>(startPhase));
    }

    /**
     * @brief Returns the calculated phase increment per sample.
     * @return Increment value.
     */
    [[nodiscard]] T getIncrement() const noexcept { return static_cast<T>(increment_); }

private:
    /**
     * @brief Pre-calculates the phase increment per sample.
     */
    void updateIncrement() noexcept
    {
        increment_ = static_cast<double>(frequency_) * invSampleRate_;
    }

    /**
     * @brief Rounds the double accumulator to T without breaking the [0, 1)
     * contract: a phase just below 1.0 can round UP to exactly 1.0 in float,
     * and a table oscillator indexing phase*N would read one past the end.
     */
    [[nodiscard]] T currentPhase() const noexcept
    {
        const T v = static_cast<T>(phase_);
        constexpr T belowOne = T(1) - std::numeric_limits<T>::epsilon() / T(2);
        return (v >= T(1)) ? belowOne : v;
    }

    /**
     * @brief Fast fallback for phase wrapping with arbitrary values (e.g., extreme FM).
     * Replaces std::floor with truncation. Named wrapUnit to avoid confusion
     * with dspark::wrapPhase, which wraps to [0, 2*pi) instead of [0, 1).
     */
    static double wrapUnit(double p) noexcept
    {
        if (p >= 0.0 && p < 1.0) return p; // Fast exit
        // Slow path (rare): sanitise NaN here, where it costs nothing in the
        // common case - otherwise a single NaN input would stick in the
        // recursive accumulator until the next reset.
        if (p != p) return 0.0;
        // std::trunc instead of a cast to int: casting is UB once |p| exceeds
        // INT_MAX, which extreme FM excursions can reach.
        double wrapped = p - std::trunc(p);
        if (wrapped < 0.0) wrapped += 1.0;
        // Guard the contract's half-open range: for tiny negative inputs the
        // addition above rounds to exactly 1.0.
        if (wrapped >= 1.0) wrapped -= 1.0;
        return wrapped;
    }

    double sampleRate_ = 48000.0;
    double invSampleRate_ = 1.0 / 48000.0;

    T frequency_ = T(0);
    double phase_ = 0.0;
    double increment_ = 0.0;
};

} // namespace dspark
