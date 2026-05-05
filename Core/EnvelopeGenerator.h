// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file EnvelopeGenerator.h
 * @brief ADSR envelope generator for synthesis and dynamics.
 *
 * Provides per-sample envelope generation with configurable attack, decay,
 * sustain, and release stages. The envelope output is a gain value in [0, 1]
 * that can be applied to audio signals, filter cutoffs, or any modulation target.
 *
 * Dependencies: DspMath.h, AudioSpec.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"

#include <algorithm>
#include <cmath>

namespace dspark {

/**
 * @class ADSREnvelope
 * @brief Classic ADSR envelope generator with exponential curves.
 *
 * Uses an asymptotic RC-style algorithm for natural-sounding dynamics.
 * Optimized for real-time block processing and cache coherence.
 *
 * State machine: Idle → Attack → Decay → Sustain → Release → Idle.
 *
 * @tparam T Output type (float or double). Constraints to IEEE 754 floats.
 */
template <FloatType T>
class ADSREnvelope
{
public:
    enum class State { Idle, Attack, Decay, Sustain, Release };

    /** * @brief Prepares the envelope for the given sample rate. 
     * @param sampleRate The operating sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        if (sampleRate <= 0.0) return;
        sampleRate_ = sampleRate;
        sampleRateMs_ = static_cast<T>(sampleRate) * T(0.001); // Pre-calculate for ms conversion
        recalculate();
    }

    /** * @brief Prepares from AudioSpec (unified API). 
     * @param spec Audio specification containing the sample rate.
     */
    void prepare(const AudioSpec& spec) { prepare(spec.sampleRate); }

    // -- Parameters (in milliseconds) ------------------------------------------

    /** @brief Sets attack time in milliseconds. */
    void setAttack(T ms) noexcept  { attackMs_ = std::max(ms, T(0.01)); recalculate(); }

    /** @brief Sets decay time in milliseconds. */
    void setDecay(T ms) noexcept   { decayMs_ = std::max(ms, T(0.01)); recalculate(); }

    /** @brief Sets sustain level (0.0 to 1.0). */
    void setSustain(T level) noexcept { sustainLevel_ = std::clamp(level, T(0), T(1)); }

    /** @brief Sets release time in milliseconds. */
    void setRelease(T ms) noexcept { releaseMs_ = std::max(ms, T(0.01)); recalculate(); }

    /**
     * @brief Sets all ADSR parameters at once.
     * @param attackMs  Attack time in ms.
     * @param decayMs   Decay time in ms.
     * @param sustain   Sustain level (0–1).
     * @param releaseMs Release time in ms.
     */
    void setParameters(T attackMs, T decayMs, T sustain, T releaseMs) noexcept
    {
        attackMs_     = std::max(attackMs, T(0.01));
        decayMs_      = std::max(decayMs, T(0.01));
        sustainLevel_ = std::clamp(sustain, T(0), T(1));
        releaseMs_    = std::max(releaseMs, T(0.01));
        recalculate();
    }

    /**
     * @brief Sets the curvature of the exponential stages.
     * @param curve Curvature factor (default: 3.0, range: 0.1 to 10.0).
     */
    void setCurvature(T curve) noexcept
    {
        curvature_ = std::clamp(curve, T(0.1), T(10));
        recalculate();
    }

    // -- Trigger ---------------------------------------------------------------

    /** @brief Triggers the attack phase (note on). */
    void noteOn() noexcept
    {
        state_ = State::Attack;
    }

    /** @brief Triggers the release phase (note off). */
    void noteOff() noexcept
    {
        if (state_ != State::Idle)
            state_ = State::Release;
    }

    /** @brief Resets the envelope to idle with zero output. */
    void reset() noexcept
    {
        state_ = State::Idle;
        currentValue_ = T(0);
    }

    // -- Processing ------------------------------------------------------------

    /**
     * @brief Returns the next envelope value and advances the state machine.
     * @return Envelope value in [0, 1].
     */
    [[nodiscard]] T getNextValue() noexcept
    {
        switch (state_)
        {
            case State::Idle:
                return T(0);

            case State::Attack:
                currentValue_ += attackRate_ * (T(1) + targetOvershoot_ - currentValue_);
                if (currentValue_ >= T(1))
                {
                    currentValue_ = T(1);
                    state_ = State::Decay;
                }
                break;

            case State::Decay:
                currentValue_ += decayRate_ * (sustainLevel_ - targetOvershoot_ - currentValue_);
                // Modulating sustain upwards safety check: if we drop below OR cross the new threshold
                if (currentValue_ <= sustainLevel_)
                {
                    currentValue_ = sustainLevel_;
                    state_ = State::Sustain;
                }
                break;

            case State::Sustain:
                // Allows dynamic sustain modulation during hold phase
                currentValue_ += decayRate_ * (sustainLevel_ - currentValue_);
                break;

            case State::Release:
                currentValue_ += releaseRate_ * (T(0) - targetOvershoot_ - currentValue_);
                if (currentValue_ <= T(0.0001))
                {
                    currentValue_ = T(0);
                    state_ = State::Idle;
                }
                break;
        }

        return currentValue_;
    }

    /**
     * @brief Fills a buffer with envelope values. Optimized for DSP loops.
     * @param output Buffer to fill with envelope values.
     * @param numSamples Number of samples to generate.
     */
    void processBlock(T* output, int numSamples) noexcept
    {
        // Unrolling the block processing prevents the switch statement 
        // from destroying CPU branch prediction on every single sample.
        int i = 0;
        while (i < numSamples)
        {
            State currentState = state_;
            
            // Process chunks while the state remains constant
            while (i < numSamples && state_ == currentState)
            {
                output[i++] = getNextValue();
            }
        }
    }

    // -- Getters ---------------------------------------------------------------

    /** @brief Returns the current envelope value. */
    [[nodiscard]] T getCurrentValue() const noexcept { return currentValue_; }

    /** @brief Returns the current state. */
    [[nodiscard]] State getState() const noexcept { return state_; }

    /** @brief Returns true if the envelope is actively producing output. */
    [[nodiscard]] bool isActive() const noexcept { return state_ != State::Idle; }

private:
    void recalculate() noexcept
    {
        // Standardize overshoot so timings are mathematically consistent across all stages
        targetOvershoot_ = std::exp(-curvature_);

        auto calcRate = [this](T timeMs) -> T {
            T samples = std::max(sampleRateMs_ * timeMs, T(1));
            return T(1) - std::exp(-curvature_ / samples);
        };

        attackRate_  = calcRate(attackMs_);
        decayRate_   = calcRate(decayMs_);
        releaseRate_ = calcRate(releaseMs_);
    }

    double sampleRate_ = 48000.0;
    T sampleRateMs_    = T(48.0); // Optimized multiplier

    T attackMs_     = T(10);
    T decayMs_      = T(100);
    T sustainLevel_ = T(0.7);
    T releaseMs_    = T(200);
    T curvature_    = T(3.0);

    T attackRate_   = T(0);
    T decayRate_    = T(0);
    T releaseRate_  = T(0);
    T targetOvershoot_ = T(0.05); // Derived from curvature

    T currentValue_ = T(0);
    State state_ = State::Idle;
};

} // namespace dspark