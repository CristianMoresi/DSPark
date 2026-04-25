// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file EnvelopeGenerator.h
 * @brief ADSR and multi-segment envelope generators for synthesis and dynamics.
 *
 * Provides per-sample envelope generation with configurable attack, decay,
 * sustain, and release stages. The envelope output is a gain value in [0, 1]
 * that can be applied to audio signals, filter cutoffs, or any modulation target.
 *
 * Two classes:
 *
 * - **ADSREnvelope\<T\>**: Classic 4-stage ADSR with exponential curves and
 *   configurable curvature. Suitable for synthesisers and dynamic processors.
 *
 * - **EnvelopeFollower\<T\>**: Simple attack/release envelope for tracking
 *   signal dynamics (e.g., compressor side-chain). This complements
 *   LevelFollower which measures levels — EnvelopeFollower generates a
 *   smooth control signal from an external trigger.
 *
 * Dependencies: DspMath.h only.
 *
 * @code
 *   dspark::ADSREnvelope<float> env;
 *   env.prepare(48000.0);
 *   env.setAttack(10.0f);   // 10 ms
 *   env.setDecay(100.0f);   // 100 ms
 *   env.setSustain(0.7f);   // 70%
 *   env.setRelease(200.0f); // 200 ms
 *
 *   env.noteOn();
 *
 *   for (int i = 0; i < numSamples; ++i)
 *       output[i] = oscillator.getNextSample() * env.getNextValue();
 *
 *   env.noteOff();
 * @endcode
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
 * Each stage uses an exponential curve for natural-sounding dynamics.
 * The curvature can be adjusted: higher values produce faster initial
 * response (more "snappy"), lower values produce more linear curves.
 *
 * State machine: Idle → Attack → Decay → Sustain → Release → Idle.
 *
 * @tparam T Output type (float or double).
 */
template <typename T>
class ADSREnvelope
{
public:
    enum class State { Idle, Attack, Decay, Sustain, Release };

    /** @brief Prepares the envelope for the given sample rate. */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        recalculate();
    }

    /** @brief Prepares from AudioSpec (unified API). */
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
     *
     * Higher values = more exponential (snappy attack, slow tail).
     * Lower values = more linear.
     *
     * **Calibration note:** the envelope uses an "asymptotic overshoot"
     * topology (`current += rate · (target ± overshoot − current)`). With
     * the default `curvature = 3`, the stage reaches ~95 % of the target
     * in roughly **half** the programmed time. If you need "a 100 ms
     * attack hitting ~95 % at 100 ms" (ADSR convention in many plugins),
     * use `curvature ≈ 1.5` and tune to taste, or read the asymptotic
     * behaviour as a feature. Kept at 3 by default for musical snap.
     *
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
        // Start from current value for re-triggering without clicks
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
                currentValue_ = T(0);
                break;

            case State::Attack:
                currentValue_ += attackRate_ * (T(1) + attackCoeff_ - currentValue_);
                if (currentValue_ >= T(1))
                {
                    currentValue_ = T(1);
                    state_ = State::Decay;
                }
                break;

            case State::Decay:
                currentValue_ += decayRate_ * (sustainLevel_ - decayCoeff_ - currentValue_);
                if (currentValue_ <= sustainLevel_)
                {
                    currentValue_ = sustainLevel_;
                    state_ = State::Sustain;
                }
                break;

            case State::Sustain:
                currentValue_ = sustainLevel_;
                break;

            case State::Release:
                currentValue_ += releaseRate_ * (T(0) - releaseCoeff_ - currentValue_);
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
     * @brief Fills a buffer with envelope values.
     * @param output Buffer to fill with envelope values.
     * @param numSamples Number of samples to generate.
     */
    void processBlock(T* output, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            output[i] = getNextValue();
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
        if (sampleRate_ <= 0.0) return;

        // One-pole coefficient: rate = 1 - exp(-curvature / (time_ms * sampleRate / 1000))
        auto calcRate = [this](T timeMs) -> T {
            T samples = static_cast<T>(sampleRate_) * timeMs / T(1000);
            return T(1) - std::exp(-curvature_ / std::max(samples, T(1)));
        };

        attackRate_  = calcRate(attackMs_);
        decayRate_   = calcRate(decayMs_);
        releaseRate_ = calcRate(releaseMs_);

        // Overshoot coefficients for natural exponential curves
        attackCoeff_  = T(0.3);
        decayCoeff_   = T(0.0001);
        releaseCoeff_ = T(0.0001);
    }

    double sampleRate_ = 48000.0;

    T attackMs_     = T(10);
    T decayMs_      = T(100);
    T sustainLevel_ = T(0.7);
    T releaseMs_    = T(200);
    T curvature_    = T(3.0);

    T attackRate_   = T(0);
    T decayRate_    = T(0);
    T releaseRate_  = T(0);
    T attackCoeff_  = T(0.3);
    T decayCoeff_   = T(0.0001);
    T releaseCoeff_ = T(0.0001);

    T currentValue_ = T(0);
    State state_ = State::Idle;
};

} // namespace dspark
