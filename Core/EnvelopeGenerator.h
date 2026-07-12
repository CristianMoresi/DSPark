// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

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
 * Each stage runs an asymptotic RC-style one-pole toward a target that
 * overshoots the stage endpoint, so the stage completes in finite time with
 * the exact duration set by its parameter:
 *
 * - **Attack**: 0 -> 1 in attackMs (from a retrigger it starts at the
 *   current value and arrives earlier, click-free legato).
 * - **Decay**: 1 -> sustain in decayMs for any sustain level (the overshoot
 *   scales with the stage depth, matching the self-similar RC discharge).
 * - **Release**: calibrated for full scale -> silence in releaseMs. Like an
 *   analog RC release (fixed dB-per-second slope), releasing from a lower
 *   sustain reaches silence proportionally sooner.
 *
 * Curvature shapes each stage: high values are strongly exponential, 0.1 is
 * near-linear. Stage durations stay exact across the whole curvature range.
 *
 * State machine: Idle -> Attack -> Decay -> Sustain -> Release -> Idle.
 * The recursion state is double regardless of T (the framework rule for
 * recursive state): in float, very long stages stall short of their target
 * when the per-sample increment rounds to zero.
 *
 * Threading is owner-managed: call setters, triggers and processing from the
 * owning (audio) thread, the pattern of a synth voice. A default-constructed
 * envelope is functional at 48 kHz; prepare() adapts it to the real rate.
 *
 * @tparam T Output type (float or double). Constrained to IEEE 754 floats.
 */
template <FloatType T>
class ADSREnvelope
{
public:
    enum class State { Idle, Attack, Decay, Sustain, Release };

    ADSREnvelope() noexcept { recalculate(); }

    /**
     * @brief Prepares the envelope for the given sample rate.
     * @param sampleRate The operating sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        if (sampleRate <= 0.0) return;
        sampleRate_ = sampleRate;
        recalculate();
    }

    /**
     * @brief Prepares from AudioSpec (unified API).
     * @param spec Audio specification containing the sample rate.
     */
    void prepare(const AudioSpec& spec) noexcept { prepare(spec.sampleRate); }

    // -- Parameters (in milliseconds) ------------------------------------------

    /** @brief Sets attack time in milliseconds (0 -> 1). */
    void setAttack(T ms) noexcept  { attackMs_ = std::max(ms, T(0.01)); recalculate(); }

    /** @brief Sets decay time in milliseconds (1 -> sustain). */
    void setDecay(T ms) noexcept   { decayMs_ = std::max(ms, T(0.01)); recalculate(); }

    /** @brief Sets sustain level (0.0 to 1.0). */
    void setSustain(T level) noexcept { sustainLevel_ = std::clamp(level, T(0), T(1)); }

    /** @brief Sets release time in milliseconds (full scale -> silence). */
    void setRelease(T ms) noexcept { releaseMs_ = std::max(ms, T(0.01)); recalculate(); }

    /**
     * @brief Sets all ADSR parameters at once.
     * @param attackMs  Attack time in ms.
     * @param decayMs   Decay time in ms.
     * @param sustain   Sustain level (0-1).
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
     * @brief Sets the curvature of all exponential stages at once.
     * @param curve Curvature factor (default: 3.0, range: 0.1 to 10.0).
     */
    void setCurvature(T curve) noexcept
    {
        curve = std::clamp(curve, T(0.1), T(10));
        attackCurve_ = decayCurve_ = releaseCurve_ = curve;
        recalculate();
    }

    /**
     * @brief Sets independent curvatures per stage.
     *
     * Classic analog envelopes shape each stage differently (near-linear
     * attack, strongly exponential release). Higher values = more curved.
     *
     * @param attackCurve  Attack curvature (0.1 to 10.0; low = near-linear).
     * @param decayCurve   Decay curvature.
     * @param releaseCurve Release curvature.
     */
    void setCurvature(T attackCurve, T decayCurve, T releaseCurve) noexcept
    {
        attackCurve_  = std::clamp(attackCurve,  T(0.1), T(10));
        decayCurve_   = std::clamp(decayCurve,   T(0.1), T(10));
        releaseCurve_ = std::clamp(releaseCurve, T(0.1), T(10));
        recalculate();
    }

    // -- Trigger ---------------------------------------------------------------

    /** @brief Triggers the attack phase (note on). Legato: continues from the current value. */
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
        currentValue_ = 0.0;
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
                currentValue_ += attackRate_ * (1.0 + kAttack_ - currentValue_);
                if (currentValue_ >= 1.0)
                {
                    currentValue_ = 1.0;
                    state_ = State::Decay;
                }
                break;

            case State::Decay:
            {
                const double s = static_cast<double>(sustainLevel_);
                const double prev = currentValue_;
                currentValue_ += decayRate_ * (s - (1.0 - s) * kDecay_ - currentValue_);
                if (currentValue_ <= s)
                {
                    // Normal completion (crossed from above): land exactly on s.
                    // Sustain raised above the current value mid-decay: keep the
                    // value and let the Sustain chase close the gap click-free.
                    if (prev >= s)
                        currentValue_ = s;
                    state_ = State::Sustain;
                }
                break;
            }

            case State::Sustain:
                // Chases sustain changes at the decay rate (dynamic modulation).
                currentValue_ += decayRate_ * (static_cast<double>(sustainLevel_) - currentValue_);
                break;

            case State::Release:
                currentValue_ += releaseRate_ * (-kRelease_ - currentValue_);
                if (currentValue_ <= kSilence)
                {
                    currentValue_ = 0.0;
                    state_ = State::Idle;
                }
                break;
        }

        return static_cast<T>(currentValue_);
    }

    /**
     * @brief Fills a buffer with envelope values.
     *
     * Runs each stage as a tight loop with its constants hoisted (no state
     * dispatch per sample). Sample-exact match with getNextValue().
     *
     * @param output Buffer to fill with envelope values.
     * @param numSamples Number of samples to generate.
     */
    void processBlock(T* output, int numSamples) noexcept
    {
        // Each stage runs as a tight loop on locals (stage constants hoisted,
        // recursion state in a register): no state dispatch or member traffic
        // per sample. Arithmetic is sample-exact with getNextValue().
        int i = 0;
        while (i < numSamples)
        {
            switch (state_)
            {
                case State::Idle:
                    while (i < numSamples)
                        output[i++] = T(0);
                    break;

                case State::Attack:
                {
                    const double target = 1.0 + kAttack_;
                    const double r = attackRate_;
                    double v = currentValue_;
                    while (i < numSamples)
                    {
                        v += r * (target - v);
                        if (v >= 1.0)
                        {
                            v = 1.0;
                            state_ = State::Decay;
                            output[i++] = T(1);
                            break;
                        }
                        output[i++] = static_cast<T>(v);
                    }
                    currentValue_ = v;
                    break;
                }

                case State::Decay:
                {
                    const double s = static_cast<double>(sustainLevel_);
                    const double target = s - (1.0 - s) * kDecay_;
                    const double r = decayRate_;
                    double v = currentValue_;
                    while (i < numSamples)
                    {
                        const double prev = v;
                        v += r * (target - v);
                        if (v <= s)
                        {
                            if (prev >= s)
                                v = s;
                            state_ = State::Sustain;
                            output[i++] = static_cast<T>(v);
                            break;
                        }
                        output[i++] = static_cast<T>(v);
                    }
                    currentValue_ = v;
                    break;
                }

                case State::Sustain:
                {
                    const double s = static_cast<double>(sustainLevel_);
                    const double r = decayRate_;
                    double v = currentValue_;
                    while (i < numSamples)
                    {
                        v += r * (s - v);
                        output[i++] = static_cast<T>(v);
                    }
                    currentValue_ = v;
                    break;
                }

                case State::Release:
                {
                    const double target = -kRelease_;
                    const double r = releaseRate_;
                    double v = currentValue_;
                    while (i < numSamples)
                    {
                        v += r * (target - v);
                        if (v <= kSilence)
                        {
                            v = 0.0;
                            state_ = State::Idle;
                            output[i++] = T(0);
                            break;
                        }
                        output[i++] = static_cast<T>(v);
                    }
                    currentValue_ = v;
                    break;
                }
            }
        }
    }

    // -- Getters ---------------------------------------------------------------

    /** @brief Returns the current envelope value. */
    [[nodiscard]] T getCurrentValue() const noexcept { return static_cast<T>(currentValue_); }

    /** @brief Returns the current state. */
    [[nodiscard]] State getState() const noexcept { return state_; }

    /** @brief Returns true if the envelope is actively producing output. */
    [[nodiscard]] bool isActive() const noexcept { return state_ != State::Idle; }

private:
    // Release ends when the value falls below this floor (about -80 dB);
    // the strictly negative release target guarantees it is crossed.
    static constexpr double kSilence = 1e-4;

    void recalculate() noexcept
    {
        // Asymptotic one-pole stage design. With ov = exp(-curve), running the
        // recursion v += rate * (target - v) for exactly `samples` steps
        // multiplies the distance to the target by ov. Overshooting the stage
        // endpoint by k = ov / (1 - ov) OF THE STAGE DEPTH makes the endpoint
        // land exactly on the last step, for any curvature:
        //
        //   endpoint = target + depth_to_target * ov  with  target = end + k*depth
        //
        // (A first-order shortcut like target = end + ov breaks the timing
        // contract: at curve 0.1 the stages run ~7x their parameter, and a
        // fixed absolute overshoot makes decay/release durations drift with
        // the sustain level.)
        auto stage = [this](T timeMs, T curve, double& rate, double& k) noexcept {
            const double c = static_cast<double>(curve);
            const double ov = std::exp(-c);
            k = ov / (1.0 - ov);
            const double samples = std::max(sampleRate_ * 0.001 * static_cast<double>(timeMs), 1.0);
            rate = 1.0 - std::exp(-c / samples);
        };

        stage(attackMs_,  attackCurve_,  attackRate_,  kAttack_);
        stage(decayMs_,   decayCurve_,   decayRate_,   kDecay_);
        stage(releaseMs_, releaseCurve_, releaseRate_, kRelease_);
    }

    double sampleRate_ = 48000.0;

    T attackMs_     = T(10);
    T decayMs_      = T(100);
    T sustainLevel_ = T(0.7);
    T releaseMs_    = T(200);
    T attackCurve_  = T(3.0);
    T decayCurve_   = T(3.0);
    T releaseCurve_ = T(3.0);

    // Stage coefficients and overshoot fractions (see recalculate()).
    double attackRate_  = 0.0;
    double decayRate_   = 0.0;
    double releaseRate_ = 0.0;
    double kAttack_     = 0.0;
    double kDecay_      = 0.0;
    double kRelease_    = 0.0;

    double currentValue_ = 0.0;
    State state_ = State::Idle;
};

} // namespace dspark
