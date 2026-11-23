// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TriggeredEnvelope.h
 * @brief Envelope with delay and hold stages, curves that bend either way,
 *        and a trigger from notes or from the audio itself.
 *
 * Stages: Idle -> Delay -> Attack -> Hold -> Decay -> Sustain -> Release
 * -> Idle. Every timed stage takes exactly the samples its time asks for (a
 * phase from 0 to 1 at a fixed step), and each moving stage follows a curve
 * that bends from slow start (-1) through a straight line (0) to fast start
 * (+1):
 *
 *     f(x) = (1 - e^(-k x)) / (1 - e^(-k)),   k = 6 * curve
 *
 * The output never steps when a control moves:
 * - a stage time changes the speed of the phase, not the phase;
 * - a curve that changes in the middle of a stage takes the phase at which
 *   the new curve gives the value the output is at;
 * - a sustain that changes during the decay keeps the decay's phase and
 *   bends what is left of it to the new level; during the sustain the
 *   output moves to the new level at the decay's pace (a full-scale move
 *   in the decay time);
 * - the attack starts from the value the output is at (a retrigger arrives
 *   earlier), and so does the release.
 *
 * The trigger is noteOn()/noteOff(), or the audio of a key signal: the
 * envelope starts when the key's peak level rises above a threshold, and
 * lets go when the level has stayed below the threshold minus a hysteresis
 * (6 dB by default) for a wait (50 ms by default), so a note's decay or the
 * gaps of a tremolo do not chatter it.
 *
 * The output is 0..1, or, in the bipolar mode, centred on the sustain: 0
 * while it sustains, up to +1 at the top of the attack, down to -1 at rest.
 *
 * Threading: owner-managed. Setters, triggers and processing belong to the
 * owning (audio) thread. No allocation anywhere.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dspark {

/**
 * @class TriggeredEnvelope
 * @brief DAHDSR envelope with bending curves, triggered by notes or by audio.
 *
 * @code
 *   dspark::TriggeredEnvelope<float> env;
 *   env.prepare(48000.0);
 *   env.setTrigger(dspark::TriggeredEnvelope<float>::Trigger::Audio);
 *   env.setThresholdDb(-30.0f);
 *   env.setTimes(0.0f, 5.0f, 20.0f, 150.0f, 300.0f);   // delay, attack, hold, decay, release
 *   env.setSustain(0.4f);
 *   env.processBlock(keySignal, modulation, numSamples);
 * @endcode
 *
 * @tparam T Sample type (float or double). The phases are double.
 */
template <FloatType T>
class TriggeredEnvelope
{
public:
    enum class Stage { Idle, Delay, Attack, Hold, Decay, Sustain, Release };
    enum class Trigger { Notes, Audio };

    TriggeredEnvelope() noexcept { recalculate(); }

    /** @brief Prepares for a sample rate (invalid rates are ignored). */
    void prepare(double sampleRate) noexcept
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) return;
        sampleRate_ = sampleRate;
        recalculate();
    }

    // -- Times, levels and curves (owning thread) ------------------------------

    /** @brief Delay before the attack, in ms (>= 0; non-finite ignored). */
    void setDelay(T ms) noexcept { setTime(delayMs_, ms); }
    /** @brief Attack, 0 to 1, in ms. */
    void setAttack(T ms) noexcept { setTime(attackMs_, ms); }
    /** @brief Hold at the top, in ms. */
    void setHold(T ms) noexcept { setTime(holdMs_, ms); }
    /** @brief Decay, 1 to the sustain, in ms. */
    void setDecay(T ms) noexcept { setTime(decayMs_, ms); }
    /** @brief Release, full scale to 0, in ms (from a lower value it arrives earlier). */
    void setRelease(T ms) noexcept { setTime(releaseMs_, ms); }

    /** @brief All five times at once, in ms. */
    void setTimes(T delayMs, T attackMs, T holdMs, T decayMs, T releaseMs) noexcept
    {
        setDelay(delayMs);
        setAttack(attackMs);
        setHold(holdMs);
        setDecay(decayMs);
        setRelease(releaseMs);
    }

    /**
     * @brief The sustain level, 0..1 (non-finite ignored). During the decay
     *        the rest of the decay bends to it; during the sustain the
     *        output moves to it at the decay's pace.
     */
    void setSustain(T level) noexcept
    {
        if (!std::isfinite(level)) return;
        const double s = std::clamp(static_cast<double>(level), 0.0, 1.0);
        if (s == sustain_) return;
        const double now = value_;
        sustain_ = s;
        if (stage_ == Stage::Decay)
        {
            // Keep the phase: what is left of the decay goes to the new level.
            const double f = shape(x_, decayCurve_);
            if (f < 1.0 - 1e-12) from_ = (now - sustain_ * f) / (1.0 - f);
            to_ = sustain_;
        }
        else if (stage_ == Stage::Sustain)
        {
            // Move there at the decay's pace.
            from_ = now;
            to_ = sustain_;
            x_ = 0.0;
            decayScale_ = std::abs(sustain_ - now);
            stage_ = Stage::Decay;
            curveNow_ = decayCurve_;
            aim(stageSamples());
        }
    }

    /**
     * @brief The curves of the attack, the decay and the release, -1 (slow
     *        start) to 1 (fast start), 0 straight (non-finite ignored). A
     *        curve that changes in the middle of its stage keeps the output
     *        where it is.
     */
    void setCurves(T attack, T decay, T release) noexcept
    {
        if (!std::isfinite(attack) || !std::isfinite(decay) || !std::isfinite(release)) return;
        const double a = std::clamp(static_cast<double>(attack), -1.0, 1.0);
        const double d = std::clamp(static_cast<double>(decay), -1.0, 1.0);
        const double r = std::clamp(static_cast<double>(release), -1.0, 1.0);
        const double* mine = curveOf(stage_, a, d, r);
        if (mine != nullptr && *mine != curveNow_)
        {
            // The phase at which the new curve gives the output's place in the stage.
            const double place = (to_ != from_) ? std::clamp((value_ - from_) / (to_ - from_), 0.0, 1.0) : 1.0;
            x_ = inverse(place, *mine);
            curveNow_ = *mine;
            aim(stageSamples());
        }
        attackCurve_ = a;
        decayCurve_ = d;
        releaseCurve_ = r;
    }

    /** @brief Output centred on the sustain (see the file notes). */
    void setBipolar(bool on) noexcept { bipolar_ = on; }
    [[nodiscard]] bool isBipolar() const noexcept { return bipolar_; }

    // -- Trigger (owning thread) ------------------------------------------------

    /** @brief Notes (noteOn()/noteOff()) or the audio of the key signal. */
    void setTrigger(Trigger trigger) noexcept { trigger_ = trigger; }
    [[nodiscard]] Trigger getTrigger() const noexcept { return trigger_; }

    /** @brief Level of the key above which the audio trigger starts the envelope, in dB (non-finite ignored). */
    void setThresholdDb(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        thresholdDb_ = std::clamp(static_cast<double>(db), -120.0, 24.0);
        recalculateTrigger();
    }

    /** @brief How far below the threshold the key must fall to let go, in dB (0..40; 6 by default). */
    void setHysteresisDb(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        hysteresisDb_ = std::clamp(static_cast<double>(db), 0.0, 40.0);
        recalculateTrigger();
    }

    /** @brief How long the key must stay below that before the release, in ms (>= 0; 50 by default). */
    void setReleaseWaitMs(T ms) noexcept
    {
        if (!(ms >= T(0)) || !std::isfinite(ms)) return;
        waitMs_ = static_cast<double>(ms);
        recalculateTrigger();
    }

    /** @brief Starts the envelope (from where the output is). */
    void noteOn() noexcept
    {
        held_ = value_;
        x_ = 0.0;
        stage_ = Stage::Delay;
        if (delaySteps_ <= 0.0) enterAttack();
    }

    /** @brief Lets the envelope go to its release (from where the output is). */
    void noteOff() noexcept
    {
        if (stage_ == Stage::Idle || stage_ == Stage::Release) return;
        enterRelease();
    }

    /** @brief Back to rest at 0, the trigger disarmed. */
    void reset() noexcept
    {
        stage_ = Stage::Idle;
        value_ = 0.0;
        x_ = 0.0;
        gate_ = false;
        level_ = 0.0;
        below_ = 0;
    }

    [[nodiscard]] Stage getStage() const noexcept { return stage_; }
    /** @brief The envelope now, 0..1 (before the bipolar mapping). */
    [[nodiscard]] T getValue() const noexcept { return static_cast<T>(value_); }
    [[nodiscard]] bool isActive() const noexcept { return stage_ != Stage::Idle; }

    // -- Processing (owning thread) ----------------------------------------------

    /**
     * @brief One sample: the key (read by the audio trigger), then the output.
     * @param key Key signal sample (ignored with the note trigger).
     */
    T process(T key = T(0)) noexcept
    {
        if (trigger_ == Trigger::Audio) listen(static_cast<double>(key));
        advance();
        return output();
    }

    /**
     * @brief A block: `key` drives the audio trigger (may be null with the
     *        note trigger), the envelope goes to `out`.
     */
    void processBlock(const T* key, T* out, int numSamples) noexcept
    {
        if (out == nullptr || numSamples <= 0) return;
        for (int i = 0; i < numSamples; ++i)
            out[i] = process(key != nullptr ? key[i] : T(0));
    }

private:
    void setTime(double& slot, T ms) noexcept
    {
        if (!(ms >= T(0)) || !std::isfinite(ms)) return;
        slot = static_cast<double>(ms);
        recalculate();
    }

    [[nodiscard]] double samplesFor(double ms) const noexcept { return std::round(ms * 0.001 * sampleRate_); }

    /// Samples a whole moving stage takes now.
    [[nodiscard]] double stageSamples() const noexcept
    {
        switch (stage_)
        {
            case Stage::Attack:  return attackN_;
            case Stage::Decay:   return samplesFor(decayMs_ * decayScale_);
            case Stage::Release: return releaseN_;
            default:             return 0.0;
        }
    }

    /// Counts down what is left of a moving stage of `samples` samples in
    /// whole samples, so the stage lands on its last sample exactly.
    void aim(double samples) noexcept
    {
        left_ = std::max<int64_t>(1, static_cast<int64_t>(std::llround((1.0 - x_) * samples)));
        moveStep_ = (1.0 - x_) / static_cast<double>(left_);
    }

    void recalculate() noexcept
    {
        delaySteps_ = std::round(delayMs_ * 0.001 * sampleRate_);
        holdSteps_ = std::round(holdMs_ * 0.001 * sampleRate_);
        attackN_ = samplesFor(attackMs_);
        releaseN_ = samplesFor(releaseMs_);
        if (stage_ == Stage::Attack || stage_ == Stage::Decay || stage_ == Stage::Release) aim(stageSamples());
        recalculateTrigger();
    }

    void recalculateTrigger() noexcept
    {
        threshold_ = std::pow(10.0, thresholdDb_ / 20.0);
        lowLevel_ = std::pow(10.0, (thresholdDb_ - hysteresisDb_) / 20.0);
        waitSamples_ = static_cast<int64_t>(std::llround(waitMs_ * 0.001 * sampleRate_));
        levelFall_ = std::exp(-1.0 / (kDetectorReleaseSeconds * sampleRate_));
    }

    /// The curve, 0..1 onto 0..1.
    [[nodiscard]] static double shape(double x, double curve) noexcept
    {
        if (curve == 0.0) return x;
        const double k = 6.0 * curve;
        return std::expm1(-k * x) / std::expm1(-k);
    }

    /// The phase at which the curve gives `y`.
    [[nodiscard]] static double inverse(double y, double curve) noexcept
    {
        if (curve == 0.0) return y;
        const double k = 6.0 * curve;
        return std::clamp(-std::log1p(y * std::expm1(-k)) / k, 0.0, 1.0);
    }

    [[nodiscard]] const double* curveOf(Stage stage, const double& a, const double& d, const double& r) const noexcept
    {
        switch (stage)
        {
            case Stage::Attack:  return &a;
            case Stage::Decay:   return &d;
            case Stage::Release: return &r;
            default:             return nullptr;
        }
    }

    void enterAttack() noexcept
    {
        stage_ = Stage::Attack;
        from_ = 0.0;
        to_ = 1.0;
        curveNow_ = attackCurve_;
        x_ = inverse(std::clamp(held_, 0.0, 1.0), curveNow_);   // from where the output is
        if (x_ >= 1.0) { enterHold(); return; }
        aim(attackN_);
    }

    void enterHold() noexcept
    {
        value_ = 1.0;
        stage_ = Stage::Hold;
        x_ = 0.0;
        if (holdSteps_ <= 0.0) enterDecay();
    }

    void enterDecay() noexcept
    {
        stage_ = Stage::Decay;
        from_ = 1.0;
        to_ = sustain_;
        curveNow_ = decayCurve_;
        decayScale_ = 1.0;
        x_ = 0.0;
        aim(stageSamples());
    }

    void enterRelease() noexcept
    {
        stage_ = Stage::Release;
        from_ = 1.0;
        to_ = 0.0;
        curveNow_ = releaseCurve_;
        x_ = inverse(1.0 - std::clamp(value_, 0.0, 1.0), curveNow_);   // from where the output is
        aim(releaseN_);
    }

    void advance() noexcept
    {
        switch (stage_)
        {
            case Stage::Idle:
                value_ = 0.0;
                return;
            case Stage::Delay:
                value_ = held_;
                if (++x_ >= delaySteps_) enterAttack();
                return;
            case Stage::Hold:
                value_ = 1.0;
                if (++x_ >= holdSteps_) enterDecay();
                return;
            case Stage::Sustain:
                value_ = sustain_;
                return;
            case Stage::Attack:
            case Stage::Decay:
            case Stage::Release:
                break;
        }
        x_ += moveStep_;
        if (--left_ <= 0)
        {
            value_ = to_;
            if (stage_ == Stage::Attack) enterHold();
            else if (stage_ == Stage::Decay) { stage_ = Stage::Sustain; value_ = sustain_; }
            else { stage_ = Stage::Idle; value_ = 0.0; }
            return;
        }
        value_ = from_ + (to_ - from_) * shape(x_, curveNow_);
    }

    /// The audio trigger: a peak level of the key, with the hysteresis and the wait.
    void listen(double key) noexcept
    {
        const double a = std::isfinite(key) ? std::abs(key) : 0.0;
        level_ = std::max(a, level_ * levelFall_);
        if (!gate_)
        {
            if (level_ > threshold_)
            {
                gate_ = true;
                below_ = 0;
                noteOn();
            }
            return;
        }
        if (level_ >= lowLevel_) { below_ = 0; return; }
        if (++below_ >= waitSamples_)
        {
            gate_ = false;
            noteOff();
        }
    }

    [[nodiscard]] T output() const noexcept
    {
        if (!bipolar_) return static_cast<T>(value_);
        const double s = sustain_;
        if (value_ >= s) return static_cast<T>(s < 1.0 ? (value_ - s) / (1.0 - s) : 0.0);
        return static_cast<T>(s > 0.0 ? (value_ - s) / s : 0.0);
    }

    static constexpr double kDetectorReleaseSeconds = 0.01;   ///< Fall of the key's peak level.

    double sampleRate_ = 48000.0;
    double delayMs_ = 0.0, attackMs_ = 10.0, holdMs_ = 0.0, decayMs_ = 100.0, releaseMs_ = 200.0;
    double sustain_ = 0.7;
    double attackCurve_ = 0.0, decayCurve_ = 0.0, releaseCurve_ = 0.0;
    bool bipolar_ = false;

    double delaySteps_ = 0.0, holdSteps_ = 0.0;
    double attackN_ = 0.0, releaseN_ = 0.0;   ///< Samples of a whole attack and release.

    Stage stage_ = Stage::Idle;
    double value_ = 0.0;
    double x_ = 0.0;          ///< The phase of a moving stage, or the samples of a waiting one.
    double from_ = 0.0, to_ = 0.0;
    double curveNow_ = 0.0;
    double moveStep_ = 0.0;
    int64_t left_ = 0;        ///< Samples left of a moving stage.
    double decayScale_ = 1.0; ///< Share of the decay time a decay segment takes (a sustain move: its distance).
    double held_ = 0.0;       ///< The value the delay holds.

    Trigger trigger_ = Trigger::Notes;
    double thresholdDb_ = -40.0, hysteresisDb_ = 6.0, waitMs_ = 50.0;
    double threshold_ = 0.01, lowLevel_ = 0.005, levelFall_ = 0.99;
    int64_t waitSamples_ = 2400;
    bool gate_ = false;
    double level_ = 0.0;
    int64_t below_ = 0;
};

} // namespace dspark
