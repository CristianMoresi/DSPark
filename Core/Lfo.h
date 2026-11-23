// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Lfo.h
 * @brief Low-frequency oscillator for modulation: classic shapes, seeded
 *        random shapes, free or tempo-synced rate, delay and fade-in.
 *
 * Every shape is a pure function of the absolute cycle position and a seed
 * (valueAt()): the random shapes draw one value per cycle from a hash of the
 * cycle index, so two LFOs with the same rate, phase and seed agree exactly,
 * and an owner can read a shared free-running LFO at any instant of a block
 * without stepping it sample by sample. Shapes are bipolar, -1 .. +1, and
 * not band-limited (control signals; an owner that needs a smooth result
 * interpolates or filters them).
 *
 * - Sine, Triangle: start at 0 rising, peak +1 at a quarter cycle.
 * - SawUp / SawDown: -1 to +1 / +1 to -1 over the cycle.
 * - Square: +1 for the first half cycle, -1 for the second.
 * - SampleAndHold: a new uniform random value every cycle.
 * - SmoothRandom: a cosine glide from one random value to the next over each
 *   cycle: continuous in value, with zero slope at every cycle boundary.
 * - Steps: a sequence of up to 16 steps over the cycle (setNumSteps(),
 *   setStep()). Each step holds its value, or glides to it from the step
 *   before over a share of its length along a curve; a random step draws a
 *   value per cycle from the seed that is always at least a quarter of the
 *   room away from the step before; a swing moves the boundary inside each
 *   pair of steps; the values can be quantized to a grid (semitones, when
 *   the owner maps the output to a pitch range). It depends on the
 *   instance's steps, so it is read with stepsAt(), not valueAt().
 *
 * reset() can glide from the output it was at to the new start over a few
 * milliseconds (setRetriggerGlideMs()), so a retrigger does not step.
 *
 * The rate is in Hz, or a note division of a tempo in BPM. Delay and fade-in
 * shape the output's depth after reset(): silent for the delay, then a
 * linear rise to full depth over the fade-in. They concern the depth only;
 * the cycle runs from reset() on.
 *
 * Threading: owner-managed. Setters, reset() and generation belong to the
 * owning thread (a synthesizer voice or its engine). valueAt() is a pure
 * static function; stepsAt() reads the instance (and caches the cycle it
 * read last). No allocation anywhere.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace dspark {

/**
 * @class Lfo
 * @brief Modulation oscillator with tempo sync, delay, fade-in and seeded
 *        random shapes.
 *
 * @code
 *   dspark::Lfo<float> lfo;
 *   lfo.prepare(48000.0);
 *   lfo.setShape(dspark::Lfo<float>::Shape::Triangle);
 *   lfo.setRateHz(5.0);
 *   lfo.setDelayMs(200.0);       // vibrato that waits...
 *   lfo.setFadeInMs(300.0);      // ...and grows in
 *   lfo.reset();                 // at note-on
 *   float v = lfo.advance(16);   // value now, then 16 samples on
 * @endcode
 *
 * @tparam T Output type (float or double). Phase arithmetic is double.
 */
template <FloatType T>
class Lfo
{
public:
    enum class Shape
    {
        Sine,
        Triangle,
        SawUp,
        SawDown,
        Square,
        SampleAndHold,
        SmoothRandom,
        Steps
    };
    static constexpr int kNumShapes = 8;

    /// Most steps of the Steps shape.
    static constexpr int kMaxSteps = 16;

    /** @brief Note values for tempo sync (four beats to the whole note). */
    enum class Division
    {
        EightBars, FourBars, TwoBars, Whole,
        HalfDotted, Half, HalfTriplet,
        QuarterDotted, Quarter, QuarterTriplet,
        EighthDotted, Eighth, EighthTriplet,
        SixteenthDotted, Sixteenth, SixteenthTriplet,
        ThirtySecond, ThirtySecondTriplet, SixtyFourth
    };
    static constexpr int kNumDivisions = 19;

    /** @brief Beats (quarter notes) per cycle of a division. */
    [[nodiscard]] static constexpr double beatsPerCycle(Division d) noexcept
    {
        switch (d)
        {
            case Division::EightBars:           return 32.0;
            case Division::FourBars:            return 16.0;
            case Division::TwoBars:             return 8.0;
            case Division::Whole:               return 4.0;
            case Division::HalfDotted:          return 3.0;
            case Division::Half:                return 2.0;
            case Division::HalfTriplet:         return 4.0 / 3.0;
            case Division::QuarterDotted:       return 1.5;
            case Division::Quarter:             return 1.0;
            case Division::QuarterTriplet:      return 2.0 / 3.0;
            case Division::EighthDotted:        return 0.75;
            case Division::Eighth:              return 0.5;
            case Division::EighthTriplet:       return 1.0 / 3.0;
            case Division::SixteenthDotted:     return 0.375;
            case Division::Sixteenth:           return 0.25;
            case Division::SixteenthTriplet:    return 1.0 / 6.0;
            case Division::ThirtySecond:        return 0.125;
            case Division::ThirtySecondTriplet: return 1.0 / 12.0;
            case Division::SixtyFourth:         return 0.0625;
        }
        return 1.0;
    }

    /// Highest free rate (Hz).
    static constexpr double kMaxRateHz = 1000.0;

    /** @brief Prepares for a sample rate (invalid rates are ignored). */
    void prepare(double sampleRate) noexcept
    {
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) return;
        sampleRate_ = sampleRate;
        updateIncrement();
        updateEnvelope();
    }

    /**
     * @brief Restarts the cycle at the start phase, the cycle count at zero
     *        and the delay and fade-in from the beginning.
     */
    void reset() noexcept
    {
        // The glide of a retrigger starts from where the output is now.
        retrigFrom_ = static_cast<double>(value());
        retrigLeft_ = retrigSamples_;
        cycles_ = startPhase_;
        age_ = 0;
    }

    // -- Parameters (owning thread) -------------------------------------------

    void setShape(Shape s) noexcept { shape_ = s; }
    [[nodiscard]] Shape getShape() const noexcept { return shape_; }

    /** @brief Free rate in Hz, 0..kMaxRateHz (NaN ignored). Used while sync is off. */
    void setRateHz(double hz) noexcept
    {
        if (hz != hz) return;
        rateHz_ = std::clamp(hz, 0.0, kMaxRateHz);
        updateIncrement();
    }

    /** @brief Tempo sync on or off; when on, the rate is the division of the tempo. */
    void setTempoSync(bool on) noexcept { sync_ = on; updateIncrement(); }
    void setDivision(Division d) noexcept { division_ = d; updateIncrement(); }

    /** @brief Tempo in BPM for tempo sync, 1..999 (invalid values ignored). */
    void setTempo(double bpm) noexcept
    {
        if (!(bpm >= 1.0) || !(bpm <= 999.0)) return;
        bpm_ = bpm;
        updateIncrement();
    }

    /** @brief Cycle position reset() starts at, 0..1 (wrapped). */
    void setStartPhase(double phase) noexcept
    {
        if (!std::isfinite(phase)) return;
        startPhase_ = phase - std::floor(phase);
    }

    /** @brief Silence before the depth starts rising, in ms (>= 0). */
    void setDelayMs(double ms) noexcept
    {
        if (!(ms >= 0.0) || !std::isfinite(ms)) return;
        delayMs_ = ms;
        updateEnvelope();
    }

    /** @brief Linear rise to full depth after the delay, in ms (>= 0). */
    void setFadeInMs(double ms) noexcept
    {
        if (!(ms >= 0.0) || !std::isfinite(ms)) return;
        fadeMs_ = ms;
        updateEnvelope();
    }

    /** @brief Seed of the random shapes (the same seed repeats the same sequence). */
    void setSeed(uint64_t seed) noexcept { seed_ = seed; stepCacheCycle_ = kNoCycle; }

    /**
     * @brief How long reset() glides from the output it was at to the new
     *        start, in ms (0, the default, jumps; >= 0, non-finite ignored).
     */
    void setRetriggerGlideMs(double ms) noexcept
    {
        if (!(ms >= 0.0) || !std::isfinite(ms)) return;
        retrigMs_ = ms;
        updateEnvelope();
    }

    // -- The Steps shape (owning thread) ---------------------------------------

    /** @brief Steps in a cycle, 1..kMaxSteps (16, the default). */
    void setNumSteps(int count) noexcept
    {
        numSteps_ = std::clamp(count, 1, kMaxSteps);
        stepCacheCycle_ = kNoCycle;
    }
    [[nodiscard]] int getNumSteps() const noexcept { return numSteps_; }

    /**
     * @brief One step: its value, how much of its length it glides to it
     *        from the step before, and the curve of that glide.
     *
     * @param index  0..kMaxSteps-1 (others are ignored).
     * @param value  -1..1.
     * @param glide  0 (holds from its start) to 1 (glides over all of it).
     * @param curve  -1 (slow start) to 1 (fast start); 0 is a straight line.
     * Non-finite values are ignored.
     */
    void setStep(int index, double value, double glide = 0.0, double curve = 0.0) noexcept
    {
        if (index < 0 || index >= kMaxSteps || !std::isfinite(value) || !std::isfinite(glide) || !std::isfinite(curve)) return;
        const auto i = static_cast<size_t>(index);
        stepValue_[i] = std::clamp(value, -1.0, 1.0);
        stepGlide_[i] = std::clamp(glide, 0.0, 1.0);
        stepCurve_[i] = std::clamp(curve, -1.0, 1.0);
        stepCacheCycle_ = kNoCycle;
    }

    /** @brief Makes a step draw its value from the seed every cycle (see the file notes). */
    void setStepRandom(int index, bool random) noexcept
    {
        if (index < 0 || index >= kMaxSteps) return;
        stepRandom_[static_cast<size_t>(index)] = random;
        stepCacheCycle_ = kNoCycle;
    }

    [[nodiscard]] double getStepValue(int index) const noexcept { return stepValue_[stepIndex(index)]; }
    [[nodiscard]] double getStepGlide(int index) const noexcept { return stepGlide_[stepIndex(index)]; }
    [[nodiscard]] double getStepCurve(int index) const noexcept { return stepCurve_[stepIndex(index)]; }
    [[nodiscard]] bool isStepRandom(int index) const noexcept { return stepRandom_[stepIndex(index)]; }

    /**
     * @brief Moves the boundary inside each pair of steps, -1..1: at 0.5 the
     *        first step of a pair lasts 1.25 steps and the second 0.75
     *        (the boundary moves by half the value, in steps).
     */
    void setStepSwing(double swing) noexcept
    {
        if (std::isfinite(swing)) swing_ = std::clamp(swing, -1.0, 1.0);
    }
    [[nodiscard]] double getStepSwing() const noexcept { return swing_; }

    /**
     * @brief Quantizes the step values to a grid of 1/divisions (0, the
     *        default, leaves them free; 12 puts them on semitones when the
     *        owner maps -1..1 to -12..12 semitones). Glides move between
     *        the quantized values.
     */
    void setStepQuantize(int divisions) noexcept
    {
        stepQuantize_ = std::clamp(divisions, 0, 1200);
        stepCacheCycle_ = kNoCycle;
    }
    [[nodiscard]] int getStepQuantize() const noexcept { return stepQuantize_; }

    /**
     * @brief The Steps shape's value at an absolute cycle position (the
     *        integer part is the cycle whose random steps are drawn).
     */
    [[nodiscard]] double stepsAt(double cycles) const noexcept
    {
        if (!std::isfinite(cycles)) cycles = 0.0;
        const double index = std::floor(cycles);
        const double ph = cycles - index;
        const auto cycle = static_cast<int64_t>(index);
        if (cycle != stepCacheCycle_) fillStepCache(cycle);
        // Where the phase is in the steps, with the swing inside each pair.
        const double pos = ph * static_cast<double>(numSteps_);
        int step = static_cast<int>(pos);
        double within = pos - static_cast<double>(step);
        if (swing_ != 0.0 && numSteps_ >= 2)
        {
            const int pair = step / 2;
            if (2 * pair + 1 < numSteps_)
            {
                const double inPair = pos - 2.0 * static_cast<double>(pair);   // 0..2
                const double edge = 1.0 + 0.5 * swing_;
                if (inPair < edge) { step = 2 * pair; within = inPair / edge; }
                else { step = 2 * pair + 1; within = (inPair - edge) / (2.0 - edge); }
            }
        }
        step = std::clamp(step, 0, numSteps_ - 1);
        const double to = stepCache_[static_cast<size_t>(step)];
        const double from = step > 0 ? stepCache_[static_cast<size_t>(step - 1)] : stepCacheBefore_;
        const double glide = stepGlide_[static_cast<size_t>(step)];
        if (!(within < glide)) return to;
        const double x = within / glide;
        const double c = stepCurve_[static_cast<size_t>(step)];
        const double bent = c > 0.0 ? 1.0 - std::pow(1.0 - x, 1.0 + 4.0 * c)
                          : c < 0.0 ? std::pow(x, 1.0 - 4.0 * c) : x;
        return from + (to - from) * bent;
    }

    /** @brief Effective rate in Hz (the synced one when tempo sync is on). */
    [[nodiscard]] double getRateHz() const noexcept { return increment_ * sampleRate_; }

    /** @brief Cycles run since reset(), start phase included. */
    [[nodiscard]] double getCycles() const noexcept { return cycles_; }

    // -- Generation (owning thread) --------------------------------------------

    /** @brief The output now (shape times delay/fade depth), without advancing. */
    [[nodiscard]] T value() const noexcept
    {
        const double shaped = (shape_ == Shape::Steps) ? stepsAt(cycles_) : valueAt(shape_, cycles_, seed_);
        const double v = shaped * depthAt(age_);
        if (retrigLeft_ <= 0) return static_cast<T>(v);
        const double share = static_cast<double>(retrigLeft_) / static_cast<double>(retrigSamples_);
        return static_cast<T>(v + (retrigFrom_ - v) * share);
    }

    /**
     * @brief Returns the output now, then advances by `samples` at the
     *        current rate (a control-rate owner advances its control period).
     */
    T advance(int samples = 1) noexcept
    {
        const T v = value();
        const auto n = static_cast<int64_t>(std::max(samples, 0));
        cycles_ += increment_ * static_cast<double>(n);
        age_ += n;
        retrigLeft_ = std::max<int64_t>(0, retrigLeft_ - n);
        return v;
    }

    /**
     * @brief Advances by `samples` at `rateScale` times the current rate
     *        (rate modulation; the scale is clamped to 0..64).
     */
    T advanceScaled(int samples, double rateScale) noexcept
    {
        const T v = value();
        const auto n = static_cast<int64_t>(std::max(samples, 0));
        const double scale = (rateScale >= 0.0) ? std::min(rateScale, 64.0) : 0.0;
        cycles_ += increment_ * scale * static_cast<double>(n);
        age_ += n;
        retrigLeft_ = std::max<int64_t>(0, retrigLeft_ - n);
        return v;
    }

    /**
     * @brief Depth of the delay and fade-in `samples` after reset(): 0 during
     *        the delay, then a linear rise to 1.
     */
    [[nodiscard]] double depthAt(int64_t samples) const noexcept
    {
        if (samples < delaySamples_) return 0.0;
        if (fadeSamples_ <= 0) return 1.0;
        const double x = static_cast<double>(samples - delaySamples_) / static_cast<double>(fadeSamples_);
        return std::min(x, 1.0);
    }

    /**
     * @brief The shape's value at an absolute cycle position.
     *
     * The integer part of `cycles` is the cycle index the random shapes hash
     * with `seed`; the fraction is the phase within the cycle. Steps depends
     * on an instance's steps: it reads 0 here (use stepsAt()).
     */
    [[nodiscard]] static double valueAt(Shape shape, double cycles, uint64_t seed) noexcept
    {
        if (!std::isfinite(cycles)) cycles = 0.0;
        const double index = std::floor(cycles);
        const double ph = cycles - index;
        switch (shape)
        {
            case Shape::Sine:     return std::sin(twoPi<double> * ph);
            case Shape::Triangle: return (ph < 0.25) ? 4.0 * ph
                                       : (ph < 0.75) ? 2.0 - 4.0 * ph
                                                     : 4.0 * ph - 4.0;
            case Shape::SawUp:    return 2.0 * ph - 1.0;
            case Shape::SawDown:  return 1.0 - 2.0 * ph;
            case Shape::Square:   return (ph < 0.5) ? 1.0 : -1.0;
            case Shape::SampleAndHold:
                return random(seed, static_cast<int64_t>(index));
            case Shape::SmoothRandom:
            {
                const double a = random(seed, static_cast<int64_t>(index));
                const double b = random(seed, static_cast<int64_t>(index) + 1);
                return a + (b - a) * 0.5 * (1.0 - std::cos(pi<double> * ph));
            }
            case Shape::Steps: return 0.0;
        }
        return 0.0;
    }

    /** @brief Uniform value in [-1, 1) for a cycle index (splitmix64 hash). */
    [[nodiscard]] static double random(uint64_t seed, int64_t index) noexcept
    {
        uint64_t x = seed + 0x9E3779B97F4A7C15ull * (static_cast<uint64_t>(index) + 1ull);
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        x ^= x >> 31;
        return static_cast<double>(x >> 11) * (2.0 / 9007199254740992.0) - 1.0;
    }

private:
    void updateIncrement() noexcept
    {
        const double hz = sync_ ? (bpm_ / 60.0) / beatsPerCycle(division_) : rateHz_;
        increment_ = hz / sampleRate_;
    }

    void updateEnvelope() noexcept
    {
        delaySamples_ = static_cast<int64_t>(std::llround(delayMs_ * 0.001 * sampleRate_));
        fadeSamples_ = static_cast<int64_t>(std::llround(fadeMs_ * 0.001 * sampleRate_));
        retrigSamples_ = static_cast<int64_t>(std::llround(retrigMs_ * 0.001 * sampleRate_));
        retrigLeft_ = std::min(retrigLeft_, retrigSamples_);
    }

    [[nodiscard]] static size_t stepIndex(int index) noexcept { return static_cast<size_t>(std::clamp(index, 0, kMaxSteps - 1)); }

    /// A step's value in a cycle: its own, or drawn from the seed at least a
    /// quarter of the room away from `before`, toward the side with more
    /// room; then on the grid.
    [[nodiscard]] double stepValueIn(int64_t cycle, int step, double before) const noexcept
    {
        double v = stepValue_[static_cast<size_t>(step)];
        if (stepRandom_[static_cast<size_t>(step)])
        {
            const double u = 0.5 * (random(seed_ ^ 0x5DEECE66Dull, cycle * kMaxSteps + step) + 1.0);   // 0..1
            const bool up = before < 0.0;
            const double room = up ? 1.0 - before : before + 1.0;   // 1..2
            const double away = room * (0.25 + 0.75 * u);
            v = std::clamp(up ? before + away : before - away, -1.0, 1.0);
        }
        return quantized(v);
    }

    /// A value on the grid of setStepQuantize().
    [[nodiscard]] double quantized(double v) const noexcept
    {
        if (stepQuantize_ <= 0) return v;
        const double q = static_cast<double>(stepQuantize_);
        return std::clamp(std::round(v * q) / q, -1.0, 1.0);
    }

    /// The step values of a cycle, and the last value of the cycle before
    /// (where its first step glides from). A cycle depends on its own draws
    /// only: a random first step moves away from the value the last step
    /// is set to (the value it plays when it is not random), so any cycle
    /// reads alike whatever was read before it.
    void fillStepCache(int64_t cycle) const noexcept
    {
        auto walk = [&](int64_t c, std::array<double, kMaxSteps>& out) {
            const int last = numSteps_ - 1;
            double before = quantized(stepValue_[static_cast<size_t>(last)]);
            for (int s = 0; s < numSteps_; ++s)
            {
                out[static_cast<size_t>(s)] = stepValueIn(c, s, before);
                before = out[static_cast<size_t>(s)];
            }
        };
        std::array<double, kMaxSteps> previous {};
        walk(cycle - 1, previous);
        stepCacheBefore_ = previous[static_cast<size_t>(numSteps_ - 1)];
        walk(cycle, stepCache_);
        stepCacheCycle_ = cycle;
    }

    double sampleRate_ = 48000.0;
    Shape shape_ = Shape::Sine;
    double rateHz_ = 1.0;
    bool sync_ = false;
    Division division_ = Division::Quarter;
    double bpm_ = 120.0;
    double startPhase_ = 0.0;
    double delayMs_ = 0.0;
    double fadeMs_ = 0.0;
    uint64_t seed_ = 0;

    // The Steps shape.
    static constexpr int64_t kNoCycle = INT64_MIN;
    int numSteps_ = kMaxSteps;
    std::array<double, kMaxSteps> stepValue_ {};
    std::array<double, kMaxSteps> stepGlide_ {};
    std::array<double, kMaxSteps> stepCurve_ {};
    std::array<bool, kMaxSteps> stepRandom_ {};
    double swing_ = 0.0;
    int stepQuantize_ = 0;
    mutable std::array<double, kMaxSteps> stepCache_ {};   ///< The step values of stepCacheCycle_.
    mutable double stepCacheBefore_ = 0.0;                 ///< The last step value of the cycle before.
    mutable int64_t stepCacheCycle_ = kNoCycle;
    // The glide of a retrigger.
    double retrigMs_ = 0.0;
    int64_t retrigSamples_ = 0;
    int64_t retrigLeft_ = 0;
    double retrigFrom_ = 0.0;

    double increment_ = 1.0 / 48000.0;
    double cycles_ = 0.0;
    int64_t age_ = 0;
    int64_t delaySamples_ = 0;
    int64_t fadeSamples_ = 0;
};

} // namespace dspark
