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
 *
 * The rate is in Hz, or a note division of a tempo in BPM. Delay and fade-in
 * shape the output's depth after reset(): silent for the delay, then a
 * linear rise to full depth over the fade-in. They concern the depth only;
 * the cycle runs from reset() on.
 *
 * Threading: owner-managed. Setters, reset() and generation belong to the
 * owning thread (a synthesizer voice or its engine). valueAt() is a pure
 * static function. No allocation anywhere.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>
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
        SmoothRandom
    };
    static constexpr int kNumShapes = 7;

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
    void setSeed(uint64_t seed) noexcept { seed_ = seed; }

    /** @brief Effective rate in Hz (the synced one when tempo sync is on). */
    [[nodiscard]] double getRateHz() const noexcept { return increment_ * sampleRate_; }

    /** @brief Cycles run since reset(), start phase included. */
    [[nodiscard]] double getCycles() const noexcept { return cycles_; }

    // -- Generation (owning thread) --------------------------------------------

    /** @brief The output now (shape times delay/fade depth), without advancing. */
    [[nodiscard]] T value() const noexcept
    {
        return static_cast<T>(valueAt(shape_, cycles_, seed_) * depthAt(age_));
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
     * with `seed`; the fraction is the phase within the cycle.
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

    double increment_ = 1.0 / 48000.0;
    double cycles_ = 0.0;
    int64_t age_ = 0;
    int64_t delaySamples_ = 0;
    int64_t fadeSamples_ = 0;
};

} // namespace dspark
