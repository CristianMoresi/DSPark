// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file NormalizedLadder.h
 * @brief Second-order section as a normalized ladder: two rotations and three
 *        taps, passive under any change of its coefficients.
 *
 * The normalized ladder (A. H. Gray and J. D. Markel, "A Normalized Digital
 * Filter Structure", IEEE Trans. ASSP 23(3), 1975) realises the denominator
 * 1 + a1 z^-1 + a2 z^-2 as two plane rotations, one per reflection
 * coefficient (k2 = a2, k1 = a1 / (1 + a2)), and the numerator as three taps
 * on the signals between them:
 *
 * @code
 *   f1 = c2 u  - s2 x2        g2 = s2 u  + c2 x2
 *   f0 = c1 f1 - s1 x1        g1 = s1 f1 + c1 x1
 *   y  = v0 f0 + v1 g1 + v2 g2
 *   x2 <- g1                  x1 <- f0
 * @endcode
 *
 * with s_i = k_i and c_i = sqrt(1 - k_i^2). The transfer functions from the
 * input are f0 = c1 c2 / A, g1 = c2 (k1 + z^-1) / A and
 * g2 = (a2 + a1 z^-1 + z^-2) / A, so the taps of a numerator
 * b0 + b1 z^-1 + b2 z^-2 are v2 = b2, v1 = (b1 - b2 a1) / c2 and
 * v0 = (b0 - b2 a2 - (b1 - b2 a1) k1) / (c1 c2).
 *
 * What the structure is for is in the rotations: the step from (u, x1, x2) to
 * (g2, x1, x2) is orthogonal, so with no input the energy of the memory can
 * only leave, through g2, whatever the angles are and however they change
 * from one sample to the next. A direct form whose coefficients jump can add
 * energy to its own memory; this one cannot, at audio-rate modulation or
 * under a hard switch. A linear ramp between two settings keeps each pair
 * (c, s) on or inside the unit circle, so it is passive along the way as
 * well. The price is arithmetic: eleven products a sample against five.
 *
 * A filter that sits still is better served by Biquad or BiquadCascade; this
 * one is for sections whose coefficients move fast.
 *
 * Threading: owner-managed, like BiquadCascade. Every method belongs to the
 * thread that processes (or to a single thread, offline).
 *
 * Dependencies: AudioBuffer.h, Biquad.h, DenormalGuard.h.
 */

#include "AudioBuffer.h"
#include "Biquad.h"
#include "DenormalGuard.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace dspark {

/**
 * @class NormalizedLadder
 * @brief One second-order section in normalized ladder form, per channel.
 *
 * @tparam T           Sample type of the buffers (float or double); the
 *                     recursion and its memory are double either way.
 * @tparam MaxChannels Largest number of channels.
 */
template <typename T, int MaxChannels = 8>
class NormalizedLadder
{
public:
    /** @brief The two rotations and the three taps (the identity by default). */
    struct Coeffs
    {
        double c1 = 1.0, s1 = 0.0;   ///< Inner rotation: s1 = k1 = a1 / (1 + a2).
        double c2 = 1.0, s2 = 0.0;   ///< Outer rotation: s2 = k2 = a2.
        double v0 = 1.0, v1 = 0.0, v2 = 0.0;   ///< Taps on f0, g1 and g2.
    };

    /**
     * @brief The ladder of a biquad.
     *
     * A set whose poles are on or outside the unit circle has a reflection
     * coefficient of magnitude one or more; it is brought just inside, so the
     * section stays passive whatever it is given.
     *
     * @param b Normalised biquad coefficients.
     * @return Rotations and taps with the same transfer function.
     */
    [[nodiscard]] static Coeffs fromBiquad(const BiquadCoeffs& b) noexcept
    {
        constexpr double kLimit = 1.0 - 1e-12;
        Coeffs l;
        const double k2 = std::clamp(b.a2, -kLimit, kLimit);
        const double k1 = std::clamp(b.a1 / (1.0 + k2), -kLimit, kLimit);
        // The denominator the rotations realise (the given one for a stable set).
        const double a1 = k1 * (1.0 + k2);
        const double a2 = k2;
        l.s1 = k1; l.c1 = std::sqrt(1.0 - k1 * k1);
        l.s2 = k2; l.c2 = std::sqrt(1.0 - k2 * k2);
        l.v2 = b.b2;
        const double middle = b.b1 - b.b2 * a1;
        l.v1 = middle / l.c2;
        l.v0 = (b.b0 - b.b2 * a2 - middle * k1) / (l.c1 * l.c2);
        return l;
    }

    /**
     * @brief The biquad a ladder realises (for curves and checks).
     * @param l Rotations and taps.
     * @return Normalised biquad coefficients.
     */
    [[nodiscard]] static BiquadCoeffs toBiquad(const Coeffs& l) noexcept
    {
        BiquadCoeffs b;
        b.a2 = l.s2;
        b.a1 = l.s1 * (1.0 + l.s2);
        b.b2 = l.v2;
        b.b1 = l.v2 * b.a1 + l.v1 * l.c2;
        b.b0 = l.v2 * b.a2 + l.v1 * l.c2 * l.s1 + l.v0 * l.c1 * l.c2;
        return b;
    }

    /** @brief Makes a biquad's response the section's own from the next sample; ends a ramp. */
    void setCoeffs(const BiquadCoeffs& b) noexcept { setLadder(fromBiquad(b)); }

    /** @brief Sets rotations and taps directly from the next sample; ends a ramp. */
    void setLadder(const Coeffs& l) noexcept
    {
        current_ = l;
        target_ = l;
        rampLeft_ = 0;
    }

    /**
     * @brief Moves the rotations and the taps linearly to those of a biquad.
     *
     * The ramp starts from the values in use and lands exactly on the target
     * after `samples` samples. Along it each rotation stays on or inside the
     * unit circle: the section never adds energy to its memory.
     *
     * @param target  Response to arrive at.
     * @param samples Length of the ramp in samples; 0 or less sets it at once.
     */
    void rampCoeffs(const BiquadCoeffs& target, int samples) noexcept
    {
        if (samples <= 0) { setCoeffs(target); return; }
        target_ = fromBiquad(target);
        rampLeft_ = samples;
        const double inverse = 1.0 / static_cast<double>(samples);
        step_.c1 = (target_.c1 - current_.c1) * inverse; step_.s1 = (target_.s1 - current_.s1) * inverse;
        step_.c2 = (target_.c2 - current_.c2) * inverse; step_.s2 = (target_.s2 - current_.s2) * inverse;
        step_.v0 = (target_.v0 - current_.v0) * inverse; step_.v1 = (target_.v1 - current_.v1) * inverse;
        step_.v2 = (target_.v2 - current_.v2) * inverse;
    }

    /** @brief Rotations and taps in use now (mid ramp, where the ramp is). */
    [[nodiscard]] Coeffs getLadder() const noexcept { return current_; }

    /** @brief Whether the coefficients are moving. */
    [[nodiscard]] bool isRamping() const noexcept { return rampLeft_ > 0; }

    /** @brief Clears the memory of every channel. Coefficients and ramp are kept. */
    void reset() noexcept
    {
        for (auto& each : state_) each = {};
    }

    /** @brief The energy in a channel's memory, x1^2 + x2^2 (it only falls with no input). */
    [[nodiscard]] double getStateEnergy(int channel) const noexcept
    {
        const State& s = state_[static_cast<size_t>(std::clamp(channel, 0, MaxChannels - 1))];
        return s.x1 * s.x1 + s.x2 * s.x2;
    }

    /**
     * @brief Runs a block through the section, in place.
     *
     * Channels beyond MaxChannels are left untouched; every channel walks
     * the same ramp. A memory a non-finite sample has reached is cleared at
     * the end of the block.
     *
     * @param buffer Audio to filter.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numChannels = std::min(buffer.getNumChannels(), MaxChannels);
        const int numSamples = buffer.getNumSamples();
        if (numChannels <= 0 || numSamples <= 0) return;

        DenormalGuard guard;
        const int moving = std::min(rampLeft_, numSamples);
        Coeffs after = current_;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            T* data = buffer.getChannel(ch);
            State& s = state_[static_cast<size_t>(ch)];
            double x1 = s.x1, x2 = s.x2;
            Coeffs c = current_;
            for (int i = 0; i < moving; ++i)
            {
                c.c1 += step_.c1; c.s1 += step_.s1; c.c2 += step_.c2; c.s2 += step_.s2;
                c.v0 += step_.v0; c.v1 += step_.v1; c.v2 += step_.v2;
                data[i] = static_cast<T>(advance(static_cast<double>(data[i]), c, x1, x2));
            }
            if (moving > 0 && moving == rampLeft_) c = target_;   // landed: exactly the target from here
            for (int i = moving; i < numSamples; ++i)
                data[i] = static_cast<T>(advance(static_cast<double>(data[i]), c, x1, x2));
            if (!std::isfinite(x1 + x2)) { x1 = 0.0; x2 = 0.0; }
            s.x1 = x1;
            s.x2 = x2;
            after = c;
        }
        current_ = after;
        rampLeft_ -= moving;
    }

    /**
     * @brief One sample of one channel, in double, with the coefficients as they are.
     *
     * Ramps advance in processBlock(), or by the caller's advanceRamp() once
     * per frame.
     *
     * @pre channel in [0, MaxChannels).
     */
    [[nodiscard]] double processSample(double input, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels && "Channel index out of bounds");
        State& s = state_[static_cast<size_t>(channel)];
        return advance(input, current_, s.x1, s.x2);
    }

    /**
     * @brief Moves the ramp on by a number of samples (for processSample() callers).
     * @param samples Samples gone by; a ramp with fewer left lands on its target.
     */
    void advanceRamp(int samples) noexcept
    {
        if (rampLeft_ <= 0 || samples <= 0) return;
        if (samples >= rampLeft_)
        {
            current_ = target_;
            rampLeft_ = 0;
            return;
        }
        rampLeft_ -= samples;
        for (int i = 0; i < samples; ++i)
        {
            current_.c1 += step_.c1; current_.s1 += step_.s1; current_.c2 += step_.c2; current_.s2 += step_.s2;
            current_.v0 += step_.v0; current_.v1 += step_.v1; current_.v2 += step_.v2;
        }
    }

private:
    struct State
    {
        double x1 = 0.0;   ///< f0 one sample ago.
        double x2 = 0.0;   ///< g1 one sample ago.
    };

    /** @brief The two rotations and the taps for one sample. */
    static double advance(double u, const Coeffs& c, double& x1, double& x2) noexcept
    {
        const double f1 = c.c2 * u - c.s2 * x2;
        const double g2 = c.s2 * u + c.c2 * x2;
        const double f0 = c.c1 * f1 - c.s1 * x1;
        const double g1 = c.s1 * f1 + c.c1 * x1;
        x2 = g1;
        x1 = f0;
        return c.v0 * f0 + c.v1 * g1 + c.v2 * g2;
    }

    Coeffs current_ {};
    Coeffs target_ {};
    Coeffs step_ {};
    int rampLeft_ = 0;
    std::array<State, MaxChannels> state_ {};
};

} // namespace dspark
