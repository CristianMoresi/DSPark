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
 * one is for sections whose coefficients move fast. NormalizedLadderN is the
 * same structure for a whole filter of any order up to 16, designed from its
 * polynomials (no roots, no factoring).
 *
 * Threading: owner-managed, like BiquadCascade. Every method belongs to the
 * thread that processes (or to a single thread, offline).
 *
 * Dependencies: AudioBuffer.h, Biquad.h, DenormalGuard.h.
 */

#include "AudioBuffer.h"
#include "Biquad.h"
#include "DenormalGuard.h"
#include "SimdOps.h"

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
     * @brief Runs one channel's samples through the section, in double and in
     *        place, walking the ramp from where it stands without moving it on.
     *
     * For owners that run their channels one after another over the same
     * frames: every channel walks the same coefficients, and advanceRamp()
     * by the same count, once all of them have run, moves the ramp on (the
     * result is processBlock()'s, to the bit).
     *
     * @param data       Samples of the channel, filtered in place.
     * @param numSamples Number of samples.
     * @param channel    Channel whose memory the samples run through.
     * @pre channel in [0, MaxChannels).
     */
    void processSegment(double* data, int numSamples, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels && "Channel index out of bounds");
        if (numSamples <= 0) return;
        State& s = state_[static_cast<size_t>(channel)];
        double x1 = s.x1, x2 = s.x2;
        const int moving = std::min(rampLeft_, numSamples);
        Coeffs c = current_;
        for (int i = 0; i < moving; ++i)
        {
            c.c1 += step_.c1; c.s1 += step_.s1; c.c2 += step_.c2; c.s2 += step_.s2;
            c.v0 += step_.v0; c.v1 += step_.v1; c.v2 += step_.v2;
            data[i] = advance(data[i], c, x1, x2);
        }
        if (moving > 0 && moving == rampLeft_) c = target_;
        for (int i = moving; i < numSamples; ++i)
            data[i] = advance(data[i], c, x1, x2);
        if (!std::isfinite(x1 + x2)) { x1 = 0.0; x2 = 0.0; }
        s.x1 = x1;
        s.x2 = x2;
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


/**
 * @class NormalizedLadderN
 * @brief A whole filter of order N as one normalized ladder (Gray and Markel),
 *        designed from its polynomials, without roots.
 *
 * The denominator 1 + a1 z^-1 + ... + aN z^-N is stepped down to its
 * reflection coefficients: k_m is the last coefficient of the polynomial of
 * order m, and the polynomial of order m - 1 is (A_m - k_m B_m) / (1 - k_m^2),
 * where B_m is A_m reversed (z^-m A_m(1/z)). Each k_m is a plane rotation
 * (s = k_m, c = sqrt(1 - k_m^2)); the input enters at the top and runs down
 * through the rotations:
 *
 * @code
 *   f_N = u
 *   f_(m-1) = c_m f_m - s_m x_m      g_m = s_m f_m + c_m x_m      (m = N .. 1)
 *   y = v_0 f_0 + v_1 g_1 + ... + v_N g_N
 *   x_1 <- f_0, x_m <- g_(m-1)
 * @endcode
 *
 * The signal g_m is (c_(m+1) ... c_N) B_m / A, so the taps of a numerator
 * come down from the top: v_m is what is left of it at z^-m over the product
 * of the c above m, and v_m times that product times B_m comes off what is
 * left. The filter is stable when every |k_m| < 1, which the design checks.
 *
 * As the second-order NormalizedLadder, the step from the input and the
 * memory to the memory and g_N is orthogonal, so with no input the memory
 * can only lose energy whatever the coefficients do: a filter of order five
 * whose knob moves at audio rate keeps bounded. A ramp moves every c, s and
 * v linearly; the pairs (c, s) stay on or inside the unit circle on the way.
 * FilterDesign::bilinearPolynomial() takes an analog network of any order
 * (its numerator and denominator as polynomials in s) to the digital
 * polynomials this class takes, so a network from components needs no
 * factoring.
 *
 * Threading: owner-managed, like NormalizedLadder.
 *
 * @tparam T           Sample type of the buffers (float or double); the
 *                     recursion and its memory are double either way.
 * @tparam Order       Order of the filter, 1 to 16.
 * @tparam MaxChannels Largest number of channels.
 */
template <typename T, int Order, int MaxChannels = 8>
class NormalizedLadderN
{
    static_assert(Order >= 1 && Order <= 16, "orders 1 to 16");

public:
    /** @brief The ladder's rotations and taps. */
    struct Coeffs
    {
        std::array<double, Order + 1> c {};   ///< c_1 .. c_N (index 0 unused).
        std::array<double, Order + 1> s {};   ///< s_1 .. s_N = the reflection coefficients.
        std::array<double, Order + 1> v {};   ///< v_0 .. v_N, the taps.
    };

    /**
     * @brief The ladder of a digital filter b(z) / a(z), each Order + 1
     *        coefficients from z^0 (a[0] scales both).
     * @return False, and `out` untouched, when the denominator is not
     *         stable (a reflection coefficient of 1 or more) or a[0] is 0.
     */
    [[nodiscard]] static bool design(const double* b, const double* a, Coeffs& out) noexcept
    {
        if (!(std::abs(a[0]) > 0.0)) return false;
        std::array<double, Order + 1> poly {}, num {};
        for (int i = 0; i <= Order; ++i)
        {
            poly[static_cast<size_t>(i)] = a[i] / a[0];
            num[static_cast<size_t>(i)] = b[i] / a[0];
        }
        // The polynomials of every order, top down (kept for the taps).
        std::array<std::array<double, Order + 1>, Order + 1> stepped {};
        stepped[static_cast<size_t>(Order)] = poly;
        Coeffs l;
        for (int m = Order; m >= 1; --m)
        {
            const auto& am = stepped[static_cast<size_t>(m)];
            const double k = am[static_cast<size_t>(m)];
            if (!(std::abs(k) < 1.0)) return false;
            l.s[static_cast<size_t>(m)] = k;
            l.c[static_cast<size_t>(m)] = std::sqrt((1.0 - k) * (1.0 + k));
            auto& below = stepped[static_cast<size_t>(m - 1)];
            const double inverse = 1.0 / ((1.0 - k) * (1.0 + k));
            for (int i = 0; i < m; ++i)
                below[static_cast<size_t>(i)] = (am[static_cast<size_t>(i)] - k * am[static_cast<size_t>(m - i)]) * inverse;
        }
        // The taps, from the top: B_m is A_m reversed.
        double above = 1.0;   // the product of the c above m
        std::array<double, Order + 1> prod {};
        for (int m = Order; m >= 0; --m)
        {
            prod[static_cast<size_t>(m)] = above;
            if (m >= 1) above *= l.c[static_cast<size_t>(m)];
        }
        for (int m = Order; m >= 0; --m)
        {
            const double w = num[static_cast<size_t>(m)];
            l.v[static_cast<size_t>(m)] = w / prod[static_cast<size_t>(m)];
            const auto& am = stepped[static_cast<size_t>(m)];
            for (int i = 0; i <= m; ++i) num[static_cast<size_t>(i)] -= w * am[static_cast<size_t>(m - i)];
        }
        out = l;
        return true;
    }

    /** @brief Takes a filter b(z) / a(z) at once; false (nothing changes) if it is not stable. */
    bool setCoeffs(const double* b, const double* a) noexcept
    {
        Coeffs l;
        if (!design(b, a, l)) return false;
        current_ = target_ = l;
        rampLeft_ = 0;
        return true;
    }

    /** @brief Moves every rotation and tap linearly to a filter's over `samples`; false if it is not stable. */
    bool rampCoeffs(const double* b, const double* a, int samples) noexcept
    {
        Coeffs l;
        if (!design(b, a, l)) return false;
        if (samples <= 0)
        {
            current_ = target_ = l;
            rampLeft_ = 0;
            return true;
        }
        target_ = l;
        rampLeft_ = samples;
        const double inverse = 1.0 / static_cast<double>(samples);
        for (int m = 0; m <= Order; ++m)
        {
            const auto i = static_cast<size_t>(m);
            step_.c[i] = (l.c[i] - current_.c[i]) * inverse;
            step_.s[i] = (l.s[i] - current_.s[i]) * inverse;
            step_.v[i] = (l.v[i] - current_.v[i]) * inverse;
        }
        return true;
    }

    [[nodiscard]] Coeffs getLadder() const noexcept { return current_; }
    [[nodiscard]] bool isRamping() const noexcept { return rampLeft_ > 0; }

    /** @brief Clears the memory of every channel. */
    void reset() noexcept
    {
        for (auto& s : state_) s = {};
    }

    /** @brief The energy of a channel's memory (the sum of the squares of its states). */
    [[nodiscard]] double getStateEnergy(int channel) const noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return 0.0;
        double e = 0.0;
        for (double x : state_[static_cast<size_t>(channel)]) e += x * x;
        return e;
    }

    /** @brief Runs a block, in place; a ramp moves on by its samples. Channels beyond MaxChannels are left untouched. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numChannels = std::min(buffer.getNumChannels(), MaxChannels);
        const int numSamples = buffer.getNumSamples();
        if (numChannels <= 0 || numSamples <= 0) return;
        DenormalGuard guard;
        const int moving = std::min(rampLeft_, numSamples);
        Coeffs after = current_;
        // Channels go two at a time, as the two lanes of one register where
        // the target has one: each walks the same coefficients.
        for (int ch = 0; ch < numChannels; ch += 2)
        {
            const bool pair = ch + 1 < numChannels;
            auto& left = state_[static_cast<size_t>(ch)];
            auto& right = state_[static_cast<size_t>(pair ? ch + 1 : ch)];
            Coeffs c = current_;
            if constexpr (kPairChannels)
            {
                if (pair)
                {
                    runLanes<simd::kVecDoubleNarrow>(buffer.getChannel(ch), buffer.getChannel(ch + 1), numSamples,
                                                     moving, c, left, right);
                    after = c;
                    continue;
                }
            }
            runLanes<1>(buffer.getChannel(ch), nullptr, numSamples, moving, c, left, left);
            if (pair)
            {
                c = current_;
                runLanes<1>(buffer.getChannel(ch + 1), nullptr, numSamples, moving, c, right, right);
            }
            after = c;
        }
        current_ = after;
        rampLeft_ -= moving;
    }

    /**
     * @brief One sample of one channel, in double, with the coefficients as
     *        they are (a ramp moves on with advanceRamp()).
     * @pre channel in [0, MaxChannels).
     */
    [[nodiscard]] double processSample(double input, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels && "Channel index out of bounds");
        return advance(input, current_, state_[static_cast<size_t>(channel)]);
    }

    /** @brief Moves a ramp on by a number of samples (for processSample() callers). */
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
        const double n = static_cast<double>(samples);
        for (int m = 0; m <= Order; ++m)
        {
            const auto k = static_cast<size_t>(m);
            current_.c[k] += step_.c[k] * n;
            current_.s[k] += step_.s[k] * n;
            current_.v[k] += step_.v[k] * n;
        }
    }

private:
    using State = std::array<double, Order + 1>;   ///< x_1 .. x_N (index 0 unused).

    /** @brief The rotations, top down, the taps and the memory for one sample. */
    static double advance(double u, const Coeffs& c, State& x) noexcept
    {
        std::array<double, Order + 1> g {};
        double f = u;
        for (int m = Order; m >= 1; --m)
        {
            const auto k = static_cast<size_t>(m);
            const double xm = x[k];
            g[k] = c.s[k] * f + c.c[k] * xm;
            f = c.c[k] * f - c.s[k] * xm;
        }
        double y = c.v[0] * f;
        for (int m = 1; m <= Order; ++m) y += c.v[static_cast<size_t>(m)] * g[static_cast<size_t>(m)];
        for (int m = Order; m >= 2; --m) x[static_cast<size_t>(m)] = g[static_cast<size_t>(m - 1)];
        x[1] = f;
        return y;
    }

    static constexpr bool kPairChannels = simd::kVecDoubleNarrow == 2 && sizeof(void*) == 8;

    /**
     * @brief A block of one channel (W = 1) or of a pair (the two lanes),
     *        `moving` samples of them with the ramp's steps first.
     */
    template <int W>
    void runLanes(T* left, T* right, int numSamples, int moving, Coeffs& c, State& memLeft, State& memRight) noexcept
    {
        using O = simd::Vec<double, W>;
        using V = typename O::V;
        const auto lanes = [](double a, double b) noexcept -> V {
            if constexpr (W == 1) { (void)b; return a; }
            else return O::set2(a, b);
        };
        std::array<V, Order + 1> x {};
        for (int m = 1; m <= Order; ++m)
            x[static_cast<size_t>(m)] = lanes(memLeft[static_cast<size_t>(m)], memRight[static_cast<size_t>(m)]);

        std::array<V, Order + 1> vc {}, vs {}, vv {};
        const auto broadcast = [&]() noexcept {
            for (int m = 0; m <= Order; ++m)
            {
                const auto k = static_cast<size_t>(m);
                vc[k] = O::set1(c.c[k]);
                vs[k] = O::set1(c.s[k]);
                vv[k] = O::set1(c.v[k]);
            }
        };
        const auto sample = [&](int i) noexcept {
            V f;
            if constexpr (W == 1) f = static_cast<double>(left[i]);
            else f = lanes(static_cast<double>(left[i]), static_cast<double>(right[i]));
            std::array<V, Order + 1> g {};
            for (int m = Order; m >= 1; --m)
            {
                const auto k = static_cast<size_t>(m);
                const V xm = x[k];
                g[k] = O::mulAdd(vs[k], f, vc[k], xm);
                f = O::mulSub(vc[k], f, vs[k], xm);
            }
            V y = O::mul(vv[0], f);
            for (int m = 1; m <= Order; ++m) y = O::madd(vv[static_cast<size_t>(m)], g[static_cast<size_t>(m)], y);
            for (int m = Order; m >= 2; --m) x[static_cast<size_t>(m)] = g[static_cast<size_t>(m - 1)];
            x[1] = f;
            if constexpr (W == 1) left[i] = static_cast<T>(y);
            else
            {
                left[i] = static_cast<T>(O::lane0(y));
                right[i] = static_cast<T>(O::lane1(y));
            }
        };
        for (int i = 0; i < moving; ++i)
        {
            for (int m = 0; m <= Order; ++m)
            {
                const auto k = static_cast<size_t>(m);
                c.c[k] += step_.c[k];
                c.s[k] += step_.s[k];
                c.v[k] += step_.v[k];
            }
            broadcast();
            sample(i);
        }
        if (moving > 0 && moving == rampLeft_) c = target_;
        broadcast();
        for (int i = moving; i < numSamples; ++i) sample(i);

        double sum = 0.0;
        for (int m = 1; m <= Order; ++m)
        {
            const auto k = static_cast<size_t>(m);
            if constexpr (W == 1) memLeft[k] = x[k];
            else
            {
                memLeft[k] = O::lane0(x[k]);
                memRight[k] = O::lane1(x[k]);
            }
            sum += memLeft[k] + memRight[k];
        }
        if (!std::isfinite(sum))
        {
            memLeft = {};
            memRight = {};
        }
    }

    Coeffs current_ {}, target_ {}, step_ {};
    int rampLeft_ = 0;
    std::array<State, MaxChannels> state_ {};
};

} // namespace dspark
