// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Saturation.h
 * @brief Professional multi-algorithm saturation processor with analog simulation.
 *
 * This class provides a high-quality saturation pipeline featuring 10 distinct
 * algorithms ranging from soft clipping to complex magnetic tape and
 * transformer modeling. It oversamples 2x by default, so its latency is the
 * oversampler's group delay (see getLatency()); at 1x it is zero. Second-order
 * antialiasing (setAntialiasing(Antialiasing::SecondOrder), opt-in) adds one
 * sample, which the dry path matches.
 *
 * @details
 * **Architecture & Performance:**
 * - Real-Time Safe: no memory allocations or blocking calls in the process
 *   path (parameter recovery uses a bounded try-lock, never a wait).
 * - No virtual dispatch in the per-sample hot path: the algorithm is resolved
 *   statically once per block.
 * - Lock-free parameters: changes travel as complete snapshots through a
 *   single-producer/single-consumer queue.
 * - Per-channel state is bounded to 16 channels; channels beyond the prepared
 *   spec (or beyond 16) bypass the saturation stage.
 *
 * Dependencies: Core/AudioBuffer.h, Core/AudioSpec.h, Core/Biquad.h,
 * Core/DryWetMixer.h, Core/DspMath.h, Core/Oversampling.h, Core/Smoothers.h,
 * Core/AnalogRandom.h, Core/SpscQueue.h, Core/SpinLock.h, Core/StateBlob.h,
 * Core/detail/Adaa.h, Core/detail/LogCosh.h, Effects/DCBlocker.h, Effects/MidSide.h.
 *
 * Threading: prepare(), setOversampling() and setState() belong to the setup
 * thread; processBlock()/process() and reset() to the audio thread. The
 * remaining setters and getters may be called from any thread (snapshot
 * queue / relaxed atomics).
 *
 * @tparam SampleType The floating-point precision to use (must be `float` or `double`).
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/Oversampling.h"
#include "../Core/Smoothers.h"
#include "../Core/AnalogRandom.h"
#include "../Core/SpscQueue.h"
#include "../Core/SpinLock.h"
#include "../Core/StateBlob.h"
#include "DCBlocker.h"
#include "MidSide.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include "../Core/detail/Adaa.h"
#include "../Core/detail/LogCosh.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <type_traits>
#include <vector>

namespace dspark {

template <typename SampleType> class Saturation;

namespace detail {

template <typename T>
class SaturationAlgorithm
{
public:
    static constexpr int kAaCh = 16;

    virtual ~SaturationAlgorithm() = default;

    /** @brief Prepares the algorithm with the current audio specification. */
    virtual void prepare(const AudioSpec& spec) noexcept = 0;

    /** @brief Resets internal states (filters, phase, memory). */
    virtual void reset() noexcept = 0;

    /** @brief Updates internal coefficients dependent on block-rate parameters. */
    virtual void update(T /*driveGain*/, T /*character*/, const AudioSpec& /*spec*/) noexcept {}

    /** @brief Identifies the exact algorithm type for CRTP static dispatch. */
    virtual typename Saturation<T>::Algorithm getType() const noexcept = 0;

    /**
     * @brief Antiderivative antialiasing order on the memoryless curves:
     * 0 off, 1 first order (SoftClip/Tube/HardClip), 2 second order
     * (SoftClip/Tube/HardClip/Wavefolder/Transformer/Exciter/Tape and every
     * MultiStage stage from 2x, Bilbao et al. 2017). Order 2 delays every
     * algorithm by the same number of processing samples (latchOrder()):
     * the kernels delay by one and are padded to it, the others are delayed
     * by all of it, so a crossfade between any two stays aligned. Default 0.
     */
    void setAntialias(int order) noexcept { antialias_.store(std::clamp(order, 0, 2), std::memory_order_relaxed); }

    /**
     * @brief Latches the order for the coming block (audio thread, before
     * update()) and the common second-order delay in processing samples
     * (1 to kMaxDelay; Saturation passes min(factor, 3)). Entering order 2
     * arms every channel: its first sample seeds the kernel history and the
     * delay line with the current input, so the delay is inserted as held
     * samples instead of a step from stale history. The off and first-order
     * paths keep no second-order history at all.
     */
    void latchOrder(int commonDelay = 1) noexcept
    {
        const int o = antialias_.load(std::memory_order_relaxed);
        if (o == 2 && order_ != 2) { armMask_ = ~0u; delayArm_ = ~0u; }
        order_ = o;
        delay_ = std::clamp(commonDelay, 1, kMaxDelay);
    }

    /** @brief The latched order (audio thread). */
    [[nodiscard]] int order() const noexcept { return order_; }

    /** @brief The latched common second-order delay (processing samples). */
    [[nodiscard]] int commonDelay() const noexcept { return delay_; }

    /** Full common delay, for algorithms without a second-order kernel. */
    T align(T y, int ch) noexcept { return delayBy(y, ch, delay_); }

    /** The common delay less the one sample of a second-order kernel. */
    T padKernel(T y, int ch) noexcept { return delay_ > 1 ? delayBy(y, ch, delay_ - 1) : y; }

    static constexpr int kMaxDelay = 3;

protected:
    std::atomic<int> antialias_ { 0 };
    int order_ = 0;            ///< Latched order (audio thread).
    int delay_ = 1;            ///< Latched common second-order delay.
    unsigned armMask_ = ~0u;   ///< Channels whose order-2 history must be seeded.
    unsigned delayArm_ = ~0u;  ///< Channels whose delay line must be seeded.

    /** True once per channel after entering order 2 (or a reset). */
    bool takeArm(int c) noexcept
    {
        const unsigned bit = 1u << (c & (kAaCh - 1));
        if ((armMask_ & bit) == 0u) return false;
        armMask_ &= ~bit;
        return true;
    }

    /** Delays y by n (1 to kMaxDelay) samples on channel ch. */
    T delayBy(T y, int ch, int n) noexcept
    {
        const auto c = static_cast<size_t>(ch & (kAaCh - 1));
        auto& line = alignHold_[c];
        const unsigned bit = 1u << c;
        if (delayArm_ & bit) [[unlikely]]
        {
            delayArm_ &= ~bit;
            line.fill(y);
        }
        const T out = line[static_cast<size_t>(n - 1)];
        for (int k = kMaxDelay - 1; k > 0; --k) line[static_cast<size_t>(k)] = line[static_cast<size_t>(k - 1)];
        line[0] = y;
        return out;
    }

    void resetAlign() noexcept
    {
        for (auto& line : alignHold_) line.fill(T(0));
        armMask_ = ~0u;
        delayArm_ = ~0u;
    }
    std::array<std::array<T, kMaxDelay>, kAaCh> alignHold_ {};
};

// -- SoftClip (tanh) ---------------------------------------------------------
// The ADAA difference quotients below run in double whatever T is: the
// antiderivative is O(1) while successive inputs can differ by 1e-5, so in
// float the quotient kept only a few correct bits (measured on 20-200 Hz
// tones: errors up to -48 dB re peak, spiking at every waveform crest).
template <typename T>
class TanhAlgorithm final : public SaturationAlgorithm<T>
{
    std::array<double, SaturationAlgorithm<T>::kAaCh> prevX_ {};
    std::array<detail::Adaa2, SaturationAlgorithm<T>::kAaCh> adaa2_ {};
    std::array<double, SaturationAlgorithm<T>::kAaCh> prevBias_ {};
public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    void reset() noexcept override
    {
        prevX_.fill(0.0);
        for (auto& a : adaa2_) a.reset();
        prevBias_.fill(0.0);
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::SoftClip; }

    /** Second-order sample (Saturation::processCore runs it in its own loop
     *  when order() is 2, so the off and first-order loops are unchanged). */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const T x    = sample * drive;
        const T bias = character * T(0.3);
        const int c  = ch & (SaturationAlgorithm<T>::kAaCh - 1);
        // Second order on u = x + bias; the resting offset tanh(bias)
        // is taken one sample back, where the output is centred.
        const double u = static_cast<double>(x) + static_cast<double>(bias);
        if (this->takeArm(c)) [[unlikely]]
        {
            adaa2_[static_cast<size_t>(c)].reset(u);
            prevBias_[static_cast<size_t>(c)] = static_cast<double>(bias);
        }
        const double y = adaa2_[static_cast<size_t>(c)].template process<detail::AdaaTanh>(u);
        const double tb = std::tanh(prevBias_[static_cast<size_t>(c)]);
        prevBias_[static_cast<size_t>(c)] = static_cast<double>(bias);
        prevX_[static_cast<size_t>(c)] = static_cast<double>(x);
        return static_cast<T>(y - tb);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        T x    = sample * drive;
        T bias = character * T(0.3);
        const int c = ch & (SaturationAlgorithm<T>::kAaCh - 1);
        const int order = this->order_;
        if (order == 0)
        {
            // Keep the ADAA history current, so enabling ADAA mid-stream
            // starts from the real previous input (a stale one glitched).
            prevX_[static_cast<size_t>(c)] = static_cast<double>(x);
            return fastTanh(x + bias) - fastTanh(bias);
        }

        // ADAA of f(x) = tanh(x+bias) - tanh(bias); antiderivative log(cosh(x+bias)) - tanh(bias)*x.
        const double xd = static_cast<double>(x);
        const double b  = static_cast<double>(bias);
        const double x0 = prevX_[static_cast<size_t>(c)];
        prevX_[static_cast<size_t>(c)] = xd;
        const double tb = std::tanh(b);
        const double dx = xd - x0;
        if (std::abs(dx) > 1e-7)
            return static_cast<T>((logCosh(xd + b) - logCosh(x0 + b)) / dx - tb);
        return static_cast<T>(std::tanh(0.5 * (xd + x0) + b) - tb);
    }
};

// -- Tube (12AX7-style asymmetric triode model) -----------------------------
// A biased tanh, DC-free at rest and normalised to unit small-signal slope:
// f(x) = (tanh(x + b) - tanh(b)) / (1 - tanh(b)^2). The bias moves the
// operating point, so the curve is smooth everywhere and its even harmonics
// grow with level as a triode's do. (The previous form switched between
// tanh(x) and tanh(1.15 x) at x = 0: a slope kink that left a 2nd harmonic of
// -30.6 dBc at every level down to -60 dBFS and aliased at -42.7 dBc at 2x.)
// b = -0.7 * (asym - 1), asym = 1.15 + 0.5 * character, matches the former
// 2nd harmonic at -6 dBFS; negative excursions still compress first.
template <typename T>
class TubeAlgorithm final : public SaturationAlgorithm<T>
{
    std::array<double, SaturationAlgorithm<T>::kAaCh> prevX_ {};
    std::array<detail::Adaa2, SaturationAlgorithm<T>::kAaCh> adaa2_ {};
    std::array<double, SaturationAlgorithm<T>::kAaCh> prevBias_ {};

public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    void reset() noexcept override
    {
        prevX_.fill(0.0);
        for (auto& a : adaa2_) a.reset();
        prevBias_.fill(0.0);
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Tube; }

    /** Second-order sample (Saturation::processCore runs it in its own loop
     *  when order() is 2, so the off and first-order loops are unchanged). */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const T x = sample * drive;
        const T b = T(-0.7) * (T(0.15) + character * T(0.5));
        const int c = ch & (SaturationAlgorithm<T>::kAaCh - 1);
        // Second order on u = x + b; offset and slope normalisation use
        // the bias one sample back, where the output is centred.
        const double u = static_cast<double>(x) + static_cast<double>(b);
        if (this->takeArm(c)) [[unlikely]]
        {
            adaa2_[static_cast<size_t>(c)].reset(u);
            prevBias_[static_cast<size_t>(c)] = static_cast<double>(b);
        }
        const double y = adaa2_[static_cast<size_t>(c)].template process<detail::AdaaTanh>(u);
        const double tb = std::tanh(prevBias_[static_cast<size_t>(c)]);
        prevBias_[static_cast<size_t>(c)] = static_cast<double>(b);
        prevX_[static_cast<size_t>(c)] = static_cast<double>(x);
        return static_cast<T>((y - tb) / (1.0 - tb * tb));
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        const T x = sample * drive;
        const T b = T(-0.7) * (T(0.15) + character * T(0.5));
        const int c = ch & (SaturationAlgorithm<T>::kAaCh - 1);
        const int order = this->order_;
        if (order == 0)
        {
            prevX_[static_cast<size_t>(c)] = static_cast<double>(x);
            const T tb = fastTanh(b);
            return (fastTanh(x + b) - tb) / (T(1) - tb * tb);
        }

        // ADAA: antiderivative (log(cosh(x + b)) - tanh(b) * x) / (1 - tanh(b)^2).
        const double xd = static_cast<double>(x);
        const double bd = static_cast<double>(b);
        const double tb = std::tanh(bd);
        const double norm = 1.0 / (1.0 - tb * tb);
        const double x0 = prevX_[static_cast<size_t>(c)];
        prevX_[static_cast<size_t>(c)] = xd;
        const double dx = xd - x0;
        if (std::abs(dx) > 1e-7)
            return static_cast<T>(((logCosh(xd + bd) - logCosh(x0 + bd)) / dx - tb) * norm);
        return static_cast<T>((std::tanh(0.5 * (xd + x0) + bd) - tb) * norm);
    }
};

// -- HardClip ----------------------------------------------------------------
// First-order ADAA over every sample, in double. (A "linear body" shortcut
// used to pass unclipped pairs through un-averaged: it switched between a 0
// and a 0.5 sample delay at every clip edge, and that made the antialiased
// path alias MORE than the plain one, e.g. -51.6 vs -135.2 dBc at 2x, +6 dB.)
template <typename T>
class HardClipAlgorithm final : public SaturationAlgorithm<T>
{
    std::array<double, SaturationAlgorithm<T>::kAaCh> prevX_ {};
    std::array<detail::Adaa2, SaturationAlgorithm<T>::kAaCh> adaa2_ {};
    std::array<double, SaturationAlgorithm<T>::kAaCh> prevBias_ {};

    /** Antiderivative of clamp(u,-1,1): u^2/2 inside, |u|-1/2 outside. */
    static inline double g(double u) noexcept
    {
        const double a = std::abs(u);
        return (a <= 1.0) ? 0.5 * u * u : a - 0.5;
    }
public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    void reset() noexcept override
    {
        prevX_.fill(0.0);
        for (auto& a : adaa2_) a.reset();
        prevBias_.fill(0.0);
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::HardClip; }

    /** Second-order sample (Saturation::processCore runs it in its own loop
     *  when order() is 2, so the off and first-order loops are unchanged). */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const T bias = character * T(0.3);
        const T d = sample * drive;
        const int c = ch & (SaturationAlgorithm<T>::kAaCh - 1);
        // Exact second-order kernel (Core/detail/Adaa.h) on u = d + bias.
        const double u = static_cast<double>(d) + static_cast<double>(bias);
        if (this->takeArm(c)) [[unlikely]]
        {
            adaa2_[static_cast<size_t>(c)].reset(u);
            prevBias_[static_cast<size_t>(c)] = static_cast<double>(bias);
        }
        const double y = adaa2_[static_cast<size_t>(c)].template process<detail::AdaaHardClip>(u);
        const double cbPrev = std::clamp(prevBias_[static_cast<size_t>(c)], -1.0, 1.0);
        prevBias_[static_cast<size_t>(c)] = static_cast<double>(bias);
        prevX_[static_cast<size_t>(c)] = static_cast<double>(d);
        return static_cast<T>(y - cbPrev);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        const T bias = character * T(0.3);
        const T cb = std::clamp(bias, T(-1), T(1));
        const T d = sample * drive;
        const int c = ch & (SaturationAlgorithm<T>::kAaCh - 1);
        const int order = this->order_;
        if (order == 0)
        {
            prevX_[static_cast<size_t>(c)] = static_cast<double>(d);
            return std::clamp(d + bias, T(-1), T(1)) - cb;
        }

        const double dd = static_cast<double>(d), b = static_cast<double>(bias);
        const double d0 = prevX_[static_cast<size_t>(c)];
        prevX_[static_cast<size_t>(c)] = dd;
        const double u = dd + b, u0 = d0 + b, diff = dd - d0;
        if (std::abs(diff) > 1e-7)
            return static_cast<T>((g(u) - g(u0)) / diff - static_cast<double>(cb));
        return static_cast<T>(std::clamp(0.5 * (u + u0), -1.0, 1.0) - static_cast<double>(cb));
    }
};

// -- Exciter (polynomial waveshaper) -----------------------------------------

/**
 * Exact second-order kernel of the Exciter curve f(u) = clamp(p(u), -1, 1),
 * p(u) = u + A u^2 - B u^3 on |u| <= 10 and constant beyond (the plain path
 * clamps u to +-10 first). f is piecewise polynomial, so the average over the
 * linear B-spline with knots at three inputs (Bilbao et al. 2017) has a
 * closed form: the interval between the outer knots is split at the middle
 * knot, at +-10 and where p crosses +-1 (bracketed Newton on the monotone
 * pieces of p between its two critical points), and each piece contributes
 * its exact moment against the linear weight, written about the piece centre
 * so that no difference of large terms appears. When no crossing can lie
 * between the knots (the extremes of p over the interval stay inside
 * [-1, 1]) the average is the closed form of the cubic in the complete
 * symmetric polynomials of the knots.
 */
struct AdaaExciter
{
    static constexpr double kB = 0.15, kLimit = 10.0;

    double A = 0.0;                          ///< Quadratic coefficient (0.25 character).
    double crit[2] = { 0.0, 0.0 };           ///< Critical points of p, ascending.
    double critValue[2] = { 0.0, 0.0 };

    void setCharacter(double a) noexcept
    {
        if (a == A && critValue[1] != 0.0) return;
        A = a;
        const double r = std::sqrt(A * A + 3.0 * kB);
        crit[0] = (A - r) / (3.0 * kB);
        crit[1] = (A + r) / (3.0 * kB);
        critValue[0] = p(crit[0]);
        critValue[1] = p(crit[1]);
    }

    [[nodiscard]] double p(double u) const noexcept { return u * (1.0 + u * (A - kB * u)); }
    [[nodiscard]] double f(double u) const noexcept
    {
        return std::clamp(p(std::clamp(u, -kLimit, kLimit)), -1.0, 1.0);
    }

    /** Average of f over the linear B-spline with knots u0, u1, u2. */
    [[nodiscard]] double average(double u0, double u1, double u2) const noexcept
    {
        double lo = u0, mid = u1, hi = u2;
        if (lo > mid) std::swap(lo, mid);
        if (mid > hi) std::swap(mid, hi);
        if (lo > mid) std::swap(lo, mid);
        const double span = hi - lo;
        if (!(span > 1e-9)) return f((u0 + u1 + u2) * (1.0 / 3.0));
        if (lo >= -kLimit && hi <= kLimit)
        {
            double top = std::max(p(lo), p(hi)), bottom = std::min(p(lo), p(hi));
            for (int k = 0; k < 2; ++k)
                if (crit[k] > lo && crit[k] < hi)
                {
                    top = std::max(top, critValue[k]);
                    bottom = std::min(bottom, critValue[k]);
                }
            if (top <= 1.0 && bottom >= -1.0)
            {
                // E[u^k] = 2 h_k / ((k + 1)(k + 2)) for the complete symmetric h_k.
                const double h1 = u0 + u1 + u2;
                const double h2 = u0 * u0 + u1 * u1 + u2 * u2 + u0 * u1 + u1 * u2 + u0 * u2;
                const double h3 = u0 * u0 * u0 + u1 * u1 * u1 + u2 * u2 * u2
                                + u0 * u0 * (u1 + u2) + u1 * u1 * (u0 + u2) + u2 * u2 * (u0 + u1)
                                + u0 * u1 * u2;
                return h1 * (1.0 / 3.0) + A * h2 * (1.0 / 6.0) - kB * h3 * (1.0 / 10.0);
            }
            if (bottom >= 1.0) return 1.0;
            if (top <= -1.0) return -1.0;
        }

        // General case: break points inside (lo, hi), at most 1 + 2 + 6.
        double cuts[12];
        int n = 0;
        cuts[n++] = lo;
        cuts[n++] = mid;
        for (const double edge : { -kLimit, kLimit })
            if (edge > lo && edge < hi) cuts[n++] = edge;
        const double a = std::max(lo, -kLimit), b = std::min(hi, kLimit);
        if (a < b)
        {
            double piece[4] = { a, 0.0, 0.0, 0.0 };
            int pieces = 1;
            for (int k = 0; k < 2; ++k)
                if (crit[k] > a && crit[k] < b) piece[pieces++] = crit[k];
            piece[pieces] = b;
            for (int k = 0; k < pieces; ++k)
                for (const double level : { -1.0, 1.0 })
                {
                    double s = piece[k], e = piece[k + 1];
                    double fs = p(s) - level, fe = p(e) - level;
                    if (!(fs * fe < 0.0)) continue;
                    double t = s + (e - s) * fs / (fs - fe);
                    for (int it = 0; it < 60; ++it)
                    {
                        const double ft = p(t) - level;
                        if (ft == 0.0) break;
                        if ((ft < 0.0) == (fs < 0.0)) { s = t; fs = ft; } else { e = t; fe = ft; }
                        const double slope = 1.0 + t * (2.0 * A - 3.0 * kB * t);
                        double next = t - ft / slope;
                        if (!(next > s && next < e)) next = 0.5 * (s + e);
                        if (std::abs(next - t) <= 1e-15 * (1.0 + std::abs(t))) { t = next; break; }
                        t = next;
                    }
                    if (t > lo && t < hi) cuts[n++] = t;
                }
        }
        cuts[n++] = hi;
        for (int k = 1; k < n; ++k)   // insertion sort: at most 11 points
            for (int j = k; j > 0 && cuts[j - 1] > cuts[j]; --j) std::swap(cuts[j - 1], cuts[j]);

        // Linear weight: rising on [lo, mid], falling on [mid, hi]; a side
        // narrower than 1e-12 of the span carries no measurable mass.
        const bool rising = mid - lo > 1e-12 * span, falling = hi - mid > 1e-12 * span;
        const double riseSlope = rising ? 2.0 / (span * (mid - lo)) : 0.0;
        const double fallSlope = falling ? -2.0 / (span * (hi - mid)) : 0.0;
        double sum = 0.0;
        for (int k = 0; k + 1 < n; ++k)
        {
            const double s = cuts[k], e = cuts[k + 1];
            if (!(e > s)) continue;
            const double c = 0.5 * (s + e), hh = 0.5 * (e - s);
            const bool up = c < mid;
            if ((up && !rising) || (!up && !falling)) continue;
            const double w0 = up ? riseSlope * (c - lo) : -fallSlope * (hi - c);
            const double w1 = up ? riseSlope : fallSlope;
            double g0, g1 = 0.0, g2 = 0.0, g3 = 0.0;
            if (std::abs(c) > kLimit) g0 = f(c);
            else
            {
                g0 = p(c);
                if (g0 > 1.0 || g0 < -1.0) g0 = g0 > 1.0 ? 1.0 : -1.0;
                else
                {
                    g1 = 1.0 + c * (2.0 * A - 3.0 * kB * c);
                    g2 = A - 3.0 * kB * c;
                    g3 = -kB;
                }
            }
            const double h3 = hh * hh * hh;
            sum += 2.0 * hh * g0 * w0 + (2.0 / 3.0) * h3 * (g2 * w0 + g1 * w1)
                 + 0.4 * h3 * hh * hh * g3 * w1;
        }
        return sum;
    }
};

template <typename T>
class ExciterAlgorithm final : public SaturationAlgorithm<T>
{
    std::array<double, SaturationAlgorithm<T>::kAaCh> u1_ {}, u2_ {};
    AdaaExciter curve_;

public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Exciter; }

    void reset() noexcept override
    {
        u1_.fill(0.0);
        u2_.fill(0.0);
        this->resetAlign();
    }

    /** Second-order sample: the exact kernel of AdaaExciter on u = in * drive
     *  (one sample of delay; the curve has no resting offset). */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const double u = static_cast<double>(sample) * static_cast<double>(drive);
        const auto c = static_cast<size_t>(ch & (SaturationAlgorithm<T>::kAaCh - 1));
        if (this->takeArm(static_cast<int>(c))) [[unlikely]] u1_[c] = u2_[c] = u;
        curve_.setCharacter(0.25 * static_cast<double>(character));
        const double y = curve_.average(u, u1_[c], u2_[c]);
        u2_[c] = u1_[c];
        u1_[c] = u;
        return static_cast<T>(y);
    }

    inline T processSample(T sample, T drive, T character, int) noexcept
    {
        T x      = std::clamp(sample * drive, T(-10), T(10));
        T x2     = x * x;
        T x3     = x2 * x;
        T result = x + character * T(0.25) * x2 - T(0.15) * x3;
        return std::clamp(result, T(-1), T(1));
    }
};

// -- Wavefolder (sin + first-order ADAA) -------------------------------------
template <typename T>
class WavefolderAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<T, kMaxCh> lastX_ {};
    std::array<detail::Adaa2, kMaxCh> adaa2_ {};
    std::array<double, kMaxCh> prevBias_ {};

public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    void reset() noexcept override
    {
        lastX_.fill(T(0));
        for (auto& a : adaa2_) a.reset();
        prevBias_.fill(0.0);
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Wavefolder; }

    /**
     * Folder curve sin(in*drive + bias) - sin(bias): unit small-signal gain
     * over in*drive and a +-1 ceiling like every other algorithm in the
     * pool. At neutral drive the curve never reaches the first peak
     * (|x| <= 1 < pi/2), i.e. gentle tanh-like rounding; folds appear once
     * in*drive crosses pi/2 and multiply with drive. The previous scaling,
     * sin(in*drive*2pi), had a 2pi (+16 dB) hidden gain - +56 dB onto small
     * signals at full drive - and folded fully at -6 dBFS even at neutral
     * drive. character biases the fold point (asymmetry / even harmonics);
     * subtracting sin(bias) keeps the resting output DC-free.
     */
    /** Second-order sample (Saturation::processCore runs it in its own loop
     *  when order() is 2, so the off and first-order loops are unchanged). */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const T bias = character * (pi<T> / T(4));
        const T x    = sample * drive + bias;
        const int c  = ch & (kMaxCh - 1);
        // Second order on sin (antiderivatives -cos, -sin); the resting
        // offset sin(bias) is taken one sample back, at the centre.
        if (this->takeArm(c)) [[unlikely]]
        {
            adaa2_[static_cast<size_t>(c)].reset(static_cast<double>(x));
            prevBias_[static_cast<size_t>(c)] = static_cast<double>(bias);
        }
        const double y = adaa2_[static_cast<size_t>(c)].template process<detail::AdaaSine>(static_cast<double>(x));
        const double sbPrev = std::sin(prevBias_[static_cast<size_t>(c)]);
        prevBias_[static_cast<size_t>(c)] = static_cast<double>(bias);
        lastX_[static_cast<size_t>(c)] = x;
        return static_cast<T>(y - sbPrev);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        const T bias = character * (pi<T> / T(4));
        const T x    = sample * drive + bias;
        const int c  = ch & (kMaxCh - 1);
        const T sb   = std::sin(bias);

        // Mean of sin over [x0, x], (cos x0 - cos x) / (x - x0), written as
        // sin(mid) * sinc(half) via the sum-to-product identity: exact, with
        // no cancellation (the difference of cosines lost most of its digits
        // in float whenever consecutive inputs were close).
        const T half = T(0.5) * (x - lastX_[static_cast<size_t>(c)]);
        const T mid  = T(0.5) * (x + lastX_[static_cast<size_t>(c)]);
        const T sinc = (std::abs(half) > T(1e-4)) ? std::sin(half) / half
                                                  : T(1) - half * half / T(6);
        lastX_[static_cast<size_t>(c)] = x;
        return std::sin(mid) * sinc - sb;
    }
};

// -- Bitcrusher (TPDF dither) ------------------------------------------------
template <typename T>
class BitcrusherAlgorithm final : public SaturationAlgorithm<T>
{
    // Per-sample TPDF dither needs a true white PRNG. (The previous
    // AnalogRandom-based source was a 1 Hz sample-and-hold: two consecutive
    // reads were almost always identical, so the dither was effectively zero
    // and the crusher truncated with audible quantisation distortion.)
    uint32_t rngState_ = 0x9E3779B9u;
    T steps_ = T(1);
    T invSteps_ = T(1);

    [[nodiscard]] inline T nextRandom() noexcept
    {
        rngState_ ^= rngState_ << 13;
        rngState_ ^= rngState_ >> 17;
        rngState_ ^= rngState_ << 5;
        constexpr T scale = T(1) / static_cast<T>(0xFFFFFFFFu);
        return static_cast<T>(rngState_) * scale - T(0.5);
    }

public:
    void prepare(const AudioSpec&) noexcept override
    {
        if (rngState_ == 0) rngState_ = 1; // xorshift fixed point guard
    }
    void reset() noexcept override { this->resetAlign(); }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Bitcrusher; }

    void update(T drive, T /*character*/, const AudioSpec&) noexcept override
    {
        T clamped  = std::clamp(drive, T(1), T(100));
        T bitDepth = mapRange(clamped, T(1), T(100), T(16), T(2));
        steps_     = std::pow(T(2), bitDepth);
        invSteps_  = T(1) / steps_;
    }

    inline T processSample(T sample, T, T, int) noexcept
    {
        // True TPDF: difference of two independent uniforms, +-1 LSB peak.
        T dither = (nextRandom() - nextRandom()) * invSteps_;
        return invSteps_ * std::round((sample + dither) * steps_);
    }
};

// -- Tape second-order kernel -------------------------------------------------

/**
 * Antiderivatives of the anhysteretic tape curve, for its second-order
 * kernel. With x the solution of H = a x - rho L(x) (a = 1/(3(1 - alpha)),
 * rho = alpha/(1 - alpha); see TapeAlgorithm) the curve output is L(x), and
 * with G = ln(sinh x / x) (dG/dx = L), P = int_0^x G and Q = int_0^x L^2,
 *
 *   F1(H) = a G - rho L^2 / 2,
 *   F2(H) = a^2 P - a rho G L + (a rho / 2) Q + rho^2 L^3 / 6,
 *
 * found by integrating in x (dH = (a - rho dL/dx) dx) and by parts. F2 is not
 * elementary: P(x) = x (1 + G) - x^2/2 - pi^2/12 + A(x) and
 * Q(x) = x - L - 2 R(x), R(x) = int_0^x L(t)/t dt = (x - G) - ln 2 + 1/x
 * + C + B(x), where A = Li2(e)/2 - x ln(1 - e) and B = ln(1 - e)
 * - 2 sum_k E1(2 k x) (e = exp(-2x)) are smooth and exponentially small. A and
 * B are tabulated once per process as degree-11 Chebyshev pieces, eight per
 * octave on [0.5, 32), from their series (dilogarithm, exponential integral
 * by its continued fraction) and are zero beyond 32 (below 1e-26); below 0.5
 * P, Q, L, its slope and G come from their Taylor series. P and Q stay
 * within 5e-15 (relative, floor 1) of a compensated Gauss quadrature of their
 * integrands.
 */
class AdaaTapeTable
{
public:
    /** Shared instance, built on first use (setup thread: TapeAlgorithm::prepare()). */
    static const AdaaTapeTable& instance()
    {
        static const AdaaTapeTable table;
        return table;
    }

    /** L and its slope for x >= 0. */
    static void langevin(double x, double& L, double& Lp) noexcept
    {
        if (x < kStart)
        {
            const double u = x * x;
            double l = kL[12], lp = kLp[12];
            for (int n = 11; n >= 0; --n) { l = l * u + kL[n]; lp = lp * u + kLp[n]; }
            L = x * l;
            Lp = lp;
            return;
        }
        const double e = std::exp(-2.0 * x), om = 1.0 - e;
        L = (1.0 + e) / om - 1.0 / x;
        Lp = 1.0 / (x * x) - 4.0 * e / (om * om);
    }

    /** G = ln(sinh x / x), P and Q for x >= 0, given L(x). */
    void antiderivatives(double x, double L, double& G, double& P, double& Q) const noexcept
    {
        if (x < kStart)
        {
            const double u = x * x, x3 = u * x;
            double g = kG[12], pp = kP[12], q = kQ[12];
            for (int n = 11; n >= 0; --n) { g = g * u + kG[n]; pp = pp * u + kP[n]; q = q * u + kQ[n]; }
            G = u * g;
            P = x3 * pp;
            Q = x3 * q;
            return;
        }
        G = x + std::log((1.0 - std::exp(-2.0 * x)) / (2.0 * x));
        double a = 0.0, b = 0.0;
        if (x < kEnd)
        {
            // Eight cells per octave: exponent and top three mantissa bits.
            const auto k = static_cast<size_t>((std::bit_cast<std::uint64_t>(x) >> 49) - kFirstCell);
            const double t = (x - mid_[k]) * invHalf_[k];
            a = clenshaw(a_[k], t);
            b = clenshaw(b_[k], t);
        }
        P = x * (1.0 + G) - 0.5 * x * x - kPiSquaredOver12 + a;
        Q = x - L - 2.0 * ((x - G) - kLn2 + 1.0 / x + cR_ + b);
    }

private:
    static constexpr int kDeg = 11, kCells = 48;
    static constexpr double kStart = 0.5, kEnd = 32.0;
    static constexpr std::uint64_t kFirstCell = 0x1FF0u;   // bits(0.5) >> 49
    static constexpr double kPiSquaredOver12 = 0.82246703342411321824;
    static constexpr double kLn2 = 0.69314718055994530942;
    // Taylor coefficients in u = x^2 (from the Bernoulli numbers): L = x sum
    // kL u^n, its slope sum kLp u^n, G = u sum kG u^n, P = x^3 sum kP u^n,
    // Q = x^3 sum kQ u^n. Thirteen terms reach rounding below x = 0.5.
    static constexpr double kL[13] = { 0.3333333333333333, -0.022222222222222223, 0.0021164021164021165,
        -0.00021164021164021165, 2.1377799155576935e-05, -2.1644042808063972e-06, 2.1925947851873778e-07,
        -2.2214608789979678e-08, 2.2507846516808994e-09, -2.2805151204592183e-10, 2.3106432599002624e-11,
        -2.3411706819824882e-12, 2.3721017400233653e-13 };
    static constexpr double kLp[13] = { 0.3333333333333333, -0.06666666666666667, 0.010582010582010581,
        -0.0014814814814814814, 0.0001924001924001924, -2.380844708887037e-05, 2.8503732207435913e-06,
        -3.332191318496952e-07, 3.8263339078575285e-08, -4.332978728872515e-09, 4.852350845790551e-10,
        -5.3846925685597234e-11, 5.930254350058414e-12 };
    static constexpr double kG[13] = { 0.16666666666666666, -0.005555555555555556, 0.0003527336860670194,
        -2.6455026455026456e-05, 2.1377799155576935e-06, -1.803670234005331e-07, 1.5661391322766983e-08,
        -1.3884130493737299e-09, 1.2504359176004997e-10, -1.1402575602296091e-11, 1.0502923908637557e-12,
        -9.754877841593701e-14, 9.123468230859098e-15 };
    static constexpr double kP[13] = { 0.05555555555555555, -0.0011111111111111111, 5.039052658100277e-05,
        -2.9394473838918285e-06, 1.9434362868706303e-07, -1.3874386415425623e-08, 1.0440927548511323e-09,
        -8.167135584551352e-11, 6.581241671581577e-12, -5.429797905855281e-13, 4.5664886559293725e-14,
        -3.9019511366374804e-15, 3.379062307725592e-16 };
    static constexpr double kQ[13] = { 0.037037037037037035, -0.002962962962962963, 0.00027210884353741496,
        -2.6128421190149586e-05, 2.5579323318621057e-06, -2.5299170598315896e-07, 2.5176556628643635e-08,
        -2.5155828459962992e-09, 2.520569343665452e-10, -2.5307045227479064e-11, 2.5447507412853134e-12,
        -2.5618698792252346e-13, 2.5814742395034203e-14 };

    using Row = std::array<double, kDeg + 1>;

    static double clenshaw(const Row& c, double t) noexcept
    {
        double b1 = 0.0, b2 = 0.0;
        const double t2 = 2.0 * t;
        for (int n = kDeg; n >= 1; --n)
        {
            const double b0 = c[static_cast<size_t>(n)] + t2 * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        return c[0] + t * b1 - b2;
    }

    /** E1(z) for z >= 1: modified Lentz evaluation of its continued fraction. */
    static double e1(double z) noexcept
    {
        double b = z + 1.0, c = 1e300, d = 1.0 / b, h = d;
        for (int i = 1; i < 300; ++i)
        {
            const double an = -static_cast<double>(i) * i;
            b += 2.0;
            d = 1.0 / (an * d + b);
            c = b + an / c;
            const double del = c * d;
            h *= del;
            if (std::abs(del - 1.0) < 4e-16) break;
        }
        return h * std::exp(-z);
    }

    /** T(x) = int_x^inf 2 / (t (e^2t - 1)) dt = 2 sum_k E1(2 k x), x >= 0.5. */
    static double tail(double x) noexcept
    {
        double t = 0.0;
        for (int k = 1; k < 400; ++k)
        {
            const double v = e1(2.0 * k * x);
            t += v;
            if (v < 1e-25) break;
        }
        return 2.0 * t;
    }

    AdaaTapeTable()
    {
        for (int k = 0; k < kCells; ++k)
        {
            const double lo = std::ldexp(1.0 + (k % 8) / 8.0, k / 8 - 1);
            const double hi = lo + std::ldexp(1.0 / 8.0, k / 8 - 1);
            mid_[static_cast<size_t>(k)] = 0.5 * (lo + hi);
            invHalf_[static_cast<size_t>(k)] = 2.0 / (hi - lo);
            double av[kDeg + 1], bv[kDeg + 1];
            for (int j = 0; j <= kDeg; ++j)
            {
                const double x = 0.5 * (lo + hi)
                               + 0.5 * (hi - lo) * std::cos(std::numbers::pi * (j + 0.5) / (kDeg + 1));
                const double e = std::exp(-2.0 * x);
                double li = 0.0, ek = e;
                for (int m = 1; m < 400 && ek > 1e-26; ++m, ek *= e)
                    li += ek / (static_cast<double>(m) * m);
                av[j] = 0.5 * li - x * std::log1p(-e);
                bv[j] = std::log1p(-e) - tail(x);
            }
            for (int n = 0; n <= kDeg; ++n)
            {
                double sa = 0.0, sb = 0.0;
                for (int j = 0; j <= kDeg; ++j)
                {
                    const double w = std::cos(std::numbers::pi * n * (j + 0.5) / (kDeg + 1));
                    sa += av[j] * w;
                    sb += bv[j] * w;
                }
                a_[static_cast<size_t>(k)][static_cast<size_t>(n)] = (n == 0 ? 1.0 : 2.0) * sa / (kDeg + 1);
                b_[static_cast<size_t>(k)][static_cast<size_t>(n)] = (n == 0 ? 1.0 : 2.0) * sb / (kDeg + 1);
            }
        }
        // C = R(0.5) - ln 0.5 - 2 + T(0.5), R(0.5) = sum kL[n] 0.5^(2n+1) / (2n+1).
        double r = 0.0;
        for (int n = 12; n >= 0; --n) r = r * 0.25 + kL[n] / (2 * n + 1);
        cR_ = 0.5 * r + kLn2 - 2.0 + tail(0.5);
    }

    std::array<Row, kCells> a_ {}, b_ {};
    std::array<double, kCells> mid_ {}, invHalf_ {};
    double cR_ = 0.0;
};

/**
 * Second-order kernel of the tape curve (one stream): the curve's average
 * over the linear B-spline with knots at the last three inputs, from F2
 * differences. Close inputs use the Euler-Maclaurin form of the divided
 * difference, (F1(a) + F1(b))/2 - (a - b)(f(a) - f(b))/12, and the confluent
 * and coincident forms of Core/detail/Adaa.h written with the same
 * per-sample data, so no solve is needed between samples. Within 1e-7 of a
 * brute-force Gauss quadrature of the curve over the same B-spline, for
 * random knots from 1e-7 to 50 apart. A change of the curve (character)
 * re-evaluates the two held inputs on the new curve.
 */
struct AdaaTape
{
    struct Point { double H = 0.0, x = 0.0, f = 0.0, fp = 0.0, F1 = 0.0, F2 = 0.0; };

    const AdaaTapeTable* table = nullptr;
    double alpha = -1.0, a = 1.0, rho = 0.0;   ///< Curve of the points held.
    double h1 = 0.0, h2 = 0.0;                 ///< Inputs one and two samples ago.
    Point p1, p2;

    /** Holds u as the whole history (arming). */
    void reset(double u, double alphaIn) noexcept
    {
        h1 = h2 = u;
        setCurve(alphaIn);
        p1 = point(u, 3.0 * u);
        p2 = p1;
    }

    /** The antialiased output for input H, centred one sample back. */
    double process(double H, double alphaIn) noexcept
    {
        if (alphaIn != alpha)
        {
            setCurve(alphaIn);
            p1 = point(h1, p1.x);
            p2 = point(h2, p2.x);
        }
        const Point p0 = point(H, p1.x);
        const double y = average(p0, p1, p2);
        p2 = p1;
        p1 = p0;
        h2 = h1;
        h1 = H;
        return y;
    }

    void setCurve(double alphaIn) noexcept
    {
        alpha = alphaIn;
        a = 1.0 / (3.0 * (1.0 - alpha));
        rho = alpha / (1.0 - alpha);
    }

    /** Solves H = a x - rho L(x) by Newton from the seed (slope in [1/3, a]). */
    [[nodiscard]] Point point(double H, double seed) const noexcept
    {
        const double s = H < 0.0 ? -1.0 : 1.0, h = std::abs(H);
        double x = std::max(0.0, s * seed), L = 0.0, Lp = 0.0;
        for (int it = 0; it < 40; ++it)
        {
            AdaaTapeTable::langevin(x, L, Lp);
            const double step = (a * x - rho * L - h) / (a - rho * Lp);
            x = std::max(0.0, x - step);
            if (!(std::abs(step) > 1e-15 * (1.0 + x))) break;
        }
        AdaaTapeTable::langevin(x, L, Lp);
        double G = 0.0, P = 0.0, Q = 0.0;
        table->antiderivatives(x, L, G, P, Q);
        Point p;
        p.H = H;
        p.x = s * x;
        p.f = s * L;
        p.fp = Lp / (a - rho * Lp);
        p.F1 = a * G - 0.5 * rho * L * L;
        p.F2 = s * (a * a * P - a * rho * G * L + 0.5 * a * rho * Q + rho * rho * L * L * L * (1.0 / 6.0));
        return p;
    }

    [[nodiscard]] static double div(const Point& p, const Point& q) noexcept
    {
        const double d = p.H - q.H;
        if (std::abs(d) > 3e-3) return (p.F2 - q.F2) / d;
        return 0.5 * (p.F1 + q.F1) - d * (p.f - q.f) * (1.0 / 12.0);
    }

    [[nodiscard]] static double average(const Point& p0, const Point& p1, const Point& p2) noexcept
    {
        const double d02 = p0.H - p2.H;
        if (std::abs(d02) > 1e-3) return 2.0 * (div(p0, p1) - div(p1, p2)) / d02;
        // p0 and p2 merge into their mean b: F1, F2 and f at b from both sides.
        const double dd = 0.125 * d02 * d02;
        Point b;
        b.H = 0.5 * (p0.H + p2.H);
        b.fp = 0.5 * (p0.fp + p2.fp);
        b.f = 0.5 * (p0.f + p2.f);
        b.F1 = 0.5 * (p0.F1 + p2.F1) - dd * b.fp;
        b.F2 = 0.5 * (p0.F2 + p2.F2) - dd * b.f;
        const double e = b.H - p1.H;
        if (std::abs(e) > 1e-4) return 2.0 * (b.F1 - div(b, p1)) / e;
        return p1.f + p1.fp * ((p0.H + p1.H + p2.H) * (1.0 / 3.0) - p1.H);
    }
};

// -- Tape (anhysteretic Langevin magnetisation + head bump + HF rolloff) -----
template <typename T>
class TapeAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<Biquad<T, 1>, kMaxCh> preFilters_;
    std::array<Biquad<T, 1>, kMaxCh> postFilters_;
    std::array<T, kMaxCh>            M_ {};   ///< Per-channel NR warm start.
    std::array<AdaaTape, kMaxCh>     kernel_ {};   ///< Second-order kernels (setup in prepare()).
    int numChannels_ = 0;
    T lastDrive_ = T(-1);
    double lastSampleRate_ = 0.0;

    /// Langevin function L(x) = coth(x) - 1/x. The Taylor series keeps float
    /// precision near zero, where the direct form cancels catastrophically
    /// (coth(x) and 1/x both ~1/x); the far asymptote avoids sinh overflow.
    static inline T langevin(T x) noexcept
    {
        const T ax = std::abs(x);
        if (ax < T(0.5))
        {
            const T x2 = x * x;
            return x * (T(1) / T(3) - x2 * (T(1) / T(45) - x2 * (T(2) / T(945))));
        }
        if (ax > T(20)) return std::copysign(T(1), x) - T(1) / x;
        return T(1) / std::tanh(x) - T(1) / x;
    }

    /// dL/dx = 1/x^2 - csch^2(x), with the matching series and asymptote.
    static inline T langevinDeriv(T x) noexcept
    {
        const T ax = std::abs(x);
        if (ax < T(0.5))
        {
            const T x2 = x * x;
            return T(1) / T(3) - x2 * (T(1) / T(15) - x2 * (T(2) / T(189)));
        }
        if (ax > T(20)) return T(1) / (x * x);
        const T s = std::sinh(x);
        return T(1) / (x * x) - T(1) / (s * s);
    }

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = std::min(spec.numChannels, kMaxCh); // clamp per-channel state
        for (auto& k : kernel_) k.table = &AdaaTapeTable::instance();
        // Invalidate the filter-design cache: a re-prepare can widen the
        // channel count at an unchanged rate/drive, and the new channels'
        // filters must still receive coefficients on the next update().
        lastDrive_ = T(-1);
        lastSampleRate_ = 0.0;
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : preFilters_)  f.reset();
        for (auto& f : postFilters_) f.reset();
        M_.fill(T(0));
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Tape; }

    void update(T drive, T /*character*/, const AudioSpec& spec) noexcept override
    {
        // Both filters depend only on drive and sample rate; skip the
        // trig-heavy redesign when neither changed since the last block.
        if (drive == lastDrive_ && spec.sampleRate == lastSampleRate_) return;
        lastDrive_ = drive;
        lastSampleRate_ = spec.sampleRate;

        auto driveDb = gainToDecibels(drive, T(-100));
        T bumpGain = T(1.5) + std::min(driveDb * T(0.05), T(3.0));
        auto peakCoeffs = BiquadCoeffs::makePeak(spec.sampleRate, 80.0, 0.6, static_cast<double>(bumpGain));
        auto lpFreq = std::max(6000.0, 19000.0 - static_cast<double>(driveDb) * 200.0);
        auto lpCoeffs = BiquadCoeffs::makeLowPass(spec.sampleRate, lpFreq, 0.55);

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            preFilters_[ch].setCoeffs(peakCoeffs);
            postFilters_[ch].setCoeffs(lpCoeffs);
        }
    }

    /**
     * Tape under correct AC bias magnetises along the ANHYSTERETIC curve
     * (the bias linearises the loop away) - so the static nonlinearity is
     * M = Ms * L((H + alpha*M)/a), solved per sample by Newton-Raphson from
     * the previous sample's M (warm start: 3 iterations are plenty, and
     * f' = 1 - 3*alpha*L' >= 1 - alpha > 0 has no singularity).
     *
     * Scaling: Ms = 3a and out = (1-alpha)*M give exactly unit small-signal
     * gain over H = in*drive and a +-1 ceiling, matching the contract of
     * every other algorithm in the pool. The previous formulation stepped M
     * by dM ~ ManDiff*|dH|/(a+|dH|) - a coercivity of ~1.0 in absolute
     * full-scale units, i.e. tape with NO bias: signals below ~0 dBFS*drive
     * fell into a dead zone (measured: -60 dBFS in -> -121 dBFS out, output
     * ~ input^2), wiping out quiet passages and low-level harmonic detail.
     */
    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        const T filtered = preFilters_[ch].processSample(sample, 0);
        const T H = filtered * drive;

        // character (clamped [-1,1] upstream) sets the mean-field coupling:
        // harder knee and more mid-level bloom as alpha rises.
        const T alpha = T(0.35) + T(0.15) * character;
        const T a     = T(1) / (T(3) * (T(1) - alpha));
        const T Ms    = T(3) * a;

        T M = M_[ch];
        for (int it = 0; it < 3; ++it)
        {
            const T x  = (H + alpha * M) / a;
            const T f  = M - Ms * langevin(x);
            const T fp = T(1) - T(3) * alpha * langevinDeriv(x);
            M -= f / fp;
        }
        M = std::clamp(M, -Ms, Ms);
        M_[ch] = M;

        return postFilters_[ch].processSample((T(1) - alpha) * M, 0);
    }

    /** Second-order sample: the AdaaTape kernel on the magnetisation curve
     *  between the same head-bump and roll-off filters (one sample of delay).
     *  The plain path's warm start follows, so switching back is seamless. */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const T filtered = preFilters_[ch].processSample(sample, 0);
        const double H = static_cast<double>(filtered) * static_cast<double>(drive);
        const T alpha = T(0.35) + T(0.15) * character;
        auto& k = kernel_[static_cast<size_t>(ch)];
        if (this->takeArm(ch)) [[unlikely]] k.reset(H, static_cast<double>(alpha));
        const double y = k.process(H, static_cast<double>(alpha));
        M_[ch] = static_cast<T>(k.p1.f / (1.0 - k.alpha));   // M = Ms L, Ms = 1 / (1 - alpha)
        return postFilters_[ch].processSample(static_cast<T>(y), 0);
    }
};

// -- Transformer (frequency-dependent: heavy LF, light HF) ------------------
template <typename T>
class TransformerAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<Biquad<T, 1>, kMaxCh> lpFilters_;
    std::array<detail::Adaa2, kMaxCh> adaaLow_ {}, adaaHigh_ {};
    std::array<double, kMaxCh> prevRestLow_ {}, prevRestHigh_ {};
    std::array<double, kMaxCh> firstLow_ {}, firstHigh_ {};   ///< First-order history (MultiStage).
    int numChannels_ = 0;
    double lastSampleRate_ = 0.0;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = std::min(spec.numChannels, kMaxCh); // clamp per-channel state
        lastSampleRate_ = 0.0; // force a redesign so channels added by a re-prepare get coefficients
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : lpFilters_) f.reset();
        for (auto& a : adaaLow_) a.reset();
        for (auto& a : adaaHigh_) a.reset();
        prevRestLow_.fill(0.0);
        prevRestHigh_.fill(0.0);
        firstLow_.fill(0.0);
        firstHigh_.fill(0.0);
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Transformer; }

    void update(T /*drive*/, T /*character*/, const AudioSpec& spec) noexcept override
    {
        // The crossover is fixed at 250 Hz: only recompute when the effective
        // sample rate changes (update() runs every block from the pipeline).
        if (spec.sampleRate == lastSampleRate_) return;
        lastSampleRate_ = spec.sampleRate;

        auto c = BiquadCoeffs::makeLowPass(spec.sampleRate, 250.0, 0.707);
        for (int ch = 0; ch < numChannels_; ++ch)
            lpFilters_[ch].setCoeffs(c);
    }

    /**
     * Band-dependent saturation THRESHOLDS at unity small-signal gain: core
     * flux scales like V/f, so lows reach saturation first (harder knee,
     * tanh(1.4x)/1.4, ceiling 0.71) while highs barely do (tanh(0.85x)/0.85,
     * ceiling 1.18). A real transformer's linear response is flat - the
     * previous form left the band factors unnormalised, which was a fixed
     * +2.3/-3.1 dB shelf tilt at 250 Hz that buried mids and highs (worst in
     * the MultiStage cascade), and its resting-bias subtraction ignored the
     * drive scaling, leaking drive-dependent DC.
     */
    /** Second-order sample (Saturation::processCore runs it in its own loop
     *  when order() is 2, so the off and first-order loops are unchanged). */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const T low  = lpFilters_[ch].processSample(sample, 0);
        const T high = sample - low;
        const T bias = character * T(0.2);
        const T kLo  = drive * T(1.4);
        const T kHi  = drive * T(0.85);
        const auto c = static_cast<size_t>(ch & (kMaxCh - 1));
        // Second order on each band's tanh input; resting offsets one
        // sample back, where the outputs are centred.
        const double uLo = static_cast<double>((low + bias) * kLo);
        const double uHi = static_cast<double>((high + bias) * kHi);
        if (this->takeArm(static_cast<int>(c))) [[unlikely]]
        {
            adaaLow_[c].reset(uLo);
            adaaHigh_[c].reset(uHi);
            prevRestLow_[c] = static_cast<double>(bias * kLo);
            prevRestHigh_[c] = static_cast<double>(bias * kHi);
        }
        const double yLo = adaaLow_[c].template process<detail::AdaaTanh>(uLo) - std::tanh(prevRestLow_[c]);
        const double yHi = adaaHigh_[c].template process<detail::AdaaTanh>(uHi) - std::tanh(prevRestHigh_[c]);
        prevRestLow_[c] = static_cast<double>(bias * kLo);
        prevRestHigh_[c] = static_cast<double>(bias * kHi);
        return static_cast<T>(yLo / 1.4 + yHi / 0.85);
    }

    /** First-order sample (first-order antiderivative of each band's tanh,
     *  half a sample of delay), for the MultiStage cascade at 2x: with the
     *  tube stage at first order and the tape kernel it keeps the cascade at
     *  two processing samples. The history is seeded when order 2 is armed. */
    inline T processFirst(T sample, T drive, T character, int ch) noexcept
    {
        const T low  = lpFilters_[ch].processSample(sample, 0);
        const T high = sample - low;
        const T bias = character * T(0.2);
        const T kLo  = drive * T(1.4);
        const T kHi  = drive * T(0.85);
        const auto c = static_cast<size_t>(ch & (kMaxCh - 1));
        const double uLo = static_cast<double>((low + bias) * kLo);
        const double uHi = static_cast<double>((high + bias) * kHi);
        if (this->takeArm(static_cast<int>(c))) [[unlikely]]
        {
            firstLow_[c] = uLo;
            firstHigh_[c] = uHi;
        }
        const auto mean = [](double u, double u0) noexcept {
            const double d = u - u0;
            return std::abs(d) > 1e-7 ? (logCosh(u) - logCosh(u0)) / d : std::tanh(0.5 * (u + u0));
        };
        const double yLo = mean(uLo, firstLow_[c]) - std::tanh(static_cast<double>(bias * kLo));
        const double yHi = mean(uHi, firstHigh_[c]) - std::tanh(static_cast<double>(bias * kHi));
        firstLow_[c] = uLo;
        firstHigh_[c] = uHi;
        return static_cast<T>(yLo / 1.4 + yHi / 0.85);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        const T low  = lpFilters_[ch].processSample(sample, 0);
        const T high = sample - low;
        const T bias = character * T(0.2);
        const T kLo  = drive * T(1.4);
        const T kHi  = drive * T(0.85);
        const T satLow  = (fastTanh((low  + bias) * kLo) - fastTanh(bias * kLo)) / T(1.4);
        const T satHigh = (fastTanh((high + bias) * kHi) - fastTanh(bias * kHi)) / T(0.85);
        return satLow + satHigh;
    }
};

// -- Downsample (sample rate reduction) --------------------------------------
template <typename T>
class DownsampleAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<Biquad<T, 1>, kMaxCh> aaFilters_;
    std::array<T, kMaxCh>            lastSample_ {};
    std::array<int, kMaxCh>          counter_    {};
    int numChannels_ = 0;
    int reduction_   = 1;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = std::min(spec.numChannels, kMaxCh); // clamp per-channel state
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : aaFilters_) f.reset();
        lastSample_.fill(T(0));
        counter_.fill(0);
        this->resetAlign();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Downsample; }

    void update(T drive, T, const AudioSpec& spec) noexcept override
    {
        T clamped = std::clamp(drive, T(1), T(100));
        reduction_ = std::max(1, static_cast<int>(mapRange(clamped, T(1), T(100), T(1), T(50))));
        
        auto c = BiquadCoeffs::makeLowPass(spec.sampleRate, spec.sampleRate / (2.5 * reduction_), 0.707);
        for (int ch = 0; ch < numChannels_; ++ch)
            aaFilters_[ch].setCoeffs(c);
    }

    inline T processSample(T sample, T, T, int ch) noexcept
    {
        T filtered = aaFilters_[ch].processSample(sample, 0);
        if (++counter_[ch] >= reduction_)
        {
            counter_[ch] = 0;
            lastSample_[ch] = filtered;
        }
        return lastSample_[ch];
    }
};

// -- MultiStage (Tube -> Tape -> Transformer cascade) -------------------------
template <typename T>
class MultiStageAlgorithm final : public SaturationAlgorithm<T>
{
    TubeAlgorithm<T>        tube_;
    TapeAlgorithm<T>        tape_;
    TransformerAlgorithm<T> xfmr_;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        tube_.prepare(spec);
        tape_.prepare(spec);
        xfmr_.prepare(spec);
    }
    void reset() noexcept override { tube_.reset(); tape_.reset(); xfmr_.reset(); }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::MultiStage; }

    /// processSecond() delays by exactly the common delay (no padding).
    static constexpr bool kSelfAligned = true;

    void update(T drive, T character, const AudioSpec& spec) noexcept override
    {
        // The stages are private instances (not the pool's), so the owner's
        // setAntialiasing() never reaches them directly: forward the orders
        // that processSecond() needs for the common delay (see there).
        const bool second = this->order_ == 2;
        const int delay = this->delay_;
        tube_.setAntialias(second ? (delay == 2 ? 1 : 2) : this->order_);
        tube_.latchOrder();
        tape_.setAntialias(second && delay >= 2 ? 2 : 0);
        tape_.latchOrder();
        xfmr_.setAntialias(second && delay >= 2 ? 2 : 0);
        xfmr_.latchOrder();
        tape_.update(drive * T(0.6), character, spec);
        xfmr_.update(drive * T(0.8), character, spec);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        // Inter-stage makeup undoes each stage's fixed small-signal factor
        // (1/0.5, 1/0.6, 1/0.8) so the cascade is gain-staged: transparent
        // at neutral drive instead of a flat -15 dB drop, and every stage
        // keeps receiving a healthy level. (The last stage's 0.8 used to be
        // left in, a -1.9 dB loss that the old Tube curve's +0.6 dB kink gain
        // partly hid.)
        T tubeOut  = tube_.processSample(sample,  drive * T(0.5), character, ch) * T(2);
        T tapeOut  = tape_.processSample(tubeOut, drive * T(0.6), character, ch) * (T(1) / T(0.6));
        return       xfmr_.processSample(tapeOut, drive * T(0.8), character, ch) * (T(1) / T(0.8));
    }

    /**
     * Second order. Each stage regenerates the harmonics the previous one's
     * kernel removed, and the transformer's low band amplifies whatever
     * aliasing reaches it (a stage's fold-back lands near DC), so every stage
     * needs a kernel; the cascade's delay must equal the common delay:
     * - 1 (1x): the tube kernel only (one sample);
     * - 2 (2x): first-order tube, tape kernel, first-order transformer
     *   (0.5 + 1 + 0.5 samples);
     * - 3 (4x and above): second-order kernels on all three stages.
     * Worst alias over 1-19.25 kHz tones at -6/-18 dBFS, +12/+24 dB:
     * -37.1/-28.0 dBc at 2x and -70.6/-49.7 dBc at 4x, against
     * -16.0/-6.8 and -32.4/-6.0 with the tube kernel alone.
     */
    inline T processSecond(T sample, T drive, T character, int ch) noexcept
    {
        const int delay = this->delay_;
        if (delay == 1)
        {
            T tubeOut  = tube_.processSecond(sample, drive * T(0.5), character, ch) * T(2);
            T tapeOut  = tape_.processSample(tubeOut, drive * T(0.6), character, ch) * (T(1) / T(0.6));
            return       xfmr_.processSample(tapeOut, drive * T(0.8), character, ch) * (T(1) / T(0.8));
        }
        if (delay == 2)
        {
            T tubeOut  = tube_.processSample(sample, drive * T(0.5), character, ch) * T(2);
            T tapeOut  = tape_.processSecond(tubeOut, drive * T(0.6), character, ch) * (T(1) / T(0.6));
            return       xfmr_.processFirst(tapeOut, drive * T(0.8), character, ch) * (T(1) / T(0.8));
        }
        T tubeOut  = tube_.processSecond(sample, drive * T(0.5), character, ch) * T(2);
        T tapeOut  = tape_.processSecond(tubeOut, drive * T(0.6), character, ch) * (T(1) / T(0.6));
        return       xfmr_.processSecond(tapeOut, drive * T(0.8), character, ch) * (T(1) / T(0.8));
    }
};

} // namespace detail

// ============================================================================
// Saturation - Public API
// ============================================================================

/**
 * @class Saturation
 * @brief Professional multi-algorithm saturation processor with analog simulation.
 *
 * This class provides a high-quality saturation pipeline featuring 10 distinct
 * algorithms ranging from soft clipping to complex magnetic tape and
 * transformer modeling. Zero latency at 1x; with oversampling enabled the
 * latency equals the oversampler group delay, plus one sample with
 * second-order antialiasing (see getLatency()).
 *
 * Channel handling: the saturation stage processes up to 16 channels and no
 * more than the prepared spec's channel count; any further channels in the
 * incoming view pass through the linear stages (filters, gain, mix) only.
 *
 * @tparam SampleType The floating-point precision to use (must be `float` or `double`).
 */
template <typename SampleType>
class Saturation
{
    static_assert(std::is_floating_point_v<SampleType>,
                  "Saturation: SampleType must be float or double.");

public:
    // -- Enums ---------------------------------------------------------------

    /**
     * @brief Defines the harmonic generation topology.
     */
    enum class Algorithm
    {
        Tube,        /**< Asymmetric triode emulation. Dominant 2nd harmonic, compresses negative excursions. */
        Tape,        /**< Anhysteretic Langevin magnetisation (tape under AC bias) with dynamic head-bump and HF roll-off. */
        Transformer, /**< Core saturation model. LF saturates before HF. */
        SoftClip,    /**< Symmetric Tanh curve. Odd harmonics only (3rd, 5th, 7th). */
        HardClip,    /**< Digital hard-clipping. Extreme odd harmonics, highly aliasing without oversampling. */
        Exciter,     /**< Polynomial waveshaper designed to synthesize high-frequency harmonic content. */
        Wavefolder,  /**< First-order ADAA (Antiderivative Anti-Aliasing) sine wavefolder. */
        Bitcrusher,  /**< Quantization noise generator with TPDF dithering. */
        Downsample,  /**< Sample-and-hold rate reduction with anti-aliasing pre-filtering. */
        MultiStage   /**< Serial cascade: Tube -> Tape -> Transformer. Gain-staged internally. */
    };

    /** @brief Determines how the stereo field is processed. */
    enum class ProcessingMode { Stereo, MidOnly, SideOnly, MidSide };

    /** @brief Determines the final output signal routing. */
    enum class OutputMode { Normal, WetOnly, Delta };

    /**
     * @brief Antiderivative antialiasing order (see setAntialiasing()).
     *
     * - Off: plain curves (the Wavefolder keeps its built-in first-order form).
     * - FirstOrder: SoftClip/Tube/HardClip, half a processing sample of
     *   uncompensated delay, -2.0 dB at 20 kHz at the default 2x.
     * - SecondOrder (Bilbao, Esqueda, Parker and Valimaki 2017): a kernel on
     *   every curve except the Bitcrusher and Downsample
     *   (SoftClip/Tube/HardClip/Wavefolder/Transformer/Exciter/Tape), and on
     *   all three MultiStage stages from 2x (the tube stage only at 1x).
     *   Every algorithm is delayed by the same min(factor, 3) processing
     *   samples and the wet path is padded to one base sample, which
     *   getLatency() reports and the dry path matches. -1.3 dB at 10 kHz and
     *   -5.9 dB at 20 kHz at 2x (-1.3 dB at 20 kHz at 4x); MultiStage -2.3
     *   and -9.9 dB at 2x, -3.9 dB at 20 kHz at 4x. Worst alias over
     *   1-19.25 kHz tones at -6/-18 dBFS, 2x, +24 dB drive: SoftClip
     *   -42.1 dBc against -29.7 for FirstOrder and -15.2 off; Transformer
     *   -43.2 against -15.7; Tape -38.0 against -7.7; Exciter -27.5 against
     *   -9.8; MultiStage -28.0 against -5.5 (at 4x -49.7 against -5.7).
     *   Meant for oversampled use: at 1x the kernel's -12 dB at 20 kHz lowers
     *   high fundamentals more than it removes alias (SoftClip +24 dB:
     *   -2.1 dBc against -10.0 off over the same tones). Cost at 2x, stereo
     *   float, ns per frame: Tape 510 against 300 plain, MultiStage 820
     *   against 260 plain, Exciter 150 against 35.
     */
    enum class Antialiasing { Off, FirstOrder, SecondOrder };

    // -- Lifecycle -----------------------------------------------------------

    Saturation()
    {
        pool_[0]  = std::make_unique<detail::TubeAlgorithm<SampleType>>();
        pool_[1]  = std::make_unique<detail::TapeAlgorithm<SampleType>>();
        pool_[2]  = std::make_unique<detail::TransformerAlgorithm<SampleType>>();
        pool_[3]  = std::make_unique<detail::TanhAlgorithm<SampleType>>();
        pool_[4]  = std::make_unique<detail::HardClipAlgorithm<SampleType>>();
        pool_[5]  = std::make_unique<detail::ExciterAlgorithm<SampleType>>();
        pool_[6]  = std::make_unique<detail::WavefolderAlgorithm<SampleType>>();
        pool_[7]  = std::make_unique<detail::BitcrusherAlgorithm<SampleType>>();
        pool_[8]  = std::make_unique<detail::DownsampleAlgorithm<SampleType>>();
        pool_[9]  = std::make_unique<detail::MultiStageAlgorithm<SampleType>>();

        active_.store(pool_[static_cast<int>(Algorithm::SoftClip)].get());
        next_.store(nullptr);
    }

    ~Saturation() = default;
    Saturation(const Saturation&)            = delete;
    Saturation& operator=(const Saturation&) = delete;

    /**
     * @brief Prepares all internal resources, filters, and buffers.
     * 
     * @note **NOT Real-Time Safe.** This method performs memory allocations (`std::vector::resize`) 
     * and filter coefficient calculations. It must be called from the main/setup thread before 
     * audio processing begins.
     * 
     * @param spec The audio environment specifications (sample rate, max block size, channels).
     *             An invalid spec (non-positive or NaN fields) is ignored and
     *             the previous state is kept.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        spec_ = spec;
        for (auto& algo : pool_)
            if (algo) algo->prepare(spec);

        preFilter_.reset();
        postFilter_.reset();
        // Order 2 at 5 Hz: the same Butterworth design this stage always used,
        // now run in the double core (see DCBlocker.h). Prepared for the full
        // channel range so it covers every channel the pipeline touches, not
        // just the first eight.
        dcBlocker_.setOrder(2);
        dcBlocker_.prepare(spec.sampleRate, DCBlocker<SampleType>::kMaxChannels, 5.0);
        
        lastPreHpFreq_    = -1.0f;
        lastPostTiltFreq_ = -1.0f;
        lastPostTiltGain_ = std::numeric_limits<float>::quiet_NaN();
        dryWetMixer_.prepare(spec);
        
        if (oversamplingFactor_ > 1 && !oversampler_)
            oversampler_ = std::make_unique<Oversampling<SampleType>>(oversamplingFactor_);
        if (oversampler_) oversampler_->prepare(spec);

        tempBuffer_.resize(spec.numChannels, spec.maxBlockSize * std::max(1, oversamplingFactor_));
        driftBuffer_.resize(spec.numChannels, spec.maxBlockSize * std::max(1, oversamplingFactor_));
        msKeepBuffer_.resize(1, spec.maxBlockSize * std::max(1, oversamplingFactor_));
        dryAligned_.resize(spec.numChannels, spec.maxBlockSize);
        driveRamp_.assign(static_cast<size_t>(spec.maxBlockSize * std::max(1, oversamplingFactor_)), SampleType(1));
        charRamp_.assign(static_cast<size_t>(spec.maxBlockSize * std::max(1, oversamplingFactor_)), SampleType(0));

        // Keep the dry path aligned with the (latent) oversampled wet path so the
        // dry/wet, Delta and adaptive-blend mixes do not comb-filter.
        dryWetMixer_.setLatencyCompensation(
            (oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0);

        // Base-rate smoothers (consumed once per base sample).
        auto sr = spec.sampleRate;
        mixSmoother_.reset(sr, 20.0f, 1.0f);
        preHpSmoother_.reset(sr, 30.0f, 0.707f, 20.0f);
        postTiltFreqSmoother_.reset(sr, 30.0f, 0.707f, 1000.0f);
        postTiltGainSmoother_.reset(sr, 30.0f, 0.0f);
        outputGainSmoother_.reset(sr, 20.0f, 0.0f);
        // Processing-rate state (drive, character, drift, crossfade).
        configureProcessingRate(false);

        // Re-apply the parameters: the smoother resets above restored the
        // defaults, and a second prepare() (a host re-activation) used to
        // drop every setting until the next setter (-9.8 dB measured for
        // drive +24, mix 0.5, output -6 while getState() still reported them).
        {
            Params pending;
            while (paramQueue_.pop(pending)) {}
            paramsPending_.store(false, std::memory_order_relaxed);
            Params current;
            {
                SpinLock::ScopedLock guard(paramsLock_);
                current = lastParams_;
            }
            applyParamSnapshot(current);
        }

        reset();
        prepared_ = true;
    }

    /**
     * @brief Clears all internal states, phase memory, and history buffers.
     * 
     * @note **Real-Time Safe.** Can be called safely from the audio thread to prevent clicks 
     * when the transport stops or loops.
     */
    void reset() noexcept
    {
        // Complete any pending algorithm switch instantly: every algorithm's
        // state is cleared below anyway, and after a reset the most recently
        // requested algorithm must be the one playing. (Leaving next_ armed
        // made a later request of that same algorithm a silent no-op.)
        if (auto* pending = next_.load())
        {
            active_.store(pending);
            next_.store(nullptr);
        }

        for (auto& algo : pool_)
            if (algo) algo->reset();

        preFilter_.reset();
        postFilter_.reset();
        dcBlocker_.reset();
        dryWetMixer_.reset();
        if (oversampler_) oversampler_->reset();

        driveSmoother_.skip();
        mixSmoother_.skip();
        characterSmoother_.skip();
        driftSmoother_.skip();
        preHpSmoother_.skip();
        postTiltFreqSmoother_.skip();
        postTiltGainSmoother_.skip();
        outputGainSmoother_.skip();
        crossfader_.setCurrentAndTargetValue(1.0f);

        prevBlendSample_.fill(SampleType(0));
        prevSlewSample_.fill(SampleType(0));
        alignFresh_ = true;
    }

    // -- Audio Processing ----------------------------------------------------

    /**
     * @brief Compatibility alias of processBlock(), with the same thread and
     * preparation contract. Prefer processBlock() in new code.
     * @param buffer Mutable view of the audio data.
     */
    void process(AudioBufferView<SampleType> buffer) noexcept { processBlock(buffer); }

    /**
     * @brief Processes an audio block in-place (AudioProcessor standard contract).
     *
     * Executes the following sequence: parameter updates (lock-free) -> Mid/Side Encoding ->
     * Pre-filtering -> Slew detection -> Drift generation -> Saturation -> Adaptive Blend ->
     * Mid/Side Decoding -> Post-filtering -> DC Blocking -> Dry/Wet Mix.
     *
     * @note **Real-Time Safe.** Call this inside your audio callback.
     * @note Pass-through until prepare() succeeds.
     * 
     * @param buffer Mutable view of the audio data. Will be modified in-place.
     */
    void processBlock(AudioBufferView<SampleType> buffer) noexcept
    {
        if (!prepared_) return;
        handleParameterChanges();

        // Blocks longer than the prepared maximum are processed in pieces
        // (the scratch buffers and the oversampler hold one maximum block).
        const int total = buffer.getNumSamples();
        const int maxBlock = std::max(1, spec_.maxBlockSize);
        for (int off = 0; off < total; off += maxBlock)
            processChunk(buffer.getSubView(off, std::min(maxBlock, total - off)));
    }

private:
    void processChunk(AudioBufferView<SampleType> buffer) noexcept
    {
        // Second-order antialiasing delays the wet path by one base sample
        // (one processing sample inside the algorithms, the rest padded
        // below); the dry capture and the untouched M/S channel follow.
        const bool second = antialiasOrder_.load(std::memory_order_relaxed) == 2;
        if (second != secondActive_)
        {
            secondActive_ = second;
            alignFresh_ = true;   // hold the first sample in the delay lines
        }
        const bool seed = second && alignFresh_;
        if (second) alignFresh_ = false;
        if (second)
        {
            const int nCh = std::min({buffer.getNumChannels(), dryAligned_.getNumChannels(), kMaxCh});
            const int nS = std::min(buffer.getNumSamples(), dryAligned_.getNumSamples());
            for (int ch = 0; ch < nCh; ++ch)
            {
                const SampleType* in = buffer.getChannel(ch);
                SampleType* out = dryAligned_.getChannel(ch);
                SampleType& hold = dryHold_[static_cast<size_t>(ch)];
                if (seed && nS > 0) hold = in[0];
                for (int i = 0; i < nS; ++i)
                {
                    out[i] = hold;
                    hold = in[i];
                }
            }
            dryWetMixer_.pushDry(dryAligned_.toView().getSubView(0, nS));
        }
        else
        {
            dryWetMixer_.pushDry(buffer);
        }

        // Pre-filter
        {
            const int numSamples = buffer.getNumSamples();
            constexpr int kCoefRefresh = 16;
            for (int i = 0; i < numSamples; i += kCoefRefresh)
            {
                const int chunk = std::min(kCoefRefresh, numSamples - i);
                float curFreq = lastPreHpFreq_;
                for (int k = 0; k < chunk; ++k)
                    curFreq = preHpSmoother_.getNextValue();

                if (curFreq != lastPreHpFreq_)
                {
                    auto c = BiquadCoeffs::makeHighPass(spec_.sampleRate, static_cast<double>(curFreq));
                    preFilter_.setCoeffs(c);
                    lastPreHpFreq_ = curFreq;
                }
                auto subView = buffer.getSubView(i, chunk);
                preFilter_.processBlock(subView);
            }
        }

        const bool isMidSide = (procMode_ == ProcessingMode::MidOnly || procMode_ == ProcessingMode::SideOnly || procMode_ == ProcessingMode::MidSide) && buffer.getNumChannels() == 2;
        if (isMidSide) MidSide<SampleType>::encode(buffer);

        // MidOnly / SideOnly: snapshot the channel that must stay UNPROCESSED
        // and restore it after the saturation pipeline. The snapshot lives in
        // the SAME domain the pipeline runs in (oversampled when OS is active),
        // so the restored channel shares the wet path's half-band round-trip
        // and group delay - perfectly time-aligned with the processed channel.
        // (The old in-pipeline routing read the DryWetMixer's base-rate L/R
        // capture from inside the oversampled loop: wrong domain AND an
        // out-of-bounds read, caught by AddressSanitizer.)
        const int keepChannel =
            (isMidSide && procMode_ == ProcessingMode::MidOnly)  ? 1 :
            (isMidSide && procMode_ == ProcessingMode::SideOnly) ? 0 : -1;

        if (oversamplingFactor_ > 1 && oversampler_)
        {
            auto upView = oversampler_->upsample(buffer);

            const int upSamples = std::min(upView.getNumSamples(), msKeepBuffer_.getNumSamples());
            if (keepChannel >= 0 && keepChannel < upView.getNumChannels() && upSamples > 0)
                std::memcpy(msKeepBuffer_.getChannel(0), upView.getChannel(keepChannel),
                            static_cast<std::size_t>(upSamples) * sizeof(SampleType));

            processSaturationPipeline(upView);
            if (second)
                alignSecondOrder(upView, keepChannel, upSamples, oversamplingFactor_, seed);

            if (keepChannel >= 0 && keepChannel < upView.getNumChannels() && upSamples > 0)
                std::memcpy(upView.getChannel(keepChannel), msKeepBuffer_.getChannel(0),
                            static_cast<std::size_t>(upSamples) * sizeof(SampleType));

            oversampler_->downsample(buffer);
        }
        else
        {
            const int baseSamples = std::min(buffer.getNumSamples(), msKeepBuffer_.getNumSamples());
            if (keepChannel >= 0 && keepChannel < buffer.getNumChannels() && baseSamples > 0)
                std::memcpy(msKeepBuffer_.getChannel(0), buffer.getChannel(keepChannel),
                            static_cast<std::size_t>(baseSamples) * sizeof(SampleType));

            processSaturationPipeline(buffer);
            if (second)
                alignSecondOrder(buffer, keepChannel, baseSamples, 1, seed);

            if (keepChannel >= 0 && keepChannel < buffer.getNumChannels() && baseSamples > 0)
                std::memcpy(buffer.getChannel(keepChannel), msKeepBuffer_.getChannel(0),
                            static_cast<std::size_t>(baseSamples) * sizeof(SampleType));
        }

        if (isMidSide) MidSide<SampleType>::decode(buffer);

        // Program-dependent adaptive blend, applied at BASE rate in the L/R
        // domain where the latency-compensated dry capture is valid.
        if (adaptiveBlend_.load(std::memory_order_relaxed))
            applyAdaptiveBlend(buffer);

        // Post-filter
        {
            const int numSamples = buffer.getNumSamples();
            constexpr int kCoefRefresh = 16;
            for (int i = 0; i < numSamples; i += kCoefRefresh)
            {
                const int chunk = std::min(kCoefRefresh, numSamples - i);
                float curFreq = lastPostTiltFreq_;
                float curGain = lastPostTiltGain_;
                for (int k = 0; k < chunk; ++k)
                {
                    curFreq = postTiltFreqSmoother_.getNextValue();
                    curGain = postTiltGainSmoother_.getNextValue();
                }

                if (curFreq != lastPostTiltFreq_ || curGain != lastPostTiltGain_)
                {
                    // A genuine tilt shelf, as the setter documents. (This used
                    // to be makePeak: a bell AT the pivot, which mid-boosted
                    // instead of brightening/darkening.)
                    auto c = BiquadCoeffs::makeTilt(spec_.sampleRate, static_cast<double>(curFreq), static_cast<double>(curGain));
                    postFilter_.setCoeffs(c);
                    lastPostTiltFreq_ = curFreq;
                    lastPostTiltGain_ = curGain;
                }
                auto subView = buffer.getSubView(i, chunk);
                postFilter_.processBlock(subView);
            }
        }

        if (dcBlockingEnabled_) dcBlocker_.processBlock(buffer);
        applyOutputGain(buffer);

        // Mix Output
        if (outputMode_ == OutputMode::WetOnly) {} 
        else if (outputMode_ == OutputMode::Delta)
        {
            const int nCh = std::min(buffer.getNumChannels(), dryWetMixer_.getDryNumChannels());
            const int nS  = std::min(buffer.getNumSamples(), dryWetMixer_.getDryCapturedSamples());
            for (int ch = 0; ch < nCh; ++ch)
            {
                SampleType*       wet = buffer.getChannel(ch);
                const SampleType* dry = dryWetMixer_.getDryChannel(ch);
                for (int i = 0; i < nS; ++i) wet[i] -= dry[i];
            }
        }
        else 
        {
            dryWetMixer_.mixWet(buffer, static_cast<SampleType>(mixSmoother_.getTargetValue()));
        }
    }

public:

    /**
     * @brief Sets the saturation algorithm topology.
     * @note Thread-Safe. Changes are crossfaded internally over 10ms to prevent clicks.
     * @param algo The desired algorithm (e.g., Algorithm::Tube).
     */
    void setAlgorithm(Algorithm algo)       { pushParam([&](auto& p){ p.algorithm = algo; }); }

    /**
     * @brief Sets the input drive gain.
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param dB Drive level in decibels. Range: [-24.0, +48.0]. 0 dB = unity gain.
     */
    void setDrive(SampleType dB)            { pushParam([&](auto& p){ p.driveDb = dB; }); }

    /**
     * @brief Sets the global Dry/Wet blend.
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param mix01 Mix ratio. Range: [0.0 (fully dry), 1.0 (fully wet)].
     */
    void setMix(SampleType mix01)           { pushParam([&](auto& p){ p.mix = mix01; }); }

    /**
     * @brief Adjusts the specific character/bias of the selected algorithm.
     * 
     * Behavior varies per algorithm:
     * - Tube: Controls waveform asymmetry.
     * - Tape: Controls the magnetic coupling (knee hardness and mid-level bloom).
     * - Transformer: Adjusts bias voltage.
     * - Wavefolder: Biases the fold point (asymmetry, even harmonics).
     * 
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param c Character amount. Range: [-1.0, 1.0]. 0.0 represents the neutral state.
     */
    void setCharacter(SampleType c)         { pushParam([&](auto& p){ p.character = c; }); }

    /**
     * @brief Sets the routing configuration for multi-channel processing.
     * @note Thread-Safe. Applies instantly on the next block.
     * @param m The processing mode (Stereo, MidOnly, SideOnly, MidSide).
     */
    void setProcessingMode(ProcessingMode m){ pushParam([&](auto& p){ p.processingMode = m; }); }

    /**
     * @brief Sets the output signal path.
     * @note Thread-Safe. 
     * @param m The output mode (Normal, WetOnly, Delta). Delta mode outputs ONLY the generated harmonics.
     */
    void setOutputMode(OutputMode m)        { pushParam([&](auto& p){ p.outputMode = m; }); }

    /**
     * @brief Injects true-stereo pseudo-random low-frequency modulation (drift) into the saturation drive.
     * @note Thread-Safe. Smoothed internally over 500ms.
     * @param i Intensity of the drift. Range: [0.0, 1.0].
     */
    void setAnalogDrift(SampleType i)       { pushParam([&](auto& p){ p.analogDrift = i; }); }

    /**
     * @brief Configures a pre-saturation high-pass filter.
     * @note Thread-Safe. Smoothed internally over 30ms.
     * @param hz Cutoff frequency in Hertz. Range: [10.0, Nyquist].
     */
    void setPreFilterHpFrequency(SampleType hz) { pushParam([&](auto& p){ p.preFilterHpFreq = hz; }); }

    /**
     * @brief Sets the post-saturation make-up or trim gain.
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param dB Gain in decibels.
     */
    void setOutputGain(SampleType dB)       { pushParam([&](auto& p){ p.outputGain = dB; }); }

    /**
     * @brief Enables or disables the fixed 5 Hz DC blocker.
     * @note Thread-Safe.
     * @param on True to enable DC blocking (default).
     */
    void setDcBlocking(bool on)             { pushParam([&](auto& p){ p.dcBlocking = on; }); }

    /**
     * @brief Enables program-dependent saturation density.
     * @note Thread-Safe. Uses memory_order_relaxed atomic assignment.
     * @param on True to enable adaptive dynamic blending.
     */
    void setAdaptiveBlend(bool on) noexcept { adaptiveBlend_.store(on, std::memory_order_relaxed); }

    /**
     * @brief Enables first-order antiderivative anti-aliasing (ADAA) on the
     * memoryless curves (SoftClip / Tube / HardClip): true selects
     * Antialiasing::FirstOrder, false Off. Best combined with oversampling
     * (Parker/Zavalishin DAFx-16, Bilbao et al. 2017). Off by default.
     * @note Thread-Safe.
     * @note First-order ADAA shifts the wet signal by half a sample. With a
     * partial dry/wet mix this causes a very slight HF comb (< 0.2 dB below
     * 10 kHz at 48 kHz) - inherent to the technique and far smaller than the
     * aliasing it removes.
     */
    void setAntialiasing(bool on) noexcept
    {
        setAntialiasing(on ? Antialiasing::FirstOrder : Antialiasing::Off);
    }

    /**
     * @brief Selects the antialiasing order (see Antialiasing). Any thread;
     * SecondOrder changes getLatency() at once and the audio from the next
     * block, so notify the host. Out-of-range values are clamped.
     */
    void setAntialiasing(Antialiasing mode) noexcept
    {
        const int order = std::clamp(static_cast<int>(mode), 0, 2);
        antialiasOrder_.store(order, std::memory_order_relaxed);   // serialization readback
        for (auto& a : pool_) if (a) a->setAntialias(order);
    }

    /** @brief The selected antialiasing order. */
    [[nodiscard]] Antialiasing getAntialiasing() const noexcept
    {
        return static_cast<Antialiasing>(antialiasOrder_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Sets a derivative-based (slew rate) saturation multiplier.
     * @note Thread-Safe. Uses memory_order_relaxed atomic assignment.
     *       Non-finite values are ignored.
     * @param amount Sensitivity multiplier. Range: [0.0 (off), 1.0 (max)].
     */
    void setSlewSensitivity(SampleType amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        slewSensitivity_.store(std::clamp(amount, SampleType(0), SampleType(1)), std::memory_order_relaxed);
    }
    
    /**
     * @brief Configures a post-saturation first-order tilt EQ.
     *
     * Tilts the spectrum around the pivot: -amountDb/2 at DC, unity at the
     * pivot, +amountDb/2 at Nyquist (total span = amountDb). Positive values
     * brighten the signal; negative values darken it.
     *
     * @note Thread-Safe. Both parameters smoothed internally over 30ms.
     * @param centerHz Pivot frequency in Hertz. Range: [100.0, Nyquist].
     * @param amountDb Total tilt span in decibels. Range: [-12.0, 12.0].
     */
    void setPostFilterTilt(SampleType centerHz, SampleType amountDb)
    {
        pushParam([&](auto& p){ p.postFilterTiltFreq = centerHz; p.postFilterTiltGain = amountDb; });
    }

    /**
     * @brief Configures internal polyphase oversampling to reduce aliasing.
     * 
     * @note **NOT Real-Time Safe.** Must be called before `prepare()`, or `prepare()` 
     * must be called immediately after.
     * 
     * @param factor The oversampling multiplier. Must be a power of 2 (1, 2, 4, 8, 16). 1 = Off.
     *        Other values are ignored.
     *        Default 2: at 1x the curves fold their harmonics back into the
     *        band - a 10.1 kHz tone at -6 dBFS through the default SoftClip
     *        left an alias at 17.7 kHz only 34 dB down (46 dB with ADAA);
     *        at 2x it measures 124 dB down, for the oversampler's latency
     *        (64 samples at 48 kHz, reported by getLatency()).
     */
    void setOversampling(int factor)
    {
        if (factor < 1 || factor > 16 || (factor & (factor - 1)) != 0) return;
        oversamplingFactor_ = factor;
        if (factor > 1)
        {
            oversampler_ = std::make_unique<Oversampling<SampleType>>(factor);
            if (spec_.sampleRate > 0) oversampler_->prepare(spec_);
        }
        else
            oversampler_.reset();

        // If prepare() already ran, grow the scratch buffers to hold the
        // upsampled block and realign the dry path; otherwise prepare() does it.
        if (prepared_)
        {
            const int upBlock = spec_.maxBlockSize * std::max(1, oversamplingFactor_);
            if (tempBuffer_.getNumSamples() < upBlock)
                tempBuffer_.resize(spec_.numChannels, upBlock);
            if (driftBuffer_.getNumSamples() < upBlock)
                driftBuffer_.resize(spec_.numChannels, upBlock);
            if (msKeepBuffer_.getNumSamples() < upBlock)
                msKeepBuffer_.resize(1, upBlock);

            dryWetMixer_.setLatencyCompensation(
                (oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0);

            // The drive/character/drift/crossfade clocks follow the factor.
            configureProcessingRate(true);
            reset();
        }
    }

    // -- Thread-Safe Getters (GUI / Metering) --------------------------------

    /** 
     * @brief Retrieves the current oversampling factor. 
     * @return Multiplier (1, 2, 4, 8, 16).
     */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return oversamplingFactor_; }

    /**
     * @brief Reports the processor's algorithmic latency in samples.
     * @note The oversampler group delay (0 when oversampling is off), plus
     *       one sample with second-order antialiasing. Report this to the
     *       host for plugin delay compensation (PDC).
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return ((oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0)
             + (antialiasOrder_.load(std::memory_order_relaxed) == 2 ? 1 : 0);
    }

    /** @brief Compatibility alias of getLatency(), in prepared-rate samples. */
    [[nodiscard]] int getLatencySamples() const noexcept { return getLatency(); }

    /** 
     * @brief Retrieves the currently active underlying algorithm.
     * @note Real-Time Safe reading via relaxed atomics.
     * @return The active Algorithm enum.
     */
    [[nodiscard]] Algorithm getCurrentAlgorithm() const noexcept { return currentAlgoType_.load(std::memory_order_relaxed); }

    /** 
     * @brief Calculates the peak gain reduction (clipping amount) for metering.
     * @note Real-Time Safe reading via relaxed atomics.
     * @return Gain reduction in decibels (always <= 0.0).
     */
    [[nodiscard]] SampleType getGainReductionDb() const noexcept { return gainReductionDb_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        Params p;
        {
            SpinLock::ScopedLock guard(paramsLock_);
            p = lastParams_;
        }
        StateWriter w(stateId("SATU"), 1);
        w.write("algorithm", static_cast<int32_t>(p.algorithm));
        w.write("procMode", static_cast<int32_t>(p.processingMode));
        w.write("outMode", static_cast<int32_t>(p.outputMode));
        w.write("drive", static_cast<float>(p.driveDb));
        w.write("mix", static_cast<float>(p.mix));
        w.write("character", static_cast<float>(p.character));
        w.write("drift", static_cast<float>(p.analogDrift));
        w.write("preHp", static_cast<float>(p.preFilterHpFreq));
        w.write("tiltFreq", static_cast<float>(p.postFilterTiltFreq));
        w.write("tiltGain", static_cast<float>(p.postFilterTiltGain));
        w.write("outputGain", static_cast<float>(p.outputGain));
        w.write("dcBlocking", p.dcBlocking);
        w.write("antialias", antialiasOrder_.load(std::memory_order_relaxed) != 0);
        w.write("antialiasOrder", antialiasOrder_.load(std::memory_order_relaxed));
        w.write("adaptiveBlend", adaptiveBlend_.load(std::memory_order_relaxed));
        w.write("slewSens", static_cast<float>(slewSensitivity_.load(std::memory_order_relaxed)));
        w.write("oversampling", oversamplingFactor_);
        return w.blob();
    }

    /** @brief Restores parameters from a blob (setup/UI threads: it forwards
     *  the stored oversampling factor to setOversampling, which may allocate).
     *  Out-of-range enum values from foreign or corrupt blobs are clamped. */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("SATU")) return false;
        setAlgorithm(static_cast<Algorithm>(r.read("algorithm", 0)));
        setProcessingMode(static_cast<ProcessingMode>(r.read("procMode", 0)));
        setOutputMode(static_cast<OutputMode>(r.read("outMode", 0)));
        setDrive(static_cast<SampleType>(r.read("drive", 0.0f)));
        setMix(static_cast<SampleType>(r.read("mix", 1.0f)));
        setCharacter(static_cast<SampleType>(r.read("character", 0.0f)));
        setAnalogDrift(static_cast<SampleType>(r.read("drift", 0.0f)));
        setPreFilterHpFrequency(static_cast<SampleType>(r.read("preHp", 20.0f)));
        setPostFilterTilt(static_cast<SampleType>(r.read("tiltFreq", 1000.0f)),
                          static_cast<SampleType>(r.read("tiltGain", 0.0f)));
        setOutputGain(static_cast<SampleType>(r.read("outputGain", 0.0f)));
        setDcBlocking(r.read("dcBlocking", true));
        // "antialiasOrder" (0/1/2) supersedes the older boolean, which
        // meant first order.
        setAntialiasing(static_cast<Antialiasing>(
            r.read("antialiasOrder", r.read("antialias", false) ? 1 : 0)));
        setAdaptiveBlend(r.read("adaptiveBlend", false));
        setSlewSensitivity(static_cast<SampleType>(r.read("slewSens", 0.0f)));
        setOversampling(r.read("oversampling", 1));
        return true;
    }

protected:
    /** Prepares the state that runs at the processing (oversampled) rate:
     *  the drive, character and drift smoothers, the algorithm crossfade and
     *  the drift generators. They were prepared at the base rate, so their
     *  times shrank by the factor (drive t63 18.7 ms instead of 36 at 2x,
     *  crossfade 5.3 ms instead of 10, drift retriggering at 2 Hz).
     *  keepTargets snaps each smoother to its current target. Setup thread. */
    void configureProcessingRate(bool keepTargets)
    {
        const double pr = spec_.sampleRate * static_cast<double>(std::max(1, oversamplingFactor_));
        driveSmoother_.reset(pr, 20.0f, 0.707f, keepTargets ? driveSmoother_.getTargetValue() : 0.0f);
        characterSmoother_.reset(pr, 20.0f, keepTargets ? characterSmoother_.getTargetValue() : 0.0f);
        driftSmoother_.reset(pr, 500.0f, keepTargets ? driftSmoother_.getTargetValue() : 0.0f);
        crossfader_.reset(pr, 10.0f, 1.0f);

        leftDrift_.prepare(pr);
        rightDrift_.prepare(pr);
        leftDrift_.reseed(0x9E3779B97F4A7C15ULL);
        rightDrift_.reseed(0xBF58476D1CE4E5B9ULL);
        // Smooth the random drift targets (~100 ms): without smoothing the
        // generators step once per second, modulating the drive in audible
        // jumps instead of an analog-style slow wander.
        leftDrift_.setSmoothing(true, SampleType(100));
        rightDrift_.setSmoothing(true, SampleType(100));

        // Slew sensitivity is defined per sample at 96 kHz (48 kHz at the
        // default 2x): a raw sample delta halves with every doubling of rate.
        slewRateScale_ = static_cast<SampleType>(pr / 96000.0);

        const auto up = static_cast<size_t>(std::max(1, spec_.maxBlockSize) * std::max(1, oversamplingFactor_));
        if (driveRamp_.size() < up) driveRamp_.assign(up, SampleType(1));
        if (charRamp_.size() < up) charRamp_.assign(up, SampleType(0));
    }

    struct Params
    {
        Algorithm      algorithm      = Algorithm::SoftClip;
        ProcessingMode processingMode = ProcessingMode::Stereo;
        OutputMode     outputMode     = OutputMode::Normal;
        SampleType     driveDb        = SampleType(0);
        SampleType     mix            = SampleType(1);
        SampleType     character      = SampleType(0);
        SampleType     analogDrift    = SampleType(0);
        SampleType     preFilterHpFreq    = SampleType(20);
        SampleType     postFilterTiltFreq = SampleType(1000);
        SampleType     postFilterTiltGain = SampleType(0);
        SampleType     outputGain         = SampleType(0);
        bool           dcBlocking         = true;
    };

    /** Keeps every snapshot the audio thread can ever see well-formed: NaN/Inf
     *  from the GUI reverts to the previous value (setter-with-NaN == ignored,
     *  the framework-wide policy - without this the smoothers and filter
     *  designs are poisoned permanently), and enum fields are clamped so a
     *  cast-from-int caller or a corrupt state blob can never index outside
     *  pool_. Also keeps lastParams_ finite for getState(). */
    static void sanitizeParams(Params& p, const Params& prev) noexcept
    {
        const auto keepFinite = [](SampleType& v, SampleType old) noexcept
        {
            if (!std::isfinite(v)) v = old;
        };
        keepFinite(p.driveDb,            prev.driveDb);
        keepFinite(p.mix,                prev.mix);
        keepFinite(p.character,          prev.character);
        keepFinite(p.analogDrift,        prev.analogDrift);
        keepFinite(p.preFilterHpFreq,    prev.preFilterHpFreq);
        keepFinite(p.postFilterTiltFreq, prev.postFilterTiltFreq);
        keepFinite(p.postFilterTiltGain, prev.postFilterTiltGain);
        keepFinite(p.outputGain,         prev.outputGain);

        const auto clampEnum = [](auto& e, int hi) noexcept
        {
            using E = std::remove_reference_t<decltype(e)>;
            e = static_cast<E>(std::clamp(static_cast<int>(e), 0, hi));
        };
        clampEnum(p.algorithm,      kNumAlgorithms - 1);
        clampEnum(p.processingMode, static_cast<int>(ProcessingMode::MidSide));
        clampEnum(p.outputMode,     static_cast<int>(OutputMode::Delta));
    }

    template <typename Fn>
    void pushParam(Fn&& mutate)
    {
        SpinLock::ScopedLock guard(paramsLock_);
        const Params prev = lastParams_;
        mutate(lastParams_);
        sanitizeParams(lastParams_, prev);
        if (!paramQueue_.push(lastParams_))
        {
            // Queue full (a burst of GUI edits within one audio block).
            // lastParams_ already holds the complete state, so flag the audio
            // thread to fetch it directly; without this the newest edit would
            // silently wait for the NEXT parameter change to become audible.
            paramsPending_.store(true, std::memory_order_release);
        }
    }

    void handleParameterChanges()
    {
        Params p;
        while (paramQueue_.pop(p))
            applyParamSnapshot(p);

        // Recover an edit that could not be queued. lastParams_ is always the
        // newest complete state, so applying it can never move backwards.
        if (paramsPending_.exchange(false, std::memory_order_acquire))
        {
            SpinLock::ScopedTryLock guard(paramsLock_);
            if (guard.isLocked())
                applyParamSnapshot(lastParams_);
            else
                paramsPending_.store(true, std::memory_order_release); // retry next block
        }
    }

    void applyParamSnapshot(const Params& p)
    {
        driveSmoother_.setTargetValue(std::clamp(static_cast<float>(p.driveDb), -24.0f, 48.0f));
        mixSmoother_.setTargetValue(std::clamp(static_cast<float>(p.mix), 0.0f, 1.0f));
        characterSmoother_.setTargetValue(std::clamp(static_cast<float>(p.character), -1.0f, 1.0f));
        driftSmoother_.setTargetValue(std::clamp(static_cast<float>(p.analogDrift), 0.0f, 1.0f));
        outputGainSmoother_.setTargetValue(static_cast<float>(p.outputGain));

        float nyquist = static_cast<float>(spec_.sampleRate) / 2.0f;
        preHpSmoother_.setTargetValue(std::clamp(static_cast<float>(p.preFilterHpFreq), 10.0f, nyquist));
        postTiltFreqSmoother_.setTargetValue(std::clamp(static_cast<float>(p.postFilterTiltFreq), 100.0f, nyquist));
        postTiltGainSmoother_.setTargetValue(std::clamp(static_cast<float>(p.postFilterTiltGain), -12.0f, 12.0f));

        dcBlockingEnabled_ = p.dcBlocking;
        procMode_   = p.processingMode;
        outputMode_ = p.outputMode;
        currentAlgoType_.store(p.algorithm, std::memory_order_relaxed);

        // Algorithm switching state machine. The crossfader fades active_ out
        // towards next_ (1 -> 0); processSaturationPipeline resolves the end
        // of the fade by its final value (0 = promote next_, 1 = discard it).
        auto* requested = pool_[static_cast<int>(p.algorithm)].get();
        auto* act = active_.load();
        auto* nxt = next_.load();
        if (requested == act)
        {
            // Return to the algorithm still playing: steer any running fade
            // back to 1 so the pending algorithm is discarded when it lands.
            // (Ignoring this case made a quick B-then-back-to-A edit complete
            // the fade to B: the wrong algorithm played forever while
            // getCurrentAlgorithm() reported A.)
            if (nxt != nullptr)
            {
                crossfader_.setTargetValue(1.0f);
                if (!crossfader_.isSmoothing())
                {
                    // The fade had not moved yet (still at 1): discard now, or
                    // the un-smoothing crossfader would leave nxt armed and a
                    // later reset() would wrongly promote it.
                    nxt->reset();
                    next_.store(nullptr);
                }
            }
        }
        else if (requested == nxt)
        {
            // Re-request of the incoming algorithm (e.g. after a revert
            // started): make sure the fade heads towards it again.
            crossfader_.setTargetValue(0.0f);
        }
        else
        {
            // New target. It has not been audible for a while, so clear its
            // state: stale filter/ADAA history would leak an old-signal
            // transient into the fade-in. If a fade was already running, its
            // pending algorithm is simply replaced (one fade-weighted step,
            // bounded like a hard switch; the abandoned one is re-cleared
            // here whenever it is next requested).
            requested->reset();
            next_.store(requested);
            crossfader_.setTargetValue(0.0f);
        }
    }

    /** Processing samples of second-order delay inside the algorithms:
     *  min(factor, 3), padded to one base sample by alignSecondOrder(). */
    [[nodiscard]] int secondOrderDelay() const noexcept { return std::clamp(oversamplingFactor_, 1, 3); }

    // Data-Oriented Pipeline to eliminate per-sample virtual dispatch
    void processSaturationPipeline(AudioBufferView<SampleType> buffer) noexcept
    {
        auto* primary   = active_.load();
        auto* secondary = next_.load();
        bool  xfading   = secondary != nullptr && crossfader_.isSmoothing();

        // Clamp to per-channel state capacity (prevSlewSample_/prevBlendSample_
        // are kMaxCh) AND to the prepared channel count: driftBuffer_ and
        // tempBuffer_ only hold spec_.numChannels channels, so a caller view
        // with more channels used to read/write past them (out of bounds in
        // release) as soon as drift or an algorithm crossfade was active.
        // Channels beyond the prepared spec now bypass the saturation stage.
        const int nCh = std::min({ buffer.getNumChannels(), kMaxCh, spec_.numChannels });
        const int nS  = buffer.getNumSamples();

        auto driveDbTarget = static_cast<SampleType>(driveSmoother_.getTargetValue());
        auto driveGainTarget = decibelsToGain(driveDbTarget);
        auto characterTarget = static_cast<SampleType>(characterSmoother_.getTargetValue());

        AudioSpec updateSpec = spec_;
        if (oversamplingFactor_ > 1) updateSpec.sampleRate *= oversamplingFactor_;

        const int commonDelay = secondOrderDelay();
        if (primary)   { primary->latchOrder(commonDelay);   primary->update(driveGainTarget, characterTarget, updateSpec); }
        if (secondary) { secondary->latchOrder(commonDelay); secondary->update(driveGainTarget, characterTarget, updateSpec); }

        SampleType peakInOriginal = SampleType(0);
        for (int ch = 0; ch < nCh; ++ch) {
            const SampleType* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i) peakInOriginal = std::max(peakInOriginal, std::abs(d[i]));
        }

        // 1. Slew (In-place)
        auto slewAmt = slewSensitivity_.load(std::memory_order_relaxed);
        if (slewAmt > SampleType(0))
        {
            for (int ch = 0; ch < nCh; ++ch) {
                SampleType* data = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i) {
                    SampleType dry = data[i];
                    SampleType delta = (dry - prevSlewSample_[ch]) * slewRateScale_;
                    data[i] += std::tanh(std::abs(delta)) * slewAmt * dry;
                    prevSlewSample_[ch] = dry;
                }
            }
        }

        // 2. Pre-generate Drift
        auto driftIntensity = driftSmoother_.getTargetValue();
        bool useDrift = driftIntensity > 0.01f;
        if (useDrift)
        {
            auto driftView = driftBuffer_.toView();
            for (int i = 0; i < nS; ++i) {
                auto driftS = driftSmoother_.getNextValue();
                // Exactly one draw per generator per sample (stereo-identical
                // to drawing inside the channel loop). Channels beyond the
                // second share the right generator's value; drawing per
                // channel instead advanced its clock nCh-1 times per sample,
                // speeding the wander up with the channel count.
                const SampleType noiseL = leftDrift_.getNextSample();
                const SampleType noiseR = (nCh > 1) ? rightDrift_.getNextSample() : SampleType(0);
                for (int ch = 0; ch < nCh; ++ch) {
                    driftView.getChannel(ch)[i] =
                        SampleType(1) + static_cast<SampleType>(driftS) * (ch == 0 ? noiseL : noiseR);
                }
            }
        }

        // 3. Drive and character trajectories, advanced ONCE per sample and
        //    shared by both algorithms of a crossfade (each used to advance
        //    the smoothers again, so a preset change depended on the host's
        //    block partition by up to 0.45 of full scale).
        for (int i = 0; i < nS; ++i)
        {
            driveRamp_[static_cast<size_t>(i)] =
                decibelsToGain(static_cast<SampleType>(driveSmoother_.getNextValue()));
            charRamp_[static_cast<size_t>(i)] = static_cast<SampleType>(characterSmoother_.getNextValue());
        }

        // 4. Primary Saturation (and secondary if xfade)
        if (xfading)
        {
            auto tempView = tempBuffer_.toView().getSubView(0, nS);
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(tempView.getChannel(ch), buffer.getChannel(ch), static_cast<std::size_t>(nS) * sizeof(SampleType));
            
            dispatchSaturator(primary, buffer, useDrift);
            dispatchSaturator(secondary, tempView, useDrift);

            for (int i = 0; i < nS; ++i) {
                auto fade = static_cast<SampleType>(crossfader_.getNextValue());
                for (int ch = 0; ch < nCh; ++ch) {
                    SampleType* out = buffer.getChannel(ch);
                    const SampleType* alt = tempView.getChannel(ch);
                    out[i] = out[i] * fade + alt[i] * (SampleType(1) - fade);
                }
            }

            if (!crossfader_.isSmoothing()) {
                if (crossfader_.getCurrentValue() < 0.5f) {
                    // Fade ran to 0: the pending algorithm takes over.
                    active_.store(secondary);
                    next_.store(nullptr);
                    if (primary) primary->reset();
                } else {
                    // Reverted fade landed back on the active algorithm:
                    // discard the pending one.
                    secondary->reset();
                    next_.store(nullptr);
                }
                crossfader_.setCurrentAndTargetValue(1.0f);
            }
        }
        else
        {
            dispatchSaturator(primary, buffer, useDrift);
        }

        // (Adaptive blend and the MidOnly/SideOnly channel passthrough run at
        //  BASE rate in process() - see the snapshot/restore logic there.)

        // Gain Reduction Tracking
        SampleType peakOut = SampleType(0);
        for (int ch = 0; ch < nCh; ++ch) {
            const SampleType* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i) peakOut = std::max(peakOut, std::abs(d[i]));
        }

        SampleType peakInDriven = peakInOriginal * driveGainTarget;
        if (peakInDriven > SampleType(1e-6)) {
            SampleType ratio = std::min(peakOut / peakInDriven, SampleType(1));
            gainReductionDb_.store(gainToDecibels(ratio, SampleType(-100)), std::memory_order_relaxed);
        } else {
            gainReductionDb_.store(SampleType(0), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Completes the second-order delay to one base sample.
     *
     * The algorithms delay by min(N, 3) processing samples
     * (secondOrderDelay()), so a factor-N stage adds the rest of N to every
     * saturated channel; the M/S snapshot that is restored unprocessed is
     * delayed by the full N. Rings hold 16 samples;
     * @p seed fills them with each channel's first sample (no step from
     * stale or zero history when the order is switched on).
     */
    void alignSecondOrder(AudioBufferView<SampleType> view, int keepChannel, int keepSamples,
                          int factor, bool seed) noexcept
    {
        const int nCh = std::min({view.getNumChannels(), kMaxCh, spec_.numChannels});
        const int nS = view.getNumSamples();
        const int pad = std::clamp(factor, 1, kAlignRing) - std::clamp(factor, 1, 3);
        int pos = alignPos_;
        if (pad > 0)
            for (int ch = 0; ch < nCh; ++ch)
            {
                if (ch == keepChannel) continue;
                SampleType* d = view.getChannel(ch);
                auto& ring = alignRing_[static_cast<size_t>(ch)];
                if (seed && nS > 0) ring.fill(d[0]);
                pos = alignPos_;
                for (int i = 0; i < nS; ++i)
                {
                    ring[static_cast<size_t>(pos)] = d[i];
                    d[i] = ring[static_cast<size_t>((pos - pad) & (kAlignRing - 1))];
                    pos = (pos + 1) & (kAlignRing - 1);
                }
            }
        if (keepChannel >= 0 && keepSamples > 0)
        {
            SampleType* d = msKeepBuffer_.getChannel(0);
            auto& ring = keepRing_;
            if (seed) ring.fill(d[0]);
            pos = alignPos_;
            const int delay = std::clamp(factor, 1, kAlignRing - 1);
            for (int i = 0; i < keepSamples; ++i)
            {
                ring[static_cast<size_t>(pos)] = d[i];
                d[i] = ring[static_cast<size_t>((pos - delay) & (kAlignRing - 1))];
                pos = (pos + 1) & (kAlignRing - 1);
            }
        }
        alignPos_ = (alignPos_ + nS) & (kAlignRing - 1);
    }

    /** @brief Program-dependent dry/wet density blend (base rate, L/R domain).
     *  The dry reference is the DryWetMixer capture, which carries the same
     *  latency compensation as the wet path - no comb filtering. */
    void applyAdaptiveBlend(AudioBufferView<SampleType> buffer) noexcept
    {
        const int nCh = std::min({ buffer.getNumChannels(),
                                   dryWetMixer_.getDryNumChannels(), kMaxCh });
        const int nS  = std::min(buffer.getNumSamples(),
                                 dryWetMixer_.getDryCapturedSamples());

        for (int ch = 0; ch < nCh; ++ch)
        {
            SampleType* wetData = buffer.getChannel(ch);
            const SampleType* dryData = dryWetMixer_.getDryChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                SampleType dry = dryData[i];
                SampleType wet = wetData[i];
                SampleType avg = (std::abs(prevBlendSample_[ch]) + std::abs(dry)) * SampleType(0.5);
                SampleType apply = std::clamp(avg, SampleType(0), SampleType(1));
                wetData[i] = dry * (SampleType(1) - apply) + wet * apply;
                prevBlendSample_[ch] = dry;
            }
        }
    }

    // Compile-time resolver (Zero virtual dispatch in hot path)
    void dispatchSaturator(detail::SaturationAlgorithm<SampleType>* baseAlgo, AudioBufferView<SampleType> buffer, bool useDrift) noexcept
    {
        if (!baseAlgo) return;
        switch (baseAlgo->getType())
        {
            case Algorithm::Tube:        processCore(static_cast<detail::TubeAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Tape:        processCore(static_cast<detail::TapeAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Transformer: processCore(static_cast<detail::TransformerAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::SoftClip:    processCore(static_cast<detail::TanhAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::HardClip:    processCore(static_cast<detail::HardClipAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Exciter:     processCore(static_cast<detail::ExciterAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Wavefolder:  processCore(static_cast<detail::WavefolderAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Bitcrusher:  processCore(static_cast<detail::BitcrusherAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Downsample:  processCore(static_cast<detail::DownsampleAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::MultiStage:  processCore(static_cast<detail::MultiStageAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
        }
    }

    template <typename ExactAlgo>
    void processCore(ExactAlgo* algo, AudioBufferView<SampleType> buffer, bool useDrift) noexcept
    {
        // Same channel clamp as processSaturationPipeline: per-channel algorithm
        // state is bounded to kMaxCh and driftBuffer_ to spec_.numChannels.
        const int nCh = std::min({ buffer.getNumChannels(), kMaxCh, spec_.numChannels });
        const int nS  = buffer.getNumSamples();
        auto driftView = driftBuffer_.toView();

        if (algo->order() == 2)
        {
            // Second order: the curve's kernel where one exists, otherwise
            // the plain curve delayed by one sample to the same alignment.
            for (int i = 0; i < nS; ++i)
            {
                const SampleType driveGainS = driveRamp_[static_cast<size_t>(i)];
                const SampleType charS      = charRamp_[static_cast<size_t>(i)];
                for (int ch = 0; ch < nCh; ++ch)
                {
                    SampleType drift = useDrift ? driftView.getChannel(ch)[i] : SampleType(1);
                    SampleType* data = buffer.getChannel(ch);
                    if constexpr (requires { ExactAlgo::kSelfAligned; })
                        data[i] = algo->processSecond(data[i], driveGainS * drift, charS * drift, ch);
                    else if constexpr (requires { algo->processSecond(data[i], driveGainS, charS, ch); })
                        data[i] = algo->padKernel(algo->processSecond(data[i], driveGainS * drift, charS * drift, ch), ch);
                    else
                        data[i] = algo->align(algo->processSample(data[i], driveGainS * drift, charS * drift, ch), ch);
                }
            }
            return;
        }

        for (int i = 0; i < nS; ++i)
        {
            const SampleType driveGainS = driveRamp_[static_cast<size_t>(i)];
            const SampleType charS      = charRamp_[static_cast<size_t>(i)];

            for (int ch = 0; ch < nCh; ++ch)
            {
                SampleType drift = useDrift ? driftView.getChannel(ch)[i] : SampleType(1);
                SampleType* data = buffer.getChannel(ch);
                data[i] = algo->processSample(data[i], driveGainS * drift, charS * drift, ch);
            }
        }
    }

    void applyOutputGain(AudioBufferView<SampleType> buffer) noexcept
    {
        auto targetGainDb = outputGainSmoother_.getTargetValue();
        if (std::abs(targetGainDb - outputGainSmoother_.getCurrentValue()) < 0.001f)
        {
            buffer.applyGain(decibelsToGain(static_cast<SampleType>(targetGainDb)));
        }
        else
        {
            const int nCh = buffer.getNumChannels();
            const int nS  = buffer.getNumSamples();
            for (int i = 0; i < nS; ++i)
            {
                auto gain = decibelsToGain(static_cast<SampleType>(outputGainSmoother_.getNextValue()));
                for (int ch = 0; ch < nCh; ++ch) buffer.getChannel(ch)[i] *= gain;
            }
        }
    }

    AudioSpec spec_ {};
    bool prepared_ = false;

    static constexpr int kNumAlgorithms = 10;
    std::array<std::unique_ptr<detail::SaturationAlgorithm<SampleType>>, kNumAlgorithms> pool_;
    std::atomic<detail::SaturationAlgorithm<SampleType>*> active_ { nullptr };
    std::atomic<detail::SaturationAlgorithm<SampleType>*> next_   { nullptr };

    SpscQueue<Params>  paramQueue_;
    Params             lastParams_;
    mutable SpinLock   paramsLock_;
    std::atomic<bool>  paramsPending_ { false }; // set when a full queue dropped a snapshot
    std::atomic<int>   antialiasOrder_ { 0 };      ///< Antialiasing order, mirror for getState.

    ProcessingMode procMode_   = ProcessingMode::Stereo;
    OutputMode     outputMode_ = OutputMode::Normal;
    bool           dcBlockingEnabled_ = true;
    std::atomic<Algorithm> currentAlgoType_ { Algorithm::SoftClip };

    Smoothers::StateVariableSmoother driveSmoother_, preHpSmoother_, postTiltFreqSmoother_;
    Smoothers::LinearSmoother        mixSmoother_, characterSmoother_, driftSmoother_,
                                     outputGainSmoother_, postTiltGainSmoother_;
    Smoothers::LinearSmoother        crossfader_;

    Biquad<SampleType> preFilter_, postFilter_;
    // A dedicated DCBlocker, not a Biquad: a 5 Hz high-pass keeps its poles
    // 1e-4 from the unit circle, where a float biquad recursion cannot hold
    // the zero at DC (it degenerates into an integrator of its own rounding
    // error above ~96 kHz and sources the very offset it is here to remove).
    DCBlocker<SampleType> dcBlocker_;
    DryWetMixer<SampleType> dryWetMixer_;

    AnalogRandom::Generator<SampleType> leftDrift_, rightDrift_;

    AudioBuffer<SampleType> tempBuffer_;
    AudioBuffer<SampleType> driftBuffer_;
    AudioBuffer<SampleType> msKeepBuffer_;  ///< MidOnly/SideOnly channel snapshot.

    std::unique_ptr<Oversampling<SampleType>> oversampler_;
    int oversamplingFactor_ = 2;   ///< 2x by default: the curves alias at 1x (see setOversampling()).

    std::atomic<SampleType> gainReductionDb_ { SampleType(0) };

    std::atomic<bool> adaptiveBlend_ { false };
    static constexpr int kMaxCh = 16;
    std::array<SampleType, kMaxCh> prevBlendSample_ {};

    std::atomic<SampleType> slewSensitivity_ { SampleType(0) };
    SampleType slewRateScale_ = SampleType(1);   ///< Processing rate / 96 kHz.
    std::vector<SampleType> driveRamp_, charRamp_;   ///< Per-sample drive/character (processing rate).
    std::array<SampleType, kMaxCh> prevSlewSample_ {};

    float lastPreHpFreq_    = -1.0f;
    float lastPostTiltFreq_ = -1.0f;
    float lastPostTiltGain_ = std::numeric_limits<float>::quiet_NaN();

    // Second-order antialiasing alignment (audio thread).
    static constexpr int kAlignRing = 32;   ///< > largest delay (16 processing samples).
    bool secondActive_ = false;
    bool alignFresh_ = true;
    std::array<std::array<SampleType, kAlignRing>, kMaxCh> alignRing_ {};
    std::array<SampleType, kAlignRing> keepRing_ {};
    int alignPos_ = 0;
    std::array<SampleType, kMaxCh> dryHold_ {};
    AudioBuffer<SampleType> dryAligned_;
};

} // namespace dspark
