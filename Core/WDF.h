// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file WDF.h
 * @brief Wave Digital Filters: physical circuit modelling building blocks.
 *
 * Implements the classical WDF toolkit (Fettweis 1986): one-port elements
 * (resistor, capacitor, inductor, resistive voltage source), adapted
 * three-port series/parallel connectors, a polarity inverter, and circuit
 * roots -- an ideal voltage source for linear networks and nonlinear roots
 * (Shockley diode, antiparallel diode pair) solved in closed form to full
 * double precision at a fixed cost per sample: for silicon-like constants a
 * full-precision Wright omega table gives the root directly (no exponential
 * and no division on the sample's path), otherwise an omega estimate is
 * followed by one exact correction step, and at low drive an exact series
 * reversion of the diode equation needs no transcendental function at all.
 *
 * Wave convention: voltage waves with a = v + R*i (incident) and
 * b = v - R*i (reflected), so v = (a+b)/2 and i = (a-b)/(2R). Reactances
 * use the bilinear (trapezoidal) discretization: a WDF network is therefore
 * sample-exact against the bilinear transform of its analog transfer
 * function, which is how this header is verified (RC and RLC against the
 * analytic discrete response; diode clippers against a high-precision
 * trapezoidal reference solve of the circuit ODE -- the same integration
 * SPICE `.tran` uses).
 *
 * Trees are composed statically from references -- no virtual dispatch, all
 * scattering inlines. Per sample: call `root.process()` after updating
 * source voltages, then read element voltages/currents.
 *
 * @code
 *   // Diode clipper: Vin --[Rs]--+--> out across antiparallel pair
 *   //                            |
 *   //                           [C] (parallel)
 *   using namespace dspark::wdf;
 *   ResistiveVoltageSource<float> vs { 2200.0f };
 *   Capacitor<float> c { 10e-9f };
 *   Parallel<float, decltype(vs), decltype(c)> tree { vs, c };
 *   DiodePairRoot<float, decltype(tree)> clipper { tree };   // 1N4148-ish
 *
 *   clipper.prepare(48000.0);
 *   // per sample:
 *   vs.setVoltage(in);
 *   clipper.process();
 *   out = clipper.getVoltage();
 * @endcode
 *
 * Beyond series/parallel trees, the header provides an **R-type adaptor**
 * (after K. Werner's thesis): an N-port connector for arbitrary -- non
 * series/parallel -- topologies. Its scattering matrix is derived numerically
 * from the interconnection network via Modified Nodal Analysis: each port is
 * replaced by its instantaneous Thevenin equivalent (source a_i, resistance
 * R_i), the linear node system is solved once per topology/parameter change,
 * and S = 2M - I where M maps incident waves to port voltages. The up-facing
 * port is adapted to the Thevenin resistance seen looking into the network,
 * so R-types nest under any root. `ToneStackFMV` builds on it: the exact
 * Fender '59 Bassman treble/bass/middle network (topology and verification
 * transfer function after Yeh & Smith, DAFx-06).
 *
 * Scope note: nonlinear roots cover one-port nonlinearities; multi-port
 * nonlinear roots (triode inside the WDF tree) remain future work.
 *
 * A non-finite incident wave at a root (a NaN or infinite source voltage)
 * resets the tree instead of being scattered, so it cannot stay in the
 * reactive states; the next finite input continues from rest.
 *
 * Memory: the omega tables (detail::WrightOmega, 22 KB, and
 * detail::DiodeOmegaTable, 103 KB) are built once per process when the first
 * diode root is constructed; nothing is allocated afterwards. A
 * DiodePairRoot holds 9.8 KB of its own for the local expansions of its
 * constants (detail::DiodeLocal), a DiodeRoot 6.2 KB.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <tuple>
#include <utility>

// Keeps the cold solver paths (large diode constants, extreme drive) out of
// the per-sample root, so the hot path inlines into the caller's loop.
// Locally defined and guarded, as in Core/Biquad.h.
#ifndef DSPARK_NOINLINE
  #if defined(_MSC_VER)
    #define DSPARK_NOINLINE __declspec(noinline)
  #elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_NOINLINE __attribute__((noinline))
  #else
    #define DSPARK_NOINLINE
  #endif
#endif

// Forces the kernel of the local expansions into the root that reads a
// cell: it has several callers, and with them a compiler may leave it out
// of line (MSVC did for the pair once the single diode read cells too, 1 ns
// per sample there). Forcing the table and series paths as well was
// measured and is not done: no gain with GCC, a loss with Clang and MSVC.
#ifndef DSPARK_WDF_INLINE
  #if defined(_MSC_VER) && !defined(__clang__)
    #define DSPARK_WDF_INLINE __forceinline
  #elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_WDF_INLINE inline __attribute__((always_inline))
  #else
    #define DSPARK_WDF_INLINE inline
  #endif
#endif

namespace dspark {
namespace wdf {

// ============================================================================
// One-port leaf elements
// ============================================================================

/** @brief Ideal resistor. Absorbs its incident wave (b = 0). */
template <FloatType T>
class Resistor
{
public:
    explicit Resistor(T resistanceOhms) noexcept : value_(resistanceOhms) {}

    void setResistance(T ohms) noexcept { value_ = std::max(ohms, T(1e-9)); }

    /** @brief Current resistance in ohms. */
    [[nodiscard]] T getResistance() const noexcept { return value_; }

    void prepare(double) noexcept {}
    void updatePorts() noexcept { R_ = static_cast<double>(value_); }
    void reset() noexcept { a_ = 0; }

    [[nodiscard]] double portResistance() const noexcept { return R_; }
    [[nodiscard]] T reflected() noexcept { return T(0); }
    void incident(T a) noexcept { a_ = a; }

    /** @brief Voltage across the resistor (valid after the root scattered). */
    [[nodiscard]] T getVoltage() const noexcept { return a_ * T(0.5); }
    /** @brief Current through the resistor. */
    [[nodiscard]] T getCurrent() const noexcept
    {
        return static_cast<T>(static_cast<double>(a_) * 0.5 / R_);
    }

private:
    T value_;
    double R_ = 1.0;
    T a_ = 0;
};

/** @brief Capacitor, bilinear discretization: b[n] = a[n-1], Rp = 1/(2 fs C). */
template <FloatType T>
class Capacitor
{
public:
    explicit Capacitor(T farads) noexcept : value_(farads) {}

    void setCapacitance(T farads) noexcept { value_ = std::max(farads, T(1e-18)); }

    /** @brief Current capacitance in farads. */
    [[nodiscard]] T getCapacitance() const noexcept { return value_; }

    void prepare(double sampleRate) noexcept { fs_ = sampleRate; }
    void updatePorts() noexcept { R_ = 1.0 / (2.0 * fs_ * static_cast<double>(value_)); }
    void reset() noexcept { state_ = 0; a_ = 0; b_ = 0; }

    /** @brief Shifts the voltage reference without changing capacitor current.
     *  Stream-owner only; the finite offset is added to both wave variables
     *  and the delayed wave, preserving their difference. */
    void offsetVoltage(T offset) noexcept
    {
        state_ += offset;
        a_ += offset;
        b_ += offset;
    }

    [[nodiscard]] double portResistance() const noexcept { return R_; }
    [[nodiscard]] T reflected() noexcept { b_ = state_; return b_; }
    void incident(T a) noexcept { a_ = a; state_ = a; }

    [[nodiscard]] T getVoltage() const noexcept { return (a_ + b_) * T(0.5); }
    [[nodiscard]] T getCurrent() const noexcept
    {
        return static_cast<T>(static_cast<double>(a_ - b_) * 0.5 / R_);
    }

private:
    T value_;
    double fs_ = 48000.0;
    double R_ = 1.0;
    T state_ = 0, a_ = 0, b_ = 0;
};

/** @brief Inductor, bilinear discretization: b[n] = -a[n-1], Rp = 2 fs L. */
template <FloatType T>
class Inductor
{
public:
    explicit Inductor(T henries) noexcept : value_(henries) {}

    void setInductance(T henries) noexcept { value_ = std::max(henries, T(1e-12)); }

    void prepare(double sampleRate) noexcept { fs_ = sampleRate; }
    void updatePorts() noexcept { R_ = 2.0 * fs_ * static_cast<double>(value_); }
    void reset() noexcept { state_ = 0; a_ = 0; b_ = 0; }

    [[nodiscard]] double portResistance() const noexcept { return R_; }
    [[nodiscard]] T reflected() noexcept { b_ = -state_; return b_; }
    void incident(T a) noexcept { a_ = a; state_ = a; }

    [[nodiscard]] T getVoltage() const noexcept { return (a_ + b_) * T(0.5); }
    [[nodiscard]] T getCurrent() const noexcept
    {
        return static_cast<T>(static_cast<double>(a_ - b_) * 0.5 / R_);
    }

private:
    T value_;
    double fs_ = 48000.0;
    double R_ = 1.0;
    T state_ = 0, a_ = 0, b_ = 0;
};

/**
 * @brief Voltage source with series resistance (Thevenin leaf): b = Vs.
 *
 * The series resistance doubles as the port resistance, which is what makes
 * the source usable anywhere in the tree (an ideal source can only be a root).
 */
template <FloatType T>
class ResistiveVoltageSource
{
public:
    explicit ResistiveVoltageSource(T seriesResistanceOhms) noexcept
        : value_(seriesResistanceOhms) {}

    void setResistance(T ohms) noexcept { value_ = std::max(ohms, T(1e-9)); }
    void setVoltage(T volts) noexcept { vs_ = volts; }

    void prepare(double) noexcept {}
    void updatePorts() noexcept { R_ = static_cast<double>(value_); }
    void reset() noexcept { a_ = 0; }

    [[nodiscard]] double portResistance() const noexcept { return R_; }
    [[nodiscard]] T reflected() noexcept { return vs_; }
    void incident(T a) noexcept { a_ = a; }

    /** @brief Voltage at the source terminals (after the series resistance). */
    [[nodiscard]] T getVoltage() const noexcept { return (a_ + vs_) * T(0.5); }

private:
    T value_;
    double R_ = 1.0;
    T vs_ = 0, a_ = 0;
};

// ============================================================================
// Connectors
// ============================================================================

/**
 * @brief Adapted three-port series connector.
 *
 * Children connect to ports 1-2; the up-facing port 3 is reflection-free
 * (R3 = R1 + R2), which is what allows nesting under any parent or root.
 *
 * A polarity inversion is folded into the up-facing port. The raw Fettweis
 * series adaptor enforces KVL as v1 + v2 + v3 = 0, which makes elements
 * read "upside down" relative to circuit intuition (and flips again at
 * every nesting level). With the folded inversion the natural convention
 * holds instead: the voltage across the up port equals v1 + v2, so a
 * source driving Series(R, C) yields the textbook divider with all element
 * voltages in their expected polarity, at any nesting depth.
 */
template <FloatType T, typename Child1, typename Child2>
class Series
{
public:
    Series(Child1& c1, Child2& c2) noexcept : c1_(c1), c2_(c2) {}

    void prepare(double sampleRate) noexcept
    {
        c1_.prepare(sampleRate);
        c2_.prepare(sampleRate);
    }
    void updatePorts() noexcept
    {
        c1_.updatePorts();
        c2_.updatePorts();
        const double r1 = c1_.portResistance();
        const double r2 = c2_.portResistance();
        R_ = r1 + r2;
        gamma1_ = static_cast<T>(r1 / R_);
    }
    void reset() noexcept { c1_.reset(); c2_.reset(); a1_ = 0; a2_ = 0; }

    [[nodiscard]] double portResistance() const noexcept { return R_; }

    [[nodiscard]] T reflected() noexcept
    {
        a1_ = c1_.reflected();
        a2_ = c2_.reflected();
        return a1_ + a2_;             // -(a1+a2) with the up-port inversion folded in
    }

    void incident(T a3) noexcept
    {
        // b_k = a_k - gamma_k * (a1 + a2 + a3'), a3' = -a3 (folded inversion),
        // with gamma1 + gamma2 = 1 on the adapted port.
        const T sum = a1_ + a2_ - a3;
        c1_.incident(a1_ - gamma1_ * sum);
        c2_.incident(a2_ - (sum - gamma1_ * sum));   // a2 - gamma2 * sum
    }

private:
    Child1& c1_;
    Child2& c2_;
    double R_ = 2.0;
    T gamma1_ = T(0.5);
    T a1_ = 0, a2_ = 0;
};

/**
 * @brief Adapted three-port parallel connector.
 *
 * Children connect to ports 1-2; the up-facing port 3 is reflection-free
 * (G3 = G1 + G2).
 */
template <FloatType T, typename Child1, typename Child2>
class Parallel
{
public:
    Parallel(Child1& c1, Child2& c2) noexcept : c1_(c1), c2_(c2) {}

    void prepare(double sampleRate) noexcept
    {
        c1_.prepare(sampleRate);
        c2_.prepare(sampleRate);
    }
    void updatePorts() noexcept
    {
        c1_.updatePorts();
        c2_.updatePorts();
        const double g1 = 1.0 / c1_.portResistance();
        const double g2 = 1.0 / c2_.portResistance();
        R_ = 1.0 / (g1 + g2);
        d1_ = static_cast<T>(g1 / (g1 + g2));
    }
    void reset() noexcept { c1_.reset(); c2_.reset(); a1_ = 0; a2_ = 0; bUp_ = 0; }

    [[nodiscard]] double portResistance() const noexcept { return R_; }

    [[nodiscard]] T reflected() noexcept
    {
        a1_ = c1_.reflected();
        a2_ = c2_.reflected();
        bUp_ = d1_ * a1_ + (T(1) - d1_) * a2_;   // weighted node wave
        return bUp_;
    }

    void incident(T a3) noexcept
    {
        // Common node wave: bNode = bUp + a3; then b_k = bNode - a_k.
        const T bNode = bUp_ + a3;
        c1_.incident(bNode - a1_);
        c2_.incident(bNode - a2_);
    }

private:
    Child1& c1_;
    Child2& c2_;
    double R_ = 0.5;
    T d1_ = T(0.5);
    T a1_ = 0, a2_ = 0, bUp_ = 0;
};

/** @brief Two-port polarity inverter (flips the connected subtree's polarity). */
template <FloatType T, typename Child>
class Inverter
{
public:
    explicit Inverter(Child& c) noexcept : c_(c) {}

    void prepare(double sampleRate) noexcept { c_.prepare(sampleRate); }
    void updatePorts() noexcept { c_.updatePorts(); }
    void reset() noexcept { c_.reset(); }

    [[nodiscard]] double portResistance() const noexcept { return c_.portResistance(); }
    [[nodiscard]] T reflected() noexcept { return -c_.reflected(); }
    void incident(T a) noexcept { c_.incident(-a); }

private:
    Child& c_;
};

// ============================================================================
// Roots
// ============================================================================

/**
 * @brief Ideal voltage source closing a linear tree: b = 2 Vs - a.
 *
 * The whole tree hangs across the source. Set the voltage, call process(),
 * then read voltages/currents anywhere in the tree.
 */
template <FloatType T, typename Tree>
class IdealVoltageSourceRoot
{
public:
    explicit IdealVoltageSourceRoot(Tree& tree) noexcept : tree_(tree) {}

    void setVoltage(T volts) noexcept { vs_ = volts; }

    void prepare(double sampleRate) noexcept
    {
        tree_.prepare(sampleRate);
        tree_.updatePorts();
        tree_.reset();
    }
    void reset() noexcept { tree_.reset(); }

    void process() noexcept
    {
        const T a = tree_.reflected();
        // A non-finite source or wave would stay in the reactive states for
        // good: clear the tree instead, so the next finite input recovers.
        if (!std::isfinite(a) || !std::isfinite(vs_)) [[unlikely]]
        {
            tree_.reset();
            return;
        }
        tree_.incident(T(2) * vs_ - a);
    }

private:
    Tree& tree_;
    T vs_ = 0;
};

template <FloatType T, typename Tree> class DiodePairRoot;
template <FloatType T, typename Tree> class DiodeRoot;

namespace detail {

/**
 * @brief Bisection-safeguarded Newton-Raphson for monotonic wave equations.
 *
 * Solves f(v) = v + i(v)*R_total - a = 0 where i(v) is monotonically
 * increasing, so the solution is bracketed by [min(0,a), max(0,a)]. Newton
 * steps that leave the shrinking bracket fall back to bisection -- guaranteed
 * convergence for any drive level, typically 1-3 iterations at audio rates
 * thanks to the previous-sample seed.
 *
 * @tparam F  Functor: void(double v, double& f, double& fprime).
 */
template <typename F>
[[nodiscard]] inline double solveMonotonic(double a, double seed, F&& evaluate) noexcept
{
    double lo = std::min(0.0, a);
    double hi = std::max(0.0, a);
    double v = std::clamp(seed, lo, hi);
    double prevAbsF = 1e300;

    for (int it = 0; it < 48; ++it)
    {
        double f = 0.0, fp = 1.0;
        evaluate(v, f, fp);

        const double absF = std::abs(f);
        if (absF < 1e-10)
            break;
        if (f > 0.0) hi = v;
        else         lo = v;

        // Accept the Newton step only while it makes real progress. Deep in
        // the exponential region Newton crawls at ~nVt per step (the classic
        // diode pathology), so a stalled |f| switches to bisection, which
        // halves the bracket unconditionally. Quadratic near the root,
        // bisection-guaranteed everywhere.
        const double vn = v - f / fp;
        if (vn > lo && vn < hi && absF < 0.7 * prevAbsF)
            v = vn;
        else
            v = 0.5 * (lo + hi);
        prevAbsF = absF;
    }
    return v;
}

/**
 * @brief Wright omega function omega(z), the solution y of y + ln(y) = z
 *        (y = W(e^z) with W the principal Lambert W branch).
 *
 * Werner et al., "An improved and generalized diode clipper model for wave
 * digital filters" (AES 2015), write the single-diode root in terms of
 * omega. This evaluation seeds one exact correction step, so it is tuned
 * for latency rather than for the last bits:
 * - z in (-32, 32): degree-4 Taylor expansion about the nearest of 257
 *   nodes 0.25 apart (exact nodal derivatives), 1.1e-8 worst error relative
 *   to 1 + omega. The node is found with a round-to-nearest add, without a
 *   float-to-integer conversion on the critical path.
 * - z in [32, 2^20): degree-4 Taylor expansion about the midpoint of one
 *   of 16 cells per octave, indexed by the exponent and leading mantissa
 *   bits of z, 1.1e-8 relative to 1 + omega.
 * - beyond: two Newton steps from the asymptote z - ln z; at or below -32:
 *   0 (omega < 1.3e-14).
 * The 22 KB of coefficients are computed once per process on first use
 * (instance()) by a full-precision Newton solve per node. Thread-safe and
 * allocation free after construction; operator() is a pure function of z.
 */
class WrightOmega
{
public:
    [[nodiscard]] static const WrightOmega& instance() noexcept
    {
        static const WrightOmega table;
        return table;
    }

    [[nodiscard]] double operator()(double z) const noexcept
    {
        if (z < kUniformMax)
        {
            if (!(z > kUniformMin)) return 0.0;
            // p + 1.5 * 2^52 rounds p to the nearest integer, which then
            // sits in the low mantissa bits: no float-to-integer conversion.
            constexpr double kRound = 6755399441055744.0;
            const double p = (z - kUniformMin) * kInvStep;
            const double rounded = p + kRound;
            const double t = p - (rounded - kRound);        // [-0.5, 0.5]
            const auto& k = taylor_[static_cast<size_t>(std::bit_cast<std::uint64_t>(rounded) & 0x1FFu)];
            const double t2 = t * t;
            return (k[0] + k[1] * t) + t2 * ((k[2] + k[3] * t) + t2 * k[4]);
        }
        if (z < kOctaveMax)
        {
            // Cell = exponent and top four mantissa bits: 16 equal cells per
            // octave from 32 upward, each expanded about its midpoint. The
            // inverse width 2^(4 - e) comes from the exponent bits.
            const auto bits = std::bit_cast<std::uint64_t>(z);
            const auto& k = upper_[static_cast<size_t>((bits >> 48) - kFirstCellBits)];
            const double t = (z - k[5]) * std::bit_cast<double>(kInverseSixteenth - (bits & kExponentMask));
            const double t2 = t * t;
            return (k[0] + k[1] * t) + t2 * ((k[2] + k[3] * t) + t2 * k[4]);
        }
        return farField(z);
    }

private:
    static constexpr double kUniformMin = -32.0, kUniformMax = 32.0, kStep = 0.25, kInvStep = 4.0;
    static constexpr int kUniformNodes = 257;
    static constexpr int kOctaves = 15;                           // 32 .. 2^20
    static constexpr int kUpperCells = 16 * kOctaves;
    static constexpr double kOctaveMax = 1048576.0;
    static constexpr std::uint64_t kFirstCellBits = 0x4040u;      // bits(32.0) >> 48
    static constexpr std::uint64_t kExponentMask = 0x7FF0000000000000ull;
    static constexpr std::uint64_t kInverseSixteenth = std::uint64_t(2050) << 52; // 2^(4 - e)

    /** Degree-4 Taylor coefficients of omega about z, scaled to a step h. */
    static std::array<double, 5> expansion(double z, double h) noexcept
    {
        const double y = solve(z);
        // omega' = w/(1+w), omega'' = w/(1+w)^3, omega''' = w(1-2w)/(1+w)^5,
        // omega'''' = w(1-8w+6w^2)/(1+w)^7.
        const double u = 1.0 / (1.0 + y), u2 = u * u;
        const double d1 = y * u, d2 = d1 * u2, d3 = d2 * (1.0 - 2.0 * y) * u2;
        const double d4 = d1 * (1.0 - 8.0 * y + 6.0 * y * y) * u2 * u2 * u2;
        return { y, d1 * h, d2 * h * h / 2.0, d3 * h * h * h / 6.0, d4 * h * h * h * h / 24.0 };
    }

    /** Beyond the tables (z >= 2^20): two Newton steps from z - ln z. */
    DSPARK_NOINLINE static double farField(double z) noexcept
    {
        double y = z - std::log(z);
        for (int it = 0; it < 2; ++it)
            y -= (y + std::log(y) - z) * y / (1.0 + y);
        return y;
    }

    static double solve(double z) noexcept
    {
        double y = z < 1.0 ? std::exp(z) : z - std::log(z);
        for (int it = 0; it < 64; ++it)
        {
            const double step = (y + std::log(y) - z) * y / (1.0 + y);
            y -= step;
            if (!(std::abs(step) > 1e-17 * y)) break;
        }
        return y;
    }

    WrightOmega() noexcept
    {
        for (int i = 0; i < kUniformNodes; ++i)
            taylor_[static_cast<size_t>(i)] = expansion(kUniformMin + kStep * i, kStep);
        for (int c = 0; c < kUpperCells; ++c)
        {
            const double width = std::ldexp(1.0, 1 + c / 16);     // 2^e / 16, e = 5 + c/16
            const double mid = std::ldexp(1.0 + (c % 16) / 16.0, 5 + c / 16) + 0.5 * width;
            const auto e = expansion(mid, width);
            upper_[static_cast<size_t>(c)] = { e[0], e[1], e[2], e[3], e[4], mid };
        }
    }

    std::array<std::array<double, 5>, kUniformNodes> taylor_ {};
    std::array<std::array<double, 6>, kUpperCells> upper_ {};   ///< Taylor terms + midpoint.
};

/**
 * e^a to twice the working precision, as hi + lo (|a| below 700): a = k ln 2
 * + r with the product by ln 2 and the difference kept with their rounding
 * errors, the series of e^r summed in double-double arithmetic, and the
 * scaling by 2^k. Setup code: about 300 operations.
 */
inline void expTwice(double a, double& hi, double& lo) noexcept
{
    constexpr double kLn2Hi = 0.6931471805599453, kLn2Lo = 2.3190468138462996e-17;
    const double k = std::nearbyint(a * 1.4426950408889634);
    const double product = k * kLn2Hi;
    const double productLow = std::fma(k, kLn2Hi, -product) + k * kLn2Lo;
    const double first = a - product;
    const double firstPart = first - a;
    const double tail = ((a - (first - firstPart)) - (product + firstPart)) - productLow;
    const double rh = first + tail;
    const double rPart = rh - first;
    const double rl = (first - (rh - rPart)) + (tail - rPart);
    // 1 + r (1 + r / 2 (1 + r / 3 (...))): |r| < 0.35, so 22 terms pass 1e-31.
    double sh = 1.0, sl = 0.0;
    for (int n = 22; n >= 1; --n)
    {
        const double p = sh * rh;
        const double pl = std::fma(sh, rh, -p) + (sh * rl + sl * rh);
        const double q = p / n;
        const double ql = (std::fma(-q, static_cast<double>(n), p) + pl) / n;
        const double t = 1.0 + q;
        const double tl = (q - (t - 1.0)) + ql;
        sh = t + tl;
        sl = tl - (sh - t);
    }
    const int e = static_cast<int>(k);
    hi = std::ldexp(sh, e);
    lo = std::ldexp(sl, e);
}

/**
 * @brief Full-precision Wright omega tables for the diode roots (no
 *        correction step needed).
 *
 * Where the single-diode root s = A - omega(z) (z = ln c + A) is evaluated
 * in the form that keeps its rounding small, a polynomial of degree 9 in the
 * offset from the nearest table point reaches the root to double precision,
 * so neither an exponential nor a division is left on the sample's critical
 * path:
 * - z in [-40, 32): 289 nodes 0.25 apart. Below z = 0 a node expands
 *   -omega (the root is A plus the expansion), above it ln(omega) = z - omega
 *   (the root is the expansion minus ln c, free of the cancellation between
 *   A and a large omega). Both are the series about the node through the
 *   power 13 with the powers 10 to 13 folded into the nine kept, each
 *   replaced by the polynomial of lower degree closest to it over the node
 *   (Chebyshev economisation): cut at degree 9 the series would leave up
 *   to 4e-16 at the edges of the nodes nearest the branch points of omega,
 *   z = -1 +- i pi, and the fold leaves 1/512 of the first omitted power.
 *   Every node also expands R = 1 / (omega (1 + omega)) to degree 9 and
 *   H = 1 / (1 + omega)^2 to degree 6, which give the antiparallel pair's
 *   reverse-diode correction (see DiodePairRoot; the lower degrees serve the
 *   small corrections, the full ones the larger corrections of c > 1e-4).
 * - z in [32, 2^20): ln(omega) to degree 9 and R to degree 7 about the
 *   midpoint of each of 16 cells per octave, indexed by the exponent and
 *   leading mantissa bits of z.
 * What the constant term of a node or cell leaves out of -omega or
 * ln(omega) there is kept apart, to twice the working precision
 * (nodeLow(), from a double-double exponential and one Newton step): the
 * single diode with a constant that is not small reads its root as the
 * difference of two terms of similar size and needs both whole.
 * Truncation of R and H (largest omitted Taylor terms at the cell edges)
 * stays below 1e-16 for the pair's correction terms up to c = 0.02. The
 * coefficients follow from the Taylor recurrences of P = 1 / (1 + omega)
 * and R at each node, computed once per process on first use (instance(),
 * 103 KB); thread-safe and allocation free afterwards.
 */
class DiodeOmegaTable
{
public:
    static constexpr double kLo = -40.0, kHi = 32.0;
    static constexpr int kNodes = 289;
    static constexpr int kFirstLogNode = 160;                     ///< z = 0.
    static constexpr double kOctaveMax = 1048576.0;
    static constexpr int kCells = 16 * 15;                        ///< 32 .. 2^20.

    struct Node { double p[10]; double r[10]; double h[7]; };     ///< Scaled to t in [-0.5, 0.5].
    struct Cell { double mid; double p[10]; double r[8]; };

    [[nodiscard]] static const DiodeOmegaTable& instance() noexcept
    {
        static const DiodeOmegaTable table;
        return table;
    }

    /** Degree-9 polynomial without its constant term c[0] (Estrin scheme). */
    [[nodiscard]] static double poly9Rest(const double* c, double t) noexcept
    {
        const double t2 = t * t, t4 = t2 * t2;
        const double a = (c[1] * t) + t2 * (c[2] + c[3] * t);
        const double b = (c[4] + c[5] * t) + t2 * (c[6] + c[7] * t);
        return (a + t4 * b) + (t4 * t4) * (c[8] + c[9] * t);
    }

private:
    // The lookups return references into the table, which is immutable after
    // construction; only the two diode roots read them.
    template <FloatType U, typename Tr> friend class wdf::DiodePairRoot;
    template <FloatType U, typename Tr> friend class wdf::DiodeRoot;

    /**
     * Nearest node of z = z4 / 4 in [kLo, kHi) (z4 = 4 z, which the caller
     * forms as 4 x + 4 ln c with the same single rounding as z) and the
     * offset t = z4 - 4 z_k in [-0.5, 0.5]. The nodes are the multiples of
     * 1/4, so z4 + 1.5 * 2^52 rounds 4 z_k into the low mantissa bits (offset
     * by the node index of z = 0) and t is exact.
     */
    [[nodiscard]] const Node& node(double z4, double& t, int& index) const noexcept
    {
        constexpr double kRound = 6755399441055744.0 + kFirstLogNode;
        const double rounded = z4 + kRound;
        index = static_cast<int>(std::bit_cast<std::uint64_t>(rounded) & 0x1FFu);
        t = z4 - (rounded - kRound);
        return nodes_[static_cast<size_t>(index)];
    }

    /** Cell of z in [kHi, kOctaveMax) and the scaled offset in [-0.5, 0.5]. */
    [[nodiscard]] const Cell& cell(double z, double& t) const noexcept
    {
        const auto bits = std::bit_cast<std::uint64_t>(z);
        const Cell& c = cells_[static_cast<size_t>((bits >> 48) - kFirstCellBits)];
        t = (z - c.mid) * std::bit_cast<double>(kInverseSixteenth - (bits & kExponentMask));
        return c;
    }

    /** The same, with the factor between z and the scaled offset and what
     *  the p[0] of the cell leaves out of ln(omega) at its midpoint. */
    [[nodiscard]] const Cell& cell(double z, double& t, double& scale, double& low) const noexcept
    {
        const auto bits = std::bit_cast<std::uint64_t>(z);
        const auto index = static_cast<size_t>((bits >> 48) - kFirstCellBits);
        const Cell& c = cells_[index];
        scale = std::bit_cast<double>(kInverseSixteenth - (bits & kExponentMask));
        t = (z - c.mid) * scale;
        low = cellLow_[index];
        return c;
    }

    /** What the p[0] of the node leaves out of -omega or ln(omega) there. */
    [[nodiscard]] double nodeLow(int index) const noexcept { return nodeLow_[static_cast<size_t>(index)]; }

public:
    /** Degree-7 polynomial (Estrin scheme). */
    [[nodiscard]] static double poly7(const double* c, double t) noexcept
    {
        const double t2 = t * t;
        const double a = (c[0] + c[1] * t) + t2 * (c[2] + c[3] * t);
        const double b = (c[4] + c[5] * t) + t2 * (c[6] + c[7] * t);
        return a + (t2 * t2) * b;
    }

private:
    static constexpr std::uint64_t kFirstCellBits = 0x4040u;      // bits(32.0) >> 48
    static constexpr std::uint64_t kExponentMask = 0x7FF0000000000000ull;
    static constexpr std::uint64_t kInverseSixteenth = std::uint64_t(2050) << 52; // 2^(4 - e)
    static constexpr int kOrder = 14;
    static constexpr int kFolded = 13;        ///< Highest power folded into the nine kept.

    /**
     * Lowers the polynomial sum q[k] t^k (k up to @p from) to degree @p to
     * over |t| <= 1/2: each power above it, from the highest down, is
     * replaced by the lower-degree polynomial closest to it there (the power
     * minus its multiple of the Chebyshev polynomial of that degree), which
     * leaves 2^(1 - 2n) of its coefficient instead of 2^-n. The change of
     * the constant term is added to @p constant instead of q[0].
     */
    static void economise(double* q, int from, int to, double& constant) noexcept
    {
        double cheb[kOrder + 1][kOrder + 1] {};            // T_n in powers of u = 2 t
        cheb[0][0] = 1.0;
        cheb[1][1] = 1.0;
        for (int n = 1; n < from; ++n)
            for (int j = 0; j <= n + 1; ++j)
                cheb[n + 1][j] = (j > 0 ? 2.0 * cheb[n][j - 1] : 0.0) - cheb[n - 1][j];
        for (int n = from; n > to; --n)
        {
            // t^n = u^n / 2^n, and u^n = (T_n(u) - its lower powers) / 2^(n - 1).
            const double factor = -q[n] * std::ldexp(1.0, 1 - 2 * n);
            for (int j = n - 2; j >= 0; j -= 2)
            {
                const double change = factor * cheb[n][j] * std::ldexp(1.0, j);
                if (j == 0) constant += change;
                else q[j] += change;
            }
            q[n] = 0.0;
        }
    }

    static double solve(double z) noexcept
    {
        double y = z < 1.0 ? std::exp(z) : z - std::log(z);
        for (int it = 0; it < 64; ++it)
        {
            const double step = (y + std::log(y) - z) * y / (1.0 + y);
            y -= step;
            if (!(std::abs(step) > 1e-17 * y)) break;
        }
        return y;
    }

    /**
     * Taylor coefficients about z (unscaled, degree kOrder) of omega (w),
     * ln omega (l), R = 1 / (omega (1 + omega)) (r) and, when @p h is set,
     * 1 / (1 + omega)^2 (degree kOrder - 1). They follow from
     * P = 1 / (1 + omega), which obeys P' = -P^2 (1 - P), and R' = -R P (2 - P):
     * every product is of terms of one sign, so the coefficients keep their
     * relative accuracy where omega is large (the recurrence of omega itself,
     * omega' (1 + omega) = omega, cancels there).
     */
    static void series(double z, double* w, double* l, double* r, double* h = nullptr,
                       double* low = nullptr) noexcept
    {
        const double y = solve(z);
        if (low)
        {
            // What the doubles y and ln y leave out of omega and ln(omega)
            // (low[0], low[1]): ln y to twice the precision from e^(ln y),
            // then one Newton step of y + ln y = z with that residual.
            const double lh = std::log(y);
            double eh = 0.0, el = 0.0;
            expTwice(lh, eh, el);
            const double ll = ((y - eh) - el) / eh;
            const double sum = y + lh;
            const double part = sum - y;
            const double sumLow = (y - (sum - part)) + (lh - part);
            const double dy = -(((sum - z) + sumLow) + ll) * y / (1.0 + y);
            low[0] = dy;
            low[1] = ll + dy / y;
        }
        double p[kOrder + 1], q[kOrder + 1], pp[kOrder + 1], rp[kOrder + 1];
        p[0] = 1.0 / (1.0 + y);
        q[0] = y * p[0];                                   // 1 - P
        r[0] = p[0] / y;
        for (int n = 0; n < kOrder; ++n)
        {
            double s = 0.0;
            for (int j = 0; j <= n; ++j) s += p[j] * p[n - j];
            pp[n] = s;                                     // [P^2]_n
            s = 0.0;
            for (int j = 0; j <= n; ++j) s += pp[j] * q[n - j];
            p[n + 1] = -s / (n + 1);
            q[n + 1] = -p[n + 1];
            s = 0.0;
            for (int j = 0; j <= n; ++j) s += r[j] * p[n - j];
            rp[n] = s;                                     // [R P]_n
            s = rp[n];
            for (int j = 0; j <= n; ++j) s += rp[j] * q[n - j];
            r[n + 1] = -s / (n + 1);
        }
        if (h)
            for (int n = 0; n < kOrder; ++n) h[n] = pp[n];
        w[0] = y;
        l[0] = std::log(y);
        for (int n = 0; n < kOrder; ++n)
        {
            w[n + 1] = q[n] / (n + 1);
            l[n + 1] = p[n] / (n + 1);
        }
    }

    DiodeOmegaTable() noexcept
    {
        double w[kOrder + 1], l[kOrder + 1], r[kOrder + 1], hh[kOrder + 1], low[2], q[kOrder + 1];
        for (int i = 0; i < kNodes; ++i)
        {
            series(kLo + 0.25 * i, w, l, r, hh, low);
            Node& n = nodes_[static_cast<size_t>(i)];
            // -omega or ln(omega) in t = 4 (z - z_k), with the powers 10 to
            // 13 folded into the nine kept: cut at degree 9, the series
            // would leave up to 4e-16 at the edges of the nodes nearest the
            // branch points of omega (z = -1 +- i pi).
            double scale = 1.0;
            for (int k = 0; k <= kFolded; ++k, scale *= 0.25)
                q[k] = (i < kFirstLogNode ? -w[k] : l[k]) * scale;
            double shift = 0.0;
            economise(q, kFolded, 9, shift);
            for (int k = 0; k < 10; ++k) n.p[k] = q[k];
            nodeLow_[static_cast<size_t>(i)] = (i < kFirstLogNode ? -low[0] : low[1]) + shift;
            scale = 1.0;
            for (int k = 0; k < 10; ++k, scale *= 0.25) n.r[k] = r[k] * scale;
            const double u = 1.0 / (1.0 + w[0]);
            n.h[0] = u * u;
            n.h[1] = -2.0 * w[0] * u * u * u * u * 0.25;
            scale = 0.0625;
            for (int k = 2; k < 7; ++k, scale *= 0.25) n.h[k] = hh[k] * scale;
        }
        for (int c = 0; c < kCells; ++c)
        {
            const double width = std::ldexp(1.0, 1 + c / 16);     // 2^e / 16, e = 5 + c/16
            const double mid = std::ldexp(1.0 + (c % 16) / 16.0, 5 + c / 16) + 0.5 * width;
            series(mid, w, l, r, nullptr, low);
            cellLow_[static_cast<size_t>(c)] = low[1];
            Cell& cl = cells_[static_cast<size_t>(c)];
            cl.mid = mid;
            double scale = 1.0;
            for (int k = 0; k < 10; ++k, scale *= width) cl.p[k] = l[k] * scale;
            scale = 1.0;
            for (int k = 0; k < 8; ++k, scale *= width) cl.r[k] = r[k] * scale;
        }
    }

    std::array<Node, kNodes> nodes_ {};
    std::array<Cell, kCells> cells_ {};
    std::array<double, kNodes> nodeLow_ {};     ///< Remainders of the p[0] (single diode).
    std::array<double, kCells> cellLow_ {};
};

/**
 * @brief Exact low-drive roots of the normalised diode equations by series
 *        reversion (no transcendental function and no division per sample).
 *
 * With the drive normalised by the linear slope, the antiparallel pair reads
 * y = s + kappa (sinh s - s) (y = x / (1 + 2c), kappa = 2c / (1 + 2c)) and the
 * single diode y = s + kappa (e^s - 1 - s) (y = x / (1 + c), kappa = c / (1 + c)).
 * Lagrange inversion gives s = y + sum_k b_k y^k, where every b_k is a
 * polynomial in kappa with rational coefficients (exact for any kappa; printed
 * here to double precision). The roots keep the terms through y^15 and use
 * them while |y| < Y, with Y set so that each of the two first omitted terms
 * stays below 1e-17 relative to y (Y <= 6). Verified against 50-digit roots
 * for kappa from 1e-30 to 0.04: worst truncation error 2e-17 relative; above
 * kappa = 0.04 the series is not used. For the default 1N4148 constants
 * behind 707 ohm Y is 1.17 (pair) and 1.09 (single), incident waves up to
 * 53 and 49 mV: the low-level region where the omega estimate and its
 * correction step cost the most relative to the signal. For the pair,
 * pairLong() carries the same reversion through y^35 under the same rule
 * (first omitted terms y^37 and y^39), which reaches 1.2 to 2.5 units of nVt
 * further for kappa from 2e-4 to 0.04 (see DiodePairRoot).
 */
struct DiodeSeries
{
    /// Coefficients of y^3, y^5, ..., y^39 (pair), each a polynomial
    /// c1 kappa + c2 kappa^2 + ... of degree (k - 1) / 2, lowest power first:
    /// pair() reads them through y^19, pairLong() all of them.
    static constexpr double kPairCoefficients[190] = {
        -0.16666666666666666, -0.008333333333333333, 0.08333333333333333,
        -0.0001984126984126984, 0.011111111111111112, -0.05555555555555555,
        -2.7557319223985893e-06, 0.000677910052910053, -0.01273148148148148,
        0.04243827160493827, -2.505210838544172e-08, 2.535273368606702e-05,
        -0.0013326719576719577, 0.01404320987654321, -0.03510802469135803,
        -1.6059043836821613e-10, 6.555301694190583e-07, -8.616255144032922e-05,
        0.0021347736625514403, -0.015303497942386831, 0.030606995884773662,
        -7.647163731819816e-13, 1.2516877596242676e-08, -3.901925528909656e-06,
        0.00020154810895551636, -0.003085684891240447, 0.01661522633744856,
        -0.0276920438957476, -2.8114572543455206e-15, 1.8420105639020982e-10,
        -1.3265171685310574e-07, 1.3412058788629556e-05, -0.00038840127743484227,
        0.004199620627572016, -0.01803037122770919, 0.025757673182441702,
        -8.22063524662433e-18, 2.154825793386156e-12, -3.537694774007972e-09,
        6.748958893377618e-07, -3.461694730904479e-05, 0.0006657831313824112,
        -0.005497982872275568, 0.019581856805365037, -0.024477321006706295,
        -1.9572941063391263e-20, 2.052328564378316e-14, -7.63054132550079e-11,
        2.6857233826151793e-08, -2.3437074499064687e-06, 7.502594069606581e-05,
        -0.0010560215634321858, 0.007007507328205631, -0.021295269275834475,
        0.023661410306482753, -3.868170170630684e-23, 1.6224188783272867e-16,
        -1.3615508265959045e-12, 8.718744862893881e-10, -1.2608500024975103e-07,
        6.444848572066936e-06, -0.00014466849341383163, 0.0015856217815772857,
        -0.008759798795847214, 0.02319379350200681, -0.02319379350200681,
        -6.446950284384474e-26, 1.0816170984167235e-18, -2.0454899596721922e-14,
        2.3622256014636148e-11, -5.560999346458007e-09, 4.4139111094536364e-07,
        -1.5151959626692243e-05, 0.0002566754712713968, -0.0022861625245067354,
        0.010791464659987286, -0.025300563078439094, 0.023000511889490084,
        -9.183689863795546e-29, 6.163067369443176e-21, -2.624379797399636e-16,
        5.435863409317769e-13, -2.0576040831492097e-10, 2.486734120772481e-08,
        -1.2686390623297178e-06, 3.18473810836589e-05, -0.0004280069091063092,
        0.0031952644376229754, -0.013144553833792228, 0.02763993138172911,
        -0.023033276151440924, -1.1309962886447716e-31, 3.0359947054677823e-23,
        -2.90983516988768e-18, 1.0773119729318394e-14, -6.500516401322334e-12,
        1.1793526431273427e-09, -8.760273119582742e-08, 3.166099301127302e-06,
        -6.155872734178536e-05, 0.0006803442075780342, -0.004357681886385206,
        0.01586718791319178, -0.03023826647467491, 0.023260204980519162,
        -1.216125041553518e-34, 1.305804281413749e-25, -2.8164494022894016e-20,
        1.8598856226380004e-16, -1.7781683941160714e-13, 4.792391011366623e-11,
        -5.105092927875956e-09, 2.6013585118536054e-07, -7.1081371288621065e-06,
        0.00011151636823255261, -0.0010411731217924711, 0.005826556292754563,
        -0.019014342187516006, 0.033124532770107074, -0.02366038055007648,
        -1.151633562077195e-37, 4.946228446941997e-28, -2.400577152778124e-22,
        2.823876221983734e-18, -4.259542492017361e-15, 1.692153810932095e-12,
        -2.555530089128816e-10, 1.8074100859420624e-08, -6.795666888993754e-07,
        1.4701125115609984e-05, -0.0001918618460532977, 0.0015450912017424953,
        -0.0076648654207181945, 0.022648764847580313, -0.03633078320260323,
        0.024220522135068822, -9.67759295863189e-41, 1.6625978069690206e-30,
        -1.815631965979403e-24, 3.80152886681811e-20, -9.020063567411856e-17,
        5.251035194309445e-14, -1.1143997834640651e-11, 1.0808330297892926e-09,
        -5.503300371729764e-08, 1.6071863715423982e-06, -2.848117077960185e-05,
        0.0003165434147228053, -0.002235380660678097, 0.009947103846933474,
        -0.026842037472040975, 0.039892624693054525, -0.02493289043315908,
        -7.265460179153071e-44, 4.992786214815564e-33, -1.2268004484704633e-26,
        4.569781510266617e-22, -1.7020604798406413e-18, 1.4456351083934227e-15,
        -4.281836851327378e-13, 5.641424476816121e-11, -3.842042611039085e-09,
        1.490510231519924e-07, -3.511190846188237e-06, 5.231861394876127e-05,
        -0.0005044424531680884, 0.0031658955995590354, -0.012761233058852326,
        0.03167578795742484, -0.04384969153807692, 0.02579393619886878,
        -4.902469756513544e-47, 1.3475806253306053e-35, -7.45026815204885e-29,
        4.9358060995432006e-24, -2.88161452928429e-20, 3.5590142851266987e-17,
        -1.463339766188361e-14, 2.599510558910379e-12, -2.3448394274072986e-10,
        1.1933073304746268e-08, -3.6768172678110105e-07, 7.190746000583771e-06,
        -9.19608815095438e-05, 0.0007807836711677919, -0.004403321514323129,
        0.016210944378911955, -0.03724307230981259, 0.04824614786994889,
        -0.02680341548330494
    };
    /// Coefficients of y^2, y^3, ..., y^17 (single), degree k - 1 in kappa.
    static constexpr double kSingleCoefficients[136] = {
        -0.5, -0.16666666666666666, 0.5, -0.041666666666666664, 0.4166666666666667, -0.625,
        -0.008333333333333333, 0.20833333333333334, -0.875, 0.875, -0.001388888888888889,
        0.07777777777777778, -0.6805555555555556, 1.75, -1.3125, -0.0001984126984126984,
        0.02361111111111111, -0.38055555555555554, 1.875, -3.4375, 2.0625,
        -2.48015873015873e-05, 0.006101190476190476, -0.16927083333333334, 1.4131944444444444,
        -4.7265625, 6.703125, -3.3515625, -2.7557319223985893e-06, 0.0013806216931216931,
        -0.06320271164021164, 0.8349729938271605, -4.510127314814815, 11.296006944444445,
        -13.033854166666666, 5.5859375, -2.755731922398589e-07, 0.0002788800705467372,
        -0.020479497354497354, 0.4098324514991182, -3.340528549382716, 13.091782407407408,
        -26.067708333333332, 25.322916666666668, -9.49609375, -2.505210838544172e-08,
        5.09810405643739e-05, -0.005889550264550265, 0.173233024691358, -2.0400270061728394,
        11.696238425925927, -35.63113425925926, 58.703125, -49.20703125, 16.40234375,
        -2.08767569878681e-09, 8.521892202447758e-06, -0.001527617026748971,
        0.06461548353909465, -1.066552533436214, 8.568155221193416, -37.354499421296296,
        92.58211805555555, -129.85188802083334, 95.68033854166667, -28.7041015625,
        -1.6059043836821613e-10, 1.3131480145369033e-06, -0.0003617044285447063,
        0.02164432227366255, -0.4897392618312757, 5.3492557227366255, -31.97756076388889,
        111.56864872685185, -232.33148871527777, 283.3610026041667, -186.20865885416666,
        50.7841796875, -1.1470745597729725e-11, 1.8775316394364013e-07, -7.891503613229804e-05,
        0.006597468567954679, -0.20118927055776015, 2.9222029320987652, -23.24848457708701,
        109.43574322089947, -316.70893012152777, 567.4930555555555, -611.8284505208334,
        362.744140625, -90.68603515625, -7.647163731819816e-13, 2.5045225938083082e-08,
        -1.598327726799949e-05, 0.0018487619924045188, -0.07498005246746567,
        1.4237209155460024, -14.74556337877229, 90.91360812573487, -350.53111479690256,
        863.9490632233797, -1356.6630859375, 1309.909396701389, -707.35107421875,
        163.23486328125, -4.779477332387385e-14, 3.131417958633567e-09,
        -3.0234750403091175e-06, 0.0004801716850276291, -0.025626671262063385,
        0.627620408914083, -8.319817679867006, 65.73195370867933, -327.86206375400155,
        1065.7648367099662, -2282.6262828685617, 3187.009562174479, -2785.194854736328,
        1380.6948852539062, -295.8631896972656, -2.8114572543455206e-15,
        3.6844990755374353e-10, -5.368700378931033e-07, 0.00011636075093390529,
        -0.008102164118286442, 0.2531695617639899, -4.238021926775227, 42.188908018022104,
        -266.3720860467246, 1109.0705859382176, -3106.9867568243117, 5875.001434326172,
        -7379.047108968099, 5888.257598876953, -2697.576141357422, 539.5152282714844
    };

    static constexpr double kTolerance = 1e-17;   ///< Relative size of the first omitted terms.
    static constexpr double kMaxKappa = 0.04;
    static constexpr double kMaxY = 6.0;

    /** Fills b[0..6] (y^3 .. y^15) and returns the series limit Y (0 = unused). */
    static double pair(double kappa, double* b) noexcept
    {
        if (!(kappa > 0.0 && kappa <= kMaxKappa)) return 0.0;
        double tail[2] {};
        int offset = 0;
        for (int i = 0; i < 9; ++i)
        {
            const int degree = i + 1;
            double v = 0.0;
            for (int j = degree - 1; j >= 0; --j) v = v * kappa + kPairCoefficients[offset + j];
            offset += degree;
            if (i < 7) b[i] = v * kappa;
            else tail[i - 7] = v * kappa;
        }
        return limit(tail, 17);
    }

    /** Fills b[0..16] (y^3 .. y^35) and returns its series limit Y (0 = unused):
     *  the same reversion carried further, for the drives just above pair()'s
     *  limit at larger kappa. */
    static double pairLong(double kappa, double* b) noexcept
    {
        if (!(kappa > 0.0 && kappa <= kMaxKappa)) return 0.0;
        double tail[2] {};
        int offset = 0;
        for (int i = 0; i < 19; ++i)
        {
            const int degree = i + 1;
            double v = 0.0;
            for (int j = degree - 1; j >= 0; --j) v = v * kappa + kPairCoefficients[offset + j];
            offset += degree;
            if (i < 17) b[i] = v * kappa;
            else tail[i - 17] = v * kappa;
        }
        return limit(tail, 37);
    }

    /** Fills d[0..13] (y^2 .. y^15) and returns the series limit Y (0 = unused). */
    static double single(double kappa, double* d) noexcept
    {
        if (!(kappa > 0.0 && kappa <= kMaxKappa)) return 0.0;
        double tail[2] {};
        int offset = 0;
        for (int i = 0; i < 16; ++i)
        {
            const int degree = i + 1;
            double v = 0.0;
            for (int j = degree - 1; j >= 0; --j) v = v * kappa + kSingleCoefficients[offset + j];
            offset += degree;
            if (i < 14) d[i] = v * kappa;
            else tail[i - 14] = v * kappa;
        }
        return limit(tail, 16);
    }

private:
    /** Largest |y| at which both omitted terms (powers p, p + step) stay
     *  below kTolerance relative to y. */
    static double limit(const double* tail, int firstPower) noexcept
    {
        const int step = (firstPower & 1) != 0 ? 2 : 1;   // the pair has odd powers only
        double y = kMaxY;
        for (int i = 0; i < 2; ++i)
        {
            const double t = std::abs(tail[i]);
            if (t > 0.0)
                y = std::min(y, std::pow(kTolerance / t, 1.0 / (firstPower + step * i - 1)));
        }
        return y;
    }
};

/**
 * @brief Local expansions of the antiparallel pair's root (built per
 *        instance), for the drives between the low-drive series and those
 *        where the reverse diode is a small correction.
 *
 * In that range the root s(x) of s + 2c sinh(s) = x has no small parameter,
 * but it is analytic: about a point of the curve its Taylor series in x
 * converges up to the nearest branch point, the image in x of a zero of
 * 1 + 2c cosh(s) (at pi from the real axis for c below 1/2 and further
 * beyond). The range is cut into cells, eight per octave of x plus an offset
 * chosen from the branch point, so that a cell is found from the exponent
 * and leading mantissa bits of that sum. Each cell holds the root at its
 * centre to twice the working precision and the powers 1 .. 16 of the
 * distance to it: no table lookup beyond the cell, no division and no
 * transcendental function per sample.
 *
 * The cells are derived from anchors, expansions about fixed points
 * s_m = k / 4 whose sinh is tabulated to twice the working precision, so
 * that x_m = s_m + 2c sinh(s_m) is known to about 1e-32. An anchor's
 * coefficients follow from g' (1 + 2c cosh g) = 1 with the series of cosh g
 * and sinh g carried along (C' = S g', S' = C g'); it keeps the powers
 * 1 .. 32 and reaches as far as its first two omitted terms stay below
 * 1e-17 of half its root, with the offset no larger than x_m / 2 and the
 * root no smaller than s_m / 2. A cell is the Taylor shift of the anchor
 * nearest its centre, which must contain the whole cell; its own first two
 * omitted terms are checked the same way. The same recurrence about the
 * origin, in the variable of DiodeSeries, gives the low-drive series for the
 * constants its tabulated polynomials do not cover.
 *
 * The single diode, s + c (e^s - 1) = x, is served the same way on both
 * sides of the origin for the constants beyond those of DiodeSeries: its
 * curve has one branch point above the real axis, at -(ln c + 1 + c) + i pi,
 * the anchors are expansions about s_m = k / 4 with e^(s_m) tabulated to
 * twice the working precision (g' (1 + c e^g) = 1 with E = e^g carried
 * along, E' = E g'), and the cells run from the end of the series about the
 * origin to the drives of two fixed roots, -1 and 1.5, beyond which the
 * omega table answers (singleTableDrives()).
 */
struct DiodeLocal
{
    static constexpr int kTerms = 32;             ///< Powers 1 .. 32 of an anchor.
    static constexpr int kCellTerms = 16;         ///< Powers 1 .. 16 of a cell.
    static constexpr int kMaxCells = 56;
    static constexpr int kCentreCount = 24;
    static constexpr double kTolerance = 1e-17;   ///< Relative size of the first omitted terms.
    /// Offset of the cell grid over the distance of the branch point from the
    /// real axis: with it a cell's half-width stays below 0.07 of the
    /// distance from its centre to the branch point.
    static constexpr double kOffset = 1.5;
    static constexpr int kKeyShift = 49;          ///< Exponent and three mantissa bits.

    /// s, sinh(s) as a double and its remainder, cosh(s).
    struct Centre { double s, sinhHi, sinhLo, cosh; };
    static constexpr Centre kCentres[kCentreCount] = {
        { 0.25, 0.2526123168081683, -5.550352555023456e-18, 1.0314130998795732 },
        { 0.5, 0.5210953054937474, -2.3328183476404597e-17, 1.1276259652063807 },
        { 0.75, 0.82231673193583, -1.1335882233129953e-17, 1.2946832846768448 },
        { 1.0, 1.1752011936438014, 7.849672142285669e-17, 1.5430806348152437 },
        { 1.25, 1.6019190803008256, 4.943508499611119e-19, 1.8884238771610158 },
        { 1.5, 2.1292794550948173, 1.8859829935660394e-16, 2.352409615243247 },
        { 1.75, 2.7904143662776426, 4.462297999014824e-17, 2.9641883097280877 },
        { 2.0, 3.6268604078470186, 1.9291196578353674e-16, 3.7621956910836314 },
        { 2.25, 4.691168305898331, -2.0060583610320798e-16, 4.796567530460195 },
        { 2.5, 6.0502044810397875, -1.5266669624477375e-16, 6.132289479663686 },
        { 2.75, 7.789352011490732, 4.362551404194279e-17, 7.853279872697439 },
        { 3.0, 10.017874927409903, -6.97789774734877e-16, 10.067661995777765 },
        { 3.25, 12.87578285468067, -2.578015334190764e-16, 12.914557062512392 },
        { 3.5, 16.542627287634996, 1.388991286007725e-15, 16.572824671057315 },
        { 3.75, 21.248782127103386, 1.0986521939233124e-15, 21.272299872959398 },
        { 4.0, 27.289917197127753, -7.633630011960991e-16, 27.308232836016487 },
        { 4.25, 35.04557405638943, 2.7362100014654743e-15, 35.05983829029843 },
        { 4.5, 45.003011151991785, 3.6708731211657254e-16, 45.014120148530026 },
        { 4.75, 57.78781641599227, 1.1560809076983308e-15, 57.79646811119539 },
        { 5.0, 74.20321057778875, 6.687957117824193e-15, 74.20994852478785 },
        { 5.25, 95.2805104701154, 5.1967560907177715e-15, 95.28575798851459 },
        { 5.5, 122.34392274639096, 3.0463153468293817e-15, 122.34800951782942 },
        { 5.75, 157.09373875244884, 1.3507577081271248e-15, 157.09692153324536 },
        { 6.0, 201.71315737027922, 8.575934656268528e-15, 201.7156361224559 },
    };

    static constexpr int kSingleCentreCount = 18;
    /// Cells an instance of the single diode has room for: its cells run
    /// between two roots that do not depend on the constant, and 3000
    /// constants up to 1e5 need at most 24.
    static constexpr int kSingleMaxCells = 32;
    /// s and e^s as a double and its remainder: the centres of the single
    /// diode's anchors, on both sides of the origin, as far as its cells go.
    struct ExpCentre { double s, expHi, expLo; };
    static constexpr ExpCentre kSingleCentres[kSingleCentreCount] = {
        { -2.0, 0.1353352832366127, -1.042381423288669e-17 },
        { -1.75, 0.17377394345044514, -1.2999409968044821e-17 },
        { -1.5, 0.22313016014842982, 1.0887723699042263e-17 },
        { -1.25, 0.2865047968601901, 8.479830077607644e-18 },
        { -1.0, 0.36787944117144233, -1.2428753672788363e-17 },
        { -0.75, 0.4723665527410147, 1.7984004434373214e-17 },
        { -0.5, 0.6065306597126334, -6.593178415491414e-19 },
        { -0.25, 0.7788007830714049, -1.0231869534531498e-17 },
        { 0.25, 1.2840254166877414, 8.968972781793724e-17 },
        { 0.5, 1.6487212707001282, -4.731568479435833e-17 },
        { 0.75, 2.117000016612675, -1.1571006249440234e-16 },
        { 1.0, 2.718281828459045, 1.4456468917292502e-16 },
        { 1.25, 3.4903429574618414, -4.604261945372796e-17 },
        { 1.5, 4.4816890703380645, 3.0481759556536343e-16 },
        { 1.75, 5.754602676005731, -2.8457593299092424e-16 },
        { 2.0, 7.38905609893065, -1.7971139497839148e-16 },
        { 2.25, 9.487735836358526, -5.218563277933494e-16 },
        { 2.5, 12.182493960703473, 2.0334002173348147e-16 },
    };

    struct Cell
    {
        double centre = 0.0;                   ///< x the cell's series is about.
        double rootHi = 0.0, rootLo = 0.0;     ///< Root there, to twice the precision.
        double a[kCellTerms] {};               ///< Coefficients of the powers 1 .. 16 of x - centre.
    };

private:
    /// An anchor: the root about a fixed point of the curve.
    struct Expansion
    {
        double xHi = 0.0, xLo = 0.0;   ///< x_m, to twice the precision.
        double width = 0.0;            ///< Half-width in x.
        double s = 0.0;                ///< s_m.
        double a[kTerms + 1] {};       ///< a[k] = g_k times the half-width to the k (a[0] = 0).
    };

public:
    /** Cell of the sum of x and the grid's offset. */
    [[nodiscard]] static DSPARK_WDF_INLINE std::uint64_t key(double shifted) noexcept
    {
        return std::bit_cast<std::uint64_t>(shifted) >> kKeyShift;
    }

    /** Root at x from the cell that holds it. */
    [[nodiscard]] static DSPARK_WDF_INLINE double evaluate(const Cell& cell, double x) noexcept
    {
        const double d = x - cell.centre;
        const double* a = cell.a;
        const double d2 = d * d, d4 = d2 * d2;
        const double q0 = (a[0] + a[1] * d) + d2 * (a[2] + a[3] * d);
        const double q1 = (a[4] + a[5] * d) + d2 * (a[6] + a[7] * d);
        const double q2 = (a[8] + a[9] * d) + d2 * (a[10] + a[11] * d);
        const double q3 = (a[12] + a[13] * d) + d2 * (a[14] + a[15] * d);
        return cell.rootHi + (cell.rootLo + d * ((q0 + d4 * q1) + (d4 * d4) * (q2 + d4 * q3)));
    }

    /**
     * Low-drive series for any constant: fills b[0..16] with the coefficients
     * of y^3 .. y^35 of the root in y = x / (1 + 2c) (kappa = 2c / (1 + 2c),
     * the form of DiodeSeries::pairLong()) and returns the limit in y, set by
     * the terms y^37 and y^39 in the same way.
     */
    static double origin(double kappa, double* b) noexcept
    {
        double g[kMaxOrder] {};
        series(1.0, -kappa, 1.0, 0.0, 39, g);
        for (int i = 0; i < 17; ++i) b[i] = g[2 * i + 2];
        double y = 6.0;
        for (int k = 37; k <= 39; k += 2)
        {
            const double tail = std::abs(g[k - 1]);
            if (tail > 0.0) y = std::min(y, std::pow(kTolerance / tail, 1.0 / (k - 1)));
        }
        return y;
    }

    /**
     * A build in progress: start() lays the grid out for a constant and a
     * range of drives, and every step() adds one cell, usable at once, so
     * that a running circuit can spread the work over its samples (an anchor
     * costs about 1700 multiply-adds, a cell about 500 more).
     */
    struct Builder
    {
        double offset = 0.0;                   ///< Offset of the cell grid.
        std::uint64_t firstKey = 0;            ///< Key of the first cell.
        int count = 0;                         ///< Cells built so far.
        bool done = true;                      ///< Nothing left to build.
        double twoC = 0.0, from = 0.0, to = 0.0;
        double c = 0.0;                        ///< Constant of the single diode.
        bool single = false;                   ///< Single diode instead of the pair.
        bool clipTop = false;                  ///< The last cell ends at `to`.
        int capacity = kMaxCells;              ///< Cells the caller has room for.
        int nearest = 0;                       ///< Anchor nearest the last cell.
        int anchorCentre[2] = { -1, -1 };      ///< Centres of the two anchors held.
        double anchorWidth[2] = { 0.0, 0.0 };
        Expansion anchor[2];
    };

    /** Lays the grid out for the drives [from, to) of the constant c. */
    static void start(Builder& b, double c, double from, double to) noexcept
    {
        b.count = 0;
        b.nearest = 0;
        b.anchorCentre[0] = b.anchorCentre[1] = -1;
        b.twoC = 2.0 * c;
        b.c = c;
        b.single = false;
        b.clipTop = false;
        b.capacity = kMaxCells;
        b.from = from;
        b.to = to;
        // Branch point nearest the positive real axis: the image of
        // acosh(1 / 2c) + i pi for c < 1/2, of i acos(-1 / 2c) beyond.
        double branchRe = 0.0, branchIm = std::numbers::pi;
        if (c < 0.5)
        {
            const double root = std::sqrt(1.0 - 4.0 * c * c);
            branchRe = std::log((1.0 + root) / b.twoC) - root;
        }
        else
        {
            branchIm = std::acos(-0.5 / c) + std::sqrt(4.0 * c * c - 1.0);
        }
        // x + offset stays positive, at least a quarter of x.
        b.offset = std::max(kOffset * branchIm - branchRe, -0.75 * from);
        b.firstKey = 0;
        b.done = !(to > from) || !(from + b.offset > 0.0);
        if (!b.done) b.firstKey = key(from + b.offset);
    }

    /**
     * Adds the next cell to cells[] and sets @p end to the drive below which
     * the cells built so far answer: it may pass the end of the range within
     * the last cell. Returns false, leaving @p end alone, when no cell could
     * be added (the range is covered, or the anchors or the cells ran out);
     * the builder is done then, as it is after the last cell that fits.
     */
    static bool step(Builder& b, Cell* cells, double& end) noexcept
    {
        if (b.done) return false;
        b.done = true;
        const std::uint64_t bits = (b.firstKey + static_cast<std::uint64_t>(b.count)) << kKeyShift;
        const double low = std::bit_cast<double>(bits);
        const double high = std::bit_cast<double>(bits + (std::uint64_t(1) << kKeyShift));
        if (!(low - b.offset < b.to)) return false;
        // The part of the cell the root uses: the first one starts at
        // `from`, and its series is about the middle of that part.
        const double lowX = std::max(low - b.offset, b.from);
        const double highX = b.clipTop ? std::min(high - b.offset, b.to) : high - b.offset;
        const double half = 0.5 * (highX - lowX);
        const double centre = lowX + half;
        const int centres = b.single ? kSingleCentreCount : kCentreCount;
        // Nearest anchor in x (their x grows with s), then the one of it
        // and its upper neighbour that leaves more room around the cell.
        while (b.nearest + 1 < centres
               && std::abs(centreX(b, b.nearest + 1) - centre) < std::abs(centreX(b, b.nearest) - centre))
            ++b.nearest;
        int slot = anchorOf(b, b.nearest);
        double room = b.anchorWidth[slot] - std::abs(centre - b.anchor[slot].xHi);
        if (room < 2.0 * half && b.nearest + 1 < centres)
        {
            const int upper = anchorOf(b, b.nearest + 1);
            const double upperRoom = b.anchorWidth[upper] - std::abs(centre - b.anchor[upper].xHi);
            slot = anchorOf(b, b.nearest);   // still held: only the lower centre is replaced
            if (upperRoom > room) { slot = upper; room = upperRoom; }
        }
        if (!(room >= 1.001 * half)) return false;
        if (!shift(b.anchor[slot], centre, half, cells[b.count])) return false;
        ++b.count;
        // Just below the cell's upper edge, so that the rounded sum of a
        // drive under it and the offset stays inside the cell.
        end = b.clipTop && !(high - b.offset < b.to) ? b.to : high * (1.0 - 0x1p-40) - b.offset;
        b.done = b.count >= b.capacity;
        return true;
    }

    /**
     * Every cell for the drives [from, to) of the constant c at once. Fills
     * cells[] and the grid's @p offset and @p firstKey, and returns the number
     * of cells built (0: none). @p end receives the drive below which they
     * answer, as step() leaves it (-1 without cells).
     */
    static int cells(double c, double from, double to, Cell* cells, double& offset,
                     std::uint64_t& firstKey, double& end) noexcept
    {
        Builder b;
        start(b, c, from, to);
        end = -1.0;
        while (step(b, cells, end)) {}
        offset = b.offset;
        firstKey = b.firstKey;
        return b.count;
    }

    // ---- Single diode: s + c (e^s - 1) = x, either sign --------------------

    /**
     * Low-drive series of the single diode for any constant: fills q[0..31]
     * with the coefficients of y^2 .. y^33 of the root in y = x / (1 + c)
     * (kappa = c / (1 + c)) and returns the limit in |y|, set by the terms
     * y^34 and y^35.
     */
    static double originSingle(double kappa, double* q) noexcept
    {
        double g[kMaxOrder] {};
        seriesSingle(1.0, -kappa, 1.0, 35, g);
        for (int i = 0; i < 32; ++i) q[i] = g[i + 1];
        double y = 6.0;
        for (int k = 34; k <= 35; ++k)
        {
            const double tail = std::abs(g[k - 1]);
            if (tail > 0.0) y = std::min(y, std::pow(kTolerance / tail, 1.0 / (k - 1)));
        }
        return y;
    }

    /// Roots between which the single diode's table root is not used (see
    /// singleTableDrives()).
    static constexpr double kSingleTableReverse = -1.0, kSingleTableForward = 1.5;

    /**
     * Drives from which the single diode reads its root from the omega table
     * for a constant beyond those of DiodeSeries: the images of the roots
     * kSingleTableReverse and kSingleTableForward. Outside them the closed
     * form, whose two terms are both known to twice the working precision,
     * no longer cancels.
     */
    static void singleTableDrives(double c, double& reverse, double& forward) noexcept
    {
        reverse = kSingleTableReverse + c * std::expm1(kSingleTableReverse);
        forward = kSingleTableForward + c * std::expm1(kSingleTableForward);
    }

    /** Offset of the single diode's cell grid: its branch point lies at
     *  -(ln c + 1 + c) + i pi, and the cells run on both sides of it. */
    [[nodiscard]] static double singleOffset(double c) noexcept
    {
        return kOffset * std::numbers::pi + (std::log(c) + 1.0 + c);
    }

    /**
     * Lays the grid out for the drives [from, to) of the single diode of
     * constant c, of one sign, on the grid of @p offset. With @p clipTop the
     * last cell ends at @p to (the range below the origin, which must not
     * reach into the series). @p capacity is the room left for cells.
     */
    static void startSingle(Builder& b, double c, double from, double to, double offset,
                            bool clipTop, int capacity) noexcept
    {
        b.count = 0;
        b.nearest = 0;
        b.anchorCentre[0] = b.anchorCentre[1] = -1;
        b.twoC = 2.0 * c;
        b.c = c;
        b.single = true;
        b.clipTop = clipTop;
        b.capacity = capacity;
        b.from = from;
        b.to = to;
        b.offset = offset;
        b.firstKey = 0;
        b.done = !(to > from) || !(from + offset > 0.0) || capacity <= 0;
        if (!b.done) b.firstKey = key(from + offset);
    }

private:
    static constexpr int kMaxOrder = 40;

    /** Slot of the builder that holds the anchor of centre m, built if it
     *  is not there (in place of the lower of the two centres held). */
    static int anchorOf(Builder& b, int m) noexcept
    {
        for (int i = 0; i < 2; ++i)
            if (b.anchorCentre[i] == m) return i;
        const int slot = b.anchorCentre[0] <= b.anchorCentre[1] ? 0 : 1;
        b.anchorWidth[slot] = b.single ? buildSingle(b.c, kSingleCentres[m], b.anchor[slot])
                                       : build(b.twoC, kCentres[m], b.anchor[slot]);
        b.anchorCentre[slot] = m;
        return slot;
    }

    [[nodiscard]] static double centreX(const Builder& b, int m) noexcept
    {
        return b.single ? kSingleCentres[m].s + b.c * (kSingleCentres[m].expHi - 1.0)
                        : kCentres[m].s + b.twoC * kCentres[m].sinhHi;
    }

    /**
     * g_1 .. g_order of the inverse of a function f with f' = 1 / invD at the
     * point and f'' / f' = -feedback sinh there: about a point of the curve
     * in x, invD = 1 / (1 + 2c cosh) and feedback = -2c invD; about the
     * origin in y, 1 and -kappa. cosh0 and sinh0 are those of the root.
     */
    static void series(double invD, double feedback, double cosh0, double sinh0, int order, double* g) noexcept
    {
        double p[kMaxOrder] {}, C[kMaxOrder] {}, S[kMaxOrder] {};   // g' = sum p_n d^n, cosh g, sinh g
        C[0] = cosh0;
        S[0] = sinh0;
        p[0] = invD;
        for (int n = 0; n + 1 < order; ++n)
        {
            double fromS = 0.0, fromC = 0.0;
            for (int j = 0; j <= n; ++j)
            {
                fromS += S[j] * p[n - j];
                fromC += C[j] * p[n - j];
            }
            const double inv = 1.0 / (n + 1);
            C[n + 1] = fromS * inv;
            S[n + 1] = fromC * inv;
            double sum = 0.0;
            for (int j = 1; j <= n + 1; ++j) sum += C[j] * p[n + 1 - j];
            p[n + 1] = feedback * sum;
        }
        for (int k = 1; k <= order; ++k) g[k - 1] = p[k - 1] / k;
    }

    /** Anchor about the centre m; returns its half-width in x. */
    static double build(double twoC, const Centre& m, Expansion& e) noexcept
    {
        double g[kMaxOrder] {};
        const double invD = 1.0 / (1.0 + twoC * m.cosh);
        series(invD, -twoC * invD, m.cosh, m.sinhHi, kTerms + 2, g);
        // x_m = s_m + 2c (sinh hi + sinh lo): the product and the sum with
        // their rounding errors.
        const double product = twoC * m.sinhHi;
        const double productLow = std::fma(twoC, m.sinhHi, -product) + twoC * m.sinhLo;
        const double sum = m.s + product;
        const double part = sum - m.s;
        const double sumLow = (m.s - (sum - part)) + (product - part);
        e.xHi = sum;
        e.xLo = sumLow + productLow;
        e.s = m.s;
        // No further than x_m / 2 and than the root s_m / 2 (to first order).
        double width = 0.5 * std::min(sum, m.s / g[0]);
        const double reference = kTolerance * 0.5 * m.s;
        for (int k = kTerms + 1; k <= kTerms + 2; ++k)
        {
            const double tail = std::abs(g[k - 1]);
            if (tail > 0.0) width = std::min(width, std::pow(reference / tail, 1.0 / k));
        }
        e.width = width;
        e.a[0] = 0.0;
        double power = width;
        for (int k = 1; k <= kTerms; ++k, power *= width) e.a[k] = g[k - 1] * power;
        return width;
    }

    /**
     * g_1 .. g_order of the inverse of the single diode's curve, as series()
     * does for the pair: g' (1 + c e^g) = 1 with the series of E = e^g
     * carried along (E' = E g'). About a point in x, invD = 1 / (1 + c e^s)
     * and feedback = -c invD; about the origin in y, 1 and -kappa.
     */
    static void seriesSingle(double invD, double feedback, double exp0, int order, double* g) noexcept
    {
        double p[kMaxOrder] {}, E[kMaxOrder] {};
        E[0] = exp0;
        p[0] = invD;
        for (int n = 0; n + 1 < order; ++n)
        {
            double fromE = 0.0;
            for (int j = 0; j <= n; ++j) fromE += E[j] * p[n - j];
            E[n + 1] = fromE / (n + 1);
            double sum = 0.0;
            for (int j = 1; j <= n + 1; ++j) sum += E[j] * p[n + 1 - j];
            p[n + 1] = feedback * sum;
        }
        for (int k = 1; k <= order; ++k) g[k - 1] = p[k - 1] / k;
    }

    /** Anchor of the single diode about the centre m; returns its
     *  half-width in x. */
    static double buildSingle(double c, const ExpCentre& m, Expansion& e) noexcept
    {
        double g[kMaxOrder] {};
        const double invD = 1.0 / (1.0 + c * m.expHi);
        seriesSingle(invD, -c * invD, m.expHi, kTerms + 2, g);
        // x_m = s_m + c (e^s - 1): the product, the difference and the sum
        // with their rounding errors.
        const double product = c * m.expHi;
        const double productLow = std::fma(c, m.expHi, -product) + c * m.expLo;
        const double excess = product - c;
        const double excessPart = excess - product;
        const double excessLow = (product - (excess - excessPart)) + (-c - excessPart);
        const double sum = m.s + excess;
        const double part = sum - m.s;
        const double sumLow = (m.s - (sum - part)) + (excess - part);
        e.xHi = sum;
        e.xLo = (sumLow + excessLow) + productLow;
        e.s = m.s;
        // No further than |x_m| / 2 and than the root s_m / 2 (to first order).
        double width = 0.5 * std::min(std::abs(sum), std::abs(m.s) / g[0]);
        const double reference = kTolerance * 0.5 * std::abs(m.s);
        for (int k = kTerms + 1; k <= kTerms + 2; ++k)
        {
            const double tail = std::abs(g[k - 1]);
            if (tail > 0.0) width = std::min(width, std::pow(reference / tail, 1.0 / k));
        }
        e.width = width;
        e.a[0] = 0.0;
        double power = width;
        for (int k = 1; k <= kTerms; ++k, power *= width) e.a[k] = g[k - 1] * power;
        return width;
    }

    /**
     * Taylor shift of the anchor to @p centre: the cell's root there and its
     * coefficients. False when the first two omitted terms of the cell, at
     * its half-width, pass kTolerance of half that root.
     */
    static bool shift(const Expansion& e, double centre, double half, Cell& cell) noexcept
    {
        const double invWidth = 1.0 / e.width;
        const double tau = ((centre - e.xHi) - e.xLo) * invWidth;
        double work[kTerms + 1];
        for (int k = 0; k <= kTerms; ++k) work[k] = e.a[k];
        // Repeated synthetic division: after pass j, work[j] is the
        // coefficient of the power j about tau.
        for (int j = 0; j <= kCellTerms + 2; ++j)
            for (int i = kTerms - 1; i >= j; --i) work[i] += tau * work[i + 1];
        // Root at the centre: s_m plus the linear term with its rounding
        // error plus the rest, summed without losing the low part.
        const double linear = e.a[1] * tau;
        double rest = e.a[kTerms];
        for (int k = kTerms - 1; k >= 2; --k) rest = rest * tau + e.a[k];
        rest = rest * (tau * tau) + std::fma(e.a[1], tau, -linear);
        const double sum = e.s + linear;
        const double part = sum - e.s;
        const double low = ((e.s - (sum - part)) + (linear - part)) + rest;
        cell.centre = centre;
        cell.rootHi = sum + low;
        cell.rootLo = low - (cell.rootHi - sum);
        double power = invWidth;
        for (int j = 1; j <= kCellTerms; ++j, power *= invWidth) cell.a[j - 1] = work[j] * power;
        const double reach = half * invWidth, reference = kTolerance * 0.5 * std::abs(cell.rootHi);
        double term = 1.0;
        for (int j = 0; j <= kCellTerms; ++j) term *= reach;   // reach to the 17
        return std::abs(work[kCellTerms + 1]) * term <= reference
            && std::abs(work[kCellTerms + 2]) * (term * reach) <= reference;
    }
};

} // namespace detail

/**
 * @brief Antiparallel diode pair root (the classic clipper nonlinearity).
 *
 * Shockley model: i(v) = 2 Is sinh(v / (n Vt)). Defaults approximate a
 * 1N4148 (Is = 2.52 nA, n = 1.752, Vt = 25.85 mV).
 *
 * Solver: in the normalised form s + c (e^s - e^-s) = |a| / nVt (s = v/nVt,
 * c = R Is / nVt) the root is read from series and tables built for the
 * constants of the circuit, with no transcendental function and no division
 * per sample, by drive:
 * - Low drive: the exact reversion series of detail::DiodeSeries, to y^15
 *   for incident waves below about 46-53 mV (default constants behind
 *   0.7-2.2 k ohm; 23 mV at 0.35 M ohm) and, for 1e-4 < c <= 0.02, to y^35
 *   above that, 1.2 to 2.5 units of nVt further (incident waves up to
 *   159 mV behind 2.2 k ohm and 80 mV behind 0.33 M ohm). Beyond c = 0.02
 *   (silicon diodes behind more than about 0.36 M ohm, germanium behind a
 *   few k ohm) the series to y^35 comes from a recurrence for the constant
 *   (detail::DiodeLocal::origin()).
 * - Medium drive, for c > 1e-4 where the series end below it: local
 *   expansions of the root itself (detail::DiodeLocal), up to 56 cells of
 *   sixteen terms found from the bits of the drive. They run from the end
 *   of the series to the drive from which the table's reverse-diode
 *   correction is short.
 * - High drive, up to ln c + x = 2^20: detail::DiodeOmegaTable gives the
 *   forward diode's root s1 = x - omega(ln c + x) to double precision, and
 *   the reverse diode's correction comes from the same table point, to
 *   three orders (c^2 / (omega (1 + omega)) <= 4.05e-5 and
 *   c^2 / (1 + omega)^2 <= 1.47e-5 there for c > 1e-4; for c <= 1e-4 the
 *   table starts at the end of the series).
 *
 * The local expansions belong to one set of constants. prepare() builds
 * them for the circuit as it stands; when a value of the circuit or of the
 * diodes changes while running they are dropped, and once the new constants
 * have held for 256 samples they are built again one cell per sample (at
 * most about 4000 multiply-adds in a sample, 10 to 45 microseconds in all).
 * Until then, and for c > 1e6 or beyond ln c + x = 2^20, the root comes
 * from the closed forms of the two diodes with correction steps on the
 * exact equation: each diode alone has the root s = A - omega(ln c + A)
 * (Wright omega, Werner et al. 2015), the estimate
 * s0 = A - omega(ln c + A) + omega(ln c - A) keeps both and is off by
 * O(c^2), and a Newton step (c <= 1e-4), a Halley step (up to c = 0.02,
 * where the table root with its full correction to seventh order answers
 * first wherever it reaches) or repeated Halley steps until one is below
 * 1e-6 relative (beyond) land on the root, with one expm1() and one
 * division each. A circuit whose values move every sample therefore pays
 * that path for c > 0.02 and at most the table's full correction below.
 *
 * Accuracy: against a long double reference over drives from 1e-6 to 1e6
 * times (1 + 2c), the worst relative error is 3.2e-16 for c from 1e-9 to
 * 1e5; with the correction steps alone, 1.1e-15 for c from 1e-15 to 1e5.
 * The result does not depend on the previous sample.
 *
 * Cost for an RC clipper at 48 kHz, measured on four machines (GCC, Clang
 * and MSVC on x64, GCC on arm64) next to the single-omega approximation
 * that ignores the reverse diode, from 0.05 to 8 V peak: behind 2.2 k to
 * 330 k ohm with the default constants (c up to 0.018) 15 to 39 ns per
 * sample, 0.54 to 0.88 times the approximation's cost (whose error is -39
 * to -88 dB); behind 1 M to 10 M ohm (c from 0.05 to 0.5) 18 to 39 ns, 0.55
 * to 0.81 times the approximation (-12 to -49 dB).
 *
 * Memory: 9.8 KB per instance, most of it the cells.
 */
template <FloatType T, typename Tree>
class DiodePairRoot
{
public:
    explicit DiodePairRoot(Tree& tree, T saturationCurrent = T(2.52e-9),
                           T idealityTimesVt = T(1.752 * 0.02585)) noexcept
        : tree_(tree), is_(saturationCurrent), nvt_(idealityTimesVt),
          omega_(&detail::WrightOmega::instance()), exact_(&detail::DiodeOmegaTable::instance()) {}

    void setSaturationCurrent(T amps) noexcept { is_ = std::max(amps, T(1e-15)); }
    void setIdealityTimesVt(T volts) noexcept { nvt_ = std::max(volts, T(1e-4)); }

    void prepare(double sampleRate) noexcept
    {
        tree_.prepare(sampleRate);
        tree_.updatePorts();
        tree_.reset();
        v_ = 0.0;
        // The constants of the circuit as it stands, with their local
        // expansions, so that the first sample already finds them.
        const double k = 2.0 * tree_.portResistance() * static_cast<double>(is_);
        const double nvt = static_cast<double>(nvt_);
        if (k != kCached_ || nvt != nvtCached_) updateConstants(k, nvt);
        while (localPending_) stepLocal();
    }
    void reset() noexcept { tree_.reset(); v_ = 0.0; }

    void process() noexcept
    {
        const double a = static_cast<double>(tree_.reflected());
        // Non-finite wave (a NaN/Inf source): clear the tree rather than
        // leave it in the capacitor states forever.
        if (!(std::abs(a) <= kMaxFinite)) [[unlikely]]
        {
            tree_.reset();
            v_ = 0.0;
            return;
        }
        const double k = 2.0 * tree_.portResistance() * static_cast<double>(is_);
        const double nvt = static_cast<double>(nvt_);
        if (k != kCached_ || nvt != nvtCached_) [[unlikely]]
            updateConstants(k, nvt);
        else if (localPending_) [[unlikely]]
        {
            // Constants changed while running: the local expansions wait
            // until they have held for kSettle samples, so that a circuit
            // whose values move every sample never pays for them, and are
            // then built one cell per sample.
            if (settle_ < kSettle) ++settle_;
            else stepLocal();
        }

        const double magnitude = std::abs(a);
        const double x = magnitude * invNvt_;
        if (x < seriesLimit_)
        {
            // Low drive: the exact reversion series (detail::DiodeSeries).
            const double y = x * invBeta_, u = y * y, u2 = u * u, u4 = u2 * u2;
            const double p = ((series_[0] + series_[1] * u) + u2 * (series_[2] + series_[3] * u))
                           + u4 * ((series_[4] + series_[5] * u) + u2 * series_[6]);
            const double s = y + (y * u) * p;
            tree_.incident(static_cast<T>(std::copysign(s * (2.0 * nvt), a) - a));
            v_ = std::copysign(s * nvt, a);
            return;
        }
        if (x < series2Limit_)
        {
            // Just above it, for c > 1e-4 (where the omega table's
            // correction takes many terms or does not reach): the same
            // series to y^35, from the tabulated polynomials up to c = 0.02
            // and from detail::DiodeLocal::origin() beyond.
            const double y = x * invBeta_, u = y * y, u2 = u * u, u4 = u2 * u2, u8 = u4 * u4;
            const double* b = series2_;
            const double p = ((((b[0] + b[1] * u) + u2 * (b[2] + b[3] * u)) + u4 * ((b[4] + b[5] * u) + u2 * (b[6] + b[7] * u)))
                              + u8 * (((b[8] + b[9] * u) + u2 * (b[10] + b[11] * u)) + u4 * ((b[12] + b[13] * u) + u2 * (b[14] + b[15] * u))))
                           + (u8 * u8) * b[16];
            const double s = y + (y * u) * p;
            tree_.incident(static_cast<T>(std::copysign(s * (2.0 * nvt), a) - a));
            v_ = std::copysign(s * nvt, a);
            return;
        }
        if (x < localTo_)
        {
            // Between the series and the table's short correction: the cell
            // that holds x (detail::DiodeLocal), found from the bits of x
            // plus the grid's offset.
            const auto cell = static_cast<std::size_t>(detail::DiodeLocal::key(x + localOffset_) - localFirstKey_);
            const double s = detail::DiodeLocal::evaluate(local_[cell], x);
            tree_.incident(static_cast<T>(std::copysign(s * (2.0 * nvt), a) - a));
            v_ = std::copysign(s * nvt, a);
            return;
        }
        if (x < kClosedFormLimit) [[likely]]
        {
            const double s = solveNormalised(x, magnitude);
            tree_.incident(static_cast<T>(std::copysign(s * (2.0 * nvt), a) - a));
            v_ = std::copysign(s * nvt, a);
            return;
        }
        v_ = solveFarField(a, k, nvt);
        tree_.incident(static_cast<T>(2.0 * v_ - a));
    }

    /** @brief Voltage across the pair (the clipper output). */
    [[nodiscard]] T getVoltage() const noexcept { return static_cast<T>(v_); }

private:
    /// Drives beyond |a| = 1e12 nVt leave the double range of the estimate;
    /// they take the bracketed Newton path instead.
    static constexpr double kClosedFormLimit = 1e12;
    /// NaN and +-Inf fail |a| <= max (a plain compare, cheaper than isfinite).
    static constexpr double kMaxFinite = std::numeric_limits<double>::max();

    DSPARK_NOINLINE void updateConstants(double k, double nvt) noexcept
    {
        kCached_ = k;
        nvtCached_ = nvt;
        invNvt_ = 1.0 / nvt;
        invNvt4_ = 4.0 * invNvt_;
        c_ = 0.5 * k * invNvt_;
        logC_ = std::log(c_);
        logC4_ = 4.0 * logC_;
        bound_ = 1.0 / (1.0 + 2.0 * c_);
        cPlus_ = 1.0 + 2.0 * c_;
        mode_ = c_ <= 1e-4 ? Mode::Newton : (c_ <= 0.02 ? Mode::Halley : Mode::Iterate);
        c2_ = c_ * c_;
        halfC2_ = 0.5 * c2_;
        // First node of solveTableWide() whose whole cell has q <= kShortQ,
        // r <= kShortR and a root above 1.25 (R and H fall with z): with
        // omega(1 + omega) >= c^2 / kShortQ, 1 + omega >= c / sqrt(kShortR)
        // and x = z - ln c >= 1.25 there, the omitted terms (largest
        // (8/3) q^4 and (27/8) r q^3) total under 1.1e-17.
        shortFrom_ = std::numeric_limits<int>::max();
        if (mode_ == Mode::Halley)
        {
            const double wq = 0.5 * (std::sqrt(1.0 + 4.0 * c2_ / kShortQ) - 1.0);
            const double wr = c_ / std::sqrt(kShortR) - 1.0;
            const double w = std::max({wq, wr, 1e-300});
            const double z = std::max(w + std::log(w), 1.25 + logC_);
            const double from = std::ceil((z + 0.125 - detail::DiodeOmegaTable::kLo) * 4.0);
            shortFrom_ = static_cast<int>(std::clamp(from, 0.0, double(detail::DiodeOmegaTable::kNodes)));
        }
        // ln c = logC_ + ln(c e^-logC_): the residual, to about 1e-16.
        logCLow_ = std::fma(c_, std::exp(-logC_), -1.0);
        invBeta_ = bound_;
        seriesLimit_ = cPlus_ * detail::DiodeSeries::pair(2.0 * c_ * bound_, series_);
        // The table root holds where q = c^2 / (omega (1 + omega)) <= kTableQ:
        // above the series limit for c <= 1e-4; from omega(1 + omega) =
        // c^2 / kWideTableQ up to c = 0.02, with the correction series of
        // solveTableWide() scaled by powers of c^2. In that range the longer
        // series takes every drive it reaches above the short one: it costs
        // less than the table root with its full correction, and for
        // c > 0.0062, where the table starts above the short series, it
        // covers the drives between (at c = 0.02 it reaches 1.764 against
        // 1.748 nVt). The omega estimate with its correction step would
        // cover any remaining gap.
        series2Limit_ = seriesLimit_;
        tableFrom_ = seriesLimit_;
        if (mode_ == Mode::Halley)
        {
            const double w = 0.5 * (std::sqrt(1.0 + 4.0 * c2_ / kWideTableQ) - 1.0);
            const double wideStart = w + std::log(w) - logC_;
            series2Limit_ = std::max(seriesLimit_, cPlus_ * detail::DiodeSeries::pairLong(2.0 * c_ * bound_, series2_));
            tableFrom_ = std::max(series2Limit_, wideStart);
            int o = 0;
            for (int j = 0; j < 5; ++j)
            {
                double scale = 1.0;
                for (int m = 0; m <= j; ++m) scale *= c2_;
                for (int m = 0; m < 7 - j; ++m, scale *= c2_)
                    wide_[o++] = kWideCoefficients[j][m] * scale;
            }
        }
        else if (mode_ == Mode::Iterate)
            tableFrom_ = std::numeric_limits<double>::infinity();

        // Local expansions (detail::DiodeLocal), built in prepare() or once
        // these constants have held for kSettle samples. Up to c = 0.02 they
        // run from the longer series to the first node with the short
        // correction, in place of the table root with its full correction.
        // Beyond, from the low-drive series of that constant (built with
        // them) to the drive from which the reverse diode's correction is
        // short (q <= kShortQ and r <= kShortR), where the table root takes
        // over on every node and cell; until they are built the repeated
        // Halley steps answer below that drive.
        localTo_ = -1.0;
        settle_ = 0;
        localPending_ = false;
        localStarted_ = false;
        if (mode_ == Mode::Halley)
        {
            localFrom_ = series2Limit_;
            localGoal_ = (detail::DiodeOmegaTable::kLo + 0.25 * shortFrom_ - 0.125) - logC_;
            localPending_ = shortFrom_ < detail::DiodeOmegaTable::kNodes && localGoal_ > localFrom_;
        }
        else if (mode_ == Mode::Iterate && c_ <= kLocalMaxC)
        {
            const double wq = 0.5 * (std::sqrt(1.0 + 4.0 * c2_ / kShortQ) - 1.0);
            const double wr = c_ / std::sqrt(kShortR) - 1.0;
            const double w = std::max(wq, wr);
            localFrom_ = 0.0;
            localGoal_ = (w + std::log(w)) - logC_;
            tableFrom_ = localGoal_;
            shortFrom_ = 0;
            localPending_ = true;
        }
    }

    /**
     * One step of the local expansions of the current constants: the first
     * lays the grid out (and, beyond c = 0.02, builds the low-drive series),
     * each of the next adds one cell, which answers from then on. A step
     * costs at most two anchors and one cell, about 4000 multiply-adds.
     */
    DSPARK_NOINLINE void stepLocal() noexcept
    {
        if (!localStarted_)
        {
            localStarted_ = true;
            if (mode_ == Mode::Iterate)
            {
                // The low-drive series of this constant, which the tabulated
                // polynomials of DiodeSeries do not cover.
                series2Limit_ = std::max(seriesLimit_, cPlus_ * detail::DiodeLocal::origin(2.0 * c_ * bound_, series2_));
                localFrom_ = series2Limit_;
            }
            detail::DiodeLocal::start(localBuild_, c_, localFrom_, localGoal_);
            localOffset_ = localBuild_.offset;
            localFirstKey_ = localBuild_.firstKey;
            localPending_ = !localBuild_.done;
            return;
        }
        double end = localTo_;
        const bool added = detail::DiodeLocal::step(localBuild_, local_, end);
        localTo_ = end;
        localPending_ = added && !localBuild_.done;
    }

    /** logValue on the nodes of the ln(omega) form (index >= kFirstLogNode),
     *  else linearValue, selected by a bit mask: a mid-level signal crosses
     *  that boundary (z = 0) twice per half cycle, and a branch there
     *  mispredicted (behind 0.33 M ohm at 0.5 V peak: 24.3 against 22.9 ns per
     *  sample with the longer series as well). */
    [[nodiscard]] static double byForm(int index, double logValue, double linearValue) noexcept
    {
        const std::uint64_t mask = std::uint64_t(0)
            - static_cast<std::uint64_t>(index >= detail::DiodeOmegaTable::kFirstLogNode);
        return std::bit_cast<double>((std::bit_cast<std::uint64_t>(logValue) & mask)
                                     | (std::bit_cast<std::uint64_t>(linearValue) & ~mask));
    }

    /** Root s >= 0 of s + c (e^s - e^-s) = x for x = magnitude / nVt >= 0. */
    [[nodiscard]] double solveNormalised(double x, double magnitude) const noexcept
    {
        if (x >= tableFrom_) [[likely]]
        {
            // 4 (ln c + x) with one rounding. The product is 4 x exactly (the
            // factor 4 only moves the exponent), formed from the wave so that
            // the table index does not wait for x.
            const double z4 = magnitude * invNvt4_ + logC4_;
            if (z4 < 4.0 * detail::DiodeOmegaTable::kOctaveMax) [[likely]]
                return mode_ == Mode::Newton ? solveTable(x, z4) : solveTableWide(x, z4);
        }
        const auto& omega = *omega_;
        // Both diodes' closed forms.
        const double s = (x + omega(logC_ - x)) - omega(logC_ + x);
        if (mode_ == Mode::Newton) [[likely]]
        {
            // One expm1 and one division. With u = e^s, g and g' share the
            // factor 1/u, which cancels in the step, and u^2 - 1 = m (2 + m)
            // keeps small s free of cancellation.
            const double m = std::expm1(s);
            const double sh = c_ * (m * (2.0 + m));        // c (u^2 - 1)
            const double g = (s - x) * (1.0 + m) + sh;     // u g
            const double g1 = (cPlus_ + m) + sh;           // u g' = u + c (u^2 + 1)
            return s - g / g1;
        }
        return refineHalley(s, x);
    }

    /**
     * Table root for c <= 1e-4 and kLo <= z < kOctaveMax (z = ln c + x):
     * the forward diode alone has the root s1 = x - omega(z), and the reverse
     * diode moves it by d, the root of d + omega (e^d - 1) = (c^2 / omega) e^-d.
     * With q = c^2 R = c^2 / (omega (1 + omega)) and h = 1 / (1 + omega)^2,
     * d = q (1 - c^2 h / 2) - q^2 (1 - 1.5 q) + O(q^4): q < 5e-5 above the
     * series limit for c <= 1e-4, so the omitted terms stay below 1e-17 of
     * the root. Below z = kLo, omega < e^-40 and the root is x.
     */
    [[nodiscard]] double solveTable(double x, double z4) const noexcept
    {
        using Table = detail::DiodeOmegaTable;
        if (z4 < 4.0 * Table::kHi)
        {
            if (!(z4 >= 4.0 * Table::kLo)) return x;
            double t = 0.0;
            int index = 0;
            const auto& n = exact_->node(z4, t, index);
            const double rest = Table::poly9Rest(n.p, t);
            // q = c^2 R with c^2 applied to the two Estrin halves of R (one
            // multiply less on the path), and d = q (1 - h) - q^2 + 1.5 q^3.
            const double t2 = t * t;
            const double* r = n.r;
            const double ra = (r[0] + r[1] * t) + t2 * (r[2] + r[3] * t);
            const double rb = (r[4] + r[5] * t) + t2 * (r[6] + r[7] * t);
            const double q = c2_ * ra + (c2_ * (t2 * t2)) * rb;
            const double h = halfC2_ * (n.h[0] + n.h[1] * t);
            const double q2 = q * q;
            const double d = (q * (1.0 - h) - q2) + (1.5 * q) * q2;
            // Root = base + p0 + rest + d, with the large base + p0 summed
            // exactly (|base| > |p0|) so that only the final add rounds.
            const double base = byForm(index, -logC_, x);
            const double s0 = base + n.p[0];
            const double e0 = n.p[0] - (s0 - base);
            return s0 + (((rest + e0) - byForm(index, logCLow_, 0.0)) + d);
        }
        // Above z = 32, q < c^2 / 818: q^2 is below 1e-24 of the root.
        double t = 0.0;
        const auto& cl = exact_->cell(0.25 * z4, t);
        const double r = Table::poly7(cl.r, t);
        const double rest = Table::poly9Rest(cl.p, t);
        const double s0 = cl.p[0] - logC_;
        const double v = s0 - cl.p[0];
        const double e0 = (cl.p[0] - (s0 - v)) - (logC_ + v);   // two-sum error
        return s0 + (((rest + e0) - logCLow_) + c2_ * r);
    }

    /**
     * Table root for 1e-4 < c <= 0.02, where q = c^2 R is bounded by
     * kWideTableQ instead of by the series limit. With r = c^2 H = q w
     * (w = omega / (1 + omega)), d = sum a_nj r^j q^(n-j) over n <= 7 and
     * j <= 4, the reversion of d + w (e^d - 1 - d) = q e^-d; where q is
     * large w is small (q w <= c^2), and the omitted terms stay below 9e-18 of
     * the root for c up to 0.02 from the table start (checked against
     * 40-digit solutions). Evaluated as R (B0(R) + H (B1(R) + ...)) with the
     * powers of c^2 in the instance's coefficients, so the correction does
     * not wait for omega. From the node shortFrom_ on (q <= kShortQ and
     * r <= kShortR over the whole node) the series stops at q^3, r q^2 and
     * r^2 q, three multiply-adds after R. Above z = 32 (q < c^2 / 818) the
     * series stops at q^2.
     */
    [[nodiscard]] double solveTableWide(double x, double z4) const noexcept
    {
        using Table = detail::DiodeOmegaTable;
        if (z4 < 4.0 * Table::kHi)
        {
            if (!(z4 >= 4.0 * Table::kLo)) return x;
            double t = 0.0;
            int index = 0;
            const auto& n = exact_->node(z4, t, index);
            const double rest = Table::poly9Rest(n.p, t);
            const double R = n.r[0] + Table::poly9Rest(n.r, t);
            const double t2 = t * t, t4 = t2 * t2;
            const double H = ((n.h[0] + n.h[1] * t) + t2 * (n.h[2] + n.h[3] * t))
                           + t4 * ((n.h[4] + n.h[5] * t) + t2 * n.h[6]);
            const double base = byForm(index, -logC_, x);
            const double s0 = base + n.p[0];
            const double e0 = n.p[0] - (s0 - base);
            const double tail = (rest + e0) - byForm(index, logCLow_, 0.0);
            if (index >= shortFrom_)
            {
                // q <= kShortQ and r <= kShortR on this node (see
                // updateConstants()): every term past q^3, r q^2 and r^2 q
                // stays below 9e-18 of the root, and the correction is three
                // multiply-adds deep after R instead of seven.
                const double q = c2_ * R, r = c2_ * H;
                return s0 + (tail + q * (((1.0 - q) + 1.5 * (q * q)) + r * ((-0.5 + (4.0 / 3.0) * q) + 0.5 * r)));
            }
            {
                const double* a = wide_;
                const double R2 = R * R, R4 = R2 * R2, H2 = H * H;
                const double B0 = ((a[0] + a[1] * R) + R2 * (a[2] + a[3] * R)) + R4 * ((a[4] + a[5] * R) + R2 * a[6]);
                const double B1 = ((a[7] + a[8] * R) + R2 * (a[9] + a[10] * R)) + R4 * (a[11] + a[12] * R);
                const double B2 = ((a[13] + a[14] * R) + R2 * (a[15] + a[16] * R)) + R4 * a[17];
                const double B3 = (a[18] + a[19] * R) + R2 * (a[20] + a[21] * R);
                const double B4 = (a[22] + a[23] * R) + R2 * a[24];
                return s0 + (tail + R * ((B0 + H * B1) + H2 * ((B2 + H * B3) + H2 * B4)));
            }
        }
        double t = 0.0;
        const auto& cl = exact_->cell(0.25 * z4, t);
        const double R = Table::poly7(cl.r, t);
        const double rest = Table::poly9Rest(cl.p, t);
        const double w = 1.0 - (0.25 * z4 - (cl.p[0] + rest)) * R;
        const double q = c2_ * R;
        const double s0 = cl.p[0] - logC_;
        const double v = s0 - cl.p[0];
        const double e0 = (cl.p[0] - (s0 - v)) - (logC_ + v);   // two-sum error
        if (mode_ == Mode::Iterate)
        {
            // c > 0.02: q is bounded by kShortQ here, not by c^2 / 818, and
            // the short correction is needed in full (r = q w).
            const double r = q * w;
            const double lead = q - (q * q) * (1.0 - 1.5 * q);
            const double cross = (-0.5 + (4.0 / 3.0) * q) + 0.5 * r;
            return s0 + (((rest + e0) - logCLow_) + (lead + (q * r) * cross));
        }
        return s0 + (((rest + e0) - logCLow_) + (q - (q * q) * (1.0 + 0.5 * w)));
    }

    /** Halley steps (u g'' = c (u^2 - 1) as well), once for c <= 0.02, else
     *  from inside the bracket [0, x / (1 + 2c)] until a step is below 1e-6
     *  relative. sinh(s) >= s bounds the root by x / (1 + 2c). */
    [[nodiscard]] DSPARK_NOINLINE double refineHalley(double s, double x) const noexcept
    {
        if (mode_ == Mode::Iterate) s = std::clamp(s, 0.0, x * bound_);
        for (int it = 0; it < 32; ++it)
        {
            const double m = std::expm1(s);
            const double sh = c_ * (m * (2.0 + m));
            const double g = (s - x) * (1.0 + m) + sh;
            const double g1 = (cPlus_ + m) + sh;
            const double step = (2.0 * g) * g1 / ((2.0 * g1) * g1 - g * sh);
            s -= step;
            if (mode_ == Mode::Halley || !(std::abs(step) > 1e-6 * std::abs(s))) break;
        }
        return s;
    }

    /** Bracketed Newton in volts, for drives beyond kClosedFormLimit. */
    [[nodiscard]] DSPARK_NOINLINE static double solveFarField(double a, double k, double nvt) noexcept
    {
        return detail::solveMonotonic(a, nvt * std::asinh(a / k),
                                      [k, nvt, a](double v, double& f, double& fp)
        {
            // cosh overflows around |x| > 710 in double, far outside any
            // solution the bracket allows anyway.
            const double x = std::clamp(v / nvt, -700.0, 700.0);
            f  = v + k * std::sinh(x) - a;
            fp = 1.0 + (k / nvt) * std::cosh(x);
        });
    }

    Tree& tree_;
    T is_, nvt_;
    const detail::WrightOmega* omega_;
    const detail::DiodeOmegaTable* exact_;
    double v_ = 0.0;
    double kCached_ = -1.0, nvtCached_ = -1.0;
    double c2_ = 1.0, halfC2_ = 0.5;   ///< c^2 and c^2 / 2 (table path).
    double logCLow_ = 0.0;              ///< ln c - logC_ (table path).
    double logC4_ = 0.0;                ///< 4 logC_ (table index).
    double tableFrom_ = 0.0;            ///< Table root from this x on (see updateConstants()).
    static constexpr double kTableQ = 3e-5;   ///< 8.75 kTableQ^4 < 1e-17.
    static constexpr double kWideTableQ = 3.5e-3;   ///< Table start of solveTableWide().
    /// a_nj of the correction series, by power j of r and then of q.
    static constexpr double kWideCoefficients[5][7] = {
        {1.0, -1.0, 1.5, -8.0 / 3.0, 125.0 / 24.0, -54.0 / 5.0, 16807.0 / 720.0},
        {-0.5, 4.0 / 3.0, -27.0 / 8.0, 128.0 / 15.0, -3125.0 / 144.0, 1944.0 / 35.0, 0.0},
        {0.5, -25.0 / 12.0, 85.0 / 12.0, -8057.0 / 360.0, 5451.0 / 80.0, 0.0, 0.0},
        {-0.625, 3.5, -1057.0 / 72.0, 4927.0 / 90.0, 0.0, 0.0, 0.0},
        {0.875, -6.125, 485.0 / 16.0, 0.0, 0.0, 0.0, 0.0}};
    double wide_[25] {};   ///< kWideCoefficients times powers of c^2 (Halley mode).
    /// Bounds of q and r for the short correction of solveTableWide() (the
    /// values at z = 0 for c = 6e-3).
    static constexpr double kShortQ = 4.05e-5, kShortR = 1.47e-5;
    int shortFrom_ = std::numeric_limits<int>::max();   ///< First node with the short correction.
    enum class Mode { Newton, Halley, Iterate };
    double invNvt_ = 1.0, c_ = 1.0, logC_ = 0.0, bound_ = 1.0, cPlus_ = 3.0;
    double invNvt4_ = 4.0;                        ///< 4 / nVt (table index).
    double invBeta_ = 1.0, seriesLimit_ = -1.0;   ///< Series path below |a| / nVt = seriesLimit_.
    double series_[7] {};                         ///< y^3 .. y^15 coefficients.
    double series2Limit_ = -1.0;                  ///< Longer series from seriesLimit_ to here.
    double series2_[17] {};                       ///< y^3 .. y^35 coefficients (c > 1e-4).
    Mode mode_ = Mode::Iterate;
    /// Local expansions are not built beyond this c (their coefficients
    /// would leave the double range); the Halley steps keep that range.
    static constexpr double kLocalMaxC = 1e6;
    static constexpr int kSettle = 256;           ///< Samples the constants hold before a build.
    double localTo_ = -1.0;                       ///< Local expansions answer below this x.
    double localFrom_ = 0.0, localGoal_ = 0.0;    ///< Range they are built for.
    int settle_ = 0;
    bool localPending_ = false;                   ///< Cells still to build.
    bool localStarted_ = false;                   ///< Their grid is laid out.
    detail::DiodeLocal::Builder localBuild_;
    double localOffset_ = 0.0;                    ///< Offset of the cell grid.
    std::uint64_t localFirstKey_ = 0;             ///< Key of the first cell.
    detail::DiodeLocal::Cell local_[detail::DiodeLocal::kMaxCells];
};

/**
 * @brief Single Shockley diode root: i(v) = Is (e^{v/(n Vt)} - 1).
 *
 * Conducts for positive v, blocks (down to -Is) for negative v.
 *
 * Solver: s + c (e^s - 1) = a / nVt (s = v/nVt, c = R Is / nVt) has the
 * exact root s = A + c - omega(ln c + A + c) (Werner et al. 2015). It is
 * read from series and tables, with no transcendental function and no
 * division per sample, by constant and drive:
 * - c up to 0.04 (the default constants behind up to about 0.7 M ohm): the
 *   exact reversion series of detail::DiodeSeries for incident waves below
 *   about 42-49 mV of either sign (default constants behind 0.7-2.2 k ohm),
 *   where A + c and omega would cancel, and the closed form from
 *   detail::DiodeOmegaTable beyond, up to ln c + A + c = 2^20.
 * - c from 0.04 to 1e5 (germanium diodes, very large resistances): at low
 *   drive a series to y^33 from a recurrence for the constant
 *   (detail::DiodeLocal::originSingle()); outside the drives of the roots
 *   -1 and 1.5 the closed form from the table, with x + c, ln c and the
 *   constant term of the table each carried to twice the working precision
 *   so that their difference keeps it; and between the series and those
 *   two drives, on either side, local expansions of the root
 *   (detail::DiodeLocal): cells of sixteen terms found from the bits of the
 *   drive, at most 24 of them for 3000 constants over that range, in room
 *   for 32.
 *
 * The series and the cells of a constant beyond 0.04 belong to it.
 * prepare() builds them for the circuit as it stands (2 to 24 microseconds);
 * when a value of the circuit or of the diode changes while running they
 * are dropped, and once the new constants have held for 256 samples they
 * are built again, the series in one sample and then one cell per sample.
 * Until then between the two table drives, there as well for c beyond 1e5,
 * at every drive for c beyond 1e7 and beyond ln c + A + c = 2^20, a
 * tabulated omega estimate takes correction steps on the exact equation,
 * with one exponential and one division each: one Newton step while
 * c <= 1e-4, one Halley step up to c = 10, repeated Halley steps beyond.
 *
 * Accuracy: against a long double reference over drives of either sign from
 * 1e-6 to 1e6 times (1 + c), the worst relative error is 3.6e-16 for c from
 * 1e-9 to 1e5 (3.3e-16 beyond c = 0.04), and under reverse drive it stays
 * within 3.5e-16 up to c = 1e9. The result does not depend on the
 * previous sample.
 *
 * Cost for an RC circuit at 48 kHz, measured on four machines (GCC, Clang
 * and MSVC on x64, GCC on arm64) next to the closed form with a fast omega
 * estimate (D'Angelo et al., DAFx-19), from 0.05 to 8 V peak: behind 2.2 k
 * to 330 k ohm with the default constants (c up to 0.018) 15 to 26 ns per
 * sample, 0.48 to 0.81 times the estimate's cost (whose error is -53 to
 * -153 dB); behind 1 M to 10 M ohm (c from 0.05 to 0.5) 17 to 27 ns, 0.49
 * to 0.87 times the estimate (-43 to -80 dB).
 *
 * Memory: 6.2 KB per instance, most of it the cells.
 */
template <FloatType T, typename Tree>
class DiodeRoot
{
public:
    explicit DiodeRoot(Tree& tree, T saturationCurrent = T(2.52e-9),
                       T idealityTimesVt = T(1.752 * 0.02585)) noexcept
        : tree_(tree), is_(saturationCurrent), nvt_(idealityTimesVt),
          omega_(&detail::WrightOmega::instance()), exact_(&detail::DiodeOmegaTable::instance()) {}

    void setSaturationCurrent(T amps) noexcept { is_ = std::max(amps, T(1e-15)); }
    void setIdealityTimesVt(T volts) noexcept { nvt_ = std::max(volts, T(1e-4)); }

    void prepare(double sampleRate) noexcept
    {
        tree_.prepare(sampleRate);
        tree_.updatePorts();
        tree_.reset();
        v_ = 0.0;
        // The constants of the circuit as it stands, with their local
        // expansions, so that the first sample already finds them.
        const double k = tree_.portResistance() * static_cast<double>(is_);
        const double nvt = static_cast<double>(nvt_);
        if (k != kCached_ || nvt != nvtCached_) updateConstants(k, nvt);
        while (localPending_) stepLocal();
    }
    void reset() noexcept { tree_.reset(); v_ = 0.0; }

    void process() noexcept
    {
        const double a = static_cast<double>(tree_.reflected());
        if (!std::isfinite(a)) [[unlikely]]
        {
            tree_.reset();
            v_ = 0.0;
            return;
        }
        const double k = tree_.portResistance() * static_cast<double>(is_);
        const double nvt = static_cast<double>(nvt_);
        if (k != kCached_ || nvt != nvtCached_) [[unlikely]]
            updateConstants(k, nvt);
        else if (localPending_) [[unlikely]]
        {
            // Constants changed while running: the local expansions wait
            // until they have held for kSettle samples, so that a circuit
            // whose values move every sample never pays for them, and are
            // then built one cell per sample.
            if (settle_ < kSettle) ++settle_;
            else stepLocal();
        }

        const double x = a * invNvt_;
        if (std::abs(x) < seriesLimit_)
        {
            // Low drive: the exact reversion series (detail::DiodeSeries).
            const double y = x * invBeta_, y2 = y * y, y4 = y2 * y2, y8 = y4 * y4;
            const double* d = series_;
            const double q = (((d[0] + d[1] * y) + y2 * (d[2] + d[3] * y))
                              + y4 * ((d[4] + d[5] * y) + y2 * (d[6] + d[7] * y)))
                           + y8 * (((d[8] + d[9] * y) + y2 * (d[10] + d[11] * y))
                                   + y4 * (d[12] + d[13] * y));
            v_ = (y + y2 * q) * nvt;
        }
        else if (seriesLimit_ > 0.0 && x * 4.0 + logCc4_ < 4.0 * detail::DiodeOmegaTable::kOctaveMax) [[likely]]
        {
            v_ = solveTable(x, x * 4.0 + logCc4_) * nvt;
        }
        else if (std::abs(x) < 1e12)
        {
            // 4 (ln c + c + x) with one rounding, formed from the wave so
            // that the table index does not wait for x.
            v_ = (large_ ? solveLarge(x, a * invNvt4_ + logCc4_) : solveIterate(x)) * nvt;
        }
        else
        {
            v_ = solveFarField(a, k, nvt);
        }

        tree_.incident(static_cast<T>(2.0 * v_ - a));
    }

    /** @brief Voltage across the diode. */
    [[nodiscard]] T getVoltage() const noexcept { return static_cast<T>(v_); }

private:
    DSPARK_NOINLINE void updateConstants(double k, double nvt) noexcept
    {
        kCached_ = k;
        nvtCached_ = nvt;
        invNvt_ = 1.0 / nvt;
        c_ = k * invNvt_;
        logC_ = std::log(c_);
        steps_ = c_ <= 1e-4 ? 0 : (c_ <= 10.0 ? 1 : 32);
        logCc4_ = 4.0 * (logC_ + c_);
        logCLow_ = std::fma(c_, std::exp(-logC_), -1.0);
        invNvt4_ = 4.0 * invNvt_;
        invBeta_ = 1.0 / (1.0 + c_);
        seriesLimit_ = (1.0 + c_) * detail::DiodeSeries::single(c_ * invBeta_, series_);

        // Beyond the constants the tabulated series cover (c above 0.0417)
        // the table root answers outside two drives that depend on the
        // constant (detail::DiodeLocal::singleTableDrives()), and up to
        // kLocalMaxC a low-drive series and cells built for the constant
        // (detail::DiodeLocal) answer between them: in prepare(), or once
        // these constants have held for kSettle samples.
        large_ = !(seriesLimit_ > 0.0) && c_ <= kTableMaxC;
        longLimit_ = 0.0;
        forwardTo_ = -1.0;
        reverseFrom_ = std::numeric_limits<double>::infinity();
        reverseTo_ = -std::numeric_limits<double>::infinity();
        reverseCount_ = 0;
        settle_ = 0;
        localPhase_ = 0;
        localPending_ = large_ && c_ <= kLocalMaxC;
        if (large_)
        {
            // ln c to twice the precision, and what logCc4_ leaves out of
            // 4 (ln c + c).
            double eh = 0.0, el = 0.0;
            detail::expTwice(logC_, eh, el);
            logCLow_ = ((c_ - eh) - el) / eh;
            const double sum = logC_ + c_;
            const double part = sum - logC_;
            logCcLow4_ = 4.0 * (((logC_ - (sum - part)) + (c_ - part)) + logCLow_);
            detail::DiodeLocal::singleTableDrives(c_, tableReverse_, tableForward_);
        }
    }

    /**
     * One step of the local expansions of the current constants: the first
     * builds the low-drive series and lays the grid out, the next ones add
     * one cell each, below the origin first and then above it. A step costs
     * at most two anchors and one cell, about 3000 multiply-adds.
     */
    DSPARK_NOINLINE void stepLocal() noexcept
    {
        using Local = detail::DiodeLocal;
        if (localPhase_ == 0)
        {
            const double limit = (1.0 + c_) * Local::originSingle(c_ * invBeta_, long_);
            localOffset_ = Local::singleOffset(c_);
            Local::startSingle(localBuild_, c_, tableReverse_, -limit, localOffset_, true, Local::kSingleMaxCells);
            reverseFirstKey_ = localBuild_.firstKey;
            longLimit_ = limit;
            localPhase_ = 1;
            return;
        }
        if (localPhase_ == 1)
        {
            double end = reverseTo_;
            if (Local::step(localBuild_, local_, end))
            {
                reverseFrom_ = tableReverse_;
                reverseTo_ = end;
                reverseCount_ = localBuild_.count;
                if (!localBuild_.done) return;
            }
            // Below the origin is done: above it, in the room that is left.
            Local::startSingle(localBuild_, c_, longLimit_, tableForward_, localOffset_, false,
                               Local::kSingleMaxCells - reverseCount_);
            forwardFirstKey_ = localBuild_.firstKey;
            localPhase_ = 2;
            localPending_ = !localBuild_.done;
            return;
        }
        double end = forwardTo_;
        const bool added = Local::step(localBuild_, local_ + reverseCount_, end);
        forwardTo_ = end;
        localPending_ = added && !localBuild_.done;
    }

    /** Root for 0.0417 < c <= kLocalMaxC: the series and the cells of the
     *  constant where they are built, the table root where it is accurate,
     *  and the estimate with its correction steps for whatever is left. */
    [[nodiscard]] double solveLarge(double x, double z4) const noexcept
    {
        using Local = detail::DiodeLocal;
        if (std::abs(x) < longLimit_)
        {
            // s = y + y^2 (q_0 + q_1 y + ... + q_31 y^31).
            const double y = x * invBeta_;
            const double* q = long_;
            const double y2 = y * y, y4 = y2 * y2, y8 = y4 * y4;
            const double q0 = (q[0] + q[1] * y) + y2 * (q[2] + q[3] * y);
            const double q1 = (q[4] + q[5] * y) + y2 * (q[6] + q[7] * y);
            const double q2 = (q[8] + q[9] * y) + y2 * (q[10] + q[11] * y);
            const double q3 = (q[12] + q[13] * y) + y2 * (q[14] + q[15] * y);
            const double q4 = (q[16] + q[17] * y) + y2 * (q[18] + q[19] * y);
            const double q5 = (q[20] + q[21] * y) + y2 * (q[22] + q[23] * y);
            const double q6 = (q[24] + q[25] * y) + y2 * (q[26] + q[27] * y);
            const double q7 = (q[28] + q[29] * y) + y2 * (q[30] + q[31] * y);
            const double low = (q0 + y4 * q1) + y8 * (q2 + y4 * q3);
            const double high = (q4 + y4 * q5) + y8 * (q6 + y4 * q7);
            return y + y2 * (low + (y8 * y8) * high);
        }
        if (x > 0.0)
        {
            if (x < forwardTo_)
            {
                const auto cell = static_cast<std::size_t>(Local::key(x + localOffset_) - forwardFirstKey_);
                return Local::evaluate(local_[static_cast<std::size_t>(reverseCount_) + cell], x);
            }
            if (x >= tableForward_) return solveTableLarge(x, z4);
        }
        else
        {
            if (x >= reverseFrom_ && x < reverseTo_)
            {
                const auto cell = static_cast<std::size_t>(Local::key(x + localOffset_) - reverseFirstKey_);
                return Local::evaluate(local_[cell], x);
            }
            if (x <= tableReverse_) return solveTableLarge(x, z4);
        }
        return solveIterate(x);
    }

    /** Bracketed Newton in volts, for drives beyond 1e12 nVt. Far-field
     *  analytic seed: forward drive lands near nVt ln(1 + a/k); reverse
     *  drive blocks, so the root sits near a. */
    [[nodiscard]] DSPARK_NOINLINE static double solveFarField(double a, double k, double nvt) noexcept
    {
        const double seed = (a > 0.0) ? nvt * std::log1p(a / k) : a;
        return detail::solveMonotonic(a, seed, [k, nvt, a](double v, double& f, double& fp)
        {
            const double xx = std::clamp(v / nvt, -700.0, 700.0);
            f  = v + k * std::expm1(xx) - a;
            fp = 1.0 + (k / nvt) * std::exp(xx);
        });
    }

    /** Tabulated omega estimate and correction steps on the exact equation
     *  (one exponential and one division each). Out of line: the series and
     *  the tables answer every drive of a prepared root up to c = 1e5. */
    [[nodiscard]] DSPARK_NOINLINE double solveIterate(double x) const noexcept
    {
        double s = x + c_ - (*omega_)(logC_ + x + c_);
        // x + c with its rounding error, for the residual below s = -0.7.
        const double sum = x + c_;
        const double part = sum - x;
        const double sumLow = (x - (sum - part)) + (c_ - part);
        for (int it = 0; it <= steps_; ++it)
        {
            double e = 0.0, g = 0.0;                // c e^s = g' - 1 = g'', residual
            if (s < -0.7)
            {
                // s + c e^s - (x + c): c (e^s - 1) would lose what c e^s
                // holds below the last place of c.
                e = c_ * std::exp(s);
                g = ((s - sum) - sumLow) + e;
            }
            else
            {
                const double m = std::expm1(s);
                e = c_ * (1.0 + m);
                g = (s - x) + c_ * m;
            }
            const double g1 = 1.0 + e;
            if (steps_ == 0)
            {
                s -= g / g1;
                break;
            }
            const double step = (2.0 * g) * g1 / ((2.0 * g1) * g1 - g * e);
            s -= step;
            if (steps_ == 1 || !(std::abs(step) > 1e-6 * std::abs(s))) break;
        }
        return s;
    }

    /**
     * The closed form from the table for a constant that is not small, at
     * the drives where it is accurate (see updateConstants()). The node comes
     * from z4, the rounded 4 (ln c + c + x), so that its coefficients do not
     * wait for anything else; what the rounding and the stored constant leave
     * out of z is known exactly (dz) and enters through the first two
     * derivatives of the polynomial of the node. Every leading term is
     * carried to twice the working precision: x + c, which cancels under
     * reverse drive, is summed with its error, the table holds what its
     * constant terms leave out of -omega and ln(omega), and ln c has its
     * remainder. Below z = kLo, omega < e^-40 and the root is x + c.
     */
    [[nodiscard]] double solveTableLarge(double x, double z4) const noexcept
    {
        using Table = detail::DiodeOmegaTable;
        if (!(z4 < 4.0 * Table::kOctaveMax)) return solveIterate(x);
        const double x4 = 4.0 * x;
        const double part = z4 - x4;
        const double dz = ((x4 - (z4 - part)) + (logCc4_ - part)) + logCcLow4_;
        if (z4 < 4.0 * Table::kHi)
        {
            if (!(z4 >= 4.0 * Table::kLo)) return x + c_;
            double t = 0.0;
            int index = 0;
            const auto& n = exact_->node(z4, t, index);
            // The small terms are gathered apart, so that only the
            // polynomial and two sums follow the node.
            const double extra = dz * (n.p[1] + n.p[2] * (2.0 * t + dz)) + exact_->nodeLow(index);
            const double rest = Table::poly9Rest(n.p, t);
            if (index < Table::kFirstLogNode)
            {
                const double sum = x + c_;                  // |x| > c wherever z < 0
                const double low = c_ - (sum - x);
                const double s0 = sum + n.p[0];
                const double v = s0 - sum;
                const double e0 = (sum - (s0 - v)) + (n.p[0] - v);
                return s0 + (rest + ((e0 + low) + extra));
            }
            const double s0 = n.p[0] - logC_;
            const double v = s0 - n.p[0];
            const double e0 = (n.p[0] - (s0 - v)) - (logC_ + v);
            return s0 + (rest + ((e0 - logCLow_) + extra));
        }
        double t = 0.0, scale = 0.0, p0Low = 0.0;
        const auto& cl = exact_->cell(0.25 * z4, t, scale, p0Low);
        const double dt = (0.25 * dz) * scale;
        const double extra = dt * (cl.p[1] + cl.p[2] * (2.0 * t + dt)) + p0Low;
        const double rest = Table::poly9Rest(cl.p, t);
        const double s0 = cl.p[0] - logC_;
        const double v = s0 - cl.p[0];
        const double e0 = (cl.p[0] - (s0 - v)) - (logC_ + v);
        return s0 + (rest + ((e0 - logCLow_) + extra));
    }

    /**
     * Exact root s = x + c - omega(z) (z = ln c + x + c, 4 z = z4) for
     * c <= 0.04 (the series covers the small drives, where x + c and omega
     * would cancel) and z < kOctaveMax, from the full-precision table: x + c - omega
     * below z = 0 (x plus the -omega expansion, summed exactly, then c), and
     * ln(omega) - ln c above. Below z = kLo, omega < e^-40 and the root is x + c.
     */
    [[nodiscard]] double solveTable(double x, double z4) const noexcept
    {
        using Table = detail::DiodeOmegaTable;
        if (z4 < 4.0 * Table::kHi)
        {
            if (!(z4 >= 4.0 * Table::kLo)) return x + c_;
            double t = 0.0;
            int index = 0;
            const auto& n = exact_->node(z4, t, index);
            const double rest = Table::poly9Rest(n.p, t);
            if (index < Table::kFirstLogNode)
            {
                const double s0 = x + n.p[0];               // |x| > omega
                const double e0 = n.p[0] - (s0 - x);
                return s0 + ((rest + e0) + c_);
            }
            const double s0 = n.p[0] - logC_;               // -ln c > |ln omega|
            const double e0 = n.p[0] - (s0 + logC_);
            return s0 + ((rest + e0) - logCLow_);
        }
        double t = 0.0;
        const auto& cl = exact_->cell(0.25 * z4, t);
        const double rest = Table::poly9Rest(cl.p, t);
        const double s0 = cl.p[0] - logC_;
        const double v = s0 - cl.p[0];
        const double e0 = (cl.p[0] - (s0 - v)) - (logC_ + v);   // two-sum error
        return s0 + ((rest + e0) - logCLow_);
    }

    Tree& tree_;
    T is_, nvt_;
    const detail::WrightOmega* omega_;
    const detail::DiodeOmegaTable* exact_;
    double v_ = 0.0;
    double kCached_ = -1.0, nvtCached_ = -1.0;
    double invNvt_ = 1.0, c_ = 1.0, logC_ = 0.0;
    double logCc4_ = 0.0, logCLow_ = 0.0;   ///< 4 (logC_ + c), ln c - logC_ (table path).
    double invNvt4_ = 4.0, logCcLow4_ = 0.0; ///< 4 / nVt, 4 (ln c + c) - logCc4_.
    double invBeta_ = 1.0, seriesLimit_ = -1.0;   ///< Series path below |a| / nVt = seriesLimit_.
    double series_[14] {};                        ///< y^2 .. y^15 coefficients.
    int steps_ = 1;   ///< 0: one Newton step, 1: one Halley step, else Halley until converged.
    /// The table root is not used beyond this c (the rounding of z, which
    /// grows with c, would pass what its compensation holds), nor the local
    /// expansions beyond kLocalMaxC (from there on the forward table drive
    /// passes the end of the table): the correction steps keep those
    /// constants.
    static constexpr double kTableMaxC = 1e7;
    static constexpr double kLocalMaxC = 1e5;
    static constexpr int kSettle = 256;           ///< Samples the constants hold before a build.
    bool large_ = false;                          ///< 0.0417 < c <= kTableMaxC.
    bool localPending_ = false;                   ///< Local expansions still to build.
    int localPhase_ = 0;                          ///< 0: series and grid, 1: below the origin, 2: above.
    int settle_ = 0;
    int reverseCount_ = 0;                        ///< Cells below the origin (the first ones).
    double longLimit_ = 0.0;                      ///< Series of the constant below this |x|.
    double tableForward_ = 0.0, tableReverse_ = 0.0;   ///< Table root from these drives outwards.
    double forwardTo_ = -1.0;                     ///< Cells above the origin answer below this x.
    double reverseFrom_ = std::numeric_limits<double>::infinity();
    double reverseTo_ = -std::numeric_limits<double>::infinity();   ///< Cells below the origin: [from, to).
    double localOffset_ = 0.0;                    ///< Offset of the cell grid.
    std::uint64_t forwardFirstKey_ = 0, reverseFirstKey_ = 0;
    double long_[32] {};                          ///< y^2 .. y^33 coefficients of the constant.
    detail::DiodeLocal::Builder localBuild_;
    detail::DiodeLocal::Cell local_[detail::DiodeLocal::kSingleMaxCells];
};

// ============================================================================
// R-type adaptor (arbitrary topologies)
// ============================================================================

/**
 * @brief N-port R-type adaptor for non-series/parallel interconnections.
 *
 * Port 0 faces up (toward the root) and is adapted: its resistance is set to
 * the Thevenin resistance seen into the network, making S(0,0) = 0. Ports
 * 1..N-1 connect the children, in declaration order. The topology is a list
 * of (node+, node-) pairs over a small node graph (-1 = ground reference).
 *
 * On every updatePorts() the scattering matrix is rebuilt from Modified
 * Nodal Analysis with the children's current port resistances -- O(nodes^3),
 * a few microseconds for typical circuits, fine for pot automation.
 *
 * @tparam T        Sample type of the wave interface.
 * @tparam Children Child node types, one per port 1..N-1.
 */
template <FloatType T, typename... Children>
class RType
{
    template <FloatType> friend class ToneStackFMV;
public:
    static constexpr int kNumPorts = static_cast<int>(sizeof...(Children)) + 1;
    static constexpr int kMaxNodes = 12;
    static_assert(sizeof...(Children) >= 1, "RType needs at least one child");

    /**
     * @param portNodes (node+, node-) per port, port 0 first; -1 is ground.
     * @param numNodes  Number of non-ground nodes in the graph.
     * @param children  Child elements bound to ports 1..N-1, in order.
     */
    RType(const std::array<std::pair<int, int>, static_cast<size_t>(kNumPorts)>& portNodes,
          int numNodes, Children&... children) noexcept
        : children_(children...), portNodes_(portNodes),
          numNodes_(std::clamp(numNodes, 1, kMaxNodes))
    {
        assert(numNodes >= 1 && numNodes <= kMaxNodes);
        // Release-safe node sanitising: an out-of-range node index would
        // stamp the conductance/RHS arrays out of bounds. Clamp into
        // [-1, numNodes_-1] (-1 = ground); a clamped topology is wrong but
        // cannot corrupt memory.
        for (auto& [p, m] : portNodes_)
        {
            assert(p >= -1 && p < numNodes_ && m >= -1 && m < numNodes_);
            p = std::clamp(p, -1, numNodes_ - 1);
            m = std::clamp(m, -1, numNodes_ - 1);
        }
    }

    void prepare(double sampleRate) noexcept
    {
        std::apply([&](auto&... ch) { (ch.prepare(sampleRate), ...); }, children_);
    }

    void updatePorts() noexcept
    {
        std::apply([&](auto&... ch) { (ch.updatePorts(), ...); }, children_);

        // Gather child port resistances (port 0 resolved by adaptation).
        {
            int i = 1;
            std::apply([&](auto&... ch)
                       { ((portR_[static_cast<size_t>(i++)] = ch.portResistance()), ...); },
                       children_);
        }

        // 1) Thevenin resistance looking into the network from port 0:
        //    assemble MNA without port 0, inject 1 A across its nodes.
        assembleConductance(false);
        factor();
        double rhs[kMaxNodes] = {};
        stampCurrent(rhs, 0, 1.0);
        solve(rhs);
        double rth = portVoltage(rhs, 0);
        portR_[0] = std::clamp(rth, 1e-6, 1e12);

        // 2) Full MNA with every port loaded; S = 2M - I where column j of M
        //    is the port-voltage response to a_j = 1 (Norton: 1/R_j).
        assembleConductance(true);
        factor();
        for (int j = 0; j < kNumPorts; ++j)
        {
            double v[kMaxNodes] = {};
            stampCurrent(v, j, 1.0 / portR_[static_cast<size_t>(j)]);
            solve(v);
            for (int i = 0; i < kNumPorts; ++i)
            {
                const double m = portVoltage(v, i);
                s_[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    2.0 * m - (i == j ? 1.0 : 0.0);
            }
        }
        // Adaptation leaves no instantaneous reflection at the up port.
        s_[0][0] = 0.0;
    }

    void reset() noexcept
    {
        std::apply([&](auto&... ch) { (ch.reset(), ...); }, children_);
        a_.fill(0.0);
    }

    [[nodiscard]] double portResistance() const noexcept { return portR_[0]; }

    [[nodiscard]] T reflected() noexcept
    {
        int i = 1;
        std::apply([&](auto&... ch)
                   { ((a_[static_cast<size_t>(i++)] = static_cast<double>(ch.reflected())), ...); },
                   children_);
        double b0 = 0.0;
        for (int j = 1; j < kNumPorts; ++j)
            b0 += s_[0][static_cast<size_t>(j)] * a_[static_cast<size_t>(j)];
        return static_cast<T>(b0);
    }

    void incident(T aUp) noexcept
    {
        a_[0] = static_cast<double>(aUp);
        int i = 1;
        std::apply([&](auto&... ch)
                   {
                       ((ch.incident(static_cast<T>(rowDot(i))), ++i), ...);
                   },
                   children_);
    }

private:
    // Preserve a topology-proven zero-current voltage mode after the nodal
    // solve: S*v = v. For the FMV network the source and three coupling
    // capacitors share a common DC voltage; every resistor has zero voltage.
    // Correcting only the final nonzero column removes factorization roundoff
    // in this identity without adding arithmetic to sample processing.
    void preserveDcMode(const std::array<double, static_cast<size_t>(kNumPorts)>& mode) noexcept
    {
        int pivot = kNumPorts - 1;
        while (pivot > 0 && mode[static_cast<size_t>(pivot)] == 0.0) --pivot;
        if (pivot == 0) return; // Keep the adapted up-port reflection at zero.
        for (int i = 0; i < kNumPorts; ++i)
        {
            double remainder = 0.0;
            for (int j = 0; j < kNumPorts; ++j)
                if (j != pivot)
                    remainder += s_[static_cast<size_t>(i)][static_cast<size_t>(j)]
                               * mode[static_cast<size_t>(j)];
            s_[static_cast<size_t>(i)][static_cast<size_t>(pivot)] =
                (mode[static_cast<size_t>(i)] - remainder) / mode[static_cast<size_t>(pivot)];
        }
    }

    [[nodiscard]] double rowDot(int row) const noexcept
    {
        double acc = 0.0;
        for (int j = 0; j < kNumPorts; ++j)
            acc += s_[static_cast<size_t>(row)][static_cast<size_t>(j)] * a_[static_cast<size_t>(j)];
        return acc;
    }

    void assembleConductance(bool includePort0) noexcept
    {
        for (auto& row : g_) row.fill(0.0);
        for (int j = includePort0 ? 0 : 1; j < kNumPorts; ++j)
        {
            const double g = 1.0 / portR_[static_cast<size_t>(j)];
            const int p = portNodes_[static_cast<size_t>(j)].first;
            const int m = portNodes_[static_cast<size_t>(j)].second;
            if (p >= 0) g_[static_cast<size_t>(p)][static_cast<size_t>(p)] += g;
            if (m >= 0) g_[static_cast<size_t>(m)][static_cast<size_t>(m)] += g;
            if (p >= 0 && m >= 0)
            {
                g_[static_cast<size_t>(p)][static_cast<size_t>(m)] -= g;
                g_[static_cast<size_t>(m)][static_cast<size_t>(p)] -= g;
            }
        }
    }

    void stampCurrent(double* rhs, int port, double amps) const noexcept
    {
        const int p = portNodes_[static_cast<size_t>(port)].first;
        const int m = portNodes_[static_cast<size_t>(port)].second;
        if (p >= 0) rhs[p] += amps;
        if (m >= 0) rhs[m] -= amps;
    }

    [[nodiscard]] double portVoltage(const double* v, int port) const noexcept
    {
        const int p = portNodes_[static_cast<size_t>(port)].first;
        const int m = portNodes_[static_cast<size_t>(port)].second;
        return (p >= 0 ? v[p] : 0.0) - (m >= 0 ? v[m] : 0.0);
    }

    /** @brief LU factorization with partial pivoting of the node matrix. */
    void factor() noexcept
    {
        const int n = numNodes_;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
                lu_[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                    g_[static_cast<size_t>(i)][static_cast<size_t>(j)];
            piv_[static_cast<size_t>(i)] = i;
        }
        for (int k = 0; k < n; ++k)
        {
            int p = k;
            for (int i = k + 1; i < n; ++i)
                if (std::abs(lu_[static_cast<size_t>(i)][static_cast<size_t>(k)])
                    > std::abs(lu_[static_cast<size_t>(p)][static_cast<size_t>(k)]))
                    p = i;
            if (p != k)
            {
                std::swap(lu_[static_cast<size_t>(p)], lu_[static_cast<size_t>(k)]);
                std::swap(piv_[static_cast<size_t>(p)], piv_[static_cast<size_t>(k)]);
            }
            double d = lu_[static_cast<size_t>(k)][static_cast<size_t>(k)];
            if (std::abs(d) < 1e-300)
                d = (d >= 0.0 ? 1e-300 : -1e-300);
            const double invD = 1.0 / d;
            for (int i = k + 1; i < n; ++i)
            {
                const double f = lu_[static_cast<size_t>(i)][static_cast<size_t>(k)] * invD;
                lu_[static_cast<size_t>(i)][static_cast<size_t>(k)] = f;
                for (int j = k + 1; j < n; ++j)
                    lu_[static_cast<size_t>(i)][static_cast<size_t>(j)]
                        -= f * lu_[static_cast<size_t>(k)][static_cast<size_t>(j)];
            }
        }
    }

    /** @brief Solves the factored system in place (rhs -> solution). */
    void solve(double* rhs) const noexcept
    {
        const int n = numNodes_;
        double y[kMaxNodes];
        for (int i = 0; i < n; ++i)
            y[i] = rhs[piv_[static_cast<size_t>(i)]];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j)
                y[i] -= lu_[static_cast<size_t>(i)][static_cast<size_t>(j)] * y[j];
        for (int i = n - 1; i >= 0; --i)
        {
            for (int j = i + 1; j < n; ++j)
                y[i] -= lu_[static_cast<size_t>(i)][static_cast<size_t>(j)] * y[j];
            double d = lu_[static_cast<size_t>(i)][static_cast<size_t>(i)];
            if (std::abs(d) < 1e-300)
                d = (d >= 0.0 ? 1e-300 : -1e-300);
            y[i] /= d;
        }
        for (int i = 0; i < n; ++i)
            rhs[i] = y[i];
    }

    std::tuple<Children&...> children_;
    std::array<std::pair<int, int>, static_cast<size_t>(kNumPorts)> portNodes_;
    int numNodes_;

    std::array<double, static_cast<size_t>(kNumPorts)> portR_ {};
    std::array<double, static_cast<size_t>(kNumPorts)> a_ {};
    std::array<std::array<double, static_cast<size_t>(kNumPorts)>,
               static_cast<size_t>(kNumPorts)> s_ {};

    std::array<std::array<double, kMaxNodes>, kMaxNodes> g_ {};
    std::array<std::array<double, kMaxNodes>, kMaxNodes> lu_ {};
    std::array<int, kMaxNodes> piv_ {};
};

// ============================================================================
// Fender '59 Bassman FMV tone stack (reference R-type circuit)
// ============================================================================

/**
 * @brief Exact Fender '59 Bassman treble/bass/middle tone stack.
 *
 * Topology and component values after Yeh & Smith, "Discretization of the
 * '59 Fender Bassman Tone Stack" (DAFx-06), figure 1: C1 = 250 pF into the
 * treble pot (R1 = 250 k ohm, output at the wiper), R4 = 56 k ohm into the slope
 * node, C2 = C3 = 20 nF from the slope node to the bass rheostat (l*R2,
 * R2 = 1 M ohm) and to the middle pot wiper (R3 = 25 k ohm) respectively. The
 * source is modelled with a series output resistance (cathode-follower
 * buffer) and the wiper drives a grid-load resistance.
 *
 * Solved exactly as one 12-port R-type. The suite verifies it sample-exact
 * against the bilinear transform of the paper's symbolic transfer function.
 * Controls are not orthogonal -- that interaction is the circuit's character.
 */
template <FloatType T>
class ToneStackFMV
{
public:
    /**
     * @param sourceResistance Driving stage output impedance (default 1 k ohm).
     * @param loadResistance   Following stage grid load (default 1 M ohm).
     */
    explicit ToneStackFMV(double sourceResistance = 1e3, double loadResistance = 1e6)
        : rOut_(static_cast<T>(sourceResistance)), c1_(T(0.25e-9)),
          r1Top_(T(125e3)), r1Bot_(T(125e3)), r4_(T(56e3)), c2_(T(20e-9)),
          r2_(T(500e3)), c3_(T(20e-9)), r3Top_(T(12.5e3)), r3Bot_(T(12.5e3)),
          rLoad_(static_cast<T>(loadResistance)),
          rtype_({ { { kSrc, -1 },          // port 0: ideal source root
                    { kSrc, kVi },          // rOut
                    { kVi, kA },            // C1
                    { kA, kVo },            // (1-t) R1
                    { kVo, kB },            // t R1
                    { kVi, kS },            // R4
                    { kS, kB },             // C2
                    { kB, kC },             // l R2 (rheostat)
                    { kS, kW },             // C3 to the middle wiper
                    { kC, kW },             // (1-m) R3
                    { kW, -1 },             // m R3 to ground
                    { kVo, -1 } } },        // grid load
                 kNumNodes,
                 rOut_, c1_, r1Top_, r1Bot_, r4_, c2_, r2_, c3_, r3Top_, r3Bot_, rLoad_),
          root_(rtype_)
    {
    }

    /** @brief Prepares the network (allocates nothing). */
    void prepare(double sampleRate) noexcept
    {
        root_.prepare(sampleRate);
        setControls(treble_, bass_, middle_);
    }

    /** @brief Clears capacitor states. RT-safe. */
    void reset() noexcept { root_.reset(); }

    /** @brief Copies the capacitor history from an identically prepared stack.
     *
     * Stream-owner only: neither stack may be processed concurrently. Both
     * stacks must have the same sample rate and controls. Port connections
     * remain owned by the destination; no allocation or scattering rebuild
     * occurs. An optional input offset shifts all three coupling capacitors
     * by the same DC voltage; the subsequent input must use that reference.
     * The stack blocks DC, so this reference shift preserves its output.
     */
    void copyStateFrom(const ToneStackFMV& source, T inputOffset = T(0)) noexcept
    {
        c1_ = source.c1_;
        c2_ = source.c2_;
        c3_ = source.c3_;
        if (inputOffset != T(0))
        {
            c1_.offsetVoltage(inputOffset);
            c2_.offsetVoltage(inputOffset);
            c3_.offsetVoltage(inputOffset);
        }
    }

    /**
     * @brief Sets the three controls, [0, 1] each.
     *
     * Treble and middle pots are linear, the bass pot is a log-taper
     * rheostat (squared approximation), matching the original circuit.
     */
    void setControls(T treble, T bass, T middle) noexcept
    {
        treble_ = std::clamp(treble, T(0), T(1));
        bass_ = std::clamp(bass, T(0), T(1));
        middle_ = std::clamp(middle, T(0), T(1));

        const double t = static_cast<double>(treble_);
        const double l = static_cast<double>(bass_) * static_cast<double>(bass_);
        const double m = static_cast<double>(middle_);
        constexpr double kRmin = 0.5;   // keeps the node matrix well-posed

        r1Top_.setResistance(static_cast<T>((1.0 - t) * 250e3 + kRmin));
        r1Bot_.setResistance(static_cast<T>(t * 250e3 + kRmin));
        r2_.setResistance(static_cast<T>(l * 1e6 + kRmin));
        r3Top_.setResistance(static_cast<T>((1.0 - m) * 25e3 + kRmin));
        r3Bot_.setResistance(static_cast<T>(m * 25e3 + kRmin));
        rtype_.updatePorts();           // rebuild scattering, keep states
        rtype_.preserveDcMode({1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0});
    }

    /** @brief Processes one sample (input volts -> wiper volts). */
    [[nodiscard]] T processSample(T input) noexcept
    {
        root_.setVoltage(input);
        root_.process();
        return rLoad_.getVoltage();
    }

    /**
     * @brief Continuous-time state space of the same network.
     *
     * States are the capacitor voltages (C1: input to treble pot, C2: slope
     * node to bass rheostat, C3: slope node to middle wiper), the input is the
     * source voltage and the output is the wiper voltage:
     * dx/dt = a x + b u, y = c x + d u. The matrices are obtained by nodal
     * analysis of the identical element values and topology used by the
     * R-type adaptor, with each capacitor replaced by a voltage source.
     * a = -diag(capacitance)^-1 G for a symmetric positive definite G.
     * Allocation free; cost of a 10x10 elimination.
     */
    struct AnalogStateSpace
    {
        double a[3][3];
        double b[3];
        double c[3];
        double d;
        double capacitance[3];
    };

    [[nodiscard]] AnalogStateSpace analogStateSpace() const noexcept
    {
        // Unknowns: node voltages kVi..kW (7) then the three capacitor currents.
        constexpr int kN = 10;
        struct Element { int p, m; double value; bool capacitor; };
        const Element elements[] = {
            { kSrc, kVi, static_cast<double>(rOut_.getResistance()), false },
            { kVi, kA, static_cast<double>(c1_.getCapacitance()), true },
            { kA, kVo, static_cast<double>(r1Top_.getResistance()), false },
            { kVo, kB, static_cast<double>(r1Bot_.getResistance()), false },
            { kVi, kS, static_cast<double>(r4_.getResistance()), false },
            { kS, kB, static_cast<double>(c2_.getCapacitance()), true },
            { kB, kC, static_cast<double>(r2_.getResistance()), false },
            { kS, kW, static_cast<double>(c3_.getCapacitance()), true },
            { kC, kW, static_cast<double>(r3Top_.getResistance()), false },
            { kW, -1, static_cast<double>(r3Bot_.getResistance()), false },
            { kVo, -1, static_cast<double>(rLoad_.getResistance()), false } };
        double m[kN][kN] {};
        double rhs[kN][4] {};   // columns: unit C1, C2, C3 voltages, unit source
        int capIndex = 0;
        AnalogStateSpace out {};
        for (const auto& e : elements)
        {
            const int p = e.p - 1, q = e.m < 0 ? -1 : e.m - 1;   // kSrc maps to -1
            if (e.capacitor)
            {
                const int row = 7 + capIndex;
                if (p >= 0) { m[p][row] += 1.0; m[row][p] += 1.0; }
                if (q >= 0) { m[q][row] -= 1.0; m[row][q] -= 1.0; }
                rhs[row][capIndex] = 1.0;
                out.capacitance[capIndex] = e.value;
                ++capIndex;
                continue;
            }
            const double g = 1.0 / e.value;
            if (e.p == kSrc)
            {
                // Source node is driven: its conductance stamps the RHS.
                m[q][q] += g;
                rhs[q][3] += g;
                continue;
            }
            if (p >= 0) m[p][p] += g;
            if (q >= 0) m[q][q] += g;
            if (p >= 0 && q >= 0) { m[p][q] -= g; m[q][p] -= g; }
        }
        // Gaussian elimination with partial pivoting, four right-hand sides.
        for (int k = 0; k < kN; ++k)
        {
            int piv = k;
            for (int i = k + 1; i < kN; ++i)
                if (std::abs(m[i][k]) > std::abs(m[piv][k])) piv = i;
            if (piv != k)
                for (int j = 0; j < kN; ++j) std::swap(m[piv][j], m[k][j]);
            if (piv != k)
                for (int j = 0; j < 4; ++j) std::swap(rhs[piv][j], rhs[k][j]);
            for (int i = k + 1; i < kN; ++i)
            {
                const double f = m[i][k] / m[k][k];
                if (f == 0.0) continue;
                for (int j = k; j < kN; ++j) m[i][j] -= f * m[k][j];
                for (int j = 0; j < 4; ++j) rhs[i][j] -= f * rhs[k][j];
            }
        }
        for (int i = kN - 1; i >= 0; --i)
            for (int j = 0; j < 4; ++j)
            {
                double v = rhs[i][j];
                for (int k = i + 1; k < kN; ++k) v -= m[i][k] * rhs[k][j];
                rhs[i][j] = v / m[i][i];
            }
        // Capacitor current flows from its first node to its second node.
        for (int j = 0; j < 4; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                const double dv = rhs[7 + k][j] / out.capacitance[k];
                if (j < 3) out.a[k][j] = dv;
                else out.b[k] = dv;
            }
            if (j < 3) out.c[j] = rhs[kVo - 1][j];
            else out.d = rhs[kVo - 1][j];
        }
        return out;
    }

private:
    static constexpr int kSrc = 0, kVi = 1, kA = 2, kVo = 3,
                         kB = 4, kS = 5, kC = 6, kW = 7;
    static constexpr int kNumNodes = 8;

    Resistor<T> rOut_;
    Capacitor<T> c1_;
    Resistor<T> r1Top_, r1Bot_, r4_;
    Capacitor<T> c2_;
    Resistor<T> r2_;
    Capacitor<T> c3_;
    Resistor<T> r3Top_, r3Bot_, rLoad_;

    using StackRType = RType<T, Resistor<T>, Capacitor<T>, Resistor<T>, Resistor<T>,
                             Resistor<T>, Capacitor<T>, Resistor<T>, Capacitor<T>,
                             Resistor<T>, Resistor<T>, Resistor<T>>;
    StackRType rtype_;
    IdealVoltageSourceRoot<T, StackRType> root_;

    T treble_ = T(0.5), bass_ = T(0.5), middle_ = T(0.5);
};

} // namespace wdf
} // namespace dspark
