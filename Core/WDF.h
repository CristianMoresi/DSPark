// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WDF.h
 * @brief Wave Digital Filters: physical circuit modelling building blocks.
 *
 * Implements the classical WDF toolkit (Fettweis 1986): one-port elements
 * (resistor, capacitor, inductor, resistive voltage source), adapted
 * three-port series/parallel connectors, a polarity inverter, and circuit
 * roots — an ideal voltage source for linear networks and Newton-Raphson
 * nonlinear roots (Shockley diode, antiparallel diode pair).
 *
 * Wave convention: voltage waves with a = v + R·i (incident) and
 * b = v − R·i (reflected), so v = (a+b)/2 and i = (a−b)/(2R). Reactances
 * use the bilinear (trapezoidal) discretization: a WDF network is therefore
 * sample-exact against the bilinear transform of its analog transfer
 * function, which is how this header is verified (RC and RLC against the
 * analytic discrete response; diode clippers against a high-precision
 * trapezoidal reference solve of the circuit ODE — the same integration
 * SPICE `.tran` uses).
 *
 * Trees are composed statically from references — no virtual dispatch, all
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
 * Scope note: nonlinear roots cover one-port nonlinearities. Multi-port
 * nonlinear roots (triode via R-type adaptors, Werner et al.) are a later,
 * separate step — the planned TubePreamp models its triode stage outside
 * the WDF and uses WDF for the (linear) tone stack.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>

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

    void prepare(double sampleRate) noexcept { fs_ = sampleRate; }
    void updatePorts() noexcept { R_ = 1.0 / (2.0 * fs_ * static_cast<double>(value_)); }
    void reset() noexcept { state_ = 0; a_ = 0; b_ = 0; }

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
 * @brief Voltage source with series resistance (Thévenin leaf): b = Vs.
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
        tree_.incident(T(2) * vs_ - a);
    }

private:
    Tree& tree_;
    T vs_ = 0;
};

namespace detail {

/**
 * @brief Bisection-safeguarded Newton-Raphson for monotonic wave equations.
 *
 * Solves f(v) = v + i(v)·R_total - a = 0 where i(v) is monotonically
 * increasing, so the solution is bracketed by [min(0,a), max(0,a)]. Newton
 * steps that leave the shrinking bracket fall back to bisection — guaranteed
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

} // namespace detail

/**
 * @brief Antiparallel diode pair root (the classic clipper nonlinearity).
 *
 * Shockley model: i(v) = 2 Is sinh(v / (n Vt)). Defaults approximate a
 * 1N4148 (Is = 2.52 nA, n = 1.752, Vt = 25.85 mV).
 */
template <FloatType T, typename Tree>
class DiodePairRoot
{
public:
    explicit DiodePairRoot(Tree& tree, T saturationCurrent = T(2.52e-9),
                           T idealityTimesVt = T(1.752 * 0.02585)) noexcept
        : tree_(tree), is_(saturationCurrent), nvt_(idealityTimesVt) {}

    void setSaturationCurrent(T amps) noexcept { is_ = std::max(amps, T(1e-15)); }
    void setIdealityTimesVt(T volts) noexcept { nvt_ = std::max(volts, T(1e-4)); }

    void prepare(double sampleRate) noexcept
    {
        tree_.prepare(sampleRate);
        tree_.updatePorts();
        tree_.reset();
        v_ = 0.0;
    }
    void reset() noexcept { tree_.reset(); v_ = 0.0; }

    void process() noexcept
    {
        const double a = static_cast<double>(tree_.reflected());
        const double k = 2.0 * tree_.portResistance() * static_cast<double>(is_);
        const double nvt = static_cast<double>(nvt_);

        // Far-field analytic seed: neglecting the linear term in
        // v + k sinh(v/nVt) = a gives v0 = nVt asinh(a/k), within ~nVt of the
        // root at any drive level (deep in the exponential, plain Newton
        // would crawl at ~nVt per iteration). Newton then converges in 1-3.
        const double seed = nvt * std::asinh(a / k);

        v_ = detail::solveMonotonic(a, seed, [k, nvt, a](double v, double& f, double& fp)
        {
            // Guard the argument: cosh overflows around |x| > 710 in double,
            // far outside any solution the bracket allows anyway.
            const double x = std::clamp(v / nvt, -700.0, 700.0);
            f  = v + k * std::sinh(x) - a;
            fp = 1.0 + (k / nvt) * std::cosh(x);
        });

        tree_.incident(static_cast<T>(2.0 * v_ - a));
    }

    /** @brief Voltage across the pair (the clipper output). */
    [[nodiscard]] T getVoltage() const noexcept { return static_cast<T>(v_); }

private:
    Tree& tree_;
    T is_, nvt_;
    double v_ = 0.0;
};

/**
 * @brief Single Shockley diode root: i(v) = Is (e^{v/(n Vt)} - 1).
 *
 * Conducts for positive v, blocks (down to -Is) for negative v.
 */
template <FloatType T, typename Tree>
class DiodeRoot
{
public:
    explicit DiodeRoot(Tree& tree, T saturationCurrent = T(2.52e-9),
                       T idealityTimesVt = T(1.752 * 0.02585)) noexcept
        : tree_(tree), is_(saturationCurrent), nvt_(idealityTimesVt) {}

    void setSaturationCurrent(T amps) noexcept { is_ = std::max(amps, T(1e-15)); }
    void setIdealityTimesVt(T volts) noexcept { nvt_ = std::max(volts, T(1e-4)); }

    void prepare(double sampleRate) noexcept
    {
        tree_.prepare(sampleRate);
        tree_.updatePorts();
        tree_.reset();
        v_ = 0.0;
    }
    void reset() noexcept { tree_.reset(); v_ = 0.0; }

    void process() noexcept
    {
        const double a = static_cast<double>(tree_.reflected());
        const double k = tree_.portResistance() * static_cast<double>(is_);
        const double nvt = static_cast<double>(nvt_);

        // Far-field analytic seed: forward drive lands near
        // nVt ln(1 + a/k); reverse drive blocks, so the root sits near a.
        const double seed = (a > 0.0) ? nvt * std::log1p(a / k) : a;

        v_ = detail::solveMonotonic(a, seed, [k, nvt, a](double v, double& f, double& fp)
        {
            const double x = std::clamp(v / nvt, -700.0, 700.0);
            f  = v + k * std::expm1(x) - a;
            fp = 1.0 + (k / nvt) * std::exp(x);
        });

        tree_.incident(static_cast<T>(2.0 * v_ - a));
    }

    /** @brief Voltage across the diode. */
    [[nodiscard]] T getVoltage() const noexcept { return static_cast<T>(v_); }

private:
    Tree& tree_;
    T is_, nvt_;
    double v_ = 0.0;
};

} // namespace wdf
} // namespace dspark
