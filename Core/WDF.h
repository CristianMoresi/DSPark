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
 * Beyond series/parallel trees, the header provides an **R-type adaptor**
 * (after K. Werner's thesis): an N-port connector for arbitrary — non
 * series/parallel — topologies. Its scattering matrix is derived numerically
 * from the interconnection network via Modified Nodal Analysis: each port is
 * replaced by its instantaneous Thévenin equivalent (source a_i, resistance
 * R_i), the linear node system is solved once per topology/parameter change,
 * and S = 2M − I where M maps incident waves to port voltages. The up-facing
 * port is adapted to the Thévenin resistance seen looking into the network,
 * so R-types nest under any root. `ToneStackFMV` builds on it: the exact
 * Fender '59 Bassman treble/bass/middle network (topology and verification
 * transfer function after Yeh & Smith, DAFx-06).
 *
 * Scope note: nonlinear roots cover one-port nonlinearities; multi-port
 * nonlinear roots (triode inside the WDF tree) remain future work.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>

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

// ============================================================================
// R-type adaptor (arbitrary topologies)
// ============================================================================

/**
 * @brief N-port R-type adaptor for non-series/parallel interconnections.
 *
 * Port 0 faces up (toward the root) and is adapted: its resistance is set to
 * the Thévenin resistance seen into the network, making S(0,0) = 0. Ports
 * 1..N-1 connect the children, in declaration order. The topology is a list
 * of (node+, node-) pairs over a small node graph (-1 = ground reference).
 *
 * On every updatePorts() the scattering matrix is rebuilt from Modified
 * Nodal Analysis with the children's current port resistances — O(nodes³),
 * a few microseconds for typical circuits, fine for pot automation.
 *
 * @tparam T        Sample type of the wave interface.
 * @tparam Children Child node types, one per port 1..N-1.
 */
template <FloatType T, typename... Children>
class RType
{
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

        // 1) Thévenin resistance looking into the network from port 0:
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
 * treble pot (R1 = 250 kΩ, output at the wiper), R4 = 56 kΩ into the slope
 * node, C2 = C3 = 20 nF from the slope node to the bass rheostat (l·R2,
 * R2 = 1 MΩ) and to the middle pot wiper (R3 = 25 kΩ) respectively. The
 * source is modelled with a series output resistance (cathode-follower
 * buffer) and the wiper drives a grid-load resistance.
 *
 * Solved exactly as one 12-port R-type. The suite verifies it sample-exact
 * against the bilinear transform of the paper's symbolic transfer function.
 * Controls are not orthogonal — that interaction is the circuit's character.
 */
template <FloatType T>
class ToneStackFMV
{
public:
    /**
     * @param sourceResistance Driving stage output impedance (default 1 kΩ).
     * @param loadResistance   Following stage grid load (default 1 MΩ).
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
    }

    /** @brief Processes one sample (input volts -> wiper volts). */
    [[nodiscard]] T processSample(T input) noexcept
    {
        root_.setVoltage(input);
        root_.process();
        return rLoad_.getVoltage();
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
