// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Adaa.h
 * @brief Second-order antiderivative antialiasing (ADAA) for memoryless curves.
 *
 * Bilbao, Esqueda, Parker and Valimaki ("Antiderivative antialiasing for
 * memoryless nonlinearities", IEEE Signal Processing Letters 24(7), 2017)
 * replace y[n] = f(x[n]) by twice the second divided difference of the
 * second antiderivative F2:
 *
 *   y[n] = 2 F2[x[n], x[n-1], x[n-2]]
 *        = 2 / (x[n] - x[n-2]) * (D(x[n], x[n-1]) - D(x[n-1], x[n-2])),
 *   D(a, b) = (F2(a) - F2(b)) / (a - b),
 *
 * which is the average of f over the linear B-spline with knots at the three
 * inputs. The output is delayed by exactly one sample and attenuates a
 * linear signal by (1 + 2 cos w) / 3 (-1.3 dB at a quarter of the rate),
 * which is why it belongs inside an oversampled stage.
 *
 * Numerics. The divided differences cancel: with F2 of order u^2/2 at a
 * large input u, the direct formula loses about log10(u^2 / dx^2) digits. The
 * curves here therefore evaluate each saturated region exactly (its F2 is a
 * quadratic plus, for tanh, an exponentially small remainder, and for the
 * rational knee a w ln w term with a stable closed form), and the generic
 * path falls back to Taylor forms when inputs nearly coincide:
 * D(a, b) = F1(m) + f'(m) (a - b)^2 / 24 for |a - b| < 1e-3, the confluent
 * form 2 F2[xb, xb, x1] for |x[n] - x[n-2]| < 1e-4, and f(mean) when all
 * three lie within 1e-4. Everything runs in double.
 *
 * Curves (unit ceiling, odd unless stated): hard clip, tanh, the clamped
 * fastSin() polynomial of the Clipper's Analog mode, the golden-ratio
 * rational knee, and sin (not saturating, for the wavefolder).
 *
 * Threading: Adaa2 is per-stream state owned by one thread; the curves are
 * pure functions. No allocation. Dependencies: DspMath.h, LogCosh.h.
 */
#include "../DspMath.h"
#include "LogCosh.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

namespace dspark::detail
{

/** One input sample with the per-point values a curve needs. */
struct AdaaPoint
{
    double u = 0.0;    ///< Input.
    double g = 0.0;    ///< Second antiderivative F2(u).
    double aux = 0.0;  ///< Curve specific (tanh remainder, knee logarithm).
};

/** Generic second-difference machinery for a function family G with known
 *  derivatives d1 = G', d2 = G'', d3 = G'''. */
template <class Family>
struct AdaaSecondDifference
{
    static constexpr double kTolDiv = 1e-3;
    static constexpr double kTolDd2 = 1e-4;

    [[nodiscard]] static double div(double a, double b, double ga, double gb) noexcept
    {
        const double d = a - b;
        if (std::abs(d) > kTolDiv) return (ga - gb) / d;
        const double m = 0.5 * (a + b);
        return Family::d1(m) + Family::d3(m) * (d * d * (1.0 / 24.0));
    }

    /** 2 G[x0, x1, x2]. */
    [[nodiscard]] static double twice(double x0, double x1, double x2,
                                      double g0, double g1, double g2) noexcept
    {
        const double d02 = x0 - x2;
        if (std::abs(d02) > kTolDd2)
            return 2.0 * (div(x0, x1, g0, g1) - div(x1, x2, g1, g2)) / d02;
        // Confluent form: x0 and x2 merge into their mean.
        const double xb = 0.5 * (x0 + x2), e = xb - x1;
        if (std::abs(e) > kTolDd2)
            return 2.0 * (Family::d1(xb) - div(xb, x1, Family::g(xb), g1)) / e;
        return Family::d2((x0 + x1 + x2) * (1.0 / 3.0));
    }
};

// ---------------------------------------------------------------------------
// Hard clip: f = clamp(u, -1, 1)
// ---------------------------------------------------------------------------
struct AdaaHardClip
{
    static double g(double u) noexcept
    {
        const double a = std::abs(u);
        return a <= 1.0 ? u * u * u * (1.0 / 6.0)
                        : std::copysign(0.5 * a * a - 0.5 * a + 1.0 / 6.0, u);
    }
    static double d1(double u) noexcept
    {
        const double a = std::abs(u);
        return a <= 1.0 ? 0.5 * u * u : a - 0.5;
    }
    static double d2(double u) noexcept { return std::clamp(u, -1.0, 1.0); }
    static double d3(double u) noexcept { return std::abs(u) < 1.0 ? 1.0 : 0.0; }

    static AdaaPoint point(double u) noexcept { return {u, g(u), 0.0}; }

    /** Closed form for every input triple (see exact()), so Adaa2 keeps
     *  only the inputs for this curve. */
    static constexpr bool kClosedForm = true;
    static double value(double u0, double u1, double u2) noexcept
    {
        double y = 0.0;
        (void)exact({u0, 0.0, 0.0}, {u1, 0.0, 0.0}, {u2, 0.0, 0.0}, y);
        return y;
    }

    /** Exact in every case: clamp(t) = t - (t - 1)+ + (-1 - t)+, and the
     *  B-spline average of a ramp (t - tau)+ has a closed form with no
     *  subtraction of nearly equal terms, so a kink crossed in tiny steps
     *  costs no accuracy (the divided-difference formula lost up to five
     *  digits there). Three points on one linear piece return before any
     *  ramp is evaluated. */
    static bool exact(const AdaaPoint& p0, const AdaaPoint& p1, const AdaaPoint& p2,
                      double& y) noexcept
    {
        double lo = p0.u, mid = p1.u, hi = p2.u;
        if (lo > mid) std::swap(lo, mid);
        if (mid > hi) std::swap(mid, hi);
        if (lo > mid) std::swap(lo, mid);
        if (lo >= 1.0) { y = 1.0; return true; }
        if (hi <= -1.0) { y = -1.0; return true; }
        const double mean = (lo + mid + hi) * (1.0 / 3.0);
        if (hi <= 1.0 && lo >= -1.0) { y = mean; return true; }
        y = mean - rampAbove(lo, mid, hi, mean, 1.0) + rampBelow(lo, mid, hi, mean, -1.0);
        return true;
    }

    /** Average of (t - tau)+ over the linear B-spline with knots a <= b <= c. */
    static double rampAbove(double a, double b, double c, double mean, double tau) noexcept
    {
        if (tau >= c) return 0.0;
        if (tau <= a) return mean - tau;
        if (tau >= b)
        {
            const double r = c - tau;
            return r * r * r / (3.0 * (c - a) * (c - b));
        }
        const double l = tau - a;
        return (mean - tau) + l * l * l / (3.0 * (c - a) * (b - a));
    }

    /** Average of (tau - t)+ over the same B-spline (mirror of rampAbove). */
    static double rampBelow(double a, double b, double c, double mean, double tau) noexcept
    {
        if (tau <= a) return 0.0;
        if (tau >= c) return tau - mean;
        if (tau <= b)
        {
            const double l = tau - a;
            return l * l * l / (3.0 * (c - a) * (b - a));
        }
        const double r = c - tau;
        return (tau - mean) + r * r * r / (3.0 * (c - a) * (c - b));
    }
};

// ---------------------------------------------------------------------------
// tanh: F1 = log cosh, F2 = integral of log cosh
// ---------------------------------------------------------------------------
struct AdaaTanh
{
    /// Beyond |u| = 2, F2 = sign (u^2/2 - |u| ln 2 + pi^2/24) + R with
    /// |R| < 0.01: the quadratic part has an exact second difference.
    static constexpr double kOuter = 2.0;

    static double g(double u) noexcept { return integralLogCosh(u); }
    static double d1(double u) noexcept { return logCosh(u); }
    static double d2(double u) noexcept { return std::tanh(u); }
    static double d3(double u) noexcept { const double t = std::tanh(u); return 1.0 - t * t; }

    /** Remainder family R(u) = F2 - quadratic, on one side |u| >= kOuter. */
    struct Remainder
    {
        // The tail is negative, so the odd extension multiplies by the sign
        // (copysign would take its magnitude).
        static double g(double u) noexcept
        {
            const double r = tail(std::abs(u));
            return u < 0.0 ? -r : r;
        }
        static double d1(double u) noexcept { return std::log1p(std::exp(-2.0 * std::abs(u))); }
        static double d2(double u) noexcept
        {
            const double z = std::exp(-2.0 * std::abs(u));
            const double v = 2.0 * z / (1.0 + z);            // tanh(u) - sign(u)
            return u < 0.0 ? v : -v;
        }
        static double d3(double u) noexcept { const double t = std::tanh(u); return 1.0 - t * t; }
    };

    /** integralLogCoshTail for a >= 1, with the series cut to the terms
     *  that matter at the argument (z = e^{-2a} <= e^{-4} needs nine). */
    static double tail(double a) noexcept
    {
        if (a < 2.0) return integralLogCoshTail(a);
        if (a > 19.0) return 0.0;                 // |tail| < 1.6e-17
        const double z = std::exp(-2.0 * a);
        double li = -1.0 / 81.0;
        for (int k = 8; k >= 1; --k)
            li = li * z + (k % 2 == 0 ? 1.0 : -1.0) / (k * k);
        return 0.5 * (li * z);
    }

    static AdaaPoint point(double u) noexcept
    {
        const double a = std::abs(u);
        if (a < 1.0) return {u, integralLogCosh(u), 0.0};
        const double r = tail(a);
        return {u, std::copysign(0.5 * a * a - a * 0.6931471805599453 + (r + 0.4112335167120566), u),
                u < 0.0 ? -r : r};
    }

    static bool exact(const AdaaPoint& p0, const AdaaPoint& p1, const AdaaPoint& p2,
                      double& y) noexcept
    {
        const double lo = std::min(p0.u, std::min(p1.u, p2.u)), hi = std::max(p0.u, std::max(p1.u, p2.u));
        if (!(lo >= kOuter || hi <= -kOuter)) return false;
        const double s = lo >= kOuter ? 1.0 : -1.0;
        y = s + AdaaSecondDifference<Remainder>::twice(p0.u, p1.u, p2.u, p0.aux, p1.aux, p2.aux);
        return true;
    }
};

// ---------------------------------------------------------------------------
// Clamped fastSin() polynomial: f = p(clamp(u, -pi/2, pi/2)), p(pi/2) = 1
// ---------------------------------------------------------------------------
/** The fastSin() polynomial p(v) and its first two antiderivatives
 *  (constexpr, so the clamp-edge values fold into constants). */
struct AdaaSinePolynomialTerms
{
    static constexpr double poly(double v) noexcept
    {
        constexpr auto c = fastSinPolynomial<double>;
        const double v2 = v * v;
        return v * (c[0] + v2 * (c[1] + v2 * (c[2] + v2 * (c[3] + v2 * c[4]))));
    }
    static constexpr double poly1(double v) noexcept   // antiderivative, even
    {
        constexpr auto c = fastSinPolynomial<double>;
        const double v2 = v * v;
        return v2 * (c[0] / 2.0 + v2 * (c[1] / 4.0 + v2 * (c[2] / 6.0 + v2 * (c[3] / 8.0 + v2 * c[4] / 10.0))));
    }
    static constexpr double poly2(double v) noexcept   // second antiderivative, odd
    {
        constexpr auto c = fastSinPolynomial<double>;
        const double v2 = v * v;
        return v * v2 * (c[0] / 6.0 + v2 * (c[1] / 20.0 + v2 * (c[2] / 42.0 + v2 * (c[3] / 72.0 + v2 * c[4] / 110.0))));
    }
    static constexpr double polySlope(double v) noexcept
    {
        constexpr auto c = fastSinPolynomial<double>;
        const double v2 = v * v;
        return c[0] + v2 * (3.0 * c[1] + v2 * (5.0 * c[2] + v2 * (7.0 * c[3] + v2 * 9.0 * c[4])));
    }
};

struct AdaaSinePolynomial : AdaaSinePolynomialTerms
{
    static constexpr double kHalfPi = std::numbers::pi * 0.5;
    static constexpr double kEdge = poly(kHalfPi), kEdge1 = poly1(kHalfPi), kEdge2 = poly2(kHalfPi);

    static double g(double u) noexcept
    {
        const double a = std::abs(u);
        if (a <= kHalfPi) return poly2(u);
        const double e = a - kHalfPi;
        return std::copysign(kEdge2 + kEdge1 * e + 0.5 * kEdge * e * e, u);
    }
    static double d1(double u) noexcept
    {
        const double a = std::abs(u);
        return a <= kHalfPi ? poly1(u) : kEdge1 + kEdge * (a - kHalfPi);
    }
    static double d2(double u) noexcept { return poly(std::clamp(u, -kHalfPi, kHalfPi)); }
    static double d3(double u) noexcept { return std::abs(u) < kHalfPi ? polySlope(u) : 0.0; }

    static AdaaPoint point(double u) noexcept { return {u, g(u), 0.0}; }

    static bool exact(const AdaaPoint& p0, const AdaaPoint& p1, const AdaaPoint& p2,
                      double& y) noexcept
    {
        const double lo = std::min(p0.u, std::min(p1.u, p2.u)), hi = std::max(p0.u, std::max(p1.u, p2.u));
        if (lo >= kHalfPi) { y = kEdge; return true; }
        if (hi <= -kHalfPi) { y = -kEdge; return true; }
        return false;
    }
};

// ---------------------------------------------------------------------------
// Golden-ratio rational knee: f = u for |u| <= k = 1/phi, else
// sign (1 - r^2 / w) with r = 1 - k and w = |u| - k + r.
// ---------------------------------------------------------------------------
struct AdaaRationalKnee
{
    static constexpr double kKnee = 0.61803398874989484820;   // 1 / phi
    static constexpr double kRange = 1.0 - kKnee;
    static constexpr double kLogRange = -0.96242365011920689500; // ln(1 - 1/phi)

    static double g(double u) noexcept
    {
        const double a = std::abs(u);
        if (a <= kKnee) return u * u * u * (1.0 / 6.0);
        const double e = a - kKnee, w = e + kRange;
        return std::copysign(kKnee * kKnee * kKnee / 6.0 + 0.5 * kKnee * kKnee * e + 0.5 * e * e
                             - kRange * kRange * (w * (std::log(w) - kLogRange) - e), u);
    }
    static double d1(double u) noexcept
    {
        const double a = std::abs(u);
        if (a <= kKnee) return 0.5 * u * u;
        const double e = a - kKnee;
        return 0.5 * kKnee * kKnee + e - kRange * kRange * std::log1p(e / kRange);
    }
    static double d2(double u) noexcept
    {
        const double a = std::abs(u);
        return a <= kKnee ? u : std::copysign(1.0 - kRange * kRange / (a - kKnee + kRange), u);
    }
    static double d3(double u) noexcept
    {
        const double a = std::abs(u);
        if (a <= kKnee) return 1.0;
        const double w = a - kKnee + kRange;
        return kRange * kRange / (w * w);
    }

    static AdaaPoint point(double u) noexcept
    {
        const double a = std::abs(u);
        if (a <= kKnee) return {u, u * u * u * (1.0 / 6.0), 0.0};
        const double e = a - kKnee, w = e + kRange, lw = std::log(w);
        return {u, std::copysign(kKnee * kKnee * kKnee / 6.0 + 0.5 * kKnee * kKnee * e + 0.5 * e * e
                                 - kRange * kRange * (w * (lw - kLogRange) - e), u), lw};
    }

    /** ln(x / y) / (x - y) for x, y > 0, with the logarithms given. */
    static double logSlope(double x, double y, double lx, double ly) noexcept
    {
        const double q = (x - y) / (x + y);
        if (std::abs(q) > 0.05) return (lx - ly) / (x - y);
        // ln(x/y) = 2 atanh(q): odd series in q, no cancellation.
        const double q2 = q * q;
        return 2.0 / (x + y)
             * (1.0 + q2 * (1.0 / 3 + q2 * (1.0 / 5 + q2 * (1.0 / 7 + q2 * (1.0 / 9 + q2 * (1.0 / 11))))));
    }

    static bool exact(const AdaaPoint& p0, const AdaaPoint& p1, const AdaaPoint& p2,
                      double& y) noexcept
    {
        const double lo = std::min(p0.u, std::min(p1.u, p2.u)), hi = std::max(p0.u, std::max(p1.u, p2.u));
        if (lo >= -kKnee && hi <= kKnee) { y = (p0.u + p1.u + p2.u) * (1.0 / 3.0); return true; }
        const bool up = lo >= kKnee, down = hi <= -kKnee;
        if (!(up || down) || !(hi - lo > AdaaSecondDifference<AdaaRationalKnee>::kTolDd2))
            return false;
        // Average of 1 - r^2/w over the B-spline with knots a <= b <= c in w:
        // 2/(c - a) * (c ln(c/b)/(c - b) - a ln(b/a)/(b - a)).
        // Knots in w = |u| - k + r, sorted with their logarithms.
        double w0 = std::abs(p0.u) - kKnee + kRange, w1 = std::abs(p1.u) - kKnee + kRange;
        double w2 = std::abs(p2.u) - kKnee + kRange, l0 = p0.aux, l1 = p1.aux, l2 = p2.aux;
        if (w0 > w1) { std::swap(w0, w1); std::swap(l0, l1); }
        if (w1 > w2) { std::swap(w1, w2); std::swap(l1, l2); }
        if (w0 > w1) { std::swap(w0, w1); std::swap(l0, l1); }
        const double h = 2.0 / (w2 - w0) * (w2 * logSlope(w2, w1, l2, l1) - w0 * logSlope(w1, w0, l1, l0));
        y = std::copysign(1.0 - kRange * kRange * h, up ? 1.0 : -1.0);
        return true;
    }
};

// ---------------------------------------------------------------------------
// sin (not saturating): F1 = -cos, F2 = -sin; bounded, so always generic.
// ---------------------------------------------------------------------------
struct AdaaSine
{
    static double g(double u) noexcept { return -std::sin(u); }
    static double d1(double u) noexcept { return -std::cos(u); }
    static double d2(double u) noexcept { return std::sin(u); }
    static double d3(double u) noexcept { return std::cos(u); }
    static AdaaPoint point(double u) noexcept { return {u, -std::sin(u), 0.0}; }
    static bool exact(const AdaaPoint&, const AdaaPoint&, const AdaaPoint&, double&) noexcept
    {
        return false;
    }
};

/**
 * @brief Per-stream second-order ADAA state.
 *
 * process<Curve>(u) returns the antialiased curve output for the input
 * history (u, u[n-1], u[n-2]), centred on u[n-1]. The cached point values
 * belong to one curve: call invalidate() when the caller switches curves
 * (they are rebuilt from the raw input history on the next process()).
 * track(u) only records the input, so antialiasing can start later from
 * real history; reset(u) holds u as the whole history.
 */
class Adaa2
{
public:
    void reset(double u = 0.0) noexcept
    {
        u1_ = u2_ = u;
        valid_ = false;
    }

    void track(double u) noexcept
    {
        u2_ = u1_;
        u1_ = u;
        valid_ = false;
    }

    void invalidate() noexcept { valid_ = false; }

    template <class Curve>
    [[nodiscard]] double process(double u) noexcept
    {
        double y = 0.0;
        if constexpr (requires { Curve::kClosedForm; })
        {
            y = Curve::value(u, u1_, u2_);
        }
        else
        {
            if (!valid_)
            {
                p1_ = Curve::point(u1_);
                p2_ = Curve::point(u2_);
                valid_ = true;
            }
            const AdaaPoint p0 = Curve::point(u);
            if (!Curve::exact(p0, p1_, p2_, y))
                y = AdaaSecondDifference<Curve>::twice(p0.u, p1_.u, p2_.u, p0.g, p1_.g, p2_.g);
            p2_ = p1_;
            p1_ = p0;
        }
        u2_ = u1_;
        u1_ = u;
        return y;
    }

    /** Input one sample ago (the sample the output is centred on). */
    [[nodiscard]] double previousInput() const noexcept { return u1_; }

private:
    AdaaPoint p1_ {}, p2_ {};
    double u1_ = 0.0, u2_ = 0.0;
    bool valid_ = false;
};

} // namespace dspark::detail
