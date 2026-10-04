// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file ContinuousClip.h
 * @brief Shared continuous reconstruction and integration of clipping curves.
 *
 * Internal mathematical kernels used by offline and streaming processing.
 * The interpolation, quadrature, knee isolation and compensation are shared;
 * resampling policy and error reporting belong to their calling processor.
 * These functions do not implement or claim physical-device models.
 */
#include "ClipperShape.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace dspark::detail::continuous_clip
{
using Curve = ClipperCurve;
using Moments = std::array<double, 4>;
inline double choose(int n, int k)
{
    double v = 1;
    for (int i = 1; i <= k; ++i)
        v *= double(n - i + 1) / i;
    return v;
}
template <int N> struct Gauss
{
    std::array<double, N> nodes{}, weights{};
    Gauss()
    {
        for (int i = 0; i < (N + 1) / 2; ++i)
        {
            double x = std::cos(pi<double> * (i + .75) / (N + .5)), derivative = 0;
            for (int it = 0; it < 30; ++it)
            {
                double p0 = 1, p1 = x;
                for (int j = 2; j <= N; ++j)
                {
                    const double p = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
                    p0 = p1;
                    p1 = p;
                }
                derivative = N * (x * p1 - p0) / (x * x - 1);
                const double step = p1 / derivative;
                x -= step;
                if (std::abs(step) < 2e-16)
                    break;
            }
            const double w = 1 / ((1 - x * x) * derivative * derivative);
            nodes[i] = (1 - x) / 2;
            nodes[N - 1 - i] = (1 + x) / 2;
            weights[i] = weights[N - 1 - i] = w;
        }
    }
};

template <Curve C, int Degree, bool NativeSlope = false, bool SubtractLinear = true,
          int MomentCount = 4> class Interval
{
    static_assert(Degree >= 1 && Degree <= 11 && Degree % 2 == 1);
    static_assert(MomentCount >= 1 && MomentCount <= 8);
    static_assert(Degree + MomentCount <= 16); // Eight-point exact polynomial quadrature.
    using Moments = std::array<double, MomentCount>;
    static constexpr bool rational =
        C == Curve::GoldenRatio || C == Curve::SymmetricKnee || C == Curve::AsymmetricKnee;
    static constexpr int count = Degree + 1;
    static constexpr double referenceSlope = NativeSlope ? clipperSmallSignalSlope<C, double>() : 1.;
    using Poly = std::array<double, count>;
    std::array<Poly, count> matrix_{}, bernsteinMatrix_{};
    Gauss<8> linearQuadrature_;
    Gauss<16> nonlinearQuadrature_;
    double ceiling_;
    double reconstructionBound_ = 1;
    double arithmeticBound_ = 1;
    std::array<double, 11> levels_{};
    int levelCount_ = 0;
    static double value(const Poly &p, double t)
    {
        double v = p[Degree];
        for (int i = Degree - 1; i >= 0; --i)
            v = v * t + p[i];
        return v;
    }
    static double slope(const Poly &p, double t)
    {
        double v = Degree * p[Degree];
        for (int i = Degree - 1; i >= 1; --i)
            v = v * t + i * p[i];
        return v;
    }
    static void add(Moments &a, const Moments &b)
    {
        for (int i = 0; i < MomentCount; ++i)
            a[i] += b[i];
    }
    template <int N>
    Moments quadrature(const Poly &p, double a, double b, const Gauss<N> &q, bool fixed = false,
                       double fixedValue = 0) const
    {
        Moments out{};
        for (int i = 0; i < N; ++i)
        {
            const double t = a + (b - a) * q.nodes[i], x = value(p, t);
            const double reference = SubtractLinear ? referenceSlope * x : 0.;
            double v = (b - a) * q.weights[i] *
                       (fixed ? fixedValue - reference : clipperShape<C>(x, ceiling_) - reference);
            for (int j = 0; j < MomentCount; ++j)
            {
                out[j] += v;
                v *= t;
            }
        }
        return out;
    }
    std::array<Moments, 2> tanhQuadrature(const Poly &p, double a, double b) const
    {
        // Embedded G7/K15 rules reuse seven evaluations in the error estimate.
        // Symmetric abscissae and weights are defined on [-1,1]. This changes
        // neither the curve nor the adaptive tolerance below.
        static constexpr std::array<double, 8> nodes{
            .99145537112081263921, .94910791234275852453, .86486442335976907279,
            .74153118559939443986, .58608723546769113029, .40584515137739716691,
            .20778495500789846760, 0.};
        static constexpr std::array<double, 8> fineWeights{
            .02293532201052922496, .06309209262997855329, .10479001032225018384,
            .14065325971552591875, .16900472663926790283, .19035057806478540991,
            .20443294007529889241, .20948214108472782801};
        static constexpr std::array<double, 4> coarseWeights{
            .12948496616886969327, .27970539148927666790,
            .38183005050511894495, .41795918367346938776};
        std::array<Moments, 2> result{};
        const double half = (b - a) * .5, center = a + half;
        const auto accumulate = [&](double t, double fineWeight, double coarseWeight)
        {
            const double x = value(p, t);
            const double reference = SubtractLinear ? referenceSlope * x : 0.;
            double v = half * (clipperShape<C>(x, ceiling_) - reference);
            for (int j = 0; j < MomentCount; ++j)
            {
                result[1][j] += fineWeight * v;
                if (coarseWeight != 0.) result[0][j] += coarseWeight * v;
                v *= t;
            }
        };
        accumulate(center, fineWeights[7], coarseWeights[3]);
        for (int i = 0; i < 7; ++i)
        {
            const double coarseWeight = i % 2 ? coarseWeights[i / 2] : 0.;
            accumulate(center - half * nodes[i], fineWeights[i], coarseWeight);
            accumulate(center + half * nodes[i], fineWeights[i], coarseWeight);
        }
        return result;
    }
    Moments adaptive(const Poly &p, double a, double b, int depth = 0) const
    {
        const auto estimates = [&] {
            if constexpr (C == Curve::Tanh) return tanhQuadrature(p, a, b);
            else return std::array<Moments, 2>{quadrature(p, a, b, linearQuadrature_),
                                             quadrature(p, a, b, nonlinearQuadrature_)};
        }();
        const auto &coarse = estimates[0], &fine = estimates[1];
        const double scale =
            1 + std::abs(value(p, a)) + std::abs(value(p, b)) + std::abs(value(p, (a + b) / 2));
        const double tolerance = (b - a) * 1e-13 * scale;
        double error = 0;
        for (int i = 0; i < MomentCount; ++i)
            error = std::max(error, std::abs(fine[i] - coarse[i]));
        if (error <= tolerance)
            return fine;
        if (depth >= 24)
            return {std::numeric_limits<double>::quiet_NaN()};
        auto out = adaptive(p, a, (a + b) / 2, depth + 1);
        add(out, adaptive(p, (a + b) / 2, b, depth + 1));
        return out;
    }
    Moments linearMoments(const Poly &p, double a, double b) const
    {
        // Integrate a complete, provably linear interval algebraically. Retain
        // the established four-moment arithmetic for existing callers.
        if constexpr (MomentCount != 4)
        {
            if (a == 0 && b == 1)
            {
                static constexpr auto weights = [] {
                    std::array<Poly, MomentCount> result{};
                    for (int j = 0; j < MomentCount; ++j)
                        for (int k = 0; k < count; ++k)
                            result[j][k] = 1. / (j + k + 1);
                    return result;
                }();
                Moments out{};
                for (int j = 0; j < MomentCount; ++j)
                    for (int k = 0; k < count; ++k)
                        out[j] += p[k] * weights[j][k];
                return out;
            }
        }
        return quadrature(p, a, b, linearQuadrature_);
    }
    Moments segment(const Poly &p, double a, double b) const
    {
        const double x = value(p, (a + b) / 2);
        if constexpr (C == Curve::Hard)
        {
            if (std::abs(x) <= ceiling_)
            {
                if constexpr (SubtractLinear) return {};
                else return linearMoments(p, a, b);
            }
            return quadrature(p, a, b, linearQuadrature_, true, std::copysign(ceiling_, x));
        }
        else if constexpr (C == Curve::Sine)
        {
            if (std::abs(x) >= ceiling_ * halfPi<double>)
                return quadrature(p, a, b, linearQuadrature_, true, clipperShape<C>(x, ceiling_));
        }
        else if constexpr (rational)
        {
            const auto limits = clipperLinearLimits<C>(ceiling_);
            if (x >= limits[0] && x <= limits[1])
            {
                if constexpr (SubtractLinear) return {};
                else return linearMoments(p, a, b);
            }
        }
        else if constexpr (C == Curve::Tanh)
        {
            if (std::abs(x) >= 24 * ceiling_)
                return quadrature(p, a, b, linearQuadrature_, true, std::copysign(ceiling_, x));
        }
        if constexpr (C == Curve::Tanh || rational)
            return adaptive(p, a, b);
        else
            return quadrature(p, a, b, nonlinearQuadrature_);
    }
    double root(const Poly &p, double a, double b, double target) const
    {
        double fa = value(p, a) - target, fb = value(p, b) - target;
        if (fa == 0)
            return a;
        if (fb == 0)
            return b;
        if ((fa > 0) == (fb > 0))
            return std::numeric_limits<double>::quiet_NaN();
        double x = a + (b - a) * (-fa) / (fb - fa);
        for (int i = 0; i < 64; ++i)
        {
            const double f = value(p, x) - target;
            if (std::abs(f) <= 2e-15 || b - a < 2e-14)
                return x;
            if ((f > 0) == (fa > 0))
            {
                a = x;
                fa = f;
            }
            else
            {
                b = x;
                fb = f;
            }
            const double d = slope(p, x);
            const double candidate = d != 0 ? x - f / d : (a + b) / 2;
            x = candidate > a && candidate < b ? candidate : (a + b) / 2;
        }
        return std::numeric_limits<double>::quiet_NaN();
    }
    Moments recurse(const Poly &p, const Poly &bernstein, double a, double b, int depth) const
    {
        const auto bounds = std::minmax_element(bernstein.begin(), bernstein.end());
        const double low = *bounds.first, high = *bounds.second;
        if constexpr (C == Curve::Hard)
        {
            if (low >= -ceiling_ && high <= ceiling_)
            {
                if constexpr (SubtractLinear) return {};
                else return linearMoments(p, a, b);
            }
        }
        if constexpr (rational)
        {
            const auto limits = clipperLinearLimits<C>(ceiling_);
            if (low >= limits[0] && high <= limits[1])
            {
                if constexpr (SubtractLinear) return {};
                else return linearMoments(p, a, b);
            }
        }
        bool crosses = false;
        for (int i = 0; i < levelCount_; ++i)
            crosses = crosses || (low < levels_[i] && levels_[i] < high);
        if (!crosses)
            return segment(p, a, b);
        // Clipping curves are 1-Lipschitz; this branch's moment error is bounded
        // by width*range. Its width-scaled bound also sums across all leaves.
        if (high - low < 1e-13)
            return quadrature(p, a, b, nonlinearQuadrature_);
        bool increasing = true, decreasing = true;
        for (int i = 1; i < count; ++i)
        {
            increasing = increasing && bernstein[i] >= bernstein[i - 1];
            decreasing = decreasing && bernstein[i] <= bernstein[i - 1];
        }
        if (increasing || decreasing)
        {
            std::array<double, 13> edges{};
            int n = 0;
            edges[n++] = a;
            const double va = value(p, a), vb = value(p, b);
            for (int i = 0; i < levelCount_; ++i)
                if (levels_[i] > std::min(va, vb) && levels_[i] < std::max(va, vb))
                {
                    const double crossing = root(p, a, b, levels_[i]);
                    if (!std::isfinite(crossing))
                        return {std::numeric_limits<double>::quiet_NaN()};
                    edges[n++] = crossing;
                }
            edges[n++] = b;
            std::sort(edges.begin(), edges.begin() + n);
            Moments out{};
            for (int i = 0; i + 1 < n; ++i)
                add(out, segment(p, edges[i], edges[i + 1]));
            return out;
        }
        if (depth >= 32)
            return {std::numeric_limits<double>::quiet_NaN()};
        Poly work = bernstein, left{}, right{};
        left[0] = work[0];
        right[Degree] = work[Degree];
        for (int level = 1; level < count; ++level)
        {
            for (int i = 0; i < count - level; ++i)
                work[i] = (work[i] + work[i + 1]) * .5;
            left[level] = work[0];
            right[Degree - level] = work[Degree - level];
        }
        auto out = recurse(p, left, a, (a + b) / 2, depth + 1);
        add(out, recurse(p, right, (a + b) / 2, b, depth + 1));
        return out;
    }

  public:
    std::vector<double> linearKernel() const
    {
        static_assert(MomentCount == 4); // Cubic B-spline synthesis below.
        constexpr std::array<Moments, 4> kernels{
            {{1, -3, 3, -1}, {4, 0, -6, 3}, {1, 3, 3, -3}, {0, 0, 0, 1}}};
        std::vector<double> result(Degree + 4);
        for (int i = 0; i < count; ++i)
        {
            Moments moments{};
            for (int j = 0; j < 4; ++j)
                for (int k = 0; k < count; ++k)
                    moments[j] += matrix_[i][k] / (k + j + 1);
            for (int lag = 0; lag < 4; ++lag)
                for (int j = 0; j < 4; ++j)
                    result[Degree - i + lag] += moments[j] * kernels[lag][j] / 6;
        }
        return result;
    }
    explicit Interval(double ceiling) : ceiling_(ceiling)
    {
        constexpr int begin = -(Degree - 1) / 2;
        for (int node = 0; node < count; ++node)
        {
            std::array<long double, count> p{};
            p[0] = 1;
            int order = 0;
            const int x = begin + node;
            for (int other = 0; other < count; ++other)
                if (other != node)
                {
                    const int y = begin + other;
                    std::array<long double, count> next{};
                    for (int j = 0; j <= order; ++j)
                    {
                        next[j] -= p[j] * y / (x - y);
                        next[j + 1] += p[j] / (x - y);
                    }
                    p = next;
                    ++order;
                }
            for (int j = 0; j < count; ++j)
                matrix_[node][j] = static_cast<double>(p[j]);
        }
        for (int i = 0; i < count; ++i)
            for (int j = 0; j <= i; ++j)
                bernsteinMatrix_[i][j] = choose(i, j) / choose(Degree, j);
        // Convex-hull bound on the complete reconstructed interval. It permits
        // exact linear-region bypass before the per-sample matrix products.
        for (int row = 0; row < count; ++row)
        {
            long double norm = 0, arithmeticNorm = 0;
            for (int node = 0; node < count; ++node)
            {
                long double coefficient = 0;
                for (int column = 0; column <= row; ++column)
                {
                    coefficient += static_cast<long double>(bernsteinMatrix_[row][column]) *
                                   matrix_[node][column];
                    arithmeticNorm +=
                        std::abs(static_cast<long double>(bernsteinMatrix_[row][column]) *
                                 matrix_[node][column]);
                }
                norm += std::abs(coefficient);
            }
            reconstructionBound_ = std::max(reconstructionBound_, static_cast<double>(norm));
            arithmeticBound_ = std::max(arithmeticBound_, static_cast<double>(arithmeticNorm));
        }
        setCeiling(ceiling);
    }
    void setCeiling(double ceiling)
    {
        ceiling_ = ceiling;
        if constexpr (C == Curve::Tanh)
        {
            levels_ = {-24 * ceiling, -12 * ceiling, -6 * ceiling, -3 * ceiling, -ceiling,    0,
                       ceiling,       3 * ceiling,   6 * ceiling,  12 * ceiling, 24 * ceiling};
            levelCount_ = 11;
        }
        else if constexpr (rational)
        {
            const auto limits = clipperLinearLimits<C>(ceiling);
            levels_[0] = limits[0];
            levels_[1] = limits[1];
            levelCount_ = 2;
        }
        else
        {
            const double limit = C == Curve::Sine ? ceiling * halfPi<double> : ceiling;
            levels_[0] = -limit;
            levels_[1] = limit;
            levelCount_ = 2;
        }
    }
    // Convex-hull norm for the complete interpolation interval. Offline
    // callers can bound finite-source tails using the same reconstruction.
    [[nodiscard]] double reconstructionBound() const noexcept
    {
        return reconstructionBound_;
    }
    /** @brief Integrates an already reconstructed power polynomial on [0,1].
     * Returns integral(t^j * shape(p(t)), dt), j = 0 .. MomentCount-1,
     * subtracting the linear reference when SubtractLinear is enabled.
     * Shares knee isolation and quadrature with the sample interpolator.
     * Nonfinite coefficients or failed convergence produce NaN.
     */
    [[nodiscard]] Moments polynomial(const Poly &power) const noexcept
    {
        Poly bernstein{};
        for (double coefficient : power)
            if (!std::isfinite(coefficient))
                return {std::numeric_limits<double>::quiet_NaN()};
        if (std::all_of(power.begin(), power.end(), [](double value) { return value == 0; }))
            return {};
        for (int i = 0; i < count; ++i)
            for (int j = 0; j <= i; ++j)
                bernstein[i] += power[j] * bernsteinMatrix_[i][j];
        return recurse(power, bernstein, 0, 1, 0);
    }
    Moments operator()(const std::array<double, count> &samples) const noexcept
    {
        if (std::all_of(samples.begin(), samples.end(), [](double value) { return value == 0; }))
            return {};
        if constexpr (SubtractLinear && (C == Curve::Hard || rational))
        {
            double low = samples[0], high = samples[0];
            for (double sample : samples)
            {
                if (!std::isfinite(sample))
                    return {std::numeric_limits<double>::quiet_NaN()};
                low = std::min(low, sample);
                high = std::max(high, sample);
            }
            const double center = 0.5 * low + 0.5 * high;
            const double radius = (0.5 * high - 0.5 * low) * reconstructionBound_;
            // Covers both dot-product reductions, the rounded basis and the
            // midpoint/radius arithmetic; count is at most twelve.
            const double margin = 128 * std::numeric_limits<double>::epsilon() * arithmeticBound_ *
                                      (std::abs(center) + radius) +
                                  256 * std::numeric_limits<double>::denorm_min();
            const auto limits = [&] {
                if constexpr (rational)
                    return clipperLinearLimits<C>(ceiling_);
                else
                    return std::array<double, 2>{-ceiling_, ceiling_};
            }();
            if (center - radius - margin >= limits[0] && center + radius + margin <= limits[1])
                return {};
        }
        Poly power{}, bernstein{};
        for (int i = 0; i < count; ++i)
            for (int j = 0; j < count; ++j)
                power[j] += samples[i] * matrix_[i][j];
        for (int i = 0; i < count; ++i)
            for (int j = 0; j <= i; ++j)
                bernstein[i] += power[j] * bernsteinMatrix_[i][j];
        return recurse(power, bernstein, 0, 1, 0);
    }
};

inline std::vector<double> inverseBoxPower(int order)
{
    // (asin(sqrt(s))/sqrt(s))^4, where s=(2-z-z^-1)/4. Restores the
    // integrated cubic B-spline passband without an IIR pole near Nyquist.
    std::vector<double> series(static_cast<std::size_t>(order + 1));
    for (int k = 0; k <= order; ++k)
        series[k] = choose(2 * k, k) / std::pow(4., k) / (2 * k + 1);
    std::vector<double> combined(static_cast<std::size_t>(order + 1));
    combined[0] = 1;
    for (int power = 0; power < 4; ++power)
    {
        std::vector<double> next(combined.size());
        for (int i = 0; i <= order; ++i)
            for (int j = 0; i + j <= order; ++j)
                next[i + j] += combined[i] * series[j];
        combined = std::move(next);
    }
    std::vector<double> taps(static_cast<std::size_t>(2 * order + 1)), basis(taps.size());
    basis[order] = 1;
    for (int k = 0; k <= order; ++k)
    {
        for (std::size_t i = 0; i < taps.size(); ++i)
            taps[i] += combined[k] * basis[i];
        if (k == order)
            break;
        std::vector<double> next(taps.size());
        for (int j = order - k; j <= order + k; ++j)
        {
            next[j - 1] -= .25 * basis[j];
            next[j] += .5 * basis[j];
            next[j + 1] -= .25 * basis[j];
        }
        basis = std::move(next);
    }
    return taps;
}

/** @brief Streaming cubic B-spline moments of a reconstructed clipping residual.
 *
 * NativeSlope selects subtraction of the implemented curve's small-signal
 * slope. The default retains the original unit-slope residual contract.
 * SubtractLinear=false integrates the complete shaped signal. It permits a
 * caller to avoid cancellation of large linear terms at very small ceilings.
 * Construction computes the quadrature tables. process() and reset() allocate
 * nothing, use bounded recursion and never throw. Non-convergence produces NaN,
 * which callers must handle as a numerical error; it never changes the curve or
 * substitutes a lower-quality result. linearKernel() allocates during setup only.
 * Threading: each stream owns its state; no concurrent calls on one instance.
 */
template <Curve C, int Degree = 11, bool NativeSlope = false, bool SubtractLinear = true> class Residual final
{
    static constexpr int degree = Degree;
    Interval<C, degree, NativeSlope, SubtractLinear> integral_{1};
    std::array<double, degree + 1> samples_{};
    std::array<Moments, 4> history_{};
    int sampleWrite_ = 0, momentWrite_ = 0;

  public:
    /** @brief Resets reconstruction history and sets the positive finite ceiling. */
    void reset(double ceiling) noexcept
    {
        integral_.setCeiling(ceiling);
        samples_.fill(0);
        history_.fill({});
        sampleWrite_ = momentWrite_ = 0;
    }
    /** @brief Constructs the matching linear FIR during setup. */
    [[nodiscard]] std::vector<double> linearKernel() const
    {
        return integral_.linearKernel();
    }
    /** @brief Processes one high-rate sample, returning its integrated residual. */
    double process(double x) noexcept
    {
        samples_[sampleWrite_] = x;
        sampleWrite_ = (sampleWrite_ + 1) % (degree + 1);
        std::array<double, degree + 1> ordered{};
        for (int i = 0; i <= degree; ++i)
            ordered[i] = samples_[(sampleWrite_ + i) % (degree + 1)];
        history_[momentWrite_] = integral_(ordered);
        const auto &a = history_[(momentWrite_ + 1) % 4];
        const auto &b = history_[(momentWrite_ + 2) % 4];
        const auto &c = history_[(momentWrite_ + 3) % 4];
        const auto &d = history_[momentWrite_];
        const double out = (a[3] + b[0] + 3 * b[1] + 3 * b[2] - 3 * b[3] + 4 * c[0] - 6 * c[2] +
                            3 * c[3] + d[0] - 3 * d[1] + 3 * d[2] - d[3]) /
                           6;
        momentWrite_ = (momentWrite_ + 1) % 4;
        return out;
    }
};
} // namespace dspark::detail::continuous_clip
