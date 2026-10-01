// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License
#pragma once

#include <cmath>

namespace dspark::detail {

/** Stable log(cosh(x)), shared by the tanh antiderivative kernels. */
template <typename T>
inline T logCosh(T x) noexcept
{
    const T a = std::abs(x);
    // The usual large-argument identity subtracts nearly equal numbers
    // around zero. This series also preserves small-signal ADAA accuracy.
    if (a < T(0.01))
    {
        const T z = x * x;
        return z * (T(0.5) + z * (T(-1.0 / 12.0)
            + z * (T(1.0 / 45.0) + z * T(-17.0 / 2520.0))));
    }
    return a + std::log1p(std::exp(T(-2) * a)) - T(0.6931471805599453);
}

/** Integral of log(cosh(x)), with value zero at the origin.
 *  A Chebyshev polynomial handles |x| < 1 without cancellation. The outer
 *  branch integrates the exponentially convergent log1p expansion; its
 *  omitted dilogarithm tail is below 5e-18 at |x| >= 1.
 */
inline double integralLogCosh(double x) noexcept
{
    const double a = std::abs(x);
    if (a < 1.0)
    {
        const double z = a * a;
        constexpr double coefficients[] = {
            -7.98389023185534e-11, 8.6684720203723614e-10,
            -4.7997095383377807e-9, 1.8850617466346369e-8,
            -6.1974137503985441e-8, 1.8914247025489175e-7,
            -5.6812797392185565e-7, 1.7247571922095787e-6,
            -5.3522004257752879e-6, 1.7105344959469917e-5,
            -5.6815608632124677e-5, 0.00019881353180096906,
            -0.00074955908287299841, 0.0031746031746026004,
            -0.016666666666666659, 0.16666666666666666
        };
        double p = 0.0;
        for (double c : coefficients) p = p * z + c;
        return x * z * p;
    }
    const double z = std::exp(-2.0 * a);
    double li = 1.0 / 256.0;
    for (int k = 15; k >= 1; --k)
        li = li * z + (k % 2 == 0 ? 1.0 : -1.0) / (k * k);
    li *= z;
    return std::copysign(0.5 * a * a - a * 0.6931471805599453
        + 0.5 * (li + 0.8224670334241132), x);
}

} // namespace dspark::detail
