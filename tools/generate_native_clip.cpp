// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

// Rebuild the fixed source-rate clipping banks of the offline clippers:
//   c++ -std=c++20 -O2 tools/generate_native_clip.cpp -o generate_native_clip
//   ./generate_native_clip Core/detail/NativeClipCoefficients.h
// Self-contained: the dense design grid and the normal equations below are
// coefficient design only, never audio processing.
//
// Reconstruction: on each unit interval [0,1] between samples 0 and 1, the
// degree-eleven polynomial that interpolates, at twelve Chebyshev nodes, the
// Kaiser (beta 8) windowed-sinc reconstruction from the 32 samples -15..16.
//
// Projection: five moment FIRs of 64 taps. The moments of interval m are
// integral(t^q r(m + t) dt, 0, 1), q = 0..4, of a continuous residual r. The
// banks together form a piecewise-quartic continuous kernel h, and the
// output is integral(r(tau) h(n - tau) dtau). The taps minimize the weighted
// squared error of h's continuous spectrum against an ideal delay on
// |f| <= 0.40 fs (weight 1) and zero on |f| >= 0.46 fs (weight (1 + |f|)^-2,
// to 12 fs): unlike a single compactly supported average with inverse
// compensation, the free moment weights place the spectral images of the
// sampled output, where distortion folded across Nyquist lands, in the
// stopband.
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <vector>

namespace
{
constexpr double pi = std::numbers::pi_v<double>;
constexpr int half = 16, reconstructionTaps = 2 * half, powers = 12;
constexpr int momentTaps = 64, moments = 5;
constexpr double beta = 8, passEdge = .40, stopEdge = .46, decay = 2, ridge = 1e-13;

double besselI0(double x)
{
    double sum = 1, term = 1;
    for (int k = 1; k < 200; ++k)
    {
        term *= (x / (2 * k)) * (x / (2 * k));
        sum += term;
        if (term < sum * 1e-18)
            break;
    }
    return sum;
}

double windowedSinc(double u)
{
    if (std::abs(u) >= half)
        return 0;
    const double r = u / half;
    const double window = besselI0(beta * std::sqrt(1 - r * r)) / besselI0(beta);
    return (u == 0 ? 1 : std::sin(pi * u) / (pi * u)) * window;
}

// Power-basis coefficients of the polynomials interpolating each Chebyshev
// node on [0,1] (the inverse Vandermonde matrix), in long double.
std::array<std::array<double, powers>, powers> chebyshevBasis(std::array<double, powers> &nodes)
{
    for (int i = 0; i < powers; ++i)
        nodes[i] = .5 - .5 * std::cos(pi * (i + .5) / powers);
    std::array<std::array<double, powers>, powers> basis{};
    for (int i = 0; i < powers; ++i)
    {
        std::array<long double, powers + 1> p{};
        p[0] = 1;
        int order = 0;
        for (int other = 0; other < powers; ++other)
        {
            if (other == i)
                continue;
            const long double scale = 1.L / (static_cast<long double>(nodes[i]) - nodes[other]);
            std::array<long double, powers + 1> next{};
            for (int j = 0; j <= order; ++j)
            {
                next[j] -= p[j] * nodes[other] * scale;
                next[j + 1] += p[j] * scale;
            }
            p = next;
            ++order;
        }
        for (int j = 0; j < powers; ++j)
            basis[i][j] = static_cast<double>(p[j]);
    }
    return basis;
}

// psi_q(f) = integral(t^q exp(i 2 pi f t), 0, 1).
void psi(double f, std::array<std::complex<double>, moments> &out)
{
    const double w = 2 * pi * f;
    if (std::abs(w) < .5)
    {
        for (int q = 0; q < moments; ++q)
        {
            std::complex<double> sum = 0, term = 1;
            for (int k = 0; k < 40; ++k)
            {
                sum += term / double(q + k + 1);
                term *= std::complex<double>(0, w) / double(k + 1);
            }
            out[q] = sum;
        }
        return;
    }
    const std::complex<double> e = std::polar(1.0, w), iw(0, w);
    out[0] = (e - 1.0) / iw;
    for (int q = 1; q < moments; ++q)
        out[q] = (e - double(q) * out[q - 1]) / iw;
}

// Shifted Legendre polynomials P_k(2t - 1) in t^q: a better conditioned
// design basis, converted back to moments of t^q at the end.
constexpr double legendre[moments][moments] = {{1, 0, 0, 0, 0},
                                               {-1, 2, 0, 0, 0},
                                               {1, -6, 6, 0, 0},
                                               {-1, 12, -30, 20, 0},
                                               {1, -20, 90, -140, 70}};

std::vector<double> designProjection()
{
    // Tap t weights the moments of the interval at lags [t, t + 1]; the
    // kernel support is [0, 64] and its centre, 32, the design delay.
    constexpr int n = moments * momentTaps;
    constexpr double delay = momentTaps / 2.;
    std::vector<double> gram(static_cast<std::size_t>(n) * n), rhs(n);
    std::vector<std::complex<double>> row(n);
    const double step = 1. / (16. * momentTaps);
    for (double f = step / 2; f < 12; f += step)
    {
        double weight;
        std::complex<double> target = 0;
        if (f <= passEdge)
        {
            weight = 1;
            target = std::polar(1.0, -2 * pi * f * delay);
        }
        else if (f >= stopEdge)
            weight = std::pow(1 + f, -decay);
        else
            continue;
        std::array<std::complex<double>, moments> p;
        psi(f, p);
        for (int t = 0; t < momentTaps; ++t)
        {
            // Lag in [t, t+1] at interval offset u: h = sum_q c_q (t + 1 - lag)^q,
            // so the tap's response is exp(-i 2 pi f (t + 1)) sum_q c_q psi_q(f).
            const auto shift = std::polar(1.0, -2 * pi * f * (t + 1));
            for (int k = 0; k < moments; ++k)
            {
                std::complex<double> value = 0;
                for (int q = 0; q < moments; ++q)
                    value += legendre[k][q] * p[q];
                row[t * moments + k] = shift * value;
            }
        }
        for (int r = 0; r < n; ++r)
        {
            const auto conjugate = std::conj(row[r]) * weight;
            rhs[r] += (conjugate * target).real();
            for (int c = r; c < n; ++c)
                gram[static_cast<std::size_t>(r) * n + c] += (conjugate * row[c]).real();
        }
    }
    double trace = 0;
    for (int r = 0; r < n; ++r)
    {
        for (int c = 0; c < r; ++c)
            gram[static_cast<std::size_t>(r) * n + c] = gram[static_cast<std::size_t>(c) * n + r];
        trace += gram[static_cast<std::size_t>(r) * n + r];
    }
    for (int r = 0; r < n; ++r)
        gram[static_cast<std::size_t>(r) * n + r] += ridge * trace / n;
    // Cholesky solve.
    for (int c = 0; c < n; ++c)
    {
        double s = gram[static_cast<std::size_t>(c) * n + c];
        for (int k = 0; k < c; ++k)
            s -= gram[static_cast<std::size_t>(c) * n + k] * gram[static_cast<std::size_t>(c) * n + k];
        const double d = std::sqrt(s);
        gram[static_cast<std::size_t>(c) * n + c] = d;
        for (int r = c + 1; r < n; ++r)
        {
            double v = gram[static_cast<std::size_t>(r) * n + c];
            for (int k = 0; k < c; ++k)
                v -= gram[static_cast<std::size_t>(r) * n + k] * gram[static_cast<std::size_t>(c) * n + k];
            gram[static_cast<std::size_t>(r) * n + c] = v / d;
        }
    }
    std::vector<double> y(n);
    for (int r = 0; r < n; ++r)
    {
        double v = rhs[r];
        for (int k = 0; k < r; ++k)
            v -= gram[static_cast<std::size_t>(r) * n + k] * y[k];
        y[r] = v / gram[static_cast<std::size_t>(r) * n + r];
    }
    for (int r = n - 1; r >= 0; --r)
    {
        double v = y[r];
        for (int k = r + 1; k < n; ++k)
            v -= gram[static_cast<std::size_t>(k) * n + r] * y[k];
        y[r] = v / gram[static_cast<std::size_t>(r) * n + r];
    }
    // Lag-ordered taps, moments of t^q.
    std::vector<double> taps(static_cast<std::size_t>(n));
    for (int t = 0; t < momentTaps; ++t)
        for (int k = 0; k < moments; ++k)
            for (int q = 0; q < moments; ++q)
                taps[t * moments + q] += legendre[k][q] * y[t * moments + k];
    return taps;
}

void writeArray(std::ostream &out, const std::vector<double> &values)
{
    out << "{\n    ";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        out << values[i];
        if (i + 1 < values.size())
            out << ',' << ((i + 1) % 4 ? " " : "\n    ");
    }
    out << "\n}";
}
} // namespace

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: generate_native_clip OUTPUT_HEADER\n";
        return 1;
    }
    std::array<double, powers> nodes{};
    const auto basis = chebyshevBasis(nodes);
    // Correlation order: tap j reads the window's j-th oldest sample, -15 + j.
    std::vector<double> reconstruction(static_cast<std::size_t>(reconstructionTaps) * powers);
    for (int j = 0; j < reconstructionTaps; ++j)
        for (int i = 0; i < powers; ++i)
        {
            const double value = windowedSinc(nodes[i] - (j - half + 1));
            for (int k = 0; k < powers; ++k)
                reconstruction[static_cast<std::size_t>(j) * powers + k] += value * basis[i][k];
        }
    const auto lagOrdered = designProjection();
    // One row per moment, correlation order: tap j reads the frame of lag
    // momentTaps - 1 - j.
    std::vector<double> projection(lagOrdered.size());
    for (int q = 0; q < moments; ++q)
        for (int j = 0; j < momentTaps; ++j)
            projection[static_cast<std::size_t>(q) * momentTaps + j] =
                lagOrdered[static_cast<std::size_t>(momentTaps - 1 - j) * moments + q];
    std::ofstream out(argv[1], std::ios::binary);
    if (!out)
        return 2;
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "// DSPark - Professional Audio DSP Framework\n"
           "// Copyright (c) 2026 Cristian Moresi - MIT License\n\n"
           "#pragma once\n\n"
           "/** @file NativeClipCoefficients.h\n"
           " * @brief Fixed source-rate reconstruction and moment projection banks.\n"
           " * Generated by tools/generate_native_clip.cpp; do not hand edit.\n"
           " * Reconstruction: degree-eleven Chebyshev fit of the Kaiser (beta 8)\n"
           " * windowed sinc over 32 samples, per unit interval. Projection: five\n"
           " * moment FIRs of 64 taps, a least-squares continuous kernel with\n"
           " * passband 0.40 fs and stopband from 0.46 fs. The reconstruction is\n"
           " * tap-interleaved, the projection one row per moment; both are in\n"
           " * correlation order (tap 0 meets the oldest input).\n"
           " * Shared read-only payload: 5632 bytes, independent of instances.\n"
           " */\n#include <array>\n\nnamespace dspark::detail::native_clip_coefficients\n{\n"
           "inline constexpr int reconstructionTaps = 32, powers = 12, projectionTaps = 64, moments = 5;\n"
           "/** @brief Delay of the projected output behind the newest input, in samples. */\n"
           "inline constexpr int delay = reconstructionTaps / 2 - 1 + projectionTaps / 2;\n"
           "inline constexpr std::array<double, reconstructionTaps * powers> reconstruction";
    writeArray(out, reconstruction);
    out << ";\ninline constexpr std::array<double, projectionTaps * moments> projection";
    writeArray(out, projection);
    out << ";\n} // namespace dspark::detail::native_clip_coefficients\n";
    return out ? 0 : 3;
}
