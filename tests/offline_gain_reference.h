// DSPark - Independent offline gain signal references for tests.
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once
#include "../Core/FFT.h"
#include "../Core/OfflineProcessing.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark::test::offline_reference
{
// Direct interpolation-basis integration, independent of the production
// Hilbert factorization, convolution and far-field approximation.
inline long double basisIntegral(int i, int j, int n)
{
    if (i == j && i == n)
        return .75L;
    if ((i & 1) == (j & 1) && (j & 1) == (n & 1))
        return 0;
    const int odd = (i & 1) == (j & 1) ? n : (i & 1) == (n & 1) ? j : i;
    const int a = odd == n ? i : n;
    const int b = odd == i ? j : odd == j ? i : j;
    return 1 / (std::numbers::pi_v<long double> * std::numbers::pi_v<long double> *
                (odd - a) * (odd - b));
}

// Whole-array linear convolution is deliberately expensive test-only storage.
// It neither uses Convolver nor truncates an infinite impulse at a block edge.
inline std::vector<double> linearHilbert(const std::vector<double> &x, bool allLags = false)
{
    const auto n = x.size();
    std::size_t size = 2;
    while (size < 3 * n - 2)
        size *= 2;
    FFTComplex<double> fft(size);
    std::vector<double> a(2 * size), h(2 * size);
    for (std::size_t i = 0; i < n; ++i)
        a[2 * i] = x[i];
    for (std::int64_t k = 1 - static_cast<std::int64_t>(n);
         k < static_cast<std::int64_t>(n); ++k)
        if (k != 0 && (allLags || k % 2))
            h[2 * static_cast<std::size_t>(k + n - 1)] = 1 / (std::numbers::pi * k);
    fft.forward(a.data());
    fft.forward(h.data());
    for (std::size_t i = 0; i < size; ++i)
    {
        const double real = a[2 * i] * h[2 * i] - a[2 * i + 1] * h[2 * i + 1];
        a[2 * i + 1] = a[2 * i] * h[2 * i + 1] + a[2 * i + 1] * h[2 * i];
        a[2 * i] = real;
    }
    fft.inverse(a.data());
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = a[2 * (i + n - 1)];
    return out;
}
inline std::vector<double> finiteReferenceDelta(const std::vector<double> &x,
                                   const std::vector<double> &gain)
{
    const double boundary = (gain.front() + gain.back()) * .5;
    std::vector<double> g(gain.size()), combined(gain.size());
    for (std::size_t i = 0; i < g.size(); ++i)
        g[i] = gain[i] - boundary;
    const auto ax = linearHilbert(x), ag = linearHilbert(g);
    for (std::size_t i = 0; i < g.size(); ++i)
        combined[i] = x[i] * ag[i] + g[i] * ax[i];
    auto result = linearHilbert(combined);
    for (std::size_t i = 0; i < g.size(); ++i)
        result[i] = (boundary - 1) * x[i] + .75 * x[i] * g[i] + ax[i] * ag[i] - result[i];
    return result;
}

// Independent finite half-integer sums start at the exact psi(1/2) and
// psi'(1/2) values. No asymptotic expansion or production weight cache is used.
inline std::vector<std::array<long double, 2>> endpointWeights(int n, double left, double right)
{
    std::vector<std::array<long double, 2>> half(static_cast<std::size_t>(n / 2 + 1)), out(n);
    half[0] = {-std::numbers::egamma_v<long double> - 2 * std::log(2.L),
               std::numbers::pi_v<long double> * std::numbers::pi_v<long double> / 2};
    for (std::size_t i = 1; i < half.size(); ++i)
    {
        const long double reciprocal = 1 / (static_cast<long double>(i) - .5L);
        half[i] = {half[i - 1][0] + reciprocal, half[i - 1][1] - reciprocal * reciprocal};
    }
    for (int i = 0; i < n; ++i)
    {
        const auto l = half[static_cast<std::size_t>((i + 1) / 2)];
        const auto r = half[static_cast<std::size_t>((n - i) / 2)];
        out[i] = {(.5L * (left * l[0] - right * r[0])),
                  (left * l[1] + right * r[1]) /
                      (4 * std::numbers::pi_v<long double> * std::numbers::pi_v<long double>)};
    }
    return out;
}

inline long double endpointIntegral(const std::vector<std::array<long double, 2>> &weights,
                                    int i, int n)
{
    if (i == n)
        return weights[n][1];
    return ((n - i) % 2 ? -1.L : 1.L) * (weights[n][0] - weights[i][0]) /
           (std::numbers::pi_v<long double> * std::numbers::pi_v<long double> * (n - i));
}

inline std::vector<double> referenceDelta(const std::vector<double> &x,
                                           const std::vector<double> &gain,
                                           double left, double right)
{
    auto result = finiteReferenceDelta(x, gain);
    const double baseline = (gain.front() + gain.back()) * .5;
    const auto weights = endpointWeights(static_cast<int>(x.size()), left - baseline,
                                         right - baseline);
    std::vector<double> a(x.size()), fa(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        a[i] = (i % 2 ? -1 : 1) * x[i];
        fa[i] = static_cast<double>(weights[i][0]) * a[i];
    }
    const auto ca = linearHilbert(a, true), cfa = linearHilbert(fa, true);
    for (std::size_t i = 0; i < x.size(); ++i)
        result[i] += static_cast<double>(weights[i][1]) * x[i] +
            (i % 2 ? -1. : 1.) / std::numbers::pi *
                (static_cast<double>(weights[i][0]) * ca[i] - cfa[i]);
    return result;
}
inline std::vector<double> referenceDelta(const std::vector<double> &x,
                                           const std::vector<double> &gain)
{
    return referenceDelta(x, gain, gain.front(), gain.back());
}

// Normalize the oracle independently for extreme/subnormal PCM. Each channel is
// reconstructed separately; the linked gain and exact protection are shared.
template <class T, class Plan>
AudioBuffer<T> render(const AudioBuffer<T> &input, const Plan &plan,
                      const OfflineGainRenderInfo &info, double feather = 1)
{
    const int n = input.getNumSamples();
    std::vector<double> gain(n);
    double normalizer = 0;
    for (int i = 0; i < n; ++i)
    {
        gain[i] = plan.gainAt(i);
        for (int c = 0; c < input.getNumChannels(); ++c)
            normalizer = std::max(normalizer, std::abs(static_cast<double>(input.getChannel(c)[i])));
    }
    if (normalizer == 0)
        normalizer = 1;
    AudioBuffer<T> result;
    result.resize(input.getNumChannels(), n);
    for (int c = 0; c < input.getNumChannels(); ++c)
    {
        std::vector<double> x(n);
        for (int i = 0; i < n; ++i)
            x[i] = input.getChannel(c)[i] / normalizer;
        const auto delta = referenceDelta(x, gain, info.leftBoundaryGain, info.rightBoundaryGain);
        for (int i = 0; i < n; ++i)
        {
            double mask = 1;
            for (auto region : plan.exclusions())
            {
                if (i >= region.begin && i < region.end)
                {
                    mask = 0;
                    break;
                }
                const double distance = i < region.begin ? double(region.begin - i)
                                                         : double(i - region.end);
                const double t = std::clamp(distance / feather, 0., 1.);
                const double u = std::min(t, 1 - t);
                const double v = u * u * u * u * (35 - u * (84 - u * (70 - 20 * u)));
                mask *= t <= .5 ? v : 1 - v;
            }
            const auto original = input.getChannel(c)[i];
            if (!info.bandlimited)
                result.getChannel(c)[i] = gain[i] == 1 ? original : T(original * gain[i]);
            else if (mask == 0 || info.deltaScale == 0)
                result.getChannel(c)[i] = original;
            else
            {
                const double scalar = x[i] * (gain[i] - 1);
                const double constrained = scalar + mask * (delta[i] - scalar);
                result.getChannel(c)[i] = T((x[i] + info.deltaScale * constrained) * normalizer);
            }
        }
    }
    return result;
}
} // namespace dspark::test::offline_reference
