// DSPark - Independent finite-source product and worker-resource contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Core/detail/OfflineProduct.h"
#include "dspark_test.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

using namespace dspark;

DSPARK_TEST(OfflineProduct_ConvolutionWindowMatchesDirectToeplitz)
{
    for (int block : {128, 512, 4096})
        for (double shift : {-.875, 0., .125, .5})
        {
            // The unshifted kernel is the principal-value Cauchy kernel.
            const auto impulse = [shift](int lag) {
                if (shift == 0)
                    return lag ? 1 / (std::numbers::pi * lag) : 0.;
                return (lag % 2 ? -1 : 1) * std::sin(std::numbers::pi * shift) /
                       (std::numbers::pi * (lag + shift));
            };
            const OfflineJobOptions options;
            detail::OfflineSession job(options);
            detail::OfflineConvolutionWindow window(job, block, impulse);
            std::vector<double> input(3 * block);
            for (int signal = 0; signal < 8; ++signal)
            {
                std::fill(input.begin(), input.end(), 0.);
                if (signal < 6)
                    input[std::array<int, 6>{0, block - 1, block, 2 * block - 1,
                                            2 * block, 3 * block - 1}[signal]] = 1;
                else
                    for (int i = 0; i < 3 * block; ++i)
                        input[i] = signal == 6 ? .4 * std::cos(.219 * i) + .3 * std::sin(2.731 * i)
                                               : (i % 2 ? -.9 : .9);
                std::copy(input.begin(), input.end(), window.input());
                const double *output = window.process();
                for (int i = 0; i < block; ++i)
                {
                    if (block > 128 && i != 0 && i != 1 && i != block / 2 && i != block - 1)
                        continue;
                    long double expected = 0;
                    for (int j = 0; j < 3 * block; ++j)
                        expected += static_cast<long double>(input[j]) * impulse(block + i - j);
                    EXPECT_NEAR(output[i], static_cast<double>(expected), 3e-12);
                }
            }
        }
}

DSPARK_TEST(OfflineProduct_WideGridKeepsExternalReadsBounded)
{
    for (int block : {8192, 16384, 32768, 65536})
    {
        const int frames = 3 * block + 17;
        std::vector<double> x(frames), out(block);
        for (int i = 0; i < frames; ++i)
            x[i] = .3 * std::sin(.217 * i) + .2 * std::cos(3.131 * i);
        const OfflineJobOptions options;
        detail::OfflineSession job(options);
        detail::OfflineProductWorkspace work(job, frames, block);
        int largestRead = 0;
        auto reader = [&](std::int64_t first, int count, double *output) {
            EXPECT_TRUE(first >= 0 && count > 0 && first + count <= frames);
            EXPECT_TRUE(count <= 4096);
            largestRead = std::max(largestRead, count);
            std::copy_n(x.data() + first, count, output);
        };
        detail::OfflineHilbertMap map(work, reader);
        for (bool alternating : {false, true})
            for (std::size_t leaf = 0; leaf < work.leaves(); ++leaf)
            {
                map.evaluateCauchy(work, leaf, reader, out.data(), alternating);
                for (int offset : {0, 1, block / 2, block - 1})
                {
                    const auto frame = static_cast<std::int64_t>(leaf) * block + offset;
                    if (frame >= frames)
                        continue;
                    long double expected = 0;
                    for (int i = 0; i < frames; ++i)
                        if (i != frame)
                            expected += (alternating && i % 2 ? -1.L : 1.L) * x[i] /
                                (std::numbers::pi_v<long double> * (frame - i));
                    EXPECT_NEAR(out[offset], static_cast<double>(expected), 3e-12);
                }
            }
        EXPECT_EQ(largestRead, 4096);
    }
}

DSPARK_TEST(OfflineProduct_SharedCauchyAndAlternatingMomentsMatchDirectSums)
{
    for (int n : {1, 2, 3, 63, 128, 257, 4131, 12345})
    {
        std::vector<double> x(n);
        for (int i = 0; i < n; ++i)
            x[i] = std::sin(.217 * i) + .2 * std::cos(.763 * i);
        const OfflineJobOptions options;
        detail::OfflineSession job(options);
        detail::OfflineProductWorkspace work(job, n, 128);
        auto reader = [&](std::int64_t first, int count, double *output) {
            std::copy_n(x.data() + first, count, output);
        };
        detail::OfflineHilbertMap map(work, reader);
        std::array<double, 128> out{};
        for (bool alternating : {false, true})
            for (std::size_t leaf = 0; leaf < work.leaves(); ++leaf)
            {
                map.evaluateCauchy(work, leaf, reader, out.data(), alternating);
                for (int k = 0; k < 128 && static_cast<int>(leaf * 128) + k < n; ++k)
                {
                    const int frame = static_cast<int>(leaf * 128) + k;
                    if (n > 1000 && frame % 137 && frame != n - 1)
                        continue;
                    long double expected = 0;
                    for (int i = 0; i < n; ++i)
                        if (i != frame)
                            expected += (alternating && i % 2 ? -1.L : 1.L) * x[i] /
                                (std::numbers::pi_v<long double> * (frame - i));
                    EXPECT_NEAR(out[k], static_cast<double>(expected), 3e-12);
                }
            }
        // Independent fractional-position sums exclude exactly the three
        // neighbouring leaves, matching the exposed far-field contract.
        for (bool alternating : {false, true})
            for (double shift : {-1., -.875, -.5, -.125, 0., .125, .5, .875, 1.})
                for (std::size_t leaf = 0; leaf < work.leaves(); ++leaf)
                {
                    map.evaluateCauchyFar(leaf, shift, out.data(), alternating);
                    for (int k : {0, 1, 63, 126, 127})
                    {
                        const auto frame = static_cast<std::int64_t>(leaf) * 128 + k;
                        long double expected = 0;
                        for (int i = 0; i < n; ++i)
                            if (std::abs(i / 128 - static_cast<std::int64_t>(leaf)) > 1)
                                expected += (alternating && i % 2 ? -1.L : 1.L) * x[i] /
                                    (std::numbers::pi_v<long double> *
                                     (frame + static_cast<long double>(shift) - i));
                        EXPECT_NEAR(out[k], static_cast<double>(expected), 3e-12);
                    }
                }
        for (double shift : {-1.001, 1.001, std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()})
        {
            bool rejected = false;
            try
            {
                map.evaluateCauchyFar(0, shift, out.data());
            }
            catch (const detail::OfflineFailure &error)
            {
                rejected = error.status == OfflineStatus::InvalidInput;
            }
            EXPECT_TRUE(rejected);
        }
        bool rejected = false;
        try
        {
            map.evaluateCauchyFar(work.leaves(), 0, out.data());
        }
        catch (const detail::OfflineFailure &error)
        {
            rejected = error.status == OfflineStatus::InvalidInput;
        }
        EXPECT_TRUE(rejected);
    }
}

namespace
{
struct ProductResult
{
    std::vector<double> samples;
    OfflineStatus status = OfflineStatus::Success;
    std::size_t bytes = 0;
    int largestRead = 0;
};
ProductResult product(const std::vector<double> &x, const std::vector<double> &g, int block = 128,
                      const OfflineJobOptions &options = {})
{
    ProductResult result;
    result.samples.resize(x.size());
    try
    {
        detail::OfflineSession job(options);
        const auto reader = [&](const std::vector<double> &data) {
            return [&data, &result, block](std::int64_t first, int count, double *out) {
                if (first < 0 || count < 1 || count > block ||
                    static_cast<std::uint64_t>(first) + count > data.size())
                    detail::offlineFail(OfflineStatus::SourceReadFailed);
                result.largestRead = std::max(result.largestRead, count);
                std::copy_n(data.data() + first, count, out);
            };
        };
        std::int64_t next = 0;
        detail::offlineBandlimitedProduct(
            job, static_cast<std::int64_t>(x.size()), reader(x), reader(g),
            [&](std::int64_t first, int count, const double *data) {
                if (first != next || count < 1 || count > block ||
                    static_cast<std::uint64_t>(first) + count > result.samples.size())
                    detail::offlineFail(OfflineStatus::SinkFailed);
                std::copy_n(data, count, result.samples.data() + first);
                next += count;
            },
            block);
        if (next != static_cast<std::int64_t>(x.size()))
            detail::offlineFail(OfflineStatus::SinkFailed);
        result.bytes = job.bytes();
    }
    catch (...)
    {
        result.status = detail::offlineExceptionStatus();
    }
    return result;
}

// Integral over the real line of sinc(t-i)*sinc(t-j)*sinc(t-n).
// This oracle evaluates the individual interpolation basis products. It uses
// neither a Hilbert transform nor the production factorization/tree/FFT.
long double tripleSinc(std::int64_t i, std::int64_t j, std::int64_t n)
{
    if (i == j && j == n)
        return .75L;
    if (i % 2 == j % 2 && j % 2 == n % 2)
        return 0;
    std::int64_t odd = n, a = i, b = j;
    if (i % 2 == n % 2)
    {
        odd = j;
        b = n;
    }
    else if (j % 2 == n % 2)
    {
        odd = i;
        a = j;
        b = n;
    }
    return 1 / (std::numbers::pi_v<long double> * std::numbers::pi_v<long double> *
                static_cast<long double>(odd - a) * static_cast<long double>(odd - b));
}
std::vector<long double> directProduct(const std::vector<double> &x, const std::vector<double> &g)
{
    std::vector<long double> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        if (x[i] != 0)
            for (std::size_t j = 0; j < g.size(); ++j)
                if (g[j] != 0)
                    for (std::size_t n = 0; n < y.size(); ++n)
                        y[n] += static_cast<long double>(x[i]) * g[j] *
                                tripleSinc(static_cast<std::int64_t>(i), static_cast<std::int64_t>(j),
                                           static_cast<std::int64_t>(n));
    return y;
}
} // namespace

DSPARK_TEST(OfflineProduct_IndependentTripleSincFiniteOracle)
{
    for (int n : {1, 2, 3, 17, 65})
    {
        std::vector<double> x(n), g(n);
        for (int i = 0; i < n; ++i)
        {
            x[i] = std::sin(1.317 * i) + .2;
            g[i] = .7 + .3 * std::cos(2.198 * i);
        }
        const auto expected = directProduct(x, g);
        const auto result = product(x, g);
        EXPECT_TRUE(result.status == OfflineStatus::Success);
        for (int i = 0; i < n; ++i)
            EXPECT_NEAR(result.samples[i], static_cast<double>(expected[i]), 2e-12);
    }
}

DSPARK_TEST(OfflineProduct_SparseRaggedLongInputsDoNotWrap)
{
    for (int n : {127, 128, 129, 4095, 4096, 4097, 49153})
    {
        std::vector<double> x(n), g(n);
        x[0] = 1;
        x[n / 2] = -.75;
        x[n - 1] = .125;
        g[1] = .5;
        g[n / 2 + 1] = -.25;
        g[n - 1] = 1;
        const auto expected = directProduct(x, g);
        for (int block : {128, 4096})
        {
            const auto result = product(x, g, block);
            EXPECT_TRUE(result.status == OfflineStatus::Success);
            EXPECT_TRUE(result.largestRead <= block);
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(result.samples[i], static_cast<double>(expected[i]), 2e-12);
        }
    }
}

DSPARK_TEST(OfflineProduct_CommutativityLinearityAndTimeReversal)
{
    constexpr int n = 1031;
    std::vector<double> x(n), g(n), h(n), sum(n);
    for (int i = 0; i < n; ++i)
    {
        x[i] = std::cos(3.131 * i);
        g[i] = .4 * std::sin(.0037 * i);
        h[i] = .6 * std::cos(.17 * i);
        sum[i] = g[i] + h[i];
    }
    const auto xg = product(x, g), gx = product(g, x), xh = product(x, h), xs = product(x, sum);
    EXPECT_TRUE(xg.status == OfflineStatus::Success && gx.status == OfflineStatus::Success &&
                xh.status == OfflineStatus::Success && xs.status == OfflineStatus::Success);
    std::reverse(x.begin(), x.end());
    std::reverse(g.begin(), g.end());
    const auto reversed = product(x, g);
    EXPECT_TRUE(reversed.status == OfflineStatus::Success);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(xg.samples[i], gx.samples[i], 2e-12);
        EXPECT_NEAR(xs.samples[i], xg.samples[i] + xh.samples[i], 2e-12);
        EXPECT_NEAR(xg.samples[i], reversed.samples[n - 1 - i], 2e-12);
    }
}

DSPARK_TEST(OfflineProduct_FullBandModulationRejectsScalarAliasing)
{
    constexpr int n = 65;
    for (double frequency : {0., .1, .25, .48, .499, .5})
    {
        std::vector<double> x(n), g(n);
        for (int i = 0; i < n; ++i)
        {
            x[i] = std::cos(2 * std::numbers::pi * frequency * i);
            g[i] = .5 + .45 * std::cos(2 * std::numbers::pi * .08 * i);
        }
        const auto expected = directProduct(x, g);
        const auto result = product(x, g);
        EXPECT_TRUE(result.status == OfflineStatus::Success);
        long double error = 0, rawError = 0, inputPower = 0;
        for (int i = 0; i < n; ++i)
        {
            const auto delta = result.samples[i] - expected[i];
            const auto raw = x[i] * g[i] - expected[i];
            error += delta * delta;
            rawError += raw * raw;
            inputPower += x[i] * x[i];
        }
        EXPECT_LT(static_cast<double>(error / inputPower), 1e-9); // -90 dB relative to input.
        if (frequency >= .48)
            EXPECT_GT(static_cast<double>(rawError / inputPower), 1e-3);
    }
}

DSPARK_TEST(OfflineProduct_BoundedStorageBudgetAndInt64Preflight)
{
    const std::vector<double> x(513, .25), g(513, .5);
    const auto initial = product(x, g);
    EXPECT_TRUE(initial.status == OfflineStatus::Success);
    OfflineJobOptions options;
    options.memoryBudgetBytes = initial.bytes;
    EXPECT_TRUE(product(x, g, 128, options).status == OfflineStatus::Success);
    --options.memoryBudgetBytes;
    EXPECT_TRUE(product(x, g, 128, options).status == OfflineStatus::MemoryLimit);
    for (auto length : {std::int64_t(1) << 34, std::numeric_limits<std::int64_t>::max()})
    {
        bool read = false;
        OfflineStatus status = OfflineStatus::Success;
        try
        {
            detail::OfflineSession job(options);
            auto source = [&](std::int64_t, int, double *) { read = true; };
            detail::offlineBandlimitedProduct(job, length, source, source,
                                             [](std::int64_t, int, const double *) {});
        }
        catch (...)
        {
            status = detail::offlineExceptionStatus();
        }
        EXPECT_TRUE(status == OfflineStatus::MemoryLimit);
        EXPECT_FALSE(read);
    }
}

DSPARK_TEST(OfflineProduct_CancellationInvalidInputAndZeroOperand)
{
    const std::vector<double> x(1031, .25), g(1031, 0);
    const auto zero = product(x, g);
    EXPECT_TRUE(zero.status == OfflineStatus::Success);
    for (double y : zero.samples)
        EXPECT_EQ(y, 0.);
    std::atomic<bool> cancelled{true};
    OfflineJobOptions options;
    options.cancel = &cancelled;
    EXPECT_TRUE(product(x, g, 128, options).status == OfflineStatus::Cancelled);
    options.cancel = nullptr;
    int calls = 0;
    options.progressContext = &calls;
    options.progress = [](void *state, OfflineProgress) { return ++*static_cast<int *>(state) < 7; };
    EXPECT_TRUE(product(x, g, 128, options).status == OfflineStatus::Cancelled);
    options.progress = [](void *, OfflineProgress) -> bool { throw 1; };
    EXPECT_TRUE(product(x, g, 128, options).status == OfflineStatus::CallbackFailed);
    EXPECT_TRUE(product(x, g, 127).status == OfflineStatus::InvalidInput);
    auto bad = x;
    bad[19] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(product(bad, g).status == OfflineStatus::NonFiniteInput);
}
