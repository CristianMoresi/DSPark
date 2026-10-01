// DSPark - Offline gain-render signal, source and publication contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Effects/OfflineLeveler.h"
#include "../Effects/OfflinePeakCompressor.h"
#include "../Effects/OfflinePunch.h"
#include "../Core/FFT.h"
#include "dspark_test.h"
#include "offline_gain_reference.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

using namespace dspark;
using dspark::test::offline_reference::basisIntegral;
using dspark::test::offline_reference::referenceDelta;

namespace
{
struct GainReport
{
    double outputSamplePeakDb = 0, outputTruePeakDb = 0;
    bool peaksMeasured = false;
    OfflineGainRenderInfo renderInfo;
};
struct GainResult
{
    OfflineStatus status = OfflineStatus::EmptyInput;
    GainReport report;
    std::size_t memoryBytes = 0;
    int largestRead = 0;
};
template <class T> class GainSink final : public OfflineAudioSink<T>
{
  public:
    AudioBuffer<T> audio;
    bool begun = false, committed = false, aborted = false;
    int begins = 0;
    std::int64_t next = 0;
    int largestWrite = 0;
    bool begin(const OfflineAudioSpec &spec) override
    {
        begun = true;
        ++begins;
        audio.resize(spec.channels, static_cast<int>(spec.frames));
        return true;
    }
    bool write(std::int64_t first, AudioBufferView<const T> block) override
    {
        if (first != next)
            return false;
        for (int c = 0; c < block.getNumChannels(); ++c)
            std::copy_n(block.getChannel(c), block.getNumSamples(), audio.getChannel(c) + first);
        next += block.getNumSamples();
        largestWrite = std::max(largestWrite, block.getNumSamples());
        return true;
    }
    bool commit() override
    {
        committed = next == audio.getNumSamples();
        return committed;
    }
    void abort() noexcept override { aborted = true; }
};
template <class T> class GainSource final : public OfflineAudioSource<T>
{
  public:
    explicit GainSource(AudioBuffer<T> &audio) : audio_(audio) {}
    int reads = 0, changeAt = 0, largestRead = 0;
    [[nodiscard]] OfflineAudioSpec getSpec() const noexcept override
    {
        return {48000, audio_.getNumChannels(), audio_.getNumSamples()};
    }
    bool read(std::int64_t first, AudioBufferView<T> block) override
    {
        ++reads;
        largestRead = std::max(largestRead, block.getNumSamples());
        for (int c = 0; c < block.getNumChannels(); ++c)
            std::copy_n(audio_.getChannel(c) + first, block.getNumSamples(), block.getChannel(c));
        if (reads == changeAt)
            block.getChannel(0)[0] *= T(.5);
        return true;
    }

  private:
    AudioBuffer<T> &audio_;
};
template <class T>
GainResult renderControl(AudioBuffer<T> &audio, const std::vector<double> &gain,
                         GainSink<T> &sink, const OfflineJobOptions &options = {},
                         bool peak = false, int changeAt = 0, int *reads = nullptr,
                         const detail::OfflineGainRenderSettings *settings = nullptr)
{
    GainSource<T> source(audio);
    source.changeAt = changeAt;
    OfflineFingerprint fingerprint;
    double maximum = 0;
    for (int i = 0; i < audio.getNumSamples(); ++i)
        for (int c = 0; c < audio.getNumChannels(); ++c)
        {
            const T x = audio.getChannel(c)[i];
            detail::offlineHash(fingerprint, x);
            maximum = std::max(maximum, std::abs(static_cast<double>(x)));
        }
    auto result = detail::offlineRenderGain<GainResult>(
        source, source.getSpec(), fingerprint, maximum, true, true, GainReport{}, sink, options,
        [&](detail::OfflineSession &) {
            return [&](std::int64_t frame) { return gain[static_cast<std::size_t>(frame)]; };
        },
        settings ? *settings : detail::OfflineGainRenderSettings{nullptr, 1, peak});
    if (reads)
        *reads = source.reads;
    result.largestRead = source.largestRead;
    return result;
}

} // namespace

DSPARK_TEST(OfflineGainRender_EndpointHalfLinesMatchOriginalBasisWithConvergedGuards)
{
    using namespace dspark::test::offline_reference;
    for (int n : {1, 2, 7, 18, 65})
    {
        const auto weights = endpointWeights(n, -.7, .2);
        for (int i : {0, n / 2, n - 1})
            for (int k : {0, n - 1})
            {
                std::array<long double, 4> sums{};
                for (int level = 0; level < 4; ++level)
                {
                    const int guard = 4096 << level;
                    for (int j = -guard; j < 0; ++j)
                        sums[level] += -.7L * basisIntegral(i, j, k);
                    for (int j = n; j < n + guard; ++j)
                        sums[level] += .2L * basisIntegral(i, j, k);
                }
                const long double extrapolated =
                    (-sums[0] + 14 * sums[1] - 56 * sums[2] + 64 * sums[3]) / 21;
                EXPECT_NEAR(static_cast<double>(extrapolated),
                            static_cast<double>(endpointIntegral(weights, i, k)), 3e-12);
            }
    }
}

DSPARK_TEST(OfflineGainRender_EndpointCalibrationBothEndsAndTimeReversal)
{
    constexpr int n = 1025, window = 90;
    for (double frequency : {.0, .13, .25, .499})
        for (bool reverse : {false, true})
        {
            AudioBuffer<double> input;
            input.resize(2, n);
            std::vector<double> gain(n, 1), x(n);
            double leftPeak = 0, rightPeak = 0;
            for (int i = 0; i < window; ++i)
            {
                x[i] = std::exp(-.02 * i) * std::cos(twoPi<double> * frequency * i);
                x[n - 1 - i] = .8 * std::exp(-.03 * i) * std::cos(twoPi<double> * frequency * i + .3);
                leftPeak = std::max(leftPeak, std::abs(x[i]));
                rightPeak = std::max(rightPeak, std::abs(x[n - 1 - i]));
            }
            const double g = decibelsToGain(-18.);
            std::fill_n(gain.begin(), 200, g);
            std::fill_n(gain.end() - 200, 200, g);
            if (reverse)
            {
                std::reverse(x.begin(), x.end());
                std::swap(leftPeak, rightPeak);
            }
            for (int i = 0; i < n; ++i)
            {
                input.getChannel(0)[i] = x[i];
                input.getChannel(1)[i] = -x[i];
            }
            detail::OfflineGainRenderSettings settings;
            settings.boundaryTargets = {{{{0, window}, leftPeak * g},
                                         {{n - window, n}, rightPeak * g}}};
            GainSink<double> sink;
            const auto result = renderControl(input, gain, sink, {}, false, 0, nullptr, &settings);
            EXPECT_TRUE(result.status == OfflineStatus::Success && sink.committed);
            EXPECT_EQ(result.report.renderInfo.minimumControlGain, g);
            const auto &info = result.report.renderInfo;
            EXPECT_TRUE(info.leftBoundaryGain >= 0 && info.leftBoundaryGain <= g);
            EXPECT_TRUE(info.rightBoundaryGain >= 0 && info.rightBoundaryGain <= g);
            const auto expected = referenceDelta(x, gain, info.leftBoundaryGain, info.rightBoundaryGain);
            double outputLeft = 0, outputRight = 0;
            for (int i = 0; i < n; ++i)
            {
                EXPECT_NEAR(sink.audio.getChannel(0)[i], x[i] + expected[i], 3e-12);
                EXPECT_EQ(sink.audio.getChannel(1)[i], -sink.audio.getChannel(0)[i]);
                if (i < window)
                    outputLeft = std::max(outputLeft, std::abs(sink.audio.getChannel(0)[i]));
                if (i >= n - window)
                    outputRight = std::max(outputRight, std::abs(sink.audio.getChannel(0)[i]));
            }
            EXPECT_TRUE(outputLeft <= leftPeak * g + 3e-12);
            EXPECT_TRUE(outputRight <= rightPeak * g + 3e-12);
            EXPECT_NEAR(20 * std::log10(leftPeak / outputLeft), 18., .05);
            EXPECT_NEAR(20 * std::log10(rightPeak / outputRight), 18., .05);
        }
}

DSPARK_TEST(OfflineGainRender_IndependentFiniteBasisAndEndpointContinuation)
{
    constexpr int n = 37;
    for (double frequency : {0., .1, .25, .48, .499, .5})
    {
        AudioBuffer<double> input;
        input.resize(2, n);
        std::vector<double> gain(n);
        for (int i = 0; i < n; ++i)
        {
            input.getChannel(0)[i] = std::cos(2 * std::numbers::pi * frequency * i);
            input.getChannel(1)[i] = .7 * std::sin(.397 * i);
            gain[i] = 1.5 + .4 * std::cos(.177 * i);
        }
        GainSink<double> sink;
        const auto result = renderControl(input, gain, sink);
        EXPECT_TRUE(result.status == OfflineStatus::Success);
        EXPECT_TRUE(result.report.renderInfo.bandlimited && sink.committed);
        const double boundary = (gain.front() + gain.back()) * .5;
        EXPECT_EQ(result.report.renderInfo.boundaryGain, boundary);
        EXPECT_EQ(result.report.renderInfo.leftBoundaryGain, gain.front());
        EXPECT_EQ(result.report.renderInfo.rightBoundaryGain, gain.back());
        const auto weights = dspark::test::offline_reference::endpointWeights(
            n, gain.front() - boundary, gain.back() - boundary);
        long double error = 0, power = 0, scalarError = 0;
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < n; ++k)
            {
                long double expected = boundary * input.getChannel(c)[k];
                for (int i = 0; i < n; ++i)
                    expected += input.getChannel(c)[i] *
                        dspark::test::offline_reference::endpointIntegral(weights, i, k);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < n; ++j)
                        expected += input.getChannel(c)[i] * (gain[j] - boundary) *
                                    basisIntegral(i, j, k);
                const long double difference = sink.audio.getChannel(c)[k] - expected;
                error += difference * difference;
                const auto raw = input.getChannel(c)[k] * gain[k] - expected;
                scalarError += raw * raw;
                power += input.getChannel(c)[k] * input.getChannel(c)[k];
            }
        EXPECT_LT(static_cast<double>(error / power), 1e-9); // Unchanged -90 dB gate.
        if (frequency >= .48)
            EXPECT_GT(static_cast<double>(scalarError / power), 1e-4);
    }
}

DSPARK_TEST(OfflineGainRender_ConstantGainAndNeutralPcmAreExact)
{
    const auto check = []<class T>() {
        for (int n : {1, 2, 129, 4097})
        {
            AudioBuffer<T> input;
            input.resize(2, n);
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < n; ++i)
                    input.getChannel(c)[i] = i % 3 ? T(.1 * std::cos(i + c)) : -T(0);
            for (double value : {.5, 1., 2.})
            {
                const std::vector<double> gain(n, value);
                GainSink<T> sink;
                const auto result = renderControl(input, gain, sink);
                EXPECT_TRUE(result.status == OfflineStatus::Success);
                EXPECT_FALSE(result.report.renderInfo.bandlimited);
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < n; ++i)
                    {
                        const T expected = static_cast<T>(input.getChannel(c)[i] * value);
                        EXPECT_EQ(
                            std::memcmp(&sink.audio.getChannel(c)[i], &expected, sizeof(T)), 0);
                    }
            }
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

DSPARK_TEST(OfflineGainRender_PublicProcessorsMatchIndependentFullBandOracle)
{
    const auto check = []<class Processor>(bool exclude) {
        constexpr bool peak = std::is_same_v<Processor, OfflinePeakCompressor<double>>;
        constexpr bool punch = std::is_same_v<Processor, OfflinePunch<double>>;
        constexpr int rate = 8000, n = peak || punch ? 4017 : 96017;
        AudioBuffer<double> input;
        input.resize(2, n);
        std::vector<double> x(n);
        for (int i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(i) / rate;
            double amplitude = .003;
            if constexpr (peak || punch)
            {
                for (double start : {.1, .3})
                    if (t >= start && t < start + .06)
                        amplitude += .8 * std::exp(-(t - start) / .012) *
                                     std::pow(std::clamp((start + .06 - t) / .01, 0., 1.), 2);
            }
            else
                amplitude = t < 3 ? .025 : t < 7 ? .18 : .07;
            x[i] = amplitude * std::cos(twoPi<double> * .499 * i);
            input.getChannel(0)[i] = x[i];
            input.getChannel(1)[i] = -.7 * x[i];
        }
        OfflineBufferSource<double> source(input.toView(), rate);
        typename Processor::Options options;
        if constexpr (peak)
            options.reductionDb = 6;
        else if constexpr (punch)
            options.boostDb = 6;
        else
            options.amount = 1;
        const OfflineRegion region{peak || punch ? 944 : 32960, peak || punch ? 1240 : 38800};
        if (exclude)
            options.exclusions = {&region, 1};
        const auto plan = Processor().analyze(source, options);
        EXPECT_TRUE(plan.succeeded());
        GainSink<double> sink;
        const auto result = Processor().render(source, plan.plan, sink);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(result.report.renderInfo.bandlimited);
        std::vector<double> g(n), delta;
        double target = 0;
        for (int i = 0; i < n; ++i)
        {
            g[i] = plan.plan.gainAt(i);
            target = std::max(target, std::abs(x[i] * g[i]));
        }
        delta = referenceDelta(x, g);
        double lo = 0, hi = peak ? 1 / (1 - *std::min_element(g.begin(), g.end())) : 1;
        for (int i = 0; i < n; ++i)
        {
            double weight = 1;
            if (exclude)
            {
                const double distance = i < region.begin ? double(region.begin - i)
                                                         : double(i - region.end);
                const double t = std::clamp(distance / (peak ? 16. : punch ? 40. : 1600.), 0., 1.);
                weight = t * t * t * t * (35 - t * (84 - t * (70 - 20 * t)));
                if (i >= region.begin && i < region.end)
                    weight = 0;
            }
            const double scalar = x[i] * (g[i] - 1);
            delta[i] = scalar + weight * (delta[i] - scalar);
            if constexpr (peak)
                if (delta[i] != 0)
                {
                    const double a = (-target - x[i]) / delta[i];
                    const double b = (target - x[i]) / delta[i];
                    lo = std::max(lo, std::min(a, b));
                    hi = std::min(hi, std::max(a, b));
                }
        }
        EXPECT_TRUE(lo <= hi);
        const double scale = std::clamp(1., lo, hi);
        EXPECT_NEAR(result.report.renderInfo.deltaScale, scale, 2e-10);
        double error = 0, power = 0, scalarError = 0;
        for (int i = 0; i < n; ++i)
        {
            const double expected = x[i] + scale * delta[i];
            error += std::pow(sink.audio.getChannel(0)[i] - expected, 2);
            scalarError += std::pow(x[i] * g[i] - expected, 2);
            power += x[i] * x[i];
            EXPECT_NEAR(sink.audio.getChannel(1)[i], -.7 * sink.audio.getChannel(0)[i], 2e-12);
        }
        EXPECT_LT(error / power, 1e-9);
        if (!exclude)
            EXPECT_GT(scalarError / power, 1e-8);
        if (exclude)
            for (int c = 0; c < 2; ++c)
                EXPECT_EQ(
                    std::memcmp(input.getChannel(c) + region.begin,
                                sink.audio.getChannel(c) + region.begin,
                                static_cast<std::size_t>(region.end - region.begin) * sizeof(double)),
                    0);
    };
    for (bool exclude : {false, true})
    {
        check.template operator()<OfflinePeakCompressor<double>>(exclude);
        check.template operator()<OfflinePunch<double>>(exclude);
        check.template operator()<OfflineLeveler<double>>(exclude);
    }
}

DSPARK_TEST(OfflineGainRender_InfeasibleScaleCannotPublish)
{
    AudioBuffer<double> input;
    input.resize(1, 17);
    input.getChannel(0)[8] = 1;
    std::vector<double> gain(17, 1);
    gain[8] = .1;
    GainSink<double> sink;
    const auto result = renderControl(input, gain, sink, {}, true);
    EXPECT_TRUE(result.status == OfflineStatus::NumericalFailure);
    EXPECT_FALSE(result.report.renderInfo.targetFeasible);
    EXPECT_FALSE(sink.begun || sink.committed);
    // Independent impulse basis: even the maximum nonnegative control scale
    // leaves 0.25 at the impulse, above the requested sample target of 0.1.
    EXPECT_NEAR(1 + (1 / .9) * (-.9 * static_cast<double>(basisIntegral(8, 8, 8))), .25, 1e-15);
}

DSPARK_TEST(OfflineGainRender_PeakExtendsOnlyNecessaryAnticipation)
{
    constexpr int rate = 48000, n = rate / 2 + 17;
    AudioBuffer<double> input;
    input.resize(1, n);
    for (int i = 0; i < n; ++i)
    {
        const double t = double(i) / rate;
        double a = .003;
        for (double start : {.1, .3})
            if (t >= start && t < start + .06)
                a += .8 * std::exp(-(t - start) / .012) *
                     std::pow(std::clamp((start + .06 - t) / .01, 0., 1.), 2);
        input.getChannel(0)[i] = a * std::cos(twoPi<double> * 23952 * t);
    }
    OfflineBufferSource<double> source(input.toView(), rate);
    OfflinePeakCompressor<double> processor;
    OfflinePeakCompressor<double>::Options options{18, .1, 80};
    const auto plan = processor.analyze(source, options);
    EXPECT_TRUE(plan.succeeded());
    GainSink<double> sink;
    const auto result = processor.render(source, plan.plan, sink);
    EXPECT_TRUE(result.succeeded() && sink.committed);
    EXPECT_EQ(sink.begins, 1);
    EXPECT_GT(result.report.lookaheadMs, .1);
    EXPECT_TRUE(result.report.lookaheadMs <= 10);
    EXPECT_GT(result.report.renderAttempts, 1);
    EXPECT_EQ(result.report.requestedLookaheadMs, .1);
    EXPECT_EQ(result.report.releaseMs, 80.);
    EXPECT_NEAR(result.report.achievedReductionDb, result.report.effectiveReductionDb, .001);
    EXPECT_NEAR(result.report.renderInfo.minimumControlGain,
                1 + result.report.renderInfo.deltaScale *
                        (decibelsToGain(-result.report.maximumReductionDb) - 1), 2e-12);
    EXPECT_TRUE(result.report.renderInfo.minimumControlGain >= 0);
    EXPECT_TRUE(result.report.renderInfo.maximumControlGain <= 1);
    options.lookaheadMs = result.report.lookaheadMs;
    const auto extended = processor.analyze(source, options);
    EXPECT_TRUE(extended.succeeded());
    GainSink<double> reference;
    const auto direct = processor.render(source, extended.plan, reference);
    EXPECT_TRUE(direct.succeeded());
    EXPECT_EQ(direct.report.renderAttempts, 1);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(sink.audio.getChannel(0)[i], reference.audio.getChannel(0)[i], 2e-12);
    OfflineJobOptions budget;
    budget.memoryBudgetBytes = result.memoryBytes;
    GainSink<double> exact;
    EXPECT_TRUE(processor.render(source, plan.plan, exact, budget).succeeded());
    --budget.memoryBudgetBytes;
    GainSink<double> insufficient;
    EXPECT_TRUE(processor.render(source, plan.plan, insufficient, budget).status ==
                OfflineStatus::MemoryLimit);
    EXPECT_FALSE(insufficient.begun);
}

DSPARK_TEST(OfflineGainRender_RepeatedReadVerificationBudgetAndCancellation)
{
    constexpr int n = 8193;
    AudioBuffer<double> input;
    input.resize(2, n);
    std::vector<double> gain(n);
    for (int i = 0; i < n; ++i)
    {
        input.getChannel(0)[i] = .5 + .25 * std::cos(3.01 * i);
        input.getChannel(1)[i] = -.25 + .1 * std::sin(.37 * i);
        gain[i] = 1.5 + .4 * std::cos(.017 * i);
    }
    GainSink<double> baseline;
    int reads = 0;
    OfflineJobOptions options;
    options.blockFrames = 37;
    const auto original = renderControl(input, gain, baseline, options, false, 0, &reads);
    EXPECT_TRUE(original.status == OfflineStatus::Success);
    EXPECT_TRUE(baseline.largestWrite <= 37);
    EXPECT_TRUE(original.largestRead <= 37);
    for (int changed : {1, 2, reads / 2, reads - 1, reads})
    {
        GainSink<double> sink;
        const auto result = renderControl(input, gain, sink, options, false, changed);
        EXPECT_TRUE(result.status == OfflineStatus::SourceMismatch);
        EXPECT_FALSE(sink.committed);
        EXPECT_TRUE(!sink.begun || sink.aborted);
    }
    options.memoryBudgetBytes = original.memoryBytes;
    GainSink<double> exact;
    EXPECT_TRUE(renderControl(input, gain, exact, options).status == OfflineStatus::Success);
    --options.memoryBudgetBytes;
    GainSink<double> insufficient;
    EXPECT_TRUE(renderControl(input, gain, insufficient, options).status ==
                OfflineStatus::MemoryLimit);
    EXPECT_FALSE(insufficient.begun);
    options = {};
    options.progress = [](void *, OfflineProgress p) { return p.phase != OfflinePhase::Render; };
    GainSink<double> cancelled;
    EXPECT_TRUE(renderControl(input, gain, cancelled, options).status == OfflineStatus::Cancelled);
    EXPECT_FALSE(cancelled.committed);
    EXPECT_TRUE(!cancelled.begun || cancelled.aborted);
}

DSPARK_TEST(OfflineGainRender_FloatingPointLimitCapsOnlyProcessingDelta)
{
    const auto check = []<class T>() {
        constexpr int n = 129;
        AudioBuffer<T> input;
        input.resize(2, n);
        std::vector<double> gain(n);
        for (int i = 0; i < n; ++i)
        {
            input.getChannel(0)[i] = static_cast<T>(std::numeric_limits<T>::max() *
                                                   (.5 + .25 * std::cos(.31 * i)));
            input.getChannel(1)[i] = -input.getChannel(0)[i];
            gain[i] = 2.5 + .4 * std::sin(.077 * i);
        }
        GainSink<T> sink;
        const auto result = renderControl(input, gain, sink);
        EXPECT_TRUE(result.status == OfflineStatus::Success);
        EXPECT_TRUE(result.report.renderInfo.representabilityLimited);
        EXPECT_GT(result.report.renderInfo.deltaScale, 0.);
        EXPECT_LT(result.report.renderInfo.deltaScale, 1.);
        for (int i = 0; i < n; ++i)
        {
            EXPECT_TRUE(std::isfinite(sink.audio.getChannel(0)[i]));
            EXPECT_EQ(sink.audio.getChannel(1)[i], -sink.audio.getChannel(0)[i]);
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}
