// DSPark - Offline transient boost signal and transaction contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Core/ProcessorTraits.h"
#include "../Effects/OfflinePunch.h"
#include "dspark_test.h"
#include "offline_gain_reference.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflinePunch<float>, float>);

namespace
{
template <class T> AudioBuffer<T> punchSong(int rate)
{
    AudioBuffer<T> input;
    input.resize(2, rate);
    for (int i = 0; i < rate; ++i)
    {
        double x = .001 * std::sin(twoPi<double> * 73 * i / rate);
        for (int event = 0; event < 3; ++event)
        {
            const auto age = i - (event + 1) * rate / 5;
            if (age >= 0 && age < rate * .008)
                x += (.25 - event * .05) * std::cos(twoPi<double> * 1000 * age / rate);
        }
        input.getChannel(0)[i] = static_cast<T>(x);
        input.getChannel(1)[i] = static_cast<T>(-x);
    }
    return input;
}
template <class T> void punchAmounts()
{
    for (int rate : {8000, 44100, 48000, 192000})
    {
        auto input = punchSong<T>(rate);
        OfflineBufferSource<T> source(input.toView(), rate);
        auto map = OfflineTransientAnalyzer<T>().analyze(source);
        EXPECT_TRUE(map.succeeded());
        EXPECT_EQ(map.analysis.attacks().size(), std::size_t(3));
        for (double amount : {0., .1, 3., 6., 12.})
        {
            OfflinePunch<T> processor;
            auto plan = processor.makePlan(source, map.analysis, {amount});
            EXPECT_TRUE(plan.succeeded());
            EXPECT_NEAR(plan.plan.getReport().effectiveBoostDb, amount, 1e-12);
            for (const auto &event : map.analysis.attacks())
                EXPECT_NEAR(plan.plan.gainAt(event.begin), std::pow(10., amount / 20), 1e-12);
            AudioBuffer<T> output;
            const auto result = processor.run(input, output, rate, {amount});
            EXPECT_TRUE(result.succeeded());
            EXPECT_TRUE(result.report.peaksMeasured);
            EXPECT_EQ(output.getNumSamples(), rate);
            const auto reference =
                test::offline_reference::render(input, plan.plan, result.report.renderInfo);
            double peak = 0;
            for (int i = 0; i < rate; ++i)
            {
                EXPECT_EQ(output.getChannel(1)[i], -output.getChannel(0)[i]);
                peak = std::max(peak, std::abs(static_cast<double>(output.getChannel(0)[i])));
                if (i < rate / 10 || i > rate * .75)
                    EXPECT_EQ(plan.plan.gainAt(i), 1.);
                EXPECT_NEAR(output.getChannel(0)[i], reference.getChannel(0)[i],
                            (std::is_same_v<T, float> ? 2e-7 : 2e-12));
            }
            EXPECT_NEAR(result.report.outputSamplePeakDb, 20 * std::log10(peak), 1e-10);
            for (int event = 0; event < 3; ++event)
            {
                const int frame = (event + 1) * rate / 5;
                const double gain = output.getChannel(0)[frame] / input.getChannel(0)[frame];
                EXPECT_NEAR(20 * std::log10(gain), amount, .1);
            }
            if (amount == 0)
                for (int c = 0; c < 2; ++c)
                    EXPECT_EQ(std::memcmp(input.getChannel(c), output.getChannel(c),
                                          static_cast<std::size_t>(rate) * sizeof(T)),
                              0);
        }
    }
}
class PunchSink final : public OfflineAudioSink<float>
{
  public:
    bool begun = false, committed = false, aborted = false, fail = false;
    bool begin(const OfflineAudioSpec &) override
    {
        begun = true;
        return true;
    }
    bool write(std::int64_t, AudioBufferView<const float>) override
    {
        return !fail;
    }
    bool commit() override
    {
        committed = true;
        return true;
    }
    void abort() noexcept override
    {
        aborted = true;
    }
};
} // namespace

DSPARK_TEST(OfflinePunch_IndependentAmountsStereoLinkAndSourceTiming_Float)
{
    punchAmounts<float>();
}
DSPARK_TEST(OfflinePunch_IndependentAmountsStereoLinkAndSourceTiming_Double)
{
    punchAmounts<double>();
}

DSPARK_TEST(OfflinePunch_CosineMaximumAndFiniteKernelIndependentOracle)
{
    for (int rate : {8000, 44100, 48000, 192000, 384000})
    {
        const int span = static_cast<int>(std::llround(.002 * rate));
        const int rise = static_cast<int>(std::llround(.005 * rate));
        const int fall = static_cast<int>(std::llround(.045 * rate));
        const int start = rate / 10;
        const std::array<std::int64_t, 4> onsets{
            {start, start + rate * 3 / 100, start + rate * 8 / 100, start + rate * 15 / 100}};
        const std::array<double, 4> durations{{.05, .001, .03, .09}};
        OfflineJobOptions options;
        detail::OfflineSession job(options);
        detail::OfflinePunchEnvelope curve;
        curve.prepare(job, onsets.size(), rate);
        for (std::size_t i = 0; i < onsets.size(); ++i)
            curve.set(i, onsets[i], static_cast<int>(std::llround(durations[i] * rate)));
        // Convolve four unit boxes independently, without the production formula.
        std::vector<long double> kernel(1, 1);
        for (int box = 0; box < 4; ++box)
        {
            const auto size = static_cast<std::size_t>(span / 4 + (box < span % 4) + 1);
            std::vector<long double> next(kernel.size() + size - 1);
            for (std::size_t i = 0; i < kernel.size(); ++i)
                for (std::size_t k = 0; k < size; ++k)
                    next[i + k] += kernel[i] / size;
            kernel = std::move(next);
        }
        const auto raw = [&](std::int64_t f) {
            double weight = 0;
            for (std::size_t i = 0; i < onsets.size(); ++i)
            {
                const auto hold = static_cast<std::int64_t>(
                    std::llround(std::clamp(durations[i], .015, .05) * rate));
                const auto age = f - onsets[i];
                const double up =
                    std::clamp(static_cast<double>(age + rise - span) / (rise - span), 0.0, 1.0);
                const double down =
                    std::clamp(static_cast<double>(age - hold - span) / (fall - span), 0.0, 1.0);
                weight = std::max(weight, age < 0 ? .5 - .5 * std::cos(pi<double> * up)
                                                  : .5 + .5 * std::cos(pi<double> * down));
            }
            return weight;
        };
        detail::OfflinePunchEnvelope::Cursor cursor(curve, job);
        for (int frame = 0; frame < rate / 2; ++frame)
        {
            const auto value = cursor(frame);
            EXPECT_TRUE(value >= 0 && value <= 1);
            if (frame % 131 == 0)
            {
                long double expected = 0;
                for (int k = 0; k <= span; ++k)
                    expected += kernel[k] * raw(frame + k);
                EXPECT_NEAR(value, static_cast<double>(expected), 2e-13);
                EXPECT_NEAR(curve.weightAt(frame), static_cast<double>(expected), 2e-14);
            }
        }
        EXPECT_EQ(cursor(start - rise), 0.);
        EXPECT_EQ(cursor(rate - 1), 0.);
        EXPECT_NEAR(cursor(start), 1., 2e-14);
        EXPECT_NEAR(curve.weightAt(start + rate / 20), 1., 2e-14);
        EXPECT_LT(curve.weightAt(start - 1), 1.);
    }
}

DSPARK_TEST(OfflinePunch_DenseAttacksHaveEqualBoundedBoostWithoutTempo)
{
    constexpr int rate = 48000;
    AudioBuffer<double> input, output;
    input.resize(2, rate);
    for (int event = 0; event < 16; ++event)
    {
        const int start = 6000 + event * 1440;
        for (int i = 0; i < 192; ++i)
        {
            const double x = (event % 3 == 0 ? .1 : .7) * std::exp(-i / 60.) *
                             std::cos(twoPi<double> * 1700 * i / rate);
            input.getChannel(0)[start + i] = x;
            input.getChannel(1)[start + i] = -x;
        }
    }
    OfflineBufferSource<double> source(input.toView(), rate);
    OfflineTransientAnalyzer<double>::Options features;
    features.pulses = false;
    auto map = OfflineTransientAnalyzer<double>().analyze(source, features);
    EXPECT_TRUE(map.succeeded());
    EXPECT_EQ(map.analysis.attacks().size(), std::size_t(16));
    EXPECT_TRUE(map.analysis.pulses().empty());
    auto plan = OfflinePunch<double>().makePlan(source, map.analysis, {12});
    EXPECT_TRUE(plan.succeeded());
    for (int event = 0; event < 16; ++event)
        EXPECT_NEAR(20 * std::log10(plan.plan.gainAt(6000 + event * 1440)), 12., .1);
    for (int i = 0; i < rate; i += 17)
        EXPECT_TRUE(plan.plan.gainAt(i) >= 1 &&
                    plan.plan.gainAt(i) <=
                        std::pow(10., .6) * (1 + 8 * std::numeric_limits<double>::epsilon()));
    auto result = OfflinePunch<double>().run(input, output, rate, {12});
    EXPECT_TRUE(result.succeeded());
    EXPECT_GT(output.getChannel(0)[7440], 1.); // No hidden full-scale limiter.
}

DSPARK_TEST(OfflinePunch_ExactExclusionsAndSignedZero)
{
    auto input = punchSong<float>(8000);
    input.getChannel(0)[100] = -0.0f;
    input.getChannel(0)[1600] = -0.0f;
    const std::array<OfflineRegion, 2> regions{{{1600, 1750}, {1500, 1601}}};
    OfflinePunch<float>::Options options{6, regions};
    AudioBuffer<float> output;
    auto result = OfflinePunch<float>().run(input, output, 8000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(std::signbit(output.getChannel(0)[1600]));
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c) + 1500, output.getChannel(c) + 1500,
                              250 * sizeof(float)),
                  0);
    EXPECT_GT(output.getChannel(0)[3200], input.getChannel(0)[3200]);
    OfflineRegion all{0, 8000};
    options.exclusions = {&all, 1};
    result = OfflinePunch<float>().run(input, output, 8000, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(result.report.reason == OfflinePunch<float>::Reason::AllExcluded);
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c), output.getChannel(c), 8000 * sizeof(float)), 0);
}

DSPARK_TEST(OfflinePunch_BlockIdentityMoveAndInPlaceWeakerPlan)
{
    auto input = punchSong<double>(8000);
    AudioBuffer<double> reference;
    OfflinePunch<double> processor;
    EXPECT_TRUE(processor.run(input, reference, 8000, {3}).succeeded());
    for (int block : {1, 17, 4093, 999999})
    {
        OfflineJobOptions job;
        job.blockFrames = block;
        AudioBuffer<double> output;
        EXPECT_TRUE(processor.run(input, output, 8000, {3}, job).succeeded());
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(
                std::memcmp(output.getChannel(c), reference.getChannel(c), 8000 * sizeof(double)),
                0);
    }
    OfflineBufferSource<double> source(input.toView(), 8000, 91, 3, 9007199254741000ll);
    auto stronger = processor.analyze(source, {12});
    auto weaker = processor.analyze(source, {3});
    EXPECT_TRUE(stronger.succeeded() && weaker.succeeded());
    EXPECT_LT(weaker.plan.gainAt(1600), stronger.plan.gainAt(1600));
    auto moved = std::move(weaker.plan);
    EXPECT_FALSE(weaker.plan.isValid());
    EXPECT_EQ(moved.getSpec().timelineOrigin, 9007199254741000ll);
    EXPECT_NEAR(moved.gainAt(1600), std::pow(10., 3. / 20), 1e-12);
    EXPECT_TRUE(processor.run(input, input, 8000, {3}).succeeded());
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c), reference.getChannel(c), 8000 * sizeof(double)),
                  0);
}

DSPARK_TEST(OfflinePunch_InvalidStaleAndCancelledTransactions)
{
    auto input = punchSong<float>(8000);
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflinePunch<float> processor;
    auto plan = processor.analyze(source, {6});
    EXPECT_TRUE(plan.succeeded());
    PunchSink failure;
    failure.fail = true;
    EXPECT_TRUE(processor.render(source, plan.plan, failure).status == OfflineStatus::SinkFailed);
    EXPECT_TRUE(failure.aborted && !failure.committed);
    std::atomic<bool> cancel{true};
    OfflineJobOptions job;
    job.cancel = &cancel;
    PunchSink cancelled;
    EXPECT_TRUE(processor.render(source, plan.plan, cancelled, job).status ==
                OfflineStatus::Cancelled);
    EXPECT_FALSE(cancelled.begun);
    input.getChannel(0)[100] += .001f;
    PunchSink stale;
    EXPECT_TRUE(processor.render(source, plan.plan, stale).status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(stale.begun);
    AudioBuffer<float> output;
    output.resize(1, 1);
    output.getChannel(0)[0] = .125f;
    for (double invalid : {-1., 13., std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_TRUE(processor.run(input, output, 8000, {invalid}).status ==
                    OfflineStatus::InvalidInput);
        EXPECT_EQ(output.getChannel(0)[0], .125f);
    }
    input.getChannel(0)[50] = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(processor.run(input, output, 8000, {6}).status == OfflineStatus::NonFiniteInput);
    EXPECT_EQ(output.getChannel(0)[0], .125f);
}

DSPARK_TEST(OfflinePunch_RepresentabilityCapUsesAffectedSamplesOnly)
{
    const auto verify = []<class T>() {
        for (double proportion : {1., .5, .1})
        {
            AudioBuffer<T> input, output;
            input.resize(2, 8000);
            const T peak = static_cast<T>(std::numeric_limits<T>::max() * proportion);
            input.getChannel(0)[1000] = peak;
            input.getChannel(1)[1000] = -peak;
            auto result = OfflinePunch<T>().run(input, output, 8000, {12});
            EXPECT_TRUE(result.succeeded());
            EXPECT_TRUE(std::isfinite(output.getChannel(0)[1000]));
            EXPECT_TRUE(std::isfinite(result.report.outputTruePeakDb));
            EXPECT_NEAR(result.report.effectiveBoostDb, std::min(12., -20 * std::log10(proportion)),
                        1e-5);
            if (proportion >= .5)
                EXPECT_TRUE(result.report.reason ==
                            OfflinePunch<T>::Reason::RepresentabilityLimited);
            else
                EXPECT_GT(output.getChannel(0)[1000], peak);
        }
        AudioBuffer<T> input, output;
        input.resize(2, 8000);
        input.getChannel(0)[1000] = std::numeric_limits<T>::max();
        input.getChannel(0)[4000] = std::numeric_limits<T>::max() / T(100);
        OfflineRegion protectedPeak{990, 1100};
        typename OfflinePunch<T>::Options options{12, {&protectedPeak, 1}};
        auto result = OfflinePunch<T>().run(input, output, 8000, options);
        EXPECT_TRUE(result.succeeded());
        EXPECT_NEAR(result.report.effectiveBoostDb, 12., 1e-12);
        EXPECT_EQ(output.getChannel(0)[1000], input.getChannel(0)[1000]);
        EXPECT_GT(output.getChannel(0)[4000], input.getChannel(0)[4000]);
    };
    verify.template operator()<float>();
    verify.template operator()<double>();
}

DSPARK_TEST(OfflinePunch_ExtremeScaleAndSubnormalPcm)
{
    for (double scale : {1e-250, 1e250, std::numeric_limits<double>::denorm_min()})
    {
        AudioBuffer<double> input, output;
        input.resize(2, 8000);
        input.getChannel(0)[1000] = scale;
        input.getChannel(1)[1000] = -scale;
        auto result = OfflinePunch<double>().run(input, output, 8000, {6});
        EXPECT_TRUE(result.succeeded());
        OfflineBufferSource<double> source(input.toView(), 8000);
        const auto plan = OfflinePunch<double>().analyze(source, {6});
        EXPECT_TRUE(plan.succeeded());
        EXPECT_NEAR(plan.plan.gainAt(1000), std::pow(10., 6. / 20), 4e-14);
        const auto reference =
            test::offline_reference::render(input, plan.plan, result.report.renderInfo);
        for (int i = 0; i < input.getNumSamples(); ++i)
            EXPECT_NEAR(output.getChannel(0)[i] / scale,
                        reference.getChannel(0)[i] / scale, 4e-14);
        EXPECT_EQ(output.getChannel(1)[1000], -output.getChannel(0)[1000]);
        EXPECT_TRUE(std::isfinite(result.report.outputTruePeakDb));
    }
}

DSPARK_TEST(OfflinePunch_StationaryBodyControlAndBandlimitedOutput)
{
    constexpr int rate = 48000;
    for (double frequency : {31.25, 100., 440., 11000.})
    {
        AudioBuffer<double> input, output;
        input.resize(2, 2 * rate);
        for (int f = 0; f < input.getNumSamples(); ++f)
        {
            const double t = static_cast<double>(f) / rate;
            const double x =
                (.6 + .2 * std::sin(twoPi<double> * 5 * t)) *
                std::sin(twoPi<double> * frequency * t + .3 * std::sin(twoPi<double> * 6 * t));
            input.getChannel(0)[f] = x;
            input.getChannel(1)[f] = -x;
        }
        const auto result = OfflinePunch<double>().run(input, output, rate, {12});
        EXPECT_TRUE(result.succeeded());
        OfflineBufferSource<double> source(input.toView(), rate);
        const auto plan = OfflinePunch<double>().analyze(source, {12});
        EXPECT_TRUE(plan.succeeded());
        const auto reference =
            test::offline_reference::render(input, plan.plan, result.report.renderInfo);
        for (int f = 0; f < input.getNumSamples(); ++f)
            EXPECT_NEAR(output.getChannel(0)[f], reference.getChannel(0)[f], 2e-12);
        for (int f = rate / 2; f < 2 * rate; ++f)
            EXPECT_EQ(plan.plan.gainAt(f), 1.);
    }
}

DSPARK_TEST(OfflinePunch_ReuseFeatureSelectionAndExactMemoryBudget)
{
    auto input = punchSong<float>(8000);
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflineTransientAnalyzer<float>::Options options;
    options.attacks = options.pulses = false;
    auto energy = OfflineTransientAnalyzer<float>().analyze(source, options);
    EXPECT_TRUE(energy.succeeded());
    OfflinePunch<float> processor;
    EXPECT_TRUE(processor.makePlan(source, energy.analysis, {6}).status ==
                OfflineStatus::InvalidInput);
    EXPECT_TRUE(processor.makePlan(source, energy.analysis, {}).succeeded());
    auto map = OfflineTransientAnalyzer<float>().analyze(source);
    EXPECT_TRUE(map.succeeded());
    auto plan = processor.makePlan(source, map.analysis, {6});
    EXPECT_TRUE(plan.succeeded());
    OfflineJobOptions job;
    job.memoryBudgetBytes = plan.memoryBytes;
    EXPECT_TRUE(processor.makePlan(source, map.analysis, {6}, job).succeeded());
    --job.memoryBudgetBytes;
    EXPECT_TRUE(processor.makePlan(source, map.analysis, {6}, job).status ==
                OfflineStatus::MemoryLimit);
    job = {};
    job.progress = [](void *, OfflineProgress p) {
        return !(p.phase == OfflinePhase::Plan && p.completed == p.total);
    };
    auto cancelled = processor.makePlan(source, map.analysis, {6}, job);
    EXPECT_TRUE(cancelled.status == OfflineStatus::Cancelled);
    EXPECT_FALSE(cancelled.plan.isValid());
}

DSPARK_TEST(OfflinePunch_LongClockAndFirstLastImpulse)
{
    OfflineJobOptions options;
    detail::OfflineSession job(options);
    detail::OfflinePunchEnvelope distant;
    distant.prepare(job, 1, 48000);
    const std::int64_t begin = 9007199254741000ll;
    distant.set(0, begin, 1);
    EXPECT_NEAR(distant.weightAt(begin), 1., 1e-14);
    EXPECT_EQ(distant.weightAt(begin - 240), 0.);
    EXPECT_EQ(distant.weightAt(begin + 720 + 2160), 0.);
    detail::OfflinePunchEnvelope::Cursor cursor(distant, job);
    EXPECT_NEAR(cursor(begin - 127), distant.weightAt(begin - 127), 2e-14);
    for (int size : {1, 2, 31, 4097, 48000})
        for (int position : {0, size - 1})
        {
            AudioBuffer<double> input, output;
            input.resize(1, size);
            input.getChannel(0)[position] = .25;
            const auto result = OfflinePunch<double>().run(input, output, 48000, {6});
            EXPECT_TRUE(result.succeeded());
            OfflineBufferSource<double> source(input.toView(), 48000);
            const auto plan = OfflinePunch<double>().analyze(source, {6});
            EXPECT_TRUE(plan.succeeded());
            EXPECT_NEAR(plan.plan.gainAt(position), std::pow(10., .3), 1e-12);
            const auto reference =
                test::offline_reference::render(input, plan.plan, result.report.renderInfo);
            for (int i = 0; i < size; ++i)
            {
                EXPECT_NEAR(output.getChannel(0)[i], reference.getChannel(0)[i], 1e-12);
                EXPECT_TRUE(std::abs(output.getChannel(0)[i]) <=
                            output.getChannel(0)[position] * (1 + 1e-12));
            }
        }
}
