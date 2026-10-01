// DSPark - Offline peak compression signal and transaction contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Core/ProcessorTraits.h"
#include "../Effects/OfflinePeakCompressor.h"
#include "dspark_test.h"
#include "offline_gain_reference.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflinePeakCompressor<float>, float>);

namespace
{
template <class T> AudioBuffer<T> peakSong(int rate)
{
    AudioBuffer<T> result;
    result.resize(2, 4 * rate + 17);
    for (int i = 0; i < result.getNumSamples(); ++i)
    {
        double x = .002 * std::sin(6.283185307179586 * 67 * i / rate);
        for (int event = 0; event < 3; ++event)
        {
            const int age = i - (event + 1) * rate;
            const double duration = event == 0 ? .008 : event == 1 ? .03 : .09;
            if (age >= 0 && age < duration * rate)
                x += (event == 0   ? 1
                      : event == 1 ? .4
                                   : .35) *
                     std::cos(6.283185307179586 * 1000 * age / rate);
        }
        result.getChannel(0)[i] = static_cast<T>(x);
        result.getChannel(1)[i] = static_cast<T>(-.37 * x);
    }
    return result;
}
template <class T> double maxSample(const AudioBuffer<T> &audio)
{
    double result = 0;
    for (int c = 0; c < audio.getNumChannels(); ++c)
        for (int i = 0; i < audio.getNumSamples(); ++i)
            result = std::max(result, std::abs(static_cast<double>(audio.getChannel(c)[i])));
    return result;
}
template <class T> void amountsAndBounds()
{
    for (int rate : {8000, 48000, 192000})
    {
        auto input = peakSong<T>(rate);
        const double peak = maxSample(input);
        for (double reduction : {0., 3., 6., 12., 18.})
        {
            AudioBuffer<T> output;
            typename OfflinePeakCompressor<T>::Options options;
            options.reductionDb = reduction;
            auto result = OfflinePeakCompressor<T>().run(input, output, rate, options);
            EXPECT_TRUE(result.succeeded());
            EXPECT_NEAR(result.report.achievedReductionDb, result.report.effectiveReductionDb,
                        .001);
            EXPECT_TRUE(result.report.maximumReductionDb <= reduction + 1e-10);
            EXPECT_TRUE(maxSample(output) <=
                        peak * decibelsToGain(-result.report.effectiveReductionDb) * (1 + 2e-7));
            EXPECT_EQ(output.getNumSamples(), input.getNumSamples());
            if (reduction <= 12)
                EXPECT_NEAR(result.report.achievedReductionDb, reduction, .1);
            if (reduction == 18)
            {
                EXPECT_TRUE(result.report.reason == OfflinePeakCompressor<T>::Reason::CrestLimited);
                EXPECT_LT(result.report.achievedReductionDb, 18.);
            }
            for (int i = 0; i < input.getNumSamples(); ++i)
            {
                EXPECT_TRUE(std::isfinite(output.getChannel(0)[i]));
                EXPECT_NEAR(output.getChannel(1)[i], -.37 * output.getChannel(0)[i], 2e-7);
            }
            if (reduction == 0)
                for (int c = 0; c < 2; ++c)
                    EXPECT_EQ(
                        std::memcmp(input.getChannel(c), output.getChannel(c),
                                    static_cast<std::size_t>(input.getNumSamples()) * sizeof(T)),
                        0);
        }
    }
}
class PeakSink final : public OfflineAudioSink<float>
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

DSPARK_TEST(OfflinePeak_IndependentReductionsAndCrestCap_Float)
{
    amountsAndBounds<float>();
}
DSPARK_TEST(OfflinePeak_IndependentReductionsAndCrestCap_Double)
{
    amountsAndBounds<double>();
}

DSPARK_TEST(OfflinePeak_EnvelopeMinimumMatchesIndependentOracle)
{
    OfflineJobOptions options;
    detail::OfflineSession job(options);
    detail::OfflineAttenuation envelope;
    struct Hold
    {
        std::int64_t begin, end;
        double gain;
    };
    const std::array<Hold, 5> holds{
        {{0, 1, .5}, {501, 600, .3}, {609, 611, .7}, {615, 700, .4}, {5000, 5001, .2}}};
    envelope.prepare(job, holds.size(), 16, 640);
    for (std::size_t i = 0; i < holds.size(); ++i)
        envelope.set(i, holds[i].begin, holds[i].end, holds[i].gain);
    detail::OfflineAttenuation::Cursor cursor(envelope, job);
    for (std::int64_t frame = 0; frame < 30000; ++frame)
    {
        double expected = 0;
        constexpr int weights[]{1, 4, 10, 20, 35, 52, 68, 80, 85, 80, 68, 52, 35, 20, 10, 4, 1};
        for (int k = 0; k < 17; ++k)
        {
            const auto rawFrame = frame + k;
            double raw = 1;
            for (const auto &hold : holds)
            {
                double gain = 1;
                if (rawFrame >= hold.begin && rawFrame <= hold.end + 16)
                    gain = hold.gain;
                else if (rawFrame > hold.end + 16)
                    gain = 1 - (1 - hold.gain) *
                                   std::exp(-static_cast<double>(rawFrame - hold.end - 16) / 640);
                raw = std::min(raw, gain);
            }
            expected += raw * weights[k] / 625;
        }
        EXPECT_NEAR(envelope.gainAt(frame), expected, 5e-15);
        EXPECT_NEAR(cursor(frame), expected, 2e-13);
    }
    EXPECT_EQ(envelope.gainAt(30000), 1.);
    // Integer source coordinates remain precise beyond the exact-double range.
    const std::int64_t origin = 9007199254741000ll;
    detail::OfflineAttenuation distant;
    distant.prepare(job, 1, 16, 640);
    distant.set(0, origin, origin + 1, .5);
    EXPECT_EQ(distant.gainAt(origin), .5);
    const double factor =
        std::exp(16. / 640) * std::pow(std::expm1(-5. / 640) / (5 * std::expm1(-1. / 640)), 4);
    const double expected = 1 - .5 / std::exp(1.) * factor;
    EXPECT_NEAR(distant.gainAt(origin + 641), expected, 1e-15);
    EXPECT_EQ(cursor(30000), 1.);
}

DSPARK_TEST(OfflinePeak_FirstLastSamplesAndNoAudioShift)
{
    for (int size : {1, 2, 31, 4097, 96000})
        for (int position : {0, size - 1})
        {
            AudioBuffer<double> input, output;
            input.resize(1, size);
            input.getChannel(0)[position] = 1;
            auto result = OfflinePeakCompressor<double>().run(input, output, 48000, {6});
            EXPECT_TRUE(result.succeeded());
            EXPECT_NEAR(output.getChannel(0)[position], decibelsToGain(-6.), 1e-12);
            OfflineBufferSource<double> source(input.toView(), 48000);
            const auto plan = OfflinePeakCompressor<double>().analyze(
                source, {6, result.report.lookaheadMs, 80});
            EXPECT_TRUE(plan.succeeded());
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

DSPARK_TEST(OfflinePeak_LocalHoldAndReleaseDoNotUseLongestEvent)
{
    constexpr int rate = 48000;
    auto input = peakSong<double>(rate);
    OfflineBufferSource<double> source(input.toView(), rate);
    auto map = OfflineTransientAnalyzer<double>().analyze(source);
    EXPECT_TRUE(map.succeeded());
    auto plan = OfflinePeakCompressor<double>().makePlan(source, map.analysis, {6});
    EXPECT_TRUE(plan.succeeded());
    EXPECT_GT(plan.plan.getReport().transientHolds, std::size_t(0));
    auto first = map.analysis.attacks()[0];
    const double held = plan.plan.gainAt(first.begin);
    EXPECT_NEAR(held, decibelsToGain(-6.), .0001);
    EXPECT_NEAR(plan.plan.gainAt(first.end - 1), held, 1e-14);
    for (int ms : {0, 20, 60, 80, 100, 200})
    {
        const double factor = std::exp(96. / 3840) *
                              std::pow(std::expm1(-25. / 3840) / (25 * std::expm1(-1. / 3840)), 4);
        const double expected = ms == 0 ? held : 1 - (1 - held) * std::exp(-ms / 80.) * factor;
        EXPECT_NEAR(plan.plan.gainAt(first.end + ms * 48), expected, 1e-12);
    }
    EXPECT_LT(plan.plan.gainAt(first.begin - 24), 1.);
    EXPECT_EQ(plan.plan.gainAt(first.begin - 97), 1.);
}

DSPARK_TEST(OfflinePeak_ExclusionsAreExactAndReportProtectedPeaks)
{
    auto input = peakSong<float>(8000);
    input.getChannel(1)[7999] = -0.0f;
    const std::array<OfflineRegion, 3> regions{{{8050, 8200}, {7980, 8051}, {8001, 8050}}};
    OfflinePeakCompressor<float>::Options opt;
    opt.reductionDb = 6;
    opt.exclusions = regions;
    AudioBuffer<float> output;
    const auto result = OfflinePeakCompressor<float>().run(input, output, 8000, opt);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.report.reason == OfflinePeakCompressor<float>::Reason::ProtectedPeaks);
    EXPECT_NEAR(result.report.achievedReductionDb, 0., 1e-6);
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c) + 7980, output.getChannel(c) + 7980,
                              220 * sizeof(float)),
                  0);
    OfflineRegion all{0, input.getNumSamples()};
    opt.exclusions = {&all, 1};
    const auto unchanged = OfflinePeakCompressor<float>().run(input, input, 8000, opt);
    EXPECT_TRUE(unchanged.succeeded());
    EXPECT_TRUE(unchanged.report.reason == OfflinePeakCompressor<float>::Reason::AllExcluded);
}

DSPARK_TEST(OfflinePeak_BlockInvarianceInPlaceAndAnalysisReuse)
{
    auto input = peakSong<double>(8000);
    OfflineBufferSource<double> source(input.toView(), 8000, 19, 7, 9000000000ll);
    auto map = OfflineTransientAnalyzer<double>().analyze(source);
    EXPECT_TRUE(map.succeeded());
    OfflinePeakCompressor<double> processor;
    auto reference = processor.makePlan(source, map.analysis, {6});
    EXPECT_TRUE(reference.succeeded());
    EXPECT_EQ(reference.plan.getSpec().timelineOrigin, 9000000000ll);
    AudioBuffer<double> previous;
    for (int block : {1, 17, 4093, 999999})
    {
        OfflineJobOptions job;
        job.blockFrames = block;
        auto plan = processor.makePlan(source, map.analysis, {6}, job);
        EXPECT_TRUE(plan.succeeded());
        for (int i = 0; i < input.getNumSamples(); ++i)
            EXPECT_EQ(plan.plan.gainAt(i), reference.plan.gainAt(i));
        AudioBuffer<double> output;
        auto rendered = processor.run(input, output, 8000, {6}, job);
        EXPECT_TRUE(rendered.succeeded());
        if (previous.getNumSamples())
            for (int c = 0; c < 2; ++c)
                EXPECT_EQ(
                    std::memcmp(previous.getChannel(c), output.getChannel(c),
                                static_cast<std::size_t>(input.getNumSamples()) * sizeof(double)),
                    0);
        previous = std::move(output);
    }
    EXPECT_TRUE(processor.run(input, input, 8000, {6}).succeeded());
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c), previous.getChannel(c),
                              static_cast<std::size_t>(input.getNumSamples()) * sizeof(double)),
                  0);
    auto moved = std::move(reference.plan);
    EXPECT_FALSE(reference.plan.isValid());
    EXPECT_EQ(reference.plan.gainAt(8000), 1.);
    EXPECT_TRUE(moved.isValid());
}

DSPARK_TEST(OfflinePeak_SustainedCarriersHaveNoCycleGainRipple)
{
    constexpr int rate = 48000;
    for (double frequency : {20., 31.25, 50., 100., 440., 4000., 15000., 20000.})
    {
        AudioBuffer<double> input, output;
        input.resize(1, 3 * rate);
        for (int i = 0; i < input.getNumSamples(); ++i)
            input.getChannel(0)[i] = .9 * std::sin(6.283185307179586 * frequency * i / rate);
        const auto result = OfflinePeakCompressor<double>().run(input, output, rate, {6});
        EXPECT_TRUE(result.succeeded());
        const double expected = decibelsToGain(-result.report.effectiveReductionDb);
        double squaredError = 0, squaredReference = 0;
        OfflineBufferSource<double> source(input.toView(), rate);
        const auto plan = OfflinePeakCompressor<double>().analyze(
            source, {6, result.report.lookaheadMs, 80});
        EXPECT_TRUE(plan.succeeded());
        const auto reference =
            test::offline_reference::render(input, plan.plan, result.report.renderInfo);
        double renderError = 0, renderPower = 0;
        for (int i = 0; i < input.getNumSamples(); ++i)
        {
            renderError += std::pow(output.getChannel(0)[i] - reference.getChannel(0)[i], 2);
            renderPower += reference.getChannel(0)[i] * reference.getChannel(0)[i];
        }
        EXPECT_LT(renderError / renderPower, 1e-20);
        for (int i = rate; i < 2 * rate; ++i)
        {
            const double ref = input.getChannel(0)[i] * expected;
            const double planned = input.getChannel(0)[i] * plan.plan.gainAt(i);
            squaredError += (planned - ref) * (planned - ref);
            squaredReference += ref * ref;
        }
        EXPECT_LT(squaredError / squaredReference, 1e-20);
    }
}

DSPARK_TEST(OfflinePeak_InvalidStaleCancelledAndSinkFailurePreserveOutput)
{
    auto input = peakSong<float>(8000);
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflinePeakCompressor<float> processor;
    auto plan = processor.analyze(source, {6});
    EXPECT_TRUE(plan.succeeded());
    std::atomic<bool> cancel{true};
    OfflineJobOptions job;
    job.cancel = &cancel;
    PeakSink cancelled;
    EXPECT_TRUE(processor.render(source, plan.plan, cancelled, job).status ==
                OfflineStatus::Cancelled);
    EXPECT_FALSE(cancelled.begun);
    PeakSink failure;
    failure.fail = true;
    EXPECT_TRUE(processor.render(source, plan.plan, failure).status == OfflineStatus::SinkFailed);
    EXPECT_TRUE(failure.aborted);
    EXPECT_FALSE(failure.committed);
    input.getChannel(0)[1] += .001f;
    PeakSink stale;
    EXPECT_TRUE(processor.render(source, plan.plan, stale).status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(stale.begun);
    AudioBuffer<float> output;
    output.resize(1, 1);
    output.getChannel(0)[0] = .123f;
    for (double invalid : {-1., 19., std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_TRUE(processor.run(input, output, 8000, {invalid}).status ==
                    OfflineStatus::InvalidInput);
        EXPECT_EQ(output.getChannel(0)[0], .123f);
    }
    input.getChannel(0)[500] = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(processor.run(input, output, 8000, {6}).status == OfflineStatus::NonFiniteInput);
    EXPECT_EQ(output.getChannel(0)[0], .123f);
}

DSPARK_TEST(OfflinePeak_ExtremeFiniteAndMemoryBudget)
{
    for (double scale : {1e-250, 1e250, std::numeric_limits<double>::max()})
    {
        AudioBuffer<double> input, output;
        input.resize(2, 1000);
        input.getChannel(0)[99] = scale;
        input.getChannel(1)[99] = -scale;
        const auto result = OfflinePeakCompressor<double>().run(input, output, 48000, {6});
        EXPECT_TRUE(result.succeeded());
        EXPECT_NEAR(output.getChannel(0)[99] / scale, decibelsToGain(-6.), 1e-14);
        EXPECT_TRUE(std::isfinite(result.report.outputTruePeakDb));
    }
    auto input = peakSong<float>(8000);
    OfflineBufferSource<float> source(input.toView(), 8000);
    auto analysis = OfflineTransientAnalyzer<float>().analyze(source);
    EXPECT_TRUE(analysis.succeeded());
    OfflinePeakCompressor<float> processor;
    auto reference = processor.makePlan(source, analysis.analysis, {6});
    EXPECT_TRUE(reference.succeeded());
    OfflineJobOptions budget;
    budget.memoryBudgetBytes = reference.memoryBytes;
    EXPECT_TRUE(processor.makePlan(source, analysis.analysis, {6}, budget).succeeded());
    --budget.memoryBudgetBytes;
    auto limited = processor.makePlan(source, analysis.analysis, {6}, budget);
    EXPECT_TRUE(limited.status == OfflineStatus::MemoryLimit);
    EXPECT_FALSE(limited.plan.isValid());
}

DSPARK_TEST(OfflinePeak_ModulatedCarriersAndWeakerPlanReplacePriorGain)
{
    constexpr int rate = 48000;
    for (double frequency : {31.25, 440., 11000.})
    {
        AudioBuffer<double> input, output;
        input.resize(2, 2 * rate);
        for (int i = 0; i < input.getNumSamples(); ++i)
        {
            const double t = static_cast<double>(i) / rate;
            const double x =
                (.6 + .2 * std::sin(twoPi<double> * 5 * t)) *
                std::sin(twoPi<double> * frequency * t + .3 * std::sin(twoPi<double> * 6 * t));
            input.getChannel(0)[i] = x;
            input.getChannel(1)[i] = -x;
        }
        OfflineBufferSource<double> source(input.toView(), rate);
        auto map = OfflineTransientAnalyzer<double>().analyze(source);
        EXPECT_TRUE(map.succeeded());
        OfflinePeakCompressor<double> processor;
        auto stronger = processor.makePlan(source, map.analysis, {6});
        EXPECT_TRUE(stronger.succeeded());
        for (double release : {60., 80., 100.})
        {
            OfflinePeakCompressor<double>::Options options;
            options.reductionDb = 1;
            options.releaseMs = release;
            auto weaker = processor.makePlan(source, map.analysis, options);
            EXPECT_TRUE(weaker.succeeded());
            for (int i = 0; i < input.getNumSamples(); ++i)
                EXPECT_TRUE(weaker.plan.gainAt(i) >= decibelsToGain(-1.) - 1e-14);
            auto result = processor.run(input, output, rate, options);
            EXPECT_TRUE(result.succeeded());
            EXPECT_NEAR(result.report.achievedReductionDb, result.report.effectiveReductionDb,
                        .001);
            for (int i = 0; i < input.getNumSamples(); ++i)
                EXPECT_EQ(output.getChannel(0)[i], -output.getChannel(1)[i]);
        }
    }
}

DSPARK_TEST(OfflinePeak_ChangedPlanningSourceAndLateCancellationCannotPublish)
{
    auto input = peakSong<float>(8000);
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflinePeakCompressor<float> processor;
    auto map = OfflineTransientAnalyzer<float>().analyze(source);
    EXPECT_TRUE(map.succeeded());
    const float saved = input.getChannel(0)[10];
    input.getChannel(0)[10] = .04f;
    auto changed = processor.makePlan(source, map.analysis, {6});
    EXPECT_TRUE(changed.status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(changed.plan.isValid());
    input.getChannel(0)[10] = saved;
    auto plan = processor.makePlan(source, map.analysis, {6});
    EXPECT_TRUE(plan.succeeded());
    OfflineJobOptions job;
    job.progress = [](void *, OfflineProgress p) {
        return p.phase != OfflinePhase::Render || p.completed < p.total;
    };
    PeakSink sink;
    EXPECT_TRUE(processor.render(source, plan.plan, sink, job).status == OfflineStatus::Cancelled);
    EXPECT_TRUE(sink.begun && sink.aborted && !sink.committed);
    job.progress = [](void *, OfflineProgress p) -> bool {
        if (p.phase == OfflinePhase::Plan && p.completed > 0)
            throw 1;
        return true;
    };
    auto failed = processor.makePlan(source, map.analysis, {6}, job);
    EXPECT_TRUE(failed.status == OfflineStatus::CallbackFailed);
    EXPECT_FALSE(failed.plan.isValid());
}

DSPARK_TEST(OfflinePeak_BoxKernelRaggedSpansAndCursorSeeks)
{
    for (int span : {1, 2, 3, 4, 5, 7, 8, 15, 17, 19, 31, 32, 95, 97, 959, 3840})
    {
        OfflineJobOptions options;
        detail::OfflineSession job(options);
        detail::OfflineAttenuation curve;
        curve.prepare(job, 4, span, 640);
        constexpr std::int64_t begins[]{0, 11, 19, 6000}, ends[]{1, 15, 25, 6001};
        constexpr double gains[]{.7, .2, .4, .3};
        for (std::size_t i = 0; i < 4; ++i)
            curve.set(i, begins[i], ends[i], gains[i]);
        std::vector<long double> kernel(1, 1);
        for (int stage = 0; stage < 4; ++stage)
        {
            const int length = span / 4 + (stage < span % 4) + 1;
            std::vector<long double> product(kernel.size() + static_cast<std::size_t>(length) - 1);
            for (std::size_t i = 0; i < kernel.size(); ++i)
                for (int j = 0; j < length; ++j)
                    product[i + static_cast<std::size_t>(j)] += kernel[i] / length;
            kernel = std::move(product);
        }
        detail::OfflineAttenuation::Cursor cursor(curve, job);
        for (std::int64_t frame :
             {0, 1, 9, 11, 15, 19, 25, 29, 33, 39, 5989, 6000, 6001, 7000, 30000, 17})
        {
            long double expected = 0;
            for (int k = 0; k <= span; ++k)
            {
                const auto rawFrame = frame + k;
                long double gain = 1;
                for (int i = 0; i < 4; ++i)
                {
                    if (rawFrame < begins[i])
                        continue;
                    const long double time = static_cast<long double>(std::max(
                                                 std::int64_t(0), rawFrame - ends[i] - span)) /
                                             640;
                    gain = std::min(gain,
                                    1 - (1 - static_cast<long double>(gains[i])) * std::exp(-time));
                }
                expected += gain * kernel[static_cast<std::size_t>(k)];
            }
            EXPECT_NEAR(curve.gainAt(frame), static_cast<double>(expected), 3e-14);
            EXPECT_NEAR(cursor(frame), static_cast<double>(expected), 2e-13);
        }
    }
}

DSPARK_TEST(OfflineExclusions_OverlappingFeathersHaveSmoothCentre)
{
    OfflineJobOptions options;
    detail::OfflineSession job(options);
    detail::OfflineExclusions exclusions;
    constexpr OfflineRegion regions[]{{0, 8000}, {9024, 15000}};
    exclusions.assign(job, regions, 20000);
    for (double gain : {.125, 16.})
    {
        constexpr std::int64_t middle = 8512;
        const auto at = [&](std::int64_t frame) { return exclusions.apply(frame, gain, 1024); };
        EXPECT_EQ(at(7999), 1.);
        EXPECT_EQ(at(8000), 1.);
        EXPECT_EQ(at(9024), 1.);
        EXPECT_EQ(at(16024), gain);
        EXPECT_NEAR(at(middle), 1 + (gain - 1) * .25, 1e-15);
        const double second = std::abs(at(middle - 1) - 2 * at(middle) + at(middle + 1));
        EXPECT_LT(second * 1024 * 1024 / std::abs(gain - 1), 12.);
        for (int frame = 8000; frame <= 9024; ++frame)
        {
            const double value = at(frame);
            EXPECT_TRUE(value >= std::min(1., gain) && value <= std::max(1., gain));
            EXPECT_NEAR(value, at(17024 - frame), 2e-14);
        }
    }
}

DSPARK_TEST(OfflinePeak_SubnormalResolutionCannotSilentlyOverReduce)
{
    const auto check = []<class T>() {
        const T unit = std::numeric_limits<T>::denorm_min();
        for (int steps : {1, 2, 3, 7, 17, 1000})
            for (double reduction : {6., 12.})
            {
                AudioBuffer<T> input, output;
                input.resize(1, 31);
                input.getChannel(0)[7] = T(steps) * unit;
                const auto result =
                    OfflinePeakCompressor<T>().run(input, output, 48000, {reduction});
                EXPECT_TRUE(result.succeeded());
                const int targetSteps =
                    static_cast<int>(std::ceil(steps * decibelsToGain(-reduction)));
                const double expected = 20 * std::log10(static_cast<double>(steps) / targetSteps);
                EXPECT_EQ(output.getChannel(0)[7], T(targetSteps) * unit);
                EXPECT_NEAR(result.report.effectiveReductionDb, expected, 1e-10);
                EXPECT_NEAR(result.report.achievedReductionDb, expected, 1e-10);
                EXPECT_TRUE(result.report.achievedReductionDb <= reduction);
                EXPECT_TRUE(result.report.reason ==
                            OfflinePeakCompressor<T>::Reason::ResolutionLimited);
                if (targetSteps == steps)
                    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
            }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}
