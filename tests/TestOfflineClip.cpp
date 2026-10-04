// DSPark - Offline clipping signal, provenance and transactional contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Core/FFT.h"
#include "../Core/ProcessorTraits.h"
#include "../Effects/OfflineHardClipper.h"
#include "../Effects/OfflineSoftClipper.h"
#include "dspark_test.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflineHardClipper<float>, float>);
static_assert(!AudioProcessor<OfflineSoftClipper<double>, double>);

namespace
{
template <class T> double clipPeak(const AudioBuffer<T> &audio)
{
    double peak = 0;
    for (int c = 0; c < audio.getNumChannels(); ++c)
        for (int i = 0; i < audio.getNumSamples(); ++i)
            peak = std::max(peak, std::abs(static_cast<double>(audio.getChannel(c)[i])));
    return peak;
}
template <class T> AudioBuffer<T> clipFixture(int channels = 2, int frames = 1025)
{
    AudioBuffer<T> audio;
    audio.resize(channels, frames);
    for (int i = 0; i < frames; ++i)
        for (int c = 0; c < channels; ++c)
            audio.getChannel(c)[i] =
                static_cast<T>((c ? -1 : 1) * (.62 * std::cos(.4 * twoPi<double> * i) +
                                               .38 * std::cos(.07 * twoPi<double> * i)));
    return audio;
}
template <class Processor, class T> void clipAmounts()
{
    auto input = clipFixture<T>();
    const double peak = clipPeak(input);
    for (double reduction : {.001, .1, 3., 6., 12.})
    {
        typename Processor::Options options;
        options.reductionDb = reduction;
        AudioBuffer<T> output;
        auto result = Processor().run(input, output, 48000, options);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(result.report.targetMet);
        EXPECT_EQ(output.getNumSamples(), input.getNumSamples());
        EXPECT_NEAR(20 * std::log10(peak / clipPeak(output)), reduction, .005);
        EXPECT_NEAR(result.report.achievedReductionDb, 20 * std::log10(peak / clipPeak(output)),
                    1e-12);
        EXPECT_GT(result.report.calibrationPasses, 0);
        EXPECT_TRUE(result.report.calibrationPasses <= 24);
        EXPECT_EQ(result.report.compensatedLatencyFrames, 0);
        for (int i = 0; i < output.getNumSamples(); ++i)
            EXPECT_NEAR(output.getChannel(1)[i], -output.getChannel(0)[i], 1e-12);
        EXPECT_TRUE(std::isfinite(result.report.outputTruePeakDb));
    }
}
template <class T> class ClipSource final : public OfflineAudioSource<T>
{
  public:
    AudioBuffer<T> audio = clipFixture<T>();
    OfflineAudioSpec spec{48000, 2, 1025, 123, 7, 9007199254741000ll};
    bool fail = false, throws = false;
    std::int64_t reads = 0;
    int maxRead = 0;
    OfflineAudioSpec getSpec() const noexcept override
    {
        return spec;
    }
    bool read(std::int64_t first, AudioBufferView<T> output) override
    {
        if (throws)
            throw std::runtime_error("source");
        if (fail)
            return false;
        reads += output.getNumSamples();
        maxRead = std::max(maxRead, output.getNumSamples());
        for (int c = 0; c < output.getNumChannels(); ++c)
            std::copy_n(audio.getChannel(c) + first, output.getNumSamples(), output.getChannel(c));
        return true;
    }
};
template <class T> class ClipSink final : public OfflineAudioSink<T>
{
  public:
    bool begun = false, committed = false, aborted = false, throws = false;
    int fail = 0;
    std::int64_t written = 0;
    OfflineAudioSpec spec;
    AudioBuffer<T> output;
    bool response(int call)
    {
        if (fail != call)
            return true;
        if (throws)
            throw std::runtime_error("sink");
        return false;
    }
    bool begin(const OfflineAudioSpec &s) override
    {
        begun = true;
        spec = s;
        output.resize(s.channels, static_cast<int>(s.frames));
        return response(1);
    }
    bool write(std::int64_t first, AudioBufferView<const T> block) override
    {
        if (first != written || block.getNumSamples() > spec.frames - first)
            return false;
        for (int c = 0; c < block.getNumChannels(); ++c)
            std::copy_n(block.getChannel(c), block.getNumSamples(), output.getChannel(c) + first);
        written += block.getNumSamples();
        return response(2);
    }
    bool commit() override
    {
        if (written != spec.frames || !response(3))
            return false;
        committed = true;
        return true;
    }
    void abort() noexcept override
    {
        aborted = true;
    }
};
} // namespace

DSPARK_TEST(OfflineClip_HardActualReduction_Float)
{
    clipAmounts<OfflineHardClipper<float>, float>();
}
DSPARK_TEST(OfflineClip_HardActualReduction_Double)
{
    clipAmounts<OfflineHardClipper<double>, double>();
}
DSPARK_TEST(OfflineClip_SoftActualReduction_Float)
{
    clipAmounts<OfflineSoftClipper<float>, float>();
}
DSPARK_TEST(OfflineClip_SoftActualReduction_Double)
{
    clipAmounts<OfflineSoftClipper<double>, double>();
}
namespace
{
template <class T> void clipSmallReductions()
{
    for (double rate : {8000., 48000.})
    {
        AudioBuffer<T> input, output;
        input.resize(2, 257);
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 257; ++i)
                input.getChannel(c)[i] = static_cast<T>(
                    (c ? .55 : .8) * std::cos(twoPi<double> * 20 * i / rate + .3 * c));
        const auto exercise = [&](const auto &processor, auto options)
        {
            for (double amount : {1e-6, .001, .1})
            {
                options.reductionDb = amount;
                const auto result = processor.run(input, output, rate, options);
                if (!result.succeeded())
                {
                    std::cerr << "Small-reduction calibration: rate=" << rate
                              << " requested=" << amount
                              << " effective=" << result.report.effectiveReductionDb
                              << " error=" << result.report.targetErrorDb
                              << " passes=" << result.report.calibrationPasses;
                    if constexpr (requires { options.curve; })
                        std::cerr << " soft-curve=" << static_cast<int>(options.curve);
                    std::cerr << '\n';
                }
                EXPECT_TRUE(result.succeeded());
                EXPECT_TRUE(result.report.targetMet);
                const double measured = 20 * std::log10(clipPeak(input) / clipPeak(output));
                const double effective = result.report.effectiveReductionDb;
                EXPECT_GT(effective, 0.);
                EXPECT_GT(measured, 0.);
                EXPECT_NEAR(measured, effective, std::min(.005, effective / 2) + 1e-12);
                EXPECT_NEAR(result.report.achievedReductionDb, measured, 1e-12);
            }
            // A protected peak cannot satisfy a representable positive request,
            // even when the request is smaller than the ordinary 0.005 dB bound.
            const std::array<OfflineRegion, 1> regions{{{0, 257}}};
            options.exclusions = regions;
            options.reductionDb = .001;
            const auto protectedResult = processor.run(input, output, rate, options);
            EXPECT_TRUE(protectedResult.status == OfflineStatus::NoChange);
            EXPECT_FALSE(protectedResult.report.targetMet);
            for (int c = 0; c < 2; ++c)
                EXPECT_EQ(std::memcmp(input.getChannel(c), output.getChannel(c), 257 * sizeof(T)),
                          0);
        };
        exercise(OfflineHardClipper<T>(), typename OfflineHardClipper<T>::Options{});
        for (auto curve : {OfflineSoftClipCurve::Sine, OfflineSoftClipCurve::Tanh,
                           OfflineSoftClipCurve::GoldenRatio})
        {
            typename OfflineSoftClipper<T>::Options options;
            options.curve = curve;
            exercise(OfflineSoftClipper<T>(), options);
        }
    }
}
} // namespace
DSPARK_TEST(OfflineClip_SmallReductionsAndProtectedPeaks_Float)
{
    clipSmallReductions<float>();
}
DSPARK_TEST(OfflineClip_SmallReductionsAndProtectedPeaks_Double)
{
    clipSmallReductions<double>();
}
DSPARK_TEST(OfflineClip_SoftCurvesAndExplicitFactors)
{
    auto input = clipFixture<double>(1, 31);
    for (auto curve : {OfflineSoftClipCurve::Sine, OfflineSoftClipCurve::Tanh,
                       OfflineSoftClipCurve::GoldenRatio})
        for (int factor : {1, 2, 4, 8, 16})
        {
            OfflineSoftClipper<double>::Options options;
            options.curve = curve;
            options.oversamplingFactor = factor;
            options.reductionDb = 6;
            AudioBuffer<double> output;
            const auto result = OfflineSoftClipper<double>().run(input, output, 8000, options);
            EXPECT_TRUE(result.succeeded());
            EXPECT_TRUE(result.report.targetMet);
            EXPECT_NEAR(-20 * std::log10(clipPeak(output)), 6., .005);
            EXPECT_EQ(result.report.oversamplingFactor, factor);
        }
}
DSPARK_TEST(OfflineClip_ZeroSilenceAndAliasedOwningOutput)
{
    auto input = clipFixture<float>();
    input.getChannel(0)[5] = -0.f;
    AudioBuffer<float> output;
    const auto neutral = OfflineSoftClipper<float>().run(input, output, 48000);
    EXPECT_TRUE(neutral.status == OfflineStatus::NoChange);
    EXPECT_EQ(neutral.report.calibrationPasses, 0);
    EXPECT_EQ(neutral.report.compensatedLatencyFrames, 0);
    OfflineSoftClipper<float>::Options tiny;
    tiny.reductionDb = 1e-9;
    const auto rounded = OfflineSoftClipper<float>().run(input, output, 48000, tiny);
    EXPECT_TRUE(rounded.status == OfflineStatus::NoChange);
    EXPECT_TRUE(rounded.report.representabilityLimited);
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c), output.getChannel(c), 1025 * sizeof(float)), 0);
    OfflineHardClipper<float>::Options options;
    options.reductionDb = 6;
    const auto aliased = OfflineHardClipper<float>().run(input, input, 48000, options);
    EXPECT_TRUE(aliased.succeeded());
    EXPECT_NEAR(-20 * std::log10(clipPeak(input)), 6., .005);
    input.clear();
    const auto silence = OfflineHardClipper<float>().run(input, output, 48000, options);
    EXPECT_TRUE(silence.status == OfflineStatus::NoChange);
    EXPECT_TRUE(silence.report.reason == OfflineHardClipper<float>::Reason::NoSignal);
    EXPECT_EQ(clipPeak(output), 0.);
}
DSPARK_TEST(OfflineClip_ShortFilesAndSourceClockEndpoints)
{
    for (int frames : {1, 2, 31, 257})
        for (int location : {0, frames - 1})
        {
            AudioBuffer<double> input, output;
            input.resize(1, frames);
            input.getChannel(0)[location] = 1;
            OfflineHardClipper<double>::Options options;
            options.reductionDb = 12;
            const auto result = OfflineHardClipper<double>().run(input, output, 192000, options);
            EXPECT_TRUE(result.succeeded());
            EXPECT_NEAR(result.report.achievedReductionDb, 12., .005);
            EXPECT_EQ(output.getNumSamples(), frames);
            const int maximum = static_cast<int>(
                std::max_element(output.getChannel(0), output.getChannel(0) + frames,
                                 [](double a, double b) { return std::abs(a) < std::abs(b); }) -
                output.getChannel(0));
            EXPECT_EQ(maximum, location);
        }
}
DSPARK_TEST(OfflineClip_NyquistHasNoResidualPeakFloor)
{
    AudioBuffer<double> input, output;
    input.resize(1, 8192);
    for (int i = 0; i < 8192; ++i)
        input.getChannel(0)[i] = i % 2 ? 1 : -1;
    OfflineHardClipper<double>::Options options;
    options.reductionDb = 12;
    const auto result = OfflineHardClipper<double>().run(input, output, 48000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_NEAR(20 * std::log10(1 / clipPeak(output)), 12., .005);
}
DSPARK_TEST(OfflineClip_ProtectedPcmAndConstraintFailure)
{
    AudioBuffer<double> input, output;
    input.resize(1, 12000);
    for (int i = 0; i < 12000; ++i)
        input.getChannel(0)[i] = (i < 4000 ? .05 : 1) * std::cos(.4 * twoPi<double> * i);
    const std::array<OfflineRegion, 2> regions{{{120, 240}, {1, 150}}};
    OfflineSoftClipper<double>::Options options;
    options.reductionDb = 6;
    options.exclusions = regions;
    auto result = OfflineSoftClipper<double>().run(input, output, 48000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_NEAR(result.report.achievedReductionDb, 6., .005);
    EXPECT_EQ(std::memcmp(input.getChannel(0) + 1, output.getChannel(0) + 1, 239 * sizeof(double)),
              0);
    // A protected global maximum makes global reduction impossible. Do not trim it.
    const OfflineRegion protectedMaximum{4000, 4100};
    options.exclusions = {&protectedMaximum, 1};
    const double previous = output.getChannel(0)[0];
    result = OfflineSoftClipper<double>().run(input, output, 48000, options);
    EXPECT_TRUE(result.status == OfflineStatus::TargetUnreachable);
    EXPECT_TRUE(result.report.reason == OfflineSoftClipper<double>::Reason::ConstraintLimited);
    EXPECT_EQ(output.getChannel(0)[0], previous);
    const OfflineRegion all{0, 12000};
    options.exclusions = {&all, 1};
    result = OfflineSoftClipper<double>().run(input, output, 48000, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_FALSE(result.report.targetMet);
    EXPECT_EQ(std::memcmp(input.getChannel(0), output.getChannel(0), 12000 * sizeof(double)), 0);
}
DSPARK_TEST(OfflineClip_PlanMoveReuseAndBlockDivision)
{
    ClipSource<float> source;
    OfflineSoftClipper<float> processor;
    OfflineSoftClipper<float>::Options options;
    options.reductionDb = 6;
    std::array<OfflineRegion, 1> regions{{{5, 19}}};
    options.exclusions = regions;
    auto analysis = OfflineEnergyAnalyzer<float>().analyze(source);
    auto prepared = processor.makePlan(source, analysis.analysis, options);
    EXPECT_TRUE(prepared.succeeded());
    regions[0] = {900, 910};
    auto moved = std::move(prepared.plan);
    EXPECT_FALSE(prepared.plan.isValid());
    EXPECT_TRUE(prepared.plan.exclusions().empty());
    EXPECT_EQ(prepared.plan.retainedBytes(), std::size_t(0));
    EXPECT_EQ(moved.exclusions()[0].begin, std::int64_t(5));
    // Replan without constraints for the block-division comparison.
    options.exclusions = {};
    prepared = processor.makePlan(source, analysis.analysis, options);
    ClipSink<float> first, second;
    OfflineJobOptions job;
    job.blockFrames = 127;
    auto result = processor.render(source, prepared.plan, first, job);
    EXPECT_TRUE(result.succeeded());
    job.blockFrames = 31;
    result = processor.render(source, prepared.plan, second, job);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(first.spec == source.spec && second.spec == source.spec);
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(first.output.getChannel(c), second.output.getChannel(c),
                              1025 * sizeof(float)),
                  0);
    EXPECT_TRUE(source.maxRead <= 1025);
}
DSPARK_TEST(OfflineClip_SourceSpecFingerprintAndReadFailures)
{
    for (int fault = 0; fault < 5; ++fault)
    {
        ClipSource<float> source;
        OfflineHardClipper<float> processor;
        auto prepared = processor.analyze(source);
        EXPECT_TRUE(prepared.succeeded());
        if (fault == 0)
            ++source.spec.revision;
        if (fault == 1)
            source.audio.getChannel(0)[10] *= .5f;
        if (fault == 2)
            source.fail = true;
        if (fault == 3)
            source.throws = true;
        if (fault == 4)
            source.audio.getChannel(0)[10] = std::numeric_limits<float>::quiet_NaN();
        ClipSink<float> sink;
        auto result = processor.render(source, prepared.plan, sink);
        EXPECT_TRUE(result.status == (fault <= 1   ? OfflineStatus::SourceMismatch
                                      : fault <= 3 ? OfflineStatus::SourceReadFailed
                                                   : OfflineStatus::NonFiniteInput));
        EXPECT_FALSE(sink.committed);
        EXPECT_TRUE(!sink.begun || sink.aborted);
    }
}
DSPARK_TEST(OfflineClip_SinkFailureAtEveryPublicationStage)
{
    ClipSource<float> source;
    OfflineHardClipper<float> processor;
    auto plan = processor.analyze(source);
    for (int stage : {1, 2, 3})
        for (bool throws : {false, true})
        {
            ClipSink<float> sink;
            sink.fail = stage;
            sink.throws = throws;
            const auto result = processor.render(source, plan.plan, sink);
            EXPECT_TRUE(result.status == OfflineStatus::SinkFailed);
            EXPECT_TRUE(sink.aborted);
            EXPECT_FALSE(sink.committed);
        }
}
DSPARK_TEST(OfflineClip_CancellationCallbacksAndLateSourceMutation)
{
    ClipSource<float> source;
    OfflineHardClipper<float> processor;
    OfflineHardClipper<float>::Options options;
    options.reductionDb = 6;
    auto plan = processor.analyze(source, options);
    struct Context
    {
        OfflinePhase phase;
        bool throws;
        ClipSource<float> *source;
        bool mutate;
    };
    for (auto phase :
         {OfflinePhase::Analyze, OfflinePhase::Plan, OfflinePhase::Verify, OfflinePhase::Render})
        for (bool throws : {false, true})
        {
            Context context{phase, throws, &source, false};
            OfflineJobOptions job;
            job.progressContext = &context;
            job.progress = [](void *p, OfflineProgress progress)
            {
                const auto &c = *static_cast<Context *>(p);
                if (c.phase != progress.phase)
                    return true;
                if (c.throws)
                    throw std::runtime_error("progress");
                return false;
            };
            const auto expected = throws ? OfflineStatus::CallbackFailed : OfflineStatus::Cancelled;
            if (phase == OfflinePhase::Analyze || phase == OfflinePhase::Plan)
                EXPECT_TRUE(processor.analyze(source, options, job).status == expected);
            else
            {
                ClipSink<float> sink;
                EXPECT_TRUE(processor.render(source, plan.plan, sink, job).status == expected);
                EXPECT_FALSE(sink.committed);
                EXPECT_TRUE(!sink.begun || sink.aborted);
            }
        }
    OfflineJobOptions job;
    job.progressContext = &source;
    job.progress = [](void *p, OfflineProgress progress)
    {
        if (progress.phase == OfflinePhase::Render && progress.completed == 0)
            static_cast<ClipSource<float> *>(p)->audio.getChannel(0)[10] *= .5f;
        return true;
    };
    ClipSink<float> sink;
    const auto changed = processor.render(source, plan.plan, sink, job);
    EXPECT_TRUE(changed.status == OfflineStatus::SourceMismatch);
    EXPECT_TRUE(sink.aborted);
    EXPECT_FALSE(sink.committed);
    std::atomic<bool> cancel{true};
    job = {};
    job.cancel = &cancel;
    EXPECT_TRUE(processor.analyze(source, options, job).status == OfflineStatus::Cancelled);
}
DSPARK_TEST(OfflineClip_InvalidOptionsAndBudgetPreserveOutput)
{
    auto input = clipFixture<float>();
    AudioBuffer<float> output;
    output.resize(1, 1);
    output.getChannel(0)[0] = 123;
    OfflineSoftClipper<float> processor;
    for (double amount : {-1., 12.01, std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()})
    {
        OfflineSoftClipper<float>::Options options;
        options.reductionDb = amount;
        EXPECT_TRUE(processor.run(input, output, 48000, options).status ==
                    OfflineStatus::InvalidInput);
    }
    OfflineSoftClipper<float>::Options options;
    options.oversamplingFactor = 3;
    EXPECT_TRUE(processor.run(input, output, 48000, options).status == OfflineStatus::InvalidInput);
    options = {};
    options.curve = static_cast<OfflineSoftClipCurve>(99);
    EXPECT_TRUE(processor.run(input, output, 48000, options).status == OfflineStatus::InvalidInput);
    options = {};
    options.reductionDb = 6;
    OfflineJobOptions job;
    job.memoryBudgetBytes = 1024;
    EXPECT_TRUE(processor.run(input, output, 48000, options, job).status ==
                OfflineStatus::MemoryLimit);
    EXPECT_EQ(output.getNumSamples(), 1);
    EXPECT_EQ(output.getChannel(0)[0], 123.f);
}

namespace
{
template <class T> void clipExtremes()
{
    for (T magnitude :
         {std::numeric_limits<T>::denorm_min(), T(16) * std::numeric_limits<T>::denorm_min(),
          std::numeric_limits<T>::min(), std::numeric_limits<T>::max()})
    {
        AudioBuffer<T> input, output;
        input.resize(1, 31);
        for (int i = 0; i < 31; ++i)
            input.getChannel(0)[i] = i % 2 ? -magnitude : magnitude;
        typename OfflineHardClipper<T>::Options options;
        options.reductionDb = 6;
        const auto result = OfflineHardClipper<T>().run(input, output, 48000, options);
        EXPECT_TRUE(result.succeeded());
        for (int i = 0; i < 31; ++i)
            EXPECT_TRUE(std::isfinite(output.getChannel(0)[i]));
        if (magnitude == std::numeric_limits<T>::denorm_min())
        {
            EXPECT_TRUE(result.status == OfflineStatus::NoChange);
            EXPECT_TRUE(result.report.representabilityLimited);
            EXPECT_EQ(std::memcmp(input.getChannel(0), output.getChannel(0), 31 * sizeof(T)), 0);
        }
        else
            EXPECT_NEAR(result.report.achievedReductionDb, result.report.effectiveReductionDb,
                        .005);
    }
}
} // namespace
DSPARK_TEST(OfflineClip_ExtremeAndSubnormalPcm_Float)
{
    clipExtremes<float>();
}
DSPARK_TEST(OfflineClip_ExtremeAndSubnormalPcm_Double)
{
    clipExtremes<double>();
}

DSPARK_TEST(OfflineClip_DirectMomentsRemainScaledAtTinyCeilings)
{
    // Integrate clamp(t-root,-c,c)*t^k analytically. This is an independent
    // moment oracle, including the sign-function limit as the ceiling shrinks.
    for (double ceiling : {.05, .2, 1e-12, 1e-24, 1e-100, 1e-300})
        for (double root : {.125, .37, .875})
        {
            detail::continuous_clip::Interval<detail::ClipperCurve::Hard, 11, false, false>
                integral(ceiling);
            std::array<double, 12> samples{};
            for (int i = 0; i < 12; ++i)
                samples[i] = i - 5. - root;
            const auto actual = integral(samples);
            const long double ua =
                std::max(-static_cast<long double>(root), -static_cast<long double>(ceiling));
            const long double ub = std::min(1.L - root, static_cast<long double>(ceiling));
            const long double a = root + ua, b = root + ub;
            for (int k = 0; k < 4; ++k)
            {
                const auto power = [k](long double t) { return std::pow(t, k + 1); };
                // Integrate u*(root+u)^k in local coordinates. Subtracting
                // two O(ceiling) antiderivatives would destroy this O(c^3)
                // contribution for tiny ceilings, even in the reference.
                long double linear = 0, binomial = 1;
                for (int j = 0; j <= k; ++j)
                {
                    linear += binomial * std::pow(static_cast<long double>(root), k - j) *
                              (std::pow(ub, j + 2) - std::pow(ua, j + 2)) / (j + 2);
                    binomial *= static_cast<long double>(k - j) / (j + 1);
                }
                const long double expected = ceiling * (1 - power(b) - power(a)) / (k + 1) + linear;
                EXPECT_TRUE(std::isfinite(actual[k]));
                EXPECT_NEAR(actual[k] / ceiling, static_cast<double>(expected / ceiling), 2e-10);
            }
        }
}

DSPARK_TEST(OfflineClip_NativeSineSlopeUsesTheExistingCurve)
{
    constexpr double ceiling = .8;
    constexpr double slope = detail::clipperSmallSignalSlope<detail::ClipperCurve::Sine, double>();
    static_assert(slope < 1 && slope > .9999999);
    detail::continuous_clip::Interval<detail::ClipperCurve::Sine, 11, true> native(ceiling);
    detail::continuous_clip::Interval<detail::ClipperCurve::Sine, 11> original(ceiling);
    for (double sample : {-.02, -1e-6, 0., 1e-6, .02})
    {
        std::array<double, 12> samples{};
        samples.fill(sample);
        const auto a = native(samples), b = original(samples);
        const long double u = static_cast<long double>(sample) / ceiling;
        // Independent evaluation of the documented minimax odd polynomial.
        const long double residual =
            ceiling * u * u * u *
            (-.16666646026660671L +
             u * u *
                 (.0083328727116396326L +
                  u * u * (-.00019799239565814083L + u * u * 2.5871610835732768e-6L)));
        for (int k = 0; k < 4; ++k)
        {
            EXPECT_NEAR(a[k], static_cast<double>(residual / (k + 1)), 2e-16);
            EXPECT_NEAR(b[k] - a[k], (slope - 1) * sample / (k + 1), 2e-16);
        }
    }
}

DSPARK_TEST(OfflineClip_SincPhaseHandlesEndpointShifts)
{
    constexpr int frames = 257, block = 128;
    std::array<double, frames> input{};
    std::array<double, block> output{};
    for (int i = 0; i < frames; ++i)
        input[i] = .3 * std::sin(.217 * i) + .2 * std::cos(.763 * i);
    const OfflineJobOptions options;
    detail::OfflineSession job(options);
    detail::OfflineProductWorkspace work(job, frames, block,
                                         detail::OfflineProductWorkspace::NearFields::External);
    auto reader = [&](std::int64_t first, int count, double *out) {
        std::copy_n(input.data() + first, count, out);
    };
    detail::OfflineHilbertMap map(work, reader);
    for (double shift : {std::nextafter(-1., 0.), -.9375, -.5, 0., .5, .9375,
                          std::nextafter(1., 0.)})
    {
        detail::offline_sinc::SincPhase phase(work, shift);
        for (std::size_t leaf = 0; leaf < work.leaves(); ++leaf)
        {
            phase.evaluate(map, leaf, reader, output.data());
            for (int i = 0; i < block && static_cast<int>(leaf) * block + i < frames; ++i)
            {
                const int frame = static_cast<int>(leaf) * block + i;
                long double expected = 0;
                for (int j = 0; j < frames; ++j)
                {
                    const long double distance = frame + static_cast<long double>(shift) - j;
                    const long double sinc = distance == 0 ? 1.L :
                        std::sin(std::numbers::pi_v<long double> * distance) /
                        (std::numbers::pi_v<long double> * distance);
                    expected += input[j] * sinc;
                }
                EXPECT_NEAR(output[i], static_cast<double>(expected), 3e-12);
            }
        }
    }
}

DSPARK_TEST(OfflineClip_IndependentHardFourierOracle)
{
    const detail::continuous_clip::Interval<detail::ClipperCurve::Hard, 11> reconstruction(1.);
    // Independent Lagrange basis evaluation checks the shared convex-hull
    // bound used by offline callers to account for interpolation tails.
    for (int point = 0; point <= 1024; ++point)
    {
        const long double t = static_cast<long double>(point) / 1024;
        long double norm = 0;
        for (int node = -5; node <= 6; ++node)
        {
            long double basis = 1;
            for (int other = -5; other <= 6; ++other)
                if (node != other)
                    basis *= (t - other) / (node - other);
            norm += std::abs(basis);
        }
        EXPECT_TRUE(norm <= reconstruction.reconstructionBound() + 1e-12);
    }
    // Fourier coefficients from an analytic integral of clamp(cos(t), -c, c).
    // No interpolation, quadrature, filter design or DSPark waveshaper in the oracle.
    // Periodic boundary fixture for the actual reconstructed stage, using ideal
    // periodic interpolation/projection. The worker uses finite zero extension;
    // its independent impulse oracle below tests that different boundary model.
    // Retain the original phase and all converter-transition frequencies.
    for (const auto [period, bin] : std::array<std::array<int, 2>, 6>{
             {{2048, 7}, {2048, 317}, {2048, 919}, {16384, 2730}, {16384, 2731}, {16384, 8191}}})
        for (double ceiling : {.05, .5, 2.})
        {
            OfflineJobOptions options;
            detail::OfflineSession job(options);
            detail::offline_clip::ReconstructedStage<detail::ClipperCurve::Hard> stage(job, 4);
            stage.reset(ceiling);
            constexpr int origin = 38922, factor = 4;
            const int highSize = period * factor;
            std::vector<double> actual(period), expected(period);
            std::vector<double> shaped(period), difference(period), nativeSpectrum(period + 2);
            std::vector<double> highSpectrum(highSize + 2), highDifference(highSize),
                highResidual(highSize), highLinear(highSize);
            FFTReal<double> nativeFft(period), highFft(highSize);
            for (int i = 0; i < period; ++i)
            {
                const double x = std::cos(twoPi<double> * bin * (origin + i) / period);
                shaped[i] = std::clamp(x, -ceiling, ceiling);
                difference[i] = x - shaped[i];
            }
            nativeFft.forward(difference.data(), nativeSpectrum.data());
            for (int i = 0; i < period + 2; ++i)
                highSpectrum[i] = factor * nativeSpectrum[i];
            // Split the native Nyquist coefficient between its two signs.
            highSpectrum[period] *= .5;
            highSpectrum[period + 1] = 0;
            highFft.inverse(highSpectrum.data(), highDifference.data());
            std::array<double, 256> data{}, reference{};
            const int end = highSize + stage.delay();
            for (int first = -128; first < end;)
            {
                const int count = std::min(256, end - first);
                for (int i = 0; i < count; ++i)
                {
                    const int at = first + i;
                    data[i] = std::cos(twoPi<double> * bin *
                                       (origin + static_cast<double>(at) / factor) / period);
                    reference[i] = highDifference[(at % highSize + highSize) % highSize];
                }
                stage.process(data.data(), reference.data(), count,
                              detail::offline_clip::Exterior::Linear);
                for (int i = 0; i < count; ++i)
                {
                    const int at = first + i - stage.delay();
                    if (at >= 0 && at < highSize)
                    {
                        highResidual[at] = data[i];
                        highLinear[at] = reference[i];
                    }
                }
                first += count;
            }
            highFft.forward(highResidual.data(), highSpectrum.data());
            for (int i = 0; i < period + 2; ++i)
                nativeSpectrum[i] = highSpectrum[i] / factor;
            nativeSpectrum[period] *= 2;
            nativeSpectrum[period + 1] = 0;
            nativeFft.inverse(nativeSpectrum.data(), actual.data());
            for (int i = 0; i < period; ++i)
                actual[i] += shaped[i] + highLinear[i * factor];
            for (int n = 1; n * bin < period / 2; n += 2)
            {
                double coefficient = n == 1 ? 1 : 0;
                if (ceiling < 1)
                {
                    const double a = std::acos(ceiling);
                    const double integral = n == 1 ? (halfPi<double> - a) / 2 - std::sin(2 * a) / 4
                                                   : -std::sin((n - 1) * a) / (2 * (n - 1)) -
                                                         std::sin((n + 1) * a) / (2 * (n + 1));
                    coefficient = 4 / pi<double> * (ceiling * std::sin(n * a) / n + integral);
                }
                for (int i = 0; i < period; ++i)
                    expected[i] +=
                        coefficient * std::cos(twoPi<double> * n * bin * (origin + i) / period);
            }
            double error = 0, energy = 0;
            for (int i = 0; i < period; ++i)
            {
                error += (actual[i] - expected[i]) * (actual[i] - expected[i]);
                energy += expected[i] * expected[i];
            }
            EXPECT_LT(10 * std::log10(std::max(error, 1e-300) / energy), -90.);
        }
}

DSPARK_TEST(OfflineClip_FiniteImpulseIndependentIntegral)
{
    // sinc(t) exceeds c only on [-a,a] for these ceilings. Its complete clipped
    // reconstruction is delta[n] + integral((c-sinc(t))*sinc(n-t), -a, a).
    // This source-rate reference uses no DSPark shaping, interpolation or filters.
    constexpr int frames = 65, centre = 32;
    constexpr long double piValue = 3.141592653589793238462643383279502884L;
    const auto sinc = [](long double t)
    {
        if (t == 0)
            return 1.L;
        return std::sin(piValue * t) / (piValue * t);
    };
    AudioBuffer<double> input, output;
    input.resize(1, frames);
    input.getChannel(0)[centre] = 1;
    for (double reduction : {3., 6.})
    {
        OfflineHardClipper<double>::Options options;
        options.reductionDb = reduction;
        const auto result = OfflineHardClipper<double>().run(input, output, 48000, options);
        EXPECT_TRUE(result.succeeded());
        const long double ceiling = result.report.normalizedCeiling;
        EXPECT_GT(ceiling, .218L);
        EXPECT_LT(ceiling, 1.L);
        long double left = 0, right = 1;
        for (int i = 0; i < 80; ++i)
        {
            const long double middle = (left + right) / 2;
            if (sinc(middle) > ceiling)
                left = middle;
            else
                right = middle;
        }
        const long double edge = (left + right) / 2;
        long double error = 0, energy = 0;
        for (int n = 0; n < frames; ++n)
        {
            const auto integral = [&](int steps)
            {
                const long double h = 2 * edge / steps;
                long double sum = 0;
                for (int i = 0; i <= steps; ++i)
                {
                    const long double t = -edge + i * h;
                    const int weight = i == 0 || i == steps ? 1 : (i % 2 == 0 ? 2 : 4);
                    sum += weight * (ceiling - sinc(t)) * sinc(n - centre - t);
                }
                return (n == centre ? 1.L : 0.L) + h * sum / 3;
            };
            const long double expected = integral(4096);
            EXPECT_NEAR(static_cast<double>(expected), static_cast<double>(integral(2048)), 1e-12);
            const long double delta = output.getChannel(0)[n] - expected;
            error += delta * delta;
            energy += expected * expected;
        }
        EXPECT_LT(static_cast<double>(10 * std::log10(error / energy)), -90.);
    }
}

namespace
{
// Independent finite hard-clip reference: locate every clipping interval of a
// direct cardinal-sinc sum, then integrate its correction against sinc(n-t).
// The L1 exterior bound proves the correction is zero outside this domain.
// Grid and Simpson refinements are compared by the caller, not assumed exact.
std::vector<long double> finiteHardReference(const double *input, int frames, long double ceiling,
                                             int grid, int steps)
{
    constexpr long double p = 3.141592653589793238462643383279502884L;
    long double mass = 0;
    std::vector<long double> alternating(frames), result(frames);
    for (int i = 0; i < frames; ++i)
    {
        result[i] = input[i];
        alternating[i] = i % 2 ? -input[i] : input[i];
        mass += std::abs(input[i]);
    }
    const int padding = static_cast<int>(std::ceil(mass / (p * ceiling))) + 4;
    const auto source = [&](long double t)
    {
        const auto index = static_cast<int>(std::round(t));
        if (t == index)
            return index >= 0 && index < frames ? static_cast<long double>(input[index]) : 0.L;
        long double sum = 0;
        for (int i = 0; i < frames; ++i)
            sum += alternating[i] / (t - i);
        return std::sin(p * t) * sum / p;
    };
    std::vector<std::array<long double, 2>> intervals;
    long double start = 0;
    bool inside = false;
    for (int at = -padding * grid; at < (frames - 1 + padding) * grid; ++at)
    {
        const long double first = static_cast<long double>(at) / grid;
        const long double last = static_cast<long double>(at + 1) / grid;
        const bool next = std::abs(source(last)) > ceiling;
        if (inside == next)
            continue;
        long double a = first, b = last;
        for (int iteration = 0; iteration < 64; ++iteration)
        {
            const long double middle = (a + b) / 2;
            if ((std::abs(source(middle)) > ceiling) == inside)
                a = middle;
            else
                b = middle;
        }
        const long double edge = (a + b) / 2;
        if (inside)
            intervals.push_back({start, edge});
        else
            start = edge;
        inside = next;
    }
    if (inside)
        throw std::runtime_error("reference tail is not enclosed");
    for (const auto &interval : intervals)
    {
        const long double h = (interval[1] - interval[0]) / steps;
        const long double sign = source((interval[0] + interval[1]) / 2) > 0 ? 1.L : -1.L;
        for (int k = 0; k <= steps; ++k)
        {
            const long double t = interval[0] + k * h;
            const int weight = k == 0 || k == steps ? 1 : (k % 2 ? 4 : 2);
            const long double value = h * weight * (sign * ceiling - source(t)) / 3;
            const int integer = static_cast<int>(std::round(t));
            if (t == integer)
            {
                if (integer >= 0 && integer < frames)
                    result[integer] += value;
            }
            else
            {
                const long double numerator = value * std::sin(p * t) / p;
                for (int n = 0; n < frames; ++n)
                    result[n] += (n % 2 ? -numerator : numerator) / (t - n);
            }
        }
    }
    return result;
}
} // namespace

DSPARK_TEST(OfflineClip_PublicFiniteNyquistOracle)
{
    constexpr int frames = 65;
    AudioBuffer<double> input, output;
    input.resize(1, frames);
    for (int i = 0; i < frames; ++i)
        input.getChannel(0)[i] = std::cos(twoPi<double> * 8191 * i / 16384);
    OfflineHardClipper<double>::Options options;
    options.reductionDb = 12;
    const auto rendered = OfflineHardClipper<double>().run(input, output, 48000, options);
    EXPECT_TRUE(rendered.succeeded());
    EXPECT_TRUE(rendered.report.targetMet);
    const double ceiling = rendered.report.normalizedCeiling;
    const auto expected = finiteHardReference(input.getChannel(0), frames, ceiling, 64, 1024);
    const auto refined = finiteHardReference(input.getChannel(0), frames, ceiling, 128, 2048);
    long double error = 0, energy = 0;
    for (int i = 0; i < frames; ++i)
    {
        EXPECT_NEAR(static_cast<double>(expected[i]), static_cast<double>(refined[i]), 1e-10);
        const long double delta = output.getChannel(0)[i] - refined[i];
        error += delta * delta;
        energy += refined[i] * refined[i];
    }
    EXPECT_LT(static_cast<double>(10 * std::log10(error / energy)), -90.);
}

DSPARK_TEST(OfflineClip_LongLogicalSourceAndInvalidFormat)
{
    class Source final : public OfflineAudioSource<float>
    {
      public:
        OfflineAudioSpec spec{48000, 1, 2147483905ll, 8, 4, 9007199254741000ll};
        std::int64_t framesRead = 0;
        int largest = 0;
        OfflineAudioSpec getSpec() const noexcept override
        {
            return spec;
        }
        bool read(std::int64_t, AudioBufferView<float> block) override
        {
            framesRead += block.getNumSamples();
            largest = std::max(largest, block.getNumSamples());
            std::fill_n(block.getChannel(0), block.getNumSamples(), .1f);
            return true;
        }
    } source;
    OfflineJobOptions job;
    job.blockFrames = 127;
    job.memoryBudgetBytes = 16 * 1024 * 1024;
    job.progress = [](void *, OfflineProgress progress) { return progress.completed == 0; };
    const auto cancelled = OfflineHardClipper<float>().analyze(source, {}, job);
    EXPECT_TRUE(cancelled.status == OfflineStatus::Cancelled);
    EXPECT_EQ(source.framesRead, std::int64_t(127));
    EXPECT_EQ(source.largest, 127);
    source.spec.frames = std::numeric_limits<std::int64_t>::max();
    EXPECT_TRUE(OfflineHardClipper<float>().analyze(source).status == OfflineStatus::InvalidInput);
    source.spec.timelineOrigin = 0;
    EXPECT_TRUE(OfflineHardClipper<float>().analyze(source, {}, job).status ==
                OfflineStatus::MemoryLimit);
    source.spec.frames = 0;
    EXPECT_TRUE(OfflineHardClipper<float>().analyze(source).status == OfflineStatus::EmptyInput);
    source.spec.frames = 3;
    source.spec.channels = 3;
    EXPECT_TRUE(OfflineHardClipper<float>().analyze(source).status ==
                OfflineStatus::UnsupportedLayout);
}
