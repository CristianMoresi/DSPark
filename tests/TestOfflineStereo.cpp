// DSPark - Offline stereo transport, provenance and transaction contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Core/ProcessorTraits.h"
#include "../Effects/OfflineStereoGenerator.h"
#include "../Effects/OfflineStereoBalance.h"
#include "dspark_test.h"
#include "offline_gain_reference.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflineStereoGenerator<double>, double>);
static_assert(!AudioProcessor<OfflineStereoBalance<double>, double>);

namespace
{
template <class T> AudioBuffer<T> stereoFixture(int channels = 2, int frames = 1025)
{
    AudioBuffer<T> result;
    result.resize(channels, frames);
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < frames; ++i)
            result.getChannel(c)[i] =
                i % 17 == 0
                    ? -T(0)
                    : static_cast<T>(.5 * std::sin(.13 * i + c) + .2 * std::cos(.91 * i - c));
    return result;
}

template <class T> bool exact(const AudioBuffer<T> &a, const AudioBuffer<T> &b)
{
    if (a.getNumChannels() != b.getNumChannels() || a.getNumSamples() != b.getNumSamples())
        return false;
    for (int c = 0; c < a.getNumChannels(); ++c)
        if (std::memcmp(a.getChannel(c), b.getChannel(c),
                        static_cast<std::size_t>(a.getNumSamples()) * sizeof(T)) != 0)
            return false;
    return true;
}

template <class T> class StereoSource final : public OfflineAudioSource<T>
{
  public:
    explicit StereoSource(int channels = 2, int frames = 1025)
        : audio(stereoFixture<T>(channels, frames))
    {
        spec.channels = channels;
        spec.frames = frames;
    }
    AudioBuffer<T> audio;
    OfflineAudioSpec spec{48000, 2, 1025, 157, 3, 9007199254741041ll};
    std::int64_t readFrames = 0;
    int maxRead = 0, readCalls = 0, failCall = 0, mutateCall = 0;
    bool throws = false;
    OfflineAudioSpec getSpec() const noexcept override
    {
        return spec;
    }
    bool read(std::int64_t first, AudioBufferView<T> output) override
    {
        ++readCalls;
        if (readCalls == failCall)
        {
            if (throws)
                throw std::runtime_error("source");
            return false;
        }
        if (readCalls == mutateCall)
            ++spec.revision;
        readFrames += output.getNumSamples();
        maxRead = std::max(maxRead, output.getNumSamples());
        for (int c = 0; c < output.getNumChannels(); ++c)
            std::copy_n(audio.getChannel(c) + first, output.getNumSamples(), output.getChannel(c));
        return true;
    }
};

template <class T> class StereoSink final : public OfflineAudioSink<T>
{
  public:
    OfflineAudioSpec spec;
    AudioBuffer<T> output;
    std::int64_t written = 0;
    bool begun = false, committed = false, aborted = false, throws = false;
    int fail = 0;
    bool answer(int stage)
    {
        if (fail != stage)
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
        return answer(1);
    }
    bool write(std::int64_t first, AudioBufferView<const T> block) override
    {
        if (first != written || block.getNumSamples() > spec.frames - first ||
            block.getNumChannels() != spec.channels)
            return false;
        for (int c = 0; c < spec.channels; ++c)
            std::copy_n(block.getChannel(c), block.getNumSamples(), output.getChannel(c) + first);
        written += block.getNumSamples();
        return answer(2);
    }
    bool commit() override
    {
        if (written != spec.frames || !answer(3))
            return false;
        committed = true;
        return true;
    }
    void abort() noexcept override
    {
        aborted = true;
    }
};

using Processor = OfflineStereoGenerator<double>;

// Exercise the adapter against the separately tested continuous core. Feed and
// flush it directly, then strip its stated latency without using offline helpers.
template <class T>
void compareCore(double rate, int factor, float cut, int channels, int frames, int block,
                 std::int64_t origin)
{
    StereoSource<T> source(channels, frames);
    source.audio.getChannel(0)[0] = T(.125); // Keep the one-frame fixture active.
    source.spec.sampleRate = rate;
    source.spec.timelineOrigin = origin;
    typename OfflineStereoGenerator<T>::Options options;
    options.width = 1;
    options.oversamplingFactor = factor;
    options.lowCutHz = cut;
    options.duplicateMono = channels == 1;
    OfflineJobOptions job;
    job.blockFrames = block;
    OfflineStereoGenerator<T> processor;
    const auto plan = processor.analyze(source, options, job);
    EXPECT_TRUE(plan.succeeded());
    StereoSink<T> sink;
    const auto result = processor.render(source, plan.plan, sink, job);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(sink.committed && !sink.aborted);
    EXPECT_EQ(source.readFrames, 2ll * frames);
    EXPECT_TRUE(source.maxRead <= block);
    EXPECT_EQ(sink.spec.channels, 2);
    EXPECT_EQ(sink.spec.timelineOrigin, origin);
    StereoGenerator<double> core;
    core.setWidth(1);
    EXPECT_TRUE(core.prepare({rate, 127, 2}, {factor, cut}));
    core.resetAtFrame(static_cast<std::uint64_t>(origin));
    const int latency = core.getLatency();
    EXPECT_EQ(result.report.compensatedLatencyFrames, latency);
    AudioBuffer<double> input;
    input.resize(2, 127);
    for (int first = 0; first < frames + latency;)
    {
        const int count = std::min(127, frames + latency - first);
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < count; ++i)
                input.getChannel(c)[i] =
                    first + i < frames ? source.audio.getChannel(c % channels)[first + i] : 0;
        EXPECT_TRUE(core.processBlock(input.toView().getSubView(0, count)));
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < count; ++i)
                if (first + i >= latency)
                    EXPECT_EQ(sink.output.getChannel(c)[first + i - latency],
                              static_cast<T>(input.getChannel(c)[i]));
        first += count;
    }
}

template <class T> void neutralAndOwning()
{
    using Effect = OfflineStereoGenerator<T>;
    auto input = stereoFixture<T>();
    AudioBuffer<T> output;
    auto result = Effect().run(input, output, 48000);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(exact(input, output));
    EXPECT_TRUE(result.report.peaksMeasured);
    EXPECT_EQ(result.report.compensatedLatencyFrames, 0);
    typename Effect::Options options;
    options.width = 1;
    const OfflineRegion region{0, input.getNumSamples()};
    options.exclusions = {&region, 1};
    result = Effect().run(input, input, 48000, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(result.report.reason == Effect::Reason::AllExcluded);
    EXPECT_TRUE(exact(input, output));
    options.exclusions = {};
    EXPECT_TRUE(Effect().run(input, output, 48000, options).succeeded());
    EXPECT_TRUE(Effect().run(input, input, 48000, options).succeeded());
    EXPECT_TRUE(exact(input, output));
    auto mono = stereoFixture<T>(1);
    const auto monoCopy = stereoFixture<T>(1);
    EXPECT_TRUE(Effect().run(mono, output, 48000).status == OfflineStatus::UnsupportedLayout);
    options.width = 0;
    options.duplicateMono = true;
    EXPECT_TRUE(Effect().run(mono, mono, 48000, options).succeeded());
    EXPECT_EQ(mono.getNumChannels(), 2);
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(mono.getChannel(c), monoCopy.getChannel(0), 1025 * sizeof(T)), 0);
    AudioBuffer<T, 1> limited;
    limited.resize(1, 1);
    limited.getChannel(0)[0] = T(.125);
    EXPECT_TRUE(Effect().run(limited, limited, 48000, options).status ==
                OfflineStatus::UnsupportedLayout);
    EXPECT_EQ(limited.getChannel(0)[0], T(.125));
}
} // namespace

DSPARK_TEST(OfflineStereo_matches_continuous_core_and_source_clock)
{
    for (const auto rate : {8000., 48000., 96000., 192000., 384000.})
        for (int factor : {1, 2, 4, 8, 16})
            compareCore<double>(rate, factor, 0, 2, 1025, 127, 9007199254741041ll);
    compareCore<float>(44100, 4, 175, 2, 8193, 251, 63);
    compareCore<float>(44100, 1, 175, 2, 8193, 251, 63);
    compareCore<float>(96000, 8, 5000, 1, 513, 1, 17);
    compareCore<double>(8000, 2, 5000, 2, 7, 3, 19);
}

DSPARK_TEST(OfflineStereo_short_sources_flush_once_and_clock_never_wraps)
{
    for (int size : {1, 2, 17, 257})
        for (int block : {1, 127, 4096})
        for (int factor : {1, 4})
            compareCore<double>(48000, factor, 175, 1, size, block,
                                std::numeric_limits<std::int64_t>::max() - size);
}

DSPARK_TEST(OfflineStereo_neutral_protected_and_owning_alias_preserve_native_pcm)
{
    neutralAndOwning<float>();
    neutralAndOwning<double>();
}

DSPARK_TEST(OfflineStereo_exclusions_feather_delta_and_preserve_mid)
{
    StereoSource<double> source(2, 4097);
    Processor effect;
    Processor::Options options;
    options.width = 1;
    const auto base = effect.analyze(source, options);
    StereoSink<double> full, protectedSink;
    EXPECT_TRUE(effect.render(source, base.plan, full).succeeded());
    const std::array<OfflineRegion, 2> regions{{{900, 1000}, {1200, 1250}}};
    options.width = .375f;
    options.exclusions = regions;
    const auto plan = effect.replan(source, base.plan, options);
    EXPECT_EQ(plan.plan.exclusions().size(), 2u);
    EXPECT_TRUE(effect.render(source, plan.plan, protectedSink).succeeded());
    double changed = 0;
    const auto smooth = [](double t) {
        const double u = std::clamp(t, 0., 1.);
        return u * u * u * u * (35 - 84 * u + 70 * u * u - 20 * u * u * u);
    };
    for (int i = 0; i < 4097; ++i)
    {
        double mask = 1;
        for (const auto region : regions)
            mask *= i >= region.begin && i < region.end
                        ? 0
                        : smooth(static_cast<double>(i < region.begin ? region.begin - i
                                                                      : i - region.end) /
                                 240);
        for (int c = 0; c < 2; ++c)
        {
            const double raw = source.audio.getChannel(c)[i];
            const double expected = raw + (full.output.getChannel(c)[i] - raw) * .375 * mask;
            EXPECT_NEAR(protectedSink.output.getChannel(c)[i], expected, 3e-15);
            if (mask == 0)
                EXPECT_EQ(std::memcmp(&raw, protectedSink.output.getChannel(c) + i, sizeof(double)),
                          0);
            changed = std::max(changed, std::abs(protectedSink.output.getChannel(c)[i] - raw));
        }
        EXPECT_NEAR(protectedSink.output.getChannel(0)[i] + protectedSink.output.getChannel(1)[i],
                    source.audio.getChannel(0)[i] + source.audio.getChannel(1)[i], 4e-16);
    }
    EXPECT_GT(changed, .001);
}

DSPARK_TEST(OfflineStereo_cache_reuses_canonical_delta_for_width_and_exclusions)
{
    for (int channels : {1, 2})
    {
        StereoSource<float> source(channels);
        OfflineStereoGenerator<float> effect;
        OfflineStereoGenerator<float>::Options options;
        options.duplicateMono = channels == 1;
        options.lowCutHz = 175;
        auto base = effect.analyze(source, options);
        StereoSink<double> delta;
        auto cached = effect.buildDeltaCache(source, base.plan, delta);
        EXPECT_TRUE(cached.succeeded() && cached.cache.isValid());
        EXPECT_EQ(delta.spec.channels, 1);
        EXPECT_TRUE(cached.cache.getSourceSpec() == source.spec);
        auto certificate = std::move(cached.cache);
        EXPECT_FALSE(cached.cache.isValid());
        OfflineBufferSource<double> cacheSource(delta.output.toView(), delta.spec.sampleRate,
                                                delta.spec.sourceId, delta.spec.revision,
                                                delta.spec.timelineOrigin);
        const OfflineRegion region{217, 341};
        for (float width : {0.f, .375f, 1.f})
        {
            options.width = width;
            options.exclusions = {&region, 1};
            const auto before = source.readFrames;
            auto plan = effect.replan(source, base.plan, options);
            EXPECT_EQ(source.readFrames, before);
            StereoSink<float> direct, replay;
            OfflineJobOptions job;
            job.blockFrames = 7;
            const auto result = effect.render(source, plan.plan, direct, job);
            job.blockFrames = 251;
            const auto reused =
                effect.renderCached(source, plan.plan, cacheSource, certificate, replay, job);
            EXPECT_TRUE(result.succeeded() && reused.succeeded());
            EXPECT_TRUE(reused.report.usedDeltaCache);
            EXPECT_TRUE(exact(direct.output, replay.output));
            EXPECT_EQ(result.report.outputSamplePeakDb, reused.report.outputSamplePeakDb);
            EXPECT_EQ(result.report.outputTruePeakDb, reused.report.outputTruePeakDb);
        }
    }
}

DSPARK_TEST(OfflineStereo_cache_rejects_content_configuration_and_provenance_changes)
{
    StereoSource<double> source;
    Processor effect;
    Processor::Options options;
    options.width = 1;
    auto plan = effect.analyze(source, options);
    StereoSink<double> delta;
    const auto cached = effect.buildDeltaCache(source, plan.plan, delta);
    EXPECT_TRUE(cached.succeeded());
    StereoSource<double> data(1);
    data.spec = delta.spec;
    data.audio = std::move(delta.output);
    const auto reject = [&](const Processor::Plan &p) {
        StereoSink<double> sink;
        const auto result = effect.renderCached(source, p, data, cached.cache, sink);
        EXPECT_TRUE(result.status == OfflineStatus::SourceMismatch);
        EXPECT_FALSE(sink.committed);
        EXPECT_TRUE(!sink.begun || sink.aborted);
    };
    for (int field = 0; field < 3; ++field)
    {
        const auto original = source.spec;
        if (field == 0)
            ++source.spec.timelineOrigin;
        if (field == 1)
            ++source.spec.revision;
        if (field == 2)
            ++source.spec.sourceId;
        reject(plan.plan);
        source.spec = original;
    }
    options.oversamplingFactor = 8;
    const auto factor = effect.replan(source, plan.plan, options);
    reject(factor.plan);
    options.oversamplingFactor = 1;
    options.lowCutHz = 175;
    const auto highPass = effect.replan(source, plan.plan, options);
    reject(highPass.plan);
    const double original = data.audio.getChannel(0)[1024];
    data.audio.getChannel(0)[1024] *= .5;
    reject(plan.plan);
    data.audio.getChannel(0)[1024] = std::numeric_limits<double>::quiet_NaN();
    reject(plan.plan);
    data.audio.getChannel(0)[1024] = original;
    source.audio.getChannel(0)[1024] *= .5;
    reject(plan.plan);
    source.audio.getChannel(0)[1024] *= 2;
    ++data.spec.timelineOrigin;
    reject(plan.plan);
    --data.spec.timelineOrigin;
    StereoSink<double> recovered;
    EXPECT_TRUE(effect.renderCached(source, plan.plan, data, cached.cache, recovered).succeeded());
    options.lowCutHz = 0;
    options.width = 0;
    const auto neutral = effect.replan(source, plan.plan, options);
    data.audio.getChannel(0)[1024] = -original; // Same peak, different complete content.
    reject(neutral.plan);
    data.audio.getChannel(0)[1024] = original;
    data.failCall = data.readCalls + 2;
    StereoSink<double> failedRead;
    EXPECT_TRUE(effect.renderCached(source, neutral.plan, data, cached.cache, failedRead).status ==
                OfflineStatus::SourceReadFailed);
    EXPECT_TRUE(failedRead.aborted && !failedRead.committed);
}

DSPARK_TEST(OfflineStereo_analysis_reuse_moves_and_validation)
{
    StereoSource<double> source;
    Processor effect;
    Processor::Options options;
    options.width = .5f;
    const auto energy = OfflineEnergyAnalyzer<double>().analyze(source);
    EXPECT_TRUE(energy.succeeded());
    const auto before = source.readFrames;
    auto plan = effect.makePlan(source, energy.analysis, options);
    EXPECT_TRUE(plan.succeeded());
    EXPECT_EQ(source.readFrames, before);
    auto moved = std::move(plan.plan);
    EXPECT_FALSE(plan.plan.isValid());
    EXPECT_EQ(plan.plan.retainedBytes(), 0u);
    EXPECT_EQ(plan.plan.exclusions().size(), 0u);
    StereoSink<double> sink;
    EXPECT_TRUE(effect.render(source, plan.plan, sink).status == OfflineStatus::InvalidInput);
    EXPECT_TRUE(effect.render(source, moved, sink).succeeded());
    for (float width : {-1.f, 1.001f, std::numeric_limits<float>::quiet_NaN()})
    {
        options.width = width;
        EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::InvalidInput);
    }
    options.width = 1;
    options.oversamplingFactor = 3;
    EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::InvalidInput);
    options.oversamplingFactor = 1;
    options.lowCutHz = 19;
    EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::InvalidInput);
    options.lowCutHz = 0;
    const OfflineRegion invalid{100, 99};
    options.exclusions = {&invalid, 1};
    EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::InvalidInput);
    options.exclusions = {};
    source.spec.sampleRate = 7999;
    EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::InvalidInput);
    source.spec.sampleRate = 48000;
    source.spec.frames = 0;
    EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::EmptyInput);
    source.spec.frames = 1025;
    source.spec.channels = 3;
    EXPECT_TRUE(effect.analyze(source, options).status == OfflineStatus::UnsupportedLayout);
}

DSPARK_TEST(OfflineStereo_failures_abort_direct_cached_and_cache_construction)
{
    StereoSource<double> source;
    Processor effect;
    Processor::Options options;
    options.width = 1;
    const auto plan = effect.analyze(source, options);
    StereoSink<double> delta;
    const auto cache = effect.buildDeltaCache(source, plan.plan, delta);
    OfflineBufferSource<double> data(delta.output.toView(), delta.spec.sampleRate,
                                     delta.spec.sourceId, delta.spec.revision,
                                     delta.spec.timelineOrigin);
    for (int path : {0, 1, 2})
        for (bool throwing : {false, true})
            for (int failure : {1, 2, 3})
            {
                StereoSink<double> sink;
                sink.fail = failure;
                sink.throws = throwing;
                const auto status =
                    path == 0 ? effect.render(source, plan.plan, sink).status
                    : path == 1
                        ? effect.buildDeltaCache(source, plan.plan, sink).status
                        : effect.renderCached(source, plan.plan, data, cache.cache, sink).status;
                EXPECT_TRUE(status == OfflineStatus::SinkFailed);
                EXPECT_TRUE(sink.begun && sink.aborted && !sink.committed);
            }
    for (bool throwing : {false, true})
    {
        source.failCall = source.readCalls + 2;
        source.throws = throwing;
        StereoSink<double> sink;
        EXPECT_TRUE(effect.render(source, plan.plan, sink).status ==
                    OfflineStatus::SourceReadFailed);
        EXPECT_TRUE(sink.aborted && !sink.committed);
    }
    source.failCall = 0;
    source.mutateCall = source.readCalls + 2;
    StereoSink<double> sink;
    EXPECT_TRUE(effect.render(source, plan.plan, sink).status == OfflineStatus::SourceMismatch);
    EXPECT_TRUE(sink.aborted && !sink.committed);
}

DSPARK_TEST(OfflineStereo_cancellation_callback_failure_and_memory_limits)
{
    StereoSource<double> source;
    Processor effect;
    Processor::Options options;
    options.width = 1;
    const auto plan = effect.analyze(source, options);
    StereoSink<double> delta;
    const auto cached = effect.buildDeltaCache(source, plan.plan, delta);
    OfflineBufferSource<double> data(delta.output.toView(), delta.spec.sampleRate,
                                     delta.spec.sourceId, delta.spec.revision,
                                     delta.spec.timelineOrigin);
    std::atomic<bool> stop{true};
    OfflineJobOptions job;
    job.cancel = &stop;
    EXPECT_TRUE(effect.analyze(source, options, job).status == OfflineStatus::Cancelled);
    job.cancel = nullptr;
    struct Context
    {
        bool throwing;
    } context{false};
    job.progressContext = &context;
    job.progress = [](void *ctx, OfflineProgress p) {
        if (p.phase == OfflinePhase::Render && p.completed > 0)
        {
            if (static_cast<Context *>(ctx)->throwing)
                throw std::runtime_error("progress");
            return false;
        }
        return true;
    };
    for (int path : {0, 1, 2})
        for (bool throwing : {false, true})
        {
            context.throwing = throwing;
            StereoSink<double> sink;
            const auto status =
                path == 0 ? effect.render(source, plan.plan, sink, job).status
                : path == 1
                    ? effect.buildDeltaCache(source, plan.plan, sink, job).status
                    : effect.renderCached(source, plan.plan, data, cached.cache, sink, job).status;
            EXPECT_TRUE(status ==
                        (throwing ? OfflineStatus::CallbackFailed : OfflineStatus::Cancelled));
            EXPECT_TRUE(sink.aborted && !sink.committed);
        }
    job = {};
    job.memoryBudgetBytes = 0;
    EXPECT_TRUE(effect.analyze(source, options, job).status == OfflineStatus::MemoryLimit);
    StereoSink<double> sink;
    EXPECT_TRUE(effect.render(source, plan.plan, sink, job).status == OfflineStatus::MemoryLimit);
    EXPECT_FALSE(sink.begun);
}

DSPARK_TEST(OfflineStereo_silence_extreme_finite_and_nonfinite_input)
{
    auto input = stereoFixture<double>(2, 17);
    AudioBuffer<double> output;
    Processor effect;
    Processor::Options options;
    options.width = 1;
    input.clear();
    auto result = effect.run(input, output, 48000, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(result.report.reason == Processor::Reason::NoSignal);
    EXPECT_TRUE(exact(input, output));
    for (double magnitude :
         {std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
          std::numeric_limits<double>::max()})
    {
        options.width = 0;
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 17; ++i)
                input.getChannel(c)[i] = i % 2 ? magnitude : -magnitude;
        result = effect.run(input, output, 48000, options);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(exact(input, output));
        EXPECT_TRUE(std::isfinite(result.report.outputTruePeakDb));
        EXPECT_NEAR(result.report.inputSamplePeakDb, 20 * std::log10(magnitude), 1e-10);
    }
    output.resize(1, 1);
    output.getChannel(0)[0] = .123;
    options.width = 1;
    EXPECT_TRUE(effect.run(input, output, 48000, options).status ==
                OfflineStatus::NumericalFailure);
    EXPECT_EQ(output.getNumSamples(), 1);
    EXPECT_EQ(output.getChannel(0)[0], .123);
    input.getChannel(0)[16] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(effect.run(input, output, 48000, options).status == OfflineStatus::NonFiniteInput);
    EXPECT_EQ(output.getChannel(0)[0], .123);
}

DSPARK_TEST(OfflineMidSide_energy_actual_intervals_and_complete_source_weighting)
{
    StereoSource<double> source(2, 5);
    source.spec.sampleRate = 200;
    constexpr double left[]{2, 0, 3, 1, 4}, right[]{0, 2, -1, -3, 0};
    std::copy_n(left, 5, source.audio.getChannel(0));
    std::copy_n(right, 5, source.audio.getChannel(1));
    OfflineEnergyAnalyzer<double> analyzer;
    auto measured = analyzer.analyzeMidSide(source);
    EXPECT_TRUE(measured.succeeded());
    EXPECT_TRUE(measured.analysis.getSpec() == source.spec);
    EXPECT_EQ(measured.analysis.binFrames(), 2);
    const auto bins = measured.analysis.bins();
    EXPECT_EQ(bins.size(), std::size_t(3));
    EXPECT_EQ(bins[0].frames, 2);
    EXPECT_EQ(bins[1].frames, 2);
    EXPECT_EQ(bins[2].frames, 1);
    EXPECT_EQ(bins[0].midRms, 1);
    EXPECT_EQ(bins[0].sideRms, 1);
    EXPECT_EQ(bins[1].midRms, 1);
    EXPECT_EQ(bins[1].sideRms, 2);
    EXPECT_EQ(bins[2].midRms, 2);
    EXPECT_EQ(bins[2].sideRms, 2);
    EXPECT_NEAR(measured.analysis.midRms(), std::sqrt(8. / 5), 1e-15);
    EXPECT_NEAR(measured.analysis.sideRms(), std::sqrt(14. / 5), 1e-15);
    EXPECT_EQ(measured.analysis.samplePeak(), 4);
    EXPECT_EQ(measured.analysis.sidePeak(), 2);
    const auto linked = analyzer.analyze(source);
    EXPECT_TRUE(linked.succeeded());
    EXPECT_EQ(linked.analysis.binFrames(), 20);
    EXPECT_EQ(linked.analysis.bins().size(), std::size_t(1));
    EXPECT_NEAR(linked.analysis.bins()[0].rms, std::sqrt(4.4), 1e-15);
    EXPECT_TRUE(linked.analysis.fingerprint() == measured.analysis.fingerprint());
    const auto retained = measured.analysis.retainedBytes();
    auto moved = std::move(measured.analysis);
    EXPECT_FALSE(measured.analysis.isValid());
    EXPECT_TRUE(measured.analysis.bins().empty());
    EXPECT_EQ(measured.analysis.retainedBytes(), std::size_t(0));
    EXPECT_EQ(moved.retainedBytes(), retained);
}

DSPARK_TEST(OfflineMidSide_partitions_rates_provenance_and_side_fingerprint)
{
    const auto check = []<typename T>()
    {
        for (double rate : {8000., 44100., 44100.125, 48000., 96000., 192000., 384000.})
        {
            StereoSource<T> source;
            source.spec.sampleRate = rate;
            OfflineEnergyAnalyzer<T> analyzer;
            const auto reference = analyzer.analyzeMidSide(source);
            EXPECT_TRUE(reference.succeeded());
            for (int block : {1, 7, 257, 4096})
            {
                source.readFrames = source.maxRead = 0;
                OfflineJobOptions job;
                job.blockFrames = block;
                const auto measured = analyzer.analyzeMidSide(source, job);
                EXPECT_TRUE(measured.succeeded());
                EXPECT_EQ(source.readFrames, source.spec.frames);
                EXPECT_TRUE(source.maxRead <= block);
                EXPECT_TRUE(measured.analysis.getSpec() == source.spec);
                EXPECT_TRUE(measured.analysis.fingerprint() == reference.analysis.fingerprint());
                EXPECT_TRUE(measured.analysis.sideFingerprint() ==
                            reference.analysis.sideFingerprint());
                EXPECT_EQ(measured.analysis.midRms(), reference.analysis.midRms());
                EXPECT_EQ(measured.analysis.sideRms(), reference.analysis.sideRms());
                const auto bins = measured.analysis.bins(), expected = reference.analysis.bins();
                EXPECT_EQ(bins.size(), expected.size());
                for (std::size_t i = 0; i < bins.size(); ++i)
                {
                    EXPECT_EQ(bins[i].frames, expected[i].frames);
                    EXPECT_EQ(bins[i].midRms, expected[i].midRms);
                    EXPECT_EQ(bins[i].sideRms, expected[i].sideRms);
                    EXPECT_EQ(bins[i].midPeak, expected[i].midPeak);
                    EXPECT_EQ(bins[i].sidePeak, expected[i].sidePeak);
                }
            }
        }
    };
    check.operator()<float>();
    check.operator()<double>();
    StereoSource<double> source(2, 3);
    source.audio.clear();
    source.audio.getChannel(0)[1] = .5;
    source.audio.getChannel(1)[1] = -.25;
    OfflineEnergyAnalyzer<double> analyzer;
    const auto first = analyzer.analyzeMidSide(source);
    source.audio.getChannel(0)[1] += .25;
    source.audio.getChannel(1)[1] += .25;
    const auto changedMid = analyzer.analyzeMidSide(source);
    EXPECT_TRUE(first.succeeded() && changedMid.succeeded());
    EXPECT_TRUE(first.analysis.sideFingerprint() == changedMid.analysis.sideFingerprint());
    EXPECT_FALSE(first.analysis.fingerprint() == changedMid.analysis.fingerprint());
    OfflineFingerprint expectedSide;
    for (int i = 0; i < 3; ++i)
        detail::offlineHash(expectedSide,
                            (source.audio.getChannel(0)[i] - source.audio.getChannel(1)[i]) * .5);
    EXPECT_TRUE(expectedSide == changedMid.analysis.sideFingerprint());
}

DSPARK_TEST(OfflineMidSide_silence_and_finite_extremes_preserve_small_values)
{
    StereoSource<double> source(2, 1);
    OfflineEnergyAnalyzer<double> analyzer;
    for (double magnitude :
         {0., std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
          std::numeric_limits<double>::max()})
        for (int sign : {-1, 1})
        {
            source.audio.getChannel(0)[0] = magnitude;
            source.audio.getChannel(1)[0] = sign * magnitude;
            const auto result = analyzer.analyzeMidSide(source);
            EXPECT_TRUE(result.succeeded());
            EXPECT_EQ(result.analysis.samplePeak(), magnitude);
            EXPECT_EQ(result.analysis.midRms(), sign == 1 ? magnitude : 0.);
            EXPECT_EQ(result.analysis.sideRms(), sign == -1 ? magnitude : 0.);
            EXPECT_EQ(result.analysis.sidePeak(), sign == -1 ? magnitude : 0.);
        }
}

DSPARK_TEST(OfflineMidSide_failures_discard_partial_analysis)
{
    StereoSource<double> source;
    OfflineEnergyAnalyzer<double> analyzer;
    OfflineJobOptions job;
    job.blockFrames = 17;
    const auto expect = [&](OfflineStatus status)
    {
        auto result = analyzer.analyzeMidSide(source, job);
        EXPECT_TRUE(result.status == status);
        EXPECT_FALSE(result.analysis.isValid());
        EXPECT_TRUE(result.analysis.bins().empty());
        EXPECT_EQ(result.analysis.retainedBytes(), std::size_t(0));
    };
    job.memoryBudgetBytes = 0;
    expect(OfflineStatus::MemoryLimit);
    job.memoryBudgetBytes = 256u * 1024u * 1024u;
    source.failCall = source.readCalls + 2;
    expect(OfflineStatus::SourceReadFailed);
    source.failCall = 0;
    source.mutateCall = source.readCalls + 2;
    expect(OfflineStatus::SourceMismatch);
    source.mutateCall = 0;
    job.progress = [](void *, OfflineProgress p) { return p.completed == 0; };
    expect(OfflineStatus::Cancelled);
    job.progress = nullptr;
    const double original = source.audio.getChannel(1)[512];
    for (double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        source.audio.getChannel(1)[512] = invalid;
        expect(OfflineStatus::NonFiniteInput);
    }
    source.audio.getChannel(1)[512] = original;
    source.spec.channels = 1;
    expect(OfflineStatus::UnsupportedLayout);
    source.spec.channels = 2;
    source.spec.frames = 0;
    expect(OfflineStatus::EmptyInput);
}

DSPARK_TEST(OfflineStereoBalance_neutral_and_known_energy_ratio)
{
    const auto check = []<typename T>()
    {
        auto input = stereoFixture<T>(2, 257);
        AudioBuffer<T> output;
        OfflineStereoBalance<T> processor;
        auto neutral = processor.run(input, output, 8000);
        EXPECT_TRUE(neutral.succeeded());
        EXPECT_TRUE(exact(input, output));
        EXPECT_TRUE(neutral.report.peaksMeasured);
        for (int i = 0; i < input.getNumSamples(); ++i)
        {
            input.getChannel(0)[i] = T(.3125);
            input.getChannel(1)[i] = T(.1875);
        }
        typename OfflineStereoBalance<T>::Options options;
        options.levelingAmount = 1;
        options.targetSideShare = .2;
        options.guard = true;
        options.guardMarginDb = 0;
        OfflineBufferSource<T> source(input.toView(), 8000);
        const auto planned = processor.analyze(source, options);
        EXPECT_TRUE(planned.succeeded());
        const double boost = 20 * std::log10(2.);
        EXPECT_NEAR(planned.plan.getReport().minimumGainDb, boost, 1e-12);
        EXPECT_NEAR(planned.plan.getReport().maximumGainDb, boost, 1e-12);
        auto cutOptions = options;
        cutOptions.targetSideShare = 1. / 65;
        const auto attenuated = processor.analyze(source, cutOptions);
        EXPECT_TRUE(attenuated.succeeded());
        EXPECT_NEAR(attenuated.plan.getReport().minimumGainDb, -boost, 1e-12);
        EXPECT_NEAR(attenuated.plan.getReport().maximumGainDb, -boost, 1e-12);
        const auto result = processor.run(input, output, 8000, options);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(result.report.guardMeasured && result.report.guardTargetMet);
        EXPECT_EQ(result.report.violatedWindows, std::size_t(0));
        EXPECT_NEAR(result.report.outputSideShare, .2, 1e-12);
        EXPECT_EQ(output.getNumSamples(), input.getNumSamples());
        for (int i = 0; i < input.getNumSamples(); ++i)
        {
            EXPECT_NEAR(output.getChannel(0)[i], .375, 1e-12);
            EXPECT_NEAR(output.getChannel(1)[i], .125, 1e-12);
        }
    };
    check.operator()<float>();
    check.operator()<double>();
}

DSPARK_TEST(OfflineStereoBalance_guard_measures_actual_windows_and_preserves_mid)
{
    constexpr int rate = 8000, frames = 12017, hop = 80;
    auto input = stereoFixture<double>(2, frames);
    for (int i = 0; i < frames; ++i)
    {
        const double mid = .2 * std::sin(twoPi<double> * 317 * i / rate);
        const double amplitude = i < 4000 ? .04 : i < 8000 ? .5 : .1;
        const double side = amplitude * std::cos(twoPi<double> * 997 * i / rate);
        input.getChannel(0)[i] = mid + side;
        input.getChannel(1)[i] = mid - side;
    }
    OfflineStereoBalance<double> processor;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = .5;
    options.guard = true;
    options.guardMarginDb = 0;
    options.targetSideShare = .2;
    AudioBuffer<double> output;
    const auto result = processor.run(input, output, rate, options);
    EXPECT_TRUE(result.succeeded());
    if (!result.succeeded())
        return;
    EXPECT_TRUE(result.report.guardTargetMet);
    EXPECT_TRUE(result.report.guardCalibrationPasses >= 1);
    EXPECT_TRUE(result.report.guardCalibrationPasses <= 16);
    EXPECT_EQ(result.report.violatedWindows, std::size_t(0));
    for (int i = 0; i < frames; ++i)
        EXPECT_NEAR(output.getChannel(0)[i] + output.getChannel(1)[i],
                    input.getChannel(0)[i] + input.getChannel(1)[i], 3e-16);
    for (int begin = 0; begin < frames; begin += hop)
    {
        double midPower = 0, sidePower = 0;
        for (int i = begin; i < std::min(frames, begin + 4 * hop); ++i)
        {
            const double m = (output.getChannel(0)[i] + output.getChannel(1)[i]) * .5;
            const double s = (output.getChannel(0)[i] - output.getChannel(1)[i]) * .5;
            midPower += m * m;
            sidePower += s * s;
        }
        EXPECT_TRUE(sidePower <= .25 * midPower * (1 + 1e-9));
    }
}

DSPARK_TEST(OfflineStereoBalance_caps_exclusions_and_unreachable_guard)
{
    StereoSource<double> source(2, 801);
    source.spec.sampleRate = 8000;
    for (int i = 0; i < 801; ++i)
    {
        source.audio.getChannel(0)[i] = .25 + .001;
        source.audio.getChannel(1)[i] = .25 - .001;
    }
    OfflineStereoBalance<double> processor;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = 1;
    options.maximumBoostDb = 6;
    options.targetSideShare = .2;
    const auto plan = processor.analyze(source, options);
    EXPECT_TRUE(plan.succeeded());
    EXPECT_TRUE(plan.plan.getReport().reason == OfflineStereoBalance<double>::Reason::GainLimited);
    EXPECT_NEAR(plan.plan.gainAt(100), std::pow(10., .3), 1e-14);
    const std::array<OfflineRegion, 1> excluded{{{80, 241}}};
    options.exclusions = excluded;
    auto protectedPlan = processor.analyze(source, options);
    StereoSink<double> sink;
    const auto protectedResult = processor.render(source, protectedPlan.plan, sink);
    EXPECT_TRUE(protectedResult.succeeded());
    EXPECT_TRUE(sink.committed);
    for (int i = 80; i < 241; ++i)
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(sink.output.getChannel(c)[i], source.audio.getChannel(c)[i]);
    for (int i = 0; i < 801; ++i)
    {
        source.audio.getChannel(0)[i] = .75;
        source.audio.getChannel(1)[i] = -.25;
    }
    const std::array<OfflineRegion, 1> all{{{0, 801}}};
    options.exclusions = all;
    options.guard = true;
    options.guardMarginDb = 0;
    auto impossible = processor.analyze(source, options);
    StereoSink<double> refused;
    const auto failed = processor.render(source, impossible.plan, refused);
    EXPECT_TRUE(failed.status == OfflineStatus::TargetUnreachable);
    EXPECT_FALSE(refused.committed);
    EXPECT_FALSE(refused.begun);
    EXPECT_FALSE(failed.report.guardTargetMet);
}

DSPARK_TEST(OfflineStereoBalance_near_nyquist_uses_the_independent_gain_reference)
{
    constexpr int rate = 8000, frames = 20001;
    StereoSource<double> source(2, frames);
    source.spec.sampleRate = rate;
    std::vector<double> side(frames), gain(frames);
    for (int i = 0; i < frames; ++i)
    {
        const double mid = .2 * std::sin(twoPi<double> * 233 * i / rate);
        const double amplitude = i < 6000 ? .015 : i < 12000 ? .06 : .15;
        const double s = amplitude * std::sin(twoPi<double> * 3917 * i / rate);
        source.audio.getChannel(0)[i] = mid + s;
        source.audio.getChannel(1)[i] = mid - s;
        side[i] = (source.audio.getChannel(0)[i] - source.audio.getChannel(1)[i]) * .5;
    }
    OfflineStereoBalance<double> processor;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = 1;
    options.targetSideShare = .2;
    const auto plan = processor.analyze(source, options);
    EXPECT_TRUE(plan.succeeded());
    for (int i = 0; i < frames; ++i)
        gain[i] = plan.plan.gainAt(i);
    StereoSink<double> sink;
    const auto result = processor.render(source, plan.plan, sink);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.report.renderInfo.bandlimited);
    const auto delta = test::offline_reference::referenceDelta(side, gain);
    double error = 0, power = 0, scalarError = 0;
    for (int i = 0; i < frames; ++i)
    {
        const double measured = (sink.output.getChannel(0)[i] - sink.output.getChannel(1)[i]) * .5;
        const double expected = side[i] + delta[i];
        error += (measured - expected) * (measured - expected);
        power += expected * expected;
        scalarError += (side[i] * gain[i] - expected) * (side[i] * gain[i] - expected);
    }
    EXPECT_LT(error / power, 1e-9);
    EXPECT_GT(scalarError, error * 100);
}

DSPARK_TEST(OfflineStereoBalance_partitions_reports_and_true_peak_tail)
{
    StereoSource<double> source(2, 1025);
    OfflineStereoBalance<double> processor;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = 1;
    options.targetSideShare = .2;
    const auto plan = processor.analyze(source, options);
    EXPECT_TRUE(plan.succeeded());
    EXPECT_TRUE(plan.plan.getSpec() == source.spec);
    StereoSink<double> baseline;
    auto initial = processor.render(source, plan.plan, baseline);
    EXPECT_TRUE(initial.succeeded());
    for (int block : {1, 7, 257, 4096})
    {
        OfflineJobOptions job;
        job.blockFrames = block;
        StereoSink<double> sink;
        const auto result = processor.render(source, plan.plan, sink, job);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(exact(sink.output, baseline.output));
        EXPECT_TRUE(sink.spec == source.spec);
        EXPECT_EQ(result.report.analysisHopFrames, 480);
        EXPECT_EQ(result.report.guardWindowFrames, 1920);
        TruePeakDetector<double, 2> meter;
        double peak = 0, truePeak = 0;
        for (int i = 0; i < 1025; ++i)
            for (int c = 0; c < 2; ++c)
            {
                const double x = sink.output.getChannel(c)[i];
                peak = std::max(peak, std::abs(x));
                truePeak = std::max(truePeak, meter.processSample(x, c));
            }
        for (int i = 0; i < TruePeakDetector<double, 2>::getTaps() - 1; ++i)
            for (int c = 0; c < 2; ++c)
                truePeak = std::max(truePeak, meter.processSample(0, c));
        EXPECT_NEAR(result.report.outputSamplePeakDb, 20 * std::log10(peak), 1e-12);
        EXPECT_NEAR(result.report.outputTruePeakDb, 20 * std::log10(truePeak), 1e-12);
    }
}

DSPARK_TEST(OfflineStereoBalance_provenance_cancellation_and_sink_failures)
{
    StereoSource<double> source(2, 129);
    for (int i = 0; i < 129; ++i)
    {
        source.audio.getChannel(0)[i] = .3125;
        source.audio.getChannel(1)[i] = .1875;
    }
    OfflineStereoBalance<double> processor;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = 1;
    options.targetSideShare = .2;
    const auto plan = processor.analyze(source, options);
    for (int stage : {1, 2, 3})
        for (bool throwing : {false, true})
        {
            StereoSink<double> sink;
            sink.fail = stage;
            sink.throws = throwing;
            const auto result = processor.render(source, plan.plan, sink);
            EXPECT_TRUE(result.status == OfflineStatus::SinkFailed);
            EXPECT_TRUE(sink.aborted && !sink.committed);
        }
    OfflineJobOptions job;
    job.blockFrames = 7;
    job.progress = [](void *, OfflineProgress p)
    { return !(p.phase == OfflinePhase::Render && p.completed > 0); };
    StereoSink<double> cancelled;
    EXPECT_TRUE(processor.render(source, plan.plan, cancelled, job).status ==
                OfflineStatus::Cancelled);
    EXPECT_TRUE(cancelled.aborted && !cancelled.committed);
    job = {};
    job.memoryBudgetBytes = 0;
    StereoSink<double> limited;
    EXPECT_TRUE(processor.render(source, plan.plan, limited, job).status ==
                OfflineStatus::MemoryLimit);
    EXPECT_FALSE(limited.begun);
    for (int i = 0; i < 129; ++i)
    {
        source.audio.getChannel(0)[i] += .25;
        source.audio.getChannel(1)[i] += .25;
    }
    StereoSink<double> stale;
    EXPECT_TRUE(processor.render(source, plan.plan, stale).status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(stale.committed);
    EXPECT_TRUE(stale.aborted);
    ++source.spec.timelineOrigin;
    StereoSink<double> clock;
    EXPECT_TRUE(processor.render(source, plan.plan, clock).status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(clock.begun);
}

DSPARK_TEST(OfflineStereoBalance_absent_components_extremes_and_invalid_options)
{
    auto input = stereoFixture<double>(2, 81);
    AudioBuffer<double> output;
    OfflineStereoBalance<double> processor;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = 1;
    input.clear();
    auto result = processor.run(input, output, 8000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.report.reason == OfflineStereoBalance<double>::Reason::Silence);
    EXPECT_TRUE(std::isnan(result.report.inputSideShare));
    EXPECT_TRUE(std::isnan(result.report.targetSideShare));
    EXPECT_TRUE(exact(input, output));
    for (double magnitude :
         {std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(), .25,
          std::numeric_limits<double>::max()})
        for (int sign : {-1, 1})
        {
            for (int i = 0; i < 81; ++i)
            {
                input.getChannel(0)[i] = magnitude;
                input.getChannel(1)[i] = sign * magnitude;
            }
            result = processor.run(input, output, 8000, options);
            EXPECT_TRUE(result.succeeded());
            EXPECT_TRUE(exact(input, output));
            EXPECT_TRUE(std::isfinite(result.report.outputTruePeakDb));
        }
    options.targetSideShare = .5;
    EXPECT_TRUE(processor.run(input, output, 8000, options).status == OfflineStatus::InvalidInput);
    options.targetSideShare.reset();
    options.levelingAmount = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(processor.run(input, output, 8000, options).status == OfflineStatus::InvalidInput);
    options = {};
    input.getChannel(1)[80] = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(processor.run(input, output, 8000, options).status ==
                OfflineStatus::NonFiniteInput);
}

DSPARK_TEST(OfflineStereoBalance_pure_side_guard_reports_cap_and_measures_silence_gate)
{
    const auto check = []<typename T>()
    {
        constexpr int frames = 1601, hop = 80;
        StereoSource<T> source(2, frames);
        source.spec.sampleRate = 8000;
        long double inputEnergy = 0;
        for (int i = 0; i < frames; ++i)
        {
            const T side = static_cast<T>(i < 800 ? .1 : .3);
            source.audio.getChannel(0)[i] = side;
            source.audio.getChannel(1)[i] = -side;
            inputEnergy += static_cast<long double>(side) * side;
        }
        OfflineStereoBalance<T> processor;
        typename OfflineStereoBalance<T>::Options options;
        options.guard = true;
        auto plan = processor.analyze(source, options);
        EXPECT_TRUE(plan.succeeded());
        StereoSink<T> refused;
        const auto limited = processor.render(source, plan.plan, refused);
        EXPECT_TRUE(limited.status == OfflineStatus::TargetUnreachable);
        EXPECT_TRUE(limited.report.reason == OfflineStereoBalance<T>::Reason::NoMid);
        EXPECT_TRUE(limited.report.guardMeasured);
        EXPECT_FALSE(limited.report.guardTargetMet);
        EXPECT_GT(limited.report.violatedWindows, std::size_t(0));
        EXPECT_FALSE(refused.begun);
        EXPECT_FALSE(refused.committed);
        const long double gate = std::max(1e-14L, inputEnergy / frames * 1e-6L);
        // The louder region still exceeds the fixed gate at the 60 dB cap.
        EXPECT_GT(.09L * 1e-6L, gate);
        options.maximumGuardCutDb = 120;
        plan = processor.analyze(source, options);
        EXPECT_TRUE(plan.succeeded());
        StereoSink<T> accepted;
        const auto reduced = processor.render(source, plan.plan, accepted);
        EXPECT_TRUE(reduced.succeeded());
        EXPECT_TRUE(accepted.committed);
        EXPECT_TRUE(reduced.report.reason == OfflineStereoBalance<T>::Reason::NoMid);
        EXPECT_TRUE(reduced.report.guardTargetMet);
        EXPECT_EQ(reduced.report.violatedWindows, std::size_t(0));
        // No mid has been synthesized. Every rendered window is below the
        // independent input-referenced silence gate, including the short tail.
        for (int begin = 0; begin < frames; begin += hop)
        {
            const int end = std::min(frames, begin + 4 * hop);
            long double energy = 0;
            for (int i = begin; i < end; ++i)
            {
                const long double left = accepted.output.getChannel(0)[i];
                const long double right = accepted.output.getChannel(1)[i];
                EXPECT_EQ(left + right, 0.L);
                const long double side = (left - right) / 2;
                energy += side * side;
            }
            EXPECT_TRUE(energy <= gate * (end - begin));
        }
    };
    check.operator()<float>();
    check.operator()<double>();
}
