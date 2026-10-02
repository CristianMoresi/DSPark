// DSPark - Offline stereo transport, provenance and transaction contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Core/ProcessorTraits.h"
#include "../Effects/OfflineStereoGenerator.h"
#include "dspark_test.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflineStereoGenerator<double>, double>);

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
