// DSPark - Offline beat compression and local release signal contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

// A Windows host may include the SDK before DSPark. Keep its legacy macros
// active while compiling the public header to cover that include order.
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "../Core/detail/OfflineAttenuation.h"
#include "../Core/detail/OfflinePunchEnvelope.h"
#include "../Core/ProcessorTraits.h"
#include "../Effects/OfflineBeatCompressor.h"
#include "dspark_test.h"
#include "offline_gain_reference.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflineBeatCompressor<float>, float>);

namespace
{
struct ReferenceHold
{
    int begin, end;
    double gain, release;
};

// Deliberately dense independent recurrence and four explicit box convolutions.
// No production prefix maximum, sparse lookup or smoothing weights are reused.
std::vector<double> attenuationReference(std::span<const ReferenceHold> holds, int attack, int n)
{
    std::vector<double> raw(static_cast<std::size_t>(n + attack));
    double tail = 0, release = holds.front().release;
    for (int f = 0; f < n + attack; ++f)
    {
        tail *= std::exp(-1 / release);
        for (const auto &h : holds)
            if (f == h.end + attack)
            {
                tail = std::max(tail, 1 - h.gain);
                release = h.release;
            }
        double deficit = tail;
        for (const auto &h : holds)
            if (f >= h.begin && f < h.end + attack)
                deficit = std::max(deficit, 1 - h.gain);
        raw[static_cast<std::size_t>(f)] = deficit;
    }
    for (int box = 0; box < 4; ++box)
    {
        const int length = attack / 4 + (box < attack % 4) + 1;
        auto out = raw;
        for (std::size_t f = 0; f < raw.size(); ++f)
        {
            long double sum = 0;
            for (int k = 0; k < length; ++k)
                if (f >= static_cast<std::size_t>(k))
                    sum += raw[f - k];
            out[f] = static_cast<double>(sum / length);
        }
        raw = std::move(out);
    }
    for (int f = 0; f < n; ++f)
        raw[static_cast<std::size_t>(f)] = 1 - raw[static_cast<std::size_t>(f + attack)];
    raw.resize(static_cast<std::size_t>(n));
    return raw;
}
}

DSPARK_TEST(OfflineBeat_local_release_matches_independent_time_varying_clock)
{
    const std::array<ReferenceHold, 5> holds{{
        {0, 53, .2, 500}, {48, 129, .7, 31}, {250, 275, .8, 1300},
        {400, 450, 1, 71}, {401, 451, .65, 100}}};
    for (int attack : {1, 7, 16, 81})
    {
        const OfflineJobOptions options;
        detail::OfflineSession job(options);
        detail::OfflineAttenuation curve;
        curve.prepare(job, holds.size(), attack, 250, true);
        for (std::size_t i = 0; i < holds.size(); ++i)
            curve.set(i, holds[i].begin, holds[i].end, holds[i].gain, holds[i].release);
        const auto reference = attenuationReference(holds, attack, 2000);
        detail::OfflineAttenuation::Cursor cursor(curve, job);
        for (int f = 0; f < 2000; ++f)
        {
            EXPECT_NEAR(cursor(f), reference[static_cast<std::size_t>(f)], 3e-13);
            EXPECT_NEAR(curve.gainAt(f), reference[static_cast<std::size_t>(f)], 3e-13);
        }
        for (int f : {120, 0, 700, 52, 1300})
            EXPECT_NEAR(cursor(f), reference[static_cast<std::size_t>(f)], 3e-13);
    }
}

DSPARK_TEST(OfflineBeat_constant_release_preserves_existing_curve_and_payload)
{
    for (double release : {31., 800., 48000.})
    {
        const OfflineJobOptions options;
        detail::OfflineSession job(options);
        detail::OfflineAttenuation fixed, local;
        fixed.prepare(job, 3, 19, release);
        local.prepare(job, 3, 19, release, true);
        for (std::size_t i = 0; i < 3; ++i)
        {
            const auto begin = static_cast<std::int64_t>(100 * i);
            fixed.set(i, begin, begin + 23, .4 + .15 * i);
            local.set(i, begin, begin + 23, .4 + .15 * i, release);
        }
        EXPECT_EQ(local.retainedBytes() - fixed.retainedBytes(), 3 * sizeof(double));
        detail::OfflineAttenuation::Cursor a(fixed, job), b(local, job);
        for (int f = 0; f < 8200; ++f)
        {
            EXPECT_EQ(a(f), b(f));
            if ((f % 19) == 0)
                EXPECT_EQ(fixed.gainAt(f), local.gainAt(f));
        }
        const auto changed = local.withAttack(job, 37);
        EXPECT_TRUE(changed.isValid());
        const std::array<ReferenceHold, 3> oracle{{
            {0, 23, .4, release}, {100, 123, .55, release}, {200, 223, .7, release}}};
        const auto reference = attenuationReference(oracle, 37, 500);
        for (int f = 0; f < 500; ++f)
            EXPECT_NEAR(changed.gainAt(f), reference[static_cast<std::size_t>(f)], 3e-13);
    }
}

DSPARK_TEST(OfflineBeat_release_retains_int64_positions_on_long_logical_timeline)
{
    constexpr std::int64_t origin = (std::int64_t(1) << 54) + 7;
    const OfflineJobOptions options;
    detail::OfflineSession job(options);
    detail::OfflineAttenuation shortCurve, longCurve;
    shortCurve.prepare(job, 2, 9, 400, true);
    longCurve.prepare(job, 2, 9, 400, true);
    for (std::size_t i = 0; i < 2; ++i)
    {
        const auto begin = static_cast<std::int64_t>(i * 100);
        shortCurve.set(i, begin, begin + 50, .3 + .3 * i, 400. / (i + 1));
        longCurve.set(i, origin + begin, origin + begin + 50, .3 + .3 * i, 400. / (i + 1));
    }
    for (std::int64_t f = 0; f < 1000; ++f)
        EXPECT_EQ(shortCurve.gainAt(f), longCurve.gainAt(origin + f));
}

namespace
{
template <class T> AudioBuffer<T> beatSong(int rate, int beats = 16, bool uneven = true)
{
    AudioBuffer<T> input;
    input.resize(2, rate * beats / 2 + rate / 4);
    input.clear();
    for (int beat = 0; beat < beats; ++beat)
        for (int i = 0; i < rate / 25; ++i)
        {
            const double t = static_cast<double>(i) / rate;
            const double amplitude = uneven && beat % 4 == 0 ? .8 : .2;
            const T x = static_cast<T>(amplitude * std::exp(-80 * t) *
                                      (std::cos(twoPi<double> * 113 * t) +
                                       .5 * std::cos(twoPi<double> * 2113 * t)));
            input.getChannel(0)[beat * rate / 2 + i] = x;
            input.getChannel(1)[beat * rate / 2 + i] = -x;
        }
    return input;
}
template <class T> class BeatSink final : public OfflineAudioSink<T>
{
  public:
    AudioBuffer<T> audio;
    bool begun = false, committed = false, aborted = false, failWrite = false;
    bool begin(const OfflineAudioSpec &spec) override
    {
        begun = true;
        audio.resize(spec.channels, static_cast<int>(spec.frames));
        return true;
    }
    bool write(std::int64_t first, AudioBufferView<const T> block) override
    {
        if (failWrite)
            return false;
        for (int c = 0; c < block.getNumChannels(); ++c)
            std::copy_n(block.getChannel(c), block.getNumSamples(), audio.getChannel(c) + first);
        return true;
    }
    bool commit() override { committed = true; return true; }
    void abort() noexcept override { aborted = true; }
};
template <class T> class BeatSource final : public OfflineAudioSource<T>
{
  public:
    AudioBuffer<T> audio = beatSong<T>(8000);
    std::int64_t reads = 0;
    int maximumRead = 0;
    OfflineAudioSpec spec{8000, 2, audio.getNumSamples(), 71, 2, (std::int64_t(1) << 54) + 31};
    bool failRead = false;
    OfflineAudioSpec getSpec() const noexcept override { return spec; }
    bool read(std::int64_t first, AudioBufferView<T> block) override
    {
        reads += block.getNumSamples();
        maximumRead = std::max(maximumRead, block.getNumSamples());
        if (failRead)
            return false;
        for (int c = 0; c < spec.channels; ++c)
            std::copy_n(audio.getChannel(c) + first, block.getNumSamples(), block.getChannel(c));
        return true;
    }
};
}

DSPARK_TEST(OfflineBeat_upper_median_cap_and_weaker_replan)
{
    for (int rate : {8000, 48000, 192000})
    {
        auto input = beatSong<double>(rate);
        OfflineBufferSource<double> source(input.toView(), rate);
        const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, false});
        EXPECT_TRUE(features.succeeded());
        EXPECT_EQ(features.analysis.pulses().size(), std::size_t(16));
        for (double amount : {1., 6., 18., 2.})
        {
            OfflineBeatCompressor<double>::Options options;
            options.reductionDb = amount;
            options.manualBpm = 120;
            auto result = OfflineBeatCompressor<double>().makePlan(source, features.analysis, options);
            EXPECT_TRUE(result.succeeded());
            const auto report = result.plan.getReport();
            EXPECT_NEAR(report.referencePeakDb, 20 * std::log10(.3), 1e-12);
            EXPECT_NEAR(report.maximumExcessDb, 20 * std::log10(4.), 1e-12);
            EXPECT_EQ(report.reducedPulses, std::size_t(4));
            EXPECT_EQ(report.fallbackPulses, std::size_t(0));
            for (std::size_t i = 0; i < result.plan.pulses().size(); ++i)
            {
                const auto &pulse = result.plan.pulses()[i];
                const double expected = i % 4 == 0 ? std::min(amount, 20 * std::log10(4.)) : 0;
                EXPECT_NEAR(pulse.reductionDb, expected, 1e-12);
                EXPECT_EQ(pulse.releaseMs, 500.);
                EXPECT_TRUE(pulse.tempoSource == OfflineBeatCompressor<double>::TempoSource::Manual);
                EXPECT_TRUE(result.plan.gainAt(pulse.begin) >= std::pow(10., -amount / 20) - 2e-15);
                if (i % 4 == 0)
                    EXPECT_NEAR(result.plan.gainAt(pulse.begin), std::pow(10., -expected / 20), 2e-14);
            }
        }
    }
}

DSPARK_TEST(OfflineBeat_auto_tempo_first_last_pulses_and_manual_note_values)
{
    auto input = beatSong<double>(8000, 32, false);
    OfflineBufferSource<double> source(input.toView(), 8000);
    const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, true});
    EXPECT_TRUE(features.succeeded());
    const auto tempo = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    EXPECT_TRUE(tempo.succeeded());
    OfflineBeatCompressor<double> processor;
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 6;
    auto automatic = processor.makePlan(source, features.analysis, tempo.analysis, options);
    EXPECT_TRUE(automatic.succeeded());
    EXPECT_EQ(automatic.plan.getReport().fallbackPulses, std::size_t(0));
    EXPECT_TRUE(automatic.plan.getReport().reason == OfflineBeatCompressor<double>::Reason::EvenPulses);
    EXPECT_EQ(automatic.plan.pulses().size(), std::size_t(32));
    for (const auto &pulse : automatic.plan.pulses())
        EXPECT_NEAR(pulse.releaseMs, 500., 2.);
    int index = 0;
    for (auto note : {OfflineBeatCompressor<double>::Note::Eighth,
                     OfflineBeatCompressor<double>::Note::Quarter,
                     OfflineBeatCompressor<double>::Note::Half,
                     OfflineBeatCompressor<double>::Note::Whole})
    {
        options.releaseNote = note;
        options.manualBpm = 20;
        const auto manual = processor.makePlan(source, features.analysis, options);
        EXPECT_TRUE(manual.succeeded());
        for (const auto &pulse : manual.plan.pulses())
            EXPECT_EQ(pulse.releaseMs, 1500. * (1 << index));
        ++index;
    }
}

DSPARK_TEST(OfflineBeat_unknown_tempo_reports_fallback_without_fabricated_bpm)
{
    auto input = beatSong<float>(8000, 1);
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflineBeatCompressor<float>::Options options;
    options.reductionDb = 6;
    auto result = OfflineBeatCompressor<float>().analyze(source, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(result.plan.pulses().size(), std::size_t(1));
    EXPECT_EQ(result.plan.getReport().fallbackPulses, std::size_t(1));
    EXPECT_EQ(result.plan.pulses()[0].releaseMs, 250.);
    EXPECT_TRUE(result.plan.pulses()[0].tempoSource == OfflineBeatCompressor<float>::TempoSource::Fallback);
}

DSPARK_TEST(OfflineBeat_shared_analysis_source_provenance_and_bounded_single_scan)
{
    BeatSource<double> source;
    const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, true});
    EXPECT_TRUE(features.succeeded());
    const auto tempo = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    EXPECT_TRUE(tempo.succeeded());
    OfflineBeatCompressor<double> processor;
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 6;
    OfflineJobOptions job;
    job.blockFrames = 257;
    source.reads = 0;
    source.maximumRead = 0;
    auto planned = processor.makePlan(source, features.analysis, tempo.analysis, options, job);
    EXPECT_TRUE(planned.succeeded());
    EXPECT_EQ(source.reads, source.spec.frames);
    EXPECT_EQ(source.maximumRead, 257);
    EXPECT_TRUE(planned.plan.getSpec() == source.spec);
    source.audio.getChannel(0)[63000] = .01;
    const auto changed = processor.makePlan(source, features.analysis, tempo.analysis, options, job);
    EXPECT_TRUE(changed.status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(changed.plan.isValid());
    BeatSink<double> sink;
    EXPECT_TRUE(processor.render(source, planned.plan, sink).status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(sink.begun);
    ++source.spec.revision;
    EXPECT_TRUE(processor.makePlan(source, features.analysis, tempo.analysis, options).status ==
                OfflineStatus::SourceMismatch);
    auto moved = std::move(planned.plan);
    EXPECT_TRUE(moved.isValid());
    EXPECT_FALSE(planned.plan.isValid());
    EXPECT_TRUE(planned.plan.pulses().empty());
}

DSPARK_TEST(OfflineBeat_render_matches_independent_full_signal_and_preserves_channels)
{
    auto input = beatSong<double>(8000, 8);
    OfflineBufferSource<double> source(input.toView(), 8000);
    OfflineBeatCompressor<double> processor;
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 6;
    options.manualBpm = 120;
    const auto prepared = processor.analyze(source, options);
    EXPECT_TRUE(prepared.succeeded());
    std::vector<double> x(static_cast<std::size_t>(input.getNumSamples())), gain(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x[i] = input.getChannel(0)[i];
        gain[i] = prepared.plan.gainAt(static_cast<std::int64_t>(i));
    }
    BeatSink<double> sink;
    const auto rendered = processor.render(source, prepared.plan, sink);
    const auto delta = dspark::test::offline_reference::referenceDelta(
        x, gain, rendered.report.renderInfo.leftBoundaryGain, rendered.report.renderInfo.rightBoundaryGain);
    EXPECT_TRUE(rendered.succeeded());
    EXPECT_TRUE(sink.committed);
    EXPECT_TRUE(rendered.report.renderInfo.bandlimited);
    EXPECT_EQ(rendered.report.renderInfo.deltaScale, 1.);
    EXPECT_TRUE(rendered.report.peaksMeasured);
    EXPECT_EQ(sink.audio.getNumSamples(), input.getNumSamples());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(sink.audio.getChannel(0)[i], x[i] + delta[i], 3e-12);
        EXPECT_EQ(sink.audio.getChannel(1)[i], -sink.audio.getChannel(0)[i]);
    }
}

DSPARK_TEST(OfflineBeat_zero_even_and_excluded_pcm_are_exact)
{
    OfflineBeatCompressor<float> processor;
    auto input = beatSong<float>(8000, 8);
    AudioBuffer<float> output;
    const auto neutral = processor.run(input, output, 8000);
    EXPECT_TRUE(neutral.succeeded());
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(input.getChannel(c), output.getChannel(c),
                              input.getNumSamples() * sizeof(float)), 0);
    const std::array<OfflineRegion, 3> protectedRegions{{{100, 701}, {13000, 16000}, {0, 100}}};
    OfflineBeatCompressor<float>::Options options;
    options.reductionDb = 18;
    options.manualBpm = 120;
    options.exclusions = protectedRegions;
    const auto rendered = processor.run(input, output, 8000, options);
    EXPECT_TRUE(rendered.succeeded());
    for (const auto region : protectedRegions)
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(std::memcmp(input.getChannel(c) + region.begin,
                                  output.getChannel(c) + region.begin,
                                  static_cast<std::size_t>(region.end - region.begin) * sizeof(float)), 0);
    auto even = beatSong<float>(8000, 8, false);
    options.exclusions = {};
    const auto unchanged = processor.run(even, output, 8000, options);
    EXPECT_TRUE(unchanged.status == OfflineStatus::NoChange);
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(even.getChannel(c), output.getChannel(c),
                              even.getNumSamples() * sizeof(float)), 0);
}

DSPARK_TEST(OfflineBeat_changing_tempo_uses_local_evidence_without_lowering_confidence_gate)
{
    constexpr int rate = 8000;
    AudioBuffer<double> input;
    input.resize(2, rate * 27);
    input.clear();
    int position = 0;
    for (int beat = 0; beat < 48; ++beat)
    {
        for (int i = 0; i < 320; ++i)
        {
            const double t = double(i) / rate;
            const double x = .2 * std::exp(-80 * t) *
                (std::cos(twoPi<double> * 113 * t) + .5 * std::cos(twoPi<double> * 2113 * t));
            input.getChannel(0)[position + i] = x;
            input.getChannel(1)[position + i] = -x;
        }
        position += beat < 24 ? 4000 : 4800;
    }
    OfflineBufferSource<double> source(input.toView(), rate);
    const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, true});
    EXPECT_TRUE(features.succeeded());
    const auto tempo = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    EXPECT_TRUE(tempo.succeeded());
    EXPECT_LT(tempo.analysis.getReport().confidence, .25);
    EXPECT_GT(tempo.analysis.getReport().localWindows, std::size_t(1));
    EXPECT_GT(tempo.analysis.getReport().locallyConfirmedIntervals, std::size_t(40));
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 6;
    const auto result = OfflineBeatCompressor<double>().makePlan(source, features.analysis, tempo.analysis, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(result.plan.pulses().size(), std::size_t(48));
    EXPECT_EQ(result.plan.getReport().fallbackPulses, std::size_t(0));
    for (const auto &pulse : result.plan.pulses())
        EXPECT_NEAR(pulse.releaseMs, pulse.begin < 12 * rate ? 500. : 600., 2.);
}

DSPARK_TEST(OfflineBeat_release_is_a_measured_amplitude_time_constant)
{
    BeatSource<double> source;
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 6;
    options.manualBpm = 120;
    const auto result = OfflineBeatCompressor<double>().analyze(source, options);
    EXPECT_TRUE(result.succeeded());
    const auto end = result.plan.pulses()[0].end;
    const double a = 1 - result.plan.gainAt(end + 80);
    const double b = 1 - result.plan.gainAt(end + 1680);
    EXPECT_GT(a, 0.);
    EXPECT_GT(b, 0.);
    EXPECT_NEAR(-200. / std::log(b / a), 500., 2.);
    for (std::int64_t frame = 0; frame < source.spec.frames; ++frame)
    {
        const auto gain = result.plan.gainAt(frame);
        EXPECT_TRUE(gain >= std::pow(10., -.3) - 1e-15 && gain <= 1);
    }
}

DSPARK_TEST(OfflineBeat_ragged_blocks_render_identically_and_in_place_is_safe)
{
    auto input = beatSong<float>(8000, 8);
    OfflineBeatCompressor<float> processor;
    OfflineBeatCompressor<float>::Options options;
    options.reductionDb = 9;
    options.manualBpm = 120;
    AudioBuffer<float> reference;
    EXPECT_TRUE(processor.run(input, reference, 8000, options).succeeded());
    for (int block : {17, 257})
    {
        OfflineJobOptions job;
        job.blockFrames = block;
        AudioBuffer<float> output;
        EXPECT_TRUE(processor.run(input, output, 8000, options, job).succeeded());
        for (int c = 0; c < 2; ++c)
            EXPECT_EQ(std::memcmp(reference.getChannel(c), output.getChannel(c),
                                  reference.getNumSamples() * sizeof(float)), 0);
    }
    EXPECT_TRUE(processor.run(input, input, 8000, options).succeeded());
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(reference.getChannel(c), input.getChannel(c),
                              reference.getNumSamples() * sizeof(float)), 0);
}

DSPARK_TEST(OfflineBeat_source_scale_and_channel_swap_preserve_the_control)
{
    OfflineBeatCompressor<double> processor;
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 9;
    options.manualBpm = 120;
    auto input = beatSong<double>(8000, 8);
    OfflineBufferSource<double> source(input.toView(), 8000);
    const auto original = processor.analyze(source, options);
    EXPECT_TRUE(original.succeeded());
    for (double scale : {1e-250, -1e250})
    {
        auto scaled = beatSong<double>(8000, 8);
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < scaled.getNumSamples(); ++f)
                scaled.getChannel(c)[f] *= scale;
        OfflineBufferSource<double> scaledSource(scaled.toView(), 8000);
        const auto plan = processor.analyze(scaledSource, options);
        EXPECT_TRUE(plan.succeeded());
        EXPECT_EQ(plan.plan.pulses().size(), original.plan.pulses().size());
        for (int f = 0; f < scaled.getNumSamples(); f += 7)
            EXPECT_NEAR(plan.plan.gainAt(f), original.plan.gainAt(f), 3e-13);
        BeatSink<double> sink;
        EXPECT_TRUE(processor.render(scaledSource, plan.plan, sink).succeeded());
        for (int f = 0; f < scaled.getNumSamples(); ++f)
            EXPECT_TRUE(std::isfinite(sink.audio.getChannel(0)[f]));
    }
}

DSPARK_TEST(OfflineBeat_budget_cancellation_source_and_sink_failures_publish_nothing)
{
    BeatSource<float> source;
    OfflineBeatCompressor<float> processor;
    OfflineBeatCompressor<float>::Options options;
    options.reductionDb = 6;
    options.manualBpm = 120;
    const auto features = OfflineTransientAnalyzer<float>().analyze(source, {false, true, false});
    EXPECT_TRUE(features.succeeded());
    auto prepared = processor.makePlan(source, features.analysis, options);
    EXPECT_TRUE(prepared.succeeded());
    OfflineJobOptions job;
    job.memoryBudgetBytes = prepared.memoryBytes;
    EXPECT_TRUE(processor.makePlan(source, features.analysis, options, job).succeeded());
    --job.memoryBudgetBytes;
    const auto limited = processor.makePlan(source, features.analysis, options, job);
    EXPECT_TRUE(limited.status == OfflineStatus::MemoryLimit);
    EXPECT_FALSE(limited.plan.isValid());
    for (int stop : {1, 3, 15})
    {
        int calls = 0;
        struct Context { int *calls; int stop; } context{&calls, stop};
        job = {};
        job.progressContext = &context;
        job.progress = [](void *p, OfflineProgress) {
            const auto &c = *static_cast<Context *>(p);
            return ++*c.calls < c.stop;
        };
        const auto cancelled = processor.makePlan(source, features.analysis, options, job);
        EXPECT_TRUE(cancelled.status == OfflineStatus::Cancelled);
        EXPECT_FALSE(cancelled.plan.isValid());
    }
    job = {};
    job.progress = [](void *, OfflineProgress) -> bool { throw 7; };
    EXPECT_TRUE(processor.makePlan(source, features.analysis, options, job).status == OfflineStatus::CallbackFailed);
    source.failRead = true;
    EXPECT_TRUE(processor.makePlan(source, features.analysis, options).status == OfflineStatus::SourceReadFailed);
    source.failRead = false;
    BeatSink<float> sink;
    sink.failWrite = true;
    EXPECT_TRUE(processor.render(source, prepared.plan, sink).status == OfflineStatus::SinkFailed);
    EXPECT_TRUE(sink.aborted && !sink.committed);
    BeatSink<float> cancelledSink;
    job.progress = [](void *, OfflineProgress p) { return p.phase != OfflinePhase::Render; };
    EXPECT_TRUE(processor.render(source, prepared.plan, cancelledSink, job).status == OfflineStatus::Cancelled);
    EXPECT_TRUE(cancelledSink.aborted && !cancelledSink.committed);
}

DSPARK_TEST(OfflineBeat_invalid_options_and_nonfinite_audio_preserve_existing_output)
{
    OfflineBeatCompressor<float> processor;
    auto input = beatSong<float>(8000, 4);
    AudioBuffer<float> output;
    output.resize(1, 1);
    output.getChannel(0)[0] = .37f;
    for (int kind = 0; kind < 7; ++kind)
    {
        OfflineBeatCompressor<float>::Options options;
        options.reductionDb = 6;
        if (kind == 0) options.reductionDb = -1;
        if (kind == 1) options.reductionDb = 19;
        if (kind == 2) options.manualBpm = std::numeric_limits<double>::quiet_NaN();
        if (kind == 3) options.manualBpm = 19;
        if (kind == 4) options.releaseNote = static_cast<OfflineBeatCompressor<float>::Note>(90);
        if (kind == 5) options.tempo.minimumConfidence = 2;
        if (kind == 6) options.tempo.minimumBpm = 500;
        EXPECT_TRUE(processor.run(input, output, 8000, options).status == OfflineStatus::InvalidInput);
        EXPECT_EQ(output.getNumSamples(), 1);
        EXPECT_EQ(output.getChannel(0)[0], .37f);
    }
    input.getChannel(1)[911] = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(processor.run(input, output, 8000).status == OfflineStatus::NonFiniteInput);
    EXPECT_EQ(output.getChannel(0)[0], .37f);
}

DSPARK_TEST(OfflineBeat_full_reduction_endpoint_and_near_nyquist_signal_reference)
{
    constexpr int rate = 8000;
    AudioBuffer<double> input;
    input.resize(1, 18000);
    input.clear();
    for (int beat = 0; beat < 4; ++beat)
        for (int i = 0; i < 320; ++i)
        {
            const double t = double(i) / rate;
            input.getChannel(0)[beat * 4000 + i] = (beat == 0 ? .8 : .08) *
                std::exp(-80 * t) * std::cos(twoPi<double> * 3992 * t);
        }
    OfflineBufferSource<double> source(input.toView(), rate);
    OfflineBeatCompressor<double> processor;
    OfflineBeatCompressor<double>::Options options;
    options.reductionDb = 18;
    options.manualBpm = 120;
    const auto prepared = processor.analyze(source, options);
    EXPECT_TRUE(prepared.succeeded());
    EXPECT_EQ(prepared.plan.getReport().maximumReductionDb, 18.);
    EXPECT_EQ(prepared.plan.pulses().size(), std::size_t(4));
    EXPECT_NEAR(prepared.plan.getReport().referencePeakDb, 20 * std::log10(.08), 1e-12);
    std::vector<double> x(static_cast<std::size_t>(input.getNumSamples())), gain(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x[i] = input.getChannel(0)[i];
        gain[i] = prepared.plan.gainAt(static_cast<std::int64_t>(i));
    }
    BeatSink<double> sink;
    const auto result = processor.render(source, prepared.plan, sink);
    const auto delta = dspark::test::offline_reference::referenceDelta(
        x, gain, result.report.renderInfo.leftBoundaryGain, result.report.renderInfo.rightBoundaryGain);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.report.renderInfo.boundaryCalibrated);
    EXPECT_TRUE(result.report.renderInfo.leftBoundaryGain < gain.front());
    EXPECT_NEAR(result.report.renderInfo.rightBoundaryGain, gain.back(), 2e-12);
    EXPECT_TRUE(result.report.renderInfo.minimumControlGain >= std::pow(10., -.9) - 1e-15);
    EXPECT_NEAR(20 * std::log10(.8 / std::abs(sink.audio.getChannel(0)[0])), 18., .05);
    for (std::size_t i = 0; i < x.size(); ++i)
        EXPECT_NEAR(sink.audio.getChannel(0)[i], x[i] + delta[i], 3e-12);
}

DSPARK_TEST(OfflineBeat_linked_control_blocks_are_identical_after_arbitrary_reloads)
{
    const OfflineJobOptions options;
    detail::OfflineSession job(options);
    detail::OfflineAttenuation curve;
    curve.prepare(job, 3, 31, 3500, true);
    curve.set(0, 0, 700, .2, 3500);
    curve.set(1, 6200, 6700, .7, 4700);
    curve.set(2, 10500, 10900, .4, 2300);
    detail::OfflineAttenuation::Cursor sequential(curve, job), reloaded(curve, job);
    std::array<double, 16384> expected{};
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(expected.size()); ++i)
        expected[static_cast<std::size_t>(i)] = sequential(i);
    for (int block : {3, 0, 2, 1, 2, 3, 0, 1, 2, 3})
        for (int i = block * 4096; i < (block + 1) * 4096; ++i)
            EXPECT_EQ(reloaded(i), expected[static_cast<std::size_t>(i)]);
    detail::OfflinePunchEnvelope punch;
    punch.prepare(job, 3, 48000);
    punch.set(0, 3700, 400);
    punch.set(1, 7900, 1500);
    punch.set(2, 12100, 700);
    detail::OfflinePunchEnvelope::Cursor forwardPunch(punch, job), reloadPunch(punch, job);
    for (std::int64_t i = 0; i < static_cast<std::int64_t>(expected.size()); ++i)
        expected[static_cast<std::size_t>(i)] = forwardPunch(i);
    for (int block : {3, 0, 2, 1, 2, 3, 0, 1, 2, 3})
        for (int i = block * 4096; i < (block + 1) * 4096; ++i)
            EXPECT_EQ(reloadPunch(i), expected[static_cast<std::size_t>(i)]);
}

DSPARK_TEST(OfflineBeat_truncated_rising_pulses_keep_their_onset_and_full_reduction)
{
    const auto check = []<class T>() {
        for (const auto& configuration : std::array<std::pair<int, double>, 3>{
                 {{8000, .499}, {48000, .25}, {192000, .499}}})
        {
            const int rate = configuration.first, period = rate / 2;
            const int burst = static_cast<int>(.024 * rate), n = 4 * period + burst;
            AudioBuffer<T> input, output;
            input.resize(2, n);
            input.clear();
            for (int beat = 0; beat < 5; ++beat)
                for (int i = 0; i < burst; ++i)
                {
                    const int phase = beat == 4 ? burst - 1 - i : i;
                    const T x = static_cast<T>((beat == 0 || beat == 4 ? .8 : .08) *
                        std::exp(-80. * phase / rate) *
                        std::cos(twoPi<double> * configuration.second * phase));
                    input.getChannel(0)[beat * period + i] = x;
                    input.getChannel(1)[beat * period + i] = -x;
                }
            OfflineBufferSource<T> source(input.toView(), rate);
            OfflineBeatCompressor<T> processor;
            typename OfflineBeatCompressor<T>::Options options;
            options.reductionDb = 18;
            options.manualBpm = 120;
            const auto plan = processor.analyze(source, options);
            EXPECT_TRUE(plan.succeeded());
            EXPECT_EQ(plan.plan.pulses().size(), std::size_t(5));
            EXPECT_TRUE(std::abs(plan.plan.pulses().back().begin - 4 * period) <= rate / 1000);
            EXPECT_EQ(plan.plan.pulses().back().end, n);
            BeatSink<T> sink;
            const auto result = processor.render(source, plan.plan, sink);
            EXPECT_TRUE(result.succeeded());
            double left = 0, right = 0;
            for (int i = 0; i < burst; ++i)
            {
                left = std::max(left, std::abs(static_cast<double>(sink.audio.getChannel(0)[i])));
                right = std::max(right, std::abs(static_cast<double>(sink.audio.getChannel(0)[n - 1 - i])));
            }
            EXPECT_NEAR(20 * std::log10(.8 / left), 18., .05);
            EXPECT_NEAR(20 * std::log10(.8 / right), 18., .05);
            for (int i = 0; i < n; ++i)
                EXPECT_EQ(sink.audio.getChannel(1)[i], -sink.audio.getChannel(0)[i]);
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}
