// DSPark - Offline macro leveling and source/transaction contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "dspark_test.h"
#include "offline_gain_reference.h"
#include "../Effects/OfflineLeveler.h"
#include "../Core/ProcessorTraits.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace dspark;

static_assert(!AudioProcessor<OfflineLeveler<float>, float>);
static_assert(!AudioProcessor<OfflineEnergyAnalyzer<float>, float>);

namespace
{

template <typename T> AudioBuffer<T> levelerSong(int rate, int channels = 2)
{
    AudioBuffer<T> audio;
    audio.resize(channels, 36 * rate);
    constexpr double amplitudes[]{0.04, 0.16, 0.08};
    constexpr double frequencies[]{173, 511, 997};
    for (int f = 0; f < audio.getNumSamples(); ++f)
    {
        const int part = f / (12 * rate);
        const auto x = static_cast<T>(amplitudes[part] *
                                      std::sin(6.283185307179586 * frequencies[part] * f / rate));
        for (int c = 0; c < channels; ++c)
            audio.getChannel(c)[f] = c == 0 ? x : -x;
    }
    return audio;
}

template <typename T> double sectionDb(const AudioBuffer<T> &audio, int rate, int from, int to)
{
    double sum = 0;
    for (int i = from * rate; i < to * rate; ++i)
        sum += static_cast<double>(audio.getChannel(0)[i]) * audio.getChannel(0)[i];
    return 10 * std::log10(sum / ((to - from) * rate));
}

template <typename T> std::vector<T> offlineSnapshot(const AudioBuffer<T> &audio)
{
    std::vector<T> out;
    for (int c = 0; c < audio.getNumChannels(); ++c)
        out.insert(out.end(), audio.getChannel(c), audio.getChannel(c) + audio.getNumSamples());
    return out;
}

template <typename T>
bool offlineUnchanged(const AudioBuffer<T> &audio, const std::vector<T> &before)
{
    const auto after = offlineSnapshot(audio);
    return before.size() == after.size() &&
           std::memcmp(before.data(), after.data(), before.size() * sizeof(T)) == 0;
}

template <typename T> void verifyLevelerAmounts()
{
    constexpr int rate = 8000;
    auto in = levelerSong<T>(rate);
    const auto original = offlineSnapshot(in);
    OfflineLeveler<T> leveler;
    const double gap = sectionDb(in, rate, 15, 21) - sectionDb(in, rate, 3, 9);
    for (double amount : {0.0, 0.25, 0.5, 1.0})
    {
        AudioBuffer<T> out;
        typename OfflineLeveler<T>::Options options;
        options.amount = amount;
        const auto result = leveler.run(in, out, rate, options);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(offlineUnchanged(in, original));
        EXPECT_EQ(out.getNumSamples(), in.getNumSamples());
        EXPECT_EQ(out.getNumChannels(), 2);
        const double remaining = sectionDb(out, rate, 15, 21) - sectionDb(out, rate, 3, 9);
        EXPECT_NEAR(remaining, (1 - amount) * gap, 0.3);
        EXPECT_TRUE(result.report.minimumGainDb >= 0.0);
        if (amount == 0)
            EXPECT_TRUE(offlineUnchanged(out, original));
        if (amount == 1)
            EXPECT_GT(sectionDb(out, rate, 3, 9) - sectionDb(in, rate, 3, 9), 3.0);
        for (int f = 0; f < out.getNumSamples(); ++f)
            EXPECT_EQ(out.getChannel(0)[f], -out.getChannel(1)[f]);
    }
}

class OfflineTestSink final : public OfflineAudioSink<float>
{
  public:
    bool begun = false, committed = false, aborted = false;
    bool failWrite = false;
    std::int64_t written = 0;
    bool begin(const OfflineAudioSpec &) override
    {
        begun = true;
        return true;
    }
    bool write(std::int64_t, AudioBufferView<const float> block) override
    {
        written += block.getNumSamples();
        return !failWrite;
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

class OfflineVirtualSource final : public OfflineAudioSource<float>
{
  public:
    OfflineAudioSpec spec{48000, 2, 0};
    int reads = 0;
    bool failRead = false;
    OfflineAudioSpec getSpec() const noexcept override
    {
        return spec;
    }
    bool read(std::int64_t, AudioBufferView<float> block) override
    {
        ++reads;
        if (failRead)
            return false;
        block.clear();
        return true;
    }
};

} // namespace

DSPARK_TEST(OfflineLeveler_IndependentSections_Float)
{
    verifyLevelerAmounts<float>();
}
DSPARK_TEST(OfflineLeveler_IndependentSections_Double)
{
    verifyLevelerAmounts<double>();
}

DSPARK_TEST(OfflineLeveler_ExclusionsAreExactAndCanonical)
{
    constexpr int rate = 8000;
    auto in = levelerSong<float>(rate);
    OfflineBufferSource<float> source(in.toView(), rate);
    const std::array<OfflineRegion, 3> excluded{
        {{6 * rate, 7 * rate}, {5 * rate, 6 * rate}, {6 * rate, 7 * rate}}};
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    options.exclusions = excluded;
    OfflineLeveler<float> leveler;
    auto prepared = leveler.analyze(source, options);
    EXPECT_TRUE(prepared.succeeded());
    EXPECT_EQ(prepared.plan.exclusions().size(), std::size_t(1));
    EXPECT_GT(prepared.plan.gainAt(3LL * rate), 1.0);
    for (int f = 5 * rate; f < 7 * rate; ++f)
        EXPECT_EQ(prepared.plan.gainAt(f), 1.0);
    EXPECT_EQ(prepared.plan.gainAt(-1), 1.0);
    EXPECT_EQ(prepared.plan.gainAt(in.getNumSamples()), 1.0);
    AudioBuffer<float> out;
    EXPECT_TRUE(leveler.run(in, out, rate, options).succeeded());
    for (int c = 0; c < 2; ++c)
        EXPECT_EQ(std::memcmp(in.getChannel(c) + 5 * rate, out.getChannel(c) + 5 * rate,
                              2 * rate * sizeof(float)),
                  0);
}

DSPARK_TEST(OfflineLeveler_AnalysisReuseAndChunkIndependence)
{
    auto in = levelerSong<float>(8000);
    OfflineBufferSource<float> source(in.toView(), 8000);
    OfflineEnergyAnalyzer<float> analyzer;
    OfflineJobOptions small, large;
    small.blockFrames = 17;
    large.blockFrames = 8192;
    auto a = analyzer.analyze(source, small);
    auto b = analyzer.analyze(source, large);
    EXPECT_TRUE(a.succeeded() && b.succeeded());
    EXPECT_EQ(a.analysis.bins().size(), std::size_t(360));
    EXPECT_TRUE(a.analysis.fingerprint() == b.analysis.fingerprint());
    for (std::size_t i = 0; i < a.analysis.bins().size(); ++i)
        EXPECT_EQ(a.analysis.bins()[i].rms, b.analysis.bins()[i].rms);
    OfflineLeveler<float> leveler;
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    auto plan = leveler.makePlan(a.analysis, options);
    EXPECT_TRUE(plan.succeeded());
    EXPECT_EQ(plan.plan.controlPoints(), std::size_t(361));
    EXPECT_LT(plan.plan.retainedBytes(), std::size_t(4000));
    options.amount = 0.5;
    auto half = leveler.makePlan(a.analysis, options);
    EXPECT_NEAR(20 * std::log10(half.plan.gainAt(6 * 8000)), 6.0206, 0.01);
    EXPECT_GT(plan.plan.gainAt(6 * 8000), half.plan.gainAt(6 * 8000));
}

DSPARK_TEST(OfflineLeveler_RaggedRendersAndInPlacePublication)
{
    auto in = levelerSong<float>(8000);
    OfflineLeveler<float> leveler;
    OfflineLeveler<float>::Options options;
    options.amount = 0.7;
    OfflineJobOptions a, b;
    a.blockFrames = 97;
    b.blockFrames = 4096;
    AudioBuffer<float> first;
    EXPECT_TRUE(leveler.run(in, first, 8000, options, a).succeeded());
    const auto expected = offlineSnapshot(first);
    EXPECT_TRUE(leveler.run(in, in, 8000, options, b).succeeded());
    EXPECT_TRUE(offlineUnchanged(in, expected));
}

DSPARK_TEST(OfflineLeveler_NoHiddenPeakLimitAndUpwardControl)
{
    auto in = levelerSong<double>(8000, 1);
    for (int f = 0; f < in.getNumSamples(); ++f)
        in.getChannel(0)[f] *= 8;
    AudioBuffer<double> out;
    OfflineLeveler<double>::Options options;
    options.amount = 1;
    auto result = OfflineLeveler<double>().run(in, out, 8000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_GT(result.report.outputTruePeakDb, 0.0);
    OfflineBufferSource<double> source(in.toView(), 8000);
    const auto plan = OfflineLeveler<double>().analyze(source, options);
    EXPECT_TRUE(plan.succeeded());
    const auto reference =
        test::offline_reference::render(in, plan.plan, result.report.renderInfo);
    double error = 0, power = 0;
    for (int f = 0; f < in.getNumSamples(); ++f)
    {
        // Upward automation is a control contract. The signal oracle includes
        // the baseband projection, which need not increase each sample modulus.
        EXPECT_TRUE(std::abs(in.getChannel(0)[f] * plan.plan.gainAt(f)) >=
                    std::abs(in.getChannel(0)[f]));
        error += std::pow(out.getChannel(0)[f] - reference.getChannel(0)[f], 2);
        power += in.getChannel(0)[f] * in.getChannel(0)[f];
    }
    EXPECT_LT(error / power, 1e-20);

    // A finite, full-scale-for-the-TYPE spike must not hide lost correction.
    // This is a representability limit, not an audio peak ceiling at 0 dBFS.
    in = levelerSong<double>(8000, 1);
    for (int f = 0; f < in.getNumSamples(); ++f)
        in.getChannel(0)[f] *= 1e308;
    in.getChannel(0)[8000 * 6] = std::numeric_limits<double>::max();
    result = OfflineLeveler<double>().run(in, out, 8000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_TRUE(result.report.reason == OfflineLeveler<double>::Reason::RepresentabilityLimited);
    EXPECT_GT(result.report.representabilityLimitedBins, std::size_t(0));
    EXPECT_EQ(out.getChannel(0)[8000 * 6], std::numeric_limits<double>::max());
    for (int f = 0; f < out.getNumSamples(); ++f)
        EXPECT_TRUE(std::isfinite(out.getChannel(0)[f]));
}

DSPARK_TEST(OfflineLeveler_NoActivityIsAnExplicitNoChange)
{
    AudioBuffer<float> in, out;
    in.resize(1, 8000);
    in.getChannel(0)[53] = -0.0f;
    const auto original = offlineSnapshot(in);
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    auto result = OfflineLeveler<float>().run(in, out, 8000, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(result.report.reason == OfflineLeveler<float>::Reason::NoActivity);
    EXPECT_TRUE(offlineUnchanged(out, original));

    in = levelerSong<float>(8000, 1);
    const auto activeOriginal = offlineSnapshot(in);
    options.maximumBoostDb = 0;
    result = OfflineLeveler<float>().run(in, out, 8000, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(result.report.reason == OfflineLeveler<float>::Reason::BoostDisabled);
    EXPECT_TRUE(offlineUnchanged(out, activeOriginal));
}

DSPARK_TEST(OfflineLeveler_InvalidInputPreservesDestination)
{
    auto in = levelerSong<float>(8000, 1);
    AudioBuffer<float> out;
    out.resize(1, 3);
    out.getChannel(0)[1] = 0.123f;
    const auto original = offlineSnapshot(out);
    OfflineLeveler<float> leveler;
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    in.getChannel(0)[31] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(leveler.run(in, out, 8000, options).status == OfflineStatus::NonFiniteInput);
    EXPECT_TRUE(offlineUnchanged(out, original));
    in.getChannel(0)[31] = 0;
    options.amount = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(leveler.run(in, out, 8000, options).status == OfflineStatus::InvalidInput);
    EXPECT_TRUE(offlineUnchanged(out, original));
    options.amount = 1;
    EXPECT_TRUE(leveler.run(in, out, 0, options).status == OfflineStatus::InvalidInput);
}

DSPARK_TEST(OfflineLeveler_CancelledRenderCannotPublish)
{
    auto in = levelerSong<float>(8000);
    AudioBuffer<float> out;
    out.resize(1, 2);
    out.getChannel(0)[0] = 0.4f;
    const auto original = offlineSnapshot(out);
    OfflineJobOptions job;
    job.progress = [](void *, OfflineProgress p) {
        return p.phase != OfflinePhase::Render || p.completed == 0;
    };
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    auto result = OfflineLeveler<float>().run(in, out, 8000, options, job);
    EXPECT_TRUE(result.status == OfflineStatus::Cancelled);
    EXPECT_TRUE(offlineUnchanged(out, original));
    std::atomic<bool> cancelled{true};
    job.progress = nullptr;
    job.cancel = &cancelled;
    EXPECT_TRUE(OfflineLeveler<float>().run(in, out, 8000, options, job).status ==
                OfflineStatus::Cancelled);
}

DSPARK_TEST(OfflineLeveler_StaleSourceRejectedBeforeSinkBegin)
{
    auto in = levelerSong<float>(8000);
    OfflineBufferSource<float> source(in.toView(), 8000, 19, 1);
    OfflineLeveler<float> leveler;
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    auto prepared = leveler.analyze(source, options);
    EXPECT_TRUE(prepared.succeeded());
    in.getChannel(0)[100] += 0.01f;
    OfflineTestSink sink;
    EXPECT_TRUE(leveler.render(source, prepared.plan, sink).status ==
                OfflineStatus::SourceMismatch);
    EXPECT_FALSE(sink.begun);
    OfflineBufferSource<float> other(in.toView(), 8000, 19, 2);
    EXPECT_TRUE(leveler.render(other, prepared.plan, sink).status == OfflineStatus::SourceMismatch);
}

DSPARK_TEST(OfflineLeveler_BoundedMemoryAndLongSourcePreflight)
{
    OfflineVirtualSource source;
    source.spec.frames = std::numeric_limits<std::int64_t>::max();
    OfflineJobOptions job;
    job.memoryBudgetBytes = 1024;
    auto analyzed = OfflineEnergyAnalyzer<float>().analyze(source, job);
    EXPECT_TRUE(analyzed.status == OfflineStatus::MemoryLimit);
    EXPECT_EQ(source.reads, 0);
    EXPECT_FALSE(analyzed.analysis.isValid());
    source.spec.frames = 100;
    source.spec.channels = 3;
    EXPECT_TRUE(OfflineLeveler<float>().analyze(source).status == OfflineStatus::UnsupportedLayout);
}

DSPARK_TEST(OfflineLeveler_SourceAndSinkFailuresAbort)
{
    OfflineVirtualSource broken;
    broken.spec.frames = 10;
    broken.failRead = true;
    auto failed = OfflineEnergyAnalyzer<float>().analyze(broken);
    EXPECT_TRUE(failed.status == OfflineStatus::SourceReadFailed);
    EXPECT_FALSE(failed.analysis.isValid());
    auto in = levelerSong<float>(8000, 1);
    OfflineBufferSource<float> source(in.toView(), 8000);
    OfflineLeveler<float> leveler;
    auto plan = leveler.analyze(source);
    EXPECT_TRUE(plan.succeeded());
    OfflineTestSink sink;
    sink.failWrite = true;
    EXPECT_TRUE(leveler.render(source, plan.plan, sink).status == OfflineStatus::SinkFailed);
    EXPECT_TRUE(sink.begun && sink.aborted);
    EXPECT_FALSE(sink.committed);
}

DSPARK_TEST(OfflineEnergy_FinalBinAndExtremeFiniteValues)
{
    AudioBuffer<double> in;
    in.resize(1, 801);
    for (int i = 0; i < 800; ++i)
        in.getChannel(0)[i] = 1e200;
    in.getChannel(0)[800] = 1e-200;
    OfflineBufferSource<double> source(in.toView(), 8000);
    auto result = OfflineEnergyAnalyzer<double>().analyze(source);
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(result.analysis.bins().size(), std::size_t(2));
    EXPECT_EQ(result.analysis.bins()[0].frames, std::int64_t(800));
    EXPECT_EQ(result.analysis.bins()[1].frames, std::int64_t(1));
    EXPECT_NEAR(result.analysis.bins()[0].rms / 1e200, 1.0, 1e-14);
    EXPECT_NEAR(result.analysis.bins()[1].rms / 1e-200, 1.0, 1e-14);
}

DSPARK_TEST(OfflineLeveler_ContinuousCrescendoAndTransientContrast)
{
    constexpr int rate = 8000;
    AudioBuffer<double> ramp;
    ramp.resize(1, 30 * rate);
    for (int f = 0; f < ramp.getNumSamples(); ++f)
        ramp.getChannel(0)[f] = 0.02 * std::pow(10.0, 0.6 * f / ramp.getNumSamples()) *
                                std::sin(twoPi<double> * 173 * f / rate);
    const double gap = sectionDb(ramp, rate, 24, 27) - sectionDb(ramp, rate, 3, 6);
    OfflineLeveler<double> leveler;
    for (double amount : {0.25, 0.5, 1.0})
    {
        OfflineLeveler<double>::Options options;
        options.amount = amount;
        AudioBuffer<double> out;
        EXPECT_TRUE(leveler.run(ramp, out, rate, options).succeeded());
        EXPECT_NEAR(sectionDb(out, rate, 24, 27) - sectionDb(out, rate, 3, 6), gap * (1 - amount),
                    0.15);
    }
    for (int f = 0; f < ramp.getNumSamples(); ++f)
    {
        const double t = static_cast<double>(f % (rate / 2)) / rate;
        ramp.getChannel(0)[f] =
            (0.008 + 0.15 * std::exp(-50 * t)) * std::sin(twoPi<double> * 220 * f / rate);
    }
    OfflineBufferSource<double> source(ramp.toView(), rate);
    for (double speed : {0.0, 0.5, 1.0})
    {
        OfflineLeveler<double>::Options options;
        options.amount = 1;
        options.speed = speed;
        auto plan = leveler.analyze(source, options);
        EXPECT_TRUE(plan.succeeded());
        for (int beat = 6; beat < 42; ++beat)
        {
            const std::int64_t first = static_cast<std::int64_t>(beat) * rate / 2;
            EXPECT_NEAR(gainToDecibels(plan.plan.gainAt(first)),
                        gainToDecibels(plan.plan.gainAt(first + rate / 25)), 0.12);
        }
    }
}

DSPARK_TEST(OfflineLeveler_PlannedGridBoundAndRenderedEnergyOracle)
{
    constexpr int rate = 8000, hop = rate / 10, window = rate * 3;
    AudioBuffer<double> in;
    in.resize(1, 12 * rate + 1);
    for (int f = 0; f < in.getNumSamples(); ++f)
    {
        const double envelope = f < 3 * rate ? 0.04 : f < 7 * rate ? 0.17 : 0.065;
        in.getChannel(0)[f] = envelope * std::sin(twoPi<double> * 173 * f / rate);
    }
    in.getChannel(0)[in.getNumSamples() - 1] = 0.9;
    const auto prefix = [](const AudioBuffer<double> &audio) {
        std::vector<double> p(static_cast<std::size_t>(audio.getNumSamples()) + 1);
        for (int f = 0; f < audio.getNumSamples(); ++f)
            p[static_cast<std::size_t>(f) + 1] =
                p[static_cast<std::size_t>(f)] + audio.getChannel(0)[f] * audio.getChannel(0)[f];
        return p;
    };
    const auto original = prefix(in);
    double reference = 0;
    const int bins = (in.getNumSamples() - 1) / hop + 1;
    for (int b = 0; b + 30 <= bins; ++b)
    {
        const int first = b * hop, end = std::min(in.getNumSamples(), first + window);
        reference = std::max(reference, (original[static_cast<std::size_t>(end)] -
                                         original[static_cast<std::size_t>(first)]) /
                                            (end - first));
    }
    OfflineLeveler<double>::Options options;
    options.amount = 1;
    AudioBuffer<double> out;
    const auto rendered = OfflineLeveler<double>().run(in, out, rate, options);
    EXPECT_TRUE(rendered.succeeded());
    OfflineBufferSource<double> source(in.toView(), rate);
    const auto plan = OfflineLeveler<double>().analyze(source, options);
    EXPECT_TRUE(plan.succeeded());
    AudioBuffer<double> planned;
    planned.resize(1, in.getNumSamples());
    for (int f = 0; f < in.getNumSamples(); ++f)
        planned.getChannel(0)[f] = in.getChannel(0)[f] * plan.plan.gainAt(f);
    const auto plannedEnergy = prefix(planned);
    const auto projected =
        test::offline_reference::render(in, plan.plan, rendered.report.renderInfo);
    const auto referenceEnergy = prefix(projected);
    const auto processed = prefix(out);
    for (int b = 0; b + 30 <= bins; ++b)
    {
        const int first = b * hop, end = std::min(out.getNumSamples(), first + window);
        const auto windowPower = [&](const auto &energy) {
            return (energy[static_cast<std::size_t>(end)] -
                    energy[static_cast<std::size_t>(first)]) / (end - first);
        };
        EXPECT_TRUE(windowPower(plannedEnergy) <= reference * (1 + 1e-10));
        EXPECT_NEAR(windowPower(processed), windowPower(referenceEnergy), reference * 1e-10);
    }
}

DSPARK_TEST(OfflineLeveler_MusicRatesShareMacroLevelContract)
{
    for (int rate : {44100, 48000, 88200, 96000, 176400, 192000})
    {
        auto in = levelerSong<float>(rate, 1);
        OfflineBufferSource<float> source(in.toView(), rate);
        OfflineLeveler<float>::Options options;
        options.amount = 1;
        auto result = OfflineLeveler<float>().analyze(source, options);
        EXPECT_TRUE(result.succeeded());
        EXPECT_NEAR(gainToDecibels(result.plan.gainAt(6LL * rate)), 12.0412, 0.01);
        EXPECT_NEAR(gainToDecibels(result.plan.gainAt(18LL * rate)), 0.0, 0.01);
        EXPECT_NEAR(gainToDecibels(result.plan.gainAt(30LL * rate)), 6.0206, 0.01);
        EXPECT_EQ(result.plan.controlPoints(), std::size_t(361));
    }
}

DSPARK_TEST(OfflineLeveler_ExclusionsDoNotMoveReferenceAndAllExcludedIsIdentity)
{
    constexpr int rate = 8000;
    auto in = levelerSong<double>(rate, 1);
    OfflineBufferSource<double> source(in.toView(), rate);
    OfflineLeveler<double> leveler;
    OfflineLeveler<double>::Options options;
    options.amount = 1;
    auto original = leveler.analyze(source, options);
    const std::array<OfflineRegion, 1> loud{{{12 * rate, 24 * rate}}};
    options.exclusions = loud;
    auto excluded = leveler.analyze(source, options);
    EXPECT_TRUE(original.succeeded() && excluded.succeeded());
    EXPECT_EQ(original.plan.getReport().referenceRmsDb, excluded.plan.getReport().referenceRmsDb);
    EXPECT_EQ(original.plan.gainAt(6LL * rate), excluded.plan.gainAt(6LL * rate));
    const std::array<OfflineRegion, 1> all{{{0, 36 * rate}}};
    options.exclusions = all;
    AudioBuffer<double> out;
    auto result = leveler.run(in, out, rate, options);
    EXPECT_TRUE(result.status == OfflineStatus::NoChange);
    EXPECT_TRUE(result.report.reason == OfflineLeveler<double>::Reason::AllExcluded);
    EXPECT_TRUE(offlineUnchanged(out, offlineSnapshot(in)));
}

DSPARK_TEST(OfflineLeveler_MovedPlanIsInvalidAndHugeTimelineOriginIsPreserved)
{
    auto in = levelerSong<float>(8000, 1);
    constexpr std::int64_t origin = std::int64_t(1) << 42;
    OfflineBufferSource<float> source(in.toView(), 8000, 71, 7, origin);
    auto first = OfflineLeveler<float>().analyze(source);
    EXPECT_TRUE(first.succeeded());
    auto second = std::move(first.plan);
    EXPECT_FALSE(first.plan.isValid());
    EXPECT_EQ(first.plan.gainAt(10), 1.0);
    EXPECT_EQ(first.plan.controlPoints(), std::size_t(0));
    EXPECT_EQ(second.getSpec().timelineOrigin, origin);
    OfflineTestSink sink;
    EXPECT_TRUE(OfflineLeveler<float>().render(source, second, sink).succeeded());
    EXPECT_TRUE(sink.committed && !sink.aborted);
    EXPECT_EQ(sink.written, std::int64_t(in.getNumSamples()));
}

DSPARK_TEST(OfflineLeveler_CallbackFailureAndMutationDuringRenderCannotCommit)
{
    auto in = levelerSong<float>(8000, 1);
    OfflineBufferSource<float> source(in.toView(), 8000);
    OfflineLeveler<float> leveler;
    auto prepared = leveler.analyze(source);
    EXPECT_TRUE(prepared.succeeded());
    OfflineJobOptions job;
    job.progress = [](void *, OfflineProgress) -> bool { throw 1; };
    auto failed = leveler.analyze(source, {}, job);
    EXPECT_TRUE(failed.status == OfflineStatus::CallbackFailed);
    EXPECT_FALSE(failed.plan.isValid());
    job.progressContext = &in;
    job.progress = [](void *data, OfflineProgress p) {
        if (p.phase == OfflinePhase::Render && p.completed == 0)
            static_cast<AudioBuffer<float> *>(data)->getChannel(0)[0] = 0.75f;
        return true;
    };
    OfflineTestSink sink;
    auto result = leveler.render(source, prepared.plan, sink, job);
    EXPECT_TRUE(result.status == OfflineStatus::SourceMismatch);
    EXPECT_TRUE(sink.begun && sink.aborted && !sink.committed);
}

DSPARK_TEST(OfflineLeveler_ShortInputAndTruePeakTail)
{
    AudioBuffer<double> in, out;
    in.resize(1, 1);
    in.getChannel(0)[0] = 0.25;
    OfflineLeveler<double>::Options options;
    options.amount = 1;
    auto result = OfflineLeveler<double>().run(in, out, 48000, options);
    EXPECT_TRUE(result.succeeded());
    EXPECT_EQ(out.getChannel(0)[0], in.getChannel(0)[0]);
    in.resize(1, 101);
    for (int f = 0; f < 101; ++f)
        in.getChannel(0)[f] = 0.5 * std::sin(twoPi<double> * 0.25 * f + pi<double> / 4);
    options.amount = 0;
    result = OfflineLeveler<double>().run(in, out, 48000, options);
    EXPECT_TRUE(result.succeeded());
    TruePeakDetector<double, 1> oracle;
    double peak = 0;
    for (int f = 0; f < 101; ++f)
        peak = std::max(peak, oracle.processSample(in.getChannel(0)[f], 0));
    for (int f = 0; f < TruePeakDetector<double, 1>::getTaps() - 1; ++f)
        peak = std::max(peak, oracle.processSample(0, 0));
    EXPECT_NEAR(result.report.outputTruePeakDb, gainToDecibels(peak), 1e-11);
    EXPECT_GT(result.report.outputTruePeakDb, result.report.outputSamplePeakDb + 2.0);
}
