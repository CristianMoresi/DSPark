// DSPark - Shared complete-source tempo contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Analysis/OfflineTempoAnalyzer.h"
#include "../Core/ProcessorTraits.h"
#include "dspark_test.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace dspark;
static_assert(!AudioProcessor<OfflineTempoAnalyzer<float>, float>);

namespace
{
template <typename T>
AudioBuffer<T> tempoFixture(int rate, int channels = 2, bool gap = false,
                            double bpm = 120, int seconds = 16)
{
    AudioBuffer<T> audio;
    audio.resize(channels, seconds * rate + 37);
    audio.clear();
    const int beatCount = static_cast<int>((seconds - 1) * bpm / 60);
    for (int beat = 0; beat < beatCount; ++beat)
    {
        if (gap && beat >= 10 && beat <= 17)
            continue;
        const int begin = static_cast<int>((0.25 + 60 * beat / bpm) * rate);
        for (int i = 0; i < rate / 20; ++i)
        {
            const double time = static_cast<double>(i) / rate;
            const T value = static_cast<T>((0.3 + 0.05 * (beat % 3)) * std::exp(-80 * time) *
                                          (std::cos(2 * pi<double> * 113 * time) +
                                           0.5 * std::cos(2 * pi<double> * 2113 * time)));
            for (int c = 0; c < channels; ++c)
                audio.getChannel(c)[begin + i] = c == 1 ? -value : value;
        }
    }
    return audio;
}

template <typename T> void compareLegacyFeed()
{
    for (const int rate : {8000, 48000, 192000})
    {
        auto audio = tempoFixture<T>(rate, 1);
        BeatTracker<T> legacy;
        legacy.prepare({static_cast<double>(rate), 4096, 1});
        const auto expected = legacy.analyze(audio.toView());
        OnsetDetector<T> frontend;
        frontend.prepare({static_cast<double>(rate), 4096, 1});
        const int hop = frontend.getHopSize();
        std::vector<typename OnsetDetector<T>::OdfFrame> frames;
        for (int first = 0; first + hop <= audio.getNumSamples(); first += hop)
        {
            frontend.pushSamples({audio.getChannel(0) + first, static_cast<std::size_t>(hop)});
            frames.push_back(frontend.getLastOdfFrame());
        }
        OfflineJobOptions options;
        detail::OfflineSession job(options);
        const auto actual = detail::OfflineBeatEngine<T>::analyze(
            rate, hop, frames.size(), [&](auto i) { return frames[i]; }, 40, 240, 25, job);
        EXPECT_EQ(actual.tempoBpm, expected.tempoBpm);
        EXPECT_EQ(actual.secondaryTempoBpm, expected.secondaryTempoBpm);
        EXPECT_EQ(actual.confidence, expected.confidence);
        EXPECT_TRUE(std::equal(actual.beats().begin(), actual.beats().end(),
                               expected.beatSamples.begin(), expected.beatSamples.end()));
        EXPECT_GT(job.bytes(), std::size_t(0));
        EXPECT_NEAR(static_cast<double>(actual.tempoBpm), 120.0, 0.1);
    }
}

template <typename T> class CountingSource final : public OfflineAudioSource<T>
{
  public:
    explicit CountingSource(AudioBuffer<T> &audio, int rate)
        : input_(audio.toView(), rate, 99, 7, (std::int64_t(1) << 54) + 13)
    {
    }
    OfflineAudioSpec getSpec() const noexcept override { return input_.getSpec(); }
    bool read(std::int64_t first, AudioBufferView<T> block) override
    {
        readFrames += block.getNumSamples();
        maximumRead = std::max(maximumRead, block.getNumSamples());
        return input_.read(first, block);
    }
    std::int64_t readFrames = 0;
    int maximumRead = 0;

  private:
    OfflineBufferSource<T> input_;
};

template <typename T> void checkSharedTempo()
{
    auto input = tempoFixture<T>(48000);
    CountingSource<T> source(input, 48000);
    OfflineJobOptions job;
    job.blockFrames = 257;
    const auto features = OfflineTransientAnalyzer<T>().analyze(source, {true, true, true}, job);
    EXPECT_TRUE(features.succeeded());
    const auto before = source.readFrames;
    const auto reused = OfflineTempoAnalyzer<T>().analyze(features.analysis, {}, job);
    EXPECT_TRUE(reused.succeeded());
    EXPECT_EQ(source.readFrames, before);
    const auto direct = OfflineTempoAnalyzer<T>().analyze(source, {}, job);
    EXPECT_TRUE(direct.succeeded());
    if (!reused.succeeded() || !direct.succeeded())
        return;
    EXPECT_EQ(source.readFrames - before, 2 * source.getSpec().frames);
    EXPECT_TRUE(source.maximumRead <= 257);
    EXPECT_TRUE((reused.analysis.getSpec()) == (source.getSpec()));
    EXPECT_TRUE((reused.analysis.fingerprint()) == (features.analysis.energy().fingerprint()));
    const auto report = reused.analysis.getReport();
    std::cout << "tempo " << report.tempoBpm << " confidence " << report.confidence
              << " reliable " << report.reliableIntervals << '\n';
    EXPECT_NEAR(report.tempoBpm, 120, 0.1);
    EXPECT_GT(report.reliableIntervals, std::size_t(20));
    EXPECT_TRUE((report.reason) == (OfflineTempoAnalyzer<T>::Reason::Tracked));
    EXPECT_EQ(report.tempoBpm, direct.analysis.getReport().tempoBpm);
    EXPECT_EQ(report.confidence, direct.analysis.getReport().confidence);
    const auto a = reused.analysis.beats(), b = direct.analysis.beats();
    EXPECT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
    {
        EXPECT_EQ(a[i].frame, b[i].frame);
        EXPECT_EQ(a[i].nextIntervalBpm, b[i].nextIntervalBpm);
        EXPECT_EQ(a[i].reliable, b[i].reliable);
        EXPECT_TRUE(a[i].frame >= 0 && a[i].frame < source.getSpec().frames);
        if (a[i].reliable && i > 2 && i + 3 < a.size())
            EXPECT_NEAR(60000.0 / a[i].nextIntervalBpm, 500, 2.0);
    }
    EXPECT_EQ(reused.analysis.tempoAt(-1), 0.0);
    EXPECT_TRUE((reused.analysis.tempoAt(source.getSpec().frames)) == (0.0));
    EXPECT_GT(reused.analysis.tempoAt(5 * 48000), 0.0);
}
}

DSPARK_TEST(OfflineTempo_SharedEngineMatchesLegacyFloatAndDouble)
{
    compareLegacyFeed<float>();
    compareLegacyFeed<double>();
}

DSPARK_TEST(OfflineTempo_PooledFeaturesReuseSourceClockAndTwoScans)
{
    checkSharedTempo<float>();
    checkSharedTempo<double>();
}

DSPARK_TEST(OfflineTempo_StereoPolarityChannelSwapAndSourceGain)
{
    auto input = tempoFixture<double>(8000);
    const auto run = [&] {
        OfflineBufferSource<double> source(input.toView(), 8000);
        return OfflineTempoAnalyzer<double>().analyze(source);
    };
    const auto antiphase = run();
    EXPECT_TRUE(antiphase.succeeded());
    for (int i = 0; i < input.getNumSamples(); ++i)
        input.getChannel(1)[i] = input.getChannel(0)[i];
    const auto dual = run();
    EXPECT_TRUE(dual.succeeded());
    EXPECT_EQ(dual.analysis.getReport().tempoBpm, antiphase.analysis.getReport().tempoBpm);
    EXPECT_EQ(dual.analysis.getReport().confidence, antiphase.analysis.getReport().confidence);
    for (int i = 0; i < input.getNumSamples(); ++i)
    {
        input.getChannel(1)[i] *= 1e250;
        input.getChannel(0)[i] = 0;
    }
    const auto right = run();
    EXPECT_TRUE(right.succeeded());
    for (int i = 0; i < input.getNumSamples(); ++i)
        std::swap(input.getChannel(0)[i], input.getChannel(1)[i]);
    const auto left = run();
    EXPECT_TRUE(left.succeeded());
    EXPECT_EQ(right.analysis.getReport().tempoBpm, left.analysis.getReport().tempoBpm);
    EXPECT_EQ(right.analysis.getReport().confidence, left.analysis.getReport().confidence);
    EXPECT_NEAR(left.analysis.getReport().tempoBpm, 120, 0.1);
}

DSPARK_TEST(OfflineTempo_SilentGapAndUncertainIntervalsDoNotInventReleaseTempo)
{
    auto input = tempoFixture<float>(8000, 2, true);
    OfflineBufferSource<float> source(input.toView(), 8000);
    auto result = OfflineTempoAnalyzer<float>().analyze(source);
    EXPECT_TRUE(result.succeeded());
    EXPECT_NEAR(result.analysis.getReport().tempoBpm, 120, 0.2);
    EXPECT_GT(result.analysis.tempoAt(3 * 8000), 0.0);
    EXPECT_EQ(result.analysis.tempoAt(7 * 8000), 0.0);
    EXPECT_GT(result.analysis.tempoAt(12 * 8000), 0.0);
    EXPECT_EQ(result.analysis.tempoAt(16 * 8000), 0.0);
    input.clear();
    const auto silent = OfflineTempoAnalyzer<float>().analyze(source);
    EXPECT_TRUE(silent.succeeded());
    EXPECT_TRUE((silent.analysis.getReport().reason) == (OfflineTempoAnalyzer<float>::Reason::NoSignal));
    EXPECT_TRUE(silent.analysis.beats().empty());
    EXPECT_EQ(silent.analysis.tempoAt(100), 0.0);
    auto moved = std::move(result.analysis);
    EXPECT_FALSE(result.analysis.isValid());
    EXPECT_TRUE(moved.isValid());
}

DSPARK_TEST(OfflineTempo_BudgetsCancellationAndInvalidFeaturesPublishNothing)
{
    auto input = tempoFixture<double>(8000);
    OfflineBufferSource<double> source(input.toView(), 8000);
    const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, true});
    EXPECT_TRUE(features.succeeded());
    const auto baseline = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    EXPECT_TRUE(baseline.succeeded());
    OfflineJobOptions job;
    job.memoryBudgetBytes = baseline.memoryBytes;
    EXPECT_TRUE(OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job).succeeded());
    --job.memoryBudgetBytes;
    const auto limited = OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job);
    EXPECT_TRUE((limited.status) == (OfflineStatus::MemoryLimit));
    EXPECT_FALSE(limited.analysis.isValid());
    job = {};
    std::atomic<bool> cancel{true};
    job.cancel = &cancel;
    EXPECT_TRUE((OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job).status) == (OfflineStatus::Cancelled));
    job = {};
    for (int stop : {1, 5, 50, 300})
    {
        int remaining = stop;
        job.progressContext = &remaining;
        job.progress = [](void *p, OfflineProgress) {
            return --*static_cast<int *>(p) > 0;
        };
        const auto stopped = OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job);
        EXPECT_TRUE((stopped.status) == (OfflineStatus::Cancelled));
        EXPECT_FALSE(stopped.analysis.isValid());
    }
    job = {};
    job.progress = [](void *, OfflineProgress) -> bool { throw 7; };
    EXPECT_TRUE((OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job).status) == (OfflineStatus::CallbackFailed));
    const auto notRetained = OfflineTransientAnalyzer<double>().analyze(source);
    EXPECT_TRUE((OfflineTempoAnalyzer<double>().analyze(notRetained.analysis).status) == (OfflineStatus::InvalidInput));
    for (const auto options : {OfflineTempoAnalyzer<double>::Options{0, 240, 0.25},
                               {240, 40, 0.25}, {40, 481, 0.25}, {40, 240, -1},
                               {40, 240, std::numeric_limits<double>::quiet_NaN()}})
        EXPECT_TRUE((OfflineTempoAnalyzer<double>().analyze(features.analysis, options).status) == (OfflineStatus::InvalidInput));
}

DSPARK_TEST(OfflineTempo_AccountingAllocatorHonorsAlignmentAndGrowth)
{
    struct alignas(64) Value { double data[8]{}; };
    std::size_t charged = 0;
    detail::AllocationAccount account{&charged, [](void *p, std::size_t bytes) {
                                         *static_cast<std::size_t *>(p) += bytes;
                                     }};
    std::vector<Value, detail::AccountedAllocator<Value>> values{
        detail::AccountedAllocator<Value>{&account}};
    values.resize(100);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values.data()) % alignof(Value), std::uintptr_t(0));
    values[0].data[0] = 3;
    const auto before = charged;
    values.resize(1000);
    EXPECT_GT(charged, before);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(values.data()) % alignof(Value), std::uintptr_t(0));
    EXPECT_EQ(values[0].data[0], 3.0);
    auto copy = values;
    EXPECT_EQ(copy[0].data[0], 3.0);
    EXPECT_TRUE(copy.get_allocator() == values.get_allocator());
    const auto *original = values.data();
    account.request = [](void *, std::size_t) { throw std::bad_alloc(); };
    bool rejected = false;
    try { values.resize(100000); }
    catch (const std::bad_alloc &) { rejected = true; }
    EXPECT_TRUE(rejected);
    EXPECT_TRUE(values.data() == original);
    EXPECT_EQ(values.size(), std::size_t(1000));
}

DSPARK_TEST(OfflineTempo_SearchEndpointsAndLocalTimingAvoidSpuriousFallback)
{
    for (int rate : {8000, 48000, 384000})
        for (double bpm : {20.0, 240.0})
        {
            auto input = tempoFixture<double>(rate, 1, false, bpm, 24);
            OfflineBufferSource<double> source(input.toView(), rate);
            OfflineTempoAnalyzer<double>::Options options{20, 240, 0.25};
            const auto result = OfflineTempoAnalyzer<double>().analyze(source, options);
            EXPECT_TRUE(result.succeeded());
            EXPECT_NEAR(result.analysis.getReport().tempoBpm, bpm, 0.02);
            const auto map = result.analysis.beats();
            EXPECT_GT(map.size(), std::size_t(4));
            for (std::size_t i = 1; i + 2 < map.size(); ++i)
            {
                const auto midpoint = map[i].frame + (map[i + 1].frame - map[i].frame) / 2;
                if (result.analysis.tempoAt(midpoint) == 0 ||
                    std::abs(60000 / map[i].nextIntervalBpm - 60000 / bpm) > 2)
                    std::cout << "endpoint rate=" << rate << " bpm=" << bpm << " interval=" << i
                              << " frames=" << map[i + 1].frame - map[i].frame
                              << " localMs=" << 60000 / map[i].nextIntervalBpm
                              << " confidence=" << result.analysis.getReport().confidence << '\n';
                EXPECT_GT(result.analysis.tempoAt(midpoint), 0.0);
                EXPECT_NEAR(60000 / map[i].nextIntervalBpm, 60000 / bpm, 2.0);
            }
        }
}
