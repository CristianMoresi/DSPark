// DSPark - Complete-source transient analysis contracts.
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Analysis/OfflineTransientAnalyzer.h"
#include "../Core/ProcessorTraits.h"
#include "dspark_test.h"
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace dspark;

static_assert(!AudioProcessor<OfflineTransientAnalyzer<float>, float>);

namespace
{
constexpr std::array<double, 8> eventTimes{0.0, 0.32, 0.70, 1.1, 1.5, 1.85, 2.22, 2.65};
constexpr std::array<double, 8> eventLengths{0.012, 0.035, 0.065, 0.1, 0.02, 0.08, 0.045, 0.009};

template <typename T>
AudioBuffer<T> transientFixture(int rate, int channels = 2, bool antiphase = true)
{
    AudioBuffer<T> buffer;
    buffer.resize(channels, 3 * rate + 1);
    buffer.clear();
    for (std::size_t e = 0; e < eventTimes.size(); ++e)
    {
        const int begin = static_cast<int>(std::lround(eventTimes[e] * rate));
        const int length = static_cast<int>(std::lround(eventLengths[e] * rate));
        for (int f = 0; f < length; ++f)
        {
            const double t = static_cast<double>(f) / rate;
            const T value = static_cast<T>(0.18 * (std::cos(6.283185307179586 * 313 * t) +
                                                   std::cos(6.283185307179586 * 997 * t) +
                                                   std::cos(6.283185307179586 * 2701 * t)));
            for (int c = 0; c < channels; ++c)
                buffer.getChannel(c)[begin + f] = antiphase && c == 1 ? -value : value;
        }
    }
    return buffer;
}

template <typename T> void checkAnnotatedEvents()
{
    OfflineTransientAnalyzer<T> analyzer;
    for (int rate : {8000, 44100, 48000, 96000, 192000, 384000})
    {
        auto input = transientFixture<T>(rate);
        OfflineBufferSource<T> source(input.toView(), rate, 123, 456, 9000000000LL);
        const auto result = analyzer.analyze(source);
        EXPECT_TRUE(result.succeeded());
        if (!result.succeeded())
            continue;
        for (auto map : {result.analysis.attacks(), result.analysis.pulses()})
        {
            EXPECT_EQ(map.size(), eventTimes.size());
            if (map.size() != eventTimes.size())
            {
                std::cout << "rate " << rate << " events";
                for (auto e : map)
                    std::cout << " " << double(e.begin) / rate << ":" << double(e.end) / rate;
                std::cout << "\n";
                continue;
            }
            for (std::size_t i = 0; i < map.size(); ++i)
            {
                const auto &e = map[i];
                EXPECT_NEAR(static_cast<double>(e.begin) / rate, eventTimes[i], 0.002);
                EXPECT_NEAR(static_cast<double>(e.end) / rate, eventTimes[i] + eventLengths[i],
                            0.004);
                EXPECT_TRUE(e.begin <= e.peak && e.peak < e.end);
                EXPECT_GT(e.peakAmplitude, 0.4);
                EXPECT_TRUE(e.strength > 0 && e.strength <= 1);
                EXPECT_TRUE(e.contrast > 0 && e.contrast <= 1);
                EXPECT_FALSE(e.endLimited);
            }
        }
        EXPECT_EQ(result.analysis.energy().getSpec().timelineOrigin, 9000000000LL);
        EXPECT_TRUE(result.analysis.features().empty());
        EXPECT_LT(result.analysis.retainedBytes(), std::size_t(6000));
    }
}

template <typename T>
bool sameEvents(std::span<const typename OfflineTransientAnalyzer<T>::Event> a,
                std::span<const typename OfflineTransientAnalyzer<T>::Event> b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i].begin != b[i].begin || a[i].peak != b[i].peak || a[i].end != b[i].end ||
            a[i].strength != b[i].strength || a[i].contrast != b[i].contrast ||
            a[i].peakAmplitude != b[i].peakAmplitude || a[i].sustainRms != b[i].sustainRms ||
            a[i].endLimited != b[i].endLimited || a[i].overlapsNext != b[i].overlapsNext)
            return false;
    return true;
}

class MutableTransientSource final : public OfflineAudioSource<float>
{
  public:
    OfflineAudioSpec spec{8000, 1, 8000, 4, 7, 0};
    int reads = 0;
    bool changed = false, fail = false;
    OfflineAudioSpec getSpec() const noexcept override
    {
        return spec;
    }
    bool read(std::int64_t first, AudioBufferView<float> block) override
    {
        if (fail)
            return false;
        if (first == 0 && reads > 0)
            changed = true;
        ++reads;
        for (int f = 0; f < block.getNumSamples(); ++f)
            block.getChannel(0)[f] = (first + f) % 1000 < 10 ? (changed ? 0.5f : 1.0f) : 0.0f;
        return true;
    }
};
} // namespace

DSPARK_TEST(OfflineTransient_AnnotatedIntervals_Float)
{
    checkAnnotatedEvents<float>();
}
DSPARK_TEST(OfflineTransient_AnnotatedIntervals_Double)
{
    checkAnnotatedEvents<double>();
}

DSPARK_TEST(OfflineTransient_Polarity_Chunking_And_ChannelSwap)
{
    auto input = transientFixture<float>(48000);
    OfflineTransientAnalyzer<float> analyzer;
    OfflineBufferSource<float> source(input.toView(), 48000);
    OfflineTransientAnalyzer<float>::Options options;
    options.retainFeatures = true;
    const auto original = analyzer.analyze(source, options);
    EXPECT_TRUE(original.succeeded());
    if (!original.succeeded())
        return;
    EXPECT_GT(original.analysis.features().size(), std::size_t(500));
    for (int size : {1, 17, 4093, 999999})
    {
        OfflineJobOptions job;
        job.blockFrames = size;
        const auto result = analyzer.analyze(source, options, job);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(sameEvents<float>(original.analysis.attacks(), result.analysis.attacks()));
        EXPECT_TRUE(sameEvents<float>(original.analysis.pulses(), result.analysis.pulses()));
        EXPECT_TRUE(original.analysis.energy().fingerprint() ==
                    result.analysis.energy().fingerprint());
        EXPECT_EQ(original.analysis.features().size(), result.analysis.features().size());
        if (original.analysis.features().size() == result.analysis.features().size())
            for (std::size_t i = 0; i < result.analysis.features().size(); ++i)
            {
                EXPECT_EQ(original.analysis.features()[i].attack,
                          result.analysis.features()[i].attack);
                EXPECT_EQ(original.analysis.features()[i].pulse,
                          result.analysis.features()[i].pulse);
            }
    }
    for (int i = 0; i < input.getNumSamples(); ++i)
        input.getChannel(1)[i] = input.getChannel(0)[i];
    const auto dualMono = analyzer.analyze(source);
    EXPECT_TRUE(sameEvents<float>(original.analysis.attacks(), dualMono.analysis.attacks()));
    EXPECT_TRUE(sameEvents<float>(original.analysis.pulses(), dualMono.analysis.pulses()));
    for (int i = 0; i < input.getNumSamples(); ++i)
        input.getChannel(0)[i] = 0;
    const auto right = analyzer.analyze(source);
    for (int i = 0; i < input.getNumSamples(); ++i)
        std::swap(input.getChannel(0)[i], input.getChannel(1)[i]);
    const auto left = analyzer.analyze(source);
    EXPECT_EQ(right.analysis.attacks().size(), eventTimes.size());
    EXPECT_EQ(right.analysis.pulses().size(), eventTimes.size());
    EXPECT_TRUE(sameEvents<float>(right.analysis.attacks(), left.analysis.attacks()));
    EXPECT_TRUE(sameEvents<float>(right.analysis.pulses(), left.analysis.pulses()));
}

DSPARK_TEST(OfflineTransient_First_Last_And_Short_Impulse)
{
    OfflineTransientAnalyzer<double> analyzer;
    for (int length : {1, 2, 31, 511, 8193})
    {
        for (int at : {0, length - 1})
        {
            AudioBuffer<double> input;
            input.resize(1, length);
            input.clear();
            input.getChannel(0)[at] = 1;
            OfflineBufferSource<double> source(input.toView(), 48000);
            auto result = analyzer.analyze(source);
            EXPECT_TRUE(result.succeeded());
            for (auto events : {result.analysis.attacks(), result.analysis.pulses()})
            {
                EXPECT_EQ(events.size(), std::size_t(1));
                if (events.size() != 1)
                    continue;
                EXPECT_EQ(events[0].begin, at);
                EXPECT_EQ(events[0].peak, at);
                EXPECT_TRUE(events[0].end > at && events[0].end <= length);
            }
        }
    }
}

DSPARK_TEST(OfflineTransient_Stationary_And_Vibrato_Do_Not_Invent_Attacks)
{
    for (double base : {50.0, 440.0, 4000.0})
    {
        constexpr int rate = 48000;
        AudioBuffer<float> input;
        input.resize(1, 3 * rate);
        double phase = 0;
        for (int i = 0; i < input.getNumSamples(); ++i)
        {
            phase += 6.283185307179586 * base *
                     (1 + 0.03 * std::sin(6.283185307179586 * 6 * i / rate)) / rate;
            const double tremolo = 0.8 + 0.2 * std::sin(6.283185307179586 * 5 * i / rate);
            input.getChannel(0)[i] = static_cast<float>(0.5 * std::sin(phase) * tremolo);
        }
        OfflineBufferSource<float> source(input.toView(), rate);
        auto result = OfflineTransientAnalyzer<float>().analyze(source);
        EXPECT_TRUE(result.succeeded());
        for (auto event : result.analysis.attacks())
            EXPECT_LT(event.begin, std::int64_t(rate / 20));
    }
}

DSPARK_TEST(OfflineTransient_Extreme_Finite_Scaling_And_Feature_Selection)
{
    auto input = transientFixture<double>(8000, 1);
    OfflineBufferSource<double> source(input.toView(), 8000);
    OfflineTransientAnalyzer<double> analyzer;
    auto baseline = analyzer.analyze(source);
    EXPECT_TRUE(baseline.succeeded());
    for (double scale : {1e-250, 1e250})
    {
        auto scaled = transientFixture<double>(8000, 1);
        for (int i = 0; i < scaled.getNumSamples(); ++i)
            scaled.getChannel(0)[i] *= scale;
        OfflineBufferSource<double> scaledSource(scaled.toView(), 8000);
        const auto result = analyzer.analyze(scaledSource);
        EXPECT_TRUE(result.succeeded());
        EXPECT_EQ(result.analysis.attacks().size(), baseline.analysis.attacks().size());
        if (result.analysis.attacks().size() == baseline.analysis.attacks().size())
            for (std::size_t i = 0; i < baseline.analysis.attacks().size(); ++i)
            {
                EXPECT_EQ(result.analysis.attacks()[i].begin, baseline.analysis.attacks()[i].begin);
                EXPECT_EQ(result.analysis.attacks()[i].end, baseline.analysis.attacks()[i].end);
                EXPECT_TRUE(std::isfinite(result.analysis.attacks()[i].peakAmplitude));
            }
    }
    const auto energyOnly = analyzer.analyze(source, {false, false, false});
    const auto energy = OfflineEnergyAnalyzer<double>().analyze(source);
    EXPECT_TRUE(energyOnly.succeeded());
    EXPECT_EQ(energyOnly.memoryBytes, energy.memoryBytes);
    EXPECT_TRUE(energyOnly.analysis.attacks().empty());
    EXPECT_TRUE(energyOnly.analysis.pulses().empty());
    auto moved = std::move(baseline.analysis);
    EXPECT_TRUE(moved.isValid());
    EXPECT_FALSE(baseline.analysis.isValid());
    EXPECT_TRUE(baseline.analysis.attacks().empty());
}

DSPARK_TEST(OfflineTransient_Resource_Source_And_Cancellation_Failures)
{
    OfflineTransientAnalyzer<float> analyzer;
    MutableTransientSource mutableSource;
    auto mismatch = analyzer.analyze(mutableSource);
    EXPECT_TRUE(mismatch.status == OfflineStatus::SourceMismatch);
    EXPECT_FALSE(mismatch.analysis.isValid());
    mutableSource.reads = 0;
    OfflineJobOptions job;
    job.memoryBudgetBytes = 128;
    auto limited = analyzer.analyze(mutableSource, {}, job);
    EXPECT_TRUE(limited.status == OfflineStatus::MemoryLimit);
    EXPECT_EQ(mutableSource.reads, 0);
    mutableSource.spec.frames = std::numeric_limits<std::int64_t>::max() - 100;
    auto huge = analyzer.analyze(mutableSource);
    EXPECT_TRUE(huge.status == OfflineStatus::MemoryLimit);
    EXPECT_EQ(mutableSource.reads, 0);
    mutableSource.spec.frames = 8000;
    std::atomic<bool> cancel{true};
    job = {};
    job.cancel = &cancel;
    auto cancelled = analyzer.analyze(mutableSource, {}, job);
    EXPECT_TRUE(cancelled.status == OfflineStatus::Cancelled);
    EXPECT_EQ(mutableSource.reads, 0);
    auto input = transientFixture<float>(8000);
    input.getChannel(1)[17] = std::numeric_limits<float>::quiet_NaN();
    OfflineBufferSource<float> source(input.toView(), 8000);
    auto invalid = analyzer.analyze(source);
    EXPECT_TRUE(invalid.status == OfflineStatus::NonFiniteInput);
    EXPECT_FALSE(invalid.analysis.isValid());
}

DSPARK_TEST(OfflineTransient_Bass_Attacks_And_Dense_Pulses)
{
    constexpr int rate = 48000;
    for (bool bass : {false, true})
    {
        AudioBuffer<double> input;
        input.resize(2, 3 * rate);
        input.clear();
        for (int event = 0; event < 16; ++event)
        {
            const int begin = 137 + 7000 * event;
            const int duration = bass ? 5500 : 2400;
            for (int f = 0; f < duration; ++f)
            {
                const double seconds = static_cast<double>(f) / rate;
                const double attack = f < 240 ? 0.5 - 0.5 * std::cos(pi<double> * f / 240) : 1;
                const double x = bass ? attack * std::exp(-seconds / 0.035) *
                                            std::sin(twoPi<double> * 70 * seconds)
                                      : 0.3 * (std::cos(twoPi<double> * 431 * seconds) +
                                               std::cos(twoPi<double> * 2731 * seconds));
                input.getChannel(0)[begin + f] += x;
                input.getChannel(1)[begin + f] -= x;
            }
        }
        OfflineBufferSource<double> source(input.toView(), rate);
        auto result = OfflineTransientAnalyzer<double>().analyze(source);
        EXPECT_TRUE(result.succeeded());
        for (auto events : {result.analysis.attacks(), result.analysis.pulses()})
        {
            EXPECT_EQ(events.size(), std::size_t(16));
            if (events.size() != 16)
                continue;
            for (std::size_t i = 0; i < events.size(); ++i)
            {
                EXPECT_NEAR(static_cast<double>(events[i].begin), double(137 + 7000 * i),
                            bass ? 240.0 : 96.0);
                EXPECT_GT(events[i].end - events[i].begin, std::int64_t(500));
                EXPECT_LT(events[i].end - events[i].begin, std::int64_t(6000));
            }
        }
    }
}

DSPARK_TEST(OfflineTransient_CloseAndQuietSourceRestarts)
{
    const auto verify = []<class T>() {
        for (int rate : {8000, 44100, 48000, 96000, 192000, 384000})
        {
            AudioBuffer<T> input;
            input.resize(2, rate);
            std::array<int, 17> expected{};
            for (int event = 0; event < 17; ++event)
            {
                const int begin = event < 16 ? static_cast<int>(std::llround(rate * .125) +
                                                                event * std::llround(rate * .03))
                                             : static_cast<int>(std::llround(rate * .85));
                expected[event] = begin;
                const double amplitude = event == 16 ? .005 : event % 3 == 0 ? .1 : .7;
                for (int i = 0; i < static_cast<int>(rate * .004); ++i)
                {
                    const double time = static_cast<double>(i) / rate;
                    const T x = static_cast<T>(amplitude * std::exp(-time / .00125) *
                                               std::cos(twoPi<double> * 1700 * time));
                    input.getChannel(0)[begin + i] = x;
                    input.getChannel(1)[begin + i] = -x;
                }
            }
            OfflineBufferSource<T> source(input.toView(), rate);
            auto result = OfflineTransientAnalyzer<T>().analyze(source);
            EXPECT_TRUE(result.succeeded());
            EXPECT_EQ(result.analysis.attacks().size(), expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                EXPECT_NEAR(static_cast<double>(result.analysis.attacks()[i].begin),
                            static_cast<double>(expected[i]), rate * .001);
        }
    };
    verify.template operator()<float>();
    verify.template operator()<double>();
}

DSPARK_TEST(OfflineTransient_Silence_Invalid_Spec_And_Cancelled_Refinement)
{
    AudioBuffer<float> input;
    input.resize(1, 100);
    input.clear();
    OfflineBufferSource<float> silence(input.toView(), 48000);
    OfflineTransientAnalyzer<float> analyzer;
    auto quiet = analyzer.analyze(silence);
    EXPECT_TRUE(quiet.succeeded());
    EXPECT_TRUE(quiet.analysis.isValid());
    EXPECT_TRUE(quiet.analysis.attacks().empty());
    EXPECT_TRUE(quiet.analysis.pulses().empty());
    for (double rate : {0.0, 7999.0, 384001.0, std::numeric_limits<double>::infinity()})
    {
        OfflineBufferSource<float> source(input.toView(), rate);
        EXPECT_TRUE(analyzer.analyze(source).status == OfflineStatus::InvalidInput);
    }
    input.resize(3, 20);
    OfflineBufferSource<float> surround(input.toView(), 48000);
    EXPECT_TRUE(analyzer.analyze(surround).status == OfflineStatus::UnsupportedLayout);
    input.resize(1, 0);
    OfflineBufferSource<float> empty(input.toView(), 48000);
    EXPECT_TRUE(analyzer.analyze(empty).status == OfflineStatus::EmptyInput);
    input = transientFixture<float>(8000, 1);
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflineJobOptions options;
    options.progress = [](void *, OfflineProgress progress) {
        return progress.phase != OfflinePhase::Plan;
    };
    auto cancelled = analyzer.analyze(source, {}, options);
    EXPECT_TRUE(cancelled.status == OfflineStatus::Cancelled);
    EXPECT_FALSE(cancelled.analysis.isValid());
    options.progress = [](void *, OfflineProgress progress) -> bool {
        if (progress.phase == OfflinePhase::Plan)
            throw 42;
        return true;
    };
    EXPECT_TRUE(analyzer.analyze(source, {}, options).status == OfflineStatus::CallbackFailed);
    auto recovered = analyzer.analyze(source);
    EXPECT_TRUE(recovered.succeeded());
    EXPECT_EQ(recovered.analysis.attacks().size(), eventTimes.size());
}

DSPARK_TEST(OfflineTransient_Maximum_Finite_Input_Produces_Finite_Map)
{
    AudioBuffer<double> input;
    input.resize(2, 48000);
    input.clear();
    for (int i = 0; i < 7000; ++i)
    {
        input.getChannel(0)[i] = std::numeric_limits<double>::max();
        input.getChannel(1)[i] = -std::numeric_limits<double>::max();
    }
    input.getChannel(0)[47999] = std::numeric_limits<double>::max();
    OfflineBufferSource<double> source(input.toView(), 48000);
    auto result = OfflineTransientAnalyzer<double>().analyze(source);
    EXPECT_TRUE(result.succeeded());
    for (auto map : {result.analysis.attacks(), result.analysis.pulses()})
        for (auto event : map)
        {
            EXPECT_TRUE(std::isfinite(event.peakAmplitude));
            EXPECT_TRUE(std::isfinite(event.sustainRms));
        }
}

DSPARK_TEST(OfflineTransient_Soft_Bass_Notes_Over_Sustained_Bed)
{
    for (int rate : {48000, 96000})
    {
        AudioBuffer<float> input;
        input.resize(1, 4 * rate);
        for (int f = 0; f < input.getNumSamples(); ++f)
            input.getChannel(0)[f] =
                static_cast<float>(0.015 * std::sin(twoPi<double> * 113 * f / rate));
        constexpr double notes[]{32.70, 36.71, 41.20, 43.65, 49.00, 55.00, 61.74, 65.40};
        for (int e = 0; e < 8; ++e)
        {
            const int begin = static_cast<int>((0.2 + 0.45 * e) * rate);
            for (int f = 0; f < rate / 5; ++f)
            {
                const double envelope =
                    std::min({1.0, double(f) / (rate / 100), double(rate / 5 - f) / (rate / 100)});
                double value = 0;
                for (int h = 1; h <= 4; ++h)
                    value += std::sin(twoPi<double> * notes[e] * h * f / rate) / h;
                input.getChannel(0)[begin + f] += static_cast<float>(0.2 * envelope * value);
            }
        }
        OfflineBufferSource<float> source(input.toView(), rate);
        auto result = OfflineTransientAnalyzer<float>().analyze(source);
        EXPECT_TRUE(result.succeeded());
        for (auto map : {result.analysis.attacks(), result.analysis.pulses()})
        {
            std::vector<std::int64_t> detected;
            for (auto event : map)
                if (event.begin > rate / 10)
                    detected.push_back(event.begin);
            if (detected.size() != 8)
            {
                std::cout << "soft notes rate " << rate;
                for (auto frame : detected)
                    std::cout << " " << double(frame) / rate;
                std::cout << "\n";
            }
            EXPECT_EQ(detected.size(), std::size_t(8));
            for (std::size_t e = 0; e < detected.size(); ++e)
                EXPECT_NEAR(double(detected[e]) / rate, 0.2 + 0.45 * static_cast<double>(e), 0.010);
        }
    }
}
