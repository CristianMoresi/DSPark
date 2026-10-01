// DSPark Tests -- bounded streaming sample-interval analysis
#include "dspark_test.h"
#include "../Analysis/AudioIntervalAnalyzer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {
template <class T>
using Analyzer = AudioIntervalAnalyzer<T>;

template <class T>
typename Analyzer<T>::Result measure(const std::vector<T>& left, const std::vector<T>& right,
    typename Analyzer<T>::Interval interval, int blockSize,
    typename Analyzer<T>::Context context = Analyzer<T>::Context::Independent,
    std::int64_t origin = 0)
{
    Analyzer<T> analyzer;
    const int channels = right.empty() ? 1 : 2;
    if (!analyzer.prepare({ 48000, blockSize, channels }, interval, context, origin))
        throw std::runtime_error("unexpected invalid test setup");
    for (size_t offset = 0; offset < left.size(); offset += static_cast<size_t>(blockSize))
    {
        const int count = static_cast<int>(std::min(left.size() - offset, static_cast<size_t>(blockSize)));
        const T* data[] = { left.data() + offset, right.empty() ? nullptr : right.data() + offset };
        if (!analyzer.processBlock({ data, channels, count }, origin + static_cast<std::int64_t>(offset)))
            throw std::runtime_error("unexpected invalid test block");
    }
    return analyzer.finish();
}

template <class T>
void sameMeasurement(const typename Analyzer<T>::Measurement& a,
                     const typename Analyzer<T>::Measurement& b)
{
    EXPECT_TRUE(a.validity == b.validity);
    EXPECT_TRUE(a.support == b.support);
    if (a.hasValue()) EXPECT_EQ(a.value, b.value);
    else { EXPECT_TRUE(std::isnan(a.value)); EXPECT_TRUE(std::isnan(b.value)); }
}

template <class T>
void sameLoudness(const typename Analyzer<T>::Loudness& a,
                  const typename Analyzer<T>::Loudness& b)
{
    EXPECT_TRUE(a.input == b.input);
    EXPECT_EQ(a.committedEnd, b.committedEnd);
    EXPECT_EQ(a.pendingFrames, b.pendingFrames);
    EXPECT_EQ(a.hopFrames, b.hopFrames);
    EXPECT_EQ(a.integratedBlocks, b.integratedBlocks);
    EXPECT_EQ(a.rangeBlocks, b.rangeBlocks);
    sameMeasurement<T>(a.momentary, b.momentary);
    sameMeasurement<T>(a.shortTerm, b.shortTerm);
    sameMeasurement<T>(a.integrated, b.integrated);
    sameMeasurement<T>(a.range, b.range);
}

template <class T>
void checkPartitions()
{
    using A = Analyzer<T>;
    std::vector<T> left(4 * 48000 + 117), right(left.size());
    for (size_t n = 0; n < left.size(); ++n)
    {
        left[n] = static_cast<T>((static_cast<int>(n % 97) - 48) / 256.0);
        right[n] = static_cast<T>((static_cast<int>(n % 113) - 56) / 512.0);
    }
    // Source-clock precision must not pass through float or double arithmetic.
    constexpr std::int64_t origin = 9007199254740997LL;
    const typename A::Interval range { origin + 37, origin + static_cast<std::int64_t>(left.size()) - 19 };
    const auto reference = measure(left, right, range, 4096, A::Context::IncludeContinuous, origin);
    EXPECT_TRUE(reference.complete());
    EXPECT_TRUE(reference.measured == range);
    for (int block : { 1, 127, 512, 1009 })
    {
        const auto actual = measure(left, right, range, block, A::Context::IncludeContinuous, origin);
        EXPECT_TRUE(actual.complete());
        EXPECT_TRUE(actual.measured == reference.measured);
        sameMeasurement<T>(actual.rms, reference.rms);
        sameMeasurement<T>(actual.samplePeak, reference.samplePeak);
        sameMeasurement<T>(actual.truePeakDbtp, reference.truePeakDbtp);
        sameLoudness<T>(actual.region, reference.region);
        sameLoudness<T>(actual.continuous, reference.continuous);
    }
}
} // namespace

DSPARK_TEST(AudioInterval_partition_independence_float_and_double)
{
    checkPartitions<float>();
    checkPartitions<double>();
}

DSPARK_TEST(AudioInterval_exact_RMS_and_peaks_at_half_open_boundaries)
{
    using A = Analyzer<double>;
    const std::vector<double> left { 0, 0, 10, .25, -.5, .125, 0, -10, 0 };
    const std::vector<double> right { 0, 0, 10, -.25, .25, -.125, 0, -10, 0 };
    const auto result = measure(left, right, { 3, 7 }, 5);
    EXPECT_TRUE(result.complete());
    EXPECT_EQ(result.measured.begin, 3);
    EXPECT_EQ(result.measured.end, 7);
    EXPECT_NEAR(result.rms.value, std::sqrt(0.46875 / 8), 1e-15);
    EXPECT_EQ(result.samplePeak.value, .5);
    EXPECT_LT(result.truePeakDbtp.value, -4.0);
    EXPECT_TRUE(result.region.integrated.validity == A::Validity::TooShort);
    EXPECT_TRUE(result.region.shortTerm.validity == A::Validity::TooShort);
    EXPECT_TRUE(std::isnan(result.region.integrated.value));
    EXPECT_TRUE(result.continuous.integrated.validity == A::Validity::NotRequested);
}

DSPARK_TEST(AudioInterval_finite_tail_matches_direct_meter_without_outside_peaks)
{
    using A = Analyzer<float>;
    std::vector<float> audio(48000 + 14, .05f);
    audio[6] = 20;
    audio[48005] = .9f;
    audio[48006] = -.9f;
    audio[48007] = -20;
    const auto result = measure(audio, {}, { 7, 48007 }, 1024, A::Context::IncludeContinuous);
    EXPECT_TRUE(result.complete());
    EXPECT_EQ(result.samplePeak.value, static_cast<double>(.9f));
    EXPECT_NEAR(result.truePeakDbtp.value, -0.244683381, 0.000002);
    LoudnessMeter<float> direct;
    direct.prepare(48000, 1);
    direct.process(audio.data() + 7, 48000);
    direct.finalizeTruePeak();
    EXPECT_EQ(result.truePeakDbtp.value, direct.getTruePeakDb());
    EXPECT_EQ(result.region.integrated.value, direct.getIntegratedLUFS());
    EXPECT_EQ(result.region.pendingFrames, 0);
    EXPECT_EQ(result.region.committedEnd, 48007);
}

DSPARK_TEST(AudioInterval_empty_short_silent_and_below_gate_are_distinct)
{
    using A = Analyzer<double>;
    const auto empty = measure<double>({}, {}, { 7, 7 }, 512);
    EXPECT_TRUE(empty.complete());
    EXPECT_TRUE(empty.rms.validity == A::Validity::Empty);
    EXPECT_TRUE(std::isnan(empty.rms.value));
    const auto one = measure<double>({ .5 }, {}, { 0, 1 }, 512);
    EXPECT_EQ(one.rms.value, .5);
    EXPECT_NEAR(one.truePeakDbtp.value, -6.020599913279624, 1e-12);
    EXPECT_TRUE(one.region.momentary.validity == A::Validity::TooShort);
    EXPECT_TRUE(one.region.integrated.validity == A::Validity::TooShort);
    std::vector<double> silence(3 * 48000, 0);
    const auto silent = measure(silence, {}, { 0, 3 * 48000 }, 317);
    EXPECT_TRUE(silent.rms.validity == A::Validity::Silence);
    EXPECT_EQ(silent.rms.value, 0);
    EXPECT_TRUE(silent.region.integrated.validity == A::Validity::Silence);
    EXPECT_EQ(silent.region.integrated.value, -100);
    EXPECT_TRUE(silent.region.range.validity == A::Validity::Silence);
    EXPECT_EQ(silent.region.range.value, 0);
    for (size_t n = 0; n < silence.size(); ++n)
        silence[n] = n % 2 ? 1e-8 : -1e-8;
    const auto quiet = measure(silence, {}, { 0, 3 * 48000 }, 317);
    EXPECT_TRUE(quiet.region.integrated.validity == A::Validity::BelowGate);
    EXPECT_TRUE(quiet.region.range.validity == A::Validity::BelowGate);
    EXPECT_TRUE(quiet.region.momentary.validity == A::Validity::Valid);
    EXPECT_EQ(quiet.region.momentary.value, -100);
}

DSPARK_TEST(AudioInterval_window_coverage_and_partial_hop_are_explicit)
{
    using A = Analyzer<double>;
    std::vector<double> audio(144019);
    for (size_t n = 0; n < audio.size(); ++n) audio[n] = n % 2 ? .1 : -.1;
    for (const std::int64_t length : { 19199LL, 19200LL, 143999LL, 144000LL, 144011LL })
    {
        const auto result = measure(audio, {}, { 7, 7 + length }, 613);
        EXPECT_TRUE(result.complete());
        EXPECT_EQ(result.region.committedEnd, 7 + (length / 4800) * 4800);
        EXPECT_EQ(result.region.pendingFrames, length % 4800);
        EXPECT_TRUE(result.region.integrated.hasValue() == (length >= 19200));
        EXPECT_TRUE(result.region.shortTerm.hasValue() == (length >= 144000));
        EXPECT_TRUE(result.region.range.hasValue() == (length >= 144000));
        if (result.region.momentary.hasValue())
        {
            EXPECT_EQ(result.region.momentary.support.end, result.region.committedEnd);
            EXPECT_EQ(result.region.momentary.support.end - result.region.momentary.support.begin, 19200);
        }
        else EXPECT_TRUE(result.region.momentary.validity == A::Validity::TooShort);
    }
}

DSPARK_TEST(AudioInterval_continuous_observations_keep_their_own_scope)
{
    using A = Analyzer<double>;
    std::vector<double> audio(8 * 48000);
    for (size_t n = 0; n < audio.size(); ++n)
        audio[n] = (n < 4 * 48000 ? .4 : .04) *
            std::sin(twoPi<double> * 997.0 * static_cast<double>(n) / 48000.0);
    const auto shortRegion = measure(audio, {}, { 192000, 194400 }, 1024, A::Context::IncludeContinuous);
    EXPECT_TRUE(shortRegion.region.momentary.validity == A::Validity::TooShort);
    EXPECT_TRUE(shortRegion.continuous.momentary.hasValue());
    EXPECT_EQ(shortRegion.continuous.input.begin, 0);
    EXPECT_EQ(shortRegion.continuous.input.end, 194400);
    EXPECT_EQ(shortRegion.continuous.momentary.support.begin, 172800);
    EXPECT_EQ(shortRegion.continuous.momentary.support.end, 192000);
    const auto longRegion = measure(audio, {}, { 192007, 383991 }, 1024, A::Context::IncludeContinuous);
    EXPECT_GT(longRegion.continuous.integrated.value - longRegion.region.integrated.value, 10.0);
    EXPECT_EQ(longRegion.region.input.begin, 192007);
    EXPECT_EQ(longRegion.continuous.input.begin, 0);
    LoudnessMeter<double> reference;
    reference.prepare(48000, 1);
    reference.process(audio.data(), 383991);
    EXPECT_EQ(longRegion.continuous.integrated.value, reference.getIntegratedLUFS());
    EXPECT_EQ(longRegion.continuous.momentary.value, reference.getMomentaryLUFS());
    EXPECT_EQ(longRegion.continuous.shortTerm.value, reference.getShortTermLUFS());
}

DSPARK_TEST(AudioInterval_integrated_uses_gated_energy_not_mean_LUFS)
{
    std::vector<double> audio(32 * 48000);
    for (size_t n = 0; n < audio.size(); ++n)
    {
        const double amplitude = n < 16 * 48000 ? .1 : std::sqrt(.001);
        audio[n] = amplitude * std::sin(twoPi<double> * 997.0 * static_cast<double>(n) / 48000.0);
    }
    const auto result = measure(audio, audio, { 0, 32 * 48000 }, 997);
    // Equal-duration -20 and -30 LUFS sections retain both levels after gating.
    // Their energy mean is -22.596373 LUFS; their incorrect LUFS mean is -25.
    EXPECT_NEAR(result.region.integrated.value, -22.59637310505756, 0.1);
    EXPECT_GT(std::abs(result.region.integrated.value - (-25)), 2.0);
    EXPECT_NEAR(result.region.range.value, 10.0, 0.1);
}

DSPARK_TEST(AudioInterval_truncated_and_invalid_streams_never_report_valid_metrics)
{
    using A = Analyzer<double>;
    const std::vector<double> audio(20, .1);
    const auto shortFile = measure(audio, {}, { 7, 25 }, 9);
    EXPECT_TRUE(shortFile.status == A::Status::Truncated);
    EXPECT_EQ(shortFile.measured.begin, 7);
    EXPECT_EQ(shortFile.measured.end, 20);
    EXPECT_TRUE(shortFile.rms.validity == A::Validity::Incomplete);
    EXPECT_TRUE(std::isnan(shortFile.rms.value));
    A analyzer;
    EXPECT_FALSE(analyzer.prepare({ 48000, 512, 3 }, { 0, 20 }));
    EXPECT_TRUE(analyzer.finish().status == A::Status::InvalidArgument);
    EXPECT_TRUE(analyzer.prepare({ 48000, 512, 1 }, { 0, 20 }));
    const double* channel[] = { audio.data() };
    EXPECT_FALSE(analyzer.processBlock({ channel, 1, 10 }, 1));
    const auto gap = analyzer.finish();
    EXPECT_TRUE(gap.status == A::Status::DiscontinuousInput);
    EXPECT_FALSE(gap.rms.hasValue());
    EXPECT_TRUE(analyzer.finish().status == gap.status);
}

DSPARK_TEST(AudioInterval_nonfinite_only_matters_inside_measured_scopes)
{
    using A = Analyzer<double>;
    std::vector<double> audio(12, .1);
    audio[3] = audio[8] = std::numeric_limits<double>::quiet_NaN();
    const auto isolated = measure(audio, {}, { 4, 8 }, 12);
    EXPECT_TRUE(isolated.complete());
    A analyzer;
    const double* channel[] = { audio.data() };
    EXPECT_TRUE(analyzer.prepare({ 48000, 12, 1 }, { 4, 8 }, A::Context::IncludeContinuous));
    EXPECT_FALSE(analyzer.processBlock({ channel, 1, 12 }, 0));
    EXPECT_TRUE(analyzer.finish().status == A::Status::NonFiniteInput);
    audio[5] = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(analyzer.prepare({ 48000, 12, 1 }, { 4, 8 }));
    EXPECT_FALSE(analyzer.processBlock({ channel, 1, 12 }, 0));
    const auto bad = analyzer.finish();
    EXPECT_TRUE(bad.status == A::Status::NonFiniteInput);
    EXPECT_FALSE(bad.samplePeak.hasValue());
}

DSPARK_TEST(AudioInterval_extreme_finite_RMS_and_numerical_loudness_diagnostic)
{
    using A = Analyzer<double>;
    for (double value : { 1e-200, 1e200 })
    {
        const auto result = measure<double>({ value, -value }, {}, { 0, 2 }, 2);
        EXPECT_TRUE(result.complete());
        EXPECT_EQ(result.rms.value, value);
        EXPECT_EQ(result.samplePeak.value, value);
        if (value > 1)
        {
            EXPECT_TRUE(result.region.integrated.validity == A::Validity::NumericalError);
            EXPECT_FALSE(result.truePeakDbtp.hasValue());
        }
    }
}
