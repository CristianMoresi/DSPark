// DSPark - LoopFinder contract tests

#include "dspark_test.h"
#include "../Analysis/LoopFinder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

using namespace dspark;

namespace {

template <typename T>
AudioBufferView<const T> constView(const std::vector<std::vector<T>>& audio)
{
    std::array<const T*, 16> pointers {};
    for (std::size_t ch = 0; ch < audio.size(); ++ch)
        pointers[ch] = audio[ch].data();
    return { pointers.data(), static_cast<int>(audio.size()),
             audio.empty() ? 0 : static_cast<int>(audio.front().size()) };
}

template <typename T>
AudioBufferView<T> mutableView(std::vector<std::vector<T>>& audio)
{
    std::array<T*, 16> pointers {};
    for (std::size_t ch = 0; ch < audio.size(); ++ch)
        pointers[ch] = audio[ch].data();
    return { pointers.data(), static_cast<int>(audio.size()),
             audio.empty() ? 0 : static_cast<int>(audio.front().size()) };
}

template <typename T>
std::vector<std::vector<T>> plantedSource(int channels = 2)
{
    constexpr int samples = 640;
    constexpr int start = 96;
    constexpr int end = 417;
    constexpr int window = 64;
    std::vector<std::vector<T>> audio(
        static_cast<std::size_t>(channels),
        std::vector<T>(samples, T(0)));
    for (int ch = 0; ch < channels; ++ch)
    {
        for (int n = 0; n < samples; ++n)
        {
            const long double phase = 2.0L * std::numbers::pi_v<long double>
                * (7.31L + 0.73L * ch) * n / 257.0L + 0.19L * ch;
            audio[static_cast<std::size_t>(ch)][static_cast<std::size_t>(n)] =
                static_cast<T>(0.24L * std::sin(phase)
                               + 0.08L * std::sin(2.17L * phase + 0.3L));
        }
        std::copy_n(audio[static_cast<std::size_t>(ch)].begin() + start,
                    window,
                    audio[static_cast<std::size_t>(ch)].begin() + end - window);
    }
    return audio;
}

template <typename T>
typename LoopFinder<T>::Settings exactSettings()
{
    typename LoopFinder<T>::Settings settings;
    settings.comparisonLength = 64;
    settings.crossfadeLength = 16;
    settings.coarseStride = 1;
    settings.refinementRadius = 0;
    settings.maxCoarsePairs = 4096;
    settings.maxRefinementPairs = 4096;
    settings.maximumCost = T(1);
    settings.minimumActivityRms = T(1.0e-5);
    return settings;
}

template <typename T>
void expectNeutral(const typename LoopFinder<T>::Result& result,
                   typename LoopFinder<T>::Status status)
{
    EXPECT_TRUE(result.status == status);
    EXPECT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(result.start, std::int64_t(0));
    EXPECT_EQ(result.end, std::int64_t(0));
    EXPECT_EQ(result.crossfadeLength, 0);
    EXPECT_EQ(result.seamCost, T(1));
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
}

// Independent scalar reimplementation of the published candidate cost.
// It shares no code with the header: means, variances, D0, D1 and the
// Jensen-Shannon spectral term are recomputed in long double, and the
// spectral probabilities come from a direct O(W^2) DFT instead of an FFT.
template <typename T>
struct ScalarCost
{
    bool admissible = false;
    long double cost = 1.0L;
};

template <typename T>
ScalarCost<T> independentCost(const std::vector<std::vector<T>>& audio,
                              std::int64_t start, std::int64_t end,
                              int window, long double activityFloor)
{
    const int channels = static_cast<int>(audio.size());
    long double scale = 0.0L;
    for (int ch = 0; ch < channels; ++ch)
        for (int i = 0; i < window; ++i)
        {
            scale = std::max(scale, std::abs(static_cast<long double>(
                audio[static_cast<std::size_t>(ch)]
                     [static_cast<std::size_t>(end - window + i)])));
            scale = std::max(scale, std::abs(static_cast<long double>(
                audio[static_cast<std::size_t>(ch)]
                     [static_cast<std::size_t>(start + i)])));
        }
    if (scale == 0.0L) return {};

    std::vector<long double> weights;
    std::vector<long double> costs;
    long double varianceSumA = 0.0L;
    long double varianceSumB = 0.0L;
    int included = 0;
    for (int ch = 0; ch < channels; ++ch)
    {
        std::vector<long double> a(static_cast<std::size_t>(window));
        std::vector<long double> b(static_cast<std::size_t>(window));
        long double meanA = 0.0L;
        long double meanB = 0.0L;
        for (int i = 0; i < window; ++i)
        {
            a[static_cast<std::size_t>(i)] = static_cast<long double>(
                audio[static_cast<std::size_t>(ch)]
                     [static_cast<std::size_t>(end - window + i)]) / scale;
            b[static_cast<std::size_t>(i)] = static_cast<long double>(
                audio[static_cast<std::size_t>(ch)]
                     [static_cast<std::size_t>(start + i)]) / scale;
            meanA += a[static_cast<std::size_t>(i)];
            meanB += b[static_cast<std::size_t>(i)];
        }
        meanA /= window;
        meanB /= window;
        long double varianceA = 0.0L;
        long double varianceB = 0.0L;
        for (int i = 0; i < window; ++i)
        {
            const long double da = a[static_cast<std::size_t>(i)] - meanA;
            const long double db = b[static_cast<std::size_t>(i)] - meanB;
            varianceA += da * da;
            varianceB += db * db;
        }
        varianceA = std::clamp(varianceA / window, 0.0L, 1.0L);
        varianceB = std::clamp(varianceB / window, 0.0L, 1.0L);
        const long double rmsA = scale * std::sqrt(varianceA);
        const long double rmsB = scale * std::sqrt(varianceB);
        if (std::max(rmsA, rmsB) < activityFloor) continue;
        ++included;
        varianceSumA += varianceA;
        varianceSumB += varianceB;

        long double d0Numerator = 0.0L;
        long double d0Denominator = 0.0L;
        long double d1Numerator = 0.0L;
        long double d1Denominator = 0.0L;
        for (int i = 0; i < window; ++i)
        {
            const long double difference = a[static_cast<std::size_t>(i)]
                - b[static_cast<std::size_t>(i)];
            d0Numerator += difference * difference;
            d0Denominator += a[static_cast<std::size_t>(i)]
                    * a[static_cast<std::size_t>(i)]
                + b[static_cast<std::size_t>(i)]
                    * b[static_cast<std::size_t>(i)];
            if (i > 0)
            {
                const long double derivativeA = a[static_cast<std::size_t>(i)]
                    - a[static_cast<std::size_t>(i - 1)];
                const long double derivativeB = b[static_cast<std::size_t>(i)]
                    - b[static_cast<std::size_t>(i - 1)];
                const long double derivativeDifference =
                    derivativeA - derivativeB;
                d1Numerator += derivativeDifference * derivativeDifference;
                d1Denominator += derivativeA * derivativeA
                    + derivativeB * derivativeB;
            }
        }
        const long double d0 = d0Denominator == 0.0L
            ? 0.0L : 0.5L * d0Numerator / d0Denominator;
        const long double d1 = d1Denominator == 0.0L
            ? 0.0L : 0.5L * d1Numerator / d1Denominator;

        // Direct DFT of the mean-removed periodic-Hann-windowed windows.
        const int bins = window / 2 + 1;
        std::vector<long double> powerA(static_cast<std::size_t>(bins));
        std::vector<long double> powerB(static_cast<std::size_t>(bins));
        long double totalA = 0.0L;
        long double totalB = 0.0L;
        for (int bin = 0; bin < bins; ++bin)
        {
            long double realA = 0.0L, imagA = 0.0L;
            long double realB = 0.0L, imagB = 0.0L;
            for (int i = 0; i < window; ++i)
            {
                const long double hann = 0.5L - 0.5L
                    * std::cos(2.0L * std::numbers::pi_v<long double> * i
                               / window);
                const long double angle = -2.0L
                    * std::numbers::pi_v<long double> * bin * i / window;
                const long double c = std::cos(angle);
                const long double s = std::sin(angle);
                const long double xa =
                    (a[static_cast<std::size_t>(i)] - meanA) * hann;
                const long double xb =
                    (b[static_cast<std::size_t>(i)] - meanB) * hann;
                realA += xa * c; imagA += xa * s;
                realB += xb * c; imagB += xb * s;
            }
            const long double weight = (bin == 0 || bin == bins - 1)
                ? 1.0L : 2.0L;
            powerA[static_cast<std::size_t>(bin)] =
                weight * (realA * realA + imagA * imagA);
            powerB[static_cast<std::size_t>(bin)] =
                weight * (realB * realB + imagB * imagB);
            totalA += powerA[static_cast<std::size_t>(bin)];
            totalB += powerB[static_cast<std::size_t>(bin)];
        }
        long double djs = 0.0L;
        if ((totalA == 0.0L) != (totalB == 0.0L))
        {
            djs = 1.0L;
        }
        else if (totalA != 0.0L)
        {
            long double divergenceA = 0.0L;
            long double divergenceB = 0.0L;
            for (int bin = 0; bin < bins; ++bin)
            {
                const long double pa =
                    powerA[static_cast<std::size_t>(bin)] / totalA;
                const long double pb =
                    powerB[static_cast<std::size_t>(bin)] / totalB;
                const long double midpoint = (pa + pb) * 0.5L;
                if (pa > 0.0L) divergenceA += pa * std::log(pa / midpoint);
                if (pb > 0.0L) divergenceB += pb * std::log(pb / midpoint);
            }
            djs = std::clamp((divergenceA + divergenceB) * 0.5L
                / std::numbers::ln2_v<long double>, 0.0L, 1.0L);
        }
        weights.push_back(varianceA + varianceB);
        costs.push_back(std::clamp(0.45L * d0 + 0.20L * d1 + 0.35L * djs,
                                   0.0L, 1.0L));
    }
    if (included == 0) return {};
    const long double aggregateRmsA = scale * std::sqrt(
        std::min(1.0L, varianceSumA / included));
    const long double aggregateRmsB = scale * std::sqrt(
        std::min(1.0L, varianceSumB / included));
    if (aggregateRmsA < activityFloor || aggregateRmsB < activityFloor)
        return {};

    std::vector<std::size_t> order(weights.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                  return weights[lhs] < weights[rhs]
                      || (weights[lhs] == weights[rhs]
                          && costs[lhs] < costs[rhs]);
              });
    long double weighted = 0.0L;
    long double totalWeight = 0.0L;
    long double maximum = 0.0L;
    for (const std::size_t index : order)
    {
        weighted += weights[index] * costs[index];
        totalWeight += weights[index];
        maximum = std::max(maximum, costs[index]);
    }
    ScalarCost<T> result;
    result.admissible = true;
    result.cost = std::clamp(0.75L * weighted / totalWeight + 0.25L * maximum,
                             0.0L, 1.0L);
    return result;
}

template <typename T>
void exerciseSearchAndRender()
{
    using Finder = LoopFinder<T>;
    Finder finder;
    auto audio = plantedSource<T>();
    auto settings = exactSettings<T>();
    const auto result = finder.find(constView(audio), { 96, 96 }, { 417, 417 },
                                    321, 321, settings);
    EXPECT_TRUE(static_cast<bool>(result));
    EXPECT_EQ(result.start, std::int64_t(96));
    EXPECT_EQ(result.end, std::int64_t(417));
    EXPECT_EQ(result.crossfadeLength, 16);
    EXPECT_TRUE(result.seamCost <= T(64) * std::numeric_limits<T>::epsilon());
    EXPECT_EQ(result.renderedLength(), std::int64_t(305));

    std::vector<std::vector<T>> output(2, std::vector<T>(305, T(-9)));
    EXPECT_TRUE(Finder::renderLoop(constView(audio), result,
                                   mutableView(output)) == Finder::Status::Success);
    for (int ch = 0; ch < 2; ++ch)
    {
        EXPECT_EQ(output[static_cast<std::size_t>(ch)][0],
                  audio[static_cast<std::size_t>(ch)][401]);
        EXPECT_EQ(output[static_cast<std::size_t>(ch)][15],
                  audio[static_cast<std::size_t>(ch)][111]);
        for (int i = 16; i < 305; ++i)
            EXPECT_EQ(output[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)],
                      audio[static_cast<std::size_t>(ch)][static_cast<std::size_t>(96 + i)]);

        // The expectation is rebuilt from the equal-power curve itself, out of
        // pure operands. A crossfader publishes its gain pair from the
        // processing call, so asking it for the blend and for the gains to
        // divide the blend by inside one expression leaves the two calls
        // unsequenced: the divisor is then whichever pair the compiler
        // happened to read first, and the expectation stops being the same
        // number on every toolchain.
        for (int i = 1; i < 15; ++i)
        {
            const T tail = audio[static_cast<std::size_t>(ch)]
                                [static_cast<std::size_t>(401 + i)];
            const T head = audio[static_cast<std::size_t>(ch)]
                                [static_cast<std::size_t>(96 + i)];
            const T scale = std::max(std::abs(tail), std::abs(head));
            const T position = static_cast<T>(i) / T(15);
            const T gainTail = std::sqrt(T(1) - position);
            const T gainHead = std::sqrt(position);
            const T expected = scale == T(0) ? T(0) : scale * std::clamp(
                (tail / scale * gainTail + head / scale * gainHead)
                    / (gainTail + gainHead),
                T(-1), T(1));
            EXPECT_NEAR(output[static_cast<std::size_t>(ch)]
                              [static_cast<std::size_t>(i)],
                        expected, T(8) * std::numeric_limits<T>::epsilon());
        }
    }
}

// The overlap divides the blend by the sum of the two gains it just applied,
// so a seam between identical constant signals must come back as that
// constant, exactly, at every position of every fade length. Any other divisor
// - a gain pair read before the blend that produced it, for one - is an
// amplitude error here, on every toolchain and with no tolerance to hide in.
template <typename T>
void exerciseConstantSeam()
{
    using Finder = LoopFinder<T>;
    const T level = T(0.37);
    const std::vector<std::vector<T>> audio(2, std::vector<T>(640, level));
    const std::array<int, 5> fades { 2, 3, 16, 64, 160 };
    for (const int fade : fades)
    {
        const typename Finder::Result loop { Finder::Status::Success, 96, 417,
                                             fade, T(0) };
        const int rendered = static_cast<int>(loop.renderedLength());
        EXPECT_EQ(rendered, 321 - fade);

        std::vector<std::vector<T>> output(
            2, std::vector<T>(static_cast<std::size_t>(rendered), T(-9)));
        EXPECT_TRUE(Finder::renderLoop(constView(audio), loop,
                                       mutableView(output))
                    == Finder::Status::Success);
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < rendered; ++i)
                EXPECT_EQ(output[static_cast<std::size_t>(ch)]
                                [static_cast<std::size_t>(i)], level);
    }
}

} // namespace

DSPARK_TEST(LoopFinder_public_api_and_guarded_length)
{
    using Finder = LoopFinder<double>;
    Finder finder;
    Finder::Settings settings;
    Finder::Range range;
    (void)finder; (void)settings; (void)range;

    Finder::Result result;
    EXPECT_FALSE(static_cast<bool>(result));
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
    result.status = Finder::Status::Success;
    result.start = 64;
    result.end = 192;
    result.crossfadeLength = 32;
    EXPECT_EQ(result.renderedLength(), std::int64_t(96));

    // Forged payloads must return zero without signed overflow.
    result.start = std::numeric_limits<std::int64_t>::min();
    result.end = std::numeric_limits<std::int64_t>::max();
    result.crossfadeLength = -1;
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
    result = { Finder::Status::Success,
               std::numeric_limits<std::int64_t>::min(),
               std::numeric_limits<std::int64_t>::max(), -1, 0.0 };
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
    result = { Finder::Status::NoAcceptableLoop, 64, 192, 32, 0.0 };
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
    result = { Finder::Status::Success, 100, 100, 2, 0.0 };
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
    result = { Finder::Status::Success, 0, 3, 2, 0.0 };
    EXPECT_EQ(result.renderedLength(), std::int64_t(0));
    result = { Finder::Status::Success, 0,
               std::numeric_limits<std::int64_t>::max(),
               std::numeric_limits<int>::max(), 0.0 };
    EXPECT_EQ(result.renderedLength(),
              std::numeric_limits<std::int64_t>::max()
                  - std::numeric_limits<int>::max());
}

DSPARK_TEST(LoopFinder_validation_status_precedence)
{
    using Finder = LoopFinder<double>;
    Finder finder;
    auto settings = exactSettings<double>();

    std::vector<std::vector<double>> none;
    expectNeutral<double>(finder.find(constView(none), {}, {}, 1, 1, settings),
                          Finder::Status::EmptyInput);

    std::array<const double*, 1> nullPointer { nullptr };
    AudioBufferView<const double> nullView(nullPointer.data(), 1, 128);
    expectNeutral<double>(finder.find(nullView, { 0, 0 }, { 65, 65 }, 65, 65,
                                      settings), Finder::Status::InvalidArgument);

    auto active = plantedSource<double>(1);
    const auto view = constView(active);

    // Malformed ranges and loop ordering are InvalidArgument.
    expectNeutral<double>(finder.find(view, { -1, 0 }, { 65, 65 }, 65, 65,
                                      settings), Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 5, 4 }, { 65, 65 }, 65, 65,
                                      settings), Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 0, 640 }, { 65, 65 }, 65, 65,
                                      settings), Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 0, 0 }, { 65, 641 }, 65, 65,
                                      settings), Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 0, 0 },
                                      { 0, std::numeric_limits<std::int64_t>::max() },
                                      65, 65, settings),
                          Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 0, 0 }, { 65, 65 }, 0, 65,
                                      settings), Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 0, 0 }, { 65, 65 }, 66, 65,
                                      settings), Finder::Status::InvalidArgument);
    expectNeutral<double>(finder.find(view, { 0, 0 }, { 65, 65 }, 65, 641,
                                      settings), Finder::Status::InvalidArgument);

    // Every invalid setting field is InvalidArgument, never silently clamped.
    for (int field = 0; field < 10; ++field)
    {
        auto bad = settings;
        switch (field)
        {
            case 0: bad.comparisonLength = 63; break;
            case 1: bad.comparisonLength = 8192; break;
            case 2: bad.crossfadeLength = 1; break;
            case 3: bad.crossfadeLength = bad.comparisonLength + 1; break;
            case 4: bad.coarseStride = 0; break;
            case 5: bad.refinementRadius = -1; break;
            case 6: bad.maxCoarsePairs = 0; break;
            case 7: bad.maxRefinementPairs = 2000000; break;
            case 8: bad.maximumCost = std::nan(""); break;
            case 9: bad.minimumActivityRms = 0.0; break;
        }
        expectNeutral<double>(finder.find(view, { 0, 0 }, { 65, 65 }, 65, 65,
                                          bad), Finder::Status::InvalidArgument);
    }

    std::array<const double*, 1> hugePointer { active[0].data() };
    AudioBufferView<const double> huge(hugePointer.data(), 1, (1 << 26) + 1);
    expectNeutral<double>(finder.find(huge, { 0, 0 }, { 65, 65 }, 65, 65,
                                      settings), Finder::Status::InputTooLong);

    auto nonFinite = active;
    nonFinite[0][500] = std::numeric_limits<double>::infinity();
    expectNeutral<double>(finder.find(constView(nonFinite), { 0, 0 }, { 65, 65 },
                                      65, 65, settings),
                          Finder::Status::NonFiniteInput);

    // W=64, L=16: N=64 is InputTooShort; the sole exposed pair P=64 is never
    // scored (NoAcceptableLoop); P=65 is the first legal boundary length.
    std::vector<std::vector<double>> shortInput(1, std::vector<double>(64));
    for (int i = 0; i < 64; ++i) shortInput[0][i] = (i & 1) ? 0.1 : -0.1;
    expectNeutral<double>(finder.find(constView(shortInput), { 0, 0 }, { 64, 64 },
                                      64, 64, settings),
                          Finder::Status::InputTooShort);

    std::vector<std::vector<double>> boundary(1, std::vector<double>(65));
    for (int i = 0; i < 65; ++i) boundary[0][i] = (i & 1) ? 0.1 : -0.1;
    expectNeutral<double>(finder.find(constView(boundary), { 0, 0 }, { 64, 64 },
                                      1, 65, settings),
                          Finder::Status::NoAcceptableLoop);
    const auto firstLegal = finder.find(constView(boundary), { 0, 0 }, { 65, 65 },
                                        1, 65, settings);
    EXPECT_TRUE(firstLegal.status == Finder::Status::Success);
    EXPECT_EQ(firstLegal.start, std::int64_t(0));
    EXPECT_EQ(firstLegal.end, std::int64_t(65));
    EXPECT_EQ(firstLegal.end - firstLegal.start, std::int64_t(65));
    EXPECT_EQ(firstLegal.crossfadeLength, 16);
    EXPECT_EQ(firstLegal.seamCost, 0.65);

    std::vector<std::vector<double>> silence(1, std::vector<double>(128, 0.0));
    expectNeutral<double>(finder.find(constView(silence), { 0, 0 }, { 65, 65 },
                                      65, 65, settings),
                          Finder::Status::InsufficientActivity);

    std::vector<std::vector<double>> dc(1, std::vector<double>(128, 0.25));
    expectNeutral<double>(finder.find(constView(dc), { 0, 0 }, { 65, 65 },
                                      65, 65, settings),
                          Finder::Status::InsufficientActivity);

    std::vector<std::vector<double>> belowFloor(1, std::vector<double>(128));
    for (int i = 0; i < 128; ++i)
        belowFloor[0][static_cast<std::size_t>(i)] = (i & 1) ? 9.0e-6 : -9.0e-6;
    expectNeutral<double>(finder.find(constView(belowFloor), { 0, 0 }, { 65, 65 },
                                      65, 65, settings),
                          Finder::Status::InsufficientActivity);

    auto capSettings = settings;
    capSettings.maxCoarsePairs = 1;
    expectNeutral<double>(finder.find(view, { 0, 2 }, { 65, 67 },
                                      65, 67, capSettings),
                          Finder::Status::CandidateLimitExceeded);
    // Exactly at the conservative pre-count the request proceeds.
    capSettings.maxCoarsePairs = 9;
    const auto atCap = finder.find(view, { 0, 2 }, { 65, 67 },
                                   65, 67, capSettings);
    EXPECT_TRUE(atCap.status == Finder::Status::Success
                || atCap.status == Finder::Status::NoAcceptableLoop);

    expectNeutral<double>(finder.find(view, { 0, 0 }, { 64, 64 },
                                      64, 64, settings),
                          Finder::Status::NoAcceptableLoop);
}

DSPARK_TEST(LoopFinder_find_and_render_float)
{
    exerciseSearchAndRender<float>();
}

DSPARK_TEST(LoopFinder_find_and_render_double)
{
    exerciseSearchAndRender<double>();
}

DSPARK_TEST(LoopFinder_constant_seam_renders_unchanged_float)
{
    exerciseConstantSeam<float>();
}

DSPARK_TEST(LoopFinder_constant_seam_renders_unchanged_double)
{
    exerciseConstantSeam<double>();
}

DSPARK_TEST(LoopFinder_render_minimal_fade_and_cyclic_step)
{
    using Finder = LoopFinder<double>;
    auto audio = plantedSource<double>(1);
    Finder::Result loop { Finder::Status::Success, 96, 417, 2, 0.0 };
    std::vector<std::vector<double>> output(1, std::vector<double>(319, -3.0));
    EXPECT_TRUE(Finder::renderLoop(constView(audio), loop, mutableView(output))
                == Finder::Status::Success);
    // L=2 has only the two exact endpoints: tail at p=0, head at p=1.
    EXPECT_EQ(output[0][0], audio[0][415]);
    EXPECT_EQ(output[0][1], audio[0][97]);
    for (int i = 2; i < 319; ++i)
        EXPECT_EQ(output[0][static_cast<std::size_t>(i)],
                  audio[0][static_cast<std::size_t>(96 + i)]);
    // The cyclic wrap of the rendered cycle jumps between two adjacent
    // source samples: y[0] - y[M-1] equals x[end-L] - x[end-L-1].
    const double cyclicStep = output[0][0] - output[0][318];
    const double sourceStep = audio[0][415] - audio[0][414];
    EXPECT_NEAR(cyclicStep, sourceStep,
                8.0 * std::numeric_limits<double>::epsilon());
}

DSPARK_TEST(LoopFinder_transactional_render_rejections)
{
    using Finder = LoopFinder<double>;
    auto audio = plantedSource<double>(1);
    Finder::Result loop { Finder::Status::Success, 96, 417, 16, 0.0 };

    std::vector<std::vector<double>> wrong(1, std::vector<double>(304, -7.0));
    EXPECT_TRUE(Finder::renderLoop(constView(audio), loop, mutableView(wrong))
                == Finder::Status::InvalidOutput);
    for (double value : wrong[0]) EXPECT_EQ(value, -7.0);

    std::array<double*, 1> aliasPointer { audio[0].data() + 10 };
    AudioBufferView<double> alias(aliasPointer.data(), 1, 305);
    EXPECT_TRUE(Finder::renderLoop(constView(audio), loop, alias)
                == Finder::Status::InvalidOutput);

    std::vector<std::vector<double>> sentinel(1, std::vector<double>(305, -8.0));
    Finder::Result forged { Finder::Status::Success, 96, 417, 200, 0.0 };
    EXPECT_TRUE(Finder::renderLoop(constView(audio), forged,
                                   mutableView(sentinel))
                == Finder::Status::InvalidArgument);
    for (double value : sentinel[0]) EXPECT_EQ(value, -8.0);

    audio[0][500] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(Finder::renderLoop(constView(audio), loop, mutableView(sentinel))
                == Finder::Status::NumericalFailure);
    for (double value : sentinel[0]) EXPECT_EQ(value, -8.0);

    loop.status = Finder::Status::NoAcceptableLoop;
    EXPECT_TRUE(Finder::renderLoop(constView(audio), loop, mutableView(sentinel))
                == Finder::Status::InvalidArgument);
}

DSPARK_TEST(LoopFinder_exhaustive_oracle_matches_independent_scalar)
{
    using Finder = LoopFinder<double>;
    Finder finder;
    auto audio = plantedSource<double>(2);
    auto settings = exactSettings<double>();

    const Finder::Range startRange { 88, 104 };
    const Finder::Range endRange { 404, 424 };
    const std::int64_t minLoop = 300;
    const std::int64_t maxLoop = 336;
    const auto result = finder.find(constView(audio), startRange, endRange,
                                    minLoop, maxLoop, settings);

    // Independent full enumeration of the identical candidate set.
    bool found = false;
    std::int64_t bestStart = 0;
    std::int64_t bestEnd = 0;
    long double bestCost = 2.0L;
    const std::int64_t minimumPeriod = std::max<std::int64_t>(
        settings.comparisonLength + 1, 2 * settings.crossfadeLength);
    for (std::int64_t s = startRange.first; s <= startRange.last; ++s)
        for (std::int64_t e = endRange.first; e <= endRange.last; ++e)
        {
            const std::int64_t period = e - s;
            if (period < minimumPeriod || period < minLoop || period > maxLoop)
                continue;
            const auto scored = independentCost<double>(
                audio, s, e, settings.comparisonLength,
                static_cast<long double>(settings.minimumActivityRms));
            if (!scored.admissible) continue;
            if (!found || scored.cost < bestCost
                || (scored.cost == bestCost
                    && (s < bestStart || (s == bestStart && e < bestEnd))))
            {
                found = true;
                bestStart = s;
                bestEnd = e;
                bestCost = scored.cost;
            }
        }

    EXPECT_TRUE(found);
    EXPECT_TRUE(static_cast<bool>(result));
    EXPECT_EQ(result.start, bestStart);
    EXPECT_EQ(result.end, bestEnd);
    EXPECT_EQ(result.start, std::int64_t(96));
    EXPECT_EQ(result.end, std::int64_t(417));
    EXPECT_NEAR(static_cast<long double>(result.seamCost), bestCost,
                1.0e-6L + 8.0L * std::numeric_limits<double>::epsilon()
                    * std::max(1.0L, bestCost));
}

DSPARK_TEST(LoopFinder_tie_law_and_cost_dominance)
{
    using Finder = LoopFinder<double>;
    Finder finder;
    auto settings = exactSettings<double>();

    // Alternating +/-0.2 with period 2: every candidate whose start and end
    // share parity compares two identical windows and costs exactly zero;
    // opposite parity flips the sign and costs 0.45 + 0.20 = 0.65.
    std::vector<std::vector<double>> tied(1, std::vector<double>(640));
    for (int i = 0; i < 640; ++i)
        tied[0][static_cast<std::size_t>(i)] = (i & 1) ? 0.2 : -0.2;

    // Same-parity start/end pairs compare identical windows and tie at cost
    // zero: (95,419), (96,418), (96,420), (97,419), (98,418) and (98,420).
    // Candidate (95,418) enumerates first and must lose on cost; among the
    // zero-cost tie the smaller start, then the smaller end, must win.
    const auto tie = finder.find(constView(tied), { 95, 98 }, { 418, 420 },
                                 318, 325, settings);
    EXPECT_TRUE(static_cast<bool>(tie));
    EXPECT_EQ(tie.start, std::int64_t(95));
    EXPECT_EQ(tie.end, std::int64_t(419));
    EXPECT_TRUE(tie.seamCost <= 64.0 * std::numeric_limits<double>::epsilon());

    // Same fixture narrowed so the zero-cost tie is between ends only.
    const auto endTie = finder.find(constView(tied), { 96, 96 }, { 418, 420 },
                                    318, 325, settings);
    EXPECT_TRUE(static_cast<bool>(endTie));
    EXPECT_EQ(endTie.start, std::int64_t(96));
    EXPECT_EQ(endTie.end, std::int64_t(418));

    // The opposite-parity candidate really is the expensive one.
    const auto costly = finder.find(constView(tied), { 95, 95 }, { 418, 418 },
                                    318, 325, settings);
    EXPECT_TRUE(static_cast<bool>(costly));
    EXPECT_GT(costly.seamCost, 0.6);
}

DSPARK_TEST(LoopFinder_activity_channel_and_permutation_laws)
{
    using Finder = LoopFinder<double>;
    Finder finder;
    auto mono = plantedSource<double>(1);
    auto stereo = mono;
    stereo.emplace_back(mono[0].size(), 0.0);
    auto settings = exactSettings<double>();

    // A channel silent at both endpoints is excluded, not diluting.
    const auto monoResult = finder.find(constView(mono), { 96, 96 }, { 417, 417 },
                                        321, 321, settings);
    const auto stereoResult = finder.find(constView(stereo), { 96, 96 },
                                          { 417, 417 }, 321, 321, settings);
    EXPECT_TRUE(monoResult);
    EXPECT_TRUE(stereoResult);
    EXPECT_NEAR(monoResult.seamCost, stereoResult.seamCost,
                8.0 * std::numeric_limits<double>::epsilon());

    // Channel permutation: identical boundaries, cost within 8 ulp, and the
    // rendered output is the same permutation of channels.
    auto twoChannel = plantedSource<double>(2);
    auto swapped = twoChannel;
    std::swap(swapped[0], swapped[1]);
    const auto direct = finder.find(constView(twoChannel), { 88, 104 },
                                    { 404, 424 }, 300, 336, settings);
    const auto permuted = finder.find(constView(swapped), { 88, 104 },
                                      { 404, 424 }, 300, 336, settings);
    EXPECT_TRUE(direct);
    EXPECT_TRUE(permuted);
    EXPECT_EQ(direct.start, permuted.start);
    EXPECT_EQ(direct.end, permuted.end);
    EXPECT_NEAR(direct.seamCost, permuted.seamCost,
                8.0 * std::numeric_limits<double>::epsilon());

    const int rendered = static_cast<int>(direct.renderedLength());
    std::vector<std::vector<double>> outDirect(
        2, std::vector<double>(static_cast<std::size_t>(rendered), 0.0));
    std::vector<std::vector<double>> outPermuted(
        2, std::vector<double>(static_cast<std::size_t>(rendered), 0.0));
    EXPECT_TRUE(Finder::renderLoop(constView(twoChannel), direct,
                                   mutableView(outDirect))
                == Finder::Status::Success);
    EXPECT_TRUE(Finder::renderLoop(constView(swapped), permuted,
                                   mutableView(outPermuted))
                == Finder::Status::Success);
    for (int i = 0; i < rendered; ++i)
    {
        EXPECT_NEAR(outDirect[0][static_cast<std::size_t>(i)],
                    outPermuted[1][static_cast<std::size_t>(i)],
                    8.0 * std::numeric_limits<double>::epsilon());
        EXPECT_NEAR(outDirect[1][static_cast<std::size_t>(i)],
                    outPermuted[0][static_cast<std::size_t>(i)],
                    8.0 * std::numeric_limits<double>::epsilon());
    }
}

DSPARK_TEST(LoopFinder_determinism_and_monotonic_supersets)
{
    using Finder = LoopFinder<double>;
    Finder finder;
    auto audio = plantedSource<double>(2);
    auto settings = exactSettings<double>();

    const auto reference = finder.find(constView(audio), { 88, 104 },
                                       { 404, 424 }, 300, 336, settings);
    EXPECT_TRUE(reference);
    for (int repeat = 0; repeat < 5; ++repeat)
    {
        Finder fresh;
        // An unrelated search in between must not perturb the result.
        (void)fresh.find(constView(audio), { 0, 32 }, { 100, 132 }, 65, 132,
                         settings);
        const auto again = fresh.find(constView(audio), { 88, 104 },
                                      { 404, 424 }, 300, 336, settings);
        EXPECT_TRUE(again.status == reference.status);
        EXPECT_EQ(again.start, reference.start);
        EXPECT_EQ(again.end, reference.end);
        EXPECT_EQ(std::memcmp(&again.seamCost, &reference.seamCost,
                              sizeof(double)), 0);
    }

    // Widening either endpoint range (a candidate superset at stride 1)
    // must never increase the best cost.
    const auto widened = finder.find(constView(audio), { 80, 112 },
                                     { 396, 432 }, 300, 352, settings);
    EXPECT_TRUE(widened);
    EXPECT_TRUE(widened.seamCost <= reference.seamCost);

    // Narrowing to exactly the returned optimum preserves it.
    const auto narrowed = finder.find(constView(audio),
                                      { reference.start, reference.start },
                                      { reference.end, reference.end },
                                      reference.end - reference.start,
                                      reference.end - reference.start,
                                      settings);
    EXPECT_TRUE(narrowed);
    EXPECT_EQ(narrowed.start, reference.start);
    EXPECT_EQ(narrowed.end, reference.end);
    EXPECT_NEAR(narrowed.seamCost, reference.seamCost,
                8.0 * std::numeric_limits<double>::epsilon());
}
