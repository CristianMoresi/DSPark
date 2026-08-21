// DSPark - SpectralFreeze contract tests

#include "dspark_test.h"
#include "../Effects/SpectralFreeze.h"
#include "../Core/SpectralProcessor.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

using namespace dspark;

namespace {

template <typename T>
AudioBufferView<T> viewOf(std::vector<std::vector<T>>& channels,
                          int offset, int length)
{
    std::array<T*, 16> pointers {};
    for (std::size_t ch = 0; ch < channels.size(); ++ch)
        pointers[ch] = channels[ch].data() + offset;
    return AudioBufferView<T>(pointers.data(), static_cast<int>(channels.size()),
                              length);
}

template <typename T>
std::vector<std::vector<T>> makeTone(int channels, int samples, int fftSize)
{
    std::vector<std::vector<T>> result(
        static_cast<std::size_t>(channels),
        std::vector<T>(static_cast<std::size_t>(samples), T(0)));
    for (int ch = 0; ch < channels; ++ch)
    {
        for (int i = 0; i < samples; ++i)
        {
            const T phase = twoPi<T> * (T(23) + T(0.37))
                * static_cast<T>(i) / static_cast<T>(fftSize)
                + static_cast<T>(ch) * T(0.31);
            result[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)] =
                T(0.2) * std::sin(phase);
        }
    }
    return result;
}

template <typename T>
void processInBlocks(SpectralFreeze<T>& freeze,
                     std::vector<std::vector<T>>& audio,
                     const std::vector<int>& partitions,
                     int freezeAt = -1)
{
    int position = 0;
    std::size_t partition = 0;
    const int samples = static_cast<int>(audio.front().size());
    while (position < samples)
    {
        if (position == freezeAt) freeze.setFrozen(true);
        int block = partitions[partition++ % partitions.size()];
        block = std::min(block, samples - position);
        if (freezeAt > position && freezeAt < position + block)
            block = freezeAt - position;
        freeze.processBlock(viewOf(audio, position, block));
        position += block;
    }
}

// One owner-published control event: publish at an absolute sample boundary,
// after any callback fired by the preceding samples and before the sample at
// that index enters the processor.
struct ControlEvent
{
    int sample = 0;
    enum class Kind : int { Freeze, Unfreeze, ModeTonal, ModeDiffuse } kind
        = Kind::Freeze;
};

template <typename T>
void runSchedule(SpectralFreeze<T>& freeze,
                 std::vector<std::vector<T>>& audio,
                 const std::vector<int>& partitions,
                 std::vector<ControlEvent> events)
{
    std::sort(events.begin(), events.end(),
              [](const ControlEvent& a, const ControlEvent& b) {
                  return a.sample < b.sample;
              });
    int position = 0;
    std::size_t partition = 0;
    std::size_t nextEvent = 0;
    const int samples = static_cast<int>(audio.front().size());
    while (position < samples)
    {
        while (nextEvent < events.size()
               && events[nextEvent].sample == position)
        {
            switch (events[nextEvent].kind)
            {
                case ControlEvent::Kind::Freeze:
                    freeze.setFrozen(true); break;
                case ControlEvent::Kind::Unfreeze:
                    freeze.setFrozen(false); break;
                case ControlEvent::Kind::ModeTonal:
                    freeze.setPhaseMode(SpectralFreeze<T>::PhaseMode::Tonal);
                    break;
                case ControlEvent::Kind::ModeDiffuse:
                    freeze.setPhaseMode(SpectralFreeze<T>::PhaseMode::Diffuse);
                    break;
            }
            ++nextEvent;
        }
        int block = partitions[partition++ % partitions.size()];
        block = std::min(block, samples - position);
        if (nextEvent < events.size()
            && events[nextEvent].sample > position
            && events[nextEvent].sample < position + block)
            block = events[nextEvent].sample - position;
        freeze.processBlock(viewOf(audio, position, block));
        position += block;
    }
}

// Independent complex Hann-window projection of one output segment onto one
// exact frequency, in long double. This is the test's own analyzer; it calls
// no DSPark transform.
inline long double principalArgumentLd(long double phase)
{
    constexpr long double pi = std::numbers::pi_v<long double>;
    long double wrapped = std::fmod(phase + pi, 2.0L * pi);
    if (wrapped < 0.0L) wrapped += 2.0L * pi;
    return wrapped - pi;
}

template <typename T>
std::complex<long double> projectTone(const std::vector<T>& samples,
                                      int startIndex, int windowLength,
                                      long double cyclesPerSample)
{
    std::complex<long double> sum(0.0L, 0.0L);
    for (int n = 0; n < windowLength; ++n)
    {
        const long double window = 0.5L - 0.5L
            * std::cos(2.0L * std::numbers::pi_v<long double> * n
                       / windowLength);
        const long double angle = -2.0L * std::numbers::pi_v<long double>
            * cyclesPerSample * (startIndex + n);
        const long double value = static_cast<long double>(
            samples[static_cast<std::size_t>(startIndex + n)]);
        sum += std::complex<long double>(
            value * window * std::cos(angle), value * window * std::sin(angle));
    }
    return sum;
}

// Temporal phase-locking value of one frequency track: the magnitude of the
// mean unit phasor of phase differences taken frameSeparation hops apart.
template <typename T>
long double temporalPlv(const std::vector<T>& samples, int firstStart,
                        int frameCount, int hop, int windowLength,
                        int frameSeparation, long double cyclesPerSample)
{
    std::complex<long double> mean(0.0L, 0.0L);
    int used = 0;
    for (int j = 0; j + frameSeparation < frameCount; ++j)
    {
        const auto early = projectTone(samples, firstStart + j * hop,
                                       windowLength, cyclesPerSample);
        const auto late = projectTone(samples,
                                      firstStart + (j + frameSeparation) * hop,
                                      windowLength, cyclesPerSample);
        if (std::abs(early) <= 0.0L || std::abs(late) <= 0.0L) continue;
        const long double difference = principalArgumentLd(
            std::arg(late) - std::arg(early));
        mean += std::complex<long double>(std::cos(difference),
                                          std::sin(difference));
        ++used;
    }
    if (used == 0) return 0.0L;
    return std::abs(mean) / used;
}

template <typename T>
void exerciseApi()
{
    using Freeze = SpectralFreeze<T>;
    Freeze freeze;
    EXPECT_EQ(freeze.getLatency(), 0);
    EXPECT_EQ(freeze.getFFTSize(), 0);
    EXPECT_EQ(freeze.getHopSize(), 0);
    EXPECT_EQ(freeze.getTransitionSamples(), 0);
    EXPECT_FALSE(freeze.isFrozen());
    freeze.setFrozen(true);
    EXPECT_TRUE(freeze.isFrozen());
    freeze.setPhaseMode(Freeze::PhaseMode::Diffuse);
    EXPECT_TRUE(freeze.getPhaseMode() == Freeze::PhaseMode::Diffuse);
    freeze.setPhaseMode(static_cast<typename Freeze::PhaseMode>(255));
    EXPECT_TRUE(freeze.getPhaseMode() == Freeze::PhaseMode::Tonal);
    freeze.reset();

    std::vector<std::vector<T>> empty(1, std::vector<T>(8, T(0.25)));
    freeze.processBlock(viewOf(empty, 0, 8));
    for (T sample : empty.front()) EXPECT_EQ(sample, T(0.25));

    AudioSpec spec { 48000.0, 64, 2 };
    freeze.prepare(spec, 1000, 1);
    EXPECT_EQ(freeze.getFFTSize(), 1024);
    EXPECT_EQ(freeze.getHopSize(), 256);
    EXPECT_EQ(freeze.getLatency(), 1024);
    EXPECT_EQ(freeze.getTransitionSamples(), 1024);
    EXPECT_TRUE(freeze.isFrozen());

    freeze.prepare(AudioSpec { std::numeric_limits<double>::quiet_NaN(), 64, 2 },
                   4096, 1024);
    EXPECT_EQ(freeze.getFFTSize(), 1024);
    EXPECT_EQ(freeze.getHopSize(), 256);
}

template <typename T>
void liveIdentity()
{
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr int count = 9 * n + 37;
    AudioSpec spec { 48000.0, 257, 2 };
    auto wrapped = makeTone<T>(2, count, n);
    auto direct = wrapped;

    SpectralFreeze<T> freeze;
    freeze.prepare(spec, n, h);
    SpectralProcessor<T> spectral;
    spectral.prepare(spec, n, h);

    int position = 0;
    const std::array<int, 5> blocks { 1, 7, 64, 257, 113 };
    std::size_t blockIndex = 0;
    while (position < count)
    {
        const int block = std::min(blocks[blockIndex++ % blocks.size()],
                                   count - position);
        freeze.processBlock(viewOf(wrapped, position, block));
        auto noChange = [](T*, int) noexcept {};
        spectral.processBlock(viewOf(direct, position, block), noChange);
        position += block;
    }

    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < count; ++i)
            EXPECT_EQ(wrapped[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)],
                      direct[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)]);
}

template <typename T>
void freezeAndPartitionInvariance(typename SpectralFreeze<T>::PhaseMode mode)
{
    constexpr int n = 1024;
    constexpr int samples = 10 * n;
    AudioSpec spec { 48000.0, 257, 2 };
    auto source = makeTone<T>(2, samples, n);
    for (int ch = 0; ch < 2; ++ch)
        std::fill(source[static_cast<std::size_t>(ch)].begin() + 3 * n,
                  source[static_cast<std::size_t>(ch)].end(), T(0));
    auto fixed = source;
    auto chopped = source;

    SpectralFreeze<T> a;
    SpectralFreeze<T> b;
    a.setPhaseMode(mode);
    b.setPhaseMode(mode);
    a.prepare(spec, n, n / 4);
    b.prepare(spec, n, n / 4);
    processInBlocks(a, fixed, { 64 }, 3 * n);
    processInBlocks(b, chopped, { 1, 7, 113, 257, 19 }, 3 * n);

    long double heldEnergy = 0.0L;
    for (int ch = 0; ch < 2; ++ch)
    {
        for (int i = 0; i < samples; ++i)
            EXPECT_EQ(fixed[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)],
                      chopped[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)]);
        for (int i = 7 * n; i < 9 * n; ++i)
        {
            const long double value = static_cast<long double>(
                fixed[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)]);
            heldEnergy += value * value;
        }
    }
    EXPECT_GT(heldEnergy, 1.0L);
}

template <typename T>
void resetReproducibility()
{
    constexpr int n = 1024;
    constexpr int samples = 8 * n;
    AudioSpec spec { 48000.0, 128, 2 };
    const auto source = makeTone<T>(2, samples, n);

    SpectralFreeze<T> reused;
    reused.setFrozen(true);
    reused.setPhaseMode(SpectralFreeze<T>::PhaseMode::Diffuse);
    reused.prepare(spec, n, n / 4);
    auto throwaway = source;
    processInBlocks(reused, throwaway, { 37, 128 });
    reused.reset();
    auto afterReset = source;
    processInBlocks(reused, afterReset, { 37, 128 });

    SpectralFreeze<T> fresh;
    fresh.setFrozen(true);
    fresh.setPhaseMode(SpectralFreeze<T>::PhaseMode::Diffuse);
    fresh.prepare(spec, n, n / 4);
    auto reference = source;
    processInBlocks(fresh, reference, { 37, 128 });

    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < samples; ++i)
            EXPECT_EQ(afterReset[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)],
                      reference[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)]);
}

} // namespace

DSPARK_TEST(SpectralFreeze_public_api_float_double)
{
    exerciseApi<float>();
    exerciseApi<double>();
}

DSPARK_TEST(SpectralFreeze_live_is_exact_SpectralProcessor)
{
    liveIdentity<float>();
    liveIdentity<double>();
}

DSPARK_TEST(SpectralFreeze_tonal_hold_and_partition_invariance)
{
    freezeAndPartitionInvariance<float>(SpectralFreeze<float>::PhaseMode::Tonal);
    freezeAndPartitionInvariance<double>(SpectralFreeze<double>::PhaseMode::Tonal);
}

DSPARK_TEST(SpectralFreeze_diffuse_hold_and_partition_invariance)
{
    freezeAndPartitionInvariance<float>(SpectralFreeze<float>::PhaseMode::Diffuse);
    freezeAndPartitionInvariance<double>(SpectralFreeze<double>::PhaseMode::Diffuse);
}

DSPARK_TEST(SpectralFreeze_reset_restores_diffuse_generation)
{
    resetReproducibility<float>();
    resetReproducibility<double>();
}

DSPARK_TEST(SpectralFreeze_channel_and_nonfinite_contract)
{
    constexpr int samples = 4096;
    AudioSpec spec { 48000.0, 512, 1 };
    SpectralFreeze<float> freeze;
    freeze.prepare(spec, 1024, 256);

    std::vector<std::vector<float>> audio(
        2, std::vector<float>(static_cast<std::size_t>(samples), 0.125f));
    audio[0][17] = std::numeric_limits<float>::infinity();
    audio[0][18] = std::numeric_limits<float>::quiet_NaN();
    audio[1][17] = std::numeric_limits<float>::infinity();
    const auto dryBits = std::bit_cast<std::uint32_t>(audio[1][17]);
    freeze.processBlock(viewOf(audio, 0, samples));

    for (float sample : audio[0]) EXPECT_TRUE(std::isfinite(sample));
    EXPECT_TRUE(std::isinf(audio[1][17]));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(audio[1][17]), dryBits);
    for (int i = 0; i < samples; ++i)
    {
        if (i == 17) continue;
        EXPECT_EQ(audio[1][static_cast<std::size_t>(i)], 0.125f);
    }

    // The exact law: every non-finite processed sample is replaced with zero
    // before it enters analysis, so the output must be bit-identical to
    // processing the same stream with those samples already zeroed. A
    // sanitizer that only cleans the output would zero whole frames instead
    // and cannot pass this equality.
    std::vector<std::vector<float>> scrubbed(
        1, std::vector<float>(static_cast<std::size_t>(samples), 0.125f));
    scrubbed[0][17] = 0.0f;
    scrubbed[0][18] = 0.0f;
    SpectralFreeze<float> twin;
    twin.prepare(spec, 1024, 256);
    twin.processBlock(viewOf(scrubbed, 0, samples));
    for (int i = 0; i < samples; ++i)
        EXPECT_EQ(audio[0][static_cast<std::size_t>(i)],
                  scrubbed[0][static_cast<std::size_t>(i)]);
}

namespace {

template <typename T>
void exercisePrepareMatrix()
{
    using Freeze = SpectralFreeze<T>;
    Freeze freeze;
    const AudioSpec baseline { 48000.0, 512, 2 };
    freeze.prepare(baseline, 2048, 0);
    EXPECT_EQ(freeze.getFFTSize(), 2048);
    EXPECT_EQ(freeze.getHopSize(), 512);
    EXPECT_EQ(freeze.getLatency(), 2048);
    EXPECT_EQ(freeze.getTransitionSamples(), 2048);

    // Every invalid specification field preserves the prepared queries.
    const AudioSpec invalidSpecs[] = {
        { 0.0, 512, 2 },
        { -1.0, 512, 2 },
        { std::numeric_limits<double>::quiet_NaN(), 512, 2 },
        { std::numeric_limits<double>::infinity(), 512, 2 },
        { 48000.0, 0, 2 },
        { 48000.0, 65537, 2 },
        { 48000.0, 512, 0 },
        { 48000.0, 512, 17 },
    };
    for (const AudioSpec& spec : invalidSpecs)
    {
        freeze.prepare(spec, 4096, 1024);
        EXPECT_EQ(freeze.getFFTSize(), 2048);
        EXPECT_EQ(freeze.getHopSize(), 512);
        EXPECT_EQ(freeze.getLatency(), 2048);
        EXPECT_EQ(freeze.getTransitionSamples(), 2048);
    }

    // FFT requests clamp to [256,32768] and round upward to a power of two.
    const std::pair<int, int> fftCases[] = {
        { -1, 256 }, { 1, 256 }, { 255, 256 }, { 256, 256 }, { 257, 512 },
        { 1000, 1024 }, { 2048, 2048 }, { 32768, 32768 }, { 32769, 32768 },
    };
    for (const auto& [request, expected] : fftCases)
    {
        freeze.prepare(baseline, request, 0);
        EXPECT_EQ(freeze.getFFTSize(), expected);
        EXPECT_EQ(freeze.getLatency(), expected);
        EXPECT_EQ(freeze.getHopSize(), expected / 4);
    }

    // Only N/8, N/4 and N/2 are accepted; every other request selects N/4.
    freeze.prepare(baseline, 2048, -1);
    EXPECT_EQ(freeze.getHopSize(), 512);
    freeze.prepare(baseline, 2048, 0);
    EXPECT_EQ(freeze.getHopSize(), 512);
    freeze.prepare(baseline, 2048, 256);
    EXPECT_EQ(freeze.getHopSize(), 256);
    freeze.prepare(baseline, 2048, 512);
    EXPECT_EQ(freeze.getHopSize(), 512);
    freeze.prepare(baseline, 2048, 1024);
    EXPECT_EQ(freeze.getHopSize(), 1024);
    freeze.prepare(baseline, 2048, 1);
    EXPECT_EQ(freeze.getHopSize(), 512);
    freeze.prepare(baseline, 2048, 682);
    EXPECT_EQ(freeze.getHopSize(), 512);
    freeze.prepare(baseline, 2048, 2048);
    EXPECT_EQ(freeze.getHopSize(), 512);
}

// Sound gate, transition law: a completed exit is not approximately live, it
// is exactly live. After the last synthesis frame that still carried any
// frozen contribution has fully decayed, the wrapper's output must be
// bit-identical to a never-frozen SpectralProcessor identity twin.
template <typename T>
void exactExitIdentity(typename SpectralFreeze<T>::PhaseMode mode)
{
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr int total = 12 * n;
    AudioSpec spec { 48000.0, 257, 2 };
    auto wrapped = makeTone<T>(2, total, n);
    auto direct = wrapped;

    SpectralFreeze<T> freeze;
    freeze.setPhaseMode(mode);
    freeze.prepare(spec, n, h);
    SpectralProcessor<T> twin;
    twin.prepare(spec, n, h);

    runSchedule(freeze, wrapped, { 64, 257, 7 },
                { { 3 * n, ControlEvent::Kind::Freeze },
                  { 6 * n, ControlEvent::Kind::Unfreeze } });
    int position = 0;
    const std::array<int, 3> blocks { 64, 257, 7 };
    std::size_t blockIndex = 0;
    while (position < total)
    {
        int block = std::min(blocks[blockIndex++ % blocks.size()],
                             total - position);
        if (3 * n > position && 3 * n < position + block)
            block = 3 * n - position;
        if (6 * n > position && 6 * n < position + block)
            block = 6 * n - position;
        auto identity = [](T*, int) noexcept {};
        twin.processBlock(viewOf(direct, position, block), identity);
        position += block;
    }

    // Unfreeze publishes at 6N and loads at x0 = 6N + H. The last blended
    // frame starts at x0 + N - H = 7N and its synthesis support ends at
    // x0 + 2N - H = 8N; from 8N onward every contributing frame is unblended.
    bool sawHeldDifference = false;
    for (int ch = 0; ch < 2; ++ch)
    {
        for (int i = 5 * n; i < 6 * n; ++i)
            if (wrapped[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)]
                != direct[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)])
                sawHeldDifference = true;
        for (int i = 8 * n; i < total; ++i)
            EXPECT_EQ(
                wrapped[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)],
                direct[static_cast<std::size_t>(ch)][static_cast<std::size_t>(i)]);
    }
    EXPECT_TRUE(sawHeldDifference);
}

} // namespace

DSPARK_TEST(SpectralFreeze_prepare_matrix_and_query_contract)
{
    exercisePrepareMatrix<float>();
    exercisePrepareMatrix<double>();
}

DSPARK_TEST(SpectralFreeze_sound_gate_exact_exit_identity_tonal)
{
    exactExitIdentity<float>(SpectralFreeze<float>::PhaseMode::Tonal);
    exactExitIdentity<double>(SpectralFreeze<double>::PhaseMode::Tonal);
}

DSPARK_TEST(SpectralFreeze_sound_gate_exact_exit_identity_diffuse)
{
    exactExitIdentity<float>(SpectralFreeze<float>::PhaseMode::Diffuse);
    exactExitIdentity<double>(SpectralFreeze<double>::PhaseMode::Diffuse);
}

DSPARK_TEST(SpectralFreeze_sound_gate_tonal_held_phase_law)
{
    // Off-bin tone: the held partial must keep advancing at the captured
    // instantaneous frequency, not at the nominal bin center. Bound: the
    // measured frequency error stays within 3 cents (the accepted 99th
    // percentile bound; the median law is 1 cent) and the temporal
    // phase-locking value of N-separated frames is at least 0.97.
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr int k = 4;
    constexpr double sampleRate = 48000.0;
    const long double cyclesPerSample = (128.0L + 0.37L) / n;
    constexpr int total = 7 * n + 40 * h + 2 * n;
    AudioSpec spec { sampleRate, 512, 1 };

    std::vector<std::vector<double>> audio(
        1, std::vector<double>(static_cast<std::size_t>(total)));
    for (int i = 0; i < total; ++i)
        audio[0][static_cast<std::size_t>(i)] = static_cast<double>(
            0.25L * std::sin(2.0L * std::numbers::pi_v<long double>
                             * cyclesPerSample * i + 0.41L));

    SpectralFreeze<double> freeze;
    freeze.prepare(spec, n, h);
    runSchedule(freeze, audio, { 512 }, { { 3 * n, ControlEvent::Kind::Freeze } });

    // Stationary window: the changing-q support of the enter transition ends
    // before 7N for a request published at 3N.
    const int firstStart = 7 * n;
    const int frames = 36;
    const long double plv = temporalPlv(audio[0], firstStart, frames, h, n, k,
                                        cyclesPerSample);
    EXPECT_GT(plv, 0.97L);

    // Phase-regression frequency estimate against the generator frequency.
    long double previous = 0.0L;
    long double unwrapped = 0.0L;
    long double sumJ = 0.0L, sumPhi = 0.0L, sumJJ = 0.0L, sumJPhi = 0.0L;
    for (int j = 0; j < frames; ++j)
    {
        const auto z = projectTone(audio[0], firstStart + j * h, n,
                                   cyclesPerSample);
        const long double phase = std::arg(z);
        if (j == 0) unwrapped = phase;
        else unwrapped += principalArgumentLd(phase - previous);
        previous = phase;
        sumJ += j; sumPhi += unwrapped;
        sumJJ += static_cast<long double>(j) * j;
        sumJPhi += j * unwrapped;
    }
    const long double slope = (frames * sumJPhi - sumJ * sumPhi)
        / (frames * sumJJ - sumJ * sumJ);
    const long double frequencyError = slope / (2.0L
        * std::numbers::pi_v<long double> * h) * sampleRate;
    const long double toneHz = cyclesPerSample * sampleRate;
    const long double cents = 1200.0L
        * std::log2((toneHz + frequencyError) / toneHz);
    EXPECT_LT(std::abs(cents), 3.0L);
}

DSPARK_TEST(SpectralFreeze_sound_gate_diffuse_stability_and_decorrelation)
{
    // Harmonic capture held in Diffuse mode: deterministic phase evolution
    // must decorrelate N-separated frames (temporal PLV at most 0.35 per the
    // accepted quantile law) while epoch-averaged energy stays within the
    // accepted 0.75 dB global / 1.5 dB per-band / 2 percent centroid bounds.
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr int k = 4;
    constexpr double sampleRate = 48000.0;
    constexpr int epochLength = 48000;
    constexpr int epochs = 3;
    constexpr int total = 7 * n + epochs * epochLength + 2 * n;
    AudioSpec spec { sampleRate, 512, 1 };

    const int fundamentalBin = 16;
    const int partials = 8;
    std::vector<std::vector<double>> source(
        1, std::vector<double>(static_cast<std::size_t>(total), 0.0));
    for (int m = 1; m <= partials; ++m)
        for (int i = 0; i < total; ++i)
            source[0][static_cast<std::size_t>(i)] = static_cast<double>(
                source[0][static_cast<std::size_t>(i)] + (0.11L / m)
                    * std::sin(2.0L * std::numbers::pi_v<long double>
                               * (static_cast<long double>(fundamentalBin) * m / n)
                               * i + 0.37L * m));

    auto held = source;
    SpectralFreeze<double> diffuse;
    diffuse.setPhaseMode(SpectralFreeze<double>::PhaseMode::Diffuse);
    diffuse.prepare(spec, n, h);
    runSchedule(diffuse, held, { 512 },
                { { 3 * n, ControlEvent::Kind::Freeze } });

    auto tonalHeld = source;
    SpectralFreeze<double> tonal;
    tonal.prepare(spec, n, h);
    runSchedule(tonal, tonalHeld, { 512 },
                { { 3 * n, ControlEvent::Kind::Freeze } });

    const int firstStart = 7 * n;
    long double diffusePlvSum = 0.0L;
    long double diffusePlvWorst = 0.0L;
    long double tonalPlvWorst = 1.0L;
    for (int m = 1; m <= partials; ++m)
    {
        const long double cyclesPerSample =
            static_cast<long double>(fundamentalBin) * m / n;
        const long double diffusePlv = temporalPlv(
            held[0], firstStart, 100, h, n, k, cyclesPerSample);
        const long double tonalPlv = temporalPlv(
            tonalHeld[0], firstStart, 36, h, n, k, cyclesPerSample);
        diffusePlvSum += diffusePlv;
        diffusePlvWorst = std::max(diffusePlvWorst, diffusePlv);
        tonalPlvWorst = std::min(tonalPlvWorst, tonalPlv);
    }
    EXPECT_LT(diffusePlvSum / partials, 0.20L);
    EXPECT_LT(diffusePlvWorst, 0.35L);
    EXPECT_GT(tonalPlvWorst, 0.80L);

    // Epoch-averaged stability of the diffuse hold. The spectral-centroid
    // clause of the stability law needs the full banded spectrum analyzer of
    // the retained quality matrix; this suite enforces the global-energy and
    // per-band clauses, whose estimators are exact at this scale.
    std::array<long double, epochs> epochEnergy {};
    std::array<std::array<long double, 8>, epochs> bandEnergy {};
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        const int begin = firstStart + epoch * epochLength;
        for (int i = begin; i < begin + epochLength; ++i)
        {
            const long double value = static_cast<long double>(
                held[0][static_cast<std::size_t>(i)]);
            epochEnergy[static_cast<std::size_t>(epoch)] += value * value;
        }
        // Band power integrates a small bin neighborhood around each partial,
        // like the banded analyzer of the accepted stability law: the diffuse
        // phase sequence smears a partial across adjacent bins, and only the
        // integrated band energy is claimed stable, not one spectral slice.
        for (int m = 1; m <= partials; ++m)
        {
            const long double cyclesPerSample =
                static_cast<long double>(fundamentalBin) * m / n;
            long double power = 0.0L;
            for (int j = 0; j < 90; ++j)
                for (int offset = -2; offset <= 2; ++offset)
                {
                    const auto z = projectTone(
                        held[0], begin + j * (n / 2), n,
                        cyclesPerSample
                            + static_cast<long double>(offset) / n);
                    power += std::norm(z);
                }
            bandEnergy[static_cast<std::size_t>(epoch)]
                      [static_cast<std::size_t>(m - 1)] = power;
        }
    }
    for (int epoch = 1; epoch < epochs; ++epoch)
    {
        const long double globalDriftDb = 10.0L
            * std::log10(epochEnergy[static_cast<std::size_t>(epoch)]
                         / epochEnergy[0]);
        EXPECT_LT(std::abs(globalDriftDb), 0.75L);
        for (int band = 0; band < partials; ++band)
        {
            const long double bandDriftDb = 10.0L * std::log10(
                bandEnergy[static_cast<std::size_t>(epoch)]
                          [static_cast<std::size_t>(band)]
                / bandEnergy[0][static_cast<std::size_t>(band)]);
            EXPECT_LT(std::abs(bandDriftDb), 1.5L);
        }
    }
}

DSPARK_TEST(SpectralFreeze_sound_gate_stereo_coherence)
{
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr double sampleRate = 48000.0;
    constexpr int total = 7 * n + 30 * h + 2 * n;
    AudioSpec spec { sampleRate, 512, 2 };
    const long double cyclesPerSample = (96.0L + 0.37L) / n;

    for (const bool diffuseMode : { false, true })
    {
        const auto mode = diffuseMode
            ? SpectralFreeze<double>::PhaseMode::Diffuse
            : SpectralFreeze<double>::PhaseMode::Tonal;

        // Identical stereo stays bit-identical through enter, hold and exit.
        std::vector<std::vector<double>> identical(
            2, std::vector<double>(static_cast<std::size_t>(total)));
        for (int i = 0; i < total; ++i)
        {
            const double value = 0.25
                * std::sin(2.0 * std::numbers::pi_v<double>
                           * static_cast<double>(cyclesPerSample) * i);
            identical[0][static_cast<std::size_t>(i)] = value;
            identical[1][static_cast<std::size_t>(i)] = value;
        }
        SpectralFreeze<double> freeze;
        freeze.setPhaseMode(mode);
        freeze.prepare(spec, n, h);
        runSchedule(freeze, identical, { 257, 64 },
                    { { 3 * n, ControlEvent::Kind::Freeze },
                      { 10 * n, ControlEvent::Kind::Unfreeze } });
        for (int i = 0; i < total; ++i)
            EXPECT_EQ(identical[0][static_cast<std::size_t>(i)],
                      identical[1][static_cast<std::size_t>(i)]);

        // A 9-sample delayed stereo pair: the captured inter-channel phase
        // relation must be preserved by the hold in both modes. Bound: the
        // cross-channel phase at the captured partial drifts at most half a
        // degree across the measured plateau.
        std::vector<std::vector<double>> delayed(
            2, std::vector<double>(static_cast<std::size_t>(total), 0.0));
        for (int i = 0; i < total; ++i)
            delayed[0][static_cast<std::size_t>(i)] = 0.25
                * std::sin(2.0 * std::numbers::pi_v<double>
                           * static_cast<double>(cyclesPerSample) * i);
        for (int i = 9; i < total; ++i)
            delayed[1][static_cast<std::size_t>(i)] =
                delayed[0][static_cast<std::size_t>(i - 9)];

        SpectralFreeze<double> stereoFreeze;
        stereoFreeze.setPhaseMode(mode);
        stereoFreeze.prepare(spec, n, h);
        runSchedule(stereoFreeze, delayed, { 512 },
                    { { 3 * n, ControlEvent::Kind::Freeze } });

        const int firstStart = 7 * n;
        long double referenceOffset = 0.0L;
        long double worstDriftDegrees = 0.0L;
        for (int j = 0; j < 24; ++j)
        {
            const auto left = projectTone(delayed[0], firstStart + j * h, n,
                                          cyclesPerSample);
            const auto right = projectTone(delayed[1], firstStart + j * h, n,
                                           cyclesPerSample);
            const long double offset = principalArgumentLd(
                std::arg(left) - std::arg(right));
            if (j == 0) referenceOffset = offset;
            else
                worstDriftDegrees = std::max(worstDriftDegrees,
                    std::abs(principalArgumentLd(offset - referenceOffset))
                        * 180.0L / std::numbers::pi_v<long double>);
        }
        EXPECT_LT(worstDriftDegrees, 0.5L);

        // Swapping the source channels swaps the output.
        auto swappedInput = delayed;
        std::swap(swappedInput[0], swappedInput[1]);
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < total; ++i)
                swappedInput[static_cast<std::size_t>(ch)]
                            [static_cast<std::size_t>(i)] = 0.0;
        for (int i = 0; i < total; ++i)
            swappedInput[1][static_cast<std::size_t>(i)] = 0.25
                * std::sin(2.0 * std::numbers::pi_v<double>
                           * static_cast<double>(cyclesPerSample) * i);
        for (int i = 9; i < total; ++i)
            swappedInput[0][static_cast<std::size_t>(i)] =
                swappedInput[1][static_cast<std::size_t>(i - 9)];

        SpectralFreeze<double> swappedFreeze;
        swappedFreeze.setPhaseMode(mode);
        swappedFreeze.prepare(spec, n, h);
        runSchedule(swappedFreeze, swappedInput, { 512 },
                    { { 3 * n, ControlEvent::Kind::Freeze } });
        for (int i = 0; i < total; ++i)
        {
            EXPECT_NEAR(swappedInput[1][static_cast<std::size_t>(i)],
                        delayed[0][static_cast<std::size_t>(i)],
                        8.0 * std::numeric_limits<double>::epsilon());
            EXPECT_NEAR(swappedInput[0][static_cast<std::size_t>(i)],
                        delayed[1][static_cast<std::size_t>(i)],
                        8.0 * std::numeric_limits<double>::epsilon());
        }
    }
}

namespace {

// Shared measurement for the transition-artifact gate: 1 ms RMS windows at a
// 0.25 ms hop over the derived changing-q support, matched 50 ms stationary
// controls immediately before and after, first differences on the support
// and the 1 ms mean-square energy of the second-difference click residual.
struct TransitionMetrics
{
    long double supportRms = 0.0L;
    long double preRms = 0.0L;
    long double postRms = 0.0L;
    long double supportRmsMinimum = 0.0L;
    long double supportStep = 0.0L;
    long double controlStep = 0.0L;
    long double supportClick = 0.0L;
    long double controlClick = 0.0L;
};

template <typename T>
long double windowRms(const std::vector<T>& samples, int begin, int length)
{
    long double sum = 0.0L;
    for (int i = begin; i < begin + length; ++i)
    {
        const long double value = static_cast<long double>(
            samples[static_cast<std::size_t>(i)]);
        sum += value * value;
    }
    return std::sqrt(sum / length);
}

template <typename T>
long double maxAbsFirstDifference(const std::vector<T>& samples,
                                  int begin, int end)
{
    long double maximum = 0.0L;
    for (int i = std::max(begin, 1); i < end; ++i)
        maximum = std::max(maximum, std::abs(
            static_cast<long double>(samples[static_cast<std::size_t>(i)])
            - static_cast<long double>(
                samples[static_cast<std::size_t>(i - 1)])));
    return maximum;
}

template <typename T>
long double maxClickEnergy(const std::vector<T>& samples, int begin, int end,
                           int windowLength, int hop)
{
    long double maximum = 0.0L;
    for (int start = begin; start + windowLength <= end; start += hop)
    {
        long double sum = 0.0L;
        for (int i = std::max(start, 2); i < start + windowLength; ++i)
        {
            const long double residual =
                static_cast<long double>(samples[static_cast<std::size_t>(i)])
                - 2.0L * static_cast<long double>(
                    samples[static_cast<std::size_t>(i - 1)])
                + static_cast<long double>(
                    samples[static_cast<std::size_t>(i - 2)]);
            sum += residual * residual;
        }
        maximum = std::max(maximum, sum / windowLength);
    }
    return maximum;
}

// Both sides of every comparison use the same 1 ms / 0.25 ms grid, so a
// material's own short-window crest and trough appear in the controls
// exactly as they appear on the transition support and the comparison
// measures the transition, not the material. supportRms/preRms carry the
// support and control window maxima; supportRmsMinimum/postRms carry the
// support and control window minima.
template <typename T>
TransitionMetrics measureTransition(const std::vector<T>& samples,
                                    int supportBegin, int supportEnd,
                                    int controlLength, int sampleRate)
{
    TransitionMetrics metrics;
    const int rmsWindow = (sampleRate + 999) / 1000;
    const int rmsHop = (sampleRate + 3999) / 4000;
    metrics.supportRms = 0.0L;
    metrics.supportRmsMinimum = std::numeric_limits<long double>::max();
    for (int start = supportBegin; start + rmsWindow <= supportEnd;
         start += rmsHop)
    {
        const long double rms = windowRms(samples, start, rmsWindow);
        metrics.supportRms = std::max(metrics.supportRms, rms);
        metrics.supportRmsMinimum = std::min(metrics.supportRmsMinimum, rms);
    }
    long double controlMaximum = 0.0L;
    long double controlMinimum = std::numeric_limits<long double>::max();
    for (const int controlBegin : { supportBegin - controlLength, supportEnd })
    {
        for (int start = controlBegin;
             start + rmsWindow <= controlBegin + controlLength;
             start += rmsHop)
        {
            const long double rms = windowRms(samples, start, rmsWindow);
            controlMaximum = std::max(controlMaximum, rms);
            controlMinimum = std::min(controlMinimum, rms);
        }
    }
    metrics.preRms = controlMaximum;
    metrics.postRms = controlMinimum;
    metrics.supportStep = maxAbsFirstDifference(samples, supportBegin,
                                                supportEnd + 1);
    metrics.controlStep = std::max(
        maxAbsFirstDifference(samples, supportBegin - controlLength,
                              supportBegin),
        maxAbsFirstDifference(samples, supportEnd, supportEnd + controlLength));
    metrics.supportClick = maxClickEnergy(samples, supportBegin,
                                          supportEnd + 2, rmsWindow, rmsHop);
    metrics.controlClick = std::max(
        maxClickEnergy(samples, supportBegin - controlLength, supportBegin,
                       rmsWindow, rmsHop),
        maxClickEnergy(samples, supportEnd, supportEnd + controlLength,
                       rmsWindow, rmsHop));
    return metrics;
}

inline long double decibelRatio(long double numerator, long double denominator)
{
    return 20.0L * std::log10(numerator / denominator);
}

} // namespace

DSPARK_TEST(SpectralFreeze_sound_gate_transition_artifacts)
{
    // Derived-bound artifact gate over the exact changing-q supports of one
    // enter and one exit, per the accepted transition-window laws: support
    // RMS within +0.5 dB of the larger control (and within -0.5 dB of the
    // smaller for this stationary tone), first difference within +1.5 dB of
    // the control maximum, 1 ms click-residual energy within +2 dB of the
    // control maximum.
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr int sampleRate = 48000;
    constexpr int controlLength = (sampleRate + 19) / 20;

    // Enter at A and exit at X per the hop-aligned schedule law:
    // A = ceil_H(N + C - 2H), X = ceil_H(A + 2N + C - 2H).
    const auto ceilToHop = [](int value) {
        return h * ((value + h - 1) / h);
    };
    const int enterAt = ceilToHop(n + controlLength - 2 * h);
    const int exitAt = ceilToHop(enterAt + 2 * n + controlLength - 2 * h);
    const int renderLength = ceilToHop(exitAt + 2 * n + controlLength) + 2 * n;

    for (const bool harmonic : { false, true })
    {
        std::vector<std::vector<double>> audio(
            1, std::vector<double>(static_cast<std::size_t>(renderLength)));
        for (int i = 0; i < renderLength; ++i)
        {
            if (!harmonic)
                audio[0][static_cast<std::size_t>(i)] = 0.25
                    * std::sin(2.0 * std::numbers::pi_v<double>
                               * ((128.0 + 0.375) / n) * i);
            else
            {
                double value = 0.0;
                for (int m = 1; m <= 8; ++m)
                    value += (0.06 / m)
                        * std::sin(2.0 * std::numbers::pi_v<double>
                                   * (16.0 * m / n) * i + 0.31 * m);
                audio[0][static_cast<std::size_t>(i)] = value;
            }
        }

        AudioSpec spec { static_cast<double>(sampleRate), 512, 1 };
        SpectralFreeze<double> freeze;
        freeze.prepare(spec, n, h);
        runSchedule(freeze, audio, { 512 },
                    { { enterAt, ControlEvent::Kind::Freeze },
                      { exitAt, ControlEvent::Kind::Unfreeze } });

        const std::pair<int, int> supports[] = {
            { enterAt + 2 * h, enterAt + 2 * n },
            { exitAt + 2 * h, exitAt + 2 * n },
        };
        for (const auto& [begin, end] : supports)
        {
            const auto metrics = measureTransition(audio[0], begin, end,
                                                   controlLength, sampleRate);
            const long double largerControl =
                std::max(metrics.preRms, metrics.postRms);
            const long double smallerControl =
                std::min(metrics.preRms, metrics.postRms);
            EXPECT_LT(decibelRatio(metrics.supportRms, largerControl), 0.5L);
            EXPECT_GT(decibelRatio(metrics.supportRmsMinimum, smallerControl),
                      -0.5L);
            EXPECT_LT(decibelRatio(metrics.supportStep, metrics.controlStep),
                      1.5L);
            EXPECT_LT(10.0L * std::log10(metrics.supportClick
                                         / metrics.controlClick), 2.0L);
        }
    }
}

DSPARK_TEST(SpectralFreeze_sound_gate_reversals_retain_capture)
{
    // Sixteen alternating requests inside one transition must reverse the
    // blend without releasing or recapturing: after the storm resolves to a
    // full hold, the held spectrum is still the originally captured tone
    // even though the live input changed underneath.
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr int total = 16 * n;
    constexpr double sampleRate = 48000.0;
    AudioSpec spec { sampleRate, 512, 1 };
    const long double capturedCycles = (96.0L + 0.37L) / n;
    const long double laterCycles = (200.0L + 0.11L) / n;

    std::vector<std::vector<double>> audio(
        1, std::vector<double>(static_cast<std::size_t>(total)));
    for (int i = 0; i < total; ++i)
    {
        const long double cycles = i < 4 * n ? capturedCycles : laterCycles;
        audio[0][static_cast<std::size_t>(i)] = static_cast<double>(
            0.25L * std::sin(
                2.0L * std::numbers::pi_v<long double> * cycles * i));
    }

    std::vector<ControlEvent> events { { 3 * n, ControlEvent::Kind::Freeze } };
    for (int r = 1; r <= 16; ++r)
        events.push_back({ 3 * n + (r + 1) * h,
                           (r & 1) ? ControlEvent::Kind::Unfreeze
                                   : ControlEvent::Kind::Freeze });

    SpectralFreeze<double> freeze;
    freeze.prepare(spec, n, h);
    runSchedule(freeze, audio, { 512 }, events);

    const int firstStart = 9 * n;
    long double capturedPower = 0.0L;
    long double laterPower = 0.0L;
    for (int j = 0; j < 16; ++j)
    {
        capturedPower += std::norm(projectTone(audio[0], firstStart + j * h, n,
                                               capturedCycles));
        laterPower += std::norm(projectTone(audio[0], firstStart + j * h, n,
                                            laterCycles));
    }
    EXPECT_GT(capturedPower, 1.0e-3L);
    EXPECT_GT(capturedPower / std::max(laterPower, 1.0e-30L), 1.0e3L);
}

DSPARK_TEST(SpectralFreeze_sound_gate_mode_change_deferred_to_recapture)
{
    // A phase-mode request while captured is recorded but deferred: the
    // running tonal hold keeps its phase-locked texture; only the fresh
    // capture after a complete exit adopts the diffuse law.
    constexpr int n = 1024;
    constexpr int h = 256;
    constexpr double sampleRate = 48000.0;
    constexpr int holdOneStart = 7 * n;
    constexpr int exitAt = 11 * n;
    constexpr int refreezeAt = 14 * n;
    constexpr int holdTwoStart = refreezeAt + 4 * n;
    constexpr int total = holdTwoStart + 30 * h + 2 * n;
    AudioSpec spec { sampleRate, 512, 1 };
    const int fundamentalBin = 16;
    const int partials = 8;

    std::vector<std::vector<double>> audio(
        1, std::vector<double>(static_cast<std::size_t>(total), 0.0));
    for (int m = 1; m <= partials; ++m)
        for (int i = 0; i < total; ++i)
            audio[0][static_cast<std::size_t>(i)] += (0.11 / m)
                * std::sin(2.0 * std::numbers::pi_v<double>
                           * (static_cast<double>(fundamentalBin) * m / n) * i
                           + 0.37 * m);

    SpectralFreeze<double> freeze;
    freeze.prepare(spec, n, h);
    runSchedule(freeze, audio, { 512 },
                { { 3 * n, ControlEvent::Kind::Freeze },
                  { 5 * n, ControlEvent::Kind::ModeDiffuse },
                  { exitAt, ControlEvent::Kind::Unfreeze },
                  { refreezeAt, ControlEvent::Kind::Freeze } });

    long double tonalWorst = 1.0L;
    long double diffuseSum = 0.0L;
    for (int m = 1; m <= partials; ++m)
    {
        const long double cyclesPerSample =
            static_cast<long double>(fundamentalBin) * m / n;
        tonalWorst = std::min(tonalWorst, temporalPlv(
            audio[0], holdOneStart, 20, h, n, 4, cyclesPerSample));
        diffuseSum += temporalPlv(
            audio[0], holdTwoStart, 20, h, n, 4, cyclesPerSample);
    }
    EXPECT_GT(tonalWorst, 0.80L);
    EXPECT_LT(diffuseSum / partials, 0.35L);
}
