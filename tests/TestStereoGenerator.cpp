// DSPark Tests - Stereo generator lifecycle and signal contracts
#include "../Core/ProcessorTraits.h"
#include "../Effects/StereoGenerator.h"
#include "dspark_test.h"
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

static_assert(AudioProcessor<StereoGenerator<float>, float>);
static_assert(AudioProcessor<StereoGenerator<double>, double>);

namespace
{
using Frame = std::array<double, 2>;
Frame signal(int i)
{
    return {.55 * std::sin(.13 * i) + .17 * std::cos(.977 * i),
            .32 * std::cos(.113 * i) - .13 * std::sin(1.413 * i)};
}

std::vector<Frame> render(StereoGenerator<double> &effect, int count, int block)
{
    std::vector<Frame> output(static_cast<std::size_t>(count));
    AudioBuffer<double> buffer;
    buffer.resize(2, block);
    for (int first = 0; first < count; first += block)
    {
        const int size = std::min(block, count - first);
        for (int i = 0; i < size; ++i)
        {
            const auto x = signal(first + i);
            buffer.getChannel(0)[i] = x[0];
            buffer.getChannel(1)[i] = x[1];
        }
        if (!effect.processBlock(buffer.toView().getSubView(0, size)))
            return {};
        for (int i = 0; i < size; ++i)
            output[first + i] = {buffer.getChannel(0)[i], buffer.getChannel(1)[i]};
    }
    return output;
}

template <typename T> bool identity(double rate, int factor, float cut)
{
    StereoGenerator<T> effect;
    if (!effect.prepare({rate, 8193, 2}, {factor, cut}))
        return false;
    const int latency = effect.getLatency();
    AudioBuffer<T> buffer, original;
    buffer.resize(2, 8193);
    original.resize(2, 8193);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 8193; ++i)
            buffer.getChannel(ch)[i] = original.getChannel(ch)[i] =
                i % 7 == 0 ? -T(0) : static_cast<T>(signal(i)[ch]);
    if (!effect.processBlock(buffer.toView()))
        return false;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 8193; ++i)
        {
            const T expected = i < latency ? T(0) : original.getChannel(ch)[i - latency];
            const T actual = buffer.getChannel(ch)[i];
            if (actual != expected || std::signbit(actual) != std::signbit(expected))
                return false;
        }
    return true;
}
} // namespace

DSPARK_TEST(StereoGenerator_prepare_validation_preserves_active_stream)
{
    StereoGenerator<double> effect;
    AudioBuffer<double> buffer;
    buffer.resize(2, 128);
    EXPECT_FALSE(effect.processBlock(buffer.toView()));
    EXPECT_TRUE(effect.getStatus() == StereoGenerator<double>::Status::NotPrepared);
    EXPECT_TRUE(effect.setWidth(.5f));
    EXPECT_TRUE(effect.prepare({48000, 1024, 2}));
    EXPECT_EQ(effect.getOptions().oversampling, 1);
    EXPECT_EQ(effect.getLatency(), 256);
    EXPECT_TRUE(effect.processBlock(buffer.toView()));
    for (const int factor : {0, 3, 5, 32})
        EXPECT_FALSE(effect.prepare({48000, 1024, 2}, {factor, 0}));
    for (const float cut : {-1.f, 1.f, 19.f, 5001.f, std::numeric_limits<float>::infinity()})
        EXPECT_FALSE(effect.prepare({48000, 1024, 2}, {4, cut}));
    for (const AudioSpec spec : {AudioSpec{0, 128, 2},
                                 {7999, 128, 2},
                                 {384001, 128, 2},
                                 {48000, 0, 2},
                                 {48000, 128, 1},
                                 {48000, 128, 3}})
        EXPECT_FALSE(effect.prepare(spec));
    EXPECT_EQ(effect.getLatency(), 256);
    EXPECT_EQ(effect.getSourceFrame(), 128u);
    EXPECT_EQ(effect.getWidth(), .5f);
    EXPECT_FALSE(effect.setWidth(std::numeric_limits<float>::quiet_NaN()));
    EXPECT_FALSE(effect.setWidth(1.01f));
    EXPECT_FALSE(effect.setWidth(-.01f));
    EXPECT_EQ(effect.getWidth(), .5f);
}

DSPARK_TEST(StereoGenerator_delta_capture_matches_addback_and_rejects_aliases)
{
    StereoGenerator<double> effect;
    effect.setWidth(.375f);
    EXPECT_TRUE(effect.prepare({48000, 1025, 2}));
    AudioBuffer<double> input, original;
    input.resize(2, 1025); original.resize(2, 1025);
    std::array<double, 1025> delta{};
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 1025; ++i)
            input.getChannel(c)[i] = original.getChannel(c)[i] = signal(i)[c];
    EXPECT_FALSE(effect.processBlock(input.toView(), {delta.data(), 1024}));
    EXPECT_FALSE(effect.processBlock(input.toView(), {input.getChannel(0), 1025}));
    EXPECT_FALSE(effect.processBlock(input.toView(), {input.getChannel(1), 1025}));
    EXPECT_EQ(effect.getSourceFrame(), 0u);
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 1025; ++i)
            EXPECT_EQ(input.getChannel(c)[i], original.getChannel(c)[i]);
    EXPECT_TRUE(effect.processBlock(input.toView(), delta));
    const int latency = effect.getLatency();
    double largest = 0;
    for (int i = 0; i < 1025; ++i)
    {
        largest = std::max(largest, std::abs(delta[i]));
        for (int c = 0; c < 2; ++c)
        {
            const double dry = i < latency ? 0 : original.getChannel(c)[i-latency];
            EXPECT_EQ(input.getChannel(c)[i], dry + (c == 0 ? delta[i] : -delta[i]));
        }
    }
    EXPECT_GT(largest, .001);
    EXPECT_EQ(StereoGenerator<double>::getPrepareMemoryBound({48000, 127, 1}, {4, 0}), 0u);
    EXPECT_EQ(StereoGenerator<double>::getPrepareMemoryBound({48000, 127, 2}, {3, 0}), 0u);
}

DSPARK_TEST(StereoGenerator_zero_width_exact_delayed_pcm)
{
    for (double rate : {8000., 44100., 48000., 96000., 192000., 384000.})
    {
        EXPECT_TRUE(identity<float>(rate, 1, 0));
        EXPECT_TRUE(identity<double>(rate, 1, 0));
        EXPECT_TRUE(identity<float>(rate, 4, 0));
        EXPECT_TRUE(identity<double>(rate, 4, 0));
    }
    EXPECT_TRUE(identity<double>(48000, 2, 175));
    EXPECT_TRUE(identity<float>(48000, 16, 175));
}

DSPARK_TEST(StereoGenerator_mid_width_and_block_invariance)
{
    StereoGenerator<double> full, half, single, large;
    for (auto *effect : {&full, &half, &single, &large})
    {
        EXPECT_TRUE(effect->setWidth(effect == &half ? .5f : 1.f));
        EXPECT_TRUE(effect->prepare({48000, 1024, 2}));
    }
    const auto a = render(full, 8193, 127), b = render(half, 8193, 127);
    const auto c = render(single, 8193, 1), d = render(large, 8193, 1024);
    EXPECT_EQ(a.size(), 8193u);
    EXPECT_TRUE(a == c);
    EXPECT_TRUE(a == d);
    double energy = 0;
    for (int i = 0; i < 8193; ++i)
    {
        const Frame dry = i < full.getLatency() ? Frame{} : signal(i - full.getLatency());
        EXPECT_NEAR(a[i][0] + a[i][1], dry[0] + dry[1], 8e-16);
        for (int ch = 0; ch < 2; ++ch)
            EXPECT_NEAR(b[i][ch], dry[ch] + .5 * (a[i][ch] - dry[ch]), 4e-16);
        energy += (a[i][0] - dry[0]) * (a[i][0] - dry[0]);
    }
    EXPECT_GT(energy, .01);
}

DSPARK_TEST(StereoGenerator_width_automation_arrives_in_five_ms)
{
    StereoGenerator<double> automated, reference;
    EXPECT_TRUE(reference.setWidth(1));
    EXPECT_TRUE(automated.prepare({48000, 1024, 2}));
    EXPECT_TRUE(reference.prepare({48000, 1024, 2}));
    EXPECT_EQ(render(automated, 4096, 127).size(), 4096u);
    EXPECT_EQ(render(reference, 4096, 127).size(), 4096u);
    EXPECT_TRUE(automated.setWidth(1));
    const auto a = render(automated, 1024, 7), b = render(reference, 1024, 256);
    EXPECT_EQ(a.size(), 1024u);
    const int latency = automated.getLatency();
    for (int i = 0; i < 1024; ++i)
    {
        const auto dry = signal(i < latency ? 4096 + i - latency : i - latency);
        const double width = std::min((i + 1) / 240., 1.);
        for (int ch = 0; ch < 2; ++ch)
            EXPECT_NEAR(a[i][ch], dry[ch] + width * (b[i][ch] - dry[ch]), 2e-14);
    }
    EXPECT_EQ(automated.getLatency(), latency);
}

DSPARK_TEST(StereoGenerator_reset_source_grid_and_64bit_clock)
{
    StereoGenerator<double> a, b;
    EXPECT_TRUE(a.setWidth(1));
    EXPECT_TRUE(b.setWidth(1));
    EXPECT_TRUE(a.prepare({48000, 1024, 2}));
    EXPECT_TRUE(b.prepare({48000, 1024, 2}));
    for (std::uint64_t frame : {std::uint64_t(1031), (std::uint64_t(1) << 40) + 37})
    {
        a.resetAtFrame(frame);
        b.resetAtFrame(frame - frame % 64);
        AudioBuffer<double> zeros;
        zeros.resize(2, static_cast<int>(frame % 64));
        EXPECT_TRUE(b.processBlock(zeros.toView()));
        const auto x = render(a, 2048, 127), y = render(b, 2048, 127);
        EXPECT_EQ(x.size(), y.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            for (int ch = 0; ch < 2; ++ch)
                EXPECT_NEAR(x[i][ch], y[i][ch], 2e-13);
        EXPECT_EQ(a.getSourceFrame(), frame + 2048);
    }
    a.reset();
    const auto original = render(a, 2048, 127);
    a.reset();
    EXPECT_TRUE(original == render(a, 2048, 127));
    a.resetAtFrame(std::numeric_limits<std::uint64_t>::max() - 4);
    AudioBuffer<double> tooLong;
    tooLong.resize(2, 5);
    EXPECT_FALSE(a.processBlock(tooLong.toView()));
    EXPECT_TRUE(a.getStatus() == StereoGenerator<double>::Status::ClockOverflow);
    EXPECT_EQ(a.getSourceFrame(), std::numeric_limits<std::uint64_t>::max() - 4);
    EXPECT_TRUE(a.processBlock(tooLong.toView().getSubView(0, 4)));
    EXPECT_EQ(a.getSourceFrame(), std::numeric_limits<std::uint64_t>::max());
}

DSPARK_TEST(StereoGenerator_invalid_input_does_not_advance_or_mutate)
{
    StereoGenerator<double> a, b;
    EXPECT_TRUE(a.setWidth(1));
    EXPECT_TRUE(b.setWidth(1));
    EXPECT_TRUE(a.prepare({48000, 127, 2}));
    EXPECT_TRUE(b.prepare({48000, 127, 2}));
    AudioBuffer<double> bad;
    bad.resize(2, 128);
    bad.getChannel(0)[0] = .125;
    EXPECT_FALSE(a.processBlock(bad.toView()));
    EXPECT_EQ(bad.getChannel(0)[0], .125);
    bad.getChannel(1)[126] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(a.processBlock(bad.toView().getSubView(0, 127)));
    EXPECT_EQ(bad.getChannel(0)[0], .125);
    double *overlap[]{bad.getChannel(0), bad.getChannel(0) + 1};
    EXPECT_FALSE(a.processBlock({overlap, 2, 127}));
    double *null[]{bad.getChannel(0), nullptr};
    EXPECT_FALSE(a.processBlock({null, 2, 127}));
    EXPECT_FALSE(a.processBlock({overlap, 1, 127}));
    EXPECT_TRUE(a.getStatus() == StereoGenerator<double>::Status::InvalidInput);
    EXPECT_EQ(a.getSourceFrame(), 0u);
    EXPECT_TRUE(render(a, 2048, 127) == render(b, 2048, 127));
}

DSPARK_TEST(StereoGenerator_numerical_failure_latches_and_reset_recovers)
{
    StereoGenerator<double> effect;
    EXPECT_TRUE(effect.setWidth(1));
    EXPECT_TRUE(effect.prepare({48000, 4096, 2}));
    AudioBuffer<double> extreme;
    extreme.resize(2, 4096);
    for (int i = 0; i < 4096; ++i)
    {
        extreme.getChannel(0)[i] = std::numeric_limits<double>::max();
        extreme.getChannel(1)[i] = -std::numeric_limits<double>::max();
    }
    EXPECT_FALSE(effect.processBlock(extreme.toView()));
    EXPECT_TRUE(effect.getStatus() == StereoGenerator<double>::Status::NumericalFailure);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
            EXPECT_TRUE(std::isfinite(extreme.getChannel(ch)[i]));
    extreme.clear();
    EXPECT_FALSE(effect.processBlock(extreme.toView()));
    effect.reset();
    EXPECT_TRUE(effect.getStatus() == StereoGenerator<double>::Status::Ok);
    EXPECT_EQ(render(effect, 4096, 127).size(), 4096u);
}

DSPARK_TEST(StereoGenerator_presets_validate_before_publication)
{
    StereoGenerator<double> original, restored;
    EXPECT_TRUE(original.setWidth(.625f));
    EXPECT_TRUE(original.prepare({48000, 1024, 2}, {8, 175}));
    const auto state = original.getState();
    EXPECT_TRUE(restored.setState(state.data(), state.size()));
    EXPECT_EQ(restored.getWidth(), .625f);
    EXPECT_TRUE(restored.getOptions() == original.getOptions());
    EXPECT_TRUE(restored.prepare({48000, 1024, 2}));
    EXPECT_TRUE(render(original, 8193, 127) == render(restored, 8193, 127));
    for (int variant = 0; variant < 6; ++variant)
    {
        StateWriter invalid(stateId(variant == 0 ? "NOPE" : "STGN"), variant == 1 ? 2 : 1);
        if (variant == 2)
            invalid.write("width", std::numeric_limits<float>::infinity());
        if (variant == 3)
            invalid.write("oversampling", 3);
        if (variant == 4)
            invalid.write("width", 1);
        if (variant == 5)
        {
            invalid.write("width", .1f);
            invalid.write("width", .2f);
        }
        const auto bytes = invalid.blob();
        EXPECT_FALSE(restored.setState(bytes.data(), bytes.size()));
        EXPECT_TRUE(restored.getState() == state);
    }
    EXPECT_FALSE(restored.setState(state.data(), state.size() - 1));
    StateWriter partial(stateId("STGN"), 1);
    partial.write("width", .25f);
    const auto bytes = partial.blob();
    EXPECT_TRUE(restored.setState(bytes.data(), bytes.size()));
    EXPECT_EQ(restored.getWidth(), .25f);
    EXPECT_TRUE(restored.getOptions() == original.getOptions());
}

DSPARK_TEST(ContinuousClip_power_moments_match_exact_integrals)
{
    using Integral = detail::continuous_clip::Interval<detail::ClipperCurve::Hard, 7, false, false, 8>;
    Integral integral(1);
    // p(t) = t^7 never leaves the linear region. All eight integrals are 1/(8+j).
    std::array<double, 8> polynomial{};
    polynomial[7] = 1;
    const auto linear = integral.polynomial(polynomial);
    for (int j = 0; j < 8; ++j) EXPECT_NEAR(linear[j], 1. / (8 + j), 2e-15);
    // clamp(4t-2,-1,1): exact knee crossings at 1/4 and 3/4.
    polynomial = {-2, 4, 0, 0, 0, 0, 0, 0};
    const auto clipped = integral.polynomial(polynomial);
    for (int j = 0; j < 8; ++j)
    {
        const double low = std::pow(.25, j + 1), high = std::pow(.75, j + 1);
        const double expected = -low / (j + 1) +
                                4 * (std::pow(.75, j + 2) - std::pow(.25, j + 2)) / (j + 2) -
                                2 * (high - low) / (j + 1) + (1 - high) / (j + 1);
        EXPECT_NEAR(clipped[j], expected, 3e-15);
    }
    polynomial.fill(0);
    EXPECT_TRUE((integral.polynomial(polynomial) == std::array<double, 8>{}));
    polynomial[0] = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isnan(integral.polynomial(polynomial)[0]));
}

DSPARK_TEST(StereoGenerator_source_rate_color_rejects_foldback_and_retains_source_dc)
{
    // Independent direct-curve Fourier integrals, converged at 2^19/2^20
    // phase points. At these coherent frequencies all harmonics above the
    // fundamental lie outside Nyquist. Thus the complete steady-state reference
    // is one sinusoid multiplied by the original source-clock DC transfer.
    // The 0.475 Fs guard's passband error here is below this -120 dB input-relative
    // error budget. Include the low-level, low-frequency linear DC case separately.
    const auto check = [&]<detail::ClipperCurve C>(std::array<double, 3> fundamentals) {
        for (int source : {1, 2, 4, 8})
        for (int level = 0; level < 3; ++level)
        for (int bin : {5123, 6827})
        {
            constexpr int length = 16384, block = 127;
            const int frequencyBin = level == 0 ? 107 : bin;
            const double amplitude = std::array<double, 3>{.03, .1, .3}[level];
            const double omega = twoPi<double> * frequencyBin / length;
            const auto z = std::exp(std::complex<double>(0, -omega / source));
            const auto transfer = (1. - z) / (1. - .9995 * z);
            const int warm = source == 1 ? 65536 : source == 2 ? 32768 : 16384;
            detail::StereoColor<C> color(1, source);
            EXPECT_EQ(color.latency(), 256);
            std::array<double, block> samples{};
            double error = 0;
            int measured = 0;
            for (int first = 0; first < warm + length + color.latency(); first += block)
            {
                const int count = std::min(block, warm + length + color.latency() - first);
                for (int i = 0; i < count; ++i)
                    samples[i] = amplitude * std::cos(omega * (first + i));
                auto *channel = samples.data();
                color.process({&channel, 1, count});
                for (int i = 0; i < count; ++i)
                {
                    const int frame = first + i - color.latency();
                    if (frame < warm) continue;
                    const double reference = fundamentals[level] *
                        std::real(transfer * std::exp(std::complex<double>(0, omega * frame)));
                    error += (samples[i] - reference) * (samples[i] - reference);
                    ++measured;
                }
            }
            EXPECT_EQ(measured, length);
            EXPECT_LT(std::sqrt(error / length) / (amplitude / std::sqrt(2.)), 1e-6);
        }
    };
    check.template operator()<detail::ClipperCurve::SymmetricKnee>(
        {.03, .087663081431628861, .10005984701175599});
    check.template operator()<detail::ClipperCurve::AsymmetricKnee>(
        {.03, .086451374133307807, .099419622996524501});
}

DSPARK_TEST(StereoGenerator_float_matches_rounded_double_path)
{
    StereoGenerator<float> narrow;
    StereoGenerator<double> wide;
    EXPECT_TRUE(narrow.setWidth(1));
    EXPECT_TRUE(wide.setWidth(1));
    EXPECT_TRUE(narrow.prepare({48000, 127, 2}));
    EXPECT_TRUE(wide.prepare({48000, 127, 2}));
    AudioBuffer<float> a;
    a.resize(2, 127);
    AudioBuffer<double> b;
    b.resize(2, 127);
    for (int first = 0; first < 8193; first += 127)
    {
        for (int i = 0; i < 127; ++i)
            for (int ch = 0; ch < 2; ++ch)
                b.getChannel(ch)[i] = a.getChannel(ch)[i] =
                    static_cast<float>(signal(first + i)[ch]);
        EXPECT_TRUE(narrow.processBlock(a.toView()));
        EXPECT_TRUE(wide.processBlock(b.toView()));
        for (int i = 0; i < 127; ++i)
            for (int ch = 0; ch < 2; ++ch)
                EXPECT_EQ(a.getChannel(ch)[i], static_cast<float>(b.getChannel(ch)[i]));
    }
}

DSPARK_TEST(StereoGenerator_low_cut_only_changes_delta)
{
    StereoGenerator<double> effect;
    EXPECT_TRUE(effect.setWidth(1));
    EXPECT_TRUE(effect.prepare({48000, 1024, 2}, {4, 175}));
    EXPECT_EQ(effect.getLatency(), 4047);
    const auto output = render(effect, 8193, 127);
    EXPECT_EQ(output.size(), 8193u);
    for (int i = effect.getLatency(); i < 8193; ++i)
    {
        const auto dry = signal(i - effect.getLatency());
        EXPECT_NEAR(output[i][0] + output[i][1], dry[0] + dry[1], 8e-16);
    }
    EXPECT_TRUE(effect.prepare({8000, 1024, 2}, {4, 5000}));
    const auto rejected = render(effect, 2048, 127);
    EXPECT_EQ(rejected.size(), 2048u);
    for (int i = effect.getLatency(); i < 2048; ++i)
        EXPECT_TRUE(rejected[i] == signal(i - effect.getLatency()));
}

DSPARK_TEST(StereoGenerator_matches_independent_complete_signal_reference)
{
    // Independent 256x SciPy upfirdn/lfilter oracle: trapezoidal band equations,
    // direct rational curves and separately designed Kaiser FIRs. The 128x/256x
    // whole-signal convergence is below -145 dBFS RMS for all six fixtures.
    // These are reference values, not captured DSPark output. The public absolute
    // tolerance covers the source-rate and explicit four-times color paths.
    constexpr int length = 16384;
    constexpr std::array<int, 16> positions{0,    1,    126,  127,  128,  255,  256,   512,
                                            1024, 4095, 4096, 4097, 7000, 8192, 12000, 16383};
    constexpr std::array<std::array<double, 16>, 6> reference{
        {{1.0996589521028626e-08, -1.2324424521897758e-08, 8.8743294397457395e-05,
          1.0168831250423536, 0.13810142099542272, -0.019384476860522675, -0.018900071865284678,
          -0.0029465484383227438, 0.00092980273448502995, 6.9650200931611939e-06,
          6.9874058654827145e-06, 7.0094384285031751e-06, 0.92763998373515866,
          0.0003093279500493727, 9.1751833036932606e-07, 1.019085954290981e-09},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, -2.909258411924876e-06, 0.43619490857595983,
          0.56686407036827147, 0.44669297670519575, 0.44990259337068145, 0.012620874820312454,
          -0.0003601878624189901},
         {-0.00063870555139679101, 0.6391023712899373, 1.6500186465027658, 0.11121630896361978,
          -1.5296265162055349, 1.6928927210072038, 1.8960139305061734, -1.5333037831980074,
          1.8985210589424026, 1.6869089618826516, 1.8922917580188428, 0.61290446546328359,
          -1.8866928360004975, -1.5131385286698, -0.3689469521680861, 1.6585855488698549},
         {0.00031581138252476473, 0.79730805433236884, -0.36470067259830807, -0.16706380812946608,
          0.410362370980631, -0.25552333308654596, -0.30592273222084121, 0.40809185142851589,
          -0.30446887957142832, -0.25524508339425989, -0.30424449051974867, 0.33467622041129474,
          0.30386410000327979, 0.41050345030197954, -0.10826739039402584, -0.26066779707755744},
         {0.39903651764292186, 0.44515330316412877, 1.3786579496434224, -0.3173369163748162,
          -1.0200179762982502, 1.4305275619051723, 0.83832666672467382, -1.0174670548890241,
          0.83748250023713333, 1.4274980196963185, 0.83460147631446902, 0.1756796120772583,
          -0.83210330796453758, -1.0105720780175484, 0.17712771289685758, 1.4223036954486856},
         {2.1309501725477985e-08, 0.0099699850553699055, 1.0599523294480031, 1.0619006713807446,
          1.063536804587109, 0.35197402576957248, 0.34784541852190665, -0.65969131878704568,
          0.56552782441525362, 0.061122049215794116, 0.066013865794886439, 0.070895142585134741,
          0.18300948146335416, 0.43330034718246935, 0.15849963510950182, -0.42789936911742432}}};
    const auto source = [](int which, int frame) -> Frame {
        const double t = static_cast<double>(frame) / 48000;
        const double a = std::sin(twoPi<double> * 7000 * t);
        const double b = std::cos(twoPi<double> * 13000 * t);
        switch (which)
        {
        case 0:
            return {frame == 127 || frame == 7000 ? .9 : 0.,
                    frame == 127 || frame == 9000 ? -.7 : 0.};
        case 1:
            return {frame >= 4096 && frame < 12000 ? .45 : 0.,
                    frame >= 4096 && frame < 12000 ? .2 : 0.};
        case 2:
            return {.9 * a, .9 * a};
        case 3: {
            const double x = .9 * std::sin(twoPi<double> * 13000 * t);
            return {x, x};
        }
        case 4:
            return {.6 * a + .35 * b, .4 * a - .3 * b};
        default: {
            const double x = .9 * std::sin(twoPi<double> * 73 * t);
            return {x, -x};
        }
        }
    };
    for (int factor : {1, 4})
    for (int which = 0; which < 6; ++which)
    {
        StereoGenerator<double> effect;
        EXPECT_TRUE(effect.setWidth(1));
        EXPECT_TRUE(factor == 1 ? effect.prepare({48000, 127, 2})
                               : effect.prepare({48000, 127, 2}, {factor, 0}));
        const int latency = effect.getLatency();
        AudioBuffer<double> buffer;
        buffer.resize(2, 127);
        std::size_t checked = 0;
        for (int offset = 0; offset < length + latency; offset += 127)
        {
            const int count = std::min(127, length + latency - offset);
            for (int i = 0; i < count; ++i)
            {
                const auto x = offset + i < length ? source(which, offset + i) : Frame{};
                buffer.getChannel(0)[i] = x[0];
                buffer.getChannel(1)[i] = x[1];
            }
            EXPECT_TRUE(effect.processBlock(buffer.toView().getSubView(0, count)));
            for (int i = 0; i < count && checked < positions.size(); ++i)
                if (offset + i - latency == positions[checked])
                {
                    EXPECT_NEAR(buffer.getChannel(0)[i], reference[which][checked], 2e-7);
                    ++checked;
                }
        }
        EXPECT_EQ(checked, positions.size());
    }
}
