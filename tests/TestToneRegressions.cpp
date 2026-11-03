// DSPark Tests - Tone and modulation regressions
// Filters, Equalizer, Delay, Chorus, Phaser, Vibrato, FrequencyShifter,
// Panner, StereoWidth, DCBlocker, NoiseGenerator, Goertzel, LevelFollower.
//
// Each case states the defect it pins and the number the old code measured.

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Analysis/Goertzel.h"
#include "../Analysis/LevelFollower.h"
#include "../Effects/Chorus.h"
#include "../Effects/DCBlocker.h"
#include "../Effects/Delay.h"
#include "../Effects/Equalizer.h"
#include "../Effects/Filters.h"
#include "../Effects/FrequencyShifter.h"
#include "../Effects/NoiseGenerator.h"
#include "../Effects/Panner.h"
#include "../Effects/Phaser.h"
#include "../Effects/StereoWidth.h"
#include "../Effects/Vibrato.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

constexpr double kPiT = 3.14159265358979323846;

/// Planar multi-channel storage with block views.
struct Planar
{
    std::vector<std::vector<float>> d;
    std::vector<float*> p;
    Planar(int ch, int n) : d(static_cast<size_t>(ch), std::vector<float>(static_cast<size_t>(n), 0.0f)),
                            p(static_cast<size_t>(ch))
    {
        for (int c = 0; c < ch; ++c) p[static_cast<size_t>(c)] = d[static_cast<size_t>(c)].data();
    }
    int size() const { return static_cast<int>(d[0].size()); }
    dspark::AudioBufferView<float> view(int off, int n)
    {
        std::vector<float*>& q = scratch;
        q.resize(p.size());
        for (size_t c = 0; c < p.size(); ++c) q[c] = p[c] + off;
        return dspark::AudioBufferView<float>(q.data(), static_cast<int>(q.size()), n);
    }
    std::vector<float*> scratch;
};

template <class P>
void runBlocks(P& proc, Planar& x, int block, int from = 0, int to = -1)
{
    if (to < 0) to = x.size();
    for (int o = from; o < to; o += block)
        proc.processBlock(x.view(o, std::min(block, to - o)));
}

/// Least-squares amplitude of a tone of frequency f over x[a, b).
double toneAmp(const float* x, int a, int b, double f, double fs)
{
    double c = 0, s = 0, cc = 0, ss = 0, cs = 0;
    for (int i = a; i < b; ++i)
    {
        const double w = 2.0 * kPiT * f * i / fs, co = std::cos(w), si = std::sin(w);
        c += x[i] * co; s += x[i] * si; cc += co * co; ss += si * si; cs += co * si;
    }
    const double det = cc * ss - cs * cs;
    const double A = (c * ss - s * cs) / det, B = (s * cc - c * cs) / det;
    return std::sqrt(A * A + B * B);
}

/// Phase of a tone of frequency f over x[a, b) (radians).
double tonePhase(const float* x, int a, int b, double f, double fs)
{
    double c = 0, s = 0;
    for (int i = a; i < b; ++i)
    {
        const double w = 2.0 * kPiT * f * i / fs;
        c += x[i] * std::cos(w); s += x[i] * std::sin(w);
    }
    return std::atan2(-s, c);
}

/// |DFT| of x[a, b) at frequency f (sample indices relative to a).
double dftMag(const float* x, int a, int b, double f, double fs)
{
    double re = 0, im = 0;
    for (int i = a; i < b; ++i)
    {
        const double w = 2.0 * kPiT * f * (i - a) / fs;
        re += x[i] * std::cos(w); im -= x[i] * std::sin(w);
    }
    return std::sqrt(re * re + im * im);
}

double maxStep(const float* x, int a, int b)
{
    double m = 0;
    for (int i = std::max(a, 1); i < b; ++i) m = std::max(m, static_cast<double>(std::abs(x[i] - x[i - 1])));
    return m;
}

int argmaxAbs(const float* x, int n)
{
    int k = 0;
    for (int i = 1; i < n; ++i) if (std::abs(x[i]) > std::abs(x[k])) k = i;
    return k;
}

void fillSine(Planar& x, double f, double fs, float amp)
{
    for (auto& ch : x.d)
        for (size_t i = 0; i < ch.size(); ++i)
            ch[i] = amp * static_cast<float>(std::sin(2.0 * kPiT * f * static_cast<double>(i) / fs));
}

bool allFinite(const Planar& x)
{
    for (const auto& ch : x.d) for (float v : ch) if (!std::isfinite(v)) return false;
    return true;
}

} // namespace

// ============================================================================
// FilterEngine / Equalizer
// ============================================================================

// After prepare() the filter glided for ~150 ms from 1 kHz / 0 dB to the set
// band: a +12 dB bell at 200 Hz sounded at 0.27 of its settled level in the
// first 10-30 ms, in FilterEngine and Equalizer alike.
DSPARK_TEST(Tone_filter_and_eq_start_settled)
{
    const double fs = 48000.0;
    Planar a(1, 48000), b(1, 48000);
    fillSine(a, 200.0, fs, 0.1f);
    fillSine(b, 200.0, fs, 0.1f);

    FilterEngine<float> f;
    f.prepare(spec(fs, 512, 1));
    f.setPeaking(200.0f, 12.0f, 1.0f);
    runBlocks(f, a, 512);
    const double early = toneAmp(a.p[0], 480, 1440, 200.0, fs);
    const double late = toneAmp(a.p[0], 24000, 48000, 200.0, fs);
    EXPECT_GT(early / late, 0.95);

    Equalizer<float> eq;
    eq.setBand(0, 200.0f, 12.0f, 1.0f);
    eq.prepare(spec(fs, 512, 1));
    runBlocks(eq, b, 512);
    const double eqEarly = toneAmp(b.p[0], 480, 1440, 200.0, fs);
    const double eqLate = toneAmp(b.p[0], 24000, 48000, 200.0, fs);
    EXPECT_GT(eqEarly / eqLate, 0.95);
}

// The cutoff glided in linear Hz at Q 0.707 and overshot: a 10 kHz -> 500 Hz
// jump on a low-pass dipped a 150 Hz tone by 9.8 dB. Log-frequency, critically
// damped: no dip (this case measured an 8.4 dB dip).
DSPARK_TEST(Tone_filter_cutoff_glide_does_not_overshoot)
{
    const double fs = 48000.0;
    Planar x(1, 96000);
    fillSine(x, 150.0, fs, 0.25f);
    FilterEngine<float> f;
    f.prepare(spec(fs, 480, 1));
    f.setLowPass(10000.0f, 0.707f, 12);
    runBlocks(f, x, 480, 0, 48000);
    f.setFrequency(500.0f);
    runBlocks(f, x, 480, 48000, 96000);
    const double fin = toneAmp(x.p[0], 86400, 96000, 150.0, fs);
    double worst = 0.0;
    for (int a = 48000; a + 960 <= 72000; a += 240)
        worst = std::min(worst, 20.0 * std::log10(toneAmp(x.p[0], a, a + 960, 150.0, fs) / fin));
    EXPECT_GT(worst, -0.5);
}

// The 16-sample coefficient refresh grid restarted every block, so with the
// nonlinearity on, the output depended on the host's block sizes: 0.07 apart
// between blocks of 512 and of 7 here (0.54 on louder material).
DSPARK_TEST(Tone_filter_output_independent_of_block_partition)
{
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    Planar a(1, 20000), b(1, 20000);
    for (int i = 0; i < 20000; ++i) a.d[0][static_cast<size_t>(i)] = b.d[0][static_cast<size_t>(i)] = u(rng);
    auto make = [] {
        auto f = std::make_unique<FilterEngine<float>>();
        f->prepare(spec(48000.0, 512, 1));
        f->setPeaking(1000.0f, 6.0f, 1.0f);
        f->setNonlinearity(0.5f);
        return f;
    };
    auto fa = make();
    auto fb = make();
    runBlocks(*fa, a, 512);
    runBlocks(*fb, b, 7);
    double diff = 0;
    for (int i = 0; i < 20000; ++i)
        diff = std::max(diff, static_cast<double>(std::abs(a.d[0][static_cast<size_t>(i)] - b.d[0][static_cast<size_t>(i)])));
    EXPECT_EQ(diff, 0.0);
}

// setState() dropped the resonance for the shelf and tilt shapes (the shelf
// helpers take no Q), so a restored engine switched to a bell sounded at
// Q 0.707 (0.118 apart here); and before prepare() the engine designed at a
// 0 Hz rate and produced NaN instead of passing audio through.
DSPARK_TEST(Tone_filter_state_and_unprepared_pass_through)
{
    FilterEngine<float> a, b;
    a.prepare(spec(48000.0, 512, 1));
    b.prepare(spec(48000.0, 512, 1));
    a.setLowShelf(300.0f, 6.0f, 0.8f);
    a.setResonance(4.0f);
    const auto blob = a.getState();
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    a.setShape(FilterEngine<float>::Shape::Peak);
    b.setShape(FilterEngine<float>::Shape::Peak);
    Planar xa(1, 4096), xb(1, 4096);
    fillSine(xa, 2000.0, 48000.0, 0.3f);
    fillSine(xb, 2000.0, 48000.0, 0.3f);
    runBlocks(a, xa, 512);
    runBlocks(b, xb, 512);
    double diff = 0;
    for (int i = 0; i < 4096; ++i)
        diff = std::max(diff, static_cast<double>(std::abs(xa.d[0][static_cast<size_t>(i)] - xb.d[0][static_cast<size_t>(i)])));
    EXPECT_LT(diff, 1e-6);

    FilterEngine<float> raw;
    raw.setLowPass(1000.0f);
    Equalizer<float> rawEq;
    rawEq.setBand(0, 1000.0f, 6.0f);
    Planar x(1, 512), ref(1, 512);
    fillSine(x, 440.0, 48000.0, 0.5f);
    fillSine(ref, 440.0, 48000.0, 0.5f);
    raw.processBlock(x.view(0, 512));
    rawEq.processBlock(x.view(0, 512));
    EXPECT_TRUE(allFinite(x));
    EXPECT_EQ(x.d[0][300], ref.d[0][300]);
}

// The linear-phase kernel was 2 x maxBlockSize taps, so the response depended
// on the host buffer: a 100 Hz, Q 2, +12 dB bell gave +0.69 dB with a 64-sample
// host block (+5.89 dB at 512). Now a fixed kernel: within 0.3 dB of the
// drawn curve, with the latency reported exactly.
DSPARK_TEST(Tone_eq_linear_phase_is_independent_of_host_block)
{
    using EQ = Equalizer<float>;
    const double fs = 48000.0;
    for (int block : { 64, 4096 })
    {
        auto eq = std::make_unique<EQ>();
        eq->setFilterMode(EQ::FilterMode::LinearPhase);
        eq->setBand(0, 100.0f, 12.0f, 2.0f);
        eq->prepare(spec(fs, block, 1));
        const int latency = eq->getLatency();
        const int total = latency + 40000;
        Planar h(1, total);
        h.d[0][0] = 1.0f;
        runBlocks(*eq, h, block);
        EXPECT_EQ(argmaxAbs(h.p[0], total), latency);
        float freq = 100.0f, drawn = 0.0f;
        eq->getMagnitudeForFrequencyArray(&freq, &drawn, 1);
        const double heard = dftMag(h.p[0], 0, total, 100.0, fs);
        EXPECT_NEAR(20.0 * std::log10(heard), 20.0 * std::log10(static_cast<double>(drawn)), 0.3);
    }

    // The kernel length is a setting of its own (latency kernel/2 + kernel/16).
    auto setLength = [](auto& e) -> bool {
        if constexpr (requires { e.setLinearPhaseKernelLength(4096); })
        {
            e.setLinearPhaseKernelLength(4096);
            return true;
        }
        else { return false; }
    };
    auto eq = std::make_unique<EQ>();
    eq->setFilterMode(EQ::FilterMode::LinearPhase);
    EXPECT_TRUE(setLength(*eq));
    eq->prepare(spec(fs, 512, 1));
    EXPECT_EQ(eq->getLatency(), 2048 + 256);
}

// Switching a band off stepped the output (4.2x the steady sample-to-sample
// step here, MinimumPhase); a band change in LinearPhase swapped kernels
// mid-stream (5.08x). Both are crossfaded now.
DSPARK_TEST(Tone_eq_band_changes_are_click_free)
{
    using EQ = Equalizer<float>;
    const double fs = 48000.0;
    {
        EQ eq;
        eq.prepare(spec(fs, 512, 1));
        eq.setBand(0, 1000.0f, 12.0f, 2.0f);
        Planar x(1, 48128);
        fillSine(x, 1000.0, fs, 0.2f);
        runBlocks(eq, x, 512, 0, 24064);
        eq.setBandEnabled(0, false);
        runBlocks(eq, x, 512, 24064, 48128);
        const double steady = maxStep(x.p[0], 12000, 24000);
        EXPECT_LT(maxStep(x.p[0], 24064, 28000), 1.3 * steady);
    }
    {
        auto eq = std::make_unique<EQ>();
        eq->setFilterMode(EQ::FilterMode::LinearPhase);
        eq->setBand(0, 1000.0f, 12.0f, 2.0f);
        eq->prepare(spec(fs, 512, 1));
        const int n = 48000 * 3;
        Planar x(1, n);
        fillSine(x, 1000.0, fs, 0.2f);
        runBlocks(*eq, x, 512, 0, 48128);
        eq->setBand(0, 1000.0f, 0.0f, 2.0f);
        runBlocks(*eq, x, 512, 48128, n);
        const double steady = maxStep(x.p[0], 30000, 48000);
        EXPECT_LT(maxStep(x.p[0], 48128, n), 1.3 * steady);
    }
}

// ============================================================================
// Delay
// ============================================================================

namespace {
std::vector<float> delayImpulse(Delay<float>& d, int n, int block = 512)
{
    Planar x(1, n);
    x.d[0][0] = 1.0f;
    runBlocks(d, x, block);
    return x.d[0];
}
} // namespace

// The delay was stored in samples: a time set before prepare() was stored as 0
// (echo at the 3-sample floor), re-prepare at another rate kept the old count
// (and the smoothers started at 0: echo at +3 samples while the getter said
// 4800), the start-up glide from 0 played an early copy (out[103] = 1.0), a
// state blob from 48 kHz restored 100 ms as 50 ms at 96 kHz, and a feedback
// corner set before prepare() kept the 48 kHz coefficient (a 2 kHz low-pass
// read -0.96 dB at 2 kHz on a 96 kHz delay instead of -3.0).
DSPARK_TEST(Tone_delay_keeps_its_time_across_prepare_and_rates)
{
    {
        Delay<float> d;
        d.setDelayMs(100.0f);
        d.setMix(1.0f);
        d.prepare(spec(48000.0, 512, 1));
        auto y = delayImpulse(d, 9000);
        EXPECT_EQ(argmaxAbs(y.data(), 9000), 4800);
        double early = 0;
        for (int i = 0; i < 4790; ++i) early = std::max(early, static_cast<double>(std::abs(y[static_cast<size_t>(i)])));
        EXPECT_LT(early, 1e-6);

        d.prepare(spec(96000.0, 512, 1));
        auto z = delayImpulse(d, 12000);
        EXPECT_EQ(argmaxAbs(z.data(), 12000), 9600);
    }
    {
        Delay<float> src;
        src.prepare(spec(48000.0, 512, 1));
        src.setDelayMs(100.0f);
        src.setMix(1.0f);
        const auto blob = src.getState();
        Delay<float> dst;
        dst.prepare(spec(96000.0, 512, 1));
        EXPECT_TRUE(dst.setState(blob.data(), blob.size()));
        auto y = delayImpulse(dst, 12000);
        EXPECT_EQ(argmaxAbs(y.data(), 12000), 9600);
    }
    {
        Delay<float> d;
        d.setFeedbackLpHz(2000.0f);
        d.setFeedbackMode(Delay<float>::FeedbackMode::Clean);
        d.setMix(1.0f);
        d.prepare(spec(96000.0, 512, 1));
        d.setDelayMs(10.0f);
        d.setFeedback(0.5f);
        auto y = delayImpulse(d, 4000);
        const double first = dftMag(y.data(), 950, 1900, 2000.0, 96000.0);
        const double second = dftMag(y.data(), 1910, 2870, 2000.0, 96000.0);
        EXPECT_NEAR(20.0 * std::log10(second / first), 20.0 * std::log10(0.5) - 3.01, 0.3);
    }
}

// The feedback filters were skipped whenever their input was exactly 0, so
// (a) with the feedback at 0 their state froze and played back as a step when
// it returned (0.9 -> 0 -> 0.9 at -60 dBFS: a 0.152 peak here, 0.318 in
// review), and (b) digital silence truncated their decay (gated against
// dithered burst: 0.313 apart).
DSPARK_TEST(Tone_delay_feedback_filters_survive_zero_feedback_and_silence)
{
    const double fs = 48000.0;
    {
        Delay<float> d;
        d.prepare(spec(fs, 512, 1));
        d.setDelayMs(50.0f);
        d.setFeedbackLpHz(3000.0f);
        d.setFeedbackHpHz(100.0f);
        d.setMix(1.0f);
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        Planar x(1, 48000 * 3);
        for (int i = 0; i < 48000 * 3; ++i)
            x.d[0][static_cast<size_t>(i)] = (i < 48000 ? 0.5f : 0.001f) * u(rng);
        d.setFeedback(0.9f);
        runBlocks(d, x, 512, 0, 48128);
        d.setFeedback(0.0f);
        runBlocks(d, x, 512, 48128, 96256);
        d.setFeedback(0.9f);
        runBlocks(d, x, 512, 96256, 48000 * 3);
        double peak = 0;
        for (int i = 96256; i < 48000 * 3; ++i) peak = std::max(peak, static_cast<double>(std::abs(x.d[0][static_cast<size_t>(i)])));
        EXPECT_LT(peak, 0.05);
    }
    {
        auto run = [&](float tail) {
            Delay<float> d;
            d.prepare(spec(fs, 512, 1));
            d.setDelayMs(20.0f);
            d.setFeedback(0.7f);
            d.setFeedbackLpHz(1000.0f);
            d.setMix(1.0f);
            std::mt19937 rng(5);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            Planar x(1, 48000);
            for (int i = 0; i < 48000; ++i)
                x.d[0][static_cast<size_t>(i)] = i < 9600 ? 0.5f * u(rng) : tail * ((i & 1) ? 1.0f : -1.0f);
            runBlocks(d, x, 512);
            return x.d[0];
        };
        const auto gated = run(0.0f);
        const auto dithered = run(1e-9f);
        double diff = 0;
        for (size_t i = 0; i < gated.size(); ++i) diff = std::max(diff, static_cast<double>(std::abs(gated[i] - dithered[i])));
        EXPECT_LT(diff, 1e-3);
    }
}

// Under SmootherType::None the read point was never recorded, so switching to
// a smoother restarted it from 0 (echo at +3 samples); reset() followed by
// setSmoothingTime() did the same.
DSPARK_TEST(Tone_delay_smoother_changes_keep_the_echo)
{
    {
        Delay<float> d;
        d.setSmoother(Delay<float>::SmootherType::None);
        d.prepare(spec(48000.0, 512, 1));
        d.setDelayMs(100.0f);
        d.setMix(1.0f);
        Planar warm(1, 4096);
        runBlocks(d, warm, 512);
        d.setSmoother(Delay<float>::SmootherType::Linear);
        auto y = delayImpulse(d, 9000);
        EXPECT_NEAR(argmaxAbs(y.data(), 9000), 4800, 1);
    }
    {
        Delay<float> d;
        d.prepare(spec(48000.0, 512, 1));
        d.setDelayMs(100.0f);
        d.setMix(1.0f);
        Planar warm(1, 4096);
        runBlocks(d, warm, 512);
        d.reset();
        d.setSmoothingTime(30.0f);
        auto y = delayImpulse(d, 9000);
        EXPECT_NEAR(argmaxAbs(y.data(), 9000), 4800, 1);
    }
}

// Modulated reads were Hermite only: -21 dB of error at 12 kHz. The optional
// sinc reader is transparent (setInterpolation(); the default stays Hermite).
DSPARK_TEST(Tone_delay_optional_sinc_read_is_transparent)
{
    const double fs = 48000.0, f = 12000.0, delay = 100.37;
    Delay<float> d;
    d.setInterpolation(Delay<float>::Interpolation::Sinc);
    d.prepare(spec(fs, 512, 1));
    d.setDelaySamples(static_cast<float>(delay));
    d.setMix(1.0f);
    Planar x(1, 8192);
    fillSine(x, f, fs, 0.5f);
    runBlocks(d, x, 512);
    double err = 0, ref = 0;
    for (int i = 2048; i < 8192; ++i)
    {
        const double want = 0.5 * std::sin(2.0 * kPiT * f * (i - delay) / fs);
        err += (x.d[0][static_cast<size_t>(i)] - want) * (x.d[0][static_cast<size_t>(i)] - want);
        ref += want * want;
    }
    EXPECT_LT(10.0 * std::log10(err / ref), -80.0);
}

// ============================================================================
// Modulation
// ============================================================================

// The FIR Hilbert lost single sideband below ~500 Hz: the image of a 200 Hz
// tone shifted by +50 Hz sat at -12 dB (-3.5 dB at 60 Hz), with 95 samples of
// latency. The allpass-pair network keeps it below -50 dB, latency 0.
DSPARK_TEST(Tone_frequency_shifter_rejects_the_image_in_the_bass)
{
    const double fs = 48000.0;
    struct Case { double tone, shift; };
    for (const Case c : { Case{ 200.0, 50.0 }, Case{ 60.0, 30.0 } })
    {
        FrequencyShifter<float> fsh;
        fsh.prepare(spec(fs, 512, 1));
        fsh.setShift(static_cast<float>(c.shift));
        fsh.setMix(1.0f);
        Planar x(1, 96000);
        fillSine(x, c.tone, fs, 0.5f);
        runBlocks(fsh, x, 512);
        const double wanted = toneAmp(x.p[0], 48000, 96000, c.tone + c.shift, fs);
        const double image = toneAmp(x.p[0], 48000, 96000, c.tone - c.shift, fs);
        EXPECT_LT(20.0 * std::log10(image / wanted), -50.0);
    }
    EXPECT_EQ(FrequencyShifter<float>::getLatency(), 0);
}

// Chorus spread 1.0 put the right channel only 11.25 degrees off the left
// (1/32 cycle): the delay curves of a stereo pair barely moved apart (38
// samples peak to peak of L-R difference where 180 degrees gives 384).
DSPARK_TEST(Tone_chorus_full_spread_is_180_degrees)
{
    const double fs = 48000.0, tone = 50.0;
    Chorus<float> c;
    c.setVoices(1);
    c.setRate(0.25f);
    c.setDepthMs(2.0f);
    c.setCenterDelay(7.0f);
    c.setMix(1.0f);
    c.setStereoSpread(1.0f);
    c.prepare(spec(fs, 512, 2));
    const int n = 48000 * 6;
    Planar x(2, n);
    fillSine(x, tone, fs, 0.5f);
    runBlocks(c, x, 512);
    double lo = 1e9, hi = -1e9;
    for (int a = 96000; a + 960 <= n; a += 960)
    {
        double dphi = tonePhase(x.p[1], a, a + 960, tone, fs) - tonePhase(x.p[0], a, a + 960, tone, fs);
        dphi = std::remainder(dphi, 2.0 * kPiT);
        const double samples = dphi / (2.0 * kPiT * tone) * fs;
        lo = std::min(lo, samples);
        hi = std::max(hi, samples);
    }
    EXPECT_GT(hi - lo, 300.0);
}

// Every spread or voice-count change reset the LFO phases: a 0.46 step against
// 0.083 steady here. The LFOs now glide to their new phases and voices fade.
DSPARK_TEST(Tone_chorus_spread_and_voice_changes_are_click_free)
{
    const double fs = 48000.0;
    for (int which = 0; which < 2; ++which)
    {
        Chorus<float> c;
        c.setStereoSpread(0.0f);
        c.setVoices(2);
        c.setMix(1.0f);
        c.prepare(spec(fs, 512, 2));
        const int n = 48000 * 2;
        Planar x(2, n);
        fillSine(x, 1000.0, fs, 0.3f);
        runBlocks(c, x, 512, 0, 48128);
        if (which == 0) c.setStereoSpread(1.0f); else c.setVoices(4);
        runBlocks(c, x, 512, 48128, n);
        for (int ch = 0; ch < 2; ++ch)
        {
            const double steady = maxStep(x.p[ch], 24000, 48000);
            EXPECT_LT(maxStep(x.p[ch], 48128, n), 1.5 * steady);
        }
    }
}

// A host block longer than the prepared maximum left its tail unmixed
// (fully wet past the prepared length: 0.62 off at mix 0) in Chorus and Phaser.
DSPARK_TEST(Tone_chorus_and_phaser_mix_oversized_blocks)
{
    const double fs = 48000.0;
    Planar ref(2, 1024);
    fillSine(ref, 700.0, fs, 0.4f);
    {
        Chorus<float> c;
        c.setMix(0.0f);
        c.prepare(spec(fs, 256, 2));
        Planar x(2, 1024);
        fillSine(x, 700.0, fs, 0.4f);
        c.processBlock(x.view(0, 1024));
        double diff = 0;
        for (int i = 0; i < 1024; ++i) diff = std::max(diff, static_cast<double>(std::abs(x.d[0][static_cast<size_t>(i)] - ref.d[0][static_cast<size_t>(i)])));
        EXPECT_LT(diff, 1e-6);
    }
    {
        Phaser<float> p;
        p.setMix(0.0f);
        p.prepare(spec(fs, 256, 2));
        Planar x(2, 1024);
        fillSine(x, 700.0, fs, 0.4f);
        p.processBlock(x.view(0, 1024));
        double diff = 0;
        for (int i = 0; i < 1024; ++i) diff = std::max(diff, static_cast<double>(std::abs(x.d[0][static_cast<size_t>(i)] - ref.d[0][static_cast<size_t>(i)])));
        EXPECT_LT(diff, 1e-6);
    }
}

// Vibrato<float> kept the LFO phase and read position in float: at 4
// semitones and 0.1 Hz (read point ~35k samples back) a 10 kHz tone strayed
// -54.8 dB from the double instance. Both now track in double. (0.125 Hz
// here: a rate exact in float, so only the tracking precision differs.)
DSPARK_TEST(Tone_vibrato_float_tracks_double_at_full_depth)
{
    const double fs = 48000.0, f = 10000.0;
    const int n = 48000 * 4;
    Vibrato<float> vf;
    Vibrato<double> vd;
    for (auto* v : { &vf })
    {
        v->setRate(0.125f); v->setDepth(4.0f); v->prepare(spec(fs, 512, 1));
    }
    vd.setRate(0.125); vd.setDepth(4.0); vd.prepare(spec(fs, 512, 1));
    Planar x(1, n);
    fillSine(x, f, fs, 0.25f);
    std::vector<double> xd(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) xd[static_cast<size_t>(i)] = static_cast<double>(x.d[0][static_cast<size_t>(i)]);
    runBlocks(vf, x, 512);
    for (int o = 0; o < n; o += 512)
    {
        double* p = xd.data() + o;
        vd.processBlock(AudioBufferView<double>(&p, 1, 512));
    }
    double err = 0, ref = 0;
    for (int i = n - 48000; i < n; ++i)
    {
        const double e = static_cast<double>(x.d[0][static_cast<size_t>(i)]) - xd[static_cast<size_t>(i)];
        err += e * e;
        ref += xd[static_cast<size_t>(i)] * xd[static_cast<size_t>(i)];
    }
    EXPECT_LT(10.0 * std::log10(err / ref), -80.0);
}

// ============================================================================
// Stereo, utilities and analysis
// ============================================================================

// Binaural and Haas delay both ears by a common 4 samples that getLatency()
// did not exist to report (PDC off by 4 samples).
DSPARK_TEST(Tone_panner_reports_its_haas_latency)
{
    auto latencyOf = [](auto& p) -> int {
        if constexpr (requires { p.getLatency(); }) return p.getLatency();
        else return -1;
    };
    Panner<float> p;
    p.setAlgorithm(Panner<float>::Algorithm::Haas);
    p.setPan(0.0f);
    p.prepare(spec(48000.0, 512, 2));
    Planar x(2, 2048);
    x.d[0][0] = x.d[1][0] = 1.0f;
    runBlocks(p, x, 512);
    EXPECT_EQ(latencyOf(p), 4);
    EXPECT_EQ(argmaxAbs(x.p[0], 2048), 4);
    EXPECT_EQ(argmaxAbs(x.p[1], 2048), 4);
}

// One NaN latched the bass-mono side filter (StereoWidth) and the DC blocker
// recursion forever: every later output sample was NaN.
DSPARK_TEST(Tone_stereo_width_and_dc_blocker_recover_from_nan)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    {
        StereoWidth<float> w;
        w.prepare(spec(48000.0, 512, 2));
        w.setBassMono(true, 120.0);
        w.setWidth(1.5f);
        Planar x(2, 4096);
        fillSine(x, 300.0, 48000.0, 0.3f);
        x.d[0][10] = nan;
        runBlocks(w, x, 512);
        bool finite = true;
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 512; i < 4096; ++i) finite = finite && std::isfinite(x.d[static_cast<size_t>(ch)][static_cast<size_t>(i)]);
        EXPECT_TRUE(finite);
    }
    for (int order : { 1, 4 })
    {
        DCBlocker<float> d;
        d.prepare(spec(48000.0, 512, 1));
        d.setOrder(order);
        Planar x(1, 4096);
        fillSine(x, 300.0, 48000.0, 0.3f);
        x.d[0][10] = nan;
        runBlocks(d, x, 512);
        bool finite = true;
        for (int i = 512; i < 4096; ++i) finite = finite && std::isfinite(x.d[0][static_cast<size_t>(i)]);
        EXPECT_TRUE(finite);
        (void)d.processSample(0, nan);
        EXPECT_TRUE(std::isfinite(d.processSample(0, 0.25f)));
    }
}

// Brown noise leaked by a fixed 0.995 per sample, so its corner doubled with
// the rate (38 Hz at 48 kHz, 77 Hz at 96 kHz): the 2 ms autocorrelation was
// 0.62 at 48 kHz but 0.38 at 96 kHz. Now the same at both rates.
DSPARK_TEST(Tone_brown_noise_corner_is_rate_independent)
{
    auto lagCorrelation = [](double fs) {
        NoiseGenerator<float> g;
        g.setType(NoiseGenerator<float>::Type::Brown);
        g.prepare(spec(fs, 512, 1));
        const int n = static_cast<int>(fs * 30.0);
        Planar x(1, n);
        runBlocks(g, x, 512);
        const int lag = static_cast<int>(fs * 0.002);
        double num = 0, den = 0, mean = 0;
        for (float v : x.d[0]) mean += v;
        mean /= n;
        for (int i = 0; i + lag < n; ++i)
        {
            const double a = x.d[0][static_cast<size_t>(i)] - mean, b = x.d[0][static_cast<size_t>(i + lag)] - mean;
            num += a * b;
            den += a * a;
        }
        return num / den;
    };
    EXPECT_NEAR(lagCorrelation(96000.0), lagCorrelation(48000.0), 0.08);
}

// The float Goertzel resonator misread a 27.5 Hz tone over 65536 samples at
// 192 kHz by -3.96 dB. The resonator runs in double now.
DSPARK_TEST(Tone_goertzel_float_reads_low_tones)
{
    const double fs = 192000.0, f = 27.5;
    const int n = 65536;
    std::vector<float> xf(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) xf[static_cast<size_t>(i)] = 0.5f * static_cast<float>(std::sin(2.0 * kPiT * f * i / fs));
    std::vector<double> xd(xf.begin(), xf.end());
    Goertzel<float> gf;
    Goertzel<double> gd;
    gf.prepare(fs, f, n);
    gd.prepare(fs, f, n);
    gf.processBlock(xf.data(), n);
    gd.processBlock(xd.data(), n);
    EXPECT_NEAR(20.0 * std::log10(static_cast<double>(gf.getMagnitude())),
                20.0 * std::log10(gd.getMagnitude()), 0.05);
}

// The meter's anti-denormal offset settled a 3 s RMS window at 192 kHz at
// -92.5 dB in silence (-94.0 after the 3.5 s here), above its own -100 dB
// readout floor.
DSPARK_TEST(Tone_level_follower_silence_reads_the_floor)
{
    const double fs = 192000.0;
    LevelFollower<float> m;
    m.prepare(spec(fs, 4096, 1));
    m.setRmsWindowMs(3000.0f);
    std::vector<float> zeros(4096, 0.0f);
    const float* p = zeros.data();
    for (int i = 0; i < static_cast<int>(fs * 3.5) / 4096; ++i)
        m.process(AudioBufferView<const float>(&p, 1, 4096));
    EXPECT_LT(m.getRmsLevelDb(0), -99.9f);
}
