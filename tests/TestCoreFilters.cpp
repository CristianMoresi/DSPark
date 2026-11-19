// DSPark Tests - Core Filters
// Biquad, StateVariableFilter, LadderFilter, FIRFilter

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/Biquad.h"
#include "../Core/BiquadCascade.h"
#include "../Core/NormalizedLadder.h"
#include "../Core/StateVariableFilter.h"
#include "../Core/LadderFilter.h"
#include "../Core/FIRFilter.h"
#include "../Core/WDF.h"
#include "../Analysis/EnvelopeFollower.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// Helper: measure filter response at a given frequency
// ============================================================================

// Feed a sine through a biquad and measure output amplitude
template <typename FilterT>
static float measureFilterResponse(FilterT& filter, float freq, float sampleRate, int numSamples = 8192)
{
    for (int i = 0; i < 2048; ++i)
    {
        float in = std::sin(twoPi<float> * freq * static_cast<float>(i) / sampleRate);
        (void)filter.processSample(in, 0);
    }
    // Measure steady-state amplitude
    float peak = 0.0f;
    for (int i = 0; i < numSamples; ++i)
    {
        float in = std::sin(twoPi<float> * freq * static_cast<float>(2048 + i) / sampleRate);
        float out = filter.processSample(in, 0);
        float a = std::abs(out);
        if (a > peak) peak = a;
    }
    return peak;
}

// ============================================================================
// Biquad
// ============================================================================

DSPARK_TEST(Biquad_LP_passband)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 5000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // 100 Hz should pass through nearly unchanged
    float response = measureFilterResponse(bq, 100.0f, 44100.0f);
    EXPECT_GT(response, 0.95f);
}

DSPARK_TEST(Biquad_LP_stopband)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // 10 kHz should be heavily attenuated
    float response = measureFilterResponse(bq, 10000.0f, 44100.0f);
    EXPECT_LT(response, 0.1f);
}

DSPARK_TEST(Biquad_LP_cutoff_minus3dB)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // At cutoff, should be ~-3dB = ~0.707
    float response = measureFilterResponse(bq, 1000.0f, 44100.0f);
    EXPECT_GT(response, 0.60f);
    EXPECT_LT(response, 0.80f);
}

DSPARK_TEST(Biquad_HP_passband)
{
    auto coeffs = BiquadCoeffs::makeHighPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // 10 kHz should pass
    float response = measureFilterResponse(bq, 10000.0f, 44100.0f);
    EXPECT_GT(response, 0.90f);
}

DSPARK_TEST(Biquad_HP_stopband)
{
    auto coeffs = BiquadCoeffs::makeHighPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // 100 Hz should be attenuated
    float response = measureFilterResponse(bq, 100.0f, 44100.0f);
    EXPECT_LT(response, 0.15f);
}

DSPARK_TEST(Biquad_Peak_boost)
{
    auto coeffs = BiquadCoeffs::makePeak(44100.0, 1000.0, 2.0, 6.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    float atCenter = measureFilterResponse(bq, 1000.0f, 44100.0f);
    float offCenter = measureFilterResponse(bq, 100.0f, 44100.0f);

    // At center should be boosted (~+6dB = ~2.0x)
    EXPECT_GT(atCenter, 1.5f);
    // Off-center should be near unity
    EXPECT_GT(offCenter, 0.85f);
    EXPECT_LT(offCenter, 1.15f);
}

DSPARK_TEST(Biquad_AllPass_unity_magnitude)
{
    auto coeffs = BiquadCoeffs::makeAllPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // Allpass should have unity magnitude at all frequencies
    for (float freq : { 100.0f, 500.0f, 1000.0f, 5000.0f, 10000.0f })
    {
        bq.reset();
        float response = measureFilterResponse(bq, freq, 44100.0f);
        EXPECT_NEAR(response, 1.0f, 0.05f);
    }
}

DSPARK_TEST(Biquad_silence_in_silence_out)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    bq.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
    EXPECT_SILENT(tb.ch(1), 512, 1e-10f);
}

DSPARK_TEST(Biquad_processBlock_matches_processSample)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 2000.0);

    Biquad<float> bq1;
    bq1.setCoeffs(coeffs);
    auto tb = makeMonoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);

    std::vector<float> input(256);
    std::copy(tb.ch(0), tb.ch(0) + 256, input.begin());

    bq1.processBlock(tb.view());

    Biquad<float> bq2;
    bq2.setCoeffs(coeffs);
    std::vector<float> sampleOut(256);
    for (int i = 0; i < 256; ++i)
        sampleOut[i] = bq2.processSample(input[i], 0);

    // Should be identical
    for (int i = 0; i < 256; ++i)
        EXPECT_NEAR(tb.ch(0)[i], sampleOut[i], 1e-6f);
}

DSPARK_TEST(Biquad_reset)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 1000.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    for (int i = 0; i < 100; ++i)
        (void)bq.processSample(1.0f, 0);

    bq.reset();

    // After reset, silence should remain silent
    float out = bq.processSample(0.0f, 0);
    EXPECT_NEAR(out, 0.0f, 1e-10f);
}

DSPARK_TEST(Biquad_double_template)
{
    auto coeffs = BiquadCoeffs::makeLowPass(44100.0, 1000.0);
    Biquad<double> bq;
    bq.setCoeffs(coeffs);
    double out = bq.processSample(1.0, 0);
    EXPECT_FALSE(std::isnan(out));
    EXPECT_FALSE(std::isinf(out));
}

// ============================================================================
// StateVariableFilter
// ============================================================================

DSPARK_TEST(SVF_LP_passband)
{
    StateVariableFilter<float> svf;
    svf.prepare(defaultSpec());
    svf.setCutoff(5000.0f);
    svf.setResonance(0.0f);
    svf.setMode(StateVariableFilter<float>::Mode::LowPass);

    float response = measureFilterResponse(svf, 100.0f, 44100.0f);
    EXPECT_GT(response, 0.90f);
}

DSPARK_TEST(SVF_LP_stopband)
{
    StateVariableFilter<float> svf;
    svf.prepare(defaultSpec());
    svf.setCutoff(1000.0f);
    svf.setResonance(0.0f);
    svf.setMode(StateVariableFilter<float>::Mode::LowPass);

    float response = measureFilterResponse(svf, 10000.0f, 44100.0f);
    EXPECT_LT(response, 0.15f);
}

DSPARK_TEST(SVF_HP_passband)
{
    StateVariableFilter<float> svf;
    svf.prepare(defaultSpec());
    svf.setCutoff(1000.0f);
    svf.setResonance(0.0f);
    svf.setMode(StateVariableFilter<float>::Mode::HighPass);

    float response = measureFilterResponse(svf, 10000.0f, 44100.0f);
    EXPECT_GT(response, 0.85f);
}

DSPARK_TEST(SVF_multiOutput_consistency)
{
    StateVariableFilter<float> svf;
    svf.prepare(defaultSpec());
    svf.setCutoff(2000.0f);
    svf.setResonance(0.3f);

    // Feed signal and check that LP + HP ~ input for notch equivalent
    for (int i = 0; i < 4096; ++i)
    {
        float in = std::sin(twoPi<float> * 1000.0f * static_cast<float>(i) / 44100.0f);
        auto [lp, hp, bp] = svf.processMultiOutput(in, 0);
        EXPECT_FALSE(std::isnan(lp));
        EXPECT_FALSE(std::isnan(hp));
        EXPECT_FALSE(std::isnan(bp));
    }
}

DSPARK_TEST(SVF_silence_in_silence_out)
{
    StateVariableFilter<float> svf;
    svf.prepare(defaultSpec());
    svf.setCutoff(1000.0f);
    svf.setMode(StateVariableFilter<float>::Mode::LowPass);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    svf.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

DSPARK_TEST(SVF_AllPass_unity_magnitude)
{
    StateVariableFilter<float> svf;
    svf.prepare(defaultSpec());
    svf.setCutoff(2000.0f);
    svf.setResonance(0.3f);
    svf.setMode(StateVariableFilter<float>::Mode::AllPass);

    for (float freq : { 200.0f, 1000.0f, 2000.0f, 5000.0f })
    {
        svf.reset();
        float response = measureFilterResponse(svf, freq, 44100.0f);
        EXPECT_NEAR(response, 1.0f, 0.1f);
    }
}

// ============================================================================
// LadderFilter
// ============================================================================

DSPARK_TEST(Ladder_LP24_passband)
{
    LadderFilter<float> ladder;
    ladder.prepare(defaultSpec());
    ladder.setCutoff(5000.0f);
    ladder.setResonance(0.0f);
    ladder.setMode(LadderFilter<float>::Mode::LP24);

    float response = measureFilterResponse(ladder, 100.0f, 44100.0f);
    EXPECT_GT(response, 0.85f);
}

DSPARK_TEST(Ladder_LP24_stopband)
{
    LadderFilter<float> ladder;
    ladder.prepare(defaultSpec());
    ladder.setCutoff(1000.0f);
    ladder.setResonance(0.0f);
    ladder.setMode(LadderFilter<float>::Mode::LP24);

    // LP24 should be much steeper than LP12
    float response24 = measureFilterResponse(ladder, 8000.0f, 44100.0f);
    EXPECT_LT(response24, 0.01f); // -40dB+ at 3 octaves above cutoff
}

DSPARK_TEST(Ladder_LP24_steeper_than_LP12)
{
    LadderFilter<float> ladder12, ladder24;
    auto s = defaultSpec();
    ladder12.prepare(s);
    ladder12.setCutoff(1000.0f);
    ladder12.setResonance(0.0f);
    ladder12.setMode(LadderFilter<float>::Mode::LP12);

    ladder24.prepare(s);
    ladder24.setCutoff(1000.0f);
    ladder24.setResonance(0.0f);
    ladder24.setMode(LadderFilter<float>::Mode::LP24);

    float resp12 = measureFilterResponse(ladder12, 5000.0f, 44100.0f);
    float resp24 = measureFilterResponse(ladder24, 5000.0f, 44100.0f);

    // LP24 should attenuate more than LP12 in stopband
    EXPECT_LT(resp24, resp12);
}

DSPARK_TEST(Ladder_silence)
{
    LadderFilter<float> ladder;
    ladder.prepare(defaultSpec());
    ladder.setCutoff(1000.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    ladder.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

DSPARK_TEST(Ladder_resonance_peak)
{
    LadderFilter<float> ladder;
    ladder.prepare(defaultSpec());
    ladder.setCutoff(2000.0f);
    ladder.setResonance(0.8f);
    ladder.setMode(LadderFilter<float>::Mode::LP24);

    // Near cutoff should be boosted by resonance
    float atCutoff = measureFilterResponse(ladder, 2000.0f, 44100.0f);
    EXPECT_GT(atCutoff, 1.0f);
}

// Steady-state gain via quadrature projection over an exact-cycle window
// (the peak-based helper above cannot resolve percent-level accuracy).
template <typename FilterT>
static float tptGainAt(FilterT& filter, float freq, float fs)
{
    constexpr int settle = 4800, measure = 4800;  // whole cycles for 1k/4k @ 48k
    double re = 0.0, im = 0.0;
    for (int i = 0; i < settle + measure; ++i)
    {
        const double ph = twoPi<double> * freq * i / fs;
        const float out = filter.processSample(static_cast<float>(std::sin(ph)), 0);
        if (i >= settle)
        {
            re += out * std::sin(ph);
            im += out * std::cos(ph);
        }
    }
    return static_cast<float>(2.0 * std::sqrt(re * re + im * im) / measure);
}

static AudioSpec spec48k()
{
    AudioSpec spec;
    spec.sampleRate = 48000.0;
    spec.maxBlockSize = 512;
    spec.numChannels = 2;
    return spec;
}

DSPARK_TEST(Ladder_mode_gains_match_analog_prototype)
{
    // At the prewarped cutoff each TPT one-pole equals the analog prototype
    // L = 1/(1+j) exactly, so every tap gain is analytic: LP6 = 1/sqrt(2),
    // LP12 = 1/2, LP18 = 1/(2 sqrt(2)), LP24 = 1/4, HP24 = |1-L|^4 = 1/4,
    // BP12 = |L| * |1 - L^2| = 0.790569.
    using M = LadderFilter<float>::Mode;
    const struct { M mode; float expected; } cases[] = {
        { M::LP6,  0.707107f }, { M::LP12, 0.5f },      { M::LP18, 0.353553f },
        { M::LP24, 0.25f },     { M::BP12, 0.790569f }, { M::HP24, 0.25f },
    };
    for (const auto& tc : cases)
    {
        LadderFilter<float> ladder;
        ladder.prepare(spec48k());
        ladder.setCutoff(1000.0f);
        ladder.setResonance(0.0f);
        ladder.setMode(tc.mode);
        const float gain = tptGainAt(ladder, 1000.0f, 48000.0f);
        EXPECT_NEAR(gain, tc.expected, tc.expected * 0.015f);
    }

    // Two octaves up the LP24 slope is exact in the prewarped domain:
    // |H| = 1/(1 + r^2)^2 with r = tan(pi*4k/fs) / tan(pi*1k/fs).
    LadderFilter<float> lp24;
    lp24.prepare(spec48k());
    lp24.setCutoff(1000.0f);
    lp24.setResonance(0.0f);
    lp24.setMode(M::LP24);
    const double r = std::tan(pi<double> * 4000.0 / 48000.0) / std::tan(pi<double> * 1000.0 / 48000.0);
    const float expected4k = static_cast<float>(1.0 / ((1.0 + r * r) * (1.0 + r * r)));
    const float gain4k = tptGainAt(lp24, 4000.0f, 48000.0f);
    EXPECT_NEAR(gain4k, expected4k, expected4k * 0.05f);
}

DSPARK_TEST(Ladder_resonance_peak_and_dc_are_analytic)
{
    // At the cutoff L^4 = -1/4, so |H_LP24| = 0.25/(1 - k/4): res 0.9 -> 2.5.
    LadderFilter<float> ladder;
    ladder.prepare(spec48k());
    ladder.setCutoff(1000.0f);
    ladder.setResonance(0.9f);
    ladder.setMode(LadderFilter<float>::Mode::LP24);
    const float peak = tptGainAt(ladder, 1000.0f, 48000.0f);
    EXPECT_NEAR(peak, 2.5f, 0.075f);

    // DC gain with resonance is 1/(1 + k): res 0.5 -> 1/3 (the analog
    // passband loss). HP24 must reject that same DC (binomial applied to u,
    // the post-feedback ladder input - the historic fix, now pinned).
    LadderFilter<float> lp, hp;
    for (auto* f : { &lp, &hp })
    {
        f->prepare(spec48k());
        f->setCutoff(1000.0f);
        f->setResonance(0.5f);
    }
    lp.setMode(LadderFilter<float>::Mode::LP24);
    hp.setMode(LadderFilter<float>::Mode::HP24);
    float lpOut = 0.0f, hpOut = 0.0f;
    for (int i = 0; i < 4800; ++i)
    {
        lpOut = lp.processSample(1.0f, 0);
        hpOut = hp.processSample(1.0f, 0);
    }
    EXPECT_NEAR(lpOut, 1.0f / 3.0f, 0.002f);
    EXPECT_LT(std::abs(hpOut), 1e-3f);
}

DSPARK_TEST(Ladder_selfosc_bounded_with_drive)
{
    // resonance 1 + drive > 1: the clamped fastTanh bounds the feedback, so
    // the marginal loop cannot blow up. Excite with a burst, then run 2 s.
    LadderFilter<float> ladder;
    ladder.prepare(spec48k());
    ladder.setCutoff(1000.0f);
    ladder.setResonance(1.0f);
    ladder.setDrive(2.0f);
    ladder.setMode(LadderFilter<float>::Mode::LP24);

    uint32_t rng = 0xA5A5u;
    float maxAbs = 0.0f;
    bool finite = true;
    for (int i = 0; i < 96000; ++i)
    {
        float in = 0.0f;
        if (i < 256)
        {
            rng = rng * 1664525u + 1013904223u;
            in = ((rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
        }
        const float out = ladder.processSample(in, 0);
        if (!std::isfinite(out)) { finite = false; break; }
        maxAbs = std::max(maxAbs, std::abs(out));
    }
    EXPECT_TRUE(finite);
    EXPECT_LT(maxAbs, 4.0f);
}

DSPARK_TEST(Ladder_invalid_inputs_ignored_and_getters_honest)
{
    // Twin A/B: NaN/Inf setters and an invalid prepare must leave the filter
    // bit-identical to an untouched twin. With the old header
    // setResonance(NaN) poisoned the loop (k = NaN -> all-NaN output,
    // permanent) and setDrive(Inf) produced 0 * Inf = NaN inside fastTanh.
    LadderFilter<float> a, b;
    for (auto* f : { &a, &b })
    {
        f->prepare(spec48k());
        f->setCutoff(800.0f);
        f->setResonance(0.7f);
        f->setDrive(3.0f);
        f->setMode(LadderFilter<float>::Mode::LP24);
    }
    b.setCutoff(std::numeric_limits<float>::quiet_NaN());
    b.setResonance(std::numeric_limits<float>::quiet_NaN());
    b.setDrive(std::numeric_limits<float>::quiet_NaN());
    b.setDrive(std::numeric_limits<float>::infinity());
    AudioSpec bad;
    bad.sampleRate = -1.0;
    b.prepare(bad);  // ignored: keeps state AND the 48k coefficients

    EXPECT_NEAR(b.getCutoff(), 800.0f, 1e-6f);
    EXPECT_NEAR(b.getResonance(), 0.7f, 1e-6f);
    EXPECT_NEAR(b.getDrive(), 3.0f, 1e-6f);

    uint32_t rng = 0x1234u;
    int mismatches = 0;
    for (int i = 0; i < 4800; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        const float in = ((rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
        if (a.processSample(in, 0) != b.processSample(in, 0)) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0);

    // Pre-prepare cutoff requests are clamped once the rate arrives, and the
    // getter reports the clamped value (it used to keep reporting the raw 5).
    LadderFilter<float> c;
    c.setCutoff(5.0f);
    c.prepare(spec48k());
    EXPECT_NEAR(c.getCutoff(), 20.0f, 1e-6f);
}

DSPARK_TEST(Ladder_block_matches_per_sample)
{
    // The block path must match the per-sample path (same snapshot, same
    // coefficients, same step) across modes and drive. The match is bit-exact
    // unless the compiler contracts multiply-adds: it may fuse the two inlined
    // copies differently, and then the last bits move but nothing else may.
#if defined(__FP_FAST_FMAF) || defined(__AVX2__)
    constexpr float kTolerance = 4e-6f;
#else
    constexpr float kTolerance = 0.0f;
#endif
    using M = LadderFilter<float>::Mode;
    for (M mode : { M::LP24, M::HP24, M::BP12 })
    {
        LadderFilter<float> blockF, sampleF;
        for (auto* f : { &blockF, &sampleF })
        {
            f->prepare(spec48k());
            f->setCutoff(900.0f);
            f->setResonance(0.85f);
            f->setDrive(2.5f);
            f->setMode(mode);
        }
        uint32_t rng = 0xFACEu;
        int mismatches = 0;
        const int sizes[] = { 64, 129, 37, 300, 11 };
        for (int blockSize : sizes)
        {
            auto tb = makeStereoBuffer(blockSize);
            std::vector<float> in(static_cast<size_t>(blockSize));
            for (int i = 0; i < blockSize; ++i)
            {
                rng = rng * 1664525u + 1013904223u;
                in[static_cast<size_t>(i)] = ((rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
                tb.ch(0)[i] = in[static_cast<size_t>(i)];
                tb.ch(1)[i] = in[static_cast<size_t>(i)];
            }
            blockF.processBlock(tb.view());
            for (int i = 0; i < blockSize; ++i)
                if (std::abs(tb.ch(0)[i] - sampleF.processSample(in[static_cast<size_t>(i)], 0)) > kTolerance)
                    ++mismatches;
        }
        EXPECT_EQ(mismatches, 0);
    }
}

// ============================================================================
// LadderFilter - per-stage saturation (Saturation::Stages)
// ============================================================================

namespace
{
// Oversampled RK4 of the per-stage tanh ladder ODE the Stages model solves:
// dy_i/dt = wc (phi(y_{i-1}) - phi(y_i)), stage-0 input x - k y_4,
// phi(v) = tanh(d v) / d.
struct LadderOdeReference
{
    double wc, k, d;
    std::array<double, 4> y{};
    void deriv(const std::array<double, 4>& v, double x, std::array<double, 4>& dv) const
    {
        double prev = std::tanh(d * (x - k * v[3])) / d;
        for (size_t i = 0; i < 4; ++i)
        {
            const double ph = std::tanh(d * v[i]) / d;
            dv[i] = wc * (prev - ph);
            prev = ph;
        }
    }
    template <class X> void step(double t, double h, X xin)
    {
        std::array<double, 4> k1{}, k2{}, k3{}, k4{}, tmp{};
        const double x0 = xin(t), xm = xin(t + 0.5 * h), x1 = xin(t + h);
        deriv(y, x0, k1);
        for (size_t i = 0; i < 4; ++i) tmp[i] = y[i] + 0.5 * h * k1[i];
        deriv(tmp, xm, k2);
        for (size_t i = 0; i < 4; ++i) tmp[i] = y[i] + 0.5 * h * k2[i];
        deriv(tmp, xm, k3);
        for (size_t i = 0; i < 4; ++i) tmp[i] = y[i] + h * k3[i];
        deriv(tmp, x1, k4);
        for (size_t i = 0; i < 4; ++i) y[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }
};

double ladderHarmonic(const std::vector<double>& s, int perPeriod, int periods, int m)
{
    const int n0 = static_cast<int>(s.size()) - perPeriod * periods, n = perPeriod * periods;
    std::complex<double> acc(0, 0);
    for (int i = 0; i < n; ++i)
        acc += s[static_cast<size_t>(n0 + i)] * std::polar(1.0, -twoPi<double> * m * i / perPeriod);
    return 2.0 * std::abs(acc) / n;
}

template <class F, class S>
std::vector<double> ladderRender(F& filter, S, double fs, double fc, double res, double drive, bool stages,
                                 int total, double f0)
{
    filter.prepare(AudioSpec{ fs, 512, 1 });
    filter.setCutoff(static_cast<S>(fc));
    filter.setResonance(static_cast<S>(res));
    filter.setDrive(static_cast<S>(drive));
    filter.setMode(F::Mode::LP24);
    filter.setSaturation(stages ? F::Saturation::Stages : F::Saturation::Feedback);
    std::vector<double> out(static_cast<size_t>(total));
    for (int n = 0; n < total; ++n)
        out[static_cast<size_t>(n)] =
            filter.processSample(static_cast<S>(std::sin(twoPi<double> * f0 * (n + 1) / fs)), 0);
    return out;
}
} // namespace

DSPARK_TEST(Ladder_stages_mode_matches_the_ladder_ode)
{
    // Per-stage saturation against a 32x oversampled RK4 of the same ODE,
    // 110 Hz sine through fc 2 kHz, resonance 0.5, drive 3. The Stages model
    // tracks the fundamental and third harmonic to a fraction of a dB, in
    // float and double; the feedback-only model misses the third harmonic by
    // several dB, the stage distortion it does not model.
    constexpr double fs = 48000.0, fc = 2000.0, res = 0.5, drive = 3.0;
    constexpr int perPeriod = 436, periods = 70, measured = 10, os = 32;
    const int total = perPeriod * periods;
    const double f0 = fs / perPeriod;
    auto xin = [&](double t) { return std::sin(twoPi<double> * f0 * t); };
    LadderOdeReference ode{ twoPi<double> * fc, 4.0 * res, drive };
    std::vector<double> ref(static_cast<size_t>(total));
    const double h = 1.0 / (fs * os);
    for (int n = 0; n < total; ++n)
    {
        for (int j = 0; j < os; ++j) ode.step((n * os + j) * h, h, xin);
        ref[static_cast<size_t>(n)] = ode.y[3];
    }
    LadderFilter<double> sd, fd;
    LadderFilter<float> sf;
    const auto stagesD = ladderRender(sd, 0.0, fs, fc, res, drive, true, total, f0);
    const auto stagesF = ladderRender(sf, 0.0f, fs, fc, res, drive, true, total, f0);
    const auto feedback = ladderRender(fd, 0.0, fs, fc, res, drive, false, total, f0);
    for (int m : { 1, 3 })
    {
        const double r = ladderHarmonic(ref, perPeriod, measured, m);
        EXPECT_NEAR(20 * std::log10(ladderHarmonic(stagesD, perPeriod, measured, m) / r), 0.0, 0.3);
        EXPECT_NEAR(20 * std::log10(ladderHarmonic(stagesF, perPeriod, measured, m) / r), 0.0, 0.3);
    }
    const double r3 = ladderHarmonic(ref, perPeriod, measured, 3);
    EXPECT_GT(std::abs(20 * std::log10(ladderHarmonic(feedback, perPeriod, measured, 3) / r3)), 3.0);
}

DSPARK_TEST(Ladder_stages_self_oscillation_pitch_and_level)
{
    // Resonance 1.1 at fc 1 kHz, drive 1, kicked by 1 ms of 0.1: a 128x RK4
    // of the ladder ODE settles at 979.54 Hz with a 0.1497 peak. The Stages
    // model lands within a cent and 0.2 dB of that limit cycle.
    constexpr double fs = 48000.0;
    LadderFilter<float> f;
    f.prepare(AudioSpec{ fs, 512, 1 });
    f.setCutoff(1000.0f);
    f.setResonance(1.1f);
    f.setDrive(1.0f);
    f.setMode(LadderFilter<float>::Mode::LP24);
    f.setSaturation(LadderFilter<float>::Saturation::Stages);
    const int total = static_cast<int>(fs * 3), n0 = total - static_cast<int>(fs);
    std::vector<double> y(static_cast<size_t>(total));
    for (int n = 0; n < total; ++n)
        y[static_cast<size_t>(n)] = f.processSample((n + 1) / fs < 1e-3 ? 0.1f : 0.0f, 0);
    double first = -1, last = -1, peak = 0;
    int crossings = 0;
    for (int i = n0; i < total - 1; ++i)
    {
        const double a = y[static_cast<size_t>(i)], b = y[static_cast<size_t>(i + 1)];
        peak = std::max(peak, std::abs(a));
        if (a <= 0 && b > 0)
        {
            const double t = i + a / (a - b);
            if (first < 0) first = t;
            last = t;
            ++crossings;
        }
    }
    EXPECT_TRUE(crossings > 100);
    if (crossings < 2) return;
    const double freq = fs * (crossings - 1) / (last - first);
    EXPECT_NEAR(1200.0 * std::log2(freq / 979.54), 0.0, 1.0);
    EXPECT_NEAR(20 * std::log10(peak / 0.1497), 0.0, 0.2);
}

DSPARK_TEST(Ladder_stages_block_matches_per_sample_and_state_round_trips)
{
    using L = LadderFilter<double>;
    L blockF, sampleF;
    for (auto* f : { &blockF, &sampleF })
    {
        f->prepare(AudioSpec{ 48000.0, 512, 1 });
        f->setCutoff(1500.0);
        f->setResonance(0.9);
        f->setDrive(4.0);
        f->setSaturation(L::Saturation::Stages);
    }
    std::vector<double> x(700);
    uint32_t rng = 0xBEEFu;
    for (auto& v : x)
    {
        rng = rng * 1664525u + 1013904223u;
        v = ((rng >> 8) * (1.0 / 16777216.0)) - 0.5;
    }
    std::vector<double> blockOut = x;
    for (int at = 0; at < 700; at += 175)
    {
        double* part[1]{ blockOut.data() + at };
        blockF.processBlock(AudioBufferView<double>(part, 1, 175));
    }
    double worst = 0;
    for (size_t i = 0; i < x.size(); ++i)
        worst = std::max(worst, std::abs(blockOut[i] - sampleF.processSample(x[i], 0)));
    EXPECT_LT(worst, 1e-12);

    const auto blob = blockF.getState();
    L restored;
    restored.prepare(AudioSpec{ 48000.0, 512, 1 });
    EXPECT_TRUE(restored.setState(blob.data(), blob.size()));
    EXPECT_TRUE(restored.getSaturation() == L::Saturation::Stages);
    EXPECT_NEAR(restored.getCutoff(), 1500.0, 1e-9);
    EXPECT_NEAR(restored.getResonance(), 0.9, 1e-6);
    EXPECT_NEAR(restored.getDrive(), 4.0, 1e-6);
    restored.setSaturation(static_cast<L::Saturation>(7));   // wild value
    EXPECT_TRUE(restored.getSaturation() == L::Saturation::Feedback);
}

DSPARK_TEST(Ladder_stages_either_start_reaches_the_converged_solution)
{
    // Below kLadderStagesStateStartGain Newton starts at the integrator
    // states (56 instead of 74 ns per sample at 1 kHz, drive 3), above it at
    // the fixed-pivot solve. Either way the step lands on the solution a
    // deeply converged solve (tolerance 1e-14 knee widths) reaches, here on
    // loud noise on both sides of the switch, with and without resonance.
    for (double fc : { 300.0, 2200.0, 2300.0, 9000.0 })
        for (double res : { 0.0, 0.9, 1.1 })
            for (double drive : { 1.0, 4.0 })
            {
                const double g = std::tan(pi<double> * fc / 48000.0);
                const auto c = detail::makeLadderStageCoeffs(g, res, drive);
                auto deep = c;
                deep.tol = 1e-14;
                std::array<double, 4> z{}, zDeep{};
                uint32_t rng = 0xC0FFEEu;
                double worst = 0;
                for (int i = 0; i < 2000; ++i)
                {
                    rng = rng * 1664525u + 1013904223u;
                    const double x = 1.6 * (((rng >> 8) * (1.0 / 16777216.0)) - 0.5);
                    const double y = detail::ladderStepStages<detail::LadderTap::LP24>(x, z, c);
                    const double yDeep = detail::ladderStepStages<detail::LadderTap::LP24>(x, zDeep, deep);
                    worst = std::max(worst, std::abs(y - yDeep) * drive);
                    zDeep = z;   // same states for the next sample
                }
                EXPECT_LT(worst, 1e-11);
            }
    EXPECT_GT(detail::kLadderStagesStateStartGain, std::tan(pi<double> * 2200.0 / 48000.0));
    EXPECT_LT(detail::kLadderStagesStateStartGain, std::tan(pi<double> * 2300.0 / 48000.0));
}

DSPARK_TEST(Ladder_stages_linear_start_reaches_the_converged_solution)
{
    // Above the state-start gain Newton now starts from the drive-free
    // ladder solved exactly while its stages stay within two knee widths
    // (a fixed-pivot solve otherwise): fewer steps for the same root. Pin:
    // on sines, a saw and noise at high cutoffs, every sample lands on the
    // solution of a deeply converged solve from the same states.
    for (double fc : { 5000.0, 15000.0, 21000.0 })
        for (double drive : { 0.5, 1.0, 3.0, 10.0 })
            for (int signal = 0; signal < 3; ++signal)
            {
                const double g = std::tan(pi<double> * fc / 48000.0);
                const auto c = detail::makeLadderStageCoeffs(g, 0.9, drive);
                auto deep = c;
                deep.tol = 1e-14;
                std::array<double, 4> z{}, zDeep{};
                uint32_t rng = 0xACE1u;
                double worst = 0;
                for (int i = 0; i < 2000; ++i)
                {
                    rng = rng * 1664525u + 1013904223u;
                    const double t = static_cast<double>(i) / 48000.0;
                    const double x = signal == 0 ? 0.9 * std::sin(2.0 * pi<double> * 110.0 * t)
                                   : signal == 1 ? 2.0 * std::fmod(110.0 * t, 1.0) - 1.0
                                                 : 1.6 * (((rng >> 8) * (1.0 / 16777216.0)) - 0.5);
                    const double y = detail::ladderStepStages<detail::LadderTap::LP24>(x, z, c);
                    const double yDeep = detail::ladderStepStages<detail::LadderTap::LP24>(x, zDeep, deep);
                    worst = std::max(worst, std::abs(y - yDeep) * drive);
                    zDeep = z;
                }
                EXPECT_LT(worst, 1e-11);
            }
}

DSPARK_TEST(Ladder_stages_channel_pairs_match_single_channels)
{
    // Speed-up pin: the Stages block loop advances channel pairs together
    // (their solves overlap). Each channel's output must stay bit-identical
    // to the same channel processed alone, for an odd channel count too.
    using L = LadderFilter<float>;
    constexpr int kCh = 3, kN = 700;
    std::vector<std::vector<float>> in(kCh, std::vector<float>(kN));
    uint32_t rng = 0x5EEDu;
    for (auto& ch : in)
        for (auto& v : ch)
        {
            rng = rng * 1664525u + 1013904223u;
            v = static_cast<float>(((rng >> 8) * (1.0 / 16777216.0)) - 0.5) * 1.8f;
        }
    const auto setup = [](L& f, int channels) {
        f.prepare(AudioSpec{ 48000.0, 256, channels });
        f.setCutoff(15000.0f);
        f.setResonance(0.9f);
        f.setDrive(3.0f);
        f.setSaturation(L::Saturation::Stages);
    };
    L multi;
    setup(multi, kCh);
    auto out = in;
    for (int at = 0; at < kN; at += 175)
    {
        float* ptr[kCh] = { out[0].data() + at, out[1].data() + at, out[2].data() + at };
        multi.processBlock(AudioBufferView<float>(ptr, kCh, 175));
    }
    for (int ch = 0; ch < kCh; ++ch)
    {
        L single;
        setup(single, 1);
        auto alone = in[static_cast<size_t>(ch)];
        for (int at = 0; at < kN; at += 175)
        {
            float* ptr[1] = { alone.data() + at };
            single.processBlock(AudioBufferView<float>(ptr, 1, 175));
        }
        EXPECT_TRUE(alone == out[static_cast<size_t>(ch)]);
    }
}

// ============================================================================
// LadderFilter - input saturation (Saturation::Input)
// ============================================================================

namespace
{
// The Input model's equations at a rate high enough to stand for the
// continuous system: trapezoidal stages, exact tanh on the loop input, the
// loop solved to convergence.
struct LadderInputReference
{
    double G = 0.0, H = 1.0, G4 = 0.0, k = 0.0, d = 1.0;
    std::array<double, 4> s{};

    LadderInputReference(double rate, double fc, double res, double drive)
    {
        const double g = std::tan(pi<double> * fc / rate);
        G = g / (1.0 + g);
        H = 1.0 - G;
        G4 = (G * G) * (G * G);
        k = 4.0 * res;
        d = drive;
    }

    double step(double x)
    {
        const double sum = H * (((G * G) * G) * s[0] + (G * G) * s[1] + G * s[2] + s[3]);
        const double c = k * G4, r = d * (x - k * sum);
        double u = r;
        for (int it = 0; it < 60; ++it)
        {
            const double t = std::tanh(u);
            const double delta = (u + c * t - r) / (1.0 + c * (1.0 - t * t));
            u -= delta;
            if (std::abs(delta) < 1e-15) break;
        }
        double in = std::tanh(u) / d;
        for (size_t i = 0; i < 4; ++i)
        {
            const double v = (in - s[i]) * G;
            const double y = v + s[i];
            s[i] = y + v;
            in = y;
        }
        return in;
    }
};

template <class F>
void ladderInputSetup(F& filter, double fs, int channels, double fc, double res, double drive)
{
    using S = decltype(filter.getCutoff());
    filter.prepare(AudioSpec{ fs, 512, channels });
    filter.setCutoff(static_cast<S>(fc));
    filter.setResonance(static_cast<S>(res));
    filter.setDrive(static_cast<S>(drive));
    filter.setMode(F::Mode::LP24);
    filter.setSaturation(F::Saturation::Input);
}

// Magnitude of the discrete-time Fourier transform of h at freq.
double ladderResponseAt(const std::vector<double>& h, double freq, double fs)
{
    const std::complex<double> step = std::polar(1.0, -twoPi<double> * freq / fs);
    std::complex<double> acc(0, 0), w(1, 0);
    for (double v : h)
    {
        acc += v * w;
        w *= step;
    }
    return std::abs(acc);
}
} // namespace

DSPARK_TEST(Ladder_input_mode_matches_its_equations)
{
    // One tanh on the loop input, linear stages, four sub-steps per sample.
    // Against the same equations 32 times oversampled (exact tanh, the loop
    // solved to convergence), a full-scale 110 Hz sine at resonance 0.5 and
    // drive 3: the harmonics through the seventh land within 0.1 dB at a
    // 2 kHz cutoff and at 12 kHz, in float and double. The per-stage model at
    // 12 kHz is 2 dB off its own equations: it solves at the sample rate.
    constexpr double fs = 48000.0, res = 0.5, drive = 3.0;
    constexpr int perPeriod = 436, periods = 70, measured = 10, os = 32;
    const int total = perPeriod * periods;
    const double f0 = fs / perPeriod;
    for (double fc : { 2000.0, 12000.0 })
    {
        LadderInputReference reference(fs * os, fc, res, drive);
        std::vector<double> truth(static_cast<size_t>(total));
        for (int n = 0; n < total; ++n)
        {
            double y = 0.0;
            for (int j = 0; j < os; ++j)
                y = reference.step(std::sin(twoPi<double> * f0 * (n + double(j + 1) / os) / fs));
            truth[static_cast<size_t>(n)] = y;
        }
        LadderFilter<double> fd;
        LadderFilter<float> ff;
        ladderInputSetup(fd, fs, 1, fc, res, drive);
        ladderInputSetup(ff, fs, 1, fc, res, drive);
        std::vector<double> outD(truth.size()), outF(truth.size());
        for (int n = 0; n < total; ++n)
        {
            const double x = std::sin(twoPi<double> * f0 * (n + 1) / fs);
            outD[static_cast<size_t>(n)] = fd.processSample(x, 0);
            outF[static_cast<size_t>(n)] = ff.processSample(static_cast<float>(x), 0);
        }
        const double r1 = ladderHarmonic(truth, perPeriod, measured, 1);
        EXPECT_GT(ladderHarmonic(truth, perPeriod, measured, 3), 0.03 * r1);   // it does saturate
        for (int m : { 1, 3, 5, 7 })
        {
            const double r = ladderHarmonic(truth, perPeriod, measured, m);
            EXPECT_NEAR(20 * std::log10(ladderHarmonic(outD, perPeriod, measured, m) / r), 0.0, 0.1);
            EXPECT_NEAR(20 * std::log10(ladderHarmonic(outF, perPeriod, measured, m) / r), 0.0, 0.1);
        }
    }
}

DSPARK_TEST(Ladder_input_resonance_is_in_tune_and_the_open_filter_is_flat)
{
    // Small signal. The analog ladder at resonance 0.95 (k = 3.8) peaks at
    // 0.98714 of its cutoff. The Input model's peak lies within 1.5 cents of
    // that at 1, 8 and 16 kHz, where the models that integrate at the sample
    // rate are 0.05, 3.8 and 13 cents sharp. With the cutoff at 20 kHz and no
    // resonance, 5 and 10 kHz pass within 0.75 dB of the analog four-pole
    // gain (the level lost to the sub-steps is corrected).
    constexpr double fs = 48000.0;
    double peakRatio = 0.0, best = 0.0;
    for (double x = 0.9; x < 1.05; x += 1e-6)
    {
        const std::complex<double> s(1.0, x);
        const double m = 1.0 / std::abs((s * s) * (s * s) + 3.8);
        if (m > best) { best = m; peakRatio = x; }
    }
    EXPECT_NEAR(peakRatio, 0.98714, 1e-4);
    for (double fc : { 1000.0, 8000.0, 16000.0 })
    {
        LadderFilter<double> f;
        ladderInputSetup(f, fs, 1, fc, 0.95, 1.0);
        std::vector<double> h(32768);
        for (size_t n = 0; n < h.size(); ++n) h[n] = f.processSample(n == 0 ? 1e-3 : 0.0, 0);
        const double target = peakRatio * fc;
        double bestCents = 0.0, bestMag = 0.0;
        for (double cents = -4.0; cents <= 4.0; cents += 0.125)
        {
            const double mag = ladderResponseAt(h, target * std::pow(2.0, cents / 1200.0), fs);
            if (mag > bestMag) { bestMag = mag; bestCents = cents; }
        }
        EXPECT_LT(std::abs(bestCents), 1.5);
    }
    for (double freq : { 5000.0, 10000.0 })
    {
        LadderFilter<double> f;
        ladderInputSetup(f, fs, 1, 20000.0, 0.0, 1.0);
        const int period = 48, settle = 100 * period, measure = 100 * period;
        const double w = twoPi<double> * freq / fs;
        double re = 0.0, im = 0.0;
        for (int n = 0; n < settle + measure; ++n)
        {
            const double y = f.processSample(1e-3 * std::sin(w * n), 0);
            if (n >= settle) { re += y * std::cos(w * n); im += y * std::sin(w * n); }
        }
        const double gainDb = 20.0 * std::log10(2.0 * std::sqrt(re * re + im * im) / measure / 1e-3);
        const double analogDb = -40.0 * std::log10(1.0 + (freq / 20000.0) * (freq / 20000.0));
        EXPECT_NEAR(gainDb, analogDb, 0.75);
    }
}

DSPARK_TEST(Ladder_input_block_pair_and_per_sample_paths_agree)
{
    // The model's state is double and one function advances it: a block, the
    // same samples one by one and a channel of a pair give the same output,
    // bit for bit, across a cutoff change and in every mode.
    using L = LadderFilter<float>;
    constexpr int kN = 700;
    std::vector<std::vector<float>> in(3, std::vector<float>(kN));
    uint32_t rng = 0x1ADDu;
    for (auto& ch : in)
        for (auto& v : ch)
        {
            rng = rng * 1664525u + 1013904223u;
            v = static_cast<float>(((rng >> 8) * (1.0 / 16777216.0)) - 0.5) * 1.8f;
        }
    for (L::Mode mode : { L::Mode::LP24, L::Mode::LP6, L::Mode::BP12, L::Mode::HP24 })
    {
        L multi;
        ladderInputSetup(multi, 48000.0, 3, 3000.0, 0.9, 3.0);
        multi.setMode(mode);
        auto out = in;
        for (int at = 0; at < kN; at += 175)
        {
            if (at == 350) multi.setCutoff(9000.0f);
            float* ptr[3] = { out[0].data() + at, out[1].data() + at, out[2].data() + at };
            multi.processBlock(AudioBufferView<float>(ptr, 3, 175));
        }
        for (int ch = 0; ch < 3; ++ch)
        {
            L single;
            ladderInputSetup(single, 48000.0, 1, 3000.0, 0.9, 3.0);
            single.setMode(mode);
            std::vector<float> alone(kN);
            for (int n = 0; n < kN; ++n)
            {
                if (n == 350) single.setCutoff(9000.0f);
                alone[static_cast<size_t>(n)] = single.processSample(in[static_cast<size_t>(ch)][static_cast<size_t>(n)], 0);
            }
            EXPECT_TRUE(alone == out[static_cast<size_t>(ch)]);
        }
    }
}

DSPARK_TEST(Ladder_input_switches_self_oscillation_and_state)
{
    using L = LadderFilter<float>;
    // Switching the saturation model mid-stream hands the integrators over:
    // a 220 Hz tone stays finite and bounded through every change.
    {
        L f;
        ladderInputSetup(f, 48000.0, 1, 3000.0, 0.3, 1.0);
        const L::Saturation order[] = { L::Saturation::Feedback, L::Saturation::Input, L::Saturation::Stages,
                                        L::Saturation::Input, L::Saturation::Feedback };
        double peak = 0.0;
        bool finite = true;
        int n = 0;
        for (L::Saturation model : order)
        {
            f.setSaturation(model);
            for (int i = 0; i < 2400; ++i, ++n)
            {
                const float y = f.processSample(0.5f * static_cast<float>(std::sin(twoPi<double> * 220.0 * n / 48000.0)), 0);
                finite = finite && std::isfinite(y);
                peak = std::max(peak, static_cast<double>(std::abs(y)));
            }
        }
        EXPECT_TRUE(finite);
        EXPECT_LT(peak, 1.0);
        EXPECT_GT(peak, 0.1);
    }
    // Above resonance 1 the saturator bounds the oscillation: kicked by 1 ms
    // of 0.1, it is still ringing half a second later and never leaves the
    // range a knee-wide swing allows.
    {
        L f;
        ladderInputSetup(f, 48000.0, 1, 2000.0, 1.15, 1.0);
        double peak = 0.0, tail = 0.0;
        for (int n = 0; n < 24000; ++n)
        {
            const double y = f.processSample(n < 48 ? 0.1f : 0.0f, 0);
            peak = std::max(peak, std::abs(y));
            if (n >= 19200) tail += y * y;
        }
        EXPECT_LT(peak, 2.0);
        EXPECT_GT(std::sqrt(tail / 4800.0), 0.01);
    }
    // The model survives a state round trip, and a NaN sample does not
    // silence the channel past the block it arrived in.
    {
        L f;
        ladderInputSetup(f, 48000.0, 1, 1500.0, 0.9, 4.0);
        const auto blob = f.getState();
        L restored;
        restored.prepare(AudioSpec{ 48000.0, 512, 1 });
        EXPECT_TRUE(restored.setState(blob.data(), blob.size()));
        EXPECT_TRUE(restored.getSaturation() == L::Saturation::Input);
        std::vector<float> block(256, 0.25f);
        block[10] = std::numeric_limits<float>::quiet_NaN();
        float* ptr[1] = { block.data() };
        f.processBlock(AudioBufferView<float>(ptr, 1, 256));
        std::fill(block.begin(), block.end(), 0.25f);
        f.processBlock(AudioBufferView<float>(ptr, 1, 256));
        EXPECT_TRUE(std::isfinite(block[255]));
        EXPECT_GT(std::abs(block[255]), 0.01f);
    }
}

// ============================================================================
// StateVariableFilter - analytic exactness and robustness
// ============================================================================

DSPARK_TEST(SVF_core_gains_match_analog_prototype)
{
    // At the prewarped cutoff the analog SVF prototype gives |LP| = |HP| =
    // |BP| = Q exactly, and Butterworth Q keeps the passband maximally flat.
    using M = StateVariableFilter<float>::Mode;
    const float q = 0.707107f;
    const struct { M mode; float freq; float expected; } cases[] = {
        { M::LowPass,  1000.0f, q },     // |H(fc)| = Q, all three taps
        { M::HighPass, 1000.0f, q },
        { M::BandPass, 1000.0f, q },
        { M::LowPass,   100.0f, 1.0f },  // Butterworth passband is flat
    };
    for (const auto& tc : cases)
    {
        StateVariableFilter<float> svf;
        svf.prepare(spec48k());
        svf.setCutoff(1000.0f);
        svf.setQ(q);
        svf.setMode(tc.mode);
        const float gain = tptGainAt(svf, tc.freq, 48000.0f);
        EXPECT_NEAR(gain, tc.expected, tc.expected * 0.015f);
    }

    // Resonant case: |BP(fc)| = Q exactly, for a high Q too.
    StateVariableFilter<float> res;
    res.prepare(spec48k());
    res.setCutoff(1000.0f);
    res.setQ(8.0f);
    res.setMode(M::BandPass);
    const float bpPeak = tptGainAt(res, 1000.0f, 48000.0f);
    EXPECT_NEAR(bpPeak, 8.0f, 0.2f);
}

DSPARK_TEST(SVF_notch_allpass_and_reconstruction)
{
    // Notch (LP+HP) nulls the centre, allpass is unity magnitude everywhere,
    // and lp + 2R*bp + hp == input reconstructs by construction.
    StateVariableFilter<float> notch;
    notch.prepare(spec48k());
    notch.setCutoff(1000.0f);
    notch.setQ(0.707107f);
    notch.setMode(StateVariableFilter<float>::Mode::Notch);
    EXPECT_LT(tptGainAt(notch, 1000.0f, 48000.0f), 0.02f);

    StateVariableFilter<float> ap;
    ap.prepare(spec48k());
    ap.setCutoff(1000.0f);
    ap.setQ(0.707107f);
    ap.setMode(StateVariableFilter<float>::Mode::AllPass);
    for (float freq : { 100.0f, 1000.0f, 10000.0f })
    {
        ap.reset();
        EXPECT_NEAR(tptGainAt(ap, freq, 48000.0f), 1.0f, 0.01f);
    }

    StateVariableFilter<float> svf;
    svf.prepare(spec48k());
    svf.setCutoff(1500.0f);
    svf.setQ(2.0f);
    const float twoR = 1.0f / 2.0f;  // 2R = 1/Q
    uint32_t rng = 0x5EED5u;
    float maxErr = 0.0f;
    for (int i = 0; i < 2000; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        const float in = ((rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
        auto [lp, hp, bp] = svf.processMultiOutput(in, 0);
        maxErr = std::max(maxErr, std::abs(lp + twoR * bp + hp - in));
    }
    EXPECT_LT(maxErr, 1e-4f);
}

DSPARK_TEST(SVF_bell_and_shelf_gains_are_exact)
{
    using M = StateVariableFilter<float>::Mode;

    // Bell: gain at the centre is exactly 10^(dB/20), boost AND cut (the cut
    // sign fix), and out-of-band stays unity.
    for (float dB : { 12.0f, -12.0f })
    {
        StateVariableFilter<float> bell;
        bell.prepare(spec48k());
        bell.setCutoff(1000.0f);
        bell.setQ(1.0f);
        bell.setGain(dB);
        bell.setMode(M::Bell);
        const float expected = std::pow(10.0f, dB / 20.0f);
        EXPECT_NEAR(tptGainAt(bell, 1000.0f, 48000.0f), expected, expected * 0.02f);
        bell.reset();
        EXPECT_NEAR(tptGainAt(bell, 50.0f, 48000.0f), 1.0f, 0.02f);
    }

    // LowShelf +18 dB: full gain in the low band, exactly half the dB at the
    // nominal frequency (Simper sqrt(A) prewarp), unity far above.
    StateVariableFilter<float> ls;
    ls.prepare(spec48k());
    ls.setCutoff(1000.0f);
    ls.setQ(0.707107f);
    ls.setGain(18.0f);
    ls.setMode(M::LowShelf);
    EXPECT_NEAR(tptGainAt(ls, 50.0f, 48000.0f), std::pow(10.0f, 18.0f / 20.0f),
                std::pow(10.0f, 18.0f / 20.0f) * 0.03f);
    ls.reset();
    EXPECT_NEAR(tptGainAt(ls, 1000.0f, 48000.0f), std::pow(10.0f, 9.0f / 20.0f),
                std::pow(10.0f, 9.0f / 20.0f) * 0.03f);
    ls.reset();
    EXPECT_NEAR(tptGainAt(ls, 20000.0f, 48000.0f), 1.0f, 0.03f);

    // HighShelf -12 dB mirror: unity low, half-dB at fc, full cut high.
    StateVariableFilter<float> hs;
    hs.prepare(spec48k());
    hs.setCutoff(1000.0f);
    hs.setQ(0.707107f);
    hs.setGain(-12.0f);
    hs.setMode(M::HighShelf);
    EXPECT_NEAR(tptGainAt(hs, 50.0f, 48000.0f), 1.0f, 0.03f);
    hs.reset();
    EXPECT_NEAR(tptGainAt(hs, 1000.0f, 48000.0f), std::pow(10.0f, -6.0f / 20.0f),
                std::pow(10.0f, -6.0f / 20.0f) * 0.03f);
    hs.reset();
    EXPECT_NEAR(tptGainAt(hs, 20000.0f, 48000.0f), std::pow(10.0f, -12.0f / 20.0f),
                std::pow(10.0f, -12.0f / 20.0f) * 0.03f);
}

DSPARK_TEST(SVF_invalid_inputs_ignored_and_channel_guard)
{
    // Twin A/B: NaN/Inf setters, an invalid prepare AND out-of-range channel
    // calls must leave the filter bit-identical to an untouched twin. With
    // the old header setResonance(NaN) poisoned R_ permanently and
    // processSample(x, 16) wrote out of bounds (measured: it landed on
    // spec_.sampleRate, corrupting every later cutoff clamp).
    StateVariableFilter<float> a, b;
    for (auto* f : { &a, &b })
    {
        f->prepare(spec48k());
        f->setCutoff(800.0f);
        f->setQ(3.0f);
        f->setGain(6.0f);
        f->setMode(StateVariableFilter<float>::Mode::Bell);
    }
    b.setCutoff(std::numeric_limits<float>::quiet_NaN());
    b.setCutoff(std::numeric_limits<float>::infinity());
    b.setResonance(std::numeric_limits<float>::quiet_NaN());
    b.setQ(std::numeric_limits<float>::quiet_NaN());
    b.setGain(std::numeric_limits<float>::quiet_NaN());
    AudioSpec bad;
    bad.sampleRate = 0.0;
    b.prepare(bad);                                   // ignored
    EXPECT_NEAR(b.processSample(0.5f, -1), 0.5f, 0.0f);   // guarded pass-through
    EXPECT_NEAR(b.processSample(0.5f, 16), 0.5f, 0.0f);
    const auto mo = b.processMultiOutput(0.25f, 99);
    EXPECT_NEAR(mo.lowpass, 0.25f, 0.0f);
    EXPECT_NEAR(mo.highpass, 0.0f, 0.0f);
    EXPECT_NEAR(mo.bandpass, 0.0f, 0.0f);

    EXPECT_NEAR(b.getCutoff(), 800.0f, 1e-6f);
    EXPECT_NEAR(b.getQ(), 3.0f, 1e-5f);
    EXPECT_NEAR(b.getGain(), 6.0f, 1e-6f);

    uint32_t rng = 0x777u;
    int mismatches = 0;
    for (int i = 0; i < 4800; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        const float in = ((rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
        if (a.processSample(in, 0) != b.processSample(in, 0)) ++mismatches;
    }
    EXPECT_EQ(mismatches, 0);
}

DSPARK_TEST(SVF_block_matches_per_sample)
{
    // The block path must be bit-identical to the per-sample path, across
    // plain, bell and shelf modes.
    using M = StateVariableFilter<float>::Mode;
    for (M mode : { M::LowPass, M::Bell, M::HighShelf, M::AllPass })
    {
        StateVariableFilter<float> blockF, sampleF;
        for (auto* f : { &blockF, &sampleF })
        {
            f->prepare(spec48k());
            f->setCutoff(1100.0f);
            f->setQ(4.0f);
            f->setGain(-9.0f);
            f->setMode(mode);
        }
        uint32_t rng = 0xFACEu;
        int mismatches = 0;
        const int sizes[] = { 64, 129, 37, 300, 11 };
        for (int blockSize : sizes)
        {
            auto tb = makeStereoBuffer(blockSize);
            std::vector<float> in(static_cast<size_t>(blockSize));
            for (int i = 0; i < blockSize; ++i)
            {
                rng = rng * 1664525u + 1013904223u;
                in[static_cast<size_t>(i)] = ((rng >> 8) * (1.0f / 16777216.0f)) - 0.5f;
                tb.ch(0)[i] = in[static_cast<size_t>(i)];
                tb.ch(1)[i] = in[static_cast<size_t>(i)];
            }
            blockF.processBlock(tb.view());
            for (int i = 0; i < blockSize; ++i)
                if (tb.ch(0)[i] != sampleF.processSample(in[static_cast<size_t>(i)], 0)) ++mismatches;
        }
        EXPECT_EQ(mismatches, 0);
    }
}

// ============================================================================
// FIRFilter
// ============================================================================

DSPARK_TEST(FIR_LP_passband)
{
    auto taps = FIRDesign<float>::lowPass(44100.0f, 2000.0f, 65);

    FIRFilter<float> fir;
    fir.prepare(static_cast<int>(taps.size()), 1);
    fir.setCoefficients(taps);

    for (int i = 0; i < 256; ++i)
    {
        float in = std::sin(twoPi<float> * 200.0f * static_cast<float>(i) / 44100.0f);
        (void)fir.processSample(in, 0);
    }

    // Measure
    float peak = 0.0f;
    for (int i = 0; i < 4096; ++i)
    {
        float in = std::sin(twoPi<float> * 200.0f * static_cast<float>(256 + i) / 44100.0f);
        float out = fir.processSample(in, 0);
        float a = std::abs(out);
        if (a > peak) peak = a;
    }
    EXPECT_GT(peak, 0.85f);
}

DSPARK_TEST(FIR_LP_stopband)
{
    auto taps = FIRDesign<float>::lowPass(44100.0f, 2000.0f, 65);

    FIRFilter<float> fir;
    fir.prepare(static_cast<int>(taps.size()), 1);
    fir.setCoefficients(taps);

    for (int i = 0; i < 256; ++i)
    {
        float in = std::sin(twoPi<float> * 10000.0f * static_cast<float>(i) / 44100.0f);
        (void)fir.processSample(in, 0);
    }

    float peak = 0.0f;
    for (int i = 0; i < 4096; ++i)
    {
        float in = std::sin(twoPi<float> * 10000.0f * static_cast<float>(256 + i) / 44100.0f);
        float out = fir.processSample(in, 0);
        float a = std::abs(out);
        if (a > peak) peak = a;
    }
    EXPECT_LT(peak, 0.1f);
}

DSPARK_TEST(FIR_linear_phase_symmetry)
{
    auto taps = FIRDesign<float>::lowPass(44100.0f, 5000.0f, 33);

    // Linear phase FIR should have symmetric coefficients
    int n = static_cast<int>(taps.size());
    for (int i = 0; i < n / 2; ++i)
        EXPECT_NEAR(taps[i], taps[n - 1 - i], 1e-6f);
}

namespace {
// |H(f)| of a FIR kernel, evaluated directly in double.
float firMagnitudeAt(const std::vector<float>& h, float freq, float fs)
{
    double re = 0.0, im = 0.0;
    const double w = 2.0 * 3.14159265358979323846 * freq / fs;
    for (size_t k = 0; k < h.size(); ++k)
    {
        re += static_cast<double>(h[k]) * std::cos(w * static_cast<double>(k));
        im -= static_cast<double>(h[k]) * std::sin(w * static_cast<double>(k));
    }
    return static_cast<float>(std::sqrt(re * re + im * im));
}
float magDb(float m) { return 20.0f * std::log10(std::max(m, 1e-12f)); }
} // namespace

// highPass / bandPass / bandStop had NO coverage at all: pin their measured
// frequency responses at passband and stopband probes (101/151 taps with
// beta 7.857 gives ~2.4 / 1.6 kHz transition bands at 48 kHz, so every probe
// sits safely inside its band).
DSPARK_TEST(FIR_design_highpass_bandpass_bandstop_responses)
{
    constexpr float fs = 48000.0f;
    constexpr float beta = 7.857f; // ~-80 dB stopband class

    auto hp = FIRDesign<float>::highPass(fs, 4000.0f, 101, beta);
    EXPECT_LT(magDb(firMagnitudeAt(hp, 0.0f, fs)),     -70.0f); // DC rejected
    EXPECT_LT(magDb(firMagnitudeAt(hp, 1000.0f, fs)),  -70.0f); // stopband
    EXPECT_NEAR(magDb(firMagnitudeAt(hp, 20000.0f, fs)), 0.0f, 0.1f); // passband

    auto bp = FIRDesign<float>::bandPass(fs, 2000.0f, 8000.0f, 151, beta);
    EXPECT_LT(magDb(firMagnitudeAt(bp, 0.0f, fs)),     -70.0f);
    EXPECT_LT(magDb(firMagnitudeAt(bp, 300.0f, fs)),   -70.0f);
    EXPECT_NEAR(magDb(firMagnitudeAt(bp, 5000.0f, fs)), 0.0f, 0.2f);
    EXPECT_LT(magDb(firMagnitudeAt(bp, 15000.0f, fs)), -70.0f);

    auto bs = FIRDesign<float>::bandStop(fs, 2000.0f, 8000.0f, 151, beta);
    EXPECT_NEAR(magDb(firMagnitudeAt(bs, 0.0f, fs)),     0.0f, 0.1f);
    EXPECT_LT(magDb(firMagnitudeAt(bs, 4500.0f, fs)),  -60.0f); // notch floor
    EXPECT_NEAR(magDb(firMagnitudeAt(bs, 20000.0f, fs)), 0.0f, 0.2f);

    // All designs must stay symmetric (linear phase).
    for (const auto& taps : { hp, bp, bs })
    {
        const int n = static_cast<int>(taps.size());
        for (int i = 0; i < n / 2; ++i)
            EXPECT_NEAR(taps[static_cast<size_t>(i)],
                        taps[static_cast<size_t>(n - 1 - i)], 1e-7f);
    }
}

// Seqlock stress: a control thread hammers alternating coefficient sets while
// the audio thread convolves a DC signal. Set A = 32 taps of 1/32 (sum 1),
// set B = 64 taps of 3/64 (sum 3). With the whole delay line holding 1.0 the
// output equals the active kernel's sum EXACTLY, so any torn copy (mixed
// per-tap values and/or mismatched count) lands strictly between the two
// legal values and is caught.
DSPARK_TEST(FIR_seqlock_coefficient_swap_is_atomic)
{
    FIRFilter<float> fir;
    fir.prepare(64, 1);

    std::vector<float> setA(32, 1.0f / 32.0f);
    std::vector<float> setB(64, 3.0f / 64.0f);
    fir.setCoefficients(setA);

    // Fill the whole 64-tap history with DC 1.0.
    for (int i = 0; i < 128; ++i) (void)fir.processSample(1.0f, 0);

    std::atomic<bool> stop{false};
    // 64-bit: the producer spins far faster than the ASan-instrumented consumer
    // under parallel build load, so a 32-bit counter overflowed to a negative
    // value (observed -1878489712) and the >2000 liveness check saw a false
    // failure. int64 cannot wrap in any realistic run; the correctness assertion
    // (torn == 0) is unchanged.
    std::atomic<long long> published{0};
    std::thread producer([&] {
        bool useA = false;
        while (!stop.load(std::memory_order_relaxed))
        {
            if (useA) fir.setCoefficients(setA);
            else      fir.setCoefficients(setB);
            useA = !useA;
            published.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // The overlap is guaranteed by construction: the reader keeps going until it
    // has done its reads AND the producer has landed its publications. A fixed
    // read count finished before the producer thread was first scheduled on a
    // hosted macOS runner (0 publications), leaving nothing to check. The cap
    // makes a producer that never runs fail loudly.
    constexpr long long kReads = 400000;
    constexpr long long kMinPublications = 2001;
    constexpr long long kReadCap = 80000000LL;

    int torn = 0;
    long long reads = 0;
    while (reads < kReadCap
           && (reads < kReads
               || published.load(std::memory_order_relaxed) < kMinPublications))
    {
        const float out = fir.processSample(1.0f, 0);
        const bool isA = std::abs(out - 1.0f) < 1e-4f;
        const bool isB = std::abs(out - 3.0f) < 1e-4f;
        if (!isA && !isB) ++torn;
        ++reads;
    }

    stop.store(true, std::memory_order_relaxed);
    producer.join();

    EXPECT_EQ(torn, 0);
    EXPECT_TRUE(reads < kReadCap);   // the producer really ran
    EXPECT_GT(published.load(std::memory_order_relaxed), 2000);
}

// ============================================================================
// Bounded audio-thread seqlock reads (the real-time bound, not race freedom)
// ============================================================================
//
// The audio thread's coefficient pull enters its seqlock loop only when the
// dirty flag is set, and setCoefficients() raises that flag AFTER it leaves its
// critical section. So a read that begins while a publication STRICTLY NEWER
// than the set the reader holds has already completed is guaranteed to enter
// the loop. If such a read comes back still holding the older set, it did not
// adopt anything: the attempt bound ran out, the flag was re-armed, and the
// give-up path ran. That is a statement about which code executed, derived
// from published counters, and it needs no clock.
//
// The reachability of the give-up path is left to a writer that publishes
// CONTINUOUSLY rather than to a scheduled instant. This is deliberate, and it
// is the difference between this protocol and the one it replaces. The earlier
// shape armed the flag with a short publication, signalled, and then began a
// long one, hoping the reader would arrive while that long one was in flight;
// on a host that gives the writer no core between the signal and the start of
// the publication, the reader arrives to an idle writer, adopts, and the pin
// reports zero give-ups -- a red that says nothing about the code. Under
// continuous publication the pressure runs the other way: every attempt the
// reader makes must survive a full copy of the staged set without the counter
// moving, and the busier the host, the less likely that is. Starvation makes
// this pin easier to satisfy, never harder, which is the property a pin on a
// shared runner has to have.
//
// Every publication carries the full kBigTaps set, so an attempt costs a copy
// of that set and a writer that so much as STARTS a publication during it
// invalidates it. The buffer is stamped in place rather than rebuilt, so the
// writer allocates nothing between publications and the window in which its
// counter is even stays short.

static constexpr int kTagTaps = 512;        ///< The taps that carry the tag.
static constexpr int kBigTaps = 1 << 18;    ///< A publish long enough to collide with.

// A kernel whose first kTagTaps user taps are tag/kTagTaps and whose remaining
// taps are zero. FIRFilter stores coefficients reversed, so those taps multiply
// the NEWEST kTagTaps input samples: with the delay line holding DC 1.0, one
// processSample() returns exactly `tag` for any tap count, and a set mixed from
// two publications lands somewhere else. Every value here is a dyadic rational,
// so the sum is exact in float and the comparisons below can be exact.
static std::vector<float> tagKernel(float tag, int taps)
{
    std::vector<float> c(static_cast<size_t>(taps), 0.0f);
    const float v = tag / static_cast<float>(kTagTaps);
    for (int i = 0; i < kTagTaps; ++i)
        c[static_cast<size_t>(i)] = v;
    return c;
}

struct BoundedReadTrials
{
    long long reads   = 0;   ///< Audio-thread reads performed.
    long long opportunities = 0; ///< Reads GUARANTEED to enter the seqlock loop.
    long long publications  = 0; ///< Publications the writer completed.
    int deferrals     = 0;   ///< Reads that gave up and kept the previous set.
    int adoptions     = 0;   ///< Reads that adopted a newer published set.
    int illegal       = 0;   ///< Reads whose result was no published set at all.
    int regressions   = 0;   ///< Reads that went BACK to an older set.
    int reArmFailures = 0;   ///< Deferred updates NOT picked up once the writer stops.
    int quiesceTimeouts = 0; ///< The writer never acknowledged a pause request.
    int capHit        = 0;   ///< The read cap ran out before the run finished.
    int stalls        = 0;   ///< A single read made no progress for the budget.
    long long medianReadNs = 0;  ///< Median COLLIDING read.
    long long maxReadNs    = 0;
    long long quietReadNs  = 0;  ///< Median UNCONTENDED read of the same set.
    float lastTag  = 0.0f;       ///< Tag of the final publication of the run.
    float finalTag = 0.0f;       ///< Tag in force after the writer is joined.
};

/// The integer tag a published set carries, or -1 for a value no publication
/// could have produced. Every tap of a set is tag/kTagTaps and the delay line
/// holds DC 1.0, so a whole set reads back as its own tag exactly, and any
/// mixture of two sets lands strictly between two integers.
static long long tagOf(float out)
{
    const float rounded = std::floor(out + 0.5f);
    if (out != rounded || rounded < 1.0f) return -1;
    return static_cast<long long>(rounded);
}

// Runs the protocol above until it has seen `wantDeferrals` give-ups, or until
// `maxOpportunities` reads that were guaranteed to enter the loop have gone by,
// and reports what the audio-thread reads did.
//
// Nothing here is asserted in wall-clock terms, deliberately. The tempting
// assertion -- "a colliding read is much faster than one publish" -- compares a
// reader-side duration against a writer-side one, and a sanitizer build does
// not instrument the two sides equally: the same code that shows a 1169x margin
// under ASan on one host inverts on another. The property being pinned is not a
// duration, it is that the reader CAME BACK from a read it was guaranteed to
// enter, without the publication that was in flight. An unbounded reader cannot
// do that in any build, at any optimisation level, because its loop has no exit
// other than a validated read. The durations are still measured, and reported
// out of band by the probe in tests/results, where a number is evidence rather
// than a pass/fail condition.
static BoundedReadTrials runBoundedReadTrials(int wantDeferrals,
                                              long long maxOpportunities,
                                              long long readCap = 20000)
{
    using clock = std::chrono::steady_clock;
    auto nsSince = [](clock::time_point t0) {
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   clock::now() - t0).count());
    };

    BoundedReadTrials st;

    FIRFilter<float> fir;
    fir.prepare(kBigTaps, 1);

    // One buffer, stamped in place: the writer allocates nothing between
    // publications, so its counter spends nearly all of its time odd.
    std::vector<float> staged(static_cast<size_t>(kBigTaps), 0.0f);
    auto stamp = [&staged](long long tag) {
        const float v = static_cast<float>(tag) / static_cast<float>(kTagTaps);
        for (int i = 0; i < kTagTaps; ++i) staged[static_cast<size_t>(i)] = v;
    };

    stamp(1);
    fir.setCoefficients(staged);
    for (int i = 0; i < kTagTaps + 8; ++i) (void) fir.processSample(1.0f, 0);
    long long heldTag = 1;

    std::atomic<bool> stop { false };
    std::atomic<bool> pauseRequested { false };
    std::atomic<bool> writerIdle { false };
    std::atomic<long long> publishedTag { 1 };

    // A read that never returns cannot be caught by any cap on the loop that
    // would perform the next one, and that is exactly what a reader with no
    // bound does under a writer that publishes continuously: it waits inside
    // the call. So the loop is watched from outside it. If no read completes
    // for the budget below, the watchdog records a stall and tears the writer
    // down, which lets the stuck read finish so the case can FAIL BY NAME
    // instead of being killed by a test-runner timeout that names nothing.
    //
    // This is a diagnostic budget, not a performance assertion. A bounded read
    // costs about 20 us here; the budget is twenty seconds, six orders of
    // magnitude above it, and the only thing that can consume it is the
    // unbounded wait the attempt limit exists to prevent.
    std::atomic<long long> readTicket { 0 };
    std::atomic<int> stallCount { 0 };
    std::atomic<bool> watchdogStop { false };
    std::thread watchdog([&] {
        long long last = -1;
        int idleTicks = 0;
        while (!watchdogStop.load(std::memory_order_relaxed))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const long long now = readTicket.load(std::memory_order_relaxed);
            if (now != last) { last = now; idleTicks = 0; continue; }
            if (++idleTicks >= 400)          // 20 s with no read completing
            {
                stallCount.fetch_add(1, std::memory_order_relaxed);
                stop.store(true, std::memory_order_relaxed);
                pauseRequested.store(false, std::memory_order_release);
                idleTicks = 0;
            }
        }
    });

    std::thread writer([&] {
        long long tag = 1;
        while (!stop.load(std::memory_order_relaxed))
        {
            if (pauseRequested.load(std::memory_order_acquire))
            {
                writerIdle.store(true, std::memory_order_release);
                while (pauseRequested.load(std::memory_order_acquire)
                       && !stop.load(std::memory_order_relaxed))
                    std::this_thread::yield();
                writerIdle.store(false, std::memory_order_release);
                continue;
            }
            ++tag;
            stamp(tag);
            fir.setCoefficients(staged);
            publishedTag.store(tag, std::memory_order_release);
        }
    });

    // Every loop below is capped, including the outer one, and hitting a cap
    // is a loud failure rather than a quiet exit. That matters in BOTH
    // directions: a writer that never runs must not hang the suite, and a
    // reader whose bound has been lost must not either. An unbounded reader
    // never gives up, so it turns every opportunity into an adoption and would
    // otherwise walk the opportunity budget at the cost of a full stalled read
    // apiece; the caps here are what make that end in seconds, by name,
    // instead of in a test-runner timeout that names nothing.
    constexpr long long kSpinCap = 2000000000LL;
    std::vector<long long> readNs;

    while (st.opportunities < maxOpportunities && st.deferrals < wantDeferrals
           && st.reads < readCap)
    {
        const long long before = publishedTag.load(std::memory_order_acquire);

        const auto t0 = clock::now();
        const float out = fir.processSample(1.0f, 0);   // the audio-thread read
        const long long ns = nsSince(t0);
        ++st.reads;
        readTicket.fetch_add(1, std::memory_order_relaxed);   // the watchdog's pulse
        if (readNs.size() < 4096) readNs.push_back(ns);

        const long long outTag = tagOf(out);
        if (outTag < 0)          { ++st.illegal;     heldTag = -1; continue; }
        if (outTag < heldTag)    { ++st.regressions; heldTag = outTag; continue; }

        // A publication strictly newer than the set in force had already
        // COMPLETED when this read began, so the dirty flag was necessarily
        // set and the read necessarily entered the seqlock loop.
        if (before <= heldTag) { heldTag = outTag; continue; }
        ++st.opportunities;

        if (outTag > heldTag) { ++st.adoptions; heldTag = outTag; continue; }

        // Same set as before: the loop was entered and nothing was adopted.
        ++st.deferrals;

        // The deferred update must have been re-armed rather than dropped.
        // Stopping the writer settles the counter at an even value, and an
        // uncontended read validates on its first attempt by construction, so
        // this check does not depend on how the two threads are scheduled.
        pauseRequested.store(true, std::memory_order_release);
        long long spins = 0;
        while (!writerIdle.load(std::memory_order_acquire) && spins < kSpinCap
               && !stop.load(std::memory_order_relaxed))
        {
            ++spins;
            std::this_thread::yield();
        }
        if (spins >= kSpinCap)
        {
            ++st.quiesceTimeouts;
        }
        else
        {
            const long long newest = publishedTag.load(std::memory_order_acquire);
            const float again = fir.processSample(1.0f, 0);
            ++st.reads;
            if (tagOf(again) != newest) ++st.reArmFailures;
            heldTag = tagOf(again);
        }
        pauseRequested.store(false, std::memory_order_release);
    }

    if (st.reads >= readCap) st.capHit = 1;

    watchdogStop.store(true, std::memory_order_relaxed);
    watchdog.join();
    st.stalls = stallCount.load(std::memory_order_relaxed);

    stop.store(true, std::memory_order_relaxed);
    pauseRequested.store(false, std::memory_order_release);
    writer.join();
    st.publications = publishedTag.load(std::memory_order_acquire) - 1;

    // With the writer joined the counter is even and stable, so this last
    // publication must be in force after one read: nothing was lost.
    const long long finalTagValue = publishedTag.load(std::memory_order_acquire) + 1;
    stamp(finalTagValue);
    fir.setCoefficients(staged);
    st.lastTag  = static_cast<float>(finalTagValue);
    st.finalTag = fir.processSample(1.0f, 0);

    // The denominator for the bound check: the same reader, the same set, the
    // same build, with nobody publishing. Measured after the writer is joined
    // so it is genuinely uncontended.
    {
        std::vector<long long> quiet;
        for (int i = 0; i < 32; ++i)
        {
            stamp(finalTagValue + 1 + i);
            fir.setCoefficients(staged);
            const auto q0 = clock::now();
            (void) fir.processSample(1.0f, 0);
            quiet.push_back(nsSince(q0));
        }
        std::nth_element(quiet.begin(), quiet.begin() + static_cast<long>(quiet.size() / 2),
                         quiet.end());
        st.quietReadNs = quiet[quiet.size() / 2];
    }

    if (!readNs.empty())
    {
        st.maxReadNs = *std::max_element(readNs.begin(), readNs.end());
        std::nth_element(readNs.begin(), readNs.begin() + static_cast<long>(readNs.size() / 2),
                         readNs.end());
        st.medianReadNs = readNs[readNs.size() / 2];
    }
    return st;
}

// The bound itself. The audio thread is made to read while a publish is in
// flight, under a protocol that guarantees the read enters the seqlock loop,
// and it comes back holding a set OLDER than the publication that completed
// before it began. Only a bounded reader can do that: the unbounded loop has no
// exit other than a validated read, so it must wait for the writer and must
// come back with the new set. That is a property of the code and holds at any
// optimisation level and under any sanitizer, which a duration does not.
//
// The two failure modes are told apart rather than conflated. A writer that
// never ran is reported by `opportunities` and `publications`, which are the
// preconditions; a reader that never had to give up is reported by `deferrals`,
// which is the property. A busy host makes the second EASIER to satisfy, so
// scheduling pressure cannot turn this pin red.
DSPARK_TEST(FIR_bounded_seqlock_read_does_not_wait_for_the_publisher)
{
    const BoundedReadTrials st = runBoundedReadTrials(4, 64);

    EXPECT_GT(st.publications, 0);                // the writer really published
    EXPECT_GT(st.opportunities, 0);               // reads really entered the loop
    EXPECT_GT(st.deferrals, 0);                   // it came back without waiting
    EXPECT_EQ(st.illegal, 0);                     // never a set nobody published
    EXPECT_EQ(st.regressions, 0);                 // and never an older one
    EXPECT_EQ(st.reArmFailures, 0);               // a deferred update is not dropped
    EXPECT_EQ(st.quiesceTimeouts, 0);             // the writer answered every pause
    EXPECT_EQ(st.capHit, 0);                      // and the run ended on its own terms
    EXPECT_EQ(st.stalls, 0);                      // no read waited for the publisher

    // What this case does NOT pin, said here rather than left to be assumed:
    // the SIZE of the attempt limit. It pins that the give-up path exists and
    // is reached. A limit raised to a billion still gives up eventually, so
    // the counters above stay green while a single read stalls for hundreds of
    // milliseconds -- and no duration written here can separate that from an
    // ordinary preemption: measured on this host, the worst colliding read is
    // 27 ms on a loaded machine with the limit intact and 448 ms with the
    // limit at a billion, while the MEDIAN is 21 us in both. A threshold
    // between those maxima is a coin toss on a busy runner, which is the
    // failure this protocol was reshaped to remove.
    //
    // The size of the limit is guarded instead by tools/verify_threading_doc.py,
    // which checks every site agrees on the declared value and that the reader
    // is a counted loop over it. Both breaks make it exit 1, and it runs on
    // every push. The durations below are recorded by the probe alongside this
    // suite, where a number is evidence rather than a pass condition.
    EXPECT_TRUE(st.quietReadNs > 0);
    EXPECT_NEAR(st.finalTag, st.lastTag, 1e-4f);  // nothing was lost on the way
}

// No tear under the bound. Phase 1 hammers publications at the audio thread and
// checks every set it adopts is one whole publication: with all taps equal to
// tag/kTagTaps and the delay line at DC 1.0, the output IS the tag, and a
// mixture of two publications is off by at least 1/kTagTaps. Phase 2 then
// proves the give-up path was exercised, so a green phase 1 cannot be green
// merely because the reader never reached the bound.
DSPARK_TEST(FIR_bounded_seqlock_read_never_adopts_a_torn_set)
{
    FIRFilter<float> fir;
    fir.prepare(kTagTaps, 1);
    fir.setCoefficients(tagKernel(1.0f, kTagTaps));
    for (int i = 0; i < kTagTaps + 8; ++i) (void) fir.processSample(1.0f, 0);

    std::vector<std::vector<float>> sets;
    for (int t = 1; t <= 4; ++t) sets.push_back(tagKernel(static_cast<float>(t), kTagTaps));

    std::atomic<bool> stop { false };
    std::atomic<long long> published { 0 };
    std::thread producer([&] {
        for (long long i = 0; !stop.load(std::memory_order_relaxed); ++i)
        {
            fir.setCoefficients(sets[static_cast<size_t>(i & 3)]);
            published.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // The overlap is guaranteed by construction, not asserted after the fact: the
    // reader keeps going until it has done its reads AND the producer has landed
    // its publications, so a build where one side runs far slower than the other
    // still exercises the race instead of failing a liveness floor. The cap is
    // there so a producer that never runs fails this pin loudly.
    constexpr long long kReads = 200000;
    constexpr long long kMinPublications = 2000;
    constexpr long long kReadCap = 40000000LL;

    int torn = 0;
    long long reads = 0;
    while (reads < kReadCap
           && (reads < kReads
               || published.load(std::memory_order_relaxed) < kMinPublications))
    {
        const float out = fir.processSample(1.0f, 0);
        bool legal = false;
        for (int t = 1; t <= 4; ++t)
            if (std::abs(out - static_cast<float>(t)) < 1e-4f) legal = true;
        if (!legal) ++torn;
        ++reads;
    }

    stop.store(true, std::memory_order_relaxed);
    producer.join();

    EXPECT_EQ(torn, 0);
    EXPECT_TRUE(reads < kReadCap);   // the producer really ran
    EXPECT_TRUE(published.load(std::memory_order_relaxed) >= kMinPublications);

    // Not vacuous: the same header, driven so that the reader must give up.
    const BoundedReadTrials st = runBoundedReadTrials(4, 64);
    EXPECT_GT(st.publications, 0);
    EXPECT_GT(st.opportunities, 0);
    EXPECT_GT(st.deferrals, 0);
    EXPECT_EQ(st.illegal, 0);
    EXPECT_EQ(st.regressions, 0);
    EXPECT_EQ(st.quiesceTimeouts, 0);
    EXPECT_EQ(st.capHit, 0);
    EXPECT_EQ(st.stalls, 0);
}

// The flat delay-line layout must keep channels fully independent.
DSPARK_TEST(FIR_multichannel_isolation)
{
    auto taps = FIRDesign<float>::lowPass(48000.0f, 5000.0f, 65);
    FIRFilter<float> fir;
    fir.prepare(65, 2);
    fir.setCoefficients(taps);

    float peakSilent = 0.0f;
    for (int i = 0; i < 1024; ++i)
    {
        const float hot = std::sin(twoPi<float> * 1000.0f * static_cast<float>(i) / 48000.0f);
        (void)fir.processSample(hot, 0);
        peakSilent = std::max(peakSilent, std::abs(fir.processSample(0.0f, 1)));
    }
    EXPECT_EQ(peakSilent, 0.0f);
}

// Coverage for the seqlock coefficient handoff at double precision: the
// staging words are std::atomic<T>, and that must hold for T = double too.
// Checks (1) the lock-free precondition of the design (a non-lock-free atomic
// would take a lock on the audio thread), (2) FIRFilter<double> behaves
// (DC gain == tap sum exactly characterises the adopted kernel), and (3)
// re-prepare() is well-formed: std::vector<std::atomic<T>> is move-ASSIGNED a
// fresh vector, which must steal the buffer without any element copy/move
// (std::atomic is neither copyable nor movable).
DSPARK_TEST(FIR_double_instantiation_lockfree_and_reprepare)
{
    // Hard RT-safety precondition of the atomic-word seqlock design.
    EXPECT_TRUE(std::atomic<float>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<double>::is_always_lock_free);

    // Heap-allocate the <double> object (repo test convention for large ones).
    auto fir = std::make_unique<FIRFilter<double>>();
    fir->prepare(127, 2);
    const auto taps = FIRDesign<double>::lowPass(48000.0, 4000.0, 127);
    fir->setCoefficients(taps);

    // Convolving long DC 1.0 converges to sum(taps) through the atomic-word
    // publish + private-active copy path.
    double dcOut = 0.0;
    for (int i = 0; i < 400; ++i) dcOut = fir->processSample(1.0, 0);
    double tapSum = 0.0;
    for (double t : taps) tapSum += t;
    EXPECT_NEAR(dcOut, tapSum, 1e-12);
    EXPECT_EQ(fir->getLatency(), 63);

    // Re-prepare with a different size: exercises the vector<atomic> move
    // assignment onto an engaged vector and must leave a working filter.
    fir->prepare(31, 1);
    const auto taps2 = FIRDesign<double>::lowPass(48000.0, 2000.0, 31);
    fir->setCoefficients(taps2);
    double dc2 = 0.0;
    for (int i = 0; i < 200; ++i) dc2 = fir->processSample(1.0, 0);
    double sum2 = 0.0;
    for (double t : taps2) sum2 += t;
    EXPECT_NEAR(dc2, sum2, 1e-12);
    EXPECT_EQ(fir->getLatency(), 15);
}


// ============================================================================
// WDF (wave digital filters)
// ============================================================================

DSPARK_TEST(WDF_RC_lowpass_matches_bilinear_analytic)
{
    // A WDF network IS the bilinear discretization of its circuit, so the
    // series RC divider must match the analytic bilinear one-pole exactly.
    const double R = 1000.0, C = 100e-9, fs = 48000.0;

    wdf::Resistor<double> r { R };
    wdf::Capacitor<double> c { C };
    wdf::Series<double, decltype(r), decltype(c)> tree { r, c };
    wdf::IdealVoltageSourceRoot<double, decltype(tree)> root { tree };
    root.prepare(fs);

    const double cb = 1.0 / (2.0 * fs * R * C);
    double x1 = 0, y1 = 0, maxErr = 0;
    for (int n = 0; n < 4800; ++n)
    {
        const double x = std::sin(2.0 * 3.14159265358979 * 1000.0 * n / fs)
                       + 0.5 * std::sin(2.0 * 3.14159265358979 * 7900.0 * n / fs);
        root.setVoltage(x);
        root.process();
        const double yRef = (cb * (x + x1) - (cb - 1.0) * y1) / (1.0 + cb);
        x1 = x; y1 = yRef;
        maxErr = std::max(maxErr, std::abs(c.getVoltage() - yRef));
    }
    EXPECT_LT(maxErr, 1e-12);
}

DSPARK_TEST(WDF_series_RLC_matches_bilinear_analytic)
{
    const double R = 220.0, L = 10e-3, C = 47e-9, fs = 48000.0;

    wdf::Resistor<double> r { R };
    wdf::Inductor<double> l { L };
    wdf::Capacitor<double> c { C };
    wdf::Series<double, decltype(l), decltype(c)> lc { l, c };
    wdf::Series<double, decltype(r), decltype(lc)> tree { r, lc };
    wdf::IdealVoltageSourceRoot<double, decltype(tree)> root { tree };
    root.prepare(fs);

    const double K = 2.0 * fs;
    const double lck = L * C * K * K, rck = R * C * K;
    const double a0 = lck + rck + 1.0;
    const double b0 = 1.0 / a0, b1c = 2.0 / a0, b2 = 1.0 / a0;
    const double a1 = (2.0 - 2.0 * lck) / a0;
    const double a2 = (lck - rck + 1.0) / a0;

    double x1 = 0, x2 = 0, y1 = 0, y2 = 0, maxErr = 0;
    for (int n = 0; n < 9600; ++n)
    {
        const double x = (n == 0) ? 1.0 : 0.0;
        root.setVoltage(x);
        root.process();
        const double yRef = b0 * x + b1c * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x;
        y2 = y1; y1 = yRef;
        maxErr = std::max(maxErr, std::abs(c.getVoltage() - yRef));
    }
    EXPECT_LT(maxErr, 1e-9);
}

DSPARK_TEST(WDF_diode_clipper_DC_sweep_matches_node_equation)
{
    // Static clipper: WDF root must solve (vin - v)/R = 2 Is sinh(v/nVt)
    // to the same answer as an independent 200-step bisection.
    const double R = 2200.0, Is = 2.52e-9, nVt = 1.752 * 0.02585;

    wdf::ResistiveVoltageSource<double> vs { R };
    wdf::DiodePairRoot<double, decltype(vs)> clip { vs, Is, nVt };
    clip.prepare(48000.0);

    double maxErr = 0;
    for (double vin = -10.0; vin <= 10.0; vin += 0.1)
    {
        vs.setVoltage(vin);
        clip.process();

        double lo = std::min(0.0, vin), hi = std::max(0.0, vin);
        for (int it = 0; it < 200; ++it)
        {
            const double mid = 0.5 * (lo + hi);
            const double f = (vin - mid) / R - 2.0 * Is * std::sinh(mid / nVt);
            if (f > 0.0) lo = mid; else hi = mid;
        }
        maxErr = std::max(maxErr, std::abs(clip.getVoltage() - 0.5 * (lo + hi)));
    }
    EXPECT_LT(maxErr, 1e-9);
}

DSPARK_TEST(WDF_RC_diode_clipper_matches_trapezoidal_reference)
{
    // Dynamic clipper (C parallel to the pair): the WDF bilinear capacitor
    // equals the trapezoidal rule, so a high-precision trapezoidal+Newton
    // solve of C dv/dt = (vin-v)/R - 2 Is sinh(v/nVt) - SPICE's .tran
    // method - must null against the WDF to solver precision.
    const double R = 2200.0, C = 10e-9, Is = 2.52e-9, nVt = 1.752 * 0.02585;
    const double fs = 48000.0, h = 1.0 / fs;

    wdf::ResistiveVoltageSource<double> vs { R };
    wdf::Capacitor<double> c { C };
    wdf::Parallel<double, decltype(vs), decltype(c)> tree { vs, c };
    wdf::DiodePairRoot<double, decltype(tree)> clip { tree, Is, nVt };
    clip.prepare(fs);

    auto fOf = [&](double vin, double v) {
        return ((vin - v) / R - 2.0 * Is * std::sinh(v / nVt)) / C;
    };

    double vRef = 0.0, vinPrev = 0.0, err = 0, ref = 0;
    for (int n = 0; n < 9600; ++n)
    {
        const double vin = 6.0 * std::sin(2.0 * 3.14159265358979 * 2000.0 * n / fs);
        vs.setVoltage(vin);
        clip.process();

        const double fPrev = fOf(vinPrev, vRef);
        double v = vRef;
        for (int it = 0; it < 80; ++it)
        {
            const double g  = v - vRef - 0.5 * h * (fPrev + fOf(vin, v));
            const double gp = 1.0 + 0.5 * h * (1.0 / R + (2.0 * Is / nVt) * std::cosh(v / nVt)) / C;
            const double dv = g / gp;
            v -= dv;
            if (std::abs(dv) < 1e-15) break;
        }
        vRef = v;
        vinPrev = vin;

        const double e = clip.getVoltage() - vRef;
        err += e * e;
        ref += vRef * vRef;
    }
    const double db = 10.0 * std::log10((err + 1e-300) / (ref + 1e-300));
    EXPECT_LT(db, -180.0);
}

DSPARK_TEST(WDF_single_diode_half_wave_rectifies)
{
    // Forward drive conducts (output clamps near the knee); reverse blocks
    // (output follows the input through the resistor with ~zero current).
    const double R = 2200.0;
    wdf::ResistiveVoltageSource<double> vs { R };
    wdf::DiodeRoot<double, decltype(vs)> diode { vs };
    diode.prepare(48000.0);

    vs.setVoltage(5.0);
    diode.process();
    const double vFwd = diode.getVoltage();
    EXPECT_GT(vFwd, 0.4);
    EXPECT_LT(vFwd, 0.9);          // silicon knee region

    vs.setVoltage(-5.0);
    diode.process();
    EXPECT_NEAR(diode.getVoltage(), -5.0, 0.01);   // blocked: all of vin drops here
}

DSPARK_TEST(WDF_inverter_flips_subtree_polarity)
{
    // Two identical resistive dividers driven by the same source; wrapping
    // R2 in an Inverter must flip the voltage that element reads while
    // leaving the electrical divider (and the other element) untouched.
    // Inverter had no dedicated coverage before this test.
    const double fs = 48000.0;

    wdf::Resistor<double> r1a { 1000.0 }, r2a { 2000.0 };
    wdf::Series<double, decltype(r1a), decltype(r2a)> treeA { r1a, r2a };
    wdf::IdealVoltageSourceRoot<double, decltype(treeA)> rootA { treeA };
    rootA.prepare(fs);

    wdf::Resistor<double> r1b { 1000.0 }, r2b { 2000.0 };
    wdf::Inverter<double, decltype(r2b)> inv { r2b };
    wdf::Series<double, decltype(r1b), decltype(inv)> treeB { r1b, inv };
    wdf::IdealVoltageSourceRoot<double, decltype(treeB)> rootB { treeB };
    rootB.prepare(fs);

    rootA.setVoltage(3.0);
    rootA.process();
    rootB.setVoltage(3.0);
    rootB.process();

    EXPECT_NEAR(r2a.getVoltage(), 2.0, 1e-12);                 // 3 V * 2k/3k
    EXPECT_NEAR(r2b.getVoltage(), -r2a.getVoltage(), 1e-12);   // inverted view
    EXPECT_NEAR(r1b.getVoltage(), r1a.getVoltage(), 1e-12);    // sibling intact
}


// ============================================================================
// WDF R-type adaptor: FMV tone stack vs Yeh & Smith (DAFx-06) symbolic TF
// ============================================================================

namespace {

void yehCoeffs(double t, double l, double m,
               double& b1, double& b2, double& b3,
               double& a1, double& a2, double& a3,
               double R1 = 250e3, double R3 = 25e3)
{
    const double C1 = 0.25e-9, C2 = 20e-9, C3 = 20e-9;
    const double R2 = 1e6, R4 = 56e3;
    const double R3sq = R3 * R3;

    b1 = t * C1 * R1 + m * C3 * R3 + l * (C1 * R2 + C2 * R2) + (C1 * R3 + C2 * R3);
    b2 = t * (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4)
       - m * m * (C1 * C3 * R3sq + C2 * C3 * R3sq)
       + m * (C1 * C3 * R1 * R3 + C1 * C3 * R3sq + C2 * C3 * R3sq)
       + l * (C1 * C2 * R1 * R2 + C1 * C2 * R2 * R4 + C1 * C3 * R2 * R4)
       + l * m * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
       + (C1 * C2 * R1 * R3 + C1 * C2 * R3 * R4 + C1 * C3 * R3 * R4);
    b3 = l * m * (C1 * C2 * C3 * R1 * R2 * R3 + C1 * C2 * C3 * R2 * R3 * R4)
       - m * m * (C1 * C2 * C3 * R1 * R3sq + C1 * C2 * C3 * R3sq * R4)
       + m * (C1 * C2 * C3 * R1 * R3sq + C1 * C2 * C3 * R3sq * R4)
       + t * C1 * C2 * C3 * R1 * R3 * R4 - t * m * C1 * C2 * C3 * R1 * R3 * R4
       + t * l * C1 * C2 * C3 * R1 * R2 * R4;
    a1 = (C1 * R1 + C1 * R3 + C2 * R3 + C2 * R4 + C3 * R4)
       + m * C3 * R3 + l * (C1 * R2 + C2 * R2);
    a2 = m * (C1 * C3 * R1 * R3 - C2 * C3 * R3 * R4 + C1 * C3 * R3sq + C2 * C3 * R3sq)
       + l * m * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
       - m * m * (C1 * C3 * R3sq + C2 * C3 * R3sq)
       + l * (C1 * C2 * R2 * R4 + C1 * C2 * R1 * R2 + C1 * C3 * R2 * R4 + C2 * C3 * R2 * R4)
       + (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4 + C1 * C2 * R3 * R4
          + C1 * C2 * R1 * R3 + C1 * C3 * R3 * R4 + C2 * C3 * R3 * R4);
    a3 = l * m * (C1 * C2 * C3 * R1 * R2 * R3 + C1 * C2 * C3 * R2 * R3 * R4)
       - m * m * (C1 * C2 * C3 * R1 * R3sq + C1 * C2 * C3 * R3sq * R4)
       + m * (C1 * C2 * C3 * R3sq * R4 + C1 * C2 * C3 * R1 * R3sq
              - C1 * C2 * C3 * R1 * R3 * R4)
       + l * C1 * C2 * C3 * R1 * R2 * R4 + C1 * C2 * C3 * R1 * R3 * R4;
}

DSPARK_TEST(WDF_FMV_state_fork_preserves_capacitors_without_sharing_ports)
{
    for (double sampleRate : {44100.0, 192000.0})
        for (double tone : {0.0, 1.0})
          for (double offset : {0.0, -173.0, 211.0})
        {
            wdf::ToneStackFMV<double> source(38e3, 1e6), twin(38e3, 1e6), fork(38e3, 1e6);
            for (auto* stack : {&source, &twin, &fork})
            {
                stack->prepare(sampleRate);
                stack->setControls(tone, 1.0 - tone, tone);
            }
            for (int i = 0; i < 3072; ++i)
            {
                const double x = 0.3 + std::sin(2.0 * std::numbers::pi * 173.0 * i / sampleRate);
                (void)source.processSample(x);
                (void)twin.processSample(x);
            }
            fork.copyStateFrom(source, offset);
            // Changing the source after the copy must not modify the fork's
            // controls, port connections or capacitor history.
            source.setControls(0.25, 0.75, 0.5);
            double error = 0.0;
            for (int i = 3072; i < 6144; ++i)
            {
                const double x = 0.3 + std::sin(2.0 * std::numbers::pi * 173.0 * i / sampleRate);
                (void)source.processSample(-x);
                error = std::max(error, std::abs(fork.processSample(x + offset) - twin.processSample(x)));
            }
            EXPECT_NEAR(error, 0.0, offset == 0.0 ? 1e-14 : 1e-10);
        }
}

double fmvResidualDb(double t, double l, double m)
{
    const double fs = 48000.0;
    wdf::ToneStackFMV<double> stack(0.01, 1e10);
    stack.prepare(fs);
    stack.setControls(t, std::sqrt(l), m);   // class maps bass^2 -> l

    constexpr double kRmin = 0.5;
    const double tEff = (t * 250e3 + kRmin) / (250e3 + 2.0 * kRmin);
    const double lEff = (l * 1e6 + kRmin) / 1e6;
    const double mEff = (m * 25e3 + kRmin) / (25e3 + 2.0 * kRmin);

    double b1, b2, b3, a1, a2, a3;
    yehCoeffs(tEff, lEff, mEff, b1, b2, b3, a1, a2, a3);
    const double c = 2.0 * fs, c2 = c * c, c3 = c2 * c;
    double B[4] = { -b1 * c - b2 * c2 - b3 * c3, -b1 * c + b2 * c2 + 3.0 * b3 * c3,
                     b1 * c + b2 * c2 - 3.0 * b3 * c3, b1 * c - b2 * c2 + b3 * c3 };
    double A[4] = { -1.0 - a1 * c - a2 * c2 - a3 * c3, -3.0 - a1 * c + a2 * c2 + 3.0 * a3 * c3,
                    -3.0 + a1 * c + a2 * c2 - 3.0 * a3 * c3, -1.0 + a1 * c - a2 * c2 + a3 * c3 };
    for (int i = 3; i >= 0; --i) { B[i] /= A[0]; A[i] /= A[0]; }

    double x[4] = {}, y[4] = {};
    uint32_t rng = 0xBEEF5EEDu;
    double err = 0, pw = 0;
    for (int i = 0; i < 24000; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        const double in = static_cast<double>(rng >> 8) / 8388608.0 - 1.0;
        const double yw = stack.processSample(in);
        x[3] = x[2]; x[2] = x[1]; x[1] = x[0]; x[0] = in;
        const double yr = B[0] * x[0] + B[1] * x[1] + B[2] * x[2] + B[3] * x[3]
                        - A[1] * y[0] - A[2] * y[1] - A[3] * y[2];
        y[2] = y[1]; y[1] = y[0]; y[0] = yr;
        if (i > 2400) { err += (yw - yr) * (yw - yr); pw += yr * yr; }
    }
    return 10.0 * std::log10((err + 1e-300) / (pw + 1e-300));
}

} // namespace

DSPARK_TEST(WDF_RType_FMV_matches_Yeh_transfer_function)
{
    // The 12-port R-type solve of the Bassman stack must null against the
    // bilinear transform of the published symbolic transfer function.
    for (double t : { 0.1, 0.9 })
        for (double l : { 0.1, 0.9 })
            for (double m : { 0.1, 0.9 })
                EXPECT_LT(fmvResidualDb(t, l, m), -80.0);
    EXPECT_LT(fmvResidualDb(0.5, 0.5, 0.5), -80.0);
}

namespace {

std::complex<double> stateSpaceResponse(const wdf::ToneStackFMV<double>::AnalogStateSpace& ss,
                                        std::complex<double> s)
{
    // c (sI - a)^-1 b + d by complex elimination with partial pivoting.
    std::complex<double> m[3][4];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) m[i][j] = (i == j ? s : 0.0) - ss.a[i][j];
        m[i][3] = ss.b[i];
    }
    for (int k = 0; k < 3; ++k)
    {
        int pivot = k;
        for (int i = k + 1; i < 3; ++i)
            if (std::abs(m[i][k]) > std::abs(m[pivot][k])) pivot = i;
        for (int j = 0; j < 4; ++j) std::swap(m[pivot][j], m[k][j]);
        for (int i = k + 1; i < 3; ++i)
        {
            const std::complex<double> f = m[i][k] / m[k][k];
            for (int j = k; j < 4; ++j) m[i][j] -= f * m[k][j];
        }
    }
    std::complex<double> x[3];
    for (int i = 2; i >= 0; --i)
    {
        std::complex<double> v = m[i][3];
        for (int k = i + 1; k < 3; ++k) v -= m[i][k] * x[k];
        x[i] = v / m[i][i];
    }
    return ss.c[0] * x[0] + ss.c[1] * x[1] + ss.c[2] * x[2] + ss.d;
}

} // namespace

DSPARK_TEST(WDF_FMV_analog_state_space_matches_Yeh_transfer_function)
{
    // The continuous-time form must equal the published ANALOG transfer
    // function, not only its bilinear image. The 0.5 ohm end resistances of
    // every pot are part of the network, so the pot totals are R + 1 ohm.
    // With a near-ideal source and load the residual is 8e-9; the wrong pot
    // totals alone leave 4e-5.
    constexpr double kRmin = 0.5;
    for (double t : {0.0, 0.1, 0.5, 0.9, 1.0})
        for (double l : {0.0, 0.1, 0.5, 0.9, 1.0})
            for (double m : {0.0, 0.1, 0.5, 0.9, 1.0})
            {
                wdf::ToneStackFMV<double> stack(1e-6, 1e14);
                stack.prepare(48000.0);
                stack.setControls(t, std::sqrt(l), m);
                const auto ss = stack.analogStateSpace();
                const double tEff = (t * 250e3 + kRmin) / (250e3 + 2.0 * kRmin);
                const double lEff = (l * 1e6 + kRmin) / 1e6;
                const double mEff = (m * 25e3 + kRmin) / (25e3 + 2.0 * kRmin);
                double b1, b2, b3, a1, a2, a3;
                yehCoeffs(tEff, lEff, mEff, b1, b2, b3, a1, a2, a3,
                          250e3 + 2.0 * kRmin, 25e3 + 2.0 * kRmin);
                for (double f : {10.0, 100.0, 1000.0, 10000.0, 40000.0})
                {
                    const std::complex<double> s(0.0, 2.0 * 3.14159265358979323846 * f);
                    const auto expected = (b1 * s + b2 * s * s + b3 * s * s * s)
                                        / (1.0 + a1 * s + a2 * s * s + a3 * s * s * s);
                    EXPECT_LT(std::abs(stateSpaceResponse(ss, s) - expected) / std::abs(expected), 1e-7);
                }
                // a = -C^-1 G with G symmetric: C a is symmetric.
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        EXPECT_NEAR(ss.capacitance[i] * ss.a[i][j], ss.capacitance[j] * ss.a[j][i],
                                    1e-10 * std::max(std::abs(ss.capacitance[i] * ss.a[i][i]),
                                                     std::abs(ss.capacitance[j] * ss.a[j][j])));
            }
}


// ============================================================================
// Analog-matched peaking (Vicanek)
// ============================================================================

namespace {

double biquadMagDb(const BiquadCoeffs& c, double freq, double fs)
{
    const std::complex<double> j(0.0, 1.0);
    const double w = 2.0 * 3.14159265358979 * freq / fs;
    const auto z1 = std::exp(-j * w);
    const auto z2 = z1 * z1;
    const auto h = (c.b0 + c.b1 * z1 + c.b2 * z2) / (1.0 + c.a1 * z1 + c.a2 * z2);
    return 20.0 * std::log10(std::abs(h));
}

// The cookbook bell's analog prototype, H(s) = (s^2 + s*W0*sqrt(G)/Q + W0^2)
// / (s^2 + s*W0/(sqrt(G)*Q) + W0^2): the response makePeak() converges to at
// low frequencies, and the one a de-cramped design must follow up high.
double analogPeakDb(double f0, double Q, double gainDb, double freq)
{
    const double G = std::pow(10.0, gainDb / 20.0);
    const double W0 = 2.0 * 3.14159265358979 * f0;
    const double B = W0 / (Q * std::sqrt(G));
    const double W = 2.0 * 3.14159265358979 * freq;
    const double d2 = (W * W - W0 * W0) * (W * W - W0 * W0);
    return 10.0 * std::log10((d2 + G * G * B * B * W * W) / (d2 + B * B * W * W));
}

double worstAudibleErrorDb(const BiquadCoeffs& c, double fs, double f0, double Q, double g)
{
    double worst = 0.0;
    for (double f = 20.0; f <= 20000.0; f *= 1.02)
        worst = std::max(worst, std::abs(biquadMagDb(c, f, fs) - analogPeakDb(f0, Q, g, f)));
    return worst;
}

} // namespace

DSPARK_TEST(BiquadCoeffs_matched_peak_follows_the_analog_bell)
{
    // Regression: the previous (Orfanidis) design prescribed its Nyquist gain
    // from a prototype with the wrong pole bandwidth (W0/Q instead of
    // W0/(Q*sqrt(G))), and broke down once the centre passed
    // (1 - 1/(2Q)) * Nyquist. Worst 20 Hz - 20 kHz deviation from the bell it
    // claims to match, old -> cookbook -> now:
    //   48 kHz 16 kHz Q1 +6 dB:    2.53 -> 3.16 -> 0.18 dB
    //   48 kHz 10 kHz Q0.7 +12 dB: 2.73 -> 4.40 -> 0.56 dB
    //   44.1 kHz 15 kHz Q0.7 -12:  10.45 -> 7.47 -> 1.14 dB
    struct Case { double fs, f0, Q, g, maxErr; };
    for (const Case k : { Case { 48000.0, 16000.0, 1.0,   6.0, 0.25 },
                          Case { 48000.0, 12000.0, 1.0,  -9.0, 0.90 },
                          Case { 48000.0, 10000.0, 0.7, 12.0, 0.65 },
                          Case { 44100.0, 15000.0, 0.7, -12.0, 1.25 } })
    {
        const auto m = BiquadCoeffs::makePeakMatched(k.fs, k.f0, k.Q, k.g);
        const auto r = BiquadCoeffs::makePeak(k.fs, k.f0, k.Q, k.g);
        EXPECT_NEAR(biquadMagDb(m, k.f0, k.fs), k.g, 0.01);   // exact peak gain
        const double errM = worstAudibleErrorDb(m, k.fs, k.f0, k.Q, k.g);
        EXPECT_LT(errM, k.maxErr);
        EXPECT_LT(errM, 0.3 * worstAudibleErrorDb(r, k.fs, k.f0, k.Q, k.g));
    }
}

DSPARK_TEST(BiquadCoeffs_matched_peak_converges_to_cookbook_at_LF)
{
    const auto m = BiquadCoeffs::makePeakMatched(48000.0, 500.0, 1.0, 6.0);
    const auto r = BiquadCoeffs::makePeak(48000.0, 500.0, 1.0, 6.0);
    for (double f = 50.0; f <= 8000.0; f *= 1.2)
        EXPECT_NEAR(biquadMagDb(m, f, 48000.0), biquadMagDb(r, f, 48000.0), 0.15);

    // Cut symmetry and the identity fallback.
    const auto cut = BiquadCoeffs::makePeakMatched(48000.0, 12000.0, 1.0, -9.0);
    EXPECT_NEAR(biquadMagDb(cut, 12000.0, 48000.0), -9.0, 0.05);
    const auto flat = BiquadCoeffs::makePeakMatched(48000.0, 1000.0, 1.0, 0.0);
    EXPECT_NEAR(biquadMagDb(flat, 1000.0, 48000.0), 0.0, 1e-9);
}

DSPARK_TEST(BiquadCoeffs_matched_peak_is_stable_and_finite_everywhere)
{
    // Includes the corners the closed forms are most exposed to: a tiny Q
    // (real poles; e^(-q w0) * cosh(r w0) overflowed there), centres at the
    // Nyquist clamp, and deep boosts/cuts at 10 Hz / 768 kHz.
    for (double fs : { 44100.0, 48000.0, 768000.0 })
        for (double f0 : { 10.0, 1000.0, 20000.0, fs * 0.499 })
            for (double Q : { 0.001, 0.3, 0.707, 10.0, 100.0 })
                for (double g : { -30.0, -0.5, 0.5, 30.0 })
                {
                    const auto c = BiquadCoeffs::makePeakMatched(fs, f0, Q, g);
                    EXPECT_TRUE(std::isfinite(c.b0) && std::isfinite(c.b1) && std::isfinite(c.b2)
                                && std::isfinite(c.a1) && std::isfinite(c.a2));
                    // Stability triangle: |a2| < 1 and |a1| < 1 + a2.
                    EXPECT_LT(std::abs(c.a2), 1.0);
                    EXPECT_LT(std::abs(c.a1), 1.0 + c.a2 + 1e-12);
                }
}

// ============================================================================
// First-order sections and tilt (analytic cross-checks)
// ============================================================================

DSPARK_TEST(BiquadCoeffs_first_order_sections_match_bilinear_prototype)
{
    const double fs = 48000.0, fc = 1000.0;
    const auto lp = BiquadCoeffs::makeFirstOrderLowPass(fs, fc);
    const auto hp = BiquadCoeffs::makeFirstOrderHighPass(fs, fc);

    // The exact bilinear image of the analog RC/CR prototype: with the
    // prewarped normalised frequency w = tan(pi f / fs) / tan(pi fc / fs),
    // |LP| = 1/sqrt(1+w^2) and |HP| = w/sqrt(1+w^2) at every frequency.
    const double t0 = std::tan(std::numbers::pi * fc / fs);
    for (double f = 20.0; f <= 22000.0; f *= 1.3)
    {
        const double w = std::tan(std::numbers::pi * f / fs) / t0;
        EXPECT_NEAR(biquadMagDb(lp, f, fs),
                    20.0 * std::log10(1.0 / std::sqrt(1.0 + w * w)), 1e-6);
        EXPECT_NEAR(biquadMagDb(hp, f, fs),
                    20.0 * std::log10(w / std::sqrt(1.0 + w * w)), 1e-6);
    }

    // Cutoff sits at -3.01 dB for both, passband limits are exact.
    EXPECT_NEAR(biquadMagDb(lp, fc, fs), -3.0103, 0.001);
    EXPECT_NEAR(biquadMagDb(hp, fc, fs), -3.0103, 0.001);
    EXPECT_NEAR(biquadMagDb(lp, 0.0, fs), 0.0, 1e-9);
    EXPECT_NEAR(biquadMagDb(hp, fs / 2.0, fs), 0.0, 1e-9);
}

DSPARK_TEST(BiquadCoeffs_tilt_pivot_and_endpoint_gains)
{
    const double fs = 48000.0, pivot = 1000.0, g = 8.0;
    const auto bright = BiquadCoeffs::makeTilt(fs, pivot, g);

    // Contract: total swing = g dB, split as -g/2 at DC, +g/2 at Nyquist,
    // and exactly unity at the pivot (by construction of the prototype).
    EXPECT_NEAR(biquadMagDb(bright, 0.0, fs), -g / 2.0, 0.001);
    EXPECT_NEAR(biquadMagDb(bright, pivot, fs), 0.0, 0.001);
    EXPECT_NEAR(biquadMagDb(bright, fs / 2.0, fs), g / 2.0, 0.001);

    // Dark direction mirrors the bright one.
    const auto dark = BiquadCoeffs::makeTilt(fs, pivot, -g);
    EXPECT_NEAR(biquadMagDb(dark, 0.0, fs), g / 2.0, 0.001);
    EXPECT_NEAR(biquadMagDb(dark, fs / 2.0, fs), -g / 2.0, 0.001);

    // getMagnitude() agrees with the independent std::complex evaluation.
    EXPECT_NEAR(20.0 * std::log10(static_cast<double>(bright.getMagnitude(3000.0, fs))),
                biquadMagDb(bright, 3000.0, fs), 1e-9);
}

// ============================================================================
// Biquad concurrency and move semantics
// ============================================================================

DSPARK_TEST(Biquad_concurrent_setCoeffs_is_tear_free)
{
    // A control thread hammers setCoeffs() alternating two recognisable sets
    // while this thread promotes and inspects. The active set must always be
    // one of the two, homogeneous: a mixed set means the seqlock tore.
    BiquadCoeffs setA;
    setA.b0 = 1.0f; setA.b1 = 1.0f; setA.b2 = 1.0f; setA.a1 = 1.0f; setA.a2 = 1.0f;
    BiquadCoeffs setB;
    setB.b0 = 2.0f; setB.b1 = 2.0f; setB.b2 = 2.0f; setB.a1 = 2.0f; setB.a2 = 2.0f;

    Biquad<float> bq;
    std::atomic<bool> stop { false };
    std::atomic<long long> published { 0 };
    std::thread gui([&] {
        for (unsigned i = 0; !stop.load(std::memory_order_relaxed); ++i)
        {
            bq.setCoeffs((i & 1u) ? setB : setA);
            published.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // The reader stops when it has seen enough adoptions, or when the WRITER
    // stops making progress -- never on an iteration count. A count is a proxy
    // for time, and on a host that hands both threads the same core the reader
    // burns through it while the writer is descheduled, which would turn a
    // scheduling decision into a verdict about the seqlock. Yielding when
    // there is nothing to adopt is part of the same reasoning: it gives the
    // writer the core it needs to produce the next publication.
    constexpr long long kIdleCap = 20000000LL;
    int torn = 0, seen = 0;
    long long lastPublished = 0;
    long long idleSpins = 0;
    while (seen < 2000 && idleSpins < kIdleCap)
    {
        if (bq.applyPendingCoeffs())
        {
            const auto& c = bq.getCoeffs();
            const bool homogeneous = c.b0 == c.b1 && c.b1 == c.b2
                                  && c.b2 == c.a1 && c.a1 == c.a2
                                  && (c.b0 == 1.0f || c.b0 == 2.0f);
            if (!homogeneous)
                ++torn;
            ++seen;
            idleSpins = 0;
            continue;
        }

        const long long now = published.load(std::memory_order_relaxed);
        if (now != lastPublished) { lastPublished = now; idleSpins = 0; }
        else                      { ++idleSpins; }
        // Spin for a while before yielding. On a machine with a core to spare
        // the yield would cost far more than the poll it replaces, and on one
        // without, giving the slice back every so often is enough for the
        // writer to keep publishing.
        if ((idleSpins & 0x3ffLL) == 0) std::this_thread::yield();
    }
    stop.store(true, std::memory_order_relaxed);
    gui.join();

    EXPECT_EQ(torn, 0);
    EXPECT_GT(published.load(std::memory_order_relaxed), 0);  // the writer ran
    EXPECT_TRUE(seen >= 2000);  // the race actually ran; not a vacuous pass
}

DSPARK_TEST(Biquad_move_carries_state_and_pending_coeffs)
{
    // Contract relied upon by CrossoverFilter's containers: moving a Biquad
    // during (single-threaded) setup preserves active coefficients, filter
    // state AND a staged-but-not-yet-promoted setCoeffs() update.
    const auto lp = BiquadCoeffs::makeLowPass(48000.0, 2000.0);
    const auto hp = BiquadCoeffs::makeHighPass(48000.0, 500.0);

    Biquad<float> ref, src;
    ref.setCoeffs(lp);
    src.setCoeffs(lp);
    for (int i = 0; i < 128; ++i)
    {
        const float x = std::sin(0.13f * static_cast<float>(i));
        (void)ref.processSample(x, 0);
        (void)src.processSample(x, 0);
    }

    // Stage a pending update on both, then relocate src.
    ref.setCoeffs(hp);
    src.setCoeffs(hp);
    Biquad<float> dst(std::move(src));

    // Identical inputs must produce bit-identical outputs: the move carried
    // the TDF-II state and the pending coefficients land on the next sample.
    for (int i = 0; i < 128; ++i)
    {
        const float x = std::sin(0.31f * static_cast<float>(i));
        EXPECT_EQ(dst.processSample(x, 0), ref.processSample(x, 0));
    }
}

DSPARK_TEST(Biquad_setCoeffsNow_is_immediate_and_bypasses_staging)
{
    // The stream-owner direct set (origin rule, docs/threading.md): the set
    // becomes active with no staging, no adoption and no dirty flag. A filter
    // written through setCoeffsNow() must be bit-identical to one whose
    // coefficients were baked in from the start -- including on the very
    // first sample after the write, where a staged update would still be one
    // gate away if the flag logic were involved at all.
    const auto lp = BiquadCoeffs::makeLowPass(48000.0, 2000.0);
    const auto hp = BiquadCoeffs::makeHighPass(48000.0, 500.0);

    Biquad<float> ref, direct;
    ref.setCoeffs(lp);
    ref.applyPendingCoeffs();
    direct.setCoeffsNow(lp);

    // getCoeffs() reflects the direct write immediately, no promotion call.
    EXPECT_EQ(direct.getCoeffs().b0, lp.b0);
    EXPECT_EQ(direct.getCoeffs().a2, lp.a2);

    for (int i = 0; i < 96; ++i)
    {
        const float x = std::sin(0.17f * static_cast<float>(i));
        // Re-write mid-stream every 16 samples, as a modulating owner would.
        if ((i & 15) == 0)
        {
            const auto& c = (i & 16) ? hp : lp;
            ref.setCoeffs(c);
            ref.applyPendingCoeffs(); // staged path needs explicit promotion
            direct.setCoeffsNow(c);   // direct path must already be there
        }
        EXPECT_EQ(direct.processSample(x, 0), ref.processSample(x, 0));
    }
}

DSPARK_TEST(Biquad_pending_publication_overwrites_a_direct_set_at_the_gate)
{
    // Dual-master semantics documented at setCoeffsNow(): on an instance
    // receiving BOTH a staged publication and a direct set, the armed dirty
    // flag is a LATER write by the adopting thread, so the publication wins
    // at the next gate. Single-threaded here, so the interleaving is exact
    // and the outcome deterministic.
    const auto staged = BiquadCoeffs::makeLowPass(48000.0, 1000.0);
    const auto direct = BiquadCoeffs::makeHighPass(48000.0, 4000.0);

    Biquad<float> bq;
    bq.setCoeffs(staged);    // publication pending, dirty armed
    bq.setCoeffsNow(direct); // direct set active right now...
    EXPECT_EQ(bq.getCoeffs().b0, direct.b0);

    (void)bq.processSample(0.5f, 0); // ...until the gate adopts the pending set
    EXPECT_EQ(bq.getCoeffs().b0, staged.b0);
    EXPECT_EQ(bq.getCoeffs().a1, staged.a1);

    // And with nothing pending, a direct set is not disturbed by the gate.
    bq.setCoeffsNow(direct);
    (void)bq.processSample(0.5f, 0);
    EXPECT_EQ(bq.getCoeffs().b0, direct.b0);
}

// ============================================================================
// EnvelopeFollower
// ============================================================================

DSPARK_TEST(EnvelopeFollower_attack_release_and_rms)
{
    EnvelopeFollower<float> env;
    env.prepare(spec(48000.0, 512, 1));
    env.setAttack(5.0f);
    env.setRelease(100.0f);

    auto buf = makeMonoBuffer(512);
    for (int i = 0; i < 512; ++i) buf.ch(0)[i] = 1.0f;
    int rise = -1, n = 0;
    for (int b = 0; b < 40 && rise < 0; ++b)
    {
        env.processBlock(AudioBufferView<const float>(buf.view()));
        n += 512;
        if (env.getEnvelope(0) > 0.63f) rise = n;
    }
    EXPECT_GT(rise, 0);
    EXPECT_LT(rise / 48000.0, 0.012);

    buf.fillSilence();
    int fall = -1; n = 0;
    for (int b = 0; b < 100 && fall < 0; ++b)
    {
        env.processBlock(AudioBufferView<const float>(buf.view()));
        n += 512;
        if (env.getEnvelope(0) < 0.37f) fall = n;
    }
    EXPECT_GT(fall, 0);
    EXPECT_NEAR(fall / 48000.0, 0.1, 0.03);

    EnvelopeFollower<float> rms;
    rms.prepare(spec(48000.0, 512, 1));
    rms.setMode(EnvelopeFollower<float>::Mode::RMS);
    rms.setAttack(50.0f);
    rms.setRelease(50.0f);
    for (int b = 0; b < 100; ++b)
    {
        for (int i = 0; i < 512; ++i)
            buf.ch(0)[i] = std::sin(2.0f * 3.14159265f * 440.0f
                                    * static_cast<float>(b * 512 + i) / 48000.0f);
        rms.processBlock(AudioBufferView<const float>(buf.view()));
    }
    EXPECT_NEAR(rms.getEnvelope(0), 0.707f, 0.03f);
}

DSPARK_TEST(EnvelopeFollower_invalid_inputs_are_ignored)
{
    auto buf = makeMonoBuffer(512);
    auto fill = [&buf](long& n, float amp) {
        for (int i = 0; i < 512; ++i, ++n)
            buf.ch(0)[i] = amp * std::sin(2.0f * 3.14159265f * 440.0f
                                          * static_cast<float>(n) / 48000.0f);
    };

    // (a) NaN/Inf setters + invalid re-prepares: bit-identical to a twin.
    //     Old: NaN published forever / follower frozen at the peak.
    {
        EnvelopeFollower<float> ef;
        EnvelopeFollower<float> twin;
        ef.prepare(spec(48000.0, 512, 1));
        twin.prepare(spec(48000.0, 512, 1));
        ef.setAttack(5.0f);   twin.setAttack(5.0f);
        ef.setRelease(80.0f); twin.setRelease(80.0f);
        long n = 0;
        for (int b = 0; b < 10; ++b)
        {
            fill(n, 0.5f);
            ef.processBlock(AudioBufferView<const float>(buf.view()));
            twin.processBlock(AudioBufferView<const float>(buf.view()));
        }
        ef.setAttack(std::numeric_limits<float>::quiet_NaN());
        ef.setRelease(std::numeric_limits<float>::infinity());
        AudioSpec bad = spec(48000.0, 512, 1);
        bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
        ef.prepare(bad);
        bad.sampleRate = std::numeric_limits<double>::infinity();
        ef.prepare(bad);
        float maxDiff = 0.0f;
        int nonFinite = 0;
        for (int b = 0; b < 20; ++b)
        {
            fill(n, 0.5f);
            ef.processBlock(AudioBufferView<const float>(buf.view()));
            twin.processBlock(AudioBufferView<const float>(buf.view()));
            if (!std::isfinite(ef.getEnvelope(0))) ++nonFinite;
            maxDiff = std::max(maxDiff, std::abs(ef.getEnvelope(0) - twin.getEnvelope(0)));
        }
        EXPECT_EQ(nonFinite, 0);              // old: NaN published every block
        EXPECT_EQ(maxDiff, 0.0f);             // old: diverges (frozen/reset)
        EXPECT_NEAR(ef.getAttack(), 5.0f, 1e-6f);
        EXPECT_NEAR(ef.getRelease(), 80.0f, 1e-6f);
        ef.setMode(static_cast<EnvelopeFollower<float>::Mode>(99));
        EXPECT_TRUE(ef.getMode() == EnvelopeFollower<float>::Mode::RMS);
    }

    // (b) One non-finite sample must not kill the follower for good.
    {
        EnvelopeFollower<float> ef;
        ef.prepare(spec(48000.0, 512, 1));
        long n = 0;
        for (int b = 0; b < 10; ++b)
        {
            fill(n, 0.5f);
            ef.processBlock(AudioBufferView<const float>(buf.view()));
        }
        fill(n, 0.5f);
        buf.ch(0)[100] = std::numeric_limits<float>::quiet_NaN();
        ef.processBlock(AudioBufferView<const float>(buf.view()));
        EXPECT_TRUE(std::isfinite(ef.getEnvelope(0)));
        for (int b = 0; b < 30; ++b)
        {
            fill(n, 0.5f);
            ef.processBlock(AudioBufferView<const float>(buf.view()));
        }
        EXPECT_NEAR(ef.getEnvelope(0), 0.45f, 0.1f);   // old: NaN forever
    }
}

DSPARK_TEST(EnvelopeFollower_processSample_publishes)
{
    EnvelopeFollower<float> ef;
    ef.prepare(spec(48000.0, 512, 1));
    float last = 0.0f;
    for (int i = 0; i < 4800; ++i)
        last = ef.processSample(0.5f);
    EXPECT_NEAR(last, 0.5f, 0.01f);
    EXPECT_EQ(ef.getEnvelope(0), last);       // old: readout stuck at 0
}

// Pin: the WDF diode-pair Newton solver must converge (finite, clamped, no
// hang) at extreme drive - reflected-wave amplitudes to +/-1e6 V.
DSPARK_TEST(WDF_diode_pair_extreme_drive_converges)
{
    const double R = 1000.0, Is = 2.52e-9, nVt = 0.045;
    wdf::ResistiveVoltageSource<double> vs { R };
    wdf::DiodePairRoot<double, decltype(vs)> clip { vs, Is, nVt };
    clip.prepare(48000.0);
    for (double vin : { -1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6 })
    {
        vs.setVoltage(vin);
        clip.process();
        const double v = clip.getVoltage();
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LT(std::fabs(v), 5.0);   // diode conduction clamps far below drive
    }
}

// ============================================================================
// BiquadCascade: sections in one pass, ramps, bypass
// ============================================================================

namespace {

// A varied cascade: bells, shelves and cuts across the band.
BiquadCoeffs cascadeSection(int index, double rate)
{
    switch (index % 6)
    {
        case 0:  return BiquadCoeffs::makePeak(rate, 120.0 + 37.0 * index, 1.4, 6.0);
        case 1:  return BiquadCoeffs::makeHighShelf(rate, 6000.0, -4.5);
        case 2:  return BiquadCoeffs::makePeak(rate, 2500.0 + 311.0 * index, 4.0, -9.0);
        case 3:  return BiquadCoeffs::makeHighPass(rate, 40.0, 0.9);
        case 4:  return BiquadCoeffs::makeLowShelf(rate, 200.0, 3.0);
        default: return BiquadCoeffs::makeLowPass(rate, 15000.0, 0.6);
    }
}

double cascadeNoise(uint32_t& state)
{
    state = state * 1664525u + 1013904223u;
    return (static_cast<double>(state >> 8) / 8388608.0) - 1.0;
}

} // namespace

DSPARK_TEST(BiquadCascade_matches_a_chain_of_biquads)
{
    constexpr double kRate = 48000.0;
    constexpr int kChannels = 3;     // a pair in one register and a channel on its own
    constexpr int kTotal = 6000;
    for (const int sections : { 1, 2, 3, 4, 5, 7, 8 })
    {
        BiquadCascade<double, 8, 4> cascade;
        cascade.setNumSections(sections);
        std::array<Biquad<double, 4>, 8> chain;
        for (int s = 0; s < sections; ++s)
        {
            const BiquadCoeffs c = cascadeSection(s, kRate);
            cascade.setCoeffs(s, c);
            chain[static_cast<size_t>(s)].setCoeffsNow(c);
        }
        EXPECT_EQ(cascade.getNumSections(), sections);

        AudioBuffer<double> block;
        block.resize(kChannels, 512);
        uint32_t seed = 12345u + static_cast<uint32_t>(sections);
        double worst = 0.0, peak = 0.0;
        int done = 0, turn = 0;
        while (done < kTotal)
        {
            // Lengths that cross the internal chunks every way.
            static constexpr int kLengths[] = { 1, 63, 64, 65, 200, 7, 512 };
            const int length = std::min(kLengths[turn++ % 7], kTotal - done);
            std::array<std::vector<double>, kChannels> expected;
            for (int ch = 0; ch < kChannels; ++ch)
            {
                expected[static_cast<size_t>(ch)].resize(static_cast<size_t>(length));
                for (int i = 0; i < length; ++i)
                {
                    const double x = 0.5 * cascadeNoise(seed);
                    block.getChannel(ch)[i] = x;
                    double y = x;
                    for (int s = 0; s < sections; ++s)
                        y = chain[static_cast<size_t>(s)].processSampleCore(y, ch);
                    expected[static_cast<size_t>(ch)][static_cast<size_t>(i)] = y;
                }
            }
            cascade.processBlock(block.toView().getSubView(0, length));
            for (int ch = 0; ch < kChannels; ++ch)
                for (int i = 0; i < length; ++i)
                {
                    const double want = expected[static_cast<size_t>(ch)][static_cast<size_t>(i)];
                    worst = std::max(worst, std::abs(block.getChannel(ch)[i] - want));
                    peak = std::max(peak, std::abs(want));
                }
            done += length;
        }
        // Two realisations of the same transfer function: they differ by rounding.
        EXPECT_GT(peak, 0.05);
        EXPECT_LT(worst, 1e-9);
    }
}

DSPARK_TEST(BiquadCascade_is_exact_across_block_partitions_and_sample_form)
{
    constexpr double kRate = 48000.0;
    constexpr int kTotal = 3000;
    std::vector<double> input(kTotal);
    uint32_t seed = 99u;
    for (double& v : input) v = 0.5 * cascadeNoise(seed);

    const auto render = [&](int blockLength, bool withRamp) {
        BiquadCascade<double, 8, 2> cascade;
        cascade.setNumSections(7);
        for (int s = 0; s < 7; ++s) cascade.setCoeffs(s, cascadeSection(s, kRate));
        std::vector<double> out(input);
        AudioBuffer<double> block;
        block.resize(1, blockLength);
        int pos = 0;
        while (pos < kTotal)
        {
            const int length = std::min(blockLength, kTotal - pos);
            if (withRamp && pos <= 1000 && pos + length > 1000)
            {
                // The ramp starts at a block edge in every partition used here.
                cascade.rampCoeffs(2, BiquadCoeffs::makePeak(kRate, 900.0, 2.0, 9.0), 700);
            }
            std::copy_n(input.data() + pos, length, block.getChannel(0));
            cascade.processBlock(block.toView().getSubView(0, length));
            std::copy_n(block.getChannel(0), length, out.data() + pos);
            pos += length;
        }
        return out;
    };

    // The stream does not depend on how it is cut (same lane form throughout).
    const auto whole = render(3000, false);
    for (const int blockLength : { 1, 50, 64, 250 })
    {
        const auto cut = render(blockLength, false);
        bool same = true;
        for (int i = 0; i < kTotal; ++i) same = same && cut[static_cast<size_t>(i)] == whole[static_cast<size_t>(i)];
        EXPECT_TRUE(same);
    }
    // Nor does it with a ramp in flight (block edges at sample 1000 in each).
    const auto rampWhole = render(1000, true);
    for (const int blockLength : { 50, 250, 500 })
    {
        const auto cut = render(blockLength, true);
        bool same = true;
        for (int i = 0; i < kTotal; ++i) same = same && cut[static_cast<size_t>(i)] == rampWhole[static_cast<size_t>(i)];
        EXPECT_TRUE(same);
    }
    bool rampHeard = false;
    for (int i = 1000; i < kTotal; ++i)
        rampHeard = rampHeard || std::abs(rampWhole[static_cast<size_t>(i)] - whole[static_cast<size_t>(i)]) > 1e-3;
    EXPECT_TRUE(rampHeard);

    // The per-sample form walks the same ramp and gives the same samples to rounding.
    {
        BiquadCascade<double, 8, 2> blockForm, sampleForm;
        for (auto* each : { &blockForm, &sampleForm })
        {
            each->setNumSections(5);
            for (int s = 0; s < 5; ++s) each->setCoeffs(s, cascadeSection(s, kRate));
        }
        const BiquadCoeffs target = BiquadCoeffs::makePeak(kRate, 3000.0, 3.0, -12.0);
        blockForm.rampCoeffs(4, target, 333);
        sampleForm.rampCoeffs(4, target, 333);
        EXPECT_TRUE(blockForm.isRamping());
        EXPECT_TRUE(blockForm.isRamping(4));
        EXPECT_FALSE(blockForm.isRamping(0));

        AudioBuffer<double> block;
        block.resize(1, 1000);
        std::copy_n(input.data(), 1000, block.getChannel(0));
        blockForm.processBlock(block.toView());
        double worst = 0.0;
        for (int i = 0; i < 1000; ++i)
        {
            sampleForm.advanceRamps(1);
            worst = std::max(worst, std::abs(sampleForm.processSample(input[static_cast<size_t>(i)], 0) - block.getChannel(0)[i]));
        }
        EXPECT_LT(worst, 1e-9);

        // Both ramps landed exactly on the target.
        for (auto* each : { &blockForm, &sampleForm })
        {
            EXPECT_FALSE(each->isRamping());
            const BiquadCoeffs& c = each->getCoeffs(4);
            EXPECT_TRUE(c.b0 == target.b0 && c.b1 == target.b1 && c.b2 == target.b2
                        && c.a1 == target.a1 && c.a2 == target.a2);
        }
    }
}

DSPARK_TEST(BiquadCascade_bypass_leaves_and_returns_without_a_click)
{
    constexpr double kRate = 48000.0;
    constexpr double kPi = 3.14159265358979323846;
    constexpr int kBlock = 4800;
    const BiquadCoeffs bell = BiquadCoeffs::makePeak(kRate, 1000.0, 4.0, 12.0);
    const auto tone = [](int64_t n) { return 0.2 * std::sin(2.0 * kPi * 1000.0 * static_cast<double>(n) / kRate); };

    // The transparent set has the bell's poles and a flat response.
    const BiquadCoeffs flat = bell.bypassed();
    EXPECT_TRUE(flat.a1 == bell.a1 && flat.a2 == bell.a2 && flat.b0 == 1.0 && flat.b1 == bell.a1 && flat.b2 == bell.a2);
    EXPECT_NEAR(flat.getMagnitude(1000.0, kRate), 1.0, 1e-12);
    EXPECT_NEAR(flat.getMagnitude(15000.0, kRate), 1.0, 1e-12);

    BiquadCascade<double, 4, 2> cascade;
    cascade.setNumSections(1);
    cascade.setCoeffs(0, bell);

    AudioBuffer<double> block;
    block.resize(1, kBlock);
    int64_t n = 0;
    const auto run = [&]() {
        for (int i = 0; i < kBlock; ++i) block.getChannel(0)[i] = tone(n + i);
        cascade.processBlock(block.toView());
        n += kBlock;
    };
    const auto largestStep = [&](double previous) {
        double step = 0.0;
        for (int i = 0; i < kBlock; ++i)
        {
            step = std::max(step, std::abs(block.getChannel(0)[i] - previous));
            previous = block.getChannel(0)[i];
        }
        return step;
    };

    run();
    run();                                     // settled: the tone 12 dB up
    double last = block.getChannel(0)[kBlock - 1];
    const double boostedStep = largestStep(block.getChannel(0)[0]);
    EXPECT_NEAR(boostedStep, 0.2 * 3.981 * 2.0 * std::sin(kPi * 1000.0 / kRate), 0.01);

    // Out: the largest step between samples never exceeds what the boosted
    // tone itself was doing, and the output ends on the input.
    cascade.rampToBypass(0, 96);
    run();
    EXPECT_LT(largestStep(last), boostedStep * 1.02);
    run();
    double residue = 0.0;
    for (int i = 0; i < kBlock; ++i)
        residue = std::max(residue, std::abs(block.getChannel(0)[i] - tone(n - kBlock + i)));
    EXPECT_LT(residue, 1e-12);

    // Transparent, it is switched off: the input goes through untouched.
    cascade.setSectionEnabled(0, false);
    EXPECT_FALSE(cascade.isSectionEnabled(0));
    run();
    bool untouched = true;
    for (int i = 0; i < kBlock; ++i) untouched = untouched && block.getChannel(0)[i] == tone(n - kBlock + i);
    EXPECT_TRUE(untouched);
    last = block.getChannel(0)[kBlock - 1];

    // Back in: enabled it starts from its input's history, and the bell comes
    // up along the ramp with no step larger than the boosted tone's.
    cascade.setSectionEnabled(0, true);
    cascade.rampCoeffs(0, bell, 96);
    run();
    EXPECT_LT(largestStep(last), boostedStep * 1.02);
    run();
    double peak = 0.0;
    for (int i = 0; i < kBlock; ++i) peak = std::max(peak, std::abs(block.getChannel(0)[i]));
    EXPECT_NEAR(peak, 0.2 * 3.981, 0.005);

}

DSPARK_TEST(BiquadCascade_sections_enter_leave_and_recover)
{
    constexpr double kRate = 48000.0;
    BiquadCascade<float, 6, 2> cascade;
    cascade.setNumSections(4);
    for (int s = 0; s < 6; ++s) cascade.setCoeffs(s, cascadeSection(s, kRate));

    // Out-of-range arguments are ignored or clamped.
    cascade.setCoeffs(-1, BiquadCoeffs {});
    cascade.setCoeffs(6, BiquadCoeffs {});
    cascade.rampCoeffs(99, BiquadCoeffs {}, 10);
    cascade.setSectionEnabled(17, false);
    EXPECT_FALSE(cascade.isSectionEnabled(17));
    cascade.setNumSections(99);
    EXPECT_EQ(cascade.getNumSections(), 6);
    cascade.setNumSections(-3);
    EXPECT_EQ(cascade.getNumSections(), 0);

    // No sections: the block is left as it is.
    auto block = makeStereoBuffer(256);
    block.fillNoise(0.5f);
    std::vector<float> before(block.ch(0), block.ch(0) + 256);
    cascade.processBlock(block.view());
    bool same = true;
    for (int i = 0; i < 256; ++i) same = same && block.ch(0)[i] == before[static_cast<size_t>(i)];
    EXPECT_TRUE(same);

    // A transparent section that enters the path changes nothing audible.
    cascade.setNumSections(1);
    cascade.setCoeffs(0, cascadeSection(0, kRate).bypassed());
    block.fillNoise(0.5f, 77u);
    before.assign(block.ch(0), block.ch(0) + 256);
    cascade.processBlock(block.view());
    float worst = 0.0f;
    for (int i = 0; i < 256; ++i) worst = std::max(worst, std::abs(block.ch(0)[i] - before[static_cast<size_t>(i)]));
    EXPECT_LT(worst, 1e-6f);

    // A sample that is not finite does not stay in the memory past its block.
    cascade.setNumSections(4);
    for (int s = 0; s < 4; ++s) cascade.setCoeffs(s, cascadeSection(s, kRate));
    block.fillNoise(0.5f, 5u);
    block.ch(0)[10] = std::numeric_limits<float>::quiet_NaN();
    block.ch(1)[20] = std::numeric_limits<float>::infinity();
    cascade.processBlock(block.view());
    block.fillNoise(0.5f, 6u);
    cascade.processBlock(block.view());
    EXPECT_FALSE(hasNaN(block.ch(0), 256));
    EXPECT_FALSE(hasNaN(block.ch(1), 256));
    EXPECT_GT(measureRMS(block.ch(0), 256), 0.01f);

    // reset() clears the memory: silence in, exact silence out.
    cascade.reset();
    block.fillSilence();
    cascade.processBlock(block.view());
    bool silent = true;
    for (int i = 0; i < 256; ++i) silent = silent && block.ch(0)[i] == 0.0f && block.ch(1)[i] == 0.0f;
    EXPECT_TRUE(silent);
}

// ============================================================================
// NormalizedLadder: rotations and taps
// ============================================================================

DSPARK_TEST(NormalizedLadder_realises_every_biquad)
{
    constexpr double kRate = 48000.0;
    const BiquadCoeffs sets[] = {
        BiquadCoeffs::makeLowPass(kRate, 1000.0, 0.707),
        BiquadCoeffs::makeLowPass(kRate, 80.0, 12.0),
        BiquadCoeffs::makeHighPass(kRate, 40.0, 0.9),
        BiquadCoeffs::makeBandPass(kRate, 3000.0, 5.0),
        BiquadCoeffs::makeNotch(kRate, 1000.0, 8.0),
        BiquadCoeffs::makeAllPass(kRate, 500.0, 0.7),
        BiquadCoeffs::makePeak(kRate, 1000.0, 4.0, 12.0),
        BiquadCoeffs::makePeak(kRate, 9000.0, 0.5, -18.0),
        BiquadCoeffs::makeLowShelf(kRate, 200.0, 6.0),
        BiquadCoeffs::makeHighShelf(kRate, 8000.0, -9.0),
        BiquadCoeffs {},   // the identity
    };
    for (const BiquadCoeffs& c : sets)
    {
        // The rotations are rotations and the round trip gives the biquad back.
        const auto ladder = NormalizedLadder<double, 1>::fromBiquad(c);
        EXPECT_NEAR(ladder.c1 * ladder.c1 + ladder.s1 * ladder.s1, 1.0, 1e-14);
        EXPECT_NEAR(ladder.c2 * ladder.c2 + ladder.s2 * ladder.s2, 1.0, 1e-14);
        const BiquadCoeffs back = NormalizedLadder<double, 1>::toBiquad(ladder);
        EXPECT_NEAR(back.b0, c.b0, 1e-12);
        EXPECT_NEAR(back.b1, c.b1, 1e-12);
        EXPECT_NEAR(back.b2, c.b2, 1e-12);
        EXPECT_NEAR(back.a1, c.a1, 1e-12);
        EXPECT_NEAR(back.a2, c.a2, 1e-12);

        // The same samples as the transposed form, to rounding.
        NormalizedLadder<double, 1> filter;
        filter.setCoeffs(c);
        Biquad<double, 1> reference;
        reference.setCoeffsNow(c);
        uint32_t seed = 4242u;
        double worst = 0.0, peak = 0.0;
        for (int i = 0; i < 6000; ++i)
        {
            const double x = 0.5 * cascadeNoise(seed);
            const double want = reference.processSampleCore(x, 0);
            worst = std::max(worst, std::abs(filter.processSample(x, 0) - want));
            peak = std::max(peak, std::abs(want));
        }
        EXPECT_GT(peak, 0.01);
        EXPECT_LT(worst, 1e-9);
    }

    // A set with poles outside the circle is brought inside: the section stays bounded.
    BiquadCoeffs wild;
    wild.b0 = 1.0; wild.a1 = -2.5; wild.a2 = 1.7;
    NormalizedLadder<double, 1> safe;
    safe.setCoeffs(wild);
    double largest = 0.0;
    uint32_t seed = 1u;
    for (int i = 0; i < 20000; ++i)
        largest = std::max(largest, std::abs(safe.processSample(i < 100 ? cascadeNoise(seed) : 0.0, 0)));
    EXPECT_TRUE(std::isfinite(largest));
    EXPECT_LT(safe.getStateEnergy(0), 1e6);
}

DSPARK_TEST(NormalizedLadder_never_adds_energy_to_its_memory)
{
    constexpr double kRate = 48000.0;
    constexpr double kPi = 3.14159265358979323846;
    NormalizedLadder<double, 1> filter;
    filter.setCoeffs(BiquadCoeffs::makeLowPass(kRate, 300.0, 20.0));
    uint32_t seed = 31u;
    for (int i = 0; i < 4000; ++i) (void)filter.processSample(0.5 * cascadeNoise(seed), 0);
    double energy = filter.getStateEnergy(0);
    EXPECT_GT(energy, 1e-3);

    // No input, and a different stable set of poles every sample: the energy
    // in the memory never rises, whatever the jumps.
    bool rose = false;
    for (int i = 0; i < 20000; ++i)
    {
        const double radius = 0.5 + 0.4999 * (0.5 * cascadeNoise(seed) + 0.5);
        const double angle = kPi * (0.5 * cascadeNoise(seed) + 0.5);
        BiquadCoeffs c;
        c.b0 = 1.0;
        c.a1 = -2.0 * radius * std::cos(angle);
        c.a2 = radius * radius;
        filter.setCoeffs(c);
        (void)filter.processSample(0.0, 0);
        const double now = filter.getStateEnergy(0);
        rose = rose || now > energy * (1.0 + 1e-12) + 1e-300;
        energy = now;
    }
    EXPECT_FALSE(rose);

    // Nor along a ramp between two distant settings.
    filter.setCoeffs(BiquadCoeffs::makeLowPass(kRate, 150.0, 30.0));
    for (int i = 0; i < 4000; ++i) (void)filter.processSample(0.5 * cascadeNoise(seed), 0);
    energy = filter.getStateEnergy(0);
    filter.rampCoeffs(BiquadCoeffs::makeLowPass(kRate, 12000.0, 0.5), 300);
    rose = false;
    for (int i = 0; i < 300; ++i)
    {
        filter.advanceRamp(1);
        const auto& l = filter.getLadder();
        // Linear interpolation keeps each rotation on or inside the circle.
        rose = rose || l.c1 * l.c1 + l.s1 * l.s1 > 1.0 + 1e-12 || l.c2 * l.c2 + l.s2 * l.s2 > 1.0 + 1e-12;
        (void)filter.processSample(0.0, 0);
        const double now = filter.getStateEnergy(0);
        rose = rose || now > energy * (1.0 + 1e-12) + 1e-300;
        energy = now;
    }
    EXPECT_FALSE(rose);
    EXPECT_FALSE(filter.isRamping());
}

DSPARK_TEST(NormalizedLadder_ramp_lands_exactly_and_block_matches_sample_form)
{
    constexpr double kRate = 48000.0;
    const BiquadCoeffs from = BiquadCoeffs::makePeak(kRate, 700.0, 2.0, 6.0);
    const BiquadCoeffs to = BiquadCoeffs::makePeak(kRate, 2500.0, 3.0, -9.0);
    NormalizedLadder<float, 2> blockForm;
    NormalizedLadder<float, 2> sampleForm;
    blockForm.setCoeffs(from);
    sampleForm.setCoeffs(from);
    blockForm.rampCoeffs(to, 700);
    sampleForm.rampCoeffs(to, 700);
    EXPECT_TRUE(blockForm.isRamping());

    auto block = makeStereoBuffer(512);
    double worst = 0.0;
    for (int pass = 0; pass < 3; ++pass)
    {
        block.fillNoise(0.5f, 100u + static_cast<uint32_t>(pass));
        std::vector<float> left(block.ch(0), block.ch(0) + 512), right(block.ch(1), block.ch(1) + 512);
        blockForm.processBlock(block.view());
        for (int i = 0; i < 512; ++i)
        {
            sampleForm.advanceRamp(1);
            const double l = sampleForm.processSample(static_cast<double>(left[static_cast<size_t>(i)]), 0);
            const double r = sampleForm.processSample(static_cast<double>(right[static_cast<size_t>(i)]), 1);
            worst = std::max(worst, std::abs(l - static_cast<double>(block.ch(0)[i])));
            worst = std::max(worst, std::abs(r - static_cast<double>(block.ch(1)[i])));
        }
    }
    EXPECT_LT(worst, 1e-6);   // the block form writes float samples

    // Both landed exactly on the target's rotations and taps.
    const auto target = NormalizedLadder<float, 2>::fromBiquad(to);
    for (const auto* each : { &blockForm, &sampleForm })
    {
        EXPECT_FALSE(each->isRamping());
        const auto& l = each->getLadder();
        EXPECT_TRUE(l.c1 == target.c1 && l.s1 == target.s1 && l.c2 == target.c2 && l.s2 == target.s2
                    && l.v0 == target.v0 && l.v1 == target.v1 && l.v2 == target.v2);
    }

    // The transparent set of a bell: its poles with a flat response.
    NormalizedLadder<double, 1> bypassed;
    bypassed.setCoeffs(from);
    uint32_t seed = 8u;
    for (int i = 0; i < 2000; ++i) (void)bypassed.processSample(0.5 * cascadeNoise(seed), 0);
    bypassed.setCoeffs(from.bypassed());
    const auto& flat = bypassed.getLadder();
    EXPECT_NEAR(flat.v0, flat.c1 * flat.c2, 1e-14);
    EXPECT_NEAR(flat.v1, flat.s1 * flat.c2, 1e-14);
    EXPECT_NEAR(flat.v2, flat.s2, 1e-14);
    for (int i = 0; i < 4000; ++i) (void)bypassed.processSample(0.5 * cascadeNoise(seed), 0);
    double residue = 0.0;
    for (int i = 0; i < 1000; ++i)
    {
        const double x = 0.5 * cascadeNoise(seed);
        residue = std::max(residue, std::abs(bypassed.processSample(x, 0) - x));
    }
    EXPECT_LT(residue, 1e-12);

    // A non-finite sample does not stay in the memory past its block.
    block.fillNoise(0.5f, 3u);
    block.ch(0)[7] = std::numeric_limits<float>::quiet_NaN();
    blockForm.processBlock(block.view());
    block.fillNoise(0.5f, 4u);
    blockForm.processBlock(block.view());
    EXPECT_FALSE(hasNaN(block.ch(0), 512));
    EXPECT_FALSE(hasNaN(block.ch(1), 512));
}

DSPARK_TEST(BiquadCascade_low_corners_at_high_rates_stay_clean)
{
    // 768 kHz: a 20 Hz high-pass and a 30 Hz bell of a tenth of a decibel,
    // corners four orders of magnitude under the rate.
    constexpr double kRate = 768000.0;
    constexpr double kPi = 3.14159265358979323846;
    const BiquadCoeffs highPass = BiquadCoeffs::makeHighPass(kRate, 20.0, 0.7071067811865476);
    const BiquadCoeffs bell = BiquadCoeffs::makePeak(kRate, 30.0, 1.0, 0.1);
    BiquadCascade<double, 2, 1> cascade;
    cascade.setNumSections(2);
    cascade.setCoeffs(0, highPass);
    cascade.setCoeffs(1, bell);
    Biquad<double, 1> first, second;
    first.setCoeffsNow(highPass);
    second.setCoeffsNow(bell);

    constexpr int kTotal = 1152000;   // a second and a half
    double worst = 0.0, mean = 0.0;
    for (int n = 0; n < kTotal; ++n)
    {
        // A constant level under a 1 kHz tone: the level must not come through.
        const double x = 0.25 + 0.25 * std::sin(2.0 * kPi * 1000.0 * static_cast<double>(n) / kRate);
        const double y = cascade.processSample(x, 0);
        const double want = second.processSampleCore(first.processSampleCore(x, 0), 0);
        if (n > kTotal / 2) worst = std::max(worst, std::abs(y - want));
        if (n >= kTotal - 768) mean += y / 768.0;   // one whole period of the tone
    }
    EXPECT_LT(worst, 1e-8);
    EXPECT_LT(std::abs(mean), 1e-9);
}
