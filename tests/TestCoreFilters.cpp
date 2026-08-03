// DSPark Tests - Core Filters
// Biquad, StateVariableFilter, LadderFilter, FIRFilter

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/Biquad.h"
#include "../Core/StateVariableFilter.h"
#include "../Core/LadderFilter.h"
#include "../Core/FIRFilter.h"
#include "../Core/WDF.h"
#include "../Analysis/EnvelopeFollower.h"

#include <algorithm>
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
    // The block path must be bit-identical to the per-sample path (same
    // snapshot, same coefficients, same FP order), across modes and drive.
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
                if (tb.ch(0)[i] != sampleF.processSample(in[static_cast<size_t>(i)], 0)) ++mismatches;
        }
        EXPECT_EQ(mismatches, 0);
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

    int torn = 0;
    for (int i = 0; i < 400000; ++i)
    {
        const float out = fir.processSample(1.0f, 0);
        const bool isA = std::abs(out - 1.0f) < 1e-4f;
        const bool isB = std::abs(out - 3.0f) < 1e-4f;
        if (!isA && !isB) ++torn;
    }

    stop.store(true, std::memory_order_relaxed);
    producer.join();

    EXPECT_EQ(torn, 0);
    EXPECT_GT(published.load(std::memory_order_relaxed), 2000);
}

// ============================================================================
// Bounded audio-thread seqlock reads (the real-time bound, not race freedom)
// ============================================================================
//
// The audio thread's coefficient pull enters its seqlock loop only when the
// dirty flag is set, and setCoefficients() raises that flag AFTER it leaves its
// critical section. A reader can therefore only meet a publish in progress if
// an EARLIER publish is still unconsumed:
//
//   1. publish P1 completes           -> dirty = true, nothing adopted yet
//   2. signal, then start publish P2  -> the sequence counter goes odd
//   3. the audio thread reads         -> dirty was true, so it MUST enter the
//                                        loop, and it meets P2 in flight
//
// Step 3's read is guaranteed to enter the loop, so an audio thread that comes
// out of it still holding a set OLDER than P1 can only have got there by giving
// up: the attempt bound ran out, nothing was adopted, the flag was re-armed.
// That is what makes "the give-up path ran" a proof below and not a guess.
//
// P2 is published with a much larger tap count than P1 on purpose. The staged
// tap COUNT is stored at the end of the critical section, so while P2 is in
// flight a reader still sees P1's small count and retries cheaply instead of
// hiding the wait inside one huge copy -- which is what lets the timing
// assertion see the wait at all.

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
    int trialsRun     = 0;   ///< Trials actually executed (the loop stops early).
    int deferrals     = 0;   ///< Reads that gave up and kept the previous set.
    int adoptions     = 0;   ///< Reads that adopted one of the two new sets.
    int illegal       = 0;   ///< Reads whose result was no published set at all.
    int reArmFailures = 0;   ///< Deferred updates NOT picked up by the next read.
    int signalTimeouts = 0;  ///< Trials where the writer never signalled in time.
    int resetMismatches = 0; ///< Trials whose uncontended reset read the wrong set.
    long long medianReadNs = 0;  ///< Reported, never asserted on -- see below.
    long long maxReadNs    = 0;
    long long publishNs    = 0;  ///< One uncontended publish of P2's size.
    float lastTag  = 0.0f;       ///< Tag of the final publication of the run.
    float finalTag = 0.0f;       ///< Tag in force after the writers are all joined.
};

// Runs the protocol above until it has seen `wantDeferrals` give-ups, or until
// `maxTrials` trials have run, and reports what the audio-thread read did.
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
static BoundedReadTrials runBoundedReadTrials(int wantDeferrals, int maxTrials)
{
    using clock = std::chrono::steady_clock;
    auto nsSince = [](clock::time_point t0) {
        return static_cast<long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   clock::now() - t0).count());
    };

    BoundedReadTrials st;

    // Calibration, on a filter of its own so it disturbs nothing: how long one
    // publish of P2's size holds the sequence counter odd on THIS machine. The
    // pin compares against that instead of a hard-coded microsecond count.
    {
        FIRFilter<float> cal;
        cal.prepare(kBigTaps, 1);
        const std::vector<float> big = tagKernel(1.0f, kBigTaps);
        cal.setCoefficients(big);                       // fault the pages in
        st.publishNs = std::numeric_limits<long long>::max();
        for (int i = 0; i < 5; ++i)
        {
            const auto t0 = clock::now();
            cal.setCoefficients(big);
            st.publishNs = std::min(st.publishNs, nsSince(t0));
        }
    }

    FIRFilter<float> fir;
    fir.prepare(kBigTaps, 1);

    float live = 1.0f;
    fir.setCoefficients(tagKernel(live, kTagTaps));
    for (int i = 0; i < kTagTaps + 8; ++i) (void) fir.processSample(1.0f, 0);

    std::vector<long long> readNs;
    readNs.reserve(static_cast<size_t>(maxTrials));

    for (int trial = 0; trial < maxTrials && st.deferrals < wantDeferrals; ++trial)
    {
        ++st.trialsRun;
        const float t1 = 2.0f + 2.0f * static_cast<float>(trial);
        const float t2 = t1 + 1.0f;
        const std::vector<float> k1 = tagKernel(t1, kTagTaps);
        const std::vector<float> k2 = tagKernel(t2, kBigTaps);

        std::atomic<bool> inFlight { false };
        std::thread writer([&] {
            fir.setCoefficients(k1);                    // P1: completes, raises dirty
            inFlight.store(true, std::memory_order_release);
            fir.setCoefficients(k2);                    // P2: the long critical section
        });

        // Hard-capped rather than open: a writer thread that never runs must
        // fail this pin loudly, not hang the suite.
        long long spins = 0;
        while (!inFlight.load(std::memory_order_acquire) && spins < 2000000000LL)
        {
            ++spins;
            std::this_thread::yield();
        }
        if (spins >= 2000000000LL) ++st.signalTimeouts;

        const auto t0 = clock::now();
        const float out = fir.processSample(1.0f, 0);   // the audio-thread read
        readNs.push_back(nsSince(t0));

        writer.join();

        if (out == live)
        {
            ++st.deferrals;
            // A deferred publication must land on the very next read: the
            // give-up re-armed the dirty flag instead of dropping the update.
            const float next = fir.processSample(1.0f, 0);
            if (next == t2) ; else ++st.reArmFailures;
            live = next;
        }
        else if (out == t1 || out == t2)
        {
            ++st.adoptions;
            live = out;
        }
        else
        {
            ++st.illegal;
            live = out;
        }

        // Put a small set back in force so the next trial reads cheaply again.
        // With no writer running the counter is even and stable, so this single
        // uncontended read must adopt: it is the control for every trial.
        fir.setCoefficients(tagKernel(live, kTagTaps));
        if (fir.processSample(1.0f, 0) != live) ++st.resetMismatches;
    }

    st.lastTag = 2.0f * static_cast<float>(maxTrials) + 7.0f;
    fir.setCoefficients(tagKernel(st.lastTag, kTagTaps));
    st.finalTag = fir.processSample(1.0f, 0);

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
// before the writer signalled. Only a bounded reader can do that: the unbounded
// loop has no exit other than a validated read, so it must wait for the writer
// and must come back with the new set. That is a property of the code and holds
// at any optimisation level and under any sanitizer, which a duration does not.
DSPARK_TEST(FIR_bounded_seqlock_read_does_not_wait_for_the_publisher)
{
    const BoundedReadTrials st = runBoundedReadTrials(4, 96);

    EXPECT_GT(st.deferrals, 0);                   // it came back without waiting
    EXPECT_EQ(st.illegal, 0);                     // never a set nobody published
    EXPECT_EQ(st.reArmFailures, 0);               // the update landed on the next read
    EXPECT_EQ(st.signalTimeouts, 0);              // every trial really raced
    EXPECT_EQ(st.resetMismatches, 0);             // and an uncontended read always adopts
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
    const BoundedReadTrials st = runBoundedReadTrials(4, 96);
    EXPECT_GT(st.deferrals, 0);
    EXPECT_EQ(st.illegal, 0);
    EXPECT_EQ(st.signalTimeouts, 0);
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
               double& a1, double& a2, double& a3)
{
    const double C1 = 0.25e-9, C2 = 20e-9, C3 = 20e-9;
    const double R1 = 250e3, R2 = 1e6, R3 = 25e3, R4 = 56e3;
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


// ============================================================================
// Orfanidis matched peaking (prescribed Nyquist gain)
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

double analogPeakDb(double f0, double Q, double gainDb, double freq)
{
    const double G = std::pow(10.0, gainDb / 20.0);
    const double W0 = 2.0 * 3.14159265358979 * f0, DW = W0 / Q;
    const double W = 2.0 * 3.14159265358979 * freq;
    const double d2 = (W * W - W0 * W0) * (W * W - W0 * W0);
    return 10.0 * std::log10((d2 + G * G * DW * DW * W * W) / (d2 + DW * DW * W * W));
}

} // namespace

DSPARK_TEST(BiquadCoeffs_matched_peak_prescribes_f0_and_Nyquist)
{
    const double fs = 48000.0, f0 = 16000.0, Q = 1.0, g = 6.0;
    const auto m = BiquadCoeffs::makePeakMatched(fs, f0, Q, g);
    EXPECT_NEAR(biquadMagDb(m, f0, fs), g, 0.01);
    EXPECT_NEAR(biquadMagDb(m, fs / 2.0, fs), analogPeakDb(f0, Q, g, fs / 2.0), 0.01);

    // And it de-cramps: residual vs analog under half the cookbook's.
    const auto r = BiquadCoeffs::makePeak(fs, f0, Q, g);
    double errM = 0, errR = 0;
    for (double f = 1000.0; f <= 23000.0; f *= 1.05)
    {
        const double a = analogPeakDb(f0, Q, g, f);
        errM = std::max(errM, std::abs(biquadMagDb(m, f, fs) - a));
        errR = std::max(errR, std::abs(biquadMagDb(r, f, fs) - a));
    }
    EXPECT_LT(errM, errR * 0.45);
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
    std::thread gui([&] {
        for (unsigned i = 0; !stop.load(std::memory_order_relaxed); ++i)
            bq.setCoeffs((i & 1u) ? setB : setA);
    });

    int torn = 0, seen = 0;
    long long iters = 0;
    while (seen < 2000 && iters < 50000000LL)
    {
        ++iters;
        if (bq.applyPendingCoeffs())
        {
            const auto& c = bq.getCoeffs();
            const bool homogeneous = c.b0 == c.b1 && c.b1 == c.b2
                                  && c.b2 == c.a1 && c.a1 == c.a2
                                  && (c.b0 == 1.0f || c.b0 == 2.0f);
            if (!homogeneous)
                ++torn;
            ++seen;
        }
    }
    stop.store(true, std::memory_order_relaxed);
    gui.join();

    EXPECT_EQ(torn, 0);
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
