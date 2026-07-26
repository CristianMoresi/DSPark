// DSPark Tests - Effects Stereo
// MidSide, StereoWidth, Crossfade, Panner

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Effects/MidSide.h"
#include "../Effects/StereoWidth.h"
#include "../Effects/Crossfade.h"
#include "../Effects/Panner.h"

#include <algorithm>
#include <limits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// MidSide
// ============================================================================

DSPARK_TEST(MidSide_encode_decode_roundtrip)
{
    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);
    // Make L and R different
    generateSine(tb.ch(1), 256, 880.0f, 44100.0f);

    // Save originals
    std::vector<float> origL(tb.ch(0), tb.ch(0) + 256);
    std::vector<float> origR(tb.ch(1), tb.ch(1) + 256);

    // Encode L/R -> M/S
    MidSide<float>::encode(tb.view());
    // Decode M/S -> L/R
    MidSide<float>::decode(tb.view());

    // Should match original
    for (int i = 0; i < 256; ++i)
    {
        EXPECT_NEAR(tb.ch(0)[i], origL[i], 1e-5f);
        EXPECT_NEAR(tb.ch(1)[i], origR[i], 1e-5f);
    }
}

DSPARK_TEST(MidSide_mono_signal_all_mid)
{
    auto tb = makeStereoBuffer(128);
    // Same signal on both channels = mono
    generateDC(tb.ch(0), 128, 0.8f);
    generateDC(tb.ch(1), 128, 0.8f);

    MidSide<float>::encode(tb.view());

    // Mid should contain the signal, Side should be zero
    float midPeak = measurePeak(tb.ch(0), 128);
    float sidePeak = measurePeak(tb.ch(1), 128);

    EXPECT_GT(midPeak, 0.5f);
    EXPECT_NEAR(sidePeak, 0.0f, 1e-5f);
}

DSPARK_TEST(MidSide_silence)
{
    auto tb = makeStereoBuffer(64);
    tb.fillSilence();
    MidSide<float>::encode(tb.view());
    EXPECT_SILENT(tb.ch(0), 64, 1e-10f);
    EXPECT_SILENT(tb.ch(1), 64, 1e-10f);
}

DSPARK_TEST(MidSide_exact_values_and_sample_api_bit_matches_block)
{
    // Known asymmetric pair: L=0.8, R=-0.4 -> M=0.2, S=0.6 (exact in float).
    float l = 0.8f, r = -0.4f;
    MidSide<float>::encodeSample(l, r);
    EXPECT_NEAR(l, 0.2f, 1e-7f);
    EXPECT_NEAR(r, 0.6f, 1e-7f);
    MidSide<float>::decodeSample(l, r);
    EXPECT_NEAR(l, 0.8f, 1e-7f);
    EXPECT_NEAR(r, -0.4f, 1e-7f);

    // The per-sample API must be bit-identical to the block API.
    auto tb = makeStereoBuffer(256);
    uint32_t rng = 21u;
    std::vector<float> ls(256), rs(256);
    for (int i = 0; i < 256; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        tb.ch(0)[i] = ls[static_cast<size_t>(i)] =
            static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
        rng = rng * 1664525u + 1013904223u;
        tb.ch(1)[i] = rs[static_cast<size_t>(i)] =
            static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }
    MidSide<float>::encode(tb.view());
    int diffs = 0;
    for (int i = 0; i < 256; ++i)
    {
        MidSide<float>::encodeSample(ls[static_cast<size_t>(i)], rs[static_cast<size_t>(i)]);
        if (ls[static_cast<size_t>(i)] != tb.ch(0)[i]
            || rs[static_cast<size_t>(i)] != tb.ch(1)[i]) ++diffs;
    }
    MidSide<float>::decode(tb.view());
    for (int i = 0; i < 256; ++i)
    {
        MidSide<float>::decodeSample(ls[static_cast<size_t>(i)], rs[static_cast<size_t>(i)]);
        if (ls[static_cast<size_t>(i)] != tb.ch(0)[i]
            || rs[static_cast<size_t>(i)] != tb.ch(1)[i]) ++diffs;
    }
    EXPECT_TRUE(diffs == 0);
}

DSPARK_TEST(MidSide_extra_channels_untouched)
{
    // Channels beyond the first two are ignored by contract.
    auto tb = makeBuffer(3, 128);
    generateSine(tb.ch(0), 128, 440.0f, 48000.0f);
    generateSine(tb.ch(1), 128, 880.0f, 48000.0f);
    generateSine(tb.ch(2), 128, 1320.0f, 48000.0f);
    std::vector<float> third(tb.ch(2), tb.ch(2) + 128);

    MidSide<float>::encode(tb.view());
    MidSide<float>::decode(tb.view());

    for (int i = 0; i < 128; ++i)
        EXPECT_TRUE(tb.ch(2)[i] == third[static_cast<size_t>(i)]);
}

// ============================================================================
// StereoWidth
// ============================================================================

DSPARK_TEST(StereoWidth_invalid_inputs_are_ignored)
{
    // setWidth(NaN) used to resolve max(0, NaN) to 0: a silent collapse to
    // mono (measured: a pure-side signal came out as full silence). And
    // setBassMono(true, NaN) built a NaN crossover coefficient that poisoned
    // the side filter state permanently (100% non-finite output measured).
    const float nan = std::numeric_limits<float>::quiet_NaN();

    StereoWidth<float> sw;
    sw.prepare(48000.0);
    sw.setWidth(nan);                          // ignored: width stays 1
    auto tb = makeStereoBuffer(1024);
    for (int i = 0; i < 1024; ++i) { tb.ch(0)[i] = 0.5f; tb.ch(1)[i] = -0.5f; } // pure side
    sw.processBlock(tb.view());
    EXPECT_NEAR(tb.ch(0)[100], 0.5f, 1e-5f);   // old header: 0.0 (mono collapse)
    EXPECT_NEAR(tb.ch(1)[100], -0.5f, 1e-5f);

    StereoWidth<float> bm;
    bm.prepare(48000.0);
    bm.setBassMono(true, nan);                 // cutoff kept, toggle applies
    bm.prepare(std::numeric_limits<double>::quiet_NaN()); // ignored
    auto tb2 = makeStereoBuffer(1024);
    generateSine(tb2.ch(0), 1024, 1000.0f, 48000.0f, 0.5f);
    for (int i = 0; i < 1024; ++i) tb2.ch(1)[i] = -tb2.ch(0)[i];
    bm.processBlock(tb2.view());
    EXPECT_NO_NAN(tb2.ch(0), 1024);            // old header: all NaN
    EXPECT_NO_NAN(tb2.ch(1), 1024);
}

DSPARK_TEST(StereoWidth_1_passthrough)
{
    StereoWidth<float> sw;
    sw.prepare(44100.0);
    sw.setWidth(1.0f);

    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);
    generateSine(tb.ch(1), 256, 880.0f, 44100.0f);

    std::vector<float> origL(tb.ch(0), tb.ch(0) + 256);
    std::vector<float> origR(tb.ch(1), tb.ch(1) + 256);

    sw.processBlock(tb.view());

    for (int i = 0; i < 256; ++i)
    {
        EXPECT_NEAR(tb.ch(0)[i], origL[i], 1e-4f);
        EXPECT_NEAR(tb.ch(1)[i], origR[i], 1e-4f);
    }
}

DSPARK_TEST(StereoWidth_0_mono)
{
    StereoWidth<float> sw;
    sw.prepare(44100.0);
    sw.setWidth(0.0f);

    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);
    generateSine(tb.ch(1), 256, 880.0f, 44100.0f);

    sw.processBlock(tb.view());

    // L and R should be identical (mono)
    for (int i = 0; i < 256; ++i)
        EXPECT_NEAR(tb.ch(0)[i], tb.ch(1)[i], 1e-4f);
}

DSPARK_TEST(StereoWidth_silence)
{
    StereoWidth<float> sw;
    sw.prepare(44100.0);
    sw.setWidth(0.5f);

    auto tb = makeStereoBuffer(128);
    tb.fillSilence();
    sw.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 128, 1e-10f);
}

// ============================================================================
// Crossfade
// ============================================================================

DSPARK_TEST(Crossfade_invalid_inputs_are_ignored)
{
    // setPosition(NaN) poisoned the gains (recomputed every call since
    // NaN != lastPos was always true): 100% non-finite output measured. A
    // wild Curve id hit no switch case, leaving processAutomated's OUT
    // parameters as UNINITIALIZED locals (undefined behaviour: measured as a
    // fully muted crossfade). NaN in the automation buffer now resolves to 0
    // (100% A) instead of poisoning the block.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Crossfade<float> cf, twin;
    cf.setPosition(0.25f);
    twin.setPosition(0.25f);
    cf.setPosition(nan);                        // ignored

    std::vector<float> a(1024, 0.5f), b(1024, -0.5f), o1(1024), o2(1024);
    cf.process(a.data(), b.data(), o1.data(), 1024);
    twin.process(a.data(), b.data(), o2.data(), 1024);
    EXPECT_NO_NAN(o1.data(), 1024);
    float maxDiff = 0.0f;
    for (int i = 0; i < 1024; ++i) maxDiff = std::max(maxDiff, std::abs(o1[i] - o2[i]));
    EXPECT_TRUE(maxDiff == 0.0f);

    Crossfade<float> wild;
    wild.setCurve(static_cast<Crossfade<float>::Curve>(99)); // clamps to SCurve
    std::vector<float> pos(1024, 0.25f), o3(1024);
    wild.processAutomated(a.data(), b.data(), pos.data(), o3.data(), 1024);
    // Smoothstep(0.25) = 0.15625: out = 0.5*0.84375 - 0.5*0.15625 = 0.34375.
    EXPECT_NEAR(o3[0], 0.34375f, 1e-5f);        // old header: 0.0 (muted by UB)

    Crossfade<float> autoCf;
    std::vector<float> posNan(1024, nan), o4(1024);
    autoCf.processAutomated(a.data(), b.data(), posNan.data(), o4.data(), 1024);
    EXPECT_NO_NAN(o4.data(), 1024);
    EXPECT_NEAR(o4[0], 0.5f, 1e-5f);            // NaN position -> 100% A
}

DSPARK_TEST(Crossfade_pos0_returns_A)
{
    Crossfade<float> cf;
    cf.setPosition(0.0f);

    float result = cf.process(1.0f, 0.0f);
    EXPECT_NEAR(result, 1.0f, 1e-5f);
}

DSPARK_TEST(Crossfade_pos1_returns_B)
{
    Crossfade<float> cf;
    cf.setPosition(1.0f);

    float result = cf.process(1.0f, 0.0f);
    EXPECT_NEAR(result, 0.0f, 1e-5f);
}

DSPARK_TEST(Crossfade_EqualPower_midpoint)
{
    Crossfade<float> cf;
    cf.setCurve(Crossfade<float>::Curve::EqualPower);
    cf.setPosition(0.5f);
    (void)cf.process(1.0f, 0.0f); // refresh the audio-thread gain cache from the new position

    // At midpoint with equal-power, each side should be ~-3dB ~ 0.707
    float gainA = cf.getGainA();
    float gainB = cf.getGainB();
    EXPECT_NEAR(gainA, 0.707f, 0.05f);
    EXPECT_NEAR(gainB, 0.707f, 0.05f);
}

DSPARK_TEST(Crossfade_Linear_midpoint)
{
    Crossfade<float> cf;
    cf.setCurve(Crossfade<float>::Curve::Linear);
    cf.setPosition(0.5f);
    (void)cf.process(1.0f, 0.0f); // refresh the audio-thread gain cache from the new position

    float gainA = cf.getGainA();
    float gainB = cf.getGainB();
    EXPECT_NEAR(gainA, 0.5f, 0.01f);
    EXPECT_NEAR(gainB, 0.5f, 0.01f);
}

// ============================================================================
// Panner
// ============================================================================

DSPARK_TEST(Panner_invalid_inputs_are_ignored)
{
    // NaN setters used to poison the engine in two ways: setPan(NaN) parked
    // the pan smoother forever (the pan silently froze wherever it was), and
    // setSpectralFrequency(NaN) built NaN shelf coefficients (100% non-finite
    // output measured). All setters now ignore non-finite values, and an
    // invalid prepare() is a no-op.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Panner<float> subject, twin;
    for (auto* p : { &subject, &twin })
    {
        p->prepare(spec(44100.0, 512, 2));
        p->setAlgorithm(Panner<float>::Algorithm::EqualPower);
        p->setPan(0.25f);
    }
    subject.setPan(nan);
    subject.setSmoothingTime(nan);
    subject.setBinauralMaxITD(nan);
    subject.setHaasMaxDelay(nan);
    subject.setSpectralFrequency(nan);
    subject.setSpectralMaxGain(nan);
    AudioSpec bad;
    bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
    bad.maxBlockSize = 0; bad.numChannels = -2;
    subject.prepare(bad);
    subject.setAlgorithm(static_cast<Panner<float>::Algorithm>(99)); // clamps

    auto ta = makeStereoBuffer(8192);
    auto tb = makeStereoBuffer(8192);
    for (auto* t : { &ta, &tb })
    {
        generateSine(t->ch(0), 8192, 500.0f, 44100.0f, 0.5f);
        generateSine(t->ch(1), 8192, 500.0f, 44100.0f, 0.5f);
    }
    // The clamped wild id lands on Spectral: exercise it with the (guarded)
    // NaN frequency edit above, then compare the clean algorithm on both.
    subject.processBlock(ta.view());
    EXPECT_NO_NAN(ta.ch(0), 8192);
    EXPECT_NO_NAN(ta.ch(1), 8192);
    EXPECT_TRUE(static_cast<int>(subject.getState().size()) > 0); // serializes finite state

    subject.setAlgorithm(Panner<float>::Algorithm::EqualPower);
    generateSine(ta.ch(0), 8192, 500.0f, 44100.0f, 0.5f);
    generateSine(ta.ch(1), 8192, 500.0f, 44100.0f, 0.5f);
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());
    const float rmsL_s = measureRMS(ta.ch(0) + 4096, 4096);
    const float rmsR_s = measureRMS(ta.ch(1) + 4096, 4096);
    const float rmsL_t = measureRMS(tb.ch(0) + 4096, 4096);
    const float rmsR_t = measureRMS(tb.ch(1) + 4096, 4096);
    EXPECT_NEAR(rmsL_s, rmsL_t, 0.01f);   // pan target survived every bad edit
    EXPECT_NEAR(rmsR_s, rmsR_t, 0.01f);
}

DSPARK_TEST(Panner_smoothing_time_applies_live)
{
    // setSmoothingTime() used to only take effect on the NEXT prepare(): a
    // hot change from a 5-second ramp to 1 ms left the pan crawling on the
    // old time constant (measured: still mid-ramp after 0.12 s). It now
    // reconfigures the smoother at the top of the next processBlock().
    Panner<float> p;
    p.prepare(spec(44100.0, 512, 2));
    p.setAlgorithm(Panner<float>::Algorithm::EqualPower);
    p.setSmoothingTime(5000.0f);
    p.prepare(spec(44100.0, 512, 2));    // bake the slow ramp in
    p.setSmoothingTime(1.0f);            // hot change, no re-prepare
    p.setPan(1.0f);                      // hard right

    auto tb = makeStereoBuffer(8192);    // ~0.19 s: plenty for a 1 ms ramp
    generateSine(tb.ch(0), 8192, 500.0f, 44100.0f, 0.5f);
    generateSine(tb.ch(1), 8192, 500.0f, 44100.0f, 0.5f);
    p.processBlock(tb.view());

    const float rmsL = measureRMS(tb.ch(0) + 4096, 4096);
    const float rmsR = measureRMS(tb.ch(1) + 4096, 4096);
    EXPECT_LT(rmsL, 0.02f);              // settled hard right (old header: mid-ramp)
    EXPECT_GT(rmsR, 0.3f);
}

DSPARK_TEST(Panner_center_equal_LR)
{
    Panner<float> pan;
    pan.prepare(defaultSpec());
    pan.setPan(0.0f); // Center

    auto tb = makeStereoBuffer(4096);
    // Mono signal on both channels
    generateSine(tb.ch(0), 4096, 440.0f, 44100.0f);
    generateSine(tb.ch(1), 4096, 440.0f, 44100.0f);

    pan.processBlock(tb.view());

    // L and R should be approximately equal
    float rmsL = measureRMS(tb.ch(0) + 512, 3000);
    float rmsR = measureRMS(tb.ch(1) + 512, 3000);
    EXPECT_NEAR(rmsL, rmsR, rmsL * 0.1f);
}

DSPARK_TEST(Panner_hard_left)
{
    Panner<float> pan;
    pan.prepare(defaultSpec());
    pan.setAlgorithm(Panner<float>::Algorithm::EqualPower);
    pan.setPan(-1.0f); // Hard left

    auto tb = makeStereoBuffer(4096);
    generateSine(tb.ch(0), 4096, 440.0f, 44100.0f);
    generateSine(tb.ch(1), 4096, 440.0f, 44100.0f);

    pan.processBlock(tb.view());

    float rmsL = measureRMS(tb.ch(0) + 512, 3000);
    float rmsR = measureRMS(tb.ch(1) + 512, 3000);

    // Left should be loud, right should be quiet
    EXPECT_GT(rmsL, rmsR * 3.0f);
}

DSPARK_TEST(Panner_silence)
{
    Panner<float> pan;
    pan.prepare(defaultSpec());
    pan.setPan(0.3f);

    auto tb = makeStereoBuffer(256);
    tb.fillSilence();
    pan.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 256, 1e-8f);
    EXPECT_SILENT(tb.ch(1), 256, 1e-8f);
}

DSPARK_TEST(Panner_no_NaN)
{
    Panner<float> pan;
    pan.prepare(defaultSpec());

    for (auto algo : { Panner<float>::Algorithm::EqualPower,
                       Panner<float>::Algorithm::Binaural,
                       Panner<float>::Algorithm::Haas })
    {
        pan.setAlgorithm(algo);
        pan.setPan(0.5f);

        auto tb = makeStereoBuffer(2048);
        tb.fillNoise();
        pan.processBlock(tb.view());
        EXPECT_NO_NAN(tb.ch(0), 2048);
        EXPECT_NO_NAN(tb.ch(1), 2048);
    }
}

// M-006 AG-5 C1: Panner front-door non-finite guard. The Spectral algorithm's
// Biquad shelves (and Binaural/Haas delay lines) latch a NaN/Inf input forever;
// a transient glitch must not poison the panner. Revert-check: removing the
// guard turns this RED. Same defect class as M-005 C1.
DSPARK_TEST(Panner_survives_nonfinite_input)
{
    Panner<float> pan;
    pan.prepare(spec(48000.0, 512, 2));
    pan.setAlgorithm(Panner<float>::Algorithm::Spectral);
    pan.setPan(0.5f);
    auto p = makeBuffer(2, 512);
    p.fillSine(220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(1)[192] = std::numeric_limits<float>::infinity();
    pan.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 30; ++blk)
    {
        auto b = makeBuffer(2, 512);
        b.fillSine(220.0f, 48000.0f, 0.3f);
        pan.processBlock(b.view());
        if (blk >= 25)
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < 512; ++i)
                {
                    if (!std::isfinite(b.ch(c)[i])) finite = false;
                    energy += double(b.ch(c)[i]) * double(b.ch(c)[i]);
                }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);
}
