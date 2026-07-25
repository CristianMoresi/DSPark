// DSPark Tests - Effects Dynamics
// Compressor, Limiter, NoiseGate, Gain, AutoGain

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Effects/Compressor.h"
#include "../Effects/Limiter.h"
#include "../Effects/NoiseGate.h"
#include "../Effects/Gain.h"
#include "../Effects/AutoGain.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// Gain
// ============================================================================

DSPARK_TEST(Gain_0dB_passthrough)
{
    Gain<float> g;
    g.prepare(44100.0);
    g.setGainDb(0.0f);
    g.skipRamp();

    auto tb = makeMonoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);
    std::vector<float> original(tb.ch(0), tb.ch(0) + 256);

    g.processBlock(tb.view());

    for (int i = 0; i < 256; ++i)
        EXPECT_NEAR(tb.ch(0)[i], original[i], 1e-5f);
}

DSPARK_TEST(Gain_minus6dB_halves)
{
    Gain<float> g;
    g.prepare(44100.0);
    g.setGainDb(-6.02f);
    g.skipRamp();

    auto tb = makeMonoBuffer(256);
    generateDC(tb.ch(0), 256, 1.0f);

    g.processBlock(tb.view());

    EXPECT_NEAR(tb.ch(0)[200], 0.5f, 0.01f);
}

DSPARK_TEST(Gain_mute)
{
    Gain<float> g;
    g.prepare(44100.0);
    g.setMuted(true);
    g.skipRamp();

    auto tb = makeMonoBuffer(512);
    tb.fillSine(440.0f, 44100.0f);

    g.processBlock(tb.view());

    // After ramp, should be silent
    EXPECT_SILENT(tb.ch(0) + 256, 256, 0.01f);
}

DSPARK_TEST(Gain_invert_polarity)
{
    Gain<float> g;
    g.prepare(44100.0);
    g.setGainLinear(1.0f);
    g.setInverted(true);
    g.skipRamp();

    auto tb = makeMonoBuffer(64);
    generateDC(tb.ch(0), 64, 0.7f);

    g.processBlock(tb.view());
    EXPECT_NEAR(tb.ch(0)[32], -0.7f, 0.01f);
}

DSPARK_TEST(Gain_silence_stays_silent)
{
    Gain<float> g;
    g.prepare(44100.0);
    g.setGainDb(-12.0f);
    g.skipRamp();

    auto tb = makeStereoBuffer(256);
    tb.fillSilence();
    g.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 256, 1e-10f);
}

DSPARK_TEST(Gain_invalid_inputs_are_ignored)
{
    // setGainLinear(NaN) used to resolve max(0, NaN) to 0: a silent
    // accidental MUTE (measured: the signal ramped to ~0 while the twin
    // stayed at unity). setGainDb(NaN) parked a NaN in the target atomic
    // (poisoned getters and serialized state). Both are inert now, as is an
    // invalid prepare().
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Gain<float> subject, twin;
    subject.prepare(48000.0);
    twin.prepare(48000.0);
    subject.setGainLinear(1.0f);
    twin.setGainLinear(1.0f);

    subject.setGainLinear(nan);
    subject.setGainDb(nan);
    subject.setRampTime(nan);
    subject.prepare(std::numeric_limits<double>::quiet_NaN());
    subject.prepare(0.0);

    auto ta = makeBuffer(1, 4096);
    auto tb = makeBuffer(1, 4096);
    for (int i = 0; i < 4096; ++i) { ta.ch(0)[i] = 0.5f; tb.ch(0)[i] = 0.5f; }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 4096);
    float maxDiff = 0.0f;
    for (int i = 0; i < 4096; ++i)
        maxDiff = std::max(maxDiff, std::abs(ta.ch(0)[i] - tb.ch(0)[i]));
    EXPECT_TRUE(maxDiff == 0.0f);              // no accidental mute, fully inert
    EXPECT_NEAR(subject.getGainLinear(), 1.0f, 1e-6f); // target survived clean
}

// ============================================================================
// Compressor
// ============================================================================

DSPARK_TEST(Compressor_below_threshold_passthrough)
{
    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(defaultSpec());
    comp.setThreshold(0.0f);  // 0 dBFS threshold
    comp.setRatio(4.0f);
    comp.setAttack(1.0f);
    comp.setRelease(50.0f);

    // Signal at -20 dBFS - well below threshold
    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f, 0.1f); // ~-20 dBFS

    float peakBefore = measurePeak(tb.ch(0), 4096);
    comp.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0), 4096);

    // Should be nearly unchanged
    EXPECT_NEAR(peakAfter, peakBefore, peakBefore * 0.1f);
}

DSPARK_TEST(Compressor_above_threshold_reduces)
{
    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(defaultSpec());
    comp.setThreshold(-20.0f);
    comp.setRatio(10.0f);
    comp.setAttack(0.1f);
    comp.setRelease(50.0f);

    // Signal at 0 dBFS - 20dB above threshold
    auto tb = makeStereoBuffer(8192);
    tb.fillSine(440.0f, 44100.0f, 1.0f);

    comp.processBlock(tb.view());

    float peakAfter = measurePeak(tb.ch(0), 8192);
    // With high ratio, should be significantly reduced
    EXPECT_LT(peakAfter, 0.5f);
}

DSPARK_TEST(Compressor_silence_stays_silent)
{
    Compressor<float> comp;
    comp.prepare(defaultSpec());
    comp.setThreshold(-20.0f);
    comp.setRatio(4.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    comp.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-8f);
}

DSPARK_TEST(Compressor_no_NaN)
{
    Compressor<float> comp;
    comp.prepare(defaultSpec());
    comp.setThreshold(-10.0f);
    comp.setRatio(4.0f);
    comp.setAttack(1.0f);
    comp.setRelease(50.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise(1.0f);
    comp.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);
}

// ============================================================================
// Limiter
// ============================================================================

DSPARK_TEST(Limiter_never_exceeds_ceiling)
{
    Limiter<float> lim;
    lim.prepare(44100.0);
    lim.setCeiling(-1.0f); // -1 dBFS

    float ceiling = decibelsToGain(-1.0f);

    // Feed loud random signal
    auto tb = makeStereoBuffer(8192);
    tb.fillNoise(2.0f); // Peaks at +/-2.0

    // Process multiple blocks to ensure limiter settles
    for (int block = 0; block < 4; ++block)
    {
        tb.fillNoise(2.0f, static_cast<uint32_t>(block * 1000 + 1));
        lim.processBlock(tb.view());

        // Check all samples (skip first block for lookahead settling)
        if (block > 0)
        {
            for (int i = 0; i < 8192; ++i)
            {
                EXPECT_TRUE(std::abs(tb.ch(0)[i]) <= ceiling + 0.05f);
                EXPECT_TRUE(std::abs(tb.ch(1)[i]) <= ceiling + 0.05f);
            }
        }
    }
}

DSPARK_TEST(Limiter_below_ceiling_passthrough)
{
    Limiter<float> lim;
    lim.prepare(44100.0);
    lim.setCeiling(0.0f);

    // Signal at -20 dBFS
    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f, 0.1f);

    float peakBefore = measurePeak(tb.ch(0), 4096);
    lim.processBlock(tb.view());

    // Account for limiter latency - check in the later portion
    float peakAfter = measurePeak(tb.ch(0) + 512, 3000);
    EXPECT_NEAR(peakAfter, peakBefore, peakBefore * 0.15f);
}

DSPARK_TEST(Limiter_silence)
{
    Limiter<float> lim;
    lim.prepare(44100.0);
    lim.setCeiling(-1.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    lim.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-8f);
}

DSPARK_TEST(Limiter_degenerate_channel_count_is_safe)
{
    // prepare() clamps the channel count to [1, kMaxChannels]; the delay-line
    // allocation must follow the clamped value. A non-positive numChannels used
    // to under-allocate (empty vector) and make processBlock read out of bounds.
    {
        Limiter<float> lim;
        lim.prepare(44100.0, 0);          // degenerate -> clamps to 1 channel
        lim.setCeiling(-1.0f);
        auto tb = makeMonoBuffer(256);
        tb.fillNoise(2.0f);
        lim.processBlock(tb.view());      // must not crash / read OOB
        EXPECT_NO_NAN(tb.ch(0), 256);
        EXPECT_BOUNDED(tb.ch(0), 256, -1.5f, 1.5f);
    }
    {
        Limiter<float> lim;
        lim.prepare(44100.0, 64);         // over-range -> clamps to kMaxChannels
        lim.setCeiling(-1.0f);
        auto tb = makeStereoBuffer(256);
        tb.fillNoise(2.0f);
        lim.processBlock(tb.view());      // must not crash
        EXPECT_NO_NAN(tb.ch(0), 256);
    }
}

DSPARK_TEST(Limiter_invalid_inputs_are_ignored)
{
    // setRelease(NaN) used to poison the release coefficient permanently
    // (measured: 1024/1024 non-finite samples forever), setCeiling(NaN)
    // parked NaN in the state blob and killed the limiter on the next
    // prepare() (measured: 2.0 peaks pass with the meter reading 0 dB GR),
    // setCeiling(+Inf) slipped past the smoother's NaN-only guard and
    // silently disabled limiting, setLookahead(NaN) collapsed getLatency()
    // from 96 to 1 through a UB float->int cast, and an invalid prepare()
    // resized the delay lines through UB casts (measured: full NaN output
    // with adaptive release). All inert now: bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Limiter<float> subject, twin;
    auto setup = [](Limiter<float>& l) {
        l.prepare(48000.0, 2);
        l.setCeiling(-6.0f);
        l.setRelease(80.0f);
        l.setAdaptiveRelease(true);
    };
    setup(subject); setup(twin);

    subject.setCeiling(nan);   subject.setCeiling(inf);  subject.setCeiling(-inf);
    subject.setRelease(nan);   subject.setRelease(inf);
    subject.setLookahead(nan); subject.setLookahead(inf);
    subject.prepare(std::numeric_limits<double>::quiet_NaN(), 2);
    subject.prepare(0.0);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));

    EXPECT_EQ(subject.getLatency(), 96);            // 2 ms @ 48 kHz survived it all
    EXPECT_NEAR(subject.getLookahead(), 2.0f, 1e-6f);

    // Wild channels: exact pass-through, no state touched.
    EXPECT_EQ(subject.processSample(0.33f, 99), 0.33f);
    EXPECT_EQ(subject.processSample(0.5f, -1), 0.5f);

    // The dirty parameters never reach the serialized state.
    auto blob = subject.getState();
    StateReader r(blob.data(), blob.size());
    EXPECT_NEAR(r.read("ceiling", 999.0f), -6.0f, 1e-6f);
    EXPECT_NEAR(r.read("release", 999.0f), 80.0f, 1e-6f);

    auto ta = makeBuffer(2, 4096);
    auto tb = makeBuffer(2, 4096);
    for (int ch = 0; ch < 2; ++ch)
    {
        generateSine(ta.ch(ch), 4096, ch == 0 ? 440.0f : 620.0f, 48000.0f, 1.4f);
        generateSine(tb.ch(ch), 4096, ch == 0 ? 440.0f : 620.0f, 48000.0f, 1.4f);
    }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 4096);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
}

DSPARK_TEST(Limiter_block_matches_per_sample)
{
    // The per-sample path used to skip the lookahead peak hold, so a transient
    // reached the delayed output after the envelope had already released and
    // was hard-clipped at the ceiling instead of smoothly limited (measured:
    // maxDiff 0.01 vs the block path on an impulse train). The paths are now
    // bit-identical for mono, with every feature branch enabled.
    Limiter<float> lb, ls;
    auto setup = [](Limiter<float>& l) {
        // Ceiling before prepare(): the smoother starts settled at -6 dB, so
        // the brickwall bound below holds from sample 0 (no 30 ms glide).
        l.setCeiling(-6.0f);
        l.prepare(48000.0, 1);
        l.setTruePeak(true);
        l.setAdaptiveRelease(true);
        l.setSafetyClip(true);
    };
    setup(lb); setup(ls);

    const int n = 4096;
    auto in = makeBuffer(1, n);
    generateSine(in.ch(0), n, 330.0f, 48000.0f, 0.05f);
    for (int k = 500; k < n; k += 997) in.ch(0)[k] = 1.0f; // impulse train

    auto blockOut = makeBuffer(1, n);
    std::copy(in.ch(0), in.ch(0) + n, blockOut.ch(0));
    for (int off = 0; off < n; off += 512)
    {
        float* p[1] = { blockOut.ch(0) + off };
        lb.processBlock(AudioBufferView<float>(p, 1, 512));
    }

    float maxDiff = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        const float s = ls.processSample(in.ch(0)[i], 0);
        maxDiff = std::max(maxDiff, std::abs(s - blockOut.ch(0)[i]));
    }
    EXPECT_TRUE(maxDiff == 0.0f); // bit-exact, hold + glide + safety clip included

    // And the brickwall holds on the per-sample path too.
    const float ceiling = decibelsToGain(-6.0f);
    for (int i = 0; i < n; ++i)
        EXPECT_TRUE(std::abs(blockOut.ch(0)[i]) <= ceiling + 1e-6f);
}

DSPARK_TEST(Limiter_prepare_preserves_lookahead)
{
    // prepare(AudioSpec) used to reset a configured lookahead back to the
    // 2 ms constructor default (measured: getLatency() 384 -> 96 after a
    // host re-activation), silently desyncing the host's delay compensation.
    Limiter<float> lim;
    lim.prepare(48000.0, 2);
    lim.setLookahead(8.0f);
    EXPECT_EQ(lim.getLatency(), 384);          // published immediately

    lim.prepare(spec(48000.0, 512, 2));        // host re-activation
    EXPECT_EQ(lim.getLatency(), 384);          // parameter survived
    EXPECT_NEAR(lim.getLookahead(), 8.0f, 1e-6f);

    lim.prepare(44100.0, 2, 3.0);              // explicit override still works
    EXPECT_EQ(lim.getLatency(), 132);          // 3 ms @ 44.1 kHz
    EXPECT_NEAR(lim.getLookahead(), 3.0f, 1e-6f);
}

// ============================================================================
// NoiseGate
// ============================================================================

DSPARK_TEST(NoiseGate_below_threshold_silences)
{
    NoiseGate<float> gate;
    gate.prepare(44100.0);
    gate.setThreshold(-10.0f);
    gate.setAttack(0.1f);
    gate.setHold(1.0f);
    gate.setRelease(10.0f);
    gate.setRange(-80.0f);

    // Quiet signal: -40 dBFS (well below -10dB threshold)
    auto tb = makeMonoBuffer(8192);
    tb.fillSine(440.0f, 44100.0f, 0.01f);

    gate.processBlock(tb.view());

    // After gate settles, output should be very quiet
    float peakLate = measurePeak(tb.ch(0) + 4096, 4096);
    EXPECT_LT(peakLate, 0.002f);
}

DSPARK_TEST(NoiseGate_above_threshold_passes)
{
    NoiseGate<float> gate;
    gate.prepare(44100.0);
    gate.setThreshold(-40.0f);
    gate.setAttack(0.1f);
    gate.setRelease(50.0f);

    // Loud signal: 0 dBFS
    auto tb = makeMonoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f, 1.0f);

    float peakBefore = measurePeak(tb.ch(0), 4096);
    gate.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 512, 3000);

    EXPECT_GT(peakAfter, peakBefore * 0.8f);
}

DSPARK_TEST(NoiseGate_silence_stays_silent)
{
    NoiseGate<float> gate;
    gate.prepare(44100.0);
    gate.setThreshold(-60.0f);

    auto tb = makeMonoBuffer(512);
    tb.fillSilence();
    gate.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

DSPARK_TEST(NoiseGate_invalid_inputs_are_ignored)
{
    // setThreshold(NaN) used to leave the gate permanently closed (measured:
    // 5e-5 peak on a 0.5 signal), setAttack/setRange(NaN) poisoned the gain
    // (6k+ non-finite samples), setHysteresis/setHold(NaN) left it permanently
    // open, a negative HPF cutoff made the detector filter unstable (gate
    // stuck open), an HPF NaN collapsed the level to zero (permanent mute),
    // prepare(NaN) poisoned the envelope into a silent permanent mute, and
    // prepare(0) stored the dead rate so every future setter was ignored.
    // All inert now: bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    NoiseGate<float> subject, twin;
    auto setup = [](NoiseGate<float>& g) {
        g.prepare(48000.0);
        g.setThreshold(-40.0f); g.setHysteresis(4.0f);
        g.setAttack(0.5f); g.setHold(20.0f); g.setRelease(50.0f);
        g.setRange(-80.0f);
        g.setSidechainHPF(true, 80.0);
    };
    setup(subject); setup(twin);

    subject.setThreshold(nan);   subject.setThreshold(inf);
    subject.setHysteresis(nan);  subject.setAttack(nan);
    subject.setHold(nan);        subject.setRelease(nan);
    subject.setRange(nan);       subject.setRange(-inf);
    subject.setSidechainHPF(true, -500.0);
    subject.setSidechainHPF(true, std::nan(""));
    subject.setGateMode(static_cast<NoiseGate<float>::GateMode>(99)); // clamps
    subject.setGateMode(NoiseGate<float>::GateMode::Amplitude);
    subject.prepare(0.0);
    subject.prepare(std::numeric_limits<double>::quiet_NaN());
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));

    // Setters must still work after an invalid prepare (it used to store the
    // dead rate and silently ignore every later parameter change).
    subject.setThreshold(-30.0f);
    twin.setThreshold(-30.0f);

    // The dirty parameters never reach the serialized state.
    auto blob = subject.getState();
    StateReader r(blob.data(), blob.size());
    EXPECT_NEAR(r.read("threshold", 999.0f), -30.0f, 1e-6f);
    EXPECT_NEAR(r.read("hysteresis", 999.0f), 4.0f, 1e-6f);
    EXPECT_NEAR(r.read("scHpfFreq", 999.0f), 80.0f, 1e-6f);
    EXPECT_TRUE(r.read("gateMode", 7) == 0);

    auto ta = makeBuffer(2, 6144);
    auto tb = makeBuffer(2, 6144);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 6144; ++i)
        {
            const float amp = (i / 1500) % 2 == 0 ? 0.5f : 0.001f; // bursts
            const float v = amp * std::sin(6.2831853f * 440.0f * i / 48000.0f);
            ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
        }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 6144);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 6144; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin

    // And the gate still actually gates (opens on the burst).
    float burstPeak = measurePeak(ta.ch(0) + 512, 900);
    EXPECT_GT(burstPeak, 0.3f);
}

DSPARK_TEST(NoiseGate_block_matches_per_sample)
{
    // The block path used to measure the zero-crossing rate on the RAW
    // channel 0 while the per-sample path measured it on the HPF'd detection
    // signal: with strong lows plus weak highs the adaptive hold diverged
    // (measured: maxDiff 0.005 between paths). Both paths now track the same
    // (filtered) signal and are bit-identical for mono.
    for (int cfg = 0; cfg < 2; ++cfg)
    {
        NoiseGate<float> gb, gs;
        auto setup = [&](NoiseGate<float>& g) {
            g.prepare(48000.0);
            g.setThreshold(-40.0f); g.setRelease(20.0f);
            if (cfg == 1) { g.setHold(0.0f); g.setSidechainHPF(true, 1000.0); g.setAdaptiveHold(true); }
            else          { g.setHold(10.0f); }
        };
        setup(gb); setup(gs);

        const int n = 6144;
        auto in = makeBuffer(1, n);
        for (int i = 0; i < n; ++i)
        {
            // Strong 60 Hz + weak 5 kHz: raw and HPF'd zero-cross rates differ
            // wildly, so a divergent adaptive hold shows immediately.
            const float amp = (i / 1500) % 2 == 0 ? 0.5f : 0.001f;
            in.ch(0)[i] = amp * (std::sin(6.2831853f * 60.0f * i / 48000.0f)
                                 + 0.05f * std::sin(6.2831853f * 5000.0f * i / 48000.0f));
        }

        auto blockOut = makeBuffer(1, n);
        std::copy(in.ch(0), in.ch(0) + n, blockOut.ch(0));
        for (int off = 0; off < n; off += 512)
        {
            float* p[1] = { blockOut.ch(0) + off };
            gb.processBlock(AudioBufferView<float>(p, 1, 512));
        }

        float maxDiff = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            const float s = gs.processSample(in.ch(0)[i]);
            maxDiff = std::max(maxDiff, std::abs(s - blockOut.ch(0)[i]));
        }
        EXPECT_TRUE(maxDiff == 0.0f); // bit-exact in both configurations
    }
}

DSPARK_TEST(NoiseGate_sidechain_path_matches_mono_path)
{
    // The multi-channel external-sidechain path used to silently ignore
    // Frequency mode and the adaptive hold (measured: maxDiff 0.233 against
    // the mono/mono fast path on identical material). Detection and gating
    // now behave identically in both paths.
    NoiseGate<float> g1, g2;
    auto setup = [](NoiseGate<float>& g) {
        g.prepare(48000.0);
        g.setThreshold(-40.0f); g.setHold(10.0f); g.setRelease(20.0f);
        g.setGateMode(NoiseGate<float>::GateMode::Frequency);
    };
    setup(g1); setup(g2);

    const int n = 4096;
    auto audio = makeBuffer(1, n);
    auto sc = makeBuffer(1, n);
    for (int i = 0; i < n; ++i)
    {
        const float amp = (i / 1000) % 2 == 0 ? 0.5f : 0.001f;
        audio.ch(0)[i] = 0.4f * std::sin(6.2831853f * 330.0f * i / 48000.0f);
        sc.ch(0)[i] = amp * std::sin(6.2831853f * 440.0f * i / 48000.0f);
    }

    // Mono/mono fast path.
    auto outMono = makeBuffer(1, n);
    std::copy(audio.ch(0), audio.ch(0) + n, outMono.ch(0));
    g1.processBlock(outMono.view(), sc.view());

    // Duplicated 2-channel path must produce the same result per channel.
    auto a2 = makeBuffer(2, n);
    auto s2 = makeBuffer(2, n);
    for (int ch = 0; ch < 2; ++ch)
    {
        std::copy(audio.ch(0), audio.ch(0) + n, a2.ch(ch));
        std::copy(sc.ch(0), sc.ch(0) + n, s2.ch(ch));
    }
    g2.processBlock(a2.view(), s2.view());

    float maxDiff = 0.0f;
    float processed = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        maxDiff = std::max(maxDiff, std::abs(a2.ch(0)[i] - outMono.ch(0)[i]));
        processed = std::max(processed, std::abs(outMono.ch(0)[i] - audio.ch(0)[i]));
    }
    EXPECT_TRUE(maxDiff == 0.0f);  // both paths bit-identical
    EXPECT_GT(processed, 0.01f);   // and the frequency gate actually engaged
}

// ============================================================================
// Compressor - Upward Compression
// ============================================================================

DSPARK_TEST(Compressor_upward_boosts_quiet)
{
    auto s = spec(48000.0, 4096, 1);
    Compressor<float> comp;
    comp.prepare(s);
    comp.setThreshold(-10.0f);   // Threshold at -10 dB
    comp.setRatio(4.0f);
    comp.setMode(Compressor<float>::Mode::Upward);
    comp.setAttack(0.1f);
    comp.setRelease(10.0f);

    // Generate a quiet signal (-30 dB)
    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 440.0f, 48000.0f, 0.03f); // ~-30 dBFS

    // Measure input energy
    float inEnergy = 0.0f;
    for (int i = 0; i < 4096; ++i)
        inEnergy += tb.ch(0)[i] * tb.ch(0)[i];

    comp.processBlock(tb.view());

    // Output should be louder (upward compression boosts)
    float outEnergy = 0.0f;
    for (int i = 512; i < 4096; ++i) // skip attack transient
        outEnergy += tb.ch(0)[i] * tb.ch(0)[i];

    // inEnergy for the same range
    float inEnergySame = 0.0f;
    auto ref = makeBuffer(1, 4096);
    generateSine(ref.ch(0), 4096, 440.0f, 48000.0f, 0.03f);
    for (int i = 512; i < 4096; ++i)
        inEnergySame += ref.ch(0)[i] * ref.ch(0)[i];

    EXPECT_GT(outEnergy, inEnergySame * 1.5f); // Should be significantly louder
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(Compressor_upward_leaves_loud_untouched)
{
    auto s = spec(48000.0, 4096, 1);
    Compressor<float> comp;
    comp.prepare(s);
    comp.setThreshold(-20.0f);
    comp.setRatio(4.0f);
    comp.setMode(Compressor<float>::Mode::Upward);

    // Generate a loud signal (0 dBFS) - above threshold
    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 440.0f, 48000.0f, 1.0f);
    auto ref = makeBuffer(1, 4096);
    generateSine(ref.ch(0), 4096, 440.0f, 48000.0f, 1.0f);

    comp.processBlock(tb.view());

    // Energy should be similar (no boost above threshold)
    float outE = 0.0f, inE = 0.0f;
    for (int i = 512; i < 4096; ++i)
    {
        outE += tb.ch(0)[i] * tb.ch(0)[i];
        inE += ref.ch(0)[i] * ref.ch(0)[i];
    }
    EXPECT_NEAR(outE, inE, inE * 0.2f); // Within 20%
}

DSPARK_TEST(Compressor_downward_still_works)
{
    // Regression: default mode should still compress downward
    auto s = spec(48000.0, 4096, 1);
    Compressor<float> comp;
    comp.prepare(s);
    comp.setThreshold(-20.0f);
    comp.setRatio(8.0f);
    comp.setAttack(0.1f);

    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 440.0f, 48000.0f, 1.0f);

    comp.processBlock(tb.view());

    // Should show gain reduction
    EXPECT_LT(comp.getGainReductionDb(), -1.0f);
}

// ============================================================================
// Compressor dynamics-audit regressions (2026-07 P0 fixes)
// ============================================================================

DSPARK_TEST(Compressor_splitpolarity_is_samplerate_invariant)
{
    // The SplitPolarity detector coefficients were fixed per-sample constants:
    // the same preset compressed 2.2 dB less at 96 kHz than at 48 kHz.
    float gainAtFs[2] = { 0.0f, 0.0f };
    const double rates[2] = { 48000.0, 96000.0 };

    for (int k = 0; k < 2; ++k)
    {
        const double fs = rates[k];
        const int n = static_cast<int>(fs * 1.5);
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 100.0f, static_cast<float>(fs), 1.0f);
        const int tail = static_cast<int>(fs * 0.5);
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(5.0f);
        comp.setRelease(200.0f);
        comp.setDetector(Compressor<float>::DetectorType::SplitPolarity);
        comp.processBlock(tb.view());

        gainAtFs[k] = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    }

    EXPECT_NEAR(gainAtFs[0], gainAtFs[1], 0.5f);
    EXPECT_LT(gainAtFs[0], -8.0f); // still genuinely compressing
}

DSPARK_TEST(Compressor_sidechain_hpf_works_in_feedback)
{
    // The HPF used to filter the (unused) input while the FeedBack detector
    // read the raw compressed output: the filter was inaudible in FeedBack.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);
    float gainHpf[2] = { 0.0f, 0.0f };

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 60.0f, static_cast<float>(fs), 0.5f); // -6 dBFS
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(5.0f);
        comp.setRelease(100.0f);
        comp.setTopology(Compressor<float>::Topology::FeedBack);
        comp.setSidechainHPF(k == 1, 500.0f);
        comp.processBlock(tb.view());

        gainHpf[k] = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    }

    EXPECT_LT(gainHpf[0], -3.0f);           // HPF off: 60 Hz drives compression
    EXPECT_GT(gainHpf[1], -0.5f);           // HPF on: 60 Hz filtered out of the key
}

DSPARK_TEST(Compressor_makeup_change_is_click_free)
{
    // setMakeupGain() used to apply per-block with no smoothing: a +12 dB
    // change produced a 4.5x sample step (a hard click).
    const double fs = 44100.0;
    const int n = static_cast<int>(fs);
    const float amp = 0.0316f; // ~-30 dBFS
    // Non-integer frequency: the switch lands at t = 0.5 s and any integer
    // frequency sits exactly on a zero crossing there, hiding the step.
    const float freq = 991.7f;

    auto tb = makeBuffer(1, n);
    generateSine(tb.ch(0), n, freq, static_cast<float>(fs), amp);

    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.setThreshold(0.0f); // no gain reduction, isolate the makeup path
    comp.setRatio(1.0f);
    comp.prepare(spec(fs, 64, 1));

    bool switched = false;
    for (int pos = 0; pos < n; pos += 64)
    {
        if (!switched && pos >= n / 2) { comp.setMakeupGain(12.0f); switched = true; }
        const int len = std::min(64, n - pos);
        float* chans[1] = { tb.ch(0) + pos };
        AudioBufferView<float> view(chans, 1, len);
        comp.processBlock(view);
    }

    float maxStep = 0.0f;
    for (int i = 1; i < n; ++i)
        maxStep = std::max(maxStep, std::abs(tb.ch(0)[i] - tb.ch(0)[i - 1]));

    // Natural sample delta of the boosted sine, with headroom for smoothing.
    const float naturalStep = amp * 3.981f * dspark::twoPi<float> * freq / static_cast<float>(fs);
    EXPECT_LT(maxStep, naturalStep * 1.5f);
}

DSPARK_TEST(Compressor_upward_silence_guard)
{
    // Upward boost must fade out well below the threshold; the detector floor
    // used to receive the largest boost of all (noise pumped up in pauses).
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);
    const float levels[2] = { 5.6e-5f /* ~-85 dBFS */, 0.01f /* -40 dBFS */ };
    float netGain[2] = { 0.0f, 0.0f };

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), levels[k]);
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(2.0f);
        comp.setAttack(5.0f);
        comp.setRelease(100.0f);
        comp.setMode(Compressor<float>::Mode::Upward);
        comp.processBlock(tb.view());

        netGain[k] = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    }

    EXPECT_NEAR(netGain[0], 0.0f, 1.0f); // -85 dB: guard active, no boost
    EXPECT_GT(netGain[1], 6.0f);         // -40 dB: normal upward boost intact
}

DSPARK_TEST(Compressor_release_clamped_to_1ms)
{
    // Sub-millisecond releases disabled envelope smoothing entirely (the gain
    // followed the rectified waveform). Requests below 1 ms now clamp to 1 ms.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 0.5);
    float out[2][2] = {{ 0.0f, 0.0f }, { 0.0f, 0.0f }}; // [instance][probe]

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 100.0f, static_cast<float>(fs), 1.0f);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(1.0f);
        comp.setRelease(k == 0 ? 0.01f : 1.0f);
        comp.processBlock(tb.view());

        out[k][0] = tb.ch(0)[n / 2];
        out[k][1] = measureRMSDb(tb.ch(0) + n / 2, n / 2);
    }

    EXPECT_NEAR(out[0][0], out[1][0], 1e-6f);
    EXPECT_NEAR(out[0][1], out[1][1], 1e-4f);
}

DSPARK_TEST(Compressor_default_automakeup_is_off)
{
    // A fresh compressor must actually lower the level (the old default kept
    // the average output loudness-matched to the input).
    const double fs = 48000.0;
    const int n = static_cast<int>(fs);
    const int tail = static_cast<int>(fs * 0.25);

    auto tb = makeBuffer(1, n);
    generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), 1.0f);
    const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

    Compressor<float> comp; // defaults untouched on purpose
    comp.prepare(spec(fs, n, 1));
    comp.setThreshold(-20.0f);
    comp.setRatio(8.0f);
    comp.setAttack(0.1f);
    comp.setRelease(100.0f);
    comp.processBlock(tb.view());

    EXPECT_LT(measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb, -8.0f);
}

DSPARK_TEST(Compressor_processSample_shared_state_is_channel_count_invariant)
{
    // Shared smoothers used to advance once per processSample() call, so their
    // time constants scaled with the caller's channel count. Channel 0 now
    // drives them: a mono and a stereo per-sample stream must match exactly.
    const double fs = 48000.0;
    const int n = 4096;

    Compressor<float> mono, stereo;
    for (Compressor<float>* c : { &mono, &stereo })
    {
        c->setAutoMakeup(false);
        c->prepare(spec(fs, 64, 2));
        c->setThreshold(0.0f);
        c->setRatio(4.0f);
        c->setAttack(1.0f);
        c->setRelease(50.0f);
    }

    float maxDiff = 0.0f;
    const float inc = dspark::twoPi<float> * 997.0f / static_cast<float>(fs);
    for (int i = 0; i < n; ++i)
    {
        if (i == n / 2) // mid-stream change exercises the parameter smoothers
        {
            mono.setThreshold(-30.0f);
            stereo.setThreshold(-30.0f);
        }
        const float x = 0.9f * std::sin(inc * static_cast<float>(i));
        const float a = mono.processSample(x, 0);
        const float b0 = stereo.processSample(x, 0);
        (void)stereo.processSample(x, 1);
        maxDiff = std::max(maxDiff, std::abs(a - b0));
    }

    EXPECT_NEAR(maxDiff, 0.0f, 1e-7f);
}

// ============================================================================
// Compressor character models (2026-07 P1: log-domain + T4/1176 ballistics)
// ============================================================================

namespace {

// Feeds a burst followed by a quiet floor, processing in 1-sample blocks, and
// returns the GR meter trace (one reading per sample).
inline std::vector<float> compressorGrTrace(Compressor<float>& comp, double fs,
                                            double burstSec, double tailSec)
{
    const int nBurst = static_cast<int>(fs * burstSec);
    const int nTail  = static_cast<int>(fs * tailSec);
    std::vector<float> in(static_cast<size_t>(nBurst + nTail));
    const float inc = dspark::twoPi<float> * 1000.0f / static_cast<float>(fs);
    for (int i = 0; i < nBurst + nTail; ++i)
    {
        const float amp = (i < nBurst) ? 1.0f : 0.001f; // 0 dBFS then -60 dB
        in[static_cast<size_t>(i)] = amp * std::sin(inc * static_cast<float>(i));
    }

    std::vector<float> trace(in.size());
    for (size_t i = 0; i < in.size(); ++i)
    {
        float* chans[1] = { &in[i] };
        AudioBufferView<float> view(chans, 1, 1);
        comp.processBlock(view);
        trace[i] = comp.getGainReductionDb();
    }
    return trace;
}

inline void setupCharacterComp(Compressor<float>& comp, double fs,
                               Compressor<float>::Character ch, float releaseMs)
{
    comp.setAutoMakeup(false);
    comp.prepare(spec(fs, 64, 1));
    comp.setThreshold(-20.0f);
    comp.setRatio(4.0f);
    comp.setAttack(1.0f);
    comp.setRelease(releaseMs);
    comp.setCharacter(ch);
}

} // namespace

DSPARK_TEST(Compressor_clean_release_knob_is_t63_in_dB)
{
    // Log-domain contract: after a fully settled burst, the remaining GR one
    // release time after the drop must be ~exp(-1) of the settled depth.
    const double fs = 48000.0;
    Compressor<float> comp;
    setupCharacterComp(comp, fs, Compressor<float>::Character::Clean, 400.0f);

    auto trace = compressorGrTrace(comp, fs, 1.0, 1.5);
    const size_t t0 = static_cast<size_t>(fs * 1.0);
    const float depth = -trace[t0 - 1];
    const float at400 = -trace[t0 + static_cast<size_t>(fs * 0.4)];

    EXPECT_GT(depth, 12.0f);
    EXPECT_NEAR(at400 / depth, 0.37f, 0.07f);
}

DSPARK_TEST(Compressor_opto_recovers_half_at_release_time)
{
    // T4 model: with the slow stage fully charged, ~50% of the reduction
    // recovers in one release time and the rest hangs on the memory tail.
    const double fs = 48000.0;
    Compressor<float> comp;
    setupCharacterComp(comp, fs, Compressor<float>::Character::Opto, 100.0f);

    auto trace = compressorGrTrace(comp, fs, 4.0, 1.0); // 4 s: full charge
    const size_t t0 = static_cast<size_t>(fs * 4.0);
    const float depth = -trace[t0 - 1];
    const float at100 = -trace[t0 + static_cast<size_t>(fs * 0.1)];

    EXPECT_GT(depth, 12.0f);
    EXPECT_NEAR(at100 / depth, 0.50f, 0.10f);
}

DSPARK_TEST(Compressor_opto_release_has_memory)
{
    // Brief peaks must release fast; sustained compression must leave a slow
    // tail (photocell memory). The old model had no history state at all.
    const double fs = 48000.0;
    float residual[2] = { 0.0f, 0.0f };
    const double bursts[2] = { 0.15, 4.0 };

    for (int k = 0; k < 2; ++k)
    {
        Compressor<float> comp;
        setupCharacterComp(comp, fs, Compressor<float>::Character::Opto, 100.0f);
        auto trace = compressorGrTrace(comp, fs, bursts[k], 1.0);
        const size_t probe = static_cast<size_t>(fs * bursts[k]) + static_cast<size_t>(fs * 0.3);
        residual[k] = -trace[probe];
    }

    EXPECT_GT(residual[1], residual[0] + 2.0f); // sustained leaves a real tail
    EXPECT_LT(residual[0], 2.5f);               // brief peak lets go quickly
}

DSPARK_TEST(Compressor_fet_release_t63_tracks_knob)
{
    // 1176 model: the compound two-stage release passes ~t63 near the knob.
    const double fs = 48000.0;
    Compressor<float> comp;
    setupCharacterComp(comp, fs, Compressor<float>::Character::FET, 200.0f);

    auto trace = compressorGrTrace(comp, fs, 4.0, 1.0);
    const size_t t0 = static_cast<size_t>(fs * 4.0);
    const float depth = -trace[t0 - 1];
    const float at200 = -trace[t0 + static_cast<size_t>(fs * 0.2)];

    EXPECT_GT(depth, 6.0f); // well developed (P3: the loop settles on the panel curve)
    EXPECT_NEAR(at200 / depth, 0.35f, 0.10f);
}

DSPARK_TEST(Compressor_fet_attack_clamps_to_hardware_range)
{
    // A 10 ms attack request must clamp to the 1176's 0.8 ms maximum: the
    // gain reduction has to be well developed within the first millisecond.
    const double fs = 48000.0;
    float grAt1ms[2] = { 0.0f, 0.0f };
    const Compressor<float>::Character chars[2] = {
        Compressor<float>::Character::FET, Compressor<float>::Character::Clean };

    for (int k = 0; k < 2; ++k)
    {
        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, 64, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(10.0f); // out of the FET hardware range on purpose
        comp.setRelease(200.0f);
        comp.setCharacter(chars[k]);

        auto trace = compressorGrTrace(comp, fs, 0.05, 0.0);
        grAt1ms[k] = -trace[static_cast<size_t>(fs * 0.001)];
    }

    EXPECT_GT(grAt1ms[0], 3.0f); // FET: clamped to <= 0.8 ms, nearly settled
    EXPECT_LT(grAt1ms[1], 2.0f); // Clean honours the 10 ms knob
}

DSPARK_TEST(Compressor_fet_detects_in_feedback)
{
    // The FET character must run feedback detection like the 1176 even when
    // the user set FeedForward. Since P3 the feedback static curve lands on
    // the panel values (depth no longer discriminates the topology), so the
    // probe is the feedback-defining behaviour instead: the detector is wired
    // to the output, so a hot external sidechain key must be ignored, while
    // the same key ducks a feed-forward compressor hard.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 1.0);
    const int tail = static_cast<int>(fs * 0.25);
    float gainFor[2] = { 0.0f, 0.0f };
    const Compressor<float>::Character chars[2] = {
        Compressor<float>::Character::FET, Compressor<float>::Character::Clean };

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);  // main programme at -30 dBFS (below threshold)
        generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), 0.0316f);
        auto key = makeBuffer(1, n); // hot external key at 0 dBFS
        generateSine(key.ch(0), n, 1000.0f, static_cast<float>(fs), 1.0f);
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(0.1f);
        comp.setRelease(100.0f);
        comp.setTopology(Compressor<float>::Topology::FeedForward);
        comp.setCharacter(chars[k]);
        comp.processBlock(tb.view(), key.view());

        gainFor[k] = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    }

    EXPECT_NEAR(gainFor[0], 0.0f, 0.5f); // FET: the feedback loop ignores the key
    EXPECT_LT(gainFor[1], -12.0f);       // feed-forward Clean ducks on the key
}

DSPARK_TEST(Compressor_varimu_has_knee_floor)
{
    // A variable-mu stage cannot form a hard corner: with knee set to 0 and
    // the level exactly at threshold, Varimu must still show the soft-knee
    // reduction of its 10 dB physical floor (~0.94 dB); Clean must show none.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);
    float gainAt[2] = { 0.0f, 0.0f };
    const Compressor<float>::Character chars[2] = {
        Compressor<float>::Character::Varimu, Compressor<float>::Character::Clean };

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), 0.1f); // -20 dBFS
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(0.05f);
        comp.setRelease(800.0f);
        comp.setKnee(0.0f);
        comp.setCharacter(chars[k]);
        comp.processBlock(tb.view());

        gainAt[k] = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    }

    EXPECT_NEAR(gainAt[0], -0.94f, 0.4f);
    EXPECT_NEAR(gainAt[1], 0.0f, 0.15f);
}

DSPARK_TEST(Compressor_upward_taper_follows_sustained_level)
{
    // At -70 dBFS (halfway down the guard fade) the boost must be about half
    // of the static curve: guard gates on the peak-held sustained level, so
    // waveform zero crossings must not punch holes in the boost.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);

    auto tb = makeBuffer(1, n);
    generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), 3.16e-4f); // -70 dBFS
    const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(spec(fs, n, 1));
    comp.setThreshold(-20.0f);
    comp.setRatio(2.0f);
    comp.setAttack(5.0f);
    comp.setRelease(100.0f);
    comp.setMode(Compressor<float>::Mode::Upward);
    comp.processBlock(tb.view());

    const float gain = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    EXPECT_NEAR(gain, 12.5f, 2.0f); // static 25 dB x 0.5 guard
}

// ============================================================================
// Compressor detector alignment and static makeup (2026-07 P2)
// ============================================================================

DSPARK_TEST(Compressor_hilbert_detector_is_latency_compensated)
{
    // The Hilbert FIR detects ~95 samples late; the audio path is delayed by
    // the same amount, so a hot step must leave already gain-reduced instead
    // of escaping unprocessed for 2 ms (the pre-P2 behaviour).
    const double fs = 48000.0;
    const int pre = static_cast<int>(fs * 0.1);
    const int n = pre + static_cast<int>(fs * 0.3);

    auto tb = makeBuffer(1, n);
    tb.fillSilence();
    generateSine(tb.ch(0) + pre, n - pre, 1000.0f, static_cast<float>(fs), 1.0f);

    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(spec(fs, 64, 1));
    comp.setThreshold(-20.0f);
    comp.setRatio(4.0f);
    comp.setAttack(0.1f);
    comp.setRelease(200.0f);
    comp.setDetector(Compressor<float>::DetectorType::Hilbert);

    for (int pos = 0; pos < n; pos += 64)
    {
        float* chans[1] = { tb.ch(0) + pos };
        AudioBufferView<float> view(chans, 1, std::min(64, n - pos));
        comp.processBlock(view);
    }

    EXPECT_EQ(comp.getLatency(), 95); // Hilbert<float>::getLatencySamples()
    const float escape = measurePeak(tb.ch(0) + pre, 600);
    EXPECT_LT(escape, 0.35f); // uncompensated, the full 1.0 step leaked through
}

DSPARK_TEST(Compressor_getLatency_matches_active_configuration)
{
    // Latency must reflect lookahead + Hilbert alignment immediately after a
    // setter (hosts re-read it before the next block), and feedback operation
    // (FeedBack topology or the FET character) must always report 0.
    const double fs = 48000.0;
    Compressor<float> comp;
    comp.prepare(spec(fs, 64, 1));

    EXPECT_EQ(comp.getLatency(), 0);
    comp.setLookahead(5.0f);
    EXPECT_EQ(comp.getLatency(), 240);
    comp.setDetector(Compressor<float>::DetectorType::Hilbert);
    EXPECT_EQ(comp.getLatency(), 335);
    comp.setTopology(Compressor<float>::Topology::FeedBack);
    EXPECT_EQ(comp.getLatency(), 0);
    comp.setTopology(Compressor<float>::Topology::FeedForward);
    comp.setDetector(Compressor<float>::DetectorType::Peak);
    comp.setCharacter(Compressor<float>::Character::FET);
    EXPECT_EQ(comp.getLatency(), 0);
}

DSPARK_TEST(Compressor_static_automakeup_is_program_independent)
{
    // Static mode adds half the full-scale static gain reduction as a fixed
    // offset: thr -20, ratio 4 -> +7.5 dB regardless of the programme level.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);
    const float levels[2] = { 0.5f /* -6 dBFS */, 0.158f /* -16 dBFS */ };
    const float grTheory[2] = { -10.5f, -3.0f };

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), levels[k]);
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(0.05f);
        comp.setRelease(800.0f);
        comp.setAutoMakeup(Compressor<float>::AutoMakeupMode::Static);
        comp.processBlock(tb.view());

        const float net = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
        EXPECT_NEAR(net - grTheory[k], 7.5f, 0.4f); // implied makeup, both levels
    }
}

// ============================================================================
// Compressor feedback calibration + FET colour (2026-07 P3)
// ============================================================================

namespace {

// Single-bin DFT magnitude over a buffer (double accumulation).
inline double compToneMagAt(const float* x, int n, double freq, double fs)
{
    double re = 0.0, im = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double ph = 2.0 * 3.14159265358979 * freq * static_cast<double>(i) / fs;
        re += static_cast<double>(x[i]) * std::cos(ph);
        im += static_cast<double>(x[i]) * std::sin(ph);
    }
    return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(n);
}

} // namespace

DSPARK_TEST(Compressor_feedback_honours_panel_ratio)
{
    // A raw feed-forward law inside the feedback loop can never compress past
    // 2:1 observed (the pre-P3 loop settled an 8:1 request at 1.87:1). The
    // element law is now the closed-form inverse of the user's curve, so the
    // settled feedback GR must land on it: thr -20, r8, 0 dBFS input
    // -> 20 * (1 - 1/8) = 17.5 dB.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.5);
    const int tail = static_cast<int>(fs * 0.5);

    auto tb = makeBuffer(1, n);
    generateSine(tb.ch(0), n, 997.0f, static_cast<float>(fs), 1.0f);
    const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(spec(fs, n, 1));
    comp.setThreshold(-20.0f);
    comp.setRatio(8.0f);
    comp.setAttack(0.5f);
    comp.setRelease(100.0f);
    comp.setTopology(Compressor<float>::Topology::FeedBack);
    comp.processBlock(tb.view());

    const float gain = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    EXPECT_NEAR(gain, -17.5f, 0.8f); // the pre-P3 loop sat at -6.4 dB here
}

DSPARK_TEST(Compressor_fet_static_curve_matches_panel)
{
    // The 1176's panel ratios describe the observed curve (UA manual: 4/8
    // compress, 12/20 limit). Settled GR with the level 20 dB over threshold
    // must land on 20 * (1 - 1/R): 15 dB at 4:1, 19 dB at 20:1 (a sine's
    // rectifier duty reads a few tenths shallow; pre-P3 sat at 8.6/9.4 dB).
    const double fs = 48000.0;
    const float ratios[2] = { 4.0f, 20.0f };
    const float theory[2] = { -15.0f, -19.0f };

    for (int k = 0; k < 2; ++k)
    {
        const int n = static_cast<int>(fs * 2.5);
        const int tail = static_cast<int>(fs * 0.5);
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 997.0f, static_cast<float>(fs), 1.0f);
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(ratios[k]);
        comp.setAttack(0.2f);
        comp.setRelease(200.0f);
        comp.setCharacter(Compressor<float>::Character::FET);
        comp.processBlock(tb.view());

        const float gain = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
        EXPECT_NEAR(gain, theory[k], 1.2f);
    }
}

DSPARK_TEST(Compressor_fet_fast_attack_is_stable_at_high_ratio)
{
    // 20 us attack at 20:1 puts a loop gain of 19 against a one-sample
    // ballistics step: the explicit iteration rings there; the semi-implicit
    // solve must stay clean AND reach hardware speed. Constant-level drive
    // (square wave) isolates the loop from the rectifier duty.
    const double fs = 48000.0;
    const int nPre = static_cast<int>(fs * 0.2); // smoother settling (silence)
    const int n = static_cast<int>(fs * 0.5);

    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(spec(fs, 64, 1));
    comp.setThreshold(-20.0f);
    comp.setRatio(20.0f);
    comp.setAttack(0.02f);
    comp.setRelease(50.0f);
    comp.setCharacter(Compressor<float>::Character::FET);

    std::vector<float> zeros(static_cast<size_t>(nPre), 0.0f);
    for (int pos = 0; pos < nPre; pos += 64)
    {
        float* chans[1] = { zeros.data() + pos };
        AudioBufferView<float> v(chans, 1, std::min(64, nPre - pos));
        comp.processBlock(v);
    }

    const float inc = dspark::twoPi<float> * 997.0f / static_cast<float>(fs);
    std::vector<float> gr(static_cast<size_t>(n));
    bool finite = true;
    for (int i = 0; i < n; ++i)
    {
        float x = std::sin(inc * static_cast<float>(i)) >= 0.0f ? 1.0f : -1.0f;
        float* chans[1] = { &x };
        AudioBufferView<float> v(chans, 1, 1);
        comp.processBlock(v);
        gr[static_cast<size_t>(i)] = comp.getGainReductionDb();
        if (!std::isfinite(x)) finite = false;
    }

    float settled = 0.0f, mn = 0.0f, mx = -100.0f;
    for (int i = n - n / 5; i < n; ++i)
    {
        const float v = gr[static_cast<size_t>(i)];
        settled += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    settled /= static_cast<float>(n / 5);

    EXPECT_TRUE(finite);
    EXPECT_NEAR(settled, -19.4f, 1.0f);  // panel curve (square reads the full peak)
    EXPECT_LT(mx - mn, 1.0f);            // no loop ringing at steady state
    int t63 = n;
    for (int i = 0; i < n; ++i)
        if (gr[static_cast<size_t>(i)] <= 0.632f * settled) { t63 = i; break; }
    EXPECT_LT(t63, 4);                   // 20 us observed: settles within samples
}

DSPARK_TEST(Compressor_opto_has_knee_floor)
{
    // A photocell's resistance curve is as gradual as a remote-cutoff tube's
    // transfer: with knee 0 and the level exactly at threshold, Opto must
    // show soft-knee reduction from its 10 dB floor while Clean shows none.
    // The 10 ms attack floor averages the rectifier ripple, so the settled
    // depth reads shallower than the instantaneous-theory 0.94 dB.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);
    float gainAt[2] = { 0.0f, 0.0f };
    const Compressor<float>::Character chars[2] = {
        Compressor<float>::Character::Opto, Compressor<float>::Character::Clean };

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 1000.0f, static_cast<float>(fs), 0.1f); // -20 dBFS
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(0.05f);
        comp.setRelease(800.0f);
        comp.setKnee(0.0f);
        comp.setCharacter(chars[k]);
        comp.processBlock(tb.view());

        gainAt[k] = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
    }

    EXPECT_GT(-gainAt[0], 0.25f);  // Opto: knee floor engages at the threshold
    EXPECT_LT(-gainAt[0], 1.3f);   // but stays a knee, not a jump
    EXPECT_LT(-gainAt[1], 0.15f);  // Clean honours the hard corner
}

DSPARK_TEST(Compressor_fet_color_meets_1176_thd_spec)
{
    // setCharacterColor(1) adds the FET's 2nd-order channel modulation,
    // calibrated to the published 1176 figure: < 0.5% THD while limiting
    // (UA manual, 50 Hz-15 kHz; measured here at -6 dBFS programme level).
    // Colour 0 must stay clean and the AC coupling must keep DC out.
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 3.0);
    const int tail = static_cast<int>(fs);
    double thd[2] = { 0.0, 0.0 };
    double dcWorst = 0.0;

    for (int k = 0; k < 2; ++k)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 997.0f, static_cast<float>(fs), 1.0f);

        Compressor<float> comp;
        comp.setAutoMakeup(false);
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-6.545f); // 12:1 limiting lands the output at -6 dBFS
        comp.setRatio(12.0f);
        comp.setAttack(0.8f);
        comp.setRelease(1100.0f);
        comp.setCharacter(Compressor<float>::Character::FET);
        comp.setCharacterColor(k == 1 ? 1.0f : 0.0f);
        comp.processBlock(tb.view());

        const float* x = tb.ch(0) + (n - tail);
        const double h1 = compToneMagAt(x, tail, 997.0, fs);
        double hs = 0.0;
        for (int h = 2; h <= 5; ++h)
        {
            const double m = compToneMagAt(x, tail, 997.0 * h, fs);
            hs += m * m;
        }
        thd[k] = std::sqrt(hs) / h1;

        double dc = 0.0;
        for (int i = 0; i < tail; ++i) dc += static_cast<double>(x[i]);
        dcWorst = std::max(dcWorst, std::abs(dc / tail));
    }

    EXPECT_LT(thd[0], 5e-4);    // colour off: clean gain riding only (~0.009%)
    EXPECT_GT(thd[1], 2.5e-3);  // colour on: the signature is really there...
    EXPECT_LT(thd[1], 5e-3);    // ...and within the 1176 spec (0.42% measured)
    EXPECT_LT(dcWorst, 1e-4);   // squared term stays AC-coupled
}

DSPARK_TEST(Compressor_invalid_inputs_are_ignored)
{
    // Every numeric setter used to pass NaN through max/clamp into the time
    // constants (measured: 4096/4096 non-finite samples, permanent), -Inf
    // slipped past the smoother's NaN-only guard (muted output), a negative
    // HPF cutoff made exp() exceed 1 (runaway filter), lookahead NaN made
    // getLatency() report INT_MIN to the host, an invalid prepare() crashed
    // on the UB int cast, wild enums fell through every switch case, and a
    // wild processSample channel was an out-of-bounds write (access
    // violation measured). All inert now: bit-identical to a clean twin.
    using Comp = Compressor<float>;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Comp subject, twin;
    auto setup = [](Comp& c) {
        c.prepare(spec(48000.0, 512, 2));
        c.setThreshold(-20.0f); c.setRatio(4.0f);
        c.setAttack(5.0f); c.setRelease(100.0f);
        c.setAutoMakeup(false);
        c.setSidechainHPF(true, 120.0f);
    };
    setup(subject); setup(twin);

    subject.setThreshold(nan);  subject.setThreshold(-inf);
    subject.setRatio(nan);      subject.setAttack(nan);
    subject.setRelease(nan);    subject.setKnee(nan);
    subject.setMakeupGain(nan); subject.setStereoLink(nan);
    subject.setMix(nan);        subject.setLookahead(nan);
    subject.setHoldTime(nan);   subject.setRange(nan);
    subject.setCharacterColor(nan);
    subject.setSidechainHPF(true, nan);
    subject.setSidechainHPF(true, -500.0f);
    subject.setRmsWindow(nan);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    subject.prepare(0.0);

    // Wild enums clamp to the last member and the getters stay honest.
    subject.setCharacter(static_cast<Comp::Character>(99));
    EXPECT_EQ(static_cast<int>(subject.getCharacter()), 3);
    subject.setCharacter(Comp::Character::Clean);
    subject.setDetector(static_cast<Comp::DetectorType>(77));
    EXPECT_EQ(static_cast<int>(subject.getDetector()), 4);
    subject.setDetector(Comp::DetectorType::Peak);

    EXPECT_EQ(subject.getLatency(), 0); // NaN lookahead never reached the host

    // Wild channels: exact pass-through, no state touched.
    EXPECT_EQ(subject.processSample(0.33f, 99), 0.33f);
    EXPECT_EQ(subject.processSample(0.5f, -1), 0.5f);

    auto ta = makeBuffer(2, 4096);
    auto tb = makeBuffer(2, 4096);
    for (int ch = 0; ch < 2; ++ch)
    {
        generateSine(ta.ch(ch), 4096, ch == 0 ? 440.0f : 620.0f, 48000.0f, 0.8f);
        generateSine(tb.ch(ch), 4096, ch == 0 ? 440.0f : 620.0f, 48000.0f, 0.8f);
    }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 4096);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
}

DSPARK_TEST(Compressor_rms_window_is_clean_after_reprepare)
{
    // The RMS window used to be clamped against capacity() instead of the
    // live size(): after a re-prepare to a lower rate the vector keeps its
    // taller capacity, and a wide window read stale squares from the dead
    // tail of the previous stream (measured: -9.2 dB of gain reduction on
    // pure silence). The window now clamps to the live element count.
    using Comp = Compressor<float>;
    Comp comp;
    comp.prepare(spec(96000.0, 512, 1));
    comp.setThreshold(-20.0f); comp.setRatio(4.0f);
    comp.setAttack(1.0f); comp.setRelease(50.0f);
    comp.setAutoMakeup(false);
    comp.setDetector(Comp::DetectorType::Rms);
    comp.setRmsWindow(500.0f);

    auto hot = makeBuffer(1, 48000); // hot stream fills the tall 96k buffer
    for (int i = 0; i < 48000; ++i) hot.ch(0)[i] = 1.0f;
    comp.processBlock(hot.view());

    comp.prepare(spec(44100.0, 512, 1)); // shrinks live size, keeps capacity
    comp.setRmsWindow(600.0f);           // beyond the 500 ms contract: clamps

    auto silence = makeBuffer(1, 8192);
    for (int i = 0; i < 8192; ++i) silence.ch(0)[i] = 0.0f;
    comp.processBlock(silence.view());

    // No ghost of the previous stream may drive the detector.
    EXPECT_NEAR(comp.getGainReductionDb(), 0.0f, 1e-3f);
    EXPECT_SILENT(silence.ch(0), 8192, 1e-8f);
}

// ============================================================================
// AutoGain
// ============================================================================

DSPARK_TEST(AutoGain_compensates_boost)
{
    auto s = spec(48000.0, 4096, 1);
    AutoGain<float> ag;
    ag.prepare(s);
    ag.setSmoothingTime(0.01f); // Near-instant smoothing for test convergence

    // Run multiple blocks so smoothing converges
    for (int block = 0; block < 20; ++block)
    {
        auto tb = makeBuffer(1, 4096);
        generateSine(tb.ch(0), 4096, 440.0f, 48000.0f, 0.5f);

        ag.pushReference(tb.view());

        // Simulate +6 dB boost
        for (int i = 0; i < 4096; ++i)
            tb.ch(0)[i] *= 2.0f;

        ag.compensate(tb.view());
    }

    // After convergence, compensation should be close to -6 dB
    float compDb = ag.getCompensationDb();
    EXPECT_NEAR(compDb, -6.0f, 1.0f);
}

DSPARK_TEST(AutoGain_silence_safe)
{
    auto s = spec(48000.0, 512, 1);
    AutoGain<float> ag;
    ag.prepare(s);

    auto tb = makeBuffer(1, 512);
    tb.fillSilence();

    ag.pushReference(tb.view());
    ag.compensate(tb.view());

    // Should not explode with infinite gain
    EXPECT_NO_NAN(tb.ch(0), 512);
    EXPECT_SILENT(tb.ch(0), 512, 0.001f);
}

DSPARK_TEST(AutoGain_max_compensation_respected)
{
    auto s = spec(48000.0, 4096, 1);
    AutoGain<float> ag;
    ag.prepare(s);
    ag.setMaxCompensation(6.0f);

    // Reference at -6 dBFS
    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 440.0f, 48000.0f, 0.5f);
    ag.pushReference(tb.view());

    // Attenuate output to -40 dBFS (would need +34 dB compensation)
    for (int i = 0; i < 4096; ++i)
        tb.ch(0)[i] *= 0.01f;

    ag.compensate(tb.view());

    // Compensation should be clamped to +/-6 dB
    float compDb = std::abs(ag.getCompensationDb());
    EXPECT_TRUE(compDb <= 6.5f); // Allow small smoothing overshoot
}

DSPARK_TEST(AutoGain_invalid_inputs_are_ignored)
{
    // setSmoothingTime(NaN) poisoned compensationDb_ PERMANENTLY (measured
    // 5120 non-finite samples after restoring a valid time; only reset()
    // cleared it), setMaxCompensation(NaN) silently DISABLED the clamp
    // (clamp(x, -NaN, NaN) returns x: compensation converged to +40 dB with
    // a 12 dB limit), an invalid prepare() poisoned the smoothing alpha, and
    // an empty pushReference() view floored the reference through 0/0 ->
    // gainToDecibels(NaN) = -100, pinning -12 dB of spurious attenuation on
    // live audio. All inert now.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    auto s = spec(48000.0, 512, 1);

    // (a) NaN/Inf setters + invalid prepare: bit-identical to a clean twin.
    AutoGain<float> subject, twin;
    subject.prepare(s); twin.prepare(s);
    subject.setSmoothingTime(10.0f); twin.setSmoothingTime(10.0f);

    auto ta = makeBuffer(1, 512);
    auto tb = makeBuffer(1, 512);
    float maxDiff = 0.0f;
    int badCount = 0;
    for (int blk = 0; blk < 12; ++blk)
    {
        if (blk == 2)
        {
            subject.setSmoothingTime(nan); subject.setSmoothingTime(inf);
            subject.setMaxCompensation(nan); subject.setMaxCompensation(inf);
        }
        if (blk == 6)
        {
            subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 1));
            subject.prepare(spec(48000.0, 512, -3));
        }
        for (int i = 0; i < 512; ++i)
        {
            const float v = 0.5f * std::sin(6.2831853f * 440.0f * (blk * 512 + i) / 48000.0f);
            ta.ch(0)[i] = v; tb.ch(0)[i] = v;
        }
        subject.pushReference(ta.view());
        twin.pushReference(tb.view());
        for (int i = 0; i < 512; ++i) { ta.ch(0)[i] *= 2.0f; tb.ch(0)[i] *= 2.0f; }
        subject.compensate(ta.view());
        twin.compensate(tb.view());
        for (int i = 0; i < 512; ++i)
        {
            if (!std::isfinite(ta.ch(0)[i])) ++badCount;
            maxDiff = std::max(maxDiff, std::abs(ta.ch(0)[i] - tb.ch(0)[i]));
        }
    }
    EXPECT_EQ(badCount, 0);       // old build: permanent NaN storm
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
    EXPECT_NEAR(subject.getMaxCompensation(), 12.0f, 1e-6f);
    EXPECT_NEAR(subject.getSmoothingTime(), 10.0f, 1e-6f);

    // (b) The clamp survives a NaN limit (old build converged to +40 dB).
    AutoGain<float> cl;
    cl.prepare(spec(48000.0, 4096, 1));
    cl.setSmoothingTime(5.0f);
    cl.setMaxCompensation(nan); // ignored: limit stays at the default 12
    auto big = makeBuffer(1, 4096);
    for (int blk = 0; blk < 30; ++blk)
    {
        generateSine(big.ch(0), 4096, 440.0f, 48000.0f, 0.5f);
        cl.pushReference(big.view());
        for (int i = 0; i < 4096; ++i) big.ch(0)[i] *= 0.01f; // -40 dB drop
        cl.compensate(big.view());
    }
    EXPECT_TRUE(cl.getCompensationDb() <= 12.5f); // old: 40.0

    // (c) An empty reference view keeps the previous reference (old build
    // floored it to -100 dB and pinned -12 dB on live audio).
    AutoGain<float> ep;
    ep.prepare(spec(48000.0, 4096, 1));
    ep.setSmoothingTime(5.0f);
    for (int blk = 0; blk < 20; ++blk)
    {
        generateSine(big.ch(0), 4096, 440.0f, 48000.0f, 0.5f);
        ep.pushReference(big.view());
        for (int i = 0; i < 4096; ++i) big.ch(0)[i] *= 2.0f;
        ep.compensate(big.view());
    }
    const float before = ep.getCompensationDb();
    float* nullPtr = nullptr;
    ep.pushReference(AudioBufferView<float>(&nullPtr, 1, 0));
    for (int blk = 0; blk < 10; ++blk)
    {
        generateSine(big.ch(0), 4096, 440.0f, 48000.0f, 1.0f);
        ep.compensate(big.view());
    }
    EXPECT_NEAR(ep.getCompensationDb(), before, 1.0f); // old: drifts to -12
}
