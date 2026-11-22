// DSPark Tests - Effects Dynamics
// Compressor, Limiter, NoiseGate, Gain, AutoGain, and the dynamics audit
// regression cases (DynamicEQ, DeEsser, Expander, TransientDesigner,
// CrossoverFilter) at the end of the file.

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Effects/Compressor.h"
#include "../Effects/Limiter.h"
#include "../Effects/NoiseGate.h"
#include "../Effects/Gain.h"
#include "../Effects/AutoGain.h"
#include "../Effects/CrossoverFilter.h"
#include "../Effects/DeEsser.h"
#include "../Effects/DynamicEQ.h"
#include "../Effects/Expander.h"
#include "../Effects/TransientDesigner.h"
#include "../Core/FFT.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <initializer_list>
#include <memory>
#include <utility>
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

// A ramp time of zero is a jump: the new gain is in force from the first
// sample of the next block. And prepare(spec) keeps the ramp time the caller
// set, so re-preparing a host's Gain at a new rate does not quietly restore
// the 10 ms default under it.
DSPARK_TEST(Gain_zero_ramp_is_instant_and_prepare_keeps_the_ramp)
{
    auto stepResponse = [](Gain<float>& g) {
        g.setGainDb(-6.0206f);
        auto tb = makeMonoBuffer(512);
        generateDC(tb.ch(0), 512, 1.0f);
        g.processBlock(tb.view());
        return std::vector<float>(tb.ch(0), tb.ch(0) + 512);
    };

    Gain<float> jump;
    jump.prepare(48000.0);
    jump.setGainDb(0.0f);
    jump.skipRamp();
    jump.setRampTime(0.0);
    const auto y = stepResponse(jump);
    EXPECT_NEAR(y[0], 0.5f, 1e-4f);
    EXPECT_NEAR(y[511], 0.5f, 1e-4f);

    // 50 ms set before prepare(spec): after one 512-sample block (10.7 ms)
    // the gain is still far from its target, as a 50 ms ramp must be and a
    // 10 ms one would not be.
    Gain<float> slow;
    slow.setRampTime(50.0);
    slow.prepare(AudioSpec{ 48000.0, 512, 1 });
    slow.setGainDb(0.0f);
    slow.skipRamp();
    const auto z = stepResponse(slow);
    EXPECT_GT(z[511], 0.6f);

    Gain<float> ref;
    ref.prepare(AudioSpec{ 48000.0, 512, 1 });   // default 10 ms
    ref.setGainDb(0.0f);
    ref.skipRamp();
    const auto r = stepResponse(ref);
    EXPECT_LT(r[511], z[511]);
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
    // bit-identical for mono, with every feature branch enabled, in both hold
    // modes (the true-peak estimator runs the same block FIRs one sample at a
    // time on the per-sample path).
  for (bool everyPeak : { false, true })
  {
    Limiter<float> lb, ls;
    auto setup = [everyPeak](Limiter<float>& l) {
        // Ceiling before prepare(): the smoother starts settled at -6 dB, so
        // the brickwall bound below holds from sample 0 (no 30 ms glide).
        l.setCeiling(-6.0f);
        if (everyPeak) l.setHoldMode(Limiter<float>::HoldMode::EveryPeak);
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
    // Uneven block sizes, so processBlock()'s internal chunking sees full,
    // partial and single-sample chunks.
    const int sizes[] = { 512, 37, 300, 1, 129, 700 };
    for (int off = 0, k = 0; off < n; ++k)
    {
        const int len = std::min(sizes[k % 6], n - off);
        float* p[1] = { blockOut.ch(0) + off };
        lb.processBlock(AudioBufferView<float>(p, 1, len));
        off += len;
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

namespace {

// Runs a mono -1 dB Limiter<double> one sample per block and counts the
// samples whose delayed input times the applied gain exceeds the ceiling:
// the samples the final clamp has to hard-clip because the gain computer
// did not turn them down in time.
int limiterClampHits(const std::vector<double>& x, double releaseMs, bool truePeak)
{
    Limiter<double> lim;
    lim.setCeiling(-1.0);
    lim.prepare(48000.0, 1);
    lim.setRelease(releaseMs);
    lim.setTruePeak(truePeak);
    const double ceiling = decibelsToGain(-1.0);
    const size_t latency = static_cast<size_t>(lim.getLatency());
    int hits = 0;
    for (size_t n = 0; n < x.size(); ++n)
    {
        double s = x[n];
        double* p[1] = { &s };
        lim.processBlock(AudioBufferView<double>(p, 1, 1));
        const double gain = decibelsToGain(lim.getGainReductionDb());
        if (n >= latency && std::abs(x[n - latency]) * gain > ceiling * (1.0 + 1e-9))
            ++hits;
    }
    return hits;
}

} // namespace

DSPARK_TEST(Limiter_close_peaks_with_fast_release_are_never_hard_clipped)
{
    // The old gain computer held a peak for the lookahead and then released,
    // so a second peak inside the release met a gain that had already
    // recovered (measured: 1 ms release, a peak reached 1.64x the ceiling
    // before the clamp flattened it). The sliding minimum plus the
    // S-curve ramp now turns every peak down before it reaches the output.
    std::vector<double> x(4000);
    for (size_t n = 0; n < x.size(); ++n)
        x[n] = 0.5 * std::sin(2.0 * 3.14159265358979 * 1000.0 * static_cast<double>(n) / 48000.0);
    x[1000] = 2.0;
    x[1090] = 1.6;

    for (double releaseMs : { 1.0, 5.0, 50.0 })
        for (bool truePeak : { false, true })
            EXPECT_EQ(limiterClampHits(x, releaseMs, truePeak), 0);
}

// Without a hold the gain recovers between the crests of a low tone and dips
// again at the next one: a 50 Hz sine driven 6 dB over the ceiling measured
// -56.8 dB THD+N (5 ms lookahead, 50 ms release). The default 10 ms hold
// keeps the reduction across the crests: -149.6 dB, and 40 Hz -149.7 dB.
DSPARK_TEST(Limiter_hold_keeps_low_tones_clean)
{
    auto thdn = [](double hz, bool hold) {
        Limiter<double> lim;
        lim.prepare(48000.0, 1, 5.0);
        lim.setCeiling(-1.0);
        lim.setRelease(50.0);
        if (!hold) lim.setHold(0.0);
        const double amp = 2.0 * std::pow(10.0, -1.0 / 20.0);
        std::vector<double> y(48000 * 2);
        for (size_t n = 0; n < y.size(); ++n)
            y[n] = amp * std::sin(2.0 * 3.14159265358979 * hz * static_cast<double>(n) / 48000.0);
        double* ch[1] = { y.data() };
        lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(y.size())));
        // Least-squares fit of the tone over the second second; the rest is
        // distortion and noise.
        double cc = 0, ss = 0, cs = 0, xc = 0, xs = 0;
        for (size_t n = 48000; n < y.size(); ++n)
        {
            const double ph = 2.0 * 3.14159265358979 * hz * static_cast<double>(n) / 48000.0;
            const double c = std::cos(ph), s = std::sin(ph);
            cc += c * c; ss += s * s; cs += c * s; xc += y[n] * c; xs += y[n] * s;
        }
        const double det = cc * ss - cs * cs;
        const double a = (xc * ss - xs * cs) / det, b = (xs * cc - xc * cs) / det;
        double res = 0, sig = 0;
        for (size_t n = 48000; n < y.size(); ++n)
        {
            const double ph = 2.0 * 3.14159265358979 * hz * static_cast<double>(n) / 48000.0;
            const double fit = a * std::cos(ph) + b * std::sin(ph);
            res += (y[n] - fit) * (y[n] - fit);
            sig += fit * fit;
        }
        return 10.0 * std::log10(res / sig + 1e-300);
    };
    EXPECT_NEAR(Limiter<double>().getHold(), 10.0, 1e-12);
    EXPECT_LT(thdn(50.0, true), -120.0);
    EXPECT_LT(thdn(40.0, true), -120.0);
    EXPECT_GT(thdn(50.0, false), -70.0);   // control: no hold
}

DSPARK_TEST(Limiter_dense_material_is_never_hard_clipped)
{
    // Drum-like material: noise bursts every 3000 samples over a noise bed.
    // The old design left the clamp to hard-clip 195 samples at 1 ms release
    // (up to 1.20x the ceiling) and 17 at 50 ms; the gain computer now meets
    // the ceiling by itself, with and without true-peak detection.
    std::vector<double> x(48000);
    uint32_t state = 0x12345678u;
    auto noise = [&state] {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        return static_cast<double>(state) / 2147483648.0 - 1.0;
    };
    for (size_t n = 0; n < x.size(); ++n)
    {
        const double phase = static_cast<double>(n % 3000);
        const double env = std::exp(-phase / 400.0) * (1.5 + 0.8 * std::sin(static_cast<double>(n) * 0.0007));
        x[n] = 0.3 * noise() + 1.4 * env * noise();
    }

    for (double releaseMs : { 1.0, 5.0, 50.0 })
        for (bool truePeak : { false, true })
            EXPECT_EQ(limiterClampHits(x, releaseMs, truePeak), 0);
}

namespace {

// Independent true-peak meter for the Limiter tests: an 8x reconstruction
// with a Kaiser-windowed sinc of 128 taps per phase (beta 14, cut at
// Nyquist), in double. The first and last 256 samples are faded so the edges
// of the excerpt do not ring.
double reconstructionPeak(const std::vector<float>& y, size_t from, size_t to)
{
    constexpr int kHalf = 64, kOver = 8;
    const double pi = 3.14159265358979323846;
    const auto i0 = [](double x) {
        double s = 1.0, t = 1.0;
        for (int k = 1; k < 80; ++k) { t *= (x / (2.0 * k)) * (x / (2.0 * k)); s += t; }
        return s;
    };
    std::vector<double> h(static_cast<size_t>(kOver * 2 * kHalf));
    for (int p = 0; p < kOver; ++p)
        for (int k = 0; k < 2 * kHalf; ++k)
        {
            const double d = (k - kHalf + 1) - static_cast<double>(p) / kOver;
            const double s = std::abs(d) < 1e-12 ? 1.0 : std::sin(pi * d) / (pi * d);
            const double u = d / kHalf;
            const double w = std::abs(u) < 1.0 ? i0(14.0 * std::sqrt(1.0 - u * u)) / i0(14.0) : 0.0;
            h[static_cast<size_t>(p * 2 * kHalf + k)] = s * w;
        }
    const size_t len = to - from;
    std::vector<double> z(len + 4 * kHalf, 0.0);
    for (size_t i = 0; i < len; ++i)
    {
        double g = 1.0;
        constexpr size_t kFade = 256;
        if (i < kFade) g = 0.5 - 0.5 * std::cos(pi * (static_cast<double>(i) + 0.5) / kFade);
        if (len - 1 - i < kFade) g = std::min(g, 0.5 - 0.5 * std::cos(pi * (static_cast<double>(len - 1 - i) + 0.5) / kFade));
        z[i + 2 * kHalf] = y[from + i] * g;
    }
    double peak = 0.0;
    for (size_t i = 0; i + 2 * kHalf <= z.size() - 2 * kHalf; ++i)
        for (int p = 0; p < kOver; ++p)
        {
            double v = 0.0;
            const double* c = &h[static_cast<size_t>(p * 2 * kHalf)];
            for (int k = 0; k < 2 * kHalf; ++k) v += c[k] * z[i + kHalf + static_cast<size_t>(k)];
            peak = std::max(peak, std::abs(v));
        }
    return peak;
}

// Stereo float Limiter at its defaults with true-peak detection and a -1 dB
// ceiling; returns the left output.
std::vector<float> limitTruePeak(const std::vector<float>& x, float lookaheadMs = 2.0f)
{
    auto lim = std::make_unique<Limiter<float>>();
    lim->setCeiling(-1.0f);
    lim->setTruePeak(true);
    lim->setLookahead(lookaheadMs);
    lim->prepare(48000.0, 2);
    std::vector<float> l = x, r = x;
    for (size_t o = 0; o < x.size(); o += 512)
    {
        const int m = static_cast<int>(std::min<size_t>(512, x.size() - o));
        float* ch[2] = { l.data() + o, r.data() + o };
        lim->processBlock(AudioBufferView<float>(ch, 2, m));
    }
    return l;
}

} // namespace

DSPARK_TEST(Limiter_true_peak_follows_the_reconstruction)
{
    // The BS.1770 Annex 2 estimate the sidechain used alone has 12 taps per
    // phase and under-reads peaks built from content near the top of the
    // band: on an independent 8x reconstruction the limited output read
    // +0.77 dB over the ceiling on a hard-clipped 5 kHz tone and +1.26 dB on
    // differentiated white noise. With the 64-tap estimate and vertex
    // refinement: +0.13 and +0.27 dB. An Annex 2 meter still reads the
    // output at or below the ceiling.
    const size_t n = 48000;
    const double ceiling = std::pow(10.0, -1.0 / 20.0);
    std::vector<float> clipped(n), hfNoise(n);
    for (size_t i = 0; i < n; ++i)
        clipped[i] = static_cast<float>(std::clamp(8.0 * std::sin(2.0 * 3.14159265358979 * 5003.0 * static_cast<double>(i) / 48000.0), -1.0, 1.0));
    uint32_t state = 0x2468ace1u;
    double prev = 0.0, peakIn = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        const double w = static_cast<double>(state) / 2147483648.0 - 1.0;
        hfNoise[i] = static_cast<float>(w - prev);
        prev = w;
        peakIn = std::max(peakIn, std::abs(static_cast<double>(hfNoise[i])));
    }
    for (auto& v : hfNoise) v = static_cast<float>(v * 1.6 / peakIn); // about 5 dB over the ceiling

    const struct { const std::vector<float>* x; double maxOverDb; } cases[] = {
        { &clipped, 0.3 }, { &hfNoise, 0.45 },
    };
    for (const auto& c : cases)
    {
        const std::vector<float> y = limitTruePeak(*c.x);
        const double overDb = 20.0 * std::log10(reconstructionPeak(y, 4800, n) / ceiling);
        EXPECT_LT(overDb, c.maxOverDb);

        TruePeakDetector<float, 1> annex;
        double annexPeak = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double v = annex.processSample(y[i], 0);
            if (i > 4800) annexPeak = std::max(annexPeak, v);
        }
        EXPECT_LT(20.0 * std::log10(annexPeak / ceiling), 0.01);
    }
}

DSPARK_TEST(Limiter_true_peak_raises_short_lookahead)
{
    // A reading reaches 33 samples back, so true-peak mode needs at least 49
    // samples of lookahead (a 16-sample ramp); getLatency() reports it at
    // once. The sample-domain contract still holds at that setting.
    Limiter<float> lim;
    lim.prepare(48000.0, 2);
    lim.setLookahead(0.5f);
    EXPECT_EQ(lim.getLatency(), 24);
    lim.setTruePeak(true);
    EXPECT_EQ(lim.getLatency(), 49);
    lim.setTruePeak(false);
    EXPECT_EQ(lim.getLatency(), 24);

    std::vector<double> x(4000);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = 0.5 * std::sin(2.0 * 3.14159265358979 * 1000.0 * static_cast<double>(i) / 48000.0);
    x[1000] = 2.0;
    x[1090] = 1.6;
    Limiter<double> dl;
    dl.setCeiling(-1.0);
    dl.prepare(48000.0, 1);
    dl.setLookahead(0.5);
    dl.setTruePeak(true);
    const double ceil = decibelsToGain(-1.0);
    const size_t latency = static_cast<size_t>(dl.getLatency());
    EXPECT_EQ(latency, size_t(49));
    int hits = 0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        double s = x[i];
        double* p[1] = { &s };
        dl.processBlock(AudioBufferView<double>(p, 1, 1));
        const double gain = decibelsToGain(dl.getGainReductionDb());
        if (i >= latency && std::abs(x[i - latency]) * gain > ceil * (1.0 + 1e-9)) ++hits;
    }
    EXPECT_EQ(hits, 0);
}

DSPARK_TEST(Limiter_every_peak_hold_holds_each_peak)
{
    // Deepest mode restarts the hold only when a peak pushes the reduction
    // down, so a shallower peak 5 ms after a deeper one is released with the
    // deeper one. EveryPeak holds it for the full hold after it has passed.
    auto gainBeforeSecondHoldEnds = [](bool everyPeak) {
        Limiter<double> lim;
        lim.setCeiling(-1.0);
        lim.prepare(48000.0, 1);          // 2 ms lookahead = 96 samples
        lim.setHold(10.0);                // 480 samples
        lim.setRelease(5.0);
        if (everyPeak) lim.setHoldMode(Limiter<double>::HoldMode::EveryPeak);
        const size_t p1 = 1000, p2 = p1 + 240, probe = p2 + 96 + 480 - 20;
        double gain = 1.0;
        for (size_t i = 0; i <= probe; ++i)
        {
            double s = (i == p1) ? 2.0 : (i == p2 ? 1.4 : 0.0);
            double* p[1] = { &s };
            lim.processBlock(AudioBufferView<double>(p, 1, 1));
            gain = decibelsToGain(lim.getGainReductionDb());
        }
        return gain;
    };
    const double r2 = decibelsToGain(-1.0) / 1.4;
    EXPECT_LT(gainBeforeSecondHoldEnds(true), r2 + 1e-9);
    EXPECT_GT(gainBeforeSecondHoldEnds(false), r2 + 0.05);   // control: Deepest
    EXPECT_TRUE(Limiter<double>().getHoldMode() == Limiter<double>::HoldMode::Deepest);
}

DSPARK_TEST(Limiter_every_peak_hold_removes_intermodulation)
{
    // A 60 Hz + 7 kHz two-tone driven 6 dB over the ceiling: in Deepest mode
    // the gain rises and falls between crests of unequal height and
    // modulates the 7 kHz tone (sidebands -61.7 dBc at the defaults);
    // EveryPeak keeps the gain flat across them (-87.1 dBc).
    auto worstSidebandDbc = [](bool everyPeak) {
        Limiter<double> lim;
        lim.setCeiling(-6.0);
        lim.prepare(48000.0, 1);
        if (everyPeak) lim.setHoldMode(Limiter<double>::HoldMode::EveryPeak);
        const double amp = std::pow(10.0, 0.0 / 20.0);
        const size_t n = 96000;
        std::vector<double> y(n);
        for (size_t i = 0; i < n; ++i)
        {
            const double t = static_cast<double>(i) / 48000.0;
            y[i] = amp * (0.8 * std::sin(2.0 * 3.14159265358979 * 60.0 * t) + 0.2 * std::sin(2.0 * 3.14159265358979 * 7000.0 * t));
        }
        for (size_t o = 0; o < n; o += 512)
        {
            double* ch[1] = { y.data() + o };
            lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(std::min<size_t>(512, n - o))));
        }
        // Every product of the 50 ms modulation pattern falls on an exact bin
        // of the last second, so single-bin projections do not leak.
        auto amplitude = [&](double hz) {
            double c = 0.0, s = 0.0;
            for (size_t i = n - 48000; i < n; ++i)
            {
                const double ph = 2.0 * 3.14159265358979 * hz * static_cast<double>(i) / 48000.0;
                c += y[i] * std::cos(ph);
                s += y[i] * std::sin(ph);
            }
            return std::sqrt(c * c + s * s);
        };
        const double carrier = amplitude(7000.0);
        double worst = 0.0;
        for (int k = 1; k <= 6; ++k)
            worst = std::max({ worst, amplitude(7000.0 - 60.0 * k), amplitude(7000.0 + 60.0 * k) });
        return 20.0 * std::log10(worst / carrier);
    };
    EXPECT_LT(worstSidebandDbc(true), -80.0);
    EXPECT_GT(worstSidebandDbc(false), -70.0);   // control: Deepest
}

namespace {

/// Worst sideband of the 7 kHz tone of a 60 Hz + 7 kHz pair driven 6 dB over
/// a -6 dB ceiling, in dB re that tone, over the last of two seconds.
template <class T>
double limiterSidebandDbc(bool truePeak, double lookaheadMs, bool highPrecision = false)
{
    constexpr double kTau = 6.283185307179586;
    Limiter<T> lim;
    lim.setCeiling(T(-6));
    lim.prepare(48000.0, 1);
    if (lookaheadMs > 0.0) lim.setLookahead(static_cast<T>(lookaheadMs));
    lim.setTruePeak(truePeak);
    if (highPrecision) lim.setTruePeakPrecision(Limiter<T>::TruePeakPrecision::High);
    const size_t n = 96000;
    std::vector<T> y(n);
    for (size_t i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i) / 48000.0;
        y[i] = static_cast<T>(0.8 * std::sin(kTau * 60.0 * t) + 0.2 * std::sin(kTau * 7000.0 * t));
    }
    for (size_t o = 0; o < n; o += 512)
    {
        T* ch[1] = { y.data() + o };
        lim.processBlock(AudioBufferView<T>(ch, 1, static_cast<int>(std::min<size_t>(512, n - o))));
    }
    const auto amplitude = [&](double hz) {
        double c = 0.0, s = 0.0;
        for (size_t i = n - 48000; i < n; ++i)
        {
            const double ph = kTau * hz * static_cast<double>(i) / 48000.0;
            c += static_cast<double>(y[i]) * std::cos(ph);
            s += static_cast<double>(y[i]) * std::sin(ph);
        }
        return std::sqrt(c * c + s * s);
    };
    const double carrier = amplitude(7000.0);
    double worst = 0.0;
    for (int k = 1; k <= 6; ++k)
        worst = std::max({ worst, amplitude(7000.0 - 60.0 * k), amplitude(7000.0 + 60.0 * k) });
    return 20.0 * std::log10(worst / carrier);
}

/// Deterministic noise in [-1, 1).
struct LimiterTestNoise
{
    uint32_t state;
    double next()
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / 8388608.0 - 1.0;
    }
};

}   // namespace

// The sidebands a limiter leaves around a tone that rides a larger slow wave
// follow the length of its attack ramp. True-peak mode gives 33 samples of
// the lookahead to its reading, so at the 2 ms default the 60 Hz + 7 kHz pair
// reads -59.1 dBc (double -60.5) against -61.9 (-61.5) in sample-peak mode;
// 2.7 ms of lookahead gives the ramp its 96 samples back and reads -64.1 dBc
// (-65.2). Two gain envelopes multiplied (sample peaks on the long ramp, the
// true-peak excess on the short one) were measured instead: -60.1 dBc, with
// the output 1.9 to 10 dB quieter than the ceiling asks, because the second
// envelope cannot know how far the first one already is below the peak.
DSPARK_TEST(Limiter_intermodulation_follows_the_attack_ramp)
{
    for (const bool wide : { false, true })
    {
        const double sample = wide ? limiterSidebandDbc<double>(false, 0.0) : limiterSidebandDbc<float>(false, 0.0);
        const double truePeak = wide ? limiterSidebandDbc<double>(true, 0.0) : limiterSidebandDbc<float>(true, 0.0);
        const double longer = wide ? limiterSidebandDbc<double>(true, 2.7) : limiterSidebandDbc<float>(true, 2.7);
        EXPECT_LT(sample, -59.0);
        EXPECT_LT(truePeak, -56.5);
        EXPECT_LT(longer, -61.5);
        EXPECT_LT(longer, truePeak - 2.0);
    }
}

// The limited output read with TruePeakAnalyzer (an 8x grid from a 512-tap
// first stage, which also resolves content between 0.45 fs and Nyquist), in
// dB over a -1 dB ceiling at the defaults: a 48-partial mix driven 12 dB over
// the ceiling +0.004 (sample-peak mode +0.66), a hard-clipped 5 kHz tone
// +0.19 (+2.56), white noise +0.77 (+5.92) and its first difference +1.10
// (+4.24). What true-peak mode leaves is content above 0.45 fs, which a
// reading that answers within 33 samples cannot follow.
DSPARK_TEST(Limiter_true_peak_output_read_on_an_independent_grid)
{
    constexpr double kTau = 6.283185307179586;
    const size_t n = 192000;
    const double ceiling = std::pow(10.0, -1.0 / 20.0);
    const auto overDb = [&](std::vector<double> x, double driveDb, bool truePeak) {
        double m = 0.0;
        for (double v : x) m = std::max(m, std::abs(v));
        const double scale = ceiling * std::pow(10.0, driveDb / 20.0) / m;
        for (double& v : x) v *= scale;
        Limiter<double> lim;
        lim.setCeiling(-1.0);
        lim.prepare(48000.0, 1);
        lim.setTruePeak(truePeak);
        for (size_t o = 0; o < n; o += 512)
        {
            double* ch[1] = { x.data() + o };
            lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(std::min<size_t>(512, n - o))));
        }
        TruePeakAnalyzer<double> analyzer;
        const auto state = std::make_unique<TruePeakAnalyzer<double>::Channel>();
        double peak = 0.0;
        for (size_t o = 0; o < n; o += 128)
            peak = std::max(peak, analyzer.processMax(*state, x.data() + o,
                                                      static_cast<int>(std::min<size_t>(128, n - o)), peak));
        peak = std::max(peak, analyzer.getTailPeak(*state));
        return 20.0 * std::log10(peak / ceiling);
    };

    std::vector<double> dense(n, 0.0), clipped(n), white(n), bright(n);
    LimiterTestNoise phases { 0x1234u };
    for (int k = 0; k < 48; ++k)
    {
        const double hz = 40.0 * std::pow(18000.0 / 40.0, k / 47.0);
        const double phase = 0.5 * kTau * phases.next();
        const double level = 1.0 / std::sqrt(1.0 + k);
        for (size_t i = 0; i < n; ++i)
            dense[i] += level * std::sin(kTau * hz * static_cast<double>(i) / 48000.0 + phase);
    }
    LimiterTestNoise noise { 0x9876u };
    double previous = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        clipped[i] = std::clamp(4.0 * std::sin(kTau * 5000.0 * static_cast<double>(i) / 48000.0), -1.0, 1.0);
        white[i] = noise.next();
        bright[i] = white[i] - previous;
        previous = white[i];
    }

    EXPECT_LT(overDb(dense, 12.0, true), 0.05);
    EXPECT_LT(overDb(clipped, 6.0, true), 0.35);
    EXPECT_LT(overDb(white, 6.0, true), 1.0);
    EXPECT_LT(overDb(bright, 6.0, true), 1.35);
    // Sample-peak mode leaves the peaks between samples alone.
    EXPECT_GT(overDb(clipped, 6.0, false), 2.0);
    EXPECT_GT(overDb(white, 6.0, false), 4.0);
}

namespace {

struct LimiterPrecisionRun
{
    double overDb;      ///< Output true peak over the ceiling (TruePeakAnalyzer).
    double annexOverDb; ///< The same as a BS.1770 Annex 2 meter reads it.
    double rmsDb;       ///< Output level.
    int latency;
};

/// A mono double Limiter at the defaults and a -1 dB ceiling over x, scaled
/// to peak driveDb over the ceiling and followed by silence so the limiter
/// sees the programme's real end.
LimiterPrecisionRun limitAndRead(std::vector<double> x, double driveDb, bool truePeak, bool highPrecision)
{
    const double ceiling = std::pow(10.0, -1.0 / 20.0);
    double m = 0.0;
    for (double v : x) m = std::max(m, std::abs(v));
    const double scale = ceiling * std::pow(10.0, driveDb / 20.0) / m;
    for (double& v : x) v *= scale;
    x.resize(x.size() + 4096, 0.0);

    Limiter<double> lim;
    lim.setCeiling(-1.0);
    lim.prepare(48000.0, 1);
    lim.setTruePeak(truePeak);
    if (highPrecision) lim.setTruePeakPrecision(Limiter<double>::TruePeakPrecision::High);
    for (size_t o = 0; o < x.size(); o += 512)
    {
        double* ch[1] = { x.data() + o };
        lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(std::min<size_t>(512, x.size() - o))));
    }

    TruePeakAnalyzer<double> analyzer;
    const auto state = std::make_unique<TruePeakAnalyzer<double>::Channel>();
    TruePeakDetector<double, 1> annex;
    double peak = 0.0, annexPeak = 0.0, sum = 0.0;
    for (size_t o = 0; o < x.size(); o += 128)
        peak = std::max(peak, analyzer.processMax(*state, x.data() + o,
                                                  static_cast<int>(std::min<size_t>(128, x.size() - o)), peak));
    peak = std::max(peak, analyzer.getTailPeak(*state));
    for (double v : x)
    {
        annexPeak = std::max(annexPeak, annex.processSample(v, 0));
        sum += v * v;
    }
    return { 20.0 * std::log10(peak / ceiling), 20.0 * std::log10(annexPeak / ceiling),
             10.0 * std::log10(sum / static_cast<double>(x.size())), lim.getLatency() };
}

}   // namespace

// The standard true-peak reading answers within 33 samples and cannot follow
// content between 0.45 fs and Nyquist, whose crests are built by samples
// hundreds of positions away. TruePeakPrecision::High reads with
// TruePeakAnalyzer (a 512-tap first stage) and pays its 262 samples of reach
// in latency.
DSPARK_TEST(Limiter_high_precision_true_peak_follows_content_near_nyquist)
{
    constexpr double kTau = 6.283185307179586;
    const size_t n = 96000;
    std::vector<double> dense(n, 0.0), white(n), bright(n), bursts(n);
    LimiterTestNoise phases { 0x1234u };
    for (int k = 0; k < 48; ++k)
    {
        const double hz = 40.0 * std::pow(18000.0 / 40.0, k / 47.0);
        const double phase = 0.5 * kTau * phases.next();
        const double level = 1.0 / std::sqrt(1.0 + k);
        for (size_t i = 0; i < n; ++i)
            dense[i] += level * std::sin(kTau * hz * static_cast<double>(i) / 48000.0 + phase);
    }
    LimiterTestNoise noise { 0x9876u };
    double previous = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        white[i] = noise.next();
        bright[i] = white[i] - previous;
        previous = white[i];
        // Bursts of a 0.47 fs tone: 20 ms on, 20 ms off, abrupt edges.
        bursts[i] = ((i / 960) % 2 == 0) ? std::sin(kTau * 0.47 * static_cast<double>(i) + 0.3) : 0.0;
    }

    // Output true peak over the ceiling, Standard and High: white noise
    // +0.77 and +0.01 dB, its first difference +0.80 and +0.02, the bursts
    // +0.59 and +0.001, the dense mix (nothing above 0.45 fs) +0.004 and
    // 0.000 with High 0.05 dB louder: its readings wait for each other, so
    // none turns the gain down before its own sample.
    const struct { const std::vector<double>* x; double driveDb; double highBelow; double standardAbove; } cases[] = {
        { &white, 6.0, 0.10, 0.5 }, { &bright, 6.0, 0.10, 0.5 }, { &bursts, 6.0, 0.05, 0.35 }, { &dense, 12.0, 0.03, -1.0 },
    };
    for (const auto& c : cases)
    {
        const LimiterPrecisionRun standard = limitAndRead(*c.x, c.driveDb, true, false);
        const LimiterPrecisionRun high = limitAndRead(*c.x, c.driveDb, true, true);
        EXPECT_LT(high.overDb, c.highBelow);
        EXPECT_GT(standard.overDb, c.standardAbove);
        EXPECT_LT(high.annexOverDb, 0.02);          // an Annex 2 meter agrees
        EXPECT_EQ(high.latency, standard.latency + 262);
        if (c.standardAbove < 0.0)
            EXPECT_GT(high.rmsDb, standard.rmsDb - 0.02);
    }
}

// High precision adds the 262 samples its reading reaches back to the
// latency, so the lookahead stays all attack ramp as in sample-peak mode;
// below the ceiling the output is the input that many samples late; and the
// choice travels in the state.
DSPARK_TEST(Limiter_high_precision_true_peak_latency_ramp_and_state)
{
    using Lim = Limiter<double>;
    const auto high = Lim::TruePeakPrecision::High;
    {
        Lim lim;
        lim.prepare(48000.0, 2);
        EXPECT_TRUE(lim.getTruePeakPrecision() == Lim::TruePeakPrecision::Standard);
        EXPECT_EQ(lim.getLatency(), 96);
        lim.setTruePeakPrecision(high);                // acts with true peak on only
        EXPECT_EQ(lim.getLatency(), 96);
        lim.setTruePeak(true);
        EXPECT_EQ(lim.getLatency(), 96 + 262);
        lim.setLookahead(0.5);
        EXPECT_EQ(lim.getLatency(), 24 + 262);          // no 49-sample floor: the ramp is the lookahead
        lim.setTruePeakPrecision(Lim::TruePeakPrecision::Standard);
        EXPECT_EQ(lim.getLatency(), 49);
        lim.setTruePeakPrecision(high);
        lim.prepare(44100.0, 2);                        // survives a re-preparation
        EXPECT_TRUE(lim.getTruePeakPrecision() == high);
        EXPECT_EQ(lim.getLatency(), 22 + 262);
    }

    // Below the ceiling: the input, getLatency() samples late.
    {
        Lim lim;
        lim.setCeiling(-1.0);
        lim.prepare(48000.0, 1);
        lim.setTruePeak(true);
        lim.setTruePeakPrecision(high);
        const size_t latency = static_cast<size_t>(lim.getLatency());
        std::vector<double> x(8000);
        LimiterTestNoise noise { 0x77u };
        for (double& v : x) v = 0.25 * noise.next();
        std::vector<double> y = x;
        for (size_t o = 0; o < y.size(); o += 300)
        {
            double* ch[1] = { y.data() + o };
            lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(std::min<size_t>(300, y.size() - o))));
        }
        double worst = 0.0;
        for (size_t i = 0; i < y.size(); ++i)
            worst = std::max(worst, std::abs(y[i] - (i >= latency ? x[i - latency] : 0.0)));
        EXPECT_LT(worst, 1e-15);
    }

    // The sample-domain contract at the shortest lookahead: the gain applied
    // to every sample keeps it at or below the ceiling.
    {
        std::vector<double> x(6000);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 0.5 * std::sin(2.0 * 3.14159265358979 * 1000.0 * static_cast<double>(i) / 48000.0);
        x[1000] = 2.0;
        x[1090] = 1.6;
        x[1400] = -1.9;
        Lim lim;
        lim.setCeiling(-1.0);
        lim.prepare(48000.0, 1);
        lim.setLookahead(0.5);
        lim.setTruePeak(true);
        lim.setTruePeakPrecision(high);
        const double ceil = decibelsToGain(-1.0);
        const size_t latency = static_cast<size_t>(lim.getLatency());
        int hits = 0;
        for (size_t i = 0; i < x.size(); ++i)
        {
            double s = x[i];
            double* p[1] = { &s };
            lim.processBlock(AudioBufferView<double>(p, 1, 1));
            const double gain = decibelsToGain(lim.getGainReductionDb());
            if (i >= latency && std::abs(x[i - latency]) * gain > ceil * (1.0 + 1e-9)) ++hits;
        }
        EXPECT_EQ(hits, 0);
    }

    // The ramp is the whole lookahead, so the intermodulation is that of
    // sample-peak mode at the same setting: -63.0 dBc (double -58.9) at the
    // default against -61.9 (-61.5), -68.1 (-63.9) at 2.7 ms against -66.7
    // (-66.3). Which crest restarts the hold depends on the last bit of a
    // reading, hence the spread between float and double.
    for (const bool wide : { false, true })
    {
        const double atDefault = wide ? limiterSidebandDbc<double>(true, 0.0, true) : limiterSidebandDbc<float>(true, 0.0, true);
        const double longer = wide ? limiterSidebandDbc<double>(true, 2.7, true) : limiterSidebandDbc<float>(true, 2.7, true);
        EXPECT_LT(atDefault, -54.5);
        EXPECT_LT(longer, -59.5);
    }

    // The state carries the choice; a blob without it restores Standard.
    {
        Limiter<float> a;
        a.setTruePeak(true);
        a.setTruePeakPrecision(Limiter<float>::TruePeakPrecision::High);
        const auto blob = a.getState();
        StateReader r(blob.data(), blob.size());
        EXPECT_EQ(r.read("truePeakPrecision", int32_t(-1)), int32_t(1));
        Limiter<float> b;
        EXPECT_TRUE(b.setState(blob.data(), blob.size()));
        EXPECT_TRUE(b.isTruePeakEnabled());
        EXPECT_TRUE(b.getTruePeakPrecision() == Limiter<float>::TruePeakPrecision::High);

        Limiter<float> plain;
        const auto old = plain.getState();
        EXPECT_TRUE(b.setState(old.data(), old.size()));
        EXPECT_TRUE(b.getTruePeakPrecision() == Limiter<float>::TruePeakPrecision::Standard);
    }
}

// In high precision the per-sample path is bit-identical to the block path
// for mono (the analyzer runs the same block filters one sample at a time),
// and the reading can be switched during playback: the histories are rebuilt
// from the delay lines, so the output after a switch is that of a limiter
// that had it on all along.
DSPARK_TEST(Limiter_high_precision_true_peak_per_sample_path_and_live_switch)
{
    for (const bool everyPeak : { false, true })
    {
        Limiter<float> lb, ls;
        for (Limiter<float>* l : { &lb, &ls })
        {
            l->setCeiling(-6.0f);
            if (everyPeak) l->setHoldMode(Limiter<float>::HoldMode::EveryPeak);
            l->prepare(48000.0, 1);
            l->setTruePeak(true);
            l->setTruePeakPrecision(Limiter<float>::TruePeakPrecision::High);
            l->setAdaptiveRelease(true);
            l->setSafetyClip(true);
        }
        const int n = 6000;
        std::vector<float> in(static_cast<size_t>(n));
        LimiterTestNoise noise { 0x99u };
        for (int i = 0; i < n; ++i)
            in[static_cast<size_t>(i)] = static_cast<float>(0.05 * std::sin(6.283185307179586 * 330.0 * i / 48000.0)
                                                            + (i % 997 == 500 ? 1.0 : 0.0) + 0.4 * noise.next());
        std::vector<float> blockOut = in;
        const int sizes[] = { 512, 37, 300, 1, 129, 700 };
        for (int off = 0, k = 0; off < n; ++k)
        {
            const int len = std::min(sizes[k % 6], n - off);
            float* p[1] = { blockOut.data() + off };
            lb.processBlock(AudioBufferView<float>(p, 1, len));
            off += len;
        }
        float maxDiff = 0.0f;
        for (int i = 0; i < n; ++i)
            maxDiff = std::max(maxDiff, std::abs(ls.processSample(in[static_cast<size_t>(i)], 0) - blockOut[static_cast<size_t>(i)]));
        EXPECT_TRUE(maxDiff == 0.0f);
        const float ceiling = decibelsToGain(-6.0f);
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(std::abs(blockOut[static_cast<size_t>(i)]) <= ceiling + 1e-6f);
    }

    // Switching during playback.
    {
        const size_t n = 96000, from = 140 * 512;
        const double ceiling = std::pow(10.0, -1.0 / 20.0);
        std::vector<double> x(n);
        LimiterTestNoise noise { 0x9876u };
        for (double& v : x) v = 2.0 * ceiling * noise.next();
        using Lim = Limiter<double>;
        const auto run = [&](bool switching) {
            Lim lim;
            lim.setCeiling(-1.0);
            lim.prepare(48000.0, 1);
            if (!switching)
            {
                lim.setTruePeak(true);
                lim.setTruePeakPrecision(Lim::TruePeakPrecision::High);
            }
            std::vector<double> y = x;
            int block = 0;
            for (size_t o = 0; o < n; o += 512, ++block)
            {
                if (switching)
                {
                    if (block == 20) lim.setTruePeak(true);
                    if (block == 50) lim.setTruePeakPrecision(Lim::TruePeakPrecision::High);
                    if (block == 90) lim.setTruePeakPrecision(Lim::TruePeakPrecision::Standard);
                    if (block == 110) { lim.setTruePeak(false); lim.setTruePeakPrecision(Lim::TruePeakPrecision::High); }
                    if (block == 130) lim.setTruePeak(true);
                }
                double* ch[1] = { y.data() + o };
                lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(std::min<size_t>(512, n - o))));
            }
            return y;
        };
        const auto peakFrom = [&](const std::vector<double>& y) {
            TruePeakAnalyzer<double> analyzer;
            const auto state = std::make_unique<TruePeakAnalyzer<double>::Channel>();
            double peak = 0.0;
            for (size_t o = from; o < n; o += 128)
                peak = std::max(peak, analyzer.processMax(*state, y.data() + o,
                                                          static_cast<int>(std::min<size_t>(128, n - o)), peak));
            return 20.0 * std::log10(peak / ceiling);
        };
        const std::vector<double> switched = run(true);
        const std::vector<double> always = run(false);
        bool finite = true;
        double worstSample = 0.0;
        for (double v : switched)
        {
            finite = finite && std::isfinite(v);
            worstSample = std::max(worstSample, std::abs(v));
        }
        EXPECT_TRUE(finite);
        EXPECT_LT(worstSample, ceiling * (1.0 + 1e-9));
        // +0.013 dB over the ceiling after the last switch, as with the
        // reading on from the start.
        EXPECT_LT(peakFrom(switched), 0.08);
        EXPECT_NEAR(peakFrom(switched), peakFrom(always), 0.03);
    }
}

// The gain a frame requires is ceiling / max(ceiling, peak). A ceiling of
// -100 dB or below is zero, and over silence that ratio is 0 / 0: the gain
// computer ignores it, the output stays silent and finite, and the limiter
// passes signal again once the ceiling comes back.
DSPARK_TEST(Limiter_zero_ceiling_over_silence_stays_finite)
{
    const auto check = []<class T>()
    {
        for (const bool truePeak : { false, true })
        {
            Limiter<T> lim;
            lim.setCeiling(T(-120));
            lim.prepare(48000.0, 1);
            lim.setTruePeak(truePeak);
            LimiterTestNoise noise { 0x31u };
            const auto process = [&](std::vector<T>& x) {
                for (size_t o = 0; o < x.size(); o += 256)
                {
                    T* ch[1] = { x.data() + o };
                    lim.processBlock(AudioBufferView<T>(ch, 1, static_cast<int>(std::min<size_t>(256, x.size() - o))));
                }
            };

            // Silence, a burst of noise, silence.
            std::vector<T> x(4096, T(0));
            for (size_t i = 1024; i < 2048; ++i) x[i] = static_cast<T>(noise.next());
            process(x);
            bool silent = true;
            for (const T v : x) silent = silent && (v == T(0));
            EXPECT_TRUE(silent);
            EXPECT_TRUE(std::isfinite(lim.getGainReductionDb()));

            lim.setCeiling(T(-6));
            std::vector<T> y(24000);
            for (T& v : y) v = static_cast<T>(0.25 * noise.next());
            process(y);
            T late = T(0);
            bool finite = true;
            for (size_t i = 0; i < y.size(); ++i)
            {
                finite = finite && std::isfinite(y[i]);
                if (i >= y.size() - 4800) late = std::max(late, std::abs(y[i]));
            }
            EXPECT_TRUE(finite);
            EXPECT_GT(late, T(0.2));
            EXPECT_LT(late, T(0.2501));
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
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
    for (int cfg = 0; cfg < 3; ++cfg)
    {
        NoiseGate<float> gb, gs;
        auto setup = [&](NoiseGate<float>& g) {
            g.prepare(48000.0);
            g.setThreshold(-40.0f); g.setRelease(20.0f);
            if (cfg == 1) { g.setHold(0.0f); g.setSidechainHPF(true, 1000.0); g.setAdaptiveHold(true); }
            else          { g.setHold(10.0f); }
            if (cfg == 2) g.setLookahead(2.0f); // the delay line rides along too
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

// Without lookahead the gate opens only as a transient arrives, so the attack
// ramp chops its first milliseconds (here the first millisecond of a burst
// comes out ~8 dB down with a 1 ms attack). With lookahead the audio is
// delayed, the gate is already open when the burst reaches the output, and
// getLatency() reports the delay for compensation.
DSPARK_TEST(NoiseGate_lookahead_keeps_the_transient_onset)
{
    auto onsetLossDb = [](float lookaheadMs, int& latency) {
        NoiseGate<float> g;
        g.prepare(48000.0, 1);
        g.setThreshold(-40.0f);
        g.setAttack(1.0f);
        g.setLookahead(lookaheadMs);
        latency = g.getLatency();
        const int n0 = 9600, n = 14400;
        std::vector<float> x(static_cast<size_t>(n), 0.0f);
        for (int i = n0; i < n; ++i)
            x[static_cast<size_t>(i)] = 0.5f * std::sin(6.2831853f * 1000.0f * (i - n0) / 48000.0f);
        std::vector<float> y = x;
        for (int off = 0; off < n; off += 256)
        {
            float* p[1] = { y.data() + off };
            g.processBlock(AudioBufferView<float>(p, 1, std::min(256, n - off)));
        }
        double eIn = 0.0, eOut = 0.0;
        for (int k = 0; k < 48; ++k)
        {
            eIn += static_cast<double>(x[static_cast<size_t>(n0 + k)]) * x[static_cast<size_t>(n0 + k)];
            eOut += static_cast<double>(y[static_cast<size_t>(n0 + latency + k)]) * y[static_cast<size_t>(n0 + latency + k)];
        }
        return 10.0 * std::log10(eOut / eIn);
    };
    int latency = -1;
    const double without = onsetLossDb(0.0f, latency);
    EXPECT_EQ(latency, 0);
    const double with = onsetLossDb(3.0f, latency);
    EXPECT_EQ(latency, 144);            // 3 ms @ 48 kHz
    EXPECT_LT(without, -3.0);           // the onset is chopped without lookahead
    EXPECT_GT(with, -0.5);              // and kept with it
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

// The three-stage cell keeps the two published figures of the two-stage one
// (half the reduction back in one release time, fully charged; a complete
// release of 0.5 to 5 s) and adds a medium memory: a short phrase leaves more
// tail than with two stages, sustained compression a smaller deep tail.
DSPARK_TEST(Compressor_opto_three_stage_cell_keeps_the_spec_and_adds_a_medium_memory)
{
    const double fs = 48000.0;
    auto residualAfter = [fs](bool three, double burst, double after, float* depth) {
        Compressor<float> comp;
        setupCharacterComp(comp, fs, Compressor<float>::Character::Opto, 100.0f);
        if (three) comp.setOptoCell(Compressor<float>::OptoCell::ThreeStage);
        auto trace = compressorGrTrace(comp, fs, burst, 3.5);
        const size_t t0 = static_cast<size_t>(fs * burst);
        if (depth != nullptr) *depth = -trace[t0 - 1];
        return -trace[t0 + static_cast<size_t>(fs * after)];
    };

    float depth = 0.0f;
    const float half = residualAfter(true, 4.0, 0.1, &depth);
    EXPECT_GT(depth, 12.0f);
    EXPECT_NEAR(half / depth, 0.50f, 0.10f);

    // A 150 ms phrase: the medium memory has charged, the deep one hardly.
    const float shortThree = residualAfter(true, 0.15, 0.3, nullptr);
    const float shortTwo = residualAfter(false, 0.15, 0.3, nullptr);
    EXPECT_GT(shortThree, shortTwo + 0.3f);
    EXPECT_LT(shortThree, 4.0f);

    // Sustained: still releasing after 3 s, with a smaller deep tail.
    const float longThree = residualAfter(true, 4.0, 3.0, nullptr);
    const float longTwo = residualAfter(false, 4.0, 3.0, nullptr);
    EXPECT_GT(longThree, 0.1f);
    EXPECT_LT(longThree, longTwo);

    // The default is the two-stage cell.
    Compressor<float> comp;
    EXPECT_TRUE((comp.getOptoCell() == Compressor<float>::OptoCell::TwoStage));
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

DSPARK_TEST(Compressor_hilbert_detector_catches_steps_without_latency)
{
    // The detector used to be a 191-tap FIR, 95 samples late, with the audio
    // delayed to match. It is now a zero-latency allpass pair, and the
    // rectified input backs it up at onsets, so a hot step leaves already
    // gain-reduced with no latency to report. Measured: the step escapes at
    // 0.317, as with the peak detector; the pair alone let 0.720 through.
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

    EXPECT_EQ(comp.getLatency(), 0);
    const float escape = measurePeak(tb.ch(0) + pre, 600);
    EXPECT_LT(escape, 0.35f);
}

// A low tone driven into the compressor must come out as a tone, not
// modulated at its own frequency. The peak detector's gain follows every
// half-cycle (THD+N -35 dB at 30 Hz with 5/100 ms); the Hilbert detector's
// envelope is flat, so the gain is too. Measured -96 dB at 30 Hz, -107 dB at
// 20 Hz; the FIR detector it replaced measured -35 dB, no better than Peak.
// The static curve holds within 0.02 dB (14 dB over a 4:1 threshold).
DSPARK_TEST(Compressor_hilbert_detector_leaves_low_tones_undistorted)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 3.0);
    for (const double f : { 20.0, 30.0, 50.0 })
    {
        Compressor<double> comp;
        comp.prepare(AudioSpec{ fs, 512, 1 });
        comp.setThreshold(-20.0);
        comp.setRatio(4.0);
        comp.setAttack(5.0);
        comp.setRelease(100.0);
        comp.setDetector(Compressor<double>::DetectorType::Hilbert);

        std::vector<double> y(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            y[static_cast<size_t>(i)] = 0.5 * std::sin(2.0 * pi<double> * f * i / fs);
        for (int pos = 0; pos < n; pos += 512)
        {
            double* chans[1] = { y.data() + pos };
            comp.processBlock(AudioBufferView<double>(chans, 1, std::min(512, n - pos)));
        }

        // Least-squares fit of the tone over the second half; the residual
        // is everything the gain modulation added.
        double ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
        for (int i = n / 2; i < n; ++i)
        {
            const double s = std::sin(2.0 * pi<double> * f * i / fs);
            const double c = std::cos(2.0 * pi<double> * f * i / fs);
            const double v = y[static_cast<size_t>(i)];
            ss += s * s; sc += s * c; cc += c * c; ys += v * s; yc += v * c;
        }
        const double det = ss * cc - sc * sc;
        const double a = (ys * cc - yc * sc) / det, b = (yc * ss - ys * sc) / det;
        double err = 0, sig = 0;
        for (int i = n / 2; i < n; ++i)
        {
            const double fit = a * std::sin(2.0 * pi<double> * f * i / fs)
                             + b * std::cos(2.0 * pi<double> * f * i / fs);
            const double e = y[static_cast<size_t>(i)] - fit;
            err += e * e; sig += fit * fit;
        }
        const double thdn = 10.0 * std::log10(err / sig);
        const double outDb = 20.0 * std::log10(std::sqrt(a * a + b * b));
        std::cout << "  " << f << " Hz: THD+N " << thdn << " dB, level " << outDb << " dBFS\n";
        EXPECT_LT(thdn, -85.0);
        EXPECT_NEAR(outDb, -16.5051, 0.03);   // -6.02 dBFS in, 4:1 above -20
    }
}

DSPARK_TEST(Compressor_getLatency_matches_active_configuration)
{
    // Latency must reflect the lookahead immediately after a setter (hosts re-read it before the next block), and feedback operation
    // (FeedBack topology or the FET character) must always report 0.
    const double fs = 48000.0;
    Compressor<float> comp;
    comp.prepare(spec(fs, 64, 1));

    EXPECT_EQ(comp.getLatency(), 0);
    comp.setLookahead(5.0f);
    EXPECT_EQ(comp.getLatency(), 240);
    comp.setDetector(Compressor<float>::DetectorType::Hilbert);
    EXPECT_EQ(comp.getLatency(), 240);   // the detector adds no latency
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

// The policy, pinned where it is most visible: a passage that never reaches
// the threshold. Static is a constant, curve-derived offset - that is what
// makes it free of pumping and click-free under parameter automation - so it
// lifts this passage by the full +7.5 dB it adds everywhere (-20 dB, 4:1).
// A caller who wants makeup only where the compressor actually works selects
// Adaptive, which leaves the passage untouched; Off leaves it untouched too.
DSPARK_TEST(Compressor_automakeup_policy_below_threshold)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    const int tail = static_cast<int>(fs * 0.5);
    using Mode = Compressor<float>::AutoMakeupMode;
    const struct { Mode mode; float gainDb; } cases[] = {
        { Mode::Off, 0.0f }, { Mode::Static, 7.5f }, { Mode::Adaptive, 0.0f } };
    for (const auto& c : cases)
    {
        auto tb = makeBuffer(1, n);
        generateSine(tb.ch(0), n, 440.0f, static_cast<float>(fs), 0.01f);   // -40 dBFS peak
        const float inDb = measureRMSDb(tb.ch(0) + (n - tail), tail);
        Compressor<float> comp;
        comp.prepare(spec(fs, n, 1));
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAutoMakeup(c.mode);
        comp.processBlock(tb.view());
        const float gain = measureRMSDb(tb.ch(0) + (n - tail), tail) - inDb;
        EXPECT_NEAR(gain, c.gainDb, 0.1f);
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
    EXPECT_EQ(static_cast<int>(subject.getCharacter()), 4);
    subject.setCharacter(Comp::Character::Clean);
    subject.setDetector(static_cast<Comp::DetectorType>(77));
    EXPECT_EQ(static_cast<int>(subject.getDetector()), 5);
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

// The match used raw per-block RMS: sub-bass the ear barely hears moved it
// by several dB (measured: a 20 Hz rumble under a 1 kHz tone pulled the
// output down 7.55 dB), and with small blocks every block's ratio was a new
// noisy estimate, so an effect with latency made the gain wobble (measured
// -0.23..+0.11 dB on a 40 Hz tone through a 100-sample delay, 64-sample
// blocks). Both sides are now K-weighted and integrated alike.
DSPARK_TEST(AutoGain_matches_loudness_and_stays_steady)
{
    auto rumbleMatch = [](AutoGain<double>::Weighting w) {
        AutoGain<double> ag;
        ag.prepare(spec(48000.0, 512, 1));
        ag.setSmoothingTime(10.0);
        ag.setWeighting(w);
        std::vector<double> b(512);
        double* p[1] = { b.data() };
        for (int blk = 0; blk < 400; ++blk)
        {
            for (int i = 0; i < 512; ++i)
                b[static_cast<size_t>(i)] = 0.2 * std::sin(twoPi<double> * 1000.0 * (blk * 512 + i) / 48000.0);
            ag.pushReference(AudioBufferView<double>(p, 1, 512));
            for (int i = 0; i < 512; ++i)
                b[static_cast<size_t>(i)] += 0.4 * std::sin(twoPi<double> * 20.0 * (blk * 512 + i) / 48000.0);
            ag.compensate(AudioBufferView<double>(p, 1, 512));
        }
        return static_cast<double>(ag.getCompensationDb());
    };
    EXPECT_GT(rumbleMatch(AutoGain<double>::Weighting::KWeighted), -1.5);  // ~ -0.6 dB
    EXPECT_NEAR(rumbleMatch(AutoGain<double>::Weighting::Flat), -7.0, 0.3); // plain RMS

    AutoGain<double> ag;
    ag.prepare(spec(48000.0, 64, 1));
    std::vector<double> delay(100, 0.0), b(64);
    size_t dp = 0;
    double lo = 1e9, hi = -1e9;
    double* p[1] = { b.data() };
    for (int blk = 0; blk < 2250; ++blk)
    {
        for (int i = 0; i < 64; ++i)
            b[static_cast<size_t>(i)] = 0.5 * std::sin(twoPi<double> * 40.0 * (blk * 64 + i) / 48000.0);
        ag.pushReference(AudioBufferView<double>(p, 1, 64));
        for (int i = 0; i < 64; ++i)
        {
            const double v = b[static_cast<size_t>(i)];
            b[static_cast<size_t>(i)] = delay[dp];
            delay[dp] = v;
            dp = (dp + 1) % delay.size();
        }
        ag.compensate(AudioBufferView<double>(p, 1, 64));
        if (blk > 1500)
        {
            lo = std::min(lo, static_cast<double>(ag.getCompensationDb()));
            hi = std::max(hi, static_cast<double>(ag.getCompensationDb()));
        }
    }
    EXPECT_LT(hi - lo, 0.02);
    EXPECT_LT(std::abs(hi), 0.05);
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

// ============================================================================
// Dynamics audit regressions (2026-11). Each case pins a defect; its comment
// states the defect and the number the unfixed code produced.
// ============================================================================

namespace {

using Vec = std::vector<float>;
constexpr double kPiD = 3.14159265358979323846;

Vec makeSine(double freq, double amp, double fs, int n)
{
    Vec v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = static_cast<float>(amp * std::sin(2.0 * kPiD * freq * i / fs));
    return v;
}

// Feeds every channel through `fn` in blocks of `block` samples.
template <class F>
void processInBlocks(std::vector<Vec>& ch, int block, F&& fn)
{
    const int n = static_cast<int>(ch[0].size());
    const int nc = static_cast<int>(ch.size());
    std::vector<float*> ptr(static_cast<size_t>(nc));
    for (int off = 0; off < n; off += block)
    {
        const int m = std::min(block, n - off);
        for (int c = 0; c < nc; ++c) ptr[static_cast<size_t>(c)] = ch[static_cast<size_t>(c)].data() + off;
        AudioBufferView<float> v(ptr.data(), nc, m);
        fn(v);
    }
}

// Coherent (Hann-weighted) amplitude of a tone.
double toneAmplitude(const float* x, int n, double freq, double fs)
{
    double c = 0, s = 0, w = 0;
    for (int i = 0; i < n; ++i)
    {
        const double wi = 0.5 - 0.5 * std::cos(2.0 * kPiD * (i + 0.5) / n);
        const double p = 2.0 * kPiD * freq * i / fs;
        c += wi * x[i] * std::cos(p);
        s += wi * x[i] * std::sin(p);
        w += wi;
    }
    return 2.0 * std::sqrt(c * c + s * s) / w;
}

double toDb(double x) { return 20.0 * std::log10(std::max(x, 1e-30)); }

double rmsOf(const float* x, int n)
{
    double s = 0;
    for (int i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
    return std::sqrt(s / n);
}

double peakOf(const float* x, int n)
{
    double p = 0;
    for (int i = 0; i < n; ++i) p = std::max(p, std::abs(static_cast<double>(x[i])));
    return p;
}

bool allFinite(const float* x, int n)
{
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(x[i])) return false;
    return true;
}

// Cookbook analog bell, H(s) = (s^2 + s A/Q + 1) / (s^2 + s/(A Q) + 1).
double analogBellDb(double f, double f0, double q, double gainDb)
{
    const double a = std::pow(10.0, gainDb / 40.0);
    const std::complex<double> s(0.0, f / f0);
    return 20.0 * std::log10(std::abs((s * s + s * (a / q) + 1.0) / (s * s + s / (a * q) + 1.0)));
}

// Steady single-band DynamicEQ, held at its range by an external key at the
// band centre; returns the gain (dB) the audio path applies to a probe tone.
double dynamicEqProbeGainDb(double fs, double f0, double probeHz, float rangeDb, bool boost)
{
    using DE = DynamicEQ<float>;
    auto d = std::make_unique<DE>();
    d->prepare(AudioSpec{ fs, 512, 1 });
    d->setNumBands(1);
    DE::BandConfig c;
    c.frequency = static_cast<float>(f0); c.q = 1.0f; c.threshold = -60.0f;
    c.aboveRatio = 1000.0f; c.aboveRangeDb = rangeDb; c.aboveBoost = boost;
    c.aboveAttackMs = 1.0f; c.aboveReleaseMs = 2000.0f;
    d->setBand(0, c);
    const int n = 36000;
    Vec probe = makeSine(probeHz, 0.01, fs, n);
    Vec key = makeSine(f0, 0.5, fs, n);
    for (int off = 0; off < n; off += 512)
    {
        const int m = std::min(512, n - off);
        float* a = probe.data() + off;
        float* k = key.data() + off;
        AudioBufferView<float> va(&a, 1, m), vk(&k, 1, m);
        d->processBlock(va, vk);
    }
    return toDb(toneAmplitude(probe.data() + n - 9600, 9600, probeHz, fs) / 0.01);
}

} // namespace

// ============================================================================
// DynamicEQ
// ============================================================================

// Defect: the precomputed bell terms used the raw centre frequency, so a band
// above Nyquist got a negative alpha and an unstable filter. Old: a 30 kHz cut
// band at 44.1 kHz on an 18 kHz tone peaked at 3.3e37 and went NaN.
DSPARK_TEST(DynamicEQ_bell_above_nyquist_stays_bounded)
{
    using DE = DynamicEQ<float>;
    for (float f : { 22060.0f, 30000.0f, 40000.0f })
    {
        auto d = std::make_unique<DE>();
        d->prepare(AudioSpec{ 44100.0, 512, 2 });
        DE::BandConfig c;
        c.frequency = f; c.threshold = -60.0f; c.aboveRatio = 8.0f;
        c.aboveAttackMs = 1.0f; c.aboveReleaseMs = 30.0f; c.aboveRangeDb = 18.0f;
        d->setNumBands(1);
        d->setBand(0, c);
        Vec s = makeSine(18000.0, 0.5, 44100.0, 44100);
        std::vector<Vec> x { s, s };
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        EXPECT_TRUE(allFinite(x[0].data(), 44100));
        EXPECT_LT(peakOf(x[0].data(), 44100), 1.0);
    }
}

// Defect: cookbook bells cramp towards Nyquist. Old: a +12 dB bell at 16 kHz
// (Q 1, 48 kHz) gave +3.42 dB at 20 kHz against +9.65 dB for the prototype.
DSPARK_TEST(DynamicEQ_bell_follows_analog_prototype_near_nyquist)
{
    const double g = dynamicEqProbeGainDb(48000.0, 16000.0, 20000.0, 12.0f, true);
    EXPECT_NEAR(g, analogBellDb(20000.0, 16000.0, 1.0, 12.0), 1.0);
    EXPECT_NEAR(dynamicEqProbeGainDb(48000.0, 16000.0, 16000.0, 12.0f, true), 12.0, 0.1);
}

// Defect: a band count increase replayed the stale filter and gain state of
// the re-activated band. Old: a boosted band re-added on a 0.001 tone peaked at
// 5.37 (cut band: 0.20).
DSPARK_TEST(DynamicEQ_reactivated_band_starts_clean)
{
    using DE = DynamicEQ<float>;
    for (bool boost : { false, true })
    {
        auto d = std::make_unique<DE>();
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        d->setNumBands(2);
        DE::BandConfig b0; b0.frequency = 200.0f; b0.aboveRatio = 1.0f;
        d->setBand(0, b0);
        DE::BandConfig c; c.frequency = 1000.0f; c.q = 8.0f; c.threshold = -40.0f; c.aboveRatio = 8.0f;
        c.aboveAttackMs = 2.0f; c.aboveReleaseMs = 50.0f; c.aboveRangeDb = 18.0f; c.aboveBoost = boost;
        d->setBand(1, c);
        std::vector<Vec> loud { makeSine(1000.0, 0.8, 48000.0, 24000), makeSine(1000.0, 0.8, 48000.0, 24000) };
        processInBlocks(loud, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        d->setNumBands(1);
        std::vector<Vec> q1 { makeSine(1000.0, 0.001, 48000.0, 2048), makeSine(1000.0, 0.001, 48000.0, 2048) };
        processInBlocks(q1, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        d->setNumBands(2);
        std::vector<Vec> q2 { makeSine(1000.0, 0.001, 48000.0, 4096), makeSine(1000.0, 0.001, 48000.0, 4096) };
        processInBlocks(q2, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        EXPECT_LT(peakOf(q2[0].data(), 4096), 0.0015);
    }
}

// Defect: with oversampling the samples past maxBlockSize passed dry. Old:
// 1024-sample blocks on a 256 spec left the tail at rms 0.355 (processed 0.045).
DSPARK_TEST(DynamicEQ_oversampled_oversize_block_is_fully_processed)
{
    using DE = DynamicEQ<float>;
    auto d = std::make_unique<DE>();
    d->setOversampling(2);
    d->prepare(AudioSpec{ 48000.0, 256, 2 });
    DE::BandConfig c; c.frequency = 1000.0f; c.threshold = -30.0f; c.aboveRatio = 8.0f; c.aboveRangeDb = 18.0f;
    d->setNumBands(1);
    d->setBand(0, c);
    std::vector<Vec> x { makeSine(1000.0, 0.5, 48000.0, 24576), makeSine(1000.0, 0.5, 48000.0, 24576) };
    processInBlocks(x, 1024, [&](AudioBufferView<float> v) { d->processBlock(v); });
    const double head = rmsOf(x[0].data() + 1024 * 20 + 64, 128);
    const double tail = rmsOf(x[0].data() + 1024 * 20 + 640, 256);
    EXPECT_LT(tail, 0.1);
    EXPECT_NEAR(tail, head, 0.01);
}

// Defect: reset() kept the analytic detectors and the oversamplers. Old: at
// 2x, silence after reset() came out with a 0.84 peak and a -13 dB gain.
DSPARK_TEST(DynamicEQ_reset_clears_detector_and_oversampler)
{
    using DE = DynamicEQ<float>;
    for (int os : { 1, 2 })
    {
        auto d = std::make_unique<DE>();
        d->setOversampling(os);
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        DE::BandConfig c; c.frequency = 1000.0f; c.threshold = -20.0f; c.aboveRatio = 8.0f;
        c.aboveAttackMs = 0.1f; c.aboveReleaseMs = 5.0f; c.aboveRangeDb = 24.0f;
        d->setNumBands(1);
        d->setBand(0, c);
        std::vector<Vec> x { makeSine(1000.0, 0.9, 48000.0, 4096), makeSine(1000.0, 0.9, 48000.0, 4096) };
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        d->reset();
        std::vector<Vec> z(2, Vec(512, 0.0f));
        processInBlocks(z, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        EXPECT_LT(peakOf(z[0].data(), 512), 1e-6);
        EXPECT_NEAR(d->getBandGainDb(0), 0.0f, 0.01f);
    }
}

// Defect: the band count started at 0, so setBand() alone did nothing. Old:
// 0.00 dB of action from a band configured for 24 dB of cut.
DSPARK_TEST(DynamicEQ_setBand_without_setNumBands_is_active)
{
    using DE = DynamicEQ<float>;
    auto d = std::make_unique<DE>();
    d->prepare(AudioSpec{ 48000.0, 512, 2 });
    DE::BandConfig c; c.frequency = 1000.0f; c.threshold = -40.0f; c.aboveRatio = 8.0f;
    c.aboveAttackMs = 1.0f; c.aboveRangeDb = 24.0f;
    d->setBand(0, c);
    std::vector<Vec> x { makeSine(1000.0, 0.5, 48000.0, 24000), makeSine(1000.0, 0.5, 48000.0, 24000) };
    processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
    EXPECT_LT(toDb(rmsOf(x[0].data() + 19200, 4800) / (0.5 / std::sqrt(2.0))), -20.0);
}

// Defect: getState() did not store the lookahead. Old: latency 240 came back as 0.
DSPARK_TEST(DynamicEQ_state_round_trip_keeps_lookahead)
{
    using DE = DynamicEQ<float>;
    auto a = std::make_unique<DE>();
    a->prepare(AudioSpec{ 48000.0, 512, 2 });
    a->setLookahead(5.0f);
    a->setNumBands(1);
    const auto blob = a->getState();
    auto b = std::make_unique<DE>();
    b->prepare(AudioSpec{ 48000.0, 512, 2 });
    EXPECT_TRUE(b->setState(blob.data(), blob.size()));
    EXPECT_EQ(b->getLatency(), a->getLatency());
    EXPECT_EQ(b->getLatency(), 240);
}

// Defect: the 16-sample coefficient refresh restarted at every block, so the
// audio depended on the host's block partition. Old: 7.6e-3 (attack 5 ms) and
// 2.1e-2 (attack 1 ms) between 7- and 512-sample blocks.
DSPARK_TEST(DynamicEQ_output_does_not_depend_on_block_partition)
{
    using DE = DynamicEQ<float>;
    const int n = 48000;
    Vec a(static_cast<size_t>(n));
    uint32_t seed = 5u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double t = i / 48000.0;
        const double env = 0.15 + 0.6 * std::exp(-std::fmod(t, 0.25) * 18.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.5 * std::sin(2.0 * kPiD * 1730.3 * t)
                                  + 0.1 * (static_cast<double>(seed) / 4294967296.0 - 0.5)));
    }
    auto run = [&](int block) {
        auto d = std::make_unique<DE>();
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        DE::BandConfig c; c.frequency = 1700.0f; c.q = 2.0f; c.threshold = -30.0f; c.aboveRatio = 4.0f; c.aboveAttackMs = 1.0f;
        d->setNumBands(1);
        d->setBand(0, c);
        std::vector<Vec> x { a, a };
        processInBlocks(x, block, [&](AudioBufferView<float> v) { d->processBlock(v); });
        return x[0];
    };
    const Vec ref = run(512);
    for (int block : { 7, 33, 100 })
    {
        const Vec x = run(block);
        double md = 0;
        for (int i = 0; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(i)]) - ref[static_cast<size_t>(i)]));
        EXPECT_EQ(md, 0.0);
    }
}

// Defect: one NaN sample latched in the detector and filter state. Old: the
// output stayed NaN for the remaining 38000 samples.
DSPARK_TEST(DynamicEQ_non_finite_sample_does_not_latch)
{
    using DE = DynamicEQ<float>;
    auto make = [] {
        auto d = std::make_unique<DE>();
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        DE::BandConfig c; c.frequency = 1000.0f; c.threshold = -30.0f; c.aboveRatio = 4.0f;
        d->setNumBands(1);
        d->setBand(0, c);
        return d;
    };
    auto d = make(), twin = make();
    const int n = 48000;
    Vec s = makeSine(997.0, 0.5, 48000.0, n);
    std::vector<Vec> x { s, s }, y { s, s };
    x[0][10000] = std::numeric_limits<float>::quiet_NaN();
    x[1][10000] = std::numeric_limits<float>::infinity();
    processInBlocks(x, 256, [&](AudioBufferView<float> v) { d->processBlock(v); });
    processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
    for (int c = 0; c < 2; ++c)
    {
        EXPECT_TRUE(allFinite(x[static_cast<size_t>(c)].data() + 12000, n - 12000));
        double md = 0;
        for (int i = n - 4800; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(c)][static_cast<size_t>(i)]) - y[static_cast<size_t>(c)][static_cast<size_t>(i)]));
        EXPECT_LT(md, 1e-4);
    }
}

namespace {

// Runs a DynamicEQ over audio with its own key, in blocks.
void dynEqRun(DynamicEQ<float>& eq, std::vector<Vec>& audio, std::vector<Vec>& key, int block = 512)
{
    const int n = static_cast<int>(audio[0].size());
    std::vector<float*> a(audio.size()), k(key.size());
    for (int off = 0; off < n; off += block)
    {
        const int m = std::min(block, n - off);
        for (size_t c = 0; c < audio.size(); ++c) a[c] = audio[c].data() + off;
        for (size_t c = 0; c < key.size(); ++c) k[c] = key[c].data() + off;
        AudioBufferView<float> va(a.data(), static_cast<int>(a.size()), m);
        AudioBufferView<float> vk(k.data(), static_cast<int>(k.size()), m);
        eq.processBlock(va, vk);
    }
}

std::unique_ptr<DynamicEQ<float>> dynEqMake(bool mixed, int channels, const DynamicEQ<float>::BandConfig& band)
{
    auto d = std::make_unique<DynamicEQ<float>>();
    d->setMixedBands(mixed);
    d->prepare(AudioSpec{ 48000.0, 512, channels });
    d->setNumBands(1);
    d->setBand(0, band);
    return d;
}

// The gain (dB) a one-band DynamicEQ gives a quiet probe at the centre of its
// band while a key tone of the same frequency holds its level.
double dynEqHeldGainDb(bool mixed, const DynamicEQ<float>::BandConfig& band, double keyAmplitude)
{
    auto d = dynEqMake(mixed, 1, band);
    const int n = 48000;
    const double hz = static_cast<double>(band.frequency);
    std::vector<Vec> probe { makeSine(hz, 0.01, 48000.0, n) };
    std::vector<Vec> key { makeSine(hz, keyAmplitude, 48000.0, n) };
    dynEqRun(*d, probe, key);
    return toDb(toneAmplitude(probe[0].data() + n - 9600, 9600, hz, 48000.0) / 0.01);
}

DynamicEQ<float>::BandConfig dynEqCut(float hz, float q, float threshold, float ratio, float range)
{
    DynamicEQ<float>::BandConfig c;
    c.frequency = hz; c.q = q; c.threshold = threshold;
    c.aboveRatio = ratio; c.aboveRangeDb = range;
    c.aboveAttackMs = 1.0f; c.aboveReleaseMs = 50.0f;
    return c;
}

// Milliseconds a band takes, once its key stops, to give back all but 10 % of
// a 6 dB cut. The key lasts keyMs.
double dynEqReleaseMs(const DynamicEQ<float>::BandConfig& band, double keyMs)
{
    auto d = dynEqMake(false, 1, band);
    const int on = static_cast<int>(keyMs * 48.0), tail = 48000;
    std::vector<Vec> audio { Vec(static_cast<size_t>(on + tail), 0.0f) };
    std::vector<Vec> key { makeSine(static_cast<double>(band.frequency), 0.0251189, 48000.0, on + tail) };
    std::fill(key[0].begin() + on, key[0].end(), 0.0f);
    std::vector<float*> a(1), k(1);
    double deepest = 0.0;
    for (int off = 0; off < on + tail; off += 64)
    {
        const int m = std::min(64, on + tail - off);
        a[0] = audio[0].data() + off;
        k[0] = key[0].data() + off;
        AudioBufferView<float> va(a.data(), 1, m), vk(k.data(), 1, m);
        d->processBlock(va, vk);
        const double gain = static_cast<double>(d->getBandGainDb(0));
        deepest = std::min(deepest, gain);
        if (off >= on && gain > -0.6) return deepest < -5.0 ? (off + m - on) / 48.0 : -1.0;
    }
    return 1e9;
}

} // namespace

// Improvement: a band made of fixed filters mixed by the gain sits exactly on
// the gain the ballistics ask for, at rest, half way and at the end of its
// range, and so does the default engine with the gain at rest that is new.
DSPARK_TEST(DynamicEQ_band_sits_on_the_gain_it_is_asked_in_both_engines)
{
    for (bool mixed : { false, true })
    {
        // 12 dB over the threshold at 2:1 is 6 dB of cut; far over it, the range.
        auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 2.0f, 12.0f);
        EXPECT_NEAR(dynEqHeldGainDb(mixed, c, 0.0398107), -6.0, 0.1);
        EXPECT_NEAR(dynEqHeldGainDb(mixed, c, 0.5), -12.0, 0.1);
        EXPECT_NEAR(dynEqHeldGainDb(mixed, c, 0.0001), 0.0, 0.02);

        // A band that rests at +4 dB: there with no key, and 6 dB below it with one.
        c.gain = 4.0f;
        EXPECT_NEAR(dynEqHeldGainDb(mixed, c, 0.0001), 4.0, 0.1);
        EXPECT_NEAR(dynEqHeldGainDb(mixed, c, 0.0398107), -2.0, 0.1);

        // A boost, and a low shelf heard well below its corner.
        auto lift = dynEqCut(3000.0f, 1.0f, -40.0f, 4.0f, 9.0f);
        lift.aboveBoost = true;
        EXPECT_NEAR(dynEqHeldGainDb(mixed, lift, 0.5), 9.0, 0.1);
        auto shelf = dynEqCut(2000.0f, 1.0f, -40.0f, 2.0f, 12.0f);
        shelf.shape = DynamicEQ<float>::BandShape::LowShelf;
        auto held = dynEqMake(mixed, 1, shelf);         // probe and key at 100 Hz, under a shelf at 2 kHz
        const int n = 48000;
        std::vector<Vec> probe { makeSine(100.0, 0.01, 48000.0, n) };
        std::vector<Vec> key { makeSine(100.0, 0.0398107, 48000.0, n) };
        dynEqRun(*held, probe, key);
        EXPECT_NEAR(toDb(toneAmplitude(probe[0].data() + n - 9600, 9600, 100.0, 48000.0) / 0.01), -6.0, 0.15);
    }
}

// Improvement: a mixed band that rests at 0 dB and is not asked to act returns
// its input bit for bit, with its filters running.
DSPARK_TEST(DynamicEQ_mixed_band_at_rest_returns_its_input)
{
    auto d = dynEqMake(true, 2, dynEqCut(1000.0f, 2.0f, -20.0f, 4.0f, 12.0f));
    d->setBand(1, dynEqCut(5000.0f, 1.0f, -20.0f, 4.0f, 12.0f));
    const int n = 24000;
    Vec s = makeSine(1000.0, 0.01, 48000.0, n);
    uint32_t seed = 9u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        s[static_cast<size_t>(i)] += 0.004f * (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f);
    }
    std::vector<Vec> x { s, s };
    processInBlocks(x, 480, [&](AudioBufferView<float> v) { d->processBlock(v); });
    EXPECT_TRUE(std::memcmp(x[0].data(), s.data(), sizeof(float) * static_cast<size_t>(n)) == 0);
    EXPECT_TRUE(std::memcmp(x[1].data(), s.data(), sizeof(float) * static_cast<size_t>(n)) == 0);
    EXPECT_NEAR(static_cast<double>(d->getBandGainDb(0)), 0.0, 1e-9);
}

// Defect of the default engine that the mixed one does not have: the band is
// designed again every 16 samples, so its gain moves in steps and a steady
// tone under a moving band grows lines 3 kHz away (48000 / 16). A 6 kHz tone
// under a band that a key opens and closes 25 times a second: the default
// engine leaves them 50 dB under the tone or so; mixed, they are not there.
DSPARK_TEST(DynamicEQ_mixed_band_moves_without_steps)
{
    auto c = dynEqCut(6000.0f, 1.0f, -30.0f, 4.0f, 12.0f);
    c.aboveAttackMs = 5.0f; c.aboveReleaseMs = 20.0f;
    const int n = 96000;
    auto lines = [&](bool mixed) {
        auto d = dynEqMake(mixed, 1, c);
        std::vector<Vec> tone { makeSine(6000.0, 0.25, 48000.0, n) };
        std::vector<Vec> key { makeSine(6000.0, 0.5, 48000.0, n) };
        for (int i = 0; i < n; ++i)
            if ((i / 960) % 2 == 1) key[0][static_cast<size_t>(i)] = 0.0f;      // 20 ms on, 20 ms off
        dynEqRun(*d, tone, key);
        const float* y = tone[0].data() + 48000;                               // one second, 25 periods
        const double carrier = toneAmplitude(y, 48000, 6000.0, 48000.0);
        const double worst = std::max({ toneAmplitude(y, 48000, 8975.0, 48000.0),
                                        toneAmplitude(y, 48000, 3025.0, 48000.0),
                                        toneAmplitude(y, 48000, 9025.0, 48000.0),
                                        toneAmplitude(y, 48000, 2975.0, 48000.0) });
        return toDb(worst / carrier);
    };
    const double stepped = lines(false), mixed = lines(true);
    EXPECT_LT(mixed, -80.0);
    EXPECT_LT(mixed, stepped - 20.0);
}

// New: a knee around the threshold, given or automatic. With the level on the
// threshold a knee of W dB at ratio R gives (1 - 1/R) W / 8 dB; the automatic
// knee is 3 dB per (1 + octaves of width) at a range of 12 dB and grows with
// the root of the range.
DSPARK_TEST(DynamicEQ_knee_bends_the_curve_around_the_threshold)
{
    using DE = DynamicEQ<float>;
    auto c = dynEqCut(1000.0f, 1.0f, -30.0f, 4.0f, 12.0f);
    const double onThreshold = 0.0316228, sixOver = 0.0630957;

    EXPECT_NEAR(dynEqHeldGainDb(false, c, onThreshold), 0.0, 0.06);        // a corner: nothing yet
    c.knee = 12.0f;
    EXPECT_NEAR(dynEqHeldGainDb(false, c, onThreshold), -1.125, 0.08);
    EXPECT_NEAR(dynEqHeldGainDb(true, c, onThreshold), -1.125, 0.08);
    EXPECT_NEAR(dynEqHeldGainDb(false, c, sixOver), -4.5, 0.1);            // where the knee ends

    c.knee = DE::kAutomaticKnee;
    const double narrow = dynEqHeldGainDb(false, c, onThreshold);          // Q 1: 7.16 dB of knee
    EXPECT_NEAR(narrow, -0.671, 0.08);
    c.aboveRangeDb = 24.0f;
    const double far = dynEqHeldGainDb(false, c, onThreshold);             // 10.13 dB
    EXPECT_NEAR(far, -0.950, 0.08);
    c.aboveRangeDb = 12.0f; c.q = 0.5f;
    const double wide = dynEqHeldGainDb(false, c, onThreshold);            // 10.63 dB
    EXPECT_NEAR(wide, -0.997, 0.08);
    EXPECT_LT(far, narrow - 0.15);
    EXPECT_LT(wide, narrow - 0.15);
}

// New: program timing. A held tone lands on the static curve, and the band
// gives back a 6 dB cut several times sooner after a short event (the fast
// follower: about 20 ms) than after a held one (the slow one: about 330 ms).
DSPARK_TEST(DynamicEQ_program_timing_lets_go_of_short_events_sooner)
{
    using DE = DynamicEQ<float>;
    auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 4.0f, 12.0f);
    c.timing = DE::Timing::Program;
    c.aboveAttackMs = 500.0f; c.aboveReleaseMs = 5000.0f;                  // not used
    EXPECT_NEAR(dynEqHeldGainDb(false, c, 0.0251189), -6.0, 0.1);          // 8 dB over at 4:1
    EXPECT_NEAR(dynEqHeldGainDb(true, c, 0.0251189), -6.0, 0.1);

    const double afterShort = dynEqReleaseMs(c, 15.0), afterLong = dynEqReleaseMs(c, 2000.0);
    EXPECT_GT(afterShort, 5.0);
    EXPECT_LT(afterShort, 60.0);
    EXPECT_GT(afterLong, 200.0);
    EXPECT_LT(afterLong, 600.0);
    EXPECT_GT(afterLong, 4.0 * afterShort);

    // The times come from the band: an octave and a half lower and twice as
    // narrow, the fast follower is six times slower.
    c.frequency = 350.0f; c.q = 4.0f;
    const double lowShort = dynEqReleaseMs(c, 60.0);
    EXPECT_GT(lowShort, 1.5 * afterShort);
}

// New: a threshold that counts from the band's own recent level. The band acts
// the same 40 dB lower, lets a quieter passage through, comes back to it as
// its reference falls (2 dB per second), and when it is moved to a quieter
// zone a second reference takes over after half a second instead of the band
// waiting for the first to fall.
DSPARK_TEST(DynamicEQ_relative_threshold_follows_the_level_of_the_band)
{
    auto c = dynEqCut(1000.0f, 2.0f, -6.0f, 4.0f, 12.0f);
    c.relativeThreshold = true;
    EXPECT_NEAR(dynEqHeldGainDb(false, c, 0.5), -4.5, 0.15);               // 6 dB over at 4:1
    EXPECT_NEAR(dynEqHeldGainDb(false, c, 0.005), -4.5, 0.15);
    EXPECT_NEAR(dynEqHeldGainDb(true, c, 0.005), -4.5, 0.15);

    auto d = dynEqMake(false, 1, c);
    auto play = [&](double hz, double amplitude, double seconds) {
        const int n = static_cast<int>(seconds * 48000.0);
        std::vector<Vec> audio { Vec(static_cast<size_t>(n), 0.0f) };
        std::vector<Vec> key { makeSine(hz, amplitude, 48000.0, n) };
        dynEqRun(*d, audio, key, 480);
        return static_cast<double>(d->getBandGainDb(0));
    };
    EXPECT_NEAR(play(1000.0, 0.5, 1.0), -4.5, 0.15);
    EXPECT_NEAR(play(1000.0, 0.158114, 0.5), 0.0, 0.2);                    // 10 dB quieter: under the threshold
    EXPECT_NEAR(play(1000.0, 0.158114, 6.0), -4.5, 0.3);                   // the reference has come down to it

    // Moved to 8 kHz, where the key is 20 dB quieter than the reference.
    c.frequency = 8000.0f;
    d->setBand(0, c);
    EXPECT_NEAR(play(8000.0, 0.0158114, 0.4), 0.0, 0.3);                   // the second reference still listens
    EXPECT_NEAR(play(8000.0, 0.0158114, 0.4), -4.5, 0.4);                  // and has taken over
}

// New: a band acts on, and listens to, the left, the right, the middle or the
// sides of the pair, in both engines.
DSPARK_TEST(DynamicEQ_band_scope_acts_and_listens_where_it_is_set)
{
    using DE = DynamicEQ<float>;
    const int n = 48000;
    const Vec probe = makeSine(1000.0, 0.02, 48000.0, n), other = makeSine(1000.0, 0.01, 48000.0, n);
    const Vec loud = makeSine(1000.0, 0.5, 48000.0, n), silent(static_cast<size_t>(n), 0.0f);
    auto gainOf = [&](const Vec& y, double reference) {
        return toDb(toneAmplitude(y.data() + n - 9600, 9600, 1000.0, 48000.0) / reference);
    };
    auto farFrom = [&](const Vec& y, const Vec& x) {
        double worst = 0.0;
        for (int i = 0; i < n; ++i)
            worst = std::max(worst, std::abs(static_cast<double>(y[static_cast<size_t>(i)]) - x[static_cast<size_t>(i)]));
        return worst;
    };

    for (bool mixed : { false, true })
    {
        auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 100.0f, 12.0f);

        // The key is on the left only.
        c.scope = DE::Scope::Left;
        {
            auto d = dynEqMake(mixed, 2, c);
            std::vector<Vec> x { probe, probe }, key { loud, silent };
            dynEqRun(*d, x, key);
            EXPECT_NEAR(gainOf(x[0], 0.02), -12.0, 0.2);
            EXPECT_EQ(farFrom(x[1], probe), 0.0);
        }
        c.scope = DE::Scope::Right;
        {
            auto d = dynEqMake(mixed, 2, c);
            std::vector<Vec> x { probe, probe }, key { loud, silent };
            dynEqRun(*d, x, key);
            EXPECT_EQ(farFrom(x[0], probe), 0.0);
            EXPECT_LT(farFrom(x[1], probe), 1e-6);                          // it hears silence
        }

        // The audio: `probe` in the middle and `other` on the sides. The key is all middle.
        Vec left(static_cast<size_t>(n)), right(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            left[static_cast<size_t>(i)] = probe[static_cast<size_t>(i)] + other[static_cast<size_t>(i)];
            right[static_cast<size_t>(i)] = probe[static_cast<size_t>(i)] - other[static_cast<size_t>(i)];
        }
        c.scope = DE::Scope::Mid;
        {
            auto d = dynEqMake(mixed, 2, c);
            std::vector<Vec> x { left, right }, key { loud, loud };
            dynEqRun(*d, x, key);
            Vec middle(static_cast<size_t>(n)), sides(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                middle[static_cast<size_t>(i)] = 0.5f * (x[0][static_cast<size_t>(i)] + x[1][static_cast<size_t>(i)]);
                sides[static_cast<size_t>(i)] = 0.5f * (x[0][static_cast<size_t>(i)] - x[1][static_cast<size_t>(i)]);
            }
            EXPECT_NEAR(gainOf(middle, 0.02), -12.0, 0.2);
            EXPECT_NEAR(gainOf(sides, 0.01), 0.0, 0.01);
        }
        c.scope = DE::Scope::Side;
        {
            auto d = dynEqMake(mixed, 2, c);
            std::vector<Vec> x { left, right }, key { loud, loud };         // no sides in the key: nothing to do
            dynEqRun(*d, x, key);
            EXPECT_LT(farFrom(x[0], left), 1e-6);
            EXPECT_LT(farFrom(x[1], right), 1e-6);
            std::vector<Vec> y { left, right }, sideKey { loud, silent };   // half of it is sides
            auto e = dynEqMake(mixed, 2, c);
            dynEqRun(*e, y, sideKey);
            Vec middle(static_cast<size_t>(n)), sides(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                middle[static_cast<size_t>(i)] = 0.5f * (y[0][static_cast<size_t>(i)] + y[1][static_cast<size_t>(i)]);
                sides[static_cast<size_t>(i)] = 0.5f * (y[0][static_cast<size_t>(i)] - y[1][static_cast<size_t>(i)]);
            }
            EXPECT_NEAR(gainOf(sides, 0.01), -12.0, 0.2);
            EXPECT_NEAR(gainOf(middle, 0.02), 0.0, 0.01);
        }

        // One channel: the left and the middle are it; it has no right and no sides.
        for (auto place : { DE::Scope::Left, DE::Scope::Mid, DE::Scope::Right, DE::Scope::Side })
        {
            c.scope = place;
            auto d = dynEqMake(mixed, 1, c);
            std::vector<Vec> x { probe }, key { loud };
            dynEqRun(*d, x, key);
            const bool acts = place == DE::Scope::Left || place == DE::Scope::Mid;
            EXPECT_NEAR(gainOf(x[0], 0.02), acts ? -12.0 : 0.0, acts ? 0.2 : 0.001);
        }
    }
}

// Design of the mixed band: two fixed curves cannot be the nominal band at
// every depth, so the end of a side is chosen for the band to be exactly the
// bell of its Q half way through the range (in decibels). At a quarter of the
// range a cut is then wider than nominal, and at the end of it narrower.
DSPARK_TEST(DynamicEQ_mixed_band_is_the_nominal_bell_half_way)
{
    auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 4.0f, 12.0f);
    auto mixed = dynEqMake(true, 1, c), plain = dynEqMake(false, 1, c);
    const float hz[7] = { 250.0f, 500.0f, 800.0f, 1000.0f, 1250.0f, 2000.0f, 4000.0f };
    float a[7] = {}, b[7] = {};
    mixed->getBandMagnitude(0, -6.0f, hz, a, 7);
    plain->getBandMagnitude(0, -6.0f, hz, b, 7);
    for (int i = 0; i < 7; ++i) EXPECT_NEAR(toDb(a[i]), toDb(b[i]), 0.08);

    mixed->getBandMagnitude(0, -3.0f, hz, a, 7);
    plain->getBandMagnitude(0, -3.0f, hz, b, 7);
    EXPECT_NEAR(toDb(a[3]), -3.0, 0.05);
    EXPECT_NEAR(toDb(a[2]), -1.90, 0.1);            // nominal: -1.66 at 800 Hz
    EXPECT_LT(toDb(a[2]), toDb(b[2]) - 0.1);

    mixed->getBandMagnitude(0, -12.0f, hz, a, 7);
    plain->getBandMagnitude(0, -12.0f, hz, b, 7);
    EXPECT_NEAR(toDb(a[3]), -12.0, 0.05);
    EXPECT_NEAR(toDb(a[2]), -4.77, 0.15);           // nominal: -6.55
    EXPECT_GT(toDb(a[2]), toDb(b[2]) + 0.3);

    // A boost mirrors it: narrower than nominal near the rest.
    c.aboveBoost = true;
    mixed->setBand(0, c);
    plain->setBand(0, c);
    mixed->getBandMagnitude(0, 6.0f, hz, a, 7);
    plain->getBandMagnitude(0, 6.0f, hz, b, 7);
    for (int i = 0; i < 7; ++i) EXPECT_NEAR(toDb(a[i]), toDb(b[i]), 0.08);
    mixed->getBandMagnitude(0, 3.0f, hz, a, 7);
    plain->getBandMagnitude(0, 3.0f, hz, b, 7);
    EXPECT_LT(toDb(a[2]), toDb(b[2]) - 0.1);
}

// New fields: kept by the state, and guarded on the way in like the rest.
DSPARK_TEST(DynamicEQ_new_band_fields_are_kept_and_guarded)
{
    using DE = DynamicEQ<float>;
    auto c = dynEqCut(700.0f, 1.5f, -25.0f, 3.0f, 9.0f);
    c.gain = 3.0f; c.knee = 6.0f; c.scope = DE::Scope::Side;
    c.timing = DE::Timing::Program; c.relativeThreshold = true;

    auto a = dynEqMake(true, 2, c);
    const auto blob = a->getState();
    auto b = std::make_unique<DE>();
    b->prepare(AudioSpec{ 48000.0, 512, 2 });
    EXPECT_TRUE(b->setState(blob.data(), blob.size()));
    EXPECT_TRUE(b->isMixedBands());
    EXPECT_TRUE(b->getState() == blob);

    // What is not a value keeps the band's own; what is out of range is brought in.
    auto wild = c;
    wild.gain = std::numeric_limits<float>::quiet_NaN();
    wild.knee = std::numeric_limits<float>::infinity();
    wild.scope = static_cast<DE::Scope>(9);
    wild.timing = static_cast<DE::Timing>(-3);
    a->setBand(0, wild);
    EXPECT_TRUE(a->getState() == blob);

    auto far = c, held = c;
    far.gain = 90.0f; far.knee = 300.0f;
    held.gain = 24.0f; held.knee = 40.0f;
    auto e = dynEqMake(true, 2, far), f = dynEqMake(true, 2, held);
    EXPECT_TRUE(e->getState() == f->getState());
    far.knee = -7.0f; held.knee = DE::kAutomaticKnee;
    e->setBand(0, far); f->setBand(0, held);
    EXPECT_TRUE(e->getState() == f->getState());

    // The curve an interface draws is the gain the audio gets, in both engines.
    for (bool mixed : { false, true })
    {
        auto band = dynEqCut(1000.0f, 2.0f, -40.0f, 2.0f, 12.0f);
        band.gain = 4.0f;
        auto d = dynEqMake(mixed, 1, band);
        const float hz[3] = { 1000.0f, 100.0f, 1400.0f };
        float drawn[3] = {};
        d->getBandMagnitude(0, -6.0f, hz, drawn, 3);
        EXPECT_NEAR(toDb(drawn[0]), -2.0, 0.05);
        EXPECT_NEAR(toDb(drawn[1]), 0.0, 0.1);
        auto played = dynEqMake(mixed, 1, band);
        const int n = 48000;
        std::vector<Vec> probe { makeSine(1400.0, 0.01, 48000.0, n) };
        std::vector<Vec> key { makeSine(1000.0, 0.0398107, 48000.0, n) };
        dynEqRun(*played, probe, key);
        EXPECT_NEAR(toDb(toneAmplitude(probe[0].data() + n - 9600, 9600, 1400.0, 48000.0) / 0.01), toDb(drawn[2]), 0.1);
    }
}

// The mixed engine keeps what the default one guarantees: the output does not
// depend on how the host cuts the stream, and a sample that is not a number
// neither stays in a filter nor in a follower.
DSPARK_TEST(DynamicEQ_mixed_engine_is_block_independent_and_recovers)
{
    using DE = DynamicEQ<float>;
    const int n = 48000;
    Vec a(static_cast<size_t>(n));
    uint32_t seed = 5u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double t = i / 48000.0;
        const double env = 0.15 + 0.6 * std::exp(-std::fmod(t, 0.25) * 18.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.5 * std::sin(2.0 * kPiD * 1730.3 * t)
                                  + 0.1 * (static_cast<double>(seed) / 4294967296.0 - 0.5)));
    }
    auto make = [] {
        auto c = dynEqCut(1700.0f, 2.0f, -30.0f, 4.0f, 12.0f);
        c.gain = 2.0f;
        auto d = dynEqMake(true, 2, c);
        auto second = dynEqCut(6000.0f, 1.0f, -45.0f, 3.0f, 9.0f);
        second.scope = DE::Scope::Mid;
        second.timing = DE::Timing::Program;
        second.aboveBoost = true;
        d->setBand(1, second);
        return d;
    };
    auto run = [&](int block) {
        auto d = make();
        std::vector<Vec> x { a, a };
        for (int i = 0; i < n; ++i) x[1][static_cast<size_t>(i)] *= 0.7f;
        processInBlocks(x, block, [&](AudioBufferView<float> v) { d->processBlock(v); });
        return x;
    };
    const auto ref = run(512);
    for (int block : { 7, 33, 100 })
    {
        const auto x = run(block);
        double md = 0;
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < n; ++i)
                md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(c)][static_cast<size_t>(i)])
                                           - ref[static_cast<size_t>(c)][static_cast<size_t>(i)]));
        EXPECT_LT(md, 1e-6);
    }

    auto d = make(), twin = make();
    std::vector<Vec> x { a, a }, y { a, a };
    x[0][10000] = std::numeric_limits<float>::quiet_NaN();
    x[1][10000] = std::numeric_limits<float>::infinity();
    processInBlocks(x, 256, [&](AudioBufferView<float> v) { d->processBlock(v); });
    processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
    EXPECT_TRUE(std::isnan(x[0][10000]));
    EXPECT_TRUE(std::isinf(x[1][10000]));
    for (int c = 0; c < 2; ++c)
    {
        EXPECT_TRUE(allFinite(x[static_cast<size_t>(c)].data() + 10001, n - 10001));
        double md = 0;
        for (int i = n - 4800; i < n; ++i)
            md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(c)][static_cast<size_t>(i)])
                                       - y[static_cast<size_t>(c)][static_cast<size_t>(i)]));
        EXPECT_LT(md, 1e-4);
    }
}

namespace {

std::unique_ptr<DynamicEQ<float>> dynEqLinear(int channels, const DynamicEQ<float>::BandConfig& band)
{
    auto d = std::make_unique<DynamicEQ<float>>();
    d->setLinearPhase(true);
    d->prepare(AudioSpec{ 48000.0, 512, channels });
    d->setNumBands(1);
    d->setBand(0, band);
    return d;
}

struct DynEqTone { double gainDb = 0.0, phaseDeg = 0.0; };

// The gain and the phase a processed tone has, over its last `span` samples, against its input
// `latency` samples earlier.
DynEqTone dynEqToneAgainst(const Vec& y, const Vec& x, double hz, int latency, int span)
{
    const int n = static_cast<int>(y.size());
    std::complex<double> out(0.0, 0.0), in(0.0, 0.0);
    for (int i = n - span; i < n; ++i)
    {
        const std::complex<double> turn = std::polar(1.0, -2.0 * kPiD * hz * i / 48000.0);
        out += static_cast<double>(y[static_cast<size_t>(i)]) * turn;
        in += static_cast<double>(x[static_cast<size_t>(i - latency)]) * turn;
    }
    DynEqTone t;
    t.gainDb = 20.0 * std::log10(std::abs(out) / std::abs(in));
    t.phaseDeg = std::arg(out / in) * 180.0 / kPiD;
    return t;
}

} // namespace

// New mode: linear phase. With no band acting it is the signal, late by half
// the kernel plus one partition, to the bit.
DSPARK_TEST(DynamicEQ_linear_phase_is_the_signal_late_when_no_band_acts)
{
    auto d = dynEqLinear(2, dynEqCut(1000.0f, 2.0f, -20.0f, 4.0f, 12.0f));
    d->setBand(1, dynEqCut(5000.0f, 1.0f, -20.0f, 4.0f, 12.0f));
    EXPECT_EQ(d->getLatency(), 2560);
    EXPECT_EQ(d->getLinearPhaseKernelLength(), 4096);

    const int n = 24000;
    Vec s = makeSine(1000.0, 0.01, 48000.0, n);
    uint32_t seed = 9u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        s[static_cast<size_t>(i)] += 0.004f * (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f);
    }
    std::vector<Vec> x { s, s };
    processInBlocks(x, 480, [&](AudioBufferView<float> v) { d->processBlock(v); });
    bool late = true, quiet = true;
    for (int i = 0; i + 2560 < n; ++i)
        late = late && x[0][static_cast<size_t>(i + 2560)] == s[static_cast<size_t>(i)]
                    && x[1][static_cast<size_t>(i + 2560)] == s[static_cast<size_t>(i)];
    for (int i = 0; i < 2560; ++i) quiet = quiet && x[0][static_cast<size_t>(i)] == 0.0f;
    EXPECT_TRUE(late);
    EXPECT_TRUE(quiet);

    // A longer kernel, and what it costs in latency: half of it plus an eighth.
    auto longer = std::make_unique<DynamicEQ<float>>();
    longer->setLinearPhase(true);
    longer->setLinearPhaseKernelLength(8000);       // goes up to 8192
    longer->prepare(AudioSpec{ 48000.0, 512, 2 });
    EXPECT_EQ(longer->getLinearPhaseKernelLength(), 8192);
    EXPECT_EQ(longer->getLatency(), 5120);
    longer->setLinearPhase(false);
    longer->prepare(AudioSpec{ 48000.0, 512, 2 });
    EXPECT_EQ(longer->getLatency(), 0);
}

// The band sits on the gain it is asked, as in the other engines, and turns no
// phase: 200 Hz below the centre of a 6 dB cut the mixed engine is 19 degrees
// off the input and this one is on it.
DSPARK_TEST(DynamicEQ_linear_phase_band_sits_on_its_gain_and_turns_no_phase)
{
    const int n = 48000;
    auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 2.0f, 12.0f);
    auto held = [&](DynamicEQ<float>& eq, double probeHz, double keyAmplitude) {
        const Vec in = makeSine(probeHz, 0.01, 48000.0, n);
        std::vector<Vec> probe { in };
        std::vector<Vec> key { makeSine(1000.0, keyAmplitude, 48000.0, n) };
        dynEqRun(eq, probe, key);
        return dynEqToneAgainst(probe[0], in, probeHz, eq.getLatency(), 9600);
    };

    {
        auto d = dynEqLinear(1, c);
        const DynEqTone centre = held(*d, 1000.0, 0.0398107);           // 12 dB over at 2:1
        EXPECT_NEAR(centre.gainDb, -6.0, 0.2);
        EXPECT_NEAR(centre.phaseDeg, 0.0, 0.3);
    }
    {
        auto d = dynEqLinear(1, c);
        const DynEqTone below = held(*d, 800.0, 0.0398107);
        const float hz[1] = { 800.0f };
        float drawn[1] = {};
        d->getBandMagnitude(0, -6.0f, hz, drawn, 1);
        EXPECT_NEAR(below.gainDb, -3.30, 0.25);                         // the nominal bell at half the range
        EXPECT_NEAR(below.gainDb, toDb(drawn[0]), 0.2);
        EXPECT_NEAR(below.phaseDeg, 0.0, 0.5);

        auto mixed = dynEqMake(true, 1, c);
        const DynEqTone turned = held(*mixed, 800.0, 0.0398107);
        EXPECT_GT(std::abs(turned.phaseDeg), 10.0);
    }
    {
        auto d = dynEqLinear(1, c);
        EXPECT_NEAR(held(*d, 1000.0, 0.5).gainDb, -12.0, 0.3);          // the end of the range
    }
    {
        c.gain = 4.0f;
        auto d = dynEqLinear(1, c);
        EXPECT_NEAR(held(*d, 1000.0, 0.0001).gainDb, 4.0, 0.2);         // at rest
        auto e = dynEqLinear(1, c);
        EXPECT_NEAR(held(*e, 1000.0, 0.0398107).gainDb, -2.0, 0.3);
    }
}

// The gain is held back to meet the audio it was computed from: a key that
// starts at one sample bites the output that many samples plus the latency
// later, and with lookahead that much sooner, at no cost in latency.
DSPARK_TEST(DynamicEQ_linear_phase_keeps_time_with_its_detector)
{
    const int n = 24000, starts = 8000;
    auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 100.0f, 12.0f);
    auto level = [](const Vec& y, int from) {
        return toDb(rmsOf(y.data() + from, 96) * std::sqrt(2.0) / 0.01);
    };
    for (float lookaheadMs : { 0.0f, 5.0f })
    {
        auto d = dynEqLinear(1, c);
        d->setLookahead(lookaheadMs);
        EXPECT_EQ(d->getLatency(), 2560);
        std::vector<Vec> probe { makeSine(1000.0, 0.01, 48000.0, n) };
        std::vector<Vec> key { makeSine(1000.0, 0.5, 48000.0, n) };
        std::fill(key[0].begin(), key[0].begin() + starts, 0.0f);
        dynEqRun(*d, probe, key);
        const int bites = starts + 2560 - static_cast<int>(lookaheadMs * 48.0f);
        EXPECT_NEAR(level(probe[0], bites - 400), 0.0, 0.3);            // not yet
        EXPECT_NEAR(level(probe[0], bites + 720), -12.0, 0.5);          // 15 ms in: the whole cut
        if (lookaheadMs > 0.0f) EXPECT_LT(level(probe[0], starts + 2560 - 96), -9.0);
    }
}

// The mode keeps the guarantees of the others (block independence, recovery
// from a sample that is not a number), its placements, and its state.
DSPARK_TEST(DynamicEQ_linear_phase_blocks_scopes_and_state)
{
    using DE = DynamicEQ<float>;
    const int n = 48000;
    Vec a(static_cast<size_t>(n));
    uint32_t seed = 5u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double t = i / 48000.0;
        const double env = 0.15 + 0.6 * std::exp(-std::fmod(t, 0.25) * 18.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.5 * std::sin(2.0 * kPiD * 1730.3 * t)
                                  + 0.1 * (static_cast<double>(seed) / 4294967296.0 - 0.5)));
    }
    auto make = [] {
        auto c = dynEqCut(1700.0f, 2.0f, -30.0f, 4.0f, 12.0f);
        c.gain = 2.0f;
        auto d = dynEqLinear(2, c);
        auto second = dynEqCut(6000.0f, 1.0f, -45.0f, 3.0f, 9.0f);
        second.scope = DE::Scope::Mid;
        second.timing = DE::Timing::Program;
        second.aboveBoost = true;
        d->setBand(1, second);
        return d;
    };
    auto run = [&](int block) {
        auto d = make();
        std::vector<Vec> x { a, a };
        for (int i = 0; i < n; ++i) x[1][static_cast<size_t>(i)] *= 0.7f;
        processInBlocks(x, block, [&](AudioBufferView<float> v) { d->processBlock(v); });
        return x;
    };
    const auto ref = run(512);
    EXPECT_GT(rmsOf(ref[0].data() + 24000, 24000), 0.01);
    for (int block : { 7, 100 })
    {
        const auto x = run(block);
        double md = 0;
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < n; ++i)
                md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(c)][static_cast<size_t>(i)])
                                           - ref[static_cast<size_t>(c)][static_cast<size_t>(i)]));
        EXPECT_LT(md, 1e-5);
    }

    {
        auto d = make(), twin = make();
        std::vector<Vec> x { a, a }, y { a, a };
        x[0][10000] = std::numeric_limits<float>::quiet_NaN();
        processInBlocks(x, 256, [&](AudioBufferView<float> v) { d->processBlock(v); });
        processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
        EXPECT_TRUE(std::isnan(x[0][10000 + 2560]));                    // it comes out late, like the rest
        EXPECT_TRUE(allFinite(x[0].data() + 10000 + 2561, n - 12561));
        EXPECT_TRUE(allFinite(x[1].data(), n));
        double md = 0;
        for (int i = n - 4800; i < n; ++i)
            md = std::max(md, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
        EXPECT_LT(md, 1e-4);
    }

    // A band on the middle: the middle takes the cut, the sides are the sides, late.
    {
        auto c = dynEqCut(1000.0f, 2.0f, -40.0f, 100.0f, 12.0f);
        c.scope = DE::Scope::Mid;
        auto d = dynEqLinear(2, c);
        const Vec probe = makeSine(1000.0, 0.02, 48000.0, n), other = makeSine(1000.0, 0.01, 48000.0, n);
        const Vec loud = makeSine(1000.0, 0.5, 48000.0, n);
        Vec left(static_cast<size_t>(n)), right(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            left[static_cast<size_t>(i)] = probe[static_cast<size_t>(i)] + other[static_cast<size_t>(i)];
            right[static_cast<size_t>(i)] = probe[static_cast<size_t>(i)] - other[static_cast<size_t>(i)];
        }
        std::vector<Vec> x { left, right }, key { loud, loud };
        dynEqRun(*d, x, key);
        Vec middle(static_cast<size_t>(n)), sides(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            middle[static_cast<size_t>(i)] = 0.5f * (x[0][static_cast<size_t>(i)] + x[1][static_cast<size_t>(i)]);
            sides[static_cast<size_t>(i)] = 0.5f * (x[0][static_cast<size_t>(i)] - x[1][static_cast<size_t>(i)]);
        }
        EXPECT_NEAR(toDb(toneAmplitude(middle.data() + n - 9600, 9600, 1000.0, 48000.0) / 0.02), -12.0, 0.3);
        EXPECT_NEAR(toDb(toneAmplitude(sides.data() + n - 9600, 9600, 1000.0, 48000.0) / 0.01), 0.0, 0.01);
    }

    // The state carries the mode and its kernel.
    {
        auto d = std::make_unique<DE>();
        d->setLinearPhase(true);
        d->setLinearPhaseKernelLength(8192);
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        d->setNumBands(1);
        const auto blob = d->getState();
        auto e = std::make_unique<DE>();
        EXPECT_TRUE(e->setState(blob.data(), blob.size()));
        EXPECT_TRUE(e->isLinearPhase());
        e->prepare(AudioSpec{ 48000.0, 512, 2 });
        EXPECT_EQ(e->getLatency(), 5120);
        EXPECT_TRUE(e->getState() == blob);
    }
}

// ============================================================================
// Compressor: automatic threshold, lookahead inside a latency, amount, soft range
// ============================================================================

namespace {

// Gain reduction a compressor shows at the end of a tone that lasts `seconds`.
template <class Setup>
double compHeldReduction(double amplitude, double seconds, Setup&& setup)
{
    Compressor<float> c;
    c.setThreshold(-12.0f);
    c.setRatio(4.0f);
    c.setAttack(1.0f);
    c.setRelease(50.0f);
    setup(c);
    c.prepare(AudioSpec{ 48000.0, 512, 1 });
    std::vector<Vec> v { makeSine(1000.0, amplitude, 48000.0, static_cast<int>(seconds * 48000.0)) };
    processInBlocks(v, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
    return static_cast<double>(c.getGainReductionDb());
}

} // namespace

// New: a threshold that counts from the key's own recent peak. The compressor
// acts the same 40 dB lower (with an absolute threshold it would not act at
// all there), in both topologies, and comes back to a quieter passage as its
// reference falls 1 dB per second.
DSPARK_TEST(Compressor_automatic_threshold_acts_the_same_at_any_level)
{
    using C = Compressor<float>;
    for (bool feedback : { false, true })
    {
        auto setup = [&](C& c) {
            c.setAutoThreshold(true);
            if (feedback) c.setTopology(C::Topology::FeedBack);
        };
        const double loud = compHeldReduction(0.5, 1.0, setup), quiet = compHeldReduction(0.005, 1.0, setup);
        EXPECT_LT(loud, -3.0);
        EXPECT_NEAR(quiet, loud, 0.05);
        auto absolute = [&](C& c) { if (feedback) c.setTopology(C::Topology::FeedBack); };
        EXPECT_NEAR(compHeldReduction(0.005, 1.0, absolute), 0.0, 0.01);
    }

    C c;
    c.setThreshold(-12.0f);
    c.setRatio(4.0f);
    c.setAttack(1.0f);
    c.setRelease(50.0f);
    c.setAutoThreshold(true);
    c.setDetector(C::DetectorType::Rms);
    c.prepare(AudioSpec{ 48000.0, 512, 1 });
    EXPECT_TRUE(c.isAutoThreshold());
    auto play = [&](double amplitude, double seconds) {
        std::vector<Vec> v { makeSine(1000.0, amplitude, 48000.0, static_cast<int>(seconds * 48000.0)) };
        processInBlocks(v, 480, [&](AudioBufferView<float> b) { c.processBlock(b); });
        return static_cast<double>(c.getGainReductionDb());
    };
    const double first = play(0.5, 1.0);                    // on its reference: 12 dB over at 4:1
    EXPECT_NEAR(first, -9.0, 0.3);
    EXPECT_NEAR(play(0.25, 0.5), -4.9, 0.5);                // 6 dB down: 6.5 dB over (the reference fell 0.5)
    EXPECT_NEAR(play(0.25, 7.0), first, 0.3);               // the reference has come down to it
}

// New: a latency kept for the lookahead. The lookahead then moves inside it
// without the latency changing, and still does its job: with 5 ms of it the
// gain is down before a burst arrives; with none the first peak gets through.
DSPARK_TEST(Compressor_lookahead_moves_inside_a_latency_of_its_own)
{
    using C = Compressor<float>;
    const int n = 14400, starts = 4800;
    int latency = 0;
    auto firstPeak = [&](float lookaheadMs, int& firstSound) {
        C c;
        c.setThreshold(-20.0f);
        c.setRatio(8.0f);
        c.setAttack(1.0f);
        c.setRelease(100.0f);
        c.setMaxLookahead(10.0f);
        c.setLookahead(lookaheadMs);
        c.prepare(AudioSpec{ 48000.0, 512, 1 });
        latency = c.getLatency();
        Vec x = makeSine(1000.0, 0.5, 48000.0, n);
        std::fill(x.begin(), x.begin() + starts, 0.0f);
        std::vector<Vec> v { x };
        processInBlocks(v, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        firstSound = 0;
        while (firstSound < n && v[0][static_cast<size_t>(firstSound)] == 0.0f) ++firstSound;
        return peakOf(v[0].data() + starts + 480, 48);      // the first cycle of the burst
    };
    int soundsLate = 0, soundsNone = 0;
    const double late = firstPeak(5.0f, soundsLate), none = firstPeak(0.0f, soundsNone);
    EXPECT_EQ(latency, 480);
    EXPECT_TRUE(soundsLate == starts + 480 || soundsLate == starts + 481);    // the burst, one latency late
    EXPECT_EQ(soundsNone, soundsLate);
    EXPECT_LT(late, 0.2);
    EXPECT_GT(none, 0.3);

    C c;
    c.setLookahead(5.0f);
    c.prepare(AudioSpec{ 48000.0, 512, 1 });
    EXPECT_EQ(c.getLatency(), 240);                         // none kept: the latency is the lookahead
    c.setMaxLookahead(10.0f);
    EXPECT_EQ(c.getLatency(), 480);
    c.setLookahead(1.0f);
    EXPECT_EQ(c.getLatency(), 480);
    c.setMaxLookahead(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(static_cast<double>(c.getMaxLookahead()), 10.0, 1e-6);
    c.setTopology(C::Topology::FeedBack);
    EXPECT_EQ(c.getLatency(), 0);
}

// New: the amount, a scale of the gain change in decibels. Half of it is half
// the reduction at every instant; none of it is the input, bit for bit; and
// in FeedBack the loop keeps the whole reduction while half is heard.
DSPARK_TEST(Compressor_amount_scales_the_reduction_in_decibels)
{
    using C = Compressor<float>;
    for (bool feedback : { false, true })
    {
        auto with = [&](float amount) {
            return compHeldReduction(0.5, 1.0, [&](C& c) {
                c.setThreshold(-24.0f);
                c.setAmount(amount);
                if (feedback) c.setTopology(C::Topology::FeedBack);
            });
        };
        const double whole = with(1.0f);
        EXPECT_LT(whole, -4.0);
        EXPECT_NEAR(with(0.5f), 0.5 * whole, 0.05);
        EXPECT_NEAR(with(0.25f), 0.25 * whole, 0.05);
    }

    C c;
    c.setThreshold(-24.0f);
    c.setAmount(0.0f);
    c.prepare(AudioSpec{ 48000.0, 512, 2 });
    const Vec x = makeSine(1000.0, 0.5, 48000.0, 9600);
    std::vector<Vec> v { x, x };
    processInBlocks(v, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
    EXPECT_TRUE(std::memcmp(v[0].data(), x.data(), sizeof(float) * x.size()) == 0);
    c.setAmount(std::numeric_limits<float>::quiet_NaN());
    c.setAmount(7.0f);
    EXPECT_NEAR(static_cast<double>(c.getAmount()), 1.0, 1e-6);
}

// New: a range the gain change approaches without a corner. A target of the
// size of the range comes out 16 % short of it, half of it 1.5 % short, and
// far over it, on it; without it the range is a cut.
DSPARK_TEST(Compressor_soft_range_approaches_the_range_without_a_corner)
{
    using C = Compressor<float>;
    // A 1 kHz tone of amplitude 0.5 read by the RMS detector is at -9.03 dB.
    auto reduction = [](float threshold, bool soft) {
        return compHeldReduction(0.5, 1.0, [&](C& c) {
            c.setDetector(C::DetectorType::Rms);
            c.setRatio(2.0f);
            c.setThreshold(threshold);
            c.setRange(6.0f);
            c.setSoftRange(soft);
        });
    };
    EXPECT_NEAR(reduction(-21.03f, false), -6.0, 0.05);     // a target of 6 dB: on the range
    EXPECT_NEAR(reduction(-21.03f, true), -5.045, 0.05);
    EXPECT_NEAR(reduction(-15.03f, false), -3.0, 0.05);     // of 3 dB
    EXPECT_NEAR(reduction(-15.03f, true), -2.955, 0.05);
    EXPECT_NEAR(reduction(-60.0f, false), -6.0, 0.05);      // of 25 dB
    EXPECT_NEAR(reduction(-60.0f, true), -5.995, 0.05);
}

// The hold freezes the release and leaves the attack free: a louder passage
// during a hold deepens the reduction at once, and it is then kept for the
// hold time after the sound stops.
DSPARK_TEST(Compressor_hold_freezes_the_release_and_leaves_the_attack_free)
{
    using C = Compressor<float>;
    C c;
    c.setThreshold(-30.0f);
    c.setRatio(4.0f);
    c.setAttack(1.0f);
    c.setRelease(50.0f);
    c.setHoldTime(200.0f);
    c.prepare(AudioSpec{ 48000.0, 512, 1 });
    auto play = [&](double amplitude, double seconds) {
        std::vector<Vec> v { makeSine(1000.0, amplitude, 48000.0, static_cast<int>(seconds * 48000.0)) };
        processInBlocks(v, 480, [&](AudioBufferView<float> b) { c.processBlock(b); });
        return static_cast<double>(c.getGainReductionDb());
    };
    const double low = play(0.1, 0.25);                     // 10 dB over: 7.5 dB at the peaks
    EXPECT_LT(low, -5.0);
    EXPECT_GT(low, -8.0);
    const double high = play(0.5, 0.02);                    // 24 dB over, during the hold
    EXPECT_LT(high, -14.0);
    EXPECT_LT(play(0.0, 0.15), -14.0);                      // still held
    EXPECT_GT(play(0.0, 0.5), -1.0);                        // let go
}

// The new controls keep what the class guarantees: one channel through
// processSample() is the block path bit for bit, in both topologies, and the
// state carries them.
DSPARK_TEST(Compressor_new_controls_sample_path_and_state)
{
    using C = Compressor<float>;
    const int n = 24000;
    Vec a(static_cast<size_t>(n));
    uint32_t seed = 3u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double t = i / 48000.0;
        const double env = 0.1 + 0.7 * std::exp(-std::fmod(t, 0.2) * 20.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.6 * std::sin(2.0 * kPiD * 440.0 * t)
                                  + 0.1 * (static_cast<double>(seed) / 4294967296.0 - 0.5)));
    }
    for (bool feedback : { false, true })
    {
        auto make = [&](C& c) {
            c.setThreshold(-10.0f);
            c.setRatio(4.0f);
            c.setAttack(2.0f);
            c.setRelease(80.0f);
            c.setRange(9.0f);
            c.setSoftRange(true);
            c.setAutoThreshold(true);
            c.setAmount(0.6f);
            if (feedback) c.setTopology(C::Topology::FeedBack);
            c.prepare(AudioSpec{ 48000.0, 512, 1 });
        };
        C block, single;
        make(block);
        make(single);
        std::vector<Vec> x { a };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { block.processBlock(b); });
        Vec y = a;
        for (auto& s : y) s = single.processSample(s, 0);
        double md = 0.0;
        for (int i = 0; i < n; ++i)
            md = std::max(md, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[static_cast<size_t>(i)]));
        EXPECT_EQ(md, 0.0);
        double moved = 0.0;
        for (int i = 0; i < n; ++i)
            moved = std::max(moved, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - a[static_cast<size_t>(i)]));
        EXPECT_GT(moved, 0.01);
    }

    C c;
    c.setAutoThreshold(true);
    c.setMaxLookahead(8.0f);
    c.setLookahead(3.0f);
    c.setAmount(0.4f);
    c.setSoftRange(true);
    const auto blob = c.getState();
    C other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.isAutoThreshold());
    EXPECT_TRUE(other.isSoftRange());
    EXPECT_NEAR(static_cast<double>(other.getAmount()), 0.4, 1e-6);
    EXPECT_NEAR(static_cast<double>(other.getMaxLookahead()), 8.0, 1e-6);
    EXPECT_TRUE(other.getState() == blob);
}

// ============================================================================
// Compressor: analytic makeup, precision of the control signal, slow followers
// ============================================================================

namespace {

// Gaussian noise with equal power per octave, at an RMS level in dBFS.
Vec compPinkNoise(int n, double rmsDb, uint32_t seed)
{
    std::vector<double> v(static_cast<size_t>(n));
    uint32_t s = seed;
    auto uni = [&]() { s = s * 1664525u + 1013904223u; return (static_cast<double>(s >> 8) + 0.5) / 16777216.0; };
    double b0 = 0.0, b1 = 0.0, b2 = 0.0, sum = 0.0;
    for (auto& x : v)
    {
        const double g = std::sqrt(-2.0 * std::log(uni())) * std::cos(2.0 * kPiD * uni());
        b0 = 0.99765 * b0 + g * 0.0990460;
        b1 = 0.96300 * b1 + g * 0.2965164;
        b2 = 0.57000 * b2 + g * 1.0526913;
        x = b0 + b1 + b2 + g * 0.1848;
        sum += x * x;
    }
    const double scale = std::pow(10.0, rmsDb / 20.0) / std::sqrt(sum / n);
    Vec out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) out[static_cast<size_t>(i)] = static_cast<float>(v[static_cast<size_t>(i)] * scale);
    return out;
}

// Level a compressor leaves over the input's, in dB, past the first quarter of the signal.
template <class Setup>
double compLevelChange(const Vec& in, Setup&& setup)
{
    auto c = std::make_unique<Compressor<float>>();
    setup(*c);
    c->prepare(AudioSpec{ 48000.0, 512, 1 });
    std::vector<Vec> x { in };
    processInBlocks(x, 512, [&](AudioBufferView<float> b) { c->processBlock(b); });
    double before = 0.0, after = 0.0;
    for (size_t i = in.size() / 4; i < in.size(); ++i)
    {
        before += static_cast<double>(in[i]) * in[i];
        after += static_cast<double>(x[0][i]) * x[0][i];
    }
    return 10.0 * std::log10(after / before);
}

// The largest line of a compressor's steady output that is no harmonic of the tone it is fed, and
// the gain of the tone, in dB. The tone sits on bin `bin` of a 32768-point transform.
struct CompLines
{
    double gainDb = 0.0, worstDb = 0.0;
};

// The gain of a bin-locked tone through a processor and the largest line of
// its output that is no harmonic of it (dB re the tone), over the last of six
// 32768-sample spans.
template <class Processor, class Setup>
CompLines toneLines(int bin, double amplitude, Setup&& setup)
{
    constexpr int n = 32768;
    auto c = std::make_unique<Processor>();
    setup(*c);
    c->prepare(AudioSpec{ 48000.0, 512, 1 });
    std::vector<double> x(static_cast<size_t>(n) * 6);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = amplitude * std::sin(2.0 * kPiD * bin * static_cast<double>(i % static_cast<size_t>(n)) / n);
    for (size_t off = 0; off < x.size(); off += 512)
    {
        double* p = x.data() + off;
        AudioBufferView<double> v(&p, 1, 512);
        c->processBlock(v);
    }
    FFTReal<double> fft(static_cast<size_t>(n));
    std::vector<double> spec(static_cast<size_t>(n) + 2);
    fft.forward(x.data() + (x.size() - static_cast<size_t>(n)), spec.data());
    auto power = [&](int k) {
        const double re = spec[static_cast<size_t>(2 * k)], im = spec[static_cast<size_t>(2 * k + 1)];
        return re * re + im * im;
    };
    double worst = 0.0;
    for (int k = 1; k <= n / 2; ++k)
        if (k % bin != 0) worst = std::max(worst, power(k));
    CompLines r;
    r.gainDb = 10.0 * std::log10(power(bin)) - 20.0 * std::log10(amplitude * n / 2.0);
    r.worstDb = 10.0 * std::log10(std::max(worst / power(bin), 1e-30));
    return r;
}

template <class Setup>
CompLines compLines(int bin, double amplitude, Setup&& setup)
{
    return toneLines<Compressor<double>>(bin, amplitude, std::forward<Setup>(setup));
}

} // namespace

// New: a makeup worked out from the settings. Over dense programme at the
// level it assumes, it leaves the output where the input was, whatever the
// curve, the times, the detector or the character; the constant half of the
// reduction of a full-scale signal (Static) misses by several dB as soon as
// the attack is slow.
DSPARK_TEST(Compressor_analytic_makeup_leaves_the_level_of_dense_programme)
{
    using C = Compressor<float>;
    using D = C::DetectorType;
    using K = C::Character;
    using T = C::Topology;
    const Vec pink = compPinkNoise(48000 * 8, -18.0, 7u);
    struct Case
    {
        const char* name;
        D det; K character; T topology; bool decoupled;
        float threshold, ratio, knee, attack, release;
        bool upward;
        double least;       // level change with no makeup, at least this far from 0
        double tolerance;   // level left with the analytic makeup
    };
    const Case cases[] = {
        { "peak 5/100 -20 4:1",          D::Peak,     K::Clean,  T::FeedForward, false, -20, 4, 0, 5, 100, false, 3.5, 0.3 },
        { "peak 5/100 -30 4:1",          D::Peak,     K::Clean,  T::FeedForward, false, -30, 4, 0, 5, 100, false, 9.0, 0.3 },
        { "peak 0.5/100 -20 4:1",        D::Peak,     K::Clean,  T::FeedForward, false, -20, 4, 0, 0.5f, 100, false, 5.0, 0.3 },
        { "peak 100/100 -20 4:1",        D::Peak,     K::Clean,  T::FeedForward, false, -20, 4, 0, 100, 100, false, 0.8, 0.3 },
        { "peak 5/1000 -20 4:1 knee 12", D::Peak,     K::Clean,  T::FeedForward, false, -20, 4, 12, 5, 1000, false, 5.0, 0.3 },
        { "peak decoupled 5/100",        D::Peak,     K::Clean,  T::FeedForward, true,  -20, 4, 0, 5, 100, false, 7.0, 0.6 },
        { "peak feedback 5/100",         D::Peak,     K::Clean,  T::FeedBack,    false, -20, 4, 0, 5, 100, false, 3.5, 0.5 },
        { "rms 30/300 -30 3:1",          D::Rms,      K::Clean,  T::FeedForward, false, -30, 3, 6, 30, 300, false, 7.0, 0.6 },
        { "hilbert 5/100 -20 4:1",       D::Hilbert,  K::Clean,  T::FeedForward, false, -20, 4, 0, 5, 100, false, 5.0, 0.5 },
        { "true peak 5/100 -20 4:1",     D::TruePeak, K::Clean,  T::FeedForward, false, -20, 4, 0, 5, 100, false, 4.5, 0.6 },
        { "opto 10/100 -25 4:1",         D::Peak,     K::Opto,   T::FeedForward, false, -25, 4, 0, 10, 100, false, 5.0, 0.4 },
        { "fet 0.2/200 -20 4:1",         D::Peak,     K::FET,    T::FeedBack,    false, -20, 4, 0, 0.2f, 200, false, 6.0, 0.7 },
        { "varimu 10/300 -25 3:1",       D::Peak,     K::Varimu, T::FeedForward, false, -25, 3, 0, 10, 300, false, 7.0, 0.4 },
        { "upward 5/100 -10 2:1",        D::Peak,     K::Clean,  T::FeedForward, false, -10, 2, 0, 5, 100, true,  2.0, 0.3 },
    };
    for (const auto& k : cases)
    {
        const auto with = [&](C::AutoMakeupMode mode) {
            return compLevelChange(pink, [&](C& c) {
                c.setDetector(k.det);
                c.setCharacter(k.character);
                c.setTopology(k.topology);
                c.setCleanBallistics(k.decoupled ? C::CleanBallistics::Decoupled : C::CleanBallistics::Branching);
                c.setThreshold(k.threshold);
                c.setRatio(k.ratio);
                c.setKnee(k.knee);
                c.setAttack(k.attack);
                c.setRelease(k.release);
                c.setMode(k.upward ? C::Mode::Upward : C::Mode::Downward);
                c.setAutoMakeup(mode);
            });
        };
        const double none = with(C::AutoMakeupMode::Off);
        const double analytic = with(C::AutoMakeupMode::Analytic);
        std::cout << "  " << k.name << ": " << none << " dB with no makeup, " << analytic << " dB with the analytic one\n";
        EXPECT_GT(std::abs(none), k.least);
        EXPECT_LT(std::abs(analytic), k.tolerance);
    }

    // The constant of the Static mode does not know the times.
    const auto slow = [&](C::AutoMakeupMode mode) {
        return compLevelChange(pink, [&](C& c) {
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(100.0f);
            c.setRelease(100.0f);
            c.setAutoMakeup(mode);
        });
    };
    EXPECT_GT(slow(C::AutoMakeupMode::Static), 5.0);
    EXPECT_LT(std::abs(slow(C::AutoMakeupMode::Analytic)), 0.3);

    // It follows the amount, and a change of setting moves it without a step.
    const double half = compLevelChange(pink, [&](C& c) {
        c.setThreshold(-30.0f);
        c.setRatio(4.0f);
        c.setAmount(0.5f);
        c.setAutoMakeup(C::AutoMakeupMode::Analytic);
    });
    EXPECT_LT(std::abs(half), 0.4);

    C moving;
    moving.setThreshold(-10.0f);
    moving.setRatio(4.0f);
    moving.setAutoMakeup(C::AutoMakeupMode::Analytic);
    moving.prepare(AudioSpec{ 48000.0, 512, 1 });
    std::vector<Vec> quiet { makeSine(1000.0, 0.001, 48000.0, 48000) };
    float last = 0.0f, largestStep = 0.0f;
    bool first = true;
    int block = 0;
    processInBlocks(quiet, 512, [&](AudioBufferView<float> b) {
        if (block++ == 20) moving.setThreshold(-40.0f);   // the makeup grows by several dB
        moving.processBlock(b);
    });
    for (int i = 12; i < 48000; i += 48)                   // the crests of the tone
    {
        const float crest = std::abs(quiet[0][static_cast<size_t>(i)]);
        if (!first) largestStep = std::max(largestStep, std::abs(crest - last) / std::max(last, 1e-9f));
        last = crest;
        first = false;
    }
    EXPECT_GT(std::abs(quiet[0][47964]) / std::abs(quiet[0][12]), 1.5f);   // it did move
    EXPECT_LT(largestStep, 0.15f);                                          // 1 ms at a time, never a step
}

// With the automatic threshold the analytic makeup knows where the programme
// sits against the threshold: the same result 30 dB lower.
DSPARK_TEST(Compressor_analytic_makeup_with_automatic_threshold_holds_at_any_level)
{
    using C = Compressor<float>;
    double left[2] = {}, lost[2] = {};
    const double levels[2] = { -36.0, -6.0 };
    for (int which = 0; which < 2; ++which)
    {
        const Vec pink = compPinkNoise(48000 * 8, levels[which], 7u);
        const auto with = [&](C::AutoMakeupMode mode) {
            return compLevelChange(pink, [&](C& c) {
                c.setThreshold(-20.0f);
                c.setRatio(3.0f);
                c.setKnee(6.0f);
                c.setAttack(20.0f);
                c.setRelease(200.0f);
                c.setAutoThreshold(true);
                c.setAutoMakeup(mode);
            });
        };
        lost[which] = with(C::AutoMakeupMode::Off);
        left[which] = with(C::AutoMakeupMode::Analytic);
        std::cout << "  programme at " << levels[which] << " dBFS: " << lost[which] << " dB, " << left[which]
                  << " dB with the analytic makeup\n";
        EXPECT_LT(lost[which], -5.0);
        EXPECT_LT(std::abs(left[which]), 0.4);
    }
    EXPECT_NEAR(left[0], left[1], 0.1);
}

// Defect: an envelope within 1e-4 dB of zero was put back on zero whatever
// its target. Leaving zero in steps shorter than that (a slow attack, a
// shallow target), each step was undone and the gain never moved: at 96 kHz,
// 200 ms of attack ignored a 0.3 dB target for good. And the memory stage of
// Opto, which charges over two seconds with a release of 500 ms, never
// charged under 9.6 dB of reduction at 48 kHz, so the long tail that makes
// the character was missing on light compression.
DSPARK_TEST(Compressor_slow_followers_leave_zero)
{
    using C = Compressor<float>;
    for (double rate : { 48000.0, 96000.0 })
    {
        C c;
        c.setDetector(C::DetectorType::Rms);
        c.setThreshold(-20.0f);
        c.setRatio(2.0f);
        c.setAttack(200.0f);
        c.setRelease(200.0f);
        c.prepare(AudioSpec{ rate, 512, 1 });
        // RMS 0.6 dB over the threshold: the curve asks for 0.3 dB.
        const double amplitude = std::pow(10.0, -19.4 / 20.0) * std::sqrt(2.0);
        std::vector<Vec> x { makeSine(1000.0, amplitude, rate, static_cast<int>(rate * 4.0)) };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        EXPECT_NEAR(static_cast<double>(c.getGainReductionDb()), -0.30, 0.02);
    }

    C opto;
    opto.setCharacter(C::Character::Opto);
    opto.setDetector(C::DetectorType::Rms);
    opto.setThreshold(-20.0f);
    opto.setRatio(2.0f);
    opto.setRelease(500.0f);
    opto.prepare(AudioSpec{ 48000.0, 512, 1 });
    const double amplitude = std::pow(10.0, -14.0 / 20.0) * std::sqrt(2.0);     // 6 dB over
    std::vector<Vec> passage { makeSine(1000.0, amplitude, 48000.0, 48000 * 10) };
    processInBlocks(passage, 512, [&](AudioBufferView<float> b) { opto.processBlock(b); });
    EXPECT_NEAR(static_cast<double>(opto.getGainReductionDb()), -3.0, 0.1);
    std::vector<Vec> pause { Vec(48000, 0.0f) };
    processInBlocks(pause, 512, [&](AudioBufferView<float> b) { opto.processBlock(b); });
    // One second on, the fast stage is back (0.07 dB left of it); what holds is the memory.
    EXPECT_LT(static_cast<double>(opto.getGainReductionDb()), -0.7);
    EXPECT_GT(static_cast<double>(opto.getGainReductionDb()), -1.5);
}

// New: Precision::High works the control signal out between the samples. The
// lines it leaves that are no harmonics of the tone fall by tens of dB, most
// where the rectifier folds hardest (a tone near the top of the band), and
// the level the detector reads stays where it was. An average of the
// rectifier over consecutive samples, tried first, read such a tone up to
// 10 dB low.
DSPARK_TEST(Compressor_high_precision_lowers_the_lines_that_are_no_harmonics)
{
    using C = Compressor<double>;
    const double amplitude = std::pow(10.0, -8.0 / 20.0);
    struct Case { const char* name; bool fet; double threshold, ratio, attack; };
    const Case cases[] = {
        { "peak, 0.1 ms, 4:1, 12 dB over", false, -20.0, 4.0, 0.1 },
        { "FET, 20 us, 20:1, 30 dB over",  true,  -38.0, 20.0, 0.02 },
    };
    for (const auto& k : cases)
    {
        for (int bin : { 683, 12289 })                       // 1.0 and 18.0 kHz
        {
            const auto at = [&](C::Precision precision) {
                return compLines(bin, amplitude, [&](C& c) {
                    if (k.fet) c.setCharacter(C::Character::FET);
                    c.setThreshold(k.threshold);
                    c.setRatio(k.ratio);
                    c.setAttack(k.attack);
                    c.setRelease(50.0);
                    c.setPrecision(precision);
                });
            };
            const CompLines standard = at(C::Precision::Standard);
            const CompLines high = at(C::Precision::High);
            std::cout << "  " << k.name << ", " << 48000.0 * bin / 32768.0 << " Hz: " << standard.worstDb << " -> "
                      << high.worstDb << " dBc, gain " << standard.gainDb << " -> " << high.gainDb << " dB\n";
            EXPECT_LT(high.worstDb, standard.worstDb - 15.0);
            EXPECT_LT(high.worstDb, -60.0);
            EXPECT_NEAR(high.gainDb, standard.gainDb, 0.15);
        }
    }

    // The detectors whose reading is already smooth keep one level per sample and gain the steps.
    const auto hilbert = [&](C::Precision precision) {
        return compLines(683, amplitude, [&](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0);
            c.setAttack(0.1);
            c.setRelease(50.0);
            c.setPrecision(precision);
        });
    };
    const CompLines smooth = hilbert(C::Precision::Standard), smoother = hilbert(C::Precision::High);
    EXPECT_LT(smoother.worstDb, smooth.worstDb - 10.0);
    EXPECT_NEAR(smoother.gainDb, smooth.gainDb, 0.01);
}

// The detectors read between the samples seven samples late, and the audio
// waits for them: the latency reported is the delay measured, with and
// without lookahead, in feed-forward and with the FET character; the
// detectors that keep one level per sample add none.
DSPARK_TEST(Compressor_high_precision_latency_paths_and_state)
{
    using C = Compressor<float>;
    // Where the impulse comes out (-1: nowhere) and how many other samples are not silence.
    const auto impulseAt = [](C& c, int channels) {
        c.setThreshold(0.0f);
        c.setPrecision(C::Precision::High);
        c.prepare(AudioSpec{ 48000.0, 512, channels });
        std::vector<Vec> x(static_cast<size_t>(channels), Vec(2048, 0.0f));
        for (auto& ch : x) ch[100] = 0.5f;
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        int found = -1;
        for (int i = 0; i < 2048; ++i)
        {
            if (x[0][static_cast<size_t>(i)] == 0.5f) found = i - 100;
            else if (x[0][static_cast<size_t>(i)] != 0.0f) return -2;
        }
        return found;
    };
    {
        C c;
        EXPECT_EQ(impulseAt(c, 1), 7);
        EXPECT_EQ(c.getLatency(), 7);
    }
    {
        C c;
        c.setLookahead(2.0f);
        EXPECT_EQ(impulseAt(c, 2), 96 + 7);
        EXPECT_EQ(c.getLatency(), 96 + 7);
    }
    {
        C c;
        c.setMaxLookahead(5.0f);
        c.setLookahead(1.0f);
        EXPECT_EQ(impulseAt(c, 1), 240 + 7);
        EXPECT_EQ(c.getLatency(), 240 + 7);
    }
    {
        C c;
        c.setCharacter(C::Character::FET);
        EXPECT_EQ(impulseAt(c, 1), 7);
        EXPECT_EQ(c.getLatency(), 7);
    }
    {
        C c;
        c.setDetector(C::DetectorType::TruePeak);
        EXPECT_EQ(impulseAt(c, 1), 7);
        EXPECT_EQ(c.getLatency(), 7);
    }
    for (auto det : { C::DetectorType::Hilbert, C::DetectorType::Rms, C::DetectorType::SplitPolarity })
    {
        C c;
        c.setDetector(det);
        EXPECT_EQ(impulseAt(c, 1), 0);
        EXPECT_EQ(c.getLatency(), 0);
    }
    {
        C c;                                    // a feedback loop that is not solved against the input
        c.setTopology(C::Topology::FeedBack);
        c.setDetector(C::DetectorType::Rms);
        EXPECT_EQ(impulseAt(c, 1), 0);
        EXPECT_EQ(c.getLatency(), 0);
    }

    // Linked channels share one gain; unlinked, a channel under the threshold is left alone.
    const int n = 24000;
    const auto stereo = [&](float link, std::vector<Vec>& x) {
        C c;
        c.setThreshold(-20.0f);
        c.setRatio(4.0f);
        c.setAttack(1.0f);
        c.setRelease(50.0f);
        c.setStereoLink(link);
        c.setPrecision(C::Precision::High);
        c.prepare(AudioSpec{ 48000.0, 512, 2 });
        x = { makeSine(1000.0, 0.5, 48000.0, n), makeSine(700.0, 0.01, 48000.0, n) };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
    };
    std::vector<Vec> linked, apart;
    stereo(1.0f, linked);
    stereo(0.0f, apart);
    const Vec quiet = makeSine(700.0, 0.01, 48000.0, n);
    double touched = 0.0, alone = 0.0;
    for (int i = 12000; i < n; ++i)
    {
        touched = std::max(touched, std::abs(static_cast<double>(linked[1][static_cast<size_t>(i)])));
        alone = std::max(alone, std::abs(static_cast<double>(apart[1][static_cast<size_t>(i)])
                                         - quiet[static_cast<size_t>(i - 7)]));
    }
    EXPECT_LT(touched, 0.006);                  // the loud channel's reduction, on the quiet one
    EXPECT_GT(touched, 0.003);
    EXPECT_LT(alone, 1e-6);                     // itself, seven samples late
    double same = 0.0;
    for (int i = 12000; i < n; ++i)
        same = std::max(same, std::abs(static_cast<double>(linked[0][static_cast<size_t>(i)])
                                       - apart[0][static_cast<size_t>(i)]));
    EXPECT_LT(same, 1e-4);                      // the loud channel drives both ways

    // processSample() stays at Standard: there is no delay line for the audio there.
    C single, plain;
    for (C* c : { &single, &plain })
    {
        c->setThreshold(-20.0f);
        c->setAttack(0.5f);
        c->prepare(AudioSpec{ 48000.0, 512, 1 });
    }
    single.setPrecision(C::Precision::High);
    const Vec tone = makeSine(3000.0, 0.5, 48000.0, 4800);
    double apartBy = 0.0;
    for (float s : tone)
        apartBy = std::max(apartBy, std::abs(static_cast<double>(single.processSample(s, 0)) - plain.processSample(s, 0)));
    EXPECT_EQ(apartBy, 0.0);

    // State and wild values.
    C c;
    c.setPrecision(C::Precision::High);
    c.setAutoMakeup(C::AutoMakeupMode::Analytic);
    const auto blob = c.getState();
    C other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.getPrecision() == C::Precision::High);
    EXPECT_TRUE(other.getState() == blob);
    other.setPrecision(static_cast<C::Precision>(99));
    EXPECT_TRUE(other.getPrecision() == C::Precision::High);
    other.setAutoMakeup(static_cast<C::AutoMakeupMode>(99));
    C third;
    const auto wild = other.getState();
    EXPECT_TRUE(third.setState(wild.data(), wild.size()));
    EXPECT_TRUE(third.getState() == wild);
}

// ============================================================================
// Compressor: timing that follows the programme, and the Hybrid detector
// ============================================================================

namespace {

// The gain reduction a compressor shows every 16 samples over a signal.
template <class Setup>
std::vector<double> compTrace(const Vec& in, Setup&& setup)
{
    Compressor<float> c;
    setup(c);
    c.prepare(AudioSpec{ 48000.0, 16, 1 });
    std::vector<Vec> x { in };
    std::vector<double> trace;
    processInBlocks(x, 16, [&](AudioBufferView<float> b) {
        c.processBlock(b);
        trace.push_back(static_cast<double>(c.getGainReductionDb()));
    });
    return trace;
}

// Milliseconds after sample `from` for the reduction to come back to 1/e of what it was there.
double compReleaseMs(const std::vector<double>& trace, int from)
{
    const size_t at = static_cast<size_t>(from / 16);
    const double start = std::abs(trace[at - 1]);
    for (size_t i = at; i < trace.size(); ++i)
        if (std::abs(trace[i]) <= start * 0.36788) return static_cast<double>(i - at + 1) * 16.0 * 1000.0 / 48000.0;
    return 1e9;
}

// Milliseconds after sample `from` for the reduction to cover 63 % of the way from what it was
// there to `settled`.
double compAttackMs(const std::vector<double>& trace, int from, double settled)
{
    const size_t at = static_cast<size_t>(from / 16);
    const double start = trace[at - 1];
    const double mark = start + 0.63212 * (settled - start);
    for (size_t i = at; i < trace.size(); ++i)
        if (trace[i] <= mark) return static_cast<double>(i - at + 1) * 16.0 * 1000.0 / 48000.0;
    return 1e9;
}

// A tone that lasts `seconds`, then silence up to `total` seconds.
Vec compToneThenSilence(double hz, double amplitude, double seconds, double total)
{
    Vec v = makeSine(hz, amplitude, 48000.0, static_cast<int>(total * 48000.0));
    for (size_t i = static_cast<size_t>(seconds * 48000.0); i < v.size(); ++i) v[i] = 0.0f;
    return v;
}

} // namespace

// New: a release that knows how long the compressor has been working. After
// 30 ms of reduction it lets go at the knob's time; after three seconds, six
// to eight times slower. At zero the two are the same release.
DSPARK_TEST(Compressor_program_release_follows_how_long_the_reduction_lasted)
{
    using C = Compressor<float>;
    const auto release = [&](double seconds, float amount) {
        const Vec in = compToneThenSilence(1000.0, 0.4, seconds, seconds + 2.5);
        const auto trace = compTrace(in, [&](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(100.0f);
            c.setProgramRelease(amount);
        });
        return compReleaseMs(trace, static_cast<int>(seconds * 48000.0));
    };
    const double briefFixed = release(0.03, 0.0f), longFixed = release(3.0, 0.0f);
    const double briefProgram = release(0.03, 1.0f), longProgram = release(3.0, 1.0f);
    const double longHalf = release(3.0, 0.5f);
    std::cout << "  release after 30 ms / 3 s: fixed " << briefFixed << " / " << longFixed << " ms, programme "
              << briefProgram << " / " << longProgram << " ms (half weight " << longHalf << " ms)\n";
    EXPECT_NEAR(briefFixed, 100.0, 6.0);
    EXPECT_NEAR(longFixed, 100.0, 6.0);
    EXPECT_LT(briefProgram, 125.0);
    EXPECT_GT(longProgram, 600.0);
    EXPECT_LT(longProgram, 820.0);
    EXPECT_GT(longHalf, 300.0);
    EXPECT_LT(longHalf, longProgram - 150.0);
}

// New: an attack that depends on how far the target is. A step of 1 dB over
// a settled reduction is taken several times slower than the knob says; a
// step of 12 dB is caught nearly at the knob's time.
DSPARK_TEST(Compressor_adaptive_attack_takes_a_small_step_slowly_and_a_large_one_fast)
{
    using C = Compressor<float>;
    const int n = 48000 * 3, from = 48000;
    const auto attack = [&](double stepDb, float amount) {
        Vec in = makeSine(1000.0, 0.1, 48000.0, n);
        const float up = static_cast<float>(std::pow(10.0, stepDb / 20.0));
        for (int i = from; i < n; ++i) in[static_cast<size_t>(i)] *= up;
        const auto trace = compTrace(in, [&](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-30.0f);
            c.setRatio(4.0f);
            c.setAttack(20.0f);
            c.setRelease(200.0f);
            c.setAdaptiveAttack(amount);
        });
        return compAttackMs(trace, from, trace.back());
    };
    const double smallFixed = attack(1.0, 0.0f), largeFixed = attack(12.0, 0.0f);
    const double smallAdaptive = attack(1.0, 1.0f), largeAdaptive = attack(12.0, 1.0f);
    std::cout << "  attack to 63 % of a 1 dB / 12 dB step: fixed " << smallFixed << " / " << largeFixed
              << " ms, adaptive " << smallAdaptive << " / " << largeAdaptive << " ms\n";
    EXPECT_NEAR(smallFixed, 20.0, 3.0);
    EXPECT_NEAR(largeFixed, 20.0, 3.0);
    EXPECT_GT(smallAdaptive, 2.5 * smallFixed);
    EXPECT_LT(smallAdaptive, 4.5 * smallFixed);
    EXPECT_LT(largeAdaptive, 1.8 * largeFixed);
    EXPECT_GT(largeAdaptive, largeFixed);
}

// New: times that follow the spectral balance of the key. The same second of
// a tone releases several times sooner at 6 kHz than at 100 Hz; with the
// weight at zero, at the same time.
DSPARK_TEST(Compressor_spectral_timing_is_fast_on_a_bright_key_and_slow_on_a_dark_one)
{
    using C = Compressor<float>;
    const auto release = [&](double hz, float amount) {
        const Vec in = compToneThenSilence(hz, 0.4, 1.0, 3.5);
        const auto trace = compTrace(in, [&](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(5.0f);
            c.setRelease(100.0f);
            c.setSpectralTiming(amount);
        });
        return compReleaseMs(trace, 48000);
    };
    const double darkFixed = release(100.0, 0.0f), brightFixed = release(6000.0, 0.0f);
    const double dark = release(100.0, 1.0f), bright = release(6000.0, 1.0f);
    std::cout << "  release after 100 Hz / 6 kHz: fixed " << darkFixed << " / " << brightFixed << " ms, spectral "
              << dark << " / " << bright << " ms\n";
    EXPECT_NEAR(darkFixed, brightFixed, 25.0);
    EXPECT_GT(dark, 240.0);
    EXPECT_LT(dark, 345.0);
    EXPECT_GT(bright, 25.0);
    EXPECT_LT(bright, 45.0);

    // Programme with the same power in every octave stays on the knobs.
    const Vec pink = compPinkNoise(48000 * 4, -14.0, 3u);
    Vec passage = pink;
    passage.resize(static_cast<size_t>(48000) * 6, 0.0f);
    const auto level = [&](float amount) {
        const auto trace = compTrace(passage, [&](C& c) {
            c.setDetector(C::DetectorType::Rms);
            c.setRmsWindow(50.0f);
            c.setThreshold(-30.0f);
            c.setRatio(4.0f);
            c.setRelease(100.0f);
            c.setSpectralTiming(amount);
        });
        return compReleaseMs(trace, 48000 * 4);
    };
    const double plain = level(0.0f), balanced = level(1.0f);
    std::cout << "  release after pink noise: " << plain << " ms fixed, " << balanced << " ms spectral\n";
    EXPECT_GT(balanced, plain * 0.75);
    EXPECT_LT(balanced, plain * 1.35);
}

// New: the emphasis of the attack. At +1 the leading edge of a reduction
// goes well past where it settles; at -1 it follows a smoothed course and
// lets the onset through; either way the settled gain is the same.
DSPARK_TEST(Compressor_attack_emphasis_deepens_or_softens_the_leading_edge)
{
    using C = Compressor<float>;
    Vec in = makeSine(1000.0, 0.4, 48000.0, 48000 * 2);
    for (int i = 0; i < 4800; ++i) in[static_cast<size_t>(i)] = 0.0f;
    const auto trace = [&](float emphasis) {
        return compTrace(in, [&](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(10.0f);
            c.setRelease(200.0f);
            c.setAttackEmphasis(emphasis);
        });
    };
    const auto plain = trace(0.0f), more = trace(1.0f), less = trace(-1.0f);
    const auto deepest = [](const std::vector<double>& t) {
        double d = 0.0;
        for (double v : t) d = std::min(d, v);
        return d;
    };
    const size_t early = static_cast<size_t>((4800 + 480) / 16);     // 10 ms after the onset
    std::cout << "  settled " << plain.back() << " dB; deepest " << deepest(plain) << " / " << deepest(more)
              << " dB plain / emphasised; 10 ms in: " << plain[early] << " / " << less[early] << " dB plain / softened\n";
    EXPECT_NEAR(more.back(), plain.back(), 0.02);
    EXPECT_NEAR(less.back(), plain.back(), 0.02);
    EXPECT_LT(deepest(more), deepest(plain) - 2.0);
    EXPECT_NEAR(deepest(less), deepest(plain), 0.05);
    EXPECT_GT(less[early], plain[early] + 2.0);
}

// New: the Hybrid detector. A steady tone reads its peak, with none of the
// ripple the rectifier leaves in the gain; an onset is read exactly as the
// Peak detector reads it until the running mean catches up.
DSPARK_TEST(Compressor_hybrid_detector_holds_a_tone_and_catches_an_onset)
{
    using C = Compressor<float>;
    const Vec tone = makeSine(1000.0, 0.4, 48000.0, 48000);
    const auto trace = [&](C::DetectorType det) {
        return compTrace(tone, [&](C& c) {
            c.setDetector(det);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(5.0f);
        });
    };
    const auto ripple = [](const std::vector<double>& t, double& mean) {
        double lo = 1e9, hi = -1e9, sum = 0.0;
        const size_t from = t.size() - 600;
        for (size_t i = from; i < t.size(); ++i) { lo = std::min(lo, t[i]); hi = std::max(hi, t[i]); sum += t[i]; }
        mean = sum / 600.0;
        return hi - lo;
    };
    double hybridMean = 0.0, peakMean = 0.0;
    const double hybridRipple = ripple(trace(C::DetectorType::Hybrid), hybridMean);
    const double peakRipple = ripple(trace(C::DetectorType::Peak), peakMean);
    const double onTheCurve = -0.75 * (20.0 * std::log10(0.4) + 20.0);
    std::cout << "  1 kHz tone: hybrid " << hybridMean << " dB, ripple " << hybridRipple << " dB; peak " << peakMean
              << " dB, ripple " << peakRipple << " dB; the curve asks " << onTheCurve << " dB\n";
    EXPECT_NEAR(hybridMean, onTheCurve, 0.1);
    EXPECT_LT(hybridRipple, 0.08);
    EXPECT_GT(peakRipple, 0.15);
    EXPECT_GT(peakRipple, 3.0 * hybridRipple);

    // An onset: the first samples come out as with the Peak detector.
    Vec onset(4800, 0.0f);
    for (int i = 2400; i < 4800; ++i)
        onset[static_cast<size_t>(i)] = static_cast<float>(0.9 * std::sin(2.0 * kPiD * 1000.0 * (i - 2400) / 48000.0));
    const auto out = [&](C::DetectorType det) {
        C c;
        c.setDetector(det);
        c.setThreshold(-20.0f);
        c.setRatio(8.0f);
        c.setAttack(0.1f);
        c.prepare(AudioSpec{ 48000.0, 512, 1 });
        std::vector<Vec> x { onset };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        return x[0];
    };
    const Vec hybrid = out(C::DetectorType::Hybrid), peak = out(C::DetectorType::Peak);
    double apart = 0.0;
    for (int i = 2400; i < 2410; ++i)
        apart = std::max(apart, std::abs(static_cast<double>(hybrid[static_cast<size_t>(i)]) - peak[static_cast<size_t>(i)]));
    EXPECT_LT(apart, 1e-6);
    EXPECT_GT(std::abs(peak[2409]), 0.05f);     // and the reduction had begun
    EXPECT_LT(std::abs(peak[2409]), 0.8f);
}

// The programme-dependent timing keeps what the class guarantees: one
// channel through processSample() is the block path bit for bit, the precise
// path honours it, the state carries it, and where it does not apply (Opto,
// FET, feedback, Decoupled) the output is the one without it.
DSPARK_TEST(Compressor_program_timing_paths_scope_and_state)
{
    using C = Compressor<float>;
    const int n = 24000;
    Vec a(static_cast<size_t>(n));
    uint32_t seed = 9u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double t = i / 48000.0;
        const double env = 0.1 + 0.7 * std::exp(-std::fmod(t, 0.2) * 20.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.6 * std::sin(2.0 * kPiD * 440.0 * t)
                                  + 0.2 * (static_cast<double>(seed) / 4294967296.0 - 0.5)));
    }
    const auto timed = [](C& c) {
        c.setThreshold(-18.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(80.0f);
        c.setProgramRelease(0.8f);
        c.setAdaptiveAttack(0.6f);
        c.setSpectralTiming(0.7f);
        c.setAttackEmphasis(0.4f);
    };
    const auto blockOut = [&](auto&& setup) {
        C c;
        setup(c);
        c.prepare(AudioSpec{ 48000.0, 512, 1 });
        std::vector<Vec> x { a };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        return x[0];
    };
    const auto farthest = [&](const Vec& p, const Vec& q) {
        double d = 0.0;
        for (int i = 0; i < n; ++i)
            d = std::max(d, std::abs(static_cast<double>(p[static_cast<size_t>(i)]) - q[static_cast<size_t>(i)]));
        return d;
    };

    const Vec block = blockOut(timed);
    const Vec plain = blockOut([](C& c) { c.setThreshold(-18.0f); c.setRatio(4.0f); c.setAttack(2.0f); c.setRelease(80.0f); });
    EXPECT_GT(farthest(block, plain), 0.01);

    C single;
    timed(single);
    single.prepare(AudioSpec{ 48000.0, 512, 1 });
    Vec y = a;
    for (auto& s : y) s = single.processSample(s, 0);
    EXPECT_EQ(farthest(block, y), 0.0);

    // Where the timing does not apply, nothing changes.
    const auto unchanged = [&](auto&& routing) {
        const Vec with = blockOut([&](C& c) { timed(c); routing(c); });
        const Vec without = blockOut([&](C& c) {
            c.setThreshold(-18.0f);
            c.setRatio(4.0f);
            c.setAttack(2.0f);
            c.setRelease(80.0f);
            routing(c);
        });
        return farthest(with, without);
    };
    EXPECT_EQ(unchanged([](C& c) { c.setCharacter(C::Character::Opto); }), 0.0);
    EXPECT_EQ(unchanged([](C& c) { c.setCharacter(C::Character::FET); }), 0.0);
    EXPECT_EQ(unchanged([](C& c) { c.setTopology(C::Topology::FeedBack); }), 0.0);
    EXPECT_EQ(unchanged([](C& c) { c.setCleanBallistics(C::CleanBallistics::Decoupled); }), 0.0);
    EXPECT_GT(unchanged([](C& c) { c.setCharacter(C::Character::Varimu); }), 0.01);

    // At Precision::High the release still knows how long the reduction lasted.
    const auto release = [&](float amount) {
        const Vec in = compToneThenSilence(1000.0, 0.4, 3.0, 5.5);
        const auto trace = compTrace(in, [&](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(100.0f);
            c.setProgramRelease(amount);
            c.setPrecision(C::Precision::High);
        });
        return compReleaseMs(trace, 48000 * 3);
    };
    const double fixed = release(0.0f), program = release(1.0f);
    EXPECT_NEAR(fixed, 100.0, 6.0);
    EXPECT_GT(program, 600.0);

    // State and wild values.
    C c;
    timed(c);
    c.setDetector(C::DetectorType::Hybrid);
    const auto blob = c.getState();
    C other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_NEAR(static_cast<double>(other.getProgramRelease()), 0.8, 1e-6);
    EXPECT_NEAR(static_cast<double>(other.getAdaptiveAttack()), 0.6, 1e-6);
    EXPECT_NEAR(static_cast<double>(other.getSpectralTiming()), 0.7, 1e-6);
    EXPECT_NEAR(static_cast<double>(other.getAttackEmphasis()), 0.4, 1e-6);
    EXPECT_TRUE(other.getDetector() == C::DetectorType::Hybrid);
    EXPECT_TRUE(other.getState() == blob);
    other.setProgramRelease(std::numeric_limits<float>::quiet_NaN());
    other.setAttackEmphasis(-7.0f);
    other.setDetector(static_cast<C::DetectorType>(99));
    EXPECT_NEAR(static_cast<double>(other.getProgramRelease()), 0.8, 1e-6);
    EXPECT_NEAR(static_cast<double>(other.getAttackEmphasis()), -1.0, 1e-6);
    EXPECT_TRUE(other.getDetector() == C::DetectorType::Hybrid);
}

// ============================================================================
// Compressor: the equalizer of the key and the key listen
// ============================================================================

namespace {

using CompKeyEq = KeyEqualizer<float, 16>;

// A signal through a key equalizer of its own, as the compressor's would shape it.
template <class Setup>
Vec compShapedKey(const Vec& in, Setup&& setup)
{
    auto eq = std::make_unique<CompKeyEq>();
    eq->prepare(48000.0);
    setup(*eq);
    (void)eq->sync();
    Vec out = in;
    for (auto& s : out) s = eq->processSample(s, 0);
    return out;
}

double compFarthest(const Vec& p, const Vec& q, size_t from = 0)
{
    double d = 0.0;
    for (size_t i = from; i < p.size(); ++i)
        d = std::max(d, std::abs(static_cast<double>(p[i]) - static_cast<double>(q[i])));
    return d;
}

// A loud low tone under a quiet high one, with a slow swell on both.
Vec compLowOverHigh(int n)
{
    Vec v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = i / 48000.0;
        const double swell = 0.7 + 0.3 * std::sin(2.0 * kPiD * 1.7 * t);
        v[static_cast<size_t>(i)] = static_cast<float>(swell * (0.5 * std::sin(2.0 * kPiD * 60.0 * t)
                                                               + 0.05 * std::sin(2.0 * kPiD * 3000.0 * t)));
    }
    return v;
}

} // namespace

// New: an equalizer on the key. A low cut takes the low tone out of what the
// detector weighs and the reduction goes with it; the audio is the one of an
// external key shaped by the same filter (to rounding) in the block path, in
// the per-sample path and at Precision::High; and in feedback it shapes what
// the loop reads.
DSPARK_TEST(Compressor_key_equalizer_decides_what_the_detector_weighs)
{
    using C = Compressor<float>;
    const int n = 48000;
    const Vec in = compLowOverHigh(n);
    C::KeyBand cut;
    cut.shape = C::KeyShape::LowCut;
    cut.frequency = 500.0f;
    cut.slope = 24.0f;

    const auto settings = [](C& c) {
        c.setThreshold(-20.0f);
        c.setRatio(4.0f);
        c.setAttack(5.0f);
        c.setRelease(100.0f);
    };
    // Output and the deepest reduction of the last tenth of a second.
    const auto render = [&](auto&& setup, const Vec* key, double& deepest) {
        auto c = std::make_unique<C>();
        settings(*c);
        setup(*c);
        c->prepare(AudioSpec{ 48000.0, 512, 1 });
        std::vector<Vec> x { in };
        Vec k = key ? *key : Vec();
        int at = 0;
        deepest = 0.0;
        processInBlocks(x, 512, [&](AudioBufferView<float> b) {
            if (key)
            {
                float* ptr = k.data() + at;
                AudioBufferView<float> side(&ptr, 1, b.getNumSamples());
                c->processBlock(b, side);
            }
            else
            {
                c->processBlock(b);
            }
            at += b.getNumSamples();
            if (at > n - 4800) deepest = std::min(deepest, static_cast<double>(c->getGainReductionDb()));
        });
        return x[0];
    };

    double plainGr = 0.0, cutGr = 0.0, sideGr = 0.0;
    const Vec plain = render([](C&) {}, nullptr, plainGr);
    const Vec shaped = render([&](C& c) { c.setSidechainBand(0, cut); }, nullptr, cutGr);
    EXPECT_LT(plainGr, -5.0);                 // the low tone alone asks for some 7 dB
    EXPECT_GT(cutGr, -0.05);                  // without it the key is under the threshold
    EXPECT_GT(compFarthest(plain, shaped), 0.1);

    // The same filter on an external key gives the same audio.
    const Vec key = compShapedKey(in, [&](CompKeyEq& eq) { eq.setBand(0, cut); });
    const Vec outside = render([](C&) {}, &key, sideGr);
    EXPECT_LT(compFarthest(shaped, outside), 1e-5);

    // A bell that lifts the quiet tone over the threshold, in every path.
    C::KeyBand bell;
    bell.shape = C::KeyShape::Bell;
    bell.frequency = 3000.0f;
    bell.gainDb = 24.0f;
    bell.q = 2.0f;
    const auto both = [&](C& c) {
        c.setSidechainBand(0, cut);
        c.setSidechainBand(3, bell);
    };
    const Vec bellKey = compShapedKey(in, [&](CompKeyEq& eq) { eq.setBand(0, cut); eq.setBand(3, bell); });
    double bellGr = 0.0;
    const Vec lifted = render(both, nullptr, bellGr);
    EXPECT_LT(bellGr, -1.0);
    EXPECT_GT(compFarthest(lifted, shaped), 0.01);
    EXPECT_LT(compFarthest(lifted, render([](C&) {}, &bellKey, sideGr)), 1e-5);

    auto single = std::make_unique<C>();
    settings(*single);
    both(*single);
    single->prepare(AudioSpec{ 48000.0, 512, 1 });
    Vec y = in;
    for (auto& s : y) s = single->processSample(s, 0);
    EXPECT_LT(compFarthest(lifted, y), 1e-5);

    const auto precise = [&](C& c) { both(c); c.setPrecision(C::Precision::High); };
    const Vec fine = render(precise, nullptr, bellGr);
    EXPECT_LT(bellGr, -1.0);
    EXPECT_LT(compFarthest(fine, render([](C& c) { c.setPrecision(C::Precision::High); }, &bellKey, sideGr)), 1e-5);

    // In feedback the loop reads the compressed output through the equalizer.
    double loopGr = 0.0, loopCutGr = 0.0;
    const auto loop = [](C& c) { c.setTopology(C::Topology::FeedBack); c.setDetector(C::DetectorType::Rms); };
    const Vec fed = render(loop, nullptr, loopGr);
    const Vec fedCut = render([&](C& c) { loop(c); c.setSidechainBand(0, cut); }, nullptr, loopCutGr);
    EXPECT_LT(loopGr, -3.0);
    EXPECT_GT(loopCutGr, -0.05);
    for (float s : fedCut) EXPECT_TRUE(std::isfinite(s));

    // A band set while it runs is taken by the next block.
    {
        auto c = std::make_unique<C>();
        settings(*c);
        c->prepare(AudioSpec{ 48000.0, 64, 1 });
        std::vector<Vec> x { in };
        double before = 0.0, last = 0.0;
        int at = 0;
        processInBlocks(x, 64, [&](AudioBufferView<float> b) {
            if (at == 24000)
            {
                before = static_cast<double>(c->getGainReductionDb());
                c->setSidechainBand(0, cut);
            }
            c->processBlock(b);
            at += b.getNumSamples();
            last = static_cast<double>(c->getGainReductionDb());
        });
        EXPECT_LT(before, -3.0);
        EXPECT_GT(last, -0.05);
        for (float s : x[0]) EXPECT_TRUE(std::isfinite(s));
    }

    // The bands are part of the state; the curve for a display follows them.
    C a;
    both(a);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.getSidechainBand(0).shape == C::KeyShape::LowCut);
    EXPECT_EQ(b.getSidechainBand(0).frequency, 500.0f);
    EXPECT_EQ(b.getSidechainBand(0).slope, 24.0f);
    EXPECT_TRUE(b.getSidechainBand(3).shape == C::KeyShape::Bell);
    EXPECT_EQ(b.getSidechainBand(3).gainDb, 24.0f);
    EXPECT_TRUE(b.getSidechainBand(1).shape == C::KeyShape::Off);
    EXPECT_TRUE(b.getState() == blob);
    b.prepare(AudioSpec{ 48000.0, 512, 1 });
    const auto curve = b.getSidechainResponse();
    EXPECT_NEAR(curve.magnitudeDb(3000.0), 24.0, 0.3);
    EXPECT_LT(curve.magnitudeDb(60.0), -60.0);
}

// New: key listen. The output is what each detector hears (the key after the
// high-pass and the equalizer, to rounding), in every path, ahead of the lookahead; the
// compressor keeps working underneath, so switching it off returns to the
// audio as if it had never been on; and it is not part of the state.
DSPARK_TEST(Compressor_key_listen_plays_what_the_detector_hears)
{
    using C = Compressor<float>;
    const int n = 24000;
    const Vec left = compLowOverHigh(n);
    Vec right(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        right[static_cast<size_t>(i)] = static_cast<float>(0.4 * std::sin(2.0 * kPiD * 220.0 * i / 48000.0));
    C::KeyBand bell;
    bell.shape = C::KeyShape::Bell;
    bell.frequency = 3000.0f;
    bell.gainDb = 12.0f;
    bell.q = 1.5f;
    const auto shape = [&](CompKeyEq& eq) { eq.setBand(2, bell); };
    const Vec heardLeft = compShapedKey(left, shape), heardRight = compShapedKey(right, shape);
    const size_t resumed = 12288;       // first block boundary past the middle

    const auto render = [&](auto&& setup, int block, bool listenSecondHalfOff, double& deepest) {
        auto c = std::make_unique<C>();
        c->setThreshold(-24.0f);
        c->setRatio(4.0f);
        c->setAttack(2.0f);
        c->setRelease(60.0f);
        c->setStereoLink(0.0f);
        c->setSidechainBand(2, bell);
        setup(*c);
        c->prepare(AudioSpec{ 48000.0, block, 2 });
        std::vector<Vec> x { left, right };
        int at = 0;
        deepest = 0.0;
        processInBlocks(x, block, [&](AudioBufferView<float> b) {
            if (listenSecondHalfOff && at >= static_cast<int>(resumed)) c->setSidechainListen(false);
            c->processBlock(b);
            at += b.getNumSamples();
            deepest = std::min(deepest, static_cast<double>(c->getGainReductionDb()));
        });
        return x;
    };
    const auto listening = [](C& c) { c.setSidechainListen(true); };

    // The block path: the key, and the meter still reads.
    double gr = 0.0;
    auto out = render(listening, 512, false, gr);
    EXPECT_LT(compFarthest(out[0], heardLeft), 1e-6);
    EXPECT_LT(compFarthest(out[1], heardRight), 1e-6);
    EXPECT_LT(gr, -3.0);

    // With a latency the key leaves late by it, like the audio would: the
    // lookahead (5 ms: 240 samples), or the 7 samples the detector of
    // Precision::High reads late (the per-sample loop of the block path).
    const auto farthestLate = [&](const std::vector<Vec>& got, size_t late) {
        double worst = 0.0;
        for (size_t i = late; i < got[0].size(); ++i)
        {
            worst = std::max(worst, std::abs(static_cast<double>(got[0][i]) - static_cast<double>(heardLeft[i - late])));
            worst = std::max(worst, std::abs(static_cast<double>(got[1][i]) - static_cast<double>(heardRight[i - late])));
        }
        return worst;
    };
    out = render([&](C& c) { listening(c); c.setLookahead(5.0f); }, 512, false, gr);
    EXPECT_LT(farthestLate(out, 240), 1e-6);

    out = render([&](C& c) { listening(c); c.setPrecision(C::Precision::High); }, 512, false, gr);
    EXPECT_LT(farthestLate(out, 7), 1e-6);

    // processSample() has no latency: the key as it arrives.
    {
        auto c = std::make_unique<C>();
        c->setThreshold(-24.0f);
        c->setSidechainBand(2, bell);
        c->setSidechainListen(true);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        Vec l = left, r = right;
        for (int i = 0; i < n; ++i)
        {
            l[static_cast<size_t>(i)] = c->processSample(l[static_cast<size_t>(i)], 0);
            r[static_cast<size_t>(i)] = c->processSample(r[static_cast<size_t>(i)], 1);
        }
        EXPECT_LT(compFarthest(l, heardLeft), 1e-6);
        EXPECT_LT(compFarthest(r, heardRight), 1e-6);
        EXPECT_LT(static_cast<double>(c->getGainReductionDb()), -1.0);
    }

    // An external key is what is heard, each channel its own.
    {
        auto c = std::make_unique<C>();
        c->setSidechainBand(2, bell);
        c->setSidechainListen(true);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        std::vector<Vec> x { left, left };
        Vec keyA = right, keyB = left;
        int at = 0;
        processInBlocks(x, 512, [&](AudioBufferView<float> b) {
            float* ptr[2] = { keyA.data() + at, keyB.data() + at };
            AudioBufferView<float> side(ptr, 2, b.getNumSamples());
            c->processBlock(b, side);
            at += b.getNumSamples();
        });
        EXPECT_LT(compFarthest(x[0], heardRight), 1e-6);
        EXPECT_LT(compFarthest(x[1], heardLeft), 1e-6);
    }

    // Switched off half way, the audio is the one of a compressor that never listened.
    const auto never = render([](C&) {}, 512, false, gr);
    const auto half = render(listening, 512, true, gr);
    EXPECT_GT(compFarthest(half[0], heardLeft), 0.01);
    EXPECT_EQ(compFarthest(never[0], half[0], resumed), 0.0);
    EXPECT_EQ(compFarthest(never[1], half[1], resumed), 0.0);
    const auto neverFine = render([](C& c) { c.setPrecision(C::Precision::High); }, 512, false, gr);
    const auto halfFine = render([&](C& c) { listening(c); c.setPrecision(C::Precision::High); }, 512, true, gr);
    EXPECT_EQ(compFarthest(neverFine[0], halfFine[0], resumed), 0.0);

    // A monitoring switch: the state does not carry it.
    C a;
    a.setSidechainListen(true);
    EXPECT_TRUE(a.isSidechainListen());
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_FALSE(b.isSidechainListen());
}

// New: the gate and the expander shape their key with the same equalizer.
// A hum that holds them open stops doing so once a low cut takes it out of
// the key; the audio is the one of an external key shaped by the same
// filter; key listen plays what the detector hears while they keep working;
// and the bands are part of the state.
DSPARK_TEST(Gate_and_expander_shape_and_play_their_key)
{
    const int n = 24000;
    const Vec hum = makeSine(60.0, 0.3, 48000.0, n);
    Vec other(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        other[static_cast<size_t>(i)] = static_cast<float>(0.2 * std::sin(2.0 * kPiD * 90.0 * i / 48000.0));
    CompKeyEq::Band cut;
    cut.shape = CompKeyEq::Shape::LowCut;
    cut.frequency = 500.0f;
    cut.slope = 24.0f;
    const auto shape = [&](CompKeyEq& eq) { eq.setBand(1, cut); };
    const Vec heardHum = compShapedKey(hum, shape), heardOther = compShapedKey(other, shape);
    const auto rmsOfTail = [&](const Vec& v) {
        double sum = 0.0;
        for (int i = n / 2; i < n; ++i) sum += static_cast<double>(v[static_cast<size_t>(i)]) * v[static_cast<size_t>(i)];
        return std::sqrt(sum / (n / 2));
    };
    const double humRms = rmsOfTail(hum);

    // One processor over the two channels, keyed by itself or by `key`.
    const auto render = [&](auto& p, const std::vector<Vec>* key) {
        std::vector<Vec> x { hum, other };
        std::vector<Vec> k = key ? *key : std::vector<Vec>();
        int at = 0;
        processInBlocks(x, 256, [&](AudioBufferView<float> b) {
            if (key)
            {
                float* ptr[2] = { k[0].data() + at, k[1].data() + at };
                AudioBufferView<float> side(ptr, 2, b.getNumSamples());
                p.processBlock(b, side);
            }
            else
            {
                p.processBlock(b);
            }
            at += b.getNumSamples();
        });
        return x;
    };

    // The gate.
    {
        using G = NoiseGate<float>;
        const auto make = [&](bool shaped, bool listen) {
            auto g = std::make_unique<G>();
            g->setThreshold(-30.0f);
            g->setHold(5.0f);
            g->setRelease(20.0f);
            if (shaped) g->setSidechainBand(1, cut);
            g->setSidechainListen(listen);
            g->prepare(48000.0, 2);
            return g;
        };
        auto open = make(false, false);
        const auto plain = render(*open, nullptr);
        EXPECT_GT(rmsOfTail(plain[0]), humRms * 0.9);            // the hum holds it open
        auto closed = make(true, false);
        const auto shaped = render(*closed, nullptr);
        EXPECT_LT(rmsOfTail(shaped[0]), humRms * 0.001);         // out of the key, it closes

        // An external key through the same filter.
        const std::vector<Vec> key { heardHum, heardOther };
        auto outside = make(false, false);
        const auto keyed = render(*outside, &key);
        EXPECT_LT(compFarthest(shaped[0], keyed[0]), 1e-5);
        EXPECT_LT(compFarthest(shaped[1], keyed[1]), 1e-5);

        // Key listen, in the block path and one sample at a time.
        auto listening = make(true, true);
        const auto heard = render(*listening, nullptr);
        EXPECT_LT(compFarthest(heard[0], heardHum), 1e-6);
        EXPECT_LT(compFarthest(heard[1], heardOther), 1e-6);
        auto single = make(true, true);
        Vec y = hum;
        for (auto& s : y) s = single->processSample(s);
        EXPECT_LT(compFarthest(y, heardHum), 1e-6);

        const auto blob = closed->getState();
        G other2;
        EXPECT_TRUE(other2.setState(blob.data(), blob.size()));
        EXPECT_TRUE(other2.getSidechainBand(1).shape == G::KeyShape::LowCut);
        EXPECT_EQ(other2.getSidechainBand(1).frequency, 500.0f);
        EXPECT_TRUE(other2.getState() == blob);
        const auto listenBlob = listening->getState();
        EXPECT_TRUE(other2.setState(listenBlob.data(), listenBlob.size()));
        EXPECT_FALSE(other2.isSidechainListen());
        other2.prepare(48000.0, 2);
        EXPECT_LT(other2.getSidechainResponse().magnitudeDb(60.0), -60.0);
    }

    // The expander.
    {
        using E = Expander<float>;
        const auto make = [&](bool shaped, bool listen) {
            auto e = std::make_unique<E>();
            e->setThreshold(-30.0f);
            e->setRatio(4.0f);
            e->setHold(5.0f);
            if (shaped) e->setSidechainBand(1, cut);
            e->setSidechainListen(listen);
            e->prepare(48000.0, 2);
            return e;
        };
        auto open = make(false, false);
        const auto plain = render(*open, nullptr);
        EXPECT_GT(rmsOfTail(plain[0]), humRms * 0.9);
        auto closed = make(true, false);
        const auto shaped = render(*closed, nullptr);
        EXPECT_LT(rmsOfTail(shaped[0]), humRms * 0.1);

        const std::vector<Vec> key { heardHum, heardOther };
        auto outside = make(false, false);
        const auto keyed = render(*outside, &key);
        EXPECT_LT(compFarthest(shaped[0], keyed[0]), 1e-5);
        EXPECT_LT(compFarthest(shaped[1], keyed[1]), 1e-5);

        auto listening = make(true, true);
        const auto heard = render(*listening, nullptr);
        EXPECT_LT(compFarthest(heard[0], heardHum), 1e-6);
        EXPECT_LT(compFarthest(heard[1], heardOther), 1e-6);

        const auto blob = closed->getState();
        E other2;
        EXPECT_TRUE(other2.setState(blob.data(), blob.size()));
        EXPECT_TRUE(other2.getSidechainBand(1).shape == E::KeyShape::LowCut);
        EXPECT_EQ(other2.getSidechainBand(1).slope, 24.0f);
        EXPECT_TRUE(other2.getState() == blob);
        const auto listenBlob = listening->getState();
        EXPECT_TRUE(other2.setState(listenBlob.data(), listenBlob.size()));
        EXPECT_FALSE(other2.isSidechainListen());
    }
}

// ============================================================================
// Compressor: the Smooth character, the push-pull element and the gain after
// the ballistics
// ============================================================================

namespace {

// Level in dB of the line at `hz` over the last tenth of a second of a signal at 48 kHz
// (hz a multiple of 10 Hz, so the stretch holds whole periods).
double compLineDb(const Vec& v, double hz)
{
    const size_t n = 4800, from = v.size() - n;
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double phase = 2.0 * kPiD * hz * static_cast<double>(i) / 48000.0;
        re += static_cast<double>(v[from + i]) * std::cos(phase);
        im += static_cast<double>(v[from + i]) * std::sin(phase);
    }
    return 20.0 * std::log10(std::max(2.0 * std::sqrt(re * re + im * im) / static_cast<double>(n), 1e-12));
}

// A mono render in blocks of 512.
template <class Setup>
Vec compRender(const Vec& in, Setup&& setup)
{
    auto c = std::make_unique<Compressor<float>>();
    setup(*c);
    c->prepare(AudioSpec{ 48000.0, 512, 1 });
    std::vector<Vec> x { in };
    processInBlocks(x, 512, [&](AudioBufferView<float> b) { c->processBlock(b); });
    return x[0];
}

// The same, one sample at a time.
template <class Setup>
Vec compRenderSamples(const Vec& in, Setup&& setup)
{
    auto c = std::make_unique<Compressor<float>>();
    setup(*c);
    c->prepare(AudioSpec{ 48000.0, 512, 1 });
    Vec y = in;
    for (auto& s : y) s = c->processSample(s, 0);
    return y;
}

} // namespace

// New: the Smooth character. Two layers make its curve: half the slope from
// 6 dB under the threshold, the whole from 6 dB over, and the ratio's own
// curve far above. Its followers take longer the farther they are, so a deep
// reduction comes back more slowly than a shallow one and an onset is caught
// later than by a one-pole; Topology and Precision do not apply to it; in
// Upward mode it is the Clean character.
DSPARK_TEST(Compressor_smooth_character_levels_under_and_catches_over)
{
    using C = Compressor<float>;
    const auto smooth = [](C& c) {
        c.setCharacter(C::Character::Layered);
        c.setDetector(C::DetectorType::Hilbert);
        c.setThreshold(-20.0f);
        c.setRatio(4.0f);
        c.setAttack(5.0f);
        c.setRelease(100.0f);
    };
    const auto clean = [](C& c) {
        c.setDetector(C::DetectorType::Hilbert);
        c.setThreshold(-20.0f);
        c.setRatio(4.0f);
        c.setAttack(5.0f);
        c.setRelease(100.0f);
    };

    // The static curve, on steady tones (decibels of reduction at a level).
    const auto settled = [&](double levelDb, auto&& setup) {
        const Vec in = makeSine(1000.0, std::pow(10.0, levelDb / 20.0), 48000.0, 48000 * 4);
        return compTrace(in, setup).back();
    };
    EXPECT_NEAR(settled(-34.0, smooth), 0.0, 0.02);
    EXPECT_NEAR(settled(-26.0, smooth), -0.5625, 0.08);
    EXPECT_NEAR(settled(-20.0, smooth), -2.25, 0.08);
    EXPECT_NEAR(settled(-14.0, smooth), -4.78, 0.08);
    EXPECT_NEAR(settled(-8.0, smooth), -9.0, 0.08);
    EXPECT_NEAR(settled(-2.0, smooth), -13.5, 0.08);
    EXPECT_NEAR(settled(-8.0, clean), -9.0, 0.08);         // far over, the ratio's own curve
    EXPECT_NEAR(settled(-20.0, clean), 0.0, 0.02);

    // The release: the time to come back to 1/e of the reduction, and the
    // share of a deep reduction given back in the first 100 ms. A one-pole
    // rushes (63 % in its time); the layers leave at a steady pace.
    double smoothEarly = 0.0, cleanEarly = 0.0;
    const auto release = [&](double levelDb, auto&& setup, double& early) {
        const Vec in = compToneThenSilence(1000.0, std::pow(10.0, levelDb / 20.0), 3.0, 7.0);
        const auto trace = compTrace(in, setup);
        const size_t stop = static_cast<size_t>(48000 * 3 / 16);
        early = 1.0 - trace[stop + 300] / trace[stop - 1];
        return compReleaseMs(trace, 48000 * 3);
    };
    const double shallow = release(-14.0, smooth, smoothEarly), deep = release(0.0, smooth, smoothEarly);
    const double cleanShallow = release(-14.0, clean, cleanEarly), cleanDeep = release(0.0, clean, cleanEarly);
    EXPECT_NEAR(cleanShallow, cleanDeep, 4.0);
    EXPECT_GT(shallow, cleanShallow * 3.0);
    EXPECT_GT(deep, cleanDeep * 3.0);
    EXPECT_LT(deep, 2500.0);
    EXPECT_GT(cleanEarly, 0.55);
    EXPECT_LT(smoothEarly, 0.3);

    // An onset 12 dB over the threshold: caught later, and to the same depth.
    Vec step = makeSine(1000.0, 0.01, 48000.0, 48000 * 3);
    for (size_t i = 48000; i < step.size(); ++i) step[i] *= 40.0f;     // -40 -> -8 dBFS
    const auto smoothTrace = compTrace(step, smooth), cleanTrace = compTrace(step, clean);
    const double smoothAttack = compAttackMs(smoothTrace, 48000, -9.0);
    const double cleanAttack = compAttackMs(cleanTrace, 48000, -9.0);
    EXPECT_GT(smoothAttack, cleanAttack * 1.5);
    EXPECT_LT(smoothAttack, 200.0);
    EXPECT_NEAR(smoothTrace.back(), cleanTrace.back(), 0.1);

    // The paths: one sample at a time is the block; Topology and Precision change nothing.
    const Vec burst = compLowOverHigh(24000);
    const auto dense = [](C& c) {
        c.setCharacter(C::Character::Layered);
        c.setThreshold(-24.0f);
        c.setRatio(3.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setHoldTime(3.0f);
        c.setRange(8.0f);
    };
    const Vec block = compRender(burst, dense);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, dense)), 1e-6);
    EXPECT_EQ(compFarthest(block, compRender(burst, [&](C& c) { dense(c); c.setTopology(C::Topology::FeedBack); })), 0.0);
    EXPECT_EQ(compFarthest(block, compRender(burst, [&](C& c) { dense(c); c.setPrecision(C::Precision::High); })), 0.0);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { dense(c); c.setCharacter(C::Character::Clean); })), 0.01);
    {
        C c;
        dense(c);
        c.setPrecision(C::Precision::High);
        c.prepare(AudioSpec{ 48000.0, 512, 1 });
        EXPECT_EQ(c.getLatency(), 0);
        c.setCharacter(C::Character::Clean);
        EXPECT_GT(c.getLatency(), 0);
    }
    // The range bounds the sum of the two layers.
    {
        const Vec loud = makeSine(1000.0, 0.9, 48000.0, 48000);
        const auto trace = compTrace(loud, dense);
        double deepest = 0.0;
        for (double gr : trace) deepest = std::min(deepest, gr);
        EXPECT_NEAR(deepest, -8.0, 0.01);
    }
    // Upward: one envelope on the knobs, as Clean.
    const auto upward = [](C& c) { c.setMode(C::Mode::Upward); c.setThreshold(-30.0f); c.setRatio(2.0f); };
    EXPECT_EQ(compFarthest(compRender(burst, [&](C& c) { upward(c); c.setCharacter(C::Character::Layered); }),
                           compRender(burst, upward)), 0.0);

    // Switching characters while it runs carries the gain across.
    {
        auto c = std::make_unique<C>();
        clean(*c);
        c->prepare(AudioSpec{ 48000.0, 16, 1 });
        std::vector<Vec> x { makeSine(1000.0, 0.4, 48000.0, 48000 * 2) };
        int at = 0;
        double last = 0.0, jump = 0.0;
        processInBlocks(x, 16, [&](AudioBufferView<float> b) {
            if (at == 48000) c->setCharacter(C::Character::Layered);
            c->processBlock(b);
            at += b.getNumSamples();
            const double gr = static_cast<double>(c->getGainReductionDb());
            if (at > 24000) jump = std::max(jump, std::abs(gr - last));
            last = gr;
        });
        EXPECT_LT(jump, 0.2);                               // dB between two blocks of 16 samples
    }

    C a;
    a.setCharacter(C::Character::Layered);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.getCharacter() == C::Character::Layered);
}

// New: with the Varimu character the colour applies the gain through a
// push-pull pair. Its harmonics are odd, grow with the reduction and with
// the level, a small signal gets the gain exactly, and the other characters
// are untouched.
DSPARK_TEST(Compressor_varimu_colour_is_a_push_pull_element_that_bends_with_the_reduction)
{
    using C = Compressor<float>;
    // A reduction pinned by the range: the gain is the same every sample.
    const auto pinned = [](float rangeDb, float colour) {
        return [=](C& c) {
            c.setCharacter(C::Character::Varimu);
            c.setThreshold(-60.0f);
            c.setRatio(20.0f);
            c.setAttack(1.0f);
            c.setRelease(100.0f);
            c.setRange(rangeDb);
            c.setHoldTime(5.0f);          // the target stays down through the zero crossings
            c.setCharacterColor(colour);
        };
    };
    const Vec tone = makeSine(1000.0, 0.5, 48000.0, 48000);
    const auto third = [&](const Vec& out) { return compLineDb(out, 3000.0) - compLineDb(out, 1000.0); };

    const Vec plain = compRender(tone, pinned(12.0f, 0.0f));
    EXPECT_NEAR(compLineDb(plain, 1000.0), -6.02 - 12.0, 0.02);
    EXPECT_LT(third(plain), -100.0);

    const Vec twelve = compRender(tone, pinned(12.0f, 1.0f));
    EXPECT_NEAR(third(twelve), -37.0, 1.0);
    EXPECT_LT(compLineDb(twelve, 2000.0) - compLineDb(twelve, 1000.0), -100.0);   // odd harmonics only
    EXPECT_LT(compLineDb(twelve, 4000.0) - compLineDb(twelve, 1000.0), -100.0);
    EXPECT_NEAR(compLineDb(twelve, 1000.0), -18.02 + 0.39, 0.1);

    EXPECT_NEAR(third(compRender(tone, pinned(20.0f, 1.0f))), -31.2, 1.0);
    EXPECT_NEAR(third(compRender(tone, pinned(6.0f, 1.0f))), -69.5, 3.0);         // the minimum of the pair
    EXPECT_NEAR(third(compRender(tone, pinned(12.0f, 0.5f))), -48.2, 1.0);

    // At rest (no reduction) the pair is nearly straight.
    const auto resting = [](C& c) {
        c.setCharacter(C::Character::Varimu);
        c.setThreshold(0.0f);
        c.setCharacterColor(1.0f);
    };
    EXPECT_NEAR(third(compRender(tone, resting)), -61.5, 1.0);

    // A small signal gets the gain, and nothing else.
    const Vec quiet = makeSine(1000.0, 0.01, 48000.0, 48000);
    const Vec quietOut = compRender(quiet, pinned(12.0f, 1.0f));
    EXPECT_NEAR(compLineDb(quietOut, 1000.0), -40.0 - 12.0, 0.01);
    EXPECT_LT(third(quietOut), -95.0);

    // One sample at a time; in a feedback loop; and the characters without it.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C& c) {
        c.setCharacter(C::Character::Varimu);
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(3.0f);
        c.setRelease(80.0f);
        c.setCharacterColor(0.8f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setCharacterColor(0.0f); })), 1e-3);
    const Vec looped = compRender(burst, [&](C& c) { working(c); c.setTopology(C::Topology::FeedBack); });
    for (float s : looped) EXPECT_TRUE(std::isfinite(s) && std::abs(s) < 2.0f);
    const auto others = [&](C::Character type) {
        const auto base = [=](C& c) { c.setCharacter(type); c.setThreshold(-24.0f); c.setRatio(4.0f); };
        return compFarthest(compRender(burst, base),
                            compRender(burst, [&](C& c) { base(c); c.setCharacterColor(1.0f); }));
    };
    EXPECT_EQ(others(C::Character::Clean), 0.0);
    EXPECT_EQ(others(C::Character::Opto), 0.0);
    EXPECT_EQ(others(C::Character::Layered), 0.0);
}

// New: the gain can be smoothed after the ballistics. Two sections round its
// corners: an attack that was a step arrives over the time set, the ripple a
// fast release leaves on a low tone goes, and the depth it settles on stays.
DSPARK_TEST(Compressor_gain_smoothing_rounds_the_gain_after_the_ballistics)
{
    using C = Compressor<float>;
    const auto fast = [](float smoothingMs) {
        return [=](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(0.1f);
            c.setRelease(200.0f);
            c.setGainSmoothing(smoothingMs);
        };
    };
    Vec step = makeSine(1000.0, 0.01, 48000.0, 48000 * 2);
    for (size_t i = 48000; i < step.size(); ++i) step[i] *= 40.0f;     // -40 -> -8 dBFS
    const auto sharp = compTrace(step, fast(0.0f)), round = compTrace(step, fast(10.0f));
    const double sharpMs = compAttackMs(sharp, 48000, sharp.back());
    EXPECT_LT(sharpMs, 5.0);
    EXPECT_NEAR(compAttackMs(round, 48000, round.back()) - sharpMs, 10.7, 2.0);
    EXPECT_NEAR(round.back(), sharp.back(), 0.02);
    EXPECT_NEAR(round.back(), -9.0, 0.1);

    // The ripple of the gain on a 100 Hz tone with a quick release.
    const auto riding = [](float smoothingMs) {
        return [=](C& c) {
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(20.0f);
            c.setGainSmoothing(smoothingMs);
        };
    };
    const auto ripple = [&](float smoothingMs, double& mean) {
        const auto trace = compTrace(makeSine(100.0, 0.5, 48000.0, 48000), riding(smoothingMs));
        double lo = 0.0, hi = -100.0, sum = 0.0;
        const size_t from = trace.size() - 600;                         // the last fifth of a second
        for (size_t i = from; i < trace.size(); ++i)
        {
            lo = std::min(lo, trace[i]);
            hi = std::max(hi, trace[i]);
            sum += trace[i];
        }
        mean = sum / 600.0;
        return hi - lo;
    };
    double meanSharp = 0.0, meanRound = 0.0;
    const double rippleSharp = ripple(0.0f, meanSharp), rippleRound = ripple(20.0f, meanRound);
    EXPECT_GT(rippleSharp, 0.5);
    EXPECT_LT(rippleRound, rippleSharp * 0.05);
    EXPECT_NEAR(meanRound, meanSharp, 0.1);

    // The paths; a feedback loop reads the gain before the smoothing and stays put.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C& c) {
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setGainSmoothing(5.0f);
        c.setAmount(0.8f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setGainSmoothing(0.0f); })), 1e-3);
    const auto loop = [](float smoothingMs) {
        return [=](C& c) {
            c.setTopology(C::Topology::FeedBack);
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(100.0f);
            c.setGainSmoothing(smoothingMs);
        };
    };
    const Vec steady = makeSine(1000.0, 0.4, 48000.0, 48000);
    EXPECT_NEAR(compTrace(steady, loop(20.0f)).back(), compTrace(steady, loop(0.0f)).back(), 0.05);

    // Switched on while the gain is down, it starts from where the gain is.
    {
        auto c = std::make_unique<C>();
        fast(0.0f)(*c);
        c->prepare(AudioSpec{ 48000.0, 16, 1 });
        std::vector<Vec> x { makeSine(1000.0, 0.4, 48000.0, 48000) };
        int at = 0;
        double last = 0.0, jump = 0.0;
        processInBlocks(x, 16, [&](AudioBufferView<float> b) {
            if (at == 24000) c->setGainSmoothing(30.0f);
            c->processBlock(b);
            at += b.getNumSamples();
            const double gr = static_cast<double>(c->getGainReductionDb());
            if (at > 12000) jump = std::max(jump, std::abs(gr - last));
            last = gr;
        });
        EXPECT_LT(jump, 0.05);
    }

    // State and wild values.
    C a;
    a.setGainSmoothing(12.5f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getGainSmoothing(), 12.5f);
    b.setGainSmoothing(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(b.getGainSmoothing(), 12.5f);
    b.setGainSmoothing(900.0f);
    EXPECT_EQ(b.getGainSmoothing(), 50.0f);
    b.setGainSmoothing(-3.0f);
    EXPECT_EQ(b.getGainSmoothing(), 0.0f);
}

// ============================================================================
// Limiter: a latency kept for the lookahead
// ============================================================================

// New: setMaxLookahead() fixes the latency and lets the lookahead move under
// it. The output is the one of the same lookahead with no latency kept, later
// by the difference and to the bit; the ceiling holds wherever the lookahead
// sits; and the per-sample path is the block path.
DSPARK_TEST(Limiter_lookahead_moves_inside_a_latency_of_its_own)
{
    using L = Limiter<float>;
    const int n = 48000;
    Vec in(static_cast<size_t>(n));
    uint32_t seed = 5u;
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double t = i / 48000.0;
        const double burst = 0.2 + 1.3 * std::exp(-std::fmod(t, 0.25) * 30.0);
        in[static_cast<size_t>(i)] = static_cast<float>(burst * (0.7 * std::sin(2.0 * kPiD * 220.0 * t)
                                   + 0.3 * (static_cast<double>(seed) / 4294967296.0 - 0.5)));
    }
    const auto run = [&](float lookMs, float maxMs, int& latency) {
        L lim;
        lim.setCeiling(-6.0f);
        lim.setRelease(50.0f);
        lim.setLookahead(lookMs);
        lim.setMaxLookahead(maxMs);
        lim.prepare(AudioSpec{ 48000.0, 512, 1 });
        latency = lim.getLatency();
        std::vector<Vec> x { in };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { lim.processBlock(b); });
        return x[0];
    };
    int plainLatency = 0, keptLatency = 0;
    const Vec plain = run(2.0f, 0.0f, plainLatency);
    const Vec kept = run(2.0f, 5.0f, keptLatency);
    EXPECT_EQ(plainLatency, 96);
    EXPECT_EQ(keptLatency, 240);
    int differ = 0;
    for (int i = 144; i < n; ++i)
        if (kept[static_cast<size_t>(i)] != plain[static_cast<size_t>(i - 144)]) ++differ;
    EXPECT_EQ(differ, 0);
    for (int i = 0; i < 240; ++i) EXPECT_EQ(kept[static_cast<size_t>(i)], 0.0f);

    const float ceiling = std::pow(10.0f, -6.0f / 20.0f);
    for (float look : { 0.5f, 1.0f, 3.0f, 5.0f })
    {
        int latency = 0;
        const Vec y = run(look, 5.0f, latency);
        EXPECT_EQ(latency, 240);
        float peak = 0.0f;
        for (float s : y) peak = std::max(peak, std::abs(s));
        EXPECT_LT(peak, ceiling * 1.0001f);
        EXPECT_GT(peak, ceiling * 0.9f);
    }
    int under = 0;
    (void)run(5.0f, 2.0f, under);              // a maximum under the lookahead changes nothing
    EXPECT_EQ(under, 240);

    // The per-sample path, and the state.
    L block, single;
    for (L* lim : { &block, &single })
    {
        lim->setCeiling(-6.0f);
        lim->setLookahead(1.0f);
        lim->setMaxLookahead(4.0f);
        lim->prepare(AudioSpec{ 48000.0, 512, 1 });
    }
    std::vector<Vec> x { in };
    processInBlocks(x, 512, [&](AudioBufferView<float> b) { block.processBlock(b); });
    int apart = 0;
    for (int i = 0; i < n; ++i)
        if (single.processSample(in[static_cast<size_t>(i)], 0) != x[0][static_cast<size_t>(i)]) ++apart;
    EXPECT_EQ(apart, 0);

    const auto blob = block.getState();
    L other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_NEAR(static_cast<double>(other.getMaxLookahead()), 4.0, 1e-6);
    EXPECT_TRUE(other.getState() == blob);
    other.setMaxLookahead(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(static_cast<double>(other.getMaxLookahead()), 4.0, 1e-6);
}

// ============================================================================
// Expander, NoiseGate, DeEsser: a threshold counted from the programme
// ============================================================================

namespace {

// A second of a tone and a second of the same tone 25 dB lower, twice over.
Vec dynLoudAndQuiet(double scale)
{
    const int n = 48000 * 4;
    Vec v(static_cast<size_t>(n));
    const double low = std::pow(10.0, -25.0 / 20.0);
    for (int i = 0; i < n; ++i)
    {
        const double t = i / 48000.0;
        const double amp = 0.3 * scale * (std::fmod(t, 2.0) < 1.0 ? 1.0 : low);
        v[static_cast<size_t>(i)] = static_cast<float>(amp * std::sin(2.0 * kPiD * 440.0 * t));
    }
    return v;
}

double dynRms(const Vec& v, int from, int to)
{
    double s = 0.0;
    for (int i = from; i < to; ++i) s += static_cast<double>(v[static_cast<size_t>(i)]) * v[static_cast<size_t>(i)];
    return std::sqrt(s / (to - from));
}

// Runs a mono processor over a signal at two levels 30.5 dB apart and returns how far the two outputs
// are from being one the other scaled (relative to the output's peak), and the gain of the quiet and
// of the loud passage at the higher level, in dB.
struct DynScaled
{
    double apart = 0.0, quietDb = 0.0, loudDb = 0.0;
};

template <class Make>
DynScaled dynScaled(Make&& make)
{
    const double scale = 0.03;
    const Vec loudIn = dynLoudAndQuiet(1.0), softIn = dynLoudAndQuiet(scale);
    std::vector<Vec> loud { loudIn }, soft { softIn };
    {
        auto p = make();
        processInBlocks(loud, 512, [&](AudioBufferView<float> b) { p->processBlock(b); });
    }
    {
        auto p = make();
        processInBlocks(soft, 512, [&](AudioBufferView<float> b) { p->processBlock(b); });
    }
    DynScaled r;
    double peak = 0.0;
    for (size_t i = 0; i < loudIn.size(); ++i)
    {
        peak = std::max(peak, std::abs(static_cast<double>(loud[0][i])));
        r.apart = std::max(r.apart, std::abs(static_cast<double>(loud[0][i]) - soft[0][i] / scale));
    }
    r.apart /= peak;
    r.quietDb = toDb(dynRms(loud[0], 48000 * 3 + 24000, 48000 * 4) / dynRms(loudIn, 48000 * 3 + 24000, 48000 * 4));
    r.loudDb = toDb(dynRms(loud[0], 48000 * 2 + 24000, 48000 * 3) / dynRms(loudIn, 48000 * 2 + 24000, 48000 * 3));
    return r;
}

} // namespace

// New: with the automatic threshold the expander and the gate act on how far
// under the programme's recent peaks a passage lies, so the output of a
// signal 30 dB lower is the same output 30 dB lower. With the threshold in
// dBFS the quieter signal is treated otherwise.
DSPARK_TEST(Expander_and_gate_automatic_threshold_act_the_same_at_any_level)
{
    for (bool automatic : { true, false })
    {
        const DynScaled expander = dynScaled([&] {
            auto e = std::make_unique<Expander<float>>();
            e->setThreshold(-15.0f);
            e->setRatio(4.0f);
            e->setAttack(1.0f);
            e->setHold(10.0f);
            e->setRelease(50.0f);
            e->setRange(-60.0f);
            e->setAutoThreshold(automatic);
            e->prepare(AudioSpec{ 48000.0, 512, 1 });
            return e;
        });
        const DynScaled gate = dynScaled([&] {
            auto g = std::make_unique<NoiseGate<float>>();
            g->setThreshold(-15.0f);
            g->setAttack(0.5f);
            g->setHold(20.0f);
            g->setRelease(50.0f);
            g->setRange(-60.0f);
            g->setAutoThreshold(automatic);
            g->prepare(48000.0, 1);
            return g;
        });
        std::cout << "  automatic " << automatic << ": expander apart " << expander.apart << ", quiet " << expander.quietDb
                  << " dB, loud " << expander.loudDb << " dB; gate apart " << gate.apart << ", quiet " << gate.quietDb
                  << " dB, loud " << gate.loudDb << " dB\n";
        if (automatic)
        {
            EXPECT_LT(expander.apart, 1e-5);
            EXPECT_LT(gate.apart, 1e-5);
            EXPECT_LT(expander.quietDb, -20.0);     // 25 dB under the peaks, 10 under the threshold
            EXPECT_GT(expander.loudDb, -0.5);
            EXPECT_LT(gate.quietDb, -40.0);
            EXPECT_GT(gate.loudDb, -0.5);
        }
        else
        {
            EXPECT_GT(expander.apart, 0.1);         // the lower signal lies under -15 dBFS throughout
            EXPECT_GT(gate.apart, 0.1);
        }
    }
}

// The reference falls 1 dB per second while the programme sounds and a
// quarter as fast under the closing threshold: a floor 45 dB under the peaks
// stays closed under a -40 dB threshold for some twenty seconds, and is let
// through once the reference has come down to it.
DSPARK_TEST(Expander_and_gate_automatic_threshold_come_down_slowly_in_a_pause)
{
    const int rate = 48000;
    Vec in(static_cast<size_t>(rate) * 36);
    const double floorLevel = 0.3 * std::pow(10.0, -45.0 / 20.0);
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = static_cast<float>((i < static_cast<size_t>(rate) ? 0.3 : floorLevel)
                                   * std::sin(2.0 * kPiD * 440.0 * static_cast<double>(i) / rate));
    const auto gainAt = [&](const Vec& out, int second) {
        return toDb(dynRms(out, rate * second, rate * second + 4800) / dynRms(in, rate * second, rate * second + 4800));
    };

    Expander<float> e;
    e.setThreshold(-40.0f);
    e.setRatio(8.0f);
    e.setHysteresis(2.0f);
    e.setRange(-60.0f);
    e.setAutoThreshold(true);
    e.prepare(AudioSpec{ 48000.0, 512, 1 });
    std::vector<Vec> x { in };
    processInBlocks(x, 512, [&](AudioBufferView<float> b) { e.processBlock(b); });
    EXPECT_LT(gainAt(x[0], 5), -20.0);       // 4 s into the pause: the threshold came down 1 dB
    EXPECT_GT(gainAt(x[0], 34), -1.0);       // 33 s in: 8 dB, past the floor

    NoiseGate<float> g;
    g.setThreshold(-40.0f);
    g.setHysteresis(2.0f);
    g.setRange(-60.0f);
    g.setAutoThreshold(true);
    g.prepare(48000.0, 1);
    std::vector<Vec> y { in };
    processInBlocks(y, 512, [&](AudioBufferView<float> b) { g.processBlock(b); });
    EXPECT_LT(gainAt(y[0], 5), -50.0);
    EXPECT_GT(gainAt(y[0], 34), -1.0);

    // State.
    const auto eb = e.getState();
    Expander<float> e2;
    EXPECT_TRUE(e2.setState(eb.data(), eb.size()));
    EXPECT_TRUE(e2.isAutoThreshold());
    const auto gb = g.getState();
    NoiseGate<float> g2;
    EXPECT_TRUE(g2.setState(gb.data(), gb.size()));
    EXPECT_TRUE(g2.isAutoThreshold());
}

// New: the de-esser's threshold counted from the level of the whole input.
// It then acts on how loud the sibilance is against the voice: the same
// reduction 26 dB lower, where a threshold in dBFS no longer acts at all.
DSPARK_TEST(DeEsser_automatic_threshold_follows_the_voice)
{
    const int n = 48000 * 3;
    const auto voiceAt = [&](double scale) {
        Vec v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            const double t = i / 48000.0;
            const double hiss = (std::fmod(t, 1.0) >= 0.5) ? 0.1 * std::sin(2.0 * kPiD * 7000.0 * t) : 0.0;
            v[static_cast<size_t>(i)] = static_cast<float>(scale * (0.3 * std::sin(2.0 * kPiD * 300.0 * t) + hiss));
        }
        return v;
    };
    const auto run = [&](double scale, bool automatic, float& reduction) {
        auto d = std::make_unique<DeEsser<float>>();
        d->prepare(AudioSpec{ 48000.0, 512, 1 });
        d->setFrequency(7000.0f);
        d->setBandwidth(1.0f);
        d->setThreshold(-20.0f);
        d->setReduction(12.0f);
        d->setAutoThreshold(automatic);
        std::vector<Vec> x { voiceAt(scale) };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { d->processBlock(b); });
        reduction = d->getGainReductionDb();
        return x[0];
    };
    float loudReduction = 0.0f, softReduction = 0.0f, fixedReduction = 0.0f;
    const double scale = 0.05;
    const Vec loud = run(1.0, true, loudReduction);
    const Vec soft = run(scale, true, softReduction);
    (void)run(scale, false, fixedReduction);
    double apart = 0.0;
    for (int i = 0; i < n; ++i)
        apart = std::max(apart, std::abs(static_cast<double>(loud[static_cast<size_t>(i)])
                                         - soft[static_cast<size_t>(i)] / scale));
    std::cout << "  reduction " << loudReduction << " dB, " << softReduction << " dB 26 dB lower, "
              << fixedReduction << " dB there with the threshold in dBFS; outputs apart " << apart << "\n";
    EXPECT_GT(loudReduction, 3.0f);
    EXPECT_NEAR(softReduction, loudReduction, 0.05f);
    EXPECT_LT(apart, 1e-5);
    EXPECT_EQ(fixedReduction, 0.0f);

    DeEsser<float> d;
    d.setAutoThreshold(true);
    const auto blob = d.getState();
    DeEsser<float> other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.isAutoThreshold());
}

// ============================================================================
// DeEsser
// ============================================================================

// Defect: cookbook bell cramping. Old: a -12 dB, one-octave cut at 16 kHz read
// -2.11 dB at 20 kHz (48 kHz) against -8.25 dB for the prototype.
DSPARK_TEST(DeEsser_bell_follows_analog_prototype_near_nyquist)
{
    auto d = std::make_unique<DeEsser<float>>();
    d->prepare(AudioSpec{ 48000.0, 512, 2 });
    d->setFrequency(16000.0f);
    d->setBandwidth(1.0f);
    d->setThreshold(-60.0f);
    d->setReduction(12.0f);
    const int n = 24000;
    std::vector<Vec> x { makeSine(16000.0, 0.3, 48000.0, n), makeSine(20000.0, 0.01, 48000.0, n) };
    processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
    const double q = 1.0 / (2.0 * std::sinh(0.34657359));
    const double g = toDb(toneAmplitude(x[1].data() + n - 9600, 9600, 20000.0, 48000.0) / 0.01);
    EXPECT_NEAR(d->getGainReductionDb(), 12.0f, 0.1f);
    EXPECT_NEAR(g, analogBellDb(20000.0, 16000.0, q, -12.0), 1.5);
}

// Defect: block-relative refresh grid. Old: 4.6e-2 between 7- and 512-sample blocks.
DSPARK_TEST(DeEsser_output_does_not_depend_on_block_partition)
{
    const int n = 48000;
    Vec a(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = i / 48000.0;
        const double env = 0.1 + 0.8 * std::exp(-std::fmod(t, 0.2) * 25.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.4 * std::sin(2.0 * kPiD * 7100.0 * t) + 0.2 * std::sin(2.0 * kPiD * 300.0 * t)));
    }
    auto run = [&](int block) {
        auto d = std::make_unique<DeEsser<float>>();
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        d->setFrequency(7000.0f);
        d->setThreshold(-40.0f);
        d->setReduction(12.0f);
        std::vector<Vec> x { a, a };
        processInBlocks(x, block, [&](AudioBufferView<float> v) { d->processBlock(v); });
        return x[0];
    };
    const Vec ref = run(512);
    for (int block : { 7, 33, 100 })
    {
        const Vec x = run(block);
        double md = 0;
        for (int i = 0; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(i)]) - ref[static_cast<size_t>(i)]));
        EXPECT_EQ(md, 0.0);
    }
}

// Defect: one NaN latched in the detector and the cut filter. Old: NaN for the
// remaining 38000 samples.
DSPARK_TEST(DeEsser_non_finite_sample_does_not_latch)
{
    auto make = [] {
        auto d = std::make_unique<DeEsser<float>>();
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        d->setFrequency(1000.0f);
        d->setThreshold(-30.0f);
        return d;
    };
    auto d = make(), twin = make();
    const int n = 48000;
    Vec s = makeSine(997.0, 0.5, 48000.0, n);
    std::vector<Vec> x { s, s }, y { s, s };
    x[0][10000] = std::numeric_limits<float>::quiet_NaN();
    processInBlocks(x, 256, [&](AudioBufferView<float> v) { d->processBlock(v); });
    processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
    EXPECT_TRUE(allFinite(x[0].data() + 12000, n - 12000));
    double md = 0;
    for (int i = n - 4800; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
    EXPECT_LT(md, 1e-4);
}

// New: the mixed bell (setMixedBell()). With no reduction it returns its
// input bit for bit, and its output does not depend on how the stream is cut.
DSPARK_TEST(DeEsser_mixed_bell_at_rest_and_any_partition)
{
    const int n = 24000;
    const Vec quiet = makeSine(7000.0, 0.001, 48000.0, n);
    {
        auto d = std::make_unique<DeEsser<float>>();
        d->setMixedBell(true);
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        d->setThreshold(-20.0f);
        std::vector<Vec> x { quiet, quiet };
        processInBlocks(x, 480, [&](AudioBufferView<float> v) { d->processBlock(v); });
        EXPECT_TRUE(std::memcmp(x[0].data(), quiet.data(), sizeof(float) * static_cast<size_t>(n)) == 0);
        EXPECT_TRUE(d->isMixedBell());
    }
    Vec a(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = i / 48000.0;
        const double env = 0.1 + 0.8 * std::exp(-std::fmod(t, 0.2) * 25.0);
        a[static_cast<size_t>(i)] = static_cast<float>(env * (0.4 * std::sin(2.0 * kPiD * 7100.0 * t) + 0.2 * std::sin(2.0 * kPiD * 300.0 * t)));
    }
    auto run = [&](int block) {
        auto d = std::make_unique<DeEsser<float>>();
        d->setMixedBell(true);
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        d->setFrequency(7000.0f);
        d->setThreshold(-40.0f);
        d->setReduction(12.0f);
        std::vector<Vec> x { a, a };
        processInBlocks(x, block, [&](AudioBufferView<float> v) { d->processBlock(v); });
        return x[0];
    };
    const Vec ref = run(512);
    for (int block : { 7, 33, 100, 1024 })
    {
        const Vec x = run(block);
        double md = 0;
        for (int i = 0; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[static_cast<size_t>(i)]) - ref[static_cast<size_t>(i)]));
        EXPECT_EQ(md, 0.0);
    }
}

// New: at the centre the mixed bell cuts exactly the reduction, as the
// default bell does.
DSPARK_TEST(DeEsser_mixed_bell_cuts_the_centre_by_the_reduction)
{
    for (bool mixed : { false, true })
    {
        auto d = std::make_unique<DeEsser<float>>();
        d->setMixedBell(mixed);
        d->prepare(AudioSpec{ 48000.0, 512, 1 });
        d->setFrequency(7000.0f);
        d->setBandwidth(1.0f);
        d->setThreshold(-40.0f);
        d->setReduction(12.0f);
        const int n = 48000;
        std::vector<Vec> x { makeSine(7000.0, 0.5, 48000.0, n) };
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        const double g = toDb(toneAmplitude(x[0].data() + n - 9600, 9600, 7000.0, 48000.0) / 0.5);
        EXPECT_NEAR(d->getGainReductionDb(), 12.0f, 0.1f);
        EXPECT_NEAR(g, -12.0, 0.3);
    }
}

// Defect of the default bell that the mixed one does not have: designed
// again every 16 samples, its gain moves in steps, and a tone under a bell
// that moves fast grows lines 3 kHz away (48000 / 16). A 6 kHz tone that jumps
// 28 dB 25 times a second, on edges of 4 ms (smooth: its own spectrum is down
// by then 3 kHz away), under a bell with a 5 ms release. Measured: -46 to
// -52 dBc stepped for releases of 1 and 5 ms whatever the attack, -64 to -96
// mixed (the floor of the signal's own edges); with the 20 ms default both
// sit on that floor.
DSPARK_TEST(DeEsser_mixed_bell_moves_without_steps)
{
    const int n = 96000;
    auto lines = [&](bool mixed) {
        auto d = std::make_unique<DeEsser<float>>();
        d->setMixedBell(mixed);
        d->prepare(AudioSpec{ 48000.0, 512, 1 });
        d->setFrequency(6000.0f);
        d->setBandwidth(1.0f);
        d->setThreshold(-30.0f);
        d->setReduction(12.0f);
        d->setAttack(0.5f);
        d->setRelease(5.0f);
        Vec s(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            const double t = i / 48000.0;
            const double phase = std::fmod(t, 0.04);              // 40 ms: 16 low, 4 up, 16 high, 4 down
            double up = 0.0;
            if (phase < 0.016) up = 0.0;
            else if (phase < 0.020) up = 0.5 - 0.5 * std::cos(kPiD * (phase - 0.016) / 0.004);
            else if (phase < 0.036) up = 1.0;
            else up = 0.5 + 0.5 * std::cos(kPiD * (phase - 0.036) / 0.004);
            const double amp = 0.02 + 0.48 * up;
            s[static_cast<size_t>(i)] = static_cast<float>(amp * std::sin(2.0 * kPiD * 6000.0 * t));
        }
        std::vector<Vec> x { s };
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        const float* y = x[0].data() + 48000;
        const double carrier = toneAmplitude(y, 48000, 6000.0, 48000.0);
        double worst = 0.0;
        for (int k = -4; k <= 4; ++k)
            for (double centre : { 3000.0, 9000.0 })
                worst = std::max(worst, toneAmplitude(y, 48000, centre + 25.0 * k, 48000.0));
        return toDb(worst / carrier);
    };
    const double stepped = lines(false), mixed = lines(true);
    std::cout << "  [mixed bell] lines 3 kHz away: stepped " << stepped << " dBc, mixed " << mixed << " dBc\n";
    EXPECT_LT(mixed, -80.0);
    EXPECT_LT(mixed, stepped - 25.0);
}

// ============================================================================
// NoiseGate
// ============================================================================

// Defect: an external sidechain shorter than the audio was read past its end.
// Here the 16-sample view sits at the start of a loud buffer, so the old
// over-read opened the gate (output = input) where the key is silent.
DSPARK_TEST(NoiseGate_short_sidechain_falls_back_to_internal_key)
{
    NoiseGate<float> g;
    g.setThreshold(-40.0f);
    g.prepare(48000.0, 1);
    Vec keyStorage(512, 0.9f);
    std::fill(keyStorage.begin(), keyStorage.begin() + 16, 0.0f);
    double inRms = 0, outRms = 0;
    for (int b = 0; b < 40; ++b)
    {
        Vec a = makeSine(1000.0, 0.001, 48000.0, 512); // -60 dBFS: below the threshold
        const double r = rmsOf(a.data(), 512);
        float* pa = a.data();
        float* pk = keyStorage.data();
        AudioBufferView<float> va(&pa, 1, 512), vk(&pk, 1, 16);
        g.processBlock(va, vk);
        if (b >= 20) { inRms += r; outRms += rmsOf(a.data(), 512); }
    }
    EXPECT_LT(outRms, 0.1 * inRms);
}

// Defect: processSample() latched a NaN in the envelope. Old: stuck open,
// passing a -60 dBFS tone at 0.001 where a closed gate gives about 1e-7.
DSPARK_TEST(NoiseGate_processSample_non_finite_does_not_stick_open)
{
    NoiseGate<float> g;
    g.setThreshold(-30.0f);
    g.setRelease(10.0f);
    g.setHold(5.0f);
    g.prepare(48000.0, 1);
    Vec s = makeSine(997.0, 0.5, 48000.0, 4800);
    for (float v : s) (void)g.processSample(v);
    (void)g.processSample(std::numeric_limits<float>::quiet_NaN());
    float o = 0.0f;
    for (int i = 0; i < 48000; ++i) o = g.processSample((i % 2) ? 0.001f : -0.001f);
    EXPECT_LT(std::abs(o), 1e-5f);
}

// Defect: Frequency mode latched a NaN in its filters. Old: NaN for the
// remaining 38000 samples.
DSPARK_TEST(NoiseGate_frequency_mode_non_finite_does_not_latch)
{
    auto make = [] {
        auto g = std::make_unique<NoiseGate<float>>();
        g->setThreshold(-30.0f);
        g->setGateMode(NoiseGate<float>::GateMode::Frequency);
        g->prepare(48000.0, 2);
        return g;
    };
    auto g = make(), twin = make();
    const int n = 48000;
    Vec s = makeSine(997.0, 0.5, 48000.0, n);
    std::vector<Vec> x { s, s }, y { s, s };
    x[0][10000] = std::numeric_limits<float>::quiet_NaN();
    processInBlocks(x, 256, [&](AudioBufferView<float> v) { g->processBlock(v); });
    processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
    EXPECT_TRUE(allFinite(x[0].data() + 10001, n - 10001));
    double md = 0;
    for (int i = n - 4800; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
    EXPECT_LT(md, 1e-4);
}

// ============================================================================
// Compressor
// ============================================================================

// Defect: one Inf sample made the dB envelope inf - inf; one NaN froze the
// Hilbert detector (mono) and the FET colour state. Old: NaN output for the
// remaining 38000 samples (Peak + Inf, FET colour + NaN), no compression at all
// afterwards (Hilbert + NaN).
DSPARK_TEST(Compressor_non_finite_sample_does_not_latch)
{
    using C = Compressor<float>;
    struct Case { C::DetectorType det; C::Character ch; float bad; };
    const Case cases[] = {
        { C::DetectorType::Peak, C::Character::Clean, std::numeric_limits<float>::infinity() },
        { C::DetectorType::Hilbert, C::Character::Clean, std::numeric_limits<float>::quiet_NaN() },
        { C::DetectorType::SplitPolarity, C::Character::Clean, std::numeric_limits<float>::quiet_NaN() },
        { C::DetectorType::Peak, C::Character::FET, std::numeric_limits<float>::quiet_NaN() },
    };
    for (const auto& k : cases)
    {
        auto make = [&] {
            auto c = std::make_unique<C>();
            c->setThreshold(-24.0f);
            c->setDetector(k.det);
            c->setCharacter(k.ch);
            c->setCharacterColor(1.0f);
            c->prepare(AudioSpec{ 48000.0, 512, 1 });
            return c;
        };
        auto c = make(), twin = make();
        const int n = 48000;
        Vec s = makeSine(997.0, 0.5, 48000.0, n);
        std::vector<Vec> x { s }, y { s };
        x[0][10000] = k.bad;
        processInBlocks(x, 256, [&](AudioBufferView<float> v) { c->processBlock(v); });
        processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
        EXPECT_TRUE(allFinite(x[0].data() + 10001, n - 10001));
        double md = 0;
        for (int i = n - 4800; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
        EXPECT_LT(md, 1e-4);
    }
}

// Defect: the explicit feedback loop fed the detector after makeup. Old: RMS
// detector, makeup +12 dB: -16.5 dB out instead of -7.5 (22.5 dB of reduction
// instead of 13.5).
DSPARK_TEST(Compressor_feedback_detector_excludes_makeup)
{
    using C = Compressor<float>;
    for (auto det : { C::DetectorType::Rms, C::DetectorType::TruePeak })
    {
        auto c = std::make_unique<C>();
        c->setThreshold(-24.0f); c->setRatio(4.0f); c->setTopology(C::Topology::FeedBack);
        c->setDetector(det); c->setMakeupGain(12.0f); c->setAttack(5.0f); c->setRelease(50.0f);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        const float v = static_cast<float>(std::pow(10.0, -6.0 / 20.0));
        std::vector<Vec> x(2, Vec(96000, v));
        processInBlocks(x, 256, [&](AudioBufferView<float> b) { c->processBlock(b); });
        EXPECT_NEAR(toDb(std::abs(x[0][95999])), -7.5, 0.1);
    }
}

// Defect: Upward + FeedBack applied the feed-forward law to the output, which
// caps the observed ratio near 2:1. Old: in -40 dB, threshold -20, ratio 8:
// out -30.67 dB (observed 1.87:1) instead of -22.5.
DSPARK_TEST(Compressor_upward_feedback_honours_ratio)
{
    using C = Compressor<float>;
    for (float knee : { 0.0f, 6.0f })
    {
        auto c = std::make_unique<C>();
        c->setThreshold(-20.0f); c->setRatio(8.0f); c->setKnee(knee); c->setMode(C::Mode::Upward);
        c->setTopology(C::Topology::FeedBack); c->setAttack(5.0f); c->setRelease(100.0f);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        const float v = static_cast<float>(std::pow(10.0, -40.0 / 20.0));
        std::vector<Vec> x(2, Vec(96000, v));
        processInBlocks(x, 256, [&](AudioBufferView<float> b) { c->processBlock(b); });
        EXPECT_NEAR(toDb(std::abs(x[0][95999])), -22.5, 0.05);
    }
}

// Defect: processSample() never recomputed the sidechain filter, so a cutoff
// set after prepare() stayed at the old value. Old: HPF 2 kHz on a 100 Hz tone
// left rms 0.043 (strong reduction) instead of 0.354.
DSPARK_TEST(Compressor_processSample_applies_sidechain_hpf_set_after_prepare)
{
    Compressor<float> c;
    c.setThreshold(-30.0f); c.setRatio(8.0f); c.setAttack(1.0f); c.setRelease(20.0f);
    c.prepare(AudioSpec{ 48000.0, 512, 2 });
    c.setSidechainHPF(true, 2000.0f);
    const int n = 48000;
    Vec s = makeSine(100.0, 0.5, 48000.0, n);
    double e = 0;
    for (int i = 0; i < n; ++i)
    {
        const float o = c.processSample(s[static_cast<size_t>(i)], 0);
        if (i >= n / 2) e += static_cast<double>(o) * o;
    }
    EXPECT_GT(std::sqrt(e / (n / 2)), 0.3);
}

// Defect: float dB envelopes stall once the per-sample step drops below an
// ulp. Old: a 3 s release at 192 kHz parked at -26.82 dB, target -26.25.
DSPARK_TEST(Compressor_long_release_settles_on_target)
{
    const double fs = 192000.0;
    Compressor<float> c;
    c.setThreshold(-40.0f); c.setRatio(8.0f); c.setAttack(1.0f); c.setRelease(3000.0f);
    c.prepare(AudioSpec{ fs, 4096, 1 });
    const float l1 = static_cast<float>(std::pow(10.0, -4.0 / 20.0));
    const float l2 = static_cast<float>(std::pow(10.0, -10.0 / 20.0));
    Vec buf(4096);
    const int total = static_cast<int>(fs * 30.0);
    for (int off = 0; off < total; off += 4096)
    {
        std::fill(buf.begin(), buf.end(), off < fs * 0.5 ? l1 : l2);
        float* p = buf.data();
        AudioBufferView<float> v(&p, 1, 4096);
        c.processBlock(v);
    }
    EXPECT_NEAR(c.getGainReductionDb(), -26.25f, 0.05f);
}

// Defect: a live character change re-weighted the fast/slow envelopes without
// re-seating them. Old: Clean to Opto at a steady -14.66 dB stepped to -8.79 dB
// in one sample.
DSPARK_TEST(Compressor_live_character_change_is_click_free)
{
    using C = Compressor<float>;
    for (auto ch : { C::Character::Opto, C::Character::FET })
    {
        Compressor<float> c;
        c.setThreshold(-30.0f); c.setRatio(4.0f); c.setAttack(5.0f); c.setRelease(100.0f);
        c.prepare(AudioSpec{ 48000.0, 512, 2 });
        std::vector<Vec> x(2, Vec(512 * 120, 0.3f));
        int cnt = 0;
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { if (++cnt == 100) c.setCharacter(ch); c.processBlock(v); });
        const int at = 99 * 512;
        EXPECT_NEAR(toDb(x[0][static_cast<size_t>(at)] / 0.3), toDb(x[0][static_cast<size_t>(at - 1)] / 0.3), 0.1);
    }
}

// ============================================================================
// Limiter
// ============================================================================

// Defect: adaptive release only counted frames where the envelope was still
// falling, which sustained limiting barely produces. Old: 87 ms recovery with
// and without adaptive release after 2 s of limiting.
DSPARK_TEST(Limiter_adaptive_release_slows_after_sustained_limiting)
{
    auto recovery = [](bool adaptive) {
        Limiter<float> l;
        l.setCeiling(-6.0f); l.setRelease(50.0f); l.setAdaptiveRelease(adaptive);
        l.prepare(48000.0, 1);
        const int n = static_cast<int>(48000 * 2.6);
        Vec x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            const double t = i / 48000.0;
            x[static_cast<size_t>(i)] = static_cast<float>((t < 2.0 ? 1.0 : 0.05) * std::sin(2.0 * kPiD * 1000.0 * t));
        }
        std::vector<Vec> c { x };
        processInBlocks(c, 256, [&](AudioBufferView<float> v) { l.processBlock(v); });
        const int k = 96000 + l.getLatency();
        for (int i = k; i + 48 < n; i += 48)
            if (toDb(rmsOf(c[0].data() + i, 48)) > toDb(0.05 / std::sqrt(2.0)) - 1.0) return i - k;
        return n;
    };
    EXPECT_GT(recovery(true), 2 * recovery(false));
}

// ============================================================================
// Expander
// ============================================================================

// Defect: the smoothed rectifier under-read a steady tone by about 1 dB, which
// the ratio multiplied. Old: R = 3, 10 dB under the threshold: -21.31 dB at
// 100 Hz, -22.15 dB at 1 kHz, -22.36 dB at 10 kHz instead of -20.
DSPARK_TEST(Expander_static_curve_holds_across_frequency)
{
    for (double f : { 100.0, 1000.0, 10000.0 })
    {
        Expander<float> x;
        x.setThreshold(-30.0f); x.setRatio(3.0f); x.setHysteresis(0.0f); x.setAttack(5.0f);
        x.setRelease(50.0f); x.setHold(0.0f); x.setRange(-80.0f);
        x.prepare(AudioSpec{ 48000.0, 512, 2 });
        const int n = 96000;
        const double a = std::pow(10.0, -40.0 / 20.0);
        std::vector<Vec> c { makeSine(f, a, 48000.0, n), makeSine(f, a, 48000.0, n) };
        processInBlocks(c, 256, [&](AudioBufferView<float> v) { x.processBlock(v); });
        const int len = static_cast<int>(std::round(24000.0 * f / 48000.0) * 48000.0 / f);
        EXPECT_NEAR(toDb(toneAmplitude(c[0].data() + n - len, len, f, 48000.0) / a), -20.0, 0.1);
    }
}

// Defect: one Inf sample made the envelope inf - inf. Old: muted for good
// (output rms 0.0005 against 0.41).
DSPARK_TEST(Expander_non_finite_sample_does_not_mute)
{
    auto make = [] {
        auto x = std::make_unique<Expander<float>>();
        x->setThreshold(-30.0f);
        x->prepare(AudioSpec{ 48000.0, 512, 2 });
        return x;
    };
    auto e = make(), twin = make();
    const int n = 48000;
    Vec s = makeSine(997.0, 0.5, 48000.0, n);
    std::vector<Vec> x { s, s }, y { s, s };
    x[0][10000] = std::numeric_limits<float>::infinity();
    processInBlocks(x, 256, [&](AudioBufferView<float> v) { e->processBlock(v); });
    processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
    EXPECT_NEAR(rmsOf(x[0].data() + n - 4800, 4800), rmsOf(y[0].data() + n - 4800, 4800), 1e-3);
}

// ============================================================================
// TransientDesigner
// ============================================================================

// Defect: amounts applied per block without smoothing. Old: setAttack(100) one
// block into a burst stepped the output from -0.241 to -3.80 in one sample.
DSPARK_TEST(TransientDesigner_amount_change_is_click_free)
{
    TransientDesigner<float> t;
    t.prepare(AudioSpec{ 48000.0, 512, 1 });
    const int n = 4096;
    Vec x(static_cast<size_t>(n), 0.0f);
    for (int i = 1024; i < n; ++i) x[static_cast<size_t>(i)] = static_cast<float>(0.25 * std::sin(2.0 * kPiD * 300.0 * i / 48000.0));
    std::vector<Vec> c { x };
    int cnt = 0;
    processInBlocks(c, 64, [&](AudioBufferView<float> v) { if (++cnt == 18) t.setAttack(100.0f); t.processBlock(v); });
    const int at = 17 * 64;
    EXPECT_LT(std::abs(static_cast<double>(c[0][static_cast<size_t>(at)]) - c[0][static_cast<size_t>(at - 1)]), 0.02);
}

// Defect: one NaN latched in the envelopes and the analytic detector. Old: NaN
// for the remaining 38000 samples.
DSPARK_TEST(TransientDesigner_non_finite_sample_does_not_latch)
{
    auto make = [] {
        auto t = std::make_unique<TransientDesigner<float>>();
        t->prepare(AudioSpec{ 48000.0, 512, 2 });
        t->setAttack(50.0f);
        return t;
    };
    auto t = make(), twin = make();
    const int n = 48000;
    Vec s = makeSine(997.0, 0.5, 48000.0, n);
    std::vector<Vec> x { s, s }, y { s, s };
    x[0][10000] = std::numeric_limits<float>::quiet_NaN();
    processInBlocks(x, 256, [&](AudioBufferView<float> v) { t->processBlock(v); });
    processInBlocks(y, 256, [&](AudioBufferView<float> v) { twin->processBlock(v); });
    EXPECT_TRUE(allFinite(x[0].data() + 10001, n - 10001));
    double md = 0;
    for (int i = n - 4800; i < n; ++i) md = std::max(md, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
    EXPECT_LT(md, 1e-4);
}

// ============================================================================
// CrossoverFilter
// ============================================================================

// Defect: band views with fewer channels than the input were written through
// a null channel pointer. Old: stereo input with mono band views crashed.
DSPARK_TEST(CrossoverFilter_narrow_band_views_are_safe)
{
    for (auto mode : { CrossoverFilter<float>::FilterMode::MinimumPhase, CrossoverFilter<float>::FilterMode::LinearPhase })
    {
        CrossoverFilter<float> x;
        x.setNumBands(2);
        x.setFilterMode(mode);
        x.prepare(AudioSpec{ 48000.0, 256, 2 });
        Vec l(256, 0.25f), r(256, 0.25f), o0(256), o1(256);
        float* ip[2] = { l.data(), r.data() };
        AudioBufferView<float> in(ip, 2, 256);
        float* p0 = o0.data();
        float* p1 = o1.data();
        AudioBufferView<float> out[2] = { AudioBufferView<float>(&p0, 1, 256), AudioBufferView<float>(&p1, 1, 256) };
        EXPECT_EQ(x.processBlock(in, out, 2), 2);
        EXPECT_TRUE(allFinite(o0.data(), 256));
        EXPECT_TRUE(allFinite(o1.data(), 256));
    }
}

// Defect: one NaN latched in the IIR sections of every band. Old: 92060
// non-finite samples per band over the next 100k.
DSPARK_TEST(CrossoverFilter_non_finite_input_does_not_latch)
{
    CrossoverFilter<float> x;
    x.setNumBands(3);
    x.prepare(AudioSpec{ 48000.0, 512, 2 });
    Vec s = makeSine(997.0, 0.5, 48000.0, 512 * 40);
    bool lateBad = false;
    for (int b = 0; b < 40; ++b)
    {
        Vec l(s.begin() + b * 512, s.begin() + (b + 1) * 512), r = l;
        if (b == 10) l[100] = std::numeric_limits<float>::quiet_NaN();
        Vec o[3][2];
        float* pp[3][2];
        std::vector<AudioBufferView<float>> views;
        for (int k = 0; k < 3; ++k)
        {
            o[k][0].assign(512, 0.0f); o[k][1].assign(512, 0.0f);
            pp[k][0] = o[k][0].data(); pp[k][1] = o[k][1].data();
            views.emplace_back(pp[k], 2, 512);
        }
        float* ip[2] = { l.data(), r.data() };
        AudioBufferView<float> in(ip, 2, 512);
        x.processBlock(in, views.data(), 3);
        if (b > 10)
            for (int k = 0; k < 3; ++k)
                if (!allFinite(o[k][0].data(), 512)) lateBad = true;
    }
    EXPECT_FALSE(lateBad);
}

// Defect: rounded Butterworth Q values (0.7071, 0.5412, 1.3066). Old: the
// 8-band sum deviated 3.35e-4 dB (LR24) and 1.05e-3 dB (LR48) from flat.
DSPARK_TEST(CrossoverFilter_band_sum_is_flat)
{
    for (int order : { 24, 48 })
    {
        CrossoverFilter<float> x;
        x.setNumBands(8);
        x.setOrder(order);
        x.prepare(AudioSpec{ 48000.0, 4096, 1 });
        const int n = 1 << 15;
        Vec in(static_cast<size_t>(n), 0.0f);
        in[8] = 1.0f;
        std::vector<Vec> o(8, Vec(static_cast<size_t>(n)));
        for (int off = 0; off < n; off += 4096)
        {
            float* ip = in.data() + off;
            AudioBufferView<float> iv(&ip, 1, 4096);
            float* p[8];
            std::vector<AudioBufferView<float>> views;
            for (int b = 0; b < 8; ++b) { p[b] = o[static_cast<size_t>(b)].data() + off; views.emplace_back(&p[b], 1, 4096); }
            x.processBlock(iv, views.data(), 8);
        }
        std::vector<double> sum(static_cast<size_t>(n), 0.0), spec(static_cast<size_t>(n) + 2);
        for (int b = 0; b < 8; ++b)
            for (int i = 0; i < n; ++i) sum[static_cast<size_t>(i)] += o[static_cast<size_t>(b)][static_cast<size_t>(i)];
        FFTReal<double> fft(n);
        fft.forward(sum.data(), spec.data());
        double worst = 0;
        for (int k = 0; k <= n / 2; ++k)
        {
            const double re = spec[static_cast<size_t>(2 * k)], im = spec[static_cast<size_t>(2 * k + 1)];
            worst = std::max(worst, std::abs(10.0 * std::log10(re * re + im * im)));
        }
        EXPECT_LT(worst, 1e-5);
    }
}

// ============================================================================
// Round 2: decoupled Clean ballistics, feed-forward block path, feedback RMS
// loop floor, fixed-length linear-phase crossover, block-independent glide.
// ============================================================================

namespace {

// Least-squares fit of a tone of known frequency over [from, to); returns the
// residual-to-tone power ratio (THD+N, dB) and the tone's peak level (dBFS).
template <typename T>
std::pair<double, double> toneThdnAndLevel(const std::vector<T>& y, int from, int to, double f, double fs)
{
    double ss = 0, sc = 0, cc = 0, ys = 0, yc = 0;
    for (int i = from; i < to; ++i)
    {
        const double s = std::sin(2.0 * kPiD * f * i / fs), c = std::cos(2.0 * kPiD * f * i / fs);
        const double v = static_cast<double>(y[static_cast<size_t>(i)]);
        ss += s * s; sc += s * c; cc += c * c; ys += v * s; yc += v * c;
    }
    const double det = ss * cc - sc * sc;
    const double a = (ys * cc - yc * sc) / det, b = (yc * ss - ys * sc) / det;
    double err = 0, sig = 0;
    for (int i = from; i < to; ++i)
    {
        const double fit = a * std::sin(2.0 * kPiD * f * i / fs) + b * std::cos(2.0 * kPiD * f * i / fs);
        const double e = static_cast<double>(y[static_cast<size_t>(i)]) - fit;
        err += e * e; sig += fit * fit;
    }
    return { 10.0 * std::log10(err / sig), 20.0 * std::log10(std::sqrt(a * a + b * b)) };
}

// Runs a mono buffer through a CrossoverFilter in blocks (random sizes in
// [1, block] when `randomSizes`), collecting every band.
std::vector<Vec> crossoverSplit(CrossoverFilter<float>& x, const Vec& in, int bands, int block,
                                bool randomSizes = false)
{
    const int n = static_cast<int>(in.size());
    std::vector<Vec> out(static_cast<size_t>(bands), Vec(static_cast<size_t>(n), 0.0f));
    Vec work = in;
    uint32_t rng = 0x2468ACE1u;
    for (int off = 0; off < n;)
    {
        int m = block;
        if (randomSizes)
        {
            rng = rng * 1664525u + 1013904223u;
            m = 1 + static_cast<int>((rng >> 8) % static_cast<uint32_t>(block));
        }
        m = std::min(m, n - off);
        float* ip = work.data() + off;
        AudioBufferView<float> iv(&ip, 1, m);
        std::vector<float*> p(static_cast<size_t>(bands));
        std::vector<AudioBufferView<float>> views;
        for (int b = 0; b < bands; ++b)
        {
            p[static_cast<size_t>(b)] = out[static_cast<size_t>(b)].data() + off;
            views.emplace_back(&p[static_cast<size_t>(b)], 1, m);
        }
        x.processBlock(iv, views.data(), bands);
        off += m;
    }
    return out;
}

// Zero-phase magnitude (dB) of a kernel centred at `centre`, at `f`.
double kernelResponseDb(const Vec& h, int centre, int halfSpan, double f, double fs)
{
    double re = 0, im = 0;
    for (int i = -halfSpan; i < halfSpan; ++i)
    {
        const double v = h[static_cast<size_t>(centre + i)];
        const double ph = -2.0 * kPiD * f * i / fs;
        re += v * std::cos(ph); im += v * std::sin(ph);
    }
    return 10.0 * std::log10(re * re + im * im);
}

} // namespace

// Defect: the Clean peak detector's branching one-pole released toward every
// zero crossing of a tone, so the gain rode the waveform: a 1 kHz tone 14 dB
// over a 4:1 threshold at 5/100 ms measured THD+N -65.5 dB, and the level sat
// 0.8 dB above the peak-referenced static curve (the envelope sagged between
// peaks). The opt-in decoupled ballistics measure -102 dB and land on the
// curve.
DSPARK_TEST(Compressor_clean_peak_gain_does_not_ride_a_tone)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 3.0);
    Compressor<double> comp;
    comp.prepare(AudioSpec{ fs, 512, 1 });
    comp.setThreshold(-20.0);
    comp.setRatio(4.0);
    comp.setAttack(5.0);
    comp.setRelease(100.0);
    comp.setCleanBallistics(Compressor<double>::CleanBallistics::Decoupled);
    // Default detector (Peak) and character (Clean).

    const double amp = std::pow(10.0, -6.0 / 20.0); // peak 14 dB over the threshold
    std::vector<double> y(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) y[static_cast<size_t>(i)] = amp * std::sin(2.0 * kPiD * 1000.0 * i / fs);
    for (int pos = 0; pos < n; pos += 512)
    {
        double* chans[1] = { y.data() + pos };
        comp.processBlock(AudioBufferView<double>(chans, 1, std::min(512, n - pos)));
    }
    const auto [thdn, level] = toneThdnAndLevel(y, n - 48000, n, 1000.0, fs);
    std::cout << "  THD+N " << thdn << " dB, peak level " << level << " dBFS\n";
    EXPECT_LT(thdn, -95.0);                       // branching: -65.5 dB
    EXPECT_NEAR(level, -6.0 - 14.0 * 0.75, 0.05); // branching: 0.8 dB above
}

// The decoupled ballistics keep the knob contract: attack and release t63 of
// the dB envelope on the knob values (a hold stage plus a smoothing pole, with
// the hold's release solved so the series still crosses 1/e at the knob).
DSPARK_TEST(Compressor_clean_decoupled_ballistics_keep_t63_on_the_knobs)
{
    const double fs = 48000.0;
    constexpr int kB = 4; // fine meter grid: the 1 ms attack is only 48 samples
    const std::pair<double, double> times[] = { { 5.0, 100.0 }, { 1.0, 30.0 }, { 40.0, 60.0 } };
    for (const auto& [att, rel] : times)
    {
        Compressor<double> comp;
        comp.prepare(AudioSpec{ fs, 64, 1 });
        comp.setThreshold(-20.0);
        comp.setRatio(4.0);
        comp.setAttack(att);
        comp.setRelease(rel);
        comp.setCleanBallistics(Compressor<double>::CleanBallistics::Decoupled);
        // A constant level gives an ideal step to the static target.
        const int up = static_cast<int>(fs * 2.0), down = static_cast<int>(fs * 1.0);
        std::vector<double> gr;
        std::vector<double> x(kB);
        auto run = [&](double level, int count) {
            for (int k = 0; k < count; k += kB)
            {
                std::fill(x.begin(), x.end(), level);
                double* ch[1] = { x.data() };
                comp.processBlock(AudioBufferView<double>(ch, 1, kB));
                gr.push_back(comp.getGainReductionDb());
            }
        };
        run(1.0, up);   // 0 dBFS: 20 dB over, 15 dB of reduction
        run(0.0, down);
        const size_t edge = static_cast<size_t>(up / kB);
        const double depth = gr[edge - 1];
        EXPECT_NEAR(depth, -15.0, 1e-3);
        // Interpolated t63 crossings on the kB-sample meter (sampled at the
        // end of each block: the crossing time is measured from that grid).
        auto t63 = [&](size_t from, size_t to, double start, double end) {
            const double level = end + (start - end) * std::exp(-1.0);
            double prev = start;
            for (size_t b = from; b < to; ++b)
            {
                const double a = prev - level, c = gr[b] - level;
                if ((a < 0) != (c < 0))
                    return (static_cast<double>(b - from) + a / (a - c)) * kB / fs * 1000.0;
                prev = gr[b];
            }
            return -1.0;
        };
        const double attMs = t63(0, edge, 0.0, depth);
        const double relMs = t63(edge, gr.size(), depth, 0.0);
        std::cout << "  " << att << "/" << rel << " ms: attack t63 " << attMs << " ms, release t63 " << relMs << " ms\n";
        EXPECT_NEAR(attMs, att, att * 0.03 + 0.05);
        EXPECT_NEAR(relMs, rel, rel * 0.03 + 0.05);
    }
}

// The feed-forward block path runs its passes over chunks of frames; each
// frame does the arithmetic of the per-sample path. Pinned bit-exact against
// processSample() on a mono stream with live parameter automation, for the
// decoupled Clean ballistics, the Opto dual envelope and the branching option.
DSPARK_TEST(Compressor_feed_forward_block_path_matches_processSample)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 1.0);
    uint32_t rng = 12345u;
    std::vector<float> x(static_cast<size_t>(n));
    for (auto& v : x)
    {
        rng = rng * 1664525u + 1013904223u;
        v = 0.4f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
    }
    for (int setup = 0; setup < 3; ++setup)
    {
        Compressor<float> block, sample;
        for (auto* c : { &block, &sample })
        {
            c->prepare(AudioSpec{ fs, 256, 1 });
            c->setThreshold(-24.0f);
            c->setRatio(3.0f);
            c->setKnee(6.0f);
            c->setAttack(2.0f);
            c->setRelease(80.0f);
            if (setup == 1) c->setCharacter(Compressor<float>::Character::Opto);
            if (setup == 2) c->setCleanBallistics(Compressor<float>::CleanBallistics::Decoupled);
        }
        std::vector<float> yb = x, ys = x;
        for (int pos = 0; pos < n; pos += 256)
        {
            if (pos == n / 2) // a threshold glide through the smoothers
                for (auto* c : { &block, &sample }) c->setThreshold(-30.0f);
            const int m = std::min(256, n - pos);
            float* ch[1] = { yb.data() + pos };
            block.processBlock(AudioBufferView<float>(ch, 1, m));
            for (int i = pos; i < pos + m; ++i)
                ys[static_cast<size_t>(i)] = sample.processSample(ys[static_cast<size_t>(i)], 0);
        }
        float maxDiff = 0.0f;
        for (int i = 0; i < n; ++i)
            maxDiff = std::max(maxDiff, std::abs(yb[static_cast<size_t>(i)] - ys[static_cast<size_t>(i)]));
        EXPECT_EQ(maxDiff, 0.0f);
    }
}

// Defect: in FeedBack with the RMS detector the loop ran faster than the
// detector's window could report its own effect, so a release shorter than
// the window overshot and rang: a 12 dB level drop overshot the settled gain
// by 3.0 dB (window 10 ms, release 1 ms) and a rise over-compressed by 4.5 dB
// (attack 0.5 ms), with ~40 direction changes. The loop advance is now capped
// at the critically damped limit for the window's delay (twice that on the
// attack side): no overshoot either way.
DSPARK_TEST(Compressor_feedback_rms_loop_does_not_ring)
{
    const double fs = 48000.0;
    const std::pair<float, float> times[] = { { 0.5f, 1.0f }, { 5.0f, 5.0f } };
    for (const auto& [att, rel] : times)
    {
        Compressor<float> comp;
        comp.prepare(AudioSpec{ fs, 64, 1 });
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        comp.setAttack(att);
        comp.setRelease(rel);
        comp.setTopology(Compressor<float>::Topology::FeedBack);
        comp.setDetector(Compressor<float>::DetectorType::Rms);
        comp.setRmsWindow(10.0f);
        const int seg = static_cast<int>(fs * 1.0);
        std::vector<float> gr;
        std::vector<float> buf(64);
        int t = 0;
        for (const float amp : { 1.0f, 0.25f, 1.0f })
        {
            for (int k = 0; k < seg; k += 64)
            {
                for (int i = 0; i < 64; ++i, ++t)
                    buf[static_cast<size_t>(i)] = amp * static_cast<float>(std::sin(2.0 * kPiD * 1000.0 * t / fs));
                float* ch[1] = { buf.data() };
                comp.processBlock(AudioBufferView<float>(ch, 1, 64));
                gr.push_back(comp.getGainReductionDb());
            }
        }
        const size_t b1 = static_cast<size_t>(seg / 64), b2 = 2 * b1;
        const float hi = gr[b1 - 1], lo = gr[b2 - 1];
        float overshoot = 0.0f, undershoot = 0.0f;
        for (size_t b = b1; b < b2; ++b) overshoot = std::max(overshoot, gr[b] - lo);
        for (size_t b = b2; b < gr.size(); ++b) undershoot = std::max(undershoot, hi - gr[b]);
        std::cout << "  attack " << att << " release " << rel << ": overshoot " << overshoot
                  << " dB, undershoot " << undershoot << " dB\n";
        EXPECT_NEAR(hi, -12.74f, 0.05f); // the static curve on the RMS level
        EXPECT_LT(overshoot, 0.05f);
        EXPECT_LT(undershoot, 0.05f);
    }
}

// Defect: the linear-phase FIR length was the host's maxBlockSize, so its
// low-frequency resolution followed the host: band 0 of a 200 Hz LR24 split
// read -12.6 dB at 100 Hz with 64-sample blocks (-1.7 dB at 512) against the
// ideal -0.527 dB. The FIR now has a fixed ~85 ms length with a flat-top
// window: the ideal response at every block size, and one fixed latency.
DSPARK_TEST(CrossoverFilter_linear_phase_resolution_is_block_independent)
{
    const double fs = 48000.0;
    int latency = -1;
    for (int bs : { 64, 512, 4096 })
    {
        CrossoverFilter<float> x;
        x.setNumBands(2);
        x.setOrder(24);
        x.setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
        x.prepare(AudioSpec{ fs, bs, 1 });
        x.setCrossoverFrequency(0, 200.0f);
        const int n = 1 << 15, t0 = 8192;
        Vec in(static_cast<size_t>(n), 0.0f);
        in[static_cast<size_t>(t0)] = 1.0f;
        const auto out = crossoverSplit(x, in, 2, bs);
        if (latency < 0) latency = x.getLatency();
        EXPECT_EQ(x.getLatency(), latency);
        const int centre = t0 + x.getLatency();
        const double at100 = kernelResponseDb(out[0], centre, 4096, 100.0, fs);
        const double at400 = kernelResponseDb(out[0], centre, 4096, 400.0, fs);
        std::cout << "  maxBlock " << bs << ": band 0 " << at100 << " dB @100 Hz, " << at400 << " dB @400 Hz\n";
        EXPECT_NEAR(at100, -0.527, 0.02); // 64-sample blocks used to read -12.6
        EXPECT_NEAR(at400, -24.61, 0.05);
    }
    EXPECT_EQ(latency, 3072);
}

// Defect: band b carried only the high-pass of the split right below it, so
// close splits did not telescope to 1: 100/150/225 Hz (LR24) summed 0.87 dB
// off flat, 1.4e-3 away from the delayed input. Each band now carries the
// high-pass of every split below it, like the IIR tree.
DSPARK_TEST(CrossoverFilter_linear_phase_close_splits_sum_to_delayed_input)
{
    CrossoverFilter<float> x;
    x.setNumBands(4);
    x.setOrder(24);
    x.setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
    x.prepare(AudioSpec{ 48000.0, 512, 1 });
    x.setCrossoverFrequency(0, 100.0f);
    x.setCrossoverFrequency(1, 150.0f);
    x.setCrossoverFrequency(2, 225.0f);
    const int n = 1 << 15, t0 = 4096;
    Vec in(static_cast<size_t>(n), 0.0f);
    in[static_cast<size_t>(t0)] = 1.0f;
    const auto out = crossoverSplit(x, in, 4, 512);
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (int b = 0; b < 4; ++b) s += out[static_cast<size_t>(b)][static_cast<size_t>(i)];
        worst = std::max(worst, std::abs(s - ((i == t0 + x.getLatency()) ? 1.0 : 0.0)));
    }
    EXPECT_LT(worst, 1e-5);
}

// The partitioned linear-phase engine buffers internally: the same input in
// any block partition gives bit-identical bands.
DSPARK_TEST(CrossoverFilter_linear_phase_output_is_block_size_independent)
{
    const int n = 24000;
    uint32_t rng = 777u;
    Vec in(static_cast<size_t>(n));
    for (auto& v : in)
    {
        rng = rng * 1664525u + 1013904223u;
        v = 0.3f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
    }
    auto make = [] {
        auto x = std::make_unique<CrossoverFilter<float>>();
        x->setNumBands(3);
        x->setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
        x->prepare(AudioSpec{ 48000.0, 512, 1 });
        return x;
    };
    auto ref = make();
    const auto a = crossoverSplit(*ref, in, 3, 512);
    for (int bs : { 1, 7, 333 })
    {
        auto x = make();
        const auto b = crossoverSplit(*x, in, 3, bs, bs == 333);
        float maxDiff = 0.0f;
        for (int band = 0; band < 3; ++band)
            for (int i = 0; i < n; ++i)
                maxDiff = std::max(maxDiff, std::abs(a[static_cast<size_t>(band)][static_cast<size_t>(i)]
                                                     - b[static_cast<size_t>(band)][static_cast<size_t>(i)]));
        EXPECT_EQ(maxDiff, 0.0f);
    }
}

// Defect: the IIR glide refreshed its coefficients on a 32-sample grid that
// restarted at every block, so a crossover glide depended on the host block
// size (up to 0.14 between 1-sample and 512-sample blocks on a 1 kHz tone).
// The grid now runs across blocks: bit-identical at any block size.
DSPARK_TEST(CrossoverFilter_glide_is_block_size_independent)
{
    const int n = 12000;
    const Vec in = makeSine(1000.0, 0.5, 48000.0, n);
    auto make = [] {
        auto x = std::make_unique<CrossoverFilter<float>>();
        x->setNumBands(2);
        x->prepare(AudioSpec{ 48000.0, 512, 1 });
        x->setCrossoverFrequency(0, 500.0f);
        const Vec warm(4096, 0.0f);
        (void)crossoverSplit(*x, warm, 2, 512);
        x->setCrossoverFrequency(0, 4000.0f); // glides over the next blocks
        return x;
    };
    auto ref = make();
    const auto a = crossoverSplit(*ref, in, 2, 512);
    for (int bs : { 1, 100, 333 })
    {
        auto x = make();
        const auto b = crossoverSplit(*x, in, 2, bs, bs == 333);
        float maxDiff = 0.0f;
        for (int i = 0; i < n; ++i)
            maxDiff = std::max(maxDiff, std::abs(a[0][static_cast<size_t>(i)] - b[0][static_cast<size_t>(i)]));
        EXPECT_EQ(maxDiff, 0.0f);
    }
}

// The ladder structure: with the splits still, every band of every order is
// the direct form's; with a split jumping between 100 Hz and 10 kHz every
// block, the band sum stays as bounded as in direct form (the 5 ms glide of
// the split keeps the direct form's memory consistent, so the ladder does not
// lower that peak: 0.65 and 0.70 against 0.65 and 0.64 at LR24 and LR48).
DSPARK_TEST(CrossoverFilter_ladder_structure_matches_and_holds_under_jumps)
{
    using X = CrossoverFilter<float>;
    const int n = 24000;
    Vec in(static_cast<size_t>(n));
    uint32_t rng = 0x13572468u;
    for (auto& v : in)
    {
        rng = rng * 1664525u + 1013904223u;
        v = 0.3f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
    }
    for (int order : { 12, 24, 48 })
    {
        auto make = [order](X::Structure structure) {
            auto x = std::make_unique<X>();
            x->setNumBands(4);
            x->setOrder(order);
            x->setStructure(structure);
            x->prepare(AudioSpec{ 48000.0, 512, 1 });
            return x;
        };
        auto direct = make(X::Structure::DirectForm);
        auto ladder = make(X::Structure::Ladder);
        EXPECT_TRUE((ladder->getStructure() == X::Structure::Ladder));
        const auto a = crossoverSplit(*direct, in, 4, 333);
        const auto b = crossoverSplit(*ladder, in, 4, 333);
        float worst = 0.0f;
        for (int band = 0; band < 4; ++band)
            for (int i = 0; i < n; ++i)
                worst = std::max(worst, std::abs(a[static_cast<size_t>(band)][static_cast<size_t>(i)]
                                                 - b[static_cast<size_t>(band)][static_cast<size_t>(i)]));
        EXPECT_LT(worst, 1e-6f);
    }

    float peak[2] = { 0.0f, 0.0f };
    for (int k = 0; k < 2; ++k)
    {
        auto x = std::make_unique<X>();
        x->setNumBands(2);
        x->setOrder(48);
        x->setStructure(k == 0 ? X::Structure::DirectForm : X::Structure::Ladder);
        x->prepare(AudioSpec{ 48000.0, 64, 1 });
        Vec low(64), high(64);
        for (int off = 0, block = 0; off + 64 <= n; off += 64, ++block)
        {
            x->setCrossoverFrequency(0, (block % 2 == 0) ? 100.0f : 10000.0f);
            Vec work(in.begin() + off, in.begin() + off + 64);
            float* ip = work.data();
            float* lp = low.data();
            float* hp = high.data();
            AudioBufferView<float> iv(&ip, 1, 64);
            AudioBufferView<float> views[2] = { AudioBufferView<float>(&lp, 1, 64), AudioBufferView<float>(&hp, 1, 64) };
            x->processBlock(iv, views, 2);
            for (int i = 0; i < 64; ++i)
                peak[k] = std::max(peak[k], std::abs(low[static_cast<size_t>(i)] + high[static_cast<size_t>(i)]));
        }
    }
    EXPECT_TRUE(std::isfinite(peak[1]));
    EXPECT_LT(peak[1], 1.0f);
    EXPECT_LT(peak[1], 1.25f * peak[0]);
}

// Defect: in linear-phase mode a live band-count change left the bands it
// added without an output partition of their own, so the band sum missed or
// replayed content until the next partition: 0.155 of error against the
// delayed input (0.70 with the first fixed-length engine) over random
// band-count, frequency and order changes. Each change now lands as one
// complete design; until then the playing design's bands are kept, silent or
// folded into the last requested band, so the sum stays the delayed input.
DSPARK_TEST(CrossoverFilter_linear_phase_bands_sum_through_live_changes)
{
    const int n = 48000 * 6, maxBlock = 700;
    auto x = std::make_unique<CrossoverFilter<float>>();
    x->setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
    x->setNumBands(4);
    x->prepare(AudioSpec{ 48000.0, maxBlock, 2 });
    const int latency = x->getLatency();
    Vec in[2], sum[2];
    uint32_t s = 7u;
    for (int c = 0; c < 2; ++c)
    {
        in[c].resize(static_cast<size_t>(n));
        sum[c].assign(static_cast<size_t>(n), 0.0f);
        for (auto& v : in[c]) { s = s * 1664525u + 1013904223u; v = static_cast<float>(s >> 8) / 16777216.0f - 0.5f; }
    }
    std::vector<Vec> out(12, Vec(static_cast<size_t>(2 * maxBlock), 0.0f));
    uint32_t r = 99u;
    int changes = 0;
    for (int pos = 0; pos < n;)
    {
        r = r * 1103515245u + 12345u;
        const int m = std::min(n - pos, 1 + static_cast<int>((r >> 16) % static_cast<uint32_t>(maxBlock)));
        r = r * 1103515245u + 12345u;
        const int ev = static_cast<int>((r >> 16) % 40u);
        if (ev == 0) { x->setNumBands(2 + static_cast<int>((r >> 8) % 11u)); ++changes; }
        else if (ev == 1) { x->setCrossoverFrequency(0, 60.0f + static_cast<float>((r >> 4) % 400u)); ++changes; }
        else if (ev == 2) { x->setOrder((r & 1u) ? 48 : 12); ++changes; }
        else if (ev == 3) { x->setOrder(24); ++changes; }
        float* ich[2] = { in[0].data() + pos, in[1].data() + pos };
        float* och[12][2];
        std::vector<AudioBufferView<float>> views;
        for (int b = 0; b < 12; ++b)
        {
            och[b][0] = out[static_cast<size_t>(b)].data();
            och[b][1] = out[static_cast<size_t>(b)].data() + maxBlock;
            views.emplace_back(och[b], 2, m);
        }
        const int got = x->processBlock(AudioBufferView<float>(ich, 2, m), views.data(), 12);
        for (int c = 0; c < 2; ++c)
            for (int b = 0; b < got; ++b)
                for (int i = 0; i < m; ++i)
                    sum[c][static_cast<size_t>(pos + i)] += och[b][c][i];
        pos += m;
    }
    double err = 0.0;
    for (int c = 0; c < 2; ++c)
        for (int i = latency; i < n; ++i)
            err = std::max(err, static_cast<double>(std::abs(sum[c][static_cast<size_t>(i)]
                                                             - in[c][static_cast<size_t>(i - latency)])));
    std::cout << "  " << changes << " live changes: max |band sum - delayed input| " << err << "\n";
    EXPECT_GT(changes, 30);
    EXPECT_LT(err, 1e-5);
}

// A linear-phase change is designed over a few blocks into the kernel set
// that is not playing, then swapped in whole: no block pays for the whole
// design. Pinned: after a move the bands become bit-identical to those of a
// crossover prepared with the new frequency, and stay so, within 8192
// samples at small and large blocks (mono; about 3600 samples measured).
DSPARK_TEST(CrossoverFilter_linear_phase_change_lands_on_the_new_design)
{
    for (int block : { 64, 512 })
    {
        auto a = std::make_unique<CrossoverFilter<float>>();
        auto b = std::make_unique<CrossoverFilter<float>>();
        for (auto* x : { a.get(), b.get() })
        {
            x->setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
            x->setNumBands(4);
        }
        b->setCrossoverFrequency(0, 230.0f);
        for (auto* x : { a.get(), b.get() }) x->prepare(AudioSpec{ 48000.0, block, 1 });
        const int warm = 16384 / block, total = warm + 2 * 48000 / block;
        std::vector<Vec> oa(4, Vec(static_cast<size_t>(block))), ob(4, Vec(static_cast<size_t>(block)));
        Vec in(static_cast<size_t>(block));
        int landed = -1;
        uint32_t s = 3u;
        for (int k = 0; k < total; ++k)
        {
            if (k == warm) a->setCrossoverFrequency(0, 230.0f);
            for (auto& v : in) { s = s * 1664525u + 1013904223u; v = static_cast<float>(s >> 8) / 16777216.0f - 0.5f; }
            for (int t = 0; t < 2; ++t)
            {
                auto& o = t == 0 ? oa : ob;
                float* ip = in.data();
                float* p[4];
                std::vector<AudioBufferView<float>> views;
                for (int band = 0; band < 4; ++band)
                {
                    p[band] = o[static_cast<size_t>(band)].data();
                    views.emplace_back(&p[band], 1, block);
                }
                (t == 0 ? *a : *b).processBlock(AudioBufferView<float>(&ip, 1, block), views.data(), 4);
            }
            bool same = true;
            for (int band = 0; band < 4; ++band)
                same = same && std::memcmp(oa[static_cast<size_t>(band)].data(), ob[static_cast<size_t>(band)].data(),
                                           sizeof(float) * static_cast<size_t>(block)) == 0;
            if (k >= warm && same && landed < 0) landed = k - warm;
            if (k >= warm && !same) landed = -1;
        }
        std::cout << "  block " << block << ": new design heard after " << landed << " blocks\n";
        EXPECT_GT(landed, -1);
        EXPECT_LT(landed * block, 8193);
    }
}

// Defect: the linear-phase design budget followed the steady work, which
// scales with the channel count, so a change took about twice as long to be
// heard in mono (14 against 8 blocks of 512 samples, 149 against 85 ms). The
// design now runs at a fixed share of the stereo work per sample: the same
// landing in blocks for one, two and six channels, at small and large blocks.
DSPARK_TEST(CrossoverFilter_linear_phase_landing_is_channel_independent)
{
    auto landing = [](int block, int chans) {
        auto a = std::make_unique<CrossoverFilter<float>>();
        auto b = std::make_unique<CrossoverFilter<float>>();
        for (auto* x : { a.get(), b.get() })
        {
            x->setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
            x->setNumBands(4);
        }
        b->setCrossoverFrequency(0, 230.0f);
        for (auto* x : { a.get(), b.get() }) x->prepare(AudioSpec{ 48000.0, block, chans });
        const int warm = 16384 / block, total = warm + 2 * 48000 / block;
        const size_t len = static_cast<size_t>(block) * static_cast<size_t>(chans);
        std::vector<Vec> oa(4, Vec(len)), ob(4, Vec(len));
        Vec in(len);
        int landed = -1;
        uint32_t s = 3u;
        for (int k = 0; k < total; ++k)
        {
            if (k == warm) a->setCrossoverFrequency(0, 230.0f);
            for (auto& v : in) { s = s * 1664525u + 1013904223u; v = static_cast<float>(s >> 8) / 16777216.0f - 0.5f; }
            std::vector<float*> ip(static_cast<size_t>(chans));
            for (int c = 0; c < chans; ++c) ip[static_cast<size_t>(c)] = in.data() + c * block;
            for (int t = 0; t < 2; ++t)
            {
                auto& o = t == 0 ? oa : ob;
                std::vector<std::vector<float*>> p(4, std::vector<float*>(static_cast<size_t>(chans)));
                std::vector<AudioBufferView<float>> views;
                for (int band = 0; band < 4; ++band)
                {
                    for (int c = 0; c < chans; ++c)
                        p[static_cast<size_t>(band)][static_cast<size_t>(c)] = o[static_cast<size_t>(band)].data() + c * block;
                    views.emplace_back(p[static_cast<size_t>(band)].data(), chans, block);
                }
                (t == 0 ? *a : *b).processBlock(AudioBufferView<float>(ip.data(), chans, block), views.data(), 4);
            }
            bool same = true;
            for (int band = 0; band < 4; ++band)
                same = same && std::memcmp(oa[static_cast<size_t>(band)].data(), ob[static_cast<size_t>(band)].data(),
                                           sizeof(float) * len) == 0;
            if (k >= warm && same && landed < 0) landed = k - warm;
            if (k >= warm && !same) landed = -1;
        }
        return landed;
    };
    for (int block : { 64, 512 })
    {
        const int mono = landing(block, 1), stereo = landing(block, 2), six = landing(block, 6);
        std::cout << "  block " << block << ": heard after " << mono << " / " << stereo << " / " << six
                  << " blocks (1 / 2 / 6 channels)\n";
        EXPECT_GT(mono, -1);
        EXPECT_EQ(mono, stereo); // 14 against 8 at 512-sample blocks before
        EXPECT_EQ(six, stereo);
        EXPECT_LT(mono * block, 8193);
    }
}

// The linear-phase bands against the ideal Linkwitz-Riley magnitudes
// |LP| = 1/(1 + (f/fc)^N), each band carrying the high-pass of every split
// below it: 12 bands at the default splits (LR24) and 3 bands at 50 Hz and
// 5 kHz (LR12), 120 frequencies from 20 Hz to 20 kHz. Pins the partition plan,
// the residual last band and the dropped negligible partitions: the
// worst error is 7.2e-7 (2.0e-7 with every partition convolved) and 7.9e-5
// (the 50 Hz LR12 skirt the FIR length allows).
DSPARK_TEST(CrossoverFilter_linear_phase_bands_match_the_ideal_response)
{
    struct Cfg { int bands; int order; float f0, f1; double tol; };
    for (const Cfg c : { Cfg{ 12, 24, 0.0f, 0.0f, 2e-6 }, Cfg{ 3, 12, 50.0f, 5000.0f, 1.5e-4 } })
    {
        CrossoverFilter<float> x;
        x.setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
        x.setNumBands(c.bands);
        x.setOrder(c.order);
        if (c.f0 > 0.0f) { x.setCrossoverFrequency(0, c.f0); x.setCrossoverFrequency(1, c.f1); }
        x.prepare(AudioSpec{ 48000.0, 512, 1 });
        const int n = 1 << 15, t0 = 4096, half = 2048;
        Vec in(static_cast<size_t>(n), 0.0f);
        in[static_cast<size_t>(t0)] = 1.0f;
        const auto out = crossoverSplit(x, in, c.bands, 512);
        const int centre = t0 + x.getLatency();
        const int expo = c.order / 6;
        double worst = 0.0;
        for (int fi = 0; fi < 120; ++fi)
        {
            const double f = 20.0 * std::pow(1000.0, fi / 119.0);
            double carry = 1.0;
            for (int b = 0; b < c.bands; ++b)
            {
                double ideal = carry;
                if (b < c.bands - 1)
                {
                    const double r = std::pow(f / static_cast<double>(x.getCrossoverFrequency(b)), expo);
                    ideal = carry / (1.0 + r);
                    carry *= r / (1.0 + r);
                }
                const double mag = std::pow(10.0, kernelResponseDb(out[static_cast<size_t>(b)], centre, half, f, 48000.0) / 20.0);
                worst = std::max(worst, std::abs(mag - ideal));
            }
        }
        std::cout << "  " << c.bands << " bands LR" << c.order << ": worst |H - ideal| " << worst << "\n";
        EXPECT_LT(worst, c.tol);
    }
}

// New: delta listen. The output of the delta twin plus the output of the
// plain one is the input, late by the latency: the compressor (no makeup),
// the de-esser and the dynamic equalizer.
namespace {

std::vector<float> l13Burst(size_t n)
{
    std::vector<float> v(n);
    uint32_t seed = 21u;
    for (size_t i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        const double env = (i / 4800) % 2 == 0 ? 0.5 : 0.05;
        v[i] = static_cast<float>(env * (0.7 * std::sin(2.0 * 3.14159265358979323846 * 900.0 * static_cast<double>(i) / 48000.0)
                                         + 0.3 * (static_cast<double>(seed >> 8) / 8388608.0 - 1.0)));
    }
    return v;
}

template <typename P>
std::vector<float> l13Through(P& p, std::vector<float> x, int block = 128)
{
    std::vector<float> r = x;
    for (size_t off = 0; off < x.size(); off += static_cast<size_t>(block))
    {
        const int n = static_cast<int>(std::min<size_t>(static_cast<size_t>(block), x.size() - off));
        float* ch[2] = { x.data() + off, r.data() + off };
        p.processBlock(AudioBufferView<float>(ch, 2, n));
    }
    return x;
}

} // namespace

DSPARK_TEST(Delta_listen_is_the_complement_of_the_output)
{
    const std::vector<float> in = l13Burst(48000);
    const auto check = [&](const std::vector<float>& a, const std::vector<float>& b, int lat) {
        double worst = 0.0;
        for (size_t i = static_cast<size_t>(lat); i < in.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(a[i] + b[i] - in[i - static_cast<size_t>(lat)])));
        return worst;
    };
    {
        Compressor<float> a, b;
        for (auto* c : { &a, &b })
        {
            c->setThreshold(-24.0f); c->setRatio(4.0f); c->setAttack(5.0f); c->setRelease(80.0f);
            c->prepare(spec(48000.0, 128, 2));
        }
        b.setRemovalListen(true);
        EXPECT_TRUE(b.isRemovalListen());
        const auto plain = l13Through(a, in);
        const auto delta = l13Through(b, in);
        EXPECT_LT(check(plain, delta, a.getLatency()), 1e-5);
        double taken = 0.0;
        for (float x : delta) taken = std::max(taken, std::abs(static_cast<double>(x)));
        EXPECT_GT(taken, 0.05);
    }
    {
        DeEsser<float> a, b;
        for (auto* d : { &a, &b }) { d->setThreshold(-40.0f); d->prepare(spec(48000.0, 128, 2)); }
        b.setRemovalListen(true);
        const auto plain = l13Through(a, in);
        const auto delta = l13Through(b, in);
        EXPECT_LT(check(plain, delta, 0), 1e-6);
    }
    {
        auto a = std::make_unique<DynamicEQ<float>>();
        auto b = std::make_unique<DynamicEQ<float>>();
        for (auto* d : { a.get(), b.get() })
        {
            DynamicEQ<float>::BandConfig band;
            band.enabled = true; band.frequency = 900.0f; band.threshold = -30.0f; band.aboveRatio = 4.0f; band.aboveRangeDb = 12.0f;
            d->setBand(0, band);
            d->prepare(spec(48000.0, 128, 2));
        }
        b->setRemovalListen(true);
        const auto plain = l13Through(*a, in);
        const auto delta = l13Through(*b, in);
        EXPECT_LT(check(plain, delta, a->getLatency()), 1e-5);
    }
}

// ============================================================================
// The dominant link of a stereo pair (Compressor, DynamicEQ)
// ============================================================================

namespace
{
// A 1 kHz tone with a gain per channel, through a processor for a second,
// in blocks of 256; returns what read() gives at the end.
template <class P, class Read>
double pairReading(P& p, float gainL, float gainR, Read read)
{
    constexpr int kBlock = 256;
    std::vector<float> l(kBlock), r(kBlock);
    long n = 0;
    for (int b = 0; b < 48000 / kBlock; ++b)
    {
        for (int i = 0; i < kBlock; ++i, ++n)
        {
            const float x = 0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979323846 * 1000.0 * static_cast<double>(n) / 48000.0));
            l[static_cast<size_t>(i)] = gainL * x;
            r[static_cast<size_t>(i)] = gainR * x;
        }
        float* ch[2] = { l.data(), r.data() };
        p.processBlock(AudioBufferView<float>(ch, 2, kBlock));
    }
    return read(p);
}
} // namespace

DSPARK_TEST(Compressor_dominant_link_reads_a_source_wherever_it_is_panned)
{
    // An equal-power pan: centred, the tone is 0.707 of itself in each
    // channel; panned hard, all of it in one. The loudest link reads the hard
    // pan 3 dB hotter (2.25 dB more reduction at 4:1); the dominant link reads
    // the same source the same anywhere. Both paths: passes and per sample.
    constexpr float kCentre = 0.70710678f;
    using C = Compressor<float>;
    for (const auto precision : { C::Precision::Standard, C::Precision::High })
    {
        double spread[2] = {};
        for (const auto link : { C::LinkMode::Loudest, C::LinkMode::Dominant })
        {
            double gr[2] = {};
            for (int pan = 0; pan < 2; ++pan)
            {
                auto c = std::make_unique<C>();
                c->setThreshold(-30.0f); c->setRatio(4.0f); c->setKnee(0.0f);
                c->setAttack(2.0f); c->setRelease(60.0f);
                c->setDetector(C::DetectorType::Rms);
                c->setPrecision(precision);
                c->setLinkMode(link);
                c->prepare(spec(48000.0, 256, 2));
                gr[pan] = pairReading(*c, pan == 0 ? kCentre : 1.0f, pan == 0 ? kCentre : 0.0f,
                                      [](C& x) { return static_cast<double>(x.getGainReductionDb()); });
            }
            EXPECT_LT(gr[0], -5.0);   // it compresses
            spread[link == C::LinkMode::Dominant ? 1 : 0] = std::abs(gr[0] - gr[1]);
        }
        EXPECT_GT(spread[0], 1.5);    // loudest: the hard pan reads hotter
        EXPECT_LT(spread[1], 0.3);    // dominant: the same anywhere
    }
}

DSPARK_TEST(DynamicEQ_dominant_link_reads_a_source_wherever_it_is_panned)
{
    constexpr float kCentre = 0.70710678f;
    using D = DynamicEQ<float>;
    double spread[2] = {};
    for (const auto link : { D::LinkMode::Loudest, D::LinkMode::Dominant })
    {
        double gain[2] = {};
        for (int pan = 0; pan < 2; ++pan)
        {
            auto d = std::make_unique<D>();
            D::BandConfig band;
            band.frequency = 1000.0f; band.q = 1.0f; band.threshold = -30.0f;
            band.aboveRatio = 4.0f; band.aboveRangeDb = 24.0f;
            band.aboveAttackMs = 5.0f; band.aboveReleaseMs = 60.0f;
            d->setBand(0, band);
            d->setLinkMode(link);
            d->prepare(spec(48000.0, 256, 2));
            gain[pan] = pairReading(*d, pan == 0 ? kCentre : 1.0f, pan == 0 ? kCentre : 0.0f,
                                    [](D& x) { return static_cast<double>(x.getBandGainDb(0)); });
        }
        EXPECT_LT(gain[0], -5.0);
        spread[link == D::LinkMode::Dominant ? 1 : 0] = std::abs(gain[0] - gain[1]);
    }
    EXPECT_GT(spread[0], 1.5);
    EXPECT_LT(spread[1], 0.3);
}

DSPARK_TEST(DynamicEQ_and_DeEsser_ladder_structure_matches_the_direct_form)
{
    // The dynamic filter is designed again every 16 samples as the gain
    // moves; as a ladder it has the same response, so the output stays with
    // the direct form's (closely while the gain swings, the two memories
    // taking the jumps differently).
    std::vector<float> burst(48000);
    for (size_t i = 0; i < burst.size(); ++i)
    {
        const double t = static_cast<double>(i) / 48000.0;
        const double env = (i / 4800) % 2 == 0 ? 0.6 : 0.05;
        burst[i] = static_cast<float>(env * (std::sin(2.0 * 3.14159265358979323846 * 2200.0 * t)
                                             + 0.5 * std::sin(2.0 * 3.14159265358979323846 * 6500.0 * t)));
    }
    const auto run = [&](auto& p) {
        std::vector<float> l = burst, r = burst;
        for (size_t off = 0; off < burst.size(); off += 256)
        {
            const int n = static_cast<int>(std::min<size_t>(256, burst.size() - off));
            float* ch[2] = { l.data() + off, r.data() + off };
            p.processBlock(AudioBufferView<float>(ch, 2, n));
        }
        return l;
    };
    {
        using D = DynamicEQ<float>;
        std::vector<float> out[2];
        for (int ladder = 0; ladder < 2; ++ladder)
        {
            auto d = std::make_unique<D>();
            D::BandConfig band;
            band.frequency = 2200.0f; band.threshold = -24.0f; band.aboveRatio = 4.0f; band.aboveRangeDb = 18.0f;
            d->setBand(0, band);
            if (ladder) d->setStructure(D::Structure::Ladder);
            d->prepare(spec(48000.0, 256, 2));
            EXPECT_TRUE(d->getStructure() == (ladder ? D::Structure::Ladder : D::Structure::DirectForm));
            out[ladder] = run(*d);
        }
        // Where a burst starts the bell jumps in from the identity: the direct
        // form starts it from an empty memory, the ladder from the history it
        // kept, so the two part there for a few milliseconds. Compared over
        // the settled part of each burst, and in energy over the whole.
        double worst = 0.0, peak = 0.0, e0 = 0.0, e1 = 0.0;
        for (size_t i = 0; i < out[0].size(); ++i)
        {
            e0 += static_cast<double>(out[0][i]) * out[0][i];
            e1 += static_cast<double>(out[1][i]) * out[1][i];
            peak = std::max(peak, std::abs(static_cast<double>(out[0][i])));
            if (i % 4800 >= 2400)
                worst = std::max(worst, std::abs(static_cast<double>(out[0][i]) - static_cast<double>(out[1][i])));
        }
        const double energyDb = 10.0 * std::log10(e1 / e0);
        std::cout << "  dynamic EQ: settled difference " << worst / peak << " of the peak, energy "
                  << energyDb << " dB\n";
        EXPECT_LT(worst / peak, 2e-2);
        EXPECT_LT(std::abs(energyDb), 0.2);
    }
    {
        using S = DeEsser<float>;
        std::vector<float> out[2];
        for (int ladder = 0; ladder < 2; ++ladder)
        {
            auto s = std::make_unique<S>();
            s->setThreshold(-30.0f);
            if (ladder) s->setStructure(S::Structure::Ladder);
            s->prepare(spec(48000.0, 256, 2));
            EXPECT_TRUE(s->getStructure() == (ladder ? S::Structure::Ladder : S::Structure::DirectForm));
            out[ladder] = run(*s);
        }
        // The bell jumps as the reduction attacks at each burst: compared
        // over the settled part of each burst, and in energy over the whole.
        double worst = 0.0, peak = 0.0, e0 = 0.0, e1 = 0.0;
        for (size_t i = 0; i < out[0].size(); ++i)
        {
            e0 += static_cast<double>(out[0][i]) * out[0][i];
            e1 += static_cast<double>(out[1][i]) * out[1][i];
            peak = std::max(peak, std::abs(static_cast<double>(out[0][i])));
            if (i % 4800 >= 2400)
                worst = std::max(worst, std::abs(static_cast<double>(out[0][i]) - static_cast<double>(out[1][i])));
        }
        const double energyDb = 10.0 * std::log10(e1 / e0);
        std::cout << "  de-esser: settled difference " << worst / peak << " of the peak, energy "
                  << energyDb << " dB\n";
        EXPECT_LT(worst / peak, 2e-2);
        EXPECT_LT(std::abs(energyDb), 0.2);
    }
}

// ============================================================================
// CharacterStage (Core/CharacterStage.h) and the colour stage of the Compressor
// ============================================================================

namespace {

// One tone through a fresh stage with these settings, in blocks of 512.
Vec stageRun(const CharacterStage<float>::Settings& s, double hz, double amp, int n, double fs = 48000.0)
{
    auto st = std::make_unique<CharacterStage<float>>();
    st->setSettings(s);
    st->prepare(AudioSpec{ fs, 512, 1 });
    std::vector<Vec> x { makeSine(hz, amp, fs, n) };
    processInBlocks(x, 512, [&](AudioBufferView<float> v) { st->process(v); });
    return x[0];
}

// Level of harmonic h of a tone at hz over the last 9600 samples, against the tone, in dB.
double stageHarmonicDb(const Vec& y, double hz, int h, double fs = 48000.0)
{
    const int n = static_cast<int>(y.size());
    const float* tail = y.data() + n - 9600;
    return toDb(toneAmplitude(tail, 9600, hz * h, fs) / toneAmplitude(tail, 9600, hz, fs));
}

} // namespace

// New: at a low level the stage has unity gain at every frequency: the
// saturators are divided by their small-signal gain and the de-emphasis
// undoes the emphasis.
DSPARK_TEST(CharacterStage_is_transparent_at_a_low_level)
{
    const CharacterStage<float>::Settings s {};
    for (double hz : { 100.0, 1000.0, 10000.0, 18000.0 })
    {
        const Vec y = stageRun(s, hz, 0.001, 48000);
        const double g = toDb(toneAmplitude(y.data() + 48000 - 9600, 9600, hz, 48000.0) / 0.001);
        EXPECT_NEAR(g, 0.0, 0.05);
    }
}

// New: the colour grows with the drive; the asymmetry is what brings the
// even harmonics; the emphasis makes the highs saturate first.
DSPARK_TEST(CharacterStage_colour_follows_drive_asymmetry_and_emphasis)
{
    using S = CharacterStage<float>::Settings;
    double h3[3];
    for (int k = 0; k < 3; ++k)
    {
        S s; s.driveDb = static_cast<float>(12 * k);
        h3[k] = stageHarmonicDb(stageRun(s, 1000.0, 0.5, 48000), 1000.0, 3);
    }
    std::cout << "  [stage] H3 at drive 0/12/24: " << h3[0] << " " << h3[1] << " " << h3[2] << " dBc\n";
    EXPECT_LT(h3[0], h3[1] - 6.0);
    EXPECT_LT(h3[1], h3[2] - 1.0);   // near its ceiling the curve grows slower

    S even; even.driveDb = 12.0f; even.asymmetry = 0.6f;
    S odd = even; odd.asymmetry = 0.0f;
    const double h2even = stageHarmonicDb(stageRun(even, 1000.0, 0.5, 48000), 1000.0, 2);
    const double h2odd = stageHarmonicDb(stageRun(odd, 1000.0, 0.5, 48000), 1000.0, 2);
    std::cout << "  [stage] H2 with asymmetry 0.6 / 0: " << h2even << " / " << h2odd << " dBc\n";
    EXPECT_GT(h2even, -60.0);
    EXPECT_LT(h2odd, -100.0);

    S flat; flat.driveDb = 12.0f; flat.emphasisDb = 0.0f;
    S bright = flat; bright.emphasisDb = 12.0f;
    const double highFlat = stageHarmonicDb(stageRun(flat, 6000.0, 0.25, 48000), 6000.0, 3);
    const double highBright = stageHarmonicDb(stageRun(bright, 6000.0, 0.25, 48000), 6000.0, 3);
    const double lowFlat = stageHarmonicDb(stageRun(flat, 200.0, 0.25, 48000), 200.0, 3);
    const double lowBright = stageHarmonicDb(stageRun(bright, 200.0, 0.25, 48000), 200.0, 3);
    std::cout << "  [stage] H3 of 6 kHz with emphasis 0 / 12: " << highFlat << " / " << highBright
              << " dBc; of 200 Hz: " << lowFlat << " / " << lowBright << " dBc\n";
    EXPECT_GT(highBright, highFlat + 2.0);   // measured +3.8 dB
    EXPECT_NEAR(lowBright, lowFlat, 2.0);
}

// New: the noise follows the signal: silence stays silent, and under a tone
// the noise sits at its setting from the tone's envelope (its mean
// magnitude, 2 / pi of the peak).
DSPARK_TEST(CharacterStage_noise_follows_the_signal)
{
    using S = CharacterStage<float>::Settings;
    S quiet; quiet.noiseDb = -140.0f;
    S noisy; noisy.noiseDb = -60.0f;
    const Vec silent = stageRun(noisy, 1000.0, 0.0, 24000);
    double peak = 0.0;
    for (float v : silent) peak = std::max(peak, std::abs(static_cast<double>(v)));
    EXPECT_EQ(peak, 0.0);

    const Vec a = stageRun(quiet, 1000.0, 0.1, 48000);
    const Vec b = stageRun(noisy, 1000.0, 0.1, 48000);
    Vec residual(a.size());
    for (size_t i = 0; i < a.size(); ++i) residual[i] = b[i] - a[i];
    const double noise = toDb(rmsOf(residual.data() + 24000, 24000) / (0.1 / std::sqrt(2.0)));
    const double expected = -60.0 + toDb(2.0 / kPiD * std::sqrt(2.0));
    std::cout << "  [stage] noise under a tone " << noise << " dB (expected " << expected << ")\n";
    EXPECT_NEAR(noise, expected, 1.5);
}

// New: the internal rate is the host's times 4, 2 or 1 (about 176 kHz or
// more), and the delay reported is where an impulse comes out.
DSPARK_TEST(CharacterStage_rate_latency_and_settings)
{
    for (double fs : { 48000.0, 96000.0, 192000.0 })
    {
        auto st = std::make_unique<CharacterStage<float>>();
        st->prepare(AudioSpec{ fs, 512, 1 });
        EXPECT_EQ(st->getFactor(), fs < 90000.0 ? 4 : fs < 150000.0 ? 2 : 1);
        std::vector<Vec> x { Vec(2048, 0.0f) };
        x[0][100] = 0.001f;
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { st->process(v); });
        int at = 0;
        for (int i = 0; i < 2048; ++i)
            if (std::abs(x[0][static_cast<size_t>(i)]) > std::abs(x[0][static_cast<size_t>(at)])) at = i;
        EXPECT_NEAR(at - 100, st->getLatency(), 1);
    }
    CharacterStage<float> st;
    CharacterStage<float>::Settings s;
    s.driveDb = 99.0f; s.emphasisDb = std::numeric_limits<float>::quiet_NaN(); s.noiseDb = -500.0f;
    st.setSettings(s);
    const auto back = st.settings();
    EXPECT_EQ(back.driveDb, 24.0f);
    EXPECT_EQ(back.emphasisDb, 6.0f);
    EXPECT_EQ(back.noiseDb, -140.0f);
}

// New: drift is reproducible and moves the colour a little; sag adds colour
// as the level rises.
DSPARK_TEST(CharacterStage_drift_and_sag)
{
    using S = CharacterStage<float>::Settings;
    S drifting; drifting.driveDb = 12.0f; drifting.drift = 1.0f;
    const Vec a = stageRun(drifting, 1000.0, 0.5, 96000), b = stageRun(drifting, 1000.0, 0.5, 96000);
    EXPECT_TRUE(std::memcmp(a.data(), b.data(), sizeof(float) * a.size()) == 0);
    S still = drifting; still.drift = 0.0f;
    const Vec c = stageRun(still, 1000.0, 0.5, 96000);
    double moved = 0.0;
    for (size_t i = 48000; i < a.size(); ++i) moved = std::max(moved, std::abs(static_cast<double>(a[i] - c[i])));
    std::cout << "  [stage] drift moves the output by up to " << moved << "\n";
    EXPECT_GT(moved, 1e-4);
    EXPECT_LT(moved, 0.1);

    S light; light.driveDb = 0.0f;
    S sagging = light; sagging.sag = 1.0f;
    const double h3 = stageHarmonicDb(stageRun(light, 1000.0, 0.5, 48000), 1000.0, 3);
    const double h3sag = stageHarmonicDb(stageRun(sagging, 1000.0, 0.5, 48000), 1000.0, 3);
    std::cout << "  [stage] H3 without and with sag: " << h3 << " / " << h3sag << " dBc\n";
    EXPECT_GT(h3sag, h3 + 2.0);
}

// New: the colour stage of the Compressor. Under the threshold the
// compressor passes the tone; the stage in place delays it by the latency it
// reports and colours a loud one; off, there is neither.
DSPARK_TEST(Compressor_character_stage_colours_and_reports_its_delay)
{
    using C = Compressor<float>;
    const auto run = [](C::StagePosition position, double amp, int& latency) {
        auto c = std::make_unique<C>();
        c->setThreshold(0.0f);
        c->setCharacterStage(position);
        C::StageSettings s; s.driveDb = 12.0f;
        c->setCharacterStageSettings(s);
        c->prepare(AudioSpec{ 48000.0, 512, 1 });
        latency = c->getLatency();
        std::vector<Vec> x { makeSine(1000.0, amp, 48000.0, 48000) };
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { c->processBlock(v); });
        return x[0];
    };
    int off = -1, before = -1, after = -1;
    const Vec plain = run(C::StagePosition::Off, 0.5, off);
    const Vec coloured = run(C::StagePosition::After, 0.5, after);
    const Vec quietIn = run(C::StagePosition::Before, 0.001, before);
    EXPECT_EQ(off, 0);
    EXPECT_GT(after, 0);
    EXPECT_EQ(before, after);
    EXPECT_NEAR(toDb(toneAmplitude(quietIn.data() + 38400, 9600, 1000.0, 48000.0) / 0.001), 0.0, 0.05);
    const double h3plain = stageHarmonicDb(plain, 1000.0, 3), h3coloured = stageHarmonicDb(coloured, 1000.0, 3);
    std::cout << "  [compressor stage] latency " << after << ", H3 off " << h3plain << " / after " << h3coloured << " dBc\n";
    EXPECT_LT(h3plain, -100.0);
    EXPECT_GT(h3coloured, -60.0);
}

// New: a change of place of the colour stage fades over 20 ms: no step in
// the output larger than the tone's own.
DSPARK_TEST(Compressor_character_stage_moves_without_a_click)
{
    using C = Compressor<float>;
    auto c = std::make_unique<C>();
    c->setThreshold(0.0f);
    c->prepare(AudioSpec{ 48000.0, 256, 1 });
    const int n = 48000;
    std::vector<Vec> x { makeSine(1000.0, 0.25, 48000.0, n) };
    int block = 0;
    processInBlocks(x, 256, [&](AudioBufferView<float> v) {
        if (block == 40) c->setCharacterStage(C::StagePosition::Before);
        if (block == 80) c->setCharacterStage(C::StagePosition::After);
        if (block == 120) c->setCharacterStage(C::StagePosition::Off);
        ++block;
        c->processBlock(v);
    });
    const double natural = 0.25 * 2.0 * kPiD * 1000.0 / 48000.0;
    double worst = 0.0;
    for (int i = 1; i < n; ++i)
        worst = std::max(worst, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)] - x[0][static_cast<size_t>(i - 1)])));
    std::cout << "  [compressor stage] largest step " << worst << " (tone " << natural << ")\n";
    EXPECT_LT(worst, natural * 1.2);
    EXPECT_TRUE(c->getCharacterStage() == C::StagePosition::Off);
}

// New: the threshold in use. Without the automatic threshold it is the one
// set; with it, the key's recent peak plus the one set: a tone peaking at
// -10 dBFS and a threshold of -20 dB put it at -30 dBFS, in feed-forward and
// in feedback (where the reference follows the input).
DSPARK_TEST(Compressor_threshold_in_use_follows_the_automatic_reference)
{
    using C = Compressor<float>;
    for (int topology = 0; topology < 2; ++topology)
        for (int autoThr = 0; autoThr < 2; ++autoThr)
        {
            auto c = std::make_unique<C>();
            c->setThreshold(-20.0f);
            c->setTopology(topology ? C::Topology::FeedBack : C::Topology::FeedForward);
            c->setAutoThreshold(autoThr != 0);
            c->prepare(AudioSpec{ 48000.0, 256, 2 });
            const int n = 48000;
            std::vector<Vec> x { makeSine(1000.0, 0.316227766, 48000.0, n), makeSine(1000.0, 0.316227766, 48000.0, n) };
            processInBlocks(x, 256, [&](AudioBufferView<float> v) { c->processBlock(v); });
            const double inUse = static_cast<double>(c->getThresholdInUseDb());
            EXPECT_NEAR(inUse, autoThr ? -30.0 : -20.0, 0.25);
        }
}

// New: Precision::High reads the signal at eight points a sample and runs the
// gain computer's window, hold, release and ramp at eight steps a sample. The
// corners of the gain fall where the crests are, not on the sample grid, and
// the lines that are no harmonics of a steady tone fall by tens of dB; the
// level of the tone stays where it was.
DSPARK_TEST(Limiter_high_precision_lowers_the_lines_that_are_no_harmonics)
{
    using Lim = Limiter<double>;
    struct Row { double standard, high, gainStandard, gainHigh; };
    std::vector<Row> rows;
    for (int bin : { 68, 683, 3413 })                         // 100 Hz, 1 and 5 kHz
        for (double release : { 20.0, 100.0 })
            for (const bool hold : { true, false })           // the default 10 ms hold, and none
            {
                const auto at = [&](Lim::Precision precision) {
                    return toneLines<Lim>(bin, 1.0, [&](Lim& l) {
                        l.setCeiling(-6.0);
                        l.setRelease(release);
                        if (!hold) l.setHold(0.0);
                        l.setPrecision(precision);
                    });
                };
                const CompLines standard = at(Lim::Precision::Standard);
                const CompLines high = at(Lim::Precision::High);
                std::cout << "  limiter, " << 48000.0 * bin / 32768.0 << " Hz, release " << release << " ms, "
                          << (hold ? "hold 10 ms" : "no hold") << ": " << standard.worstDb << " -> "
                          << high.worstDb << " dBc, gain " << standard.gainDb << " -> " << high.gainDb << " dB\n";
                rows.push_back({ standard.worstDb, high.worstDb, standard.gainDb, high.gainDb });
            }
    for (const Row& r : rows)
    {
        EXPECT_LT(r.high, r.standard - 20.0);
        EXPECT_LT(r.high, -100.0);
        EXPECT_NEAR(r.gainHigh, r.gainStandard, 0.1);
    }
}

// The points concern the sample seven frames back: the latency is seven
// samples longer, the input comes out that late below the ceiling, every
// sample leaves at or below the ceiling, the per-sample path is the block
// path to the bit for mono, a switch during playback stays under the
// ceiling, and the state carries the choice. With true peak on the precision
// does not apply.
DSPARK_TEST(Limiter_high_precision_latency_ceiling_paths_and_state)
{
    using Lim = Limiter<double>;
    {
        Lim lim;
        lim.prepare(48000.0, 2);
        EXPECT_TRUE(lim.getPrecision() == Lim::Precision::Standard);
        EXPECT_EQ(lim.getLatency(), 96);
        lim.setPrecision(Lim::Precision::High);
        EXPECT_EQ(lim.getLatency(), 96 + 7);
        lim.setTruePeak(true);
        EXPECT_EQ(lim.getLatency(), 96);
        lim.setTruePeak(false);
        lim.prepare(44100.0, 2);                         // survives a re-preparation
        EXPECT_TRUE(lim.getPrecision() == Lim::Precision::High);
        EXPECT_EQ(lim.getLatency(), 88 + 7);
    }

    // Below the ceiling: the input, getLatency() samples late.
    {
        Lim lim;
        lim.setCeiling(-1.0);
        lim.setPrecision(Lim::Precision::High);
        lim.prepare(48000.0, 1);
        const size_t latency = static_cast<size_t>(lim.getLatency());
        std::vector<double> x(8000);
        LimiterTestNoise noise { 0x51u };
        for (double& v : x) v = 0.25 * noise.next();
        std::vector<double> y = x;
        for (size_t o = 0; o < y.size(); o += 300)
        {
            double* ch[1] = { y.data() + o };
            lim.processBlock(AudioBufferView<double>(ch, 1, static_cast<int>(std::min<size_t>(300, y.size() - o))));
        }
        double worst = 0.0;
        for (size_t i = 0; i < y.size(); ++i)
            worst = std::max(worst, std::abs(y[i] - (i >= latency ? x[i - latency] : 0.0)));
        EXPECT_LT(worst, 1e-15);
    }

    // The sample-domain contract at the shortest lookahead, with isolated
    // crests over a tone.
    {
        std::vector<double> x(6000);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 0.5 * std::sin(2.0 * 3.14159265358979 * 1000.0 * static_cast<double>(i) / 48000.0);
        x[1000] = 2.0;
        x[1090] = 1.6;
        x[1400] = -1.9;
        Lim lim;
        lim.setCeiling(-1.0);
        lim.setPrecision(Lim::Precision::High);
        lim.prepare(48000.0, 1);
        lim.setLookahead(0.5);
        const double ceil = decibelsToGain(-1.0);
        std::vector<double> y = x;
        for (size_t i = 0; i < y.size(); ++i)
        {
            double* p[1] = { &y[i] };
            lim.processBlock(AudioBufferView<double>(p, 1, 1));
        }
        int hits = 0;
        for (const double v : y) hits += std::abs(v) > ceil * (1.0 + 1e-12) ? 1 : 0;
        EXPECT_EQ(hits, 0);
    }

    // The per-sample path is the block path to the bit, with every option on.
    for (const bool everyPeak : { false, true })
    {
        Limiter<float> lb, ls;
        for (Limiter<float>* l : { &lb, &ls })
        {
            l->setCeiling(-6.0f);
            if (everyPeak) l->setHoldMode(Limiter<float>::HoldMode::EveryPeak);
            l->setPrecision(Limiter<float>::Precision::High);
            l->prepare(48000.0, 1);
            l->setAdaptiveRelease(true);
            l->setSafetyClip(true);
        }
        const int n = 6000;
        std::vector<float> in(static_cast<size_t>(n));
        LimiterTestNoise noise { 0x77u };
        for (int i = 0; i < n; ++i)
            in[static_cast<size_t>(i)] = static_cast<float>(0.05 * std::sin(6.283185307179586 * 330.0 * i / 48000.0)
                                                            + (i % 997 == 500 ? 1.0 : 0.0) + 0.4 * noise.next());
        std::vector<float> blockOut = in;
        const int sizes[] = { 512, 37, 300, 1, 129, 700 };
        for (int off = 0, k = 0; off < n; ++k)
        {
            const int len = std::min(sizes[k % 6], n - off);
            float* p[1] = { blockOut.data() + off };
            lb.processBlock(AudioBufferView<float>(p, 1, len));
            off += len;
        }
        float maxDiff = 0.0f;
        for (int i = 0; i < n; ++i)
            maxDiff = std::max(maxDiff, std::abs(ls.processSample(in[static_cast<size_t>(i)], 0) - blockOut[static_cast<size_t>(i)]));
        EXPECT_TRUE(maxDiff == 0.0f);
        const float ceiling = decibelsToGain(-6.0f);
        for (int i = 0; i < n; ++i)
            EXPECT_TRUE(std::abs(blockOut[static_cast<size_t>(i)]) <= ceiling + 1e-6f);
    }

    // Switched on and off during playback over noise twice the ceiling.
    {
        const double ceiling = std::pow(10.0, -1.0 / 20.0);
        std::vector<double> x(96000);
        LimiterTestNoise noise { 0x4321u };
        for (double& v : x) v = 2.0 * ceiling * noise.next();
        Lim lim;
        lim.setCeiling(-1.0);
        lim.prepare(48000.0, 2);
        std::vector<double> left = x, right = x;
        int block = 0;
        double worst = 0.0;
        bool finite = true;
        for (size_t o = 0; o < x.size(); o += 512, ++block)
        {
            if (block == 40) lim.setPrecision(Lim::Precision::High);
            if (block == 120) lim.setPrecision(Lim::Precision::Standard);
            const int len = static_cast<int>(std::min<size_t>(512, x.size() - o));
            double* ch[2] = { left.data() + o, right.data() + o };
            lim.processBlock(AudioBufferView<double>(ch, 2, len));
            for (int i = 0; i < len; ++i)
            {
                finite = finite && std::isfinite(left[o + static_cast<size_t>(i)]);
                worst = std::max(worst, std::abs(left[o + static_cast<size_t>(i)]));
            }
        }
        EXPECT_TRUE(finite);
        EXPECT_TRUE(worst <= ceiling * (1.0 + 1e-12));
    }

    // The state carries the choice; a blob without it restores Standard.
    {
        Limiter<float> a;
        a.setPrecision(Limiter<float>::Precision::High);
        const auto blob = a.getState();
        StateReader r(blob.data(), blob.size());
        EXPECT_EQ(r.read("precision", int32_t(-1)), int32_t(1));
        Limiter<float> b;
        EXPECT_TRUE(b.setState(blob.data(), blob.size()));
        EXPECT_TRUE(b.getPrecision() == Limiter<float>::Precision::High);
        Limiter<float> plain;
        const auto old = plain.getState();
        EXPECT_TRUE(b.setState(old.data(), old.size()));
        EXPECT_TRUE(b.getPrecision() == Limiter<float>::Precision::Standard);
    }
}

namespace {

// A stereo tone through a processor in blocks of 256, with a hook per block.
template <class P, class Hook>
std::vector<float> m3Run(P& p, double hz, double fs, size_t n, double amp, Hook&& hook)
{
    std::vector<float> l(n), r(n);
    for (size_t i = 0; i < n; ++i)
        l[i] = r[i] = static_cast<float>(amp * std::sin(2.0 * 3.14159265358979323846 * hz * static_cast<double>(i) / fs));
    for (size_t off = 0; off < n; off += 256)
    {
        hook(off);
        float* ch[2] = { l.data() + off, r.data() + off };
        p.processBlock(AudioBufferView<float>(ch, 2, static_cast<int>(std::min<size_t>(256, n - off))));
    }
    return l;
}

double m3RmsDb(const std::vector<float>& v, size_t from, size_t to)
{
    double s = 0.0;
    for (size_t i = from; i < to; ++i) s += static_cast<double>(v[i]) * v[i];
    return 10.0 * std::log10(s / static_cast<double>(to - from) + 1e-30);
}

} // namespace

// New: the detectors of the de-esser and of the dynamic equalizer keep the
// width of the band they move up to the top of the audio band. Their
// band-pass is matched to the analog one: a tone half an octave under a
// 10 kHz band and one half an octave over it, at the same level, read the
// same at 44.1 kHz (the bilinear band-pass narrows toward Nyquist: the
// de-esser read 5.2 and 3.2 dB of reduction, the equalizer 4.2 and 2.7).
DSPARK_TEST(DeEsser_and_DynamicEQ_detectors_keep_their_width_near_the_top)
{
    const double fs = 44100.0, centre = 10000.0, edges[2] = { centre / std::sqrt(2.0), centre * std::sqrt(2.0) };
    double reduction[2] = { 0.0, 0.0 }, gain[2] = { 0.0, 0.0 };
    for (int k = 0; k < 2; ++k)
    {
        auto d = std::make_unique<DeEsser<float>>();
        d->setFrequency(static_cast<float>(centre));
        d->setBandwidth(1.0f);
        d->setThreshold(-30.0f);
        d->setReduction(12.0f);
        d->prepare(spec(fs, 256, 2));
        (void)m3Run(*d, edges[k], fs, 44100, 0.1, [](size_t) {});
        reduction[k] = static_cast<double>(d->getGainReductionDb());

        auto e = std::make_unique<DynamicEQ<float>>();
        e->prepare(spec(fs, 256, 2));
        DynamicEQ<float>::BandConfig c;
        c.frequency = static_cast<float>(centre);
        c.q = 1.4142f;
        c.threshold = -30.0f;
        c.aboveRatio = 4.0f;
        c.aboveRangeDb = 12.0f;
        e->setBand(0, c);
        (void)m3Run(*e, edges[k], fs, 44100, 0.1, [](size_t) {});
        gain[k] = static_cast<double>(e->getBandGainDb(0));
    }
    std::cout << "  [band detectors] de-esser " << reduction[0] << " / " << reduction[1] << " dB, dynamic eq "
              << gain[0] << " / " << gain[1] << " dB\n";
    EXPECT_GT(std::abs(reduction[0]), 5.0);
    EXPECT_NEAR(reduction[0], reduction[1], 0.5);
    EXPECT_LT(gain[0], -4.0);
    EXPECT_NEAR(gain[0], gain[1], 0.5);
}

// New: the range of a mixed band changed while the band is at work. The
// weight of the band's end follows the 20 ms glide of its filter, so a tone
// held 6 dB down goes to 18 dB down without rising on the way (the weight
// used to jump to the new range while the filter was still at the old one:
// the tone came up 2.5 dB first).
DSPARK_TEST(DynamicEQ_mixed_band_follows_a_change_of_range_without_a_bump)
{
    const double fs = 48000.0;
    auto e = std::make_unique<DynamicEQ<float>>();
    e->setMixedBands(true);
    e->prepare(spec(fs, 256, 2));
    DynamicEQ<float>::BandConfig c;
    c.frequency = 1000.0f;
    c.q = 1.0f;
    c.threshold = -50.0f;
    c.aboveRatio = 20.0f;
    c.aboveRangeDb = 6.0f;
    e->setBand(0, c);
    const size_t n = 96000, at = 48000 / 256 * 256;
    const auto out = m3Run(*e, 1000.0, fs, n, 0.3, [&](size_t off) {
        if (off == at) { c.aboveRangeDb = 18.0f; e->setBand(0, c); }
    });
    const double in = 20.0 * std::log10(0.3 / std::sqrt(2.0));
    const double before = m3RmsDb(out, at - 4800, at) - in;
    double highest = -300.0;
    for (size_t w = at; w + 48 <= at + 4800; w += 48) highest = std::max(highest, m3RmsDb(out, w, w + 48) - in);
    const double settled = m3RmsDb(out, n - 4800, n) - in;
    std::cout << "  [mixed band range] before " << before << " dB, highest on the way " << highest << " dB, settled "
              << settled << " dB\n";
    EXPECT_NEAR(before, -6.0, 0.3);
    EXPECT_NEAR(settled, -18.0, 0.3);
    EXPECT_LT(highest, before + 0.5);
}

// ============================================================================
// The listened key, the lookahead inside a kept latency, the hold's window
// ============================================================================

namespace {

// A pair of equal channels through a processor with a key of its own.
template <class P>
void m5RunKeyed(P& p, std::vector<float>& audio, std::vector<float>& key)
{
    std::vector<float> audio2 = audio, key2 = key;
    for (size_t off = 0; off < audio.size(); off += 256)
    {
        const int n = static_cast<int>(std::min<size_t>(256, audio.size() - off));
        float* ch[2] = { audio.data() + off, audio2.data() + off };
        float* kc[2] = { key.data() + off, key2.data() + off };
        p.processBlock(AudioBufferView<float>(ch, 2, n), AudioBufferView<float>(kc, 2, n));
    }
}

// How late an impulse in the key leaves when the key is listened to.
template <class P>
int m5ListenedDelay(P& p)
{
    const size_t n = 4096, at = 600;
    std::vector<float> audio(n, 0.0f), key(n, 0.0f);
    key[at] = 0.5f;
    m5RunKeyed(p, audio, key);
    size_t top = 0;
    for (size_t i = 1; i < n; ++i)
        if (std::abs(audio[i]) > std::abs(audio[top])) top = i;
    return static_cast<int>(top) - static_cast<int>(at);
}

// The largest sample that goes out of a processor whose key, a tone at half
// scale under a bell of +24 dB, is listened to.
template <class P>
double m5ListenedPeak()
{
    auto p = std::make_unique<P>();
    typename P::KeyBand bell;
    bell.shape = decltype(bell.shape)::Bell;
    bell.frequency = 1000.0f;
    bell.gainDb = 24.0f;
    bell.q = 1.0f;
    p->setSidechainBand(0, bell);
    p->setSidechainListen(true);
    p->prepare(AudioSpec{ 48000.0, 256, 2 });
    const auto out = m3Run(*p, 1000.0, 48000.0, 24000, 0.5, [](size_t) {});
    double top = 0.0;
    for (size_t i = 12000; i < out.size(); ++i) top = std::max(top, std::abs(static_cast<double>(out[i])));
    return top;
}

template <class P>
int m5ListenedDelayWithLookahead(float ms)
{
    auto p = std::make_unique<P>();
    p->setLookahead(ms);
    p->setSidechainListen(true);
    p->prepare(AudioSpec{ 48000.0, 256, 2 });
    const int late = m5ListenedDelay(*p);
    return late - p->getLatency();
}

} // namespace

// Defect: the listened key left without the delay of the lookahead, so with a
// lookahead it played early by the whole latency the processor reports, and
// an equalized key went out as loud as its bands made it. It now leaves late
// by the latency and held under full scale, in the compressor, the gate and
// the expander; without a lookahead and under full scale it is the key
// itself.
DSPARK_TEST(Dynamics_listened_key_keeps_its_place_and_stays_under_full_scale)
{
    EXPECT_EQ(m5ListenedDelayWithLookahead<Compressor<float>>(5.0f), 0);
    EXPECT_EQ(m5ListenedDelayWithLookahead<NoiseGate<float>>(5.0f), 0);
    EXPECT_EQ(m5ListenedDelayWithLookahead<Expander<float>>(5.0f), 0);
    EXPECT_EQ(m5ListenedDelayWithLookahead<Compressor<float>>(0.0f), 0);
    {
        auto c = std::make_unique<Compressor<float>>();
        c->setLookahead(5.0f);
        c->prepare(AudioSpec{ 48000.0, 256, 2 });
        EXPECT_EQ(c->getLatency(), 240);
    }
    const double comp = m5ListenedPeak<Compressor<float>>(), gate = m5ListenedPeak<NoiseGate<float>>(),
                 expander = m5ListenedPeak<Expander<float>>();
    std::cout << "  [listen] a key with a bell of +24 dB goes out at " << comp << " (compressor), " << gate << " (gate), "
              << expander << " (expander) of full scale\n";
    for (const double peak : { comp, gate, expander })
    {
        EXPECT_LT(peak, 1.0 + 1e-6);
        EXPECT_GT(peak, 0.9);
    }
    // Under full scale and with no lookahead the listened key is the key.
    {
        auto c = std::make_unique<Compressor<float>>();
        c->setSidechainListen(true);
        c->prepare(AudioSpec{ 48000.0, 256, 2 });
        const auto out = m3Run(*c, 440.0, 48000.0, 4096, 0.4, [](size_t) {});
        size_t differing = 0;
        for (size_t i = 0; i < out.size(); ++i)
            if (out[i] != static_cast<float>(0.4 * std::sin(2.0 * 3.14159265358979323846 * 440.0 * static_cast<double>(i) / 48000.0))) ++differing;
        EXPECT_EQ(differing, static_cast<size_t>(0));
    }
}

// Defect: with a latency kept for the lookahead, moving the lookahead jumped
// the gain to another point of its own past: a step of level whenever the
// gain was moving. The delay of the gain now slides to its new value.
DSPARK_TEST(Compressor_lookahead_moves_inside_its_latency_without_a_step)
{
    auto c = std::make_unique<Compressor<float>>();
    c->setThreshold(-20.0f);
    c->setRatio(4.0f);
    c->setAttack(5.0f);
    c->setRelease(60.0f);
    c->setMaxLookahead(10.0f);
    c->setLookahead(8.0f);
    c->prepare(AudioSpec{ 48000.0, 256, 2 });
    const size_t n = 256 * 560, at = 256 * 375;
    std::vector<float> l(n), r;
    for (size_t i = 0; i < n; ++i)
        l[i] = static_cast<float>(0.5 * (1.0 + 0.6 * std::sin(2.0 * 3.14159265358979323846 * 3.0 * static_cast<double>(i) / 48000.0)));
    r = l;
    for (size_t off = 0; off < n; off += 256)
    {
        if (off == at) c->setLookahead(0.0f);
        float* ch[2] = { l.data() + off, r.data() + off };
        c->processBlock(AudioBufferView<float>(ch, 2, 256));
    }
    double steady = 0.0, moving = 0.0;
    for (size_t i = 24000; i < at; ++i) steady = std::max(steady, std::abs(static_cast<double>(l[i]) - l[i - 1]));
    for (size_t i = at; i < at + 9600; ++i) moving = std::max(moving, std::abs(static_cast<double>(l[i]) - l[i - 1]));
    std::cout << "  [lookahead] largest step of the output: " << steady << " before the move, " << moving << " during it\n";
    EXPECT_LT(moving, 3.0 * steady);
    EXPECT_EQ(c->getLatency(), 480);
}

// New: HoldMode::EveryPeak holds every gain change for the hold time (the
// deepest of a sliding window). On a tone that dies away, with a hold longer
// than its period, the reduction comes back in steps from crest to crest;
// with HoldMode::Deepest it falls into the wave whenever a hold ends. Either
// way a reduction is kept for the hold time, and then let go.
DSPARK_TEST(Compressor_hold_every_peak_follows_the_crests)
{
    using C = Compressor<float>;
    const auto make = [](C::HoldMode mode, float holdMs, float releaseMs) {
        auto c = std::make_unique<C>();
        c->setThreshold(-40.0f);
        c->setRatio(4.0f);
        c->setAttack(0.1f);
        c->setRelease(releaseMs);
        c->setHoldTime(holdMs);
        c->setHoldMode(mode);
        c->prepare(AudioSpec{ 48000.0, 8, 2 });
        return c;
    };
    const auto trace = [](C& c, const std::vector<float>& x) {
        std::vector<float> l = x, r = x;
        std::vector<double> gr;
        for (size_t off = 0; off + 8 <= l.size(); off += 8)
        {
            float* ch[2] = { l.data() + off, r.data() + off };
            c.processBlock(AudioBufferView<float>(ch, 2, 8));
            gr.push_back(static_cast<double>(c.getGainReductionDb()));
        }
        return gr;
    };
    // A 50 Hz tone that dies away; a hold of three periods.
    std::vector<float> dying(96000);
    for (size_t i = 0; i < dying.size(); ++i)
    {
        const double t = static_cast<double>(i) / 48000.0;
        dying[i] = static_cast<float>(0.8 * std::exp(-1.5 * t) * std::sin(2.0 * 3.14159265358979323846 * 50.0 * t));
    }
    const auto deeperAgain = [&](C::HoldMode mode) {
        auto c = make(mode, 60.0f, 20.0f);
        const auto gr = trace(*c, dying);
        double shallowest = -1e9, again = 0.0;
        for (size_t i = 480; i < gr.size(); ++i)   // from 80 ms on
        {
            shallowest = std::max(shallowest, gr[i]);
            again = std::max(again, shallowest - gr[i]);
        }
        return again;
    };
    const double deepest = deeperAgain(C::HoldMode::Deepest), every = deeperAgain(C::HoldMode::EveryPeak);
    std::cout << "  [hold] on a tone that dies away the reduction goes deeper again by " << deepest << " dB (deepest) and "
              << every << " dB (every peak)\n";
    EXPECT_GT(deepest, 4.0 * every);
    EXPECT_LT(every, 1.0);
    // A level that stops: the reduction stays for the hold time, then goes.
    std::vector<float> burst(48000, 0.0f);
    for (size_t i = 0; i < 4800; ++i) burst[i] = 0.5f;
    for (const C::HoldMode mode : { C::HoldMode::Deepest, C::HoldMode::EveryPeak })
    {
        auto c = make(mode, 100.0f, 5.0f);
        const auto gr = trace(*c, burst);
        const double held = gr[(4800 + 4300) / 8], atEnd = gr[(4800 + 4800 + 1500) / 8], depth = gr[4700 / 8];
        EXPECT_LT(depth, -6.0);
        EXPECT_NEAR(held, depth, 0.5);       // 90 ms after the level stopped
        EXPECT_GT(atEnd, -1.0);              // 30 ms after the hold ended
    }
    // The mode is part of the state.
    auto from = std::make_unique<C>();
    from->setHoldMode(C::HoldMode::EveryPeak);
    const auto blob = from->getState();
    auto to = std::make_unique<C>();
    EXPECT_TRUE(to->setState(blob.data(), blob.size()));
    EXPECT_TRUE(to->getHoldMode() == C::HoldMode::EveryPeak);
    EXPECT_TRUE(std::make_unique<C>()->getHoldMode() == C::HoldMode::Deepest);
}

// New: LookaheadShape::Ramp brings each reduction in along an S-shaped ramp
// that ends exactly when its peak leaves the delay. On a level that jumps
// 20 dB with a fast attack, the gain of the default shape falls in a few
// samples a whole lookahead early; the ramp takes the lookahead to come
// down, has the settled reduction when the jump arrives and lets no more
// through than the default.
DSPARK_TEST(Compressor_lookahead_ramp_arrives_with_the_peak)
{
    using C = Compressor<float>;
    struct Reading { double atArrival = 0.0, settled = 0.0, step = 0.0, peakOver = 0.0; };
    const auto read = [](C::LookaheadShape shape, float keptMs) {
        auto c = std::make_unique<C>();
        c->setThreshold(-30.0f);
        c->setRatio(4.0f);
        c->setAttack(0.1f);
        c->setRelease(50.0f);
        if (keptMs > 0.0f) c->setMaxLookahead(keptMs);
        c->setLookahead(5.0f);
        c->setLookaheadShape(shape);
        c->prepare(AudioSpec{ 48000.0, 256, 2 });
        const size_t n = 256 * 300, at = 38400, len = 4800;
        std::vector<float> l(n, 0.03f), r;
        for (size_t i = at; i < at + len; ++i) l[i] = 0.3f;
        r = l;
        const std::vector<float> in = l;
        for (size_t off = 0; off < n; off += 256)
        {
            float* ch[2] = { l.data() + off, r.data() + off };
            c->processBlock(AudioBufferView<float>(ch, 2, 256));
        }
        const auto lat = static_cast<size_t>(c->getLatency());
        const auto gainAt = [&](size_t i) { return 20.0 * std::log10(std::max(1e-9, static_cast<double>(l[i]) / in[i - lat])); };
        const size_t arrive = at + lat;
        Reading out;
        out.atArrival = gainAt(arrive);
        out.settled = gainAt(arrive + len - 10);
        double top = 0.0;
        for (size_t i = arrive - 2000; i < arrive + len; ++i) out.step = std::max(out.step, std::abs(gainAt(i) - gainAt(i - 1)));
        for (size_t i = arrive; i < arrive + len; ++i) top = std::max(top, static_cast<double>(l[i]));
        out.peakOver = 20.0 * std::log10(top / static_cast<double>(l[arrive + len - 10]));
        return out;
    };
    const Reading lead = read(C::LookaheadShape::Lead, 0.0f), ramp = read(C::LookaheadShape::Ramp, 0.0f),
                  kept = read(C::LookaheadShape::Ramp, 10.0f);
    std::cout << "  [ramp] largest step of the gain: " << lead.step << " dB per sample (lead), " << ramp.step << " (ramp), " << kept.step
              << " (ramp inside a kept latency); gain when the jump arrives " << ramp.atArrival << " dB, settled " << ramp.settled
              << " dB; the output peaks " << ramp.peakOver << " dB over its settled level (lead " << lead.peakOver << ")\n";
    EXPECT_LT(ramp.step, 0.2 * lead.step);
    EXPECT_LT(kept.step, 0.2 * lead.step);
    EXPECT_NEAR(ramp.atArrival, ramp.settled, 0.3);
    EXPECT_NEAR(kept.atArrival, kept.settled, 0.3);
    EXPECT_LT(ramp.peakOver, 0.3);
    EXPECT_LT(ramp.settled, -10.0);
    // The shape is part of the state.
    auto from = std::make_unique<C>();
    from->setLookaheadShape(C::LookaheadShape::Ramp);
    const auto blob = from->getState();
    auto to = std::make_unique<C>();
    EXPECT_TRUE(to->setState(blob.data(), blob.size()));
    EXPECT_TRUE(to->getLookaheadShape() == C::LookaheadShape::Ramp);
    EXPECT_TRUE(std::make_unique<C>()->getLookaheadShape() == C::LookaheadShape::Lead);
}

// New: setFeedbackKeyed(true) closes a feedback loop around the key ahead of
// the audio. With the programme as the key and no lookahead it is the same
// compressor; a lookahead then acts in feedback, with its latency and the
// same settled level; and an external key, which a plain feedback loop
// ignores, drives the reduction.
DSPARK_TEST(Compressor_feedback_keyed_takes_a_lookahead_and_an_external_key)
{
    using C = Compressor<float>;
    const size_t n = 256 * 190;
    std::vector<float> programme(n);
    for (size_t i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i) / 48000.0;
        programme[i] = static_cast<float>((0.25 + 0.25 * std::sin(2.0 * 3.14159265358979323846 * 2.0 * t))
                                          * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * t));
    }
    struct Run { std::vector<float> out; int latency = 0; double reduction = 0.0; };
    const auto render = [&](const std::vector<float>& x, const std::vector<float>* key, float ratio, bool keyed, float lookaheadMs) {
        auto c = std::make_unique<C>();
        c->setCharacter(C::Character::FET);
        c->setThreshold(-24.0f);
        c->setRatio(ratio);
        c->setFeedbackKeyed(keyed);
        c->setLookahead(lookaheadMs);
        c->prepare(AudioSpec{ 48000.0, 256, 2 });
        Run run;
        run.out = x;
        std::vector<float> right = x, keyL, keyR;
        if (key != nullptr) { keyL = *key; keyR = *key; }
        for (size_t off = 0; off < n; off += 256)
        {
            float* ch[2] = { run.out.data() + off, right.data() + off };
            if (key != nullptr)
            {
                float* kc[2] = { keyL.data() + off, keyR.data() + off };
                c->processBlock(AudioBufferView<float>(ch, 2, 256), AudioBufferView<float>(kc, 2, 256));
            }
            else c->processBlock(AudioBufferView<float>(ch, 2, 256));
        }
        run.latency = c->getLatency();
        run.reduction = static_cast<double>(c->getGainReductionDb());
        return run;
    };
    // The same compressor with the programme as its key.
    const Run plain = render(programme, nullptr, 4.0f, false, 0.0f), keyed = render(programme, nullptr, 4.0f, true, 0.0f);
    double apart = 0.0;
    for (size_t i = 0; i < n; ++i) apart = std::max(apart, std::abs(static_cast<double>(plain.out[i]) - keyed.out[i]));
    EXPECT_LT(apart, 1e-6);
    EXPECT_EQ(keyed.latency, 0);
    // A lookahead: ignored by the plain loop, taken by the keyed one, which settles at the same level.
    std::vector<float> steady(n);
    for (size_t i = 0; i < n; ++i) steady[i] = static_cast<float>(0.5 * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * static_cast<double>(i) / 48000.0));
    for (const float ratio : { 4.0f, 20.0f })
    {
        const Run none = render(steady, nullptr, ratio, false, 3.0f), ahead = render(steady, nullptr, ratio, true, 3.0f);
        EXPECT_EQ(none.latency, 0);
        EXPECT_EQ(ahead.latency, 144);
        const double a = m3RmsDb(none.out, n - 12000, n), b = m3RmsDb(ahead.out, n - 12000, n);
        std::cout << "  [keyed] ratio " << ratio << ": settled level " << a << " dB plain, " << b << " dB keyed with a lookahead\n";
        EXPECT_NEAR(b, a, 0.05);
    }
    // An external key over a quiet programme.
    std::vector<float> quiet(n), loud(n);
    for (size_t i = 0; i < n; ++i)
    {
        quiet[i] = static_cast<float>(0.03 * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * static_cast<double>(i) / 48000.0));
        loud[i] = static_cast<float>(0.5 * std::sin(2.0 * 3.14159265358979323846 * 300.0 * static_cast<double>(i) / 48000.0));
    }
    const Run ignored = render(quiet, &loud, 4.0f, false, 0.0f), driven = render(quiet, &loud, 4.0f, true, 0.0f);
    std::cout << "  [keyed] a loud key over a quiet programme: reduction " << ignored.reduction << " dB plain, " << driven.reduction << " dB keyed\n";
    EXPECT_GT(ignored.reduction, -0.5);
    EXPECT_LT(driven.reduction, -6.0);
    // Part of the state.
    auto from = std::make_unique<C>();
    from->setFeedbackKeyed(true);
    const auto blob = from->getState();
    auto to = std::make_unique<C>();
    EXPECT_TRUE(to->setState(blob.data(), blob.size()));
    EXPECT_TRUE(to->isFeedbackKeyed());
    EXPECT_FALSE(std::make_unique<C>()->isFeedbackKeyed());
}

// New: a zone of its own for the detector of a band. A band at 300 Hz that
// listens between 50 and 90 Hz leaves a tone at 300 Hz alone and gives way to
// one in the zone, which it reads at its level in the middle and 3 dB down on
// an edge (less what the other edge takes and the scale gives back). One edge
// alone is a half of the spectrum. In both engines, kept by the state and
// guarded on the way in.
DSPARK_TEST(DynamicEQ_band_listens_to_a_zone_of_its_own)
{
    using DE = DynamicEQ<float>;
    const int n = 48000;
    auto gainAt300 = [&](const DE::BandConfig& band, bool mixed, double keyHz, double keyAmplitude) {
        auto d = dynEqMake(mixed, 1, band);
        std::vector<Vec> x { makeSine(300.0, 0.01, 48000.0, n) }, key { makeSine(keyHz, keyAmplitude, 48000.0, n) };
        dynEqRun(*d, x, key);
        return toDb(toneAmplitude(x[0].data() + n - 9600, 9600, 300.0, 48000.0) / 0.01);
    };

    const auto plain = dynEqCut(300.0f, 2.0f, -20.0f, 4.0f, 12.0f);
    auto zone = plain;
    zone.listenLow = 50.0f; zone.listenHigh = 90.0f;
    auto above = plain, below = plain;
    above.listenLow = 2000.0f;
    below.listenHigh = 100.0f;
    const double middle = std::sqrt(50.0 * 90.0);

    for (bool mixed : { false, true })
    {
        // Where it acts, as before: 14 dB over the threshold at 4:1.
        EXPECT_NEAR(gainAt300(plain, mixed, 300.0, 0.5), -10.5, 0.25);
        const double own = gainAt300(zone, mixed, 300.0, 0.5), inZone = gainAt300(zone, mixed, middle, 0.5);
        const double onEdge = gainAt300(zone, mixed, 50.0, 0.5);
        const double low = gainAt300(above, mixed, 300.0, 0.5), high = gainAt300(above, mixed, 5000.0, 0.5);
        const double top = gainAt300(below, mixed, 5000.0, 0.5), bottom = gainAt300(below, mixed, 50.0, 0.5);
        std::printf("  [zone] %s engine, a band at 300 Hz listening to 50-90 Hz: its own tone %.2f dB, a tone in the middle of the zone %.2f, one on its low edge %.2f; listening above 2 kHz: 300 Hz %.2f, 5 kHz %.2f; below 100 Hz: 5 kHz %.2f, 50 Hz %.2f\n",
                    mixed ? "mixed" : "default", own, inZone, onEdge, low, high, top, bottom);
        EXPECT_NEAR(own, 0.0, 0.05);            // 300 Hz reads 18.6 dB lower in the zone: under the threshold
        EXPECT_NEAR(inZone, -10.48, 0.25);      // read at its level
        EXPECT_NEAR(onEdge, -9.68, 0.3);        // -3.01 (its edge) - 0.40 (the other) + 2.34 (the scale)
        EXPECT_NEAR(low, 0.0, 0.05);
        EXPECT_NEAR(high, -10.40, 0.3);
        EXPECT_NEAR(top, 0.0, 0.05);
        EXPECT_NEAR(bottom, -10.29, 0.3);
    }

    // Program timing takes its times from the zone: the band still lands on the curve.
    auto timed = zone;
    timed.timing = DE::Timing::Program;
    EXPECT_NEAR(gainAt300(timed, false, middle, 0.5), -10.48, 0.3);

    // The state keeps the zone; edges the wrong way round are put in order, what is
    // not a value keeps the band's own, and an edge below zero is no edge.
    auto a = dynEqMake(false, 1, zone);
    const auto blob = a->getState();
    auto b = std::make_unique<DE>();
    b->prepare(AudioSpec{ 48000.0, 512, 1 });
    EXPECT_TRUE(b->setState(blob.data(), blob.size()));
    EXPECT_TRUE(b->getState() == blob);
    EXPECT_FALSE(blob == dynEqMake(false, 1, plain)->getState());
    auto wild = zone;
    wild.listenLow = 90.0f; wild.listenHigh = 50.0f;
    a->setBand(0, wild);
    EXPECT_TRUE(a->getState() == blob);
    wild.listenLow = std::numeric_limits<float>::quiet_NaN();
    wild.listenHigh = std::numeric_limits<float>::infinity();
    a->setBand(0, wild);
    EXPECT_TRUE(a->getState() == blob);
    wild = zone;
    wild.listenLow = -5.0f;
    auto none = zone;
    none.listenLow = 0.0f;
    EXPECT_TRUE(dynEqMake(false, 1, wild)->getState() == dynEqMake(false, 1, none)->getState());
}

// New: key listen. The output is what one band's detector hears: the key
// through the band's detector filters (its zone when it has one), late by the
// latency, on the channels of the band's scope. The bands keep working
// underneath, so the output is the processed signal again once it ends.
DSPARK_TEST(DynamicEQ_key_listen_gives_what_a_band_hears)
{
    using DE = DynamicEQ<float>;
    const int n = 48000;
    Vec in = makeSine(1000.0, 0.3, 48000.0, n);
    {
        const Vec low = makeSine(100.0, 0.3, 48000.0, n);
        for (int i = 0; i < n; ++i) in[static_cast<size_t>(i)] += low[static_cast<size_t>(i)];
    }
    const auto band = dynEqCut(1000.0f, 2.0f, -30.0f, 4.0f, 12.0f);

    // What the detector of that band hears.
    Vec heard(static_cast<size_t>(n));
    {
        Biquad<float> detector;
        detector.setCoeffsNow(BiquadCoeffs::makeBandPassMatched(48000.0, 1000.0f, 2.0f));
        for (int i = 0; i < n; ++i) heard[static_cast<size_t>(i)] = detector.processSample(in[static_cast<size_t>(i)], 0);
    }
    auto apart = [&](const Vec& y, const Vec& reference, int late, int from) {
        double worst = 0.0;
        for (int i = std::max(from, late); i < n; ++i)
            worst = std::max(worst, std::abs(static_cast<double>(y[static_cast<size_t>(i)])
                                             - static_cast<double>(reference[static_cast<size_t>(i - late)])));
        return worst;
    };

    // No latency, a lookahead, and the linear-phase mode: the key, late by each.
    for (int mode = 0; mode < 3; ++mode)
    {
        auto d = mode == 2 ? dynEqLinear(1, band) : dynEqMake(false, 1, band);
        if (mode == 1) d->setLookahead(5.0f);
        d->setKeyListen(0);
        std::vector<Vec> x { in };
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        const int late = d->getLatency();
        const double off = apart(x[0], heard, late, late + 4800);
        std::printf("  [key listen] latency %d: the output is the band's key within %.1e; 1 kHz at %.4f of 0.3, 100 Hz at %.4f\n",
                    late, off, toneAmplitude(x[0].data() + n - 9600, 9600, 1000.0, 48000.0),
                    toneAmplitude(x[0].data() + n - 9600, 9600, 100.0, 48000.0));
        if (mode == 0) { EXPECT_EQ(late, 0); }
        if (mode == 1) { EXPECT_EQ(late, 240); }
        if (mode == 2) { EXPECT_GT(late, 1000); }
        EXPECT_LT(off, 1e-5);
        EXPECT_NEAR(toneAmplitude(x[0].data() + n - 9600, 9600, 1000.0, 48000.0), 0.3, 0.003);
        EXPECT_LT(toneAmplitude(x[0].data() + n - 9600, 9600, 100.0, 48000.0), 0.02);
    }

    // It comes in through a fade, and the processing is back, bit for bit, once it has gone.
    {
        auto d = dynEqMake(false, 1, band), twin = dynEqMake(false, 1, band);
        std::vector<Vec> x { in }, y { in };
        std::vector<float*> px(1), py(1);
        double jump = 0.0;
        float before = 0.0f;
        for (int off = 0; off < n; off += 512)
        {
            const int m = std::min(512, n - off);
            if (off == 512 * 20) d->setKeyListen(0);
            if (off == 512 * 50) d->setKeyListen(-1);
            px[0] = x[0].data() + off;
            py[0] = y[0].data() + off;
            AudioBufferView<float> vx(px.data(), 1, m), vy(py.data(), 1, m);
            d->processBlock(vx);
            twin->processBlock(vy);
            for (int i = 0; i < m; ++i)
            {
                jump = std::max(jump, std::abs(static_cast<double>(px[0][i]) - static_cast<double>(before)));
                before = px[0][i];
            }
        }
        double listening = 0.0, after = 0.0;
        for (int i = 512 * 30; i < 512 * 50; ++i)
            listening = std::max(listening, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
        for (int i = 512 * 54; i < n; ++i)
            after = std::max(after, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) - y[0][static_cast<size_t>(i)]));
        // The input moves 0.045 from one sample to the next at most: nothing steps on the way in or out.
        std::printf("  [key listen] in and out of it: largest move between samples %.4f; apart from the processing %.3f while listening and %.1e after\n",
                    jump, listening, after);
        EXPECT_LT(jump, 0.06);
        EXPECT_GT(listening, 0.1);
        EXPECT_EQ(after, 0.0);
    }

    // An external key and the scopes: a band of the sides is heard with its channels
    // opposed, one of the left on the left alone, and none of the audio is in it.
    {
        const Vec audio = makeSine(3000.0, 0.25, 48000.0, n), shared = makeSine(100.0, 0.3, 48000.0, n);
        const Vec apartTone = makeSine(1000.0, 0.2, 48000.0, n);
        Vec keyL(static_cast<size_t>(n)), keyR(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            keyL[static_cast<size_t>(i)] = shared[static_cast<size_t>(i)] + apartTone[static_cast<size_t>(i)];
            keyR[static_cast<size_t>(i)] = shared[static_cast<size_t>(i)] - apartTone[static_cast<size_t>(i)];
        }
        auto side = band;
        side.scope = DE::Scope::Side;
        auto d = dynEqMake(false, 2, side);
        d->setKeyListen(0);
        std::vector<Vec> x { audio, audio }, key { keyL, keyR };
        dynEqRun(*d, x, key);
        double sum = 0.0;
        for (int i = 4800; i < n; ++i)
            sum = std::max(sum, std::abs(static_cast<double>(x[0][static_cast<size_t>(i)]) + x[1][static_cast<size_t>(i)]));
        EXPECT_EQ(sum, 0.0);
        EXPECT_NEAR(toneAmplitude(x[0].data() + n - 9600, 9600, 1000.0, 48000.0), 0.2, 0.002);
        EXPECT_LT(toneAmplitude(x[0].data() + n - 9600, 9600, 3000.0, 48000.0), 1e-4);

        auto left = band;
        left.scope = DE::Scope::Left;
        auto e = dynEqMake(false, 2, left);
        e->setKeyListen(0);
        std::vector<Vec> z { audio, audio };
        dynEqRun(*e, z, key);
        double right = 0.0;
        for (int i = 4800; i < n; ++i) right = std::max(right, std::abs(static_cast<double>(z[1][static_cast<size_t>(i)])));
        EXPECT_EQ(right, 0.0);
        EXPECT_NEAR(toneAmplitude(z[0].data() + n - 9600, 9600, 1000.0, 48000.0), 0.2, 0.002);
    }

    // A band with a zone: what is heard is the zone, at the level it is read at.
    {
        auto zone = band;
        zone.listenLow = 50.0f; zone.listenHigh = 90.0f;
        auto d = dynEqMake(false, 1, zone);
        d->setKeyListen(0);
        const double middle = std::sqrt(50.0 * 90.0);
        std::vector<Vec> x { makeSine(middle, 0.3, 48000.0, n) };
        const Vec distant = makeSine(1000.0, 0.3, 48000.0, n);
        for (int i = 0; i < n; ++i) x[0][static_cast<size_t>(i)] += distant[static_cast<size_t>(i)];
        processInBlocks(x, 512, [&](AudioBufferView<float> v) { d->processBlock(v); });
        EXPECT_NEAR(toneAmplitude(x[0].data() + n - 19200, 19200, middle, 48000.0), 0.3, 0.006);
        EXPECT_LT(toneAmplitude(x[0].data() + n - 19200, 19200, 1000.0, 48000.0), 0.3 * 0.012);
    }

    // The control: a band that is not one is no listen, and it is not part of the state.
    auto d = dynEqMake(false, 1, band);
    const auto blob = d->getState();
    EXPECT_EQ(d->getKeyListen(), -1);
    d->setKeyListen(99);
    EXPECT_EQ(d->getKeyListen(), -1);
    d->setKeyListen(-7);
    EXPECT_EQ(d->getKeyListen(), -1);
    d->setKeyListen(0);
    EXPECT_EQ(d->getKeyListen(), 0);
    EXPECT_TRUE(d->getState() == blob);
}

// ============================================================================
// Compressor: what stands between the ballistics and the gain, and the range
// ============================================================================

// New: the landing of the release. The gain that is heard takes every deeper
// reduction at once and comes back behind the ballistics, slower the nearer it
// is: a release keeps its start and its last decibel takes several times
// longer, with no bump on the way.
DSPARK_TEST(Compressor_release_landing_brings_the_last_decibels_back_slowly)
{
    using C = Compressor<float>;
    const auto with = [](float landingMs) {
        return [=](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-30.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(100.0f);
            c.setReleaseLanding(landingMs);
        };
    };
    // 16 dB over the threshold for a second (12 dB of reduction), then 40 dB lower.
    Vec x = makeSine(1000.0, 0.2, 48000.0, 48000 * 4);
    for (size_t i = 48000; i < x.size(); ++i) x[i] *= 0.01f;
    const auto plain = compTrace(x, with(0.0f)), landed = compTrace(x, with(300.0f));
    const size_t drop = 48000 / 16;
    const auto reaches = [&](const std::vector<double>& trace, double depth) {
        for (size_t i = drop; i < trace.size(); ++i)
            if (std::abs(trace[i]) <= depth) return static_cast<double>(i - drop) * 16.0 * 1000.0 / 48000.0;
        return 1e9;
    };
    const double plainFirst = reaches(plain, 6.0), landedFirst = reaches(landed, 6.0);
    const double plainLast = reaches(plain, 0.1) - reaches(plain, 1.0), landedLast = reaches(landed, 0.1) - reaches(landed, 1.0);
    std::printf("  [landing] from 12 dB of reduction: 6 dB back after %.1f ms plain and %.1f ms with a 300 ms landing; the stretch from 1 dB to 0.1 dB takes %.1f and %.1f ms; held reduction %.3f and %.3f dB\n",
                plainFirst, landedFirst, plainLast, landedLast, plain[drop - 1], landed[drop - 1]);
    EXPECT_NEAR(plain[drop - 1], -12.0, 0.1);
    EXPECT_NEAR(landed[drop - 1], plain[drop - 1], 0.02);              // what is held is the same
    EXPECT_LT(landedFirst, plainFirst * 1.6);                           // the release starts as it did
    EXPECT_GT(landedLast, plainLast * 1.5);                             // and lands slowly
    double bump = 0.0, attack = 0.0;
    for (size_t i = drop + 1; i < landed.size(); ++i) bump = std::max(bump, landed[i - 1] - landed[i]);
    for (size_t i = 0; i < drop; ++i) attack = std::max(attack, std::abs(landed[i] - plain[i]));
    EXPECT_LT(bump, 1e-4);                                              // it only comes back
    EXPECT_LT(attack, 0.1);                                             // a deeper reduction is taken at once (the onset wobbles 0.06 dB)

    // One sample at a time is the block path; off, it is not there.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C& c) {
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setReleaseLanding(150.0f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setReleaseLanding(0.0f); })), 1e-3);

    // State and wild values.
    C a;
    a.setReleaseLanding(250.0f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getReleaseLanding(), 250.0f);
    b.setReleaseLanding(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(b.getReleaseLanding(), 250.0f);
    b.setReleaseLanding(1e6f);
    EXPECT_EQ(b.getReleaseLanding(), 2000.0f);
    b.setReleaseLanding(-3.0f);
    EXPECT_EQ(b.getReleaseLanding(), 0.0f);
}

// New: a lead-lag network in the control path. Of a sudden move of the gain a
// share arrives at once and the rest with the time constant of the corner;
// what the gain settles on does not move, and the ripple a fast release
// leaves on a low tone falls by the depth of the network.
DSPARK_TEST(Compressor_control_network_applies_a_share_of_a_fast_move_at_once)
{
    using C = Compressor<float>;
    const auto fast = [](float depthDb) {
        return [=](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-30.0f);
            c.setRatio(4.0f);
            c.setAttack(0.1f);
            c.setRelease(200.0f);
            c.setControlNetwork(depthDb, 5.0f);
        };
    };
    Vec step = makeSine(1000.0, 0.002, 48000.0, 48000 * 2);
    for (size_t i = 48000; i < step.size(); ++i) step[i] *= 100.0f;    // -54 -> -14 dBFS: 12 dB of reduction
    const auto plain = compTrace(step, fast(0.0f)), half = compTrace(step, fast(6.0206f));
    const size_t at = 48000 / 16;
    const size_t soon = at + 6;                                         // 2 ms after the step
    const double share = half[soon] / plain[soon];
    std::printf("  [network] 6 dB at 5 Hz: 2 ms after a step the gain is at %.2f dB of %.2f (a share of %.3f); settled %.3f and %.3f dB\n",
                half[soon], plain[soon], share, half.back(), plain.back());
    EXPECT_LT(plain[soon], -11.0);                                      // the attack is over (the detector overshoots a step by a decibel)
    EXPECT_GT(share, 0.49);
    EXPECT_LT(share, 0.60);                                             // half at once, 6 % of the rest in 2 ms
    EXPECT_NEAR(half.back(), plain.back(), 0.01);                       // the static curve does not move

    // The ripple of the gain on a 40 Hz tone with a quick release.
    const auto riding = [](float depthDb) {
        return [=](C& c) {
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(20.0f);
            c.setControlNetwork(depthDb, 5.0f);
        };
    };
    const auto ripple = [&](float depthDb, double& mean) {
        const auto trace = compTrace(makeSine(40.0, 0.5, 48000.0, 96000), riding(depthDb));
        double lo = 0.0, hi = -100.0, sum = 0.0;
        const size_t from = trace.size() - 1500;                        // the last half second: 20 periods
        for (size_t i = from; i < trace.size(); ++i)
        {
            lo = std::min(lo, trace[i]);
            hi = std::max(hi, trace[i]);
            sum += trace[i];
        }
        mean = sum / 1500.0;
        return hi - lo;
    };
    double meanPlain = 0.0, meanHalf = 0.0;
    const double ripplePlain = ripple(0.0f, meanPlain), rippleHalf = ripple(6.0206f, meanHalf);
    std::printf("  [network] gain ripple on a 40 Hz tone: %.2f dB plain, %.2f dB through the network; mean %.3f and %.3f dB\n",
                ripplePlain, rippleHalf, meanPlain, meanHalf);
    EXPECT_GT(ripplePlain, 1.0);
    EXPECT_NEAR(rippleHalf / ripplePlain, 0.5, 0.06);
    EXPECT_NEAR(meanHalf, meanPlain, 0.05);

    // The paths; a feedback loop reads the gain before the network and stays put.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C& c) {
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setControlNetwork(4.0f, 8.0f);
        c.setReleaseLanding(100.0f);
        c.setGainSmoothing(3.0f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setControlNetwork(0.0f); })), 1e-3);
    const auto loop = [](float depthDb) {
        return [=](C& c) {
            c.setTopology(C::Topology::FeedBack);
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(100.0f);
            c.setControlNetwork(depthDb, 5.0f);
        };
    };
    const Vec steady = makeSine(1000.0, 0.4, 48000.0, 48000);
    EXPECT_NEAR(compTrace(steady, loop(6.0f)).back(), compTrace(steady, loop(0.0f)).back(), 0.05);

    // Switched on while the gain is down, it starts from where the gain is.
    {
        auto c = std::make_unique<C>();
        fast(0.0f)(*c);
        c->prepare(AudioSpec{ 48000.0, 16, 1 });
        std::vector<Vec> x { makeSine(1000.0, 0.4, 48000.0, 48000) };
        int played = 0;
        double last = 0.0, jump = 0.0;
        processInBlocks(x, 16, [&](AudioBufferView<float> b) {
            if (played == 24000) c->setControlNetwork(9.0f, 5.0f);
            c->processBlock(b);
            played += b.getNumSamples();
            const double gr = static_cast<double>(c->getGainReductionDb());
            if (played > 12000) jump = std::max(jump, std::abs(gr - last));
            last = gr;
        });
        EXPECT_LT(jump, 0.05);
    }

    // State and wild values.
    C a;
    a.setControlNetwork(4.5f, 20.0f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getControlNetworkDepth(), 4.5f);
    EXPECT_EQ(b.getControlNetworkCorner(), 20.0f);
    b.setControlNetwork(std::numeric_limits<float>::quiet_NaN(), 50.0f);
    EXPECT_EQ(b.getControlNetworkDepth(), 4.5f);
    EXPECT_EQ(b.getControlNetworkCorner(), 20.0f);
    b.setControlNetwork(90.0f, 5000.0f);
    EXPECT_EQ(b.getControlNetworkDepth(), 12.0f);
    EXPECT_EQ(b.getControlNetworkCorner(), 100.0f);
    b.setControlNetwork(-2.0f, std::numeric_limits<float>::infinity());
    EXPECT_EQ(b.getControlNetworkDepth(), 0.0f);
    EXPECT_EQ(b.getControlNetworkCorner(), 100.0f);
}

// New: a range the gain change meets in a stretch of its own. Under the
// stretch the target is untouched, over it it is the range exactly, and in it
// a quadratic joins the two with no corner; the stretch is as wide as the
// range when the range is small.
DSPARK_TEST(Compressor_local_range_bends_only_around_the_range)
{
    using C = Compressor<float>;
    // A 1 kHz tone of amplitude 0.5 read by the RMS detector is at -9.03 dB: at 2:1 the
    // target is half of what that is over the threshold.
    auto reduction = [](double target, float range, C::RangeShape shape) {
        return compHeldReduction(0.5, 1.0, [&](C& c) {
            c.setDetector(C::DetectorType::Rms);
            c.setRatio(2.0f);
            c.setThreshold(static_cast<float>(-9.03 - 2.0 * target));
            c.setRange(range);
            c.setRangeShape(shape);
        });
    };
    const auto local = C::RangeShape::Local;
    EXPECT_NEAR(reduction(2.0, 6.0f, local), -2.0, 0.03);       // under the stretch (3 to 9 dB)
    EXPECT_NEAR(reduction(3.0, 6.0f, local), -3.0, 0.03);       // where it starts
    EXPECT_NEAR(reduction(3.2, 6.0f, local), -3.1967, 0.03);    // the slope goes on from 1
    EXPECT_NEAR(reduction(6.0, 6.0f, local), -5.25, 0.03);      // on the range: an eighth of the stretch short
    EXPECT_NEAR(reduction(8.8, 6.0f, local), -5.9967, 0.03);    // and arrives at 0
    EXPECT_NEAR(reduction(9.0, 6.0f, local), -6.0, 0.03);       // where it ends
    EXPECT_NEAR(reduction(25.0, 6.0f, local), -6.0, 0.03);      // far over: the range exactly
    EXPECT_NEAR(reduction(2.0, 2.0f, local), -1.75, 0.03);      // a range of 2 dB: a stretch of 2 dB
    EXPECT_NEAR(reduction(0.9, 2.0f, local), -0.9, 0.03);
    EXPECT_NEAR(reduction(4.0, 2.0f, local), -2.0, 0.03);
    // The other two shapes are what they were.
    EXPECT_NEAR(reduction(12.0, 6.0f, C::RangeShape::Asymptotic), -5.912, 0.03);
    EXPECT_NEAR(reduction(12.0, 6.0f, C::RangeShape::Corner), -6.0, 0.03);

    // The older switch is two of the shapes; the state keeps the third.
    C c;
    EXPECT_TRUE(c.getRangeShape() == C::RangeShape::Corner);
    c.setSoftRange(true);
    EXPECT_TRUE(c.getRangeShape() == C::RangeShape::Asymptotic);
    c.setRangeShape(local);
    EXPECT_TRUE(c.isSoftRange());
    const auto blob = c.getState();
    C other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.getRangeShape() == local);
    EXPECT_TRUE(other.getState() == blob);
    other.setRangeShape(static_cast<C::RangeShape>(40));
    EXPECT_TRUE(other.getRangeShape() == local);
    other.setSoftRange(false);
    EXPECT_TRUE(other.getRangeShape() == C::RangeShape::Corner);

    // One sample at a time is the block path.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [&](C& k) {
        k.setThreshold(-30.0f);
        k.setRatio(4.0f);
        k.setAttack(2.0f);
        k.setRelease(60.0f);
        k.setRange(8.0f);
        k.setRangeShape(local);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& k) { working(k); k.setRangeShape(C::RangeShape::Corner); })), 1e-3);
}

// New: a tone that changes with the reduction. The gain acts on a copy of the
// signal that has passed a nearly flat filter, y = x - (1 - g) F(x): with no
// reduction the output is the input bit for bit, and with 12 dB of it each
// tone is where that expression puts it.
DSPARK_TEST(Compressor_reduction_tone_leans_the_response_with_the_reduction)
{
    using C = Compressor<float>;
    const int n = 96000;
    const double hz[5] = { 30.0, 200.0, 1000.0, 4000.0, 16000.0 };
    const Vec key = makeSine(1000.0, 0.2, 48000.0, n);              // 16 dB over the threshold: 12 dB of reduction
    Vec probes(static_cast<size_t>(n), 0.0f);
    for (const double f : hz)
    {
        const Vec tone = makeSine(f, 0.01, 48000.0, n);
        for (int i = 0; i < n; ++i) probes[static_cast<size_t>(i)] += tone[static_cast<size_t>(i)];
    }
    double reductionDb = 0.0;
    const auto keyed = [&](float lowDb, float highDb) {
        C c;
        c.setDetector(C::DetectorType::Hilbert);
        c.setThreshold(-30.0f);
        c.setRatio(4.0f);
        c.setAttack(1.0f);
        c.setRelease(100.0f);
        c.setReductionTone(lowDb, highDb);
        c.prepare(AudioSpec{ 48000.0, 512, 1 });
        Vec audio = probes, k = key;
        float* pa[1]; float* pk[1];
        for (int off = 0; off < n; off += 512)
        {
            const int m = std::min(512, n - off);
            pa[0] = audio.data() + off;
            pk[0] = k.data() + off;
            AudioBufferView<float> va(pa, 1, m), vk(pk, 1, m);
            c.processBlock(va, vk);
        }
        reductionDb = static_cast<double>(c.getGainReductionDb());
        return audio;
    };
    const auto response = [](const BiquadCoeffs& s, double f) {
        const std::complex<double> z1 = std::polar(1.0, -2.0 * kPiD * f / 48000.0), z2 = z1 * z1;
        return (s.b0 + s.b1 * z1 + s.b2 * z2) / (1.0 + s.a1 * z1 + s.a2 * z2);
    };
    const auto level = [&](const Vec& y, double f) {
        return toDb(toneAmplitude(y.data() + n - 19200, 19200, f, 48000.0) / 0.01);
    };

    const Vec plain = keyed(0.0f, 0.0f);
    EXPECT_NEAR(reductionDb, -12.0, 0.05);
    for (const double f : hz) EXPECT_NEAR(level(plain, f), reductionDb, 0.02);

    for (int which = 0; which < 2; ++which)
    {
        const float lowDb = which == 0 ? 1.5f : -1.0f, highDb = which == 0 ? -1.5f : 1.0f;
        const Vec y = keyed(lowDb, highDb);
        const double g = std::pow(10.0, reductionDb / 20.0);
        const BiquadCoeffs low = BiquadCoeffs::makeLowShelfMatched(48000.0, C::kReductionToneLowHz, static_cast<double>(lowDb));
        const BiquadCoeffs high = BiquadCoeffs::makeHighShelfMatched(48000.0, C::kReductionToneHighHz, static_cast<double>(highDb));
        double worst = 0.0, shown[5] = {};
        for (int i = 0; i < 5; ++i)
        {
            const std::complex<double> filter = response(low, hz[i]) * response(high, hz[i]);
            const double expected = toDb(std::abs(1.0 - (1.0 - g) * filter));
            shown[i] = level(y, hz[i]);
            worst = std::max(worst, std::abs(shown[i] - expected));
        }
        std::printf("  [tone] %+.1f dB low and %+.1f dB high under %.2f dB of reduction: 30 Hz %.2f dB, 200 Hz %.2f, 1 kHz %.2f, 4 kHz %.2f, 16 kHz %.2f; farthest from y = x - (1 - g) F(x): %.3f dB\n",
                    static_cast<double>(lowDb), static_cast<double>(highDb), reductionDb,
                    shown[0], shown[1], shown[2], shown[3], shown[4], worst);
        EXPECT_LT(worst, 0.05);
        // Deeper where the filter is over 0 dB and shallower where it is under.
        if (which == 0) { EXPECT_LT(shown[0], reductionDb - 3.0); EXPECT_GT(shown[4], reductionDb + 2.0); }
        else            { EXPECT_GT(shown[0], reductionDb + 1.5); EXPECT_LT(shown[4], reductionDb - 2.0); }
    }

    // No reduction: the input, bit for bit.
    {
        C c;
        c.setThreshold(0.0f);
        c.setReductionTone(2.0f, -2.0f);
        c.prepare(AudioSpec{ 48000.0, 512, 2 });
        const Vec x = compLowOverHigh(24000);
        std::vector<Vec> v { x, x };
        processInBlocks(v, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        EXPECT_TRUE(std::memcmp(v[0].data(), x.data(), sizeof(float) * x.size()) == 0);
        EXPECT_TRUE(std::memcmp(v[1].data(), x.data(), sizeof(float) * x.size()) == 0);
    }

    // One sample at a time is the block path; the push-pull element applies the gain its own way.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C& c) {
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setReductionTone(1.0f, -2.0f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setReductionTone(0.0f, 0.0f); })), 1e-3);
    const auto pair = [](float toneDb) {
        return [=](C& c) {
            c.setCharacter(C::Character::Varimu);
            c.setThreshold(-24.0f);
            c.setRatio(4.0f);
            c.setCharacterColor(0.8f);
            c.setReductionTone(toneDb, toneDb);
        };
    };
    EXPECT_EQ(compFarthest(compRender(burst, pair(2.0f)), compRender(burst, pair(0.0f))), 0.0);

    // State and wild values.
    C a;
    a.setReductionTone(1.25f, -0.75f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getReductionToneLow(), 1.25f);
    EXPECT_EQ(b.getReductionToneHigh(), -0.75f);
    b.setReductionTone(std::numeric_limits<float>::quiet_NaN(), 40.0f);
    EXPECT_EQ(b.getReductionToneLow(), 1.25f);
    EXPECT_EQ(b.getReductionToneHigh(), 3.0f);
    b.setReductionTone(-40.0f, std::numeric_limits<float>::infinity());
    EXPECT_EQ(b.getReductionToneLow(), -3.0f);
    EXPECT_EQ(b.getReductionToneHigh(), 3.0f);
}

// New: the leak of the control. The moves of the gain, through a high-pass of
// a few Hz, are added to the output of the element: a thump at each attack
// and release in proportion to the amount, and nothing under a settled gain.
DSPARK_TEST(Compressor_control_leak_leaves_a_thump_where_the_gain_moves)
{
    using C = Compressor<float>;
    const int n = 96000;
    // A tone that comes and goes: 100 ms on, 150 ms off, 16 dB over the threshold.
    Vec bursts = makeSine(1000.0, 0.2, 48000.0, n);
    for (int i = 0; i < n; ++i)
        if ((i % 12000) >= 4800) bursts[static_cast<size_t>(i)] = 0.0f;
    const auto with = [](float amount) {
        return [=](C& c) {
            c.setDetector(C::DetectorType::Hilbert);
            c.setThreshold(-30.0f);
            c.setRatio(4.0f);
            c.setAttack(1.0f);
            c.setRelease(50.0f);
            c.setControlLeak(amount);
        };
    };
    const Vec none = compRender(bursts, with(0.0f)), half = compRender(bursts, with(0.5f)), whole = compRender(bursts, with(1.0f));
    double sumHalf = 0.0, sumWhole = 0.0, peak = 0.0, apart = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double h = static_cast<double>(half[static_cast<size_t>(i)]) - none[static_cast<size_t>(i)];
        const double w = static_cast<double>(whole[static_cast<size_t>(i)]) - none[static_cast<size_t>(i)];
        sumHalf += h * h;
        sumWhole += w * w;
        peak = std::max(peak, std::abs(w));
        apart = std::max(apart, std::abs(w - 2.0 * h));
    }
    const double doubling = 10.0 * std::log10(sumWhole / sumHalf);
    std::printf("  [leak] bursts under 12 dB of reduction: what the leak adds peaks at %.4f at an amount of 1 and rises %.2f dB from 0.5 to 1 (farthest from twice: %.1e)\n",
                peak, doubling, apart);
    EXPECT_NEAR(doubling, 6.02, 0.05);
    EXPECT_GT(peak, 0.02);                      // a gain that falls by 0.75 leaks 0.05 of that
    EXPECT_LT(peak, 0.04);

    // Under a gain that has settled it adds nothing.
    const Vec held = makeSine(1000.0, 0.2, 48000.0, n);
    const Vec heldNone = compRender(held, with(0.0f)), heldWhole = compRender(held, with(1.0f));
    const double settled = compFarthest(heldWhole, heldNone, static_cast<size_t>(n - 9600));
    std::printf("  [leak] a held tone, last 200 ms: what the leak adds stays under %.1e\n", settled);
    EXPECT_LT(settled, 1e-6);

    // Switched on while the gain is down, it starts from that gain: no thump.
    {
        auto c = std::make_unique<C>();
        with(0.0f)(*c);
        c->prepare(AudioSpec{ 48000.0, 512, 1 });
        std::vector<Vec> x { held };
        int played = 0;
        processInBlocks(x, 512, [&](AudioBufferView<float> b) {
            if (played == 512 * 90) c->setControlLeak(1.0f);
            c->processBlock(b);
            played += b.getNumSamples();
        });
        EXPECT_LT(compFarthest(x[0], heldNone, static_cast<size_t>(512 * 90)), 1e-4);
    }

    // One sample at a time is the block path.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C& c) {
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setControlLeak(0.7f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setControlLeak(0.0f); })), 1e-4);

    // State and wild values.
    C a;
    a.setControlLeak(0.35f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getControlLeak(), 0.35f);
    b.setControlLeak(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(b.getControlLeak(), 0.35f);
    b.setControlLeak(9.0f);
    EXPECT_EQ(b.getControlLeak(), 1.0f);
    b.setControlLeak(-9.0f);
    EXPECT_EQ(b.getControlLeak(), 0.0f);
}

// New: a detector that reads the two halves of the wave differently. A signal
// that is not symmetric gets a different reduction from its inverse, the gain
// moves at the frequency of the signal (a second harmonic that grows with the
// asymmetry), and a held symmetric tone settles where it did.
DSPARK_TEST(Compressor_asymmetric_detector_reads_the_two_halves_differently)
{
    using C = Compressor<float>;
    const int n = 96000;
    // Peaks of +0.3 and -0.6: a fundamental and its second harmonic.
    Vec lopsided(static_cast<size_t>(n)), inverse(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const double t = 2.0 * kPiD * 100.0 * i / 48000.0;
        lopsided[static_cast<size_t>(i)] = static_cast<float>(0.4 * std::sin(t) + 0.2 * std::cos(2.0 * t));
        inverse[static_cast<size_t>(i)] = -lopsided[static_cast<size_t>(i)];
    }
    const auto held = [](float asymmetry) {
        return [=](C& c) {
            c.setThreshold(-30.0f);
            c.setRatio(4.0f);
            c.setAttack(0.1f);
            c.setRelease(500.0f);
            c.setDetectorAsymmetry(asymmetry);
        };
    };
    const auto reduction = [&](const Vec& x, float asymmetry) { return compTrace(x, held(asymmetry)).back(); };
    const double even = reduction(lopsided, 0.0f) - reduction(inverse, 0.0f);
    const double toPositive = reduction(lopsided, 0.5f), fromPositive = reduction(inverse, 0.5f);
    std::printf("  [asymmetry] peaks of +0.3 and -0.6: the reduction of the signal and of its inverse differ by %.3f dB at 0 and by %.2f dB at 0.5 (%.2f and %.2f dB)\n",
                even, toPositive - fromPositive, toPositive, fromPositive);
    EXPECT_NEAR(even, 0.0, 0.02);
    EXPECT_GT(toPositive - fromPositive, 1.5);                  // its larger half is the one read less
    EXPECT_NEAR(reduction(lopsided, -0.5f), fromPositive, 0.05);    // the sign chooses the half
    // A symmetric tone settles where it did.
    const Vec tone = makeSine(1000.0, 0.5, 48000.0, n);
    EXPECT_NEAR(reduction(tone, 0.5f), reduction(tone, 0.0f), 0.1);
    EXPECT_NEAR(reduction(tone, -0.9f), reduction(tone, 0.0f), 0.1);

    // The second harmonic of a low tone whose gain rides the wave.
    const auto riding = [](float asymmetry) {
        return [=](C& c) {
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setAttack(0.5f);
            c.setRelease(5.0f);
            c.setDetectorAsymmetry(asymmetry);
        };
    };
    const Vec low = makeSine(100.0, 0.5, 48000.0, 48000);
    double second[4] = {}, third[4] = {};
    const float amounts[4] = { 0.0f, 0.2f, 0.5f, 0.9f };
    for (int i = 0; i < 4; ++i)
    {
        const Vec y = compRender(low, riding(amounts[i]));
        second[i] = compLineDb(y, 200.0) - compLineDb(y, 100.0);
        third[i] = compLineDb(y, 300.0) - compLineDb(y, 100.0);
    }
    std::printf("  [asymmetry] 100 Hz with a gain that rides the wave, second / third harmonic in dBc at 0, 0.2, 0.5 and 0.9: %.1f / %.1f, %.1f / %.1f, %.1f / %.1f, %.1f / %.1f\n",
                second[0], third[0], second[1], third[1], second[2], third[2], second[3], third[3]);
    EXPECT_LT(second[0], -70.0);
    EXPECT_GT(second[1], second[0] + 20.0);
    EXPECT_GT(second[2], second[1] + 3.0);
    EXPECT_GT(second[3], second[2]);
    for (int i = 1; i < 4; ++i) EXPECT_GT(second[i] - third[i], second[i - 1] - third[i - 1] + 0.5);    // against the third, too

    // A detector that does not rectify reads the key as it is; one sample at a time is the block path.
    const Vec burst = compLowOverHigh(24000);
    const auto mean = [](float asymmetry) {
        return [=](C& c) {
            c.setDetector(C::DetectorType::Rms);
            c.setThreshold(-24.0f);
            c.setRatio(4.0f);
            c.setDetectorAsymmetry(asymmetry);
        };
    };
    EXPECT_LT(compFarthest(compRender(burst, mean(0.8f)), compRender(burst, mean(0.0f))), 1e-6);
    const auto working = [](C& c) {
        c.setThreshold(-24.0f);
        c.setRatio(4.0f);
        c.setAttack(2.0f);
        c.setRelease(60.0f);
        c.setDetectorAsymmetry(0.6f);
    };
    const Vec block = compRender(burst, working);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working)), 1e-6);
    EXPECT_GT(compFarthest(block, compRender(burst, [&](C& c) { working(c); c.setDetectorAsymmetry(0.0f); })), 1e-3);

    // State and wild values.
    C a;
    a.setDetectorAsymmetry(-0.4f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getDetectorAsymmetry(), -0.4f);
    b.setDetectorAsymmetry(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(b.getDetectorAsymmetry(), -0.4f);
    b.setDetectorAsymmetry(7.0f);
    EXPECT_EQ(b.getDetectorAsymmetry(), 1.0f);
    b.setDetectorAsymmetry(-7.0f);
    EXPECT_EQ(b.getDetectorAsymmetry(), -1.0f);
}

// New: Mode::Both. One detector, two curves: what is over the threshold comes
// down with the ratio and what is under a second, lower threshold comes up
// with a ratio and a range of its own; between them, nothing. With no rise it
// is Downward bit for bit, and in a feedback loop it is Downward.
DSPARK_TEST(Compressor_both_mode_brings_the_loud_down_and_the_quiet_up)
{
    using C = Compressor<float>;
    // The RMS detector reads a tone of amplitude a at 20 log10(a) - 3.01 dB.
    const auto gainAt = [](double levelDb, C::Mode mode) {
        return compHeldReduction(std::pow(10.0, (levelDb + 3.0103) / 20.0), 1.0, [&](C& c) {
            c.setDetector(C::DetectorType::Rms);
            c.setThreshold(-20.0f);
            c.setRatio(4.0f);
            c.setMode(mode);
            c.setLift(30.0f, 2.0f, 6.0f);           // a lower threshold at -50 dB
        });
    };
    const double levels[10] = { 0.0, -10.0, -20.0, -30.0, -45.0, -50.0, -56.0, -62.0, -70.0, -80.0 };
    const double wanted[10] = { -15.0, -7.5, 0.0, 0.0, 0.0, 0.0, 3.0, 6.0, 6.0, 6.0 };
    double worst = 0.0, shown[10] = {};
    for (int i = 0; i < 10; ++i)
    {
        shown[i] = gainAt(levels[i], C::Mode::Both);
        worst = std::max(worst, std::abs(shown[i] - wanted[i]));
    }
    std::printf("  [both] gain at 0, -10, -20, -30, -45, -50, -56, -62, -70 and -80 dB: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f dB; farthest from the curve %.3f dB\n",
                shown[0], shown[1], shown[2], shown[3], shown[4], shown[5], shown[6], shown[7], shown[8], shown[9], worst);
    EXPECT_LT(worst, 0.1);
    EXPECT_NEAR(gainAt(-62.0, C::Mode::Downward), 0.0, 0.01);

    // No rise: Downward, bit for bit. In a loop: Downward too.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C::Mode mode, float riseDb, bool loop) {
        return [=](C& c) {
            c.setThreshold(-24.0f);
            c.setRatio(4.0f);
            c.setAttack(2.0f);
            c.setRelease(60.0f);
            c.setMode(mode);
            c.setLift(20.0f, 3.0f, riseDb);
            if (loop) c.setTopology(C::Topology::FeedBack);
        };
    };
    EXPECT_EQ(compFarthest(compRender(burst, working(C::Mode::Both, 0.0f, false)),
                           compRender(burst, working(C::Mode::Downward, 0.0f, false))), 0.0);
    EXPECT_EQ(compFarthest(compRender(burst, working(C::Mode::Both, 9.0f, true)),
                           compRender(burst, working(C::Mode::Downward, 9.0f, true))), 0.0);
    // With a rise it is another signal, and one sample at a time is the block path.
    Vec quiet = burst;
    for (auto& s : quiet) s *= 0.02f;
    const Vec block = compRender(quiet, working(C::Mode::Both, 9.0f, false));
    EXPECT_GT(compFarthest(block, compRender(quiet, working(C::Mode::Downward, 9.0f, false))), 1e-4);
    EXPECT_LT(compFarthest(block, compRenderSamples(quiet, working(C::Mode::Both, 9.0f, false))), 1e-6);

    // State and wild values.
    C a;
    a.setMode(C::Mode::Both);
    a.setLift(24.0f, 3.0f, 9.0f);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.getState() == blob);
    EXPECT_EQ(b.getLiftBelow(), 24.0f);
    EXPECT_EQ(b.getLiftRatio(), 3.0f);
    EXPECT_EQ(b.getLiftRange(), 9.0f);
    b.setLift(std::numeric_limits<float>::quiet_NaN(), 0.2f, 900.0f);
    EXPECT_EQ(b.getLiftBelow(), 24.0f);
    EXPECT_EQ(b.getLiftRatio(), 1.0f);
    EXPECT_EQ(b.getLiftRange(), 40.0f);
    b.setLift(900.0f, std::numeric_limits<float>::infinity(), -4.0f);
    EXPECT_EQ(b.getLiftBelow(), 80.0f);
    EXPECT_EQ(b.getLiftRatio(), 1.0f);
    EXPECT_EQ(b.getLiftRange(), 0.0f);
}

// New: Mode::Leveler. No ratio and no knee: the gain is 1 / (1 + level /
// threshold), 6 dB at the threshold and harder by itself the louder the
// input, so the output never passes the threshold; the range bounds it.
DSPARK_TEST(Compressor_leveler_mode_hardens_by_itself)
{
    using C = Compressor<float>;
    const auto gainAt = [](double levelDb, float ratio, float knee, float range) {
        return compHeldReduction(std::pow(10.0, (levelDb + 3.0103) / 20.0), 1.0, [&](C& c) {
            c.setDetector(C::DetectorType::Rms);
            c.setThreshold(-20.0f);
            c.setRatio(ratio);
            c.setKnee(knee);
            c.setRange(range);
            c.setMode(C::Mode::Leveler);
        });
    };
    const double levels[6] = { 0.0, -10.0, -20.0, -30.0, -40.0, -60.0 };
    double worst = 0.0, shown[6] = {};
    for (int i = 0; i < 6; ++i)
    {
        shown[i] = gainAt(levels[i], 4.0f, 0.0f, 100.0f);
        const double wanted = -20.0 * std::log10(1.0 + std::pow(10.0, (levels[i] + 20.0) / 20.0));
        worst = std::max(worst, std::abs(shown[i] - wanted));
        EXPECT_LT(levels[i] + shown[i], -20.0);                 // the output stays under the threshold
    }
    std::printf("  [leveler] gain at 0, -10, -20, -30, -40 and -60 dB under a threshold of -20: %.2f %.2f %.2f %.2f %.2f %.3f dB; farthest from 1 / (1 + level / threshold) %.3f dB\n",
                shown[0], shown[1], shown[2], shown[3], shown[4], shown[5], worst);
    EXPECT_LT(worst, 0.05);
    EXPECT_NEAR(shown[2], -6.02, 0.05);
    EXPECT_NEAR(gainAt(0.0, 20.0f, 12.0f, 100.0f), shown[0], 0.01);     // neither the ratio nor the knee counts
    EXPECT_NEAR(gainAt(0.0, 4.0f, 0.0f, 6.0f), -6.0, 0.02);             // the range bounds it

    // The static makeup counts its own curve: half of what a signal at full scale loses.
    {
        C c;
        c.setThreshold(-20.0f);
        c.setMode(C::Mode::Leveler);
        c.setAutoMakeup(C::AutoMakeupMode::Static);
        c.prepare(AudioSpec{ 48000.0, 512, 1 });
        const Vec in = makeSine(1000.0, 0.001, 48000.0, 48000);         // 40 dB under the threshold
        std::vector<Vec> x { in };
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c.processBlock(b); });
        const double change = toDb(toneAmplitude(x[0].data() + 38400, 9600, 1000.0, 48000.0) / 0.001);
        // 20.83 dB at full scale: +10.41; the curve itself takes 0.09 at this level.
        EXPECT_NEAR(change, 10.41 - 0.09, 0.15);
    }

    // In a loop the curve is the ratio's; one sample at a time is the block path.
    const Vec burst = compLowOverHigh(24000);
    const auto working = [](C::Mode mode, bool loop) {
        return [=](C& c) {
            c.setThreshold(-24.0f);
            c.setRatio(4.0f);
            c.setAttack(2.0f);
            c.setRelease(60.0f);
            c.setMode(mode);
            if (loop) c.setTopology(C::Topology::FeedBack);
        };
    };
    EXPECT_EQ(compFarthest(compRender(burst, working(C::Mode::Leveler, true)),
                           compRender(burst, working(C::Mode::Downward, true))), 0.0);
    const Vec block = compRender(burst, working(C::Mode::Leveler, false));
    EXPECT_GT(compFarthest(block, compRender(burst, working(C::Mode::Downward, false))), 1e-3);
    EXPECT_LT(compFarthest(block, compRenderSamples(burst, working(C::Mode::Leveler, false))), 1e-6);

    // The state keeps the mode; a mode that is not one is the last.
    C a;
    a.setMode(C::Mode::Leveler);
    const auto blob = a.getState();
    C b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.getState() == blob);
    a.setMode(static_cast<C::Mode>(77));
    EXPECT_TRUE(a.getState() == blob);
}

// New: the routes of a pair. The compressor works on the middle and the
// sides in place of left and right: on both, on one (the other passes
// untouched and in time), or on one with the gain the other asks for.
DSPARK_TEST(Compressor_routes_work_on_the_middle_and_the_sides)
{
    using C = Compressor<float>;
    const int n = 48000;
    const Vec mid = makeSine(1000.0, 0.4, 48000.0, n);              // 12 dB over the threshold: 9 dB of reduction
    const Vec side = makeSine(300.0, 0.05, 48000.0, n);             // under it
    const auto pairOf = [&](const Vec& m, const Vec& s) {
        std::vector<Vec> x { Vec(static_cast<size_t>(n)), Vec(static_cast<size_t>(n)) };
        for (int i = 0; i < n; ++i)
        {
            x[0][static_cast<size_t>(i)] = m[static_cast<size_t>(i)] + s[static_cast<size_t>(i)];
            x[1][static_cast<size_t>(i)] = m[static_cast<size_t>(i)] - s[static_cast<size_t>(i)];
        }
        return x;
    };
    const auto partOf = [&](const std::vector<Vec>& x, bool sides) {
        Vec v(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] = 0.5f * (sides ? x[0][static_cast<size_t>(i)] - x[1][static_cast<size_t>(i)]
                                                      : x[0][static_cast<size_t>(i)] + x[1][static_cast<size_t>(i)]);
        return v;
    };
    const auto late = [&](const Vec& y, const Vec& x, int by) {
        double worst = 0.0;
        for (int i = by; i < n; ++i)
            worst = std::max(worst, std::abs(static_cast<double>(y[static_cast<size_t>(i)]) - x[static_cast<size_t>(i - by)]));
        return worst;
    };
    const auto base = [](C& c) {
        c.setDetector(C::DetectorType::Hilbert);
        c.setThreshold(-20.0f);
        c.setRatio(4.0f);
        c.setAttack(1.0f);
        c.setRelease(100.0f);
    };
    const auto run = [&](std::vector<Vec> x, C::Route route, float lookaheadMs, float link, int* latency = nullptr) {
        auto c = std::make_unique<C>();
        base(*c);
        c->setRoute(route);
        c->setLookahead(lookaheadMs);
        c->setStereoLink(link);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c->processBlock(b); });
        if (latency != nullptr) *latency = c->getLatency();
        return x;
    };
    const auto levelDb = [&](const Vec& v, double hz, double reference) {
        return toDb(toneAmplitude(v.data() + n - 9600, 9600, hz, 48000.0) / reference);
    };

    // The middle alone: 9 dB off the middle, and the sides as they came.
    {
        const auto y = run(pairOf(mid, side), C::Route::Mid, 0.0f, 1.0f);
        const double untouched = late(partOf(y, true), side, 0);
        std::printf("  [routes] the middle alone: the middle changes %.2f dB and the sides come out within %.1e of what went in\n",
                    levelDb(partOf(y, false), 1000.0, 0.4), untouched);
        EXPECT_NEAR(levelDb(partOf(y, false), 1000.0, 0.4), -9.03, 0.1);
        EXPECT_LT(untouched, 1e-7);
    }
    // The sides alone, loud this time; and the part that is not treated waits for the other.
    {
        const auto y = run(pairOf(side, mid), C::Route::Side, 0.0f, 1.0f);
        EXPECT_NEAR(levelDb(partOf(y, true), 1000.0, 0.4), -9.03, 0.1);
        EXPECT_LT(late(partOf(y, false), side, 0), 1e-7);
        int latency = 0;
        const auto ahead = run(pairOf(mid, side), C::Route::Mid, 5.0f, 1.0f, &latency);
        EXPECT_EQ(latency, 240);
        EXPECT_LT(late(partOf(ahead, true), side, 240), 1e-7);
        EXPECT_NEAR(levelDb(partOf(ahead, false), 1000.0, 0.4), -9.03, 0.1);
    }
    // Both: each with its gain when they are not linked, one gain for the two when they are.
    {
        const auto apart = run(pairOf(mid, side), C::Route::MidSide, 0.0f, 0.0f);
        EXPECT_NEAR(levelDb(partOf(apart, false), 1000.0, 0.4), -9.03, 0.1);
        EXPECT_NEAR(levelDb(partOf(apart, true), 300.0, 0.05), 0.0, 0.02);
        const auto linked = run(pairOf(mid, side), C::Route::MidSide, 0.0f, 1.0f);
        EXPECT_NEAR(levelDb(partOf(linked, true), 300.0, 0.05), -9.03, 0.15);
    }
    // Crossed: the middle of the pair asks, and the sides get the gain.
    {
        const auto y = run(pairOf(mid, side), C::Route::MidKeysSide, 0.0f, 1.0f);
        EXPECT_NEAR(levelDb(partOf(y, true), 300.0, 0.05), -9.03, 0.1);
        EXPECT_LT(late(partOf(y, false), mid, 0), 1e-7);
        const auto other = run(pairOf(side, mid), C::Route::SideKeysMid, 0.0f, 1.0f);
        EXPECT_NEAR(levelDb(partOf(other, false), 300.0, 0.05), -9.03, 0.1);
        EXPECT_LT(late(partOf(other, true), mid, 0), 1e-7);
        // A middle that comes and goes: the sides follow it down and come back.
        Vec bursts = mid;
        for (int i = 0; i < n; ++i)
            if ((i % 24000) >= 12000) bursts[static_cast<size_t>(i)] = 0.0f;
        const auto moving = run(pairOf(bursts, side), C::Route::MidKeysSide, 0.0f, 1.0f);
        const Vec sides = partOf(moving, true);
        const double during = toDb(toneAmplitude(sides.data() + 24000 + 7200, 4800, 300.0, 48000.0) / 0.05);
        const double after = toDb(toneAmplitude(sides.data() + 36000 + 7200, 4800, 300.0, 48000.0) / 0.05);
        std::printf("  [routes] the middle keys the sides: the sides at %.2f dB while the middle sounds and %.2f dB once it has stopped\n", during, after);
        EXPECT_NEAR(during, -9.03, 0.2);
        EXPECT_GT(after, -2.5);                                 // 150 to 250 ms into a release of 100 ms
    }
    // No reduction: the pair comes back from its middle and its sides as it went.
    {
        auto c = std::make_unique<C>();
        c->setThreshold(0.0f);
        c->setRoute(C::Route::MidSide);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        const auto in = pairOf(mid, side);
        auto x = in;
        processInBlocks(x, 512, [&](AudioBufferView<float> b) { c->processBlock(b); });
        EXPECT_LT(std::max(compFarthest(x[0], in[0]), compFarthest(x[1], in[1])), 1e-7);
    }
    // A key given apart is split the same way; listened to, the part that is not treated is left out.
    {
        auto c = std::make_unique<C>();
        base(*c);
        c->setRoute(C::Route::Mid);
        c->prepare(AudioSpec{ 48000.0, 512, 2 });
        const Vec quiet = makeSine(1000.0, 0.01, 48000.0, n);
        auto x = pairOf(quiet, side);
        std::vector<Vec> key { mid, mid };                          // all middle
        std::vector<float*> pa(2), pk(2);
        for (int off = 0; off < n; off += 512)
        {
            const int m = std::min(512, n - off);
            for (size_t ch = 0; ch < 2; ++ch) { pa[ch] = x[ch].data() + off; pk[ch] = key[ch].data() + off; }
            AudioBufferView<float> va(pa.data(), 2, m), vk(pk.data(), 2, m);
            c->processBlock(va, vk);
        }
        EXPECT_NEAR(levelDb(partOf(x, false), 1000.0, 0.01), -9.03, 0.1);
        EXPECT_LT(late(partOf(x, true), side, 0), 1e-7);

        auto d = std::make_unique<C>();
        base(*d);
        d->setRoute(C::Route::Mid);
        d->setSidechainListen(true);
        d->prepare(AudioSpec{ 48000.0, 512, 2 });
        auto heard = pairOf(mid, side);
        processInBlocks(heard, 512, [&](AudioBufferView<float> b) { d->processBlock(b); });
        EXPECT_LT(compFarthest(heard[0], heard[1], 4800), 1e-7);    // the middle of the key, and nothing of the sides
        EXPECT_GT(toneAmplitude(heard[0].data() + n - 9600, 9600, 1000.0, 48000.0), 0.3);
    }
    // Any other width runs as left and right; the state keeps the route.
    {
        const auto wide = [&](C::Route route) {
            auto c = std::make_unique<C>();
            base(*c);
            c->setRoute(route);
            c->prepare(AudioSpec{ 48000.0, 512, 3 });
            std::vector<Vec> x { mid, side, mid };
            processInBlocks(x, 512, [&](AudioBufferView<float> b) { c->processBlock(b); });
            return x;
        };
        const auto a = wide(C::Route::Mid), b = wide(C::Route::Stereo);
        EXPECT_EQ(std::max({ compFarthest(a[0], b[0]), compFarthest(a[1], b[1]), compFarthest(a[2], b[2]) }), 0.0);
        C c;
        c.setRoute(C::Route::SideKeysMid);
        const auto blob = c.getState();
        C other;
        EXPECT_TRUE(other.setState(blob.data(), blob.size()));
        EXPECT_TRUE(other.getRoute() == C::Route::SideKeysMid);
        other.setRoute(static_cast<C::Route>(-3));
        EXPECT_TRUE(other.getRoute() == C::Route::Stereo);
    }
}

// The measures of the programme count in the knobs' own times. A reduction
// held for three release times lengthens the release as much with a release
// of 50 ms as with one of a second, and the emphasis of the attack overshoots
// as far with an attack of 2 ms as with one of 20 (the measures had fixed
// windows: 1.5 s and 30 ms, whatever the knobs said).
DSPARK_TEST(Compressor_programme_measures_count_in_the_knobs_own_times)
{
    using C = Compressor<float>;
    const auto stretch = [&](float releaseMs) {
        const double held = 3.0 * static_cast<double>(releaseMs) / 1000.0;
        const Vec in = compToneThenSilence(1000.0, 0.4, held, held + 12.0 * static_cast<double>(releaseMs) / 1000.0 + 0.5);
        const auto time = [&](float amount) {
            const auto trace = compTrace(in, [&](C& c) {
                c.setDetector(C::DetectorType::Hilbert);
                c.setThreshold(-20.0f);
                c.setRatio(4.0f);
                c.setAttack(1.0f);
                c.setRelease(releaseMs);
                c.setProgramRelease(amount);
            });
            return compReleaseMs(trace, static_cast<int>(held * 48000.0));
        };
        return time(1.0f) / time(0.0f);
    };
    const double quick = stretch(50.0f), usual = stretch(100.0f), slow = stretch(1000.0f);
    std::printf("  [windows] a reduction held for three release times lengthens the release %.3f times at 50 ms, %.3f at 100 ms and %.3f at 1 s\n",
                quick, usual, slow);
    EXPECT_GT(usual, 1.05);
    EXPECT_NEAR(quick / usual, 1.0, 0.1);
    EXPECT_NEAR(slow / usual, 1.0, 0.1);

    const auto overshoot = [&](float attackMs) {
        Vec in = makeSine(1000.0, 0.4, 48000.0, 48000 * 2);
        for (int i = 0; i < 4800; ++i) in[static_cast<size_t>(i)] = 0.0f;
        const auto deepest = [&](float emphasis) {
            const auto trace = compTrace(in, [&](C& c) {
                c.setDetector(C::DetectorType::Hilbert);
                c.setThreshold(-20.0f);
                c.setRatio(4.0f);
                c.setAttack(attackMs);
                c.setRelease(2000.0f);
                c.setAttackEmphasis(emphasis);
            });
            double d = 0.0;
            for (double v : trace) d = std::min(d, v);
            return d;
        };
        return deepest(0.0f) - deepest(1.0f);
    };
    const double sharp = overshoot(2.0f), broad = overshoot(20.0f);
    std::printf("  [windows] the emphasis of the attack goes %.2f dB past the plain attack at 2 ms and %.2f dB at 20 ms\n", sharp, broad);
    EXPECT_GT(sharp, 1.0);
    EXPECT_NEAR(sharp / broad, 1.0, 0.2);
}

// A band of the key's equalizer on the middle or on the sides of a pair. A
// low cut on the middle takes a centred low tone out of what the detector
// weighs and one on the sides leaves it there; for a tone that is only sides
// it is the other way round. On both channels it takes either.
DSPARK_TEST(Compressor_key_bands_sit_on_the_middle_or_on_the_sides)
{
    using C = Compressor<float>;
    const int n = 24000;
    const auto reduction = [&](bool wide, int place, int block) {
        auto c = std::make_unique<C>();
        c->setThreshold(-20.0f);
        c->setRatio(4.0f);
        c->setAttack(5.0f);
        c->setRelease(100.0f);
        if (place >= 0)
        {
            C::KeyBand cut;
            cut.shape = C::KeyShape::LowCut;
            cut.frequency = 500.0f;
            cut.slope = 24.0f;
            cut.placement = static_cast<C::KeyPlacement>(place);
            c->setSidechainBand(0, cut);
        }
        c->prepare(AudioSpec{ 48000.0, block, 2 });
        std::vector<Vec> x(2, Vec(static_cast<size_t>(n)));
        for (int i = 0; i < n; ++i)
        {
            const float v = static_cast<float>(0.5 * std::sin(2.0 * kPiD * 60.0 * i / 48000.0));
            x[0][static_cast<size_t>(i)] = v;
            x[1][static_cast<size_t>(i)] = wide ? -v : v;
        }
        double deepest = 0.0;
        int at = 0;
        processInBlocks(x, block, [&](AudioBufferView<float> b) {
            c->processBlock(b);
            at += b.getNumSamples();
            if (at > n - 4800) deepest = std::min(deepest, static_cast<double>(c->getGainReductionDb()));
        });
        return deepest;
    };
    const int stereo = static_cast<int>(C::KeyPlacement::Stereo), mid = static_cast<int>(C::KeyPlacement::Mid),
              side = static_cast<int>(C::KeyPlacement::Side);

    const double centred = reduction(false, -1, 512), wideTone = reduction(true, -1, 512);
    std::printf("  [key place] no band: centred %.3f dB, wide %.3f dB\n", centred, wideTone);
    EXPECT_LT(centred, -5.0);
    EXPECT_NEAR(wideTone, centred, 0.01);

    const double centredStereo = reduction(false, stereo, 512), centredMid = reduction(false, mid, 512),
                 centredSide = reduction(false, side, 512);
    const double wideStereo = reduction(true, stereo, 512), wideMid = reduction(true, mid, 512),
                 wideSide = reduction(true, side, 512);
    std::printf("  [key place] centred tone: cut on both %.3f, on the middle %.3f, on the sides %.3f dB\n",
                centredStereo, centredMid, centredSide);
    std::printf("  [key place] wide tone:    cut on both %.3f, on the middle %.3f, on the sides %.3f dB\n",
                wideStereo, wideMid, wideSide);
    EXPECT_GT(centredStereo, -0.05);
    EXPECT_GT(centredMid, -0.05);
    EXPECT_NEAR(centredSide, centred, 0.01);
    EXPECT_GT(wideStereo, -0.05);
    EXPECT_NEAR(wideMid, wideTone, 0.01);
    EXPECT_GT(wideSide, -0.05);

    // The size of the block is not in it.
    EXPECT_NEAR(reduction(true, mid, 37), wideMid, 0.01);
    EXPECT_NEAR(reduction(false, side, 1), centredSide, 0.01);
}
