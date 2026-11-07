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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
