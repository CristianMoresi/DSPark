// DSPark Tests - Effects Modulation
// Delay, Chorus, Phaser, Reverb, AlgorithmicReverb, Tremolo, Vibrato,
// RingModulator, FrequencyShifter, DeEsser

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Effects/Delay.h"
#include "../Effects/Chorus.h"
#include "../Effects/Phaser.h"
#include "../Effects/AlgorithmicReverb.h"
#include "../Effects/Tremolo.h"
#include "../Effects/Vibrato.h"
#include "../Effects/RingModulator.h"
#include "../Effects/FrequencyShifter.h"
#include "../Effects/DeEsser.h"
#include "../Effects/NoiseGenerator.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// Delay
// ============================================================================

DSPARK_TEST(Delay_exact_sample_delay)
{
    constexpr int delaySamples = 100;
    constexpr int bufLen = 512;
    constexpr float sr = 44100.0f;
    float delayMs = static_cast<float>(delaySamples) / sr * 1000.0f;

    auto s = spec(static_cast<double>(sr), bufLen, 1);
    Delay<float> delay;
    delay.prepare(s, 1.0);
    delay.setSmoother(Delay<float>::SmootherType::None);

    auto tb = makeBuffer(1, bufLen);
    tb.ch(0)[0] = 1.0f; // Impulse at sample 0, rest is silence

    delay.processBlock(tb.view(), delayMs, 0.0f);

    // Sample 0 should be zero (delayed)
    EXPECT_NEAR(tb.ch(0)[0], 0.0f, 1e-5f);

    // The impulse should appear at approximately sample 'delaySamples'
    float peak = 0.0f;
    int peakPos = 0;
    for (int i = 0; i < bufLen; ++i)
    {
        if (std::abs(tb.ch(0)[i]) > peak)
        {
            peak = std::abs(tb.ch(0)[i]);
            peakPos = i;
        }
    }
    EXPECT_GT(peak, 0.5f); // Impulse came through
    EXPECT_NEAR(peakPos, delaySamples, 2); // Within 2 samples of expected
}

DSPARK_TEST(Delay_feedback_decay)
{
    constexpr int bufLen = 4410; // 100ms at 44100
    auto s = spec(44100.0, bufLen, 1);
    Delay<float> delay;
    delay.prepare(s, 0.5);
    delay.setSmoother(Delay<float>::SmootherType::None);

    // Impulse at sample 0 with 10ms delay (441 samples) and 0.5 feedback
    auto tb = makeBuffer(1, bufLen);
    tb.ch(0)[0] = 1.0f;
    delay.processBlock(tb.view(), 10.0f, 0.5f);

    // Within this block: echoes at 441, 882, 1323, ... with decaying amplitude
    int delaySmp = 441;
    float prevPeak = 2.0f;
    bool decaying = true;
    for (int echo = 1; echo <= 3; ++echo)
    {
        int center = echo * delaySmp;
        if (center + 5 >= bufLen) break;
        float peak = 0.0f;
        for (int j = center - 5; j <= center + 5; ++j)
        {
            float a = std::abs(tb.ch(0)[j]);
            if (a > peak) peak = a;
        }
        if (peak >= prevPeak) decaying = false;
        prevPeak = peak;
    }
    EXPECT_TRUE(decaying);
    // First echo should be significant
    float firstEcho = 0.0f;
    for (int j = delaySmp - 5; j <= delaySmp + 5; ++j)
    {
        float a = std::abs(tb.ch(0)[j]);
        if (a > firstEcho) firstEcho = a;
    }
    EXPECT_GT(firstEcho, 0.3f);
}

DSPARK_TEST(Delay_silence)
{
    Delay<float> delay;
    delay.prepare(defaultSpec(), 0.5);
    delay.setDelayMs(50.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    delay.processBlock(tb.view(), 50.0f);
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

DSPARK_TEST(Delay_mix_ramp_is_channel_coherent)
{
    // mixWetToDry used to advance the shared mix smoother inside a
    // channel-outer loop: the ramp ran numChannels times too fast and each
    // channel got a DIFFERENT segment of it (channel 1 a whole block ahead),
    // tilting the stereo image on every mix change (measured |L-R| up to 0.149
    // on identical channel content). One smoother value per sample now.
    Delay<float> d;
    d.prepare(spec(44100.0, 512, 2), 0.5);
    d.setSmoother(Delay<float>::SmootherType::Linear);
    d.setSmoothingTime(40.0f);                 // 1764-sample ramp, longer than the block

    auto tb = makeStereoBuffer(512);
    for (int i = 0; i < 512; ++i) { tb.ch(0)[i] = 0.5f; tb.ch(1)[i] = 0.5f; }
    d.pushDryToWet(tb.view());
    d.processWet(50.0f, 0.0f);
    d.mixWetToDry(tb.view(), 0.0f);            // target 0 from initial 1: ramp active

    float maxLr = 0.0f;
    for (int i = 0; i < 512; ++i) maxLr = std::max(maxLr, std::abs(tb.ch(0)[i] - tb.ch(1)[i]));
    EXPECT_TRUE(maxLr == 0.0f);                // identical input -> identical mix per sample
}

DSPARK_TEST(Delay_wet_path_keeps_time_on_short_blocks)
{
    // processWet used to run the FULL wet-buffer capacity regardless of how
    // many samples the last pushDryToWet delivered, so with caller blocks
    // shorter than maxBlockSize the delay line advanced past the real stream:
    // an impulse echoed 44 samples late after just 8 blocks of 128 on a
    // 512-sample spec (and drifting further every block). It now processes
    // exactly the pushed length.
    constexpr int kMaxBlock = 512, kBlock = 128, kDelay = 300, kBlocks = 8;
    Delay<float> d;
    d.prepare(spec(44100.0, kMaxBlock, 1), 0.5);
    d.setSmoother(Delay<float>::SmootherType::None);
    const float delayMs = kDelay * 1000.0f / 44100.0f;

    std::vector<float> stream;
    auto tb = makeBuffer(1, kBlock);
    for (int b = 0; b < kBlocks; ++b)
    {
        tb.fillSilence();
        if (b == 0) tb.ch(0)[0] = 1.0f;        // impulse at stream sample 0
        d.pushDryToWet(tb.view());
        d.processWet(delayMs, 0.0f);
        auto wet = d.getWetView();
        stream.insert(stream.end(), wet.getChannel(0), wet.getChannel(0) + kBlock);
    }
    int argmax = 0; float peak = 0.0f;
    for (size_t i = 0; i < stream.size(); ++i)
        if (std::abs(stream[i]) > peak) { peak = std::abs(stream[i]); argmax = static_cast<int>(i); }
    EXPECT_GT(peak, 0.5f);
    EXPECT_NEAR(argmax, kDelay, 1);            // old header: lands at 344
}

DSPARK_TEST(Delay_pingpong_bounded_and_crossfeeds)
{
    // The ping-pong cross-feed used to skip the FeedbackMode nonlinearity
    // entirely: |feedback| >= 1 grew without bound (measured 3e17 after 200
    // blocks). It now passes through the same tanh (Analog) / +-2 clamp
    // (Clean) as the straight path. Also fixes the alternation contract:
    // an impulse fed to L only echoes L at d, R at 2d, L at 3d...
    constexpr int kBlock = 512, kDelaySmp = 441;
    Delay<float> d;
    d.prepare(spec(44100.0, kBlock, 2), 0.5);
    d.setSmoother(Delay<float>::SmootherType::None);
    d.setFeedbackMode(Delay<float>::FeedbackMode::Analog);
    const float delayMs = kDelaySmp * 1000.0f / 44100.0f;

    std::vector<float> streamL, streamR;
    auto tb = makeStereoBuffer(kBlock);
    for (int b = 0; b < 4; ++b)
    {
        tb.fillSilence();
        if (b == 0) tb.ch(0)[0] = 1.0f;        // impulse on LEFT only
        d.pushDryToWet(tb.view());
        d.processPingPong(delayMs, 0.6f);
        auto wet = d.getWetView();
        streamL.insert(streamL.end(), wet.getChannel(0), wet.getChannel(0) + kBlock);
        streamR.insert(streamR.end(), wet.getChannel(1), wet.getChannel(1) + kBlock);
    }
    auto peakNear = [](const std::vector<float>& s, int center) {
        float p = 0.0f;
        for (int j = std::max(0, center - 3); j <= center + 3; ++j)
            p = std::max(p, std::abs(s[static_cast<size_t>(j)]));
        return p;
    };
    EXPECT_GT(peakNear(streamL, kDelaySmp), 0.5f);       // 1st echo on L
    EXPECT_LT(peakNear(streamR, kDelaySmp), 0.05f);      // nothing on R yet
    EXPECT_GT(peakNear(streamR, 2 * kDelaySmp), 0.2f);   // cross-fed echo on R
    EXPECT_LT(peakNear(streamL, 2 * kDelaySmp), 0.05f);  // and not on L

    // Runaway guard: |feedback| > 1 in Analog mode must stay bounded.
    Delay<float> hot;
    hot.prepare(spec(44100.0, 256, 2), 0.5);
    hot.setSmoother(Delay<float>::SmootherType::None);
    auto tb2 = makeStereoBuffer(256);
    float maxAbs = 0.0f;
    bool finite = true;
    for (int b = 0; b < 200; ++b)
    {
        tb2.fillSilence();
        if (b == 0) tb2.ch(0)[0] = 1.0f;
        hot.pushDryToWet(tb2.view());
        hot.processPingPong(5.0f, 1.2f);
        auto wet = hot.getWetView();
        for (int i = 0; i < 256; ++i)
        {
            if (!std::isfinite(wet.getChannel(0)[i])) finite = false;
            maxAbs = std::max(maxAbs, std::abs(wet.getChannel(0)[i]));
        }
    }
    EXPECT_TRUE(finite);
    EXPECT_LT(maxAbs, 10.0f);                  // old header: ~3e17 and climbing
}

DSPARK_TEST(Delay_invalid_inputs_are_ignored)
{
    // NaN setters used to poison the delay buffer permanently (the NaN
    // recirculates through the feedback path; 100% of the output went
    // non-finite). Invalid prepare() and wild channel indices must also be
    // inert, and setDelaySamples before prepare() no longer hits
    // std::clamp(x, 0, -1) (undefined behaviour).
    const float nan = std::numeric_limits<float>::quiet_NaN();

    Delay<float> early;
    early.setDelaySamples(100.0f);             // before prepare: must be safe

    Delay<float> subject, twin;
    for (auto* d : { &subject, &twin })
    {
        d->prepare(spec(44100.0, 256, 1), 0.5);
        d->setSmoother(Delay<float>::SmootherType::None);
        d->setDelayMs(10.0f);
        d->setFeedback(0.5f);
    }
    subject.setFeedback(nan);
    subject.setDelaySamples(nan);
    subject.setFeedbackLpHz(nan);
    subject.setFeedbackHpHz(nan);
    subject.setSmoothingTime(nan);
    AudioSpec bad;
    bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
    bad.maxBlockSize = 0; bad.numChannels = -3;
    subject.prepare(bad, 0.5);
    subject.prepare(spec(44100.0, 256, 1), std::numeric_limits<double>::quiet_NaN());
    const int cap = subject.getMaxDelaySamples();
    subject.advanceWriteIndex(99);
    subject.advanceWriteIndex(-7);
    EXPECT_EQ(subject.getMaxDelaySamples(), cap);

    auto ta = makeBuffer(1, 256);
    auto tb = makeBuffer(1, 256);
    int nonFinite = 0; float maxDiff = 0.0f;
    for (int blk = 0; blk < 10; ++blk)
    {
        ta.fillSilence(); tb.fillSilence();
        if (blk == 0) { ta.ch(0)[0] = 1.0f; tb.ch(0)[0] = 1.0f; }
        for (int i = 0; i < 256; ++i)
        {
            ta.ch(0)[i] = subject.processSample(0, ta.ch(0)[i]);
            tb.ch(0)[i] = twin.processSample(0, tb.ch(0)[i]);
        }
        for (int i = 0; i < 256; ++i)
        {
            if (!std::isfinite(ta.ch(0)[i])) ++nonFinite;
            maxDiff = std::max(maxDiff, std::abs(ta.ch(0)[i] - tb.ch(0)[i]));
        }
    }
    EXPECT_EQ(nonFinite, 0);
    EXPECT_TRUE(maxDiff == 0.0f);              // every invalid edit fully inert
}

// ============================================================================
// Chorus
// ============================================================================

DSPARK_TEST(Chorus_modifies_signal)
{
    Chorus<float> chorus;
    chorus.prepare(defaultSpec());
    chorus.setRate(1.0f);
    chorus.setDepthMs(5.0f);
    chorus.setMix(1.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);

    std::vector<float> original(tb.ch(0), tb.ch(0) + 4096);
    chorus.processBlock(tb.view());

    // Output should differ from input (modulated)
    bool differs = false;
    for (int i = 512; i < 4096; ++i)
    {
        if (std::abs(tb.ch(0)[i] - original[i]) > 0.01f)
        {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs);
}

DSPARK_TEST(Chorus_no_NaN)
{
    Chorus<float> chorus;
    chorus.prepare(defaultSpec());
    chorus.setRate(5.0f);
    chorus.setDepthMs(10.0f);
    chorus.setFeedback(0.5f);

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    chorus.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);
}

DSPARK_TEST(Chorus_silence)
{
    Chorus<float> chorus;
    chorus.prepare(defaultSpec());

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    chorus.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-6f);
}

DSPARK_TEST(Chorus_invalid_inputs_are_ignored)
{
    // Every float setter passed NaN through std::clamp into the modulated
    // delay (or the feedback loop, which recirculates it forever): measured
    // 4096/4096 non-finite samples for depth/center/feedback and 4095 for
    // rate (through the LFO frequency). An invalid prepare() corrupted the
    // delay sizing and smoothing state (0.45 divergence). All inert now:
    // bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Chorus<float> subject, twin;
    auto setup = [](Chorus<float>& c) {
        c.prepare(spec(48000.0, 512, 2));
        c.setMix(0.5f); c.setDepthMs(5.0f); c.setCenterDelay(7.0f);
        c.setFeedback(0.4f); c.setRate(1.5f); c.setStereoSpread(0.5f);
    };
    setup(subject); setup(twin);

    subject.setRate(nan);         subject.setRate(inf);
    subject.setDepthMs(nan);      subject.setCenterDelay(nan);
    subject.setFeedback(nan);     subject.setMix(nan);
    subject.setStereoSpread(nan);
    subject.setModWaveform(static_cast<Oscillator<float>::Waveform>(99)); // clamps
    subject.setModWaveform(Oscillator<float>::Waveform::Sine);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    subject.prepare(spec(48000.0, 0, 2));

    auto ta = makeBuffer(2, 4096);
    auto tb = makeBuffer(2, 4096);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
        {
            const float v = 0.4f * std::sin(6.2831853f * (ch == 0 ? 440.0f : 620.0f) * i / 48000.0f);
            ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
        }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 4096);
    EXPECT_NO_NAN(ta.ch(1), 4096);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
}

namespace
{
// Pitch-glide duration (ms) of a 15 ms -> 5 ms center-delay step: the
// shrinking delay Doppler-shifts a 1 kHz sine, so the upward zero-crossing
// spacing deviates from nominal until the parameter smoother settles.
double chorusGlideMs(double fs)
{
    Chorus<float> ch;
    ch.prepare(spec(fs, 512, 1));
    ch.setMix(1.0f);
    ch.setVoices(1);
    ch.setDepthMs(0.0f);
    ch.setFeedback(0.0f);
    ch.setStereoSpread(0.0f);
    ch.setCenterDelay(15.0f);

    const int B = 512;
    std::vector<float> buf(B);
    long g = 0; // continuous sine phase across warmup + measurement
    auto runBlock = [&](std::vector<float>* sink) {
        for (int i = 0; i < B; ++i)
        {
            buf[static_cast<size_t>(i)] =
                0.5f * std::sin(static_cast<float>(2.0 * 3.14159265358979 * 1000.0 * g / fs));
            ++g;
        }
        float* p[1] = { buf.data() };
        ch.processBlock(AudioBufferView<float>(p, 1, B));
        if (sink)
            for (int i = 0; i < B; ++i) sink->push_back(buf[static_cast<size_t>(i)]);
    };

    for (int blk = 0; blk < static_cast<int>(fs / B); ++blk)
        runBlock(nullptr);                     // settle mixer + delay at 15 ms

    ch.setCenterDelay(5.0f);                   // the step (t = 0)
    std::vector<float> out;
    const int total = static_cast<int>(fs * 0.15); // 150 ms
    for (int blk = 0; blk < total / B; ++blk)
        runBlock(&out);

    const double nominal = fs / 1000.0;
    double lastBad = 0.0;
    int prev = -1;
    for (int i = 1; i < static_cast<int>(out.size()); ++i)
    {
        if (out[static_cast<size_t>(i - 1)] <= 0.0f && out[static_cast<size_t>(i)] > 0.0f)
        {
            if (prev >= 0)
            {
                const double spacing = i - prev;
                if (std::abs(spacing - nominal) > 0.05 * nominal)
                    lastBad = i;
            }
            prev = i;
        }
    }
    return 1000.0 * lastBad / fs;
}
} // namespace

DSPARK_TEST(Chorus_smoothing_is_rate_invariant)
{
    // The parameter smoothing used a fixed 0.005 per-sample coefficient, so a
    // center-delay glide lasted half as long (in ms) at 96 kHz as at 48 kHz
    // (measured: 16.2 ms vs 9.8 ms). The coefficient now derives from the
    // sample rate (240/fs, bit-identical at 48 kHz) and the glide duration is
    // rate-invariant.
    const double s48 = chorusGlideMs(48000.0);
    const double s96 = chorusGlideMs(96000.0);
    EXPECT_GT(s48, 8.0);   // the glide is real and measurable
    EXPECT_LT(s48, 40.0);
    EXPECT_GT(s96 / s48, 0.8);  // the old fixed coefficient measured 0.60
    EXPECT_LT(s96 / s48, 1.25);
}

// ============================================================================
// Phaser
// ============================================================================

DSPARK_TEST(Phaser_invalid_inputs_are_ignored)
{
    // Every float setter passed NaN through std::clamp (or max with NaN in
    // the first argument) into the allpass coefficients, permanently
    // poisoning the chain state (measured: 4095-4096/4096 non-finite samples
    // for rate/depth/feedback/center/range). An invalid prepare() corrupted
    // the smoothing coefficient (0.31 divergence), and setCenterFrequency()
    // before prepare() stored maxFreq = 0 (log(0) sweep). All inert now:
    // bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Phaser<float> subject, twin;
    auto setup = [](Phaser<float>& p) {
        p.prepare(spec(48000.0, 512, 2));
        p.setRate(0.8f); p.setDepth(0.8f); p.setMix(0.5f);
        p.setFeedback(0.6f); p.setStages(6); p.setStereoSpread(0.5f);
    };
    setup(subject); setup(twin);

    subject.setRate(nan);            subject.setRate(inf);
    subject.setDepth(nan);           subject.setMix(nan);
    subject.setFeedback(nan);        subject.setStereoSpread(nan);
    subject.setCenterFrequency(nan); subject.setCenterFrequency(-100.0f);
    subject.setFrequencyRange(nan, nan);
    subject.setLfoWaveform(static_cast<Oscillator<float>::Waveform>(99)); // clamps
    subject.setLfoWaveform(Oscillator<float>::Waveform::Sine);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    subject.prepare(spec(48000.0, 0, 2));

    // The serialized state never carries the poison.
    auto blob = subject.getState();
    StateReader r(blob.data(), blob.size());
    EXPECT_NEAR(r.read("rate", 999.0f), 0.8f, 1e-6f);
    EXPECT_NEAR(r.read("minFreq", 999.0f), 200.0f, 1e-3f);
    EXPECT_NEAR(r.read("maxFreq", 999.0f), 6000.0f, 1e-3f);

    auto ta = makeBuffer(2, 4096);
    auto tb = makeBuffer(2, 4096);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
        {
            const float v = 0.4f * std::sin(6.2831853f * (ch == 0 ? 440.0f : 620.0f) * i / 48000.0f);
            ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
        }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 4096);
    EXPECT_NO_NAN(ta.ch(1), 4096);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin

    // Pre-prepare setCenterFrequency stores a sane range (it used to clamp
    // maxFreq against the unprepared 0 Hz sample rate).
    Phaser<float> fresh;
    fresh.setCenterFrequency(1000.0f);
    auto blob2 = fresh.getState();
    StateReader r2(blob2.data(), blob2.size());
    EXPECT_GT(r2.read("maxFreq", 0.0f), r2.read("minFreq", 0.0f));
    EXPECT_GT(r2.read("minFreq", 0.0f), 19.0f);
}

DSPARK_TEST(Phaser_stage_increase_is_clean)
{
    // Stages re-activated by a live increase used to replay whatever history
    // they held when last active: raising 2 -> 12 during a 0.001-level
    // passage after a loud one produced a full-scale click (measured: max
    // sample jump 0.944 vs 0.003 with the state cleared on activation).
    Phaser<float> ph;
    ph.prepare(spec(48000.0, 512, 2));
    ph.setMix(1.0f); ph.setStages(12); ph.setRate(0.3f);

    auto loud = makeBuffer(2, 512);
    for (int blk = 0; blk < 8; ++blk)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                loud.ch(ch)[i] = 0.9f * std::sin(6.2831853f * 220.0f * (blk * 512 + i) / 48000.0f);
        ph.processBlock(loud.view());
    }
    ph.setStages(2);
    for (int blk = 0; blk < 40; ++blk)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                loud.ch(ch)[i] = 0.001f * std::sin(6.2831853f * 220.0f * i / 48000.0f);
        ph.processBlock(loud.view());
    }
    ph.setStages(12); // re-activation must start from cleared state

    float maxJump = 0.0f, prev = 0.0f;
    for (int blk = 0; blk < 2; ++blk)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                loud.ch(ch)[i] = 0.001f * std::sin(6.2831853f * 220.0f * i / 48000.0f);
        ph.processBlock(loud.view());
        for (int i = 0; i < 512; ++i)
        {
            maxJump = std::max(maxJump, std::abs(loud.ch(0)[i] - prev));
            prev = loud.ch(0)[i];
        }
    }
    EXPECT_LT(maxJump, 0.01f); // the old stale-history click measured 0.94
}

DSPARK_TEST(Phaser_modifies_signal)
{
    Phaser<float> phaser;
    phaser.prepare(defaultSpec());
    phaser.setRate(2.0f);
    phaser.setDepth(1.0f);
    phaser.setMix(1.0f);
    phaser.setStages(4);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);

    std::vector<float> original(tb.ch(0), tb.ch(0) + 4096);
    phaser.processBlock(tb.view());

    bool differs = false;
    for (int i = 256; i < 4096; ++i)
    {
        if (std::abs(tb.ch(0)[i] - original[i]) > 0.01f)
        {
            differs = true;
            break;
        }
    }
    EXPECT_TRUE(differs);
}

DSPARK_TEST(Phaser_stable_with_high_feedback)
{
    Phaser<float> phaser;
    phaser.prepare(defaultSpec());
    phaser.setRate(1.0f);
    phaser.setDepth(1.0f);
    phaser.setFeedback(0.95f);
    phaser.setStages(8);

    auto tb = makeStereoBuffer(8192);
    tb.fillSine(440.0f, 44100.0f);
    phaser.processBlock(tb.view());

    EXPECT_NO_NAN(tb.ch(0), 8192);
    float peak = measurePeak(tb.ch(0), 8192);
    EXPECT_LT(peak, 10.0f); // Should not explode
}

DSPARK_TEST(Phaser_silence)
{
    Phaser<float> phaser;
    phaser.prepare(defaultSpec());

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    phaser.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-6f);
}

// ============================================================================
// AlgorithmicReverb
// ============================================================================

DSPARK_TEST(AlgoReverb_impulse_produces_tail)
{
    AlgorithmicReverb<float> reverb;
    reverb.prepare(defaultSpec());
    reverb.setDecay(1.0f);
    reverb.setMix(1.0f);

    // Feed impulse
    auto tb = makeStereoBuffer(512);
    tb.fillImpulse();
    reverb.processBlock(tb.view());

    float peakFirst = measurePeak(tb.ch(0), 512);

    // Feed silence - should still produce output (reverb tail)
    tb.fillSilence();
    reverb.processBlock(tb.view());
    float peakTail = measurePeak(tb.ch(0), 512);

    EXPECT_GT(peakTail, 0.001f); // Tail should be audible
    EXPECT_LT(peakTail, peakFirst + 0.1f); // Tail should be decaying
}

DSPARK_TEST(AlgoReverb_different_types_differ)
{
    AlgorithmicReverb<float> r1, r2;
    r1.prepare(defaultSpec());
    r2.prepare(defaultSpec());

    r1.setType(AlgorithmicReverb<float>::Type::Room);
    r2.setType(AlgorithmicReverb<float>::Type::Cathedral);
    r1.setDecay(1.0f); r1.setMix(1.0f);
    r2.setDecay(1.0f); r2.setMix(1.0f);

    auto tb1 = makeStereoBuffer(4096);
    auto tb2 = makeStereoBuffer(4096);
    tb1.fillImpulse();
    tb2.fillImpulse();

    r1.processBlock(tb1.view());
    r2.processBlock(tb2.view());

    // RMS should differ between types
    float rms1 = measureRMS(tb1.ch(0), 4096);
    float rms2 = measureRMS(tb2.ch(0), 4096);
    EXPECT_NE(rms1, rms2);
}

DSPARK_TEST(AlgoReverb_silence)
{
    AlgorithmicReverb<float> reverb;
    reverb.prepare(defaultSpec());
    reverb.setDecay(0.5f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    reverb.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-6f);
}

DSPARK_TEST(AlgoReverb_no_NaN)
{
    AlgorithmicReverb<float> reverb;
    reverb.prepare(defaultSpec());
    reverb.setDecay(2.0f);
    reverb.setMix(0.5f);

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    reverb.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);
}

// ============================================================================
// Tremolo
// ============================================================================

DSPARK_TEST(Tremolo_modulates_amplitude)
{
    Tremolo<float> trem;
    trem.prepare(defaultSpec());
    trem.setRate(4.0f);
    trem.setDepth(1.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);

    // Store original peak
    float origPeak = 0.0f;
    for (int i = 0; i < 4096; ++i)
        origPeak = std::max(origPeak, std::abs(tb.ch(0)[i]));

    trem.processBlock(tb.view());

    // With full depth, some samples should be near zero (when LFO dips)
    float minAbs = 1e30f;
    float maxAbs = 0.0f;
    for (int i = 256; i < 4096; ++i) // skip initial transient
    {
        float a = std::abs(tb.ch(0)[i]);
        minAbs = std::min(minAbs, a);
        maxAbs = std::max(maxAbs, a);
    }
    EXPECT_LT(minAbs, 0.1f);    // Should reach near silence
    EXPECT_GT(maxAbs, 0.5f);    // Should still have loud parts
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(Tremolo_stereo_offset)
{
    Tremolo<float> trem;
    trem.setStereo(true);
    trem.prepare(defaultSpec());
    trem.setRate(2.0f);
    trem.setDepth(1.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillDC(1.0f);
    trem.processBlock(tb.view());

    // L and R should be different (180-degree phase offset)
    bool different = false;
    for (int i = 100; i < 4096; ++i)
    {
        if (std::abs(tb.ch(0)[i] - tb.ch(1)[i]) > 0.1f)
        {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

DSPARK_TEST(Tremolo_zero_depth_passthrough)
{
    Tremolo<float> trem;
    trem.prepare(defaultSpec());
    trem.setDepth(0.0f);

    // Let the depth smoother settle to 0 so we test true zero-depth passthrough
    // (the depth parameter is de-zippered, not applied instantaneously).
    for (int w = 0; w < 8; ++w)
    {
        auto wb = makeMonoBuffer(512);
        wb.fillSine(440.0f, 44100.0f);
        trem.processBlock(wb.view());
    }

    auto tb = makeMonoBuffer(512);
    tb.fillSine(440.0f, 44100.0f);
    auto ref = makeMonoBuffer(512);
    ref.fillSine(440.0f, 44100.0f);

    trem.processBlock(tb.view());

    for (int i = 0; i < 512; ++i)
        EXPECT_NEAR(tb.ch(0)[i], ref.ch(0)[i], 1e-5f);
}

// ============================================================================
// Vibrato
// ============================================================================

DSPARK_TEST(Vibrato_alters_signal)
{
    Vibrato<float> vib;
    vib.prepare(defaultSpec());
    vib.setRate(5.0f);
    vib.setDepth(1.0f);

    auto tb = makeMonoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);
    auto ref = makeMonoBuffer(4096);
    ref.fillSine(440.0f, 44100.0f);

    vib.processBlock(tb.view());

    // Output should differ from input (pitch is modulated)
    float diff = 0.0f;
    for (int i = 256; i < 4096; ++i)
        diff += std::abs(tb.ch(0)[i] - ref.ch(0)[i]);
    EXPECT_GT(diff, 1.0f);
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(Vibrato_bounded_output)
{
    Vibrato<float> vib;
    vib.prepare(defaultSpec());
    vib.setRate(6.0f);
    vib.setDepth(2.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);
    vib.processBlock(tb.view());

    for (int i = 0; i < 4096; ++i)
        EXPECT_TRUE(std::abs(tb.ch(0)[i]) <= 2.0f);
}

DSPARK_TEST(Vibrato_invalid_inputs_are_ignored)
{
    // setRate/setDepth(NaN) went through std::max/std::clamp into the delay
    // computation (measured: 4608 non-finite samples, NaN getters and state
    // blob), setRate(+Inf) slipped past the Phasor's NaN-only guard and made
    // wrapUnit(Inf) return NaN (a continuous storm, plus a permanent LFO
    // phase offset after recovery), and an invalid prepare() either poisoned
    // the rings (NaN rate) or threw from resize (negative channels). All
    // inert now: bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Vibrato<float> subject, twin;
    subject.prepare(spec(48000.0, 512, 2));
    twin.prepare(spec(48000.0, 512, 2));
    subject.setRate(5.0f);     twin.setRate(5.0f);
    subject.setDepth(1.0f);    twin.setDepth(1.0f);
    subject.setModRate(2.0f);  twin.setModRate(2.0f);
    subject.setModDepth(0.3f); twin.setModDepth(0.3f);

    auto ta = makeBuffer(2, 512);
    auto tb = makeBuffer(2, 512);
    float maxDiff = 0.0f;
    for (int blk = 0; blk < 10; ++blk)
    {
        if (blk == 2)
        {
            subject.setRate(nan);     subject.setRate(inf);
            subject.setDepth(nan);    subject.setDepth(inf);
            subject.setModRate(nan);  subject.setModRate(inf);
            subject.setModDepth(nan); subject.setModDepth(inf);
        }
        if (blk == 5)
        {
            subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
            subject.prepare(spec(48000.0, 512, -3)); // used to throw from resize
        }
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.5f * std::sin(6.2831853f * 440.0f * (blk * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
            }
        subject.processBlock(ta.view());
        twin.processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    }
    EXPECT_NO_NAN(ta.ch(0), 512);
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
    EXPECT_NEAR(subject.getRate(), 5.0f, 1e-6f);
    EXPECT_NEAR(subject.getDepth(), 1.0f, 1e-6f);
    EXPECT_NEAR(subject.getModRate(), 2.0f, 1e-6f);
    EXPECT_NEAR(subject.getModDepth(), 0.3f, 1e-6f);
}

DSPARK_TEST(Vibrato_worst_case_depth_is_not_clamped)
{
    // prepare() sized the ring for deviation + 128 but the sweep requests
    // centre + deviation = 2 * deviation + 4. At 44.1k with the worst case
    // the sizing claims to cover (4 st at the 0.1 Hz floor) the top HALF of
    // the LFO cycle parked at the capacity clamp: pitch frozen, with a 4 st
    // pitch jump on clamp entry/exit (measured max delay 16380 = old cap,
    // theory 32440). An impulse train reveals the instantaneous delay.
    Vibrato<float> vib;
    vib.setRate(0.1f);
    vib.setDepth(4.0f);
    vib.prepare(spec(44100.0, 512, 1));

    const int total = 1292 * 512; // ~15 s at 44.1k, whole blocks
    const int spacing = 36000; // > max possible delay: windows never mix impulses
    std::vector<float> out(static_cast<size_t>(total), 0.0f);
    auto blk = makeBuffer(1, 512);
    for (int base = 0; base < total; base += 512)
    {
        for (int i = 0; i < 512; ++i)
            blk.ch(0)[i] = ((base + i) % spacing == 0) ? 1.0f : 0.0f;
        vib.processBlock(blk.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(base + i)] = blk.ch(0)[i];
    }

    float maxDelay = 0.0f, minDelay = 1e9f;
    int peaks = 0;
    for (int k = spacing; k + spacing < total; k += spacing)
    {
        if (k < 50000) continue; // ring fully written
        int best = -1; float bestAbs = 0.0f;
        for (int m = k + 1; m < k + spacing; ++m)
            if (std::abs(out[static_cast<size_t>(m)]) > bestAbs)
            {
                bestAbs = std::abs(out[static_cast<size_t>(m)]);
                best = m;
            }
        if (best < 0 || bestAbs < 0.05f) continue; // impulse skipped (falling slope)
        const float d = static_cast<float>(best - k);
        maxDelay = std::max(maxDelay, d);
        minDelay = std::min(minDelay, d);
        ++peaks;
    }
    EXPECT_GT(peaks, 7);           // metric sanity
    EXPECT_GT(maxDelay, 30000.0f); // full sweep reached (old build parks at 16380)
    EXPECT_TRUE(minDelay < 2000.0f); // and the trough still comes back down
}

DSPARK_TEST(Vibrato_mod_depth_change_is_click_free)
{
    // The FM depth used to be applied per block WITHOUT smoothing: it scales
    // the instantaneous rate, and deviation and centre follow that rate, so a
    // step jumped the delay-line read position by hundreds of samples.
    // Measured max |x[n]-x[n-1]| around a 0.45 -> 0.15 step: 1.18 (a
    // full-scale click, 47x the settled regime's 0.025). Smoothed now.
    Vibrato<float> vib;
    vib.setRate(5.0f);
    vib.setDepth(1.5f);
    vib.setModRate(3.0f);
    vib.setModDepth(0.15f);
    vib.prepare(spec(48000.0, 512, 1));

    const int nBlocks = 32, N = 512;
    std::vector<float> out;
    out.reserve(static_cast<size_t>(nBlocks) * N);
    auto blk = makeBuffer(1, N);
    for (int b = 0; b < nBlocks; ++b)
    {
        if (b == 20) vib.setModDepth(0.45f); // step up   (mod LFO at sin = -0.77)
        if (b == 25) vib.setModDepth(0.15f); // step down (mod LFO at sin = -0.95)
        for (int i = 0; i < N; ++i)
            blk.ch(0)[i] = 0.8f * std::sin(6.2831853f * 220.0f * (b * N + i) / 48000.0f);
        vib.processBlock(blk.view());
        out.insert(out.end(), blk.ch(0), blk.ch(0) + N);
    }

    auto maxStep = [&](int from, int to) {
        float m = 0.0f;
        for (int i = std::max(1, from); i < to; ++i)
            m = std::max(m, std::abs(out[static_cast<size_t>(i)] - out[static_cast<size_t>(i - 1)]));
        return m;
    };
    const float base = maxStep(8 * N, 20 * N);      // settled FM 0.15 regime
    const float up   = maxStep(20 * N - 4, 21 * N); // around the 0.15 -> 0.45 step
    const float down = maxStep(25 * N - 4, 26 * N); // around the 0.45 -> 0.15 step
    EXPECT_TRUE(base < 0.1f);  // regime sanity (measured 0.025)
    EXPECT_TRUE(up < 0.12f);   // old build: 0.24
    EXPECT_TRUE(down < 0.15f); // old build: 1.18
}

// ============================================================================
// RingModulator
// ============================================================================

DSPARK_TEST(RingModulator_produces_sum_diff)
{
    RingModulator<float> ring;
    ring.prepare(defaultSpec());
    ring.setFrequency(100.0f);
    ring.setMix(1.0f);

    auto tb = makeMonoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f, 0.5f);
    ring.processBlock(tb.view());

    // Output should be nonzero and different from pure sine
    float energy = 0.0f;
    for (int i = 0; i < 4096; ++i)
        energy += tb.ch(0)[i] * tb.ch(0)[i];
    EXPECT_GT(energy, 10.0f);
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(RingModulator_invalid_inputs_are_ignored)
{
    // setFrequency had no validation at all: a NaN parked in currentFreq_
    // was IRRECOVERABLE (the ramp accumulates, never reassigns), leaving the
    // carrier deaf to every future setFrequency (measured: 293 Hz forever
    // with getFrequency() reporting 1000), +Inf slipped past the Phasor's
    // NaN-only guard (2048 non-finite samples), setMix(NaN) was a permanent
    // NaN output (4096 measured, surviving "recovery"), setSoar(NaN)
    // poisoned GM mode, and prepare(ch=-3) silently killed the effect
    // (pass-through, diff 1.08 vs twin). All inert now.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    RingModulator<float> subject, twin;
    subject.prepare(spec(48000.0, 512, 2));
    twin.prepare(spec(48000.0, 512, 2));
    subject.setFrequency(500.0f); twin.setFrequency(500.0f);
    subject.setMix(0.9f);         twin.setMix(0.9f);
    subject.setSoar(0.05f);       twin.setSoar(0.05f);
    subject.setMode(RingModulator<float>::Mode::GeometricMean);
    twin.setMode(RingModulator<float>::Mode::GeometricMean);

    auto ta = makeBuffer(2, 512);
    auto tb = makeBuffer(2, 512);
    float maxDiff = 0.0f;
    for (int blk = 0; blk < 10; ++blk)
    {
        if (blk == 2)
        {
            subject.setFrequency(nan); subject.setFrequency(inf);
            subject.setMix(nan);       subject.setMix(inf);
            subject.setSoar(nan);      subject.setSoar(inf);
        }
        if (blk == 5)
        {
            subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
            subject.prepare(spec(48000.0, 512, -3)); // old: effect silently died
        }
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.5f * std::sin(6.2831853f * 440.0f * (blk * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
            }
        subject.processBlock(ta.view());
        twin.processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    }
    EXPECT_NO_NAN(ta.ch(0), 512);
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
    EXPECT_NEAR(subject.getFrequency(), 500.0f, 1e-6f);
    EXPECT_NEAR(subject.getMix(), 0.9f, 1e-6f);
    EXPECT_NEAR(subject.getSoar(), 0.05f, 1e-6f);

    // Wild enum: the old build kept 99 in the atomic (getter lied) while the
    // audio ran the Classic branch. Clamped now: getter and audio agree.
    subject.setMode(static_cast<RingModulator<float>::Mode>(99));
    EXPECT_EQ(static_cast<int>(subject.getMode()), 1); // GeometricMean (last member)
}

DSPARK_TEST(RingModulator_geometric_mean_mode_works)
{
    // The GeometricMean mode had zero coverage. With a DC 1.0 probe at
    // mix 1, a Classic-mode twin outputs the raw carrier c, so the GM output
    // must equal sqrt(|c|) * sign(c) sample by sample (soar 0). The soar
    // floor must fill the carrier's zero-crossing notch, and the amplitude
    // scaling must keep silence exactly silent (no DC pumping).
    RingModulator<float> gm, cl;
    gm.setMode(RingModulator<float>::Mode::GeometricMean);
    gm.setFrequency(440.0f); gm.setMix(1.0f); gm.setSoar(0.0f);
    cl.setMode(RingModulator<float>::Mode::Classic);
    cl.setFrequency(440.0f); cl.setMix(1.0f);
    gm.prepare(spec(48000.0, 512, 1));
    cl.prepare(spec(48000.0, 512, 1));

    auto ga = makeBuffer(1, 512);
    auto ca = makeBuffer(1, 512);
    float gmMin = 1e9f;
    for (int blk = 0; blk < 4; ++blk)
    {
        for (int i = 0; i < 512; ++i) { ga.ch(0)[i] = 1.0f; ca.ch(0)[i] = 1.0f; }
        gm.processBlock(ga.view());
        cl.processBlock(ca.view());
        for (int i = 0; i < 512; ++i)
        {
            // Compare in the squared domain: y = sqrt(|c|)*sign(c) implies
            // y*|y| == c, and unlike sqrt (infinite slope at 0) the square is
            // well conditioned at the carrier's zero crossings, where the
            // half-ulp noise of the twin's mix arithmetic would otherwise
            // blow up to ~5e-5 through the sqrt.
            const float c = ca.ch(0)[i]; // Classic with DC 1.0 emits the carrier
            const float y = ga.ch(0)[i];
            EXPECT_NEAR(y * std::abs(y), c, 1e-5f);
            gmMin = std::min(gmMin, std::abs(y));
        }
    }

    // Soar floors the zero-crossing notch: min |out| rises by ~sqrt(soar).
    RingModulator<float> gs;
    gs.setMode(RingModulator<float>::Mode::GeometricMean);
    gs.setFrequency(440.0f); gs.setMix(1.0f); gs.setSoar(0.09f);
    gs.prepare(spec(48000.0, 512, 1));
    float soarMin = 1e9f;
    for (int blk = 0; blk < 4; ++blk)
    {
        for (int i = 0; i < 512; ++i) ga.ch(0)[i] = 1.0f;
        gs.processBlock(ga.view());
        for (int i = 0; i < 512; ++i)
            soarMin = std::min(soarMin, std::abs(ga.ch(0)[i]));
    }
    EXPECT_GT(soarMin, gmMin + 0.1f); // notch filled (sqrt(0.09) = 0.3 floor)

    // Silence stays exactly silent even with soar (the amplitude scaling).
    for (int i = 0; i < 512; ++i) ga.ch(0)[i] = 0.0f;
    gs.processBlock(ga.view());
    for (int i = 0; i < 512; ++i)
        EXPECT_TRUE(ga.ch(0)[i] == 0.0f);
}

DSPARK_TEST(RingModulator_silence_in_silence_out)
{
    RingModulator<float> ring;
    ring.prepare(defaultSpec());
    ring.setFrequency(1000.0f);

    auto tb = makeMonoBuffer(512);
    tb.fillSilence();
    ring.processBlock(tb.view());

    for (int i = 0; i < 512; ++i)
        EXPECT_NEAR(tb.ch(0)[i], 0.0f, 1e-6f);
}

// ============================================================================
// FrequencyShifter
// ============================================================================

DSPARK_TEST(FrequencyShifter_shifts_signal)
{
    FrequencyShifter<float> fs;
    fs.prepare(defaultSpec());
    fs.setShift(100.0f);
    fs.setMix(1.0f);

    auto tb = makeMonoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);
    auto ref = makeMonoBuffer(4096);
    ref.fillSine(440.0f, 44100.0f);

    fs.processBlock(tb.view());

    float diff = 0.0f;
    for (int i = 256; i < 4096; ++i)
        diff += std::abs(tb.ch(0)[i] - ref.ch(0)[i]);
    EXPECT_GT(diff, 1.0f); // Output should differ
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(FrequencyShifter_invalid_inputs_are_ignored)
{
    // setShift(NaN) poisoned phase_ PERMANENTLY (phase += NaN*n, then
    // fmod(NaN) = NaN forever: measured 3072 non-finite samples AFTER
    // restoring a valid shift; only reset()/prepare() could clear it), +Inf
    // did the same through fmod(Inf), setMix(NaN) was a per-block NaN storm,
    // and an invalid prepare() either parked a NaN sample rate in the
    // carrier math (2047 non-finite) or threw from resize. All inert now.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    FrequencyShifter<float> subject, twin;
    subject.prepare(spec(48000.0, 512, 2));
    twin.prepare(spec(48000.0, 512, 2));
    subject.setShift(50.0f); twin.setShift(50.0f);
    subject.setMix(0.8f);    twin.setMix(0.8f);

    auto ta = makeBuffer(2, 512);
    auto tb = makeBuffer(2, 512);
    float maxDiff = 0.0f;
    int badCount = 0;
    for (int blk = 0; blk < 10; ++blk)
    {
        if (blk == 2)
        {
            subject.setShift(nan); subject.setShift(inf);
            subject.setMix(nan);   subject.setMix(inf);
        }
        if (blk == 5)
        {
            subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
            subject.prepare(spec(48000.0, 512, -3)); // used to throw from resize
        }
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.5f * std::sin(6.2831853f * 440.0f * (blk * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
            }
        subject.processBlock(ta.view());
        twin.processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(ta.ch(ch)[i])) ++badCount;
                maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
            }
    }
    EXPECT_EQ(badCount, 0);       // old build: permanent NaN storm
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
    EXPECT_NEAR(subject.getShift(), 50.0f, 1e-6f);
    EXPECT_NEAR(subject.getMix(), 0.8f, 1e-6f);
}

DSPARK_TEST(FrequencyShifter_shift_frequency_is_exact)
{
    // The core SSB property (a 480 Hz probe lands at exactly 480 + shift,
    // sign included) had no test anywhere in the suite. Measured by
    // zero-crossing count over ~170 ms after the Hilbert settles.
    for (float sh : { 100.0f, -100.0f })
    {
        FrequencyShifter<float> fs;
        fs.setShift(sh);
        fs.setMix(1.0f);
        fs.prepare(spec(48000.0, 512, 1));

        auto blk = makeBuffer(1, 512);
        std::vector<float> tail;
        for (int b = 0; b < 20; ++b)
        {
            for (int i = 0; i < 512; ++i)
                blk.ch(0)[i] = 0.7f * std::sin(6.2831853f * 480.0f * (b * 512 + i) / 48000.0f);
            fs.processBlock(blk.view());
            if (b >= 4) tail.insert(tail.end(), blk.ch(0), blk.ch(0) + 512);
        }
        int zc = 0;
        for (size_t i = 1; i < tail.size(); ++i)
            if (tail[i - 1] <= 0.0f && tail[i] > 0.0f) ++zc;
        const float measured = zc * 48000.0f / static_cast<float>(tail.size());
        EXPECT_NEAR(measured, 480.0f + sh, 15.0f); // measured 580.1 / 380.9
    }
}

DSPARK_TEST(FrequencyShifter_mix_change_is_click_free)
{
    // The mix used to be applied per block WITHOUT smoothing: with the wet
    // path out of phase with the dry, a full step jumped the crossfade by up
    // to |shifted - dry| in one sample (measured 0.93, 20x the settled
    // regime's 0.046). Smoothed now (linear per-block ramp).
    FrequencyShifter<float> fs;
    fs.setShift(50.0f);
    fs.setMix(0.0f);
    fs.prepare(spec(48000.0, 512, 1));

    const int nBlocks = 30, N = 512;
    std::vector<float> out;
    out.reserve(static_cast<size_t>(nBlocks) * N);
    auto blk = makeBuffer(1, N);
    for (int b = 0; b < nBlocks; ++b)
    {
        if (b == 20) fs.setMix(1.0f);
        if (b == 25) fs.setMix(0.0f);
        for (int i = 0; i < N; ++i)
            blk.ch(0)[i] = 0.8f * std::sin(6.2831853f * 440.0f * (b * N + i) / 48000.0f);
        fs.processBlock(blk.view());
        out.insert(out.end(), blk.ch(0), blk.ch(0) + N);
    }
    auto maxStep = [&](int from, int to) {
        float m = 0.0f;
        for (int i = std::max(1, from); i < to; ++i)
            m = std::max(m, std::abs(out[static_cast<size_t>(i)] - out[static_cast<size_t>(i - 1)]));
        return m;
    };
    const float base = maxStep(8 * N, 20 * N);
    const float up   = maxStep(20 * N - 4, 21 * N);
    const float down = maxStep(25 * N - 4, 26 * N);
    EXPECT_TRUE(base < 0.12f); // regime sanity (measured 0.046)
    EXPECT_TRUE(up < 0.15f);   // old build: 0.58
    EXPECT_TRUE(down < 0.15f); // old build: 0.93
}

DSPARK_TEST(FrequencyShifter_zero_shift_passthrough)
{
    FrequencyShifter<float> fs;
    fs.prepare(defaultSpec());
    fs.setShift(0.0f);
    fs.setMix(1.0f);

    auto tb = makeMonoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);
    auto ref = makeMonoBuffer(4096);
    ref.fillSine(440.0f, 44100.0f);

    fs.processBlock(tb.view());

    // With 0 Hz shift, the Hilbert's real path output should preserve energy.
    // The allpass network introduces phase shift but preserves magnitude.
    float outEnergy = 0.0f, inEnergy = 0.0f;
    for (int i = 512; i < 4096; ++i)
    {
        outEnergy += tb.ch(0)[i] * tb.ch(0)[i];
        inEnergy += ref.ch(0)[i] * ref.ch(0)[i];
    }
    EXPECT_NEAR(outEnergy, inEnergy, inEnergy * 0.3f); // Within 30%
}

// ============================================================================
// DeEsser
// ============================================================================

DSPARK_TEST(DeEsser_invalid_inputs_are_ignored)
{
    // setFrequency/setBandwidth(NaN) poisoned BOTH biquads' recursive state
    // permanently (5120 non-finite samples measured AFTER restoring valid
    // values), setAttack/setThreshold(NaN) silently KILLED the reduction
    // (NaN comparisons kept grDb at 0 forever: GR 12 dB -> 0.00 with clean
    // audio, an invisible failure), prepare()'s old `<= 0` guard let a NaN
    // rate through (2048 non-finite), a negative channel count silently
    // disabled the effect, and a wild detection enum made the getter lie.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    DeEsser<float> subject, twin;
    for (auto* p : { &subject, &twin })
    {
        p->setFrequency(7000.0f); p->setBandwidth(2.0f);
        p->setThreshold(-20.0f);  p->setReduction(12.0f);
        p->prepare(spec(48000.0, 512, 2));
    }

    auto sibProbe = [](long n) {
        return 0.5f * std::sin(6.2831853f * 7500.0f * n / 48000.0f)
             + 0.3f * std::sin(6.2831853f * 300.0f * n / 48000.0f);
    };

    auto ta = makeBuffer(2, 512);
    auto tb = makeBuffer(2, 512);
    float maxDiff = 0.0f;
    int badCount = 0;
    for (int blk = 0; blk < 12; ++blk)
    {
        if (blk == 2)
        {
            subject.setFrequency(nan);  subject.setFrequency(inf);
            subject.setBandwidth(nan);  subject.setBandwidth(inf);
            subject.setAttack(nan);     subject.setRelease(nan);
            subject.setThreshold(nan);  subject.setReduction(nan);
        }
        if (blk == 6)
        {
            subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
            subject.prepare(spec(48000.0, 512, -3));
        }
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = sibProbe(blk * 512 + i) * (ch == 0 ? 1.0f : 0.8f);
                ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
            }
        subject.processBlock(ta.view());
        twin.processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(ta.ch(ch)[i])) ++badCount;
                maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
            }
    }
    EXPECT_EQ(badCount, 0);       // old build: permanent NaN storm
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
    EXPECT_NEAR(subject.getThreshold(), -20.0f, 1e-6f);
    EXPECT_NEAR(subject.getFrequency(), 7000.0f, 1e-6f);
    EXPECT_GT(subject.getGainReductionDb(), 3.0f); // still reducing (old: dead at 0)

    subject.setDetectionMode(static_cast<DeEsser<float>::DetectionMode>(99));
    EXPECT_EQ(static_cast<int>(subject.getDetectionMode()), 1); // Derivative (last member)
}

DSPARK_TEST(DeEsser_beyond_nyquist_frequency_is_stable)
{
    // The bandpass factory clamps its frequency internally, but the
    // precomputed peak-bell terms used the RAW frequency: beyond Nyquist
    // sin(w0) goes negative, alpha flips sign and the dynamic bell turns
    // unstable as soon as gain reduction engages (measured: output 8e+37
    // with 12113 non-finite samples). The clamp is shared now.
    DeEsser<float> de;
    de.setFrequency(30000.0f); // > Nyquist at 44.1k
    de.setThreshold(-80.0f);
    de.setReduction(18.0f);
    de.prepare(spec(44100.0, 512, 1));

    auto blk = makeBuffer(1, 512);
    float maxOut = 0.0f;
    int badCount = 0;
    for (int b = 0; b < 24; ++b)
    {
        for (int i = 0; i < 512; ++i)
            blk.ch(0)[i] = 0.7f * std::sin(6.2831853f * 18000.0f * (b * 512 + i) / 44100.0f);
        de.processBlock(blk.view());
        for (int i = 0; i < 512; ++i)
        {
            if (!std::isfinite(blk.ch(0)[i])) ++badCount;
            else maxOut = std::max(maxOut, std::abs(blk.ch(0)[i]));
        }
    }
    EXPECT_EQ(badCount, 0);
    EXPECT_TRUE(maxOut < 2.0f); // bounded (measured 0.70)
}

DSPARK_TEST(DeEsser_reduces_sibilance)
{
    DeEsser<float> de;
    de.prepare(spec(48000.0, 4096, 1));
    de.setFrequency(7000.0f);
    de.setThreshold(-30.0f);
    de.setReduction(12.0f);

    // Generate a loud signal at the sibilance frequency
    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 7000.0f, 48000.0f, 0.8f);

    de.processBlock(tb.view());

    // Output energy should be reduced compared to input
    float outEnergy = 0.0f;
    for (int i = 512; i < 4096; ++i)
        outEnergy += tb.ch(0)[i] * tb.ch(0)[i];

    float inEnergy = 0.0f;
    auto ref = makeBuffer(1, 4096);
    generateSine(ref.ch(0), 4096, 7000.0f, 48000.0f, 0.8f);
    for (int i = 512; i < 4096; ++i)
        inEnergy += ref.ch(0)[i] * ref.ch(0)[i];

    EXPECT_LT(outEnergy, inEnergy * 0.8f); // At least 20% reduction
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(DeEsser_passes_low_freq)
{
    DeEsser<float> de;
    de.prepare(spec(48000.0, 4096, 1));
    de.setFrequency(7000.0f);
    de.setThreshold(-30.0f);
    de.setReduction(12.0f);

    // Low frequency should pass through mostly unchanged
    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 200.0f, 48000.0f, 0.5f);
    auto ref = makeBuffer(1, 4096);
    generateSine(ref.ch(0), 4096, 200.0f, 48000.0f, 0.5f);

    de.processBlock(tb.view());

    // Energy should be similar (within 10%)
    float outEnergy = 0.0f, inEnergy = 0.0f;
    for (int i = 256; i < 4096; ++i)
    {
        outEnergy += tb.ch(0)[i] * tb.ch(0)[i];
        inEnergy += ref.ch(0)[i] * ref.ch(0)[i];
    }
    EXPECT_GT(outEnergy, inEnergy * 0.85f);
}

DSPARK_TEST(DeEsser_gain_reduction_meter)
{
    DeEsser<float> de;
    de.prepare(spec(48000.0, 4096, 1));
    de.setFrequency(7000.0f);
    de.setThreshold(-40.0f);
    de.setReduction(12.0f);

    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 7000.0f, 48000.0f, 0.8f);

    de.processBlock(tb.view());

    // Should report some gain reduction
    EXPECT_GT(de.getGainReductionDb(), 0.5f);
}

// ============================================================================
// NoiseGenerator
// ============================================================================

DSPARK_TEST(Tremolo_invalid_inputs_are_ignored)
{
    // setDepth(NaN) reached the LinearSmoother (which has no NaN guard) and
    // poisoned the gain (measured 1263 non-finite samples), setRate had no
    // validation at all (getRate()/state blob carried NaN), and a wild shape
    // enum skipped the dispatch switch entirely: the tremolo was silently OFF
    // with getShape() lying (measured: flat output at full depth). All inert
    // now: bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Tremolo<float> subject, twin;
    subject.prepare(spec(48000.0, 512, 2));
    twin.prepare(spec(48000.0, 512, 2));
    subject.setRate(5.0f);  twin.setRate(5.0f);
    subject.setDepth(0.7f); twin.setDepth(0.7f);

    auto ta = makeBuffer(2, 512);
    auto tb = makeBuffer(2, 512);
    float maxDiff = 0.0f;
    for (int blk = 0; blk < 8; ++blk)
    {
        if (blk == 2)
        {
            subject.setDepth(nan); subject.setDepth(inf);
            subject.setRate(nan);  subject.setRate(inf);
            subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
        }
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.5f * std::sin(6.2831853f * 440.0f * (blk * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
            }
        subject.processBlock(ta.view());
        twin.processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    }
    EXPECT_NO_NAN(ta.ch(0), 512);
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
    EXPECT_NEAR(subject.getRate(), 5.0f, 1e-6f);
    EXPECT_NEAR(subject.getDepth(), 0.7f, 1e-6f);

    // Wild shape enum clamps (the old switch skipped the whole block: the
    // tremolo silently stopped modulating) and the getter stays honest.
    subject.setShape(static_cast<Tremolo<float>::Shape>(99));
    EXPECT_EQ(static_cast<int>(subject.getShape()), 2); // Square (last member)
    subject.setDepth(1.0f);
    // DC input over two full 5 Hz LFO periods: the square gate must swing the
    // output between ~0 and 0.5 somewhere in the stream.
    float mn = 1e9f, mx = -1e9f;
    for (int blk = 0; blk < 40; ++blk)
    {
        for (int i = 0; i < 512; ++i) { ta.ch(0)[i] = 0.5f; ta.ch(1)[i] = 0.5f; }
        subject.processBlock(ta.view());
        for (int i = 0; i < 512; ++i) { mn = std::min(mn, ta.ch(0)[i]); mx = std::max(mx, ta.ch(0)[i]); }
    }
    EXPECT_GT(mx - mn, 0.3f); // still modulating (the old build stayed flat)
}

DSPARK_TEST(NoiseGenerator_invalid_inputs_and_reset_determinism)
{
    // setGain(NaN) used to output a full block of NaN (measured 2048/2048)
    // with getGain() reporting inf after setLevel(+inf); a wild type enum
    // made getType() lie (99); and reset() did NOT restore the per-channel
    // seeds, so the doc's offline-render determinism claim was false
    // (measured: different block hashes after reset). All fixed: invalid
    // inputs are inert (bit-identical twin), -inf level stays the documented
    // silence, and reset() reproduces the exact same sequence.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    NoiseGenerator<float> subject, twin;
    subject.prepare(spec(48000.0, 512, 2));
    twin.prepare(spec(48000.0, 512, 2));
    subject.setGain(0.5f);
    twin.setGain(0.5f);

    subject.setGain(nan);
    subject.setGain(inf);
    subject.setLevel(nan);
    subject.setLevel(inf);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    subject.prepare(spec(48000.0, 512, 0));
    EXPECT_NEAR(subject.getGain(), 0.5f, 1e-9f);

    auto ta = makeBuffer(2, 1024);
    auto tb = makeBuffer(2, 1024);
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());
    EXPECT_NO_NAN(ta.ch(0), 1024);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 1024; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin

    // -inf level is the documented silence.
    subject.setLevel(-inf);
    subject.processBlock(ta.view());
    subject.processBlock(ta.view());
    EXPECT_SILENT(ta.ch(0), 1024, 1e-12f);

    // Wild enum clamps; the getter stays honest.
    subject.setType(static_cast<NoiseGenerator<float>::Type>(99));
    EXPECT_EQ(static_cast<int>(subject.getType()), 2); // Brown (last member)

    // reset() reproduces the exact post-prepare sequence (offline renders).
    NoiseGenerator<float> ng;
    ng.prepare(spec(48000.0, 512, 2));
    ng.setType(NoiseGenerator<float>::Type::Pink);
    auto b1 = makeBuffer(2, 512);
    auto b2 = makeBuffer(2, 512);
    ng.processBlock(b1.view());
    ng.processBlock(b2.view()); // advance the PRNG past the first block
    ng.reset();
    ng.processBlock(b2.view());
    bool identical = true;
    for (int ch = 0; ch < 2 && identical; ++ch)
        for (int i = 0; i < 512; ++i)
            if (std::memcmp(&b1.ch(ch)[i], &b2.ch(ch)[i], sizeof(float)) != 0)
            { identical = false; break; }
    EXPECT_TRUE(identical); // the old reset continued the PRNG mid-stream
}

DSPARK_TEST(NoiseGenerator_produces_output)
{
    NoiseGenerator<float> ng;
    ng.prepare(defaultSpec());
    ng.setType(NoiseGenerator<float>::Type::White);

    auto tb = makeStereoBuffer(1024);
    tb.fillSilence();
    ng.processBlock(tb.view());

    float energy = 0.0f;
    for (int i = 0; i < 1024; ++i)
        energy += tb.ch(0)[i] * tb.ch(0)[i];
    EXPECT_GT(energy, 1.0f);
    EXPECT_NO_NAN(tb.ch(0), 1024);
}

DSPARK_TEST(NoiseGenerator_stereo_uncorrelated)
{
    NoiseGenerator<float> ng;
    ng.prepare(defaultSpec());

    auto tb = makeStereoBuffer(4096);
    ng.processBlock(tb.view());

    // L and R should NOT be identical
    float diff = 0.0f;
    for (int i = 0; i < 4096; ++i)
        diff += std::abs(tb.ch(0)[i] - tb.ch(1)[i]);
    EXPECT_GT(diff, 10.0f);
}

DSPARK_TEST(NoiseGenerator_level_control)
{
    NoiseGenerator<float> ng;
    ng.prepare(defaultSpec());
    ng.setLevel(-60.0f); // very quiet

    // Settle the level smoother (the gain is de-zippered from its default), then
    // measure on a fresh block so the peak reflects the steady -60 dB level.
    for (int w = 0; w < 4; ++w)
    {
        auto wb = makeMonoBuffer(4096);
        ng.processBlock(wb.view());
    }

    auto tb = makeMonoBuffer(4096);
    ng.processBlock(tb.view());

    float peak = 0.0f;
    for (int i = 0; i < 4096; ++i)
        peak = std::max(peak, std::abs(tb.ch(0)[i]));
    EXPECT_LT(peak, 0.01f); // -60 dB = ~0.001 gain
}

DSPARK_TEST(NoiseGenerator_pink_type)
{
    NoiseGenerator<float> ng;
    ng.prepare(defaultSpec());
    ng.setType(NoiseGenerator<float>::Type::Pink);

    auto tb = makeMonoBuffer(4096);
    ng.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);

    float energy = 0.0f;
    for (int i = 0; i < 4096; ++i)
        energy += tb.ch(0)[i] * tb.ch(0)[i];
    EXPECT_GT(energy, 0.1f);
}

// ============================================================================
// M-005 AG-4 audit pins (additive): sideband accuracy + RT denormal hygiene
// ============================================================================

// FrequencyShifter single-sideband: +shift keeps the wanted sideband and
// rejects the image and carrier by >30 dB; output stays finite (DenormalGuard
// on the per-sample Hilbert path, F1).
DSPARK_TEST(FrequencyShifter_ssb_rejection_and_finite)
{
    const double fs = 48000.0, f0 = 2000.0, shift = 300.0, amp = 0.5;
    FrequencyShifter<float> fsh;
    fsh.prepare(spec(fs, 512, 1));
    fsh.setShift(static_cast<float>(shift));
    fsh.setMix(1.0f);
    const int N = 48000;
    std::vector<float> x(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) x[static_cast<size_t>(i)] = static_cast<float>(amp * std::sin(twoPi<double> * f0 * i / fs));
    for (int off = 0; off < N; off += 512)
    {
        const int n = std::min(512, N - off);
        float* ch[1] = { x.data() + off };
        AudioBufferView<float> v(ch, 1, n);
        fsh.processBlock(v);
    }
    auto goertzel = [&](double f) {
        const double w = twoPi<double> * f / fs, c = 2.0 * std::cos(w);
        double s1 = 0, s2 = 0;
        for (int i = 4800; i < N; ++i) { const double s0 = x[static_cast<size_t>(i)] + c * s1 - s2; s2 = s1; s1 = s0; }
        return 2.0 * std::sqrt((s1 - s2 * std::cos(w)) * (s1 - s2 * std::cos(w)) + (s2 * std::sin(w)) * (s2 * std::sin(w))) / (N - 4800);
    };
    const double wanted = goertzel(f0 + shift), image = goertzel(f0 - shift), carrier = goertzel(f0);
    EXPECT_LT(20.0 * std::log10(image / wanted), -30.0);
    EXPECT_LT(20.0 * std::log10(carrier / wanted), -30.0);
    for (int i = 0; i < N; ++i) EXPECT_TRUE(std::isfinite(x[static_cast<size_t>(i)]));
}

// RingModulator GeometricMean: the scaled soar term must not inject static DC -
// a silent input produces exact silence (R2/R3), output finite.
DSPARK_TEST(RingModulator_geomean_no_dc_leak_on_silence)
{
    RingModulator<float> rm;
    rm.prepare(spec(48000.0, 512, 1));
    rm.setFrequency(300.0f);
    rm.setMix(1.0f);
    rm.setSoar(0.8f);
    rm.setMode(RingModulator<float>::Mode::GeometricMean);
    float maxAbs = 0.0f;
    for (int blk = 0; blk < 94; ++blk)
    {
        auto b = makeBuffer(1, 512);
        std::memset(b.ch(0), 0, sizeof(float) * 512);
        rm.processBlock(b.view());
        for (int i = 0; i < 512; ++i) maxAbs = std::max(maxAbs, std::fabs(b.ch(0)[i]));
    }
    EXPECT_LT(maxAbs, 1e-9f);
}

// ============================================================================
// M-006 AG-5 C1: front-door non-finite input guards. A transient NaN/Inf input
// must NOT permanently poison recursive state (feedback filters, allpass, FDN).
// After a poison block, sustained clean input must recover to finite, non-zero
// output WITHOUT an intervening reset(). Revert-check: removing the guard turns
// each of these RED (stuck NaN / silence). Same defect class as M-005 C1.
// ============================================================================
DSPARK_TEST(Delay_survives_nonfinite_input)
{
    Delay<float> d;
    d.prepare(spec(48000.0, 512, 2), 1.0);
    auto p = makeBuffer(2, 512);
    p.fillSine(220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(1)[192] = std::numeric_limits<float>::infinity();
    d.processBlock(p.view(), 50.0f, 0.7f, 5000.0f, 0.0f); // feedback + LP recirculation
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 30; ++blk)
    {
        auto b = makeBuffer(2, 512);
        b.fillSine(220.0f, 48000.0f, 0.3f);
        d.processBlock(b.view(), 50.0f, 0.7f, 5000.0f, 0.0f);
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

DSPARK_TEST(Chorus_survives_nonfinite_input)
{
    Chorus<float> c;
    c.prepare(spec(48000.0, 512, 2));
    c.setFeedback(0.7f);
    c.setMix(0.5f);
    auto p = makeBuffer(2, 512);
    p.fillSine(220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(1)[192] = -std::numeric_limits<float>::infinity();
    c.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 30; ++blk)
    {
        auto b = makeBuffer(2, 512);
        b.fillSine(220.0f, 48000.0f, 0.3f);
        c.processBlock(b.view());
        if (blk >= 25)
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 512; ++i)
                {
                    if (!std::isfinite(b.ch(ch)[i])) finite = false;
                    energy += double(b.ch(ch)[i]) * double(b.ch(ch)[i]);
                }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);
}

DSPARK_TEST(Phaser_survives_nonfinite_input)
{
    Phaser<float> ph;
    ph.prepare(spec(48000.0, 512, 2));
    ph.setFeedback(0.5f);
    ph.setMix(0.5f);
    auto p = makeBuffer(2, 512);
    p.fillSine(220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::infinity();
    p.ch(1)[192] = std::numeric_limits<float>::quiet_NaN();
    ph.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 30; ++blk)
    {
        auto b = makeBuffer(2, 512);
        b.fillSine(220.0f, 48000.0f, 0.3f);
        ph.processBlock(b.view());
        if (blk >= 25)
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 512; ++i)
                {
                    if (!std::isfinite(b.ch(ch)[i])) finite = false;
                    energy += double(b.ch(ch)[i]) * double(b.ch(ch)[i]);
                }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);
}

DSPARK_TEST(AlgoReverb_survives_nonfinite_input)
{
    AlgorithmicReverb<float> r;
    r.prepare(spec(48000.0, 512, 2));
    r.setDecay(2.0f);
    r.setMix(1.0f);
    auto p = makeBuffer(2, 512);
    p.fillSine(220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(1)[192] = std::numeric_limits<float>::infinity();
    r.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 60; ++blk)
    {
        auto b = makeBuffer(2, 512);
        b.fillSine(220.0f, 48000.0f, 0.3f);
        r.processBlock(b.view());
        if (blk >= 55)
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 512; ++i)
                {
                    if (!std::isfinite(b.ch(ch)[i])) finite = false;
                    energy += double(b.ch(ch)[i]) * double(b.ch(ch)[i]);
                }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);
}
