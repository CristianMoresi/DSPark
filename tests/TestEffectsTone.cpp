// DSPark Tests - Effects Tone
// Filters (FilterEngine), Equalizer, Saturation, DCBlocker

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/Biquad.h"
#include "../Core/Hysteresis.h"
#include "../Effects/Filters.h"
#include "../Effects/Equalizer.h"
#include "../Effects/Saturation.h"
#include "../Effects/DCBlocker.h"
#include "../Effects/TapeMachine.h"
#include "../Effects/TubePreamp.h"
#include "../Effects/TransformerModel.h"
#include "../Effects/Clipper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <array>
#include <atomic>
#include <thread>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// FilterEngine
// ============================================================================

DSPARK_TEST(FilterEngine_LP_passes_low)
{
    FilterEngine<float> filt;
    filt.prepare(defaultSpec());
    filt.setLowPass(2000.0f);

    // Generate low-frequency sine
    auto tb = makeStereoBuffer(8192);
    tb.fillSine(200.0f, 44100.0f);

    float peakBefore = measurePeak(tb.ch(0), 8192);
    filt.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 2048, 4096);

    // Low freq should pass through
    EXPECT_GT(peakAfter, peakBefore * 0.8f);
}

DSPARK_TEST(FilterEngine_LP_attenuates_high)
{
    FilterEngine<float> filt;
    filt.prepare(defaultSpec());
    filt.setLowPass(1000.0f);

    auto tb = makeStereoBuffer(8192);
    tb.fillSine(10000.0f, 44100.0f);

    filt.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 2048, 4096);

    EXPECT_LT(peakAfter, 0.15f);
}

DSPARK_TEST(FilterEngine_HP_attenuates_low)
{
    FilterEngine<float> filt;
    filt.prepare(defaultSpec());
    filt.setHighPass(1000.0f);

    auto tb = makeStereoBuffer(8192);
    tb.fillSine(100.0f, 44100.0f);

    filt.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 2048, 4096);

    EXPECT_LT(peakAfter, 0.15f);
}

DSPARK_TEST(FilterEngine_silence)
{
    FilterEngine<float> filt;
    filt.prepare(defaultSpec());
    filt.setLowPass(1000.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    filt.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

namespace {

// Steady-state gain (dB) of a sine through a FilterEngine (RMS of the tail).
float engineToneGainDb(FilterEngine<float>& f, float freq, float sr, int n = 16384)
{
    const int kSkip = n / 4;
    auto tb = makeBuffer(1, n);
    tb.fillSine(freq, sr, 0.1f);
    f.processBlock(tb.view());
    double acc = 0.0;
    for (int i = kSkip; i < n; ++i)
        acc += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i];
    const double rms = std::sqrt(acc / (n - kSkip));
    return 20.0f * static_cast<float>(std::log10(rms / (0.1 / std::sqrt(2.0)) + 1e-30));
}

} // namespace

DSPARK_TEST(FilterEngine_lp_resonance_is_honoured)
{
    // The Butterworth cascade table used to silently override the user Q for
    // every LP/HP slope: setLowPass(f, 8.0f) sounded identical to Butterworth
    // and setResonance() was dead. The user Q now scales the final stage:
    // neutral at the 0.707 default (bit-identical cascade), and Q = 8 peaks
    // 20*log10(8) = +18.06 dB at the cutoff (measured exactly that).
    FilterEngine<float> butter, resonant, steep;
    for (auto* f : { &butter, &resonant, &steep }) f->prepare(spec(44100.0, 16384, 1));
    butter.setLowPass(1000.0f, 0.707f, 12);
    resonant.setLowPass(1000.0f, 8.0f, 12);
    steep.setLowPass(1000.0f, 0.707f, 24);

    const float gB = engineToneGainDb(butter,  1000.0f, 44100.0f);
    const float gR = engineToneGainDb(resonant, 1000.0f, 44100.0f);
    const float gS = engineToneGainDb(steep,   1000.0f, 44100.0f);
    EXPECT_NEAR(gB, -3.01f, 0.5f);   // Butterworth -3 dB at fc
    EXPECT_NEAR(gR, 18.06f, 1.0f);   // |H(fc)| = Q (old header: -3.01, Q ignored)
    EXPECT_NEAR(gS, -3.01f, 0.5f);   // higher-order Butterworth stays -3 dB at fc
}

DSPARK_TEST(FilterEngine_reprepare_rebuilds_for_new_rate)
{
    // The static-path coefficient cache was not invalidated by prepare(): with
    // parameters equal to the smoother defaults (so no smoothing kicks in
    // after re-prepare), the old-rate coefficients kept running and the
    // cutoff landed at freq * newRate / oldRate. Measured: a 1 kHz LP
    // re-prepared 44.1k -> 88.2k read -3 dB at 2 kHz (cutoff doubled)
    // instead of the correct -12 dB.
    FilterEngine<float> f;
    f.prepare(spec(44100.0, 16384, 1));
    f.setLowPass(1000.0f, 0.707f, 12);       // == smoother defaults
    (void)engineToneGainDb(f, 500.0f, 44100.0f); // builds + caches at 44.1k

    f.prepare(spec(88200.0, 16384, 1));
    const float g2k = engineToneGainDb(f, 2000.0f, 88200.0f);
    EXPECT_LT(g2k, -8.0f);                   // one octave above the cutoff
}

DSPARK_TEST(FilterEngine_invalid_inputs_are_ignored)
{
    // NaN setters poisoned the smoothers and coefficients (whole block went
    // non-finite); processSample with a wild channel hit the biquad state out
    // of bounds (assert in debug, UB in release); prepare() accepted invalid
    // specs. All inert now.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    FilterEngine<float> subject, twin;
    for (auto* f : { &subject, &twin })
    {
        f->prepare(spec(44100.0, 4096, 1));
        f->setLowPass(2000.0f, 0.707f, 12);
    }
    subject.setFrequency(nan);
    subject.setResonance(nan);
    subject.setGain(nan);
    subject.setNonlinearity(nan);
    AudioSpec bad;
    bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
    bad.maxBlockSize = 0; bad.numChannels = -1;
    subject.prepare(bad);

    auto ta = makeBuffer(1, 4096);
    auto tb = makeBuffer(1, 4096);
    ta.fillSine(500.0f, 44100.0f, 0.1f);
    tb.fillSine(500.0f, 44100.0f, 0.1f);
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 4096);
    float maxDiff = 0.0f;
    for (int i = 0; i < 4096; ++i)
        maxDiff = std::max(maxDiff, std::abs(ta.ch(0)[i] - tb.ch(0)[i]));
    EXPECT_TRUE(maxDiff == 0.0f);            // invalid edits fully inert

    // Wild channel index on the per-sample API: unprocessed pass-through.
    float acc = 0.0f;
    for (int i = 0; i < 100; ++i) acc += subject.processSample(0.5f, 99);
    for (int i = 0; i < 100; ++i) acc += subject.processSample(0.5f, -3);
    EXPECT_NEAR(acc, 100.0f, 1e-4f);         // 200 calls x 0.5 passed through

    // Wild shape id from a cast/corrupt blob: clamped into range.
    subject.setShape(static_cast<FilterEngine<float>::Shape>(99));
    EXPECT_TRUE(static_cast<int>(subject.getShape()) <= static_cast<int>(FilterEngine<float>::Shape::Tilt));
}

DSPARK_TEST(FilterEngine_no_NaN)
{
    FilterEngine<float> filt;
    filt.prepare(defaultSpec());
    filt.setPeaking(1000.0f, 12.0f, 2.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    filt.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

// ============================================================================
// Equalizer
// ============================================================================

DSPARK_TEST(Equalizer_flat_passthrough)
{
    Equalizer<float> eq;
    eq.prepare(defaultSpec());
    eq.setNumBands(6);

    // All bands at 0dB gain = passthrough
    for (int i = 0; i < 6; ++i)
        eq.setBand(i, 1000.0f, 0.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);
    std::vector<float> original(tb.ch(0), tb.ch(0) + 4096);

    eq.processBlock(tb.view());

    // Should be very close to original (IIR adds minimal delay)
    float maxDiff = 0.0f;
    for (int i = 256; i < 4096; ++i)
    {
        float diff = std::abs(tb.ch(0)[i] - original[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    EXPECT_LT(maxDiff, 0.05f);
}

DSPARK_TEST(Equalizer_boost_measurable)
{
    Equalizer<float> eq;
    eq.prepare(defaultSpec());
    eq.setNumBands(1);
    eq.setBand(0, 1000.0f, 6.0f, 2.0f); // +6dB at 1kHz

    // Feed 1kHz sine
    auto tb = makeStereoBuffer(8192);
    tb.fillSine(1000.0f, 44100.0f);

    float peakBefore = measurePeak(tb.ch(0), 8192);
    eq.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 2048, 4096);

    // Should be boosted
    EXPECT_GT(peakAfter, peakBefore * 1.5f);
}

DSPARK_TEST(Equalizer_silence)
{
    Equalizer<float> eq;
    eq.prepare(defaultSpec());
    eq.setNumBands(4);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    eq.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

DSPARK_TEST(Equalizer_mode_switch_after_prepare_works)
{
    // setFilterMode(LinearPhase) after prepare() used to fall back silently to
    // the IIR path (the LP engine was only allocated when the mode was already
    // set at prepare time) while getLatency() still reported maxBlockSize:
    // measured argmax 0 with a 512-sample latency claim, desyncing host PDC.
    Equalizer<float> eq;
    eq.prepare(spec(48000.0, 512, 1));
    eq.setBand(0, 2000.0f, 6.0f);
    eq.setFilterMode(Equalizer<float>::FilterMode::LinearPhase);
    EXPECT_EQ(eq.getLatency(), 512);

    auto tb = makeBuffer(1, 512);
    int argmax = -1;
    float best = 0.0f;
    for (int blk = 0; blk < 4; ++blk)
    {
        for (int i = 0; i < 512; ++i) tb.ch(0)[i] = 0.0f;
        if (blk == 0) tb.ch(0)[0] = 1.0f;
        eq.processBlock(tb.view());
        for (int i = 0; i < 512; ++i)
            if (std::abs(tb.ch(0)[i]) > best) { best = std::abs(tb.ch(0)[i]); argmax = blk * 512 + i; }
    }
    EXPECT_EQ(argmax, 512); // the reported latency is the real one

    eq.setFilterMode(Equalizer<float>::FilterMode::MinimumPhase);
    EXPECT_EQ(eq.getLatency(), 0);
}

DSPARK_TEST(Equalizer_invalid_inputs_are_ignored)
{
    // A NaN band config used to zero the whole linear-phase kernel (total
    // silent mute, drawn curve at 0), poison the serialized state, and an
    // invalid prepare() with negative channels THREW from the huge size_t
    // cast. All inert now: bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    using EQ = Equalizer<float>;
    EQ subject, twin;
    auto setup = [](EQ& e) {
        e.setFilterMode(EQ::FilterMode::LinearPhase);
        e.prepare(spec(48000.0, 512, 2));
        e.setBand(0, 1000.0f, 6.0f, 1.0f);
    };
    setup(subject); setup(twin);

    subject.setBand(0, nan, nan, nan);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    subject.prepare(spec(48000.0, 512, -3)); // must not throw, must not resize
    subject.prepare(spec(48000.0, 0, 2));

    EQ::BandConfig wild;
    wild.frequency = 500.0f; wild.gain = 3.0f; wild.q = 1.0f;
    wild.type = static_cast<EQ::BandType>(99); wild.slope = 999;
    subject.setBand(1, wild);
    twin.setBand(1, wild); // both get the sanitized copy
    EXPECT_EQ(static_cast<int>(subject.getBandConfig(1).type), 7); // clamped to Tilt
    EXPECT_EQ(subject.getBandConfig(1).slope, 48);

    // The NaN never reached the stored config nor the state blob.
    EXPECT_NEAR(subject.getBandConfig(0).frequency, 1000.0f, 1e-3f);
    auto blob = subject.getState();
    StateReader r(blob.data(), blob.size());
    EXPECT_NEAR(r.read("b0.freq", 999.0f), 1000.0f, 1e-3f);

    EXPECT_EQ(subject.getLatency(), 512);

    auto ta = makeBuffer(2, 4096);
    auto tb = makeBuffer(2, 4096);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
        {
            const float v = 0.4f * std::sin(6.2831853f * (ch == 0 ? 440.0f : 620.0f) * i / 48000.0f);
            ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
        }
    // Process per 512-sample block (the LP engine's prepared block size).
    for (int off = 0; off < 4096; off += 512)
    {
        float* pa[2] = { ta.ch(0) + off, ta.ch(1) + off };
        float* pb[2] = { tb.ch(0) + off, tb.ch(1) + off };
        subject.processBlock(AudioBufferView<float>(pa, 2, 512));
        twin.processBlock(AudioBufferView<float>(pb, 2, 512));
    }

    EXPECT_NO_NAN(ta.ch(0), 4096);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 4096; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
}

namespace
{
// Steady-state sine gain (dB) through the EQ at `freq` (mono block path, 48 kHz).
float eqSineGainDb(Equalizer<float>& eq, float freq)
{
    const int B = 512;
    std::vector<float> buf(B);
    double sumSq = 0.0; int count = 0;
    for (int blk = 0; blk < 16; ++blk)
    {
        for (int i = 0; i < B; ++i)
            buf[i] = 0.25f * std::sin(6.2831853f * freq * (blk * B + i) / 48000.0f);
        float* p[1] = { buf.data() };
        eq.processBlock(AudioBufferView<float>(p, 1, B));
        if (blk >= 12)
        {
            for (int i = 0; i < B; ++i) sumSq += double(buf[i]) * buf[i];
            count += B;
        }
    }
    const double rms = std::sqrt(sumSq / count);
    return static_cast<float>(20.0 * std::log10(rms / (0.25 / std::sqrt(2.0))));
}
} // namespace

DSPARK_TEST(Equalizer_analysis_matches_audio)
{
    // (a) setMatchedBells() used to change the drawn curve and the LP kernel
    // but NOT the IIR engines (measured: 8.6 dB between the curve and the
    // audio at 21 kHz for a 16 kHz bell). The matched design now reaches the
    // engines through FilterEngine::setMatchedPeak().
    {
        Equalizer<float> eq;
        eq.prepare(spec(48000.0, 512, 1));
        eq.setMatchedBells(true);
        Equalizer<float>::BandConfig cfg;
        cfg.frequency = 16000.0f; cfg.gain = 12.0f; cfg.q = 1.0f;
        cfg.type = Equalizer<float>::BandType::Peak;
        eq.setBand(0, cfg);

        const float probe = 21000.0f;
        float mag = 1.0f;
        eq.getMagnitudeForFrequencyArray(&probe, &mag, 1);
        const float drawnDb = 20.0f * std::log10(mag);
        const float measDb = eqSineGainDb(eq, probe);
        EXPECT_NEAR(drawnDb, measDb, 1.0f);
    }
    // (b) The analysis cascade ignored the user Q on LP/HP bands while the
    // engine applied it (measured: 21 dB apart at fc with Q = 8).
    {
        Equalizer<float> eq;
        eq.prepare(spec(48000.0, 512, 1));
        Equalizer<float>::BandConfig cfg;
        cfg.frequency = 1000.0f; cfg.gain = 0.0f; cfg.q = 8.0f;
        cfg.type = Equalizer<float>::BandType::LowPass; cfg.slope = 24;
        eq.setBand(0, cfg);

        const float probe = 1000.0f;
        float mag = 1.0f;
        eq.getMagnitudeForFrequencyArray(&probe, &mag, 1);
        const float drawnDb = 20.0f * std::log10(mag);
        const float measDb = eqSineGainDb(eq, probe);
        EXPECT_GT(measDb, 15.0f); // the resonance is real
        EXPECT_NEAR(drawnDb, measDb, 1.0f);
    }
}

DSPARK_TEST(Equalizer_processSample_applies_band_changes)
{
    // The per-sample path never consumed configDirty_ nor rebuilt the engine
    // coefficients, so setBand() had no effect for per-sample streams
    // (measured: a +12 dB bell returned 0.00 dB). It now applies pending
    // changes immediately (without the block path's smoothing).
    Equalizer<float> eq;
    eq.prepare(spec(48000.0, 512, 1));
    eq.setBand(0, 10000.0f, 12.0f, 1.0f);

    double inSq = 0.0, outSq = 0.0;
    for (int i = 0; i < 8192; ++i)
    {
        const float x = 0.25f * std::sin(6.2831853f * 10000.0f * i / 48000.0f);
        const float y = eq.processSample(x, 0);
        if (i > 4096) { inSq += double(x) * x; outSq += double(y) * y; }
    }
    const float ratioDb = static_cast<float>(10.0 * std::log10(outSq / inSq));
    EXPECT_NEAR(ratioDb, 12.0f, 1.0f);
}

// ============================================================================
// Saturation
// ============================================================================

DSPARK_TEST(Saturation_output_bounded)
{
    Saturation<float> sat;
    sat.prepare(defaultSpec());
    sat.setDrive(24.0f); // Heavy drive
    sat.setMix(1.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f, 1.0f);

    sat.process(tb.view());

    // Output should be bounded (saturation clips/limits)
    EXPECT_BOUNDED(tb.ch(0), 4096, -5.0f, 5.0f); // Generous bound
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(Saturation_antialiasing_bounded_and_active)
{
    // Antiderivative anti-aliasing (ADAA) on a fast, driven signal (default
    // SoftClip, so no algorithm crossfade): the output must stay finite/bounded
    // and must differ from the plain path, confirming the ADAA branch is engaged.
    Saturation<float> aa;
    aa.prepare(defaultSpec());
    aa.setDrive(18.0f);
    aa.setMix(1.0f);
    aa.setAntialiasing(true);
    auto tbAa = makeStereoBuffer(4096);
    tbAa.fillSine(2000.0f, 44100.0f, 1.0f);
    aa.process(tbAa.view());
    EXPECT_BOUNDED(tbAa.ch(0), 4096, -2.0f, 2.0f);
    EXPECT_NO_NAN(tbAa.ch(0), 4096);

    Saturation<float> plain;
    plain.prepare(defaultSpec());
    plain.setDrive(18.0f);
    plain.setMix(1.0f);
    auto tbPl = makeStereoBuffer(4096);
    tbPl.fillSine(2000.0f, 44100.0f, 1.0f);
    plain.process(tbPl.view());

    float maxDiff = 0.0f;
    for (int i = 0; i < 4096; i++) maxDiff = std::max(maxDiff, std::abs(tbAa.ch(0)[i] - tbPl.ch(0)[i]));
    EXPECT_GT(maxDiff, 0.001f);
}

DSPARK_TEST(Saturation_silence)
{
    Saturation<float> sat;
    sat.prepare(defaultSpec());
    sat.setDrive(6.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    sat.process(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 0.001f);
}

DSPARK_TEST(Saturation_adds_harmonics)
{
    Saturation<float> sat;
    sat.prepare(defaultSpec());
    sat.setDrive(18.0f);
    sat.setMix(1.0f);

    constexpr int N = 8192;
    auto tb = makeStereoBuffer(N);
    tb.fillSine(440.0f, 44100.0f);

    sat.process(tb.view());

    // Should have harmonics at 880 Hz (2nd harmonic)
    float mag440 = measureFrequencyMagnitude(tb.ch(0), N, 440.0f, 44100.0f);
    float mag880 = measureFrequencyMagnitude(tb.ch(0), N, 880.0f, 44100.0f);

    // Saturation should introduce harmonics (even slight amount counts)
    EXPECT_GT(mag880, 0.001f);
    EXPECT_GT(mag440, mag880); // Fundamental still dominant
}

namespace {

// Steady-state gain (dB) of a 1 kHz sine at a given amplitude through a
// Saturation algorithm at neutral drive. RMS over the last 3/4 of the block
// (skips the 20 ms parameter smoothers).
float satToneGainDb(Saturation<float>::Algorithm algo, float amp)
{
    constexpr int N = 16384, kSkip = 4096;
    Saturation<float> sat;
    sat.prepare(spec(44100.0, N, 2));
    sat.setAlgorithm(algo);
    sat.setDrive(0.0f);
    sat.setMix(1.0f);
    auto tb = makeStereoBuffer(N);
    tb.fillSine(1000.0f, 44100.0f, amp);
    sat.process(tb.view());

    double acc = 0.0;
    for (int i = kSkip; i < N; ++i)
        acc += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i];
    const double rms = std::sqrt(acc / (N - kSkip));
    return 20.0f * static_cast<float>(std::log10(rms / (amp / std::sqrt(2.0)) + 1e-30));
}

} // namespace

DSPARK_TEST(Saturation_tape_no_low_level_dead_zone)
{
    // The old pseudo-hysteresis stepped M by ManDiff*|dH|/(a+|dH|) with a~1:
    // a coercivity of ~1.0 in absolute full-scale units, i.e. tape with NO
    // AC bias. Output scaled like input^2 (-60 dBFS in -> -121 dBFS out),
    // muting quiet passages and low-level harmonic detail. The anhysteretic
    // Langevin model must be transparent at low level and level-independent.
    const float gQuiet = satToneGainDb(Saturation<float>::Algorithm::Tape, 0.00316f); // -50 dBFS
    const float gMid   = satToneGainDb(Saturation<float>::Algorithm::Tape, 0.1f);     // -20 dBFS
    EXPECT_NEAR(gQuiet, 0.0f, 1.5f);
    EXPECT_NEAR(gMid,   0.0f, 1.5f);
    EXPECT_LT(std::abs(gQuiet - gMid), 1.0f);
}

DSPARK_TEST(Saturation_multistage_is_gain_staged)
{
    // The cascade undoes each stage's fixed small-signal factor; at neutral
    // drive it must be roughly transparent (not the -15 dB drop of the
    // unstaged chain) and stay linear across levels (the tape stage used to
    // add a dead zone here too).
    const float gQuiet = satToneGainDb(Saturation<float>::Algorithm::MultiStage, 0.00316f);
    const float gMid   = satToneGainDb(Saturation<float>::Algorithm::MultiStage, 0.1f);
    EXPECT_NEAR(gQuiet, 0.0f, 1.5f);
    EXPECT_LT(std::abs(gQuiet - gMid), 1.5f);
}

namespace {

// Gain (dB) at an arbitrary frequency through a Saturation algorithm at
// neutral drive (RMS over the steady tail, like satToneGainDb).
float satFreqGainDb(Saturation<float>::Algorithm algo, float freq, float amp)
{
    constexpr int N = 16384, kSkip = 4096;
    Saturation<float> sat;
    sat.prepare(spec(44100.0, N, 2));
    sat.setAlgorithm(algo);
    sat.setDrive(0.0f);
    sat.setMix(1.0f);

    auto tb = makeStereoBuffer(N);
    tb.fillSine(freq, 44100.0f, amp);
    sat.process(tb.view());

    double acc = 0.0;
    for (int i = kSkip; i < N; ++i)
        acc += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i];
    const double rms = std::sqrt(acc / (N - kSkip));
    return 20.0f * static_cast<float>(std::log10(rms / (amp / std::sqrt(2.0)) + 1e-30));
}

} // namespace

DSPARK_TEST(Saturation_transformer_and_multistage_are_spectrally_balanced)
{
    // A transformer's linear (small-signal) response is flat: the band split
    // may only move the saturation THRESHOLD, never the linear gain. The old
    // unnormalised band factors (1.3/0.7) were a fixed +2.3/-3.1 dB shelf at
    // 250 Hz, which in the MultiStage cascade buried mids and highs ~6 dB
    // below the bass ("all mids and highs gone" in DSParkLab).
    using A = Saturation<float>::Algorithm;
    for (auto algo : { A::Transformer, A::MultiStage })
    {
        const float gLow  = satFreqGainDb(algo, 100.0f,  0.1f);
        const float gMid  = satFreqGainDb(algo, 3000.0f, 0.1f);
        EXPECT_LT(std::abs(gLow - gMid), 3.0f);
    }
}

DSPARK_TEST(Saturation_all_algorithms_honour_neutral_drive)
{
    // Pool contract: at 0 dB drive every algorithm is roughly transparent to
    // low-level signals - no hidden gain (the Wavefolder used to carry a 2*pi
    // = +16 dB factor that exploded to +56 dB of small-signal gain at full
    // drive) and no dead zone / expansion (the Tape used to scale output like
    // input^2). Transformer (-3.5 dB HF split) and MultiStage (its cascade)
    // sit lower by design but must still be linear across levels.
    for (int ai = 0; ai < 10; ++ai)
    {
        const auto algo = static_cast<Saturation<float>::Algorithm>(ai);
        const float gQuiet = satToneGainDb(algo, 0.00316f); // -50 dBFS
        const float gMid   = satToneGainDb(algo, 0.1f);     // -20 dBFS
        EXPECT_GT(gQuiet, -10.0f);
        EXPECT_LT(gQuiet,  1.5f);
        EXPECT_LT(std::abs(gQuiet - gMid), 1.5f);
    }
}

namespace {

// Push nBlocks of a phase-continuous 500 Hz stereo sine through sat (blocks of
// 256) and return channel 0 of the LAST block.
std::vector<float> satRunSine(Saturation<float>& sat, double& phase, int nBlocks)
{
    constexpr int kBlock = 256;
    auto tb = makeBuffer(2, kBlock);
    for (int b = 0; b < nBlocks; ++b)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            const float v = 0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * phase));
            tb.ch(0)[i] = v;
            tb.ch(1)[i] = v;
            phase += 500.0 / 44100.0;
            if (phase >= 1.0) phase -= 1.0;
        }
        sat.process(tb.view());
    }
    return std::vector<float>(tb.ch(0), tb.ch(0) + kBlock);
}

double satRmsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double d = static_cast<double>(a[i]) - b[i];
        acc += d * d;
    }
    return std::sqrt(acc / (a.empty() ? 1 : a.size()));
}

} // namespace

DSPARK_TEST(Saturation_algorithm_switch_reverts_cleanly)
{
    // Request B mid-crossfade, then request the still-active algorithm back.
    // The old state machine ignored the second edit (active == requested was a
    // no-op) and completed the fade to B: the WRONG algorithm played forever
    // while getCurrentAlgorithm() reported the requested one. Measured with
    // the old header: distance to the requested SoftClip tail 0.94 RMS
    // (essentially all Downsample); fixed: 0.0015.
    using A = Saturation<float>::Algorithm;
    Saturation<float> subject, refSoft, refDown;
    for (auto* s : { &subject, &refSoft, &refDown })
    {
        s->prepare(spec(44100.0, 256, 2));
        s->setDrive(30.0f);
        s->setMix(1.0f);
    }
    refDown.setAlgorithm(A::Downsample);

    double phS = 0.0, phA = 0.0, phB = 0.0;
    satRunSine(subject, phS, 1); satRunSine(refSoft, phA, 1); satRunSine(refDown, phB, 1);
    subject.setAlgorithm(A::Downsample);              // crossfade starts (10 ms > one block)
    satRunSine(subject, phS, 1); satRunSine(refSoft, phA, 1); satRunSine(refDown, phB, 1);
    subject.setAlgorithm(A::SoftClip);                // revert mid-fade
    const auto tailSub  = satRunSine(subject, phS, 40);
    const auto tailSoft = satRunSine(refSoft, phA, 40);
    const auto tailDown = satRunSine(refDown, phB, 40);

    EXPECT_LT(satRmsDiff(tailSub, tailSoft), 0.01);
    EXPECT_GT(satRmsDiff(tailSub, tailDown), 0.5);
    EXPECT_TRUE(subject.getCurrentAlgorithm() == A::SoftClip);
}

DSPARK_TEST(Saturation_reset_completes_pending_switch)
{
    // reset() during a crossfade used to park the crossfader while leaving
    // next_ armed: the OLD algorithm kept playing, and because next_ still
    // pointed at the pending one, a later request of that same algorithm was
    // silently ignored forever. reset() now promotes the pending algorithm
    // (every algorithm's state is cleared anyway), so what plays after a
    // reset is what was last requested.
    using A = Saturation<float>::Algorithm;
    Saturation<float> subject, refDown;
    for (auto* s : { &subject, &refDown })
    {
        s->prepare(spec(44100.0, 256, 2));
        s->setDrive(30.0f);
        s->setMix(1.0f);
    }
    refDown.setAlgorithm(A::Downsample);

    double phS = 0.0, phB = 0.0;
    subject.setAlgorithm(A::Downsample);
    satRunSine(subject, phS, 1);                      // fade under way
    subject.reset();
    refDown.reset();
    phS = phB = 0.0;                                  // restart the stimulus for both

    const auto tailSub  = satRunSine(subject, phS, 40);
    const auto tailDown = satRunSine(refDown, phB, 40);
    EXPECT_LT(satRmsDiff(tailSub, tailDown), 0.01);   // measured 0.0011 (old header: 0.92)
}

namespace {

// Steady-state gain (dB, relative to the input) of a sine through SoftClip at
// -24 dB drive (linear regime) with the post tilt EQ configured.
float satTiltGainDb(float freq, float tiltDb)
{
    constexpr int N = 16384, kSkip = 4096;
    Saturation<float> sat;
    sat.prepare(spec(44100.0, N, 2));
    sat.setDrive(-24.0f);
    sat.setMix(1.0f);
    sat.setPostFilterTilt(1000.0f, tiltDb);
    auto tb = makeStereoBuffer(N);
    tb.fillSine(freq, 44100.0f, 0.01f);
    sat.process(tb.view());
    double acc = 0.0;
    for (int i = kSkip; i < N; ++i)
        acc += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i];
    const double rms = std::sqrt(acc / (N - kSkip));
    return 20.0f * static_cast<float>(std::log10(rms / (0.01 / std::sqrt(2.0)) + 1e-30));
}

} // namespace

DSPARK_TEST(Saturation_post_tilt_is_a_tilt)
{
    // setPostFilterTilt documents a tilt (brighten/darken around the pivot)
    // but was wired to makePeak: a bell AT the pivot (+6 dB at 1 kHz, ~0 at
    // the extremes; measured span 200 Hz -> 8 kHz of -0.4 dB). With makeTilt
    // the response is the documented shelf: pivot unity, -3 dB towards DC,
    // +3 dB towards Nyquist (measured span +5.7 dB).
    const float ref  = satTiltGainDb(1000.0f, 0.0f);  // linear-path reference
    const float g200 = satTiltGainDb(200.0f,  6.0f) - ref;
    const float g1k  = satTiltGainDb(1000.0f, 6.0f) - ref;
    const float g8k  = satTiltGainDb(8000.0f, 6.0f) - ref;

    EXPECT_NEAR(g1k, 0.0f, 0.75f);   // pivot stays unity (the old bell: +6 dB)
    EXPECT_GT(g8k - g200, 3.0f);     // genuine spectral tilt across the pivot
    EXPECT_GT(g8k, 1.0f);            // brightens above
    EXPECT_LT(g200, -1.0f);          // darkens below
}

DSPARK_TEST(Saturation_invalid_inputs_are_ignored)
{
    // NaN/Inf setters, an invalid prepare() and an out-of-range Algorithm id
    // must all be inert. The old header poisoned every smoother (the full
    // block went NaN), fed numChannels=-3 straight into AudioBuffer::resize,
    // and indexed pool_[97]: an access violation (0xC0000005 measured) from a
    // garbage vtable pointer.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    Saturation<float> subject, twin;
    for (auto* s : { &subject, &twin })
    {
        s->prepare(spec(44100.0, 256, 2));
        s->setDrive(12.0f);
        s->setMix(1.0f);
    }

    subject.setDrive(nan);
    subject.setCharacter(inf);
    subject.setMix(nan);
    subject.setPreFilterHpFrequency(nan);
    subject.setPostFilterTilt(nan, nan);
    subject.setOutputGain(nan);
    subject.setAnalogDrift(nan);
    subject.setSlewSensitivity(nan);
    AudioSpec bad;
    bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
    bad.maxBlockSize = 0;
    bad.numChannels = -3;
    subject.prepare(bad);
    subject.setAlgorithm(static_cast<Saturation<float>::Algorithm>(97)); // clamps to the last id
    twin.setAlgorithm(Saturation<float>::Algorithm::MultiStage);

    double phS = 0.0, phT = 0.0;
    const auto tailSub  = satRunSine(subject, phS, 10);
    const auto tailTwin = satRunSine(twin, phT, 10);

    EXPECT_NO_NAN(tailSub.data(), static_cast<int>(tailSub.size()));
    float maxDiff = 0.0f;
    for (size_t i = 0; i < tailSub.size(); ++i)
        maxDiff = std::max(maxDiff, std::abs(tailSub[i] - tailTwin[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // bit-identical: the invalid edits were fully inert

    // PDC contract: the framework-wide latency name reports the same figure.
    Saturation<float> os;
    os.setOversampling(4);
    os.prepare(spec(44100.0, 512, 2));
    EXPECT_EQ(os.getLatency(), os.getLatencySamples());
    EXPECT_GT(os.getLatency(), 0);
}

DSPARK_TEST(Saturation_extra_channels_are_safe)
{
    // A view with more channels than the prepared spec used to index
    // driftBuffer_/tempBuffer_ (sized to the spec) out of bounds as soon as
    // drift or an algorithm crossfade was active: assert in debug, OOB
    // read/write in release. Channels beyond the spec now bypass the
    // saturation stage and must come out finite.
    using A = Saturation<float>::Algorithm;
    Saturation<float> sat;
    sat.prepare(spec(44100.0, 256, 2));      // prepared for TWO channels
    sat.setDrive(12.0f);
    sat.setMix(1.0f);
    sat.setAnalogDrift(1.0f);                // exercises driftBuffer_
    sat.setAlgorithm(A::Tube);               // exercises tempBuffer_ (crossfade)

    auto tb = makeBuffer(4, 256);            // FOUR-channel view
    for (int b = 0; b < 20; ++b)
    {
        tb.fillSine(500.0f, 44100.0f, 0.5f);
        sat.process(tb.view());
    }
    for (int ch = 0; ch < 4; ++ch)
    {
        EXPECT_NO_NAN(tb.ch(ch), 256);
        EXPECT_BOUNDED(tb.ch(ch), 256, -2.0f, 2.0f);
    }
}

DSPARK_TEST(Saturation_dc_blocking_works_at_high_rates)
{
    // The internal DC blocker was a private Biquad<float>: the same 5 Hz design
    // DCBlocker uses, but run in float, where the denominator at DC collapses
    // at high rates (see DCBlocker_rejection_is_rate_independent).
    // Measured at 192 kHz with 0.3 of DC at the input and dcBlocking on: +0.0739
    // of DC left at the output and the peak inflated from 0.664 to 0.748. It now
    // shares the DCBlocker core, so the offset is gone at any rate.
    constexpr double kTwoPi = 6.283185307179586;
    auto residualDc = [](double rate) {
        Saturation<float> sat;
        sat.setOversampling(2);
        sat.prepare(spec(rate, 512, 2));
        sat.setAlgorithm(Saturation<float>::Algorithm::SoftClip);
        sat.setDrive(6.0f);
        sat.setDcBlocking(true);

        const int total = static_cast<int>(rate);                  // 1 s
        const int start = total - static_cast<int>(rate * 0.25);   // settled tail
        auto tb = makeBuffer(2, 512);
        double acc = 0.0;
        int count = 0;
        for (int i = 0; i < total; i += 512)
        {
            for (int k = 0; k < 512; ++k)
            {
                const auto x = static_cast<float>(
                    0.3 + 0.4 * std::sin(kTwoPi * 1000.0 * (i + k) / rate));
                tb.ch(0)[k] = tb.ch(1)[k] = x;
            }
            sat.process(tb.view());
            for (int k = 0; k < 512; ++k)
                if (i + k >= start) { acc += static_cast<double>(tb.ch(0)[k]); ++count; }
        }
        return acc / count;
    };

    EXPECT_LT(std::abs(residualDc(48000.0)), 1e-3);
    EXPECT_LT(std::abs(residualDc(192000.0)), 1e-3);   // old header: +0.0739
}

DSPARK_TEST(DCBlocker_prepare_keeps_configured_cutoff)
{
    // prepare(AudioSpec) silently restored the 5 Hz default, so a host
    // re-activating the plugin (or the Lab changing device) threw away a
    // configured cutoff while the UI kept showing it (Limiter had the same
    // bug with its lookahead).
    DCBlocker<float> d;
    d.prepare(48000.0, 2, 5.0);
    d.setCutoff(80.0f);
    d.prepare(spec(96000.0, 512, 2));                 // host re-activation
    EXPECT_NEAR(static_cast<double>(d.getCutoff()), 80.0, 1e-6);

    // ... and the audio agrees: 80 Hz is 3 dB down at its own cutoff, while a
    // filter that fell back to 5 Hz would pass it untouched.
    constexpr int n = 32768;
    auto tb = makeBuffer(1, n);
    for (int i = 0; i < n; ++i)
        tb.ch(0)[i] = static_cast<float>(0.5 * std::sin(6.283185307179586 * 80.0 * i / 96000.0));
    d.processBlock(tb.view());
    double acc = 0.0;
    for (int i = n / 2; i < n; ++i) acc += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i];
    const double db = 20.0 * std::log10(std::sqrt(acc / (n / 2)) / (0.5 / std::sqrt(2.0)));
    EXPECT_NEAR(db, -3.01, 0.3);                      // old header: ~0 dB (5 Hz)
}

DSPARK_TEST(DCBlocker_live_order_change_rebuilds_and_invalid_inputs_ignored)
{
    // Rebuilds were gated on CUTOFF changes only: a live setOrder(8) ran a
    // broken hybrid (first stage with the old order's Q + identity stages)
    // until the cutoff happened to move. Measured 2 octaves below a 100 Hz
    // cutoff: -24 dB (hybrid) vs -94.5 dB (true order 8). Also: setCutoff(NaN)
    // passed std::max(NaN, 1) and poisoned every coefficient (100% non-finite
    // output), and the per-channel APIs indexed the state arrays out of
    // bounds with wild channel indices.
    auto toneGainDb = [](DCBlocker<float>& d, float freq) {
        constexpr int n = 32768, kSkip = n / 2;
        auto tb = makeBuffer(1, n);
        tb.fillSine(freq, 44100.0f, 0.5f);
        d.processBlock(tb.view());
        double acc = 0.0;
        for (int i = kSkip; i < n; ++i)
            acc += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i];
        return 20.0f * static_cast<float>(std::log10(std::sqrt(acc / (n - kSkip)) / (0.5 / std::sqrt(2.0)) + 1e-30));
    };

    DCBlocker<float> d;
    d.prepare(44100.0, 1, 100.0);          // order 1 baked in
    (void)toneGainDb(d, 1000.0f);
    d.setOrder(8);                         // LIVE order change
    EXPECT_LT(toneGainDb(d, 25.0f), -60.0f); // true order-8 slope (old: -24)

    DCBlocker<float> subject, twin;
    subject.prepare(44100.0, 1, 5.0);
    twin.prepare(44100.0, 1, 5.0);
    subject.setCutoff(std::numeric_limits<float>::quiet_NaN());
    subject.prepare(std::numeric_limits<double>::quiet_NaN());
    auto ta = makeBuffer(1, 4096);
    auto tb2 = makeBuffer(1, 4096);
    ta.fillDC(0.25f);
    tb2.fillDC(0.25f);
    subject.processBlock(ta.view());
    twin.processBlock(tb2.view());
    EXPECT_NO_NAN(ta.ch(0), 4096);
    float maxDiff = 0.0f;
    for (int i = 0; i < 4096; ++i)
        maxDiff = std::max(maxDiff, std::abs(ta.ch(0)[i] - tb2.ch(0)[i]));
    EXPECT_TRUE(maxDiff == 0.0f);

    // Wild channel indices: pass-through / no-op, state arrays untouched.
    float x[16] = {};
    subject.processBlock(99, x, 16);
    subject.processBlock(-3, x, 16);
    float acc = 0.0f;
    for (int i = 0; i < 100; ++i) acc += subject.processSample(99, 0.5f);
    EXPECT_NEAR(acc, 50.0f, 1e-5f);        // old header: OOB state gave 48.3
}

// ============================================================================
// DCBlocker
// ============================================================================

DSPARK_TEST(DCBlocker_removes_DC)
{
    DCBlocker<float> dc;
    dc.prepare(44100.0);

    constexpr int N = 44100; // 1 second
    auto tb = makeMonoBuffer(N);

    // DC + sine
    for (int i = 0; i < N; ++i)
        tb.ch(0)[i] = 0.5f + 0.5f * std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);

    dc.processBlock(tb.view());

    // Measure DC content in the latter half (after settling)
    float sum = 0.0f;
    int start = N / 2;
    for (int i = start; i < N; ++i)
        sum += tb.ch(0)[i];
    float avgDC = sum / static_cast<float>(N - start);

    // DC should be near zero
    EXPECT_NEAR(avgDC, 0.0f, 0.02f);
}

DSPARK_TEST(DCBlocker_passes_AC)
{
    DCBlocker<float> dc;
    dc.prepare(44100.0);

    constexpr int N = 8192;
    auto tb = makeMonoBuffer(N);
    tb.fillSine(440.0f, 44100.0f);

    float peakBefore = measurePeak(tb.ch(0), N);
    dc.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 2048, N - 2048);

    // AC signal should pass through mostly unchanged
    EXPECT_GT(peakAfter, peakBefore * 0.9f);
}

DSPARK_TEST(DCBlocker_silence)
{
    DCBlocker<float> dc;
    dc.prepare(44100.0);

    auto tb = makeMonoBuffer(512);
    tb.fillSilence();
    dc.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-10f);
}

DSPARK_TEST(DCBlocker_rejection_is_rate_independent)
{
    // The filter core used to run in the sample type. With float, the
    // denominator evaluated at DC (1 + a1 + a2 = (2 - 2*cos w0)/a0, i.e. 2.7e-8
    // at 5 Hz / 192 kHz) is a cancellation of terms of size 2: it rounded to
    // 4.77e-7 at 44.1 kHz (6 % off) and to EXACTLY 0 at 192 and 768 kHz, which
    // puts a realised pole ON the unit circle. From there every rounding error
    // of the recursion was integrated forever instead of decaying and the DC
    // blocker SOURCED DC. Measured with the old header on a DC-free 1 kHz tone
    // at -6 dBFS: -0.72 of DC at 768 kHz order 2 (output peaking at 1.42 from a
    // 0.5 input), -0.059 at 192 kHz order 4, and ~1e-6 on the 1-pole path;
    // a 0.5 DC input left +0.0722 at 192 kHz order 2, and the realised DC gain
    // (sum of the impulse response) reached -53.7 at 768 kHz order 4.
    constexpr double kTwoPi = 6.283185307179586;

    auto settledDc = [](double rate, int order, bool dcInput) {
        DCBlocker<float> d;
        d.setOrder(order);
        d.prepare(rate, 1, 5.0);

        const int total = static_cast<int>(rate * 2.0);
        const int start = total - static_cast<int>(rate * 0.5);  // last 0.5 s
        constexpr int kBlock = 512;
        std::vector<float> buf(kBlock);
        double acc = 0.0;
        int count = 0;
        for (int i = 0; i < total; i += kBlock)
        {
            const int n = std::min(kBlock, total - i);
            for (int k = 0; k < n; ++k)
                buf[static_cast<size_t>(k)] = dcInput
                    ? 0.5f
                    : static_cast<float>(0.5 * std::sin(kTwoPi * 1000.0 * (i + k) / rate));
            float* ptrs[1] = { buf.data() };
            d.processBlock(AudioBufferView<float>(ptrs, 1, n));
            for (int k = 0; k < n; ++k)
                if (i + k >= start)
                {
                    acc += static_cast<double>(buf[static_cast<size_t>(k)]);
                    ++count;
                }
        }
        return acc / count;
    };

    for (double rate : { 44100.0, 48000.0, 96000.0, 192000.0, 384000.0, 768000.0 })
        for (int order : { 1, 2, 4 })
        {
            EXPECT_LT(std::abs(settledDc(rate, order, false)), 1e-7);  // DC-free in, DC-free out
            EXPECT_LT(std::abs(settledDc(rate, order, true)), 1e-7);   // steady DC decays away
        }

    // Identical channels must stay bit-identical (per-channel state only).
    DCBlocker<float> st;
    st.setOrder(4);
    st.prepare(192000.0, 2, 5.0);
    auto tb = makeBuffer(2, 4096);
    for (int i = 0; i < 4096; ++i)
    {
        const auto x = static_cast<float>(0.4 * std::sin(kTwoPi * 997.0 * i / 192000.0) + 0.05);
        tb.ch(0)[i] = tb.ch(1)[i] = x;
    }
    st.processBlock(tb.view());
    int mismatches = 0;
    for (int i = 0; i < 4096; ++i)
        if (std::memcmp(&tb.ch(0)[i], &tb.ch(1)[i], sizeof(float)) != 0) ++mismatches;
    EXPECT_EQ(mismatches, 0);
}

DSPARK_TEST(DCBlocker_response_matches_butterworth_design)
{
    // Same root cause seen from the response side: quantising a1/a2 to float
    // misplaces poles that sit 2.6e-5 away from the unit circle. The order-4
    // cascade at 48 kHz measured +0.534 dB at 10 Hz and +0.182 dB at 20 Hz --
    // a passband bump a Butterworth high-pass cannot have (its magnitude is
    // monotonic and never exceeds unity). The double core now lands on the
    // analog prototype to better than 0.001 dB.
    constexpr double kTwoPi = 6.283185307179586;

    auto gainDb = [](double rate, int order, double freq) {
        DCBlocker<float> d;
        d.setOrder(order);
        d.prepare(rate, 1, 5.0);

        const int settle = static_cast<int>(rate);                            // 1 s
        const int window = static_cast<int>(std::lround(10.0 * rate / freq)); // 10 whole periods
        const double w = kTwoPi * freq / rate;
        constexpr int kBlock = 512;
        std::vector<float> buf(kBlock);
        double si = 0.0, co = 0.0;
        for (int i = 0; i < settle + window; i += kBlock)
        {
            const int n = std::min(kBlock, settle + window - i);
            for (int k = 0; k < n; ++k)
                buf[static_cast<size_t>(k)] = static_cast<float>(0.5 * std::sin(w * (i + k)));
            float* ptrs[1] = { buf.data() };
            d.processBlock(AudioBufferView<float>(ptrs, 1, n));
            for (int k = 0; k < n; ++k)
                if (i + k >= settle)
                {
                    const double ph = w * (i + k);
                    const double y  = static_cast<double>(buf[static_cast<size_t>(k)]);
                    si += y * std::sin(ph);
                    co += y * std::cos(ph);
                }
        }
        const double amp = std::sqrt(si * si + co * co) * 2.0 / window;
        return 20.0 * std::log10(amp / 0.5);
    };

    // |H(f)|^2 = (f/fc)^2n / (1 + (f/fc)^2n): the Butterworth prototype the
    // cascade is designed from (warping is negligible this far below Nyquist).
    auto ideal = [](int order, double freq) {
        const double r = std::pow(freq / 5.0, 2.0 * order);
        return 10.0 * std::log10(r / (1.0 + r));
    };

    for (double rate : { 44100.0, 48000.0 })
        for (int order : { 2, 4 })
            for (double f : { 10.0, 20.0, 50.0, 100.0 })
                EXPECT_NEAR(gainDb(rate, order, f), ideal(order, f), 0.02);
}

DSPARK_TEST(DCBlocker_processSample_matches_processBlock)
{
    // The per-sample path now polls the published parameters like the block
    // path does (it used to honour setOrder() immediately - running stages
    // that held no coefficients - and to ignore setCutoff() entirely).
    constexpr int kN = 6000;
    constexpr int kBlock = 512;

    DCBlocker<float> blockPath, samplePath;
    blockPath.setOrder(4);
    samplePath.setOrder(4);
    blockPath.prepare(96000.0, 1, 5.0);
    samplePath.prepare(96000.0, 1, 5.0);

    std::vector<float> src(kN), a(kN), b(kN);
    uint32_t rng = 22222u;
    for (int i = 0; i < kN; ++i)
    {
        rng = rng * 1664525u + 1013904223u;
        src[static_cast<size_t>(i)] = static_cast<float>(static_cast<int32_t>(rng) / 2147483648.0) * 0.4f + 0.1f;
    }
    a = src;
    b = src;

    for (int i = 0; i < kN; i += kBlock)
    {
        const int n = std::min(kBlock, kN - i);
        if (i == 2 * kBlock)    // live cutoff change, on a block boundary for both
        {
            blockPath.setCutoff(20.0f);
            samplePath.setCutoff(20.0f);
        }
        float* ptrs[1] = { a.data() + i };
        blockPath.processBlock(AudioBufferView<float>(ptrs, 1, n));
        for (int k = 0; k < n; ++k)
            b[static_cast<size_t>(i + k)] = samplePath.processSample(0, b[static_cast<size_t>(i + k)]);
    }

    int mismatches = 0;
    for (int i = 0; i < kN; ++i)
        if (std::memcmp(&a[static_cast<size_t>(i)], &b[static_cast<size_t>(i)], sizeof(float)) != 0) ++mismatches;
    EXPECT_EQ(mismatches, 0);
    // The cutoff change must actually have reached both paths.
    EXPECT_NEAR(static_cast<double>(blockPath.getCutoff()), 20.0, 1e-6);
}

// ============================================================================
// Tilt EQ (via Biquad + Equalizer)
// ============================================================================

DSPARK_TEST(Biquad_Tilt_boosts_highs_cuts_lows)
{
    // Tilt +6 dB with pivot at 1000 Hz
    auto coeffs = BiquadCoeffs::makeTilt(44100.0, 1000.0, 6.0);
    Biquad<float> bqLow, bqHigh;
    bqLow.setCoeffs(coeffs);
    bqHigh.setCoeffs(coeffs);

    auto measureResponse = [](Biquad<float>& bq, float freq, float sr) -> float
    {
        for (int i = 0; i < 2048; ++i)
        {
            float in = std::sin(twoPi<float> * freq * static_cast<float>(i) / sr);
            (void)bq.processSample(in, 0);
        }
        float peak = 0.0f;
        for (int i = 0; i < 8192; ++i)
        {
            float in = std::sin(twoPi<float> * freq * static_cast<float>(2048 + i) / sr);
            float out = bq.processSample(in, 0);
            float a = std::abs(out);
            if (a > peak) peak = a;
        }
        return peak;
    };

    float respLow = measureResponse(bqLow, 100.0f, 44100.0f);
    float respHigh = measureResponse(bqHigh, 10000.0f, 44100.0f);

    // Tilt +6 dB: high freq should be louder than low freq
    EXPECT_GT(respHigh, respLow);
    // 100 Hz should be attenuated (below unity)
    EXPECT_LT(respLow, 1.0f);
    // 10 kHz should be boosted (above unity)
    EXPECT_GT(respHigh, 1.0f);
}

DSPARK_TEST(Biquad_Tilt_0dB_passthrough)
{
    auto coeffs = BiquadCoeffs::makeTilt(44100.0, 1000.0, 0.0);
    Biquad<float> bq;
    bq.setCoeffs(coeffs);

    // Feed sine and measure - should be nearly unity
    for (int i = 0; i < 2048; ++i)
    {
        float in = std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);
        (void)bq.processSample(in, 0);
    }
    float peak = 0.0f;
    for (int i = 0; i < 8192; ++i)
    {
        float in = std::sin(twoPi<float> * 440.0f * static_cast<float>(2048 + i) / 44100.0f);
        float out = bq.processSample(in, 0);
        float a = std::abs(out);
        if (a > peak) peak = a;
    }
    EXPECT_NEAR(peak, 1.0f, 0.05f);
}

DSPARK_TEST(Equalizer_Tilt_band_works)
{
    // Use two separate instances to avoid smoother reset issues
    auto s = defaultSpec();

    typename Equalizer<float>::BandConfig cfg;
    cfg.frequency = 1000.0f;
    cfg.gain      = 6.0f;
    cfg.type      = Equalizer<float>::BandType::Tilt;

    // Measure low frequency (100 Hz)
    Equalizer<float> eqLow;
    eqLow.prepare(s);
    eqLow.setNumBands(1);
    eqLow.setBand(0, cfg);

    // Process several blocks so smoothers settle
    auto tbLow = makeStereoBuffer(4096);
    for (int i = 0; i < 4; ++i)
    {
        tbLow.fillSine(100.0f, 44100.0f);
        eqLow.processBlock(tbLow.view());
    }
    float peakLow = measurePeak(tbLow.ch(0) + 1024, 2048);

    // Measure high frequency (10 kHz)
    Equalizer<float> eqHigh;
    eqHigh.prepare(s);
    eqHigh.setNumBands(1);
    eqHigh.setBand(0, cfg);

    auto tbHigh = makeStereoBuffer(4096);
    for (int i = 0; i < 4; ++i)
    {
        tbHigh.fillSine(10000.0f, 44100.0f);
        eqHigh.processBlock(tbHigh.view());
    }
    float peakHigh = measurePeak(tbHigh.ch(0) + 1024, 2048);

    EXPECT_GT(peakHigh, peakLow);
}


// ============================================================================
// Hysteresis (Jiles-Atherton core)
// ============================================================================

namespace {

struct HystLoop
{
    std::vector<double> h, m;
};

HystLoop runHystLoop(double kParam, double cParam, double amp)
{
    Hysteresis<double> hyst;
    hyst.prepare(96000.0);
    hyst.setParameters(3.5e5, 2.2e4, 1.6e-3, kParam, cParam);

    const int perCycle = 960;   // 100 Hz at 96k
    HystLoop loop;
    loop.h.resize(static_cast<size_t>(perCycle));
    loop.m.resize(static_cast<size_t>(perCycle));
    for (int cyc = 0; cyc < 12; ++cyc)
        for (int i = 0; i < perCycle; ++i)
        {
            const double h = amp * std::sin(2.0 * 3.14159265358979 * i / perCycle);
            const double m = hyst.processSample(h);
            if (cyc == 11)
            {
                loop.h[static_cast<size_t>(i)] = h;
                loop.m[static_cast<size_t>(i)] = m;
            }
        }
    return loop;
}

double hystLoopArea(const HystLoop& loop)
{
    double area = 0.0;
    const size_t n = loop.h.size();
    for (size_t i = 0; i < n; ++i)
    {
        const size_t j = (i + 1) % n;
        area += 0.5 * (loop.m[i] + loop.m[j]) * (loop.h[j] - loop.h[i]);
    }
    return std::abs(area);
}

} // namespace

DSPARK_TEST(Hysteresis_BH_loop_has_odd_symmetry)
{
    const auto loop = runHystLoop(2.7e4, 0.17, 3.0 * 2.2e4);
    double worst = 0.0;
    const size_t n = loop.m.size();
    for (size_t i = 0; i < n / 2; ++i)
        worst = std::max(worst, std::abs(loop.m[i] + loop.m[i + n / 2]) / 3.5e5);
    EXPECT_LT(worst, 0.005);   // steady-state loop is odd-symmetric
    for (double m : loop.m)
        EXPECT_TRUE(std::isfinite(m) && std::abs(m) <= 3.5e5);
}

DSPARK_TEST(Hysteresis_loop_area_grows_with_k_and_shrinks_with_c)
{
    // Loop area is the per-cycle magnetic loss: k widens the loop, c -> 1
    // collapses it toward the reversible anhysteretic curve.
    const double amp = 3.0 * 2.2e4;
    EXPECT_LT(hystLoopArea(runHystLoop(1.0e4, 0.17, amp)),
              hystLoopArea(runHystLoop(2.7e4, 0.17, amp)));
    EXPECT_LT(hystLoopArea(runHystLoop(2.7e4, 0.90, amp)),
              hystLoopArea(runHystLoop(2.7e4, 0.17, amp)));
}

DSPARK_TEST(Hysteresis_shows_remanence_after_field_removal)
{
    Hysteresis<double> hyst;
    hyst.prepare(96000.0);
    hyst.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);

    const int n = 9600;
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = hyst.processSample(3.0 * 2.2e4 * i / n);
    for (int i = 0; i < n; ++i)
        m = hyst.processSample(3.0 * 2.2e4 * (1.0 - static_cast<double>(i) / n));
    for (int i = 0; i < 4800; ++i)
        m = hyst.processSample(0.0);
    EXPECT_GT(m / 3.5e5, 0.02);   // magnetic memory remains at zero field
}

// Parameter sets that violate the JA stability condition c*alpha*Ms/(3a) < 1
// must degrade gracefully: the reversible-branch denominator sweeps through
// zero as Q moves, and the guard bounds the resulting dM/dt spike while the
// physical clamp keeps |M| <= Ms. Pins finite output for a full second.
DSPARK_TEST(Hysteresis_survives_unstable_parameter_sets)
{
    Hysteresis<double> hyst;
    hyst.prepare(48000.0);
    // alpha = 1, c = 0.9, Ms/a huge: c*alpha*Ms/(3a) = 0.9 * 350 >> 1.
    hyst.setParameters(3.5e5, 3.3e2, 1.0, 2.7e4, 0.9);

    bool finite = true;
    double m = 0.0;
    for (int i = 0; i < 48000; ++i)
    {
        const double h = 5.0 * 2.2e4 * std::sin(2.0 * 3.14159265358979 * 137.0 * i / 48000.0);
        m = hyst.processSample(h);
        if (!std::isfinite(m) || std::abs(m) > 3.5e5) { finite = false; break; }
    }
    EXPECT_TRUE(finite);
}

// The susceptibility getter (used by TapeMachine for exact small-signal
// makeup) must match the measured small-signal slope dM/dH.
DSPARK_TEST(Hysteresis_small_signal_susceptibility_matches_measurement)
{
    Hysteresis<double> hyst;
    hyst.prepare(96000.0);
    hyst.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
    const double chi = hyst.getSmallSignalSusceptibility();

    // Tiny sine, well inside the reversible region; measure output amplitude
    // over the settled tail.
    const double amp = 1e-3 * 2.2e4;
    double peak = 0.0;
    for (int i = 0; i < 9600; ++i)
    {
        const double h = amp * std::sin(2.0 * 3.14159265358979 * 100.0 * i / 96000.0);
        const double m = hyst.processSample(h);
        if (i > 4800) peak = std::max(peak, std::abs(m));
    }
    EXPECT_NEAR(peak / amp, chi, 0.05 * chi);   // within 5%
}

// ============================================================================
// TapeMachine
// ============================================================================

namespace {

std::vector<float> runTapeSine(dspark::TapeMachine<float>& tape, double freq,
                               float amp, double seconds)
{
    const int total = static_cast<int>(48000.0 * seconds / 512) * 512;
    std::vector<float> out(static_cast<size_t>(total));
    auto buf = makeStereoBuffer(512);
    int n = 0;
    for (int b = 0; b < total / 512; ++b)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            const float v = amp * static_cast<float>(
                std::sin(2.0 * 3.14159265358979 * freq * n / 48000.0));
            buf.ch(0)[i] = v;
            buf.ch(1)[i] = v;
        }
        tape.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(b * 512 + i)] = buf.ch(0)[i];
    }
    return out;
}

double tapeToneMag(const std::vector<float>& x, double freq)
{
    const size_t from = x.size() / 2;
    const size_t n = x.size() - from;
    double re = 0, im = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const double ph = 2.0 * 3.14159265358979 * freq * static_cast<double>(i) / 48000.0;
        re += static_cast<double>(x[from + i]) * std::cos(ph);
        im += static_cast<double>(x[from + i]) * std::sin(ph);
    }
    return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(n);
}

void neutralTape(TapeMachine<float>& tape)
{
    tape.prepare(spec(48000.0, 512, 2));
    tape.setLossEffects(0.0f);
    tape.setHeadBump(0.0f);
    tape.setWowFlutter(0.0f);
}

} // namespace

DSPARK_TEST(TapeMachine_response_flat_at_reference_level)
{
    // Record EQ must be the exact inverse of playback EQ: any mismatch shows
    // up as tilt across the band at the calibration level.
    for (double f : { 100.0, 1000.0, 8000.0, 14000.0 })
    {
        TapeMachine<float> tape;
        neutralTape(tape);
        auto out = runTapeSine(tape, f, 0.25f, 1.0);
        const double db = 20.0 * std::log10(tapeToneMag(out, f) / 0.25);
        EXPECT_NEAR(db, 0.0, 0.75);
    }
}

DSPARK_TEST(TapeMachine_low_end_is_aligned)
{
    // The JA loop settles on a history-dependent branch whose LF gain used to
    // exceed the 1 kHz gain by +1.8..+2.8 dB below 100 Hz at programme level
    // (unremovable by any control - user-reported bass bloat), and the chain
    // had no subsonic roll-off, so loop-remanence DC (-0.034 measured) passed
    // straight through. Pinned here: aligned low end and blocked DC.
    for (double f : { 40.0, 60.0, 90.0 })
    {
        TapeMachine<float> tape;
        neutralTape(tape);
        auto out = runTapeSine(tape, f, 0.25f, 1.5);
        const double db = 20.0 * std::log10(tapeToneMag(out, f) / 0.25);
        EXPECT_NEAR(db, 0.0, 1.0);   // old header: +1.8 dB at 40 Hz
    }
    {
        TapeMachine<float> tape;
        neutralTape(tape);
        auto out = runTapeSine(tape, 25.0, 0.25f, 2.0);
        double dc = 0.0;
        const size_t from = out.size() / 2;
        for (size_t i = from; i < out.size(); ++i) dc += out[i];
        dc /= static_cast<double>(out.size() - from);
        EXPECT_LT(std::fabs(dc), 0.005); // old header: 0.034 standing DC
    }
}

DSPARK_TEST(TapeMachine_distortion_is_odd_dominant_and_grows_with_drive)
{
    // With real push-pull AC bias the even orders cancel (H2 measured 40+ dB
    // below H3) and a calibrated machine at reference level is CLEAN (~0.1%);
    // saturation is the drive knob's job, reaching honest tape crunch at the
    // top of the range.
    double thdPrev = -1.0;
    for (float drive : { 0.0f, 6.0f, 12.0f, 24.0f })
    {
        TapeMachine<float> tape;
        neutralTape(tape);
        tape.setDrive(drive);
        auto out = runTapeSine(tape, 1000.0, 0.5f, 1.0);
        const double h1 = tapeToneMag(out, 1000.0);
        const double h2 = tapeToneMag(out, 2000.0);
        const double h3 = tapeToneMag(out, 3000.0);
        EXPECT_GT(h3, h2 * 3.0);            // tape symmetry: odd >> even
        const double thd = h3 / h1;
        EXPECT_GT(thd, thdPrev);            // more drive, more saturation
        thdPrev = thd;
    }
    EXPECT_GT(thdPrev, 0.04);               // real crunch at +24 dB drive
    EXPECT_LT(thdPrev, 0.30);               // but not broken
}

DSPARK_TEST(TapeMachine_wow_flutter_modulates_pitch)
{
    auto devFor = [](float wf) {
        TapeMachine<float> tape;
        neutralTape(tape);
        tape.setDrive(-6.0f);
        tape.setWowFlutter(wf);
        auto out = runTapeSine(tape, 3000.0, 0.25f, 4.0);

        double prevCross = -1.0, minP = 1e9, maxP = 0.0;
        for (size_t i = out.size() / 2; i + 1 < out.size(); ++i)
        {
            if (out[i] <= 0.0f && out[i + 1] > 0.0f)
            {
                const double fr = static_cast<double>(out[i])
                                / (static_cast<double>(out[i]) - out[i + 1]);
                const double tc = static_cast<double>(i) + fr;
                if (prevCross >= 0.0)
                {
                    minP = std::min(minP, tc - prevCross);
                    maxP = std::max(maxP, tc - prevCross);
                }
                prevCross = tc;
            }
        }
        return (maxP - minP) / (48000.0 / 3000.0);
    };
    EXPECT_LT(devFor(0.0f), 1e-4);     // transport off: no modulation
    const double dev = devFor(1.0f);
    EXPECT_GT(dev, 0.002);             // full depth: clearly audible wow
    EXPECT_LT(dev, 0.02);              // but within worn-machine territory
}

DSPARK_TEST(TapeMachine_latency_matches_reported)
{
    TapeMachine<float> tape;
    neutralTape(tape);
    tape.setDrive(-12.0f);
    const int latency = tape.getLatency();

    auto buf = makeStereoBuffer(512);
    std::vector<float> out;
    for (int b = 0; b < 8; ++b)
    {
        buf.fillSilence();
        if (b == 0) buf.ch(0)[0] = buf.ch(1)[0] = 0.05f;
        tape.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out.push_back(buf.ch(0)[i]);
    }
    int peakAt = 0;
    float peak = 0;
    for (size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(out[i]));
        if (std::abs(out[i]) > peak) { peak = std::abs(out[i]); peakAt = static_cast<int>(i); }
    }
    EXPECT_NEAR(static_cast<double>(peakAt), static_cast<double>(latency), 1.0);
}


// ============================================================================
// TubePreamp
// ============================================================================

DSPARK_TEST(TubePreamp_single_stage_is_even_dominant)
{
    // Asymmetric triode at moderate drive: H2 above H3 (plan acceptance).
    TubePreamp<float> amp;
    amp.prepare(spec(48000.0, 512, 2));
    amp.setStages(1);
    amp.setSag(0.0f);
    amp.setDrive(6.0f);

    auto buf = makeStereoBuffer(512);
    const int total = (48000 / 512) * 512;
    std::vector<float> out(static_cast<size_t>(total));
    int n = 0;
    for (int b = 0; b < total / 512; ++b)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            buf.ch(0)[i] = 0.5f * std::sin(2.0f * 3.14159265f * 220.0f
                                           * static_cast<float>(n) / 48000.0f);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        amp.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(b * 512 + i)] = buf.ch(0)[i];
    }
    auto hMag = [&](double freq) {
        const size_t from = out.size() / 2;
        double re = 0, im = 0;
        for (size_t i = from; i < out.size(); ++i)
        {
            const double ph = 2.0 * 3.14159265358979 * freq
                            * static_cast<double>(i - from) / 48000.0;
            re += static_cast<double>(out[i]) * std::cos(ph);
            im += static_cast<double>(out[i]) * std::sin(ph);
        }
        return std::sqrt(re * re + im * im);
    };
    const double h1 = hMag(220.0), h2 = hMag(440.0), h3 = hMag(660.0);
    EXPECT_GT(h2, h3);                  // triode asymmetry
    EXPECT_GT(h2 / h1, 0.005);          // audibly warm
    EXPECT_NO_NAN(out.data(), static_cast<int>(out.size()));
}

DSPARK_TEST(TubePreamp_supply_sags_under_load)
{
    TubePreamp<float> amp;
    amp.prepare(spec(48000.0, 512, 2));
    amp.setStages(2);
    amp.setDrive(12.0f);
    amp.setSag(1.0f);

    auto buf = makeStereoBuffer(512);
    for (int b = 0; b < 50; ++b)
    {
        buf.fillSine(220.0f, 48000.0f, 0.5f);
        amp.processBlock(buf.view());
    }
    EXPECT_LT(amp.getSupplyVoltage(), 270.0f);   // B+ drooped from 300 V
    EXPECT_GT(amp.getSupplyVoltage(), 150.0f);   // but sane
}

DSPARK_TEST(TubePreamp_stable_at_max_drive)
{
    TubePreamp<float> amp;
    amp.prepare(spec(48000.0, 512, 2));
    amp.setStages(2);
    amp.setDrive(36.0f);
    amp.setSag(1.0f);

    auto buf = makeStereoBuffer(512);
    uint32_t rng = 77u;
    for (int b = 0; b < 100; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            rng = rng * 1664525u + 1013904223u;
            buf.ch(0)[i] = (b / 10) % 2 ? 0.0f
                : (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        amp.processBlock(buf.view());
        EXPECT_NO_NAN(buf.ch(0), 512);
    }
}


// ============================================================================
// TransformerModel
// ============================================================================

namespace {

double xfmrThdAt(double freq, float amp, float drive)
{
    TransformerModel<float> tr;
    tr.prepare(spec(48000.0, 512, 2));
    tr.setDrive(drive);
    tr.setResonance(0.0f);

    const int total = (48000 / 512) * 512;
    std::vector<float> out(static_cast<size_t>(total));
    auto buf = makeStereoBuffer(512);
    int n = 0;
    for (int b = 0; b < total / 512; ++b)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            buf.ch(0)[i] = amp * static_cast<float>(
                std::sin(2.0 * 3.14159265358979 * freq * n / 48000.0));
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        tr.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(b * 512 + i)] = buf.ch(0)[i];
    }
    auto hm = [&](double f) {
        const size_t from = out.size() / 2;
        double re = 0, im = 0;
        for (size_t i = from; i < out.size(); ++i)
        {
            const double ph = 2.0 * 3.14159265358979 * f
                            * static_cast<double>(i - from) / 48000.0;
            re += static_cast<double>(out[i]) * std::cos(ph);
            im += static_cast<double>(out[i]) * std::sin(ph);
        }
        return std::sqrt(re * re + im * im);
    };
    const double h1 = hm(freq);
    double harm = 0;
    for (int k = 2; k <= 5; ++k)
    {
        const double h = hm(freq * k);
        harm += h * h;
    }
    return std::sqrt(harm) / h1;
}

} // namespace

DSPARK_TEST(TransformerModel_distortion_is_LF_weighted)
{
    // Core saturation acts on flux (integral of voltage): at equal level,
    // distortion must concentrate in the low end - the transformer signature.
    const double t60 = xfmrThdAt(60.0, 0.5f, 6.0f);
    const double t1k = xfmrThdAt(1000.0, 0.5f, 6.0f);
    EXPECT_GT(t60, 5.0 * t1k);
    EXPECT_GT(t60, 0.03);            // audible iron at +6 dB drive
    EXPECT_LT(t60, 0.5);             // but not broken
}

DSPARK_TEST(TransformerModel_program_level_transparent)
{
    // Min drive, big core, PROGRAM level (-12 dBFS at 1 kHz): the
    // int -> JA -> exact-inverse path must be near unity (validates the
    // algebraic inverse and the calibration). Small-signal probes are the
    // wrong reference here: they ride the JA virgin curve, whose
    // susceptibility sits an order of magnitude below the major loop -
    // calibrating to them made program material explode at high drive.
    TransformerModel<float> tr;
    tr.prepare(spec(48000.0, 512, 2));
    tr.setDrive(-12.0f);
    tr.setCoreSize(1.0f);
    tr.setResonance(0.0f);

    auto buf = makeStereoBuffer(512);
    const int total = (48000 / 512) * 512;
    std::vector<float> out(static_cast<size_t>(total));
    int n = 0;
    for (int b = 0; b < total / 512; ++b)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            buf.ch(0)[i] = 0.25f * std::sin(2.0f * 3.14159265f * 1000.0f
                                            * static_cast<float>(n) / 48000.0f);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        tr.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(b * 512 + i)] = buf.ch(0)[i];
    }
    const size_t from = out.size() / 2;
    double re = 0, im = 0;
    for (size_t i = from; i < out.size(); ++i)
    {
        const double ph = 2.0 * 3.14159265358979 * 1000.0
                        * static_cast<double>(i - from) / 48000.0;
        re += static_cast<double>(out[i]) * std::cos(ph);
        im += static_cast<double>(out[i]) * std::sin(ph);
    }
    const double g = 2.0 * std::sqrt(re * re + im * im)
                   / static_cast<double>(out.size() - from) / 0.25;
    EXPECT_NEAR(20.0 * std::log10(g), 0.0, 2.0);
}

DSPARK_TEST(TransformerModel_stable_under_bursts)
{
    TransformerModel<float> tr;
    tr.prepare(spec(48000.0, 512, 2));
    tr.setDrive(24.0f);

    auto buf = makeStereoBuffer(512);
    uint32_t rng = 5u;
    for (int b = 0; b < 100; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            rng = rng * 1664525u + 1013904223u;
            buf.ch(0)[i] = (b / 10) % 2 ? 0.0f
                : (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        tr.processBlock(buf.view());
        EXPECT_NO_NAN(buf.ch(0), 512);
    }
}

// ============================================================================
// Analog category contract: neutral insertion loudness, click-free drags
// ============================================================================

namespace {

// Pink-weighted multitone (one tone per octave, 100..6400 Hz): program-like.
float analogProgram(int n, float amp)
{
    static const double f[] = { 100, 200, 400, 800, 1600, 3200, 6400 };
    double s = 0;
    for (int k = 0; k < 7; ++k)
        s += std::sin(2.0 * 3.14159265358979 * f[k] * n / 48000.0 + k * 1.7);
    return amp * static_cast<float>(s / 2.6457513);
}

// Program-loudness insertion gain (dB) at the effect's defaults.
template <typename FX>
double analogInsertionDb(FX& fx, float amp = 0.1f)
{
    auto buf = makeStereoBuffer(512);
    const int total = 96000, measFrom = 48000;
    double inSq = 0, outSq = 0;
    int n = 0;
    while (n < total)
    {
        for (int i = 0; i < 512; ++i)
        {
            buf.ch(0)[i] = analogProgram(n + i, amp);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        fx.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            if (n + i >= measFrom)
            {
                const double s = analogProgram(n + i, amp);
                inSq += s * s;
                outSq += static_cast<double>(buf.ch(0)[i]) * buf.ch(0)[i];
            }
        n += 512;
    }
    return 10.0 * std::log10(outSq / std::max(inSq, 1e-30));
}

// Worst |sample-to-sample delta| of a 1 kHz tone, with the drive either held
// (worst over 3 settings) or dragged across its range one step per block.
template <typename FX, typename SetDrive>
double analogWorstDelta(FX& fx, SetDrive setDrive, float dbFrom, float dbTo, bool drag)
{
    auto buf = makeStereoBuffer(512);
    auto pass = [&](float fixedDb) {
        fx.reset();
        setDrive(fixedDb);
        double md = 0; float prev = 0;
        int n = 0, block = 0;
        const int total = drag ? 96000 : 36000;
        while (n < total)
        {
            if (drag && n >= 24000)
            {
                const float t = std::fmod(block * 0.02f, 2.0f);
                const float x = t < 1.0f ? t : 2.0f - t;
                setDrive(dbFrom + (dbTo - dbFrom) * x);
            }
            for (int i = 0; i < 512; ++i)
            {
                buf.ch(0)[i] = 0.25f * std::sin(2.0f * 3.14159265f * 1000.0f
                                                * static_cast<float>(n + i) / 48000.0f);
                buf.ch(1)[i] = buf.ch(0)[i];
            }
            fx.processBlock(buf.view());
            for (int i = 0; i < 512; ++i)
            {
                const float v = buf.ch(0)[i];
                if (n + i > 9600)
                    md = std::max(md, static_cast<double>(std::abs(v - prev)));
                prev = v;
            }
            n += 512; ++block;
        }
        return md;
    };
    if (drag)
        return pass(dbFrom);
    double worst = 0;
    for (float db : { dbFrom, 0.5f * (dbFrom + dbTo), dbTo })
        worst = std::max(worst, pass(db));
    return worst;
}

} // namespace

DSPARK_TEST(TubePreamp_insertion_loudness_is_neutral)
{
    // Activating the effect at the Lab defaults must not change loudness.
    // The old 1 kHz/10 ms calibration measured INSIDE the supply-sag
    // settling step and buried the wet path: -22.9 dB measured insertion.
    TubePreamp<float> amp;
    amp.prepare(spec(48000.0, 512, 2));
    EXPECT_NEAR(static_cast<float>(analogInsertionDb(amp)), 0.0f, 3.0f);
}

DSPARK_TEST(TubePreamp_drive_knob_is_alive_and_monotonic)
{
    // Per-drive measured loudness link (prepare LUT) + 0.25 dB/dB residual:
    // backing off must audibly drop level; pushing must never LOSE level.
    // The analytic 1/drive^x link failed exactly there: once the triode
    // pinned at its ceiling, level FELL hard with drive (user-reported).
    auto levelAt = [](float driveDb, float amp) {
        TubePreamp<float> tp;
        tp.prepare(spec(48000.0, 512, 2));
        tp.setDrive(driveDb);
        return analogInsertionDb(tp, amp);
    };
    // Reference program level: monotonic rise across the whole knob.
    const double lo  = levelAt(-12.0f, 0.1f);
    const double mid = levelAt(0.0f, 0.1f);
    const double hi  = levelAt(24.0f, 0.1f);
    const double max = levelAt(36.0f, 0.1f);
    EXPECT_LT(lo, mid - 1.0);
    EXPECT_GT(hi, mid - 1.0);
    EXPECT_LT(max, mid + 11.0);
    EXPECT_GT(max, mid - 2.0);
    // Hot program (-10 dBFS): heavy saturation may flatten the curve but
    // must never collapse the level anywhere on the knob.
    const double hotMid = levelAt(0.0f, 0.32f);
    const double hotMax = levelAt(36.0f, 0.32f);
    EXPECT_GT(hotMax, hotMid - 3.0);
}

DSPARK_TEST(TapeMachine_drive_knob_is_alive_and_monotonic)
{
    // Same contract as TubePreamp: the per-drive calibration used to pin
    // loudness exactly, leaving the knob dead below 0 dB and attenuating
    // above. The +0.25 dB/dB residual slope keeps it alive.
    auto levelAt = [](float driveDb) {
        TapeMachine<float> tm;
        tm.prepare(spec(48000.0, 512, 2));
        tm.setDrive(driveDb);
        return analogInsertionDb(tm);
    };
    const double lo = levelAt(-12.0f), mid = levelAt(0.0f), hi = levelAt(24.0f);
    EXPECT_LT(lo, mid - 1.0);
    EXPECT_GT(hi, mid - 1.0);
    EXPECT_LT(hi, mid + 8.0);
}

DSPARK_TEST(TubePreamp_default_tone_is_not_overcoloured)
{
    // The raw FMV stack at neutral knobs is a ~10 dB mid scoop - far too
    // much fixed colour for an insert. The reference flattener must keep
    // the default response within a few dB across the audible band while
    // leaving the tone knobs fully active (they act relative to flat).
    auto gainAt = [](double freq) {
        TubePreamp<float> amp;
        amp.prepare(spec(48000.0, 512, 2));
        auto buf = makeStereoBuffer(512);
        const int total = 96000, measFrom = 48000;
        double inSq = 0, outSq = 0;
        int n = 0;
        while (n < total)
        {
            for (int i = 0; i < 512; ++i)
            {
                buf.ch(0)[i] = 0.0316f * std::sin(2.0f * 3.14159265f
                                * static_cast<float>(freq) * (n + i) / 48000.0f);
                buf.ch(1)[i] = buf.ch(0)[i];
            }
            amp.processBlock(buf.view());
            for (int i = 0; i < 512; ++i)
                if (n + i >= measFrom)
                {
                    const double s = 0.0316 * std::sin(2.0 * 3.14159265358979
                                    * freq * (n + i) / 48000.0);
                    inSq += s * s;
                    outSq += static_cast<double>(buf.ch(0)[i]) * buf.ch(0)[i];
                }
            n += 512;
        }
        return 10.0 * std::log10(outSq / std::max(inSq, 1e-30));
    };
    double lo = 1e9, hi = -1e9;
    for (double f : { 100.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0 })
    {
        const double g = gainAt(f);
        lo = std::min(lo, g);
        hi = std::max(hi, g);
    }
    EXPECT_LT(hi - lo, 6.0);   // was 10.6 dB peak-to-valley before flattening
}

DSPARK_TEST(TransformerModel_insertion_loudness_is_neutral)
{
    TransformerModel<float> tr;
    tr.prepare(spec(48000.0, 512, 2));
    EXPECT_NEAR(static_cast<float>(analogInsertionDb(tr)), 0.0f, 3.0f);
}

DSPARK_TEST(TubePreamp_drive_drag_is_click_free)
{
    // Dragging drive across its full range while audio plays must not click
    // or jump: the worst sample delta may not exceed twice the worst static
    // behaviour anywhere in the range (was x171 with flat per-block gain
    // steps and a small-signal-only loudness reference).
    TubePreamp<float> amp;
    amp.prepare(spec(48000.0, 512, 2));
    auto set = [&](float db){ amp.setDrive(db); };
    const double stat = analogWorstDelta(amp, set, -12.0f, 36.0f, false);
    const double drag = analogWorstDelta(amp, set, -12.0f, 36.0f, true);
    EXPECT_LT(drag, 2.0 * stat + 1e-4);
}

DSPARK_TEST(TransformerModel_drive_drag_is_click_free)
{
    // The model differentiates its output (x 2*fs): any gain step becomes a
    // huge click (was x21), and linear ramping of the (hScale, mScale)
    // compensation pair transits through over-gained states - the geometric
    // ramps must hold the drag at the static behaviour.
    TransformerModel<float> tr;
    tr.prepare(spec(48000.0, 512, 2));
    auto set = [&](float db){ tr.setDrive(db); };
    const double stat = analogWorstDelta(tr, set, -12.0f, 24.0f, false);
    const double drag = analogWorstDelta(tr, set, -12.0f, 24.0f, true);
    EXPECT_LT(drag, 2.0 * stat + 1e-4);
}

// Invalid inputs must be ignored. The old header: NaN drive/bias poisoned the
// JA hysteresis state PERMANENTLY (12800 non-finite samples measured AFTER
// reposting valid values), NaN wow/flutter poisoned the playback shelf state
// through the delay read (15360 measured, also permanent), an invalid
// re-prepare stormed permanently (10240), and wild enums ran a silent
// fallback that diverged from any honest setting.
DSPARK_TEST(TapeMachine_invalid_inputs_are_ignored)
{
    auto dut  = std::make_unique<TapeMachine<float>>();
    auto twin = std::make_unique<TapeMachine<float>>();
    const auto s = spec(48000.0, 512, 2);
    for (auto* t : { dut.get(), twin.get() })
    {
        t->setDrive(6.0f);
        t->setBias(0.5f);
        t->setWowFlutter(0.15f);
        t->prepare(s);
    }

    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();
    dut->setDrive(kNan);      dut->setDrive(kInf);
    dut->setBias(kNan);
    dut->setLossEffects(kNan);
    dut->setHeadBump(kNan);
    dut->setWowFlutter(kNan);
    dut->setNoise(kNan);
    dut->setMix(-kInf);
    dut->setSpeed(static_cast<TapeMachine<float>::Speed>(99));
    dut->setStandard(static_cast<TapeMachine<float>::Standard>(77));
    dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    dut->prepare(spec(48000.0, 512, -3));

    // Wild enums clamp to the last member; sync the twin to the same values.
    EXPECT_TRUE(dut->getSpeed() == TapeMachine<float>::Speed::IPS_30);
    EXPECT_TRUE(dut->getStandard() == TapeMachine<float>::Standard::CCIR);
    twin->setSpeed(TapeMachine<float>::Speed::IPS_30);
    twin->setStandard(TapeMachine<float>::Standard::CCIR);

    EXPECT_NEAR(dut->getDrive(), 6.0f, 1e-6f);
    EXPECT_NEAR(dut->getBias(), 0.5f, 1e-6f);
    EXPECT_NEAR(dut->getWowFlutter(), 0.15f, 1e-6f);
    EXPECT_NEAR(dut->getMix(), 1.0f, 1e-6f);

    auto ta = makeStereoBuffer(512);
    auto tb = makeStereoBuffer(512);
    float maxDiff = 0.0f;
    int nonFinite = 0;
    for (int k = 0; k < 25; ++k)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.7f * std::sin(2.0f * 3.14159265f * (180.0f + 40.0f * ch)
                                                * float(k * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v;
                tb.ch(ch)[i] = v;
            }
        dut->processBlock(ta.view());
        twin->processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(ta.ch(ch)[i])) ++nonFinite;
                maxDiff = std::max(maxDiff, std::fabs(ta.ch(ch)[i] - tb.ch(ch)[i]));
            }
    }
    EXPECT_EQ(nonFinite, 0);
    EXPECT_EQ(maxDiff, 0.0f);
}

// Invalid inputs must be ignored. The old header CRASHED with an access
// violation on setDrive(NaN): the NaN passed the clamp into recompute(),
// where the drive LUT position clamp is NaN-blind and the float-to-int cast
// produced a wild index (OOB read). A NaN sample rate also passed the old
// `<= 0` prepare gate and stormed the whole circuit (10240 non-finite
// samples measured).
DSPARK_TEST(TubePreamp_invalid_inputs_are_ignored)
{
    auto dut  = std::make_unique<TubePreamp<float>>();
    auto twin = std::make_unique<TubePreamp<float>>();
    const auto s = spec(48000.0, 512, 2);
    for (auto* t : { dut.get(), twin.get() })
    {
        t->setDrive(6.0f);
        t->setSag(0.3f);
        t->prepare(s);
    }

    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();
    dut->setDrive(kNan);   dut->setDrive(kInf);   // old: crash in recompute
    dut->setTreble(kNan);
    dut->setBass(kNan);
    dut->setMiddle(kNan);
    dut->setSag(kNan);
    dut->setOutput(kNan);
    dut->setMix(-kInf);
    dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));

    EXPECT_NEAR(dut->getDrive(), 6.0f, 1e-6f);
    EXPECT_NEAR(dut->getTreble(), 0.5f, 1e-6f);
    EXPECT_NEAR(dut->getSag(), 0.3f, 1e-6f);
    EXPECT_NEAR(dut->getMix(), 1.0f, 1e-6f);
    EXPECT_EQ(dut->getStages(), 1);

    auto ta = makeStereoBuffer(512);
    auto tb = makeStereoBuffer(512);
    float maxDiff = 0.0f;
    int nonFinite = 0;
    for (int k = 0; k < 25; ++k)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.4f * std::sin(2.0f * 3.14159265f * (220.0f + 60.0f * ch)
                                                * float(k * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v;
                tb.ch(ch)[i] = v;
            }
        dut->processBlock(ta.view());
        twin->processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(ta.ch(ch)[i])) ++nonFinite;
                maxDiff = std::max(maxDiff, std::fabs(ta.ch(ch)[i] - tb.ch(ch)[i]));
            }
    }
    EXPECT_EQ(nonFinite, 0);
    EXPECT_EQ(maxDiff, 0.0f);
}

// Invalid inputs must be ignored. The old header: NaN drive/coreSize/
// resonance poisoned the flux integrator, the JA loop, the bell AND the
// geometric anti-zipper ramp permanently (12800 non-finite samples measured
// AFTER reposting valid values), and a NaN rate passed the old prepare gate
// (10240 measured). Also pins the mix ramp: a hard mix flip on the
// differentiated wet stream used to click at 4.6x the steady-state delta.
DSPARK_TEST(TransformerModel_invalid_inputs_are_ignored)
{
    auto dut  = std::make_unique<TransformerModel<float>>();
    auto twin = std::make_unique<TransformerModel<float>>();
    const auto s = spec(48000.0, 512, 2);
    for (auto* t : { dut.get(), twin.get() })
    {
        t->setDrive(6.0f);
        t->prepare(s);
    }

    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();
    dut->setDrive(kNan);     dut->setDrive(kInf);
    dut->setCoreSize(kNan);
    dut->setResonance(kNan);
    dut->setMix(-kInf);
    dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));

    EXPECT_NEAR(dut->getDrive(), 6.0f, 1e-6f);
    EXPECT_NEAR(dut->getCoreSize(), 0.5f, 1e-6f);
    EXPECT_NEAR(dut->getResonance(), 0.3f, 1e-6f);
    EXPECT_NEAR(dut->getMix(), 1.0f, 1e-6f);

    auto ta = makeStereoBuffer(512);
    auto tb = makeStereoBuffer(512);
    float maxDiff = 0.0f;
    int nonFinite = 0;
    for (int k = 0; k < 25; ++k)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.4f * std::sin(2.0f * 3.14159265f * (110.0f + 40.0f * ch)
                                                * float(k * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v;
                tb.ch(ch)[i] = v;
            }
        dut->processBlock(ta.view());
        twin->processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(ta.ch(ch)[i])) ++nonFinite;
                maxDiff = std::max(maxDiff, std::fabs(ta.ch(ch)[i] - tb.ch(ch)[i]));
            }
    }
    EXPECT_EQ(nonFinite, 0);
    EXPECT_EQ(maxDiff, 0.0f);
}

// ============================================================================
// Clipper
// ============================================================================

// Invalid inputs must be ignored. The old header: NaN setters stormed the
// output (2048 non-finite samples measured while published), a wild enum left
// the clipper INERT (no switch case ran, so the wet path was the raw input)
// with getMode() lying, and an invalid re-prepare with oversampling active
// CRASHED (0xC0000409 fast-fail from the filter design).
DSPARK_TEST(Clipper_invalid_inputs_are_ignored)
{
    auto dut  = std::make_unique<Clipper<float>>();
    auto twin = std::make_unique<Clipper<float>>();
    const auto s = spec(48000.0, 512, 2);
    for (auto* c : { dut.get(), twin.get() })
    {
        c->setMode(Clipper<float>::Mode::Soft);
        c->setCeiling(-1.0f);
        c->setInputGain(12.0f);
        c->setSlewLimit(0.3f);
        c->setMix(0.8f);
        c->setOversampling(2);
        c->prepare(s);
    }

    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();
    dut->setCeiling(kNan);  dut->setCeiling(kInf);
    dut->setInputGain(kNan); dut->setInputGain(-kInf);
    dut->setSlewLimit(kNan);
    dut->setMix(kNan);
    dut->prepare(spec(48000.0, 512, -3));                               // old: crash
    dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 0, 2)); // old: poisoned spec

    EXPECT_NEAR(dut->getCeiling(), -1.0f, 1e-6f);
    EXPECT_NEAR(dut->getInputGain(), 12.0f, 1e-6f);
    EXPECT_NEAR(dut->getSlewLimit(), 0.3f, 1e-6f);
    EXPECT_NEAR(dut->getMix(), 0.8f, 1e-6f);

    dut->setMode(static_cast<Clipper<float>::Mode>(99)); // wild enum
    EXPECT_TRUE(dut->getMode() == Clipper<float>::Mode::GoldenRatio);
    dut->setMode(Clipper<float>::Mode::Soft);            // back in sync with twin

    auto ta = makeStereoBuffer(512);
    auto tb = makeStereoBuffer(512);
    float maxDiff = 0.0f;
    int nonFinite = 0;
    for (int k = 0; k < 15; ++k)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.9f * std::sin(2.0f * 3.14159265f * (220.0f + 200.0f * ch)
                                                * float(k * 512 + i) / 48000.0f);
                ta.ch(ch)[i] = v;
                tb.ch(ch)[i] = v;
            }
        dut->processBlock(ta.view());
        twin->processBlock(tb.view());
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(ta.ch(ch)[i])) ++nonFinite;
                maxDiff = std::max(maxDiff, std::fabs(ta.ch(ch)[i] - tb.ch(ch)[i]));
            }
    }
    EXPECT_EQ(nonFinite, 0);
    EXPECT_EQ(maxDiff, 0.0f);

    // An instance that never saw a valid prepare passes audio through.
    auto fresh = std::make_unique<Clipper<float>>();
    auto tf = makeBuffer(1, 256);
    for (int i = 0; i < 256; ++i)
        tf.ch(0)[i] = 1.5f * std::sin(2.0f * 3.14159265f * 300.0f * float(i) / 48000.0f);
    std::vector<float> orig(tf.ch(0), tf.ch(0) + 256);
    fresh->processBlock(tf.view());
    float passDiff = 0.0f;
    for (int i = 0; i < 256; ++i)
        passDiff = std::max(passDiff, std::fabs(tf.ch(0)[i] - orig[i]));
    EXPECT_EQ(passDiff, 0.0f); // old header hard-clipped peaks above 1.0 here
}

// Functional coverage the Clipper never had: every mode is bounded by the
// ceiling under heavy drive, and passes quiet material at unity with no drive.
DSPARK_TEST(Clipper_modes_clip_to_ceiling_and_pass_quiet)
{
    const float ceiling = std::pow(10.0f, -1.0f / 20.0f);
    for (int m = 0; m < 4; ++m)
    {
        auto c = std::make_unique<Clipper<float>>();
        c->setMode(static_cast<Clipper<float>::Mode>(m));
        c->setCeiling(-1.0f);
        c->setInputGain(18.0f);
        c->prepare(spec(48000.0, 512, 1));
        auto tb = makeBuffer(1, 512);
        generateSine(tb.ch(0), 512, 997.0f, 48000.0f, 0.9f);
        c->processBlock(tb.view());
        const float peak = measurePeak(tb.ch(0), 512);
        EXPECT_LT(peak, ceiling * 1.001f); // bounded by the ceiling
        EXPECT_GT(peak, ceiling * 0.5f);   // and actually driven into it

        auto q = std::make_unique<Clipper<float>>();
        q->setMode(static_cast<Clipper<float>::Mode>(m));
        q->setCeiling(0.0f);
        q->prepare(spec(48000.0, 512, 1));
        auto tq = makeBuffer(1, 512);
        generateSine(tq.ch(0), 512, 997.0f, 48000.0f, 0.05f);
        std::vector<float> orig(tq.ch(0), tq.ch(0) + 512);
        q->processBlock(tq.view());
        float quietDiff = 0.0f;
        for (int i = 0; i < 512; ++i)
            quietDiff = std::max(quietDiff, std::fabs(tq.ch(0)[i] - orig[i]));
        EXPECT_LT(quietDiff, 2e-4f); // unity for quiet material in every mode
    }
}

// ============================================================================
// Oversampling-transparency pins and nonlinear defect regression pins
// ============================================================================

// Transparency: TapeMachine gains a configurable oversampling factor incl.
// 1x, with the added latency reported per factor and persisted in state.
DSPARK_TEST(TapeMachine_oversampling_configurable_and_reported)
{
    TapeMachine<float> t;
    t.prepare(spec(48000.0, 512, 2));
    const int base4x = t.getLatency();                 // default factor is 4
    t.setOversampling(1); const int l1 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 1);
    t.setOversampling(2); const int l2 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 2);
    t.setOversampling(4); const int l4 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 4);
    t.setOversampling(8); const int l8 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 8);
    EXPECT_EQ(l4, base4x);                              // default unchanged
    EXPECT_LT(l1, l2); EXPECT_LT(l2, l4); EXPECT_LT(l4, l8);   // monotone in factor
    EXPECT_EQ(t.getLatencySamples(), t.getLatency());  // the alias agrees
    // Invalid / non-power-of-two factors are ignored (stay at 8).
    t.setOversampling(3); t.setOversampling(0); t.setOversampling(-4); t.setOversampling(32);
    EXPECT_EQ(t.getOversamplingFactor(), 8);
    // State persists the factor.
    auto blob = t.getState();
    TapeMachine<float> u; u.prepare(spec(48000.0, 512, 2));
    EXPECT_TRUE(u.setState(blob.data(), blob.size()));
    EXPECT_EQ(u.getOversamplingFactor(), 8);
}

// Transparency: TubePreamp gains a configurable oversampling factor incl.
// 1x=off (zero added latency), reported per factor and persisted in state.
DSPARK_TEST(TubePreamp_oversampling_configurable_and_reported)
{
    TubePreamp<float> t;
    t.prepare(spec(48000.0, 512, 2));
    t.setOversampling(1); const int l1 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 1);
    EXPECT_EQ(l1, 0);                                   // 1x = off, no added latency
    t.setOversampling(2); const int l2 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 2);
    t.setOversampling(4); const int l4 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 4);
    EXPECT_LT(l1, l2); EXPECT_LT(l2, l4);
    EXPECT_EQ(t.getLatencySamples(), t.getLatency());
    t.setOversampling(5); t.setOversampling(0);         // ignored
    EXPECT_EQ(t.getOversamplingFactor(), 4);
    auto blob = t.getState();
    TubePreamp<float> u; u.prepare(spec(48000.0, 512, 2));
    EXPECT_TRUE(u.setState(blob.data(), blob.size()));
    EXPECT_EQ(u.getOversamplingFactor(), 4);
}

// A transient NaN/Inf field must not poison the Jiles-Atherton core
// forever - after the bad samples the output recovers to finite, bounded M.
DSPARK_TEST(Hysteresis_survives_nonfinite_input)
{
    Hysteresis<double> h;
    h.prepare(192000.0);
    h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
    for (int i = 0; i < 500; ++i) (void)h.processSample(2.0e4 * std::sin(0.01 * i));
    (void)h.processSample(std::numeric_limits<double>::quiet_NaN());
    (void)h.processSample(std::numeric_limits<double>::infinity());
    bool finite = true;
    for (int i = 0; i < 4000; ++i)
    {
        const double m = h.processSample(2.0e4 * std::sin(0.01 * i));
        if (!std::isfinite(m) || std::abs(m) > h.getSaturation() * 1.0000001) finite = false;
    }
    EXPECT_TRUE(finite);
}

// A NaN/Inf input sample must not poison the transformer channel -
// a clean block after a poisoned block is fully finite.
DSPARK_TEST(TransformerModel_survives_nonfinite_input)
{
    TransformerModel<float> t;
    t.prepare(spec(48000.0, 512, 1));
    t.setDrive(12.0f);
    auto b1 = makeBuffer(1, 512);
    generateSine(b1.ch(0), 512, 220.0f, 48000.0f, 0.3f);
    b1.ch(0)[100] = std::numeric_limits<float>::quiet_NaN();
    b1.ch(0)[300] = std::numeric_limits<float>::infinity();
    t.processBlock(b1.view());
    auto b2 = makeBuffer(1, 512);
    generateSine(b2.ch(0), 512, 220.0f, 48000.0f, 0.3f);
    t.processBlock(b2.view());
    EXPECT_NO_NAN(b2.ch(0), 512);
}

// A transient NaN/Inf input must not poison TapeMachine forever (the
// JA/EQ/FIR/transport state otherwise mutes the channel to exact silence until
// reset()). After a poison block, sustained clean input must produce finite AND
// non-zero output WITHOUT an intervening reset().
DSPARK_TEST(TapeMachine_survives_nonfinite_input)
{
    TapeMachine<float> t;
    t.prepare(spec(48000.0, 512, 1));
    t.setWowFlutter(0.0f);
    // Poison block: mixed NaN/Inf.
    auto p = makeBuffer(1, 512);
    generateSine(p.ch(0), 512, 220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(0)[192] = std::numeric_limits<float>::infinity();
    p.ch(0)[300] = -std::numeric_limits<float>::infinity();
    t.processBlock(p.view());
    // Sustained clean input (well past the 223-sample latency).
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 60; ++blk)
    {
        auto b = makeBuffer(1, 512);
        generateSine(b.ch(0), 512, 220.0f, 48000.0f, 0.3f);
        t.processBlock(b.view());
        if (blk >= 55)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(b.ch(0)[i])) finite = false;
                energy += double(b.ch(0)[i]) * double(b.ch(0)[i]);
            }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);   // recovered signal, not stuck silence
}

// A transient NaN/Inf input must not poison TubePreamp forever (the
// recursive triode NR / WDF tone / DC-blocker state otherwise emits all-NaN
// until reset()). After a poison block, clean input must recover to finite,
// non-zero output WITHOUT an intervening reset().
DSPARK_TEST(TubePreamp_survives_nonfinite_input)
{
    TubePreamp<float> t;
    t.prepare(spec(48000.0, 512, 1));
    t.setDrive(6.0f);
    auto p = makeBuffer(1, 512);
    generateSine(p.ch(0), 512, 220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(0)[192] = std::numeric_limits<float>::infinity();
    p.ch(0)[300] = -std::numeric_limits<float>::infinity();
    t.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 40; ++blk)
    {
        auto b = makeBuffer(1, 512);
        generateSine(b.ch(0), 512, 220.0f, 48000.0f, 0.3f);
        t.processBlock(b.view());
        if (blk >= 35)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(b.ch(0)[i])) finite = false;
                energy += double(b.ch(0)[i]) * double(b.ch(0)[i]);
            }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);
}

// ============================================================================
// Two guards pinned here. C1: Equalizer/FilterEngine front-door
// non-finite guard (recursive biquad cascade poisons forever on NaN/Inf). C2:
// FilterEngine shelf-slope / Equalizer shelf-band Q change must be audible on
// the MinimumPhase static fast path -- the rebuild guard used to omit
// shelfSlope_, silently dropping a shelf Q change in min-phase steady state
// while LinearPhase and the analysis curve reflected it. Revert-check: removing
// either fix turns the corresponding pin RED.
// ============================================================================
DSPARK_TEST(Equalizer_survives_nonfinite_input)
{
    Equalizer<float> eq;
    eq.prepare(spec(48000.0, 512, 2));
    eq.setBand(0, 1000.0f, 6.0f, 2.0f);
    auto p = makeBuffer(2, 512);
    p.fillSine(220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::quiet_NaN();
    p.ch(1)[192] = std::numeric_limits<float>::infinity();
    eq.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 20; ++blk)
    {
        auto b = makeBuffer(2, 512);
        b.fillSine(220.0f, 48000.0f, 0.3f);
        eq.processBlock(b.view());
        if (blk >= 15)
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

DSPARK_TEST(FilterEngine_survives_nonfinite_input)
{
    FilterEngine<float> fe;
    fe.prepare(spec(48000.0, 512, 1));
    fe.setPeaking(1000.0f, 6.0f, 2.0f);
    auto p = makeBuffer(1, 512);
    generateSine(p.ch(0), 512, 220.0f, 48000.0f, 0.3f);
    p.ch(0)[64]  = std::numeric_limits<float>::infinity();
    p.ch(0)[192] = std::numeric_limits<float>::quiet_NaN();
    fe.processBlock(p.view());
    double energy = 0.0; bool finite = true;
    for (int blk = 0; blk < 20; ++blk)
    {
        auto b = makeBuffer(1, 512);
        generateSine(b.ch(0), 512, 220.0f, 48000.0f, 0.3f);
        fe.processBlock(b.view());
        if (blk >= 15)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(b.ch(0)[i])) finite = false;
                energy += double(b.ch(0)[i]) * double(b.ch(0)[i]);
            }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(energy, 1e-3);
}

// C2 regression: a shelf-slope-only change (same freq/gain/shape) must alter the
// MinimumPhase static-path output. The bug: the fast-path rebuild guard omitted
// shelfSlope_, so once the freq/gain smoothers had settled bit-exact to target
// (the normal steady state after ~100 ms of no change), a shelf slope/Q-only
// change was silently dropped -- coefficients were never rebuilt. The LONG
// pre-settle (kSettle blocks) is essential: with a short settle the smoother is
// still inching toward target, so f/g differ every block and the rebuild fires
// regardless, masking the defect. Revert-check: without the fix delta == 0.
static constexpr int kShelfSettle = 500;   // reach bit-exact smoother settle
static double feEnergy(dspark::FilterEngine<float>& fe, int nblk)
{
    using namespace dspark;
    double e = 0.0;
    for (int blk = 0; blk < nblk; ++blk)
    {
        auto b = makeBuffer(1, 512);
        generateSine(b.ch(0), 512, 120.0f, 48000.0f, 0.3f);
        fe.processBlock(b.view());
        e = 0.0;
        for (int i = 0; i < 512; ++i) e += double(b.ch(0)[i]) * double(b.ch(0)[i]);
    }
    return e;
}

DSPARK_TEST(FilterEngine_shelf_slope_change_is_audible)
{
    FilterEngine<float> fe;
    fe.prepare(spec(48000.0, 512, 1));
    fe.setLowShelf(200.0f, 12.0f, 0.4f);       // settle bit-exact: gentle slope
    const double gentle = feEnergy(fe, kShelfSettle);
    fe.setLowShelf(200.0f, 12.0f, 1.6f);       // ONLY the slope changes (same freq/gain)
    const double steep  = feEnergy(fe, 16);
    EXPECT_GT(std::abs(steep - gentle), 1e-3 * gentle);
}

static double eqEnergy(dspark::Equalizer<float>& eq, int nblk)
{
    using namespace dspark;
    double e = 0.0;
    for (int blk = 0; blk < nblk; ++blk)
    {
        auto b = makeBuffer(1, 512);
        generateSine(b.ch(0), 512, 120.0f, 48000.0f, 0.3f);
        eq.processBlock(b.view());
        e = 0.0;
        for (int i = 0; i < 512; ++i) e += double(b.ch(0)[i]) * double(b.ch(0)[i]);
    }
    return e;
}

DSPARK_TEST(Equalizer_shelf_band_Q_change_is_audible_in_min_phase)
{
    Equalizer<float> eq;
    eq.prepare(spec(48000.0, 512, 1));
    typename Equalizer<float>::BandConfig cfg;
    cfg.frequency = 200.0f; cfg.gain = 12.0f;
    cfg.type = Equalizer<float>::BandType::LowShelf; cfg.enabled = true;
    cfg.q = 0.3f; eq.setBand(0, cfg);          // settle bit-exact: low Q -> gentle slope
    const double loQ = eqEnergy(eq, kShelfSettle);
    cfg.q = 1.2f; eq.setBand(0, cfg);          // ONLY Q changes (same freq/gain/type)
    const double hiQ = eqEnergy(eq, 16);
    EXPECT_GT(std::abs(hiQ - loQ), 1e-3 * loQ);
}

// ---------------------------------------------------------------------------
// CHANGE-REQUEST M-008C-TEST-1 (ADDITIVE ONLY): concurrent publication pin for
// Effects/Equalizer.h. Nothing above this marker is modified.
//
// Why this pin can actually fail, which is the whole point of it:
// before the band configs became a seqlock, this exact schedule adopted a
// config that was never published -- ~560k-680k torn adoptions per 5 s run,
// ~3.3% of all adoptions, reproduced three times out of three
// (tests/results/M-008C/torn-probe-before.log). The two configs below differ
// in EVERY field, and the audio thread reads back what it just adopted through
// the band's own FilterEngine, ON THE AUDIO THREAD, so the read-back cannot
// itself invent a mismatch. A (shape, slope) pair that is neither of the two
// published pairs can only come from one updateActiveFilters() pass reading
// cfg.type from one publication and cfg.slope from another.
//
// The liveness floor makes a vacuous pass impossible: if the threads never
// overlapped, adoptions would not accumulate and the test fails instead of
// silently proving nothing.
DSPARK_TEST(Equalizer_concurrent_setBand_is_never_torn)
{
    using EQ = Equalizer<float, 8>;
    using Shape = FilterEngine<float>::Shape;

    auto eq = std::make_unique<EQ>();
    AudioSpec spec; spec.sampleRate = 48000.0; spec.maxBlockSize = 16; spec.numChannels = 1;
    eq->prepare(spec);
    eq->setNumBands(1);

    EQ::BandConfig a;
    a.frequency = 100.0f;   a.gain = 12.0f;   a.q = 0.5f;
    a.type = EQ::BandType::LowPass;  a.slope = 12; a.enabled = true;
    EQ::BandConfig b;
    b.frequency = 15000.0f; b.gain = -18.0f;  b.q = 8.0f;
    b.type = EQ::BandType::HighPass; b.slope = 48; b.enabled = true;
    eq->setBand(0, a);

    std::atomic<bool> stop{ false };
    std::atomic<long long> torn{ 0 };
    std::atomic<long long> adoptions{ 0 };

    std::thread control([&] {
        bool even = true;
        while (!stop.load(std::memory_order_relaxed))
        {
            eq->setBand(0, even ? a : b);
            even = !even;
        }
    });

    std::thread audio([&] {
        std::array<float, 16> buf{};
        buf.fill(0.05f);
        float* chans[1] = { buf.data() };
        long long localTorn = 0, localAdopt = 0;
        for (int i = 0; i < 200000; ++i)
        {
            AudioBufferView<float> view(chans, 1, 16);
            eq->processBlock(view);

            const auto& fe = eq->getBandFilter(0);
            const Shape shape = fe.getShape();
            const int   slope = fe.getSlopeDb();
            const bool isA = (shape == Shape::LowPass  && slope == 12);
            const bool isB = (shape == Shape::HighPass && slope == 48);
            if (isA || isB) ++localAdopt; else ++localTorn;
        }
        torn.store(localTorn);
        adoptions.store(localAdopt);
        stop.store(true, std::memory_order_relaxed);
    });

    control.join();
    audio.join();

    EXPECT_EQ(torn.load(), 0LL);                 // the invariant
    EXPECT_GT(adoptions.load(), 1000LL);         // liveness: no vacuous pass
}
