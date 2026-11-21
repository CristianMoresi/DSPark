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
#include "../Effects/MultibandSaturation.h"
#include "../Effects/LoFi.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <array>
#include <atomic>
#include <complex>
#include <functional>
#include <numbers>
#include <thread>
#include <utility>
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

namespace
{
// Hann-windowed DFT energy of x[first, first + n) summed over [lo, hi] Hz.
double glideBandEnergy(const std::vector<float>& x, int first, int n, double lo, double hi, double fs)
{
    double sum = 0.0;
    const double step = fs / n;
    for (double f = lo; f <= hi; f += step)
    {
        std::complex<double> a = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double w = 0.5 - 0.5 * std::cos(twoPi<double> * i / (n - 1));
            a += w * static_cast<double>(x[static_cast<size_t>(first + i)]) *
                 std::polar(1.0, -twoPi<double> * f * i / fs);
        }
        sum += std::norm(a);
    }
    return sum;
}
} // namespace

DSPARK_TEST(FilterEngine_cutoff_glide_has_no_chunk_rate_sidebands)
{
    // Glides design coefficients every 16 samples. Stepping them there put
    // chunk-rate sidebands on a tone at -55 dB; interpolating them sample by
    // sample between the designs leaves no sideband above -90 dB.
    const double fs = 48000.0;
    for (float target : { 1500.0f, 8000.0f })
    {
        FilterEngine<float> fe;
        fe.prepare(AudioSpec{ fs, 512, 1 });
        fe.setLowPass(target == 1500.0f ? 8000.0f : 1500.0f, 0.707f, 24);
        fe.applyParametersNow();
        std::vector<float> x(8192);
        for (size_t n = 0; n < x.size(); ++n)
            x[n] = 0.5f * static_cast<float>(std::sin(twoPi<double> * 300.0 * static_cast<double>(n) / fs));
        fe.setFrequency(target);
        for (int o = 0; o < 8192; o += 512)
        {
            float* ch[1] = { x.data() + o };
            fe.processBlock(AudioBufferView<float>(ch, 1, 512));
        }
        const double side = glideBandEnergy(x, 0, 4096, 2000.0, 4000.0, fs);
        const double tone = glideBandEnergy(x, 0, 4096, 250.0, 350.0, fs);
        EXPECT_LT(10.0 * std::log10(side / tone), -90.0);
    }
}

DSPARK_TEST(FilterEngine_cutoff_glide_lands_in_its_ramp_time)
{
    // The cutoff glide is a cubic that arrives exactly 165 ms after the
    // change; the critically damped glide it replaced was still tens of
    // cents away then (1 cent at 331 ms). A 500 Hz tone at the new low-pass
    // cutoff reads its settled level from 170 ms on.
    const double fs = 48000.0;
    FilterEngine<float> fe;
    fe.prepare(AudioSpec{ fs, 480, 1 });
    fe.setLowPass(10000.0f, 0.707f, 12);
    fe.applyParametersNow();
    std::vector<float> x(48000);
    for (size_t n = 0; n < x.size(); ++n)
        x[n] = 0.25f * static_cast<float>(std::sin(twoPi<double> * 500.0 * static_cast<double>(n) / fs));
    fe.setFrequency(500.0f);
    for (int o = 0; o < 48000; o += 480)
    {
        float* ch[1] = { x.data() + o };
        fe.processBlock(AudioBufferView<float>(ch, 1, 480));
    }
    auto rms = [&](int first, int n) {
        double s = 0.0;
        for (int i = first; i < first + n; ++i) s += double(x[size_t(i)]) * double(x[size_t(i)]);
        return std::sqrt(s / n);
    };
    // 960 samples = ten periods of 500 Hz.
    const double early = rms(8160, 960), settled = rms(40000, 960);
    EXPECT_NEAR(20.0 * std::log10(early / settled), 0.0, 0.02);
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
    // Fixed kernel (32768 taps at 48 kHz): latency kernel/2 + kernel/16,
    // independent of the host block (it used to equal maxBlockSize).
    EXPECT_EQ(eq.getLatency(), 16384 + 2048);

    auto tb = makeBuffer(1, 512);
    int argmax = -1;
    float best = 0.0f;
    const int blocks = (eq.getLatency() + 2048) / 512;
    for (int blk = 0; blk < blocks; ++blk)
    {
        for (int i = 0; i < 512; ++i) tb.ch(0)[i] = 0.0f;
        if (blk == 0) tb.ch(0)[0] = 1.0f;
        eq.processBlock(tb.view());
        for (int i = 0; i < 512; ++i)
            if (std::abs(tb.ch(0)[i]) > best) { best = std::abs(tb.ch(0)[i]); argmax = blk * 512 + i; }
    }
    EXPECT_EQ(argmax, eq.getLatency()); // the reported latency is the real one

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
    EXPECT_EQ(static_cast<int>(subject.getBandConfig(1).type), 0); // not a type: the band keeps its own
    EXPECT_EQ(subject.getBandConfig(1).slope, 96);                  // clamped (the cookbook engine runs 48)

    // The NaN never reached the stored config nor the state blob.
    EXPECT_NEAR(subject.getBandConfig(0).frequency, 1000.0f, 1e-3f);
    auto blob = subject.getState();
    StateReader r(blob.data(), blob.size());
    EXPECT_NEAR(r.read("b0.freq", 999.0f), 1000.0f, 1e-3f);

    EXPECT_EQ(subject.getLatency(), 16384 + 2048); // fixed linear-phase kernel

    // Long enough to get past the linear-phase latency into real output.
    const int kLen = 4096 + 16384 + 2048;
    auto ta = makeBuffer(2, kLen);
    auto tb = makeBuffer(2, kLen);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < kLen; ++i)
        {
            const float v = 0.4f * std::sin(6.2831853f * (ch == 0 ? 440.0f : 620.0f) * i / 48000.0f);
            ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
        }
    // Process per 512-sample block (the LP engine's prepared block size).
    for (int off = 0; off < kLen; off += 512)
    {
        float* pa[2] = { ta.ch(0) + off, ta.ch(1) + off };
        float* pb[2] = { tb.ch(0) + off, tb.ch(1) + off };
        subject.processBlock(AudioBufferView<float>(pa, 2, 512));
        twin.processBlock(AudioBufferView<float>(pb, 2, 512));
    }

    EXPECT_NO_NAN(ta.ch(0), kLen);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < kLen; ++i)
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

// The ADAA difference quotients ran in T: for float the antiderivative (order
// 1) minus its previous value, divided by input steps as small as 1e-5, kept
// only a few correct bits (measured on 20-200 Hz tones: errors up to -48 dB
// re peak for Tube/SoftClip and -51 dB for the Wavefolder, spiking at every
// crest). The float path must now track the double path to float precision.
DSPARK_TEST(Saturation_adaa_is_accurate_in_float)
{
    using Algo = Saturation<float>::Algorithm;
    for (Algo algo : { Algo::Tube, Algo::SoftClip, Algo::Wavefolder })
    {
        for (double f : { 20.0, 50.0 })
        {
            Saturation<float> sf;
            Saturation<double> sd;
            sf.setAlgorithm(algo);
            sd.setAlgorithm(static_cast<Saturation<double>::Algorithm>(algo));
            sf.setDrive(6.0f); sd.setDrive(6.0);
            sf.setAntialiasing(true); sd.setAntialiasing(true);
            sf.prepare(spec(48000.0, 512, 1));
            sd.prepare(spec(48000.0, 512, 1));
            const int n = 24000;
            std::vector<float> xf(static_cast<size_t>(n));
            std::vector<double> xd(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i)
            {
                xf[static_cast<size_t>(i)] = static_cast<float>(0.5 * std::sin(twoPi<double> * f * i / 48000.0));
                xd[static_cast<size_t>(i)] = static_cast<double>(xf[static_cast<size_t>(i)]);
            }
            for (int off = 0; off < n; off += 512)
            {
                float* pf[1] = { xf.data() + off };
                double* pd[1] = { xd.data() + off };
                sf.processBlock(AudioBufferView<float>(pf, 1, std::min(512, n - off)));
                sd.processBlock(AudioBufferView<double>(pd, 1, std::min(512, n - off)));
            }
            double worst = 0.0;
            for (int i = n / 2; i < n; ++i)
                worst = std::max(worst, std::abs(static_cast<double>(xf[static_cast<size_t>(i)]) - xd[static_cast<size_t>(i)]));
            EXPECT_LT(worst, 1e-5);   // was up to 5e-3
        }
    }
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
    twin.setAlgorithm(Saturation<float>::Algorithm::Curve);

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

// The block path runs channels in pairs (two lanes of one register). Every
// order, with three channels (a pair and one alone) and a non-finite sample,
// must give what one channel at a time gives.
DSPARK_TEST(DCBlocker_channel_pairs_match_single_channels)
{
    constexpr int kN = 7000;
    for (int order = 1; order <= 10; ++order)
    {
        DCBlocker<double> wide;
        DCBlocker<double> alone[3];
        wide.setOrder(order);
        wide.prepare(48000.0, 3, 12.0);
        for (auto& f : alone)
        {
            f.setOrder(order);
            f.prepare(48000.0, 1, 12.0);
        }
        std::vector<double> data[3], single[3];
        uint32_t rng = 777u + static_cast<uint32_t>(order);
        for (int ch = 0; ch < 3; ++ch)
        {
            data[ch].resize(kN);
            for (int i = 0; i < kN; ++i)
            {
                rng = rng * 1664525u + 1013904223u;
                data[ch][static_cast<size_t>(i)] = static_cast<double>(static_cast<int32_t>(rng)) / 2147483648.0 * 0.5 + 0.2 * ch;
            }
            single[ch] = data[ch];
        }
        data[1][3000] = std::numeric_limits<double>::quiet_NaN();
        single[1][3000] = std::numeric_limits<double>::quiet_NaN();
        double worst = 0.0;
        for (int i = 0; i < kN; i += 333)
        {
            const int n = std::min(333, kN - i);
            double* ptrs[3] = { data[0].data() + i, data[1].data() + i, data[2].data() + i };
            wide.processBlock(AudioBufferView<double>(ptrs, 3, n));
            for (int ch = 0; ch < 3; ++ch)
            {
                double* one[1] = { single[ch].data() + i };
                alone[ch].processBlock(AudioBufferView<double>(one, 1, n));
                for (int k = 0; k < n; ++k)
                    worst = std::max(worst, std::abs(data[ch][static_cast<size_t>(i + k)] - single[ch][static_cast<size_t>(i + k)]));
            }
        }
        // Rounding only: where the compiler fuses multiply-adds, it may fuse the
        // lane form and the single form differently.
        EXPECT_LT(worst, 1e-9);
        EXPECT_TRUE(std::isfinite(data[1][kN - 1]));
    }
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

DSPARK_TEST(TubePreamp_default_rejects_high_gain_foldback)
{
    // At 2x the 15 kHz, -6 dBFS tone folded 18.2 dB below its fundamental
    // through two stages at +24 dB. Also cover the worse neighboring tones.
    // The ADAA core reached -36.2 dBc here; the continuous-time core
    // measures -95.8 / -97.4 / -98.9 dBc.
    for (double sourceFrequency : {14000.0, 15000.0, 18000.0})
    {
        TubePreamp<double> amp;
        amp.setStages(2);
        amp.setDrive(24.0);
        amp.prepare(spec(48000.0, 256, 2));
        AudioBuffer<double> audio;
        audio.resize(2, 256);
        constexpr int length = 48000;
        std::vector<double> tail;
        tail.reserve(length / 2);
        for (int pos = 0; pos < length; pos += 256)
        {
            const int count = std::min(256, length - pos);
            for (int i = 0; i < count; ++i)
                audio.getChannel(0)[i] = audio.getChannel(1)[i] =
                    std::pow(10.0, -6.0 / 20.0)
                    * std::sin(2.0 * 3.14159265358979323846 * sourceFrequency * (pos + i) / 48000.0);
            amp.processBlock(audio.toView().getSubView(0, count));
            for (int i = 0; i < count; ++i)
                if (pos + i >= length / 2) tail.push_back(audio.getChannel(0)[i]);
        }
        const auto amplitude = [&](double frequency) {
            double re = 0.0, im = 0.0;
            for (size_t i = 0; i < tail.size(); ++i)
            {
                const double phase = 2.0 * 3.14159265358979323846 * frequency
                                   * static_cast<double>(i) / 48000.0;
                re += tail[i] * std::cos(phase);
                im += tail[i] * std::sin(phase);
            }
            return std::hypot(re, im);
        };
        const double fundamental = amplitude(sourceFrequency);
        double folded = 0.0;
        const int spacing = std::gcd(static_cast<int>(sourceFrequency), 48000);
        for (int frequency = spacing; frequency <= 20000; frequency += spacing)
        {
            const double harmonic = frequency / sourceFrequency;
            if (harmonic >= 1.0 && std::abs(harmonic - std::round(harmonic)) < 1e-9) continue;
            folded = std::max(folded, amplitude(frequency));
        }
        EXPECT_GT(fundamental, 1.0);
        EXPECT_LT(20.0 * std::log10(folded / fundamental), -80.0);
        EXPECT_EQ(amp.getOversamplingFactor(), 2);
        EXPECT_EQ(amp.getLatency(), 71);
    }
}

namespace {
// Direct current law, with no production derivatives or interpolation. The
// circuit reference below bisects Kirchhoff's equation to a unique root.
long double referenceTriodeCurrent(long double vp, long double vg)
{
    vp = std::max(vp, 0.0L);
    const long double z = 600.0L * (0.01L + vg / std::sqrt(300.0L + vp * vp));
    const long double e = vp / 600.0L * std::log1p(std::exp(z));
    return 2.0L / 1060.0L * std::pow(e, 1.4L);
}

long double referenceTriodeLoad(double supply, double grid, double cathodeR)
{
    long double lo = 0.0L, hi = supply / (100000.0L + cathodeR);
    for (int i = 0; i < 65; ++i)
    {
        const long double mid = 0.5L * (lo + hi);
        const long double cathode = cathodeR * mid;
        if (mid > referenceTriodeCurrent(supply - 100000.0L * mid - cathode, grid - cathode))
            hi = mid;
        else
            lo = mid;
    }
    return 0.5L * (lo + hi);
}
} // namespace

DSPARK_TEST(TubePreamp_current_surface_matches_implicit_circuit)
{
    // The most curved cells are at low supply near cutoff. A nearest or
    // bilinear lookup fails this tolerance; it also protects the capacitor
    // feedback term, which changes with the internal sample rate.
    for (double rate : {8000.0, 44100.0, 96000.0, 384000.0, 3072000.0})
    {
        auto table = std::make_unique<detail::TubePreampCurrentTable>(rate);
        const double h = 0.5 / (rate * 22e-6);
        const double cathodeR = h / (1.0 + h / 1500.0);
        uint32_t seed = 51;
        for (int n = 0; n < 6000; ++n)
        {
            seed = seed * 1664525u + 1013904223u;
            const double supply = 80.0 + 224.0 * (static_cast<double>(seed) / 4294967296.0);
            seed = seed * 1664525u + 1013904223u;
            const double grid = -16.0 + 17.0 * (static_cast<double>(seed) / 4294967296.0);
            const double measured = table->eval(supply, grid);
            const double expected = static_cast<double>(referenceTriodeLoad(supply, grid, cathodeR));
            EXPECT_NEAR(measured, expected, 5e-10); // < 50 uV at the plate load
            EXPECT_TRUE(measured >= 0.0);
        }
        // Rounded coordinates at the final grid edge used to risk reading
        // past the surface even though covers() accepted the input.
        for (double supply : {80.0, 192.0, std::nextafter(304.0, 0.0)})
            for (double grid : {-16.0, -1.03125, std::nextafter(1.0, 0.0)})
            {
                EXPECT_TRUE(table->covers(supply, grid));
                EXPECT_NEAR(table->eval(supply, grid),
                    static_cast<double>(referenceTriodeLoad(supply, grid, cathodeR)), 5e-10);
            }
        // Deep cutoff is bounded by the unloaded Koren current; substituting
        // zero here is below 1e-27 A, unlike clamping to a nonzero endpoint.
        for (double supply : {80.0, 128.0, 192.0, 303.999})
        {
            EXPECT_LT(referenceTriodeCurrent(supply, -0.08L * supply), 1e-27L);
            EXPECT_TRUE(table->covers(supply, -1000.0));
            EXPECT_NEAR(table->eval(supply, -1000.0), 0.0, 1e-30);
            for (double ratio : {-0.04, -0.025})
            {
                const double grid = supply * ratio;
                for (double offset : {-1e-8, 0.0, 1e-8})
                    EXPECT_NEAR(table->eval(supply, grid + offset),
                        static_cast<double>(referenceTriodeLoad(supply, grid + offset, cathodeR)), 5e-10);
            }
        }
        EXPECT_FALSE(table->covers(304.0, 0.0));
        EXPECT_FALSE(table->covers(300.0, std::numeric_limits<double>::quiet_NaN()));
        EXPECT_FALSE(table->covers(300.0, -std::numeric_limits<double>::infinity()));
    }
}

DSPARK_TEST(TubePreamp_phi_functions_match_their_definition)
{
    // phi_k(w) = sum_j w^j / (j + k)! weights the exact forced response of
    // each tone-circuit mode. Independent evaluation: a 60-term series for
    // |w| < 1, otherwise (e^w - sum_{j<k} w^j / j!) / w^k. The points cover
    // both production branches and the stiff range of the stack.
    for (double w : {-2000.0, -1042.0, -40.0, -7.5, -1.0, -0.5000001, -0.4999999, -0.3,
                     -0.1, -0.02, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 0.02, 0.1, 0.3,
                     0.4999999, 0.5000001, 1.0, 3.0})
    {
        double p[5];
        detail::tubePreampPhi(w, p);
        for (int k = 0; k <= 4; ++k)
        {
            long double expected = 0.0L;
            if (std::abs(w) < 1.0)
            {
                long double term = 1.0L;
                for (int j = 1; j <= k; ++j) term /= j;
                for (int j = 0; j < 60; ++j)
                {
                    expected += term;
                    term *= static_cast<long double>(w) / (j + k + 1);
                }
            }
            else
            {
                long double partial = 0.0L, term = 1.0L;
                for (int j = 0; j < k; ++j)
                {
                    partial += term;
                    term *= static_cast<long double>(w) / (j + 1);
                }
                expected = (std::exp(static_cast<long double>(w)) - partial)
                         / std::pow(static_cast<long double>(w), k);
            }
            EXPECT_NEAR(p[k], static_cast<double>(expected),
                        1e-12 * std::abs(static_cast<double>(expected)));
        }
    }
}

namespace {
std::complex<double> tubeStackResponse(const wdf::ToneStackFMV<double>::AnalogStateSpace& ss,
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

DSPARK_TEST(TubePreamp_tone_modes_reproduce_the_analog_state_space)
{
    // The modal form propagated by the continuous core must keep the
    // transfer function of the analog state space over the whole control
    // cube, including bass 0 / middle 1, where one mode is so fast that a
    // single 96 kHz interval spans about a thousand of its time constants.
    constexpr double rate = 96000.0;
    double stiffest = 0.0;
    for (double t : {0.0, 0.5, 1.0})
        for (double b : {0.0, 0.5, 1.0})
            for (double m : {0.0, 0.5, 1.0})
            {
                wdf::ToneStackFMV<double> stack(38e3, 1e6);
                stack.prepare(rate);
                stack.setControls(t, b, m);
                const auto ss = stack.analogStateSpace();
                detail::TubePreampToneModes modes;
                modes.design(ss, rate);
                for (int k = 0; k < 3; ++k)
                {
                    EXPECT_LT(modes.lambda[k], 0.0);
                    stiffest = std::min(stiffest, modes.lambda[k]);
                }
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                    {
                        double product = 0.0;
                        for (int k = 0; k < 3; ++k) product += modes.toModal[i][k] * modes.toPhysical[k][j];
                        EXPECT_NEAR(product, i == j ? 1.0 : 0.0, 1e-12);
                    }
                for (double f : {20.0, 200.0, 2000.0, 20000.0, 45000.0})
                {
                    const std::complex<double> s(0.0, 2.0 * std::numbers::pi * f);
                    std::complex<double> modal = modes.direct;
                    for (int k = 0; k < 3; ++k)
                        modal += modes.gamma[k] * modes.beta[k] * rate / (s - modes.lambda[k] * rate);
                    const auto expected = tubeStackResponse(ss, s);
                    EXPECT_LT(std::abs(modal - expected) / std::abs(expected), 1e-9);
                }
            }
    EXPECT_LT(stiffest, -500.0);
}

DSPARK_TEST(TubePreamp_core_reconstruction_and_projection_are_accurate)
{
    // Input reconstruction: the Farrow polynomial of each internal interval
    // reproduces every complex tone up to 22 kHz at 48 kHz (scaled with the
    // factor) at all interval positions. Output projection: the order-6
    // B-spline pieces form a partition of unity, and the linear-phase FIR
    // inverts their sinc^6 droop over the same band.
    constexpr double bounds[] = {-80.0, -100.0, -104.0, -100.0};
    int index = 0;
    for (int factor : {2, 4, 8, 16})
    {
        const detail::TubePreampCoreDesign design(factor);
        const int taps = design.taps, degree = design.degree;
        const double band = 22000.0 / (48000.0 * factor);
        double worst = 0.0;
        for (int iw = 0; iw <= 200; ++iw)
        {
            const double w = 2.0 * std::numbers::pi * band * iw / 200.0;
            for (int it = 0; it <= 100; ++it)
            {
                const double tau = it / 100.0;
                std::complex<double> value = 0.0;
                for (int j = 0; j < taps; ++j)
                {
                    const double* row = design.farrow + j * (degree + 1);
                    double c = row[degree];
                    for (int p = degree - 1; p >= 0; --p) c = c * tau + row[p];
                    value += c * std::polar(1.0, w * (j - taps / 2 + 1));
                }
                worst = std::max(worst, std::abs(value - std::polar(1.0, w * tau)));
            }
        }
        EXPECT_LT(20.0 * std::log10(worst), bounds[index]);

        for (int i = 0; i <= 100; ++i)
        {
            const double d = -0.5 + i / 100.0;
            double sum = 0.0;
            for (const auto& piece : design.kernel)
            {
                double v = 0.0;
                for (int p = detail::TubePreampCoreDesign::kKernelOrder - 1; p >= 0; --p)
                    v = v * d + piece[static_cast<size_t>(p)];
                EXPECT_GT(v, -1e-15);
                sum += v;
            }
            EXPECT_NEAR(sum, 1.0, 1e-13);
        }

        const auto& c = design.compensation;
        size_t first = 0;
        while (first < c.size() && c[first] == 0.0) ++first;
        const size_t half = (c.size() - 1 - first) / 2, centre = first + half;
        for (size_t q = 1; q <= half; ++q) EXPECT_EQ(c[centre - q], c[centre + q]);
        for (int i = 0; i <= 400; ++i)
        {
            const double f = band * i / 400.0;
            double response = 0.0;
            for (size_t n = 0; n < c.size(); ++n)
                response += c[n] * std::cos(2.0 * std::numbers::pi * f
                                            * (static_cast<double>(n) - static_cast<double>(centre)));
            const double sinc = f == 0.0 ? 1.0 : std::sin(std::numbers::pi * f) / (std::numbers::pi * f);
            EXPECT_LT(std::abs(20.0 * std::log10(response * std::pow(sinc, 6))), 1e-4);
        }
        ++index;
    }
}

namespace {
// Worst component below 20 kHz other than DC and the exact harmonics, in dB
// relative to the tone. Coherent 16384-point measurement at an odd bin.
// The caller prepares the amp once per setting; reset() restores the settled
// operating point before every tone.
double tubeCoherentSpurDbc(TubePreamp<double>& amp, double nominal, double level)
{
    constexpr int size = 16384, block = 256;
    int bin = static_cast<int>(nominal * size / 48000.0 + 0.5);
    if ((bin & 1) == 0) ++bin;
    amp.reset();
    AudioBuffer<double> audio;
    audio.resize(1, block);
    std::vector<double> capture(size), spectrum(size + 2);
    for (int pos = 0; pos < 6 * size; pos += block)
    {
        for (int j = 0; j < block; ++j)
            audio.getChannel(0)[j] = std::pow(10.0, level / 20.0)
                * std::sin(2.0 * std::numbers::pi * bin * (pos + j) / size);
        amp.processBlock(audio.toView());
        if (pos >= 5 * size)
            std::copy_n(audio.getChannel(0), block, capture.data() + pos - 5 * size);
    }
    FFTReal<double> fft(size);
    fft.forward(capture.data(), spectrum.data());
    const auto magnitude = [&](int k) {
        return std::hypot(spectrum[static_cast<size_t>(2 * k)], spectrum[static_cast<size_t>(2 * k + 1)]);
    };
    double folded = 0.0;
    for (int k = 1; k * 48000 < 20000 * size; ++k)
        if (k % bin != 0) folded = std::max(folded, magnitude(k));
    return 20.0 * std::log10(folded / magnitude(bin));
}
} // namespace

DSPARK_TEST(TubePreamp_coherent_sweep_resolves_aliases_next_to_fundamental)
{
    // Exact coherent bins: a +/-2-bin Hann mask would hide the fifth-harmonic
    // alias next to 16 kHz. Include between-kHz failures from independent QA.
    // The published ADAA core measured -36.2 dBc at +24 dB (two stages) and
    // -19.2 / -22.5 dBc at +36 dB (two / one stage) on this sweep.
    struct Setting { int stages; double drive, bound; };
    for (const Setting setting : {Setting{2, 24.0, -75.0}, Setting{2, 36.0, -75.0},
                                  Setting{1, 36.0, -75.0}})
    {
        TubePreamp<double> amp;
        amp.setStages(setting.stages);
        amp.setDrive(setting.drive);
        amp.prepare(spec(48000.0, 256, 1));
        for (double nominal : {1000., 2000., 3000., 4000., 5000., 6000., 7000., 8000.,
                               9000., 10000., 11000., 12000., 12750., 13000., 14000.,
                               15000., 15250., 16000., 17000., 18000., 19000., 19250., 20000.})
        {
            const double ratio = tubeCoherentSpurDbc(amp, nominal, -6.0);
            if (ratio >= setting.bound)
                std::cerr << "TubePreamp " << setting.stages << " stage(s) +" << setting.drive
                          << " dB tone " << nominal << " Hz: " << ratio << " dBc\n";
            EXPECT_LT(ratio, setting.bound);
        }
    }
}

DSPARK_TEST(TubePreamp_wet_path_latency_matches_point_circuit)
{
    // The point circuit at 1x has no resampler and no core delay: its wet
    // phase anchors the reported latency of every oversampled factor. At
    // -12 dB drive the stages are nearly linear; the remaining difference is
    // the bilinear warping of the 1x circuit (0.014 samples, 0.025 dB measured).
    constexpr int size = 8192, block = 256;
    constexpr std::array<int, 4> bins {171, 853, 1707, 2561};   // about 1, 5, 10, 15 kHz
    std::array<std::array<std::array<std::complex<double>, 4>, 5>, 2> phasor {};
    int row = 0;
    for (int factor : {1, 2, 4, 8, 16})
    {
        TubePreamp<double> amp;
        amp.setOversampling(factor);
        amp.setDrive(-12.0);
        amp.setSag(0.0);
        amp.prepare(spec(48000.0, block, 1));
        const int latency = amp.getLatency();
        AudioBuffer<double> audio;
        audio.resize(1, block);
        for (int stages : {1, 2})
        {
            amp.setStages(stages);
            for (size_t column = 0; column < bins.size(); ++column)
            {
                const int bin = bins[column];
                amp.reset();
                std::complex<double> sum = 0.0;
                const int total = 2 * size + latency;
                for (int pos = 0; pos < total; pos += block)
                {
                    const int count = std::min(block, total - pos);
                    for (int j = 0; j < count; ++j)
                        audio.getChannel(0)[j] = 0.1 * std::sin(2.0 * std::numbers::pi * bin * (pos + j) / size);
                    amp.processBlock(audio.toView().getSubView(0, count));
                    for (int j = 0; j < count; ++j)
                    {
                        const int n = pos + j - latency - size;
                        if (n >= 0)
                            sum += audio.getChannel(0)[j] * std::polar(1.0, -2.0 * std::numbers::pi * bin * n / size);
                    }
                }
                phasor[static_cast<size_t>(stages - 1)][static_cast<size_t>(row)][column] = sum;
            }
        }
        ++row;
    }
    for (const auto& stage : phasor)
        for (size_t factor = 1; factor < stage.size(); ++factor)
            for (size_t k = 0; k < bins.size(); ++k)
            {
                const auto ratio = stage[factor][k] / stage[0][k];
                const double delay = -std::arg(ratio) * size / (2.0 * std::numbers::pi * bins[k]);
                EXPECT_LT(std::abs(delay), 0.05);
                EXPECT_LT(std::abs(20.0 * std::log10(std::abs(ratio))), 0.1);
            }
}

namespace {
std::vector<double> renderTubeTone(int factor, double rate, int stages, double drive,
                                   double treble, double bass, double middle, double hz)
{
    TubePreamp<double> amp;
    amp.setOversampling(factor);
    amp.setStages(stages);
    amp.setDrive(drive);
    amp.setTreble(treble);
    amp.setBass(bass);
    amp.setMiddle(middle);
    amp.prepare(spec(rate, 4096, 1));
    const int latency = amp.getLatency();
    const size_t total = static_cast<size_t>(0.15 * rate) + static_cast<size_t>(latency);
    std::vector<double> y(total);
    for (size_t i = 0; i < total; ++i)
        y[i] = 0.25 * std::sin(2.0 * std::numbers::pi * hz * static_cast<double>(i) / rate);
    for (size_t pos = 0; pos < total; pos += 4096)
    {
        const int count = static_cast<int>(std::min<size_t>(4096, total - pos));
        double* channels[] = { y.data() + pos };
        amp.processBlock({ channels, 1, count });
    }
    y.erase(y.begin(), y.begin() + latency);
    return y;
}

std::complex<double> tubeHarmonic(const std::vector<double>& y, double rate, double hz, int harmonic)
{
    // The last 30 periods (an integer number of samples at both rates used).
    const size_t count = static_cast<size_t>(30.0 * rate / hz + 0.5), from = y.size() - count;
    std::complex<double> sum = 0.0;
    for (size_t i = from; i < y.size(); ++i)
        sum += y[i] * std::polar(1.0, -2.0 * std::numbers::pi * harmonic * hz
                                     * static_cast<double>(i - from) / rate);
    return 2.0 * sum / static_cast<double>(count);
}
} // namespace

DSPARK_TEST(TubePreamp_continuous_core_matches_independent_point_circuit)
{
    // Independent reference: the point-sampled circuit (factor 1: implicit
    // current table with trapezoidal cathode, WDF tone stack and sag) run at
    // 32x the base rate, where it agrees with a 64x run. At the default 2x
    // the continuous core must reproduce the reference gain and harmonic
    // structure (each harmonic below 20 kHz relative to the fundamental,
    // with the fundamental's phase removed). The published ADAA core
    // measured -15.1 / -39.3 dBc and -2.36 / -0.35 dB on these two cases.
    struct Case { int stages; double drive, treble, bass, middle, hz, bound; };
    for (const Case c : {Case{2, 36.0, 1.0, 1.0, 1.0, 3000.0, -55.0},
                         Case{2, 24.0, 0.5, 0.0, 1.0, 1000.0, -70.0}})
    {
        const double referenceRate = 48000.0 * 32.0;
        const auto reference = renderTubeTone(1, referenceRate, c.stages, c.drive,
                                              c.treble, c.bass, c.middle, c.hz);
        const auto actual = renderTubeTone(2, 48000.0, c.stages, c.drive,
                                           c.treble, c.bass, c.middle, c.hz);
        const auto r1 = tubeHarmonic(reference, referenceRate, c.hz, 1);
        const auto a1 = tubeHarmonic(actual, 48000.0, c.hz, 1);
        EXPECT_LT(std::abs(20.0 * std::log10(std::abs(a1) / std::abs(r1))), 0.05);
        double worst = 0.0;
        for (int h = 2; h * c.hz <= 20000.0; ++h)
        {
            const auto r = tubeHarmonic(reference, referenceRate, c.hz, h)
                         / std::pow(r1 / std::abs(r1), h) / std::abs(r1);
            const auto a = tubeHarmonic(actual, 48000.0, c.hz, h)
                         / std::pow(a1 / std::abs(a1), h) / std::abs(a1);
            worst = std::max(worst, std::abs(a - r));
        }
        EXPECT_LT(20.0 * std::log10(worst), c.bound);
    }
}

DSPARK_TEST(TubePreamp_dry_mix_and_stage_changes_keep_reported_latency)
{
    constexpr int total = 513;
    for (int factor : {1, 2, 4, 8, 16})
    {
        TubePreamp<double> dry, wet, mixed;
        for (auto* amp : {&dry, &wet, &mixed})
        {
            amp->setOversampling(factor);
            amp->prepare(spec(48000.0, 64, 1));
        }
        dry.setMix(0.0); mixed.setMix(0.5);
        dry.reset(); mixed.reset();
        const int latency = dry.getLatency();
        std::array<double, total> source{};
        for (int n = 0; n < total; ++n)
            source[static_cast<size_t>(n)] = 0.05 * std::sin(0.7 * n) + (n == 7 ? 0.4 : 0.0);
        AudioBuffer<double> d, w, m;
        d.resize(1, 37); w.resize(1, 37); m.resize(1, 37);
        for (int pos = 0; pos < total; pos += 37)
        {
            const int count = std::min(37, total - pos);
            const int stages = (pos / 111) % 2 + 1;
            for (auto* amp : {&dry, &wet, &mixed}) amp->setStages(stages);
            for (int j = 0; j < count; ++j)
                d.getChannel(0)[j] = w.getChannel(0)[j] = m.getChannel(0)[j] = source[static_cast<size_t>(pos + j)];
            dry.processBlock(d.toView().getSubView(0, count));
            wet.processBlock(w.toView().getSubView(0, count));
            mixed.processBlock(m.toView().getSubView(0, count));
            EXPECT_EQ(dry.getLatency(), latency);
            EXPECT_EQ(wet.getLatency(), latency);
            for (int j = 0; j < count; ++j)
            {
                const double expected = pos + j >= latency ? source[static_cast<size_t>(pos + j - latency)] : 0.0;
                EXPECT_NEAR(d.getChannel(0)[j], expected, 1e-14);
                EXPECT_NEAR(m.getChannel(0)[j], 0.5 * (expected + w.getChannel(0)[j]), 1e-13);
            }
        }
    }
}

DSPARK_TEST(TubePreamp_dc_operating_point_and_legacy_state)
{
    for (int stages : {1, 2})
    {
        TubePreamp<double> amp;
        amp.setStages(stages);
        amp.setSag(0.3);
        amp.prepare(spec(48000.0, 256, 1));
        AudioBuffer<double> audio;
        audio.resize(1, 256);
        for (int n = 0; n < 50; ++n)
        {
            std::fill_n(audio.getChannel(0), 256, 0.0);
            amp.processBlock(audio.toView());
        }
        // At DC, the cathode is Ip*Rk and the common supply is
        // 300 - stages*Rsag*Ip. Solve that complete circuit independently.
        long double lo = 0.0L, hi = 0.003L;
        for (int n = 0; n < 65; ++n)
        {
            const long double ip = 0.5L * (lo + hi);
            const long double current = referenceTriodeCurrent(
                300.0L - (101500.0L + stages * 12000.0L) * ip, -1500.0L * ip);
            if (ip > current) hi = ip;
            else lo = ip;
        }
        const double expectedSupply = static_cast<double>(300.0L - stages * 12000.0L * (lo + hi) * 0.5L);
        EXPECT_NEAR(amp.getSupplyVoltage(), expectedSupply, 2e-5);

        StateWriter legacy(stateId("TUBE"), 1);
        legacy.write("drive", 0.0f);
        const auto oldState = legacy.blob();
        EXPECT_TRUE(amp.setState(oldState.data(), oldState.size()));
        EXPECT_EQ(amp.getOversamplingFactor(), 2);
        EXPECT_EQ(amp.getLatency(), 71);
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

// The loop's third harmonic of a 10.1 kHz tone is at 30.3 kHz: at 1x it
// folds to 17.7 kHz, at the 2x default it is removed by the downsampler. One
// second of a tone on a whole number of hertz puts both on exact DFT bins.
DSPARK_TEST(TransformerModel_oversampling_keeps_folded_harmonics_out)
{
    auto foldedDb = [](int factor, int& latency) {
        TransformerModel<double> tr;
        tr.setOversampling(factor);
        tr.prepare(AudioSpec{ 48000.0, 512, 1 });
        tr.setDrive(24.0);
        tr.setCoreSize(0.0);
        latency = tr.getLatency();
        std::vector<double> x(96000);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 0.5 * std::sin(2.0 * 3.14159265358979 * 10103.0 * static_cast<double>(i) / 48000.0);
        for (size_t off = 0; off < x.size(); off += 512)
        {
            double* p = x.data() + off;
            tr.processBlock(AudioBufferView<double>(&p, 1, static_cast<int>(std::min<size_t>(512, x.size() - off))));
        }
        auto bin = [&](double f) {
            double re = 0.0, im = 0.0;
            for (size_t i = 48000; i < 96000; ++i)
            {
                const double ph = 2.0 * 3.14159265358979 * f * static_cast<double>(i - 48000) / 48000.0;
                re += x[i] * std::cos(ph);
                im += x[i] * std::sin(ph);
            }
            return std::sqrt(re * re + im * im);
        };
        return 20.0 * std::log10(bin(17691.0) / bin(10103.0));
    };
    int lat1 = -1, lat2 = -1;
    const double at1x = foldedDb(1, lat1);
    const double at2x = foldedDb(2, lat2);
    std::cout << "  10.1 kHz third folded to 17.7 kHz: " << at1x << " dB at 1x (latency "
              << lat1 << "), " << at2x << " dB at 2x (latency " << lat2 << ")\n";
    EXPECT_GT(at1x, -45.0);    // the defect the default removes: -37.5 dB
    EXPECT_LT(at2x, -90.0);
    EXPECT_EQ(lat1, 0);
    EXPECT_GT(lat2, 0);
}

// Fully dry, the output is the input delayed by exactly getLatency(), at
// every factor and whatever the blocking; the factor travels in the state.
DSPARK_TEST(TransformerModel_dry_path_matches_the_reported_latency)
{
    for (const int factor : { 1, 2, 4 })
    {
        TransformerModel<float> tr;
        tr.setOversampling(factor);
        tr.setMix(0.0f);   // before prepare(), which seeds the mix ramp at its target
        tr.prepare(AudioSpec{ 48000.0, 256, 1 });
        const int lat = tr.getLatency();
        std::vector<float> x(4000), y;
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.3f * std::sin(0.037f * static_cast<float>(i));
        y = x;
        size_t off = 0;
        for (const int len : { 1, 700, 256, 33, 2000, 1010 })   // 700 and 2000 exceed the prepared maximum
        {
            float* p = y.data() + off;
            tr.processBlock(AudioBufferView<float>(&p, 1, len));
            off += static_cast<size_t>(len);
        }
        double worst = 0.0;
        for (size_t i = static_cast<size_t>(lat); i < x.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(y[i] - x[i - static_cast<size_t>(lat)])));
        EXPECT_LT(worst, 1e-7);
        EXPECT_EQ(tr.getOversamplingFactor(), factor);
    }

    TransformerModel<float> a, b;
    a.setOversampling(4);
    b.prepare(AudioSpec{ 48000.0, 512, 2 });
    const auto blob = a.getState();
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getOversamplingFactor(), 4);
    EXPECT_EQ(b.getLatency(), b.getLatencySamples());
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

// The bias carrier's own harmonics must not mirror the programme into the
// audio band. At 0.375 of the internal rate the carrier's third harmonic
// folded onto the base-rate Nyquist frequency and every tone came out with
// an image at 24 kHz minus its frequency: 9.9 dB below a 10.1 kHz tone at
// -30 dBFS, 22 dB below a 5 kHz one. At a quarter of the internal rate the
// image measures 80 dB down.
DSPARK_TEST(TapeMachine_bias_carrier_leaves_no_mirror_image)
{
    for (const double f : { 10100.0, 5000.0 })
    {
        TapeMachine<float> tape;
        tape.prepare(spec(48000.0, 512, 2));
        tape.setWowFlutter(0.0f);
        auto out = runTapeSine(tape, f, 0.0316f, 1.0);
        const double image = 20.0 * std::log10(tapeToneMag(out, 24000.0 - f) / tapeToneMag(out, f));
        std::cout << "  " << f << " Hz: image at " << 24000.0 - f << " Hz " << image << " dB\n";
        EXPECT_LT(image, -60.0);
    }
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

namespace {
template <class Sample>
std::array<std::vector<Sample>, 2> renderTubeAutomation(int factor, int partition,
                                                       int control)
{
    constexpr int frames = 4096, boundary = 173, maximumBlock = 257;
    TubePreamp<Sample> amp;
    amp.setOversampling(factor);
    amp.setDrive(Sample(0));
    amp.setOutput(Sample(-3));
    amp.setStages(1);
    amp.prepare({ 48000.0, maximumBlock, 2 });
    std::array<std::vector<Sample>, 2> result;
    for (auto& channel : result) channel.resize(frames);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < frames; ++i)
            result[static_cast<size_t>(ch)][static_cast<size_t>(i)] =
                static_cast<Sample>(0.12 * std::sin(2.0 * std::numbers::pi
                    * (317.0 + ch * 701.0) * i / 48000.0));

    constexpr std::array<int, 6> irregular { 7, 89, 1, 113, 53, 257 };
    int position = 0;
    size_t block = 0;
    while (position < frames)
    {
        if (position == boundary)
        {
            if (control == 0) amp.setDrive(Sample(24));
            if (control == 1) amp.setOutput(Sample(12));
            if (control == 2) amp.setStages(2);
        }
        if (control == 2)
        {
            // Revert during warm-up, request again, reverse an audible fade,
            // and finally settle. These events also split inside host blocks.
            if (position == 311 || position == 1907) amp.setStages(1);
            if (position == 503 || position == 2341) amp.setStages(2);
        }
        const int requested = partition == 0 ? maximumBlock
                            : partition == 1 ? irregular[block++ % irregular.size()] : 1;
        int count = std::min(requested, frames - position);
        if (position < boundary) count = std::min(count, boundary - position);
        if (control == 2)
            for (int event : {311, 503, 1907, 2341})
                if (position < event) count = std::min(count, event - position);
        Sample* channels[] = { result[0].data() + position, result[1].data() + position };
        amp.processBlock({ channels, 2, count });
        position += count;
    }
    return result;
}

template <class Sample>
void verifyTubeAutomationPartitions()
{
    for (int factor : { 1, 2, 16 })
        for (int control = 0; control < 3; ++control)
        {
            const auto reference = renderTubeAutomation<Sample>(factor, 0, control);
            for (int partition : { 1, 2 })
            {
                const auto actual = renderTubeAutomation<Sample>(factor, partition, control);
                double maximumDifference = 0.0;
                for (size_t ch = 0; ch < actual.size(); ++ch)
                    for (size_t i = 0; i < actual[ch].size(); ++i)
                        maximumDifference = std::max(maximumDifference,
                            std::abs(static_cast<double>(actual[ch][i])
                                   - static_cast<double>(reference[ch][i])));
                EXPECT_NEAR(maximumDifference, 0.0, 1e-11);
            }
        }
}
} // namespace

DSPARK_TEST(TubePreamp_gain_automation_is_independent_of_block_partition)
{
    verifyTubeAutomationPartitions<float>();
    verifyTubeAutomationPartitions<double>();
}

DSPARK_TEST(TubePreamp_stage_switch_preserves_silence_and_latency)
{
    for (int factor : {1, 2, 16})
      for (double input : {0.0, 0.1})
    {
        TubePreamp<double> amp;
        amp.setOversampling(factor);
        amp.setDrive(36.0);
        amp.setSag(1.0);
        amp.prepare({44100.0, 257, 1});
        const int latency = amp.getLatencySamples();
        std::array<double, 257> samples {};
        double peak = 0.0;
        // Let the constant-input response settle before checking topology
        // changes. Its physical start-up transient is not a switching click.
        for (int block = input == 0.0 ? 0 : -96; block < 100; ++block)
        {
            // Covers both directions, a warm-up cancellation, an in-flight
            // reversal and returning to a previously dormant path.
            if (block == 8 || block == 10 || block == 48) amp.setStages(2);
            if (block == 9 || block == 13 || block == 80) amp.setStages(1);
            samples.fill(input);
            double* p = samples.data();
            amp.processBlock({&p, 1, static_cast<int>(samples.size())});
            for (double x : samples)
            {
                EXPECT_TRUE(std::isfinite(x));
                if (block >= 0) peak = std::max(peak, std::abs(x));
            }
            EXPECT_EQ(amp.getLatencySamples(), latency);
        }
        // A topology change must not turn its changed quiescent plate load
        // into a signal. The former absolute-state copy produced -34 dBFS.
        EXPECT_LT(peak, input == 0.0 ? 1e-7 : 1e-6);
    }
}

DSPARK_TEST(TubePreamp_stage_switch_is_smooth_and_reaches_the_requested_circuit)
{
    constexpr int frameCount = 24576, change = 8197, maxBlock = 257;
    constexpr double sampleRate = 48000.0;
    for (double drive : {-12.0, 36.0})
    {
        std::array<std::vector<double>, 4> output;
        for (int run = 0; run < 4; ++run)
        {
            auto& samples = output[static_cast<size_t>(run)];
            samples.resize(frameCount);
            for (int i = 0; i < frameCount; ++i)
                samples[static_cast<size_t>(i)] = 0.2 * std::sin(
                    2.0 * std::numbers::pi * 40.0 * i / sampleRate);
            TubePreamp<double> amp;
            const int initial = run % 2 + 1;
            amp.setStages(initial);
            amp.setDrive(drive);
            amp.setSag(1.0);
            amp.prepare({sampleRate, maxBlock, 1});
            int pos = 0;
            while (pos < frameCount)
            {
                if (run >= 2)
                {
                    if (pos == change || pos == change + 911) amp.setStages(3 - initial);
                    if (pos == change + 733) amp.setStages(initial);
                }
                int count = std::min(maxBlock, frameCount - pos);
                for (int event : {change, change + 733, change + 911})
                    if (pos < event) count = std::min(count, event - pos);
                double* p = samples.data() + pos;
                amp.processBlock({&p, 1, count});
                pos += count;
            }
        }
        double staticDelta = 0.0, movingDelta = 0.0;
        for (int i = change; i < frameCount; ++i)
        {
            const auto n = static_cast<size_t>(i);
            for (int mode = 0; mode < 2; ++mode)
            {
                const auto& fixed = output[static_cast<size_t>(mode)];
                const auto& moving = output[static_cast<size_t>(mode + 2)];
                staticDelta = std::max(staticDelta, std::abs(fixed[n] - fixed[n - 1]));
                movingDelta = std::max(movingDelta, std::abs(moving[n] - moving[n - 1]));
            }
        }
        // Compare against both actual circuits, including their nonlinear
        // transients. The old switch exceeded this by orders of magnitude.
        EXPECT_LT(movingDelta, 1.15 * staticDelta + 1e-6);
        for (int mode = 0; mode < 2; ++mode)
        {
            const auto& target = output[static_cast<size_t>(1 - mode)];
            const auto& moving = output[static_cast<size_t>(mode + 2)];
            double error = 0.0, energy = 0.0;
            for (int i = frameCount - 2048; i < frameCount; ++i)
            {
                const auto n = static_cast<size_t>(i);
                error += (moving[n] - target[n]) * (moving[n] - target[n]);
                energy += target[n] * target[n];
            }
            // The incoming circuit runs independently after the fade; it must
            // approach the always-running requested circuit, without a mute,
            // retained old path, or an arbitrary output gain correction.
            EXPECT_LT(error, energy * 1e-5);
        }
    }
}

DSPARK_TEST(TubePreamp_output_automation_follows_the_sample_clock)
{
    constexpr int frames = 4096, boundary = 173;
    TubePreamp<double> reference, automated;
    for (auto* amp : { &reference, &automated })
    {
        amp->setOversampling(1);
        amp->setDrive(12.0);
        amp->setOutput(-3.0);
        amp->prepare({ 48000.0, frames, 1 });
    }
    std::array<double, frames> baseline{}, output{};
    for (int i = 0; i < frames; ++i)
        baseline[static_cast<size_t>(i)] = output[static_cast<size_t>(i)] =
            0.1 * std::sin(2.0 * std::numbers::pi * 997.0 * i / 48000.0);
    double* a[] = { baseline.data() };
    double* b[] = { output.data() };
    reference.processBlock({ a, 1, frames });
    automated.processBlock({ b, 1, boundary });
    automated.setOutput(12.0);
    b[0] += boundary;
    automated.processBlock({ b, 1, frames - boundary });
    double maximumError = 0.0;
    for (int i = boundary; i < frames; ++i)
    {
        // Independent closed-form response of a 30 ms one-pole in dB.
        const double db = 15.0 * (1.0 - std::exp(-(i - boundary + 1.0) / (0.030 * 48000.0)));
        const double expected = baseline[static_cast<size_t>(i)] * std::pow(10.0, db / 20.0);
        maximumError = std::max(maximumError, std::abs(output[static_cast<size_t>(i)] - expected));
    }
    EXPECT_LT(maximumError, 1e-11);
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
    EXPECT_TRUE(dut->getSpeed() == TapeMachine<float>::Speed::IPS_3_75);
    EXPECT_TRUE(dut->getStandard() == TapeMachine<float>::Standard::CCIR);
    twin->setSpeed(TapeMachine<float>::Speed::IPS_3_75);
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

// Input gain and ceiling used to step once per block (a +12 dB drive move
// jumped straight to 4x at the block edge, zippering under automation). They
// now glide linearly over 20 ms at the processing rate.
DSPARK_TEST(Clipper_input_gain_glides_instead_of_stepping)
{
    auto c = std::make_unique<Clipper<float>>();
    c->setMode(Clipper<float>::Mode::Hard);
    c->prepare(spec(48000.0, 512, 1));
    auto tb = makeBuffer(1, 512);
    for (int blk = 0; blk < 12; ++blk)
    {
        if (blk == 10) c->setInputGain(12.0f);   // +12 dB, far below clipping
        for (int i = 0; i < 512; ++i)
            tb.ch(0)[i] = 0.05f * std::sin(6.2831853f * 441.0f * static_cast<float>(blk * 512 + i) / 48000.0f);
        std::vector<float> in(tb.ch(0), tb.ch(0) + 512);
        c->processBlock(tb.view());
        if (blk == 10)
        {
            // Effective gain early in the block: barely moved (old: ~3.98).
            double num = 0.0, den = 0.0;
            for (int i = 0; i < 48; ++i) { num += tb.ch(0)[i] * in[static_cast<size_t>(i)]; den += in[static_cast<size_t>(i)] * in[static_cast<size_t>(i)]; }
            EXPECT_LT(num / den, 1.3);
        }
        if (blk == 11)
        {
            // 20 ms later (960 samples) the full +12 dB is reached.
            double num = 0.0, den = 0.0;
            for (int i = 500; i < 512; ++i) { num += tb.ch(0)[i] * in[static_cast<size_t>(i)]; den += in[static_cast<size_t>(i)] * in[static_cast<size_t>(i)]; }
            EXPECT_NEAR(num / den, std::pow(10.0, 12.0 / 20.0), 1e-3);
        }
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
// 1x disables internal resampling/ADAA; latency is persisted by factor.
DSPARK_TEST(TubePreamp_oversampling_configurable_and_reported)
{
    TubePreamp<float> t;
    t.prepare(spec(48000.0, 512, 2));
    t.setOversampling(1); const int l1 = t.getLatency();
    EXPECT_EQ(t.getOversamplingFactor(), 1);
    EXPECT_EQ(l1, 0);
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
// Non-finite configuration is ignored too: prepare(NaN) used to pass the
// `<= 0` test and a NaN parameter passed std::max/std::clamp, each silencing
// the core for good.
DSPARK_TEST(Hysteresis_survives_nonfinite_input)
{
    Hysteresis<double> h;
    h.prepare(192000.0);
    h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, 0.17);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    h.prepare(nan);
    h.prepare(std::numeric_limits<double>::infinity());
    h.setParameters(nan, 2.2e4, 1.6e-3, 2.7e4, 0.17);
    h.setParameters(3.5e5, 2.2e4, 1.6e-3, 2.7e4, nan);
    EXPECT_NEAR(h.getSaturation(), 3.5e5, 0.0);
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

// ============================================================================
// TapeMachine: the machine around the tape
// ============================================================================

namespace {

using Tape10 = TapeMachine<float>;

std::vector<float> l10Tone(double hz, double amp, int n)
{
    std::vector<float> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = static_cast<float>(amp * std::sin(2.0 * 3.14159265358979323846 * hz * i / 48000.0));
    return v;
}

std::pair<std::vector<float>, std::vector<float>> l10Render(Tape10& tape, const std::vector<float>& left,
                                                            const std::vector<float>& right)
{
    std::vector<float> l = left, r = right;
    for (size_t off = 0; off < l.size(); off += 512)
    {
        const int n = static_cast<int>(std::min<size_t>(512, l.size() - off));
        float* p[2] = { l.data() + off, r.data() + off };
        tape.processBlock(AudioBufferView<float>(p, 2, n));
    }
    return { l, r };
}

template <class Setup>
std::pair<std::vector<float>, std::vector<float>> l10Run(Setup&& setup, const std::vector<float>& left,
                                                         const std::vector<float>& right)
{
    auto tape = std::make_unique<Tape10>();
    setup(*tape);
    tape->prepare(spec(48000.0, 512, 2));
    return l10Render(*tape, left, right);
}

// Level of a frequency over the second half, Hann window, in dB.
double l10Line(const std::vector<float>& v, double hz)
{
    const size_t from = v.size() / 2, n = v.size() - from;
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double ph = 2.0 * 3.14159265358979323846 * hz * static_cast<double>(i) / 48000.0;
        const double w = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * static_cast<double>(i) / static_cast<double>(n));
        re += w * static_cast<double>(v[from + i]) * std::cos(ph);
        im += w * static_cast<double>(v[from + i]) * std::sin(ph);
    }
    return 20.0 * std::log10(std::max(4.0 * std::sqrt(re * re + im * im) / static_cast<double>(n), 1e-15));
}

// Mean level of a noise between two frequencies (lines on a grid), in dB.
double l10Band(const std::vector<float>& v, double lo, double hi)
{
    double sum = 0.0;
    int count = 0;
    for (double f = lo; f <= hi; f *= 1.05, ++count) sum += std::pow(10.0, l10Line(v, f) / 10.0);
    return 10.0 * std::log10(sum / count);
}

double l10RmsDb(const std::vector<float>& v)
{
    double s = 0.0;
    for (size_t i = v.size() / 2; i < v.size(); ++i) s += static_cast<double>(v[i]) * v[i];
    return 10.0 * std::log10(std::max(s / static_cast<double>(v.size() - v.size() / 2), 1e-30));
}

// Largest minus smallest period of a tone over its nominal, second half.
double l10PitchSpread(const std::vector<float>& v, double hz)
{
    double prev = -1.0, lo = 1e9, hi = 0.0;
    for (size_t i = v.size() / 2; i + 1 < v.size(); ++i)
        if (v[i] <= 0.0f && v[i + 1] > 0.0f)
        {
            const double t = static_cast<double>(i) + static_cast<double>(v[i]) / (static_cast<double>(v[i]) - v[i + 1]);
            if (prev >= 0.0) { lo = std::min(lo, t - prev); hi = std::max(hi, t - prev); }
            prev = t;
        }
    return (hi - lo) / (48000.0 / hz);
}

// THD of a 1 kHz tone (harmonics 2 to 5), as a ratio.
double l10Thd(const std::vector<float>& v)
{
    const double f = l10Line(v, 1000.0);
    double sum = 0.0;
    for (int k = 2; k <= 5; ++k) sum += std::pow(10.0, (l10Line(v, 1000.0 * k) - f) / 10.0);
    return std::sqrt(sum);
}

} // namespace

// New: the speed transport. The reproduce pitch follows a speed state whose
// deviation scales with the depth and grows at slower speeds; with no depth it
// is steady, and a change of speed glides with no jump of pitch.
DSPARK_TEST(TapeMachine_speed_transport_follows_the_speed)
{
    const auto in = l10Tone(3000.0, 0.1, 48000 * 6);
    const auto spread = [&](Tape10::Speed sp, float depth) {
        return l10PitchSpread(l10Run([=](Tape10& t) {
            t.setTransport(Tape10::Transport::Speed);
            t.setSpeed(sp);
            t.setWowFlutter(depth);
            t.setLossEffects(0.0f);
        }, in, in).first, 3000.0);
    };
    EXPECT_LT(spread(Tape10::Speed::IPS_15, 0.0f), 1e-4);
    const double at15 = spread(Tape10::Speed::IPS_15, 1.0f);
    const double at75 = spread(Tape10::Speed::IPS_7_5, 1.0f);
    EXPECT_GT(at15, 0.0015);
    EXPECT_LT(at15, 0.015);
    EXPECT_GT(at75, 1.2 * at15);

    auto tape = std::make_unique<Tape10>();
    tape->setTransport(Tape10::Transport::Speed);
    tape->setWowFlutter(0.0f);
    tape->prepare(spec(48000.0, 512, 2));
    EXPECT_TRUE(tape->getTransport() == Tape10::Transport::Speed);
    std::vector<float> l = l10Tone(3000.0, 0.1, 48000 * 4), r = l;
    for (size_t off = 0; off < l.size(); off += 512)
    {
        if (off == 512 * 94) tape->setSpeed(Tape10::Speed::IPS_7_5);
        float* p[2] = { l.data() + off, r.data() + off };
        tape->processBlock(AudioBufferView<float>(p, 2, static_cast<int>(std::min<size_t>(512, l.size() - off))));
    }
    for (float x : l) EXPECT_TRUE(std::isfinite(x));
    EXPECT_LT(l10PitchSpread(l, 3000.0), 0.01);
}

// New: the hiss is laid on the tape, so it comes out through the reproduce
// head and the de-emphasis (darker than white noise), at the level set, on
// independent channels, and moved by the track width and the operating level;
// the hum comes after the tape, in phase, with its second harmonic 6 dB down.
DSPARK_TEST(TapeMachine_hiss_is_read_off_the_tape_and_hum_comes_after)
{
    const std::vector<float> silence(96000, 0.0f);
    const auto hiss = [&](auto&& setup) {
        return l10Run([&](Tape10& t) { t.setNoise(-50.0f); setup(t); }, silence, silence);
    };
    const auto plain = hiss([](Tape10&) {});
    EXPECT_NEAR(l10RmsDb(plain.first), -50.0, 1.0);
    EXPECT_GT(l10Band(plain.first, 200.0, 1000.0) - l10Band(plain.first, 6000.0, 14000.0), 4.0);
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 48000; i < 96000; ++i)
    {
        ab += static_cast<double>(plain.first[i]) * plain.second[i];
        aa += static_cast<double>(plain.first[i]) * plain.first[i];
        bb += static_cast<double>(plain.second[i]) * plain.second[i];
    }
    EXPECT_LT(std::abs(ab / std::sqrt(aa * bb)), 0.1);
    EXPECT_NEAR(l10RmsDb(hiss([](Tape10& t) { t.setTrackFormat(Tape10::TrackFormat::Full); }).first) - l10RmsDb(plain.first), -5.0, 0.7);
    EXPECT_NEAR(l10RmsDb(hiss([](Tape10& t) { t.setTrackFormat(Tape10::TrackFormat::Quarter); }).first) - l10RmsDb(plain.first), 3.0, 0.7);
    EXPECT_NEAR(l10RmsDb(hiss([](Tape10& t) { t.setCalibrationLevel(6.0f); }).first) - l10RmsDb(plain.first), -6.0, 0.7);

    for (int m = 0; m < 2; ++m)
    {
        const auto hum = l10Run([=](Tape10& t) { t.setHum(-60.0f, static_cast<Tape10::Mains>(m)); }, silence, silence);
        const double f = m ? 60.0 : 50.0;
        EXPECT_NEAR(l10RmsDb(hum.first), -60.0, 0.3);
        EXPECT_NEAR(l10Line(hum.first, f) - l10Line(hum.first, 2.0 * f), 6.02, 0.3);
        double worst = 0.0;
        for (size_t i = 0; i < silence.size(); ++i) worst = std::max(worst, std::abs(static_cast<double>(hum.first[i] - hum.second[i])));
        EXPECT_EQ(worst, 0.0);
    }
}

// New: crosstalk leaks each channel of a pair into the other at the level set
// at 1 kHz, rising 6 dB per octave up to 10 kHz; off, nothing leaks.
DSPARK_TEST(TapeMachine_channel_bleed_rises_with_frequency)
{
    const std::vector<float> silence(72000, 0.0f);
    const auto leak = [&](double hz, bool on) {
        const auto out = l10Run([=](Tape10& t) { if (on) t.setChannelBleed(-50.0f); }, l10Tone(hz, 0.25, 72000), silence);
        return l10Line(out.second, hz) - l10Line(out.first, hz);
    };
    EXPECT_NEAR(leak(1000.0, true), -50.0, 1.0);
    EXPECT_NEAR(leak(10000.0, true), -50.0 + 18.39, 1.0);   // the high-pass corner kept at 10 kHz
    EXPECT_LT(leak(1000.0, false), -110.0);
}

// New: the even asymmetry raises the second harmonic in proportion to its
// amount; the transformer saturates the low end and leaves 1 kHz alone, and
// once its share has glided back to 0 the output is that of a machine that
// never had it.
DSPARK_TEST(TapeMachine_asymmetry_and_transformer_add_their_harmonics)
{
    const auto tone = l10Tone(1000.0, 0.25, 72000);
    const auto h2 = [&](float a) {
        const auto out = l10Run([=](Tape10& t) { t.setEvenHarmonics(a); }, tone, tone).first;
        return l10Line(out, 2000.0) - l10Line(out, 1000.0);
    };
    const double none = h2(0.0f), half = h2(0.5f), full = h2(1.0f);
    EXPECT_GT(full, none + 20.0);
    EXPECT_NEAR(full - half, 6.02, 1.5);

    // The tape cancels its own even orders, so the transformer's second
    // harmonic stands out at 40 Hz (measured -116 -> -41 dBc at -6 dBFS) and
    // stays 20 dB lower at 1 kHz, where the fundamental and the third are left
    // as they were.
    const auto h2Of = [&](double hz, float a) {
        const auto in = l10Tone(hz, 0.5, 96000);
        const auto out = l10Run([=](Tape10& t) { t.setOutputIron(a); }, in, in).first;
        return l10Line(out, 2.0 * hz) - l10Line(out, hz);
    };
    EXPECT_GT(h2Of(40.0, 1.0f), h2Of(40.0, 0.0f) + 40.0);
    EXPECT_GT(h2Of(40.0, 1.0f), h2Of(1000.0, 1.0f) + 15.0);
    const auto mid = l10Tone(1000.0, 0.5, 72000);
    const auto plainMid = l10Run([](Tape10&) {}, mid, mid).first;
    const auto bentMid = l10Run([](Tape10& t) { t.setOutputIron(1.0f); }, mid, mid).first;
    EXPECT_NEAR(l10Line(bentMid, 1000.0), l10Line(plainMid, 1000.0), 0.1);
    EXPECT_NEAR(l10Line(bentMid, 3000.0), l10Line(plainMid, 3000.0), 0.5);

    auto a = std::make_unique<Tape10>(), b = std::make_unique<Tape10>();
    a->setOutputIron(1.0f);
    a->prepare(spec(48000.0, 512, 2));
    b->prepare(spec(48000.0, 512, 2));
    std::vector<float> la = l10Tone(60.0, 0.4, 96000), ra = la, lb = la, rb = la;
    for (size_t off = 0; off < la.size(); off += 512)
    {
        if (off == 512 * 40) a->setOutputIron(0.0f);
        float* pa[2] = { la.data() + off, ra.data() + off };
        float* pb[2] = { lb.data() + off, rb.data() + off };
        a->processBlock(AudioBufferView<float>(pa, 2, static_cast<int>(std::min<size_t>(512, la.size() - off))));
        b->processBlock(AudioBufferView<float>(pb, 2, static_cast<int>(std::min<size_t>(512, lb.size() - off))));
    }
    double worst = 0.0;
    for (size_t i = 48000; i < la.size(); ++i) worst = std::max(worst, std::abs(static_cast<double>(la[i] - lb[i])));
    EXPECT_EQ(worst, 0.0);
}

// New: the head contour, the circuit electronics and the alignment trims,
// at low level: the contour dips where the plain bump does not, the circuit
// matches the ideal electronics in the band and rolls off a little at the top,
// and the trims move the high and low ends by about their amount.
DSPARK_TEST(TapeMachine_head_contour_circuit_and_trims)
{
    const auto response = [](auto&& setup, double hz) {
        const auto in = l10Tone(hz, 0.0316, 36000);
        return l10Line(l10Run(setup, in, in).first, hz) - l10Line(in, hz);
    };
    const auto bump = [](Tape10& t) { t.setHeadBump(1.0f); };
    const auto contour = [](Tape10& t) { t.setHeadBump(1.0f); t.setHeadModel(Tape10::HeadModel::Contour); };
    EXPECT_LT(response(contour, 130.0) - response(bump, 130.0), -0.5);
    EXPECT_NEAR(response(contour, 1000.0), response(bump, 1000.0), 0.1);

    const auto ideal = [](Tape10&) {};
    const auto circuit = [](Tape10& t) { t.setElectronics(Tape10::Electronics::Circuit); };
    EXPECT_NEAR(response(circuit, 1000.0), response(ideal, 1000.0), 0.3);
    EXPECT_NEAR(response(circuit, 100.0), response(ideal, 100.0), 0.5);
    EXPECT_LT(response(circuit, 20000.0), response(ideal, 20000.0) - 0.2);

    const double hfTrim = response([](Tape10& t) { t.setAlignment(3.0f, 0.0f); }, 16000.0) - response(ideal, 16000.0);
    const double lfTrim = response([](Tape10& t) { t.setAlignment(0.0f, -3.0f); }, 31.5) - response(ideal, 31.5);
    EXPECT_GT(hfTrim, 2.4);
    EXPECT_LT(hfTrim, 3.3);
    EXPECT_GT(lfTrim, -3.3);
    EXPECT_LT(lfTrim, -2.2);
}

// New: the operating level drives the tape harder at the same output level
// (more distortion at -12 dBFS), the formulations move the distortion the way
// their headroom says, the input monitor is the input delayed like the tape,
// and the sync head loses more of the top end than the reproduce head.
DSPARK_TEST(TapeMachine_calibration_formulation_and_monitor)
{
    const auto tone = l10Tone(1000.0, 0.25, 72000);
    const auto plain = l10Run([](Tape10&) {}, tone, tone).first;
    const auto hot = l10Run([](Tape10& t) { t.setCalibrationLevel(6.0f); }, tone, tone).first;
    EXPECT_GT(l10Thd(hot), 2.5 * l10Thd(plain));
    EXPECT_NEAR(l10Line(hot, 1000.0), l10Line(plain, 1000.0), 0.5);
    const auto high = l10Run([](Tape10& t) { t.setFormulation(Tape10::Formulation::HighOutput); }, tone, tone).first;
    const auto vintage = l10Run([](Tape10& t) { t.setFormulation(Tape10::Formulation::Vintage); }, tone, tone).first;
    EXPECT_LT(l10Thd(high), l10Thd(plain));
    EXPECT_GT(l10Thd(vintage), l10Thd(plain));

    std::vector<float> prog(48000);
    uint32_t seed = 5u;
    for (auto& x : prog) { seed = seed * 1664525u + 1013904223u; x = 0.2f * (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f); }
    auto input = std::make_unique<Tape10>();
    input->setMonitor(Tape10::Monitor::Source);
    input->prepare(spec(48000.0, 512, 2));
    const auto monitored = l10Render(*input, prog, prog).first;
    const size_t lat = static_cast<size_t>(input->getLatency());
    double worst = 0.0;
    for (size_t i = lat; i < prog.size(); ++i) worst = std::max(worst, std::abs(static_cast<double>(monitored[i] - prog[i - lat])));
    EXPECT_LT(worst, 1e-6);

    const auto top = [](auto&& setup) {
        const auto in = l10Tone(15000.0, 0.0316, 36000);
        return l10Line(l10Run(setup, in, in).first, 15000.0);
    };
    EXPECT_LT(top([](Tape10& t) { t.setMonitor(Tape10::Monitor::RecordHead); }), top([](Tape10&) {}) - 0.3);
}

// New: the controls of the machine around the tape clamp what is out of range
// and ignore what is not finite.
DSPARK_TEST(TapeMachine_machine_controls_clamp_and_ignore_invalid_values)
{
    Tape10 t;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    t.setHum(nan);
    t.setChannelBleed(std::numeric_limits<float>::infinity());
    t.setEvenHarmonics(nan);
    t.setOutputIron(-1.0f);
    t.setAlignment(nan, 2.0f);
    t.setCalibrationLevel(20.0f);
    t.setTransport(static_cast<Tape10::Transport>(7));
    t.setMonitor(static_cast<Tape10::Monitor>(-3));
    t.setFormulation(static_cast<Tape10::Formulation>(9));
    t.setTrackFormat(static_cast<Tape10::TrackFormat>(5));
    t.setHeadModel(static_cast<Tape10::HeadModel>(4));
    t.setElectronics(static_cast<Tape10::Electronics>(-1));
    EXPECT_EQ(t.getHum(), -200.0f);
    EXPECT_EQ(t.getChannelBleed(), -200.0f);
    EXPECT_EQ(t.getEvenHarmonics(), 0.0f);
    EXPECT_EQ(t.getOutputIron(), 0.0f);
    EXPECT_EQ(t.getAlignmentHf(), 0.0f);
    EXPECT_EQ(t.getAlignmentLf(), 0.0f);
    EXPECT_EQ(t.getCalibrationLevel(), 9.0f);
    EXPECT_TRUE(t.getTransport() == Tape10::Transport::Speed);
    EXPECT_TRUE(t.getMonitor() == Tape10::Monitor::PlayHead);
    EXPECT_TRUE(t.getFormulation() == Tape10::Formulation::Vintage);
    EXPECT_TRUE(t.getTrackFormat() == Tape10::TrackFormat::Quarter);
    EXPECT_TRUE(t.getHeadModel() == Tape10::HeadModel::Contour);
    EXPECT_TRUE(t.getElectronics() == Tape10::Electronics::Ideal);
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

DSPARK_TEST(FilterEngine_granular_shelf_slope_setter)
{
    // setShelfSlope() is the one-parameter counterpart of the slope argument
    // of setLowShelf(): same audible result, and a NaN slope (which would pass
    // the design clamp and poison every coefficient) is ignored.
    FilterEngine<float> a, b;
    a.prepare(spec(48000.0, 512, 1));
    b.prepare(spec(48000.0, 512, 1));
    a.setLowShelf(200.0f, 12.0f, 0.4f);
    b.setLowShelf(200.0f, 12.0f);
    b.setShelfSlope(0.4f);
    EXPECT_NEAR(b.getShelfSlope(), 0.4f, 1e-7f);
    EXPECT_NEAR(feEnergy(a, kShelfSettle), feEnergy(b, kShelfSettle), 1e-9);

    b.setShelfSlope(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(b.getShelfSlope(), 0.4f, 1e-7f);
    EXPECT_TRUE(std::isfinite(feEnergy(b, 4)));

    Equalizer<float> eq;
    EXPECT_TRUE(eq.isMatchedBells());   // matched bells are the default
    eq.setMatchedBells(false);
    EXPECT_FALSE(eq.isMatchedBells());
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
// Concurrent publication pin for Effects/Equalizer.h.
//
// Why this pin can actually fail, which is the whole point of it:
// before the band configs became a seqlock, this exact schedule adopted a
// config that was never published -- several hundred thousand mixed configs
// per five-second run, roughly one adoption in thirty, on every attempt.
// The two configs below differ
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

// A pinned drift seed makes the analog drift reproducible: two instances with
// the same seed render identical output, a reset() instance replays its own
// first render, and a different seed or the default unique seed drifts
// differently. Without the seed every instance draws its own sequence and no
// render of a drifting filter could be repeated.
DSPARK_TEST(FilterEngine_drift_seed_makes_renders_reproducible)
{
    constexpr int kBlock = 512, kBlocks = 40;
    auto input = [](int n) {
        std::vector<float> x(static_cast<size_t>(n));
        uint32_t s = 12345u;
        for (auto& v : x) { s = s * 1664525u + 1013904223u; v = static_cast<float>(s >> 8) / 16777216.0f - 0.5f; }
        return x;
    };
    const std::vector<float> x = input(kBlock * kBlocks);
    auto render = [&](FilterEngine<float>& f) {
        std::vector<float> out;
        auto tb = makeStereoBuffer(kBlock);
        for (int b = 0; b < kBlocks; ++b)
        {
            for (int c = 0; c < 2; ++c)
                std::copy_n(x.data() + b * kBlock, kBlock, tb.ch(c));
            f.processBlock(tb.view());
            out.insert(out.end(), tb.ch(0), tb.ch(0) + kBlock);
        }
        return out;
    };
    auto make = [](std::unique_ptr<FilterEngine<float>>& f, std::uint64_t seed) {
        f = std::make_unique<FilterEngine<float>>();
        f->prepare(defaultSpec());
        f->setLowPass(1200.0f, 2.0f, 24);
        f->enableAnalogDrift(AnalogRandom::AnalogComponent::TapeMachine, 1.0f);
        f->setDriftSeed(seed);
    };

    std::unique_ptr<FilterEngine<float>> a, b, c, d;
    make(a, 42u); make(b, 42u); make(c, 7u); make(d, 0u);
    EXPECT_TRUE(a->getDriftSeed() == 42u);
    EXPECT_TRUE(d->getDriftSeed() == 0u);
    const auto ya = render(*a), yb = render(*b), yc = render(*c), yd = render(*d);
    EXPECT_TRUE(ya == yb);
    EXPECT_FALSE(ya == yc);
    EXPECT_FALSE(ya == yd);

    // The drift is audible in the output, so the equalities above test it.
    std::unique_ptr<FilterEngine<float>> still;
    make(still, 42u);
    still->disableAnalogDrift();
    EXPECT_FALSE(render(*still) == ya);

    // reset() restarts a pinned drift whatever ran before it: an instance
    // that has rendered once and one that has rendered three times replay the
    // same output after a reset.
    std::unique_ptr<FilterEngine<float>> e;
    make(e, 42u);
    for (int k = 0; k < 3; ++k) (void)render(*e);
    a->reset();
    e->reset();
    const auto ra = render(*a);
    EXPECT_TRUE(render(*e) == ra);
}

// ============================================================================
// Equalizer: the analog response (bands designed from their prototypes and run
// in a cascade with ramped coefficients)
// ============================================================================

namespace {

using AnalogEq = Equalizer<double>;

// Runs a signal through an equalizer in blocks (one channel).
void eqRun(AnalogEq& eq, std::vector<double>& x, int block = 256)
{
    const int total = static_cast<int>(x.size());
    for (int off = 0; off < total; off += block)
    {
        double* ch[1] = { x.data() + off };
        eq.processBlock(AudioBufferView<double>(ch, 1, std::min(block, total - off)));
    }
}

// Level in decibels of a response at a frequency (the sum of its discrete Fourier transform there).
double eqLevelDb(const std::vector<double>& h, double hz, double fs)
{
    const double w = 2.0 * 3.14159265358979323846 * hz / fs;
    double re = 0.0, im = 0.0;
    for (size_t n = 0; n < h.size(); ++n)
    {
        re += h[n] * std::cos(w * static_cast<double>(n));
        im -= h[n] * std::sin(w * static_cast<double>(n));
    }
    return 10.0 * std::log10(std::max(re * re + im * im, 1e-30));
}

// The response an equalizer gives to an impulse.
std::vector<double> eqImpulse(AnalogEq& eq, int length = 16384)
{
    std::vector<double> h(static_cast<size_t>(length), 0.0);
    h[0] = 1.0;
    eqRun(eq, h);
    return h;
}

AnalogEq::BandConfig eqBand(AnalogEq::BandType type, double hz, double gain, double q, double slope = 12.0, int order = 1)
{
    AnalogEq::BandConfig cfg;
    cfg.type = type;
    cfg.frequency = hz;
    cfg.gain = gain;
    cfg.q = q;
    cfg.slope = slope;
    cfg.order = static_cast<float>(order);
    return cfg;
}

// Worst distance in decibels, over a list of frequencies, between what an equalizer with one band plays
// and an analog prototype.
double eqWorstAgainst(const AnalogCascade& proto, const AnalogEq::BandConfig& cfg, bool analog,
                      const std::vector<double>& freqs, double fs = 48000.0, bool linearPhase = false)
{
    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setAnalogResponse(analog);
    if (linearPhase) eq.setFilterMode(AnalogEq::FilterMode::LinearPhase);
    eq.setBand(0, cfg);
    eq.prepare({ fs, 256, 1 });
    const std::vector<double> h = eqImpulse(eq, linearPhase ? 3 * 32768 : 16384);
    double worst = 0.0;
    for (const double hz : freqs)
    {
        const double want = proto.magnitudeDb(hz);
        if (want > -70.0) worst = std::max(worst, std::abs(eqLevelDb(h, hz, fs) - want));
    }
    return worst;
}

AnalogCascade eqOne(const AnalogSection& section)
{
    AnalogCascade c;
    c.add(section);
    return c;
}

} // namespace

// The gap: next to Nyquist the cookbook bands shrink (a high shelf reaches its
// plateau early, a bell narrows). With the analog response the audio stays on
// the analog curve in both phase modes.
DSPARK_TEST(Equalizer_analog_response_stays_on_the_analog_curves)
{
    using Type = AnalogEq::BandType;
    const std::vector<double> top = { 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000 };
    const std::vector<double> wide = { 50, 200, 1000, 4000, 8000, 12000, 16000, 19000 };

    // A high shelf of +12 dB at 12 kHz (Q 0.7071 is the plain shelf).
    {
        const AnalogEq::BandConfig cfg = eqBand(Type::HighShelf, 12000.0, 12.0, 0.7071);
        const AnalogCascade proto = eqOne(AnalogSection::highShelf(12000.0, 12.0));
        const double plain = eqWorstAgainst(proto, cfg, false, top), analog = eqWorstAgainst(proto, cfg, true, top);
        EXPECT_GT(plain, 0.8);
        EXPECT_LT(analog, 0.15);
        EXPECT_LT(eqWorstAgainst(proto, cfg, true, wide), 0.15);
    }
    // A bell of -18 dB at 15 kHz, Q 1: the closed-form bell the equalizer uses by default is off by decibels.
    {
        const AnalogEq::BandConfig cfg = eqBand(Type::Peak, 15000.0, -18.0, 1.0);
        const AnalogCascade proto = eqOne(AnalogSection::peak(15000.0, 1.0, -18.0));
        const double before = eqWorstAgainst(proto, cfg, false, top), analog = eqWorstAgainst(proto, cfg, true, top);
        EXPECT_GT(before, 1.0);
        EXPECT_LT(analog, 0.3);
        EXPECT_LT(3.0 * analog, before);
    }
    // A notch at 14 kHz and a tilt: the cookbook forms have no matched design at all.
    {
        const AnalogEq::BandConfig cfg = eqBand(Type::Notch, 14000.0, 0.0, 2.0);
        const AnalogCascade proto = eqOne(AnalogSection::notch(14000.0, 2.0));
        const std::vector<double> around = { 6000, 9000, 11000, 12500, 13500, 14500, 15500, 17000, 19000 };
        EXPECT_GT(eqWorstAgainst(proto, cfg, false, around), 1.0);
        EXPECT_LT(eqWorstAgainst(proto, cfg, true, around), 0.5);
    }
    // The linear-phase kernel is built from the prototype itself.
    {
        const AnalogEq::BandConfig cfg = eqBand(Type::HighShelf, 12000.0, 12.0, 0.7071);
        const AnalogCascade proto = eqOne(AnalogSection::highShelf(12000.0, 12.0));
        EXPECT_GT(eqWorstAgainst(proto, cfg, false, top, 48000.0, true), 0.8);
        EXPECT_LT(eqWorstAgainst(proto, cfg, true, top, 48000.0, true), 0.05);
    }
}

// A cut takes any slope from 0 to 96 dB per octave with the analog response;
// the cookbook engine keeps its whole slopes from 6 to 48.
DSPARK_TEST(Equalizer_analog_cuts_take_any_slope)
{
    using Type = AnalogEq::BandType;
    const std::vector<double> freqs = { 100, 500, 1000, 1500, 2000, 2500, 3000, 4000, 6000, 8000, 12000, 16000 };
    double before = 0.0;
    for (const double slope : { 3.0, 9.0, 15.0, 30.0, 45.5, 72.0, 96.0 })
    {
        const AnalogEq::BandConfig cfg = eqBand(Type::LowPass, 2000.0, 0.0, 0.7071, slope);
        EXPECT_LT(eqWorstAgainst(FilterDesign::highCut(2000.0, slope), cfg, true, freqs), 0.3);

        auto eqHeap = std::make_unique<AnalogEq>();
        AnalogEq& eq = *eqHeap;
        eq.setAnalogResponse(true);
        eq.setBand(0, cfg);
        eq.prepare({ 48000.0, 256, 1 });
        EXPECT_NEAR(static_cast<double>(eq.getBandConfig(0).slope), slope, 1e-9);
        const std::vector<double> h = eqImpulse(eq);
        if (slope >= 6.0) EXPECT_NEAR(eqLevelDb(h, 2000.0, 48000.0), -3.0103, 0.05);
        // The steeper the slope, the lower an octave above the corner.
        const double octave = eqLevelDb(h, 4000.0, 48000.0);
        if (before != 0.0) EXPECT_LT(octave, before - 1.0);
        before = octave;
    }
    // A low cut of 30 dB per octave at 200 Hz, with resonance.
    {
        const AnalogEq::BandConfig cfg = eqBand(Type::HighPass, 200.0, 0.0, 2.0, 30.0);
        const std::vector<double> low = { 40, 80, 120, 160, 200, 260, 400, 1000, 5000, 15000 };
        EXPECT_LT(eqWorstAgainst(FilterDesign::lowCut(200.0, 30.0, 2.0), cfg, true, low), 0.3);
    }
    // The cookbook engine runs a slope past its range as its steepest, and a slope in between as the nearest whole one.
    {
        AnalogEq steep, steepest, between, whole;
        steep.setBand(0, eqBand(Type::LowPass, 2000.0, 0.0, 0.7071, 96.0));
        steepest.setBand(0, eqBand(Type::LowPass, 2000.0, 0.0, 0.7071, 48.0));
        between.setBand(0, eqBand(Type::LowPass, 2000.0, 0.0, 0.7071, 23.6));
        whole.setBand(0, eqBand(Type::LowPass, 2000.0, 0.0, 0.7071, 24.0));
        for (AnalogEq* eq : { &steep, &steepest, &between, &whole }) eq->prepare({ 48000.0, 256, 1 });
        EXPECT_TRUE(eqImpulse(steep, 2048) == eqImpulse(steepest, 2048));
        EXPECT_TRUE(eqImpulse(between, 2048) == eqImpulse(whole, 2048));
        EXPECT_EQ(static_cast<double>(steep.getBandConfig(0).slope), 96.0);
    }
}

// Bells, shelves, notches and band-passes of any order, the all-pass, the flat
// tilt and the walls. A band only the analog response can make runs with it
// whatever the switch says.
DSPARK_TEST(Equalizer_analog_bands_of_any_order_and_the_new_types)
{
    using Type = AnalogEq::BandType;
    for (const bool analog : { false, true })
    {
        const double half = 0.25, root = std::sqrt(1.0 + half * half);      // Q 2
        const std::vector<double> bell = { 300, 1000, 2000, 3000 * (root - half), 3000, 3000 * (root + half), 5000, 10000, 18000 };
        EXPECT_LT(eqWorstAgainst(FilterDesign::bell(3000.0, 2.0, 12.0, 4), eqBand(Type::Peak, 3000.0, 12.0, 2.0, 12.0, 4), analog, bell), 0.05);
        EXPECT_LT(eqWorstAgainst(FilterDesign::lowShelf(400.0, -9.0, 6), eqBand(Type::LowShelf, 400.0, -9.0, 0.7071, 12.0, 6), analog, bell), 0.05);
        EXPECT_LT(eqWorstAgainst(FilterDesign::highShelf(6000.0, 6.0, 3), eqBand(Type::HighShelf, 6000.0, 6.0, 0.7071, 12.0, 3), analog, bell), 0.1);
        EXPECT_LT(eqWorstAgainst(FilterDesign::bandPass(3000.0, 2.0, 3), eqBand(Type::BandPass, 3000.0, 0.0, 2.0, 12.0, 3), analog, bell), 0.7);
        EXPECT_LT(eqWorstAgainst(FilterDesign::bandStop(3000.0, 2.0, 2), eqBand(Type::Notch, 3000.0, 0.0, 2.0, 12.0, 2), analog,
                                 { 300, 1000, 2000, 2400, 3700, 5000, 10000, 18000 }), 0.1);
        EXPECT_LT(eqWorstAgainst(FilterDesign::tilt(1000.0, 3.0), eqBand(Type::ConstantTilt, 1000.0, 3.0, 0.7071), analog,
                                 { 30, 100, 250, 1000, 4000, 10000, 16000, 19000 }), 0.1);
        EXPECT_LT(eqWorstAgainst(FilterDesign::ellipticHighCut(8000.0), eqBand(Type::LowPassWall, 8000.0, 0.0, 0.7071), analog,
                                 { 100, 1000, 4000, 6000, 7000, 7600, 7900, 8000 }), 0.06);    // inside its ripple of 0.05 dB
        EXPECT_LT(eqWorstAgainst(FilterDesign::ellipticLowCut(200.0), eqBand(Type::HighPassWall, 200.0, 0.0, 0.7071), analog,
                                 { 205, 220, 300, 1000, 8000, 19000 }), 0.1);

        // The wall is down by its floor a quarter of an octave past its corner.
        AnalogEq wall;
        wall.setAnalogResponse(analog);
        wall.setBand(0, eqBand(Type::LowPassWall, 8000.0, 0.0, 0.7071));
        wall.prepare({ 48000.0, 256, 1 });
        const std::vector<double> h = eqImpulse(wall);
        for (const double hz : { 9400.0, 9600.0, 11000.0, 14000.0, 18000.0, 23000.0 }) EXPECT_LT(eqLevelDb(h, hz, 48000.0), -99.0);

        // An all-pass keeps the level at every frequency and turns the phase.
        for (const int order : { 1, 2, 5 })
        {
            AnalogEq all;
            all.setAnalogResponse(analog);
            all.setBand(0, eqBand(Type::AllPass, 1000.0, 0.0, 1.5, 12.0, order));
            all.prepare({ 48000.0, 256, 1 });
            const std::vector<double> a = eqImpulse(all, 32768);
            for (const double hz : { 60.0, 500.0, 1000.0, 2000.0, 9000.0, 20000.0 }) EXPECT_NEAR(eqLevelDb(a, hz, 48000.0), 0.0, 1e-6);
            EXPECT_TRUE(std::abs(a[0]) < 0.999);      // not the identity
        }
    }
    // The types of the cookbook engine keep to it with the switch off.
    {
        AnalogEq plain;
        plain.setBand(0, eqBand(Type::Peak, 3000.0, 12.0, 2.0));
        plain.prepare({ 48000.0, 256, 1 });
        BiquadCoeffs sections[16];
        EXPECT_EQ(plain.buildBandSections(plain.getBandConfig(0), sections, 16), 1);
        EXPECT_EQ(plain.buildBandSections(eqBand(Type::Peak, 3000.0, 12.0, 2.0, 12.0, 4), sections, 16), 4);
        EXPECT_EQ(plain.buildBandSections(eqBand(Type::LowPassWall, 3000.0, 0.0, 0.7071), sections, 16), 6);
        EXPECT_EQ(plain.buildBandSections(eqBand(Type::Peak, 3000.0, 12.0, 2.0, 12.0, 16), sections, 4), 4);
    }
}

// The curve an interface draws is the audio, section for section.
DSPARK_TEST(Equalizer_analog_analysis_curve_is_the_audio)
{
    using Type = AnalogEq::BandType;
    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setAnalogResponse(true);
    eq.setBand(0, eqBand(Type::HighPass, 80.0, 0.0, 1.2, 21.0));
    eq.setBand(1, eqBand(Type::Peak, 2500.0, -7.0, 3.0, 12.0, 5));
    eq.setBand(2, eqBand(Type::HighShelf, 9000.0, 8.0, 0.9));
    eq.setBand(3, eqBand(Type::Tilt, 700.0, 4.0, 0.7071));
    eq.setBand(4, eqBand(Type::LowPass, 17000.0, 0.0, 0.7071, 13.0));
    eq.prepare({ 48000.0, 256, 1 });
    const std::vector<double> h = eqImpulse(eq, 65536);
    const double freqs[] = { 40, 80, 150, 700, 2000, 2500, 3200, 9000, 15000, 19000 };
    double drawn[10];
    eq.getMagnitudeForFrequencyArray(freqs, drawn, 10);
    for (int i = 0; i < 10; ++i)
        EXPECT_NEAR(eqLevelDb(h, freqs[i], 48000.0), 20.0 * std::log10(drawn[i]), 2e-3);
}

// A change of a band glides: nothing steps, a small move takes its time and a
// jump is quick, and the result does not depend on how the host cuts the
// stream.
DSPARK_TEST(Equalizer_analog_parameter_changes_glide)
{
    using Type = AnalogEq::BandType;
    const double fs = 48000.0;
    const auto tone = [&](int n) { return 0.5 * std::sin(2.0 * 3.14159265358979323846 * 100.0 * n / fs); };
    const auto run = [&](double toGain, int block, std::vector<double>& out) {
        auto eqHeap = std::make_unique<AnalogEq>();
        AnalogEq& eq = *eqHeap;
        eq.setAnalogResponse(true);
        eq.setBand(0, eqBand(Type::LowShelf, 400.0, 0.0, 0.7071));
        eq.prepare({ fs, 512, 1 });
        out.resize(9600);
        for (size_t n = 0; n < out.size(); ++n) out[n] = tone(static_cast<int>(n));
        for (int off = 0; off < 9600; off += block)
        {
            if (off == 2400) eq.setBand(0, eqBand(Type::LowShelf, 400.0, toGain, 0.7071));
            double* ch[1] = { out.data() + off };
            eq.processBlock(AudioBufferView<double>(ch, 1, std::min(block, 9600 - off)));
        }
    };
    std::vector<double> jump, small, cut;
    run(12.0, 240, jump);
    run(0.5, 240, small);
    run(12.0, 48, cut);
    EXPECT_TRUE(jump == cut);                                  // the same for any partition

    // No step anywhere: the largest move between samples is that of the tone at its final level.
    const double steady = 0.5 * std::pow(10.0, 12.0 / 20.0) * 2.0 * 3.14159265358979323846 * 100.0 / fs;
    double largest = 0.0;
    for (size_t n = 1; n < jump.size(); ++n) largest = std::max(largest, std::abs(jump[n] - jump[n - 1]));
    EXPECT_LT(largest, 1.1 * steady);

    // Before the change the band rests: the tone passes untouched.
    for (int n = 0; n < 2400; ++n) EXPECT_TRUE(jump[static_cast<size_t>(n)] == tone(n));
    // The jump has landed in 20 ms; the small move is still on its way then and lands in 60.
    const auto gainAt = [&](const std::vector<double>& y, int from) {
        double in = 0.0, out = 0.0;
        for (int n = from; n < from + 480; ++n) { in += tone(n) * tone(n); out += y[static_cast<size_t>(n)] * y[static_cast<size_t>(n)]; }
        return 10.0 * std::log10(out / in);
    };
    EXPECT_NEAR(gainAt(jump, 2400 + 960), 11.9, 0.3);
    EXPECT_LT(gainAt(small, 2400 + 480), 0.45);
    EXPECT_NEAR(gainAt(small, 2400 + 2880), 0.5, 0.02);
}

// A gain band at 0 dB is out of the path: the samples pass untouched, and it
// comes back and leaves again without a step.
DSPARK_TEST(Equalizer_analog_band_at_zero_gain_leaves_the_path)
{
    using Type = AnalogEq::BandType;
    const double fs = 48000.0;
    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setAnalogResponse(true);
    eq.setBand(0, eqBand(Type::Peak, 1000.0, 0.0, 2.0));
    eq.setBand(1, eqBand(Type::HighShelf, 6000.0, 0.01, 0.7071));
    eq.setBand(2, eqBand(Type::ConstantTilt, 1000.0, 0.0, 0.7071));
    eq.prepare({ fs, 256, 1 });
    uint32_t seed = 77u;
    const auto noise = [&]() { seed = seed * 1664525u + 1013904223u; return (static_cast<double>(seed >> 8) / 8388608.0 - 1.0) * 0.4; };

    std::vector<double> x(4800), y;
    for (double& v : x) v = noise();
    y = x;
    eqRun(eq, y);
    EXPECT_TRUE(y == x);

    // In: 6 dB at 1 kHz. Out again: untouched once the glide has landed and its tail has gone.
    eq.setBand(0, eqBand(Type::Peak, 1000.0, 6.0, 2.0));
    std::vector<double> in(9600);
    for (size_t n = 0; n < in.size(); ++n) in[n] = 0.3 * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * static_cast<double>(n) / fs);
    y = in;
    eqRun(eq, y);
    double peak = 0.0, step = 0.0;
    for (size_t n = 4800; n < y.size(); ++n) peak = std::max(peak, std::abs(y[n]));
    for (size_t n = 1; n < y.size(); ++n) step = std::max(step, std::abs(y[n] - y[n - 1]));
    EXPECT_NEAR(20.0 * std::log10(peak / 0.3), 6.0, 0.05);
    EXPECT_LT(step, 1.1 * 0.3 * 2.0 * 2.0 * 3.14159265358979323846 * 1000.0 / fs);

    eq.setBand(0, eqBand(Type::Peak, 1000.0, 0.0, 2.0));
    for (size_t n = 0; n < in.size(); ++n) in[n] = 0.3 * std::sin(2.0 * 3.14159265358979323846 * 1000.0 * static_cast<double>(n + 9600) / fs);
    y = in;
    eqRun(eq, y);
    step = 0.0;
    for (size_t n = 1; n < y.size(); ++n) step = std::max(step, std::abs(y[n] - y[n - 1]));
    EXPECT_LT(step, 1.1 * 0.3 * 2.0 * 2.0 * 3.14159265358979323846 * 1000.0 / fs);
    std::vector<double> tail(4800);
    for (double& v : tail) v = noise();
    y = tail;
    eqRun(eq, y);
    EXPECT_TRUE(y == tail);
}

// The per-sample form plays what the block form plays, channel by channel, and
// the state carries the new fields.
DSPARK_TEST(Equalizer_analog_per_sample_form_and_state)
{
    using Type = AnalogEq::BandType;
    AnalogEq block, single;
    for (AnalogEq* eq : { &block, &single })
    {
        eq->setAnalogResponse(true);
        eq->setBand(0, eqBand(Type::Peak, 900.0, 7.0, 3.0, 12.0, 3));
        eq->setBand(1, eqBand(Type::LowPass, 9000.0, 0.0, 1.1, 27.5));
        eq->setBand(2, eqBand(Type::HighPassWall, 60.0, 0.0, 0.7071));
        eq->prepare({ 48000.0, 128, 2 });
    }
    uint32_t seed = 5u;
    std::vector<double> l(4096), r(4096);
    for (size_t n = 0; n < l.size(); ++n)
    {
        seed = seed * 1664525u + 1013904223u; l[n] = (static_cast<double>(seed >> 8) / 8388608.0 - 1.0) * 0.5;
        seed = seed * 1664525u + 1013904223u; r[n] = (static_cast<double>(seed >> 8) / 8388608.0 - 1.0) * 0.5;
    }
    std::vector<double> bl = l, br = r;
    for (int off = 0; off < 4096; off += 128)
    {
        double* ch[2] = { bl.data() + off, br.data() + off };
        block.processBlock(AudioBufferView<double>(ch, 2, 128));
    }
    double worst = 0.0;
    for (size_t n = 0; n < l.size(); ++n)
    {
        worst = std::max(worst, std::abs(single.processSample(l[n], 0) - bl[n]));
        worst = std::max(worst, std::abs(single.processSample(r[n], 1) - br[n]));
    }
    EXPECT_LT(worst, 1e-9);

    const std::vector<uint8_t> blob = block.getState();
    AnalogEq other;
    EXPECT_FALSE(other.isAnalogResponse());
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.isAnalogResponse());
    EXPECT_EQ(other.getBandConfig(0).order, 3);
    EXPECT_NEAR(static_cast<double>(other.getBandConfig(1).slope), 27.5, 1e-6);
    EXPECT_TRUE(other.getBandConfig(2).type == Type::HighPassWall);
    EXPECT_TRUE(other.getState() == blob);

    // Values that are not a band: the order and the slope come back inside their ranges.
    AnalogEq::BandConfig wild = eqBand(Type::Peak, 1000.0, 3.0, 1.0, 500.0, 99);
    other.setBand(3, wild);
    EXPECT_EQ(other.getBandConfig(3).order, 16);
    EXPECT_EQ(static_cast<double>(other.getBandConfig(3).slope), 96.0);
    wild.slope = std::numeric_limits<double>::quiet_NaN();
    wild.order = -4;
    other.setBand(3, wild);
    EXPECT_EQ(other.getBandConfig(3).order, 1);
    EXPECT_EQ(static_cast<double>(other.getBandConfig(3).slope), 96.0);
}

// ============================================================================
// Equalizer: where in the stereo pair a band acts
// ============================================================================

namespace {

struct EqStereo { std::vector<double> l, r; };

EqStereo eqStereoNoise(int n, uint32_t seed)
{
    EqStereo x;
    x.l.resize(static_cast<size_t>(n));
    x.r.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        seed = seed * 1664525u + 1013904223u; x.l[static_cast<size_t>(i)] = (static_cast<double>(seed >> 8) / 8388608.0 - 1.0) * 0.4;
        seed = seed * 1664525u + 1013904223u; x.r[static_cast<size_t>(i)] = (static_cast<double>(seed >> 8) / 8388608.0 - 1.0) * 0.4;
    }
    return x;
}

void eqRunStereo(AnalogEq& eq, EqStereo& x, int block = 256)
{
    const int total = static_cast<int>(x.l.size());
    for (int off = 0; off < total; off += block)
    {
        double* ch[2] = { x.l.data() + off, x.r.data() + off };
        eq.processBlock(AudioBufferView<double>(ch, 2, std::min(block, total - off)));
    }
}

double eqWorstDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

AnalogEq::BandConfig eqPlaced(AnalogEq::BandConfig cfg, AnalogEq::Scope where)
{
    cfg.scope = where;
    return cfg;
}

} // namespace

// A band placed on the left, the right, the middle or the sides acts there and
// nowhere else, in both engines: the same samples as running the band on that
// signal alone.
DSPARK_TEST(Equalizer_band_scope_acts_where_it_is_set)
{
    using Type = AnalogEq::BandType;
    using Place = AnalogEq::Scope;
    const AnalogEq::BandConfig bell = eqBand(Type::Peak, 1000.0, 12.0, 1.5);
    const AnalogEq::BandConfig shelf = eqBand(Type::HighShelf, 5000.0, -9.0, 0.7071);
    for (const bool analog : { false, true })
    {
        const EqStereo in = eqStereoNoise(4096, 11u);
        // What one band does to one signal.
        const auto alone = [&](const AnalogEq::BandConfig& cfg, std::vector<double> x) {
            auto eqHeap = std::make_unique<AnalogEq>();
            AnalogEq& eq = *eqHeap;
            eq.setAnalogResponse(analog);
            eq.setBand(0, cfg);
            eq.prepare({ 48000.0, 256, 1 });
            eqRun(eq, x);
            return x;
        };
        const auto placed = [&](std::initializer_list<AnalogEq::BandConfig> bands) {
            auto eqHeap = std::make_unique<AnalogEq>();
            AnalogEq& eq = *eqHeap;
            eq.setAnalogResponse(analog);
            int index = 0;
            for (const AnalogEq::BandConfig& cfg : bands) eq.setBand(index++, cfg);
            eq.prepare({ 48000.0, 256, 2 });
            EqStereo x = in;
            eqRunStereo(eq, x);
            return x;
        };
        std::vector<double> mid(in.l.size()), side(in.l.size());
        for (size_t i = 0; i < mid.size(); ++i) { mid[i] = 0.5 * (in.l[i] + in.r[i]); side[i] = 0.5 * (in.l[i] - in.r[i]); }

        EqStereo out = placed({ eqPlaced(bell, Place::Left) });
        EXPECT_LT(eqWorstDiff(out.l, alone(bell, in.l)), 1e-12);
        EXPECT_TRUE(out.r == in.r);

        out = placed({ eqPlaced(bell, Place::Right) });
        EXPECT_TRUE(out.l == in.l);
        EXPECT_LT(eqWorstDiff(out.r, alone(bell, in.r)), 1e-12);

        out = placed({ eqPlaced(bell, Place::Mid) });
        {
            const std::vector<double> m = alone(bell, mid);
            double worst = 0.0;
            for (size_t i = 0; i < m.size(); ++i)
                worst = std::max({ worst, std::abs(out.l[i] - (m[i] + side[i])), std::abs(out.r[i] - (m[i] - side[i])) });
            EXPECT_LT(worst, 1e-12);
        }
        out = placed({ eqPlaced(bell, Place::Side) });
        {
            const std::vector<double> s = alone(bell, side);
            double worst = 0.0;
            for (size_t i = 0; i < s.size(); ++i)
                worst = std::max({ worst, std::abs(out.l[i] - (mid[i] + s[i])), std::abs(out.r[i] - (mid[i] - s[i])) });
            EXPECT_LT(worst, 1e-12);
        }
        // Left and stereo bands first, then the middle and the sides.
        out = placed({ eqPlaced(bell, Place::Mid), eqPlaced(shelf, Place::Left), eqPlaced(shelf, Place::Side) });
        {
            const std::vector<double> left = alone(shelf, in.l);
            std::vector<double> m(left.size()), s(left.size());
            for (size_t i = 0; i < m.size(); ++i) { m[i] = 0.5 * (left[i] + in.r[i]); s[i] = 0.5 * (left[i] - in.r[i]); }
            m = alone(bell, m);
            s = alone(shelf, s);
            double worst = 0.0;
            for (size_t i = 0; i < m.size(); ++i)
                worst = std::max({ worst, std::abs(out.l[i] - (m[i] + s[i])), std::abs(out.r[i] - (m[i] - s[i])) });
            EXPECT_LT(worst, 1e-12);
        }

        // One channel: the left and middle bands act on it, the right and side bands do nothing.
        AnalogEq mono;
        mono.setAnalogResponse(analog);
        mono.setBand(0, eqPlaced(bell, Place::Mid));
        mono.setBand(1, eqPlaced(shelf, Place::Right));
        mono.setBand(2, eqPlaced(shelf, Place::Side));
        mono.prepare({ 48000.0, 256, 1 });
        std::vector<double> x = in.l;
        eqRun(mono, x);
        EXPECT_LT(eqWorstDiff(x, alone(bell, in.l)), 1e-12);

        // The per-sample form runs the left and right bands on their channels (the middle and the
        // sides need whole frames).
        AnalogEq block, single;
        for (AnalogEq* eq : { &block, &single })
        {
            eq->setAnalogResponse(analog);
            eq->setBand(0, eqPlaced(bell, Place::Left));
            eq->setBand(1, eqPlaced(shelf, Place::Right));
            eq->setBand(2, eqBand(Type::Peak, 300.0, 4.0, 2.0));
            eq->prepare({ 48000.0, 256, 2 });
        }
        EqStereo viaBlock = in;
        eqRunStereo(block, viaBlock);
        double worst = 0.0;
        for (size_t i = 0; i < in.l.size(); ++i)
        {
            worst = std::max(worst, std::abs(single.processSample(in.l[i], 0) - viaBlock.l[i]));
            worst = std::max(worst, std::abs(single.processSample(in.r[i], 1) - viaBlock.r[i]));
        }
        EXPECT_LT(worst, analog ? 1e-9 : 1e-4);      // the cookbook engine smooths its parameters by block
    }
}

// In LinearPhase the placement is one matrix of kernels: the latency is the
// same, and each kernel has the response the two steps would give.
DSPARK_TEST(Equalizer_band_scope_in_linear_phase_is_one_matrix)
{
    using Type = AnalogEq::BandType;
    using Place = AnalogEq::Scope;
    const double fs = 48000.0;
    const auto make = [&](std::initializer_list<AnalogEq::BandConfig> bands, AnalogEq& eq) {
        eq.setFilterMode(AnalogEq::FilterMode::LinearPhase);
        int index = 0;
        for (const AnalogEq::BandConfig& cfg : bands) eq.setBand(index++, cfg);
        eq.prepare({ fs, 512, 2 });
    };
    AnalogEq plain, placed;
    make({ eqBand(Type::Peak, 1000.0, 12.0, 1.0) }, plain);
    make({ eqPlaced(eqBand(Type::Peak, 1000.0, 12.0, 1.0), Place::Mid),
           eqPlaced(eqBand(Type::HighShelf, 6000.0, -6.0, 0.7071), Place::Left) }, placed);
    EXPECT_EQ(placed.getLatency(), plain.getLatency());

    // An impulse on the left alone: what reaches each channel.
    const int length = 3 * 32768;
    EqStereo x;
    x.l.assign(static_cast<size_t>(length), 0.0);
    x.r.assign(static_cast<size_t>(length), 0.0);
    x.l[0] = 1.0;
    eqRunStereo(placed, x, 512);
    // The bell is 12 dB at 1 kHz in the middle and nothing at the sides: half the sum reaches the
    // left and half the difference crosses to the right. The shelf (-6 dB above 6 kHz) is on the left.
    const double m = std::pow(10.0, 12.0 / 20.0);
    EXPECT_NEAR(eqLevelDb(x.l, 1000.0, fs), 20.0 * std::log10(0.5 * (m + 1.0)), 0.05);
    EXPECT_NEAR(eqLevelDb(x.r, 1000.0, fs), 20.0 * std::log10(0.5 * (m - 1.0)), 0.05);
    EXPECT_NEAR(eqLevelDb(x.l, 60.0, fs), 0.0, 0.05);
    EXPECT_LT(eqLevelDb(x.r, 60.0, fs), -40.0);
    EXPECT_NEAR(eqLevelDb(x.l, 16000.0, fs), -6.0, 0.1);
    EXPECT_LT(eqLevelDb(x.r, 16000.0, fs), -40.0);

    // The same signal on both channels is all middle: the bell at full height on both. Opposite
    // signals are all sides: the bell does not touch them.
    for (const double sign : { 1.0, -1.0 })
    {
        auto eqHeap = std::make_unique<AnalogEq>();
        AnalogEq& eq = *eqHeap;
        make({ eqPlaced(eqBand(Type::Peak, 1000.0, 12.0, 1.0), Place::Mid) }, eq);
        EqStereo y;
        y.l.assign(static_cast<size_t>(length), 0.0);
        y.r.assign(static_cast<size_t>(length), 0.0);
        y.l[0] = 1.0;
        y.r[0] = sign;
        eqRunStereo(eq, y, 512);
        EXPECT_NEAR(eqLevelDb(y.l, 1000.0, fs), sign > 0.0 ? 12.0 : 0.0, 0.05);
        EXPECT_NEAR(eqLevelDb(y.r, 1000.0, fs), sign > 0.0 ? 12.0 : 0.0, 0.05);
    }
}

DSPARK_TEST(Equalizer_band_scope_state_and_curves)
{
    using Type = AnalogEq::BandType;
    using Place = AnalogEq::Scope;
    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setBand(0, eqPlaced(eqBand(Type::Peak, 1000.0, 6.0, 1.0), Place::Side));
    eq.setBand(1, eqPlaced(eqBand(Type::Peak, 1000.0, -4.0, 1.0), Place::Left));
    eq.setBand(2, eqBand(Type::Peak, 1000.0, 3.0, 1.0));
    eq.prepare({ 48000.0, 256, 2 });
    AnalogEq other;
    const std::vector<uint8_t> blob = eq.getState();
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.getBandConfig(0).scope == Place::Side);
    EXPECT_TRUE(other.getBandConfig(1).scope == Place::Left);
    EXPECT_TRUE(other.getBandConfig(2).scope == Place::Stereo);
    EXPECT_TRUE(other.getState() == blob);

    // A placement that is not one keeps the band's own.
    AnalogEq::BandConfig wild = eq.getBandConfig(0);
    wild.scope = static_cast<Place>(42);
    eq.setBand(0, wild);
    EXPECT_TRUE(eq.getBandConfig(0).scope == Place::Side);

    // The curve of each place: the stereo bands and the ones placed there.
    const double hz[1] = { 1000.0 };
    double level[1];
    eq.getMagnitudeForFrequencyArray(hz, level, 1, Place::Side);
    EXPECT_NEAR(20.0 * std::log10(level[0]), 9.0, 0.05);
    eq.getMagnitudeForFrequencyArray(hz, level, 1, Place::Left);
    EXPECT_NEAR(20.0 * std::log10(level[0]), -1.0, 0.05);
    eq.getMagnitudeForFrequencyArray(hz, level, 1, Place::Mid);
    EXPECT_NEAR(20.0 * std::log10(level[0]), 3.0, 0.05);
    eq.getMagnitudeForFrequencyArray(hz, level, 1);
    EXPECT_NEAR(20.0 * std::log10(level[0]), 5.0, 0.05);
}

// ============================================================================
// Equalizer: the natural-phase mode
// ============================================================================

namespace {

// The response of an equalizer at one frequency, with its latency taken out.
std::complex<double> eqPlayedAt(const std::vector<double>& h, double hz, double fs, int latency)
{
    const double w = 2.0 * 3.14159265358979323846 * hz / fs;
    std::complex<double> sum(0.0, 0.0);
    for (size_t n = 0; n < h.size(); ++n)
        sum += h[n] * std::polar(1.0, -w * (static_cast<double>(n) - latency));
    return sum;
}

// The response of an analog prototype at one frequency.
std::complex<double> eqAnalogAt(const AnalogCascade& proto, double hz)
{
    const double w = 2.0 * 3.14159265358979323846 * hz;
    std::complex<double> h(1.0, 0.0);
    for (int i = 0; i < proto.count; ++i)
    {
        const AnalogSection& s = proto.sections[static_cast<size_t>(i)];
        h *= std::complex<double>(s.b0 - s.b2 * w * w, s.b1 * w) / std::complex<double>(s.a0 - s.a2 * w * w, s.a1 * w);
    }
    return h;
}

struct EqDistance { double level = 0.0, phase = 0.0; };

// Worst distance in level (dB) and in phase (degrees), from 200 Hz to 20 kHz, between one band as an
// equalizer plays it in a mode and its analog prototype. Nothing is compared below -60 dB.
EqDistance eqDistance(const AnalogCascade& proto, const AnalogEq::BandConfig& cfg, AnalogEq::FilterMode mode,
                      double fs = 48000.0)
{
    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setAnalogResponse(true);
    eq.setFilterMode(mode);
    eq.setBand(0, cfg);
    eq.prepare({ fs, 256, 1 });
    const std::vector<double> h = eqImpulse(eq);
    const int latency = eq.getLatency();
    EqDistance worst;
    for (double hz = 200.0; hz <= 20000.0; hz *= 1.03)
    {
        const std::complex<double> want = eqAnalogAt(proto, hz), got = eqPlayedAt(h, hz, fs, latency);
        if (std::abs(want) < 1e-3) continue;
        worst.level = std::max(worst.level, std::abs(20.0 * std::log10(std::abs(got) / std::abs(want))));
        worst.phase = std::max(worst.phase, std::abs(std::arg(got / want)) * 180.0 / 3.14159265358979323846);
    }
    return worst;
}

} // namespace

// New mode: with no band at work the natural-phase mode is the signal, 32
// samples late, to the bit, in double and in float.
DSPARK_TEST(Equalizer_prototype_phase_at_rest_is_the_signal_late_to_the_bit)
{
    using Type = AnalogEq::BandType;
    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setFilterMode(AnalogEq::FilterMode::PrototypePhase);
    eq.setBand(0, eqBand(Type::Peak, 1000.0, 0.0, 1.0));
    eq.setBand(1, eqBand(Type::HighShelf, 8000.0, 0.0, 0.7071));
    eq.prepare({ 48000.0, 256, 2 });
    EXPECT_EQ(eq.getLatency(), 32);

    const EqStereo x = eqStereoNoise(8192, 11u);
    EqStereo y = x;
    eqRunStereo(eq, y, 200);
    bool late = true, quiet = true;
    for (size_t i = 0; i + 32 < x.l.size(); ++i)
        late = late && y.l[i + 32] == x.l[i] && y.r[i + 32] == x.r[i];
    for (size_t i = 0; i < 32; ++i) quiet = quiet && y.l[i] == 0.0 && y.r[i] == 0.0;
    EXPECT_TRUE(late);
    EXPECT_TRUE(quiet);

    eq.setFilterMode(AnalogEq::FilterMode::MinimumPhase);
    EXPECT_EQ(eq.getLatency(), 0);

    Equalizer<float> single;
    single.setFilterMode(Equalizer<float>::FilterMode::PrototypePhase);
    single.setBand(0, 3000.0f, 0.0f, 1.0f);
    single.prepare({ 44100.0, 128, 1 });
    std::vector<float> in(4096), out;
    uint32_t seed = 7u;
    for (auto& v : in) { seed = seed * 1664525u + 1013904223u; v = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f; }
    out = in;
    for (int off = 0; off < 4096; off += 100)
    {
        float* ch[1] = { out.data() + off };
        single.processBlock(AudioBufferView<float>(ch, 1, std::min(100, 4096 - off)));
    }
    bool same = true;
    for (size_t i = 0; i + 32 < in.size(); ++i) same = same && out[i + 32] == in[i];
    EXPECT_TRUE(same);
    EXPECT_EQ(single.getLatency(), 32);
}

// The mode keeps the guarantees of the others: the output does not depend on
// how the host cuts the stream, and a change of a band is faded in (a 15 kHz
// tone under a shelf that jumps 6 dB leaves nothing 60 dB under it at 3 kHz).
DSPARK_TEST(Equalizer_prototype_phase_does_not_depend_on_the_blocks_and_fades_a_change)
{
    using Type = AnalogEq::BandType;
    auto make = [](AnalogEq& eq) {
        eq.setFilterMode(AnalogEq::FilterMode::PrototypePhase);
        eq.setBand(0, eqBand(Type::HighShelf, 10000.0, 9.0, 0.7071));
        eq.setBand(1, eqPlaced(eqBand(Type::Peak, 14000.0, -6.0, 2.0), AnalogEq::Scope::Mid));
        eq.setBand(2, eqBand(Type::LowPass, 17000.0, 0.0, 0.7071, 18.0));
        eq.prepare({ 48000.0, 512, 2 });
    };
    const EqStereo x = eqStereoNoise(24000, 5u);
    AnalogEq first;
    make(first);
    EqStereo reference = x;
    eqRunStereo(first, reference, 512);
    for (int block : { 37, 64, 480 })
    {
        auto eqHeap = std::make_unique<AnalogEq>();
        AnalogEq& eq = *eqHeap;
        make(eq);
        EqStereo y = x;
        eqRunStereo(eq, y, block);
        EXPECT_LT(eqWorstDiff(y.l, reference.l), 1e-12);
        EXPECT_LT(eqWorstDiff(y.r, reference.r), 1e-12);
    }

    auto eqHeap = std::make_unique<AnalogEq>();
    AnalogEq& eq = *eqHeap;
    eq.setFilterMode(AnalogEq::FilterMode::PrototypePhase);
    eq.setBand(0, eqBand(Type::HighShelf, 10000.0, 9.0, 0.7071));
    eq.prepare({ 48000.0, 256, 1 });
    const int n = 48000;
    std::vector<double> tone(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) tone[static_cast<size_t>(i)] = 0.25 * std::sin(2.0 * 3.14159265358979323846 * 15000.0 * i / 48000.0);
    std::vector<double> head(tone.begin(), tone.begin() + 24064), tail(tone.begin() + 24064, tone.end());
    eqRun(eq, head);
    eq.setBand(0, eqBand(Type::HighShelf, 10000.0, 3.0, 0.7071));
    eqRun(eq, tail);
    std::vector<double> y = head;
    y.insert(y.end(), tail.begin(), tail.end());
    // What the change leaves at 3 kHz, over 8192 samples around it (Hann), against the tone.
    double re = 0.0, im = 0.0, carrierRe = 0.0, carrierIm = 0.0;
    for (int i = 0; i < 8192; ++i)
    {
        const double window = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / 8192.0);
        const double v = y[static_cast<size_t>(24064 - 4096 + i)] * window;
        const double a = 2.0 * 3.14159265358979323846 * 3000.0 * i / 48000.0, b = 2.0 * 3.14159265358979323846 * 15000.0 * i / 48000.0;
        re += v * std::cos(a); im += v * std::sin(a);
        carrierRe += v * std::cos(b); carrierIm += v * std::sin(b);
    }
    const double splash = 10.0 * std::log10((re * re + im * im) / (carrierRe * carrierRe + carrierIm * carrierIm));
    EXPECT_LT(splash, -60.0);
}

// With placed bands the correction is a matrix between the two channels: a
// band on the left leaves the right channel alone (the signal, late, to the
// bit) and gives the left what a one-channel equalizer gives; a band on the
// middle gives the middle that, and leaves the sides alone.
DSPARK_TEST(Equalizer_prototype_phase_with_scoped_bands_is_one_matrix)
{
    using Type = AnalogEq::BandType;
    using Place = AnalogEq::Scope;
    const AnalogEq::BandConfig band = eqBand(Type::HighShelf, 10000.0, 9.0, 0.7071);
    const EqStereo x = eqStereoNoise(8192, 23u);
    const size_t n = x.l.size();

    AnalogEq one;
    one.setFilterMode(AnalogEq::FilterMode::PrototypePhase);
    one.setBand(0, band);
    one.prepare({ 48000.0, 256, 1 });
    std::vector<double> alone = x.l;
    eqRun(one, alone);

    AnalogEq left;
    left.setFilterMode(AnalogEq::FilterMode::PrototypePhase);
    left.setBand(0, eqPlaced(band, Place::Left));
    left.prepare({ 48000.0, 256, 2 });
    EqStereo y = x;
    eqRunStereo(left, y);
    EXPECT_LT(eqWorstDiff(y.l, alone), 1e-12);
    bool late = true;
    for (size_t i = 0; i + 32 < n; ++i) late = late && y.r[i + 32] == x.r[i];
    EXPECT_TRUE(late);

    std::vector<double> middle(n), sides(n);
    for (size_t i = 0; i < n; ++i) { middle[i] = 0.5 * (x.l[i] + x.r[i]); sides[i] = 0.5 * (x.l[i] - x.r[i]); }
    one.reset();
    std::vector<double> middleAlone = middle;
    eqRun(one, middleAlone);

    AnalogEq mid;
    mid.setFilterMode(AnalogEq::FilterMode::PrototypePhase);
    mid.setBand(0, eqPlaced(band, Place::Mid));
    mid.prepare({ 48000.0, 256, 2 });
    EqStereo z = x;
    eqRunStereo(mid, z);
    double worstMiddle = 0.0, worstSides = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        worstMiddle = std::max(worstMiddle, std::abs(0.5 * (z.l[i] + z.r[i]) - middleAlone[i]));
        if (i >= 32) worstSides = std::max(worstSides, std::abs(0.5 * (z.l[i] - z.r[i]) - sides[i - 32]));
    }
    EXPECT_LT(worstMiddle, 1e-9);
    EXPECT_LT(worstSides, 1e-9);
}

// New mode: a minimum-phase filter that follows the magnitude of an analog
// one does not have its phase near Nyquist. The natural-phase mode brings the
// bands the rest of the way: the phase of the analog curve up to 20 kHz, and
// what was left of the level.
DSPARK_TEST(Equalizer_prototype_phase_follows_the_phase_of_the_analog_curve)
{
    using Type = AnalogEq::BandType;
    using Mode = AnalogEq::FilterMode;
    struct Case { AnalogEq::BandConfig cfg; AnalogCascade proto; };
    const Case cases[] = {
        { eqBand(Type::HighShelf, 12000.0, 12.0, 0.7071067811865476), eqOne(AnalogSection::highShelf(12000.0, 12.0)) },
        { eqBand(Type::Peak, 12000.0, 12.0, 2.0), eqOne(AnalogSection::peak(12000.0, 2.0, 12.0)) },
        { eqBand(Type::Peak, 15000.0, -18.0, 1.0), eqOne(AnalogSection::peak(15000.0, 1.0, -18.0)) },
        { eqBand(Type::LowPass, 10000.0, 0.0, 0.7071067811865476, 24.0), FilterDesign::highCut(10000.0, 24.0) },
    };
    for (const Case& c : cases)
    {
        const EqDistance minimum = eqDistance(c.proto, c.cfg, Mode::MinimumPhase);
        const EqDistance natural = eqDistance(c.proto, c.cfg, Mode::PrototypePhase);
        EXPECT_GT(minimum.phase, 15.0);                 // 17 to 178 degrees: what the mode is for
        EXPECT_LT(natural.phase, 1.5);                  // 0.08 to 1.05
        EXPECT_LT(natural.phase, 0.1 * minimum.phase);
        EXPECT_LT(natural.level, 0.25);
    }

    // At 44.1 kHz there is less room between 20 kHz and Nyquist for the correction to fade out, and a
    // steep cut near the top keeps a few degrees.
    {
        const EqDistance bell = eqDistance(eqOne(AnalogSection::peak(15000.0, 1.0, -18.0)),
                                           eqBand(Type::Peak, 15000.0, -18.0, 1.0), Mode::PrototypePhase, 44100.0);
        EXPECT_LT(bell.phase, 2.0);                     // minimum phase: 38
        EXPECT_LT(bell.level, 0.1);
        const EqDistance cut = eqDistance(FilterDesign::highCut(10000.0, 24.0),
                                          eqBand(Type::LowPass, 10000.0, 0.0, 0.7071067811865476, 24.0),
                                          Mode::PrototypePhase, 44100.0);
        EXPECT_LT(cut.phase, 8.0);                      // minimum phase: 179
        EXPECT_LT(cut.level, 1.0);
    }

    // A band in the middle of the spectrum needs no correction and takes no harm from one.
    const AnalogEq::BandConfig low = eqBand(Type::Peak, 1000.0, 12.0, 1.0);
    const EqDistance natural = eqDistance(eqOne(AnalogSection::peak(1000.0, 1.0, 12.0)), low, Mode::PrototypePhase);
    EXPECT_LT(natural.phase, 0.3);
    EXPECT_LT(natural.level, 0.05);
}

// ============================================================================
// Saturation: the curve family, minimum-phase oversampling, and the stages
// around the curve (asymmetry, programme noise, dynamics, tuned feedback,
// drive compensation)
// ============================================================================

namespace {

using Sat8 = Saturation<float>;
using Sat8Vec = std::vector<float>;
constexpr double kSat8Pi = 3.14159265358979323846;

Sat8Vec sat8Sine(double hz, double amp, int n)
{
    Sat8Vec v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = static_cast<float>(amp * std::sin(2.0 * kSat8Pi * hz * i / 48000.0));
    return v;
}

// Pink noise at an RMS level in dBFS.
Sat8Vec sat8Pink(int n, double rmsDb, uint32_t seed)
{
    Sat8Vec v(static_cast<size_t>(n));
    double b0 = 0.0, b1 = 0.0, b2 = 0.0, sum = 0.0;
    for (auto& s : v)
    {
        seed = seed * 1664525u + 1013904223u;
        const double white = static_cast<double>(static_cast<int32_t>(seed)) / 2147483648.0;
        b0 = 0.99765 * b0 + white * 0.0990460;
        b1 = 0.96300 * b1 + white * 0.2965164;
        b2 = 0.57000 * b2 + white * 1.0526913;
        const double pink = b0 + b1 + b2 + white * 0.1848;
        s = static_cast<float>(pink);
        sum += pink * pink;
    }
    const double scale = std::pow(10.0, rmsDb / 20.0) / std::sqrt(sum / n);
    for (auto& s : v) s = static_cast<float>(s * scale);
    return v;
}

// A mono render in blocks, the processor set up before prepare().
template <class Setup>
Sat8Vec sat8Run(const Sat8Vec& in, Setup&& setup, int block = 512)
{
    auto s = std::make_unique<Sat8>();
    setup(*s);
    s->prepare(AudioSpec{ 48000.0, block, 1 });
    Sat8Vec x = in;
    for (size_t off = 0; off < x.size(); off += static_cast<size_t>(block))
    {
        float* p = x.data() + off;
        const int m = static_cast<int>(std::min(static_cast<size_t>(block), x.size() - off));
        s->processBlock(AudioBufferView<float>(&p, 1, m));
    }
    return x;
}

// Level in dB of the line at hz over the last 4800 samples (hz a multiple of 10).
double sat8Line(const Sat8Vec& v, double hz)
{
    const size_t n = 4800, from = v.size() - n;
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double ph = 2.0 * kSat8Pi * hz * static_cast<double>(i) / 48000.0;
        re += static_cast<double>(v[from + i]) * std::cos(ph);
        im += static_cast<double>(v[from + i]) * std::sin(ph);
    }
    return 20.0 * std::log10(std::max(2.0 * std::sqrt(re * re + im * im) / static_cast<double>(n), 1e-15));
}

double sat8Rms(const Sat8Vec& v, size_t from, size_t to)
{
    double sum = 0.0;
    for (size_t i = from; i < to; ++i) sum += static_cast<double>(v[i]) * static_cast<double>(v[i]);
    return std::sqrt(sum / static_cast<double>(to - from));
}

// Harmonic k of a curve on a sine of amplitude amp, in dB, by its Fourier sum.
template <class Curve>
double sat8Harmonic(Curve&& curve, double amp, int k)
{
    const int n = 4096;
    double re = 0.0, im = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double ph = 2.0 * kSat8Pi * i / n;
        const double y = curve(amp * std::sin(ph));
        re += y * std::cos(k * ph);
        im += y * std::sin(k * ph);
    }
    return 20.0 * std::log10(std::max(2.0 * std::sqrt(re * re + im * im) / n, 1e-15));
}

} // namespace

// New: Algorithm::Curve. At its defaults it is tanh; hardness keeps it a
// straight line up to a share of its ceiling, curvature changes how fast it
// turns, and asymmetry gives the negative half-cycle a ceiling of its own.
DSPARK_TEST(Saturation_curve_family_shapes_the_knee_the_turn_and_the_halves)
{
    const auto curve = [](float hardness, float curvature, float asymmetry, float drive) {
        return [=](Sat8& s) {
            s.setAlgorithm(Sat8::Algorithm::Curve);
            s.setOversampling(4);
            s.setDrive(drive);
            s.setCurveShape(hardness, curvature, asymmetry);
        };
    };
    const Sat8Vec tone = sat8Sine(1000.0, 0.5, 24000);

    // tanh, to its Fourier series.
    const Sat8Vec plain = sat8Run(tone, curve(0.0f, 1.0f, 0.0f, 12.0f));
    const auto tanhOf = [](double x) { return std::tanh(x); };
    const double g = std::pow(10.0, 12.0 / 20.0);
    EXPECT_NEAR(sat8Line(plain, 1000.0), sat8Harmonic(tanhOf, 0.5 * g, 1), 0.05);
    EXPECT_NEAR(sat8Line(plain, 3000.0), sat8Harmonic(tanhOf, 0.5 * g, 3), 0.1);
    EXPECT_NEAR(sat8Line(plain, 5000.0), sat8Harmonic(tanhOf, 0.5 * g, 5), 0.3);
    EXPECT_LT(sat8Line(plain, 2000.0) - sat8Line(plain, 1000.0), -100.0);

    // Hardness: a peak under it passes straight; the whole tanh bends it.
    const Sat8Vec soft = sat8Run(tone, curve(0.0f, 1.0f, 0.0f, 0.0f));
    const Sat8Vec straight = sat8Run(tone, curve(0.6f, 1.0f, 0.0f, 0.0f));
    EXPECT_GT(sat8Line(soft, 3000.0) - sat8Line(soft, 1000.0), -40.0);
    EXPECT_LT(sat8Line(straight, 3000.0) - sat8Line(straight, 1000.0), -100.0);

    // Curvature: over 1 it stays near the line longer.
    const Sat8Vec late = sat8Run(tone, curve(0.0f, 4.0f, 0.0f, 0.0f));
    EXPECT_LT(sat8Line(late, 3000.0) - sat8Line(late, 1000.0), sat8Line(soft, 3000.0) - sat8Line(soft, 1000.0) - 15.0);
    const Sat8Vec early = sat8Run(tone, curve(0.0f, 0.5f, 0.0f, 0.0f));
    EXPECT_GT(sat8Line(early, 3000.0) - sat8Line(early, 1000.0), sat8Line(soft, 3000.0) - sat8Line(soft, 1000.0) + 3.0);

    // Asymmetry: the negative half meets a ceiling half as high, and even harmonics appear.
    const Sat8Vec uneven = sat8Run(tone, [&](Sat8& s) { curve(0.0f, 1.0f, 1.0f, 12.0f)(s); s.setDcBlocking(false); });
    float top = 0.0f, bottom = 0.0f;
    for (size_t i = 12000; i < uneven.size(); ++i)
    {
        top = std::max(top, uneven[i]);
        bottom = std::min(bottom, uneven[i]);
    }
    // tanh(2) over tanh(4) / 2.
    EXPECT_NEAR(static_cast<double>(top / -bottom), std::tanh(2.0) / (0.5 * std::tanh(4.0)), 0.05);
    EXPECT_GT(sat8Line(uneven, 2000.0) - sat8Line(uneven, 1000.0), -30.0);
    // The other way round, the positive half is the one held down: neither goes past 1.
    const Sat8Vec mirrored = sat8Run(tone, [&](Sat8& s) { curve(0.0f, 1.0f, -1.0f, 12.0f)(s); s.setDcBlocking(false); });
    float topM = 0.0f, bottomM = 0.0f;
    for (size_t i = 12000; i < mirrored.size(); ++i)
    {
        topM = std::max(topM, mirrored[i]);
        bottomM = std::min(bottomM, mirrored[i]);
    }
    EXPECT_NEAR(static_cast<double>(-bottomM / topM), std::tanh(2.0) / (0.5 * std::tanh(4.0)), 0.05);
    EXPECT_LT(static_cast<double>(-bottomM), 1.0);

    // Wild values; a block split does not change it.
    Sat8 s;
    s.setCurveShape(0.3f, 2.0f, -0.5f);
    s.setCurveShape(std::numeric_limits<float>::quiet_NaN(), 99.0f, -7.0f);
    EXPECT_EQ(s.getCurveHardness(), 0.3f);
    EXPECT_EQ(s.getCurveCurvature(), 4.0f);
    EXPECT_EQ(s.getCurveAsymmetry(), -1.0f);
    const Sat8Vec small = sat8Run(tone, curve(0.2f, 1.5f, 0.3f, 6.0f), 64);
    const Sat8Vec large = sat8Run(tone, curve(0.2f, 1.5f, 0.3f, 6.0f), 2048);
    double farthest = 0.0;
    for (size_t i = 0; i < small.size(); ++i)
        farthest = std::max(farthest, static_cast<double>(std::abs(small[i] - large[i])));
    EXPECT_LT(farthest, 1e-6);
}

// New: minimum-phase oversampling. A few samples of latency instead of the
// factor's, and the dry path goes through the same filters: a half mix of a
// signal the curve leaves alone is the wet signal itself at every frequency,
// and the Delta output of it is next to nothing.
DSPARK_TEST(Saturation_minimum_phase_oversampling_keeps_the_mix_in_phase)
{
    const auto base = [](Sat8::OversamplingMode mode, float mix) {
        return [=](Sat8& s) {
            s.setAlgorithm(Sat8::Algorithm::SoftClip);
            s.setOversampling(2);
            s.setOversamplingMode(mode);
            s.setMix(mix);
            s.setPreFilterHpFrequency(10.0f);
        };
    };
    Sat8 linear, minimum;
    base(Sat8::OversamplingMode::LinearPhase, 1.0f)(linear);
    base(Sat8::OversamplingMode::MinimumPhase, 1.0f)(minimum);
    linear.prepare(AudioSpec{ 48000.0, 512, 1 });
    minimum.prepare(AudioSpec{ 48000.0, 512, 1 });
    EXPECT_GT(linear.getLatency(), 20);
    EXPECT_GT(minimum.getLatency(), 0);
    EXPECT_LT(minimum.getLatency(), 8);
    EXPECT_TRUE(minimum.getOversamplingMode() == Sat8::OversamplingMode::MinimumPhase);

    for (double hz : { 500.0, 2000.0, 8000.0, 15000.0, 19000.0 })
    {
        const Sat8Vec in = sat8Sine(hz, 0.01, 24000);
        const double wet = sat8Line(sat8Run(in, base(Sat8::OversamplingMode::MinimumPhase, 1.0f)), hz);
        const double half = sat8Line(sat8Run(in, base(Sat8::OversamplingMode::MinimumPhase, 0.5f)), hz);
        EXPECT_NEAR(half, wet, 0.03);
        if (hz < 16000.0)
        {
            const double wetLinear = sat8Line(sat8Run(in, base(Sat8::OversamplingMode::LinearPhase, 1.0f)), hz);
            const double halfLinear = sat8Line(sat8Run(in, base(Sat8::OversamplingMode::LinearPhase, 0.5f)), hz);
            EXPECT_NEAR(halfLinear, wetLinear, 0.05);
        }
    }

    // (What is left is the phase of the input and DC filters of the wet path.)
    const Sat8Vec in = sat8Sine(8000.0, 0.01, 24000);
    const Sat8Vec delta = sat8Run(in, [&](Sat8& s) {
        base(Sat8::OversamplingMode::MinimumPhase, 1.0f)(s);
        s.setOutputMode(Sat8::OutputMode::Delta);
    });
    EXPECT_LT(sat8Rms(delta, 12000, 24000), sat8Rms(in, 12000, 24000) * 0.01);

    // With second-order antialiasing too.
    const Sat8Vec deltaSecond = sat8Run(in, [&](Sat8& s) {
        base(Sat8::OversamplingMode::MinimumPhase, 1.0f)(s);
        s.setAntialiasing(Sat8::Antialiasing::SecondOrder);
        s.setOutputMode(Sat8::OutputMode::Delta);
    });
    // (Here the kernel's own loss at 8 kHz is in it too.)
    EXPECT_LT(sat8Rms(deltaSecond, 12000, 24000), sat8Rms(in, 12000, 24000) * 0.2);
}

// New: setAsymmetry(). A share of the square of the signal before any curve:
// second harmonic in proportion, none at zero.
DSPARK_TEST(Saturation_asymmetry_adds_even_harmonics_before_any_curve)
{
    const Sat8Vec tone = sat8Sine(1000.0, 0.5, 24000);
    const auto second = [&](float amount, Sat8::Algorithm algo) {
        const Sat8Vec out = sat8Run(tone, [=](Sat8& s) {
            s.setAlgorithm(algo);
            s.setAsymmetry(amount);
        });
        return sat8Line(out, 2000.0) - sat8Line(out, 1000.0);
    };
    EXPECT_LT(second(0.0f, Sat8::Algorithm::SoftClip), -100.0);
    const double full = second(1.0f, Sat8::Algorithm::SoftClip);
    EXPECT_NEAR(full, -24.1, 2.5);
    EXPECT_NEAR(second(0.5f, Sat8::Algorithm::SoftClip), full - 6.0, 1.0);
    EXPECT_NEAR(second(-1.0f, Sat8::Algorithm::SoftClip), full, 0.5);
    EXPECT_GT(second(1.0f, Sat8::Algorithm::Curve), -30.0);

    Sat8 s;
    s.setAsymmetry(0.4f);
    s.setAsymmetry(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(s.getAsymmetry(), 0.4f);
    s.setAsymmetry(5.0f);
    EXPECT_EQ(s.getAsymmetry(), 1.0f);
}

// New: setProgramNoise(). Noise counted from the peak of the programme: there
// while the tone is, at its level under the peak, and gone after it.
DSPARK_TEST(Saturation_program_noise_follows_the_programme_and_leaves_silence_silent)
{
    Sat8Vec in = sat8Sine(1000.0, 0.25, 48000 * 9 / 2);
    for (size_t i = 48000; i < in.size(); ++i) in[i] = 0.0f;
    const Sat8Vec quiet = sat8Run(in, [](Sat8& s) { s.setAlgorithm(Sat8::Algorithm::SoftClip); });
    const Sat8Vec noisy = sat8Run(in, [](Sat8& s) { s.setAlgorithm(Sat8::Algorithm::SoftClip); s.setProgramNoise(-40.0f); });
    Sat8Vec difference(in.size());
    for (size_t i = 0; i < in.size(); ++i) difference[i] = noisy[i] - quiet[i];
    // White noise in [-1, 1): an RMS of 1/sqrt(3) of its peak.
    EXPECT_NEAR(sat8Rms(difference, 9600, 43200), 0.25 * 0.01 / std::sqrt(3.0), 0.0005);
    double tail = 0.0;
    for (size_t i = in.size() - 14400; i < in.size(); ++i) tail = std::max(tail, static_cast<double>(std::abs(difference[i])));
    EXPECT_LT(tail, 1e-7);

    Sat8 s;
    s.setProgramNoise(-200.0f);
    EXPECT_EQ(s.getProgramNoise(), -120.0f);
    s.setProgramNoise(-5.0f);
    EXPECT_EQ(s.getProgramNoise(), -20.0f);
    s.setProgramNoise(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(s.getProgramNoise(), -20.0f);
}

// New: setLevelTracking(). At +1 a quiet and a loud tone saturate alike and come
// out at the level they went in at; at -1 the loud one saturates far more;
// with a focus the follower hears only the low end.
DSPARK_TEST(Saturation_dynamics_hold_the_amount_of_saturation_whatever_the_level)
{
    const auto measure = [](double amp, float dynamics, float focus, double hz, double& third) {
        const Sat8Vec in = sat8Sine(hz, amp, 48000);
        const Sat8Vec out = sat8Run(in, [=](Sat8& s) {
            s.setAlgorithm(Sat8::Algorithm::SoftClip);
            s.setDrive(12.0f);
            s.setLevelTracking(dynamics);
            s.setTrackingCorner(focus);
        });
        third = sat8Line(out, 3.0 * hz) - sat8Line(out, hz);
        return sat8Line(out, hz) - sat8Line(in, hz);
    };
    double quiet = 0.0, loud = 0.0;
    (void)measure(0.0316, 0.0f, 0.0f, 1000.0, quiet);
    (void)measure(0.5, 0.0f, 0.0f, 1000.0, loud);
    EXPECT_GT(loud - quiet, 20.0);

    double quietHeld = 0.0, loudHeld = 0.0;
    const double gainQuiet = measure(0.0316, 1.0f, 0.0f, 1000.0, quietHeld);
    const double gainLoud = measure(0.5, 1.0f, 0.0f, 1000.0, loudHeld);
    EXPECT_LT(std::abs(loudHeld - quietHeld), 1.5);
    EXPECT_LT(std::abs(gainLoud - gainQuiet), 0.3);

    double quietSpread = 0.0, loudSpread = 0.0;
    (void)measure(0.0316, -1.0f, 0.0f, 1000.0, quietSpread);
    (void)measure(0.5, -1.0f, 0.0f, 1000.0, loudSpread);
    EXPECT_GT(loudSpread - quietSpread, loud - quiet + 3.0);

    double heard = 0.0, deaf = 0.0;
    (void)measure(0.5, -1.0f, 0.0f, 3000.0, heard);
    (void)measure(0.5, -1.0f, 200.0f, 3000.0, deaf);
    EXPECT_LT(deaf, heard - 6.0);

    Sat8 s;
    s.setLevelTracking(-3.0f);
    EXPECT_EQ(s.getLevelTracking(), -1.0f);
    s.setTrackingCorner(5.0f);
    EXPECT_EQ(s.getTrackingCorner(), 20.0f);
    s.setTrackingCorner(-1.0f);
    EXPECT_EQ(s.getTrackingCorner(), 0.0f);
}

// New: setCombLoop(). The output of the curve returns to its input, inverted,
// one period of the frequency later: dips at the frequency and its
// multiples, peaks half way, the other way round for a negative amount; any
// block size gives the same output, and it stays bounded at full drive.
DSPARK_TEST(Saturation_feedback_tunes_a_comb_inside_the_curve)
{
    const auto gain = [](float amount, double hz, int block) {
        const Sat8Vec in = sat8Sine(hz, 0.01, 48000);
        const Sat8Vec out = sat8Run(in, [=](Sat8& s) {
            s.setAlgorithm(Sat8::Algorithm::SoftClip);
            s.setCombLoop(amount, 200.0f);
        }, block);
        return sat8Line(out, hz) - sat8Line(in, hz);
    };
    const double dip = 20.0 * std::log10(1.0 / 1.8), peak = 20.0 * std::log10(1.0 / 0.2);
    EXPECT_NEAR(gain(0.8f, 200.0, 512), dip, 0.3);
    EXPECT_NEAR(gain(0.8f, 300.0, 512), peak, 0.8);
    EXPECT_NEAR(gain(0.8f, 400.0, 512), dip, 0.3);
    EXPECT_NEAR(gain(-0.8f, 200.0, 512), peak, 0.8);
    EXPECT_NEAR(gain(-0.8f, 300.0, 512), dip, 0.3);

    Sat8 s;
    s.setCombLoop(0.5f, 200.0f);
    s.prepare(AudioSpec{ 48000.0, 512, 1 });
    EXPECT_NEAR(static_cast<double>(s.getCombLoopFrequency()), 200.0, 0.01);
    s.setCombLoop(0.5f, 20000.0f);
    EXPECT_NEAR(static_cast<double>(s.getCombLoopFrequency()), 48000.0 / (16.0 + s.getLatency()), 0.01);
    s.setCombLoop(3.0f, std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(static_cast<double>(s.getCombLoop()), 0.95, 1e-6);

    const Sat8Vec in = sat8Sine(330.0, 0.5, 48000);
    const auto render = [&](int block) {
        return sat8Run(in, [](Sat8& x) {
            x.setAlgorithm(Sat8::Algorithm::SoftClip);
            x.setDrive(24.0f);
            x.setCombLoop(0.95f, 150.0f);
        }, block);
    };
    const Sat8Vec small = render(64), large = render(2048);
    double farthest = 0.0, top = 0.0;
    for (size_t i = 0; i < small.size(); ++i)
    {
        farthest = std::max(farthest, static_cast<double>(std::abs(small[i] - large[i])));
        top = std::max(top, static_cast<double>(std::abs(small[i])));
    }
    EXPECT_EQ(farthest, 0.0);
    EXPECT_TRUE(std::isfinite(top) && top < 2.0);
}

// New: setDriveCompensation(). Of the level the drive adds on programme, a
// quarter is left; the compensation follows a drive moved while it runs and
// is part of the state.
DSPARK_TEST(Saturation_drive_compensation_leaves_a_quarter_of_the_level_the_drive_adds)
{
    const Sat8Vec pink = sat8Pink(49152, -18.0, 77u);
    const auto level = [&](float drive, bool compensated) {
        const Sat8Vec out = sat8Run(pink, [=](Sat8& s) {
            s.setAlgorithm(Sat8::Algorithm::SoftClip);
            s.setDrive(drive);
            s.setDriveCompensation(compensated);
        });
        return 20.0 * std::log10(sat8Rms(out, 9600, 48000) / sat8Rms(pink, 9600, 48000));
    };
    const double plain = level(24.0f, false) - level(0.0f, false);
    const double kept = level(24.0f, true) - level(0.0f, true);
    EXPECT_GT(plain, 6.0);
    EXPECT_NEAR(kept, 0.25 * plain, 0.6);
    EXPECT_LT(std::abs(level(0.0f, true) - level(0.0f, false)), 0.5);

    // A drive moved while it runs.
    auto s = std::make_unique<Sat8>();
    s->setDriveCompensation(true);
    s->prepare(AudioSpec{ 48000.0, 256, 1 });
    Sat8Vec x = pink;
    for (size_t off = 0; off < x.size(); off += 256)
    {
        if (off == 12288) s->setDrive(24.0f);
        float* p = x.data() + off;
        s->processBlock(AudioBufferView<float>(&p, 1, 256));
    }
    EXPECT_LT(static_cast<double>(s->getDriveCompensationDb()), -5.0);

    const auto blob = s->getState();
    Sat8 other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.isDriveCompensation());
    s->setDriveCompensation(false);
    float* p = x.data();
    s->processBlock(AudioBufferView<float>(&p, 1, 256));
    EXPECT_EQ(static_cast<double>(s->getDriveCompensationDb()), 0.0);
}

// ============================================================================
// MultibandSaturation
// ============================================================================

namespace {

using MbSat = MultibandSaturation<float>;

// A stereo render in blocks of 512; the processor set up before prepare().
template <class Setup>
std::vector<Sat8Vec> mbRun(const Sat8Vec& left, const Sat8Vec& right, Setup&& setup, int block = 512,
                           const std::function<void(MbSat&, size_t)>& during = nullptr)
{
    auto m = std::make_unique<MbSat>();
    setup(*m);
    m->prepare(AudioSpec{ 48000.0, block, 2 });
    std::vector<Sat8Vec> x { left, right };
    for (size_t off = 0; off < left.size(); off += static_cast<size_t>(block))
    {
        if (during) during(*m, off);
        float* p[2] = { x[0].data() + off, x[1].data() + off };
        const int n = static_cast<int>(std::min(static_cast<size_t>(block), left.size() - off));
        m->processBlock(AudioBufferView<float>(p, 2, n));
    }
    return x;
}

Sat8Vec mbTwoTones(int n)
{
    Sat8Vec v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        v[static_cast<size_t>(i)] = static_cast<float>(0.25 * std::sin(2.0 * kSat8Pi * 100.0 * i / 48000.0)
                                                     + 0.25 * std::sin(2.0 * kSat8Pi * 3000.0 * i / 48000.0));
    return v;
}

} // namespace

// New: MultibandSaturation. At no drive the bands sum back to the input's
// level at every frequency, with the same latency the processor reports, and
// they stay in time when one band runs another oversampling factor (it is
// delayed to the slowest one).
DSPARK_TEST(MultibandSaturation_sums_the_bands_back_flat_and_in_time)
{
    const auto three = [](MbSat& m) {
        m.setNumBands(3);
        m.setCrossoverFrequency(0, 300.0f);
        m.setCrossoverFrequency(1, 3000.0f);
    };
    const auto uneven = [&](MbSat& m) {
        three(m);
        m.getBandSaturation(1).setOversampling(4);
        m.getBandSaturation(2).setOversampling(8);
    };
    for (double hz : { 100.0, 300.0, 1000.0, 3000.0, 10000.0 })
    {
        const Sat8Vec in = sat8Sine(hz, 0.01, 24000);
        const auto even = mbRun(in, in, three);
        const auto mixed = mbRun(in, in, uneven);
        EXPECT_NEAR(sat8Line(even[0], hz) - sat8Line(in, hz), 0.0, 0.1);
        EXPECT_NEAR(sat8Line(mixed[0], hz) - sat8Line(in, hz), 0.0, 0.15);
        EXPECT_NEAR(sat8Line(mixed[1], hz), sat8Line(mixed[0], hz), 1e-6);
    }

    MbSat m;
    uneven(m);
    m.prepare(AudioSpec{ 48000.0, 512, 2 });
    int slowest = 0;
    for (int b = 0; b < 3; ++b) slowest = std::max(slowest, m.getBandSaturation(b).getLatency());
    EXPECT_EQ(m.getLatency(), slowest);
    EXPECT_GT(m.getBandSaturation(1).getLatency(), m.getBandSaturation(0).getLatency());
}

// New: each band saturates on its own: drive the high band and the low tone
// comes out clean while the high one gains its harmonics.
DSPARK_TEST(MultibandSaturation_drives_one_band_and_leaves_the_others_clean)
{
    const Sat8Vec in = mbTwoTones(24000);
    const auto split = [](float highDrive) {
        return [=](MbSat& m) {
            m.setNumBands(2);
            m.setCrossoverFrequency(0, 1000.0f);
            m.setBandDrive(1, highDrive);
        };
    };
    const auto out = mbRun(in, in, split(24.0f));
    const auto calm = mbRun(in, in, split(0.0f));
    EXPECT_GT(sat8Line(out[0], 9000.0) - sat8Line(out[0], 3000.0), -40.0);
    EXPECT_LT(sat8Line(calm[0], 9000.0) - sat8Line(calm[0], 3000.0), -40.0);
    // The low band is what it was (its own soft curve at no drive included).
    EXPECT_NEAR(sat8Line(out[0], 300.0), sat8Line(calm[0], 300.0), 0.5);
    EXPECT_NEAR(sat8Line(out[0], 100.0), sat8Line(calm[0], 100.0), 0.05);
}

// New: level, pan, mute and solo per band, with fades and no click; a band
// that is not heard is not computed and comes back clean.
DSPARK_TEST(MultibandSaturation_level_pan_mute_and_solo)
{
    const Sat8Vec in = mbTwoTones(48000);
    const auto two = [](MbSat& m) {
        m.setNumBands(2);
        m.setCrossoverFrequency(0, 1000.0f);
    };
    const auto plain = mbRun(in, in, two);

    const auto quieter = mbRun(in, in, [&](MbSat& m) { two(m); m.setBandLevel(0, -6.0f); });
    EXPECT_NEAR(sat8Line(quieter[0], 100.0) - sat8Line(plain[0], 100.0), -6.02, 0.1);
    EXPECT_NEAR(sat8Line(quieter[0], 3000.0) - sat8Line(plain[0], 3000.0), 0.0, 0.05);

    const auto panned = mbRun(in, in, [&](MbSat& m) { two(m); m.setBandPan(1, -1.0f); });
    // (What is left on the right is the low band's own share of the tone, 38 dB down.)
    EXPECT_LT(sat8Line(panned[1], 3000.0) - sat8Line(plain[1], 3000.0), -30.0);
    EXPECT_NEAR(sat8Line(panned[0], 3000.0) - sat8Line(plain[0], 3000.0), 0.0, 0.05);

    // Muted half way: the low tone fades out without a click and is not computed.
    bool computedAfter = true;
    const auto muted = mbRun(in, in, two, 512, [&](MbSat& m, size_t at) {
        if (at == 24576) m.setBandMute(0, true);
        if (at + 512 >= 48000) computedAfter = m.isBandComputed(0);
    });
    EXPECT_LT(sat8Line(muted[0], 100.0) - sat8Line(plain[0], 100.0), -70.0);
    EXPECT_FALSE(computedAfter);
    double step = 0.0, steady = 0.0;
    for (size_t i = 1; i < 48000; ++i)
    {
        const double d = std::abs(static_cast<double>(muted[0][i] - muted[0][i - 1]));
        if (i > 20000 && i < 24000) steady = std::max(steady, d);
        if (i >= 24576 && i < 26000) step = std::max(step, d);
    }
    EXPECT_LT(step, steady * 1.1);

    // Soloed: only the soloed band is heard; unmuted later, it comes back.
    const auto soloed = mbRun(in, in, [&](MbSat& m) { two(m); m.setBandSolo(1, true); });
    EXPECT_LT(sat8Line(soloed[0], 100.0) - sat8Line(plain[0], 100.0), -70.0);
    // (Without the low band the tone loses its share of the crossover: -0.11 dB.)
    EXPECT_NEAR(sat8Line(soloed[0], 3000.0) - sat8Line(plain[0], 3000.0), 20.0 * std::log10(81.0 / 82.0), 0.05);
    const auto back = mbRun(in, in, [&](MbSat& m) { two(m); m.setBandMute(0, true); }, 512, [&](MbSat& m, size_t at) {
        if (at == 12288) m.setBandMute(0, false);
    });
    EXPECT_NEAR(sat8Line(back[0], 100.0), sat8Line(plain[0], 100.0), 0.05);

    // Wild values and the state.
    MbSat a;
    two(a);
    a.setBandLevel(0, -3.5f);
    a.setBandLevel(0, std::numeric_limits<float>::quiet_NaN());
    a.setBandLevel(9, 5.0f);
    a.setBandPan(1, 4.0f);
    a.setBandSolo(1, true);
    a.setBandDrive(1, 9.0f);
    EXPECT_EQ(a.getBandLevel(0), -3.5f);
    EXPECT_EQ(a.getBandPan(1), 1.0f);
    const auto blob = a.getState();
    MbSat b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_EQ(b.getNumBands(), 2);
    EXPECT_EQ(b.getBandLevel(0), -3.5f);
    EXPECT_EQ(b.getBandPan(1), 1.0f);
    EXPECT_TRUE(b.isBandSoloed(1));
    EXPECT_TRUE(b.getState() == blob);
}

// New: the TubePreamp's cabinet stage. It darkens the top as a speaker does,
// adds no latency, fades in without a click and is part of the state.
DSPARK_TEST(TubePreamp_cabinet_stage_colours_the_wet_signal_without_latency)
{
    double after = 0.0;
    const auto run = [&after](double hz, bool cabinet, size_t switchAt, double& step, double& steady) {
        auto tp = std::make_unique<TubePreamp<float>>();
        tp->setCabinet(cabinet && switchAt == 0);
        tp->prepare(AudioSpec{ 48000.0, 256, 1 });
        Sat8Vec x = sat8Sine(hz, 0.05, 49152);
        float last = 0.0f;
        step = 0.0;
        steady = 0.0;
        after = 0.0;
        for (size_t off = 0; off < x.size(); off += 256)
        {
            if (switchAt != 0 && off == switchAt) tp->setCabinet(true);
            float* p = x.data() + off;
            tp->processBlock(AudioBufferView<float>(&p, 1, 256));
            for (size_t i = off; i < off + 256; ++i)
            {
                const double d = std::abs(static_cast<double>(x[i] - last));
                last = x[i];
                if (off + 4096 < switchAt && off > 8192) steady = std::max(steady, d);
                if (switchAt != 0 && off >= switchAt && off < switchAt + 2048) step = std::max(step, d);
                if (switchAt != 0 && off >= switchAt + 4096 && off < switchAt + 8192) after = std::max(after, d);
            }
        }
        return x;
    };
    double step = 0.0, steady = 0.0;
    const Sat8Vec open = run(8000.0, false, 0, step, steady);
    const Sat8Vec boxed = run(8000.0, true, 0, step, steady);
    EXPECT_LT(sat8Line(boxed, 8000.0), sat8Line(open, 8000.0) - 10.0);
    const Sat8Vec mid = run(1000.0, true, 0, step, steady);
    const Sat8Vec midOpen = run(1000.0, false, 0, step, steady);
    EXPECT_NEAR(sat8Line(mid, 1000.0), sat8Line(midOpen, 1000.0), 2.0);

    // Switched on while it plays: no click (the speaker's presence makes
    // the slopes of the tone a little steeper once it is on).
    (void)run(1000.0, true, 24576, step, steady);
    EXPECT_LT(step, std::max(steady, after) * 1.1);

    TubePreamp<float> a;
    a.prepare(AudioSpec{ 48000.0, 256, 1 });
    const int latency = a.getLatency();
    a.setCabinet(true);
    a.getCabinet().setSize(0.8f);
    a.getCabinet().setDistance(0.6f);
    EXPECT_EQ(a.getLatency(), latency);
    const auto blob = a.getState();
    TubePreamp<float> b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.isCabinet());
    EXPECT_EQ(b.getCabinet().getSize(), 0.8f);
    EXPECT_EQ(b.getCabinet().getDistance(), 0.6f);
}

// ============================================================================
// LoFi
// ============================================================================

namespace {

// A mono render of a LoFi in blocks of 512, set up before prepare().
template <class Setup>
Sat8Vec lofiRun(const Sat8Vec& in, Setup&& setup)
{
    auto lofi = std::make_unique<LoFi<float>>();
    setup(*lofi);
    lofi->prepare(AudioSpec{ 48000.0, 512, 1 });
    Sat8Vec x = in;
    for (size_t off = 0; off < x.size(); off += 512)
    {
        float* p = x.data() + off;
        lofi->processBlock(AudioBufferView<float>(&p, 1, static_cast<int>(std::min<size_t>(512, x.size() - off))));
    }
    return x;
}

} // namespace

// New: LoFi. At its defaults the output is the input to the bit, and so at a
// mix of 0 whatever the stages do; fewer bits round to their grid, with the
// error of a step's rounding or, with dither, of half a step.
DSPARK_TEST(LoFi_is_transparent_off_and_rounds_to_its_bits)
{
    const Sat8Vec pink = sat8Pink(24000, -14.0, 11u);
    const Sat8Vec plain = lofiRun(pink, [](LoFi<float>&) {});
    double farthest = 0.0;
    for (size_t i = 0; i < pink.size(); ++i) farthest = std::max(farthest, static_cast<double>(std::abs(plain[i] - pink[i])));
    EXPECT_EQ(farthest, 0.0);
    const Sat8Vec dry = lofiRun(pink, [](LoFi<float>& l) { l.setMix(0.0f); l.setBits(2.0f); l.setFold(12.0f); });
    farthest = 0.0;
    for (size_t i = 0; i < pink.size(); ++i) farthest = std::max(farthest, static_cast<double>(std::abs(dry[i] - pink[i])));
    EXPECT_EQ(farthest, 0.0);

    const Sat8Vec rounded = lofiRun(pink, [](LoFi<float>& l) { l.setBits(4.0f); });
    double offGrid = 0.0, error = 0.0;
    for (size_t i = 0; i < pink.size(); ++i)
    {
        const double scaled = static_cast<double>(rounded[i]) * 8.0;
        offGrid = std::max(offGrid, std::abs(scaled - std::round(scaled)));
        const double e = static_cast<double>(rounded[i] - pink[i]);
        error += e * e;
    }
    EXPECT_LT(offGrid, 1e-5);
    EXPECT_NEAR(std::sqrt(error / pink.size()), (1.0 / 8.0) / std::sqrt(12.0), 0.007);
    const Sat8Vec dithered = lofiRun(pink, [](LoFi<float>& l) { l.setBits(4.0f); l.setDither(true); });
    error = 0.0;
    for (size_t i = 0; i < pink.size(); ++i)
    {
        const double e = static_cast<double>(dithered[i] - pink[i]);
        error += e * e;
    }
    EXPECT_NEAR(std::sqrt(error / pink.size()), 0.5 / 8.0, 0.008);
}

// New: the rate of the hold folds what is above half of it back, as a held
// signal does; the filter in front takes that out first.
DSPARK_TEST(LoFi_rate_holds_and_its_filter_takes_out_what_would_fold)
{
    const Sat8Vec tone = sat8Sine(1000.0, 0.5, 24000);
    const Sat8Vec held = lofiRun(tone, [](LoFi<float>& l) { l.setRate(8000.0f); });
    // A hold of six samples: the image at 7 kHz is the zero-order hold's.
    const double image = std::sin(kSat8Pi * 7.0 / 8.0) / (kSat8Pi * 7.0 / 8.0)
                       / (std::sin(kSat8Pi / 8.0) / (kSat8Pi / 8.0));
    EXPECT_NEAR(sat8Line(held, 7000.0) - sat8Line(held, 1000.0), 20.0 * std::log10(image), 0.3);
    for (size_t i = 1; i < held.size(); ++i)
        if (i % 6 != 0) EXPECT_EQ(held[i], held[i - 1]);

    const Sat8Vec high = sat8Sine(6000.0, 0.5, 24000);
    const Sat8Vec folded = lofiRun(high, [](LoFi<float>& l) { l.setRate(8000.0f); });
    const Sat8Vec clean = lofiRun(high, [](LoFi<float>& l) { l.setRate(8000.0f); l.setRateFilter(true); });
    EXPECT_GT(sat8Line(folded, 2000.0) - sat8Line(high, 6000.0), -15.0);
    EXPECT_LT(sat8Line(clean, 2000.0), sat8Line(folded, 2000.0) - 25.0);
}

// New: the fold keeps the level of a sine at -12 dBFS whatever its gain,
// while its harmonics grow; the grains drop the pitch and smear.
DSPARK_TEST(LoFi_fold_keeps_its_level_and_the_grains_drop_the_pitch)
{
    // 997 Hz: a period that is not a whole number of samples, so the samples
    // of the folded tone cover its phases (at 1 kHz the same 48 phases repeat
    // and the mean square of a fold of 36 dB reads 1.8 dB away from the tone's).
    const double hz = 997.0;
    const Sat8Vec tone = sat8Sine(hz, 0.25, 48000);
    const double inLevel = sat8Rms(tone, 24000, 48000);
    for (float fold : { 6.0f, 12.0f, 24.0f, 36.0f })
    {
        const Sat8Vec out = lofiRun(tone, [=](LoFi<float>& l) { l.setFold(fold); });
        EXPECT_NEAR(20.0 * std::log10(sat8Rms(out, 24000, 48000) / inLevel), 0.0, 0.1);
        if (fold >= 24.0f) EXPECT_GT(sat8Line(out, 3.0 * hz) - sat8Line(out, hz), -20.0);
    }

    const Sat8Vec down = lofiRun(tone, [](LoFi<float>& l) { l.setPitch(-12.0f); });
    EXPECT_GT(sat8Line(down, 0.5 * hz), sat8Line(down, hz) + 10.0);
    const Sat8Vec smeared = lofiRun(tone, [](LoFi<float>& l) { l.setSmear(1.0f); });
    EXPECT_NEAR(20.0 * std::log10(sat8Rms(smeared, 24000, 48000) / inLevel), 0.0, 3.0);
    double moved = 0.0;
    for (size_t i = 24000; i < 48000; ++i) moved = std::max(moved, static_cast<double>(std::abs(smeared[i] - tone[i])));
    EXPECT_GT(moved, 0.05);

    // Wild values and the state.
    LoFi<float> a;
    a.setBits(3.5f); a.setRate(11025.0f); a.setRateFilter(true); a.setDither(true);
    a.setFold(9.0f); a.setPitch(-5.0f); a.setSmear(0.4f); a.setMix(0.7f); a.setOutput(-3.0f);
    a.setBits(std::numeric_limits<float>::quiet_NaN());
    a.setRate(10.0f);
    a.setPitch(-40.0f);
    EXPECT_EQ(a.getBits(), 3.5f);
    EXPECT_EQ(a.getRate(), 100.0f);
    EXPECT_EQ(a.getPitch(), -24.0f);
    const auto blob = a.getState();
    LoFi<float> b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.getState() == blob);
    EXPECT_EQ(LoFi<float>::getLatency(), 0);
}

// ============================================================================
// Oversampled effects with minimum-phase filters: the dry copy crosses them
// ============================================================================

namespace
{
// A tone in blocks of 256 through an effect; returns the left channel.
template <class P>
std::vector<float> toneThrough(P& p, double freq, float amp, int length)
{
    constexpr int kBlock = 256;
    std::vector<float> outL(static_cast<size_t>(length));
    std::vector<float> l(kBlock), r(kBlock);
    for (int off = 0; off < length; off += kBlock)
    {
        const int n = std::min(kBlock, length - off);
        for (int i = 0; i < n; ++i)
        {
            const float x = amp * static_cast<float>(std::sin(2.0 * std::numbers::pi * freq * (off + i) / 48000.0));
            l[static_cast<size_t>(i)] = x;
            r[static_cast<size_t>(i)] = x;
        }
        float* ch[2] = { l.data(), r.data() };
        p.processBlock(AudioBufferView<float>(ch, 2, n));
        std::copy_n(l.data(), n, outL.data() + off);
    }
    return outL;
}

// The same tone through a bare round trip of an engine, late by `delay`.
std::vector<float> roundTripThrough(Oversampling<float>& os, double freq, float amp, int length, int delay)
{
    constexpr int kBlock = 256;
    std::vector<float> out(static_cast<size_t>(length), 0.0f);
    std::vector<float> l(kBlock), r(kBlock);
    for (int off = 0; off < length; off += kBlock)
    {
        const int n = std::min(kBlock, length - off);
        for (int i = 0; i < n; ++i)
        {
            const float x = amp * static_cast<float>(std::sin(2.0 * std::numbers::pi * freq * (off + i) / 48000.0));
            l[static_cast<size_t>(i)] = x;
            r[static_cast<size_t>(i)] = x;
        }
        float* ch[2] = { l.data(), r.data() };
        AudioBufferView<float> view(ch, 2, n);
        [[maybe_unused]] const auto high = os.upsample(view);
        os.downsample(view);
        for (int i = 0; i < n; ++i)
            if (off + i + delay < length) out[static_cast<size_t>(off + i + delay)] = l[static_cast<size_t>(i)];
    }
    return out;
}

double worstDifference(const std::vector<float>& a, const std::vector<float>& b, int from)
{
    double worst = 0.0;
    for (size_t i = static_cast<size_t>(from); i < a.size(); ++i)
        worst = std::max(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return worst;
}
} // namespace

DSPARK_TEST(Clipper_minimum_phase_oversampling_mixes_in_phase)
{
    // Below the ceiling the hard clip does nothing, so the wet signal is the
    // oversampler's round trip; behind minimum-phase filters the dry copy
    // crosses the same filters, so all dry and all wet agree at 18 kHz,
    // where the filters' delay is fractional (against the rounded delay they
    // differ by a large share of the tone).
    using C = Clipper<float>;
    int latency[2] = {};
    for (const auto mode : { C::OversamplingMode::LinearPhase, C::OversamplingMode::MinimumPhase })
    {
        std::vector<float> out[2];
        for (int wet = 0; wet < 2; ++wet)
        {
            auto c = std::make_unique<C>();
            c->setMode(C::Mode::Hard);
            c->setCeiling(0.0f);
            c->setMix(wet ? 1.0f : 0.0f);
            c->setOversampling(2);
            c->setOversamplingMode(mode);
            c->prepare(spec(48000.0, 256, 2));
            EXPECT_TRUE(c->getOversamplingMode() == mode);
            latency[mode == C::OversamplingMode::MinimumPhase ? 1 : 0] = c->getLatency();
            out[wet] = toneThrough(*c, 18000.0, 0.25f, 9600);
        }
        if (mode == C::OversamplingMode::MinimumPhase)
        {
            EXPECT_LT(worstDifference(out[0], out[1], 2400), 1e-5);
        }
    }
    EXPECT_LT(latency[1], latency[0]);
}

DSPARK_TEST(TubePreamp_and_TapeMachine_dry_path_crosses_minimum_phase_filters)
{
    // With the mix at zero the output is the dry copy: the oversampler's
    // round trip of the input, late by the rest of the effect's latency.
    {
        using P = TubePreamp<float>;
        auto p = std::make_unique<P>();
        p->setMix(0.0f);
        p->setOversamplingMode(P::OversamplingMode::MinimumPhase);
        p->prepare(spec(48000.0, 256, 2));
        auto linear = std::make_unique<P>();
        linear->prepare(spec(48000.0, 256, 2));
        EXPECT_LT(p->getLatency(), linear->getLatency());

        Oversampling<float> os(2, Oversampling<float>::Quality::High, Oversampling<float>::Mode::PolyphaseIir,
                               Oversampling<float>::Depth::Preset);
        os.prepare(spec(48000.0, 256, 2));
        const auto got = toneThrough(*p, 15000.0, 0.2f, 9600);
        const auto want = roundTripThrough(os, 15000.0, 0.2f, 9600, p->getLatency() - os.getLatency());
        EXPECT_LT(worstDifference(got, want, 2400), 1e-6);
    }
    {
        using P = TapeMachine<float>;
        auto p = std::make_unique<P>();
        p->setMix(0.0f);
        p->setOversamplingMode(P::OversamplingMode::MinimumPhase);
        p->prepare(spec(48000.0, 256, 2));
        auto linear = std::make_unique<P>();
        linear->prepare(spec(48000.0, 256, 2));
        EXPECT_LT(p->getLatency(), linear->getLatency());

        Oversampling<float> os(4, Oversampling<float>::Quality::High, Oversampling<float>::Mode::PolyphaseIir);
        os.prepare(spec(48000.0, 256, 2));
        const auto got = toneThrough(*p, 15000.0, 0.2f, 9600);
        const auto want = roundTripThrough(os, 15000.0, 0.2f, 9600, p->getLatency() - os.getLatency());
        EXPECT_LT(worstDifference(got, want, 2400), 1e-6);
    }
}

DSPARK_TEST(FilterEngine_ladder_structure_matches_the_direct_form)
{
    // The same response at rest and through a glide (the two ramp in their
    // own coefficient spaces, so the glide agrees closely, not to the bit).
    using F = FilterEngine<float>;
    std::vector<float> out[2];
    for (int ladder = 0; ladder < 2; ++ladder)
    {
        F f;
        f.setLowPass(900.0f, 3.0f, 24);
        if (ladder) f.setStructure(F::Structure::Ladder);
        f.prepare(spec(48000.0, 256, 2));
        EXPECT_TRUE(f.getStructure() == (ladder ? F::Structure::Ladder : F::Structure::DirectForm));
        std::vector<float> noise(48000);
        uint32_t seed = 12345u;
        for (auto& v : noise) { seed = seed * 1664525u + 1013904223u; v = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f; v *= 0.3f; }
        out[ladder] = noise;
        for (int off = 0; off < 48000; off += 256)
        {
            if (off == 23808) f.setFrequency(6000.0f);   // a glide in the second half
            float* ch[1] = { out[ladder].data() + off };
            f.processBlock(AudioBufferView<float>(ch, 1, std::min(256, 48000 - off)));
        }
    }
    double peak = 0.0;
    for (float v : out[0]) peak = std::max(peak, std::abs(static_cast<double>(v)));
    EXPECT_LT(worstDifference(out[0], out[1], 0) / peak, 2e-2);
    // At rest, before the glide: rounding apart.
    double rest = 0.0;
    for (size_t i = 0; i < 23808; ++i)
        rest = std::max(rest, std::abs(static_cast<double>(out[0][i]) - static_cast<double>(out[1][i])));
    EXPECT_LT(rest / peak, 1e-4);
}

DSPARK_TEST(Equalizer_ladder_structure_matches_the_direct_form)
{
    // The bands of the minimum-phase engine as ladders: the same response.
    using E = Equalizer<float>;
    std::vector<float> out[2];
    std::vector<float> in(24000);
    uint32_t seed = 4242u;
    for (auto& v : in) { seed = seed * 1664525u + 1013904223u; v = 0.3f * (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f); }
    for (int ladder = 0; ladder < 2; ++ladder)
    {
        auto e = std::make_unique<E>();
        e->setNumBands(3);
        e->setBand(0, 120.0f, 5.0f, 0.8f);
        e->setBand(1, 1800.0f, -7.0f, 2.0f);
        e->setBand(2, 9000.0f, 3.0f, 0.7f);
        if (ladder) e->setStructure(E::Structure::Ladder);
        e->prepare(spec(48000.0, 256, 1));
        EXPECT_TRUE(e->getStructure() == (ladder ? E::Structure::Ladder : E::Structure::DirectForm));
        out[ladder] = in;
        for (int off = 0; off < 24000; off += 256)
        {
            float* ch[1] = { out[ladder].data() + off };
            e->processBlock(AudioBufferView<float>(ch, 1, std::min(256, 24000 - off)));
        }
    }
    double worst = 0.0, peak = 0.0;
    for (size_t i = 0; i < out[0].size(); ++i)
    {
        worst = std::max(worst, std::abs(static_cast<double>(out[0][i]) - static_cast<double>(out[1][i])));
        peak = std::max(peak, std::abs(static_cast<double>(out[0][i])));
    }
    EXPECT_LT(worst / peak, 1e-4);
}

// ============================================================================
// The comb loop that glides and closes, the crossover's slopes, the multiband
// mix and its output clipper
// ============================================================================

namespace {

// A mono render in blocks of 256 with a hook before each block.
template <class Setup, class Hook>
Sat8Vec m6Run(const Sat8Vec& in, Setup&& setup, Hook&& hook)
{
    auto s = std::make_unique<Sat8>();
    setup(*s);
    s->prepare(AudioSpec{ 48000.0, 256, 1 });
    Sat8Vec x = in;
    for (size_t off = 0; off < x.size(); off += 256)
    {
        hook(*s, off);
        float* p = x.data() + off;
        s->processBlock(AudioBufferView<float>(&p, 1, static_cast<int>(std::min<size_t>(256, x.size() - off))));
    }
    return x;
}

// A crossover run on one channel: every band's output.
std::vector<Sat8Vec> m6Split(CrossoverFilter<float, 6>& x, const Sat8Vec& in, int bands)
{
    std::vector<Sat8Vec> out(static_cast<size_t>(bands), Sat8Vec(in.size(), 0.0f));
    Sat8Vec feed = in;
    for (size_t off = 0; off < in.size(); off += 256)
    {
        const int n = static_cast<int>(std::min<size_t>(256, in.size() - off));
        float* ip = feed.data() + off;
        float* ptr[6] = {};
        std::array<AudioBufferView<float>, 6> views;
        for (int b = 0; b < bands; ++b)
        {
            ptr[b] = out[static_cast<size_t>(b)].data() + off;
            views[static_cast<size_t>(b)] = AudioBufferView<float>(&ptr[b], 1, n);
        }
        x.processBlock(AudioBufferView<float>(&ip, 1, n), views.data(), bands);
    }
    return out;
}

Sat8Vec m6Noise(int n, double amp, uint32_t seed)
{
    Sat8Vec v(static_cast<size_t>(n));
    for (auto& x : v) { seed = seed * 1664525u + 1013904223u; x = static_cast<float>(amp * (static_cast<double>(static_cast<int32_t>(seed)) / 2147483648.0)); }
    return v;
}

} // namespace

// Defect: a new frequency of the comb loop moved the read point of its line
// at once, by a whole number of samples: a cut in a signal that recirculates,
// a click each time. The delay now glides to its new length (an octave in a
// tenth of a second) and is read between samples on the way.
DSPARK_TEST(Saturation_comb_loop_glides_to_a_new_frequency)
{
    const Sat8Vec in = sat8Sine(150.0, 0.2, 48000);
    const size_t at = 256 * 100;
    const Sat8Vec out = m6Run(in, [](Sat8& s) { s.setCombLoop(0.9f, 200.0f); },
                              [&](Sat8& s, size_t off) { if (off == at) s.setCombLoop(0.9f, 150.0f); });
    // The largest step between samples before the change, on the way and
    // once the loop has settled on the new frequency (where the tone has
    // another level): the way adds no step of its own.
    double steady = 0.0, moving = 0.0, settled = 0.0;
    for (size_t i = at - 9600; i < at; ++i) steady = std::max(steady, std::abs(static_cast<double>(out[i]) - out[i - 1]));
    for (size_t i = at; i < at + 9600; ++i) moving = std::max(moving, std::abs(static_cast<double>(out[i]) - out[i - 1]));
    for (size_t i = out.size() - 4800; i < out.size(); ++i) settled = std::max(settled, std::abs(static_cast<double>(out[i]) - out[i - 1]));
    std::printf("  [comb] largest step of the output: %.5f before the new frequency, %.5f while the loop moves to it, %.5f settled on it\n",
                steady, moving, settled);
    EXPECT_LT(moving, 1.5 * std::max(steady, settled));
    // It lands on the new frequency: a loop of 150 Hz at the end.
    const Sat8Vec direct = m6Run(in, [](Sat8& x) { x.setCombLoop(0.9f, 150.0f); }, [](Sat8&, size_t) {});
    double apart = 0.0;
    for (size_t i = in.size() - 4800; i < in.size(); ++i) apart = std::max(apart, std::abs(static_cast<double>(out[i]) - direct[i]));
    EXPECT_LT(apart, 1e-3);
}

// New: setCombLoopClosing(true). With a loop gain near one the comb rings on
// after the programme has stopped; closing follows the level of the input
// and shuts the loop once there is none.
DSPARK_TEST(Saturation_comb_loop_closes_when_the_programme_stops)
{
    Sat8Vec in = m6Noise(48000, 0.3, 7u);
    std::fill(in.begin() + 24000, in.end(), 0.0f);
    const auto tail = [&](bool closing) {
        const Sat8Vec out = m6Run(in, [=](Sat8& s) { s.setCombLoop(0.95f, 200.0f); s.setCombLoopClosing(closing); }, [](Sat8&, size_t) {});
        // From 250 to 350 ms after the programme stops, in dB.
        return 20.0 * std::log10(sat8Rms(out, 24000 + 12000, 24000 + 16800) + 1e-30);
    };
    const double open = tail(false), closed = tail(true);
    std::printf("  [comb] the ring 300 ms after the programme stops: %.1f dB as it was, %.1f dB closing\n", open, closed);
    EXPECT_LT(closed, open - 20.0);
    // While there is programme it changes nothing.
    const Sat8Vec loud = m6Noise(24000, 0.3, 9u);
    const Sat8Vec a = m6Run(loud, [](Sat8& s) { s.setCombLoop(0.9f, 200.0f); }, [](Sat8&, size_t) {});
    const Sat8Vec b = m6Run(loud, [](Sat8& s) { s.setCombLoop(0.9f, 200.0f); s.setCombLoopClosing(true); }, [](Sat8&, size_t) {});
    double apart = 0.0;
    for (size_t i = 0; i < a.size(); ++i) apart = std::max(apart, std::abs(static_cast<double>(a[i]) - b[i]));
    EXPECT_LT(apart, 1e-6);
    Sat8 state;
    state.setCombLoopClosing(true);
    const auto blob = state.getState();
    Sat8 other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_TRUE(other.getCombLoopClosing());
}

// New: a split of 6 dB per octave is a first-order low-pass and its
// complement: the bands add up to the input itself, in the minimum-phase
// engine with either structure, and to the delayed input in linear phase.
DSPARK_TEST(CrossoverFilter_six_decibel_split_sums_to_the_input)
{
    using X = CrossoverFilter<float, 6>;
    const Sat8Vec in = m6Noise(24000, 0.5, 21u);
    for (const X::Structure structure : { X::Structure::DirectForm, X::Structure::Ladder })
    {
        auto x = std::make_unique<X>();
        x->setNumBands(4);
        x->setOrder(6);
        x->setStructure(structure);
        x->setCrossoverFrequency(0, 200.0f);
        x->setCrossoverFrequency(1, 1000.0f);
        x->setCrossoverFrequency(2, 5000.0f);
        x->prepare(AudioSpec{ 48000.0, 256, 1 });
        EXPECT_EQ(x->getOrder(), 6);
        const auto bands = m6Split(*x, in, 4);
        double worst = 0.0;
        for (size_t i = 0; i < in.size(); ++i)
        {
            double sum = 0.0;
            for (const auto& band : bands) sum += static_cast<double>(band[i]);
            worst = std::max(worst, std::abs(sum - static_cast<double>(in[i])));
        }
        EXPECT_LT(worst, 2e-6);
    }
    // The slope: an octave over the split, the low band of a two-band split is 7 dB down (a first-order filter).
    {
        auto x = std::make_unique<X>();
        x->setNumBands(2);
        x->setOrder(6);
        x->setCrossoverFrequency(0, 1000.0f);
        x->prepare(AudioSpec{ 48000.0, 256, 1 });
        const Sat8Vec tone = sat8Sine(2000.0, 0.25, 24000);
        const auto bands = m6Split(*x, tone, 2);
        EXPECT_NEAR(sat8Line(bands[0], 2000.0) - sat8Line(tone, 2000.0), -7.0, 0.3);
    }
    // Linear phase: the bands sum to the input, late by the latency.
    {
        auto x = std::make_unique<X>();
        x->setNumBands(3);
        x->setOrder(6);
        x->setFilterMode(X::FilterMode::LinearPhase);
        x->setCrossoverFrequency(0, 300.0f);
        x->setCrossoverFrequency(1, 3000.0f);
        x->prepare(AudioSpec{ 48000.0, 256, 1 });
        const auto bands = m6Split(*x, in, 3);
        const auto lat = static_cast<size_t>(x->getLatency());
        double worst = 0.0;
        for (size_t i = lat; i < in.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(bands[0][i]) + bands[1][i] + bands[2][i] - in[i - lat]));
        EXPECT_LT(worst, 1e-4);
    }
}

// New: a split of 36 dB per octave (two third-order Butterworth filters per
// branch). The bands sum flat, each branch is 6 dB down at the split and 36
// dB down an octave past it, in both structures and in linear phase.
DSPARK_TEST(CrossoverFilter_thirty_six_decibel_split_is_flat_and_steep)
{
    using X = CrossoverFilter<float, 6>;
    const auto make = [](int bands, X::Structure structure, X::FilterMode mode) {
        auto x = std::make_unique<X>();
        x->setNumBands(bands);
        x->setOrder(36);
        x->setStructure(structure);
        x->setFilterMode(mode);
        if (bands == 2) x->setCrossoverFrequency(0, 1000.0f);
        else { x->setCrossoverFrequency(0, 250.0f); x->setCrossoverFrequency(1, 1000.0f); x->setCrossoverFrequency(2, 4000.0f); }
        x->prepare(AudioSpec{ 48000.0, 256, 1 });
        return x;
    };
    for (const X::Structure structure : { X::Structure::DirectForm, X::Structure::Ladder })
    {
        // The branches of one split.
        for (const double hz : { 500.0, 1000.0, 2000.0 })
        {
            auto x = make(2, structure, X::FilterMode::MinimumPhase);
            const Sat8Vec tone = sat8Sine(hz, 0.25, 24000);
            const auto bands = m6Split(*x, tone, 2);
            const double low = sat8Line(bands[0], hz) - sat8Line(tone, hz), high = sat8Line(bands[1], hz) - sat8Line(tone, hz);
            if (hz == 1000.0) { EXPECT_NEAR(low, -6.02, 0.05); EXPECT_NEAR(high, -6.02, 0.05); }
            if (hz == 2000.0) { EXPECT_NEAR(low, -36.1, 0.5); }
            if (hz == 500.0) { EXPECT_NEAR(high, -36.1, 0.5); }
        }
        // Four bands sum flat at the splits and between them.
        for (const double hz : { 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 10000.0 })
        {
            auto x = make(4, structure, X::FilterMode::MinimumPhase);
            const Sat8Vec tone = sat8Sine(hz, 0.25, 24000);
            const auto bands = m6Split(*x, tone, 4);
            Sat8Vec sum(tone.size(), 0.0f);
            for (const auto& band : bands)
                for (size_t i = 0; i < sum.size(); ++i) sum[i] += band[i];
            EXPECT_NEAR(sat8Line(sum, hz) - sat8Line(tone, hz), 0.0, 0.01);
        }
    }
    // Linear phase: the same slope, and the bands sum to the delayed input.
    {
        auto x = make(2, X::Structure::DirectForm, X::FilterMode::LinearPhase);
        const Sat8Vec tone = sat8Sine(2000.0, 0.25, 24000);
        const auto bands = m6Split(*x, tone, 2);
        EXPECT_NEAR(sat8Line(bands[0], 2000.0) - sat8Line(tone, 2000.0), -36.2, 0.7);
        const Sat8Vec noise = m6Noise(24000, 0.5, 33u);
        auto y = make(4, X::Structure::DirectForm, X::FilterMode::LinearPhase);
        const auto split = m6Split(*y, noise, 4);
        const auto lat = static_cast<size_t>(y->getLatency());
        double worst = 0.0;
        for (size_t i = lat; i < noise.size(); ++i)
            worst = std::max(worst, std::abs(static_cast<double>(split[0][i]) + split[1][i] + split[2][i] + split[3][i] - noise[i - lat]));
        EXPECT_LT(worst, 1e-4);
    }
    // Slopes that do not exist are still ignored.
    X x;
    x.setOrder(36);
    x.setOrder(18);
    EXPECT_EQ(x.getOrder(), 36);
}

// New: MultibandSaturation::setMix(). The dry side is the programme as the
// crossover leaves it, as late as the bands, so any mix is as flat as the
// crossover: at the splits too, where a dry signal in another phase would
// carve a notch.
DSPARK_TEST(MultibandSaturation_mix_keeps_the_phase_of_the_crossover)
{
    const auto three = [](float mix) {
        return [=](MbSat& m) {
            m.setNumBands(3);
            m.setCrossoverFrequency(0, 300.0f);
            m.setCrossoverFrequency(1, 3000.0f);
            m.getBandSaturation(1).setOversampling(4);     // the bands are not equally late
            m.setMix(mix);
        };
    };
    for (const double hz : { 100.0, 300.0, 1000.0, 3000.0, 10000.0 })
    {
        const Sat8Vec in = sat8Sine(hz, 0.01, 24000);
        const auto amplitude = [&](float mix) { return std::pow(10.0, sat8Line(mbRun(in, in, three(mix))[0], hz) / 20.0); };
        const double source = std::pow(10.0, sat8Line(in, hz) / 20.0);
        const double dry = amplitude(0.0f), half = amplitude(0.5f), wet = amplitude(1.0f);
        EXPECT_NEAR(dry / source, 1.0, 0.005);                 // the dry side: the flat sum of the crossover
        // It meets the bands in phase. Under the lowest split the bands' own
        // filters (each Saturation's 20 Hz high-pass and its DC filter) lead
        // the dry side a little: 20 degrees at 100 Hz, 0.14 dB at this mix.
        EXPECT_NEAR(half, 0.5 * (dry + wet), (hz < 200.0 ? 0.02 : 0.003) * source);
    }
    // A driven band under a mix of one half: half way, in amplitude, between the dry and the wet.
    const Sat8Vec in = sat8Sine(1000.0, 0.25, 24000);
    const auto driven = [](float mix) {
        return [=](MbSat& m) { m.setNumBands(2); m.setCrossoverFrequency(0, 200.0f); m.setBandLevel(1, -12.0f); m.setMix(mix); };
    };
    const double dry = std::pow(10.0, sat8Line(mbRun(in, in, driven(0.0f))[0], 1000.0) / 20.0);
    const double wet = std::pow(10.0, sat8Line(mbRun(in, in, driven(1.0f))[0], 1000.0) / 20.0);
    const double half = std::pow(10.0, sat8Line(mbRun(in, in, driven(0.5f))[0], 1000.0) / 20.0);
    EXPECT_NEAR(half, 0.5 * (dry + wet), 0.01 * dry);
    EXPECT_LT(wet, 0.5 * dry);
    // Part of the state.
    MbSat m;
    m.setMix(0.25f);
    m.setOutputClip(true);
    m.setOutputCeiling(-6.0f);
    const auto blob = m.getState();
    MbSat other;
    EXPECT_TRUE(other.setState(blob.data(), blob.size()));
    EXPECT_NEAR(static_cast<double>(other.getMix()), 0.25, 1e-6);
    EXPECT_TRUE(other.getOutputClip());
    EXPECT_NEAR(static_cast<double>(other.getOutputCeiling()), -6.0, 1e-6);
}

// New: MultibandSaturation::setOutputClip(). Driven bands add up to more than
// full scale; with the clipper after the sum no sample leaves over its
// ceiling, and a programme under the ceiling leaves as it would without it,
// late by the clipper's latency.
DSPARK_TEST(MultibandSaturation_output_clip_holds_its_ceiling)
{
    const Sat8Vec loud = m6Noise(24000, 0.9, 41u);
    const auto six = [](bool clip, float ceilingDb, float level) {
        return [=](MbSat& m) {
            m.setNumBands(6);
            for (int b = 0; b < 6; ++b) { m.setBandDrive(b, 24.0f); m.setBandLevel(b, level); }
            m.setOutputClip(clip);
            m.setOutputCeiling(ceilingDb);
        };
    };
    const auto peak = [](const Sat8Vec& v) { double top = 0.0; for (float x : v) top = std::max(top, std::abs(static_cast<double>(x))); return top; };
    const double free = peak(mbRun(loud, loud, six(false, 0.0f, 6.0f))[0]);
    const double held = peak(mbRun(loud, loud, six(true, 0.0f, 6.0f))[0]);
    const double lower = peak(mbRun(loud, loud, six(true, -6.0f, 6.0f))[0]);
    std::printf("  [clip] six driven bands peak at %.2f of full scale; %.4f with the clipper, %.4f with its ceiling at -6 dB\n", free, held, lower);
    EXPECT_GT(free, 1.5);
    EXPECT_LT(held, 1.0 + 1e-6);
    EXPECT_LT(lower, 0.5012 + 1e-6);
    // Under the ceiling: the same signal, late by the clipper.
    const Sat8Vec quiet = sat8Sine(1000.0, 0.05, 24000);
    const auto calm = [](bool clip) { return [=](MbSat& m) { m.setNumBands(3); m.setOutputClip(clip); }; };
    const auto plain = mbRun(quiet, quiet, calm(false)), clipped = mbRun(quiet, quiet, calm(true));
    MbSat a, b;
    calm(false)(a); calm(true)(b);
    a.prepare(AudioSpec{ 48000.0, 512, 2 }); b.prepare(AudioSpec{ 48000.0, 512, 2 });
    const int late = b.getLatency() - a.getLatency();
    EXPECT_GT(late, 0);
    double apart = 0.0;
    for (size_t i = 4800; i + static_cast<size_t>(late) < quiet.size(); ++i)
        apart = std::max(apart, std::abs(static_cast<double>(clipped[0][i + static_cast<size_t>(late)]) - plain[0][i]));
    EXPECT_LT(apart, 1e-4);
}

// ============================================================================
// The tape machine's input, its bias against the top end, the record emphasis
// trim, the iron's low end, the head by speed and track, rest and block size.
// ============================================================================

namespace {

// A sum of tones (frequency, amplitude) over a DC offset.
std::vector<float> m7Tones(const std::vector<std::pair<double, double>>& list, int n, double dc = 0.0)
{
    std::vector<float> v(static_cast<size_t>(n), static_cast<float>(dc));
    for (const auto& tone : list)
        for (int i = 0; i < n; ++i)
            v[static_cast<size_t>(i)] += static_cast<float>(tone.second * std::sin(2.0 * 3.14159265358979323846 * tone.first * i / 48000.0));
    return v;
}

// The left channel of a machine with a steady transport.
template <class Setup>
std::vector<float> m7Tape(Setup&& setup, const std::vector<float>& in)
{
    return l10Run([&](Tape10& t) { t.setWowFlutter(0.0f); setup(t); }, in, in).first;
}

// Gain of a tone through the machine, in dB.
template <class Setup>
double m7Gain(Setup&& setup, double hz, double amp, int n = 96000)
{
    const std::vector<float> in = m7Tones({ { hz, amp } }, n);
    return l10Line(m7Tape(setup, in), hz) - l10Line(in, hz);
}

} // namespace

// New: the input is AC coupled. DC at the input reached the tape and leaned
// its loop to one side: a second harmonic that no later stage removes (the
// output coupling only took the DC itself). It no longer reaches it, and the
// band stays where it was.
DSPARK_TEST(TapeMachine_input_is_ac_coupled)
{
    const int n = 96000;
    const auto second = [](const std::vector<float>& v) { return l10Line(v, 2000.0) - l10Line(v, 1000.0); };
    const double clean = second(m7Tape([](Tape10&) {}, m7Tones({ { 1000.0, 0.25 } }, n)));
    const double offset = second(m7Tape([](Tape10&) {}, m7Tones({ { 1000.0, 0.25 } }, n, 0.1)));
    std::printf("  [tape] second harmonic of 1 kHz: %.1f dB clean, %.1f dB with DC 0.1 at the input\n", clean, offset);
    EXPECT_LT(offset, std::max(clean + 3.0, -75.0));
    EXPECT_NEAR(m7Gain([](Tape10&) {}, 40.0, 0.02), m7Gain([](Tape10&) {}, 1000.0, 0.02), 1.0);
}

// New: the bias moves the top end both ways. Under the optimum the short
// wavelengths record more strongly and the top rises; over it they erase
// themselves and it falls, from the optimum on.
DSPARK_TEST(TapeMachine_bias_moves_the_top_end_both_ways)
{
    const std::vector<float> in = m7Tones({ { 1000.0, 0.02 }, { 10000.0, 0.02 } }, 96000);
    double last = 1e9, atOptimum = 0.0, low = 0.0, high = 0.0;
    for (const float bias : { 0.2f, 0.35f, 0.5f, 0.54f, 0.65f, 0.8f })
    {
        const std::vector<float> out = m7Tape([=](Tape10& t) { t.setBias(bias); }, in);
        const double top = l10Line(out, 10000.0) - l10Line(out, 1000.0);
        std::printf("  [tape] bias %.2f: 10 kHz against 1 kHz %+.2f dB\n", static_cast<double>(bias), top);
        EXPECT_LT(top, last - 0.05);
        last = top;
        if (bias == 0.5f) atOptimum = top;
        if (bias == 0.2f) low = top;
        if (bias == 0.8f) high = top;
    }
    EXPECT_GT(low, atOptimum + 1.0);
    EXPECT_LT(high, atOptimum - 1.0);
}

// New: the record emphasis trim. More emphasis on the record side, taken off
// by the reproduce side: flat at low level, the top end reaches the tape
// hotter (a loud tone up there distorts more) and the tape's hiss falls.
DSPARK_TEST(TapeMachine_record_emphasis_is_flat_at_low_level_and_loads_the_top)
{
    const auto with = [](float db) { return [=](Tape10& t) { t.setRecordEmphasis(db); }; };
    for (const double hz : { 100.0, 1000.0, 10000.0 })
        EXPECT_NEAR(m7Gain(with(6.0f), hz, 0.01), m7Gain(with(0.0f), hz, 0.01), 0.3);
    // The third harmonic of a 5 kHz tone at -6 dBFS, against the tone.
    const auto third = [&](float db) {
        const std::vector<float> in = m7Tones({ { 5000.0, 0.5 } }, 96000);
        const std::vector<float> out = m7Tape(with(db), in);
        return l10Line(out, 15000.0) - l10Line(out, 5000.0);
    };
    const double plain = third(0.0f), loaded = third(6.0f), spared = third(-6.0f);
    std::printf("  [tape] third harmonic of 5 kHz at -6 dBFS: %.1f dB with the trim at 0, %.1f at +6, %.1f at -6\n", plain, loaded, spared);
    EXPECT_GT(loaded, plain + 0.5);
    EXPECT_LT(spared, plain - 1.0);

    const std::vector<float> silence(96000, 0.0f);
    const auto hissTop = [&](float db) {
        return l10Band(m7Tape([=](Tape10& t) { t.setRecordEmphasis(db); t.setNoise(-50.0f); }, silence), 8000.0, 16000.0);
    };
    EXPECT_LT(hissTop(6.0f), hissTop(0.0f) - 3.0);

    Tape10 a;
    a.setRecordEmphasis(4.0f);
    a.setRecordEmphasis(std::numeric_limits<float>::quiet_NaN());
    const auto blob = a.getState();
    Tape10 b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_NEAR(static_cast<double>(b.getRecordEmphasis()), 4.0, 1e-6);
    b.setRecordEmphasis(40.0f);
    EXPECT_NEAR(static_cast<double>(b.getRecordEmphasis()), 6.0, 1e-6);
}

// New: the head bump sits where the wavelength meets the head's contact
// with the tape: it follows the speed (half the speed, half the frequency)
// and the track format (each has its head).
DSPARK_TEST(TapeMachine_head_bump_follows_the_speed_and_the_track)
{
    using S = Tape10::Speed;
    using F = Tape10::TrackFormat;
    const auto lift = [](S speed, F format, double hz) {
        const auto with = [=](float amount) {
            return [=](Tape10& t) { t.setSpeed(speed); t.setTrackFormat(format); t.setHeadBump(amount); };
        };
        return m7Gain(with(1.0f), hz, 0.008) - m7Gain(with(0.0f), hz, 0.008);
    };
    EXPECT_GT(lift(S::IPS_15, F::Half, 90.0), lift(S::IPS_15, F::Half, 45.0) + 0.2);
    EXPECT_GT(lift(S::IPS_15, F::Half, 90.0), lift(S::IPS_15, F::Half, 180.0) + 0.2);
    EXPECT_GT(lift(S::IPS_15, F::Full, 72.0), lift(S::IPS_15, F::Full, 112.0) + 0.1);
    EXPECT_GT(lift(S::IPS_15, F::Quarter, 112.0), lift(S::IPS_15, F::Quarter, 72.0) + 0.1);
    EXPECT_GT(lift(S::IPS_7_5, F::Half, 45.0), lift(S::IPS_7_5, F::Half, 22.5) + 0.2);
    EXPECT_GT(lift(S::IPS_3_75, F::Half, 22.5), lift(S::IPS_3_75, F::Half, 45.0) + 0.2);
}

// New: 3.75 ips. The slower the tape, the shorter the wavelengths and the
// more of the top the head loses; the reference level stays where it is. The
// speed comes last in the enumeration, so stored states keep their meaning.
DSPARK_TEST(TapeMachine_three_and_three_quarter_ips)
{
    using S = Tape10::Speed;
    double last = -1e9;
    for (const S speed : { S::IPS_3_75, S::IPS_7_5, S::IPS_15, S::IPS_30 })
    {
        const double top = m7Gain([=](Tape10& t) { t.setSpeed(speed); t.setLossEffects(1.0f); }, 10000.0, 0.02);
        std::printf("  [tape] speed %d: 10 kHz at full losses %.2f dB\n", static_cast<int>(speed), top);
        EXPECT_GT(top, last + 0.5);
        last = top;
    }
    for (const int standard : { 0, 1 })
        EXPECT_NEAR(m7Gain([=](Tape10& t) { t.setSpeed(S::IPS_3_75); t.setStandard(static_cast<Tape10::Standard>(standard)); }, 1000.0, 0.25),
                    0.0, 0.5);
    EXPECT_EQ(static_cast<int>(S::IPS_7_5), 0);
    EXPECT_EQ(static_cast<int>(S::IPS_30), 2);
    Tape10 a;
    a.setSpeed(S::IPS_3_75);
    const auto blob = a.getState();
    Tape10 b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(b.getSpeed() == S::IPS_3_75);
}

// New: the output iron loses its low end in a linear way: a first-order
// high-pass at 20 Hz at the full amount, whatever the level.
DSPARK_TEST(TapeMachine_output_iron_loses_its_low_end)
{
    const auto drop = [](double hz) {
        return m7Gain([](Tape10& t) { t.setOutputIron(1.0f); }, hz, 0.005, 192000) - m7Gain([](Tape10&) {}, hz, 0.005, 192000);
    };
    const double at20 = drop(20.0), at100 = drop(100.0), at1k = drop(1000.0);
    std::printf("  [tape] the iron at 1 against 0: %.2f dB at 20 Hz, %.2f at 100 Hz, %.2f at 1 kHz\n", at20, at100, at1k);
    EXPECT_NEAR(at20, -3.0, 0.5);
    EXPECT_GT(at100, -0.3);
    EXPECT_NEAR(at1k, 0.0, 0.05);
}

// New: the machine rests while it is not heard (the mix on the dry side, or
// the source monitored): the output is the input, late by the latency, to
// the bit. Asked for again it starts from rest and comes in under the fade:
// no step, and it ends where a machine that never rested is.
DSPARK_TEST(TapeMachine_rests_when_it_is_not_heard_and_wakes_without_a_step)
{
    const std::vector<float> in = m7Tones({ { 1000.0, 0.25 } }, 96000);
    const std::vector<float> never = m7Tape([](Tape10&) {}, in);
    const size_t at = 512 * 60;
    for (const int way : { 0, 1 })
    {
        auto tape = std::make_unique<Tape10>();
        tape->setWowFlutter(0.0f);
        if (way == 0) tape->setMix(0.0f);
        else tape->setMonitor(Tape10::Monitor::Source);
        tape->prepare(spec(48000.0, 512, 2));
        std::vector<float> l = in, r = in;
        for (size_t off = 0; off < l.size(); off += 512)
        {
            if (off == at)
            {
                if (way == 0) tape->setMix(1.0f);
                else tape->setMonitor(Tape10::Monitor::PlayHead);
            }
            float* p[2] = { l.data() + off, r.data() + off };
            tape->processBlock(AudioBufferView<float>(p, 2, static_cast<int>(std::min<size_t>(512, l.size() - off))));
        }
        const size_t late = static_cast<size_t>(tape->getLatency());
        double apart = 0.0;
        for (size_t i = late; i < at; ++i) apart = std::max(apart, std::abs(static_cast<double>(l[i]) - in[i - late]));
        EXPECT_EQ(apart, 0.0);
        double resting = 0.0, waking = 0.0, settled = 0.0;
        for (size_t i = at - 4800; i < at; ++i) resting = std::max(resting, std::abs(static_cast<double>(l[i]) - l[i - 1]));
        for (size_t i = at; i < at + 4800; ++i) waking = std::max(waking, std::abs(static_cast<double>(l[i]) - l[i - 1]));
        for (size_t i = l.size() - 4800; i < l.size(); ++i) settled = std::max(settled, std::abs(static_cast<double>(l[i]) - l[i - 1]));
        std::printf("  [tape] largest step: %.5f at rest, %.5f waking, %.5f settled\n", resting, waking, settled);
        EXPECT_LT(waking, 1.2 * std::max(resting, settled));
        EXPECT_NEAR(l10Line(l, 1000.0), l10Line(never, 1000.0), 0.05);
    }
}

// New: asked for another block size alone, a prepared machine keeps its
// memory: the output goes on as if nothing had been asked. The same spec
// again still starts it from rest.
DSPARK_TEST(TapeMachine_keeps_its_memory_when_only_the_block_size_changes)
{
    const std::vector<float> in = m7Tones({ { 440.0, 0.25 } }, 48000);
    const auto render = [&](int asked) {
        auto tape = std::make_unique<Tape10>();
        tape->prepare(spec(48000.0, 512, 2));
        std::vector<float> l = in, r = in;
        for (size_t off = 0; off < l.size(); off += 256)
        {
            if (off == 24064 && asked == 1) tape->prepare(spec(48000.0, 256, 2));
            if (off == 24064 && asked == 2) tape->prepare(spec(48000.0, 512, 2));
            float* p[2] = { l.data() + off, r.data() + off };
            tape->processBlock(AudioBufferView<float>(p, 2, static_cast<int>(std::min<size_t>(256, l.size() - off))));
        }
        return l;
    };
    const std::vector<float> steady = render(0), resized = render(1), again = render(2);
    double apart = 0.0, restart = 0.0;
    for (size_t i = 0; i < steady.size(); ++i)
    {
        apart = std::max(apart, std::abs(static_cast<double>(steady[i]) - resized[i]));
        restart = std::max(restart, std::abs(static_cast<double>(steady[i]) - again[i]));
    }
    EXPECT_EQ(apart, 0.0);
    EXPECT_GT(restart, 0.05);
}
