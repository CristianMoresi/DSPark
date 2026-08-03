// DSPark Tests - Analysis
// LevelFollower, SpectrumAnalyzer, LoudnessMeter, Goertzel, PitchDetector

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Analysis/LevelFollower.h"
#include "../Analysis/SpectrumAnalyzer.h"
#include "../Analysis/LoudnessMeter.h"
#include "../Analysis/Goertzel.h"
#include "../Analysis/PitchDetector.h"
#include "../Analysis/PitchFollower.h"
#include "../Analysis/PhaseCorrelation.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// LevelFollower
// ============================================================================

DSPARK_TEST(LevelFollower_sine_0dBFS_peak)
{
    LevelFollower<float> lf;
    lf.prepare(defaultSpec());
    lf.setAttackMs(1.0f);
    lf.setReleaseMs(50.0f);

    // Feed full-scale sine for several blocks
    for (int block = 0; block < 10; ++block)
    {
        auto tb = makeStereoBuffer(512);
        tb.fillSine(440.0f, 44100.0f);
        lf.process(tb.view());
    }

    float peakDb = lf.getPeakLevelDb(0);
    EXPECT_NEAR(peakDb, 0.0f, 1.0f); // +/-1 dB
}

DSPARK_TEST(LevelFollower_sine_minus6dBFS)
{
    LevelFollower<float> lf;
    lf.prepare(defaultSpec());
    lf.setAttackMs(1.0f);
    lf.setReleaseMs(50.0f);

    for (int block = 0; block < 10; ++block)
    {
        auto tb = makeStereoBuffer(512);
        tb.fillSine(440.0f, 44100.0f, 0.5f); // -6 dBFS
        lf.process(tb.view());
    }

    float peakDb = lf.getPeakLevelDb(0);
    EXPECT_NEAR(peakDb, -6.0f, 1.5f);
}

DSPARK_TEST(LevelFollower_silence)
{
    LevelFollower<float> lf;
    lf.prepare(defaultSpec());

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    lf.process(tb.view());

    float peak = lf.getPeakLevel(0);
    EXPECT_NEAR(peak, 0.0f, 1e-6f);
}

DSPARK_TEST(LevelFollower_RMS_sine)
{
    LevelFollower<float> lf;
    lf.prepare(defaultSpec());
    lf.setAttackMs(1.0f);
    lf.setReleaseMs(100.0f);
    lf.setRmsWindowMs(30.0f); // short time constant: fully settled below

    // The RMS detector is a symmetric one-pole over x^2 (exponential RMS):
    // once settled, a full-scale sine must read its true RMS, -3.01 dB.
    for (int block = 0; block < 20; ++block)
    {
        auto tb = makeStereoBuffer(512);
        tb.fillSine(440.0f, 44100.0f);
        lf.process(tb.view());
    }

    float rmsDb = lf.getRmsLevelDb(0);
    EXPECT_NEAR(rmsDb, -3.01f, 0.3f);
}

DSPARK_TEST(LevelFollower_invalid_inputs_are_ignored)
{
    // DUT receives NaN/Inf setters and invalid prepares mid-stream; readings
    // must stay BIT-identical to an untouched twin. The old header fails
    // three ways: setReleaseMs(+Inf) silently froze the meter (coeff = 1),
    // setAttackMs(NaN) silently became 0.001 ms, and prepare with a NaN rate
    // published NaN straight to the UI readout.
    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();

    LevelFollower<float> dut, ref;
    dut.prepare(defaultSpec());
    ref.prepare(defaultSpec());
    dut.setAttackMs(5.0f);      ref.setAttackMs(5.0f);
    dut.setReleaseMs(90.0f);    ref.setReleaseMs(90.0f);
    dut.setRmsWindowMs(150.0f); ref.setRmsWindowMs(150.0f);

    float maxDiff  = 0.0f;
    int   badReads = 0;
    for (int b = 0; b < 30; ++b)
    {
        if (b == 5)
        {
            dut.setAttackMs(kNan);
            dut.setReleaseMs(kInf);
            dut.setRmsWindowMs(-kInf);
        }
        if (b == 10) { AudioSpec bad = defaultSpec(); bad.sampleRate = kNan; dut.prepare(bad); }
        if (b == 15) { AudioSpec bad = defaultSpec(); bad.numChannels = -3; dut.prepare(bad); }

        auto tb = makeStereoBuffer(512);
        tb.fillSine(440.0f, 44100.0f, 0.6f);
        dut.process(tb.view());
        ref.process(tb.view());

        for (int ch = 0; ch < 2; ++ch)
        {
            const float dp = std::fabs(dut.getPeakLevel(ch) - ref.getPeakLevel(ch));
            const float dr = std::fabs(dut.getRmsLevel(ch) - ref.getRmsLevel(ch));
            if (!std::isfinite(dp) || !std::isfinite(dr))
                ++badReads;
            else
                maxDiff = std::max(maxDiff, std::max(dp, dr));
        }
    }
    EXPECT_EQ(badReads, 0);
    EXPECT_EQ(maxDiff, 0.0f);

    // Getters stay honest after the rejected values.
    EXPECT_NEAR(dut.getAttackMs(), 5.0f, 1e-6f);
    EXPECT_NEAR(dut.getReleaseMs(), 90.0f, 1e-6f);
    EXPECT_NEAR(dut.getRmsWindowMs(), 150.0f, 1e-6f);

    // A NaN sample in the SIGNAL must not stick: the meter sanitizes at
    // publish time and recovers on the next block (old: NaN forever).
    LevelFollower<float> lf;
    lf.prepare(defaultSpec());
    lf.setRmsWindowMs(30.0f);
    auto tb = makeStereoBuffer(512);
    tb.fillSine(440.0f, 44100.0f, 0.25f);
    for (int b = 0; b < 5; ++b) lf.process(tb.view());
    tb.ch(0)[100] = kNan;
    lf.process(tb.view());
    tb.fillSine(440.0f, 44100.0f, 0.25f);
    for (int b = 0; b < 40; ++b) lf.process(tb.view());
    EXPECT_TRUE(std::isfinite(lf.getPeakLevel(0)));
    EXPECT_NEAR(lf.getPeakLevel(0), 0.25f, 0.02f);
    EXPECT_NEAR(lf.getRmsLevelDb(0), -15.05f, 0.5f); // 0.25 sine: 10*log10(A^2/2)

    // Out-of-range channel reads are safe and floored.
    EXPECT_EQ(lf.getPeakLevel(-1), 0.0f);
    EXPECT_EQ(lf.getPeakLevel(99), 0.0f);
    EXPECT_NEAR(lf.getRmsLevelDb(99), -100.0f, 1e-6f);
}

// ============================================================================
// SpectrumAnalyzer
// ============================================================================

DSPARK_TEST(SpectrumAnalyzer_sine_peak_at_frequency)
{
    SpectrumAnalyzer<float> sa;
    sa.prepare(44100.0, 2048);

    // Push enough samples for at least one FFT frame
    constexpr int N = 4096;
    std::vector<float> buf(N);
    generateSine(buf.data(), N, 1000.0f, 44100.0f);

    sa.pushSamples(buf.data(), N);

    // Find the peak bin
    const float* spectrum = sa.getMagnitudesDb();
    int numBins = 2048 / 2 + 1;

    int peakBin = 0;
    float peakVal = -200.0f;
    for (int i = 1; i < numBins; ++i)
    {
        if (spectrum[i] > peakVal)
        {
            peakVal = spectrum[i];
            peakBin = i;
        }
    }

    float peakFreq = sa.binToFrequency(peakBin);
    EXPECT_NEAR(peakFreq, 1000.0f, 50.0f); // Within ~50 Hz (bin resolution)
}

DSPARK_TEST(SpectrumAnalyzer_silence_floor)
{
    SpectrumAnalyzer<float> sa;
    sa.prepare(44100.0, 1024);
    sa.setFloorDb(-100.0f);

    std::vector<float> buf(2048, 0.0f);
    sa.pushSamples(buf.data(), 2048);

    const float* spectrum = sa.getMagnitudesDb();
    // All bins should be at or near floor
    for (int i = 0; i < 513; ++i)
        EXPECT_LT(spectrum[i], -80.0f);
}

DSPARK_TEST(SpectrumAnalyzer_amplitude_is_calibrated)
{
    // Single-sided scaling contract: a bin-exact sine of amplitude A reads
    // 20*log10(A) with smoothing off. Never pinned by any test before.
    SpectrumAnalyzer<float> sa;
    sa.prepare(48000.0, 2048, SpectrumAnalyzer<float>::WindowType::Hann);
    sa.setSmoothing(0.0f);

    // bin 100 at 48k/2048 = 2343.75 Hz, amplitude 0.5 -> -6.02 dB
    std::vector<float> x(2048);
    double ph = 0.0;
    const double w = 2.0 * 3.141592653589793 * 100.0 / 2048.0;
    for (int b = 0; b < 4; ++b)
    {
        for (auto& s : x) { s = 0.5f * static_cast<float>(std::sin(ph)); ph += w; }
        sa.pushSamples(x.data(), 2048);
    }
    EXPECT_NEAR(sa.getMagnitudesDb()[100], -6.02f, 0.2f);

    // Peak hold: rises to the signal, then decays at the configured dB/s.
    sa.setPeakHoldEnabled(true);
    sa.setPeakDecay(60.0f);
    for (int b = 0; b < 2; ++b)
    {
        for (auto& s : x) { s = 0.5f * static_cast<float>(std::sin(ph)); ph += w; }
        sa.pushSamples(x.data(), 2048);
    }
    const float peakAtSignal = sa.getPeakHoldDb()[100];
    EXPECT_NEAR(peakAtSignal, -6.02f, 0.3f);

    std::fill(x.begin(), x.end(), 0.0f);
    for (int b = 0; b < 4; ++b) sa.pushSamples(x.data(), 2048); // 8 frames of silence
    // 8 hops * 1024 / 48000 s * 60 dB/s = 10.24 dB of decay
    const float peakAfter = sa.getPeakHoldDb()[100];
    EXPECT_NEAR(peakAtSignal - peakAfter, 10.24f, 1.0f);
}

DSPARK_TEST(SpectrumAnalyzer_invalid_inputs_are_ignored)
{
    const float kNan = std::numeric_limits<float>::quiet_NaN();

    // pushSamples before prepare: the old header dereferenced a null FFT and
    // wrote into an empty ring (access violation measured); now a clean no-op
    // with honest getters.
    {
        SpectrumAnalyzer<float> sa;
        std::vector<float> x(256, 0.25f);
        sa.pushSamples(x.data(), 256);
        EXPECT_EQ(sa.getNumBins(), 0);
        EXPECT_EQ(sa.getFFTSize(), 0);
        EXPECT_NEAR(sa.binToFrequency(10), 0.0f, 1e-9f);
    }

    // Wild window enum: the old header left the window all zeros and the
    // analyser reported the floor forever on live signal; now falls back to
    // Hann and stays alive.
    {
        SpectrumAnalyzer<float> sa;
        sa.prepare(48000.0, 2048, static_cast<SpectrumAnalyzer<float>::WindowType>(99));
        EXPECT_TRUE(sa.getWindowType() == SpectrumAnalyzer<float>::WindowType::Hann);
        sa.setSmoothing(0.0f);
        std::vector<float> x(2048);
        double ph = 0.0;
        const double w = 2.0 * 3.141592653589793 * 100.0 / 2048.0;
        for (int b = 0; b < 4; ++b)
        {
            for (auto& s : x) { s = 0.5f * static_cast<float>(std::sin(ph)); ph += w; }
            sa.pushSamples(x.data(), 2048);
        }
        EXPECT_GT(sa.getMagnitudesDb()[100], -12.0f); // old: -100 (dead)
    }

    // Twin: NaN setters and an invalid re-prepare mid-stream must leave the
    // DUT bit-identical to an untouched reference. The old header parked NaN
    // in the smoothing one-pole (spectrum stuck at the floor for good) and
    // published NaN dB through a NaN floor.
    SpectrumAnalyzer<float> dut, ref;
    dut.prepare(48000.0, 1024); ref.prepare(48000.0, 1024);
    dut.setSmoothing(0.6f);       ref.setSmoothing(0.6f);
    dut.setPeakHoldEnabled(true); ref.setPeakHoldEnabled(true);
    dut.setPeakDecay(20.0f);      ref.setPeakDecay(20.0f);

    std::vector<float> x(512);
    double ph = 0.0;
    const double w = 2.0 * 3.141592653589793 * 50.0 / 1024.0;
    float maxDiff = 0.0f;
    int badReads = 0;
    for (int b = 0; b < 20; ++b)
    {
        if (b == 4)
        {
            dut.setSmoothing(kNan);
            dut.setPeakDecay(kNan);
            dut.setFloorDb(kNan);
        }
        if (b == 8)
            dut.prepare(std::numeric_limits<double>::quiet_NaN(), 1024);

        for (auto& s : x) { s = 0.4f * static_cast<float>(std::sin(ph)); ph += w; }
        dut.pushSamples(x.data(), 512);
        ref.pushSamples(x.data(), 512);

        const float* md = dut.getMagnitudesDb();
        const float* mr = ref.getMagnitudesDb();
        const float* pd = dut.getPeakHoldDb();
        const float* pr = ref.getPeakHoldDb();
        for (int k = 0; k < ref.getNumBins(); ++k)
        {
            const float d1 = std::fabs(md[k] - mr[k]);
            const float d2 = std::fabs(pd[k] - pr[k]);
            if (!std::isfinite(d1) || !std::isfinite(d2))
                ++badReads;
            else
                maxDiff = std::max(maxDiff, std::max(d1, d2));
        }
    }
    EXPECT_EQ(badReads, 0);
    EXPECT_EQ(maxDiff, 0.0f);

    // Getters stay honest after the rejected values.
    EXPECT_NEAR(dut.getSmoothing(), 0.6f, 1e-6f);
    EXPECT_NEAR(dut.getPeakDecay(), 20.0f, 1e-6f);
    EXPECT_NEAR(dut.getFloorDb(), -100.0f, 1e-6f);
    EXPECT_TRUE(dut.isPeakHoldEnabled());
    EXPECT_NEAR(dut.binToFrequency(50), 2343.75f, 0.01f); // rate survived the NaN prepare
}

// ============================================================================
// LoudnessMeter
// ============================================================================

DSPARK_TEST(LoudnessMeter_sine_997Hz_reference)
{
    LoudnessMeter<float> meter;
    meter.prepare(48000.0, 2);

    // Generate several seconds of 997 Hz sine at -23 dBFS (EBU reference level)
    float amplitude = decibelsToGain(-23.0f); // For LUFS, this is approximate
    constexpr int blockSize = 4800; // 100ms at 48kHz
    constexpr int numBlocks = 40;   // 4 seconds

    AudioBuffer<float> buf;
    buf.resize(2, blockSize);

    for (int b = 0; b < numBlocks; ++b)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            float sample = amplitude * std::sin(twoPi<float> * 997.0f *
                           static_cast<float>(b * blockSize + i) / 48000.0f);
            buf.getChannel(0)[i] = sample;
            buf.getChannel(1)[i] = sample;
        }
        meter.processBlock(std::as_const(buf).toView());
    }

    float lufs = meter.getIntegratedLUFS();
    // Should be approximately -23 LUFS (within +/-2 LU due to K-weighting)
    EXPECT_GT(lufs, -27.0f);
    EXPECT_LT(lufs, -19.0f);
}

DSPARK_TEST(LoudnessMeter_invalid_inputs_are_ignored)
{
    const double kPi = 3.141592653589793;
    auto fill997 = [&](std::vector<float>& L, std::vector<float>& R, double& ph, double db)
    {
        const double a = std::pow(10.0, db / 20.0);
        const double w = 2.0 * kPi * 997.0 / 48000.0;
        for (size_t i = 0; i < L.size(); ++i)
        {
            const float s = static_cast<float>(a * std::sin(ph));
            L[i] = s; R[i] = s; ph += w;
        }
    };
    std::vector<float> L(480), R(480);

    // (a) prepare with a NaN rate on a hot meter: the old gate was NaN-blind
    // (NaN <= 0 is false) - K coefficients went NaN, blockSamples_ took a UB
    // cast, reset() wiped the program history and momentary published NaN.
    // Must be a conservative no-op: bit-identical to an untouched twin.
    {
        LoudnessMeter<float> dut, ref;
        dut.prepare(48000.0, 2); ref.prepare(48000.0, 2);
        double ph = 0.0;
        for (int b = 0; b < 200; ++b)
        {
            fill997(L, R, ph, -23.0);
            dut.process(L.data(), R.data(), 480);
            ref.process(L.data(), R.data(), 480);
        }
        dut.prepare(std::numeric_limits<double>::quiet_NaN(), 2);
        AudioSpec bad; bad.sampleRate = -48000.0; bad.maxBlockSize = 512; bad.numChannels = 2;
        dut.prepare(bad);
        for (int b = 0; b < 200; ++b)
        {
            fill997(L, R, ph, -23.0);
            dut.process(L.data(), R.data(), 480);
            ref.process(L.data(), R.data(), 480);
        }
        EXPECT_TRUE(std::isfinite(dut.getMomentaryLUFS()));
        EXPECT_EQ(dut.getMomentaryLUFS(), ref.getMomentaryLUFS());
        EXPECT_EQ(dut.getIntegratedLUFS(), ref.getIntegratedLUFS());
        EXPECT_EQ(dut.getTruePeakDb(), ref.getTruePeakDb());
    }

    // (b) one NaN sample in the SIGNAL: the old K-filter recursion never
    // drained it (momentary NaN forever, integrated histogram frozen). The
    // meter now drops the poisoned 100 ms block, clears the filter states
    // and keeps measuring: after a level change it must track the new level.
    {
        LoudnessMeter<float> m;
        m.prepare(48000.0, 2);
        double ph = 0.0;
        for (int b = 0; b < 100; ++b) { fill997(L, R, ph, -23.0); m.process(L.data(), R.data(), 480); }
        fill997(L, R, ph, -23.0);
        L[100] = std::numeric_limits<float>::quiet_NaN();
        m.process(L.data(), R.data(), 480);
        for (int b = 0; b < 200; ++b) { fill997(L, R, ph, -14.0); m.process(L.data(), R.data(), 480); }
        EXPECT_TRUE(std::isfinite(m.getMomentaryLUFS()));   // old: NaN forever
        EXPECT_NEAR(m.getMomentaryLUFS(), -14.0f, 0.5f);
        EXPECT_GT(m.getIntegratedLUFS(), -18.0f);           // old: frozen at -23
        EXPECT_LT(m.getIntegratedLUFS(), -13.0f);
    }

    // (c) process before prepare: the old header measured WITHOUT the
    // K-weighting stages (silently ~0.7 LU miscalibrated); now a clean no-op.
    {
        LoudnessMeter<float> m;
        double ph = 0.0;
        for (int b = 0; b < 100; ++b) { fill997(L, R, ph, -23.0); m.process(L.data(), R.data(), 480); }
        EXPECT_NEAR(m.getIntegratedLUFS(), -100.0f, 1e-6f);
        EXPECT_NEAR(m.getMomentaryLUFS(), -100.0f, 1e-6f);
    }
}

DSPARK_TEST(LoudnessMeter_silence)
{
    LoudnessMeter<float> meter;
    meter.prepare(44100.0, 2);

    auto tb = makeStereoBuffer(4410);
    tb.fillSilence();
    meter.processBlock(std::as_const(tb.buffer).toView());

    float lufs = meter.getMomentaryLUFS();
    EXPECT_LT(lufs, -70.0f); // Should be very low
}

DSPARK_TEST(LoudnessMeter_louder_signal_higher_LUFS)
{
    LoudnessMeter<float> m1, m2;
    m1.prepare(48000.0, 2);
    m2.prepare(48000.0, 2);

    constexpr int blockSize = 4800;
    constexpr int numBlocks = 40;

    AudioBuffer<float> buf;
    buf.resize(2, blockSize);

    for (int b = 0; b < numBlocks; ++b)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            float t = static_cast<float>(b * blockSize + i) / 48000.0f;
            float s = std::sin(twoPi<float> * 997.0f * t);
            buf.getChannel(0)[i] = s * 0.1f;  // Quiet
            buf.getChannel(1)[i] = s * 0.1f;
        }
        m1.processBlock(std::as_const(buf).toView());

        for (int i = 0; i < blockSize; ++i)
        {
            float t = static_cast<float>(b * blockSize + i) / 48000.0f;
            float s = std::sin(twoPi<float> * 997.0f * t);
            buf.getChannel(0)[i] = s * 0.5f;  // Loud
            buf.getChannel(1)[i] = s * 0.5f;
        }
        m2.processBlock(std::as_const(buf).toView());
    }

    float lufs1 = m1.getIntegratedLUFS();
    float lufs2 = m2.getIntegratedLUFS();
    EXPECT_GT(lufs2, lufs1); // Louder signal = higher LUFS
}

// Pin: the true-peak max-hold must NOT be permanently poisoned by a single
// non-finite (Inf/NaN) input sample. The K-filter / histogram path
// already drops a poisoned block and self-recovers; before the fix std::max()
// latched +Inf into truePeakMax_ (and the raw |sample| leaked it through the
// detector ring), so getTruePeakDb() returned +Inf dBTP forever. Now only
// FINITE per-sample estimates fold into the max-hold, so the reading recovers.
DSPARK_TEST(LoudnessMeter_true_peak_recovers_after_non_finite_sample)
{
    LoudnessMeter<float> meter;
    meter.prepare(48000.0, 2);

    std::vector<float> L(480), R(480);
    double ph = 0.0;
    const double w = twoPi<double> * 997.0 / 48000.0;
    auto fill = [&](double amp)
    {
        for (size_t i = 0; i < L.size(); ++i)
        { const float s = static_cast<float>(amp * std::sin(ph)); L[i] = s; R[i] = s; ph += w; }
    };

    // Warm up on a clean -6 dBFS tone: finite, roughly -6 dBTP.
    for (int b = 0; b < 60; ++b) { fill(std::pow(10.0, -6.0 / 20.0)); meter.process(L.data(), R.data(), 480); }
    EXPECT_TRUE(std::isfinite(meter.getTruePeakDb()));

    // Inject ONE block carrying a non-finite sample on each channel.
    fill(std::pow(10.0, -6.0 / 20.0));
    L[100] = std::numeric_limits<float>::infinity();
    R[240] = std::numeric_limits<float>::quiet_NaN();
    meter.process(L.data(), R.data(), 480);

    // Keep metering clean signal: the true-peak readout must be FINITE again
    // (old header: stuck at +Inf dBTP forever).
    for (int b = 0; b < 60; ++b) { fill(std::pow(10.0, -6.0 / 20.0)); meter.process(L.data(), R.data(), 480); }
    const float tp = meter.getTruePeakDb();
    EXPECT_TRUE(std::isfinite(tp));   // old: +Inf
    EXPECT_LT(tp, 0.0f);              // a -6 dBFS tone tops out well under 0 dBTP
    EXPECT_GT(tp, -12.0f);
}

// ============================================================================
// Goertzel
// ============================================================================

DSPARK_TEST(Goertzel_detects_target_frequency)
{
    Goertzel<float> g;
    g.prepare(44100.0, 440.0, 4096);

    std::vector<float> buf(4096);
    generateSine(buf.data(), 4096, 440.0f, 44100.0f);

    g.processBlock(buf.data(), 4096);

    float mag = g.getMagnitude();
    EXPECT_GT(mag, 0.5f); // Should detect the frequency strongly
}

DSPARK_TEST(Goertzel_rejects_other_frequency)
{
    Goertzel<float> g;
    g.prepare(44100.0, 440.0, 4096);

    // Feed 880 Hz - should not trigger 440 Hz detector
    std::vector<float> buf(4096);
    generateSine(buf.data(), 4096, 880.0f, 44100.0f);

    g.processBlock(buf.data(), 4096);

    float mag440 = g.getMagnitude();

    // Now measure at actual frequency
    Goertzel<float> g2;
    g2.prepare(44100.0, 880.0, 4096);
    g2.processBlock(buf.data(), 4096);
    float mag880 = g2.getMagnitude();

    EXPECT_GT(mag880, mag440 * 5.0f); // Target freq should be much stronger
}

DSPARK_TEST(Goertzel_silence)
{
    Goertzel<float> g;
    g.prepare(44100.0, 1000.0, 1024);

    std::vector<float> buf(1024, 0.0f);
    g.processBlock(buf.data(), 1024);

    EXPECT_NEAR(g.getMagnitude(), 0.0f, 1e-6f);
}

DSPARK_TEST(Goertzel_streaming_matches_block)
{
    Goertzel<float> gBlock, gStream;
    gBlock.prepare(44100.0, 440.0, 1024);
    gStream.prepare(44100.0, 440.0, 1024);

    std::vector<float> buf(1024);
    generateSine(buf.data(), 1024, 440.0f, 44100.0f);

    // Block mode
    gBlock.processBlock(buf.data(), 1024);

    // Streaming mode
    for (int i = 0; i < 1024; ++i)
        gStream.pushSample(buf[i]);

    EXPECT_NEAR(gBlock.getMagnitude(), gStream.getMagnitude(), 0.01f);
}

DSPARK_TEST(Goertzel_nyquist_and_dc_normalization)
{
    // Exact Nyquist has no mirrored bin: the old 2/N normalization read an
    // alternating full-band signal 2x (+6 dB) high. Verified in double where
    // the recursion is numerically exact (at DC/Nyquist the resonator sits
    // on a double pole and its state grows O(N^2), so float needs short
    // windows there - documented in the header).
    Goertzel<double> g;
    g.prepare(48000.0, 24000.0, 4800);
    std::vector<double> x(4800);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (i & 1) ? -0.5 : 0.5;
    g.processBlock(x.data(), 4800);
    EXPECT_NEAR(g.getMagnitude(), 0.5, 1e-9); // old: 1.0

    Goertzel<float> gf;
    gf.prepare(48000.0, 24000.0, 480);
    std::vector<float> xf(480);
    for (size_t i = 0; i < xf.size(); ++i) xf[i] = (i & 1) ? -0.5f : 0.5f;
    gf.processBlock(xf.data(), 480);
    EXPECT_NEAR(gf.getMagnitude(), 0.5f, 0.01f); // old: 1.0

    // DC keeps its 1/N (regression pin) and the new getters are honest.
    Goertzel<float> gd;
    gd.prepare(48000.0, 0.0, 480);
    std::vector<float> dc(480, 0.25f);
    gd.processBlock(dc.data(), 480);
    EXPECT_NEAR(gd.getMagnitude(), 0.25f, 0.01f);
    EXPECT_EQ(gd.getBlockSize(), 480);
    EXPECT_NEAR(static_cast<float>(gd.getSampleRate()), 48000.0f, 1e-3f);
}

// ============================================================================
// PitchDetector
// ============================================================================

DSPARK_TEST(PitchDetector_invalid_inputs_are_ignored)
{
    const double kPi = 3.141592653589793;
    auto fill = [&](std::vector<float>& x, double& ph, double freq)
    {
        const double w = 2.0 * kPi * freq / 48000.0;
        for (auto& s : x) { s = static_cast<float>(0.5 * std::sin(ph)); ph += w; }
    };
    std::vector<float> x(512);

    // pushSamples before prepare: the old header wrote into an empty vector
    // and dereferenced a null FFT (access violation measured); now a no-op.
    {
        PitchDetector<float> pd;
        pd.pushSamples(std::span<const float>(x.data(), x.size()));
        EXPECT_NEAR(pd.getFrequencyHz(), 0.0f, 1e-9f);
        EXPECT_EQ(pd.getMidiNote(), -1);
    }

    // setThreshold(NaN): clamp(NaN) parked NaN and every comparison went
    // false - the detector reported unvoiced FOREVER on live signal. Now
    // ignored, with an honest getter (the API was write-only).
    {
        PitchDetector<float> pd;
        pd.prepare(48000.0, 2048, 512);
        double ph = 0.0;
        for (int b = 0; b < 8; ++b) { fill(x, ph, 440.0); pd.pushSamples(std::span<const float>(x.data(), x.size())); }
        pd.setThreshold(std::numeric_limits<float>::quiet_NaN());
        for (int b = 0; b < 8; ++b) { fill(x, ph, 440.0); pd.pushSamples(std::span<const float>(x.data(), x.size())); }
        EXPECT_NEAR(pd.getFrequencyHz(), 440.0f, 2.0f); // old: 0.0 (dead)
        EXPECT_NEAR(pd.getThreshold(), 0.10f, 1e-6f);
    }

    // prepare with a NaN rate: the old header stored it and published NaN
    // frequency; now a conservative no-op keeping the configuration.
    {
        PitchDetector<float> pd;
        pd.prepare(48000.0, 2048, 512);
        double ph = 0.0;
        for (int b = 0; b < 8; ++b) { fill(x, ph, 440.0); pd.pushSamples(std::span<const float>(x.data(), x.size())); }
        pd.prepare(std::numeric_limits<double>::quiet_NaN(), 2048, 512);
        for (int b = 0; b < 8; ++b) { fill(x, ph, 440.0); pd.pushSamples(std::span<const float>(x.data(), x.size())); }
        EXPECT_TRUE(std::isfinite(pd.getFrequencyHz()));
        EXPECT_NEAR(pd.getFrequencyHz(), 440.0f, 2.0f);
    }

    // A NaN sample in the signal: the old CMND filled with zeros and the
    // detector published fs/2 = 24000 Hz at confidence 1.0 - a FAKE
    // detection at maximum confidence. Now it reads unvoiced while the bad
    // sample is in the window and self-recovers once it flushes out.
    {
        PitchDetector<float> pd;
        pd.prepare(48000.0, 2048, 512);
        double ph = 0.0;
        for (int b = 0; b < 8; ++b) { fill(x, ph, 440.0); pd.pushSamples(std::span<const float>(x.data(), x.size())); }
        fill(x, ph, 440.0);
        x[100] = std::numeric_limits<float>::quiet_NaN();
        pd.pushSamples(std::span<const float>(x.data(), x.size()));
        EXPECT_NEAR(pd.getFrequencyHz(), 0.0f, 1e-9f);  // old: 24000
        EXPECT_NEAR(pd.getConfidence(), 0.0f, 1e-9f);   // old: 1.0
        for (int b = 0; b < 10; ++b) { fill(x, ph, 440.0); pd.pushSamples(std::span<const float>(x.data(), x.size())); }
        EXPECT_NEAR(pd.getFrequencyHz(), 440.0f, 2.0f);
    }
}

DSPARK_TEST(PitchDetector_detects_440)
{
    PitchDetector<float> pd;
    pd.prepare(44100.0, 2048);

    // Generate 440 Hz sine and push enough for detection
    std::vector<float> buf(4096);
    generateSine(buf.data(), 4096, 440.0f, 44100.0f);

    pd.pushSamples(buf);

    float freq = pd.getFrequencyHz();
    EXPECT_NEAR(freq, 440.0f, 5.0f); // Within 5 Hz
    EXPECT_GT(pd.getConfidence(), 0.7f);
}

DSPARK_TEST(PitchDetector_midi_note_A4)
{
    PitchDetector<float> pd;
    pd.prepare(44100.0, 2048);

    std::vector<float> buf(4096);
    generateSine(buf.data(), 4096, 440.0f, 44100.0f);
    pd.pushSamples(buf);

    EXPECT_EQ(pd.getMidiNote(), 69); // A4 = MIDI 69
    EXPECT_NEAR(pd.getCentsOffset(), 0.0f, 15.0f);
}

DSPARK_TEST(PitchDetector_low_freq)
{
    PitchDetector<float> pd;
    pd.prepare(44100.0, 4096); // Larger window for low frequency

    std::vector<float> buf(8192);
    generateSine(buf.data(), 8192, 100.0f, 44100.0f);
    pd.pushSamples(buf);

    EXPECT_NEAR(pd.getFrequencyHz(), 100.0f, 3.0f);
}

DSPARK_TEST(PitchDetector_silence_no_pitch)
{
    PitchDetector<float> pd;
    pd.prepare(44100.0, 2048);

    std::vector<float> buf(4096, 0.0f);
    pd.pushSamples(buf);

    EXPECT_NEAR(pd.getFrequencyHz(), 0.0f, 0.1f);
    EXPECT_EQ(pd.getMidiNote(), -1);
}


// ============================================================================
// PitchFollower
// ============================================================================

DSPARK_TEST(PitchFollower_tracks_glide_without_jumps)
{
    // A2->A3 glide with 5.5 Hz vibrato and periodic noise consonants: the
    // smoothed output must never jump more than 1 semitone between blocks
    // (the plan acceptance criterion) and must land on the final tone.
    PitchFollower<float> pf;
    pf.prepare(spec(48000.0, 512, 2));
    pf.setRange(60.0f, 1000.0f);
    pf.setGlide(60.0f);

    auto buf = makeStereoBuffer(512);
    const int totalBlocks = static_cast<int>(6.0 * 48000.0 / 512.0);
    double phase = 0.0;
    uint32_t rng = 0x2026u;
    float prevSt = -999.0f;
    float maxJump = 0.0f;
    float finalHz = 0.0f;
    double finalF0 = 0.0;

    for (int b = 0; b < totalBlocks; ++b)
    {
        const double t = b * 512.0 / 48000.0;
        const bool consonant = std::fmod(t, 0.9) < 0.07;
        const double glide = std::min(t / 4.0, 1.0);
        const double f0 = 110.0 * std::pow(2.0, glide)
                        * (1.0 + 0.02 * std::sin(2.0 * 3.14159265358979 * 5.0 * t));

        for (int i = 0; i < 512; ++i)
        {
            phase += f0 / 48000.0;
            if (phase >= 1.0) phase -= 1.0;
            float v = 0.4f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * phase));
            if (consonant)
            {
                rng = rng * 1664525u + 1013904223u;
                v = 0.1f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
            }
            buf.ch(0)[i] = v;
            buf.ch(1)[i] = v;
        }
        pf.processBlock(AudioBufferView<const float>(buf.view()));

        const float hz = pf.getSmoothedHz();
        if (hz > 0.0f)
        {
            const float st = 12.0f * std::log2(hz / 440.0f);
            if (prevSt > -900.0f)
                maxJump = std::max(maxJump, std::abs(st - prevSt));
            prevSt = st;
        }
        if (b == totalBlocks - 1) { finalHz = hz; finalF0 = f0; }
    }

    EXPECT_GT(prevSt, -900.0f);          // it locked at some point
    EXPECT_LT(maxJump, 1.0f);            // no jump > 1 semitone between blocks
    EXPECT_TRUE(pf.isTracking());
    const double errSt = 12.0 * std::log2(static_cast<double>(finalHz) / finalF0);
    EXPECT_NEAR(errSt, 0.0, 0.6);        // lands on the tone (vibrato is +/-0.34 st)
}

DSPARK_TEST(PitchFollower_freezes_during_silence)
{
    PitchFollower<float> pf;
    pf.prepare(spec(48000.0, 512, 1));

    // Phase-continuous sine across blocks (per-block fillSine restarts the
    // phase, creating a 93.75 Hz periodicity the detector would rightly lock).
    auto buf = makeMonoBuffer(512);
    double phase = 0.0;
    for (int b = 0; b < 100; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            buf.ch(0)[i] = 0.5f * static_cast<float>(std::sin(2.0 * 3.14159265358979 * phase));
            phase += 220.0 / 48000.0;
            if (phase >= 1.0) phase -= 1.0;
        }
        pf.processBlock(AudioBufferView<const float>(buf.view()));
    }
    const float locked = pf.getSmoothedHz();
    EXPECT_NEAR(locked, 220.0f, 2.0f);
    EXPECT_TRUE(pf.isTracking());

    for (int b = 0; b < 100; ++b)        // ~1 s of silence
    {
        buf.fillSilence();
        pf.processBlock(AudioBufferView<const float>(buf.view()));
    }
    EXPECT_NEAR(pf.getSmoothedHz(), locked, 0.01f);   // frozen, not decayed
    EXPECT_FALSE(pf.isTracking());                     // but no longer tracking
}

DSPARK_TEST(PitchFollower_range_gate_rejects_out_of_band)
{
    PitchFollower<float> pf;
    pf.prepare(spec(48000.0, 512, 1));
    pf.setRange(100.0f, 800.0f);

    auto buf = makeMonoBuffer(512);
    for (int b = 0; b < 100; ++b)
    {
        buf.fillSine(50.0f, 48000.0f, 0.5f);   // below the accepted range
        pf.processBlock(AudioBufferView<const float>(buf.view()));
    }
    EXPECT_EQ(pf.getSmoothedHz(), 0.0f);       // never locked
    EXPECT_FALSE(pf.isTracking());
}

DSPARK_TEST(PitchFollower_invalid_inputs_are_ignored)
{
    // Phase-continuous sine (TestBuffer::fillSine restarts the phase every
    // block, which makes the whole 512-sample block the true signal period
    // and YIN honestly reports 93.75 Hz).
    auto fillCont = [](TestBuffer& b, double& ph, double freq)
    {
        const double w = 2.0 * 3.141592653589793 * freq / 48000.0;
        for (int i = 0; i < 512; ++i) { b.ch(0)[i] = static_cast<float>(0.5 * std::sin(ph)); ph += w; }
    };

    // (a) NaN parameters: the old max()/clamp() passed NaN through and the
    // validity gate closed FOREVER - the follower froze on the last pitch
    // (still reporting it) while the music moved on. Now ignored.
    {
        PitchFollower<float> pf;
        pf.prepare(spec(48000.0, 512, 1));
        pf.setGlide(0.0f);
        auto buf = makeMonoBuffer(512);
        double ph = 0.0;
        for (int b = 0; b < 40; ++b)
        {
            fillCont(buf, ph, 220.0);
            pf.processBlock(AudioBufferView<const float>(buf.view()));
        }
        pf.setRange(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN());
        pf.setConfidence(std::numeric_limits<float>::quiet_NaN());
        pf.setGlide(std::numeric_limits<float>::quiet_NaN());
        for (int b = 0; b < 60; ++b)
        {
            fillCont(buf, ph, 300.0); // +5.4 st: tracked directly
            pf.processBlock(AudioBufferView<const float>(buf.view()));
        }
        EXPECT_NEAR(pf.getSmoothedHz(), 300.0f, 3.0f); // old: stuck at 220
        EXPECT_TRUE(pf.isTracking());                  // old: false
        EXPECT_NEAR(pf.getMinHz(), 60.0f, 1e-6f);
        EXPECT_NEAR(pf.getMaxHz(), 1200.0f, 1e-6f);
        EXPECT_NEAR(pf.getConfidenceThreshold(), 0.85f, 1e-6f);
        EXPECT_NEAR(pf.getGlide(), 0.0f, 1e-6f);
    }

    // (b) prepare with an invalid spec on a hot tracker: the old NaN-blind
    // gate reset the tracking state and poisoned the wrapped detector - the
    // follower lost the pitch for good. Now a conservative no-op,
    // bit-identical to an untouched twin.
    {
        PitchFollower<float> pf, ref;
        pf.prepare(spec(48000.0, 512, 1)); ref.prepare(spec(48000.0, 512, 1));
        pf.setGlide(0.0f); ref.setGlide(0.0f);
        auto buf = makeMonoBuffer(512);
        double ph = 0.0;
        for (int b = 0; b < 40; ++b)
        {
            fillCont(buf, ph, 220.0);
            pf.processBlock(AudioBufferView<const float>(buf.view()));
            ref.processBlock(AudioBufferView<const float>(buf.view()));
        }
        AudioSpec bad = spec(48000.0, 512, 1);
        bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
        pf.prepare(bad);
        float maxDiff = 0.0f;
        for (int b = 0; b < 40; ++b)
        {
            fillCont(buf, ph, 220.0);
            pf.processBlock(AudioBufferView<const float>(buf.view()));
            ref.processBlock(AudioBufferView<const float>(buf.view()));
            maxDiff = std::max(maxDiff, std::fabs(pf.getSmoothedHz() - ref.getSmoothedHz()));
        }
        EXPECT_EQ(maxDiff, 0.0f);        // old: loses the pitch (diff 220)
        EXPECT_TRUE(pf.isTracking());
    }
}


// ============================================================================
// PhaseCorrelation
// ============================================================================

DSPARK_TEST(PhaseCorrelation_detects_mono_inverted_and_uncorrelated)
{
    auto runCase = [](auto fill) {
        PhaseCorrelation<float> pc;
        pc.prepare(spec(48000.0, 512, 2));
        auto buf = makeStereoBuffer(512);
        uint32_t rng = 99u;
        for (int b = 0; b < 100; ++b)
        {
            for (int i = 0; i < 512; ++i)
                fill(b * 512 + i, buf.ch(0)[i], buf.ch(1)[i], rng);
            pc.processBlock(AudioBufferView<const float>(buf.view()));
        }
        return pc.getCorrelation();
    };

    const float rMono = runCase([](int n, float& l, float& r, uint32_t&) {
        l = r = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(n) / 48000.0f);
    });
    EXPECT_GT(rMono, 0.99f);

    const float rInv = runCase([](int n, float& l, float& r, uint32_t&) {
        l = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f * static_cast<float>(n) / 48000.0f);
        r = -l;
    });
    EXPECT_LT(rInv, -0.99f);

    const float rUnc = runCase([](int, float& l, float& r, uint32_t& rng) {
        rng = rng * 1664525u + 1013904223u;
        l = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
        rng = rng * 1664525u + 1013904223u;
        r = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    });
    EXPECT_LT(std::abs(rUnc), 0.2f);
}

DSPARK_TEST(PhaseCorrelation_balance_and_gonio)
{
    PhaseCorrelation<float> pc;
    pc.prepare(spec(48000.0, 512, 2));
    auto buf = makeStereoBuffer(512);
    for (int b = 0; b < 100; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            buf.ch(0)[i] = 0.0f;
            buf.ch(1)[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f
                                           * static_cast<float>(b * 512 + i) / 48000.0f);
        }
        pc.processBlock(AudioBufferView<const float>(buf.view()));
    }
    EXPECT_GT(pc.getBalance(), 0.9f);

    PhaseCorrelation<float>::GonioPoint pts[64];
    const int got = pc.getGonioPoints(pts, 64);
    EXPECT_EQ(got, 64);
    bool any = false;
    for (int i = 0; i < got; ++i)
        any = any || std::abs(pts[i].mid) > 1e-6f || std::abs(pts[i].side) > 1e-6f;
    EXPECT_TRUE(any);
}

DSPARK_TEST(PhaseCorrelation_invalid_inputs_are_ignored)
{
    auto buf = makeStereoBuffer(512);
    auto fillCorr = [&buf](long& counter) {
        for (int i = 0; i < 512; ++i, ++counter)
        {
            const float s = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f
                                            * static_cast<float>(counter) / 48000.0f);
            buf.ch(0)[i] = s;
            buf.ch(1)[i] = 0.7f * s;
        }
    };

    // (a) Invalid re-prepares on a hot meter are conservative no-ops; a NaN
    //     window falls back to the default. Bit-identical to a twin.
    {
        PhaseCorrelation<float> pc;
        PhaseCorrelation<float> twin;
        pc.prepare(spec(48000.0, 512, 2));
        twin.prepare(spec(48000.0, 512, 2));
        long n = 0;
        for (int b = 0; b < 20; ++b)
        {
            fillCorr(n);
            pc.processBlock(AudioBufferView<const float>(buf.view()));
            twin.processBlock(AudioBufferView<const float>(buf.view()));
        }
        AudioSpec bad = spec(48000.0, 512, 2);
        bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
        pc.prepare(bad);                                     // NaN rate: no-op
        bad.sampleRate = std::numeric_limits<double>::infinity();
        pc.prepare(bad);                                     // inf passes isValid(): no-op
        pc.prepare(spec(48000.0, 512, 2), std::numeric_limits<double>::quiet_NaN());
        twin.prepare(spec(48000.0, 512, 2), 300.0);          // NaN window == default
        float maxDiff = 0.0f;
        for (int b = 0; b < 20; ++b)
        {
            fillCorr(n);
            pc.processBlock(AudioBufferView<const float>(buf.view()));
            twin.processBlock(AudioBufferView<const float>(buf.view()));
            maxDiff = std::max(maxDiff, std::abs(pc.getCorrelation() - twin.getCorrelation()));
            maxDiff = std::max(maxDiff, std::abs(pc.getBalance() - twin.getBalance()));
        }
        EXPECT_EQ(maxDiff, 0.0f);              // old: correlation dies at 0 forever
        EXPECT_GT(pc.getCorrelation(), 0.99f);
        EXPECT_TRUE(pc.getWindowMs() == 300.0);
    }

    // (b) One non-finite sample must not kill the meter for good, and the
    //     goniometer ring never serves non-finite points to the GUI.
    {
        PhaseCorrelation<float> pc;
        pc.prepare(spec(48000.0, 512, 2));
        long n = 0;
        for (int b = 0; b < 10; ++b)
        {
            fillCorr(n);
            pc.processBlock(AudioBufferView<const float>(buf.view()));
        }
        fillCorr(n);
        buf.ch(0)[100] = std::numeric_limits<float>::quiet_NaN();
        pc.processBlock(AudioBufferView<const float>(buf.view()));
        PhaseCorrelation<float>::GonioPoint pts[PhaseCorrelation<float>::kGonioSize];
        int got = pc.getGonioPoints(pts, PhaseCorrelation<float>::kGonioSize);
        int nBad = 0;
        for (int i = 0; i < got; ++i)
            if (!std::isfinite(pts[i].mid) || !std::isfinite(pts[i].side)) ++nBad;
        EXPECT_EQ(nBad, 0);                    // old: NaN point served to the display
        for (int b = 0; b < 30; ++b)
        {
            fillCorr(n);
            pc.processBlock(AudioBufferView<const float>(buf.view()));
        }
        EXPECT_GT(pc.getCorrelation(), 0.99f); // old: stuck at 0 forever
        EXPECT_EQ(pc.getGonioPoints(nullptr, 64), 0);   // old: null deref crash
    }
}

DSPARK_TEST(PhaseCorrelation_reset_clears_ring_and_mono_is_dual_mono)
{
    PhaseCorrelation<float> pc;
    pc.prepare(spec(48000.0, 512, 2), 30.0);
    auto st = makeStereoBuffer(512);
    long n = 0;
    for (int b = 0; b < 10; ++b)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            const float s = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f
                                            * static_cast<float>(n) / 48000.0f);
            st.ch(0)[i] = s;
            st.ch(1)[i] = -s;                  // side-heavy program
        }
        pc.processBlock(AudioBufferView<const float>(st.view()));
    }
    EXPECT_LT(pc.getCorrelation(), -0.99f);

    // reset() clears the goniometer ring for real (the documented contract).
    pc.reset();
    PhaseCorrelation<float>::GonioPoint pts[PhaseCorrelation<float>::kGonioSize];
    int got = pc.getGonioPoints(pts, PhaseCorrelation<float>::kGonioSize);
    float ringMax = 0.0f;
    for (int i = 0; i < got; ++i)
        ringMax = std::max({ ringMax, std::abs(pts[i].mid), std::abs(pts[i].side) });
    EXPECT_EQ(ringMax, 0.0f);                  // old: full of stale side energy

    // A mono buffer measures as dual mono: r = +1 with signal and the gonio
    // stays ALIVE, collapsed onto the mid axis (old: frozen stale cloud).
    auto mono = makeMonoBuffer(512);
    long m = 0;
    for (int b = 0; b < 20; ++b)
    {
        for (int i = 0; i < 512; ++i, ++m)
            mono.ch(0)[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f
                                            * static_cast<float>(m) / 48000.0f);
        pc.processBlock(AudioBufferView<const float>(mono.view()));
    }
    EXPECT_GT(pc.getCorrelation(), 0.99f);
    got = pc.getGonioPoints(pts, 64);
    float maxMid = 0.0f, maxSide = 0.0f;
    for (int i = 0; i < got; ++i)
    {
        maxMid = std::max(maxMid, std::abs(pts[i].mid));
        maxSide = std::max(maxSide, std::abs(pts[i].side));
    }
    EXPECT_GT(maxMid, 0.1f);
    EXPECT_LT(maxSide, 1e-6f);
}

// ============================================================================
// Cross-thread publication pins (concurrent, with a reachability argument)
// ============================================================================

// Shared probe signal for the SpectrumAnalyzer publication pins: two tones
// of EQUAL amplitude at 1500 Hz and 18750 Hz, which at 48 kHz land exactly
// on bins fftSize/32 and fftSize*25/64 for every power-of-two fftSize the
// analyser accepts. The phase index is reduced modulo 64 -- the tones'
// exact common period (2 and 25 whole cycles) -- so both stay on-bin
// however long the writer has been running. Reducing it is load-bearing,
// not cosmetic: with the raw index, static_cast<float>(n) stops being exact
// at n = 2^24 = 16777216 and the 18750 Hz argument, 12.5x larger than the
// 1500 Hz one, loses its low bits first, so the upper probe bin
// decorrelates while the lower one survives. The pins then report tens of
// dB of "tear" on a handoff that never tore, and their verdict becomes a
// function of how many frames the writer got through -- i.e. of machine
// speed and -O level. SpectrumAnalyzer_probe_signal_stays_on_bin_past_2p24
// below is the deterministic guard for exactly that.
static void fillProbeTones(float* dst, int numSamples, long long& n, float amp)
{
    const float w1 = twoPi<float> * 1500.0f / 48000.0f;
    const float w2 = twoPi<float> * 18750.0f / 48000.0f;
    for (int i = 0; i < numSamples; ++i, ++n)
    {
        const float ph = static_cast<float>(n & 63);
        dst[static_cast<size_t>(i)] = amp * (std::sin(w1 * ph) + std::sin(w2 * ph));
    }
}

// Pin for the SpectrumAnalyzer triple-buffer handoff (audio writer vs GUI
// reader), the largest un-oracled lock-free structure in the library before
// this pin. The writer publishes spectra whose two probe bins ALWAYS carry
// equal-amplitude tones (1500 Hz -> bin 32 and 18750 Hz -> bin 400 at
// fs 48 kHz, fftSize 1024), toggling the common amplitude by 24 dB every 8
// hops with smoothing off. Every coherent frame therefore has
// |dB[32] - dB[400]| of only a few dB (equal on-bin tones; the amplitude
// STEP's spectral splatter at 368 bins distance is >30 dB under either
// tone). A reader that ever observes a slot the writer is concurrently
// writing mixes a high-level bin 32 with a low-level bin 400 (or vice
// versa) and trips the 12 dB delta bound. That "every coherent frame" claim
// holds for any run length only because fillProbeTones() reduces the phase
// index; read its comment before touching the signal.
// Reachability argument: breaking the ownership handoff -- a mutated
// computeSpectrum() that publishes the finished slot but KEEPS
// writing into it (no slot adoption) -- makes this exact assertion fail with
// tens of thousands of violations per run, so the pin demonstrably reaches
// the structure it guards rather than merely running beside it. Liveness
// is guaranteed by construction: the reader keeps polling until the writer
// has published at least 64 frames (hard-capped so a starved writer fails
// loudly instead of hanging), and the fresh-adoption floor proves the
// handoff actually cycled.
DSPARK_TEST(SpectrumAnalyzer_triple_buffer_concurrent_readout_is_tear_free)
{
    auto sa = std::make_unique<SpectrumAnalyzer<float>>();
    sa->prepare(48000.0, 1024);
    sa->setSmoothing(0.0f);
    sa->setPeakHoldEnabled(false);
    sa->setFloorDb(-200.0f);

    std::atomic<bool> stop{ false };
    std::atomic<int> frames{ 0 };

    std::thread audio([&] {
        std::vector<float> block(512);
        long long n = 0;
        int hop = 0;
        while (!stop.load(std::memory_order_relaxed))
        {
            const float amp = ((hop / 8) & 1) ? 0.025f : 0.4f; // 24 dB toggle
            fillProbeTones(block.data(), 512, n, amp);
            sa->pushSamples(block.data(), 512);
            ++hop;
            frames.fetch_add(1, std::memory_order_relaxed); // 1 hop = 1 frame
        }
    });

    int violations = 0;
    int fresh = 0;
    long long reads = 0;
    const long long kMaxReads = 50000000; // hard cap: never hang, fail loudly
    while ((frames.load(std::memory_order_relaxed) < 64 || reads < 300000)
           && reads < kMaxReads)
    {
        ++reads;
        if (sa->isNewDataReady()) ++fresh;
        const float* m = sa->getMagnitudesDb();
        const float d = m[32] - m[400];
        if (std::abs(d) > 12.0f) ++violations;
    }

    stop.store(true, std::memory_order_relaxed);
    audio.join();

    EXPECT_EQ(violations, 0);
    EXPECT_GT(frames.load(std::memory_order_relaxed), 63); // writer liveness
    EXPECT_GT(fresh, 16);                                  // adoption liveness
}

// Deterministic guard for the probe signal the two publication pins above
// and below rely on. NO threads: one writer, one reader, same call, so a
// failure here can only be the signal or the analyser, never a handoff.
// It drives the sample index across 2^24 = 16777216 -- the last integer a
// float represents exactly -- which is where the raw-index form of this
// signal silently stops being two equal on-bin tones. Starting the index
// just below the cliff instead of counting up to it keeps the pin at a few
// hundred frames.
// Reachability argument: with fillProbeTones()'s `n & 63` replaced by the
// raw `n` this pin fails unconditionally -- measured on this tree,
// single-threaded at -O2, all 204 checked frames violate and |d| reaches
// 27.22 dB against a bound of 1 dB -- which is exactly the reading the
// concurrent pin above reports as a torn snapshot when it is really just
// this. With the reduction in place every frame gives |d| = 0.00 dB.
DSPARK_TEST(SpectrumAnalyzer_probe_signal_stays_on_bin_past_2p24)
{
    auto sa = std::make_unique<SpectrumAnalyzer<float>>();
    sa->prepare(48000.0, 1024);
    sa->setSmoothing(0.0f);
    sa->setPeakHoldEnabled(false);
    sa->setFloorDb(-200.0f);

    std::vector<float> block(512);
    // Start 8 hops below the 2^24 cliff and run 200 hops past it.
    long long n = (1LL << 24) - 8LL * 512LL;
    int violations = 0;
    float worst = 0.0f;
    int checked = 0;

    for (int hop = 0; hop < 208; ++hop)
    {
        const float amp = ((hop / 8) & 1) ? 0.025f : 0.4f; // 24 dB toggle
        fillProbeTones(block.data(), 512, n, amp);
        sa->pushSamples(block.data(), 512);
        if (hop < 4) continue; // let the analysis window fill
        const float* m = sa->getMagnitudesDb();
        const float d = std::abs(m[32] - m[400]);
        if (d > worst) worst = d;
        if (d > 1.0f) ++violations;
        ++checked;
    }

    EXPECT_EQ(violations, 0);
    EXPECT_LT(worst, 1.0f);
    EXPECT_GT(checked, 200);
    EXPECT_GT(n, 1LL << 24); // the run really crossed the cliff
}

// Frame coherence of a getter snapshot against a SLOW reader, which is the
// case the copy-out getters exist for (a GUI thread stalled by a page
// fault, a preemption or a debugger). fftSize 16384 makes the copy 8193
// bins -- the longest the analyser can be asked for and 16x the pin above
// -- so the reader spends as much of its life inside the copy loop as the
// public API allows, while the writer keeps publishing.
// Two distinct assertions, and the second is the one this pin adds:
//  - no snapshot is ever a mixture of two frames (the |d| bound, as above);
//  - getStaleSnapshotCount() stays 0, i.e. the reader never had to fall
//    back on the previous frame, which is the health of the handoff itself.
// Reachability argument, measured on this tree with a mutated
// computeSpectrum() that publishes the finished slot and then KEEPS writing
// into it (the `writeSlot_ = old & kSlotMask` adoption deleted): against
// the pre-seqlock header that mutation makes the reader RETURN torn frames
// (2450-2592 per 300k reads, worst |d| 18.59 dB); against this header it
// returns none and reports 257288-262313 stale snapshots instead. So the
// two assertions cover the two ways the handoff can go wrong, and neither
// is vacuous.
DSPARK_TEST(SpectrumAnalyzer_slow_reader_snapshot_is_frame_coherent)
{
    auto sa = std::make_unique<SpectrumAnalyzer<float>>();
    sa->prepare(48000.0, 16384); // 8193 bins, hop 8192
    sa->setSmoothing(0.0f);
    sa->setPeakHoldEnabled(false);
    sa->setFloorDb(-200.0f);
    EXPECT_EQ(sa->getNumBins(), 8193);

    std::atomic<bool> stop{ false };
    std::atomic<int> frames{ 0 };

    std::thread audio([&] {
        std::vector<float> block(8192);
        long long n = 0;
        int hop = 0;
        while (!stop.load(std::memory_order_relaxed))
        {
            const float amp = ((hop / 4) & 1) ? 0.025f : 0.4f; // 24 dB toggle
            fillProbeTones(block.data(), 8192, n, amp);
            sa->pushSamples(block.data(), 8192);
            ++hop;
            frames.fetch_add(1, std::memory_order_relaxed);
        }
    });

    int violations = 0;
    int fresh = 0;
    long long reads = 0;
    const long long kMaxReads = 5000000; // hard cap: never hang, fail loudly
    while ((frames.load(std::memory_order_relaxed) < 24 || reads < 4000)
           && reads < kMaxReads)
    {
        ++reads;
        if (sa->isNewDataReady()) ++fresh;
        const float* m = sa->getMagnitudesDb();
        // 1500 Hz -> bin 512, 18750 Hz -> bin 6400 at fftSize 16384.
        if (std::abs(m[512] - m[6400]) > 12.0f) ++violations;
    }

    stop.store(true, std::memory_order_relaxed);
    audio.join();

    EXPECT_EQ(violations, 0);
    EXPECT_EQ(sa->getStaleSnapshotCount(), 0LL);           // handoff health
    EXPECT_GT(frames.load(std::memory_order_relaxed), 23); // writer liveness
    EXPECT_GT(fresh, 8);                                   // adoption liveness
}

// Paired-getter returned-pointer LIFETIME. The tear pin above reads only
// getMagnitudesDb(), so it could never reach the defect this pin covers --
// that one needs two getters in play at once. With the earlier code, the
// pointer returned by one getter outlived
// the reader's slot ownership -- the next acquisition (by EITHER getter)
// handed that slot back to the writer, which then mutated the array behind
// the still-held pointer through documented usage.
// Reachability argument:
//  - Part 1 is a deterministic public-API schedule that provokes the
//    defect on demand. Against the pre-fix header (commit 3478aaa) it
//    fails unconditionally: all 129 bins mutate behind the held pointer.
//    Against the fixed header it passes, because each getter now copies
//    the acquired slot into reader-private storage and returns a pointer
//    to that copy, which no other thread can reach.
//  - Part 2 exercises BOTH getters concurrently with the writer while
//    re-reading through the first getter's held pointer, so CI TSan (the
//    C++11-aware oracle) now sees this access pattern: pre-fix it is a
//    plain-word data race TSan reports; the live pre-fix mutation rate was
//    ~263k observed mutations per 200k frames, measured live against the
//    pre-fix header -- far above this pin's zero tolerance.
// Liveness floors prove real overlap; hard caps fail loudly instead of
// hanging.
DSPARK_TEST(SpectrumAnalyzer_paired_getter_held_pointer_never_mutates)
{
    // Part 1: deterministic defect schedule (single-threaded, public API).
    {
        auto sa = std::make_unique<SpectrumAnalyzer<float>>();
        sa->prepare(48000.0, 256); // hop = 128 -> one frame per 128 samples
        sa->setSmoothing(0.0f);
        sa->setPeakHoldEnabled(true);
        sa->setFloorDb(-200.0f);

        std::vector<float> block(128);
        auto pushHop = [&](float amp) {
            std::fill(block.begin(), block.end(), amp);
            sa->pushSamples(block.data(), 128);
        };

        pushHop(0.5f);                          // frame 1 published
        const float* held = sa->getMagnitudesDb();
        const int nb = sa->getNumBins();
        std::vector<float> snap(held, held + nb);

        pushHop(0.01f);                         // frame 2 published (fresh)
        (void)sa->getPeakHoldDb();              // second acquisition
        pushHop(0.9f);                          // pre-fix: writer adopts the
        pushHop(0.7f);                          // surrendered slot and writes it

        int changed = 0;
        for (int k = 0; k < nb; ++k)
            if (held[k] != snap[static_cast<size_t>(k)]) ++changed;
        EXPECT_EQ(changed, 0); // pre-fix: 129/129
    }

    // Part 2: live paired-getter usage under a concurrent writer.
    {
        auto sa = std::make_unique<SpectrumAnalyzer<float>>();
        sa->prepare(48000.0, 1024);
        sa->setSmoothing(0.0f);
        sa->setPeakHoldEnabled(true);
        sa->setFloorDb(-200.0f);
        const int nb = sa->getNumBins();

        std::atomic<bool> stop{ false };
        std::atomic<int> frames{ 0 };

        std::thread audio([&] {
            std::vector<float> block(512);
            const float w1 = twoPi<float> * 1500.0f / 48000.0f;
            long long n = 0;
            int hop = 0;
            while (!stop.load(std::memory_order_relaxed))
            {
                const float amp = (hop & 1) ? 0.01f : 0.9f; // hop-rate toggle
                for (int i = 0; i < 512; ++i, ++n)
                    block[static_cast<size_t>(i)] =
                        amp * std::sin(w1 * static_cast<float>(n));
                sa->pushSamples(block.data(), 512);
                ++hop;
                frames.fetch_add(1, std::memory_order_relaxed);
            }
        });

        long long mutationsBehindHeldPtr = 0;
        int violations = 0;
        int fresh = 0;
        long long reads = 0;
        const long long kMaxReads = 50000000; // hard cap: never hang
        std::vector<float> snap(static_cast<size_t>(nb));

        while ((frames.load(std::memory_order_relaxed) < 64 || reads < 100000)
               && reads < kMaxReads)
        {
            ++reads;
            if (sa->isNewDataReady()) ++fresh;
            const float* m = sa->getMagnitudesDb();          // held pointer
            for (int k = 0; k < nb; ++k)
                snap[static_cast<size_t>(k)] = m[k];
            const float* p = sa->getPeakHoldDb();            // 2nd acquisition
            if (!std::isfinite(p[32]) || p[32] < -200.0f || p[32] > 20.0f)
                ++violations;                                // peaks stay sane
            // Documented straddle: keep reading through the FIRST pointer
            // after the second acquisition. Pre-fix, the writer mutates the
            // array behind it; post-fix it is reader-private snapshot
            // storage and must never change under our feet.
            for (int r = 0; r < 4; ++r)
                for (int k = 0; k < nb; ++k)
                    if (m[k] != snap[static_cast<size_t>(k)])
                        ++mutationsBehindHeldPtr;
        }

        stop.store(true, std::memory_order_relaxed);
        audio.join();

        EXPECT_EQ(mutationsBehindHeldPtr, 0LL);
        EXPECT_EQ(violations, 0);
        EXPECT_GT(frames.load(std::memory_order_relaxed), 63); // writer liveness
        EXPECT_GT(fresh, 16);                                  // adoption liveness
    }
}
