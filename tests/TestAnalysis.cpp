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
#include "../Analysis/DelayEstimator.h"
#include "../Core/FFT.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <iostream>
#include <cstddef>
#include <cstdint>
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

    // K-weighting at 997 Hz adds the 0.691 dB the LUFS constant removes, so the
    // tone reads its own per-channel peak level. Tech 3341 allows +/-0.1 LU.
    EXPECT_NEAR(meter.getIntegratedLUFS(), -23.0f, 0.05f);
}

// Integrated loudness reads the block energies themselves, not a 0.1 LU grid.
// Holding only a count per bin read -23.00 for a -23.04 tone and -30.10 for a
// -30.07 tone; an exact block-mode reference reads the input level on all three.
DSPARK_TEST(LoudnessMeter_integrated_is_not_quantized_to_the_histogram_grid)
{
    for (double level : { -23.04, -18.33, -30.07 })
    {
        LoudnessMeter<float> meter;
        meter.prepare(48000.0, 2);
        const double gain = std::pow(10.0, level / 20.0);
        std::vector<float> block(1024);
        std::int64_t position = 0;
        for (int b = 0; b < 200; ++b)
        {
            for (float& sample : block)
            {
                sample = static_cast<float>(gain * std::sin(2.0 * 3.141592653589793 * 997.0
                                                            * static_cast<double>(position++) / 48000.0));
            }
            meter.process(block.data(), block.data(), static_cast<int>(block.size()));
        }
        EXPECT_NEAR(meter.getIntegratedLUFS(), level, 0.002);
    }
}

// The relative gate keeps a bin when its centre clears the threshold, as an
// exact block-mode reference does. Rounding the gate position down kept the
// bin below it.
DSPARK_TEST(LoudnessMeter_relative_gate_tests_each_bin_at_its_centre)
{
    // 20 s at -20 and 2 s at q: the first pass means about -20.3, which puts the
    // relative gate near -30.3, between the bin centres -30.4 and -30.3.
    auto integrated = [](double quietDb)
    {
        LoudnessMeter<double> meter;
        meter.prepare(48000.0, 2);
        std::vector<double> block(1024);
        std::int64_t position = 0;
        auto run = [&](double db, double seconds)
        {
            const double gain = std::pow(10.0, db / 20.0);
            for (auto left = static_cast<std::int64_t>(seconds * 48000.0); left > 0; left -= 1024)
            {
                for (double& sample : block)
                    sample = gain * std::sin(2.0 * 3.141592653589793 * 997.0
                                             * static_cast<double>(position++) / 48000.0);
                meter.process(block.data(), block.data(), 1024);
            }
        };
        run(-20.0, 20.0);
        run(quietDb, 2.0);
        return static_cast<double>(meter.getIntegratedLUFS());
    };
    // Well below the gate the quiet segment cannot move the reading.
    EXPECT_NEAR(integrated(-45.0), -20.0, 0.05);
    // The bin at -30.4 is below the gate. Rounding the position down kept it and
    // read about -20.3.
    EXPECT_NEAR(integrated(-30.4), -20.0, 0.05);
    // The bin at -30.2 clears the gate, so it counts and pulls the mean down.
    EXPECT_LT(integrated(-30.2), -20.2);
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

namespace {
template <class T>
void checkLoudnessFloor()
{
    // Steady 1 kHz powers below, across and above both the former power-domain
    // switch and the actual -100 LUFS floor. The first two reported levels are
    // -100.70 and -100.69 LUFS before flooring; neither may fall below -100.
    constexpr double base = 1.3037063414911167e-5;
    constexpr double steps[] = { -20, -1, -0.01, 0, 0.01, 0.1, 0.69, 0.70,
                                 0.71, 0.8, 1, 2 };
    std::vector<T> tone(6 * 48000);
    T previousMomentary = T(-100), previousShortTerm = T(-100);
    for (double step : steps)
    {
        const double amplitude = base * std::pow(10.0, step / 20.0);
        for (size_t i = 0; i < tone.size(); ++i)
            tone[i] = static_cast<T>(amplitude * std::sin(
                twoPi<double> * 1000.0 * static_cast<double>(i) / 48000.0));
        LoudnessMeter<T> meter;
        meter.prepare(48000.0, 1);
        meter.process(tone.data(), static_cast<int>(tone.size()));
        const T momentary = meter.getMomentaryLUFS();
        const T shortTerm = meter.getShortTermLUFS();
        EXPECT_TRUE(momentary >= previousMomentary);
        EXPECT_TRUE(shortTerm >= previousShortTerm);
        EXPECT_NEAR(momentary, std::max(-100.0, -100.70 + step), 0.00002);
        EXPECT_NEAR(shortTerm, std::max(-100.0, -100.70 + step), 0.00002);
        if (step <= 0.69)
        {
            EXPECT_EQ(momentary, T(-100));
            EXPECT_EQ(shortTerm, T(-100));
        }
        previousMomentary = momentary;
        previousShortTerm = shortTerm;
    }
}

template <class T>
void checkFiniteTruePeak()
{
    // Test mono, then stereo with the hard ending on either channel alone.
    for (int layout = 0; layout < 3; ++layout)
    {
        LoudnessMeter<T> meter;
        meter.finalizeTruePeak(); // Unprepared is a no-op.
        const T silence = meter.getTruePeakDb();
        meter.prepare(48000.0, layout == 0 ? 1 : 2);
        meter.finalizeTruePeak();
        EXPECT_EQ(meter.getTruePeakDb(), silence);
        std::vector<T> hardEnd(48000, T(0.05)), quiet(48000, T(0.05));
        hardEnd[47998] = T(0.9);
        hardEnd[47999] = T(-0.9);
        if (layout == 0) meter.process(hardEnd.data(), 48000);
        else if (layout == 1) meter.process(hardEnd.data(), quiet.data(), 48000);
        else meter.process(quiet.data(), hardEnd.data(), 48000);
        EXPECT_NEAR(meter.getTruePeakDb(), -0.9151498112135024, 0.000002);
        const T momentary = meter.getMomentaryLUFS();
        const T shortTerm = meter.getShortTermLUFS();
        const T integrated = meter.getIntegratedLUFS();
        const T range = meter.getLoudnessRange();
        meter.finalizeTruePeak();
        // The finite hard ending reconstructs to 0.990809 (-0.0802 dB, the
        // exact band-limited sum of the zero-extended programme). The
        // long-kernel reading is 0.987980; the BS.1770 Annex 2 table alone
        // read 0.972223 (-0.2447 dB).
        EXPECT_NEAR(meter.getTruePeakDb(), -0.1050385, 0.000002);
        EXPECT_NEAR(meter.getTruePeakDb(), -0.0802, 0.03);
        const T finalPeak = meter.getTruePeakDb();
        meter.finalizeTruePeak();
        EXPECT_EQ(meter.getTruePeakDb(), finalPeak);
        EXPECT_EQ(meter.getMomentaryLUFS(), momentary);
        EXPECT_EQ(meter.getShortTermLUFS(), shortTerm);
        EXPECT_EQ(meter.getIntegratedLUFS(), integrated);
        EXPECT_EQ(meter.getLoudnessRange(), range);
        EXPECT_TRUE(meter.isMeasurementValid());
        meter.reset();
        meter.finalizeTruePeak();
        EXPECT_EQ(meter.getTruePeakDb(), silence);
        EXPECT_EQ(meter.getMomentaryLUFS(), T(-100));
        EXPECT_TRUE(meter.isMeasurementValid());
    }
}

template <class T>
void checkFinalizerLoudnessHistory()
{
    LoudnessMeter<T> finalized, untouched;
    finalized.prepare(48000.0, 1);
    untouched.prepare(48000.0, 1);
    std::vector<T> audio(7 * 48000 - 3);
    for (size_t i = 0; i < audio.size(); ++i)
        audio[i] = static_cast<T>((0.03 + 0.02 * ((i / 48000) % 4)) *
            std::sin(twoPi<double> * 1000.0 * static_cast<double>(i) / 48000.0));
    finalized.process(audio.data(), static_cast<int>(audio.size()));
    untouched.process(audio.data(), static_cast<int>(audio.size()));
    finalized.finalizeTruePeak();
    // Three samples from the next 100 ms commit. A whole-meter zero flush
    // would shift this boundary and all subsequent overlapping windows.
    const T next = T(0.125);
    for (int i = 0; i < 4800 + 3; ++i)
    {
        EXPECT_EQ(finalized.getMomentaryLUFS(), untouched.getMomentaryLUFS());
        EXPECT_EQ(finalized.getShortTermLUFS(), untouched.getShortTermLUFS());
        if (i < 5 || i == 4800)
        {
            EXPECT_EQ(finalized.getIntegratedLUFS(), untouched.getIntegratedLUFS());
            EXPECT_EQ(finalized.getLoudnessRange(), untouched.getLoudnessRange());
        }
        finalized.process(&next, 1);
        untouched.process(&next, 1);
    }
}
} // namespace

DSPARK_TEST(LoudnessMeter_floor_is_monotone_for_float_and_double)
{
    checkLoudnessFloor<float>();
    checkLoudnessFloor<double>();
}

DSPARK_TEST(LoudnessMeter_finite_true_peak_finalization)
{
    checkFiniteTruePeak<float>();
    checkFiniteTruePeak<double>();
}

DSPARK_TEST(LoudnessMeter_true_peak_finalization_preserves_loudness_clock)
{
    checkFinalizerLoudnessHistory<float>();
    checkFinalizerLoudnessHistory<double>();
}

DSPARK_TEST(TruePeakDetector_tail_query_preserves_live_channel_history)
{
    const auto check = []<class T>()
    {
        TruePeakDetector<T, 2> detector, liveReference, drained;
        for (int n = 0; n < 100; ++n)
            for (int ch = 0; ch < 2; ++ch)
            {
                const T input = static_cast<T>((n % 17 - 8) * (ch == 0 ? 0.125 : -0.0625));
                (void)detector.processSample(input, ch);
                (void)liveReference.processSample(input, ch);
                (void)drained.processSample(input, ch);
            }
        for (int ch = 0; ch < 2; ++ch)
        {
            T tail = T(0);
            for (int n = 0; n < drained.getTaps() - 1; ++n)
                tail = std::max(tail, drained.processSample(T(0), ch));
            EXPECT_EQ(detector.getTailPeak(ch), tail);
            EXPECT_EQ(detector.getTailPeak(ch), tail);
        }
        for (int n = 0; n < 100; ++n)
            for (int ch = 0; ch < 2; ++ch)
            {
                const T input = static_cast<T>((n % 13 - 6) * 0.0625);
                EXPECT_EQ(detector.processSample(input, ch), liveReference.processSample(input, ch));
            }
        detector.reset();
        EXPECT_EQ(detector.getTailPeak(0), T(0));
        EXPECT_EQ(detector.getTailPeak(1), T(0));
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

namespace {

/// Raised-cosine faded sine, amplitude 0.5, `pad` zeros either side. Its
/// exact band-limited reconstruction peaks at `exact` dB (measured on the
/// zero-extended sum with a +-2048-tap kernel).
template <class T>
std::vector<T> fadedTone(double cyclesPerSample, int fade, int steady, int pad = 64)
{
    const int n = 2 * fade + steady;
    std::vector<T> x(static_cast<size_t>(n + 2 * pad), T(0));
    for (int i = 0; i < n; ++i)
    {
        double e = 1.0;
        if (i < fade)
            e = 0.5 - 0.5 * std::cos(pi<double> * i / fade);
        else if (i >= n - fade)
            e = 0.5 - 0.5 * std::cos(pi<double> * (n - 1 - i) / fade);
        const double c = cyclesPerSample * i;
        x[static_cast<size_t>(i + pad)] = static_cast<T>(static_cast<float>(
            0.5 * e * std::sin(twoPi<double> * (c - std::floor(c)) + 0.7)));
    }
    return x;
}

struct ToneCase { double cyclesPerSample; int fade, steady; double exactDb; };

// Steady tones (30 ms fades) and 1 ms bursts. The BS.1770 Annex 2 table
// reads the first three 0.16 to 0.22 dB above their true peak and the
// bursts up to 0.04 dB below it.
constexpr ToneCase kToneCases[] = {
    { 0.085, 1440, 4800, -6.02060 }, { 0.25, 1440, 4800, -6.02060 },
    { 0.39,  1440, 4800, -6.02060 }, { 0.45, 1440, 4800, -6.02058 },
    { 0.49,  1440, 4800, -6.02059 },
    { 0.30, 48, 96, -6.02016 }, { 0.42, 48, 96, -6.01436 }, { 0.45, 48, 96, -6.00346 },
};

}   // namespace

// The long-kernel estimator reads the reconstruction peak, keeps the live
// history across a tail query, and with the Annex 2 guard never reads below
// the Annex 2 table.
DSPARK_TEST(TruePeakEstimator_reads_the_reconstruction_peak)
{
    const auto check = []<class T>()
    {
        using E = TruePeakEstimator<T>;
        for (const ToneCase& c : kToneCases)
        {
            const std::vector<T> x = fadedTone<T>(c.cyclesPerSample, c.fade, c.steady);
            E estimator, guarded, runningMax;
            typename E::Channel state, guardedState, runningState;
            TruePeakDetector<T, 1> annex;
            T peak = T(0), guardedPeak = T(0), annexPeak = T(0), running = T(0);
            T readings[E::kChunk], guardedReadings[E::kChunk];
            for (size_t start = 0; start < x.size(); start += E::kChunk)
            {
                const int n = static_cast<int>(std::min<size_t>(E::kChunk, x.size() - start));
                std::fill(readings, readings + n, T(0));
                std::fill(guardedReadings, guardedReadings + n, T(0));
                estimator.process(state, x.data() + start, n, readings);
                guarded.process(guardedState, x.data() + start, n, guardedReadings, true);
                const T chunkPeak = runningMax.processMax(runningState, x.data() + start, n, running);
                if (chunkPeak > running) running = chunkPeak;
                for (int i = 0; i < n; ++i)
                {
                    peak = std::max(peak, readings[i]);
                    guardedPeak = std::max(guardedPeak, guardedReadings[i]);
                    annexPeak = std::max(annexPeak, annex.processSample(x[start + static_cast<size_t>(i)], 0));
                }
            }
            // The running-maximum form skips refinements that cannot win: same result.
            EXPECT_EQ(running, peak);
            EXPECT_TRUE(std::equal(std::begin(state.history), std::end(state.history), std::begin(runningState.history)));
            const typename E::Channel before = state;
            peak = std::max(peak, estimator.getTailPeak(state));
            guardedPeak = std::max(guardedPeak, guarded.getTailPeak(guardedState));
            annexPeak = std::max(annexPeak, annex.getTailPeak(0));
            EXPECT_TRUE(std::equal(std::begin(before.history), std::end(before.history), std::begin(state.history)));
            EXPECT_NEAR(20.0 * std::log10(static_cast<double>(peak)), c.exactDb, 0.008);
            EXPECT_TRUE(guardedPeak >= annexPeak);
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

// The meter's true-peak readout is the long-kernel reading: steady tones read
// their true peak where the Annex 2 table read up to 0.22 dB high.
DSPARK_TEST(LoudnessMeter_true_peak_reads_steady_sines_at_their_peak)
{
    const auto check = []<class T>()
    {
        for (const ToneCase& c : kToneCases)
        {
            const std::vector<T> x = fadedTone<T>(c.cyclesPerSample, c.fade, c.steady);
            LoudnessMeter<T> meter;
            meter.prepare(48000.0, 1);
            for (size_t start = 0; start < x.size(); start += 1000)
                meter.process(x.data() + start, static_cast<int>(std::min<size_t>(1000, x.size() - start)));
            meter.finalizeTruePeak();
            EXPECT_NEAR(static_cast<double>(meter.getTruePeakDb()), c.exactDb, 0.008);
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
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

// ----------------------------------------------------------------------------
// Loudness range (EBU Tech 3342)
// ----------------------------------------------------------------------------

namespace {

double lraOf(const std::vector<float>& L, const std::vector<float>& R)
{
    LoudnessMeter<float> m;
    m.prepare(48000.0, 2);
    for (size_t i = 0; i < L.size(); i += 512)
    {
        const int n = static_cast<int>(std::min<size_t>(512, L.size() - i));
        m.process(L.data() + i, R.data() + i, n);
    }
    return static_cast<double>(m.getLoudnessRange());
}

/// Tech 3342 table 1 tone: 1 kHz, in phase on both channels, `db` dBFS peak.
void appendTone(std::vector<float>& L, std::vector<float>& R, double db, double seconds)
{
    const double a = std::pow(10.0, db / 20.0);
    const auto n = static_cast<size_t>(seconds * 48000.0);
    const size_t base = L.size();
    for (size_t i = 0; i < n; ++i)
    {
        const auto s = static_cast<float>(a * std::sin(2.0 * 3.141592653589793 * 1000.0
                                                       * static_cast<double>(base + i) / 48000.0));
        L.push_back(s); R.push_back(s);
    }
}

// Deterministic music-like programme: drums, bass and chords under a
// section-by-section level plan (verse / chorus / breakdown), 48 kHz stereo.
void lraMusicFragment(double seconds, std::vector<float>& L, std::vector<float>& R)
{
    const int fs = 48000;
    const auto n = static_cast<size_t>(seconds * fs);
    L.assign(n, 0.0f); R.assign(n, 0.0f);
    const double pi = 3.141592653589793;
    const double sectionDb[] = { -14, -12, -20, -22, -10, -9, -17, -26, -11, -13, -19, -15 };
    const double bpm = 124.0, beat = 60.0 / bpm;
    uint32_t lcg = 12345u;
    auto noise = [&]() { lcg = lcg * 1664525u + 1013904223u; return (static_cast<double>(lcg >> 8) / 8388608.0) - 1.0; };
    const double chordHz[4][3] = { {261.63, 329.63, 392.0}, {220.0, 261.63, 329.63},
                                   {174.61, 220.0, 261.63}, {196.0, 246.94, 293.66} };
    for (size_t i = 0; i < n; ++i)
    {
        const double t = static_cast<double>(i) / fs;
        // Section level, 3.5 s sections with 0.5 s linear crossfades.
        const double secLen = 3.5;
        const int sIdx = static_cast<int>(t / secLen);
        const double frac = t - sIdx * secLen;
        const double dbA = sectionDb[sIdx % 12], dbB = sectionDb[(sIdx + 1) % 12];
        const double w = frac > secLen - 0.5 ? (frac - (secLen - 0.5)) / 0.5 : 0.0;
        const double gain = std::pow(10.0, ((1.0 - w) * dbA + w * dbB) / 20.0);
        const double bt = std::fmod(t, beat);
        const int beatIdx = static_cast<int>(t / beat);
        double kick = std::sin(2.0 * pi * (50.0 * bt + 30.0 * 0.03 * (1.0 - std::exp(-bt / 0.03))))
                    * std::exp(-bt / 0.15);
        double snare = (beatIdx % 2 == 1) ? noise() * std::exp(-bt / 0.08) * 0.6 : 0.0;
        double hat = noise() * std::exp(-std::fmod(t, beat * 0.5) / 0.02) * 0.15;
        const int chord = (beatIdx / 8) % 4;
        double pad = 0.0;
        for (double f : chordHz[chord]) pad += std::sin(2.0 * pi * f * t) * 0.12;
        const double bassF = chordHz[chord][0] * 0.25;
        double bass = std::sin(2.0 * pi * bassF * t) * 0.3 * std::exp(-bt / 0.4);
        const double mid = gain * (kick * 0.7 + snare + bass);
        L[i] = static_cast<float>(mid + gain * (pad * 1.1 + hat));
        R[i] = static_cast<float>(mid + gain * (pad * 0.9 - hat));
    }
}

}   // namespace

// A steady programme has no loudness range at any length. Below 10 gated
// short-term values the 10th percentile used to resolve to rank 0 before
// any value was counted and read the relative-gate threshold - 20 LU below
// the programme - so a 5 s or 10 s steady tone read LRA 20.
DSPARK_TEST(LoudnessMeter_LRA_steady_tone_is_zero_at_every_length)
{
    for (double seconds : { 2.0, 3.0, 4.0, 5.0, 10.0, 12.0, 30.0 })
    {
        std::vector<float> L, R;
        appendTone(L, R, -23.0, seconds);
        EXPECT_NEAR(lraOf(L, R), 0.0, 0.05);
    }
}

// EBU Tech 3342-2023 table 1, cases 1-4 (the tone cases; 5 and 6 are
// programme files, covered by the conformance suite when the official set
// is available). The standard accepts +-1 LU; the meter is held to 0.1.
DSPARK_TEST(LoudnessMeter_LRA_EBU_3342_tone_vectors)
{
    struct Case { std::vector<double> levels; double lra; };
    const Case cases[] = {
        { { -20.0, -30.0 }, 10.0 },
        { { -20.0, -15.0 }, 5.0 },
        { { -40.0, -20.0 }, 20.0 },
        { { -50.0, -35.0, -20.0, -35.0, -50.0 }, 15.0 },
    };
    for (const auto& c : cases)
    {
        std::vector<float> L, R;
        for (double db : c.levels) appendTone(L, R, db, 20.0);
        EXPECT_NEAR(lraOf(L, R), c.lra, 0.1);
    }
}

// A music-like programme against the Tech 3342 reference code on the same
// samples: rank round((n - 1) * p) applied verbatim to short-term values
// taken every 100 ms by an independent BS.1770 implementation. The meter
// holds every short-term value to 0.005 LU; held at 0.1 LU it was off by
// 0.056 to 0.075 LU here.
DSPARK_TEST(LoudnessMeter_LRA_music_fragment_matches_the_reference)
{
    struct Case { double seconds; double spec; };
    const Case cases[] = {
        { 8.0,  1.6404 },
        { 35.2, 12.3559 },
        { 42.0, 11.9255 },
    };
    for (const auto& c : cases)
    {
        std::vector<float> L, R;
        lraMusicFragment(c.seconds, L, R);
        EXPECT_NEAR(lraOf(L, R), c.spec, 0.01);
    }
}

// Tech 3342 requires the short-term series at 10 Hz or more (since V3). A
// programme whose level steps every 3 s, against the reference code applied
// to short-term values taken every 100 ms (16.6027 LU) and every second
// (17.5869 LU) by an independent BS.1770 implementation: the meter reads the
// 10 Hz value (held at 0.1 LU it read 16.700), far from the 1 Hz one.
DSPARK_TEST(LoudnessMeter_LRA_stepped_programme_reads_the_10_Hz_series)
{
    const int fs = 48000;
    const auto n = static_cast<size_t>(30 * fs);
    std::vector<float> L(n), R(n);
    uint32_t a = 5u, b = 6u;
    double la = 0.0, lb = 0.0;
    auto uni = [](uint32_t& s) { s = s * 1664525u + 1013904223u; return static_cast<double>(s >> 8) / 8388608.0 - 1.0; };
    for (size_t i = 0; i < n; ++i)
    {
        const int step = static_cast<int>(i / static_cast<size_t>(3 * fs)) % 5;
        const double gain = std::pow(10.0, (-30.0 + 5.0 * step) / 20.0);
        la = 0.7 * la + 0.3 * uni(a);
        lb = 0.7 * lb + 0.3 * uni(b);
        L[i] = static_cast<float>(gain * 2.0 * la);
        R[i] = static_cast<float>(gain * 2.0 * lb);
    }
    const double lra = lraOf(L, R);
    EXPECT_NEAR(lra, 16.6027, 0.01);
    EXPECT_LT(lra, 17.5869 - 0.5);
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

DSPARK_TEST(PitchDetector_is_tuner_accurate_up_to_1760_hz)
{
    // The sub-sample parabola was fitted to the CMND, whose tau / running-sum
    // weight tilts the dip: at 48 kHz a pure 1760 Hz tone read 0.99 cent off,
    // 1 kHz 0.39 cent. On the raw difference function, with the raised-cosine
    // bias inverted, pure tones stay within 0.02 cent, and a harmonic-rich
    // tone at 1760 Hz within 0.5 cent (1.45 before).
    auto worstCents = [](double hz, bool harmonic) {
        PitchDetector<float> pd;
        pd.prepare(48000.0);
        std::vector<float> x(4096);
        double ph = 0.0, worst = 0.0;
        for (int b = 0; b < 30; ++b)
        {
            for (auto& v : x)
            {
                v = harmonic ? static_cast<float>(0.3 * std::sin(ph) + 0.25 * std::sin(2.0 * ph)
                                                  + 0.2 * std::sin(3.0 * ph) + 0.1 * std::sin(4.0 * ph))
                             : static_cast<float>(0.5 * std::sin(ph));
                ph += 6.283185307179586 * hz / 48000.0;
            }
            pd.pushSamples(std::span<const float>(x.data(), x.size()));
            if (b > 4)
                worst = std::max(worst, std::abs(1200.0 * std::log2(pd.getFrequencyHz() / hz)));
        }
        return worst;
    };
    EXPECT_LT(worstCents(440.0, false), 0.05);
    EXPECT_LT(worstCents(1000.0, false), 0.05);
    EXPECT_LT(worstCents(1760.0, false), 0.05);
    EXPECT_LT(worstCents(1760.0, true), 0.8);
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

namespace {

/// Uniform in [0, 1) from a 64-bit LCG, and a Gaussian from it (Box-Muller).
struct NoiseSource
{
    uint64_t state;
    double uniform() noexcept
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(state >> 11) * 0x1.0p-53;
    }
    double gauss() noexcept
    {
        const double a = uniform() + 1e-300, b = uniform();
        return std::sqrt(-2.0 * std::log(a)) * std::cos(2.0 * pi<double> * b);
    }
};

/// Readings of 40 harmonic tones (80 Hz to 1 kHz unless given, spectral tilt
/// -3 to -12 dB/octave, random partial phases) in white noise at `snrDb`,
/// each read once after 6000 samples at 48 kHz. Counts readings within 50
/// cents, readings an octave or two off, and wrong readings at confidence
/// 0.85 or more.
struct NoisyToneScore { int correct = 0, octave = 0, confidentWrong = 0, voiced = 0; };

NoisyToneScore scoreNoisyTones(double snrDb, bool suppression = false, double lowHz = 80.0,
                               double highHz = 1000.0)
{
    NoisyToneScore score;
    NoiseSource rng { 2024u };
    const int n = 6000;
    std::vector<double> tone(static_cast<size_t>(n));
    std::vector<float> x(static_cast<size_t>(n));
    for (int k = 0; k < 40; ++k)
    {
        const double f0 = lowHz * std::pow(highHz / lowHz, k / 39.0);
        const double tilt = 3.0 + 9.0 * rng.uniform();
        std::fill(tone.begin(), tone.end(), 0.0);
        for (int h = 1; h <= 40 && h * f0 < 20000.0; ++h)
        {
            const double amp = std::pow(10.0, -tilt * std::log2(static_cast<double>(h)) / 20.0);
            const double phase = 2.0 * pi<double> * rng.uniform();
            for (int i = 0; i < n; ++i)
                tone[static_cast<size_t>(i)] += amp * std::sin(2.0 * pi<double> * f0 * h * i / 48000.0 + phase);
        }
        double power = 0.0;
        for (const double v : tone) power += v * v;
        const double sigma = std::sqrt(power / n / std::pow(10.0, snrDb / 10.0));
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = static_cast<float>(0.2 * (tone[static_cast<size_t>(i)] + sigma * rng.gauss()));

        PitchDetector<float> pd;
        pd.prepare(48000.0);
        pd.setNoiseSuppression(suppression);
        pd.pushSamples(std::span<const float>(x.data(), x.size()));
        const double hz = static_cast<double>(pd.getFrequencyHz());
        if (hz <= 0.0) continue;
        ++score.voiced;
        const double cents = 1200.0 * std::log2(hz / f0);
        if (std::abs(cents) < 50.0) { ++score.correct; continue; }
        if (std::abs(std::abs(cents) - 1200.0) < 50.0 || std::abs(std::abs(cents) - 2400.0) < 50.0)
            ++score.octave;
        if (pd.getConfidence() >= 0.85f) ++score.confidentWrong;
    }
    return score;
}

} // namespace

// One threshold of 0.1 on the normalised difference read every frame of a
// tone at 5 dB SNR as unvoiced (its dip sits near the noise share of the
// power, 0.24): 0 of 40 correct. The vote over a threshold prior (Mauch &
// Dixon 2014) with a voicing ceiling of 0.30 reads 36 of 40, and all 40 now
// that each threshold reads the deepest minimum of its dip (a clean-room
// port of that frame stage scores 87% on such tones). Noisy readings carry
// an honest confidence below 0.85, so a confidence gate such as
// PitchFollower's still ignores them.
DSPARK_TEST(PitchDetector_tracks_tones_at_5_db_snr)
{
    const NoisyToneScore s = scoreNoisyTones(5.0);
    std::cout << "  [pitch noise] 5 dB SNR: " << s.correct << "/40 correct, "
              << s.voiced << " voiced, " << s.octave << " octave errors\n";
    EXPECT_GT(s.correct, 31);
    EXPECT_EQ(s.confidentWrong, 0);
}

// At 10 dB SNR the dip at the period sits near 0.1, so the first-dip rule
// often found a multiple of the period below the threshold first and read an
// octave or two low at confidence above 0.9 (6.8% of 400 tones in white
// noise, 3 of these 40). The vote prefers the earlier, nearly as deep dip.
DSPARK_TEST(PitchDetector_noisy_tones_avoid_octave_errors)
{
    const NoisyToneScore s = scoreNoisyTones(10.0);
    std::cout << "  [pitch noise] 10 dB SNR: " << s.correct << "/40 correct, "
              << s.octave << " octave errors, " << s.confidentWrong << " confidently wrong\n";
    EXPECT_GT(s.correct, 33);
    EXPECT_EQ(s.octave, 0);
}

// A sawtooth sweeping 1 to 4 kHz at 44.1 kHz: when the period falls between
// two lags, the dip at the period is shallow at its samples while a multiple
// near an integer lag is deep. Scored at the samples, the vote read such
// frames an octave low at confidence 0.997; the first-dip rule read one of
// 256 off. Scored at each dip's parabola vertex, no reading is off.
DSPARK_TEST(PitchDetector_high_sweep_reads_no_octave_errors)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 3.0);
    std::vector<float> x(static_cast<size_t>(n));
    std::vector<double> inst(static_cast<size_t>(n));
    double phase = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double f = 987.8 * std::pow(2.0, 2.0 * i / n);
        inst[static_cast<size_t>(i)] = f;
        phase += 2.0 * pi<double> * f / fs;
        double s = 0.0;
        for (int k = 1; k <= 30 && k * f < 0.45 * fs; ++k) s += std::sin(k * phase) / k;
        x[static_cast<size_t>(i)] = static_cast<float>(0.3 * s);
    }
    PitchDetector<float> pd;
    pd.prepare(fs);
    const int hop = pd.getHopSize(), window = pd.getWindowSize();
    int wrong = 0, voiced = 0;
    for (int at = 0; at + hop <= n; at += hop)
    {
        pd.pushSamples(std::span<const float>(x.data() + at, static_cast<size_t>(hop)));
        const double hz = static_cast<double>(pd.getFrequencyHz());
        if (hz <= 0.0) continue;
        ++voiced;
        // The sweep moves 0.7 semitone across one window: accept any pitch
        // the window held.
        bool held = false;
        for (int i = std::max(0, at + hop - window); i < at + hop && !held; i += 8)
            held = std::abs(1200.0 * std::log2(hz / inst[static_cast<size_t>(i)])) < 50.0;
        if (!held) ++wrong;
    }
    std::cout << "  [pitch sweep] " << voiced << " voiced, " << wrong << " off\n";
    EXPECT_GT(voiced, 200);
    EXPECT_EQ(wrong, 0);
}

// Noise without a period must stay unvoiced: the voicing ceiling (0.30)
// sits far below the white-noise floor of the normalised difference (about
// 0.85 at the default window) and of strongly low-passed noise (0.43).
DSPARK_TEST(PitchDetector_noise_alone_reads_unvoiced)
{
    for (const double pole : { 0.0, 0.9, 0.95 })
    {
        NoiseSource rng { 77u };
        std::vector<float> x(96000);
        double state = 0.0;
        for (auto& v : x)
        {
            state = pole * state + rng.gauss();
            v = static_cast<float>(0.05 * state);
        }
        PitchDetector<float> pd;
        pd.prepare(48000.0);
        int voiced = 0;
        for (size_t at = 0; at + 512 <= x.size(); at += 512)
        {
            pd.pushSamples(std::span<const float>(x.data() + at, 512));
            if (pd.getFrequencyHz() > 0.0f) ++voiced;
        }
        EXPECT_EQ(voiced, 0);
    }
}

// Across a legato step the window can hold two notes whose periods share a
// longer one (4 periods of B5 are 5 of D#6): the dip there was moderately
// deep in both halves and read at confidence 0.75 to 0.89, two octaves or
// more below both notes. Each note's own period repeats in one half only,
// which marks the shared period; no reading lands outside the window's notes.
DSPARK_TEST(PitchDetector_legato_change_reads_no_shared_period)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 3.0);
    for (const double base : { 987.8, 1760.0 })
    {
        std::vector<float> x(static_cast<size_t>(n));
        std::vector<double> inst(static_cast<size_t>(n));
        double phase = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double f = base * std::pow(2.0, ((i / 6000) % 5) / 12.0);
            inst[static_cast<size_t>(i)] = f;
            phase += 2.0 * pi<double> * f / fs;
            double s = 0.0;
            for (int k = 1; k <= 30 && k * f < 0.45 * fs; ++k) s += std::sin(k * phase) / k;
            x[static_cast<size_t>(i)] = static_cast<float>(0.3 * s);
        }
        PitchDetector<float> pd;
        pd.prepare(fs);
        const int window = pd.getWindowSize();
        int voiced = 0, wrong = 0;
        for (int at = 0; at + 256 <= n; at += 256)
        {
            pd.pushSamples(std::span<const float>(x.data() + at, 256));
            const double hz = static_cast<double>(pd.getFrequencyHz());
            if (hz <= 0.0) continue;
            ++voiced;
            const int p = ((at + 256) / pd.getHopSize()) * pd.getHopSize();
            bool held = false;
            for (int i = std::max(0, p - window); i < p && !held; i += 4)
                held = std::abs(1200.0 * std::log2(hz / inst[static_cast<size_t>(i)])) < 50.0;
            if (!held) ++wrong;
        }
        std::cout << "  [pitch legato] " << base << " Hz steps: " << voiced << " voiced, " << wrong << " off\n";
        EXPECT_GT(voiced, 450);
        EXPECT_EQ(wrong, 0);
    }
}

// At 0 dB SNR the dip at the period sits at about half the mean difference,
// above any voicing ceiling: the detector reads unvoiced, and a frame stage
// that always reads is wrong two times in three there. With noise floor
// suppression a steady window that reads unvoiced is read again with its
// noise removed, the period taken from the input next to the cleaned dip:
// 37 of these 40 tones read right (84.5% and 0.2% wrong of 400 measured);
// with the period refined over its multiples and confirmed by the detections
// before it, all 40 (99.8% and none), at the input's own confidence, about
// 0.5. Off, nothing changes.
DSPARK_TEST(PitchDetector_noise_suppression_reads_through_0_db_noise)
{
    const NoisyToneScore off = scoreNoisyTones(0.0, false);
    const NoisyToneScore on = scoreNoisyTones(0.0, true);
    std::cout << "  [pitch noise] 0 dB SNR: off " << off.voiced << " voiced, on "
              << on.correct << "/40 correct, " << on.voiced - on.correct << " wrong\n";
    EXPECT_EQ(off.voiced, 0);
    EXPECT_GT(on.correct, 33);
    EXPECT_LT(on.voiced - on.correct, 3);
    EXPECT_EQ(on.confidentWrong, 0);
    // 5 dB readings stay those of the first reading or better.
    EXPECT_TRUE(scoreNoisyTones(5.0, true).correct >= scoreNoisyTones(5.0, false).correct);

    // White and pink noise alone stay unvoiced.
    PitchDetector<float> pd;
    pd.prepare(48000.0);
    pd.setNoiseSuppression(true);
    EXPECT_TRUE(pd.getNoiseSuppression());
    for (const double pole : { 0.0, 0.9 })
    {
        NoiseSource rng { 91u };
        std::vector<float> x(96000);
        double state = 0.0;
        for (auto& v : x)
        {
            state = pole * state + rng.gauss();
            v = static_cast<float>(0.05 * state);
        }
        pd.reset();
        int voiced = 0;
        for (size_t at = 0; at + 512 <= x.size(); at += 512)
        {
            pd.pushSamples(std::span<const float>(x.data() + at, 512));
            if (pd.getFrequencyHz() > 0.0f) ++voiced;
        }
        EXPECT_EQ(voiced, 0);
    }
}

namespace {

/// Windows of `seconds` of band-pass noise (constant-peak band-pass biquad on
/// white noise) that read voiced at 48 kHz.
int bandNoiseVoicedWindows(double centreHz, double q, bool suppression, uint64_t seed,
                           double seconds = 2.0)
{
    const double fs = 48000.0, w0 = 2.0 * pi<double> * centreHz / fs, alpha = std::sin(w0) / (2.0 * q);
    const double a0 = 1.0 + alpha, b0 = alpha / a0, b2 = -alpha / a0;
    const double a1 = -2.0 * std::cos(w0) / a0, a2 = (1.0 - alpha) / a0;
    NoiseSource rng { seed };
    const int length = static_cast<int>(fs * seconds);
    std::vector<float> x(static_cast<size_t>(length));
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
    for (int i = 0; i < 4000 + length; ++i)
    {
        const double w = rng.gauss();
        const double y = b0 * w + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = w; y2 = y1; y1 = y;
        if (i >= 4000) x[static_cast<size_t>(i - 4000)] = static_cast<float>(0.3 * y);
    }
    PitchDetector<float> pd;
    pd.prepare(fs);
    pd.setNoiseSuppression(suppression);
    int voiced = 0;
    for (size_t at = 0; at + 512 <= x.size(); at += 512)
    {
        pd.pushSamples(std::span<const float>(x.data() + at, 512));
        if (at >= 4096 && pd.getFrequencyHz() > 0.0f) ++voiced;
    }
    return voiced;
}

} // namespace

// The cleaned window of a narrow band of noise can dip like a period's: the
// subtraction keeps its strongest bins, a few hertz apart. A tone repeats at
// every multiple of its period; such a band decorrelates at each one, so the
// suppression path now keeps a reading only when the input's periodic
// correlation at the multiples holds up. Of 179 windows of a Q 5 band at
// 1 kHz, 67 read voiced through the second read and 5 by the first alone;
// then 8, now 5. A Q 2 band at 300 Hz: 16, against 0, then 6, now 1.
DSPARK_TEST(PitchDetector_noise_suppression_ignores_narrow_band_noise)
{
    const int q5Off = bandNoiseVoicedWindows(1000.0, 5.0, false, 6u);
    const int q5On = bandNoiseVoicedWindows(1000.0, 5.0, true, 6u);
    const int q2Off = bandNoiseVoicedWindows(300.0, 2.0, false, 6u);
    const int q2On = bandNoiseVoicedWindows(300.0, 2.0, true, 6u);
    std::cout << "  [pitch noise] band noise voiced windows of 179: Q 5 at 1 kHz off " << q5Off
              << " on " << q5On << ", Q 2 at 300 Hz off " << q2Off << " on " << q2On << "\n";
    EXPECT_LT(q5On, q5Off + 15);
    EXPECT_LT(q2On, q2Off + 11);
}

// The multiples of a period are now read over the last two windows of input,
// not over the analysis window alone: a narrow band of noise has more
// multiples to decorrelate at, over a longer span. Over 20 s of four bands
// (Q 5 at 1 kHz and 300 Hz, Q 2 at 300 and 120 Hz) the second read added 182
// windows to those the first read voices; then 65. A reading taken through
// the noise must now also agree with the detections before it, which a
// narrow band of noise, wandering in pitch, rarely does: 4.
DSPARK_TEST(PitchDetector_noise_suppression_tests_multiples_over_two_windows)
{
    int excess = 0;
    for (const auto& band : { std::pair { 1000.0, 5.0 }, std::pair { 300.0, 5.0 },
                              std::pair { 300.0, 2.0 }, std::pair { 120.0, 2.0 } })
    {
        const int off = bandNoiseVoicedWindows(band.first, band.second, false, 6u, 20.0);
        const int on = bandNoiseVoicedWindows(band.first, band.second, true, 6u, 20.0);
        EXPECT_TRUE(on >= off);
        excess += on - off;
    }
    std::cout << "  [pitch noise] band noise, 20 s of four bands: second read adds " << excess
              << " voiced windows\n";
    EXPECT_LT(excess, 16);
}

// Notes below 67 Hz leave no multiple of their period inside the 48 kHz
// window, so the second read used to drop them: 50, 55 and 62 Hz harmonic
// tones at 2 dB SNR read 0 of 513 detections. Two windows of history hold
// three periods of the lowest note the detector reads: 280, none wrong;
// refined over the multiples and confirmed by recent detections, 513.
DSPARK_TEST(PitchDetector_noise_suppression_reads_notes_below_67_hz)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 2.0);
    int correct = 0, wrong = 0, total = 0;
    for (const double f0 : { 50.0, 55.0, 62.0 })
    {
        NoiseSource rng { 77u };
        std::vector<double> tone(static_cast<size_t>(n), 0.0);
        for (int h = 1; h <= 40 && h * f0 < 20000.0; ++h)
        {
            const double amp = std::pow(10.0, -6.0 * std::log2(static_cast<double>(h)) / 20.0);
            const double phase = 2.0 * pi<double> * rng.uniform();
            for (int i = 0; i < n; ++i)
                tone[static_cast<size_t>(i)] += amp * std::sin(2.0 * pi<double> * f0 * h * i / fs + phase);
        }
        double power = 0.0;
        for (const double v : tone) power += v * v;
        const double sigma = std::sqrt(power / n / std::pow(10.0, 2.0 / 10.0));
        std::vector<float> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = static_cast<float>(0.2 * (tone[static_cast<size_t>(i)] + sigma * rng.gauss()));
        PitchDetector<float> pd;
        pd.prepare(fs);
        pd.setNoiseSuppression(true);
        for (int at = 0; at + 512 <= n; at += 512)
        {
            pd.pushSamples(std::span<const float>(x.data() + at, 512));
            if (at < 8192) continue;
            ++total;
            const double hz = static_cast<double>(pd.getFrequencyHz());
            if (hz <= 0.0) continue;
            if (std::abs(1200.0 * std::log2(hz / f0)) < 50.0) ++correct;
            else ++wrong;
        }
    }
    std::cout << "  [pitch noise] 50-62 Hz at 2 dB SNR: " << correct << " of " << total
              << " right, " << wrong << " wrong\n";
    EXPECT_GT(correct, 200);
    EXPECT_EQ(wrong, 0);
}

// A broad dip of a low note carries dozens of small minima that noise
// leaves on its walls. Each threshold of the vote used to pick its first
// minimum, so the dip's mass split across them and one on the wall won:
// 47 to 67 Hz tones at 10 dB SNR read 50 to 140 cents sharp at confidence
// 0.81 to 0.89 (31.5% of 200). A threshold now picks the deepest minimum of
// the stretch below it.
DSPARK_TEST(PitchDetector_low_notes_in_noise_read_their_period)
{
    const NoisyToneScore s = scoreNoisyTones(10.0, false, 47.0, 67.0);
    std::cout << "  [pitch noise] 47-67 Hz at 10 dB SNR: " << s.correct << "/40 correct, "
              << s.voiced - s.correct << " wrong\n";
    EXPECT_GT(s.correct, 37);
    EXPECT_EQ(s.voiced - s.correct, 0);
}

// Where the cleaned window of a low note does not dip below the threshold
// (its partials crowd the floor estimate), the input's own dip is now the
// candidate, refined over the period's multiples and confirmed by the
// detections before it: 47 to 67 Hz tones at 0 dB SNR in white noise read
// 40.5% right before, 99.5% now (200 tones), none wrong.
DSPARK_TEST(PitchDetector_noise_suppression_reads_low_notes_at_0_db)
{
    const NoisyToneScore s = scoreNoisyTones(0.0, true, 47.0, 67.0);
    std::cout << "  [pitch noise] 47-67 Hz at 0 dB SNR: " << s.correct << "/40 correct, "
              << s.voiced - s.correct << " wrong\n";
    EXPECT_GT(s.correct, 35);
    EXPECT_LT(s.voiced - s.correct, 2);
    EXPECT_EQ(s.confidentWrong, 0);
}

// A reading through the noise must agree with the detections before it,
// within a tolerance that grows with their age, so a 50-cent vibrato keeps
// agreeing: continuous notes of 70 to 880 Hz with 50 cents of vibrato at
// 5.5 Hz and 2 dB SNR read 83.4% of detections right before, 99.6% now.
DSPARK_TEST(PitchDetector_noise_suppression_follows_vibrato)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 1.5);
    int correct = 0, wrong = 0, total = 0;
    for (const double f0 : { 110.0, 330.0, 880.0 })
    {
        NoiseSource rng { 41u };
        std::vector<double> tone(static_cast<size_t>(n), 0.0), inst(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            inst[static_cast<size_t>(i)] = f0 * std::pow(2.0, 0.5 / 12.0 * std::sin(2.0 * pi<double> * 5.5 * i / fs));
        for (int h = 1; h <= 40 && h * f0 < 20000.0; ++h)
        {
            const double amp = std::pow(10.0, -6.0 * std::log2(static_cast<double>(h)) / 20.0);
            double phase = 2.0 * pi<double> * rng.uniform();
            for (int i = 0; i < n; ++i)
            {
                phase += 2.0 * pi<double> * inst[static_cast<size_t>(i)] * h / fs;
                tone[static_cast<size_t>(i)] += amp * std::sin(phase);
            }
        }
        double power = 0.0;
        for (const double v : tone) power += v * v;
        const double sigma = std::sqrt(power / n / std::pow(10.0, 2.0 / 10.0));
        std::vector<float> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = static_cast<float>(0.2 * (tone[static_cast<size_t>(i)] + sigma * rng.gauss()));
        PitchDetector<float> pd;
        pd.prepare(fs);
        pd.setNoiseSuppression(true);
        for (int at = 0; at + 512 <= n; at += 512)
        {
            pd.pushSamples(std::span<const float>(x.data() + at, 512));
            const int end = at + 512;
            if (end < 4096) continue;
            ++total;
            double mean = 0.0;
            for (int i = end - 2048; i < end; ++i) mean += inst[static_cast<size_t>(i)];
            mean /= 2048.0;
            const double hz = static_cast<double>(pd.getFrequencyHz());
            if (hz <= 0.0) continue;
            if (std::abs(1200.0 * std::log2(hz / mean)) < 50.0) ++correct;
            else ++wrong;
        }
    }
    std::cout << "  [pitch noise] 50-cent vibrato at 2 dB SNR: " << correct << " of " << total
              << " right, " << wrong << " wrong\n";
    EXPECT_GT(correct, (total * 95) / 100);
    EXPECT_EQ(wrong, 0);
}

namespace {

/// A phase-continuous harmonic tone (10 partials, 1/k) of `hz` into `x`.
void harmonicTone(std::vector<float>& x, int from, int to, double hz, double& phase, double fs)
{
    for (int i = from; i < to; ++i)
    {
        phase += 2.0 * pi<double> * hz / fs;
        double s = 0.0;
        for (int k = 1; k <= 10; ++k) s += std::sin(k * phase) / k;
        x[static_cast<size_t>(i)] = static_cast<float>(0.25 * s);
    }
}

} // namespace

// Every reading whole octaves from the target used to be folded back to it,
// so a melody's octave leap never came through: a 220 Hz note followed by
// 440 Hz kept reading 220 Hz for as long as the second note lasted (21.9% of
// voiced frames an octave off on legato melodies with an octave leap in one
// step of eight). An octave leap now arrives like any large interval, once
// two consecutive readings agree on it.
DSPARK_TEST(PitchFollower_follows_an_octave_leap)
{
    const double fs = 48000.0;
    const int n = static_cast<int>(fs * 1.2);
    std::vector<float> x(static_cast<size_t>(n));
    double phase = 0.0;
    harmonicTone(x, 0, n / 2, 220.0, phase, fs);
    harmonicTone(x, n / 2, n, 440.0, phase, fs);
    for (const float glide : { 0.0f, 60.0f })
    {
        PitchFollower<float> pf;
        pf.prepare(spec(fs, 512, 1));
        pf.setGlide(glide);
        int arrival = -1;
        for (int at = 0; at + 512 <= n; at += 512)
        {
            pf.pushSamples(std::span<const float>(x.data() + at, 512));
            if (at >= n / 2 && arrival < 0 && std::abs(1200.0 * std::log2(pf.getSmoothedHz() / 440.0)) < 50.0)
                arrival = at + 512 - n / 2;
        }
        std::cout << "  [pitch follower] octave leap, glide " << glide << " ms: "
                  << pf.getSmoothedHz() << " Hz at the end, arrived after " << arrival << " samples\n";
        EXPECT_NEAR(pf.getSmoothedHz(), 440.0f, 4.0f);
        EXPECT_TRUE(arrival > 0 && arrival < 6000);
    }
}

// isVoiced() follows each detection, where isTracking() holds for 250 ms;
// setNoiseSuppression() reaches the wrapped detector, so with a lower gate
// the follower reads a tone through 0 dB of white noise.
DSPARK_TEST(PitchFollower_reports_voicing_and_reads_through_noise)
{
    const double fs = 48000.0;
    {
        const int n = static_cast<int>(fs * 0.6);
        std::vector<float> x(static_cast<size_t>(n), 0.0f);
        double phase = 0.0;
        harmonicTone(x, 0, static_cast<int>(fs * 0.5), 220.0, phase, fs);
        PitchFollower<float> pf;
        pf.prepare(spec(fs, 512, 1));
        bool voicedInTone = false;
        for (int at = 0; at + 512 <= n; at += 512)
        {
            pf.pushSamples(std::span<const float>(x.data() + at, 512));
            if (at + 512 == 23552) voicedInTone = pf.isVoiced();
        }
        EXPECT_TRUE(voicedInTone);
        EXPECT_FALSE(pf.isVoiced());   // 4600 samples of silence
        EXPECT_TRUE(pf.isTracking());  // still within the 250 ms hold
        pf.reset();
        EXPECT_FALSE(pf.isVoiced());
    }
    {
        const int n = static_cast<int>(fs * 1.5);
        NoiseSource rng { 5u };
        std::vector<float> tone(static_cast<size_t>(n));
        double phase = 0.0;
        harmonicTone(tone, 0, n, 220.0, phase, fs);
        double power = 0.0;
        for (const float v : tone) power += static_cast<double>(v) * v;
        const double sigma = std::sqrt(power / n);   // 0 dB SNR
        std::vector<float> x(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] = static_cast<float>(0.5 * (tone[static_cast<size_t>(i)] + sigma * rng.gauss()));
        PitchFollower<float> pf;
        pf.prepare(spec(fs, 512, 1));
        pf.setNoiseSuppression(true);
        EXPECT_TRUE(pf.getNoiseSuppression());
        pf.setConfidence(0.45f);
        pf.setGlide(0.0f);
        int good = 0, total = 0;
        for (int at = 0; at + 512 <= n; at += 512)
        {
            pf.pushSamples(std::span<const float>(x.data() + at, 512));
            if (at < 9600) continue;
            ++total;
            if (pf.isVoiced() && std::abs(1200.0 * std::log2(pf.getSmoothedHz() / 220.0)) < 50.0) ++good;
        }
        std::cout << "  [pitch follower] 220 Hz at 0 dB SNR with noise suppression: " << good << " of "
                  << total << " detections voiced and right\n";
        EXPECT_GT(good, (total * 9) / 10);
    }
}

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

struct SpectrumPublicationProof
{
    static constexpr int publications = 12;
    int observable = 0;
    int consumedOnce = 0;
    int coherent = 0;
};

// isNewDataReady() is a coalescing Boolean, not a frame counter. Prove its
// publication contract on a finite cross-thread schedule: the writer cannot
// publish frame N+1 until the reader has observed, consumed and adopted frame
// N. The acknowledgement is independent of the analyser notification, so a
// missing notification fails the counts below instead of hanging this proof.
static SpectrumPublicationProof runSpectrumPublicationProof()
{
    SpectrumAnalyzer<float> sa;
    sa.prepare(48000.0, 1024);
    sa.setSmoothing(0.0f);
    sa.setPeakHoldEnabled(false);
    sa.setFloorDb(-200.0f);

    std::vector<float> block(512);
    long long n = 0;
    for (int warmup = 0; warmup < 4; ++warmup)
    {
        fillProbeTones(block.data(), 512, n, 0.4f);
        sa.pushSamples(block.data(), 512);
    }
    (void)sa.isNewDataReady();
    (void)sa.getMagnitudesDb();

    std::atomic<int> requested{ 0 };
    std::atomic<int> published{ 0 };
    std::atomic<int> acknowledged{ 0 };
    std::thread publisher([&] {
        for (int publication = 1;
             publication <= SpectrumPublicationProof::publications;
             ++publication)
        {
            int observed = requested.load(std::memory_order_acquire);
            while (observed < publication)
            {
                requested.wait(observed, std::memory_order_acquire);
                observed = requested.load(std::memory_order_acquire);
            }

            fillProbeTones(block.data(), 512, n, 0.4f);
            sa.pushSamples(block.data(), 512);
            published.store(publication, std::memory_order_release);
            published.notify_one();

            observed = acknowledged.load(std::memory_order_acquire);
            while (observed < publication)
            {
                acknowledged.wait(observed, std::memory_order_acquire);
                observed = acknowledged.load(std::memory_order_acquire);
            }
        }
    });

    SpectrumPublicationProof proof;
    for (int publication = 1;
         publication <= SpectrumPublicationProof::publications;
         ++publication)
    {
        requested.store(publication, std::memory_order_release);
        requested.notify_one();

        int observed = published.load(std::memory_order_acquire);
        while (observed < publication)
        {
            published.wait(observed, std::memory_order_acquire);
            observed = published.load(std::memory_order_acquire);
        }

        const bool ready = sa.isNewDataReady();
        const bool consumed = !sa.isNewDataReady();
        if (ready) ++proof.observable;
        if (ready && consumed) ++proof.consumedOnce;
        const float* m = sa.getMagnitudesDb();
        if (std::abs(m[32] - m[400]) <= 12.0f) ++proof.coherent;

        acknowledged.store(publication, std::memory_order_release);
        acknowledged.notify_one();
    }
    publisher.join();
    return proof;
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
// loudly instead of hanging). Notification publication is proved separately
// by runSpectrumPublicationProof(), without counting scheduler slices.
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
    long long reads = 0;
    const long long kMaxReads = 50000000; // hard cap: never hang, fail loudly
    while ((frames.load(std::memory_order_relaxed) < 64 || reads < 300000)
           && reads < kMaxReads)
    {
        ++reads;
        (void)sa->isNewDataReady();
        const float* m = sa->getMagnitudesDb();
        const float d = m[32] - m[400];
        if (std::abs(d) > 12.0f) ++violations;
    }

    stop.store(true, std::memory_order_relaxed);
    audio.join();

    EXPECT_EQ(violations, 0);
    EXPECT_GT(frames.load(std::memory_order_relaxed), 63); // writer liveness
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
    long long reads = 0;
    const long long kMaxReads = 5000000; // hard cap: never hang, fail loudly
    while ((frames.load(std::memory_order_relaxed) < 24 || reads < 4000)
           && reads < kMaxReads)
    {
        ++reads;
        (void)sa->isNewDataReady();
        const float* m = sa->getMagnitudesDb();
        // 1500 Hz -> bin 512, 18750 Hz -> bin 6400 at fftSize 16384.
        if (std::abs(m[512] - m[6400]) > 12.0f) ++violations;
    }

    stop.store(true, std::memory_order_relaxed);
    audio.join();

    EXPECT_EQ(violations, 0);
    EXPECT_EQ(sa->getStaleSnapshotCount(), 0LL);           // handoff health
    EXPECT_GT(frames.load(std::memory_order_relaxed), 23); // writer liveness

    const SpectrumPublicationProof proof = runSpectrumPublicationProof();
    EXPECT_EQ(proof.observable, SpectrumPublicationProof::publications);
    EXPECT_EQ(proof.consumedOnce, SpectrumPublicationProof::publications);
    EXPECT_EQ(proof.coherent, SpectrumPublicationProof::publications);
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
// Free-running progress floors prove real overlap; the finite
// request/publish/acknowledge proof below pins notification/adoption without
// making its verdict depend on how many writer publications one reader slice
// happens to observe. Hard caps fail loudly instead of hanging.
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
        long long reads = 0;
        const long long kMaxReads = 50000000; // hard cap: never hang
        std::vector<float> snap(static_cast<size_t>(nb));

        while ((frames.load(std::memory_order_relaxed) < 64 || reads < 100000)
               && reads < kMaxReads)
        {
            ++reads;
            (void)sa->isNewDataReady();
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
    }

    const SpectrumPublicationProof proof = runSpectrumPublicationProof();
    EXPECT_EQ(proof.observable, SpectrumPublicationProof::publications);
    EXPECT_EQ(proof.consumedOnce, SpectrumPublicationProof::publications);
    EXPECT_EQ(proof.coherent, SpectrumPublicationProof::publications);
}

// ============================================================================
// Rate invariance of the shipped analysis defaults (Analysis/).
//
// Every default below used to be a fixed SAMPLE COUNT, which makes the
// property it delivers -- a musical register, a display resolution, a
// selectivity in Hz -- a function of the sample rate. The pins are counts, Hz
// and dB only: no wall-clock durations are asserted anywhere here, so nothing
// a build or an instrumented run can rescale is load-bearing.
// ============================================================================

namespace {

// Band-limited sawtooth: harmonic-rich, so YIN has a genuine period to find.
std::vector<float> rateInvSaw(int n, double fs, double f0)
{
    std::vector<float> out(static_cast<size_t>(n), 0.0f);
    const int H = std::max(1, static_cast<int>(std::min(8000.0, fs * 0.45) / f0));
    for (int i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (int h = 1; h <= H; ++h)
            s += std::sin(dspark::twoPi<double> * f0 * h * i / fs) / h;
        out[static_cast<size_t>(i)] = 0.5f * static_cast<float>(s * 0.5);
    }
    return out;
}

double centsOff(double got, double want)
{
    if (!(got > 0.0)) return 1e9;
    return 1200.0 * std::log2(got / want);
}

} // namespace

// The YIN lag search reaches down to fs/(windowSize/2 - 1). With the old fixed
// 2048-sample default that floor was 46.9 Hz at 48 kHz but 93.8 Hz at 96 kHz
// and 187.7 Hz at 192 kHz, so A1/E2 -- and at 192 kHz A2 -- read as unvoiced
// with the DEFAULT settings. Measured pre-fix: 0 Hz, confidence 0.
DSPARK_TEST(PitchDetector_default_window_holds_the_register_across_rates)
{
    struct Case { double fs; int expectedWindow; };
    const Case cases[] = {
        { 44100.0, 2048 }, { 48000.0, 2048 },
        { 88200.0, 4096 }, { 96000.0, 4096 },
        { 176400.0, 8192 }, { 192000.0, 8192 },
    };
    for (const Case& c : cases)
    {
        PitchDetector<float> pd;
        pd.prepare(c.fs);
        EXPECT_EQ(pd.getWindowSize(), c.expectedWindow);
        EXPECT_EQ(pd.getHopSize(), c.expectedWindow / 4);   // constant overlap
    }

    // Correctness at the professional rates, not just a map: A1 (55 Hz) and
    // E2 (82.41 Hz) with the DEFAULT settings.
    for (double fs : { 96000.0, 176400.0, 192000.0 })
    {
        for (double f0 : { 55.0, 82.4069 })
        {
            const int n = static_cast<int>(fs * 1.2);
            std::vector<float> sig = rateInvSaw(n, fs, f0);
            PitchDetector<float> pd;
            pd.prepare(fs);
            pd.pushSamples(std::span<const float>(sig.data(), static_cast<size_t>(n)));
            const double hz = static_cast<double>(pd.getFrequencyHz());
            std::cout << "  [rate-inv] pitch fs=" << fs << " f0=" << f0
                      << " -> " << hz << " Hz conf=" << pd.getConfidence() << "\n";
            EXPECT_GT(hz, 0.0);
            EXPECT_LT(std::abs(centsOff(hz, f0)), 25.0);
            EXPECT_GT(static_cast<double>(pd.getConfidence()), 0.8);
        }
    }

    // Explicit windows are still honoured verbatim, with their documented
    // (rate-dependent) register. 2048 samples at 192 kHz put the floor at
    // 187.7 Hz, so A1 is out of reach: the documented outcome is a reading
    // that is NOT the played note. If this ever starts reporting 55 Hz, the
    // register documentation is stale -- update it.
    PitchDetector<float> pinned;
    pinned.prepare(192000.0, 2048, 512);
    EXPECT_EQ(pinned.getWindowSize(), 2048);
    EXPECT_EQ(pinned.getHopSize(), 512);
    {
        const int n = static_cast<int>(192000.0 * 1.2);
        std::vector<float> sig = rateInvSaw(n, 192000.0, 55.0);
        pinned.pushSamples(std::span<const float>(sig.data(), static_cast<size_t>(n)));
        const double hz = static_cast<double>(pinned.getFrequencyHz());
        std::cout << "  [rate-inv] explicit 2048 at 192 kHz, A1 -> " << hz << " Hz\n";
        EXPECT_FALSE(hz > 0.0 && std::abs(centsOff(hz, 55.0)) < 25.0);
    }
}

// The wrapper inherits the register: its documented low-end claim has to hold
// at professional rates too. Pre-fix at 192 kHz a 55 Hz sawtooth left
// getSmoothedHz() at 0 and isTracking() false whatever setRange() said.
DSPARK_TEST(PitchFollower_tracks_bass_at_professional_rates)
{
    for (double fs : { 96000.0, 192000.0 })
    {
        PitchFollower<float> pf;
        pf.prepare(spec(fs, 512, 1));
        EXPECT_EQ(pf.getWindowSize(), (fs > 100000.0) ? 8192 : 4096);
        pf.setRange(40.0f, 1200.0f);

        const int n = static_cast<int>(fs * 1.5);
        std::vector<float> sig = rateInvSaw(n, fs, 55.0);
        pf.pushSamples(std::span<const float>(sig.data(), static_cast<size_t>(n)));
        std::cout << "  [rate-inv] follower fs=" << fs << " A1 -> "
                  << pf.getSmoothedHz() << " Hz tracking=" << pf.isTracking() << "\n";
        EXPECT_TRUE(pf.isTracking());
        EXPECT_NEAR(static_cast<double>(pf.getSmoothedHz()), 55.0, 0.5);
    }
}

// The display resolution is 4*fs/fftSize (Hann main lobe). With the old fixed
// 2048-point default, two partials a fifth apart in the mid register (D4 and
// A4, 146.3 Hz) merged into ONE peak at 176.4 and 192 kHz.
DSPARK_TEST(SpectrumAnalyzer_default_size_holds_the_resolution_across_rates)
{
    struct Case { double fs; int expectedFft; };
    const Case cases[] = {
        { 44100.0, 2048 }, { 48000.0, 2048 },
        { 88200.0, 4096 }, { 96000.0, 4096 },
        { 176400.0, 8192 }, { 192000.0, 8192 },
    };
    for (const Case& c : cases)
    {
        SpectrumAnalyzer<float> sa;
        sa.prepare(c.fs);
        EXPECT_EQ(sa.getFFTSize(), c.expectedFft);
        EXPECT_EQ(sa.getNumBins(), c.expectedFft / 2 + 1);
        // Bin width is the resolution; it must not move with the rate.
        EXPECT_NEAR(static_cast<double>(sa.binToFrequency(1)), c.fs / c.expectedFft, 1e-6);
        EXPECT_LT(static_cast<double>(sa.binToFrequency(1)), 24.0);
    }

    // Two tones a fifth apart must read as TWO peaks at every rate.
    for (double fs : { 96000.0, 176400.0, 192000.0 })
    {
        SpectrumAnalyzer<float> sa;
        sa.prepare(fs);
        sa.setSmoothing(0.0f);
        const int N = sa.getFFTSize();
        const int numBins = sa.getNumBins();
        const int n = 24 * N;
        std::vector<float> sig(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            sig[static_cast<size_t>(i)] = 0.45f * static_cast<float>(
                std::sin(dspark::twoPi<double> * 293.665 * i / fs)
                + std::sin(dspark::twoPi<double> * 440.0 * i / fs));
        sa.pushSamples(sig.data(), n);
        const float* mag = sa.getMagnitudesDb();

        const double binHz = fs / N;
        const int b1 = static_cast<int>(std::lround(293.665 / binHz));
        const int b2 = static_cast<int>(std::lround(440.0 / binHz));
        EXPECT_GT(b2 - b1, 1);
        double p1 = -300.0, p2 = -300.0, dip = 300.0;
        for (int k = std::max(0, b1 - 2); k <= std::min(numBins - 1, b1 + 2); ++k)
            p1 = std::max(p1, static_cast<double>(mag[k]));
        for (int k = std::max(0, b2 - 2); k <= std::min(numBins - 1, b2 + 2); ++k)
            p2 = std::max(p2, static_cast<double>(mag[k]));
        for (int k = b1 + 1; k < b2; ++k) dip = std::min(dip, static_cast<double>(mag[k]));
        std::cout << "  [rate-inv] spectrum fs=" << fs << " N=" << N
                  << " D4=" << p1 << " dip=" << dip << " A4=" << p2 << " dB\n";
        EXPECT_GT(std::min(p1, p2) - dip, 3.0);   // a real valley between them
    }

    // Explicit sizes remain honoured verbatim.
    SpectrumAnalyzer<float> pinned;
    pinned.prepare(192000.0, 2048);
    EXPECT_EQ(pinned.getFFTSize(), 2048);
}

// Goertzel takes its block length from the caller by design; what the header
// documents is the LAW. This pins the law itself: a count that separates two
// semitones at 48 kHz does not separate them at 96 or 192 kHz, and a length
// scaled with the rate does.
DSPARK_TEST(Goertzel_semitone_selectivity_follows_the_rate_law)
{
    auto rejectionDb = [](double fs, int N) {
        auto magAt = [&](double toneHz) {
            Goertzel<double> g;
            g.prepare(fs, 261.6255653, N);
            std::vector<double> sig(static_cast<size_t>(N));
            for (int i = 0; i < N; ++i)
                sig[static_cast<size_t>(i)] = std::sin(dspark::twoPi<double> * toneHz * i / fs);
            g.processBlock(sig.data(), N);
            return static_cast<double>(g.getMagnitude());
        };
        const double on = magAt(261.6255653);      // C4
        const double off = magAt(277.1826310);     // C#4, one semitone up
        return 20.0 * std::log10(std::max(on, 1e-30) / std::max(off, 1e-30));
    };

    // The 48 kHz reference block separates the semitones.
    EXPECT_GT(rejectionDb(48000.0, 2048), 6.0);

    // Reusing that COUNT at higher rates does not (measured 1.3 dB at 96 kHz,
    // 0.2 dB at 192 kHz): the two semitones become one reading.
    EXPECT_LT(rejectionDb(96000.0, 2048), 3.0);
    EXPECT_LT(rejectionDb(192000.0, 2048), 3.0);

    // Scaling the block with the rate -- N >= fs / (wanted bin width) --
    // restores it. This is the rule ChordDetector applies to its note bank.
    for (double fs : { 96000.0, 192000.0 })
    {
        int n = 1024;
        const double target = fs * (2048.0 / 48000.0);
        while (n < 16384 && static_cast<double>(n) < target) n *= 2;
        const double r = rejectionDb(fs, n);
        std::cout << "  [rate-inv] goertzel fs=" << fs << " N=" << n
                  << " semitone rejection=" << r << " dB\n";
        EXPECT_GT(r, 6.0);
    }
}

// ============================================================================
// DelayEstimator
// ============================================================================

namespace {

// Delays x by d samples exactly (band-limited, circular) in the frequency
// domain; x's length is a power of two.
std::vector<double> exactDelay(const std::vector<double>& x, double d)
{
    const size_t n = x.size();
    FFTReal<double> f(n);
    std::vector<double> b(n + 2);
    f.forward(x.data(), b.data());
    for (size_t k = 0; k <= n / 2; ++k)
    {
        std::complex<double> c(b[2 * k], b[2 * k + 1]);
        c *= std::polar(1.0, -2.0 * pi<double> * static_cast<double>(k) * d / static_cast<double>(n));
        if (k == n / 2) c = std::complex<double>(c.real(), 0.0);
        b[2 * k] = c.real();
        b[2 * k + 1] = c.imag();
    }
    std::vector<double> y(n);
    f.inverse(b.data(), y.data());
    double sx = 0.0, sy = 0.0;
    for (size_t i = 0; i < n; ++i) { sx += x[i] * x[i]; sy += y[i] * y[i]; }
    const double g = std::sqrt(sx / sy);   // undo the transform pair's scale
    for (auto& v : y) v *= g;
    return y;
}

struct DelayCase { std::vector<float> ref, del; };

DelayCase makeDelayCase(bool coloured, double delay, double snrDb, double gain, uint32_t seed)
{
    const size_t n = size_t(1) << 18;   // 5.5 s at 48 kHz
    uint32_t s = seed;
    auto gauss = [&s]() {
        double acc = 0.0;
        for (int k = 0; k < 12; ++k) { s = s * 1664525u + 1013904223u; acc += static_cast<double>(s >> 8) / 16777216.0; }
        return acc - 6.0;
    };
    std::vector<double> x(n);
    double lp = 0.0, lp2 = 0.0;
    for (auto& v : x)
    {
        const double w = gauss();
        lp = 0.95 * lp + 0.05 * w;
        lp2 = 0.9 * lp2 + 0.1 * lp;
        v = coloured ? 8.0 * lp2 : 0.1 * w;
    }
    auto y = exactDelay(x, delay);
    double p = 0.0;
    for (const double v : y) p += v * v;
    const double noise = std::sqrt(p / static_cast<double>(n) * std::pow(10.0, -snrDb / 10.0));
    for (auto& v : y) v = gain * (v + noise * gauss());
    auto toFloat = [](const std::vector<double>& v) {
        std::vector<float> f(v.size());
        for (size_t i = 0; i < v.size(); ++i) f[i] = static_cast<float>(v[i]);
        return f;
    };
    return { toFloat(x), toFloat(y) };
}

DelayEstimator<float>::Result estimateDelay(const DelayCase& c, int maxDelay)
{
    const float* a = c.ref.data();
    const float* b = c.del.data();
    return DelayEstimator<float>::estimate(
        AudioBufferView<const float>(&a, 1, static_cast<int>(c.ref.size())),
        AudioBufferView<const float>(&b, 1, static_cast<int>(c.del.size())), 48000.0, maxDelay);
}

} // namespace

// Known whole and fractional delays, either way, on white and on coloured
// material, with noise and a gain difference, are recovered to a small
// fraction of a sample (measured within 0.0005 samples on white noise down
// to 0 dB SNR, within 0.003 on coloured noise at 20 dB).
DSPARK_TEST(DelayEstimator_recovers_known_delays_to_a_fraction_of_a_sample)
{
    for (const bool coloured : { false, true })
        for (const double d : { 0.0, 0.5, 12.37, 37.0, -250.4, 4000.25 })
            for (const double snr : { 60.0, 20.0 })
            {
                const auto c = makeDelayCase(coloured, d, snr, 0.5, 11u);
                const auto r = estimateDelay(c, 6000);
                const double err = r.delaySamples - d;
                std::cout << "  " << (coloured ? "coloured" : "white") << " delay " << d << " at "
                          << snr << " dB: " << r.delaySamples << " (" << err << "), confidence "
                          << r.confidence << "\n";
                EXPECT_TRUE(r.valid);
                EXPECT_FALSE(r.inverted);
                EXPECT_LT(std::abs(err), coloured ? 0.01 : 0.002);
                EXPECT_GT(r.confidence, 0.95);
                EXPECT_NEAR(r.delaySeconds, r.delaySamples / 48000.0, 1e-12);
            }
}

// Noise as loud as the signal still leaves one delay to find; an inverted
// delivery is reported as inverted, not as another delay; unrelated material
// reads as having no delay to speak of.
DSPARK_TEST(DelayEstimator_noise_polarity_and_unrelated_material)
{
    const auto noisy = makeDelayCase(false, 12.37, 0.0, 1.0, 21u);
    const auto r = estimateDelay(noisy, 1000);
    EXPECT_LT(std::abs(r.delaySamples - 12.37), 0.005);

    const auto inv = makeDelayCase(false, 100.3, 60.0, -0.1, 23u);
    const auto ri = estimateDelay(inv, 1000);
    EXPECT_TRUE(ri.inverted);
    EXPECT_LT(std::abs(ri.delaySamples - 100.3), 0.002);
    EXPECT_GT(ri.confidence, 0.95);

    const auto a = makeDelayCase(false, 0.0, 60.0, 1.0, 31u);
    const auto b = makeDelayCase(false, 0.0, 60.0, 1.0, 97u);
    DelayCase unrelated { a.ref, b.ref };
    const auto ru = estimateDelay(unrelated, 1000);
    std::cout << "  unrelated: confidence " << ru.confidence << "\n";
    EXPECT_LT(ru.confidence, 0.1);
}

// Pushed in blocks of any size, the estimate is the one the whole-signal
// call gives: memory is a few frames, not the program. Neither path appends
// zeros after a programme of at least one frame.
DSPARK_TEST(DelayEstimator_block_feeding_matches_the_whole_signal_call)
{
    const auto c = makeDelayCase(true, 37.25, 30.0, 0.7, 5u);
    const auto whole = estimateDelay(c, 2000);
    for (const int block : { 1, 64, 1000, 4096 })
    {
        DelayEstimator<float> est;
        est.prepare(48000.0, 2000);
        const int n = static_cast<int>(c.ref.size());
        for (int p = 0; p < n; p += block)
        {
            const float* a = c.ref.data() + p;
            const float* b = c.del.data() + p;
            const int len = std::min(block, n - p);
            est.push(AudioBufferView<const float>(&a, 1, len), AudioBufferView<const float>(&b, 1, len));
        }
        const auto r = est.estimate();
        EXPECT_NEAR(r.delaySamples, whole.delaySamples, 1e-9);
        EXPECT_NEAR(r.confidence, whole.confidence, 1e-9);
    }
    DelayEstimator<float> idle;
    idle.prepare(48000.0, 100);
    EXPECT_FALSE(idle.estimate().valid);
}

// While the first half of the window is still digital silence, the difference
// function is exactly zero at the small lags, and the CMND read that 0/0 as a
// perfect dip: a note after silence reported 24000 Hz at confidence 1.0 for
// 16 polls of 64 samples, 21 ms.
DSPARK_TEST(PitchDetector_note_after_silence_is_not_a_confident_nyquist)
{
    PitchDetector<float> detector;
    detector.prepare(48000.0);

    std::vector<float> x(96000, 0.0f);
    for (int i = 48000; i < 96000; ++i)
        x[static_cast<size_t>(i)] = 0.5f * static_cast<float>(
            std::sin(2.0 * pi<double> * 440.0 * i / 48000.0));

    int wrong = 0;
    for (int at = 0; at + 64 <= 96000; at += 64)
    {
        detector.pushSamples(std::span<const float>(x.data() + at, 64));
        const float hz = detector.getFrequencyHz();
        if (hz > 0.0f && std::abs(std::log2(hz / 440.0f)) > 1.0f / 12.0f) ++wrong;
    }
    EXPECT_EQ(wrong, 0);
}

namespace {

/// The largest distance in cents of the follower's output from `heldHz`, with
/// a stretch of `spellHz` over [48000, 48000 + spell) in a note of `heldHz`.
double worstDeviationCents(double heldHz, double spellHz, int spell, int block)
{
    PitchFollower<float> follower;
    AudioSpec spec;
    spec.sampleRate = 48000.0;
    spec.maxBlockSize = 8192;
    spec.numChannels = 1;
    follower.prepare(spec);
    follower.setGlide(0.0f);

    const int total = 72000;
    std::vector<float> x(static_cast<size_t>(total));
    double phase = 0.0;
    for (int i = 0; i < total; ++i)
    {
        const double hz = (i >= 48000 && i < 48000 + spell) ? spellHz : heldHz;
        phase += 2.0 * pi<double> * hz / 48000.0;
        x[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(phase));
    }

    double worst = 0.0;
    for (int at = 0; at < total; at += block)
    {
        follower.pushSamples(std::span<const float>(
            x.data() + at, static_cast<size_t>(std::min(block, total - at))));
        const double hz = static_cast<double>(follower.getSmoothedHz());
        if (at > 40000 && hz > 0.0)
            worst = std::max(worst, std::abs(1200.0 * std::log2(hz / heldHz)));
    }
    return worst;
}

/// Samples after the change until the follower stands on the new note.
int intervalArrival(int block)
{
    PitchFollower<float> follower;
    AudioSpec spec;
    spec.sampleRate = 48000.0;
    spec.maxBlockSize = 8192;
    spec.numChannels = 1;
    follower.prepare(spec);
    follower.setGlide(0.0f);

    const int total = 96000;
    std::vector<float> x(static_cast<size_t>(total));
    double phase = 0.0;
    for (int i = 0; i < total; ++i)
    {
        phase += 2.0 * pi<double> * (i < 48000 ? 220.0 : 660.0) / 48000.0;
        x[static_cast<size_t>(i)] = 0.4f * static_cast<float>(std::sin(phase));
    }

    for (int at = 0; at < total; at += block)
    {
        const int n = std::min(block, total - at);
        follower.pushSamples(std::span<const float>(x.data() + at, static_cast<size_t>(n)));
        const double hz = static_cast<double>(follower.getSmoothedHz());
        if (at >= 48000 && std::abs(1200.0 * std::log2(hz / 660.0)) < 50.0)
            return at + n - 48000;
    }
    return total;
}

} // namespace

// The follower judged the detector once per pushSamples() call, not once per
// detection. At blocks of 170 samples or less one reading was counted three
// times and confirmed itself, and a single bad detection moved a held 220 Hz
// note by 2614 cents; at the 512-sample hop the same reading was rejected.
DSPARK_TEST(PitchFollower_one_bad_reading_is_rejected_at_every_block_size)
{
    for (const int block : { 32, 64, 170, 512, 1024 })
        EXPECT_LT(worstDeviationCents(220.0, 996.0, 1200, block), 50.0);
}

// The same cause from the other side: above the hop it saw one detection in
// several, and a real 220 to 660 Hz interval needed three blocks instead of
// three hops, 48 ms at block 64 and 536 ms at block 8192.
DSPARK_TEST(PitchFollower_interval_latency_does_not_depend_on_block_size)
{
    const int reference = intervalArrival(64);
    for (const int block : { 512, 4096, 8192 })
        EXPECT_LT(intervalArrival(block), reference + block + 64);
}

// The whole-signal call flushed zeros after the programme. Both signals stop on
// the same sample, so that adds a step at lag 0 in both, and on a bass stem,
// which has no highs of its own, the shared click wins: the old code read this
// 2000-sample delay as -11.9 samples, confidence 0.02.
DSPARK_TEST(DelayEstimator_programme_end_is_not_a_shared_event)
{
    const int n = 48000 * 11;
    uint32_t s = 7u;
    auto gauss = [&s]() {
        double acc = 0.0;
        for (int k = 0; k < 12; ++k) { s = s * 1664525u + 1013904223u; acc += static_cast<double>(s >> 8) / 16777216.0; }
        return acc - 6.0;
    };
    constexpr int delay = 2000;
    std::vector<double> bass(static_cast<size_t>(n + delay));
    double a = 0.0, b = 0.0, c = 0.0;
    for (auto& v : bass)
    {
        a += 0.025 * (gauss() - a);
        b += 0.025 * (a - b);
        c += 0.025 * (b - c);
        v = c;
    }
    std::vector<float> ref(static_cast<size_t>(n)), del(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        ref[static_cast<size_t>(i)] = static_cast<float>(bass[static_cast<size_t>(i + delay)]);
        del[static_cast<size_t>(i)] = static_cast<float>(bass[static_cast<size_t>(i)] + 2e-5 * gauss());
    }
    const float* rp = ref.data();
    const float* dp = del.data();
    const auto r = DelayEstimator<float>::estimate(AudioBufferView<const float>(&rp, 1, n),
                                                   AudioBufferView<const float>(&dp, 1, n),
                                                   48000.0, 4800);
    EXPECT_TRUE(r.valid);
    EXPECT_NEAR(r.delaySamples, static_cast<double>(delay), 0.05);
    EXPECT_GT(r.confidence, 0.9);
    EXPECT_FALSE(r.inverted);
}
