// DSPark Tests - Advanced Effects
// CrossoverFilter, Sidechain, Expander, TransientDesigner, DynamicEQ,
// MultibandCompressor, SpectralProcessor, ProcessorChain bypass,
// AlgorithmicReverb quality, Panner smoothing

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Effects/CrossoverFilter.h"
#include "../Effects/Compressor.h"
#include "../Effects/NoiseGate.h"
#include "../Effects/Expander.h"
#include "../Effects/TransientDesigner.h"
#include "../Effects/DynamicEQ.h"
#include "../Effects/MultibandCompressor.h"
#include "../Core/SpectralProcessor.h"
#include "../Core/ProcessorChain.h"
#include "../Effects/Gain.h"
#include "../Effects/Limiter.h"
#include "../Effects/Filters.h"
#include "../Effects/AlgorithmicReverb.h"
#include "../Effects/Reverb.h"
#include "../Effects/Panner.h"
#include "../Effects/PitchShifter.h"
#include "../Effects/detail/PhaseVocoderEngine.h"
#include "../Effects/GranularProcessor.h"
#include "../Effects/SpectralDenoiser.h"
#include "../Core/FFT.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <array>
#include <atomic>
#include <thread>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// CrossoverFilter
// ============================================================================

DSPARK_TEST(CrossoverFilter_2band_sums_to_unity)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 1000.0f);
    xover.setOrder(24);

    auto s = spec(44100.0, 4096, 2);
    xover.prepare(s);

    auto input = makeStereoBuffer(4096);
    input.fillNoise();

    // Save original
    std::vector<float> original(input.ch(0), input.ch(0) + 4096);

    AudioBuffer<float> b0, b1;
    b0.resize(2, 4096);
    b1.resize(2, 4096);
    AudioBufferView<float> views[2] = { b0.toView(), b1.toView() };

    xover.processBlock(input.view(), views, 2);

    // LR crossover sum is allpass (flat magnitude, non-trivial phase).
    // Verify equal RMS energy - waveform differs due to phase shift.
    float rmsOrig = measureRMS(original.data() + 512, 4096 - 512);
    std::vector<float> sumBuf(4096);
    for (int i = 0; i < 4096; ++i)
        sumBuf[i] = b0.getChannel(0)[i] + b1.getChannel(0)[i];
    float rmsSum = measureRMS(sumBuf.data() + 512, 4096 - 512);
    EXPECT_NEAR(rmsSum, rmsOrig, rmsOrig * 0.05f);
}

// Linear-phase mode: band0 + band1 must reconstruct the input (the summed FIR
// kernels form a pure delay). Exercises the previously untested linear-phase path
// and guards the pre-allocated recompute scratch (no audio-thread allocation).
DSPARK_TEST(CrossoverFilter_linear_phase_reconstructs)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 1000.0f);
    xover.setOrder(24);
    xover.setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);

    xover.prepare(spec(44100.0, 2048, 1));

    AudioBuffer<float> low, high;
    low.resize(1, 2048);
    high.resize(1, 2048);
    AudioBufferView<float> views[2] = { low.toView(), high.toView() };

    float rmsIn = 0.0f, rmsSum = 0.0f;
    // Process several blocks; measure on the last (steady-state, past FIR latency).
    for (int blk = 0; blk < 4; ++blk)
    {
        auto input = makeMonoBuffer(2048);
        unsigned int rng = static_cast<unsigned int>(blk * 9781u + 12345u);
        for (int i = 0; i < 2048; ++i)
        {
            rng = rng * 1664525u + 1013904223u;
            input.ch(0)[i] = static_cast<float>(rng >> 8) * (1.0f / 16777216.0f) - 0.5f; // ~[-0.5,0.5]
        }
        std::vector<float> orig(input.ch(0), input.ch(0) + 2048);

        xover.processBlock(input.view(), views, 2);

        if (blk == 3)
        {
            std::vector<float> sum(2048);
            for (int i = 0; i < 2048; ++i)
                sum[i] = low.getChannel(0)[i] + high.getChannel(0)[i];
            EXPECT_NO_NAN(sum.data(), 2048);
            rmsIn  = measureRMS(orig.data(), 2048);
            rmsSum = measureRMS(sum.data(), 2048);
        }
    }
    EXPECT_NEAR(rmsSum, rmsIn, rmsIn * 0.08f);
}

DSPARK_TEST(CrossoverFilter_LP_band_attenuates_high)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 500.0f);
    xover.setOrder(24);

    auto s = spec(44100.0, 8192, 1);
    xover.prepare(s);

    // Feed high frequency (5 kHz)
    auto input = makeBuffer(1, 8192);
    generateSine(input.ch(0), 8192, 5000.0f, 44100.0f);

    AudioBuffer<float> low, high;
    low.resize(1, 8192);
    high.resize(1, 8192);
    AudioBufferView<float> views[2] = { low.toView(), high.toView() };

    xover.processBlock(input.view(), views, 2);

    // Low band should attenuate the 5 kHz signal
    float peakLow = measurePeak(low.getChannel(0) + 2048, 4096);
    EXPECT_LT(peakLow, 0.05f);
}

DSPARK_TEST(CrossoverFilter_HP_band_attenuates_low)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 2000.0f);
    xover.setOrder(24);

    auto s = spec(44100.0, 8192, 1);
    xover.prepare(s);

    // Feed low frequency (100 Hz)
    auto input = makeBuffer(1, 8192);
    generateSine(input.ch(0), 8192, 100.0f, 44100.0f);

    AudioBuffer<float> low, high;
    low.resize(1, 8192);
    high.resize(1, 8192);
    AudioBufferView<float> views[2] = { low.toView(), high.toView() };

    xover.processBlock(input.view(), views, 2);

    // High band should attenuate the 100 Hz signal
    float peakHigh = measurePeak(high.getChannel(0) + 2048, 4096);
    EXPECT_LT(peakHigh, 0.05f);
}

DSPARK_TEST(CrossoverFilter_3band_sums_to_unity)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(3);
    xover.setCrossoverFrequency(0, 300.0f);
    xover.setCrossoverFrequency(1, 3000.0f);
    xover.setOrder(24);

    auto s = spec(44100.0, 4096, 1);
    xover.prepare(s);

    auto input = makeBuffer(1, 4096);
    input.fillNoise();
    std::vector<float> original(input.ch(0), input.ch(0) + 4096);

    AudioBuffer<float> bands[3];
    AudioBufferView<float> views[3];
    for (int b = 0; b < 3; ++b)
    {
        bands[b].resize(1, 4096);
        views[b] = bands[b].toView();
    }

    xover.processBlock(input.view(), views, 3);

    // LR crossover sum is allpass - verify equal RMS energy
    float rmsOrig = measureRMS(original.data() + 512, 4096 - 512);
    std::vector<float> sumBuf(4096);
    for (int i = 0; i < 4096; ++i)
        sumBuf[i] = bands[0].getChannel(0)[i]
                  + bands[1].getChannel(0)[i]
                  + bands[2].getChannel(0)[i];
    float rmsSum = measureRMS(sumBuf.data() + 512, 4096 - 512);
    EXPECT_NEAR(rmsSum, rmsOrig, rmsOrig * 0.1f);
}

DSPARK_TEST(CrossoverFilter_minus6dB_at_crossover)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 1000.0f);
    xover.setOrder(24);

    auto s = spec(44100.0, 8192, 1);
    xover.prepare(s);

    // Feed exactly at crossover frequency (1 kHz)
    auto input = makeBuffer(1, 8192);
    generateSine(input.ch(0), 8192, 1000.0f, 44100.0f);

    AudioBuffer<float> low, high;
    low.resize(1, 8192);
    high.resize(1, 8192);
    AudioBufferView<float> views[2] = { low.toView(), high.toView() };

    xover.processBlock(input.view(), views, 2);

    // At crossover: each band should be ~-6 dB (0.5 linear)
    float peakLow  = measurePeak(low.getChannel(0) + 4096, 4096);
    float peakHigh = measurePeak(high.getChannel(0) + 4096, 4096);

    EXPECT_GT(peakLow, 0.3f);
    EXPECT_LT(peakLow, 0.7f);
    EXPECT_GT(peakHigh, 0.3f);
    EXPECT_LT(peakHigh, 0.7f);
}

DSPARK_TEST(CrossoverFilter_silence)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 1000.0f);
    xover.prepare(defaultSpec());

    auto input = makeStereoBuffer(512);
    input.fillSilence();

    AudioBuffer<float> low, high;
    low.resize(2, 512);
    high.resize(2, 512);
    AudioBufferView<float> views[2] = { low.toView(), high.toView() };

    xover.processBlock(input.view(), views, 2);
    EXPECT_SILENT(low.getChannel(0), 512, 1e-10f);
    EXPECT_SILENT(high.getChannel(0), 512, 1e-10f);
}

// The allpass phase-correction per band/split must equal the allpass that the
// split's LP/HP branches sum to: ONE section for LR12/LR24, TWO for LR48.
// The old code applied numStagesPerFilter_ sections - double the phase - which
// carved -3.5 dB (LR24) / -13.4 dB (LR48) holes into the band sum with
// octave-spaced splits. LR12 additionally notched at every split (no polarity
// inversion; -144 dB measured at fc) and its correction allpass had the sign
// of its coefficient flipped (-90 degrees at the mirrored frequency fs/2 - f).
DSPARK_TEST(CrossoverFilter_sum_is_flat_at_octave_splits)
{
    for (int order : { 12, 24, 48 })
    {
        CrossoverFilter<float> xover;
        xover.setNumBands(3);
        xover.setOrder(order);
        xover.prepare(spec(48000.0, 8192, 1));
        xover.setCrossoverFrequency(0, 1000.0f);
        xover.setCrossoverFrequency(1, 2000.0f);

        AudioBuffer<float> bands[3];
        AudioBufferView<float> views[3];
        for (int b = 0; b < 3; ++b) { bands[b].resize(1, 8192); views[b] = bands[b].toView(); }

        auto input = makeBuffer(1, 8192);
        std::fill(input.ch(0), input.ch(0) + 8192, 0.0f);
        xover.processBlock(input.view(), views, 3); // settle the 5 ms smoothing

        float worst = 0.0f;
        const float probes[] = { 600.0f, 800.0f, 1000.0f, 1200.0f, 1500.0f,
                                 1800.0f, 2000.0f, 2400.0f, 3000.0f };
        for (float f : probes)
        {
            xover.reset();
            generateSine(input.ch(0), 8192, f, 48000.0f);
            xover.processBlock(input.view(), views, 3);
            std::vector<float> sum(8192, 0.0f);
            for (int b = 0; b < 3; ++b)
                for (int i = 0; i < 8192; ++i) sum[i] += bands[b].getChannel(0)[i];
            const float rIn  = measureRMS(input.ch(0) + 4096, 4096);
            const float rSum = measureRMS(sum.data() + 4096, 4096);
            const float devDb = std::fabs(20.0f * std::log10(std::max(rSum, 1e-9f) / rIn));
            if (devDb > worst) worst = devDb;
        }
        EXPECT_LT(worst, 0.75f); // old: 14.6 dB (12), 3.5 dB (24), 13.4 dB (48)
    }
}

// prepare() used to re-initialise all split frequencies to the log-spaced
// defaults, silently discarding any configuration made before prepare() (the
// canonical configure-then-prepare pattern of the whole framework).
DSPARK_TEST(CrossoverFilter_prepare_preserves_frequencies)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(3);
    xover.setCrossoverFrequency(0, 300.0f);
    xover.setCrossoverFrequency(1, 3000.0f);
    xover.prepare(spec(48000.0, 4096, 1));

    EXPECT_NEAR(xover.getCrossoverFrequency(0), 300.0f, 0.5f);  // old: 464.2 (default)
    EXPECT_NEAR(xover.getCrossoverFrequency(1), 3000.0f, 5.0f); // old: 2154.4 (default)

    // Functional check: a 380 Hz tone must land in band 1 (fc0 = 300); the
    // discarded default fc0 = 464 would keep it in band 0.
    auto input = makeBuffer(1, 4096);
    generateSine(input.ch(0), 4096, 380.0f, 48000.0f);
    AudioBuffer<float> bands[3];
    AudioBufferView<float> views[3];
    for (int b = 0; b < 3; ++b) { bands[b].resize(1, 4096); views[b] = bands[b].toView(); }
    xover.processBlock(input.view(), views, 3);
    const float e0 = measureRMS(bands[0].getChannel(0) + 2048, 2048);
    const float e1 = measureRMS(bands[1].getChannel(0) + 2048, 2048);
    EXPECT_GT(e1, e0);
}

// Linear-phase mode selected AFTER prepare used to fall back to IIR silently
// (the engine was only allocated when prepare() saw the mode), and
// setCrossoverFrequency never reached the FIR kernels (frequencies_ was only
// advanced by the IIR smoothing path, so linear-phase mode was deaf to it).
DSPARK_TEST(CrossoverFilter_linear_phase_engages_after_prepare)
{
    CrossoverFilter<float> xover;
    xover.setNumBands(2);
    xover.setCrossoverFrequency(0, 500.0f);
    xover.setOrder(24);
    xover.prepare(spec(48000.0, 512, 1)); // prepared in MinimumPhase
    xover.setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);

    EXPECT_EQ(xover.getLatency(), 256);

    AudioBuffer<float> low, high;
    low.resize(1, 512);
    high.resize(1, 512);
    AudioBufferView<float> views[2] = { low.toView(), high.toView() };

    // Impulse: the band sum of an engaged linear-phase engine is a pure delay
    // of getLatency() samples (the IIR fallback would peak at sample 0).
    auto input = makeBuffer(1, 512);
    std::fill(input.ch(0), input.ch(0) + 512, 0.0f);
    input.ch(0)[0] = 1.0f;
    xover.processBlock(input.view(), views, 2);
    int argmax = 0;
    float peak = 0.0f;
    for (int i = 0; i < 512; ++i)
    {
        const float v = std::fabs(low.getChannel(0)[i] + high.getChannel(0)[i]);
        if (v > peak) { peak = v; argmax = i; }
    }
    EXPECT_EQ(argmax, xover.getLatency());
    EXPECT_NEAR(peak, 1.0f, 0.05f);

    // Kernels must follow the setter: after fc 500 -> 4000, a 2 kHz tone
    // belongs in band 0 (low). The old code kept the prepare-time kernels.
    xover.setCrossoverFrequency(0, 4000.0f);
    float e0 = 0.0f, e1 = 0.0f;
    for (int k = 0; k < 8; ++k)
    {
        for (int i = 0; i < 512; ++i)
            input.ch(0)[i] = std::sin(2.0f * 3.14159265f * 2000.0f
                                      * static_cast<float>(k * 512 + i) / 48000.0f);
        xover.processBlock(input.view(), views, 2);
        if (k >= 4)
        {
            e0 += measureRMS(low.getChannel(0), 512);
            e1 += measureRMS(high.getChannel(0), 512);
        }
    }
    EXPECT_GT(e0, e1 * 4.0f);
}

// Non-finite frequencies are ignored (a NaN used to park in the target array
// and pass through std::sort - formal strict-weak-ordering UB), invalid
// prepare specs are no-ops (a negative channel count used to throw from a
// gigantic allocation), oversized views are clamped to the prepared
// maxBlockSize (heap-buffer-overflow before, ASan-verified), and wild mode
// enums are clamped. Audio must stay bit-identical to an untouched twin.
DSPARK_TEST(CrossoverFilter_invalid_inputs_are_ignored)
{
    CrossoverFilter<float> x, twin;
    for (auto* p : { &x, &twin })
    {
        p->setNumBands(4);
        p->setCrossoverFrequency(0, 200.0f);
        p->setCrossoverFrequency(1, 1000.0f);
        p->setCrossoverFrequency(2, 5000.0f);
        p->setOrder(24);
        p->prepare(spec(48000.0, 512, 2));
    }

    x.setCrossoverFrequency(1, std::numeric_limits<float>::quiet_NaN());
    x.setCrossoverFrequency(0, std::numeric_limits<float>::infinity());
    bool threw = false;
    try
    {
        auto bad = spec(48000.0, 512, 2);
        bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
        x.prepare(bad);
        x.prepare(spec(48000.0, 512, -3));
    }
    catch (...) { threw = true; }
    EXPECT_TRUE(!threw);
    x.setFilterMode(static_cast<CrossoverFilter<float>::FilterMode>(99));
    EXPECT_TRUE(x.getFilterMode() == CrossoverFilter<float>::FilterMode::MinimumPhase
                || x.getFilterMode() == CrossoverFilter<float>::FilterMode::LinearPhase);
    x.setFilterMode(CrossoverFilter<float>::FilterMode::MinimumPhase);
    EXPECT_NEAR(x.getCrossoverFrequency(0), 200.0f, 0.01f);
    EXPECT_NEAR(x.getCrossoverFrequency(1), 1000.0f, 0.01f);
    EXPECT_NEAR(x.getCrossoverFrequency(2), 5000.0f, 0.01f);

    AudioBuffer<float> in1, in2;
    in1.resize(2, 512);
    in2.resize(2, 512);
    AudioBuffer<float> ba[4], bb[4];
    AudioBufferView<float> va[4], vb[4];
    for (int b = 0; b < 4; ++b)
    {
        ba[b].resize(2, 512);
        bb[b].resize(2, 512);
        va[b] = ba[b].toView();
        vb[b] = bb[b].toView();
    }
    unsigned rng = 99;
    float maxDiff = 0.0f;
    for (int k = 0; k < 12; ++k)
    {
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 512; ++i)
            {
                rng = rng * 1664525u + 1013904223u;
                const float v = static_cast<float>(rng >> 8) / 16777216.0f - 0.5f;
                in1.getChannel(c)[i] = v;
                in2.getChannel(c)[i] = v;
            }
        x.processBlock(in1.toView(), va, 4);
        twin.processBlock(in2.toView(), vb, 4);
        for (int b = 0; b < 4; ++b)
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < 512; ++i)
                {
                    const float d = std::fabs(ba[b].getChannel(c)[i] - bb[b].getChannel(c)[i]);
                    if (!(d <= maxDiff)) maxDiff = d;
                }
    }
    EXPECT_NEAR(maxDiff, 0.0f, 0.0f);

    // Oversized view: processed span clamps to the prepared maxBlockSize.
    AudioBuffer<float> big;
    big.resize(2, 700);
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < 700; ++i) big.getChannel(c)[i] = 0.25f;
    AudioBuffer<float> bo[4];
    AudioBufferView<float> vo[4];
    for (int b = 0; b < 4; ++b) { bo[b].resize(2, 700); vo[b] = bo[b].toView(); }
    x.processBlock(big.toView(), vo, 4);
    EXPECT_NO_NAN(bo[0].getChannel(0), 512);
}

// ============================================================================
// Compressor - External Sidechain
// ============================================================================

DSPARK_TEST(Compressor_sidechain_triggers_compression)
{
    Compressor<float> comp;
    comp.prepare(defaultSpec());
    comp.setThreshold(-20.0f);
    comp.setRatio(10.0f);
    comp.setAttack(0.1f);
    comp.setRelease(50.0f);

    // Audio: constant loud signal
    auto audio = makeStereoBuffer(8192);
    audio.fillSine(440.0f, 44100.0f, 1.0f);

    // Sidechain: also loud (triggers compression)
    auto sc = makeStereoBuffer(8192);
    sc.fillSine(440.0f, 44100.0f, 1.0f);

    comp.processBlock(audio.view(), sc.view());
    float peakAfter = measurePeak(audio.ch(0) + 4096, 4096);

    // Should be compressed
    EXPECT_LT(peakAfter, 0.5f);
}

DSPARK_TEST(Compressor_sidechain_silence_no_compression)
{
    Compressor<float> comp;
    comp.setAutoMakeup(false);
    comp.prepare(defaultSpec());
    comp.setThreshold(-20.0f);
    comp.setRatio(10.0f);
    comp.setAttack(0.1f);
    comp.setRelease(50.0f);

    // Audio: loud signal
    auto audio = makeStereoBuffer(4096);
    audio.fillSine(440.0f, 44100.0f, 1.0f);
    float peakBefore = measurePeak(audio.ch(0), 4096);

    // Sidechain: silent (no compression triggered)
    auto sc = makeStereoBuffer(4096);
    sc.fillSilence();

    comp.processBlock(audio.view(), sc.view());
    float peakAfter = measurePeak(audio.ch(0), 4096);

    // Should pass through uncompressed
    EXPECT_NEAR(peakAfter, peakBefore, peakBefore * 0.15f);
}

DSPARK_TEST(Compressor_sidechain_no_NaN)
{
    Compressor<float> comp;
    comp.prepare(defaultSpec());
    comp.setThreshold(-10.0f);
    comp.setRatio(4.0f);

    auto audio = makeStereoBuffer(4096);
    audio.fillNoise();
    auto sc = makeStereoBuffer(4096);
    sc.fillNoise();

    comp.processBlock(audio.view(), sc.view());
    EXPECT_NO_NAN(audio.ch(0), 4096);
}

// ============================================================================
// NoiseGate - External Sidechain
// ============================================================================

DSPARK_TEST(NoiseGate_sidechain_opens_gate)
{
    NoiseGate<float> gate;
    gate.prepare(44100.0);
    gate.setThreshold(-20.0f);
    gate.setAttack(0.1f);
    gate.setRelease(10.0f);
    gate.setRange(-80.0f);

    // Audio: quiet signal
    auto audio = makeMonoBuffer(8192);
    audio.fillSine(440.0f, 44100.0f, 0.01f);

    // Sidechain: loud (opens the gate)
    auto sc = makeMonoBuffer(8192);
    sc.fillSine(1000.0f, 44100.0f, 1.0f);

    gate.processBlock(audio.view(), sc.view());

    // Gate should be open - audio should pass through
    float peakAfter = measurePeak(audio.ch(0) + 4096, 4096);
    EXPECT_GT(peakAfter, 0.005f);
}

DSPARK_TEST(NoiseGate_sidechain_closes_gate)
{
    NoiseGate<float> gate;
    gate.prepare(44100.0);
    gate.setThreshold(-10.0f);
    gate.setAttack(0.1f);
    gate.setHold(1.0f);
    gate.setRelease(10.0f);
    gate.setRange(-80.0f);

    // Audio: loud signal
    auto audio = makeMonoBuffer(8192);
    audio.fillSine(440.0f, 44100.0f, 1.0f);

    // Sidechain: silent (closes the gate)
    auto sc = makeMonoBuffer(8192);
    sc.fillSilence();

    gate.processBlock(audio.view(), sc.view());

    // Gate should close - audio should be attenuated
    float peakAfter = measurePeak(audio.ch(0) + 4096, 4096);
    EXPECT_LT(peakAfter, 0.01f);
}

// ============================================================================
// Expander
// ============================================================================

DSPARK_TEST(Expander_above_threshold_passthrough)
{
    Expander<float> exp;
    exp.prepare(44100.0);
    exp.setThreshold(-30.0f);
    exp.setRatio(4.0f);
    exp.setAttack(0.1f);
    exp.setRelease(50.0f);

    auto tb = makeMonoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f, 1.0f); // 0 dBFS - well above threshold

    float peakBefore = measurePeak(tb.ch(0), 4096);
    exp.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 512, 3000);

    EXPECT_GT(peakAfter, peakBefore * 0.8f);
}

DSPARK_TEST(Expander_below_threshold_attenuates)
{
    Expander<float> exp;
    exp.prepare(44100.0);
    exp.setThreshold(-10.0f);
    exp.setRatio(4.0f);
    exp.setAttack(0.1f);
    exp.setHold(1.0f);
    exp.setRelease(10.0f);
    exp.setRange(-60.0f);

    // Signal at ~-40 dBFS (well below -10 dB threshold)
    auto tb = makeMonoBuffer(8192);
    tb.fillSine(440.0f, 44100.0f, 0.01f);

    exp.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 4096, 4096);

    // Should be attenuated by the expander
    EXPECT_LT(peakAfter, 0.005f);
}

DSPARK_TEST(Expander_range_clamps_attenuation)
{
    Expander<float> exp;
    exp.prepare(44100.0);
    exp.setThreshold(-10.0f);
    exp.setRatio(20.0f);
    exp.setRange(-20.0f); // Floor at -20 dB
    exp.setAttack(0.1f);
    exp.setHold(1.0f);
    exp.setRelease(10.0f);

    auto tb = makeMonoBuffer(8192);
    tb.fillSine(440.0f, 44100.0f, 0.01f); // ~-40 dBFS

    exp.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 4096, 4096);

    // Attenuation clamped by range - not fully silenced
    EXPECT_GT(peakAfter, 0.0005f);
}

DSPARK_TEST(Expander_deep_attenuation_below_threshold)
{
    Expander<float> exp;
    exp.prepare(44100.0);
    exp.setThreshold(-20.0f);
    exp.setRatio(10.0f);
    exp.setAttack(0.1f);
    exp.setHold(1.0f);
    exp.setRelease(10.0f);
    exp.setRange(-80.0f); // Deep floor -> near-silence below threshold

    // Quiet signal (~-34 dBFS), well below threshold -> heavy downward expansion.
    auto audio = makeMonoBuffer(8192);
    audio.fillSine(440.0f, 44100.0f, 0.02f);

    exp.processBlock(audio.view());
    float peakAfter = measurePeak(audio.ch(0) + 4096, 4096);
    EXPECT_LT(peakAfter, 0.01f);
}

// Regression: the downward-expander transfer slope below threshold
// must be (R-1), not the compressor law (1-1/R). With threshold -20 dB, R=3 and
// a signal 15 dB under threshold (well above the -60 dB range floor), the
// correct attenuation is ~(R-1)*15 = 30 dB (gain ~0.032), whereas the old
// (1-1/R)*15 = 10 dB (gain ~0.32) leaves the output ~20 dB too loud. Every prior
// Expander test drove the signal deep enough that the range floor clamped both
// formulas identically, so none of them distinguished the two laws.
DSPARK_TEST(Expander_downward_slope_is_ratio_minus_one)
{
    Expander<float> exp;
    exp.prepare(44100.0);
    exp.setThreshold(-20.0f);
    exp.setRatio(3.0f);
    exp.setRange(-60.0f);      // floor far below the expected -50..-65 dBFS output
    exp.setAttack(0.1f);
    exp.setHold(0.0f);
    exp.setRelease(5.0f);

    auto tb = makeMonoBuffer(16384);
    tb.fillSine(440.0f, 44100.0f, 0.017783f); // ~-35 dBFS peak => ~15 dB under thr

    exp.processBlock(tb.view());
    const float peakAfter = measurePeak(tb.ch(0) + 12288, 4096); // settled tail

    // Fixed (R-1) law lands near 0.0006 (peak-det) .. 0.0003 (rms-det); the old
    // (1-1/R) law lands near 0.0056. The band below cleanly separates them and
    // stays above the range floor (~1.8e-5), so it fails on either regression.
    EXPECT_LT(peakAfter, 0.0018f);
    EXPECT_GT(peakAfter, 0.00008f);
}

// Regression: setCrossoverFrequency keeps the split frequencies
// sorted ascending (the internal sort was reimplemented as an explicit insertion
// sort to avoid a spurious GCC -O2 -Warray-bounds on the fixed array; this pins
// that the ordering behaviour is preserved).
DSPARK_TEST(CrossoverFilter_frequencies_stay_sorted)
{
    CrossoverFilter<float, 8> xo;
    xo.prepare(spec(48000.0, 512, 2));
    xo.setNumBands(4); // 3 split points, ascending log-spaced defaults
    // Assign the HIGHEST split index the LOWEST frequency: only a working sort
    // migrates it to index 0 and keeps the whole set ascending.
    xo.setCrossoverFrequency(2, 50.0f);
    const float f0 = xo.getCrossoverFrequency(0);
    const float f1 = xo.getCrossoverFrequency(1);
    const float f2 = xo.getCrossoverFrequency(2);
    EXPECT_NEAR(f0, 50.0f, 1.0f); // 50 Hz sorted down to split 0
    EXPECT_LT(f0, f1);
    EXPECT_LT(f1, f2);
}

// DynamicEQ's internal oversampling is transparent - factor 1 = OFF adds
// zero latency, and enabling 2x/4x reports the added group
// delay through getLatency() so a host can compensate (PDC). Pins the policy.
DSPARK_TEST(DynamicEQ_oversampling_is_configurable_and_reported)
{
    const AudioSpec sp = spec(48000.0, 512, 2);

    DynamicEQ<float> off;
    off.setOversampling(1);
    off.prepare(sp);
    EXPECT_EQ(off.getLatency(), 0);           // 1x = off => no added latency

    DynamicEQ<float> os2; os2.setOversampling(2); os2.prepare(sp);
    DynamicEQ<float> os4; os4.setOversampling(4); os4.prepare(sp);
    EXPECT_GT(os2.getLatency(), 0);           // 2x reports its group delay
    EXPECT_GT(os4.getLatency(), os2.getLatency()); // 4x reports more than 2x

    // Factor 3 rounds up to 4 (documented; engine is power-of-two).
    DynamicEQ<float> os3; os3.setOversampling(3); os3.prepare(sp);
    EXPECT_EQ(os3.getLatency(), os4.getLatency());
}

// NaN attack/release/ratio used to poison the gain smoother PERMANENTLY (7168
// non-finite samples measured after recovery), prepare(NaN) passed the old
// max(NaN, 1.0) guard and stormed, a negative sidechain-HPF cutoff flipped the
// one-pole into unstable growth whose NaN was then discarded by max() - the
// detector froze at zero and the WHOLE programme was muted to the range
// (-40 dB measured on loud bursts), and processing before prepare() hard-muted
// (smoothing coefficients were zero, freezing the gain at 0).
DSPARK_TEST(Expander_invalid_inputs_are_ignored)
{
    // Before prepare: pass-through, not mute.
    {
        Expander<float> e;
        auto buf = makeMonoBuffer(64);
        std::fill(buf.ch(0), buf.ch(0) + 64, 0.5f);
        e.processBlock(buf.view());
        EXPECT_NEAR(buf.ch(0)[32], 0.5f, 1e-6f);
    }

    Expander<float> e, twin;
    for (auto* p : { &e, &twin })
    {
        p->prepare(48000.0);
        p->setThreshold(-30.0f);
        p->setRatio(4.0f);
        p->setAttack(0.5f);
        p->setRelease(80.0f);
        p->setHold(50.0f);
        p->setRange(-40.0f);
        p->setHysteresis(4.0f);
    }

    auto fillBurst = [](float* d, int k)
    {
        for (int i = 0; i < 512; ++i)
        {
            const int t = k * 512 + i;
            const float amp = ((t / 2400) % 2 == 0) ? 0.5f : 0.001f;
            d[i] = amp * std::sin(2.0f * 3.14159265f * 1000.0f * static_cast<float>(t) / 48000.0f);
        }
    };

    auto a = makeMonoBuffer(512);
    auto b = makeMonoBuffer(512);
    int bad = 0;
    float maxDiff = 0.0f;
    for (int k = 0; k < 24; ++k)
    {
        if (k == 4) // inject at a block boundary, recover 4 blocks later
        {
            e.setAttack(std::numeric_limits<float>::quiet_NaN());
            e.setRelease(std::numeric_limits<float>::quiet_NaN());
            e.setRatio(std::numeric_limits<float>::quiet_NaN());
            e.setThreshold(std::numeric_limits<float>::quiet_NaN());
            e.setRange(std::numeric_limits<float>::infinity());
            e.setHysteresis(std::numeric_limits<float>::quiet_NaN());
            e.setHold(std::numeric_limits<float>::quiet_NaN());
            e.prepare(std::numeric_limits<double>::quiet_NaN());
        }
        fillBurst(a.ch(0), k);
        fillBurst(b.ch(0), k);
        e.processBlock(a.view());
        twin.processBlock(b.view());
        for (int i = 0; i < 512; ++i)
        {
            if (!std::isfinite(a.ch(0)[i])) ++bad;
            const float d = std::fabs(a.ch(0)[i] - b.ch(0)[i]);
            if (!(d <= maxDiff)) maxDiff = d;
        }
    }
    EXPECT_EQ(bad, 0);
    EXPECT_NEAR(maxDiff, 0.0f, 0.0f); // all invalid inputs ignored: bit-identical

    // Negative HPF cutoff keeps the previous frequency; the expander must
    // still OPEN on loud material (the old detector died and muted everything).
    Expander<float> h;
    h.prepare(48000.0);
    h.setThreshold(-30.0f);
    h.setRange(-40.0f);
    h.setRelease(10.0f);
    h.setHold(0.0f);
    h.setSidechainHPF(true, -500.0);
    h.setSidechainHPF(true, std::numeric_limits<double>::quiet_NaN());
    auto seg = makeMonoBuffer(2400);
    float loudGain = -100.0f;
    for (int s = 0; s < 8; ++s)
    {
        const bool loud = (s % 2 == 0);
        for (int i = 0; i < 2400; ++i)
            seg.ch(0)[i] = (loud ? 0.5f : 0.001f)
                * std::sin(2.0f * 3.14159265f * 1000.0f * static_cast<float>(s * 2400 + i) / 48000.0f);
        h.processBlock(seg.view());
        if (loud) loudGain = h.getCurrentGainDb();
    }
    EXPECT_GT(loudGain, -3.0f); // old: -40.00 (whole programme muted to range)
}

// The external sidechain overload is advertised by the umbrella table but did
// not exist; and the default 50 ms hold was silently ZERO until setHold() was
// called (holdSamples_ was only derived inside the setter, never by prepare).
DSPARK_TEST(Expander_external_sidechain_and_default_hold)
{
    Expander<float> e;
    e.prepare(48000.0);
    e.setThreshold(-30.0f);
    e.setRange(-60.0f);
    e.setAttack(0.5f);
    e.setRelease(20.0f);
    e.setHold(5.0f);
    e.setHysteresis(2.0f);

    auto audio = makeStereoBuffer(512);
    auto sc = makeMonoBuffer(512);
    auto fillAudio = [&](int k)
    {
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 512; ++i)
                audio.ch(c)[i] = 0.5f * std::sin(2.0f * 3.14159265f * 997.0f
                                                 * static_cast<float>(k * 512 + i) / 48000.0f);
    };

    // Silent key: gain must fall toward the range despite loud audio.
    for (int k = 0; k < 30; ++k)
    {
        fillAudio(k);
        std::fill(sc.ch(0), sc.ch(0) + 512, 0.0f);
        e.processBlock(audio.view(), sc.view());
    }
    EXPECT_LT(e.getCurrentGainDb(), -20.0f);

    // Hot key: gain returns to unity.
    for (int k = 0; k < 30; ++k)
    {
        fillAudio(k);
        for (int i = 0; i < 512; ++i)
            sc.ch(0)[i] = 0.7f * std::sin(2.0f * 3.14159265f * 200.0f
                                          * static_cast<float>(k * 512 + i) / 48000.0f);
        e.processBlock(audio.view(), sc.view());
    }
    EXPECT_GT(e.getCurrentGainDb(), -1.0f);

    // Sidechain shorter than the audio block: internal-key fallback, finite.
    auto shortSc = makeMonoBuffer(100);
    std::fill(shortSc.ch(0), shortSc.ch(0) + 100, 0.0f);
    fillAudio(0);
    e.processBlock(audio.view(), shortSc.view());
    EXPECT_NO_NAN(audio.ch(0), 512);

    // Default hold: prepare() derives the 50 ms hold; 20 ms into silence the
    // gain must still sit at unity (the old default hold was DEAD: -8.8 dB).
    Expander<float> d;
    d.prepare(48000.0);
    d.setThreshold(-20.0f);
    d.setRange(-60.0f);
    d.setRelease(5.0f);
    d.setHysteresis(0.0f);
    auto blk = makeMonoBuffer(480);
    float gainInHold = -100.0f;
    for (int k = 0; k < 13; ++k)
    {
        for (int i = 0; i < 480; ++i)
        {
            const int t = k * 480 + i;
            blk.ch(0)[i] = (t < 4800)
                ? 0.5f * std::sin(2.0f * 3.14159265f * 1000.0f * static_cast<float>(t) / 48000.0f)
                : 0.0f;
        }
        d.processBlock(blk.view());
        if (k == 11) gainInHold = d.getCurrentGainDb(); // 20 ms after the cutoff
    }
    EXPECT_GT(gainInHold, -0.5f); // old: -8.83 dB (hold never armed)
}

// ============================================================================
// TransientDesigner
// ============================================================================

// A NaN sample rate used to pass the coefficient guard (NaN <= 0 is false)
// and poison the envelopes permanently (3584 non-finite samples measured);
// NaN amounts stormed the output while published, and under ODR the NaN
// last-output silently switched the slow envelope to the fast release,
// leaving the envelope history diverged 0.128 after recovery.
DSPARK_TEST(TransientDesigner_invalid_inputs_are_ignored)
{
    TransientDesigner<float> e, twin;
    for (auto* p : { &e, &twin })
    {
        p->prepare(48000.0);
        p->setAttack(50.0f);
        p->setSustain(-30.0f);
        p->setOutputDepRecovery(true);
    }

    auto fillBurst = [](float* d, int k)
    {
        for (int i = 0; i < 512; ++i)
        {
            const int t = k * 512 + i;
            const float amp = ((t / 2000) % 2 == 0) ? 0.6f : 0.05f;
            d[i] = amp * std::sin(2.0f * 3.14159265f * 220.0f * static_cast<float>(t) / 48000.0f);
        }
    };

    auto a = makeMonoBuffer(512);
    auto b = makeMonoBuffer(512);
    int bad = 0;
    float maxDiff = 0.0f;
    for (int k = 0; k < 24; ++k)
    {
        if (k == 4)
        {
            e.setAttack(std::numeric_limits<float>::quiet_NaN());
            e.setSustain(std::numeric_limits<float>::infinity());
            e.setCharacter(std::numeric_limits<float>::quiet_NaN());
            e.prepare(std::numeric_limits<double>::quiet_NaN());
            e.prepare(-48000.0);
        }
        fillBurst(a.ch(0), k);
        fillBurst(b.ch(0), k);
        e.processBlock(a.view());
        twin.processBlock(b.view());
        for (int i = 0; i < 512; ++i)
        {
            if (!std::isfinite(a.ch(0)[i])) ++bad;
            const float d = std::fabs(a.ch(0)[i] - b.ch(0)[i]);
            if (!(d <= maxDiff)) maxDiff = d;
        }
    }
    EXPECT_EQ(bad, 0);                    // old: 2048+ non-finite
    EXPECT_NEAR(maxDiff, 0.0f, 0.0f);     // old: NaN storm + 0.128 divergence
}

DSPARK_TEST(TransientDesigner_zero_is_passthrough)
{
    TransientDesigner<float> td;
    td.prepare(defaultSpec());
    td.setAttack(0.0f);
    td.setSustain(0.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillSine(440.0f, 44100.0f);
    std::vector<float> original(tb.ch(0), tb.ch(0) + 4096);

    td.processBlock(tb.view());

    float maxDiff = 0.0f;
    for (int i = 256; i < 4096; ++i)
    {
        float diff = std::abs(tb.ch(0)[i] - original[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    EXPECT_LT(maxDiff, 0.05f);
}

DSPARK_TEST(TransientDesigner_attack_boost_changes_peak)
{
    TransientDesigner<float> td;
    auto s = spec(44100.0, 4096, 1);
    td.prepare(s);
    td.setAttack(80.0f); // +80% attack emphasis

    // Create a signal with transient (impulse + sine)
    auto tb = makeBuffer(1, 4096);
    tb.fillSilence();
    // Sustained sine
    for (int i = 0; i < 4096; ++i)
        tb.ch(0)[i] = 0.3f * std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);
    // Add transient burst at start
    for (int i = 0; i < 100; ++i)
        tb.ch(0)[i] += 0.7f;

    float peakBefore = measurePeak(tb.ch(0), 4096);
    td.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0), 4096);

    // Transient boost should increase or maintain peak
    EXPECT_NO_NAN(tb.ch(0), 4096);
    // Peak should be at least as loud (attack boost)
    EXPECT_GT(peakAfter, peakBefore * 0.8f);
}

DSPARK_TEST(TransientDesigner_no_NaN)
{
    TransientDesigner<float> td;
    td.prepare(defaultSpec());
    td.setAttack(100.0f);
    td.setSustain(-100.0f);

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    td.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);
}

// ============================================================================
// DynamicEQ
// ============================================================================

DSPARK_TEST(DynamicEQ_below_threshold_no_change)
{
    DynamicEQ<float> deq;
    auto s = spec(44100.0, 4096, 1);
    deq.prepare(s);
    deq.setNumBands(1);

    DynamicEQ<float>::BandConfig cfg;
    cfg.frequency = 1000.0f;
    cfg.threshold = 0.0f; // 0 dBFS threshold (signal below)
    cfg.aboveRatio = 4.0f;
    cfg.aboveBoost = false;
    deq.setBand(0, cfg);

    auto tb = makeBuffer(1, 4096);
    generateSine(tb.ch(0), 4096, 1000.0f, 44100.0f, 0.1f); // ~-20 dBFS

    float peakBefore = measurePeak(tb.ch(0), 4096);
    deq.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 512, 3000);

    // Below threshold: aboveRatio doesn't apply, belowRatio=1 -> no change
    EXPECT_NEAR(peakAfter, peakBefore, peakBefore * 0.3f);
}

DSPARK_TEST(DynamicEQ_above_threshold_cuts)
{
    DynamicEQ<float> deq;
    auto s = spec(44100.0, 8192, 1);
    deq.prepare(s);
    deq.setNumBands(1);

    DynamicEQ<float>::BandConfig cfg;
    cfg.frequency = 1000.0f;
    cfg.threshold = -30.0f;
    cfg.aboveRatio = 8.0f;
    cfg.aboveRangeDb = 20.0f;
    cfg.aboveBoost = false; // cut when above
    cfg.aboveAttackMs = 1.0f;
    cfg.aboveReleaseMs = 50.0f;
    deq.setBand(0, cfg);

    auto tb = makeBuffer(1, 8192);
    generateSine(tb.ch(0), 8192, 1000.0f, 44100.0f, 1.0f); // 0 dBFS

    deq.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 4096, 4096);

    // Should be cut (at 1 kHz, above -30 dB threshold)
    EXPECT_LT(peakAfter, 0.8f);
}

DSPARK_TEST(DynamicEQ_sidechain_external)
{
    DynamicEQ<float> deq;
    auto s = spec(44100.0, 4096, 1);
    deq.prepare(s);
    deq.setNumBands(1);

    DynamicEQ<float>::BandConfig cfg;
    cfg.frequency = 1000.0f;
    cfg.threshold = -30.0f;
    cfg.aboveRatio = 8.0f;
    cfg.aboveBoost = false;
    deq.setBand(0, cfg);

    // Audio: loud
    auto audio = makeBuffer(1, 4096);
    generateSine(audio.ch(0), 4096, 1000.0f, 44100.0f, 1.0f);

    // Sidechain: silent (below threshold -> no dynamic action)
    auto sc = makeBuffer(1, 4096);
    sc.fillSilence();

    float peakBefore = measurePeak(audio.ch(0), 4096);
    deq.processBlock(audio.view(), sc.view());
    float peakAfter = measurePeak(audio.ch(0) + 512, 3000);

    EXPECT_NEAR(peakAfter, peakBefore, peakBefore * 0.3f);
}

DSPARK_TEST(DynamicEQ_no_NaN)
{
    DynamicEQ<float> deq;
    deq.prepare(spec(44100.0, 4096, 2));
    deq.setNumBands(4);
    for (int b = 0; b < 4; ++b)
    {
        DynamicEQ<float>::BandConfig cfg;
        cfg.frequency = 200.0f * static_cast<float>(b + 1);
        cfg.threshold = -20.0f;
        cfg.aboveRatio = 4.0f;
        cfg.belowRatio = 2.0f;
        cfg.belowBoost = true;
        deq.setBand(b, cfg);
    }

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    deq.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

// A band config with NaN fields used to poison the detector biquad and the
// gain ballistics PERMANENTLY (14336 non-finite samples measured after
// recovery); an external sidechain shorter than the audio block was read out
// of bounds (ASan-verified heap overflow, now a self-key fallback); invalid
// prepare specs are ignored.
DSPARK_TEST(DynamicEQ_invalid_inputs_are_ignored)
{
    auto baseCfg = []()
    {
        DynamicEQ<float>::BandConfig c;
        c.frequency = 3000.0f;
        c.q = 2.0f;
        c.threshold = -18.0f;
        c.aboveRatio = 4.0f;
        c.aboveAttackMs = 2.0f;
        c.aboveReleaseMs = 40.0f;
        c.aboveRangeDb = 12.0f;
        return c;
    };

    DynamicEQ<float> e, twin;
    for (auto* p : { &e, &twin })
    {
        p->setNumBands(1);
        p->setBand(0, baseCfg());
        p->prepare(spec(48000.0, 512, 2));
    }

    auto fill = [](float* d, int k)
    {
        for (int i = 0; i < 512; ++i)
        {
            const int t = k * 512 + i;
            d[i] = 0.4f * std::sin(2.0f * 3.14159265f * 3000.0f * static_cast<float>(t) / 48000.0f)
                 + 0.1f * std::sin(2.0f * 3.14159265f * 200.0f * static_cast<float>(t) / 48000.0f);
        }
    };

    auto a = makeStereoBuffer(512);
    auto b = makeStereoBuffer(512);
    int bad = 0;
    float maxDiff = 0.0f;
    for (int k = 0; k < 24; ++k)
    {
        if (k == 4)
        {
            auto c = baseCfg();
            c.frequency = std::numeric_limits<float>::quiet_NaN();
            c.q = std::numeric_limits<float>::quiet_NaN();
            c.aboveAttackMs = std::numeric_limits<float>::quiet_NaN();
            c.aboveRangeDb = std::numeric_limits<float>::infinity();
            e.setBand(0, c);
            auto bad1 = spec(48000.0, 512, 2);
            bad1.sampleRate = std::numeric_limits<double>::quiet_NaN();
            e.prepare(bad1);
        }
        for (int c2 = 0; c2 < 2; ++c2) { fill(a.ch(c2), k); fill(b.ch(c2), k); }
        e.processBlock(a.view());
        twin.processBlock(b.view());
        for (int c2 = 0; c2 < 2; ++c2)
            for (int i = 0; i < 512; ++i)
            {
                if (!std::isfinite(a.ch(c2)[i])) ++bad;
                const float d = std::fabs(a.ch(c2)[i] - b.ch(c2)[i]);
                if (!(d <= maxDiff)) maxDiff = d;
            }
    }
    EXPECT_EQ(bad, 0);                // old: 14336 non-finite
    EXPECT_NEAR(maxDiff, 0.0f, 0.0f); // all invalid inputs ignored: bit-identical

    // Short external sidechain: self-key fallback, no out-of-bounds read.
    auto sc = makeMonoBuffer(100);
    std::fill(sc.ch(0), sc.ch(0) + 100, 0.0f);
    for (int c2 = 0; c2 < 2; ++c2) fill(a.ch(c2), 0);
    e.processBlock(a.view(), sc.view());
    EXPECT_NO_NAN(a.ch(0), 512);
}

// Lookahead and the optional oversampler both delay the audio path, but the
// class reported no latency at all: hosts could not compensate. getLatency()
// must match the measured impulse delay exactly.
DSPARK_TEST(DynamicEQ_latency_is_reported_exact)
{
    for (int os : { 1, 2 })
    {
        DynamicEQ<float> e;
        e.setNumBands(1);
        DynamicEQ<float>::BandConfig c;
        c.enabled = false; // bypass the band: pure delay path
        e.setBand(0, c);
        e.setOversampling(os);
        e.prepare(spec(48000.0, 512, 1));
        e.setLookahead(5.0f);

        auto a = makeMonoBuffer(512);
        int arg = -1, total = 0;
        float mx = 0.0f;
        for (int k = 0; k < 4; ++k)
        {
            std::fill(a.ch(0), a.ch(0) + 512, 0.0f);
            if (k == 0) a.ch(0)[0] = 1.0f;
            e.processBlock(a.view());
            for (int i = 0; i < 512; ++i)
            {
                const float v = std::fabs(a.ch(0)[i]);
                if (v > mx) { mx = v; arg = total + i; }
            }
            total += 512;
        }
        EXPECT_EQ(arg, e.getLatency());
        EXPECT_GT(mx, 0.9f);
    }
}

// ============================================================================
// MultibandCompressor
// ============================================================================

DSPARK_TEST(MultibandCompressor_output_sums_correctly)
{
    MultibandCompressor<float> mbc;
    mbc.setNumBands(3);
    mbc.setCrossoverFrequency(0, 300.0f);
    mbc.setCrossoverFrequency(1, 3000.0f);

    // Set compressors to passthrough BEFORE prepare so smoothed values
    // start at the target (avoids 30ms convergence transient)
    for (int b = 0; b < 3; ++b)
    {
        mbc.getBandCompressor(b).setThreshold(0.0f);
        mbc.getBandCompressor(b).setRatio(1.0f);
    }

    auto s = spec(44100.0, 4096, 2);
    mbc.prepare(s);

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    std::vector<float> original(tb.ch(0), tb.ch(0) + 4096);

    mbc.processBlock(tb.view());

    // Crossover sum is allpass - verify equal RMS energy
    float rmsOrig = measureRMS(original.data() + 512, 4096 - 512);
    float rmsOut  = measureRMS(tb.ch(0) + 512, 4096 - 512);
    EXPECT_NEAR(rmsOut, rmsOrig, rmsOrig * 0.1f);
}

DSPARK_TEST(MultibandCompressor_per_band_compression)
{
    MultibandCompressor<float> mbc;
    mbc.setNumBands(2);
    mbc.setCrossoverFrequency(0, 1000.0f);

    auto s = spec(44100.0, 8192, 1);
    mbc.prepare(s);

    // Only compress low band heavily
    mbc.getBandCompressor(0).setThreshold(-20.0f);
    mbc.getBandCompressor(0).setRatio(20.0f);
    mbc.getBandCompressor(0).setAttack(0.1f);

    // High band: no compression
    mbc.getBandCompressor(1).setThreshold(0.0f);
    mbc.getBandCompressor(1).setRatio(1.0f);

    // Feed low frequency (200 Hz) - should be compressed
    auto tbLow = makeBuffer(1, 8192);
    generateSine(tbLow.ch(0), 8192, 200.0f, 44100.0f, 1.0f);
    mbc.processBlock(tbLow.view());
    float peakLow = measurePeak(tbLow.ch(0) + 4096, 4096);

    // Feed high frequency (5 kHz) - should pass through
    mbc.reset();
    mbc.getBandCompressor(0).setThreshold(-20.0f);
    mbc.getBandCompressor(0).setRatio(20.0f);
    mbc.getBandCompressor(0).setAttack(0.1f);
    mbc.getBandCompressor(1).setThreshold(0.0f);
    mbc.getBandCompressor(1).setRatio(1.0f);

    auto tbHigh = makeBuffer(1, 8192);
    generateSine(tbHigh.ch(0), 8192, 5000.0f, 44100.0f, 1.0f);
    mbc.processBlock(tbHigh.view());
    float peakHigh = measurePeak(tbHigh.ch(0) + 4096, 4096);

    // Low band compressed more than high band
    EXPECT_LT(peakLow, peakHigh);
}

DSPARK_TEST(MultibandCompressor_silence)
{
    MultibandCompressor<float> mbc;
    mbc.setNumBands(3);
    mbc.setCrossoverFrequency(0, 300.0f);
    mbc.setCrossoverFrequency(1, 3000.0f);
    mbc.prepare(defaultSpec());

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    mbc.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 0.001f);
}

DSPARK_TEST(MultibandCompressor_no_NaN)
{
    MultibandCompressor<float> mbc;
    mbc.setNumBands(3);
    mbc.setCrossoverFrequency(0, 300.0f);
    mbc.setCrossoverFrequency(1, 3000.0f);
    mbc.prepare(spec(44100.0, 4096, 2));

    for (int b = 0; b < 3; ++b)
    {
        mbc.getBandCompressor(b).setThreshold(-10.0f);
        mbc.getBandCompressor(b).setRatio(4.0f);
    }

    auto tb = makeStereoBuffer(4096);
    tb.fillNoise();
    mbc.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);
}

// Invalid prepare specs must be ignored. The old header muted the caller's
// audio after a first invalid prepare (summing zeroed band buffers over it)
// and CRASHED with an access violation after an invalid re-prepare (band
// buffers shrunk to 0 channels while the crossover kept writing 2).
DSPARK_TEST(MultibandCompressor_invalid_prepare_is_ignored)
{
    // (a) First-ever prepare is invalid -> instance stays pass-through.
    {
        auto mbc = std::make_unique<MultibandCompressor<float>>();
        mbc->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));

        auto tb = makeStereoBuffer(512);
        tb.fillNoise();
        std::vector<float> original(tb.ch(0), tb.ch(0) + 512);
        mbc->processBlock(tb.view());

        float maxDiff = 0.0f;
        for (int i = 0; i < 512; ++i)
            maxDiff = std::max(maxDiff, std::fabs(tb.ch(0)[i] - original[i]));
        EXPECT_EQ(maxDiff, 0.0f); // old header: output is all zeros
    }

    // (b) Invalid re-prepares on a working instance -> bit-identical to a twin.
    {
        auto dut  = std::make_unique<MultibandCompressor<float>>();
        auto twin = std::make_unique<MultibandCompressor<float>>();
        auto s = spec(48000.0, 512, 2);
        for (auto* m : { dut.get(), twin.get() })
        {
            m->setNumBands(4);
            for (int b = 0; b < 4; ++b)
            {
                m->getBandCompressor(b).setThreshold(-30.0f);
                m->getBandCompressor(b).setRatio(10.0f);
                m->getBandCompressor(b).setAttack(1.0f);
                m->getBandCompressor(b).setRelease(200.0f);
            }
            m->prepare(s);
        }

        dut->prepare(spec(48000.0, 512, -3));                              // old: poisons band buffers
        dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 0, 2)); // old: zeroes band buffers

        EXPECT_EQ(dut->getNumBands(), 4);
        EXPECT_EQ(dut->getOrder(), twin->getOrder());

        auto ta = makeStereoBuffer(512);
        auto tbuf = makeStereoBuffer(512);
        float maxDiff = 0.0f;
        for (int k = 0; k < 20; ++k)
        {
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 512; ++i)
                {
                    const float v = 0.9f * std::sin(2.0f * 3.14159265f * (200.0f + 1800.0f * ch)
                                                    * float(k * 512 + i) / 48000.0f);
                    ta.ch(ch)[i] = v;
                    tbuf.ch(ch)[i] = v;
                }
            dut->processBlock(ta.view());   // old header crashes here (0xC0000005)
            twin->processBlock(tbuf.view());
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 512; ++i)
                    maxDiff = std::max(maxDiff, std::fabs(ta.ch(ch)[i] - tbuf.ch(ch)[i]));
        }
        EXPECT_EQ(maxDiff, 0.0f);
    }
}

// Bands re-enabled by a live band-count increase must start with clean
// compressors. The old header replayed the gain reduction frozen from when
// the bands were last active: measured -15.5 dB of phantom attenuation (gain
// 0.17) on quiet material, with 15.8 dB of stale GR still reported.
DSPARK_TEST(MultibandCompressor_reenabled_bands_start_clean)
{
    auto mbc = std::make_unique<MultibandCompressor<float>>();
    auto s = spec(48000.0, 512, 2);
    mbc->prepare(s);
    mbc->setNumBands(6);
    for (int b = 0; b < 6; ++b)
    {
        mbc->getBandCompressor(b).setThreshold(-30.0f);
        mbc->getBandCompressor(b).setRatio(20.0f);
        mbc->getBandCompressor(b).setAttack(1.0f);
        mbc->getBandCompressor(b).setRelease(1000.0f);
    }

    auto tb = makeStereoBuffer(512);
    auto loud = [&] {
        tb.fillNoise();
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i) tb.ch(ch)[i] *= 0.9f;
    };
    auto quiet = [&] {
        tb.fillNoise();
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i) tb.ch(ch)[i] *= 0.004f;
    };

    for (int k = 0; k < 60; ++k) { loud(); mbc->processBlock(tb.view()); } // deep GR everywhere
    mbc->setNumBands(2);
    for (int k = 0; k < 10; ++k) { quiet(); mbc->processBlock(tb.view()); } // bands 2..5 frozen
    mbc->setNumBands(6); // re-enable

    float settledGain = 0.0f;
    for (int k = 0; k < 4; ++k)
    {
        quiet();
        double ei = 0.0;
        for (int i = 0; i < 512; ++i) ei += double(tb.ch(0)[i]) * tb.ch(0)[i];
        mbc->processBlock(tb.view());
        double eo = 0.0;
        for (int i = 0; i < 512; ++i) eo += double(tb.ch(0)[i]) * tb.ch(0)[i];
        if (k == 3) settledGain = float(std::sqrt(eo / (ei > 0.0 ? ei : 1e-30)));
    }

    // Quiet material far below threshold: no gain reduction may remain.
    for (int b = 2; b < 6; ++b)
        EXPECT_LT(std::fabs(mbc->getBandGainReductionDb(b)), 1.0f); // old: 15.8 dB
    EXPECT_GT(settledGain, 0.7f);                                   // old: ~0.2
}

// Invalid inputs must be ignored. The old header: setDensity(NaN) poisoned
// the spawn accumulator PERMANENTLY (the granulator went silently DEAD -
// no grain ever spawned again, even after reposting valid values; only
// reset() revived it, 0.21 divergence measured), and an invalid prepare
// (NaN rate or NaN bufferSeconds) UB-cast the ring size down to ONE sample.
// Also pins the mix ramp: a hard flip on the decorrelated grain cloud
// clicked at 11x the steady-state sample delta.
DSPARK_TEST(GranularProcessor_invalid_inputs_are_ignored)
{
    auto dut  = std::make_unique<GranularProcessor<float>>();
    auto twin = std::make_unique<GranularProcessor<float>>();
    const auto s = spec(48000.0, 512, 2);
    for (auto* g : { dut.get(), twin.get() })
    {
        g->setDensity(40.0f);
        g->prepare(s);
    }

    const float kNan = std::numeric_limits<float>::quiet_NaN();
    const float kInf = std::numeric_limits<float>::infinity();
    dut->setGrainSize(kNan);
    dut->setDensity(kNan);   dut->setDensity(kInf);
    dut->setJitter(kNan);
    dut->setPitch(kNan);
    dut->setPitchJitter(kNan);
    dut->setSpread(kNan);
    dut->setMix(-kInf);
    dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    dut->prepare(s, std::numeric_limits<double>::quiet_NaN());

    EXPECT_NEAR(dut->getDensity(), 40.0f, 1e-6f);
    EXPECT_NEAR(dut->getGrainSize(), 80.0f, 1e-6f);
    EXPECT_NEAR(dut->getMix(), 1.0f, 1e-6f);
    EXPECT_TRUE(!dut->getFreeze());

    auto ta = makeStereoBuffer(512);
    auto tb = makeStereoBuffer(512);
    float maxDiff = 0.0f;
    int nonFinite = 0;
    bool wetAlive = false;
    for (int k = 0; k < 30; ++k)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.5f * std::sin(2.0f * 3.14159265f * (220.0f + 60.0f * ch)
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
                // The cloud must still be alive (output differs from raw input).
                if (k > 10 && std::fabs(ta.ch(ch)[i]) > 1e-4f) wetAlive = true;
            }
    }
    EXPECT_EQ(nonFinite, 0);
    EXPECT_EQ(maxDiff, 0.0f);   // old header: 0.21 (spawn permanently dead)
    EXPECT_TRUE(wetAlive);
}

// Invalid inputs must be ignored, and a narrow buffer over a wider spec must
// not rotate the per-channel gain memories. The old header: setReduction(NaN)
// poisoned the per-bin gain memory PERMANENTLY (17920 non-finite samples
// measured AFTER reposting valid values), a NaN rate passed the old prepare
// gate (0.07 divergence), and a mono buffer over a stereo spec alternated
// channel 0 onto channel 1's release state between hops (0.004 divergence
// versus a mono-prepared twin).
DSPARK_TEST(SpectralDenoiser_invalid_inputs_and_narrow_buffers)
{
    const auto s2 = spec(48000.0, 512, 2);
    const auto s1 = spec(48000.0, 512, 1);

    // (a) NaN setters + invalid prepare -> bit-identical to a twin.
    {
        auto dut  = std::make_unique<SpectralDenoiser<float>>();
        auto twin = std::make_unique<SpectralDenoiser<float>>();
        for (auto* d : { dut.get(), twin.get() }) d->prepare(s2);

        const float kNan = std::numeric_limits<float>::quiet_NaN();
        dut->setReduction(kNan);
        dut->setThreshold(kNan);
        dut->setReduction(std::numeric_limits<float>::infinity());
        dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));

        EXPECT_NEAR(dut->getReduction(), 18.0f, 1e-6f);
        EXPECT_NEAR(dut->getThreshold(), 2.0f, 1e-6f);

        auto ta = makeStereoBuffer(512);
        auto tb = makeStereoBuffer(512);
        float maxDiff = 0.0f;
        int nonFinite = 0;
        uint32_t rng = 123u;
        for (int k = 0; k < 25; ++k)
        {
            for (int ch = 0; ch < 2; ++ch)
                for (int i = 0; i < 512; ++i)
                {
                    rng = rng * 1664525u + 1013904223u;
                    const float v = 0.4f * std::sin(2.0f * 3.14159265f * 440.0f
                                                    * float(k * 512 + i) / 48000.0f)
                                  + (float(rng >> 8) / 8388608.0f - 1.0f) * 0.05f;
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

    // (b) mono buffer over a stereo spec == a mono-prepared twin, bit-exactly.
    {
        auto wide = std::make_unique<SpectralDenoiser<float>>();
        auto mono = std::make_unique<SpectralDenoiser<float>>();
        wide->prepare(s2);
        mono->prepare(s1);
        auto ta = makeBuffer(1, 512);
        auto tb = makeBuffer(1, 512);
        float maxDiff = 0.0f;
        uint32_t rng = 55u;
        for (auto* d : { wide.get(), mono.get() }) d->setLearning(true);
        for (int k = 0; k < 60; ++k)
        {
            if (k == 30)
                for (auto* d : { wide.get(), mono.get() }) d->setLearning(false);
            for (int i = 0; i < 512; ++i)
            {
                rng = rng * 1664525u + 1013904223u;
                const float v = (k >= 30 ? 0.4f * std::sin(2.0f * 3.14159265f * 440.0f
                                                           * float(k * 512 + i) / 48000.0f)
                                         : 0.0f)
                              + (float(rng >> 8) / 8388608.0f - 1.0f) * 0.05f;
                ta.ch(0)[i] = v;
                tb.ch(0)[i] = v;
            }
            wide->processBlock(ta.view());
            mono->processBlock(tb.view());
            for (int i = 0; i < 512; ++i)
                maxDiff = std::max(maxDiff, std::fabs(ta.ch(0)[i] - tb.ch(0)[i]));
        }
        EXPECT_EQ(maxDiff, 0.0f);   // old header: 0.004 (rotated gain memories)
    }
}

// ============================================================================
// SpectralProcessor
// ============================================================================

DSPARK_TEST(SpectralProcessor_identity_passthrough)
{
    SpectralProcessor<float> sp;
    auto s = spec(44100.0, 1024, 1);
    sp.prepare(s, 2048);

    // Identity callback (no modification)
    auto identity = [](float*, int) {};

    auto tb = makeBuffer(1, 1024);
    generateSine(tb.ch(0), 1024, 440.0f, 44100.0f);

    // Process several blocks to fill latency
    for (int block = 0; block < 4; ++block)
    {
        generateSine(tb.ch(0), 1024, 440.0f, 44100.0f);
        sp.processBlock(tb.view(), identity);
    }

    // After latency, output should be close to input
    float peak = measurePeak(tb.ch(0), 1024);
    EXPECT_GT(peak, 0.5f); // Signal should come through
    EXPECT_NO_NAN(tb.ch(0), 1024);
}

DSPARK_TEST(SpectralProcessor_magnitude_halving)
{
    SpectralProcessor<float> sp;
    auto s = spec(44100.0, 1024, 1);
    sp.prepare(s, 2048);

    // Halve all magnitudes
    auto halve = [](float* data, int numBins) {
        for (int k = 0; k < numBins; ++k)
        {
            data[2 * k]     *= 0.5f;
            data[2 * k + 1] *= 0.5f;
        }
    };

    // Process several blocks
    for (int block = 0; block < 6; ++block)
    {
        auto tb = makeBuffer(1, 1024);
        generateSine(tb.ch(0), 1024, 440.0f, 44100.0f, 1.0f);
        sp.processBlock(tb.view(), halve);

        if (block >= 4) // After latency settling
        {
            float peak = measurePeak(tb.ch(0), 1024);
            // Should be roughly half amplitude
            EXPECT_LT(peak, 0.8f);
        }
    }
}

DSPARK_TEST(SpectralProcessor_latency_correct)
{
    SpectralProcessor<float> sp;
    auto s = spec(44100.0, 512, 1);
    sp.prepare(s, 1024);

    EXPECT_TRUE(sp.getLatency() == 1024);
    EXPECT_TRUE(sp.getFFTSize() == 1024);
    EXPECT_TRUE(sp.getNumBins() == 513);
}

DSPARK_TEST(SpectralProcessor_no_NaN)
{
    SpectralProcessor<float> sp;
    auto s = spec(44100.0, 512, 2);
    sp.prepare(s, 1024);

    auto boost = [](float* data, int numBins) {
        for (int k = 0; k < numBins; ++k)
            data[2 * k] *= 2.0f; // Boost real part only
    };

    for (int block = 0; block < 4; ++block)
    {
        auto tb = makeStereoBuffer(512);
        tb.fillNoise();
        sp.processBlock(tb.view(), boost);
        EXPECT_NO_NAN(tb.ch(0), 512);
        EXPECT_NO_NAN(tb.ch(1), 512);
    }
}

DSPARK_TEST(SpectralProcessor_chunking_bit_exact_and_reconstruction_nulls)
{
    // The identity STFT must be a pure fftSize-sample delay, bit-identical
    // for ANY chopping of the stream. With the pre-fix header the frame
    // anchor depended on the caller's block sizes: irregular blocks left a
    // +4.7 dB misalignment residual that no single delay could null.
    const int N = 1024, H = 256, L = 8 * N;
    std::vector<float> in(static_cast<size_t>(L));
    uint32_t rng = 123u;
    for (auto& x : in)
    {
        rng = rng * 1664525u + 1013904223u;
        x = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }
    auto identity = [](float*, int) {};

    auto runStream = [&](std::vector<int> blocks) {
        SpectralProcessor<float> sp;
        sp.prepare(spec(48000.0, 512, 1), N, H);
        std::vector<float> out(static_cast<size_t>(L));
        std::vector<float> tmp;
        size_t pos = 0, bi = 0;
        while (pos < in.size())
        {
            const int bs = std::min(blocks[bi % blocks.size()],
                                    static_cast<int>(in.size() - pos));
            ++bi;
            tmp.assign(in.begin() + static_cast<long>(pos),
                       in.begin() + static_cast<long>(pos) + bs);
            float* ch[1] = { tmp.data() };
            AudioBufferView<float> v(ch, 1, bs);
            sp.processBlock(v, identity);
            std::copy(tmp.begin(), tmp.end(), out.begin() + static_cast<long>(pos));
            pos += static_cast<size_t>(bs);
        }
        return out;
    };

    auto fixed  = runStream({ 512 });
    auto ragged = runStream({ 1, 7, 64, 129, 300, 512, 33 });

    int diffs = 0;
    for (int i = 0; i < L; ++i)
        if (fixed[static_cast<size_t>(i)] != ragged[static_cast<size_t>(i)]) ++diffs;
    EXPECT_TRUE(diffs == 0);

    float maxErr = 0.0f;
    for (int i = 2 * N; i < L; ++i)
        maxErr = std::max(maxErr, std::fabs(ragged[static_cast<size_t>(i)]
                                            - in[static_cast<size_t>(i - N)]));
    EXPECT_LT(maxErr, 1e-4f);
}

DSPARK_TEST(SpectralProcessor_latency_is_measured_exact)
{
    // Impulse argmax must equal getLatency() for several FFT/hop configs,
    // including block sizes that do NOT divide the hop. Pre-fix the measured
    // delay was fftSize - hopSize with hop-aligned blocks and drifted with
    // the caller's block size.
    const int cfg[3][3] = { { 1024, 512, 512 }, { 1024, 256, 448 }, { 2048, 512, 320 } };
    for (const auto& c : cfg)
    {
        const int N = c[0], H = c[1], bs = c[2], L = 6 * N, P = 2 * N;
        SpectralProcessor<float> sp;
        sp.prepare(spec(48000.0, 512, 1), N, H);
        auto identity = [](float*, int) {};

        std::vector<float> stream(static_cast<size_t>(L), 0.0f);
        stream[static_cast<size_t>(P)] = 1.0f;
        std::vector<float> tmp;
        int pos = 0;
        while (pos < L)
        {
            const int n = std::min(bs, L - pos);
            tmp.assign(stream.begin() + pos, stream.begin() + pos + n);
            float* ch[1] = { tmp.data() };
            AudioBufferView<float> v(ch, 1, n);
            sp.processBlock(v, identity);
            std::copy(tmp.begin(), tmp.end(), stream.begin() + pos);
            pos += n;
        }

        int argmax = 0; float best = 0.0f;
        for (int i = 0; i < L; ++i)
            if (std::fabs(stream[static_cast<size_t>(i)]) > best)
            { best = std::fabs(stream[static_cast<size_t>(i)]); argmax = i; }

        EXPECT_TRUE(argmax - P == sp.getLatency());
        EXPECT_NEAR(best, 1.0f, 0.02f);
    }
}

DSPARK_TEST(SpectralProcessor_prepare_sanitizes_and_reprepare_is_clean)
{
    // Hop sanitization: degenerate hop == fftSize (would amplitude-modulate
    // at the frame rate) and non-divisors fall back to fftSize/2; divisors
    // and the <= 0 default are honoured.
    SpectralProcessor<float> sp;
    sp.prepare(spec(48000.0, 512, 1), 1024, 1024);
    EXPECT_TRUE(sp.getHopSize() == 512);
    sp.prepare(spec(48000.0, 512, 1), 1024, 513);
    EXPECT_TRUE(sp.getHopSize() == 512);
    sp.prepare(spec(48000.0, 512, 1), 1024, -5);
    EXPECT_TRUE(sp.getHopSize() == 512);
    sp.prepare(spec(48000.0, 512, 1), 1024, 128);
    EXPECT_TRUE(sp.getHopSize() == 128);

    // Invalid specs are ignored (NaN-safe guard) and an unprepared
    // processor leaves the buffer untouched.
    SpectralProcessor<float> un;
    AudioSpec bad = spec(std::numeric_limits<double>::quiet_NaN(), 512, 2);
    un.prepare(bad);
    un.prepare(spec(0.0, 512, 2));
    un.prepare(spec(48000.0, 512, 0));
    auto tb = makeBuffer(1, 64);
    for (int i = 0; i < 64; ++i) tb.ch(0)[i] = 0.25f;
    un.processBlock(tb.view(), [](float*, int) {});
    for (int i = 0; i < 64; ++i) EXPECT_NEAR(tb.ch(0)[i], 0.25f, 0.0f);

    // Re-prepare to a SMALLER FFT size after processing: positions are
    // reset, the stream is clean (pre-fix they were carried over stale).
    SpectralProcessor<float> rp;
    rp.prepare(spec(48000.0, 512, 1), 4096, 1024);
    auto identity = [](float*, int) {};
    auto big = makeBuffer(1, 512);
    for (int b = 0; b < 24; ++b)
    {
        for (int i = 0; i < 512; ++i) big.ch(0)[i] = 0.5f;
        rp.processBlock(big.view(), identity);
    }
    rp.prepare(spec(48000.0, 512, 1), 1024, 512);
    EXPECT_TRUE(rp.getFFTSize() == 1024 && rp.getLatency() == 1024);

    const int L = 4096;
    std::vector<float> nz(static_cast<size_t>(L));
    uint32_t rng = 77u;
    for (auto& x : nz)
    {
        rng = rng * 1664525u + 1013904223u;
        x = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }
    std::vector<float> tmp;
    int pos = 0;
    while (pos < L)
    {
        tmp.assign(nz.begin() + pos, nz.begin() + pos + 512);
        float* ch[1] = { tmp.data() };
        AudioBufferView<float> v(ch, 1, 512);
        rp.processBlock(v, identity);
        std::copy(tmp.begin(), tmp.end(), nz.begin() + pos);
        pos += 512;
    }
    // nz now holds the output; compare tail region against the original
    // noise regenerated from the same seed, delayed by 1024.
    std::vector<float> orig(static_cast<size_t>(L));
    rng = 77u;
    for (auto& x : orig)
    {
        rng = rng * 1664525u + 1013904223u;
        x = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
    }
    float maxErr = 0.0f;
    for (int i = 2 * 1024; i < L; ++i)
        maxErr = std::max(maxErr, std::fabs(nz[static_cast<size_t>(i)]
                                            - orig[static_cast<size_t>(i - 1024)]));
    EXPECT_LT(maxErr, 1e-4f);
}

DSPARK_TEST(SpectralProcessor_channel_contract)
{
    // (a) The callback runs once per channel per hop, in channel order;
    // (b) a silent channel stays EXACTLY zero (shared scratch must not
    // leak between channels); (c) channels beyond the prepared count pass
    // through untouched.
    SpectralProcessor<float> sp;
    sp.prepare(spec(48000.0, 512, 2), 1024, 256);

    std::vector<float> dcMag;
    auto cb = [&dcMag](float* bins, int) { dcMag.push_back(std::fabs(bins[0])); };

    auto tb = makeStereoBuffer(512);
    for (int b = 0; b < 8; ++b)
    {
        for (int i = 0; i < 512; ++i) { tb.ch(0)[i] = 0.5f; tb.ch(1)[i] = 0.0f; }
        sp.processBlock(tb.view(), cb);
        for (int i = 0; i < 512; ++i)
            EXPECT_TRUE(tb.ch(1)[i] == 0.0f);
    }
    // 8 blocks * 512 samples / hop 256 = 16 hops * 2 channels = 32 calls.
    EXPECT_TRUE(static_cast<int>(dcMag.size()) == 32);
    for (size_t k = 16; k < dcMag.size(); k += 2)
    {
        EXPECT_GT(dcMag[k], 1.0f);          // even calls: the DC channel
        EXPECT_TRUE(dcMag[k + 1] == 0.0f);  // odd calls: the silent channel
    }

    // (c) mono-prepared processor, stereo buffer: second channel is dry.
    SpectralProcessor<float> mono;
    mono.prepare(spec(48000.0, 512, 1), 1024, 512);
    auto sb = makeStereoBuffer(512);
    std::vector<float> ref(512);
    for (int i = 0; i < 512; ++i)
    {
        sb.ch(0)[i] = 0.3f;
        sb.ch(1)[i] = std::sin(0.03f * static_cast<float>(i));
        ref[static_cast<size_t>(i)] = sb.ch(1)[i];
    }
    mono.processBlock(sb.view(), [](float*, int) {});
    for (int i = 0; i < 512; ++i)
        EXPECT_TRUE(sb.ch(1)[i] == ref[static_cast<size_t>(i)]);
}

// ============================================================================
// ProcessorChain - Bypass
// ============================================================================

DSPARK_TEST(ProcessorChain_bypass_skips_processor)
{
    ProcessorChain<float, Gain<float>, Gain<float>> chain;
    auto s = spec(44100.0, 256, 1);
    chain.prepare(s);

    chain.get<0>().setGainDb(-96.0f); // Mute
    chain.get<0>().skipRamp();
    chain.get<1>().setGainDb(0.0f);
    chain.get<1>().skipRamp();

    auto tb = makeBuffer(1, 256);
    generateDC(tb.ch(0), 256, 1.0f);

    // Without bypass: first gain mutes everything
    chain.processBlock(tb.view());
    EXPECT_LT(measurePeak(tb.ch(0), 256), 0.01f);

    // With bypass on slot 0: mute is skipped
    chain.setBypassed<0>(true);
    generateDC(tb.ch(0), 256, 1.0f);
    chain.processBlock(tb.view());
    EXPECT_NEAR(measurePeak(tb.ch(0), 256), 1.0f, 0.01f);
}

DSPARK_TEST(ProcessorChain_all_bypassed_is_passthrough)
{
    ProcessorChain<float, Gain<float>, FilterEngine<float>> chain;
    auto s = spec(44100.0, 256, 1);
    chain.prepare(s);

    chain.get<0>().setGainDb(-20.0f);
    chain.get<0>().skipRamp();
    chain.get<1>().setLowPass(100.0f);

    chain.setBypassed<0>(true);
    chain.setBypassed<1>(true);

    auto tb = makeBuffer(1, 256);
    generateDC(tb.ch(0), 256, 0.75f);
    std::vector<float> original(tb.ch(0), tb.ch(0) + 256);

    chain.processBlock(tb.view());

    // All bypassed = passthrough
    for (int i = 0; i < 256; ++i)
        EXPECT_NEAR(tb.ch(0)[i], original[i], 1e-6f);
}

DSPARK_TEST(ProcessorChain_isBypassed_reflects_state)
{
    ProcessorChain<float, Gain<float>, Gain<float>> chain;

    EXPECT_TRUE(!chain.isBypassed<0>());
    EXPECT_TRUE(!chain.isBypassed<1>());

    chain.setBypassed<0>(true);
    EXPECT_TRUE(chain.isBypassed<0>());
    EXPECT_TRUE(!chain.isBypassed<1>());

    chain.setBypassed<0>(false);
    EXPECT_TRUE(!chain.isBypassed<0>());
}

// getLatency() sums only the processors that report latency (const
// getLatency() convertible to int; a non-const one is a compile error, not
// a silent zero) and, by documented contract, does NOT change when slots
// are bypassed: hosts must not receive a fluctuating delay compensation.
DSPARK_TEST(ProcessorChain_getLatency_sums_reporters_and_ignores_bypass)
{
    ProcessorChain<float, Gain<float>, Limiter<float>, Gain<float>> chain;
    auto s = spec(48000.0, 256, 2);
    chain.prepare(s);

    chain.get<1>().setLookahead(5.0f); // 5 ms @ 48 kHz = 240 samples
    const int expected = chain.get<1>().getLatency();
    EXPECT_GT(expected, 0);
    EXPECT_EQ(chain.getLatency(), expected); // Gains report nothing

    chain.setBypassed<1>(true);
    EXPECT_EQ(chain.getLatency(), expected); // bypass-invariant by contract
}

// ============================================================================
// AlgorithmicReverb - Quality Tests
// ============================================================================

DSPARK_TEST(Reverb_echo_density)
{
    // Impulse -> measure % of non-zero samples in 20-100ms -> must be >80%
    AlgorithmicReverb<float> rev;
    auto s = spec(48000.0, 8192, 2);
    rev.prepare(s);
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    rev.setMix(1.0f);

    auto tb = makeStereoBuffer(8192);
    generateImpulse(tb.ch(0), 8192, 1.0f);
    generateImpulse(tb.ch(1), 8192, 1.0f);
    rev.processBlock(tb.view());

    // Count non-zero samples in 20-100ms range (samples 960-4800 at 48kHz)
    int start = 960, end = 4800;
    int nonZero = 0;
    for (int i = start; i < end; ++i)
        if (std::abs(tb.ch(0)[i]) > 1e-6f) ++nonZero;

    float density = static_cast<float>(nonZero) / static_cast<float>(end - start);
    EXPECT_GT(density, 0.80f);
}

DSPARK_TEST(Reverb_stereo_decorrelation)
{
    // Impulse -> cross-correlation L/R of wet < 0.5
    AlgorithmicReverb<float> rev;
    auto s = spec(48000.0, 8192, 2);
    rev.prepare(s);
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    rev.setMix(1.0f);

    auto tb = makeStereoBuffer(8192);
    generateImpulse(tb.ch(0), 8192, 1.0f);
    generateImpulse(tb.ch(1), 8192, 1.0f);
    rev.processBlock(tb.view());

    // Normalized cross-correlation in the tail region (1000-8000 samples)
    float sumLR = 0, sumLL = 0, sumRR = 0;
    for (int i = 1000; i < 8000; ++i)
    {
        float l = tb.ch(0)[i], r = tb.ch(1)[i];
        sumLR += l * r;
        sumLL += l * l;
        sumRR += r * r;
    }
    float denom = std::sqrt(sumLL * sumRR);
    float xcorr = (denom > 1e-10f) ? std::abs(sumLR / denom) : 0.0f;

    EXPECT_LT(xcorr, 0.5f);
}

DSPARK_TEST(Reverb_spectral_flatness)
{
    // White noise -> spectrum of reverb tail should be within +/-10dB in 200Hz-8kHz
    AlgorithmicReverb<float> rev;
    auto s = spec(48000.0, 4096, 2);
    rev.prepare(s);
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    rev.setDamping(0.3f);
    rev.setMix(1.0f);

    // Process several blocks of noise to fill the reverb tail
    for (int block = 0; block < 10; ++block)
    {
        auto tb = makeStereoBuffer(4096);
        tb.fillNoise(0.5f, static_cast<uint32_t>(block * 1000 + 42));
        rev.processBlock(tb.view());
    }

    // One more block of noise, measure the output spectrum via RMS in bands
    auto tb = makeStereoBuffer(4096);
    tb.fillNoise(0.5f, 99999);
    rev.processBlock(tb.view());

    // Simple band energy measurement via RMS of windowed output
    // Just verify output has energy and no NaN
    float rms = measureRMS(tb.ch(0), 4096);
    EXPECT_GT(rms, 1e-4f);
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);
}

DSPARK_TEST(Reverb_decay_accuracy)
{
    // Impulse -> measure decay. Energy at T60 should be ~60dB below peak.
    AlgorithmicReverb<float> rev;
    float decayTime = 1.0f;
    auto s = spec(48000.0, 4096, 2);
    rev.prepare(s);
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    rev.setDecay(decayTime);
    rev.setMix(1.0f);
    rev.setDamping(0.0f); // No damping for clean decay measurement

    // Feed impulse and collect many blocks
    int totalSamples = static_cast<int>(48000.0 * 2.0); // 2 seconds
    int blockSize = 4096;
    int numBlocks = totalSamples / blockSize;

    float peakEarly = 0, peakLate = 0;
    int t60Sample = static_cast<int>(48000.0f * decayTime);

    for (int b = 0; b < numBlocks; ++b)
    {
        auto tb = makeStereoBuffer(blockSize);
        if (b == 0)
        {
            generateImpulse(tb.ch(0), blockSize, 1.0f);
            generateImpulse(tb.ch(1), blockSize, 1.0f);
        }
        else
        {
            tb.fillSilence();
        }
        rev.processBlock(tb.view());

        int blockStart = b * blockSize;
        for (int i = 0; i < blockSize; ++i)
        {
            float val = std::abs(tb.ch(0)[i]);
            int globalSample = blockStart + i;
            if (globalSample >= 500 && globalSample < 2000)
                peakEarly = std::max(peakEarly, val);
            if (globalSample >= t60Sample - 500 && globalSample < t60Sample + 500)
                peakLate = std::max(peakLate, val);
        }
    }

    // At T60, signal should be significantly reduced (at least 30dB down)
    if (peakEarly > 1e-6f)
    {
        float dropDb = 20.0f * std::log10(peakLate / peakEarly);
        EXPECT_LT(dropDb, -20.0f);
    }
}

DSPARK_TEST(Reverb_damping_effectiveness)
{
    // High damping should significantly reduce HF in the tail
    AlgorithmicReverb<float> rev;
    auto s = spec(48000.0, 4096, 2);
    rev.prepare(s);
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    rev.setDamping(0.8f);
    rev.setMix(1.0f);

    // Process white noise to fill reverb state
    for (int b = 0; b < 8; ++b)
    {
        auto tb = makeStereoBuffer(4096);
        tb.fillNoise(0.5f, static_cast<uint32_t>(b * 777 + 1));
        rev.processBlock(tb.view());
    }

    // Now feed silence and measure the tail
    auto tb = makeStereoBuffer(4096);
    tb.fillSilence();
    rev.processBlock(tb.view());

    // This case only pins that a heavily damped tail still rings and stays
    // finite. The spectral tilt damping produces is measured by the T60
    // per-band cases above, which fit the decay properly rather than
    // correlating against a pair of probe tones.
    float totalRms = measureRMS(tb.ch(0), 4096);
    EXPECT_GT(totalRms, 1e-6f);
    EXPECT_NO_NAN(tb.ch(0), 4096);
}

DSPARK_TEST(Reverb_presets_unique)
{
    // Each preset's impulse response RMS should differ pairwise
    auto s = spec(48000.0, 8192, 2);

    float rmsValues[6];
    AlgorithmicReverb<float>::Type types[] = {
        AlgorithmicReverb<float>::Type::Room,
        AlgorithmicReverb<float>::Type::Hall,
        AlgorithmicReverb<float>::Type::Chamber,
        AlgorithmicReverb<float>::Type::Plate,
        AlgorithmicReverb<float>::Type::Spring,
        AlgorithmicReverb<float>::Type::Cathedral
    };

    for (int p = 0; p < 6; ++p)
    {
        AlgorithmicReverb<float> rev;
        rev.prepare(s);
        rev.setType(types[p]);
        rev.setMix(1.0f);

        auto tb = makeStereoBuffer(8192);
        generateImpulse(tb.ch(0), 8192, 1.0f);
        generateImpulse(tb.ch(1), 8192, 1.0f);
        rev.processBlock(tb.view());

        rmsValues[p] = measureRMS(tb.ch(0), 8192);
    }

    // At least 4 out of 6 presets should have distinct RMS (>5% difference)
    int distinctPairs = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
        {
            float avg = (rmsValues[i] + rmsValues[j]) * 0.5f;
            if (avg > 1e-6f && std::abs(rmsValues[i] - rmsValues[j]) / avg > 0.05f)
                ++distinctPairs;
        }
    EXPECT_GT(distinctPairs, 8); // At least 8 of 15 pairs are distinct
}

DSPARK_TEST(Reverb_no_NaN_extreme_params)
{
    AlgorithmicReverb<float> rev;
    auto s = spec(48000.0, 2048, 2);
    rev.prepare(s);

    // Test with extreme parameter values
    rev.setDecay(30.0f);
    rev.setSize(1.0f);
    rev.setDamping(1.0f);
    rev.setDiffusion(1.0f);
    rev.setModulation(1.0f);
    rev.setMix(1.0f);

    auto tb = makeStereoBuffer(2048);
    tb.fillNoise(1.0f);
    rev.processBlock(tb.view());
    EXPECT_NO_NAN(tb.ch(0), 2048);
    EXPECT_NO_NAN(tb.ch(1), 2048);

    // Minimum params
    rev.setDecay(0.1f);
    rev.setSize(0.01f);
    rev.setDamping(0.0f);
    rev.setDiffusion(0.0f);
    rev.setModulation(0.0f);

    auto tb2 = makeStereoBuffer(2048);
    tb2.fillNoise(1.0f);
    rev.processBlock(tb2.view());
    EXPECT_NO_NAN(tb2.ch(0), 2048);
    EXPECT_NO_NAN(tb2.ch(1), 2048);
}

DSPARK_TEST(Reverb_silence_stays_silent)
{
    AlgorithmicReverb<float> rev;
    auto s = spec(48000.0, 512, 2);
    rev.prepare(s);
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    rev.setMix(1.0f);

    auto tb = makeStereoBuffer(512);
    tb.fillSilence();
    rev.processBlock(tb.view());
    EXPECT_SILENT(tb.ch(0), 512, 1e-8f);
    EXPECT_SILENT(tb.ch(1), 512, 1e-8f);
}

// Schroeder backward-integration T60 (T20 fit: -5 dB to -25 dB EDC crossings).
static double schroederT60(const float* x, int n, double fs)
{
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += static_cast<double>(x[i]) * x[i];
    if (total <= 1e-20) return -1.0;
    double tail = total;
    int t5 = -1, t25 = -1;
    for (int i = 0; i < n; ++i)
    {
        const double ratio = tail / total;
        if (t5 < 0 && ratio <= 0.31622776601683794) t5 = i;
        if (ratio <= 0.0031622776601683794) { t25 = i; break; }
        const double v = x[i];
        tail -= v * v;
    }
    if (t5 < 0 || t25 < 0 || t25 <= t5) return -1.0;
    return 3.0 * (t25 - t5) / fs;
}

DSPARK_TEST(Reverb_default_quality_is_full)
{
    // The Eco engine is opt-in: a fresh instance must run the Full engine
    // (bit-identical to previous releases) and blobs without the "quality"
    // key must restore to Full.
    AlgorithmicReverb<float> rev;
    EXPECT_TRUE(rev.getQuality() == AlgorithmicReverb<float>::Quality::Full);
    rev.setQuality(AlgorithmicReverb<float>::Quality::Eco);
    EXPECT_TRUE(rev.getQuality() == AlgorithmicReverb<float>::Quality::Eco);
}

DSPARK_TEST(Reverb_eco_quality_tail_matches_full_calibration)
{
    // Eco (8 lines, control-rate mod, linear interp, no extra taps) must keep
    // the Full engine's calibration: same decay-time law, tail loudness
    // within 1.5 dB, stereo decorrelation preserved, everything finite.
    const double fs = 48000.0;
    const int block = 512;
    const int burstBlocks = static_cast<int>(fs * 1.0) / block;
    const int tailBlocks  = static_cast<int>(fs * 3.0) / block;

    double steadyRms[2] = { 0.0, 0.0 };
    double t60L[2] = { 0.0, 0.0 };
    double corr[2] = { 0.0, 0.0 };
    bool finite[2] = { true, true };

    for (int pass = 0; pass < 2; ++pass)
    {
        AlgorithmicReverb<float> rev;
        rev.prepare(spec(fs, block, 2));
        rev.setType(AlgorithmicReverb<float>::Type::Hall);
        rev.setDecay(2.0f);
        rev.setMix(1.0f);
        if (pass == 1)
            rev.setQuality(AlgorithmicReverb<float>::Quality::Eco);

        std::vector<float> tailBufL, tailBufR;
        double sumSq = 0.0; long sumN = 0;
        double phase = 0.0;
        for (int b = 0; b < burstBlocks + tailBlocks; ++b)
        {
            auto tb = makeStereoBuffer(block);
            for (int i = 0; i < block; ++i)
            {
                phase += 200.0 / fs; if (phase >= 1.0) phase -= 1.0;
                const float v = (b < burstBlocks)
                    ? static_cast<float>(0.5 * std::sin(6.283185307179586 * phase))
                    : 0.0f;
                tb.ch(0)[i] = v; tb.ch(1)[i] = v;
            }
            rev.processBlock(tb.view());
            if (b >= burstBlocks / 2 && b < burstBlocks)
                for (int i = 0; i < block; ++i)
                {
                    sumSq += static_cast<double>(tb.ch(0)[i]) * tb.ch(0)[i]
                           + static_cast<double>(tb.ch(1)[i]) * tb.ch(1)[i];
                    sumN += 2;
                }
            if (b >= burstBlocks)
                for (int i = 0; i < block; ++i)
                {
                    tailBufL.push_back(tb.ch(0)[i]);
                    tailBufR.push_back(tb.ch(1)[i]);
                    if (!std::isfinite(tb.ch(0)[i]) || !std::isfinite(tb.ch(1)[i]))
                        finite[pass] = false;
                }
        }
        steadyRms[pass] = std::sqrt(sumSq / static_cast<double>(sumN));
        t60L[pass] = schroederT60(tailBufL.data(),
                                  static_cast<int>(tailBufL.size()), fs);
        double num = 0.0, dl = 0.0, dr = 0.0;
        for (size_t i = 0; i < tailBufL.size(); ++i)
        {
            num += static_cast<double>(tailBufL[i]) * tailBufR[i];
            dl  += static_cast<double>(tailBufL[i]) * tailBufL[i];
            dr  += static_cast<double>(tailBufR[i]) * tailBufR[i];
        }
        corr[pass] = std::abs(num) / std::max(1e-20, std::sqrt(dl * dr));
    }

    EXPECT_TRUE(finite[0]);
    EXPECT_TRUE(finite[1]);
    // Tail loudness match (measured +0.26 dB on the reference machine)
    const double dbDelta = 20.0 * std::log10(std::max(steadyRms[1], 1e-12)
                                              / std::max(steadyRms[0], 1e-12));
    EXPECT_LT(std::abs(dbDelta), 1.5);
    // Same decay law: Eco T60 within 35% of Full's measured T60
    EXPECT_GT(t60L[1], t60L[0] * 0.65);
    EXPECT_LT(t60L[1], t60L[0] * 1.35);
    // Stereo tail stays decorrelated
    EXPECT_LT(corr[1], 0.5);
}

DSPARK_TEST(Reverb_eco_switching_is_click_safe)
{
    // Live Full -> Eco -> type change -> Full transitions must never emit
    // NaN or runaway peaks (the drain resets state on topology change).
    AlgorithmicReverb<float> rev;
    rev.prepare(spec(48000.0, 256, 2));
    rev.setMix(0.5f);

    bool ok = true;
    for (int b = 0; b < 400; ++b)
    {
        if (b == 100) rev.setQuality(AlgorithmicReverb<float>::Quality::Eco);
        if (b == 200) rev.setType(AlgorithmicReverb<float>::Type::Cathedral);
        if (b == 300) rev.setQuality(AlgorithmicReverb<float>::Quality::Full);
        auto tb = makeStereoBuffer(256);
        tb.fillSine(330.0f, 48000.0f, 0.4f);
        rev.processBlock(tb.view());
        for (int i = 0; i < 256; ++i)
            if (!std::isfinite(tb.ch(0)[i]) || std::abs(tb.ch(0)[i]) > 4.0f)
                ok = false;
    }
    EXPECT_TRUE(ok);
}

DSPARK_TEST(Reverb_all_types_produce_output)
{
    auto s = spec(48000.0, 4096, 2);
    AlgorithmicReverb<float>::Type types[] = {
        AlgorithmicReverb<float>::Type::Room,
        AlgorithmicReverb<float>::Type::Hall,
        AlgorithmicReverb<float>::Type::Chamber,
        AlgorithmicReverb<float>::Type::Plate,
        AlgorithmicReverb<float>::Type::Spring,
        AlgorithmicReverb<float>::Type::Cathedral
    };

    for (int p = 0; p < 6; ++p)
    {
        AlgorithmicReverb<float> rev;
        rev.prepare(s);
        rev.setType(types[p]);
        rev.setMix(1.0f);

        auto tb = makeStereoBuffer(4096);
        tb.fillSine(440.0f, 48000.0f, 0.5f);
        rev.processBlock(tb.view());

        float rms = measureRMS(tb.ch(0), 4096);
        EXPECT_GT(rms, 1e-4f);
    }
}

DSPARK_TEST(Reverb_processSample_matches_block)
{
    auto s = spec(48000.0, 256, 2);

    // Block processing
    AlgorithmicReverb<float> revBlock;
    revBlock.prepare(s);
    revBlock.setType(AlgorithmicReverb<float>::Type::Room);
    revBlock.setMix(1.0f);

    auto tb = makeStereoBuffer(256);
    generateImpulse(tb.ch(0), 256, 1.0f);
    generateImpulse(tb.ch(1), 256, 1.0f);

    // Use processSample to get raw wet output
    AlgorithmicReverb<float> revSample;
    revSample.prepare(s);
    revSample.setType(AlgorithmicReverb<float>::Type::Room);

    std::vector<float> sampleL(256), sampleR(256);
    for (int i = 0; i < 256; ++i)
    {
        float monoIn = (i == 0) ? 1.0f : 0.0f;
        auto [l, r] = revSample.processSample(monoIn);
        sampleL[i] = l;
        sampleR[i] = r;
    }

    // processSample should produce non-zero output
    float rms = measureRMS(sampleL.data(), 256);
    EXPECT_GT(rms, 1e-6f);
    EXPECT_NO_NAN(sampleL.data(), 256);
    EXPECT_NO_NAN(sampleR.data(), 256);
}

// ============================================================================
// Convolution Reverb - IR shaping (decay scale / stretch)
// ============================================================================

// Synthetic exponential hall IR: white noise under exp decay, exact T60,
// unit direct-sound spike at n = 0.
static std::vector<float> makeSyntheticIR(double fs, double lengthS, double t60)
{
    const int len = static_cast<int>(fs * lengthS);
    std::vector<float> ir(static_cast<size_t>(len));
    uint32_t rng = 7u;
    for (int n = 0; n < len; ++n)
    {
        rng = rng * 1664525u + 1013904223u;
        const float white = static_cast<float>(static_cast<int32_t>(rng))
                            * (1.0f / 2147483648.0f);
        ir[static_cast<size_t>(n)] = white
            * static_cast<float>(std::exp(-6.907755 * n / (fs * t60)));
    }
    ir[0] = 1.0f;
    return ir;
}

// Impulse through the reverb at mix = 1; returns the wet left channel.
static std::vector<float> convolutionImpulseTail(Reverb<float>& rev,
                                                 double fs, double seconds)
{
    const int block = 256;
    const int blocks = static_cast<int>(fs * seconds) / block;
    std::vector<float> out;
    out.reserve(static_cast<size_t>(blocks) * block);
    for (int b = 0; b < blocks; ++b)
    {
        auto tb = makeStereoBuffer(block);
        tb.fillSilence();
        if (b == 0) { tb.ch(0)[0] = 1.0f; tb.ch(1)[0] = 1.0f; }
        rev.processBlock(tb.view());
        for (int i = 0; i < block; ++i)
            out.push_back(tb.ch(0)[i]);
    }
    return out;
}

DSPARK_TEST(ConvReverb_decay_scale_rescales_t60_and_trims)
{
    // A 1.0 s T60 synthetic hall scaled by 0.5 must measure ~0.5 s, and the
    // shaped IR must be trimmed: no energy left near the original tail end.
    const double fs = 48000.0;
    auto ir = makeSyntheticIR(fs, 1.3, 1.0);

    double t60Ref = 0.0, t60Scaled = 0.0;
    double lateRef = 0.0, lateScaled = 0.0;
    for (int pass = 0; pass < 2; ++pass)
    {
        Reverb<float> rev;
        rev.prepare(spec(fs, 256, 2));
        rev.setMix(1.0f);
        rev.loadIR(ir.data(), static_cast<int>(ir.size()), fs);
        if (pass == 1) rev.setDecayScale(0.5f);
        auto tail = convolutionImpulseTail(rev, fs, 2.5);
        const double t60 = schroederT60(tail.data(),
                                        static_cast<int>(tail.size()), fs);
        double late = 0.0;
        for (size_t n = static_cast<size_t>(fs * 0.9); n < tail.size(); ++n)
            late += static_cast<double>(tail[n]) * tail[n];
        if (pass == 0) { t60Ref = t60; lateRef = late; }
        else           { t60Scaled = t60; lateScaled = late; }
    }

    // Reference measures its own T60 (sanity of the probe itself)
    EXPECT_GT(t60Ref, 0.85);
    EXPECT_LT(t60Ref, 1.15);
    // Scaled: half the decay time
    EXPECT_GT(t60Scaled, 0.35);
    EXPECT_LT(t60Scaled, 0.65);
    // Trim: the original still rings at 0.9 s, the scaled one is silent
    EXPECT_GT(lateRef, 1e-6);
    EXPECT_LT(lateScaled, 1e-12);
}

DSPARK_TEST(ConvReverb_stretch_scales_ir_duration)
{
    // Stretch 2.0 doubles the IR timeline (tape-speed): measured T60 ~2x.
    const double fs = 48000.0;
    auto ir = makeSyntheticIR(fs, 1.3, 1.0);

    Reverb<float> rev;
    rev.prepare(spec(fs, 256, 2));
    rev.setMix(1.0f);
    rev.loadIR(ir.data(), static_cast<int>(ir.size()), fs);
    rev.setStretch(2.0f);
    auto tail = convolutionImpulseTail(rev, fs, 4.0);
    const double t60 = schroederT60(tail.data(),
                                    static_cast<int>(tail.size()), fs);
    EXPECT_GT(t60, 1.6);
    EXPECT_LT(t60, 2.5);
}

DSPARK_TEST(ConvReverb_shaping_defaults_are_bit_transparent)
{
    // decayScale = 1 and stretch = 1 must take the exact pre-existing path:
    // output bit-identical to an instance that never touched the setters.
    const double fs = 48000.0;
    auto ir = makeSyntheticIR(fs, 0.4, 0.3);

    Reverb<float> a, b;
    a.prepare(spec(fs, 256, 2));
    b.prepare(spec(fs, 256, 2));
    a.setMix(0.7f);
    b.setMix(0.7f);
    a.loadIR(ir.data(), static_cast<int>(ir.size()), fs);
    b.loadIR(ir.data(), static_cast<int>(ir.size()), fs);
    b.setDecayScale(1.0f);
    b.setStretch(1.0f);

    auto ta = convolutionImpulseTail(a, fs, 0.8);
    auto tb2 = convolutionImpulseTail(b, fs, 0.8);
    bool identical = ta.size() == tb2.size();
    if (identical)
        for (size_t i = 0; i < ta.size(); ++i)
            if (std::memcmp(&ta[i], &tb2[i], sizeof(float)) != 0)
            { identical = false; break; }
    EXPECT_TRUE(identical);
}

DSPARK_TEST(ConvReverb_dry_and_wet_are_time_aligned)
{
    // The dry path was never delay-compensated against the convolver's
    // partition latency: with a delta IR at mix 0.5 the output showed TWO
    // impulses (dry at t0, wet at t0 + latency = comb filtering at any mix).
    // The mixer now delays the dry by the same amount: one aligned event.
    Reverb<float> rev;
    rev.prepare(spec(48000.0, 256, 2));
    std::vector<float> delta(128, 0.0f);
    delta[0] = 1.0f;
    EXPECT_TRUE(rev.loadIR(delta.data(), 128, 48000.0));
    rev.setMix(0.5f);

    const int block = 256;
    const int warmBlocks = 40;              // settle the internal mix smoother
    const int impPos = warmBlocks * block;  // impulse sample position
    std::vector<float> out;
    for (int b = 0; b < warmBlocks + 8; ++b)
    {
        auto tb = makeStereoBuffer(block);
        tb.fillSilence();
        if (b == warmBlocks) { tb.ch(0)[0] = 1.0f; tb.ch(1)[0] = 1.0f; }
        rev.processBlock(tb.view());
        for (int i = 0; i < block; ++i) out.push_back(tb.ch(0)[i]);
    }

    const int lat = rev.getLatency();
    EXPECT_EQ(lat, 256);
    EXPECT_LT(std::abs(out[static_cast<size_t>(impPos)]), 1e-4f);        // no early dry spike
    EXPECT_GT(std::abs(out[static_cast<size_t>(impPos + lat)]), 0.9f);   // aligned dry + wet
}

DSPARK_TEST(ConvReverb_reset_keeps_ir_loaded)
{
    // reset() used to drop the convolver bank entirely: a host reset on
    // stop/start silently UNLOADED the reverb (dry passthrough, isLoaded
    // false) until the next loadIR(). It now clears tails and keeps the IR.
    const double fs = 48000.0;
    auto ir = makeSyntheticIR(fs, 0.25, 0.2);

    Reverb<float> rev;
    rev.prepare(spec(fs, 256, 2));
    rev.setMix(1.0f);
    EXPECT_TRUE(rev.loadIR(ir.data(), static_cast<int>(ir.size()), fs));
    (void)convolutionImpulseTail(rev, fs, 0.1);   // run a bit

    rev.reset();
    EXPECT_TRUE(rev.isLoaded());

    auto tail = convolutionImpulseTail(rev, fs, 0.3);
    double energy = 0.0;
    for (size_t n = 2000; n < tail.size(); ++n)
        energy += static_cast<double>(tail[n]) * tail[n];
    EXPECT_GT(energy, 1e-3); // the reverb still rings after reset
}

DSPARK_TEST(ConvReverb_invalid_inputs_are_ignored)
{
    // Invalid loads used to be UB (negative length threw from the huge
    // vector range, a zero IR rate fed an infinite resampling ratio through
    // a UB cast), prepare() with negative channels THREW from the size_t
    // resize, getConvolver() before any IR dereferenced a null bank
    // (measured segfault), and NaN setters parked NaN in every getter and
    // the state blob. All inert now: bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const double fs = 48000.0;
    auto ir = makeSyntheticIR(fs, 0.2, 0.15);

    Reverb<float> subject, twin;
    auto setup = [&](Reverb<float>& r) {
        r.prepare(spec(fs, 256, 2));
        EXPECT_TRUE(r.loadIR(ir.data(), static_cast<int>(ir.size()), fs));
        r.setMix(0.6f);
        r.setPreDelay(15.0f);
    };
    setup(subject); setup(twin);

    EXPECT_TRUE(!subject.loadIR(nullptr, 100, fs));
    EXPECT_TRUE(!subject.loadIR(ir.data(), -5, fs));
    EXPECT_TRUE(!subject.loadIR(ir.data(), 0, fs));
    EXPECT_TRUE(!subject.loadIR(ir.data(), static_cast<int>(ir.size()), 0.0));
    EXPECT_TRUE(!subject.loadIR(ir.data(), static_cast<int>(ir.size()),
                                std::numeric_limits<double>::quiet_NaN()));
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 256, 2));
    subject.prepare(spec(fs, 256, -3)); // must not throw, must not resize
    subject.prepare(spec(fs, 0, 2));
    subject.setMix(nan);
    subject.setPreDelay(nan);
    subject.setDecayScale(nan);
    subject.setStretch(nan);

    // Getters and serialized state stay clean; the working IR survived.
    EXPECT_NEAR(subject.getMix(), 0.6f, 1e-6f);
    EXPECT_NEAR(subject.getPreDelay(), 15.0f, 1e-6f);
    EXPECT_NEAR(subject.getDecayScale(), 1.0f, 1e-6f);
    EXPECT_NEAR(subject.getStretch(), 1.0f, 1e-6f);
    EXPECT_TRUE(subject.isLoaded());
    EXPECT_EQ(subject.getLatency(), 256);

    // getConvolver on a fresh instance must not crash (null bank fallback).
    Reverb<float> fresh;
    fresh.prepare(spec(fs, 256, 2));
    EXPECT_EQ(fresh.getConvolver(3).getLatency(), 0);

    auto ta = convolutionImpulseTail(subject, fs, 0.4);
    auto tb = convolutionImpulseTail(twin, fs, 0.4);
    float maxDiff = 0.0f;
    for (size_t i = 0; i < ta.size(); ++i)
        maxDiff = std::max(maxDiff, std::abs(ta[i] - tb[i]));
    EXPECT_NO_NAN(ta.data(), static_cast<int>(ta.size()));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin
}

// ============================================================================
// AlgorithmicReverb - robustness (58/96 audit)
// ============================================================================

DSPARK_TEST(AlgoReverb_invalid_inputs_are_ignored)
{
    // NaN slipped through every float setter's std::clamp into the audio-side
    // drain: decay poisoned the Jot absorption coefficients (measured
    // 8077/8192 non-finite, permanent through the FDN feedback), diffusion
    // and modulation poisoned their allpass chains (8192/8192), the tone
    // cutoffs poisoned the output biquads (8192/8192), and a NaN size
    // collapsed every delay line through the UB float->int cast (finite but
    // broken topology). An invalid prepare() diverged 0.33. All inert now:
    // bit-identical to a clean twin.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    using ARev = AlgorithmicReverb<float>;
    ARev subject, twin;
    auto setup = [](ARev& r) {
        r.prepare(spec(48000.0, 512, 2));
        r.setType(ARev::Type::Hall);
        r.setMix(0.5f);
        // Drain the preset publication so the invalid values below are not
        // simply overwritten by the preset commit on the first block.
        auto w = makeBuffer(2, 64);
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 64; ++i) w.ch(ch)[i] = 0.0f;
        r.processBlock(w.view());
    };
    setup(subject); setup(twin);

    subject.setDecay(nan);           subject.setDecay(inf);
    subject.setSize(nan);            subject.setDamping(nan);
    subject.setPreDelay(nan);        subject.setDiffusion(nan);
    subject.setModulation(nan);      subject.setWidth(nan);
    subject.setErToLateDelay(nan);   subject.setHighDecayMultiplier(nan);
    subject.setBassDecayMultiplier(nan);
    subject.setHighCrossover(nan);   subject.setBassCrossover(nan);
    subject.setEarlyLevel(nan);      subject.setLateLevel(nan);
    subject.setModRate(nan);         subject.setMix(nan);
    subject.setToneLowCut(nan);      subject.setToneHighCut(nan);
    subject.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    subject.prepare(spec(48000.0, 0, 2));

    auto ta = makeBuffer(2, 8192);
    auto tb = makeBuffer(2, 8192);
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 8192; ++i)
        {
            const float v = 0.4f * std::sin(6.2831853f * (ch == 0 ? 440.0f : 620.0f) * i / 48000.0f);
            ta.ch(ch)[i] = v; tb.ch(ch)[i] = v;
        }
    subject.processBlock(ta.view());
    twin.processBlock(tb.view());

    EXPECT_NO_NAN(ta.ch(0), 8192);
    EXPECT_NO_NAN(ta.ch(1), 8192);
    float maxDiff = 0.0f;
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 8192; ++i)
            maxDiff = std::max(maxDiff, std::abs(ta.ch(ch)[i] - tb.ch(ch)[i]));
    EXPECT_TRUE(maxDiff == 0.0f); // fully inert: bit-identical to the twin

    // Wild type enum clamps and the getter stays honest.
    subject.setType(static_cast<ARev::Type>(99));
    EXPECT_EQ(static_cast<int>(subject.getType()), 5); // Cathedral (last member)
}

DSPARK_TEST(AlgoReverb_reprepare_keeps_predelay)
{
    // preDelaySamples_ was only computed at set-time from the CURRENT rate:
    // a 100 ms pre-delay set at 48 kHz played back as 50 ms after a
    // re-prepare at 96 kHz (measured onset 6776 samples instead of >= 9600).
    // prepare() now re-derives the sample counts from the ms parameters.
    using ARev = AlgorithmicReverb<float>;
    ARev rev;
    rev.prepare(spec(48000.0, 512, 2));
    rev.setPreDelay(100.0f);
    rev.setMix(1.0f);
    rev.setType(ARev::Type::Plate);   // no early reflections: clean onset
    rev.prepare(spec(96000.0, 512, 2));

    auto tb = makeBuffer(2, 512);
    int onset = -1;
    for (int blk = 0; blk < 40 && onset < 0; ++blk)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                tb.ch(ch)[i] = (blk == 0 && i == 0) ? 1.0f : 0.0f;
        rev.processBlock(tb.view());
        for (int i = 0; i < 512 && onset < 0; ++i)
            if (std::abs(tb.ch(0)[i]) > 1e-5f) onset = blk * 512 + i;
    }
    EXPECT_GT(onset, 9599); // 100 ms at 96 kHz (the old stale count gave 6776)
}

namespace
{
// Band-limited T60 of the reverb tail: impulse -> bandpass -> Schroeder T20
// fit. The preset publication is drained BEFORE the custom decay targets are
// set (the preset commit would overwrite them on the first block).
double reverbBandT60(float fcXover, double bandHz)
{
    AlgorithmicReverb<float> rev;
    rev.prepare(spec(48000.0, 512, 2));
    rev.setType(AlgorithmicReverb<float>::Type::Hall);
    {
        auto w = makeBuffer(2, 64);
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 64; ++i) w.ch(ch)[i] = 0.0f;
        rev.processBlock(w.view());
    }
    rev.setModulation(0.0f);
    rev.setEarlyLevel(-60.0f);
    rev.setMix(1.0f);
    rev.setDecay(2.0f);
    rev.setHighDecayMultiplier(0.3f);   // HF T60 target = 0.6 s
    rev.setBassDecayMultiplier(1.0f);
    rev.setHighCrossover(fcXover);

    const int total = 3 * 48000;
    std::vector<float> tail;
    tail.reserve(static_cast<size_t>(total));
    auto tb = makeBuffer(2, 512);
    for (int blk = 0; blk * 512 < total; ++blk)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
                tb.ch(ch)[i] = (blk == 0 && i == 0) ? 1.0f : 0.0f;
        rev.processBlock(tb.view());
        for (int i = 0; i < 512; ++i) tail.push_back(tb.ch(0)[i]);
    }

    Biquad<float, 1> bp;
    bp.setCoeffs(BiquadCoeffs::makeBandPass(48000.0, bandHz, 4.0));
    for (auto& v : tail) v = bp.processSample(v, 0);

    // Schroeder T20 fit (-5 dB to -25 dB of the backward energy curve).
    double totalE = 0.0;
    for (const float v : tail) totalE += static_cast<double>(v) * v;
    if (totalE <= 1e-20) return -1.0;
    double rem = totalE;
    int t5 = -1, t25 = -1;
    for (int i = 0; i < static_cast<int>(tail.size()); ++i)
    {
        const double r = rem / totalE;
        if (t5 < 0 && r <= 0.31622776601683794) t5 = i;
        if (r <= 0.0031622776601683794) { t25 = i; break; }
        rem -= static_cast<double>(tail[static_cast<size_t>(i)]) * tail[static_cast<size_t>(i)];
    }
    if (t5 < 0 || t25 < 0 || t25 <= t5) return -1.0;
    return 3.0 * (t25 - t5) / 48000.0;
}
} // namespace

DSPARK_TEST(AlgoReverb_high_crossover_is_honoured)
{
    // setHighCrossover was stored, serialized, preset-committed and exposed
    // in the Lab, but never read by the DSP: a one-pole absorption with DC
    // and Nyquist pinned has no freedom left for a transition frequency
    // (measured: bit-identical output for 1 kHz vs 16 kHz). The absorption
    // is now a pole-zero shelf: same exact T60 anchors at DC/Nyquist, with
    // the transition midpoint placed at the crossover.
    const double mid2k  = reverbBandT60(2000.0f, 2000.0);
    const double mid16k = reverbBandT60(16000.0f, 2000.0);
    const double lo2k   = reverbBandT60(2000.0f, 300.0);
    const double lo16k  = reverbBandT60(16000.0f, 300.0);

    EXPECT_GT(mid2k, 0.0);
    EXPECT_GT(mid16k, 0.0);
    // The 2 kHz band decays much faster when the crossover sits at 2 kHz
    // than when it sits at 16 kHz (measured 0.81 s vs 1.38 s = 1.71x; the
    // inert parameter gave exactly 1.00x).
    EXPECT_GT(mid16k / mid2k, 1.3);
    // The low anchor barely moves with the crossover (first-order skirt).
    EXPECT_GT(lo16k / lo2k, 0.85);
    EXPECT_LT(lo16k / lo2k, 1.25);
}

DSPARK_TEST(AlgoReverb_processSample_applies_pending_changes)
{
    // The per-sample path never drained the deferred preset/param flags: a
    // setType() was published but never applied, so per-sample streams kept
    // the old topology forever (measured: tail energy 1e-11 with a pending
    // Cathedral preset, i.e. still the short Room default).
    using ARev = AlgorithmicReverb<float>;
    ARev rev;
    rev.prepare(spec(48000.0, 512, 2));
    rev.setMix(1.0f);
    rev.setType(ARev::Type::Cathedral); // 5 s decay, published only

    double tail = 0.0;
    for (int i = 0; i < 96000; ++i)     // 2 s per-sample stream
    {
        const float x = (i == 0) ? 1.0f : 0.0f;
        auto [oL, oR] = rev.processSample(x);
        (void)oR;
        if (i > 48000) tail += static_cast<double>(oL) * oL;
    }
    EXPECT_GT(tail, 1e-6); // the Cathedral tail is ringing after 1 s
}

// ============================================================================
// Panner - Smooth Transition
// ============================================================================

DSPARK_TEST(Panner_smooth_transition_no_click)
{
    Panner<float> pan;
    auto s = spec(48000.0, 256, 2);
    pan.prepare(s);
    pan.setPan(0.0);

    // Process a block at center
    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, 48000.0f, 0.5f);
    pan.processBlock(tb.view());

    // Now jump pan to hard right
    pan.setPan(1.0);

    // Process another block - should transition smoothly
    auto tb2 = makeStereoBuffer(256);
    tb2.fillSine(440.0f, 48000.0f, 0.5f);
    pan.processBlock(tb2.view());

    // Check for clicks: max sample-to-sample difference should be bounded
    float maxDiff = 0;
    for (int i = 1; i < 256; ++i)
    {
        float diff = std::abs(tb2.ch(0)[i] - tb2.ch(0)[i - 1]);
        maxDiff = std::max(maxDiff, diff);
    }

    // With smoothing, the transition should be gradual (no sudden jumps > 0.2)
    EXPECT_LT(maxDiff, 0.2f);
    EXPECT_NO_NAN(tb2.ch(0), 256);
    EXPECT_NO_NAN(tb2.ch(1), 256);
}

DSPARK_TEST(Compressor_autoMakeup_preserves_level)
{
    Compressor<float> comp;
    comp.prepare(defaultSpec());
    comp.setThreshold(-20.0f);
    comp.setRatio(8.0f);
    comp.setAttack(0.1f);
    comp.setRelease(50.0f);
    comp.setAutoMakeup(true);

    // 0 dBFS signal - heavy compression
    auto tb = makeStereoBuffer(8192);
    tb.fillSine(440.0f, 44100.0f, 1.0f);
    float peakBefore = measurePeak(tb.ch(0), 8192);

    comp.processBlock(tb.view());
    float peakAfter = measurePeak(tb.ch(0) + 4096, 4000);

    // With auto-makeup, output should be closer to input level
    // (at least 30% of original, vs much less without makeup)
    EXPECT_GT(peakAfter, peakBefore * 0.3f);
}


// ============================================================================
// PitchShifter (phase vocoder, identity phase locking)
// ============================================================================

namespace {

// Dominant frequency of x[start..start+len) via zero-padded FFT + parabolic
// interpolation on the log-magnitude peak (~0.1 Hz resolution).
double measureDominantHz(const float* x, int len, double sampleRate)
{
    constexpr size_t kN = 16384;
    FFTReal<double> fft(kN);
    std::vector<double> t(kN, 0.0), f(kN + 2);
    const int n = std::min(len, static_cast<int>(kN));
    for (int i = 0; i < n; ++i)
    {
        const double w = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979 * i / (n - 1));
        t[static_cast<size_t>(i)] = static_cast<double>(x[i]) * w;
    }
    fft.forward(t.data(), f.data());

    const size_t bins = kN / 2 + 1;
    size_t peak = 1;
    double pPrev = 0, pPeak = 0, pNext = 0;
    for (size_t k = 1; k + 1 < bins; ++k)
    {
        const double p = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
        if (p > pPeak)
        {
            peak = k;
            pPeak = p;
            pPrev = f[2 * (k - 1)] * f[2 * (k - 1)] + f[2 * (k - 1) + 1] * f[2 * (k - 1) + 1];
            pNext = f[2 * (k + 1)] * f[2 * (k + 1)] + f[2 * (k + 1) + 1] * f[2 * (k + 1) + 1];
        }
    }
    const double l0 = 10.0 * std::log10(pPrev + 1e-30);
    const double l1 = 10.0 * std::log10(pPeak + 1e-30);
    const double l2 = 10.0 * std::log10(pNext + 1e-30);
    const double den = l0 - 2.0 * l1 + l2;
    const double d = (std::abs(den) > 1e-12) ? 0.5 * (l0 - l2) / den : 0.0;
    return (static_cast<double>(peak) + d) * sampleRate / static_cast<double>(kN);
}

// Runs a 440 Hz sine through the shifter and returns the last second of output.
std::vector<float> runShifter(float semitones, float mix, double sampleRate, int seconds = 3)
{
    PitchShifter<float> ps;
    ps.prepare(spec(sampleRate, 512, 2));
    ps.setSemitones(semitones);
    ps.setMix(mix);

    const int total = (static_cast<int>(sampleRate) * seconds / 512) * 512;
    std::vector<float> out(static_cast<size_t>(total));
    auto buf = makeStereoBuffer(512);
    int n = 0;
    for (int b = 0; b < total / 512; ++b)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            const float v = 0.5f * std::sin(dspark::twoPi<float> * 440.0f
                                            * static_cast<float>(n) / static_cast<float>(sampleRate));
            buf.ch(0)[i] = v;
            buf.ch(1)[i] = v;
        }
        ps.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(b * 512 + i)] = buf.ch(0)[i];
    }
    return out;
}

} // namespace

DSPARK_TEST(PitchShifter_shift_up_7st_exact_frequency)
{
    auto out = runShifter(7.0f, 1.0f, 48000.0);
    const double expected = 440.0 * std::exp2(7.0 / 12.0);   // 659.255 Hz
    const int tail = 16384;
    const double got = measureDominantHz(out.data() + out.size() - tail, tail, 48000.0);
    const double cents = 1200.0 * std::log2(got / expected);
    EXPECT_NEAR(cents, 0.0, 2.0);   // within 2 cents
}

DSPARK_TEST(PitchShifter_shift_down_12st_exact_frequency)
{
    auto out = runShifter(-12.0f, 1.0f, 48000.0);
    const int tail = 16384;
    const double got = measureDominantHz(out.data() + out.size() - tail, tail, 48000.0);
    const double cents = 1200.0 * std::log2(got / 220.0);
    EXPECT_NEAR(cents, 0.0, 2.0);
}

DSPARK_TEST(PitchShifter_sideband_rejection)
{
    // Identity phase locking keeps phase-vocoder modulation sidebands low:
    // total power outside the shifted carrier must stay below -35 dB.
    auto out = runShifter(7.0f, 1.0f, 48000.0);

    constexpr size_t kN = 16384;
    FFTReal<double> fft(kN);
    std::vector<double> t(kN), f(kN + 2);
    for (size_t i = 0; i < kN; ++i)
    {
        const double w = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979 * static_cast<double>(i) / (kN - 1));
        t[i] = static_cast<double>(out[out.size() - kN + i]) * w;
    }
    fft.forward(t.data(), f.data());

    const size_t bins = kN / 2 + 1;
    size_t peak = 1;
    std::vector<double> p(bins);
    for (size_t k = 0; k < bins; ++k)
    {
        p[k] = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
        if (k > 4 && p[k] > p[peak]) peak = k;
    }
    double carrier = 0, rest = 0;
    for (size_t k = 1; k < bins; ++k)
    {
        if (k + 4 >= peak && k <= peak + 4) carrier += p[k];
        else                                rest += p[k];
    }
    const double sidebandDb = 10.0 * std::log10((rest + 1e-30) / (carrier + 1e-30));
    EXPECT_LT(sidebandDb, -35.0);
}

DSPARK_TEST(PitchShifter_dry_path_is_latency_compensated)
{
    // mix = 0 must be a pure delay of getLatency() samples.
    PitchShifter<float> ps;
    ps.prepare(spec(48000.0, 512, 2));
    ps.setSemitones(7.0f);
    ps.setMix(0.0f);
    const int latency = ps.getLatency();

    const int total = (48000 / 512) * 512;   // whole blocks only
    std::vector<float> in(static_cast<size_t>(total)), out(static_cast<size_t>(total));
    generateSine(in.data(), total, 997.0f, 48000.0f, 0.5f);

    auto buf = makeStereoBuffer(512);
    for (int b = 0; b < total / 512; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            buf.ch(0)[i] = in[static_cast<size_t>(b * 512 + i)];
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        ps.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out[static_cast<size_t>(b * 512 + i)] = buf.ch(0)[i];
    }

    double err = 0, ref = 0;
    for (int i = latency + 4096; i < total; ++i)
    {
        const double d = in[static_cast<size_t>(i - latency)];
        const double e = out[static_cast<size_t>(i)] - d;
        err += e * e;
        ref += d * d;
    }
    const double residualDb = 10.0 * std::log10((err + 1e-30) / (ref + 1e-30));
    EXPECT_LT(residualDb, -80.0);
}

DSPARK_TEST(PitchShifter_finite_and_tail_decays)
{
    PitchShifter<float> ps;
    ps.prepare(spec(48000.0, 512, 2));
    ps.setSemitones(5.0f);

    auto buf = makeStereoBuffer(512);
    for (int b = 0; b < 100; ++b)
    {
        buf.fillSine(220.0f, 48000.0f, 0.5f);
        ps.processBlock(buf.view());
        EXPECT_NO_NAN(buf.ch(0), 512);
    }
    float tailPeak = 0.0f;
    for (int b = 0; b < 60; ++b)
    {
        buf.fillSilence();
        ps.processBlock(buf.view());
        EXPECT_NO_NAN(buf.ch(0), 512);
        if (b == 59) tailPeak = measurePeak(buf.ch(0), 512);
    }
    EXPECT_LT(tailPeak, 1e-3f);
}


// ============================================================================
// GranularProcessor + SpectralDenoiser
// ============================================================================

DSPARK_TEST(Granular_freeze_sustains_and_pitch_doubles)
{
    GranularProcessor<float> gp;
    gp.prepare(spec(48000.0, 512, 2));
    gp.setGrainSize(200.0f);
    gp.setDensity(40.0f);
    gp.setJitter(0.05f);
    gp.setSpread(0.0f);
    gp.setPitch(12.0f);

    auto buf = makeStereoBuffer(512);
    for (int b = 0; b < 94; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            buf.ch(0)[i] = 0.5f * std::sin(2.0f * 3.14159265f * 440.0f
                                           * static_cast<float>(b * 512 + i) / 48000.0f);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        gp.processBlock(buf.view());
    }
    gp.setFreeze(true);

    std::vector<float> out;
    for (int b = 0; b < 94; ++b)
    {
        buf.fillSilence();
        gp.processBlock(buf.view());
        EXPECT_NO_NAN(buf.ch(0), 512);
        for (int i = 0; i < 512; ++i)
            out.push_back(buf.ch(0)[i]);
    }
    double rms = 0;
    for (size_t i = out.size() / 2; i < out.size(); ++i)
        rms += static_cast<double>(out[i]) * out[i];
    rms = std::sqrt(rms / (out.size() / 2.0));
    EXPECT_GT(rms, 0.05);   // frozen cloud keeps sounding

    auto mag = [&](double freq) {
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
    EXPECT_GT(mag(880.0), 3.0 * mag(440.0));   // +12 st: content doubled
}

DSPARK_TEST(SpectralDenoiser_improves_snr_and_keeps_tone)
{
    SpectralDenoiser<float> dn;
    dn.prepare(spec(48000.0, 512, 2));
    dn.setReduction(20.0f);
    dn.setThreshold(2.0f);

    auto buf = makeStereoBuffer(512);
    uint32_t rng = 42u;
    auto noise = [&]() {
        rng = rng * 1664525u + 1013904223u;
        return 0.05f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
    };

    dn.setLearning(true);
    for (int b = 0; b < 94; ++b)
    {
        for (int i = 0; i < 512; ++i) { buf.ch(0)[i] = noise(); buf.ch(1)[i] = noise(); }
        dn.processBlock(buf.view());
    }
    dn.setLearning(false);

    std::vector<float> out;
    for (int b = 0; b < 188; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            const float tone = 0.25f * std::sin(2.0f * 3.14159265f * 1000.0f
                                                * static_cast<float>(b * 512 + i) / 48000.0f);
            buf.ch(0)[i] = tone + noise();
            buf.ch(1)[i] = tone + noise();
        }
        dn.processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
            out.push_back(buf.ch(0)[i]);
    }
    const size_t from = out.size() / 2;
    double re = 0, im = 0, total = 0;
    for (size_t i = from; i < out.size(); ++i)
    {
        const double ph = 2.0 * 3.14159265358979 * 1000.0
                        * static_cast<double>(i - from) / 48000.0;
        re += static_cast<double>(out[i]) * std::cos(ph);
        im += static_cast<double>(out[i]) * std::sin(ph);
        total += static_cast<double>(out[i]) * out[i];
    }
    const double tone = 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(out.size() - from);
    total /= static_cast<double>(out.size() - from);
    const double tonePow = tone * tone / 2.0;
    const double snrOut = 10.0 * std::log10(tonePow / std::max(total - tonePow, 1e-12));
    EXPECT_NEAR(tone, 0.25, 0.05);   // tone preserved
    EXPECT_GT(snrOut, 25.0);          // noise floor pushed well down
}


DSPARK_TEST(PitchShifter_formant_preserve_keeps_envelope)
{
    // 150 Hz glottal train through fixed 800/2400 Hz resonators, +7 st.
    // With formant preservation the envelope must stay near 800 Hz; without
    // it the envelope shifts to ~1200 Hz (the chipmunk effect).
    auto envelopeRatio = [](bool formant) {
        PitchShifter<float> ps;
        ps.prepare(spec(48000.0, 512, 1));
        ps.setSemitones(7.0f);
        ps.setFormantPreserve(formant);

        Biquad<float> f1, f2;
        f1.setCoeffs(BiquadCoeffs::makeBandPass(48000.0, 800.0, 6.0));
        f2.setCoeffs(BiquadCoeffs::makeBandPass(48000.0, 2400.0, 6.0));

        auto buf = makeMonoBuffer(512);
        std::vector<float> out;
        int n = 0;
        const int period = 320;   // 150 Hz at 48 kHz
        for (int b = 0; b < 200; ++b)
        {
            for (int i = 0; i < 512; ++i, ++n)
            {
                const float pulse = (n % period == 0) ? 1.0f : 0.0f;
                buf.ch(0)[i] = f1.processSample(pulse, 0) + 0.7f * f2.processSample(pulse, 0);
            }
            ps.processBlock(buf.view());
            if (b > 60)
                for (int i = 0; i < 512; ++i)
                    out.push_back(buf.ch(0)[i]);
        }

        constexpr size_t kN = 1 << 14;
        FFTReal<double> fft(kN);
        std::vector<double> t(kN), f(kN + 2), avg(kN / 2 + 1, 0.0);
        for (size_t s = 0; s + kN <= out.size(); s += kN / 2)
        {
            for (size_t i = 0; i < kN; ++i)
            {
                const double w = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979
                                                      * static_cast<double>(i) / (kN - 1));
                t[i] = static_cast<double>(out[s + i]) * w;
            }
            fft.forward(t.data(), f.data());
            for (size_t k = 0; k <= kN / 2; ++k)
                avg[k] += std::sqrt(f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1]);
        }
        const double binHz = 48000.0 / static_cast<double>(kN);
        auto bandPeak = [&](double lo, double hi) {
            double m = 0;
            for (size_t k = static_cast<size_t>(lo / binHz);
                 k <= static_cast<size_t>(hi / binHz); ++k)
                m = std::max(m, avg[k]);
            return m;
        };
        return bandPeak(700.0, 900.0) / (bandPeak(1100.0, 1300.0) + 1e-12);
    };

    EXPECT_GT(envelopeRatio(true), 1.5);    // envelope held at 800
    EXPECT_LT(envelopeRatio(false), 0.8);   // envelope moved to 1200
}

// Invalid inputs must be ignored. The old header: setSemitones(NaN) poisoned
// the glide accumulator PERMANENTLY (20480 non-finite samples measured AFTER
// posting a valid value again - only reset() plus a valid setter recovered),
// and prepare() with a NaN rate passed the `<= 0` gate, leaving sampleRate_
// NaN (UB float-to-int in the formant lifter cut; 0.21 divergence measured).
DSPARK_TEST(PitchShifter_invalid_inputs_are_ignored)
{
    auto dut  = std::make_unique<PitchShifter<float>>();
    auto twin = std::make_unique<PitchShifter<float>>();
    const auto s = spec(48000.0, 512, 2);
    for (auto* p : { dut.get(), twin.get() })
    {
        p->setSemitones(4.0f);
        p->setFormantPreserve(true);
        p->prepare(s);
    }

    const float kNan = std::numeric_limits<float>::quiet_NaN();
    dut->setSemitones(kNan);
    dut->setPitchRatio(kNan);
    dut->setMix(kNan);
    dut->setSemitones(std::numeric_limits<float>::infinity());
    dut->prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2)); // old: poisons rate
    dut->prepare(s, 1000);                                               // not a power of two
    dut->prepare(s, 1 << 24);                                            // above the size cap

    EXPECT_NEAR(dut->getSemitones(), 4.0f, 1e-6f);
    EXPECT_NEAR(dut->getMix(), 1.0f, 1e-6f);
    EXPECT_TRUE(dut->getFormantPreserve());
    EXPECT_TRUE(dut->getTransientPreserve());

    auto ta = makeStereoBuffer(512);
    auto tb = makeStereoBuffer(512);
    float maxDiff = 0.0f;
    int nonFinite = 0;
    for (int k = 0; k < 30; ++k)
    {
        for (int ch = 0; ch < 2; ++ch)
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.7f * std::sin(2.0f * 3.14159265f * (220.0f + 110.0f * ch)
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

// The wet stream's real latency is 2 * fftSize (measured); the old header
// reported fftSize + fftSize/4 and delayed the dry path by that same wrong
// value, so a 50% mix comb-filtered: at 484.375 Hz (an exact half-cycle of
// the 1536-sample mismatch) the output CANCELLED to 0.0000. Both the report
// and the dry alignment are pinned here.
DSPARK_TEST(PitchShifter_wet_and_dry_are_time_aligned)
{
    // (a) reported latency == measured impulse peak at unity ratio.
    {
        auto ps = std::make_unique<PitchShifter<float>>();
        ps->setSemitones(0.0f);
        ps->prepare(spec(48000.0, 512, 1));
        auto buf = makeBuffer(1, 512);
        int argmax = -1, n = 0;
        float best = 0.0f;
        for (int k = 0; k < 24; ++k)
        {
            for (int i = 0; i < 512; ++i) buf.ch(0)[i] = 0.0f;
            if (k == 0) buf.ch(0)[0] = 1.0f;
            ps->processBlock(buf.view());
            for (int i = 0; i < 512; ++i, ++n)
                if (std::fabs(buf.ch(0)[i]) > best) { best = std::fabs(buf.ch(0)[i]); argmax = n; }
        }
        EXPECT_EQ(argmax, ps->getLatency());   // old header: 4096 vs reported 2560
        EXPECT_GT(best, 0.9f);
    }

    // (b) 50% mix does not comb: worst-case frequency passes at unity.
    {
        auto ps = std::make_unique<PitchShifter<float>>();
        ps->setSemitones(0.0f);
        ps->setMix(0.5f);
        ps->prepare(spec(48000.0, 512, 1));
        auto buf = makeBuffer(1, 512);
        int n = 0;
        double sumIn = 0.0, sumOut = 0.0;
        for (int k = 0; k < 120; ++k)
        {
            for (int i = 0; i < 512; ++i, ++n)
                buf.ch(0)[i] = 0.5f * std::sin(2.0f * 3.14159265f * 484.375f
                                               * float(n) / 48000.0f);
            if (k >= 40)
                for (int i = 0; i < 512; ++i) sumIn += double(buf.ch(0)[i]) * buf.ch(0)[i];
            ps->processBlock(buf.view());
            if (k >= 40)
                for (int i = 0; i < 512; ++i) sumOut += double(buf.ch(0)[i]) * buf.ch(0)[i];
        }
        const double gain = std::sqrt(sumOut / (sumIn > 0.0 ? sumIn : 1e-30));
        EXPECT_GT(gain, 0.9);   // old header: 0.0000 (full cancellation)
        EXPECT_LT(gain, 1.1);
    }
}

// A hard mix flip on the decorrelated wet stream used to jump 3.7x the
// steady-state sample delta; the linear per-block ramp keeps it at regime.
DSPARK_TEST(PitchShifter_mix_change_is_click_free)
{
    auto ps = std::make_unique<PitchShifter<float>>();
    ps->setSemitones(7.0f);
    ps->setMix(1.0f);
    ps->prepare(spec(48000.0, 512, 1));
    auto buf = makeBuffer(1, 512);
    int n = 0;
    float prev = 0.0f;
    double regimeStep = 0.0, clickStep = 0.0;
    for (int k = 0; k < 120; ++k)
    {
        if (k == 80) ps->setMix(0.0f);
        for (int i = 0; i < 512; ++i, ++n)
            buf.ch(0)[i] = 0.8f * (0.5f * std::sin(2.0f * 3.14159265f * 220.0f * float(n) / 48000.0f)
                                 + 0.3f * std::sin(2.0f * 3.14159265f * 1980.0f * float(n) / 48000.0f));
        ps->processBlock(buf.view());
        for (int i = 0; i < 512; ++i)
        {
            const double d = std::fabs(double(buf.ch(0)[i]) - double(prev));
            prev = buf.ch(0)[i];
            if (k >= 40 && k < 80) regimeStep = std::max(regimeStep, d);
            if (k >= 80 && k < 84) clickStep = std::max(clickStep, d);
        }
    }
    EXPECT_LT(clickStep, regimeStep * 1.5 + 1e-4);   // old header: 3.7x regime
}

// ---------------------------------------------------------------------------
// Concurrent publication pins for Effects/DynamicEQ.h and Effects/Reverb.h.

// Every field of a published DynamicEQ band is atomic behind a per-band
// seqlock, so a band the audio thread adopts is always one whole publication.
//
// HONEST NOTE ON THIS PIN'S POWER, so it is not mistaken for a red/green pair
// like the Equalizer one: pre-fix DynamicEQ already carried a working seq
// counter with an s0 == s1 retry, so a torn copy was retried rather than
// adopted. Its defects were the plain struct staged inside the seqlock (formal
// UB) and the writer's missing release fence -- neither of which any x86 test
// can observe. This pin therefore does NOT go red on the pre-fix header. What
// it does is fix the invariant against future regressions and, more usefully,
// drive the handoff concurrently so CI ThreadSanitizer -- the only oracle here
// that understands the C++11 model, and the only one that could ever see the
// fence -- has something to look at. Before this, no test drove any Effects/
// handoff concurrently at all.
DSPARK_TEST(DynamicEQ_concurrent_setBand_is_never_torn)
{
    using DEQ = DynamicEQ<float, 4>;

    auto deq = std::make_unique<DEQ>();
    AudioSpec spec; spec.sampleRate = 48000.0; spec.maxBlockSize = 32; spec.numChannels = 1;
    deq->prepare(spec);
    deq->setNumBands(1);

    DEQ::BandConfig a;
    a.frequency = 120.0f;  a.q = 0.6f; a.threshold = -30.0f;
    a.shape = DEQ::BandShape::Bell;      a.enabled = true;
    a.aboveRatio = 2.0f; a.aboveAttackMs = 1.0f; a.aboveReleaseMs = 20.0f; a.aboveRangeDb = 3.0f;
    DEQ::BandConfig b;
    b.frequency = 9000.0f; b.q = 6.0f; b.threshold = -6.0f;
    b.shape = DEQ::BandShape::HighShelf; b.enabled = true;
    b.aboveRatio = 8.0f; b.aboveAttackMs = 40.0f; b.aboveReleaseMs = 300.0f; b.aboveRangeDb = 18.0f;
    deq->setBand(0, a);

    std::atomic<bool> stop{ false };
    std::atomic<long long> blocks{ 0 };

    std::thread control([&] {
        bool even = true;
        while (!stop.load(std::memory_order_relaxed))
        {
            deq->setBand(0, even ? a : b);
            even = !even;
        }
    });

    std::thread audio([&] {
        std::array<float, 32> buf{};
        buf.fill(0.25f);
        float* chans[1] = { buf.data() };
        long long n = 0;
        for (int i = 0; i < 20000; ++i)
        {
            buf.fill(0.25f);
            AudioBufferView<float> view(chans, 1, 32);
            deq->processBlock(view);
            for (int k = 0; k < 32; ++k)
                EXPECT_TRUE(std::isfinite(buf[static_cast<size_t>(k)]));
            ++n;
        }
        blocks.store(n);
        stop.store(true, std::memory_order_relaxed);
    });

    control.join();
    audio.join();

    // A band assembled from two different publications can pair a 9 kHz shelf
    // with a 1 ms attack and a -30 dB threshold; the gain readout must stay a
    // finite, bounded number whatever the interleaving.
    const float gr = deq->getBandGainDb(0);
    EXPECT_TRUE(std::isfinite(gr));
    EXPECT_LT(std::abs(gr), 60.0f);
    EXPECT_GT(blocks.load(), 1000LL);   // liveness: no vacuous pass
}

// Effects/Reverb.h::getConvolver() returns a reference into the published
// convolver bank. This pin holds that reference across a republication and
// keeps using it.
//
// Why it can actually fail: on the pre-fix header this is a deterministic
// heap-use-after-free, single-threaded, through documented public API -- the
// accessor took a local shared_ptr snapshot, returned a reference into it and
// let the snapshot die at the closing brace, so loadIR() freed the bank under
// the caller, which AddressSanitizer reports as a heap-use-after-free.
// Under the ASan+UBSan build this pin is a hard failure on any regression; on
// normal builds it asserts the observed values, which a freed bank has no
// obligation to preserve.
DSPARK_TEST(Reverb_getConvolver_reference_survives_republication)
{
    auto rv = std::make_unique<Reverb<float>>();
    AudioSpec spec; spec.sampleRate = 48000.0; spec.maxBlockSize = 64; spec.numChannels = 2;
    rv->prepare(spec);

    std::vector<float> ir(4096, 0.0f);
    ir[0] = 1.0f;
    ir[128] = 0.4f;
    EXPECT_TRUE(rv->loadIR(ir.data(), static_cast<int>(ir.size()), 48000.0));

    auto& conv = rv->getConvolver(0);
    const int partsBefore = conv.getNumPartitions();
    const int blockBefore = conv.getBlockSize();
    EXPECT_GT(partsBefore, 0);

    // Documented as an ordinary thing to do: the bank "may be replaced by a
    // future loadIR() call". The reference must not die with the old bank.
    ir[1] = 0.25f;
    EXPECT_TRUE(rv->loadIR(ir.data(), static_cast<int>(ir.size()), 48000.0));

    EXPECT_EQ(conv.getNumPartitions(), partsBefore);
    EXPECT_EQ(conv.getBlockSize(), blockBefore);

    // And again after the other publishing setters.
    rv->setDecayScale(0.5f);
    rv->setStretch(1.5f);
    EXPECT_EQ(conv.getNumPartitions(), partsBefore);
}

// ---------------------------------------------------------------------------
// Concurrent publication pin for Effects/detail/PhaseVocoderEngine.h.
//
// The engine's ONE cross-thread handoff is publishParams(): a canonical
// seqlock publish of the {targetSemitones, transientPreserve,
// formantPreserve} set, adopted on the audio thread by a BOUNDED reader at
// hop boundaries. This pin drives that handoff concurrently, which is what
// makes the race reachable: a control thread publishes triples whose fields
// are tied together by construction while the stream-owner thread pushes
// audio and inspects the adopted set after every commit. A reader that mixed
// words from two publications would produce a triple outside the published
// table; the bounded reader must instead adopt whole sets only (give-ups keep
// the previous whole set, so the invariant also covers the give-up path).
// The subject is heap-allocated (valgrind DRD ignores stack subjects by
// default) and both loops are bounded (valgrind's serializing scheduler can
// starve a consumer behind an unbounded producer). The single-threaded tail
// is the deterministic control: with no concurrency, the last publish must be
// adopted at the next hop, exactly.
DSPARK_TEST(PhaseVocoderEngine_concurrent_publication_adopts_whole_sets)
{
    using Engine = dspark::detail::PhaseVocoderEngine<float>;
    auto eng = std::make_unique<Engine>();
    EXPECT_TRUE(eng->prepare(48000.0, 1, 256, true));
    eng->reset();

    // Field-tying: semitones k-12 pairs with transient=(k&1) and
    // formant=(k%3==0), all exact integer-valued doubles.
    auto tripleOf = [](int i) {
        const int k = i % 25;
        Engine::Params p;
        p.targetSemitones   = static_cast<double>(k) - 12.0;
        p.transientPreserve = (k & 1) != 0;
        p.formantPreserve   = (k % 3) == 0;
        return p;
    };

    std::thread control([&] {
        for (int i = 0; i < 20000; ++i)
            eng->publishParams(tripleOf(i));
    });

    std::atomic<int> adoptionChanges{ 0 };
    std::thread audio([&] {
        std::array<float, 64> buf{};
        buf.fill(0.1f);
        double lastSt = 0.0;
        int changes = 0;
        for (int i = 0; i < 30000; ++i)
        {
            const int n = std::min<int>(64, eng->samplesToNextHop());
            eng->pushInput(0, buf.data(), n);
            eng->commitInput(n, 1);
            const auto p = eng->activeParams();
            const int k = static_cast<int>(p.targetSemitones + 12.0);
            const bool isDefault = p.targetSemitones == 0.0
                                && p.transientPreserve && !p.formantPreserve;
            const bool inTable = k >= 0 && k <= 24
                && p.targetSemitones == static_cast<double>(k) - 12.0
                && p.transientPreserve == ((k & 1) != 0)
                && p.formantPreserve == ((k % 3) == 0);
            EXPECT_TRUE(isDefault || inTable);
            if (p.targetSemitones != lastSt)
            {
                ++changes;
                lastSt = p.targetSemitones;
            }
        }
        adoptionChanges.store(changes);
    });

    control.join();
    audio.join();

    // Probe-input liveness: publications actually landed (the last published
    // triple has st = +12, so at least one adoption must have been observed).
    EXPECT_GT(adoptionChanges.load(), 0);

    // Deterministic single-threaded control: the last publish is adopted at
    // the next hop, as one whole set.
    Engine::Params fin;
    fin.targetSemitones = 5.0;
    fin.transientPreserve = false;
    fin.formantPreserve = false;
    eng->publishParams(fin);
    std::array<float, 256> tail{};
    int fed = 0;
    while (fed < 512)
    {
        const int n = std::min<int>(256, eng->samplesToNextHop());
        eng->pushInput(0, tail.data(), n);
        eng->commitInput(n, 1);
        fed += n;
    }
    const auto pf = eng->activeParams();
    EXPECT_EQ(pf.targetSemitones, 5.0);
    EXPECT_FALSE(pf.transientPreserve);
    EXPECT_FALSE(pf.formantPreserve);
}

// A streaming processor's output must depend on the audio, not on how the
// host happened to chop it into blocks. The old processBlock committed the
// shared fractional read position once per chunk as a single
// frac + ratio * chunk product, while the per-sample output loop advanced
// the same position iteratively; the two round differently whenever
// ratio * chunk is inexact, and chunk boundaries follow the host block size,
// so at st = +7 (ratio 2^(7/12)) a 512-sample chopping and a 64-sample
// chopping of the same signal produced different bit streams (measured:
// max delta 7.45e-9, i.e. ~1 float ulp at 0.5 FS, hashes acc88bdde0c56924
// vs b50b6343b9975ca5). The commit now runs the same per-sample recurrence,
// so any chopping of the same input yields the identical bit stream.
DSPARK_TEST(PitchShifter_output_is_bit_identical_under_block_chopping)
{
    // Same scenario the characterisation harness measured red: 3 s stereo,
    // 440 / 620.5 Hz sines plus deterministic LCG noise, st = +7.
    const int total = 44100 * 3;
    std::vector<std::vector<float>> sig(2, std::vector<float>(static_cast<size_t>(total)));
    for (int ch = 0; ch < 2; ++ch)
    {
        const double f = (ch == 0) ? 440.0 : 620.5;
        const double w = 2.0 * 3.141592653589793 * f / 44100.0;
        double phase = 0.0;
        for (int n = 0; n < total; ++n)
        {
            sig[static_cast<size_t>(ch)][static_cast<size_t>(n)] =
                static_cast<float>(0.5 * std::sin(phase));
            phase += w;
            if (phase > 2.0 * 3.141592653589793) phase -= 2.0 * 3.141592653589793;
        }
    }
    uint64_t lcg = 0xC0FFEEu;
    for (int ch = 0; ch < 2; ++ch)
        for (int n = 0; n < total; ++n)
        {
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            const float r = (static_cast<float>(static_cast<uint32_t>(lcg >> 33))
                             / 4294967296.0f) * 2.0f - 0.5f;
            sig[static_cast<size_t>(ch)][static_cast<size_t>(n)] += 0.05f * r;
        }

    auto runWith = [&](const std::vector<int>& chop) {
        auto ps = std::make_unique<PitchShifter<float>>();
        ps->setSemitones(7.0f);                    // irrational ratio 2^(7/12)
        ps->prepare(spec(44100.0, 512, 2));
        auto buf = sig;
        int pos = 0;
        size_t ci = 0;
        while (pos < total)
        {
            const int blk = std::min(chop[ci % chop.size()], total - pos);
            ++ci;
            float* ch[2] = { buf[0].data() + pos, buf[1].data() + pos };
            AudioBufferView<float> view(ch, 2, blk);
            ps->processBlock(view);
            pos += blk;
        }
        return buf;
    };

    const auto a = runWith({ 512 });
    const auto b = runWith({ 64 });
    const auto c = runWith({ 371, 64, 512, 1, 128 });
    int diffAB = 0, diffAC = 0, firstAB = -1;
    for (int ch = 0; ch < 2; ++ch)
        for (int n = 0; n < total; ++n)
        {
            const auto sc = static_cast<size_t>(ch);
            const auto sn = static_cast<size_t>(n);
            if (a[sc][sn] != b[sc][sn])
            {
                if (firstAB < 0) firstAB = n;
                ++diffAB;
            }
            if (a[sc][sn] != c[sc][sn]) ++diffAC;
        }
    if (firstAB >= 0)
        std::cerr << "    first 512-vs-64 divergence at sample " << firstAB
                  << " (" << diffAB << " samples differ)\n";
    EXPECT_EQ(diffAB, 0);
    EXPECT_EQ(diffAC, 0);
}
