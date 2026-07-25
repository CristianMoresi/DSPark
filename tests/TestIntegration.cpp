// DSPark Tests - Integration
// ProcessorChain composed, full signal chain, dual-type, processor concepts

#include "dspark_test.h"
#include "TestSignals.h"

#include "../DSPark.h" // full umbrella: the concept matrix below touches everything

#include <cmath>
#include <cstdio>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// ProcessorChain - composed effects
// ============================================================================

DSPARK_TEST(Integration_DCBlocker_Gain_Limiter_chain)
{
    // Chain: DCBlocker -> Gain -> Limiter
    ProcessorChain<float, DCBlocker<float>, Gain<float>, Limiter<float>> chain;
    auto s = defaultSpec();
    chain.prepare(s);

    // Configure
    chain.get<1>().setGainDb(6.0f);  // Boost
    chain.get<1>().skipRamp();
    chain.get<2>().setCeiling(-1.0f); // Limit at -1dBFS

    // Input: sine + DC offset
    auto tb = makeStereoBuffer(4096);
    for (int i = 0; i < 4096; ++i)
    {
        float sample = 0.3f + 0.7f * std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);
        tb.ch(0)[i] = sample;
        tb.ch(1)[i] = sample;
    }

    chain.processBlock(tb.view());

    // Verify: no NaN
    EXPECT_NO_NAN(tb.ch(0), 4096);
    EXPECT_NO_NAN(tb.ch(1), 4096);

    // DC should be removed (check average in latter portion)
    float sum = 0.0f;
    for (int i = 2048; i < 4096; ++i)
        sum += tb.ch(0)[i];
    float avgDC = sum / 2048.0f;
    EXPECT_LT(std::abs(avgDC), 0.1f);
}

// ============================================================================
// Full signal chain: WAV -> process -> WAV -> read back
// ============================================================================

DSPARK_TEST(Integration_WAV_process_roundtrip)
{
    const char* inPath = "dspark_test_int_in.wav";
    const char* outPath = "dspark_test_int_out.wav";

    // Cleanup on exit
    struct Cleanup {
        const char* p1;
        const char* p2;
        ~Cleanup() { std::remove(p1); std::remove(p2); }
    } cleanup { inPath, outPath };

    constexpr int N = 8192;

    // Write input WAV
    {
        WavFile w;
        AudioFileInfo info;
        info.sampleRate = 44100.0;
        info.numChannels = 2;
        info.bitsPerSample = 16;
        info.numSamples = N;

        EXPECT_TRUE(w.openWrite(inPath, info));
        AudioBuffer<float> buf;
        buf.resize(2, N);
        generateSine(buf.getChannel(0), N, 440.0f, 44100.0f, 0.9f);
        generateSine(buf.getChannel(1), N, 440.0f, 44100.0f, 0.9f);
        EXPECT_TRUE(w.writeSamples(std::as_const(buf).toView()));
        w.close();
    }

    // Read, process, write
    {
        WavFile reader;
        EXPECT_TRUE(reader.openRead(inPath));
        auto info = reader.getInfo();

        AudioBuffer<float> buf;
        buf.resize(info.numChannels, static_cast<int>(info.numSamples));
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        // Apply gain
        Gain<float> gain;
        gain.prepare(info.sampleRate);
        gain.setGainDb(-6.0f);
        gain.skipRamp();
        gain.processBlock(buf.toView());

        // Write output
        WavFile writer;
        AudioFileInfo outInfo = info;
        outInfo.bitsPerSample = 16;
        EXPECT_TRUE(writer.openWrite(outPath, outInfo));
        EXPECT_TRUE(writer.writeSamples(std::as_const(buf).toView()));
        writer.close();
    }

    // Read back and verify
    {
        WavFile reader;
        EXPECT_TRUE(reader.openRead(outPath));
        auto info = reader.getInfo();

        AudioBuffer<float> buf;
        buf.resize(info.numChannels, static_cast<int>(info.numSamples));
        EXPECT_TRUE(reader.readSamples(buf.toView()));
        reader.close();

        // Peak should be ~half of original (-6 dB)
        float peak = measurePeak(buf.getChannel(0), static_cast<int>(info.numSamples));
        EXPECT_GT(peak, 0.35f);
        EXPECT_LT(peak, 0.55f);
    }
}

// ============================================================================
// Dual-type: double chain compiles and runs
// ============================================================================

DSPARK_TEST(Integration_double_template_compiles)
{
    ProcessorChain<double, DCBlocker<double>, Gain<double>> chain;
    AudioSpec s { .sampleRate = 48000.0, .maxBlockSize = 256, .numChannels = 2 };
    chain.prepare(s);
    chain.get<1>().setGainDb(-3.0);
    chain.get<1>().skipRamp();

    AudioBuffer<double> buf;
    buf.resize(2, 256);
    for (int ch = 0; ch < 2; ++ch)
        generateSine(buf.getChannel(ch), 256, 440.0, 48000.0);

    chain.processBlock(buf.toView());

    // Just verify it doesn't crash and produces valid output
    for (int i = 0; i < 256; ++i)
    {
        EXPECT_FALSE(std::isnan(buf.getChannel(0)[i]));
        EXPECT_FALSE(std::isinf(buf.getChannel(0)[i]));
    }
}

// ============================================================================
// ProcessorTraits - concept contract across the framework
// ============================================================================

// Every in-place single-buffer insert must satisfy AudioProcessor: the
// ProcessorChain static_assert rejects anything that does not (Saturation
// failed exactly this before its reset() was marked noexcept). Multi-buffer
// or per-call-parameterised utilities (Crossfade, CrossoverFilter, MidSide,
// AutoGain, Delay, Phasor) are intentionally out of contract; that scope is
// documented in ProcessorTraits.h and not asserted here so they may adopt
// the contract later without breaking this test.
DSPARK_TEST(ProcessorTraits_insert_effects_satisfy_AudioProcessor)
{
    EXPECT_TRUE((AudioProcessor<AlgorithmicReverb<float>, float>));
    EXPECT_TRUE((AudioProcessor<Chorus<float>, float>));
    EXPECT_TRUE((AudioProcessor<Clipper<float>, float>));
    EXPECT_TRUE((AudioProcessor<Compressor<float>, float>));
    EXPECT_TRUE((AudioProcessor<DCBlocker<float>, float>));
    EXPECT_TRUE((AudioProcessor<DeEsser<float>, float>));
    EXPECT_TRUE((AudioProcessor<DynamicEQ<float>, float>));
    EXPECT_TRUE((AudioProcessor<Equalizer<float>, float>));
    EXPECT_TRUE((AudioProcessor<Expander<float>, float>));
    EXPECT_TRUE((AudioProcessor<FrequencyShifter<float>, float>));
    EXPECT_TRUE((AudioProcessor<Gain<float>, float>));
    EXPECT_TRUE((AudioProcessor<GranularProcessor<float>, float>));
    EXPECT_TRUE((AudioProcessor<Limiter<float>, float>));
    EXPECT_TRUE((AudioProcessor<MultibandCompressor<float>, float>));
    EXPECT_TRUE((AudioProcessor<NoiseGate<float>, float>));
    EXPECT_TRUE((AudioProcessor<NoiseGenerator<float>, float>));
    EXPECT_TRUE((AudioProcessor<Panner<float>, float>));
    EXPECT_TRUE((AudioProcessor<Phaser<float>, float>));
    EXPECT_TRUE((AudioProcessor<PitchShifter<float>, float>));
    EXPECT_TRUE((AudioProcessor<Reverb<float>, float>));
    EXPECT_TRUE((AudioProcessor<RingModulator<float>, float>));
    EXPECT_TRUE((AudioProcessor<Saturation<float>, float>));
    EXPECT_TRUE((AudioProcessor<SpectralDenoiser<float>, float>));
    EXPECT_TRUE((AudioProcessor<StereoWidth<float>, float>));
    EXPECT_TRUE((AudioProcessor<TapeMachine<float>, float>));
    EXPECT_TRUE((AudioProcessor<TransformerModel<float>, float>));
    EXPECT_TRUE((AudioProcessor<TransientDesigner<float>, float>));
    EXPECT_TRUE((AudioProcessor<Tremolo<float>, float>));
    EXPECT_TRUE((AudioProcessor<TubePreamp<float>, float>));
    EXPECT_TRUE((AudioProcessor<Vibrato<float>, float>));

    // Double-precision spot checks (the contract is type-parametric).
    EXPECT_TRUE((AudioProcessor<Gain<double>, double>));
    EXPECT_TRUE((AudioProcessor<Saturation<double>, double>));
}

// SampleProcessor refines AudioProcessor with per-sample scalar processing;
// GeneratorProcessor is the source contract (getSample + generateBlock).
DSPARK_TEST(ProcessorTraits_sample_and_generator_contracts)
{
    EXPECT_TRUE((SampleProcessor<Compressor<float>, float>));
    EXPECT_TRUE((SampleProcessor<DCBlocker<float>, float>));
    EXPECT_TRUE((SampleProcessor<Equalizer<float>, float>));
    EXPECT_TRUE((SampleProcessor<Limiter<float>, float>));

    EXPECT_TRUE((GeneratorProcessor<Oscillator<float>, float>));
    EXPECT_TRUE((GeneratorProcessor<WavetableOscillator<float>, float>));
    EXPECT_TRUE((GeneratorProcessor<Oscillator<double>, double>));
}
