// DSPark Tests - Integration
// ProcessorChain composed, full signal chain, dual-type, processor concepts

#include "dspark_test.h"
#include "TestSignals.h"

#include "../DSPark.h" // full umbrella: the concept matrix below touches everything
#include "../plugin/DSParkPlugin.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

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
    // TimeStretch also offers a rate-changing pair of entry points beside the
    // in-place block call. That pair is deliberately outside this contract,
    // but the block call is inside it and must stay inside it: an insert that
    // stopped satisfying this would break every chain it sits in, and adding
    // a streaming surface is exactly the kind of change that could do it.
    EXPECT_TRUE((AudioProcessor<TimeStretch<float>, float>));
    EXPECT_TRUE((AudioProcessor<TransformerModel<float>, float>));
    EXPECT_TRUE((AudioProcessor<TransientDesigner<float>, float>));
    EXPECT_TRUE((AudioProcessor<Tremolo<float>, float>));
    EXPECT_TRUE((AudioProcessor<TubePreamp<float>, float>));
    EXPECT_TRUE((AudioProcessor<Vibrato<float>, float>));

    // Double-precision spot checks (the contract is type-parametric).
    EXPECT_TRUE((AudioProcessor<Gain<double>, double>));
    EXPECT_TRUE((AudioProcessor<Saturation<double>, double>));
    EXPECT_TRUE((AudioProcessor<TimeStretch<double>, double>));
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

// The assertions above say what must be accepted. They cannot, on their own,
// tell anyone what the contract EXCLUDES -- and for a long time it excluded
// almost nothing beyond a missing noexcept, because a class that only reads
// the buffer satisfied it too. These four shapes are the boundary, written as
// types so the boundary is checked rather than described.
namespace {

// The shape the contract must reject: reads the buffer, cannot write it.
struct ReadOnlyInsert
{
    void prepare(const AudioSpec&) {}
    void processBlock(AudioBufferView<const float>) noexcept {}
    void reset() noexcept {}
};

// The shape the contract must NOT reject: const handle, mutable samples. This
// is in-place and several inserts are free to be declared this way.
struct ConstHandleInsert
{
    void prepare(const AudioSpec&) {}
    void processBlock(const AudioBufferView<float>&) noexcept {}
    void reset() noexcept {}
};

// The same boundary on the generator side.
struct ReadOnlyGenerator
{
    void prepare(const AudioSpec&) {}
    void generateBlock(AudioBufferView<const float>) noexcept {}
    void reset() noexcept {}
    [[nodiscard]] float getSample() noexcept { return 0.0f; }
};

struct WritingGenerator
{
    void prepare(const AudioSpec&) {}
    void generateBlock(AudioBufferView<float>) noexcept {}
    void reset() noexcept {}
    [[nodiscard]] float getSample() noexcept { return 0.0f; }
};

} // namespace

DSPARK_TEST(ProcessorTraits_contract_excludes_processors_that_cannot_write)
{
    // A processor whose block call takes a const view is not an insert: it
    // cannot produce the output the next stage of a chain expects.
    EXPECT_FALSE((AudioProcessor<ReadOnlyInsert, float>));
    EXPECT_FALSE((SampleProcessor<ReadOnlyInsert, float>));
    EXPECT_FALSE((GeneratorProcessor<ReadOnlyGenerator, float>));

    // Constness of the HANDLE is not constness of the samples. This one writes
    // through the view it is given and must stay inside the contract.
    EXPECT_TRUE((AudioProcessor<ConstHandleInsert, float>));
    EXPECT_TRUE((GeneratorProcessor<WritingGenerator, float>));
}

// The shipped read-only analysers are the same boundary in production code.
// Each takes an AudioBufferView<const T> and returns measurements, so none of
// them is an insert; putting one in a ProcessorChain would silently give the
// following stage the untouched input. They satisfied the contract until the
// in-place clause existed, which is the reason these are pinned: the exclusion
// is a property of the contract, not an accident of how they happen to be
// written today.
DSPARK_TEST(ProcessorTraits_read_only_analysers_are_outside_the_insert_contract)
{
    EXPECT_FALSE((AudioProcessor<BeatTracker<float>, float>));
    EXPECT_FALSE((AudioProcessor<ChordDetector<float>, float>));
    EXPECT_FALSE((AudioProcessor<EnvelopeFollower<float>, float>));
    EXPECT_FALSE((AudioProcessor<LoudnessMeter<float>, float>));
    EXPECT_FALSE((AudioProcessor<OnsetDetector<float>, float>));
    EXPECT_FALSE((AudioProcessor<PhaseCorrelation<float>, float>));
    EXPECT_FALSE((AudioProcessor<PitchFollower<float>, float>));
}

// ============================================================================
// Plugin layer (plugin/DSParkPlugin.h) - normalisation, state container
// ============================================================================

namespace {

struct MiniPlug
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name = "Mini", .vendor = "T", .url = "", .email = "",
        .productId = "com.dspark.test.mini", .version = "1.0.0",
    };
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("gain", "Gain", -24.0f, 24.0f, 0.0f, "dB"),
        dspark::plugin::param("mix",  "Mix",    0.0f,  1.0f, 1.0f, ""),
        dspark::plugin::toggle("on", "On", true));

    std::vector<uint8_t> lastUserBlob;
    [[nodiscard]] std::vector<uint8_t> getState() const { return { 0xAB, 0xCD }; }
    bool setState(const uint8_t* d, size_t n)
    {
        lastUserBlob.assign(d, d + n);
        return true;
    }
};

} // namespace

DSPARK_TEST(PluginLayer_normalisation_and_hashing)
{
    using namespace dspark::plugin;

    // Continuous mapping is exact at the ends and the middle.
    constexpr Param g = MiniPlug::parameters[0];
    EXPECT_NEAR(toPlain(g, 0.0), -24.0, 1e-12);
    EXPECT_NEAR(toPlain(g, 1.0),  24.0, 1e-12);
    EXPECT_NEAR(toPlain(g, 0.5),   0.0, 1e-12);
    EXPECT_NEAR(toNormalized(g, 12.0), 0.75, 1e-12);

    // Out-of-range plain values clamp instead of leaving [0,1].
    EXPECT_NEAR(toNormalized(g, 1000.0), 1.0, 1e-12);
    EXPECT_NEAR(toNormalized(g, -1000.0), 0.0, 1e-12);

    // Toggles snap to exactly two positions.
    constexpr Param t = MiniPlug::parameters[2];
    EXPECT_NEAR(toPlain(t, 0.49), 0.0, 1e-12);
    EXPECT_NEAR(toPlain(t, 0.51), 1.0, 1e-12);

    // Toggle text round-trip (the clap-validator regression of v1.4.0).
    EXPECT_EQ(parseToggleText("On"), 1);
    EXPECT_EQ(parseToggleText("off"), 0);
    EXPECT_EQ(parseToggleText("0.5"), -1);
    EXPECT_EQ(parseToggleText(nullptr), -1);

    // Ids hash uniquely and clash with no reserved state id - compile-time.
    static_assert(paramIdsUnique<MiniPlug>());

    // UIDs derive deterministically from the productId.
    constexpr auto uidA = makeUid(MiniPlug::descriptor.productId, 1);
    constexpr auto uidB = makeUid(MiniPlug::descriptor.productId, 1);
    constexpr auto uidC = makeUid(MiniPlug::descriptor.productId, 2);
    EXPECT_TRUE(uidA == uidB);
    EXPECT_TRUE(uidA != uidC);
}

DSPARK_TEST(PluginLayer_state_roundtrip_and_hostile_blobs)
{
    using namespace dspark::plugin;

    MiniPlug plug;
    double norm[3] = { 0.25, 0.75, 1.0 };
    auto blob = buildState(plug, norm, 3, 2, 1);

    // Round-trip restores every value, the program index, the bypass and
    // the user section.
    MiniPlug back;
    double restored[3] = { 0.0, 0.0, 0.0 };
    int program = -1, bypass = -1;
    EXPECT_TRUE(applyState(back, blob.data(), blob.size(), restored, &program, &bypass));
    EXPECT_NEAR(restored[0], 0.25, 1e-12);
    EXPECT_NEAR(restored[1], 0.75, 1e-12);
    EXPECT_NEAR(restored[2], 1.0, 1e-12);
    EXPECT_EQ(program, 2);
    EXPECT_EQ(bypass, 1);
    EXPECT_EQ(int(back.lastUserBlob.size()), 2);

    // Foreign blob rejected.
    const uint8_t junk[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    EXPECT_TRUE(!applyState(back, junk, 8, restored));

    // Truncated blob at every prefix: never crashes, only the full one parses.
    for (size_t cut = 0; cut < blob.size(); ++cut)
    {
        MiniPlug victim;
        double tmp[3] = { 0.5, 0.5, 0.5 };
        (void) applyState(victim, blob.data(), cut, tmp);
    }

    // Hostile user-section length (0xFFFFFFFF): the subtraction-form bounds
    // check must reject it. On 32-bit size_t targets the old addition form
    // wrapped around and passed a giant out-of-bounds span to setState.
    {
        auto evil = blob;
        const size_t userLenPos = evil.size() - 2 - 4; // [len][2 user bytes]
        evil[userLenPos]     = 0xFF;
        evil[userLenPos + 1] = 0xFF;
        evil[userLenPos + 2] = 0xFF;
        evil[userLenPos + 3] = 0xFF;
        MiniPlug victim;
        double tmp[3] = { 0.5, 0.5, 0.5 };
        EXPECT_TRUE(applyState(victim, evil.data(), evil.size(), tmp));
        EXPECT_EQ(int(victim.lastUserBlob.size()), 0); // user section refused
    }

    // NaN parameter entries keep the default instead of poisoning the shadow.
    {
        auto evil = blob;
        // First entry value starts after magic+version+count+id = 16 bytes.
        for (int b = 0; b < 8; ++b) evil[16 + size_t(b)] = 0xFF; // a quiet NaN
        MiniPlug victim;
        double tmp[3] = { 0.111, 0.222, 0.333 };
        EXPECT_TRUE(applyState(victim, evil.data(), evil.size(), tmp));
        EXPECT_NEAR(tmp[0], 0.111, 1e-12); // untouched default
        EXPECT_NEAR(tmp[1], 0.75, 1e-12);  // later entries still apply
    }
}

namespace {

struct FixedPlug
{
    static constexpr dspark::plugin::EditorSize editorSize { 400, 300 };
};

struct AspectPlug
{
    static constexpr dspark::plugin::EditorSize editorSize { 400, 300 };
    static constexpr dspark::plugin::EditorResize editorResize =
        dspark::plugin::EditorResize::KeepAspect;
};

} // namespace

DSPARK_TEST(PluginLayer_editor_size_policy)
{
    using namespace dspark::plugin;

    double w = 999.0, h = 111.0;
    constrainEditorSize<FixedPlug>(w, h, 1.0);
    EXPECT_NEAR(w, 400.0, 1e-9);
    EXPECT_NEAR(h, 300.0, 1e-9);

    // Inside-fit: never exceed the host proposal on either axis.
    w = 800.0; h = 300.0;
    constrainEditorSize<AspectPlug>(w, h, 1.0);
    EXPECT_NEAR(w, 400.0, 1e-9);   // limited by height * ratio
    EXPECT_NEAR(h, 300.0, 1e-9);
    EXPECT_TRUE(w <= 800.0 && h <= 300.0);

    // Clamped to the 0.5x..3x window.
    w = 10.0; h = 10.0;
    constrainEditorSize<AspectPlug>(w, h, 1.0);
    EXPECT_NEAR(w, 200.0, 1e-9);
    EXPECT_NEAR(h, 150.0, 1e-9);
}
