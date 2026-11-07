// DSPark Tests - Integration
// ProcessorChain composed, full signal chain, dual-type, processor concepts

// <windows.h> FIRST and without NOMINMAX, as a desktop application includes
// it: its function-like min/max macros must not break the umbrella or the
// plugin headers. Before the umbrella parked them, this translation unit did
// not compile (81 errors, the first in Core/DspMath.h).
#if defined(_WIN32)
#include <windows.h>
#endif

#include "../DSPark.h" // full umbrella: the concept matrix below touches everything
#include "../plugin/DSParkPlugin.h"
#include "../plugin/vst3/DSParkVst3.h"
#include "../plugin/clap/DSParkClap.h"

#if defined(_WIN32)
#undef min   // the rest of this file uses std::min/std::max freely
#undef max
#endif

#include "dspark_test.h"
#include "TestSignals.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
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
// AutoGain, Phasor) are intentionally out of contract; that scope is
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
    EXPECT_TRUE((AudioProcessor<Delay<float>, float>));
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

    // Named choices and stepped parameters display and parse their own text
    // (hosts round-trip it through automation lanes and typed values).
    static constexpr const char* kModes[] = { "Clean", "Warm", "Hot" };
    constexpr Param mode = choice("mode", "Mode", kModes, 1);
    static_assert(mode.steps == 2 && mode.maxValue == 2.0f && mode.defValue == 1.0f);
    char text[32] {};
    formatValue(mode, 2.0, text, sizeof(text));
    EXPECT_TRUE(std::strcmp(text, "Hot") == 0);
    double plain = -1.0;
    EXPECT_TRUE(parseValue(mode, "warm", plain));
    EXPECT_NEAR(plain, 1.0, 1e-12);
    EXPECT_TRUE(parseValue(mode, "2", plain));
    EXPECT_NEAR(plain, 2.0, 1e-12);
    EXPECT_TRUE(!parseValue(mode, "Loud", plain));

    constexpr Param voices = stepped("voices", "Voices", 1.0f, 8.0f, 4.0f, 7);
    formatValue(voices, 5.0, text, sizeof(text));
    EXPECT_TRUE(std::strcmp(text, "5") == 0);
    EXPECT_TRUE(parseValue(voices, "6.4", plain));
    EXPECT_NEAR(plain, 6.0, 1e-12);   // snapped to the nearest position

    // Garbage and NaN are refused; numbers clamp; toggles take On/Off.
    EXPECT_TRUE(!parseValue(g, "abc", plain));
    EXPECT_TRUE(!parseValue(g, "nan", plain));
    EXPECT_TRUE(parseValue(g, "100 dB", plain));
    EXPECT_NEAR(plain, 24.0, 1e-12);
    EXPECT_TRUE(parseValue(t, "OFF", plain));
    EXPECT_NEAR(plain, 0.0, 1e-12);
    EXPECT_NEAR(toNormalized(g, std::nan("")), 0.0, 1e-12);

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

// ============================================================================
// Plugin format wrappers (plugin/vst3, plugin/clap) driven through their C
// vtables with minimal host objects. Each case names the defect it pins and
// what the old wrapper did.
// ============================================================================

namespace {

struct WrapPlug
{
    static constexpr const char* kWaves[] = { "Sine", "Saw" };
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name = "Wrap", .vendor = "T", .url = "", .email = "",
        .productId = "com.dspark.test.wrap", .version = "1.0.0",
    };
    // A UTF-8 name and unit (U+00F6 U+00DF, U+00B5) and a five-position
    // stepped control on a 0..1 range.
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("size", "Gr\xC3\xB6\xC3\x9F" "e", 0.0f, 1.0f, 0.5f, "\xC2\xB5s"),
        dspark::plugin::toggle("look", "Lookahead", false),
        dspark::plugin::stepped("frac", "Frac", 0.0f, 1.0f, 0.5f, 4),
        dspark::plugin::choice("wave", "Wave", kWaves, 0));

    std::atomic<std::thread::id> lastSetter {};
    std::atomic<bool> look { false };
    std::atomic<int> resets { 0 };
    std::atomic<int> lastCc { -1 };
    std::atomic<float> lastCcValue { 0.0f };
    size_t userStateBytes = 0;
    bool userStateRestored = false;

    void prepare(const dspark::AudioSpec&) {}
    void setParameter(int index, float value) noexcept
    {
        lastSetter.store(std::this_thread::get_id());
        if (index == 1) look.store(value >= 0.5f);
    }
    void processBlock(dspark::AudioBufferView<float>) noexcept {}
    void reset() noexcept { ++resets; }
    void handleMidiEvent(const dspark::plugin::MidiEvent& e) noexcept
    {
        if (e.type == dspark::plugin::MidiEvent::Type::ControlChange)
        {
            lastCcValue.store(e.value);
            lastCc.store(e.note);
        }
    }
    [[nodiscard]] int getLatency() const noexcept { return look.load() ? 64 : 0; }
    [[nodiscard]] std::vector<uint8_t> getState() const
    { return std::vector<uint8_t>(userStateBytes, uint8_t { 0x5A }); }
    bool setState(const uint8_t*, size_t n)
    {
        userStateRestored = n == userStateBytes;
        return true;
    }
};

using Vst3Wrap = dspark::plugin::vst3::Plugin<WrapPlug>;

struct MockQueue
{
    Steinberg_Vst_IParamValueQueueVtbl* vtbl;
    uint32_t id = 0;
    std::vector<std::pair<Steinberg_int32, double>> points;
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ref(void*) { return 1; }
    static Steinberg_Vst_ParamID SMTG_STDMETHODCALLTYPE getId(void* s)
    { return static_cast<MockQueue*>(s)->id; }
    static Steinberg_int32 SMTG_STDMETHODCALLTYPE count(void* s)
    { return static_cast<Steinberg_int32>(static_cast<MockQueue*>(s)->points.size()); }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE get(void* s, Steinberg_int32 i,
                                                        Steinberg_int32* offset,
                                                        Steinberg_Vst_ParamValue* value)
    {
        const auto& p = static_cast<MockQueue*>(s)->points[static_cast<size_t>(i)];
        *offset = p.first;
        *value = p.second;
        return Steinberg_kResultOk;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE add(void*, Steinberg_int32,
                                                        Steinberg_Vst_ParamValue, Steinberg_int32*)
    { return Steinberg_kResultFalse; }
    static inline Steinberg_Vst_IParamValueQueueVtbl kVtbl = { &q, &ref, &ref, &getId, &count, &get, &add };
    MockQueue() : vtbl(&kVtbl) {}
};

struct MockChanges
{
    Steinberg_Vst_IParameterChangesVtbl* vtbl;
    std::vector<MockQueue> queues;
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ref(void*) { return 1; }
    static Steinberg_int32 SMTG_STDMETHODCALLTYPE count(void* s)
    { return static_cast<Steinberg_int32>(static_cast<MockChanges*>(s)->queues.size()); }
    static Steinberg_Vst_IParamValueQueue* SMTG_STDMETHODCALLTYPE get(void* s, Steinberg_int32 i)
    {
        return reinterpret_cast<Steinberg_Vst_IParamValueQueue*>(
            &static_cast<MockChanges*>(s)->queues[static_cast<size_t>(i)]);
    }
    static Steinberg_Vst_IParamValueQueue* SMTG_STDMETHODCALLTYPE add(void*, const Steinberg_Vst_ParamID*,
                                                                      Steinberg_int32*)
    { return nullptr; }
    static inline Steinberg_Vst_IParameterChangesVtbl kVtbl = { &q, &ref, &ref, &count, &get, &add };
    MockChanges() : vtbl(&kVtbl) {}
    void point(uint32_t id, Steinberg_int32 offset, double value)
    {
        MockQueue queue;
        queue.id = id;
        queue.points.push_back({ offset, value });
        queues.push_back(queue);
    }
};

struct MockHandler
{
    Steinberg_Vst_IComponentHandlerVtbl* vtbl;
    std::thread::id ui = std::this_thread::get_id();
    std::atomic<int> refs { 1 }, restarts { 0 }, offThread { 0 };
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE addRef(void* s)
    { return static_cast<Steinberg_uint32>(++static_cast<MockHandler*>(s)->refs); }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE release(void* s)
    { return static_cast<Steinberg_uint32>(--static_cast<MockHandler*>(s)->refs); }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE edit(void*, Steinberg_Vst_ParamID)
    { return Steinberg_kResultOk; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE perform(void*, Steinberg_Vst_ParamID,
                                                            Steinberg_Vst_ParamValue)
    { return Steinberg_kResultOk; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE restart(void* s, Steinberg_int32)
    {
        auto* h = static_cast<MockHandler*>(s);
        ++h->restarts;
        if (std::this_thread::get_id() != h->ui) ++h->offThread;
        return Steinberg_kResultOk;
    }
    static inline Steinberg_Vst_IComponentHandlerVtbl kVtbl = { &q, &addRef, &release, &edit,
                                                                &perform, &edit, &restart };
    MockHandler() : vtbl(&kVtbl) {}
};

struct MockStream
{
    Steinberg_IBStreamVtbl* vtbl;
    std::vector<uint8_t> bytes;
    size_t pos = 0;
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ref(void*) { return 1; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE read(void* s_, void* dst, Steinberg_int32 n,
                                                         Steinberg_int32* got)
    {
        auto* s = static_cast<MockStream*>(s_);
        const size_t left = s->bytes.size() - s->pos;
        const size_t take = left < static_cast<size_t>(n) ? left : static_cast<size_t>(n);
        if (take > 0) std::memcpy(dst, s->bytes.data() + s->pos, take);
        s->pos += take;
        if (got) *got = static_cast<Steinberg_int32>(take);
        return take > 0 ? Steinberg_kResultOk : Steinberg_kResultFalse;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE write(void* s_, void* src, Steinberg_int32 n,
                                                          Steinberg_int32* put)
    {
        auto* s = static_cast<MockStream*>(s_);
        const auto* p = static_cast<const uint8_t*>(src);
        s->bytes.insert(s->bytes.end(), p, p + n);
        if (put) *put = n;
        return Steinberg_kResultOk;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE seek(void* s_, Steinberg_int64 p, Steinberg_int32,
                                                         Steinberg_int64* out)
    {
        static_cast<MockStream*>(s_)->pos = static_cast<size_t>(p);
        if (out) *out = p;
        return Steinberg_kResultOk;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE tell(void* s_, Steinberg_int64* out)
    {
        *out = static_cast<Steinberg_int64>(static_cast<MockStream*>(s_)->pos);
        return Steinberg_kResultOk;
    }
    static inline Steinberg_IBStreamVtbl kVtbl = { &q, &ref, &ref, &read, &write, &seek, &tell };
    MockStream() : vtbl(&kVtbl) {}
};

Vst3Wrap* makeActiveVst3()
{
    auto* p = new Vst3Wrap();
    Steinberg_Vst_ProcessSetup setup { 0, Steinberg_Vst_SymbolicSampleSizes_kSample32, 256, 48000.0 };
    p->kProcessorVtbl.setupProcessing(p->lensPtr(1), &setup);
    p->kComponentVtbl.setActive(p->lensPtr(0), 1);
    p->kProcessorVtbl.setProcessing(p->lensPtr(1), 1);
    return p;
}

Steinberg_tresult processVst3(Vst3Wrap* p, float** out, Steinberg_int32 frames, MockChanges* changes)
{
    Steinberg_Vst_AudioBusBuffers outBus {};
    outBus.numChannels = 2;
    outBus.Steinberg_Vst_AudioBusBuffers_channelBuffers32 = out;
    Steinberg_Vst_ProcessData data {};
    data.symbolicSampleSize = Steinberg_Vst_SymbolicSampleSizes_kSample32;
    data.numSamples = frames;
    data.numOutputs = 1;
    data.outputs = &outBus;
    data.inputParameterChanges = reinterpret_cast<Steinberg_Vst_IParameterChanges*>(changes);
    return p->kProcessorVtbl.process(p->lensPtr(1), &data);
}

} // namespace

DSPARK_TEST(PluginVst3_ui_edits_reach_setParameter_on_the_audio_thread)
{
    // While the host processes, the user's setParameter must have ONE caller:
    // DSPark setters such as Equalizer::setBand are single-writer seqlocks,
    // and two concurrent writers tear a band (measured: 14386 torn band
    // readouts in 3 s). The old wrapper called it from setParamNormalized on
    // the UI thread while process() called it on the audio thread.
    auto* p = makeActiveVst3();
    p->user.lastSetter.store(std::thread::id {});
    p->kControllerVtbl.setParamNormalized(p->lensPtr(2), dspark::plugin::hash32("size"), 0.8);
    EXPECT_TRUE(p->user.lastSetter.load() != std::this_thread::get_id());
    std::thread::id audioThread;
    std::thread audio([&] {
        audioThread = std::this_thread::get_id();
        std::vector<float> l(256), r(256);
        float* out[2] = { l.data(), r.data() };
        processVst3(p, out, 256, nullptr);
    });
    audio.join();
    EXPECT_TRUE(p->user.lastSetter.load() == audioThread);
    EXPECT_NEAR(p->kControllerVtbl.getParamNormalized(p->lensPtr(2),
                                                      dspark::plugin::hash32("size")), 0.8, 1e-12);
    p->releaseImpl();
}

DSPARK_TEST(PluginVst3_restart_is_never_requested_off_the_UI_thread)
{
    // A latency change found in process() must reach restartComponent on the
    // UI thread only. With no UI tick (a Linux host whose initialize()
    // context has no IRunLoop - here: no initialize at all) the old wrapper
    // called the host straight from the audio thread.
    auto* p = makeActiveVst3();
    MockHandler handler;
    p->kControllerVtbl.setComponentHandler(p->lensPtr(2),
        reinterpret_cast<Steinberg_Vst_IComponentHandler*>(&handler));
    std::thread audio([&] {
        std::vector<float> l(256), r(256);
        float* out[2] = { l.data(), r.data() };
        MockChanges changes;
        changes.point(dspark::plugin::hash32("look"), 0, 1.0);
        processVst3(p, out, 256, &changes);
    });
    audio.join();
    EXPECT_EQ(handler.offThread.load(), 0);
    (void) p->kControllerVtbl.getParamNormalized(p->lensPtr(2), dspark::plugin::hash32("look"));
    EXPECT_EQ(handler.restarts.load(), 1);   // delivered by the UI-thread call
    EXPECT_EQ(static_cast<int>(p->kProcessorVtbl.getLatencySamples(p->lensPtr(1))), 64);
    p->kControllerVtbl.setComponentHandler(p->lensPtr(2), nullptr);
    p->releaseImpl();
}

DSPARK_TEST(PluginVst3_terminate_releases_the_component_handler)
{
    // Steinberg's EditController::terminate releases the handler; the old
    // wrapper kept its reference until the plugin's last release, after the
    // host had torn the handler down.
    auto* p = new Vst3Wrap();
    MockHandler handler;
    p->kControllerVtbl.initialize(p->lensPtr(2), nullptr);
    p->kControllerVtbl.setComponentHandler(p->lensPtr(2),
        reinterpret_cast<Steinberg_Vst_IComponentHandler*>(&handler));
    EXPECT_EQ(handler.refs.load(), 2);
    p->kControllerVtbl.terminate(p->lensPtr(2));
    EXPECT_EQ(handler.refs.load(), 1);
    p->releaseImpl();
    EXPECT_EQ(handler.refs.load(), 1);
}

DSPARK_TEST(PluginVst3_utf8_strings_become_utf16)
{
    // Names and units are UTF-8; VST3 strings are UTF-16. The old wrapper
    // widened byte by byte: "Gr\xC3\xB6\xC3\x9F" "e" became 7 units of mojibake.
    auto* p = new Vst3Wrap();
    Steinberg_Vst_ParameterInfo info {};
    p->kControllerVtbl.getParameterInfo(p->lensPtr(2), 0, &info);
    int len = 0;
    while (info.title[len] != 0) ++len;
    EXPECT_EQ(len, 5);
    EXPECT_EQ(static_cast<int>(info.title[2]), 0x00F6);
    EXPECT_EQ(static_cast<int>(info.title[3]), 0x00DF);
    EXPECT_EQ(static_cast<int>(info.units[0]), 0x00B5);
    EXPECT_EQ(static_cast<int>(info.units[1]), 's');
    // A host typing the displayed text back gets the same value.
    Steinberg_Vst_String128 text {};
    p->kControllerVtbl.getParamStringByValue(p->lensPtr(2), dspark::plugin::hash32("size"), 0.25, text);
    Steinberg_Vst_ParamValue back = -1.0;
    EXPECT_EQ(p->kControllerVtbl.getParamValueByString(p->lensPtr(2), dspark::plugin::hash32("size"),
                                                      text, &back), Steinberg_kResultOk);
    EXPECT_NEAR(back, 0.25, 1e-9);
    p->releaseImpl();
}

DSPARK_TEST(PluginVst3_midi_mapping_covers_every_controller)
{
    // Every CC, channel pressure and pitch bend maps through IMidiMapping
    // and reaches handleMidiEvent. The old table mapped only CC1, CC64,
    // pressure and bend; CC74 here was refused.
    auto* p = makeActiveVst3();
    Steinberg_Vst_ParamID cc74 = 0;
    EXPECT_EQ(p->kMidiMapVtbl.getMidiControllerAssignment(p->lensPtr(5), 0, 3, 74, &cc74),
              Steinberg_kResultTrue);
    MockChanges changes;
    changes.point(cc74, 0, 0.5);
    std::vector<float> l(256), r(256);
    float* out[2] = { l.data(), r.data() };
    processVst3(p, out, 256, &changes);
    EXPECT_EQ(p->user.lastCc.load(), 74);
    EXPECT_NEAR(p->user.lastCcValue.load(), 0.5f, 1e-6f);
    // The pitch-bend proxy rests at its declared centre (the old shadow
    // started at 0.0 = full bend down while the info said 0.5).
    EXPECT_NEAR(p->kControllerVtbl.getParamNormalized(p->lensPtr(2),
                                                      dspark::plugin::vst3::midiProxyId(0, 129)),
                0.5, 1e-12);
    p->releaseImpl();
}

DSPARK_TEST(PluginVst3_stop_resets_and_large_states_round_trip)
{
    // setProcessing(false) reaches the plugin's reset() (the old no-op never
    // did, so tails rang on across a transport stop), and a user state over
    // the old 16 MiB read cap - one getState() itself wrote - loads back.
    auto* p = makeActiveVst3();
    const int resetsBefore = p->user.resets.load();
    p->kProcessorVtbl.setProcessing(p->lensPtr(1), 0);
    EXPECT_EQ(p->user.resets.load(), resetsBefore + 1);
    p->user.userStateBytes = size_t { 17 } << 20;
    MockStream saved;
    EXPECT_EQ(p->kComponentVtbl.getState(p->lensPtr(0), reinterpret_cast<Steinberg_IBStream*>(&saved)),
              Steinberg_kResultOk);
    auto* q = new Vst3Wrap();
    q->user.userStateBytes = p->user.userStateBytes;
    saved.pos = 0;
    EXPECT_EQ(q->kComponentVtbl.setState(q->lensPtr(0), reinterpret_cast<Steinberg_IBStream*>(&saved)),
              Steinberg_kResultOk);
    EXPECT_TRUE(q->user.userStateRestored);
    q->releaseImpl();
    p->releaseImpl();
}

DSPARK_TEST(PluginVst3_processing_start_resets_the_history)
{
    // setProcessing(true) clears the processing history too, so a host that
    // starts processing again without a stop in between (or that never
    // sends the stop) does not resume the tails of the previous run. The old
    // wrapper reset only on setProcessing(false).
    auto* p = makeActiveVst3();   // setActive(true), setProcessing(true)
    const int afterFirstStart = p->user.resets.load();
    EXPECT_EQ(afterFirstStart, 1);
    std::vector<float> l(256), r(256);
    float* out[2] = { l.data(), r.data() };
    EXPECT_EQ(processVst3(p, out, 256, nullptr), Steinberg_kResultOk);
    p->kProcessorVtbl.setProcessing(p->lensPtr(1), 1);   // restart, no stop
    EXPECT_EQ(p->user.resets.load(), afterFirstStart + 1);
    p->releaseImpl();
}

DSPARK_TEST(PluginVst3_null_left_output_channel_is_not_written)
{
    // A host handing a null left channel next to a valid right one: the old
    // trim looked only at the last pointer and wrote through the null one.
    auto* p = makeActiveVst3();
    std::vector<float> r(256, 0.25f);
    float* out[2] = { nullptr, r.data() };
    EXPECT_EQ(processVst3(p, out, 256, nullptr), Steinberg_kResultOk);
    EXPECT_NEAR(r[100], 0.25f, 0.0f);
    p->releaseImpl();
}

DSPARK_TEST(PluginClap_stepped_flag_only_for_integer_grids)
{
    // CLAP_PARAM_IS_STEPPED means integer values only. A five-position 0..1
    // control flagged stepped (the old behaviour) offers two positions in a
    // host; whole-number grids (toggle, choice) keep the flag.
    using Wrap = dspark::plugin::clap_backend::Plugin<WrapPlug>;
    clap_param_info_t info {};
    EXPECT_TRUE(Wrap::kParams.get_info(nullptr, 2, &info));
    EXPECT_EQ(info.flags & CLAP_PARAM_IS_STEPPED, 0u);
    EXPECT_TRUE(Wrap::kParams.get_info(nullptr, 1, &info));
    EXPECT_NE(info.flags & CLAP_PARAM_IS_STEPPED, 0u);
    EXPECT_TRUE(Wrap::kParams.get_info(nullptr, 3, &info));
    EXPECT_NE(info.flags & CLAP_PARAM_IS_STEPPED, 0u);
    EXPECT_NE(info.flags & CLAP_PARAM_IS_ENUM, 0u);
}

namespace {

// Records every MIDI event it receives; opts out of sample-accurate
// automation, so parameter queues collapse to their last point per block.
struct MidiLogPlug
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name = "MidiLog", .vendor = "T", .url = "", .email = "",
        .productId = "com.dspark.test.midilog", .version = "1.0.0",
    };
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("gain", "Gain", 0.0f, 1.0f, 1.0f, ""));
    static constexpr bool sampleAccurateAutomation = false;

    std::vector<dspark::plugin::MidiEvent> log;

    MidiLogPlug() { log.reserve(64); }
    void prepare(const dspark::AudioSpec&) {}
    void setParameter(int, float) noexcept {}
    void processBlock(dspark::AudioBufferView<float>) noexcept {}
    void handleMidiEvent(const dspark::plugin::MidiEvent& e) noexcept
    {
        if (log.size() < log.capacity()) log.push_back(e);
    }
};

/** A minimal CLAP input-event list (storage reserved up front, so the
 *  header pointers stay valid). */
struct ClapEvents
{
    clap_input_events_t iface {};
    std::vector<clap_event_note_t> notes;
    std::vector<clap_event_midi_t> midi;
    std::vector<clap_event_param_value_t> values;
    std::vector<clap_event_param_mod_t> mods;
    std::vector<const clap_event_header_t*> order;
    static uint32_t size(const clap_input_events_t* l)
    { return static_cast<uint32_t>(static_cast<const ClapEvents*>(l->ctx)->order.size()); }
    static const clap_event_header_t* get(const clap_input_events_t* l, uint32_t i)
    { return static_cast<const ClapEvents*>(l->ctx)->order[i]; }
    ClapEvents()
    {
        iface.ctx = this;
        iface.size = &size;
        iface.get = &get;
        notes.reserve(16);
        midi.reserve(16);
        values.reserve(512);
        mods.reserve(16);
    }
    void note(uint16_t type, int16_t channel, int16_t key, uint32_t time)
    {
        clap_event_note_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = type;
        ev.note_id = -1;
        ev.port_index = 0;
        ev.channel = channel;
        ev.key = key;
        ev.velocity = 1.0;
        notes.push_back(ev);
        order.push_back(&notes.back().header);
    }
    void cc(uint8_t controller, uint8_t value, uint32_t time)
    {
        clap_event_midi_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_MIDI;
        ev.data[0] = 0xB2;   // CC on channel 3
        ev.data[1] = controller;
        ev.data[2] = value;
        midi.push_back(ev);
        order.push_back(&midi.back().header);
    }
    void value(clap_id id, double plain, uint32_t time)
    {
        clap_event_param_value_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_PARAM_VALUE;
        ev.param_id = id;
        ev.note_id = -1; ev.port_index = -1; ev.channel = -1; ev.key = -1;
        ev.value = plain;
        values.push_back(ev);
        order.push_back(&values.back().header);
    }
    void mod(clap_id id, double amount, uint32_t time, int32_t noteId = -1)
    {
        clap_event_param_mod_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_PARAM_MOD;
        ev.param_id = id;
        ev.note_id = noteId; ev.port_index = -1; ev.channel = -1; ev.key = -1;
        ev.amount = amount;
        mods.push_back(ev);
        order.push_back(&mods.back().header);
    }
    void clear()
    {
        notes.clear();
        midi.clear();
        values.clear();
        mods.clear();
        order.clear();
    }
};

template <typename Wrap>
void processClap(Wrap* s, ClapEvents& events, uint32_t frames)
{
    std::vector<float> l(frames), r(frames);
    float* out[2] = { l.data(), r.data() };
    clap_audio_buffer_t outBuf {};
    outBuf.data32 = out;
    outBuf.channel_count = 2;
    clap_process_t proc {};
    proc.steady_time = -1;
    proc.frames_count = frames;
    proc.audio_outputs = &outBuf;
    proc.audio_outputs_count = 1;
    proc.in_events = &events.iface;
    s->plugin.process(&s->plugin, &proc);
    events.clear();
}

} // namespace

DSPARK_TEST(PluginVst3_controller_points_survive_block_rate_automation)
{
    // A MIDI controller proxy carries MIDI events, not automation: a plugin
    // that opts out of sample-accurate automation must still receive every
    // controller value with its offset, as the CLAP and AU backends deliver
    // raw MIDI. The old wrapper kept only the last point of each proxy
    // queue: 1 of these 4 CC11 events reached handleMidiEvent.
    using Wrap = dspark::plugin::vst3::Plugin<MidiLogPlug>;
    auto* p = new Wrap();
    Steinberg_Vst_ProcessSetup setup { 0, Steinberg_Vst_SymbolicSampleSizes_kSample32, 512, 48000.0 };
    p->kProcessorVtbl.setupProcessing(p->lensPtr(1), &setup);
    p->kComponentVtbl.setActive(p->lensPtr(0), 1);
    p->kProcessorVtbl.setProcessing(p->lensPtr(1), 1);
    MockChanges changes;
    MockQueue queue;
    queue.id = dspark::plugin::vst3::midiProxyId(2, 11);
    for (int k = 0; k < 4; ++k)
        queue.points.push_back({ 100 * k, 0.25 * (k + 1) });
    changes.queues.push_back(queue);
    std::vector<float> l(512), r(512);
    float* out[2] = { l.data(), r.data() };
    Steinberg_Vst_AudioBusBuffers outBus {};
    outBus.numChannels = 2;
    outBus.Steinberg_Vst_AudioBusBuffers_channelBuffers32 = out;
    Steinberg_Vst_ProcessData data {};
    data.symbolicSampleSize = Steinberg_Vst_SymbolicSampleSizes_kSample32;
    data.numSamples = 512;
    data.numOutputs = 1;
    data.outputs = &outBus;
    data.inputParameterChanges = reinterpret_cast<Steinberg_Vst_IParameterChanges*>(&changes);
    EXPECT_EQ(p->kProcessorVtbl.process(p->lensPtr(1), &data), Steinberg_kResultOk);
    const auto& log = p->user.log;
    EXPECT_EQ(static_cast<int>(log.size()), 4);
    for (size_t k = 0; k < log.size() && k < 4; ++k)
    {
        EXPECT_TRUE(log[k].type == dspark::plugin::MidiEvent::Type::ControlChange);
        EXPECT_EQ(static_cast<int>(log[k].channel), 2);
        EXPECT_EQ(static_cast<int>(log[k].note), 11);
        EXPECT_EQ(log[k].sampleOffset, 100 * static_cast<int>(k));
        EXPECT_NEAR(log[k].value, 0.25f * static_cast<float>(k + 1), 1e-6f);
    }
    p->releaseImpl();

    // The same plugin through CLAP raw MIDI: every point, with its offset.
    using ClapWrap = dspark::plugin::clap_backend::Plugin<MidiLogPlug>;
    const clap_plugin_t* cp = ClapWrap::create(nullptr);
    auto* c = static_cast<ClapWrap*>(cp->plugin_data);
    cp->init(cp);
    cp->activate(cp, 48000.0, 32, 512);
    ClapEvents events;
    for (uint32_t k = 0; k < 4; ++k)
        events.cc(11, static_cast<uint8_t>(32 * (k + 1) - 1), 100 * k);
    processClap(c, events, 512);
    EXPECT_EQ(static_cast<int>(c->user.log.size()), 4);
    if (c->user.log.size() == 4)
    {
        EXPECT_EQ(static_cast<int>(c->user.log[3].channel), 2);
        EXPECT_EQ(c->user.log[3].sampleOffset, 300);
    }
    cp->deactivate(cp);
    cp->destroy(cp);
}

DSPARK_TEST(PluginClap_choke_arrives_as_a_noteoff_with_the_choke_flag)
{
    // NOTE_CHOKE means "end the voice now, without its release". It arrives
    // as a NoteOff (a plugin that ignores the flag still releases the voice)
    // with MidiEvent::choke set; a NOTE_OFF leaves it clear. The old wrapper
    // delivered a choke as a plain NoteOff, so a choked voice rang through
    // its whole release.
    using Wrap = dspark::plugin::clap_backend::Plugin<MidiLogPlug>;
    const clap_plugin_t* cp = Wrap::create(nullptr);
    auto* s = static_cast<Wrap*>(cp->plugin_data);
    cp->init(cp);
    cp->activate(cp, 48000.0, 32, 256);
    using Type = dspark::plugin::MidiEvent::Type;
    ClapEvents events;
    events.note(CLAP_EVENT_NOTE_ON, 0, 60, 0);
    events.note(CLAP_EVENT_NOTE_ON, 0, 64, 0);
    events.note(CLAP_EVENT_NOTE_ON, 0, 67, 0);
    processClap(s, events, 256);
    s->user.log.clear();
    events.note(CLAP_EVENT_NOTE_CHOKE, 0, 60, 10);   // exact address
    events.note(CLAP_EVENT_NOTE_OFF, 0, 64, 20);     // an ordinary release
    processClap(s, events, 256);
    EXPECT_EQ(static_cast<int>(s->user.log.size()), 2);
    if (s->user.log.size() == 2)
    {
        EXPECT_TRUE(s->user.log[0].type == Type::NoteOff);
        EXPECT_EQ(static_cast<int>(s->user.log[0].note), 60);
        EXPECT_TRUE(s->user.log[0].choke);
        EXPECT_TRUE(s->user.log[1].type == Type::NoteOff);
        EXPECT_EQ(static_cast<int>(s->user.log[1].note), 64);
        EXPECT_TRUE(!s->user.log[1].choke);
    }
    s->user.log.clear();
    events.note(CLAP_EVENT_NOTE_CHOKE, -1, -1, 0);   // wildcard: every sounding note
    processClap(s, events, 256);
    EXPECT_EQ(static_cast<int>(s->user.log.size()), 1);
    if (s->user.log.size() == 1)
    {
        EXPECT_EQ(static_cast<int>(s->user.log[0].note), 67);
        EXPECT_TRUE(s->user.log[0].choke);
    }
    cp->deactivate(cp);
    cp->destroy(cp);
}

namespace {

// Logs the plain value setParameter receives and, per processBlock call, the
// block length and the value in force.
struct ModLogPlug
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name = "ModLog", .vendor = "T", .url = "", .email = "",
        .productId = "com.dspark.test.modlog", .version = "1.0.0",
    };
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("gain", "Gain", 0.0f, 10.0f, 5.0f, ""),
        dspark::plugin::stepped("taps", "Taps", 0.0f, 4.0f, 2.0f, 4),
        dspark::plugin::param("other", "Other", 0.0f, 1.0f, 0.0f, ""));

    float heard[3] = { 5.0f, 2.0f, 0.0f };
    std::vector<std::pair<int, float>> blocks;
    int resets = 0;

    ModLogPlug() { blocks.reserve(64); }
    void prepare(const dspark::AudioSpec&) {}
    void setParameter(int index, float plain) noexcept { heard[index] = plain; }
    void processBlock(dspark::AudioBufferView<float> io) noexcept
    {
        if (blocks.size() < blocks.capacity()) blocks.push_back({ io.getNumSamples(), heard[0] });
    }
    void reset() noexcept { ++resets; }
};

} // namespace

DSPARK_TEST(PluginClap_start_processing_resets_the_history)
{
    // start_processing clears the processing history like reset(): a device
    // the host stops and starts again (a sleeping one waking up) used to
    // resume its old tails, the CLAP call being a no-op.
    using Wrap = dspark::plugin::clap_backend::Plugin<ModLogPlug>;
    const clap_plugin_t* cp = Wrap::create(nullptr);
    auto* s = static_cast<Wrap*>(cp->plugin_data);
    cp->init(cp);
    cp->activate(cp, 48000.0, 32, 256);
    EXPECT_TRUE(cp->start_processing(cp));
    EXPECT_EQ(s->user.resets, 1);
    cp->stop_processing(cp);
    EXPECT_TRUE(cp->start_processing(cp));
    EXPECT_EQ(s->user.resets, 2);
    cp->stop_processing(cp);
    cp->deactivate(cp);
    cp->destroy(cp);
}

DSPARK_TEST(PluginClap_modulation_adds_to_the_heard_value)
{
    // Every parameter is CLAP_PARAM_IS_MODULATABLE and a global
    // CLAP_EVENT_PARAM_MOD amount (plain units) adds to the value the plugin
    // hears, clamped and snapped, sample-accurately, while the parameter's
    // own value stays unmodulated. The old wrapper declared no parameter
    // modulatable and ignored the event, so host modulators had no effect.
    using Wrap = dspark::plugin::clap_backend::Plugin<ModLogPlug>;
    clap_param_info_t info {};
    EXPECT_TRUE(Wrap::kParams.get_info(nullptr, 0, &info));
    EXPECT_NE(info.flags & CLAP_PARAM_IS_MODULATABLE, 0u);
    EXPECT_EQ(info.flags & CLAP_PARAM_IS_MODULATABLE_PER_NOTE_ID, 0u);
    EXPECT_TRUE(Wrap::kParams.get_info(nullptr, 3, &info));   // the wrapper's Bypass
    EXPECT_EQ(info.flags & CLAP_PARAM_IS_MODULATABLE, 0u);

    const clap_plugin_t* cp = Wrap::create(nullptr);
    auto* s = static_cast<Wrap*>(cp->plugin_data);
    cp->init(cp);
    cp->activate(cp, 48000.0, 32, 256);
    const clap_id gain = dspark::plugin::hash32("gain");
    const clap_id taps = dspark::plugin::hash32("taps");
    const clap_id other = dspark::plugin::hash32("other");
    auto value = [&](clap_id id) { double v = -1.0; Wrap::kParams.get_value(cp, id, &v); return v; };
    ClapEvents events;

    // Placed like automation: the block splits at the quantum before t = 100.
    s->user.blocks.clear();
    events.mod(gain, 2.0, 100);
    processClap(s, events, 256);
    EXPECT_NEAR(s->user.heard[0], 7.0f, 1e-6f);
    EXPECT_NEAR(value(gain), 5.0, 1e-9);
    EXPECT_EQ(static_cast<int>(s->user.blocks.size()), 2);
    if (s->user.blocks.size() == 2)
    {
        EXPECT_EQ(s->user.blocks[0].first, 96);
        EXPECT_NEAR(s->user.blocks[0].second, 5.0f, 1e-6f);
        EXPECT_NEAR(s->user.blocks[1].second, 7.0f, 1e-6f);
    }

    events.value(gain, 6.0, 0);                    // value + modulation
    processClap(s, events, 64);
    EXPECT_NEAR(s->user.heard[0], 8.0f, 1e-6f);
    EXPECT_NEAR(value(gain), 6.0, 1e-9);
    events.mod(gain, 10.0, 0);                     // clamped to the range
    processClap(s, events, 64);
    EXPECT_NEAR(s->user.heard[0], 10.0f, 1e-6f);
    events.mod(gain, std::numeric_limits<double>::quiet_NaN(), 0);   // ignored
    events.mod(gain, -1.0, 0, 7);                  // per-note: not offered, ignored
    processClap(s, events, 64);
    EXPECT_NEAR(s->user.heard[0], 10.0f, 1e-6f);
    events.mod(gain, 0.0, 0);                      // modulation off
    processClap(s, events, 64);
    EXPECT_NEAR(s->user.heard[0], 6.0f, 1e-6f);
    events.mod(taps, 1.4, 0);                      // 2 + 1.4 snaps to 3
    processClap(s, events, 64);
    EXPECT_NEAR(s->user.heard[1], 3.0f, 1e-6f);
    EXPECT_NEAR(value(taps), 2.0, 1e-9);

    // A full block: a value and a modulation amount of one parameter are
    // separate points, so neither displaces the other when the event list
    // merges (both used to share one slot key).
    for (uint32_t i = 0; i < 300; ++i) events.value(other, (i % 10) / 10.0, i % 256);
    events.value(gain, 4.0, 255);
    events.mod(gain, 1.0, 255);
    processClap(s, events, 256);
    EXPECT_NEAR(value(gain), 4.0, 1e-9);
    EXPECT_NEAR(s->user.heard[0], 5.0f, 1e-6f);

    cp->deactivate(cp);
    cp->destroy(cp);
}
