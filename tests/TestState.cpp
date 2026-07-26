// DSPark Tests - State serialization (getState/setState round-trips)
//
// Every effect: configure non-default parameters, serialize, restore into a
// fresh instance, serialize again - the two blobs must be identical. This
// instantiates every getState/setState template and proves the round-trip
// is lossless and stable. Plus JSON helpers and nested (multiband) states.

#include "dspark_test.h"
#include "TestSignals.h"

#include "../DSPark.h"

#include <clocale>
#include <limits>
#include <string>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

bool blobsEqual(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// configure -> getState -> fresh setState -> getState must match.
template <typename Effect, typename Configure, typename... PrepArgs>
bool roundTrip(Configure&& configure, PrepArgs... prepArgs)
{
    const AudioSpec sp = spec(48000.0, 512, 2);
    Effect a;
    a.prepare(sp, prepArgs...);
    configure(a);
    const auto blobA = a.getState();

    Effect b;
    b.prepare(sp, prepArgs...);
    if (!b.setState(blobA.data(), blobA.size())) return false;
    const auto blobB = b.getState();
    return blobsEqual(blobA, blobB) && !blobA.empty();
}

} // namespace

DSPARK_TEST(State_roundtrip_dynamics)
{
    EXPECT_TRUE((roundTrip<Compressor<float>>([](auto& c) {
        c.setThreshold(-28.5f); c.setRatio(3.3f); c.setAttack(2.5f);
        c.setRelease(180.0f); c.setKnee(6.0f); c.setMix(0.8f);
        c.setHoldTime(12.0f); c.setSidechainHPF(true, 120.0f);
        c.setMode(Compressor<float>::Mode::Upward);
        c.setCharacter(Compressor<float>::Character::FET);
        c.setCharacterColor(0.65f);
    })));
    EXPECT_TRUE((roundTrip<Limiter<float>>([](auto& l) {
        l.setCeiling(-1.2f); l.setRelease(60.0f); l.setTruePeak(true);
        l.setAdaptiveRelease(true); l.setLookahead(7.5f);
    })));
    EXPECT_TRUE((roundTrip<NoiseGate<float>>([](auto& g) {
        g.setThreshold(-52.0f); g.setHold(80.0f); g.setDuckMode(true);
        g.setGateMode(NoiseGate<float>::GateMode::Frequency);
        g.setSidechainHPF(true, 150.0);
    })));
    EXPECT_TRUE((roundTrip<Expander<float>>([](auto& e) {
        e.setThreshold(-46.0f); e.setRatio(2.5f); e.setRange(-60.0f);
    })));
    EXPECT_TRUE((roundTrip<DeEsser<float>>([](auto& d) {
        d.setFrequency(6200.0f); d.setThreshold(-25.0f); d.setReduction(9.0f);
        d.setBandwidth(1.5f); d.setAttack(2.0f); d.setRelease(80.0f);
        d.setDetectionMode(DeEsser<float>::DetectionMode::Derivative);
    })));
    EXPECT_TRUE((roundTrip<TransientDesigner<float>>([](auto& t) {
        t.setAttack(40.0f); t.setSustain(-30.0f);   // percent units
    })));
    EXPECT_TRUE((roundTrip<Gain<float>>([](auto& g) {
        g.setGainDb(-4.5f); g.setInverted(true);
    })));
    EXPECT_TRUE((roundTrip<AutoGain<float>>([](auto& a) {
        a.setMaxCompensation(9.0f); a.setSmoothingTime(50.0f);
    })));
}

DSPARK_TEST(State_roundtrip_tone_and_modulation)
{
    EXPECT_TRUE((roundTrip<Saturation<float>>([](auto& s) {
        s.setAlgorithm(Saturation<float>::Algorithm::Tape);
        s.setDrive(7.5f); s.setMix(0.7f); s.setCharacter(0.4f);
        s.setAntialiasing(true); s.setOutputGain(-2.0f);
    })));
    EXPECT_TRUE((roundTrip<Clipper<float>>([](auto& c) {
        c.setMode(Clipper<float>::Mode::Soft); c.setCeiling(-0.5f);
        c.setInputGain(3.0f); c.setStages(3); c.setMix(0.65f);
        c.setSlewLimit(0.4f); c.setOversampling(4);
    })));
    EXPECT_TRUE((roundTrip<Chorus<float>>([](auto& c) {
        c.setRate(1.8f); c.setDepthMs(5.0f); c.setVoices(3);
        c.setFeedback(0.2f);
        c.setModWaveform(Oscillator<float>::Waveform::Triangle);
    })));
    EXPECT_TRUE((roundTrip<Phaser<float>>([](auto& p) {
        p.setRate(0.9f); p.setStages(6); p.setFeedback(0.5f);
        p.setLfoWaveform(Oscillator<float>::Waveform::Triangle);
    })));
    EXPECT_TRUE((roundTrip<Tremolo<float>>([](auto& t) {
        t.setRate(6.5f); t.setDepth(0.8f); t.setStereo(true);
    })));
    EXPECT_TRUE((roundTrip<Vibrato<float>>([](auto& v) {
        v.setRate(4.2f); v.setDepth(0.3f); v.setModRate(1.5f); v.setModDepth(0.6f);
    })));
    EXPECT_TRUE((roundTrip<RingModulator<float>>([](auto& r) {
        r.setFrequency(220.0f); r.setMix(0.6f);
        r.setMode(RingModulator<float>::Mode::GeometricMean); r.setSoar(0.03f);
    })));
    EXPECT_TRUE((roundTrip<FrequencyShifter<float>>([](auto& f) {
        f.setShift(150.0f); f.setMix(0.5f);
    })));
    EXPECT_TRUE((roundTrip<Delay<float>>([](auto& d) {
        d.setDelayMs(370.0f); d.setFeedback(0.45f);
        d.setFeedbackLpHz(4500.0f); d.setFeedbackHpHz(150.0f);
    }, 2.0)));   // prepare(spec, maxDelaySeconds)
    EXPECT_TRUE((roundTrip<DCBlocker<float>>([](auto& d) {
        d.setOrder(2); d.setCutoff(10.0f);
    })));
    EXPECT_TRUE((roundTrip<FilterEngine<float>>([](auto& f) {
        f.setLowPass(2200.0f, 0.9f, 24);
    })));
}

DSPARK_TEST(State_roundtrip_stereo_reverb_spatial)
{
    EXPECT_TRUE((roundTrip<Panner<float>>([](auto& p) {
        p.setPan(-0.4f);
        p.setAlgorithm(Panner<float>::Algorithm::Binaural);
    })));
    EXPECT_TRUE((roundTrip<StereoWidth<float>>([](auto& s) {
        s.setWidth(1.5f); s.setBassMono(true, 130.0);
    })));
    EXPECT_TRUE((roundTrip<NoiseGenerator<float>>([](auto& n) {
        n.setType(NoiseGenerator<float>::Type::Pink); n.setGain(0.3f);
    })));
    EXPECT_TRUE((roundTrip<AlgorithmicReverb<float>>([](auto& r) {
        r.setType(AlgorithmicReverb<float>::Type::Hall);
        r.setDecay(2.8f); r.setMix(0.4f); r.setDamping(0.6f);
        r.setPreDelay(18.0f); r.setWidth(1.4f);
        r.setQuality(AlgorithmicReverb<float>::Quality::Eco);
    })));
    EXPECT_TRUE((roundTrip<Reverb<float>>([](auto& r) {
        r.setMix(0.55f); r.setPreDelay(25.0f);
        r.setDecayScale(0.7f); r.setStretch(1.3f);
    })));
    EXPECT_TRUE((roundTrip<Equalizer<float>>([](auto& e) {
        Equalizer<float>::BandConfig b;
        b.frequency = 320.0f; b.gain = 4.0f; b.q = 1.4f;
        e.setBand(0, b);
        b.frequency = 8200.0f; b.gain = -3.0f;
        b.type = Equalizer<float>::BandType::HighShelf;
        e.setBand(1, b);
        e.setMatchedBells(true);
        e.setSoftMode(true);
        e.setFilterMode(Equalizer<float>::FilterMode::LinearPhase);
    })));
    EXPECT_TRUE((roundTrip<CrossoverFilter<float>>([](auto& x) {
        x.setNumBands(3);
        x.setCrossoverFrequency(0, 250.0f);
        x.setCrossoverFrequency(1, 2800.0f);
        x.setOrder(48);
        x.setFilterMode(CrossoverFilter<float>::FilterMode::LinearPhase);
    })));
    EXPECT_TRUE((roundTrip<DynamicEQ<float>>([](auto& d) {
        DynamicEQ<float>::BandConfig b;
        b.frequency = 3200.0f; b.threshold = -26.0f; b.aboveRatio = 2.5f;
        d.setNumBands(2);
        d.setBand(0, b);
        b.frequency = 240.0f; b.belowRatio = 1.8f;
        d.setBand(1, b);
    })));
}

DSPARK_TEST(State_roundtrip_new_effects)
{
    EXPECT_TRUE((roundTrip<TapeMachine<float>>([](auto& t) {
        t.setDrive(5.0f); t.setBias(0.35f); t.setWowFlutter(0.25f);
    })));
    EXPECT_TRUE((roundTrip<TubePreamp<float>>([](auto& t) {
        t.setDrive(8.0f); t.setTreble(0.7f); t.setBass(0.4f);
        t.setStages(2);
    })));
    EXPECT_TRUE((roundTrip<TransformerModel<float>>([](auto& t) {
        t.setDrive(4.0f); t.setCoreSize(0.7f);
    })));
    EXPECT_TRUE((roundTrip<PitchShifter<float>>([](auto& p) {
        p.setSemitones(7.0f); p.setMix(0.9f);
    })));
    EXPECT_TRUE((roundTrip<GranularProcessor<float>>([](auto& g) {
        g.setGrainSize(120.0f); g.setDensity(35.0f); g.setPitch(-5.0f);
        g.setFreeze(true);
    })));
    EXPECT_TRUE((roundTrip<SpectralDenoiser<float>>([](auto& d) {
        d.setReduction(24.0f); d.setThreshold(3.0f);
    })));
}

DSPARK_TEST(State_multiband_nested_blobs)
{
    const AudioSpec sp = spec(48000.0, 512, 2);
    MultibandCompressor<float> a;
    a.prepare(sp);
    a.setNumBands(3);
    a.setOrder(48);
    a.setCrossoverMode(CrossoverFilter<float, 12>::FilterMode::LinearPhase);
    a.setCrossoverFrequency(0, 180.0f);
    a.setCrossoverFrequency(1, 3500.0f);
    a.setBandThreshold(0, -24.0f);
    a.setBandRatio(0, 2.5f);
    a.setBandThreshold(2, -30.0f);
    a.setBandAttack(2, 1.5f);

    const auto blob = a.getState();
    MultibandCompressor<float> b;
    b.prepare(sp);
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    EXPECT_TRUE(blobsEqual(blob, b.getState()));
    EXPECT_EQ(b.getOrder(), 48);
    EXPECT_TRUE((b.getCrossoverMode() == CrossoverFilter<float, 12>::FilterMode::LinearPhase));
    EXPECT_NEAR(static_cast<float>(b.getCrossoverFrequency(0)), 180.0f, 1e-3f);

    // The nested band states really carried the per-band parameters.
    const auto inner = b.getBandCompressor(0).getState();
    StateReader r(inner.data(), inner.size());
    EXPECT_TRUE(r.isValid());
    EXPECT_NEAR(r.read("threshold", 0.0f), -24.0f, 1e-6f);
    EXPECT_NEAR(r.read("ratio", 0.0f), 2.5f, 1e-6f);
}

DSPARK_TEST(State_json_roundtrip_and_tolerance)
{
    const AudioSpec sp = spec(48000.0, 512, 2);
    Compressor<float> c;
    c.prepare(sp);
    c.setThreshold(-31.5f);
    c.setRatio(5.0f);
    c.setAutoMakeup(false);

    // blob -> JSON -> blob -> setState restores the same parameters.
    const auto blob = c.getState();
    const std::string json = stateToJson(blob);
    EXPECT_TRUE(json.find("\"threshold\":-31.5") != std::string::npos);
    const auto back = stateFromJson(json);
    EXPECT_TRUE(!back.empty());

    Compressor<float> d;
    d.prepare(sp);
    EXPECT_TRUE(d.setState(back.data(), back.size()));
    const auto blob2 = d.getState();
    StateReader r(blob2.data(), blob2.size());
    EXPECT_NEAR(r.read("threshold", 0.0f), -31.5f, 1e-6f);
    EXPECT_NEAR(r.read("ratio", 0.0f), 5.0f, 1e-6f);
    EXPECT_TRUE(r.read("autoMakeup", true) == false);

    // Foreign blobs are rejected; junk keys are tolerated.
    Limiter<float> lim;
    lim.prepare(sp);
    EXPECT_TRUE(!lim.setState(blob.data(), blob.size()));   // COMP into LIMT
    StateWriter w(stateId("COMP"), 99);
    w.write("unknownKey", 1.25f);
    const auto odd = w.blob();
    Compressor<float> e;
    e.prepare(sp);
    EXPECT_TRUE(e.setState(odd.data(), odd.size()));        // future version OK
}

DSPARK_TEST(State_reader_rejects_corrupt_blobs)
{
    // A nested entry whose declared length reaches past the blob must be
    // rejected. The bounds check uses the subtraction form because the
    // addition form (`pos + payload > size`) wraps on 32-bit size_t (WASM)
    // for lengths near UINT32_MAX and would pass into a giant OOB read.
    StateWriter w(stateId("TEST"), 1);
    w.write("f", 1.0f);
    auto blob = w.blob();
    {
        StateWriter inner(stateId("NEST"), 1);
        inner.write("x", 2.0f);
        StateWriter outer(stateId("TEST"), 1);
        outer.write("child", inner.blob());
        blob = outer.blob();
    }
    // Corrupt the nested length field (last 4 bytes before the child bytes):
    // entry layout is keyLen(1) "child"(5) type(1) len(4) child....
    const size_t lenPos = 14 + 1 + 5 + 1;
    blob[lenPos]     = 0xF0;
    blob[lenPos + 1] = 0xFF;
    blob[lenPos + 2] = 0xFF;
    blob[lenPos + 3] = 0xFF;   // declared nested length ~4 GB
    StateReader huge(blob.data(), blob.size());
    EXPECT_FALSE(huge.isValid());

    // Systematic truncation: every prefix of a valid blob must parse without
    // crashing, and only the full blob reports valid.
    StateWriter w2(stateId("TRNC"), 3);
    w2.write("a", 1.5f);
    w2.write("b", static_cast<int32_t>(-7));
    w2.write("c", true);
    const auto full = w2.blob();
    for (size_t n = 0; n < full.size(); ++n)
    {
        StateReader partial(full.data(), n);
        EXPECT_FALSE(partial.isValid());
    }
    StateReader whole(full.data(), full.size());
    EXPECT_TRUE(whole.isValid());

    // Wrong magic is rejected outright.
    auto bad = full;
    bad[0] = 'X';
    StateReader magic(bad.data(), bad.size());
    EXPECT_FALSE(magic.isValid());
}

DSPARK_TEST(State_json_is_locale_independent)
{
    // The classic plugin bug: a host sets a decimal-comma LC_NUMERIC and
    // printf-based JSON emits "0,5" (invalid) while strtod stops at '.'.
    // Both directions must survive any process locale.
    const char* prev = std::setlocale(LC_NUMERIC, nullptr);
    std::string saved = (prev != nullptr) ? prev : "C";
    const char* de = std::setlocale(LC_NUMERIC, "de-DE");
    if (de == nullptr)
        de = std::setlocale(LC_NUMERIC, "de_DE.UTF-8");

    StateWriter w(stateId("LOCL"), 1);
    w.write("half", 0.5f);
    w.write("freq", 1234.567f);
    const auto blob = w.blob();

    const std::string json = stateToJson(blob);
    const auto back = stateFromJson(json);

    std::setlocale(LC_NUMERIC, saved.c_str());   // restore before asserting

    if (de == nullptr)
        return;   // locale not installed on this machine: nothing to prove

    EXPECT_TRUE(json.find("0.5") != std::string::npos);        // dot, not comma
    EXPECT_TRUE(json.find("1234.567") != std::string::npos);
    EXPECT_TRUE(!back.empty());
    StateReader r(back.data(), back.size());
    EXPECT_NEAR(r.read("half", 0.0f), 0.5f, 1e-9f);
    EXPECT_NEAR(r.read("freq", 0.0f), 1234.567f, 1e-3f);
}

// ===========================================================================
// M-003 AG-8 audit regression pins (additive; CHANGE-REQUEST recorded in
// builder-report.002.json). Guard the StateBlob JSON-helper fixes.
// ===========================================================================

// D-M003-4a: keys needing JSON escaping must produce valid (escaped) JSON and
// round-trip losslessly (writer escapes, reader unescapes symmetrically).
DSPARK_TEST(State_json_escapes_hostile_keys_and_roundtrips)
{
    StateWriter w(0x4B455931u, 1);
    w.write("a\"b\\c", 0.5f);          // embedded quote + backslash
    w.write("ctrl\x01\x1f", 0.25f);    // control characters
    const auto blob = w.blob();

    const std::string json = stateToJson(blob);
    EXPECT_TRUE(json.find("\\\"") != std::string::npos);      // escaped quote
    EXPECT_TRUE(json.find("\\\\") != std::string::npos);      // escaped backslash
    EXPECT_TRUE(json.find("\\u0001") != std::string::npos);   // escaped control
    // no raw control byte leaked into the JSON text
    EXPECT_TRUE(json.find('\x01') == std::string::npos);

    const auto back = stateFromJson(json);
    EXPECT_TRUE(!back.empty());
    StateReader r(back.data(), back.size());
    EXPECT_TRUE(r.isValid());
    EXPECT_NEAR(r.read("a\"b\\c", 0.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(r.read("ctrl\x01\x1f", 0.0f), 0.25f, 1e-6f);
}

// D-M003-4b: a non-finite float payload must render as legal JSON (never bare
// nan/inf) and still parse back (rendered as 0).
// M-004 (AG-3) regression: getState() must COMPILE and round-trip for the
// double specialisation. Limiter/NoiseGate/Gain used to pass an unqualified
// T=double to StateWriter::write(), which is ambiguous (float/int32/bool) and
// broke the public getState() for the supported double type; the others cast to
// float. This pin instantiates getState()/setState() for T=double across every
// AG-3 dynamics/gain header so the ambiguity cannot silently return.
DSPARK_TEST(State_roundtrip_dynamics_double)
{
    EXPECT_TRUE((roundTrip<Limiter<double>>([](auto& l) {
        l.setCeiling(-1.2); l.setRelease(60.0); l.setTruePeak(true);
        l.setAdaptiveRelease(true); l.setLookahead(7.5);
    })));
    EXPECT_TRUE((roundTrip<NoiseGate<double>>([](auto& g) {
        g.setThreshold(-52.0); g.setHold(80.0); g.setDuckMode(true);
        g.setGateMode(NoiseGate<double>::GateMode::Frequency);
        g.setSidechainHPF(true, 150.0);
    })));
    EXPECT_TRUE((roundTrip<Gain<double>>([](auto& g) {
        g.setGainDb(-4.5); g.setInverted(true);
    })));
    EXPECT_TRUE((roundTrip<Compressor<double>>([](auto& c) {
        c.setThreshold(-28.5); c.setRatio(3.3); c.setKnee(6.0);
    })));
    EXPECT_TRUE((roundTrip<Expander<double>>([](auto& e) {
        e.setThreshold(-46.0); e.setRatio(2.5); e.setRange(-60.0);
    })));
    EXPECT_TRUE((roundTrip<DeEsser<double>>([](auto& d) {
        d.setFrequency(6200.0); d.setThreshold(-25.0); d.setReduction(9.0);
    })));
    EXPECT_TRUE((roundTrip<TransientDesigner<double>>([](auto& t) {
        t.setAttack(40.0); t.setSustain(-30.0);
    })));
    EXPECT_TRUE((roundTrip<AutoGain<double>>([](auto& a) {
        a.setMaxCompensation(9.0); a.setSmoothingTime(50.0);
    })));
    EXPECT_TRUE((roundTrip<DynamicEQ<double>>([](auto& e) {
        e.setNumBands(3);
    })));
    EXPECT_TRUE((roundTrip<CrossoverFilter<double>>([](auto& x) {
        x.setNumBands(3); x.setOrder(24);
    })));
    EXPECT_TRUE((roundTrip<MultibandCompressor<double>>([](auto& m) {
        m.setNumBands(3);
    })));
}

DSPARK_TEST(State_json_non_finite_float_is_legal)
{
    StateWriter w(0x4E414E31u, 1);
    w.write("bad", std::numeric_limits<float>::quiet_NaN());
    w.write("big", std::numeric_limits<float>::infinity());
    w.write("neg", -std::numeric_limits<float>::infinity());
    const auto blob = w.blob();

    const std::string json = stateToJson(blob);
    EXPECT_TRUE(json.find("nan") == std::string::npos);
    EXPECT_TRUE(json.find("inf") == std::string::npos);

    const auto back = stateFromJson(json);
    EXPECT_TRUE(!back.empty());
    StateReader r(back.data(), back.size());
    EXPECT_TRUE(r.isValid());
    EXPECT_NEAR(r.read("bad", -1.0f), 0.0f, 1e-9f); // non-finite -> 0
}
