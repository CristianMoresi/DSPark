// DSPark Tests - Music
// HarmonyConstants: scales, chords, note parsing, transposition

#include "dspark_test.h"
#include "TestSignals.h"
#include "../Music/HarmonyConstants.h"
#include "../Music/ChordDetector.h"

#include <cmath>
#include <limits>
#include <string_view>

using namespace dspark;
using namespace dspark::harmony;
using namespace dspark::test;

// ============================================================================
// Scales
// ============================================================================

DSPARK_TEST(Harmony_Ionian_correct_degrees)
{
    // Ionian (Major): C D E F G A B = semitones 0,2,4,5,7,9,11
    const auto& ionian = allScales[0];
    NoteSet mask = ionian.mask;

    EXPECT_TRUE((mask & (1 << 0))  != 0);  // C
    EXPECT_TRUE((mask & (1 << 2))  != 0);  // D
    EXPECT_TRUE((mask & (1 << 4))  != 0);  // E
    EXPECT_TRUE((mask & (1 << 5))  != 0);  // F
    EXPECT_TRUE((mask & (1 << 7))  != 0);  // G
    EXPECT_TRUE((mask & (1 << 9))  != 0);  // A
    EXPECT_TRUE((mask & (1 << 11)) != 0);  // B

    // Should NOT contain C#, D#, F#, G#, A#
    EXPECT_TRUE((mask & (1 << 1))  == 0);  // C#
    EXPECT_TRUE((mask & (1 << 3))  == 0);  // D#
    EXPECT_TRUE((mask & (1 << 6))  == 0);  // F#
    EXPECT_TRUE((mask & (1 << 8))  == 0);  // G#
    EXPECT_TRUE((mask & (1 << 10)) == 0);  // A#
}

DSPARK_TEST(Harmony_Aeolian_correct_degrees)
{
    // Aeolian (Natural Minor): 0,2,3,5,7,8,10
    const auto& aeolian = allScales[5];
    NoteSet mask = aeolian.mask;

    EXPECT_TRUE((mask & (1 << 0))  != 0);  // C
    EXPECT_TRUE((mask & (1 << 2))  != 0);  // D
    EXPECT_TRUE((mask & (1 << 3))  != 0);  // Eb
    EXPECT_TRUE((mask & (1 << 5))  != 0);  // F
    EXPECT_TRUE((mask & (1 << 7))  != 0);  // G
    EXPECT_TRUE((mask & (1 << 8))  != 0);  // Ab
    EXPECT_TRUE((mask & (1 << 10)) != 0);  // Bb
}

DSPARK_TEST(Harmony_scaleAtRoot_transposes)
{
    // C Ionian transposed to D (root=2) should have bits at 2,4,6,7,9,11,1
    const auto& ionian = allScales[0];
    NoteSet dMajor = scaleAtRoot(ionian.mask, 2);

    EXPECT_TRUE((dMajor & (1 << 2))  != 0);  // D
    EXPECT_TRUE((dMajor & (1 << 4))  != 0);  // E
    EXPECT_TRUE((dMajor & (1 << 6))  != 0);  // F#
    EXPECT_TRUE((dMajor & (1 << 7))  != 0);  // G
    EXPECT_TRUE((dMajor & (1 << 9))  != 0);  // A
    EXPECT_TRUE((dMajor & (1 << 11)) != 0);  // B
    EXPECT_TRUE((dMajor & (1 << 1))  != 0);  // C#
}

DSPARK_TEST(Harmony_ChromaticScale_all_12)
{
    const auto& chromatic = allScales[40]; // "Chromatic"
    EXPECT_EQ(chromatic.mask, static_cast<NoteSet>(0x0FFF));
}

// ============================================================================
// Note Parsing
// ============================================================================

DSPARK_TEST(Harmony_parseNote_basic)
{
    auto c  = parseNote("C");
    auto cs = parseNote("C#");
    auto eb = parseNote("Eb");
    auto b  = parseNote("B");

    EXPECT_TRUE(c.has_value());
    EXPECT_EQ(c.value(), 0);

    EXPECT_TRUE(cs.has_value());
    EXPECT_EQ(cs.value(), 1);

    EXPECT_TRUE(eb.has_value());
    EXPECT_EQ(eb.value(), 3);

    EXPECT_TRUE(b.has_value());
    EXPECT_EQ(b.value(), 11);
}

DSPARK_TEST(Harmony_parseNote_invalid)
{
    auto result = parseNote("X");
    EXPECT_FALSE(result.has_value());
}

DSPARK_TEST(Harmony_noteName_correct)
{
    // MIDI 60 = C, root 0 = C context
    auto name = noteName(60, 0);
    EXPECT_EQ(name, std::string_view("C"));

    // MIDI 69 = A
    auto nameA = noteName(69, 0);
    EXPECT_EQ(nameA, std::string_view("A"));
}

// ============================================================================
// Chords
// ============================================================================

DSPARK_TEST(Harmony_chordAtRootMidi_major_triad)
{
    // C Major triad at MIDI 60: should be C(60), E(64), G(67)
    auto notes = chordAtRootMidi(allChords[0], 60, 0);
    EXPECT_EQ(notes[0], 60);  // C
    EXPECT_EQ(notes[1], 64);  // E
    EXPECT_EQ(notes[2], 67);  // G
}

DSPARK_TEST(Harmony_chordAtRootMidi_first_inversion)
{
    // C Major 1st inversion: E(64), G(67), C(72)
    auto notes = chordAtRootMidi(allChords[0], 60, 1);
    EXPECT_EQ(notes[0], 64);  // E
    EXPECT_EQ(notes[1], 67);  // G
    EXPECT_EQ(notes[2], 72);  // C (octave up)
}

DSPARK_TEST(Harmony_chord_minor7)
{
    // C Minor 7th: C(60), Eb(63), G(67), Bb(70)
    auto notes = chordAtRootMidi(allChords[6], 60, 0);
    EXPECT_EQ(notes[0], 60);  // C
    EXPECT_EQ(notes[1], 63);  // Eb
    EXPECT_EQ(notes[2], 67);  // G
    EXPECT_EQ(notes[3], 70);  // Bb
}

// ============================================================================
// Transposition
// ============================================================================

DSPARK_TEST(Harmony_transposeByOctaves)
{
    EXPECT_EQ(transposeByOctaves(60, 1), 72);
    EXPECT_EQ(transposeByOctaves(60, -1), 48);
    EXPECT_EQ(transposeByOctaves(60, 0), 60);
}

DSPARK_TEST(Harmony_61_scales_defined)
{
    // All 61 scales should have non-zero masks
    for (int i = 0; i < 61; ++i)
        EXPECT_TRUE(allScales[i].mask != 0);
}

DSPARK_TEST(Harmony_15_chords_defined)
{
    // All 15 chords should have at least a root note (interval[0] >= 0)
    for (int i = 0; i < 15; ++i)
        EXPECT_TRUE(allChords[i].intervals[0] >= 0);
}


// ============================================================================
// ChordDetector
// ============================================================================

namespace {

dspark::ChordDetector<float>::Result detectChord(std::initializer_list<double> freqs)
{
    dspark::ChordDetector<float> det;
    det.prepare(spec(48000.0, 512, 2));
    auto buf = makeStereoBuffer(512);
    for (int b = 0; b < 60; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            const int n = b * 512 + i;
            double v = 0.0;
            for (const double f : freqs)
                v += std::sin(2.0 * 3.14159265358979 * f * n / 48000.0);
            buf.ch(0)[i] = static_cast<float>(0.25 * v);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        det.processBlock(AudioBufferView<const float>(buf.view()));
    }
    return det.getChord();
}

} // namespace

DSPARK_TEST(ChordDetector_recognizes_basic_chords)
{
    using CT = ChordDetector<float>::ChordType;

    const auto cMajor = detectChord({ 261.63, 329.63, 392.00 });
    EXPECT_EQ(cMajor.rootPitchClass, 0);
    EXPECT_TRUE(cMajor.type == CT::Major);
    EXPECT_GT(cMajor.confidence, 0.5f);

    const auto aMinor = detectChord({ 220.00, 261.63, 329.63 });
    EXPECT_EQ(aMinor.rootPitchClass, 9);
    EXPECT_TRUE(aMinor.type == CT::Minor);

    const auto g7 = detectChord({ 196.00, 246.94, 293.66, 349.23 });
    EXPECT_EQ(g7.rootPitchClass, 7);
    EXPECT_TRUE(g7.type == CT::Dominant7);
}

DSPARK_TEST(ChordDetector_bass_disambiguates_sus_chords)
{
    // Dsus4 and Gsus2 share the same pitch-class set; the bass decides.
    using CT = ChordDetector<float>::ChordType;
    const auto dsus4 = detectChord({ 146.83, 196.00, 220.00 });
    EXPECT_EQ(dsus4.rootPitchClass, 2);
    EXPECT_TRUE(dsus4.type == CT::Sus4);
    EXPECT_GT(dsus4.confidence, 0.5f);
}

DSPARK_TEST(ChordDetector_names_chords)
{
    using CT = ChordDetector<float>::ChordType;
    ChordDetector<float>::Result r;
    r.rootPitchClass = 6;
    r.type = CT::Minor7;
    r.confidence = 0.8f;
    char name[8];
    ChordDetector<float>::getChordName(r, name, 8);
    EXPECT_TRUE(std::string_view(name) == "F#m7");
}

// ============================================================================
// Modal families are exact rotations of their parent scale
// ============================================================================

namespace {

// Rotate a 12-bit pitch mask so the scale degree `deg` semitones above the
// old root becomes the new root: mode k of a parent scale is the parent mask
// rotated by its k-th active degree.
constexpr NoteSet rotateToDegree(NoteSet parent, int deg)
{
    unsigned b = parent & 0x0FFFu;
    return NoteSet(((b >> deg) | (b << (12 - deg))) & 0x0FFFu);
}

// Collect the active degrees (semitones) of a mask in ascending order.
constexpr std::array<int, 12> maskDegrees(NoteSet mask)
{
    std::array<int, 12> out{};
    int n = 0;
    for (int i = 0; i < 12; ++i)
        if (mask & (1u << i)) out[n++] = i;
    for (int i = n; i < 12; ++i) out[i] = -1;
    return out;
}

} // namespace

DSPARK_TEST(Harmony_modal_families_are_rotations_of_their_parent)
{
    // For the four classical 7-note families the database stores the parent
    // at the family's first slot and its modes in degree order. Every mode
    // mask must equal the parent rotated to the corresponding degree - this
    // is the definition of a mode, so it pins all 28 masks at once.
    struct Family { int parentIdx; const char* name; };
    constexpr Family families[] = {
        {0,  "major"},          // 0..6   Ionian .. Locrian
        {7,  "melodic minor"},  // 7..13
        {14, "harmonic minor"}, // 14..20
        {21, "harmonic major"}, // 21..27
    };

    for (const auto& fam : families)
    {
        const NoteSet parent = allScales[size_t(fam.parentIdx)].mask;
        const auto degrees = maskDegrees(parent);
        for (int mode = 0; mode < 7; ++mode)
        {
            const NoteSet expected = rotateToDegree(parent, degrees[size_t(mode)]);
            const NoteSet actual   = allScales[size_t(fam.parentIdx + mode)].mask;
            EXPECT_EQ(int(actual), int(expected));
        }
    }

    // The old database failed this for 5 of the 7 harmonic-major modes
    // (two of them even duplicated PhrygianDominant / LydianSharp2 exactly).
    // Spot-check that the corrected modes are all DISTINCT masks.
    for (int i = 21; i <= 27; ++i)
        for (int j = i + 1; j <= 27; ++j)
            EXPECT_TRUE(allScales[size_t(i)].mask != allScales[size_t(j)].mask);
}

DSPARK_TEST(Harmony_known_scale_masks_pinned)
{
    // Independent literature values for scales outside the modal families.
    auto maskOf = [](std::string_view name) -> NoteSet {
        for (const auto& s : allScales)
            if (s.name == name) return s.mask;
        return 0;
    };

    // Yo (Japanese anhemitonic pentatonic): D E G A B from D = 0,2,5,7,9.
    // The old table stored {0,4,5,7,11}, which contains two semitone steps
    // and is not anhemitonic at all.
    EXPECT_EQ(int(maskOf("Yo")), int(makeMask({0,2,5,7,9})));

    // Hindu = Mixolydian b6 (Slonimsky). The old table duplicated plain
    // Mixolydian.
    EXPECT_EQ(int(maskOf("Hindu")), int(makeMask({0,2,4,5,7,8,10})));
    EXPECT_TRUE(maskOf("Hindu") != maskOf("Mixolydian"));

    // Javanese = Phrygian natural-6 (seven notes).
    EXPECT_EQ(int(maskOf("Javanese")), int(makeMask({0,1,3,5,7,9,10})));
}

DSPARK_TEST(Harmony_note_spelling_follows_key_signature)
{
    // Sharp keys spell F# (G major has one sharp); flat keys spell Gb.
    EXPECT_EQ(noteName(66, 7), std::string_view("F#")); // G major context
    EXPECT_EQ(noteName(66, 1), std::string_view("Gb")); // Db major context
    // A, E and B major are sharp keys (the old table spelled them flat).
    EXPECT_EQ(noteName(61, 9),  std::string_view("C#")); // A major
    EXPECT_EQ(noteName(68, 4),  std::string_view("G#")); // E major
    EXPECT_EQ(noteName(70, 11), std::string_view("A#")); // B major
    // Flat keys stay flat.
    EXPECT_EQ(noteName(63, 10), std::string_view("Eb")); // Bb major
    EXPECT_EQ(noteName(68, 8),  std::string_view("Ab")); // Ab major
}

DSPARK_TEST(Harmony_parseNote_empty_and_tags)
{
    // parseNote("") must fail: the old lookup table declared 28 entries but
    // listed 21, so the value-initialized {"",0} tail made "" parse as C.
    EXPECT_FALSE(parseNote("").has_value());
    EXPECT_FALSE(parseNote("H").has_value());
    EXPECT_TRUE(parseNote("bb").has_value());  // case-insensitive Bb

    // ChordTag algebra: & intersects, hasTags tests containment.
    constexpr ChordTag both = ChordTag::MajorTriad | ChordTag::Major7;
    EXPECT_TRUE((both & ChordTag::Major7) == ChordTag::Major7);
    EXPECT_TRUE(hasTags(both, ChordTag::MajorTriad));
    EXPECT_TRUE(!hasTags(both, ChordTag::Minor7));
}

// ============================================================================
// ChordDetector - invalid inputs and lifecycle
// ============================================================================

DSPARK_TEST(ChordDetector_invalid_inputs_are_ignored)
{
    using CT = ChordDetector<float>::ChordType;

    ChordDetector<float> det;
    det.prepare(spec(48000.0, 512, 2));

    auto feed = [&](std::initializer_list<double> freqs, int blocks)
    {
        auto buf = makeStereoBuffer(512);
        for (int b = 0; b < blocks; ++b)
        {
            for (int i = 0; i < 512; ++i)
            {
                const int n = b * 512 + i;
                double v = 0.0;
                for (const double f : freqs)
                    v += std::sin(2.0 * 3.14159265358979 * f * n / 48000.0);
                buf.ch(0)[i] = static_cast<float>(0.25 * v);
                buf.ch(1)[i] = buf.ch(0)[i];
            }
            det.processBlock(AudioBufferView<const float>(buf.view()));
        }
    };

    // Establish C major.
    feed({ 261.63, 329.63, 392.00 }, 60);
    EXPECT_EQ(det.getChord().rootPitchClass, 0);

    // A NaN threshold must be ignored: with the old clamp(NaN) pass-through
    // every confidence comparison went false and the detector froze on the
    // held chord forever. After the fix it keeps tracking chord changes.
    det.setConfidenceThreshold(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(det.getConfidenceThreshold(), 0.55f, 1e-6f); // honest getter

    // An invalid re-prepare on a hot detector is a conservative no-op.
    det.prepare(spec(std::numeric_limits<double>::quiet_NaN(), 512, 2));
    det.prepare(spec(-48000.0, 512, 2));
    // A positive but unusably low rate is invalid for this analyser too:
    // MIDI 83 cannot sit below Nyquist. Reject it before Goertzel's debug
    // assertion and preserve the working configuration.
    det.prepare(spec(1.0, 512, 2));

    // Change to F major: the detector must follow.
    feed({ 349.23, 440.00, 523.25 }, 60);
    EXPECT_EQ(det.getChord().rootPitchClass, 5);
    EXPECT_TRUE(det.getChord().type == CT::Major);

    // reset() drops the held chord.
    det.reset();
    EXPECT_EQ(det.getChord().rootPitchClass, -1);
    EXPECT_TRUE(det.getChord().type == CT::None);

    // Unprepared instance is inert (no crash, empty result).
    ChordDetector<float> cold;
    auto buf = makeStereoBuffer(64);
    cold.processBlock(AudioBufferView<const float>(buf.view()));
    EXPECT_EQ(cold.getChord().rootPitchClass, -1);
}

// ============================================================================
// Regression pin, landed in the same commit as the fix it guards:
// locks the RELIABLE-register contract the corrected ChordDetector doc states
// (mid/upper-register root-position triads detect exactly) and the signal-path
// NaN/Inf poison-then-recover behaviour measured on the shipped header.
// ============================================================================

DSPARK_TEST(ChordDetector_reliable_register_and_signal_poison_recovery)
{
    using CT = ChordDetector<float>::ChordType;

    auto midiHz = [](int m){ return 440.0 * std::exp2((m - 69) / 12.0); };

    // Build a fresh detector (default 4096 window) driven by an equal-tempered
    // sine stack given as MIDI notes, optionally lacing NaN/+Inf into a middle
    // stretch of blocks; returns the held chord after `blocks` blocks.
    auto run = [&](std::initializer_list<int> midis, int blocks, bool poison)
    {
        ChordDetector<float> det;
        det.prepare(spec(48000.0, 512, 2));
        auto b2 = makeStereoBuffer(512);
        for (int b = 0; b < blocks; ++b)
        {
            const bool bad = poison && b >= 20 && b < 28;
            for (int i = 0; i < 512; ++i)
            {
                const int n = b * 512 + i;
                double v = 0.0;
                for (int m : midis)
                    v += std::sin(dspark::twoPi<double> * midiHz(m) * n / 48000.0);
                float s = static_cast<float>(0.25 * v);
                if (bad && (i % 37 == 0)) s = std::numeric_limits<float>::quiet_NaN();
                if (bad && (i % 53 == 0)) s = std::numeric_limits<float>::infinity();
                b2.ch(0)[i] = s;
                b2.ch(1)[i] = s;
            }
            det.processBlock(AudioBufferView<const float>(b2.view()));
        }
        return det.getChord();
    };

    // Reliable register (>= C4): root-position triads/7ths are exact.
    const auto cMaj = run({60, 64, 67}, 60, false);            // C4 major
    EXPECT_EQ(cMaj.rootPitchClass, 0);
    EXPECT_TRUE(cMaj.type == CT::Major);
    EXPECT_GT(cMaj.confidence, 0.5f);

    const auto dMin = run({62, 65, 69}, 60, false);            // D4 minor
    EXPECT_EQ(dMin.rootPitchClass, 2);
    EXPECT_TRUE(dMin.type == CT::Minor);

    const auto g7 = run({67, 71, 74, 77}, 60, false);          // G4 dominant7
    EXPECT_EQ(g7.rootPitchClass, 7);
    EXPECT_TRUE(g7.type == CT::Dominant7);

    // Signal-path poison-then-recover: a run laced with NaN/+Inf samples in the
    // middle must (a) keep confidence finite and (b) still resolve to the same
    // chord once the poisoned samples flush out of the analysis window.
    const auto poisoned = run({60, 64, 67}, 60, true);
    EXPECT_TRUE(std::isfinite(poisoned.confidence));
    EXPECT_EQ(poisoned.rootPitchClass, 0);
    EXPECT_TRUE(poisoned.type == CT::Major);
}

// ============================================================================
// Further ChordDetector and Harmony pins:
// the rate-aware automatic window, the documented sharp edges of explicit
// windows at professional rates, and the diatonicChord heptatonic contract.
// Each pins a documented behaviour, not an implementation detail.
// ============================================================================

namespace {

// Drive a fresh detector with a root-position sine stack at an arbitrary
// sample rate and window request; returns the held chord. windowSize <= 0
// selects the automatic (rate-derived) window; windowOut reports the choice.
dspark::ChordDetector<float>::Result detectAt(std::initializer_list<int> midis,
                                              double fs, int windowSize,
                                              int blocks, int* windowOut = nullptr)
{
    ChordDetector<float> det;
    det.prepare(spec(fs, 512, 2), windowSize);
    if (windowOut != nullptr) *windowOut = det.getWindowSize();
    auto buf = makeStereoBuffer(512);
    auto midiHz = [](int m) { return 440.0 * std::exp2((m - 69) / 12.0); };
    for (int b = 0; b < blocks; ++b)
    {
        for (int i = 0; i < 512; ++i)
        {
            const long long n = static_cast<long long>(b) * 512 + i;
            double v = 0.0;
            for (int m : midis)
                v += std::sin(dspark::twoPi<double> * midiHz(m)
                              * static_cast<double>(n) / fs);
            buf.ch(0)[i] = static_cast<float>(0.25 * v);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        det.processBlock(AudioBufferView<const float>(buf.view()));
    }
    return det.getChord();
}

} // namespace

DSPARK_TEST(ChordDetector_default_window_tracks_sample_rate)
{
    using CT = ChordDetector<float>::ChordType;

    // The automatic default must keep the analysis span ~85 ms so the
    // reliable register does not collapse at professional rates: with the
    // old fixed 4096 default, C4 major at 96 kHz read C Maj7 at conf 0.574
    // (confidently WRONG) and at 192 kHz latched a non-sounding B-Sus4.
    int w44 = 0, w48 = 0, w96 = 0, w192 = 0;

    const auto c44 = detectAt({60, 64, 67}, 44100.0, 0, 120, &w44);
    EXPECT_EQ(w44, 4096);
    EXPECT_EQ(c44.rootPitchClass, 0);
    EXPECT_TRUE(c44.type == CT::Major);
    EXPECT_GT(c44.confidence, 0.5f);

    const auto c48 = detectAt({60, 64, 67}, 48000.0, 0, 120, &w48);
    EXPECT_EQ(w48, 4096);                    // 48 kHz behaviour unchanged
    EXPECT_EQ(c48.rootPitchClass, 0);
    EXPECT_TRUE(c48.type == CT::Major);
    EXPECT_GT(c48.confidence, 0.5f);

    // The doubling boundaries of the auto map: 88.2 and 176.4 kHz are the
    // rates where the span requirement crosses into the next power of two.
    // Documented and measured, previously unpinned.
    int w88 = 0, w176 = 0;
    (void)detectAt({60, 64, 67}, 88200.0, 0, 8, &w88);
    EXPECT_EQ(w88, 8192);
    (void)detectAt({60, 64, 67}, 176400.0, 0, 8, &w176);
    EXPECT_EQ(w176, 16384);

    const auto c96 = detectAt({60, 64, 67}, 96000.0, 0, 240, &w96);
    EXPECT_EQ(w96, 8192);
    EXPECT_EQ(c96.rootPitchClass, 0);
    EXPECT_TRUE(c96.type == CT::Major);
    EXPECT_GT(c96.confidence, 0.5f);

    const auto c192 = detectAt({60, 64, 67}, 192000.0, 0, 480, &w192);
    EXPECT_EQ(w192, 16384);
    EXPECT_EQ(c192.rootPitchClass, 0);
    EXPECT_TRUE(c192.type == CT::Major);
    EXPECT_GT(c192.confidence, 0.5f);

    // A seventh chord at 96 kHz through the automatic window: G4 dominant 7.
    const auto g7 = detectAt({67, 71, 74, 77}, 96000.0, 0, 240);
    EXPECT_EQ(g7.rootPitchClass, 7);
    EXPECT_TRUE(g7.type == CT::Dominant7);

    // Explicit requests are still honoured verbatim (clamped to 1024..16384).
    int wExp = 0;
    (void)detectAt({60, 64, 67}, 96000.0, 4096, 8, &wExp);
    EXPECT_EQ(wExp, 4096);
}

DSPARK_TEST(ChordDetector_explicit_window_bounds_are_documented)
{
    using CT = ChordDetector<float>::ChordType;

    // Green anchor at a non-48k rate: 96 kHz with an explicit 16384 window
    // resolves C4 major exactly (measured Major confidence ~0.77, confirmed
    // by an independent register sweep).
    int w = 0;
    const auto rescued = detectAt({60, 64, 67}, 96000.0, 16384, 240, &w);
    EXPECT_EQ(w, 16384);
    EXPECT_EQ(rescued.rootPitchClass, 0);
    EXPECT_TRUE(rescued.type == CT::Major);
    EXPECT_GT(rescued.confidence, 0.5f);

    // Documented sharp edge: 96 kHz with a forced 4096 window
    // must NOT be trusted at C4 -- the ~94 Hz main lobe smears adjacent
    // semitones (measured C Maj7 conf ~0.57). If this ever resolves as a
    // correct C major, the register documentation is stale: update it.
    const auto smeared = detectAt({60, 64, 67}, 96000.0, 4096, 240);
    EXPECT_FALSE(smeared.rootPitchClass == 0 && smeared.type == CT::Major);

    // Documented sharp edge: windowSize 1024 at 48 kHz has an
    // EMPTY reliable register (measured 0/29 roots C3..E5 correct).
    const auto tiny = detectAt({60, 64, 67}, 48000.0, 1024, 120);
    EXPECT_FALSE(tiny.rootPitchClass == 0 && tiny.type == CT::Major);

    // Requests below the clamp floor land INTO 1024 (documented), and 1024
    // IS meaningful at a low rate: 8 kHz spans 128 ms and resolves C4.
    int wClamp = 0, wLow = 0;
    (void)detectAt({60, 64, 67}, 48000.0, 512, 8, &wClamp);
    EXPECT_EQ(wClamp, 1024);
    const auto low = detectAt({60, 64, 67}, 8000.0, 0, 120, &wLow);
    EXPECT_EQ(wLow, 1024);
    EXPECT_EQ(low.rootPitchClass, 0);
    EXPECT_TRUE(low.type == CT::Major);
}

DSPARK_TEST(Harmony_diatonicChord_heptatonic_contract)
{
    // C major (Ionian): textbook triad quality on every degree.
    const auto& ionian = allScales[0];
    const std::string_view triads[7] = { "M", "m", "m", "M", "M", "m", "dim" };
    const int thirds[7] = { 4, 3, 3, 4, 4, 3, 3 };
    const int fifths[7] = { 7, 7, 7, 7, 7, 7, 6 };
    for (int d = 0; d < 7; ++d)
    {
        const auto c = diatonicChord(ionian, d, ChordLevel::TriadsOnly);
        EXPECT_TRUE(c.view() == triads[d]);
        EXPECT_EQ(c.intervals[0], 0);
        EXPECT_EQ(c.intervals[1], thirds[d]);
        EXPECT_EQ(c.intervals[2], fifths[d]);
    }

    // Sevenths: I maj7, V dominant 7, vii m7b5.
    EXPECT_TRUE(diatonicChord(ionian, 0, ChordLevel::Triads7).view()
                == std::string_view("maj7"));
    EXPECT_TRUE(diatonicChord(ionian, 4, ChordLevel::Triads7).view()
                == std::string_view("7"));
    EXPECT_TRUE(diatonicChord(ionian, 6, ChordLevel::Triads7).view()
                == std::string_view("m7b5"));

    // Invalid degrees return an empty chord.
    EXPECT_TRUE(diatonicChord(ionian, -1, ChordLevel::TriadsOnly).view().empty());
    EXPECT_TRUE(diatonicChord(ionian, 7, ChordLevel::TriadsOnly).view().empty());

    // Non-heptatonic scales are documented as musically UNSPECIFIED but must
    // stay safe: every scale x degree x level yields a NUL-terminated name
    // inside the buffer and a root interval of 0.
    for (const auto& sc : allScales)
        for (int d = 0; d < 7; ++d)
        {
            const auto c = diatonicChord(sc, d, ChordLevel::Triads791113);
            EXPECT_TRUE(c.view().size() < c.name.size());
            EXPECT_EQ(c.intervals[0], 0);
        }

    // The documented example of the sharp edge: MajorPentatonic degree 0
    // stacks {0, 4, 9} (a 9-semitone "fifth") and names it "?".
    const auto& penta = allScales[33];
    const auto odd = diatonicChord(penta, 0, ChordLevel::TriadsOnly);
    EXPECT_EQ(odd.intervals[1], 4);
    EXPECT_EQ(odd.intervals[2], 9);
    EXPECT_TRUE(odd.view() == std::string_view("?"));
}

// ============================================================================
// Above the E5 root ceiling the detector has no correct answer to give, but it
// is not silent: the window-fill transient can push a WRONG chord through the
// confidence gate, and hold-last-confident then keeps it on display for as
// long as the programme runs. This pins the documented hazard and the
// documented mitigation (reset() on programme change).
// ============================================================================
DSPARK_TEST(ChordDetector_above_ceiling_fill_transient_latches_and_persists)
{
    using CT = ChordDetector<float>::ChordType;
    const double fs = 48000.0;

    ChordDetector<float> det;
    det.prepare(spec(fs, 512, 2));
    const int N = det.getWindowSize();

    // C6 root-position major triad (MIDI 84/88/91): the root is 8 semitones
    // above the E5 ceiling, so NO reading of this signal is correct.
    auto midiHz = [](int m) { return 440.0 * std::exp2((m - 69) / 12.0); };
    auto buf = makeStereoBuffer(512);
    const long long total = 12LL * N;
    long long n = 0;
    int gatePassesDuringFill = 0, gatePassesAfterFill = 0;
    ChordDetector<float>::Result firstPass {};
    while (n < total)
    {
        for (int i = 0; i < 512; ++i)
        {
            double v = 0.0;
            for (int m : { 84, 88, 91 })
                v += std::sin(dspark::twoPi<double> * midiHz(m)
                              * static_cast<double>(n + i) / fs);
            buf.ch(0)[i] = static_cast<float>(0.25 * v);
            buf.ch(1)[i] = buf.ch(0)[i];
        }
        det.processBlock(AudioBufferView<const float>(buf.view()));
        n += 512;
        const auto r = det.getChord();
        if (r.confidence >= det.getConfidenceThreshold() && r.type != CT::None)
        {
            if (n <= 2LL * N) { if (gatePassesDuringFill++ == 0) firstPass = r; }
            else              ++gatePassesAfterFill;
        }
    }

    std::cout << "  [ceiling] fill passes=" << gatePassesDuringFill
              << " steady passes=" << gatePassesAfterFill
              << " latched pc=" << firstPass.rootPitchClass
              << " type=" << static_cast<int>(firstPass.type)
              << " conf=" << firstPass.confidence << "\n";

    // The hazard: something passed the gate while the window was filling...
    EXPECT_GT(gatePassesDuringFill, 0);
    // ...it was not the chord that is playing (C major would be pc 0)...
    EXPECT_FALSE(firstPass.rootPitchClass == 0 && firstPass.type == CT::Major);
    // ...nothing passes once the window is full (so no later reading corrects
    // it)...
    EXPECT_EQ(gatePassesAfterFill, 0);
    // ...and the wrong chord is still what a caller reads at the end, at a
    // confidence BELOW the gate: a stale value, not a current one.
    const auto held = det.getChord();
    EXPECT_EQ(held.rootPitchClass, firstPass.rootPitchClass);
    EXPECT_TRUE(held.type == firstPass.type);
    EXPECT_LT(held.confidence, det.getConfidenceThreshold());

    // The documented mitigation clears it.
    det.reset();
    const auto afterReset = det.getChord();
    EXPECT_EQ(afterReset.rootPitchClass, -1);
    EXPECT_TRUE(afterReset.type == CT::None);
}
