// DSPark Tests - Music
// HarmonyConstants: scales, chords, note parsing, transposition

#include "dspark_test.h"
#include "TestSignals.h"
#include "../Music/HarmonyConstants.h"
#include "../Music/ChordDetector.h"

#include <cmath>
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
