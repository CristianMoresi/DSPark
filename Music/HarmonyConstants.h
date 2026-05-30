// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file HarmonyConstants.h
 * @brief Comprehensive musical harmony calculations — scales, chords, MIDI, theory.
 *
 * A complete, strictly constexpr toolkit for working with musical scales and chords.
 * Fully functional at compile-time (`constexpr` / `consteval`) to guarantee zero-overhead
 * runtime execution and generate static data tables for audio plugins and musical analysis.
 * 
 * Complies with DSPark strict real-time constraints:
 * - Zero allocations (no std::string, no std::vector).
 * - Cache-friendly bitmask operations (O(1) lookups and bitwise rotations).
 * - SIMD/Thread-safe (completely stateless and immutable).
 *
 * Dependencies: C++20 standard library only.
 */

#include <array>
#include <string_view>
#include <cstdint>
#include <algorithm>
#include <optional>

namespace dspark {
namespace harmony
{
    //==============================================================================
    // 0. CORE TYPEDEFS
    //==============================================================================

    /**
     * @typedef NoteSet
     * @brief A 12-bit bitmask representing the 12 pitch-classes of the chromatic scale.
     *
     * Bits 0..11 correspond to semitones above the root (0=C, 1=C#/Db, ..., 11=B).
     * Fits perfectly in CPU registers for extremely fast subset/superset evaluations.
     */
    using NoteSet = std::uint16_t;

    /**
     * @typedef Degree
     * @brief Integer index used to select a degree inside standard diatonic representations (0-based).
     */
    using Degree = int;

    //==============================================================================
    // 1. CHORDTAG - A FILTER FOR CHORD TYPES
    //==============================================================================

    /**
     * @enum ChordTag
     * @brief Bitmask flags describing chord "families" compatible with a scale.
     *
     * Used to filter scales by which chord types are naturally available
     * inside them without needing to compute intervals dynamically.
     */
    enum class ChordTag : std::uint16_t
    {
        MajorTriad      = 1u << 0,   ///< Contains a major triad (e.g., C-E-G).
        MinorTriad      = 1u << 1,   ///< Contains a minor triad (e.g., C-Eb-G).
        DiminishedTriad = 1u << 2,   ///< Contains a diminished triad (e.g., C-Eb-Gb).
        AugmentedTriad  = 1u << 3,   ///< Contains an augmented triad (e.g., C-E-G#).
        Major7          = 1u << 4,   ///< Contains a major 7th chord.
        Dominant7       = 1u << 5,   ///< Contains a dominant 7th chord.
        Minor7          = 1u << 6,   ///< Contains a minor 7th chord.
        HalfDim7        = 1u << 7,   ///< Contains a half-diminished 7th chord.
        Dim7            = 1u << 8,   ///< Contains a fully-diminished 7th chord.
        Sus2Triad       = 1u << 9,   ///< Contains a suspended-2 triad.
        Sus4Triad       = 1u << 10,  ///< Contains a suspended-4 triad.
        Major9          = 1u << 11,  ///< Contains a major 9th chord.
        Dominant9       = 1u << 12,  ///< Contains a dominant 9th chord.
        Minor9          = 1u << 13,  ///< Contains a minor 9th chord.
        Major11         = 1u << 14,  ///< Contains a major 11th chord.
        Dominant11      = 1u << 15,  ///< Contains a dominant 11th chord.
        All             = 0xFFFFu    ///< Convenience: all flags set.
    };

    /**
     * @brief Bitwise OR operator for ChordTag flags.
     * @return Combination of both flags.
     */
    [[nodiscard]] constexpr ChordTag operator|(ChordTag lhs, ChordTag rhs) noexcept
    {
        using U = std::underlying_type_t<ChordTag>;
        return static_cast<ChordTag>(static_cast<U>(lhs) | static_cast<U>(rhs));
    }

    //==============================================================================
    // 2. SCALE DESCRIPTOR
    //==============================================================================

    /**
     * @struct Scale
     * @brief Descriptor holding the name, pitch mask and chord tags for a musical scale.
     */
    struct Scale
    {
        std::string_view name; ///< Human-readable scale name (root = C in the database).
        NoteSet          mask; ///< 12-bit mask with set bits for the scale degrees.
        ChordTag         tags; ///< Flags describing chord families present in the scale.
    };

    //==============================================================================
    // 3. INTERNAL HELPERS & MASK GENERATION
    //==============================================================================

    /**
     * @brief Build a NoteSet from 12 boolean flags (b0 = C, b1 = C#/Db, ... b11 = B).
     */
    [[nodiscard]] consteval NoteSet makeMask(
        bool b0, bool b1, bool b2, bool b3,
        bool b4, bool b5, bool b6, bool b7,
        bool b8, bool b9, bool b10, bool b11) noexcept
    {
        return (static_cast<NoteSet>(b0)  << 0)  | (static_cast<NoteSet>(b1)  << 1)  |
               (static_cast<NoteSet>(b2)  << 2)  | (static_cast<NoteSet>(b3)  << 3)  |
               (static_cast<NoteSet>(b4)  << 4)  | (static_cast<NoteSet>(b5)  << 5)  |
               (static_cast<NoteSet>(b6)  << 6)  | (static_cast<NoteSet>(b7)  << 7)  |
               (static_cast<NoteSet>(b8)  << 8)  | (static_cast<NoteSet>(b9)  << 9)  |
               (static_cast<NoteSet>(b10) << 10) | (static_cast<NoteSet>(b11) << 11);
    }

    /**
     * @brief Build a NoteSet from a list of semitone degrees.
     * @details Degree values are taken modulo 12. Negative degrees are ignored.
     */
    [[nodiscard]] consteval NoteSet makeMask(std::initializer_list<int> degrees) noexcept
    {
        NoteSet m = 0;
        for (int d : degrees) {
            if (d >= 0) { m |= static_cast<NoteSet>(1u << (d % 12)); }
        }
        return m & 0x0FFFu;
    }

    namespace detail
    {
        /**
         * @brief Safe small string copy used only for compile-time-friendly name building.
         */
        constexpr void copy(char* dst, std::string_view src, std::size_t dstCapacity) noexcept
        {
            std::size_t toCopy = src.size();
            if (toCopy + 1u > dstCapacity) { toCopy = (dstCapacity > 0) ? dstCapacity - 1u : 0u; }
            for (std::size_t i = 0; i < toCopy; ++i) { dst[i] = src[i]; }
            if (dstCapacity > 0) { dst[toCopy] = '\0'; }
        }

        /**
         * @brief Extract active scale degrees and expand them sequentially across the octave.
         */
        [[nodiscard]] constexpr std::array<int, 7> activeDegrees(NoteSet mask) noexcept
        {
            std::array<int, 12> temp{};
            int tcount = 0;
            for (int i = 0; i < 12; ++i) {
                if (mask & (1u << i)) temp[tcount++] = i;
            }

            std::array<int, 7> out{};
            if (tcount == 0) {
                for (int i = 0; i < 7; ++i) out[i] = i;
                return out;
            }

            for (int i = 0; i < 7; ++i) {
                int idx = i % tcount;
                int octave = i / tcount;
                out[i] = temp[idx] + octave * 12;
            }
            return out;
        }

        /**
         * @brief Calculate semitone interval skipping 'skip' degrees in the scale.
         */
        [[nodiscard]] constexpr int interval(const std::array<int, 7>& deg, int degIdx, int skip) noexcept
        {
            int a = deg[degIdx];
            int b = deg[(degIdx + skip) % 7];
            while (b <= a) { b += 12; }
            return b - a;
        }
    } // namespace detail

    //==============================================================================
    // 4. DATABASE OF SCALES & CONSTANTS
    //==============================================================================

    inline constexpr std::array<Scale, 61> allScales = [](){
        std::array<Scale, 61> scales{};

        // Major modes
        scales[0]  = {"Ionian",          makeMask({0,2,4,5,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7 | ChordTag::Dominant7};
        scales[1]  = {"Dorian",          makeMask({0,2,3,5,7,9,10}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[2]  = {"Phrygian",        makeMask({0,1,3,5,7,8,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7};
        scales[3]  = {"Lydian",          makeMask({0,2,4,6,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7};
        scales[4]  = {"Mixolydian",      makeMask({0,2,4,5,7,9,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[5]  = {"Aeolian",         makeMask({0,2,3,5,7,8,10}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[6]  = {"Locrian",         makeMask({0,1,3,5,6,8,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7};

        // Melodic-minor modes
        scales[7]  = {"MelodicMinor",    makeMask({0,2,3,5,7,9,11}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[8]  = {"Dorianb2",        makeMask({0,1,3,5,7,9,10}), ChordTag::MinorTriad};
        scales[9]  = {"LydianAugmented", makeMask({0,2,4,6,8,9,11}), ChordTag::MajorTriad | ChordTag::AugmentedTriad};
        scales[10] = {"LydianDominant",  makeMask({0,2,4,6,7,9,10}), ChordTag::Dominant7};
        scales[11] = {"Mixolydianb6",    makeMask({0,2,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[12] = {"HalfDiminished",  makeMask({0,2,3,5,6,8,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7};
        scales[13] = {"AlteredDominant", makeMask({0,1,3,4,6,8,10}), ChordTag::Dominant7};

        // Harmonic-minor modes
        scales[14] = {"HarmonicMinor",   makeMask({0,2,3,5,7,8,11}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[15] = {"Locrian6",        makeMask({0,1,3,5,6,9,10}), ChordTag::MinorTriad | ChordTag::DiminishedTriad};
        scales[16] = {"IonianAugmented", makeMask({0,2,4,5,8,9,11}), ChordTag::MajorTriad | ChordTag::AugmentedTriad};
        scales[17] = {"DorianSharp4",    makeMask({0,2,3,6,7,9,10}), ChordTag::MinorTriad};
        scales[18] = {"PhrygianDominant",makeMask({0,1,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[19] = {"LydianSharp2",    makeMask({0,3,4,6,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7};
        scales[20] = {"UltraLocrian",    makeMask({0,1,3,4,6,8,9}),  ChordTag::DiminishedTriad | ChordTag::Dim7};

        // Harmonic-major modes
        scales[21] = {"HarmonicMajor",     makeMask({0,2,4,5,7,8,11}), ChordTag::MajorTriad | ChordTag::AugmentedTriad}; 
        scales[22] = {"Dorianb5",          makeMask({0,2,3,5,6,9,11}), ChordTag::MinorTriad | ChordTag::DiminishedTriad};
        scales[23] = {"Phrygianb4",        makeMask({0,1,3,4,6,9,10}), ChordTag::MinorTriad}; 
        scales[24] = {"LydianMinor",       makeMask({0,2,4,6,7,8,10}), ChordTag::MinorTriad | ChordTag::Minor7};
        scales[25] = {"Mixolydianb9",      makeMask({0,1,4,5,7,8,10}), ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[26] = {"LydianAugmented2",  makeMask({0,3,4,6,7,9,11}), ChordTag::MajorTriad | ChordTag::Major7};
        scales[27] = {"LocrianDiminished", makeMask({0,1,3,4,6,7,9}),  ChordTag::DiminishedTriad | ChordTag::Dim7}; 

        // Double-harmonic family
        scales[28] = {"DoubleHarmonic",    makeMask({0,1,4,5,7,8,11}), ChordTag::MajorTriad};
        scales[29] = {"HungarianMinor",    makeMask({0,2,3,6,7,8,11}), ChordTag::MinorTriad};
        scales[30] = {"Byzantine",         makeMask({0,1,4,5,7,8,11}), ChordTag::MajorTriad};
        scales[31] = {"Ionian b5",         makeMask({0,2,4,5,6,9,11}), ChordTag::MajorTriad | ChordTag::DiminishedTriad};
        scales[32] = {"Lydian #6",         makeMask({0,2,4,6,7,10,11}),ChordTag::MajorTriad | ChordTag::Major7};
        
        // Pentatonics
        scales[33] = {"MajorPentatonic",   makeMask({0,2,4,7,9}),      ChordTag::MajorTriad};
        scales[34] = {"MinorPentatonic",   makeMask({0,3,5,7,10}),     ChordTag::MinorTriad};
        scales[35] = {"EgyptianPentatonic",makeMask({0,2,5,7,10}),     ChordTag::MajorTriad};
        scales[36] = {"Hirajoshi",         makeMask({0,2,3,7,8}),      ChordTag::MinorTriad};
        scales[37] = {"InSen",             makeMask({0,1,5,7,10}),     ChordTag::MinorTriad};
        scales[38] = {"Yo",                makeMask({0,4,5,7,11}),     ChordTag::MajorTriad};

        // Symmetrical / exotic
        scales[39] = {"WholeTone",         makeMask({0,2,4,6,8,10}),               ChordTag::AugmentedTriad};
        scales[40] = {"Chromatic",         makeMask({0,1,2,3,4,5,6,7,8,9,10,11}),  ChordTag::All};
        scales[41] = {"Diminished",        makeMask({0,2,3,5,6,8,9,11}),           ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7 | ChordTag::Dim7};
        scales[42] = {"Diminished2",       makeMask({0,1,3,4,6,7,9,10}),           ChordTag::MinorTriad | ChordTag::DiminishedTriad | ChordTag::HalfDim7 | ChordTag::Dim7};
        scales[43] = {"Augmented",         makeMask({0,3,4,7,8,11}),               ChordTag::AugmentedTriad};

        // Additional Scales
        scales[44] = {"Algerian",          makeMask({0,2,3,6,7,8,11}),   ChordTag::MinorTriad};
        scales[45] = {"Arabian",           makeMask({0,2,4,5,6,8,10}),   ChordTag::MajorTriad};
        scales[46] = {"Balinese",          makeMask({0,1,3,7,8}),        ChordTag::MinorTriad};
        scales[47] = {"Chinese",           makeMask({0,4,6,7,11}),       ChordTag::MajorTriad};
        scales[48] = {"Gypsy",             makeMask({0,1,4,5,7,8,10}),   ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[49] = {"Hindu",             makeMask({0,2,4,5,7,9,10}),   ChordTag::MajorTriad | ChordTag::Dominant7};
        scales[50] = {"Hungarian",         makeMask({0,2,3,6,7,8,11}),   ChordTag::MinorTriad};
        scales[51] = {"Japanese",          makeMask({0,1,5,7,8}),        ChordTag::MinorTriad};
        scales[52] = {"Javanese",          makeMask({0,1,3,5,7,10}),     ChordTag::MajorTriad};
        scales[53] = {"Mongolian",         makeMask({0,2,4,7,9}),        ChordTag::MajorTriad};
        scales[54] = {"Neapolitan",        makeMask({0,1,3,5,7,8,11}),   ChordTag::MinorTriad};
        scales[55] = {"Oriental",          makeMask({0,1,4,5,6,9,10}),   ChordTag::MajorTriad};
        scales[56] = {"Persian",           makeMask({0,1,4,5,6,8,11}),   ChordTag::MajorTriad};
        scales[57] = {"Prometheus",        makeMask({0,2,4,6,9,10}),     ChordTag::MajorTriad};
        scales[58] = {"Spanish",           makeMask({0,1,3,4,5,7,8,10}), ChordTag::DiminishedTriad};
        scales[59] = {"Tritone",           makeMask({0,1,4,6,7,10}),     ChordTag::Dominant7};
        scales[60] = {"Ukrainian",         makeMask({0,2,3,6,7,9,10}),   ChordTag::MinorTriad};

        return scales;
    }();

    inline constexpr std::array<std::string_view, 12> sharpNames{
        "C","C#","D","D#","E","F","F#","G","G#","A","A#","B"
    };
    inline constexpr std::array<std::string_view, 12> flatNames{
        "C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"
    };
    inline constexpr std::array<bool,12> useSharpsForRoot{
        1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0
    };

    //==============================================================================
    // 5. CONTEXT-AWARE NOTE NAMES & PARSING
    //==============================================================================

    /**
     * @brief Returns a human-readable note name for the given MIDI note (0..127).
     * @param midi The MIDI note number.
     * @param root The pitch-class of the key root (0..11) to decide enharmonic spelling.
     * @return std::string_view pointing to a static note name (e.g., "C#", "Db").
     */
    [[nodiscard]] constexpr std::string_view noteName(int midi, int root = 0) noexcept
    {
        int idx = (midi % 12 + 12) % 12;
        int r   = (root % 12 + 12) % 12;
        return useSharpsForRoot[r] ? sharpNames[idx] : flatNames[idx];
    }

    /**
     * @brief Parse a simple note name (no octave) into a pitch-class (0..11).
     * @param s String containing note name (e.g., "C#", "Bb", "Fb").
     * @return Pitch-class 0..11 on success, std::nullopt on failure.
     */
    [[nodiscard]] constexpr std::optional<int> parseNote(std::string_view s) noexcept
    {
        constexpr std::array<std::pair<std::string_view,int>, 28> lut{{
            {"C",0}, {"B#",0},
            {"C#",1}, {"Db",1},
            {"D",2}, {"D#",3}, {"Eb",3},
            {"E",4}, {"E#",5}, {"Fb",4},
            {"F",5}, {"F#",6}, {"Gb",6},
            {"G",7}, {"G#",8}, {"Ab",8},
            {"A",9}, {"A#",10},{"Bb",10},
            {"B",11}, {"Cb",11}
        }};

        for (const auto& [name,val] : lut) {
            if (name.size() == s.size() &&
                std::equal(name.begin(), name.end(), s.begin(),
                           [](char a, char b){ return (a|32) == (b|32); })) {
                return val;
            }
        }
        return std::nullopt;
    }

    //==============================================================================
    // 6. TRANSPOSE A SCALE
    //==============================================================================

    /**
     * @brief Circularly rotate a NoteSet so it becomes rooted at a specific key.
     * @param base NoteSet defined with root = C.
     * @param root Desired root as semitone offset from C (0..11).
     * @return New bitmask representing the transposed scale.
     */
    [[nodiscard]] constexpr NoteSet scaleAtRoot(NoteSet base, int root) noexcept
    {
        root = (root % 12 + 12) % 12;
        std::uint32_t b = static_cast<std::uint32_t>(base) & 0x0FFFu;
        std::uint32_t res = ((b << root) | (b >> (12 - root))) & 0x0FFFu;
        return static_cast<NoteSet>(res);
    }

    //==============================================================================
    // 7. CHORD DESCRIPTOR
    //==============================================================================

    /**
     * @struct Chord
     * @brief A chord "recipe" defining the required intervals relative to its root.
     */
    struct Chord
    {
        std::string_view   name;
        std::array<int, 7> intervals; ///< Intervals in semitones from root. -1 = not present.
    };

    inline constexpr std::array<Chord, 15> allChords{{
        {"Major",      {0, 4, 7, -1, -1, -1, -1}},
        {"Minor",      {0, 3, 7, -1, -1, -1, -1}},
        {"Diminished", {0, 3, 6, -1, -1, -1, -1}},
        {"Augmented",  {0, 4, 8, -1, -1, -1, -1}},
        {"Major7",     {0, 4, 7, 11, -1, -1, -1}},
        {"Dominant7",  {0, 4, 7, 10, -1, -1, -1}},
        {"Minor7",     {0, 3, 7, 10, -1, -1, -1}},
        {"HalfDim7",   {0, 3, 6, 10, -1, -1, -1}},
        {"Dim7",       {0, 3, 6,  9, -1, -1, -1}},
        {"Sus2",       {0, 2, 7, -1, -1, -1, -1}},
        {"Sus4",       {0, 5, 7, -1, -1, -1, -1}},
        {"Major9",     {0, 4, 7, 11, 14, -1, -1}},
        {"Dominant9",  {0, 4, 7, 10, 14, -1, -1}},
        {"Minor9",     {0, 3, 7, 10, 14, -1, -1}},
        {"Major13",    {0, 4, 7, 11, 14, 17, 21}}
    }};

    //==============================================================================
    // 8. CHORD GENERATION & REVERSE LOOKUP
    //==============================================================================

    /**
     * @brief Build MIDI note numbers for a chord recipe located at a specific root.
     * @param c Chord definition to use.
     * @param rootMidi MIDI note for chord root (0..127 typical).
     * @param inversion Which chord tone to place in bass (0=root position, 1=first inversion...).
     * @return Array of 7 ints: valid MIDI numbers for present tones, unused slots = -1.
     */
    [[nodiscard]] constexpr std::array<int, 7>
    chordAtRootMidi(const Chord& c, int rootMidi, int inversion = 0) noexcept
    {
        std::array<int, 7> notes{};
        int count = 0;

        for (int i = 0; i < 7; ++i) {
            const int deg = c.intervals[i];
            if (deg < 0) break;
            notes[count++] = rootMidi + deg;
        }

        if (count == 0) return { -1,-1,-1,-1,-1,-1,-1 };

        if (count > 0) {
            inversion = (inversion >= 0) ? (inversion % count) : 0;
            for (int i = 0; i < inversion; ++i) { notes[i] += 12; }
            std::sort(notes.begin(), notes.begin() + count);
        }

        for (int i = count; i < 7; ++i) { notes[i] = -1; }
        return notes;
    }

    /**
     * @brief Reverse lookup: find up to 16 scales that fully contain `chordMask`.
     * @param chordMask A NoteSet representing the notes of the chord.
     * @return Array of pointers to compatible scales. Remaining entries are nullptr.
     */
    [[nodiscard]] constexpr std::array<const Scale*, 16>
    scalesForChordMask(NoteSet chordMask) noexcept
    {
        std::array<const Scale*, 16> out{};
        std::size_t idx = 0;
        for (const auto& s : allScales) {
            if (((s.mask & chordMask) == chordMask) && idx < out.size()) {
                out[idx++] = &s;
            }
        }
        return out;
    }

    /**
     * @brief Reverse lookup wrapper: find scales that can contain a specific Chord recipe.
     * @param chord The Chord recipe object.
     * @return Array of pointers to compatible scales.
     */
    [[nodiscard]] constexpr std::array<const Scale*, 16>
    scalesForChord(const Chord& chord) noexcept
    {
        NoteSet chordMask = 0;
        for (int d : chord.intervals) {
            if (d >= 0) { chordMask |= static_cast<NoteSet>(1u << (d % 12)); }
        }
        return scalesForChordMask(chordMask);
    }

    //==============================================================================
    // 9. DIATONIC CHORD GENERATION
    //==============================================================================

    /**
     * @enum ChordLevel
     * @brief Complexity level describing which extensions to generate for diatonic chords.
     */
    enum class ChordLevel : std::uint8_t
    {
        TriadsOnly   = 0, ///< Generate base triads only (R-3-5).
        Triads7      = 1, ///< Generate up to 7th chords.
        Triads79     = 2, ///< Up to 9ths.
        Triads7911   = 3, ///< Up to 11ths.
        Triads791113 = 4  ///< Up to 13ths.
    };

    /**
     * @struct DiatonicChord
     * @brief Contains the structure and generated symbol name of a diatonic chord.
     */
    struct DiatonicChord
    {
        std::array<char, 24> name;      ///< NUL-terminated buffer ("m7", "maj7"...). Zero alloc.
                                        ///< Sized for the longest symbol "m(maj7)(9)(11)(13)" (18+NUL).
        std::array<int, 7>   intervals; ///< Intervals in semitones (R-3-5-7-9-11-13). -1 = absent.

        /**
         * @brief Safely obtain a string_view of the internal name buffer.
         * @return A std::string_view tightly bound to the null-terminated string length.
         */
        [[nodiscard]] constexpr std::string_view view() const noexcept
        {
            std::size_t len = 0;
            while (len < name.size() && name[len] != '\0') { ++len; }
            return std::string_view(name.data(), len);
        }
    };

    /**
     * @brief Generates the diatonic chord built upon a specific degree of a scale.
     * @param sc Scale definition (assumes root=C internally).
     * @param degree The 0-based degree index inside the scale (0..6).
     * @param level Which chord extensions to include.
     * @return A DiatonicChord object with the dynamically evaluated intervals and symbol.
     */
    [[nodiscard]] constexpr DiatonicChord
    diatonicChord(const Scale& sc, Degree degree, ChordLevel level) noexcept
    {
        const auto deg = detail::activeDegrees(sc.mask);
        if (degree < 0 || degree >= 7) return {}; // Invalid input protection

        const int third      = detail::interval(deg, degree, 2);
        const int fifth      = detail::interval(deg, degree, 4);
        const int seventh    = detail::interval(deg, degree, 6);
        const int ninth      = detail::interval(deg, degree, 1);
        const int eleventh   = detail::interval(deg, degree, 3);
        const int thirteenth = detail::interval(deg, degree, 5);

        DiatonicChord c{};
        c.intervals = {0, third, fifth,
                       (level >= ChordLevel::Triads7)      ? seventh    : -1,
                       (level >= ChordLevel::Triads79)     ? ninth      : -1,
                       (level >= ChordLevel::Triads7911)   ? eleventh   : -1,
                       (level >= ChordLevel::Triads791113) ? thirteenth : -1};

        auto buildName = [&](std::string_view baseName, ChordLevel lvl)
        {
            char buf[24]{}; // Match DiatonicChord::name capacity (fits longest symbol)
            std::size_t pos = 0;
            detail::copy(buf + pos, baseName, sizeof(buf) - pos);
            pos += baseName.size();

            if (lvl >= ChordLevel::Triads79)     { detail::copy(buf + pos, "(9)",  sizeof(buf) - pos);  pos += 3; }
            if (lvl >= ChordLevel::Triads7911)   { detail::copy(buf + pos, "(11)", sizeof(buf) - pos);  pos += 4; }
            if (lvl >= ChordLevel::Triads791113) { detail::copy(buf + pos, "(13)", sizeof(buf) - pos);  pos += 4; }

            detail::copy(c.name.data(), std::string_view(buf, pos), c.name.size());
        };

        std::string_view base;
        if      (third == 4 && fifth == 7) base = "M";
        else if (third == 3 && fifth == 7) base = "m";
        else if (third == 3 && fifth == 6) base = "dim";
        else if (third == 4 && fifth == 8) base = "aug";
        else                               base = "?";

        std::string_view name7;
        if      (base == "dim" && seventh == 10) name7 = "m7b5";
        else if (base == "dim" && seventh == 9)  name7 = "dim7";
        else if (base == "M"   && seventh == 10) name7 = "7";
        else if (base == "M"   && seventh == 11) name7 = "maj7";
        else if (base == "m"   && seventh == 10) name7 = "m7";
        else if (base == "m"   && seventh == 11) name7 = "m(maj7)";
        else if (base == "aug" && seventh == 10) name7 = "aug7";
        else                                     name7 = base;

        switch (level)
        {
            case ChordLevel::TriadsOnly:   buildName(base,  ChordLevel::TriadsOnly);   break;
            case ChordLevel::Triads7:      buildName(name7, ChordLevel::Triads7);      break;
            case ChordLevel::Triads79:     buildName(name7, ChordLevel::Triads79);     break;
            case ChordLevel::Triads7911:   buildName(name7, ChordLevel::Triads7911);   break;
            case ChordLevel::Triads791113: buildName(name7, ChordLevel::Triads791113); break;
        }
        
        return c;
    }

    /**
     * @brief Convert a DiatonicChord's interval recipe into absolute MIDI notes.
     * @param c The DiatonicChord object.
     * @param rootMidi The MIDI note assigned as the root of the generated chord.
     * @return Array of 7 ints with valid MIDI notes, -1 for absent positions.
     */
    [[nodiscard]] constexpr std::array<int, 7>
    diatonicChordToMidi(const DiatonicChord& c, int rootMidi) noexcept
    {
        std::array<int, 7> notes{};
        for (std::size_t i = 0; i < 7; ++i) {
            notes[i] = (c.intervals[i] >= 0) ? rootMidi + c.intervals[i] : -1;
        }
        return notes;
    }

    //==============================================================================
    // 10. OCTAVE & NOTE HELPERS
    //==============================================================================

    /**
     * @brief Parse a note string containing an optional octave (e.g. "C#4", "C-1").
     * @param note String representation of the note.
     * @return Pitch-class (0..11) on success, or std::nullopt if format is invalid.
     */
    [[nodiscard]] constexpr std::optional<int> parseNoteWithOctave(std::string_view note) noexcept
    {
        std::size_t len = note.size();
        while (len > 0 && note[len - 1] >= '0' && note[len - 1] <= '9') { --len; }
        if (len > 0 && note[len - 1] == '-') { --len; }

        std::string_view baseNote = note.substr(0, len);
        return parseNote(baseNote);
    }

    /**
     * @brief Extract octave number from a note string. Supports negative octaves.
     * @param note String representation of the note (e.g. "C#5", "C-1").
     * @return Valid octave integer. Defaults to 4 if not present.
     */
    [[nodiscard]] constexpr int getOctaveFromNote(std::string_view note) noexcept
    {
        if (note.empty()) return 4;
        std::size_t len = note.size();
        
        while (len > 0 && note[len - 1] >= '0' && note[len - 1] <= '9') { --len; }
        
        bool isNegative = false;
        if (len > 0 && note[len - 1] == '-') {
            isNegative = true;
            --len;
        }

        if (len < note.size() && !(isNegative && len + 1 == note.size())) {
            int parsed = 0;
            std::size_t start = isNegative ? len + 1 : len;
            for (std::size_t i = start; i < note.size(); ++i) {
                parsed = parsed * 10 + (note[i] - '0');
            }
            return isNegative ? -parsed : parsed;
        }
        
        return 4; 
    }

    /**
     * @brief Transpose a MIDI note by a discrete number of full octaves.
     * @param midi Original MIDI note number.
     * @param octaveDelta Number of octaves to transpose (can be negative).
     * @return New MIDI note number.
     */
    [[nodiscard]] constexpr int transposeByOctaves(int midi, int octaveDelta) noexcept
    {
        return midi + octaveDelta * 12;
    }

} // namespace harmony
} // namespace dspark
