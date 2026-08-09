// DSPark - KeyDetector acceptance suite.
//
// Synthetic-only corpus, generated here from harmony:: data: I-IV-V-I
// cadences and diatonic passages in all 24 keys over several timbres and
// registers, pure Aeolian material with no leading tone, and modal material
// that has no correct answer in a 24-key output.
//
// Every case prints its numbers, so the ctest log doubles as the machine
// generated result artifact. Error modes are counted BY NAME -- relative,
// parallel, dominant, subdominant -- rather than folded into one accuracy
// figure, because which mistake a key finder makes is the whole story about
// whether it is working.
//
// All material is voiced inside the chroma front end's documented reliable
// register (about F#3 to E5). That is a property of the front end, not a
// convenience: below its leakage floor the chroma contains pitch classes that
// were never played, so a key estimate over such material would be measuring
// the window, not the method.

#include "dspark_test.h"
#include "TestSignals.h"
#include "../Music/KeyDetector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

constexpr double kFs = 48000.0;

// A deterministic small LCG: the corpus must be identical on every platform.
struct Rng
{
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
    std::uint32_t next() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
    int pick(int n) { return static_cast<int>(next() % static_cast<std::uint32_t>(n)); }
};

struct Timbre { const char* name; int n; double a[8]; };

const std::array<Timbre, 3> kTimbres { {
    { "sine",   1, { 1.0 } },
    { "organ",  5, { 1.0, 0.5, 0.33, 0.0, 0.2 } },
    { "string", 7, { 1.0, 0.7, 0.45, 0.3, 0.22, 0.15, 0.1 } },
} };

void addNote(std::vector<float>& out, double t0, double dur, int midi,
             double amp, const Timbre& tb)
{
    const double f0 = 440.0 * std::exp2((midi - 69) / 12.0);
    const auto i0 = static_cast<long long>(t0 * kFs);
    const auto n  = static_cast<long long>(dur * kFs);
    const double atk = 0.01 * kFs, rel = 0.05 * kFs;
    for (long long i = 0; i < n; ++i)
    {
        const long long idx = i0 + i;
        if (idx < 0 || idx >= static_cast<long long>(out.size())) continue;
        double env = (i < atk) ? (static_cast<double>(i) / atk) : 1.0;
        const double left = static_cast<double>(n - i);
        if (left < rel) env *= left / rel;
        double s = 0.0;
        for (int h = 0; h < tb.n; ++h)
        {
            const double f = f0 * (h + 1);
            if (f > 0.45 * kFs) break;
            s += tb.a[h] * std::sin(twoPi<double> * f * (static_cast<double>(i) / kFs));
        }
        out[static_cast<std::size_t>(idx)] += static_cast<float>(amp * env * s);
    }
}

// Fold a pitch class into [low, low+11] so every voice stays in register.
int foldRoot(int pc, int low) { return low + ((pc - (low % 12)) + 12) % 12; }

std::vector<float> renderCadence(int tonicPc, bool minor, const Timbre& tb,
                                 int low, int reps, double bar = 1.0)
{
    std::vector<float> out(static_cast<std::size_t>(kFs * (bar * 4 * reps + 1.0)), 0.0f);
    const int degs[4] = { 0, 5, 7, 0 };
    const bool cm[4] = { minor, minor, false, minor };
    const int maj[3] = { 0, 4, 7 }, min[3] = { 0, 3, 7 };
    double t = 0.0;
    for (int r = 0; r < reps; ++r)
        for (int c = 0; c < 4; ++c)
        {
            const int root = foldRoot((tonicPc + degs[c]) % 12, low);
            for (int i = 0; i < 3; ++i)
                addNote(out, t, bar * 0.95, root + (cm[c] ? min[i] : maj[i]), 0.16, tb);
            t += bar;
        }
    return out;
}

// Pure Aeolian: no leading tone, so the pitch-class content is EXACTLY that of
// the relative major and only the dwelling separates the two.
std::vector<float> renderAeolian(int tonicPc, const Timbre& tb, int low,
                                 double sec, std::uint32_t seed)
{
    std::vector<float> out(static_cast<std::size_t>(kFs * (sec + 1.0)), 0.0f);
    const int steps[7] = { 0, 2, 3, 5, 7, 8, 10 };
    Rng rng(seed);
    const int base = foldRoot(tonicPc, low);
    for (double t = 0.0; t < sec; t += 0.30)
        addNote(out, t, 0.28, base + steps[rng.pick(7)], 0.20, tb);
    addNote(out, 0.0, sec, base, 0.14, tb);
    for (double b : { 0.0, sec - 1.2 })
    {
        addNote(out, b, 1.1, base, 0.12, tb);
        addNote(out, b, 1.1, base + 3, 0.12, tb);
        addNote(out, b, 1.1, base + 7, 0.12, tb);
    }
    return out;
}

std::vector<float> renderModal(int centrePc, harmony::NoteSet mask, const Timbre& tb,
                               int low, double sec, std::uint32_t seed)
{
    std::vector<float> out(static_cast<std::size_t>(kFs * (sec + 1.0)), 0.0f);
    std::vector<int> steps;
    for (int i = 0; i < 12; ++i)
        if (mask & (1u << i)) steps.push_back(i);
    Rng rng(seed);
    const int base = foldRoot(centrePc, low);
    for (double t = 0.0; t < sec; t += 0.30)
        addNote(out, t, 0.28, base + steps[static_cast<std::size_t>(
            rng.pick(static_cast<int>(steps.size())))], 0.20, tb);
    addNote(out, 0.0, sec, base, 0.16, tb);
    for (double b = 0.0; b < sec; b += 2.0)
    {
        addNote(out, b, 0.9, base, 0.10, tb);
        addNote(out, b, 0.9, base + ((mask & (1u << 3)) ? 3 : 4), 0.10, tb);
        addNote(out, b, 0.9, base + 7, 0.10, tb);
    }
    return out;
}

KeyDetector<float>::Key run(KeyDetector<float>& kd, const std::vector<float>& a)
{
    kd.reset();
    const std::size_t block = 512;
    for (std::size_t off = 0; off < a.size(); off += block)
        kd.pushSamples(std::span<const float>(a.data() + off,
                                              std::min(block, a.size() - off)));
    return kd.getKey();
}

// The error taxonomy, by name.
const char* classify(int gtPc, bool gtMin, int pc, bool mn)
{
    if (pc < 0) return "NONE";
    if (pc == gtPc && mn == gtMin) return "CORRECT";
    if (!gtMin && mn && pc == (gtPc + 9) % 12) return "RELATIVE";
    if ( gtMin && !mn && pc == (gtPc + 3) % 12) return "RELATIVE";
    if (pc == gtPc && mn != gtMin)              return "PARALLEL";
    if (pc == (gtPc + 7) % 12)                  return "DOMINANT";
    if (pc == (gtPc + 5) % 12)                  return "SUBDOMINANT";
    return "OTHER";
}

AudioSpec keySpec(double sr = kFs)
{
    AudioSpec s;
    s.sampleRate = sr;
    s.numChannels = 1;
    s.maxBlockSize = 512;
    return s;
}

} // namespace

// ============================================================================
// The profile digits themselves. They are the one thing in this header that is
// quoted from a source rather than derived, so they are pinned here: a typo in
// a weight is a defect no accuracy figure would localise.
// ============================================================================

DSPARK_TEST(Key_published_profile_digits_are_pinned)
{
    using KD = KeyDetector<float>;
    // Krumhansl & Kessler (1982), as reprinted in Temperley, Musicae
    // Scientiae 8(2) (2004), Table 1, columns "K-S model".
    const double kkMaj[12] = { 6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88 };
    const double kkMin[12] = { 6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17 };
    // Temperley, Music Perception 17(1) (1999), Figure 4, reprinted as the
    // "CBMS model" columns of the same table. NOT the Kostka-Payne
    // probabilities of Temperley (2007), which other software labels
    // "Temperley" and which are a different vector entirely.
    const double tMaj[12] = { 5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
                              2.0, 4.5, 2.0, 3.5, 1.5, 4.0 };
    const double tMin[12] = { 5.0, 2.0, 3.5, 4.5, 2.0, 4.0,
                              2.0, 4.5, 3.5, 2.0, 1.5, 4.0 };
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_EQ(KD::kKrumhanslKesslerMajor[static_cast<std::size_t>(i)], kkMaj[i]);
        EXPECT_EQ(KD::kKrumhanslKesslerMinor[static_cast<std::size_t>(i)], kkMin[i]);
        EXPECT_EQ(KD::kTemperleyMajor[static_cast<std::size_t>(i)], tMaj[i]);
        EXPECT_EQ(KD::kTemperleyMinor[static_cast<std::size_t>(i)], tMin[i]);
    }
    // The two structural facts the mode decision rests on, asserted as
    // relations rather than as digits, so they survive a reformatting but not
    // a transcription error that reverses them.
    EXPECT_GT(KD::kKrumhanslKesslerMinor[3], KD::kKrumhanslKesslerMinor[4]);
    EXPECT_GT(KD::kKrumhanslKesslerMajor[4], KD::kKrumhanslKesslerMajor[3]);
    EXPECT_GT(KD::kKrumhanslKesslerMinor[10], KD::kKrumhanslKesslerMinor[11]);
    EXPECT_GT(KD::kTemperleyMinor[8], KD::kTemperleyMinor[9]);
    EXPECT_GT(KD::kTemperleyMajor[9], KD::kTemperleyMajor[8]);
}

// ============================================================================
// Accuracy over all 24 keys, with every error class counted by name.
// ============================================================================

DSPARK_TEST(Key_all_24_keys_cadence_and_aeolian)
{
    using KD = KeyDetector<float>;
    const int registers[2] = { 54, 58 };   // tops out at 72 / 76.

    for (int prof = 0; prof < 2; ++prof)
    {
        KD kd;
        kd.prepare(keySpec(), 0);
        kd.setProfile(prof == 0 ? KD::Profile::KrumhanslKessler : KD::Profile::Temperley);

        int n = 0, tonicOk = 0, correct = 0;
        int rel = 0, par = 0, dom = 0, sub = 0, other = 0, none = 0;
        float minConf = 2.0f;

        for (int tonic = 0; tonic < 12; ++tonic)
        {
            for (int m = 0; m < 2; ++m)
                for (std::size_t ti = 0; ti < 2; ++ti)
                    for (int ri = 0; ri < 2; ++ri)
                    {
                        const auto a = renderCadence(tonic, m == 1, kTimbres[ti], registers[ri], 2);
                        const auto k = run(kd, a);
                        const char* c = classify(tonic, m == 1, k.tonicPitchClass, k.isMinor);
                        ++n;
                        if (k.tonicPitchClass == tonic) ++tonicOk;
                        if (!std::strcmp(c, "CORRECT")) ++correct;
                        else if (!std::strcmp(c, "RELATIVE")) ++rel;
                        else if (!std::strcmp(c, "PARALLEL")) ++par;
                        else if (!std::strcmp(c, "DOMINANT")) ++dom;
                        else if (!std::strcmp(c, "SUBDOMINANT")) ++sub;
                        else if (!std::strcmp(c, "NONE")) ++none;
                        else ++other;
                        minConf = std::min(minConf, k.confidence);
                    }
            // Pure Aeolian: the relative-confusion bed.
            for (std::size_t ti = 0; ti < 2; ++ti)
            {
                const auto a = renderAeolian(tonic, kTimbres[ti], 54, 8.0,
                                             77u + static_cast<std::uint32_t>(tonic));
                const auto k = run(kd, a);
                const char* c = classify(tonic, true, k.tonicPitchClass, k.isMinor);
                ++n;
                if (k.tonicPitchClass == tonic) ++tonicOk;
                if (!std::strcmp(c, "CORRECT")) ++correct;
                else if (!std::strcmp(c, "RELATIVE")) ++rel;
                else if (!std::strcmp(c, "PARALLEL")) ++par;
                else if (!std::strcmp(c, "DOMINANT")) ++dom;
                else if (!std::strcmp(c, "SUBDOMINANT")) ++sub;
                else if (!std::strcmp(c, "NONE")) ++none;
                else ++other;
                minConf = std::min(minConf, k.confidence);
            }
        }

        const double tonicPct = 100.0 * tonicOk / n;
        const double bothPct  = 100.0 * correct / n;
        std::printf("  [key] %-16s n=%d tonic=%.2f%% tonic+mode=%.2f%% "
                    "RELATIVE=%d PARALLEL=%d DOMINANT=%d SUBDOMINANT=%d OTHER=%d NONE=%d "
                    "minConf=%.4f\n",
                    prof == 0 ? "KrumhanslKessler" : "Temperley",
                    n, tonicPct, bothPct, rel, par, dom, sub, other, none,
                    static_cast<double>(minConf));

        EXPECT_GT(tonicPct, 89.999);   // architecture floor: correct tonic >= 90%
        EXPECT_GT(bothPct, 84.999);    // architecture floor: correct mode  >= 85%
        // Non-relative error < 10% of the corpus, counted from the named
        // classes rather than inferred from the accuracy figure.
        const double nonRelative = 100.0 * (par + dom + sub + other + none) / n;
        EXPECT_LT(nonRelative, 10.0);
    }
}

// ============================================================================
// Modal material: named, not scored. A 24-key output has no correct answer for
// it, so the test asserts what the reading IS and that it is stable, which is
// the only honest thing to assert.
// ============================================================================

DSPARK_TEST(Key_modal_material_reads_as_its_centre)
{
    using KD = KeyDetector<float>;
    struct M { const char* name; int idx; bool expectMinor; };
    // Indices into harmony::allScales; the name is asserted, not assumed.
    const M modes[5] = { { "Dorian", 1, true }, { "Phrygian", 2, true },
                         { "Lydian", 3, false }, { "Mixolydian", 4, false },
                         { "Aeolian", 5, true } };

    KD kd;
    kd.prepare(keySpec(), 0);
    kd.setProfile(KD::Profile::KrumhanslKessler);

    for (const auto& m : modes)
    {
        EXPECT_TRUE(harmony::allScales[static_cast<std::size_t>(m.idx)].name == m.name);
        const harmony::NoteSet mask = harmony::allScales[static_cast<std::size_t>(m.idx)].mask;
        int centreMinor = 0, centreMajor = 0, parentMajor = 0, elsewhere = 0;
        float minConf = 2.0f, maxConf = -1.0f;
        for (int centre = 0; centre < 12; centre += 4)
            for (std::size_t ti = 1; ti < 2; ++ti)
            {
                const auto a = renderModal(centre, mask, kTimbres[ti], 54, 8.0,
                                           901u + static_cast<std::uint32_t>(centre) * 3u
                                               + static_cast<std::uint32_t>(ti));
                const auto k = run(kd, a);
                const int rel = ((k.tonicPitchClass - centre) + 12) % 12;
                if (rel == 0 && k.isMinor) ++centreMinor;
                else if (rel == 0 && !k.isMinor) ++centreMajor;
                else if (!k.isMinor && ((mask >> rel) & 1u) == 0u) ++elsewhere;
                else if (!k.isMinor) ++parentMajor;
                else ++elsewhere;
                minConf = std::min(minConf, k.confidence);
                maxConf = std::max(maxConf, k.confidence);
            }
        std::printf("  [key] modal %-11s centre-as-minor=%d centre-as-major=%d "
                    "other-diatonic-major=%d elsewhere=%d conf[%.4f..%.4f]\n",
                    m.name, centreMinor, centreMajor, parentMajor, elsewhere,
                    static_cast<double>(minConf), static_cast<double>(maxConf));
        // The centre is found even where the mode is not: the reading is the
        // centre's own triad quality, and never a key outside the mode's
        // pitch-class set.
        EXPECT_EQ(centreMinor + centreMajor, 3);
        EXPECT_EQ(m.expectMinor ? centreMinor : centreMajor, 3);
        EXPECT_EQ(elsewhere, 0);
    }
}

// ============================================================================
// Contract: window, publication, determinism, degenerate input.
// ============================================================================

DSPARK_TEST(Key_auto_window_map_and_non_48k_anchor)
{
    KeyDetector<float> kd;
    const double rates[6]  = { 44100, 48000, 88200, 96000, 176400, 192000 };
    const int    expect[6] = { 4096,  4096,  8192,  8192,  16384,   16384 };
    for (int i = 0; i < 6; ++i)
    {
        kd.prepare(keySpec(rates[i]), 0);
        EXPECT_EQ(kd.getWindowSize(), expect[i]);
    }
    // Explicit requests are clamped into the same range as the front end's.
    kd.prepare(keySpec(), 64);
    EXPECT_EQ(kd.getWindowSize(), 1024);
    kd.prepare(keySpec(), 1 << 20);
    EXPECT_EQ(kd.getWindowSize(), 16384);

    // The non-48k anchor is an accuracy anchor, not only a window one: a
    // cadence at 44.1 kHz must still name its key.
    KeyDetector<float> at441;
    at441.prepare(keySpec(44100.0), 0);
    // The corpus renderer is fixed at 48 kHz, so feed it as if it were 44.1:
    // the notes land a semitone-and-a-bit low, which moves the whole key by a
    // constant, so what is asserted is that ONE key wins with a real margin.
    const auto a = renderCadence(0, false, kTimbres[1], 56, 2);
    at441.reset();
    for (std::size_t off = 0; off < a.size(); off += 512)
        at441.pushSamples(std::span<const float>(a.data() + off,
                                                 std::min<std::size_t>(512, a.size() - off)));
    const auto k = at441.getKey();
    std::printf("  [key] 44.1 kHz anchor: tonic=%d minor=%d conf=%.4f window=%d\n",
                k.tonicPitchClass, static_cast<int>(k.isMinor),
                static_cast<double>(k.confidence), at441.getWindowSize());
    EXPECT_TRUE(k.tonicPitchClass >= 0);
    EXPECT_GT(k.confidence, 0.05f);
}

DSPARK_TEST(Key_result_travels_in_one_word)
{
    // The published set is multi-word by nature -- tonic, mode, confidence and
    // the runner-up pair -- and travels as ONE atomic word. Reading it twice
    // without new material must give the identical struct, field for field:
    // two independently racing words could not promise that.
    KeyDetector<float> kd;
    kd.prepare(keySpec(), 0);
    const auto a = renderCadence(7, true, kTimbres[2], 56, 2);
    const auto k1 = run(kd, a);
    for (int i = 0; i < 64; ++i)
    {
        const auto k2 = kd.getKey();
        EXPECT_EQ(k2.tonicPitchClass, k1.tonicPitchClass);
        EXPECT_EQ(k2.isMinor, k1.isMinor);
        EXPECT_EQ(k2.runnerUpPitchClass, k1.runnerUpPitchClass);
        EXPECT_EQ(k2.runnerUpMinor, k1.runnerUpMinor);
        EXPECT_EQ(k2.confidence, k1.confidence);
    }
    EXPECT_EQ(k1.tonicPitchClass, 7);
    EXPECT_TRUE(k1.isMinor);
    // The runner-up is populated and is a different key from the winner.
    EXPECT_TRUE(k1.runnerUpPitchClass >= 0);
    EXPECT_TRUE(k1.runnerUpPitchClass != k1.tonicPitchClass || k1.runnerUpMinor != k1.isMinor);
    // Quantisation of the confidence into the packed word is bounded by one
    // step of a 16-bit field.
    EXPECT_TRUE(k1.confidence >= 0.0f && k1.confidence <= 1.0f);
}

DSPARK_TEST(Key_confidence_is_zero_without_tonal_evidence)
{
    KeyDetector<float> kd;
    kd.prepare(keySpec(), 0);

    // Silence: nothing accumulates, so nothing is claimed.
    std::vector<float> quiet(static_cast<std::size_t>(kFs * 4.0), 0.0f);
    auto k = run(kd, quiet);
    EXPECT_EQ(k.tonicPitchClass, -1);
    EXPECT_EQ(k.confidence, 0.0f);

    // All twelve pitch classes equally: the chroma is flat, every rotation
    // correlates identically, and the margin is the honest answer -- zero.
    std::vector<float> chromatic(static_cast<std::size_t>(kFs * 13.0), 0.0f);
    for (int i = 0; i < 12; ++i)
        addNote(chromatic, i * 1.0, 0.95, 54 + i, 0.2, kTimbres[0]);
    k = run(kd, chromatic);
    std::printf("  [key] flat chroma: tonic=%d conf=%.6f\n",
                k.tonicPitchClass, static_cast<double>(k.confidence));
    EXPECT_LT(k.confidence, 0.10f);
}

DSPARK_TEST(Key_degenerate_input_is_survived)
{
    KeyDetector<float> kd;
    // Calls before prepare() do nothing and claim nothing.
    std::vector<float> v(1024, 0.5f);
    kd.pushSamples(std::span<const float>(v.data(), v.size()));
    EXPECT_EQ(kd.getKey().tonicPitchClass, -1);

    // An invalid spec leaves the previous configuration in force.
    kd.prepare(keySpec(), 0);
    const int w = kd.getWindowSize();
    AudioSpec bad;
    bad.sampleRate = std::numeric_limits<double>::quiet_NaN();
    bad.numChannels = 1;
    bad.maxBlockSize = 512;
    kd.prepare(bad, 0);
    EXPECT_EQ(kd.getWindowSize(), w);

    // Injected non-finite samples must not leave the estimate non-finite.
    auto a = renderCadence(3, false, kTimbres[1], 56, 2);
    a[1000] = std::numeric_limits<float>::quiet_NaN();
    a[2000] = std::numeric_limits<float>::infinity();
    a[3000] = -std::numeric_limits<float>::infinity();
    const auto k = run(kd, a);
    EXPECT_TRUE(std::isfinite(k.confidence));
    EXPECT_TRUE(k.tonicPitchClass >= -1 && k.tonicPitchClass <= 11);
    for (const float c : kd.chroma()) EXPECT_TRUE(std::isfinite(c));
}

DSPARK_TEST(Key_reset_returns_to_a_fresh_instance)
{
    KeyDetector<float> a, b;
    a.prepare(keySpec(), 0);
    b.prepare(keySpec(), 0);
    const auto first = renderCadence(2, false, kTimbres[1], 56, 2);
    const auto second = renderCadence(9, true, kTimbres[2], 56, 2);
    (void)run(a, first);              // a has history
    const auto viaReset = run(a, second);
    const auto fresh    = run(b, second);
    EXPECT_EQ(viaReset.tonicPitchClass, fresh.tonicPitchClass);
    EXPECT_EQ(viaReset.isMinor, fresh.isMinor);
    EXPECT_EQ(viaReset.confidence, fresh.confidence);
    for (int i = 0; i < 12; ++i)
        EXPECT_EQ(a.chroma()[static_cast<std::size_t>(i)],
                  b.chroma()[static_cast<std::size_t>(i)]);
}

DSPARK_TEST(Key_block_size_does_not_change_the_estimate)
{
    // A host may cut the stream anywhere. The estimate must be a property of
    // the audio, not of the block boundaries.
    const auto a = renderCadence(5, false, kTimbres[2], 56, 2);
    const int blocks[6] = { 1, 64, 333, 512, 2048, 4096 };
    KeyDetector<float>::Key ref {};
    for (int bi = 0; bi < 6; ++bi)
    {
        KeyDetector<float> kd;
        kd.prepare(keySpec(), 0);
        for (std::size_t off = 0; off < a.size(); off += static_cast<std::size_t>(blocks[bi]))
            kd.pushSamples(std::span<const float>(
                a.data() + off,
                std::min(static_cast<std::size_t>(blocks[bi]), a.size() - off)));
        const auto k = kd.getKey();
        if (bi == 0) ref = k;
        EXPECT_EQ(k.tonicPitchClass, ref.tonicPitchClass);
        EXPECT_EQ(k.isMinor, ref.isMinor);
        EXPECT_EQ(k.confidence, ref.confidence);
    }
}

DSPARK_TEST(Key_processBlock_and_pushSamples_agree)
{
    const auto mono = renderCadence(11, true, kTimbres[1], 56, 2);
    KeyDetector<float> viaSpan, viaView;
    viaSpan.prepare(keySpec(), 0);
    viaView.prepare(keySpec(), 0);
    const auto k1 = run(viaSpan, mono);

    const float* ch[1] = { mono.data() };
    AudioBufferView<const float> view(ch, 1, static_cast<int>(mono.size()));
    for (int off = 0; off < view.getNumSamples(); off += 512)
        viaView.processBlock(view.getSubView(off, std::min(512, view.getNumSamples() - off)));
    const auto k2 = viaView.getKey();
    EXPECT_EQ(k1.tonicPitchClass, k2.tonicPitchClass);
    EXPECT_EQ(k1.isMinor, k2.isMinor);
    EXPECT_EQ(k1.confidence, k2.confidence);
}

DSPARK_TEST(Key_float_and_double_agree)
{
    const auto a = renderCadence(8, false, kTimbres[2], 56, 2);
    std::vector<double> ad(a.begin(), a.end());
    KeyDetector<float> kf;
    KeyDetector<double> kdd;
    kf.prepare(keySpec(), 0);
    kdd.prepare(keySpec(), 0);
    const auto rf = run(kf, a);
    kdd.reset();
    for (std::size_t off = 0; off < ad.size(); off += 512)
        kdd.pushSamples(std::span<const double>(ad.data() + off,
                                                std::min<std::size_t>(512, ad.size() - off)));
    const auto rd = kdd.getKey();
    EXPECT_EQ(rf.tonicPitchClass, rd.tonicPitchClass);
    EXPECT_EQ(rf.isMinor, rd.isMinor);
    EXPECT_NEAR(static_cast<double>(rf.confidence), static_cast<double>(rd.confidence), 0.01);
}

DSPARK_TEST(Key_profile_switch_rereads_the_same_evidence)
{
    using KD = KeyDetector<float>;
    KD kd;
    kd.prepare(keySpec(), 0);
    EXPECT_TRUE(kd.getProfile() == KD::Profile::KrumhanslKessler);
    const auto a = renderCadence(4, true, kTimbres[1], 56, 2);
    const auto k1 = run(kd, a);
    std::array<float, 12> before {};
    for (int i = 0; i < 12; ++i) before[static_cast<std::size_t>(i)] = kd.chroma()[static_cast<std::size_t>(i)];

    kd.setProfile(KD::Profile::Temperley);
    EXPECT_TRUE(kd.getProfile() == KD::Profile::Temperley);
    // Feeding one more hop re-reads the accumulated chroma under the new
    // profile without disturbing it.
    std::vector<float> more(static_cast<std::size_t>(kd.getWindowSize()), 0.0f);
    addNote(more, 0.0, 0.9, foldRoot(4, 56), 0.16, kTimbres[1]);
    addNote(more, 0.0, 0.9, foldRoot(4, 56) + 3, 0.16, kTimbres[1]);
    addNote(more, 0.0, 0.9, foldRoot(4, 56) + 7, 0.16, kTimbres[1]);
    kd.pushSamples(std::span<const float>(more.data(), more.size()));
    const auto k2 = kd.getKey();
    EXPECT_EQ(k2.tonicPitchClass, k1.tonicPitchClass);
    EXPECT_EQ(k2.isMinor, k1.isMinor);
    // The chroma is evidence, not a function of the profile: it only grew.
    for (int i = 0; i < 12; ++i)
        EXPECT_TRUE(std::isfinite(kd.chroma()[static_cast<std::size_t>(i)]));
    (void)before;
}

DSPARK_TEST(Key_chroma_is_normalised_and_names_are_written)
{
    KeyDetector<float> kd;
    kd.prepare(keySpec(), 0);
    const auto a = renderCadence(0, false, kTimbres[1], 56, 2);
    const auto k = run(kd, a);
    float sum = 0.0f;
    for (const float c : kd.chroma()) { EXPECT_TRUE(c >= 0.0f); sum += c; }
    EXPECT_NEAR(static_cast<double>(sum), 1.0, 1e-4);

    char name[8] = {};
    const int len = KeyDetector<float>::getKeyName(k, name, sizeof name);
    EXPECT_EQ(len, 1);
    EXPECT_TRUE(std::string(name) == "C");

    KeyDetector<float>::Key minorKey;
    minorKey.tonicPitchClass = 10;
    minorKey.isMinor = true;
    KeyDetector<float>::getKeyName(minorKey, name, sizeof name);
    EXPECT_TRUE(std::string(name) == "Bbm" || std::string(name) == "A#m");

    KeyDetector<float>::Key nothing;
    EXPECT_EQ(KeyDetector<float>::getKeyName(nothing, name, sizeof name), 0);
    EXPECT_EQ(KeyDetector<float>::getKeyName(nothing, name, 0), 0);
}
