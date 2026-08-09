// DSPark - LoudnessNormalizer acceptance suite.
//
// Two things are checked here and they are not the same kind of claim.
//
// The MEASUREMENT is normative: ITU-R BS.1770-5 fixes the K-weighting, the
// 400 ms gating block with 75% overlap, the -70 LKFS absolute and -10 LU
// relative thresholds, and the 4x true-peak interpolator. Those are verified
// against the EBU Tech 3341-2023 Table 1 signals, synthesised here from the
// printed descriptions of the test signals rather than downloaded, so the
// check runs on any machine with no external material.
//
// The NORMALIZATION is this class's own: hit the target, respect the ceiling,
// and say honestly which one gave way when they could not both be had.
//
// Every case prints its numbers. Sweeps report their WORST point with its
// sign, not their endpoints.

#include "dspark_test.h"
#include "TestSignals.h"
#include "../Analysis/LoudnessNormalizer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

constexpr double kFs = 48000.0;

struct Rng
{
    std::uint32_t s = 20120809u;
    float gauss()
    {
        // Sum of twelve uniforms minus six: deterministic, platform-identical,
        // and close enough to Gaussian for a noise bed.
        float acc = 0.0f;
        for (int i = 0; i < 12; ++i)
        {
            s = s * 1664525u + 1013904223u;
            acc += static_cast<float>(s >> 8) / 16777216.0f;
        }
        return acc - 6.0f;
    }
};

void pinkProgramme(AudioBuffer<float>& b, double sec, float scale)
{
    const int n = static_cast<int>(sec * kFs);
    b.resize(2, n);
    Rng rng;
    float p0 = 0, p1 = 0, p2 = 0;
    for (int i = 0; i < n; ++i)
    {
        const float w = rng.gauss();
        p0 = 0.99765f * p0 + w * 0.0990460f;
        p1 = 0.96300f * p1 + w * 0.2965164f;
        p2 = 0.57000f * p2 + w * 1.0526913f;
        const float p = (p0 + p1 + p2 + w * 0.1848f) * 0.06f * scale;
        const auto env = static_cast<float>(
            0.5 + 0.45 * std::sin(twoPi<double> * (i / kFs) / 7.0));
        b.getChannel(0)[i] = p * env;
        b.getChannel(1)[i] = p * env * 0.95f;
    }
}

// High crest factor: the bed on which the ceiling and the target genuinely
// compete, so a criterion written on it is not decided in advance.
void percussiveProgramme(AudioBuffer<float>& b, double sec, float scale)
{
    const int n = static_cast<int>(sec * kFs);
    b.resize(2, n);
    Rng rng;
    for (int i = 0; i < n; ++i)
    {
        const auto v = static_cast<float>(
            0.02 * std::sin(twoPi<double> * 220.0 * (i / kFs))) * scale;
        b.getChannel(0)[i] = v;
        b.getChannel(1)[i] = v * 0.9f;
    }
    for (double t0 = 0.05; t0 < sec; t0 += 0.25)
    {
        const int i0 = static_cast<int>(t0 * kFs);
        for (int k = 0; k < static_cast<int>(0.08 * kFs) && i0 + k < n; ++k)
        {
            const float e = std::exp(-static_cast<float>(k) / (0.004f * static_cast<float>(kFs)));
            const float s = 0.5f * e * rng.gauss() * scale;
            b.getChannel(0)[i0 + k] += s;
            b.getChannel(1)[i0 + k] += s * 0.98f;
        }
    }
}

// A tone at fs/4 with 45 degrees of phase: every SAMPLE sits at 0.707 of the
// waveform peak, so the inter-sample peak is 3 dB above anything a sample
// peak meter can see. A ceiling honoured on samples alone fails here.
void interSampleProgramme(AudioBuffer<float>& b, double sec, float scale)
{
    const int n = static_cast<int>(sec * kFs);
    b.resize(2, n);
    const double w = twoPi<double> * (kFs / 4.0) / kFs;
    const double p0 = 45.0 * pi<double> / 180.0;
    for (int i = 0; i < n; ++i)
    {
        const auto v = static_cast<float>(scale * 0.5 * std::sin(w * i + p0));
        b.getChannel(0)[i] = v;
        b.getChannel(1)[i] = v;
    }
}

float maxAbs(const AudioBuffer<float>& b)
{
    float m = 0;
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
        for (int i = 0; i < b.getNumSamples(); ++i)
            m = std::max(m, std::abs(b.getChannel(ch)[i]));
    return m;
}

// EBU Tech 3341 signal builders.
void toneSeq(AudioBuffer<double>& b, double fs, const double (*segs)[2], int nSeg)
{
    double total = 0.0;
    for (int i = 0; i < nSeg; ++i) total += segs[i][0];
    const int n = static_cast<int>(total * fs + 0.5);
    b.resize(2, n);
    int i = 0;
    double phase = 0.0;
    const double w = twoPi<double> * 1000.0 / fs;
    for (int s = 0; s < nSeg; ++s)
    {
        const int len = static_cast<int>(segs[s][0] * fs + 0.5);
        const double a = std::pow(10.0, segs[s][1] / 20.0);
        for (int k = 0; k < len && i < n; ++k, ++i)
        {
            const double v = a * std::sin(phase);
            phase += w;
            b.getChannel(0)[i] = v;
            b.getChannel(1)[i] = v;
        }
    }
}

void tpTone(AudioBuffer<double>& b, double fs, double divisor, double amp,
            double phaseDeg, double sec)
{
    const int n = static_cast<int>(sec * fs + 0.5);
    b.resize(2, n);
    const double w = twoPi<double> * (fs / divisor) / fs;
    const double p0 = phaseDeg * pi<double> / 180.0;
    const int fade = static_cast<int>(0.010 * fs);
    for (int i = 0; i < n; ++i)
    {
        double env = 1.0;
        if (i < fade) env = static_cast<double>(i) / fade;
        if (n - 1 - i < fade) env = std::min(env, static_cast<double>(n - 1 - i) / fade);
        const double v = amp * env * std::sin(w * i + p0);
        b.getChannel(0)[i] = v;
        b.getChannel(1)[i] = v;
    }
}

} // namespace

// ============================================================================
// The measurement the normalizer stands on, re-verified rather than assumed.
// ============================================================================

DSPARK_TEST(LoudnessNorm_EBU_3341_integrated_vectors)
{
    // EBU Tech 3341-2023 Table 1, cases 1 to 5. Stereo 1000 Hz sine applied in
    // phase, the listed per-channel PEAK levels and durations, integrated
    // loudness to +/-0.1 LU. Run at 48 kHz and at 44.1 kHz, because the
    // K-weighting coefficients are printed for 48 kHz only and every other
    // rate is a derivation.
    for (double fs : { 48000.0, 44100.0 })
    {
        const double c1[][2] = { { 20, -23 } };
        const double c2[][2] = { { 20, -33 } };
        const double c3[][2] = { { 10, -36 }, { 60, -23 }, { 10, -36 } };
        const double c4[][2] = { { 10, -72 }, { 10, -36 }, { 60, -23 },
                                 { 10, -36 }, { 10, -72 } };
        const double c5[][2] = { { 20, -26 }, { 20.1, -20 }, { 20, -26 } };
        struct C { const char* n; const double (*s)[2]; int k; double e; };
        const C cases[5] = { { "3341-1", c1, 1, -23.0 }, { "3341-2", c2, 1, -33.0 },
                             { "3341-3", c3, 3, -23.0 }, { "3341-4", c4, 5, -23.0 },
                             { "3341-5", c5, 3, -23.0 } };
        for (const auto& c : cases)
        {
            AudioBuffer<double> b;
            toneSeq(b, fs, c.s, c.k);
            LoudnessMeter<double> m;
            m.prepare(fs, 2);
            m.process(b.getChannel(0), b.getChannel(1), b.getNumSamples());
            const double got = m.getIntegratedLUFS();
            std::printf("  [ln] %s @ %.0f Hz: I = %.4f LUFS (expect %.1f, err %+.4f LU)\n",
                        c.n, fs, got, c.e, got - c.e);
            EXPECT_NEAR(got, c.e, 0.1);
        }
    }
}

DSPARK_TEST(LoudnessNorm_EBU_3341_true_peak_vectors)
{
    // EBU Tech 3341-2023 Table 1, cases 15 to 19, tolerance +0.2/-0.4 dB.
    for (double fs : { 48000.0, 44100.0 })
    {
        struct TP { const char* n; double div, amp, phase, e; };
        const TP tps[5] = {
            { "3341-15", 4.0, 0.50,  0.0, -6.0 }, { "3341-16", 4.0, 0.50, 45.0, -6.0 },
            { "3341-17", 6.0, 0.50, 60.0, -6.0 }, { "3341-18", 8.0, 0.50, 67.5, -6.0 },
            { "3341-19", 4.0, 1.41, 45.0,  3.0 },
        };
        for (const auto& t : tps)
        {
            AudioBuffer<double> b;
            tpTone(b, fs, t.div, t.amp, t.phase, 2.0);
            LoudnessMeter<double> m;
            m.prepare(fs, 2);
            m.process(b.getChannel(0), b.getChannel(1), b.getNumSamples());
            const double got = m.getTruePeakDb();
            std::printf("  [ln] %s @ %.0f Hz: TP = %.4f dBTP (expect %.1f, err %+.4f dB)\n",
                        t.n, fs, got, t.e, got - t.e);
            EXPECT_TRUE(got - t.e <= 0.2);
            EXPECT_TRUE(got - t.e >= -0.4);
        }
    }
}

// ============================================================================
// Acceptance: the target, over a declared grid, at its worst point.
// ============================================================================

DSPARK_TEST(LoudnessNorm_hits_the_target_when_the_ceiling_allows)
{
    const double targets[4]  = { -23.0, -18.0, -16.0, -14.0 };
    const double ceilings[3] = { -2.0, -1.0, -0.5 };
    double worst = 0.0;
    double worstT = 0.0, worstC = 0.0;
    int counted = 0;

    for (int quiet = 0; quiet < 2; ++quiet)
        for (double tgt : targets)
            for (double ceil : ceilings)
            {
                AudioBuffer<float> b;
                pinkProgramme(b, 20.0, quiet ? 0.05f : 1.0f);
                LoudnessNormalizer<float> ln;
                ln.setTargetLUFS(static_cast<float>(tgt));
                ln.setTruePeakCeilingDb(static_cast<float>(ceil));
                const auto r = ln.normalize(b, kFs);
                if (r.limiterPasses != 0) continue;   // ceiling bound; see the next case
                const double err = static_cast<double>(r.outLUFS) - tgt;
                ++counted;
                if (std::abs(err) > std::abs(worst)) { worst = err; worstT = tgt; worstC = ceil; }
                EXPECT_TRUE(r.targetReached);
                EXPECT_NEAR(static_cast<double>(r.outLUFS), tgt, 0.1);
                // The gain the class says it applied must be the gain it applied.
                EXPECT_NEAR(static_cast<double>(r.appliedGainDb),
                            tgt - static_cast<double>(r.measuredLUFS), 0.01);
            }
    std::printf("  [ln] target grid: %d unbound points, WORST signed error %+.4f LU "
                "at target %.1f LUFS / ceiling %.1f dBTP\n", counted, worst, worstT, worstC);
    EXPECT_GT(counted, 10);
    EXPECT_LT(std::abs(worst), 0.1);
}

// ============================================================================
// Acceptance: the ceiling, over the same grid, at its worst point. The bed
// includes material the ceiling can genuinely fail on -- with the ceiling
// stage inert the same grid runs 17.04 dB over it -- so a pass here is a
// measurement and not a property of the material.
// ============================================================================

DSPARK_TEST(LoudnessNorm_true_peak_never_exceeds_the_ceiling)
{
    const double targets[4]  = { -23.0, -16.0, -14.0, -11.0 };
    const double ceilings[3] = { -3.0, -1.0, -0.5 };
    double worstOver = -1e9;
    double worstT = 0.0, worstC = 0.0;
    float worstSample = 0.0f;
    int engaged = 0;

    for (int bed = 0; bed < 3; ++bed)
        for (double tgt : targets)
            for (double ceil : ceilings)
            {
                AudioBuffer<float> b;
                if (bed == 0) pinkProgramme(b, 12.0, 1.0f);
                else if (bed == 1) percussiveProgramme(b, 12.0, 1.0f);
                else interSampleProgramme(b, 12.0, 1.0f);

                LoudnessNormalizer<float> ln;
                ln.setTargetLUFS(static_cast<float>(tgt));
                ln.setTruePeakCeilingDb(static_cast<float>(ceil));
                const auto r = ln.normalize(b, kFs);
                if (r.limiterPasses > 0) ++engaged;

                const double over = static_cast<double>(r.outTruePeakDb) - ceil;
                if (over > worstOver) { worstOver = over; worstT = tgt; worstC = ceil; }
                worstSample = std::max(worstSample, maxAbs(b));

                EXPECT_TRUE(over <= 0.0);
                // No sample over 0 dBFS follows from the ceiling being at or
                // below it, but it is the property a caller actually cares
                // about, so it is asserted and not inferred.
                EXPECT_TRUE(maxAbs(b) <= 1.0f);
                // The reported true peak must be the buffer's true peak.
                LoudnessNormalizer<float> probe;
                EXPECT_NEAR(static_cast<double>(probe.measureTruePeakDb(b)),
                            static_cast<double>(r.outTruePeakDb), 1e-4);
            }
    std::printf("  [ln] ceiling grid: %d of 36 points engaged the limiter, "
                "WORST true peak vs ceiling %+.4f dB at target %.1f / ceiling %.1f, "
                "worst sample %.6f (%.4f dBFS)\n",
                engaged, worstOver, worstT, worstC, static_cast<double>(worstSample),
                20.0 * std::log10(std::max(1e-12f, worstSample)));
    EXPECT_GT(engaged, 8);          // the bed must actually exercise the ceiling
    EXPECT_TRUE(worstOver <= 0.0);
}

// ============================================================================
// The ceiling holds across the whole parameter grid, not only at the defaults:
// the envelope shape is a preference, the ceiling is a contract.
// ============================================================================

DSPARK_TEST(LoudnessNorm_ceiling_holds_across_the_envelope_grid)
{
    const double las[4]  = { 0.05, 0.5, 2.0, 5.0 };
    const double rels[3] = { 10.0, 100.0, 200.0 };
    double worstOver = -1e9, worstLoss = -1e9;
    double lossAtShortest = 0.0, lossAtLongest = 0.0;

    for (double la : las)
        for (double rl : rels)
        {
            AudioBuffer<float> b;
            percussiveProgramme(b, 12.0, 1.0f);
            LoudnessNormalizer<float> ln;
            ln.setTargetLUFS(-14.0f);
            ln.setTruePeakCeilingDb(-1.0f);
            ln.setLookaheadMs(static_cast<float>(la));
            ln.setReleaseMs(static_cast<float>(rl));
            const auto r = ln.normalize(b, kFs);
            const double over = static_cast<double>(r.outTruePeakDb) + 1.0;
            const double loss = -14.0 - static_cast<double>(r.outLUFS);
            worstOver = std::max(worstOver, over);
            worstLoss = std::max(worstLoss, loss);
            if (la == 0.5 && rl == 10.0) lossAtShortest = loss;
            if (la == 0.5 && rl == 200.0) lossAtLongest = loss;
            EXPECT_TRUE(over <= 0.0);
        }
    std::printf("  [ln] envelope grid 4x3: WORST over %+.4f dB, WORST loudness "
                "shortfall %+.4f LU (short release %.4f, long release %.4f)\n",
                worstOver, worstLoss, lossAtShortest, lossAtLongest);
    EXPECT_TRUE(worstOver <= 0.0);
    // Setters reject what they cannot use and clamp what they can.
    LoudnessNormalizer<float> ln;
    ln.setLookaheadMs(1.5f);
    ln.setLookaheadMs(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(ln.getLookaheadMs(), 1.5f);
    ln.setLookaheadMs(1000.0f);
    EXPECT_EQ(ln.getLookaheadMs(), 50.0f);
    ln.setReleaseMs(0.0f);
    EXPECT_EQ(ln.getReleaseMs(), 1.0f);
    ln.setTruePeakCeilingDb(5.0f);
    EXPECT_EQ(ln.getTruePeakCeilingDb(), 0.0f);
    ln.setTruePeakCeilingDb(std::numeric_limits<float>::infinity());
    EXPECT_EQ(ln.getTruePeakCeilingDb(), 0.0f);
    ln.setTargetLUFS(-23.0f);
    ln.setTargetLUFS(std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(ln.getTargetLUFS(), -23.0f);
}

// ============================================================================
// Where the two constraints cannot both be had, the ceiling wins and the class
// says so instead of reporting a target it did not reach.
// ============================================================================

DSPARK_TEST(LoudnessNorm_reports_the_target_it_could_not_reach)
{
    AudioBuffer<float> b;
    percussiveProgramme(b, 12.0, 1.0f);
    LoudnessNormalizer<float> ln;
    ln.setTargetLUFS(-9.0f);            // unreachable under a -1 dBTP ceiling here
    ln.setTruePeakCeilingDb(-1.0f);
    const auto r = ln.normalize(b, kFs);
    std::printf("  [ln] unreachable target: measured %.3f, applied %+.3f dB, "
                "out %.3f LUFS, TP %.3f dBTP, passes %d, targetReached %d\n",
                static_cast<double>(r.measuredLUFS), static_cast<double>(r.appliedGainDb),
                static_cast<double>(r.outLUFS), static_cast<double>(r.outTruePeakDb),
                r.limiterPasses, static_cast<int>(r.targetReached));
    EXPECT_FALSE(r.targetReached);
    EXPECT_LT(static_cast<double>(r.outLUFS), -9.0);
    EXPECT_TRUE(static_cast<double>(r.outTruePeakDb) <= -1.0);
    EXPECT_GT(r.limiterPasses, 0);
}

DSPARK_TEST(LoudnessNorm_defaults_are_the_broadcast_ones)
{
    // -23.0 LUFS is the EBU R 128 Programme Loudness Level and -1 dBTP its
    // maximum permitted true-peak level. Both are the defaults, and a change
    // to either is a change to what this class means out of the box.
    LoudnessNormalizer<float> ln;
    EXPECT_EQ(ln.getTargetLUFS(), -23.0f);
    EXPECT_EQ(ln.getTruePeakCeilingDb(), -1.0f);
}

DSPARK_TEST(LoudnessNorm_degenerate_input_is_survived)
{
    LoudnessNormalizer<float> ln;

    // Empty buffer, and a rate that is not a rate: no crash, nothing claimed.
    AudioBuffer<float> empty;
    auto r = ln.normalize(empty, kFs);
    EXPECT_EQ(r.appliedGainDb, 0.0f);
    AudioBuffer<float> b;
    pinkProgramme(b, 2.0, 1.0f);
    const float before = b.getChannel(0)[1000];
    r = ln.normalize(b, std::numeric_limits<double>::quiet_NaN());
    EXPECT_EQ(b.getChannel(0)[1000], before);
    r = ln.normalize(b, -1.0);
    EXPECT_EQ(b.getChannel(0)[1000], before);

    // Silence has no gain that makes it -23 LUFS. It is returned untouched
    // rather than amplified until its noise floor measures right.
    AudioBuffer<float> quiet;
    quiet.resize(2, static_cast<int>(kFs * 3.0));
    r = ln.normalize(quiet, kFs);
    EXPECT_EQ(r.appliedGainDb, 0.0f);
    EXPECT_EQ(maxAbs(quiet), 0.0f);

    // Mono is measured as one channel and normalized like any other.
    AudioBuffer<float> mono;
    mono.resize(1, static_cast<int>(kFs * 6.0));
    for (int i = 0; i < mono.getNumSamples(); ++i)
        mono.getChannel(0)[i] = 0.1f * static_cast<float>(std::sin(twoPi<double> * 997.0 * (i / kFs)));
    r = ln.normalize(mono, kFs);
    std::printf("  [ln] mono 997 Hz: measured %.3f -> out %.3f LUFS\n",
                static_cast<double>(r.measuredLUFS), static_cast<double>(r.outLUFS));
    EXPECT_NEAR(static_cast<double>(r.outLUFS), -23.0, 0.1);

    // Injected non-finite samples must not turn the whole programme into NaN.
    AudioBuffer<float> nasty;
    pinkProgramme(nasty, 6.0, 1.0f);
    nasty.getChannel(0)[5000] = std::numeric_limits<float>::quiet_NaN();
    nasty.getChannel(1)[9000] = std::numeric_limits<float>::infinity();
    r = ln.normalize(nasty, kFs);
    EXPECT_TRUE(std::isfinite(r.outTruePeakDb));
    int finite = 0;
    for (int i = 0; i < nasty.getNumSamples(); ++i)
        if (std::isfinite(nasty.getChannel(0)[i])) ++finite;
    EXPECT_GT(finite, nasty.getNumSamples() - 10);
}

DSPARK_TEST(LoudnessNorm_is_deterministic_and_idempotent)
{
    AudioBuffer<float> a, b;
    pinkProgramme(a, 8.0, 1.0f);
    pinkProgramme(b, 8.0, 1.0f);
    LoudnessNormalizer<float> l1, l2;
    const auto r1 = l1.normalize(a, kFs);
    const auto r2 = l2.normalize(b, kFs);
    EXPECT_EQ(r1.outLUFS, r2.outLUFS);
    for (int i = 0; i < a.getNumSamples(); i += 997)
        EXPECT_EQ(a.getChannel(0)[i], b.getChannel(0)[i]);

    // Normalizing an already-normalized programme must not move it again.
    const auto r3 = l1.normalize(a, kFs);
    std::printf("  [ln] second pass: applied %+.5f dB, out %.4f LUFS\n",
                static_cast<double>(r3.appliedGainDb), static_cast<double>(r3.outLUFS));
    EXPECT_LT(std::abs(static_cast<double>(r3.appliedGainDb)), 0.05);
    EXPECT_NEAR(static_cast<double>(r3.outLUFS), -23.0, 0.1);
}

DSPARK_TEST(LoudnessNorm_float_and_double_agree)
{
    AudioBuffer<float> bf;
    pinkProgramme(bf, 8.0, 1.0f);
    AudioBuffer<double> bd;
    bd.resize(2, bf.getNumSamples());
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < bf.getNumSamples(); ++i)
            bd.getChannel(ch)[i] = bf.getChannel(ch)[i];

    LoudnessNormalizer<float> lf;
    LoudnessNormalizer<double> ld;
    const auto rf = lf.normalize(bf, kFs);
    const auto rd = ld.normalize(bd, kFs);
    EXPECT_NEAR(static_cast<double>(rf.outLUFS), rd.outLUFS, 0.01);
    EXPECT_NEAR(static_cast<double>(rf.appliedGainDb), rd.appliedGainDb, 0.01);
}
