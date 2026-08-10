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
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
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

template <typename T, int MaxChannels>
T maxAbs(const AudioBuffer<T, MaxChannels>& b)
{
    T m = T(0);
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
        for (int i = 0; i < b.getNumSamples(); ++i)
            m = std::max(m, std::abs(b.getChannel(ch)[i]));
    return m;
}

template <typename T, int MaxChannels>
T truePeakLinear(const AudioBuffer<T, MaxChannels>& b, bool flushTail)
{
    TruePeakDetector<T, MaxChannels> detector;
    T peak = T(0);
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
    {
        for (int i = 0; i < b.getNumSamples(); ++i)
            peak = std::max(peak, detector.processSample(b.getChannel(ch)[i], ch));
        if (flushTail)
            for (int i = 0; i < TruePeakDetector<T, MaxChannels>::getTaps() - 1; ++i)
                peak = std::max(peak, detector.processSample(T(0), ch));
    }
    return peak;
}

template <typename T, int MaxChannels>
std::vector<T> snapshot(const AudioBuffer<T, MaxChannels>& b)
{
    std::vector<T> result;
    result.reserve(static_cast<std::size_t>(b.getNumChannels())
                   * static_cast<std::size_t>(b.getNumSamples()));
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
        result.insert(result.end(), b.getChannel(ch),
                      b.getChannel(ch) + b.getNumSamples());
    return result;
}

template <typename T, int MaxChannels>
bool bitwiseEqual(const AudioBuffer<T, MaxChannels>& b, const std::vector<T>& reference)
{
    std::size_t offset = 0;
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
    {
        const auto bytes = static_cast<std::size_t>(b.getNumSamples()) * sizeof(T);
        if (std::memcmp(b.getChannel(ch), reference.data() + offset, bytes) != 0)
            return false;
        offset += static_cast<std::size_t>(b.getNumSamples());
    }
    return offset == reference.size();
}

template <typename T>
bool finiteResult(const typename LoudnessNormalizer<T>::Result& result)
{
    return std::isfinite(result.measuredLUFS)
        && std::isfinite(result.requestedGainDb)
        && std::isfinite(result.appliedGainDb)
        && std::isfinite(result.outLUFS)
        && std::isfinite(result.outTruePeakDb);
}

template <typename T, int MaxChannels>
void verifyNonFiniteMatrix(int runtimeChannels)
{
    using Normalizer = LoudnessNormalizer<T>;
    const std::array<T, 3> poisons {
        std::numeric_limits<T>::quiet_NaN(),
        std::numeric_limits<T>::infinity(),
        -std::numeric_limits<T>::infinity(),
    };
    constexpr int numSamples = 31;
    const int positions[3] = { 0, numSamples / 2, numSamples - 1 };
    for (int channel = 0; channel < runtimeChannels; ++channel)
        for (const int position : positions)
            for (const T poison : poisons)
            {
                AudioBuffer<T, MaxChannels> b;
                b.resize(runtimeChannels, numSamples);
                for (int ch = 0; ch < runtimeChannels; ++ch)
                    for (int i = 0; i < numSamples; ++i)
                        b.getChannel(ch)[i] = static_cast<T>(
                            0.01 * (1 + ch) + 0.0001 * i);
                b.getChannel(channel)[position] = poison;
                const auto before = snapshot(b);
                Normalizer normalizer;
                const auto result = normalizer.normalize(b, kFs);
                EXPECT_TRUE(result.status == Normalizer::Status::NonFiniteInput);
                EXPECT_TRUE(bitwiseEqual(b, before));
                EXPECT_TRUE(finiteResult<T>(result));
                EXPECT_FALSE(result.targetReached);
                EXPECT_FALSE(result.ceilingLimited);
                EXPECT_EQ(result.requestedGainDb, T(0));
                EXPECT_EQ(result.appliedGainDb, T(0));
            }
}

template <typename T, int MaxChannels>
void verifyAllChannelCeiling(int runtimeChannels)
{
    using Normalizer = LoudnessNormalizer<T>;
    const int numSamples = static_cast<int>(0.6 * kFs);
    std::array<int, 2> positions { 0, runtimeChannels - 1 };
    const int cases = runtimeChannels == 1 ? 1 : 2;
    for (int caseIndex = 0; caseIndex < cases; ++caseIndex)
    {
        const int decisiveChannel = positions[static_cast<std::size_t>(caseIndex)];
        AudioBuffer<T, MaxChannels> b;
        b.resize(runtimeChannels, numSamples);
        for (int i = 0; i < numSamples; ++i)
        {
            const T tone = static_cast<T>(
                0.01 * std::sin(twoPi<double> * 997.0 * i / kFs));
            b.getChannel(0)[i] = tone;
            if (runtimeChannels > 1) b.getChannel(1)[i] = tone;
        }
        b.getChannel(decisiveChannel)[numSamples / 2] = T(0.8);

        Normalizer normalizer;
        normalizer.setTargetLUFS(T(0));
        const auto result = normalizer.normalize(b, kFs);
        const T oraclePeak = truePeakLinear(b, true);
        const double oracleDb = 20.0 * std::log10(static_cast<double>(oraclePeak));
        EXPECT_TRUE(result.status == Normalizer::Status::Success);
        EXPECT_TRUE(result.ceilingLimited);
        EXPECT_TRUE(oracleDb <= -1.0);
        EXPECT_NEAR(static_cast<double>(result.outTruePeakDb), oracleDb, 0.001);
        EXPECT_TRUE(maxAbs(b) <= T(1));
        if (runtimeChannels >= 17)
            std::printf("  [ln] all-channel width=%d decisive=%d TP=%+.6f dBTP\n",
                        runtimeChannels, decisiveChannel, oracleDb);
    }
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
                EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::Success);
                if (r.ceilingLimited) continue;   // ceiling bound; see the next case
                const double err = static_cast<double>(r.outLUFS) - tgt;
                ++counted;
                if (std::abs(err) > std::abs(worst)) { worst = err; worstT = tgt; worstC = ceil; }
                EXPECT_TRUE(r.targetReached);
                EXPECT_NEAR(static_cast<double>(r.outLUFS), tgt, 0.1);
                // The gain the class says it applied must be the gain it applied.
                EXPECT_NEAR(static_cast<double>(r.appliedGainDb),
                            tgt - static_cast<double>(r.measuredLUFS), 0.01);
                EXPECT_NEAR(static_cast<double>(r.requestedGainDb),
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
                EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::Success);
                if (r.ceilingLimited) ++engaged;

                const double over = static_cast<double>(r.outTruePeakDb) - ceil;
                if (over > worstOver) { worstOver = over; worstT = tgt; worstC = ceil; }
                worstSample = std::max(worstSample, maxAbs(b));

                EXPECT_TRUE(over <= 0.0);
                // No sample over 0 dBFS follows from the ceiling being at or
                // below it, but it is the property a caller actually cares
                // about, so it is asserted and not inferred.
                EXPECT_TRUE(maxAbs(b) <= 1.0f);
                // The reported true peak must be the buffer's true peak.
                const double oracleDb = 20.0 * std::log10(
                    static_cast<double>(truePeakLinear(b, true)));
                EXPECT_NEAR(oracleDb, static_cast<double>(r.outTruePeakDb), 1e-3);
            }
    std::printf("  [ln] ceiling grid: %d of 36 points selected ceiling gain, "
                "WORST true peak vs ceiling %+.4f dB at target %.1f / ceiling %.1f, "
                "worst sample %.6f (%.4f dBFS)\n",
                engaged, worstOver, worstT, worstC, static_cast<double>(worstSample),
                20.0 * std::log10(std::max(1e-12f, worstSample)));
    EXPECT_GT(engaged, 8);          // the bed must actually exercise the ceiling
    EXPECT_TRUE(worstOver <= 0.0);
}

// ============================================================================
// The successful path is one rounded constant multiplication at every sample.
// ============================================================================

DSPARK_TEST(LoudnessNorm_is_one_transparent_constant_gain)
{
    AudioBuffer<float> b;
    percussiveProgramme(b, 12.0, 0.4f);
    const auto before = snapshot(b);
    LoudnessNormalizer<float> ln;
    ln.setTargetLUFS(0.0f);
    const auto r = ln.normalize(b, kFs);
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::Success);
    EXPECT_TRUE(r.ceilingLimited);

    const double gain = std::pow(10.0, static_cast<double>(r.appliedGainDb) / 20.0);
    double worstResidual = 0.0;
    double maxReference = 0.0;
    std::size_t offset = 0;
    for (int ch = 0; ch < b.getNumChannels(); ++ch)
        for (int i = 0; i < b.getNumSamples(); ++i, ++offset)
        {
            const float reference = static_cast<float>(
                static_cast<double>(before[offset]) * gain);
            worstResidual = std::max(worstResidual,
                std::abs(static_cast<double>(b.getChannel(ch)[i])
                         - static_cast<double>(reference)));
            maxReference = std::max(maxReference,
                                    std::abs(static_cast<double>(reference)));
        }
    const double bound = 8.0 * std::numeric_limits<float>::epsilon()
                       * std::max(1.0, maxReference);
    std::printf("  [ln] constant gain: requested %+.5f dB, applied %+.5f dB, "
                "worst rounded-oracle residual %.9g (bound %.9g)\n",
                static_cast<double>(r.requestedGainDb),
                static_cast<double>(r.appliedGainDb), worstResidual, bound);
    EXPECT_TRUE(worstResidual <= bound);

    // Controls reject non-finite requests and clamp the finite ceiling range.
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
    std::printf("  [ln] unreachable target: measured %.3f, requested %+.3f dB, "
                "applied %+.3f dB, out %.3f LUFS, TP %.3f dBTP, "
                "ceilingLimited %d, targetReached %d\n",
                static_cast<double>(r.measuredLUFS),
                static_cast<double>(r.requestedGainDb),
                static_cast<double>(r.appliedGainDb),
                static_cast<double>(r.outLUFS), static_cast<double>(r.outTruePeakDb),
                static_cast<int>(r.ceilingLimited), static_cast<int>(r.targetReached));
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::Success);
    EXPECT_FALSE(r.targetReached);
    EXPECT_TRUE(r.ceilingLimited);
    EXPECT_LT(static_cast<double>(r.outLUFS), -9.0);
    EXPECT_TRUE(static_cast<double>(r.outTruePeakDb) <= -1.0);
    EXPECT_LT(r.appliedGainDb, r.requestedGainDb);
}

DSPARK_TEST(LoudnessNorm_defaults_are_the_broadcast_ones)
{
    // -23.0 LUFS is the EBU R 128 Programme Loudness Level and -1 dBTP its
    // maximum permitted true-peak level.
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
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::EmptyInput);
    EXPECT_TRUE(finiteResult<float>(r));
    EXPECT_EQ(r.appliedGainDb, 0.0f);
    AudioBuffer<float> b;
    pinkProgramme(b, 2.0, 1.0f);
    const auto before = snapshot(b);
    r = ln.normalize(b, std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::InvalidSampleRate);
    EXPECT_TRUE(bitwiseEqual(b, before));
    r = ln.normalize(b, -1.0);
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::InvalidSampleRate);
    EXPECT_TRUE(bitwiseEqual(b, before));

    // Silence has no gain that makes it -23 LUFS. It is returned untouched
    // rather than amplified until its noise floor measures right.
    AudioBuffer<float> quiet;
    quiet.resize(2, static_cast<int>(kFs * 3.0));
    const auto quietBefore = snapshot(quiet);
    r = ln.normalize(quiet, kFs);
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::NoMeasurableLoudness);
    EXPECT_TRUE(bitwiseEqual(quiet, quietBefore));
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
    EXPECT_TRUE(r.status == LoudnessNormalizer<float>::Status::Success);
    EXPECT_NEAR(static_cast<double>(r.outLUFS), -23.0, 0.1);
}

DSPARK_TEST(LoudnessNorm_non_finite_input_is_bitwise_rejected)
{
    verifyNonFiniteMatrix<float, 1>(1);
    verifyNonFiniteMatrix<float, 2>(2);
    verifyNonFiniteMatrix<float, 6>(6);
    verifyNonFiniteMatrix<float, 16>(16);
    verifyNonFiniteMatrix<float, 17>(17);
    verifyNonFiniteMatrix<float, 20>(20);
    verifyNonFiniteMatrix<double, 1>(1);
    verifyNonFiniteMatrix<double, 2>(2);
    verifyNonFiniteMatrix<double, 6>(6);
    verifyNonFiniteMatrix<double, 16>(16);
    verifyNonFiniteMatrix<double, 17>(17);
    verifyNonFiniteMatrix<double, 20>(20);
}

DSPARK_TEST(LoudnessNorm_ceiling_covers_every_runtime_channel)
{
    verifyAllChannelCeiling<float, 1>(1);
    verifyAllChannelCeiling<float, 2>(2);
    verifyAllChannelCeiling<float, 6>(6);
    verifyAllChannelCeiling<float, 16>(16);
    verifyAllChannelCeiling<float, 17>(17);
    verifyAllChannelCeiling<float, 20>(20);
    verifyAllChannelCeiling<float, 32>(32);
    verifyAllChannelCeiling<double, 1>(1);
    verifyAllChannelCeiling<double, 2>(2);
    verifyAllChannelCeiling<double, 6>(6);
    verifyAllChannelCeiling<double, 16>(16);
    verifyAllChannelCeiling<double, 17>(17);
    verifyAllChannelCeiling<double, 20>(20);
    verifyAllChannelCeiling<double, 32>(32);
}

DSPARK_TEST(LoudnessNorm_end_of_programme_inter_sample_peak_is_bounded)
{
    constexpr std::array<double, 12> tailPattern {
        -0.011016845703125, -0.59326171875, 0.453399658203125,
        -0.679534912109375, 0.8564453125, -0.30316162109375,
        0.321533203125, -0.90863037109375, 0.855712890625,
        0.911041259765625, -0.805908203125, 0.87860107421875,
    };
    AudioBuffer<double, 1> b;
    const int numSamples = static_cast<int>(0.6 * kFs);
    b.resize(1, numSamples);
    for (int i = 0; i < numSamples; ++i)
        b.getChannel(0)[i] = 0.01 * std::sin(twoPi<double> * 997.0 * i / kFs);
    for (std::size_t i = 0; i < tailPattern.size(); ++i)
        b.getChannel(0)[numSamples - static_cast<int>(tailPattern.size())
                        + static_cast<int>(i)] = 0.5 * tailPattern[i];

    const double peakWithoutTail = truePeakLinear(b, false);
    const double peakWithTail = truePeakLinear(b, true);
    EXPECT_GT(peakWithTail, peakWithoutTail * 1.5);

    LoudnessNormalizer<double> normalizer;
    normalizer.setTargetLUFS(0.0);
    const auto result = normalizer.normalize(b, kFs);
    const double applied = std::pow(10.0, result.appliedGainDb / 20.0);
    const double expectedSafe = std::pow(10.0, -1.0 / 20.0)
                              * (1.0 - 1.0e-4) / peakWithTail;
    const double outputPeak = truePeakLinear(b, true);
    std::printf("  [ln] end ISP: no-tail %.9f, causal-tail %.9f, "
                "requested-applied %.3f dB, out %+.6f dBTP\n",
                peakWithoutTail, peakWithTail,
                result.requestedGainDb - result.appliedGainDb,
                20.0 * std::log10(outputPeak));
    EXPECT_TRUE(result.status == LoudnessNormalizer<double>::Status::Success);
    EXPECT_TRUE(result.ceilingLimited);
    EXPECT_GT(result.requestedGainDb - result.appliedGainDb, 6.0);
    EXPECT_NEAR(applied, expectedSafe, 1e-12);
    EXPECT_TRUE(20.0 * std::log10(outputPeak) <= -1.0);
}

DSPARK_TEST(LoudnessNorm_numerical_failures_precede_mutation)
{
    AudioBuffer<float> b;
    pinkProgramme(b, 2.0, 0.2f);
    const auto before = snapshot(b);
    LoudnessNormalizer<float> normalizer;

    normalizer.setTargetLUFS(std::numeric_limits<float>::max());
    auto result = normalizer.normalize(b, kFs);
    EXPECT_TRUE(result.status == LoudnessNormalizer<float>::Status::NumericalFailure);
    EXPECT_TRUE(bitwiseEqual(b, before));
    EXPECT_TRUE(finiteResult<float>(result));

    normalizer.setTargetLUFS(std::numeric_limits<float>::lowest());
    result = normalizer.normalize(b, kFs);
    EXPECT_TRUE(result.status == LoudnessNormalizer<float>::Status::NumericalFailure);
    EXPECT_TRUE(bitwiseEqual(b, before));
    EXPECT_TRUE(finiteResult<float>(result));

    // A representable gain below -200 dB remains a successful, truthfully
    // reported constant gain rather than being clipped to a convenience floor.
    normalizer.setTargetLUFS(-250.0f);
    result = normalizer.normalize(b, kFs);
    EXPECT_TRUE(result.status == LoudnessNormalizer<float>::Status::Success);
    EXPECT_LT(result.appliedGainDb, -200.0f);
    EXPECT_NEAR(result.appliedGainDb, result.requestedGainDb, 0.001);
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
