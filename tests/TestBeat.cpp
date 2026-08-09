// DSPark - BeatTracker acceptance suite.
//
// Synthetic-only corpus, generated here: anti-aliased percussive clicks,
// tempo ramps, swung eighths, onset jitter, coloured noise beds and a
// three-against-two polyrhythm. Scoring uses a reimplemented F-measure
// (one-to-one greedy matching inside a tolerance) and a reimplemented Cemgil
// accuracy (Gaussian on the closest error, sigma 40 ms).
//
// Every case prints its numbers, so the ctest log doubles as the machine
// generated result artifact. Where a criterion is a worst case over a sweep,
// the worst value and the point it occurred at are both printed: a sweep
// reported only at its endpoints hides its own worst case.

#include "dspark_test.h"
#include "TestSignals.h"
#include "../Analysis/BeatTracker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

constexpr double kFs = 44100.0;

// ---------------------------------------------------------------------------
// Corpus generators
// ---------------------------------------------------------------------------

uint32_t rngState = 12345u;

double urand()
{
    rngState ^= rngState << 13; rngState ^= rngState >> 17; rngState ^= rngState << 5;
    return static_cast<double>(rngState) / 4294967296.0;
}

double nrand()
{
    double u1 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    const double u2 = urand();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(twoPi<double> * u2);
}

// Anti-aliased percussive click: a band-limited noise burst with a 1 ms
// raised-cosine attack and an exponential decay. Deterministic per position.
void addClick(std::vector<float>& buf, int64_t at, double fs,
              float amp = 1.0f, uint32_t seed = 7u)
{
    const double tau = 0.025 * fs;
    const int len = static_cast<int>(std::lround(tau * 5.0));
    const int att = std::max(1, static_cast<int>(0.001 * fs));
    uint32_t s = seed ^ static_cast<uint32_t>(at * 2654435761u) ^ 0x9e3779b9u;
    double lp = 0.0;
    for (int n = 0; n < len; ++n)
    {
        const int64_t i = at + n;
        if (i < 0 || i >= static_cast<int64_t>(buf.size())) continue;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const double white = static_cast<double>(s) / 4294967296.0 * 2.0 - 1.0;
        lp = 0.5 * lp + 0.5 * white;
        double env = std::exp(-static_cast<double>(n) / tau);
        if (n < att) env *= 0.5 - 0.5 * std::cos(pi<double> * n / att);
        buf[static_cast<size_t>(i)] += amp * static_cast<float>(lp * env);
    }
}

struct Corpus
{
    std::vector<float> x;
    std::vector<int64_t> beats;   ///< Ground truth, in samples.
    double bpm = 0.0;
};

// Half a second of silence leads every corpus, so that no ground-truth beat
// sits inside the analysis front end's warm-up.
constexpr double kLeadSeconds = 0.5;

Corpus clickTrain(double bpm, double seconds, double jitterMs = 0.0,
                  float amp = 1.0f, uint32_t seed = 7u)
{
    rngState = seed;
    Corpus c;
    c.bpm = bpm;
    const int64_t n = static_cast<int64_t>((seconds + kLeadSeconds + 1.0) * kFs);
    c.x.assign(static_cast<size_t>(n), 0.0f);
    const double period = 60.0 / bpm * kFs;
    const int64_t lead = static_cast<int64_t>(kLeadSeconds * kFs);
    for (int k = 0;; ++k)
    {
        const double ideal = static_cast<double>(lead) + static_cast<double>(k) * period;
        if (ideal > static_cast<double>(n) - kFs * 0.3) break;
        const double j = (jitterMs > 0.0) ? nrand() * jitterMs * 0.001 * kFs : 0.0;
        const int64_t at = static_cast<int64_t>(std::llround(ideal + j));
        if (at < 0) continue;
        addClick(c.x, at, kFs, amp, seed);
        c.beats.push_back(at);
    }
    return c;
}

Corpus rampTrain(double bpm0, double bpm1, double seconds, uint32_t seed = 11u)
{
    Corpus c;
    c.bpm = 0.5 * (bpm0 + bpm1);
    const int64_t n = static_cast<int64_t>((seconds + kLeadSeconds + 1.0) * kFs);
    c.x.assign(static_cast<size_t>(n), 0.0f);
    const int64_t lead = static_cast<int64_t>(kLeadSeconds * kFs);
    double t = 0.0;
    for (;;)
    {
        const int64_t at = lead + static_cast<int64_t>(std::llround(t * kFs));
        if (static_cast<double>(at) > static_cast<double>(n) - kFs * 0.3) break;
        addClick(c.x, at, kFs, 1.0f, seed);
        c.beats.push_back(at);
        const double frac = std::min(1.0, t / seconds);
        t += 60.0 / (bpm0 + (bpm1 - bpm0) * frac);
    }
    return c;
}

// Quarter notes are the ground truth; a quieter swung eighth sits `ratio` of
// the way through each quarter (2/3 for a triplet feel, 0.6 for 60/40).
Corpus swingTrain(double bpm, double seconds, double ratio, uint32_t seed = 13u)
{
    Corpus c;
    c.bpm = bpm;
    const int64_t n = static_cast<int64_t>((seconds + kLeadSeconds + 1.0) * kFs);
    c.x.assign(static_cast<size_t>(n), 0.0f);
    const double period = 60.0 / bpm * kFs;
    const int64_t lead = static_cast<int64_t>(kLeadSeconds * kFs);
    for (int k = 0;; ++k)
    {
        const double q = static_cast<double>(lead) + static_cast<double>(k) * period;
        if (q > static_cast<double>(n) - kFs * 0.3) break;
        addClick(c.x, static_cast<int64_t>(q), kFs, 1.0f, seed);
        c.beats.push_back(static_cast<int64_t>(q));
        const double off = q + ratio * period;
        if (off < static_cast<double>(n) - kFs * 0.3)
            addClick(c.x, static_cast<int64_t>(off), kFs, 0.7f, seed + 1u);
    }
    return c;
}

// Two pulses at once: halves and thirds of the same bar, both real.
Corpus polyrhythm(double barSeconds, double seconds, uint32_t seed = 17u)
{
    Corpus c;
    c.bpm = 60.0 / (barSeconds / 2.0);
    const int64_t n = static_cast<int64_t>((seconds + kLeadSeconds + 1.0) * kFs);
    c.x.assign(static_cast<size_t>(n), 0.0f);
    const int64_t lead = static_cast<int64_t>(kLeadSeconds * kFs);
    for (int k = 0;; ++k)
    {
        const double bar = static_cast<double>(lead)
                         + static_cast<double>(k) * barSeconds * kFs;
        if (bar > static_cast<double>(n) - kFs * 0.5) break;
        for (int j = 0; j < 2; ++j)
        {
            const double u = bar + static_cast<double>(j) * barSeconds * kFs / 2.0;
            if (u < static_cast<double>(n) - kFs * 0.3)
            {
                addClick(c.x, static_cast<int64_t>(u), kFs, 1.0f, seed);
                c.beats.push_back(static_cast<int64_t>(u));
            }
        }
        for (int j = 0; j < 3; ++j)
        {
            const double u = bar + static_cast<double>(j) * barSeconds * kFs / 3.0;
            if (u < static_cast<double>(n) - kFs * 0.3)
                addClick(c.x, static_cast<int64_t>(u), kFs, 0.85f, seed + 4u);
        }
    }
    std::sort(c.beats.begin(), c.beats.end());
    return c;
}

void addNoiseBed(std::vector<float>& x, float rms, uint32_t seed)
{
    rngState = seed;
    std::vector<double> w(x.size());
    double lp = 0.0, acc = 0.0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        lp = 0.85 * lp + 0.15 * (urand() * 2.0 - 1.0);
        w[i] = lp;
        acc += lp * lp;
    }
    const double r = std::sqrt(acc / static_cast<double>(x.size()));
    if (!(r > 0.0)) return;
    const double g = static_cast<double>(rms) / r;
    for (size_t i = 0; i < x.size(); ++i)
        x[i] += static_cast<float>(w[i] * g);
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

struct BeatScore
{
    double f = 0.0, precision = 0.0, recall = 0.0, cemgil = 0.0;
    double meanErrMs = 0.0, maxErrMs = 0.0, medianErrMs = 0.0;
    int matched = 0, nRef = 0, nEst = 0;
};

BeatScore scoreBeats(const std::vector<int64_t>& ref, const std::vector<int64_t>& est,
                     double toleranceMs)
{
    BeatScore s;
    s.nRef = static_cast<int>(ref.size());
    s.nEst = static_cast<int>(est.size());
    if (ref.empty() || est.empty()) return s;

    const double tol = toleranceMs * 0.001 * kFs;
    std::vector<char> used(est.size(), 0);
    double errSum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i)
    {
        int bestJ = -1;
        double bestD = tol + 1.0;
        for (size_t j = 0; j < est.size(); ++j)
        {
            if (used[j]) continue;
            const double d = std::abs(static_cast<double>(est[j] - ref[i]));
            if (d <= tol && d < bestD) { bestD = d; bestJ = static_cast<int>(j); }
        }
        if (bestJ >= 0)
        {
            used[static_cast<size_t>(bestJ)] = 1;
            ++s.matched;
            errSum += bestD;
            s.maxErrMs = std::max(s.maxErrMs, bestD / kFs * 1000.0);
        }
    }
    s.precision = static_cast<double>(s.matched) / static_cast<double>(est.size());
    s.recall = static_cast<double>(s.matched) / static_cast<double>(ref.size());
    s.f = (s.precision + s.recall > 0.0)
              ? 2.0 * s.precision * s.recall / (s.precision + s.recall) : 0.0;
    s.meanErrMs = (s.matched > 0) ? errSum / s.matched / kFs * 1000.0 : 0.0;

    const double sigma = 0.040 * kFs;
    double acc = 0.0;
    std::vector<double> closest;
    closest.reserve(ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double best = 1e18;
        for (size_t j = 0; j < est.size(); ++j)
            best = std::min(best, std::abs(static_cast<double>(est[j] - ref[i])));
        acc += std::exp(-0.5 * (best / sigma) * (best / sigma));
        closest.push_back(best / kFs * 1000.0);
    }
    s.cemgil = 2.0 * acc / static_cast<double>(ref.size() + est.size());
    std::sort(closest.begin(), closest.end());
    s.medianErrMs = closest.empty() ? 0.0 : closest[closest.size() / 2];
    return s;
}

// Which metrical level an estimate landed on, reported by name rather than
// absorbed into a tolerance: a tracker that is out by a factor of two is not
// "nearly right", and a criterion that cannot say so is not gating the error
// that dominates this problem.
enum class Level { Correct, Half, Double, Third, Triple, Other };

Level classify(double est, double truth, double tolPercent = 4.0)
{
    if (!(est > 0.0) || !(truth > 0.0)) return Level::Other;
    auto within = [&](double r) {
        return std::abs(est - truth * r) <= truth * r * tolPercent * 0.01;
    };
    if (within(1.0)) return Level::Correct;
    if (within(0.5)) return Level::Half;
    if (within(2.0)) return Level::Double;
    if (within(1.0 / 3.0)) return Level::Third;
    if (within(3.0)) return Level::Triple;
    return Level::Other;
}

const char* levelName(Level l)
{
    switch (l)
    {
        case Level::Correct: return "correct";
        case Level::Half:    return "HALF";
        case Level::Double:  return "DOUBLE";
        case Level::Third:   return "third";
        case Level::Triple:  return "triple";
        default:             return "other";
    }
}

BeatTracker<float>::Result analyzeCorpus(const Corpus& c, double alpha = -1.0)
{
    BeatTracker<float> bt;
    bt.prepare(AudioSpec{ kFs, 512, 1 });
    if (alpha > 0.0) bt.setTightness(static_cast<float>(alpha));
    const float* p = c.x.data();
    AudioBufferView<const float> v(&p, 1, static_cast<int>(c.x.size()));
    return bt.analyze(v);
}

struct CausalTrace
{
    std::vector<int64_t> beatSamples;
    std::vector<double> bpm;
    std::vector<int64_t> at;
    double finalConfidence = 0.0;
};

CausalTrace runCausal(const Corpus& c, int blockSize)
{
    BeatTracker<float> bt;
    bt.prepare(AudioSpec{ kFs, blockSize, 1 });
    CausalTrace tr;
    int64_t pos = 0, lastBeat = -1;
    const int64_t n = static_cast<int64_t>(c.x.size());
    while (pos < n)
    {
        const int64_t take = std::min<int64_t>(blockSize, n - pos);
        const float* p = c.x.data() + pos;
        AudioBufferView<const float> v(&p, 1, static_cast<int>(take));
        bt.processBlock(v);
        pos += take;
        if (bt.beatNow())
        {
            const int64_t b = bt.getLastBeatSample();
            if (b != lastBeat) tr.beatSamples.push_back(b);
            lastBeat = b;
        }
        float b = 0.0f, cf = 0.0f;
        bt.getTempoAndConfidence(b, cf);
        tr.bpm.push_back(static_cast<double>(b));
        tr.at.push_back(pos);
        tr.finalConfidence = static_cast<double>(cf);
    }
    return tr;
}

// The confidence below which the tracker is telling the caller not to rely on
// the grid. Documented here because two cases share it.
constexpr double kUntrustworthy = 0.5;

} // namespace

// ---------------------------------------------------------------------------
// Tempo
// ---------------------------------------------------------------------------

// Constant clean clicks, 60 to 180 BPM: the reported tempo must be within
// 0.5 BPM of the truth.
DSPARK_TEST(Beat_tempo_on_constant_clean_clicks)
{
    double worst = 0.0, worstAt = 0.0;
    for (double bpm = 60.0; bpm <= 180.5; bpm += 10.0)
    {
        const Corpus c = clickTrain(bpm, 30.0);
        const auto r = analyzeCorpus(c);
        const double err = static_cast<double>(r.tempoBpm) - bpm;
        if (std::abs(err) > std::abs(worst)) { worst = err; worstAt = bpm; }
        EXPECT_NEAR(static_cast<double>(r.tempoBpm), bpm, 0.5);
    }
    std::cout << "  tempo: worst signed error " << worst << " BPM at " << worstAt
              << " BPM (bound 0.5)\n";
}

// The whole searchable range, in 5 BPM steps. Two separate claims: the tempo
// is within 4% of the truth AT THE RIGHT METRICAL LEVEL for at least 95% of
// the sweep, and the half and double confusions are counted and reported by
// name rather than folded into that percentage.
DSPARK_TEST(Beat_tempo_across_the_range_and_metrical_level)
{
    int correct = 0, half = 0, doubled = 0, other = 0, total = 0;
    double worstPercent = 0.0, worstAt = 0.0;
    for (double bpm = 40.0; bpm <= 240.5; bpm += 5.0)
    {
        const Corpus c = clickTrain(bpm, 30.0);
        const auto r = analyzeCorpus(c);
        const Level l = classify(static_cast<double>(r.tempoBpm), bpm);
        ++total;
        switch (l)
        {
            case Level::Correct: ++correct; break;
            case Level::Half:    ++half; break;
            case Level::Double:  ++doubled; break;
            default:             ++other; break;
        }
        if (l == Level::Correct)
        {
            const double pct = (static_cast<double>(r.tempoBpm) - bpm) / bpm * 100.0;
            if (std::abs(pct) > std::abs(worstPercent)) { worstPercent = pct; worstAt = bpm; }
        }
        else
        {
            std::cout << "    " << bpm << " BPM -> " << static_cast<double>(r.tempoBpm)
                      << " (" << levelName(l) << "), secondary "
                      << static_cast<double>(r.secondaryTempoBpm) << "\n";
        }
    }
    const double accuracy = static_cast<double>(correct) / static_cast<double>(total);
    std::cout << "  range sweep: " << correct << "/" << total << " = " << accuracy
              << " at the correct level; HALF=" << half << " DOUBLE=" << doubled
              << " other=" << other << "; worst in-level error " << worstPercent
              << "% at " << worstAt << " BPM\n";
    EXPECT_GT(accuracy, 0.9499);
    EXPECT_EQ(half, 0);
    EXPECT_EQ(doubled, 0);
}

// ---------------------------------------------------------------------------
// Beat grid
// ---------------------------------------------------------------------------

// Clean clicks: every beat found, none invented, and each one placed within
// 10 ms on average and 20 ms at worst. The worst point of the sweep is the
// headline, not its endpoints.
DSPARK_TEST(Beat_grid_on_clean_clicks)
{
    double worstF = 1.0, worstCemgil = 1.0, worstMean = 0.0, worstMax = 0.0;
    double worstFAt = 0.0, worstCemgilAt = 0.0, worstMeanAt = 0.0, worstMaxAt = 0.0;
    for (double bpm : { 60.0, 75.0, 90.0, 100.0, 120.0, 140.0, 150.0, 160.0, 180.0 })
    {
        const Corpus c = clickTrain(bpm, 30.0);
        const auto r = analyzeCorpus(c);
        const auto s = scoreBeats(c.beats, r.beatSamples, 70.0);
        if (s.f < worstF) { worstF = s.f; worstFAt = bpm; }
        if (s.cemgil < worstCemgil) { worstCemgil = s.cemgil; worstCemgilAt = bpm; }
        if (s.meanErrMs > worstMean) { worstMean = s.meanErrMs; worstMeanAt = bpm; }
        if (s.maxErrMs > worstMax) { worstMax = s.maxErrMs; worstMaxAt = bpm; }
    }
    std::cout << "  grid worst case: F=" << worstF << " at " << worstFAt
              << " BPM; Cemgil=" << worstCemgil << " at " << worstCemgilAt
              << " BPM; mean error " << worstMean << " ms at " << worstMeanAt
              << " BPM; max error " << worstMax << " ms at " << worstMaxAt << " BPM\n";
    EXPECT_GT(worstF, 0.9999);
    EXPECT_GT(worstCemgil, 0.95);
    EXPECT_LT(worstMean, 10.0);
    EXPECT_LT(worstMax, 20.0);
}

// A tempo that moves. The grid must follow the tempo being played rather than
// the average of the piece: local inter-beat intervals within 5% of the local
// truth after the first eight beats, and a median beat error under 25 ms.
DSPARK_TEST(Beat_grid_follows_a_tempo_ramp)
{
    double worstLocal = 0.0, worstMedian = 0.0;
    for (auto pair : { std::pair<double, double>{ 100.0, 140.0 },
                       std::pair<double, double>{ 140.0, 100.0 },
                       std::pair<double, double>{ 90.0, 110.0 } })
    {
        const Corpus c = rampTrain(pair.first, pair.second, 40.0);
        const auto r = analyzeCorpus(c);
        const auto s = scoreBeats(c.beats, r.beatSamples, 70.0);

        double worst = 0.0;
        for (size_t k = 8; k + 1 < r.beatSamples.size(); ++k)
        {
            const double got = static_cast<double>(r.beatSamples[k + 1] - r.beatSamples[k]);
            size_t nearest = 0;
            int64_t bestD = INT64_MAX;
            for (size_t j = 0; j + 1 < c.beats.size(); ++j)
            {
                const int64_t d = std::llabs(c.beats[j] - r.beatSamples[k]);
                if (d < bestD) { bestD = d; nearest = j; }
            }
            if (nearest + 1 >= c.beats.size()) continue;
            const double truth = static_cast<double>(c.beats[nearest + 1] - c.beats[nearest]);
            const double pct = (got - truth) / truth * 100.0;
            if (std::abs(pct) > std::abs(worst)) worst = pct;
        }
        std::cout << "  ramp " << pair.first << "->" << pair.second
                  << " BPM: worst local tempo error " << worst
                  << "%, median beat error " << s.medianErrMs << " ms, F=" << s.f << "\n";
        if (std::abs(worst) > std::abs(worstLocal)) worstLocal = worst;
        worstMedian = std::max(worstMedian, s.medianErrMs);
    }
    std::cout << "  ramp worst case: local tempo error " << worstLocal
              << "% (bound 5), median beat error " << worstMedian << " ms (bound 25)\n";
    EXPECT_LT(std::abs(worstLocal), 5.0);
    EXPECT_LT(worstMedian, 25.0);
}

// Swung eighths. The quarter-note grid must be recovered within 15 ms, and
// the tracker must not lock onto the swung off-beat -- which is why the
// metrical level is asserted as well as the timing.
DSPARK_TEST(Beat_grid_on_swung_eighths)
{
    double worstMax = 0.0, worstF = 1.0;
    for (auto feel : { std::pair<const char*, double>{ "triplet 2:1", 2.0 / 3.0 },
                       std::pair<const char*, double>{ "60/40", 0.6 } })
    {
        for (double bpm : { 90.0, 120.0, 140.0 })
        {
            const Corpus c = swingTrain(bpm, 30.0, feel.second);
            const auto r = analyzeCorpus(c);
            const auto s = scoreBeats(c.beats, r.beatSamples, 15.0);
            const Level l = classify(static_cast<double>(r.tempoBpm), bpm);
            std::cout << "  " << feel.first << " at " << bpm << " BPM: tempo "
                      << static_cast<double>(r.tempoBpm) << " (" << levelName(l)
                      << "), F=" << s.f << " at 15 ms, max error " << s.maxErrMs
                      << " ms, secondary " << static_cast<double>(r.secondaryTempoBpm) << "\n";
            EXPECT_TRUE(l == Level::Correct);
            worstMax = std::max(worstMax, s.maxErrMs);
            worstF = std::min(worstF, s.f);
        }
    }
    std::cout << "  swing worst case: max error " << worstMax
              << " ms (bound 15), F=" << worstF << "\n";
    EXPECT_LT(worstMax, 15.0);
    EXPECT_GT(worstF, 0.9999);
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

// The confidence is the share of onset mass in phase with the delivered grid,
// so under Gaussian onset jitter of standard deviation s it has a closed
// form: exp(-2*pi^2*s^2/period^2), the characteristic function of the jitter.
// Checking against that is worth more than checking it merely falls, because
// a number that falls for the wrong reason also falls.
DSPARK_TEST(Beat_confidence_matches_the_closed_form_under_jitter)
{
    double previous = 2.0;
    double worstGap = 0.0, worstGapAt = 0.0;
    const double period = 0.5;   // seconds per beat at 120 BPM
    for (double jitterMs : { 0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0 })
    {
        const Corpus c = clickTrain(120.0, 30.0, jitterMs);
        const auto r = analyzeCorpus(c);
        const double s = jitterMs * 0.001;
        const double predicted = std::exp(-2.0 * pi<double> * pi<double> * s * s
                                          / (period * period));
        const double got = static_cast<double>(r.confidence);
        const double gap = std::abs(got - predicted);
        if (gap > worstGap) { worstGap = gap; worstGapAt = jitterMs; }
        std::cout << "  jitter " << jitterMs << " ms: confidence " << got
                  << ", closed form " << predicted << ", gap " << gap << "\n";
        EXPECT_LT(got, previous);       // strictly decreasing
        previous = got;
    }
    std::cout << "  jitter sweep: worst disagreement with the closed form " << worstGap
              << " at " << worstGapAt << " ms\n";
    EXPECT_LT(worstGap, 0.02);
}

// As the pulse weakens against a noise bed the confidence must fall, and it
// must be below the untrustworthy mark before the grid itself goes wrong --
// a tracker that reports a wrong grid confidently is worse than one that
// reports nothing.
DSPARK_TEST(Beat_confidence_falls_before_the_grid_does)
{
    double previous = 2.0;
    bool everWrongWhileConfident = false;
    for (double bed : { 0.0, 0.02, 0.05, 0.1, 0.2, 0.4 })
    {
        Corpus c = clickTrain(120.0, 30.0, 10.0);
        if (bed > 0.0) addNoiseBed(c.x, static_cast<float>(bed), 99u);
        const auto r = analyzeCorpus(c);
        const auto s = scoreBeats(c.beats, r.beatSamples, 70.0);
        const double conf = static_cast<double>(r.confidence);
        std::cout << "  bed RMS " << bed << ": confidence " << conf << ", tempo "
                  << static_cast<double>(r.tempoBpm) << ", F=" << s.f << "\n";
        if (s.f < 0.9 && conf >= kUntrustworthy) everWrongWhileConfident = true;
        EXPECT_LT(conf, previous);
        previous = conf;
    }
    EXPECT_FALSE(everWrongWhileConfident);
}

// The secondary hypothesis has to do something, not merely be reported. Two
// things are asserted: it names the metrical alternative where one exists in
// the searched range and is empty where none does, and where the alternative
// explains the signal nearly as well -- two genuine pulses at once -- it pulls
// the confidence down.
DSPARK_TEST(Beat_secondary_hypothesis_names_and_discounts)
{
    // 120 BPM: the half-tempo reading is inside the default range, so it is
    // named. 70 BPM: half of it is 35, below the range, so there is none.
    const auto fast = analyzeCorpus(clickTrain(120.0, 30.0));
    const auto slow = analyzeCorpus(clickTrain(70.0, 30.0));
    std::cout << "  120 BPM -> secondary " << static_cast<double>(fast.secondaryTempoBpm)
              << "; 70 BPM -> secondary " << static_cast<double>(slow.secondaryTempoBpm) << "\n";
    EXPECT_NEAR(static_cast<double>(fast.secondaryTempoBpm), 60.0, 1.0);
    EXPECT_NEAR(static_cast<double>(slow.secondaryTempoBpm), 0.0, 1e-6);

    // Unambiguous material must be untouched by the discount, and a
    // three-against-two polyrhythm must not be reported confidently.
    const auto clean = analyzeCorpus(clickTrain(120.0, 30.0));
    double worstPoly = 0.0;
    for (double bar : { 2.0, 1.4, 2.6 })
    {
        const auto p = analyzeCorpus(polyrhythm(bar, 30.0));
        std::cout << "  polyrhythm, bar " << bar << " s: tempo "
                  << static_cast<double>(p.tempoBpm) << ", secondary "
                  << static_cast<double>(p.secondaryTempoBpm) << ", confidence "
                  << static_cast<double>(p.confidence) << "\n";
        worstPoly = std::max(worstPoly, static_cast<double>(p.confidence));
    }
    std::cout << "  clean confidence " << static_cast<double>(clean.confidence)
              << " vs worst polyrhythm confidence " << worstPoly << "\n";
    EXPECT_GT(static_cast<double>(clean.confidence), 0.95);
    EXPECT_LT(worstPoly, kUntrustworthy);
}

// ---------------------------------------------------------------------------
// Causal path
// ---------------------------------------------------------------------------

// The real-time readouts, fed block by block. The lock-in window is the
// shorter of eight seconds and sixteen beats; the running tempo must reach
// within 5% inside it and stay within 4% after it, and beats must be
// attributed within 25 ms typically and 50 ms at worst. Attribution is
// measured only where ground truth exists: the tracker legitimately keeps
// predicting past the last click, and scoring that against a corpus that has
// stopped would be measuring the corpus.
DSPARK_TEST(Beat_causal_lock_in_and_beat_placement)
{
    double worstLock = 0.0, worstPost = 0.0, worstMedian = 0.0, worstMax = 0.0;
    double worstLockAt = 0.0, worstPostAt = 0.0;
    for (double bpm : { 60.0, 75.0, 90.0, 120.0, 150.0, 180.0 })
    {
        const Corpus c = clickTrain(bpm, 40.0);
        const auto tr = runCausal(c, 512);
        const double window = std::min(8.0, 16.0 * 60.0 / bpm);
        const int64_t windowSamples = static_cast<int64_t>(window * kFs);
        const int64_t lastRef = c.beats.back();

        double lock = -1.0;
        for (size_t i = 0; i < tr.bpm.size(); ++i)
            if (tr.bpm[i] > 0.0 && std::abs(tr.bpm[i] - bpm) <= 0.05 * bpm)
            { lock = static_cast<double>(tr.at[i]) / kFs; break; }

        double post = 0.0;
        for (size_t i = 0; i < tr.bpm.size(); ++i)
            if (tr.at[i] > windowSamples && tr.at[i] <= lastRef)
                post = std::max(post, std::abs(tr.bpm[i] - bpm) / bpm * 100.0);

        std::vector<double> errs;
        for (const int64_t b : tr.beatSamples)
        {
            if (b <= windowSamples || b > lastRef) continue;
            double best = 1e18;
            for (const int64_t t : c.beats)
                best = std::min(best, std::abs(static_cast<double>(t - b)));
            errs.push_back(best / kFs * 1000.0);
        }
        std::sort(errs.begin(), errs.end());
        const double median = errs.empty() ? 1e9 : errs[errs.size() / 2];
        const double maxErr = errs.empty() ? 1e9 : errs.back();
        std::cout << "  " << bpm << " BPM: window " << window << " s, locked at "
                  << lock << " s, tempo error after the window " << post
                  << "%, beat placement median " << median << " ms, max " << maxErr
                  << " ms, over " << errs.size() << " beats\n";
        EXPECT_TRUE(lock >= 0.0 && lock <= window);
        if (lock > worstLock) { worstLock = lock; worstLockAt = bpm; }
        if (post > worstPost) { worstPost = post; worstPostAt = bpm; }
        worstMedian = std::max(worstMedian, median);
        worstMax = std::max(worstMax, maxErr);
    }
    std::cout << "  causal worst case: lock-in " << worstLock << " s at " << worstLockAt
              << " BPM; tempo error after the window " << worstPost << "% at "
              << worstPostAt << " BPM; beat placement median " << worstMedian
              << " ms, max " << worstMax << " ms\n";
    EXPECT_LT(worstPost, 4.0);
    EXPECT_LT(worstMedian, 25.0);
    EXPECT_LT(worstMax, 50.0);
}

// The causal readouts must not depend on how the host happens to cut the
// stream into blocks.
DSPARK_TEST(Beat_causal_output_is_block_size_independent)
{
    const Corpus c = clickTrain(120.0, 20.0);
    const auto a = runCausal(c, 64);
    const auto b = runCausal(c, 512);
    const auto d = runCausal(c, 1000);
    std::cout << "  beats reported: 64 -> " << a.beatSamples.size() << ", 512 -> "
              << b.beatSamples.size() << ", 1000 -> " << d.beatSamples.size() << "\n";
    EXPECT_EQ(a.beatSamples.size(), b.beatSamples.size());
    EXPECT_EQ(a.beatSamples.size(), d.beatSamples.size());
    int64_t worst = 0;
    const size_t m = std::min({ a.beatSamples.size(), b.beatSamples.size(),
                                d.beatSamples.size() });
    for (size_t i = 0; i < m; ++i)
    {
        worst = std::max<int64_t>(worst, std::llabs(a.beatSamples[i] - b.beatSamples[i]));
        worst = std::max<int64_t>(worst, std::llabs(a.beatSamples[i] - d.beatSamples[i]));
    }
    std::cout << "  worst difference in attributed sample across block sizes: "
              << worst << "\n";
    EXPECT_EQ(worst, static_cast<int64_t>(0));
}

// On material the offline path also finds unreliable, the causal readout must
// stay below the same mark rather than emitting a confident grid.
DSPARK_TEST(Beat_causal_confidence_on_weak_material)
{
    for (auto material : { std::pair<double, double>{ 10.0, 0.2 },
                           std::pair<double, double>{ 10.0, 0.4 },
                           std::pair<double, double>{ 60.0, 0.3 } })
    {
        Corpus c = clickTrain(120.0, 40.0, material.first);
        addNoiseBed(c.x, static_cast<float>(material.second), 99u);
        const auto tr = runCausal(c, 512);
        std::cout << "  jitter " << material.first << " ms over a bed of "
                  << material.second << ": causal confidence " << tr.finalConfidence << "\n";
        EXPECT_LT(tr.finalConfidence, kUntrustworthy);
    }
}

// The published tempo and its confidence describe the same instant, so they
// travel as one word and come back from one load.
DSPARK_TEST(Beat_tempo_and_confidence_are_published_together)
{
    const Corpus c = clickTrain(120.0, 20.0);
    BeatTracker<float> bt;
    bt.prepare(AudioSpec{ kFs, 512, 1 });
    const int64_t n = static_cast<int64_t>(c.x.size());
    for (int64_t pos = 0; pos < n; pos += 512)
    {
        const int64_t take = std::min<int64_t>(512, n - pos);
        const float* p = c.x.data() + pos;
        AudioBufferView<const float> v(&p, 1, static_cast<int>(take));
        bt.processBlock(v);
    }
    float bpm = 0.0f, conf = 0.0f;
    bt.getTempoAndConfidence(bpm, conf);
    std::cout << "  packed readout: " << bpm << " BPM at confidence " << conf
              << "; separate getters: " << bt.getRunningTempoBpm() << " / "
              << bt.getConfidence() << "\n";
    EXPECT_NEAR(static_cast<double>(bpm), static_cast<double>(bt.getRunningTempoBpm()), 1e-9);
    EXPECT_NEAR(static_cast<double>(conf), static_cast<double>(bt.getConfidence()), 1e-9);
    EXPECT_NEAR(static_cast<double>(bpm), 120.0, 5.0);
}

// ---------------------------------------------------------------------------
// Contract and robustness
// ---------------------------------------------------------------------------

// The envelope readout the tracker consumes must describe the frame the
// caller just caused: same value the detector computed, at a position inside
// the window that produced it, advancing exactly once per hop.
DSPARK_TEST(Beat_onset_envelope_readout_advances_once_per_hop)
{
    OnsetDetector<float> od;
    od.prepare(AudioSpec{ kFs, 512, 1 });
    const int hop = od.getHopSize();
    EXPECT_GT(hop, 0);
    EXPECT_GT(od.getWarmupFrames(), 0);
    EXPECT_LT(od.getEnvelopeLatencySamples(), od.getLatencySamples());

    Corpus c = clickTrain(120.0, 4.0);
    int64_t pushed = 0;
    int64_t frames = 0;
    const int64_t n = static_cast<int64_t>(c.x.size());
    // The first frames localise to a point BEFORE the stream started: the
    // frame centre of a window that is still half empty is a negative sample
    // index, and that is the honest answer, so the comparison starts below
    // every representable index rather than at zero.
    int64_t previousReference = INT64_MIN;
    bool referencesAscend = true;
    while (pushed < n)
    {
        const int64_t take = std::min<int64_t>(hop, n - pushed);
        od.pushSamples(std::span<const float>(c.x.data() + pushed,
                                              static_cast<size_t>(take)));
        pushed += take;
        if (take == hop)
        {
            ++frames;
            const auto f = od.getLastOdfFrame();
            if (f.frameIndex != frames) referencesAscend = false;
            if (f.referenceSample <= previousReference) referencesAscend = false;
            if (pushed - f.referenceSample != od.getEnvelopeLatencySamples())
                referencesAscend = false;
            previousReference = f.referenceSample;
        }
    }
    std::cout << "  " << frames << " frames over " << n << " samples at hop " << hop
              << "; envelope delay " << od.getEnvelopeLatencySamples()
              << " samples, latch delay " << od.getLatencySamples() << "\n";
    EXPECT_TRUE(referencesAscend);
    EXPECT_EQ(frames, static_cast<int64_t>(n / hop));

    od.reset();
    EXPECT_EQ(od.getLastOdfFrame().frameIndex, static_cast<int64_t>(0));
}

// Setting the tempo range must be honoured and must not need to allocate,
// so it stays callable while audio is running.
DSPARK_TEST(Beat_tempo_range_is_honoured_and_rejects_nonsense)
{
    const Corpus c = clickTrain(160.0, 30.0);
    BeatTracker<float> narrow;
    narrow.prepare(AudioSpec{ kFs, 512, 1 });
    narrow.setTempoRange(60.0f, 100.0f);
    const float* p = c.x.data();
    AudioBufferView<const float> v(&p, 1, static_cast<int>(c.x.size()));
    const auto r = narrow.analyze(v);
    std::cout << "  160 BPM material searched over 60..100 BPM -> "
              << static_cast<double>(r.tempoBpm) << " BPM (the half-tempo reading,"
              << " which is the only one inside the range)\n";
    EXPECT_GT(static_cast<double>(r.tempoBpm), 60.0);
    EXPECT_LT(static_cast<double>(r.tempoBpm), 100.0);
    EXPECT_NEAR(static_cast<double>(r.tempoBpm), 80.0, 1.0);

    // Nonsense is ignored, leaving the previous range in force.
    BeatTracker<float> bt;
    bt.prepare(AudioSpec{ kFs, 512, 1 });
    bt.setTempoRange(std::numeric_limits<float>::quiet_NaN(), 200.0f);
    bt.setTempoRange(200.0f, 100.0f);   // inverted
    bt.setTempoRange(-5.0f, 0.0f);
    const auto still = bt.analyze(v);
    std::cout << "  after three rejected range requests: "
              << static_cast<double>(still.tempoBpm) << " BPM\n";
    EXPECT_NEAR(static_cast<double>(still.tempoBpm), 160.0, 1.0);

    bt.setTightness(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(static_cast<double>(bt.getTightness()), 400.0, 1e-3);
}

// Degenerate and hostile input must return an empty answer rather than a
// tempo fitted to nothing, and must never produce a non-finite number.
DSPARK_TEST(Beat_degenerate_input_returns_nothing_rather_than_noise)
{
    BeatTracker<float> bt;

    // Before prepare().
    std::vector<float> tiny(64, 0.0f);
    const float* tp = tiny.data();
    AudioBufferView<const float> tv(&tp, 1, 64);
    auto r = bt.analyze(tv);
    EXPECT_EQ(r.beatSamples.size(), static_cast<size_t>(0));
    bt.processBlock(tv);
    EXPECT_FALSE(bt.beatNow());

    bt.prepare(AudioSpec{ kFs, 512, 1 });

    // Too short for the slowest searched tempo.
    r = bt.analyze(tv);
    EXPECT_EQ(r.beatSamples.size(), static_cast<size_t>(0));
    EXPECT_NEAR(static_cast<double>(r.tempoBpm), 0.0, 1e-9);

    // Silence, long enough.
    std::vector<float> quiet(static_cast<size_t>(kFs * 20.0), 0.0f);
    const float* qp = quiet.data();
    AudioBufferView<const float> qv(&qp, 1, static_cast<int>(quiet.size()));
    r = bt.analyze(qv);
    std::cout << "  silence -> tempo " << static_cast<double>(r.tempoBpm)
              << ", confidence " << static_cast<double>(r.confidence) << ", "
              << r.beatSamples.size() << " beats\n";
    EXPECT_TRUE(std::isfinite(static_cast<double>(r.tempoBpm)));
    EXPECT_TRUE(std::isfinite(static_cast<double>(r.confidence)));

    // Non-finite samples must be absorbed, not propagated.
    Corpus c = clickTrain(120.0, 20.0);
    c.x[1000] = std::numeric_limits<float>::quiet_NaN();
    c.x[2000] = std::numeric_limits<float>::infinity();
    c.x[3000] = -std::numeric_limits<float>::infinity();
    const float* cp = c.x.data();
    AudioBufferView<const float> cv(&cp, 1, static_cast<int>(c.x.size()));
    r = bt.analyze(cv);
    std::cout << "  with injected NaN/Inf -> tempo " << static_cast<double>(r.tempoBpm)
              << ", confidence " << static_cast<double>(r.confidence) << "\n";
    EXPECT_TRUE(std::isfinite(static_cast<double>(r.tempoBpm)));
    EXPECT_NEAR(static_cast<double>(r.tempoBpm), 120.0, 1.0);

    // An invalid spec leaves the previous configuration alone.
    const int before = bt.getLatencySamples();
    bt.prepare(AudioSpec{ -1.0, 512, 1 });
    EXPECT_EQ(bt.getLatencySamples(), before);
}

// Both instantiations must exist and agree on a clean signal.
DSPARK_TEST(Beat_double_instantiation)
{
    const Corpus c = clickTrain(120.0, 20.0);
    std::vector<double> wide(c.x.begin(), c.x.end());

    BeatTracker<float> bf;
    bf.prepare(AudioSpec{ kFs, 512, 1 });
    const float* fp = c.x.data();
    AudioBufferView<const float> fv(&fp, 1, static_cast<int>(c.x.size()));
    const auto rf = bf.analyze(fv);

    BeatTracker<double> bd;
    bd.prepare(AudioSpec{ kFs, 512, 1 });
    const double* dp = wide.data();
    AudioBufferView<const double> dv(&dp, 1, static_cast<int>(wide.size()));
    const auto rd = bd.analyze(dv);

    std::cout << "  float " << static_cast<double>(rf.tempoBpm) << " BPM / "
              << rf.beatSamples.size() << " beats; double " << rd.tempoBpm
              << " BPM / " << rd.beatSamples.size() << " beats\n";
    EXPECT_NEAR(static_cast<double>(rf.tempoBpm), rd.tempoBpm, 0.5);
    EXPECT_EQ(rf.beatSamples.size(), rd.beatSamples.size());
}

// reset() must return the object to its state after prepare(), so that a
// second stream is not tracked through the first one's memory.
DSPARK_TEST(Beat_reset_clears_the_causal_state)
{
    const Corpus fast = clickTrain(180.0, 20.0);
    const Corpus slow = clickTrain(70.0, 20.0);
    BeatTracker<float> bt;
    bt.prepare(AudioSpec{ kFs, 512, 1 });

    auto feed = [&](const Corpus& c) {
        const int64_t n = static_cast<int64_t>(c.x.size());
        for (int64_t pos = 0; pos < n; pos += 512)
        {
            const int64_t take = std::min<int64_t>(512, n - pos);
            const float* p = c.x.data() + pos;
            AudioBufferView<const float> v(&p, 1, static_cast<int>(take));
            bt.processBlock(v);
        }
    };

    feed(fast);
    const double afterFast = static_cast<double>(bt.getRunningTempoBpm());
    bt.reset();
    EXPECT_NEAR(static_cast<double>(bt.getRunningTempoBpm()), 0.0, 1e-9);
    EXPECT_EQ(bt.getLastBeatSample(), static_cast<int64_t>(-1));
    feed(slow);
    const double afterSlow = static_cast<double>(bt.getRunningTempoBpm());

    BeatTracker<float> fresh;
    fresh.prepare(AudioSpec{ kFs, 512, 1 });
    const int64_t n = static_cast<int64_t>(slow.x.size());
    for (int64_t pos = 0; pos < n; pos += 512)
    {
        const int64_t take = std::min<int64_t>(512, n - pos);
        const float* p = slow.x.data() + pos;
        AudioBufferView<const float> v(&p, 1, static_cast<int>(take));
        fresh.processBlock(v);
    }
    const double reference = static_cast<double>(fresh.getRunningTempoBpm());
    std::cout << "  after 180 BPM: " << afterFast << "; after reset and 70 BPM: "
              << afterSlow << "; a fresh instance on the same 70 BPM: " << reference << "\n";
    EXPECT_NEAR(afterSlow, reference, 1e-9);
}
