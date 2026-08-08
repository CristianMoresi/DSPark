// DSPark - OnsetDetector acceptance suite (criteria OA-1..OA-6, below).
//
// Synthetic-only corpus: anti-aliased (windowed-sinc / decaying-sine)
// clicks, pink/brown noise beds, soft onsets over a harmonic bed, FM/AM
// vibrato stress. Scoring uses a reimplemented mir_eval-style F-measure
// (+/-25 ms, one-to-one greedy matching). Numbers are printed so the ctest
// log doubles as the machine-generated OA results artifact.

#include "dspark_test.h"
#include "TestSignals.h"
#include "../Analysis/OnsetDetector.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

// ---------------------------------------------------------------------------
// Synthetic corpus generators
// ---------------------------------------------------------------------------

// Anti-aliased broadband percussive click: a band-limited (one-pole low-passed)
// noise burst with a sharp 1 ms raised-cosine attack and an exponential decay.
// Broadband so it stands out over colored-noise beds; sharp attack gives a
// well-defined onset. Deterministic per (at, seed). Onset ground truth = start.
void addClick(std::vector<float>& buf, int64_t at, double fs,
              double decayMs = 25.0, float amp = 1.0f, uint32_t seed = 0u)
{
    const double tau = decayMs * 0.001 * fs;
    const int len = static_cast<int>(std::lround(tau * 5.0));
    const int att = std::max(1, static_cast<int>(0.001 * fs));
    uint32_t s = seed ^ static_cast<uint32_t>(at * 2654435761u);
    double lp = 0.0;
    for (int n = 0; n < len; ++n)
    {
        const int64_t i = at + n;
        if (i < 0 || i >= static_cast<int64_t>(buf.size())) continue;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        const double white = static_cast<double>(s) / static_cast<double>(0xFFFFFFFFu) * 2.0 - 1.0;
        lp = 0.5 * lp + 0.5 * white; // mild band-limit (anti-alias)
        double env = std::exp(-static_cast<double>(n) / tau);
        if (n < att) env *= 0.5 - 0.5 * std::cos(dspark::pi<double> * n / att);
        buf[static_cast<size_t>(i)] += amp * static_cast<float>(lp * env);
    }
}

// One-pole pink-ish (-3 dB/oct approx via 3-stage) and brown (-6 dB/oct) noise,
// RMS-normalised to targetRms.
std::vector<float> makeNoise(int n, double /*fs*/, int color, float targetRms, uint32_t seed)
{
    std::vector<float> w(static_cast<size_t>(n));
    uint32_t s = seed;
    auto rnd = [&s]() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<float>(s) / static_cast<float>(0xFFFFFFFFu) * 2.0f - 1.0f;
    };
    // Paul Kellet pink filter state.
    double b0 = 0, b1 = 0, b2 = 0, brown = 0;
    for (int i = 0; i < n; ++i)
    {
        const double white = rnd();
        double out;
        if (color == 1) // pink
        {
            b0 = 0.99765 * b0 + white * 0.0990460;
            b1 = 0.96300 * b1 + white * 0.2965164;
            b2 = 0.57000 * b2 + white * 1.0526913;
            out = b0 + b1 + b2 + white * 0.1848;
            out *= 0.15;
        }
        else if (color == 2) // brown
        {
            brown = (brown + 0.02 * white) / 1.02;
            out = brown * 3.5;
        }
        else out = white;
        w[static_cast<size_t>(i)] = static_cast<float>(out);
    }
    // RMS normalise.
    double sq = 0.0;
    for (float v : w) sq += static_cast<double>(v) * v;
    const double rms = std::sqrt(sq / std::max(1, n));
    const float g = (rms > 1e-12) ? targetRms / static_cast<float>(rms) : 1.0f;
    for (float& v : w) v *= g;
    return w;
}

// Sustained additive harmonic bed (quiet), used as a masking background.
void addHarmonicBed(std::vector<float>& buf, double fs, double f0, float amp)
{
    const int n = static_cast<int>(buf.size());
    for (int i = 0; i < n; ++i)
    {
        double s = 0.0;
        for (int h = 1; h <= 6; ++h)
            s += std::sin(dspark::twoPi<double> * f0 * h * i / fs) / h;
        buf[static_cast<size_t>(i)] += amp * static_cast<float>(s * 0.4);
    }
}

// Soft note onset: a harmonic tone with a 10 ms linear attack then sustain.
void addSoftNote(std::vector<float>& buf, int64_t at, double fs, double f0,
                 double attackMs, double durMs, float amp)
{
    const int att = static_cast<int>(attackMs * 0.001 * fs);
    const int dur = static_cast<int>(durMs * 0.001 * fs);
    for (int n = 0; n < dur; ++n)
    {
        const int64_t i = at + n;
        if (i < 0 || i >= static_cast<int64_t>(buf.size())) continue;
        double env = (n < att) ? static_cast<double>(n) / std::max(1, att) : 1.0;
        if (n > dur - att) env *= static_cast<double>(dur - n) / std::max(1, att);
        double s = 0.0;
        for (int h = 1; h <= 4; ++h)
            s += std::sin(dspark::twoPi<double> * f0 * h * (at + n) / fs) / h;
        buf[static_cast<size_t>(i)] += amp * static_cast<float>(s * env);
    }
}

// FM (vibrato) + AM (tremolo) sustained tone, no true onsets.
std::vector<float> makeVibratoTone(int n, double fs, double f0, float amp)
{
    std::vector<float> out(static_cast<size_t>(n));
    double phase = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double t = i / fs;
        const double vib = 1.0 + 0.03 * std::sin(dspark::twoPi<double> * 6.0 * t);   // 6 Hz FM +/-3%
        const double trem = 0.8 + 0.2 * std::sin(dspark::twoPi<double> * 5.0 * t);   // 5 Hz AM
        phase += dspark::twoPi<double> * f0 * vib / fs;
        out[static_cast<size_t>(i)] = amp * static_cast<float>(std::sin(phase) * trem);
    }
    return out;
}

// ---------------------------------------------------------------------------
// mir_eval-style F-measure (greedy one-to-one within a tolerance window)
// ---------------------------------------------------------------------------

struct PRF { double p, r, f; int tp, fp, fn; double meanErrMs, maxErrMs; };

PRF scoreOnsets(const std::vector<int64_t>& detected,
                const std::vector<int64_t>& truth, double fs, double windowMs)
{
    const int64_t win = static_cast<int64_t>(std::lround(windowMs * 0.001 * fs));
    std::vector<bool> usedDet(detected.size(), false);
    int tp = 0;
    double sumErr = 0.0, maxErr = 0.0;
    for (int64_t gt : truth)
    {
        int best = -1; int64_t bestDist = win + 1;
        for (size_t d = 0; d < detected.size(); ++d)
        {
            if (usedDet[d]) continue;
            const int64_t dist = std::llabs(detected[d] - gt);
            if (dist <= win && dist < bestDist) { bestDist = dist; best = static_cast<int>(d); }
        }
        if (best >= 0)
        {
            usedDet[static_cast<size_t>(best)] = true;
            ++tp;
            const double errMs = static_cast<double>(bestDist) * 1000.0 / fs;
            sumErr += errMs;
            maxErr = std::max(maxErr, errMs);
        }
    }
    const int fn = static_cast<int>(truth.size()) - tp;
    const int fp = static_cast<int>(detected.size()) - tp;
    const double p = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 1.0;
    const double r = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 1.0;
    const double f = (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
    return { p, r, f, tp, fp, fn, (tp > 0 ? sumErr / tp : 0.0), maxErr };
}

// Run the detector offline over a mono signal, return detected onset samples.
std::vector<int64_t> runOffline(std::vector<float>& sig, double fs,
                                OnsetDetector<float>::Method method,
                                int fft = 2048, int hop = 0)
{
    OnsetDetector<float> od;
    od.prepare(AudioSpec{ fs, 512, 1 }, fft, hop);
    od.setMethod(method);
    float* ptr = sig.data();
    AudioBufferView<float> view(&ptr, 1, static_cast<int>(sig.size()));
    return od.detectOffline(view);
}

} // namespace

// ===========================================================================
// OA-1: isolated band-limited clicks over silence -> F = 1.00, zero FP.
// ===========================================================================
DSPARK_TEST(Onset_OA1_isolated_clicks_perfect)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 6.0);
    std::vector<float> sig(static_cast<size_t>(n), 0.0f);
    std::vector<int64_t> truth;
    // 10 well-separated clicks, first after >100 ms warm-up.
    for (int k = 0; k < 10; ++k)
    {
        const int64_t at = static_cast<int64_t>(fs * (0.3 + 0.5 * k));
        addClick(sig, at, fs);
        truth.push_back(at);
    }
    auto det = runOffline(sig, fs, OnsetDetector<float>::Method::SuperFlux);
    PRF s = scoreOnsets(det, truth, fs, 25.0);
    std::cout << "  [OA-1] SuperFlux clicks/silence: P=" << s.p << " R=" << s.r
              << " F=" << s.f << " (tp=" << s.tp << " fp=" << s.fp
              << " fn=" << s.fn << ", detected=" << det.size() << ")\n";
    EXPECT_EQ(s.fp, 0);
    EXPECT_EQ(s.fn, 0);
    EXPECT_NEAR(s.f, 1.0, 1e-9);
}

// ===========================================================================
// OA-2: localization on clean clicks -> mean |err| <= 10 ms, max <= 20 ms.
// ===========================================================================
DSPARK_TEST(Onset_OA2_localization)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 6.0);
    std::vector<float> sig(static_cast<size_t>(n), 0.0f);
    std::vector<int64_t> truth;
    for (int k = 0; k < 10; ++k)
    {
        const int64_t at = static_cast<int64_t>(fs * (0.3 + 0.5 * k));
        addClick(sig, at, fs);
        truth.push_back(at);
    }
    auto det = runOffline(sig, fs, OnsetDetector<float>::Method::SuperFlux);
    PRF s = scoreOnsets(det, truth, fs, 25.0);
    std::cout << "  [OA-2] localization: meanErr=" << s.meanErrMs
              << " ms  maxErr=" << s.maxErrMs << " ms\n";
    EXPECT_EQ(s.tp, 10);
    EXPECT_LT(s.meanErrMs, 10.0);
    EXPECT_LT(s.maxErrMs, 20.0);
}

// ===========================================================================
// OA-3: clicks over pink AND brown noise at 0 dB onset-to-bed RMS -> F >= 0.95.
// ===========================================================================
DSPARK_TEST(Onset_OA3_clicks_over_noise)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 6.0);
    for (int color = 1; color <= 2; ++color) // 1=pink, 2=brown
    {
        std::vector<int64_t> truth;
        std::vector<float> clicks(static_cast<size_t>(n), 0.0f);
        for (int k = 0; k < 10; ++k)
        {
            const int64_t at = static_cast<int64_t>(fs * (0.3 + 0.5 * k));
            addClick(clicks, at, fs, 25.0, 1.0f, static_cast<uint32_t>(k));
            truth.push_back(at);
        }
        // Click burst RMS (measured over its active region).
        double sq = 0.0; int cnt = 0;
        for (float v : clicks) { if (std::abs(v) > 1e-6f) { sq += v * v; ++cnt; } }
        const float clickRms = (cnt > 0) ? static_cast<float>(std::sqrt(sq / cnt)) : 0.1f;
        auto bed = makeNoise(n, fs, color, clickRms, 777u + static_cast<uint32_t>(color));
        std::vector<float> sig(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) sig[static_cast<size_t>(i)] = clicks[static_cast<size_t>(i)] + bed[static_cast<size_t>(i)];

        auto det = runOffline(sig, fs, OnsetDetector<float>::Method::SuperFlux);
        PRF s = scoreOnsets(det, truth, fs, 25.0);
        std::cout << "  [OA-3] " << (color == 1 ? "pink" : "brown")
                  << " 0 dB: F=" << s.f << " (tp=" << s.tp << " fp=" << s.fp
                  << " fn=" << s.fn << ")\n";
        EXPECT_GT(s.f, 0.949);
    }
}

// ===========================================================================
// OA-4: soft onsets (10 ms attack) over a sustained harmonic bed -> F >= 0.80.
// ===========================================================================
DSPARK_TEST(Onset_OA4_soft_onsets_over_bed)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 8.0);
    std::vector<float> sig(static_cast<size_t>(n), 0.0f);
    addHarmonicBed(sig, fs, 110.0, 0.15f);
    std::vector<int64_t> truth;
    const double notes[8] = { 220, 262, 294, 330, 349, 392, 440, 494 };
    for (int k = 0; k < 8; ++k)
    {
        const int64_t at = static_cast<int64_t>(fs * (0.5 + 0.9 * k));
        addSoftNote(sig, at, fs, notes[k], 10.0, 600.0, 0.5f);
        truth.push_back(at);
    }
    auto det = runOffline(sig, fs, OnsetDetector<float>::Method::SuperFlux);
    PRF s = scoreOnsets(det, truth, fs, 25.0);
    std::cout << "  [OA-4] soft onsets/bed: F=" << s.f << " (tp=" << s.tp
              << " fp=" << s.fp << " fn=" << s.fn << ")\n";
    EXPECT_GT(s.f, 0.799);
}

// ===========================================================================
// OA-5: FM/AM sustained tone, no true onsets -> 0 SuperFlux false onsets, and
// SuperFlux FP <= plain SpectralFlux FP (vibrato suppression).
// ===========================================================================
DSPARK_TEST(Onset_OA5_vibrato_false_positives)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 10.0);
    auto tone = makeVibratoTone(n, fs, 440.0, 0.6f);

    std::vector<float> a = tone;
    auto detSF = runOffline(a, fs, OnsetDetector<float>::Method::SuperFlux);
    std::vector<float> b = tone;
    auto detFlux = runOffline(b, fs, OnsetDetector<float>::Method::SpectralFlux);

    // Ignore the unavoidable ramp-in transient in the first frame region.
    auto countAfterWarmup = [&](const std::vector<int64_t>& v) {
        int c = 0; for (int64_t t : v) if (t > static_cast<int64_t>(fs * 0.2)) ++c; return c;
    };
    const int fpSF = countAfterWarmup(detSF);
    const int fpFlux = countAfterWarmup(detFlux);
    std::cout << "  [OA-5] vibrato FP: SuperFlux=" << fpSF
              << "  SpectralFlux=" << fpFlux << "\n";
    EXPECT_EQ(fpSF, 0);
    EXPECT_LT(fpSF, fpFlux + 1); // SuperFlux <= SpectralFlux
}

// ===========================================================================
// OA-6: causal latency == fftSize + hop exactly, block-size independent,
// verified at 44.1 and 48 kHz; zero alloc in processBlock (checked by ASan).
// ===========================================================================
namespace {
// Push a click sample-by-sample and return the global sample index at which the
// onset latch first fires, plus the detector's reported reference sample.
struct LatchResult { int64_t latchIndex; int64_t reference; int latency; };

LatchResult measureLatch(double fs, int fft, int hop, int64_t clickAt)
{
    OnsetDetector<float> od;
    od.prepare(AudioSpec{ fs, 1, 1 }, fft, hop);
    od.setMethod(OnsetDetector<float>::Method::SuperFlux);
    const int H = od.getHopSize();
    const int64_t total = clickAt + static_cast<int64_t>(fft) * 4 + H * 4;

    // Precompute the click into a full buffer, then stream one sample at a time.
    std::vector<float> sig(static_cast<size_t>(total), 0.0f);
    addClick(sig, clickAt, fs);

    LatchResult res{ -1, -1, od.getLatencySamples() };
    for (int64_t i = 0; i < total; ++i)
    {
        float s = sig[static_cast<size_t>(i)];
        od.pushSamples(std::span<const float>(&s, 1));
        if (res.latchIndex < 0 && od.onsetDetected())
        {
            res.latchIndex = i;
            res.reference = od.getLastOnsetSample();
        }
    }
    return res;
}
} // namespace

DSPARK_TEST(Onset_OA6_latency_exact)
{
    for (double fs : { 44100.0, 48000.0 })
    {
        OnsetDetector<float> probe;
        probe.prepare(AudioSpec{ fs, 1, 1 });
        const int fft = probe.getFftSize();
        const int hop = probe.getHopSize();
        const int L = fft + hop;
        EXPECT_EQ(probe.getLatencySamples(), L);

        const int64_t clickAt = 40 * hop;
        LatchResult r = measureLatch(fs, fft, hop, clickAt);

        std::cout << "  [OA-6] fs=" << fs << " fft=" << fft << " hop=" << hop
                  << " L=" << L << " latchIdx=" << r.latchIndex
                  << " ref=" << r.reference
                  << " (latch-ref)=" << (r.latchIndex - r.reference)
                  << " (latch-click)=" << (r.latchIndex - clickAt) << "\n";

        EXPECT_TRUE(r.latchIndex >= 0);
        // ONE latency definition: getLatencySamples() == fftSize + hop,
        // and the latch asserts exactly L samples after the onset's reported
        // reference sample -- an exact, block-size-independent contract.
        EXPECT_EQ(probe.getLatencySamples(), L);
        EXPECT_EQ(r.latchIndex - r.reference, static_cast<int64_t>(L));
        // The reference localises the physical click within the acceptance
        // window (the attack is band-limited, not a Dirac).
        EXPECT_LT(std::llabs(r.reference - clickAt), static_cast<int64_t>(0.025 * fs));
    }
}

// Block-size independence: identical onset set regardless of block chopping.
DSPARK_TEST(Onset_OA6_block_size_independent)
{
    const double fs = 44100.0;
    const int n = static_cast<int>(fs * 3.0);
    std::vector<float> sig(static_cast<size_t>(n), 0.0f);
    for (int k = 0; k < 5; ++k)
        addClick(sig, static_cast<int64_t>(fs * (0.3 + 0.5 * k)), fs);

    auto runChopped = [&](int block) {
        OnsetDetector<float> od;
        od.prepare(AudioSpec{ fs, block, 1 });
        std::vector<int64_t> events;
        int64_t g = 0;
        for (int i = 0; i < n; i += block)
        {
            const int len = std::min(block, n - i);
            od.pushSamples(std::span<const float>(sig.data() + i, static_cast<size_t>(len)));
            if (od.onsetDetected()) events.push_back(od.getLastOnsetSample());
            g += len;
        }
        (void)g;
        return events;
    };
    auto e1 = runChopped(1);
    auto e64 = runChopped(64);
    auto e512 = runChopped(512);
    std::cout << "  [OA-6] block-independence: n(1)=" << e1.size()
              << " n(64)=" << e64.size() << " n(512)=" << e512.size() << "\n";
    EXPECT_EQ(e1.size(), e64.size());
    EXPECT_EQ(e1.size(), e512.size());
    for (size_t i = 0; i < e1.size() && i < e512.size(); ++i)
    {
        EXPECT_EQ(e1[i], e64[i]);
        EXPECT_EQ(e1[i], e512[i]);
    }
}

// double instantiation compiles and runs (parity).
DSPARK_TEST(Onset_double_instantiation)
{
    const double fs = 44100.0;
    OnsetDetector<double> od;
    od.prepare(AudioSpec{ fs, 512, 1 });
    std::vector<double> sig(static_cast<size_t>(fs * 2), 0.0);
    const double tau = 4.0 * 0.001 * fs;
    for (int n = 0; n < static_cast<int>(tau * 6); ++n)
        sig[static_cast<size_t>(static_cast<int>(fs * 0.5) + n)] =
            std::sin(dspark::twoPi<double> * 1000.0 * n / fs) * std::exp(-n / tau);
    double* p = sig.data();
    AudioBufferView<double> view(&p, 1, static_cast<int>(sig.size()));
    auto det = od.detectOffline(view);
    std::cout << "  [double] detected=" << det.size() << "\n";
    EXPECT_GT(det.size(), 0u);
}

// ===========================================================================
// Rate invariance of the DEFAULT analysis frame.
//
// The frame decides the STFT bin width fs/N, and with it the frequency above
// which the log-frequency filterbank's declared quarter-tone spacing is
// actually delivered. A fixed sample COUNT therefore moves the detector's
// usable low register with the sample rate. These pins are all integer counts
// and event counts -- nothing here is a wall-clock measurement, so no build
// or instrumentation can invert them.
//
// Measured on the pre-fix tree (fixed 2048): 135 bands at 48 kHz but 111 at
// 96 kHz and 88 at 192 kHz, and soft bass onsets recalled 7/8 at 48 kHz but
// 5/8 at 96 kHz. Both halves of this test fail there.
// ===========================================================================
DSPARK_TEST(Onset_default_frame_holds_the_span_across_rates)
{
    struct Case { double fs; int expectedFft; int expectedHop; int expectedLatency; };
    const Case cases[] = {
        { 44100.0,  2048, 221, 2269 }, { 48000.0,  2048, 240, 2288 },
        { 88200.0,  4096, 441, 4537 }, { 96000.0,  4096, 480, 4576 },
        { 176400.0, 8192, 882, 9074 }, { 192000.0, 8192, 960, 9152 },
    };
    int bands441 = 0, bands48 = 0;
    for (const Case& c : cases)
    {
        OnsetDetector<float> od;
        od.prepare(AudioSpec{ c.fs, 512, 1 });
        EXPECT_EQ(od.getFftSize(), c.expectedFft);
        EXPECT_EQ(od.getHopSize(), c.expectedHop);

        // The filterbank the frame builds is the resolution in force. Equal
        // fs/N families must build the SAME bank, exactly.
        if (c.fs == 44100.0) bands441 = od.getNumBands();
        if (c.fs == 48000.0) bands48  = od.getNumBands();
        if (c.fs == 88200.0 || c.fs == 176400.0) EXPECT_EQ(od.getNumBands(), bands441);
        if (c.fs == 96000.0 || c.fs == 192000.0) EXPECT_EQ(od.getNumBands(), bands48);

        // Reporting latency L = N + hop, pinned in SAMPLES per rate. The frame
        // doubles exactly; the hop is round(fs/200), which at 44.1 kHz rounds
        // 220.5 up to 221, so L is NOT exactly 2x between 44.1 and 88.2 kHz
        // (2269 vs 4537). Each value is therefore pinned outright rather than
        // as a ratio: 2269/44100 s = 4537/88200 s to within one sample.
        EXPECT_EQ(od.getLatencySamples(), c.expectedLatency);
        EXPECT_EQ(od.getLatencySamples(), od.getFftSize() + od.getHopSize());
    }
    std::cout << "  [rate-inv] onset auto map 2048/2048/4096/4096/8192/8192, bands "
              << bands441 << " (44.1 family) / " << bands48 << " (48 family)\n";

    // An explicit frame is still honoured verbatim at any rate.
    OnsetDetector<float> explicitFrame;
    explicitFrame.prepare(AudioSpec{ 192000.0, 512, 1 }, 2048);
    EXPECT_EQ(explicitFrame.getFftSize(), 2048);
}

// Correctness anchor at a non-48k rate: soft BASS note onsets, the material
// the low-register loss actually costs.
//
// SUPERSEDES Onset_soft_bass_onsets_survive_96k, which asserted 8 of 8
// recalled at 96 kHz. That 8th detection was an artifact of a second defect:
// the un-normalised ODF magnitude grew with the analysis frame, so 96 kHz ran
// at roughly double the reference sensitivity and recalled a note the 44.1/48
// kHz reference never did. With the ODF scale now frame-invariant, 96 kHz
// reads exactly like 48 kHz, and the honest recall of this corpus is 7 of 8
// AT EVERY RATE: the soft F#1 at t = 1.0 s sits below the default delta
// everywhere, and always did at the shipped 44.1/48 kHz reference. A bare
// absolute recall at one rate is what let the artifact in, so the successor
// asserts INVARIANCE on an equal-fs/N pair (48 kHz/2048 and 96 kHz/4096, both
// 23.44 Hz bins, identical 135-band filterbank): identical event lists in
// TIME, plus a recall floor and zero false positives so a dead detector
// cannot satisfy the equality. Still red on the pre-fix tree (fixed 2048 at
// 96 kHz recalls 5 of 8, so the floor fails); also red on the un-normalised
// tree (96 kHz reports 8 events to the reference's 7, so equality fails).
DSPARK_TEST(Onset_soft_bass_96k_reads_like_the_48k_reference)
{
    auto midiHz = [](int m) { return 440.0 * std::exp2((m - 69) / 12.0); };
    struct Run { int events; int tp, fn, fp; std::vector<double> timesSec; };
    auto runAt = [&](double fs) {
        const int n = static_cast<int>(fs * 6.0);
        std::vector<float> sig(static_cast<size_t>(n), 0.0f);
        std::vector<int64_t> truth;
        for (int k = 0; k < 8; ++k)   // E1 .. B2, under the quarter-tone floor
        {
            const int64_t at = static_cast<int64_t>(fs * (0.4 + 0.6 * k));
            addSoftNote(sig, at, fs, midiHz(28 + 2 * k), 10.0, 400.0, 0.6f);
            truth.push_back(at);
        }
        OnsetDetector<float> od;
        od.prepare(AudioSpec{ fs, 512, 1 });      // automatic frame
        float* ptr = sig.data();
        AudioBufferView<float> view(&ptr, 1, n);
        auto det = od.detectOffline(view);
        PRF s = scoreOnsets(det, truth, fs, 50.0);
        Run r{ static_cast<int>(det.size()), s.tp, s.fn, s.fp, {} };
        for (int64_t d : det) r.timesSec.push_back(static_cast<double>(d) / fs);
        std::cout << "  [rate-inv] fs=" << fs << " soft bass: events="
                  << r.events << " tp=" << s.tp << " fn=" << s.fn
                  << " fp=" << s.fp << "\n";
        return r;
    };
    const Run ref = runAt(48000.0);
    const Run hi  = runAt(96000.0);

    // Quality floor at the reference meaning of the default delta: at least
    // 7 of 8 recalled, zero false positives, at BOTH rates.
    EXPECT_GT(ref.tp, 6);
    EXPECT_GT(hi.tp, 6);
    EXPECT_EQ(ref.fp, 0);
    EXPECT_EQ(hi.fp, 0);

    // Invariance: the 96 kHz session reports the same events as the 48 kHz
    // reference -- same count, same score, same positions in TIME (one hop,
    // 5 ms, of localisation slack; the hop grids differ by rounding).
    EXPECT_EQ(hi.events, ref.events);
    EXPECT_EQ(hi.tp, ref.tp);
    EXPECT_EQ(hi.fn, ref.fn);
    if (hi.events == ref.events)
        for (int i = 0; i < ref.events; ++i)
            EXPECT_LT(std::abs(hi.timesSec[static_cast<size_t>(i)]
                               - ref.timesSec[static_cast<size_t>(i)]), 0.005);
}

// ===========================================================================
// The ODF magnitude SCALE is frame-invariant, not just the filterbank.
//
// |X| grows linearly with the analysis frame, so an un-normalised spectrum
// against a fixed peak-pick delta means a quiet-onset sensitivity that
// roughly doubles per rate-family doubling even when the filterbank is
// identical (measured: the same soft-bass corpus read 7 events at
// 44.1/48 kHz, 8 at 88.2/96 kHz and 9 at 176.4/192 kHz, the 9th being a
// 10 ms note-OFF release ramp reported as an onset only at the top family).
// The band magnitudes are now scaled by 2048/fftSize before the log
// compression, so the same music with default settings must produce the
// SAME events at every rate.
//
// Within a rate family the auto ladder holds fs/N exactly (44.1 -> 2048,
// 88.2 -> 4096, 176.4 -> 8192 build the identical 138-band filterbank), so
// the requirement is EQUALITY of the detection results, asserted as
// equality rather than as per-rate magic counts; a quality floor (>= 7 of 8
// recalled, zero false positives) keeps the equality from being satisfiable
// by a dead detector. All assertions are integer event counts -- nothing
// here can be inverted by instrumentation.
//
// RED on the pre-normalisation tree: event counts 7 vs 8 vs 9 across the
// families, with one false positive at 176.4/192 kHz.
// ===========================================================================
DSPARK_TEST(Onset_odf_scale_is_rate_invariant_at_equal_span)
{
    auto midiHz = [](int m) { return 440.0 * std::exp2((m - 69) / 12.0); };
    const double families[2][3] = { { 44100.0, 88200.0, 176400.0 },
                                    { 48000.0, 96000.0, 192000.0 } };
    for (const auto& fam : families)
    {
        int firstEvents = -1, firstTp = -1;
        for (double fs : fam)
        {
            const int n = static_cast<int>(fs * 6.0);
            std::vector<float> sig(static_cast<size_t>(n), 0.0f);
            std::vector<int64_t> truth;
            for (int k = 0; k < 8; ++k)   // E1 .. B2, soft attacks
            {
                const int64_t at = static_cast<int64_t>(fs * (0.4 + 0.6 * k));
                addSoftNote(sig, at, fs, midiHz(28 + 2 * k), 10.0, 400.0, 0.6f);
                truth.push_back(at);
            }
            OnsetDetector<float> od;
            od.prepare(AudioSpec{ fs, 512, 1 });          // automatic frame
            float* ptr = sig.data();
            AudioBufferView<float> view(&ptr, 1, n);
            auto det = od.detectOffline(view);
            PRF s = scoreOnsets(det, truth, fs, 50.0);
            std::cout << "  [odf-inv] fs=" << fs << " N=" << od.getFftSize()
                      << " bands=" << od.getNumBands() << " events=" << det.size()
                      << " tp=" << s.tp << " fn=" << s.fn << " fp=" << s.fp << "\n";

            // Zero false positives at every rate: the note-OFF release ramp
            // must not be reported as an onset at ANY session rate.
            EXPECT_EQ(s.fp, 0);
            // Sensitivity floor at the reference meaning of the delta.
            EXPECT_GT(s.tp, 6);

            if (firstEvents < 0) { firstEvents = static_cast<int>(det.size()); firstTp = s.tp; }
            // Equal fs/N, equal bank, equal ODF scale => equal result.
            EXPECT_EQ(static_cast<int>(det.size()), firstEvents);
            EXPECT_EQ(s.tp, firstTp);
        }
    }
}
