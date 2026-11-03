// DSPark Tests - Analog-modelling audit regressions
// Each case pins one measured defect; the comment states the defect and the
// number the unpatched code produced.

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/AnalogRandom.h"
#include "../Core/FFT.h"
#include "../Core/Hysteresis.h"
#include "../Core/WDF.h"
#include "../Effects/Clipper.h"
#include "../Effects/Saturation.h"
#include "../Effects/TapeMachine.h"
#include "../Effects/TransformerModel.h"
#include "../Effects/TubePreamp.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

double toDb(double x) { return 20.0 * std::log10(std::max(x, 1e-30)); }

// Coherent amplitude of frequency f over x[0..n).
double toneAmplitude(const std::vector<double>& x, size_t from, size_t n, double f, double fs)
{
    double re = 0.0, im = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const double w = kTwoPi * f * static_cast<double>(i) / fs;
        re += x[from + i] * std::cos(w);
        im += x[from + i] * std::sin(w);
    }
    return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(n);
}

// Runs a mono sine through a processor in blocks and returns the output.
template <class P>
std::vector<double> runSine(P& p, double fs, int block, size_t total, double f, double amp,
                            long long& t)
{
    std::vector<float> buf(static_cast<size_t>(block));
    std::vector<double> out;
    out.reserve(total + static_cast<size_t>(block));
    while (out.size() < total)
    {
        for (int i = 0; i < block; ++i)
            buf[static_cast<size_t>(i)] = static_cast<float>(
                amp * std::sin(kTwoPi * f * static_cast<double>(t++) / fs));
        float* ch[1] = { buf.data() };
        p.processBlock(AudioBufferView<float>(ch, 1, block));
        for (float v : buf) out.push_back(v);
    }
    out.resize(total);
    return out;
}

// Largest third difference: a click detector that ignores the tone itself.
double maxD3(const std::vector<double>& x)
{
    double m = 0.0;
    for (size_t i = 3; i < x.size(); ++i)
        m = std::max(m, std::abs(x[i] - 3.0 * x[i - 1] + 3.0 * x[i - 2] - x[i - 3]));
    return m;
}

double maxStep(const std::vector<double>& x, size_t from, size_t to)
{
    double m = 0.0;
    for (size_t i = std::max<size_t>(from, 1); i < to && i < x.size(); ++i)
        m = std::max(m, std::abs(x[i] - x[i - 1]));
    return m;
}

// Worst non-harmonic component below 20 kHz relative to the fundamental of a
// coherent tone at odd bin b of an N-point frame (N a power of two).
double worstSpurDbc(const std::vector<double>& frame, int b, double fs)
{
    const size_t n = frame.size();
    FFTReal<double> fft(n);
    std::vector<double> spec(n + 2);
    fft.forward(frame.data(), spec.data());
    const auto mag = [&](size_t k) { return std::hypot(spec[2 * k], spec[2 * k + 1]); };
    const double fund = mag(static_cast<size_t>(b));
    double worst = 0.0;
    const size_t maxBin = static_cast<size_t>(20000.0 / fs * static_cast<double>(n));
    for (size_t k = 1; k <= maxBin; ++k)
        if (k % static_cast<size_t>(b) != 0) worst = std::max(worst, mag(k));
    return toDb(worst / fund);
}

// Median block time of `blocks` calls, optionally changing a parameter first.
template <class P, class Set>
double medianBlockMicros(P& p, int blocks, bool automate, Set set)
{
    constexpr int kBlock = 64;
    std::vector<float> l(kBlock), r(kBlock);
    std::vector<double> times;
    long long t = 0;
    for (int k = 0; k < blocks; ++k)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            const float v = static_cast<float>(0.25 * std::sin(kTwoPi * 440.0 * static_cast<double>(t++) / 48000.0));
            l[static_cast<size_t>(i)] = v;
            r[static_cast<size_t>(i)] = v;
        }
        if (automate) set(k);
        float* ch[2] = { l.data(), r.data() };
        const auto t0 = std::chrono::steady_clock::now();
        p.processBlock(AudioBufferView<float>(ch, 2, kBlock));
        times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

} // namespace

// A drive change ran the whole makeup calibration (two JA solvers over 24 ms
// of internal samples) inside one callback: 1.68 ms for a 64-sample block at
// 48 kHz stereo, 126% of the budget and 10x the idle cost (TransformerModel:
// 0.43 ms, 16x). The calibration is now spread over the following blocks.
DSPARK_TEST(AnalogAudit_automation_does_not_spike_the_callback)
{
    {
        TapeMachine<float> tape;
        tape.prepare(spec(48000.0, 64, 2));
        (void)medianBlockMicros(tape, 200, false, [](int) {});
        const double idle = medianBlockMicros(tape, 400, false, [](int) {});
        const double busy = medianBlockMicros(tape, 300, true,
            [&](int k) { tape.setDrive(static_cast<float>(6.0 * std::sin(0.1 * k))); });
        EXPECT_LT(busy / idle, 3.0);
    }
    {
        TransformerModel<float> xf;
        xf.prepare(spec(48000.0, 64, 2));
        (void)medianBlockMicros(xf, 200, false, [](int) {});
        const double idle = medianBlockMicros(xf, 400, false, [](int) {});
        const double busy = medianBlockMicros(xf, 300, true,
            [&](int k) { xf.setDrive(static_cast<float>(6.0 * std::sin(0.1 * k))); });
        EXPECT_LT(busy / idle, 3.0);
    }
}

// TapeMachine applied drive/makeup/bias as a block-boundary step: drive
// 0 -> +24 dB on a -20 dBFS 220 Hz tone put a click of 1378x the steady
// third-difference level into the output. It now glides geometrically.
DSPARK_TEST(AnalogAudit_tape_drive_change_is_ramped)
{
    const double fs = 48000.0;
    TapeMachine<float> tape;
    tape.setWowFlutter(0.0f);
    tape.prepare(spec(fs, 64, 1));
    long long t = 0;
    (void)runSine(tape, fs, 64, 48000, 220.0, 0.1, t);
    const auto before = runSine(tape, fs, 64, 12000, 220.0, 0.1, t);
    tape.setDrive(24.0f);
    const auto during = runSine(tape, fs, 64, 9600, 220.0, 0.1, t);
    (void)runSine(tape, fs, 64, 48000, 220.0, 0.1, t);
    const auto after = runSine(tape, fs, 64, 12000, 220.0, 0.1, t);
    const double steady = std::max(maxD3(before), maxD3(after));
    EXPECT_LT(maxD3(during) / steady, 20.0);
}

// The field differentiator's pole sat on z = -1: a one-sample 0.001*a tick
// left a permanent alternating dH/dt that erased the remanence (-99.96%
// within 50 ms of zero field). The damped alpha transform keeps it.
DSPARK_TEST(AnalogAudit_hysteresis_tick_keeps_remanence)
{
    const double a = 2.2e4;
    Hysteresis<double> h;
    h.prepare(96000.0);
    const int n = 9600;
    double m = 0.0;
    for (int i = 0; i < n; ++i)
    {
        const double e = std::sin(std::numbers::pi * i / n);
        m = h.processSample(8.0 * a * e * e);
    }
    const double remanence = m;
    (void)h.processSample(0.001 * a);
    for (int i = 0; i < 4800; ++i) m = h.processSample(0.0);
    EXPECT_GT(std::abs(remanence), 1e4);
    EXPECT_LT(std::abs(m / remanence - 1.0), 0.05);
}

// The same undamped mode muted TapeMachine at 1x (its documented use at high
// host rates): after 5 s of -26 dBFS noise at 96 kHz a -20 dBFS tone came
// out at -221 dB until reset().
DSPARK_TEST(AnalogAudit_tape_at_1x_survives_noise)
{
    const double fs = 96000.0;
    TapeMachine<float> tape;
    tape.setOversampling(1);
    tape.setWowFlutter(0.0f);
    tape.prepare(spec(fs, 512, 1));
    long long t = 0;
    const auto y0 = runSine(tape, fs, 512, 3u << 15, 1000.0, 0.1, t);
    std::vector<float> buf(512);
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    for (int k = 0; k < static_cast<int>(fs * 5.0 / 512.0); ++k)
    {
        for (float& v : buf) v = nd(rng);
        float* ch[1] = { buf.data() };
        tape.processBlock(AudioBufferView<float>(ch, 1, 512));
    }
    const auto y1 = runSine(tape, fs, 512, 3u << 15, 1000.0, 0.1, t);
    const double before = toDb(toneAmplitude(y0, 2u << 15, 1u << 15, 1000.0, fs));
    const double after = toDb(toneAmplitude(y1, 2u << 15, 1u << 15, 1000.0, fs));
    EXPECT_GT(before, -21.0);
    EXPECT_LT(std::abs(after - before), 0.5);
}

// prepare() reset every Saturation smoother to its default and never
// re-applied the stored parameters: a second prepare() with the same spec
// lost drive +24, mix 0.5 and output -6 (-9.8 dB) while getState() still
// reported them.
DSPARK_TEST(AnalogAudit_saturation_reprepare_keeps_parameters)
{
    const double fs = 48000.0;
    Saturation<float> sat;
    sat.prepare(spec(fs, 512, 1));
    sat.setDrive(24.0f);
    sat.setMix(0.5f);
    sat.setOutputGain(-6.0f);
    long long t = 0;
    const auto a = runSine(sat, fs, 512, 30720, 1000.0, 0.1, t);
    sat.prepare(spec(fs, 512, 1));
    const auto b = runSine(sat, fs, 512, 30720, 1000.0, 0.1, t);
    const double ra = toDb(toneAmplitude(a, 15360, 15360, 1000.0, fs));
    const double rb = toDb(toneAmplitude(b, 15360, 15360, 1000.0, fs));
    EXPECT_LT(std::abs(ra - rb), 0.05);
}

// Blocks longer than maxBlockSize: the oversampled paths of TubePreamp,
// TapeMachine, Saturation and Clipper left the tail of the block
// unprocessed (a clipped signal's tail kept its 0.30 input peak), and the 1x
// paths of TubePreamp and Saturation (drift on) wrote past their scratch
// buffers (AddressSanitizer: heap-buffer-overflow). One 1024-sample block must
// equal four 256-sample blocks.
DSPARK_TEST(AnalogAudit_oversize_blocks_equal_split_blocks)
{
    const auto check = [](auto make) {
        auto a = make();
        auto b = make();
        std::vector<float> x(4096), ya, yb;
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = static_cast<float>(0.3 * std::sin(0.05 * static_cast<double>(i)));
        ya = x;
        yb = x;
        for (size_t off = 0; off < x.size(); off += 1024)
        {
            float* ch[1] = { ya.data() + off };
            a->processBlock(AudioBufferView<float>(ch, 1, 1024));
        }
        for (size_t off = 0; off < x.size(); off += 256)
        {
            float* ch[1] = { yb.data() + off };
            b->processBlock(AudioBufferView<float>(ch, 1, 256));
        }
        double worst = 0.0;
        for (size_t i = 0; i < x.size(); ++i)
            worst = std::max(worst, static_cast<double>(std::abs(ya[i] - yb[i])));
        return worst;
    };
    const auto s = spec(48000.0, 256, 1);
    EXPECT_LT(check([&] { auto p = std::make_unique<TubePreamp<float>>(); p->prepare(s); p->setDrive(12.0f); return p; }), 1e-6);
    EXPECT_LT(check([&] { auto p = std::make_unique<TubePreamp<float>>(); p->setOversampling(1); p->prepare(s); return p; }), 1e-6);
    EXPECT_LT(check([&] { auto p = std::make_unique<TapeMachine<float>>(); p->setWowFlutter(0.0f); p->prepare(s); return p; }), 1e-6);
    EXPECT_LT(check([&] { auto p = std::make_unique<Saturation<float>>(); p->prepare(s); p->setDrive(12.0f); return p; }), 1e-6);
    EXPECT_LT(check([&] { auto p = std::make_unique<Saturation<float>>(); p->setOversampling(1); p->prepare(s); p->setAnalogDrift(0.8f); return p; }), 1e-6);
    EXPECT_LT(check([&] { auto p = std::make_unique<Clipper<float>>(); p->setOversampling(2); p->prepare(s); p->setInputGain(24.0f); return p; }), 1e-6);
}

// Changing the Clipper stage count kept the old per-stage gain: 1 -> 4 stages
// at +18 dB drove a -20 dBFS tone to the ceiling for 20 ms (a 1.94 step
// against 0.023 steady).
DSPARK_TEST(AnalogAudit_clipper_stage_change_keeps_total_gain)
{
    const double fs = 48000.0;
    Clipper<float> clip;
    clip.prepare(spec(fs, 64, 1));
    clip.setInputGain(18.0f);
    long long t = 0;
    (void)runSine(clip, fs, 64, 9600, 220.0, 0.1, t);
    const auto before = runSine(clip, fs, 64, 9600, 220.0, 0.1, t);
    clip.setStages(4);
    const auto after = runSine(clip, fs, 64, 9600, 220.0, 0.1, t);
    EXPECT_LT(maxStep(after, 0, after.size()), 1.5 * maxStep(before, 0, before.size()));
}

// The HardClip ADAA passed unclipped pairs through un-averaged, switching
// between a 0 and a 0.5 sample delay at every clip edge: at +12 dB drive and
// 2x the antialiased path aliased MORE than the plain one (worst of three
// -6 dBFS tones: -24.6 dBc against -30.1 dBc with ADAA off).
DSPARK_TEST(AnalogAudit_hardclip_adaa_reduces_aliasing)
{
    const double fs = 48000.0;
    const int n = 16384;
    const auto worst = [&](float drive, bool adaa) {
        double w = -400.0;
        for (double f : { 5000.0, 9000.0, 15250.0 })
        {
            int b = static_cast<int>(std::lround(f / fs * n));
            if ((b & 1) == 0) ++b;
            Saturation<float> sat;
            sat.prepare(spec(fs, 512, 1));
            sat.setAlgorithm(Saturation<float>::Algorithm::HardClip);
            sat.setDrive(drive);
            sat.setAntialiasing(adaa);
            long long t = 0;
            const auto y = runSine(sat, fs, 512, 14400 + n, b * fs / n, 0.5, t);
            w = std::max(w, worstSpurDbc(std::vector<double>(y.begin() + 14400, y.end()), b, fs));
        }
        return w;
    };
    EXPECT_LT(worst(6.0f, true), -100.0);
    EXPECT_LT(worst(12.0f, true), worst(12.0f, false) - 3.0);
}

// TransformerModel::reset() assigned {} to the bell section, wiping its
// coefficients: at resonance 1 a 15 kHz tone lost 2.2 dB until the next
// parameter change.
DSPARK_TEST(AnalogAudit_transformer_reset_keeps_resonance)
{
    const double fs = 48000.0;
    TransformerModel<float> xf;
    xf.setResonance(1.0f);
    xf.prepare(spec(fs, 512, 1));
    long long t = 0;
    const auto a = runSine(xf, fs, 512, 48000, 15000.0, 0.05, t);
    xf.reset();
    const auto b = runSine(xf, fs, 512, 48000, 15000.0, 0.05, t);
    const double ga = toneAmplitude(a, 24000, 24000, 15000.0, fs);
    const double gb = toneAmplitude(b, 24000, 24000, 15000.0, fs);
    EXPECT_LT(std::abs(toDb(gb / ga)), 0.05);
}

// The Saturation Tube curve switched between tanh(x) and tanh(1.15x) at
// x = 0: a slope kink whose 2nd harmonic stayed at -30.6 dBc at every level,
// down to -60 dBFS. A smooth curve's harmonics fall with level.
DSPARK_TEST(AnalogAudit_saturation_tube_low_level_is_clean)
{
    const double fs = 48000.0;
    const int n = 32768;
    int b = static_cast<int>(std::lround(1000.0 / fs * n));
    if ((b & 1) == 0) ++b;
    const double f = b * fs / n;
    Saturation<float> sat;
    sat.prepare(spec(fs, 512, 1));
    sat.setAlgorithm(Saturation<float>::Algorithm::Tube);
    long long t = 0;
    const auto y = runSine(sat, fs, 512, 24000 + static_cast<size_t>(n), f, 0.001, t);
    const double h1 = toneAmplitude(y, 24000, static_cast<size_t>(n), f, fs);
    const double h2 = toneAmplitude(y, 24000, static_cast<size_t>(n), 2.0 * f, fs);
    EXPECT_LT(toDb(h2 / h1), -60.0);
}

// The drive, character, drift and crossfade smoothers were prepared at the
// base rate but advanced at the oversampled rate: the drive t63 at 1x and at
// the default 2x differed by 21.8 ms. It must not depend on the factor.
DSPARK_TEST(AnalogAudit_saturation_smoothing_ignores_oversampling)
{
    const double fs = 48000.0;
    const auto t63 = [&](int factor) {
        Saturation<float> sat;
        sat.setOversampling(factor);
        sat.prepare(spec(fs, 64, 1));
        sat.setAlgorithm(Saturation<float>::Algorithm::HardClip);
        long long t = 0;
        (void)runSine(sat, fs, 64, 24000, 4000.0, 0.01, t);
        sat.setDrive(12.0f);
        const auto y = runSine(sat, fs, 64, 9600, 4000.0, 0.01, t);
        // envelope per 4 kHz period (12 samples)
        const double target = 0.01 * (1.0 + 0.63 * (std::pow(10.0, 12.0 / 20.0) - 1.0));
        const double latency = static_cast<double>(sat.getLatency());
        for (size_t i = 0; i + 12 <= y.size(); i += 12)
        {
            double pk = 0.0;
            for (size_t j = i; j < i + 12; ++j) pk = std::max(pk, std::abs(y[j]));
            if (pk >= target) return (static_cast<double>(i) - latency) * 1000.0 / fs;
        }
        return 1e9;
    };
    EXPECT_LT(std::abs(t63(1) - t63(2)), 3.0);
}

// During an algorithm crossfade both algorithms advanced the drive and
// character smoothers, so a preset change (algorithm + drive) depended on the
// host's block partition: up to 0.28 of full scale between 512-sample and
// 1-sample blocks.
DSPARK_TEST(AnalogAudit_saturation_crossfade_is_partition_independent)
{
    const double fs = 48000.0;
    const int total = 24000, change = 9600;
    std::vector<float> x(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i)
        x[static_cast<size_t>(i)] = static_cast<float>(0.4 * std::sin(kTwoPi * 220.0 * i / fs)
                                                     + 0.2 * std::sin(kTwoPi * 5100.0 * i / fs));
    const auto run = [&](int block) {
        Saturation<float> sat;
        sat.prepare(spec(fs, 512, 1));
        sat.setDrive(12.0f);
        std::vector<float> y = x;
        for (int off = 0; off < total;)
        {
            if (off == change)
            {
                sat.setAlgorithm(Saturation<float>::Algorithm::HardClip);
                sat.setDrive(24.0f);
            }
            int bs = std::min(block, total - off);
            if (off < change) bs = std::min(bs, change - off);
            float* ch[1] = { y.data() + off };
            sat.processBlock(AudioBufferView<float>(ch, 1, bs));
            off += bs;
        }
        return y;
    };
    const auto a = run(512);
    const auto b = run(1);
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::abs(a[i] - b[i])));
    EXPECT_LT(worst, 1e-4);
}

// The loss FIR was fixed at 63 taps, so at 192 kHz its design grid smeared the
// reproduce-gap cut, and the over-bias one-pole fell short at 18 kHz at
// 48 kHz: 18 kHz read 2.1 dB apart between 48 and 192 kHz at the default
// bias and 3.9 dB apart at bias 1. Both now follow the rate.
DSPARK_TEST(AnalogAudit_tape_response_is_rate_independent)
{
    const auto gain18k = [](double fs, float bias) {
        TapeMachine<float> tape;
        tape.setWowFlutter(0.0f);
        tape.setBias(bias);
        tape.prepare(spec(fs, 512, 1));
        long long t = 0;
        const auto settle = static_cast<size_t>(fs * 0.25);
        const auto y = runSine(tape, fs, 512, 2 * settle, 18000.0, 0.0316, t);
        return toDb(toneAmplitude(y, settle, settle, 18000.0, fs) / 0.0316);
    };
    EXPECT_LT(std::abs(gain18k(48000.0, 0.5f) - gain18k(192000.0, 0.5f)), 0.5);
    EXPECT_LT(std::abs(gain18k(48000.0, 1.0f) - gain18k(192000.0, 1.0f)), 1.0);
}

// A sag change moved the supply at once and thumped: 0.3 -> 1 on a -20 dBFS
// tone stepped the output by 0.0975, 32x its steady sample delta.
DSPARK_TEST(AnalogAudit_tube_sag_change_is_ramped)
{
    const double fs = 48000.0;
    TubePreamp<float> tube;
    tube.prepare(spec(fs, 64, 1));
    long long t = 0;
    (void)runSine(tube, fs, 64, 24000, 220.0, 0.1, t);
    const auto before = runSine(tube, fs, 64, 9600, 220.0, 0.1, t);
    tube.setSag(1.0f);
    const auto after = runSine(tube, fs, 64, 9600, 220.0, 0.1, t);
    EXPECT_LT(maxStep(after, 0, after.size()), 2.0 * maxStep(before, 0, before.size()));
}

// AnalogRandom: setRateHz(NaN) froze the phase as NaN forever (0 retriggers
// per second after restoring 100 Hz); setRange(NaN) with smoothing and
// setQuantization(inf) latched the output at NaN; getNextDiscreteInt over the
// full int range overflowed and always returned INT_MIN.
DSPARK_TEST(AnalogAudit_analog_random_rejects_nonfinite_inputs)
{
    using Gen = AnalogRandom::Generator<float>;
    {
        Gen g(1);
        g.prepare(48000.0);
        g.setRateHz(100.0f);
        g.setRateHz(std::numeric_limits<float>::quiet_NaN());
        for (int i = 0; i < 100; ++i) (void)g.getNextSample();
        g.setRateHz(100.0f);
        int changes = 0;
        float last = g.getNextSample();
        for (int i = 0; i < 48000; ++i)
        {
            const float v = g.getNextSample();
            if (std::abs(v - last) > 1e-6f) ++changes;
            last = v;
        }
        EXPECT_GT(changes, 50);
    }
    {
        Gen g(1);
        g.prepare(48000.0);
        g.setRateHz(1000.0f);
        g.setSmoothing(true, 10.0f);
        g.setRange(std::numeric_limits<float>::quiet_NaN(), 1.0f);
        g.setQuantization(std::numeric_limits<float>::infinity());
        float v = 0.0f;
        for (int i = 0; i < 4800; ++i) v = g.getNextSample();
        EXPECT_TRUE(std::isfinite(v));
    }
    {
        AnalogRandom::Generator<double> g(1);
        g.prepare(48000.0);
        g.setRateHz(48000.0);
        int atMin = 0;
        for (int i = 0; i < 64; ++i)
            if (g.getNextDiscreteInt(INT_MIN, INT_MAX) == INT_MIN) ++atMin;
        EXPECT_LT(atMin, 8);
    }
}

// Saturation::setOversampling(32) was accepted: it reported 32 while the
// oversampler ran 16x and the curves' filters were designed for 32x.
DSPARK_TEST(AnalogAudit_saturation_rejects_unsupported_factor)
{
    Saturation<float> sat;
    sat.setOversampling(32);
    sat.prepare(spec(48000.0, 64, 1));
    EXPECT_EQ(sat.getOversamplingFactor(), 2);
}

// Enabling Saturation ADAA mid-stream averaged against a stale previous input:
// a 0.031 step on a -20 dBFS tone at +12 dB, 2.7x its steady sample delta.
DSPARK_TEST(AnalogAudit_saturation_adaa_enable_is_click_free)
{
    const double fs = 48000.0;
    Saturation<float> sat;
    sat.setOversampling(1);
    sat.prepare(spec(fs, 64, 1));
    sat.setDrive(12.0f);
    long long t = 0;
    (void)runSine(sat, fs, 64, 24000, 220.0, 0.1, t);
    const auto before = runSine(sat, fs, 64, 9600, 220.0, 0.1, t);
    sat.setAntialiasing(true);
    const auto after = runSine(sat, fs, 64, 9600, 220.0, 0.1, t);
    EXPECT_LT(maxStep(after, 0, after.size()), 2.0 * maxStep(before, 0, before.size()));
}

// TapeMachine::setNoise(dBFS) produced an RMS 13.9 dB below the setting
// (setNoise(-55) measured -68.9 dBFS).
DSPARK_TEST(AnalogAudit_tape_hiss_level_matches_setting)
{
    TapeMachine<float> tape;
    tape.setWowFlutter(0.0f);
    tape.setNoise(-50.0f);
    tape.prepare(spec(48000.0, 512, 1));
    std::vector<float> buf(512);
    double acc = 0.0;
    int count = 0;
    for (int k = 0; k < 400; ++k)
    {
        std::fill(buf.begin(), buf.end(), 0.0f);
        float* ch[1] = { buf.data() };
        tape.processBlock(AudioBufferView<float>(ch, 1, 512));
        if (k > 200)
            for (float v : buf) { acc += static_cast<double>(v) * v; ++count; }
    }
    EXPECT_NEAR(10.0 * std::log10(acc / count), -50.0, 0.5);
}

// One NaN source voltage stayed in a WDF tree's capacitor states for good
// (the diode clipper and the FMV stack returned NaN after 1000 clean samples).
DSPARK_TEST(AnalogAudit_wdf_recovers_after_nonfinite_input)
{
    using namespace dspark::wdf;
    ResistiveVoltageSource<float> vs { 2200.0f };
    Capacitor<float> cap { 10e-9f };
    Parallel<float, decltype(vs), decltype(cap)> tree { vs, cap };
    DiodePairRoot<float, decltype(tree)> root { tree };
    root.prepare(48000.0);
    vs.setVoltage(std::numeric_limits<float>::quiet_NaN());
    root.process();
    vs.setVoltage(0.1f);
    for (int i = 0; i < 1000; ++i) root.process();
    EXPECT_TRUE(std::isfinite(root.getVoltage()));
    EXPECT_NEAR(root.getVoltage(), 0.1f, 0.02f);

    ToneStackFMV<double> stack;
    stack.prepare(48000.0);
    (void)stack.processSample(std::numeric_limits<double>::quiet_NaN());
    double y = 0.0;
    for (int i = 0; i < 1000; ++i) y = stack.processSample(0.1 * std::sin(0.05 * i));
    EXPECT_TRUE(std::isfinite(y));
}
