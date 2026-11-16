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
#include <complex>
#include <cstddef>
#include <cstdint>
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

namespace {

// Runs a mono sine through a double-precision processor in blocks.
template <class P>
std::vector<double> runSineDouble(P& p, double fs, int block, size_t total, double f, double amp)
{
    std::vector<double> buf(static_cast<size_t>(block)), out;
    out.reserve(total + static_cast<size_t>(block));
    long long t = 0;
    while (out.size() < total)
    {
        for (int i = 0; i < block; ++i)
            buf[static_cast<size_t>(i)] = amp * std::sin(kTwoPi * f * static_cast<double>(t++) / fs);
        double* ch[1] = { buf.data() };
        p.processBlock(AudioBufferView<double>(ch, 1, block));
        out.insert(out.end(), buf.begin(), buf.end());
    }
    out.resize(total);
    return out;
}

// Worst spur over coherent -6 dBFS tones (7 / 12.75 / 15.25 / 19.25 kHz) after
// a 0.2 s settle; make(f) builds a prepared processor.
template <class Make>
double worstToneSpurDbc(Make make)
{
    const double fs = 48000.0;
    const int n = 16384;
    double w = -400.0;
    for (double f : { 7000.0, 12750.0, 15250.0, 19250.0 })
    {
        int b = static_cast<int>(std::lround(f / fs * n));
        if ((b & 1) == 0) ++b;
        auto p = make();
        const auto y = runSineDouble(*p, fs, 512, 9600 + n, b * fs / n, 0.5);
        w = std::max(w, worstSpurDbc(std::vector<double>(y.begin() + 9600, y.end()), b, fs));
    }
    return w;
}

// Gain of a 0.01 tone at f after 0.25 s; the dry and wet paths are aligned
// exactly when gain(mix 0.5) = (1 + gain(mix 1)) / 2 (the kernels are zero
// phase about their centre, so a slip would turn that into a comb).
template <class Make>
double toneGain(Make make, double f)
{
    const double fs = 48000.0;
    auto p = make();
    const auto y = runSineDouble(*p, fs, 512, 24000, f, 0.01);
    return toneAmplitude(y, 12000, 12000, f, fs) / 0.01;
}

} // namespace

// The Clipper had no antialiasing: hard clipping at 2x and +24 dB aliased at
// -14.5 dBc (worst over 1-19.25 kHz tones at -6/-18 dBFS). Second-order
// antiderivative antialiasing reaches -40.5 dBc there; its delay is padded to
// whole base samples, reported by getLatency() and matched by the dry path.
DSPARK_TEST(AnalogAudit_clipper_second_order_antialiasing)
{
    const auto make = [](bool aa, double mix) {
        return [aa, mix] {
            auto c = std::make_unique<Clipper<double>>();
            c->setMode(Clipper<double>::Mode::Hard);
            c->setOversampling(2);
            c->setInputGain(24.0);
            c->setMix(mix);
            c->setAntialiasing(aa);
            c->prepare(spec(48000.0, 512, 1));
            return c;
        };
    };
    const double off = worstToneSpurDbc(make(false, 1.0));
    const double on = worstToneSpurDbc(make(true, 1.0));
    EXPECT_LT(on, -38.0);
    EXPECT_LT(on, off - 15.0);
    EXPECT_EQ(make(true, 1.0)()->getLatency(), make(false, 1.0)()->getLatency() + 2);
    const auto quiet = [](double mix) {
        return [mix] {
            auto c = std::make_unique<Clipper<double>>();
            c->setMode(Clipper<double>::Mode::Hard);
            c->setOversampling(2);
            c->setMix(mix);
            c->setAntialiasing(true);
            c->prepare(spec(48000.0, 512, 1));
            return c;
        };
    };
    EXPECT_NEAR(toneGain(quiet(0.5), 15000.0), 0.5 * (1.0 + toneGain(quiet(1.0), 15000.0)), 1e-3);
}

// Saturation stopped at first-order antialiasing: SoftClip at 2x and +24 dB
// aliased at -29.7 dBc (-15.2 off). SecondOrder reaches -42.1 dBc and adds
// two base samples at 2x that getLatency() reports and the dry path matches.
DSPARK_TEST(AnalogAudit_saturation_second_order_antialiasing)
{
    using Sat = Saturation<double>;
    const auto make = [](Sat::Antialiasing order, double drive, double mix) {
        return [order, drive, mix] {
            auto s = std::make_unique<Sat>();
            s->setAlgorithm(Sat::Algorithm::SoftClip);
            s->setDrive(drive);
            s->setMix(mix);
            s->setDcBlocking(false);
            s->setAntialiasing(order);
            s->prepare(spec(48000.0, 512, 1));
            return s;
        };
    };
    const double first = worstToneSpurDbc(make(Sat::Antialiasing::FirstOrder, 24.0, 1.0));
    const double second = worstToneSpurDbc(make(Sat::Antialiasing::SecondOrder, 24.0, 1.0));
    EXPECT_LT(second, -40.0);
    EXPECT_LT(second, first - 8.0);
    EXPECT_EQ(make(Sat::Antialiasing::SecondOrder, 0.0, 1.0)()->getLatency(),
              make(Sat::Antialiasing::Off, 0.0, 1.0)()->getLatency() + 2);
    EXPECT_NEAR(toneGain(make(Sat::Antialiasing::SecondOrder, 0.0, 0.5), 15000.0),
                0.5 * (1.0 + toneGain(make(Sat::Antialiasing::SecondOrder, 0.0, 1.0), 15000.0)), 1e-3);
}

// SecondOrder had no kernel for the Exciter (a clamped cubic) or the Tape
// curve (anhysteretic magnetisation, whose second antiderivative is not
// elementary), and only the MultiStage tube stage had one: at 2x and +24 dB
// they aliased at -9.8, -7.7 and -6.8 dBc (worst over 1-19.25 kHz tones at
// -6/-18 dBFS). The Exciter kernel is exact per piece, the Tape kernel uses
// tabulated antiderivatives, and MultiStage antialiases every stage within
// the common second-order delay (two base samples at 2x, one otherwise).
DSPARK_TEST(AnalogAudit_saturation_exciter_tape_multistage_second_order)
{
    using Sat = Saturation<double>;
    const auto make = [](Sat::Algorithm algo, int factor, Sat::Antialiasing order) {
        return [algo, factor, order] {
            auto s = std::make_unique<Sat>();
            s->setAlgorithm(algo);
            s->setOversampling(factor);
            s->setDrive(24.0);
            s->setDcBlocking(false);
            s->setAntialiasing(order);
            s->prepare(spec(48000.0, 512, 1));
            return s;
        };
    };
    EXPECT_LT(worstToneSpurDbc(make(Sat::Algorithm::Exciter, 2, Sat::Antialiasing::SecondOrder)), -25.0);
    EXPECT_LT(worstToneSpurDbc(make(Sat::Algorithm::Tape, 2, Sat::Antialiasing::SecondOrder)), -35.0);
    EXPECT_LT(worstToneSpurDbc(make(Sat::Algorithm::MultiStage, 2, Sat::Antialiasing::SecondOrder)), -25.0);
    EXPECT_LT(worstToneSpurDbc(make(Sat::Algorithm::MultiStage, 4, Sat::Antialiasing::SecondOrder)), -45.0);
    for (int factor : { 1, 2, 4, 8 })
        EXPECT_EQ(make(Sat::Algorithm::MultiStage, factor, Sat::Antialiasing::SecondOrder)()->getLatency(),
                  make(Sat::Algorithm::MultiStage, factor, Sat::Antialiasing::Off)()->getLatency()
                      + (factor == 2 ? 2 : 1));

    // The common delay keeps every algorithm aligned: at a low level, where
    // the kernels are transparent at 200 Hz, the SecondOrder output is the
    // Off output two base samples later at 2x and one at the other factors
    // (a slip of one processing sample would leave a -37 dB residual at 2x).
    const auto quiet = [](Sat::Algorithm algo, int factor, Sat::Antialiasing order) {
        auto s = std::make_unique<Sat>();
        s->setAlgorithm(algo);
        s->setOversampling(factor);
        s->setDcBlocking(false);
        s->setAntialiasing(order);
        s->prepare(spec(48000.0, 512, 1));
        return runSineDouble(*s, 48000.0, 512, 24000, 200.0, 0.01);
    };
    for (int factor : { 1, 2, 4, 8 })
        for (auto algo : { Sat::Algorithm::MultiStage, Sat::Algorithm::Tape, Sat::Algorithm::Exciter,
                           Sat::Algorithm::SoftClip })
        {
            const auto off = quiet(algo, factor, Sat::Antialiasing::Off);
            const auto second = quiet(algo, factor, Sat::Antialiasing::SecondOrder);
            const size_t shift = factor == 2 ? 2 : 1;
            double err = 0.0, ref = 0.0;
            for (size_t i = 12000; i < off.size(); ++i)
            {
                err += (second[i] - off[i - shift]) * (second[i] - off[i - shift]);
                ref += off[i] * off[i];
            }
            EXPECT_LT(10.0 * std::log10(err / ref), -55.0);
        }
}

// Within one base sample of second-order delay, MultiStage at 2x had to keep
// its transformer stage at first order and aliased at -28.0 dBc at +24 dB
// (worst over 1-19.25 kHz tones at -6/-18 dBFS). With two base samples the
// transformer kernel and a 1.5-sample interpolator fit from +16 dB of drive
// (-36.2 dBc at +24 dB), while the two-sample cascade stays in use below
// +14 dB. Every algorithm takes the same delay, so getLatency() reports two
// samples at once, and the dry, Delta, unprocessed M/S and crossfade paths
// stay aligned.
DSPARK_TEST(AnalogAudit_saturation_multistage_second_order_two_samples)
{
    using Sat = Saturation<double>;
    const auto make = [](double drive) {
        return [drive] {
            auto s = std::make_unique<Sat>();
            s->setAlgorithm(Sat::Algorithm::MultiStage);
            s->setDrive(drive);
            s->setDcBlocking(false);
            s->setAntialiasing(Sat::Antialiasing::SecondOrder);
            s->prepare(spec(48000.0, 512, 1));
            return s;
        };
    };
    EXPECT_LT(worstToneSpurDbc(make(24.0)), -33.0);
    EXPECT_LT(worstToneSpurDbc(make(36.0)), -32.0);
    EXPECT_LT(worstToneSpurDbc(make(12.0)), -35.0);

    // Every algorithm reports the new latency as soon as the order is set.
    for (int a = 0; a <= static_cast<int>(Sat::Algorithm::MultiStage); ++a)
    {
        Sat s;
        s.setAlgorithm(static_cast<Sat::Algorithm>(a));
        s.prepare(spec(48000.0, 512, 1));
        const int off = s.getLatency();
        s.setAntialiasing(Sat::Antialiasing::SecondOrder);
        EXPECT_EQ(s.getLatency(), off + 2);
        EXPECT_EQ(s.getLatencySamples(), off + 2);
    }

    // Dry, Delta, unprocessed M/S and crossfade paths: at a low level the
    // kernels are transparent at 200 Hz, so each mode's SecondOrder output
    // is its Off output two base samples later (a slip of one processing
    // sample would leave about -37 dB, of half a sample about -43 dB).
    enum class Case { Switch, MidOnly, Delta, HalfMix };
    const auto run = [](Case c, Sat::Antialiasing order) {
        Sat s;
        s.setAlgorithm(c == Case::Switch ? Sat::Algorithm::SoftClip : Sat::Algorithm::MultiStage);
        s.setDcBlocking(false);
        if (c == Case::MidOnly) s.setProcessingMode(Sat::ProcessingMode::MidOnly);
        if (c == Case::Delta) s.setOutputMode(Sat::OutputMode::Delta);
        if (c == Case::HalfMix) s.setMix(0.5);
        s.setAntialiasing(order);
        s.prepare(spec(48000.0, 512, 2));
        std::vector<double> l(512), r(512), out;
        long long t = 0;
        for (int b = 0; b < 60; ++b)
        {
            if (c == Case::Switch && b == 20) s.setAlgorithm(Sat::Algorithm::MultiStage);
            for (int i = 0; i < 512; ++i, ++t)
            {
                const double ph = kTwoPi * 200.0 * static_cast<double>(t) / 48000.0;
                l[static_cast<size_t>(i)] = 0.01 * std::sin(ph);
                r[static_cast<size_t>(i)] = 0.004 * std::sin(ph + 1.3);
            }
            double* ch[2] = { l.data(), r.data() };
            s.processBlock(AudioBufferView<double>(ch, 2, 512));
            for (int i = 0; i < 512; ++i)
            {
                out.push_back(l[static_cast<size_t>(i)]);
                out.push_back(r[static_cast<size_t>(i)]);
            }
        }
        return out;
    };
    for (auto c : { Case::Switch, Case::MidOnly, Case::Delta, Case::HalfMix })
    {
        const auto off = run(c, Sat::Antialiasing::Off);
        const auto second = run(c, Sat::Antialiasing::SecondOrder);
        double err = 0.0, ref = 0.0;
        for (size_t i = 2 * 4000; i < off.size(); ++i)
        {
            err += (second[i] - off[i - 4]) * (second[i] - off[i - 4]);
            ref += off[i] * off[i];
        }
        EXPECT_LT(10.0 * std::log10(err / ref), -55.0);
    }
}

// The Exciter kernel is the exact average of its clamped cubic over the
// linear B-spline of three inputs: against an 8192-cell quadrature of the same
// curve on random knots (in and across the clamp and fold regions, with
// character) it agrees to rounding.
DSPARK_TEST(AnalogAudit_saturation_exciter_kernel_is_exact)
{
    detail::AdaaExciter curve;
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> centre(-4.0, 4.0), unit(-1.0, 1.0), logSpan(-6.0, 1.0);
    double worst = 0.0;
    for (int trial = 0; trial < 400; ++trial)
    {
        curve.setCharacter(0.5 * unit(rng));
        const double c = centre(rng), span = std::pow(10.0, logSpan(rng));
        double u[3] = { c + span * unit(rng), c + span * unit(rng), c + span * unit(rng) };
        const double y = curve.average(u[0], u[1], u[2]);
        std::sort(u, u + 3);
        double ref = 0.0;
        if (u[2] - u[0] < 1e-9)
            ref = curve.f(u[1]);
        else
        {
            const int n = 8192;
            for (int side = 0; side < 2; ++side)
            {
                const double lo = side == 0 ? u[0] : u[1], hi = side == 0 ? u[1] : u[2];
                if (!(hi > lo)) continue;
                for (int k = 0; k < n; ++k)
                {
                    // Midpoint rule on n cells plus the exact weight.
                    const double t = lo + (hi - lo) * (k + 0.5) / n;
                    const double w = side == 0 ? 2.0 * (t - u[0]) / ((u[2] - u[0]) * (u[1] - u[0]))
                                               : 2.0 * (u[2] - t) / ((u[2] - u[0]) * (u[2] - u[1]));
                    ref += curve.f(t) * w * (hi - lo) / n;
                }
            }
        }
        worst = std::max(worst, std::abs(y - ref));
    }
    EXPECT_LT(worst, 1e-6);
}

// The diode-pair and single-diode roots stopped Newton at an absolute 1e-10 V
// residual, so small signals came back with a large relative error: with a
// 2.2 k ohm source and the default 1N4148 constants 2.4e-4 (pair) and 1.2e-4
// (single diode). Both now return the root of the node equation to rounding.
DSPARK_TEST(AnalogAudit_wdf_diode_roots_are_exact_at_every_drive)
{
    using namespace dspark::wdf;
    const double r = 2200.0, is = 2.52e-9, nvt = 1.752 * 0.02585;
    const auto exact = [&](double vin, bool pair) {
        double lo = std::min(0.0, vin), hi = std::max(0.0, vin);
        const auto f = [&](double v) {
            return (vin - v) / r - (pair ? 2.0 * is * std::sinh(v / nvt) : is * std::expm1(v / nvt));
        };
        for (int i = 0; i < 3000; ++i)
        {
            const double m = 0.5 * (lo + hi);
            if (m <= lo || m >= hi) break;
            (f(m) > 0.0 ? lo : hi) = m;
        }
        return std::abs(f(lo)) < std::abs(f(hi)) ? lo : hi;
    };
    ResistiveVoltageSource<double> sourcePair { r }, sourceSingle { r };
    DiodePairRoot<double, decltype(sourcePair)> pair { sourcePair };
    DiodeRoot<double, decltype(sourceSingle)> single { sourceSingle };
    pair.prepare(48000.0);
    single.prepare(48000.0);
    double worstPair = 0.0, worstSingle = 0.0;
    for (double decade = -10.0; decade <= 1.0; decade += 0.01)
        for (double sign : { 1.0, -1.0 })
        {
            const double vin = sign * std::pow(10.0, decade);
            sourcePair.setVoltage(vin);
            sourceSingle.setVoltage(vin);
            pair.process();
            single.process();
            const double ep = exact(vin, true), es = exact(vin, false);
            worstPair = std::max(worstPair, std::abs(pair.getVoltage() - ep) / std::abs(ep));
            worstSingle = std::max(worstSingle, std::abs(single.getVoltage() - es) / std::abs(es));
        }
    EXPECT_LT(worstPair, 1e-12);
    EXPECT_LT(worstSingle, 1e-12);
}

// Speed-up pin: at low drive the diode roots now evaluate an exact series
// reversion of the node equation instead of the omega estimate and its
// correction step (pair 30 -> 14.7 ns, single 26.6 -> 11.9 ns per sample in
// an RC circuit). The output must stay the root to rounding: these are
// 50-digit roots of s + 2c sinh(s) = x (pair) and s + c (e^s - 1) = x
// (single), with nVt = 1 and R = 1 so that c = Is and x is the source.
DSPARK_TEST(AnalogAudit_wdf_low_drive_series_roots_are_exact)
{
    using namespace dspark::wdf;
    struct Point { int pair; double c, x, s; };
    static constexpr Point kPoints[] = {
        { 1, 3.93e-05, 1e-07, 9.999214061774745e-08 },
        { 1, 3.93e-05, 0.001, 0.0009999214061643786 },
        { 1, 3.93e-05, 0.05, 0.04999606867168401 },
        { 1, 3.93e-05, 0.3, 0.2999760666713768 },
        { 1, 3.93e-05, 0.4, 0.3999677176102836 },
        { 1, 3.93e-05, 1.0, 0.9999076403877263 },
        { 1, 0.000122, 1e-07, 9.997560595214768e-08 },
        { 1, 0.000122, 0.001, 0.0009997560594808498 },
        { 1, 0.000122, 0.05, 0.04998779789706584 },
        { 1, 0.000122, 0.3, 0.2999257159952651 },
        { 1, 0.000122, 0.4, 0.3998998028621683 },
        { 1, 0.0195, 1e-07, 9.624639076034648e-08 },
        { 1, 0.0195, 0.001, 0.0009624639020258159 },
        { 1, 0.0195, 0.05, 0.048122498123646704 },
        { 1, 0.0195, 0.3, 0.28858818452243246 },
        { 1, 0.0195, 0.4, 0.3846269477553365 },
        { 0, 3.93e-05, 1e-07, 9.999607015424645e-08 },
        { 0, 3.93e-05, -1e-07, -9.99960701546394e-08 },
        { 0, 3.93e-05, 0.001, 0.000999960681890196 },
        { 0, 3.93e-05, -0.001, -0.0009999607211855662 },
        { 0, 3.93e-05, 0.05, 0.04999798512915664 },
        { 0, 3.93e-05, -0.05, -0.04999808338803226 },
        { 0, 3.93e-05, 0.3, 0.2999862512782194 },
        { 0, 3.93e-05, -0.3, -0.2999898144526179 },
        { 0, 3.93e-05, 0.4, 0.3999806724225227 },
        { 0, 3.93e-05, -0.4, -0.39998704391912093 },
        { 0, 3.93e-05, 1.0, 0.9999324787370917 },
        { 0, 3.93e-05, -1.0, -0.9999751580211994 },
        { 0, 0.000122, 1e-07, 9.998780148760865e-08 },
        { 0, 0.000122, -1e-07, -9.998780148882821e-08 },
        { 0, 0.000122, 0.001, 0.0009998779538841838 },
        { 0, 0.000122, -0.001, -0.000999878075839553 },
        { 0, 0.000122, 0.05, 0.04999374572838169 },
        { 0, 0.000122, -0.05, -0.049994050680209644 },
        { 0, 0.000122, 0.3, 0.29995732425328614 },
        { 0, 0.000122, -0.3, -0.29996838268053605 },
        { 0, 0.000122, 0.4, 0.3999400083052055 },
        { 0, 0.000122, -0.4, -0.3999597823346448 },
        { 0, 0.0195, 1e-07, 9.808729760293682e-08 },
        { 0, 0.0195, -1e-07, -9.808729778696017e-08 },
        { 0, 0.0195, 0.001, 0.0009808637729458586 },
        { 0, 0.0195, -0.001, -0.0009808821752815766 },
        { 0, 0.0195, 0.05, 0.04902028767895538 },
        { 0, 0.0195, -0.05, -0.04906630102355141 },
        { 0, 0.0195, 0.3, 0.29335215900997086 },
        { 0, 0.0195, -0.3, -0.2950181031300965 },
        { 0, 0.0195, 0.4, 0.39067930317657545 },
        { 0, 0.0195, -0.4, -0.3936544488463956 }

    };
    double worst = 0.0;
    for (const auto& p : kPoints)
    {
        ResistiveVoltageSource<double> source { 1.0 };
        double v = 0.0;
        if (p.pair)
        {
            DiodePairRoot<double, decltype(source)> root { source, p.c, 1.0 };
            root.prepare(48000.0);
            source.setVoltage(p.x);
            root.process();
            v = root.getVoltage();
        }
        else
        {
            DiodeRoot<double, decltype(source)> root { source, p.c, 1.0 };
            root.prepare(48000.0);
            source.setVoltage(p.x);
            root.process();
            v = root.getVoltage();
        }
        worst = std::max(worst, std::abs(v - p.s) / std::abs(p.s));
    }
    EXPECT_LT(worst, 4.5e-16);
}

// Speed-up pin: above the low-drive series the roots now come from a
// full-precision omega table (the forward diode's closed form and, for the
// pair, the reverse diode's correction from the same table point) instead of
// an omega estimate and one correction step with an exponential and a
// division (pair 28-32 -> 19-25 ns, single 26-27 -> 13-16 ns per sample in an
// RC circuit). The output must stay the root to rounding: 50-digit roots of
// s + 2c sinh(s) = x (pair) and s + c (e^s - 1) = x (single), nVt = 1 and
// R = 1 so that c = Is and x is the source, across every table region (below
// and above omega = 0.57, and the octave cells beyond ln c + x = 32). For
// 1e-4 < c <= 0.02 the pair's table now starts where c^2 / (omega (1 + omega))
// reaches 3.5e-3 instead of 3e-5 (behind 94 k ohm, 0.5 V peak: 28.6 -> 22.6 ns),
// so the points at c = 0.00525, 0.01 and 0.0195 from about 2 to 5 nVt come
// from it (below, the longer series of the next test answers).
DSPARK_TEST(AnalogAudit_wdf_table_roots_are_exact)
{
    using namespace dspark::wdf;
    struct Point { int pair; double c, x, s; };
    static constexpr Point kPoints[] = {
        { 1, 0.00525, 0.7, 0.69213823836038093 },
        { 1, 0.00525, 1.5, 1.4781763352734014 },
        { 1, 0.00525, 3.0, 2.9044479806151392 },
        { 1, 0.00525, 5.0, 4.5185965423662986 },
        { 1, 0.00525, 10.0, 6.5017828948455905 },
        { 1, 0.01, 1.2, 1.1708535808709835 },
        { 1, 0.01, 2.5, 2.391604626842606 },
        { 1, 0.0195, 1.6, 1.5155238207326209 },
        { 1, 0.0195, 4.0, 3.4102967120194546 },
        // The first table nodes on which the wide root takes its short
        // correction (45-digit roots).
        { 1, 0.00525, 5.144058772540771, 4.6142971866928367 },
        { 1, 0.00525, 5.370317963702527, 4.7584166104719260 },
        { 1, 0.008, 6.309573444801933, 5.0550834853291726 },
        { 1, 0.0195, 9.586318637548251, 5.3751115742403040 },
        { 1, 0.0195, 9.804753956715794, 5.4163305803934470 },
        // The wide table root from where it starts above the longer series:
        // through its full correction, across the node on which the short
        // one takes over and beyond it (50-digit roots).
        { 1, 0.001664, 2.5481612075009616, 2.5274580938018504546 },
        { 1, 0.001664, 2.9351200693163801, 2.904823195193435823 },
        { 1, 0.001664, 3.3220789311317986, 3.2780076038522969377 },
        { 1, 0.001664, 3.7090377929472176, 3.6453560256989865385 },
        { 1, 0.001664, 3.7835309365831202, 3.7152339780483452846 },
        { 1, 0.001664, 5.0735309365831203, 4.8590511839774799304 },
        { 1, 0.005184, 2.3044653400129285, 2.2555505242892660767 },
        { 1, 0.005184, 3.199006281040055, 3.0857923522637703394 },
        { 1, 0.005184, 4.0935472220671816, 3.8500365807250339517 },
        { 1, 0.005184, 4.988088163094309, 4.5146224337644161504 },
        { 1, 0.005184, 5.1471783199321628, 4.6207103945937769474 },
        { 1, 0.005184, 6.4371783199321628, 5.3478042488942989922 },
        { 1, 0.008228, 2.0217642616109646, 1.9642559271775075302 },
        { 1, 0.008228, 2.3121856152022811, 2.2360791126679100514 },
        { 1, 0.008228, 2.6026069687935971, 2.5027653619614186845 },
        { 1, 0.008228, 2.850212307191426, 2.7251970084551838596 },
        { 1, 0.008228, 3.9002123071914259, 3.5994673330071171518 },
        { 1, 0.008228, 4.9502123071914266, 4.3271991075284515524 },
        { 1, 0.008228, 6.0002123071914255, 4.8977836357920288521 },
        { 1, 0.008228, 6.185212307191426, 4.9837765289318869648 },
        { 1, 0.008228, 7.475212307191426, 5.4873096035653540177 },
        { 1, 0.0125, 1.9654995142786706, 1.8850626684448883099 },
        { 1, 0.0125, 2.4284004566765418, 2.3044187493853082752 },
        { 1, 0.0125, 2.8913013990744134, 2.7051753854147111132 },
        { 1, 0.0125, 3.2320266346738813, 2.9852621007461257023 },
        { 1, 0.0125, 4.5820266346738814, 3.9397183132965643181 },
        { 1, 0.0125, 5.9320266346738819, 4.6390596507400950635 },
        { 1, 0.0125, 7.2820266346738807, 5.1425990699964181881 },
        { 1, 0.0125, 7.517026634673881, 5.2155897804623480474 },
        { 1, 0.0125, 8.8070266346738819, 5.5598326782093496828 },
        { 1, 0.0178, 1.9208907145608001, 1.8145307194699894478 },
        { 1, 0.0178, 2.5798534288378212, 2.3876784897785404566 },
        { 1, 0.0178, 3.2388161431148426, 2.9122850780699317298 },
        { 1, 0.0178, 3.6910568216840978, 3.2381122036989154442 },
        { 1, 0.0178, 5.4160568216840979, 4.2133502673511959337 },
        { 1, 0.0178, 7.1410568216840984, 4.855310185863756791 },
        { 1, 0.0178, 8.8660568216840971, 5.3000329500215785782 },
        { 1, 0.0178, 9.1635568216840984, 5.3635750363803041232 },
        { 1, 0.0178, 10.453556821684099, 5.6068663951964630288 },
        { 1, 0.0199, 1.9500401191402776, 1.8292707651438060239 },
        { 1, 0.0199, 2.6575936427453497, 2.4326901995071194079 },
        { 1, 0.0199, 3.3651471663504227, 2.9759557684844312152 },
        { 1, 0.0199, 3.8420355472516903, 3.3020949658512854998 },
        { 1, 0.0199, 5.6420355472516901, 4.2488397511675373592 },
        { 1, 0.0199, 7.4420355472516899, 4.8640884956052289336 },
        { 1, 0.0199, 9.2420355472516889, 5.2910308219683262382 },
        { 1, 0.0199, 9.5520355472516894, 5.3521219499682232424 },
        { 1, 0.0199, 10.84203554725169, 5.5779561204825996271 },
        { 1, 3.93e-05, 1.5, 1.4998326695717819 },
        { 1, 3.93e-05, 3.0, 2.9992132173824881 },
        { 1, 3.93e-05, 7.25, 7.1974922424195151 },
        { 1, 3.93e-05, 12.0, 10.529731248222374 },
        { 1, 3.93e-05, 30.0, 12.978748680095542 },
        { 1, 3.93e-05, 45.0, 13.591368774039944 },
        { 1, 3.93e-05, 300.0, 15.793985417731058 },
        { 1, 3.93e-05, 30000.0, 20.452556715344343 },
        { 1, 3.93e-05, 500000.0, 23.266602882204921 },
        { 1, 0.0001, 1.5, 1.4995743443337046 },
        { 1, 0.0001, 3.0, 2.9980004471762358 },
        { 1, 0.0001, 7.25, 7.1256539714894256 },
        { 1, 0.0001, 12.0, 9.9353170525858783 },
        { 1, 0.0001, 30.0, 12.095398139018503 },
        { 1, 0.0001, 45.0, 12.685845719253297 },
        { 1, 0.0001, 300.0, 14.863309054617536 },
        { 1, 0.0001, 30000.0, 19.51864219946787 },
        { 1, 0.0001, 500000.0, 22.332659083064819 },
        { 1, 0.001, 1.5, 1.4957613449999703 },
        { 1, 0.001, 3.0, 2.9803559490065177 },
        { 1, 0.001, 7.25, 6.5504456017814601 },
        { 1, 0.001, 12.0, 8.2338172933251528 },
        { 1, 0.001, 30.0, 9.9080733563191838 },
        { 1, 0.001, 45.0, 10.450158249712992 },
        { 1, 0.001, 300.0, 12.568739018406605 },
        { 1, 0.001, 30000.0, 17.216133903769244 },
        { 1, 0.001, 500000.0, 20.030078595426847 },
        { 1, 0.0199, 7.25, 4.8093616470431551 },
        { 1, 0.0199, 12.0, 5.7496785744283034 },
        { 1, 0.0199, 30.0, 7.0503394449089791 },
        { 1, 0.0199, 45.0, 7.5403014784389244 },
        { 1, 0.0199, 300.0, 9.5883350028141408 },
        { 1, 0.0199, 30000.0, 14.225513911638679 },
        { 1, 0.0199, 500000.0, 17.039364845345638 },
        { 0, 3.93e-05, 1.5, 1.4998631937136309 },
        { 0, 3.93e-05, -1.5, -1.4999694692830223 },
        { 0, 3.93e-05, 3.0, 2.9992505297802921 },
        { 0, 3.93e-05, -7.25, -7.2499607279109499 },
        { 0, 3.93e-05, 12.0, 10.529747156921951 },
        { 0, 3.93e-05, 45.0, 13.591369986677636 },
        { 0, 3.93e-05, -300.0, -299.99996069999997 },
        { 0, 3.93e-05, 300.0, 15.793985555526159 },
        { 0, 3.93e-05, 500000.0, 23.266602882283522 },
        { 0, 0.001, 1.5, 1.4965338183865577 },
        { 0, 0.001, -1.5, -1.4992233035318756 },
        { 0, 0.001, 3.0, 2.981286832132279 },
        { 0, 0.001, -7.25, -7.2490007108844132 },
        { 0, 0.001, 12.0, 8.2340270317346196 },
        { 0, 0.001, 45.0, 10.450186378034756 },
        { 0, 0.001, -300.0, -299.99900000000002 },
        { 0, 0.001, 300.0, 12.568742485419126 },
        { 0, 0.001, 500000.0, 20.030078597426922 },
        { 0, 0.04, 1.5, 1.3808661593177498 },
        { 0, 0.04, -1.5, -1.4692043400197221 },
        { 0, 0.04, 3.0, 2.5352277947271711 },
        { 0, 0.04, -7.25, -7.2100295654120918 },
        { 0, 0.04, 12.0, 5.1490803736122999 },
        { 0, 0.04, 45.0, 6.8611573302330067 },
        { 0, 0.04, -300.0, -299.95999999999998 },
        { 0, 0.04, 300.0, 8.8927051319114927 },
        { 0, 0.04, 500000.0, 16.341206599327862 },
    };
    double worst = 0.0;
    for (const auto& p : kPoints)
    {
        ResistiveVoltageSource<double> source { 1.0 };
        double v = 0.0;
        if (p.pair)
        {
            DiodePairRoot<double, decltype(source)> root { source, p.c, 1.0 };
            root.prepare(48000.0);
            source.setVoltage(p.x);
            root.process();
            v = root.getVoltage();
        }
        else
        {
            DiodeRoot<double, decltype(source)> root { source, p.c, 1.0 };
            root.prepare(48000.0);
            source.setVoltage(p.x);
            root.process();
            v = root.getVoltage();
        }
        worst = std::max(worst, std::abs(v - p.s) / std::abs(p.s));
    }
    EXPECT_LT(worst, 4.5e-16);
}

// Speed-up pin: for 1e-4 < c <= 0.02 the pair's root just above the low-drive
// series now comes from the same reversion series carried to y^35, instead of
// the table root with its full correction or, where the table starts higher
// (c above 0.0062), an omega estimate and a correction step with an
// exponential and a division (RC clipper behind 330 k ohm at 0.1 V peak, four
// machines: 33-44 -> 20-23 ns per sample). The output must stay the root to
// rounding:
// 50-digit roots of s + 2c sinh(s) = x, nVt = 1 and R = 1 so that c = Is and x
// is the source, at 2, 35, 70 and 98 % of the way from the short series' limit
// to the longer one's for seven values of c. The second check pins what makes
// the estimate unnecessary up to c = 0.02: the longer series reaches the drive
// at which the table root starts (c^2 / (omega (1 + omega)) = 3.5e-3) for
// every c in that range.
DSPARK_TEST(AnalogAudit_wdf_longer_series_roots_are_exact)
{
    using namespace dspark::wdf;
    struct Point { double c, x, s; };
    static constexpr Point kPoints[] = {
        { 0.00012, 1.0630061817714715, 1.0627003432684334432 },
        { 0.00012, 1.885255807670104, 1.8844840816837253615 },
        { 0.00012, 2.7573387442292594, 2.7554589828193692014 },
        { 0.00012, 3.4550050934765837, 3.4512242207692066991 },
        { 0.000556, 0.93311315148136798, 0.931920206875992625 },
        { 0.000556, 1.5931098763905771, 1.5904953675279635208 },
        { 0.000556, 2.2931064028094355, 2.2876850674595761473 },
        { 0.000556, 2.8531036239445227, 2.8435854450648909846 },
        { 0.001664, 0.77329394222969783, 0.77046849994259347469 },
        { 0.001664, 1.3492362484946661, 1.3432946663678571984 },
        { 0.001664, 1.9600841490787233, 1.9486413583926753895 },
        { 0.001664, 2.4487624695459695, 2.4300076375756727189 },
        { 0.005184, 0.67065106968902954, 0.66325900470017835264 },
        { 0.005184, 1.1706091895363711, 1.1557743008805544441 },
        { 0.005184, 1.7008678014956726, 1.6741858085976747008 },
        { 0.005184, 2.125074691063114, 2.0840558990230311063 },
        { 0.008228, 0.62761975575103701, 0.61681349022679441231 },
        { 0.008228, 1.0726278427535805, 1.0519430569324319755 },
        { 0.008228, 1.544606116847187, 1.5092087222312235214 },
        { 0.008228, 1.9221887361220724, 1.8700674749546574272 },
        { 0.0178, 0.53363144612086666, 0.5144965418830415019 },
        { 0.0178, 0.94529586924058473, 0.90832583582977136063 },
        { 0.0178, 1.3819096513372555, 1.3200296430168327874 },
        { 0.0178, 1.7312006770145918, 1.6426398440654462347 },
        { 0.0199, 0.53624299283488297, 0.51483529226756224084 },
        { 0.0199, 0.95275308057300712, 0.91125326384792666594 },
        { 0.0199, 1.3945062039316238, 1.3249335227166928306 },
        { 0.0199, 1.7479087026185169, 1.6482946736579264339 },
    };
    double worst = 0.0;
    for (const auto& p : kPoints)
    {
        ResistiveVoltageSource<double> source { 1.0 };
        DiodePairRoot<double, decltype(source)> root { source, p.c, 1.0 };
        root.prepare(48000.0);
        for (double sign : { 1.0, -1.0 })
        {
            source.setVoltage(sign * p.x);
            root.process();
            const double v = root.getVoltage();
            worst = std::max(worst, std::abs(v - sign * p.s) / std::abs(p.s));
        }
    }
    EXPECT_LT(worst, 4.5e-16);

    // Limits in units of nVt, from the expressions of the root itself.
    double leastMargin = 1e30, leastSpan = 1e30;
    for (int i = 1; i <= 400; ++i)
    {
        const double c = std::min(0.02, 1e-4 * std::pow(200.0, i / 400.0));
        double b[17] {};
        const double bound = 1.0 / (1.0 + 2.0 * c), cPlus = 1.0 + 2.0 * c;
        const double shortLimit = cPlus * dspark::wdf::detail::DiodeSeries::pair(2.0 * c * bound, b);
        const double longLimit = cPlus * dspark::wdf::detail::DiodeSeries::pairLong(2.0 * c * bound, b);
        const double w = 0.5 * (std::sqrt(1.0 + 4.0 * c * c / 3.5e-3) - 1.0);
        const double tableStart = w + std::log(w) - std::log(c);
        leastMargin = std::min(leastMargin, longLimit - tableStart);
        leastSpan = std::min(leastSpan, longLimit - shortLimit);
    }
    EXPECT_GT(leastMargin, 0.0);
    EXPECT_GT(leastSpan, 1.0);
}

// Speed-up pin: between the low-drive series and the drives where the reverse
// diode is a short correction, the pair's root now comes from local expansions
// built for the instance (detail::DiodeLocal): cells of sixteen terms about
// points of the curve, found from the bits of the drive. Up to c = 0.02 they
// replace the table root with its full correction; beyond (germanium diodes,
// very large resistances) they replace repeated Halley steps with an
// exponential and a division each, together with a low-drive series built for
// the constant and the table root at high drive (RC clipper behind 1 M ohm
// and more, four machines: 57-124 -> 18-39 ns per sample). The output must
// stay the root to rounding: 50-digit roots of s + 2c sinh(s) = x, nVt = 1 and
// R = 1 so that c = Is and x is the source, for eleven constants from 0.0125
// to 1e5: inside the series (beyond c = 0.02), across the cells from 0.4 to
// 99.6 % of their range, and above the drive where the table root takes over.
DSPARK_TEST(AnalogAudit_wdf_local_expansion_roots_are_exact)
{
    using namespace dspark::wdf;
    struct Point { double c, x, s; };
    static constexpr Point kPoints[] = {
        { 0.0125, 1.8724032881031816, 1.7989318686920240115 },
        { 0.0125, 2.585217084958511, 2.4425278504430055353 },
        { 0.0125, 3.6035225090375533, 3.2738389888449588948 },
        { 0.0125, 4.6784004566765418, 3.9977124599620940166 },
        { 0.0125, 5.8664234514354252, 4.6102198180926607084 },
        { 0.0125, 6.9413013990744137, 5.0299039979958811136 },
        { 0.0125, 7.4843976252499029, 5.2056727469192042777 },
        { 0.0125, 7.7464918556554876, 5.283449646030650628 },
        { 0.0125, 13.038649658033988, 6.2911953594047502342 },
        { 0.0125, 306.79175665962327, 10.074805295124825832 },
        { 0.0178, 1.785739663134315, 1.6923251640931200499 },
        { 0.0178, 2.7178129181315769, 2.5019906366633395731 },
        { 0.0178, 4.0493461395562367, 3.4749974053451972432 },
        { 0.0178, 5.4548534288378221, 4.2308748934003771382 },
        { 0.0178, 7.0083088538332579, 4.814336430728516487 },
        { 0.0178, 8.4138161431148433, 5.1969843871875847865 },
        { 0.0178, 9.1239671945413292, 5.3553004258060599337 },
        { 0.0178, 9.4096495788194581, 5.4138263002413871307 },
        { 0.0178, 15.838024043557503, 6.2853784367845845394 },
        { 0.0178, 372.65938926017657, 9.9222353022944859473 },
        { 0.021, 0.051258366575838418, 0.049191490632522928892 },
        { 0.021, 0.93973672055703783, 0.89681453180574201048 },
        { 0.021, 1.6573538526187757, 1.5616582928340543241 },
        { 0.021, 1.741135618236292, 1.6372582363408698543 },
        { 0.021, 2.7656226880491479, 2.5091474527470112763 },
        { 0.021, 4.2291756449246556, 3.5201895933617657374 },
        { 0.021, 5.7740370994043593, 4.2709494230243762036 },
        { 0.021, 7.4815155490924523, 4.8361174254464810173 },
        { 0.021, 9.026377003572156, 5.2041074067166017565 },
        { 0.021, 9.8069385805724281, 5.356302197457599894 },
        { 0.021, 10.252797807742443, 5.435471011519596781 },
        { 0.021, 17.257184428873419, 6.2608027671117213817 },
        { 0.021, 406.0513983264334, 9.8451677041610263509 },
        { 0.05, 0.044701531906723636, 0.040636739448523320374 },
        { 0.05, 0.81952808495660012, 0.73874801285417175076 },
        { 0.05, 1.445349531650731, 1.2795136191978958554 },
        { 0.05, 1.5541911158743869, 1.3701065985310853949 },
        { 0.05, 3.5746027638577536, 2.775437002928989037 },
        { 0.05, 6.4609051181197055, 3.9261989055495551913 },
        { 0.05, 9.5075576031739892, 4.5888760990185047429 },
        { 0.05, 12.874910349812932, 5.0527357300589015249 },
        { 0.05, 15.921562834867217, 5.353583289447599727 },
        { 0.05, 17.46092409047359, 5.4791374074176246506 },
        { 0.05, 19.478704892901213, 5.624345256289182991 },
        { 0.05, 32.785938928645606, 6.2733553041804517152 },
        { 0.05, 771.43385714460248, 9.6314195317080702005 },
        { 0.3, 0.03673303827682553, 0.022957392683876908426 },
        { 0.3, 0.67343903507513481, 0.4163493376999726721 },
        { 0.3, 1.1877015709506922, 0.71852287014752587968 },
        { 0.3, 1.5507250652444515, 0.91866108566376593736 },
        { 0.3, 11.828874429777866, 3.3435576846781150375 },
        { 0.3, 26.511944950539885, 4.304579269597916104 },
        { 0.3, 42.010741611344237, 4.8200951573677175077 },
        { 0.3, 59.140990552233255, 5.1920419290584930775 },
        { 0.3, 74.639787213037621, 5.440975292617893187 },
        { 0.3, 82.470758157444024, 5.5468053219497609267 },
        { 0.3, 84.422085541285426, 5.5715414926739496383 },
        { 0.3, 142.09657962394576, 6.1164861749561961099 },
        { 0.3, 3343.4489323281355, 9.3159407644284395698 },
        { 0.5, 0.03649650245686073, 0.018247744875199249259 },
        { 0.5, 0.66910254504244671, 0.33149880805311734258 },
        { 0.5, 1.1800535794384968, 0.57400477785993390689 },
        { 0.5, 1.7515489479855573, 0.8270017360290541962 },
        { 0.5, 18.60401322982684, 3.4148136480367346906 },
        { 0.5, 42.678962203885817, 4.3397893803096876497 },
        { 0.5, 68.091408343170301, 4.840321967134800878 },
        { 0.5, 96.17884881290577, 5.2037630712918774841 },
        { 0.5, 121.59129495219025, 5.4479905249398800636 },
        { 0.5, 134.43126773835502, 5.5520379938432671034 },
        { 0.5, 140.68048712967916, 5.5990387986465724695 },
        { 0.5, 236.78893873312336, 6.1340743791726887906 },
        { 0.5, 5571.504440779373, 9.3168939421823551094 },
        { 2, 0.067260475822293109, 0.013451770614898605898 },
        { 2, 1.2331087234087073, 0.24466315173384935448 },
        { 2, 2.1747553849208106, 0.42464857638682240383 },
        { 2, 4.3378575717824832, 0.79770451924751263588 },
        { 2, 70.356871469522943, 3.5101490514294462602 },
        { 2, 164.669748466295, 4.3839668336121444164 },
        { 2, 264.22222974066551, 4.8651182004947960357 },
        { 2, 374.25391957023288, 5.2177767765739210818 },
        { 2, 473.80640084460344, 5.4560876029445712812 },
        { 2, 524.10660190954854, 5.5579016434823790101 },
        { 2, 573.12986320833204, 5.6480738997130263904 },
        { 2, 964.67402718234109, 6.1722286121947318774 },
        { 2, 22698.212404290382, 9.3364828630530415874 },
        { 10, 0.25534238959272304, 0.01215887608211965836 },
        { 10, 4.681277142533256, 0.22119588067526855645 },
        { 10, 8.2560705968313783, 0.38408582415293264293 },
        { 10, 18.92843174583555, 0.81314538375511040718 },
        { 10, 347.06452266729616, 3.5375269108837843611 },
        { 10, 815.83036684081139, 4.3963697991063471137 },
        { 10, 1310.6387579128552, 4.8720192383895651241 },
        { 10, 1857.5322427819563, 5.2216331682072110674 },
        { 10, 2352.3406338540003, 5.4582761324558140285 },
        { 10, 2602.349084079875, 5.5594609369099541684 },
        { 10, 2811.4423606657697, 5.6368735470545239288 },
        { 10, 4732.1307060710969, 6.1582480142712259009 },
        { 10, 111344.25190755523, 9.3177132772512909723 },
        { 100, 2.3839849063514436, 0.011860344744688856598 },
        { 100, 43.706389949776465, 0.2157748054359685161 },
        { 100, 77.082178638696675, 0.37470710959690128189 },
        { 100, 183.49466025509128, 0.81877874749987999997 },
        { 100, 3460.3923066214502, 3.5437782225428774368 },
        { 100, 8141.6746585733918, 4.3991914961835648561 },
        { 100, 13083.028252300441, 4.8735868098188255093 },
        { 100, 18544.524329577704, 5.2225070879082049078 },
        { 100, 23485.877923304753, 5.4587700767560402099 },
        { 100, 25982.56184434579, 5.5598115211571177937 },
        { 100, 26584.856456032558, 5.5827309766188568634 },
        { 100, 44746.788094312222, 6.1034733467951110536 },
        { 100, 1052865.6022191113, 9.26184717594535213 },
        { 1000, 23.672978525898746, 0.011830298158493883862 },
        { 1000, 434.00460630814376, 0.21522913811421822678 },
        { 1000, 765.42630567072615, 0.37376291662358773302 },
        { 1000, 1829.2422042852327, 0.81939293604599553477 },
        { 1000, 34593.74418707638, 3.5444051364139868196 },
        { 1000, 81400.175591063729, 4.3994742842341243702 },
        { 1000, 130806.9642952726, 4.873743864711879645 },
        { 1000, 185414.46759992451, 5.2225946053695600883 },
        { 1000, 234821.25630413339, 5.4588195019579161315 },
        { 1000, 259784.68638625997, 5.5598465744013575025 },
        { 1000, 294828.73986454395, 5.6863868622066114862 },
        { 1000, 496246.39383141062, 6.2070641178473218067 },
        { 1000, 11676385.737209663, 9.3653229734489074586 },
        { 100000, 2365.4651298552717, 0.011826990790327099168 },
        { 100000, 43366.860714013317, 0.2151690734467261592 },
        { 100000, 76483.372531987116, 0.37365898220738730716 },
        { 100000, 182861.5667579812, 0.81946107979899871395 },
        { 100000, 3459262.533286354, 3.5444741166108180801 },
        { 100000, 8139835.3426126009, 4.399505397771344363 },
        { 100000, 13080439.974679194, 4.8737611440538638351 },
        { 100000, 18541108.252226483, 5.2226042336836823874 },
        { 100000, 23481712.884293079, 5.4588249390719605494 },
        { 100000, 25978018.382600408, 5.5598504302007501724 },
        { 100000, 27232603.380214736, 5.6070132800990936515 },
        { 100000, 45837055.194420837, 6.1276825482183874002 },
        { 100000, 1078518945.7510786, 9.2859291254164123304 },
    };
    double worst = 0.0;
    for (const auto& p : kPoints)
    {
        ResistiveVoltageSource<double> source { 1.0 };
        DiodePairRoot<double, decltype(source)> root { source, p.c, 1.0 };
        root.prepare(48000.0);
        for (double sign : { 1.0, -1.0 })
        {
            source.setVoltage(sign * p.x);
            root.process();
            worst = std::max(worst, std::abs(root.getVoltage() - sign * p.s) / std::abs(p.s));
        }
    }
    EXPECT_LT(worst, 4.5e-16);
}

// A change of constants while running (here the saturation current) is not
// followed by the local expansions at once: they wait until the new constants
// have held for 256 samples and are then built one cell per sample, so that a
// value that moves every sample never pays for them. The root is exact on
// every sample of that passage, by the repeated Halley steps first and by the
// cells as they appear, and ends up bit for bit where a root prepared with
// the new constants is.
DSPARK_TEST(AnalogAudit_wdf_local_expansions_follow_a_change_of_constants)
{
    using namespace dspark::wdf;
    struct Point { double x, s; };
    // 50-digit roots for c = 0.3, across its cells.
    static constexpr Point kPoints[] = {
        { 1.5507250652444515, 0.91866108566376593736 },
        { 11.828874429777866, 3.3435576846781150375 },
        { 42.010741611344237, 4.8200951573677175077 },
        { 82.470758157444024, 5.5468053219497609267 },
    };
    ResistiveVoltageSource<double> source { 1.0 }, freshSource { 1.0 };
    DiodePairRoot<double, decltype(source)> root { source, 0.05, 1.0 };
    DiodePairRoot<double, decltype(freshSource)> fresh { freshSource, 0.3, 1.0 };
    root.prepare(48000.0);
    fresh.prepare(48000.0);
    root.setSaturationCurrent(0.3);
    double worst = 0.0;
    std::vector<double> last;
    for (int sample = 0; sample < 400; ++sample)
        for (const auto& p : kPoints)
        {
            source.setVoltage(p.x);
            root.process();
            const double v = root.getVoltage();
            worst = std::max(worst, std::abs(v - p.s) / p.s);
            if (sample == 399) last.push_back(v);
        }
    EXPECT_LT(worst, 4.5e-16);
    for (size_t i = 0; i < last.size(); ++i)
    {
        freshSource.setVoltage(kPoints[i].x);
        fresh.process();
        EXPECT_EQ(last[i], fresh.getVoltage());
    }
}

// The cells reach the drive from which the table root answers for every
// constant beyond 0.02, so that no drive is left to the Halley steps once
// they are built: 600 constants up to 1e6, from the expressions of the root.
DSPARK_TEST(AnalogAudit_wdf_local_expansions_cover_their_range)
{
    using Local = dspark::wdf::detail::DiodeLocal;
    static Local::Cell cells[Local::kMaxCells];
    int shortOnes = 0, most = 0;
    for (int i = 0; i < 600; ++i)
    {
        const double c = 0.0201 * std::pow(1e6 / 0.0201, i / 599.0);
        double b[17] {};
        const double bound = 1.0 / (1.0 + 2.0 * c), cPlus = 1.0 + 2.0 * c;
        const double from = std::max(cPlus * dspark::wdf::detail::DiodeSeries::pair(2.0 * c * bound, b),
                                     cPlus * Local::origin(2.0 * c * bound, b));
        // q = c^2 / (omega (1 + omega)) <= 4.05e-5 and c^2 / (1 + omega)^2 <= 1.47e-5.
        const double w = std::max(0.5 * (std::sqrt(1.0 + 4.0 * c * c / 4.05e-5) - 1.0), c / std::sqrt(1.47e-5) - 1.0);
        const double goal = (w + std::log(w)) - std::log(c);
        double offset = 0.0, end = -1.0;
        std::uint64_t firstKey = 0;
        const int count = Local::cells(c, from, goal, cells, offset, firstKey, end);
        most = std::max(most, count);
        if (!(count > 0 && end >= goal)) ++shortOnes;
    }
    EXPECT_EQ(shortOnes, 0);
    EXPECT_TRUE(most <= Local::kMaxCells);
}

// Speed-up pin: beyond the constants of the tabulated series (c above 0.0417:
// germanium diodes, very large resistances) the single diode read its root
// from an omega estimate and Halley steps with an exponential and a division
// each. It now has a low-drive series built for the constant, cells on both
// sides of the origin (detail::DiodeLocal) and, outside the drives of the
// roots -1 and 1.5, the closed form from the omega table with every leading
// term to twice the working precision (RC circuit behind 1 M ohm and more,
// four machines: 39-69 -> 17-31 ns per sample). The output must stay the root
// to rounding: 50-digit roots of s + c (e^s - 1) = x, nVt = 1 and R = 1 so
// that c = Is and x is the source, for seven constants from 0.045 to 1e4:
// inside the series, across the cells of each side, beyond the two table
// drives and around the knee of the blocked diode (x near -c), where x + c
// and omega meet and the Halley steps were off by up to 7e-14.
DSPARK_TEST(AnalogAudit_wdf_single_diode_local_roots_are_exact)
{
    using namespace dspark::wdf;
    struct Point { double c, x, s; };
    static constexpr Point kPoints[] = {
        { 0.045, 0.028524338244960673, 0.02727984752497336077 },
        { 0.045, -0.028524338244960673, -0.027311933179314849699 },
        { 0.045, 0.71310845612401685, 0.67015421396847997414 },
        { 0.045, -0.71310845612401685, -0.69066438835397183773 },
        { 0.045, 1.3834304048805928, 1.2684415660047003478 },
        { 0.045, -1.3834304048805928, -1.3500951071601921298 },
        { 0.045, 1.4285215032072056, 1.3072085833674274306 },
        { 0.045, 1.4838316862273286, 1.3544698973313682011 },
        { 0.045, 1.5414464602066233, 1.4033500973681392599 },
        { 0.045, 1.5990612341859181, 1.4518632933025403245 },
        { 0.045, 1.6543714172060411, 1.4980818775535456053 },
        { 0.045, 1.8621408469112652, 1.6684609251376472255 },
        { 0.045, -1.0387298793987578, -1.0101178028221320117 },
        { 0.045, 3.3186668558814629, 2.6964510503457366867 },
        { 0.045, -1.8512017652651129, -1.8135401761145233039 },
        { 0.045, 55.311114264691042, 6.9800969696945814535 },
        { 0.045, -30.853362754418548, -30.808362754418549687 },
        { 0.14, 0.023771042571458158, 0.020824976458514915556 },
        { 0.14, -0.023771042571458158, -0.020878372604543813499 },
        { 0.14, 0.59427606428645396, 0.50280640047576690683 },
        { 0.14, -0.59427606428645396, -0.53617359403277231593 },
        { 0.14, 1.1528955647157206, 0.93594759775035629704 },
        { 0.14, -1.1528955647157206, -1.0613346840771281873 },
        { 0.14, 1.1965409719856521, 0.96797561979923152983 },
        { 0.14, 1.3882732138915133, 1.1054089101155107989 },
        { 0.14, 1.5879942992101186, 1.2428349563499041563 },
        { 0.14, 1.7877153845287239, 1.3743592949579141989 },
        { 0.14, 1.9794476264345851, 1.4950865025453429558 },
        { 0.14, 2.1348611147425132, 1.5890133146892096988 },
        { 0.14, -1.0993818470183581, -1.0103544345121271439 },
        { 0.14, 3.8047029767688354, 2.4005732951403928701 },
        { 0.14, -1.9592943808247965, -1.8414955916911049396 },
        { 0.14, 63.411716279480586, 6.0184757335929750347 },
        { 0.14, -32.654906347079944, -32.514906347079944931 },
        { 0.5, 0.024279426398681334, 0.016142618933584560078 },
        { 0.5, -0.024279426398681334, -0.01622994959524737608 },
        { 0.5, 0.60698565996703335, 0.37759768863485640951 },
        { 0.5, -0.60698565996703335, -0.43169074836412883423 },
        { 0.5, 1.1775521803360447, 0.68533446046145119504 },
        { 0.5, -1.1775521803360447, -0.88409591556581354734 },
        { 0.5, 1.2342400520864163, 0.71358843962950547951 },
        { 0.5, 1.7206896237428082, 0.94030651635561452418 },
        { 0.5, 2.2274079275515497, 1.1493451715329188318 },
        { 0.5, 2.7341262313602912, 1.3346993695483797837 },
        { 0.5, 3.2205758030166831, 1.4937322957279023429 },
        { 0.5, -1.3150393898194768, -0.99913777603350186442 },
        { 0.5, -1.2905380395442259, -0.97847919381070564302 },
        { 0.5, -1.2650157996741727, -0.95703141256431439714 },
        { 0.5, -1.2394935598041197, -0.93565747691505235266 },
        { 0.5, -1.2149922095288688, -0.91520872751606577 },
        { 0.5, 3.5155657821687409, 1.5823577465208963313 },
        { 0.5, -1.3292208822084217, -1.0111255195140865748 },
        { 0.5, 6.2653647603007263, 2.2095512848809399083 },
        { 0.5, -2.3689085029457018, -1.9407095158041323721 },
        { 0.5, 104.4227460050121, 5.2945919761788075583 },
        { 0.5, -39.481808382428362, -38.981808382428361887 },
        { 2, 0.034653743236579507, 0.011506941398317159574 },
        { 2, -0.034653743236579507, -0.011595896603719780508 },
        { 2, 0.86634358091448771, 0.26346918911922867779 },
        { 2, -0.86634358091448771, -0.3194461838907356412 },
        { 2, 1.680706546974106, 0.47254947182180094154 },
        { 2, -1.680706546974106, -0.68693652681626652693 },
        { 2, 1.799994071617447, 0.50059359703787515948 },
        { 2, 3.4153599065407643, 0.82977129464054810428 },
        { 2, 5.0980326512525531, 1.0985294803843062124 },
        { 2, 6.7807053959643415, 1.3169160587419891005 },
        { 2, 8.3960712308876602, 1.4932239627802498914 },
        { 2, -2.2589255780988338, -0.99693961497078933779 },
        { 2, -2.1313526287000801, -0.92467402647693570878 },
        { 2, -1.9984641397430454, -0.85177663299968286983 },
        { 2, -1.8655756507860104, -0.78124601931619284454 },
        { 2, -1.738002701387257, -0.71570116077906073642 },
        { 2, 9.6904084774293846, 1.616774347820206428 },
        { 2, -2.2868835288336866, -1.0130807824394004988 },
        { 2, 17.270034910270191, 2.1472621380344623399 },
        { 2, -4.0756340117828076, -2.2801680503158330942 },
        { 2, 287.83391517116979, 4.9589033485071313329 },
        { 2, -67.927233529713462, -65.927233529713461735 },
        { 2, -1.25, -0.48338486666685651217 },
        { 2, -3.5, -1.8230610731703216724 },
        { 2, -8.5, -6.5029978776308255263 },
        { 2, -11.5, -9.5001496812536208669 },
        { 2, -16, -14.000001663054672454 },
        { 16, 0.1392305238831846, 0.0081586216411398873038 },
        { 16, -0.1392305238831846, -0.0082217542838037016635 },
        { 16, 3.4807630970796151, 0.18718364732653727414 },
        { 16, -3.4807630970796151, -0.22732698587473514967 },
        { 16, 6.7526804083344532, 0.33716466399600378887 },
        { 16, -6.7526804083344532, -0.49600620527706941474 },
        { 16, 7.463981183471728, 0.36710850427026584804 },
        { 16, 19.522900926971683, 0.77551560442355900804 },
        { 16, 32.08427565978414, 1.0776988028077086867 },
        { 16, 44.645650392596593, 1.3106112879260720727 },
        { 16, 56.70457013609655, 1.4930656628065968469 },
        { 16, -11.072404913785945, -0.9939853443502346131 },
        { 16, -10.075828254482499, -0.85825391694595286249 },
        { 16, -9.0377275677080764, -0.73209956948446231874 },
        { 16, -7.9996268809336533, -0.61862040493760950525 },
        { 16, -7.0030502216302075, -0.51956050727267382853 },
        { 16, 64.150172520268455, 1.5912600978010721748 },
        { 16, -11.225068230669493, -1.0162520007593678715 },
        { 16, 114.3270401351319, 2.0813590852199690994 },
        { 16, -20.005072094262463, -4.2364202514358831835 },
        { 16, 1905.4506689188649, 4.7857532093402312454 },
        { 16, -333.41786823770769, -317.41786823770769388 },
        { 16, -15.25, -1.8262527871533580402 },
        { 16, -17.5, -2.6408314145993693004 },
        { 16, -22.5, -6.5234964087993294793 },
        { 16, -25.5, -9.501196197533510021 },
        { 16, -30, -14.000013304282500547 },
        { 300, 2.0681854770817369, 0.0068476275575472581888 },
        { 300, -2.0681854770817369, -0.0068946831098291212176 },
        { 300, 51.704636927043424, 0.15855831934759515313 },
        { 300, -51.704636927043424, -0.18840495107575768294 },
        { 300, 100.30699563846424, 0.28773023480111418501 },
        { 300, -100.30699563846424, -0.40497532961348906386 },
        { 300, 112.83524832656018, 0.31849434600417776576 },
        { 300, 339.05863566592001, 0.75503156399739374265 },
        { 300, 574.70799747775322, 1.0688849010546272841 },
        { 300, 810.35735928958638, 1.3074764916626905061 },
        { 300, 1036.5807466289461, 1.4929698444170823142 },
        { 300, -189.7638987106225, -0.9921976384548993323 },
        { 300, -168.82944419994718, -0.82104419052969495371 },
        { 300, -147.02272075132709, -0.66912856536996876741 },
        { 300, -125.21599730270697, -0.53716295044109836471 },
        { 300, -104.28154279203166, -0.42493648245350985297 },
        { 300, 1107.5496668291153, 1.5447251124427079941 },
        { 300, -192.54252932505298, -1.0172651780702366015 },
        { 300, 1973.8508913786213, 2.0245568649894339874 },
        { 300, -343.14510176742112, -43.145101767421124405 },
        { 300, 32897.514856310358, 4.7063060468532370905 },
        { 300, -5719.0850294570191, -5419.0850294570191181 },
        { 300, -299.25, -4.1205712455581098744 },
        { 300, -301.5, -4.5791340649452146594 },
        { 300, -306.5, -6.8256658617778689234 },
        { 300, -309.5, -9.5219676325243117741 },
        { 300, -314, -14.000249396409405351 },
        { 10000, 67.993900938604071, 0.0067757054972274495106 },
        { 10000, -67.993900938604071, -0.0068219244039657060682 },
        { 10000, 1699.8475234651016, 0.1569772994142148326 },
        { 10000, -1699.8475234651016, -0.18628876392816266125 },
        { 10000, 3297.7041955222971, 0.28498487909208799158 },
        { 10000, -3297.7041955222971, -0.40007527741356691105 },
        { 10000, 3713.8820034947075, 0.31580048367839525248 },
        { 10000, 11254.368961042814, 0.75394190645545424837 },
        { 10000, 19109.042875155425, 1.068427079658739594 },
        { 10000, 26963.716789268037, 1.3073163430215917292 },
        { 10000, 34504.203746816143, 1.4929650107678821699 },
        { 10000, -6292.9804828720225, -0.99208931518660757256 },
        { 10000, -5591.5779529467336, -0.8188825431585328083 },
        { 10000, -4860.9503176078897, -0.66558741004145663762 },
        { 10000, -4130.3226822690467, -0.53269468255152580016 },
        { 10000, -3428.9201523437569, -0.41984302312414043578 },
        { 10000, 35391.488043312733, 1.5127061801195348344 },
        { 10000, -6385.4276441684324, -1.0173301287055495181 },
        { 10000, 63073.939087091996, 1.9888594823066106375 },
        { 10000, -11379.970058914038, -1379.9700589140375087 },
        { 10000, 1051232.3181181999, 4.6645965876575060624 },
        { 10000, -189666.16764856729, -179666.16764856729424 },
        { 10000, -9999.25, -7.1442107034915909227 },
        { 10000, -10001.5, -7.4302705380725369334 },
        { 10000, -10006.5, -8.511473055089643769 },
        { 10000, -10009.5, -9.9685189984493055832 },
        { 10000, -10014, -14.008246993072805746 },
    };
    double worst = 0.0;
    for (const auto& p : kPoints)
    {
        ResistiveVoltageSource<double> source { 1.0 };
        DiodeRoot<double, decltype(source)> root { source, p.c, 1.0 };
        root.prepare(48000.0);
        source.setVoltage(p.x);
        root.process();
        worst = std::max(worst, std::abs(root.getVoltage() - p.s) / std::abs(p.s));
    }
    EXPECT_LT(worst, 4.5e-16);
}

// A change of constants while running (here the saturation current, from a
// constant of the tabulated series to one beyond them) is not followed by the
// series and the cells at once: they wait until the new constants have held
// for 256 samples and are then built, the series first and one cell per
// sample after it. The root is exact on every sample of that passage, by the
// table and the correction steps first and by the series and the cells as
// they appear, and ends up bit for bit where a root prepared with the new
// constants is.
DSPARK_TEST(AnalogAudit_wdf_single_diode_follows_a_change_of_constants)
{
    using namespace dspark::wdf;
    struct Point { double x, s; };
    // 50-digit roots for c = 2: its series, its cells on both sides, the table.
    static constexpr Point kPoints[] = {
        { 0.86634358091448771, 0.26346918911922867779 },
        { -0.86634358091448771, -0.3194461838907356412 },
        { 3.4153599065407643, 0.82977129464054810428 },
        { 6.7807053959643415, 1.3169160587419891005 },
        { -2.1313526287000801, -0.92467402647693570878 },
        { -1.8655756507860104, -0.78124601931619284454 },
        { 17.270034910270191, 2.1472621380344623399 },
        { -4.0756340117828076, -2.2801680503158330942 },
    };
    ResistiveVoltageSource<double> source { 1.0 }, freshSource { 1.0 };
    DiodeRoot<double, decltype(source)> root { source, 0.01, 1.0 };
    DiodeRoot<double, decltype(freshSource)> fresh { freshSource, 2.0, 1.0 };
    root.prepare(48000.0);
    fresh.prepare(48000.0);
    root.setSaturationCurrent(2.0);
    double worst = 0.0, worstSettled = 0.0;
    std::vector<double> last;
    for (int sample = 0; sample < 400; ++sample)
        for (const auto& p : kPoints)
        {
            source.setVoltage(p.x);
            root.process();
            const double v = root.getVoltage();
            const double error = std::abs(v - p.s) / std::abs(p.s);
            worst = std::max(worst, error);
            if (sample >= 300) worstSettled = std::max(worstSettled, error);
            if (sample == 399) last.push_back(v);
        }
    EXPECT_LT(worst, 2e-15);
    EXPECT_LT(worstSettled, 4.5e-16);
    for (size_t i = 0; i < last.size(); ++i)
    {
        freshSource.setVoltage(kPoints[i].x);
        fresh.process();
        EXPECT_EQ(last[i], fresh.getVoltage());
    }
}

// The cells of the single diode reach both table drives for every constant
// from the end of the tabulated series to 1e5, so that no drive is left to
// the Halley steps once they are built: 600 constants, from the expressions
// of the root, in the room an instance has.
DSPARK_TEST(AnalogAudit_wdf_single_diode_cells_cover_their_range)
{
    using Local = dspark::wdf::detail::DiodeLocal;
    static Local::Cell cells[Local::kSingleMaxCells];
    int shortOnes = 0, most = 0;
    for (int i = 0; i < 600; ++i)
    {
        const double c = 0.0417 * std::pow(1e5 / 0.0417, i / 599.0);
        double q[32] {};
        const double limit = (1.0 + c) * Local::originSingle(c / (1.0 + c), q);
        double reverse = 0.0, forward = 0.0;
        Local::singleTableDrives(c, reverse, forward);
        const double offset = Local::singleOffset(c);
        Local::Builder builder;
        Local::startSingle(builder, c, reverse, -limit, offset, true, Local::kSingleMaxCells);
        double reverseTo = reverse;
        while (Local::step(builder, cells, reverseTo)) {}
        const int below = builder.count;
        Local::startSingle(builder, c, limit, forward, offset, false, Local::kSingleMaxCells - below);
        double forwardTo = limit;
        while (Local::step(builder, cells + below, forwardTo)) {}
        most = std::max(most, below + builder.count);
        if (reverse < -limit && !(reverseTo >= -limit)) ++shortOnes;
        if (limit < forward && !(forwardTo >= forward)) ++shortOnes;
    }
    EXPECT_EQ(shortOnes, 0);
    EXPECT_TRUE(most <= Local::kSingleMaxCells);
}

// The omega table keeps what the constant term of each node leaves out of
// -omega or ln(omega), from an exponential to twice the working precision
// (detail::expTwice): against e^s to 34 digits at the centres of the single
// diode's anchors, and e^a e^-a = 1 to twice the working precision for
// arguments across the range the table and the constants use.
DSPARK_TEST(AnalogAudit_wdf_double_double_exponential_is_exact)
{
    using Local = dspark::wdf::detail::DiodeLocal;
    for (const auto& m : Local::kSingleCentres)
    {
        double hi = 0.0, lo = 0.0;
        dspark::wdf::detail::expTwice(m.s, hi, lo);
        EXPECT_EQ(hi, m.expHi);
        EXPECT_LT(std::abs(lo - m.expLo), 1e-29 * m.expHi);
    }
    for (double a : { -40.0, -17.3, -3.7, 0.3, 5.5, 13.9, 34.5 })
    {
        double h1 = 0.0, l1 = 0.0, h2 = 0.0, l2 = 0.0;
        dspark::wdf::detail::expTwice(a, h1, l1);
        dspark::wdf::detail::expTwice(-a, h2, l2);
        const double product = h1 * h2;
        const double excess = ((product - 1.0) + std::fma(h1, h2, -product)) + (h1 * l2 + l1 * h2);
        EXPECT_LT(std::abs(excess), 1e-30);
    }
}

// The grid clamp is referenced to ground: it compresses from 0 V although the
// cathode sits near +1.2 V, so a 1 V source reaches the grid as 0.624 V.
// GridModel::Conduction drives the grid through 68 k ohm against a grid
// current of the grid-to-cathode voltage: 0.936 V, the root of the grid node
// equation; the core keeps its accuracy against the point circuit at 32x.
DSPARK_TEST(AnalogAudit_tube_grid_conduction_is_cathode_referenced)
{
    // Independent solve of w + Rs * Ig(w) = d with the published 12AX7 fit.
    const auto ig = [](double v) {
        const double z = 9.901 * v;
        const double sp = z > 40.0 ? z : std::log1p(std::exp(z));
        return 6.177e-4 * std::pow(sp / 9.901, 1.314);
    };
    const auto root = [&](double d) {
        double lo = d - 68e3 * ig(d), hi = d;
        for (int i = 0; i < 200; ++i)
        {
            const double m = 0.5 * (lo + hi);
            (m + 68e3 * ig(m) - d > 0.0 ? hi : lo) = m;
        }
        return 0.5 * (lo + hi);
    };
    detail::TubePreampGridTable table;
    double worst = 0.0;
    for (double d = -3.0; d <= 90.0; d += 1.0 / 512.0)
        worst = std::max(worst, std::abs(table.eval(d) - root(d)));
    EXPECT_LT(worst, 1e-7);
    EXPECT_NEAR(1.2 + table.eval(1.0 - 1.2), 0.9363, 5e-4);
    EXPECT_NEAR(detail::TubePreampGridClamp::value(1.0), 0.624, 1e-3);

    // Whole circuit: the 2x core against the point circuit at 32x, both with
    // the conduction model (two stages, +24 dB, 1 kHz).
    const auto render = [](int factor, double rate) {
        TubePreamp<double> amp;
        amp.setGridModel(TubePreamp<double>::GridModel::Conduction);
        amp.setOversampling(factor);
        amp.setStages(2);
        amp.setDrive(24.0);
        amp.setBass(0.0);
        amp.setMiddle(1.0);
        amp.prepare(spec(rate, 4096, 1));
        const size_t latency = static_cast<size_t>(amp.getLatency());
        const size_t total = static_cast<size_t>(0.15 * rate) + latency;
        std::vector<double> y(total);
        for (size_t i = 0; i < total; ++i)
            y[i] = 0.25 * std::sin(kTwoPi * 1000.0 * static_cast<double>(i) / rate);
        for (size_t pos = 0; pos < total; pos += 4096)
        {
            double* ch[] = { y.data() + pos };
            amp.processBlock({ ch, 1, static_cast<int>(std::min<size_t>(4096, total - pos)) });
        }
        y.erase(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(latency));
        return y;
    };
    const auto harmonic = [](const std::vector<double>& y, double rate, int h) {
        const size_t count = static_cast<size_t>(30.0 * rate / 1000.0 + 0.5), from = y.size() - count;
        std::complex<double> sum = 0.0;
        for (size_t i = from; i < y.size(); ++i)
            sum += y[i] * std::polar(1.0, -kTwoPi * h * 1000.0 * static_cast<double>(i - from) / rate);
        return 2.0 * sum / static_cast<double>(count);
    };
    const double referenceRate = 48000.0 * 32.0;
    const auto reference = render(1, referenceRate);
    const auto actual = render(2, 48000.0);
    const auto r1 = harmonic(reference, referenceRate, 1), a1 = harmonic(actual, 48000.0, 1);
    EXPECT_LT(std::abs(toDb(std::abs(a1) / std::abs(r1))), 0.05);
    double err = 0.0;
    for (int h = 2; h <= 20; ++h)
    {
        const auto r = harmonic(reference, referenceRate, h) / std::pow(r1 / std::abs(r1), h) / std::abs(r1);
        const auto a = harmonic(actual, 48000.0, h) / std::pow(a1 / std::abs(a1), h) / std::abs(a1);
        err = std::max(err, std::abs(a - r));
    }
    EXPECT_LT(toDb(err), -70.0);

    // The model is part of the state; blobs without the key restore Clamp.
    TubePreamp<double> amp;
    amp.setGridModel(TubePreamp<double>::GridModel::Conduction);
    const auto blob = amp.getState();
    TubePreamp<double> restored;
    EXPECT_TRUE(restored.getGridModel() == TubePreamp<double>::GridModel::Clamp);
    EXPECT_TRUE(restored.setState(blob.data(), blob.size()));
    EXPECT_TRUE(restored.getGridModel() == TubePreamp<double>::GridModel::Conduction);
}

// GridModel::Conduction drove each grid straight from its source, so grid
// current left no trace: after a loud passage the stage recovered at once.
// The grids are now fed through a coupling capacitor and grid leak: grid
// current charges the capacitor and biases the stage colder (blocking
// distortion). After a 300 ms burst at +24 dB a quiet tone 2 ms later comes
// out 15.6 dB below its settled level (0.4 dB without the coupling network)
// and recovers with the 22 ms coupling time constant; the Clamp model has no
// such memory.
DSPARK_TEST(AnalogAudit_tube_conduction_models_blocking_distortion)
{
    const auto probe = [](TubePreamp<double>::GridModel model, int factor) {
        TubePreamp<double> amp;
        amp.setGridModel(model);
        amp.setOversampling(factor);
        amp.setDrive(24.0);
        amp.prepare(spec(48000.0, 512, 1));
        const size_t latency = static_cast<size_t>(amp.getLatency());
        const size_t burst = 14400, total = 14400 + 19200 + latency;
        std::vector<double> y(total);
        for (size_t i = 0; i < total; ++i)
            y[i] = (i < burst ? 0.5 : 0.01) * std::sin(kTwoPi * 1000.0 * static_cast<double>(i) / 48000.0);
        for (size_t pos = 0; pos < total; pos += 512)
        {
            double* ch[] = { y.data() + pos };
            amp.processBlock({ ch, 1, static_cast<int>(std::min<size_t>(512, total - pos)) });
        }
        // 1 kHz level over 10 ms windows starting 2 ms and 300 ms after the burst.
        const auto level = [&](double ms) {
            const size_t from = burst + static_cast<size_t>(ms * 48.0) + latency;
            std::vector<double> window(y.begin() + static_cast<std::ptrdiff_t>(from),
                                       y.begin() + static_cast<std::ptrdiff_t>(from + 480));
            return toneAmplitude(window, 0, 480, 1000.0, 48000.0);
        };
        return toDb(level(2.0) / level(300.0));
    };
    for (int factor : { 2, 1 })
    {
        EXPECT_LT(probe(TubePreamp<double>::GridModel::Conduction, factor), -10.0);
        EXPECT_GT(probe(TubePreamp<double>::GridModel::Clamp, factor), -3.0);
    }
}

// The conduction model held each coupling capacitor at its predicted value
// over the whole internal sample and stepped it at every boundary. Under
// grid current the steps are large, and the output kernel cannot remove a
// step train: at 2x and +36 dB the fifth harmonic of a high tone came back
// folded at -65.5 dBc (two stages, 19.25 kHz: 255 Hz) and -68.3 dBc (one
// stage, 20 kHz: 4 kHz). The grid reference now ramps continuously across
// each interval: -74.5 and -74.6 dBc.
DSPARK_TEST(AnalogAudit_tube_conduction_reference_is_continuous)
{
    const double fs = 48000.0;
    const int n = 16384;
    const auto worst = [&](int stages, double hz) {
        int b = static_cast<int>(std::lround(hz / fs * n));
        if ((b & 1) == 0) ++b;
        TubePreamp<double> amp;
        amp.setGridModel(TubePreamp<double>::GridModel::Conduction);
        amp.setStages(stages);
        amp.setDrive(36.0);
        amp.prepare(spec(fs, 512, 1));
        const size_t settle = 24000 + static_cast<size_t>(amp.getLatency());
        const size_t total = (settle + static_cast<size_t>(n) + 511) / 512 * 512;
        std::vector<double> y(total);
        for (size_t i = 0; i < total; ++i)
            y[i] = 0.5 * std::sin(kTwoPi * (b * fs / n) * static_cast<double>(i) / fs);
        for (size_t pos = 0; pos < total; pos += 512)
        {
            double* ch[] = { y.data() + pos };
            amp.processBlock({ ch, 1, 512 });
        }
        // Worst non-harmonic line from 20 Hz to 20 kHz (the coupling
        // network's settling stays below 20 Hz).
        std::vector<double> frame(y.begin() + static_cast<std::ptrdiff_t>(settle),
                                  y.begin() + static_cast<std::ptrdiff_t>(settle) + n);
        FFTReal<double> fft(static_cast<size_t>(n));
        std::vector<double> spectrum(static_cast<size_t>(n) + 2);
        fft.forward(frame.data(), spectrum.data());
        const auto mag = [&](size_t k) { return std::hypot(spectrum[2 * k], spectrum[2 * k + 1]); };
        double w = 0.0;
        for (size_t k = static_cast<size_t>(std::ceil(20.0 / fs * n)); k <= static_cast<size_t>(20000.0 / fs * n); ++k)
            if (k % static_cast<size_t>(b) != 0) w = std::max(w, mag(k));
        return toDb(w / mag(static_cast<size_t>(b)));
    };
    EXPECT_LT(worst(2, 19250.0), -71.0);
    EXPECT_LT(worst(1, 20000.0), -71.0);
}

// The coupling capacitor's charge by grid current was a linear ramp across
// each interval (aimed at a predicted end), and one cubic spanned the six
// decades of plate current between cutoff and 3 V below the cathode, where
// the colder-biased stage starts its pulses. At 2x and +24 dB a high tone's
// harmonics came back folded at -74.1 dBc (one stage, 19.25 kHz) and
// -73.1 dBc (two stages, 16 kHz), against -80.6 and -83.0 for Clamp over
// the documented sweep. The charge is now integrated over each piece from
// its grid currents, the stage-1 leak is exact in the source polynomial and
// stage 1 is split once more in that region: -90.8 and -90.1 dBc.
DSPARK_TEST(AnalogAudit_tube_conduction_charge_is_integrated)
{
    const double fs = 48000.0;
    const int n = 16384;
    const auto worst = [&](int stages, double drive, double hz) {
        int b = static_cast<int>(std::lround(hz / fs * n));
        if ((b & 1) == 0) ++b;
        TubePreamp<double> amp;
        amp.setGridModel(TubePreamp<double>::GridModel::Conduction);
        amp.setStages(stages);
        amp.setDrive(drive);
        amp.prepare(spec(fs, 512, 1));
        const size_t settle = 48000 + static_cast<size_t>(amp.getLatency());
        const size_t total = (settle + static_cast<size_t>(n) + 511) / 512 * 512;
        std::vector<double> y(total);
        for (size_t i = 0; i < total; ++i)
            y[i] = 0.5 * std::sin(kTwoPi * (b * fs / n) * static_cast<double>(i) / fs);
        for (size_t pos = 0; pos < total; pos += 512)
        {
            double* ch[] = { y.data() + pos };
            amp.processBlock({ ch, 1, 512 });
        }
        std::vector<double> frame(y.begin() + static_cast<std::ptrdiff_t>(settle),
                                  y.begin() + static_cast<std::ptrdiff_t>(settle) + n);
        FFTReal<double> fft(static_cast<size_t>(n));
        std::vector<double> spectrum(static_cast<size_t>(n) + 2);
        fft.forward(frame.data(), spectrum.data());
        const auto mag = [&](size_t k) { return std::hypot(spectrum[2 * k], spectrum[2 * k + 1]); };
        double w = 0.0;
        for (size_t k = static_cast<size_t>(std::ceil(20.0 / fs * n)); k <= static_cast<size_t>(20000.0 / fs * n); ++k)
            if (k % static_cast<size_t>(b) != 0) w = std::max(w, mag(k));
        return toDb(w / mag(static_cast<size_t>(b)));
    };
    EXPECT_LT(worst(1, 24.0, 19250.0), -86.0);
    EXPECT_LT(worst(2, 24.0, 16000.0), -86.0);
    EXPECT_LT(worst(2, 36.0, 15250.0), -81.0);
}

// From 4x the continuous core leaves the plate's harmonics up to the
// internal Nyquist at close to their full level, and the decimator's later
// stages were only as deep as its first (80 dB): at 4x a 20 kHz tone's
// fourth and fifth harmonics came back at -86.6 dBc (conduction, one stage,
// +36 dB), -84.9 dBc (two stages) and -93.0 dBc (Clamp, one stage, +24 dB).
// From 4x every decimation stage is now at least 100 dB deep: -109.4,
// -107.4 and -96.5 dBc; 19 kHz with two Clamp stages at +24 dB went from
// -89.4 to -109.6 dBc.
DSPARK_TEST(AnalogAudit_tube_decimation_is_deep_from_4x)
{
    const double fs = 48000.0;
    const int n = 16384;
    const auto worst = [&](bool conduction, int stages, double drive, double hz) {
        int b = static_cast<int>(std::lround(hz / fs * n));
        if ((b & 1) == 0) ++b;
        TubePreamp<double> amp;
        if (conduction) amp.setGridModel(TubePreamp<double>::GridModel::Conduction);
        amp.setOversampling(4);
        amp.setStages(stages);
        amp.setDrive(drive);
        amp.prepare(spec(fs, 512, 1));
        const size_t settle = 48000 + static_cast<size_t>(amp.getLatency());
        const size_t total = (settle + static_cast<size_t>(n) + 511) / 512 * 512;
        std::vector<double> y(total);
        for (size_t i = 0; i < total; ++i)
            y[i] = 0.5 * std::sin(kTwoPi * (b * fs / n) * static_cast<double>(i) / fs);
        for (size_t pos = 0; pos < total; pos += 512)
        {
            double* ch[] = { y.data() + pos };
            amp.processBlock({ ch, 1, 512 });
        }
        std::vector<double> frame(y.begin() + static_cast<std::ptrdiff_t>(settle),
                                  y.begin() + static_cast<std::ptrdiff_t>(settle) + n);
        FFTReal<double> fft(static_cast<size_t>(n));
        std::vector<double> spectrum(static_cast<size_t>(n) + 2);
        fft.forward(frame.data(), spectrum.data());
        const auto mag = [&](size_t k) { return std::hypot(spectrum[2 * k], spectrum[2 * k + 1]); };
        double w = 0.0;
        for (size_t k = static_cast<size_t>(std::ceil(20.0 / fs * n)); k <= static_cast<size_t>(20000.0 / fs * n); ++k)
            if (k % static_cast<size_t>(b) != 0) w = std::max(w, mag(k));
        return toDb(w / mag(static_cast<size_t>(b)));
    };
    EXPECT_LT(worst(true, 1, 36.0, 20000.0), -100.0);
    EXPECT_LT(worst(true, 2, 36.0, 20000.0), -100.0);
    EXPECT_LT(worst(false, 1, 24.0, 20000.0), -95.0);
    EXPECT_LT(worst(false, 2, 24.0, 19000.0), -100.0);

    // The deeper stages add 20 base samples at 4x; the dry path follows.
    TubePreamp<double> amp;
    amp.setOversampling(4);
    amp.prepare(spec(fs, 512, 1));
    EXPECT_EQ(amp.getLatency(), 93);
}

// Saturation and Clipper decimated through the High preset at every factor,
// whose later stages are only as deep as its first. At 8x the wavefolder at
// +24 dB, which leaves harmonics above the tone's own level across the whole
// oversampled band, aliased at -75.2 dBc; at 16x the curves with second-order
// antialiasing stopped at the last stage's 93 dB (SoftClip -95.1 dBc, hard
// clip -94.9 dBc). Saturation from 8x and Clipper at 16x now decimate with
// every stage at least 100 dB deep: -104.3, -119.3 and -114.7 dBc, for 20
// more base samples of latency at 8x and 21 at 16x. The lower factors keep
// the preset: there the folding of the curve itself is what is left.
DSPARK_TEST(AnalogAudit_saturation_and_clipper_decimate_deep_at_high_factors)
{
    const double fs = 48000.0;
    const int n = 16384;
    // Worst non-harmonic component from 20 Hz to 20 kHz relative to a
    // coherent tone of amplitude 0.5, after one second.
    const auto worst = [&](auto& p, double hz) {
        int b = static_cast<int>(std::lround(hz / fs * n));
        if ((b & 1) == 0) ++b;
        const size_t settle = 48000 + static_cast<size_t>(p.getLatency());
        const size_t total = (settle + static_cast<size_t>(n) + 511) / 512 * 512;
        std::vector<double> y(total);
        for (size_t i = 0; i < total; ++i)
            y[i] = 0.5 * std::sin(kTwoPi * (b * fs / n) * static_cast<double>(i) / fs);
        for (size_t pos = 0; pos < total; pos += 512)
        {
            double* ch[] = { y.data() + pos };
            p.processBlock(AudioBufferView<double>(ch, 1, 512));
        }
        FFTReal<double> fft(static_cast<size_t>(n));
        std::vector<double> spectrum(static_cast<size_t>(n) + 2);
        fft.forward(y.data() + settle, spectrum.data());
        const auto mag = [&](size_t k) { return std::hypot(spectrum[2 * k], spectrum[2 * k + 1]); };
        double w = 0.0;
        for (size_t k = static_cast<size_t>(std::ceil(20.0 / fs * n)); k <= static_cast<size_t>(20000.0 / fs * n); ++k)
            if (k % static_cast<size_t>(b) != 0) w = std::max(w, mag(k));
        return toDb(w / mag(static_cast<size_t>(b)));
    };
    using Sat = Saturation<double>;
    const auto saturation = [&](Sat::Algorithm algorithm, int factor, Sat::Antialiasing order) {
        auto s = std::make_unique<Sat>();
        s->setAlgorithm(algorithm);
        s->setDrive(24.0);
        s->setMix(1.0);
        s->setDcBlocking(false);
        s->setAntialiasing(order);
        s->setOversampling(factor);
        s->prepare(spec(fs, 512, 1));
        return s;
    };
    using Clip = Clipper<double>;
    const auto clipper = [&](int factor, bool antialias) {
        auto c = std::make_unique<Clip>();
        c->setMode(Clip::Mode::Hard);
        c->setOversampling(factor);
        c->setInputGain(24.0);
        c->setMix(1.0);
        c->setAntialiasing(antialias);
        c->prepare(spec(fs, 512, 1));
        return c;
    };

    for (double hz : { 15250.0, 19250.0 })
        EXPECT_LT(worst(*saturation(Sat::Algorithm::Wavefolder, 8, Sat::Antialiasing::Off), hz), -95.0);
    EXPECT_LT(worst(*saturation(Sat::Algorithm::SoftClip, 16, Sat::Antialiasing::SecondOrder), 19250.0), -110.0);
    EXPECT_LT(worst(*clipper(16, true), 19250.0), -105.0);

    // 20 more base samples at 8x and 21 at 16x; the lower factors are unchanged.
    EXPECT_EQ(saturation(Sat::Algorithm::SoftClip, 4, Sat::Antialiasing::Off)->getLatency(), 70);
    EXPECT_EQ(saturation(Sat::Algorithm::SoftClip, 8, Sat::Antialiasing::Off)->getLatency(), 93);
    EXPECT_EQ(saturation(Sat::Algorithm::SoftClip, 16, Sat::Antialiasing::Off)->getLatency(), 95);
    EXPECT_EQ(clipper(8, false)->getLatency(), 73);
    EXPECT_EQ(clipper(16, false)->getLatency(), 95);
}
