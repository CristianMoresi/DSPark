// DSPark Core regression tests for the 2026-11 Core audit fixes.
// Each case states the defect it pins and the number the unfixed code gave.

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/Biquad.h"
#include "../Core/Dither.h"
#include "../Core/DspMath.h"
#include "../Core/FIRFilter.h"
#include "../Core/Hilbert.h"
#include "../Core/Interpolation.h"
#include "../Core/LadderFilter.h"
#include "../Core/Oversampling.h"
#include "../Core/SimdOps.h"
#include "../Core/SmoothedValue.h"
#include "../Core/Smoothers.h"
#include "../Core/StateVariableFilter.h"
#include "../Effects/Delay.h"
#include "../Effects/Equalizer.h"
#include "../Effects/Filters.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

constexpr double kRegPi = 3.14159265358979323846;

// Timing comparisons only mean something in optimised, uninstrumented builds.
#if defined(_DEBUG) || defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) \
    || ((defined(__GNUC__) || defined(__clang__)) && !defined(__OPTIMIZE__))
constexpr bool kTimingReliable = false;
#elif defined(__has_feature)
  #if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || __has_feature(memory_sanitizer)
constexpr bool kTimingReliable = false;
  #else
constexpr bool kTimingReliable = true;
  #endif
#else
constexpr bool kTimingReliable = true;
#endif

volatile double gRegSink = 0.0;

template <class F>
double minNanoseconds(F&& f, int rounds = 15)
{
    double best = 1e30;
    for (int r = 0; r < rounds; ++r)
    {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return best;
}

// Amplitude of a sine at w (radians per sample) by quadrature projection.
double toneAmplitude(const float* x, int n, double w)
{
    double c = 0.0, s = 0.0;
    for (int i = 0; i < n; ++i)
    {
        c += x[i] * std::cos(w * i);
        s += x[i] * std::sin(w * i);
    }
    return 2.0 * std::hypot(c, s) / n;
}

// Analog prototypes for the matched designs.
double analogDb(int type, double f, double f0, double Q, double gainDb)
{
    const std::complex<double> s(0.0, f / f0);
    const double A = std::pow(10.0, gainDb / 40.0);
    const double r = std::sqrt(A) * std::sqrt(2.0);
    std::complex<double> h;
    switch (type)
    {
        case 0: h = 1.0 / (s * s + s / Q + 1.0); break;                          // LP
        case 1: h = s * s / (s * s + s / Q + 1.0); break;                        // HP
        case 2: h = (s / Q) / (s * s + s / Q + 1.0); break;                      // BP
        case 3: h = A * (s * s + r * s + A) / (A * s * s + r * s + 1.0); break;  // low shelf
        default: h = A * (A * s * s + r * s + 1.0) / (s * s + r * s + A); break; // high shelf
    }
    return 20.0 * std::log10(std::abs(h));
}

double worstAnalogError(const BiquadCoeffs& c, int type, double fs, double f0, double Q, double gainDb)
{
    double worst = 0.0;
    for (double f = 20.0; f <= 20000.0 && f < 0.495 * fs; f *= std::pow(2.0, 1.0 / 24.0))
    {
        const double a = analogDb(type, f, f0, Q, gainDb);
        if (a < -60.0) continue;
        worst = std::max(worst, std::abs(20.0 * std::log10(c.getMagnitude(f, fs)) - a));
    }
    return worst;
}

bool isStable(const BiquadCoeffs& c)
{
    return std::isfinite(c.b0) && std::isfinite(c.b1) && std::isfinite(c.b2)
        && std::abs(c.a2) < 1.0 && std::abs(c.a1) < 1.0 + c.a2;
}

} // namespace

// ============================================================================
// 1. Smoothers: double state
// ============================================================================

// Delay smooths its delay time through Smoothers::ExponentialSmoother (its
// default smoother). With float state the per-step coefficient rounding left a
// 19.76-sample jump of the delay at the end of a 200 ms 96000 -> 91200 ramp
// (nominal step 0.5): a click on the default path.
DSPARK_TEST(CoreReg_delay_exponential_smoothing_has_no_end_jump)
{
    auto d = std::make_unique<Delay<double>>();
    constexpr int bs = 256;
    d->prepare(spec(48000.0, bs, 1), 5.0);
    d->setSmoother(Delay<double>::SmootherType::Exponential);
    d->setSmoothingTime(200.0f);
    d->setDelaySamples(96000.0);
    d->setFeedback(0.0);
    d->setMix(1.0);
    std::vector<double> b(bs), delays;
    const double k = 1e-5;
    double n = 0.0;
    const long switchBlock = static_cast<long>(3.5 * 48000.0 / bs);
    for (long block = 0; block < static_cast<long>(4.0 * 48000.0 / bs); ++block)
    {
        if (block == switchBlock) d->setDelaySamples(91200.0);
        for (int i = 0; i < bs; ++i) b[static_cast<size_t>(i)] = (n + i) * k;   // ramp input reveals the delay
        double* c[1] = { b.data() };
        d->processBlock(AudioBufferView<double>(c, 1, bs));
        if (block >= switchBlock - 2)
            for (int i = 0; i < bs; ++i) delays.push_back((n + i) - b[static_cast<size_t>(i)] / k);
        n += bs;
    }
    double maxStep = 0.0;
    for (size_t i = 1; i < delays.size(); ++i)
        maxStep = std::max(maxStep, std::abs(delays[i] - delays[i - 1]));
    EXPECT_LT(maxStep, 0.6);                      // old: 19.76 samples
    EXPECT_LT(delays.back(), 96000.0 - 4000.0);   // the glide is under way
}

// A float SlewLimiter loses every step below half an ulp: at 10000 with a
// 20 units/s limit at 48 kHz (4.2e-4 per sample) it never moved (old: still
// 10000.000 after 50 s).
DSPARK_TEST(CoreReg_slew_limiter_moves_at_large_values)
{
    Smoothers::SlewLimiter s;
    s.reset(48000.0, 20.0f, 10000.0f);
    s.setTargetValue(11000.0f);
    float v = 0.0f;
    for (int i = 0; i < 2 * 48000; ++i) v = s.getNextValue();
    EXPECT_NEAR(v, 10040.0f, 0.05f);
}

// A float LinearSmoother accumulates rounding: a 5 s ramp at 192 kHz ended
// with a 1.3e-2 jump onto its target (old) instead of its 1.04e-6 step.
DSPARK_TEST(CoreReg_linear_smoother_long_ramp_lands_smoothly)
{
    Smoothers::LinearSmoother l;
    l.reset(192000.0, 5000.0f, 0.0f);
    l.setTargetValue(1.0f);
    double prev = 0.0, maxStep = 0.0;
    for (int i = 0; i < 5 * 192000; ++i)
    {
        const double v = l.getNextValue();
        maxStep = std::max(maxStep, std::abs(v - prev));
        prev = v;
    }
    EXPECT_LT(maxStep, 1e-5);                     // old: 1.3e-2
    EXPECT_EQ(l.getCurrentValue(), 1.0f);
}

// ============================================================================
// 2. NaN recovery in the recursive block paths; SmoothedValue
// ============================================================================

// One NaN sample latched every recursive filter: all 48128 samples of the
// following second came out NaN (old) until reset().
DSPARK_TEST(CoreReg_biquad_block_recovers_from_nan)
{
    Biquad<float, 4> q;
    q.setCoeffsNow(BiquadCoeffs::makeLowPass(48000.0, 1000.0, 0.7));
    std::vector<float> a(512), b(512), c(512);
    float* p[3] = { a.data(), b.data(), c.data() };            // a channel pair and a single
    AudioBufferView<float> v(p, 3, 512);
    for (auto* ch : p) std::fill(ch, ch + 512, 0.1f);
    a[10] = b[20] = c[30] = std::numeric_limits<float>::quiet_NaN();
    q.processBlock(v);
    int bad = 0;
    for (int r = 0; r < 94; ++r)
    {
        for (auto* ch : p) std::fill(ch, ch + 512, 0.1f);
        q.processBlock(v);
        for (auto* ch : p) for (int i = 0; i < 512; ++i) bad += !std::isfinite(ch[i]);
    }
    EXPECT_EQ(bad, 0);                            // old: 144384 (3 x 48128)
    EXPECT_NEAR(a[511], 0.1f, 1e-4f);
}

DSPARK_TEST(CoreReg_svf_and_ladder_blocks_recover_from_nan)
{
    StateVariableFilter<float> svf;
    svf.prepare(spec(48000.0, 512, 1));
    LadderFilter<float> ladder;
    ladder.prepare(spec(48000.0, 512, 1));
    std::vector<float> x(512);
    float* p[1] = { x.data() };
    AudioBufferView<float> v(p, 1, 512);
    int badSvf = 0, badLadder = 0;
    for (int pass = 0; pass < 2; ++pass)
    {
        std::fill(x.begin(), x.end(), 0.1f);
        x[10] = std::numeric_limits<float>::quiet_NaN();
        if (pass == 0) svf.processBlock(v); else ladder.processBlock(v);
        for (int r = 0; r < 94; ++r)
        {
            std::fill(x.begin(), x.end(), 0.1f);
            if (pass == 0) svf.processBlock(v); else ladder.processBlock(v);
            for (float s : x) (pass == 0 ? badSvf : badLadder) += !std::isfinite(s);
        }
    }
    EXPECT_EQ(badSvf, 0);                         // old: 48128
    EXPECT_EQ(badLadder, 0);                      // old: 48128
}

// An infinite target passed the NaN-only guard and turned the exponential
// recursion into NaN that no later target recovered (old: NaN forever).
DSPARK_TEST(CoreReg_smoothed_value_ignores_infinite_targets)
{
    SmoothedValue<float> v;
    v.prepare(48000.0, 20.0);
    v.reset(1.0f);
    v.setTargetValue(std::numeric_limits<float>::infinity());
    (void)v.getNextValue();
    v.setTargetValue(0.5f);
    float last = 0.0f;
    for (int i = 0; i < 48000; ++i) last = v.getNextValue();
    EXPECT_EQ(last, 0.5f);
    EXPECT_FALSE(v.isSmoothing());
}

// Chase took 171 ms (0 -> 0.99) at 44.1 kHz but 53 ms at 192 kHz (old), and
// with rampTimeMs = 0 it still glided (first sample 0.0004) although the
// documentation promises a snap for every curve.
DSPARK_TEST(CoreReg_smoothed_value_chase_is_rate_independent)
{
    auto settleMs = [](double fs) {
        SmoothedValue<float> c;
        c.prepare(fs, 20.0);
        c.setSmoothingType(SmoothedValue<float>::SmoothingType::Chase);
        c.reset(0.0f);
        c.setTargetValue(1.0f);
        long n = 0;
        while (c.getNextValue() < 0.99f && n < 10000000) ++n;
        return 1000.0 * static_cast<double>(n) / fs;
    };
    const double a = settleMs(44100.0), b = settleMs(192000.0);
    EXPECT_NEAR(b / a, 1.0, 0.05);                // old: 0.31
    SmoothedValue<float> z;
    z.prepare(48000.0, 0.0);
    z.setSmoothingType(SmoothedValue<float>::SmoothingType::Chase);
    z.reset(0.0f);
    z.setTargetValue(1.0f);
    EXPECT_EQ(z.getNextValue(), 1.0f);            // old: 0.0004
}

// ============================================================================
// 3. Oversampling: stage-specific half-band lengths
// ============================================================================

// Every stage used the first stage's kernel. Later stages only have to stop
// what folds back into the first stage's band, so they are shorter: 4x/8x/16x
// High reported 96/112/120 samples (old), now 70/73/74, still exact and still
// inside the documented -80 dB.
DSPARK_TEST(CoreReg_oversampling_later_stages_are_shorter)
{
    const int expected[] = { 70, 73, 74 };
    int idx = 0;
    for (int factor : { 4, 8, 16 })
    {
        Oversampling<float> os(factor, Oversampling<float>::Quality::High);
        constexpr int bs = 2048;
        os.prepare(spec(48000.0, bs, 1));
        EXPECT_EQ(os.getLatency(), expected[idx++]);
        std::vector<float> x(bs, 0.0f);
        x[64] = 1.0f;
        float* p[1] = { x.data() };
        (void)os.upsample(AudioBufferView<const float>(p, 1, bs));
        os.downsample(AudioBufferView<float>(p, 1, bs));
        int argmax = 0;
        for (int i = 0; i < bs; ++i) if (std::fabs(x[i]) > std::fabs(x[argmax])) argmax = i;
        EXPECT_EQ(argmax, 64 + os.getLatency());

        // Every image of a 15 kHz tone at the top rate (k * 48 kHz +- 15 kHz,
        // from every stage) stays below the documented -80 dB. The analysis
        // window holds a whole number of cycles of each, so no leakage.
        os.reset();
        const int n = 64;
        std::vector<float> in(static_cast<size_t>(n) * 64), hi;
        for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<float>(std::sin(2.0 * kRegPi * 15000.0 / 48000.0 * static_cast<double>(i)));
        for (int b = 0; b < 64; ++b)
        {
            const float* q[1] = { in.data() + b * n };
            auto up = os.upsample(AudioBufferView<const float>(q, 1, n));
            if (b >= 32) hi.insert(hi.end(), up.getChannel(0), up.getChannel(0) + up.getNumSamples());
            float* o[1] = { x.data() };
            os.downsample(AudioBufferView<float>(o, 1, n));
        }
        const double topRate = 48000.0 * factor;
        const int len = static_cast<int>(hi.size());
        const double tone = toneAmplitude(hi.data(), len, 2.0 * kRegPi * 15000.0 / topRate);
        double worst = 0.0;
        for (int k = 1; k < factor; ++k)
            for (double f : { k * 48000.0 - 15000.0, k * 48000.0 + 15000.0 })
                if (f < 0.5 * topRate)
                    worst = std::max(worst, toneAmplitude(hi.data(), len, 2.0 * kRegPi * f / topRate));
        EXPECT_LT(20.0 * std::log10(worst / tone + 1e-30), -80.0);
    }
}

// ============================================================================
// 4. Oversampling: opt-in polyphase IIR mode
// ============================================================================

// New mode (absent before). 2x High: a few samples of delay instead of the
// FIR's 64, a deeper stopband than the FIR preset, an honest fractional delay.
DSPARK_TEST(CoreReg_oversampling_iir_mode_delay_and_stopband)
{
    using OS = Oversampling<float>;
    OS os(2, OS::Quality::High, OS::Mode::PolyphaseIir);
    constexpr int bs = 512;
    os.prepare(spec(48000.0, bs, 1));
    EXPECT_TRUE(os.getMode() == OS::Mode::PolyphaseIir);
    EXPECT_EQ(os.getLatency(), 3);                // FIR High: 64

    // 30 kHz at 96 kHz folds to 18 kHz: designed stopband 99 dB.
    std::vector<float> out;
    std::vector<float> blk(bs, 0.0f);
    long long t = 0;
    for (int b = 0; b < 32; ++b)
    {
        std::fill(blk.begin(), blk.end(), 0.0f);
        float* p[1] = { blk.data() };
        auto up = os.upsample(AudioBufferView<const float>(p, 1, bs));
        for (int i = 0; i < up.getNumSamples(); ++i)
            up.getChannel(0)[i] = 0.5f * static_cast<float>(std::sin(2.0 * kRegPi * 30000.0 / 96000.0 * static_cast<double>(t++)));
        os.downsample(AudioBufferView<float>(p, 1, bs));
        if (b >= 16) out.insert(out.end(), blk.begin(), blk.end());
    }
    const double alias = toneAmplitude(out.data(), static_cast<int>(out.size()), 2.0 * kRegPi * 18000.0 / 48000.0) / 0.5;
    EXPECT_LT(20.0 * std::log10(alias + 1e-30), -95.0);

    // Measured round-trip phase delay at 1 kHz matches getRoundTripDelay().
    os.reset();
    const int len = 48000;
    std::vector<float> x(len), y(len);
    for (int i = 0; i < len; ++i) x[i] = static_cast<float>(std::sin(2.0 * kRegPi * 1000.0 / 48000.0 * i));
    for (int b = 0; b + bs <= len; b += bs)
    {
        std::copy_n(x.data() + b, bs, y.data() + b);
        float* p[1] = { y.data() + b };
        (void)os.upsample(AudioBufferView<const float>(p, 1, bs));
        os.downsample(AudioBufferView<float>(p, 1, bs));
    }
    double c = 0.0, s = 0.0;
    const double w = 2.0 * kRegPi * 1000.0 / 48000.0;
    for (int i = 24000; i < 24000 + 48 * 400; ++i) { c += y[i] * std::cos(w * i); s += y[i] * std::sin(w * i); }
    const double measured = std::atan2(-c, s) / w;   // phase lag of the sine, in samples
    EXPECT_NEAR(measured, os.getRoundTripDelay(1000.0), 0.01);
    EXPECT_GT(os.getRoundTripDelay(18000.0), os.getRoundTripDelay(1000.0) + 0.5);   // nonlinear phase, reported
}

// The IIR mode is exact across block partitions and recovers from a NaN.
DSPARK_TEST(CoreReg_oversampling_iir_mode_partitions_and_nan)
{
    using OS = Oversampling<float>;
    OS a(4, OS::Quality::Maximum, OS::Mode::PolyphaseIir), b(4, OS::Quality::Maximum, OS::Mode::PolyphaseIir);
    a.prepare(spec(48000.0, 512, 1));
    b.prepare(spec(48000.0, 512, 1));
    std::mt19937 g(5);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<float> in(8192);
    for (auto& v : in) v = u(g);
    std::vector<float> x = in, y = in;
    for (int pos = 0; pos < 8192; pos += 512)
    {
        float* p[1] = { x.data() + pos };
        (void)a.upsample(AudioBufferView<const float>(p, 1, 512));
        a.downsample(AudioBufferView<float>(p, 1, 512));
    }
    std::uniform_int_distribution<int> len(1, 512);
    int pos = 0;
    while (pos < 8192)
    {
        const int n = std::min(len(g), 8192 - pos);
        float* p[1] = { y.data() + pos };
        (void)b.upsample(AudioBufferView<const float>(p, 1, n));
        b.downsample(AudioBufferView<float>(p, 1, n));
        pos += n;
    }
    EXPECT_TRUE(x == y);
    std::vector<float> z(512, 0.1f);
    z[3] = std::numeric_limits<float>::quiet_NaN();
    float* p[1] = { z.data() };
    (void)a.upsample(AudioBufferView<const float>(p, 1, 512));
    a.downsample(AudioBufferView<float>(p, 1, 512));
    std::fill(z.begin(), z.end(), 0.1f);
    (void)a.upsample(AudioBufferView<const float>(p, 1, 512));
    a.downsample(AudioBufferView<float>(p, 1, 512));
    EXPECT_NO_NAN(z.data(), 512);
}

// ============================================================================
// 5. Matched (de-cramped) LP, HP, BP and shelves
// ============================================================================

// Bilinear designs cramp near Nyquist. Worst error against the analog
// prototype over 20 Hz - 20 kHz at 48 kHz (bilinear): LP 15 kHz 9.41 dB,
// HP 18 kHz 12.46 dB, BP 5 kHz 8.58 dB.
DSPARK_TEST(CoreReg_matched_lowpass_highpass_bandpass_follow_analog)
{
    const double fs = 48000.0, Q = 0.7071;
    EXPECT_LT(worstAnalogError(BiquadCoeffs::makeLowPassMatched(fs, 15000.0, Q), 0, fs, 15000.0, Q, 0.0), 0.7);
    EXPECT_LT(worstAnalogError(BiquadCoeffs::makeHighPassMatched(fs, 18000.0, Q), 1, fs, 18000.0, Q, 0.0), 0.5);
    EXPECT_LT(worstAnalogError(BiquadCoeffs::makeBandPassMatched(fs, 5000.0, Q), 2, fs, 5000.0, Q, 0.0), 1.0);
    // Stable everywhere, unity DC gain for the low-pass, sane at low corners.
    for (double sr : { 44100.0, 48000.0, 192000.0, 768000.0 })
        for (double f0 : { 5.0, 20.0, 1000.0, 0.3 * sr, 0.49 * sr })
            for (double q : { 0.1, 0.5, 0.7071, 4.0, 20.0 })
            {
                const auto lp = BiquadCoeffs::makeLowPassMatched(sr, f0, q);
                const auto hp = BiquadCoeffs::makeHighPassMatched(sr, f0, q);
                const auto bp = BiquadCoeffs::makeBandPassMatched(sr, f0, q);
                EXPECT_TRUE(isStable(lp) && isStable(hp) && isStable(bp));
                EXPECT_NEAR((lp.b0 + lp.b1 + lp.b2) / (1.0 + lp.a1 + lp.a2), 1.0, 1e-6);
                EXPECT_NEAR(20.0 * std::log10(lp.getMagnitude(f0, sr)), 20.0 * std::log10(q), 0.05);
            }
}

// Bilinear Butterworth shelves deviate up to 3.93 dB from the analog shelf on
// the 44.1/48 kHz grid (old); the matched design stays within 0.4 dB.
DSPARK_TEST(CoreReg_matched_shelves_follow_analog)
{
    double worst = 0.0;
    for (double fs : { 44100.0, 48000.0 })
        for (double f0 : { 1000.0, 5000.0, 10000.0, 15000.0, 18000.0 })
            for (double g : { -12.0, 6.0, 12.0 })
            {
                worst = std::max(worst, worstAnalogError(BiquadCoeffs::makeLowShelfMatched(fs, f0, g), 3, fs, f0, 0.7071, g));
                worst = std::max(worst, worstAnalogError(BiquadCoeffs::makeHighShelfMatched(fs, f0, g), 4, fs, f0, 0.7071, g));
            }
    EXPECT_LT(worst, 0.5);
    for (double fs : { 44100.0, 192000.0, 768000.0 })
        for (double f0 : { 10.0, 100.0, 0.25 * fs, 0.49 * fs })
            for (double g : { -40.0, -24.0, -0.5, 0.5, 24.0, 40.0 })
            {
                const auto ls = BiquadCoeffs::makeLowShelfMatched(fs, f0, g);
                const auto hs = BiquadCoeffs::makeHighShelfMatched(fs, f0, g);
                EXPECT_TRUE(isStable(ls) && isStable(hs));
                EXPECT_NEAR(20.0 * std::log10(std::abs((ls.b0 + ls.b1 + ls.b2) / (1.0 + ls.a1 + ls.a2))), g, 1e-3);
                EXPECT_NEAR(20.0 * std::log10(std::abs((hs.b0 + hs.b1 + hs.b2) / (1.0 + hs.a1 + hs.a2))), 0.0, 1e-3);
            }
}

// The matched designs are reachable from the engines (opt-in, beside the
// matched-bell control): a 15 kHz Butterworth low-pass at 48 kHz now passes
// 20 kHz at the analog level instead of the bilinear one.
DSPARK_TEST(CoreReg_matched_filters_control_in_engines)
{
    const double fs = 48000.0, f = 20000.0;
    const double analog = analogDb(0, f, 15000.0, 0.7071, 0.0);
    auto measure = [&](bool matched, bool viaEq) {
        std::vector<float> x(48000);
        for (int i = 0; i < 48000; ++i) x[i] = static_cast<float>(std::sin(2.0 * kRegPi * f / fs * i));
        float* p[1] = { x.data() };
        if (viaEq)
        {
            auto eq = std::make_unique<Equalizer<float>>();
            eq->prepare(spec(fs, 48000, 1));
            eq->setMatchedFilters(matched);
            Equalizer<float>::BandConfig cfg;
            cfg.type = Equalizer<float>::BandType::LowPass;
            cfg.frequency = 15000.0f;
            cfg.q = 0.7071f;
            cfg.slope = 12;
            cfg.enabled = true;
            eq->setBand(0, cfg);
            eq->processBlock(AudioBufferView<float>(p, 1, 48000));
        }
        else
        {
            FilterEngine<float> fe;
            fe.prepare(spec(fs, 48000, 1));
            fe.setMatchedFilters(matched);
            fe.setLowPass(15000.0f, 0.7071f, 12);
            fe.processBlock(AudioBufferView<float>(p, 1, 48000));
        }
        return 20.0 * std::log10(toneAmplitude(x.data() + 24000, 24000, 2.0 * kRegPi * f / fs));
    };
    EXPECT_NEAR(measure(true, false), analog, 0.7);
    EXPECT_NEAR(measure(true, true), analog, 0.7);
    EXPECT_LT(measure(false, false), analog - 3.0);   // bilinear default unchanged
}

// ============================================================================
// 6. Block-loop optimisations, Hilbert zero-tap skip, fastLog bit split
// ============================================================================

// FIRFilter::processBlock now runs the register-blocked firCorrelate kernel
// (63 taps: 4.3 -> 2.8 ns/sample). It must equal that kernel exactly and be
// independent of the block partition. Old: the per-sample dot product
// differed from the kernel by up to 6e-8.
DSPARK_TEST(CoreReg_fir_block_path_is_the_block_kernel)
{
    const auto h = FIRDesign<float>::lowPass(48000.0, 5000.0, 63, 7.0f);
    FIRFilter<float> a, b;
    a.prepare(63, 1);
    b.prepare(63, 1);
    a.setCoefficients(h);
    b.setCoefficients(h);
    std::mt19937 g(9);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<float> in(4000);
    for (auto& v : in) v = u(g);
    std::vector<float> hist(62, 0.0f), ref(in.size()), rev(h.rbegin(), h.rend());
    hist.insert(hist.end(), in.begin(), in.end());
    simd::firCorrelate(hist.data(), rev.data(), 63, ref.data(), static_cast<int>(in.size()), 1.0f);
    std::vector<float> x = in, y = in;
    for (int pos = 0; pos < 4000; pos += 500)
    {
        float* p[1] = { x.data() + pos };
        a.processBlock(AudioBufferView<float>(p, 1, 500));
    }
    std::uniform_int_distribution<int> len(1, 700);
    int pos = 0;
    while (pos < 4000)
    {
        const int n = std::min(len(g), 4000 - pos);
        float* p[1] = { y.data() + pos };
        b.processBlock(AudioBufferView<float>(p, 1, n));
        pos += n;
    }
    EXPECT_TRUE(x == ref);
    EXPECT_TRUE(y == ref);
}

// A channel pair costs about one channel now (stereo 4.04 -> 1.96 ns/frame).
// Old: a stereo block took twice as long as a mono one.
DSPARK_TEST(CoreReg_biquad_channel_pair_runs_in_lock_step)
{
    if constexpr (kTimingReliable && simd::kVecDoubleNarrow == 2)
    {
    Biquad<float, 2> q;
    q.setCoeffsNow(BiquadCoeffs::makeLowPass(48000.0, 1000.0, 0.7));
    std::vector<float> l(4096, 0.1f), r(4096, 0.1f);
    float* p2[2] = { l.data(), r.data() };
    float* p1[1] = { l.data() };
    const double stereo = minNanoseconds([&] { for (int k = 0; k < 20; ++k) q.processBlock(AudioBufferView<float>(p2, 2, 4096)); });
    const double mono = minNanoseconds([&] { for (int k = 0; k < 20; ++k) q.processBlock(AudioBufferView<float>(p1, 1, 4096)); });
    gRegSink = l[7];
    EXPECT_LT(stereo, 1.5 * mono);                // old: about 2.0x
    }
}

// The SVF block loop keeps state and coefficients in locals (MSVC x64 LP
// 4.74 -> 2.41 ns/sample; GCC 15 at the default parameters 6.0 -> 2.3).
// Equivalence pin only: whether the per-sample path is slower depends on how
// each compiler inlines it into the caller, so a timing check against it
// failed on GCC and Clang with an unchanged block loop. The block path
// agrees with processSample() in every mode.
DSPARK_TEST(CoreReg_svf_block_loop_matches_per_sample)
{
    using SV = StateVariableFilter<float>;
    std::mt19937 g(4);
    std::uniform_real_distribution<float> u(-0.5f, 0.5f);
    std::vector<float> in(2048);
    for (auto& v : in) v = u(g);
    int off = 0;
    for (int m = 0; m < 8; ++m)
    {
        for (float gain : { -6.0f, 6.0f })
        {
            SV a, b;
            a.prepare(spec(48000.0, 2048, 1));
            b.prepare(spec(48000.0, 2048, 1));
            for (SV* f : { &a, &b }) { f->setMode(static_cast<SV::Mode>(m)); f->setCutoff(2000.0f); f->setQ(1.3f); f->setGain(gain); }
            std::vector<float> x = in;
            float* p[1] = { x.data() };
            a.processBlock(AudioBufferView<float>(p, 1, 2048));
            for (int i = 0; i < 2048; ++i) off += std::abs(x[i] - b.processSample(in[i], 0)) > 1e-6f;
        }
    }
    EXPECT_EQ(off, 0);
}

// fastLog read the exponent through frexp(); it now reads the bit pattern
// (float, dependent sum: MSVC x64 2.85 -> 1.63 ns, GCC 15 2.10 -> 1.60 ns).
// Equivalence pin only: a timing check against the frexp form written in this
// file passed or failed with the codegen of the surrounding tests. The bit
// split gives the frexp form's results (same mantissa and exponent; the
// tolerance only allows for compilers that contract the series differently).
DSPARK_TEST(CoreReg_fastlog_bit_split_matches_frexp_form)
{
    auto viaFrexp = [](float x) {
        int e = 0;
        float m = std::frexp(x, &e);
        if (m < static_cast<float>(0.70710678118654752440)) { m *= 2.0f; --e; }
        const float s = (m - 1.0f) / (m + 1.0f), s2 = s * s;
        const float lnm = 2.0f * s * (1.0f + s2 * (static_cast<float>(1.0 / 3.0)
                        + s2 * (static_cast<float>(1.0 / 5.0) + s2 * static_cast<float>(1.0 / 7.0))));
        return lnm + static_cast<float>(e) * std::numbers::ln2_v<float>;
    };
    std::mt19937 g(11);
    std::uniform_real_distribution<float> u(-40.0f, 40.0f);
    std::vector<float> xs(4096);
    for (auto& v : xs) v = std::exp2(u(g));
    int off = 0;
    for (float v : xs)
    {
        const float r = viaFrexp(v);
        off += std::abs(fastLog(v) - r) > 1e-6f * std::max(1.0f, std::abs(r));
    }
    EXPECT_EQ(off, 0);
    EXPECT_NEAR(fastLog(1e-40f), viaFrexp(1e-40f), 1e-4f);         // subnormal: frexp path
    EXPECT_NEAR(fastLog(1.0e300), std::log(1.0e300), 1e-6);
    EXPECT_NEAR(fastLog(3.0e-310), std::log(3.0e-310), 1e-6);      // subnormal double
}

// The Hilbert FIR convolved all 191 taps although 95 are exact zeros; it now
// convolves the 96 non-zero ones (SSE2: float 10.3 -> 6.6, double 27.2 ->
// 10.2 ns/sample). Equivalence pin only: a pure speed-up has no output that
// fails before, and a timing check against a hand-written full convolution
// proved codegen-dependent. The quadrature branch equals the full
// convolution to rounding and the in-phase branch is exactly x[n - 95].
DSPARK_TEST(CoreReg_hilbert_zero_tap_skip_is_equivalent)
{
    Hilbert<double> h;
    h.prepare(48000.0);
    std::mt19937 g(2);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::vector<double> x(3000);
    for (auto& v : x) v = u(g);
    double worst = 0.0;
    int realOff = 0;
    for (int n = 0; n < 3000; ++n)
    {
        const auto r = h.process(x[static_cast<size_t>(n)]);
        double ref = 0.0;
        for (int lag = 0; lag < 191; ++lag)
        {
            if (n - lag < 0) break;
            const int k = lag - 95;
            if (k % 2 == 0) continue;
            const double w = 0.42 - 0.5 * std::cos(2.0 * kRegPi * lag / 190.0) + 0.08 * std::cos(4.0 * kRegPi * lag / 190.0);
            ref += 2.0 / (kRegPi * k) * w * x[n - lag];
        }
        worst = std::max(worst, std::abs(r.imag - ref));
        realOff += r.real != (n >= 95 ? x[static_cast<size_t>(n - 95)] : 0.0);
    }
    EXPECT_LT(worst, 1e-12);
    EXPECT_EQ(realOff, 0);
}

// ============================================================================
// 7. StretchedSincReader cutoff; gain ramps
// ============================================================================

// The stretched kernel had its cutoff AT the output Nyquist: content at 1.05x
// the output Nyquist folded back at -13.4 dB (old), -6 dB at Nyquist.
DSPARK_TEST(CoreReg_stretched_sinc_does_not_fold_above_output_nyquist)
{
    StretchedSincReader<double> rd;
    const int step = rd.stepFor(2.0);
    const int mask = (1 << 16) - 1;
    std::vector<double> ring(1 << 16);
    auto level = [&](double ratio) {
        const double f = ratio * 0.25;   // cycles per input sample; output Nyquist is 0.25 at rate 2
        for (int i = 0; i <= mask; ++i) ring[static_cast<size_t>(i)] = std::sin(2.0 * kRegPi * f * i);
        double p = 0.0;
        int n = 0;
        for (int i = 1000; i < 30000; i += 2) { const double v = rd.readRing(ring.data(), mask, i, 0.25, step); p += v * v; ++n; }
        return 10.0 * std::log10(2.0 * p / n + 1e-30);
    };
    EXPECT_LT(level(1.05), -90.0);                // old: -13.4 dB
    EXPECT_LT(level(1.0), -40.0);                 // old: -6 dB
    EXPECT_NEAR(level(0.5), 0.0, 0.1);            // passband intact
}

// The SIMD ramp accumulated its gain: over 4.8M samples it drifted 1.3e-3
// from the indexed ramp and stepped at the vector/tail seam (old).
DSPARK_TEST(CoreReg_gain_ramp_does_not_drift)
{
    const int n = 4800003;
    std::vector<float> d(static_cast<size_t>(n), 1.0f);
    simd::applyGainRamp(d.data(), 1.0f, 0.0f, n);
    const float step = (0.0f - 1.0f) / static_cast<float>(n);
    double worst = 0.0;
    for (int i = 0; i < n; ++i)
        worst = std::max(worst, std::abs(static_cast<double>(d[static_cast<size_t>(i)]) - (1.0 + static_cast<double>(step) * i)));
    EXPECT_LT(worst, 1e-6);                       // old: 1.3e-3
    std::vector<double> dd(1000003, 0.0), src(1000003, 1.0);
    simd::addWithGainRamp(dd.data(), src.data(), 0.0, 2.0, 1000003);
    double worstD = 0.0;
    const double stepD = 2.0 / 1000003.0;
    for (int i = 0; i < 1000003; ++i) worstD = std::max(worstD, std::abs(dd[static_cast<size_t>(i)] - stepD * i));
    EXPECT_LT(worstD, 1e-12);
}

// ============================================================================
// 8. Dither: Lipshitz 5-tap shaper; NaN input
// ============================================================================

// New opt-in shaper: less noise than first order in 1-5 kHz at 44.1 kHz
// (measured -113.1 vs -108.2 dB for 16 bits), output still on the grid.
DSPARK_TEST(CoreReg_dither_lipshitz_shaper_lowers_midband_noise)
{
    auto bandNoiseDb = [](bool lipshitz) {
        Dither<double> d(16, true);
        if (lipshitz) d.setShaper(Dither<double>::Shaper::Lipshitz5);
        constexpr int n = 1 << 19;
        std::vector<double> e(n);
        bool onGrid = true;
        for (int i = 0; i < n; ++i)
        {
            e[static_cast<size_t>(i)] = d.processSample(0.0);
            const double lv = e[static_cast<size_t>(i)] * 32768.0;
            onGrid = onGrid && lv == std::round(lv);
        }
        Biquad<double, 1> hp, lp;
        hp.setCoeffsNow(BiquadCoeffs::makeHighPass(44100.0, 1000.0, 0.7071));
        lp.setCoeffsNow(BiquadCoeffs::makeLowPass(44100.0, 5000.0, 0.7071));
        double p = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double y = lp.processSampleCore(hp.processSampleCore(e[static_cast<size_t>(i)], 0), 0);
            if (i > 4096) p += y * y;
        }
        return std::pair<double, bool>(10.0 * std::log10(p / (n - 4096)), onGrid);
    };
    const auto first = bandNoiseDb(false), lip = bandNoiseDb(true);
    EXPECT_TRUE(first.second && lip.second);
    EXPECT_LT(lip.first, first.first - 3.0);

    // Overload recovery stays bounded with the higher-order loop.
    Dither<float> d(16, true);
    d.setShaper(Dither<float>::Shaper::Lipshitz5);
    for (int i = 0; i < 2000; ++i) (void)d.processSample(1.2f);
    float worst = 0.0f;
    for (int i = 0; i < 100; ++i) worst = std::max(worst, std::abs(d.processSample(0.0f)));
    EXPECT_LT(worst, 20.0f / 32768.0f);
}

// One NaN input poisoned the shaping error feedback: 100000 of the next
// 100000 samples came out NaN (old) until reset().
DSPARK_TEST(CoreReg_dither_nan_input_does_not_poison_feedback)
{
    Dither<float> d(16, true);
    (void)d.processSample(std::numeric_limits<float>::quiet_NaN());
    int bad = 0;
    for (int i = 0; i < 100000; ++i) bad += !std::isfinite(d.processSample(0.25f));
    EXPECT_EQ(bad, 0);                            // old: 100000
    EXPECT_EQ(d.processSample(std::numeric_limits<float>::infinity()), 32767.0f / 32768.0f);
}
