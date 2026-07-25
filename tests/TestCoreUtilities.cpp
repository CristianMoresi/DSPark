// DSPark Tests - Core Utilities
// Smoothers, SmoothedValue, Interpolation, RingBuffer, WaveshapeTable,
// SampleAndHold, Dither, Oversampling, DryWetMixer, ProcessorChain/Traits

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/Smoothers.h"
#include "../Core/SmoothedValue.h"
#include "../Core/Interpolation.h"
#include "../Core/RingBuffer.h"
#include "../Core/WaveshapeTable.h"
#include "../Core/SampleAndHold.h"
#include "../Core/Dither.h"
#include "../Core/Oversampling.h"
#include "../Core/DryWetMixer.h"
#include "../Core/ProcessorChain.h"
#include "../Core/ProcessorTraits.h"
#include "../Effects/Gain.h" // For ProcessorChain test

#include "../Core/ModulationRouter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// Smoothers
// ============================================================================

DSPARK_TEST(LinearSmoother_converges)
{
    Smoothers::LinearSmoother s;
    s.reset(44100.0, 10.0f, 0.0f); // 10ms ramp from 0
    s.setTargetValue(1.0f);

    // After 10ms worth of samples (~441), should be at target
    for (int i = 0; i < 500; ++i)
        (void)s.getNextValue();

    EXPECT_NEAR(s.getNextValue(), 1.0f, 0.01f);
}

DSPARK_TEST(ExponentialSmoother_converges)
{
    Smoothers::ExponentialSmoother s;
    s.reset(44100.0, 20.0f, 0.0f);
    s.setTargetValue(1.0f);

    // Run for a few time constants
    for (int i = 0; i < 4410; ++i) // 100ms
        (void)s.getNextValue();

    EXPECT_NEAR(s.getNextValue(), 1.0f, 0.05f);
}

DSPARK_TEST(OnePoleSmoother_converges)
{
    Smoothers::OnePoleSmoother s;
    s.reset(44100.0, 10.0f, 0.0f);
    s.setTargetValue(0.5f);

    for (int i = 0; i < 4410; ++i)
        (void)s.getNextValue();

    EXPECT_NEAR(s.getNextValue(), 0.5f, 0.05f);
}

DSPARK_TEST(Smoothers_isSmoothing)
{
    Smoothers::LinearSmoother s;
    s.reset(44100.0, 10.0f, 0.0f);
    s.setTargetValue(0.0f); // Already at target

    // Should not be smoothing when already at target
    EXPECT_FALSE(s.isSmoothing());

    s.setTargetValue(1.0f);
    EXPECT_TRUE(s.isSmoothing());
}

DSPARK_TEST(MultiPoleSmoother_honours_common_contract)
{
    Smoothers::MultiPoleSmoother<4> s;
    s.reset(48000.0, 10.0f, 0.0f);
    EXPECT_FALSE(s.isSmoothing());

    // Regression: right after a retarget only pole 0 knows the new target;
    // asking the LAST pole reported "settled" before the ramp even started.
    s.setTargetValue(1.0f);
    EXPECT_TRUE(s.isSmoothing());
    EXPECT_NEAR(s.getTargetValue(), 1.0f, 1e-9f);

    // The chain converges to the global target and reports settled.
    for (int i = 0; i < 48000; ++i)
        (void)s.getNextValue();
    EXPECT_NEAR(s.getCurrentValue(), 1.0f, 1e-4f);
    EXPECT_FALSE(s.isSmoothing());

    // Regression: skip() must land ON the target. Skipping each pole onto
    // its stale local target used to freeze the chain mid-way.
    s.setTargetValue(0.25f);
    (void)s.getNextValue();   // chain barely started moving
    s.skip();
    EXPECT_NEAR(s.getCurrentValue(), 0.25f, 1e-6f);
    EXPECT_NEAR(s.getNextValue(), 0.25f, 1e-6f);
    EXPECT_FALSE(s.isSmoothing());
}

DSPARK_TEST(SVF_and_Butterworth_smoothers_complete_their_overshoot)
{
    // Both are Q = 0.707 second-order responses: the step response crosses
    // the target at full speed and overshoots ~4%. With a tiny target every
    // per-sample step near the crossing is below epsilon, so an error-only
    // anti-denormal check snapped exactly at the crossing and truncated the
    // trajectory (max would equal the target instead of overshooting).
    const float target = 1e-3f;

    Smoothers::StateVariableSmoother svf;
    svf.reset(48000.0, 100.0f, 0.707f, 0.0f);
    svf.setTargetValue(target);
    float maxSvf = 0.0f;
    for (int i = 0; i < 96000; ++i)
        maxSvf = std::max(maxSvf, svf.getNextValue());
    EXPECT_GT(maxSvf, target * 1.03f);              // the overshoot survived
    EXPECT_NEAR(svf.getCurrentValue(), target, 1e-6f); // and it still settles
    EXPECT_FALSE(svf.isSmoothing());

    Smoothers::ButterworthSmoother bw;
    bw.reset(48000.0, 100.0f, 0.0f);
    bw.setTargetValue(target);
    float maxBw = 0.0f;
    for (int i = 0; i < 96000; ++i)
        maxBw = std::max(maxBw, bw.getNextValue());
    EXPECT_GT(maxBw, target * 1.03f);
    EXPECT_NEAR(bw.getCurrentValue(), target, 1e-6f);
    EXPECT_FALSE(bw.isSmoothing());
}

DSPARK_TEST(CriticallyDamped_smoother_never_overshoots)
{
    Smoothers::CriticallyDampedSmoother s;
    s.reset(48000.0, 20.0f, 0.0f);
    s.setTargetValue(1.0f);
    float maxVal = 0.0f;
    for (int i = 0; i < 48000; ++i)
        maxVal = std::max(maxVal, s.getNextValue());
    EXPECT_LT(maxVal, 1.0f + 1e-4f);  // zeta = 1: monotonic approach
    EXPECT_NEAR(s.getCurrentValue(), 1.0f, 1e-5f);
}

DSPARK_TEST(SlewLimiter_zero_or_negative_rate_freezes)
{
    // A negative rate used to flip std::clamp's bounds (undefined behaviour).
    // Contract now: rate <= 0 means the value may not move at all.
    Smoothers::SlewLimiter s;
    s.reset(48000.0, -100.0f, 0.5f);
    s.setTargetValue(1.0f);
    for (int i = 0; i < 100; ++i)
        (void)s.getNextValue();
    EXPECT_NEAR(s.getCurrentValue(), 0.5f, 1e-9f);

    // And a positive rate obeys its per-second budget exactly.
    Smoothers::SlewLimiter ok;
    ok.reset(1000.0, 100.0f, 0.0f);   // 0.1 per sample
    ok.setTargetValue(1.0f);
    (void)ok.getNextValue();
    (void)ok.getNextValue();
    EXPECT_NEAR(ok.getCurrentValue(), 0.2f, 1e-6f);
}

// ============================================================================
// SmoothedValue
// ============================================================================

DSPARK_TEST(SmoothedValue_exponential_converges)
{
    SmoothedValue<float> sv;
    sv.prepare(44100.0, 20.0);
    sv.setTargetValue(1.0f);

    for (int i = 0; i < 4410; ++i)
        (void)sv.getNextValue();

    EXPECT_NEAR(sv.getCurrentValue(), 1.0f, 0.05f);
}

DSPARK_TEST(SmoothedValue_linear)
{
    SmoothedValue<float> sv;
    sv.prepare(44100.0, 10.0);
    sv.setSmoothingType(SmoothedValue<float>::SmoothingType::Linear);
    sv.setTargetValue(1.0f);

    for (int i = 0; i < 500; ++i)
        (void)sv.getNextValue();

    EXPECT_NEAR(sv.getCurrentValue(), 1.0f, 0.01f);
}

DSPARK_TEST(SmoothedValue_disabled_instant)
{
    SmoothedValue<float> sv;
    sv.prepare(44100.0, 10.0);
    sv.setSmoothingType(SmoothedValue<float>::SmoothingType::Disabled);
    sv.setTargetValue(0.75f);

    float v = sv.getNextValue();
    EXPECT_NEAR(v, 0.75f, 1e-6f);
    EXPECT_FALSE(sv.isSmoothing());
}

// The recursion state must be double: in float the one-pole stalls hundreds
// of epsilons short of the target (2.6e-5 for a 20 ms ramp, 0.58% of the
// travel for a 2 s ramp at 96 kHz) and isSmoothing() stays true forever,
// pinning callers that gate work on it (Gain bulk path, CrossoverFilter
// coefficient updates) to their slow path. With double state the smoother
// snaps exactly onto the target ~16 time constants after a unit step.
DSPARK_TEST(SmoothedValue_settles_and_snaps_exactly)
{
    {
        SmoothedValue<float> sv;
        sv.prepare(44100.0, 20.0); // tau = 882 samples
        sv.setTargetValue(1.0f);
        long long n = 0;
        const long long cap = 20 * 882;
        while (sv.isSmoothing() && n < cap) { (void)sv.getNextValue(); ++n; }
        EXPECT_LT(n, cap);                          // settled at ~16 tau (perpetual before)
        EXPECT_TRUE(sv.getCurrentValue() == 1.0f);  // exact snap, not an approximation
    }
    {
        SmoothedValue<float> sv;
        sv.prepare(96000.0, 2000.0); // the long-ramp case: float stalled 0.58% short
        sv.setTargetValue(1.0f);
        long long n = 0;
        const long long cap = 20LL * 192000;
        while (sv.isSmoothing() && n < cap) { (void)sv.getNextValue(); ++n; }
        EXPECT_LT(n, cap);
        EXPECT_TRUE(sv.getCurrentValue() == 1.0f);
    }
}

// A default-constructed smoother (no prepare) must be functional at the
// documented 44.1 kHz / 20 ms defaults: Linear had a zero rate (parameter
// frozen forever) and Exponential a zero coefficient (instant jump, clicks).
DSPARK_TEST(SmoothedValue_default_constructed_is_functional)
{
    SmoothedValue<float> lin;
    lin.setSmoothingType(SmoothedValue<float>::SmoothingType::Linear);
    lin.setTargetValue(1.0f);
    int n = 0;
    while (lin.isSmoothing() && n < 100000) { (void)lin.getNextValue(); ++n; }
    EXPECT_NEAR(lin.getCurrentValue(), 1.0f, 1e-6f); // frozen at 0 before
    EXPECT_LT(n, 2000);                              // ~882 samples for the unit step

    SmoothedValue<float> expo; // Exponential is the default type
    expo.setTargetValue(1.0f);
    const float first = expo.getNextValue();
    EXPECT_GT(first, 0.0f);   // it moves...
    EXPECT_LT(first, 0.05f);  // ...but smoothly (instant jump to 1.0 before)
}

// setTargetValue(NaN) must be ignored: a NaN target poisons the recursion
// and the arrival checks permanently (every comparison goes false).
DSPARK_TEST(SmoothedValue_nan_target_is_ignored)
{
    SmoothedValue<float> sv;
    sv.prepare(48000.0, 10.0);
    sv.setTargetValue(0.5f);
    for (int i = 0; i < 100; ++i) (void)sv.getNextValue();

    sv.setTargetValue(std::numeric_limits<float>::quiet_NaN());
    EXPECT_NEAR(sv.getTargetValue(), 0.5f, 1e-9f);
    float v = 0.0f;
    for (int i = 0; i < 10000; ++i) v = sv.getNextValue(); // past the ~16 tau snap
    EXPECT_FALSE(std::isnan(v));
    EXPECT_NEAR(v, 0.5f, 1e-6f);
}

// processBlock is the same arithmetic as getNextValue, sample-exact,
// including the per-sample arrival snap, across all four curve types and
// with a retarget landing mid-run.
DSPARK_TEST(SmoothedValue_processBlock_matches_getNextValue)
{
    float maxDiff = 0.0f;
    for (int t = 0; t < 4; ++t)
    {
        SmoothedValue<float> a, b;
        a.prepare(48000.0, 5.0);
        b.prepare(48000.0, 5.0);
        a.setSmoothingType(static_cast<SmoothedValue<float>::SmoothingType>(t));
        b.setSmoothingType(static_cast<SmoothedValue<float>::SmoothingType>(t));
        a.setTargetValue(0.8f);
        b.setTargetValue(0.8f);

        float block[301];
        for (int chunk = 0; chunk < 8; ++chunk)
        {
            if (chunk == 4)
            {
                a.setTargetValue(-0.3f);
                b.setTargetValue(-0.3f);
            }
            a.processBlock(std::span<float>(block, 301));
            for (int i = 0; i < 301; ++i)
                maxDiff = std::max(maxDiff, std::abs(block[i] - b.getNextValue()));
        }
        EXPECT_TRUE(a.isSmoothing() == b.isSmoothing());
    }
    EXPECT_TRUE(maxDiff == 0.0f);
}

// ============================================================================
// Interpolation
// ============================================================================

DSPARK_TEST(Interpolation_linear_midpoint)
{
    float buf[] = { 0.0f, 1.0f, 2.0f, 3.0f };
    float val = interpolateLinear(buf, 4, 0.5f);
    EXPECT_NEAR(val, 0.5f, 1e-5f);

    float val2 = interpolateLinear(buf, 4, 1.5f);
    EXPECT_NEAR(val2, 1.5f, 1e-5f);
}

DSPARK_TEST(Interpolation_linear_exact)
{
    float buf[] = { 10.0f, 20.0f, 30.0f, 40.0f };
    float val = interpolateLinear(buf, 4, 2.0f);
    EXPECT_NEAR(val, 30.0f, 1e-5f);
}

DSPARK_TEST(Interpolation_hermite_at_integers_matches)
{
    // Hermite should return exact buffer values at integer positions
    float buf[] = { 0.0f, 0.3f, 0.7f, 1.0f, 0.8f, 0.4f, 0.1f, 0.0f };

    for (int i = 1; i < 6; ++i)
    {
        float val = interpolateHermite(buf, 8, static_cast<float>(i));
        EXPECT_NEAR(val, buf[i], 1e-5f);
    }

    // Also verify Hermite produces valid output at fractional positions
    for (float pos = 1.0f; pos < 6.0f; pos += 0.25f)
    {
        float val = interpolateHermite(buf, 8, pos);
        EXPECT_FALSE(std::isnan(val));
        // Should stay within a reasonable range of neighboring values
        EXPECT_TRUE(val >= -0.5f && val <= 1.5f);
    }
}

DSPARK_TEST(Interpolation_polynomial_reproduction_orders)
{
    // The defining property of each interpolator order: Linear reconstructs
    // degree-1 polynomials exactly, Hermite (Catmull-Rom) degree 2,
    // Lagrange degree 3. Sampled on the integer grid, evaluated mid-segment.
    auto p1 = [](double x) { return 0.7 * x - 0.3; };
    auto p2 = [](double x) { return 0.4 * x * x - 0.9 * x + 0.2; };
    auto p3 = [](double x) { return 0.3 * x * x * x - 0.5 * x * x + 0.2 * x + 0.1; };

    double maxLin = 0.0, maxHer = 0.0, maxLag = 0.0;
    for (int k = 1; k <= 4; ++k)
    {
        for (int s = 0; s <= 16; ++s)
        {
            const double frac = s / 16.0;
            const double x = k + frac;

            double lin = interpolateLinear(p1(k), p1(k + 1.0), frac);
            maxLin = std::max(maxLin, std::abs(lin - p1(x)));

            double her = interpolateHermite(p2(k - 1.0), p2(double(k)),
                                            p2(k + 1.0), p2(k + 2.0), frac);
            maxHer = std::max(maxHer, std::abs(her - p2(x)));

            double lag = interpolateLagrange(p3(k - 1.0), p3(double(k)),
                                             p3(k + 1.0), p3(k + 2.0), frac);
            maxLag = std::max(maxLag, std::abs(lag - p3(x)));
        }
    }
    EXPECT_LT(maxLin, 1e-12);
    EXPECT_LT(maxHer, 1e-12);
    EXPECT_LT(maxLag, 1e-12);

    // Float instantiation of the same property (looser tolerance).
    float lagF = interpolateLagrange(
        static_cast<float>(p3(1.0)), static_cast<float>(p3(2.0)),
        static_cast<float>(p3(3.0)), static_cast<float>(p3(4.0)), 0.375f);
    EXPECT_NEAR(lagF, static_cast<float>(p3(2.375)), 1e-5f);
}

DSPARK_TEST(Interpolation_hermite_is_c1_across_segments)
{
    // Catmull-Rom is C1: value AND first derivative are continuous at the
    // shared knot of two adjacent segments (tangent = central difference).
    const double y[] = { 0.12, -0.68, 0.91, 0.33, -0.55, 0.77, -0.21 };

    for (int k = 1; k <= 3; ++k)
    {
        // C0: end of segment k == start of segment k+1 == y[k+1] exactly.
        double endK = interpolateHermite(y[k - 1], y[k], y[k + 1], y[k + 2], 1.0);
        double startK1 = interpolateHermite(y[k], y[k + 1], y[k + 2], y[k + 3], 0.0);
        EXPECT_NEAR(endK, y[k + 1], 1e-12);
        EXPECT_NEAR(startK1, y[k + 1], 1e-12);

        // C1: numerical slope on both sides of the knot matches the central
        // difference tangent (y[k+2] - y[k]) / 2.
        const double h = 1e-6;
        double left = interpolateHermite(y[k - 1], y[k], y[k + 1], y[k + 2], 1.0 - h);
        double right = interpolateHermite(y[k], y[k + 1], y[k + 2], y[k + 3], h);
        double slopeLeft = (endK - left) / h;
        double slopeRight = (right - startK1) / h;
        const double tangent = (y[k + 2] - y[k]) * 0.5;
        EXPECT_NEAR(slopeLeft, tangent, 1e-4);
        EXPECT_NEAR(slopeRight, tangent, 1e-4);
    }
}

DSPARK_TEST(Interpolation_buffer_overloads_wrap_circularly)
{
    // The (buffer, length, position) overloads treat the buffer as circular.
    // Verify bit-exact agreement with the raw-sample overloads fed manually
    // wrapped neighbours, at both edges.
    const float buf[] = { 0.31f, -0.72f, 0.55f, 0.18f, -0.93f, 0.42f, -0.11f, 0.66f };
    const int n = 8;

    // Near the start: position 0.25 -> y0 wraps to buf[7].
    EXPECT_TRUE(interpolateHermite(buf, n, 0.25f)
                == interpolateHermite(buf[7], buf[0], buf[1], buf[2], 0.25f));
    EXPECT_TRUE(interpolateLagrange(buf, n, 0.25f)
                == interpolateLagrange(buf[7], buf[0], buf[1], buf[2], 0.25f));

    // Near the end: position 7.5 -> y2/y3 wrap to buf[0]/buf[1].
    EXPECT_TRUE(interpolateHermite(buf, n, 7.5f)
                == interpolateHermite(buf[6], buf[7], buf[0], buf[1], 0.5f));
    EXPECT_TRUE(interpolateLagrange(buf, n, 7.5f)
                == interpolateLagrange(buf[6], buf[7], buf[0], buf[1], 0.5f));
    EXPECT_TRUE(interpolateLinear(buf, n, 7.5f)
                == interpolateLinear(buf[7], buf[0], 0.5f));

    // Penultimate segment: position 6.5 -> only y3 wraps to buf[0].
    EXPECT_TRUE(interpolateHermite(buf, n, 6.5f)
                == interpolateHermite(buf[5], buf[6], buf[7], buf[0], 0.5f));

    // interpolateCubic is an exact alias of the Hermite buffer overload.
    EXPECT_TRUE(interpolateCubic(buf, n, 3.7f) == interpolateHermite(buf, n, 3.7f));
}

DSPARK_TEST(Interpolation_allpass_delay_magnitude_and_stability)
{
    // frac = 1 -> coefficient is exactly 0 -> pure one-sample delay, bit-exact.
    {
        float state = 0.0f;
        float prev = 0.87f;
        uint32_t rng = 12345u;
        for (int i = 0; i < 64; ++i)
        {
            rng = rng * 1664525u + 1013904223u;
            // Magnitudes in [0.25, 1.0] keep the signal above the denormal cut.
            float cur = (0.25f + 0.75f * static_cast<float>(rng >> 8)
                                        / static_cast<float>(1u << 24))
                        * ((rng & 1u) ? 1.0f : -1.0f);
            float out = interpolateAllpass(cur, prev, 1.0f, state);
            EXPECT_TRUE(out == prev);
            prev = cur;
        }
    }

    // Allpass property: unity magnitude at any frequency once settled.
    // 1 kHz at 48 kHz = 48 samples/cycle; measure over 500 exact cycles.
    {
        const double fs = 48000.0, f = 1000.0;
        const double w = 2.0 * 3.14159265358979323846 * f / fs;
        double state = 0.0, prev = 0.0;
        double sumIn = 0.0, sumOut = 0.0;
        for (int i = 0; i < 25000; ++i)
        {
            double cur = std::sin(w * i);
            double out = interpolateAllpass(cur, prev, 0.5, state);
            if (i >= 1000) { sumIn += cur * cur; sumOut += out * out; }
            prev = cur;
        }
        EXPECT_NEAR(std::sqrt(sumOut / sumIn), 1.0, 1e-3);
    }

    // Low-frequency phase delay equals frac: 100 Hz, frac = 0.3, projected
    // onto quadrature references over 50 exact cycles after settling.
    {
        const double fs = 48000.0, f = 100.0;
        const double w = 2.0 * 3.14159265358979323846 * f / fs;
        double state = 0.0, prev = 0.0;
        double reIn = 0.0, imIn = 0.0, reOut = 0.0, imOut = 0.0;
        const int settle = 4800, span = 24000; // 50 cycles of 480 samples
        for (int i = 0; i < settle + span; ++i)
        {
            double cur = std::sin(w * i);
            double out = interpolateAllpass(cur, prev, 0.3, state);
            if (i >= settle)
            {
                const double c = std::cos(w * i), s = std::sin(w * i);
                reIn += cur * c;  imIn += cur * s;
                reOut += out * c; imOut += out * s;
            }
            prev = cur;
        }
        // Projections over exact cycles give atan2(re, im) = signal phase;
        // a D-sample delay shifts the phase by -w*D, so D = (in - out) / w.
        const double phaseIn = std::atan2(reIn, imIn);
        const double phaseOut = std::atan2(reOut, imOut);
        double dphi = phaseIn - phaseOut;
        while (dphi > 3.14159265358979323846) dphi -= 2.0 * 3.14159265358979323846;
        while (dphi < -3.14159265358979323846) dphi += 2.0 * 3.14159265358979323846;
        const double delaySamples = dphi / w;
        EXPECT_NEAR(delaySamples, 0.3, 0.05);
    }

    // Smallest allowed frac (pole at -0.998): impulse response decays, never
    // blows up, no NaN.
    {
        float state = 0.0f, prev = 0.0f;
        float maxAbs = 0.0f, lastAbs = 0.0f;
        for (int i = 0; i < 30000; ++i)
        {
            float cur = (i == 0) ? 1.0f : 0.0f;
            float out = interpolateAllpass(cur, prev, 0.001f, state);
            EXPECT_FALSE(std::isnan(out));
            maxAbs = std::max(maxAbs, std::abs(out));
            lastAbs = std::abs(out);
            prev = cur;
        }
        EXPECT_LT(maxAbs, 2.0f);
        EXPECT_LT(lastAbs, 1e-3f);
    }
}

// ============================================================================
// RingBuffer
// ============================================================================

DSPARK_TEST(RingBuffer_FIFO_delay)
{
    RingBuffer<float> rb;
    rb.prepare(1024);

    // Push samples 0, 1, 2, ...
    for (int i = 0; i < 100; ++i)
        rb.push(static_cast<float>(i));

    // Read with delay 0 = most recent = 99
    EXPECT_NEAR(rb.read(0), 99.0f, 1e-6f);
    // Delay 1 = 98
    EXPECT_NEAR(rb.read(1), 98.0f, 1e-6f);
    // Delay 50 = 49
    EXPECT_NEAR(rb.read(50), 49.0f, 1e-6f);
}

DSPARK_TEST(RingBuffer_interpolated_at_integer_matches_read)
{
    RingBuffer<float> rb;
    rb.prepare(1024);

    for (int i = 0; i < 200; ++i)
        rb.push(static_cast<float>(i) * 0.1f);

    // At integer delay, interpolated should match exact read.
    // Start at 1: the default (Cubic) 4-point interpolator reads intDelay-1,
    // so a delay of 0 is out of its valid domain (delay >= 1 required).
    for (int d = 1; d < 50; ++d)
    {
        float exact = rb.read(d);
        float interp = rb.readInterpolated(static_cast<float>(d));
        EXPECT_NEAR(exact, interp, 1e-4f);
    }
}

DSPARK_TEST(RingBuffer_reset)
{
    RingBuffer<float> rb;
    rb.prepare(256);
    rb.push(42.0f);
    rb.reset();
    EXPECT_NEAR(rb.read(0), 0.0f, 1e-10f);
}

DSPARK_TEST(RingBuffer_pushBlock_matches_serial_pushes)
{
    // pushBlock's two-span memcpy path (including wraps, a block equal to the
    // capacity, and a block larger than the capacity) must leave the delay
    // line bit-identical to per-sample pushes.
    RingBuffer<float> blk, ref;
    blk.prepare(64); // capacity 64
    ref.prepare(64);

    float src[300];
    for (int i = 0; i < 300; ++i)
        src[i] = std::sin(0.11f * static_cast<float>(i)) + 0.01f * static_cast<float>(i);

    // Irregular blocks: sizes below, equal to and above the capacity, chosen
    // to cross the wrap boundary several times. Sums to 300.
    const int blocks[] = { 5, 33, 64, 90, 17, 64, 27 };
    int pos = 0;
    for (int b : blocks)
    {
        blk.pushBlock(src + pos, b);
        for (int i = 0; i < b; ++i) ref.push(src[pos + i]);
        pos += b;

        // Full logical content must agree after every block.
        for (int d = 0; d < 64; ++d)
            EXPECT_TRUE(blk.read(d) == ref.read(d));
    }
}

DSPARK_TEST(RingBuffer_unprepared_calls_are_safe_noops)
{
    // The constructor promises safe no-ops before prepare() for every method.
    RingBuffer<float> rb;
    EXPECT_TRUE(rb.getCapacity() == 0);

    rb.push(1.0f);
    float block[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    rb.pushBlock(block, 8);
    rb.reset();

    EXPECT_TRUE(rb.read(0) == 0.0f);
    EXPECT_TRUE(rb.read(100) == 0.0f);
    EXPECT_TRUE(rb.readInterpolated<InterpMethod::Linear>(2.5f) == 0.0f);
    EXPECT_TRUE(rb.readInterpolated(5.5f) == 0.0f); // Cubic default
    EXPECT_TRUE(rb.readInterpolated<InterpMethod::Lagrange>(5.5f) == 0.0f);
}

DSPARK_TEST(RingBuffer_fractional_interpolation_accuracy)
{
    // A fractional delay of a sine must reproduce the analytic delayed value;
    // 4-point methods clearly better than Linear.
    RingBuffer<float> rb;
    rb.prepare(1024);

    const double w = 2.0 * 3.14159265358979323846 * 997.0 / 48000.0;
    const int n = 600;
    for (int i = 0; i < n; ++i)
        rb.push(static_cast<float>(std::sin(w * i)));

    const float d = 7.5f;
    const float exact = static_cast<float>(std::sin(w * (n - 1 - 7.5)));

    const float lin = rb.readInterpolated<InterpMethod::Linear>(d);
    const float her = rb.readInterpolated<InterpMethod::Hermite>(d);
    const float lag = rb.readInterpolated<InterpMethod::Lagrange>(d);
    const float cub = rb.readInterpolated<InterpMethod::Cubic>(d);

    EXPECT_NEAR(lin, exact, 5e-3f);  // linear droops on a 997 Hz sine
    EXPECT_NEAR(her, exact, 1e-4f);
    EXPECT_NEAR(lag, exact, 5e-5f);
    EXPECT_TRUE(cub == her);         // Cubic is the Hermite kernel

    // Integer delay returns the exact stored sample for every method.
    for (int di = 1; di < 6; ++di)
    {
        const float s = rb.read(di);
        EXPECT_TRUE(rb.readInterpolated<InterpMethod::Linear>(static_cast<float>(di)) == s);
        EXPECT_TRUE(rb.readInterpolated<InterpMethod::Hermite>(static_cast<float>(di)) == s);
        EXPECT_TRUE(rb.readInterpolated<InterpMethod::Lagrange>(static_cast<float>(di)) == s);
    }
}

// ============================================================================
// WaveshapeTable
// ============================================================================

DSPARK_TEST(WaveshapeTable_identity_passthrough)
{
    WaveshapeTable<float> wst;
    wst.buildFromFunction([](float x) { return x; });

    for (float x = -1.0f; x <= 1.0f; x += 0.01f)
    {
        float out = wst.process(x);
        EXPECT_NEAR(out, x, 0.02f); // Table interpolation has slight error
    }
}

DSPARK_TEST(WaveshapeTable_tanh_bounded)
{
    WaveshapeTable<float> wst;
    wst.buildTanh();

    for (float x = -2.0f; x <= 2.0f; x += 0.1f)
    {
        float out = wst.process(x);
        EXPECT_TRUE(out >= -1.01f && out <= 1.01f);
    }
}

DSPARK_TEST(WaveshapeTable_hardClip)
{
    WaveshapeTable<float> wst;
    wst.buildHardClip(); // Default threshold = 0.8

    EXPECT_NEAR(wst.process(0.5f), 0.5f, 0.02f);
    // The table spans +-8, so 2.0 traces the curve and clips at the threshold
    EXPECT_NEAR(wst.process(2.0f), 0.8f, 0.02f);
    EXPECT_NEAR(wst.process(-2.0f), -0.8f, 0.02f);

    // With threshold = 1.0, clips at full scale
    WaveshapeTable<float> wst2;
    wst2.buildHardClip(1.0f);
    EXPECT_NEAR(wst2.process(0.5f), 0.5f, 0.02f);
    EXPECT_NEAR(wst2.process(2.0f), 1.0f, 0.02f);
}

DSPARK_TEST(WaveshapeTable_matches_function_and_edges)
{
    // The interpolated table must reproduce the stored function closely over
    // the whole +-8 domain, including exactly at both edges (the last-index
    // corner of the lookup).
    WaveshapeTable<float> wst;
    wst.buildTanh();

    float maxErr = 0.0f;
    for (int i = 0; i <= 3200; ++i)
    {
        const float x = -8.0f + 16.0f * static_cast<float>(i) / 3200.0f;
        maxErr = std::max(maxErr, std::abs(wst.process(x) - std::tanh(x)));
    }
    EXPECT_LT(maxErr, 1e-5f);

    EXPECT_NEAR(wst.process(8.0f), std::tanh(8.0f), 1e-6f);
    EXPECT_NEAR(wst.process(-8.0f), std::tanh(-8.0f), 1e-6f);

    // preGain drives along the curve; postGain scales the output.
    EXPECT_NEAR(wst.process(1.0f, 4.0f), std::tanh(4.0f), 1e-5f);
    EXPECT_NEAR(wst.process(0.5f, 1.0f, 2.0f), 2.0f * std::tanh(0.5f), 1e-5f);
}

DSPARK_TEST(WaveshapeTable_nan_and_inf_inputs_stay_finite)
{
    // NaN or infinite inputs (or a NaN preGain) must resolve to a table edge
    // and produce a finite output instead of indexing out of bounds.
    WaveshapeTable<float> wst;
    wst.buildTanh();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    EXPECT_TRUE(std::isfinite(wst.process(nan)));
    EXPECT_NEAR(wst.process(inf), std::tanh(8.0f), 1e-5f);
    EXPECT_NEAR(wst.process(-inf), std::tanh(-8.0f), 1e-5f);
    EXPECT_TRUE(std::isfinite(wst.process(0.5f, nan)));

    // Block path with a NaN in the middle: every sample stays finite.
    float block[8] = { 0.1f, -0.2f, nan, 0.4f, inf, -0.6f, 0.7f, -inf };
    wst.process(block, 8);
    for (int i = 0; i < 8; ++i) EXPECT_TRUE(std::isfinite(block[i]));
}

DSPARK_TEST(WaveshapeTable_oversampling_lifecycle_and_getters)
{
    WaveshapeTable<float> wst;
    wst.buildTanh();

    AudioSpec spec { 48000.0, 256, 2 };
    wst.prepare(spec);
    EXPECT_TRUE(wst.getOversamplingFactor() == 1);
    EXPECT_TRUE(wst.getLatency() == 0);

    wst.setOversampling(4);
    EXPECT_TRUE(wst.getOversamplingFactor() == 4);
    EXPECT_TRUE(wst.getLatency() > 0);

    // Oversampled block processing produces bounded, non-silent output.
    AudioBuffer<float> buf;
    buf.resize(2, 256);
    double rmsAcc = 0.0;
    for (int blk = 0; blk < 8; ++blk)
    {
        for (int ch = 0; ch < 2; ++ch)
        {
            float* d = buf.toView().getChannel(ch);
            for (int i = 0; i < 256; ++i)
                d[i] = 0.8f * std::sin(0.13f * static_cast<float>(blk * 256 + i));
        }
        wst.processBlock(buf.toView(), 2.0f);
        const float* d = buf.toView().getChannel(0);
        for (int i = 0; i < 256; ++i)
        {
            // 1.1 bound: tanh caps at ~0.92 here; the downsampler FIR may add
            // a little ringing on top.
            EXPECT_TRUE(std::isfinite(d[i]) && std::abs(d[i]) <= 1.1f);
            rmsAcc += static_cast<double>(d[i]) * static_cast<double>(d[i]);
        }
    }
    EXPECT_GT(std::sqrt(rmsAcc / (8 * 256)), 0.1); // not silence

    // Back to 1x: getter and latency return to the direct path.
    wst.setOversampling(1);
    EXPECT_TRUE(wst.getOversamplingFactor() == 1);
    EXPECT_TRUE(wst.getLatency() == 0);
}

// ============================================================================
// SampleAndHold
// ============================================================================

DSPARK_TEST(SampleAndHold_hold1_passthrough)
{
    SampleAndHold<float> sh;
    sh.setHoldSamples(1);

    for (float x = 0.0f; x < 10.0f; x += 1.0f)
    {
        float out = sh.process(x);
        EXPECT_NEAR(out, x, 1e-6f);
    }
}

DSPARK_TEST(SampleAndHold_holdN_repeats)
{
    SampleAndHold<float> sh;
    sh.setHoldSamples(4);
    sh.reset(); // counter = 0: the initial held value is output until a full period elapses

    // The first (holdPeriod-1) calls output the initial held value (0); the capture
    // happens on the holdPeriod-th call, then that value is held for the next period.
    EXPECT_NEAR(sh.process(1.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(sh.process(2.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(sh.process(3.0f), 0.0f, 1e-6f);
    float captured = sh.process(4.0f); // 4th call latches the current input
    EXPECT_NEAR(captured, 4.0f, 1e-6f);
    EXPECT_NEAR(sh.process(5.0f), captured, 1e-6f);
    EXPECT_NEAR(sh.process(6.0f), captured, 1e-6f);
}

DSPARK_TEST(SampleAndHold_trigger_mode_level_semantics)
{
    // Trigger mode is level-sensitive: a single-sample pulse captures and
    // holds; a sustained trigger tracks the input sample-by-sample.
    SampleAndHold<float> sh;
    sh.setMode(SampleAndHold<float>::Mode::Trigger);
    sh.reset(0.25f);

    // No trigger: holds the initial value indefinitely.
    EXPECT_NEAR(sh.process(1.0f), 0.25f, 1e-6f);
    EXPECT_NEAR(sh.process(2.0f), 0.25f, 1e-6f);

    // Single-sample pulse: captures that input, then holds it.
    EXPECT_NEAR(sh.process(3.0f, true), 3.0f, 1e-6f);
    EXPECT_NEAR(sh.process(4.0f), 3.0f, 1e-6f);
    EXPECT_NEAR(sh.getHeldValue(), 3.0f, 1e-6f);

    // Sustained trigger: tracks every sample while true.
    EXPECT_NEAR(sh.process(5.0f, true), 5.0f, 1e-6f);
    EXPECT_NEAR(sh.process(6.0f, true), 6.0f, 1e-6f);
    EXPECT_NEAR(sh.process(7.0f), 6.0f, 1e-6f);

    // Block overload with a trigger buffer matches the per-sample twin
    // bit-exactly.
    SampleAndHold<float> blk, ref;
    blk.setMode(SampleAndHold<float>::Mode::Trigger);
    ref.setMode(SampleAndHold<float>::Mode::Trigger);
    blk.reset(0.5f);
    ref.reset(0.5f);

    float data[64];
    bool trig[64] = {};
    for (int i = 0; i < 64; ++i) data[i] = 0.01f * static_cast<float>(i * 7 % 23);
    trig[5] = trig[6] = trig[40] = true;

    float expected[64];
    for (int i = 0; i < 64; ++i) expected[i] = ref.process(data[i], trig[i]);
    blk.processBlock(data, trig, 64);
    for (int i = 0; i < 64; ++i) EXPECT_TRUE(data[i] == expected[i]);

    // Null trigger buffer is treated as "no triggers": output holds.
    float held = blk.getHeldValue();
    float more[8] = { 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f };
    blk.processBlock(more, nullptr, 8);
    for (int i = 0; i < 8; ++i) EXPECT_TRUE(more[i] == held);
}

DSPARK_TEST(SampleAndHold_setHoldRate_rounds_and_rejects_invalid)
{
    // 48000 / 9601 = 4.9995: nearest period is 5 (truncation would give 4 and
    // bias the effective rate upward by a full step).
    SampleAndHold<float> sh;
    sh.setHoldRate(9601.0, 48000.0);
    sh.reset(0.0f);
    EXPECT_NEAR(sh.process(1.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(sh.process(2.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(sh.process(3.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(sh.process(4.0f), 0.0f, 1e-6f); // truncated period 4 captures here
    EXPECT_NEAR(sh.process(5.0f), 5.0f, 1e-6f); // rounded period 5 captures here

    // Exact ratio stays exact.
    sh.setHoldRate(12000.0, 48000.0);
    sh.reset(0.0f);
    for (int i = 1; i <= 3; ++i) EXPECT_NEAR(sh.process(static_cast<float>(i)), 0.0f, 1e-6f);
    EXPECT_NEAR(sh.process(4.0f), 4.0f, 1e-6f);

    // Invalid rates (NaN / zero / negative) reset the period to 1 (transparent).
    const double nan = std::numeric_limits<double>::quiet_NaN();
    sh.setHoldRate(nan, 48000.0);
    EXPECT_NEAR(sh.process(0.7f), 0.7f, 1e-6f);
    sh.setHoldRate(9600.0, nan);
    EXPECT_NEAR(sh.process(0.8f), 0.8f, 1e-6f);
    sh.setHoldRate(0.0, 48000.0);
    EXPECT_NEAR(sh.process(0.9f), 0.9f, 1e-6f);

    // A huge ratio clamps instead of overflowing the int cast: the processor
    // simply holds for a very long time.
    sh.setHoldRate(1.0e-300, 48000.0);
    sh.reset(0.33f);
    for (int i = 0; i < 1000; ++i)
        EXPECT_TRUE(sh.process(1.0f) == 0.33f);
}

DSPARK_TEST(SampleAndHold_block_matches_per_sample_counter)
{
    // Counter-mode block processing across irregular block boundaries is
    // bit-identical to the per-sample path.
    SampleAndHold<float> blk, ref;
    blk.setHoldSamples(7);
    ref.setHoldSamples(7);
    blk.reset(0.1f);
    ref.reset(0.1f);

    float src[257];
    for (int i = 0; i < 257; ++i)
        src[i] = std::sin(0.37f * static_cast<float>(i)) + 0.001f * static_cast<float>(i);

    float expected[257];
    for (int i = 0; i < 257; ++i) expected[i] = ref.process(src[i]);

    float data[257];
    for (int i = 0; i < 257; ++i) data[i] = src[i];
    const int blocks[] = { 1, 6, 32, 100, 118 }; // sums to 257
    int pos = 0;
    for (int b : blocks)
    {
        blk.processBlock(data + pos, b);
        pos += b;
    }
    for (int i = 0; i < 257; ++i) EXPECT_TRUE(data[i] == expected[i]);
}

// ============================================================================
// Dither
// ============================================================================

DSPARK_TEST(Dither_output_in_range)
{
    Dither<float> d(16);

    for (int i = 0; i < 1000; ++i)
    {
        float in = 0.5f * std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);
        float out = d.processSample(in);
        EXPECT_FALSE(std::isnan(out));
        EXPECT_FALSE(std::isinf(out));
    }
}

DSPARK_TEST(Dither_RMS_similar_to_input)
{
    Dither<float> d(16);

    float inSum = 0.0f, outSum = 0.0f;
    constexpr int N = 44100;
    for (int i = 0; i < N; ++i)
    {
        float in = 0.5f * std::sin(twoPi<float> * 440.0f * static_cast<float>(i) / 44100.0f);
        float out = d.processSample(in);
        inSum += in * in;
        outSum += out * out;
    }
    float inRMS = std::sqrt(inSum / N);
    float outRMS = std::sqrt(outSum / N);

    // RMS should be similar (dithering adds a tiny bit of noise)
    EXPECT_NEAR(outRMS, inRMS, 0.01f);
}

namespace {

// Mean power (dB) of x at one DFT bin over Hann-windowed segments of length N.
inline double ditherBandPowerDb(const std::vector<float>& x, int N, int bin)
{
    const int segs = static_cast<int>(x.size()) / N;
    double acc = 0.0;
    for (int s = 0; s < segs; ++s)
    {
        double re = 0.0, im = 0.0;
        for (int n = 0; n < N; ++n)
        {
            const double w = 0.5 - 0.5 * std::cos(twoPi<double> * n / N);
            const double v = w * x[static_cast<size_t>(s) * N + n];
            const double ph = twoPi<double> * bin * n / N;
            re += v * std::cos(ph);
            im -= v * std::sin(ph);
        }
        acc += re * re + im * im;
    }
    return 10.0 * std::log10(acc / segs + 1e-30);
}

} // namespace

// The shaper must feed back the TOTAL requantisation error (dither +
// quantiser). Feeding back only the quantiser error leaves the dither flat
// and caps the low-frequency benefit at ~1.8 dB; the correct loop measures
// ~-17 dB at 1 kHz relative to flat dither, with the classic first-order
// rise towards Nyquist (~+7 dB at 20 kHz, 44.1 kHz frame).
DSPARK_TEST(Dither_noise_shaping_shapes_the_whole_floor)
{
    const int N = 4096, segs = 48;
    std::vector<float> flat(static_cast<size_t>(N) * segs, 0.0f);
    std::vector<float> shaped(flat);

    Dither<float> dFlat(16, false);
    dFlat.processBlock(flat.data(), static_cast<int>(flat.size()));
    Dither<float> dShaped(16, true);
    dShaped.processBlock(shaped.data(), static_cast<int>(shaped.size()));

    // Bin 93 ~ 1 kHz, bin 1858 ~ 20 kHz at fs 44100.
    const double lfDelta = ditherBandPowerDb(shaped, N, 93) - ditherBandPowerDb(flat, N, 93);
    const double hfDelta = ditherBandPowerDb(shaped, N, 1858) - ditherBandPowerDb(flat, N, 1858);
    EXPECT_LT(lfDelta, -10.0); // measured -16.8 dB; only-quantiser feedback gives -1.6
    EXPECT_LT(hfDelta, 9.0);   // measured +6.9 dB; the shaping trade, bounded
    EXPECT_GT(hfDelta, 2.0);   // and it must actually tilt (flat dither would give ~0)
}

// Every output must lie exactly on the integer grid of the target depth,
// clamped to the representable range [-2^(b-1), 2^(b-1) - 1]: a level of
// +32768 does not exist in int16, so positive full scale quantises one step
// below 1.0 and the WavFile writer round-trips the result bit-exactly.
DSPARK_TEST(Dither_output_lies_on_int16_grid_including_full_scale)
{
    for (bool shaping : { false, true })
    {
        Dither<float> d(16, shaping);
        int offGrid = 0, outOfRange = 0;
        for (int i = 0; i < 200000; ++i)
        {
            const float in = -1.0f + 2.0f * (static_cast<float>(i % 20001) / 20000.0f);
            const double lvl = static_cast<double>(d.processSample(in)) * 32768.0;
            if (std::abs(lvl - std::round(lvl)) > 1e-3) ++offGrid;
            if (lvl > 32767.0 || lvl < -32768.0) ++outOfRange;
        }
        EXPECT_EQ(offGrid, 0);
        EXPECT_EQ(outOfRange, 0);
    }
}

// The error feedback is capped at 2 LSB (a legitimate error never exceeds
// 1.5): an overload must not accumulate clip error into the loop. Without
// the cap, 1000 samples at +1.2 pin the output at full scale for ~200
// samples after the overload ends. Out-of-range channel indices must fall
// back to plain dithering without touching the shaping state.
DSPARK_TEST(Dither_shaper_recovers_after_overload_and_guards_channel)
{
    Dither<float> d(16, true);
    for (int i = 0; i < 1000; ++i) (void)d.processSample(1.2f);
    EXPECT_LT(std::abs(d.processSample(0.0f)), 0.5f);

    for (int i = 0; i < 5000; ++i) (void)d.processSample(1.0f); // legal full scale
    EXPECT_LT(std::abs(d.processSample(0.0f)), 0.5f);

    // Negative / oversized channel: valid dithered output, no state damage
    // (the ASan CI job would flag the old out-of-bounds write on -1).
    for (int ch : { -1, 99 })
    {
        const float y = d.processSample(0.25f, ch);
        const double lvl = static_cast<double>(y) * 32768.0;
        EXPECT_LT(std::abs(lvl - std::round(lvl)), 1e-3);
    }
    EXPECT_LT(std::abs(d.processSample(0.0f)), 0.5f); // channel 0 still sane
}

// ============================================================================
// Oversampling
// ============================================================================

DSPARK_TEST(Oversampling_2x_roundtrip)
{
    Oversampling<float> os(2, Oversampling<float>::Quality::Low);
    constexpr int blockSize = 256;
    auto s = spec(44100.0, blockSize, 1);
    os.prepare(s);

    // Process several blocks to let the FIR filters settle past their latency
    for (int warmup = 0; warmup < 10; ++warmup)
    {
        auto tb = makeMonoBuffer(blockSize);
        tb.fillSine(440.0f, 44100.0f);
        (void)os.upsample(tb.view());
        os.downsample(tb.view());
    }

    // Now process one more block and measure quality
    auto tb = makeMonoBuffer(blockSize);
    tb.fillSine(440.0f, 44100.0f);

    std::vector<float> original(tb.ch(0), tb.ch(0) + blockSize);

    (void)os.upsample(tb.view());
    os.downsample(tb.view());

    EXPECT_NO_NAN(tb.ch(0), blockSize);
    EXPECT_BOUNDED(tb.ch(0), blockSize, -1.5f, 1.5f);

    // After settling, the signal should be well preserved (compensate for FIR latency)
    int latency = os.getLatency();
    float errSum = 0.0f, sigSum = 0.0f;
    int start = latency + 16;          // past transient from block boundary
    int end = blockSize - 16;          // avoid end edge effects
    for (int i = start; i < end; ++i)
    {
        float err = tb.ch(0)[i] - original[i - latency];
        errSum += err * err;
        sigSum += original[i - latency] * original[i - latency];
    }
    float snrDb = -10.0f * std::log10(errSum / (sigSum + 1e-30f));
    EXPECT_GT(snrDb, 40.0f); // High SNR for in-band signal after latency compensation
}

// Regression guard: the up/down round-trip must stay sample-continuous even when
// the host feeds VARIABLE block sizes (automation, loop points). A previous
// upsample implementation shifted its history by the current block size at the
// start of each call, which only restored the previous tail correctly for a
// constant block size; variable blocks corrupted the output. Covers x2..x16.
DSPARK_TEST(Oversampling_variable_block_roundtrip)
{
    const int factors[] = { 2, 4, 8, 16 };
    for (int factor : factors)
    {
        Oversampling<float> os(factor, Oversampling<float>::Quality::High);
        constexpr int maxBlock = 512;
        os.prepare(spec(48000.0, maxBlock, 1));
        const int latency = os.getLatency();

        constexpr int total = 8192;
        std::vector<float> ref(total), out(total, 0.0f);
        for (int i = 0; i < total; ++i)
            ref[i] = 0.5f * std::sin(2.0f * 3.14159265f * 1000.0f * static_cast<float>(i) / 48000.0f);

        auto blk = makeMonoBuffer(maxBlock);
        unsigned int rng = 2246822519u + static_cast<unsigned int>(factor);
        int pos = 0;
        while (pos < total)
        {
            rng = rng * 1664525u + 1013904223u;
            int bs = 1 + static_cast<int>(rng % static_cast<unsigned int>(maxBlock));
            if (bs > total - pos) bs = total - pos;

            for (int i = 0; i < bs; ++i) blk.ch(0)[i] = ref[pos + i];
            auto sub = blk.view().getSubView(0, bs);
            (void)os.upsample(sub);
            os.downsample(sub);
            for (int i = 0; i < bs; ++i) out[pos + i] = blk.ch(0)[i];
            pos += bs;
        }

        EXPECT_NO_NAN(out.data(), total);

        float errSum = 0.0f, sigSum = 0.0f;
        for (int i = latency + 1000; i < total - 16; ++i)
        {
            float e = out[i] - ref[i - latency];
            errSum += e * e;
            sigSum += ref[i - latency] * ref[i - latency];
        }
        float snrDb = -10.0f * std::log10(errSum / (sigSum + 1e-30f));
        EXPECT_GT(snrDb, 40.0f); // continuous round-trip regardless of block size
    }
}

// getLatency() must match the REAL group delay: an impulse fed through the
// up/down round-trip has to peak exactly getLatency() samples late, for every
// factor and for both tap-count extremes. This also pins the polyphase phase
// alignment (a misaligned even/centre pair or a broken SIMD kernel would move
// or split the peak).
DSPARK_TEST(Oversampling_impulse_roundtrip_latency_is_exact)
{
    const int factors[] = { 2, 4, 8, 16 };
    const Oversampling<float>::Quality qualities[] = {
        Oversampling<float>::Quality::Low, Oversampling<float>::Quality::Maximum };

    for (auto q : qualities)
    {
        for (int factor : factors)
        {
            Oversampling<float> os(factor, q);
            constexpr int blockSize = 2048;
            os.prepare(spec(48000.0, blockSize, 1));

            auto tb = makeMonoBuffer(blockSize);
            tb.fillSilence();
            constexpr int impulseAt = 64;
            tb.ch(0)[impulseAt] = 1.0f;

            (void)os.upsample(tb.view());
            os.downsample(tb.view());

            int argmax = 0;
            float peak = 0.0f;
            for (int i = 0; i < blockSize; ++i)
            {
                float a = std::fabs(tb.ch(0)[i]);
                if (a > peak) { peak = a; argmax = i; }
            }

            EXPECT_EQ(argmax, impulseAt + os.getLatency());
            EXPECT_GT(peak, 0.7f); // impulse passes near-unity (minus supra-Nyquist energy)
        }
    }
}

// The Quality enum promises a stopband depth per preset (~-40/-60/-80/-100 dB).
// Feed a tone INSIDE the decimator's stopband at the high rate (the situation a
// nonlinear stage creates) and measure what leaks back, aliased, into the base
// band. Coherent frequencies (30 kHz @ 96k -> alias at 18 kHz @ 48k, both exact
// bins of the 4096-sample window) keep the measurement floor at ~-140 dB.
DSPARK_TEST(Oversampling_stopband_matches_quality_preset)
{
    struct Case { Oversampling<float>::Quality q; float maxDb; };
    const Case cases[] = {
        { Oversampling<float>::Quality::Low,     -35.0f },
        { Oversampling<float>::Quality::Medium,  -55.0f },
        { Oversampling<float>::Quality::High,    -75.0f },
        { Oversampling<float>::Quality::Maximum, -90.0f },
    };

    constexpr int blockSize = 512;
    constexpr int total = 8192;
    constexpr double srHigh = 96000.0;
    constexpr double toneHz = 30000.0;  // stopband of the 2x half-band at 96k
    constexpr float toneAmp = 0.5f;

    for (const auto& c : cases)
    {
        Oversampling<float> os(2, c.q);
        os.prepare(spec(48000.0, blockSize, 1));

        std::vector<float> out(total, 0.0f);
        auto blk = makeMonoBuffer(blockSize);
        int pos = 0;
        while (pos < total)
        {
            blk.fillSilence();
            auto up = os.upsample(blk.view());
            // Inject the supra-Nyquist tone at the high rate with block-continuous phase.
            float* d = up.getChannel(0);
            for (int i = 0; i < up.getNumSamples(); ++i)
            {
                const double n = static_cast<double>(pos) * 2.0 + static_cast<double>(i);
                d[i] = toneAmp * static_cast<float>(std::sin(dspark::twoPi<double> * toneHz * n / srHigh));
            }
            os.downsample(blk.view());
            for (int i = 0; i < blockSize; ++i) out[pos + i] = blk.ch(0)[i];
            pos += blockSize;
        }

        // Measure the aliased image (96k - 2*... folds 30 kHz to 18 kHz at 48k)
        // over the settled second half.
        const float aliasAmp = measureFrequencyMagnitude(out.data() + total / 2, total / 2,
                                                         18000.0f, 48000.0f);
        const float aliasDb = 20.0f * std::log10(aliasAmp / toneAmp + 1e-30f);
        EXPECT_LT(aliasDb, c.maxDb);
    }
}

// Per-channel FIR histories must be fully independent: a hot left channel may
// not leak a single sample into a silent right channel across the round-trip.
DSPARK_TEST(Oversampling_channels_are_isolated)
{
    Oversampling<float> os(4, Oversampling<float>::Quality::High);
    constexpr int blockSize = 256;
    os.prepare(spec(48000.0, blockSize, 2));

    float peakSilent = 0.0f, peakHot = 0.0f;
    for (int b = 0; b < 8; ++b)
    {
        auto tb = makeStereoBuffer(blockSize);
        tb.fillSilence();
        generateSine(tb.ch(0), blockSize, 1000.0f, 48000.0f, 0.8f);

        (void)os.upsample(tb.view());
        os.downsample(tb.view());

        for (int i = 0; i < blockSize; ++i)
        {
            peakHot    = std::max(peakHot, std::fabs(tb.ch(0)[i]));
            peakSilent = std::max(peakSilent, std::fabs(tb.ch(1)[i]));
        }
    }

    EXPECT_GT(peakHot, 0.5f);       // signal survives on the driven channel
    EXPECT_EQ(peakSilent, 0.0f);    // and never crosses into the silent one
}

// ============================================================================
// DryWetMixer
// ============================================================================

DSPARK_TEST(DryWetMixer_mix0_returns_dry)
{
    DryWetMixer<float> mixer;
    auto s = defaultSpec();
    mixer.prepare(s);

    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);

    // Save dry
    std::vector<float> dryCopy(tb.ch(0), tb.ch(0) + 256);

    mixer.pushDry(tb.view());

    // Simulate effect: zero out the buffer (extreme wet)
    tb.view().clear();

    // Mix at 0 = 100% dry
    mixer.mixWet(tb.view(), 0.0f);

    for (int i = 0; i < 256; ++i)
        EXPECT_NEAR(tb.ch(0)[i], dryCopy[i], 1e-5f);
}

DSPARK_TEST(DryWetMixer_mix1_returns_wet)
{
    DryWetMixer<float> mixer;
    mixer.prepare(defaultSpec());

    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, 44100.0f);

    mixer.pushDry(tb.view());

    // Fill with DC as "wet" signal
    generateDC(tb.ch(0), 256, 0.42f);
    generateDC(tb.ch(1), 256, 0.42f);

    // Mix at 1 = 100% wet
    mixer.mixWet(tb.view(), 1.0f);

    EXPECT_NEAR(tb.ch(0)[100], 0.42f, 1e-5f);
}

DSPARK_TEST(DryWetMixer_mix_laws_are_exact)
{
    // Static mix, dry = DC 1.0, wet = 0: output IS the dry gain of each law.
    DryWetMixer<float> linear, ep;
    linear.prepare(defaultSpec());
    ep.prepare(defaultSpec());
    ep.setMixRule(DryWetMixer<float>::MixRule::EqualPower);

    auto tb = makeMonoBuffer(256);

    generateDC(tb.ch(0), 256, 1.0f);
    linear.pushDry(tb.view());
    tb.view().clear();
    linear.mixWet(tb.view(), 0.25f);            // first call jumps (no ramp)
    EXPECT_NEAR(tb.ch(0)[200], 0.75f, 1e-6f);   // d = 1 - m

    generateDC(tb.ch(0), 256, 1.0f);
    ep.pushDry(tb.view());
    tb.view().clear();
    ep.mixWet(tb.view(), 0.25f);
    EXPECT_NEAR(tb.ch(0)[200], 0.8660254f, 1e-6f); // d = sqrt(1 - m)
}

DSPARK_TEST(DryWetMixer_equalpower_ramp_is_nan_free_on_huge_blocks)
{
    // Regression: the ramped equal-power path takes square roots of values
    // that float rounding can push past the [0, 1] ends on long blocks;
    // sqrt(-epsilon) would fill the tail of the block with NaN.
    constexpr int N = 65536;
    DryWetMixer<float> mixer;
    mixer.prepare(spec(48000.0, N, 1));
    mixer.setMixRule(DryWetMixer<float>::MixRule::EqualPower);

    AudioBuffer<float> buf;
    buf.resize(1, N);

    // Settle the smoother at 0.1 (first mixWet jumps straight there).
    generateDC(buf.getChannel(0), N, 1.0f);
    mixer.pushDry(buf.toView());
    buf.toView().clear();
    mixer.mixWet(buf.toView(), 0.1f);
    EXPECT_NEAR(buf.getChannel(0)[N - 1], std::sqrt(0.9f), 1e-6f);

    // Ramp 0.1 -> 1.0 across one huge block. Output = sqrt(1 - m_i): must be
    // NaN-free, continuous at the seam and monotonically fading out.
    generateDC(buf.getChannel(0), N, 1.0f);
    mixer.pushDry(buf.toView());
    buf.toView().clear();
    mixer.mixWet(buf.toView(), 1.0f);

    const float* out = buf.getChannel(0);
    EXPECT_NO_NAN(out, N);
    EXPECT_NEAR(out[0], std::sqrt(0.9f), 1e-4f);  // starts at the old mix
    for (int i = 1; i < N; ++i)
        if (!(out[i] <= out[i - 1] + 1e-6f)) { EXPECT_TRUE(false); break; }

    // And the settled block after the ramp is exactly fully wet.
    generateDC(buf.getChannel(0), N, 1.0f);
    mixer.pushDry(buf.toView());
    buf.toView().clear();
    mixer.mixWet(buf.toView(), 1.0f);
    EXPECT_NEAR(buf.getChannel(0)[100], 0.0f, 1e-7f);
}

DSPARK_TEST(DryWetMixer_latency_compensation_time_aligns_dry)
{
    // The compensated dry must be the input delayed by exactly D samples,
    // across pushDry() calls of irregular sizes (circular buffer wraps).
    constexpr int D = 17;
    DryWetMixer<float> mixer;
    mixer.prepare(spec(48000.0, 64, 1));
    mixer.setLatencyCompensation(D);
    EXPECT_EQ(mixer.getLatencyCompensation(), D);

    AudioBuffer<float> buf;
    buf.resize(1, 64);

    const int blockSizes[] = { 7, 32, 19 };  // total 58 > 3*D: several wraps
    int globalIndex = 0;
    for (int n : blockSizes)
    {
        for (int i = 0; i < n; ++i)  // x[g] = g + 1 (0 marks pre-history)
            buf.getChannel(0)[i] = static_cast<float>(globalIndex + i + 1);
        mixer.pushDry(buf.toView().getSubView(0, n));

        const float* dry = mixer.getDryChannel(0);
        for (int i = 0; i < n; ++i)
        {
            const int g = globalIndex + i;
            const float expected = (g < D) ? 0.0f : static_cast<float>(g - D + 1);
            EXPECT_NEAR(dry[i], expected, 1e-9f);
        }
        globalIndex += n;
    }
}

DSPARK_TEST(DryWetMixer_mixWet_without_snapshot_is_passthrough)
{
    // After reset() there is no valid dry snapshot: mixWet() must leave the
    // wet buffer untouched instead of blending against stale/cleared data.
    DryWetMixer<float> mixer;
    mixer.prepare(defaultSpec());
    mixer.reset();

    auto tb = makeMonoBuffer(256);
    generateDC(tb.ch(0), 256, 0.42f);
    mixer.mixWet(tb.view(), 0.0f);  // mix 0 would zero the buffer if it ran
    EXPECT_NEAR(tb.ch(0)[100], 0.42f, 1e-9f);
}

DSPARK_TEST(DryWetMixer_move_preserves_snapshot_and_config)
{
    // Setup-time relocation: the moved-to mixer keeps the captured dry,
    // the latency setting and the smoothing state.
    DryWetMixer<float> src;
    src.prepare(defaultSpec());

    auto tb = makeMonoBuffer(256);
    generateDC(tb.ch(0), 256, 0.5f);
    src.pushDry(tb.view());

    DryWetMixer<float> dst(std::move(src));
    EXPECT_EQ(dst.getDryCapturedSamples(), 256);

    tb.view().clear();
    dst.mixWet(tb.view(), 0.0f);  // fully dry: recovers the captured signal
    EXPECT_NEAR(tb.ch(0)[100], 0.5f, 1e-6f);
}

// ============================================================================
// ProcessorChain
// ============================================================================

DSPARK_TEST(ProcessorChain_two_gains_multiply)
{
    // Chain two Gain processors - output should be product of both gains
    ProcessorChain<float, Gain<float>, Gain<float>> chain;
    chain.prepare(defaultSpec());
    chain.get<0>().setGainLinear(0.5f);
    chain.get<0>().skipRamp();
    chain.get<1>().setGainLinear(0.5f);
    chain.get<1>().skipRamp();

    auto tb = makeMonoBuffer(256);
    generateDC(tb.ch(0), 256, 1.0f);

    chain.processBlock(tb.view());

    // 1.0 * 0.5 * 0.5 = 0.25
    EXPECT_NEAR(tb.ch(0)[255], 0.25f, 0.02f);
}

DSPARK_TEST(ProcessorChain_reset)
{
    ProcessorChain<float, Gain<float>, Gain<float>> chain;
    chain.prepare(defaultSpec());
    chain.reset(); // Should not crash
    EXPECT_TRUE(true);
}


// ============================================================================
// ModulationRouter
// ============================================================================

DSPARK_TEST(ModulationRouter_routes_scale_and_smooth)
{
    ModulationRouter<float, 4> router;
    float sourceValue = 100.0f;
    float applied = -1.0f;
    const int id = router.addRoute([&] { return sourceValue; },
                                   [&](float v) { applied = v; },
                                   0.9f, 10.0f, 0.0f);   // no smoothing
    EXPECT_EQ(id, 0);

    router.update(512, 48000.0);
    EXPECT_NEAR(applied, 10.0f + 100.0f * 0.9f, 1e-4f);

    // Smoothing: a jump approaches the target over blocks, primed at start.
    ModulationRouter<float, 4> sm;
    float v2 = 0.0f, out2 = 0.0f;
    sm.addRoute([&] { return v2; }, [&](float v) { out2 = v; }, 1.0f, 0.0f, 50.0f);
    v2 = 1.0f;
    sm.update(512, 48000.0);
    EXPECT_NEAR(out2, 1.0f, 1e-6f);    // first hit primes directly
    v2 = 0.0f;
    sm.update(512, 48000.0);
    EXPECT_GT(out2, 0.5f);              // 10.7 ms < 50 ms: still on the way
    for (int i = 0; i < 40; ++i)
        sm.update(512, 48000.0);
    EXPECT_LT(out2, 0.02f);             // converged

    EXPECT_EQ(router.getNumRoutes(), 1);
    router.clear();
    EXPECT_EQ(router.getNumRoutes(), 0);
}

DSPARK_TEST(ModulationRouter_contracts)
{
    ModulationRouter<float, 2> router;

    // Empty callables are rejected at the door (they would raise
    // bad_function_call inside the noexcept audio-thread update()).
    EXPECT_EQ(router.addRoute(nullptr, [](float) {}), -1);
    EXPECT_EQ(router.addRoute([] { return 0.0f; }, nullptr), -1);
    EXPECT_EQ(router.getNumRoutes(), 0);

    // Capacity: third route on a 2-route router fails without side effects.
    float sink = 0.0f;
    EXPECT_EQ(router.addRoute([] { return 1.0f; }, [&](float v) { sink = v; }), 0);
    EXPECT_EQ(router.addRoute([] { return 2.0f; }, [&](float v) { sink = v; }), 1);
    EXPECT_EQ(router.addRoute([] { return 3.0f; }, [&](float v) { sink = v; }), -1);
    EXPECT_EQ(router.getNumRoutes(), 2);

    // Live depth change is picked up by the next update (unsmoothed route).
    ModulationRouter<float, 2> hot;
    float out = 0.0f;
    const int id = hot.addRoute([] { return 10.0f; }, [&](float v) { out = v; },
                                1.0f, 0.0f, 0.0f);
    hot.update(256, 48000.0);
    EXPECT_NEAR(out, 10.0f, 1e-6f);
    hot.setDepth(id, 0.25f);
    hot.update(256, 48000.0);
    EXPECT_NEAR(out, 2.5f, 1e-6f);

    // clear() must release the stored captures, not just reset the count.
    auto token = std::make_shared<int>(42);
    ModulationRouter<float, 2> cap;
    cap.addRoute([token] { return static_cast<float>(*token); }, [](float) {});
    EXPECT_EQ(static_cast<int>(token.use_count()), 2);
    cap.clear();
    EXPECT_EQ(static_cast<int>(token.use_count()), 1);
}

// ===========================================================================
// M-003 AG-8 audit regression pins (additive; CHANGE-REQUEST recorded in
// builder-report.002.json). Guard the shipped fixes in the CI suite.
// ===========================================================================

// D-M003-3b: a buffer wider than the 16-channel view default must not silently
// drop channels through toView() (regression guard: toView now propagates
// MaxChannels as the view's channel capacity).
DSPARK_TEST(AudioBuffer_toView_preserves_channels_beyond_default_width)
{
    AudioBuffer<float, 32> wide;
    wide.resize(20, 16);
    EXPECT_EQ(wide.toView().getNumChannels(), 20);
    EXPECT_EQ(wide.toView().getNumSamples(), 16);
    // const overload preserves them too
    const AudioBuffer<float, 32>& cref = wide;
    EXPECT_EQ(cref.toView().getNumChannels(), 20);
}

// D-M003-5: a transient non-finite modulation source must not poison a route;
// the one-pole holds the last good value (matches DryWetMixer's guard).
DSPARK_TEST(ModulationRouter_holds_last_value_on_nonfinite_source)
{
    ModulationRouter<float, 2> router;
    float src = 0.3f; float out = -1.0f;
    router.addRoute([&] { return src; }, [&](float v) { out = v; }, 1.0f, 0.0f, 0.0f);
    router.update(64, 48000.0);
    const float good = out;
    EXPECT_TRUE(std::isfinite(good));

    src = std::numeric_limits<float>::quiet_NaN();
    router.update(64, 48000.0);
    EXPECT_TRUE(std::isfinite(out));
    EXPECT_EQ(out, good); // held, not poisoned
}
