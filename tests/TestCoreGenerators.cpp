// DSPark Tests -- Core Generators
// Oscillator, Phasor, WavetableOscillator, EnvelopeGenerator, AnalogRandom

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/Oscillator.h"
#include "../Core/Phasor.h"
#include "../Core/WavetableOscillator.h"
#include "../Core/EnvelopeGenerator.h"
#include "../Core/AnalogRandom.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <thread>
#include <vector>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// Phasor
// ============================================================================

DSPARK_TEST(Phasor_range_0_to_1)
{
    Phasor<float> p;
    p.prepare(44100.0);
    p.setFrequency(440.0f);

    for (int i = 0; i < 44100; ++i)
    {
        float phase = p.advance();
        EXPECT_TRUE(phase >= 0.0f);
        EXPECT_TRUE(phase < 1.0f);
    }
}

DSPARK_TEST(Phasor_frequency_accuracy)
{
    Phasor<float> p;
    p.prepare(44100.0);
    p.setFrequency(100.0f);

    // 100 Hz = 441 samples per cycle at 44100 Hz
    // Count zero-crossings (phase wraps)
    int wraps = 0;
    float prev = 0.0f;
    for (int i = 0; i < 44100; ++i)
    {
        float phase = p.advance();
        if (phase < prev) ++wraps; // Wrapped around
        prev = phase;
    }
    // Should have ~100 wraps in 1 second
    EXPECT_NEAR(wraps, 100, 1);
}

DSPARK_TEST(Phasor_hardSync)
{
    Phasor<float> p;
    p.prepare(44100.0);
    p.setFrequency(440.0f);

    for (int i = 0; i < 100; ++i)
        (void)p.advance();

    p.hardSync();
    EXPECT_NEAR(p.getPhase(), 0.0f, 1e-6f);
}

DSPARK_TEST(Phasor_reset)
{
    Phasor<float> p;
    p.prepare(44100.0);
    p.setFrequency(440.0f);

    for (int i = 0; i < 1000; ++i) (void)p.advance();
    p.reset(0.5f);
    EXPECT_NEAR(p.getPhase(), 0.5f, 1e-6f);
}

DSPARK_TEST(Phasor_slow_lfo_neither_stalls_nor_drifts)
{
    // Ultra-slow LFO rates must keep moving at the exact requested rate.
    // With a float accumulator, 0.002 Hz at 96 kHz freezes completely (the
    // per-sample increment rounds to zero against the phase's ulp) and
    // 0.02 Hz runs ~8% slow; the double accumulator is exact.
    auto measure = [](double freq, double fs, double seconds)
    {
        Phasor<float> p;
        p.prepare(fs);
        p.setFrequency(static_cast<float>(freq));
        p.reset(0.6f); // upper half of the cycle, where float ulp is largest

        const long long n = static_cast<long long>(seconds * fs);
        float first = p.getPhase(), last = first, prev = first;
        long long wraps = 0;
        for (long long i = 0; i < n; ++i)
        {
            last = p.advance();
            if (last < prev) ++wraps;
            prev = last;
        }
        const double moved = static_cast<double>(wraps)
                           + static_cast<double>(last) - static_cast<double>(first);
        const double expected = static_cast<double>(n - 1) * (freq / fs);
        return moved / expected;
    };

    EXPECT_NEAR(measure(0.002, 96000.0, 60.0), 1.0, 0.01); // frozen -> 0.0 before
    EXPECT_NEAR(measure(0.02, 96000.0, 60.0), 1.0, 0.01);  // ~0.92 before
}

DSPARK_TEST(Phasor_advanceWithFM_matches_advance_and_wraps)
{
    // Zero FM must be bit-identical to the unmodulated path.
    Phasor<float> a, b;
    a.prepare(48000.0);
    b.prepare(48000.0);
    a.setFrequency(440.0f);
    b.setFrequency(440.0f);
    for (int i = 0; i < 1000; ++i)
        EXPECT_TRUE(a.advanceWithFM(0.0f) == b.advance());

    // Through-zero FM produces a descending ramp.
    Phasor<float> tz;
    tz.prepare(48000.0);
    tz.setFrequency(100.0f);
    (void)tz.advance();
    float p0 = tz.getPhase();
    (void)tz.advanceWithFM(-300.0f); // net -200 Hz
    float p1 = tz.getPhase();
    EXPECT_TRUE(p1 < p0 || p1 > 0.9f); // moved backwards (or wrapped below 0)

    // Wild FM excursions (way beyond the sample rate, both signs) never
    // leave [0, 1) and never hang.
    Phasor<float> wild;
    wild.prepare(48000.0);
    wild.setFrequency(440.0f);
    for (int i = 0; i < 2000; ++i)
    {
        const float fm = ((i & 1) ? 1.0f : -1.0f) * 3.7f * 48000.0f
                       + ((i % 100 == 0) ? 1.0e9f : 0.0f);
        float ph = wild.advanceWithFM(fm);
        EXPECT_TRUE(ph >= 0.0f && ph < 1.0f);
    }
}

DSPARK_TEST(Phasor_nan_inputs_do_not_poison_the_phase)
{
    // setFrequency(NaN) is ignored: frequency and motion stay intact.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    Phasor<float> p, twin;
    p.prepare(48000.0);
    twin.prepare(48000.0);
    p.setFrequency(440.0f);
    twin.setFrequency(440.0f);
    p.setFrequency(nan);
    EXPECT_NEAR(p.getFrequency(), 440.0f, 1e-6f);
    for (int i = 0; i < 100; ++i)
        EXPECT_TRUE(p.advance() == twin.advance());

    // prepare with invalid rates (NaN / 0 / negative) is ignored too.
    p.prepare(std::numeric_limits<double>::quiet_NaN());
    p.prepare(0.0);
    p.prepare(-48000.0);
    for (int i = 0; i < 100; ++i)
        EXPECT_TRUE(p.advance() == twin.advance());

    // setPhase(NaN) resets to 0 instead of sticking.
    p.setPhase(nan);
    EXPECT_TRUE(p.getPhase() == 0.0f);

    // A single NaN FM excursion cannot poison the accumulator: the phase is
    // back in [0, 1) immediately and keeps advancing afterwards.
    Phasor<float> fm;
    fm.prepare(48000.0);
    fm.setFrequency(440.0f);
    (void)fm.advanceWithFM(nan);
    float ph = fm.getPhase();
    EXPECT_TRUE(ph >= 0.0f && ph < 1.0f);
    float before = fm.getPhase();
    (void)fm.advance();
    (void)fm.advance();
    EXPECT_TRUE(fm.getPhase() != before); // moving again
}

DSPARK_TEST(Phasor_sync_getters_and_range_contract)
{
    Phasor<float> p;
    p.prepare(48000.0);
    p.setFrequency(480.0f);

    // getIncrement reports freq / fs.
    EXPECT_NEAR(p.getIncrement(), 0.01f, 1e-9f);

    // softSync only fires at/above the threshold.
    p.setPhase(0.7f);
    p.softSync(0.5f);
    EXPECT_TRUE(p.getPhase() == 0.0f);
    p.setPhase(0.3f);
    p.softSync(0.5f);
    EXPECT_NEAR(p.getPhase(), 0.3f, 1e-7f);

    // hardSync and reset wrap their argument into [0, 1).
    p.hardSync(1.3f);
    EXPECT_NEAR(p.getPhase(), 0.3f, 1e-6f);
    p.reset(-0.25f);
    EXPECT_NEAR(p.getPhase(), 0.75f, 1e-6f);

    // Half-open contract at the wrap seam: crawl through the zone just below
    // 1.0 with a tiny increment; the T-rounded phase must never reach 1.0
    // (the double accumulator can round UP to exactly 1.0f without the guard).
    Phasor<float> seam;
    seam.prepare(96000.0);
    seam.setFrequency(1.04e-3f); // increment ~1.08e-8, well below float ulp at 1.0
    seam.setPhase(0.99999994f);  // largest float below 1.0
    for (int i = 0; i < 20; ++i)
    {
        float ph = seam.advance();
        EXPECT_TRUE(ph >= 0.0f && ph < 1.0f);
    }
}

// ============================================================================
// Oscillator
// ============================================================================

DSPARK_TEST(Oscillator_Sine_frequency)
{
    Oscillator<float> osc;
    osc.prepare(44100.0);
    osc.setFrequency(440.0f);
    osc.setWaveform(Oscillator<float>::Waveform::Sine);

    // Generate 1 second
    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = osc.getNextSample();

    // Measure fundamental using DFT
    float mag440 = measureFrequencyMagnitude(buf.data(), N, 440.0f, 44100.0f);
    float mag880 = measureFrequencyMagnitude(buf.data(), N, 880.0f, 44100.0f);

    // Fundamental should be strong, second harmonic should be negligible for sine
    EXPECT_GT(mag440, 0.9f);
    EXPECT_LT(mag880, 0.01f);
}

DSPARK_TEST(Oscillator_Sine_amplitude)
{
    Oscillator<float> osc;
    osc.prepare(44100.0);
    osc.setFrequency(440.0f);
    osc.setWaveform(Oscillator<float>::Waveform::Sine);

    float peak = 0.0f;
    for (int i = 0; i < 44100; ++i)
    {
        float s = osc.getNextSample();
        float a = std::abs(s);
        if (a > peak) peak = a;
    }
    EXPECT_NEAR(peak, 1.0f, 0.01f);
}

DSPARK_TEST(Oscillator_Saw_fundamental)
{
    Oscillator<float> osc;
    osc.prepare(44100.0);
    osc.setFrequency(440.0f);
    osc.setWaveform(Oscillator<float>::Waveform::Saw);

    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = osc.getNextSample();

    float mag440 = measureFrequencyMagnitude(buf.data(), N, 440.0f, 44100.0f);
    EXPECT_GT(mag440, 0.5f);
}

DSPARK_TEST(Oscillator_Square_fundamental)
{
    Oscillator<float> osc;
    osc.prepare(44100.0);
    osc.setFrequency(440.0f);
    osc.setWaveform(Oscillator<float>::Waveform::Square);

    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = osc.getNextSample();

    float mag440 = measureFrequencyMagnitude(buf.data(), N, 440.0f, 44100.0f);
    EXPECT_GT(mag440, 0.5f);
}

DSPARK_TEST(Oscillator_no_NaN)
{
    Oscillator<float> osc;
    osc.prepare(44100.0);
    osc.setFrequency(440.0f);

    for (auto wf : { Oscillator<float>::Waveform::Sine,
                     Oscillator<float>::Waveform::Saw,
                     Oscillator<float>::Waveform::Square,
                     Oscillator<float>::Waveform::Triangle })
    {
        osc.setWaveform(wf);
        osc.reset();
        for (int i = 0; i < 4096; ++i)
        {
            float s = osc.getNextSample();
            EXPECT_FALSE(std::isnan(s));
            EXPECT_FALSE(std::isinf(s));
        }
    }
}

DSPARK_TEST(Oscillator_double_template)
{
    Oscillator<double> osc;
    osc.prepare(48000.0);
    osc.setFrequency(1000.0);
    osc.setWaveform(Oscillator<double>::Waveform::Sine);

    double s = osc.getNextSample();
    EXPECT_FALSE(std::isnan(s));
}

// ============================================================================
// WavetableOscillator
// ============================================================================

DSPARK_TEST(WavetableOsc_Sine_fundamental)
{
    WavetableOscillator<float> wt;
    wt.prepare(44100.0);
    wt.buildSine();
    wt.setFrequency(440.0f);

    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = wt.getSample();

    float mag440 = measureFrequencyMagnitude(buf.data(), N, 440.0f, 44100.0f);
    EXPECT_GT(mag440, 0.85f);
}

DSPARK_TEST(WavetableOsc_Saw_has_harmonics)
{
    WavetableOscillator<float> wt;
    wt.prepare(44100.0);
    wt.buildSaw();
    wt.setFrequency(440.0f);

    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = wt.getSample();

    float mag440 = measureFrequencyMagnitude(buf.data(), N, 440.0f, 44100.0f);
    float mag880 = measureFrequencyMagnitude(buf.data(), N, 880.0f, 44100.0f);

    // Saw has harmonics at multiples
    EXPECT_GT(mag440, 0.3f);
    EXPECT_GT(mag880, 0.1f);
}

DSPARK_TEST(WavetableOsc_reset)
{
    WavetableOscillator<float> wt;
    wt.prepare(44100.0);
    wt.buildSine();
    wt.setFrequency(440.0f);

    float s1 = wt.getSample();
    wt.reset();
    float s2 = wt.getSample();
    EXPECT_NEAR(s1, s2, 1e-5f);
}

DSPARK_TEST(WavetableOsc_rate_change_after_build_keeps_antialiasing)
{
    // Build the tables at 96 kHz, then prepare at 44.1 kHz. prepare() must
    // re-derive the mip cutoffs; with the old fixed cutoffs the oscillator
    // picked levels designed for the higher rate and a 5 kHz saw aliased
    // loudly (harmonic 5 at 25 kHz folds to 19.1 kHz).
    WavetableOscillator<float> wt;
    wt.prepare(96000.0);
    wt.buildSaw();
    wt.prepare(44100.0);
    wt.setFrequency(5000.0f);

    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = wt.getSample();

    float fund = measureFrequencyMagnitude(buf.data(), N, 5000.0f, 44100.0f);
    float alias = measureFrequencyMagnitude(buf.data(), N, 19100.0f, 44100.0f);
    EXPECT_GT(fund, 0.2f);
    EXPECT_LT(alias / fund, 0.03f); // < -30 dB (old header: ~-15 dB, fails)
}

DSPARK_TEST(WavetableOsc_build_before_prepare_works)
{
    // Table content is rate-independent; building before prepare() and then
    // preparing at a different rate must produce the requested pitch cleanly.
    WavetableOscillator<float> wt;
    wt.buildSaw(); // built with the default 48 kHz state
    wt.prepare(44100.0);
    wt.setFrequency(440.0f);

    constexpr int N = 44100;
    std::vector<float> buf(N);
    for (int i = 0; i < N; ++i)
        buf[i] = wt.getSample();

    float mag440 = measureFrequencyMagnitude(buf.data(), N, 440.0f, 44100.0f);
    EXPECT_GT(mag440, 0.3f);
}

DSPARK_TEST(WavetableOsc_loadWavetable_reconstructs_harmonics)
{
    // A custom cycle with known harmonic ratios (1st = 1.0, 3rd = 0.5) must
    // reproduce those ratios; DC content is discarded by design.
    constexpr int cycleN = 512;
    std::vector<float> cycle(cycleN);
    for (int i = 0; i < cycleN; ++i)
    {
        const double ph = 2.0 * 3.14159265358979323846 * i / cycleN;
        cycle[static_cast<size_t>(i)] =
            static_cast<float>(0.5 + std::sin(ph) + 0.5 * std::sin(3.0 * ph));
    }

    WavetableOscillator<float> wt;
    wt.prepare(48000.0);
    wt.loadWavetable(cycle.data(), cycleN);
    wt.setFrequency(200.0f);

    constexpr int N = 48000;
    std::vector<float> buf(N);
    double mean = 0.0;
    for (int i = 0; i < N; ++i)
    {
        buf[static_cast<size_t>(i)] = wt.getSample();
        mean += buf[static_cast<size_t>(i)];
    }
    mean /= N;

    float m1 = measureFrequencyMagnitude(buf.data(), N, 200.0f, 48000.0f);
    float m3 = measureFrequencyMagnitude(buf.data(), N, 600.0f, 48000.0f);
    EXPECT_GT(m1, 0.1f);
    EXPECT_NEAR(m1 / m3, 2.0f, 0.15f);      // 1.0 : 0.5 harmonic ratio kept
    EXPECT_NEAR(static_cast<float>(mean), 0.0f, 0.01f); // DC discarded

    // Degenerate sizes are safe no-ops.
    wt.loadWavetable(cycle.data(), 2);
    wt.loadWavetable(nullptr, 100);
    EXPECT_TRUE(std::isfinite(wt.getSample()));
}

DSPARK_TEST(WavetableOsc_generateBlock_fills_all_channels)
{
    WavetableOscillator<float> wt;
    wt.prepare(48000.0);
    wt.buildSine();
    wt.setFrequency(1000.0f);

    AudioBuffer<float> buf;
    buf.resize(2, 512);
    wt.generateBlock(buf.toView());

    const float* l = buf.toView().getChannel(0);
    const float* r = buf.toView().getChannel(1);
    double rms = 0.0;
    for (int i = 0; i < 512; ++i)
    {
        EXPECT_TRUE(l[i] == r[i]); // mono source, all channels identical
        rms += static_cast<double>(l[i]) * static_cast<double>(l[i]);
    }
    EXPECT_GT(std::sqrt(rms / 512.0), 0.3); // not silence
}

// ============================================================================
// EnvelopeGenerator (ADSR)
// ============================================================================

DSPARK_TEST(ADSR_idle_state)
{
    ADSREnvelope<float> env;
    env.prepare(44100.0);
    env.setAttack(10.0f);
    env.setDecay(50.0f);
    env.setSustain(0.7f);
    env.setRelease(100.0f);

    EXPECT_NEAR(env.getNextValue(), 0.0f, 1e-6f);
    EXPECT_TRUE(env.getState() == ADSREnvelope<float>::State::Idle);
}

DSPARK_TEST(ADSR_attack_rises)
{
    ADSREnvelope<float> env;
    env.prepare(44100.0);
    env.setAttack(10.0f);
    env.setDecay(50.0f);
    env.setSustain(0.7f);
    env.setRelease(100.0f);

    env.noteOn();

    // After attack phase, should reach near 1.0
    float maxVal = 0.0f;
    int attackSamples = static_cast<int>(44100.0 * 0.010); // 10 ms
    for (int i = 0; i < attackSamples + 100; ++i)
    {
        float v = env.getNextValue();
        if (v > maxVal) maxVal = v;
    }
    EXPECT_GT(maxVal, 0.95f);
}

DSPARK_TEST(ADSR_sustain_holds)
{
    ADSREnvelope<float> env;
    env.prepare(44100.0);
    env.setAttack(5.0f);
    env.setDecay(20.0f);
    env.setSustain(0.5f);
    env.setRelease(100.0f);

    env.noteOn();

    // Run through attack + decay
    for (int i = 0; i < 4410; ++i) // ~100ms
        (void)env.getNextValue();

    // Should be near sustain level
    float v = env.getNextValue();
    EXPECT_NEAR(v, 0.5f, 0.05f);
}

DSPARK_TEST(ADSR_release_to_zero)
{
    ADSREnvelope<float> env;
    env.prepare(44100.0);
    env.setAttack(5.0f);
    env.setDecay(20.0f);
    env.setSustain(0.5f);
    env.setRelease(50.0f);

    env.noteOn();
    for (int i = 0; i < 4410; ++i) (void)env.getNextValue();

    env.noteOff();
    for (int i = 0; i < 4410; ++i) (void)env.getNextValue();

    float v = env.getNextValue();
    EXPECT_LT(v, 0.01f);
}

DSPARK_TEST(ADSR_reset)
{
    ADSREnvelope<float> env;
    env.prepare(44100.0);
    env.setAttack(10.0f);
    env.setDecay(50.0f);
    env.setSustain(0.7f);
    env.setRelease(100.0f);

    env.noteOn();
    for (int i = 0; i < 1000; ++i) (void)env.getNextValue();

    env.reset();
    EXPECT_TRUE(env.getState() == ADSREnvelope<float>::State::Idle);
    EXPECT_NEAR(env.getNextValue(), 0.0f, 1e-6f);
}

namespace {

// Samples until the envelope leaves `from` (cap = safety limit).
template <typename Env>
long long adsrSamplesUntilLeave(Env& env, typename Env::State from, long long cap)
{
    long long n = 0;
    while (env.getState() == from && n < cap)
    {
        (void)env.getNextValue();
        ++n;
    }
    return n;
}

} // namespace

// The stage parameters are hard time contracts: attack = 0 -> 1 in attackMs,
// decay = 1 -> sustain in decayMs (any sustain), release = full scale ->
// silence in releaseMs. The asymptotic targets must be the exact-completion
// overshoots: a first-order approximation runs ~7x long at curvature 0.1 and
// makes decay duration drift with the sustain level.
DSPARK_TEST(ADSR_stage_timings_are_exact)
{
    const double fs = 48000.0;
    for (float curve : { 0.5f, 3.0f })
    {
        for (float sus : { 0.3f, 0.7f })
        {
            const float aMs = 50.0f, dMs = 80.0f, rMs = 120.0f;
            const double aN = fs * 0.001 * aMs;
            const double dN = fs * 0.001 * dMs;
            const double rN = fs * 0.001 * rMs;

            ADSREnvelope<float> env;
            env.prepare(fs);
            env.setParameters(aMs, dMs, sus, rMs);
            env.setCurvature(curve);

            env.noteOn();
            const auto na = adsrSamplesUntilLeave(env, ADSREnvelope<float>::State::Attack,
                                                  static_cast<long long>(20 * aN));
            const auto nd = adsrSamplesUntilLeave(env, ADSREnvelope<float>::State::Decay,
                                                  static_cast<long long>(20 * dN));
            EXPECT_NEAR(na / aN, 1.0, 0.02);
            EXPECT_NEAR(nd / dN, 1.0, 0.02);

            // Release calibrated from full scale: retrigger to 1.0 first.
            env.noteOn();
            (void)adsrSamplesUntilLeave(env, ADSREnvelope<float>::State::Attack,
                                        static_cast<long long>(20 * aN));
            env.noteOff();
            const auto nr = adsrSamplesUntilLeave(env, ADSREnvelope<float>::State::Release,
                                                  static_cast<long long>(20 * rN));
            EXPECT_NEAR(nr / rN, 1.0, 0.02);
        }
    }
}

// Recursive state must be double regardless of T: in float, a 60 s attack at
// 96 kHz freezes short of 1.0 when the per-sample increment rounds to zero
// (the same stall class fixed in Smoothers).
DSPARK_TEST(ADSR_long_attack_completes_in_float)
{
    ADSREnvelope<float> env;
    env.prepare(96000.0);
    env.setParameters(60000.0f, 100.0f, 0.7f, 200.0f);
    env.noteOn();

    const double expected = 96000.0 * 60.0;
    const auto n = adsrSamplesUntilLeave(env, ADSREnvelope<float>::State::Attack,
                                         static_cast<long long>(1.1 * expected));
    EXPECT_NEAR(n / expected, 1.0, 0.02);
    EXPECT_TRUE(env.getState() == ADSREnvelope<float>::State::Decay);
}

// processBlock is the same arithmetic as getNextValue, sample-exact, with
// block boundaries landing mid-stage and across every state transition.
DSPARK_TEST(ADSR_processBlock_matches_getNextValue)
{
    ADSREnvelope<float> a, b;
    for (auto* e : { &a, &b })
    {
        e->prepare(48000.0);
        e->setParameters(3.0f, 10.0f, 0.55f, 25.0f);
        e->setCurvature(0.8f, 3.0f, 5.0f);
        e->noteOn();
    }

    float block[301];
    float maxDiff = 0.0f;
    bool released = false;
    for (int chunk = 0; chunk < 30; ++chunk)
    {
        if (chunk == 15 && !released) // mid-run noteOff on both
        {
            a.noteOff();
            b.noteOff();
            released = true;
        }
        a.processBlock(block, 301);
        for (int i = 0; i < 301; ++i)
            maxDiff = std::max(maxDiff, std::abs(block[i] - b.getNextValue()));
    }
    EXPECT_TRUE(maxDiff == 0.0f);
    EXPECT_TRUE(a.getState() == b.getState());
}

// Output stays in [0, 1] for every configuration (including sustain 0, where
// the decay target is negative), and a retrigger from mid-release continues
// from the current value (legato, rate-bounded steps, no jump).
DSPARK_TEST(ADSR_bounds_and_legato_retrigger)
{
    for (float curve : { 0.1f, 3.0f, 10.0f })
    {
        for (float sus : { 0.0f, 0.5f, 1.0f })
        {
            ADSREnvelope<float> env;
            env.prepare(48000.0);
            env.setParameters(2.0f, 5.0f, sus, 8.0f);
            env.setCurvature(curve);
            env.noteOn();
            for (int i = 0; i < 2000; ++i)
            {
                const float v = env.getNextValue();
                EXPECT_TRUE(v >= 0.0f && v <= 1.0f);
            }
            env.noteOff();
            for (int i = 0; i < 4000; ++i)
            {
                const float v = env.getNextValue();
                EXPECT_TRUE(v >= 0.0f && v <= 1.0f);
            }
        }
    }

    // Legato retrigger: release half-way, then noteOn must resume from the
    // current level with rate-bounded steps (no discontinuity toward 0 or 1).
    ADSREnvelope<float> env;
    env.prepare(48000.0);
    env.setParameters(10.0f, 30.0f, 0.6f, 100.0f);
    env.noteOn();
    for (int i = 0; i < 4000; ++i) (void)env.getNextValue();
    env.noteOff();
    for (int i = 0; i < 2000; ++i) (void)env.getNextValue();

    float prev = env.getCurrentValue();
    EXPECT_GT(prev, 0.05f); // still audibly releasing
    env.noteOn();
    float maxStep = 0.0f;
    for (int i = 0; i < 2000; ++i)
    {
        const float v = env.getNextValue();
        maxStep = std::max(maxStep, std::abs(v - prev));
        prev = v;
    }
    EXPECT_LT(maxStep, 0.01f);
}

// A default-constructed envelope (no prepare) must be functional at the
// documented 48 kHz default -- the rates are computed in the constructor.
DSPARK_TEST(ADSR_default_constructed_is_functional)
{
    ADSREnvelope<float> env; // no prepare(), no setters: pure defaults
    env.noteOn();

    const double aN = 48000.0 * 0.001 * 10.0; // default attack = 10 ms
    const auto na = adsrSamplesUntilLeave(env, ADSREnvelope<float>::State::Attack,
                                          static_cast<long long>(20 * aN));
    EXPECT_NEAR(na / aN, 1.0, 0.02);
    EXPECT_NEAR(env.getCurrentValue(), 1.0f, 1e-6f);
}

DSPARK_TEST(ADSR_no_NaN)
{
    ADSREnvelope<float> env;
    env.prepare(44100.0);
    env.setAttack(1.0f);
    env.setDecay(10.0f);
    env.setSustain(0.5f);
    env.setRelease(50.0f);

    env.noteOn();
    for (int i = 0; i < 10000; ++i)
    {
        float v = env.getNextValue();
        EXPECT_FALSE(std::isnan(v));
        EXPECT_FALSE(std::isinf(v));
    }
    env.noteOff();
    for (int i = 0; i < 10000; ++i)
    {
        float v = env.getNextValue();
        EXPECT_FALSE(std::isnan(v));
    }
}

// ============================================================================
// AnalogRandom
// ============================================================================

DSPARK_TEST(AnalogRandom_seeded_stream_is_deterministic_and_bounded)
{
    using dspark::AnalogRandom::Generator;
    using dspark::AnalogRandom::NoiseType;

    Generator<float> a(42), b(42);
    for (Generator<float>* g : { &a, &b })
    {
        g->prepare(48000.0);
        g->setNoiseType(NoiseType::White);
        g->setRateHz(1000.0f);
        g->setRange(-0.25f, 0.25f);
    }

    int mismatches = 0, outOfRange = 0;
    for (int i = 0; i < 4800; ++i)
    {
        const float va = a.getNextSample();
        const float vb = b.getNextSample();
        if (va != vb) ++mismatches;
        if (va < -0.2501f || va > 0.2501f) ++outOfRange;
    }
    EXPECT_EQ(mismatches, 0);
    EXPECT_EQ(outOfRange, 0);
}

DSPARK_TEST(AnalogRandom_sample_hold_rate_and_quantization)
{
    using dspark::AnalogRandom::Generator;
    using dspark::AnalogRandom::NoiseType;

    Generator<float> g(7);
    g.prepare(48000.0);
    g.setNoiseType(NoiseType::White);
    g.setRateHz(100.0f);        // new target every 480 samples
    g.setRange(0.5f, 1.0f);     // strictly positive: keeps the denormal
    g.setQuantization(0.1f);    // flip below float resolution of the value

    // Without smoothing the output is piecewise constant on the 0.1 grid,
    // stepping at the S&H rate.
    float prev = g.getNextSample();
    int steps = 0, offGrid = 0;
    for (int i = 1; i < 48000; ++i)
    {
        const float v = g.getNextSample();
        if (v != prev) ++steps;
        const float snapped = std::round(v / 0.1f) * 0.1f;
        if (std::abs(v - snapped) > 1e-5f) ++offGrid;
        prev = v;
    }
    EXPECT_GT(steps, 60);   // ~100 retargets, minus same-level repeats
    EXPECT_LT(steps, 110);
    EXPECT_EQ(offGrid, 0);
}

DSPARK_TEST(AnalogRandom_smoothing_bounds_slew_and_zero_time_is_instant)
{
    using dspark::AnalogRandom::Generator;
    using dspark::AnalogRandom::NoiseType;

    Generator<float> g(3);
    g.prepare(48000.0);
    g.setNoiseType(NoiseType::White);
    g.setRateHz(2.0f);
    g.setRange(-1.0f, 1.0f);
    g.setSmoothing(true, 50.0f);

    float prev = g.getNextSample();
    float maxDelta = 0.0f;
    for (int i = 1; i < 48000; ++i)
    {
        const float v = g.getNextSample();
        maxDelta = std::max(maxDelta, std::abs(v - prev));
        prev = v;
    }
    // One-pole with tau 50 ms: per-sample move <= range * (1 - exp(-1/(fs*tau)))
    EXPECT_LT(maxDelta, 2.0f * 4.2e-4f + 1e-6f);

    // Regression: zero smoothing time means instantaneous. It used to keep
    // the previous coefficient (initially 0), silently freezing the output
    // short of every new target while smoothing was enabled.
    Generator<float> z(9);
    z.prepare(48000.0);
    z.setNoiseType(NoiseType::White);
    z.setRateHz(4800.0f);        // retarget every 10 samples
    z.setRange(0.5f, 1.0f);
    z.setSmoothing(true, 0.0f);
    float p = z.getNextSample();
    int moves = 0;
    for (int i = 1; i < 1000; ++i)
    {
        const float v = z.getNextSample();
        if (std::abs(v - p) > 1e-3f) ++moves;
        p = v;
    }
    EXPECT_GT(moves, 50);   // frozen output would report 0
}

DSPARK_TEST(AnalogRandom_discrete_mapping_stays_in_bounds)
{
    using dspark::AnalogRandom::Generator;
    using dspark::AnalogRandom::NoiseType;

    Generator<float> g(11);
    g.prepare(48000.0);
    g.setNoiseType(NoiseType::White);
    g.setRateHz(4800.0f);
    g.setRange(-1.0f, 1.0f);

    int histo[5] = {};
    int outOfBounds = 0;
    for (int i = 0; i < 5000; ++i)
    {
        const int v = g.getNextDiscreteInt(2, 6);
        if (v < 2 || v > 6) { ++outOfBounds; continue; }
        ++histo[v - 2];
    }
    EXPECT_EQ(outOfBounds, 0);
    for (int k = 0; k < 5; ++k)
        EXPECT_GT(histo[k], 100);   // every bucket reachable
}

DSPARK_TEST(AnalogRandom_reseed_is_deterministic_across_move)
{
    using dspark::AnalogRandom::Generator;
    using dspark::AnalogRandom::NoiseType;

    // Two generators with different histories, reseeded to the same value,
    // must produce identical streams; the pending seed and every setting
    // must survive a move (NoiseGenerator stores these in std::vector).
    Generator<float> g(5), h(999);
    for (Generator<float>* p : { &g, &h })
    {
        p->prepare(48000.0);
        p->setNoiseType(NoiseType::White);
        p->setRateHz(48000.0f);  // new value every sample
        p->setRange(-1.0f, 1.0f);
        (void)p->getNextSample();  // diverge the histories
    }

    g.reseed(1234);
    h.reseed(1234);
    Generator<float> moved(std::move(h));  // pending seed rides along

    int mismatches = 0;
    for (int i = 0; i < 64; ++i)
        if (g.getNextSample() != moved.getNextSample()) ++mismatches;
    EXPECT_EQ(mismatches, 0);
}


// ============================================================================
// Oscillator hard sync (table minBLEP corrected)
// ============================================================================

namespace {

double syncAliasFloorDb(Oscillator<float>::Waveform wf, float ratio)
{
    Oscillator<float> osc;
    osc.prepare(48000.0);
    osc.setWaveform(wf);
    osc.setFrequency(440.0f);
    osc.setSyncRatio(ratio);
    osc.reset();

    constexpr size_t kN = 1 << 16;   // long window: keeps harmonic leakage
    std::vector<float> x(kN);        // out of the non-harmonic measurement
    for (size_t i = 0; i < kN; ++i)
        x[i] = osc.getNextSample();

    // Blackman-Harris analysis window: its -92 dB sidelobes (vs -57 dB of
    // Hann leakage at the tolerance edge) keep the metric floor below the
    // minBLEP's own alias floor, so we measure the kernel and not the window.
    std::vector<double> win(kN);
    WindowFunctions<double>::blackmanHarris(win.data(), (int)kN, false);

    FFTReal<double> fft(kN);
    std::vector<double> t(kN), f(kN + 2);
    for (size_t i = 0; i < kN; ++i)
        t[i] = static_cast<double>(x[i]) * win[i];
    fft.forward(t.data(), f.data());

    const double binHz = 48000.0 / static_cast<double>(kN);
    const auto maxBin = static_cast<size_t>(20000.0 / binHz);
    double fund = 0, worst = 0;
    for (size_t k = 16; k <= maxBin; ++k)
    {
        const double p = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
        const double harm = static_cast<double>(k) * binHz / 440.0;
        if (std::abs(harm - std::round(harm)) * 440.0 / binHz <= 10.0)
            fund = std::max(fund, p);
        else
            worst = std::max(worst, p);
    }
    return 10.0 * std::log10((worst + 1e-300) / (fund + 1e-300));
}

} // namespace

DSPARK_TEST(Oscillator_hard_sync_is_band_limited)
{
    using WF = Oscillator<float>::Waveform;
    // Table-minBLEP alias rejection (measured -98..-108 dB; the old 2-point
    // PolyBLEP kernel sat at -35..-44 dB on the same metric).
    EXPECT_LT(syncAliasFloorDb(WF::Saw, 1.5f), -90.0);
    EXPECT_LT(syncAliasFloorDb(WF::Saw, 2.7f), -90.0);
    EXPECT_LT(syncAliasFloorDb(WF::Square, 2.3f), -90.0);
    EXPECT_LT(syncAliasFloorDb(WF::Triangle, 3.1f), -90.0);
}

DSPARK_TEST(MinBlepTable_residual_shape)
{
    const auto& tab = MinBlepTable<float>::instance();

    // Starts at -1: right at the event the band-limited transition has
    // barely begun, so the correction must cancel ~the whole naive jump.
    EXPECT_NEAR(tab.residual(0.0f), -1.0f, 1e-2f);

    // Settles to exactly zero at the end (pinned during construction) and
    // beyond the table the step is considered settled.
    EXPECT_NEAR(tab.residual(63.999f), 0.0f, 1e-6f);
    EXPECT_EQ(tab.residual(64.0f), 0.0f);
    EXPECT_EQ(tab.residual(1000.0f), 0.0f);

    // Tail decay: the last quarter of the kernel is residual ripple only.
    float tailMax = 0.0f;
    for (float t = 48.0f; t < 64.0f; t += 1.0f / 64.0f)
        tailMax = std::max(tailMax, std::abs(tab.residual(t)));
    EXPECT_LT(tailMax, 1e-4f);
}

// The float and double tables must agree to float precision everywhere:
// generation always runs in double internally and only the final store is
// cast to T. A regression that let T leak into the cepstral build (float
// FFTs, float log/exp) would show up here long before it audibly degraded
// the sync alias floor.
DSPARK_TEST(MinBlepTable_float_and_double_tables_agree)
{
    const auto& tf = MinBlepTable<float>::instance();
    const auto& td = MinBlepTable<double>::instance();

    float maxDiff = 0.0f;
    for (int i = 0; i < MinBlepTable<float>::kTableSize; ++i)
    {
        const float t = static_cast<float>(i) / MinBlepTable<float>::kOversample;
        const auto d = static_cast<float>(td.residual(static_cast<double>(t)));
        maxDiff = std::max(maxDiff, std::abs(tf.residual(t) - d));
    }
    EXPECT_LT(maxDiff, 1e-6f);
}

DSPARK_TEST(Oscillator_sync_triangle_is_dc_free)
{
    // The synced square has inherent duty-cycle DC (the slave always restarts
    // in its +1 half). Uncompensated, the triangle integrator turned that
    // into a +0.45 offset at ratio 2.7; the analytic compensation in
    // updatePhaseInc must keep the triangle centred.
    Oscillator<float> osc;
    osc.prepare(48000.0);
    osc.setWaveform(Oscillator<float>::Waveform::Triangle);
    osc.setFrequency(440.0f);
    osc.setSyncRatio(2.7f);
    osc.reset();

    double dc = 0.0;
    bool finite = true;
    constexpr int kN = 96000;
    for (int i = 0; i < kN; ++i)
    {
        const float v = osc.getNextSample();
        if (!std::isfinite(v)) finite = false;
        dc += v;
    }
    EXPECT_TRUE(finite);
    EXPECT_LT(std::abs(dc / kN), 0.01);
}

DSPARK_TEST(Oscillator_sync_off_is_unchanged)
{
    // syncRatio <= 1 must leave the classic path bit-identical.
    Oscillator<float> a, b;
    a.prepare(48000.0); b.prepare(48000.0);
    a.setWaveform(Oscillator<float>::Waveform::Saw);
    b.setWaveform(Oscillator<float>::Waveform::Saw);
    a.setFrequency(440.0f); b.setFrequency(440.0f);
    b.setSyncRatio(0.5f);   // below 1: off
    a.reset(); b.reset();
    for (int i = 0; i < 4096; ++i)
        EXPECT_EQ(a.getNextSample(), b.getNextSample());
}

DSPARK_TEST(Oscillator_prepare_reclamps_frequency_after_rate_drop)
{
    // A frequency legal at the old rate can exceed Nyquist at a new, lower
    // one. prepare() must re-clamp it: with increment > 0.5 the two PolyBLEP
    // correction branches overlap and the output is garbage.
    Oscillator<float> osc;
    osc.prepare(96000.0);
    osc.setFrequency(40000.0f);          // legal at 96 kHz (Nyquist 48 kHz)
    osc.prepare(44100.0);                // rate drops: 40 kHz is now illegal
    EXPECT_LT(osc.getFrequency(), 22050.5f);

    osc.setWaveform(Oscillator<float>::Waveform::Saw);
    osc.reset();
    bool bounded = true;
    for (int i = 0; i < 4096; ++i)
    {
        const float v = osc.getNextSample();
        if (!std::isfinite(v) || std::abs(v) > 1.5f) bounded = false;
    }
    EXPECT_TRUE(bounded);
}

DSPARK_TEST(Oscillator_setPhase_seeds_triangle_integrator)
{
    // A triangle LFO must land on the waveform point implied by setPhase()
    // immediately -- Chorus re-phases its spread LFOs this way. Without
    // seeding, the integrator eases in from its stale state over ~1/inc
    // samples: a full second of wrong modulation depth at 1 Hz.
    Oscillator<float> osc;
    osc.prepare(48000.0);
    osc.setWaveform(Oscillator<float>::Waveform::Triangle);
    osc.setFrequency(1.0f);

    osc.setPhase(0.5f);                  // positive peak of the triangle
    EXPECT_GT(osc.getNextSample(), 0.9f);

    osc.setPhase(0.75f);                 // falling zero crossing
    EXPECT_LT(std::abs(osc.getNextSample()), 0.05f);

    osc.setPhase(0.0f);                  // negative peak (matches reset())
    EXPECT_LT(osc.getNextSample(), -0.9f);
}

DSPARK_TEST(Oscillator_triangle_free_run_amplitude)
{
    // The leaky-integrator triangle is normalised to +-1 by the analytic
    // steady-state peak; verify the free-running amplitude actually lands there.
    Oscillator<float> osc;
    osc.prepare(48000.0);
    osc.setWaveform(Oscillator<float>::Waveform::Triangle);
    osc.setFrequency(440.0f);
    osc.reset();

    const int settle = 2 * 48000 / 440;  // skip the first two cycles
    float peak = -1e9f, trough = 1e9f;
    for (int i = 0; i < 48000; ++i)
    {
        const float v = osc.getNextSample();
        if (i >= settle)
        {
            peak   = std::max(peak, v);
            trough = std::min(trough, v);
        }
    }
    EXPECT_NEAR(peak,    1.0f, 0.08f);
    EXPECT_NEAR(trough, -1.0f, 0.08f);
}

DSPARK_TEST(Oscillator_sync_reengage_is_clean)
{
    // Re-engaging sync must not replay stale ring corrections nor keep an
    // incoherent slave phase. After a sync -> off -> sync round trip the
    // stream must equal a freshly-phased oscillator with identical
    // parameters: both are pure functions of (master phase, increments)
    // once the slave is re-seeded and the ring is clean.
    Oscillator<float> a;
    a.prepare(48000.0);
    a.setWaveform(Oscillator<float>::Waveform::Saw);
    a.setFrequency(440.0f);
    a.setSyncRatio(2.7f);
    a.reset();
    for (int i = 0; i < 1000; ++i) (void) a.getNextSample();  // dirty the ring
    a.setSyncRatio(0.0f);                                     // sync off
    for (int i = 0; i < 37; ++i) (void) a.getNextSample();    // free-run a bit
    a.setSyncRatio(2.7f);                                     // re-engage

    Oscillator<float> b;
    b.prepare(48000.0);
    b.setWaveform(Oscillator<float>::Waveform::Saw);
    b.setFrequency(440.0f);
    b.setSyncRatio(2.7f);
    b.setPhase(a.getPhase());   // same master phase -> same slave state

    int mismatches = 0;
    for (int i = 0; i < 2048; ++i)
        if (a.getNextSample() != b.getNextSample()) ++mismatches;
    EXPECT_EQ(mismatches, 0);
}

// ===========================================================================
// Regression pins for the AnalogRandom and Oscillator fixes below. They
// exist so the CI suite goes red if any of those fixes is reverted: the
// probes that originally found the defects were throwaway programs, so the
// permanence lives here, in the suite.
// ===========================================================================

// Pink noise must measure ~-3 dB/oct across the audio band. The prior
// 3-of-7-pole filter produced -5 dB/oct; this pins the full Kellett filter.
DSPARK_TEST(AnalogRandom_pink_slope_is_minus3_dB_per_octave)
{
    using dspark::AnalogRandom::Generator;
    using dspark::AnalogRandom::NoiseType;

    const double fs = 48000.0; const int N = 8192; const int segs = 48;
    Generator<float> g(12345);
    g.prepare(fs);
    g.setNoiseType(NoiseType::Pink);
    g.setRateHz(static_cast<float>(fs)); // fresh sample every tick -> continuous noise
    g.setRange(-1.0f, 1.0f);
    g.setSmoothing(false);

    std::vector<double> psd(static_cast<size_t>(N / 2 + 1), 0.0);
    std::vector<float> win(static_cast<size_t>(N));
    WindowFunctions<float>::hann(win.data(), N, true);
    double wp = 0.0; for (int i = 0; i < N; ++i) wp += double(win[static_cast<size_t>(i)]) * win[static_cast<size_t>(i)];
    FFTReal<float> fft(static_cast<size_t>(N));
    std::vector<float> buf(static_cast<size_t>(N)), freq(static_cast<size_t>(N + 2)), mag(static_cast<size_t>(N / 2 + 1));
    for (int s = 0; s < segs; ++s)
    {
        for (int i = 0; i < N; ++i) buf[static_cast<size_t>(i)] = g.getNextSample() * win[static_cast<size_t>(i)];
        fft.forward(buf.data(), freq.data());
        fft.computeMagnitudes(freq.data(), mag.data());
        for (int k = 0; k <= N / 2; ++k) psd[static_cast<size_t>(k)] += double(mag[static_cast<size_t>(k)]) * double(mag[static_cast<size_t>(k)]);
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int n = 0;
    for (int k = 1; k <= N / 2; ++k)
    {
        const double f = k * fs / N; if (f < 100.0 || f > 8000.0) continue;
        const double x = std::log2(f);
        const double y = 10.0 * std::log10(psd[static_cast<size_t>(k)] / segs / wp + 1e-30);
        sx += x; sy += y; sxx += x * x; sxy += x * y; ++n;
    }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    EXPECT_GT(slope, -3.75);
    EXPECT_LT(slope, -2.25);
}

// setFrequency(NaN) must not poison the oscillator (std::clamp(NaN)
// returns NaN); output stays finite and a valid frequency afterwards recovers.
DSPARK_TEST(Oscillator_setFrequency_NaN_recovers_finite)
{
    Oscillator<float> osc; osc.prepare(48000.0);
    osc.setWaveform(Oscillator<float>::Waveform::Sine);
    osc.setFrequency(std::numeric_limits<float>::quiet_NaN());

    bool allFinite = true;
    for (int i = 0; i < 256; ++i) if (!std::isfinite(osc.getNextSample())) allFinite = false;
    EXPECT_TRUE(allFinite);

    osc.setFrequency(440.0f); // recovery
    double energy = 0.0;
    for (int i = 0; i < 512; ++i) { const float s = osc.getNextSample(); if (!std::isfinite(s)) allFinite = false; energy += double(s) * s; }
    EXPECT_TRUE(allFinite);
    EXPECT_GT(energy, 1.0);
}

// ============================================================================
// Cross-thread publication pins (concurrent, with a reachability argument)
// ============================================================================

// Pin for the AnalogRandom readout fix: getCurrentValue()/getPhase() must be
// lock-free atomic readouts, race-free while the audio thread generates and
// the control thread reseeds and moves parameters.
// Reachability argument: before the fix (commit 3cb669e) both readouts
// returned PLAIN members the generator writes on every sample -- this exact
// three-thread interleaving is a C++ data race that CI ThreadSanitizer
// reports at the readout locators (locally proven by the DRD red run:
// conflicting plain-word frames at AnalogRandom.h getNextSample/updatePhase
// vs getCurrentValue/getPhase, which is exactly this interleaving).
// The primary failure path is the oracle: CI ThreadSanitizer (and the
// registered DRD harness, proven red pre-fix) reports the pre-fix plain-word
// reads in this exact interleaving. The bounds assertions are a secondary
// net scoped to targets and variants where torn or stale readouts are
// actually producible (weakly-ordered hardware, broken publication
// variants); on the x86 hosts running this suite an aligned float load does
// not tear, so pre-fix they would essentially never fire locally. The
// liveness floors prove the threads actually overlapped rather than running
// back-to-back.
DSPARK_TEST(AnalogRandom_concurrent_readout_is_published_and_bounded)
{
    auto gen = std::make_unique<AnalogRandom::Generator<float>>(777u);
    gen->prepare(48000.0);
    gen->setNoiseType(AnalogRandom::NoiseType::Pink);
    gen->setRateHz(500.0f);
    gen->setSmoothing(true, 2.0f);
    gen->setRange(-1.0f, 1.0f);

    std::atomic<bool> stop{ false };
    std::atomic<long long> generated{ 0 };

    std::thread audio([&] {
        std::array<float, 64> block{};
        while (!stop.load(std::memory_order_relaxed))
        {
            for (int i = 0; i < 64; ++i) (void)gen->getNextSample();
            gen->getNextBlock(std::span<float>(block.data(), block.size()));
            generated.fetch_add(128, std::memory_order_relaxed);
        }
    });
    std::thread control([&] {
        int i = 0;
        while (!stop.load(std::memory_order_relaxed))
        {
            gen->setRange((i & 1) ? -0.5f : -1.0f, (i & 1) ? 0.5f : 1.0f);
            gen->setQuantization((i & 2) ? 0.05f : 0.0f);
            if ((i++ & 63) == 0) gen->reseed(static_cast<std::uint64_t>(i) + 5u);
        }
    });

    int violations = 0;
    int distinct = 0;
    float lastV = 0.0f;
    long long reads = 0;
    const long long kMaxReads = 50000000; // hard cap: never hang, fail loudly
    while ((generated.load(std::memory_order_relaxed) < 4096 || reads < 200000)
           && reads < kMaxReads)
    {
        ++reads;
        const float v = gen->getCurrentValue();
        const float p = gen->getPhase();
        // Published value stays inside the widest configured range (the
        // smoother converges within [min, max]; the denormal offset is
        // 1e-18); the published phase is a wrapped accumulator in [0, 1].
        if (!std::isfinite(v) || std::abs(v) > 1.001f) ++violations;
        if (!std::isfinite(p) || p < 0.0f || p > 1.0f) ++violations;
        if (v != lastV) { ++distinct; lastV = v; }
    }

    stop.store(true, std::memory_order_relaxed);
    audio.join();
    control.join();

    EXPECT_EQ(violations, 0);
    EXPECT_GT(distinct, 100);                                     // publication liveness
    EXPECT_GT(generated.load(std::memory_order_relaxed), 4095LL); // real overlap
}
