// DSPark Tests -- Synthesis
// Synthesizer and Lfo: pitch, timing, determinism, envelopes, voices,
// filters, modulation, FM against Bessel, the sample source against Sampler,
// oversampling latency, state.

#include "dspark_test.h"

#include "../Core/Synthesizer.h"
#include "../Core/Sampler.h"
#include "../Core/Lfo.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

using namespace dspark;
using namespace dspark::test;

namespace {

using Synth = Synthesizer<double>;
using P = Synth::Param;
constexpr double kFs = 48000.0;

/** Renders `total` samples in blocks, calling `events(synth, blockStart,
 *  blockLength)` before each block; returns interleaved L/R. */
std::vector<double> renderStereo(Synth& s, int total, int block,
                                 const std::function<void(Synth&, int, int)>& events)
{
    std::vector<double> out(static_cast<size_t>(2 * total));
    std::vector<double> l(static_cast<size_t>(block)), r(static_cast<size_t>(block));
    for (int pos = 0; pos < total; pos += block)
    {
        const int n = std::min(block, total - pos);
        events(s, pos, n);
        double* ch[2] = { l.data(), r.data() };
        s.processBlock(AudioBufferView<double>(ch, 2, n));
        for (int i = 0; i < n; ++i)
        {
            out[static_cast<size_t>(2 * (pos + i))] = l[static_cast<size_t>(i)];
            out[static_cast<size_t>(2 * (pos + i) + 1)] = r[static_cast<size_t>(i)];
        }
    }
    return out;
}

std::vector<double> left(const std::vector<double>& stereo)
{
    std::vector<double> l(stereo.size() / 2);
    for (size_t i = 0; i < l.size(); ++i) l[i] = stereo[2 * i];
    return l;
}

/** An event at absolute sample `at`, delivered in the block that holds it. */
void at(int at, int pos, int n, const std::function<void(int)>& fire)
{
    if (at >= pos && at < pos + n) fire(at - pos);
}

/** Frequency of a near-sinusoid by the phase advance of its Hann-windowed
 *  demodulation between two consecutive 16384-sample windows. */
double toneHz(const std::vector<double>& x, size_t start, double expected)
{
    constexpr size_t N = 16384;
    auto phaseAt = [&](size_t a) {
        double re = 0.0, im = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            const double w = 0.5 - 0.5 * std::cos(2.0 * pi<double> * static_cast<double>(i) / N);
            const double t = static_cast<double>(a + i);
            re += x[a + i] * w * std::cos(2.0 * pi<double> * expected * t / kFs);
            im -= x[a + i] * w * std::sin(2.0 * pi<double> * expected * t / kFs);
        }
        return std::atan2(im, re);
    };
    double d = phaseAt(start + N) - phaseAt(start);
    while (d > pi<double>) d -= 2.0 * pi<double>;
    while (d < -pi<double>) d += 2.0 * pi<double>;
    return expected + d * kFs / (2.0 * pi<double> * static_cast<double>(N));
}

double cents(double measured, double expected) { return 1200.0 * std::log2(measured / expected); }

double rms(const std::vector<double>& x, size_t from, size_t to)
{
    double s = 0.0;
    for (size_t i = from; i < to; ++i) s += x[i] * x[i];
    return std::sqrt(s / static_cast<double>(to - from));
}

/** Bessel function of the first kind by its power series (|x| small). */
double besselJ(int n, double x)
{
    double sum = 0.0, term = std::pow(0.5 * x, n);
    for (int k = 1; k <= n; ++k) term /= k;
    for (int k = 0; k < 40; ++k)
    {
        sum += term;
        term *= -(0.25 * x * x) / (static_cast<double>(k + 1) * static_cast<double>(k + 1 + n));
    }
    return sum;
}

/** A plain sine voice: one sine oscillator, filter off, instant envelope. */
void sineVoice(Synth& s)
{
    s.setParameter(P::Osc1Wave, 0.0f);
    s.setParameter(P::FilterType, 0.0f);
    s.setParameter(P::Env1Attack, 0.0f);
    s.setParameter(P::MasterVolume, 0.0f);
}

} // namespace

// ============================================================================
// Parameter table
// ============================================================================

DSPARK_TEST(Synth_parameter_table_is_complete_and_addressable)
{
    EXPECT_EQ(Synth::kNumParams, static_cast<int>(synthesizer::kParams.size()));
    for (int i = 0; i < Synth::kNumParams; ++i)
    {
        const auto& info = Synth::getParamInfo(static_cast<P>(i));
        EXPECT_EQ(Synth::findParam(info.id), i);
        if (info.labels != nullptr)
        {
            // A choice names each of its steps + 1 positions.
            for (int k = 0; k <= info.steps; ++k)
                EXPECT_TRUE(info.labels[k] != nullptr && info.labels[k][0] != '\0');
        }
    }
    EXPECT_EQ(Synth::findParam("no.such.param"), -1);

    Synth s;
    s.setParameter(P::FilterCutoff, 1.0e9f);
    EXPECT_EQ(s.getParameter(P::FilterCutoff), 20000.0f);          // clamped
    s.setParameter(P::Unison, 3.4f);
    EXPECT_EQ(s.getParameter(P::Unison), 3.0f);                     // rounded to its steps
    s.setParameter(P::Unison, std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(s.getParameter(P::Unison), 3.0f);                     // NaN ignored
    EXPECT_TRUE(s.setParameter("filter.resonance", 0.25f));
    EXPECT_EQ(s.getParameter(P::FilterResonance), 0.25f);
    EXPECT_FALSE(s.setParameter("filter.nope", 0.25f));
}

// ============================================================================
// Pitch
// ============================================================================

DSPARK_TEST(Synth_pitch_is_exact_on_every_engine_and_offset)
{
    // Sine on each engine: analog oscillator 1, FM carrier alone, wavetable
    // Basic frame 0. Expected 440 * 2^((key - 69) / 12) within 0.01 cent.
    double worst = 0.0;
    for (int engine = 0; engine < 3; ++engine)
        for (int key : { 33, 69, 93 })
        {
            Synth s;
            sineVoice(s);
            s.setParameter(P::Engine, static_cast<float>(engine));
            s.prepare({ kFs, 256, 2 }, 4);
            const auto x = left(renderStereo(s, 40000, 256, [&](Synth& sy, int pos, int n) {
                at(0, pos, n, [&](int o) { sy.noteOn(key, 1.0, o); });
            }));
            const double f = 440.0 * std::exp2((key - 69) / 12.0);
            worst = std::max(worst, std::abs(cents(toneHz(x, 4000, f), f)));
        }
    // Offsets: octave, semitone, fine, master tune and a full pitch bend.
    {
        Synth s;
        sineVoice(s);
        s.setParameter(P::Osc1Octave, -1.0f);
        s.setParameter(P::Osc1Semitone, 7.0f);
        s.setParameter(P::Osc1Fine, 25.0f);
        s.setParameter(P::MasterTune, -10.0f);
        s.setParameter(P::BendRange, 5.0f);
        s.prepare({ kFs, 256, 2 }, 4);
        const auto x = left(renderStereo(s, 40000, 256, [&](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(60, 1.0, o); sy.setPitchBend(-1.0, o); });
        }));
        const double f = 440.0 * std::exp2((60 - 69 - 12 + 7 + 0.25 - 0.10 - 5.0) / 12.0);
        worst = std::max(worst, std::abs(cents(toneHz(x, 4000, f), f)));
    }
    std::cout << "  worst pitch error " << worst << " cents\n";
    EXPECT_LT(worst, 0.01);
}

// ============================================================================
// Timing and determinism
// ============================================================================

DSPARK_TEST(Synth_note_starts_at_its_sample_offset)
{
    // A saw starts at its steady-state value (non-zero) with an instant
    // attack: nothing before the offset, sound from it on, in any block size.
    for (int block : { 64, 1000 })
    {
        Synth s;
        s.setParameter(P::Env1Attack, 0.0f);
        s.setParameter(P::FilterType, 0.0f);
        s.prepare({ kFs, 1024, 2 }, 4);
        const int onset = 777;
        const auto x = left(renderStereo(s, 2000, block, [&](Synth& sy, int pos, int n) {
            at(onset, pos, n, [&](int o) { sy.noteOn(57, 1.0, o); });
        }));
        double before = 0.0;
        for (int i = 0; i < onset; ++i) before = std::max(before, std::abs(x[static_cast<size_t>(i)]));
        EXPECT_EQ(before, 0.0);
        EXPECT_GT(std::abs(x[static_cast<size_t>(onset)]), 1e-3);
    }
}

DSPARK_TEST(Synth_output_is_identical_for_any_block_size)
{
    // Unison, drift, glide, sync, sub, noise, the filter envelope, free and
    // per-note LFOs, matrix routings, a bend, the wheel, overlapping notes
    // and a sample zone, at 1x and 2x: every value is a function of the
    // events and the seed, not of where blocks begin.
    std::vector<double> recording(30000);
    for (size_t i = 0; i < recording.size(); ++i)
        recording[i] = 0.4 * std::sin(2.0 * pi<double> * 300.0 * static_cast<double>(i) / 44100.0);
    const double* rec[1] = { recording.data() };
    for (int os : { 0, 1 })
    {
        auto setup = [&](Synth& s) {
            s.setParameter(P::Unison, 3.0f);
            s.setParameter(P::Drift, 0.7f);
            s.setParameter(P::Glide, 80.0f);
            s.setParameter(P::PhaseMode, 1.0f);
            s.setParameter(P::Osc1Wave, 3.0f);
            s.setParameter(P::Osc2Level, 0.6f);
            s.setParameter(P::Osc2Sync, 1.0f);
            s.setParameter(P::Osc2Semitone, 7.0f);
            s.setParameter(P::Osc3Level, 0.3f);
            s.setParameter(P::Osc3Wave, 1.0f);
            s.setParameter(P::SubLevel, 0.4f);
            s.setParameter(P::NoiseLevel, 0.05f);
            s.setParameter(P::SampleLevel, 0.5f);
            s.setParameter(P::FilterCutoff, 800.0f);
            s.setParameter(P::FilterResonance, 0.6f);
            s.setParameter(P::FilterEnvAmount, 3.0f);
            s.setParameter(P::Lfo1Rate, 5.5f);
            s.setParameter(P::Lfo2Shape, 6.0f);
            s.setParameter(P::Lfo2Mode, 0.0f);
            s.setParameter(P::Mod1Source, 9.0f);  s.setParameter(P::Mod1Destination, 1.0f);  s.setParameter(P::Mod1Amount, 0.3f);
            s.setParameter(P::Mod2Source, 10.0f); s.setParameter(P::Mod2Destination, 11.0f); s.setParameter(P::Mod2Amount, 0.3f);
            s.setParameter(P::Mod3Source, 13.0f); s.setParameter(P::Mod3Destination, 3.0f);  s.setParameter(P::Mod3Amount, 12.0f);
            s.setParameter(P::Mod4Source, 4.0f);  s.setParameter(P::Mod4Destination, 20.0f); s.setParameter(P::Mod4Amount, 2.0f);
            s.setParameter(P::Oversampling, static_cast<float>(os));
            s.prepare({ kFs, 512, 2 }, 6);
            s.loadSample(AudioBufferView<const double>(rec, 1, 30000), 44100.0, 62);
        };
        auto events = [](Synth& sy, int pos, int n) {
            static const int times[] = { 100, 3000, 9000, 9001, 20000, 33333, 41000, 52000 };
            static const int keys[] = { 48, 55, 60, 64, 67, 72, 48, 79 };
            for (int i = 0; i < 8; ++i)
            {
                at(times[i], pos, n, [&](int o) { sy.noteOn(keys[i], 0.5 + 0.05 * i, o); });
                at(times[i] + 15000, pos, n, [&](int o) { sy.noteOff(keys[i], 0.3, o); });
            }
            at(31000, pos, n, [&](int o) { sy.setPitchBend(0.5, o); });
            at(45000, pos, n, [&](int o) { sy.setModWheel(0.8, o); });
        };
        std::vector<double> reference;
        bool identical = true;
        for (int block : { 512, 1, 7, 64, 333 })
        {
            Synth s;
            setup(s);
            const auto out = renderStereo(s, 72000, block, events);
            if (reference.empty()) reference = out;
            else identical = identical && out == reference;
        }
        EXPECT_TRUE(identical);
        EXPECT_GT(rms(reference, 0, reference.size()), 1e-3);
    }
}

DSPARK_TEST(Synth_seed_repeats_and_changes_the_randomness)
{
    auto run = [](int seed) {
        Synth s;
        s.setParameter(P::Drift, 1.0f);
        s.setParameter(P::PhaseMode, 1.0f);
        s.setParameter(P::Unison, 2.0f);
        s.setParameter(P::NoiseLevel, 0.2f);
        s.setParameter(P::Seed, static_cast<float>(seed));
        s.prepare({ kFs, 256, 2 }, 4);
        return renderStereo(s, 20000, 256, [](Synth& sy, int pos, int n) {
            at(10, pos, n, [&](int o) { sy.noteOn(60, 0.8, o); });
        });
    };
    EXPECT_TRUE(run(7) == run(7));
    EXPECT_FALSE(run(7) == run(8));
}

// ============================================================================
// Envelope and velocity
// ============================================================================

DSPARK_TEST(Synth_amplifier_envelope_times_and_velocity)
{
    // Attack 50 ms reaches full level at 50 ms; velocity 0.5 with a 24 dB
    // range plays 12 dB down.
    Synth s;
    sineVoice(s);
    s.setParameter(P::Env1Attack, 50.0f);
    s.setParameter(P::Env1Decay, 10.0f);
    s.setParameter(P::Env1Sustain, 1.0f);
    s.prepare({ kFs, 256, 2 }, 4);
    const auto x = left(renderStereo(s, 12000, 256, [](Synth& sy, int pos, int n) {
        at(0, pos, n, [&](int o) { sy.noteOn(81, 1.0, o); });
    }));
    // Envelope reading: peak of each 1 ms window (48 samples, ~41 cycles of 880 Hz).
    int reached = -1;
    for (int w = 0; w + 48 <= 12000; w += 48)
    {
        double peak = 0.0;
        for (int i = w; i < w + 48; ++i) peak = std::max(peak, std::abs(x[static_cast<size_t>(i)]));
        if (reached < 0 && peak > 0.999) reached = w;
    }
    std::cout << "  full level reached in the window starting at " << reached / 48.0 << " ms\n";
    EXPECT_NEAR(reached / 48.0, 49.0, 1.5);

    auto level = [](double velocity) {
        Synth sv;
        sineVoice(sv);
        sv.prepare({ kFs, 256, 2 }, 4);
        const auto y = left(renderStereo(sv, 24000, 256, [&](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(69, velocity, o); });
        }));
        return rms(y, 4800, 24000);
    };
    const double drop = 20.0 * std::log10(level(0.5) / level(1.0));
    std::cout << "  velocity 0.5: " << drop << " dB\n";
    EXPECT_NEAR(drop, -12.0, 0.01);
}

// ============================================================================
// Voices
// ============================================================================

DSPARK_TEST(Synth_stealing_and_release_never_click)
{
    // Polyphony 3, eight overlapping sine notes: stolen voices fade over 3 ms.
    // The largest sample step stays within that of the summed sines at full
    // level (each contributes at most 2 pi f / fs per sample).
    Synth s;
    sineVoice(s);
    s.setParameter(P::Polyphony, 3.0f);
    s.setParameter(P::Env1Release, 5.0f);
    s.prepare({ kFs, 256, 2 }, 8);
    double maxStep = 0.0;
    int maxActive = 0;
    std::vector<double> l(256), r(256);
    double* ch[2] = { l.data(), r.data() };
    double prev = 0.0;
    for (int b = 0; b < 200; ++b)
    {
        if (b % 10 == 0 && b < 80) s.noteOn(60 + b / 10 * 2, 1.0, (b * 37) % 256);
        if (b == 120) s.allNotesOff(100);
        s.processBlock(AudioBufferView<double>(ch, 2, 256));
        for (int i = 0; i < 256; ++i)
        {
            maxStep = std::max(maxStep, std::abs(l[static_cast<size_t>(i)] - prev));
            prev = l[static_cast<size_t>(i)];
        }
        maxActive = std::max(maxActive, s.getActiveVoiceCount());
    }
    // Three sustained sines up to 74 (587 Hz), plus one fading: bound by four.
    const double bound = 4.0 * 2.0 * pi<double> * 587.33 / kFs;
    std::cout << "  max step " << maxStep << " (bound " << bound << "), voices at most " << maxActive << "\n";
    EXPECT_LT(maxStep, bound);
    EXPECT_LT(maxActive, 5);
    EXPECT_EQ(s.getActiveVoiceCount(), 0);
}

DSPARK_TEST(Synth_legato_glides_without_retriggering)
{
    // Legato, 100 ms glide: the second key moves the pitch from 60 to 72
    // over 100 ms (midway at 66) and the level never dips.
    Synth s;
    sineVoice(s);
    s.setParameter(P::VoiceMode, 2.0f);
    s.setParameter(P::Glide, 100.0f);
    s.setParameter(P::Env1Attack, 20.0f);
    s.prepare({ kFs, 256, 2 }, 4);
    const int second = 9600;
    const auto x = left(renderStereo(s, 30000, 256, [&](Synth& sy, int pos, int n) {
        at(0, pos, n, [&](int o) { sy.noteOn(60, 1.0, o); });
        at(second, pos, n, [&](int o) { sy.noteOn(72, 1.0, o); });
    }));
    // Instantaneous frequency from zero crossings around 50 ms into the glide.
    auto crossingsHz = [&](int from, int to) {
        std::vector<double> times;
        for (int i = from + 1; i < to; ++i)
            if (x[static_cast<size_t>(i - 1)] < 0.0 && x[static_cast<size_t>(i)] >= 0.0)
                times.push_back(i - 1 + x[static_cast<size_t>(i - 1)] / (x[static_cast<size_t>(i - 1)] - x[static_cast<size_t>(i)]));
        return (times.size() - 1) * kFs / (times.back() - times.front());
    };
    const double mid = crossingsHz(second + 2400 - 300, second + 2400 + 300);
    const double midSemis = 69.0 + 12.0 * std::log2(mid / 440.0);
    std::cout << "  pitch halfway through the glide " << midSemis << "\n";
    EXPECT_NEAR(midSemis, 66.0, 0.3);
    const double end = crossingsHz(second + 6000, second + 9000);
    EXPECT_NEAR(69.0 + 12.0 * std::log2(end / 440.0), 72.0, 0.01);
    // No retrigger: the level around the second key stays at full scale.
    EXPECT_GT(rms(x, static_cast<size_t>(second), static_cast<size_t>(second) + 2400), 0.69);
    EXPECT_EQ(s.getActiveVoiceCount(), 1);
}

DSPARK_TEST(Synth_sustain_pedal_holds_released_keys)
{
    Synth s;
    sineVoice(s);
    s.setParameter(P::Env1Release, 10.0f);
    s.prepare({ kFs, 256, 2 }, 4);
    const auto x = left(renderStereo(s, 48000, 256, [](Synth& sy, int pos, int n) {
        at(0, pos, n, [&](int o) { sy.setSustainPedal(true, o); sy.noteOn(64, 1.0, o); });
        at(4800, pos, n, [&](int o) { sy.noteOff(64, 0.0, o); });
        at(24000, pos, n, [&](int o) { sy.setSustainPedal(false, o); });
    }));
    EXPECT_GT(rms(x, 9600, 24000), 0.6);     // held by the pedal
    EXPECT_LT(rms(x, 26400, 48000), 1e-4);   // released with it
}

DSPARK_TEST(Synth_unison_detunes_spreads_and_keeps_its_level)
{
    auto run = [](int unison, double spread) {
        Synth s;
        s.setParameter(P::Unison, static_cast<float>(unison));
        s.setParameter(P::UnisonDetune, 30.0f);
        s.setParameter(P::UnisonSpread, static_cast<float>(spread));
        s.setParameter(P::PhaseMode, 1.0f);
        s.setParameter(P::FilterType, 0.0f);
        s.prepare({ kFs, 256, 2 }, 4);
        return renderStereo(s, 48000, 256, [](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(57, 1.0, o); });
        });
    };
    const auto one = run(1, 0.0), four = run(4, 1.0), narrow = run(4, 0.0);
    const double levelDb = 20.0 * std::log10(rms(four, 9600, four.size()) / rms(one, 9600, one.size()));
    std::cout << "  unison 4 level " << levelDb << " dB against 1\n";
    EXPECT_NEAR(levelDb, 0.0, 2.0);
    double diffSpread = 0.0, diffNarrow = 0.0;
    for (size_t i = 0; i < four.size(); i += 2)
    {
        diffSpread = std::max(diffSpread, std::abs(four[i] - four[i + 1]));
        diffNarrow = std::max(diffNarrow, std::abs(narrow[i] - narrow[i + 1]));
    }
    EXPECT_GT(diffSpread, 0.1);    // spread: left and right differ
    EXPECT_EQ(diffNarrow, 0.0);    // no spread: mono

    // Opposite pans mirror exactly.
    auto panned = [](float pan) {
        Synth s;
        s.setParameter(P::AmpPan, pan);
        s.setParameter(P::FilterType, 0.0f);
        s.prepare({ kFs, 256, 2 }, 4);
        return renderStereo(s, 4800, 256, [](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(57, 1.0, o); });
        });
    };
    const auto toRight = panned(0.3f), toLeft = panned(-0.3f);
    bool mirrored = true;
    for (size_t i = 0; i < toRight.size(); i += 2)
        mirrored = mirrored && toRight[i] == toLeft[i + 1] && toRight[i + 1] == toLeft[i];
    EXPECT_TRUE(mirrored);
    EXPECT_GT(rms(toRight, 0, toRight.size()), 0.1);
}

// ============================================================================
// Filter
// ============================================================================

DSPARK_TEST(Synth_filters_have_their_slopes)
{
    // White noise through each type, cutoff 1 kHz: the stopband falls by
    // about 24 dB per octave (ladder LP24) or 12 (SVF LP) between 4 and 8 kHz,
    // and the high-pass keeps the top while cutting below.
    auto spectrum = [](int type) {
        Synth s;
        s.setParameter(P::Osc1Level, 0.0f);
        s.setParameter(P::NoiseLevel, 1.0f);
        s.setParameter(P::FilterType, static_cast<float>(type));
        s.setParameter(P::FilterCutoff, 1000.0f);
        s.prepare({ kFs, 512, 2 }, 4);
        const auto x = left(renderStereo(s, 48000 * 4, 512, [](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(60, 1.0, o); });
        }));
        // Averaged power per band from 4096-point Hann spectra.
        constexpr size_t N = 4096;
        std::vector<double> win(N), t(N), f(N + 2), power(N / 2 + 1, 0.0);
        WindowFunctions<double>::hann(win.data(), static_cast<int>(N), false);
        FFTReal<double> fft(N);
        for (size_t start = 9600; start + N <= x.size(); start += N)
        {
            for (size_t i = 0; i < N; ++i) t[i] = x[start + i] * win[i];
            fft.forward(t.data(), f.data());
            for (size_t k = 0; k <= N / 2; ++k) power[k] += f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
        }
        return [power](double hz) {
            const auto k = static_cast<size_t>(std::llround(hz / (kFs / N)));
            double sum = 0.0;
            for (size_t j = k - 4; j <= k + 4; ++j) sum += power[j];
            return 10.0 * std::log10(sum);
        };
    };
    const auto ladder = spectrum(1), svf = spectrum(5), hp = spectrum(6);
    const double ladderSlope = ladder(4000.0) - ladder(8000.0);
    const double svfSlope = svf(4000.0) - svf(8000.0);
    std::cout << "  octave 4-8 kHz: ladder " << ladderSlope << " dB, svf " << svfSlope << " dB\n";
    EXPECT_NEAR(ladderSlope, 24.0, 3.0);
    EXPECT_NEAR(svfSlope, 12.0, 2.0);
    EXPECT_GT(hp(8000.0) - hp(250.0), 20.0);
}

DSPARK_TEST(Synth_filter_envelope_opens_and_closes_the_filter)
{
    // Cutoff 200 Hz plus 5 octaves of envelope (decay 200 ms, sustain 0): the
    // early sound carries far more energy above 1 kHz than the late one.
    Synth s;
    s.setParameter(P::FilterCutoff, 200.0f);
    s.setParameter(P::FilterEnvAmount, 5.0f);
    s.setParameter(P::Env2Attack, 1.0f);
    s.setParameter(P::Env2Decay, 200.0f);
    s.prepare({ kFs, 256, 2 }, 4);
    const auto x = left(renderStereo(s, 48000, 256, [](Synth& sy, int pos, int n) {
        at(0, pos, n, [&](int o) { sy.noteOn(45, 1.0, o); });
    }));
    // Share of the energy above 1 kHz in a 4096-sample window.
    auto highShareDb = [&](size_t from) {
        constexpr size_t N = 4096;
        std::vector<double> win(N), t(N), f(N + 2);
        WindowFunctions<double>::hann(win.data(), static_cast<int>(N), false);
        for (size_t i = 0; i < N; ++i) t[i] = x[from + i] * win[i];
        FFTReal<double> fft(N);
        fft.forward(t.data(), f.data());
        double high = 0.0, all = 0.0;
        for (size_t k = 1; k <= N / 2; ++k)
        {
            const double p = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
            all += p;
            if (static_cast<double>(k) * kFs / N > 1000.0) high += p;
        }
        return 10.0 * std::log10(high / all);
    };
    const double early = highShareDb(480), late = highShareDb(40000);
    std::cout << "  energy above 1 kHz: early " << early << " dB, late " << late << " dB\n";
    EXPECT_GT(early - late, 30.0);
}

// ============================================================================
// Modulation
// ============================================================================

DSPARK_TEST(Synth_lfo_vibrato_depth_follows_the_matrix_and_the_wheel)
{
    // LFO 1 (sine, 4 Hz) to Pitch, 0.5 semitone, via the mod wheel: no
    // vibrato with the wheel down, +-0.5 semitone with it up.
    auto excursion = [](double wheel) {
        Synth s;
        sineVoice(s);
        s.setParameter(P::Lfo1Rate, 4.0f);
        s.setParameter(P::Mod1Source, static_cast<float>(Synth::ModSource::Lfo1));
        s.setParameter(P::Mod1Via, static_cast<float>(Synth::ModSource::ModWheel));
        s.setParameter(P::Mod1Destination, static_cast<float>(Synth::ModDest::Pitch));
        s.setParameter(P::Mod1Amount, 0.5f);
        s.prepare({ kFs, 256, 2 }, 4);
        const auto x = left(renderStereo(s, 48000, 256, [&](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.setModWheel(wheel, o); sy.noteOn(69, 1.0, o); });
        }));
        // Period-by-period frequency from zero crossings.
        double lo = 1e9, hi = 0.0, last = -1.0;
        for (size_t i = 1; i < x.size(); ++i)
            if (x[i - 1] < 0.0 && x[i] >= 0.0)
            {
                const double t = static_cast<double>(i - 1) + x[i - 1] / (x[i - 1] - x[i]);
                if (last > 4800.0)
                {
                    const double st = 12.0 * std::log2(kFs / (t - last) / 440.0);
                    lo = std::min(lo, st);
                    hi = std::max(hi, st);
                }
                last = t;
            }
        return std::make_pair(lo, hi);
    };
    const auto off = excursion(0.0), on = excursion(1.0);
    std::cout << "  vibrato wheel 0: " << off.first << ".." << off.second
              << " st, wheel 1: " << on.first << ".." << on.second << " st\n";
    EXPECT_NEAR(off.first, 0.0, 1e-3);
    EXPECT_NEAR(off.second, 0.0, 1e-3);
    EXPECT_NEAR(on.first, -0.5, 0.02);
    EXPECT_NEAR(on.second, 0.5, 0.02);
}

DSPARK_TEST(Synth_free_lfo_is_shared_by_every_voice)
{
    // A free-running square LFO on the amplifier gates two notes started at
    // different times in step; a per-note one restarts with each note.
    auto gates = [](int mode) {
        Synth s;
        sineVoice(s);
        s.setParameter(P::Lfo1Shape, 4.0f);
        s.setParameter(P::Lfo1Rate, 10.0f);
        s.setParameter(P::Lfo1Mode, static_cast<float>(mode));
        s.setParameter(P::Mod1Source, static_cast<float>(Synth::ModSource::Lfo1));
        s.setParameter(P::Mod1Destination, static_cast<float>(Synth::ModDest::Amp));
        s.setParameter(P::Mod1Amount, -1.0f);   // gain 1 - lfo: 0 on high, 2 on low halves
        s.prepare({ kFs, 256, 2 }, 4);
        Synth t;
        sineVoice(t);
        t.setParameter(P::Lfo1Shape, 4.0f);
        t.setParameter(P::Lfo1Rate, 10.0f);
        t.setParameter(P::Lfo1Mode, static_cast<float>(mode));
        t.setParameter(P::Mod1Source, static_cast<float>(Synth::ModSource::Lfo1));
        t.setParameter(P::Mod1Destination, static_cast<float>(Synth::ModDest::Amp));
        t.setParameter(P::Mod1Amount, -1.0f);
        t.prepare({ kFs, 256, 2 }, 4);
        // The same note started 1000 samples later in the second synth.
        const auto a = left(renderStereo(s, 24000, 256, [](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(69, 1.0, o); });
        }));
        const auto b = left(renderStereo(t, 24000, 256, [](Synth& sy, int pos, int n) {
            at(1000, pos, n, [&](int o) { sy.noteOn(69, 1.0, o); });
        }));
        // Silent stretches (gain 0) of each, sampled every 100 samples.
        int agree = 0, count = 0;
        for (size_t i = 6000; i + 100 < a.size(); i += 100)
        {
            const bool qa = rms(a, i, i + 48) < 1e-3;
            const bool qb = rms(b, i, i + 48) < 1e-3;
            agree += (qa == qb) ? 1 : 0;
            ++count;
        }
        return static_cast<double>(agree) / count;
    };
    const double freeAgreement = gates(0), retriggerAgreement = gates(1);
    std::cout << "  gate agreement: free " << freeAgreement << ", retrigger " << retriggerAgreement << "\n";
    EXPECT_GT(freeAgreement, 0.9);
    EXPECT_LT(retriggerAgreement, 0.7);
}

// ============================================================================
// FM
// ============================================================================

DSPARK_TEST(Synth_fm_sidebands_follow_the_bessel_functions)
{
    // Operator 2 (ratio 0.25, level 0.05: index 4 pi 0.05) modulates operator
    // 1 at 1500 Hz. Everything is a multiple of 375 Hz, whose period is 128
    // samples, so a rectangular DFT over whole periods is exact: the
    // sidebands at 1500 + 375 n relative to the carrier are J_n / J_0.
    Synth s;
    s.setParameter(P::Engine, 1.0f);
    s.setParameter(P::FilterType, 0.0f);
    s.setParameter(P::Op2Level, 0.05f);
    s.setParameter(P::Op2Ratio, 0.25f);
    s.setParameter(P::Op1Attack, 0.0f);
    s.setParameter(P::Op2Attack, 0.0f);
    const double key = 69.0 + 12.0 * std::log2(1500.0 / 440.0);
    s.setParameter(P::MasterTune, static_cast<float>(100.0 * (key - std::floor(key))));
    s.prepare({ kFs, 256, 2 }, 4);
    const auto x = left(renderStereo(s, 48000, 256, [&](Synth& sy, int pos, int n) {
        at(0, pos, n, [&](int o) { sy.noteOn(static_cast<int>(std::floor(key)), 1.0, o); });
    }));
    // Refine: measure the carrier frequency and confirm it sits on 1500 Hz.
    const double fc = toneHz(x, 8000, 1500.0);
    EXPECT_NEAR(cents(fc, 1500.0), 0.0, 0.005);
    auto amplitude = [&](double hz) {
        constexpr size_t N = 128 * 200;
        double re = 0.0, im = 0.0;
        for (size_t i = 0; i < N; ++i)
        {
            const double t = static_cast<double>(i);
            re += x[12000 + i] * std::cos(2.0 * pi<double> * hz * t / kFs);
            im += x[12000 + i] * std::sin(2.0 * pi<double> * hz * t / kFs);
        }
        return std::hypot(re, im);
    };
    const double beta = 4.0 * pi<double> * 0.05;
    const double carrier = amplitude(1500.0);
    double worst = 0.0;
    for (int n = 1; n <= 3; ++n)
    {
        const double measured = 20.0 * std::log10(amplitude(1500.0 + 375.0 * n) / carrier);
        const double expected = 20.0 * std::log10(std::abs(besselJ(n, beta) / besselJ(0, beta)));
        worst = std::max(worst, std::abs(measured - expected));
    }
    std::cout << "  worst sideband error " << worst << " dB\n";
    EXPECT_LT(worst, 0.01);
}

// ============================================================================
// Wavetable
// ============================================================================

DSPARK_TEST(Synth_wavetable_position_morphs_the_timbre)
{
    // Basic bank: position 0 is a sine (no second harmonic), position 0.4 is
    // frame 2, the saw (second harmonic at half the fundamental).
    auto secondHarmonicDb = [](double position) {
        Synth s;
        s.setParameter(P::Engine, 2.0f);
        s.setParameter(P::FilterType, 0.0f);
        s.setParameter(P::Wt1Position, static_cast<float>(position));
        s.prepare({ kFs, 256, 2 }, 4);
        const auto x = left(renderStereo(s, 30000, 256, [](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOn(57, 1.0, o); });
        }));
        auto amp = [&](double hz) {
            double re = 0.0, im = 0.0;
            for (size_t i = 0; i < 19200; ++i)
            {
                const double t = static_cast<double>(i);
                re += x[9600 + i] * std::cos(2.0 * pi<double> * hz * t / kFs);
                im += x[9600 + i] * std::sin(2.0 * pi<double> * hz * t / kFs);
            }
            return std::hypot(re, im);
        };
        return 20.0 * std::log10(amp(440.0) / amp(220.0));
    };
    const double sine = secondHarmonicDb(0.0), saw = secondHarmonicDb(0.4);
    std::cout << "  second harmonic: sine frame " << sine << " dB, saw frame " << saw << " dB\n";
    EXPECT_LT(sine, -100.0);
    EXPECT_NEAR(saw, -6.0206, 0.01);
}

// ============================================================================
// Sample source
// ============================================================================

DSPARK_TEST(Synth_sample_source_matches_sampler)
{
    // The same looped zone through Sampler and through the synthesizer's
    // sample source (envelopes, velocity range and gains matched, filter
    // off): one note at the root (gather form) and one a fifth up (scatter
    // form). They share the zone reader, so they agree to rounding.
    std::vector<double> rec(40000);
    for (size_t i = 0; i < rec.size(); ++i)
    {
        const double t = static_cast<double>(i) / 44100.0;
        rec[i] = 0.5 * std::sin(2.0 * pi<double> * 330.0 * t) + 0.2 * std::sin(2.0 * pi<double> * 1271.0 * t);
    }
    const double* chans[1] = { rec.data() };
    SampleZoneSpec<double> zone;
    zone.audio = AudioBufferView<const double>(chans, 1, 40000);
    zone.sampleRate = 44100.0;
    zone.rootKey = 64;
    zone.tuneCents = 13.0;
    zone.loopMode = SampleLoopMode::Sustain;
    zone.loopStart = 10000;
    zone.loopEnd = 30000;
    zone.loopCrossfade = 800;

    for (int key : { 64, 71 })
    {
        Sampler<double> sampler;
        sampler.prepare({ kFs, 256, 2 }, 4);
        sampler.setZones(std::span<const SampleZoneSpec<double>>(&zone, 1));
        sampler.setEnvelope(1.0, 100.0, 1.0, 100.0);

        Synth synth;
        synth.setParameter(P::Osc1Level, 0.0f);
        synth.setParameter(P::SampleLevel, 1.0f);
        synth.setParameter(P::FilterType, 0.0f);
        synth.setParameter(P::MasterVolume, 0.0f);
        synth.setParameter(P::Env1Attack, 1.0f);
        synth.setParameter(P::Env1Decay, 100.0f);
        synth.setParameter(P::Env1Sustain, 1.0f);
        synth.setParameter(P::Env1Release, 100.0f);
        synth.prepare({ kFs, 256, 2 }, 4);
        synth.setSampleZones(std::span<const SampleZoneSpec<double>>(&zone, 1));

        std::vector<double> a(256), b(256), c(256), d(256);
        double* sa[2] = { a.data(), b.data() };
        double* sb[2] = { c.data(), d.data() };
        double worst = 0.0, peak = 0.0;
        for (int blk = 0; blk < 400; ++blk)
        {
            if (blk == 1) { sampler.noteOn(key, 0.8, 17); synth.noteOn(key, 0.8, 17); }
            if (blk == 250) { sampler.noteOff(key, 5); synth.noteOff(key, 0.0, 5); }
            sampler.processBlock(AudioBufferView<double>(sa, 2, 256));
            synth.processBlock(AudioBufferView<double>(sb, 2, 256));
            for (int i = 0; i < 256; ++i)
            {
                worst = std::max(worst, std::abs(a[static_cast<size_t>(i)] - c[static_cast<size_t>(i)]));
                worst = std::max(worst, std::abs(b[static_cast<size_t>(i)] - d[static_cast<size_t>(i)]));
                peak = std::max(peak, std::abs(a[static_cast<size_t>(i)]));
            }
        }
        std::cout << "  key " << key << ": peak " << peak << ", largest difference " << worst << "\n";
        EXPECT_GT(peak, 0.3);
        EXPECT_LT(worst, 1e-9);
    }
}

// ============================================================================
// Oversampling, state and robustness
// ============================================================================

DSPARK_TEST(Synth_oversampling_reports_its_latency_exactly)
{
    // A 1x and a 2x/4x render of the same slow-attack note: the half-level
    // crossing of the envelope moves by exactly the reported latency.
    auto crossing = [](int os, int& latency) {
        Synth s;
        sineVoice(s);
        s.setParameter(P::Env1Attack, 20.0f);
        s.setParameter(P::Oversampling, static_cast<float>(os));
        s.prepare({ kFs, 256, 2 }, 4);
        latency = s.getLatency();
        const auto x = left(renderStereo(s, 9600, 256, [](Synth& sy, int pos, int n) {
            at(300, pos, n, [&](int o) { sy.noteOn(45, 1.0, o); });
        }));
        // Envelope of a 110 Hz sine by the running peak over 436 samples (one period).
        for (size_t i = 436; i < x.size(); ++i)
        {
            double peak = 0.0;
            for (size_t j = i - 436; j < i; ++j) peak = std::max(peak, std::abs(x[j]));
            if (peak > 0.5) return static_cast<double>(i);
        }
        return -1.0;
    };
    int l1 = 0, l2 = 0, l4 = 0;
    const double c1 = crossing(0, l1), c2 = crossing(1, l2), c4 = crossing(2, l4);
    EXPECT_EQ(l1, 0);
    EXPECT_EQ(l2, 64);
    EXPECT_EQ(l4, 96);
    EXPECT_NEAR(c2 - c1, l2, 1.0);
    EXPECT_NEAR(c4 - c1, l4, 1.0);
}

DSPARK_TEST(Synth_state_round_trips_every_parameter)
{
    Synth a;
    for (int i = 0; i < Synth::kNumParams; ++i)
    {
        const auto& info = Synth::getParamInfo(static_cast<P>(i));
        // A value away from the default: 30% into the range, rounded for steps.
        a.setParameter(static_cast<P>(i), info.minValue + 0.3f * (info.maxValue - info.minValue));
    }
    const auto blob = a.getState();
    Synth b;
    EXPECT_TRUE(b.setState(blob.data(), blob.size()));
    int mismatches = 0;
    for (int i = 0; i < Synth::kNumParams; ++i)
        mismatches += a.getParameter(static_cast<P>(i)) != b.getParameter(static_cast<P>(i)) ? 1 : 0;
    EXPECT_EQ(mismatches, 0);
    // A foreign or damaged blob is refused and leaves the parameters alone.
    std::vector<uint8_t> junk = { 1, 2, 3 };
    EXPECT_FALSE(b.setState(junk.data(), junk.size()));
    EXPECT_EQ(b.getParameter(P::FilterCutoff), a.getParameter(P::FilterCutoff));
}

DSPARK_TEST(Synth_is_silent_and_finite_on_hostile_input)
{
    Synth s;
    double buffer[64] = {};
    double* ch[1] = { buffer };
    s.processBlock(AudioBufferView<double>(ch, 1, 64));   // before prepare: silence
    for (double v : buffer) EXPECT_EQ(v, 0.0);
    s.setParameter(P::FilterResonance, 1.0f);
    s.setParameter(P::FilterDrive, 10.0f);
    s.setParameter(P::Unison, 8.0f);
    s.setParameter(P::Engine, 1.0f);
    s.setParameter(P::Op2Level, 1.0f);
    s.setParameter(P::Op3Level, 1.0f);
    s.setParameter(P::Op4Level, 1.0f);
    s.setParameter(P::FmFeedback, 1.0f);
    s.prepare({ kFs, 64, 1 }, 2);
    s.noteOn(-5, std::numeric_limits<double>::quiet_NaN(), 3);
    s.noteOn(500, 2.0, -7);                 // clamped key, velocity and offset
    s.noteOn(127, 1.0, 1000);               // offset past the block: applied at its end
    s.setPitchBend(std::numeric_limits<double>::infinity());
    s.setController(64, 0.9);
    bool finite = true;
    double peak = 0.0;
    for (int b = 0; b < 400; ++b)
    {
        s.processBlock(AudioBufferView<double>(ch, 1, 64));
        for (double v : buffer)
        {
            finite = finite && std::isfinite(v);
            peak = std::max(peak, std::abs(v));
        }
    }
    EXPECT_TRUE(finite);
    EXPECT_GT(peak, 0.0);
    EXPECT_LT(peak, 16.0);
    // A block longer than prepared is rendered in pieces.
    std::vector<double> big(1000, 1.0);
    double* bigCh[1] = { big.data() };
    s.processBlock(AudioBufferView<double>(bigCh, 1, 1000));
    for (double v : big) EXPECT_TRUE(std::isfinite(v));
}

// ============================================================================
// Lfo
// ============================================================================

DSPARK_TEST(Lfo_shapes_rates_sync_delay_and_seeds)
{
    using L = Lfo<double>;
    // Shapes at quarter-cycle points.
    EXPECT_NEAR(L::valueAt(L::Shape::Sine, 0.25, 0), 1.0, 1e-12);
    EXPECT_NEAR(L::valueAt(L::Shape::Triangle, 0.25, 0), 1.0, 1e-12);
    EXPECT_NEAR(L::valueAt(L::Shape::Triangle, 0.75, 0), -1.0, 1e-12);
    EXPECT_NEAR(L::valueAt(L::Shape::SawUp, 0.0, 0), -1.0, 1e-12);
    EXPECT_NEAR(L::valueAt(L::Shape::SawDown, 0.5, 0), 0.0, 1e-12);
    EXPECT_EQ(L::valueAt(L::Shape::Square, 0.49, 0), 1.0);
    EXPECT_EQ(L::valueAt(L::Shape::Square, 0.51, 0), -1.0);
    // Sample and hold: constant within a cycle, repeatable per seed.
    EXPECT_EQ(L::valueAt(L::Shape::SampleAndHold, 3.1, 9), L::valueAt(L::Shape::SampleAndHold, 3.9, 9));
    EXPECT_NE(L::valueAt(L::Shape::SampleAndHold, 3.1, 9), L::valueAt(L::Shape::SampleAndHold, 4.1, 9));
    EXPECT_NE(L::valueAt(L::Shape::SampleAndHold, 3.1, 9), L::valueAt(L::Shape::SampleAndHold, 3.1, 10));
    // Smooth random: continuous at the cycle boundary.
    EXPECT_NEAR(L::valueAt(L::Shape::SmoothRandom, 5.0 - 1e-9, 4), L::valueAt(L::Shape::SmoothRandom, 5.0, 4), 1e-6);
    double lo = 1.0, hi = -1.0;
    for (int k = 0; k < 10000; ++k)
    {
        const double r = L::random(1, k);
        lo = std::min(lo, r);
        hi = std::max(hi, r);
    }
    EXPECT_TRUE(lo >= -1.0 && hi < 1.0 && lo < -0.99 && hi > 0.99);

    // Rate and tempo sync: 120 BPM eighth notes are 4 Hz.
    L lfo;
    lfo.prepare(48000.0);
    lfo.setRateHz(3.0);
    EXPECT_NEAR(lfo.getRateHz(), 3.0, 1e-12);
    lfo.setTempo(120.0);
    lfo.setDivision(L::Division::Eighth);
    lfo.setTempoSync(true);
    EXPECT_NEAR(lfo.getRateHz(), 4.0, 1e-12);
    lfo.reset();
    (void)lfo.advance(48000);
    EXPECT_NEAR(lfo.getCycles(), 4.0, 1e-9);

    // Delay 100 ms, fade-in 100 ms: silent, then a linear rise.
    lfo.setDelayMs(100.0);
    lfo.setFadeInMs(100.0);
    EXPECT_EQ(lfo.depthAt(4799), 0.0);
    EXPECT_NEAR(lfo.depthAt(4800 + 2400), 0.5, 1e-12);
    EXPECT_EQ(lfo.depthAt(9600), 1.0);
    // advance() reports the value, then moves.
    lfo.setTempoSync(false);
    lfo.setRateHz(1.0);
    lfo.setDelayMs(0.0);
    lfo.setFadeInMs(0.0);
    lfo.setStartPhase(0.25);
    lfo.reset();
    EXPECT_NEAR(lfo.advance(12000), 1.0, 1e-12);   // sine at 0.25
    EXPECT_NEAR(lfo.value(), 0.0, 1e-12);          // half a cycle later... at 0.5
}

// ============================================================================
// Individual note tuning
// ============================================================================

namespace {

template <typename T, typename Player, typename Events>
std::vector<double> tunedRender(Player& player, int total, int block, Events events)
{
    std::vector<double> out(static_cast<size_t>(total));
    std::vector<T> l(static_cast<size_t>(block)), r(static_cast<size_t>(block));
    for (int pos = 0; pos < total; pos += block)
    {
        const int n = std::min(block, total - pos);
        events(player, pos, n);
        T* channels[2] { l.data(), r.data() };
        player.processBlock(AudioBufferView<T>(channels, 2, n));
        for (int i = 0; i < n; ++i) out[static_cast<size_t>(pos + i)] = l[static_cast<size_t>(i)];
    }
    return out;
}

template <typename T>
void tunedSine(Synthesizer<T>& s, int engine = 0)
{
    using Q = typename Synthesizer<T>::Param;
    s.setParameter(Q::Engine, static_cast<float>(engine));
    s.setParameter(Q::Osc1Wave, 0.0f);
    s.setParameter(Q::FilterType, 0.0f);
    s.setParameter(Q::Env1Attack, 0.0f);
    s.setParameter(Q::MasterVolume, 0.0f);
    s.prepare({ kFs, 256, 2 }, 8);
}

template <typename T>
void tunedSample(Sampler<T>& s, double zoneCents = 0.0)
{
    std::vector<T> recording(48000);
    for (size_t i = 0; i < recording.size(); ++i)
        recording[i] = static_cast<T>(0.5 * std::sin(2.0 * pi<double> * 440.0 * static_cast<double>(i) / kFs));
    const T* input[1] { recording.data() };
    SampleZoneSpec<T> zone;
    zone.audio = AudioBufferView<const T>(input, 1, 48000);
    zone.sampleRate = kFs;
    zone.rootKey = 69;
    zone.tuneCents = zoneCents;
    zone.loopMode = SampleLoopMode::Continuous;
    zone.loopEnd = 48000;
    s.prepare({ kFs, 256, 2 }, 8);
    s.setEnvelope(T(0.01), T(1), T(1), T(10));
    s.setVelocityRange(T(0));
    EXPECT_TRUE(s.setZones(std::span<const SampleZoneSpec<T>>(&zone, 1)));
}

double tunedDifference(const std::vector<double>& a, const std::vector<double>& b)
{
    double error = 0.0;
    for (size_t i = 0; i < a.size(); ++i) error = std::max(error, std::abs(a[i] - b[i]));
    return error;
}

} // namespace

DSPARK_TEST(NoteTuning_synth_pitch_float_and_double_all_engines)
{
    auto run = []<typename T>() {
        double worst = 0.0;
        for (int engine : { 0, 1, 2 })
            for (int key : { 33, 69, 93 })
                for (double tuning : { -100.0, -37.5, 45.25, 100.0 })
                {
                    Synthesizer<T> s;
                    tunedSine(s, engine);
                    const auto x = tunedRender<T>(s, 40000, 256, [&](auto& sy, int pos, int n) {
                        at(17, pos, n, [&](int o) { sy.noteOnTuned(key, T(1), static_cast<T>(tuning), o); });
                    });
                    const double f = 440.0 * std::exp2((key - 69 + tuning / 100.0) / 12.0);
                    worst = std::max(worst, std::abs(cents(toneHz(x, 4000, f), f)));
                    EXPECT_GT(rms(x, 4000, 39000), 0.1);
                }
        std::cout << "  tuned synth " << sizeof(T) << "-byte worst pitch error " << worst << " cents\n";
        EXPECT_LT(worst, 0.01);
    };
    run.template operator()<float>();
    run.template operator()<double>();
}

DSPARK_TEST(NoteTuning_sampler_pitch_adds_zone_tuning_and_bend)
{
    auto run = []<typename T>() {
        double worst = 0.0;
        for (int key : { 57, 69, 81 })
            for (double tuning : { -100.0, -37.5, 45.25, 100.0 })
            {
                Sampler<T> s;
                tunedSample(s, 13.0);
                const auto x = tunedRender<T>(s, 40000, 127, [&](auto& sy, int pos, int n) {
                    at(17, pos, n, [&](int o) {
                        sy.setPitchBend(T(-0.5), o);
                        sy.noteOnTuned(key, T(1), static_cast<T>(tuning), o);
                    });
                });
                const double f = 440.0 * std::exp2((key - 69 + (13.0 + tuning) / 100.0 - 0.5) / 12.0);
                worst = std::max(worst, std::abs(cents(toneHz(x, 4000, f), f)));
                EXPECT_GT(rms(x, 4000, 39000), 0.2);
            }
        std::cout << "  tuned sampler " << sizeof(T) << "-byte worst pitch error " << worst << " cents\n";
        EXPECT_LT(worst, 0.01);
    };
    run.template operator()<float>();
    run.template operator()<double>();
}

DSPARK_TEST(NoteTuning_legacy_zero_and_event_partition_are_identical)
{
    auto run = []<typename T>() {
        auto render = [&](bool sampler, int block, bool tuned, T tuning) {
            auto events = [&](auto& s, int pos, int n) {
                at(37, pos, n, [&](int o) {
                    if (tuned) s.noteOnTuned(69, T(0.8), tuning, o);
                    else s.noteOn(69, T(0.8), o);
                });
                at(1451, pos, n, [&](int o) { s.setSustainPedal(true, o); });
                at(2103, pos, n, [&](int o) { s.noteOnTuned(69, T(0), T(70), o); });
                at(3177, pos, n, [&](int o) { s.setSustainPedal(false, o); });
            };
            if (sampler) { Sampler<T> s; tunedSample(s); return tunedRender<T>(s, 4800, block, events); }
            Synthesizer<T> s;
            tunedSine(s);
            s.setParameter(Synthesizer<T>::Param::Env1Release, 10.0f);
            return tunedRender<T>(s, 4800, block, events);
        };
        for (bool sampler : { false, true })
        {
            const auto legacy = render(sampler, 256, false, T(0));
            EXPECT_EQ(tunedDifference(legacy, render(sampler, 256, true, T(0))), 0.0);
            const auto reference = render(sampler, 256, true, T(-62.5));
            EXPECT_EQ(tunedDifference(reference, render(sampler, 1, true, T(-62.5))), 0.0);
            EXPECT_EQ(tunedDifference(reference, render(sampler, 17, true, T(-62.5))), 0.0);
            EXPECT_EQ(rms(reference, 0, 37), 0.0);
            EXPECT_GT(rms(reference, 2200, 3000), 0.1);
            EXPECT_EQ(rms(reference, 4400, 4800), 0.0);
        }
    };
    run.template operator()<float>();
    run.template operator()<double>();
}

DSPARK_TEST(NoteTuning_nonfinite_refused_and_cents_clamped)
{
    auto run = []<typename T>() {
        for (bool sampler : { false, true })
        {
            auto render = [&](T tuning, T velocity = T(1)) {
                auto events = [&](auto& s, int pos, int n) {
                    at(0, pos, n, [&](int o) { s.noteOnTuned(69, velocity, tuning, o); });
                };
                if (sampler) { Sampler<T> s; tunedSample(s); return tunedRender<T>(s, 1000, 256, events); }
                Synthesizer<T> s; tunedSine(s); return tunedRender<T>(s, 1000, 256, events);
            };
            EXPECT_EQ(tunedDifference(render(T(1000)), render(T(100))), 0.0);
            EXPECT_EQ(tunedDifference(render(T(-1000)), render(T(-100))), 0.0);
            for (T value : { std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN() })
            {
                EXPECT_EQ(rms(render(value), 0, 1000), 0.0);
                EXPECT_EQ(rms(render(T(0), value), 0, 1000), 0.0);
            }
        }
    };
    run.template operator()<float>();
    run.template operator()<double>();
}

DSPARK_TEST(NoteTuning_mono_and_legato_restore_held_tuning)
{
    for (float mode : { 1.0f, 2.0f })
    {
        Synth s;
        sineVoice(s);
        s.setParameter(P::VoiceMode, mode);
        s.setParameter(P::Glide, 20.0f);
        s.prepare({ kFs, 256, 2 }, 4);
        const auto x = left(renderStereo(s, 100000, 127, [&](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOnTuned(69, 1.0, -70.0, o); });
            at(40000, pos, n, [&](int o) { sy.noteOnTuned(76, 1.0, 100.0, o); });
            at(50000, pos, n, [&](int o) { sy.noteOff(76, 0.0, o); });
        }));
        const double f = 440.0 * std::exp2(-0.7 / 12.0);
        EXPECT_NEAR(cents(toneHz(x, 4000, f), f), 0.0, 0.01);
        EXPECT_NEAR(cents(toneHz(x, 55000, f), f), 0.0, 0.01);
        EXPECT_EQ(s.getActiveVoiceCount(), 1);
    }
}

DSPARK_TEST(NoteTuning_glide_accepts_pitch_below_key_zero)
{
    // A -100 cent note at key 0 is a valid previous pitch, not an unset
    // sentinel. Adding +100 master cents is the independent equivalent.
    auto render = [](bool tuned) {
        Synth s;
        sineVoice(s);
        s.setParameter(P::MasterTune, tuned ? 100.0f : 0.0f);
        s.setParameter(P::Glide, 100.0f);
        s.setParameter(P::Env1Release, 0.0f);
        s.prepare({ kFs, 256, 2 }, 4);
        return left(renderStereo(s, 12000, 256, [&](Synth& sy, int pos, int n) {
            at(0, pos, n, [&](int o) { sy.noteOnTuned(0, 1.0, tuned ? -100.0 : 0.0, o); });
            at(800, pos, n, [&](int o) { sy.noteOff(0, 0.0, o); });
            at(1000, pos, n, [&](int o) { sy.noteOnTuned(72, 1.0, tuned ? -100.0 : 0.0, o); });
        }));
    };
    const double worst = tunedDifference(render(false), render(true));
    std::cout << "  equivalent tuned glide residual " << worst << "\n";
    EXPECT_LT(worst, 1e-10);
}

DSPARK_TEST(NoteTuning_polyphonic_notes_have_independent_pitch)
{
    auto render = [](int which) {
        Synth s;
        sineVoice(s);
        s.prepare({ kFs, 256, 2 }, 4);
        return left(renderStereo(s, 4000, 256, [&](Synth& sy, int pos, int n) {
            if (which != 2) at(0, pos, n, [&](int o) { sy.noteOnTuned(60, 1.0, -100.0, o); });
            if (which != 1) at(0, pos, n, [&](int o) { sy.noteOnTuned(69, 1.0, 65.0, o); });
        }));
    };
    const auto a = render(1), b = render(2), sum = render(0);
    double worst = 0.0;
    for (size_t i = 0; i < sum.size(); ++i) worst = std::max(worst, std::abs(sum[i] - a[i] - b[i]));
    EXPECT_LT(worst, 1e-12);
}

DSPARK_TEST(NoteTuning_sample_zone_selection_uses_integer_key)
{
    std::vector<double> rec(4096, 0.5);
    const double* input[1] { rec.data() };
    SampleZoneSpec<double> zone;
    zone.audio = AudioBufferView<const double>(input, 1, 4096);
    zone.sampleRate = kFs;
    zone.rootKey = zone.keyLow = zone.keyHigh = 69;
    Sampler<double> sampler;
    sampler.prepare({ kFs, 256, 2 });
    sampler.setEnvelope(0.01, 1.0, 1.0, 10.0);
    EXPECT_TRUE(sampler.setZones(std::span<const SampleZoneSpec<double>>(&zone, 1)));
    Synth synth;
    sineVoice(synth);
    synth.setParameter(P::Osc1Level, 0.0f);
    synth.setParameter(P::SampleLevel, 1.0f);
    synth.prepare({ kFs, 256, 2 });
    EXPECT_TRUE(synth.setSampleZones(std::span<const SampleZoneSpec<double>>(&zone, 1)));
    auto outside = [](auto& s, int pos, int n) {
        at(0, pos, n, [&](int o) { s.noteOnTuned(68, 1.0, 100.0, o); });
    };
    EXPECT_EQ(rms(tunedRender<double>(sampler, 1000, 256, outside), 0, 1000), 0.0);
    EXPECT_EQ(rms(tunedRender<double>(synth, 1000, 256, outside), 0, 1000), 0.0);
    auto inside = [](auto& s, int pos, int n) {
        at(0, pos, n, [&](int o) { s.noteOnTuned(69, 1.0, 100.0, o); });
    };
    EXPECT_GT(rms(tunedRender<double>(sampler, 1000, 256, inside), 100, 1000), 0.4);
    EXPECT_GT(rms(tunedRender<double>(synth, 1000, 256, inside), 100, 1000), 0.4);
}


DSPARK_TEST(NoteTuning_published_event_capacity_is_usable_and_resets)
{
    auto check = []<typename Player>(Player& s) {
        static_assert(Player::kMaxEvents >= 256);
        for (int e = 0; e < Player::kMaxEvents - 1; ++e) s.setPitchBend(0.0, 0);
        s.noteOnTuned(69, 1.0, -37.5, 17);
        s.noteOnTuned(69, 0.0, 0.0, 18); // excess event must not corrupt the queue
        const auto x = tunedRender<double>(s, 256, 256, [](auto&, int, int) {});
        EXPECT_EQ(rms(x, 0, 17), 0.0);
        EXPECT_GT(rms(x, 64, 256), 0.1);
        EXPECT_EQ(s.getActiveVoiceCount(), 1);
        s.allNotesOff(0);
        const auto tail = tunedRender<double>(s, 48000, 256, [](auto&, int, int) {});
        EXPECT_EQ(rms(tail, 47000, 48000), 0.0);
        EXPECT_EQ(s.getActiveVoiceCount(), 0);
    };
    Synth synth;
    tunedSine(synth);
    Sampler<double> sampler;
    tunedSample(sampler);
    check(synth);
    check(sampler);
}
