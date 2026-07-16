// DSPark — Public conformance suite
// Copyright (c) 2026 Cristian Moresi — MIT License
//
// This is the PUBLIC quality gate that runs in CI on every platform:
//
//   [simd]    every SimdOps kernel against a scalar reference on the SIMD
//             path that is active for the build (SSE2/AVX/NEON/Wasm/scalar)
//   [core]    DenormalGuard flush + restore semantics on the active target
//             (real FTZ on x86/ARM, documented no-op on WebAssembly)
//   [smoke]   every effect: prepare → process → finite output → silent tail
//   [pdc]     null tests: latency-reporting processors cancel against a
//             delay-compensated dry path
//   [metrics] objective quality numbers (resampler SNR, EQ linear-phase
//             level, oscillator levels, filter DC rejection, ...)
//   [ebu]     EBU R128 conformance against the official EBU loudness test
//             set: integrated loudness (Tech 3341-2023 table 1), loudness
//             range (Tech 3342-2023 table 1) and true peak (Tech 3341 cases
//             15-23). Optional: set DSPARK_EBU_TEST_SET to the directory
//             containing the extracted official WAV files
//             (https://tech.ebu.ch/publications/ebu_loudness_test_set).
//             Skipped cleanly when the variable is not set.
//
// Zero dependencies, single translation unit, plain exit code (0 = pass).

#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "DSPark.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#ifndef DSPARK_NO_FILE_IO
#include <filesystem>
#endif

namespace {

int g_passed = 0;
int g_failed = 0;
int g_skipped = 0;

void check(bool condition, const char* section, const char* name, const char* detail = "")
{
    if (condition)
    {
        ++g_passed;
        std::printf("PASS  [%s] %s\n", section, name);
    }
    else
    {
        ++g_failed;
        std::printf("FAIL  [%s] %s %s\n", section, name, detail);
    }
}

void skip(const char* section, const char* name, const char* why)
{
    ++g_skipped;
    std::printf("SKIP  [%s] %s (%s)\n", section, name, why);
}

constexpr double kPiConf = 3.14159265358979323846;

// -----------------------------------------------------------------------------
// Shared helpers
// -----------------------------------------------------------------------------

struct StereoSignal
{
    std::vector<float> left, right;
};

StereoSignal makeSine(double freq, double sampleRate, int numSamples, float amp = 0.5f)
{
    StereoSignal s;
    s.left.resize(static_cast<size_t>(numSamples));
    s.right.resize(static_cast<size_t>(numSamples));
    for (int i = 0; i < numSamples; ++i)
    {
        const float v = amp * static_cast<float>(std::sin(2.0 * kPiConf * freq * i / sampleRate));
        s.left[static_cast<size_t>(i)] = v;
        s.right[static_cast<size_t>(i)] = v * 0.9f;
    }
    return s;
}

bool allFinite(const float* d, int n)
{
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(d[i])) return false;
    return true;
}

double rms(const float* d, int n)
{
    double acc = 0;
    for (int i = 0; i < n; ++i) acc += static_cast<double>(d[i]) * d[i];
    return std::sqrt(acc / std::max(1, n));
}

double toDb(double x) { return 20.0 * std::log10(std::max(x, 1e-12)); }

// Runs `process` over `blocks` blocks of sine, then `tailBlocks` of silence.
// Returns true if every output sample stayed finite and the tail decays.
bool smokeRun(const std::function<void(dspark::AudioBufferView<float>)>& process,
              double sampleRate = 48000.0, int blockSize = 512,
              int blocks = 40, int tailBlocks = 60, bool isGenerator = false)
{
    dspark::AudioBuffer<float> buf;
    buf.resize(2, blockSize);

    int n = 0;
    for (int b = 0; b < blocks; ++b)
    {
        for (int i = 0; i < blockSize; ++i, ++n)
        {
            const float v = 0.5f * static_cast<float>(std::sin(2.0 * kPiConf * 220.0 * n / sampleRate));
            buf.getChannel(0)[i] = v;
            buf.getChannel(1)[i] = v;
        }
        process(buf.toView());
        for (int ch = 0; ch < 2; ++ch)
            if (!allFinite(buf.getChannel(ch), blockSize)) return false;
    }

    double tailRms = 0.0;
    for (int b = 0; b < tailBlocks; ++b)
    {
        buf.clear();
        process(buf.toView());
        for (int ch = 0; ch < 2; ++ch)
            if (!allFinite(buf.getChannel(ch), blockSize)) return false;
        if (b == tailBlocks - 1)
            tailRms = rms(buf.getChannel(0), blockSize);
    }
    // Reverbs and long delays decay slowly; only demand the tail is not stuck
    // at signal level or blowing up. Generators keep producing on silence by
    // design, so only the finiteness checks apply to them.
    return isGenerator || tailRms < 0.25;
}

// -----------------------------------------------------------------------------
// [simd] — every SimdOps kernel against a scalar reference. Counts cover
// empty, sub-vector, vector-boundary and unrolled-loop sizes, so both the
// SIMD body and the scalar tail of every kernel run on the active path.
// -----------------------------------------------------------------------------

template <typename T>
void checkSimdKernels(bool& elementwise, bool& reductions, bool& cma, bool& ramps)
{
    // Deterministic LCG so every platform checks identical data.
    uint32_t seed = sizeof(T) == 4 ? 0x5EEDC0DEu : 0xC0DE5EEDu;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<T>(static_cast<int>((seed >> 16) & 0x7FFFu)) / static_cast<T>(16384)
             - static_cast<T>(1);
    };
    const double tolE = sizeof(T) == 4 ? 1e-5 : 1e-13; // element-wise (admits fused FMA)
    const double tolR = sizeof(T) == 4 ? 5e-4 : 1e-11; // accumulating kernels

    constexpr int kMax = 257;
    T srcA[kMax + 1], srcB[kMax + 1], dst[kMax + 1], ref[kMax + 1], out[kMax + 1], outRef[kMax + 1];

    for (int count : { 0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 33, 64, 65, 100, 257 })
    {
        for (int i = 0; i <= kMax; ++i)
        {
            srcA[i] = next(); srcB[i] = next(); dst[i] = next();
            ref[i] = dst[i];  out[i] = static_cast<T>(77);
        }

        // Element-wise kernels, chained so one pass covers all of them.
        dspark::simd::addWithGain(dst, srcA, static_cast<T>(0.7), count);
        dspark::simd::applyGain(dst, static_cast<T>(0.5), count);
        dspark::simd::add(dst, srcB, count);
        for (int i = 0; i < count; ++i)
            ref[i] = (ref[i] + srcA[i] * static_cast<T>(0.7)) * static_cast<T>(0.5) + srcB[i];
        for (int i = 0; i < count; ++i)
            if (std::abs(static_cast<double>(dst[i]) - static_cast<double>(ref[i])) > tolE) elementwise = false;
        if (dst[count] != ref[count]) elementwise = false; // past-the-end untouched

        dspark::simd::multiply(out, srcA, srcB, count);
        for (int i = 0; i < count; ++i)
            if (std::abs(static_cast<double>(out[i])
                       - static_cast<double>(srcA[i]) * static_cast<double>(srcB[i])) > tolE) elementwise = false;
        if (out[count] != static_cast<T>(77)) elementwise = false;
        dspark::simd::copyWithGain(out, srcA, static_cast<T>(-1.25), count);
        for (int i = 0; i < count; ++i)
            if (std::abs(static_cast<double>(out[i])
                       - static_cast<double>(srcA[i]) * -1.25) > tolE) elementwise = false;

        // Reductions. peakLevel is rounding-free, so the match is exact.
        T refPeak = static_cast<T>(0);
        double refDot = 0.0, refSos = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const T v = srcA[i] < static_cast<T>(0) ? -srcA[i] : srcA[i];
            if (v > refPeak) refPeak = v;
            refDot += static_cast<double>(srcA[i]) * static_cast<double>(srcB[i]);
            refSos += static_cast<double>(srcA[i]) * static_cast<double>(srcA[i]);
        }
        if (dspark::simd::peakLevel(srcA, count) != refPeak) reductions = false;
        if (std::abs(static_cast<double>(dspark::simd::dotProduct(srcA, srcB, count)) - refDot) > tolR)
            reductions = false;
        if (std::abs(static_cast<double>(dspark::simd::sumOfSquares(srcA, count)) - refSos) > tolR)
            reductions = false;

        // Complex multiply-accumulate over interleaved [re, im] spectra.
        const int bins = count < 129 ? count : 128;
        for (int i = 0; i <= kMax; ++i) { dst[i] = next(); ref[i] = dst[i]; }
        dspark::simd::complexMulAccum(dst, srcA, srcB, bins);
        for (int k = 0; k < bins; ++k)
        {
            const double re = static_cast<double>(ref[2 * k])
                            + static_cast<double>(srcA[2 * k]) * static_cast<double>(srcB[2 * k])
                            - static_cast<double>(srcA[2 * k + 1]) * static_cast<double>(srcB[2 * k + 1]);
            const double im = static_cast<double>(ref[2 * k + 1])
                            + static_cast<double>(srcA[2 * k]) * static_cast<double>(srcB[2 * k + 1])
                            + static_cast<double>(srcA[2 * k + 1]) * static_cast<double>(srcB[2 * k]);
            if (std::abs(static_cast<double>(dst[2 * k]) - re) > tolR)     cma = false;
            if (std::abs(static_cast<double>(dst[2 * k + 1]) - im) > tolR) cma = false;
        }
        if (2 * bins <= kMax && dst[2 * bins] != ref[2 * bins]) cma = false;

        // Linear gain ramps (block-contiguous convention).
        const T gs = static_cast<T>(0.25), ge = static_cast<T>(1.75);
        for (int i = 0; i <= kMax; ++i)
        {
            dst[i] = next(); ref[i] = dst[i];
            out[i] = next(); outRef[i] = out[i];
        }
        dspark::simd::applyGainRamp(dst, gs, ge, count);
        dspark::simd::addWithGainRamp(out, srcA, gs, ge, count);
        if (count > 0)
        {
            const T step = (ge - gs) / static_cast<T>(count);
            for (int i = 0; i < count; ++i)
            {
                const T g = gs + step * static_cast<T>(i);
                ref[i]    *= g;
                outRef[i] += srcA[i] * g;
            }
        }
        for (int i = 0; i < count; ++i)
        {
            if (std::abs(static_cast<double>(dst[i]) - static_cast<double>(ref[i])) > tolR)    ramps = false;
            if (std::abs(static_cast<double>(out[i]) - static_cast<double>(outRef[i])) > tolR) ramps = false;
        }
        if (dst[count] != ref[count] || out[count] != outRef[count]) ramps = false;
    }
}

template <typename T>
bool simdPeakLevelIgnoresNan()
{
    const T nan = std::numeric_limits<T>::quiet_NaN();
    T buf[33];
    for (int i = 0; i < 33; ++i)
        buf[i] = static_cast<T>(0.01) * static_cast<T>(i % 7) - static_cast<T>(0.02);
    buf[4]  = static_cast<T>(-1.5); // true peak, seen BEFORE the first NaN
    buf[6]  = nan;
    buf[19] = nan;
    buf[32] = nan;                  // scalar-tail NaN
    if (!(dspark::simd::peakLevel(buf, 33) == static_cast<T>(1.5))) return false;
    buf[0]  = nan;                  // NaN before any finite sample
    buf[10] = static_cast<T>(2.5);  // a larger peak AFTER a NaN must be seen
    return dspark::simd::peakLevel(buf, 33) == static_cast<T>(2.5);
}

void runSimdKernelTests()
{
    bool elementwise = true, reductions = true, cma = true, ramps = true;
    checkSimdKernels<float>(elementwise, reductions, cma, ramps);
    checkSimdKernels<double>(elementwise, reductions, cma, ramps);
    check(elementwise, "simd", "element-wise kernels match the scalar reference");
    check(reductions,  "simd", "reduction kernels match the scalar reference");
    check(cma,         "simd", "complex multiply-accumulate matches the scalar reference");
    check(ramps,       "simd", "gain ramp kernels match the scalar reference");
    check(simdPeakLevelIgnoresNan<float>() && simdPeakLevelIgnoresNan<double>(),
          "simd", "peakLevel ignores NaN samples");
}

// -----------------------------------------------------------------------------
// [core] — DenormalGuard flushes denormal results where the architecture has
// an FTZ mode, is a transparent no-op where none exists (e.g. WebAssembly),
// and always restores the previous FP state exactly.
// -----------------------------------------------------------------------------

float denormalProbe()
{
    // FLT_MIN * 0.5 is subnormal under IEEE arithmetic, exactly 0.0f under
    // FTZ. volatile forces the multiply to run under the current FP mode.
    volatile float smallest = std::numeric_limits<float>::min();
    volatile float half     = 0.5f;
    return smallest * half;
}

void runCoreGuardTests()
{
    const float before = denormalProbe();
    float inside = -1.0f;
    {
        dspark::DenormalGuard guard;
        inside = denormalProbe();
    }
    const float after = denormalProbe();

    if (dspark::DenormalGuard::isActive())
        check(inside == 0.0f, "core", "DenormalGuard flushes denormal results to zero");
    else
        check(inside == before, "core", "DenormalGuard is a transparent no-op on this target");
    check(after == before, "core", "DenormalGuard restores the previous FP state");
}

// -----------------------------------------------------------------------------
// [smoke] — every effect builds, runs, stays finite, and dies down
// -----------------------------------------------------------------------------

void runSmokeTests()
{
    const dspark::AudioSpec spec { 48000.0, 512, 2 };
    using V = dspark::AudioBufferView<float>;

    struct Case
    {
        const char* name;
        std::function<std::function<void(V)>()> make;
        bool isGenerator = false;
    };
    std::vector<Case> cases;

    cases.push_back({ "Gain", [&] {
        auto p = std::make_shared<dspark::Gain<float>>();
        p->prepare(spec); p->setGainDb(-3.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "FilterEngine", [&] {
        auto p = std::make_shared<dspark::FilterEngine<float>>();
        p->prepare(spec); p->setLowPass(4000.0f, 0.707f, 24);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Equalizer", [&] {
        auto p = std::make_shared<dspark::Equalizer<float>>();
        p->prepare(spec); p->setBand(0, 200.0f, 3.0f); p->setBand(1, 5000.0f, -2.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "EqualizerLinearPhase", [&] {
        auto p = std::make_shared<dspark::Equalizer<float>>();
        p->setFilterMode(dspark::Equalizer<float>::FilterMode::LinearPhase);
        p->prepare(spec); p->setBand(0, 1000.0f, 4.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Compressor", [&] {
        auto p = std::make_shared<dspark::Compressor<float>>();
        p->prepare(spec); p->setThreshold(-20.0f); p->setRatio(4.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Limiter", [&] {
        auto p = std::make_shared<dspark::Limiter<float>>();
        p->prepare(spec); p->setCeiling(-1.0f); p->setTruePeak(true);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "NoiseGate", [&] {
        auto p = std::make_shared<dspark::NoiseGate<float>>();
        p->prepare(spec); p->setThreshold(-40.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Expander", [&] {
        auto p = std::make_shared<dspark::Expander<float>>();
        p->prepare(spec); p->setThreshold(-45.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "DynamicEQ", [&] {
        auto p = std::make_shared<dspark::DynamicEQ<float>>();
        p->prepare(spec);
        dspark::DynamicEQ<float>::BandConfig cfg;
        cfg.frequency = 3000.0f; cfg.threshold = -30.0f; cfg.aboveRatio = 3.0f;
        p->setBand(0, cfg); p->setNumBands(1);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "MultibandCompressor", [&] {
        auto p = std::make_shared<dspark::MultibandCompressor<float>>();
        p->prepare(spec); p->setNumBands(3);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "TransientDesigner", [&] {
        auto p = std::make_shared<dspark::TransientDesigner<float>>();
        p->prepare(spec); p->setAttack(40.0f); p->setSustain(-20.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "DeEsser", [&] {
        auto p = std::make_shared<dspark::DeEsser<float>>();
        p->prepare(spec); p->setFrequency(7000.0f); p->setThreshold(-30.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "AlgorithmicReverb", [&] {
        auto p = std::make_shared<dspark::AlgorithmicReverb<float>>();
        p->prepare(spec); p->setType(dspark::AlgorithmicReverb<float>::Type::Hall);
        p->setMix(0.4f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Saturation", [&] {
        auto p = std::make_shared<dspark::Saturation<float>>();
        p->setOversampling(2); p->prepare(spec); p->setDrive(8.0f);
        p->setAlgorithm(dspark::Saturation<float>::Algorithm::Tape);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Clipper", [&] {
        auto p = std::make_shared<dspark::Clipper<float>>();
        p->setOversampling(4); p->prepare(spec);
        p->setMode(dspark::Clipper<float>::Mode::Analog); p->setInputGain(6.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Chorus", [&] {
        auto p = std::make_shared<dspark::Chorus<float>>();
        p->prepare(spec); p->setRate(0.8f); p->setMix(0.5f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Phaser", [&] {
        auto p = std::make_shared<dspark::Phaser<float>>();
        p->prepare(spec); p->setRate(0.5f); p->setStereoSpread(0.5f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Tremolo", [&] {
        auto p = std::make_shared<dspark::Tremolo<float>>();
        p->prepare(spec); p->setRate(5.0f); p->setDepth(0.7f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Vibrato", [&] {
        auto p = std::make_shared<dspark::Vibrato<float>>();
        p->prepare(spec); p->setRate(5.0f); p->setDepth(0.4f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "RingModulator", [&] {
        auto p = std::make_shared<dspark::RingModulator<float>>();
        p->prepare(spec); p->setFrequency(300.0f); p->setMix(0.8f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "FrequencyShifter", [&] {
        auto p = std::make_shared<dspark::FrequencyShifter<float>>();
        p->prepare(spec); p->setShift(50.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "PitchShifter", [&] {
        auto p = std::make_shared<dspark::PitchShifter<float>>();
        p->prepare(spec); p->setSemitones(4.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "TapeMachine", [&] {
        auto p = std::make_shared<dspark::TapeMachine<float>>();
        p->prepare(spec); p->setDrive(6.0f); p->setWowFlutter(0.3f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "TubePreamp", [&] {
        auto p = std::make_shared<dspark::TubePreamp<float>>();
        p->prepare(spec); p->setStages(2); p->setDrive(12.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "GranularProcessor", [&] {
        auto p = std::make_shared<dspark::GranularProcessor<float>>();
        p->prepare(spec); p->setDensity(30.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "SpectralDenoiser", [&] {
        auto p = std::make_shared<dspark::SpectralDenoiser<float>>();
        p->prepare(spec); p->setReduction(12.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});    cases.push_back({ "TransformerModel", [&] {
        auto p = std::make_shared<dspark::TransformerModel<float>>();
        p->prepare(spec); p->setDrive(9.0f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "Delay", [&] {
        auto p = std::make_shared<dspark::Delay<float>>();
        p->prepareMs(spec, 500.0);
        return std::function<void(V)>([p](V b) { p->processBlock(b, 120.0f, 0.4f, 6000.0f, 100.0f); });
    }});
    cases.push_back({ "Panner", [&] {
        auto p = std::make_shared<dspark::Panner<float>>();
        p->prepare(spec); p->setPan(0.4f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "StereoWidth", [&] {
        auto p = std::make_shared<dspark::StereoWidth<float>>();
        p->prepare(spec); p->setWidth(1.4f); p->setBassMono(true, 120.0);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "CrossfadeWrap", [&] {
        auto p = std::make_shared<dspark::Crossfade<float>>();
        p->setPosition(0.5f);
        return std::function<void(V)>([p](V b) {
            for (int ch = 0; ch < b.getNumChannels(); ++ch)
                p->process(b.getChannel(ch), b.getChannel(ch), b.getChannel(ch), b.getNumSamples());
        });
    }});
    cases.push_back({ "DCBlocker", [&] {
        auto p = std::make_shared<dspark::DCBlocker<float>>();
        p->prepare(48000.0, 2, 10.0); p->setOrder(4);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});
    cases.push_back({ "AutoGainWrap", [&] {
        auto p = std::make_shared<dspark::AutoGain<float>>();
        p->prepare(spec);
        return std::function<void(V)>([p](V b) { p->pushReference(b); p->compensate(b); });
    }});
    cases.push_back({ "NoiseGenerator", [&] {
        auto p = std::make_shared<dspark::NoiseGenerator<float>>();
        p->prepare(spec);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }, true });
    cases.push_back({ "MidSideRoundTrip", [&] {
        return std::function<void(V)>([](V b) {
            dspark::MidSide<float>::encode(b);
            dspark::MidSide<float>::decode(b);
        });
    }});
    cases.push_back({ "ConvolutionReverb", [&] {
        auto p = std::make_shared<dspark::Reverb<float>>();
        p->prepare(spec);
        // Small synthetic exponentially-decaying IR
        std::vector<float> ir(4800);
        for (size_t i = 0; i < ir.size(); ++i)
            ir[i] = static_cast<float>(std::exp(-3.0 * static_cast<double>(i) / 4800.0)
                  * std::sin(0.1 * static_cast<double>(i)));
        ir[0] = 1.0f;
        p->loadIR(ir.data(), static_cast<int>(ir.size()), 48000.0);
        p->setMix(0.3f);
        return std::function<void(V)>([p](V b) { p->processBlock(b); });
    }});

    for (auto& c : cases)
    {
        auto proc = c.make();
        check(smokeRun(proc, 48000.0, 512, 40, 60, c.isGenerator), "smoke", c.name);
    }
}

// -----------------------------------------------------------------------------
// [pdc] — latency-compensated null tests
// -----------------------------------------------------------------------------

// Processes `signal` through `process` and measures the residual (in dB,
// relative to the dry RMS) against the same signal delayed by `latency`.
double residualDbVsDelayedDry(const std::function<void(dspark::AudioBufferView<float>)>& process,
                              int latency, double sampleRate = 48000.0,
                              int blockSize = 512, int blocks = 60)
{
    const int total = blockSize * blocks;
    StereoSignal dry = makeSine(997.0, sampleRate, total, 0.25);

    dspark::AudioBuffer<float> buf;
    buf.resize(2, blockSize);

    std::vector<float> out(static_cast<size_t>(total));
    for (int b = 0; b < blocks; ++b)
    {
        for (int i = 0; i < blockSize; ++i)
        {
            buf.getChannel(0)[i] = dry.left[static_cast<size_t>(b * blockSize + i)];
            buf.getChannel(1)[i] = dry.right[static_cast<size_t>(b * blockSize + i)];
        }
        process(buf.toView());
        std::memcpy(out.data() + static_cast<size_t>(b) * blockSize,
                    buf.getChannel(0), static_cast<size_t>(blockSize) * sizeof(float));
    }

    // Compare steady-state region, skipping warm-up
    const int start = std::max(latency + blockSize * 4, total / 4);
    double err = 0, ref = 0;
    for (int i = start; i < total; ++i)
    {
        const double d = dry.left[static_cast<size_t>(i - latency)];
        const double e = out[static_cast<size_t>(i)] - d;
        err += e * e;
        ref += d * d;
    }
    return 10.0 * std::log10(std::max(err, 1e-30) / std::max(ref, 1e-30));
}

void runPdcNullTests()
{
    const dspark::AudioSpec spec { 48000.0, 512, 2 };

    {
        // Flat linear-phase EQ must null against dry delayed by getLatency().
        auto eq = std::make_shared<dspark::Equalizer<float>>();
        eq->setFilterMode(dspark::Equalizer<float>::FilterMode::LinearPhase);
        eq->prepare(spec);
        const double res = residualDbVsDelayedDry(
            [eq](dspark::AudioBufferView<float> b) { eq->processBlock(b); }, eq->getLatency());
        char d[64]; std::snprintf(d, sizeof(d), "(residual %.1f dB)", res);
        check(res < -60.0, "pdc", "EqualizerLinearPhase flat null", d);
    }
    {
        // Limiter far below threshold must be a pure delay.
        auto lim = std::make_shared<dspark::Limiter<float>>();
        lim->prepare(48000.0, 2, 2.0);
        lim->setCeiling(0.0f);
        const double res = residualDbVsDelayedDry(
            [lim](dspark::AudioBufferView<float> b) { lim->processBlock(b); }, lim->getLatency());
        char d[64]; std::snprintf(d, sizeof(d), "(residual %.1f dB)", res);
        check(res < -60.0, "pdc", "Limiter transparent below ceiling", d);
    }
    {
        // Oversampling round-trip nulls against dry delayed by getLatency().
        auto os = std::make_shared<dspark::Oversampling<float>>(4, dspark::Oversampling<float>::Quality::High);
        os->prepare(dspark::AudioSpec { 48000.0, 512, 2 });
        const double res = residualDbVsDelayedDry(
            [os](dspark::AudioBufferView<float> b) {
                (void)os->upsample(b);
                os->downsample(b);
            }, os->getLatency());
        char d[64]; std::snprintf(d, sizeof(d), "(residual %.1f dB)", res);
        check(res < -60.0, "pdc", "Oversampling 4x round-trip null", d);
    }
    {
        // Convolver with a unit impulse IR is a pure block delay.
        std::vector<float> ir(1, 1.0f);
        auto conv = std::make_shared<dspark::Convolver<float>>();
        conv->prepare(512, ir.data(), 1);
        const double res = residualDbVsDelayedDry(
            [conv](dspark::AudioBufferView<float> b) {
                conv->processInPlace(b.getChannel(0), b.getNumSamples());
            }, conv->getLatency());
        char d[64]; std::snprintf(d, sizeof(d), "(residual %.1f dB)", res);
        check(res < -80.0, "pdc", "Convolver delta-IR null", d);
    }
    {
        // ZeroLatencyConvolver with a delta IR must null with NO delay at all,
        // even with the IR long enough to engage all three partition levels.
        std::vector<float> ir(6000, 0.0f);
        ir[0] = 1.0f;
        auto conv = std::make_shared<dspark::ZeroLatencyConvolver<float>>();
        conv->prepare(ir.data(), static_cast<int>(ir.size()));
        const double res = residualDbVsDelayedDry(
            [conv](dspark::AudioBufferView<float> b) {
                conv->processInPlace(b.getChannel(0), b.getNumSamples());
            }, conv->getLatency());   // == 0
        char d[64]; std::snprintf(d, sizeof(d), "(residual %.1f dB)", res);
        check(res < -80.0, "pdc", "ZeroLatencyConvolver zero-latency delta null", d);
    }
}

// -----------------------------------------------------------------------------
// [metrics] — objective quality numbers
// -----------------------------------------------------------------------------

void runMetricTests()
{
    {
        // Resampler 44.1k -> 48k sine SNR (offline path).
        dspark::Resampler<double> rs;
        rs.prepare(44100.0, 48000.0, dspark::Resampler<double>::Quality::High);
        const int n = 1 << 13;
        std::vector<double> in(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) in[static_cast<size_t>(i)] = std::sin(2.0 * kPiConf * 1000.0 * i / 44100.0);
        auto out = rs.process(in.data(), n);

        double err = 0, ref = 0;
        for (size_t k = 200; k + 200 < out.size(); ++k)
        {
            const double srcPos = static_cast<double>(k) * 44100.0 / 48000.0;
            const double ideal = std::sin(2.0 * kPiConf * 1000.0 * srcPos / 44100.0);
            const double e = out[k] - ideal;
            err += e * e; ref += ideal * ideal;
        }
        const double snr = 10.0 * std::log10(ref / std::max(err, 1e-30));
        char d[64]; std::snprintf(d, sizeof(d), "(SNR %.1f dB)", snr);
        check(snr > 90.0, "metrics", "Resampler High sine SNR > 90 dB", d);
    }
    {
        // Oscillator waveform levels must match within 1 dB.
        dspark::Oscillator<float> osc;
        osc.prepare(48000.0);
        osc.setFrequency(220.0f);
        double peaks[4] = {};
        const dspark::Oscillator<float>::Waveform shapes[4] = {
            dspark::Oscillator<float>::Waveform::Sine,
            dspark::Oscillator<float>::Waveform::Saw,
            dspark::Oscillator<float>::Waveform::Square,
            dspark::Oscillator<float>::Waveform::Triangle };
        for (int w = 0; w < 4; ++w)
        {
            osc.setWaveform(shapes[w]);
            osc.reset();
            double pk = 0;
            for (int i = 0; i < 48000; ++i)
                pk = std::max(pk, std::abs(static_cast<double>(osc.getNextSample())));
            peaks[w] = pk;
        }
        const double triDb = toDb(peaks[3]);
        char d[96]; std::snprintf(d, sizeof(d), "(sine %.2f saw %.2f sq %.2f tri %.2f)",
                                  peaks[0], peaks[1], peaks[2], peaks[3]);
        check(triDb > -1.0 && triDb < 1.0, "metrics", "Oscillator triangle level ~0 dBFS", d);
    }
    {
        // Ladder HP24 rejects DC at high resonance.
        dspark::LadderFilter<float> lf;
        lf.prepare(dspark::AudioSpec { 48000.0, 512, 1 });
        lf.setMode(dspark::LadderFilter<float>::Mode::HP24);
        lf.setCutoff(1000.0f);
        lf.setResonance(0.5f);
        dspark::AudioBuffer<float> b; b.resize(1, 512);
        float last = 1.0f;
        for (int blk = 0; blk < 200; ++blk)
        {
            for (int i = 0; i < 512; ++i) b.getChannel(0)[i] = 1.0f;
            lf.processBlock(b.toView());
            last = b.getChannel(0)[511];
        }
        char d[64]; std::snprintf(d, sizeof(d), "(DC out %.5f)", static_cast<double>(last));
        check(std::abs(last) < 1e-3, "metrics", "Ladder HP24 DC rejection @ res 0.5", d);
    }
    {
        // Clipper Analog mode: unity gain for small signals.
        dspark::Clipper<float> cl;
        cl.prepare(dspark::AudioSpec { 48000.0, 512, 2 });
        cl.setMode(dspark::Clipper<float>::Mode::Analog);
        dspark::AudioBuffer<float> b; b.resize(2, 512);
        double inR = 0, outR = 0;
        for (int blk = 0; blk < 40; ++blk)
        {
            for (int i = 0; i < 512; ++i)
            {
                const float v = 0.05f * static_cast<float>(std::sin(2.0 * kPiConf * 1000.0 * (blk * 512 + i) / 48000.0));
                b.getChannel(0)[i] = v; b.getChannel(1)[i] = v;
            }
            if (blk > 4) inR += rms(b.getChannel(0), 512);
            cl.processBlock(b.toView());
            if (blk > 4) outR += rms(b.getChannel(0), 512);
        }
        const double gainDb = toDb(outR / std::max(inR, 1e-30));
        char d[64]; std::snprintf(d, sizeof(d), "(gain %+.2f dB)", gainDb);
        check(std::abs(gainDb) < 0.2, "metrics", "Clipper Analog small-signal unity gain", d);
    }
    {
        // PitchShifter +7 st on a 440 Hz sine must land on 659.255 Hz.
        dspark::PitchShifter<float> ps;
        ps.prepare(dspark::AudioSpec { 48000.0, 512, 2 });
        ps.setSemitones(7.0f);

        const int total = (3 * 48000 / 512) * 512;
        std::vector<float> outSig(static_cast<size_t>(total));
        dspark::AudioBuffer<float> b; b.resize(2, 512);
        int n = 0;
        for (int blk = 0; blk < total / 512; ++blk)
        {
            for (int i = 0; i < 512; ++i, ++n)
            {
                const float v = 0.5f * static_cast<float>(std::sin(2.0 * kPiConf * 440.0 * n / 48000.0));
                b.getChannel(0)[i] = v; b.getChannel(1)[i] = v;
            }
            ps.processBlock(b.toView());
            std::memcpy(outSig.data() + static_cast<size_t>(blk) * 512, b.getChannel(0),
                        512 * sizeof(float));
        }

        // Dominant frequency of the last 16384 samples via zero-padded FFT
        // + parabolic interpolation on the log-magnitude peak.
        constexpr size_t kN = 16384;
        dspark::FFTReal<double> fft(kN);
        std::vector<double> t(kN), f(kN + 2);
        for (size_t i = 0; i < kN; ++i)
        {
            const double w = 0.5 - 0.5 * std::cos(2.0 * kPiConf * static_cast<double>(i) / (kN - 1));
            t[i] = static_cast<double>(outSig[outSig.size() - kN + i]) * w;
        }
        fft.forward(t.data(), f.data());
        size_t peak = 1;
        double pP = 0;
        std::vector<double> p(kN / 2 + 1);
        for (size_t k = 0; k <= kN / 2; ++k)
        {
            p[k] = f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1];
            if (k > 4 && p[k] > pP) { pP = p[k]; peak = k; }
        }
        const double l0 = 10.0 * std::log10(p[peak - 1] + 1e-30);
        const double l1 = 10.0 * std::log10(p[peak] + 1e-30);
        const double l2 = 10.0 * std::log10(p[peak + 1] + 1e-30);
        const double den = l0 - 2.0 * l1 + l2;
        const double d0 = (std::abs(den) > 1e-12) ? 0.5 * (l0 - l2) / den : 0.0;
        const double got = (static_cast<double>(peak) + d0) * 48000.0 / static_cast<double>(kN);
        const double expected = 440.0 * std::exp2(7.0 / 12.0);
        const double cents = 1200.0 * std::log2(got / expected);
        char d[96]; std::snprintf(d, sizeof(d), "(got %.3f Hz, expected %.3f, %+.2f cents)",
                                  got, expected, cents);
        check(std::abs(cents) < 2.0, "metrics", "PitchShifter +7 st frequency exact", d);
    }
    {
        // TapeMachine signature: odd-dominant harmonic distortion (H3 >> H2)
        // growing with drive, at roughly constant fundamental level.
        dspark::TapeMachine<float> tape;
        tape.prepare(dspark::AudioSpec { 48000.0, 512, 2 });
        tape.setDrive(6.0f);
        tape.setLossEffects(0.0f);
        tape.setHeadBump(0.0f);
        tape.setWowFlutter(0.0f);

        const int total = (48000 / 512) * 512;
        std::vector<float> outSig(static_cast<size_t>(total));
        dspark::AudioBuffer<float> b; b.resize(2, 512);
        int n = 0;
        for (int blk = 0; blk < total / 512; ++blk)
        {
            for (int i = 0; i < 512; ++i, ++n)
            {
                const float v = 0.5f * static_cast<float>(std::sin(2.0 * kPiConf * 1000.0 * n / 48000.0));
                b.getChannel(0)[i] = v; b.getChannel(1)[i] = v;
            }
            tape.processBlock(b.toView());
            std::memcpy(outSig.data() + static_cast<size_t>(blk) * 512, b.getChannel(0),
                        512 * sizeof(float));
        }
        auto mag = [&](double freq) {
            const size_t from = outSig.size() / 2;
            const size_t cnt = outSig.size() - from;
            double re = 0, im = 0;
            for (size_t i = 0; i < cnt; ++i)
            {
                const double ph = 2.0 * kPiConf * freq * static_cast<double>(i) / 48000.0;
                re += static_cast<double>(outSig[from + i]) * std::cos(ph);
                im += static_cast<double>(outSig[from + i]) * std::sin(ph);
            }
            return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(cnt);
        };
        const double h1 = mag(1000.0), h2 = mag(2000.0), h3 = mag(3000.0);
        char d[96];
        std::snprintf(d, sizeof(d), "(H1 %.2f, H2 %.1f dB, H3 %.1f dB)",
                      h1, 20.0 * std::log10(h2 / h1 + 1e-12), 20.0 * std::log10(h3 / h1 + 1e-12));
        check(h3 > 3.0 * h2 && h3 / h1 > 0.005 && h3 / h1 < 0.2 && h1 > 0.25,
              "metrics", "TapeMachine odd-dominant tape signature", d);
    }
    {
        // WDF R-type FMV tone stack must match the bilinear transform of the
        // published symbolic transfer function (Yeh & Smith, DAFx-06) at the
        // reference setting. Validates the MNA-derived scattering end to end.
        const double fs = 48000.0;
        dspark::wdf::ToneStackFMV<double> stack(0.01, 1e10);
        stack.prepare(fs);
        const double t = 0.5, l = 0.5, m = 0.5;
        stack.setControls(t, std::sqrt(l), m);

        const double C1 = 0.25e-9, C2 = 20e-9, C3 = 20e-9;
        const double R1 = 250e3, R2 = 1e6, R3 = 25e3, R4 = 56e3;
        constexpr double kRmin = 0.5;
        const double te = (t * R1 + kRmin) / (R1 + 2.0 * kRmin);
        const double le = (l * R2 + kRmin) / R2;
        const double me = (m * R3 + kRmin) / (R3 + 2.0 * kRmin);
        const double R3sq = R3 * R3;

        const double b1 = te * C1 * R1 + me * C3 * R3 + le * (C1 * R2 + C2 * R2) + (C1 * R3 + C2 * R3);
        const double b2 = te * (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4)
            - me * me * (C1 * C3 * R3sq + C2 * C3 * R3sq)
            + me * (C1 * C3 * R1 * R3 + C1 * C3 * R3sq + C2 * C3 * R3sq)
            + le * (C1 * C2 * R1 * R2 + C1 * C2 * R2 * R4 + C1 * C3 * R2 * R4)
            + le * me * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
            + (C1 * C2 * R1 * R3 + C1 * C2 * R3 * R4 + C1 * C3 * R3 * R4);
        const double b3 = le * me * (C1 * C2 * C3 * R1 * R2 * R3 + C1 * C2 * C3 * R2 * R3 * R4)
            - me * me * (C1 * C2 * C3 * R1 * R3sq + C1 * C2 * C3 * R3sq * R4)
            + me * (C1 * C2 * C3 * R1 * R3sq + C1 * C2 * C3 * R3sq * R4)
            + te * C1 * C2 * C3 * R1 * R3 * R4 - te * me * C1 * C2 * C3 * R1 * R3 * R4
            + te * le * C1 * C2 * C3 * R1 * R2 * R4;
        const double a1 = (C1 * R1 + C1 * R3 + C2 * R3 + C2 * R4 + C3 * R4)
            + me * C3 * R3 + le * (C1 * R2 + C2 * R2);
        const double a2 = me * (C1 * C3 * R1 * R3 - C2 * C3 * R3 * R4 + C1 * C3 * R3sq + C2 * C3 * R3sq)
            + le * me * (C1 * C3 * R2 * R3 + C2 * C3 * R2 * R3)
            - me * me * (C1 * C3 * R3sq + C2 * C3 * R3sq)
            + le * (C1 * C2 * R2 * R4 + C1 * C2 * R1 * R2 + C1 * C3 * R2 * R4 + C2 * C3 * R2 * R4)
            + (C1 * C2 * R1 * R4 + C1 * C3 * R1 * R4 + C1 * C2 * R3 * R4
               + C1 * C2 * R1 * R3 + C1 * C3 * R3 * R4 + C2 * C3 * R3 * R4);
        const double a3 = le * me * (C1 * C2 * C3 * R1 * R2 * R3 + C1 * C2 * C3 * R2 * R3 * R4)
            - me * me * (C1 * C2 * C3 * R1 * R3sq + C1 * C2 * C3 * R3sq * R4)
            + me * (C1 * C2 * C3 * R3sq * R4 + C1 * C2 * C3 * R1 * R3sq
                    - C1 * C2 * C3 * R1 * R3 * R4)
            + le * C1 * C2 * C3 * R1 * R2 * R4 + C1 * C2 * C3 * R1 * R3 * R4;

        const double c = 2.0 * fs, c2 = c * c, c3 = c2 * c;
        double B[4] = { -b1 * c - b2 * c2 - b3 * c3, -b1 * c + b2 * c2 + 3.0 * b3 * c3,
                         b1 * c + b2 * c2 - 3.0 * b3 * c3, b1 * c - b2 * c2 + b3 * c3 };
        double A[4] = { -1.0 - a1 * c - a2 * c2 - a3 * c3, -3.0 - a1 * c + a2 * c2 + 3.0 * a3 * c3,
                        -3.0 + a1 * c + a2 * c2 - 3.0 * a3 * c3, -1.0 + a1 * c - a2 * c2 + a3 * c3 };
        for (int i = 3; i >= 0; --i) { B[i] /= A[0]; A[i] /= A[0]; }

        double xs[4] = {}, ys[4] = {};
        uint32_t rng = 0xBEEF5EEDu;
        double err = 0, pw = 0;
        for (int i = 0; i < 24000; ++i)
        {
            rng = rng * 1664525u + 1013904223u;
            const double x = static_cast<double>(rng >> 8) / 8388608.0 - 1.0;
            const double yw = stack.processSample(x);
            xs[3] = xs[2]; xs[2] = xs[1]; xs[1] = xs[0]; xs[0] = x;
            const double yr = B[0] * xs[0] + B[1] * xs[1] + B[2] * xs[2] + B[3] * xs[3]
                            - A[1] * ys[0] - A[2] * ys[1] - A[3] * ys[2];
            ys[2] = ys[1]; ys[1] = ys[0]; ys[0] = yr;
            if (i > 2400) { err += (yw - yr) * (yw - yr); pw += yr * yr; }
        }
        const double db = 10.0 * std::log10((err + 1e-300) / (pw + 1e-300));
        char d[64]; std::snprintf(d, sizeof(d), "(residual %.1f dB)", db);
        check(db < -80.0, "metrics", "WDF R-type FMV stack matches Yeh DAFx-06 TF", d);
    }
    {
        // WDF diode clipper solves the node equation (vin - v)/R = 2 Is sinh(v/nVt)
        // to the same answer as an independent bisection — physical-model gate.
        const double R = 2200.0, Is = 2.52e-9, nVt = 1.752 * 0.02585;
        dspark::wdf::ResistiveVoltageSource<double> vsrc { R };
        dspark::wdf::DiodePairRoot<double, decltype(vsrc)> clip { vsrc, Is, nVt };
        clip.prepare(48000.0);

        double maxErr = 0;
        for (double vin = -10.0; vin <= 10.0; vin += 0.25)
        {
            vsrc.setVoltage(vin);
            clip.process();
            double lo = std::min(0.0, vin), hi = std::max(0.0, vin);
            for (int it = 0; it < 200; ++it)
            {
                const double mid = 0.5 * (lo + hi);
                const double f = (vin - mid) / R - 2.0 * Is * std::sinh(mid / nVt);
                if (f > 0.0) lo = mid; else hi = mid;
            }
            maxErr = std::max(maxErr, std::abs(static_cast<double>(clip.getVoltage()) - 0.5 * (lo + hi)));
        }
        char d[64]; std::snprintf(d, sizeof(d), "(max err %.2e V)", maxErr);
        check(maxErr < 1e-9, "metrics", "WDF diode clipper matches node equation", d);
    }
    {
        // FFT round trip exactness.
        dspark::FFTReal<double> fft(2048);
        std::vector<double> t(2048), f(2050), t2(2048);
        for (int i = 0; i < 2048; ++i) t[static_cast<size_t>(i)] = std::sin(0.05 * i) + 0.3 * std::cos(0.31 * i);
        fft.forward(t.data(), f.data());
        fft.inverse(f.data(), t2.data());
        double err = 0;
        for (int i = 0; i < 2048; ++i) err = std::max(err, std::abs(t[static_cast<size_t>(i)] - t2[static_cast<size_t>(i)]));
        char d[64]; std::snprintf(d, sizeof(d), "(max err %.2e)", err);
        check(err < 1e-12, "metrics", "FFT exact round trip", d);
    }
}

// -----------------------------------------------------------------------------
// [ebu] — official EBU R128 test set (optional)
// -----------------------------------------------------------------------------

#ifndef DSPARK_NO_FILE_IO

std::string toLowerAscii(std::string s)
{
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// True when `nameLower` contains `tokenLower` not immediately followed by a
// digit — "3341-1" matches "seq-3341-1-16bit.wav" but not "seq-3341-10-...".
bool containsCaseToken(const std::string& nameLower, const std::string& tokenLower)
{
    size_t pos = 0;
    while ((pos = nameLower.find(tokenLower, pos)) != std::string::npos)
    {
        const size_t end = pos + tokenLower.size();
        if (end >= nameLower.size()
            || std::isdigit(static_cast<unsigned char>(nameLower[end])) == 0)
            return true;
        ++pos;
    }
    return false;
}

// Recursively locates a WAV under `dir` matching `token` (or `altToken`).
// The official set's filenames are too inconsistent to predict (-v02
// revisions, combined programme files like seq-3341-7_seq-3342-5, the year
// in seq-3341-2011-8, even double ".wav.wav"), so scan and match by token.
std::string findEbuFile(const std::string& dir, const char* token,
                        const char* altToken = nullptr)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::recursive_directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
    const fs::recursive_directory_iterator end;
    while (!ec && it != end)
    {
        std::error_code fec;
        if (it->is_regular_file(fec))
        {
            const std::string nameLower = toLowerAscii(it->path().filename().string());
            if (nameLower.size() > 4 && nameLower.compare(nameLower.size() - 4, 4, ".wav") == 0)
            {
                if (containsCaseToken(nameLower, toLowerAscii(token))
                    || (altToken != nullptr && containsCaseToken(nameLower, toLowerAscii(altToken))))
                    return it->path().string();
            }
        }
        it.increment(ec);
    }
    return {};
}

struct EbuMeasurement
{
    double integratedLufs = 0;
    double lra = 0;
    double truePeakDb = 0;
};

// Streams a whole WAV through LoudnessMeter, then feeds 2 s of silence so the
// final short-term windows drain — the Tech 3342 reference implementation
// requires >= 1.5 s of trailing silence before the LRA readout. The silence
// cannot bias any measure: it falls under the -70 LUFS absolute gate.
bool measureFile(const std::string& path, EbuMeasurement& out)
{
    dspark::WavFile wav;
    if (!wav.openRead(path)) return false;
    const auto info = wav.getInfo();
    if (info.numChannels < 1 || info.numChannels > 2) { wav.close(); return false; }

    dspark::LoudnessMeter<float> meter;
    meter.prepare(info.sampleRate, static_cast<int>(info.numChannels));

    constexpr int kBlock = 8192;
    dspark::AudioBuffer<float> buf;
    buf.resize(static_cast<int>(info.numChannels), kBlock);

    int64_t remaining = info.numSamples;
    int64_t offset = 0;
    while (remaining > 0)
    {
        const int n = static_cast<int>(std::min<int64_t>(remaining, kBlock));
        auto view = buf.toView().getSubView(0, n);
        if (!wav.readSamples(view, offset, n)) break;
        if (info.numChannels == 2)
            meter.process(view.getChannel(0), view.getChannel(1), n);
        else
            meter.process(view.getChannel(0), n);
        offset += n;
        remaining -= n;
    }
    wav.close();

    buf.clear();
    int64_t flush = static_cast<int64_t>(info.sampleRate * 2.0);
    while (flush > 0)
    {
        const int n = static_cast<int>(std::min<int64_t>(flush, kBlock));
        if (info.numChannels == 2)
            meter.process(buf.getChannel(0), buf.getChannel(1), n);
        else
            meter.process(buf.getChannel(0), n);
        flush -= n;
    }

    out.integratedLufs = static_cast<double>(meter.getIntegratedLUFS());
    out.lra            = static_cast<double>(meter.getLoudnessRange());
    out.truePeakDb     = static_cast<double>(meter.getTruePeakDb());
    return true;
}

void runEbuTests()
{
    const char* env = std::getenv("DSPARK_EBU_TEST_SET");
    if (env == nullptr || env[0] == '\0')
    {
        skip("ebu", "EBU R128 conformance", "DSPARK_EBU_TEST_SET not set");
        return;
    }
    const std::string dir = env;

    enum class Kind { Integrated, Lra, TruePeak };
    struct Case
    {
        const char* name;
        const char* token;
        const char* altToken;
        Kind        kind;
        double      expected;
    };

    // Expected values from EBU Tech 3341-2023 table 1 (integrated loudness
    // +/-0.1 LU, max true peak +0.2/-0.4 dB) and EBU Tech 3342-2023 table 1
    // (LRA +/-1 LU). Case 3341-6 is 5.0-channel: not covered by the
    // mono/stereo meter, so it is reported as a skip via the channel check.
    const Case cases[] = {
        { "3341-1 integrated -23 LUFS",   "3341-1", nullptr,       Kind::Integrated, -23.0 },
        { "3341-2 integrated -33 LUFS",   "3341-2", nullptr,       Kind::Integrated, -33.0 },
        { "3341-3 integrated -23 LUFS",   "3341-3", nullptr,       Kind::Integrated, -23.0 },
        { "3341-4 integrated -23 LUFS",   "3341-4", nullptr,       Kind::Integrated, -23.0 },
        { "3341-5 integrated -23 LUFS",   "3341-5", nullptr,       Kind::Integrated, -23.0 },
        { "3341-7 integrated -23 LUFS",   "3341-7", nullptr,       Kind::Integrated, -23.0 },
        { "3341-8 integrated -23 LUFS",   "3341-8", "3341-2011-8", Kind::Integrated, -23.0 },

        { "3342-1 LRA 10 LU",             "3342-1", nullptr,       Kind::Lra,        10.0 },
        { "3342-2 LRA 5 LU",              "3342-2", nullptr,       Kind::Lra,         5.0 },
        { "3342-3 LRA 20 LU",             "3342-3", nullptr,       Kind::Lra,        20.0 },
        { "3342-4 LRA 15 LU",             "3342-4", nullptr,       Kind::Lra,        15.0 },
        { "3342-5 LRA 5 LU",              "3342-5", nullptr,       Kind::Lra,         5.0 },
        { "3342-6 LRA 15 LU",             "3342-6", nullptr,       Kind::Lra,        15.0 },

        { "3341-15 true peak -6 dBTP",    "3341-15", nullptr,      Kind::TruePeak,   -6.0 },
        { "3341-16 true peak -6 dBTP",    "3341-16", nullptr,      Kind::TruePeak,   -6.0 },
        { "3341-17 true peak -6 dBTP",    "3341-17", nullptr,      Kind::TruePeak,   -6.0 },
        { "3341-18 true peak -6 dBTP",    "3341-18", nullptr,      Kind::TruePeak,   -6.0 },
        { "3341-19 true peak +3 dBTP",    "3341-19", nullptr,      Kind::TruePeak,    3.0 },
        { "3341-20 true peak 0 dBTP",     "3341-20", nullptr,      Kind::TruePeak,    0.0 },
        { "3341-21 true peak 0 dBTP",     "3341-21", nullptr,      Kind::TruePeak,    0.0 },
        { "3341-22 true peak 0 dBTP",     "3341-22", nullptr,      Kind::TruePeak,    0.0 },
        { "3341-23 true peak 0 dBTP",     "3341-23", nullptr,      Kind::TruePeak,    0.0 },
    };

    for (const auto& c : cases)
    {
        const std::string path = findEbuFile(dir, c.token, c.altToken);
        if (path.empty())
        {
            skip("ebu", c.name, "file not found");
            continue;
        }
        EbuMeasurement m;
        if (!measureFile(path, m))
        {
            skip("ebu", c.name, "channel layout not covered by mono/stereo meter");
            continue;
        }

        bool pass = false;
        double got = 0;
        const char* unit = "";
        switch (c.kind)
        {
        case Kind::Integrated:
            got = m.integratedLufs; unit = "LUFS";
            pass = std::abs(got - c.expected) <= 0.1;
            break;
        case Kind::Lra:
            got = m.lra; unit = "LU";
            pass = std::abs(got - c.expected) <= 1.0;
            break;
        case Kind::TruePeak:
            got = m.truePeakDb; unit = "dBTP";
            pass = got >= c.expected - 0.4 && got <= c.expected + 0.2;
            break;
        }

        char d[96];
        std::snprintf(d, sizeof(d), "(got %.2f %s, expected %.1f)", got, unit, c.expected);
        check(pass, "ebu", c.name, d);
    }
}

#else
void runEbuTests() { skip("ebu", "EBU R128 conformance", "built with DSPARK_NO_FILE_IO"); }
#endif // DSPARK_NO_FILE_IO

// -----------------------------------------------------------------------------
// [table] — public per-processor metrics table (KPI K3): THD+N, noise floor,
// spurious/aliasing floor and latency at documented settings, in Markdown.
// Run with:  dspark_conformance --metrics docs/metrics.md
// -----------------------------------------------------------------------------

struct MetricsCase
{
    const char* name;
    const char* settings;   ///< Human-readable settings shown in the table.
    const char* note;       ///< Caveat for time-based/modulated processors.
    std::function<void(dspark::AudioBufferView<float>)> process;
    std::function<int()> latency;   ///< Null when the processor is zero-latency.
    bool shifted = false;   ///< True when the output spectrum is displaced from
                            ///< the input (ring mod, shifters): THD+N/spurious
                            ///< against the input fundamental are meaningless.
};

/// Magnitude spectrum (Blackman-Harris, 32768-point) of channel 0 after a
/// warmup long enough to flush every latency/modulation transient.
struct CapturedSpectrum
{
    std::vector<double> mag;   ///< N/2+1 bin magnitudes.
    double binHz = 1.0;
    double outRms = 0.0;
};

CapturedSpectrum captureSpectrum(const std::function<void(dspark::AudioBufferView<float>)>& process,
                                 double freq, float amp)
{
    constexpr int kFft = 32768, kBlock = 512, kWarmBlocks = 80;   // ~0.85 s warmup
    dspark::AudioBuffer<float> buf;
    buf.resize(2, kBlock);
    auto view = buf.toView();

    // Silence purge: processors with long memory (granular rings, reverb
    // tails) must not carry the PREVIOUS measurement tone into this capture.
    for (int b = 0; b < 280; ++b)   // ~3 s
    {
        for (int i = 0; i < kBlock; ++i)
        {
            view.getChannel(0)[i] = 0.0f;
            view.getChannel(1)[i] = 0.0f;
        }
        process(view);
    }

    std::vector<float> cap;
    cap.reserve(kFft);
    int n = 0;
    const int total = kWarmBlocks * kBlock + kFft;
    while (n < total)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            const float s = amp * static_cast<float>(
                std::sin(2.0 * kPiConf * freq * (n + i) / 48000.0));
            view.getChannel(0)[i] = s;
            view.getChannel(1)[i] = s;
        }
        process(view);
        for (int i = 0; i < kBlock && n + i >= kWarmBlocks * kBlock
                        && static_cast<int>(cap.size()) < kFft; ++i)
            cap.push_back(view.getChannel(0)[i]);
        n += kBlock;
    }

    std::vector<double> win(kFft), t(kFft), f(kFft + 2);
    dspark::WindowFunctions<double>::blackmanHarris(win.data(), kFft, false);
    double acc = 0.0;
    for (int i = 0; i < kFft; ++i)
    {
        acc += static_cast<double>(cap[static_cast<size_t>(i)]) * cap[static_cast<size_t>(i)];
        t[static_cast<size_t>(i)] = static_cast<double>(cap[static_cast<size_t>(i)])
                                  * win[static_cast<size_t>(i)];
    }
    dspark::FFTReal<double> fft(kFft);
    fft.forward(t.data(), f.data());

    CapturedSpectrum sp;
    sp.binHz = 48000.0 / kFft;
    sp.outRms = std::sqrt(acc / kFft);
    sp.mag.resize(kFft / 2 + 1);
    for (size_t k = 0; k < sp.mag.size(); ++k)
        sp.mag[k] = std::sqrt(f[2 * k] * f[2 * k] + f[2 * k + 1] * f[2 * k + 1]);
    return sp;
}

/// THD+N (%) relative to the fundamental: spectral energy outside the
/// fundamental's +-8 bin main lobe, within 20 Hz..20 kHz.
double thdnPercent(const CapturedSpectrum& sp, double f0)
{
    const auto bin0 = static_cast<long>(std::lround(f0 / sp.binHz));
    const auto lo = static_cast<size_t>(std::max(1.0, 20.0 / sp.binHz));
    const auto hi = std::min(sp.mag.size() - 1, static_cast<size_t>(20000.0 / sp.binHz));
    double fund = 0.0, total = 0.0;
    for (size_t k = lo; k <= hi; ++k)
    {
        const double p = sp.mag[k] * sp.mag[k];
        total += p;
        if (std::llabs(static_cast<long long>(k) - bin0) <= 8)
            fund += p;
    }
    if (fund <= 0.0) return 100.0;
    return 100.0 * std::sqrt(std::max(total - fund, 0.0) / fund);
}

/// Worst spurious component (dB re fundamental peak): the largest bin that is
/// neither near DC nor within +-8 bins of any harmonic of f0.
double spuriousDb(const CapturedSpectrum& sp, double f0)
{
    const auto hi = std::min(sp.mag.size() - 1, static_cast<size_t>(20000.0 / sp.binHz));
    const auto bin0 = std::max<long>(1, std::lround(f0 / sp.binHz));
    double fund = 0.0;
    for (long k = bin0 - 8; k <= bin0 + 8; ++k)
        if (k > 0 && static_cast<size_t>(k) <= hi)
            fund = std::max(fund, sp.mag[static_cast<size_t>(k)]);

    double worst = 0.0;
    for (size_t k = static_cast<size_t>(std::max(1.0, 30.0 / sp.binHz)); k <= hi; ++k)
    {
        const double harm = static_cast<double>(k) / bin0;
        if (std::abs(harm - std::round(harm)) * bin0 <= 8.0) continue;   // harmonic lobe
        worst = std::max(worst, sp.mag[k]);
    }
    return toDb(worst / std::max(fund, 1e-12));
}

/// Output RMS (dBFS) with a silent input, after a short warmup.
double noiseFloorDbfs(const std::function<void(dspark::AudioBufferView<float>)>& process)
{
    constexpr int kBlock = 512, kWarm = 24, kMeas = 94;   // ~1 s measured
    dspark::AudioBuffer<float> buf;
    buf.resize(2, kBlock);
    auto view = buf.toView();
    double acc = 0.0;
    long count = 0;
    for (int b = 0; b < kWarm + kMeas; ++b)
    {
        for (int i = 0; i < kBlock; ++i)
        {
            view.getChannel(0)[i] = 0.0f;
            view.getChannel(1)[i] = 0.0f;
        }
        process(view);
        if (b >= kWarm)
            for (int i = 0; i < kBlock; ++i)
            {
                acc += static_cast<double>(view.getChannel(0)[i]) * view.getChannel(0)[i];
                ++count;
            }
    }
    return toDb(std::sqrt(acc / std::max(1L, count)));
}

std::vector<MetricsCase> buildMetricsCases()
{
    const dspark::AudioSpec spec { 48000.0, 512, 2 };
    using V = dspark::AudioBufferView<float>;
    std::vector<MetricsCase> cases;
    auto add = [&](const char* name, const char* settings, const char* note,
                   std::function<void(V)> process, std::function<int()> latency = nullptr,
                   bool shifted = false) {
        cases.push_back({ name, settings, note, std::move(process), std::move(latency), shifted });
    };

    { auto p = std::make_shared<dspark::Gain<float>>();
      p->prepare(spec); p->setGainDb(-3.0f);
      add("Gain", "-3 dB", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::FilterEngine<float>>();
      p->prepare(spec); p->setLowPass(16000.0f, 0.707f, 24);
      add("FilterEngine", "LP 16 kHz, 24 dB/oct", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Equalizer<float>>();
      p->prepare(spec); p->setBand(0, 500.0f, 3.0f);
      add("Equalizer (IIR)", "+3 dB bell @ 500 Hz", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::Equalizer<float>>();
      p->setFilterMode(dspark::Equalizer<float>::FilterMode::LinearPhase);
      p->prepare(spec); p->setBand(0, 500.0f, 3.0f);
      add("Equalizer (linear phase)", "+3 dB bell @ 500 Hz", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::Compressor<float>>();
      p->prepare(spec); p->setThreshold(-20.0f); p->setRatio(4.0f);
      add("Compressor", "-20 dB, 4:1", "gain-riding on steady tone",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::Limiter<float>>();
      p->prepare(spec); p->setCeiling(-1.0f); p->setTruePeak(true);
      add("Limiter", "-1 dBTP ceiling", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::NoiseGate<float>>();
      p->prepare(spec); p->setThreshold(-60.0f);
      add("NoiseGate", "-60 dB (open)", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Expander<float>>();
      p->prepare(spec); p->setThreshold(-60.0f);
      add("Expander", "-60 dB (inactive)", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::DynamicEQ<float>>();
      p->prepare(spec);
      dspark::DynamicEQ<float>::BandConfig cfg;
      cfg.frequency = 3000.0f; cfg.threshold = -30.0f; cfg.aboveRatio = 3.0f;
      p->setBand(0, cfg); p->setNumBands(1);
      add("DynamicEQ", "3 kHz band, -30 dB, 3:1", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::MultibandCompressor<float>>();
      p->prepare(spec); p->setNumBands(3);
      add("MultibandCompressor", "3 bands, defaults", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::TransientDesigner<float>>();
      p->prepare(spec); p->setAttack(20.0f); p->setSustain(0.0f);
      add("TransientDesigner", "attack +20%", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::DeEsser<float>>();
      p->prepare(spec); p->setFrequency(7000.0f); p->setThreshold(-30.0f);
      add("DeEsser", "7 kHz, -30 dB", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Saturation<float>>();
      p->setOversampling(2); p->prepare(spec); p->setDrive(0.0f);
      add("Saturation (SoftClip)", "drive 0 dB, 2x OS", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatencySamples(); }); }

    { auto p = std::make_shared<dspark::Clipper<float>>();
      p->setOversampling(4); p->prepare(spec);
      p->setMode(dspark::Clipper<float>::Mode::Analog);
      add("Clipper (Analog)", "0 dB in, 4x OS", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::TapeMachine<float>>();
      p->prepare(spec); p->setDrive(0.0f); p->setWowFlutter(0.0f);
      add("TapeMachine", "drive 0 dB, wow/flutter off", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::TubePreamp<float>>();
      p->prepare(spec); p->setStages(1); p->setDrive(0.0f);
      add("TubePreamp", "1 stage, drive 0 dB", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::TransformerModel<float>>();
      p->prepare(spec); p->setDrive(0.0f);
      add("TransformerModel", "drive 0 dB", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Chorus<float>>();
      p->prepare(spec); p->setRate(0.8f); p->setMix(0.5f);
      add("Chorus", "0.8 Hz, mix 50%", "modulated: sidebands are the effect",
          [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Phaser<float>>();
      p->prepare(spec); p->setRate(0.5f);
      add("Phaser", "0.5 Hz", "modulated: sidebands are the effect",
          [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Tremolo<float>>();
      p->prepare(spec); p->setRate(5.0f); p->setDepth(0.7f);
      add("Tremolo", "5 Hz, depth 0.7", "amplitude modulation is the effect",
          [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Vibrato<float>>();
      p->prepare(spec); p->setRate(5.0f); p->setDepth(0.4f);
      add("Vibrato", "5 Hz, depth 0.4", "FM: spectrum displaced by design",
          [p](V b){ p->processBlock(b); }, nullptr, true); }

    { auto p = std::make_shared<dspark::RingModulator<float>>();
      p->prepare(spec); p->setFrequency(300.0f); p->setMix(1.0f);
      add("RingModulator", "300 Hz carrier", "spectrum displaced by design",
          [p](V b){ p->processBlock(b); }, nullptr, true); }

    { auto p = std::make_shared<dspark::FrequencyShifter<float>>();
      p->prepare(spec); p->setShift(50.0f);
      add("FrequencyShifter", "+50 Hz", "spectrum displaced by design",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }, true); }

    { auto p = std::make_shared<dspark::PitchShifter<float>>();
      p->prepare(spec); p->setSemitones(4.0f);
      add("PitchShifter", "+4 st", "spectrum displaced by design",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }, true); }

    { auto p = std::make_shared<dspark::GranularProcessor<float>>();
      p->prepare(spec); p->setDensity(30.0f);
      add("GranularProcessor", "density 30/s", "granular texture is the effect",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::SpectralDenoiser<float>>();
      p->prepare(spec); p->setReduction(12.0f);
      add("SpectralDenoiser", "reduction 12 dB", "",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::AlgorithmicReverb<float>>();
      p->prepare(spec); p->setType(dspark::AlgorithmicReverb<float>::Type::Hall);
      p->setMix(0.4f);
      add("AlgorithmicReverb", "Hall, mix 40%", "time-based: tail energy",
          [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Reverb<float>>();
      p->prepare(spec);
      std::vector<float> ir(4800);
      for (size_t i = 0; i < ir.size(); ++i)
          ir[i] = static_cast<float>(std::exp(-3.0 * static_cast<double>(i) / 4800.0)
                * std::sin(0.1 * static_cast<double>(i)));
      ir[0] = 1.0f;
      p->loadIR(ir.data(), static_cast<int>(ir.size()), 48000.0);
      p->setMix(0.3f);
      add("ConvolutionReverb", "synthetic IR, mix 30%", "time-based: tail energy",
          [p](V b){ p->processBlock(b); }, [p]{ return p->getLatency(); }); }

    { auto p = std::make_shared<dspark::StereoWidth<float>>();
      p->prepare(spec); p->setWidth(1.4f); p->setBassMono(true, 120.0);
      add("StereoWidth", "width 1.4, bass mono 120 Hz", "",
          [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::Panner<float>>();
      p->prepare(spec); p->setPan(0.4f);
      add("Panner", "pan 0.4 R", "", [p](V b){ p->processBlock(b); }); }

    { auto p = std::make_shared<dspark::DCBlocker<float>>();
      p->prepare(48000.0, 2, 10.0); p->setOrder(4);
      add("DCBlocker", "10 Hz, order 4", "", [p](V b){ p->processBlock(b); }); }

    return cases;
}

int runMetricsTable(const char* outPath)
{
#ifndef DSPARK_NO_FILE_IO
    FILE* out = std::fopen(outPath, "w");
    if (!out)
    {
        std::printf("ERROR: cannot open %s for writing\n", outPath);
        return 1;
    }

    std::fprintf(out,
        "# DSPark — Per-Processor Quality Metrics\n\n"
        "Generated by `dspark_conformance --metrics` (48 kHz, stereo, block 512).\n\n"
        "- **THD+N**: spectral energy outside the fundamental, 1 kHz tone at -6 dBFS,\n"
        "  relative to the fundamental, 20 Hz-20 kHz.\n"
        "- **Noise floor**: output RMS (dBFS) with silent input.\n"
        "- **Spurious**: worst non-harmonic component (dB re fundamental) with a\n"
        "  10.1 kHz tone at -6 dBFS — aliasing shows up here.\n"
        "- **Latency**: as reported by `getLatency()` (samples @ 48 kHz). Validated\n"
        "  by the PDC null tests in this same suite.\n\n"
        "| Processor | Settings | Latency | THD+N @1 kHz | Noise floor | Spurious @10.1 kHz | Notes |\n"
        "|---|---|---:|---:|---:|---:|---|\n");

    // Bin-centred probe frequencies (32768-point FFT @ 48 kHz) minimise leakage.
    const double bin = 48000.0 / 32768.0;
    const double f1k = bin * std::round(1000.0 / bin);
    const double f10k = bin * std::round(10100.0 / bin);

    auto cases = buildMetricsCases();
    for (auto& c : cases)
    {
        // Noise floor FIRST, while the processor state is pristine — granular
        // and reverb tails from a previous tone are not the effect's noise.
        const double noise = noiseFloorDbfs(c.process);
        const int lat = c.latency ? c.latency() : 0;

        char thdnStr[32], spurStr[32];
        if (c.shifted)
        {
            // No fundamental survives at the input frequency: the columns do
            // not apply (the displacement IS the effect).
            std::snprintf(thdnStr, sizeof(thdnStr), "n/a");
            std::snprintf(spurStr, sizeof(spurStr), "n/a");
        }
        else
        {
            const auto sp1 = captureSpectrum(c.process, f1k, 0.5f);
            const double thdn = thdnPercent(sp1, f1k);
            const auto sp10 = captureSpectrum(c.process, f10k, 0.5f);
            const double spur = spuriousDb(sp10, f10k);
            if (thdn < 0.001) std::snprintf(thdnStr, sizeof(thdnStr), "<0.001 %%");
            else              std::snprintf(thdnStr, sizeof(thdnStr), "%.3f %%", thdn);
            std::snprintf(spurStr, sizeof(spurStr), "%.1f dB", spur);
        }

        std::fprintf(out, "| %s | %s | %d | %s | %.1f dBFS | %s | %s |\n",
                     c.name, c.settings, lat, thdnStr, noise, spurStr, c.note);
        std::printf("TABLE %-26s thdn %-9s noise %.1f dBFS  spur %-9s lat %d\n",
                    c.name, thdnStr, noise, spurStr, lat);
    }

    std::fprintf(out,
        "\nModulated, inharmonic-by-design and time-based processors necessarily show\n"
        "high \"THD+N\" — for them the column documents the character at the listed\n"
        "settings rather than a defect. Linear processors read at the measurement's\n"
        "own floor.\n");
    std::fclose(out);
    std::printf("\nMetrics table written to %s\n", outPath);
    return 0;
#else
    (void) outPath;
    std::printf("metrics table requires file IO (built with DSPARK_NO_FILE_IO)\n");
    return 1;
#endif
}

} // namespace

int main(int argc, char** argv)
{
    for (int i = 1; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--metrics") == 0)
            return runMetricsTable(argv[i + 1]);

    std::printf("DSPark public conformance suite\n");
    std::printf("================================\n\n");

    runSimdKernelTests();
    runCoreGuardTests();
    runSmokeTests();
    runPdcNullTests();
    runMetricTests();
    runEbuTests();

    std::printf("\n================================\n");
    std::printf("%d passed, %d failed, %d skipped\n", g_passed, g_failed, g_skipped);
    return g_failed == 0 ? 0 : 1;
}
