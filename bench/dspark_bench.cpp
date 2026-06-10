// DSPark — Reproducible micro-benchmarks
// Copyright (c) 2026 Cristian Moresi — MIT License
//
// Zero-dependency benchmark harness for the DSP hot paths. Prints a Markdown
// table; numbers are machine-dependent, so commit them only together with the
// CPU/compiler line this program prints first.
//
// Build (MSVC):  cl /std:c++20 /EHsc /O2 /I . bench\dspark_bench.cpp
// Build (GCC):   g++ -std=c++20 -O2 -I . bench/dspark_bench.cpp -o dspark_bench
//
// Method: each case is auto-calibrated to run >= 50 ms per timed run, 5 runs,
// median ns per processed frame reported (1 frame = 1 sample per channel).

#include "DSPark.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

volatile double g_sink = 0.0; // defeats dead-code elimination

constexpr double kPiBench = 3.14159265358979323846;
constexpr int    kBlock = 512;
constexpr double kRate = 48000.0;

template <typename F>
double medianNsPerFrame(F&& iter, double framesPerCall)
{
    for (int i = 0; i < 3; ++i)
        iter();

    std::array<double, 5> runs {};
    for (auto& r : runs)
    {
        int calls = 0;
        const auto t0 = std::chrono::steady_clock::now();
        double elapsed = 0.0;
        do
        {
            iter();
            ++calls;
            elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } while (elapsed < 0.05);
        r = elapsed * 1e9 / (framesPerCall * calls);
    }
    std::sort(runs.begin(), runs.end());
    return runs[2];
}

void printRow(const char* name, double nsPerFrame)
{
    const double msamples = 1000.0 / nsPerFrame;
    const double xRealtime = (1e9 / nsPerFrame) / kRate;
    std::printf("| %-36s | %10.2f | %9.1f | %10.0fx |\n", name, nsPerFrame, msamples, xRealtime);
}

// Source signal shared by all cases.
struct Source
{
    std::vector<float> data;
    Source()
    {
        data.resize(kBlock);
        for (int i = 0; i < kBlock; ++i)
            data[static_cast<size_t>(i)] =
                0.5f * static_cast<float>(std::sin(2.0 * kPiBench * 220.0 * i / kRate));
    }
};

} // namespace

int main()
{
    const Source src;
    const dspark::AudioSpec spec { kRate, kBlock, 2 };

    std::printf("DSPark micro-benchmarks — block %d, %.0f Hz, single thread\n", kBlock, kRate);
#if defined(_MSC_VER)
    std::printf("Compiler: MSVC %d\n\n", _MSC_VER);
#elif defined(__clang__)
    std::printf("Compiler: Clang %d.%d\n\n", __clang_major__, __clang_minor__);
#elif defined(__GNUC__)
    std::printf("Compiler: GCC %d.%d\n\n", __GNUC__, __GNUC_MINOR__);
#endif

    std::printf("| Processor (stereo unless noted)      | ns / frame | Msample/s | x realtime |\n");
    std::printf("|--------------------------------------|------------|-----------|------------|\n");

    dspark::AudioBuffer<float> stereo;
    stereo.resize(2, kBlock);
    auto fill = [&] {
        std::memcpy(stereo.getChannel(0), src.data.data(), kBlock * sizeof(float));
        std::memcpy(stereo.getChannel(1), src.data.data(), kBlock * sizeof(float));
    };

    {
        printRow("memcpy baseline", medianNsPerFrame([&] {
            fill();
            g_sink += stereo.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::Biquad<float> bq;
        bq.setCoeffs(dspark::BiquadCoeffs<float>::makeLowPass(kRate, 1000.0));
        dspark::AudioBuffer<float> mono;
        mono.resize(1, kBlock);
        printRow("Biquad low-pass (mono)", medianNsPerFrame([&] {
            std::memcpy(mono.getChannel(0), src.data.data(), kBlock * sizeof(float));
            bq.processBlock(mono.toView());
            g_sink += mono.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::FilterEngine<float> fe;
        fe.prepare(spec);
        fe.setLowPass(4000.0f, 0.707f, 24);
        printRow("FilterEngine LP 24 dB/oct", medianNsPerFrame([&] {
            fill();
            fe.processBlock(stereo.toView());
            g_sink += stereo.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::Equalizer<float> eq;
        eq.prepare(spec);
        eq.setBand(0, 120.0f, 2.5f);
        eq.setBand(1, 800.0f, -1.5f);
        eq.setBand(2, 3200.0f, 3.0f);
        eq.setBand(3, 9000.0f, -2.0f);
        printRow("Equalizer 4 bands", medianNsPerFrame([&] {
            fill();
            eq.processBlock(stereo.toView());
            g_sink += stereo.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::Compressor<float> comp;
        comp.prepare(spec);
        comp.setThreshold(-20.0f);
        comp.setRatio(4.0f);
        printRow("Compressor", medianNsPerFrame([&] {
            fill();
            comp.processBlock(stereo.toView());
            g_sink += stereo.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::AlgorithmicReverb<float> rev;
        rev.prepare(spec);
        printRow("AlgorithmicReverb", medianNsPerFrame([&] {
            fill();
            rev.processBlock(stereo.toView());
            g_sink += stereo.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::Oversampling<float> os(4, dspark::Oversampling<float>::Quality::High);
        os.prepare(spec);
        printRow("Oversampling 4x up+down", medianNsPerFrame([&] {
            fill();
            (void)os.upsample(stereo.toView());
            os.downsample(stereo.toView());
            g_sink += stereo.getChannel(0)[0];
        }, kBlock));
    }
    for (const int irLen : { 4800, 48000 })
    {
        std::vector<float> ir(static_cast<size_t>(irLen));
        for (int i = 0; i < irLen; ++i)
            ir[static_cast<size_t>(i)] = static_cast<float>(
                std::exp(-4.0 * i / irLen) * std::cos(0.07 * i));
        dspark::Convolver<float> conv;
        conv.prepare(kBlock, ir.data(), irLen);
        dspark::AudioBuffer<float> mono;
        mono.resize(1, kBlock);
        char name[64];
        std::snprintf(name, sizeof(name), "Convolver IR %.1f s (mono)", irLen / kRate);
        printRow(name, medianNsPerFrame([&] {
            std::memcpy(mono.getChannel(0), src.data.data(), kBlock * sizeof(float));
            conv.processInPlace(mono.getChannel(0), kBlock);
            g_sink += mono.getChannel(0)[0];
        }, kBlock));
    }
    {
        dspark::Resampler<double> rs;
        rs.prepare(44100.0, 48000.0, dspark::Resampler<double>::Quality::High);
        std::vector<double> in(44100);
        for (size_t i = 0; i < in.size(); ++i)
            in[i] = std::sin(2.0 * kPiBench * 1000.0 * static_cast<double>(i) / 44100.0);
        printRow("Resampler 44.1->48 High (mono, dbl)", medianNsPerFrame([&] {
            auto out = rs.process(in.data(), static_cast<int>(in.size()));
            g_sink += out.empty() ? 0.0 : out[0];
        }, 48000.0));
    }

    std::printf("\n| FFT (float, forward + inverse) | us / pair | pairs / s |\n");
    std::printf("|--------------------------------|-----------|-----------|\n");
    for (const int size : { 1024, 4096, 8192 })
    {
        dspark::FFTReal<float> fft(static_cast<size_t>(size));
        std::vector<float> t(static_cast<size_t>(size)), f(static_cast<size_t>(size) + 2);
        for (int i = 0; i < size; ++i)
            t[static_cast<size_t>(i)] = static_cast<float>(std::sin(0.05 * i));
        const double nsPerPair = medianNsPerFrame([&] {
            fft.forward(t.data(), f.data());
            fft.inverse(f.data(), t.data());
            g_sink += t[0];
        }, 1.0);
        std::printf("| FFTReal %-22d | %9.2f | %9.0f |\n", size, nsPerPair / 1000.0, 1e9 / nsPerPair);
    }

    std::printf("\n(sink %.3f)\n", g_sink == 0.0 ? 0.0 : 1.0);
    return 0;
}
