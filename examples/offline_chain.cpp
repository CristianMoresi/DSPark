// DSPark - Complete-source offline chain example.
// Copyright (c) 2026 Cristian Moresi - MIT License
// cl /std:c++20 /EHsc /O2 /I . examples\offline_chain.cpp
// c++ -std=c++20 -O2 -I . examples/offline_chain.cpp -o offline_chain

#include "Effects/OfflineBeatCompressor.h"
#include "Effects/OfflineHardClipper.h"
#include "Effects/OfflineLeveler.h"
#include "Effects/OfflinePeakCompressor.h"
#include "Effects/OfflinePunch.h"
#include "Effects/OfflineSoftClipper.h"
#include "Effects/OfflineStereoBalance.h"
#include "Effects/OfflineStereoGenerator.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <utility>

template <class T> bool example()
{
    constexpr int rate = 48000, frames = 4 * rate;
    dspark::AudioBuffer<T> input;
    input.resize(2, frames);
    for (int i = 0; i < frames; ++i)
    {
        const double t = static_cast<double>(i) / rate;
        const int beat = i / (rate / 2);
        const double age = static_cast<double>(i % (rate / 2)) / rate;
        const double amplitude = .2 + .12 * (beat % 4);
        const double mid = amplitude * std::exp(-28 * age) *
                               std::cos(dspark::twoPi<double> * 83 * t) +
                           .018 * std::sin(dspark::twoPi<double> * 317 * t);
        const double side = .009 * std::sin(dspark::twoPi<double> * 997 * t);
        input.getChannel(0)[i] = static_cast<T>(mid + side);
        input.getChannel(1)[i] = static_cast<T>(mid - side);
    }
    // Owning run() is convenient for a source already in memory. For long
    // recordings, use OfflineAudioSource/OfflineAudioSink and analyze/render.
    // Every stage below analyzes its actual input; old plans are never reused
    // after another processor changes the source PCM.
    std::array<std::array<T, 128>, 2> originalTail{};
    for (int c = 0; c < 2; ++c)
        std::copy_n(input.getChannel(c) + frames - 128, 128, originalTail[c].data());
    dspark::AudioBuffer<T> current = std::move(input), next;
    std::atomic<bool> cancel{false};
    dspark::OfflineJobOptions job;
    job.blockFrames = 257;
    job.cancel = &cancel; // A host may set this flag from its UI/control thread.
    const std::array<dspark::OfflineRegion, 1> protectedTail{{{frames - 128, frames}}};
    const auto apply = [&](const char *name, const auto &processor, auto options)
    {
        options.exclusions = protectedTail;
        const auto result = processor.run(current, next, rate, options, job);
        std::cout << name << ": status=" << static_cast<int>(result.status)
                  << " peak=" << result.report.outputSamplePeakDb
                  << " dBFS, worker-payload=" << result.memoryBytes << " bytes\n";
        if (!result.succeeded())
            return false; // Failure/cancellation leaves the owning output unchanged.
        if (next.getNumSamples() != frames || next.getNumChannels() != 2)
            return false;
        for (int c = 0; c < 2; ++c)
        {
            if (std::memcmp(next.getChannel(c) + frames - 128,
                            originalTail[c].data(), 128 * sizeof(T)) != 0)
                return false;
            for (int i = 0; i < frames; ++i)
                if (!std::isfinite(next.getChannel(c)[i]))
                    return false;
        }
        current = std::move(next);
        return true;
    };
    typename dspark::OfflineLeveler<T>::Options level;
    level.amount = .4;
    level.maximumBoostDb = 3;
    if (!apply("Leveler", dspark::OfflineLeveler<T>(), level)) return false;
    typename dspark::OfflinePeakCompressor<T>::Options peak;
    peak.reductionDb = 1;
    if (!apply("Peak", dspark::OfflinePeakCompressor<T>(), peak)) return false;
    typename dspark::OfflineBeatCompressor<T>::Options beat;
    beat.reductionDb = 1;
    if (!apply("Beat", dspark::OfflineBeatCompressor<T>(), beat)) return false;
    typename dspark::OfflinePunch<T>::Options punch;
    punch.boostDb = .5;
    if (!apply("Punch", dspark::OfflinePunch<T>(), punch)) return false;
    typename dspark::OfflineSoftClipper<T>::Options soft;
    soft.reductionDb = .5;
    if (!apply("Soft clip", dspark::OfflineSoftClipper<T>(), soft)) return false;
    typename dspark::OfflineHardClipper<T>::Options hard;
    hard.reductionDb = .5;
    if (!apply("Hard clip", dspark::OfflineHardClipper<T>(), hard)) return false;
    typename dspark::OfflineStereoGenerator<T>::Options stereo;
    stereo.width = .3f;
    if (!apply("Stereo generation", dspark::OfflineStereoGenerator<T>(), stereo)) return false;
    // Optional post-generation side leveling and guard. The actual generated
    // signal is analyzed, at the selected width, without changing its mid.
    typename dspark::OfflineStereoBalance<T>::Options balance;
    balance.levelingAmount = .5;
    balance.guard = true;
    if (!apply("Stereo balance", dspark::OfflineStereoBalance<T>(), balance)) return false;
    // Offline outputs keep the source frame count and source-aligned timing;
    // the stereo adapter has already compensated the real-time core's latency.
    return true;
}

int main()
{
    std::cout << "float\n";
    if (!example<float>()) return 1;
    std::cout << "double\n";
    return example<double>() ? 0 : 1;
}
