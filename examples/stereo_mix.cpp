// DSPark - Real-time stereo generation with an aligned parallel dry path.
// Copyright (c) 2026 Cristian Moresi - MIT License
// cl /std:c++20 /EHsc /O2 /I . examples\stereo_mix.cpp
// c++ -std=c++20 -O2 -I . examples/stereo_mix.cpp -o stereo_mix

#include "Core/DryWetMixer.h"
#include "Effects/StereoGenerator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <type_traits>

template <class T> bool example(double rate, int factor, float lowCut)
{
    constexpr int maxBlock = 257, sourceFrames = 2048;
    const dspark::AudioSpec spec{rate, maxBlock, 2};
    dspark::StereoGenerator<T> generator;
    if (!generator.prepare(spec, {factor, lowCut})) return false;

    dspark::DryWetMixer<T> mixer;
    mixer.prepare(spec);
    // Dry and wet share the original mid: linear mixing preserves its level.
    mixer.setMixRule(dspark::DryWetMixer<T>::MixRule::Linear);
    const int latency = generator.getLatency();
    mixer.setLatencyCompensation(latency); // Setup only; allocates delay storage.
    // A plugin must also publish this latency to its host before playback.
    // Width and mix changes do not change it. New factor/low-cut settings need
    // a stopped stream and preparation of both the effect and compensation.

    dspark::AudioBuffer<T> source, block, reference;
    source.resize(2, sourceFrames);
    block.resize(2, maxBlock);
    reference.resize(2, sourceFrames + latency);
    for (int i = 0; i < sourceFrames; ++i)
    {
        const double time = i / rate;
        const double mid = .21 * std::sin(dspark::twoPi<double> * 997 * time)
                         + .07 * std::sin(dspark::twoPi<double> * 11731 * time);
        const double side = .03 * std::sin(dspark::twoPi<double> * 233 * time);
        source.getChannel(0)[i] = static_cast<T>(mid + side);
        source.getChannel(1)[i] = static_cast<T>(mid - side);
    }
    // Nonzero source boundaries make a missing/double delay observable.
    source.getChannel(0)[0] = source.getChannel(1)[0] = T(.4);
    source.getChannel(0)[sourceFrames - 1] = T(-.3);
    source.getChannel(1)[sourceFrames - 1] = T(.1);

    double largestMidError = 0, largestPartitionError = 0;
    for (float width : {0.f, .65f})
    {
        for (bool ragged : {false, true})
        {
            if (!generator.setWidth(width)) return false;
            generator.resetAtFrame(0);
            mixer.reset();
            // On a real seek, reset both histories. resetAtFrame(position)
            // aligns the modulation clock; replay earlier PCM to restore the
            // filter history when seamless continuity is required.
            constexpr std::array<int, 5> sizes{1, 31, 257, 7, 113};
            int offset = 0;
            std::size_t part = 0;
            double generatedEnergy = 0;
            while (offset < sourceFrames + latency)
            {
                const int count = std::min(sourceFrames + latency - offset,
                                           ragged ? sizes[part++ % sizes.size()] : maxBlock);
                auto audio = block.toView().getSubView(0, count);
                for (int c = 0; c < 2; ++c)
                    for (int i = 0; i < count; ++i)
                        audio.getChannel(c)[i] = offset + i < sourceFrames
                            ? source.getChannel(c)[offset + i] : T(0);

                // Audio callback: all buffers and delay histories already exist.
                // The host supplies finite, writable, nonoverlapping stereo PCM.
                mixer.pushDry(audio);
                if (!generator.processBlock(audio)) return false;
                mixer.mixWet(audio, T(.5));
                // On failure stop this stream, report generator.getStatus(), and
                // reset both histories before restarting; do not continue with
                // a dry history that advanced past a rejected input block.

                // Standalone validation only; a plugin sends audio to its host.
                for (int i = 0; i < count; ++i)
                {
                    const int frame = offset + i, inputFrame = frame - latency;
                    const double left = audio.getChannel(0)[i], right = audio.getChannel(1)[i];
                    if (!std::isfinite(left) || !std::isfinite(right)) return false;
                    const double dryL = inputFrame < 0 ? 0 : source.getChannel(0)[inputFrame];
                    const double dryR = inputFrame < 0 ? 0 : source.getChannel(1)[inputFrame];
                    largestMidError = std::max(largestMidError, std::abs((left + right - dryL - dryR) * .5));
                    generatedEnergy += (left - dryL) * (left - dryL) + (right - dryR) * (right - dryR);
                    for (int c = 0; c < 2; ++c)
                    {
                        if (!ragged) reference.getChannel(c)[frame] = audio.getChannel(c)[i];
                        else largestPartitionError = std::max(largestPartitionError,
                            std::abs(double(audio.getChannel(c)[i]) - reference.getChannel(c)[frame]));
                    }
                }
                offset += count;
            }
            if (generator.getSourceFrame() != static_cast<std::uint64_t>(sourceFrames + latency)) return false;
            if (width == 0 ? generatedEnergy != 0 : !(generatedEnergy > 1e-10)) return false;
        }
    }
    // Zero padding here retrieves the delayed source, not the complete effect
    // tail. Offline hosts must define a tail policy separately from latency.
    const double tolerance = 4 * std::numeric_limits<T>::epsilon();
    const bool passed = largestMidError <= tolerance && largestPartitionError <= tolerance;
    std::cout << (std::is_same_v<T, float> ? "float" : "double") << ' ' << rate
              << " Hz, factor=" << factor << ", low-cut=" << lowCut
              << " Hz, latency=" << latency << ", mid-error=" << largestMidError
              << ", block-error=" << largestPartitionError << (passed ? " PASS\n" : " FAIL\n");
    return passed;
}

int main()
{
    for (double rate : {44100., 96000.})
        for (int factor : {1, 4})
            for (float lowCut : {0.f, 175.f})
                if (!example<float>(rate, factor, lowCut) || !example<double>(rate, factor, lowCut))
                    return 1;
    return 0;
}
