// DSPark example — real-time style vocal channel strip
// Copyright (c) 2026 Cristian Moresi — MIT License
//
// Shows the real-time usage pattern: prepare() everything up front, then a
// block loop with zero allocation, exactly like an audio callback. The chain
// is a classic vocal strip:
//
//   NoiseGate -> Equalizer -> Compressor -> DeEsser -> Limiter
//
// Input here is a synthetic "voice": a glided tone with vibrato plus bursts
// of noise (consonants), so the example runs with no input files. Replace the
// synth block with your callback's input buffer in a real application.
//
// Build:  g++ -std=c++20 -O2 -I .. channel_strip.cpp -o channel_strip

#include "DSPark.h"

#include <cmath>
#include <cstdio>

int main()
{
    constexpr double kRate  = 48000.0;
    constexpr int    kBlock = 512;
    const dspark::AudioSpec spec { kRate, kBlock, 2 };

    // --- prepare (allocation happens here, never in the loop) ---------------
    dspark::NoiseGate<float> gate;
    gate.prepare(spec);
    gate.setThreshold(-45.0f);

    dspark::Equalizer<float> eq;
    eq.prepare(spec);
    eq.setBand(0, 110.0f, -2.0f);   // rumble / proximity
    eq.setBand(1, 350.0f, -1.5f);   // boxiness
    eq.setBand(2, 3500.0f, 2.0f);   // presence
    eq.setBand(3, 11000.0f, 1.5f);  // air

    dspark::Compressor<float> comp;
    comp.prepare(spec);
    comp.setThreshold(-22.0f);
    comp.setRatio(3.0f);

    dspark::DeEsser<float> deEsser;
    deEsser.prepare(spec);
    deEsser.setFrequency(6500.0f);
    deEsser.setThreshold(-28.0f);

    dspark::Limiter<float> limiter;
    limiter.prepare(spec);
    limiter.setCeiling(-1.0f);
    limiter.setTruePeak(true);

    dspark::LoudnessMeter<float> meter;
    meter.prepare(kRate, 2);

    dspark::AudioBuffer<float> buf;
    buf.resize(2, kBlock);

    // --- the "callback" loop: 5 seconds of synthetic voice ------------------
    const int totalBlocks = static_cast<int>(5.0 * kRate / kBlock);
    double phase = 0.0;
    uint32_t rng = 0x12345678u;

    for (int b = 0; b < totalBlocks; ++b)
    {
        const double t = b * kBlock / kRate;
        const bool consonant = std::fmod(t, 0.7) < 0.06;          // noise bursts
        const double f0 = 165.0 * std::pow(2.0, 0.25 * std::sin(2.0 * 3.14159265358979 * 0.4 * t))
                        * (1.0 + 0.015 * std::sin(2.0 * 3.14159265358979 * 5.5 * t));

        float* L = buf.getChannel(0);
        float* R = buf.getChannel(1);
        for (int i = 0; i < kBlock; ++i)
        {
            phase += f0 / kRate;
            if (phase >= 1.0) phase -= 1.0;
            float v = 0.30f * static_cast<float>(
                std::sin(2.0 * 3.14159265358979 * phase)
                + 0.4 * std::sin(4.0 * 3.14159265358979 * phase));
            if (consonant)
            {
                rng = rng * 1664525u + 1013904223u;
                v = 0.15f * (static_cast<float>(rng >> 8) / 8388608.0f - 1.0f);
            }
            L[i] = v;
            R[i] = v;
        }

        auto view = buf.toView();
        gate.processBlock(view);
        eq.processBlock(view);
        comp.processBlock(view);
        deEsser.processBlock(view);
        limiter.processBlock(view);
        meter.processBlock(view);
    }

    std::printf("processed %.1f s — momentary %.1f LUFS, integrated %.1f LUFS, true peak %.2f dBTP\n",
                totalBlocks * kBlock / kRate,
                static_cast<double>(meter.getMomentaryLUFS()),
                static_cast<double>(meter.getIntegratedLUFS()),
                static_cast<double>(meter.getTruePeakDb()));
    return 0;
}
