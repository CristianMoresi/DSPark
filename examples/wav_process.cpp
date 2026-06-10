// DSPark example — offline WAV processing
// Copyright (c) 2026 Cristian Moresi — MIT License
//
// The simplest end-to-end DSPark program: read a WAV, run a small mastering
// chain (EQ -> compressor -> true-peak limiter), write the result as 24-bit
// WAV, and report the loudness of what was written.
//
// Build:  g++ -std=c++20 -O2 -I .. wav_process.cpp -o wav_process
// Usage:  ./wav_process input.wav output.wav

#include "DSPark.h"

#include <cstdio>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: %s <input.wav> <output.wav>\n", argv[0]);
        return 1;
    }

    dspark::WavFile in;
    if (!in.openRead(argv[1]))
    {
        std::printf("could not open %s\n", argv[1]);
        return 1;
    }
    const dspark::AudioFileInfo info = in.getInfo();
    std::printf("in : %s — %.0f Hz, %u ch, %lld frames\n",
                argv[1], info.sampleRate, info.numChannels,
                static_cast<long long>(info.numSamples));

    dspark::AudioFileInfo outInfo = info;
    outInfo.bitsPerSample = 24;
    dspark::WavFile out;
    if (!out.openWrite(argv[2], outInfo))
    {
        std::printf("could not create %s\n", argv[2]);
        return 1;
    }

    // --- the chain ----------------------------------------------------------
    constexpr int kBlock = 4096;
    const dspark::AudioSpec spec { info.sampleRate, kBlock,
                                   static_cast<int>(info.numChannels) };

    dspark::Equalizer<float> eq;
    eq.prepare(spec);
    eq.setBand(0, 90.0f, -1.5f);     // tame mud
    eq.setBand(1, 3000.0f, 1.0f);    // presence
    eq.setBand(2, 12000.0f, 1.5f);   // air

    dspark::Compressor<float> comp;
    comp.prepare(spec);
    comp.setThreshold(-18.0f);
    comp.setRatio(2.5f);

    dspark::Limiter<float> limiter;
    limiter.prepare(spec);
    limiter.setCeiling(-1.0f);       // -1 dBTP for streaming targets
    limiter.setTruePeak(true);

    dspark::LoudnessMeter<float> meter;
    meter.prepare(info.sampleRate, static_cast<int>(info.numChannels));

    // --- stream -------------------------------------------------------------
    dspark::AudioBuffer<float> buf;
    buf.resize(static_cast<int>(info.numChannels), kBlock);

    int64_t remaining = info.numSamples;
    int64_t offset = 0;
    while (remaining > 0)
    {
        const int n = static_cast<int>(std::min<int64_t>(remaining, kBlock));
        auto view = buf.toView().getSubView(0, n);
        if (!in.readSamples(view, offset, n))
            break;

        eq.processBlock(view);
        comp.processBlock(view);
        limiter.processBlock(view);
        meter.processBlock(view);

        if (!out.writeSamples(view))
        {
            std::printf("write failed\n");
            return 1;
        }
        offset += n;
        remaining -= n;
    }
    in.close();
    out.close();

    std::printf("out: %s — integrated %.1f LUFS, true peak %.2f dBTP\n",
                argv[2],
                static_cast<double>(meter.getIntegratedLUFS()),
                static_cast<double>(meter.getTruePeakDb()));
    return 0;
}
