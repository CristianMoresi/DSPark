// DSPark example - offline WAV processing
// Copyright (c) 2026 Cristian Moresi - MIT License
//
// The simplest end-to-end DSPark program: read a WAV, run a small mastering
// chain (EQ -> compressor -> true-peak limiter), write the result as 24-bit
// WAV, and report the loudness of what was written.
//
// Build:  g++ -std=c++20 -O2 -I .. wav_process.cpp -o wav_process
// Usage:  ./wav_process input.wav output.wav

#include "DSPark.h"

#include <algorithm>
#include <cstdint>
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
    std::printf("in : %s - %.0f Hz, %u ch, %lld frames\n",
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
    // The limiter's lookahead (and a lookahead compressor, if enabled) delay
    // the signal. An offline render compensates, so the file stays aligned
    // with its source: drop the first `latency` output frames, and feed as
    // many frames of silence past the end so the tail is not cut off.
    const int latency = comp.getLatency() + limiter.getLatency();

    dspark::AudioBuffer<float> buf;
    buf.resize(static_cast<int>(info.numChannels), kBlock);

    int64_t toSkip = latency;
    int64_t remaining = info.numSamples + latency;
    int64_t offset = 0;
    while (remaining > 0)
    {
        const int n = static_cast<int>(std::min<int64_t>(remaining, kBlock));
        auto view = buf.toView().getSubView(0, n);
        const int fromFile = static_cast<int>(
            std::clamp<int64_t>(info.numSamples - offset, 0, n));
        if (fromFile > 0 && !in.readSamples(view.getSubView(0, fromFile), offset, fromFile))
        {
            std::printf("read failed\n");
            return 1;
        }
        if (fromFile < n)
            view.getSubView(fromFile, n - fromFile).clear();   // flush silence

        eq.processBlock(view);
        comp.processBlock(view);
        limiter.processBlock(view);

        const int skip = static_cast<int>(std::min<int64_t>(toSkip, n));
        toSkip -= skip;
        if (skip < n)
        {
            auto written = view.getSubView(skip, n - skip);
            meter.processBlock(written);
            if (!out.writeSamples(written))
            {
                std::printf("write failed\n");
                return 1;
            }
        }
        offset += n;
        remaining -= n;
    }
    in.close();
    out.close();

    std::printf("out: %s - integrated %.1f LUFS, true peak %.2f dBTP\n",
                argv[2],
                static_cast<double>(meter.getIntegratedLUFS()),
                static_cast<double>(meter.getTruePeakDb()));
    return 0;
}
