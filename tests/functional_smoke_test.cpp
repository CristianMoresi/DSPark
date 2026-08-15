// Functional smoke test - feeds a sine-wave buffer through every public effect
// at default settings and verifies the output contains no NaN/Inf and is
// reasonably bounded. This catches:
//   - prepare() bugs that leave coefficients uninitialised
//   - division-by-zero / log-of-zero hazards
//   - run-away IIR feedback loops
//   - bad default parameter values
//
// Build:
//   cl /std:c++20 /O2 /EHsc /DUNICODE /DNOMINMAX /I.. functional_smoke_test.cpp
//      /Fe:functional_smoke_test.exe
//
// Returns non-zero on the first failed effect.

#include "../DSPark.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr int kSampleRate = 48000;
constexpr int kBlockSize  = 512;
constexpr int kNumBlocks  = 8;     // ~85 ms of audio
constexpr int kChannels   = 2;

// Fill a buffer with a 1 kHz sine at -6 dBFS.
void fillSine(dspark::AudioBuffer<float>& buf, double phaseStart = 0.0)
{
    constexpr double freq = 1000.0;
    constexpr double amp  = 0.5;
    const double k = 2.0 * 3.14159265358979323846 * freq / kSampleRate;
    auto view = buf.toView();
    for (int ch = 0; ch < view.getNumChannels(); ++ch)
    {
        float* d = view.getChannel(ch);
        for (int i = 0; i < view.getNumSamples(); ++i)
            d[i] = static_cast<float>(amp * std::sin(k * (i + phaseStart)));
    }
}

// Returns true if buffer contains NaN, Inf, or any sample whose magnitude
// exceeds `boundDb`. Used as a sanity check after running an effect.
bool isUnsafe(const dspark::AudioBuffer<float>& buf, float boundLinear, std::string& reason)
{
    auto view = buf.toView();
    for (int ch = 0; ch < view.getNumChannels(); ++ch)
    {
        const float* d = view.getChannel(ch);
        for (int i = 0; i < view.getNumSamples(); ++i)
        {
            float s = d[i];
            if (std::isnan(s)) { reason = "NaN at ch=" + std::to_string(ch) + " i=" + std::to_string(i); return true; }
            if (std::isinf(s)) { reason = "Inf at ch=" + std::to_string(ch) + " i=" + std::to_string(i); return true; }
            if (std::abs(s) > boundLinear)
            {
                char b[128];
                std::snprintf(b, sizeof(b), "abs=%.3f > bound=%.3f at ch=%d i=%d",
                              s, boundLinear, ch, i);
                reason = b;
                return true;
            }
        }
    }
    return false;
}

// One smoke-test slot. Each effect captures shared_ptrs by value and exposes
// prepare/process/reset hooks, mirroring the EffectSlot pattern but trimmed
// down for the test harness (no parameters).
struct Slot
{
    const char* name;
    std::function<void(const dspark::AudioSpec&)> prepare;
    std::function<void(dspark::AudioBufferView<float>)> process;
    float bound = 4.0f;  // generous: distortion + reverb tails can spike
};

template <typename Proc>
Slot make(const char* name, std::shared_ptr<Proc> p, float bound = 4.0f)
{
    Slot s;
    s.name = name;
    s.bound = bound;
    s.prepare = [p](const dspark::AudioSpec& spec) { p->prepare(spec); };
    s.process = [p](dspark::AudioBufferView<float> b) { p->processBlock(b); };
    return s;
}

int testAll()
{
    using namespace dspark;
    AudioSpec spec { static_cast<double>(kSampleRate), kBlockSize, kChannels };

    std::vector<Slot> slots;

    // Distortion (output bound is loose because saturators can boost peaks)
    slots.push_back(make("Saturation",     std::make_shared<Saturation<float>>()));
    slots.push_back(make("Clipper",        std::make_shared<Clipper<float>>()));

    // Dynamics
    slots.push_back(make("Compressor",     std::make_shared<Compressor<float>>()));
    slots.push_back(make("Limiter",        std::make_shared<Limiter<float>>()));
    slots.push_back(make("Expander",       std::make_shared<Expander<float>>()));
    slots.push_back(make("NoiseGate",      std::make_shared<NoiseGate<float>>()));
    slots.push_back(make("DeEsser",        std::make_shared<DeEsser<float>>()));
    slots.push_back(make("MultibandComp",  std::make_shared<MultibandCompressor<float>>()));
    slots.push_back(make("DynamicEQ",      std::make_shared<DynamicEQ<float>>()));
    slots.push_back(make("TransientDes",   std::make_shared<TransientDesigner<float>>()));

    // Filters / EQ
    slots.push_back(make("Equalizer",      std::make_shared<Equalizer<float>>()));
    slots.push_back(make("FilterEngine",   std::make_shared<FilterEngine<float>>()));
    slots.push_back(make("LadderFilter",   std::make_shared<LadderFilter<float>>()));
    slots.push_back(make("DCBlocker",      std::make_shared<DCBlocker<float>>()));
    // CrossoverFilter has a multi-output API (split into N bands) so it
    // doesn't fit the in-place processBlock(buffer) contract - covered by
    // MultibandCompressor which uses it internally.

    // Modulation
    slots.push_back(make("Chorus",         std::make_shared<Chorus<float>>()));
    slots.push_back(make("Phaser",         std::make_shared<Phaser<float>>()));
    slots.push_back(make("Tremolo",        std::make_shared<Tremolo<float>>()));
    slots.push_back(make("Vibrato",        std::make_shared<Vibrato<float>>()));
    slots.push_back(make("RingModulator",  std::make_shared<RingModulator<float>>()));
    slots.push_back(make("FrequencyShift", std::make_shared<FrequencyShifter<float>>()));

    // Time
    slots.push_back(make("Reverb",         std::make_shared<Reverb<float>>()));
    slots.push_back(make("AlgoReverb",     std::make_shared<AlgorithmicReverb<float>>()));

    // Stereo
    slots.push_back(make("Panner",         std::make_shared<Panner<float>>()));
    slots.push_back(make("StereoWidth",    std::make_shared<StereoWidth<float>>()));

    // Utility
    slots.push_back(make("Gain",           std::make_shared<Gain<float>>()));

    int failed = 0;
    AudioBuffer<float> buf;
    buf.resize(kChannels, kBlockSize);

    for (const auto& slot : slots)
    {
        bool slotFailed = false;
        try
        {
            slot.prepare(spec);

            // Phase 1: feed kNumBlocks of sine, expect output to stay safe.
            for (int b = 0; b < kNumBlocks && !slotFailed; ++b)
            {
                fillSine(buf, b * kBlockSize);
                slot.process(buf.toView());
                std::string reason;
                if (isUnsafe(buf, slot.bound, reason))
                {
                    std::printf("FAIL %-18s block %d: %s\n", slot.name, b, reason.c_str());
                    ++failed;
                    slotFailed = true;
                }
            }

            // Phase 2 (tail test): feed silence; reverbs/delays should decay
            // gracefully - no NaN/Inf, no run-away IIR feedback.
            for (int b = 0; b < kNumBlocks && !slotFailed; ++b)
            {
                buf.toView().clear();
                slot.process(buf.toView());
                std::string reason;
                if (isUnsafe(buf, slot.bound, reason))
                {
                    std::printf("FAIL %-18s tail block %d: %s\n", slot.name, b, reason.c_str());
                    ++failed;
                    slotFailed = true;
                }
            }

            if (!slotFailed)
                std::printf("OK   %-18s (sine + silence tail)\n", slot.name);
        }
        catch (const std::exception& e)
        {
            std::printf("FAIL %-18s exception: %s\n", slot.name, e.what());
            ++failed;
        }
    }

    // Analysis readouts do not mutate the buffer, so they sit outside the
    // in-place Slot loop. Exercise OnsetDetector on sine + a click and confirm
    // the readouts stay finite and the latency contract holds.
    try
    {
        OnsetDetector<float> onset;
        onset.prepare(spec);
        bool sawOnset = false;
        for (int b = 0; b < 40; ++b)
        {
            fillSine(buf, b * kBlockSize);
            if (b == 20) buf.getChannel(0)[0] += 4.0f; // a transient
            onset.processBlock(buf.toView());
            if (onset.onsetDetected()) sawOnset = true;
            const float st = onset.getOnsetStrength();
            if (std::isnan(st) || std::isinf(st))
            {
                std::printf("FAIL %-18s non-finite strength\n", "OnsetDetector");
                ++failed; break;
            }
        }
        if (onset.getLatencySamples() != onset.getFftSize() + onset.getHopSize())
        {
            std::printf("FAIL %-18s latency != fftSize+hop\n", "OnsetDetector");
            ++failed;
        }
        else
            std::printf("OK   %-18s (sine + transient, sawOnset=%d)\n",
                        "OnsetDetector", sawOnset ? 1 : 0);
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %-18s exception: %s\n", "OnsetDetector", e.what());
        ++failed;
    }

    // SpectralFreeze changes behavior on a control request, so the plain
    // Slot loop cannot cover its held state: run the live path, request the
    // hold mid-stream, and require finite bounded output in both states.
    try
    {
        SpectralFreeze<float> freeze;
        freeze.prepare(spec);
        bool freezeFailed = false;
        for (int b = 0; b < 24 && !freezeFailed; ++b)
        {
            if (b == 12) freeze.setFrozen(true);
            fillSine(buf, b * kBlockSize);
            freeze.processBlock(buf.toView());
            std::string reason;
            if (isUnsafe(buf, 4.0f, reason))
            {
                std::printf("FAIL %-18s block %d: %s\n", "SpectralFreeze", b,
                            reason.c_str());
                ++failed;
                freezeFailed = true;
            }
        }
        if (!freezeFailed && freeze.getLatency() != freeze.getFFTSize())
        {
            std::printf("FAIL %-18s latency != fftSize\n", "SpectralFreeze");
            ++failed;
            freezeFailed = true;
        }
        if (!freezeFailed)
            std::printf("OK   %-18s (live + frozen hold)\n", "SpectralFreeze");
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %-18s exception: %s\n", "SpectralFreeze", e.what());
        ++failed;
    }

    // LoopFinder is offline: one bounded search over a planted seam and one
    // transactional render, checked for status, shape and finite samples.
    try
    {
        constexpr int loopSamples = 4096;
        constexpr int plantedStart = 512;
        constexpr int plantedEnd = 2949;
        constexpr int contextLength = 256;
        std::vector<float> loopSource(loopSamples);
        for (int i = 0; i < loopSamples; ++i)
            loopSource[static_cast<size_t>(i)] = 0.4f
                * std::sin(2.0f * 3.14159265f * 0.019f * static_cast<float>(i))
                + 0.1f * std::sin(2.0f * 3.14159265f * 0.077f
                                  * static_cast<float>(i) + 0.3f);
        std::copy_n(loopSource.begin() + plantedStart, contextLength,
                    loopSource.begin() + plantedEnd - contextLength);

        LoopFinder<float> finder;
        LoopFinder<float>::Settings settings;
        settings.comparisonLength = 256;
        settings.crossfadeLength = 64;
        settings.coarseStride = 16;
        settings.refinementRadius = 15;
        const float* channelPointer = loopSource.data();
        AudioBufferView<const float> source(&channelPointer, 1, loopSamples);
        const auto loop = finder.find(source,
                                      { plantedStart - 64, plantedStart + 64 },
                                      { plantedEnd - 64, plantedEnd + 64 },
                                      2000, 2600, settings);
        std::vector<float> rendered;
        bool loopOk = static_cast<bool>(loop)
            && loop.start == plantedStart && loop.end == plantedEnd;
        if (loopOk)
        {
            rendered.assign(static_cast<size_t>(loop.renderedLength()), 0.0f);
            float* renderPointer = rendered.data();
            AudioBufferView<float> output(&renderPointer, 1,
                                          static_cast<int>(rendered.size()));
            loopOk = LoopFinder<float>::renderLoop(source, loop, output)
                == LoopFinder<float>::Status::Success;
            for (const float sample : rendered)
                if (!std::isfinite(sample) || std::abs(sample) > 1.0f)
                    loopOk = false;
        }
        if (!loopOk)
        {
            std::printf("FAIL %-18s find/render smoke\n", "LoopFinder");
            ++failed;
        }
        else
            std::printf("OK   %-18s (bounded find + render)\n", "LoopFinder");
    }
    catch (const std::exception& e)
    {
        std::printf("FAIL %-18s exception: %s\n", "LoopFinder", e.what());
        ++failed;
    }

#ifndef DSPARK_NO_FILE_IO
    // File codecs are exercised without a device. This smoke only pins their
    // public lifecycle; byte-exact corpora live in TestIO.
    MidiFile midi;
    FlacFile flac;
    if (!midi.create(0, 960) || midi.format() != 0
        || midi.ticksPerQuarter() != 960 || flac.isOpen()
        || FlacFile::kWriteUnsupportedReason.empty())
    {
        std::printf("FAIL %-18s public lifecycle\n", "MIDI/FLAC I/O");
        ++failed;
    }
    else
        std::printf("OK   %-18s (offline lifecycle)\n", "MIDI/FLAC I/O");
#endif

    std::printf("\n%d / %zu effects passed\n",
                static_cast<int>(slots.size()) - failed, slots.size());
    return failed;
}

} // namespace

int main()
{
    return testAll();
}
