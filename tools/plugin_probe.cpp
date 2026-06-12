// DSPark — host-contract probe plugin (test instrumentation, not an example)
//
// A deliberately bare plugin whose OUTPUT encodes what the wrapper delivered,
// so the smoke hosts can prove each host-facing capability end to end:
//
//   - gain          raw, unsmoothed multiply -> sample-accurate automation
//                   shows up as an exact step in the output
//   - transport     while playing, adds a DC of tempo/1000 (sign flips to
//                   negative under offline rendering)
//   - MIDI          one monophonic sine voice: note on/off (honouring the
//                   sample offset), pitch bend (+/-2 st), CC1 adds 0.1 DC
//   - lookahead     a toggle that flips getLatency() 0 <-> 64 so the smokes
//                   can watch the latency-changed notification (the probe
//                   does NOT actually delay: contract instrumentation only)
//   - presets       two factory presets ("Unity", "Half") differing in gain
//   - channels      mono+stereo (the default), every channel identical
//
// No DSPark processors and no smoothing on purpose: every behaviour above
// must be exactly measurable. Real plugins should smooth their parameters.

#include "../plugin/vst3/DSParkVst3.h"
#include "../plugin/clap/DSParkClap.h"

#include <atomic>
#include <cmath>

struct ProbePlugin
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark Probe",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.test.probe",
        .version   = "1.0.0",
    };

    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param ("gain",      "Gain",      0.0f, 1.0f, 1.0f, ""),
        dspark::plugin::toggle("lookahead", "Lookahead", false));

    static constexpr auto factoryPresets = dspark::plugin::presets(
        dspark::plugin::preset("Unity", 1.0f, 0.0f),
        dspark::plugin::preset("Half",  0.5f, 0.0f));

    void prepare(const dspark::AudioSpec& spec) { sampleRate_ = spec.sampleRate; }

    void setParameter(int index, float value) noexcept
    {
        if (index == 0) gain_.store(value, std::memory_order_relaxed);
        else if (index == 1) lookahead_.store(value >= 0.5f, std::memory_order_relaxed);
    }

    void setTransport(const dspark::plugin::TransportInfo& t) noexcept
    {
        tempo_.store(t.tempoValid ? t.tempoBpm : 0.0, std::memory_order_relaxed);
        playing_.store(t.playing, std::memory_order_relaxed);
    }

    void setOfflineRendering(bool offline) noexcept
    {
        offline_.store(offline, std::memory_order_relaxed);
    }

    void handleMidiEvent(const dspark::plugin::MidiEvent& ev) noexcept
    {
        using Type = dspark::plugin::MidiEvent::Type;
        switch (ev.type)
        {
        case Type::NoteOn:
            noteFreq_ = 440.0 * std::pow(2.0, (ev.note - 69) / 12.0);
            noteAmp_ = ev.value;
            noteDelay_ = ev.sampleOffset;
            phase_ = 0.0;
            break;
        case Type::NoteOff:
            noteAmp_ = 0.0f;
            break;
        case Type::PitchBend:
            bendSemis_ = ev.value * 2.0f;
            break;
        case Type::ControlChange:
            if (ev.note == 1) mod_ = ev.value;
            break;
        default:
            break;
        }
    }

    [[nodiscard]] int getLatency() const noexcept
    {
        return lookahead_.load(std::memory_order_relaxed) ? 64 : 0;
    }

    void processBlock(dspark::AudioBufferView<float> io) noexcept
    {
        const float gain = gain_.load(std::memory_order_relaxed);
        const double tempo = tempo_.load(std::memory_order_relaxed);
        const bool playing = playing_.load(std::memory_order_relaxed);
        const bool offline = offline_.load(std::memory_order_relaxed);
        const float dc = (playing ? static_cast<float>(tempo) / 1000.0f : 0.0f)
                       * (offline ? -1.0f : 1.0f)
                       + 0.1f * mod_;
        const double freq = noteFreq_ * std::pow(2.0, bendSemis_ / 12.0);
        const double phaseInc = 6.283185307179586 * freq / sampleRate_;

        const int n = io.getNumSamples();
        const int channels = io.getNumChannels();
        for (int i = 0; i < n; ++i)
        {
            float voice = 0.0f;
            if (noteDelay_ > 0)
                --noteDelay_;
            else if (noteAmp_ > 0.0f)
            {
                voice = noteAmp_ * static_cast<float>(std::sin(phase_));
                phase_ += phaseInc;
            }
            for (int ch = 0; ch < channels; ++ch)
            {
                float* data = io.getChannel(ch);
                data[i] = data[i] * gain + dc + voice;
            }
        }
    }

private:
    std::atomic<float> gain_ { 1.0f };
    std::atomic<bool>  lookahead_ { false };
    std::atomic<double> tempo_ { 0.0 };
    std::atomic<bool>  playing_ { false };
    std::atomic<bool>  offline_ { false };

    // Audio-thread-only voice state (MIDI lands on the audio thread).
    double noteFreq_ = 0.0;
    float  noteAmp_ = 0.0f;
    int    noteDelay_ = 0;
    float  bendSemis_ = 0.0f;
    float  mod_ = 0.0f;
    double phase_ = 0.0;
    double sampleRate_ = 48000.0;
};

DSPARK_VST3_PLUGIN(ProbePlugin)
DSPARK_CLAP_PLUGIN(ProbePlugin)
