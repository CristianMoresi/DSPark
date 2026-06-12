// DSPark instrument example — a polyphonic synthesizer in one file.
//
// The canonical Category::Instrument reference: no audio input in any
// format (VST3 instrument class, CLAP "instrument" feature, AU `aumu`
// music device), MIDI in through handleMidiEvent, eight voices built from
// DSPark's Oscillator + EnvelopeGenerator, factory presets, and host
// transport received per block (printed nowhere — wired so the pattern is
// on display; a tempo-synced LFO would read it the same way).
//
// Voices ADD into `io`: the wrapper hands an instrument a cleared buffer.
//
//   cl  /std:c++20 /O2 /LD /EHsc /I . synth.cpp /Fe:DSParkSynth.vst3
//   g++ -std=c++20 -O2 -fPIC -shared -I . synth.cpp -o DSParkSynth.vst3
//
// The same binary is also a CLAP plugin (copy it as DSParkSynth.clap) and,
// on macOS, an `aumu` Audio Unit (see au/Info.plist).

#include "../../plugin/vst3/DSParkVst3.h"
#include "../../plugin/clap/DSParkClap.h"
#include "../../plugin/au/DSParkAu.h"

#include "../../Core/Oscillator.h"
#include "../../Core/EnvelopeGenerator.h"

#include <atomic>
#include <cmath>

struct SynthPlugin : dspark::plugin::PluginBase<SynthPlugin>
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark Synth",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.examples.synth",
        .version   = "1.0.0",
        .category  = dspark::plugin::Category::Instrument,
    };

    static constexpr auto parameters = dspark::plugin::params(
        // 3-position discrete control (steps = 2): Sine / Saw / Square.
        dspark::plugin::Param { "wave", "Waveform", 0.0f, 2.0f, 1.0f, "", 2 },
        dspark::plugin::param("attack",  "Attack",   1.0f,  500.0f,    5.0f, "ms"),
        dspark::plugin::param("decay",   "Decay",    1.0f, 1000.0f,  120.0f, "ms"),
        dspark::plugin::param("sustain", "Sustain",  0.0f,    1.0f,    0.7f, ""),
        dspark::plugin::param("release", "Release",  1.0f, 2000.0f,  150.0f, "ms"),
        dspark::plugin::param("gain",    "Gain",   -24.0f,    6.0f,   -6.0f, "dB"));

    static constexpr auto factoryPresets = dspark::plugin::presets(
        dspark::plugin::preset("Init",  1.0f,   5.0f, 120.0f, 0.7f,  150.0f, -6.0f),
        dspark::plugin::preset("Pluck", 1.0f,   1.0f, 180.0f, 0.0f,   80.0f, -6.0f),
        dspark::plugin::preset("Pad",   0.0f, 250.0f, 400.0f, 0.8f, 1200.0f, -9.0f));

    // The synth output is inherently stereo here (both channels identical);
    // a mono configuration adds nothing, so restrict the negotiation.
    static constexpr auto channels = dspark::plugin::ChannelSupport::StereoOnly;

    void prepare(const dspark::AudioSpec& spec)
    {
        scratch_.assign(static_cast<size_t>(spec.maxBlockSize), 0.0f);
        for (auto& voice : voices_)
        {
            voice.osc.prepare(spec.sampleRate);
            voice.env.prepare(spec.sampleRate);
            voice.note = -1;
            voice.delay = 0;
        }
        applyEnvelope();
    }

    void setParameter(int index, float value) noexcept
    {
        switch (index)
        {
        case 0: wave_.store(static_cast<int>(value + 0.5f),
                            std::memory_order_relaxed); break;
        case 1: attackMs_.store(value, std::memory_order_relaxed);  break;
        case 2: decayMs_.store(value, std::memory_order_relaxed);   break;
        case 3: sustain_.store(value, std::memory_order_relaxed);   break;
        case 4: releaseMs_.store(value, std::memory_order_relaxed); break;
        case 5: gainDb_.store(value, std::memory_order_relaxed);    break;
        default: break;
        }
    }

    void handleMidiEvent(const dspark::plugin::MidiEvent& ev) noexcept
    {
        using Type = dspark::plugin::MidiEvent::Type;
        switch (ev.type)
        {
        case Type::NoteOn:
        {
            Voice& voice = allocateVoice();
            voice.note = ev.note;
            voice.velocity = ev.value;
            voice.delay = ev.sampleOffset;   // sample-accurate start
            voice.osc.setWaveform(waveformOf(
                wave_.load(std::memory_order_relaxed)));
            voice.osc.setFrequency(noteFrequency(ev.note));
            voice.osc.reset();
            applyEnvelopeTo(voice);
            voice.env.noteOn();
            break;
        }
        case Type::NoteOff:
            for (auto& voice : voices_)
                if (voice.note == ev.note && voice.env.isActive())
                    voice.env.noteOff();
            break;
        case Type::PitchBend:
            bendSemis_ = ev.value * 2.0f;
            for (auto& voice : voices_)
                if (voice.env.isActive())
                    voice.osc.setFrequency(noteFrequency(voice.note));
            break;
        default:
            break;
        }
    }

    // Wired to show the pattern: a tempo-synced voice LFO would read
    // transport_.samplesPerBeat() here. See docs/plugins.md.
    void setTransport(const dspark::plugin::TransportInfo& t) noexcept
    {
        transport_ = t;
    }

    [[nodiscard]] double getTailSeconds() const noexcept
    {
        return releaseMs_.load(std::memory_order_relaxed) / 1000.0;
    }

    void reset() noexcept
    {
        for (auto& voice : voices_)
        {
            voice.env.reset();
            voice.note = -1;
        }
    }

    void processBlock(dspark::AudioBufferView<float> io) noexcept
    {
        const int n = io.getNumSamples();
        if (n <= 0 || static_cast<size_t>(n) > scratch_.size()) return;
        const float gain = std::pow(10.0f,
            gainDb_.load(std::memory_order_relaxed) / 20.0f);

        for (auto& voice : voices_)
        {
            if (!voice.env.isActive()) continue;
            voice.osc.processBlock(scratch_.data(), static_cast<size_t>(n));
            float* left = io.getChannel(0);
            float* right = io.getNumChannels() > 1 ? io.getChannel(1) : left;
            for (int i = 0; i < n; ++i)
            {
                if (voice.delay > 0)
                {
                    --voice.delay;
                    continue;
                }
                const float sample = scratch_[static_cast<size_t>(i)]
                                   * voice.env.getNextValue()
                                   * voice.velocity * gain;
                left[i] += sample;
                if (right != left) right[i] += sample;
            }
        }
    }

private:
    struct Voice
    {
        dspark::Oscillator<float> osc;
        dspark::ADSREnvelope<float> env;
        int   note = -1;
        float velocity = 0.0f;
        int   delay = 0;       // frames until the voice starts (sample offset)
        uint32_t age = 0;      // allocation order for voice stealing
    };

    static dspark::Oscillator<float>::Waveform waveformOf(int index) noexcept
    {
        using Waveform = dspark::Oscillator<float>::Waveform;
        return index <= 0 ? Waveform::Sine
             : (index == 1 ? Waveform::Saw : Waveform::Square);
    }

    [[nodiscard]] float noteFrequency(int note) const noexcept
    {
        return 440.0f * std::pow(2.0f,
            (static_cast<float>(note) - 69.0f + bendSemis_) / 12.0f);
    }

    Voice& allocateVoice() noexcept
    {
        ++ageCounter_;
        Voice* best = &voices_[0];
        for (auto& voice : voices_)
        {
            if (!voice.env.isActive())
            {
                voice.age = ageCounter_;
                return voice;
            }
            if (voice.age < best->age) best = &voice;   // steal the oldest
        }
        best->age = ageCounter_;
        return *best;
    }

    void applyEnvelope() noexcept
    {
        for (auto& voice : voices_) applyEnvelopeTo(voice);
    }

    void applyEnvelopeTo(Voice& voice) noexcept
    {
        voice.env.setParameters(attackMs_.load(std::memory_order_relaxed),
                                decayMs_.load(std::memory_order_relaxed),
                                sustain_.load(std::memory_order_relaxed),
                                releaseMs_.load(std::memory_order_relaxed));
    }

    static constexpr int kNumVoices = 8;
    std::array<Voice, kNumVoices> voices_ {};
    std::vector<float> scratch_;
    dspark::plugin::TransportInfo transport_ {};
    float bendSemis_ = 0.0f;
    uint32_t ageCounter_ = 0;

    std::atomic<int>   wave_ { 1 };
    std::atomic<float> attackMs_ { 5.0f };
    std::atomic<float> decayMs_ { 120.0f };
    std::atomic<float> sustain_ { 0.7f };
    std::atomic<float> releaseMs_ { 150.0f };
    std::atomic<float> gainDb_ { -6.0f };
};

DSPARK_VST3_PLUGIN(SynthPlugin)
DSPARK_CLAP_PLUGIN(SynthPlugin)
DSPARK_AU_PLUGIN(SynthPlugin, "DSsy", "DSpk")
