// DSPark plugin template — every contract method, present and explained.
//
// This is the "kitchen sink" companion to docs/plugins.md: a working plugin
// that implements EVERY optional method the format wrappers can detect, with
// comments on when you actually want each one.
//
// It inherits dspark::plugin::PluginBase<T>: the base defines the WHOLE
// optional contract with safe defaults (latency 0, no tail, nothing to
// reset, no extra state), so the full menu of overridable methods is one
// Go-to-Definition away and your IDE autocompletes it — delete any override
// below and the plugin still builds, falling back to the default. A
// free-standing struct without the base works identically if you prefer
// (see examples/plugin_saturator/).
//
//   cl  /std:c++20 /O2 /LD /EHsc /I . plugin_template.cpp /Fe:MyPlugin.vst3
//   g++ -std=c++20 -O2 -fPIC -shared -I . plugin_template.cpp -o MyPlugin.vst3
//
// The same binary is also a CLAP plugin: copy it as MyPlugin.clap.

#include "../../plugin/vst3/DSParkVst3.h"
#include "../../plugin/clap/DSParkClap.h"

#include "../../Effects/Reverb.h"      // convolution reverb: latency + tail
#include "../../Effects/Limiter.h"     // lookahead: real latency to report
#include "../../Core/StateBlob.h"      // optional extra-state serializer

#include <cmath>

struct TemplatePlugin : dspark::plugin::PluginBase<TemplatePlugin>
{
    // =========================================================================
    // REQUIRED 1/5 — identity. productId is FOREVER: it derives the VST3
    // class UID and is the CLAP id. Changing it orphans every saved session.
    // =========================================================================
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name      = "DSPark Template",
        .vendor    = "DSPark",
        .url       = "https://github.com/CristianMoresi/DSPark",
        .email     = "mailto:dev@cristianmoresi.com",
        .productId = "com.dspark.examples.template",
        .version   = "1.0.0",
        .category  = dspark::plugin::Category::Fx,
    };

    // =========================================================================
    // REQUIRED 2/5 — the parameter table. The TEXT IDS are the stable
    // identity (state + automation). Reorder or insert freely in future
    // versions; never rename an id. Three kinds:
    //   param(...)            continuous, with unit for host display
    //   toggle(...)           on/off (hosts render a button)
    //   param(... ) + steps   discrete N+1 positions (set .steps yourself)
    // =========================================================================
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param ("mix",      "Mix",        0.0f,  1.0f, 0.3f, ""),
        dspark::plugin::param ("ceiling",  "Ceiling",  -24.0f,  0.0f, -1.0f, "dB"),
        dspark::plugin::toggle("truepeak", "True Peak", true),
        dspark::plugin::param ("lookahead","Lookahead",  0.5f, 10.0f,  5.0f, "ms"));

    // =========================================================================
    // OPTIONAL — factoryPresets. One PLAIN value per parameter, in table
    // order. Every format publishes them natively: a VST3 program list with
    // a program-change parameter, CLAP preset-load + preset-discovery, AU
    // factory presets — so the host's own preset browser offers them.
    // =========================================================================
    static constexpr auto factoryPresets = dspark::plugin::presets(
        dspark::plugin::preset("Subtle Glue", 0.2f, -1.0f, 1.0f, 5.0f),
        dspark::plugin::preset("Big Hall",    0.6f, -3.0f, 1.0f, 8.0f));

    // =========================================================================
    // REQUIRED 3/5 — prepare. Main thread, before audio, allocation allowed.
    // Maps to VST3 setActive(true) and CLAP activate.
    // =========================================================================
    void prepare(const dspark::AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;

        reverb_.prepare(spec);
        buildImpulseResponse();        // a small synthetic hall

        limiter_.prepare(spec);
        limiter_.setCeiling(-1.0f);
        limiter_.setTruePeak(true);
    }

    // =========================================================================
    // REQUIRED 4/5 — setParameter. CALLED FROM ANY THREAD (host automation
    // arrives on the audio thread, generic-UI edits on the main thread).
    // DSPark setters are atomic and smoothed by contract, so forwarding is
    // all you do here. Index = position in `parameters`. Values are PLAIN
    // (already in your declared min..max range).
    // =========================================================================
    void setParameter(int index, float value) noexcept
    {
        switch (index)
        {
        case 0: reverb_.setMix(value); break;
        case 1: limiter_.setCeiling(value); break;
        case 2: limiter_.setTruePeak(value >= 0.5f); break;
        case 3: limiter_.setLookahead(value); break;   // changes getLatency()!
        default: break;
        }
    }

    // =========================================================================
    // REQUIRED 5/5 — processBlock. Audio thread: no allocation, no locks.
    // Exactly the DSPark contract you already know.
    // =========================================================================
    void processBlock(dspark::AudioBufferView<float> io) noexcept
    {
        reverb_.processBlock(io);
        limiter_.processBlock(io);
    }

    // =========================================================================
    // OPTIONAL — getLatency. Implement when your chain delays the signal
    // (lookahead limiter, linear-phase EQ, oversampling, FFT processing).
    // The host shifts every other track to compensate; report it accurately
    // or your users hear phasing on parallel paths. Read after prepare()
    // AND re-read after parameter changes: when the value moves (the
    // lookahead knob above), the wrapper notifies the host on its own
    // (restartComponent(kLatencyChanged) / clap_host_latency / the AU
    // Latency property listeners) so the project re-compensates.
    // =========================================================================
    [[nodiscard]] int getLatency() const noexcept
    {
        return reverb_.getLatency() + limiter_.getLatency();
    }

    // =========================================================================
    // OPTIONAL — setTransport. The host timeline lands here once per block
    // (audio thread): tempo, musical position, play state, loop points.
    // This demo only stores it; a tempo-synced delay would derive its time
    // from transport.samplesPerBeat(sampleRate). Check the *Valid flags —
    // not every host provides every field.
    // =========================================================================
    void setTransport(const dspark::plugin::TransportInfo& transport) noexcept
    {
        lastTransport_ = transport;
    }

    // =========================================================================
    // OPTIONAL — setOfflineRendering. Hosts flip it for non-realtime
    // bounces: the moment to raise quality/cost trade-offs (oversampling,
    // longer lookahead). Called outside the audio thread.
    // =========================================================================
    void setOfflineRendering(bool offline) noexcept
    {
        offlineRender_ = offline;
    }

    // =========================================================================
    // OPTIONAL — getTailSeconds. Implement when sound continues after the
    // input stops (reverbs, delays): hosts keep processing you that long
    // instead of cutting the tail. Maps to VST3 getTailSamples and the CLAP
    // tail extension.
    // =========================================================================
    [[nodiscard]] double getTailSeconds() const noexcept
    {
        return kIrSeconds;             // our impulse response length
    }

    // =========================================================================
    // OPTIONAL — reset. Implement when you keep history that should clear
    // on transport jumps (delay lines, envelopes, reverb tails). Invoked by
    // CLAP hosts; VST3 hosts re-activate (prepare runs again) instead.
    // =========================================================================
    void reset() noexcept
    {
        reverb_.reset();
        limiter_.reset();
    }

    // =========================================================================
    // OPTIONAL — getState/setState. The wrapper ALREADY saves and restores
    // every parameter by its stable id — most plugins should delete this
    // pair. Implement it only for state BEYOND the parameters: learned noise
    // profiles, user-loaded IRs, editor layout... DSPark's StateBlob gives
    // you versioned, tolerant serialization for free. This demo persists one
    // non-parameter value to show the mechanics.
    // =========================================================================
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        dspark::StateWriter w(dspark::stateId("XMPL"), 1);
        w.write("irSeed", static_cast<int>(irSeed_));
        return w.blob();
    }

    bool setState(const uint8_t* data, size_t size)
    {
        dspark::StateReader r(data, size);
        if (!r.isValid() || r.processorId() != dspark::stateId("XMPL"))
            return false;
        irSeed_ = static_cast<uint32_t>(r.read("irSeed", 0x1234567));
        buildImpulseResponse();        // rebuild dependent state from it
        return true;
    }

    // OPTIONAL — hasEditor: inherited from PluginBase as false. Hosts show
    // their generic parameter UI. The WebView editor layer will claim this
    // hook; see docs/plugins.md.

private:
    static constexpr double kIrSeconds = 1.5;

    void buildImpulseResponse()
    {
        const int n = static_cast<int>(kIrSeconds * sampleRate_);
        ir_.assign(static_cast<size_t>(n), 0.0f);
        uint32_t rng = irSeed_;
        for (int i = 0; i < n; ++i)
        {
            rng = rng * 1664525u + 1013904223u;
            const float noise = static_cast<float>(rng >> 8) / 8388608.0f - 1.0f;
            const float decay = std::exp(-4.0f * static_cast<float>(i) / static_cast<float>(n));
            ir_[static_cast<size_t>(i)] = noise * decay * 0.25f;
        }
        ir_[0] = 1.0f;
        reverb_.loadIR(ir_.data(), n, sampleRate_);
    }

    dspark::Reverb<float>  reverb_;
    dspark::Limiter<float> limiter_;
    std::vector<float>     ir_;
    double                 sampleRate_ = 48000.0;
    uint32_t               irSeed_ = 0x1234567u;
    dspark::plugin::TransportInfo lastTransport_ {};
    bool                   offlineRender_ = false;
};

DSPARK_VST3_PLUGIN(TemplatePlugin)
DSPARK_CLAP_PLUGIN(TemplatePlugin)
