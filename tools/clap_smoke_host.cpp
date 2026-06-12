// DSPark — CLAP smoke host
//
// Loads a .clap module through the public C ABI like a DAW would and drives
// the full lifecycle: entry -> factory -> descriptor -> instantiation ->
// extensions -> activate -> process (sine blocks + a live parameter event) ->
// state save/perturb/restore -> teardown. Exit code 0 means every step
// behaved. Companion to tools/vst3_smoke_host.cpp; clap-validator is the
// recommended final gate.
//
//   clap_smoke_host <module.clap> [--expect-sidechain | --expect-instrument | --probe]
//
//   --expect-sidechain   sidechain port present + the key ducks the output
//   --expect-instrument  no audio inputs, a note port; a note must sound at
//                        the right pitch and decay
//   --probe              full host-contract battery against the DSPark probe
//                        plugin: transport DC, the render extension's offline
//                        sign flip, sample-accurate automation, note events
//                        and raw-MIDI pitch bend, the latency-changed
//                        notification, the mono ports configuration and
//                        preset-load + preset-discovery

#include "../plugin/clap/clap/clap.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
using ModuleHandle = HMODULE;
static ModuleHandle openModule(const char* p) { return LoadLibraryA(p); }
static void* moduleSym(ModuleHandle m, const char* n)
{ return reinterpret_cast<void*>(GetProcAddress(m, n)); }
#else
#include <dlfcn.h>
using ModuleHandle = void*;
static ModuleHandle openModule(const char* p) { return dlopen(p, RTLD_NOW); }
static void* moduleSym(ModuleHandle m, const char* n) { return dlsym(m, n); }
#endif

namespace {

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "PASS " : "FAIL ", what);
    if (!ok) ++g_failures;
}

/// Same FNV-1a the wrapper uses for its stable parameter ids.
constexpr uint32_t hash32(const char* s) noexcept
{
    uint32_t h = 2166136261u;
    while (*s != '\0')
    {
        h ^= static_cast<uint8_t>(*s++);
        h *= 16777619u;
    }
    return h;
}

// -- host with the extensions the wrapper may call back into -----------------------

struct SmokeHost
{
    clap_host_t host {};
    const clap_plugin_t* plugin = nullptr;   // for request_callback dispatch
    bool latencyChangedSeen = false;
    bool presetLoadedSeen = false;
    bool callbackRequested = false;

    static SmokeHost* self(const clap_host_t* h)
    {
        return static_cast<SmokeHost*>(h->host_data);
    }

    static void sLatencyChanged(const clap_host_t* h)
    {
        self(h)->latencyChangedSeen = true;
    }
    static void sParamsRescan(const clap_host_t*, clap_param_rescan_flags) {}
    static void sParamsClear(const clap_host_t*, clap_id, clap_param_clear_flags) {}
    static void sParamsRequestFlush(const clap_host_t*) {}
    static void sPresetOnError(const clap_host_t*, uint32_t, const char*,
                               const char*, int32_t, const char*) {}
    static void sPresetLoaded(const clap_host_t* h, uint32_t, const char*,
                              const char*)
    {
        self(h)->presetLoadedSeen = true;
    }

    inline static const clap_host_latency_t kLatency = { &sLatencyChanged };
    inline static const clap_host_params_t kParams = {
        &sParamsRescan, &sParamsClear, &sParamsRequestFlush
    };
    inline static const clap_host_preset_load_t kPresetLoad = {
        &sPresetOnError, &sPresetLoaded
    };

    static const void* sGetExtension(const clap_host_t*, const char* id)
    {
        if (std::strcmp(id, CLAP_EXT_LATENCY) == 0) return &kLatency;
        if (std::strcmp(id, CLAP_EXT_PARAMS) == 0) return &kParams;
        if (std::strcmp(id, CLAP_EXT_PRESET_LOAD) == 0
            || std::strcmp(id, CLAP_EXT_PRESET_LOAD_COMPAT) == 0)
            return &kPresetLoad;
        return nullptr;
    }
    static void sRequestRestart(const clap_host_t*) {}
    static void sRequestProcess(const clap_host_t*) {}
    static void sRequestCallback(const clap_host_t* h)
    {
        // The smoke IS the main thread: dispatch immediately.
        auto* s = self(h);
        s->callbackRequested = true;
        if (s->plugin != nullptr && s->plugin->on_main_thread != nullptr)
            s->plugin->on_main_thread(s->plugin);
    }

    SmokeHost()
    {
        host.clap_version = CLAP_VERSION_INIT;
        host.host_data = this;
        host.name = "DSPark smoke host";
        host.vendor = "DSPark";
        host.url = "https://github.com/CristianMoresi/DSPark";
        host.version = "1.0";
        host.get_extension = &sGetExtension;
        host.request_restart = &sRequestRestart;
        host.request_process = &sRequestProcess;
        host.request_callback = &sRequestCallback;
    }
};

// -- event list over heterogeneous event storage --------------------------------------

struct EventList
{
    clap_input_events_t iface;
    struct Slot
    {
        alignas(8) unsigned char bytes[64];
    };
    std::vector<Slot> slots;

    static uint32_t sSize(const clap_input_events_t* l)
    {
        return static_cast<uint32_t>(static_cast<const EventList*>(l->ctx)->slots.size());
    }
    static const clap_event_header_t* sGet(const clap_input_events_t* l, uint32_t i)
    {
        const auto* self = static_cast<const EventList*>(l->ctx);
        return i < self->slots.size()
             ? reinterpret_cast<const clap_event_header_t*>(self->slots[i].bytes)
             : nullptr;
    }
    EventList()
    {
        iface.ctx = this;
        iface.size = &sSize;
        iface.get = &sGet;
    }

    template <typename E>
    void push(const E& e)
    {
        static_assert(sizeof(E) <= sizeof(Slot::bytes));
        Slot slot {};
        std::memcpy(slot.bytes, &e, sizeof(E));
        slots.push_back(slot);
    }
    void clear() { slots.clear(); }

    void paramValue(uint32_t paramId, double value, uint32_t time)
    {
        clap_event_param_value_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_PARAM_VALUE;
        ev.param_id = paramId;
        ev.note_id = -1;
        ev.port_index = -1;
        ev.channel = -1;
        ev.key = -1;
        ev.value = value;
        push(ev);
    }
    void noteOn(int16_t key, double velocity, uint32_t time)
    {
        clap_event_note_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_NOTE_ON;
        ev.note_id = -1;
        ev.port_index = 0;
        ev.channel = 0;
        ev.key = key;
        ev.velocity = velocity;
        push(ev);
    }
    void noteOff(int16_t key, uint32_t time)
    {
        clap_event_note_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_NOTE_OFF;
        ev.note_id = -1;
        ev.port_index = 0;
        ev.channel = 0;
        ev.key = key;
        ev.velocity = 0.0;
        push(ev);
    }
    void midi(uint8_t b0, uint8_t b1, uint8_t b2, uint32_t time)
    {
        clap_event_midi_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_MIDI;
        ev.port_index = 0;
        ev.data[0] = b0;
        ev.data[1] = b1;
        ev.data[2] = b2;
        push(ev);
    }
};

bool outTryPush(const clap_output_events_t*, const clap_event_header_t*) { return true; }
const clap_output_events_t kOutEvents = { nullptr, &outTryPush };

// -- byte-vector streams ------------------------------------------------------------

struct ByteOStream
{
    clap_ostream_t iface;
    std::vector<uint8_t> bytes;
    static int64_t sWrite(const clap_ostream_t* s, const void* buf, uint64_t n)
    {
        auto* self = static_cast<ByteOStream*>(s->ctx);
        const auto* p = static_cast<const uint8_t*>(buf);
        self->bytes.insert(self->bytes.end(), p, p + n);
        return static_cast<int64_t>(n);
    }
    ByteOStream() { iface.ctx = this; iface.write = &sWrite; }
};

struct ByteIStream
{
    clap_istream_t iface;
    std::vector<uint8_t> bytes;
    size_t pos = 0;
    static int64_t sRead(const clap_istream_t* s, void* buf, uint64_t n)
    {
        auto* self = static_cast<ByteIStream*>(s->ctx);
        const uint64_t left = self->bytes.size() - self->pos;
        const uint64_t take = left < n ? left : n;
        if (take > 0)
        {
            std::memcpy(buf, self->bytes.data() + self->pos, take);
            self->pos += take;
        }
        return static_cast<int64_t>(take);
    }
    ByteIStream() { iface.ctx = this; iface.read = &sRead; }
};

/// Zero-crossing pitch estimate over a mono buffer (clean tones only).
double estimateFrequency(const float* x, int n, double sampleRate)
{
    int crossings = 0;
    for (int i = 1; i < n; ++i)
        if ((x[i - 1] < 0.0f && x[i] >= 0.0f) || (x[i - 1] >= 0.0f && x[i] < 0.0f))
            ++crossings;
    return crossings * sampleRate / (2.0 * n);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: clap_smoke_host <module.clap> "
                    "[--expect-sidechain | --expect-instrument | --probe]\n");
        return 2;
    }
    const bool expectSidechain = argc >= 3
        && std::strcmp(argv[2], "--expect-sidechain") == 0;
    const bool expectInstrument = argc >= 3
        && std::strcmp(argv[2], "--expect-instrument") == 0;
    const bool probeMode = argc >= 3 && std::strcmp(argv[2], "--probe") == 0;

    ModuleHandle mod = openModule(argv[1]);
    expect(mod != nullptr, "module loads");
    if (!mod) return 1;

    const auto* entry = static_cast<const clap_plugin_entry_t*>(moduleSym(mod, "clap_entry"));
    expect(entry != nullptr, "clap_entry exported");
    if (!entry) return 1;

    expect(entry->init(argv[1]), "entry init");

    const auto* factory = static_cast<const clap_plugin_factory_t*>(
        entry->get_factory(CLAP_PLUGIN_FACTORY_ID));
    expect(factory != nullptr, "plugin factory");
    if (!factory) return 1;

    expect(factory->get_plugin_count(factory) == 1, "one plugin exported");
    const clap_plugin_descriptor_t* desc = factory->get_plugin_descriptor(factory, 0);
    expect(desc != nullptr && desc->id != nullptr, "descriptor");
    std::printf("      plugin: %s (%s) by %s\n", desc->name, desc->id, desc->vendor);

    SmokeHost smokeHost;
    const clap_plugin_t* plugin = factory->create_plugin(factory, &smokeHost.host,
                                                         desc->id);
    expect(plugin != nullptr, "create_plugin");
    if (!plugin) return 1;
    smokeHost.plugin = plugin;
    expect(plugin->init(plugin), "plugin init");

    const auto* ports = static_cast<const clap_plugin_audio_ports_t*>(
        plugin->get_extension(plugin, CLAP_EXT_AUDIO_PORTS));
    expect(ports != nullptr, "audio-ports extension");
    const uint32_t expectedInPorts =
        expectInstrument ? 0u : (expectSidechain ? 2u : 1u);
    if (ports)
    {
        expect(ports->count(plugin, true) == expectedInPorts,
               expectInstrument ? "instrument has no audio input ports"
               : (expectSidechain ? "main + sidechain input ports"
                                  : "one audio input port"));
        if (!expectInstrument)
        {
            clap_audio_port_info_t info {};
            expect(ports->get(plugin, 0, true, &info) && info.channel_count == 2,
                   "stereo input port");
        }
        if (expectSidechain)
        {
            clap_audio_port_info_t aux {};
            expect(ports->get(plugin, 1, true, &aux) && aux.channel_count == 2
                       && (aux.flags & CLAP_AUDIO_PORT_IS_MAIN) == 0,
                   "sidechain port is stereo and not main");
        }
    }
    if (expectInstrument || probeMode)
    {
        const auto* notePorts = static_cast<const clap_plugin_note_ports_t*>(
            plugin->get_extension(plugin, CLAP_EXT_NOTE_PORTS));
        expect(notePorts != nullptr && notePorts->count(plugin, true) == 1,
               "one note input port");
        if (notePorts != nullptr)
        {
            clap_note_port_info_t info {};
            expect(notePorts->get(plugin, 0, true, &info)
                       && (info.supported_dialects & CLAP_NOTE_DIALECT_CLAP) != 0
                       && (info.supported_dialects & CLAP_NOTE_DIALECT_MIDI) != 0,
                   "note port accepts CLAP and MIDI dialects");
        }
    }

    const auto* params = static_cast<const clap_plugin_params_t*>(
        plugin->get_extension(plugin, CLAP_EXT_PARAMS));
    expect(params != nullptr, "params extension");
    uint32_t numParams = params ? params->count(plugin) : 0;
    expect(numParams >= 1, "has parameters");
    std::printf("      params: %u (incl. bypass)\n", numParams);

    bool sawBypass = false;
    clap_param_info_t p0 {};
    for (uint32_t i = 0; i < numParams; ++i)
    {
        clap_param_info_t pi {};
        if (!params->get_info(plugin, i, &pi)) { expect(false, "get_info"); break; }
        if (pi.flags & CLAP_PARAM_IS_BYPASS) sawBypass = true;
        if (i == 0) p0 = pi;
    }
    expect(sawBypass, "bypass parameter present");

    const auto* state = static_cast<const clap_plugin_state_t*>(
        plugin->get_extension(plugin, CLAP_EXT_STATE));
    expect(state != nullptr, "state extension");
    const auto* latency = static_cast<const clap_plugin_latency_t*>(
        plugin->get_extension(plugin, CLAP_EXT_LATENCY));
    expect(latency != nullptr, "latency extension");

    expect(plugin->activate(plugin, 48000.0, 32, 512), "activate");
    expect(plugin->start_processing(plugin), "start_processing");

    std::vector<float> inL(512), inR(512), outL(512), outR(512);
    std::vector<float> scL(512, 0.0f), scR(512, 0.0f);
    float* inCh[2]  = { inL.data(), inR.data() };
    float* outCh[2] = { outL.data(), outR.data() };
    float* scCh[2]  = { scL.data(), scR.data() };

    clap_audio_buffer_t inBufs[2] {};
    inBufs[0].data32 = inCh;
    inBufs[0].channel_count = 2;
    inBufs[1].data32 = scCh;
    inBufs[1].channel_count = 2;
    clap_audio_buffer_t outBuf {};
    outBuf.data32 = outCh;
    outBuf.channel_count = 2;

    EventList events;
    clap_event_transport_t transport {};
    transport.header.size = sizeof(transport);
    transport.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
    transport.header.type = CLAP_EVENT_TRANSPORT;

    clap_process_t proc {};
    proc.steady_time = -1;
    proc.frames_count = 512;
    proc.audio_inputs = expectedInPorts > 0 ? inBufs : nullptr;
    proc.audio_outputs = &outBuf;
    proc.audio_inputs_count = expectedInPorts;
    proc.audio_outputs_count = 1;
    proc.in_events = &events.iface;
    proc.out_events = &kOutEvents;

    auto processOnce = [&]() {
        const bool ok = plugin->process(plugin, &proc) == CLAP_PROCESS_CONTINUE;
        events.clear();
        return ok;
    };

    bool finite = true;
    int n = 0;
    // Runs the 1 kHz test tone with the sidechain fed `scLevel` (0 = silent
    // key); fires the mid-run parameter event only on the first pass.
    auto runBlocks = [&](int blocks, float scLevel, bool fireEvent) -> double {
        double settled = 0.0;
        for (int block = 0; block < blocks; ++block)
        {
            for (int i = 0; i < 512; ++i, ++n)
            {
                inL[static_cast<size_t>(i)] = 0.25f
                    * std::sin(2.0f * 3.14159265f * 1000.0f
                               * static_cast<float>(n) / 48000.0f);
                inR[static_cast<size_t>(i)] = inL[static_cast<size_t>(i)];
                const float key = scLevel
                    * std::sin(2.0f * 3.14159265f * 200.0f
                               * static_cast<float>(n) / 48000.0f);
                scL[static_cast<size_t>(i)] = key;
                scR[static_cast<size_t>(i)] = key;
            }
            if (fireEvent && block == 20)   // live parameter event mid-run
                events.paramValue(p0.id, 0.5 * (p0.min_value + p0.max_value), 0);
            if (!processOnce())
            {
                expect(false, "process returns CONTINUE");
                break;
            }
            for (int i = 0; i < 512; ++i)
            {
                const float v = outL[static_cast<size_t>(i)];
                if (!std::isfinite(v)) finite = false;
                if (block >= blocks / 2) settled += static_cast<double>(v) * v;
            }
        }
        return settled;
    };

    if (!expectInstrument)
    {
        const double energy = runBlocks(40, 0.0f, true);
        expect(finite, "output stays finite");
        expect(energy > 1e-6, "output carries signal");

        if (expectSidechain)
        {
            // The functional proof: a hot key on the sidechain port must duck
            // the main output well below the silent-key level measured above.
            const double ducked = runBlocks(40, 0.9f, false);
            std::printf("      sidechain: open %.3g -> ducked %.3g\n", energy, ducked);
            expect(finite, "ducked output stays finite");
            expect(ducked < energy * 0.5, "hot sidechain ducks the main output");
            runBlocks(20, 0.0f, false);   // let the detector release
        }

        double mid = 0.0;
        expect(params->get_value(plugin, p0.id, &mid), "get_value");
        expect(std::fabs(mid - 0.5 * (p0.min_value + p0.max_value)) < 1e-6,
               "parameter event applied");
    }

    auto fillInput = [&](float value) {
        for (int i = 0; i < 512; ++i)
        {
            inL[static_cast<size_t>(i)] = value;
            inR[static_cast<size_t>(i)] = value;
        }
    };
    auto outMean = [&]() {
        double sum = 0.0;
        for (int i = 0; i < 512; ++i) sum += outL[static_cast<size_t>(i)];
        return sum / 512.0;
    };

    if (expectInstrument)
    {
        // A note must sound at the right pitch, then decay on release.
        // Select the sine waveform first for a clean zero-crossing estimate.
        const auto* paramsExt = params;
        clap_event_param_value_t wave {};
        (void) wave;
        events.paramValue(hash32("wave"), 0.0, 0);   // plain 0 = Sine
        events.noteOn(69, 1.0, 0);
        expect(processOnce(), "process with a note on");
        double noteEnergy = 0.0;
        std::vector<float> tone;
        for (int block = 0; block < 8; ++block)
        {
            processOnce();
            for (int i = 0; i < 512; ++i)
            {
                noteEnergy += static_cast<double>(outL[static_cast<size_t>(i)])
                            * outL[static_cast<size_t>(i)];
                tone.push_back(outL[static_cast<size_t>(i)]);
            }
        }
        expect(noteEnergy > 1e-4, "note produces sound");
        const double freq = estimateFrequency(tone.data(),
                                              static_cast<int>(tone.size()), 48000.0);
        std::printf("      note pitch: %.1f Hz (expected 440)\n", freq);
        expect(std::fabs(freq - 440.0) < 22.0, "note plays at the right pitch");

        events.noteOff(69, 0);
        processOnce();
        for (int block = 0; block < 90; ++block) processOnce();   // ~1 s release
        double tailEnergy = 0.0;
        processOnce();
        for (int i = 0; i < 512; ++i)
            tailEnergy += static_cast<double>(outL[static_cast<size_t>(i)])
                        * outL[static_cast<size_t>(i)];
        expect(tailEnergy < noteEnergy / 100.0, "note decays after note off");
        (void) paramsExt;
    }

    if (probeMode)
    {
        std::printf("      -- probe: host contract battery --\n");
        const uint32_t gainId = hash32("gain");
        const uint32_t lookaheadId = hash32("lookahead");

        // 1) Transport: while playing, the probe outputs DC = tempo/1000.
        fillInput(0.0f);
        transport.flags = CLAP_TRANSPORT_HAS_TEMPO | CLAP_TRANSPORT_IS_PLAYING;
        transport.tempo = 137.5;
        proc.transport = &transport;
        processOnce();
        processOnce();
        std::printf("      transport DC: %.4f (expected 0.1375)\n", outMean());
        expect(std::fabs(outMean() - 0.1375) < 1e-3,
               "transport tempo reaches the plugin (DC = tempo/1000)");

        // 2) Render extension: offline flips the probe's DC sign.
        const auto* render = static_cast<const clap_plugin_render_t*>(
            plugin->get_extension(plugin, CLAP_EXT_RENDER));
        expect(render != nullptr, "render extension present");
        if (render != nullptr)
        {
            expect(!render->has_hard_realtime_requirement(plugin),
                   "no hard realtime requirement");
            expect(render->set(plugin, CLAP_RENDER_OFFLINE), "set offline mode");
            processOnce();
            processOnce();
            expect(std::fabs(outMean() + 0.1375) < 1e-3,
                   "offline render mode reaches the plugin (DC sign flips)");
            render->set(plugin, CLAP_RENDER_REALTIME);
        }

        // 3) Sample-accurate automation: a gain step at offset 256 must land
        //    exactly there. Stop the transport (kept valid) first.
        transport.flags = CLAP_TRANSPORT_HAS_TEMPO;
        processOnce();
        fillInput(1.0f);
        events.paramValue(gainId, 0.0, 0);
        processOnce();   // settle at gain 0
        events.paramValue(gainId, 1.0, 256);
        processOnce();
        int stepAt = -1;
        for (int i = 0; i < 512; ++i)
            if (outL[static_cast<size_t>(i)] > 0.5f)
            {
                stepAt = i;
                break;
            }
        std::printf("      automation step lands at sample %d (expected 256)\n", stepAt);
        expect(stepAt == 256, "automation step is sample-accurate");

        // 4) Note events: silence the input path, play A4 with an in-block
        //    offset, check the start position and the pitch.
        fillInput(0.0f);
        events.paramValue(gainId, 0.0, 0);
        events.noteOn(69, 1.0, 256);
        processOnce();
        double firstHalf = 0.0, secondHalf = 0.0;
        for (int i = 0; i < 256; ++i)
            firstHalf += std::fabs(outL[static_cast<size_t>(i)]);
        for (int i = 256; i < 512; ++i)
            secondHalf += std::fabs(outL[static_cast<size_t>(i)]);
        expect(firstHalf < 1e-6 && secondHalf > 0.1,
               "note starts at its sample offset");
        std::vector<float> tone;
        for (int block = 0; block < 8; ++block)
        {
            processOnce();
            for (int i = 0; i < 512; ++i)
                tone.push_back(outL[static_cast<size_t>(i)]);
        }
        const double freq = estimateFrequency(tone.data(),
                                              static_cast<int>(tone.size()), 48000.0);
        std::printf("      note pitch: %.1f Hz (expected 440)\n", freq);
        expect(std::fabs(freq - 440.0) < 22.0, "note event plays at the right pitch");

        // 5) Raw-MIDI pitch bend: full bend (0xE0 7F 7F) = +2 semitones.
        events.midi(0xE0, 0x7F, 0x7F, 0);
        processOnce();
        tone.clear();
        for (int block = 0; block < 8; ++block)
        {
            processOnce();
            for (int i = 0; i < 512; ++i)
                tone.push_back(outL[static_cast<size_t>(i)]);
        }
        const double bent = estimateFrequency(tone.data(),
                                              static_cast<int>(tone.size()), 48000.0);
        std::printf("      bent pitch: %.1f Hz (expected ~493.9)\n", bent);
        expect(std::fabs(bent - 493.9) < 25.0, "raw MIDI pitch bend reaches the plugin");
        events.midi(0xE0, 0x00, 0x40, 0);   // re-center
        events.noteOff(69, 0);
        processOnce();

        // 6) Latency change: flipping the lookahead toggle must request a
        //    main-thread callback and notify clap_host_latency.
        expect(latency->get(plugin) == 0, "initial latency is 0");
        smokeHost.latencyChangedSeen = false;
        events.paramValue(lookaheadId, 1.0, 0);
        processOnce();
        expect(smokeHost.callbackRequested, "latency change requests a callback");
        expect(smokeHost.latencyChangedSeen,
               "latency change notifies clap_host_latency");
        expect(latency->get(plugin) == 64, "new latency is reported");
        events.paramValue(lookaheadId, 0.0, 0);
        processOnce();

        // 7) Mono ports configuration: select, re-activate, process 1 channel.
        const auto* portsConfig = static_cast<const clap_plugin_audio_ports_config_t*>(
            plugin->get_extension(plugin, CLAP_EXT_AUDIO_PORTS_CONFIG));
        expect(portsConfig != nullptr, "audio-ports-config extension present");
        if (portsConfig != nullptr)
        {
            expect(portsConfig->count(plugin) == 2, "two ports configurations");
            clap_audio_ports_config_t cfg {};
            expect(portsConfig->get(plugin, 0, &cfg)
                       && cfg.main_output_channel_count == 1,
                   "first configuration is mono");
            plugin->stop_processing(plugin);
            plugin->deactivate(plugin);
            expect(portsConfig->select(plugin, cfg.id), "mono configuration selected");
            clap_audio_port_info_t monoInfo {};
            expect(ports->get(plugin, 0, true, &monoInfo)
                       && monoInfo.channel_count == 1,
                   "ports report mono after select");
            expect(plugin->activate(plugin, 48000.0, 32, 512), "re-activate mono");
            plugin->start_processing(plugin);
            inBufs[0].channel_count = 1;
            outBuf.channel_count = 1;
            fillInput(0.5f);
            events.paramValue(gainId, 1.0, 0);
            processOnce();
            processOnce();
            expect(std::fabs(outMean() - 0.5) < 1e-3, "mono processing works");
            plugin->stop_processing(plugin);
            plugin->deactivate(plugin);
            portsConfig->select(plugin, 2);   // back to stereo
            inBufs[0].channel_count = 2;
            outBuf.channel_count = 2;
            plugin->activate(plugin, 48000.0, 32, 512);
            plugin->start_processing(plugin);
        }

        // 8) Factory presets: preset-load applies, preset-discovery lists.
        const auto* presetLoad = static_cast<const clap_plugin_preset_load_t*>(
            plugin->get_extension(plugin, CLAP_EXT_PRESET_LOAD));
        expect(presetLoad != nullptr, "preset-load extension present");
        if (presetLoad != nullptr)
        {
            smokeHost.presetLoadedSeen = false;
            expect(presetLoad->from_location(plugin,
                       CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN, nullptr, "1"),
                   "preset loads from the plugin location");
            double gainNow = 0.0;
            params->get_value(plugin, gainId, &gainNow);
            expect(std::fabs(gainNow - 0.5) < 1e-9,
                   "loading the preset applies its parameter values");
            expect(smokeHost.presetLoadedSeen, "preset load reports loaded()");
            presetLoad->from_location(plugin,
                CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN, nullptr, "0");
        }
        proc.transport = nullptr;
    }

    // --- state round-trip ---------------------------------------------------------
    EventList stateEvents;
    events.paramValue(p0.id, 0.5 * (p0.min_value + p0.max_value), 0);
    params->flush(plugin, &events.iface, &kOutEvents);
    events.clear();
    (void) stateEvents;
    double mid = 0.0;
    params->get_value(plugin, p0.id, &mid);

    ByteOStream saved;
    expect(state->save(plugin, &saved.iface) && !saved.bytes.empty(),
           "state save produces a blob");

    events.paramValue(p0.id, p0.min_value, 0);
    params->flush(plugin, &events.iface, &kOutEvents);
    events.clear();

    ByteIStream restore;
    restore.bytes = saved.bytes;
    expect(state->load(plugin, &restore.iface), "state load accepts the blob");
    double after = 0.0;
    params->get_value(plugin, p0.id, &after);
    expect(std::fabs(after - mid) < 1e-6, "state round-trip restores the parameter");

    // --- gui extension (WebView editor layer; optional) -------------------------------
    // Contract-only: create/inspect/destroy without set_parent, so no real
    // window or web engine is needed — this also runs on headless CI.
    if (const auto* gui = static_cast<const clap_plugin_gui_t*>(
            plugin->get_extension(plugin, CLAP_EXT_GUI)))
    {
#if defined(_WIN32)
        const char* api = CLAP_WINDOW_API_WIN32;
#elif defined(__APPLE__)
        const char* api = CLAP_WINDOW_API_COCOA;
#else
        const char* api = CLAP_WINDOW_API_X11;
#endif
        expect(gui->is_api_supported(plugin, api, false),
               "gui supports the native embedded api");

        const char* preferred = nullptr;
        bool floating = true;
        expect(gui->get_preferred_api(plugin, &preferred, &floating)
                   && preferred != nullptr && !floating,
               "gui prefers a non-floating native api");

        expect(gui->create(plugin, api, false), "gui create");
        uint32_t width = 0, height = 0;
        expect(gui->get_size(plugin, &width, &height) && width > 0 && height > 0,
               "gui reports a usable size");
        std::printf("      editor: %u x %u%s\n", width, height,
                    gui->can_resize(plugin) ? " (resizable)" : "");

        uint32_t adjW = width, adjH = height;
        expect(gui->adjust_size(plugin, &adjW, &adjH) && adjW == width && adjH == height,
               "adjust_size accepts the reported size");
        gui->destroy(plugin);
    }
    else
        std::printf("      editor: none (host shows its generic UI)\n");

    plugin->stop_processing(plugin);
    plugin->deactivate(plugin);
    plugin->destroy(plugin);

    // --- preset discovery factory (declared presets are indexable) --------------------
    if (probeMode)
    {
        const auto* discovery = static_cast<const clap_preset_discovery_factory_t*>(
            entry->get_factory(CLAP_PRESET_DISCOVERY_FACTORY_ID));
        expect(discovery != nullptr, "preset-discovery factory present");
        if (discovery != nullptr)
        {
            expect(discovery->count(discovery) == 1, "one preset provider");
            const auto* provDesc = discovery->get_descriptor(discovery, 0);
            expect(provDesc != nullptr && provDesc->id != nullptr,
                   "provider descriptor");
            struct Indexer
            {
                clap_preset_discovery_indexer_t iface {};
                bool locationDeclared = false;
                int presetsSeen = 0;
            } indexer;
            indexer.iface.clap_version = CLAP_VERSION_INIT;
            indexer.iface.name = "DSPark smoke host";
            indexer.iface.indexer_data = &indexer;
            indexer.iface.declare_filetype =
                [](const clap_preset_discovery_indexer_t*,
                   const clap_preset_discovery_filetype_t*) { return true; };
            indexer.iface.declare_location =
                [](const clap_preset_discovery_indexer_t* idx,
                   const clap_preset_discovery_location_t* loc) {
                    auto* self = static_cast<Indexer*>(idx->indexer_data);
                    self->locationDeclared =
                        loc->kind == CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN;
                    return true;
                };
            indexer.iface.declare_soundpack =
                [](const clap_preset_discovery_indexer_t*,
                   const clap_preset_discovery_soundpack_t*) { return true; };
            indexer.iface.get_extension =
                [](const clap_preset_discovery_indexer_t*,
                   const char*) -> const void* { return nullptr; };

            const auto* provider = discovery->create(discovery, &indexer.iface,
                                                     provDesc->id);
            expect(provider != nullptr, "provider creates");
            if (provider != nullptr)
            {
                expect(provider->init(provider), "provider init");
                expect(indexer.locationDeclared,
                       "provider declares the plugin location");
                struct Receiver
                {
                    clap_preset_discovery_metadata_receiver_t iface {};
                    Indexer* indexer = nullptr;
                } receiver;
                receiver.indexer = &indexer;
                receiver.iface.receiver_data = &receiver;
                receiver.iface.on_error =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       int32_t, const char*) {};
                receiver.iface.begin_preset =
                    [](const clap_preset_discovery_metadata_receiver_t* r,
                       const char*, const char*) {
                        ++static_cast<Receiver*>(r->receiver_data)
                              ->indexer->presetsSeen;
                        return true;
                    };
                receiver.iface.add_plugin_id =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       const clap_universal_plugin_id_t*) {};
                receiver.iface.set_soundpack_id =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       const char*) {};
                receiver.iface.set_flags =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       uint32_t) {};
                receiver.iface.add_creator =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       const char*) {};
                receiver.iface.set_description =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       const char*) {};
                receiver.iface.set_timestamps =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       clap_timestamp, clap_timestamp) {};
                receiver.iface.add_feature =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       const char*) {};
                receiver.iface.add_extra_info =
                    [](const clap_preset_discovery_metadata_receiver_t*,
                       const char*, const char*) {};
                expect(provider->get_metadata(provider,
                           CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN, nullptr,
                           &receiver.iface),
                       "provider reports metadata");
                expect(indexer.presetsSeen == 2, "provider lists both presets");
                provider->destroy(provider);
            }
        }
    }

    entry->deinit();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "SMOKE OK" : "SMOKE FAILED",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
