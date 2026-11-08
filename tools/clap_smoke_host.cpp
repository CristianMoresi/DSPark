// DSPark - CLAP smoke host
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
//                        the right pitch, decay, and stop at once on a choke
//   --probe              full host-contract battery against the DSPark probe
//                        plugin: transport DC, the render extension's offline
//                        sign flip, sample-accurate automation, note events,
//                        raw-MIDI pitch bend and a CC at its sample offset,
//                        the latency-changed
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
    // clap_host_latency.changed is [main-thread & being-activated]: count
    // calls made outside activate() (a conforming host ignores them).
    bool activating = false;
    int latencyChangedOutsideActivate = 0;
    bool restartRequested = false;

    static SmokeHost* self(const clap_host_t* h)
    {
        return static_cast<SmokeHost*>(h->host_data);
    }

    static void sLatencyChanged(const clap_host_t* h)
    {
        self(h)->latencyChangedSeen = true;
        if (!self(h)->activating) ++self(h)->latencyChangedOutsideActivate;
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
    // Timer support: the Linux editor needs a host run loop to create at all
    // (the gui contract below runs without a window, so no tick is needed).
    static bool sRegisterTimer(const clap_host_t*, uint32_t, clap_id* id)
    {
        if (id != nullptr) *id = 1;
        return true;
    }
    static bool sUnregisterTimer(const clap_host_t*, clap_id) { return true; }
    inline static const clap_host_timer_support_t kTimer = {
        &sRegisterTimer, &sUnregisterTimer
    };

    // The gui host extension: resize requests are recorded and accepted.
    uint32_t resizeRequests = 0, requestedW = 0, requestedH = 0;
    static void sResizeHintsChanged(const clap_host_t*) {}
    static bool sRequestResize(const clap_host_t* h, uint32_t width, uint32_t height)
    {
        auto* s = self(h);
        ++s->resizeRequests;
        s->requestedW = width;
        s->requestedH = height;
        return true;
    }
    static bool sRequestShow(const clap_host_t*) { return true; }
    static bool sRequestHide(const clap_host_t*) { return true; }
    static void sClosed(const clap_host_t*, bool) {}
    inline static const clap_host_gui_t kGui = {
        &sResizeHintsChanged, &sRequestResize, &sRequestShow, &sRequestHide, &sClosed
    };

    static const void* sGetExtension(const clap_host_t*, const char* id)
    {
        if (std::strcmp(id, CLAP_EXT_GUI) == 0) return &kGui;
        if (std::strcmp(id, CLAP_EXT_TIMER_SUPPORT) == 0) return &kTimer;
        if (std::strcmp(id, CLAP_EXT_LATENCY) == 0) return &kLatency;
        if (std::strcmp(id, CLAP_EXT_PARAMS) == 0) return &kParams;
        if (std::strcmp(id, CLAP_EXT_PRESET_LOAD) == 0
            || std::strcmp(id, CLAP_EXT_PRESET_LOAD_COMPAT) == 0)
            return &kPresetLoad;
        return nullptr;
    }
    static void sRequestRestart(const clap_host_t* h) { self(h)->restartRequested = true; }
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
    /// A note event with an explicit (channel, key) address, -1 = wildcard.
    void note(uint16_t type, int16_t channel, int16_t key, uint32_t time)
    {
        clap_event_note_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = type;
        ev.note_id = -1;
        ev.port_index = -1;
        ev.channel = channel;
        ev.key = key;
        ev.velocity = 0.0;
        push(ev);
    }
    void noteOnWithId(int16_t key, double velocity, uint32_t time, int32_t noteId)
    {
        clap_event_note_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_NOTE_ON;
        ev.note_id = noteId;
        ev.port_index = 0;
        ev.channel = 0;
        ev.key = key;
        ev.velocity = velocity;
        push(ev);
    }
    /// A modulation amount for one voice, addressed by its note id.
    void voiceMod(uint32_t paramId, double amount, uint32_t time, int32_t noteId)
    {
        clap_event_param_mod_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_PARAM_MOD;
        ev.param_id = paramId;
        ev.note_id = noteId;
        ev.port_index = -1;
        ev.channel = -1;
        ev.key = -1;
        ev.amount = amount;
        push(ev);
    }
    /// A per-note value for one voice, addressed by its note id.
    void voiceValue(uint32_t paramId, double value, uint32_t time, int32_t noteId)
    {
        clap_event_param_value_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_PARAM_VALUE;
        ev.param_id = paramId;
        ev.note_id = noteId;
        ev.port_index = -1;
        ev.channel = -1;
        ev.key = -1;
        ev.value = value;
        push(ev);
    }
    /// A note expression addressed by (channel, key).
    void expression(clap_note_expression which, double value, uint32_t time,
                    int16_t channel, int16_t key)
    {
        clap_event_note_expression_t ev {};
        ev.header.size = sizeof(ev);
        ev.header.time = time;
        ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
        ev.header.type = CLAP_EVENT_NOTE_EXPRESSION;
        ev.expression_id = which;
        ev.note_id = -1;
        ev.port_index = -1;
        ev.channel = channel;
        ev.key = key;
        ev.value = value;
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

// The plugin's output events: NOTE_END ones are kept (voice ends).
std::vector<clap_event_note_t> g_noteEnds;
bool outTryPush(const clap_output_events_t*, const clap_event_header_t* ev)
{
    if (ev != nullptr && ev->space_id == CLAP_CORE_EVENT_SPACE_ID
        && ev->type == CLAP_EVENT_NOTE_END)
        g_noteEnds.push_back(*reinterpret_cast<const clap_event_note_t*>(ev));
    return true;
}
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
        // Every position of a discrete parameter must survive the host's
        // value -> text -> value round-trip (labels, On/Off, whole numbers).
        if (pi.flags & CLAP_PARAM_IS_STEPPED)
        {
            const int positions = static_cast<int>(pi.max_value - pi.min_value + 0.5) + 1;
            for (int k = 0; k < positions && k < 64; ++k)
            {
                const double v = pi.min_value
                    + (pi.max_value - pi.min_value) * k / (positions > 1 ? positions - 1 : 1);
                char text[128] {};
                double back = -1.0;
                if (!params->value_to_text(plugin, pi.id, v, text, sizeof(text))
                    || !params->text_to_value(plugin, pi.id, text, &back)
                    || std::fabs(back - v) > 1e-9)
                {
                    std::printf("      %s: %.3f -> \"%s\" -> %.3f\n", pi.name, v, text, back);
                    expect(false, "stepped parameter text round-trips");
                    break;
                }
            }
        }
    }
    {
        double junk = 0.0;
        expect(!params->text_to_value(plugin, p0.id, "not a number", &junk),
               "unparsable parameter text is refused");
    }
    expect(sawBypass, "bypass parameter present");

    const auto* state = static_cast<const clap_plugin_state_t*>(
        plugin->get_extension(plugin, CLAP_EXT_STATE));
    expect(state != nullptr, "state extension");
    const auto* latency = static_cast<const clap_plugin_latency_t*>(
        plugin->get_extension(plugin, CLAP_EXT_LATENCY));
    expect(latency != nullptr, "latency extension");

    auto activatePlugin = [&]() {
        smokeHost.activating = true;
        const bool ok = plugin->activate(plugin, 48000.0, 32, 512);
        smokeHost.activating = false;
        return ok;
    };
    expect(activatePlugin(), "activate");
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

        // A NOTE_CHOKE (a host choke group) must end the voice at once,
        // without its release phase: the plugin gets a NoteOff with
        // MidiEvent::choke set. Delivered as a plain NoteOff (the old
        // behaviour), the voice rang on through its whole release.
        events.noteOn(69, 1.0, 0);
        for (int block = 0; block < 4; ++block) processOnce();
        double held = 0.0;
        for (int i = 0; i < 512; ++i)
            held += static_cast<double>(outL[static_cast<size_t>(i)])
                  * outL[static_cast<size_t>(i)];
        events.note(CLAP_EVENT_NOTE_CHOKE, 0, 69, 0);
        processOnce();
        double choked = 0.0;   // after the first 128 frames of the choke block
        for (int i = 128; i < 512; ++i)
            choked += static_cast<double>(outL[static_cast<size_t>(i)])
                    * outL[static_cast<size_t>(i)];
        std::printf("      choke: held block energy %.3g -> %.3g after 128 frames\n",
                    held, choked);
        expect(held > 1e-3 && choked < held * 1e-6,
               "NOTE_CHOKE ends the voice without its release");
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

        // 6) Latency change while active: the CLAP latency extension allows
        //    clap_host_latency.changed only inside activate(), so the plugin
        //    must request a restart; the host's deactivate/activate cycle
        //    then reports the new value. The old wrapper called changed()
        //    from on_main_thread while active, which conforming hosts ignore.
        expect(latency->get(plugin) == 0, "initial latency is 0");
        smokeHost.latencyChangedSeen = false;
        smokeHost.latencyChangedOutsideActivate = 0;
        smokeHost.restartRequested = false;
        events.paramValue(lookaheadId, 1.0, 0);
        processOnce();
        expect(smokeHost.callbackRequested, "latency change requests a callback");
        expect(smokeHost.restartRequested && smokeHost.latencyChangedOutsideActivate == 0,
               "latency change while active requests a restart, not latency.changed");
        plugin->stop_processing(plugin);
        plugin->deactivate(plugin);
        expect(activatePlugin(), "re-activate after the restart request");
        plugin->start_processing(plugin);
        expect(smokeHost.latencyChangedSeen && smokeHost.latencyChangedOutsideActivate == 0,
               "the new latency is reported from activate()");
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
            expect(activatePlugin(), "re-activate mono");
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
            activatePlugin();
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

        // 9) Bypass under latency: with the lookahead engaged the probe
        //    delays its input by 64 samples. Hosts keep compensating that
        //    latency while the plugin is bypassed, so the bypassed output -
        //    and every sample of the crossfade - must stay exactly as late.
        constexpr uint32_t bypassId = 0x42595053u;   // 'BYPS'
        events.paramValue(gainId, 1.0, 0);
        events.paramValue(lookaheadId, 1.0, 0);
        std::vector<float> fed, got;
        int k = 0;
        for (int block = 0; block < 7; ++block)
        {
            for (int i = 0; i < 512; ++i, ++k)
            {
                const float v = 0.5f * std::sin(0.0113f * static_cast<float>(k))
                              + 0.2f * std::sin(0.171f * static_cast<float>(k));
                inL[static_cast<size_t>(i)] = v;
                inR[static_cast<size_t>(i)] = v;
                fed.push_back(v);
            }
            if (block == 2) events.paramValue(bypassId, 1.0, 100);
            if (block == 4) events.paramValue(bypassId, 0.0, 300);
            processOnce();
            got.insert(got.end(), outR.begin(), outR.end());
        }
        double worstLag = 0.0;
        for (size_t i = 512; i < got.size(); ++i)   // block 0 re-reports the latency
            worstLag = std::fmax(worstLag, std::fabs(got[i] - fed[i - 64]));
        std::printf("      bypass under latency: worst misalignment %.2g\n", worstLag);
        expect(worstLag < 1e-6, "bypassed output stays latency-aligned");
        events.paramValue(lookaheadId, 0.0, 0);
        processOnce();

        // 10) Oversize block: more frames than activate()'s maximum must
        //     still be processed whole, automation included (the step lands
        //     in the third chunk). Out of contract, but a wrapper must not
        //     leave host garbage in the output.
        {
            constexpr int kBig = 1200;
            std::vector<float> bigInL(kBig, 1.0f), bigInR(kBig, 1.0f),
                               bigOutL(kBig, -7.0f), bigOutR(kBig, -7.0f);
            inCh[0] = bigInL.data(); inCh[1] = bigInR.data();
            outCh[0] = bigOutL.data(); outCh[1] = bigOutR.data();
            proc.frames_count = kBig;
            events.paramValue(gainId, 0.0, 0);
            events.paramValue(gainId, 1.0, 1024);
            expect(processOnce(), "oversize block processes");
            bool ok = true;
            for (int i = 0; i < kBig; ++i)
            {
                const float want = i < 1024 ? 0.0f : 1.0f;
                ok = ok && std::fabs(bigOutL[static_cast<size_t>(i)] - want) < 1e-6f
                        && std::fabs(bigOutR[static_cast<size_t>(i)] - want) < 1e-6f;
            }
            expect(ok, "oversize block is processed whole, automation at its sample");
            inCh[0] = inL.data(); inCh[1] = inR.data();
            outCh[0] = outL.data(); outCh[1] = outR.data();
            proc.frames_count = 512;
            events.paramValue(gainId, 1.0, 0);
            processOnce();
        }

        // 11) Narrower input port than output: the wrapper must not read
        //     past the input's channel array, and the missing channel
        //     repeats the one given.
        {
            float* monoIn[1] = { inL.data() };
            inBufs[0].data32 = monoIn;
            inBufs[0].channel_count = 1;
            fillInput(0.5f);
            processOnce();
            processOnce();
            double meanR = 0.0;
            for (int i = 0; i < 512; ++i) meanR += outR[static_cast<size_t>(i)];
            expect(std::fabs(outMean() - 0.5) < 1e-3 && std::fabs(meanR / 512.0 - 0.5) < 1e-3,
                   "mono input port feeds both output channels");
            inBufs[0].data32 = inCh;
            inBufs[0].channel_count = 2;
        }
        // 12) More than 256 events in one block: 300 gain points, a note-off
        //     and a later gain point. Every note event and every parameter's
        //     final value must survive. The old list overwrote its last slot
        //     on overflow, so the note-off was replaced by the gain point
        //     after it and the voice rang on.
        {
            fillInput(1.0f);
            events.paramValue(gainId, 1.0, 0);
            events.noteOn(60, 1.0, 0);
            processOnce();
            for (uint32_t t = 0; t < 300; ++t) events.paramValue(gainId, 0.25, t);
            events.noteOff(60, 400);
            events.paramValue(gainId, 0.75, 450);
            processOnce();
            processOnce();
            double worst = 0.0;
            for (int i = 0; i < 512; ++i)
                worst = std::fmax(worst, std::fabs(outL[static_cast<size_t>(i)] - 0.75));
            std::printf("      overflow block: mean %.4f (expected 0.75), worst deviation %.3g\n",
                        outMean(), worst);
            expect(std::fabs(outMean() - 0.75) < 1e-3,
                   "event overflow keeps every parameter's final value");
            expect(worst < 1e-3, "event overflow keeps the note-off (no stuck voice)");
        }

        // 13) Wildcard NOTE_OFF (channel -1, key -1) releases every sounding
        //     note; NOTE_CHOKE stops its note. The old wrapper read -1 as
        //     channel 0 / key 0 and ignored chokes, so both voices rang on.
        {
            fillInput(0.0f);
            auto voiceEnergy = [&]() {
                double e = 0.0;
                for (int i = 0; i < 512; ++i)
                    e += static_cast<double>(outL[static_cast<size_t>(i)]) * outL[static_cast<size_t>(i)];
                return e;
            };
            events.noteOn(64, 1.0, 0);
            processOnce();
            const double sounding = voiceEnergy();
            events.note(CLAP_EVENT_NOTE_OFF, -1, -1, 0);
            processOnce();
            processOnce();
            std::printf("      wildcard note-off: energy %.3g -> %.3g\n", sounding, voiceEnergy());
            expect(sounding > 1.0 && voiceEnergy() < 1e-9,
                   "wildcard NOTE_OFF releases the sounding note");
            events.noteOn(67, 1.0, 0);
            processOnce();
            events.note(CLAP_EVENT_NOTE_CHOKE, 0, 67, 0);
            processOnce();
            processOnce();
            expect(voiceEnergy() < 1e-9, "NOTE_CHOKE stops the note");
        }

        // 14) An effect given no input port processes silence (the old code
        //     processed the host's stale output buffer as input).
        {
            proc.audio_inputs = nullptr;
            proc.audio_inputs_count = 0;
            for (int i = 0; i < 512; ++i)
            {
                outL[static_cast<size_t>(i)] = 0.7f;
                outR[static_cast<size_t>(i)] = 0.7f;
            }
            events.paramValue(gainId, 1.0, 0);
            processOnce();
            std::printf("      no input port: output mean %.4f (expected 0)\n", outMean());
            expect(std::fabs(outMean()) < 1e-6, "missing input port processes silence");
            proc.audio_inputs = inBufs;
            proc.audio_inputs_count = expectedInPorts;
        }

        // 15) Sample-accurate bypass: with the wet path silent (gain 0) a
        //     bypass event at time 384 starts the crossfade there (the old
        //     once-per-block blend began it at sample 0).
        {
            fillInput(1.0f);
            events.paramValue(gainId, 0.0, 0);
            processOnce();
            events.paramValue(bypassId, 1.0, 384);
            processOnce();
            int first = -1;
            for (int i = 0; i < 512; ++i)
                if (outL[static_cast<size_t>(i)] != 0.0f)
                {
                    first = i;
                    break;
                }
            std::printf("      bypass at 384: crossfade starts at sample %d\n", first);
            expect(first >= 352 && outL[511] > 0.0f, "bypass crossfade starts at its event");
            events.paramValue(bypassId, 0.0, 0);
            events.paramValue(gainId, 1.0, 0);
            processOnce();
            processOnce();
        }

        // 16) A state without some parameter (saved before it existed)
        //     restores that parameter's DEFAULT; the old code kept the value
        //     the instance held (0.3 here).
        {
            events.paramValue(gainId, 0.3, 0);
            processOnce();
            ByteIStream older;
            auto put32 = [&](uint32_t v) {
                for (int b = 0; b < 4; ++b)
                    older.bytes.push_back(static_cast<uint8_t>(v >> (8 * b)));
            };
            put32(0x4453504Bu);   // "DSPK"
            put32(1u);            // version
            put32(1u);            // one entry: the lookahead only
            put32(lookaheadId);
            put32(0u);            // 0.0 as a double
            put32(0u);
            put32(0u);            // no user section
            expect(state->load(plugin, &older.iface), "state load accepts an older state");
            double gainNow = 0.0;
            params->get_value(plugin, gainId, &gainNow);
            expect(std::fabs(gainNow - 1.0) < 1e-9,
                   "a parameter missing from the state takes its default");
            processOnce();
        }

        // 17) A raw MIDI controller lands at its sample offset: CC74 on
        //     channel 4 at frame 300 (off the 32-frame grid, 12 frames into
        //     its sub-block) raises the probe's DC by 0.2 from that frame.
        {
            fillInput(0.0f);
            processOnce();
            events.midi(0xB3, 74, 127, 300);
            processOnce();
            int ccAt = -1;
            for (int i = 0; i < 512; ++i)
                if (outL[static_cast<size_t>(i)] > 0.1f)
                {
                    ccAt = i;
                    break;
                }
            std::printf("      CC74 at frame 300 lands at sample %d\n", ccAt);
            expect(ccAt == 300 && std::fabs(outL[511] - 0.2f) < 1e-6f,
                   "a MIDI controller lands at its sample offset");
            events.midi(0xB3, 74, 0, 0);
            processOnce();
        }
        proc.transport = nullptr;

        // 18) Per-voice signals: "gain" declares per-voice modulation; a
        //     modulation addressed to the voice's note id and a Volume note
        //     expression each scale the voice, and its end reaches the host
        //     as NOTE_END with the note id.
        {
            uint32_t perVoiceFlags = 0;
            for (uint32_t i = 0; i < numParams; ++i)
            {
                clap_param_info_t pi {};
                if (params->get_info(plugin, i, &pi) && pi.id == gainId)
                    perVoiceFlags = pi.flags & (CLAP_PARAM_IS_MODULATABLE_PER_NOTE_ID
                                                | CLAP_PARAM_IS_MODULATABLE_PER_KEY
                                                | CLAP_PARAM_IS_MODULATABLE_PER_CHANNEL
                                                | CLAP_PARAM_IS_MODULATABLE_PER_PORT);
            }
            expect(perVoiceFlags == (CLAP_PARAM_IS_MODULATABLE_PER_NOTE_ID
                                     | CLAP_PARAM_IS_MODULATABLE_PER_KEY
                                     | CLAP_PARAM_IS_MODULATABLE_PER_CHANNEL
                                     | CLAP_PARAM_IS_MODULATABLE_PER_PORT),
                   "a perVoice parameter declares every per-voice modulation form");
            auto voicePeak = [&]() {
                processOnce();   // let the change settle a block
                processOnce();
                float peak = 0.0f;
                for (int i = 0; i < 512; ++i)
                    peak = std::fmax(peak, std::fabs(outL[static_cast<size_t>(i)]));
                return peak;
            };
            fillInput(0.0f);
            events.paramValue(gainId, 1.0, 0);
            events.noteOnWithId(69, 0.5, 0, 7);
            const float plainVoice = voicePeak();
            events.voiceMod(gainId, -0.5, 0, 7);
            const float modulated = voicePeak();
            events.expression(CLAP_NOTE_EXPRESSION_VOLUME, 0.5, 0, 0, 69);
            const float expressed = voicePeak();
            // Per-note automation: the voice's own value (0.75) plus its own
            // modulation (-0.5); the parameter itself stays at 1.
            uint32_t perNoteFlags = 0;
            for (uint32_t i = 0; i < numParams; ++i)
            {
                clap_param_info_t pi {};
                if (params->get_info(plugin, i, &pi) && pi.id == gainId)
                    perNoteFlags = pi.flags & (CLAP_PARAM_IS_AUTOMATABLE_PER_NOTE_ID
                                               | CLAP_PARAM_IS_AUTOMATABLE_PER_KEY
                                               | CLAP_PARAM_IS_AUTOMATABLE_PER_CHANNEL);
            }
            events.voiceValue(gainId, 0.75, 0, 7);
            const float automated = voicePeak();
            double gainAfter = 0.0;
            params->get_value(plugin, gainId, &gainAfter);
            g_noteEnds.clear();
            events.noteOff(69, 0);
            processOnce();
            std::printf("      voice peak %.3f -> per-voice gain %.3f -> volume expression %.3f"
                        " -> per-note value %.3f (parameter %.2f)\n",
                        plainVoice, modulated, expressed, automated, gainAfter);
            expect(std::fabs(plainVoice - 0.5f) < 0.01f && std::fabs(modulated - 0.25f) < 0.01f
                       && std::fabs(expressed - 0.125f) < 0.01f,
                   "per-voice modulation and a note expression reach the voice");
            expect(perNoteFlags == (CLAP_PARAM_IS_AUTOMATABLE_PER_NOTE_ID
                                    | CLAP_PARAM_IS_AUTOMATABLE_PER_KEY
                                    | CLAP_PARAM_IS_AUTOMATABLE_PER_CHANNEL)
                       && std::fabs(automated - 0.0625f) < 0.01f
                       && std::fabs(gainAfter - 1.0) < 1e-9,
                   "per-note automation sets the voice, not the parameter");
            expect(g_noteEnds.size() == 1 && g_noteEnds[0].note_id == 7
                       && g_noteEnds[0].key == 69,
                   "the voice's end reaches the host (NOTE_END with its note id)");
        }
    }

    // --- state round-trip ---------------------------------------------------------
    constexpr uint32_t kBypassId = 0x42595053u;   // 'BYPS', the wrapper's bypass
    events.paramValue(p0.id, 0.5 * (p0.min_value + p0.max_value), 0);
    events.paramValue(kBypassId, 1.0, 0);   // bypass is state like any parameter
    params->flush(plugin, &events.iface, &kOutEvents);
    events.clear();
    double mid = 0.0;
    params->get_value(plugin, p0.id, &mid);

    ByteOStream saved;
    expect(state->save(plugin, &saved.iface) && !saved.bytes.empty(),
           "state save produces a blob");

    events.paramValue(p0.id, p0.min_value, 0);
    events.paramValue(kBypassId, 0.0, 0);
    params->flush(plugin, &events.iface, &kOutEvents);
    events.clear();

    ByteIStream restore;
    restore.bytes = saved.bytes;
    expect(state->load(plugin, &restore.iface), "state load accepts the blob");
    double after = 0.0;
    params->get_value(plugin, p0.id, &after);
    expect(std::fabs(after - mid) < 1e-6, "state round-trip restores the parameter");
    double bypassAfter = 0.0;
    expect(params->get_value(plugin, kBypassId, &bypassAfter)
               && bypassAfter >= 0.5,
           "state round-trip restores the bypass (clap-validator demands it)");
    events.paramValue(kBypassId, 0.0, 0);   // leave the plugin un-bypassed
    params->flush(plugin, &events.iface, &kOutEvents);
    events.clear();

    // --- gui extension (WebView editor layer; optional) -------------------------------
    // Contract-only: create/inspect/destroy without set_parent, so no real
    // window or web engine is needed - this also runs on headless CI.
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

        // A resizable editor reopens at the size the user left, and a new
        // instance loading the saved state (the next session) opens there.
        if (gui->can_resize(plugin))
        {
            uint32_t bigW = width + width / 2, bigH = height + height / 2;
            gui->adjust_size(plugin, &bigW, &bigH);
            expect(gui->set_size(plugin, bigW, bigH), "gui set_size to a larger size");
            gui->destroy(plugin);
            expect(gui->create(plugin, api, false), "gui create again");
            uint32_t againW = 0, againH = 0;
            gui->get_size(plugin, &againW, &againH);
            expect(againW == bigW && againH == bigH,
                   "the editor reopens at the size the user left");
            ByteOStream session;
            expect(state->save(plugin, &session.iface), "state save with the editor size");
            // A state loaded while the editor exists asks the host to resize
            // it to the state's size (the old wrapper only remembered it).
            expect(gui->set_size(plugin, width, height), "gui set_size back");
            smokeHost.resizeRequests = 0;
            ByteIStream reload;
            reload.bytes = session.bytes;
            expect(state->load(plugin, &reload.iface), "state load with the editor open");
            uint32_t liveW = 0, liveH = 0;
            gui->get_size(plugin, &liveW, &liveH);
            std::printf("      state load with the editor open: %u resize request(s) to %u x %u,"
                        " editor %u x %u\n", smokeHost.resizeRequests, smokeHost.requestedW,
                        smokeHost.requestedH, liveW, liveH);
            expect(smokeHost.resizeRequests == 1 && smokeHost.requestedW == bigW
                       && smokeHost.requestedH == bigH && liveW == bigW && liveH == bigH,
                   "a state loaded with the editor open resizes it");
            const clap_plugin_t* next = factory->create_plugin(factory, &smokeHost.host,
                                                               desc->id);
            expect(next != nullptr && next->init(next), "a second instance");
            if (next != nullptr)
            {
                const auto* nextState = static_cast<const clap_plugin_state_t*>(
                    next->get_extension(next, CLAP_EXT_STATE));
                const auto* nextGui = static_cast<const clap_plugin_gui_t*>(
                    next->get_extension(next, CLAP_EXT_GUI));
                ByteIStream sessionIn;
                sessionIn.bytes = session.bytes;
                uint32_t nextW = 0, nextH = 0;
                if (nextState != nullptr && nextGui != nullptr
                    && nextState->load(next, &sessionIn.iface)
                    && nextGui->create(next, api, false))
                {
                    nextGui->get_size(next, &nextW, &nextH);
                    nextGui->destroy(next);
                }
                std::printf("      editor reopens at %u x %u in the next session\n",
                            nextW, nextH);
                expect(nextW == bigW && nextH == bigH,
                       "the next session opens the editor at the saved size");
                next->destroy(next);
            }
        }
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
