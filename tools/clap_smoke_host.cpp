// DSPark — CLAP smoke host
//
// Loads a .clap module through the public C ABI like a DAW would and drives
// the full lifecycle: entry -> factory -> descriptor -> instantiation ->
// extensions -> activate -> process (sine blocks + a live parameter event) ->
// state save/perturb/restore -> teardown. Exit code 0 means every step
// behaved. Companion to tools/vst3_smoke_host.cpp; clap-validator is the
// recommended final gate.
//
//   clap_smoke_host <path-to-module.clap>

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

// -- minimal host ---------------------------------------------------------------

const void* hostGetExtension(const clap_host_t*, const char*) { return nullptr; }
void hostNoop(const clap_host_t*) {}

const clap_host_t kHost = {
    CLAP_VERSION_INIT,
    nullptr,
    "DSPark smoke host", "DSPark", "https://github.com/CristianMoresi/DSPark", "1.0",
    &hostGetExtension, &hostNoop, &hostNoop, &hostNoop
};

// -- event list with zero or one param event --------------------------------------

struct EventList
{
    clap_input_events_t iface;
    clap_event_param_value_t event {};
    uint32_t count = 0;

    static uint32_t sSize(const clap_input_events_t* l)
    {
        return static_cast<const EventList*>(l->ctx)->count;
    }
    static const clap_event_header_t* sGet(const clap_input_events_t* l, uint32_t i)
    {
        const auto* self = static_cast<const EventList*>(l->ctx);
        return (i < self->count) ? &self->event.header : nullptr;
    }
    EventList()
    {
        iface.ctx = this;
        iface.size = &sSize;
        iface.get = &sGet;
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

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: clap_smoke_host <module.clap>\n");
        return 2;
    }

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

    const clap_plugin_t* plugin = factory->create_plugin(factory, &kHost, desc->id);
    expect(plugin != nullptr, "create_plugin");
    if (!plugin) return 1;
    expect(plugin->init(plugin), "plugin init");

    const auto* ports = static_cast<const clap_plugin_audio_ports_t*>(
        plugin->get_extension(plugin, CLAP_EXT_AUDIO_PORTS));
    expect(ports != nullptr, "audio-ports extension");
    if (ports)
    {
        clap_audio_port_info_t info {};
        expect(ports->count(plugin, true) == 1 && ports->get(plugin, 0, true, &info)
                   && info.channel_count == 2,
               "stereo input port");
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
    float* inCh[2]  = { inL.data(), inR.data() };
    float* outCh[2] = { outL.data(), outR.data() };

    clap_audio_buffer_t inBuf {};
    inBuf.data32 = inCh;
    inBuf.channel_count = 2;
    clap_audio_buffer_t outBuf {};
    outBuf.data32 = outCh;
    outBuf.channel_count = 2;

    EventList events;
    clap_process_t proc {};
    proc.steady_time = -1;
    proc.frames_count = 512;
    proc.audio_inputs = &inBuf;
    proc.audio_outputs = &outBuf;
    proc.audio_inputs_count = 1;
    proc.audio_outputs_count = 1;
    proc.in_events = &events.iface;
    proc.out_events = &kOutEvents;

    bool finite = true;
    double energy = 0.0;
    int n = 0;
    for (int block = 0; block < 40; ++block)
    {
        for (int i = 0; i < 512; ++i, ++n)
        {
            inL[static_cast<size_t>(i)] = 0.25f
                * std::sin(2.0f * 3.14159265f * 1000.0f * static_cast<float>(n) / 48000.0f);
            inR[static_cast<size_t>(i)] = inL[static_cast<size_t>(i)];
        }
        if (block == 20)   // live parameter event mid-run
        {
            events.event = {};
            events.event.header.size = sizeof(clap_event_param_value_t);
            events.event.header.time = 0;
            events.event.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
            events.event.header.type = CLAP_EVENT_PARAM_VALUE;
            events.event.param_id = p0.id;
            events.event.value = 0.5 * (p0.min_value + p0.max_value);
            events.count = 1;
        }
        else
            events.count = 0;

        if (plugin->process(plugin, &proc) != CLAP_PROCESS_CONTINUE)
        {
            expect(false, "process returns CONTINUE");
            break;
        }
        for (int i = 0; i < 512; ++i)
        {
            const float v = outL[static_cast<size_t>(i)];
            if (!std::isfinite(v)) finite = false;
            if (block >= 4) energy += static_cast<double>(v) * v;
        }
    }
    expect(finite, "output stays finite");
    expect(energy > 1e-6, "output carries signal");

    double mid = 0.0;
    expect(params->get_value(plugin, p0.id, &mid), "get_value");
    expect(std::fabs(mid - 0.5 * (p0.min_value + p0.max_value)) < 1e-6,
           "parameter event applied");

    // --- state round-trip ---------------------------------------------------------
    ByteOStream saved;
    expect(state->save(plugin, &saved.iface) && !saved.bytes.empty(),
           "state save produces a blob");

    EventList perturb;
    perturb.event = {};
    perturb.event.header.size = sizeof(clap_event_param_value_t);
    perturb.event.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
    perturb.event.header.type = CLAP_EVENT_PARAM_VALUE;
    perturb.event.param_id = p0.id;
    perturb.event.value = p0.min_value;
    perturb.count = 1;
    params->flush(plugin, &perturb.iface, &kOutEvents);

    ByteIStream restore;
    restore.bytes = saved.bytes;
    expect(state->load(plugin, &restore.iface), "state load accepts the blob");
    double after = 0.0;
    params->get_value(plugin, p0.id, &after);
    expect(std::fabs(after - mid) < 1e-6, "state round-trip restores the parameter");

    plugin->stop_processing(plugin);
    plugin->deactivate(plugin);
    plugin->destroy(plugin);
    entry->deinit();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "SMOKE OK" : "SMOKE FAILED",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
