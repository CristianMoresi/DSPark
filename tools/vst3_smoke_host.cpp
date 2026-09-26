// DSPark - VST3 smoke host
//
// Loads a .vst3 module through the public C ABI exactly like a DAW would and
// drives the full lifecycle: factory -> class info -> instantiation ->
// initialize -> bus/parameter introspection -> setupProcessing -> setActive ->
// process (sine blocks, parameter change mid-run) -> state round-trip ->
// teardown. Exit code 0 means every step behaved.
//
//   vst3_smoke_host <module> [--expect-sidechain | --expect-instrument | --probe]
//
//   --expect-sidechain   the plugin declares an aux input; also proves the
//                        key reaches the detector (hot key ducks the output)
//   --expect-instrument  no audio inputs, an event input bus; a note must
//                        produce sound at the right pitch and then decay
//   --probe              full host-contract battery against the DSPark probe
//                        plugin (tools/plugin_probe.cpp): transport DC,
//                        offline sign flip, sample-accurate automation step,
//                        note events + IMidiMapping pitch bend, the
//                        latency-changed notification, mono negotiation and
//                        the factory-preset program list
//
// This is the local first-line gate for plugins built with DSParkVst3.h;
// run Steinberg's validator and Tracktion's pluginval before shipping.

#include "../plugin/vst3/vst3_c_api.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#if defined(_WIN32)
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
    #include <windows.h>
    #if defined(_MSC_VER)
    #pragma comment(lib, "user32.lib")
    #endif
#elif defined(__APPLE__)
    #include <dlfcn.h>
#endif

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

// -- minimal host-side IBStream over a byte vector ------------------------------

struct MemoryStream
{
    const Steinberg_IBStreamVtbl* vtbl;
    std::vector<uint8_t> bytes;
    int64_t pos = 0;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { if (o) *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE read(void* self_, void* dst,
        Steinberg_int32 n, Steinberg_int32* got)
    {
        auto* s = static_cast<MemoryStream*>(self_);
        const int64_t left = static_cast<int64_t>(s->bytes.size()) - s->pos;
        const Steinberg_int32 take = static_cast<Steinberg_int32>(
            left < n ? (left > 0 ? left : 0) : n);
        if (take > 0)
        {
            std::memcpy(dst, s->bytes.data() + s->pos, static_cast<size_t>(take));
            s->pos += take;
        }
        if (got) *got = take;
        return take > 0 ? Steinberg_kResultOk : Steinberg_kResultFalse;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE write(void* self_, void* src,
        Steinberg_int32 n, Steinberg_int32* put)
    {
        auto* s = static_cast<MemoryStream*>(self_);
        const auto* p = static_cast<const uint8_t*>(src);
        s->bytes.insert(s->bytes.end(), p, p + n);
        s->pos += n;
        if (put) *put = n;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE seek(void* self_, Steinberg_int64 p,
        Steinberg_int32 mode, Steinberg_int64* out)
    {
        auto* s = static_cast<MemoryStream*>(self_);
        int64_t base = mode == 0 ? 0 : (mode == 1 ? s->pos
                                                  : static_cast<int64_t>(s->bytes.size()));
        s->pos = base + p;
        if (s->pos < 0) s->pos = 0;
        if (out) *out = s->pos;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE tell(void* self_, Steinberg_int64* out)
    {
        if (out) *out = static_cast<MemoryStream*>(self_)->pos;
        return Steinberg_kResultOk;
    }

    inline static const Steinberg_IBStreamVtbl kVtbl = { &q, &ar, &rel,
                                                         &read, &write, &seek, &tell };
    MemoryStream() : vtbl(&kVtbl) {}
};

// -- host-side IComponentHandler (captures restartComponent flags) ----------------

struct HostHandler
{
    const Steinberg_Vst_IComponentHandlerVtbl* vtbl;
    Steinberg_int32 restartFlags = 0;
    int restartCalls = 0;
    // VST3 allows restartComponent on the UI thread only: count violations.
    std::thread::id uiThread = std::this_thread::get_id();
    int offThreadRestarts = 0;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void* self_,
        const Steinberg_TUID iid, void** o)
    {
        if (o == nullptr) return Steinberg_kInvalidArgument;
        if (std::memcmp(iid, Steinberg_FUnknown_iid, 16) == 0
            || std::memcmp(iid, Steinberg_Vst_IComponentHandler_iid, 16) == 0)
        {
            *o = self_;
            return Steinberg_kResultOk;
        }
        *o = nullptr;
        return Steinberg_kNoInterface;
    }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE beginEdit(void*, Steinberg_Vst_ParamID)
    { return Steinberg_kResultOk; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE performEdit(void*,
        Steinberg_Vst_ParamID, Steinberg_Vst_ParamValue)
    { return Steinberg_kResultOk; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE endEdit(void*, Steinberg_Vst_ParamID)
    { return Steinberg_kResultOk; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE restartComponent(void* self_,
        Steinberg_int32 flags)
    {
        auto* h = static_cast<HostHandler*>(self_);
        h->restartFlags |= flags;
        ++h->restartCalls;
        if (std::this_thread::get_id() != h->uiThread) ++h->offThreadRestarts;
        return Steinberg_kResultOk;
    }

    inline static const Steinberg_Vst_IComponentHandlerVtbl kVtbl = {
        &q, &ar, &rel, &beginEdit, &performEdit, &endEdit, &restartComponent
    };
    HostHandler() : vtbl(&kVtbl) {}
};

// -- host application context with a Linux IRunLoop ----------------------------------
//
// Offered through initialize(): the UI-thread tick a plugin uses to hand the
// host restart requests raised on the audio thread. pumpUiThread() plays
// the host's event loop on every platform.

struct HostRunLoop
{
    const Steinberg_Linux_IRunLoopVtbl* vtbl;
    Steinberg_Linux_ITimerHandler* timer = nullptr;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void* self_, const Steinberg_TUID iid,
                                                      void** o)
    {
        if (o == nullptr) return Steinberg_kInvalidArgument;
        if (std::memcmp(iid, Steinberg_FUnknown_iid, 16) == 0
            || std::memcmp(iid, Steinberg_Linux_IRunLoop_iid, 16) == 0)
        {
            *o = self_;
            return Steinberg_kResultOk;
        }
        *o = nullptr;
        return Steinberg_kNoInterface;
    }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE regEvent(void*,
        Steinberg_Linux_IEventHandler*, Steinberg_Linux_FileDescriptor)
    { return Steinberg_kNotImplemented; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE unregEvent(void*,
        Steinberg_Linux_IEventHandler*)
    { return Steinberg_kNotImplemented; }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE regTimer(void* self_,
        Steinberg_Linux_ITimerHandler* handler, Steinberg_Linux_TimerInterval)
    {
        static_cast<HostRunLoop*>(self_)->timer = handler;
        return Steinberg_kResultOk;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE unregTimer(void* self_,
        Steinberg_Linux_ITimerHandler* handler)
    {
        auto* loop = static_cast<HostRunLoop*>(self_);
        if (loop->timer == handler) loop->timer = nullptr;
        return Steinberg_kResultOk;
    }
    inline static const Steinberg_Linux_IRunLoopVtbl kVtbl = {
        &q, &ar, &rel, &regEvent, &unregEvent, &regTimer, &unregTimer
    };
    HostRunLoop() : vtbl(&kVtbl) {}
};

struct HostContext
{
    const Steinberg_FUnknownVtbl* vtbl;
    HostRunLoop* runLoop = nullptr;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void* self_, const Steinberg_TUID iid,
                                                      void** o)
    {
        if (o == nullptr) return Steinberg_kInvalidArgument;
        auto* context = static_cast<HostContext*>(self_);
        if (std::memcmp(iid, Steinberg_Linux_IRunLoop_iid, 16) == 0)
        {
            *o = context->runLoop;
            return Steinberg_kResultOk;
        }
        if (std::memcmp(iid, Steinberg_FUnknown_iid, 16) == 0)
        {
            *o = self_;
            return Steinberg_kResultOk;
        }
        *o = nullptr;
        return Steinberg_kNoInterface;
    }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }
    inline static const Steinberg_FUnknownVtbl kVtbl = { &q, &ar, &rel };
    HostContext() : vtbl(&kVtbl) {}
};

/** One slice of the host's UI event loop: Windows messages (thread timers),
 *  the macOS main run loop (main-queue sources), the Linux IRunLoop timer. */
void pumpUiThread(HostRunLoop& runLoop)
{
#if defined(_WIN32)
    (void) runLoop;
    MSG message {};
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    Sleep(5);
#elif defined(__APPLE__)
    (void) runLoop;
    // CoreFoundation resolved at runtime: the host links no framework.
    using RunInMode = int (*)(const void*, double, unsigned char);
    static void* cf = dlopen(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation", RTLD_LAZY);
    static auto run = cf ? reinterpret_cast<RunInMode>(dlsym(cf, "CFRunLoopRunInMode"))
                         : nullptr;
    static auto mode = cf ? static_cast<const void* const*>(
                                dlsym(cf, "kCFRunLoopDefaultMode"))
                          : nullptr;
    if (run != nullptr && mode != nullptr) run(*mode, 0.005, 0);
    else std::this_thread::sleep_for(std::chrono::milliseconds(5));
#else
    if (runLoop.timer != nullptr) runLoop.timer->lpVtbl->onTimer(runLoop.timer);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
#endif
}

// -- host-side IParamValueQueue / IParameterChanges --------------------------------

struct ParamQueue
{
    const Steinberg_Vst_IParamValueQueueVtbl* vtbl;
    Steinberg_Vst_ParamID id = 0;
    struct Point { Steinberg_int32 offset; Steinberg_Vst_ParamValue value; };
    std::vector<Point> points;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { if (o) *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }
    static Steinberg_Vst_ParamID SMTG_STDMETHODCALLTYPE getParameterId(void* self_)
    { return static_cast<ParamQueue*>(self_)->id; }
    static Steinberg_int32 SMTG_STDMETHODCALLTYPE getPointCount(void* self_)
    { return static_cast<Steinberg_int32>(static_cast<ParamQueue*>(self_)->points.size()); }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE getPoint(void* self_,
        Steinberg_int32 index, Steinberg_int32* offset, Steinberg_Vst_ParamValue* value)
    {
        auto* queue = static_cast<ParamQueue*>(self_);
        if (index < 0 || index >= static_cast<Steinberg_int32>(queue->points.size()))
            return Steinberg_kInvalidArgument;
        if (offset) *offset = queue->points[static_cast<size_t>(index)].offset;
        if (value) *value = queue->points[static_cast<size_t>(index)].value;
        return Steinberg_kResultOk;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE addPoint(void*, Steinberg_int32,
        Steinberg_Vst_ParamValue, Steinberg_int32*)
    { return Steinberg_kResultFalse; }

    inline static const Steinberg_Vst_IParamValueQueueVtbl kVtbl = {
        &q, &ar, &rel, &getParameterId, &getPointCount, &getPoint, &addPoint
    };
    ParamQueue() : vtbl(&kVtbl) {}
};

struct ParamChanges
{
    const Steinberg_Vst_IParameterChangesVtbl* vtbl;
    std::vector<ParamQueue> queues;

    void clear() { queues.clear(); }
    ParamQueue& add(Steinberg_Vst_ParamID id)
    {
        queues.emplace_back();
        queues.back().id = id;
        return queues.back();
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { if (o) *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }
    static Steinberg_int32 SMTG_STDMETHODCALLTYPE getParameterCount(void* self_)
    { return static_cast<Steinberg_int32>(static_cast<ParamChanges*>(self_)->queues.size()); }
    static Steinberg_Vst_IParamValueQueue* SMTG_STDMETHODCALLTYPE getParameterData(
        void* self_, Steinberg_int32 index)
    {
        auto* c = static_cast<ParamChanges*>(self_);
        if (index < 0 || index >= static_cast<Steinberg_int32>(c->queues.size()))
            return nullptr;
        return reinterpret_cast<Steinberg_Vst_IParamValueQueue*>(
            &c->queues[static_cast<size_t>(index)]);
    }
    static Steinberg_Vst_IParamValueQueue* SMTG_STDMETHODCALLTYPE addParameterData(
        void*, const Steinberg_Vst_ParamID*, Steinberg_int32*)
    { return nullptr; }

    inline static const Steinberg_Vst_IParameterChangesVtbl kVtbl = {
        &q, &ar, &rel, &getParameterCount, &getParameterData, &addParameterData
    };
    ParamChanges() : vtbl(&kVtbl) {}
};

// -- host-side IEventList -----------------------------------------------------------

struct EventList
{
    const Steinberg_Vst_IEventListVtbl* vtbl;
    std::vector<Steinberg_Vst_Event> events;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void*, const Steinberg_TUID, void** o)
    { if (o) *o = nullptr; return Steinberg_kNoInterface; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*) { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }
    static Steinberg_int32 SMTG_STDMETHODCALLTYPE getEventCount(void* self_)
    { return static_cast<Steinberg_int32>(static_cast<EventList*>(self_)->events.size()); }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE getEvent(void* self_,
        Steinberg_int32 index, struct Steinberg_Vst_Event* e)
    {
        auto* list = static_cast<EventList*>(self_);
        if (e == nullptr || index < 0
            || index >= static_cast<Steinberg_int32>(list->events.size()))
            return Steinberg_kInvalidArgument;
        *e = list->events[static_cast<size_t>(index)];
        return Steinberg_kResultOk;
    }
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE addEvent(void*,
        struct Steinberg_Vst_Event*)
    { return Steinberg_kResultFalse; }

    inline static const Steinberg_Vst_IEventListVtbl kVtbl = {
        &q, &ar, &rel, &getEventCount, &getEvent, &addEvent
    };
    EventList() : vtbl(&kVtbl) {}

    void noteOn(Steinberg_int16 pitch, float velocity, Steinberg_int32 offset)
    {
        Steinberg_Vst_Event e {};
        e.type = Steinberg_Vst_Event_EventTypes_kNoteOnEvent;
        e.sampleOffset = offset;
        e.Steinberg_Vst_Event_noteOn.pitch = pitch;
        e.Steinberg_Vst_Event_noteOn.velocity = velocity;
        events.push_back(e);
    }
    void noteOff(Steinberg_int16 pitch, Steinberg_int32 offset)
    {
        Steinberg_Vst_Event e {};
        e.type = Steinberg_Vst_Event_EventTypes_kNoteOffEvent;
        e.sampleOffset = offset;
        e.Steinberg_Vst_Event_noteOff.pitch = pitch;
        events.push_back(e);
    }
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
        std::printf("usage: vst3_smoke_host <module> "
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

    using GetFactoryFn = Steinberg_IPluginFactory* (SMTG_STDMETHODCALLTYPE*)();
    auto getFactory = reinterpret_cast<GetFactoryFn>(moduleSym(mod, "GetPluginFactory"));
    expect(getFactory != nullptr, "GetPluginFactory exported");
    if (!getFactory) return 1;

    Steinberg_IPluginFactory* factory = getFactory();
    expect(factory != nullptr, "factory returned");
    if (!factory) return 1;

    Steinberg_PFactoryInfo finfo {};
    expect(factory->lpVtbl->getFactoryInfo(factory, &finfo) == Steinberg_kResultOk,
           "getFactoryInfo");
    std::printf("      vendor: %s\n", finfo.vendor);

    expect(factory->lpVtbl->countClasses(factory) == 1, "one class exported");

    Steinberg_PClassInfo cls {};
    expect(factory->lpVtbl->getClassInfo(factory, 0, &cls) == Steinberg_kResultOk,
           "getClassInfo");
    std::printf("      class:  %s (%s)\n", cls.name, cls.category);
    expect(std::strcmp(cls.category, "Audio Module Class") == 0, "category is audio module");

    // --- instantiate as IComponent -------------------------------------------
    void* raw = nullptr;
    expect(factory->lpVtbl->createInstance(factory, cls.cid,
               reinterpret_cast<Steinberg_FIDString>(
                   const_cast<Steinberg_int8*>(Steinberg_Vst_IComponent_iid)),
               &raw) == Steinberg_kResultOk && raw != nullptr,
           "createInstance(IComponent)");
    if (!raw) return 1;
    auto* comp = static_cast<Steinberg_Vst_IComponent*>(raw);

    HostRunLoop hostRunLoop;
    HostContext hostContext;
    hostContext.runLoop = &hostRunLoop;
    expect(comp->lpVtbl->initialize(comp,
               reinterpret_cast<Steinberg_FUnknown*>(&hostContext)) == Steinberg_kResultOk,
           "initialize");

    void* rawProc = nullptr;
    expect(comp->lpVtbl->queryInterface(comp, Steinberg_Vst_IAudioProcessor_iid, &rawProc)
               == Steinberg_kResultOk && rawProc,
           "queryInterface(IAudioProcessor)");
    auto* proc = static_cast<Steinberg_Vst_IAudioProcessor*>(rawProc);

    void* rawCtrl = nullptr;
    expect(comp->lpVtbl->queryInterface(comp, Steinberg_Vst_IEditController_iid, &rawCtrl)
               == Steinberg_kResultOk && rawCtrl,
           "queryInterface(IEditController)");
    auto* ctrl = static_cast<Steinberg_Vst_IEditController*>(rawCtrl);

    HostHandler handler;
    expect(ctrl->lpVtbl->setComponentHandler(ctrl,
               reinterpret_cast<Steinberg_Vst_IComponentHandler*>(&handler))
               == Steinberg_kResultOk,
           "setComponentHandler");

    // --- buses & parameters ----------------------------------------------------
    const Steinberg_int32 expectedInBuses =
        expectInstrument ? 0 : (expectSidechain ? 2 : 1);
    const Steinberg_int32 inBusCount = comp->lpVtbl->getBusCount(
        comp, Steinberg_Vst_MediaTypes_kAudio, Steinberg_Vst_BusDirections_kInput);
    expect(inBusCount == expectedInBuses,
           expectInstrument ? "instrument has no audio input buses"
           : (expectSidechain ? "main + sidechain input buses"
                              : "one audio input bus"));
    Steinberg_Vst_BusInfo bus {};
    expect(comp->lpVtbl->getBusInfo(comp, Steinberg_Vst_MediaTypes_kAudio,
                                    Steinberg_Vst_BusDirections_kOutput, 0, &bus)
               == Steinberg_kResultOk && bus.channelCount == 2,
           "stereo output bus");
    if (expectSidechain)
    {
        Steinberg_Vst_BusInfo aux {};
        expect(comp->lpVtbl->getBusInfo(comp, Steinberg_Vst_MediaTypes_kAudio,
                                        Steinberg_Vst_BusDirections_kInput, 1, &aux)
                   == Steinberg_kResultOk && aux.channelCount == 2
                   && aux.busType == Steinberg_Vst_BusTypes_kAux,
               "sidechain bus is a stereo aux bus");
        expect(comp->lpVtbl->activateBus(comp, Steinberg_Vst_MediaTypes_kAudio,
                                         Steinberg_Vst_BusDirections_kInput, 1, 1)
                   == Steinberg_kResultOk,
               "sidechain bus activates");
    }
    if (expectInstrument || probeMode)
    {
        const Steinberg_int32 eventBuses = comp->lpVtbl->getBusCount(
            comp, Steinberg_Vst_MediaTypes_kEvent, Steinberg_Vst_BusDirections_kInput);
        expect(eventBuses == 1, "one MIDI event input bus");
    }

    const Steinberg_int32 numParams = ctrl->lpVtbl->getParameterCount(ctrl);
    expect(numParams >= 1, "has parameters");
    std::printf("      params: %d (incl. bypass)\n", numParams);
    bool sawBypass = false;
    for (Steinberg_int32 i = 0; i < numParams; ++i)
    {
        Steinberg_Vst_ParameterInfo pi {};
        if (ctrl->lpVtbl->getParameterInfo(ctrl, i, &pi) != Steinberg_kResultOk)
        {
            expect(false, "getParameterInfo");
            break;
        }
        if (pi.flags & Steinberg_Vst_ParameterInfo_ParameterFlags_kIsBypass)
            sawBypass = true;
        // normalized <-> plain must round-trip
        const double n0 = 0.37;
        const double plain = ctrl->lpVtbl->normalizedParamToPlain(ctrl, pi.id, n0);
        const double n1 = ctrl->lpVtbl->plainParamToNormalized(ctrl, pi.id, plain);
        if (pi.stepCount == 0 && std::fabs(n1 - n0) > 1e-9)
        {
            expect(false, "param normalization round-trip");
            break;
        }
        // Every position of a discrete parameter must survive the host's
        // value -> string -> value round-trip (labels, On/Off, whole numbers).
        for (Steinberg_int32 k = 0; pi.stepCount > 0 && k <= pi.stepCount && k < 64; ++k)
        {
            const double v = static_cast<double>(k) / pi.stepCount;
            Steinberg_Vst_String128 text {};
            Steinberg_Vst_ParamValue back = -1.0;
            if (ctrl->lpVtbl->getParamStringByValue(ctrl, pi.id, v, text) != Steinberg_kResultOk
                || ctrl->lpVtbl->getParamValueByString(ctrl, pi.id, text, &back)
                       != Steinberg_kResultOk
                || std::fabs(back - v) > 1e-9)
            {
                std::printf("      param %u position %d does not round-trip (%.3f)\n",
                            static_cast<unsigned>(pi.id), static_cast<int>(k), back);
                expect(false, "stepped parameter text round-trips");
                break;
            }
        }
    }
    expect(sawBypass, "bypass parameter present");

    // --- processing -------------------------------------------------------------
    Steinberg_Vst_ProcessSetup setup {};
    setup.processMode = 0;
    setup.symbolicSampleSize = Steinberg_Vst_SymbolicSampleSizes_kSample32;
    setup.maxSamplesPerBlock = 512;
    setup.sampleRate = 48000.0;
    expect(proc->lpVtbl->setupProcessing(proc, &setup) == Steinberg_kResultOk,
           "setupProcessing");
    expect(comp->lpVtbl->setActive(comp, 1) == Steinberg_kResultOk, "setActive(true)");
    expect(proc->lpVtbl->setProcessing(proc, 1) == Steinberg_kResultOk, "setProcessing(true)");

    std::vector<float> inL(512), inR(512), outL(512), outR(512);
    std::vector<float> scL(512, 0.0f), scR(512, 0.0f);
    float* inCh[2]  = { inL.data(), inR.data() };
    float* outCh[2] = { outL.data(), outR.data() };
    float* scCh[2]  = { scL.data(), scR.data() };

    Steinberg_Vst_AudioBusBuffers inputBuses[2] {};
    inputBuses[0].numChannels = 2;
    inputBuses[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 = inCh;
    inputBuses[1].numChannels = 2;
    inputBuses[1].Steinberg_Vst_AudioBusBuffers_channelBuffers32 = scCh;
    Steinberg_Vst_AudioBusBuffers outBusBuf {};
    outBusBuf.numChannels = 2;
    outBusBuf.Steinberg_Vst_AudioBusBuffers_channelBuffers32 = outCh;

    ParamChanges changes;
    EventList events;
    Steinberg_Vst_ProcessContext context {};
    context.sampleRate = 48000.0;

    Steinberg_Vst_ProcessData data {};
    data.processMode = 0;
    data.symbolicSampleSize = Steinberg_Vst_SymbolicSampleSizes_kSample32;
    data.numSamples = 512;
    data.numInputs = expectedInBuses;
    data.numOutputs = 1;
    data.inputs = expectedInBuses > 0 ? inputBuses : nullptr;
    data.outputs = &outBusBuf;
    data.inputParameterChanges =
        reinterpret_cast<Steinberg_Vst_IParameterChanges*>(&changes);
    data.inputEvents = reinterpret_cast<Steinberg_Vst_IEventList*>(&events);

    auto processOnce = [&]() {
        const Steinberg_tresult r = proc->lpVtbl->process(proc, &data);
        changes.clear();
        events.events.clear();
        return r == Steinberg_kResultOk;
    };

    // Runs the 1 kHz test tone through `blocks` blocks with the sidechain
    // fed `scLevel` (0 = silent key) and returns the settled output energy.
    int n = 0;
    bool finite = true;
    auto runBlocks = [&](int blocks, float scLevel) -> double {
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
            if (!processOnce())
            {
                expect(false, "process returns kResultOk");
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
        const double energy = runBlocks(40, 0.0f);
        expect(finite, "output stays finite");
        expect(energy > 1e-6, "output carries signal");

        if (expectSidechain)
        {
            // The functional proof: a hot key on the aux bus must duck the
            // main output well below the silent-key level measured above.
            const double ducked = runBlocks(40, 0.9f);
            std::printf("      sidechain: open %.3g -> ducked %.3g\n", energy, ducked);
            expect(finite, "ducked output stays finite");
            expect(ducked < energy * 0.5, "hot sidechain ducks the main output");
            runBlocks(20, 0.0f);   // let the detector release before moving on
        }
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
        // A note must produce sound at the right pitch, then decay on release.
        // Select the sine waveform first for a clean zero-crossing estimate.
        ctrl->lpVtbl->setParamNormalized(ctrl, hash32("wave"), 0.0);
        events.noteOn(69, 1.0f, 0);
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
    }

    if (probeMode)
    {
        std::printf("      -- probe: host contract battery --\n");
        const Steinberg_Vst_ParamID gainId = hash32("gain");
        const Steinberg_Vst_ParamID lookaheadId = hash32("lookahead");
        const Steinberg_Vst_ParamID programId = 0x5052474Du;   // 'PRGM'

        // 1) Transport: while playing, the probe outputs DC = tempo/1000.
        void* rawReq = nullptr;
        expect(comp->lpVtbl->queryInterface(comp,
                   Steinberg_Vst_IProcessContextRequirements_iid, &rawReq)
                   == Steinberg_kResultOk && rawReq != nullptr,
               "queryInterface(IProcessContextRequirements)");
        if (rawReq != nullptr)
        {
            auto* req = static_cast<Steinberg_Vst_IProcessContextRequirements*>(rawReq);
            const Steinberg_uint32 need =
                req->lpVtbl->getProcessContextRequirements(req);
            expect((need & Steinberg_Vst_IProcessContextRequirements_Flags_kNeedTempo)
                       != 0,
                   "plugin declares it needs the tempo");
            req->lpVtbl->release(req);
        }
        fillInput(0.0f);
        context.state = Steinberg_Vst_ProcessContext_StatesAndFlags_kPlaying
                      | Steinberg_Vst_ProcessContext_StatesAndFlags_kTempoValid
                      | Steinberg_Vst_ProcessContext_StatesAndFlags_kProjectTimeMusicValid;
        context.tempo = 137.5;
        data.processContext = &context;
        processOnce();
        processOnce();
        std::printf("      transport DC: %.4f (expected 0.1375)\n", outMean());
        expect(std::fabs(outMean() - 0.1375) < 1e-3,
               "transport tempo reaches the plugin (DC = tempo/1000)");

        // 2) Offline mode: the probe flips the DC sign under kOffline.
        proc->lpVtbl->setProcessing(proc, 0);
        comp->lpVtbl->setActive(comp, 0);
        setup.processMode = Steinberg_Vst_ProcessModes_kOffline;
        expect(proc->lpVtbl->setupProcessing(proc, &setup) == Steinberg_kResultOk,
               "setupProcessing(kOffline)");
        comp->lpVtbl->setActive(comp, 1);
        proc->lpVtbl->setProcessing(proc, 1);
        processOnce();
        processOnce();
        expect(std::fabs(outMean() + 0.1375) < 1e-3,
               "offline render mode reaches the plugin (DC sign flips)");
        proc->lpVtbl->setProcessing(proc, 0);
        comp->lpVtbl->setActive(comp, 0);
        setup.processMode = 0;
        proc->lpVtbl->setupProcessing(proc, &setup);
        comp->lpVtbl->setActive(comp, 1);
        proc->lpVtbl->setProcessing(proc, 1);

        // 3) Sample-accurate automation: a gain step queued at offset 256
        //    must land exactly there (256 is on the quantum grid). Stop the
        //    transport first (kept valid, like a real host) so the probe's
        //    tempo DC does not ride on top of the step.
        context.state = Steinberg_Vst_ProcessContext_StatesAndFlags_kTempoValid;
        processOnce();   // deliver the stopped transport
        fillInput(1.0f);
        changes.add(gainId).points.push_back({ 0, 0.0 });
        processOnce();   // settle at gain 0
        changes.add(gainId).points.push_back({ 256, 1.0 });
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

        // 4) MIDI note events: silence the input path, play A4, check pitch
        //    and the in-block start offset.
        fillInput(0.0f);
        changes.add(gainId).points.push_back({ 0, 0.0 });
        events.noteOn(69, 1.0f, 256);
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

        // 5) Pitch bend through IMidiMapping: full bend = +2 semitones.
        void* rawMap = nullptr;
        expect(ctrl->lpVtbl->queryInterface(ctrl, Steinberg_Vst_IMidiMapping_iid,
                                            &rawMap) == Steinberg_kResultOk
                   && rawMap != nullptr,
               "queryInterface(IMidiMapping)");
        if (rawMap != nullptr)
        {
            auto* map = static_cast<Steinberg_Vst_IMidiMapping*>(rawMap);
            Steinberg_Vst_ParamID bendId = 0;
            expect(map->lpVtbl->getMidiControllerAssignment(map, 0, 0,
                       129 /* kPitchBend */, &bendId) == Steinberg_kResultTrue,
                   "pitch bend maps to a proxy parameter");
            map->lpVtbl->release(map);
            changes.add(bendId).points.push_back({ 0, 1.0 });
            processOnce();
            tone.clear();
            for (int block = 0; block < 8; ++block)
            {
                processOnce();
                for (int i = 0; i < 512; ++i)
                    tone.push_back(outL[static_cast<size_t>(i)]);
            }
            const double bent = estimateFrequency(tone.data(),
                                                  static_cast<int>(tone.size()),
                                                  48000.0);
            std::printf("      bent pitch: %.1f Hz (expected 493.9)\n", bent);
            expect(std::fabs(bent - 493.9) < 25.0, "pitch bend reaches the plugin");
            changes.add(bendId).points.push_back({ 0, 0.5 });   // re-center
        }
        events.noteOff(69, 0);
        processOnce();

        // 6) Latency change: flipping the lookahead toggle must notify the
        //    host and re-report.
        expect(proc->lpVtbl->getLatencySamples(proc) == 0, "initial latency is 0");
        handler.restartFlags = 0;
        ctrl->lpVtbl->setParamNormalized(ctrl, lookaheadId, 1.0);
        expect((handler.restartFlags
                & Steinberg_Vst_RestartFlags_kLatencyChanged) != 0,
               "latency change notifies restartComponent(kLatencyChanged)");
        expect(proc->lpVtbl->getLatencySamples(proc) == 64,
               "new latency is reported");
        ctrl->lpVtbl->setParamNormalized(ctrl, lookaheadId, 0.0);

        // 7) Mono negotiation: 1-in/1-out must be accepted and processed.
        proc->lpVtbl->setProcessing(proc, 0);
        comp->lpVtbl->setActive(comp, 0);
        Steinberg_Vst_SpeakerArrangement monoIn = Steinberg_Vst_SpeakerArr_kMono;
        Steinberg_Vst_SpeakerArrangement monoOut = Steinberg_Vst_SpeakerArr_kMono;
        expect(proc->lpVtbl->setBusArrangements(proc, &monoIn, 1, &monoOut, 1)
                   == Steinberg_kResultTrue,
               "mono bus arrangement accepted");
        Steinberg_Vst_BusInfo monoBus {};
        comp->lpVtbl->getBusInfo(comp, Steinberg_Vst_MediaTypes_kAudio,
                                 Steinberg_Vst_BusDirections_kOutput, 0, &monoBus);
        expect(monoBus.channelCount == 1, "bus info reports mono");
        comp->lpVtbl->setActive(comp, 1);
        proc->lpVtbl->setProcessing(proc, 1);
        inputBuses[0].numChannels = 1;
        outBusBuf.numChannels = 1;
        fillInput(0.5f);
        ctrl->lpVtbl->setParamNormalized(ctrl, gainId, 1.0);
        processOnce();
        processOnce();
        expect(std::fabs(outMean() - 0.5) < 1e-3, "mono processing works");
        proc->lpVtbl->setProcessing(proc, 0);
        comp->lpVtbl->setActive(comp, 0);
        Steinberg_Vst_SpeakerArrangement stereoArr = Steinberg_Vst_SpeakerArr_kStereo;
        proc->lpVtbl->setBusArrangements(proc, &stereoArr, 1, &stereoArr, 1);
        inputBuses[0].numChannels = 2;
        outBusBuf.numChannels = 2;
        comp->lpVtbl->setActive(comp, 1);
        proc->lpVtbl->setProcessing(proc, 1);

        // 8) Factory presets: program list + program change parameter.
        void* rawUnit = nullptr;
        expect(ctrl->lpVtbl->queryInterface(ctrl, Steinberg_Vst_IUnitInfo_iid, &rawUnit)
                   == Steinberg_kResultOk && rawUnit != nullptr,
               "queryInterface(IUnitInfo)");
        if (rawUnit != nullptr)
        {
            auto* unit = static_cast<Steinberg_Vst_IUnitInfo*>(rawUnit);
            expect(unit->lpVtbl->getProgramListCount(unit) == 1,
                   "one factory program list");
            Steinberg_Vst_ProgramListInfo listInfo {};
            expect(unit->lpVtbl->getProgramListInfo(unit, 0, &listInfo)
                       == Steinberg_kResultOk && listInfo.programCount == 2,
                   "program list reports two presets");
            Steinberg_Vst_String128 progName {};
            expect(unit->lpVtbl->getProgramName(unit, listInfo.id, 1, progName)
                       == Steinberg_kResultOk && progName[0] == 'H',
                   "second preset is named Half");
            unit->lpVtbl->release(unit);
            handler.restartFlags = 0;
            ctrl->lpVtbl->setParamNormalized(ctrl, programId, 1.0);
            expect(std::fabs(ctrl->lpVtbl->getParamNormalized(ctrl, gainId) - 0.5)
                       < 1e-9,
                   "loading the preset applies its parameter values");
            expect((handler.restartFlags
                    & Steinberg_Vst_RestartFlags_kParamValuesChanged) != 0,
                   "preset load notifies kParamValuesChanged");

            // The active program must survive a state round-trip (the
            // pluginval state-restoration test demands it).
            MemoryStream programState;
            comp->lpVtbl->getState(comp,
                reinterpret_cast<Steinberg_IBStream*>(&programState));
            ctrl->lpVtbl->setParamNormalized(ctrl, programId, 0.0);
            MemoryStream programRestore;
            programRestore.bytes = programState.bytes;
            programRestore.pos = 0;
            comp->lpVtbl->setState(comp,
                reinterpret_cast<Steinberg_IBStream*>(&programRestore));
            expect(std::fabs(ctrl->lpVtbl->getParamNormalized(ctrl, programId) - 1.0)
                       < 1e-9,
                   "program selection survives the state round-trip");
            ctrl->lpVtbl->setParamNormalized(ctrl, programId, 0.0);
        }

        // 9) Bypass under latency: with the lookahead engaged the probe
        //    delays its input by 64 samples. Hosts keep compensating that
        //    latency while the plugin is bypassed, so the bypassed output -
        //    and every sample of the crossfade - must stay exactly as late.
        constexpr Steinberg_Vst_ParamID bypassId = 0x42595053u;   // 'BYPS'
        changes.add(gainId).points.push_back({ 0, 1.0 });
        changes.add(lookaheadId).points.push_back({ 0, 1.0 });
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
            if (block == 2) changes.add(bypassId).points.push_back({ 100, 1.0 });
            if (block == 4) changes.add(bypassId).points.push_back({ 300, 0.0 });
            processOnce();
            got.insert(got.end(), outR.begin(), outR.end());
        }
        double worstLag = 0.0;
        for (size_t i = 512; i < got.size(); ++i)   // block 0 re-reports the latency
            worstLag = std::fmax(worstLag, std::fabs(got[i] - fed[i - 64]));
        std::printf("      bypass under latency: worst misalignment %.2g\n", worstLag);
        expect(worstLag < 1e-6, "bypassed output stays latency-aligned");
        changes.add(lookaheadId).points.push_back({ 0, 0.0 });
        processOnce();

        // 10) Oversize block: a host passing more frames than
        //     maxSamplesPerBlock must still get the whole block processed,
        //     automation included (the step lands in the third chunk).
        {
            constexpr int kBig = 1200;
            std::vector<float> bigInL(kBig, 1.0f), bigInR(kBig, 1.0f),
                               bigOutL(kBig, -7.0f), bigOutR(kBig, -7.0f);
            inCh[0] = bigInL.data(); inCh[1] = bigInR.data();
            outCh[0] = bigOutL.data(); outCh[1] = bigOutR.data();
            data.numSamples = kBig;
            changes.add(gainId).points.push_back({ 0, 0.0 });
            changes.add(gainId).points.push_back({ 1024, 1.0 });
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
            data.numSamples = 512;
            changes.add(gainId).points.push_back({ 0, 1.0 });
            processOnce();
        }

        // 11) Narrower input bus than output (a host ignoring the
        //     arrangement): the wrapper must not read past the input's
        //     channel array, and the missing channel repeats the one given.
        {
            float* monoIn[1] = { inL.data() };
            inputBuses[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 = monoIn;
            inputBuses[0].numChannels = 1;
            fillInput(0.5f);
            processOnce();
            processOnce();
            double meanR = 0.0;
            for (int i = 0; i < 512; ++i) meanR += outR[static_cast<size_t>(i)];
            expect(std::fabs(outMean() - 0.5) < 1e-3 && std::fabs(meanR / 512.0 - 0.5) < 1e-3,
                   "mono input bus feeds both output channels");
            inputBuses[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 = inCh;
            inputBuses[0].numChannels = 2;
        }

        // 12) Latency change detected on the AUDIO thread (automation moves
        //     the lookahead): VST3 allows restartComponent on the UI thread
        //     only, so the plugin must defer it to its UI tick - never call
        //     the host from process(), yet still deliver promptly.
        {
            handler.restartFlags = 0;
            handler.offThreadRestarts = 0;
            changes.add(lookaheadId).points.push_back({ 0, 1.0 });
            std::thread audioThread([&] { processOnce(); });
            audioThread.join();
            expect(handler.offThreadRestarts == 0,
                   "no restartComponent from the audio thread");
            const auto deadline = std::chrono::steady_clock::now()
                                + std::chrono::seconds(2);
            while ((handler.restartFlags & Steinberg_Vst_RestartFlags_kLatencyChanged) == 0
                   && std::chrono::steady_clock::now() < deadline)
                pumpUiThread(hostRunLoop);
            expect((handler.restartFlags & Steinberg_Vst_RestartFlags_kLatencyChanged) != 0
                       && handler.offThreadRestarts == 0,
                   "audio-thread latency change reaches the host on the UI thread");
            expect(proc->lpVtbl->getLatencySamples(proc) == 64,
                   "deferred latency is reported");
            ctrl->lpVtbl->setParamNormalized(ctrl, lookaheadId, 0.0);   // UI thread
        }
        data.processContext = nullptr;
        context.state = 0;
    }

    // a parameter poke through the controller while "running"
    Steinberg_Vst_ParameterInfo p0 {};
    ctrl->lpVtbl->getParameterInfo(ctrl, 0, &p0);
    expect(ctrl->lpVtbl->setParamNormalized(ctrl, p0.id, 0.8) == Steinberg_kResultOk,
           "setParamNormalized");
    expect(std::fabs(ctrl->lpVtbl->getParamNormalized(ctrl, p0.id) - 0.8) < 1e-9,
           "getParamNormalized reflects the change");

    // --- state round-trip ---------------------------------------------------------
    MemoryStream saved;
    expect(comp->lpVtbl->getState(comp,
               reinterpret_cast<Steinberg_IBStream*>(&saved)) == Steinberg_kResultOk
               && !saved.bytes.empty(),
           "getState produces a blob");

    expect(ctrl->lpVtbl->setParamNormalized(ctrl, p0.id, 0.1) == Steinberg_kResultOk,
           "perturb parameter");

    MemoryStream restore;
    restore.bytes = saved.bytes;
    restore.pos = 0;
    expect(comp->lpVtbl->setState(comp,
               reinterpret_cast<Steinberg_IBStream*>(&restore)) == Steinberg_kResultOk,
           "setState accepts the blob");
    expect(std::fabs(ctrl->lpVtbl->getParamNormalized(ctrl, p0.id) - 0.8) < 1e-9,
           "state round-trip restores the parameter");

    // --- editor view (WebView editor layer; optional) -------------------------------
    // Contract-only: create/inspect/release without attaching a real window,
    // so this also runs on headless CI. A nullptr view is valid (no editor).
    if (Steinberg_IPlugView* view = ctrl->lpVtbl->createView(ctrl, "editor"))
    {
#if defined(_WIN32)
        const Steinberg_FIDString platformType = "HWND";
#elif defined(__APPLE__)
        const Steinberg_FIDString platformType = "NSView";
#else
        const Steinberg_FIDString platformType = "X11EmbedWindowID";
#endif
        expect(view->lpVtbl->isPlatformTypeSupported(view, platformType)
                   == Steinberg_kResultTrue,
               "view supports the native platform type");

        Steinberg_ViewRect rect {};
        expect(view->lpVtbl->getSize(view, &rect) == Steinberg_kResultOk
                   && rect.right > rect.left && rect.bottom > rect.top,
               "view reports a usable size");
        std::printf("      editor: %d x %d%s\n",
                    rect.right - rect.left, rect.bottom - rect.top,
                    view->lpVtbl->canResize(view) == Steinberg_kResultTrue
                        ? " (resizable)" : "");

        Steinberg_ViewRect constrained = rect;
        expect(view->lpVtbl->checkSizeConstraint(view, &constrained)
                   == Steinberg_kResultTrue,
               "checkSizeConstraint accepts the reported size");

        void* sameView = nullptr;
        expect(view->lpVtbl->queryInterface(view, Steinberg_IPlugView_iid, &sameView)
                   == Steinberg_kResultOk && sameView == view,
               "view queryInterface(IPlugView)");
        if (sameView != nullptr) view->lpVtbl->release(view);
        expect(view->lpVtbl->release(view) == 0, "view refcount returns to zero");
    }
    else
        std::printf("      editor: none (host shows its generic UI)\n");

    // --- teardown -------------------------------------------------------------------
    expect(proc->lpVtbl->setProcessing(proc, 0) == Steinberg_kResultOk, "setProcessing(false)");
    expect(comp->lpVtbl->setActive(comp, 0) == Steinberg_kResultOk, "setActive(false)");
    expect(comp->lpVtbl->terminate(comp) == Steinberg_kResultOk, "terminate");

    proc->lpVtbl->release(proc);
    ctrl->lpVtbl->release(ctrl);
    const Steinberg_uint32 left = comp->lpVtbl->release(comp);
    expect(left == 0, "refcount returns to zero");

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "SMOKE OK" : "SMOKE FAILED",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
