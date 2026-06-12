// DSPark — VST3 smoke host
//
// Loads a .vst3 module through the public C ABI exactly like a DAW would and
// drives the full lifecycle: factory -> class info -> instantiation ->
// initialize -> bus/parameter introspection -> setupProcessing -> setActive ->
// process (sine blocks, parameter change mid-run) -> state round-trip ->
// teardown. Exit code 0 means every step behaved.
//
//   vst3_smoke_host <path-to-module>
//
// This is the local first-line gate for plugins built with DSParkVst3.h;
// run Steinberg's validator and Tracktion's pluginval before shipping.

#include "../plugin/vst3/vst3_c_api.h"

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

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: vst3_smoke_host <module>\n");
        return 2;
    }

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

    expect(comp->lpVtbl->initialize(comp, nullptr) == Steinberg_kResultOk, "initialize");

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

    // --- buses & parameters ----------------------------------------------------
    expect(comp->lpVtbl->getBusCount(comp, Steinberg_Vst_MediaTypes_kAudio,
                                     Steinberg_Vst_BusDirections_kInput) == 1,
           "one audio input bus");
    Steinberg_Vst_BusInfo bus {};
    expect(comp->lpVtbl->getBusInfo(comp, Steinberg_Vst_MediaTypes_kAudio,
                                    Steinberg_Vst_BusDirections_kOutput, 0, &bus)
               == Steinberg_kResultOk && bus.channelCount == 2,
           "stereo output bus");

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
    float* inCh[2]  = { inL.data(), inR.data() };
    float* outCh[2] = { outL.data(), outR.data() };

    Steinberg_Vst_AudioBusBuffers inBus {};
    inBus.numChannels = 2;
    inBus.Steinberg_Vst_AudioBusBuffers_channelBuffers32 = inCh;
    Steinberg_Vst_AudioBusBuffers outBusBuf {};
    outBusBuf.numChannels = 2;
    outBusBuf.Steinberg_Vst_AudioBusBuffers_channelBuffers32 = outCh;

    Steinberg_Vst_ProcessData data {};
    data.processMode = 0;
    data.symbolicSampleSize = Steinberg_Vst_SymbolicSampleSizes_kSample32;
    data.numSamples = 512;
    data.numInputs = 1;
    data.numOutputs = 1;
    data.inputs = &inBus;
    data.outputs = &outBusBuf;

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
        if (proc->lpVtbl->process(proc, &data) != Steinberg_kResultOk)
        {
            expect(false, "process returns kResultOk");
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
