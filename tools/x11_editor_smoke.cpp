// DSPark — X11 WebView editor smoke host (Linux)
//
// Loads a .vst3 module and exercises the Linux editor path exactly like an
// X11 host: createView -> setFrame (offering a real IRunLoop) -> attached to
// an X11 window created here -> drives the registered timer so GTK breathes
// -> verifies the embedded WebKitGTK page completed the JS bridge handshake
// (through the DSPARK_WEBVIEW_LOG diagnostic trace) -> removed/teardown.
// Plugins without an editor are checked to answer createView with null.
// Exit code 0 means every step behaved. Companion to tools/au_editor_smoke
// (macOS) and tools/vst3_editor_host (Windows).
//
//   x11_editor_smoke <module.vst3> --expect-editor <W>x<H>
//   x11_editor_smoke <module.vst3> --expect-no-editor
//
// Needs a running X server (CI wraps it in xvfb-run) and the WebKitGTK
// runtime the editor layer dlopens (libwebkit2gtk-4.1 or -4.0).

#if !defined(__linux__)

#include <cstdio>
int main()
{
    std::printf("x11_editor_smoke: Linux only (no-op on this platform)\n");
    return 0;
}

#else

#include "../plugin/vst3/vst3_c_api.h"

#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

namespace {

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "PASS " : "FAIL ", what);
    if (!ok) ++g_failures;
}

// -- diagnostic-log probe (same trace the AU smoke uses) ----------------------------

void logPath(char* out, size_t cap)
{
    const char* dir = std::getenv("TMPDIR");
    std::snprintf(out, cap, "%s/DSParkWebView.log", dir != nullptr ? dir : "/tmp");
}

bool logContains(const char* needle)
{
    char path[512];
    logPath(path, sizeof(path));
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return false;
    std::string text;
    char chunk[4096];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
        text.append(chunk, got);
    std::fclose(f);
    return text.find(needle) != std::string::npos;
}

// -- minimal Xlib surface, dlopen-resolved (no libx11-dev needed to build) ----------

struct X11
{
    bool ok = false;
    void*         (*openDisplay)(const char*) = nullptr;
    unsigned long (*defaultRootWindow)(void*) = nullptr;
    unsigned long (*createSimpleWindow)(void*, unsigned long, int, int,
                                        unsigned, unsigned, unsigned,
                                        unsigned long, unsigned long) = nullptr;
    int           (*mapWindow)(void*, unsigned long) = nullptr;
    int           (*flush)(void*) = nullptr;
    int           (*destroyWindow)(void*, unsigned long) = nullptr;
    int           (*closeDisplay)(void*) = nullptr;
};

X11 loadX11()
{
    X11 x {};
    void* lib = dlopen("libX11.so.6", RTLD_LAZY | RTLD_LOCAL);
    if (lib == nullptr) return x;
    auto load = [lib](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
    };
    load(x.openDisplay, "XOpenDisplay");
    load(x.defaultRootWindow, "XDefaultRootWindow");
    load(x.createSimpleWindow, "XCreateSimpleWindow");
    load(x.mapWindow, "XMapWindow");
    load(x.flush, "XFlush");
    load(x.destroyWindow, "XDestroyWindow");
    load(x.closeDisplay, "XCloseDisplay");
    x.ok = x.openDisplay && x.defaultRootWindow && x.createSimpleWindow
        && x.mapWindow && x.flush && x.destroyWindow && x.closeDisplay;
    return x;
}

// -- host-side IPlugFrame with a real IRunLoop --------------------------------------
//
// The Linux editor only comes alive when the host offers an IRunLoop and the
// plugin registers a timer on it; this frame plays that host and lets the
// main() loop below deliver the ticks.

struct SmokeFrame
{
    const Steinberg_IPlugFrameVtbl*     frameVtbl;     // lens 0
    const Steinberg_Linux_IRunLoopVtbl* runLoopVtbl;   // lens 1

    Steinberg_Linux_ITimerHandler* timerHandler = nullptr;
    Steinberg_Linux_TimerInterval  timerMs = 0;
    int resizeCalls = 0;

    static SmokeFrame* fromLens(void* iface, int lens)
    {
        return reinterpret_cast<SmokeFrame*>(static_cast<char*>(iface)
            - static_cast<ptrdiff_t>(lens) * static_cast<ptrdiff_t>(sizeof(void*)));
    }

    void* lensPtr(int lens)
    {
        return reinterpret_cast<char*>(this)
            + static_cast<ptrdiff_t>(lens) * static_cast<ptrdiff_t>(sizeof(void*));
    }

    template <int Lens>
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sQuery(void* self_,
        const Steinberg_TUID iid, void** obj)
    {
        auto* s = fromLens(self_, Lens);
        if (obj == nullptr) return Steinberg_kInvalidArgument;
        *obj = nullptr;
        if (std::memcmp(iid, Steinberg_FUnknown_iid, sizeof(Steinberg_TUID)) == 0
            || std::memcmp(iid, Steinberg_IPlugFrame_iid, sizeof(Steinberg_TUID)) == 0)
            *obj = s->lensPtr(0);
        else if (std::memcmp(iid, Steinberg_Linux_IRunLoop_iid,
                             sizeof(Steinberg_TUID)) == 0)
            *obj = s->lensPtr(1);
        return *obj != nullptr ? Steinberg_kResultOk : Steinberg_kNoInterface;
    }

    template <int Lens>
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sAddRef(void*) { return 100; }
    template <int Lens>
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sRelease(void*) { return 100; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sResizeView(void* self_,
        Steinberg_IPlugView*, Steinberg_ViewRect*)
    {
        ++fromLens(self_, 0)->resizeCalls;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sRegisterEventHandler(void*,
        Steinberg_Linux_IEventHandler*, Steinberg_Linux_FileDescriptor)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sUnregisterEventHandler(void*,
        Steinberg_Linux_IEventHandler*)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sRegisterTimer(void* self_,
        Steinberg_Linux_ITimerHandler* handler, Steinberg_Linux_TimerInterval ms)
    {
        auto* s = fromLens(self_, 1);
        if (handler == nullptr || ms == 0) return Steinberg_kInvalidArgument;
        s->timerHandler = handler;
        s->timerMs = ms;
        reinterpret_cast<Steinberg_FUnknown*>(handler)->lpVtbl->addRef(handler);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sUnregisterTimer(void* self_,
        Steinberg_Linux_ITimerHandler* handler)
    {
        auto* s = fromLens(self_, 1);
        if (handler == nullptr || s->timerHandler != handler)
            return Steinberg_kInvalidArgument;
        reinterpret_cast<Steinberg_FUnknown*>(handler)->lpVtbl->release(handler);
        s->timerHandler = nullptr;
        return Steinberg_kResultOk;
    }

    inline static const Steinberg_IPlugFrameVtbl kFrameVtbl = {
        &sQuery<0>, &sAddRef<0>, &sRelease<0>, &sResizeView
    };
    inline static const Steinberg_Linux_IRunLoopVtbl kRunLoopVtbl = {
        &sQuery<1>, &sAddRef<1>, &sRelease<1>,
        &sRegisterEventHandler, &sUnregisterEventHandler,
        &sRegisterTimer, &sUnregisterTimer
    };

    SmokeFrame() : frameVtbl(&kFrameVtbl), runLoopVtbl(&kRunLoopVtbl) {}
};

} // namespace

int main(int argc, char** argv)
{
    bool expectEditor = false;
    int expectedW = 0, expectedH = 0;
    if (argc >= 3)
    {
        if (std::strcmp(argv[2], "--expect-editor") == 0 && argc >= 4
            && std::sscanf(argv[3], "%dx%d", &expectedW, &expectedH) == 2)
            expectEditor = true;
        else if (std::strcmp(argv[2], "--expect-no-editor") != 0)
            argc = 0;
    }
    else
        argc = 0;
    if (argc == 0)
    {
        std::fprintf(stderr, "usage: x11_editor_smoke <module.vst3> "
                             "--expect-editor <W>x<H> | --expect-no-editor\n");
        return 2;
    }

    // Enable the editor's diagnostic trace BEFORE the module loads.
    setenv("DSPARK_WEBVIEW_LOG", "1", 1);
    char path[512];
    logPath(path, sizeof(path));
    unlink(path);
    std::printf("x11_editor_smoke: %s (%s)\n", argv[1],
                expectEditor ? "expecting editor" : "expecting NO editor");

    void* mod = dlopen(argv[1], RTLD_NOW);
    expect(mod != nullptr, "module loads");
    if (mod == nullptr) return 1;

    using GetFactoryFn = Steinberg_IPluginFactory* (SMTG_STDMETHODCALLTYPE*)();
    auto getFactory = reinterpret_cast<GetFactoryFn>(dlsym(mod, "GetPluginFactory"));
    expect(getFactory != nullptr, "GetPluginFactory exported");
    if (getFactory == nullptr) return 1;
    Steinberg_IPluginFactory* factory = getFactory();
    expect(factory != nullptr, "factory returned");
    if (factory == nullptr) return 1;

    Steinberg_PClassInfo cls {};
    expect(factory->lpVtbl->getClassInfo(factory, 0, &cls) == Steinberg_kResultOk,
           "getClassInfo");

    void* raw = nullptr;
    expect(factory->lpVtbl->createInstance(factory, cls.cid,
               reinterpret_cast<Steinberg_FIDString>(
                   const_cast<Steinberg_int8*>(Steinberg_Vst_IComponent_iid)),
               &raw) == Steinberg_kResultOk && raw != nullptr,
           "createInstance(IComponent)");
    if (raw == nullptr) return 1;
    auto* comp = static_cast<Steinberg_Vst_IComponent*>(raw);
    expect(comp->lpVtbl->initialize(comp, nullptr) == Steinberg_kResultOk, "initialize");

    void* rawCtrl = nullptr;
    expect(comp->lpVtbl->queryInterface(comp, Steinberg_Vst_IEditController_iid,
                                        &rawCtrl) == Steinberg_kResultOk && rawCtrl,
           "queryInterface(IEditController)");
    auto* ctrl = static_cast<Steinberg_Vst_IEditController*>(rawCtrl);

    Steinberg_IPlugView* view = ctrl->lpVtbl->createView(ctrl, "editor");

    if (!expectEditor)
    {
        expect(view == nullptr, "createView is null (host generic UI)");
        if (view != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(view)->lpVtbl->release(view);
        ctrl->lpVtbl->release(ctrl);
        comp->lpVtbl->terminate(comp);
        comp->lpVtbl->release(comp);
        std::printf("x11_editor_smoke: %s\n",
                    g_failures == 0 ? "ALL PASS" : "FAILURES");
        return g_failures == 0 ? 0 : 1;
    }

    expect(view != nullptr, "createView returns a view");
    if (view == nullptr) return 1;

    expect(view->lpVtbl->isPlatformTypeSupported(view,
               Steinberg_kPlatformTypeX11EmbedWindowID) == Steinberg_kResultTrue,
           "X11 embed window id supported (WebKitGTK resolved)");

    SmokeFrame frame;
    expect(view->lpVtbl->setFrame(view,
               reinterpret_cast<Steinberg_IPlugFrame*>(frame.lensPtr(0)))
               == Steinberg_kResultOk,
           "setFrame (frame offers IRunLoop)");

    Steinberg_ViewRect size {};
    expect(view->lpVtbl->getSize(view, &size) == Steinberg_kResultOk,
           "getSize");
    expect(size.right - size.left == expectedW && size.bottom - size.top == expectedH,
           "size matches the declared editorSize");

    const X11 x = loadX11();
    expect(x.ok, "libX11 available");
    if (!x.ok) return 1;
    void* display = x.openDisplay(nullptr);
    expect(display != nullptr, "X display opens (xvfb)");
    if (display == nullptr) return 1;
    const unsigned long window = x.createSimpleWindow(
        display, x.defaultRootWindow(display), 0, 0,
        static_cast<unsigned>(expectedW), static_cast<unsigned>(expectedH),
        0, 0, 0);
    expect(window != 0, "host X11 window created");
    x.mapWindow(display, window);
    x.flush(display);

    expect(view->lpVtbl->attached(view,
               reinterpret_cast<void*>(window),
               Steinberg_kPlatformTypeX11EmbedWindowID) == Steinberg_kResultOk,
           "attached to the X11 window");
    expect(frame.timerHandler != nullptr, "IRunLoop timer registered");
    expect(logContains("linux create: plug"), "editor trace written");

    // Drive the plugin's timer like a host run loop until the page completes
    // the bridge handshake (cold WebKit web-process spawn can be slow in CI).
    bool handshake = false;
    for (int tick = 0; tick < 450 && !handshake; ++tick)
    {
        if (frame.timerHandler != nullptr)
            frame.timerHandler->lpVtbl->onTimer(frame.timerHandler);
        usleep(33 * 1000);
        if (tick % 15 == 14)
            handshake = logContains("page metric");
    }
    expect(handshake, "JS bridge handshake completed");

    expect(view->lpVtbl->removed(view) == Steinberg_kResultOk, "removed");
    expect(frame.timerHandler == nullptr, "timer unregistered on removed");
    view->lpVtbl->setFrame(view, nullptr);
    reinterpret_cast<Steinberg_FUnknown*>(view)->lpVtbl->release(view);

    ctrl->lpVtbl->release(ctrl);
    comp->lpVtbl->terminate(comp);
    comp->lpVtbl->release(comp);

    x.destroyWindow(display, window);
    x.closeDisplay(display);

    std::printf("x11_editor_smoke: %s\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

#endif // __linux__
