// DSPark — VST3 editor host
//
// Opens a plugin's editor view in a plain native window — the 10-second
// visual check for WebView editors without launching a DAW:
//
//   vst3_editor_host <path-to-module> [seconds]
//
// Loads the module, creates the single component, asks the controller for
// its "editor" view, attaches it to a resizable top-level window and pumps
// messages until you close it (or the optional timeout fires, for scripted
// use). A minimal IComponentHandler logs begin/perform/endEdit calls, so
// dragging a control shows the full UI -> gesture -> parameter path live.
//
// Windows-only for now (the WebView editor's first-class dev platform);
// on other systems it reports so and exits 0.

#include <cstdio>

#if !defined(_WIN32)

int main()
{
    std::printf("vst3_editor_host: only implemented on Windows so far.\n");
    return 0;
}

#else

#include "../plugin/vst3/vst3_c_api.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdlib>
#include <cstring>

namespace {

// -- minimal IComponentHandler: logs automation gestures ---------------------------

struct LoggingHandler
{
    const Steinberg_Vst_IComponentHandlerVtbl* vtbl;

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE q(void* self_, const Steinberg_TUID,
                                                      void** obj)
    { if (obj) *obj = self_; return Steinberg_kResultOk; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE ar(void*)  { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE rel(void*) { return 100; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE beginEdit(void*, Steinberg_Vst_ParamID id)
    {
        std::printf("  beginEdit   0x%08x\n", id);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE performEdit(void*,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue value)
    {
        std::printf("  performEdit 0x%08x = %.4f\n", id, value);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE endEdit(void*, Steinberg_Vst_ParamID id)
    {
        std::printf("  endEdit     0x%08x\n", id);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE restart(void*, Steinberg_int32 flags)
    {
        std::printf("  restartComponent flags=0x%x\n", flags);
        return Steinberg_kResultOk;
    }

    inline static const Steinberg_Vst_IComponentHandlerVtbl kVtbl = {
        &q, &ar, &rel, &beginEdit, &performEdit, &endEdit, &restart
    };
    LoggingHandler() : vtbl(&kVtbl) {}
};

Steinberg_IPlugView* g_view = nullptr;

LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
{
    switch (msg)
    {
    case WM_SIZE:
        if (g_view != nullptr)
        {
            Steinberg_ViewRect r { 0, 0, LOWORD(lp), HIWORD(lp) };
            g_view->lpVtbl->onSize(g_view, &r);
        }
        return 0;
    case WM_TIMER:
    case WM_CLOSE:
        PostQuitMessage(0);
        return 0;
    default:
        return DefWindowProcW(hwnd, msg, wp, lp);
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: vst3_editor_host <module> [seconds]\n");
        return 2;
    }
    const int autoCloseSeconds = argc >= 3 ? std::atoi(argv[2]) : 0;

    HMODULE mod = LoadLibraryA(argv[1]);
    if (mod == nullptr) { std::printf("FAIL module load\n"); return 1; }

    using GetFactoryFn = Steinberg_IPluginFactory* (SMTG_STDMETHODCALLTYPE*)();
    auto getFactory = reinterpret_cast<GetFactoryFn>(GetProcAddress(mod, "GetPluginFactory"));
    if (getFactory == nullptr) { std::printf("FAIL GetPluginFactory\n"); return 1; }
    Steinberg_IPluginFactory* factory = getFactory();

    Steinberg_PClassInfo cls {};
    if (factory->lpVtbl->getClassInfo(factory, 0, &cls) != Steinberg_kResultOk)
    { std::printf("FAIL getClassInfo\n"); return 1; }
    std::printf("plugin: %s\n", cls.name);

    void* raw = nullptr;
    factory->lpVtbl->createInstance(factory, cls.cid,
        reinterpret_cast<Steinberg_FIDString>(
            const_cast<Steinberg_int8*>(Steinberg_Vst_IComponent_iid)), &raw);
    if (raw == nullptr) { std::printf("FAIL createInstance\n"); return 1; }
    auto* comp = static_cast<Steinberg_Vst_IComponent*>(raw);
    comp->lpVtbl->initialize(comp, nullptr);

    void* rawCtrl = nullptr;
    comp->lpVtbl->queryInterface(comp, Steinberg_Vst_IEditController_iid, &rawCtrl);
    auto* ctrl = static_cast<Steinberg_Vst_IEditController*>(rawCtrl);

    LoggingHandler handler;
    ctrl->lpVtbl->setComponentHandler(ctrl,
        reinterpret_cast<Steinberg_Vst_IComponentHandler*>(&handler));

    Steinberg_IPlugView* view = ctrl->lpVtbl->createView(ctrl, "editor");
    if (view == nullptr)
    {
        std::printf("no editor view (plugin uses the host's generic UI)\n");
        ctrl->lpVtbl->release(ctrl);
        comp->lpVtbl->terminate(comp);
        comp->lpVtbl->release(comp);
        return 0;
    }

    Steinberg_ViewRect rect {};
    view->lpVtbl->getSize(view, &rect);
    const int width  = rect.right - rect.left;
    const int height = rect.bottom - rect.top;
    std::printf("editor: %d x %d\n", width, height);

    WNDCLASSW wc {};
    wc.lpfnWndProc = &wndProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"DSParkEditorHost";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    const DWORD style = view->lpVtbl->canResize(view) == Steinberg_kResultTrue
                      ? WS_OVERLAPPEDWINDOW : (WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX));
    RECT frame { 0, 0, width, height };
    AdjustWindowRect(&frame, style, FALSE);
    HWND hwnd = CreateWindowW(L"DSParkEditorHost", L"DSPark editor host", style,
                              CW_USEDEFAULT, CW_USEDEFAULT,
                              frame.right - frame.left, frame.bottom - frame.top,
                              nullptr, nullptr, wc.hInstance, nullptr);
    if (hwnd == nullptr) { std::printf("FAIL CreateWindow\n"); return 1; }

    if (view->lpVtbl->attached(view, hwnd, "HWND") != Steinberg_kResultOk)
    {
        std::printf("FAIL attached (is the WebView2 runtime installed?)\n");
        DestroyWindow(hwnd);
        view->lpVtbl->release(view);
        ctrl->lpVtbl->release(ctrl);
        comp->lpVtbl->terminate(comp);
        comp->lpVtbl->release(comp);
        return 1;
    }
    std::printf("attached; close the window to exit\n");
    g_view = view;

    ShowWindow(hwnd, SW_SHOW);
    if (autoCloseSeconds > 0)
        SetTimer(hwnd, 1, static_cast<UINT>(autoCloseSeconds) * 1000u, nullptr);

    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0)
    {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    g_view = nullptr;
    view->lpVtbl->removed(view);
    DestroyWindow(hwnd);
    const Steinberg_uint32 viewRefs = view->lpVtbl->release(view);
    ctrl->lpVtbl->release(ctrl);
    comp->lpVtbl->terminate(comp);
    const Steinberg_uint32 compRefs = comp->lpVtbl->release(comp);
    std::printf("teardown ok (view refs %u, component refs %u)\n", viewRefs, compRefs);
    return (viewRefs == 0 && compRefs == 0) ? 0 : 1;
}

#endif // _WIN32
