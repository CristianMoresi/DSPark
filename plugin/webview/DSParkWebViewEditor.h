// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkWebViewEditor.h
 * @brief WebView editor layer: plugin GUIs written in HTML/CSS/JS, embedded
 * in the host window through the platform web engine.
 *
 * Opt-in and additive: include this header BEFORE the format headers, set
 * `hasEditor = true` and serve your page from `editorHtml()` — the VST3,
 * CLAP and AU backends then embed it inside the window the host provides
 * (AU through a Cocoa view factory announced via kAudioUnitProperty_CocoaUI).
 * Nothing else in the plugin class changes; plugins without this header keep
 * building exactly as before.
 *
 * ```cpp
 * #include "plugin/webview/DSParkWebViewEditor.h"   // FIRST
 * #include "plugin/vst3/DSParkVst3.h"
 * #include "plugin/clap/DSParkClap.h"
 *
 * struct MyPlugin
 * {
 *     // ... descriptor / parameters / prepare / setParameter / processBlock ...
 *     static constexpr bool hasEditor = true;
 *     static constexpr dspark::plugin::EditorSize editorSize { 560, 330 };
 *     static constexpr bool editorResizable = true;          // optional
 *     static const char* editorHtml() { return R"(<!doctype html>...)"; }
 * };
 * ```
 *
 * The page talks to the DSP through a tiny injected bridge (same stable text
 * ids as state and automation):
 *
 * ```js
 * dspark.onReady(params => { ... });        // parameter table + current values
 * dspark.setParam("drive", 6.0);            // UI -> DSP (plain value)
 * dspark.onParam("drive", v => { ... });    // DSP/automation -> UI (~30 Hz)
 * dspark.beginEdit("drive");                // host automation gesture (undo)
 * dspark.endEdit("drive");
 * dspark.getParam("drive"); dspark.params;  // cached value / metadata
 * ```
 *
 * Platform engines:
 * - **Windows** — WebView2 (Edge runtime, present on Win10/11) through the
 *   vendored MIT webview library (webview/webview.h), embedded as a child
 *   window of the host's HWND. Requires exceptions enabled (plugins are).
 * - **macOS** — WKWebView created directly through the Objective-C runtime
 *   (no headers, WebKit.framework loaded at runtime) and added as a subview
 *   of the host's NSView.
 * - **Linux** — WebKitGTK (GTK3) resolved entirely through dlopen at runtime
 *   (no headers, no link-time dependency), embedded in the host's X11 window
 *   with GtkPlug/XEmbed — the same route LV2's suil takes. GTK is driven
 *   from the HOST's run loop through Editor::pump() (VST3 IRunLoop timers /
 *   CLAP timer-support); a plugin must never spin its own GTK main loop.
 *   Systems without WebKitGTK (or hosts without a usable run loop) keep the
 *   host's generic parameter UI — the same plugin binary, unchanged.
 *
 * Threading: every editor call (create/destroy/resize/JS callbacks) runs on
 * the host's main/UI thread. Values flow UI -> DSP through the wrapper's
 * normalized shadows and the user's atomic `setParameter` (any-thread safe
 * by contract), and DSP -> UI by polling those shadows at ~30 Hz from the
 * page — no native timers, no locks, no allocation on the audio thread.
 */

#if defined(DSPARK_PLUGIN_VST3_INCLUDED) || defined(DSPARK_PLUGIN_CLAP_INCLUDED) \
    || defined(DSPARK_PLUGIN_AU_INCLUDED)
#error "Include plugin/webview/DSParkWebViewEditor.h BEFORE the plugin format headers"
#endif

#define DSPARK_PLUGIN_WEBVIEW 1

#include "../DSParkPlugin.h"

#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// -- platform backend selection --------------------------------------------------
//
// DSPARK_WEBVIEW_BACKEND: 1 = WebView2 via vendored webview.h (Windows),
//                         2 = WKWebView via the Objective-C runtime (macOS),
//                         3 = WebKitGTK via dlopen + GtkPlug/XEmbed (Linux/X11),
//                         0 = stub (editor reports unavailable).

#if defined(_WIN32) && !defined(DSPARK_NO_EXCEPTIONS) \
    && (defined(__cpp_exceptions) || (defined(_CPPUNWIND) && _CPPUNWIND))
    #define DSPARK_WEBVIEW_BACKEND 1
    #ifndef NOMINMAX
    #define NOMINMAX               // webview.h pulls <windows.h>; keep min/max sane
    #endif
    #if defined(_MSC_VER)
    #pragma warning(push, 1)
    #endif
    #include "webview/webview.h"
    #if defined(_MSC_VER)
    #pragma warning(pop)
    #endif
    #include <commctrl.h>      // SetWindowSubclass: follow host resizes reliably
    #if defined(_MSC_VER)
    #pragma comment(lib, "comctl32.lib")
    #endif
    #include <memory>
#elif defined(__APPLE__)
    #define DSPARK_WEBVIEW_BACKEND 2
    #include <objc/message.h>
    #include <objc/runtime.h>
    #include <dlfcn.h>
#elif defined(__linux__)
    #define DSPARK_WEBVIEW_BACKEND 3
    #include <dlfcn.h>
    #include <cstdint>
    #include <type_traits>
#else
    #define DSPARK_WEBVIEW_BACKEND 0
#endif

namespace dspark::plugin::webview_ui {

// -- diagnostics -------------------------------------------------------------------
//
// Set the environment variable DSPARK_WEBVIEW_LOG to any non-empty value and
// every editor in the process appends its host-interaction trace (attach,
// size negotiation, parent resizes) to %TEMP%/DSParkWebView.log (or
// $TMPDIR on POSIX). Costs nothing when the variable is absent.

inline void debugLog(const char* format, ...) noexcept
{
    static std::FILE* file = []() -> std::FILE* {
        char enabled[8] {};
#if defined(_WIN32)
        if (GetEnvironmentVariableA("DSPARK_WEBVIEW_LOG", enabled,
                                    sizeof(enabled)) == 0)
            return nullptr;
        char dir[MAX_PATH] {};
        if (GetEnvironmentVariableA("TEMP", dir, sizeof(dir)) == 0)
            dir[0] = '\0';
        char path[MAX_PATH + 32];
        std::snprintf(path, sizeof(path), "%s%sDSParkWebView.log",
                      dir, dir[0] != '\0' ? "\\" : "");
        std::FILE* f = nullptr;
        fopen_s(&f, path, "a");
#else
        const char* env = std::getenv("DSPARK_WEBVIEW_LOG");
        if (env == nullptr || env[0] == '\0') return nullptr;
        (void) enabled;
        const char* dir = std::getenv("TMPDIR");
        char path[512];
        std::snprintf(path, sizeof(path), "%s/DSParkWebView.log",
                      dir != nullptr ? dir : "/tmp");
        std::FILE* f = std::fopen(path, "a");
#endif
        if (f != nullptr)
            std::fprintf(f, "\n== editor session (layer built " __DATE__ " " __TIME__ ") ==\n");
        return f;
    }();
    if (file == nullptr) return;
    std::va_list args;
    va_start(args, format);
    std::vfprintf(file, format, args);
    va_end(args);
    std::fputc('\n', file);
    std::fflush(file);
}

// -- JSON utilities ---------------------------------------------------------------
//
// The bridge protocol is small enough that a focused, locale-independent
// formatter/parser beats dragging in a JSON library: host processes may run
// with any LC_NUMERIC, where printf writes "6,5" (invalid JSON) and strtod
// stops at the '.' — both classic plugin bugs.

/** @brief Appends a JSON string literal (quotes included, control chars escaped). */
inline void appendJsonString(std::string& out, const char* s)
{
    out += '"';
    for (const char* c = s; c != nullptr && *c != '\0'; ++c)
    {
        const unsigned char u = static_cast<unsigned char>(*c);
        switch (u)
        {
        case '"':  out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n";  break;
        case '\r': out += "\\r";  break;
        case '\t': out += "\\t";  break;
        default:
            if (u < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", u);
                out += buf;
            }
            else
                out += static_cast<char>(u);
            break;
        }
    }
    out += '"';
}

/** @brief Appends a JSON number, immune to the process locale. */
inline void appendJsonNumber(std::string& out, double v)
{
    if (!std::isfinite(v))
    {
        out += '0';
        return;
    }
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    for (char* c = buf; *c != '\0'; ++c)
        if (*c == ',') *c = '.';   // locale decimal comma -> JSON dot
    out += buf;
}

/** @brief Locale-independent number parse; returns the end pointer or nullptr. */
inline const char* parseJsonNumber(const char* s, double& out) noexcept
{
    bool negative = false;
    if (*s == '-' || *s == '+')
    {
        negative = (*s == '-');
        ++s;
    }
    double v = 0.0;
    bool any = false;
    while (*s >= '0' && *s <= '9')
    {
        v = v * 10.0 + (*s - '0');
        ++s;
        any = true;
    }
    if (*s == '.')
    {
        ++s;
        double f = 0.1;
        while (*s >= '0' && *s <= '9')
        {
            v += (*s - '0') * f;
            f *= 0.1;
            ++s;
            any = true;
        }
    }
    if (!any) return nullptr;
    if (*s == 'e' || *s == 'E')
    {
        ++s;
        bool expNegative = false;
        if (*s == '-' || *s == '+')
        {
            expNegative = (*s == '-');
            ++s;
        }
        int e = 0;
        bool eAny = false;
        while (*s >= '0' && *s <= '9')
        {
            e = e * 10 + (*s - '0');
            ++s;
            eAny = true;
            if (e > 308) { e = 309; }   // saturate; isfinite check downstream
        }
        if (!eAny) return nullptr;
        v *= std::pow(10.0, expNegative ? -e : e);
    }
    out = negative ? -v : v;
    return s;
}

/** @brief One decoded bridge message: `[op, id?, value?]`. */
struct PostArgs
{
    char   op[12] {};
    char   id[64] {};
    double value = 0.0;
    int    tokens = 0;
};

/** @brief Parses the bridge's JSON array of scalars (op + optional id/number). */
inline bool parsePostArgs(const char* s, PostArgs& out) noexcept
{
    auto skipWs = [&s] { while (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r') ++s; };
    auto readString = [&s](char* dst, size_t cap) -> bool
    {
        ++s;   // opening quote
        size_t n = 0;
        while (*s != '\0' && *s != '"')
        {
            char c = *s++;
            if (c == '\\')
            {
                const char esc = *s++;
                switch (esc)
                {
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                case '/': c = '/'; break;
                case 'n': c = '\n'; break;
                case 'r': c = '\r'; break;
                case 't': c = '\t'; break;
                case 'b': c = '\b'; break;
                case 'f': c = '\f'; break;
                case 'u':
                {
                    unsigned code = 0;
                    for (int k = 0; k < 4; ++k)
                    {
                        const char h = *s;
                        if (h >= '0' && h <= '9') code = code * 16 + unsigned(h - '0');
                        else if (h >= 'a' && h <= 'f') code = code * 16 + unsigned(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') code = code * 16 + unsigned(h - 'A' + 10);
                        else return false;
                        ++s;
                    }
                    c = code < 128 ? static_cast<char>(code) : '?';
                    break;
                }
                case '\0': return false;
                default: return false;
                }
            }
            if (n + 1 < cap) dst[n++] = c;
        }
        if (*s != '"') return false;
        ++s;
        dst[n] = '\0';
        return true;
    };

    out = PostArgs {};
    skipWs();
    if (*s != '[') return false;
    ++s;
    for (;;)
    {
        skipWs();
        if (*s == ']') { ++s; return true; }
        if (out.tokens > 0)
        {
            if (*s != ',') return false;
            ++s;
            skipWs();
        }
        if (*s == '"')
        {
            char* dst = out.tokens == 0 ? out.op : out.id;
            const size_t cap = out.tokens == 0 ? sizeof(out.op) : sizeof(out.id);
            if (out.tokens >= 2 || !readString(dst, cap)) return false;
        }
        else if (*s == 't' || *s == 'f' || *s == 'n')
        {
            // true/false/null can only appear as the value slot.
            out.value = (*s == 't') ? 1.0 : 0.0;
            while ((*s >= 'a' && *s <= 'z')) ++s;
        }
        else
        {
            const char* end = parseJsonNumber(s, out.value);
            if (end == nullptr) return false;
            s = end;
        }
        ++out.tokens;
        if (out.tokens > 3) return false;
    }
}

// -- the injected JS bridge ---------------------------------------------------------
//
// Runs at document start, before the page's own scripts. `__dsparkPost` is
// the platform-provided uplink (a bound RPC on Windows, a WebKit message
// handler wrapper on macOS); `__dsparkRecv` is the downlink the native side
// drives through script evaluation. Everything user-facing lives in
// `window.dspark`. The ~30 Hz poll starts after the native side answers the
// "ready" handshake, and listeners replay the cached value on registration,
// so page and native initialisation order can never race.

inline constexpr const char* kBridgeJs =
    "(function(){'use strict';"
    "if(window.dspark){return;}"
    "var ls={},vals={},meta=null,readyCbs=[];"
    "function fire(id,v){var c=ls[id];if(c){for(var i=0;i<c.length;i++){c[i](v);}}}"
    "window.__dsparkRecv=function(m){"
      "if(m.type==='params'){"
        "meta=m.params;"
        "for(var i=0;i<meta.length;i++){vals[meta[i].id]=meta[i].value;}"
        // KeepAspect editors scale their whole content with the window:
        // the page keeps its DESIGN layout and is fitted into the window
        // by the limiting axis, centered (letterboxed) when the host lets
        // the user distort the ratio. Nothing can ever clip.
        "if(m.design&&m.design.keepAspect&&!window.__dsparkFit){"
          "var dw=m.design.width,dh=m.design.height;"
          "window.__dsparkFit=function(){"
            "var z=Math.min(window.innerWidth/dw,window.innerHeight/dh);"
            "if(!(z>0&&isFinite(z))){z=1;}"
            "var s=document.body.style;"
            "s.width=dw+'px';s.height=dh+'px';"
            "s.position='absolute';s.margin='0';"
            "s.transformOrigin='0 0';s.transform='scale('+z+')';"
            "s.left=((window.innerWidth-dw*z)/2)+'px';"
            "s.top=((window.innerHeight-dh*z)/2)+'px';};"
          "window.addEventListener('resize',window.__dsparkFit);"
          "window.__dsparkFit();}"
        "var rc=readyCbs;readyCbs=[];"
        "for(var j=0;j<rc.length;j++){rc[j](meta);}"
        "for(var k=0;k<meta.length;k++){fire(meta[k].id,meta[k].value);}"
        "if(!window.__dsparkTimer){"
          "window.__dsparkTimer=setInterval(function(){window.__dsparkPost('poll');},33);}"
      "}else if(m.type==='values'){"
        "for(var id in m.values){"
          "if(vals[id]!==m.values[id]){vals[id]=m.values[id];fire(id,m.values[id]);}}"
      "}};"
    "window.dspark={"
      "setParam:function(id,v){v=+v;vals[id]=v;window.__dsparkPost('set',id,v);fire(id,v);},"
      "getParam:function(id){return vals[id];},"
      "beginEdit:function(id){window.__dsparkPost('begin',id);},"
      "endEdit:function(id){window.__dsparkPost('end',id);},"
      "onParam:function(id,cb){(ls[id]=ls[id]||[]).push(cb);"
        "if(meta!==null&&(id in vals)){cb(vals[id]);}},"
      "onReady:function(cb){if(meta!==null){cb(meta);}else{readyCbs.push(cb);}}"
    "};"
    "Object.defineProperty(window.dspark,'params',{get:function(){return meta;}});"
    "function hello(){window.__dsparkPost('ready');}"
    "if(document.readyState==='complete'||document.readyState==='interactive'){hello();}"
    "else{document.addEventListener('DOMContentLoaded',hello);}"
    "})();";

// -- host callbacks ----------------------------------------------------------------

/**
 * @brief How the editor reaches back into the format wrapper. Plain function
 * pointers + context (no std::function, no virtuals): the wrapper translates
 * to its format's gesture/automation calls and to the user's setParameter.
 * All calls arrive on the host's main/UI thread, in PLAIN parameter values.
 */
struct HostCallbacks
{
    void* context = nullptr;
    void (*setParam)(void* context, int index, double plainValue) = nullptr;
    void (*beginEdit)(void* context, int index) = nullptr;
    void (*endEdit)(void* context, int index) = nullptr;
};

#if DSPARK_WEBVIEW_BACKEND == 2

// -- macOS: minimal Objective-C runtime glue ---------------------------------------
//
// WKWebView is driven entirely through objc_msgSend, so no Apple GUI headers
// or link flags are needed; WebKit.framework is loaded on first use. The
// script-message relay is ONE runtime-registered class shared by every
// editor in the process: its ivar holds a {function, context} sink, keeping
// the IMP free of template types (two different plugins in one host would
// otherwise race to register the same class name with different code).

namespace objc_glue {

using ObjId = void*;

template <typename Ret = ObjId, typename... Args>
inline Ret call(ObjId target, const char* selector, Args... args) noexcept
{
    using Fn = Ret (*)(ObjId, SEL, Args...);
    return reinterpret_cast<Fn>(&objc_msgSend)(target, sel_registerName(selector), args...);
}

inline ObjId cls(const char* name) noexcept
{
    return reinterpret_cast<ObjId>(objc_getClass(name));
}

inline ObjId nsString(const char* utf8) noexcept
{
    return call(cls("NSString"), "stringWithUTF8String:", utf8 ? utf8 : "");
}

struct Rect { double x = 0, y = 0, w = 0, h = 0; };   // CGRect-compatible

struct MessageSink
{
    void (*fn)(void* context, const char* json) = nullptr;
    void* context = nullptr;
};

inline Class relayClass() noexcept;

inline void onScriptMessage(ObjId self, SEL, ObjId, ObjId message) noexcept
{
    Ivar ivar = class_getInstanceVariable(relayClass(), "dsparkSink");
    if (ivar == nullptr) return;
    auto* sink = reinterpret_cast<MessageSink*>(
        object_getIvar(static_cast<id>(self), ivar));
    if (sink == nullptr || sink->fn == nullptr) return;
    ObjId body = call(message, "body");
    if (body == nullptr) return;
    const char* utf8 = call<const char*>(body, "UTF8String");
    if (utf8 != nullptr)
        sink->fn(sink->context, utf8);
}

inline Class relayClass() noexcept
{
    static Class registered = [] {
        Class c = objc_allocateClassPair(objc_getClass("NSObject"),
                                         "DSParkWebViewMessageRelay", 0);
        if (c == nullptr)   // another DSPark plugin in this process won the race
            return objc_getClass("DSParkWebViewMessageRelay");
        class_addIvar(c, "dsparkSink", sizeof(void*), alignof(void*) == 8 ? 3 : 2, "^v");
        class_addMethod(c, sel_registerName("userContentController:didReceiveScriptMessage:"),
                        reinterpret_cast<IMP>(&onScriptMessage), "v@:@@");
        objc_registerClassPair(c);
        return c;
    }();
    return registered;
}

inline void setRelaySink(ObjId relay, MessageSink* sink) noexcept
{
    Ivar ivar = class_getInstanceVariable(relayClass(), "dsparkSink");
    if (ivar != nullptr)
        object_setIvar(static_cast<id>(relay), ivar, reinterpret_cast<id>(sink));
}

inline bool loadWebKit() noexcept
{
    static void* handle =
        dlopen("/System/Library/Frameworks/WebKit.framework/WebKit", RTLD_LAZY | RTLD_GLOBAL);
    return handle != nullptr;
}

} // namespace objc_glue

#endif // DSPARK_WEBVIEW_BACKEND == 2

#if DSPARK_WEBVIEW_BACKEND == 3

// -- Linux: WebKitGTK resolved at runtime -------------------------------------------
//
// Every entry point comes from ONE dlopen of libwebkit2gtk (GTK3, GLib and
// JavaScriptCore are its dependencies, so dlsym on that handle resolves them
// all): no GTK headers, no pkg-config, no link-time dependency — plugins
// keep building with the plain C++ toolchain and simply report the editor
// unavailable where WebKitGTK is missing. GTK3 is required for GtkPlug
// (XEmbed into the host's X11 window); GTK4 removed it, which is why the
// 4.x webkit2gtk line is used and not webkitgtk-6.0.

namespace gtk_glue {

/** @brief The WebKitGTK/GTK3 surface the editor uses, all dlsym-resolved.
 *  `ok` is true only when every required symbol was found. */
struct Api
{
    bool ok = false;
    // GTK3 / GLib / GObject
    int   (*gtkInitCheck)(int*, char***) = nullptr;
    void* (*plugNew)(unsigned long) = nullptr;
    void  (*containerAdd)(void*, void*) = nullptr;
    void  (*widgetShowAll)(void*) = nullptr;
    void  (*widgetSetVisible)(void*, int) = nullptr;
    void  (*widgetDestroy)(void*) = nullptr;
    void  (*windowSetDefaultSize)(void*, int, int) = nullptr;
    void  (*windowResize)(void*, int, int) = nullptr;
    unsigned long (*signalConnectData)(void*, const char*, void (*)(), void*,
                                       void*, int) = nullptr;
    void  (*signalHandlerDisconnect)(void*, unsigned long) = nullptr;
    int   (*mainContextPending)(void*) = nullptr;
    int   (*mainContextIteration)(void*, int) = nullptr;
    void  (*gFree)(void*) = nullptr;
    // WebKit
    void* (*webViewNew)() = nullptr;
    void* (*viewGetContentManager)(void*) = nullptr;
    int   (*contentManagerRegisterHandler)(void*, const char*) = nullptr;
    void  (*contentManagerUnregisterHandler)(void*, const char*) = nullptr;
    void* (*userScriptNew)(const char*, int, int, const char* const*,
                           const char* const*) = nullptr;
    void  (*contentManagerAddScript)(void*, void*) = nullptr;
    void  (*userScriptUnref)(void*) = nullptr;
    void  (*loadHtml)(void*, const char*, const char*) = nullptr;
    // JS eval: run_javascript exists on every 4.0/4.1 (deprecated since
    // 2.40); evaluate_javascript replaces it from 2.40 on. Either works.
    void  (*runJs)(void*, const char*, void*, void*, void*) = nullptr;
    void  (*evalJs)(void*, const char*, long, const char*, const char*,
                    void*, void*, void*) = nullptr;
    void* (*jsResultGetValue)(void*) = nullptr;
    char* (*jscValueToString)(void*) = nullptr;
};

inline Api loadApi() noexcept
{
    Api api {};
    void* lib = dlopen("libwebkit2gtk-4.1.so.0", RTLD_LAZY | RTLD_LOCAL);
    if (lib == nullptr)
        lib = dlopen("libwebkit2gtk-4.0.so.37", RTLD_LAZY | RTLD_LOCAL);
    if (lib == nullptr)
    {
        debugLog("linux engine: libwebkit2gtk not found (generic UI)");
        return api;
    }
    auto load = [lib](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(lib, name));
    };
    load(api.gtkInitCheck, "gtk_init_check");
    load(api.plugNew, "gtk_plug_new");
    load(api.containerAdd, "gtk_container_add");
    load(api.widgetShowAll, "gtk_widget_show_all");
    load(api.widgetSetVisible, "gtk_widget_set_visible");
    load(api.widgetDestroy, "gtk_widget_destroy");
    load(api.windowSetDefaultSize, "gtk_window_set_default_size");
    load(api.windowResize, "gtk_window_resize");
    load(api.signalConnectData, "g_signal_connect_data");
    load(api.signalHandlerDisconnect, "g_signal_handler_disconnect");
    load(api.mainContextPending, "g_main_context_pending");
    load(api.mainContextIteration, "g_main_context_iteration");
    load(api.gFree, "g_free");
    load(api.webViewNew, "webkit_web_view_new");
    load(api.viewGetContentManager, "webkit_web_view_get_user_content_manager");
    load(api.contentManagerRegisterHandler,
         "webkit_user_content_manager_register_script_message_handler");
    load(api.contentManagerUnregisterHandler,
         "webkit_user_content_manager_unregister_script_message_handler");
    load(api.userScriptNew, "webkit_user_script_new");
    load(api.contentManagerAddScript, "webkit_user_content_manager_add_script");
    load(api.userScriptUnref, "webkit_user_script_unref");
    load(api.loadHtml, "webkit_web_view_load_html");
    load(api.runJs, "webkit_web_view_run_javascript");
    load(api.evalJs, "webkit_web_view_evaluate_javascript");
    load(api.jsResultGetValue, "webkit_javascript_result_get_js_value");
    load(api.jscValueToString, "jsc_value_to_string");
    api.ok = api.gtkInitCheck != nullptr && api.plugNew != nullptr
          && api.containerAdd != nullptr && api.widgetShowAll != nullptr
          && api.widgetSetVisible != nullptr && api.widgetDestroy != nullptr
          && api.windowSetDefaultSize != nullptr && api.windowResize != nullptr
          && api.signalConnectData != nullptr
          && api.signalHandlerDisconnect != nullptr
          && api.mainContextPending != nullptr
          && api.mainContextIteration != nullptr && api.gFree != nullptr
          && api.webViewNew != nullptr && api.viewGetContentManager != nullptr
          && api.contentManagerRegisterHandler != nullptr
          && api.contentManagerUnregisterHandler != nullptr
          && api.userScriptNew != nullptr
          && api.contentManagerAddScript != nullptr
          && api.userScriptUnref != nullptr && api.loadHtml != nullptr
          && (api.runJs != nullptr || api.evalJs != nullptr)
          && api.jsResultGetValue != nullptr && api.jscValueToString != nullptr;
    if (!api.ok)
        debugLog("linux engine: libwebkit2gtk found but symbols missing");
    return api;
}

/** @brief Loaded once per process, first use; cheap to call afterwards. */
inline const Api& api() noexcept
{
    static const Api loaded = loadApi();
    return loaded;
}

} // namespace gtk_glue

#endif // DSPARK_WEBVIEW_BACKEND == 3

// -- the editor --------------------------------------------------------------------

/**
 * @brief One embedded WebView editor instance for plugin class @p P. Owned
 * by the format backends (the VST3 IPlugView object, the CLAP gui
 * extension); plugin authors never touch this class — they only provide
 * `editorHtml()` and friends.
 *
 * Lifecycle (all on the host's main thread): `create(parent, ...)` builds
 * the platform web engine inside the host window and loads the page;
 * `setBounds` follows host resizes; `destroy` tears everything down. While
 * alive, the page polls the wrapper's normalized shadows for DSP -> UI sync
 * and pushes edits through HostCallbacks for UI -> DSP.
 */
template <typename P>
class Editor
{
public:
    static_assert(!HasEditor<P> || HasEditorHtml<P>,
                  "hasEditor is true but `static const char* editorHtml()` is missing");

    /** True when this platform can embed a WebView editor (see file docs). */
    static constexpr bool kAvailable = (DSPARK_WEBVIEW_BACKEND != 0);

    /**
     * Runtime availability: kAvailable AND the platform engine can actually
     * run here. On Linux that means WebKitGTK/GTK3 resolved through dlopen
     * (first call loads it, later calls are free); elsewhere it equals
     * kAvailable. The format backends gate their editor offer on this, so
     * hosts on bare systems fall back to their generic UI cleanly.
     */
    static bool available() noexcept
    {
#if DSPARK_WEBVIEW_BACKEND == 3
        return gtk_glue::api().ok;
#else
        return kAvailable;
#endif
    }

    /**
     * Gives the engine main-loop time from the HOST's run loop. On Linux the
     * format backends call this at ~30 Hz (VST3 IRunLoop timer / CLAP
     * timer-support) to iterate the GLib main context — a plugin must never
     * spin its own GTK loop inside a host. No-op on Windows/macOS, where the
     * host's native message loop already drives the engine.
     */
    void pump() noexcept
    {
#if DSPARK_WEBVIEW_BACKEND == 3
        if (created_)
            pumpPlatform();
#endif
    }

    Editor() = default;
    ~Editor() { destroy(); }

    Editor(const Editor&) = delete;
    Editor& operator=(const Editor&) = delete;

    /**
     * Builds the web engine inside @p parentWindow (HWND on Windows, NSView*
     * on macOS), loads the user page and starts the bridge. @p shadows is
     * the wrapper's normalized shadow array (read-only here); @p host
     * receives UI edits. Returns false when the engine is unavailable (the
     * host then falls back to its generic editor).
     */
    bool create(void* parentWindow, const std::atomic<double>* shadows,
                const HostCallbacks& host) noexcept
    {
        if (created_ || parentWindow == nullptr) return false;
        shadows_ = shadows;
        host_ = host;
        parent_ = parentWindow;
        created_ = createPlatform(parentWindow);
        return created_;
    }

    /** Tears the web engine down. Safe to call twice; never touches the host window. */
    void destroy() noexcept
    {
        if (!created_) return;
        created_ = false;
        destroyPlatform();
    }

    /** Resizes the embedded view to the host-given box (origin stays 0,0). */
    void setBounds(int width, int height) noexcept
    {
        if (created_ && width > 0 && height > 0)
            setBoundsPlatform(width, height);
    }

    /** Shows/hides the embedded view (CLAP hosts drive this explicitly). */
    void setVisible(bool visible) noexcept
    {
        if (created_)
            setVisiblePlatform(visible);
    }

    /**
     * Reads the CURRENT size of the host window we are embedded in
     * (physical pixels on Windows). Lets the backends fit the page to
     * whatever box the host actually created, even when its negotiation
     * calls arrived in an unexpected order. Returns false where the parent
     * cannot be queried (then the negotiated size is used as-is).
     *
     * Note on DPI: the web engine applies the window's scale factor itself
     * (CSS pixels are device-independent), so the page never needs zooming —
     * the wrapper only converts logical <-> physical for host negotiation.
     */
    bool queryParentSize(int& width, int& height) const noexcept
    {
        return queryParentSizePlatform(width, height);
    }

    [[nodiscard]] bool created() const noexcept { return created_; }

private:
    static constexpr size_t kNumParams = P::parameters.size();

    std::atomic<double> const* shadows_ = nullptr;
    HostCallbacks host_ {};
    void* parent_ = nullptr;
    bool  created_ = false;

    static int indexOfId(const char* id) noexcept
    {
        for (size_t i = 0; i < kNumParams; ++i)
            if (std::strcmp(P::parameters[i].id, id) == 0) return static_cast<int>(i);
        return -1;
    }

    [[nodiscard]] double plainOf(size_t index) const noexcept
    {
        return toPlain(P::parameters[index],
                       shadows_[index].load(std::memory_order_relaxed));
    }

    // --- bridge protocol ---------------------------------------------------------

    void handlePost(const char* json) noexcept
    {
        PostArgs msg;
        if (!parsePostArgs(json, msg)) return;
        if (std::strcmp(msg.op, "poll") == 0)
        {
            sendValues();
        }
        else if (std::strcmp(msg.op, "set") == 0)
        {
            const int idx = indexOfId(msg.id);
            if (idx >= 0 && msg.tokens >= 3 && host_.setParam != nullptr)
                host_.setParam(host_.context, idx, msg.value);
        }
        else if (std::strcmp(msg.op, "begin") == 0)
        {
            const int idx = indexOfId(msg.id);
            if (idx >= 0 && host_.beginEdit != nullptr)
                host_.beginEdit(host_.context, idx);
        }
        else if (std::strcmp(msg.op, "end") == 0)
        {
            const int idx = indexOfId(msg.id);
            if (idx >= 0 && host_.endEdit != nullptr)
                host_.endEdit(host_.context, idx);
        }
        else if (std::strcmp(msg.op, "ready") == 0)
        {
            sendMeta();
            // Diagnostic: how the page actually sees the world (DPI bugs
            // live exactly here). Logged only with DSPARK_WEBVIEW_LOG set.
            evalPlatform("window.__dsparkPost('metric','innerWidth',window.innerWidth);"
                         "window.__dsparkPost('metric','innerHeight',window.innerHeight);"
                         "window.__dsparkPost('metric','dpr',window.devicePixelRatio||1);");
        }
        else if (std::strcmp(msg.op, "metric") == 0)
        {
            debugLog("page metric %s = %.2f", msg.id, msg.value);
        }
    }

    void sendMeta()
    {
        std::string json = "{\"type\":\"params\",\"params\":[";
        for (size_t i = 0; i < kNumParams; ++i)
        {
            const Param& p = P::parameters[i];
            if (i > 0) json += ',';
            json += "{\"id\":";
            appendJsonString(json, p.id);
            json += ",\"name\":";
            appendJsonString(json, p.name);
            json += ",\"min\":";
            appendJsonNumber(json, p.minValue);
            json += ",\"max\":";
            appendJsonNumber(json, p.maxValue);
            json += ",\"def\":";
            appendJsonNumber(json, p.defValue);
            json += ",\"unit\":";
            appendJsonString(json, p.unit);
            json += ",\"steps\":";
            appendJsonNumber(json, p.steps);
            json += ",\"value\":";
            appendJsonNumber(json, plainOf(i));
            json += '}';
        }
        json += "],\"design\":{\"width\":";
        appendJsonNumber(json, editorSizeOf<P>().width);
        json += ",\"height\":";
        appendJsonNumber(json, editorSizeOf<P>().height);
        json += ",\"keepAspect\":";
        json += editorResizeOf<P>() == EditorResize::KeepAspect ? "true" : "false";
        json += "}}";
        sendRecv(json);
    }

    void sendValues()
    {
        std::string json = "{\"type\":\"values\",\"values\":{";
        for (size_t i = 0; i < kNumParams; ++i)
        {
            if (i > 0) json += ',';
            appendJsonString(json, P::parameters[i].id);
            json += ':';
            appendJsonNumber(json, plainOf(i));
        }
        json += "}}";
        sendRecv(json);
    }

    void sendRecv(const std::string& payloadJson)
    {
        std::string js = "window.__dsparkRecv(";
        js += payloadJson;
        js += ");";
        evalPlatform(js);
    }

    static constexpr bool debugFlag() noexcept
    {
        if constexpr (HasEditorDebug<P>) return true;
        else return false;
    }

    /** The page to serve: the dev-file override when declared and readable
     *  (edit + reopen, no recompile), the embedded editorHtml() otherwise. */
    static std::string pageHtml()
    {
#if !defined(DSPARK_NO_FILE_IO)
        if constexpr (HasEditorDevFile<P>)
        {
            std::FILE* f = nullptr;
#if defined(_WIN32)
            fopen_s(&f, P::editorDevFile(), "rb");
#else
            f = std::fopen(P::editorDevFile(), "rb");
#endif
            if (f != nullptr)
            {
                std::string text;
                char chunk[4096];
                size_t got = 0;
                while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0)
                    text.append(chunk, got);
                std::fclose(f);
                if (!text.empty())
                {
                    debugLog("page: dev file %s (%u bytes)", P::editorDevFile(),
                             static_cast<unsigned>(text.size()));
                    return text;
                }
            }
            debugLog("page: dev file %s unreadable; embedded page used",
                     P::editorDevFile());
        }
#endif
        return P::editorHtml();
    }

    // ==============================================================================
    // Windows — WebView2 through the vendored webview library
    // ==============================================================================
#if DSPARK_WEBVIEW_BACKEND == 1

    std::unique_ptr<webview::webview> wv_;
    void* root_ = nullptr;               // host top-level window (frame limits)
    mutable SIZE frameChrome_ { -1, -1 }; // historical-minimum chrome; -1 = unmeasured
    bool comInitialized_ = false;

    bool createPlatform(void* parentWindow) noexcept
    {
        try
        {
            // WebView2 needs COM STA on this (UI) thread. DAWs have it there
            // already (S_FALSE, harmless); bare hosts may not. Balanced in
            // destroyPlatform. RPC_E_CHANGED_MODE (an MTA thread) fails the
            // creation below and degrades to the host's generic UI.
            comInitialized_ = SUCCEEDED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED));
            wv_ = std::make_unique<webview::webview>(debugFlag(), parentWindow);
            wv_->bind("__dsparkPost", [this](std::string request) -> std::string {
                handlePost(request.c_str());
                return std::string {};
            });
            wv_->init(kBridgeJs);
            wv_->set_html(pageHtml());
            // Track the host window directly: several hosts resize their
            // plugin area without calling the format's size callbacks, so
            // relying on those alone leaves the page stuck at the old size.
            SetWindowSubclass(static_cast<HWND>(parentWindow), &onParentMessage,
                              reinterpret_cast<UINT_PTR>(this),
                              reinterpret_cast<DWORD_PTR>(this));
            // And the host's TOP-LEVEL frame: WM_GETMINMAXINFO/WM_SIZING are
            // the OS-level drag limits — the only ones no host can bypass.
            // VST3 size negotiation only governs the inner plugin area; the
            // outer frame belongs to the user, so min/max/aspect must be
            // enforced where the WINDOW is dragged (see rootGuard for docks).
            root_ = GetAncestor(static_cast<HWND>(parentWindow), GA_ROOT);
            if (root_ != nullptr)
                SetWindowSubclass(static_cast<HWND>(root_), &onRootMessage,
                                  reinterpret_cast<UINT_PTR>(this),
                                  reinterpret_cast<DWORD_PTR>(this));
            RECT parentBox {};
            GetClientRect(static_cast<HWND>(parentWindow), &parentBox);
            debugLog("create: parent=%p root=%p client=%ldx%ld", parentWindow,
                     root_, parentBox.right, parentBox.bottom);
            return true;
        }
        catch (...)   // e.g. WebView2 runtime missing -> host falls back to generic UI
        {
            wv_.reset();
            if (comInitialized_)
            {
                CoUninitialize();
                comInitialized_ = false;
            }
            return false;
        }
    }

    static LRESULT CALLBACK onParentMessage(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp,
                                            UINT_PTR, DWORD_PTR refData) noexcept
    {
        if (msg == WM_SIZE)
        {
            debugLog("parent WM_SIZE %dx%d", int(LOWORD(lp)), int(HIWORD(lp)));
            auto* editor = reinterpret_cast<Editor*>(refData);
            editor->setBounds(static_cast<int>(LOWORD(lp)), static_cast<int>(HIWORD(lp)));
        }
        return DefSubclassProc(hwnd, msg, wp, lp);
    }

    // --- host frame limits (OS level) ----------------------------------------------

    /** DPI of the plugin area, as a 96-dpi scale factor (multi-monitor safe). */
    double frameDpiScale() const noexcept
    {
        const UINT dpi = parent_ != nullptr ? GetDpiForWindow(static_cast<HWND>(parent_))
                                            : 96u;
        return dpi > 0 ? dpi / 96.0 : 1.0;
    }

    /**
     * Only steer the top-level frame when the plugin area dominates it (a
     * floating plugin window). When the editor is docked inside a large host
     * window, that frame is the DAW's own — hands off.
     *
     * The chrome (frame + host bars around the plugin) is tracked as the
     * HISTORICAL MINIMUM of coherent measurements. That cuts the feedback
     * loop by construction — dead space the host leaves around the plugin
     * can only inflate a measurement, never deflate it, so the minimum
     * converges on the real chrome — while staying re-evaluable (a single
     * early measurement taken mid-layout cannot poison the limits forever;
     * that permanent-verdict cache was the previous round's bug).
     */
    bool rootGuard(HWND root, SIZE& chrome) const noexcept
    {
        if (parent_ == nullptr || root == nullptr) return false;
        RECT rootBox {}, parentBox {};
        const bool measured = GetWindowRect(root, &rootBox)
                           && GetClientRect(static_cast<HWND>(parent_), &parentBox);
        bool coherent = false;
        if (measured)
        {
            // Threshold separates plugin windows from docked editors: a
            // REAPER FX-chain window (plugin list beside the editor) sits
            // near 45-90% plugin area, an editor docked in a DAW main
            // window stays well under 25%.
            const double rootArea = double(rootBox.right - rootBox.left)
                                  * double(rootBox.bottom - rootBox.top);
            const double parentArea = double(parentBox.right) * double(parentBox.bottom);
            coherent = rootArea > 0.0 && parentArea > 0.0
                    && parentArea / rootArea >= 1.0 / 3.0;
        }
        if (coherent)
        {
            // No tight cap here: legitimate host chrome can be large (the
            // REAPER FX-chain window keeps its plugin list beside the
            // editor, ~450px). Underestimating it lets the window shrink
            // past the point where the plugin minimum still fits — exactly
            // a clipped editor. The historical minimum below already stops
            // dead space from inflating these numbers over time.
            auto clampChrome = [](LONG v) -> LONG {
                return v < 0 ? 0 : (v > 1024 ? 1024 : v);
            };
            const LONG cx = clampChrome((rootBox.right - rootBox.left) - parentBox.right);
            const LONG cy = clampChrome((rootBox.bottom - rootBox.top) - parentBox.bottom);
            if (frameChrome_.cx < 0)
            {
                frameChrome_ = SIZE { cx, cy };
                debugLog("frame: steering host window, chrome=%ldx%ld", cx, cy);
            }
            else
            {
                if (cx < frameChrome_.cx) frameChrome_.cx = cx;
                if (cy < frameChrome_.cy) frameChrome_.cy = cy;
            }
        }
        // A truly docked editor never produces a coherent (dominant-area)
        // measurement, so the chrome stays unmeasured and we never steer.
        if (frameChrome_.cx < 0) return false;
        chrome = frameChrome_;
        return true;
    }

    void applyMinMax(HWND root, MINMAXINFO* info) const noexcept
    {
        SIZE chrome {};
        if (info == nullptr || !rootGuard(root, chrome)) return;
        const EditorSize logical = editorSizeOf<P>();
        const double dpi = frameDpiScale();
        constexpr EditorResize mode = editorResizeOf<P>();
        const double lo = mode == EditorResize::Fixed ? 1.0 : kEditorMinSizeFactor;
        const double hi = mode == EditorResize::Fixed ? 1.0 : kEditorMaxSizeFactor;
        const LONG minW = LONG(logical.width * dpi * lo + 0.5) + chrome.cx;
        const LONG minH = LONG(logical.height * dpi * lo + 0.5) + chrome.cy;
        const LONG maxW = LONG(logical.width * dpi * hi + 0.5) + chrome.cx;
        const LONG maxH = LONG(logical.height * dpi * hi + 0.5) + chrome.cy;
        if (info->ptMinTrackSize.x < minW) info->ptMinTrackSize.x = minW;
        if (info->ptMinTrackSize.y < minH) info->ptMinTrackSize.y = minH;
        if (info->ptMaxTrackSize.x > maxW) info->ptMaxTrackSize.x = maxW;
        if (info->ptMaxTrackSize.y > maxH) info->ptMaxTrackSize.y = maxH;
        debugLog("frame minmax: track %ldx%ld .. %ldx%ld (chrome %ldx%ld)",
                 info->ptMinTrackSize.x, info->ptMinTrackSize.y,
                 info->ptMaxTrackSize.x, info->ptMaxTrackSize.y,
                 chrome.cx, chrome.cy);
    }

    void applySizing(HWND root, int edge, RECT* box) const noexcept
    {
        SIZE chrome {};
        if (box == nullptr || !rootGuard(root, chrome)) return;
        const EditorSize logical = editorSizeOf<P>();
        const double dpi = frameDpiScale();
        constexpr EditorResize mode = editorResizeOf<P>();
        // Unlike checkSizeConstraint (where the window is already fixed and
        // the plugin must fit INSIDE it), WM_SIZING reshapes the whole
        // window — so the dragged axis leads and the other may follow.
        const double lo = mode == EditorResize::Fixed ? 1.0 : kEditorMinSizeFactor;
        const double hi = mode == EditorResize::Fixed ? 1.0 : kEditorMaxSizeFactor;
        const double minW = logical.width * dpi * lo;
        const double maxW = logical.width * dpi * hi;
        const double minH = logical.height * dpi * lo;
        const double maxH = logical.height * dpi * hi;
        double w = double(box->right - box->left) - chrome.cx;
        double h = double(box->bottom - box->top) - chrome.cy;
        w = w < minW ? minW : (w > maxW ? maxW : w);
        h = h < minH ? minH : (h > maxH ? maxH : h);
        if constexpr (mode == EditorResize::KeepAspect)
        {
            // The dragged edge expresses the user's intent; the bounds above
            // are ratio-consistent, so the derived axis stays in range.
            const double ratio = double(logical.width) / logical.height;
            if (edge == WMSZ_TOP || edge == WMSZ_BOTTOM)
                w = h * ratio;
            else
                h = w / ratio;
        }
        const LONG newW = LONG(w + 0.5) + chrome.cx;
        const LONG newH = LONG(h + 0.5) + chrome.cy;
        const LONG oldW = box->right - box->left;
        const LONG oldH = box->bottom - box->top;
        // Anchor the side opposite to the dragged edge.
        if (edge == WMSZ_LEFT || edge == WMSZ_TOPLEFT || edge == WMSZ_BOTTOMLEFT)
            box->left = box->right - newW;
        else
            box->right = box->left + newW;
        if (edge == WMSZ_TOP || edge == WMSZ_TOPLEFT || edge == WMSZ_TOPRIGHT)
            box->top = box->bottom - newH;
        else
            box->bottom = box->top + newH;
        if (newW != oldW || newH != oldH)
            debugLog("frame sizing(edge %d): %ldx%ld -> %ldx%ld", edge,
                     oldW, oldH, newW, newH);
    }

    static LRESULT CALLBACK onRootMessage(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp,
                                          UINT_PTR, DWORD_PTR refData) noexcept
    {
        const LRESULT result = DefSubclassProc(hwnd, msg, wp, lp);
        auto* editor = reinterpret_cast<Editor*>(refData);
        if (msg == WM_GETMINMAXINFO)
            editor->applyMinMax(hwnd, reinterpret_cast<MINMAXINFO*>(lp));
        else if (msg == WM_SIZING)
        {
            editor->applySizing(hwnd, static_cast<int>(wp),
                                reinterpret_cast<RECT*>(lp));
            return TRUE;
        }
        return result;
    }

    void destroyPlatform() noexcept
    {
        if (parent_ != nullptr)
            RemoveWindowSubclass(static_cast<HWND>(parent_), &onParentMessage,
                                 reinterpret_cast<UINT_PTR>(this));
        if (root_ != nullptr)
        {
            RemoveWindowSubclass(static_cast<HWND>(root_), &onRootMessage,
                                 reinterpret_cast<UINT_PTR>(this));
            root_ = nullptr;
        }
        try { wv_.reset(); }
        catch (...) {}
        if (comInitialized_)
        {
            CoUninitialize();
            comInitialized_ = false;
        }
    }

    void setBoundsPlatform(int width, int height) noexcept
    {
        // The parent's real client box is the single source of truth: hosts
        // and formats disagree about who announces sizes when, but the web
        // widget must simply fill whatever the host window currently is.
        const int requestedW = width;
        const int requestedH = height;
        RECT rect {};
        if (parent_ != nullptr && GetClientRect(static_cast<HWND>(parent_), &rect)
            && rect.right > 0 && rect.bottom > 0)
        {
            width  = static_cast<int>(rect.right);
            height = static_cast<int>(rect.bottom);
        }
        debugLog("setBounds(%d,%d) -> widget %dx%d", requestedW, requestedH,
                 width, height);
        try
        {
            if (auto widget = wv_->widget(); widget.ok())
                MoveWindow(static_cast<HWND>(widget.value()), 0, 0, width, height, TRUE);
        }
        catch (...) {}
    }

    void setVisiblePlatform(bool visible) noexcept
    {
        try
        {
            if (auto widget = wv_->widget(); widget.ok())
                ShowWindow(static_cast<HWND>(widget.value()), visible ? SW_SHOW : SW_HIDE);
        }
        catch (...) {}
    }

    bool queryParentSizePlatform(int& width, int& height) const noexcept
    {
        RECT rect {};
        if (parent_ == nullptr || !GetClientRect(static_cast<HWND>(parent_), &rect))
            return false;
        if (rect.right <= 0 || rect.bottom <= 0) return false;
        width  = static_cast<int>(rect.right);
        height = static_cast<int>(rect.bottom);
        return true;
    }

    void evalPlatform(const std::string& js) noexcept
    {
        try
        {
            if (wv_) wv_->eval(js);
        }
        catch (...) {}
    }

    // ==============================================================================
    // macOS — WKWebView through the Objective-C runtime
    // ==============================================================================
#elif DSPARK_WEBVIEW_BACKEND == 2

    objc_glue::ObjId webView_ = nullptr;       // WKWebView (retained)
    objc_glue::ObjId configuration_ = nullptr; // WKWebViewConfiguration (retained)
    objc_glue::ObjId relay_ = nullptr;         // script-message relay (retained)
    objc_glue::MessageSink sink_ {};

    static void sinkTrampoline(void* context, const char* json) noexcept
    {
        static_cast<Editor*>(context)->handlePost(json);
    }

    bool createPlatform(void* parentWindow) noexcept
    {
        namespace og = objc_glue;
        if (!og::loadWebKit()) return false;
        Class relayCls = og::relayClass();
        if (relayCls == nullptr) return false;

        sink_ = { &sinkTrampoline, this };
        relay_ = og::call(og::call(reinterpret_cast<og::ObjId>(relayCls), "alloc"), "init");
        if (relay_ == nullptr) return false;
        og::setRelaySink(relay_, &sink_);

        configuration_ = og::call(og::cls("WKWebViewConfiguration"), "new");
        og::ObjId contentController = og::call(configuration_, "userContentController");
        og::call<void>(contentController, "addScriptMessageHandler:name:",
                       relay_, og::nsString("dspark"));

        // The uplink + bridge run at document start, before the page scripts.
        std::string bootstrap =
            "window.__dsparkPost=function(){"
            "window.webkit.messageHandlers.dspark.postMessage("
            "JSON.stringify(Array.prototype.slice.call(arguments)));};";
        bootstrap += kBridgeJs;
        og::ObjId script = og::call(og::call(og::cls("WKUserScript"), "alloc"),
                                    "initWithSource:injectionTime:forMainFrameOnly:",
                                    og::nsString(bootstrap.c_str()),
                                    static_cast<long>(0) /* AtDocumentStart */,
                                    static_cast<signed char>(1));
        og::call<void>(contentController, "addUserScript:", script);
        og::call<void>(script, "release");

        const og::Rect frame {};   // resized right after by the backend
        webView_ = og::call(og::call(og::cls("WKWebView"), "alloc"),
                            "initWithFrame:configuration:", frame, configuration_);
        if (webView_ == nullptr)
        {
            destroyPlatform();
            return false;
        }
        // Follow the host view on resize: width + height autoresize (2 | 16).
        og::call<void>(webView_, "setAutoresizingMask:", static_cast<unsigned long>(18));
        og::call<void>(static_cast<og::ObjId>(parentWindow), "addSubview:", webView_);
        og::call<void>(webView_, "loadHTMLString:baseURL:",
                       og::nsString(pageHtml().c_str()), static_cast<og::ObjId>(nullptr));
        return true;
    }

    void destroyPlatform() noexcept
    {
        namespace og = objc_glue;
        if (configuration_ != nullptr)
        {
            og::ObjId contentController = og::call(configuration_, "userContentController");
            og::call<void>(contentController, "removeScriptMessageHandlerForName:",
                           og::nsString("dspark"));
        }
        if (webView_ != nullptr)
        {
            og::call<void>(webView_, "removeFromSuperview");
            og::call<void>(webView_, "release");
            webView_ = nullptr;
        }
        if (configuration_ != nullptr)
        {
            og::call<void>(configuration_, "release");
            configuration_ = nullptr;
        }
        if (relay_ != nullptr)
        {
            og::call<void>(relay_, "release");
            relay_ = nullptr;
        }
    }

    void setBoundsPlatform(int width, int height) noexcept
    {
        const objc_glue::Rect frame { 0, 0, static_cast<double>(width),
                                      static_cast<double>(height) };
        objc_glue::call<void>(webView_, "setFrame:", frame);
    }

    void setVisiblePlatform(bool visible) noexcept
    {
        objc_glue::call<void>(webView_, "setHidden:",
                              static_cast<signed char>(visible ? 0 : 1));
    }

    bool queryParentSizePlatform(int&, int&) const noexcept
    {
        // Cocoa views are logical-size and the autoresizing mask follows the
        // host view, so the negotiated size is already authoritative here.
        return false;
    }

    void evalPlatform(const std::string& js) noexcept
    {
        if (webView_ != nullptr)
            objc_glue::call<void>(webView_, "evaluateJavaScript:completionHandler:",
                                  objc_glue::nsString(js.c_str()),
                                  static_cast<objc_glue::ObjId>(nullptr));
    }

    // ==============================================================================
    // Linux — WebKitGTK in a GtkPlug, embedded in the host's X11 window
    // ==============================================================================
#elif DSPARK_WEBVIEW_BACKEND == 3

    void* plug_ = nullptr;             // GtkPlug (XEmbed top-level), owns the tree
    void* webView_ = nullptr;          // WebKitWebView widget (child of the plug)
    void* contentManager_ = nullptr;   // WebKitUserContentManager (view-owned)
    unsigned long messageSignal_ = 0;

    static void onScriptMessage(void*, void* jsResult, void* userData) noexcept
    {
        auto* self = static_cast<Editor*>(userData);
        const auto& gtk = gtk_glue::api();
        void* value = gtk.jsResultGetValue(jsResult);
        if (value == nullptr) return;
        char* text = gtk.jscValueToString(value);
        if (text != nullptr)
        {
            self->handlePost(text);
            gtk.gFree(text);
        }
    }

    bool createPlatform(void* parentWindow) noexcept
    {
        const auto& gtk = gtk_glue::api();
        if (!gtk.ok) return false;
        // Idempotent when the host (or another plugin) initialised GTK; fails
        // cleanly with no usable display (Wayland-only sessions, headless).
        if (!gtk.gtkInitCheck(nullptr, nullptr))
        {
            debugLog("linux create: gtk_init_check failed (no display)");
            return false;
        }
        plug_ = gtk.plugNew(static_cast<unsigned long>(
            reinterpret_cast<std::uintptr_t>(parentWindow)));
        if (plug_ == nullptr) return false;
        webView_ = gtk.webViewNew();
        if (webView_ == nullptr)
        {
            destroyPlatform();
            return false;
        }
        contentManager_ = gtk.viewGetContentManager(webView_);
        gtk.contentManagerRegisterHandler(contentManager_, "dspark");
        messageSignal_ = gtk.signalConnectData(
            contentManager_, "script-message-received::dspark",
            reinterpret_cast<void (*)()>(&onScriptMessage), this, nullptr, 0);
        // The uplink + bridge run at document start, before the page scripts
        // (WebKitGTK exposes the same window.webkit.messageHandlers namespace
        // as WKWebView).
        std::string bootstrap =
            "window.__dsparkPost=function(){"
            "window.webkit.messageHandlers.dspark.postMessage("
            "JSON.stringify(Array.prototype.slice.call(arguments)));};";
        bootstrap += kBridgeJs;
        void* script = gtk.userScriptNew(bootstrap.c_str(),
                                         1 /* INJECT_TOP_FRAME */,
                                         0 /* INJECT_AT_DOCUMENT_START */,
                                         nullptr, nullptr);
        gtk.contentManagerAddScript(contentManager_, script);
        gtk.userScriptUnref(script);
        gtk.containerAdd(plug_, webView_);
        gtk.loadHtml(webView_, pageHtml().c_str(), nullptr);
        gtk.widgetShowAll(plug_);
        debugLog("linux create: plug for X11 window %p", parentWindow);
        pumpPlatform();   // realize + map promptly, before the first host tick
        return true;
    }

    void destroyPlatform() noexcept
    {
        const auto& gtk = gtk_glue::api();
        if (!gtk.ok) return;
        if (contentManager_ != nullptr)
        {
            if (messageSignal_ != 0)
                gtk.signalHandlerDisconnect(contentManager_, messageSignal_);
            gtk.contentManagerUnregisterHandler(contentManager_, "dspark");
        }
        if (plug_ != nullptr)
            gtk.widgetDestroy(plug_);   // destroys the whole widget tree
        plug_ = nullptr;
        webView_ = nullptr;
        contentManager_ = nullptr;
        messageSignal_ = 0;
        pumpPlatform();   // flush the destroy events while we still can
    }

    void setBoundsPlatform(int width, int height) noexcept
    {
        const auto& gtk = gtk_glue::api();
        if (plug_ == nullptr) return;
        gtk.windowSetDefaultSize(plug_, width, height);   // pre-map size
        gtk.windowResize(plug_, width, height);           // post-map size
        pumpPlatform();
    }

    void setVisiblePlatform(bool visible) noexcept
    {
        if (plug_ != nullptr)
            gtk_glue::api().widgetSetVisible(plug_, visible ? 1 : 0);
    }

    bool queryParentSizePlatform(int&, int&) const noexcept
    {
        // The plug follows the host window through XEmbed configure events,
        // so the negotiated size is authoritative here (as on macOS).
        return false;
    }

    void evalPlatform(const std::string& js) noexcept
    {
        const auto& gtk = gtk_glue::api();
        if (webView_ == nullptr) return;
        if (gtk.runJs != nullptr)
            gtk.runJs(webView_, js.c_str(), nullptr, nullptr, nullptr);
        else
            gtk.evalJs(webView_, js.c_str(), -1, nullptr, nullptr,
                       nullptr, nullptr, nullptr);
    }

    void pumpPlatform() noexcept
    {
        const auto& gtk = gtk_glue::api();
        if (!gtk.ok) return;
        while (gtk.mainContextPending(nullptr) != 0)
            gtk.mainContextIteration(nullptr, 0);
    }

    // ==============================================================================
    // Other platforms — stub (hosts fall back to their generic editor)
    // ==============================================================================
#else

    bool createPlatform(void*) noexcept { return false; }
    void destroyPlatform() noexcept {}
    void setBoundsPlatform(int, int) noexcept {}
    void setVisiblePlatform(bool) noexcept {}
    bool queryParentSizePlatform(int&, int&) const noexcept { return false; }
    void evalPlatform(const std::string&) noexcept {}

#endif
};

} // namespace dspark::plugin::webview_ui
