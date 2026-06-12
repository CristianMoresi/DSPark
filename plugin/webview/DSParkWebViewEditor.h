// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkWebViewEditor.h
 * @brief WebView editor layer: plugin GUIs written in HTML/CSS/JS, embedded
 * in the host window through the platform web engine.
 *
 * Opt-in and additive: include this header BEFORE the format headers, set
 * `hasEditor = true` and serve your page from `editorHtml()` — the VST3 and
 * CLAP backends then embed it inside the window the host provides (the AU
 * backend keeps the host's generic UI). Nothing else in the plugin class
 * changes; plugins without this header keep building exactly as before.
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
 * - **Linux** — not yet wired (no stable cross-host embedding story for
 *   WebKitGTK inside an X11 socket); the editor reports unavailable and
 *   hosts fall back to their generic parameter UI. The same plugin source
 *   builds unchanged.
 *
 * Threading: every editor call (create/destroy/resize/JS callbacks) runs on
 * the host's main/UI thread. Values flow UI -> DSP through the wrapper's
 * normalized shadows and the user's atomic `setParameter` (any-thread safe
 * by contract), and DSP -> UI by polling those shadows at ~30 Hz from the
 * page — no native timers, no locks, no allocation on the audio thread.
 */

#if defined(DSPARK_PLUGIN_VST3_INCLUDED) || defined(DSPARK_PLUGIN_CLAP_INCLUDED)
#error "Include plugin/webview/DSParkWebViewEditor.h BEFORE the plugin format headers"
#endif

#define DSPARK_PLUGIN_WEBVIEW 1

#include "../DSParkPlugin.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

// -- platform backend selection --------------------------------------------------
//
// DSPARK_WEBVIEW_BACKEND: 1 = WebView2 via vendored webview.h (Windows),
//                         2 = WKWebView via the Objective-C runtime (macOS),
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
    #include <memory>
#elif defined(__APPLE__)
    #define DSPARK_WEBVIEW_BACKEND 2
    #include <objc/message.h>
    #include <objc/runtime.h>
    #include <dlfcn.h>
#else
    #define DSPARK_WEBVIEW_BACKEND 0
#endif

namespace dspark::plugin::webview_ui {

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
    "if(window.__dsparkScale){document.documentElement.style.zoom=window.__dsparkScale;}"
    "var ls={},vals={},meta=null,readyCbs=[];"
    "function fire(id,v){var c=ls[id];if(c){for(var i=0;i<c.length;i++){c[i](v);}}}"
    "window.__dsparkRecv=function(m){"
      "if(m.type==='params'){"
        "meta=m.params;"
        "for(var i=0;i<meta.length;i++){vals[meta[i].id]=meta[i].value;}"
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
     * Applies the host's content scale: the native box is expected in
     * physical pixels and the page is zoomed to match, so CSS keeps working
     * in logical units. No-op platforms (macOS) simply never call this.
     */
    void setScale(double scale) noexcept
    {
        scale_ = scale > 0.0 ? scale : 1.0;
        if (created_)
        {
            std::string js = "document.documentElement.style.zoom=";
            appendJsonNumber(js, scale_);
            js += ";";
            evalPlatform(js);
        }
    }

    [[nodiscard]] bool created() const noexcept { return created_; }

private:
    static constexpr size_t kNumParams = P::parameters.size();

    std::atomic<double> const* shadows_ = nullptr;
    HostCallbacks host_ {};
    bool   created_ = false;
    double scale_ = 1.0;

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
        json += "]}";
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

    // ==============================================================================
    // Windows — WebView2 through the vendored webview library
    // ==============================================================================
#if DSPARK_WEBVIEW_BACKEND == 1

    std::unique_ptr<webview::webview> wv_;
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
            if (scale_ != 1.0)
            {
                std::string preset = "window.__dsparkScale=";
                appendJsonNumber(preset, scale_);
                preset += ";";
                wv_->init(preset);
            }
            wv_->init(kBridgeJs);
            wv_->set_html(P::editorHtml());
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

    void destroyPlatform() noexcept
    {
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
                       og::nsString(P::editorHtml()), static_cast<og::ObjId>(nullptr));
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

    void evalPlatform(const std::string& js) noexcept
    {
        if (webView_ != nullptr)
            objc_glue::call<void>(webView_, "evaluateJavaScript:completionHandler:",
                                  objc_glue::nsString(js.c_str()),
                                  static_cast<objc_glue::ObjId>(nullptr));
    }

    // ==============================================================================
    // Other platforms — stub (hosts fall back to their generic editor)
    // ==============================================================================
#else

    bool createPlatform(void*) noexcept { return false; }
    void destroyPlatform() noexcept {}
    void setBoundsPlatform(int, int) noexcept {}
    void setVisiblePlatform(bool) noexcept {}
    void evalPlatform(const std::string&) noexcept {}

#endif
};

} // namespace dspark::plugin::webview_ui
