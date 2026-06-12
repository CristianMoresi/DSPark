// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkClap.h
 * @brief Native CLAP backend: the same plugin class, one more macro.
 *
 * CLAP (https://cleveraudio.org, MIT, vendored under plugin/clap/clap/) is a
 * plain-C plugin ABI — the cleanest of the desktop formats. This backend
 * exposes the exact same user class as the VST3 backend:
 *
 * ```cpp
 * #include "plugin/vst3/DSParkVst3.h"
 * #include "plugin/clap/DSParkClap.h"
 * struct MyPlugin { ... };           // see plugin/DSParkPlugin.h
 * DSPARK_VST3_PLUGIN(MyPlugin)
 * DSPARK_CLAP_PLUGIN(MyPlugin)       // both ABIs can live in ONE binary:
 * ```                                // ship it as .vst3 AND as .clap
 *
 * Implementation notes:
 * - CLAP parameters travel in PLAIN values (min..max), so clap_param_info
 *   maps 1:1 onto the declarative table; the wrapper still keeps its
 *   normalized shadows so state blobs stay byte-identical with the VST3
 *   backend (presets are portable across formats by construction).
 * - All parameter events in a block are applied in order (block-rate v1,
 *   like the VST3 backend; DSPark setters smooth internally).
 * - The wrapper-owned Bypass uses CLAP_PARAM_IS_BYPASS and the same short
 *   crossfade against the kept dry signal.
 * - Extensions implemented: audio-ports (stereo in/out), params, state,
 *   latency, tail — plus gui when plugin/webview/DSParkWebViewEditor.h is
 *   included before this header and the class declares `hasEditor = true`
 *   (otherwise hosts show their generic editor). Editor edits reach the host
 *   as gesture begin/end plus CLAP_EVENT_PARAM_VALUE through a lock-free
 *   queue drained by process()/flush(), so automation recording works.
 */

#define DSPARK_PLUGIN_CLAP_INCLUDED 1

#include "../DSParkPlugin.h"

#include "clap/clap.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <new>

namespace dspark::plugin::clap_backend {

inline constexpr uint32_t kBypassParamId = 0x42595053u;   // matches the VST3 backend
inline constexpr int      kBypassRampSamples = 256;

template <typename P>
struct Plugin
{
    static constexpr size_t kNumParams = P::parameters.size();

    clap_plugin_t plugin {};            // the C-facing object (plugin_data = this)
    const clap_host_t* host = nullptr;

    P user {};
    double sampleRate = 48000.0;
    uint32_t maxFrames = 0;
    bool prepared = false;
    int cachedLatency = 0;

    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<bool>   bypass { false };
    float bypassMix = 0.0f;
    std::vector<float> dryL, dryR;

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- editor state + UI -> host event queue (single producer: main thread;
    // single consumer: process() on audio or flush() on main — never both).
    webview_ui::Editor<P> guiEditor;
    bool   guiActive = false;
    double guiScale = 1.0;
    int    guiWidth = 0, guiHeight = 0;
    bool   guiEditActive[kNumParams == 0 ? 1 : kNumParams] {};
    const clap_host_params_t* hostParams = nullptr;
#if defined(__linux__)
    // GTK is driven from the host's run loop: a timer-support tick pumps it.
    const clap_host_timer_support_t* hostTimer = nullptr;
    clap_id guiTimerId = CLAP_INVALID_ID;
#endif

    enum : uint8_t { kUiGestureBegin = 0, kUiValue = 1, kUiGestureEnd = 2 };
    struct UiEvent
    {
        uint32_t paramId;
        uint8_t  kind;
        double   value;
    };
    static constexpr uint32_t kUiQueueSize = 256;   // power of two
    UiEvent uiQueue[kUiQueueSize] {};
    std::atomic<uint32_t> uiHead { 0 }, uiTail { 0 };
#endif

    explicit Plugin(const clap_host_t* h) noexcept : host(h)
    {
        for (size_t i = 0; i < kNumParams; ++i)
            shadow[i].store(toNormalized(P::parameters[i], P::parameters[i].defValue),
                            std::memory_order_relaxed);
    }

    static Plugin* self(const clap_plugin_t* p) noexcept
    {
        return static_cast<Plugin*>(p->plugin_data);
    }

    void applyNormalized(int index, double normalized) noexcept
    {
        const auto& spec = P::parameters[static_cast<size_t>(index)];
        shadow[static_cast<size_t>(index)].store(normalized, std::memory_order_relaxed);
        user.setParameter(index, static_cast<float>(toPlain(spec, normalized)));
    }

    static int indexOfParamId(uint32_t id) noexcept
    {
        for (size_t i = 0; i < kNumParams; ++i)
            if (hash32(P::parameters[i].id) == id) return static_cast<int>(i);
        return -1;
    }

    void handleEvent(const clap_event_header_t* ev) noexcept
    {
        if (ev->space_id != CLAP_CORE_EVENT_SPACE_ID
            || ev->type != CLAP_EVENT_PARAM_VALUE)
            return;
        const auto* pv = reinterpret_cast<const clap_event_param_value_t*>(ev);
        if (pv->param_id == kBypassParamId)
        {
            bypass.store(pv->value >= 0.5, std::memory_order_relaxed);
            return;
        }
        const int idx = indexOfParamId(pv->param_id);
        if (idx < 0) return;
        // CLAP events carry PLAIN values.
        applyNormalized(idx, toNormalized(P::parameters[static_cast<size_t>(idx)], pv->value));
    }

    void drainEvents(const clap_input_events_t* events) noexcept
    {
        if (events == nullptr) return;
        const uint32_t n = events->size(events);
        for (uint32_t i = 0; i < n; ++i)
            if (const clap_event_header_t* ev = events->get(events, i))
                handleEvent(ev);
    }

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- UI -> host parameter events ---------------------------------------------

    void pushUiEvent(uint32_t paramId, uint8_t kind, double value) noexcept
    {
        const uint32_t tail = uiTail.load(std::memory_order_relaxed);
        const uint32_t next = (tail + 1) & (kUiQueueSize - 1);
        if (next == uiHead.load(std::memory_order_acquire)) return;   // full: drop
        uiQueue[tail] = UiEvent { paramId, kind, value };
        uiTail.store(next, std::memory_order_release);
    }

    void requestUiFlush() noexcept
    {
        // Host schedules process()/flush(), which drains the queue below.
        if (hostParams != nullptr && host != nullptr)
            hostParams->request_flush(host);
    }

    void drainUiEvents(const clap_output_events_t* out) noexcept
    {
        if (out == nullptr) return;
        uint32_t head = uiHead.load(std::memory_order_relaxed);
        const uint32_t tail = uiTail.load(std::memory_order_acquire);
        while (head != tail)
        {
            const UiEvent& e = uiQueue[head];
            if (e.kind == kUiValue)
            {
                clap_event_param_value_t ev {};
                ev.header.size = sizeof(ev);
                ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
                ev.header.type = CLAP_EVENT_PARAM_VALUE;
                ev.param_id = e.paramId;
                ev.note_id = -1;
                ev.port_index = -1;
                ev.channel = -1;
                ev.key = -1;
                ev.value = e.value;
                out->try_push(out, &ev.header);
            }
            else
            {
                clap_event_param_gesture_t ev {};
                ev.header.size = sizeof(ev);
                ev.header.space_id = CLAP_CORE_EVENT_SPACE_ID;
                ev.header.type = static_cast<uint16_t>(
                    e.kind == kUiGestureBegin ? CLAP_EVENT_PARAM_GESTURE_BEGIN
                                              : CLAP_EVENT_PARAM_GESTURE_END);
                ev.param_id = e.paramId;
                out->try_push(out, &ev.header);
            }
            head = (head + 1) & (kUiQueueSize - 1);
        }
        uiHead.store(head, std::memory_order_release);
    }
#endif // DSPARK_PLUGIN_WEBVIEW

    // --- clap_plugin_t callbacks ------------------------------------------------

    static bool sInit(const clap_plugin_t* p) noexcept
    {
#if defined(DSPARK_PLUGIN_WEBVIEW)
        auto* s = self(p);
        if (s->host != nullptr)
            s->hostParams = static_cast<const clap_host_params_t*>(
                s->host->get_extension(s->host, CLAP_EXT_PARAMS));
#else
        (void) p;
#endif
        return true;
    }

    static void sDestroy(const clap_plugin_t* p) noexcept { delete self(p); }

    static bool sActivate(const clap_plugin_t* p, double sr,
                          uint32_t, uint32_t maxFrames) noexcept
    {
        auto* s = self(p);
        s->sampleRate = sr;
        s->maxFrames = maxFrames;
        dspark::AudioSpec spec { sr, static_cast<int>(maxFrames), 2 };
        s->user.prepare(spec);
        for (size_t i = 0; i < kNumParams; ++i)
            s->user.setParameter(static_cast<int>(i),
                static_cast<float>(toPlain(P::parameters[i],
                                           s->shadow[i].load(std::memory_order_relaxed))));
        s->dryL.assign(maxFrames, 0.0f);
        s->dryR.assign(maxFrames, 0.0f);
        s->bypassMix = s->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if constexpr (HasLatency<P>)
            s->cachedLatency = s->user.getLatency();
        s->prepared = true;
        return true;
    }

    static void sDeactivate(const clap_plugin_t*) noexcept {}
    static bool sStartProcessing(const clap_plugin_t*) noexcept { return true; }
    static void sStopProcessing(const clap_plugin_t*) noexcept {}

    static void sReset(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
        if constexpr (HasReset<P>)
            s->user.reset();
        s->bypassMix = s->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
    }

    static clap_process_status sProcess(const clap_plugin_t* p,
                                        const clap_process_t* process) noexcept
    {
        auto* s = self(p);
        s->drainEvents(process->in_events);
#if defined(DSPARK_PLUGIN_WEBVIEW)
        s->drainUiEvents(process->out_events);
#endif

        const uint32_t n = process->frames_count;
        if (n == 0 || !s->prepared) return CLAP_PROCESS_CONTINUE;
        if (process->audio_outputs_count < 1) return CLAP_PROCESS_CONTINUE;

        auto& outPort = process->audio_outputs[0];
        float** out = outPort.data32;
        if (out == nullptr || outPort.channel_count < 1) return CLAP_PROCESS_CONTINUE;
        const uint32_t nCh = outPort.channel_count < 2 ? outPort.channel_count : 2;

        const bool haveIn = process->audio_inputs_count >= 1
                         && process->audio_inputs[0].data32 != nullptr;
        float** in = haveIn ? process->audio_inputs[0].data32 : nullptr;

        float* dry[2] = { s->dryL.data(), s->dryR.data() };
        const size_t bytes = sizeof(float) * n;
        for (uint32_t ch = 0; ch < nCh; ++ch)
        {
            const float* src = (haveIn && in[ch] != nullptr) ? in[ch] : out[ch];
            if (n <= s->dryL.size())
                std::memcpy(dry[ch], src, bytes);
            if (out[ch] != src)
                std::memcpy(out[ch], src, bytes);
        }

        dspark::AudioBufferView<float> view(out, static_cast<int>(nCh),
                                            static_cast<int>(n));
        s->user.processBlock(view);

        const float target = s->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (s->bypassMix != target || target > 0.0f)
        {
            const float step = 1.0f / static_cast<float>(kBypassRampSamples);
            float mix = s->bypassMix;
            if (n <= s->dryL.size())
            {
                for (uint32_t i = 0; i < n; ++i)
                {
                    mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
                    mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
                    for (uint32_t ch = 0; ch < nCh; ++ch)
                        out[ch][i] += (dry[ch][i] - out[ch][i]) * mix;
                }
                s->bypassMix = mix;
            }
        }
        return CLAP_PROCESS_CONTINUE;
    }

    static const void* sGetExtension(const clap_plugin_t*, const char* id) noexcept
    {
        if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &kAudioPorts;
        if (std::strcmp(id, CLAP_EXT_PARAMS) == 0)      return &kParams;
        if (std::strcmp(id, CLAP_EXT_STATE) == 0)       return &kState;
        if (std::strcmp(id, CLAP_EXT_LATENCY) == 0)     return &kLatency;
        if (std::strcmp(id, CLAP_EXT_TAIL) == 0)        return &kTail;
#if defined(DSPARK_PLUGIN_WEBVIEW)
        if constexpr (HasEditor<P>)
        {
            if (webview_ui::Editor<P>::available() && std::strcmp(id, CLAP_EXT_GUI) == 0)
                return &kGui;
#if defined(__linux__)
            if (webview_ui::Editor<P>::available()
                && std::strcmp(id, CLAP_EXT_TIMER_SUPPORT) == 0)
                return &kTimer;
#endif
        }
#endif
        return nullptr;
    }

    static void sOnMainThread(const clap_plugin_t*) noexcept {}

    // --- ext: audio ports ---------------------------------------------------------

    static uint32_t sPortCount(const clap_plugin_t*, bool) noexcept { return 1; }

    static bool sPortGet(const clap_plugin_t*, uint32_t index, bool isInput,
                         clap_audio_port_info_t* info) noexcept
    {
        if (index != 0 || info == nullptr) return false;
        std::memset(info, 0, sizeof(*info));
        info->id = isInput ? 0 : 1;
        std::snprintf(info->name, sizeof(info->name), "%s",
                      isInput ? "Input" : "Output");
        info->flags = CLAP_AUDIO_PORT_IS_MAIN;
        info->channel_count = 2;
        info->port_type = CLAP_PORT_STEREO;
        info->in_place_pair = isInput ? 1 : 0;
        return true;
    }

    inline static const clap_plugin_audio_ports_t kAudioPorts = { &sPortCount, &sPortGet };

    // --- ext: params ----------------------------------------------------------------

    static uint32_t sParamCount(const clap_plugin_t*) noexcept
    {
        return static_cast<uint32_t>(kNumParams) + 1;   // + Bypass
    }

    static bool sParamInfo(const clap_plugin_t*, uint32_t index,
                           clap_param_info_t* info) noexcept
    {
        if (info == nullptr || index > kNumParams) return false;
        std::memset(info, 0, sizeof(*info));
        if (index == kNumParams)
        {
            info->id = kBypassParamId;
            std::snprintf(info->name, sizeof(info->name), "Bypass");
            info->flags = CLAP_PARAM_IS_AUTOMATABLE | CLAP_PARAM_IS_STEPPED
                        | CLAP_PARAM_IS_BYPASS;
            info->min_value = 0.0;
            info->max_value = 1.0;
            info->default_value = 0.0;
            return true;
        }
        const auto& spec = P::parameters[index];
        info->id = hash32(spec.id);
        std::snprintf(info->name, sizeof(info->name), "%s", spec.name);
        info->flags = CLAP_PARAM_IS_AUTOMATABLE;
        if (spec.steps > 0) info->flags |= CLAP_PARAM_IS_STEPPED;
        info->min_value = spec.minValue;
        info->max_value = spec.maxValue;
        info->default_value = spec.defValue;
        return true;
    }

    static bool sParamValue(const clap_plugin_t* p, clap_id id, double* out) noexcept
    {
        auto* s = self(p);
        if (out == nullptr) return false;
        if (id == kBypassParamId)
        {
            *out = s->bypass.load(std::memory_order_relaxed) ? 1.0 : 0.0;
            return true;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return false;
        *out = toPlain(P::parameters[static_cast<size_t>(idx)],
                       s->shadow[static_cast<size_t>(idx)].load(std::memory_order_relaxed));
        return true;
    }

    static bool sParamValueToText(const clap_plugin_t*, clap_id id, double value,
                                  char* out, uint32_t size) noexcept
    {
        if (out == nullptr || size == 0) return false;
        if (id == kBypassParamId)
        {
            std::snprintf(out, size, "%s", value >= 0.5 ? "On" : "Off");
            return true;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return false;
        formatValue(P::parameters[static_cast<size_t>(idx)], value, out,
                    static_cast<int>(size));
        return true;
    }

    static bool sParamTextToValue(const clap_plugin_t*, clap_id id,
                                  const char* text, double* out) noexcept
    {
        if (text == nullptr || out == nullptr) return false;
        const int toggle = parseToggleText(text);
        if (id == kBypassParamId)
        {
            *out = toggle >= 0 ? toggle
                               : (std::strtod(text, nullptr) >= 0.5 ? 1.0 : 0.0);
            return true;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return false;
        const Param& spec = P::parameters[static_cast<size_t>(idx)];
        if (spec.steps == 1 && toggle >= 0)
            *out = toggle != 0 ? spec.maxValue : spec.minValue;
        else
            *out = std::strtod(text, nullptr);
        return true;
    }

    static void sParamFlush(const clap_plugin_t* p, const clap_input_events_t* in,
                            const clap_output_events_t* out) noexcept
    {
        self(p)->drainEvents(in);
#if defined(DSPARK_PLUGIN_WEBVIEW)
        self(p)->drainUiEvents(out);
#else
        (void) out;
#endif
    }

    inline static const clap_plugin_params_t kParams = {
        &sParamCount, &sParamInfo, &sParamValue,
        &sParamValueToText, &sParamTextToValue, &sParamFlush
    };

    // --- ext: state -------------------------------------------------------------------

    static bool sStateSave(const clap_plugin_t* p, const clap_ostream_t* stream) noexcept
    {
        auto* s = self(p);
        double norm[kNumParams == 0 ? 1 : kNumParams];
        for (size_t i = 0; i < kNumParams; ++i)
            norm[i] = s->shadow[i].load(std::memory_order_relaxed);
        const std::vector<uint8_t> blob = buildState(s->user, norm, kNumParams);
        size_t pos = 0;
        while (pos < blob.size())
        {
            const int64_t put = stream->write(stream, blob.data() + pos,
                                              blob.size() - pos);
            if (put <= 0) return false;
            pos += static_cast<size_t>(put);
        }
        return true;
    }

    static bool sStateLoad(const clap_plugin_t* p, const clap_istream_t* stream) noexcept
    {
        auto* s = self(p);
        std::vector<uint8_t> blob;
        uint8_t chunk[4096];
        for (;;)
        {
            const int64_t got = stream->read(stream, chunk, sizeof(chunk));
            if (got < 0) return false;
            if (got == 0) break;
            blob.insert(blob.end(), chunk, chunk + got);
            if (blob.size() > (16u << 20)) return false;
        }
        double norm[kNumParams == 0 ? 1 : kNumParams];
        for (size_t i = 0; i < kNumParams; ++i)
            norm[i] = s->shadow[i].load(std::memory_order_relaxed);
        if (!applyState(s->user, blob.data(), blob.size(), norm))
            return false;
        for (size_t i = 0; i < kNumParams; ++i)
            s->applyNormalized(static_cast<int>(i), norm[i]);
        return true;
    }

    inline static const clap_plugin_state_t kState = { &sStateSave, &sStateLoad };

    // --- ext: latency / tail --------------------------------------------------------

    static uint32_t sLatencyGet(const clap_plugin_t* p) noexcept
    {
        return static_cast<uint32_t>(self(p)->cachedLatency);
    }

    inline static const clap_plugin_latency_t kLatency = { &sLatencyGet };

    static uint32_t sTailGet(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
        if constexpr (HasTail<P>)
            return static_cast<uint32_t>(s->user.getTailSeconds() * s->sampleRate);
        else
        {
            (void) s;
            return 0;
        }
    }

    inline static const clap_plugin_tail_t kTail = { &sTailGet };

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- ext: gui (WebView editor layer) ----------------------------------------------
    //
    // CLAP hands the parent window AFTER create(), so create() only arms the
    // state and set_parent() builds the actual web engine. All [main-thread].

    static const char* clapWindowApi() noexcept
    {
#if defined(_WIN32)
        return CLAP_WINDOW_API_WIN32;
#elif defined(__APPLE__)
        return CLAP_WINDOW_API_COCOA;
#else
        return CLAP_WINDOW_API_X11;
#endif
    }

    static void cbGuiSetParam(void* context, int index, double plain) noexcept
    {
        auto* s = static_cast<Plugin*>(context);
        const Param& spec = P::parameters[static_cast<size_t>(index)];
        const double normalized = toNormalized(spec, plain);
        s->applyNormalized(index, normalized);
        const double snapped = toPlain(spec, normalized);
        const uint32_t id = hash32(spec.id);
        if (s->guiEditActive[static_cast<size_t>(index)])
            s->pushUiEvent(id, kUiValue, snapped);
        else
        {
            // Edits outside an explicit gesture still get one (host undo).
            s->pushUiEvent(id, kUiGestureBegin, 0.0);
            s->pushUiEvent(id, kUiValue, snapped);
            s->pushUiEvent(id, kUiGestureEnd, 0.0);
        }
        s->requestUiFlush();
    }

    static void cbGuiBeginEdit(void* context, int index) noexcept
    {
        auto* s = static_cast<Plugin*>(context);
        s->guiEditActive[static_cast<size_t>(index)] = true;
        s->pushUiEvent(hash32(P::parameters[static_cast<size_t>(index)].id),
                       kUiGestureBegin, 0.0);
        s->requestUiFlush();
    }

    static void cbGuiEndEdit(void* context, int index) noexcept
    {
        auto* s = static_cast<Plugin*>(context);
        s->guiEditActive[static_cast<size_t>(index)] = false;
        s->pushUiEvent(hash32(P::parameters[static_cast<size_t>(index)].id),
                       kUiGestureEnd, 0.0);
        s->requestUiFlush();
    }

    static bool sGuiIsApiSupported(const clap_plugin_t*, const char* api,
                                   bool isFloating) noexcept
    {
        return !isFloating && api != nullptr && std::strcmp(api, clapWindowApi()) == 0
            && webview_ui::Editor<P>::available();
    }

    static bool sGuiGetPreferredApi(const clap_plugin_t*, const char** api,
                                    bool* isFloating) noexcept
    {
        if (api == nullptr || isFloating == nullptr) return false;
        *api = clapWindowApi();
        *isFloating = false;
        return webview_ui::Editor<P>::available();
    }

    static bool sGuiCreate(const clap_plugin_t* p, const char* api,
                           bool isFloating) noexcept
    {
        auto* s = self(p);
        if (!sGuiIsApiSupported(p, api, isFloating)) return false;
#if defined(__linux__)
        // GTK breathes only when pumped from the host's run loop; without
        // timer-support the page would freeze — fall back to generic UI.
        s->hostTimer = s->host != nullptr
            ? static_cast<const clap_host_timer_support_t*>(
                  s->host->get_extension(s->host, CLAP_EXT_TIMER_SUPPORT))
            : nullptr;
        if (s->hostTimer == nullptr || s->hostTimer->register_timer == nullptr
            || !s->hostTimer->register_timer(s->host, 33, &s->guiTimerId))
        {
            webview_ui::debugLog("clap gui create: no host timer-support -> no editor");
            s->hostTimer = nullptr;
            s->guiTimerId = CLAP_INVALID_ID;
            return false;
        }
#endif
        const EditorSize logical = editorSizeOf<P>();
        s->guiWidth  = static_cast<int>(logical.width * s->guiScale + 0.5);
        s->guiHeight = static_cast<int>(logical.height * s->guiScale + 0.5);
        s->guiActive = true;
        webview_ui::debugLog("clap gui create %dx%d scale=%.2f",
                             s->guiWidth, s->guiHeight, s->guiScale);
        return true;
    }

    static void sGuiDestroy(const clap_plugin_t* p) noexcept
    {
        auto* s = self(p);
#if defined(__linux__)
        if (s->hostTimer != nullptr && s->guiTimerId != CLAP_INVALID_ID)
            s->hostTimer->unregister_timer(s->host, s->guiTimerId);
        s->hostTimer = nullptr;
        s->guiTimerId = CLAP_INVALID_ID;
#endif
        s->guiEditor.destroy();
        s->guiActive = false;
    }

    static bool sGuiSetScale(const clap_plugin_t* p, double scale) noexcept
    {
#if defined(__APPLE__)
        (void) p; (void) scale;
        return false;   // cocoa uses logical sizes; scaling is the OS's job
#else
        auto* s = self(p);
        if (scale <= 0.0) return false;
        // Only the physical-size negotiation changes; the page itself never
        // zooms (the web engine applies the window DPI to CSS pixels itself).
        const double previous = s->guiScale;
        s->guiScale = scale;
        s->guiWidth  = static_cast<int>(s->guiWidth * (scale / previous) + 0.5);
        s->guiHeight = static_cast<int>(s->guiHeight * (scale / previous) + 0.5);
        s->guiEditor.setBounds(s->guiWidth, s->guiHeight);
        return true;
#endif
    }

    static bool sGuiGetSize(const clap_plugin_t* p, uint32_t* width,
                            uint32_t* height) noexcept
    {
        auto* s = self(p);
        if (width == nullptr || height == nullptr || !s->guiActive) return false;
        *width  = static_cast<uint32_t>(s->guiWidth);
        *height = static_cast<uint32_t>(s->guiHeight);
        return true;
    }

    static bool sGuiCanResize(const clap_plugin_t*) noexcept
    {
        return editorResizeOf<P>() != EditorResize::Fixed;
    }

    static bool sGuiGetResizeHints(const clap_plugin_t*,
                                   clap_gui_resize_hints_t* hints) noexcept
    {
        if (hints == nullptr) return false;
        constexpr EditorResize mode = editorResizeOf<P>();
        const EditorSize logical = editorSizeOf<P>();
        hints->can_resize_horizontally = mode != EditorResize::Fixed;
        hints->can_resize_vertically = mode != EditorResize::Fixed;
        hints->preserve_aspect_ratio = mode == EditorResize::KeepAspect;
        hints->aspect_ratio_width = static_cast<uint32_t>(logical.width);
        hints->aspect_ratio_height = static_cast<uint32_t>(logical.height);
        return true;
    }

    static bool sGuiAdjustSize(const clap_plugin_t* p, uint32_t* width,
                               uint32_t* height) noexcept
    {
        auto* s = self(p);
        if (width == nullptr || height == nullptr) return false;
        double w = *width;
        double h = *height;
        constrainEditorSize<P>(w, h, s->guiScale);
        *width  = static_cast<uint32_t>(w + 0.5);
        *height = static_cast<uint32_t>(h + 0.5);
        return true;
    }

    static bool sGuiSetSize(const clap_plugin_t* p, uint32_t width,
                            uint32_t height) noexcept
    {
        auto* s = self(p);
        if (!s->guiActive) return false;
        // Clamp here too: not every host runs the size through adjust_size.
        double w = width;
        double h = height;
        constrainEditorSize<P>(w, h, s->guiScale);
        webview_ui::debugLog("clap gui set_size %ux%u -> %.0fx%.0f", width, height, w, h);
        s->guiWidth  = static_cast<int>(w + 0.5);
        s->guiHeight = static_cast<int>(h + 0.5);
        s->guiEditor.setBounds(s->guiWidth, s->guiHeight);
        return true;
    }

    static bool sGuiSetParent(const clap_plugin_t* p,
                              const clap_window_t* window) noexcept
    {
        auto* s = self(p);
        if (window == nullptr || !s->guiActive) return false;
#if defined(__linux__)
        // The X11 window id rides the union's dedicated field.
        void* parentHandle = reinterpret_cast<void*>(
            static_cast<std::uintptr_t>(window->x11));
#else
        void* parentHandle = window->ptr;
#endif
        webview_ui::debugLog("clap gui set_parent %p negotiated=%dx%d",
                             parentHandle, s->guiWidth, s->guiHeight);
        const webview_ui::HostCallbacks callbacks {
            s, &cbGuiSetParam, &cbGuiBeginEdit, &cbGuiEndEdit
        };
        if (!s->guiEditor.create(parentHandle, s->shadow, callbacks))
            return false;
        // Fill whatever box the host actually built (call order differs
        // between hosts); fall back to the negotiated size otherwise.
        int parentW = 0, parentH = 0;
        if (s->guiEditor.queryParentSize(parentW, parentH))
        {
            s->guiWidth  = parentW;
            s->guiHeight = parentH;
        }
        s->guiEditor.setBounds(s->guiWidth, s->guiHeight);
        return true;
    }

    static bool sGuiSetTransient(const clap_plugin_t*, const clap_window_t*) noexcept
    {
        return false;   // embedded only; no floating window mode
    }

    static void sGuiSuggestTitle(const clap_plugin_t*, const char*) noexcept {}

    static bool sGuiShow(const clap_plugin_t* p) noexcept
    {
        self(p)->guiEditor.setVisible(true);
        return true;
    }

    static bool sGuiHide(const clap_plugin_t* p) noexcept
    {
        self(p)->guiEditor.setVisible(false);
        return true;
    }

    inline static const clap_plugin_gui_t kGui = {
        &sGuiIsApiSupported, &sGuiGetPreferredApi, &sGuiCreate, &sGuiDestroy,
        &sGuiSetScale, &sGuiGetSize, &sGuiCanResize, &sGuiGetResizeHints,
        &sGuiAdjustSize, &sGuiSetSize, &sGuiSetParent, &sGuiSetTransient,
        &sGuiSuggestTitle, &sGuiShow, &sGuiHide
    };

#if defined(__linux__)

    // --- ext: timer-support (drives the GTK main context for the editor) -----------

    static void sOnTimer(const clap_plugin_t* p, clap_id) noexcept
    {
        self(p)->guiEditor.pump();
    }

    inline static const clap_plugin_timer_support_t kTimer = { &sOnTimer };

#endif // __linux__
#endif // DSPARK_PLUGIN_WEBVIEW

    // --- descriptor & factory ----------------------------------------------------------

    static const clap_plugin_descriptor_t* descriptor() noexcept
    {
        static const char* features[] = {
            CLAP_PLUGIN_FEATURE_AUDIO_EFFECT, CLAP_PLUGIN_FEATURE_STEREO, nullptr
        };
        static const clap_plugin_descriptor_t desc = {
            CLAP_VERSION_INIT,
            P::descriptor.productId,
            P::descriptor.name,
            P::descriptor.vendor,
            P::descriptor.url,
            "",                       // manual_url
            P::descriptor.url,        // support_url
            P::descriptor.version,
            "Built with DSPark",      // description
            features
        };
        return &desc;
    }

    static const clap_plugin_t* create(const clap_host_t* host) noexcept
    {
        auto* s = new (std::nothrow) Plugin(host);
        if (s == nullptr) return nullptr;
        s->plugin.desc = descriptor();
        s->plugin.plugin_data = s;
        s->plugin.init = &sInit;
        s->plugin.destroy = &sDestroy;
        s->plugin.activate = &sActivate;
        s->plugin.deactivate = &sDeactivate;
        s->plugin.start_processing = &sStartProcessing;
        s->plugin.stop_processing = &sStopProcessing;
        s->plugin.reset = &sReset;
        s->plugin.process = &sProcess;
        s->plugin.get_extension = &sGetExtension;
        s->plugin.on_main_thread = &sOnMainThread;
        return &s->plugin;
    }
};

template <typename P>
struct Factory
{
    static uint32_t sCount(const clap_plugin_factory_t*) noexcept { return 1; }

    static const clap_plugin_descriptor_t* sDescriptor(const clap_plugin_factory_t*,
                                                       uint32_t index) noexcept
    {
        return index == 0 ? Plugin<P>::descriptor() : nullptr;
    }

    static const clap_plugin_t* sCreate(const clap_plugin_factory_t*,
                                        const clap_host_t* host,
                                        const char* pluginId) noexcept
    {
        if (pluginId == nullptr
            || std::strcmp(pluginId, P::descriptor.productId) != 0)
            return nullptr;
        return Plugin<P>::create(host);
    }

    inline static const clap_plugin_factory_t kFactory = {
        &sCount, &sDescriptor, &sCreate
    };

    static bool sEntryInit(const char*) noexcept { return true; }
    static void sEntryDeinit() noexcept {}

    static const void* sEntryGetFactory(const char* factoryId) noexcept
    {
        if (factoryId != nullptr
            && std::strcmp(factoryId, CLAP_PLUGIN_FACTORY_ID) == 0)
            return &kFactory;
        return nullptr;
    }
};

} // namespace dspark::plugin::clap_backend

/**
 * @brief Declares the CLAP module entry for one plugin class. May coexist
 * with DSPARK_VST3_PLUGIN in the same translation unit: the resulting binary
 * is a valid .vst3 AND a valid .clap at the same time.
 */
#define DSPARK_CLAP_PLUGIN(PluginClass)                                            \
    extern "C" CLAP_EXPORT const clap_plugin_entry_t clap_entry = {                \
        CLAP_VERSION_INIT,                                                         \
        &dspark::plugin::clap_backend::Factory<PluginClass>::sEntryInit,           \
        &dspark::plugin::clap_backend::Factory<PluginClass>::sEntryDeinit,         \
        &dspark::plugin::clap_backend::Factory<PluginClass>::sEntryGetFactory      \
    };
