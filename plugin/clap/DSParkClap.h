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
 * - Parameter events apply SAMPLE-ACCURATELY by default: every timestamped
 *   event splits processing at kAutomationQuantum boundaries (plugins may
 *   opt out, see DSParkPlugin.h). DSPark setters smooth internally either
 *   way.
 * - The wrapper-owned Bypass uses CLAP_PARAM_IS_BYPASS and the same short
 *   crossfade against the kept dry signal.
 * - Buses follow the declared ChannelSupport through audio-ports-config
 *   (Mono / Stereo configurations); Category::Instrument drops the audio
 *   inputs and HasMidi adds a note port accepting both CLAP note events
 *   and raw MIDI (notes, pitch bend, CC, pressure).
 * - HasTransport plugins receive the per-block clap transport as a
 *   TransportInfo; HasOfflineMode plugins get the render extension; a
 *   latency change after parameter motion notifies the host through
 *   clap_host_latency on the main thread (request_callback).
 * - Factory presets publish through preset-load + preset-discovery (the
 *   plugin DSO itself is the preset container; load keys are indices).
 * - process() runs under a DenormalGuard (FTZ/DAZ) so user DSP outside
 *   DSPark's own classes is covered in hosts that do not set it.
 * - GUI: when plugin/webview/DSParkWebViewEditor.h is included before this
 *   header and the class declares `hasEditor = true`, the gui extension
 *   embeds the WebView editor (otherwise hosts show their generic editor).
 *   Editor edits reach the host as gesture begin/end plus
 *   CLAP_EVENT_PARAM_VALUE through a lock-free queue drained by
 *   process()/flush(), so automation recording works.
 */

#define DSPARK_PLUGIN_CLAP_INCLUDED 1

#include "../DSParkPlugin.h"

#include "clap/clap.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

namespace dspark::plugin::clap_backend {

inline constexpr uint32_t kBypassParamId = 0x42595053u;   // matches the VST3 backend
inline constexpr int      kBypassRampSamples = 256;

template <typename P>
struct Plugin
{
    static constexpr size_t kNumParams = P::parameters.size();
    static constexpr bool kIsInstrument =
        P::descriptor.category == Category::Instrument;
    static constexpr int kNumPresets = factoryPresetCountOf<P>();
    static_assert(!(kIsInstrument && HasSidechain<P>),
                  "an Instrument has no audio inputs; remove the sidechain "
                  "processBlock or use Category::Fx");
    static_assert(!kIsInstrument || HasMidi<P>,
                  "an Instrument needs handleMidiEvent (see HasMidi): it has "
                  "no audio input to process");

    clap_plugin_t plugin {};            // the C-facing object (plugin_data = this)
    const clap_host_t* host = nullptr;
    const clap_host_latency_t* hostLatency = nullptr;
    const clap_host_params_t* hostParams = nullptr;
    const clap_host_preset_load_t* hostPresetLoad = nullptr;

    P user {};
    double sampleRate = 48000.0;
    uint32_t maxFrames = 0;
    bool prepared = false;
    int cachedLatency = 0;
    int currentChannels = defaultChannelCount<P>();   // selected ports config
    bool offlineRender = false;          // last render-extension setting
    std::atomic<bool> latencyDirty { false };   // audio -> main thread notify

    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<bool>   bypass { false };
    std::atomic<int>    currentProgram { 0 };   // last factory preset loaded
    float bypassMix = 0.0f;
    std::vector<float> dryL, dryR;
    std::vector<float> silence;   // stand-in sidechain when not connected

#if defined(DSPARK_PLUGIN_WEBVIEW)
    // --- editor state + UI -> host event queue (single producer: main thread;
    // single consumer: process() on audio or flush() on main — never both).
    webview_ui::Editor<P> guiEditor;
    bool   guiActive = false;
    double guiScale = 1.0;
    int    guiWidth = 0, guiHeight = 0;
    bool   guiEditActive[kNumParams == 0 ? 1 : kNumParams] {};
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

    /** Applies one factory preset: every parameter, by normalized value. */
    void applyFactoryPresetIdx(int idx) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            if (idx < 0 || idx >= kNumPresets) return;
            for (size_t i = 0; i < kNumParams; ++i)
                applyNormalized(static_cast<int>(i), presetNormalized<P>(idx, i));
            currentProgram.store(idx, std::memory_order_relaxed);
        }
        else
            (void) idx;
    }

    /** Re-reads the plugin latency after parameter motion. The CLAP host
     *  must hear about it on the main thread, so a change only flags it and
     *  requests a callback; sOnMainThread delivers clap_host_latency. */
    void refreshLatency() noexcept
    {
        if constexpr (HasLatency<P>)
        {
            const int now = user.getLatency();
            if (prepared && now != cachedLatency)
            {
                cachedLatency = now;
                latencyDirty.store(true, std::memory_order_release);
                if (host != nullptr && host->request_callback != nullptr)
                    host->request_callback(host);
            }
        }
    }

    /** Translates one CLAP event into the timestamped block stream. CLAP
     *  param events carry PLAIN values; raw MIDI covers what note events
     *  don't (pitch bend, CC, pressure). */
    void collectEvent(const clap_event_header_t* ev, uint32_t numSamples,
                      BlockEvent* events, int& count) noexcept
    {
        if (ev == nullptr || ev->space_id != CLAP_CORE_EVENT_SPACE_ID) return;
        BlockEvent out {};
        out.offset = static_cast<int32_t>(ev->time);
        if (numSamples > 0 && out.offset >= static_cast<int32_t>(numSamples))
            out.offset = static_cast<int32_t>(numSamples) - 1;
        if (out.offset < 0) out.offset = 0;

        if (ev->type == CLAP_EVENT_PARAM_VALUE)
        {
            const auto* pv = reinterpret_cast<const clap_event_param_value_t*>(ev);
            if (pv->param_id == kBypassParamId)
            {
                out.kind = BlockEvent::Kind::Bypass;
                out.value = pv->value;
            }
            else
            {
                const int idx = indexOfParamId(pv->param_id);
                if (idx < 0) return;
                out.kind = BlockEvent::Kind::Param;
                out.paramId = pv->param_id;
                out.value = toNormalized(P::parameters[static_cast<size_t>(idx)],
                                         pv->value);
            }
        }
        else if constexpr (HasMidi<P>)
        {
            if (ev->type == CLAP_EVENT_NOTE_ON || ev->type == CLAP_EVENT_NOTE_OFF)
            {
                const auto* note = reinterpret_cast<const clap_event_note_t*>(ev);
                out.kind = BlockEvent::Kind::Midi;
                out.midi.type = ev->type == CLAP_EVENT_NOTE_ON
                              ? MidiEvent::Type::NoteOn : MidiEvent::Type::NoteOff;
                out.midi.channel = static_cast<uint8_t>(
                    note->channel >= 0 ? note->channel & 0x0F : 0);
                out.midi.note = static_cast<uint8_t>(
                    note->key >= 0 ? note->key & 0x7F : 0);
                out.midi.value = static_cast<float>(note->velocity);
            }
            else if (ev->type == CLAP_EVENT_MIDI)
            {
                const auto* midi = reinterpret_cast<const clap_event_midi_t*>(ev);
                const uint8_t status = midi->data[0] & 0xF0u;
                const uint8_t channel = midi->data[0] & 0x0Fu;
                const uint8_t d1 = midi->data[1] & 0x7Fu;
                const uint8_t d2 = midi->data[2] & 0x7Fu;
                out.kind = BlockEvent::Kind::Midi;
                out.midi.channel = channel;
                switch (status)
                {
                case 0x90:   // wire convention: velocity 0 means note off
                    out.midi.type = d2 > 0 ? MidiEvent::Type::NoteOn
                                           : MidiEvent::Type::NoteOff;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    break;
                case 0x80:
                    out.midi.type = MidiEvent::Type::NoteOff;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    break;
                case 0xA0:
                    out.midi.type = MidiEvent::Type::PolyPressure;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    break;
                case 0xB0:
                    out.midi.type = MidiEvent::Type::ControlChange;
                    out.midi.note = d1;
                    out.midi.value = static_cast<float>(d2) / 127.0f;
                    break;
                case 0xD0:
                    out.midi.type = MidiEvent::Type::ChannelPressure;
                    out.midi.value = static_cast<float>(d1) / 127.0f;
                    break;
                case 0xE0:
                    out.midi.type = MidiEvent::Type::PitchBend;
                    out.midi.value = (static_cast<float>((d2 << 7) | d1) - 8192.0f)
                                   / 8192.0f;
                    break;
                default:
                    return;
                }
            }
            else
                return;
        }
        else
            return;

        if (count < kMaxBlockEvents) events[count++] = out;
        else events[kMaxBlockEvents - 1] = out;
    }

    void collectEvents(const clap_input_events_t* in, uint32_t numSamples,
                       BlockEvent* events, int& count) noexcept
    {
        if (in == nullptr) return;
        const uint32_t n = in->size(in);
        for (uint32_t i = 0; i < n; ++i)
            collectEvent(in->get(in, i), numSamples, events, count);
    }

    /** Applies one event; @p blockStart rebases MIDI offsets onto the next
     *  processBlock call. Returns true when a user parameter changed. */
    bool applyBlockEvent(const BlockEvent& ev, int blockStart) noexcept
    {
        switch (ev.kind)
        {
        case BlockEvent::Kind::Bypass:
            bypass.store(ev.value >= 0.5, std::memory_order_relaxed);
            return false;
        case BlockEvent::Kind::Midi:
            if constexpr (HasMidi<P>)
            {
                MidiEvent midi = ev.midi;
                midi.sampleOffset = ev.offset - blockStart;
                if (midi.sampleOffset < 0) midi.sampleOffset = 0;
                user.handleMidiEvent(midi);
            }
            return false;
        case BlockEvent::Kind::Param:
        default:
            if (const int idx = indexOfParamId(ev.paramId); idx >= 0)
            {
                applyNormalized(idx, ev.value);
                return true;
            }
            return false;
        }
    }

    /** Forwards the per-block clap transport as a TransportInfo. */
    void forwardTransport(const clap_event_transport_t* t) noexcept
    {
        if constexpr (HasTransport<P>)
        {
            if (t == nullptr) return;
            constexpr double kBeatFactor = static_cast<double>(CLAP_BEATTIME_FACTOR);
            TransportInfo info {};
            info.playing   = (t->flags & CLAP_TRANSPORT_IS_PLAYING) != 0;
            info.recording = (t->flags & CLAP_TRANSPORT_IS_RECORDING) != 0;
            info.looping   = (t->flags & CLAP_TRANSPORT_IS_LOOP_ACTIVE) != 0;
            if ((t->flags & CLAP_TRANSPORT_HAS_TEMPO) != 0)
            {
                info.tempoBpm = t->tempo;
                info.tempoValid = true;
            }
            if ((t->flags & CLAP_TRANSPORT_HAS_BEATS_TIMELINE) != 0)
            {
                info.ppqPosition = static_cast<double>(t->song_pos_beats) / kBeatFactor;
                info.barStartPpq = static_cast<double>(t->bar_start) / kBeatFactor;
                info.positionValid = true;
                info.loopStartPpq =
                    static_cast<double>(t->loop_start_beats) / kBeatFactor;
                info.loopEndPpq = static_cast<double>(t->loop_end_beats) / kBeatFactor;
                info.loopValid = true;
            }
            if ((t->flags & CLAP_TRANSPORT_HAS_TIME_SIGNATURE) != 0)
            {
                info.timeSigNumerator = t->tsig_num;
                info.timeSigDenominator = t->tsig_denom;
                info.timeSigValid = true;
            }
            user.setTransport(info);
        }
        else
            (void) t;
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
        auto* s = self(p);
        if (s->host != nullptr)
        {
            s->hostParams = static_cast<const clap_host_params_t*>(
                s->host->get_extension(s->host, CLAP_EXT_PARAMS));
            s->hostLatency = static_cast<const clap_host_latency_t*>(
                s->host->get_extension(s->host, CLAP_EXT_LATENCY));
            if constexpr (kNumPresets > 0)
                s->hostPresetLoad = static_cast<const clap_host_preset_load_t*>(
                    s->host->get_extension(s->host, CLAP_EXT_PRESET_LOAD));
        }
        return true;
    }

    static void sDestroy(const clap_plugin_t* p) noexcept { delete self(p); }

    static bool sActivate(const clap_plugin_t* p, double sr,
                          uint32_t, uint32_t maxFrames) noexcept
    {
        auto* s = self(p);
        s->sampleRate = sr;
        s->maxFrames = maxFrames;
        dspark::AudioSpec spec { sr, static_cast<int>(maxFrames), s->currentChannels };
        s->user.prepare(spec);
        if constexpr (HasOfflineMode<P>)
            s->user.setOfflineRendering(s->offlineRender);
        for (size_t i = 0; i < kNumParams; ++i)
            s->user.setParameter(static_cast<int>(i),
                static_cast<float>(toPlain(P::parameters[i],
                                           s->shadow[i].load(std::memory_order_relaxed))));
        s->dryL.assign(maxFrames, 0.0f);
        s->dryR.assign(maxFrames, 0.0f);
        if constexpr (HasSidechain<P>)
            s->silence.assign(maxFrames, 0.0f);
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

        // FTZ/DAZ for the whole callback: DSPark processors guard their own
        // hot loops, this covers user DSP in hosts that do not set it.
        dspark::DenormalGuard denormalGuard;

#if defined(DSPARK_PLUGIN_WEBVIEW)
        s->drainUiEvents(process->out_events);
#endif

        const uint32_t n = process->frames_count;

        BlockEvent events[kMaxBlockEvents];
        int eventCount = 0;
        s->collectEvents(process->in_events, n, events, eventCount);
        sortBlockEvents(events, eventCount);

        bool paramsChanged = false;

        const bool canRender = n > 0 && s->prepared
            && process->audio_outputs_count >= 1
            && process->audio_outputs[0].data32 != nullptr
            && process->audio_outputs[0].channel_count >= 1;
        if (!canRender)
        {
            for (int i = 0; i < eventCount; ++i)
                paramsChanged |= s->applyBlockEvent(events[i], events[i].offset);
            if (paramsChanged) s->refreshLatency();
            return CLAP_PROCESS_CONTINUE;
        }

        s->forwardTransport(process->transport);

        auto& outPort = process->audio_outputs[0];
        float** out = outPort.data32;
        const uint32_t width = static_cast<uint32_t>(s->currentChannels);
        const uint32_t nCh = outPort.channel_count < width
                           ? outPort.channel_count : width;

        const bool haveIn = !kIsInstrument && process->audio_inputs_count >= 1
                         && process->audio_inputs[0].data32 != nullptr;
        float** in = haveIn ? process->audio_inputs[0].data32 : nullptr;

        if (n > s->dryL.size()) return CLAP_PROCESS_CONTINUE;   // oversize block

        // Dry copy for the bypass blend; instruments start cleared (voices
        // ADD) and bypass toward silence (their dry vectors stay zero).
        float* dry[2] = { s->dryL.data(), s->dryR.data() };
        const size_t bytes = sizeof(float) * n;
        for (uint32_t ch = 0; ch < nCh; ++ch)
        {
            if (kIsInstrument)
            {
                std::memset(out[ch], 0, bytes);
                continue;
            }
            const float* src = (haveIn && in[ch] != nullptr) ? in[ch] : out[ch];
            std::memcpy(dry[ch], src, bytes);
            if (out[ch] != src)
                std::memcpy(out[ch], src, bytes);
        }

        // Sidechain: input port 1, pre-allocated silence when not routed.
        float* scPtrs[2] = { nullptr, nullptr };
        if constexpr (HasSidechain<P>)
        {
            if (n > s->silence.size()) return CLAP_PROCESS_CONTINUE;
            scPtrs[0] = scPtrs[1] = s->silence.data();
            if (process->audio_inputs_count >= 2
                && process->audio_inputs[1].data32 != nullptr)
            {
                const auto& scPort = process->audio_inputs[1];
                const uint32_t scCh = scPort.channel_count < 2
                                    ? scPort.channel_count : 2;
                for (uint32_t ch = 0; ch < scCh; ++ch)
                    if (scPort.data32[ch] != nullptr)
                        scPtrs[ch] = scPort.data32[ch];
                if (scCh == 1 && scPort.data32[0] != nullptr)
                    scPtrs[1] = scPort.data32[0];   // mono key feeds both ears
            }
        }

        // Sub-block processing at quantum-aligned event positions (the
        // sample-accurate default); opted out, everything applies up front.
        auto processSegment = [&](int start, int length) noexcept {
            float* sub[2] = { out[0] + start,
                              nCh > 1 ? out[1] + start : out[0] + start };
            dspark::AudioBufferView<float> view(sub, static_cast<int>(nCh), length);
            if constexpr (HasSidechain<P>)
            {
                // The key view mirrors the main width (mono main, mono key).
                float* scSub[2] = { scPtrs[0] + start, scPtrs[1] + start };
                dspark::AudioBufferView<float> scView(scSub,
                                                      static_cast<int>(nCh), length);
                s->user.processBlock(view, scView);
            }
            else
                s->user.processBlock(view);
        };

        const int total = static_cast<int>(n);
        int evIdx = 0;
        if (!sampleAccurateOf<P>())
        {
            for (; evIdx < eventCount; ++evIdx)
                paramsChanged |= s->applyBlockEvent(events[evIdx], 0);
            processSegment(0, total);
        }
        else
        {
            int pos = 0;
            while (pos < total)
            {
                while (evIdx < eventCount
                       && (events[evIdx].offset / kAutomationQuantum)
                              * kAutomationQuantum <= pos)
                    paramsChanged |= s->applyBlockEvent(events[evIdx++], pos);
                int next = total;
                if (evIdx < eventCount)
                {
                    const int snapped = (events[evIdx].offset / kAutomationQuantum)
                                      * kAutomationQuantum;
                    if (snapped < next) next = snapped;
                }
                if (next <= pos) next = pos + kAutomationQuantum < total
                                      ? pos + kAutomationQuantum : total;
                processSegment(pos, next - pos);
                pos = next;
            }
            for (; evIdx < eventCount; ++evIdx)
                paramsChanged |= s->applyBlockEvent(events[evIdx], total);
        }

        const float target = s->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (s->bypassMix != target || target > 0.0f)
        {
            const float step = 1.0f / static_cast<float>(kBypassRampSamples);
            float mix = s->bypassMix;
            for (uint32_t i = 0; i < n; ++i)
            {
                mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
                mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
                for (uint32_t ch = 0; ch < nCh; ++ch)
                    out[ch][i] += (dry[ch][i] - out[ch][i]) * mix;
            }
            s->bypassMix = mix;
        }

        if (paramsChanged) s->refreshLatency();
        return CLAP_PROCESS_CONTINUE;
    }

    static const void* sGetExtension(const clap_plugin_t*, const char* id) noexcept
    {
        if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS) == 0) return &kAudioPorts;
        if (std::strcmp(id, CLAP_EXT_PARAMS) == 0)      return &kParams;
        if (std::strcmp(id, CLAP_EXT_STATE) == 0)       return &kState;
        if (std::strcmp(id, CLAP_EXT_LATENCY) == 0)     return &kLatency;
        if (std::strcmp(id, CLAP_EXT_TAIL) == 0)        return &kTail;
        if (channelSupportOf<P>() == ChannelSupport::MonoAndStereo)
        {
            if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS_CONFIG) == 0)
                return &kPortsConfig;
            if (std::strcmp(id, CLAP_EXT_AUDIO_PORTS_CONFIG_INFO) == 0
                || std::strcmp(id, CLAP_EXT_AUDIO_PORTS_CONFIG_INFO_COMPAT) == 0)
                return &kPortsConfigInfo;
        }
        if constexpr (HasMidi<P>)
            if (std::strcmp(id, CLAP_EXT_NOTE_PORTS) == 0) return &kNotePorts;
        if constexpr (HasOfflineMode<P>)
            if (std::strcmp(id, CLAP_EXT_RENDER) == 0) return &kRender;
        if constexpr (kNumPresets > 0)
            if (std::strcmp(id, CLAP_EXT_PRESET_LOAD) == 0
                || std::strcmp(id, CLAP_EXT_PRESET_LOAD_COMPAT) == 0)
                return &kPresetLoad;
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

    static void sOnMainThread(const clap_plugin_t* p) noexcept
    {
        // Deferred latency notification (the audio thread only flags it).
        auto* s = self(p);
        if (s->latencyDirty.exchange(false, std::memory_order_acq_rel)
            && s->hostLatency != nullptr && s->hostLatency->changed != nullptr)
            s->hostLatency->changed(s->host);
    }

    // --- ext: audio ports ---------------------------------------------------------

    static uint32_t sPortCount(const clap_plugin_t*, bool isInput) noexcept
    {
        if (isInput)
        {
            if (kIsInstrument) return 0;
            return HasSidechain<P> ? 2 : 1;
        }
        return 1;
    }

    static bool sPortGet(const clap_plugin_t* p, uint32_t index, bool isInput,
                         clap_audio_port_info_t* info) noexcept
    {
        if (info == nullptr || index >= sPortCount(p, isInput)) return false;
        const int width = self(p)->currentChannels;
        std::memset(info, 0, sizeof(*info));
        if (isInput && index == 1)   // the sidechain: routable, never the main pair
        {
            info->id = 2;
            std::snprintf(info->name, sizeof(info->name), "Sidechain");
            info->flags = 0;
            info->channel_count = static_cast<uint32_t>(width);
            info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
            info->in_place_pair = CLAP_INVALID_ID;
            return true;
        }
        info->id = isInput ? 0 : 1;
        std::snprintf(info->name, sizeof(info->name), "%s",
                      isInput ? "Input" : "Output");
        info->flags = CLAP_AUDIO_PORT_IS_MAIN;
        info->channel_count = static_cast<uint32_t>(width);
        info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
        info->in_place_pair = kIsInstrument ? CLAP_INVALID_ID
                                            : (isInput ? 1u : 0u);
        return true;
    }

    inline static const clap_plugin_audio_ports_t kAudioPorts = { &sPortCount, &sPortGet };

    // --- ext: audio ports config (mono / stereo selection) --------------------------

    static uint32_t sConfigCount(const clap_plugin_t*) noexcept { return 2; }

    static void fillConfig(clap_audio_ports_config_t* config, int width) noexcept
    {
        std::memset(config, 0, sizeof(*config));
        config->id = static_cast<clap_id>(width);
        std::snprintf(config->name, sizeof(config->name), "%s",
                      width == 1 ? "Mono" : "Stereo");
        config->input_port_count = sPortCount(nullptr, true);
        config->output_port_count = 1;
        config->has_main_input = !kIsInstrument;
        config->main_input_channel_count = static_cast<uint32_t>(width);
        config->main_input_port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
        config->has_main_output = true;
        config->main_output_channel_count = static_cast<uint32_t>(width);
        config->main_output_port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
    }

    static bool sConfigGet(const clap_plugin_t*, uint32_t index,
                           clap_audio_ports_config_t* config) noexcept
    {
        if (config == nullptr || index >= 2) return false;
        fillConfig(config, index == 0 ? 1 : 2);
        return true;
    }

    static bool sConfigSelect(const clap_plugin_t* p, clap_id configId) noexcept
    {
        const int width = static_cast<int>(configId);
        if (!supportsChannelCount<P>(width)) return false;
        self(p)->currentChannels = width;
        return true;
    }

    inline static const clap_plugin_audio_ports_config_t kPortsConfig = {
        &sConfigCount, &sConfigGet, &sConfigSelect
    };

    static clap_id sConfigCurrent(const clap_plugin_t* p) noexcept
    {
        return static_cast<clap_id>(self(p)->currentChannels);
    }

    static bool sConfigInfoGet(const clap_plugin_t* p, clap_id configId,
                               uint32_t portIndex, bool isInput,
                               clap_audio_port_info_t* info) noexcept
    {
        const int width = static_cast<int>(configId);
        if (info == nullptr || !supportsChannelCount<P>(width)
            || portIndex >= sPortCount(p, isInput))
            return false;
        // Same shape as sPortGet, with the width of the asked-about config.
        std::memset(info, 0, sizeof(*info));
        if (isInput && portIndex == 1)
        {
            info->id = 2;
            std::snprintf(info->name, sizeof(info->name), "Sidechain");
            info->channel_count = static_cast<uint32_t>(width);
            info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
            info->in_place_pair = CLAP_INVALID_ID;
            return true;
        }
        info->id = isInput ? 0 : 1;
        std::snprintf(info->name, sizeof(info->name), "%s",
                      isInput ? "Input" : "Output");
        info->flags = CLAP_AUDIO_PORT_IS_MAIN;
        info->channel_count = static_cast<uint32_t>(width);
        info->port_type = width == 1 ? CLAP_PORT_MONO : CLAP_PORT_STEREO;
        info->in_place_pair = kIsInstrument ? CLAP_INVALID_ID
                                            : (isInput ? 1u : 0u);
        return true;
    }

    inline static const clap_plugin_audio_ports_config_info_t kPortsConfigInfo = {
        &sConfigCurrent, &sConfigInfoGet
    };

    // --- ext: note ports (HasMidi) ----------------------------------------------------

    static uint32_t sNotePortCount(const clap_plugin_t*, bool isInput) noexcept
    {
        return (HasMidi<P> && isInput) ? 1 : 0;
    }

    static bool sNotePortGet(const clap_plugin_t*, uint32_t index, bool isInput,
                             clap_note_port_info_t* info) noexcept
    {
        if (info == nullptr || !isInput || index != 0 || !HasMidi<P>) return false;
        std::memset(info, 0, sizeof(*info));
        info->id = 10;
        info->supported_dialects = CLAP_NOTE_DIALECT_CLAP | CLAP_NOTE_DIALECT_MIDI;
        info->preferred_dialect = CLAP_NOTE_DIALECT_CLAP;
        std::snprintf(info->name, sizeof(info->name), "MIDI In");
        return true;
    }

    inline static const clap_plugin_note_ports_t kNotePorts = {
        &sNotePortCount, &sNotePortGet
    };

    // --- ext: render (HasOfflineMode) ---------------------------------------------------

    static bool sRenderHardRealtime(const clap_plugin_t*) noexcept { return false; }

    static bool sRenderSet(const clap_plugin_t* p, clap_plugin_render_mode mode) noexcept
    {
        auto* s = self(p);
        s->offlineRender = mode == CLAP_RENDER_OFFLINE;
        if constexpr (HasOfflineMode<P>)
            s->user.setOfflineRendering(s->offlineRender);
        return true;
    }

    inline static const clap_plugin_render_t kRender = {
        &sRenderHardRealtime, &sRenderSet
    };

    // --- ext: preset load (factory presets; the DSO is the container) -------------------

    static bool sPresetFromLocation(const clap_plugin_t* p, uint32_t locationKind,
                                    const char* location, const char* loadKey) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            auto* s = self(p);
            if (locationKind != CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN
                || loadKey == nullptr)
                return false;
            const int idx = std::atoi(loadKey);
            if (idx < 0 || idx >= kNumPresets) return false;
            s->applyFactoryPresetIdx(idx);
            s->refreshLatency();
            if (s->hostParams != nullptr && s->hostParams->rescan != nullptr)
                s->hostParams->rescan(s->host, CLAP_PARAM_RESCAN_VALUES);
            if (s->hostPresetLoad != nullptr && s->hostPresetLoad->loaded != nullptr)
                s->hostPresetLoad->loaded(s->host, locationKind, location, loadKey);
            return true;
        }
        else
        {
            (void) p;
            (void) locationKind;
            (void) location;
            (void) loadKey;
            return false;
        }
    }

    inline static const clap_plugin_preset_load_t kPresetLoad = {
        &sPresetFromLocation
    };

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
        auto* s = self(p);
        BlockEvent events[kMaxBlockEvents];
        int eventCount = 0;
        s->collectEvents(in, 0, events, eventCount);
        sortBlockEvents(events, eventCount);
        bool paramsChanged = false;
        for (int i = 0; i < eventCount; ++i)
            paramsChanged |= s->applyBlockEvent(events[i], events[i].offset);
        if (paramsChanged) s->refreshLatency();
#if defined(DSPARK_PLUGIN_WEBVIEW)
        s->drainUiEvents(out);
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
        const std::vector<uint8_t> blob = buildState(
            s->user, norm, kNumParams,
            kNumPresets > 0 ? s->currentProgram.load(std::memory_order_relaxed) : -1);
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
        int program = -1;
        if (!applyState(s->user, blob.data(), blob.size(), norm, &program))
            return false;
        for (size_t i = 0; i < kNumParams; ++i)
            s->applyNormalized(static_cast<int>(i), norm[i]);
        if (kNumPresets > 0 && program >= 0 && program < kNumPresets)
            s->currentProgram.store(program, std::memory_order_relaxed);
        s->refreshLatency();   // restored state may imply a new lookahead
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
            kIsInstrument ? CLAP_PLUGIN_FEATURE_INSTRUMENT
                          : CLAP_PLUGIN_FEATURE_AUDIO_EFFECT,
            channelSupportOf<P>() == ChannelSupport::MonoOnly
                ? CLAP_PLUGIN_FEATURE_MONO : CLAP_PLUGIN_FEATURE_STEREO,
            nullptr
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

    // --- preset discovery (factory presets live inside the plugin DSO) -------------
    //
    // The provider declares ONE location of kind PLUGIN and reports each
    // factory preset with its table index as the load key; preset-load's
    // from_location() applies it. Hosts with a preset browser (e.g. Bitwig)
    // index these without instantiating the plugin.

    static constexpr int kNumPresets = factoryPresetCountOf<P>();

    static const char* providerId() noexcept
    {
        // Magic static: the discovery factory contract is [thread-safe].
        static const std::array<char, 256> id = [] {
            std::array<char, 256> buf {};
            std::snprintf(buf.data(), buf.size(), "%s.presets",
                          P::descriptor.productId);
            return buf;
        }();
        return id.data();
    }

    static const clap_preset_discovery_provider_descriptor_t* providerDescriptor() noexcept
    {
        static const clap_preset_discovery_provider_descriptor_t desc = {
            CLAP_VERSION_INIT, providerId(), "DSPark Factory Presets",
            P::descriptor.vendor
        };
        return &desc;
    }

    struct ProviderState
    {
        clap_preset_discovery_provider_t provider {};
        const clap_preset_discovery_indexer_t* indexer = nullptr;
    };

    static bool sProviderInit(const clap_preset_discovery_provider_t* provider) noexcept
    {
        const auto* state = static_cast<const ProviderState*>(provider->provider_data);
        if (state->indexer == nullptr || state->indexer->declare_location == nullptr)
            return false;
        clap_preset_discovery_location_t location {};
        location.flags = CLAP_PRESET_DISCOVERY_IS_FACTORY_CONTENT;
        location.name = "Factory Presets";
        location.kind = CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN;
        location.location = nullptr;
        return state->indexer->declare_location(state->indexer, &location);
    }

    static void sProviderDestroy(const clap_preset_discovery_provider_t* provider) noexcept
    {
        delete static_cast<ProviderState*>(provider->provider_data);
    }

    static bool sProviderGetMetadata(const clap_preset_discovery_provider_t*,
                                     uint32_t locationKind, const char*,
                                     const clap_preset_discovery_metadata_receiver_t*
                                         receiver) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            if (locationKind != CLAP_PRESET_DISCOVERY_LOCATION_PLUGIN
                || receiver == nullptr)
                return false;
            const clap_universal_plugin_id_t pluginId {
                "clap", P::descriptor.productId
            };
            for (int i = 0; i < kNumPresets; ++i)
            {
                char key[16];
                std::snprintf(key, sizeof(key), "%d", i);
                if (!receiver->begin_preset(receiver,
                        P::factoryPresets[static_cast<size_t>(i)].name, key))
                    return true;   // the indexer asked to stop; not an error
                receiver->add_plugin_id(receiver, &pluginId);
                receiver->set_flags(receiver,
                                    CLAP_PRESET_DISCOVERY_IS_FACTORY_CONTENT);
            }
            return true;
        }
        else
        {
            (void) locationKind;
            (void) receiver;
            return false;
        }
    }

    static const void* sProviderGetExtension(const clap_preset_discovery_provider_t*,
                                             const char*) noexcept
    { return nullptr; }

    static uint32_t sDiscoveryCount(const clap_preset_discovery_factory_t*) noexcept
    { return kNumPresets > 0 ? 1u : 0u; }

    static const clap_preset_discovery_provider_descriptor_t* sDiscoveryDescriptor(
        const clap_preset_discovery_factory_t*, uint32_t index) noexcept
    {
        return (kNumPresets > 0 && index == 0) ? providerDescriptor() : nullptr;
    }

    static const clap_preset_discovery_provider_t* sDiscoveryCreate(
        const clap_preset_discovery_factory_t*,
        const clap_preset_discovery_indexer_t* indexer,
        const char* providerIdAsked) noexcept
    {
        if (kNumPresets == 0 || indexer == nullptr || providerIdAsked == nullptr
            || std::strcmp(providerIdAsked, providerId()) != 0)
            return nullptr;
        auto* state = new (std::nothrow) ProviderState();
        if (state == nullptr) return nullptr;
        state->indexer = indexer;
        state->provider.desc = providerDescriptor();
        state->provider.provider_data = state;
        state->provider.init = &sProviderInit;
        state->provider.destroy = &sProviderDestroy;
        state->provider.get_metadata = &sProviderGetMetadata;
        state->provider.get_extension = &sProviderGetExtension;
        return &state->provider;
    }

    inline static const clap_preset_discovery_factory_t kDiscoveryFactory = {
        &sDiscoveryCount, &sDiscoveryDescriptor, &sDiscoveryCreate
    };

    static bool sEntryInit(const char*) noexcept { return true; }
    static void sEntryDeinit() noexcept {}

    static const void* sEntryGetFactory(const char* factoryId) noexcept
    {
        if (factoryId == nullptr) return nullptr;
        if (std::strcmp(factoryId, CLAP_PLUGIN_FACTORY_ID) == 0)
            return &kFactory;
        if (kNumPresets > 0
            && (std::strcmp(factoryId, CLAP_PRESET_DISCOVERY_FACTORY_ID) == 0
                || std::strcmp(factoryId, CLAP_PRESET_DISCOVERY_FACTORY_ID_COMPAT) == 0))
            return &kDiscoveryFactory;
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
