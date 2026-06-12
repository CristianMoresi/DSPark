// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkVst3.h
 * @brief Native VST3 backend: a complete plugin from one DSPark plugin class.
 *
 * Implements the VST3 COM ABI directly against Steinberg's official C API
 * (vst3_c_api.h, vendored next to this header under Steinberg's permissive
 * 2025 license) — no VST3 C++ SDK to download, no build system beyond your
 * compiler:
 *
 * ```cpp
 * #include "plugin/vst3/DSParkVst3.h"
 * struct MyPlugin { ... };               // see plugin/DSParkPlugin.h
 * DSPARK_VST3_PLUGIN(MyPlugin)
 * ```
 * cl /std:c++20 /O2 /LD myplugin.cpp /Fe:MyPlugin.vst3        (Windows)
 * g++/clang++ -std=c++20 -O2 -fPIC -shared -o MyPlugin.so ... (Linux/macOS)
 *
 * Design notes:
 * - **Single-component plugin**: one object implements IComponent,
 *   IAudioProcessor and IEditController (the layout hosts use for the vast
 *   majority of plugins); the COM "lenses" are consecutive vtable pointers
 *   and queryInterface hands out the right one. IUnitInfo (factory presets)
 *   and IMidiMapping (MIDI plugins) ride two more lenses, surfaced only
 *   when the plugin class declares the matching capability.
 * - **Parameters**: host ParamIDs are FNV-1a hashes of the stable text ids
 *   (never indices: reordering parameters between versions must not break
 *   automation). A wrapper-owned Bypass parameter (kIsBypass) is always
 *   appended and applied as a short crossfade against the dry input.
 * - **Automation is sample-accurate by default**: every queue point becomes
 *   a timestamped event and processing splits at kAutomationQuantum
 *   boundaries (plugins may opt out, see DSParkPlugin.h). DSPark setters
 *   still smooth internally, so both modes are click-free.
 * - **Buses**: mono and stereo are negotiated through setBusArrangements
 *   per the declared ChannelSupport; a sidechain aux input follows the main
 *   width. Category::Instrument drops the audio inputs entirely and
 *   HasMidi adds an event input bus (notes natively; pitch bend / mod /
 *   sustain / channel pressure through IMidiMapping proxy parameters, the
 *   VST3 scheme for MIDI controllers).
 * - **Transport**: when the class implements setTransport, the wrapper
 *   declares its needs through IProcessContextRequirements and forwards the
 *   host ProcessContext once per block.
 * - **Latency** is re-read after parameter changes; a change notifies the
 *   host through restartComponent(kLatencyChanged).
 * - **State** uses the format-agnostic container of DSParkPlugin.h, so
 *   presets are byte-identical across the VST3/CLAP/AU backends. Factory
 *   presets publish as a program list with a program-change parameter.
 * - **Threading**: setParamNormalized arrives from the UI thread and
 *   process() from the audio thread; both funnel into the user's
 *   setParameter, which DSPark's contract requires to be atomic-based.
 *   process() runs under a DenormalGuard (FTZ/DAZ) so user DSP outside
 *   DSPark's own classes is covered in hosts that do not set it.
 * - **Editor**: when plugin/webview/DSParkWebViewEditor.h is included before
 *   this header and the class declares `hasEditor = true`, createView returns
 *   an IPlugView embedding the WebView editor (with content-scale support);
 *   otherwise hosts show their generic parameter UI.
 */

#define DSPARK_PLUGIN_VST3_INCLUDED 1

#include "../DSParkPlugin.h"

#include "vst3_c_api.h"

#include <atomic>
#include <cstring>
#include <new>

namespace dspark::plugin::vst3 {

// -- small helpers -------------------------------------------------------------

inline bool tuidEqual(const Steinberg_TUID a, const Steinberg_TUID b) noexcept
{
    return std::memcmp(a, b, sizeof(Steinberg_TUID)) == 0;
}

inline void asciiToString128(const char* src, Steinberg_Vst_String128 dst) noexcept
{
    int i = 0;
    for (; i < 127 && src[i] != '\0'; ++i)
        dst[i] = static_cast<Steinberg_char16>(static_cast<unsigned char>(src[i]));
    dst[i] = 0;
}

inline void copyAscii(const char* src, char* dst, size_t cap) noexcept
{
    std::snprintf(dst, cap, "%s", src ? src : "");
}

/// Reads the full requested range from an IBStream (hosts may chunk).
inline bool streamRead(Steinberg_IBStream* s, void* dst, Steinberg_int32 bytes) noexcept
{
    auto* p = static_cast<char*>(dst);
    while (bytes > 0)
    {
        Steinberg_int32 got = 0;
        if (s->lpVtbl->read(s, p, bytes, &got) != Steinberg_kResultOk || got <= 0)
            return false;
        p += got;
        bytes -= got;
    }
    return true;
}

inline bool streamWrite(Steinberg_IBStream* s, const void* src, Steinberg_int32 bytes) noexcept
{
    auto* p = static_cast<const char*>(src);
    while (bytes > 0)
    {
        Steinberg_int32 put = 0;
        if (s->lpVtbl->write(s, const_cast<char*>(p), bytes, &put) != Steinberg_kResultOk
            || put <= 0)
            return false;
        p += put;
        bytes -= put;
    }
    return true;
}

// -- the plugin object ----------------------------------------------------------

inline constexpr uint32_t kBypassParamId  = 0x42595053u;   // 'BYPS'
inline constexpr uint32_t kProgramParamId = 0x5052474Du;   // 'PRGM'
inline constexpr int      kBypassRampSamples = 256;
inline constexpr Steinberg_int32 kPresetProgramListId = 1;

// MIDI controllers travel as proxy parameters in VST3 (the IMidiMapping
// scheme): one hidden ParamID per (channel, controller). Four controllers
// cover playable MIDI: mod wheel, sustain, channel pressure, pitch bend.
inline constexpr uint32_t kMidiProxyBase = 0x4D440000u;    // 'MD' marker
inline constexpr int      kMidiProxyControllers[4] = { 1, 64, 128, 129 };
inline constexpr int      kNumMidiProxies = 16 * 4;        // channels x controllers

inline constexpr bool isMidiProxyId(uint32_t id) noexcept
{
    if ((id & 0xFFFF0000u) != kMidiProxyBase) return false;
    const uint32_t ctrl = id & 0xFFu;
    const uint32_t channel = (id >> 8) & 0xFFu;
    return channel < 16
        && (ctrl == 1 || ctrl == 64 || ctrl == 128 || ctrl == 129);
}

inline constexpr uint32_t midiProxyId(int channel, int controller) noexcept
{
    return kMidiProxyBase | (static_cast<uint32_t>(channel) << 8)
         | static_cast<uint32_t>(controller);
}

/** @brief Proxy parameter (normalized) -> the framework-neutral MidiEvent. */
inline MidiEvent midiProxyToEvent(uint32_t id, double normalized, int offset) noexcept
{
    MidiEvent ev {};
    ev.channel = static_cast<uint8_t>((id >> 8) & 0x0Fu);
    ev.sampleOffset = offset;
    const uint32_t ctrl = id & 0xFFu;
    if (ctrl == 129)
    {
        ev.type = MidiEvent::Type::PitchBend;
        ev.value = static_cast<float>(normalized * 2.0 - 1.0);
    }
    else if (ctrl == 128)
    {
        ev.type = MidiEvent::Type::ChannelPressure;
        ev.value = static_cast<float>(normalized);
    }
    else
    {
        ev.type = MidiEvent::Type::ControlChange;
        ev.note = static_cast<uint8_t>(ctrl);
        ev.value = static_cast<float>(normalized);
    }
    return ev;
}

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

    // COM lenses — consecutive vtable pointers; queryInterface returns the
    // address of the matching slot. Standard layout is asserted below. The
    // last two lenses surface only for the matching capability.
    const Steinberg_Vst_IComponentVtbl*       componentVtbl;
    const Steinberg_Vst_IAudioProcessorVtbl*  processorVtbl;
    const Steinberg_Vst_IEditControllerVtbl*  controllerVtbl;
    const Steinberg_Vst_IProcessContextRequirementsVtbl* contextReqVtbl;
    const Steinberg_Vst_IUnitInfoVtbl*        unitVtbl;
    const Steinberg_Vst_IMidiMappingVtbl*     midiMapVtbl;

    std::atomic<Steinberg_uint32> refs { 1 };

    P user {};
    Steinberg_Vst_ProcessSetup setup { 0, 0, 0, 48000.0 };
    bool prepared = false;
    int  cachedLatency = 0;
    int  currentChannels = defaultChannelCount<P>();   // negotiated bus width

    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<double> midiProxyShadow[HasMidi<P> ? kNumMidiProxies : 1] {};
    std::atomic<bool>   bypass { false };
    std::atomic<int>    currentProgram { 0 };
    float bypassMix = 0.0f;            // audio-thread crossfade state
    std::vector<float> dryL, dryR;     // pre-process copy for the bypass blend
    std::vector<float> silence;        // stand-in sidechain when not connected

    Steinberg_Vst_IComponentHandler* handler = nullptr;

    Plugin() noexcept
    {
        componentVtbl  = &kComponentVtbl;
        processorVtbl  = &kProcessorVtbl;
        controllerVtbl = &kControllerVtbl;
        contextReqVtbl = &kContextReqVtbl;
        unitVtbl       = &kUnitVtbl;
        midiMapVtbl    = &kMidiMapVtbl;
        for (size_t i = 0; i < kNumParams; ++i)
            shadow[i].store(toNormalized(P::parameters[i], P::parameters[i].defValue),
                            std::memory_order_relaxed);
    }

    // --- lens recovery ---------------------------------------------------------

    static Plugin* fromLens(void* iface, int lens) noexcept
    {
        return reinterpret_cast<Plugin*>(static_cast<char*>(iface)
                                         - static_cast<ptrdiff_t>(lens) * sizeof(void*));
    }

    void* lensPtr(int lens) noexcept
    {
        return reinterpret_cast<char*>(this) + static_cast<ptrdiff_t>(lens) * sizeof(void*);
    }

    // --- shared FUnknown --------------------------------------------------------

    Steinberg_tresult query(const Steinberg_TUID iid, void** obj) noexcept
    {
        if (obj == nullptr) return Steinberg_kInvalidArgument;
        *obj = nullptr;
        if (tuidEqual(iid, Steinberg_FUnknown_iid)
            || tuidEqual(iid, Steinberg_IPluginBase_iid)
            || tuidEqual(iid, Steinberg_Vst_IComponent_iid))
            *obj = lensPtr(0);
        else if (tuidEqual(iid, Steinberg_Vst_IAudioProcessor_iid))
            *obj = lensPtr(1);
        else if (tuidEqual(iid, Steinberg_Vst_IEditController_iid))
            *obj = lensPtr(2);
        else if (tuidEqual(iid, Steinberg_Vst_IProcessContextRequirements_iid))
            *obj = lensPtr(3);
        else if (kNumPresets > 0 && tuidEqual(iid, Steinberg_Vst_IUnitInfo_iid))
            *obj = lensPtr(4);
        else if (HasMidi<P> && tuidEqual(iid, Steinberg_Vst_IMidiMapping_iid))
            *obj = lensPtr(5);
        if (*obj == nullptr) return Steinberg_kNoInterface;
        refs.fetch_add(1, std::memory_order_relaxed);
        return Steinberg_kResultOk;
    }

    Steinberg_uint32 addRefImpl() noexcept
    {
        return refs.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    Steinberg_uint32 releaseImpl() noexcept
    {
        const Steinberg_uint32 left = refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (left == 0) delete this;
        return left;
    }

    template <int Lens>
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sQuery(void* self_,
                                                           const Steinberg_TUID iid,
                                                           void** obj)
    { return fromLens(self_, Lens)->query(iid, obj); }

    template <int Lens>
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sAddRef(void* self_)
    { return fromLens(self_, Lens)->addRefImpl(); }

    template <int Lens>
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sRelease(void* self_)
    { return fromLens(self_, Lens)->releaseImpl(); }

    // --- parameter plumbing ------------------------------------------------------

    static int indexOfParamId(Steinberg_Vst_ParamID id) noexcept
    {
        for (size_t i = 0; i < kNumParams; ++i)
            if (hash32(P::parameters[i].id) == id) return static_cast<int>(i);
        return -1;
    }

    void applyNormalized(int index, double normalized) noexcept
    {
        const auto& spec = P::parameters[static_cast<size_t>(index)];
        shadow[static_cast<size_t>(index)].store(normalized, std::memory_order_relaxed);
        user.setParameter(index, static_cast<float>(toPlain(spec, normalized)));
    }

    void applyAllShadows() noexcept
    {
        for (size_t i = 0; i < kNumParams; ++i)
            user.setParameter(static_cast<int>(i),
                static_cast<float>(toPlain(P::parameters[i],
                                           shadow[i].load(std::memory_order_relaxed))));
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

    /** Re-reads the plugin latency; on a change, updates the cache and asks
     *  the host to re-fetch it. Mainstream hosts accept the restart request
     *  from the audio thread (they defer internally); extraFlags lets the
     *  caller batch kParamValuesChanged from a preset load into one call. */
    void refreshLatency(Steinberg_int32 extraFlags = 0) noexcept
    {
        Steinberg_int32 flags = extraFlags;
        if constexpr (HasLatency<P>)
        {
            const int now = user.getLatency();
            if (prepared && now != cachedLatency)
            {
                cachedLatency = now;
                flags |= Steinberg_Vst_RestartFlags_kLatencyChanged;
            }
        }
        if (flags != 0 && handler != nullptr)
            handler->lpVtbl->restartComponent(handler, flags);
    }

    // --- block event plumbing -------------------------------------------------------
    //
    // Every host input lands in ONE timestamped stream (parameter points,
    // bypass points, note events), sorted and applied either at sub-block
    // boundaries (sample-accurate, the default) or all up front.

    /** Collects parameter queue points. In sample-accurate mode every point
     *  is kept with its offset; otherwise only the last point per queue. */
    void collectParameterChanges(Steinberg_Vst_IParameterChanges* changes,
                                 Steinberg_int32 numSamples,
                                 BlockEvent* events, int& count) noexcept
    {
        if (changes == nullptr) return;
        const Steinberg_int32 queues = changes->lpVtbl->getParameterCount(changes);
        for (Steinberg_int32 q = 0; q < queues; ++q)
        {
            Steinberg_Vst_IParamValueQueue* queue =
                changes->lpVtbl->getParameterData(changes, q);
            if (queue == nullptr) continue;
            const Steinberg_int32 points = queue->lpVtbl->getPointCount(queue);
            if (points <= 0) continue;
            const Steinberg_Vst_ParamID id = queue->lpVtbl->getParameterId(queue);
            const Steinberg_int32 first = sampleAccurateOf<P>() ? 0 : points - 1;
            for (Steinberg_int32 i = first; i < points; ++i)
            {
                Steinberg_int32 offset = 0;
                Steinberg_Vst_ParamValue value = 0.0;
                if (queue->lpVtbl->getPoint(queue, i, &offset, &value)
                    != Steinberg_kResultOk)
                    continue;
                if (offset < 0) offset = 0;
                if (numSamples > 0 && offset >= numSamples) offset = numSamples - 1;
                BlockEvent ev {};
                ev.offset = offset;
                ev.kind = id == kBypassParamId ? BlockEvent::Kind::Bypass
                                               : BlockEvent::Kind::Param;
                ev.paramId = id;
                ev.value = value;
                if (count < kMaxBlockEvents) events[count++] = ev;
                else events[kMaxBlockEvents - 1] = ev;   // overflow: keep the latest
            }
        }
    }

    /** Collects the VST3 note events (HasMidi plugins only). */
    void collectInputEvents(Steinberg_Vst_IEventList* list,
                            Steinberg_int32 numSamples,
                            BlockEvent* events, int& count) noexcept
    {
        if constexpr (HasMidi<P>)
        {
            if (list == nullptr) return;
            const Steinberg_int32 n = list->lpVtbl->getEventCount(list);
            for (Steinberg_int32 i = 0; i < n; ++i)
            {
                Steinberg_Vst_Event raw {};
                if (list->lpVtbl->getEvent(list, i, &raw) != Steinberg_kResultOk)
                    continue;
                MidiEvent midi {};
                if (raw.type == Steinberg_Vst_Event_EventTypes_kNoteOnEvent)
                {
                    midi.type = MidiEvent::Type::NoteOn;
                    midi.channel = static_cast<uint8_t>(
                        raw.Steinberg_Vst_Event_noteOn.channel & 0x0F);
                    midi.note = static_cast<uint8_t>(
                        raw.Steinberg_Vst_Event_noteOn.pitch & 0x7F);
                    midi.value = raw.Steinberg_Vst_Event_noteOn.velocity;
                }
                else if (raw.type == Steinberg_Vst_Event_EventTypes_kNoteOffEvent)
                {
                    midi.type = MidiEvent::Type::NoteOff;
                    midi.channel = static_cast<uint8_t>(
                        raw.Steinberg_Vst_Event_noteOff.channel & 0x0F);
                    midi.note = static_cast<uint8_t>(
                        raw.Steinberg_Vst_Event_noteOff.pitch & 0x7F);
                    midi.value = raw.Steinberg_Vst_Event_noteOff.velocity;
                }
                else if (raw.type == Steinberg_Vst_Event_EventTypes_kPolyPressureEvent)
                {
                    midi.type = MidiEvent::Type::PolyPressure;
                    midi.channel = static_cast<uint8_t>(
                        raw.Steinberg_Vst_Event_polyPressure.channel & 0x0F);
                    midi.note = static_cast<uint8_t>(
                        raw.Steinberg_Vst_Event_polyPressure.pitch & 0x7F);
                    midi.value = raw.Steinberg_Vst_Event_polyPressure.pressure;
                }
                else
                    continue;
                Steinberg_int32 offset = raw.sampleOffset;
                if (offset < 0) offset = 0;
                if (numSamples > 0 && offset >= numSamples) offset = numSamples - 1;
                BlockEvent ev {};
                ev.offset = offset;
                ev.kind = BlockEvent::Kind::Midi;
                ev.midi = midi;
                if (count < kMaxBlockEvents) events[count++] = ev;
                else events[kMaxBlockEvents - 1] = ev;
            }
        }
        else
        {
            (void) list;
            (void) numSamples;
            (void) events;
            (void) count;
        }
    }

    /** Applies one event; @p blockStart rebases MIDI offsets onto the next
     *  processBlock call. Returns true when a user parameter changed (the
     *  latency re-check trigger) — program changes report through
     *  @p programChanged instead. */
    bool applyBlockEvent(const BlockEvent& ev, int blockStart,
                         bool& programChanged) noexcept
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
            if constexpr (kNumPresets > 0)
            {
                if (ev.paramId == kProgramParamId)
                {
                    applyFactoryPresetIdx(static_cast<int>(
                        ev.value * (kNumPresets - 1) + 0.5));
                    programChanged = true;
                    return true;
                }
            }
            if constexpr (HasMidi<P>)
            {
                if (isMidiProxyId(ev.paramId))
                {
                    const int slot = midiProxySlot(ev.paramId);
                    midiProxyShadow[static_cast<size_t>(slot)].store(
                        ev.value, std::memory_order_relaxed);
                    MidiEvent midi = midiProxyToEvent(ev.paramId, ev.value,
                                                      ev.offset - blockStart);
                    if (midi.sampleOffset < 0) midi.sampleOffset = 0;
                    user.handleMidiEvent(midi);
                    return false;
                }
            }
            if (const int idx = indexOfParamId(ev.paramId); idx >= 0)
            {
                applyNormalized(idx, ev.value);
                return true;
            }
            return false;
        }
    }

    /** Table slot of a proxy id: channel * 4 + controller slot. */
    static int midiProxySlot(uint32_t id) noexcept
    {
        const int channel = static_cast<int>((id >> 8) & 0x0Fu);
        const uint32_t ctrl = id & 0xFFu;
        const int slot = ctrl == 1 ? 0 : (ctrl == 64 ? 1 : (ctrl == 128 ? 2 : 3));
        return channel * 4 + slot;
    }

    // ==========================================================================
    // Lens 0 — IComponent (+ IPluginBase)
    // ==========================================================================

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sInitialize(void*, Steinberg_FUnknown*)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sTerminate(void*)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetControllerClassId(void*, Steinberg_TUID)
    { return Steinberg_kResultFalse; }   // single component: no separate controller

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetIoMode(void*, Steinberg_Vst_IoMode)
    { return Steinberg_kResultOk; }

    /// Audio inputs: none for an Instrument; otherwise the main pair plus
    /// an aux bus when the class implements the sidechain processBlock.
    static Steinberg_int32 numInputBuses() noexcept
    {
        if (kIsInstrument) return 0;
        return HasSidechain<P> ? 2 : 1;
    }

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sGetBusCount(void*,
        Steinberg_Vst_MediaType type, Steinberg_Vst_BusDirection dir)
    {
        if (type == Steinberg_Vst_MediaTypes_kEvent)
            return (HasMidi<P> && dir == Steinberg_Vst_BusDirections_kInput) ? 1 : 0;
        if (type != Steinberg_Vst_MediaTypes_kAudio) return 0;
        return dir == Steinberg_Vst_BusDirections_kInput ? numInputBuses() : 1;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetBusInfo(void* self_,
        Steinberg_Vst_MediaType type, Steinberg_Vst_BusDirection dir,
        Steinberg_int32 index, Steinberg_Vst_BusInfo* bus)
    {
        if (bus == nullptr) return Steinberg_kInvalidArgument;
        if (type == Steinberg_Vst_MediaTypes_kEvent)
        {
            if (!HasMidi<P> || dir != Steinberg_Vst_BusDirections_kInput
                || index != 0)
                return Steinberg_kInvalidArgument;
            bus->mediaType = Steinberg_Vst_MediaTypes_kEvent;
            bus->direction = dir;
            bus->channelCount = 16;
            bus->busType = Steinberg_Vst_BusTypes_kMain;
            bus->flags = Steinberg_Vst_BusInfo_BusFlags_kDefaultActive;
            asciiToString128("MIDI In", bus->name);
            return Steinberg_kResultOk;
        }
        const Steinberg_int32 count =
            dir == Steinberg_Vst_BusDirections_kInput ? numInputBuses() : 1;
        if (type != Steinberg_Vst_MediaTypes_kAudio || index < 0 || index >= count)
            return Steinberg_kInvalidArgument;
        bus->mediaType = Steinberg_Vst_MediaTypes_kAudio;
        bus->direction = dir;
        bus->channelCount = fromLens(self_, 0)->currentChannels;
        bus->busType = index == 0 ? Steinberg_Vst_BusTypes_kMain
                                  : Steinberg_Vst_BusTypes_kAux;
        bus->flags = Steinberg_Vst_BusInfo_BusFlags_kDefaultActive;
        asciiToString128(index == 1 ? "Sidechain"
                         : (dir == Steinberg_Vst_BusDirections_kInput ? "Input"
                                                                      : "Output"),
                         bus->name);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetRoutingInfo(void*,
        Steinberg_Vst_RoutingInfo*, Steinberg_Vst_RoutingInfo*)
    { return Steinberg_kNotImplemented; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sActivateBus(void*,
        Steinberg_Vst_MediaType, Steinberg_Vst_BusDirection, Steinberg_int32,
        Steinberg_TBool)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetActive(void* self_, Steinberg_TBool state)
    {
        auto* p = fromLens(self_, 0);
        if (state)
        {
            const int maxBlock = p->setup.maxSamplesPerBlock > 0
                               ? p->setup.maxSamplesPerBlock : 512;
            const double sr = p->setup.sampleRate > 0.0 ? p->setup.sampleRate : 48000.0;
            dspark::AudioSpec spec { sr, maxBlock, p->currentChannels };
            p->user.prepare(spec);
            if constexpr (HasOfflineMode<P>)
                p->user.setOfflineRendering(
                    p->setup.processMode == Steinberg_Vst_ProcessModes_kOffline);
            p->applyAllShadows();
            p->dryL.assign(static_cast<size_t>(maxBlock), 0.0f);
            p->dryR.assign(static_cast<size_t>(maxBlock), 0.0f);
            if constexpr (HasSidechain<P>)
                p->silence.assign(static_cast<size_t>(maxBlock), 0.0f);
            p->bypassMix = p->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
            if constexpr (HasLatency<P>)
                p->cachedLatency = p->user.getLatency();
            p->prepared = true;
        }
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sComponentSetState(void* self_,
                                                                       Steinberg_IBStream* state)
    {
        auto* p = fromLens(self_, 0);
        if (state == nullptr) return Steinberg_kInvalidArgument;
        // Read whatever the host hands us (chunked); cap defensively at 16 MB.
        std::vector<uint8_t> blob;
        char chunk[4096];
        for (;;)
        {
            Steinberg_int32 got = 0;
            if (state->lpVtbl->read(state, chunk, static_cast<Steinberg_int32>(sizeof(chunk)),
                                    &got) != Steinberg_kResultOk || got <= 0)
                break;
            blob.insert(blob.end(), chunk, chunk + got);
            if (blob.size() > (16u << 20)) return Steinberg_kResultFalse;
        }
        double norm[kNumParams == 0 ? 1 : kNumParams];
        for (size_t i = 0; i < kNumParams; ++i)
            norm[i] = p->shadow[i].load(std::memory_order_relaxed);
        int program = -1;
        if (!applyState(p->user, blob.data(), blob.size(), norm, &program))
            return Steinberg_kResultFalse;
        for (size_t i = 0; i < kNumParams; ++i)
            p->applyNormalized(static_cast<int>(i), norm[i]);
        if (kNumPresets > 0 && program >= 0 && program < kNumPresets)
            p->currentProgram.store(program, std::memory_order_relaxed);
        p->refreshLatency();   // restored state may imply a new lookahead
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sComponentGetState(void* self_,
                                                                       Steinberg_IBStream* state)
    {
        auto* p = fromLens(self_, 0);
        if (state == nullptr) return Steinberg_kInvalidArgument;
        double norm[kNumParams == 0 ? 1 : kNumParams];
        for (size_t i = 0; i < kNumParams; ++i)
            norm[i] = p->shadow[i].load(std::memory_order_relaxed);
        const std::vector<uint8_t> blob = buildState(
            p->user, norm, kNumParams,
            kNumPresets > 0 ? p->currentProgram.load(std::memory_order_relaxed) : -1);
        return streamWrite(state, blob.data(), static_cast<Steinberg_int32>(blob.size()))
             ? Steinberg_kResultOk : Steinberg_kResultFalse;
    }

    inline static const Steinberg_Vst_IComponentVtbl kComponentVtbl = {
        &sQuery<0>, &sAddRef<0>, &sRelease<0>,
        &sInitialize, &sTerminate,
        &sGetControllerClassId, &sSetIoMode, &sGetBusCount, &sGetBusInfo,
        &sGetRoutingInfo, &sActivateBus, &sSetActive,
        &sComponentSetState, &sComponentGetState
    };

    // ==========================================================================
    // Lens 1 — IAudioProcessor
    // ==========================================================================

    /// Mono and stereo negotiate per the declared ChannelSupport; every bus
    /// of an instance (main in, sidechain, out) runs the same width.
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetBusArrangements(void* self_,
        Steinberg_Vst_SpeakerArrangement* inputs, Steinberg_int32 numIns,
        Steinberg_Vst_SpeakerArrangement* outputs, Steinberg_int32 numOuts)
    {
        if (numIns != numInputBuses() || numOuts != 1
            || (numIns > 0 && inputs == nullptr) || outputs == nullptr)
            return Steinberg_kResultFalse;
        const Steinberg_Vst_SpeakerArrangement want = outputs[0];
        int channels = 0;
        if (want == Steinberg_Vst_SpeakerArr_kMono) channels = 1;
        else if (want == Steinberg_Vst_SpeakerArr_kStereo) channels = 2;
        if (channels == 0 || !supportsChannelCount<P>(channels))
            return Steinberg_kResultFalse;
        for (Steinberg_int32 i = 0; i < numIns; ++i)
            if (inputs[i] != want)
                return Steinberg_kResultFalse;
        fromLens(self_, 1)->currentChannels = channels;
        return Steinberg_kResultTrue;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetBusArrangement(void* self_,
        Steinberg_Vst_BusDirection dir, Steinberg_int32 index,
        Steinberg_Vst_SpeakerArrangement* arr)
    {
        const Steinberg_int32 count =
            dir == Steinberg_Vst_BusDirections_kInput ? numInputBuses() : 1;
        if (index < 0 || index >= count || arr == nullptr)
            return Steinberg_kInvalidArgument;
        *arr = fromLens(self_, 1)->currentChannels == 1
             ? Steinberg_Vst_SpeakerArr_kMono : Steinberg_Vst_SpeakerArr_kStereo;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCanProcessSampleSize(void*,
                                                                          Steinberg_int32 size)
    {
        return size == Steinberg_Vst_SymbolicSampleSizes_kSample32
             ? Steinberg_kResultTrue : Steinberg_kResultFalse;
    }

    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sGetLatencySamples(void* self_)
    {
        return static_cast<Steinberg_uint32>(fromLens(self_, 1)->cachedLatency);
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetupProcessing(void* self_,
        Steinberg_Vst_ProcessSetup* setup)
    {
        if (setup == nullptr) return Steinberg_kInvalidArgument;
        fromLens(self_, 1)->setup = *setup;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetProcessing(void*, Steinberg_TBool)
    { return Steinberg_kResultOk; }

    /** Forwards the host ProcessContext as a TransportInfo (HasTransport). */
    void forwardTransport(const Steinberg_Vst_ProcessContext* ctx) noexcept
    {
        if constexpr (HasTransport<P>)
        {
            if (ctx == nullptr) return;
            TransportInfo t {};
            const Steinberg_uint32 s = ctx->state;
            t.playing  = (s & Steinberg_Vst_ProcessContext_StatesAndFlags_kPlaying) != 0;
            t.recording = (s & Steinberg_Vst_ProcessContext_StatesAndFlags_kRecording) != 0;
            t.looping  = (s & Steinberg_Vst_ProcessContext_StatesAndFlags_kCycleActive) != 0;
            if ((s & Steinberg_Vst_ProcessContext_StatesAndFlags_kTempoValid) != 0)
            {
                t.tempoBpm = ctx->tempo;
                t.tempoValid = true;
            }
            if ((s & Steinberg_Vst_ProcessContext_StatesAndFlags_kProjectTimeMusicValid) != 0)
            {
                t.ppqPosition = ctx->projectTimeMusic;
                t.positionValid = true;
            }
            if ((s & Steinberg_Vst_ProcessContext_StatesAndFlags_kBarPositionValid) != 0)
                t.barStartPpq = ctx->barPositionMusic;
            if ((s & Steinberg_Vst_ProcessContext_StatesAndFlags_kTimeSigValid) != 0)
            {
                t.timeSigNumerator = ctx->timeSigNumerator;
                t.timeSigDenominator = ctx->timeSigDenominator;
                t.timeSigValid = true;
            }
            if ((s & Steinberg_Vst_ProcessContext_StatesAndFlags_kCycleValid) != 0)
            {
                t.loopStartPpq = ctx->cycleStartMusic;
                t.loopEndPpq = ctx->cycleEndMusic;
                t.loopValid = true;
            }
            user.setTransport(t);
        }
        else
            (void) ctx;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sProcess(void* self_,
                                                             Steinberg_Vst_ProcessData* data)
    {
        auto* p = fromLens(self_, 1);
        if (data == nullptr) return Steinberg_kInvalidArgument;

        // FTZ/DAZ for the whole callback: DSPark processors guard their own
        // hot loops, this covers user DSP in hosts that do not set it.
        dspark::DenormalGuard denormalGuard;

        const Steinberg_int32 n = data->numSamples;

        // One timestamped stream for everything the host handed us.
        BlockEvent events[kMaxBlockEvents];
        int eventCount = 0;
        p->collectParameterChanges(data->inputParameterChanges, n, events, eventCount);
        p->collectInputEvents(data->inputEvents, n, events, eventCount);
        sortBlockEvents(events, eventCount);

        bool paramsChanged = false;
        bool programChanged = false;

        const bool canRender = n > 0
            && data->symbolicSampleSize == Steinberg_Vst_SymbolicSampleSizes_kSample32
            && data->numOutputs >= 1 && data->outputs != nullptr && p->prepared;
        if (!canRender)
        {
            // Parameter flush (n == 0) or unrenderable block: still apply
            // every event so automation/state never desynchronises.
            for (int i = 0; i < eventCount; ++i)
                paramsChanged |= p->applyBlockEvent(events[i], events[i].offset,
                                                    programChanged);
            if (paramsChanged || programChanged)
                p->refreshLatency(programChanged
                    ? Steinberg_Vst_RestartFlags_kParamValuesChanged : 0);
            if (n <= 0) return Steinberg_kResultOk;
            return data->symbolicSampleSize
                       == Steinberg_Vst_SymbolicSampleSizes_kSample32
                 ? Steinberg_kResultOk : Steinberg_kResultFalse;
        }

        p->forwardTransport(data->processContext);

        auto& outBus = data->outputs[0];
        float** out = outBus.Steinberg_Vst_AudioBusBuffers_channelBuffers32;
        if (out == nullptr || outBus.numChannels < 1) return Steinberg_kResultOk;
        const int width = p->currentChannels;
        const int nCh = outBus.numChannels < width ? outBus.numChannels : width;

        const bool haveIn = !kIsInstrument
            && data->numInputs >= 1 && data->inputs != nullptr
            && data->inputs[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 != nullptr;
        float** in = haveIn
            ? data->inputs[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 : nullptr;

        if (static_cast<size_t>(n) > p->dryL.size())
            return Steinberg_kResultOk;   // oversize block: pass through

        // Keep the dry signal for the bypass blend, then process the output
        // buffers in place (copying input over first when distinct). An
        // instrument has no input: its output starts cleared (voices ADD)
        // and its bypass blends toward silence (the dry vectors stay zero).
        float* dry[2] = { p->dryL.data(), p->dryR.data() };
        const size_t bytes = sizeof(float) * static_cast<size_t>(n);
        for (int ch = 0; ch < nCh; ++ch)
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

        // Sidechain: aux bus 1, pre-allocated silence when nothing is
        // routed (same frame count, no branches user-side). Read-only.
        float* scPtrs[2] = { nullptr, nullptr };
        if constexpr (HasSidechain<P>)
        {
            if (static_cast<size_t>(n) > p->silence.size())
                return Steinberg_kResultOk;
            scPtrs[0] = scPtrs[1] = p->silence.data();
            if (data->numInputs >= 2 && data->inputs != nullptr
                && data->inputs[1].Steinberg_Vst_AudioBusBuffers_channelBuffers32
                       != nullptr)
            {
                const auto& scBus = data->inputs[1];
                float** sc = scBus.Steinberg_Vst_AudioBusBuffers_channelBuffers32;
                const int scCh = scBus.numChannels < 2 ? scBus.numChannels : 2;
                for (int ch = 0; ch < scCh; ++ch)
                    if (sc[ch] != nullptr)
                        scPtrs[ch] = sc[ch];
                if (scCh == 1 && sc[0] != nullptr)
                    scPtrs[1] = sc[0];   // mono key feeds both detector ears
            }
        }

        // Process in sub-blocks split at quantum-aligned event positions
        // (sample-accurate default); without splitting, apply everything up
        // front and run the block in one call.
        auto processSegment = [&](int start, int length) noexcept {
            float* sub[2] = { out[0] + start,
                              nCh > 1 ? out[1] + start : out[0] + start };
            dspark::AudioBufferView<float> view(sub, nCh, length);
            if constexpr (HasSidechain<P>)
            {
                // The key view mirrors the main width (mono main, mono key).
                float* scSub[2] = { scPtrs[0] + start, scPtrs[1] + start };
                dspark::AudioBufferView<float> scView(scSub, nCh, length);
                p->user.processBlock(view, scView);
            }
            else
                p->user.processBlock(view);
        };

        int evIdx = 0;
        if (!sampleAccurateOf<P>())
        {
            for (; evIdx < eventCount; ++evIdx)
                paramsChanged |= p->applyBlockEvent(events[evIdx], 0, programChanged);
            processSegment(0, n);
        }
        else
        {
            int pos = 0;
            while (pos < n)
            {
                while (evIdx < eventCount
                       && (events[evIdx].offset / kAutomationQuantum)
                              * kAutomationQuantum <= pos)
                    paramsChanged |= p->applyBlockEvent(events[evIdx++], pos,
                                                        programChanged);
                int next = n;
                if (evIdx < eventCount)
                {
                    const int snapped = (events[evIdx].offset / kAutomationQuantum)
                                      * kAutomationQuantum;
                    if (snapped < next) next = snapped;
                }
                if (next <= pos) next = pos + kAutomationQuantum < n
                                      ? pos + kAutomationQuantum : n;
                processSegment(pos, next - pos);
                pos = next;
            }
            for (; evIdx < eventCount; ++evIdx)   // safety: events at block end
                paramsChanged |= p->applyBlockEvent(events[evIdx], n, programChanged);
        }

        // Soft bypass: short linear crossfade toward the dry signal.
        const float target = p->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (p->bypassMix != target || target > 0.0f)
        {
            const float step = 1.0f / static_cast<float>(kBypassRampSamples);
            float mix = p->bypassMix;
            for (Steinberg_int32 i = 0; i < n; ++i)
            {
                mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
                mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
                for (int ch = 0; ch < nCh; ++ch)
                    out[ch][i] += (dry[ch][i] - out[ch][i]) * mix;
            }
            p->bypassMix = mix;
        }

        if (paramsChanged || programChanged)
            p->refreshLatency(programChanged
                ? Steinberg_Vst_RestartFlags_kParamValuesChanged : 0);

        outBus.silenceFlags = 0;
        return Steinberg_kResultOk;
    }

    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sGetTailSamples(void* self_)
    {
        auto* p = fromLens(self_, 1);
        if constexpr (HasTail<P>)
            return static_cast<Steinberg_uint32>(
                p->user.getTailSeconds() * (p->setup.sampleRate > 0 ? p->setup.sampleRate
                                                                    : 48000.0));
        else
        {
            (void) p;
            return 0;
        }
    }

    inline static const Steinberg_Vst_IAudioProcessorVtbl kProcessorVtbl = {
        &sQuery<1>, &sAddRef<1>, &sRelease<1>,
        &sSetBusArrangements, &sGetBusArrangement, &sCanProcessSampleSize,
        &sGetLatencySamples, &sSetupProcessing, &sSetProcessing, &sProcess,
        &sGetTailSamples
    };

    // ==========================================================================
    // Lens 2 — IEditController (+ IPluginBase)
    // ==========================================================================

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCtrlInitialize(void*, Steinberg_FUnknown*)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCtrlTerminate(void*)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetComponentState(void* self_,
                                                                       Steinberg_IBStream* state)
    {
        // Single component: identical to IComponent::setState (lens offset!).
        return sComponentSetState(fromLens(self_, 2)->lensPtr(0), state);
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCtrlSetState(void*, Steinberg_IBStream*)
    { return Steinberg_kResultOk; }   // no controller-only state

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCtrlGetState(void*, Steinberg_IBStream*)
    { return Steinberg_kResultOk; }

    // Parameter index layout: [0, N) the user table, N the bypass, N+1 the
    // program change (factory presets only), then the hidden MIDI proxies.
    static constexpr Steinberg_int32 kBypassIndex =
        static_cast<Steinberg_int32>(kNumParams);
    static constexpr Steinberg_int32 kProgramIndex =
        kNumPresets > 0 ? kBypassIndex + 1 : -1;
    static constexpr Steinberg_int32 kFirstProxyIndex =
        kBypassIndex + 1 + (kNumPresets > 0 ? 1 : 0);

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sGetParameterCount(void*)
    {
        return kFirstProxyIndex + (HasMidi<P> ? kNumMidiProxies : 0);
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetParameterInfo(void* self_,
        Steinberg_int32 index, Steinberg_Vst_ParameterInfo* info)
    {
        (void) self_;
        if (info == nullptr || index < 0 || index >= sGetParameterCount(self_))
            return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        if (index == kBypassIndex)
        {
            info->id = kBypassParamId;
            asciiToString128("Bypass", info->title);
            asciiToString128("Byp", info->shortTitle);
            info->stepCount = 1;
            info->defaultNormalizedValue = 0.0;
            info->flags = Steinberg_Vst_ParameterInfo_ParameterFlags_kCanAutomate
                        | Steinberg_Vst_ParameterInfo_ParameterFlags_kIsBypass;
            return Steinberg_kResultOk;
        }
        if (kNumPresets > 0 && index == kProgramIndex)
        {
            info->id = kProgramParamId;
            asciiToString128("Program", info->title);
            asciiToString128("Prog", info->shortTitle);
            info->stepCount = kNumPresets - 1;
            info->defaultNormalizedValue = 0.0;
            info->unitId = Steinberg_Vst_kRootUnitId;
            info->flags = Steinberg_Vst_ParameterInfo_ParameterFlags_kCanAutomate
                        | Steinberg_Vst_ParameterInfo_ParameterFlags_kIsList
                        | Steinberg_Vst_ParameterInfo_ParameterFlags_kIsProgramChange;
            return Steinberg_kResultOk;
        }
        if (HasMidi<P> && index >= kFirstProxyIndex)
        {
            // Hidden conduits for IMidiMapping: hosts write MIDI controller
            // motion into them; they never show in generic UIs.
            const int slot = index - kFirstProxyIndex;
            const int channel = slot / 4;
            const int ctrl = kMidiProxyControllers[slot % 4];
            info->id = midiProxyId(channel, ctrl);
            asciiToString128("MIDI", info->title);
            asciiToString128("MIDI", info->shortTitle);
            info->stepCount = 0;
            info->defaultNormalizedValue = ctrl == 129 ? 0.5 : 0.0;
            info->flags = Steinberg_Vst_ParameterInfo_ParameterFlags_kIsHidden;
            return Steinberg_kResultOk;
        }
        const auto& spec = P::parameters[static_cast<size_t>(index)];
        info->id = hash32(spec.id);
        asciiToString128(spec.name, info->title);
        asciiToString128(spec.name, info->shortTitle);
        asciiToString128(spec.unit, info->units);
        info->stepCount = spec.steps;
        info->defaultNormalizedValue = toNormalized(spec, spec.defValue);
        info->flags = Steinberg_Vst_ParameterInfo_ParameterFlags_kCanAutomate;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetParamStringByValue(void* self_,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue normalized,
        Steinberg_Vst_String128 out)
    {
        (void) self_;
        if (out == nullptr) return Steinberg_kInvalidArgument;
        char text[64] = "";
        if (id == kBypassParamId)
            std::snprintf(text, sizeof(text), "%s", normalized >= 0.5 ? "On" : "Off");
        else if (kNumPresets > 0 && id == kProgramParamId)
        {
            if constexpr (HasFactoryPresets<P>)
            {
                int idx = static_cast<int>(normalized * (kNumPresets - 1) + 0.5);
                idx = idx < 0 ? 0 : (idx >= kNumPresets ? kNumPresets - 1 : idx);
                std::snprintf(text, sizeof(text), "%s",
                              P::factoryPresets[static_cast<size_t>(idx)].name);
            }
        }
        else if (HasMidi<P> && isMidiProxyId(id))
            std::snprintf(text, sizeof(text), "%.3f", normalized);
        else
        {
            const int idx = indexOfParamId(id);
            if (idx < 0) return Steinberg_kInvalidArgument;
            const auto& spec = P::parameters[static_cast<size_t>(idx)];
            formatValue(spec, toPlain(spec, normalized), text, sizeof(text));
        }
        asciiToString128(text, out);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetParamValueByString(void* self_,
        Steinberg_Vst_ParamID id, Steinberg_Vst_TChar* string,
        Steinberg_Vst_ParamValue* normalized)
    {
        (void) self_;
        if (string == nullptr || normalized == nullptr) return Steinberg_kInvalidArgument;
        char ascii[64];
        int i = 0;
        for (; i < 63 && string[i] != 0; ++i)
            ascii[i] = static_cast<char>(string[i]);
        ascii[i] = '\0';
        const int toggle = parseToggleText(ascii);
        if (id == kBypassParamId)
        {
            *normalized = toggle >= 0 ? toggle
                                      : (std::strtod(ascii, nullptr) >= 0.5 ? 1.0 : 0.0);
            return Steinberg_kResultOk;
        }
        if (kNumPresets > 0 && id == kProgramParamId)
        {
            if constexpr (HasFactoryPresets<P>)
            {
                for (int presetIdx = 0; presetIdx < kNumPresets; ++presetIdx)
                    if (std::strcmp(ascii,
                            P::factoryPresets[static_cast<size_t>(presetIdx)].name)
                        == 0)
                    {
                        *normalized = kNumPresets > 1
                            ? static_cast<double>(presetIdx) / (kNumPresets - 1)
                            : 0.0;
                        return Steinberg_kResultOk;
                    }
            }
            const double v = std::strtod(ascii, nullptr);
            *normalized = kNumPresets > 1 ? v / (kNumPresets - 1) : 0.0;
            return Steinberg_kResultOk;
        }
        if (HasMidi<P> && isMidiProxyId(id))
        {
            *normalized = std::strtod(ascii, nullptr);
            return Steinberg_kResultOk;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return Steinberg_kInvalidArgument;
        const Param& spec = P::parameters[static_cast<size_t>(idx)];
        if (spec.steps == 1 && toggle >= 0)
            *normalized = toggle;
        else
            *normalized = toNormalized(spec, std::strtod(ascii, nullptr));
        return Steinberg_kResultOk;
    }

    static Steinberg_Vst_ParamValue SMTG_STDMETHODCALLTYPE sNormalizedParamToPlain(void*,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue normalized)
    {
        if (id == kBypassParamId) return normalized >= 0.5 ? 1.0 : 0.0;
        if (kNumPresets > 0 && id == kProgramParamId)
            return static_cast<double>(
                static_cast<int>(normalized * (kNumPresets - 1) + 0.5));
        if (HasMidi<P> && isMidiProxyId(id)) return normalized;
        const int idx = indexOfParamId(id);
        return idx < 0 ? 0.0 : toPlain(P::parameters[static_cast<size_t>(idx)], normalized);
    }

    static Steinberg_Vst_ParamValue SMTG_STDMETHODCALLTYPE sPlainParamToNormalized(void*,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue plain)
    {
        if (id == kBypassParamId) return plain >= 0.5 ? 1.0 : 0.0;
        if (kNumPresets > 0 && id == kProgramParamId)
            return kNumPresets > 1 ? plain / (kNumPresets - 1) : 0.0;
        if (HasMidi<P> && isMidiProxyId(id)) return plain;
        const int idx = indexOfParamId(id);
        return idx < 0 ? 0.0 : toNormalized(P::parameters[static_cast<size_t>(idx)], plain);
    }

    static Steinberg_Vst_ParamValue SMTG_STDMETHODCALLTYPE sGetParamNormalized(void* self_,
        Steinberg_Vst_ParamID id)
    {
        auto* p = fromLens(self_, 2);
        if (id == kBypassParamId)
            return p->bypass.load(std::memory_order_relaxed) ? 1.0 : 0.0;
        if (kNumPresets > 0 && id == kProgramParamId)
            return kNumPresets > 1
                 ? static_cast<double>(p->currentProgram.load(std::memory_order_relaxed))
                       / (kNumPresets - 1)
                 : 0.0;
        if constexpr (HasMidi<P>)
        {
            if (isMidiProxyId(id))
                return p->midiProxyShadow[static_cast<size_t>(midiProxySlot(id))]
                    .load(std::memory_order_relaxed);
        }
        const int idx = indexOfParamId(id);
        return idx < 0 ? 0.0
                       : p->shadow[static_cast<size_t>(idx)].load(std::memory_order_relaxed);
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetParamNormalized(void* self_,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue value)
    {
        auto* p = fromLens(self_, 2);
        if (id == kBypassParamId)
        {
            p->bypass.store(value >= 0.5, std::memory_order_relaxed);
            return Steinberg_kResultOk;
        }
        if (kNumPresets > 0 && id == kProgramParamId)
        {
            // A program change from the host UI: apply the preset and tell
            // the host every other parameter moved (main-thread call).
            p->applyFactoryPresetIdx(
                static_cast<int>(value * (kNumPresets - 1) + 0.5));
            p->refreshLatency(Steinberg_Vst_RestartFlags_kParamValuesChanged);
            return Steinberg_kResultOk;
        }
        if constexpr (HasMidi<P>)
        {
            // Live MIDI rides the process() queues; a main-thread write only
            // keeps the proxy shadow coherent for host round-trips.
            if (isMidiProxyId(id))
            {
                p->midiProxyShadow[static_cast<size_t>(midiProxySlot(id))]
                    .store(value, std::memory_order_relaxed);
                return Steinberg_kResultOk;
            }
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return Steinberg_kInvalidArgument;
        p->applyNormalized(idx, value);   // user setters are atomic by contract
        p->refreshLatency();
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetComponentHandler(void* self_,
        Steinberg_Vst_IComponentHandler* handler)
    {
        auto* p = fromLens(self_, 2);
        if (p->handler != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(p->handler)->lpVtbl->release(p->handler);
        p->handler = handler;
        if (p->handler != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(p->handler)->lpVtbl->addRef(p->handler);
        return Steinberg_kResultOk;
    }

    static Steinberg_IPlugView* SMTG_STDMETHODCALLTYPE sCreateView(void* self_,
                                                                   Steinberg_FIDString name);

    inline static const Steinberg_Vst_IEditControllerVtbl kControllerVtbl = {
        &sQuery<2>, &sAddRef<2>, &sRelease<2>,
        &sCtrlInitialize, &sCtrlTerminate,
        &sSetComponentState, &sCtrlSetState, &sCtrlGetState,
        &sGetParameterCount, &sGetParameterInfo,
        &sGetParamStringByValue, &sGetParamValueByString,
        &sNormalizedParamToPlain, &sPlainParamToNormalized,
        &sGetParamNormalized, &sSetParamNormalized,
        &sSetComponentHandler, &sCreateView
    };

    // ==========================================================================
    // Lens 3 — IProcessContextRequirements
    // ==========================================================================

    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sGetProcessContextRequirements(void*)
    {
        if constexpr (HasTransport<P>)
            return Steinberg_Vst_IProcessContextRequirements_Flags_kNeedTempo
                 | Steinberg_Vst_IProcessContextRequirements_Flags_kNeedTransportState
                 | Steinberg_Vst_IProcessContextRequirements_Flags_kNeedProjectTimeMusic
                 | Steinberg_Vst_IProcessContextRequirements_Flags_kNeedBarPositionMusic
                 | Steinberg_Vst_IProcessContextRequirements_Flags_kNeedTimeSignature
                 | Steinberg_Vst_IProcessContextRequirements_Flags_kNeedCycleMusic;
        else
            return 0;
    }

    inline static const Steinberg_Vst_IProcessContextRequirementsVtbl kContextReqVtbl = {
        &sQuery<3>, &sAddRef<3>, &sRelease<3>,
        &sGetProcessContextRequirements
    };

    // ==========================================================================
    // Lens 4 — IUnitInfo (factory presets as a program list; surfaced by
    // queryInterface only when the class declares factoryPresets)
    // ==========================================================================

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sGetUnitCount(void*) { return 1; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetUnitInfo(void*,
        Steinberg_int32 unitIndex, Steinberg_Vst_UnitInfo* info)
    {
        if (unitIndex != 0 || info == nullptr) return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        info->id = Steinberg_Vst_kRootUnitId;
        info->parentUnitId = Steinberg_Vst_kNoParentUnitId;
        info->programListId = kNumPresets > 0 ? kPresetProgramListId
                                              : Steinberg_Vst_kNoProgramListId;
        asciiToString128("Root", info->name);
        return Steinberg_kResultOk;
    }

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sGetProgramListCount(void*)
    {
        return kNumPresets > 0 ? 1 : 0;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetProgramListInfo(void*,
        Steinberg_int32 listIndex, Steinberg_Vst_ProgramListInfo* info)
    {
        if (listIndex != 0 || info == nullptr || kNumPresets == 0)
            return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        info->id = kPresetProgramListId;
        info->programCount = kNumPresets;
        asciiToString128("Factory Presets", info->name);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetProgramName(void*,
        Steinberg_Vst_ProgramListID listId, Steinberg_int32 programIndex,
        Steinberg_Vst_String128 name)
    {
        if constexpr (HasFactoryPresets<P>)
        {
            if (listId != kPresetProgramListId || name == nullptr
                || programIndex < 0 || programIndex >= kNumPresets)
                return Steinberg_kInvalidArgument;
            asciiToString128(
                P::factoryPresets[static_cast<size_t>(programIndex)].name, name);
            return Steinberg_kResultOk;
        }
        else
        {
            (void) listId;
            (void) programIndex;
            (void) name;
            return Steinberg_kInvalidArgument;
        }
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetProgramInfo(void*,
        Steinberg_Vst_ProgramListID, Steinberg_int32, Steinberg_Vst_CString,
        Steinberg_Vst_String128)
    { return Steinberg_kResultFalse; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sHasProgramPitchNames(void*,
        Steinberg_Vst_ProgramListID, Steinberg_int32)
    { return Steinberg_kResultFalse; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetProgramPitchName(void*,
        Steinberg_Vst_ProgramListID, Steinberg_int32, Steinberg_int16,
        Steinberg_Vst_String128)
    { return Steinberg_kResultFalse; }

    static Steinberg_Vst_UnitID SMTG_STDMETHODCALLTYPE sGetSelectedUnit(void*)
    { return Steinberg_Vst_kRootUnitId; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSelectUnit(void*,
        Steinberg_Vst_UnitID unitId)
    { return unitId == Steinberg_Vst_kRootUnitId ? Steinberg_kResultOk
                                                 : Steinberg_kInvalidArgument; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetUnitByBus(void*,
        Steinberg_Vst_MediaType, Steinberg_Vst_BusDirection, Steinberg_int32,
        Steinberg_int32, Steinberg_Vst_UnitID* unitId)
    {
        if (unitId == nullptr) return Steinberg_kInvalidArgument;
        *unitId = Steinberg_Vst_kRootUnitId;
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetUnitProgramData(void*,
        Steinberg_int32, Steinberg_int32, struct Steinberg_IBStream*)
    { return Steinberg_kNotImplemented; }

    inline static const Steinberg_Vst_IUnitInfoVtbl kUnitVtbl = {
        &sQuery<4>, &sAddRef<4>, &sRelease<4>,
        &sGetUnitCount, &sGetUnitInfo, &sGetProgramListCount, &sGetProgramListInfo,
        &sGetProgramName, &sGetProgramInfo, &sHasProgramPitchNames,
        &sGetProgramPitchName, &sGetSelectedUnit, &sSelectUnit, &sGetUnitByBus,
        &sSetUnitProgramData
    };

    // ==========================================================================
    // Lens 5 — IMidiMapping (pitch bend / mod / sustain / channel pressure
    // arrive as proxy-parameter automation; surfaced only for HasMidi)
    // ==========================================================================

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetMidiControllerAssignment(void*,
        Steinberg_int32 busIndex, Steinberg_int16 channel,
        Steinberg_Vst_CtrlNumber midiControllerNumber, Steinberg_Vst_ParamID* id)
    {
        if (!HasMidi<P> || id == nullptr || busIndex != 0
            || channel < 0 || channel >= 16)
            return Steinberg_kResultFalse;
        const int ctrl = midiControllerNumber;
        if (ctrl != 1 && ctrl != 64 && ctrl != 128 && ctrl != 129)
            return Steinberg_kResultFalse;
        *id = midiProxyId(channel, ctrl);
        return Steinberg_kResultTrue;
    }

    inline static const Steinberg_Vst_IMidiMappingVtbl kMidiMapVtbl = {
        &sQuery<5>, &sAddRef<5>, &sRelease<5>,
        &sGetMidiControllerAssignment
    };
};

// -- the editor view (WebView editor layer) ---------------------------------------
//
// Compiled only when plugin/webview/DSParkWebViewEditor.h was included before
// this header, and created only for plugin classes that declare an editor.
// Without either, createView answers nullptr and hosts show their generic UI.

#if defined(DSPARK_PLUGIN_WEBVIEW)

/**
 * @brief IPlugView (+ IPlugViewContentScaleSupport) hosting the WebView
 * editor. A standalone COM object: it addRefs the owning plugin for its
 * lifetime and bridges UI edits to IComponentHandler gestures.
 */
template <typename P>
struct View
{
    static constexpr size_t kNumParams = Plugin<P>::kNumParams;

    // COM lenses, same layout trick as the plugin object. The timer lens is
    // Linux-only: it receives the host IRunLoop ticks that drive GTK.
    const Steinberg_IPlugViewVtbl*                    viewVtbl;
    const Steinberg_IPlugViewContentScaleSupportVtbl* scaleVtbl;
#if defined(__linux__)
    const Steinberg_Linux_ITimerHandlerVtbl*          timerVtbl;
#endif

    std::atomic<Steinberg_uint32> refs { 1 };
    Plugin<P>* owner;
    Steinberg_IPlugFrame* frame = nullptr;
    webview_ui::Editor<P> editor;
    int width;                 // current physical size (host units)
    int height;
    double scale = 1.0;
    bool correctingSize = false;   // re-entrancy guard for resizeView round-trips
    bool editActive[kNumParams == 0 ? 1 : kNumParams] {};
#if defined(__linux__)
    Steinberg_Linux_IRunLoop* runLoop = nullptr;
    bool timerRegistered = false;
#endif

    explicit View(Plugin<P>* plugin) noexcept : owner(plugin)
    {
        viewVtbl  = &kViewVtbl;
        scaleVtbl = &kScaleVtbl;
#if defined(__linux__)
        timerVtbl = &kTimerVtbl;
#endif
        owner->addRefImpl();
        const EditorSize logical = editorSizeOf<P>();
        width  = logical.width;
        height = logical.height;
    }

    ~View()
    {
#if defined(__linux__)
        releaseRunLoop();
#endif
        editor.destroy();
        if (frame != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(frame)->lpVtbl->release(frame);
        owner->releaseImpl();
    }

#if defined(__linux__)

    // --- host run loop: GTK breathes only when pumped from here ---------------------

    void acquireRunLoop() noexcept
    {
        if (frame == nullptr || runLoop != nullptr) return;
        void* loop = nullptr;
        if (reinterpret_cast<Steinberg_FUnknown*>(frame)->lpVtbl->queryInterface(
                frame, Steinberg_Linux_IRunLoop_iid, &loop) == Steinberg_kResultOk)
            runLoop = static_cast<Steinberg_Linux_IRunLoop*>(loop);
    }

    void startTimer() noexcept
    {
        acquireRunLoop();
        if (runLoop != nullptr && !timerRegistered && editor.created())
            timerRegistered = runLoop->lpVtbl->registerTimer(runLoop,
                reinterpret_cast<Steinberg_Linux_ITimerHandler*>(lensPtr(2)),
                33) == Steinberg_kResultOk;
    }

    void stopTimer() noexcept
    {
        if (runLoop != nullptr && timerRegistered)
            runLoop->lpVtbl->unregisterTimer(runLoop,
                reinterpret_cast<Steinberg_Linux_ITimerHandler*>(lensPtr(2)));
        timerRegistered = false;
    }

    void releaseRunLoop() noexcept
    {
        stopTimer();
        if (runLoop != nullptr)
        {
            reinterpret_cast<Steinberg_FUnknown*>(runLoop)->lpVtbl->release(runLoop);
            runLoop = nullptr;
        }
    }

    static void SMTG_STDMETHODCALLTYPE sOnTimer(void* self_)
    {
        fromLens(self_, 2)->editor.pump();
    }

#endif // __linux__

    static View* fromLens(void* iface, int lens) noexcept
    {
        return reinterpret_cast<View*>(static_cast<char*>(iface)
                                       - static_cast<ptrdiff_t>(lens) * sizeof(void*));
    }

    void* lensPtr(int lens) noexcept
    {
        return reinterpret_cast<char*>(this) + static_cast<ptrdiff_t>(lens) * sizeof(void*);
    }

    static const char* platformType() noexcept
    {
#if defined(_WIN32)
        return Steinberg_kPlatformTypeHWND;
#elif defined(__APPLE__)
        return Steinberg_kPlatformTypeNSView;
#else
        return Steinberg_kPlatformTypeX11EmbedWindowID;
#endif
    }

    // --- editor -> host/DSP bridge ------------------------------------------------

    static void cbSetParam(void* context, int index, double plain) noexcept
    {
        auto* view = static_cast<View*>(context);
        const Param& spec = P::parameters[static_cast<size_t>(index)];
        const double normalized = toNormalized(spec, plain);
        view->owner->applyNormalized(index, normalized);
        if (auto* handler = view->owner->handler)
        {
            const Steinberg_Vst_ParamID id = hash32(spec.id);
            if (view->editActive[static_cast<size_t>(index)])
                handler->lpVtbl->performEdit(handler, id, normalized);
            else
            {
                // Edits outside an explicit gesture still need one for host undo.
                handler->lpVtbl->beginEdit(handler, id);
                handler->lpVtbl->performEdit(handler, id, normalized);
                handler->lpVtbl->endEdit(handler, id);
            }
        }
    }

    static void cbBeginEdit(void* context, int index) noexcept
    {
        auto* view = static_cast<View*>(context);
        view->editActive[static_cast<size_t>(index)] = true;
        if (auto* handler = view->owner->handler)
            handler->lpVtbl->beginEdit(handler,
                hash32(P::parameters[static_cast<size_t>(index)].id));
    }

    static void cbEndEdit(void* context, int index) noexcept
    {
        auto* view = static_cast<View*>(context);
        view->editActive[static_cast<size_t>(index)] = false;
        if (auto* handler = view->owner->handler)
            handler->lpVtbl->endEdit(handler,
                hash32(P::parameters[static_cast<size_t>(index)].id));
    }

    // --- FUnknown -------------------------------------------------------------------

    Steinberg_tresult query(const Steinberg_TUID iid, void** obj) noexcept
    {
        if (obj == nullptr) return Steinberg_kInvalidArgument;
        *obj = nullptr;
        if (tuidEqual(iid, Steinberg_FUnknown_iid)
            || tuidEqual(iid, Steinberg_IPlugView_iid))
            *obj = lensPtr(0);
        else if (tuidEqual(iid, Steinberg_IPlugViewContentScaleSupport_iid))
            *obj = lensPtr(1);
#if defined(__linux__)
        else if (tuidEqual(iid, Steinberg_Linux_ITimerHandler_iid))
            *obj = lensPtr(2);
#endif
        if (*obj == nullptr) return Steinberg_kNoInterface;
        refs.fetch_add(1, std::memory_order_relaxed);
        return Steinberg_kResultOk;
    }

    template <int Lens>
    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sQuery(void* self_,
                                                           const Steinberg_TUID iid,
                                                           void** obj)
    { return fromLens(self_, Lens)->query(iid, obj); }

    template <int Lens>
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sAddRef(void* self_)
    { return fromLens(self_, Lens)->refs.fetch_add(1, std::memory_order_relaxed) + 1; }

    template <int Lens>
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sRelease(void* self_)
    {
        auto* view = fromLens(self_, Lens);
        const Steinberg_uint32 left = view->refs.fetch_sub(1, std::memory_order_acq_rel) - 1;
        if (left == 0) delete view;
        return left;
    }

    // --- IPlugView --------------------------------------------------------------------

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sIsPlatformTypeSupported(void*,
        Steinberg_FIDString type)
    {
        return (webview_ui::Editor<P>::available() && type != nullptr
                && std::strcmp(type, platformType()) == 0)
             ? Steinberg_kResultTrue : Steinberg_kResultFalse;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sAttached(void* self_, void* parent,
                                                              Steinberg_FIDString type)
    {
        auto* view = fromLens(self_, 0);
        webview_ui::debugLog("vst3 attached(parent=%p type=%s) negotiated=%dx%d scale=%.2f",
                             parent, type != nullptr ? type : "?",
                             view->width, view->height, view->scale);
        if (sIsPlatformTypeSupported(self_, type) != Steinberg_kResultTrue
            || parent == nullptr)
            return Steinberg_kResultFalse;
        const webview_ui::HostCallbacks callbacks {
            view, &cbSetParam, &cbBeginEdit, &cbEndEdit
        };
        if (!view->editor.create(parent, view->owner->shadow, callbacks))
            return Steinberg_kResultFalse;
        // Fill the box the host ACTUALLY built — hosts differ in whether
        // getSize/setContentScaleFactor/attached arrive in spec order, so the
        // parent's real client size wins over the negotiated one.
        int parentW = 0, parentH = 0;
        if (view->editor.queryParentSize(parentW, parentH))
        {
            view->width  = parentW;
            view->height = parentH;
        }
        view->editor.setBounds(view->width, view->height);
        view->editor.setVisible(true);
#if defined(__linux__)
        // GTK breathes only when pumped from the host's run loop; without a
        // usable IRunLoop the page would freeze, so fall back to generic UI.
        view->startTimer();
        if (!view->timerRegistered)
        {
            webview_ui::debugLog("vst3 attached: no usable IRunLoop -> no editor");
            view->editor.destroy();
            return Steinberg_kResultFalse;
        }
#endif
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sRemoved(void* self_)
    {
        auto* view = fromLens(self_, 0);
#if defined(__linux__)
        view->stopTimer();
#endif
        view->editor.destroy();
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnWheel(void*, float)
    { return Steinberg_kResultFalse; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnKeyDown(void*, Steinberg_char16,
                                                               Steinberg_int16, Steinberg_int16)
    { return Steinberg_kResultFalse; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnKeyUp(void*, Steinberg_char16,
                                                             Steinberg_int16, Steinberg_int16)
    { return Steinberg_kResultFalse; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetSize(void* self_,
                                                             Steinberg_ViewRect* size)
    {
        auto* view = fromLens(self_, 0);
        if (size == nullptr) return Steinberg_kInvalidArgument;
        size->left = 0;
        size->top = 0;
        size->right = view->width;
        size->bottom = view->height;
        webview_ui::debugLog("vst3 getSize -> %dx%d", view->width, view->height);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnSize(void* self_,
                                                            Steinberg_ViewRect* newSize)
    {
        auto* view = fromLens(self_, 0);
        if (newSize == nullptr) return Steinberg_kInvalidArgument;
        const double proposedW = newSize->right - newSize->left;
        const double proposedH = newSize->bottom - newSize->top;
        double w = proposedW;
        double h = proposedH;
        // Enforce the resize policy HERE, not only in checkSizeConstraint:
        // some hosts drag-resize freely and only report the result. When the
        // host overshot, apply the constrained size and ask it to follow.
        if (!view->correctingSize)
            constrainEditorSize<P>(w, h, view->scale);
        webview_ui::debugLog("vst3 onSize %.0fx%.0f -> %.0fx%.0f%s",
                             proposedW, proposedH, w, h,
                             view->correctingSize ? " (correction round-trip)" : "");
        view->width  = static_cast<int>(w + 0.5);
        view->height = static_cast<int>(h + 0.5);
        view->editor.setBounds(view->width, view->height);
        if (!view->correctingSize && view->frame != nullptr
            && (view->width - proposedW > 1.0 || proposedW - view->width > 1.0
                || view->height - proposedH > 1.0 || proposedH - view->height > 1.0))
        {
            view->correctingSize = true;
            Steinberg_ViewRect corrected { 0, 0, view->width, view->height };
            view->frame->lpVtbl->resizeView(view->frame,
                reinterpret_cast<Steinberg_IPlugView*>(view->lensPtr(0)), &corrected);
            view->correctingSize = false;
        }
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnFocus(void*, Steinberg_TBool)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetFrame(void* self_,
                                                              Steinberg_IPlugFrame* frame)
    {
        auto* view = fromLens(self_, 0);
#if defined(__linux__)
        view->releaseRunLoop();   // the run loop belongs to the old frame
#endif
        if (view->frame != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(view->frame)
                ->lpVtbl->release(view->frame);
        view->frame = frame;
        if (view->frame != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(view->frame)
                ->lpVtbl->addRef(view->frame);
#if defined(__linux__)
        if (view->editor.created())
            view->startTimer();   // frame arrived after attach (rare order)
#endif
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCanResize(void*)
    {
        return editorResizeOf<P>() != EditorResize::Fixed ? Steinberg_kResultTrue
                                                          : Steinberg_kResultFalse;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCheckSizeConstraint(void* self_,
        Steinberg_ViewRect* rect)
    {
        auto* view = fromLens(self_, 0);
        if (rect == nullptr) return Steinberg_kInvalidArgument;
        const double proposedW = rect->right - rect->left;
        const double proposedH = rect->bottom - rect->top;
        double w = proposedW;
        double h = proposedH;
        constrainEditorSize<P>(w, h, view->scale);
        rect->right  = rect->left + static_cast<Steinberg_int32>(w + 0.5);
        rect->bottom = rect->top + static_cast<Steinberg_int32>(h + 0.5);
        webview_ui::debugLog("vst3 checkSizeConstraint %.0fx%.0f -> %.0fx%.0f",
                             proposedW, proposedH, w, h);
        return Steinberg_kResultTrue;
    }

    inline static const Steinberg_IPlugViewVtbl kViewVtbl = {
        &sQuery<0>, &sAddRef<0>, &sRelease<0>,
        &sIsPlatformTypeSupported, &sAttached, &sRemoved,
        &sOnWheel, &sOnKeyDown, &sOnKeyUp,
        &sGetSize, &sOnSize, &sOnFocus,
        &sSetFrame, &sCanResize, &sCheckSizeConstraint
    };

    // --- IPlugViewContentScaleSupport ----------------------------------------------

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetContentScaleFactor(void* self_,
        Steinberg_IPlugViewContentScaleSupport_ScaleFactor factor)
    {
        auto* view = fromLens(self_, 1);
        if (factor <= 0.0f) return Steinberg_kInvalidArgument;
        const double previous = view->scale;
        view->scale = factor;
        webview_ui::debugLog("vst3 setContentScaleFactor %.2f (was %.2f)",
                             view->scale, previous);
        // Rescale the negotiated physical size (the page itself never zooms:
        // the web engine applies the window DPI to CSS pixels on its own).
        view->width  = static_cast<int>(view->width * (view->scale / previous) + 0.5);
        view->height = static_cast<int>(view->height * (view->scale / previous) + 0.5);
        if (view->editor.created() && view->frame != nullptr)
        {
            // Already on screen (e.g. dragged to another monitor): ask the
            // host to rebuild the window at the new physical size.
            Steinberg_ViewRect rect { 0, 0, view->width, view->height };
            view->frame->lpVtbl->resizeView(view->frame,
                reinterpret_cast<Steinberg_IPlugView*>(view->lensPtr(0)), &rect);
        }
        return Steinberg_kResultOk;
    }

    inline static const Steinberg_IPlugViewContentScaleSupportVtbl kScaleVtbl = {
        &sQuery<1>, &sAddRef<1>, &sRelease<1>,
        &sSetContentScaleFactor
    };

#if defined(__linux__)
    // Defined after the FUnknown thunks: a static member INITIALIZER has no
    // complete-class context, so every name it uses must be declared above.
    inline static const Steinberg_Linux_ITimerHandlerVtbl kTimerVtbl = {
        &sQuery<2>, &sAddRef<2>, &sRelease<2>, &sOnTimer
    };
#endif
};

#endif // DSPARK_PLUGIN_WEBVIEW

template <typename P>
Steinberg_IPlugView* SMTG_STDMETHODCALLTYPE Plugin<P>::sCreateView(void* self_,
                                                                   Steinberg_FIDString name)
{
    (void) self_;
    (void) name;
#if defined(DSPARK_PLUGIN_WEBVIEW)
    if constexpr (HasEditor<P>)
    {
        if (webview_ui::Editor<P>::available() && name != nullptr
            && std::strcmp(name, "editor") == 0)
        {
            auto* view = new (std::nothrow) View<P>(fromLens(self_, 2));
            if (view != nullptr)
                return reinterpret_cast<Steinberg_IPlugView*>(view->lensPtr(0));
        }
    }
#endif
    return nullptr;   // no editor: the host shows its generic parameter UI
}

// -- factory --------------------------------------------------------------------

template <typename P>
struct Factory
{
    const Steinberg_IPluginFactory3Vtbl* vtbl;

    Factory() noexcept { vtbl = &kVtbl; }

    static std::array<uint8_t, 16> classUid() noexcept
    {
        return makeUid(P::descriptor.productId, 0x56535433ull);   // 'VST3'
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sQuery(void* self_,
        const Steinberg_TUID iid, void** obj)
    {
        if (obj == nullptr) return Steinberg_kInvalidArgument;
        if (tuidEqual(iid, Steinberg_FUnknown_iid)
            || tuidEqual(iid, Steinberg_IPluginFactory_iid)
            || tuidEqual(iid, Steinberg_IPluginFactory2_iid)
            || tuidEqual(iid, Steinberg_IPluginFactory3_iid))
        {
            *obj = self_;
            return Steinberg_kResultOk;
        }
        *obj = nullptr;
        return Steinberg_kNoInterface;
    }

    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sAddRef(void*)  { return 100; }
    static Steinberg_uint32 SMTG_STDMETHODCALLTYPE sRelease(void*) { return 100; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetFactoryInfo(void*,
        Steinberg_PFactoryInfo* info)
    {
        if (info == nullptr) return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        copyAscii(P::descriptor.vendor, info->vendor, sizeof(info->vendor));
        copyAscii(P::descriptor.url, info->url, sizeof(info->url));
        copyAscii(P::descriptor.email, info->email, sizeof(info->email));
        info->flags = Steinberg_PFactoryInfo_FactoryFlags_kUnicode;
        return Steinberg_kResultOk;
    }

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sCountClasses(void*) { return 1; }

    static void fillClassCommon(Steinberg_TUID cid) noexcept
    {
        const auto uid = classUid();
        std::memcpy(cid, uid.data(), 16);
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetClassInfo(void*,
        Steinberg_int32 index, Steinberg_PClassInfo* info)
    {
        if (index != 0 || info == nullptr) return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        fillClassCommon(info->cid);
        info->cardinality = 0x7FFFFFFF;   // kManyInstances
        copyAscii("Audio Module Class", info->category, sizeof(info->category));
        copyAscii(P::descriptor.name, info->name, sizeof(info->name));
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetClassInfo2(void*,
        Steinberg_int32 index, Steinberg_PClassInfo2* info)
    {
        if (index != 0 || info == nullptr) return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        fillClassCommon(info->cid);
        info->cardinality = 0x7FFFFFFF;
        copyAscii("Audio Module Class", info->category, sizeof(info->category));
        copyAscii(P::descriptor.name, info->name, sizeof(info->name));
        info->classFlags = 0;
        copyAscii(P::descriptor.category == Category::Instrument ? "Instrument" : "Fx",
                  info->subCategories, sizeof(info->subCategories));
        copyAscii(P::descriptor.vendor, info->vendor, sizeof(info->vendor));
        copyAscii(P::descriptor.version, info->version, sizeof(info->version));
        copyAscii("VST 3.7.9", info->sdkVersion, sizeof(info->sdkVersion));
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetClassInfoUnicode(void*,
        Steinberg_int32 index, Steinberg_PClassInfoW* info)
    {
        if (index != 0 || info == nullptr) return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        fillClassCommon(info->cid);
        info->cardinality = 0x7FFFFFFF;
        copyAscii("Audio Module Class", info->category, sizeof(info->category));
        asciiToString128(P::descriptor.name, info->name);
        info->classFlags = 0;
        copyAscii(P::descriptor.category == Category::Instrument ? "Instrument" : "Fx",
                  info->subCategories, sizeof(info->subCategories));
        asciiToString128(P::descriptor.vendor, info->vendor);
        asciiToString128(P::descriptor.version, info->version);
        asciiToString128("VST 3.7.9", info->sdkVersion);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sCreateInstance(void*,
        Steinberg_FIDString cid, Steinberg_FIDString iid, void** obj)
    {
        if (cid == nullptr || iid == nullptr || obj == nullptr)
            return Steinberg_kInvalidArgument;
        *obj = nullptr;
        const auto uid = classUid();
        if (std::memcmp(cid, uid.data(), 16) != 0)
            return Steinberg_kNoInterface;

        auto* plugin = new (std::nothrow) Plugin<P>();
        if (plugin == nullptr) return Steinberg_kResultFalse;

        Steinberg_TUID requested;
        std::memcpy(requested, iid, sizeof(requested));
        const Steinberg_tresult r = plugin->query(requested, obj);
        plugin->releaseImpl();   // drop the construction reference
        return r == Steinberg_kResultOk ? Steinberg_kResultOk : Steinberg_kNoInterface;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetHostContext(void*, Steinberg_FUnknown*)
    { return Steinberg_kResultOk; }

    inline static const Steinberg_IPluginFactory3Vtbl kVtbl = {
        &sQuery, &sAddRef, &sRelease,
        &sGetFactoryInfo, &sCountClasses, &sGetClassInfo, &sCreateInstance,
        &sGetClassInfo2,
        &sGetClassInfoUnicode, &sSetHostContext
    };
};

} // namespace dspark::plugin::vst3

// -- module entry ----------------------------------------------------------------

#if defined(_WIN32)
#define DSPARK_VST3_EXPORT __declspec(dllexport)
#else
#define DSPARK_VST3_EXPORT __attribute__((visibility("default")))
#endif

/**
 * @brief Declares the VST3 module entry points for one plugin class.
 * Place once in exactly one translation unit, after the class definition.
 */
#define DSPARK_VST3_PLUGIN(PluginClass)                                            \
    static dspark::plugin::vst3::Factory<PluginClass> gDsparkVst3Factory;          \
    extern "C" {                                                                   \
    DSPARK_VST3_EXPORT Steinberg_IPluginFactory* SMTG_STDMETHODCALLTYPE            \
    GetPluginFactory()                                                             \
    {                                                                              \
        return reinterpret_cast<Steinberg_IPluginFactory*>(&gDsparkVst3Factory);   \
    }                                                                              \
    DSPARK_VST3_EXPORT bool InitDll() { return true; }                             \
    DSPARK_VST3_EXPORT bool ExitDll() { return true; }                             \
    DSPARK_VST3_EXPORT bool ModuleEntry(void*) { return true; }                    \
    DSPARK_VST3_EXPORT bool ModuleExit() { return true; }                          \
    DSPARK_VST3_EXPORT bool bundleEntry(void*) { return true; }                    \
    DSPARK_VST3_EXPORT bool bundleExit() { return true; }                          \
    }
