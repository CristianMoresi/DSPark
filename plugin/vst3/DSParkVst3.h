// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkVst3.h
 * @brief Native VST3 backend: a complete plugin from one DSPark plugin class.
 *
 * Implements the VST3 COM ABI directly against Steinberg's official C API
 * (vst3_c_api.h, vendored next to this header under Steinberg's permissive
 * 2025 license) — no VST3 C++ SDK to download, no JUCE, no build system
 * beyond your compiler:
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
 * - **Single-component effect**: one object implements IComponent,
 *   IAudioProcessor and IEditController (the layout hosts use for the vast
 *   majority of effects); the COM "lenses" are four consecutive vtable
 *   pointers and queryInterface hands out the right one.
 * - **Parameters**: host ParamIDs are FNV-1a hashes of the stable text ids
 *   (never indices: reordering parameters between versions must not break
 *   automation). A wrapper-owned Bypass parameter (kIsBypass) is always
 *   appended and applied as a short crossfade against the dry input.
 * - **Automation** is applied per block (last queue point) — every DSPark
 *   parameter setter smooths internally, so this is click-free by
 *   construction. Sample-accurate slicing is a planned v2 refinement.
 * - **State** uses the format-agnostic container of DSParkPlugin.h, so
 *   presets are byte-identical across the VST3/CLAP/AU backends.
 * - **Threading**: setParamNormalized arrives from the UI thread and
 *   process() from the audio thread; both funnel into the user's
 *   setParameter, which DSPark's contract requires to be atomic-based.
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

inline constexpr uint32_t kBypassParamId = 0x42595053u;   // 'BYPS'
inline constexpr int      kBypassRampSamples = 256;

template <typename P>
struct Plugin
{
    static constexpr size_t kNumParams = P::parameters.size();

    // COM lenses — four consecutive vtable pointers; queryInterface returns
    // the address of the matching slot. Standard layout is asserted below.
    const Steinberg_Vst_IComponentVtbl*       componentVtbl;
    const Steinberg_Vst_IAudioProcessorVtbl*  processorVtbl;
    const Steinberg_Vst_IEditControllerVtbl*  controllerVtbl;
    const Steinberg_Vst_IProcessContextRequirementsVtbl* contextReqVtbl;

    std::atomic<Steinberg_uint32> refs { 1 };

    P user {};
    Steinberg_Vst_ProcessSetup setup { 0, 0, 0, 48000.0 };
    bool prepared = false;
    int  cachedLatency = 0;

    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<bool>   bypass { false };
    float bypassMix = 0.0f;            // audio-thread crossfade state
    std::vector<float> dryL, dryR;     // pre-process copy for the bypass blend

    Steinberg_Vst_IComponentHandler* handler = nullptr;

    Plugin() noexcept
    {
        componentVtbl  = &kComponentVtbl;
        processorVtbl  = &kProcessorVtbl;
        controllerVtbl = &kControllerVtbl;
        contextReqVtbl = &kContextReqVtbl;
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

    void drainParameterChanges(Steinberg_Vst_IParameterChanges* changes) noexcept
    {
        if (changes == nullptr) return;
        const Steinberg_int32 n = changes->lpVtbl->getParameterCount(changes);
        for (Steinberg_int32 q = 0; q < n; ++q)
        {
            Steinberg_Vst_IParamValueQueue* queue = changes->lpVtbl->getParameterData(changes, q);
            if (queue == nullptr) continue;
            const Steinberg_int32 points = queue->lpVtbl->getPointCount(queue);
            if (points <= 0) continue;
            Steinberg_int32 offset = 0;
            Steinberg_Vst_ParamValue value = 0.0;
            if (queue->lpVtbl->getPoint(queue, points - 1, &offset, &value)
                != Steinberg_kResultOk)
                continue;
            const Steinberg_Vst_ParamID id = queue->lpVtbl->getParameterId(queue);
            if (id == kBypassParamId)
                bypass.store(value >= 0.5, std::memory_order_relaxed);
            else if (const int idx = indexOfParamId(id); idx >= 0)
                applyNormalized(idx, value);
        }
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

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sGetBusCount(void*,
        Steinberg_Vst_MediaType type, Steinberg_Vst_BusDirection)
    {
        return type == Steinberg_Vst_MediaTypes_kAudio ? 1 : 0;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetBusInfo(void*,
        Steinberg_Vst_MediaType type, Steinberg_Vst_BusDirection dir,
        Steinberg_int32 index, Steinberg_Vst_BusInfo* bus)
    {
        if (type != Steinberg_Vst_MediaTypes_kAudio || index != 0 || bus == nullptr)
            return Steinberg_kInvalidArgument;
        bus->mediaType = Steinberg_Vst_MediaTypes_kAudio;
        bus->direction = dir;
        bus->channelCount = 2;
        bus->busType = Steinberg_Vst_BusTypes_kMain;
        bus->flags = Steinberg_Vst_BusInfo_BusFlags_kDefaultActive;
        asciiToString128(dir == Steinberg_Vst_BusDirections_kInput ? "Input" : "Output",
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
            dspark::AudioSpec spec { sr, maxBlock, 2 };
            p->user.prepare(spec);
            p->applyAllShadows();
            p->dryL.assign(static_cast<size_t>(maxBlock), 0.0f);
            p->dryR.assign(static_cast<size_t>(maxBlock), 0.0f);
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
        if (!applyState(p->user, blob.data(), blob.size(), norm))
            return Steinberg_kResultFalse;
        for (size_t i = 0; i < kNumParams; ++i)
            p->applyNormalized(static_cast<int>(i), norm[i]);
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
        const std::vector<uint8_t> blob = buildState(p->user, norm, kNumParams);
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

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetBusArrangements(void*,
        Steinberg_Vst_SpeakerArrangement* inputs, Steinberg_int32 numIns,
        Steinberg_Vst_SpeakerArrangement* outputs, Steinberg_int32 numOuts)
    {
        if (numIns == 1 && numOuts == 1 && inputs != nullptr && outputs != nullptr
            && inputs[0] == Steinberg_Vst_SpeakerArr_kStereo
            && outputs[0] == Steinberg_Vst_SpeakerArr_kStereo)
            return Steinberg_kResultTrue;
        return Steinberg_kResultFalse;   // we stay stereo/stereo
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetBusArrangement(void*,
        Steinberg_Vst_BusDirection, Steinberg_int32 index,
        Steinberg_Vst_SpeakerArrangement* arr)
    {
        if (index != 0 || arr == nullptr) return Steinberg_kInvalidArgument;
        *arr = Steinberg_Vst_SpeakerArr_kStereo;
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

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sProcess(void* self_,
                                                             Steinberg_Vst_ProcessData* data)
    {
        auto* p = fromLens(self_, 1);
        if (data == nullptr) return Steinberg_kInvalidArgument;

        p->drainParameterChanges(data->inputParameterChanges);

        const Steinberg_int32 n = data->numSamples;
        if (n <= 0) return Steinberg_kResultOk;   // parameter flush
        if (data->symbolicSampleSize != Steinberg_Vst_SymbolicSampleSizes_kSample32)
            return Steinberg_kResultFalse;
        if (data->numOutputs < 1 || data->outputs == nullptr || !p->prepared)
            return Steinberg_kResultOk;

        auto& outBus = data->outputs[0];
        float** out = outBus.Steinberg_Vst_AudioBusBuffers_channelBuffers32;
        if (out == nullptr || outBus.numChannels < 1) return Steinberg_kResultOk;
        const int nCh = outBus.numChannels < 2 ? outBus.numChannels : 2;

        const bool haveIn = data->numInputs >= 1 && data->inputs != nullptr
            && data->inputs[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 != nullptr;
        float** in = haveIn
            ? data->inputs[0].Steinberg_Vst_AudioBusBuffers_channelBuffers32 : nullptr;

        // Keep the dry signal for the bypass blend, then process the output
        // buffers in place (copying input over first when distinct).
        float* dry[2] = { p->dryL.data(), p->dryR.data() };
        const size_t bytes = sizeof(float) * static_cast<size_t>(n);
        for (int ch = 0; ch < nCh; ++ch)
        {
            const float* src = (haveIn && in[ch] != nullptr) ? in[ch] : out[ch];
            if (static_cast<size_t>(n) <= p->dryL.size())
                std::memcpy(dry[ch], src, bytes);
            if (out[ch] != src)
                std::memcpy(out[ch], src, bytes);
        }

        dspark::AudioBufferView<float> view(out, nCh, n);
        p->user.processBlock(view);

        // Soft bypass: short linear crossfade toward the dry signal.
        const float target = p->bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (p->bypassMix != target || target > 0.0f)
        {
            const float step = 1.0f / static_cast<float>(kBypassRampSamples);
            float mix = p->bypassMix;
            if (static_cast<size_t>(n) <= p->dryL.size())
            {
                for (Steinberg_int32 i = 0; i < n; ++i)
                {
                    mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
                    mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
                    for (int ch = 0; ch < nCh; ++ch)
                        out[ch][i] += (dry[ch][i] - out[ch][i]) * mix;
                }
                p->bypassMix = mix;
            }
        }

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

    static Steinberg_int32 SMTG_STDMETHODCALLTYPE sGetParameterCount(void*)
    {
        return static_cast<Steinberg_int32>(kNumParams) + 1;   // + Bypass
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sGetParameterInfo(void* self_,
        Steinberg_int32 index, Steinberg_Vst_ParameterInfo* info)
    {
        (void) self_;
        if (info == nullptr) return Steinberg_kInvalidArgument;
        if (index < 0 || index > static_cast<Steinberg_int32>(kNumParams))
            return Steinberg_kInvalidArgument;
        std::memset(info, 0, sizeof(*info));
        if (index == static_cast<Steinberg_int32>(kNumParams))
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
        char text[64];
        if (id == kBypassParamId)
            std::snprintf(text, sizeof(text), "%s", normalized >= 0.5 ? "On" : "Off");
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
        const double plain = std::strtod(ascii, nullptr);
        if (id == kBypassParamId)
        {
            *normalized = plain >= 0.5 ? 1.0 : 0.0;
            return Steinberg_kResultOk;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return Steinberg_kInvalidArgument;
        *normalized = toNormalized(P::parameters[static_cast<size_t>(idx)], plain);
        return Steinberg_kResultOk;
    }

    static Steinberg_Vst_ParamValue SMTG_STDMETHODCALLTYPE sNormalizedParamToPlain(void*,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue normalized)
    {
        if (id == kBypassParamId) return normalized >= 0.5 ? 1.0 : 0.0;
        const int idx = indexOfParamId(id);
        return idx < 0 ? 0.0 : toPlain(P::parameters[static_cast<size_t>(idx)], normalized);
    }

    static Steinberg_Vst_ParamValue SMTG_STDMETHODCALLTYPE sPlainParamToNormalized(void*,
        Steinberg_Vst_ParamID id, Steinberg_Vst_ParamValue plain)
    {
        if (id == kBypassParamId) return plain >= 0.5 ? 1.0 : 0.0;
        const int idx = indexOfParamId(id);
        return idx < 0 ? 0.0 : toNormalized(P::parameters[static_cast<size_t>(idx)], plain);
    }

    static Steinberg_Vst_ParamValue SMTG_STDMETHODCALLTYPE sGetParamNormalized(void* self_,
        Steinberg_Vst_ParamID id)
    {
        auto* p = fromLens(self_, 2);
        if (id == kBypassParamId)
            return p->bypass.load(std::memory_order_relaxed) ? 1.0 : 0.0;
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
        const int idx = indexOfParamId(id);
        if (idx < 0) return Steinberg_kInvalidArgument;
        p->applyNormalized(idx, value);   // user setters are atomic by contract
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
        return 0;   // v1 ignores the transport; nothing required
    }

    inline static const Steinberg_Vst_IProcessContextRequirementsVtbl kContextReqVtbl = {
        &sQuery<3>, &sAddRef<3>, &sRelease<3>,
        &sGetProcessContextRequirements
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

    // Two COM lenses, same layout trick as the plugin object.
    const Steinberg_IPlugViewVtbl*                    viewVtbl;
    const Steinberg_IPlugViewContentScaleSupportVtbl* scaleVtbl;

    std::atomic<Steinberg_uint32> refs { 1 };
    Plugin<P>* owner;
    Steinberg_IPlugFrame* frame = nullptr;
    webview_ui::Editor<P> editor;
    int width;                 // current physical size (host units)
    int height;
    double scale = 1.0;
    bool editActive[kNumParams == 0 ? 1 : kNumParams] {};

    explicit View(Plugin<P>* plugin) noexcept : owner(plugin)
    {
        viewVtbl  = &kViewVtbl;
        scaleVtbl = &kScaleVtbl;
        owner->addRefImpl();
        const EditorSize logical = editorSizeOf<P>();
        width  = logical.width;
        height = logical.height;
    }

    ~View()
    {
        editor.destroy();
        if (frame != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(frame)->lpVtbl->release(frame);
        owner->releaseImpl();
    }

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
        return (webview_ui::Editor<P>::kAvailable && type != nullptr
                && std::strcmp(type, platformType()) == 0)
             ? Steinberg_kResultTrue : Steinberg_kResultFalse;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sAttached(void* self_, void* parent,
                                                              Steinberg_FIDString type)
    {
        auto* view = fromLens(self_, 0);
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
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sRemoved(void* self_)
    {
        fromLens(self_, 0)->editor.destroy();
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
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnSize(void* self_,
                                                            Steinberg_ViewRect* newSize)
    {
        auto* view = fromLens(self_, 0);
        if (newSize == nullptr) return Steinberg_kInvalidArgument;
        view->width  = newSize->right - newSize->left;
        view->height = newSize->bottom - newSize->top;
        view->editor.setBounds(view->width, view->height);
        return Steinberg_kResultOk;
    }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sOnFocus(void*, Steinberg_TBool)
    { return Steinberg_kResultOk; }

    static Steinberg_tresult SMTG_STDMETHODCALLTYPE sSetFrame(void* self_,
                                                              Steinberg_IPlugFrame* frame)
    {
        auto* view = fromLens(self_, 0);
        if (view->frame != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(view->frame)
                ->lpVtbl->release(view->frame);
        view->frame = frame;
        if (view->frame != nullptr)
            reinterpret_cast<Steinberg_FUnknown*>(view->frame)
                ->lpVtbl->addRef(view->frame);
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
        const EditorSize logical = editorSizeOf<P>();
        constexpr EditorResize mode = editorResizeOf<P>();
        if constexpr (mode == EditorResize::Fixed)
        {
            rect->right  = rect->left + static_cast<Steinberg_int32>(
                logical.width * view->scale + 0.5);
            rect->bottom = rect->top + static_cast<Steinberg_int32>(
                logical.height * view->scale + 0.5);
        }
        else
        {
            const double minW = logical.width * view->scale * kEditorMinSizeFactor;
            const double maxW = logical.width * view->scale * kEditorMaxSizeFactor;
            const double minH = logical.height * view->scale * kEditorMinSizeFactor;
            const double maxH = logical.height * view->scale * kEditorMaxSizeFactor;
            double w = rect->right - rect->left;
            double h = rect->bottom - rect->top;
            w = w < minW ? minW : (w > maxW ? maxW : w);
            h = h < minH ? minH : (h > maxH ? maxH : h);
            if constexpr (mode == EditorResize::KeepAspect)
            {
                // The proposed width drives; height follows the declared ratio.
                h = w * logical.height / logical.width;
            }
            rect->right  = rect->left + static_cast<Steinberg_int32>(w + 0.5);
            rect->bottom = rect->top + static_cast<Steinberg_int32>(h + 0.5);
        }
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
        if (webview_ui::Editor<P>::kAvailable && name != nullptr
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
