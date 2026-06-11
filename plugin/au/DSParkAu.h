// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSParkAu.h
 * @brief Native Audio Unit v2 backend: the same plugin class, for Logic Pro.
 *
 * Implements the AUv2 component ABI directly against Apple's AudioToolbox
 * (system headers — nothing to vendor or download). AUv2 remains the format
 * desktop hosts load (Logic Pro, GarageBand, MainStage, Reaper/Live on
 * macOS); AUv3's app-extension model is out of scope for a C++ framework.
 *
 * ```cpp
 * #include "plugin/au/DSParkAu.h"     // self-disables off-macOS
 * struct MyPlugin { ... };
 * DSPARK_AU_PLUGIN(MyPlugin, "Subt", "Manu")   // unique 4-char codes
 * ```
 *
 * The bundle side (Info.plist `AudioComponents` entry pointing at the
 * exported factory, `.component` folder layout, ad-hoc codesign) is produced
 * by dspark_add_plugin(); `auval -v aufx Subt Manu` is the release gate and
 * runs in this repository's macOS CI.
 *
 * Mapping notes:
 * - Parameter IDs are the same FNV-1a hashes of the stable text ids used by
 *   the VST3/CLAP backends; values travel in PLAIN units (AU convention).
 * - State (kAudioUnitProperty_ClassInfo) wraps the shared format-agnostic
 *   container in the standard aupreset dictionary, so presets remain
 *   portable across all three backends.
 * - Host bypass maps to kAudioUnitProperty_BypassEffect with the same
 *   crossfade as the other backends.
 * - Input arrives by pulling the host's render callback (or connection),
 *   the universal aufx pattern auval exercises.
 */

#if defined(__APPLE__)

#include "../DSParkPlugin.h"

#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>

#include <atomic>
#include <cstring>
#include <new>
#include <vector>

namespace dspark::plugin::au {

inline constexpr uint32_t kBypassParamId = 0x42595053u;   // matches VST3/CLAP
inline constexpr int      kBypassRampSamples = 256;

inline OSType fourCC(const char* s) noexcept
{
    return (static_cast<OSType>(static_cast<unsigned char>(s[0])) << 24)
         | (static_cast<OSType>(static_cast<unsigned char>(s[1])) << 16)
         | (static_cast<OSType>(static_cast<unsigned char>(s[2])) << 8)
         |  static_cast<OSType>(static_cast<unsigned char>(s[3]));
}

template <typename P>
struct Plugin
{
    static constexpr size_t kNumParams = P::parameters.size();

    AudioComponentInstance instance = nullptr;
    OSType subtype = 0, manufacturer = 0;

    double sampleRate = 44100.0;
    UInt32 maxFrames = 1156;            // CoreAudio's historical default
    bool initialized = false;

    P user {};
    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<bool>   bypass { false };
    float bypassMix = 0.0f;
    int cachedLatency = 0;

    AURenderCallbackStruct inputCallback {};
    AudioUnit inputConnection = nullptr;
    UInt32 inputConnectionBus = 0;

    std::vector<float> pullL, pullR, dryL, dryR;

    // Property listeners (registered/fired on the host's main thread) and
    // the user-preset name AU hosts expect through kAudioUnitProperty_PresentPreset.
    struct Listener
    {
        AudioUnitPropertyID id;
        AudioUnitPropertyListenerProc proc;
        void* user;
    };
    std::vector<Listener> listeners;
    CFStringRef presetName = nullptr;

    ~Plugin() noexcept
    {
        if (presetName != nullptr) CFRelease(presetName);
    }

    void notifyProperty(AudioUnitPropertyID id, AudioUnitScope scope,
                        AudioUnitElement element) noexcept
    {
        for (const auto& l : listeners)
            if (l.id == id && l.proc != nullptr)
                l.proc(l.user, instance, id, scope, element);
    }

    Plugin() noexcept
    {
        for (size_t i = 0; i < kNumParams; ++i)
            shadow[i].store(toNormalized(P::parameters[i], P::parameters[i].defValue),
                            std::memory_order_relaxed);
    }

    static int indexOfParamId(AudioUnitParameterID id) noexcept
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

    // --- lifecycle ---------------------------------------------------------------

    OSStatus initialize() noexcept
    {
        dspark::AudioSpec spec { sampleRate, static_cast<int>(maxFrames), 2 };
        user.prepare(spec);
        applyAllShadows();
        pullL.assign(maxFrames, 0.0f);
        pullR.assign(maxFrames, 0.0f);
        dryL.assign(maxFrames, 0.0f);
        dryR.assign(maxFrames, 0.0f);
        bypassMix = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if constexpr (HasLatency<P>)
            cachedLatency = user.getLatency();
        initialized = true;
        return noErr;
    }

    OSStatus uninitialize() noexcept
    {
        initialized = false;
        return noErr;
    }

    OSStatus reset() noexcept
    {
        if constexpr (HasReset<P>)
            user.reset();
        bypassMix = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        return noErr;
    }

    // --- stream format helpers -----------------------------------------------------

    void fillStreamFormat(AudioStreamBasicDescription& f) const noexcept
    {
        std::memset(&f, 0, sizeof(f));
        f.mSampleRate = sampleRate;
        f.mFormatID = kAudioFormatLinearPCM;
        f.mFormatFlags = static_cast<AudioFormatFlags>(kAudioFormatFlagsNativeFloatPacked)
                       | static_cast<AudioFormatFlags>(kAudioFormatFlagIsNonInterleaved);
        f.mBytesPerPacket = sizeof(float);
        f.mFramesPerPacket = 1;
        f.mBytesPerFrame = sizeof(float);
        f.mChannelsPerFrame = 2;
        f.mBitsPerChannel = 32;
    }

    bool acceptableFormat(const AudioStreamBasicDescription& f) const noexcept
    {
        return f.mFormatID == kAudioFormatLinearPCM
            && (f.mFormatFlags & kAudioFormatFlagIsFloat) != 0
            && (f.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
            && f.mChannelsPerFrame == 2
            && f.mBitsPerChannel == 32
            && f.mSampleRate > 0.0;
    }

    // --- properties -------------------------------------------------------------------

    OSStatus getPropertyInfo(AudioUnitPropertyID id, AudioUnitScope scope,
                             AudioUnitElement element, UInt32* outSize,
                             Boolean* outWritable) noexcept
    {
        UInt32 size = 0;
        Boolean writable = false;
        switch (id)
        {
        case kAudioUnitProperty_StreamFormat:
            size = sizeof(AudioStreamBasicDescription); writable = true; break;
        case kAudioUnitProperty_SampleRate:
            size = sizeof(Float64); writable = true; break;
        case kAudioUnitProperty_MaximumFramesPerSlice:
            size = sizeof(UInt32); writable = true; break;
        case kAudioUnitProperty_ElementCount:
            size = sizeof(UInt32); break;
        case kAudioUnitProperty_SupportedNumChannels:
            size = sizeof(AUChannelInfo); break;
        case kAudioUnitProperty_ParameterList:
            size = scope == kAudioUnitScope_Global
                 ? static_cast<UInt32>((kNumParams + 1) * sizeof(AudioUnitParameterID))
                 : 0;
            break;
        case kAudioUnitProperty_ParameterInfo:
            size = sizeof(AudioUnitParameterInfo); break;
        case kAudioUnitProperty_Latency:
        case kAudioUnitProperty_TailTime:
            size = sizeof(Float64); break;
        case kAudioUnitProperty_BypassEffect:
            size = sizeof(UInt32); writable = true; break;
        case kAudioUnitProperty_ClassInfo:
            size = sizeof(CFPropertyListRef); writable = true; break;
        case kAudioUnitProperty_SetRenderCallback:
            size = sizeof(AURenderCallbackStruct); writable = true; break;
        case kAudioUnitProperty_MakeConnection:
            size = sizeof(AudioUnitConnection); writable = true; break;
        case kAudioUnitProperty_InPlaceProcessing:
            size = sizeof(UInt32); break;
        case kAudioUnitProperty_PresentPreset:
        case kAudioUnitProperty_CurrentPreset:
            size = sizeof(AUPreset); writable = true; break;
        default:
            (void) element;
            return kAudioUnitErr_InvalidProperty;
        }
        if (outSize) *outSize = size;
        if (outWritable) *outWritable = writable;
        return noErr;
    }

    OSStatus getProperty(AudioUnitPropertyID id, AudioUnitScope scope,
                         AudioUnitElement element, void* outData,
                         UInt32* ioSize) noexcept
    {
        if (ioSize == nullptr) return kAudio_ParamError;
        if (outData == nullptr)
        {
            Boolean w = false;
            return getPropertyInfo(id, scope, element, ioSize, &w);
        }
        switch (id)
        {
        case kAudioUnitProperty_StreamFormat:
        {
            if (*ioSize < sizeof(AudioStreamBasicDescription))
                return kAudioUnitErr_InvalidPropertyValue;
            fillStreamFormat(*static_cast<AudioStreamBasicDescription*>(outData));
            *ioSize = sizeof(AudioStreamBasicDescription);
            return noErr;
        }
        case kAudioUnitProperty_SampleRate:
            if (*ioSize < sizeof(Float64)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<Float64*>(outData) = sampleRate;
            *ioSize = sizeof(Float64);
            return noErr;
        case kAudioUnitProperty_MaximumFramesPerSlice:
            if (*ioSize < sizeof(UInt32)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<UInt32*>(outData) = maxFrames;
            *ioSize = sizeof(UInt32);
            return noErr;
        case kAudioUnitProperty_ElementCount:
            if (*ioSize < sizeof(UInt32)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<UInt32*>(outData) = 1;
            *ioSize = sizeof(UInt32);
            return noErr;
        case kAudioUnitProperty_SupportedNumChannels:
        {
            if (*ioSize < sizeof(AUChannelInfo))
                return kAudioUnitErr_InvalidPropertyValue;
            auto* info = static_cast<AUChannelInfo*>(outData);
            info->inChannels = 2;
            info->outChannels = 2;
            *ioSize = sizeof(AUChannelInfo);
            return noErr;
        }
        case kAudioUnitProperty_ParameterList:
        {
            if (scope != kAudioUnitScope_Global)
            {
                *ioSize = 0;
                return noErr;
            }
            const UInt32 want =
                static_cast<UInt32>((kNumParams + 1) * sizeof(AudioUnitParameterID));
            if (*ioSize < want) return kAudioUnitErr_InvalidPropertyValue;
            auto* ids = static_cast<AudioUnitParameterID*>(outData);
            for (size_t i = 0; i < kNumParams; ++i)
                ids[i] = hash32(P::parameters[i].id);
            ids[kNumParams] = kBypassParamId;
            *ioSize = want;
            return noErr;
        }
        case kAudioUnitProperty_ParameterInfo:
        {
            if (*ioSize < sizeof(AudioUnitParameterInfo))
                return kAudioUnitErr_InvalidPropertyValue;
            auto* info = static_cast<AudioUnitParameterInfo*>(outData);
            std::memset(info, 0, sizeof(*info));
            info->flags = kAudioUnitParameterFlag_IsReadable
                        | kAudioUnitParameterFlag_IsWritable
                        | kAudioUnitParameterFlag_HasCFNameString
                        | kAudioUnitParameterFlag_CFNameRelease;
            if (element == kBypassParamId)
            {
                std::snprintf(info->name, sizeof(info->name), "Bypass");
                info->cfNameString = CFStringCreateWithCString(
                    nullptr, "Bypass", kCFStringEncodingUTF8);
                info->unit = kAudioUnitParameterUnit_Boolean;
                info->minValue = 0.0f;
                info->maxValue = 1.0f;
                info->defaultValue = 0.0f;
                *ioSize = sizeof(AudioUnitParameterInfo);
                return noErr;
            }
            const int idx = indexOfParamId(element);
            if (idx < 0) return kAudioUnitErr_InvalidParameter;
            const auto& spec = P::parameters[static_cast<size_t>(idx)];
            std::snprintf(info->name, sizeof(info->name), "%s", spec.name);
            info->cfNameString = CFStringCreateWithCString(
                nullptr, spec.name, kCFStringEncodingUTF8);
            info->unit = spec.steps == 1 ? kAudioUnitParameterUnit_Boolean
                       : (std::strcmp(spec.unit, "dB") == 0
                              ? kAudioUnitParameterUnit_Decibels
                              : kAudioUnitParameterUnit_Generic);
            info->minValue = spec.minValue;
            info->maxValue = spec.maxValue;
            info->defaultValue = spec.defValue;
            *ioSize = sizeof(AudioUnitParameterInfo);
            return noErr;
        }
        case kAudioUnitProperty_Latency:
            if (*ioSize < sizeof(Float64)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<Float64*>(outData) =
                sampleRate > 0.0 ? cachedLatency / sampleRate : 0.0;
            *ioSize = sizeof(Float64);
            return noErr;
        case kAudioUnitProperty_TailTime:
        {
            if (*ioSize < sizeof(Float64)) return kAudioUnitErr_InvalidPropertyValue;
            double tail = 0.0;
            if constexpr (HasTail<P>)
                tail = user.getTailSeconds();
            *static_cast<Float64*>(outData) = tail;
            *ioSize = sizeof(Float64);
            return noErr;
        }
        case kAudioUnitProperty_BypassEffect:
            if (*ioSize < sizeof(UInt32)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<UInt32*>(outData) =
                bypass.load(std::memory_order_relaxed) ? 1 : 0;
            *ioSize = sizeof(UInt32);
            return noErr;
        case kAudioUnitProperty_InPlaceProcessing:
            if (*ioSize < sizeof(UInt32)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<UInt32*>(outData) = 1;
            *ioSize = sizeof(UInt32);
            return noErr;
        case kAudioUnitProperty_PresentPreset:
        case kAudioUnitProperty_CurrentPreset:
        {
            if (*ioSize < sizeof(AUPreset)) return kAudioUnitErr_InvalidPropertyValue;
            auto* preset = static_cast<AUPreset*>(outData);
            preset->presetNumber = -1;   // user preset
            preset->presetName = presetName != nullptr
                ? static_cast<CFStringRef>(CFRetain(presetName))
                : CFStringCreateWithCString(nullptr, "Untitled",
                                            kCFStringEncodingUTF8);
            *ioSize = sizeof(AUPreset);
            return noErr;
        }
        case kAudioUnitProperty_ClassInfo:
        {
            if (*ioSize < sizeof(CFPropertyListRef))
                return kAudioUnitErr_InvalidPropertyValue;
            double norm[kNumParams == 0 ? 1 : kNumParams];
            for (size_t i = 0; i < kNumParams; ++i)
                norm[i] = shadow[i].load(std::memory_order_relaxed);
            const std::vector<uint8_t> blob = buildState(user, norm, kNumParams);

            CFMutableDictionaryRef dict = CFDictionaryCreateMutable(
                nullptr, 0, &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks);
            auto putInt = [&](const char* key, SInt32 v) {
                CFNumberRef n = CFNumberCreate(nullptr, kCFNumberSInt32Type, &v);
                CFStringRef k = CFStringCreateWithCString(nullptr, key,
                                                          kCFStringEncodingUTF8);
                CFDictionarySetValue(dict, k, n);
                CFRelease(k);
                CFRelease(n);
            };
            putInt("version", 0);
            putInt("type", static_cast<SInt32>(fourCC("aufx")));
            putInt("subtype", static_cast<SInt32>(subtype));
            putInt("manufacturer", static_cast<SInt32>(manufacturer));
            CFStringRef nameKey = CFSTR("name");
            CFStringRef nameVal = CFStringCreateWithCString(
                nullptr, P::descriptor.name, kCFStringEncodingUTF8);
            CFDictionarySetValue(dict, nameKey, nameVal);
            CFRelease(nameVal);
            CFDataRef data = CFDataCreate(nullptr, blob.data(),
                                          static_cast<CFIndex>(blob.size()));
            CFDictionarySetValue(dict, CFSTR("dspark-state"), data);
            CFRelease(data);

            *static_cast<CFPropertyListRef*>(outData) = dict;   // caller releases
            *ioSize = sizeof(CFPropertyListRef);
            return noErr;
        }
        default:
            return kAudioUnitErr_InvalidProperty;
        }
    }

    OSStatus setProperty(AudioUnitPropertyID id, AudioUnitScope scope,
                         AudioUnitElement element, const void* inData,
                         UInt32 inSize) noexcept
    {
        (void) element;
        switch (id)
        {
        case kAudioUnitProperty_StreamFormat:
        {
            if (inData == nullptr || inSize < sizeof(AudioStreamBasicDescription))
                return kAudioUnitErr_InvalidPropertyValue;
            const auto* f = static_cast<const AudioStreamBasicDescription*>(inData);
            if (!acceptableFormat(*f)) return kAudioUnitErr_FormatNotSupported;
            sampleRate = f->mSampleRate;
            return noErr;
        }
        case kAudioUnitProperty_SampleRate:
            if (inData == nullptr || inSize < sizeof(Float64))
                return kAudioUnitErr_InvalidPropertyValue;
            sampleRate = *static_cast<const Float64*>(inData);
            return noErr;
        case kAudioUnitProperty_MaximumFramesPerSlice:
            if (inData == nullptr || inSize < sizeof(UInt32))
                return kAudioUnitErr_InvalidPropertyValue;
            maxFrames = *static_cast<const UInt32*>(inData);
            notifyProperty(id, scope, element);
            return noErr;
        case kAudioUnitProperty_PresentPreset:
        case kAudioUnitProperty_CurrentPreset:
        {
            if (inData == nullptr || inSize < sizeof(AUPreset))
                return kAudioUnitErr_InvalidPropertyValue;
            const auto* preset = static_cast<const AUPreset*>(inData);
            if (preset->presetNumber >= 0)
                return kAudioUnitErr_InvalidPropertyValue;   // no factory presets
            if (presetName != nullptr) CFRelease(presetName);
            presetName = preset->presetName != nullptr
                ? static_cast<CFStringRef>(CFRetain(preset->presetName))
                : nullptr;
            notifyProperty(id, scope, element);
            return noErr;
        }
        case kAudioUnitProperty_BypassEffect:
            if (inData == nullptr || inSize < sizeof(UInt32))
                return kAudioUnitErr_InvalidPropertyValue;
            bypass.store(*static_cast<const UInt32*>(inData) != 0,
                         std::memory_order_relaxed);
            return noErr;
        case kAudioUnitProperty_SetRenderCallback:
            if (scope != kAudioUnitScope_Input || inData == nullptr
                || inSize < sizeof(AURenderCallbackStruct))
                return kAudioUnitErr_InvalidPropertyValue;
            inputCallback = *static_cast<const AURenderCallbackStruct*>(inData);
            inputConnection = nullptr;
            return noErr;
        case kAudioUnitProperty_MakeConnection:
        {
            if (inData == nullptr || inSize < sizeof(AudioUnitConnection))
                return kAudioUnitErr_InvalidPropertyValue;
            const auto* c = static_cast<const AudioUnitConnection*>(inData);
            inputConnection = c->sourceAudioUnit;
            inputConnectionBus = c->sourceOutputNumber;
            inputCallback = {};
            return noErr;
        }
        case kAudioUnitProperty_ClassInfo:
        {
            if (inData == nullptr || inSize < sizeof(CFPropertyListRef))
                return kAudioUnitErr_InvalidPropertyValue;
            CFPropertyListRef plist = *static_cast<const CFPropertyListRef*>(inData);
            if (plist == nullptr || CFGetTypeID(plist) != CFDictionaryGetTypeID())
                return kAudioUnitErr_InvalidPropertyValue;
            auto dict = static_cast<CFDictionaryRef>(plist);
            auto data = static_cast<CFDataRef>(
                CFDictionaryGetValue(dict, CFSTR("dspark-state")));
            if (data == nullptr || CFGetTypeID(data) != CFDataGetTypeID())
                return noErr;   // foreign preset: keep current state
            double norm[kNumParams == 0 ? 1 : kNumParams];
            for (size_t i = 0; i < kNumParams; ++i)
                norm[i] = shadow[i].load(std::memory_order_relaxed);
            if (applyState(user,
                           CFDataGetBytePtr(data),
                           static_cast<size_t>(CFDataGetLength(data)), norm))
                for (size_t i = 0; i < kNumParams; ++i)
                    applyNormalized(static_cast<int>(i), norm[i]);
            return noErr;
        }
        default:
            return kAudioUnitErr_InvalidProperty;
        }
    }

    // --- parameters ----------------------------------------------------------------

    OSStatus getParameter(AudioUnitParameterID id, AudioUnitScope scope,
                          AudioUnitParameterValue* outValue) noexcept
    {
        if (scope != kAudioUnitScope_Global || outValue == nullptr)
            return kAudioUnitErr_InvalidParameter;
        if (id == kBypassParamId)
        {
            *outValue = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
            return noErr;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return kAudioUnitErr_InvalidParameter;
        *outValue = static_cast<AudioUnitParameterValue>(
            toPlain(P::parameters[static_cast<size_t>(idx)],
                    shadow[static_cast<size_t>(idx)].load(std::memory_order_relaxed)));
        return noErr;
    }

    OSStatus setParameter(AudioUnitParameterID id, AudioUnitScope scope,
                          AudioUnitParameterValue value) noexcept
    {
        if (scope != kAudioUnitScope_Global) return kAudioUnitErr_InvalidParameter;
        if (id == kBypassParamId)
        {
            bypass.store(value >= 0.5f, std::memory_order_relaxed);
            return noErr;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return kAudioUnitErr_InvalidParameter;
        applyNormalized(idx,
            toNormalized(P::parameters[static_cast<size_t>(idx)], value));
        return noErr;
    }

    // --- render ----------------------------------------------------------------------

    OSStatus render(AudioUnitRenderActionFlags* ioFlags,
                    const AudioTimeStamp* timeStamp, UInt32 busNumber,
                    UInt32 frames, AudioBufferList* ioData) noexcept
    {
        if (ioData == nullptr || frames == 0) return noErr;
        if (!initialized) return kAudioUnitErr_Uninitialized;
        if (frames > maxFrames) return kAudioUnitErr_TooManyFramesToProcess;

        // Pull the input through the registered callback or connection into
        // our own buffers, then process and write to the host's buffers.
        float* pull[2] = { pullL.data(), pullR.data() };
        AudioBufferList pullList;
        pullList.mNumberBuffers = 2;
        for (int ch = 0; ch < 2; ++ch)
        {
            pullList.mBuffers[ch].mNumberChannels = 1;
            pullList.mBuffers[ch].mDataByteSize = frames * sizeof(float);
            pullList.mBuffers[ch].mData = pull[ch];
        }

        OSStatus pullStatus = noErr;
        AudioUnitRenderActionFlags pullFlags = 0;
        if (inputCallback.inputProc != nullptr)
            pullStatus = inputCallback.inputProc(inputCallback.inputProcRefCon,
                                                 &pullFlags, timeStamp, busNumber,
                                                 frames, &pullList);
        else if (inputConnection != nullptr)
            pullStatus = AudioUnitRender(inputConnection, &pullFlags, timeStamp,
                                         inputConnectionBus, frames, &pullList);
        else
            for (int ch = 0; ch < 2; ++ch)
                std::memset(pull[ch], 0, frames * sizeof(float));
        if (pullStatus != noErr) return pullStatus;

        // The callback may have substituted its own pointers.
        for (UInt32 ch = 0; ch < 2 && ch < pullList.mNumberBuffers; ++ch)
            if (pullList.mBuffers[ch].mData != nullptr)
                pull[ch] = static_cast<float*>(pullList.mBuffers[ch].mData);

        const UInt32 outCh = ioData->mNumberBuffers < 2 ? ioData->mNumberBuffers : 2;
        float* out[2] = { nullptr, nullptr };
        for (UInt32 ch = 0; ch < outCh; ++ch)
        {
            if (ioData->mBuffers[ch].mData == nullptr)
                ioData->mBuffers[ch].mData = pull[ch];
            out[ch] = static_cast<float*>(ioData->mBuffers[ch].mData);
            ioData->mBuffers[ch].mDataByteSize = frames * sizeof(float);
        }

        float* dry[2] = { dryL.data(), dryR.data() };
        for (UInt32 ch = 0; ch < outCh; ++ch)
        {
            std::memcpy(dry[ch], pull[ch], frames * sizeof(float));
            if (out[ch] != pull[ch])
                std::memcpy(out[ch], pull[ch], frames * sizeof(float));
        }

        dspark::AudioBufferView<float> view(out, static_cast<int>(outCh),
                                            static_cast<int>(frames));
        user.processBlock(view);

        const float target = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (bypassMix != target || target > 0.0f)
        {
            const float step = 1.0f / static_cast<float>(kBypassRampSamples);
            float mix = bypassMix;
            for (UInt32 i = 0; i < frames; ++i)
            {
                mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
                mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
                for (UInt32 ch = 0; ch < outCh; ++ch)
                    out[ch][i] += (dry[ch][i] - out[ch][i]) * mix;
            }
            bypassMix = mix;
        }

        (void) ioFlags;
        return noErr;
    }
};

// -- component plug-in interface -----------------------------------------------------

template <typename P>
struct Component
{
    AudioComponentPlugInInterface iface {};
    Plugin<P>* state = nullptr;

    static Component* fromSelf(void* self) noexcept
    {
        return static_cast<Component*>(self);
    }

    static OSStatus sOpen(void* self, AudioComponentInstance instance) noexcept
    {
        auto* c = fromSelf(self);
        c->state = new (std::nothrow) Plugin<P>();
        if (c->state == nullptr) return kAudio_MemFullError;
        c->state->instance = instance;
        c->state->subtype = gSubtype;
        c->state->manufacturer = gManufacturer;
        return noErr;
    }

    static OSStatus sClose(void* self) noexcept
    {
        auto* c = fromSelf(self);
        delete c->state;
        delete c;
        return noErr;
    }

    // Selector implementations (signatures from AUComponent.h).

    static OSStatus sInitialize(void* self) noexcept
    { return fromSelf(self)->state->initialize(); }

    static OSStatus sUninitialize(void* self) noexcept
    { return fromSelf(self)->state->uninitialize(); }

    static OSStatus sGetPropertyInfo(void* self, AudioUnitPropertyID id,
                                     AudioUnitScope scope, AudioUnitElement element,
                                     UInt32* outSize, Boolean* outWritable) noexcept
    { return fromSelf(self)->state->getPropertyInfo(id, scope, element, outSize, outWritable); }

    static OSStatus sGetProperty(void* self, AudioUnitPropertyID id,
                                 AudioUnitScope scope, AudioUnitElement element,
                                 void* outData, UInt32* ioSize) noexcept
    { return fromSelf(self)->state->getProperty(id, scope, element, outData, ioSize); }

    static OSStatus sSetProperty(void* self, AudioUnitPropertyID id,
                                 AudioUnitScope scope, AudioUnitElement element,
                                 const void* inData, UInt32 inSize) noexcept
    { return fromSelf(self)->state->setProperty(id, scope, element, inData, inSize); }

    static OSStatus sGetParameter(void* self, AudioUnitParameterID id,
                                  AudioUnitScope scope, AudioUnitElement element,
                                  AudioUnitParameterValue* outValue) noexcept
    {
        (void) element;
        return fromSelf(self)->state->getParameter(id, scope, outValue);
    }

    static OSStatus sSetParameter(void* self, AudioUnitParameterID id,
                                  AudioUnitScope scope, AudioUnitElement element,
                                  AudioUnitParameterValue value,
                                  UInt32 bufferOffset) noexcept
    {
        (void) element;
        (void) bufferOffset;
        return fromSelf(self)->state->setParameter(id, scope, value);
    }

    static OSStatus sReset(void* self, AudioUnitScope scope,
                           AudioUnitElement element) noexcept
    {
        (void) scope;
        (void) element;
        return fromSelf(self)->state->reset();
    }

    static OSStatus sRender(void* self, AudioUnitRenderActionFlags* ioFlags,
                            const AudioTimeStamp* timeStamp, UInt32 busNumber,
                            UInt32 frames, AudioBufferList* ioData) noexcept
    { return fromSelf(self)->state->render(ioFlags, timeStamp, busNumber, frames, ioData); }

    static OSStatus sAddPropertyListener(void* self, AudioUnitPropertyID id,
                                         AudioUnitPropertyListenerProc proc,
                                         void* user) noexcept
    {
        auto* state = fromSelf(self)->state;
        state->listeners.push_back({ id, proc, user });
        return noErr;
    }

    static OSStatus sRemovePropertyListener(void* self, AudioUnitPropertyID id,
                                            AudioUnitPropertyListenerProc proc) noexcept
    {
        auto& ls = fromSelf(self)->state->listeners;
        for (size_t i = ls.size(); i > 0; --i)
            if (ls[i - 1].id == id && ls[i - 1].proc == proc)
                ls.erase(ls.begin() + static_cast<ptrdiff_t>(i - 1));
        return noErr;
    }

    static OSStatus sRemovePropertyListenerWithUserData(void* self, AudioUnitPropertyID id,
                                                        AudioUnitPropertyListenerProc proc,
                                                        void* user) noexcept
    {
        auto& ls = fromSelf(self)->state->listeners;
        for (size_t i = ls.size(); i > 0; --i)
            if (ls[i - 1].id == id && ls[i - 1].proc == proc && ls[i - 1].user == user)
                ls.erase(ls.begin() + static_cast<ptrdiff_t>(i - 1));
        return noErr;
    }

    static OSStatus sAddRenderNotify(void*, AURenderCallback, void*) noexcept
    { return noErr; }

    static OSStatus sRemoveRenderNotify(void*, AURenderCallback, void*) noexcept
    { return noErr; }

    static OSStatus sScheduleParameters(void* self,
                                        const AudioUnitParameterEvent* events,
                                        UInt32 numEvents) noexcept
    {
        auto* c = fromSelf(self);
        for (UInt32 i = 0; events != nullptr && i < numEvents; ++i)
            if (events[i].eventType == kParameterEvent_Immediate)
                c->state->setParameter(events[i].parameter, events[i].scope,
                                       events[i].eventValues.immediate.value);
        return noErr;
    }

    static AudioComponentMethod sLookup(SInt16 selector) noexcept
    {
        switch (selector)
        {
        case kAudioUnitInitializeSelect:
            return reinterpret_cast<AudioComponentMethod>(&sInitialize);
        case kAudioUnitUninitializeSelect:
            return reinterpret_cast<AudioComponentMethod>(&sUninitialize);
        case kAudioUnitGetPropertyInfoSelect:
            return reinterpret_cast<AudioComponentMethod>(&sGetPropertyInfo);
        case kAudioUnitGetPropertySelect:
            return reinterpret_cast<AudioComponentMethod>(&sGetProperty);
        case kAudioUnitSetPropertySelect:
            return reinterpret_cast<AudioComponentMethod>(&sSetProperty);
        case kAudioUnitGetParameterSelect:
            return reinterpret_cast<AudioComponentMethod>(&sGetParameter);
        case kAudioUnitSetParameterSelect:
            return reinterpret_cast<AudioComponentMethod>(&sSetParameter);
        case kAudioUnitResetSelect:
            return reinterpret_cast<AudioComponentMethod>(&sReset);
        case kAudioUnitRenderSelect:
            return reinterpret_cast<AudioComponentMethod>(&sRender);
        case kAudioUnitAddPropertyListenerSelect:
            return reinterpret_cast<AudioComponentMethod>(&sAddPropertyListener);
        case kAudioUnitRemovePropertyListenerSelect:
            return reinterpret_cast<AudioComponentMethod>(&sRemovePropertyListener);
        case kAudioUnitRemovePropertyListenerWithUserDataSelect:
            return reinterpret_cast<AudioComponentMethod>(&sRemovePropertyListenerWithUserData);
        case kAudioUnitAddRenderNotifySelect:
            return reinterpret_cast<AudioComponentMethod>(&sAddRenderNotify);
        case kAudioUnitRemoveRenderNotifySelect:
            return reinterpret_cast<AudioComponentMethod>(&sRemoveRenderNotify);
        case kAudioUnitScheduleParametersSelect:
            return reinterpret_cast<AudioComponentMethod>(&sScheduleParameters);
        default:
            return nullptr;
        }
    }

    inline static OSType gSubtype = 0;
    inline static OSType gManufacturer = 0;

    static void* factory(const AudioComponentDescription* desc) noexcept
    {
        if (desc != nullptr && desc->componentType != fourCC("aufx"))
            return nullptr;
        auto* c = new (std::nothrow) Component();
        if (c == nullptr) return nullptr;
        c->iface.Open = &sOpen;
        c->iface.Close = &sClose;
        c->iface.Lookup = &sLookup;
        c->iface.reserved = nullptr;
        return c;
    }
};

} // namespace dspark::plugin::au

/**
 * @brief Declares the AUv2 factory for one plugin class. `subtype4` and
 * `manufacturer4` are 4-character codes (unique per plugin / per vendor) that
 * must match the bundle's Info.plist AudioComponents entry — and the
 * `auval -v aufx <subtype4> <manufacturer4>` invocation.
 */
#define DSPARK_AU_PLUGIN(PluginClass, subtype4, manufacturer4)                        \
    extern "C" __attribute__((visibility("default")))                                \
    void* DSParkAuFactory(const AudioComponentDescription* desc)                      \
    {                                                                                 \
        using Comp = dspark::plugin::au::Component<PluginClass>;                      \
        Comp::gSubtype = dspark::plugin::au::fourCC(subtype4);                        \
        Comp::gManufacturer = dspark::plugin::au::fourCC(manufacturer4);              \
        return Comp::factory(desc);                                                   \
    }

#else // !__APPLE__

// Self-disabling off macOS so one translation unit can target every format.
#define DSPARK_AU_PLUGIN(PluginClass, subtype4, manufacturer4)

#endif // __APPLE__
