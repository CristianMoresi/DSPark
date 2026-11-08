// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DSParkAu.h
 * @brief Native Audio Unit v2 backend: the same plugin class, for Logic Pro.
 *
 * Implements the AUv2 component ABI directly against Apple's AudioToolbox
 * (system headers - nothing to vendor or download). AUv2 remains the format
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
 *   Every parameter is flagged high-resolution, and a non-finite value
 *   from the host is refused (kAudioUnitErr_InvalidParameterValue). An
 *   immediate scheduled value reads back at once; the DSP hears it at its
 *   offset in the next slice.
 * - The component type follows the class: Category::Instrument is an
 *   `aumu` music device (no input elements; MIDI drives it), a HasMidi
 *   effect is an `aumf` music effect, anything else stays `aufx`. The
 *   bundle's Info.plist entry must declare the same type.
 * - MIDI arrives through the MusicDevice selectors as raw bytes, lands in a
 *   lock-free ring and is delivered as MidiEvents, time-ordered with the
 *   scheduled parameter events (sample-accurate sub-block processing, the
 *   same scheme as the VST3/CLAP backends).
 * - HasTransport plugins get the host-callback transport (BeatAndTempo /
 *   TransportState / MusicalTimeLocation) once per render.
 * - Mono and stereo follow the declared ChannelSupport through
 *   SupportedNumChannels and the stream format; all elements run the same
 *   width. kAudioUnitProperty_OfflineRender feeds setOfflineRendering.
 *   While initialized the stream format, sample rate and
 *   MaximumFramesPerSlice are fixed: a different value is refused
 *   (kAudioUnitErr_PropertyNotWritable / kAudioUnitErr_Initialized) so the
 *   host uninitializes, sets it and initializes again; re-applying the
 *   current value is accepted.
 * - State (kAudioUnitProperty_ClassInfo) wraps the shared format-agnostic
 *   container in the standard aupreset dictionary, so presets remain
 *   portable across all three backends; its "name" is the current preset's
 *   name, restored with the state (PresentPreset listeners hear of it),
 *   and a preset of another unit (subtype / manufacturer) is refused.
 *   factoryPresets publish through kAudioUnitProperty_FactoryPresets and
 *   PresentPreset. A latency or tail change after parameter motion - or at
 *   initialize - notifies the kAudioUnitProperty_Latency / TailTime
 *   listeners on the main thread.
 * - Render notifications (AudioUnitAddRenderNotify) run before and after
 *   every render, with the pre/post-render flags (and the error flag on a
 *   refused call), as the format's reference implementation runs them. A
 *   render of another output element, or into a host buffer shorter than
 *   the slice, is refused.
 * - Reset clears the plugin's history and drops the MIDI queued before it.
 * - Host bypass maps to kAudioUnitProperty_BypassEffect with the same
 *   crossfade as the other backends. The bypass is also a parameter: a
 *   change through the parameter (automation, a state load) notifies the
 *   property's listeners, and one through the property notifies the
 *   parameter's, so the host's button and its automation lane agree.
 * - Threading: while initialized, the render thread is the only caller of
 *   the user's setParameter; AudioUnitSetParameter from another thread, the
 *   editor, ClassInfo and preset loads update the shadow and are applied at
 *   the next render (DSPark setters such as Equalizer::setBand allow one
 *   writer). State and preset loads notify AU parameter listeners.
 * - Input arrives by pulling the host's render callback (or connection),
 *   the universal aufx pattern auval exercises. render() runs under a
 *   DenormalGuard (FTZ/DAZ).
 * - **Editor**: when plugin/webview/DSParkWebViewEditor.h is included before
 *   this header and the class declares `hasEditor = true`, the AU publishes
 *   kAudioUnitProperty_CocoaUI - hosts load a runtime-registered view
 *   factory from this bundle and get the same WKWebView editor as the
 *   VST3/CLAP backends; otherwise they show their generic parameter UI. A
 *   resizable editor reopens at the size the host left it (the container
 *   view records every setFrameSize:), also in the next session.
 */

#define DSPARK_PLUGIN_AU_INCLUDED 1

#if defined(__APPLE__)

#include "../DSParkPlugin.h"

#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>

// AUParameterListenerNotify (state / preset loads), AUParameterSet and
// AUEventListenerNotify (editor gestures): all in AudioToolbox.framework.
#include <AudioToolbox/AudioUnitUtilities.h>

#if defined(DSPARK_PLUGIN_WEBVIEW)
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>
#endif

#include <dispatch/dispatch.h>
#include <pthread.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

// Survive a <windows.h>-style min/max macro (restored at the end).
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max

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

#if defined(DSPARK_PLUGIN_WEBVIEW)

// -- WebView editor glue (Cocoa view factory) -----------------------------------
//
// AUv2 has no IPlugView: the host reads kAudioUnitProperty_CocoaUI, loads the
// announced bundle, instantiates the named factory class and asks it for an
// NSView. Both classes here (factory + container view) are registered through
// the Objective-C runtime on first use - no Objective-C sources, no AppKit
// link dependency in the plugin. The factory is stateless: it recovers the
// C++ plugin instance through a PRIVATE property on the AudioUnit handle
// (IDs >= 64000 are reserved for third parties), so one process-wide class
// serves every DSPark plugin loaded in the host, whichever binary won the
// registration race (same policy as the WebView message relay).

/** @brief Private property handing the view factory a way back into C++. */
inline constexpr AudioUnitPropertyID kDsparkEditorHookProperty = 64537;

/** @brief Payload of kDsparkEditorHookProperty: instance + view constructor.
 *  The function pointer lives in the SAME binary as the AudioUnit instance,
 *  so a factory class registered by another DSPark plugin still lands here. */
struct EditorHook
{
    void* context = nullptr;
    void* (*createView)(void* context, double width, double height) = nullptr;
};

/** @brief NSSize-compatible POD for the factory selector signature. */
struct ViewSize
{
    double width = 0, height = 0;
};

/** @brief Back-reference stored in the container view's ivar; the container
 *  notifies the owning wrapper when the host deallocates it. */
struct EditorViewSink
{
    void (*onDealloc)(void* context) = nullptr;
    void* context = nullptr;
    /// The host resized the container (logical points).
    void (*onResize)(void* context, double width, double height) = nullptr;
};

inline Class editorContainerClass() noexcept;

inline void editorContainerDealloc(void* self, SEL) noexcept
{
    // Instances exist only after the class registered, so the lookup cannot
    // miss; the superclass is resolved from the registered class (NOT from
    // object_getClass) so a future subclass would not recurse.
    Class cls = editorContainerClass();
    if (cls == nullptr) return;
    if (Ivar ivar = class_getInstanceVariable(cls, "dsparkSink"))
    {
        auto* sink = reinterpret_cast<EditorViewSink*>(
            object_getIvar(static_cast<id>(self), ivar));
        if (sink != nullptr && sink->onDealloc != nullptr)
            sink->onDealloc(sink->context);
    }
    objc_super super { static_cast<id>(self), class_getSuperclass(cls) };
    reinterpret_cast<void (*)(objc_super*, SEL)>(&objc_msgSendSuper)(
        &super, sel_registerName("dealloc"));
}

/** @brief setFrameSize: of the container: NSView's own, then the owning
 *  wrapper hears the new size (every host-driven resize passes here, whether
 *  the host sets the frame or the window autoresizes the view). */
inline void editorContainerSetFrameSize(void* self, SEL selector, ViewSize size) noexcept
{
    Class cls = editorContainerClass();
    if (cls == nullptr) return;
    objc_super super { static_cast<id>(self), class_getSuperclass(cls) };
    reinterpret_cast<void (*)(objc_super*, SEL, ViewSize)>(&objc_msgSendSuper)(
        &super, selector, size);
    if (Ivar ivar = class_getInstanceVariable(cls, "dsparkSink"))
    {
        auto* sink = reinterpret_cast<EditorViewSink*>(
            object_getIvar(static_cast<id>(self), ivar));
        if (sink != nullptr && sink->onResize != nullptr)
            sink->onResize(sink->context, size.width, size.height);
    }
}

/** @brief The NSView subclass hosting the editor. Returns null when AppKit is
 *  not loaded (a faceless host like auval - which never asks for views). */
inline Class editorContainerClass() noexcept
{
    // The class name carries this binary's own suffix: a process-wide name
    // let the first DSPark plugin loaded register it, and every later plugin
    // binary then ran THAT binary's dealloc code (another DSPark version, or
    // an image the host may since have unloaded).
    static Class registered = []() -> Class {
        Class viewCls = objc_getClass("NSView");
        if (viewCls == nullptr) return nullptr;
        static char name[96];
        webview_ui::objc_glue::imageUniqueClassName("DSParkAUEditorContainer",
                                                    name, sizeof(name));
        Class c = objc_allocateClassPair(viewCls, name, 0);
        if (c == nullptr)   // this image registered it already
            return objc_getClass(name);
        class_addIvar(c, "dsparkSink", sizeof(void*), alignof(void*) == 8 ? 3 : 2, "^v");
        class_addMethod(c, sel_registerName("dealloc"),
                        reinterpret_cast<IMP>(&editorContainerDealloc), "v@:");
        class_addMethod(c, sel_registerName("setFrameSize:"),
                        reinterpret_cast<IMP>(&editorContainerSetFrameSize),
                        "v@:{CGSize=dd}");
        objc_registerClassPair(c);
        return c;
    }();
    return registered;
}

inline unsigned int editorFactoryInterfaceVersion(void*, SEL) noexcept
{
    return 0;   // AUCocoaUIBase protocol version
}

inline void* editorFactoryUiView(void*, SEL, void* audioUnit, ViewSize size) noexcept
{
    EditorHook hook {};
    UInt32 ioSize = sizeof(hook);
    if (audioUnit == nullptr
        || AudioUnitGetProperty(static_cast<AudioUnit>(audioUnit),
                                kDsparkEditorHookProperty, kAudioUnitScope_Global,
                                0, &hook, &ioSize) != noErr
        || hook.createView == nullptr)
        return nullptr;
    return hook.createView(hook.context, size.width, size.height);
}

/** @brief The AUCocoaUIBase factory class announced through CocoaUI. */
inline Class editorFactoryClass() noexcept
{
    static Class registered = []() -> Class {
        static char name[96];   // per-binary name, see editorContainerClass()
        webview_ui::objc_glue::imageUniqueClassName("DSParkAUCocoaViewFactory",
                                                    name, sizeof(name));
        Class c = objc_allocateClassPair(objc_getClass("NSObject"), name, 0);
        if (c == nullptr)
            return objc_getClass(name);
        class_addMethod(c, sel_registerName("interfaceVersion"),
                        reinterpret_cast<IMP>(&editorFactoryInterfaceVersion), "I@:");
        class_addMethod(c, sel_registerName("uiViewForAudioUnit:withSize:"),
                        reinterpret_cast<IMP>(&editorFactoryUiView),
                        "@@:^{ComponentInstanceRecord=}{CGSize=dd}");
        // Hosts overwhelmingly probe by respondsToSelector; declare the formal
        // protocol too when some loaded image has registered it.
        if (Protocol* proto = objc_getProtocol("AUCocoaUIBase"))
            class_addProtocol(c, proto);
        objc_registerClassPair(c);
        return c;
    }();
    return registered;
}

/** @brief CFURL of the bundle this code lives in (caller releases), derived
 *  from the image path: .../Foo.component/Contents/MacOS/Foo. A bare dylib
 *  (tests) falls back to its directory. */
inline CFURLRef copyOwningBundleUrl() noexcept
{
    static const int anchor = 0;   // any address inside THIS image
    Dl_info info {};
    if (dladdr(&anchor, &info) == 0 || info.dli_fname == nullptr) return nullptr;
    const char* path = info.dli_fname;
    const char* marker = nullptr;
    for (const char* p = std::strstr(path, "/Contents/MacOS/"); p != nullptr;
         p = std::strstr(p + 1, "/Contents/MacOS/"))
        marker = p;   // last occurrence wins (nested bundle layouts)
    size_t length = marker != nullptr ? static_cast<size_t>(marker - path) : 0;
    if (length == 0)
        if (const char* slash = std::strrchr(path, '/'))
            length = static_cast<size_t>(slash - path);
    if (length == 0) return nullptr;
    return CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault, reinterpret_cast<const UInt8*>(path),
        static_cast<CFIndex>(length), true);
}

#endif // DSPARK_PLUGIN_WEBVIEW

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
    static_assert(paramIdsUnique<P>(),
                  "two parameter ids share a hash32 (or collide with the "
                  "reserved PRGM/BYPS/EDSW/EDSH state ids): automation and state "
                  "would cross-wire. Rename one id.");

    /** Component type implied by the class: aumu / aumf / aufx. */
    static OSType expectedType() noexcept
    {
        if (kIsInstrument) return fourCC("aumu");
        if (HasMidi<P>) return fourCC("aumf");
        return fourCC("aufx");
    }

    AudioComponentInstance instance = nullptr;
    OSType subtype = 0, manufacturer = 0;

    double sampleRate = 44100.0;
    UInt32 maxFrames = 1156;            // CoreAudio's historical default
    /// Written on the host's main thread, read on the render thread and by
    /// parameter calls from any thread.
    std::atomic<bool> initialized { false };
    int currentChannels = defaultChannelCount<P>();   // negotiated width
    UInt32 offlineRender = 0;

    P user {};
    std::atomic<double> shadow[kNumParams == 0 ? 1 : kNumParams] {};
    std::atomic<bool>   bypass { false };
    float bypassMix = 0.0f;
    /// Latency in samples, as last reported to the host. The processing call
    /// re-reads it after parameter motion while the host fetches it from its
    /// own thread, so it is a published word rather than a plain int. It
    /// stands alone -- nothing else is ordered against it, and the host is
    /// told to re-fetch through the notification path below -- so the plain
    /// store/load idiom for a single independent value is the whole contract.
    std::atomic<int> cachedLatency { 0 };

    // Input elements: 0 = main, 1 = sidechain. An instrument has none.
    static constexpr UInt32 kNumInputElements =
        kIsInstrument ? 0 : (HasSidechain<P> ? 2 : 1);
    AURenderCallbackStruct inputCallback[2] {};
    AudioUnit inputConnection[2] = { nullptr, nullptr };
    UInt32 inputConnectionBus[2] = { 0, 0 };

    std::vector<float> pullL, pullR, dryL, dryR, scL, scR;
    DryDelay dryDelay;   // latency-aligned dry path of the bypass

    // Host transport callbacks (kAudioUnitProperty_HostCallbacks).
    HostCallbackInfo hostCallbacks {};

    // Incoming MIDI: the MusicDevice selectors push, render() drains.
    // Single producer (host MIDI thread), single consumer (render thread).
    static constexpr uint32_t kMidiRingSize = 256;   // power of two
    MidiEvent midiRing[HasMidi<P> ? kMidiRingSize : 1] {};
    std::atomic<uint32_t> midiHead { 0 }, midiTail { 0 };
    // Reset discards the MIDI queued before it (a transport stop must not
    // replay notes into the next run): it records the ring position, the
    // render thread skips up to it.
    std::atomic<uint32_t> midiDiscardTo { 0 };
    std::atomic<bool>     midiDiscard { false };

    // Render notifications (AudioUnitAddRenderNotify): host callbacks run
    // before and after every render - hosts schedule parameters from the
    // pre-render one and meter from the post-render one. Add/remove arrive
    // on a control thread and reach the render thread through a
    // single-producer queue; the render thread owns the list it calls.
    struct RenderNotify
    {
        AURenderCallback proc = nullptr;
        void* refCon = nullptr;
    };
    static constexpr int kMaxRenderNotifies = 16;
    RenderNotify renderNotifies[kMaxRenderNotifies] {};
    int renderNotifyCount = 0;
    struct NotifyCommand
    {
        RenderNotify target {};
        bool add = false;
    };
    static constexpr uint32_t kNotifyQueueSize = 64;   // power of two
    NotifyCommand notifyQueue[kNotifyQueueSize] {};
    std::atomic<uint32_t> notifyHead { 0 }, notifyTail { 0 };

    // Sample-accurate parameter events (kAudioUnitScheduleParametersSelect
    // arrives on the render thread, right before the matching render call).
    // Immediate events and bypass points go straight into the block list
    // (merging past its capacity instead of dropping the newest); ramps are
    // kept whole and evaluated against the slice length inside render().
    BlockEventList blockEvents;
    struct Ramp
    {
        uint32_t paramId;
        SInt32 startOffset;          // may be negative: began in an earlier slice
        UInt32 duration;
        float startValue, endValue;  // plain
    };
    static constexpr int kMaxRamps = 64;
    Ramp ramps[kMaxRamps] {};
    int rampCount = 0;

    // One writer for the user's setParameter: the render thread while the
    // unit is initialized; edits from any other thread are handed over.
    PendingParams<kNumParams> pending;
    std::atomic<std::thread::id> renderThread {};
    UInt32 sidechainChannels = 0;    // input element 1 width (0: follow main)

    // Factory presets: AUPreset records plus the lazily built CFArray the
    // property hands out (retained per get; the strings live here).
    std::vector<AUPreset> presetStorage;
    CFArrayRef presetArray = nullptr;
    SInt32 currentPresetNumber = -1;

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

    /// Tail in seconds, as last reported to the host (see refreshReported).
    std::atomic<double> cachedTail { 0.0 };
    /// Latency in seconds the host learned at the last initialize().
    double latencySecondsAtInit = 0.0;
    /// The editor size the user left (kept without the editor layer too, so
    /// a session saved where the editor ran keeps its size here).
    EditorSizeMemory<P> editorMemory;

#if defined(DSPARK_PLUGIN_WEBVIEW)
    webview_ui::Editor<P> editor;
    void* editorContainer = nullptr;   // NSView* the HOST owns; weak here
    EditorViewSink editorSink {};
    bool editActive[kNumParams == 0 ? 1 : kNumParams] {};
#endif

    ~Plugin() noexcept
    {
        stopTick(false);   // disposing: no listener call from here
#if defined(DSPARK_PLUGIN_WEBVIEW)
        teardownEditor();
#endif
        if (presetName != nullptr) CFRelease(presetName);
        for (auto& preset : presetStorage)
            if (preset.presetName != nullptr) CFRelease(preset.presetName);
        if (presetArray != nullptr) CFRelease(presetArray);
    }

    /** Applies one factory preset: every parameter, by normalized value. */
    void applyFactoryPresetIdx(int idx) noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            if (idx < 0 || idx >= kNumPresets) return;
            for (size_t i = 0; i < kNumParams; ++i)
                applyFromAnyThread(static_cast<int>(i), presetNormalized<P>(idx, i));
            currentPresetNumber = idx;
        }
        else
            (void) idx;
    }

    /** Tail in seconds as the TailTime property reports it (a negative or
     *  NaN answer from the plugin is no tail). */
    double tailSecondsNow() const noexcept
    {
        if constexpr (HasTail<P>)
        {
            const double tail = user.getTailSeconds();
            return tail > 0.0 ? tail : 0.0;
        }
        else
            return 0.0;
    }

    /** Re-reads the plugin latency and tail after parameter motion; on a
     *  change, updates the cache and tells the kAudioUnitProperty_Latency /
     *  TailTime listeners - on the main thread. A change detected in
     *  render() (automation moved a lookahead or a decay) only raises a bit;
     *  the main-queue tick below calls the host's listeners, which must
     *  never run on the render thread. The old code never told the host
     *  about a tail change. */
    void refreshReported() noexcept
    {
        uint32_t bits = 0;
        if constexpr (HasLatency<P>)
        {
            const int now = user.getLatency();
            if (initialized && now != cachedLatency.load(std::memory_order_relaxed))
            {
                cachedLatency.store(now, std::memory_order_relaxed);
                bits |= kNotifyLatency;
            }
        }
        if constexpr (HasTail<P>)
        {
            const double now = tailSecondsNow();
            if (initialized && now != cachedTail.load(std::memory_order_relaxed))
            {
                cachedTail.store(now, std::memory_order_relaxed);
                bits |= kNotifyTail;
            }
        }
        if (bits != 0) announce(bits);
    }

    // --- property notifications (main thread; ~30 Hz main-queue tick) ------------------

    enum : uint32_t { kNotifyLatency = 1u, kNotifyTail = 2u, kNotifyBypass = 4u };
    std::atomic<uint32_t> notifyPending { 0u };
    std::atomic<bool> tickRunning { false };
    dispatch_source_t tick = nullptr;

    /** Tells the host that the latency, the tail or the bypass changed: at
     *  once on the main thread (or when no tick runs), else at the next tick. */
    void announce(uint32_t bits) noexcept
    {
        if (pthread_main_np() != 0 || !tickRunning.load(std::memory_order_acquire))
            deliverNotifications(bits);
        else
            notifyPending.fetch_or(bits, std::memory_order_release);
    }

    void deliverNotifications(uint32_t bits) noexcept
    {
        if ((bits & kNotifyLatency) != 0u)
            notifyProperty(kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0);
        if ((bits & kNotifyTail) != 0u)
            notifyProperty(kAudioUnitProperty_TailTime, kAudioUnitScope_Global, 0);
        // The host's bypass button follows the property, not the bypass
        // parameter the host itself moved (automation, a state load).
        if ((bits & kNotifyBypass) != 0u)
            notifyProperty(kAudioUnitProperty_BypassEffect, kAudioUnitScope_Global, 0);
    }

    static void onTick(void* self)
    {
        auto* plugin = static_cast<Plugin*>(self);
        const uint32_t bits = plugin->notifyPending.exchange(0u, std::memory_order_acq_rel);
        if (bits != 0u) plugin->deliverNotifications(bits);
    }

    void startTick() noexcept
    {
        if (tick != nullptr) return;
        tick = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
                                      dispatch_get_main_queue());
        if (tick == nullptr) return;
        dispatch_set_context(tick, this);
        dispatch_source_set_event_handler_f(tick, &onTick);
        dispatch_source_set_timer(tick, dispatch_time(DISPATCH_TIME_NOW, 33 * 1000000),
                                  33u * 1000000u, 5u * 1000000u);
        dispatch_resume(tick);
        tickRunning.store(true, std::memory_order_release);
    }

    /** Main thread (uninitialize/close): no tick runs after this returns;
     *  still-pending notifications are delivered first when @p flush is set. */
    void stopTick(bool flush = true) noexcept
    {
        if (tick == nullptr) return;
        tickRunning.store(false, std::memory_order_release);
        dispatch_source_cancel(tick);
        dispatch_release(tick);
        tick = nullptr;
        const uint32_t bits = notifyPending.exchange(0u, std::memory_order_acq_rel);
        if (bits != 0u && flush) deliverNotifications(bits);
    }

    /** Tells AU parameter listeners that one parameter moved. */
    void notifyParameterChanged(AudioUnitParameterID id) noexcept
    {
        AudioUnitParameter parameter {};
        parameter.mAudioUnit = instance;
        parameter.mParameterID = id;
        parameter.mScope = kAudioUnitScope_Global;
        parameter.mElement = 0;
        AUParameterListenerNotify(nullptr, nullptr, &parameter);
    }

    /** Sets the host bypass; true when it changed. */
    bool setBypass(bool on) noexcept
    {
        return bypass.exchange(on, std::memory_order_relaxed) != on;
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
        return paramIndexOf<P>(id);
    }

    void applyNormalized(int index, double normalized) noexcept
    {
        // Host values are untrusted doubles: a NaN would pass both range
        // clamps, poison the shadow and reach the user's setter. Ignore it.
        if (normalized != normalized) return;
        const auto& spec = P::parameters[static_cast<size_t>(index)];
        shadow[static_cast<size_t>(index)].store(normalized, std::memory_order_relaxed);
        user.setParameter(index, static_cast<float>(toPlain(spec, normalized)));
    }

    /**
     * A change arriving on any thread. While initialized, a thread other
     * than the render thread only writes the shadow and marks the parameter;
     * render() calls the user's setParameter at its next slice. Calling it
     * directly (the old behaviour) made the host's UI thread a second
     * concurrent writer of DSPark setters that allow one. Returns true when
     * the change reached the DSP immediately.
     */
    bool applyFromAnyThread(int index, double normalized) noexcept
    {
        if (normalized != normalized) return false;
        if (initialized
            && std::this_thread::get_id() != renderThread.load(std::memory_order_relaxed))
        {
            shadow[static_cast<size_t>(index)].store(normalized, std::memory_order_relaxed);
            pending.mark(static_cast<size_t>(index));
            return false;
        }
        applyNormalized(index, normalized);
        return true;
    }

    bool drainPending() noexcept
    {
        return pending.drain([this](int i) {
            user.setParameter(i, static_cast<float>(toPlain(
                P::parameters[static_cast<size_t>(i)],
                shadow[static_cast<size_t>(i)].load(std::memory_order_relaxed))));
        });
    }

    /** The current preset's name (caller releases): the one set through
     *  PresentPreset or restored with the state, else the active factory
     *  preset's, else "Untitled" (the format's default name). */
    CFStringRef copyPresetName() const noexcept
    {
        if (presetName != nullptr)
            return static_cast<CFStringRef>(CFRetain(presetName));
        const char* name = "Untitled";
        if constexpr (kNumPresets > 0)
            if (currentPresetNumber >= 0 && currentPresetNumber < kNumPresets)
                name = P::factoryPresets[static_cast<size_t>(currentPresetNumber)].name;
        return CFStringCreateWithCString(nullptr, name, kCFStringEncodingUTF8);
    }

    /** Tells AU parameter listeners (host automation lanes, generic views)
     *  that every parameter may have moved - the AUBase convention after a
     *  state or preset load, which the old code never followed. */
    void notifyAllParameters() noexcept
    {
        AudioUnitParameter any {};
        any.mAudioUnit = instance;
        any.mParameterID = kAUParameterListener_AnyParameter;
        any.mScope = kAudioUnitScope_Global;
        any.mElement = 0;
        AUParameterListenerNotify(nullptr, nullptr, &any);
    }

    void applyAllShadows() noexcept
    {
        pending.drain([](int) {});   // every value is re-applied right below
        for (size_t i = 0; i < kNumParams; ++i)
            user.setParameter(static_cast<int>(i),
                static_cast<float>(toPlain(P::parameters[i],
                                           shadow[i].load(std::memory_order_relaxed))));
    }

#if defined(DSPARK_PLUGIN_WEBVIEW)

    // --- WebView editor (all on the host's main thread) -----------------------------

    static constexpr bool kHasWebEditor =
        HasEditor<P> && webview_ui::Editor<P>::kAvailable;

    void sendGesture(int index, AudioUnitEventType type) noexcept
    {
        AudioUnitEvent event {};
        event.mEventType = type;
        event.mArgument.mParameter = AudioUnitParameter {
            instance, hash32(P::parameters[static_cast<size_t>(index)].id),
            kAudioUnitScope_Global, 0 };
        AUEventListenerNotify(nullptr, nullptr, &event);
    }

    // Editor -> host/DSP bridge. AUParameterSet routes through our own
    // SetParameter (shadow + user setter) AND notifies host listeners, which
    // is how AU hosts observe UI edits for automation recording.
    static void cbSetParam(void* context, int index, double plain) noexcept
    {
        auto* self = static_cast<Plugin*>(context);
        const AudioUnitParameter parameter {
            self->instance, hash32(P::parameters[static_cast<size_t>(index)].id),
            kAudioUnitScope_Global, 0 };
        const bool inGesture = self->editActive[static_cast<size_t>(index)];
        if (!inGesture)   // edits outside a drag still need a gesture for undo
            self->sendGesture(index, kAudioUnitEvent_BeginParameterChangeGesture);
        AUParameterSet(nullptr, nullptr, &parameter,
                       static_cast<AudioUnitParameterValue>(plain), 0);
        if (!inGesture)
            self->sendGesture(index, kAudioUnitEvent_EndParameterChangeGesture);
    }

    static void cbBeginEdit(void* context, int index) noexcept
    {
        auto* self = static_cast<Plugin*>(context);
        self->editActive[static_cast<size_t>(index)] = true;
        self->sendGesture(index, kAudioUnitEvent_BeginParameterChangeGesture);
    }

    static void cbEndEdit(void* context, int index) noexcept
    {
        auto* self = static_cast<Plugin*>(context);
        self->editActive[static_cast<size_t>(index)] = false;
        self->sendGesture(index, kAudioUnitEvent_EndParameterChangeGesture);
    }

    static void sOnContainerDealloc(void* context) noexcept
    {
        auto* self = static_cast<Plugin*>(context);
        self->editorContainer = nullptr;   // the host is destroying the view
        self->editor.destroy();
    }

    /** The host resized the container: keep the size (logical points) so the
     *  next view, and the next session, open at it. */
    static void sOnContainerResize(void* context, double width, double height) noexcept
    {
        static_cast<Plugin*>(context)->editorMemory.remember(width, height);
    }

    /** Plugin dies first (or the host asks for a fresh view): disconnect the
     *  container so its later dealloc no longer calls back, then tear down. */
    void teardownEditor() noexcept
    {
        if (editorContainer != nullptr)
        {
            Class cls = editorContainerClass();
            if (Ivar ivar = cls != nullptr
                    ? class_getInstanceVariable(cls, "dsparkSink") : nullptr)
                object_setIvar(static_cast<id>(editorContainer), ivar, nullptr);
            editorContainer = nullptr;
        }
        editor.destroy();
    }

    static void* sCreateEditorView(void* context, double width, double height) noexcept
    {
        return static_cast<Plugin*>(context)->createEditorView(width, height);
    }

    /** Builds the container NSView + embedded web engine. Returns the view
     *  AUTORELEASED (Cocoa factory convention: the host retains it), or null
     *  so the host falls back to its generic UI. The host's preferred size is
     *  advisory (often 0x0); the size the user last left the editor at (this
     *  session or a saved one) wins, else the declared editorSize, like the
     *  other backends. */
    void* createEditorView(double, double) noexcept
    {
        namespace og = webview_ui::objc_glue;
        teardownEditor();   // hosts may ask again without releasing the first view
        Class cls = editorContainerClass();
        if (cls == nullptr) return nullptr;   // AppKit absent: faceless host
        const EditorSize logical = editorMemory.current();
        const og::Rect frame { 0.0, 0.0, static_cast<double>(logical.width),
                               static_cast<double>(logical.height) };
        og::ObjId view = og::call(og::call(reinterpret_cast<og::ObjId>(cls), "alloc"),
                                  "initWithFrame:", frame);
        if (view == nullptr) return nullptr;
        const webview_ui::HostCallbacks callbacks {
            this, &cbSetParam, &cbBeginEdit, &cbEndEdit
        };
        if (!editor.create(view, shadow, callbacks))
        {
            og::call<void>(view, "release");
            return nullptr;
        }
        editorSink = { &sOnContainerDealloc, this, &sOnContainerResize };
        if (Ivar ivar = class_getInstanceVariable(cls, "dsparkSink"))
            object_setIvar(static_cast<id>(view),
                           ivar, reinterpret_cast<id>(&editorSink));
        editorContainer = view;
        editor.setBounds(logical.width, logical.height);
        editor.setVisible(true);
        webview_ui::debugLog("au createEditorView %dx%d container=%p",
                             logical.width, logical.height, view);
        return og::call(view, "autorelease");
    }

#endif // DSPARK_PLUGIN_WEBVIEW

    // --- lifecycle ---------------------------------------------------------------

    OSStatus initialize() noexcept
    {
        dspark::AudioSpec spec { sampleRate, static_cast<int>(maxFrames),
                                 currentChannels };
        const double tailBefore = cachedTail.load(std::memory_order_relaxed);
        user.prepare(spec);
        if constexpr (HasOfflineMode<P>)
            user.setOfflineRendering(offlineRender != 0);
        applyAllShadows();
        pullL.assign(maxFrames, 0.0f);
        pullR.assign(maxFrames, 0.0f);
        dryL.assign(maxFrames, 0.0f);
        dryR.assign(maxFrames, 0.0f);
        if constexpr (HasSidechain<P>)
        {
            scL.assign(maxFrames, 0.0f);
            scR.assign(maxFrames, 0.0f);
        }
        bypassMix = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        uint32_t changed = 0;
        if constexpr (HasLatency<P>)
        {
            const int latency = user.getLatency();
            cachedLatency.store(latency, std::memory_order_relaxed);
            dryDelay.prepare(std::max(2 * latency, kDryDelayMinCapacity),
                             static_cast<int>(maxFrames));
            // The property is in seconds: a new latency or a new rate moves
            // it. Listeners hear of it here; the old code only updated the
            // cache, so a host that read it before initializing kept 0.
            const double seconds = sampleRate > 0.0 ? latency / sampleRate : 0.0;
            if (seconds != latencySecondsAtInit) changed |= kNotifyLatency;
            latencySecondsAtInit = seconds;
        }
        if constexpr (HasTail<P>)
        {
            const double tail = tailSecondsNow();
            cachedTail.store(tail, std::memory_order_relaxed);
            if (tail != tailBefore) changed |= kNotifyTail;
        }
        else
            (void) tailBefore;
        blockEvents.clear();
        rampCount = 0;
        if constexpr (HasMidi<P>)   // MIDI queued while uninitialized is stale
        {
            midiHead.store(midiTail.load(std::memory_order_acquire),
                           std::memory_order_release);
            midiDiscard.store(false, std::memory_order_relaxed);
        }
        applyNotifyCommands();   // no render runs before `initialized` below
        renderThread.store(std::thread::id {}, std::memory_order_relaxed);
        startTick();
        initialized = true;
        if (changed != 0u) deliverNotifications(changed);   // main thread
        return noErr;
    }

    OSStatus uninitialize() noexcept
    {
        initialized = false;
        drainPending();   // no render can run now: hand edits over directly
        stopTick();
        return noErr;
    }

    /** Clears the processing history. MIDI queued before the call is
     *  dropped (the render thread skips it), as a stop or locate requires:
     *  the old code replayed those notes into the next run. */
    OSStatus reset() noexcept
    {
        if constexpr (HasReset<P>)
            user.reset();
        dryDelay.reset();
        bypassMix = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if constexpr (HasMidi<P>)
        {
            midiDiscardTo.store(midiTail.load(std::memory_order_acquire),
                                std::memory_order_relaxed);
            midiDiscard.store(true, std::memory_order_release);
        }
        return noErr;
    }

    // --- render notifications -------------------------------------------------------

    /** Adds or removes a render notification (any control thread). While
     *  uninitialized no render runs, so the list is updated at once;
     *  otherwise the render thread applies the change at its next call. */
    OSStatus queueRenderNotify(AURenderCallback proc, void* refCon, bool add) noexcept
    {
        if (proc == nullptr) return kAudio_ParamError;
        if (!initialized)
        {
            applyNotifyCommands();
            applyNotifyCommand(RenderNotify { proc, refCon }, add);
            return noErr;
        }
        const uint32_t tail = notifyTail.load(std::memory_order_relaxed);
        const uint32_t next = (tail + 1) & (kNotifyQueueSize - 1);
        if (next == notifyHead.load(std::memory_order_acquire))
            return kAudio_MemFullError;   // 63 changes with no render in between
        notifyQueue[tail] = NotifyCommand { RenderNotify { proc, refCon }, add };
        notifyTail.store(next, std::memory_order_release);
        return noErr;
    }

    void applyNotifyCommand(RenderNotify target, bool add) noexcept
    {
        int found = -1;
        for (int i = 0; i < renderNotifyCount; ++i)
            if (renderNotifies[i].proc == target.proc
                && renderNotifies[i].refCon == target.refCon)
                found = i;
        if (add)
        {
            // A second add of the same pair is a no-op, as the format defines.
            if (found < 0 && renderNotifyCount < kMaxRenderNotifies)
                renderNotifies[renderNotifyCount++] = target;
        }
        else if (found >= 0)
        {
            for (int i = found; i + 1 < renderNotifyCount; ++i)
                renderNotifies[i] = renderNotifies[i + 1];
            --renderNotifyCount;
        }
    }

    /** The queued add/remove requests, applied by the list's owner. */
    void applyNotifyCommands() noexcept
    {
        uint32_t head = notifyHead.load(std::memory_order_relaxed);
        const uint32_t tail = notifyTail.load(std::memory_order_acquire);
        while (head != tail)
        {
            applyNotifyCommand(notifyQueue[head].target, notifyQueue[head].add);
            head = (head + 1) & (kNotifyQueueSize - 1);
        }
        notifyHead.store(head, std::memory_order_release);
    }

    void callRenderNotifies(AudioUnitRenderActionFlags flags,
                            const AudioTimeStamp* timeStamp, UInt32 busNumber,
                            UInt32 frames, AudioBufferList* ioData) noexcept
    {
        for (int i = 0; i < renderNotifyCount; ++i)
            renderNotifies[i].proc(renderNotifies[i].refCon, &flags, timeStamp,
                                   busNumber, frames, ioData);
    }

    /** A slice the unit does not render: its scheduled values still take
     *  effect (as scheduled values do in the format's reference implementation - they are
     *  already in the shadow and reach the DSP at the next render), its
     *  ramps are dropped. The old code left both in the list, and the next
     *  slice replayed them at stale offsets. */
    void discardSlice() noexcept
    {
        for (int i = 0; i < blockEvents.size(); ++i)
        {
            const BlockEvent& ev = blockEvents[i];
            if (ev.kind == BlockEvent::Kind::Bypass)
            {
                if (setBypass(ev.value >= 0.5)) announce(kNotifyBypass);
            }
            else if (ev.kind == BlockEvent::Kind::Param)
            {
                if (const int idx = indexOfParamId(ev.paramId); idx >= 0)
                {
                    shadow[static_cast<size_t>(idx)].store(ev.value,
                                                           std::memory_order_relaxed);
                    pending.mark(static_cast<size_t>(idx));
                }
            }
        }
        blockEvents.clear();
        rampCount = 0;
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
        f.mChannelsPerFrame = static_cast<UInt32>(currentChannels);
        f.mBitsPerChannel = 32;
    }

    bool acceptableFormat(const AudioStreamBasicDescription& f) const noexcept
    {
        return f.mFormatID == kAudioFormatLinearPCM
            && (f.mFormatFlags & kAudioFormatFlagIsFloat) != 0
            && (f.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
            && supportsChannelCount<P>(static_cast<int>(f.mChannelsPerFrame))
            && f.mBitsPerChannel == 32
            && f.mSampleRate > 0.0;
    }

    // --- properties -------------------------------------------------------------------

    static constexpr int numChannelConfigs() noexcept
    {
        return channelSupportOf<P>() == ChannelSupport::MonoAndStereo ? 2 : 1;
    }

    /** Lazily builds the FactoryPresets CFArray (records + names live in
     *  the instance; the array holds bare AUPreset pointers, callbacks-free). */
    void buildPresetArray() noexcept
    {
        if constexpr (kNumPresets > 0)
        {
            if (presetArray != nullptr) return;
            presetStorage.resize(static_cast<size_t>(kNumPresets));
            const void* pointers[kNumPresets == 0 ? 1 : kNumPresets];
            for (int i = 0; i < kNumPresets; ++i)
            {
                presetStorage[static_cast<size_t>(i)].presetNumber = i;
                presetStorage[static_cast<size_t>(i)].presetName =
                    CFStringCreateWithCString(
                        nullptr, P::factoryPresets[static_cast<size_t>(i)].name,
                        kCFStringEncodingUTF8);
                pointers[i] = &presetStorage[static_cast<size_t>(i)];
            }
            presetArray = CFArrayCreate(nullptr, pointers,
                                        static_cast<CFIndex>(kNumPresets), nullptr);
        }
    }

    OSStatus getPropertyInfo(AudioUnitPropertyID id, AudioUnitScope scope,
                             AudioUnitElement element, UInt32* outSize,
                             Boolean* outWritable) noexcept
    {
        UInt32 size = 0;
        Boolean writable = false;
        switch (id)
        {
        // Formats are fixed while initialized (see setProperty).
        case kAudioUnitProperty_StreamFormat:
            size = sizeof(AudioStreamBasicDescription); writable = !initialized; break;
        case kAudioUnitProperty_SampleRate:
            size = sizeof(Float64); writable = !initialized; break;
        case kAudioUnitProperty_MaximumFramesPerSlice:
            size = sizeof(UInt32); writable = true; break;
        case kAudioUnitProperty_ElementCount:
            size = sizeof(UInt32); break;
        case kAudioUnitProperty_SupportedNumChannels:
            size = static_cast<UInt32>(numChannelConfigs() * sizeof(AUChannelInfo));
            break;
        case kAudioUnitProperty_HostCallbacks:
            size = sizeof(HostCallbackInfo); writable = true; break;
        case kAudioUnitProperty_OfflineRender:
            size = sizeof(UInt32); writable = true; break;
        case kAudioUnitProperty_FactoryPresets:
            if (kNumPresets == 0) return kAudioUnitErr_InvalidProperty;
            size = sizeof(CFArrayRef);
            break;
        case kAudioUnitProperty_ParameterList:
            size = scope == kAudioUnitScope_Global
                 ? static_cast<UInt32>((kNumParams + 1) * sizeof(AudioUnitParameterID))
                 : 0;
            break;
        case kAudioUnitProperty_ParameterInfo:
            size = sizeof(AudioUnitParameterInfo); break;
        case kAudioUnitProperty_ParameterValueStrings:
        {
            const int idx = indexOfParamId(element);
            if (idx < 0 || P::parameters[static_cast<size_t>(idx)].labels == nullptr)
                return kAudioUnitErr_InvalidProperty;
            size = sizeof(CFArrayRef);
            break;
        }
        case kAudioUnitProperty_ParameterStringFromValue:
            size = sizeof(AudioUnitParameterStringFromValue); break;
        case kAudioUnitProperty_ParameterValueFromString:
            size = sizeof(AudioUnitParameterValueFromString); break;
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
#if defined(DSPARK_PLUGIN_WEBVIEW)
        case kAudioUnitProperty_CocoaUI:
            if constexpr (!kHasWebEditor) return kAudioUnitErr_InvalidProperty;
            size = sizeof(AudioUnitCocoaViewInfo);
            break;
        case kDsparkEditorHookProperty:
            if constexpr (!kHasWebEditor) return kAudioUnitErr_InvalidProperty;
            size = sizeof(EditorHook);
            break;
#endif
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
            auto& f = *static_cast<AudioStreamBasicDescription*>(outData);
            fillStreamFormat(f);
            if (scope == kAudioUnitScope_Input && element == 1 && sidechainChannels != 0)
                f.mChannelsPerFrame = sidechainChannels;
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
            *static_cast<UInt32*>(outData) =
                scope == kAudioUnitScope_Input ? kNumInputElements : 1;
            *ioSize = sizeof(UInt32);
            return noErr;
        case kAudioUnitProperty_SupportedNumChannels:
        {
            const UInt32 want =
                static_cast<UInt32>(numChannelConfigs() * sizeof(AUChannelInfo));
            if (*ioSize < want) return kAudioUnitErr_InvalidPropertyValue;
            auto* info = static_cast<AUChannelInfo*>(outData);
            int slot = 0;
            // An instrument declares 0 inputs; effects run symmetric widths.
            if (supportsChannelCount<P>(1))
            {
                info[slot].inChannels = kIsInstrument ? 0 : 1;
                info[slot].outChannels = 1;
                ++slot;
            }
            if (supportsChannelCount<P>(2))
            {
                info[slot].inChannels = kIsInstrument ? 0 : 2;
                info[slot].outChannels = 2;
                ++slot;
            }
            *ioSize = want;
            return noErr;
        }
        case kAudioUnitProperty_HostCallbacks:
            if (*ioSize < sizeof(HostCallbackInfo))
                return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<HostCallbackInfo*>(outData) = hostCallbacks;
            *ioSize = sizeof(HostCallbackInfo);
            return noErr;
        case kAudioUnitProperty_OfflineRender:
            if (*ioSize < sizeof(UInt32)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<UInt32*>(outData) = offlineRender;
            *ioSize = sizeof(UInt32);
            return noErr;
        case kAudioUnitProperty_FactoryPresets:
        {
            if (kNumPresets == 0) return kAudioUnitErr_InvalidProperty;
            if (*ioSize < sizeof(CFArrayRef))
                return kAudioUnitErr_InvalidPropertyValue;
            buildPresetArray();
            if (presetArray == nullptr) return kAudioUnitErr_InvalidProperty;
            *static_cast<CFArrayRef*>(outData) =
                static_cast<CFArrayRef>(CFRetain(presetArray));   // caller releases
            *ioSize = sizeof(CFArrayRef);
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
            // IsHighResolution: a host that limits the control resolution of
            // parameters (automation, control surfaces) gives these the
            // highest it has; the old code left every parameter without it.
            info->flags = kAudioUnitParameterFlag_IsReadable
                        | kAudioUnitParameterFlag_IsWritable
                        | kAudioUnitParameterFlag_HasCFNameString
                        | kAudioUnitParameterFlag_CFNameRelease
                        | kAudioUnitParameterFlag_IsHighResolution;
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
            info->unit = isToggle(spec) ? kAudioUnitParameterUnit_Boolean
                       : spec.steps > 0 && spec.minValue == 0.0f
                             && spec.maxValue == static_cast<float>(spec.steps)
                           ? kAudioUnitParameterUnit_Indexed   // 0..N, choices
                       : (std::strcmp(spec.unit, "dB") == 0
                              ? kAudioUnitParameterUnit_Decibels
                              : kAudioUnitParameterUnit_Generic);
            if (spec.labels != nullptr)
                info->flags |= kAudioUnitParameterFlag_ValuesHaveStrings;
            info->minValue = spec.minValue;
            info->maxValue = spec.maxValue;
            info->defaultValue = spec.defValue;
            *ioSize = sizeof(AudioUnitParameterInfo);
            return noErr;
        }
        case kAudioUnitProperty_ParameterValueStrings:
        {
            // A choice's labels, one CFString per position (caller releases).
            const int idx = indexOfParamId(element);
            if (idx < 0) return kAudioUnitErr_InvalidParameter;
            const Param& spec = P::parameters[static_cast<size_t>(idx)];
            if (spec.labels == nullptr) return kAudioUnitErr_InvalidProperty;
            if (*ioSize < sizeof(CFArrayRef)) return kAudioUnitErr_InvalidPropertyValue;
            CFMutableArrayRef strings = CFArrayCreateMutable(
                nullptr, static_cast<CFIndex>(spec.steps + 1), &kCFTypeArrayCallBacks);
            if (strings == nullptr) return kAudioUnitErr_InvalidProperty;
            for (int i = 0; i <= spec.steps; ++i)
            {
                const char* label = spec.labels[i] != nullptr ? spec.labels[i] : "";
                CFStringRef text = CFStringCreateWithCString(nullptr, label,
                                                             kCFStringEncodingUTF8);
                if (text == nullptr) continue;
                CFArrayAppendValue(strings, text);
                CFRelease(text);
            }
            *static_cast<CFArrayRef*>(outData) = strings;
            *ioSize = sizeof(CFArrayRef);
            return noErr;
        }
        case kAudioUnitProperty_ParameterStringFromValue:
        {
            // Host display text (a choice's label, "3.50 dB"); caller releases.
            if (*ioSize < sizeof(AudioUnitParameterStringFromValue))
                return kAudioUnitErr_InvalidPropertyValue;
            auto* request = static_cast<AudioUnitParameterStringFromValue*>(outData);
            const int idx = indexOfParamId(request->inParamID);
            if (idx < 0) return kAudioUnitErr_InvalidParameter;
            const Param& spec = P::parameters[static_cast<size_t>(idx)];
            const double plain = request->inValue != nullptr
                ? static_cast<double>(*request->inValue)
                : toPlain(spec, shadow[static_cast<size_t>(idx)].load(std::memory_order_relaxed));
            char text[128];
            formatValue(spec, plain, text, static_cast<int>(sizeof(text)));
            request->outString = CFStringCreateWithCString(nullptr, text,
                                                           kCFStringEncodingUTF8);
            *ioSize = sizeof(AudioUnitParameterStringFromValue);
            return noErr;
        }
        case kAudioUnitProperty_ParameterValueFromString:
        {
            // The inverse: labels, On/Off or a number; garbage is refused.
            if (*ioSize < sizeof(AudioUnitParameterValueFromString))
                return kAudioUnitErr_InvalidPropertyValue;
            auto* request = static_cast<AudioUnitParameterValueFromString*>(outData);
            const int idx = indexOfParamId(request->inParamID);
            if (idx < 0) return kAudioUnitErr_InvalidParameter;
            char text[128];
            double plain = 0.0;
            if (request->inString == nullptr
                || !CFStringGetCString(request->inString, text,
                                       static_cast<CFIndex>(sizeof(text)),
                                       kCFStringEncodingUTF8)
                || !parseValue(P::parameters[static_cast<size_t>(idx)], text, plain))
                return kAudioUnitErr_InvalidPropertyValue;
            request->outValue = static_cast<AudioUnitParameterValue>(plain);
            *ioSize = sizeof(AudioUnitParameterValueFromString);
            return noErr;
        }
        case kAudioUnitProperty_Latency:
            if (*ioSize < sizeof(Float64)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<Float64*>(outData) =
                sampleRate > 0.0
                    ? cachedLatency.load(std::memory_order_relaxed) / sampleRate
                    : 0.0;
            *ioSize = sizeof(Float64);
            return noErr;
        case kAudioUnitProperty_TailTime:
        {
            if (*ioSize < sizeof(Float64)) return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<Float64*>(outData) = tailSecondsNow();
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
            preset->presetNumber = currentPresetNumber;
            preset->presetName = copyPresetName();
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
            EditorSize editorSize {};
            const bool withSize = editorMemory.remembered(editorSize);
            const std::vector<uint8_t> blob = buildState(
                user, norm, kNumParams,
                kNumPresets > 0 ? currentPresetNumber : -1,
                bypass.load(std::memory_order_relaxed) ? 1 : 0,
                withSize ? &editorSize : nullptr);

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
            putInt("type", static_cast<SInt32>(expectedType()));
            putInt("subtype", static_cast<SInt32>(subtype));
            putInt("manufacturer", static_cast<SInt32>(manufacturer));
            // "name" is the PRESET's name (the aupreset convention hosts
            // display and the reference implementation writes); the old code
            // wrote the plugin's.
            CFStringRef nameVal = copyPresetName();
            CFDictionarySetValue(dict, CFSTR("name"), nameVal);
            CFRelease(nameVal);
            CFDataRef data = CFDataCreate(nullptr, blob.data(),
                                          static_cast<CFIndex>(blob.size()));
            CFDictionarySetValue(dict, CFSTR("dspark-state"), data);
            CFRelease(data);

            *static_cast<CFPropertyListRef*>(outData) = dict;   // caller releases
            *ioSize = sizeof(CFPropertyListRef);
            return noErr;
        }
#if defined(DSPARK_PLUGIN_WEBVIEW)
        case kAudioUnitProperty_CocoaUI:
        {
            if constexpr (!kHasWebEditor) return kAudioUnitErr_InvalidProperty;
            if (*ioSize < sizeof(AudioUnitCocoaViewInfo))
                return kAudioUnitErr_InvalidPropertyValue;
            Class factory = editorFactoryClass();   // registered before the host
            CFURLRef url = copyOwningBundleUrl();   // looks the class name up
            if (factory == nullptr || url == nullptr)
            {
                if (url != nullptr) CFRelease(url);
                return kAudioUnitErr_InvalidProperty;
            }
            auto* viewInfo = static_cast<AudioUnitCocoaViewInfo*>(outData);
            viewInfo->mCocoaAUViewBundleLocation = url;   // caller releases both
            viewInfo->mCocoaAUViewClass[0] = CFStringCreateWithCString(
                nullptr, class_getName(factory), kCFStringEncodingUTF8);
            *ioSize = sizeof(AudioUnitCocoaViewInfo);
            return noErr;
        }
        case kDsparkEditorHookProperty:
        {
            if constexpr (!kHasWebEditor) return kAudioUnitErr_InvalidProperty;
            if (*ioSize < sizeof(EditorHook))
                return kAudioUnitErr_InvalidPropertyValue;
            *static_cast<EditorHook*>(outData) = EditorHook { this, &sCreateEditorView };
            *ioSize = sizeof(EditorHook);
            return noErr;
        }
#endif
        default:
            return kAudioUnitErr_InvalidProperty;
        }
    }

    OSStatus setProperty(AudioUnitPropertyID id, AudioUnitScope scope,
                         AudioUnitElement element, const void* inData,
                         UInt32 inSize) noexcept
    {
        switch (id)
        {
        case kAudioUnitProperty_StreamFormat:
        {
            if (inData == nullptr || inSize < sizeof(AudioStreamBasicDescription))
                return kAudioUnitErr_InvalidPropertyValue;
            const auto* f = static_cast<const AudioStreamBasicDescription*>(inData);
            if (!acceptableFormat(*f)) return kAudioUnitErr_FormatNotSupported;
            // While initialized the format is fixed: the plugin was prepared
            // for this rate and width. Re-applying the same format is fine;
            // a different one is refused, and the host uninitializes, sets
            // it and initializes again. Accepting it (the old behaviour) let
            // processBlock see a width it was never prepared for and made
            // kAudioUnitProperty_Latency divide by a rate the DSP never ran.
            if (HasSidechain<P> && scope == kAudioUnitScope_Input && element == 1)
            {
                // The sidechain's own width: a mono key on a stereo main is
                // pulled mono and fed to both detector ears. It never sets
                // the main width - the old code let this element switch the
                // whole unit, and a stereo track then rendered one channel.
                const UInt32 current = sidechainChannels != 0
                    ? sidechainChannels : static_cast<UInt32>(currentChannels);
                if (initialized && f->mChannelsPerFrame != current)
                    return kAudioUnitErr_PropertyNotWritable;
                sidechainChannels = f->mChannelsPerFrame;
                return noErr;
            }
            if (initialized
                && (f->mSampleRate != sampleRate
                    || static_cast<int>(f->mChannelsPerFrame) != currentChannels))
                return kAudioUnitErr_PropertyNotWritable;
            sampleRate = f->mSampleRate;
            // The negotiated width: the main elements follow it (initialize()
            // prepares the plugin at this count).
            currentChannels = static_cast<int>(f->mChannelsPerFrame);
            return noErr;
        }
        case kAudioUnitProperty_SampleRate:
        {
            if (inData == nullptr || inSize < sizeof(Float64))
                return kAudioUnitErr_InvalidPropertyValue;
            const Float64 rate = *static_cast<const Float64*>(inData);
            if (!(rate > 0.0)) return kAudioUnitErr_FormatNotSupported;   // NaN too
            if (initialized && rate != sampleRate)
                return kAudioUnitErr_PropertyNotWritable;   // see StreamFormat
            sampleRate = rate;
            return noErr;
        }
        case kAudioUnitProperty_MaximumFramesPerSlice:
        {
            if (inData == nullptr || inSize < sizeof(UInt32))
                return kAudioUnitErr_InvalidPropertyValue;
            const UInt32 frames = *static_cast<const UInt32*>(inData);
            if (frames == maxFrames) return noErr;
            // initialize() sizes the buffers for this value, so a change
            // while initialized is refused (kAudioUnitErr_Initialized tells
            // the host to uninitialize, set it and initialize again).
            // Accepting it (the old behaviour) answered noErr and then
            // failed every slice longer than the allocation with
            // kAudioUnitErr_TooManyFramesToProcess: silence. render() still
            // refuses such a slice as a last guard.
            if (initialized) return kAudioUnitErr_Initialized;
            maxFrames = frames;
            notifyProperty(id, scope, element);
            return noErr;
        }
        case kAudioUnitProperty_PresentPreset:
        case kAudioUnitProperty_CurrentPreset:
        {
            if (inData == nullptr || inSize < sizeof(AUPreset))
                return kAudioUnitErr_InvalidPropertyValue;
            const auto* preset = static_cast<const AUPreset*>(inData);
            if (preset->presetNumber >= 0)
            {
                // A factory preset by number: apply its parameter values.
                if (preset->presetNumber >= kNumPresets)
                    return kAudioUnitErr_InvalidPropertyValue;
                applyFactoryPresetIdx(preset->presetNumber);
                refreshReported();
                notifyAllParameters();
                if (presetName != nullptr) CFRelease(presetName);
                presetName = preset->presetName != nullptr
                    ? static_cast<CFStringRef>(CFRetain(preset->presetName))
                    : CFStringCreateWithCString(
                          nullptr,
                          [&]() -> const char* {
                              if constexpr (kNumPresets > 0)
                                  return P::factoryPresets[static_cast<size_t>(
                                      preset->presetNumber)].name;
                              else
                                  return "";
                          }(),
                          kCFStringEncodingUTF8);
                notifyProperty(id, scope, element);
                return noErr;
            }
            currentPresetNumber = -1;   // back to a user preset
            if (presetName != nullptr) CFRelease(presetName);
            presetName = preset->presetName != nullptr
                ? static_cast<CFStringRef>(CFRetain(preset->presetName))
                : nullptr;
            notifyProperty(id, scope, element);
            return noErr;
        }
        case kAudioUnitProperty_HostCallbacks:
        {
            if (inData == nullptr) return kAudioUnitErr_InvalidPropertyValue;
            // Hosts may pass a truncated struct (older callback sets).
            std::memset(&hostCallbacks, 0, sizeof(hostCallbacks));
            std::memcpy(&hostCallbacks, inData,
                        inSize < sizeof(hostCallbacks) ? inSize
                                                       : sizeof(hostCallbacks));
            return noErr;
        }
        case kAudioUnitProperty_OfflineRender:
            if (inData == nullptr || inSize < sizeof(UInt32))
                return kAudioUnitErr_InvalidPropertyValue;
            offlineRender = *static_cast<const UInt32*>(inData);
            if constexpr (HasOfflineMode<P>)
                user.setOfflineRendering(offlineRender != 0);
            return noErr;
        case kAudioUnitProperty_BypassEffect:
            if (inData == nullptr || inSize < sizeof(UInt32))
                return kAudioUnitErr_InvalidPropertyValue;
            // The bypass is also a parameter: its listeners (an automation
            // lane, the generic view) follow the host's button. The old code
            // left them showing the previous state.
            if (setBypass(*static_cast<const UInt32*>(inData) != 0))
                notifyParameterChanged(kBypassParamId);
            return noErr;
        case kAudioUnitProperty_SetRenderCallback:
        {
            // Instruments have no input elements yet must still ACCEPT a
            // render callback (auval and some hosts set one regardless);
            // it is stored and never pulled.
            constexpr UInt32 storable = kIsInstrument ? 2 : kNumInputElements;
            if (scope != kAudioUnitScope_Input || element >= storable
                || inData == nullptr || inSize < sizeof(AURenderCallbackStruct))
                return kAudioUnitErr_InvalidPropertyValue;
            inputCallback[element] = *static_cast<const AURenderCallbackStruct*>(inData);
            inputConnection[element] = nullptr;
            return noErr;
        }
        case kAudioUnitProperty_MakeConnection:
        {
            if (inData == nullptr || inSize < sizeof(AudioUnitConnection))
                return kAudioUnitErr_InvalidPropertyValue;
            const auto* c = static_cast<const AudioUnitConnection*>(inData);
            constexpr UInt32 storable = kIsInstrument ? 2 : kNumInputElements;
            if (c->destInputNumber >= storable)
                return kAudioUnitErr_InvalidPropertyValue;
            inputConnection[c->destInputNumber] = c->sourceAudioUnit;
            inputConnectionBus[c->destInputNumber] = c->sourceOutputNumber;
            inputCallback[c->destInputNumber] = {};
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
            // A preset of another unit is refused, as the format requires: the old
            // code loaded any dictionary carrying this format's key - a
            // preset of a different DSPark plugin fed this one its user
            // section.
            auto numberIs = [&](CFStringRef key, OSType expected) {
                auto number = static_cast<CFNumberRef>(CFDictionaryGetValue(dict, key));
                if (number == nullptr) return true;   // absent: tolerated
                SInt32 value = 0;
                return CFGetTypeID(number) == CFNumberGetTypeID()
                    && CFNumberGetValue(number, kCFNumberSInt32Type, &value)
                    && static_cast<OSType>(value) == expected;
            };
            if (!numberIs(CFSTR("subtype"), subtype)
                || !numberIs(CFSTR("manufacturer"), manufacturer))
                return kAudioUnitErr_InvalidPropertyValue;
            auto data = static_cast<CFDataRef>(
                CFDictionaryGetValue(dict, CFSTR("dspark-state")));
            if (data == nullptr || CFGetTypeID(data) != CFDataGetTypeID())
                return noErr;   // foreign preset: keep current state
            // Parameters the blob does not carry take their DEFAULTS (the old
            // code kept whatever value the instance happened to hold).
            double norm[kNumParams == 0 ? 1 : kNumParams];
            loadDefaultNormalized<P>(norm);
            int program = -1;
            int bypassState = -1;
            EditorSize editorSize { 0, 0 };
            if (applyState(user,
                           CFDataGetBytePtr(data),
                           static_cast<size_t>(CFDataGetLength(data)), norm,
                           &program, &bypassState, &editorSize))
            {
                for (size_t i = 0; i < kNumParams; ++i)
                    applyFromAnyThread(static_cast<int>(i), norm[i]);
                currentPresetNumber =
                    kNumPresets > 0 && program >= 0 && program < kNumPresets ? program : -1;
                // The preset name travels with the state (the host shows it).
                auto name = static_cast<CFStringRef>(CFDictionaryGetValue(dict, CFSTR("name")));
                if (presetName != nullptr) CFRelease(presetName);
                presetName = name != nullptr && CFGetTypeID(name) == CFStringGetTypeID()
                           ? static_cast<CFStringRef>(CFRetain(name)) : nullptr;
                if (bypassState >= 0 && setBypass(bypassState != 0))
                    announce(kNotifyBypass);
                if (editorSize.width > 0)
                    editorMemory.remember(editorSize.width, editorSize.height);
                refreshReported();   // restored state may imply a new lookahead
                notifyAllParameters();
                notifyProperty(kAudioUnitProperty_PresentPreset, kAudioUnitScope_Global, 0);
            }
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
        // A non-finite value is refused, as the format requires: the plain ->
        // normalized mapping sends NaN to 0 and +Inf to 1, so the old code
        // moved the parameter to an end of its range.
        if (!std::isfinite(value)) return kAudioUnitErr_InvalidParameterValue;
        if (id == kBypassParamId)
        {
            if (setBypass(value >= 0.5f)) announce(kNotifyBypass);
            return noErr;
        }
        const int idx = indexOfParamId(id);
        if (idx < 0) return kAudioUnitErr_InvalidParameter;
        if (applyFromAnyThread(idx,
                toNormalized(P::parameters[static_cast<size_t>(idx)], value)))
            refreshReported();
        return noErr;
    }

    // --- MIDI (MusicDevice selectors -> lock-free ring -> render) ---------------------

    /** Parses one raw MIDI message and queues it for the next render. */
    OSStatus pushMidi(UInt32 status, UInt32 data1, UInt32 data2,
                      UInt32 offsetFrame) noexcept
    {
        if constexpr (HasMidi<P>)
        {
            MidiEvent ev {};
            ev.channel = static_cast<uint8_t>(status & 0x0Fu);
            ev.sampleOffset = static_cast<int>(offsetFrame);
            const uint8_t d1 = static_cast<uint8_t>(data1 & 0x7Fu);
            const uint8_t d2 = static_cast<uint8_t>(data2 & 0x7Fu);
            switch (status & 0xF0u)
            {
            case 0x90:   // wire convention: velocity 0 means note off
                ev.type = d2 > 0 ? MidiEvent::Type::NoteOn : MidiEvent::Type::NoteOff;
                ev.note = d1;
                ev.value = static_cast<float>(d2) / 127.0f;
                break;
            case 0x80:
                ev.type = MidiEvent::Type::NoteOff;
                ev.note = d1;
                ev.value = static_cast<float>(d2) / 127.0f;
                break;
            case 0xA0:
                ev.type = MidiEvent::Type::PolyPressure;
                ev.note = d1;
                ev.value = static_cast<float>(d2) / 127.0f;
                break;
            case 0xB0:
                ev.type = MidiEvent::Type::ControlChange;
                ev.note = d1;
                ev.value = static_cast<float>(d2) / 127.0f;
                break;
            case 0xD0:
                ev.type = MidiEvent::Type::ChannelPressure;
                ev.value = static_cast<float>(d1) / 127.0f;
                break;
            case 0xE0:
                ev.type = MidiEvent::Type::PitchBend;
                ev.value = (static_cast<float>((d2 << 7) | d1) - 8192.0f) / 8192.0f;
                break;
            default:
                return noErr;   // system messages: accepted, ignored
            }
            const uint32_t tail = midiTail.load(std::memory_order_relaxed);
            const uint32_t next = (tail + 1) & (kMidiRingSize - 1);
            if (next == midiHead.load(std::memory_order_acquire))
                return noErr;   // ring full: drop (never block the MIDI thread)
            midiRing[tail] = ev;
            midiTail.store(next, std::memory_order_release);
            return noErr;
        }
        else
        {
            (void) status;
            (void) data1;
            (void) data2;
            (void) offsetFrame;
            return kAudioUnitErr_InvalidProperty;
        }
    }

    /** Forwards the host-callback transport once per render (HasTransport). */
    void forwardTransport() noexcept
    {
        if constexpr (HasTransport<P>)
        {
            TransportInfo info {};
            bool any = false;
            if (hostCallbacks.beatAndTempoProc != nullptr)
            {
                Float64 beat = 0.0, tempo = 0.0;
                if (hostCallbacks.beatAndTempoProc(hostCallbacks.hostUserData,
                                                   &beat, &tempo) == noErr)
                {
                    if (tempo > 0.0)
                    {
                        info.tempoBpm = tempo;
                        info.tempoValid = true;
                    }
                    info.ppqPosition = beat;
                    info.positionValid = true;
                    any = true;
                }
            }
            if (hostCallbacks.musicalTimeLocationProc != nullptr)
            {
                UInt32 deltaToNextBeat = 0;
                Float32 num = 4.0f;
                UInt32 den = 4;
                Float64 downbeat = 0.0;
                if (hostCallbacks.musicalTimeLocationProc(
                        hostCallbacks.hostUserData, &deltaToNextBeat, &num, &den,
                        &downbeat) == noErr)
                {
                    // The host WRITES the numerator, and this is the one
                    // format that delivers it as a floating-point number
                    // (VST3 sends an int32, CLAP a uint16), so it is the one
                    // place the value has to be bounded before it is
                    // converted: a conversion that does not fit an int is
                    // undefined, and a NaN would sail through a test written
                    // the obvious way. The ceiling is far above any meter a
                    // sequencer can express and the floor keeps the field
                    // usable as a divisor.
                    info.timeSigNumerator = toBoundedInt(num, 1, 255);
                    info.timeSigDenominator = static_cast<int>(den);
                    info.timeSigValid = true;
                    info.barStartPpq = downbeat;
                    any = true;
                }
            }
            if (hostCallbacks.transportStateProc != nullptr)
            {
                Boolean isPlaying = false, changed = false, isCycling = false;
                Float64 samplePos = 0.0, cycleStart = 0.0, cycleEnd = 0.0;
                if (hostCallbacks.transportStateProc(
                        hostCallbacks.hostUserData, &isPlaying, &changed,
                        &samplePos, &isCycling, &cycleStart, &cycleEnd) == noErr)
                {
                    info.playing = isPlaying;
                    info.looping = isCycling;
                    info.loopStartPpq = cycleStart;
                    info.loopEndPpq = cycleEnd;
                    info.loopValid = isCycling;
                    any = true;
                }
            }
            if (any)
                user.setTransport(info);
        }
    }

    // --- render ----------------------------------------------------------------------

    /** Pulls one input element through its registered callback or connection
     *  into @p dst (silence when neither is set), at the negotiated width.
     *  On success @p dst may be repointed at buffers the source substituted. */
    OSStatus pullInput(UInt32 element, float* dst[2],
                       const AudioTimeStamp* timeStamp, UInt32 frames) noexcept
    {
        // AudioBufferList declares mBuffers[1] (variable length): a stack
        // instance has storage for ONE AudioBuffer, so writing mBuffers[1]
        // corrupts the frame. Give the list real storage and lend it out
        // as AudioBufferList* - the canonical CoreAudio pattern.
        struct StereoBufferList
        {
            UInt32 mNumberBuffers;
            ::AudioBuffer mBuffers[2];   // CoreAudio's, not dspark::AudioBuffer
        } list {};
        // The sidechain element pulls at its own negotiated width.
        const UInt32 elementWidth = element == 1 && sidechainChannels != 0
                                  ? sidechainChannels
                                  : static_cast<UInt32>(currentChannels);
        const UInt32 width = elementWidth < 2u ? elementWidth : 2u;
        list.mNumberBuffers = width;
        for (UInt32 ch = 0; ch < width; ++ch)
        {
            list.mBuffers[ch].mNumberChannels = 1;
            list.mBuffers[ch].mDataByteSize = frames * sizeof(float);
            list.mBuffers[ch].mData = dst[ch];
        }
        auto* abl = reinterpret_cast<AudioBufferList*>(&list);
        OSStatus status = noErr;
        AudioUnitRenderActionFlags flags = 0;
        if (inputCallback[element].inputProc != nullptr)
            status = inputCallback[element].inputProc(
                inputCallback[element].inputProcRefCon, &flags, timeStamp,
                element, frames, abl);
        else if (inputConnection[element] != nullptr)
            status = AudioUnitRender(inputConnection[element], &flags, timeStamp,
                                     inputConnectionBus[element], frames, abl);
        else
            for (UInt32 ch = 0; ch < width; ++ch)
                std::memset(dst[ch], 0, frames * sizeof(float));
        if (status != noErr) return status;
        for (UInt32 ch = 0; ch < width && ch < list.mNumberBuffers; ++ch)
            if (list.mBuffers[ch].mData != nullptr)
                dst[ch] = static_cast<float*>(list.mBuffers[ch].mData);
        if (width == 1 && currentChannels > 1)
            dst[1] = dst[0];   // a mono key feeds both detector ears
        return noErr;
    }

    /** Expands the slice's scheduled ramps into parameter points at the
     *  slice start, at every automation-quantum boundary the ramp crosses
     *  and at its end, each carrying the ramp's linear value at that offset
     *  (AUBase semantics: startBufferOffset may be negative for a ramp that
     *  began in an earlier slice). The old code pushed the start value at
     *  the start offset and the END value at start+duration clamped into
     *  the slice, so a one-second ramp jumped to its final value within the
     *  first slice, and a re-issued ramp swung start->end in every slice. */
    void expandRamps(int frames) noexcept
    {
        for (int r = 0; r < rampCount; ++r)
        {
            const Ramp& ramp = ramps[r];
            const int idx = indexOfParamId(ramp.paramId);
            if (idx < 0) continue;
            const long long start = ramp.startOffset;
            const long long end = start + static_cast<long long>(ramp.duration);
            auto valueAt = [&](long long offset) {
                if (ramp.duration == 0 || offset >= end) return static_cast<double>(ramp.endValue);
                const double t = static_cast<double>(offset - start)
                               / static_cast<double>(ramp.duration);
                const double u = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
                return ramp.startValue + (static_cast<double>(ramp.endValue) - ramp.startValue) * u;
            };
            auto push = [&](long long offset) {
                BlockEvent ev {};
                ev.offset = static_cast<int32_t>(offset);
                ev.kind = BlockEvent::Kind::Param;
                ev.paramId = ramp.paramId;
                ev.value = toNormalized(P::parameters[static_cast<size_t>(idx)], valueAt(offset));
                blockEvents.add(ev);
            };
            long long first = start > 0 ? start : 0;
            if (first >= frames) continue;
            push(first);
            for (long long q = (first / kAutomationQuantum + 1) * kAutomationQuantum;
                 q < frames && q < end; q += kAutomationQuantum)
                push(q);
            if (end > first && end < frames && end % kAutomationQuantum != 0)
                push(end);
        }
        rampCount = 0;
    }

    /** Crossfades @p length output frames from @p start toward the dry copy
     *  at the current bypass target. */
    void blendBypass(float* const* out, float* const* dry, UInt32 outCh, int start,
                     int length) noexcept
    {
        const float target = bypass.load(std::memory_order_relaxed) ? 1.0f : 0.0f;
        if (bypassMix == target && target == 0.0f) return;
        const float step = 1.0f / static_cast<float>(kBypassRampSamples);
        float mix = bypassMix;
        for (int i = 0; i < length; ++i)
        {
            mix += (target > mix) ? step : ((target < mix) ? -step : 0.0f);
            mix = mix < 0.0f ? 0.0f : (mix > 1.0f ? 1.0f : mix);
            for (UInt32 ch = 0; ch < outCh; ++ch)
                out[ch][start + i] += (dry[ch][start + i] - out[ch][start + i]) * mix;
        }
        bypassMix = mix;
    }

    /**
     * The render selector: validates the call, runs the host's render
     * notifications around the slice (pre-render first: hosts schedule the
     * slice's parameters from it) and renders. As in the format's reference
     * implementation, a refused call
     * still ends with a post-render notification carrying the error flag,
     * and the events scheduled for a slice that does not render never reach
     * a later one (see discardSlice).
     */
    OSStatus render(AudioUnitRenderActionFlags* ioFlags,
                    const AudioTimeStamp* timeStamp, UInt32 busNumber,
                    UInt32 frames, AudioBufferList* ioData) noexcept
    {
        if (ioData == nullptr)
        {
            discardSlice();
            return kAudio_ParamError;
        }
        // One output element. Rendering any other (the old code ignored the
        // number) ran the DSP a second time in the same cycle.
        if (busNumber != 0)
        {
            discardSlice();
            return kAudioUnitErr_InvalidElement;
        }
        if (!initialized)
        {
            discardSlice();
            return kAudioUnitErr_Uninitialized;
        }

        // FTZ/DAZ for the whole callback (the notifications included):
        // DSPark processors guard their own hot loops, this covers user DSP
        // in hosts that do not set it.
        dspark::DenormalGuard denormalGuard;

        applyNotifyCommands();   // this thread owns the notification list
        using Flags = AudioUnitRenderActionFlags;
        const Flags flags = ioFlags != nullptr ? *ioFlags : Flags { 0 };
        constexpr Flags kPre = static_cast<Flags>(kAudioUnitRenderAction_PreRender);
        constexpr Flags kPost = static_cast<Flags>(kAudioUnitRenderAction_PostRender);
        constexpr Flags kError = static_cast<Flags>(kAudioUnitRenderAction_PostRenderError);
        auto refuse = [&](OSStatus error) noexcept {
            discardSlice();
            callRenderNotifies(flags | kPost | kError, timeStamp, busNumber, frames, ioData);
            return error;
        };
        // Bound by the property and by what initialize() allocated (the
        // property cannot change while initialized; the second test is a
        // last guard for the buffers).
        if (frames > maxFrames || frames > pullL.size())
            return refuse(kAudioUnitErr_TooManyFramesToProcess);
        // A host buffer shorter than the slice is refused, as the format requires:
        // the old code wrote `frames` samples into it regardless.
        const UInt32 bytes = frames * static_cast<UInt32>(sizeof(float));
        for (UInt32 b = 0; b < ioData->mNumberBuffers; ++b)
            if (ioData->mBuffers[b].mData != nullptr && ioData->mBuffers[b].mDataByteSize < bytes)
                return refuse(kAudio_ParamError);

        callRenderNotifies(flags | kPre, timeStamp, busNumber, frames, ioData);
        OSStatus status = noErr;
        if (frames == 0)
            discardSlice();
        else
            status = renderSlice(timeStamp, frames, ioData);
        callRenderNotifies(flags | kPost | (status != noErr ? kError : Flags { 0 }),
                           timeStamp, busNumber, frames, ioData);
        return status;
    }

    /** One validated slice of @p frames (at least one). */
    OSStatus renderSlice(const AudioTimeStamp* timeStamp, UInt32 frames,
                         AudioBufferList* ioData) noexcept
    {
        // This thread now owns setParameter; first apply what other threads
        // handed over, so the slice's scheduled events below still win.
        renderThread.store(std::this_thread::get_id(), std::memory_order_relaxed);
        bool paramsChanged = drainPending();
        bool bypassChanged = false;

        forwardTransport();

        // One timestamped stream: scheduled parameter events and ramps
        // (already on this thread) plus the queued MIDI from the MusicDevice
        // selectors.
        expandRamps(static_cast<int>(frames));
        if constexpr (HasMidi<P>)
        {
            uint32_t head = midiHead.load(std::memory_order_relaxed);
            const uint32_t tail = midiTail.load(std::memory_order_acquire);
            if (midiDiscard.exchange(false, std::memory_order_acquire))
            {
                // Skip what a Reset found queued - only forward: a slice
                // rendered between the Reset's two stores may be past it.
                const uint32_t to = midiDiscardTo.load(std::memory_order_relaxed);
                constexpr uint32_t mask = kMidiRingSize - 1;
                if (((to - head) & mask) <= ((tail - head) & mask)) head = to;
            }
            while (head != tail)
            {
                BlockEvent ev {};
                ev.kind = BlockEvent::Kind::Midi;
                ev.midi = midiRing[head];
                ev.offset = ev.midi.sampleOffset;
                blockEvents.add(ev);
                head = (head + 1) & (kMidiRingSize - 1);
            }
            midiHead.store(head, std::memory_order_release);
        }
        BlockEventList& events = blockEvents;
        for (int i = 0; i < events.size(); ++i)
        {
            if (events[i].offset >= static_cast<int32_t>(frames))
                events[i].offset = static_cast<int32_t>(frames) - 1;
            if (events[i].offset < 0) events[i].offset = 0;
        }
        events.sort();
        const int eventCount = events.size();

        auto applyEvent = [&](const BlockEvent& ev, int blockStart) noexcept -> bool {
            switch (ev.kind)
            {
            case BlockEvent::Kind::Bypass:
                bypassChanged |= setBypass(ev.value >= 0.5);
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
        };
        auto finish = [&]() noexcept {
            events.clear();
            if (paramsChanged) refreshReported();
            if (bypassChanged) announce(kNotifyBypass);
        };

        const UInt32 width = static_cast<UInt32>(currentChannels);
        const UInt32 outCh = ioData->mNumberBuffers < width
                           ? ioData->mNumberBuffers : width;
        if (outCh < 1)
        {
            for (int i = 0; i < eventCount; ++i)
                paramsChanged |= applyEvent(events[i], events[i].offset);
            finish();
            return noErr;
        }

        // Pull the main input through the registered callback or connection
        // into our own buffers (instruments have no input: silence), then
        // process and write to the host's buffers.
        float* pull[2] = { pullL.data(), pullR.data() };
        if constexpr (!kIsInstrument)
        {
            const OSStatus pullStatus = pullInput(0, pull, timeStamp, frames);
            if (pullStatus != noErr)
            {
                for (int i = 0; i < eventCount; ++i)   // never lose the slice's events
                    paramsChanged |= applyEvent(events[i], events[i].offset);
                finish();
                return pullStatus;
            }
        }

        float* out[2] = { nullptr, nullptr };
        for (UInt32 ch = 0; ch < outCh; ++ch)
        {
            if (ioData->mBuffers[ch].mData == nullptr)
                ioData->mBuffers[ch].mData = pull[ch];
            out[ch] = static_cast<float*>(ioData->mBuffers[ch].mData);
            ioData->mBuffers[ch].mDataByteSize = frames * sizeof(float);
        }

        // Dry copy for the bypass blend; an instrument starts cleared
        // (voices ADD) and its bypass blends toward silence (dry stays 0).
        float* dry[2] = { dryL.data(), dryR.data() };
        for (UInt32 ch = 0; ch < outCh; ++ch)
        {
            if (kIsInstrument)
            {
                std::memset(out[ch], 0, frames * sizeof(float));
                continue;
            }
            std::memcpy(dry[ch], pull[ch], frames * sizeof(float));
            if (out[ch] != pull[ch])
                std::memcpy(out[ch], pull[ch], frames * sizeof(float));
        }
        if constexpr (HasLatency<P> && !kIsInstrument)
            dryDelay.process(dry, static_cast<int>(outCh), static_cast<int>(frames),
                             cachedLatency.load(std::memory_order_relaxed));

        // Sidechain: input element 1; a missing or failing source must
        // never take the main path down - fall back to silence.
        float* sc[2] = { nullptr, nullptr };
        if constexpr (HasSidechain<P>)
        {
            sc[0] = scL.data();
            sc[1] = scR.data();
            if (pullInput(1, sc, timeStamp, frames) != noErr)
            {
                sc[0] = scL.data();
                sc[1] = scR.data();
                std::memset(sc[0], 0, frames * sizeof(float));
                std::memset(sc[1], 0, frames * sizeof(float));
            }
        }

        // Sub-block processing at quantum-aligned event positions (the
        // sample-accurate default); opted out, everything applies up front.
        auto processSegment = [&](int start, int length) noexcept {
            float* sub[2] = { out[0] + start,
                              outCh > 1 ? out[1] + start : out[0] + start };
            dspark::AudioBufferView<float> view(sub, static_cast<int>(outCh),
                                                length);
            if constexpr (HasSidechain<P>)
            {
                // The key view mirrors the main width (mono main, mono key).
                float* scSub[2] = { sc[0] + start, sc[1] + start };
                dspark::AudioBufferView<float> scView(scSub,
                                                      static_cast<int>(outCh), length);
                user.processBlock(view, scView);
            }
            else
                user.processBlock(view);
            // Soft bypass per segment: a scheduled bypass point starts its
            // crossfade where it lands (the old once-per-slice blend began
            // it at frame 0).
            blendBypass(out, dry, outCh, start, length);
        };

        const int total = static_cast<int>(frames);
        int evIdx = 0;
        if (!sampleAccurateOf<P>())
        {
            for (; evIdx < eventCount; ++evIdx)
                paramsChanged |= applyEvent(events[evIdx], 0);
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
                    paramsChanged |= applyEvent(events[evIdx++], pos);
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
                paramsChanged |= applyEvent(events[evIdx], total);
        }

        finish();
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

    /** Render notifications are honoured (the old selectors accepted and
     *  dropped them, so a host scheduling parameters or metering from a
     *  pre/post-render callback got nothing). */
    static OSStatus sAddRenderNotify(void* self, AURenderCallback proc, void* refCon) noexcept
    { return fromSelf(self)->state->queueRenderNotify(proc, refCon, true); }

    static OSStatus sRemoveRenderNotify(void* self, AURenderCallback proc, void* refCon) noexcept
    { return fromSelf(self)->state->queueRenderNotify(proc, refCon, false); }

    /** Scheduled parameter events keep their buffer offsets: AUv2 calls
     *  this on the render thread right before the matching render, so the
     *  events ride the same timestamped stream as MIDI and split processing
     *  sample-accurately. A ramp is evaluated along its line at the slice
     *  start, at every automation quantum it crosses and at its end. */
    static OSStatus sScheduleParameters(void* self,
                                        const AudioUnitParameterEvent* events,
                                        UInt32 numEvents) noexcept
    {
        auto* state = fromSelf(self)->state;
        // A non-finite value refuses the whole call, as the format requires (NaN
        // used to map to the bottom of the range).
        for (UInt32 i = 0; events != nullptr && i < numEvents; ++i)
        {
            const auto& ev = events[i];
            if (ev.eventType == kParameterEvent_Immediate
                    ? !std::isfinite(ev.eventValues.immediate.value)
                    : (ev.eventType == kParameterEvent_Ramped
                       && (!std::isfinite(ev.eventValues.ramp.startValue)
                           || !std::isfinite(ev.eventValues.ramp.endValue))))
                return kAudioUnitErr_InvalidParameterValue;
        }
        // Immediate points (bypass included, as a timestamped event so its
        // crossfade starts where it lands) go into the block list; the list
        // merges past its capacity instead of dropping the newest points
        // (the old fixed array ignored every event after the 256th).
        auto push = [&](AudioUnitParameterID id, float plain, SInt32 offset, bool immediate) {
            BlockEvent ev {};
            ev.offset = offset < 0 ? 0 : offset;
            if (id == kBypassParamId)
            {
                ev.kind = BlockEvent::Kind::Bypass;
                ev.paramId = kBypassParamId;
                ev.value = plain >= 0.5f ? 1.0 : 0.0;
            }
            else
            {
                const int idx = Plugin<P>::indexOfParamId(id);
                if (idx < 0) return;
                ev.kind = BlockEvent::Kind::Param;
                ev.paramId = id;
                ev.value = toNormalized(P::parameters[static_cast<size_t>(idx)], plain);
                // The host reads an immediate value back at once (the format sets
                // it when scheduled); the DSP still hears it at its offset.
                if (immediate)
                    state->shadow[static_cast<size_t>(idx)].store(ev.value,
                                                                  std::memory_order_relaxed);
            }
            state->blockEvents.add(ev);
        };
        for (UInt32 i = 0; events != nullptr && i < numEvents; ++i)
        {
            const auto& ev = events[i];
            if (ev.scope != kAudioUnitScope_Global) continue;
            if (ev.eventType == kParameterEvent_Immediate)
                push(ev.parameter, ev.eventValues.immediate.value,
                     static_cast<SInt32>(ev.eventValues.immediate.bufferOffset), true);
            else if (ev.eventType == kParameterEvent_Ramped)
            {
                // Kept whole: render() evaluates it against the slice length.
                if (ev.parameter != kBypassParamId && state->rampCount < Plugin<P>::kMaxRamps)
                    state->ramps[state->rampCount++] = typename Plugin<P>::Ramp {
                        ev.parameter, ev.eventValues.ramp.startBufferOffset,
                        ev.eventValues.ramp.durationInFrames,
                        ev.eventValues.ramp.startValue, ev.eventValues.ramp.endValue };
                else
                    push(ev.parameter, ev.eventValues.ramp.endValue,
                         ev.eventValues.ramp.startBufferOffset
                             + static_cast<SInt32>(ev.eventValues.ramp.durationInFrames),
                         false);
            }
        }
        return noErr;
    }

    // MusicDevice selectors (aumu / aumf): raw MIDI in, queued lock-free.

    static OSStatus sMidiEvent(void* self, UInt32 status, UInt32 data1,
                               UInt32 data2, UInt32 offsetSampleFrame) noexcept
    {
        return fromSelf(self)->state->pushMidi(status, data1, data2,
                                               offsetSampleFrame);
    }

    static OSStatus sSysEx(void* self, const UInt8*, UInt32) noexcept
    {
        (void) self;
        return noErr;   // accepted, ignored (no sysex contract in v1)
    }

    static AudioComponentMethod sLookup(SInt16 selector) noexcept
    {
        if constexpr (HasMidi<P>)
        {
            if (selector == kMusicDeviceMIDIEventSelect)
                return reinterpret_cast<AudioComponentMethod>(&sMidiEvent);
            if (selector == kMusicDeviceSysExSelect)
                return reinterpret_cast<AudioComponentMethod>(&sSysEx);
        }
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
        // The Info.plist entry must declare the type the class implies:
        // aumu (Instrument), aumf (HasMidi effect) or aufx.
        if (desc != nullptr && desc->componentType != Plugin<P>::expectedType())
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
 * must match the bundle's Info.plist AudioComponents entry - and the
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

#pragma pop_macro("max")
#pragma pop_macro("min")

#else // !__APPLE__

// Self-disabling off macOS so one translation unit can target every format.
#define DSPARK_AU_PLUGIN(PluginClass, subtype4, manufacturer4)

#endif // __APPLE__
