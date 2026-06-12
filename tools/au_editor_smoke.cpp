// DSPark — AUv2 WebView editor smoke host (macOS)
//
// Exercises the Cocoa editor contract exactly like an AU host: reads
// kAudioUnitProperty_CocoaUI, resolves the announced view factory class,
// asks it for the editor NSView, verifies the embedded WKWebView completes
// the JS bridge handshake (through the DSPARK_WEBVIEW_LOG diagnostic trace),
// then tears down in BOTH orders — view before AudioUnit and AudioUnit
// before view. Plugins without an editor are checked to NOT announce one.
// Exit code 0 means every step behaved. Companion to vst3/clap_smoke_host;
// auval remains the format-conformance gate.
//
//   au_editor_smoke <type4> <subtype4> <manu4> --expect-editor <W>x<H>
//   au_editor_smoke <type4> <subtype4> <manu4> --expect-no-editor

#if !defined(__APPLE__)

#include <cstdio>
int main()
{
    std::printf("au_editor_smoke: macOS only (no-op on this platform)\n");
    return 0;
}

#else

#include <AudioToolbox/AudioToolbox.h>
#include <CoreFoundation/CoreFoundation.h>

#include <objc/message.h>
#include <objc/runtime.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" unsigned char NSApplicationLoad(void);   // AppKit, CLI-tool bootstrap

namespace {

int g_failures = 0;

void expect(bool ok, const char* what)
{
    std::printf("%s  %s\n", ok ? "PASS " : "FAIL ", what);
    if (!ok) ++g_failures;
}

// -- minimal Objective-C runtime glue (mirrors the plugin's own) -------------------

using ObjId = void*;

template <typename Ret = ObjId, typename... Args>
Ret call(ObjId target, const char* selector, Args... args)
{
    using Fn = Ret (*)(ObjId, SEL, Args...);
    return reinterpret_cast<Fn>(&objc_msgSend)(target, sel_registerName(selector), args...);
}

struct CgSize
{
    double width = 0, height = 0;
};

struct CgRect
{
    double x = 0, y = 0, w = 0, h = 0;
};

CgRect callRect(ObjId target, const char* selector)
{
#if defined(__x86_64__)
    // 32-byte struct returns go through the hidden-pointer entry on x86_64.
    CgRect out;
    using Fn = void (*)(CgRect*, ObjId, SEL);
    reinterpret_cast<Fn>(&objc_msgSend_stret)(&out, target, sel_registerName(selector));
    return out;
#else
    using Fn = CgRect (*)(ObjId, SEL);
    return reinterpret_cast<Fn>(&objc_msgSend)(target, sel_registerName(selector));
#endif
}

ObjId poolPush()
{
    return call(call(reinterpret_cast<ObjId>(objc_getClass("NSAutoreleasePool")),
                     "alloc"), "init");
}

void poolDrain(ObjId pool)
{
    call<void>(pool, "drain");
}

// -- diagnostic-log probe ------------------------------------------------------------
//
// The editor layer traces its host interaction to $TMPDIR/DSParkWebView.log
// when DSPARK_WEBVIEW_LOG is set. The "page metric" lines are emitted only
// after the page's JS bridge completed the ready handshake with the native
// side — the strongest headless proof that the whole WKWebView pipeline
// (engine, page load, script message channel, eval downlink) works.

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

OSType fourCC(const char* s)
{
    return (static_cast<OSType>(static_cast<unsigned char>(s[0])) << 24)
         | (static_cast<OSType>(static_cast<unsigned char>(s[1])) << 16)
         | (static_cast<OSType>(static_cast<unsigned char>(s[2])) << 8)
         |  static_cast<OSType>(static_cast<unsigned char>(s[3]));
}

ObjId makeView(ObjId factory, AudioUnit au)
{
    using Fn = ObjId (*)(ObjId, SEL, AudioUnit, CgSize);
    return reinterpret_cast<Fn>(&objc_msgSend)(
        factory, sel_registerName("uiViewForAudioUnit:withSize:"), au, CgSize {});
}

} // namespace

int main(int argc, char** argv)
{
    bool expectEditor = false;
    int expectedW = 0, expectedH = 0;
    if (argc >= 5 && std::strlen(argv[1]) == 4 && std::strlen(argv[2]) == 4
        && std::strlen(argv[3]) == 4)
    {
        if (std::strcmp(argv[4], "--expect-editor") == 0 && argc >= 6
            && std::sscanf(argv[5], "%dx%d", &expectedW, &expectedH) == 2)
            expectEditor = true;
        else if (std::strcmp(argv[4], "--expect-no-editor") != 0)
            argc = 0;
    }
    else
        argc = 0;
    if (argc == 0)
    {
        std::fprintf(stderr,
            "usage: au_editor_smoke <type4> <subtype4> <manu4> "
            "--expect-editor <W>x<H> | --expect-no-editor\n");
        return 2;
    }

    // Enable the editor's diagnostic trace BEFORE the component loads.
    setenv("DSPARK_WEBVIEW_LOG", "1", 1);
    std::printf("au_editor_smoke: %s %s %s (%s)\n", argv[1], argv[2], argv[3],
                expectEditor ? "expecting editor" : "expecting NO editor");

    AudioComponentDescription desc {};
    desc.componentType = fourCC(argv[1]);
    desc.componentSubType = fourCC(argv[2]);
    desc.componentManufacturer = fourCC(argv[3]);
    AudioComponent component = AudioComponentFindNext(nullptr, &desc);
    expect(component != nullptr, "component is registered");
    if (component == nullptr) return 1;

    AudioUnit au = nullptr;
    expect(AudioComponentInstanceNew(component, &au) == noErr && au != nullptr,
           "instance created");
    if (au == nullptr) return 1;

    UInt32 infoSize = 0;
    Boolean writable = false;
    const OSStatus infoStatus = AudioUnitGetPropertyInfo(
        au, kAudioUnitProperty_CocoaUI, kAudioUnitScope_Global, 0,
        &infoSize, &writable);

    if (!expectEditor)
    {
        expect(infoStatus != noErr, "CocoaUI property absent (host generic UI)");
        expect(AudioComponentInstanceDispose(au) == noErr, "instance disposed");
        std::printf("au_editor_smoke: %s\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
        return g_failures == 0 ? 0 : 1;
    }

    expect(infoStatus == noErr, "CocoaUI property announced");
    expect(infoSize >= sizeof(AudioUnitCocoaViewInfo), "CocoaUI property size");
    expect(writable == false, "CocoaUI property is read-only");

    AudioUnitCocoaViewInfo viewInfo {};
    UInt32 ioSize = sizeof(viewInfo);
    expect(AudioUnitGetProperty(au, kAudioUnitProperty_CocoaUI,
                                kAudioUnitScope_Global, 0, &viewInfo, &ioSize) == noErr,
           "CocoaUI property read");

    char bundlePath[1024] {};
    expect(viewInfo.mCocoaAUViewBundleLocation != nullptr
               && CFURLGetFileSystemRepresentation(
                      viewInfo.mCocoaAUViewBundleLocation, true,
                      reinterpret_cast<UInt8*>(bundlePath), sizeof(bundlePath)),
           "view bundle URL is a filesystem path");
    struct stat st {};
    expect(bundlePath[0] != '\0' && stat(bundlePath, &st) == 0,
           "view bundle exists on disk");

    char className[256] {};
    expect(viewInfo.mCocoaAUViewClass[0] != nullptr
               && CFStringGetCString(viewInfo.mCocoaAUViewClass[0], className,
                                     sizeof(className), kCFStringEncodingUTF8)
               && className[0] != '\0',
           "view factory class name");
    std::printf("    bundle: %s\n    class:  %s\n", bundlePath, className);

    NSApplicationLoad();

    // Host contract: after reading the property the class must resolve
    // (NSClassFromString) — the plugin registers it when serving CocoaUI.
    Class factoryClass = objc_getClass(className);
    expect(factoryClass != nullptr, "factory class resolves");
    if (factoryClass == nullptr) return 1;

    ObjId factory = call(call(reinterpret_cast<ObjId>(factoryClass), "alloc"), "init");
    expect(factory != nullptr, "factory instantiated");
    expect(call<unsigned int>(factory, "interfaceVersion") == 0, "interfaceVersion == 0");

    char path[512];
    logPath(path, sizeof(path));
    unlink(path);   // fresh trace for this session

    // --- first view: full creation ------------------------------------------------
    ObjId pool = poolPush();
    ObjId view = makeView(factory, au);
    expect(view != nullptr, "uiViewForAudioUnit returns a view");
    if (view == nullptr) return 1;
    call<void>(view, "retain");   // play the host: keep it beyond the pool
    poolDrain(pool);

    expect(call<signed char>(view, "isKindOfClass:",
                             reinterpret_cast<ObjId>(objc_getClass("NSView"))) != 0,
           "view is an NSView");
    const CgRect frame = callRect(view, "frame");
    std::printf("    frame:  %.0fx%.0f\n", frame.w, frame.h);
    expect(static_cast<int>(frame.w) == expectedW
               && static_cast<int>(frame.h) == expectedH,
           "frame matches the declared editorSize");

    ObjId subviews = call(view, "subviews");
    const unsigned long subviewCount = call<unsigned long>(subviews, "count");
    bool webViewFound = false;
    for (unsigned long i = 0; i < subviewCount; ++i)
    {
        ObjId sub = call(subviews, "objectAtIndex:", i);
        if (std::strstr(object_getClassName(reinterpret_cast<id>(sub)), "WKWebView"))
            webViewFound = true;
    }
    expect(webViewFound, "WKWebView embedded in the container");
    expect(logContains("au createEditorView"), "editor trace written");

    // Page load + bridge handshake happen asynchronously on this runloop;
    // first-launch web-process spawn can be slow on cold CI runners, so poll
    // up to 15 s but return the moment the handshake lands.
    bool handshake = false;
    for (int i = 0; i < 30 && !handshake; ++i)
    {
        CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, false);
        handshake = logContains("page metric");
    }
    expect(handshake, "JS bridge handshake completed");

    // --- teardown order A: host releases the view, editor must re-create ----------
    pool = poolPush();
    call<void>(view, "release");
    poolDrain(pool);

    pool = poolPush();
    ObjId view2 = makeView(factory, au);
    expect(view2 != nullptr, "editor re-creates after view release");
    if (view2 != nullptr) call<void>(view2, "retain");
    poolDrain(pool);
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, false);

    // --- teardown order B: AudioUnit dies first, view follows ----------------------
    expect(AudioComponentInstanceDispose(au) == noErr,
           "AU disposes with a live editor view");
    if (view2 != nullptr)
    {
        pool = poolPush();
        call<void>(view2, "release");
        poolDrain(pool);
    }
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.25, false);

    CFRelease(viewInfo.mCocoaAUViewBundleLocation);
    CFRelease(viewInfo.mCocoaAUViewClass[0]);
    call<void>(factory, "release");

    std::printf("au_editor_smoke: %s\n", g_failures == 0 ? "ALL PASS" : "FAILURES");
    return g_failures == 0 ? 0 : 1;
}

#endif // __APPLE__
