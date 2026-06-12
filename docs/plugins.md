# Building audio plugins with DSPark

DSPark ships a native plugin layer: you write **one ordinary struct** — no
base class, no SDK download — and format macros turn it into a
loadable **VST3**, **CLAP** and **Audio Unit v2** plugin (`auval`-validated
in CI; AU is what Logic Pro and GarageBand load). This page is the complete
contract reference: everything the wrappers will detect and call, what each
piece maps to in every format, and when to use it.

For a copy-paste starting point with every optional method present and
commented, see [`examples/plugin_template/`](../examples/plugin_template/plugin_template.cpp).
For a real effect, see [`examples/plugin_saturator/`](../examples/plugin_saturator/saturator.cpp).

---

## The 60-second version

```cpp
#include "plugin/vst3/DSParkVst3.h"
#include "plugin/clap/DSParkClap.h"

struct MyPlugin
{
    static constexpr auto descriptor = dspark::plugin::Descriptor { ... };
    static constexpr auto parameters = dspark::plugin::params( ... );

    void prepare(const dspark::AudioSpec& spec);
    void setParameter(int index, float plainValue) noexcept;
    void processBlock(dspark::AudioBufferView<float> io) noexcept;
};

DSPARK_VST3_PLUGIN(MyPlugin)
DSPARK_CLAP_PLUGIN(MyPlugin)
DSPARK_AU_PLUGIN(MyPlugin, "Subt", "Manu")   // unique 4-char codes; macOS bundle
```

```
cl      /std:c++20 /O2 /LD /EHsc /I . myplugin.cpp /Fe:MyPlugin.vst3      (Windows)
g++     -std=c++20 -O2 -fPIC -shared -I . myplugin.cpp -o MyPlugin.vst3   (Linux)
clang++ -std=c++20 -O2 -fPIC -shared -I . myplugin.cpp -o MyPlugin        (macOS;
        add -framework AudioToolbox -framework CoreFoundation for the AU
        macro, and -lobjc when the WebView editor layer is included)
```

Or skip the per-platform flags entirely — the CMake helper builds the right
bundle layout, links the right system libraries and embeds the editor files
on every platform from one line (**the recommended cross-platform path**):

```cmake
include(plugin/cmake/DSParkPlugin.cmake)
dspark_add_plugin(MyPlugin SOURCES myplugin.cpp
                  FORMATS VST3 CLAP AU
                  AU_SUBTYPE Subt AU_MANUFACTURER Manu)
```

One compiled binary carries **every entry point the source declares**: ship
the same file as `.vst3` and as `.clap`; on macOS the AU `.component`
bundle wraps a copy of that same binary. Install locations:

| Format | Windows | macOS | Linux |
|---|---|---|---|
| VST3 | `C:\Program Files\Common Files\VST3` | `/Library/Audio/Plug-Ins/VST3` | `~/.vst3` |
| CLAP | `C:\Program Files\Common Files\CLAP` (or `%LOCALAPPDATA%\Programs\Common\CLAP`) | `/Library/Audio/Plug-Ins/CLAP` | `~/.clap` |
| AU | — | `/Library/Audio/Plug-Ins/Components` (`.component` bundle: binary built with `-bundle -framework AudioToolbox`, plus an `Info.plist` whose `AudioComponents` entry carries your 4-char codes — see `examples/plugin_saturator/au/`) | — |

AU presets, latency, tail, bypass and parameters map through the same layer
(plain values, same hashed ids, same state container — presets stay portable
across all three formats). Validate with `auval -v aufx Subt Manu`; this
repository's macOS CI does exactly that on every commit.

---

## Two writing styles

**Guided (recommended to start):** inherit `dspark::plugin::PluginBase<T>`.
The base defines the entire optional contract with safe defaults, so every
overridable method is one Go-to-Definition away and your IDE autocompletes
the menu — define the same signature in your class to replace a default
(plain shadowing, resolved at compile time, zero dispatch cost, no
virtuals). Full discoverability, with no machinery behind it.
`examples/plugin_template/` uses it.

```cpp
struct MyPlugin : dspark::plugin::PluginBase<MyPlugin>
{
    // descriptor + parameters + the required five...
    int getLatency() const noexcept { return limiter_.getLatency(); }  // override
    // everything you DON'T define falls back to the base default
};
```

**Free-standing:** skip the base entirely — the wrappers detect your
capabilities structurally (C++20 concepts), so a plain struct with the same
members behaves identically. `examples/plugin_saturator/` uses this style.

## The contract, in full

The wrappers detect your capabilities with C++20 concepts: implement a
method and it is wired into every format automatically; omit it (or keep
the `PluginBase` default) and the safe fallback applies. **Nothing is
virtual; nothing else is required.**

### Required

| Member | Called from | What it must do |
|---|---|---|
| `static constexpr Descriptor descriptor` | scan time | Identity. `productId` derives the VST3 class UID and IS the CLAP id — **never change it after a release** (it orphans saved sessions). `name`, `vendor`, `version`, `url`, `email` feed the hosts' plugin browsers. `category` selects effect or **instrument** (see "Instruments & MIDI"). |
| `static constexpr auto parameters` | scan time | The automatable parameter table, built with `params(param(...), toggle(...))`. The **text ids are the stable identity** of each parameter (state + automation): you may reorder/insert parameters freely between versions, but never rename an id. |
| `void prepare(const AudioSpec&)` | main thread, before audio | Allocate and configure for `sampleRate` / `maxBlockSize` / `numChannels` — the channel count is whatever the host negotiated (1 or 2, see `channels` below). Maps to VST3 `setActive(true)` (after `setupProcessing`) and CLAP `activate`. May allocate. |
| `void setParameter(int index, float plain) noexcept` | **any thread** | Receive a plain-range value for `parameters[index]`. Forward to your DSPark setters — they are atomic and smoothed by contract, which is what makes this callable from UI and audio threads alike. Never allocate or lock. |
| `void processBlock(AudioBufferView<float>) noexcept` | audio thread | In-place processing at the negotiated width (`io.getNumChannels()`), exactly like every DSPark effect. Never allocate, lock or block. (For a sidechain, implement the two-buffer form INSTEAD — see Optional below.) |

### Optional — detected automatically

| Member | Maps to | Implement it when... |
|---|---|---|
| `int getLatency() const` | VST3 `getLatencySamples`, CLAP `clap.latency`, AU `Latency` | your chain introduces delay: lookahead limiters, linear-phase EQ, oversampling, FFT processors. Sum your DSPark effects' `getLatency()`/`getLatencySamples()`. The host shifts everything to compensate — report it accurately or your users get phasing. Read after `prepare()` **and re-read after parameter changes**: when the value moves at runtime, the wrapper notifies the host on its own (`restartComponent(kLatencyChanged)` / `clap_host_latency` / the AU property listeners). |
| `double getTailSeconds() const` | VST3 `getTailSamples`, CLAP `clap.tail`, AU `TailTime` | sound continues after input stops: reverbs, delays. Hosts keep processing your plugin that long after audio ends instead of cutting the tail. |
| `void reset() noexcept` | CLAP `reset`, AU `Reset` | you keep history that should clear on transport jumps (delay lines, envelopes). VST3 has no direct equivalent — hosts re-activate instead, which re-runs `prepare()`. |
| `std::vector<uint8_t> getState() const` + `bool setState(const uint8_t*, size_t)` | VST3 `IComponent::get/setState`, CLAP `clap.state`, AU `ClassInfo` (extra section) | you have state **beyond the parameters**: learned noise profiles, loaded IRs, editor layout. The wrapper *always* saves/restores your parameter values by itself — most plugins need neither method. DSPark's `StateBlob` is the natural serializer here. |
| `void processBlock(AudioBufferView<float> io, AudioBufferView<float> sidechain) noexcept` — *instead of* the single-buffer form | a second input named "Sidechain": VST3 **aux bus**, CLAP **non-main port**, AU **input element** | the detector should follow a host-routed key signal: duckers, externally-keyed gates and de-essers. The shape matches DSPark's own dynamics, so `comp_.processBlock(io, sidechain)` forwards 1:1. The key view always mirrors the main width (mono main, mono key) and the wrapper hands you frame-aligned **silence when nothing is routed** — never branch on availability, and treat the sidechain as read-only. Reference plugin: `examples/plugin_ducker/`. |
| `void setTransport(const TransportInfo&) noexcept` | VST3 `ProcessContext`, CLAP transport event, AU host callbacks | anything in your plugin follows the song: tempo-synced delays, LFOs, gates, arpeggiators. Delivered on the audio thread before each `processBlock` whenever the host supplied timeline data — check the `*Valid` flags. `TransportInfo::samplesPerBeat(sampleRate)` is the conversion you usually want. |
| `void handleMidiEvent(const MidiEvent&) noexcept` | VST3 event bus + `IMidiMapping` proxies, CLAP note port (CLAP + raw MIDI dialects), AU MusicDevice selectors | notes drive the plugin: instruments, vocoders, MIDI-gated effects. One normalised struct covers note on/off, pitch bend, CC, channel/poly pressure. Events arrive time-ordered right before the block that contains them, with `sampleOffset` for sample-accurate voice starts. Required for `Category::Instrument`. |
| `void setOfflineRendering(bool) noexcept` | VST3 `kOffline` process mode, CLAP `clap.render`, AU `OfflineRender` | quality should rise when there is no realtime pressure: switch to higher oversampling or longer lookahead during bounces. Called outside the audio thread, before processing (re)starts. |
| `static constexpr auto channels` (`ChannelSupport`) | VST3 `setBusArrangements`, CLAP `audio-ports-config`, AU `SupportedNumChannels` | **the default is already mono+stereo** — declare it only to restrict: `StereoOnly` for inherently stereo DSP (M/S wideners), `MonoOnly` for metering utilities. All buses of an instance run the same width. |
| `static constexpr auto factoryPresets` | VST3 program list, CLAP preset-load + preset-discovery, AU factory presets | you ship starting points. Built with `presets(preset("Name", v0, v1, ...))` — one PLAIN value per parameter, in table order, checked at compile time. The host's own preset browser offers them in every format. |
| `static constexpr bool sampleAccurateAutomation = false` | — | you want to OPT OUT of sub-block automation: by default the wrappers split processing at automation points (32-frame grain) so fast curves land where the host drew them. Opt out only when a high fixed cost per `processBlock` call matters more. |
| `static constexpr bool hasEditor` | VST3 `createView`, CLAP `clap.gui`, AU `CocoaUI` | a custom GUI written in HTML/CSS/JS: pair it with `editorHtml()` (+ optional `editorSize`, `editorResize`, `editorDebug`) and include the WebView editor layer first. While false/absent, hosts show their generic parameter UI — fully usable. See "Custom UIs" below. |

### What the wrappers handle so you don't have to

- **Bypass**: a host-integrated bypass parameter (VST3 `kIsBypass`, CLAP
  `CLAP_PARAM_IS_BYPASS`, AU `BypassEffect`) with a click-free crossfade
  against the dry input (toward silence for an instrument).
- **Sample-accurate automation**: every timestamped event the host sends —
  parameter points, bypass, MIDI — lands in one time-ordered stream and
  processing splits at 32-frame quantum boundaries, so fast automation
  curves land where they were drawn instead of stepping once per block.
  DSPark setters still smooth on top. (Opt out per plugin, see above.)
- **Latency changes**: after parameter motion the wrapper re-reads
  `getLatency()` and notifies the host through the native channel
  (`restartComponent(kLatencyChanged)`, `clap_host_latency` on the main
  thread, the AU `Latency` property listeners) so projects re-compensate.
- **State container**: versioned, tolerant (unknown parameters are skipped,
  missing ones keep defaults) and **identical across formats** — a preset
  saved by the VST3 build loads in the CLAP and AU builds byte-for-byte.
- **Buses**: mono/stereo negotiation per the declared `ChannelSupport`
  (default both), instrument layouts without audio inputs, the sidechain
  following the main width, and the right answers to every arrangement
  negotiation. Value formatting/parsing for host displays. Factory metadata.
- **Denormals**: `process()`/`render()` run under DSPark's `DenormalGuard`
  (FTZ/DAZ), so your own DSP is protected even in hosts that don't set it.
- **Entry points** per platform and format, from the macros.

### Threading model

| Call | Thread |
|---|---|
| `prepare` | main (host setup) — allocation allowed |
| `processBlock` | audio — real-time rules apply |
| `setParameter` | **both** (host automation arrives on audio; generic UI edits arrive on main) — hence the atomic-setter contract |
| `setTransport` / `handleMidiEvent` | audio — real-time rules apply |
| `setOfflineRendering` | main (before processing restarts) |
| `getState`/`setState` | main |

---

## Instruments & MIDI

Set `category = Category::Instrument` in the descriptor and the plugin
becomes a generator: **no audio input in any format** (VST3 instrument
class, CLAP `"instrument"` feature, AU `aumu` music device — the AU bundle's
Info.plist must declare `aumu` too), and the wrapper hands `processBlock` a
**cleared buffer so voices ADD into it** without reading. An instrument
must implement `handleMidiEvent` — it has nothing else to react to.

```cpp
void handleMidiEvent(const dspark::plugin::MidiEvent& ev) noexcept
{
    using Type = dspark::plugin::MidiEvent::Type;
    switch (ev.type)
    {
    case Type::NoteOn:    /* ev.note, ev.value = velocity 0..1,
                             ev.sampleOffset = frames into the next block */
    case Type::NoteOff:   /* release the matching voice */
    case Type::PitchBend: /* ev.value = -1..+1 */
    default: break;
    }
}
```

The wrapper translates every format's native scheme into that one struct:
VST3 delivers notes as events and pitch bend / mod wheel / sustain /
channel pressure through its `IMidiMapping` controller scheme (handled
internally with hidden proxy parameters — you never see them); CLAP
delivers note events plus raw MIDI; AU delivers raw MIDI bytes through the
MusicDevice selectors. Events always arrive **time-ordered, right before
the `processBlock` call that contains them**; `sampleOffset` says how many
frames into that block the event belongs, so a synth can start its voice
sample-accurately (or ignore the offset and be at most one automation
quantum early).

An effect (`Category::Fx`) may also implement `handleMidiEvent` — it keeps
its audio buses and gains a MIDI input on top (a vocoder, a MIDI-gated
delay). On macOS that maps to the `aumf` music-effect type: declare `aumf`
in the AU Info.plist for MIDI effects, `aumu` for instruments.

Reference instrument: `examples/plugin_synth/` — eight voices of
`Oscillator` + `ADSREnvelope`, voice stealing, pitch bend, factory presets
and sample-accurate note starts, validated by `auval -v aumu` and both
smoke hosts in CI.

## Host transport

Implement `setTransport(const TransportInfo&)` and the wrapper feeds you
the host timeline once per block, before `processBlock`, translated from
each format's native source (VST3 `ProcessContext` — with the requirement
flags declared for you —, the CLAP transport event, the AU host callbacks):

```cpp
void setTransport(const dspark::plugin::TransportInfo& t) noexcept
{
    if (t.tempoValid)
        delaySamples_ = static_cast<int>(t.samplesPerBeat(sampleRate_) * 0.5);
    playing_ = t.playing;   // e.g. freeze an LFO while stopped
}
```

`TransportInfo` carries tempo, musical position (quarter notes), bar start,
time signature, loop points and the playing/recording/looping states, each
guarded by a `*Valid` flag — hosts differ in what they provide, so default
sensibly when a flag is false. No call means "nothing changed since the
last block".

---

## Custom UIs — the WebView editor

Plugins without an editor use the host's generic parameter UI (fully
functional: sliders, automation, state). To ship a custom GUI, write it in
**plain HTML/CSS/JS** and let the editor layer embed it in the host window —
one UI codebase for every format, no C++ GUI framework, no resource files:

```cpp
#include "plugin/webview/DSParkWebViewEditor.h"   // FIRST: before format headers
#include "plugin/vst3/DSParkVst3.h"
#include "plugin/clap/DSParkClap.h"
#include "plugin/au/DSParkAu.h"                   // self-disables off macOS

struct MyPlugin
{
    // ... descriptor / parameters / prepare / setParameter / processBlock ...
    static constexpr bool hasEditor = true;
    static constexpr dspark::plugin::EditorSize editorSize { 560, 330 };  // logical px
    static constexpr auto editorResize =              // optional (default Fixed)
        dspark::plugin::EditorResize::KeepAspect;     // Fixed | Free | KeepAspect
    // static constexpr bool editorDebug = true;      // optional: browser DevTools
    static const char* editorHtml() { return R"html(<!doctype html>...)html"; }
};
```

The wrappers embed the platform web engine inside the window each host hands
them — VST3 `IPlugView` (with `IPlugViewContentScaleSupport`, so HiDPI hosts
get a crisp, correctly-sized page), CLAP `clap.gui`, and AU through the
Cocoa view factory announced by `kAudioUnitProperty_CocoaUI` (registered at
runtime from this very binary — still no Objective-C sources to write):

| Platform | Engine | Status |
|---|---|---|
| Windows | WebView2 (Edge runtime, ships with Win10/11; vendored MIT `webview` library + BSD-3 SDK header, nothing to install) | exercised in a real window on every change (`tools/vst3_editor_host`) |
| macOS | WKWebView (system WebKit, loaded at runtime through the Objective-C runtime — no extra SDK; link `-lobjc`) | VST3, CLAP **and AU**; the AU editor contract (factory, real WKWebView, bridge handshake, teardown) runs in CI via `tools/au_editor_smoke` |
| Linux | WebKitGTK (GTK3) — resolved entirely through `dlopen` at runtime, embedded with GtkPlug/XEmbed in the host's X11 window; GTK is pumped from the host run loop (VST3 `IRunLoop` / CLAP `timer-support`) | full attach + bridge handshake runs in CI under xvfb via `tools/x11_editor_smoke`; systems without WebKitGTK (or hosts without a run loop) keep the generic UI |

A complete working example — knobs with drag/wheel/double-click, a discrete
selector, gestures, automation feedback — lives in
[`examples/plugin_webview_editor/`](../examples/plugin_webview_editor/webview_saturator.cpp).

### Per-platform notes: what you need, what your users need

The editor is an **optional capability at runtime** by design: wherever the
web engine cannot run, the editor simply isn't offered and hosts fall back
to their generic parameter UI — the DSP keeps working untouched. Your page
never needs platform branches; these notes are about toolchains and what to
tell end users.

| | Building your plugin | What end users need | Worth knowing |
|---|---|---|---|
| **Windows** | MSVC: nothing extra (system libs auto-link via `#pragma comment`). MinGW/clang: link `advapi32 comctl32 ole32 shell32 shlwapi user32 version` — `dspark_add_plugin` already does. | The WebView2 (Edge) runtime — preinstalled on Windows 10/11. | On the rare locked-down machine without the runtime, editor creation fails cleanly → generic UI. `editorDebug = true` gives you the Chromium DevTools. |
| **macOS** | Link `-lobjc`; once the AU macro is in the TU, also `-framework AudioToolbox -framework CoreFoundation`. No Apple GUI headers or Objective-C sources involved. | Nothing — WKWebView ships with macOS. | AU hosts size the window from the view's initial frame (`editorSize`); the `Free`/`KeepAspect` drag negotiation is a VST3/CLAP concept, so on AU the page's proportional fit is what absorbs host-driven resizes. |
| **Linux** | Nothing — WebKitGTK is resolved with `dlopen` at runtime: no headers, no pkg-config, no link-time dependency. | `libwebkit2gtk-4.1` (or `-4.0`) installed, and an X11 or XWayland session. Debian/Ubuntu: `apt install libwebkit2gtk-4.1-0`. | GTK must breathe from the host's run loop; the wrapper handles it (VST3 `IRunLoop` timer / CLAP `timer-support`). A host offering neither — or a system without WebKitGTK — gets the generic UI instead of a frozen page. |

### The `dspark` JS bridge

Injected before your page's own scripts. Parameters use the **same stable
text ids** as state and automation, in **plain** values:

| Call | Direction | Notes |
|---|---|---|
| `dspark.onReady(cb)` | native → UI | fires once with the parameter table + current values: `[{id, name, min, max, def, unit, steps, value}, ...]` (replayed if already received) |
| `dspark.setParam(id, v)` | UI → DSP + host | applied to the DSP immediately and forwarded to the host for automation recording |
| `dspark.onParam(id, cb)` | DSP → UI | fires on host automation, preset/state restore and your own edits; cached value replays on registration |
| `dspark.beginEdit(id)` / `dspark.endEdit(id)` | UI → host | automation gesture around a drag (host undo / touch automation); a bare `setParam` outside a gesture gets a one-shot gesture automatically |
| `dspark.getParam(id)` / `dspark.params` | cached | last seen value / the metadata table |

**How DSP → UI sync works:** the page polls the wrapper's normalized shadow
atomics at ~30 Hz (started automatically after `onReady`) and dispatches
only actual changes. No native timers, no locks, nothing on the audio
thread; when the host hides the page, the browser throttles the timer by
itself. All editor callbacks run on the host's main/UI thread and funnel
into your `setParameter` — atomic and smoothed by contract, so a drag is
click-free by construction.

### Sizing, resize and HiDPI

`editorSize` is declared in **logical pixels**; the wrapper converts to the
physical pixels VST3/CLAP negotiate in using the host-reported content
scale. The page itself never zooms — the web engine already maps CSS pixels
through the window's DPI — so your CSS works in logical units everywhere.
On attach the editor fits the box the host actually built (hosts differ in
call order during HiDPI negotiation), and `EditorResize` picks the policy:

| `editorResize` | Host behaviour |
|---|---|
| `Fixed` (default) | the window is exactly `editorSize`; no drag-resize |
| `Free` | drag-resizable between 0.5x and 3x the declared size |
| `KeepAspect` | drag-resizable, locked to the declared width:height ratio |

Make the page fluid (flexbox/grid, percentage sizes) and any of the three
modes looks right; `examples/plugin_webview_editor/` uses `KeepAspect`.

### Developing a UI

- **Separate web files, embedded at build time** (the production workflow):
  keep the interface as ordinary `editor.html` + `editor.css` + `editor.js`
  and let CMake inline and embed them —
  ```cmake
  dspark_add_plugin(MyPlugin SOURCES myplugin.cpp
                    FORMATS VST3 CLAP AU
                    AU_SUBTYPE Subt AU_MANUFACTURER Manu
                    EDITOR_HTML ui/editor.html)
  ```
  generates `MyPlugin_editor_html.h` (rebuilt whenever any referenced file
  changes); the plugin returns the generated `kDsparkEditorHtml` from
  `editorHtml()`. Complete example: `examples/plugin_webview_files/`.
- **Edit without recompiling**: declare
  `static const char* editorDevFile() { return "C:/dev/myplugin/editor.html"; }`
  and the editor loads that file from disk on every open (falling back to
  the embedded `editorHtml()` when missing). Edit, save, reopen the editor —
  no C++ rebuild. Remove it for release builds.
- `tools/vst3_editor_host.cpp` opens your editor in a bare native window —
  the 10-second check without a DAW. It reports the monitor content scale
  like a DPI-aware host, logs begin/perform/endEdit calls live and
  self-tests the resize chain (grow, shrink-below-minimum, widget fit).
- Set `editorDebug = true` during development to get the browser DevTools.
- Set the environment variable `DSPARK_WEBVIEW_LOG=1` to trace every
  host/editor interaction (attach, size negotiation, bridge handshake) to
  `%TEMP%\DSParkWebView.log` on Windows or `$TMPDIR/DSParkWebView.log` on
  macOS/Linux — invaluable when a host misbehaves.
- The page is ordinary web content: iterate it in a regular browser with a
  stubbed `window.__dsparkPost`, then paste it into `editorHtml()`.

---

## Verifying and shipping

1. **Smoke hosts** (in this repo, run in CI on every platform):
   `tools/vst3_smoke_host.cpp` and `tools/clap_smoke_host.cpp` drive your
   module through the full lifecycle like a DAW — including the editor
   contract (view creation, platform support, sizing, refcounts) — and exit
   non-zero on any misbehaviour. `tools/vst3_editor_host.cpp` additionally
   opens the editor in a real window (Windows), `tools/au_editor_smoke.cpp`
   drives the AU Cocoa view contract end to end (macOS): factory class, a
   real WKWebView, the JS bridge handshake and both teardown orders — and
   `tools/x11_editor_smoke.cpp` does the same for Linux: a real X11 window,
   an IRunLoop-driven WebKitGTK attach and the bridge handshake under xvfb.
2. **Official validators**: Tracktion's
   [`pluginval`](https://github.com/Tracktion/pluginval) (strictness 8) and
   [`clap-validator`](https://github.com/free-audio/clap-validator) gate this
   repository's CI on Windows, Linux and macOS — every example plugin passes
   both on every commit. Steinberg's `validator` (ships with the VST3 SDK)
   remains a useful extra check before release.
3. **Real hosts**: Reaper loads both formats; Cubase is the strictest VST3
   host. On macOS, distribution requires code signing + notarization
   (a developer-account step, not a framework one).

## FAQ

**How do I add a parameter in v2 of my plugin without breaking sessions?**
Append it (or insert it — order does not matter) with a NEW text id. Old
sessions load: the new parameter takes its default, every old one restores
by id.

**Can I rename a parameter's display name?** Yes — `name` is cosmetic.
Never change `id`.

**Sidechain?** Yes: implement the two-buffer `processBlock(io, sidechain)`
and every format grows a host-routable "Sidechain" input (the table above;
`examples/plugin_ducker/` is the reference). The smoke hosts verify the bus
layout — and, with `--expect-sidechain`, that a hot key actually ducks.

**Mono?** Default. Every plugin negotiates 1→1 and 2→2 unless it declares
`channels = ChannelSupport::StereoOnly`; `prepare` receives the width and
`io.getNumChannels()` reflects it. The sidechain follows the main width.

**MIDI? Instruments?** Yes: `handleMidiEvent` adds a note input to every
format, and `Category::Instrument` makes the plugin a generator (no audio
input; AU type `aumu`). See "Instruments & MIDI" above;
`examples/plugin_synth/` is the reference, and the smoke hosts prove the
note path functionally (`--expect-instrument`: pitch and release decay
measured from the output).

**Does automation step once per block?** No — it splits the block at the
host's automation points (32-frame grain) by default, and DSPark setters
smooth on top. Opt out with `sampleAccurateAutomation = false` if your
plugin has a high fixed cost per `processBlock` call.

**Tempo-synced effects?** Implement `setTransport` and read
`TransportInfo::samplesPerBeat(sampleRate)` — delivered per block from the
VST3 ProcessContext / CLAP transport / AU host callbacks.

**Where do the parameter values live?** Wherever your DSPark members keep
them. The wrapper keeps normalized shadows only for host queries and state;
your `setParameter` is the single source of truth for the DSP.
