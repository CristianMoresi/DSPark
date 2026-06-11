# Building audio plugins with DSPark

DSPark ships a native plugin layer: you write **one ordinary struct** — no
base class, no SDK download, no JUCE — and format macros turn it into a
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
cl  /std:c++20 /O2 /LD /EHsc /I . myplugin.cpp /Fe:MyPlugin.vst3     (Windows)
g++ -std=c++20 -O2 -fPIC -shared -I . myplugin.cpp -o MyPlugin.vst3 (Linux)
```

One compiled binary carries **both** entry points: ship it as `.vst3` and as
`.clap` (same file, two names). Install locations:

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
virtuals). This is the JUCE-style discoverability without the JUCE-style
machinery. `examples/plugin_template/` uses it.

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
| `static constexpr Descriptor descriptor` | scan time | Identity. `productId` derives the VST3 class UID and IS the CLAP id — **never change it after a release** (it orphans saved sessions). `name`, `vendor`, `version`, `url`, `email` feed the hosts' plugin browsers. |
| `static constexpr auto parameters` | scan time | The automatable parameter table, built with `params(param(...), toggle(...))`. The **text ids are the stable identity** of each parameter (state + automation): you may reorder/insert parameters freely between versions, but never rename an id. |
| `void prepare(const AudioSpec&)` | main thread, before audio | Allocate and configure for `sampleRate` / `maxBlockSize` / 2 channels. Maps to VST3 `setActive(true)` (after `setupProcessing`) and CLAP `activate`. May allocate. |
| `void setParameter(int index, float plain) noexcept` | **any thread** | Receive a plain-range value for `parameters[index]`. Forward to your DSPark setters — they are atomic and smoothed by contract, which is what makes this callable from UI and audio threads alike. Never allocate or lock. |
| `void processBlock(AudioBufferView<float>) noexcept` | audio thread | In-place stereo processing, exactly like every DSPark effect. Never allocate, lock or block. |

### Optional — detected automatically

| Member | Maps to | Implement it when... |
|---|---|---|
| `int getLatency() const` | VST3 `getLatencySamples`, CLAP `clap.latency` | your chain introduces delay: lookahead limiters, linear-phase EQ, oversampling, FFT processors. Sum your DSPark effects' `getLatency()`/`getLatencySamples()`. The host shifts everything to compensate — report it accurately or your users get phasing. Read after `prepare()`. |
| `double getTailSeconds() const` | VST3 `getTailSamples`, CLAP `clap.tail` | sound continues after input stops: reverbs, delays. Hosts keep processing your plugin that long after audio ends instead of cutting the tail. |
| `void reset() noexcept` | CLAP `reset` | you keep history that should clear on transport jumps (delay lines, envelopes). VST3 has no direct equivalent — hosts re-activate instead, which re-runs `prepare()`. |
| `std::vector<uint8_t> getState() const` + `bool setState(const uint8_t*, size_t)` | VST3 `IComponent::get/setState`, CLAP `clap.state` (extra section) | you have state **beyond the parameters**: learned noise profiles, loaded IRs, editor layout. The wrapper *always* saves/restores your parameter values by itself — most plugins need neither method. DSPark's `StateBlob` is the natural serializer here. |
| `static constexpr bool hasEditor` | reserved | a custom GUI. v1 reports "no editor" and every host shows its generic parameter UI — fully usable. See "Custom UIs" below. |

### What the wrappers handle so you don't have to

- **Bypass**: a host-integrated bypass parameter (VST3 `kIsBypass`, CLAP
  `CLAP_PARAM_IS_BYPASS`) with a click-free crossfade against the dry input.
- **Automation**: parameter event queues are drained and funnelled into
  `setParameter` (block-rate in v1 — inaudible because DSPark setters smooth
  internally; sample-accurate slicing is a planned refinement).
- **State container**: versioned, tolerant (unknown parameters are skipped,
  missing ones keep defaults) and **identical across formats** — a preset
  saved by the VST3 build loads in the CLAP build byte-for-byte.
- **Buses**: stereo in / stereo out, the right answers to every arrangement
  negotiation. Value formatting/parsing for host displays. Factory metadata.
- **Entry points** per platform and format, from the macros.

### Threading model (the part JUCE hides)

| Call | Thread |
|---|---|
| `prepare` | main (host setup) — allocation allowed |
| `processBlock` | audio — real-time rules apply |
| `setParameter` | **both** (host automation arrives on audio; generic UI edits arrive on main) — hence the atomic-setter contract |
| `getState`/`setState` | main |

---

## Custom UIs — the WebView direction

v1 plugins use the host's generic editor on purpose: it is fully functional
(sliders, automation, state) and lets the DSP ship without blocking on GUI
infrastructure.

The planned editor layer — designed, not yet implemented — is **WebView
based**: your plugin declares `hasEditor = true` and provides its interface
as HTML/CSS/JS; the layer embeds the platform web engine (WebView2 on
Windows, WKWebView on macOS, WebKitGTK on Linux) inside the window each
format hands us (VST3 `IPlugView`, CLAP `clap.gui` — whose spec already has
a webview draft extension, so the industry is converging on exactly this),
plus a small JS bridge:

```js
dspark.setParam("drive", 6.0);          // UI -> DSP (same stable text ids)
dspark.onParam("drive", v => { ... });  // DSP/automation -> UI
dspark.beginEdit("drive"); /* drag */ dspark.endEdit("drive");  // host undo
```

One UI codebase for every format and platform, hot-reloadable during
development, and no C++ GUI framework to learn. Until it lands, nothing in
your plugin class changes — the editor is purely additive.

---

## Verifying and shipping

1. **Smoke hosts** (in this repo, run in CI on every platform):
   `tools/vst3_smoke_host.cpp` and `tools/clap_smoke_host.cpp` drive your
   module through the full lifecycle like a DAW and exit non-zero on any
   misbehaviour.
2. **Official validators** before release: Steinberg's `validator` (ships
   with the VST3 SDK), [`clap-validator`](https://github.com/free-audio/clap-validator),
   and Tracktion's [`pluginval`](https://github.com/Tracktion/pluginval) at
   strictness 10.
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

**Mono? Sidechain? MIDI?** Not in v1 — stereo effect buses only. The layer
will grow `Category::Instrument` and bus configs without breaking the
current contract.

**Where do the parameter values live?** Wherever your DSPark members keep
them. The wrapper keeps normalized shadows only for host queries and state;
your `setParameter` is the single source of truth for the DSP.
