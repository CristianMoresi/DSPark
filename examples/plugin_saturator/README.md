# A complete VST3 + CLAP plugin with DSPark alone — no SDK download

`saturator.cpp` is a finished, loadable effect (drive / algorithm / mix /
output around `dspark::Saturation`) written against DSPark's native plugin
layer. Everything it needs ships in this repository: the format-agnostic
layer (`plugin/DSParkPlugin.h`), the VST3 backend implementing the COM ABI
directly (`plugin/vst3/DSParkVst3.h`, with Steinberg's official C API header
vendored under its permissive 2025 license) and the CLAP backend
(`plugin/clap/DSParkClap.h`, with the MIT CLAP headers vendored).

The two `DSPARK_*_PLUGIN` macros at the bottom make ONE binary carry both
entry points: ship the compiled module as `.vst3` and as `.clap`. Presets
saved in one format load in the other (shared state container).

## Build

One translation unit, one command:

```bat
:: Windows (from the repo root)
cl /std:c++20 /O2 /LD /EHsc /I . examples\plugin_saturator\saturator.cpp /Fe:DSParkSaturator.vst3
```

```bash
# Linux
g++ -std=c++20 -O2 -fPIC -shared -I . examples/plugin_saturator/saturator.cpp -o DSParkSaturator.vst3
```

Or with the bundle-aware CMake helper (correct .vst3 folder layout on every
platform, including the macOS bundle):

```cmake
include(plugin/cmake/DSParkPlugin.cmake)
dspark_add_plugin(DSParkSaturator
    SOURCES examples/plugin_saturator/saturator.cpp
    FORMATS VST3)
```

Install by copying the result into the platform VST3 folder
(`C:\Program Files\Common Files\VST3`, `~/.vst3`,
`/Library/Audio/Plug-Ins/VST3`). The plugin shows the host's generic
parameter editor; a DSPark editor layer is on the roadmap.

## Verify before shipping

```
cl /std:c++20 /O2 /EHsc /I . tools\vst3_smoke_host.cpp /Fe:vst3_smoke_host.exe
vst3_smoke_host DSParkSaturator.vst3
```

The smoke host drives the module exactly like a DAW (factory, buses,
parameters, processing, automation, state round-trip, teardown) and exits
non-zero on any misbehaviour. Steinberg's `validator` and Tracktion's
`pluginval` are recommended as final gates.

## Anatomy

```cpp
struct DSParkSaturator
{
    static constexpr auto descriptor = dspark::plugin::Descriptor { ... };
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("drive", "Drive", -12.0f, 36.0f, 0.0f, "dB"),
        ...);

    void prepare(const dspark::AudioSpec& spec);            // DSPark as usual
    void setParameter(int index, float value) noexcept;     // any thread
    void processBlock(dspark::AudioBufferView<float> io) noexcept;
    int  getLatency() const noexcept;                       // optional
};

DSPARK_VST3_PLUGIN(DSParkSaturator)
```

State saving, the bypass parameter, latency reporting, bus negotiation and
parameter automation are handled by the layer. Parameter ids are stable text
strings — reorder or insert parameters freely between versions without
breaking saved sessions.
