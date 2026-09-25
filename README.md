<p align="center">
  <img src="https://raw.githubusercontent.com/CristianMoresi/DSPark/main/docs/img/dspark-gh.png" alt="DSPark - header-only audio DSP in C++20" width="100%">
</p>

# DSPark

[![CI](https://github.com/CristianMoresi/DSPark/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/CristianMoresi/DSPark/actions/workflows/ci.yml)
[![Docs](https://github.com/CristianMoresi/DSPark/actions/workflows/docs.yml/badge.svg?branch=main)](https://cristianmoresi.github.io/DSPark/)
[![Release](https://img.shields.io/github/v/release/CristianMoresi/DSPark?label=release)](https://github.com/CristianMoresi/DSPark/releases/tag/v1.7.0)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DSPark is a header-only C++20 audio DSP library for developers building
real-time and offline audio software. It provides effects, filters, synthesis
and spectral building blocks, analysis and musical tools, native file I/O, and
an optional VST3/CLAP/Audio Unit v2 plugin layer.

The framework modules have no third-party dependencies and require no
separately compiled DSPark library. Optimized paths may use compiler-provided
SIMD intrinsics behind platform guards. The plugin adapters and the Windows-only
DSParkLab application are separate from the umbrella header and carry their own
platform integration requirements.

The v1.7.0 CI matrix covers Windows, Linux, macOS, WebAssembly, and an embedded
profile. The framework headers also target iOS and Android, although those two
platforms do not have dedicated v1.7.0 CI jobs.

**Current release:** [v1.7.0](https://github.com/CristianMoresi/DSPark/releases/tag/v1.7.0)
| [API reference](https://cristianmoresi.github.io/DSPark/)
| [Cookbook](docs/cookbook.md)
| [Examples](https://github.com/CristianMoresi/DSPark/blob/v1.7.0/examples/README.md)
| [Plugin guide](docs/plugins.md)

## Quick start

### Requirements

- A C++20 compiler.
- CMake 3.21 or newer when using the supplied CMake project.
- An optimized build for production DSP (`-O2` or `/O2`); header-only code is
  compiled with the flags of the consuming target.

### Add DSPark to a project

Vendor the repository at `third_party/DSPark`, add that directory to your
include path, and include the umbrella header:

```cpp
#include <DSPark.h>
```

For a direct compiler invocation:

```bash
c++ -std=c++20 -O2 -I third_party/DSPark app.cpp -o app
```

For CMake, the interface target supplies the include path and C++20
requirement:

```cmake
add_subdirectory(third_party/DSPark)
target_link_libraries(app PRIVATE dspark::dspark)
```

A CMake installation is consumed with `find_package(dspark CONFIG REQUIRED)`
and the same `dspark::dspark` target.

Or fetch the v1.7.0 release tag:

```cmake
include(FetchContent)
FetchContent_Declare(dspark
    GIT_REPOSITORY https://github.com/CristianMoresi/DSPark.git
    GIT_TAG v1.7.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(dspark)

target_link_libraries(app PRIVATE dspark::dspark)
```

### Prepare and process a block

A typical block processor is constructed and configured with
`prepare(AudioSpec)` before its audio-stream owner calls `processBlock()`.

```cpp
#include <DSPark.h>

int main()
{
    const dspark::AudioSpec spec { 48000.0, 512, 2 };

    dspark::Equalizer<float> equalizer;
    dspark::Compressor<float> compressor;
    dspark::Limiter<float> limiter;

    equalizer.prepare(spec);
    compressor.prepare(spec);
    limiter.prepare(spec);

    equalizer.setBand(0, 100.0f, 3.0f);
    compressor.setThreshold(-18.0f);
    compressor.setRatio(3.0f);
    limiter.setCeiling(-1.0f);

    dspark::AudioBuffer<float> audio;
    audio.resize(spec.numChannels, spec.maxBlockSize);

    auto block = audio.toView();
    equalizer.processBlock(block);
    compressor.processBlock(block);
    limiter.processBlock(block);
}
```

In an audio host, create an `AudioBufferView<float>` over the host-owned channel
pointers instead of allocating an `AudioBuffer` in the callback. See the
[real-time-style channel-strip example](https://github.com/CristianMoresi/DSPark/blob/v1.7.0/examples/channel_strip.cpp)
and the [offline WAV example](https://github.com/CristianMoresi/DSPark/blob/v1.7.0/examples/wav_process.cpp)
for complete programs.

## Capabilities

| Area | Representative APIs |
|---|---|
| Filters and equalization | `Biquad`, `StateVariableFilter`, `LadderFilter`, `FilterEngine`, `Equalizer`, `CrossoverFilter` |
| Dynamics and level control | `Compressor`, `Limiter`, `NoiseGate`, `Expander`, `DynamicEQ`, `MultibandCompressor`, `DeEsser` |
| Nonlinear and analog modeling | `Saturation`, `Clipper`, `TapeMachine`, `TubePreamp`, `TransformerModel`, `wdf::*` |
| Time, pitch, and spatial processing | `Delay`, `AlgorithmicReverb`, `Reverb`, `PitchShifter`, `TimeStretch`, `PitchCorrector`, `GranularProcessor` |
| Spectral processing | `FFTReal`, `Convolver`, `ZeroLatencyConvolver`, `SpectralProcessor`, `SpectralFreeze`, `SpectralDenoiser` |
| Analysis and metering | `SpectrumAnalyzer`, `LoudnessMeter`, `LoudnessNormalizer`, `PitchDetector`, `OnsetDetector`, `BeatTracker`, `LoopFinder` |
| Synthesis and music | `Oscillator`, `WavetableOscillator`, `ADSREnvelope`, `ChordDetector`, `KeyDetector`, `harmony::*` |
| Infrastructure | `AudioBuffer`, `ProcessorChain`, `Resampler`, `Oversampling`, `Dither`, `SpscQueue`, versioned state blobs |
| File I/O | WAV and MP3 read/write, Standard MIDI File read/write, native FLAC decode |
| Plugin development | Native VST3 and CLAP adapters, macOS Audio Unit v2, instruments, sidechains, automation, state, and optional WebView editors |

The [API reference](https://cristianmoresi.github.io/DSPark/) is the exhaustive
class and method inventory. The [cookbook](docs/cookbook.md) covers production
workflows including mastering, restoration, tempo detection, time stretching,
loop finding, and pitch correction.

### File I/O qualification

WAV and MP3 support includes reading and writing. `MidiFile` reads Standard
MIDI File formats 0, 1, and 2 and writes formats 0 and 1. `FlacFile` is a
decode-only native FLAC implementation; its write operations return `false`.
Define `DSPARK_NO_FILE_IO` before including `DSPark.h` to omit all file-I/O
headers for targets without filesystem support.

## Real-time and threading contract

Real-time safety is a per-method contract, not a blanket property of every
operation:

- Construction, `prepare()`, state restoration, and setup-only configuration
  may allocate and must not run concurrently with processing.
- A prepared processor has one audio-stream owner. Processing methods that a
  component documents as real-time-safe avoid allocation and unbounded waits;
  offline utilities are outside that contract.
- Parameter publication normally assumes one control writer and one audio
  owner. Some setters are safe during processing; setup setters such as changes
  that resize oversampling storage are not.
- Before sharing an instance, read its documented threading contract. Report
  the latency of lookahead, FFT, convolution, and oversampled paths to the host.

The complete ownership and publication rules are in the
[threading model](docs/threading.md).

## Platform and validation coverage

| Target | v1.7.0 validation |
|---|---|
| Windows | MSVC x64 and ARM64: test and conformance suites |
| Linux | GCC x64/ARM64 and Clang x64: test and conformance suites |
| macOS | Clang ARM64: test and conformance suites |
| WebAssembly | Emscripten conformance in scalar and SIMD128 configurations |
| Embedded profile | GCC compile gate with file I/O, exceptions, and RTTI disabled |
| iOS and Android | Portable framework headers; not separate targets in the v1.7.0 CI matrix |

CI also checks installed-header self-sufficiency with GCC and Clang, warning-free
builds, sanitizers, concurrent publication paths, examples, native plugin
smoke hosts and validators, and generated documentation. The official EBU
loudness-vector job is best effort because the licensed test material is
downloaded separately; the job reports a skip when the vectors are unavailable.

Measured processor and analyzer results, including settings and limitations,
are published in the [quality metrics table](docs/metrics.md).

## Native plugins

DSPark supplies native adapters for VST3 and CLAP on desktop platforms and
Audio Unit v2 on macOS. The required VST3 C API and CLAP headers are included
in the repository; the CMake helper builds the platform-specific module or
bundle layout.

The shared plugin contract covers parameters, automation, state, latency,
tail reporting, bypass, mono/stereo layouts, sidechains, transport, MIDI
instruments, offline-rendering modes, and factory presets. Custom editors can
be written in HTML/CSS/JavaScript and embedded with the platform WebView layer.
On Linux, a custom editor requires WebKitGTK at runtime; the host's generic
parameter UI remains the fallback when a WebView is unavailable.

Start with the [plugin guide](docs/plugins.md) and the
[plugin examples](https://github.com/CristianMoresi/DSPark/blob/v1.7.0/examples/README.md).

## DSParkLab

`DSParkLab/` is a Windows-only Win32/Direct3D 11 application for interactive
processor testing. Its bundled Dear ImGui and miniaudio sources are confined to
the application and are not dependencies of the DSPark framework.
Build it with `DSParkLab\build.bat` (Visual Studio 2019 or later, any edition).

## Documentation

- [API reference](https://cristianmoresi.github.io/DSPark/) - generated from
  the public headers and guides.
- [Cookbook](docs/cookbook.md) - task-oriented DSP recipes.
- [Examples](https://github.com/CristianMoresi/DSPark/blob/v1.7.0/examples/README.md) - standalone processing and plugin projects.
- [Plugin guide](docs/plugins.md) - formats, host contract, editors, and
  shipping checks.
- [Threading model](docs/threading.md) - setup, control, audio, and readout
  ownership.
- [Quality metrics](docs/metrics.md) - measured settings, results, and caveats.
- [v1.7.0 migration guide](docs/migration-v1.7.0.md) - source and latency
  changes from v1.6.1.
- [Changelog](CHANGELOG.md) - release history.

## What's new in v1.7.0

- Musical timing and editing: `OnsetDetector`, `BeatTracker`, `LoopFinder`,
  `KeyDetector`, and `LoudnessNormalizer`.
- Pitch and spectral processing: `TimeStretch`, `PitchCorrector`, and
  `SpectralFreeze`.
- Native formats: dependency-free FLAC decoding and Standard MIDI File reading
  and writing, plus hardened WAV and MP3 parsing.
- Public-contract updates: rate-aware analysis windows, bounded staged-state
  adoption, double-precision biquad coefficients, const-correct buffer views,
  and a stricter in-place `AudioProcessor` concept.

Read the [release notes](https://github.com/CristianMoresi/DSPark/releases/tag/v1.7.0),
[full changelog](CHANGELOG.md), and
[migration guide](docs/migration-v1.7.0.md).

## Build and contribute

From a repository checkout:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
```

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development workflow and conventions. Please open an issue to discuss a
proposed change before submitting a pull request. Bug reports should include
the platform, compiler version, and a minimal reproduction.

## License

DSPark is available under the [MIT License](LICENSE).

## Author

Created and maintained by [Cristian Moresi](https://github.com/CristianMoresi).
Contact: [dev@cristianmoresi.com](mailto:dev@cristianmoresi.com).
