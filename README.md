# DSPark

**DSP** + **Ark** + **The Spark**

**A header-only audio DSP framework in pure C++20. Zero external dependencies.**

**v1.2.2** — 90 headers. One `#include`. Ready to build plugins, desktop apps, WebAssembly, mobile, embedded.

**📖 Full API documentation: [cristianmoresi.github.io/DSPark](https://cristianmoresi.github.io/DSPark/)**

CI builds and tests every commit on Windows (MSVC), Linux (GCC + Clang, x64 and ARM64), macOS (ARM64) and WebAssembly (Emscripten), plus AddressSanitizer/UBSan, an exceptions-free embedded profile and a single-header amalgamation. The public conformance suite validates loudness against the official EBU R128 test vectors and generates a [per-processor quality metrics table](docs/metrics.md) (THD+N, noise floor, spurious/aliasing, latency).

```cpp
#include "DSPark/DSPark.h"

dspark::Equalizer<float> eq;
dspark::Compressor<float> comp;
dspark::AlgorithmicReverb<float> reverb;

eq.prepare(spec);
comp.prepare(spec);
reverb.prepare(spec);

eq.setBand(0, 100.0f, 3.0f);            // Boost bass
comp.setThreshold(-18.0f);              // Compress
comp.setCharacter(dspark::Compressor<float>::Character::FET);  // 1176-style
reverb.setType(dspark::AlgorithmicReverb<float>::Type::Hall);  // Concert hall

eq.processBlock(buffer);
comp.processBlock(buffer);
reverb.processBlock(buffer);
```

No build system. No linking. No configuration. Just include and go.

## Build a VST3 plugin — no JUCE required

DSPark ships a native plugin layer: describe your plugin declaratively,
implement the usual DSPark contract, add one macro — and compile a loadable
VST3 with nothing but this repository (Steinberg's official C API header is
vendored under its permissive 2025 license).

```cpp
#include "plugin/vst3/DSParkVst3.h"

struct MySaturator
{
    static constexpr auto descriptor = dspark::plugin::Descriptor {
        .name = "My Saturator", .vendor = "Me",
        .productId = "com.me.mysaturator", .version = "1.0.0",
    };
    static constexpr auto parameters = dspark::plugin::params(
        dspark::plugin::param("drive", "Drive", -12.0f, 36.0f, 0.0f, "dB"));

    void prepare(const dspark::AudioSpec& spec)   { sat_.prepare(spec); }
    void setParameter(int, float v) noexcept      { sat_.setDrive(v); }
    void processBlock(dspark::AudioBufferView<float> io) noexcept { sat_.processBlock(io); }

    dspark::Saturation<float> sat_;
};
DSPARK_VST3_PLUGIN(MySaturator)
```

```
cl /std:c++20 /O2 /LD /EHsc /I . mysaturator.cpp /Fe:MySaturator.vst3
```

Parameter automation, state save/restore, bypass, latency reporting and bus
negotiation are handled by the layer; `tools/vst3_smoke_host.cpp` drives the
result through the full VST3 lifecycle like a DAW would (it runs in CI on
Windows, Linux and macOS). See `examples/plugin_saturator/`. CLAP and AU
backends over the same plugin class are on the roadmap.

---

## Why DSPark

Most audio DSP libraries fall into two categories: either they require a massive framework dependency, or they cover only a narrow slice of what you need. Nothing gives you everything in a single, portable, dependency-free package.

DSPark was built to change that. Whether you are:

- **A software developer** adding audio features to an app (zero DSP knowledge required)
- **A mixing engineer or sound technician** building custom plugins and tools
- **A DSP engineer** designing professional audio processors or embedded systems

You get the same framework, the same headers, the same API. The difference is how deep you go.

### Progressive Disclosure API

Every processor exposes three levels of complexity:

```cpp
// Level 1 — Just works:
eq.setBand(0, 1000.0f, -3.0f);

// Level 2 — More control:
eq.setBand(0, 1000.0f, -3.0f, 1.5f);   // Adds Q factor

// Level 3 — Full control:
eq.setBand(0, { .frequency = 1000, .gain = -3, .q = 1.5,
                .type = BandType::LowShelf, .slope = 24 });
```

You never see complexity you don't need. But it's always there when you do.

### Extensible by Design

Effect classes expose **protected internals** so you can subclass them directly to reach the delay lines, filters and early reflections inside. They are deliberately **leaf classes with non-virtual destructors** — honouring the zero-virtual-dispatch design — so you extend by direct inheritance and composition rather than polymorphic base-pointer deletion:

```cpp
class MyReverb : public dspark::AlgorithmicReverb<float> {
    // Access FDN delay lines, absorption filters, early reflections...
    // Add custom processing stages, expose your own presets
};
```

---

## What's Included

### Effects (36 processors)

| Class | Description |
|---|---|
| `Equalizer<T>` | Multi-band parametric EQ with **linear-phase** (FFT) and IIR modes |
| `Compressor<T>` | Modular: 5 detectors (Peak, RMS, TruePeak, SplitPolarity, Hilbert), 2 topologies (FF/FB), 4 characters (Clean/Opto/FET/Varimu), upward/downward modes. Hold and range controls, adaptive auto-makeup gain, external sidechain. |
| `Limiter<T>` | ISP true-peak brickwall limiter with adaptive release |
| `NoiseGate<T>` | State machine with hysteresis, hold time, duck mode. External sidechain. |
| `Expander<T>` | Downward expander with threshold, ratio, hold, range |
| `DynamicEQ<T>` | Frequency-selective dynamics (above/below threshold, dual-action) |
| `MultibandCompressor<T>` | Crossover + per-band Compressor, configurable up to 12 bands (compile-time `MaxBands`) |
| `TransientDesigner<T>` | Attack/sustain shaping via envelope following |
| `DeEsser<T>` | Frequency-targeted dynamic reduction for sibilance |
| `AlgorithmicReverb<T>` | 16-line FDN with Jot (1991) frequency-dependent absorption, Householder feedback mixing, Lexicon-style smooth random + noise modulation, serial allpass per line (Infinity2-style), feedback IIR smoothing (Verbity), allpass-interpolated modulated reads, tanh soft saturation, parallel allpass input diffuser (256 echo paths), Dattorro multi-tap output, output diffusion, M/S stereo width control, progressive ER absorption. 6 presets: Room, Hall, Chamber, Plate, Spring, Cathedral. Full parameter API for custom reverb design. |
| `Reverb<T>` | Convolution reverb with IR loading, pre-delay, auto-resample |
| `Saturation<T>` | 10 algorithms (Tube, Tape, Transformer, SoftClip, HardClip, Exciter, Wavefolder, Bitcrusher, Downsample, MultiStage). Adaptive blend, slew-dependent saturation, oversampling. |
| `Clipper<T>` | 4-mode clipper (Hard/Soft/Analog/GoldenRatio), multi-stage, slew limiter, up to 16x oversampling |
| `FilterEngine<T>` | Cascaded biquads, 9 shapes, 6–48 dB/oct slopes |
| `CrossoverFilter<T>` | Linkwitz-Riley crossover (LR12/24/48), IIR + linear-phase modes |
| `Chorus<T>` | Multi-voice LFO delay, stereo spread, flanger mode |
| `Phaser<T>` | Allpass chain with LFO modulation, configurable stages, stereo LFO spread |
| `Tremolo<T>` | Amplitude modulation with configurable LFO |
| `Vibrato<T>` | Pitch modulation via modulated delay |
| `RingModulator<T>` | Ring modulation with carrier oscillator |
| `FrequencyShifter<T>` | Single-sideband frequency shift via Hilbert transform |
| `Delay<T>` | Interpolated delay with feedback (clean or analog tanh regeneration), ping-pong, filters |
| `Panner<T>` | 6 algorithms: equal-power, binaural (ITD), mid-pan, side-pan, Haas, spectral |
| `Gain<T>` | Smoothed gain with fade, mute, polarity inversion |
| `AutoGain<T>` | Automatic gain compensation based on loudness measurement |
| `Crossfade<T>` | Linear, equal-power, S-curve |
| `StereoWidth<T>` | M/S width control with bass-mono option |
| `MidSide<T>` | Stereo Mid/Side encoding and decoding |
| `NoiseGenerator<T>` | White, pink, and brown noise generation |
| `DCBlocker<T>` | DC offset removal (1-pole or Butterworth order 2–10) |
| `TapeMachine<T>` | Physical tape model: Jiles-Atherton hysteresis at 2x oversampling, NAB/CCIR record/play EQ with exact digital inverses, speed-dependent head-gap/spacing/thickness loss, head bump, common-transport wow & flutter |
| `TubePreamp<T>` | Koren triode stages (12AX7) solved per sample with Newton-Raphson, exact Fender FMV tone stack as a Wave Digital R-type network, power-supply sag |
| `TransformerModel<T>` | Audio transformer coloration: flux-domain Jiles-Atherton core (distortion rises as frequency falls — the LF "bloom"), magnetizing-inductance corner, HF resonance bell |
| `PitchShifter<T>` | Phase vocoder with identity phase locking (Laroche-Dolson), exact tuning, transient phase reset, **formant preservation** (cepstral lift), stereo-coherent |
| `GranularProcessor<T>` | 64-grain clouds over live input: per-grain pitch/pan/jitter, freeze, equal-power spread |
| `SpectralDenoiser<T>` | Learnable-noise-profile spectral gating with the standard musical-noise defenses |

### Core (40 building blocks)

| Class | Description |
|---|---|
| `StateVariableFilter<T>` | TPT SVF: 8 modes (LP/HP/BP/Notch/AP/Bell/LowShelf/HighShelf), simultaneous multi-output |
| `LadderFilter<T>` | Moog-style 4-pole TPT filter, 6 modes, drive, self-oscillation |
| `Biquad<T>` | TDF-II biquad with 9 coefficient types and lock-free auto-promote of staged coefficients |
| `BiquadCoeffs<T>` | Standalone factory for biquad coefficients (LP, HP, BP, Peak, Shelf, Notch, AP, Tilt, DC blocker) |
| `FFTComplex<T>` / `FFTReal<T>` | Radix-2 FFT with SIMD (SSE3/NEON), real-optimised |
| `Convolver<T>` | Partitioned overlap-save FFT convolution |
| `ZeroLatencyConvolver<T>` | Gardner non-uniform partitioning: zero-latency convolution with time-distributed tail FFTs (flat CPU even for second-long IRs) |
| `wdf::*` (WDF.h) | Wave Digital Filter circuit toolkit: R/L/C leaves, series/parallel adaptors, Newton-Raphson diode roots with analytic seeds, and an R-type adaptor (MNA-derived scattering) for non-adaptable topologies |
| `Hysteresis<T>` | Jiles-Atherton magnetic hysteresis with implicit trapezoidal Newton-Raphson solver (tape, transformers) |
| `ModulationRouter<T>` | Block-rate modulation routing: any source callable to any parameter setter, with depth and smoothing |
| `StateWriter`/`StateReader` | Versioned key/value preset blobs + JSON helpers — every effect implements `getState()`/`setState()` |
| `FIRFilter<T>` | FIR engine with windowed-sinc design |
| `Oversampling<T>` | 2x–16x polyphase half-band Kaiser filters (-80 dB+ rejection), transparent up/down round-trip, exact reported latency |
| `Oscillator<T>` | PolyBLEP (sine, saw, square, triangle) |
| `WavetableOscillator<T>` | Mipmapped wavetable with bandlimited harmonics |
| `Resampler<T>` | Polyphase windowed-sinc sample-rate conversion |
| `EnvelopeGenerator<T>` (`ADSREnvelope`) | ADSR with exponential curves |
| `RingBuffer<T>` | Power-of-two circular buffer with interpolated read |
| `SmoothedValue<T>` | Parameter smoother (exponential, linear, chase, or disabled) |
| `Smoothers` | 9 smoothing algorithms (linear, exponential, one-pole, asymmetric, slew, SVF, Butterworth…) |
| `ProcessorChain<T,...>` | Zero-overhead compile-time processor chain with per-slot bypass |
| `SpectralProcessor<T>` | STFT-based analysis-modification-synthesis framework |
| `Dither<T>` | TPDF dithering with noise shaping |
| `DenormalGuard` | RAII FTZ/DAZ (x86 SSE, ARM, WebAssembly) |
| `Interpolation` | 5 methods (linear, cubic, Hermite, Lagrange, allpass) |
| `Hilbert<T>` | FIR (windowed-sinc) Hilbert transform for analytic signals — flat magnitude across the audible band |
| `WindowFunctions<T>` | 8 windows (Hann, Hamming, Blackman, Kaiser…) |
| `DryWetMixer<T>` | Parallel dry/wet mixing for effects |
| `TruePeakDetector<T>` | Shared ITU-R BS.1770-4 inter-sample peak detector (used by Compressor, Limiter, LoudnessMeter) |
| `SpinLock` | RT-safe spinlock for thread-safe parameters |
| `SpscQueue<T>` | Lock-free single-producer / single-consumer queue |
| `AudioSpec` | Audio environment descriptor (sample rate, block size, channels) |
| `AudioBuffer<T>` | 32-byte aligned owning buffer (SIMD-ready) |
| `AudioBufferView<T>` | Non-owning view (what processors receive) |
| `SimdOps` | SIMD-accelerated buffer operations (SSE2/AVX/NEON with scalar fallback) |
| `DspMath` | Constants, dB ⇄ gain, fast tanh / tan / sin / cos / exp / log / pow10, range mapping |
| `Phasor<T>` | Phase accumulator for LFO and oscillator construction |
| `SampleAndHold<T>` | Sample-and-hold with configurable hold time |
| `WaveshapeTable<T>` | LUT-based waveshaping with linear / cubic interpolation |
| `AnalogRandom` | Analog-flavoured random generators (smooth, noise, jitter) |
| `AnalogConstants` | Reference constants from analog-hardware research (zero runtime cost) |
| `ProcessorTraits` | C++20 concepts: `AudioProcessor`, `SampleProcessor`, `GeneratorProcessor` |

### Analysis (8 analyzers)

| Class | Description |
|---|---|
| `LevelFollower<T>` | Peak and RMS envelope follower |
| `EnvelopeFollower<T>` | Public attack/release detector (Peak or RMS law) for sidechains, modulation and metering |
| `SpectrumAnalyzer<T>` | Real-time FFT spectrum with peak hold |
| `LoudnessMeter<T>` | EBU R128: momentary, short-term, integrated, LRA, true peak — **passes the official EBU test vectors** (Tech 3341/3342, BS.1770-5 K-weighting and true-peak interpolator) |
| `Goertzel<T>` | Single-frequency O(N) magnitude detection |
| `PitchDetector<T>` | YIN pitch detection with FFT-accelerated difference function (O(N log N)) |
| `PitchFollower<T>` | Musical pitch tracking source: confidence gating, octave-jump correction, constant-rate semitone glide |
| `PhaseCorrelation<T>` | Stereo correlation/balance meter with a goniometer (vectorscope) point feed |

### I/O (3 file handlers)

| Class | Description |
|---|---|
| `WavFile` | Read/write WAV (PCM 8/16/24/32-bit, float 32/64-bit) |
| `Mp3File` | MPEG-1 Layer III codec — read (CBR/VBR) + write (CBR encoder, 32–320 kbps) |
| `AudioFile` | Abstract base class for custom format implementations |

### Music (2 modules)

| Class | Description |
|---|---|
| `HarmonyConstants` | Constexpr musical harmony toolkit: 61 scales (bitmask representation), 15 chord recipes with inversions, MIDI/note conversion, key-aware naming (sharp/flat), diatonic chord generation. Fully `constexpr`/`consteval` — generates static tables at compile time. |
| `ChordDetector<T>` | Real-time chord recognition: per-note Goertzel chroma, template matching over ten chord families, bass-note root disambiguation, confidence-gated hold |

---

## Quick Start

### Installation

Copy the `DSPark/` folder into your project. Done.

```cpp
#include "DSPark/DSPark.h"
```

Requires a C++20 compiler. Tested with MSVC 19.50+, and compatible with GCC 12+, Clang 15+, Emscripten 3+.

### Process a WAV File

```cpp
#include "DSPark/DSPark.h"

int main()
{
    dspark::WavFile input, output;
    input.openRead("input.wav");
    auto info = input.getInfo();

    dspark::AudioSpec spec { info.sampleRate, 512, info.numChannels };
    dspark::AudioBuffer<float> buffer(info.numChannels, 512);

    dspark::Compressor<float> comp;
    dspark::Limiter<float> lim;
    comp.prepare(spec);
    lim.prepare(spec);

    comp.setThreshold(-18.0f);
    comp.setCharacter(dspark::Compressor<float>::Character::Opto);
    lim.setCeiling(-1.0f);

    output.openWrite("output.wav", info);

    while (input.readSamples(buffer.toView()))
    {
        comp.processBlock(buffer.toView());
        lim.processBlock(buffer.toView());
        output.writeSamples(buffer.toView());
    }

    output.close();
}
```

### Build a Channel Strip

```cpp
using namespace dspark;

ProcessorChain<float,
    Equalizer<float>,
    Compressor<float>,
    Limiter<float>
> channelStrip;

channelStrip.prepare(spec);
channelStrip.processBlock(buffer);
```

### Per-Sample Processing (DSP Engineers)

```cpp
dspark::StateVariableFilter<float> svf;
svf.prepare(spec);
svf.setCutoff(2000.0f);
svf.setResonance(0.7f);

for (int i = 0; i < numSamples; ++i)
{
    svf.setCutoff(lfo.getNextSample() * 2000 + 500);  // Modulate per sample
    auto [lp, hp, bp] = svf.processMultiOutput(input[i], 0);
    output[i] = lp;  // Or hp, bp, or any combination
}
```

### Draw an EQ Curve (Plugin GUI)

```cpp
dspark::Equalizer<float> eq;
eq.prepare(spec);
eq.setBand(0, 80.0f, 4.0f);
eq.setBand(1, 3000.0f, -2.0f, 2.0f);

// Get magnitude response for drawing
std::vector<float> freqs(512), mags(512);
for (int i = 0; i < 512; ++i)
    freqs[i] = 20.0f * std::pow(1000.0f, static_cast<float>(i) / 511.0f);

eq.getMagnitudeForFrequencyArray(freqs.data(), mags.data(), 512);
// mags[] now contains the combined EQ curve — plot it in your GUI
```

---

## DSParkLab — Interactive Testing App

DSParkLab is an interactive, plugin-style GUI application for real-time testing of every DSPark processor. Load any audio file, enable effects, shape them on an interactive analyzer or with parameter controls, and hear the results instantly.

```bash
# Build (requires MSVC with C++20 support + Windows SDK for D3D11)
DSParkLab\build.bat
DSParkLab\DSParkLab.exe
```

**Features:**

- Asynchronous WAV/MP3 loading with format auto-detection (audio thread never blocks on I/O)
- 34 effects organised by category (Filters, Dynamics, Distortion, Analog, Modulation, Pitch, Spatial, Utility) — including the physical tape/tube/transformer models, pitch shifter with formant preservation, granular engine and denoiser
- **Interactive analyzer**: log-frequency spectrum behind a live response curve, with **draggable nodes** for the EQ / Filter / Dynamic EQ (drag X = frequency, Y = gain, mouse-wheel = Q)
- Dynamic EQ draws its **live** per-band reaction in real time; multiband compressor shows crossover bands with per-band gain reduction
- Auto-generated parameter panels with mouse-wheel-adjustable sliders and combo boxes
- Live oversampling selection (up to 16x) on Saturation and Clipper for audible anti-alias comparison
- Convolution Reverb loads real impulse responses (WAV) or builds a synthetic one
- Per-channel level meters, waveform view, and gain-reduction metering
- A/B bypass for instant comparison, and transport controls (play, pause, stop, seek, loop)

Built with [Dear ImGui](https://github.com/ocornut/imgui) (MIT) and [miniaudio](https://miniaud.io/) (public domain). These dependencies are bundled in `DSParkLab/vendor/` and are only used by the testing app — the DSPark framework itself remains 100% dependency-free.

---

## Platform Support

| Platform | Status | Notes |
|---|---|---|
| Windows (MSVC) | Tested | C++20, /W4 /WX- (only benign C4324 `alignas` padding notices) |
| Linux (GCC) | Compatible | C++20, -Wall -Wextra |
| macOS (Clang) | Compatible | C++20, -Wall -Wextra |
| WebAssembly (Emscripten) | Compatible | Zero syscalls in audio path |
| iOS / Android | Compatible | ARM NEON denormal flush supported |
| VST3 plugins | Use as DSP engine | Pair with JUCE or iPlug2 for the wrapper |

---

## Technical Highlights

- **C++20**: Concepts (`AudioProcessor`, `SampleProcessor`, `GeneratorProcessor`), designated initializers, `std::numbers`
- **Zero allocation in audio thread**: All memory pre-allocated in `prepare()`
- **SIMD inner loops**: SSE2/SSE3/AVX/NEON-accelerated buffer operations and FFT, with automatic scalar fallback
- **Cache-friendly**: Contiguous memory, 32-byte aligned buffers
- **Thread-safe parameters**: All setters use `std::atomic` with `memory_order_relaxed` — safe to call from any thread (UI, automation, audio) with zero contention
- **Lock-free coefficient updates**: `Biquad::setCoeffs()` is consumed automatically by `processBlock()` and `processSample()` via a relaxed-load fast path. No external sequencing required.
- **No virtual dispatch in hot path**: Templates and compile-time polymorphism
- **Physically-modeled algorithms**: Tape (Chowdhury 2019 hysteresis with Langevin function, head bump pre-filter, gap-loss HF rolloff), FDN reverb (Jot 1991 absorption, Householder mixing, Dattorro 1997 multi-tap output, Lexicon-style modulation, serial allpass density, allpass interpolation, tanh soft saturation), TPT state-variable and ladder filters
- **Full Doxygen documentation**: Every public class and method documented — browse it at [cristianmoresi.github.io/DSPark](https://cristianmoresi.github.io/DSPark/)

---

## Architecture

```
DSPark/
├── DSPark.h                 # Single umbrella include + full documentation
├── Core/          (41)      # Building blocks: filters, FFT, WDF, oscillators, SIMD
├── Effects/       (36)      # Ready-to-use processors: EQ, compressor, reverb, tape...
├── Analysis/       (8)      # Metering: LUFS (EBU-verified), spectrum, pitch, correlation
├── IO/             (3)      # File I/O: WAV read/write, MP3 read/write
├── Music/          (2)      # Harmony constants + real-time chord detection
├── conformance/             # Public conformance suite (runs in CI)
├── docs/                    # Cookbook and guides
├── examples/                # WAV processing, channel strip, pitch-tracking EQ, VST3
└── DSParkLab/               # Interactive testing app (Win32 + ImGui + miniaudio)
```

---

## What's New in v1.2.2

- **Public per-processor metrics table** ([docs/metrics.md](docs/metrics.md)):
  THD+N, noise floor, spurious/aliasing and latency for 31 processors,
  generated by the conformance suite and regenerated as a CI artifact.
- **TubePreamp**: the DC operating point now settles over the active stage
  count (a ~70 ms supply transient on activation at 1-stage settings is gone
  — caught by the new table's noise-floor column).
- **DSParkLab**: test-tone generator with a live THD+N / fundamental / RMS
  readout in the meters panel.
- **API docs front page**: the README now renders as the Doxygen main page.

---

## What's New in v1.2.1

A sound-quality release: every fix below came from systematic listening
sessions in DSParkLab backed by measurement, and each one is now pinned by a
permanent contract test (insertion loudness, low-level linearity, spectral
balance, click-free parameter drags).

- **Hard sync rebuilt on a table minBLEP** (`Core/MinBlepTable.h`, new): a
  minimum-phase band-limited step built at prepare() time from first
  principles (windowed BLIT → real cepstrum → causal fold). Alias rejection
  improves from the 2-point PolyBLEP's -35..-44 dB to **-98..-108 dB**, and
  the synced triangle loses its inherent DC offset.
- **Saturation algorithms repaired**: the Tape model was a dead-zone (tape
  with no AC bias — quiet signals vanished as input²); it is now the correct
  anhysteretic Langevin curve solved per sample. The Wavefolder carried a
  hidden 2π (+16 dB) gain factor; the Transformer's band split imposed a
  fixed ±3 dB shelf tilt that buried mids and highs (worst in the MultiStage
  cascade, now properly gain-staged). All ten algorithms now honour one
  contract: unit small-signal gain at neutral drive, ±1 ceiling.
- **Analog models calibrated at program level**: TubePreamp no longer drops
  ~20 dB on activation (its calibration measured inside the supply-sag
  settling transient) and its FMV stack's fixed mid-scoop is flattened by a
  measured 3-section EQ — neutral knobs now sound neutral, tone controls act
  relative to flat. TransformerModel no longer explodes +21 dB at high drive
  (small-signal calibration probed the JA virgin curve instead of the major
  loop). Gain changes ramp geometrically, so drive drags are click-free even
  through the transformer's output differentiator.
- **Drive knobs that feel right**: TapeMachine, TubePreamp and
  TransformerModel share one loudness contract — the link is measured at
  each drive setting (the TubePreamp sweeps a prepare-time LUT) with a
  +0.25 dB/dB residual slope: backing off cleans and drops a touch, pushing
  adds density at a steady level.
- **EQ band types**: the `Equalizer` exposes its full type set per band
  (peak, shelves, cuts with 6-48 dB/oct slope, notch, bandpass, tilt) in
  DSParkLab again, drawn from the real biquad cascade
  (`buildBandStages` is now a public analysis API). `DynamicEQ` gains
  per-band **shapes** (bell / low shelf / high shelf) with shape-aware
  detectors — a framework-level addition, state-compatible with old presets.
- **API documentation now published** at
  [cristianmoresi.github.io/DSPark](https://cristianmoresi.github.io/DSPark/).

---

## What's New in v1.2

The "world-class or bust" release: new DSP, physical modelling, standards
conformance and infrastructure.

- **Analog physical models**: `TapeMachine` (Jiles-Atherton hysteresis, exact
  NAB/CCIR EQ inverses, speed-dependent loss stack, wow/flutter),
  `TubePreamp` (Koren triodes + the exact Fender tone stack as a WDF R-type
  network, verified to -94.8 dB against the published symbolic transfer
  function) and `TransformerModel` (flux-domain core, LF bloom).
- **Wave Digital Filters** (`Core/WDF.h`): a complete circuit-modelling
  toolkit verified to machine precision against analytic references.
- **Pitch tools**: `PitchShifter` (identity phase locking, exact tuning,
  formant preservation via cepstral lift), `PitchFollower` (musical tracking
  with gating/glide), `GranularProcessor` and `SpectralDenoiser`.
- **Zero-latency convolution** (`ZeroLatencyConvolver`): Gardner non-uniform
  partitioning with time-distributed tail work — null vs direct convolution
  at -132 dB and flat CPU at any block size.
- **EBU R128 conformance**: the loudness pipeline passes the official EBU
  test vectors (integrated, LRA, true peak) after exact BS.1770-5
  K-weighting and the Annex 2 polyphase true-peak interpolator.
- **State-of-the-art EQ option**: `BiquadCoeffs::makePeakMatched` implements
  the Orfanidis prescribed-Nyquist-gain design (de-cramped high bells);
  `Equalizer::setMatchedBells(true)` switches Peak bands to it.
- **Band-limited oscillator hard sync** (`setSyncRatio`) with PolyBLEP event
  bookkeeping, and a public `EnvelopeFollower` and `ModulationRouter`.
- **Presets everywhere**: every effect implements `getState()`/`setState()`
  (versioned binary blobs, tolerant restore, JSON helpers, nested multiband
  states).
- **Music analysis**: `ChordDetector` (Goertzel chroma + template matching
  with bass disambiguation) and `PhaseCorrelation` (correlation/goniometer).
- **Infrastructure**: multi-platform CI (11 jobs), public conformance suite,
  CMake package, single-header amalgamation, benchmark harness, cookbook
  (`docs/cookbook.md`), VST3 integration template, and portability fixes for
  GCC/Clang/libc++/SSE2-baseline uncovered by the first CI runs.

---

## What's New in v1.1

A deep quality and correctness pass over the entire framework:

- **Resampler**: rebuilt polyphase kernel (continuous Kaiser window, exact
  sub-sample phase) — sine-conversion error improved from audible distortion
  to **better than -100 dB**; SIMD-accelerated streaming path.
- **Equalizer**: linear-phase mode level calibration fixed (kernel scaling);
  shelf bands now map Q to slope consistently in IIR and linear-phase modes.
- **StereoWidth**: bass-mono crossover rebuilt with an exact complementary
  one-pole split (previous phase network damaged the stereo image at HF).
- **LadderFilter**: HP24 now has a true zero at DC at every resonance setting.
- **Oscillator**: triangle level normalisation corrected (was ~4 dB low).
- **Clipper**: Analog (sine) mode re-derived with unity small-signal gain.
- **Limiter**: exact brickwall guarantee (output never exceeds the ceiling),
  glitch-free live look-ahead changes.
- **LoudnessMeter**: BS.1770-4-conformant integrated gating (400 ms blocks,
  75% overlap) plus new loudness range (LRA, EBU Tech 3342) and true peak
  (dBTP) readouts.
- **PitchDetector**: YIN difference function now computed via FFT — ~20x
  faster, real-time friendly at the default window size.
- **Performance**: register-resident Biquad block processing, multi-accumulator
  SIMD dot product, SIMD double paths (AVX), an order-of-magnitude faster
  Hilbert transform, coefficient caching across FilterEngine and the TPT SVF,
  and new fast math approximations (sin / cos / log) used framework-wide.
- **Robustness**: release-safe channel clamps across the API surface, ARM/NEON
  build fix in the Convolver, allocation-safety fixes validated under
  AddressSanitizer, and a shared ITU-R BS.1770-4 true-peak detector.

---

## License

MIT License. See [LICENSE](LICENSE).

Free to use in commercial and open-source projects. Attribution appreciated.

---

## Author

**Cristian Moresi** — Software developer and music producer with professional experience in mixing engineering and sound design.

DSPark was created to provide a truly free, professional-grade DSP toolkit accessible to developers at every level of expertise — from desktop app builders to DSP engineers designing embedded audio systems. It is a genuine open-source alternative to commercial audio frameworks, built from the ground up with no dependencies and no compromises.

- GitHub: [github.com/crismoresi](https://github.com/crismoresi)
- LinkedIn: [linkedin.com/in/crismoresi](https://linkedin.com/in/crismoresi)
- Email: dev@cristianmoresi.com

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

For bug reports, include: compiler version, platform, minimal reproduction code.
