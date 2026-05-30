# DSPark

**DSP** + **Ark** + **The Spark**

**A header-only audio DSP framework in pure C++20. Zero external dependencies.**

**v1.1** — 73 headers. One `#include`. Ready to build plugins, desktop apps, WebAssembly, mobile, embedded.

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

### Effects (30 processors)

| Class | Description |
|---|---|
| `Equalizer<T>` | Multi-band parametric EQ with **linear-phase** (FFT) and IIR modes |
| `Compressor<T>` | Modular: 5 detectors (Peak, RMS, TruePeak, SplitPolarity, Hilbert), 2 topologies (FF/FB), 4 characters (Clean/Opto/FET/Varimu), upward/downward modes. Adaptive auto-makeup gain. External sidechain. |
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
| `Phaser<T>` | Allpass chain with LFO modulation, configurable stages |
| `Tremolo<T>` | Amplitude modulation with configurable LFO |
| `Vibrato<T>` | Pitch modulation via modulated delay |
| `RingModulator<T>` | Ring modulation with carrier oscillator |
| `FrequencyShifter<T>` | Single-sideband frequency shift via Hilbert transform |
| `Delay<T>` | Interpolated delay with feedback, ping-pong, filters |
| `Panner<T>` | 6 algorithms: equal-power, binaural (ITD), mid-pan, side-pan, Haas, spectral |
| `Gain<T>` | Smoothed gain with fade, mute, polarity inversion |
| `AutoGain<T>` | Automatic gain compensation based on loudness measurement |
| `Crossfade<T>` | Linear, equal-power, S-curve |
| `StereoWidth<T>` | M/S width control with bass-mono option |
| `MidSide<T>` | Stereo Mid/Side encoding and decoding |
| `NoiseGenerator<T>` | White, pink, and brown noise generation |
| `DCBlocker<T>` | DC offset removal (1-pole or Butterworth order 2–10) |

### Core (33 building blocks)

| Class | Description |
|---|---|
| `StateVariableFilter<T>` | TPT SVF: 8 modes (LP/HP/BP/Notch/AP/Bell/LowShelf/HighShelf), simultaneous multi-output |
| `LadderFilter<T>` | Moog-style 4-pole TPT filter, 6 modes, drive, self-oscillation |
| `Biquad<T>` | TDF-II biquad with 9 coefficient types and lock-free auto-promote of staged coefficients |
| `BiquadCoeffs<T>` | Standalone factory for biquad coefficients (LP, HP, BP, Peak, Shelf, Notch, AP, Tilt, DC blocker) |
| `FFTComplex<T>` / `FFTReal<T>` | Radix-2 FFT with SIMD (SSE3/NEON), real-optimised |
| `Convolver<T>` | Partitioned overlap-save FFT convolution |
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
| `SpinLock` | RT-safe spinlock for thread-safe parameters |
| `SpscQueue<T>` | Lock-free single-producer / single-consumer queue |
| `AudioSpec` | Audio environment descriptor (sample rate, block size, channels) |
| `AudioBuffer<T>` | 32-byte aligned owning buffer (SIMD-ready) |
| `AudioBufferView<T>` | Non-owning view (what processors receive) |
| `SimdOps` | SIMD-accelerated buffer operations (SSE2/AVX/NEON with scalar fallback) |
| `DspMath` | Constants, dB ⇄ gain, fast tanh / exp / pow10, range mapping |
| `Phasor<T>` | Phase accumulator for LFO and oscillator construction |
| `SampleAndHold<T>` | Sample-and-hold with configurable hold time |
| `WaveshapeTable<T>` | LUT-based waveshaping with linear / cubic interpolation |
| `AnalogRandom` | Analog-flavoured random generators (smooth, noise, jitter) |
| `ProcessorTraits` | C++20 concepts: `AudioProcessor`, `SampleProcessor`, `GeneratorProcessor` |

### Analysis (5 analyzers)

| Class | Description |
|---|---|
| `LevelFollower<T>` | Peak and RMS envelope follower |
| `SpectrumAnalyzer<T>` | Real-time FFT spectrum with peak hold |
| `LoudnessMeter<T>` | EBU R128 LUFS (momentary, short-term, integrated) |
| `Goertzel<T>` | Single-frequency O(N) magnitude detection |
| `PitchDetector<T>` | Autocorrelation-based fundamental frequency detection |

### I/O (3 file handlers)

| Class | Description |
|---|---|
| `WavFile` | Read/write WAV (PCM 8/16/24/32-bit, float 32/64-bit) |
| `Mp3File` | MPEG-1 Layer III codec — read (CBR/VBR) + write (CBR encoder, 32–320 kbps) |
| `AudioFile` | Abstract base class for custom format implementations |

### Music (1 module)

| Class | Description |
|---|---|
| `HarmonyConstants` | Constexpr musical harmony toolkit: 61 scales (bitmask representation), 15 chord recipes with inversions, MIDI/note conversion, key-aware naming (sharp/flat), diatonic chord generation. Fully `constexpr`/`consteval` — generates static tables at compile time. |

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
- 28 effects organised by category (Filters, Dynamics, Distortion, Modulation, Spatial, Utility) — including Dynamic EQ, Multiband Compressor and Convolution Reverb
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
- **Full Doxygen documentation**: Every public class and method documented

---

## Architecture

```
DSPark/
├── DSPark.h                 # Single umbrella include + full documentation
├── Core/          (33)      # Building blocks: filters, FFT, oscillators, SIMD, buffers
├── Effects/       (30)      # Ready-to-use processors: EQ, compressor, reverb...
├── Analysis/       (5)      # Metering: LUFS, spectrum, level follower, pitch
├── IO/             (3)      # File I/O: WAV read/write, MP3 read/write
├── Music/          (1)      # Harmony constants and music theory
└── DSParkLab/               # Interactive testing app (Win32 + ImGui + miniaudio)
```

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
