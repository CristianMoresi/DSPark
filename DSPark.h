// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DSPark.h
 * @brief Single-include umbrella header for the DSPark framework.
 *
 * A complete, standalone audio DSP framework in pure C++20. Zero external
 * dependencies — only the C++ standard library. Works on any platform:
 * Windows, macOS, Linux, WebAssembly, iOS, Android.
 *
 * ```cpp
 * #include "DSPark/DSPark.h"
 * ```
 *
 * @tableofcontents
 *
 * ---
 *
 * @section integration How to Add This Framework to Your Project
 *
 * This is a **header-only** library. No compilation, no linking, no build system
 * required. Just add the `DSPark/` folder to your project's include path.
 *
 * @subsection integration_vs Visual Studio (Windows Forms, WPF, Console, etc.)
 *
 * 1. Copy the `DSPark/` folder into your project directory (e.g., `MyApp/libs/DSPark/`).
 * 2. In Visual Studio: Project → Properties → C/C++ → Additional Include Directories →
 *    add the **parent** directory of `DSPark/` (e.g., `$(ProjectDir)libs`).
 * 3. Set the C++ standard to C++20: C/C++ → Language → C++ Language Standard → ISO C++20.
 * 4. In your source files:
 *    ```cpp
 *    #include "DSPark/DSPark.h"
 *    // All classes are now available under the dspark:: namespace
 *    ```
 *
 * @subsection integration_cmake CMake
 *
 * ```cmake
 * # In your CMakeLists.txt:
 * target_include_directories(MyApp PRIVATE ${CMAKE_SOURCE_DIR}/libs)
 * target_compile_features(MyApp PRIVATE cxx_std_20)
 * ```
 *
 * @subsection integration_wasm WebAssembly (Emscripten)
 *
 * ```bash
 * em++ -std=c++20 -O2 -I./libs my_processor.cpp -o processor.js
 * ```
 * The framework uses no platform-specific APIs — it compiles directly with
 * Emscripten. Pair with the Web Audio API's AudioWorklet for real-time processing.
 *
 * @subsection integration_vst3 VST3 Plugin (DAW)
 *
 * Add the `DSPark/` include path to your VST3 project. In your processor class:
 * ```cpp
 * #include "DSPark/DSPark.h"
 *
 * class MyProcessor : public Steinberg::Vst::AudioEffect {
 *     dspark::FilterEngine<float> filter_;
 *     dspark::Saturation<float> saturator_;
 *
 *     // In setupProcessing():
 *     //   dspark::AudioSpec spec { sampleRate, maxBlockSize, numChannels };
 *     //   filter_.prepare(spec);
 *     //   saturator_.prepare(spec);
 *
 *     // In process():
 *     //   dspark::AudioBufferView<float> view(outputs[0], numChannels, numSamples);
 *     //   saturator_.process(view);
 *     //   filter_.processBlock(view);
 * };
 * ```
 *
 * @subsection integration_juce JUCE Plugin
 *
 * ```cpp
 * #include "DSPark/DSPark.h"
 *
 * void MyPlugin::prepareToPlay(double sampleRate, int maxBlockSize) {
 *     dspark::AudioSpec spec { sampleRate, maxBlockSize, getTotalNumOutputChannels() };
 *     myFilter_.prepare(spec);
 * }
 *
 * void MyPlugin::processBlock(juce::AudioBuffer<float>& buffer, ...) {
 *     // Wrap JUCE buffer in a DSPark view (zero-copy):
 *     dspark::AudioBufferView<float> view(
 *         buffer.getArrayOfWritePointers(),
 *         buffer.getNumChannels(),
 *         buffer.getNumSamples());
 *     myFilter_.processBlock(view);
 * }
 * ```
 *
 * ---
 *
 * @section lifecycle Processor Lifecycle
 *
 * Every processor in this framework follows the same three-step pattern:
 *
 * 1. **Create** — Construct the processor (stack or heap, your choice).
 * 2. **Prepare** — Call `prepare(AudioSpec)` once before processing. This is the
 *    only step that may allocate memory. Call again if sample rate or block size changes.
 * 3. **Process** — Call `process()` / `processBlock()` / `processSample()` in your
 *    audio callback. These methods are real-time safe (zero allocations, no locks).
 *
 * ```
 * ┌──────────┐     ┌───────────────────┐     ┌────────────────────────────┐
 * │  Create  │ ──> │  prepare(spec)    │ ──> │  process(buffer) [repeat]  │
 * │          │     │  (allocates once) │     │  (real-time safe)          │
 * └──────────┘     └───────────────────┘     └────────────────────────────┘
 * ```
 *
 * ---
 *
 * @section concepts Key Concepts
 *
 * @subsection concept_spec AudioSpec — Describe Your Audio Environment
 *
 * Before processing, you tell each processor about your audio setup:
 *
 * ```cpp
 * dspark::AudioSpec spec {
 *     .sampleRate   = 48000.0,  // Hz (44100, 48000, 96000, etc.)
 *     .maxBlockSize = 512,      // Maximum samples per process() call
 *     .numChannels  = 2         // 1 = mono, 2 = stereo
 * };
 * ```
 *
 * @subsection concept_buffer AudioBuffer / AudioBufferView — Carry Audio Data
 *
 * - **AudioBufferView\<T\>**: A lightweight, non-owning wrapper around existing audio
 *   data (e.g., your audio driver's buffers, JUCE's AudioBuffer, or raw float**).
 *   This is what processors receive. Cheap to create, zero-copy.
 *
 * - **AudioBuffer\<T\>**: An owning buffer that manages its own memory. Allocates once
 *   in `resize()`, 32-byte aligned for SIMD. Use this for internal storage.
 *
 * ```cpp
 * // Wrapping existing raw pointers (e.g., from your audio driver):
 * float* left  = ...;
 * float* right = ...;
 * float* channels[] = { left, right };
 * dspark::AudioBufferView<float> view(channels, 2, 512);
 *
 * // Or allocating your own buffer:
 * dspark::AudioBuffer<float> buffer;
 * buffer.resize(2, 512);                    // Allocates (call in setup, not in callback)
 * dspark::AudioBufferView<float> view = buffer.toView();  // Zero-copy view
 * ```
 *
 * ---
 *
 * @section example_realtime Example: Real-Time Audio Processing
 *
 * This example shows how to set up a processing chain in a real-time audio
 * application (Windows Forms, Qt, standalone app, etc.). Your audio driver
 * (WASAPI, CoreAudio, ALSA, etc.) calls your callback with raw float pointers.
 *
 * ```cpp
 * #include "DSPark/DSPark.h"
 *
 * // --- Global processors (create once) ---
 * dspark::FilterEngine<float> highpass;
 * dspark::Saturation<float>   saturator;
 * dspark::FilterEngine<float> lowpass;
 * dspark::LevelFollower<float> meter;
 *
 * // --- Called once at startup ---
 * void setupAudio(double sampleRate, int blockSize)
 * {
 *     dspark::AudioSpec spec { sampleRate, blockSize, 2 };
 *
 *     highpass.prepare(spec);
 *     highpass.setHighPass(80.0f);           // Remove rumble below 80 Hz
 *
 *     saturator.prepare(spec);
 *     saturator.setAlgorithm(dspark::Saturation<float>::Algorithm::Tape);
 *     saturator.setDrive(6.0f);             // 6 dB of tape warmth
 *
 *     lowpass.prepare(spec);
 *     lowpass.setLowPass(12000.0f, 0.707f, 12);  // Gentle 12 dB/oct rolloff
 *
 *     meter.prepare(spec);
 *     meter.setAttackMs(5.0f);
 *     meter.setReleaseMs(100.0f);
 * }
 *
 * // --- Called by audio driver (real-time thread, ~every 5-10 ms) ---
 * void audioCallback(float** channelData, int numChannels, int numSamples)
 * {
 *     dspark::AudioBufferView<float> buffer(channelData, numChannels, numSamples);
 *
 *     highpass.processBlock(buffer);         // Pre-filter
 *     saturator.process(buffer);             // Distortion
 *     lowpass.processBlock(buffer);          // Post-filter
 *     meter.process(buffer.toView());        // Metering (non-destructive)
 *
 *     // Read levels for your GUI (thread-safe):
 *     // float peakL = meter.getPeakLevelDb(0);
 *     // float peakR = meter.getPeakLevelDb(1);
 * }
 * ```
 *
 * @section example_offline Example: Offline File Processing
 *
 * Load a WAV file, process it, and save the result. No audio driver needed.
 *
 * ```cpp
 * #include "DSPark/DSPark.h"
 * #include <cstdio>
 *
 * int main()
 * {
 *     // 1. Read input file
 *     dspark::WavFile reader;
 *     if (!reader.openRead("input.wav")) {
 *         std::printf("Failed to open input.wav\n");
 *         return 1;
 *     }
 *     auto info = reader.getInfo();
 *     std::printf("Loaded: %d ch, %.0f Hz, %lld samples\n",
 *                 info.numChannels, info.sampleRate, info.numSamples);
 *
 *     // 2. Load into buffer
 *     dspark::AudioBuffer<float> buffer;
 *     buffer.resize(info.numChannels, static_cast<int>(info.numSamples));
 *     reader.readSamples(buffer.toView());
 *     reader.close();
 *
 *     // 3. Process
 *     dspark::AudioSpec spec = info.toSpec();
 *
 *     dspark::FilterEngine<float> filter;
 *     filter.prepare(spec);
 *     filter.setHighPass(80.0f, 0.707f, 24);    // 24 dB/oct high-pass at 80 Hz
 *     filter.processBlock(buffer.toView());
 *
 *     dspark::Saturation<float> sat;
 *     sat.prepare(spec);
 *     sat.setAlgorithm(dspark::Saturation<float>::Algorithm::Tape);
 *     sat.setDrive(3.0f);
 *     sat.process(buffer.toView());
 *
 *     // 4. Write output file
 *     dspark::WavFile writer;
 *     if (!writer.openWrite("output.wav", info)) {
 *         std::printf("Failed to create output.wav\n");
 *         return 1;
 *     }
 *     writer.writeSamples(buffer.toView());
 *     writer.close();
 *
 *     std::printf("Done! Wrote output.wav\n");
 *     return 0;
 * }
 * ```
 *
 * @section example_mp3_export Example: Export to MP3
 *
 * Convert a WAV file to MP3 with processing applied:
 *
 * ```cpp
 * #include "DSPark/DSPark.h"
 *
 * int main()
 * {
 *     dspark::WavFile reader;
 *     reader.openRead("input.wav");
 *     auto info = reader.getInfo();
 *
 *     dspark::AudioBuffer<float> buffer;
 *     buffer.resize(info.numChannels, static_cast<int>(info.numSamples));
 *     reader.readSamples(buffer.toView());
 *     reader.close();
 *
 *     // Process...
 *     dspark::AudioSpec spec = info.toSpec();
 *     dspark::Limiter<float> limiter;
 *     limiter.prepare(spec);
 *     limiter.setCeiling(-1.0f);
 *     limiter.processBlock(buffer.toView());
 *
 *     // Export as MP3 at 320 kbps
 *     dspark::AudioFileInfo mp3Info;
 *     mp3Info.sampleRate   = info.sampleRate;
 *     mp3Info.numChannels  = info.numChannels;
 *     mp3Info.bitsPerSample = 320;  // Bitrate in kbps for MP3
 *
 *     dspark::Mp3File mp3;
 *     mp3.openWrite("output.mp3", mp3Info);
 *     mp3.writeSamples(buffer.toView());
 *     mp3.close();
 *
 *     return 0;
 * }
 * ```
 *
 * @section example_large_file Example: Processing Large Files in Chunks
 *
 * For files too large to fit in memory, process in blocks (streaming):
 *
 * ```cpp
 * dspark::WavFile reader, writer;
 * reader.openRead("large_file.wav");
 * auto info = reader.getInfo();
 *
 * writer.openWrite("output.wav", info);
 *
 * constexpr int kBlockSize = 4096;
 * dspark::AudioBuffer<float> block;
 * block.resize(info.numChannels, kBlockSize);
 *
 * dspark::AudioSpec spec { info.sampleRate, kBlockSize, info.numChannels };
 * dspark::FilterEngine<float> filter;
 * filter.prepare(spec);
 * filter.setLowPass(8000.0f);
 *
 * int64_t remaining = info.numSamples;
 * int64_t offset = 0;
 * while (remaining > 0)
 * {
 *     int toRead = static_cast<int>(std::min(remaining, int64_t(kBlockSize)));
 *     auto view = block.toView().getSubView(0, toRead);
 *
 *     reader.readSamples(view, offset, toRead);
 *     filter.processBlock(view);
 *     writer.writeSamples(view);
 *
 *     offset += toRead;
 *     remaining -= toRead;
 * }
 *
 * reader.close();
 * writer.close();
 * ```
 *
 * ---
 *
 * @section classes_overview Available Classes
 *
 * @subsection classes_processors Audio Processors
 *
 * | Class                    | Header                | Purpose                                         |
 * |--------------------------|-----------------------|-------------------------------------------------|
 * | `Equalizer<T>`           | Effects/Equalizer.h   | Multi-band parametric EQ (Peak/Shelf/LP/HP/Notch, 1-16 bands)       |
 * | `FilterEngine<T>`        | Effects/Filters.h     | Multi-mode cascaded filter (LP/HP/BP/Peak/Shelf/Notch, 6-48 dB/oct) |
 * | `Saturation<T>`          | Effects/Saturation.h  | 10 saturation algorithms (tube, tape, transformer, wavefolder...)    |
 * | `Delay<T>`               | Effects/Delay.h       | Delay line with interpolation, feedback filters, ping-pong           |
 * | `Reverb<T>`              | Effects/Reverb.h      | Convolution reverb with IR loading, pre-delay, dry/wet               |
 * | `Chorus<T>`              | Effects/Chorus.h      | Chorus/flanger with multi-voice LFO, stereo spread                   |
 * | `Phaser<T>`              | Effects/Phaser.h      | Allpass phaser with configurable stages, feedback, LFO               |
 * | `Panner<T>`              | Effects/Panner.h      | 6 stereo panning algorithms (equal-power, binaural, Haas, spectral)  |
 * | `MidSide<T>`             | Effects/MidSide.h     | Stereo Mid/Side encoding and decoding                                |
 * | `Gain<T>`                | Effects/Gain.h        | Smoothed gain with fade, mute, polarity inversion                    |
 * | `DCBlocker<T>`           | Effects/DCBlocker.h   | DC offset removal (1-pole or Butterworth order 2-10)                |
 * | `Crossfade<T>`           | Effects/Crossfade.h   | Crossfade with linear, equal-power, S-curve                         |
 * | `StereoWidth<T>`         | Effects/StereoWidth.h | Stereo width via M/S with bass-mono option                          |
 * | `Compressor<T>`          | Effects/Compressor.h  | Modular compressor (3 detectors, 2 topologies, 4 characters, ext. sidechain) |
 * | `Limiter<T>`             | Effects/Limiter.h     | ISP true-peak brickwall limiter with adaptive release                |
 * | `NoiseGate<T>`           | Effects/NoiseGate.h   | Noise gate with hysteresis, hold, duck mode, ext. sidechain         |
 * | `Expander<T>`            | Effects/Expander.h    | Downward expander with ratio, hysteresis, ext. sidechain            |
 * | `CrossoverFilter<T>`     | Effects/CrossoverFilter.h | Linkwitz-Riley crossover (2-12 bands, LR12/24/48, IIR + linear-phase) |
 * | `MultibandCompressor<T>` | Effects/MultibandCompressor.h | Multi-band compressor (crossover split + per-band Compressor) |
 * | `DynamicEQ<T>`           | Effects/DynamicEQ.h   | Per-band dynamic EQ (above/below threshold, ext. sidechain)        |
 * | `TransientDesigner<T>`   | Effects/TransientDesigner.h | Dual-envelope transient shaper (attack/sustain control)       |
 * | `AlgorithmicReverb<T>`   | Effects/AlgorithmicReverb.h | FDN reverb: Room/Hall/Chamber/Plate/Spring/Cathedral presets            |
 * | `NoiseGenerator<T>`      | Effects/NoiseGenerator.h | White/pink/brown noise generator (AudioProcessor contract)        |
 * | `Tremolo<T>`             | Effects/Tremolo.h     | LFO amplitude modulation with stereo auto-pan option                |
 * | `Vibrato<T>`             | Effects/Vibrato.h     | Pitch modulation via LFO-driven delay line                          |
 * | `RingModulator<T>`       | Effects/RingModulator.h | Ring modulation (signal × carrier) with mix control               |
 * | `FrequencyShifter<T>`    | Effects/FrequencyShifter.h | Constant-Hz frequency shift via Hilbert transform              |
 * | `DeEsser<T>`             | Effects/DeEsser.h     | Split-band de-esser with dynamic sibilance detection                |
 * | `AutoGain<T>`            | Effects/AutoGain.h    | Automatic gain compensation for honest A/B comparison               |
 * | `Clipper<T>`             | Effects/Clipper.h     | Multi-mode clipper (Hard/Soft/Analog/GoldenRatio, oversampling)     |
 * | `LadderFilter<T>`        | Core/LadderFilter.h   | Moog-style 4-pole resonant filter (TPT, 6 modes, drive)            |
 * | `StateVariableFilter<T>` | Core/StateVariableFilter.h | TPT SVF: LP/HP/BP/Notch/AP/Bell/Shelf, multi-output, mod-friendly |
 * | `Oversampling<T>`        | Core/Oversampling.h   | 2x-16x oversampling with FIR half-band Kaiser filters (-80 dB+)   |
 * | `Oscillator<T>`          | Core/Oscillator.h     | PolyBLEP oscillator (sine, saw, square, triangle)                   |
 * | `WavetableOscillator<T>` | Core/WavetableOscillator.h | Mipmapped wavetable oscillator (bandlimited)                   |
 * | `DryWetMixer<T>`         | Core/DryWetMixer.h    | Dry/wet parallel mix for effects                                    |
 *
 * @subsection classes_analysis Analysis & Metering
 *
 * | Class                    | Header                    | Purpose                                      |
 * |--------------------------|---------------------------|----------------------------------------------|
 * | `LevelFollower<T>`       | Analysis/LevelFollower.h  | Peak and RMS envelope follower               |
 * | `SpectrumAnalyzer<T>`    | Analysis/SpectrumAnalyzer.h | Real-time FFT spectrum analyser with peak hold |
 * | `LoudnessMeter<T>`       | Analysis/LoudnessMeter.h  | EBU R128 LUFS metering (momentary/short/integrated) |
 * | `Goertzel<T>`            | Analysis/Goertzel.h       | Single-frequency O(N) magnitude detection    |
 * | `PitchDetector<T>`       | Analysis/PitchDetector.h  | YIN monophonic pitch detection with MIDI/cents output |
 *
 * @subsection classes_io File I/O
 *
 * | Class                    | Header           | Purpose                                         |
 * |--------------------------|------------------|-------------------------------------------------|
 * | `WavFile`                | IO/WavFile.h     | Read/write WAV files (PCM 8/16/24/32, float 32/64) |
 * | `Mp3File`                | IO/Mp3File.h     | MPEG-1 Layer III codec — read (CBR/VBR) + write (CBR encoder) |
 *
 * @subsection classes_music Music Theory
 *
 * | Class                    | Header                    | Purpose                                      |
 * |--------------------------|---------------------------|----------------------------------------------|
 * | `harmony::*`             | Music/HarmonyConstants.h  | 61 scales, 15 chords, MIDI, note naming, diatonic generation |
 *
 * @subsection classes_core Core Building Blocks
 *
 * | Class                    | Header                    | Purpose                                        |
 * |--------------------------|---------------------------|-------------------------------------------------|
 * | `AudioBuffer<T>`         | Core/AudioBuffer.h        | Owning audio buffer (32-byte aligned, SIMD-ready) |
 * | `AudioBufferView<T>`     | Core/AudioBuffer.h        | Non-owning view -- what processors receive        |
 * | `AudioSpec`              | Core/AudioSpec.h          | Audio environment descriptor (rate, block, channels) |
 * | `Biquad<T>`              | Core/Biquad.h             | Single biquad filter with 9 coefficient types     |
 * | `BiquadCoeffs<T>`        | Core/Biquad.h             | Filter coefficient calculator (Audio EQ Cookbook)  |
 * | `FFTReal<T>`             | Core/FFT.h                | Radix-2 FFT with SIMD (SSE2/NEON), real-optimised |
 * | `FIRFilter<T>`           | Core/FIRFilter.h          | FIR filter with windowed-sinc design              |
 * | `Convolver<T>`           | Core/Convolver.h          | Partitioned overlap-save FFT convolution          |
 * | `Resampler<T>`           | Core/Resampler.h          | Polyphase windowed-sinc sample rate converter     |
 * | `WindowFunctions<T>`     | Core/WindowFunctions.h    | 8 window functions (Hann, Kaiser, Blackman...)    |
 * | `Smoothers`              | Core/Smoothers.h          | 9 parameter smoothing algorithms                  |
 * | `EnvelopeGenerator<T>`   | Core/EnvelopeGenerator.h  | ADSR envelope for synthesis and dynamics          |
 * | `Dither<T>`              | Core/Dither.h             | TPDF dithering with noise shaping                 |
 * | `DenormalGuard`          | Core/DenormalGuard.h      | RAII denormal flush (SSE FTZ/DAZ, ARM FZ)         |
 * | `Interpolation`          | Core/Interpolation.h      | 5 interpolation methods (linear to Lagrange)      |
 * | `Phasor<T>`              | Core/Phasor.h             | Phase accumulator for oscillators/LFOs            |
 * | `RingBuffer<T>`          | Core/RingBuffer.h         | Power-of-two circular buffer with interp. read    |
 * | `SampleAndHold<T>`       | Core/SampleAndHold.h      | S&H for bit-crush and stepped modulation          |
 * | `WaveshapeTable<T>`      | Core/WaveshapeTable.h     | Table-lookup waveshaper with presets              |
 * | `Hilbert<T>`             | Core/Hilbert.h            | Allpass Hilbert transform for analytic signals    |
 * | `TruePeakDetector<T>`    | Core/TruePeakDetector.h   | Shared ITU-R BS.1770-4 inter-sample peak detector |
 * | `AnalogConstants`        | Core/AnalogConstants.h    | Analog-hardware reference constants (zero runtime cost) |
 * | `SpinLock`               | Core/SpinLock.h           | RT-safe spinlock for thread-safe parameters       |
 * | `SpscQueue<T>`           | Core/SpscQueue.h          | Lock-free single-producer/single-consumer queue   |
 * | `AnalogRandom`           | Core/AnalogRandom.h       | Analog-style noise (white/pink/brown)             |
 * | `SmoothedValue<T>`       | Core/SmoothedValue.h      | Parameter smoother (exponential, linear, chase, disabled)  |
 * | `ProcessorChain<T,...>`  | Core/ProcessorChain.h     | Zero-overhead compile-time processor chain with per-slot bypass |
 * | `SpectralProcessor<T>`   | Core/SpectralProcessor.h  | STFT analysis-modification-synthesis with user callback |
 * | `DspMath`                | Core/DspMath.h            | dB/gain conversions, fastTanh, constants          |
 * | `SimdOps`                | Core/SimdOps.h            | SIMD-accelerated buffer ops (SSE2/AVX/NEON/scalar)|
 *
 * ---
 *
 * @section design Design Principles
 *
 * - **Zero external dependencies** — C++20 standard library only.
 * - **Real-time safe** — No allocations, no locks, no syscalls in process().
 * - **Thread-safe** — All parameter setters use `std::atomic` with `memory_order_relaxed`, callable from any thread with zero contention.
 * - **Cache-friendly** — Contiguous memory, 32-byte aligned buffers (SIMD-ready).
 * - **Multiplatform** — Windows, macOS, Linux, WebAssembly, iOS, Android.
 * - **Header-only** — No build system, no compilation, just `#include`.
 *
 * @note All classes live in the `dspark` namespace.
 *
 * ---
 *
 * @section api_contract API Contract (C++20 Concepts)
 *
 * The framework defines C++20 concepts in `ProcessorTraits.h` that formalise
 * the processor interface. These are compile-time contracts — zero overhead,
 * no vtable, clear error messages if you forget a method.
 *
 * ```cpp
 * // Any type that has prepare(AudioSpec), processBlock(AudioBufferView<T>), reset()
 * template <typename P, typename T>
 * concept AudioProcessor = ...;
 *
 * // AudioProcessor + processSample(T, int) -> T
 * template <typename P, typename T>
 * concept SampleProcessor = ...;
 *
 * // Generators: prepare(AudioSpec), reset(), getSample() -> T
 * template <typename P, typename T>
 * concept GeneratorProcessor = ...;
 * ```
 *
 * All processors in this framework satisfy `AudioProcessor`. You can use these
 * concepts in your own code to write generic functions or verify your classes:
 *
 * ```cpp
 * static_assert(dspark::AudioProcessor<dspark::Compressor<float>, float>);
 * static_assert(dspark::AudioProcessor<dspark::Gain<float>, float>);
 *
 * template <dspark::AudioProcessor<float> Proc>
 * void applyEffect(Proc& proc, dspark::AudioBufferView<float> buf) {
 *     proc.processBlock(buf);
 * }
 * ```
 *
 * @section processor_chain ProcessorChain — Compose Processors at Compile Time
 *
 * `ProcessorChain` lets you combine multiple processors into a single unit
 * with zero runtime overhead. All dispatch is resolved at compile time via
 * `std::tuple` and fold expressions.
 *
 * ```cpp
 * // Define a chain: high-pass → saturation drive → compressor → output gain
 * dspark::ProcessorChain<float,
 *     dspark::FilterEngine<float>,
 *     dspark::Compressor<float>,
 *     dspark::Gain<float>> chain;
 *
 * // Prepare all processors at once
 * dspark::AudioSpec spec { 48000.0, 512, 2 };
 * chain.prepare(spec);
 *
 * // Configure individual processors by index
 * chain.get<0>().setHighPass(80.0f, 0.707f, 12);
 * chain.get<1>().setThreshold(-20.0f);
 * chain.get<1>().setRatio(4.0f);
 * chain.get<2>().setGainDb(-3.0f);
 *
 * // In audio callback: process all in order
 * chain.processBlock(buffer);
 *
 * // Bypass individual slots at runtime (zero-cost when not bypassed)
 * chain.setBypassed<1>(true);   // Skip compressor
 * chain.processBlock(buffer);   // Only filter + gain run
 *
 * // Reset all at once
 * chain.reset();
 * ```
 *
 * @section oversampling Integrated Oversampling
 *
 * `Saturation` and `WaveshapeTable` support integrated oversampling to reduce
 * aliasing from harmonic generation. Just call `setOversampling()`:
 *
 * ```cpp
 * dspark::Saturation<float> sat;
 * sat.setOversampling(4);  // 4x oversampling
 * sat.prepare(spec);
 * sat.process(buffer);     // Automatically upsamples → saturates → downsamples
 *
 * dspark::WaveshapeTable<float> shaper;
 * shaper.buildTanh(3.0f);
 * shaper.prepare(spec);
 * shaper.setOversampling(4);
 * shaper.processBlock(buffer);  // Same: upsample → shape → downsample
 * ```
 *
 * @section progressive_disclosure Progressive Disclosure API
 *
 * Every processor uses a **single API** with three levels of depth:
 *
 * - **Level 1 (desktop developer):** Use basic setters with sensible defaults.
 *   No DSP knowledge required.
 *   ```cpp
 *   eq.setBand(0, 200.0f, -3.0f);           // Just frequency and gain
 *   comp.setThreshold(-20.0f);               // Just threshold
 *   chorus.setRate(1.5f);                    // Just speed
 *   ```
 *
 * - **Level 2 (audio developer):** Add extra parameters for fine control.
 *   ```cpp
 *   eq.setBand(0, 200.0f, -3.0f, 1.5f);     // + Q factor
 *   comp.setKnee(6.0f);                      // + soft knee
 *   comp.setMix(0.5f);                       // + parallel compression
 *   chorus.setVoices(3);                     // + voice count
 *   chorus.setFeedback(-0.7f);               // + flanger mode
 *   ```
 *
 * - **Level 3 (DSP engineer):** Full expert control over every internal.
 *   ```cpp
 *   eq.setBand(0, { .frequency=200, .gain=-3, .q=1.5,
 *                    .type=BandType::LowShelf, .slope=24 });
 *   comp.setDetector(DetectorType::TruePeak);
 *   comp.setTopology(Topology::FeedBack);
 *   comp.setCharacter(Character::Opto);
 *   comp.setLookahead(5.0f);
 *   limiter.setTruePeak(true);
 *   limiter.setAdaptiveRelease(true);
 *   ```
 *
 * @section glossary DSP Glossary
 *
 * Quick reference for common DSP terms used throughout this framework:
 *
 * | Term              | Meaning                                                        |
 * |-------------------|----------------------------------------------------------------|
 * | **Sample rate**   | Number of audio samples per second (e.g., 44100, 48000 Hz).   |
 * | **dB (decibel)**  | Logarithmic unit for loudness. +6 dB = double amplitude.       |
 * | **dBFS**          | Decibels relative to full scale. 0 dBFS = max digital level.   |
 * | **dBTP**          | Decibels True Peak — accounts for inter-sample peaks.          |
 * | **Gain**          | Amplitude multiplier. 1.0 = unity (no change).                 |
 * | **Q factor**      | Filter bandwidth. Low Q = wide, high Q = narrow/resonant.      |
 * | **Nyquist**       | Maximum representable frequency = sampleRate / 2.              |
 * | **FFT**           | Fast Fourier Transform — converts time domain to frequency.    |
 * | **Biquad**        | 2nd-order IIR filter — the workhorse of audio EQ.              |
 * | **IIR / FIR**     | Infinite/Finite Impulse Response — two fundamental filter types.|
 * | **Oversampling**  | Processing at a higher sample rate to reduce aliasing.          |
 * | **Aliasing**      | Distortion from frequencies above Nyquist folding back.        |
 * | **Threshold**     | Level (dB) above which a compressor/limiter begins acting.     |
 * | **Ratio**         | Compression ratio. 4:1 means 4 dB input over threshold = 1 dB output. |
 * | **Attack**        | Time for compressor to reach full compression after threshold.  |
 * | **Release**       | Time for compressor to return to unity after signal drops.      |
 * | **Knee**          | Transition region around threshold. Soft knee = gradual onset.  |
 * | **Makeup gain**   | Gain applied after compression to restore perceived loudness.   |
 * | **Lookahead**     | Delay allowing the processor to "see" transients in advance.    |
 * | **ISP**           | Inter-Sample Peak — peak that occurs between digital samples.   |
 * | **LUFS**          | Loudness Unit Full Scale — perceptual loudness measurement.     |
 * | **Dry/Wet**       | Unprocessed/processed signal. Mix 50/50 = parallel processing.  |
 * | **LFO**           | Low Frequency Oscillator — modulation source (0.1-20 Hz).      |
 * | **IR**            | Impulse Response — acoustic fingerprint of a space or device.   |
 * | **Denormal**      | Tiny float values that cause CPU spikes in IIR filters.         |
 * | **PolyBLEP**      | Polynomial Band-Limited Step — anti-aliasing for oscillators.   |
 * | **M/S**           | Mid/Side encoding. Mid = (L+R)/2, Side = (L-R)/2.              |
 * | **Allpass**       | Filter that changes phase without affecting magnitude.           |
 *
 */

// === Core ===================================================================

#include "Core/SimdOps.h"
#include "Core/DspMath.h"
#include "Core/AudioSpec.h"
#include "Core/AudioBuffer.h"
#include "Core/AnalogConstants.h"
#include "Core/TruePeakDetector.h"
#include "Core/SpinLock.h"
#include "Core/SpscQueue.h"
#include "Core/Biquad.h"
#include "Core/DryWetMixer.h"
#include "Core/Smoothers.h"
#include "Core/AnalogRandom.h"
#include "Core/Oscillator.h"
#include "Core/Oversampling.h"
#include "Core/FFT.h"
#include "Core/WindowFunctions.h"
#include "Core/FIRFilter.h"
#include "Core/Convolver.h"
#include "Core/Resampler.h"
#include "Core/EnvelopeGenerator.h"
#include "Core/Dither.h"
#include "Core/SmoothedValue.h"
#include "Core/ProcessorTraits.h"
#include "Core/ProcessorChain.h"
#include "Core/DenormalGuard.h"
#include "Core/Interpolation.h"
#include "Core/Phasor.h"
#include "Core/SampleAndHold.h"
#include "Core/RingBuffer.h"
#include "Core/WaveshapeTable.h"
#include "Core/WavetableOscillator.h"
#include "Core/Hilbert.h"
#include "Core/LadderFilter.h"
#include "Core/StateVariableFilter.h"
#include "Core/SpectralProcessor.h"

// === Effects ================================================================

#include "Effects/MidSide.h"
#include "Effects/Saturation.h"
#include "Effects/Delay.h"
#include "Effects/Filters.h"
#include "Effects/Panner.h"
#include "Effects/Gain.h"
#include "Effects/DCBlocker.h"
#include "Effects/Crossfade.h"
#include "Effects/StereoWidth.h"
#include "Effects/Compressor.h"
#include "Effects/Limiter.h"
#include "Effects/NoiseGate.h"
#include "Effects/Equalizer.h"
#include "Effects/Reverb.h"
#include "Effects/Chorus.h"
#include "Effects/Phaser.h"
#include "Effects/AlgorithmicReverb.h"
#include "Effects/NoiseGenerator.h"
#include "Effects/Tremolo.h"
#include "Effects/Vibrato.h"
#include "Effects/RingModulator.h"
#include "Effects/FrequencyShifter.h"
#include "Effects/DeEsser.h"
#include "Effects/AutoGain.h"
#include "Effects/CrossoverFilter.h"
#include "Effects/Expander.h"
#include "Effects/TransientDesigner.h"
#include "Effects/DynamicEQ.h"
#include "Effects/MultibandCompressor.h"
#include "Effects/Clipper.h"

// === Analysis ===============================================================

#include "Analysis/LevelFollower.h"
#include "Analysis/SpectrumAnalyzer.h"
#include "Analysis/LoudnessMeter.h"
#include "Analysis/Goertzel.h"
#include "Analysis/PitchDetector.h"

// === I/O ====================================================================

#include "IO/AudioFile.h"
#include "IO/WavFile.h"
#include "IO/Mp3File.h"

// === Music ==================================================================

#include "Music/HarmonyConstants.h"
