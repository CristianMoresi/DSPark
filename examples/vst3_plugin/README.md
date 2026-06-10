# Minimal VST3 plugin template (no JUCE)

A complete Steinberg VST3 plugin around a DSPark chain, using only the
official VST3 SDK — which stays an EXTERNAL dependency of this example,
never of the DSPark core.

## Build

1. Get the SDK: <https://github.com/steinbergmedia/vst3sdk> (clone with
   submodules; accept Steinberg's license).
2. Configure pointing at it:

```bash
cmake -B build -DVST3_SDK_DIR=/path/to/vst3sdk
cmake --build build --config Release
```

Without `VST3_SDK_DIR` the example is skipped entirely (the rest of DSPark
builds and tests as usual).

## What the template shows

- `plugin.cpp` — one `AudioEffect` subclass owning a DSPark chain
  (`TubePreamp` → `TapeMachine` → `Limiter`), the AudioSpec lifecycle in
  `setupProcessing()`, the zero-copy `AudioBufferView` wrap in `process()`,
  thread-safe parameter forwarding from the VST3 parameter queue to DSPark's
  atomic setters, and latency reporting through `getLatencySamples()`
  (DSPark processors report theirs via `getLatency()`).
- `version.h` / `entry.cpp` — boilerplate class factory and UIDs. Replace
  the GUIDs before shipping anything.

DSPark side, in short:

```cpp
void Processor::setupProcessing(Vst::ProcessSetup& s)
{
    dspark::AudioSpec spec { s.sampleRate, s.maxSamplesPerBlock, 2 };
    preamp_.prepare(spec);
    tape_.prepare(spec);
    limiter_.prepare(spec);
    setLatencySamples(static_cast<Steinberg::uint32>(
        preamp_.getLatency() + tape_.getLatency() + limiter_.getLatency()));
}

tresult Processor::process(Vst::ProcessData& data)
{
    // forward parameter changes -> dspark atomic setters (thread-safe)
    dspark::AudioBufferView<float> view(
        data.outputs[0].channelBuffers32, 2, data.numSamples);
    preamp_.processBlock(view);
    tape_.processBlock(view);
    limiter_.processBlock(view);
    return kResultOk;
}
```
