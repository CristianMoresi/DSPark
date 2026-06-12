# A sidechain ducker — the reference for the sidechain contract

`ducker.cpp` is the canonical "music dips when the voice speaks" compressor:
`dspark::Compressor` driven by an **external key signal** the host routes in.

The whole sidechain story is one signature. Implement the two-buffer process
instead of the single-buffer one —

```cpp
void processBlock(dspark::AudioBufferView<float> io,
                  dspark::AudioBufferView<float> sidechain) noexcept;
```

— and every format backend announces a second stereo input named
"Sidechain": a VST3 **aux bus**, a CLAP **non-main port**, an AU **input
element**. DAWs show it in their routing UI exactly like for any commercial
dynamics plugin. When nothing is routed, the wrapper hands the plugin
silence with the same frame count, so the plugin code never branches.

The shape matches DSPark's own dynamics (`Compressor`, `NoiseGate`,
`Expander`, `DynamicEQ` all take `processBlock(audio, sidechain)`), so the
plugin forwards the two views 1:1.

Build and install like `examples/plugin_saturator/` (one binary, ship as
`.vst3` and `.clap`; `au/Info.plist` carries the `DSdk`/`DSpk` codes for the
macOS `.component`). The VST3/CLAP smoke hosts verify the bus layout and —
with `--expect-sidechain` — that a hot key signal actually ducks the main
audio, in CI on every commit.
