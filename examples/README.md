# DSPark examples

## Using the framework directly

| Example | What it shows |
|---|---|
| [`wav_process.cpp`](wav_process.cpp) | offline file processing: WAV in → mastering chain → 24-bit WAV out |
| [`channel_strip.cpp`](channel_strip.cpp) | the real-time block-processing pattern (prepare / per-block / setters) |
| [`pitch_tracking_eq.cpp`](pitch_tracking_eq.cpp) | analysis driving DSP: PitchFollower steering a tracking low-cut |

## Building plugins (VST3 / CLAP / AU)

In learning order — each example is the canonical reference for exactly one
capability. The full guide is [`docs/plugins.md`](../docs/plugins.md).

| Example | What it teaches |
|---|---|
| [`plugin_saturator/`](plugin_saturator/saturator.cpp) | the minimum: one effect, one file, three formats |
| [`plugin_template/`](plugin_template/plugin_template.cpp) | every optional contract method, present and commented (uses `PluginBase`) |
| [`plugin_ducker/`](plugin_ducker/ducker.cpp) | a sidechain input the host can route into |
| [`plugin_synth/`](plugin_synth/synth.cpp) | an instrument: MIDI in, voices, no audio input, factory presets |
| [`plugin_webview_editor/`](plugin_webview_editor/webview_saturator.cpp) | a custom HTML/CSS/JS GUI in a single file (knobs, gestures) |
| [`plugin_webview_files/`](plugin_webview_files/) | the production GUI workflow: separate `ui/` files embedded by CMake |

Every plugin example is compiled and driven through the smoke hosts (and the
official validators) by CI on every commit.
