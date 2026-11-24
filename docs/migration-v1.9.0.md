# Migrating from DSPark v1.8.1 to v1.9.0

v1.9.0 is source compatible with v1.8.1. No public function, type or
enumerator has been removed or renamed, every enumerator keeps its value, and
every state blob saved by v1.8.1 loads. A key that an older blob does not
carry takes the v1.9.0 default, so a session saved with v1.8.1 can sound
different where that default changed. Review the sections that apply to your
product before upgrading.

## Changed processing defaults

| Processor | New default | Previous behaviour |
|---|---|---|
| `TapeMachine` | Four-level bias carrier: the gain no longer rises with the bias, and the top is cleaner at high level | `setBiasShape(TapeMachine<T>::BiasShape::TwoLevel)` restores the v1.8.1 carrier; states without the key take the new one |
| `AlgorithmicReverb` | The wet signal at full mix sits at the dry signal's level (within about +-1 dB on pink noise for every type) | The wet signal sat 7 to 16.5 dB under the dry one, depending on the type: re-balance mixes that relied on it |
| `GranularProcessor` | The grain cloud is loudness neutral at its defaults | The wet path was 7.27 dB quieter |
| `FilterEngine` | A cutoff, Q or gain change arrives in exactly 165 ms; the settled output is unchanged | A critically damped glide that took 331 ms |
| `PitchDetector` | `setNoiseSuppression()` is on, so notes are read through noise; `PitchFollower` and `PitchCorrector` keep it off | `setNoiseSuppression(false)` gives the plain reading |
| `KeyDetector` | `Profile::Recorded`, an analysis made for recorded music | `setProfile(KeyDetector<T>::Profile::KrumhanslKessler)` |
| `BeatTracker` (offline) | The beat grid is laid in two passes, with a tightness of 400 | `setTightness(25)` for the previous tightness |
| `OnsetDetector` | `Picker::Gated` with a delta of 0.0035 | `setPicker(OnsetDetector<T>::Picker::Plain)` and `setThreshold(0.03)` |

Plugin layer: starting the processing (VST3 `setProcessing(true)`, CLAP
`start_processing()`) now calls the plugin's `reset()`, as stopping already
did. VST3 maps every MIDI controller through `IMidiMapping`, so the hidden
proxy parameters grow from 64 to 2080.

## Latency

Always query the prepared instance's `getLatency()` and propagate it to the
host and to any parallel dry path. Several values move in this release:

- `Oversampling` High at 4x, 8x and 16x delays 70, 73 and 74 samples (was 96,
  112 and 120), and the effects that oversample follow.
- `TubePreamp` at 4x and above, `Saturation` from 8x and `Clipper` at 16x
  decimate through `Oversampling::Depth::Deep`: `TubePreamp` at 4x, 8x and
  16x reports 93, 94 and 96 base samples (was 73, 74 and 75).
- `Saturation` with `Antialiasing::SecondOrder` at the default 2x adds two
  base samples (one before).
- `PitchShifter::prepare(spec, 0)` takes the power of two nearest 43 ms at
  every rate: 2048 at 44.1 and 48 kHz as before, 4096 at 88.2 and 96 kHz.

## New processors and options

Everything new is opt-in or a new class, and leaves existing settings as
they were: `TonalBalance`, `ResonanceSuppressor`, `SpectralDynamicEQ`,
`MultibandSaturation`, `LoFi`, `DownbeatEstimator`, and primitives such as
`FilterDesign`, `BiquadCascade`, `KeyEqualizer`, `DetailFilter` and
`BucketBrigade`. The [changelog](../CHANGELOG.md) lists every option with its
measured figures.
