# Migrating from DSPark v1.7.0 to v1.8.0

v1.8.0 adds offline processing and analysis APIs and changes several processing
defaults, timing contracts and packaging details. Review the sections that
apply to your product before upgrading, even when existing calls still compile.

## Changed processing defaults

The following processors change their default design:

| Processor | New default | Previous behaviour |
|---|---|---|
| `Oscillator` | Saw and Square edges corrected with a table minBLEP; band-limited edges ring to about 1.45x full scale | `setAntiAliasing(Oscillator<T>::AntiAliasing::PolyBLEP)` keeps every waveform inside [-1, 1] |
| `Equalizer` | Peak bands use the analog-matched design, so high bells keep their shape near Nyquist | `setMatchedBells(false)` restores the bilinear bell |
| `SpectralDenoiser` | Decision-directed Wiener gain over a mean-power noise profile | No switch: re-tune the threshold by ear if a preset relied on the hard gate |

`AlgorithmicReverb` was rebuilt: the same parameters now drive a true-stereo
feedback delay network with a binaural early field. Presets load unchanged,
but the sound and the CPU cost differ, so re-audition mixes that
depend on it.

`Limiter` now holds reduction for 10 ms before release. `setHold(0)` restores
release-only recovery; state blobs that predate this field restore zero hold.
This changes recovery after isolated peaks, so re-audition existing settings.

## Oversampling and latency

Always query the prepared instance's `getLatency()` and propagate it to the
host and any parallel dry path. Do not hard-code previous latency values.
The analog effects retain `getLatencySamples()` as a compatibility alias.

| Processor | Current contract |
|---|---|
| `TubePreamp` | Default 2x, selectable 1/2/4/8/16. Latency is 0/71/99/113/121 base-rate samples respectively. Stage-count changes preserve this latency and prime the incoming circuit for 5 ms before a 20 ms crossfade. At 1x the point-sampled circuit remains available; it does not have the alias rejection of the 2x path. |
| `Saturation` | Default changes from 1x to 2x, adding 64 samples. `setOversampling(1)` selects zero added latency. |
| `TransformerModel` | Default 2x, selectable 1/2/4/8/16. `getLatency()` is now an instance method, not a static constant; it reports 64 samples at 2x and zero at 1x. |
| `StereoGenerator` | New class; default 1x with 256 samples before an optional delta-only low cut. Higher factors are selectable at preparation. Width zero is delayed identity, not zero latency. |

Oversampling is a setup choice; prepare again as the individual class requires.
The TubePreamp circuit rendering and TapeMachine bias correction change existing
renders. Re-audition analog presets and retain the explicit sample rate/factor
when comparing results.

## Time, pitch and sample-rate conversion

New `TimeStretch` and `PitchShifter` instances use `Quality::Studio`. At 48 kHz
with default frames, the pitch shifter reports 5184 samples; the time stretcher's
fixed-rate adaptor reports 5632. Their frame argument defaults to zero to select
the engine's frame. Select `Quality::Standard` explicitly to retain the earlier
engine (`Quality::High` also remains available on `PitchShifter`). State blobs
without the quality field restore Standard. `PitchCorrector` still selects
Standard for its existing retune dynamics.

`TimeStretch::setQuality()` takes effect at preparation or reset. Crossing
between Studio and Standard/High in `PitchShifter` restarts the stream at the
next block and changes latency; Standard/High changes crossfade live. Schedule
engine changes where a stream restart and host latency update are acceptable.

`Resampler` designs filters per rate conversion. Quality tiers specify passband
and rejection rather than a fixed tap count, so query `getFilterLength()` and
`getLatency()`. For example, High at 44.1 to 48 kHz reports 112 output samples
instead of 35. Equal-rate streaming is a one-sample delay; offline conversion
is time-aligned. `processRange()` and `getReach()` expose finite-source support.

`Reverb` now aligns and scales resampled or stretched impulse responses over
the entire conversion kernel. An IR with no initial silence can require added
latency to retain its pre-ringing; query the prepared effect after loading it.

## Offline processing and interval measurements

The new `Offline*` dynamics, clipping and stereo adapters require complete-source
analysis and run on a worker, never an audio callback. They accept bounded source
and transactional sink adapters; convenience `run()` calls also allocate an
owning output. Plans belong to the exact stage input analyzed. Update source
revision and analyze again after editing or processing that input. Exclusions
are source-relative half-open sample intervals and retain their PCM exactly.

These APIs require exceptions and are omitted from `DSPark.h` when exceptions
are disabled or `DSPARK_NO_OFFLINE` is defined. Check `DSPARK_HAS_OFFLINE`.
The real-time `StereoGenerator` core remains separately usable. Offline clipping
defaults to 4x local processing, with 1/2/4/8/16 selectable; the
[offline guide](offline-processing.md) states factor-specific quality and cost.

The new `AudioIntervalAnalyzer` reports the requested and actually measured
interval, numeric validity, and complete loudness-window support. Use
`IncludeContinuous` for additional contextual observations; regional integrated
loudness still measures the interval independently. Finalize the true-peak tail
without feeding zero padding into loudness windows. See the
[contract and compiling example](interval-analysis.md).

`LoudnessMeter::getLoudnessRange()` now uses the EBU percentile ranks and 10 Hz
short-term sampling, so previously saved numerical expectations can change.

## Parameter changes glide

Updated mix, width, gain, shape and ceiling controls use time-based smoothing
instead of stepping once per block. Durations belong to each class's setter
contract: for example, `TubePreamp` limits dry/wet slew to a full-scale change
per 20 ms, while `StereoGenerator` width ramps over 5 ms. Code that compared
the first processed block against the new target should allow for the glide.

## Plugin layer

- **Bypass is latency-aligned.** The dry path of the soft bypass is delayed by
  the reported latency, so a bypassed track stays where the host compensates
  it. Plugins with no latency are unaffected.
- **Host text parsing refuses garbage.** Text that is neither a number, a
  choice label nor On/Off now makes `getParamValueByString` return
  `kResultFalse` and CLAP `text_to_value` return false, instead of mapping to
  the minimum. Hosts keep the current value.
- **Restart requests reach the host on its UI thread.** A latency change
  detected in the audio callback is delivered by a ~30 Hz UI-thread tick
  (VST3 and AU) instead of from the callback itself. Hosts that polled
  `getLatencySamples()` right after a block see the new value; the
  notification may arrive up to one tick later.
- **New parameter helpers.** `choice(id, name, labels, default)` and
  `stepped(...)` replace hand-built `Param { ... , steps }` aggregates. The
  aggregate form still compiles; a `Param` gained a trailing `labels` member
  that defaults to null.

## CMake package and helper

- `dspark_add_plugin()` is defined by `add_subdirectory()`, `FetchContent` and
  `find_package(dspark)`. A project that defines a function of the same name
  must rename its own.
- The installed package now contains the plugin layer and the vendored
  format SDK headers with their licenses.
- `dspark_add_plugin()` accepts `VERSION` (bundle and AU component version)
  and `BUNDLE_ID`. Set `BUNDLE_ID` to your own reverse-domain identifier
  before shipping on macOS; the default is only a placeholder.

## Conan include directory

The Conan recipe now installs the headers under `include/dspark`, like the
CMake package, and exposes that directory as the include root. The same
`#include <DSPark.h>` therefore works with every package manager. Code that
wrote `#include <DSPark/DSPark.h>` against the previous recipe must drop the
`DSPark/` prefix. The recipe also loads `dspark_add_plugin()` through
CMakeDeps.

The repository supplies a local Conan 2 recipe and a vcpkg overlay port. They
pin an immutable source commit and authenticated archive checksums; they do not
follow `main`. Their presence here does not mean publication in Conan Center or
the central vcpkg registry.

## MP3 round trips

Files written by `Mp3File` now carry an Info frame with a gapless tag, and
decoding trims the encoder delay and padding such a tag announces when its
CRC checks out. A decoded file therefore has exactly the length of the
source instead of carrying the encoder delay and end padding; code that
compensated for the codec delay by hand must stop doing so.
