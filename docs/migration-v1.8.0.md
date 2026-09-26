# Migrating from DSPark v1.7.0 to v1.8.0

v1.8.0 is source-compatible with v1.7.0: no public function was removed or
renamed. The changes below alter defaults, timing or packaging, so review the
ones that apply to your product before upgrading.

## Changed processing defaults

Three processors now default to the higher-quality design. Each keeps the
previous behaviour one call away:

| Processor | New default | Previous behaviour |
|---|---|---|
| `Oscillator` | Saw and Square edges corrected with a table minBLEP; band-limited edges ring to about 1.45x full scale | `setAntiAliasing(Oscillator<T>::AntiAliasing::PolyBLEP)` keeps every waveform inside [-1, 1] |
| `Equalizer` | Peak bands use the analog-matched design, so high bells keep their shape near Nyquist | `setMatchedBells(false)` restores the bilinear bell |
| `SpectralDenoiser` | Decision-directed Wiener gain over a mean-power noise profile, free of musical noise | No switch: re-tune the threshold by ear if a preset relied on the hard gate |

`AlgorithmicReverb` was rebuilt: the same parameters now drive a true-stereo
feedback delay network with a binaural early field. Presets load unchanged,
but the sound and the CPU cost (about half) differ, so re-audition mixes that
depend on it.

## Parameter changes glide

Mix, width, gain, shape and ceiling changes across the effects now ramp over at
least 20 ms instead of stepping once per block. Automation that previously
produced an instantaneous jump now takes one short glide; code that compared
the first processed block against the new target should allow for it.

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

## MP3 round trips

Files written by `Mp3File` now carry an Info frame with a LAME-format tag,
and decoding trims the encoder delay and padding it announces (LAME, FFmpeg
or DSPark tags). A decoded file therefore has exactly the length of the
source instead of carrying the encoder delay and end padding; code that
compensated for the codec delay by hand must stop doing so.
