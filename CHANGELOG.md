# Changelog

All notable user-facing changes to DSPark are documented here.

## [Unreleased]

### Added

- `TimeStretch::Quality::Studio` and `PitchShifter::Quality::Studio`, a new
  engine and the default for new instances: phase-gradient heap integration
  on a reference summed over every channel, an unambiguous
  instantaneous-frequency estimate, and a time map anchored on strikes found
  by a look-ahead spectral-flux detector; inside a strike's lock the bins
  the strike rises into are copied unrotated, so it carries no pre-echo.
  Scored against an ideal rendering of the same scenes: time stretch
  spectral distance 4.12 dB (1.8 engine 5.01), strike timing 0.59 ms median
  (4.16); pitch shift spectral distance 1.96 dB at the default frame and
  1.47 at 4096 (1.8 engine 2.50), strike timing 0.68 ms (13.18).
- The Studio pitch shifter's latency is the same at every pitch (the 1.8
  engine's real delay drifted 64 ms late at -12 semitones and 32 ms early
  at +12 against one reported latency), and it shifts a source panned hard
  to either side exactly (the 1.8 engine, deciding phases on the first
  channel, attenuated a right-only source by 48 dB).
- `Resampler` kernels are designed per conversion from a specification - a
  passband edge and a stopband attenuation per quality tier, with the
  stopband starting at the lower Nyquist frequency - and every rational
  ratio (all common audio rates) uses exact polyphase phases with integer
  position arithmetic; other ratios read a 512-phase table with cubic
  interpolation across phases. In double precision, worst over six common
  conversions, Ultra leaves a -220 dB residual and a -214 dB stopband
  (before: -105 dB and -53 dB) with a flat passband to 0.915 of Nyquist.
  `getFilterLength()` reports the kernel length.

- `Limiter::setHold()`: the reduction is held after the last peak that
  asked for it before the release starts (default 10 ms, 0 to 50 ms). A
  50 Hz sine driven 6 dB over the ceiling measures -149.6 dB THD+N instead
  of -56.8 dB, 40 Hz -149.7 instead of -47.4, and 60 Hz + 7 kHz
  intermodulation -74.2 instead of -62.4 dB, for 0.21 dB of integrated
  loudness on a dense mix driven 12 dB over the ceiling. State blobs saved
  before the field existed restore a hold of 0.

### Changed

- `Resampler` quality tiers are now specifications (Draft 0.80 of Nyquist
  and 60 dB, Normal 0.90 and 100 dB, High 0.91 and 140 dB, Ultra 0.915 and
  210 dB), so kernels are longer than the former fixed 8/32/64/128 taps and
  latencies grow accordingly (High 44.1 -> 48 kHz: 112 output samples, was
  35). A ratio of exactly 1 is a one-sample delay. Draft's passband now ends
  at 0.80 of Nyquist.
- New `TimeStretch` and `PitchShifter` instances use Studio. Latencies at
  the default frames and 48 kHz: `PitchShifter` 5184 samples (was 4096),
  `TimeStretch`'s fixed-rate adaptor 5632 (was 2048). `prepare()`'s frame
  argument now defaults to 0, each engine's own frame. `setQuality()`
  selects the 1.8 renderings (`Standard`, and `High` for the pitch
  shifter), bit-exact as before; state blobs saved by 1.8 restore them.
  `PitchCorrector` keeps the 1.8 engine, whose retune dynamics it is tuned
  to. Crossing between Studio and a 1.8 engine in `PitchShifter` restarts
  the stream at the next block.

- `AlgorithmicReverb` mixes its tail through a time-varying feedback matrix:
  after the Hadamard mix, line pairs turn through slow Givens rotations
  (lossless, so every T60 stays exact), which spread modes alike at every
  frequency and let the delay lines wander less. A steady 3 kHz tone keeps
  11-14 dB more of its energy within +/-3 Hz in Hall and Cathedral (Hall
  -44 dB, Cathedral -45 dB outside), the modal ringing of Room and Chamber
  drops by about half, the late field of a lateral source decorrelates
  better (500 Hz coherence 0.17 instead of 0.20) and the CPU cost is
  unchanged. The Hall preset's modulation depth is 0.10 (was 0.13) and the
  Cathedral's 0.12 (was 0.16).
- `AlgorithmicReverb::setDecay()` is now the ISO 3382-1 mid-frequency
  reverberation time T_mid (mean of the 500 Hz and 1 kHz octave T60s). The
  loop's DC anchor is solved so that T_mid lands on the setting; before, the
  first-order absorption shelves left it 3-12% short (Cathedral at 12 s
  measured 10.6 s, now 11.8 s; Spring at 1.5 s measured 1.37 s, now
  1.46 s). Tails are correspondingly longer at the same setting.
- `BeatTracker::analyze()` on dense mixes: the metrical level is decided on
  a register-balanced onset envelope, correlation noise bumps are no longer
  tempo candidates, and the grid is chosen among a moving-period, a steady
  and a tighter steady reading by how much of that envelope each explains.
  `tempoBpm` is fitted to the grid with skipped or extra beats counted as
  such. On 46 synthetic pop masters (92-150 BPM) the level is right on all
  46 (35 before) and `tempoBpm` agrees with the grid's median interval
  within 0.55% (up to 17.7% before). `OnsetDetector::OdfFrame` gains
  `registers`, the flux per register group.
- `Compressor` `AutoMakeupMode::Static` keeps its textbook, program-
  independent offset, which also lifts passages that never reach the
  threshold; the behaviour is now pinned by a test and the documentation
  points to `Adaptive` for makeup only where gain is reduced.

### Fixed

- `LoudnessMeter::getLoudnessRange()` percentiles follow EBU Tech 3342:
  0-based rank round((n - 1) * p), never the relative-gate threshold, and
  short-term values sampled at 10 Hz (the minimum Tech 3342 has required
  since V3). A steady tone below 12 s read LRA 20 LU and now reads 0; on a
  music fragment the reading matches the Tech 3342 reference code within
  0.06 LU.

## [1.8.0] - 2026-09-26

### Added

- Plugin parameters: `choice()` (named positions whose labels hosts list,
  display and parse, including AU value strings and VST3 list / CLAP enum
  flags) and `stepped()` (evenly spaced discrete positions shown as whole
  numbers). Unparsable host text is refused instead of guessed.
- The installed CMake package (`find_package(dspark)`) now ships the plugin
  layer and defines `dspark_add_plugin()`, as `add_subdirectory()` and
  `FetchContent` do; the helper gains `VERSION` and `BUNDLE_ID`.
- `HilbertIIR`, a zero-latency analytic pair in quadrature from 20 Hz;
  `SincInterpolator` (32-tap windowed sinc) and `StretchedSincReader`;
  `PitchShifter` `Quality::High`; NoiseGate and Expander lookahead with
  reported latency; first-order ADAA in `WaveshapeTable`; fractional
  `SampleAndHold` periods; `FilterEngine::setShelfSlope()`.
- `Delay` insert API (`prepare(spec)`, `setMix()`, in-place dry/wet
  `processBlock`), `Crossfade` equal-power sine law with curve glides and
  `gainsFor()`.
- Gapless MP3 round trips: codec-delay flush, an Info frame with a gapless
  tag, and trimming by the tags of other encoders on decode.

### Changed

- `AlgorithmicReverb` rebuilt as a true-stereo 32-line FDN (16 in Eco) with
  exact per-band decay, a velvet-noise early field joined to the late field
  on one physical decay, a binaural stereo image with directional early
  reflections, and a dispersive spring model - at about half the CPU.
- The FFT is a split-format Stockham radix-4 engine (2x faster on SSE2, 4x
  with AVX2); `Oversampling` runs a true polyphase decimator on the shared
  `SimdOps` layer (about 2x faster).
- `Oscillator` waveforms are minBLEP band-limited by default; `Equalizer`
  bells default to the analog-matched design; `SpectralDenoiser` uses a
  decision-directed Wiener gain instead of a hard gate.
- The `Limiter` gain computer turns peaks down before the hard-clip
  backstop; AutoGain matches integrated K-weighted loudness; decibel
  conversions run on exp/log at half the cost.
- Mix, width, gain and shape changes across the effects glide over at least
  20 ms instead of stepping once per block.
- VST3 `restartComponent` and AU `Latency` listeners are called on the host's
  UI/main thread only: a latency change detected in the audio callback raises
  an atomic flag that a ~30 Hz UI-thread tick hands to the host, with no host
  call or allocation on the audio thread.
- The Conan recipe installs under `include/dspark` like the CMake package, so
  `#include <DSPark.h>` works with every package manager, and ships the plugin
  layer with `dspark_add_plugin()`. Both recipes pin the 1.8.0 source.

### Fixed

- Plugin wrappers: the soft bypass is delayed by the reported latency so a
  bypassed track stays aligned; blocks larger than the announced maximum
  are processed in chunks instead of passed through; input buses narrower
  than the output are read within bounds; the VST3 component handler is
  swapped without a use-after-free window.
- WebView editor: the Linux GTK pump is bounded so an always-ready source
  cannot stall the host UI thread.
- Delay read-position wrap under moving Binaural/Haas pans; Vibrato sweep
  leaps under FM; MultibandCompressor oversize blocks left dry; PitchDetector
  period interpolation (1760 Hz error from 1 cent to 0.02); MP3 encoder
  granules over 4095 bits; non-finite samples in integer WAV and MP3 output;
  Saturation ADAA precision in float; DynamicEQ and TransientDesigner
  detection on the analytic magnitude.
- The `wav_process` example compensates its chain latency, so the written
  file stays aligned with the source and keeps its tail.

### Migration

Changed defaults, plugin timing and packaging are covered in the
[v1.8.0 migration guide](docs/migration-v1.8.0.md).

## [1.7.0] - 2026-08-21

### Added

- `OnsetDetector`, `BeatTracker` and `LoopFinder` for transient analysis,
  tempo/phase tracking and bounded crossfade-ready loop discovery.
- `LoudnessNormalizer` for offline LUFS normalization under a true-peak ceiling.
- `TimeStretch`, `PitchCorrector` and `SpectralFreeze` for phase-vocoder time,
  pitch and spectral processing. Spectral freeze retains captured magnitudes
  while phases advance, are reconstructed or are decorrelated according to the
  selected mode.
- `FlacFile` for dependency-free native FLAC decoding, `MidiFile` for Standard
  MIDI File reading and writing, and `KeyDetector` for major/minor key
  estimation.
- Public `Biquad::setCoeffsNow()` for coefficients computed by the stream owner
  and for single-threaded or offline processing.
- Deterministic installed-header, include-order, dependent-comparison,
  concurrent-suite and float-cast-overflow sanitizer checks.

### Changed

- Automatic analysis windows now preserve their time span across sample rates.
  Resolved sizes are exposed through the relevant size getters, including
  `PitchDetector::getWindowSize()`.
- Audio-thread staged-state adoption is bounded. A contended update may be
  deferred to a later processing call instead of making the callback wait for
  the control thread.
- `BiquadCoeffs` is a non-template, double-precision coefficient set, and the
  `Biquad` recursion remains in double precision for both float and double
  buffers.
- `AudioBufferView` converting and pointer-array constructors express
  const-correctness as constraints, so type traits and `requires` expressions
  now report the legal conversion direction.
- `AudioProcessor` now describes a genuinely in-place processor. Read-only
  analysers no longer satisfy the concept.
- CI treats supported compiler warnings as errors and runs AddressSanitizer,
  UndefinedBehaviorSanitizer and float-cast-overflow coverage with GCC and
  Clang.

### Fixed

- Hardened WAV, MP3, FLAC and MIDI parsing against malformed sizes, truncated
  data and invalid metadata without adding runtime codec dependencies.
- Preserved valid processor state when `prepare()` receives non-finite or
  non-positive sample rates, including `PitchCorrector`.
- Left `AudioBuffer` safely empty after an allocation failure instead of
  retaining dangling channel views.
- Hardened plugin and DSParkLab host-input, shared-state and editor-lifetime
  boundaries while preserving real-time processing contracts.
- Removed shared fixed temporary filenames from the I/O test suite so parallel
  test processes remain isolated.

### Migration

Source and latency changes are covered in the
[v1.7.0 migration guide](docs/migration-v1.7.0.md).

## [1.6.1]

- Added the `AlgorithmicReverb` Eco quality mode.
- Added impulse-response decay scaling and tape-speed-style stretching to
  `ConvolutionReverb`.
- Added matching DSParkLab controls for the new reverb options.
