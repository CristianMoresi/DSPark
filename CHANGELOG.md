# Changelog

All notable user-facing changes to DSPark are documented here.

## [Unreleased]

### Added

- `AudioIntervalAnalyzer`: bounded block-stream analysis of exact sample
  intervals, with RMS/peak/finite true peak, scoped loudness, explicit validity,
  optional continuous observations and no retained PCM. Reuses existing meters
  and the scaled-energy kernel shared with `OfflineEnergyAnalyzer`.
- `LoudnessMeter::finalizeTruePeak()` completes a finite interpolation tail
  without advancing loudness windows or gates; `getMeasurementInfo()` exposes
  complete-hop coverage and absolute-gate counts to the stream owner.
  `TruePeakDetector::getTailPeak()` preserves live channel history.

- `OfflineStereoGenerator`: complete-source rendering through `StereoGenerator`,
  with explicit mono duplication, exact exclusions, compensated latency and
  measured output headroom. A host-owned canonical delta cache supports new
  widths/exclusions while validating source, clock, settings and cached PCM.
  Worker scratch is bounded independently of duration; output is transactional.

- `StereoGenerator`: twenty moving bands and rational color on a parallel copy,
  with delta add-back preserving the delayed original mid. Includes width
  smoothing, optional delta-only low cut, explicit local oversampling, source
  clock/reset contracts, presets, latency reporting and numerical-error status.
  Shares Core conversion and clipping kernels; processing allocates nothing.

- `OfflineSoftClipper` and `OfflineHardClipper`: full-source automatic sample-peak
  reduction using shared Clipper curves, continuous-interval antialiasing and
  finite-source sinc reconstruction through Core convolution and Cauchy maps.
  Actual rounded PCM calibrates the ceiling, with explicit local oversampling,
  original sample alignment, exact exclusions, bounded PCM caches and
  transactional output. Coarse maps scale with duration and reuse storage across
  calibration passes. No master gain trim or extra limiter.

- `OfflineBeatCompressor`: complete-source pulse peak leveling, upper-median
  reference, capped reduction, local tempo release and explicit unknown-tempo
  fallback, reusing shared analysis and the bandlimited gain renderer.
- `OfflineTempoAnalyzer`: source-bound beat intervals and ambiguity diagnostics
  using retained pooled-channel features and the existing `BeatTracker` engine.
  Adds per-job allocation accounting and cancellation without another FFT;
  unsupported pulse intervals return no reliable local tempo.
- `OfflinePunch`: source-aligned transient boost with a bounded, stereo-linked
  gain envelope; shared smoothing, exclusions, verified rendering and source
  analysis. The attack map also recovers quiet restarts masked by global novelty
  and rejects end-padding attacks, with sample-clock minimum spacing.
- `OfflinePeakCompressor`: automatic complete-source peak reduction, event-local
  holds, anticipatory smoothing and stereo-linked amplitude recovery. Reuses the
  transient analyzer, envelope follower and gain transport shared with Leveler.
  Requested reduction is capped by measured sustained headroom; exclusions and
  achieved sample-peak reduction are explicit, with no hidden audio clipping.
- `OfflineTransientAnalyzer`: separate attack and pulse maps from a shared pooled
  stereo spectrum, complete-source normalization, refined source-frame intervals,
  optional reusable novelty features and bounded worker memory. Reuses the same
  FFT, filterbank, novelty and peak-picking kernels as `OnsetDetector`.
- `OfflineLeveler`: automatic, stereo-linked upward macro RMS leveling with
  exact source-frame exclusions, reusable immutable plans, cancellation and
  transactional offline rendering. Reports output sample/true peak without
  imposing hidden limiting or attenuation. Shared exclusion feathers have three
  zero edge derivatives and smooth overlapping masks, avoiding gain cusps between
  neighboring protected regions.
- `OfflineEnergyAnalyzer` and shared offline source/sink/job contracts: complete
  mono/stereo analysis in bounded blocks, 64-bit source positions, actual final-bin
  duration, content/revision checks and payload-memory budgets. Maps retain energy
  features rather than PCM. See `docs/offline-processing.md`.
- `PitchCorrector::getState()` / `setState()` (scale, root, retune speed,
  formant preservation) and `SpectralFreeze::getState()` / `setState()`
  (freeze request, phase mode): they were the two effects with settings
  that a host could not save with a session.
- `DelayEstimator`: the delay between two recordings of the same material
  (a delivery against its reference), found by GCC-PHAT on a cross-spectrum
  averaged over the whole program and refined on its phase slope to a
  fraction of a sample, with a confidence and polarity. Memory is a few
  frames whatever the program's length; blocks of any size give the
  whole-signal answer. Measured over 11 s at 48 kHz: within 0.0005 samples
  on white noise down to 0 dB SNR and 0.003 on coloured noise at 20 dB, for
  delays from -250.4 to 4000.25 samples.
- `Delay::setPingPong()`: the insert-style `processBlock(buffer)` runs as a
  ping-pong delay (the first two channels cross their echoes) with the
  insert's blend, so one call serves both modes. Fully wet it matches the
  wet-buffer sequence sample for sample; the setting is saved in the state
  (older blobs restore it off).
- `Resampler::processRange()` and `getReach()`: offline, time-aligned
  conversion of any span of output samples, including the kernel's ringing
  before the first input sample and after the last.
- `Sampler`: a polyphonic multi-zone sampler. Zones map a recording over key
  and velocity ranges with root key, tuning, gain, pan and loop (continuous
  or sustain, with a crossfade baked at load); overlapping zones layer. The
  interpolator is a 64-tap Kaiser sinc (100 dB) whose kernel follows the
  playback rate - gathered at or below the recording's rate, scattered in
  output time above it - so transposition is clean both ways and nothing
  above the output Nyquist frequency folds: -118 to -130 dB THD+N from -12
  to +19 semitones, -110 dB of alias for an 18 kHz tone an octave up.
  Stolen voices fade over 3 ms, events are sample-accurate at any block
  size, and zones are replaced while playing through a lock-free handover.
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
- `TimeStretch::beginOffline()` / `pushOffline()` / `finishOffline()` with
  `pullOffline()`: the offline stretch fed and drained in blocks of any
  size, bit-identical to `process()` over the whole signal, holding only
  unfed input and unpulled output. `process()` is now that session run
  over one block.
- Incremental offline analysis for material that arrives in pieces:
  `BeatTracker::beginOffline()` / `pushOffline()` / `finishOffline()` and
  the same trio on `OnsetDetector`. Blocks may be any size; the result is
  bit-identical to `analyze()` / `detectOffline()` over the concatenation,
  and only the onset envelope is kept between calls, not the audio.
- `OnsetDetector::detectOfflineOnsets()` and `finishOfflineOnsets()` return
  each onset's strength with its position, on the scale
  `getOnsetStrength()` reports; `getMethod()`, `getThreshold()` and
  `getAdaptiveWhitening()` echo the settings in force.
- `FilterEngine::setDriftSeed()`: a non-zero seed makes the analog drift
  reproducible - the same seed renders the same output, and `reset()` or
  `prepare()` restarts the drift from it. Zero keeps the per-instance
  unique seed.
- `Reverb::loadIR(AudioBufferView<const T>, double)`: a multi-channel
  impulse response from memory, for hosts that keep IRs in their own asset
  store - the same resampling, per-channel convolvers and atomic
  publication as loading from a file.

### Changed

- `TubePreamp` tabulates the implicit Koren load line, including cathode
  feedback, instead of iterating the usual operating range per sample.
  The shared per-instance table occupies 149768 bytes; the independent
  current reference stays within 5e-10 A over five internal rates. A stereo
  two-tone benchmark at 2x is 4.0-5.9 times faster than the previous solver
  at the same factor, with the measurement conditions in the header.
  The default remains 2x. This does not solve high-drive aliasing: the
  updated spectral sweep includes +36 dB and finds worse 2x cases near
  14 kHz than the previous sparse measurements.
- `processBlock()` and `getLatency()` are the canonical in-place and audio
  delay names. Saturation's `process()` and the analog effects'
  `getLatencySamples()` remain compatible aliases. The cookbook documents
  separate-input utilities, analysis delay, branch alignment and setup-only
  configuration. The channel-strip example uses ProcessorChain to report
  the complete chain latency.
- `BeatTracker::analyze()` settles the metrical level with a model fitted to
  three public tempo-annotated collections: the proposed reading, its half
  and its double are scored on how each one's pulse sits in the full, the
  register-balanced and the four register envelopes, and at what rate it
  would be tapped. The header records the measured accuracy and distinguishes
  reuse of fitting material from the additional evaluation excerpts. The model is not
  consulted where the proposal's alternate beats cannot be told apart, so
  clicks, swing and quiet subdivisions keep their level; when it moves the
  level, the reading it moved from is `secondaryTempoBpm`. Viennese waltzes
  read at one beat per bar can still be moved to two; that limitation remains.
- `TransformerModel` oversamples its core 2x by default, with
  `setOversampling()` (1, 2, 4, 8, 16), `getOversamplingFactor()` and the
  factor saved in the state (older blobs restore 2x). At 1x the loop's
  harmonics of high tones folded back into the band: a 10.1 kHz tone at -6
  dBFS put its third at 17.7 kHz 61 dB down at the default drive and 37.5 dB
  down at +24 dB; at 2x the worst fold below 20 kHz over 1-15 kHz tones is
  83.4 and 59.4 dB down. `getLatency()` is no longer a static constant: it
  reports the oversampler's delay (64 samples at 2x, 0 at 1x),
  `getLatencySamples()` returns the same value as on TapeMachine and
  TubePreamp, the dry path of the mix is delayed to match, and blocks longer than the prepared maximum are
  processed in pieces. DSParkLab's Saturation and Transformer slots start at
  the library's 2x.
- `Saturation` oversamples 2x by default (it was 1x, the one saturator in
  the framework that did not): at 1x a 10.1 kHz tone at -6 dBFS through
  the default SoftClip left an alias at 17.7 kHz only 34 dB down, at 2x it
  measures 124 dB down. The oversampler adds 64 samples of latency at 48
  kHz, reported by `getLatency()`; `setOversampling(1)` restores zero
  latency.
- `Compressor::DetectorType::Hilbert` detects on the zero-latency allpass
  pair (`HilbertIIR`), backed by the rectified input at onsets, instead of
  the 191-tap FIR. The FIR was blind below about 200 Hz, so low tones were
  modulated at their own frequency exactly as with the peak detector; now a
  30 Hz tone 14 dB into 4:1 at 5/100 ms measures -96 dB THD+N (before -35
  dB) and 20 Hz -107 dB, the static curve holds within 0.02 dB, and the
  detector adds no latency (it reported 95 samples). A step escapes as it
  does with the peak detector.
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

- `LoudnessMeter`: the -100 LUFS floor is monotone across its former power-domain
  discontinuity, including positive sub-floor powers in float and double.
- `AlgorithmicReverb`: compute feedback rotation sine/cosine in double before
  rounding to the sample type, avoiding amplified float-libm discrepancies.
  Full-signal x64 MSVC/GCC/Clang fixtures now gate numerical agreement across
  three sample rates and static, preset and maximum modulation.

- Widening `AudioBufferView` conversions now retain small-capacity channel views,
  allowing `Oversampling<T, 2>` and other compact Core buffers to use the common
  processing API. Const sample access remains const; implicit capacity narrowing
  stays prohibited.

- `FIRDesign` propagates allocation failures from its coefficient factories,
  allowing callers to recover instead of terminating the process. Successful
  designs produce the same coefficients.
- `BeatTracker` no longer counts an isolated half-beat phase transition as
  a full beat when stable intervals surround it. The worst tempo error on
  six 85/100/140 BPM phase-change trains falls from 4.01% to below 0.003%.
  With the level model and beat positions unchanged, tempo accuracy on the
  same 454 ballroom excerpts rises from 75.8% to 76.0%, Salsa stays at 68.9%,
  and the expanded 3223-loop evaluation rises from 49.7% to 50.5%. On 242
  additional ballroom excerpts kept apart from the original fit and the
  slope development, accuracy rises from 73.6% to 74.8%.
- Cookbook synthesis recipe and umbrella class index use `ADSREnvelope`,
  the declared type. Windows comment-path checks normalize Git paths with
  forward slashes; the VST3 smoke host avoids a shadowed local rejected by
  strict compiler warnings.
- `TapeMachine` no longer mirrors the programme about a quarter of the
  sample rate. Its AC-bias carrier sat at 0.375 of the internal rate, where
  the carrier's third harmonic folds onto the base-rate Nyquist frequency:
  every tone came out with an image at 24 kHz minus its frequency (9.9 dB
  below a 10.1 kHz tone at -30 dBFS, 22 dB below a 5 kHz one at 48 kHz).
  The carrier now runs at a quarter of the internal rate, where all its
  harmonics fold onto 0, itself or the internal Nyquist frequency: the image
  measures 80 dB down, and the reference-level response, the odd-dominant
  saturation and the drive law are unchanged within their tests.
- `Reverb` (convolution) converts an impulse response held at another rate,
  or stretched, without changing what it does. The IR was streamed through
  the resampler: its gain moved with the rate ratio (+6.02 dB for a 48 kHz
  IR at 96 kHz, -6.02 dB the other way), the resampler's latency stayed
  inside the response (the direct sound 71 samples late from 44.1 to 48
  kHz) and was never flushed, so a short IR came out as a fragment (-18.2
  dB). It is now converted offline and time aligned over the kernel's
  whole reach, and scaled by the rate ratio: a unit impulse keeps a DC gain
  of 1.000000 with its peak exactly on `getLatency()`, for every rate pair,
  IR length and stretch. The kernel's ringing ahead of an IR that starts at
  full level is kept and declared as latency (112 samples for 44.1 into 48
  kHz), with the dry path delayed to match; an IR with pre-delay holds it
  in its own silence and adds none.
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
