# Changelog

All notable user-facing changes to DSPark are documented here.

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
