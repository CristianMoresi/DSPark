# Changelog

All notable user-facing changes to DSPark are documented here.

## [Unreleased]

### Added

- `Mp3File` decodes MPEG-2 and MPEG-2.5 Layer III (16, 22.05, 24, 8, 11.025
  and 12 kHz, ISO 13818-3), with LSF scalefactors and intensity stereo. LAME
  files at every rate match minimp3 at 131 dB or better.
- `WavFile` reads RF64 and BW64. `setRf64Mode()` writes RF64 always, or
  promotes the file only when it outgrows 4 GiB.
- `AudioFile::hasWriteError()` reports a failed write or header finalisation,
  including after `close()`.
- `Sampler::noteOnTuned` and `Synthesizer::noteOnTuned`: sample-offset
  note events with individual tuning in cents (-100..100), added to
  existing pitch controls. Integer keys still select zones and identify
  note-off/retrigger events. Synthesizer glide follows tuned pitches and
  mono/legato restores the tuning of the most recent held key. Existing
  `noteOn` remains the zero-cent entry point.
- Both instruments expose `kMaxEvents`, their shared per-block event queue
  capacity, so hosts can split dense event streams before queuing them.
- `Convolver::Partitioning::NonUniform`: the latency stays one block, while
  a long IR's tail runs in larger partitions whose work is spread over the
  samples. `getTailBlockSize()` reports the tail partition.
- `Equalizer::setLinearPhaseKernelLength()` and `getLinearPhaseKernelLength()`:
  the linear-phase FIR length, applied at `prepare()`. The default is 32768
  taps at 44.1 and 48 kHz, scaled with the rate to the nearest power of two.
- `Delay`, `Chorus` and `Vibrato` take `setInterpolation(Interpolation::Sinc)`:
  the framework's 32-tap sinc reader for modulated reads, with error below
  -80 dB at 12 kHz where Hermite reaches -21 dB, at about 5x the read cost.
  Hermite stays the default.
- `Delay::getDelayMs()`, and `Panner::getLatency()`: 4 samples for Binaural
  and Haas, 0 for the other algorithms.
- `Oversampling::Mode::PolyphaseIir` (opt-in; FIR stays the default):
  elliptic allpass half-bands designed in the class. 2x High delays 3 samples
  instead of 64 at -99 dB alias rejection; the phase is not linear and
  `getRoundTripDelay(hz)` reports the delay per frequency (3.07 at 1 kHz,
  4.23 at 18 kHz).
- Analog-matched designs: `BiquadCoeffs::makeLowPassMatched`, `makeHighPassMatched`,
  `makeBandPassMatched`, `makeLowShelfMatched` and `makeHighShelfMatched`, and
  an opt-in `setMatchedFilters()` on `FilterEngine` and `Equalizer`. At 48 kHz
  a 15 kHz low-pass deviates 0.58 dB from the analog prototype (9.4 dB bilinear).
- `Dither::setShaper(Shaper::Lipshitz5)`: a five-tap noise shaper, 4.9 dB less
  noise than the first-order shaper in 1-5 kHz. The default is unchanged.

### Changed

- Plugin layer, VST3: every MIDI controller (CC 0-127, channel pressure,
  pitch bend) now maps through `IMidiMapping`, as CLAP and AU already
  delivered; the hidden proxy parameters grow from 64 to 2080.
  `paramIdsUnique()` also rejects ids that hash into that 0x4D44xxxx range.
- Plugin layer: parameter id lookups use a compile-time id table (a block of
  128 automation points: 12.8 to 4.9 us).
- `Oversampling`: stages after the first use shorter half-bands, so 4x/8x/16x
  High delay 70/73/74 samples (was 96/112/120) and cost half to a quarter as
  much. Alias rejection is unchanged. Effect latencies follow: 197 for
  `TapeMachine` at its default 4x (was 223), 73 for `TubePreamp` and 68 for
  `Synthesizer` at 4x, 70 for the other effects at 4x.
- Faster block paths with unchanged output: `Biquad` runs channel pairs
  together (stereo about 2x faster), `StateVariableFilter` 2x on MSVC (GCC
  and Clang already ran most modes at that speed) and `LadderFilter` 13%
  faster; `FIRFilter::processBlock` 1.5x (rounding-level differences from
  `processSample()`); `Hilbert` skips its zero taps (SSE2: 1.6x float, 2.7x
  double; AVX2: 2.2x double, float unchanged); `fastLog` reads the exponent
  bits (1.3x GCC, 1.75x MSVC, float).

### Fixed

- Plugin layer: a block with more than 256 events no longer loses events.
  Intermediate parameter points give way, so final parameter values and
  note-offs survive (VST3, CLAP, AU; a stuck note before).
- Plugin layer: the plugin's `setParameter` has one caller at a time. UI
  edits, state loads and presets are applied by the audio thread while the
  host processes; two writers tore single-writer setters such as
  `Equalizer::setBand`.
- Plugin layer: a bypass crossfade starts at the bypass point, not at the
  block start; parameters missing from a loaded state take their defaults;
  states over 16 MiB load back; an effect whose input is missing processes
  silence instead of the host's stale output buffer.
- VST3: `restartComponent` is never called off the UI thread;
  `setProcessing(false)` calls the plugin's `reset()`; `terminate()` releases
  the component handler; UTF-8 names and units reach hosts as UTF-16; a null
  left output channel is no longer written; the pitch-bend proxy rests at its
  centre.
- CLAP: a latency change while active requests a host restart and is
  reported from `activate()`, the only place `latency.changed` is allowed;
  wildcard note-offs and chokes release the matching sounding notes;
  `CLAP_PARAM_IS_STEPPED` is set only for whole-number grids; editor edits
  re-read the latency.
- AU: raising `MaximumFramesPerSlice` while initialized no longer overruns the
  render buffers; ramped parameter events follow their line instead of
  jumping to the end value; a sidechain stream format no longer changes the
  main width; state and preset loads notify parameter listeners; the
  Objective-C editor classes carry a per-binary name.
- WebView editor: a poll reply in flight no longer moves a knob back during a
  drag.
- `DSPark.h` and the plugin headers compile after `<windows.h>` without
  `NOMINMAX`.
- `AlgorithmicReverb`: the modulated reads cover a fractional delay of
  [0.1, 1.1) instead of [0.5, 1.5). Where the read index steps, the old range
  changed the high-frequency phase delay and left noise: energy beyond 200 Hz
  of a steady 8 kHz tone fell from -50 to -65 dBc. Default audio changes for
  every type and quality, FDN and spring; level and T60 are unchanged.
- `AlgorithmicReverb`: `setType()` writes the preset's values on the calling
  thread. A setter called right after it was lost when the audio thread was
  applying the preset (7% of trials in a stress test).
- `AlgorithmicReverb`: width, early and late level, diffusion (including the
  discrete/diffuse split of the early taps) and the tone filters glide over
  about 20 ms instead of stepping at a block edge; the tone filters crossfade
  when switched. A change before the first block after `prepare()` or
  `reset()` still applies at once. Audio changes only while these move.
- `AlgorithmicReverb` keeps the dry signal of a stereo buffer on a mono spec
  and of a block longer than `maxBlockSize`, which it now processes in
  conforming parts; `reset()` renders like a fresh instance (it differed by
  6.6e-7); and `setState()` switches the tone filters off when the blob has
  them off.
- `Reverb` runs non-uniform partitions. A 6 s IR at 64-sample blocks costs
  0.96% of a core instead of 18%, and 20 s 2.3% instead of 248%; the latency
  is unchanged and the output changes at rounding level only.
- `Reverb::setDecayScale()` fits the IR's decay on its tail, after the direct
  sound, with one fit for all channels. A hall with a +20 dB direct sound
  scaled by 0.5 measured 0.40 instead of 1.0 of the expected T60, and the two
  channels of a stereo IR drifted apart. IRs with a strong direct sound sound
  different at scales other than 1.
- `Reverb` fades the wet signal across an IR replacement made while audio runs
  (it stepped by 6x the steady sample-to-sample change), keeps the dry delay
  of a converted IR on buffers with more channels than prepared, keeps the dry
  signal of blocks longer than `maxBlockSize`, and `Reverb<double>::getState()`
  compiles.
- `RingBuffer` stays unprepared when its byte size does not fit a 32-bit
  `size_t`; a 2^30-sample float ring got a zero-size block and wrote past it.
- `Mp3File` accepts a frame only when the next header confirms it, so tags,
  cover art and other binary data no longer decode as full-scale noise, and
  non-MP3 files are rejected.
- `Mp3File` intensity stereo starts above the right channel's last non-zero
  band. FhG files at 48-64 kbps matched the reference decoders at 32.8 dB.
- `Mp3File` bounds decoded PCM at `kMaxDecodedPcmBytes` and computes the size
  in 64 bits. A 32-bit `size_t` used to wrap and overrun the buffer.
- `Mp3File` mutes a frame whose bit reservoir is missing instead of decoding
  noise. It drops a count1 quad that overruns `part2_3_length`.
- `Mp3File` trims only the start when a gapless tag counts fewer frames than
  the file holds; concatenated files kept one copy. It also accepts tag CRCs
  taken over 190 bytes with the field zeroed, as FFmpeg writes them.
- `Mp3File` decodes 2.3 to 6.5 times faster (Huffman trees, a fast DCT-32 and
  a symmetric IMDCT), and the encoder runs 7 times faster with byte-identical
  output.
- `Mp3File::readSamples()` returns false on a writer, which used to crash.
  `Mp3File` and `WavFile` range checks no longer overflow int64.
- `Mp3File::writeSamples()` and `MidiFile::write()` report I/O failures; a
  short MIDI file returned true with nothing on disk.
- `FlacFile` decodes pre-2007 variable-block streams (RFC 9639 B.1) and fixed
  streams with unequal STREAMINFO bounds. It reads 5.4 times faster with
  identical PCM, and skips MD5 work when the stored digest is zero.
- `WavFile` pads an odd-sized data chunk to a word boundary.
- `stateFromJson()` round-trips +-FLT_MAX, rejects trailing text, and rejects
  more than 65535 parameters or keys over 255 bytes. `StateWriter::blob()`
  returns an empty blob in those cases instead of wrapping its entry count.
- `TimeStretch` and `PitchShifter`, Studio engine: one NaN or Inf input sample
  wrote a byte before a heap block and left the output NaN until `reset()`.
  Non-finite samples now read as silence, and a spectrum that overflowed
  restarts the phase of its bins.
- `SpectralDenoiser`: one NaN no longer stays in the gain memory or the learned
  profile. The frames holding it come out silent.
- `PitchFollower` judges each detection exactly once. At blocks of 170 samples
  or less one bad reading could confirm itself and move a note by 2614 cents,
  and above the hop a real interval took up to 536 ms instead of 48 ms.
- `PitchDetector`: a note after digital silence no longer reports half the
  sample rate at confidence 1.0.
- `SpectralFreeze`: a Diffuse hold keeps the input level; it dropped 3, 6 or
  9 dB at hops N/2, N/4 and N/8.
- `TimeStretch`, Standard engine: the `feedInput()`/`pullOutput()` stream now
  starts on the stretched timeline, up to 768 samples off before. Priming and
  tail figures moved with it.
- `TimeStretch`: offline sessions over 2^31 output samples no longer come out
  empty on targets with a 32-bit `long`. `stretchedLength()` is new.
- `Resampler::prepare()` ignores non-finite rates and ratios above `kMaxRatio`
  instead of overrunning the output buffer.
- `GranularProcessor`: long grains are cut to what a short capture ring holds
  instead of crossing the write head, and a non-finite input sample reads as
  silence instead of being replayed by the grains.
- `PitchShifter` and `SpectralDenoiser` report 0 latency until prepared, and
  `PitchShifter` and `TimeStretch` publish it through a relaxed atomic.
- The Studio engine queues bins in buckets instead of a binary heap, 12 to 44
  percent less CPU at equal measured fidelity, and `PitchDetector` halves its
  correlation transform: about 45 percent less CPU, 40 percent less memory.
- Looped zones no longer tick at the loop rate at or below the recording's
  rate. After a wrap the kernel read the audio before the loop, or zeros for a
  loop from sample 0, instead of the loop's end: -77 dB and -45 dB, now below
  -110 dB.
- Sample zones stay in tune far below their root key. The 1/8 rate floor, which
  only the scatter form needs, held `Sampler` at -36 semitones and
  `Synthesizer`'s sample source at -24 semitones at 2x and -12 at 4x, up to an
  octave and more sharp.
- `Sampler` and `Synthesizer` keep queued events across a zero-sample block.
  Dropping them left a released note sounding forever.
- `Synthesizer` mono and legato moves restart the sample source on a retrigger
  or a change of zone, through a 3 ms fade. A one-shot stayed silent after a
  retrigger and a move kept the previous key's zone.
- `Synthesizer` honours a mono zone's pan (+1 was silent, -1 centred) and keeps
  a stereo zone's right channel when its level comes from the matrix.
- Replacing the whole polyphony at one sample now fades every stolen voice:
  both instruments keep one spare voice per voice. They cut the voices that
  found no spare slot, about 47 dB louder at the edge, and use about 1.5x the
  memory now.
- `Sampler::processBlock()` no longer crashes on a view with no channels.
- `Synthesizer` noise keeps its level at every oversampling factor; it lost
  3 dB per doubling.
- A released Sustain loop now leaves a loop that has no read position clear of
  its crossfade, such as a crossfade as long as the loop. It used to loop on
  until the envelope ended.
- An LFO source in an LFO Rate matrix slot now works, one control period late;
  it read zero. Free LFOs still ignore LFO Rate, as documented.
- `Sampler::setGain()` is smoothed over about 10 ms instead of stepping at a
  block edge.
- `ADSREnvelope` and `Oscillator` ignore non-finite parameters and rates,
  `Phasor` wraps infinity to 0, and `ModulationRouter` survives a NaN smoothing
  time or sample rate; each used to hold NaN until reset. Its state is now
  double, so a float route settles instead of stalling 0.02% short.
- `Sampler.h` no longer claims the first pass of a crossfaded loop is the
  untouched recording: every pass of an engaged loop reads the baked region.
- `LoudnessMeter::getIntegratedLUFS()` now sums the exact energy of the blocks
  in each histogram bin instead of the bin centre, so a reading is no longer
  held to a 0.1 LU grid: it agrees with libebur128's exact block mode to four
  decimals on steady tones and within 0.004 LU on the programmes measured. The
  relative gates keep a bin only when its centre clears the threshold. The
  meter grows by 8 KB and stays allocation-free.
- `AlgorithmicReverb<float>` now keeps its modulation phase and read positions
  in double. The float versions drifted in rate and resolved a long line to
  about 0.001 samples, which put the float output only 28 dB under the signal
  away from the double result; it is now 50 to 80 dB under it.
- `AlgorithmicReverb<float>` now uses the shared double-precision `Biquad`
  core for its subsonic and stereo-coherence output high-pass filters.
  The former float recursions amplified one-ULP input perturbations into
  approximately -105 dBFS residuals in a 48 kHz Room fixture. The correction
  preserves the filter design, delay network, controls and latency; float
  output bytes change. Double processing retains its existing arithmetic.
- `LoopFinder` now scores a seam at the lag `renderLoop()` overlaps,
  period - crossfadeLength, not period - comparisonLength. With the default
  settings a tone could score near zero and render a crossfade in antiphase
  (RMS min/max 0.28, now 0.99). Chosen loop points change. Loops too close to
  the file edges are scored at the nearest lag that fits.
- `DelayEstimator::estimate()` over whole buffers no longer flushes zeros after
  a programme long enough for one frame. The shared step at lag 0 misread a
  2000-sample delay on a bass stem by about 2005 samples.
- `OfflineTempoAnalyzer` beats now carry the same localization as `BeatTracker`.
  They sat 13.6 ms early at 48 kHz and 22 ms early at 8 kHz, which marked every
  interval unreliable at 400 BPM on low sample rates. `algorithmRevision` is 3.
- `KeyDetector` frames now vote with chroma magnitude rather than energy, so a
  loud bass note no longer decides a frame: exact keys rise from 44.4% to 55.6%
  on a 216-piece synthetic bed. Keep the bed in mind: no recorded corpus was run.
- `BeatTracker` keeps its tempo range and tightness, and `OnsetDetector` its
  method, threshold and whitening, across `prepare()`, including values set
  before the first one.
- `BeatTracker` positions are no longer negative, and its running tempo stays
  inside the requested range.
- `LoudnessNormalizer` confirms its gain by measuring through it, so quiet
  programmes near the absolute gate reach the target: -25.48 LUFS became -23.00
  in the measured case. Programmes clear of the gate get the same gain as
  before, at the cost of one more measurement pass.
- `OnsetDetector` computes bin phases only for `ComplexDomain`: 71% less CPU
  with bit-identical SuperFlux output. `computesBinPhases()` reports it.
- `harmony::diatonicChord()` returns extensions as compound intervals (14, 17,
  21), matching `allChords`, and an invalid degree returns -1 for every interval.
- `KeyDetector::getKeyName()` writes C#m and G#m, not Dbm and Abm.
  `ChordDetector::getChordName()` spells roots by key, so Bb is no longer A#.
- `DynamicEQ` and `DeEsser` build their bells with
  `BiquadCoeffs::makePeakMatched()`. A band above Nyquist was unstable (a
  30 kHz band at 44.1 kHz reached 3e37, then NaN), and bells near Nyquist were
  cramped (+12 dB at 16 kHz gave +3.4 dB at 20 kHz; the prototype is +9.65,
  now +10.0). Default audio changes: bell shapes move to the prototype, by up
  to 0.05 dB for a 1 kHz bell, 2.8 dB at 20 kHz for an 8 kHz bell and 1.0 dB
  near 9.6 kHz for the default 7 kHz de-esser cut. DynamicEQ CPU +1.6%.
- `DynamicEQ`: a band re-added by `setNumBands()` starts clean (a boosted band
  peaked at 5.37 on a 0.001 tone); with oversampling, blocks longer than
  `maxBlockSize` are processed in chunks (the tail passed dry); `reset()`
  clears the analytic detectors and the oversamplers (silence after a reset
  came out at a 0.84 peak); `setBand()` raises the band count to cover the
  band (alone it did nothing); the state blob stores the lookahead (latency
  240 came back as 0).
- `DynamicEQ` and `DeEsser`: the 16-sample coefficient refresh runs on a
  free-running counter, so the output no longer depends on the host's block
  partition (it differed by up to 2.1e-2 and 4.6e-2). Unchanged for block
  sizes that are multiples of 16.
- One NaN or Inf input sample no longer latches. It reads as silence in the
  detectors and skips recursive filter state; the sample itself still passes.
  Before: Inf poisoned `Compressor` for good (every detector but
  SplitPolarity), NaN froze its mono Hilbert and SplitPolarity detectors and
  the FET colour, Inf muted `Expander` for good, `NoiseGate::processSample()`
  stuck open and Frequency mode went NaN, and `DynamicEQ`, `DeEsser`,
  `TransientDesigner` and the IIR `CrossoverFilter` (hence
  `MultibandCompressor`) stayed NaN.
- `NoiseGate`: an external sidechain shorter than the audio block was read past
  its end; it now falls back to the internal key.
- `CrossoverFilter`: band views with fewer channels than the input were written
  through a null pointer; the channel count now clamps to the narrowest view.
  Exact Butterworth Q values flatten the band sum to 1e-6 dB (3.4e-4 for
  LR24, 1.1e-3 for LR48). Default audio changes by 2e-6.
- `Compressor`, FeedBack with RMS, TruePeak or Hilbert detection: the
  detector read the signal after makeup, so +12 dB of makeup gave 22.5 dB of
  reduction instead of 13.5 and adaptive auto makeup ran away to -52 dB.
  Upward + FeedBack now lands on the panel ratio (8:1 observed 1.87:1), and the
  feedback attack compensation covers Upward too.
- `Compressor`: `processSample()` applies a sidechain HPF cutoff set after
  `prepare()` (it kept the old one); a live character change no longer steps
  the gain (Clean to Opto jumped 5.9 dB in one sample); the dB envelopes are
  double (a 3 s release at 192 kHz parked 0.57 dB short of its target).
  Default audio changes by up to 2.4e-6 (Opto 8e-6). The FET aliasing note
  states the measured -41 to -55 dBc; it claimed -72 dBc or better.
- `Limiter`: adaptive release counts the time peaks ask for reduction, so it
  slows after sustained limiting (2 s of limiting: 239 ms to recover instead
  of 87 ms, the same as without it). Off by default.
- `Expander`: the detector follows max(|x|, analytic magnitude) with an
  instant attack. The old smoothed rectifier read a tone about 1.07 dB low, so
  the gain below the threshold was (R - 1) x 1.07 dB under the documented law
  (-22.15 dB instead of -20 at ratio 3, 1 kHz), and a 30 Hz tone had -30.7 dB
  THD+N (now -87). Default audio changes: up to 3.2 dB more gain below the
  threshold at the default ratio 4. CPU 5.8 to 42 ns per stereo frame.
- `TransientDesigner`: attack and sustain changes ramp at one unit per 10 ms
  (`setAttack(100)` at an onset stepped the output by 3.56 in one sample).
  Default audio unchanged.
- `DeEsser` installs a `DenormalGuard` (silence after signal cost 20.1 instead
  of 10.6 ns per stereo frame) and writes its own coefficients with
  `setCoeffsNow()`.
- `TruePeakDetector<float>` runs the four phases as one 4-lane pass on SSE2 and
  NEON: 12.4 to 3.45 ns per sample, and `Limiter` with true peak 37.4 to
  19.9 ns per stereo frame. Readings change by rounding only (under 2e-7);
  the scalar path (double, other targets) is unchanged.

- `TapeMachine` and `TransformerModel` no longer run their makeup
  calibration inside one callback. Drive automation cost 126% of a
  64-sample block at 48 kHz stereo (976% at 192 kHz with 128-sample
  blocks); it is now 14% and 80%. The calibration runs over the next
  24 ms (tape) or 40 ms (transformer) of audio, and the new
  drive/makeup/bias set is adopted when it finishes. `TapeMachine` now
  ramps that set geometrically over about 30 ms instead of stepping it,
  so a 0 to +24 dB drive step no longer clicks. Default audio: a
  parameter change now takes effect after the calibration plus the
  ramp; settled output is unchanged.
- `Hysteresis` uses a damped alpha transform (a = 0.9) for both the
  field derivative and the magnetization integral. The plain trapezoid
  left a permanent alternating dH/dt after any transient. A 0.001*a
  field tick erased the remanence (now 1.5%), the integration stayed
  7.5 dB off after an abrupt onset (now -67.6 dB), and `TapeMachine`
  at 1x fell silent (-221 dB) after 5 s of noise at 96 kHz. Default
  audio: the default `TapeMachine` output moves at -77 dB relative to
  the signal (+0.03 dB at 18 kHz, same level), and `TransformerModel`
  at -98 dB.
- `TapeMachine` scales its loss FIR with the sample rate (63 taps at
  48 kHz, 127 at 96 kHz, 255 at 192 kHz). 18 kHz read 2.1 dB lower at
  192 kHz than at 48 kHz; it is now within 0.2 dB. Default audio at
  44.1 and 48 kHz: unchanged. At 88.2 to 96 kHz latency grows by 32
  samples, and at 192 kHz by 96.
- The `TapeMachine` over-bias roll-off (bias above about 0.57) uses a
  first-order low-pass whose zero matches the analog gain at Nyquist,
  and it starts from the signal when it switches on. At bias 1, 18 kHz
  now reads -5.2 dB at 48 kHz and -4.8 dB at 192 kHz; it was -2.6 and
  -6.7 dB. The default bias is unaffected.
- `TapeMachine::setNoise()` gives the requested RMS level. It was
  13.9 dB low, so hiss that is switched on is now 13.9 dB louder.
- `TapeMachine`, `TubePreamp`, `Saturation` and `Clipper` process
  blocks longer than the prepared maximum in pieces. The oversampled
  paths left the tail of such a block unprocessed. The 1x paths of
  `TubePreamp` and of `Saturation` (with drift or a crossfade active)
  overflowed their scratch buffers.
- `Saturation::prepare()` re-applies the stored parameters. A second
  `prepare()` reset drive, mix, output gain and the filters to their
  defaults (-9.8 dB measured) while `getState()` still reported them.
- `Saturation` runs its drive, character and drift smoothers, its drift
  generators and its algorithm crossfade at the processing rate, so
  their times no longer shrink with oversampling. During a crossfade
  the two algorithms share one drive and character trajectory, so a
  preset change no longer depends on the host block size (up to 0.28 of
  full scale before). Slew sensitivity is defined at 96 kHz. Default
  audio at the default 2x: the drive t63 is 36 ms (was 18.7 ms), the
  character ramp 20 ms (was 10.7 ms) and an algorithm crossfade 10 ms
  (was 5.3 ms), as at 1x. Drift wanders at half its former speed.
  Settled output and slew at 48 kHz/2x are unchanged.
- The `Saturation` Tube curve (also the first stage of `MultiStage`) is
  a smooth biased tanh normalised to unit small-signal slope. The
  former switch between tanh(x) and tanh(1.15x) at zero left a 2nd
  harmonic of -30.6 dBc at every level. Default audio for Tube: H2 is
  -85.5 dBc at -60 dBFS and -32.5 dBc at -6 dBFS (was -30.6 and
  -33.0). The gain is 0 dB at -50 dBFS (was +0.63 dB) and -0.50 dB at
  -9 dBFS (was +0.03 dB). The ceilings are +1.11/-0.91 at character 0
  (were +1/-1). The worst alias at 2x and drive 0 falls from -42.7 to
  -60.6 dBc. `MultiStage` now also undoes its last stage's 0.8 drive
  factor, as its comment always claimed, so its gain is +0.01 dB at
  -50 dBFS and -0.84 dB at -9 dBFS (was -1.29 and -2.28 dB).
- `Saturation` HardClip antialiasing averages every sample in double. Its
  "linear body" shortcut switched delay at clip edges and aliased more
  than the plain curve (-24.6 against -30.1 dBc at +12 dB, 2x). Turning
  antialiasing on mid-stream no longer averages against a stale input
  (a 0.031 step on a -20 dBFS tone). Antialiasing is off by default.
- `Saturation::setOversampling()` ignores factors above 16. It used to
  report them while running at 16x. The DC blocker is documented at its
  real 5 Hz.
- `Clipper` keeps its total gain continuous when the stage count
  changes. Going from 1 to 4 stages at +18 dB drove a -20 dBFS tone to
  the ceiling for 20 ms.
- `TubePreamp` ramps supply sag over about 30 ms. A 0.3 to 1 sag change
  stepped a -20 dBFS tone by 0.0975.
- `TransformerModel::reset()` keeps the resonance bell. It wiped the
  coefficients, so at resonance 1 a 15 kHz tone lost 2.2 dB until the
  next parameter change.
- `AnalogRandom::Generator` reads its reseed flag with a relaxed load
  before exchanging it, and keeps its retrigger flag as a plain member,
  since only the stream owner touches it. With identical output, a
  per-sample drift read drops from 9.6 to 2.4 ns and the audio-rate path
  that `NoiseGenerator` uses from 11.2 to 4.7 ns. Non-finite rates, BPM,
  ranges and quantization steps are ignored; a NaN or infinite value
  used to latch the phase or the output at NaN. `getNextDiscreteInt()`
  over the full int range no longer overflows (it returned `INT_MIN`
  every time).
- `wdf` roots reset the tree on a non-finite incident wave or source
  voltage. One NaN source sample used to keep a diode clipper or the
  FMV stack at NaN.
- The `TapeMachine` documentation no longer claims that bias-carrier
  products always stay out of band. At the default 4x and 48 kHz, a
  19.25 kHz tone at -6 dBFS leaves a 255 Hz product at -46 dBc.
- `Smoothers` Linear, Exponential and SlewLimiter keep double state. In float
  a `Delay` time glide jumped 19.8 samples at its end, a 5 s ramp at 192 kHz
  jumped 1.3e-2, and a slew limiter at 10000 never moved.
- `Biquad`, `StateVariableFilter` and `LadderFilter` block paths recover from
  one NaN or Inf sample; before, every later sample came out NaN until
  `reset()`.
- `SmoothedValue` ignores infinite targets (which left it NaN), Chase runs at
  the same speed at every sample rate (it was 3.2x faster at 192 kHz than at
  44.1 kHz), and a 0 ms ramp snaps for every curve.
- `Dither`: one NaN input no longer turns every later output NaN.
- `StretchedSincReader` folds content above the output Nyquist below -100 dB
  (was -13 dB at 1.05x Nyquist): the kernel cutoff moves to 0.85 of Nyquist.
- `simd::applyGainRamp` and `addWithGainRamp` compute the gain from the sample
  index, so long ramps no longer drift (1.3e-3 over 4.8 M samples).
- `DspMath` documents the measured speed of `fastTanh`, `fastExp`, `fastTan`,
  `fastSin` and `fastLog` (some were overstated).

- `Equalizer` LinearPhase runs one fixed kernel through a partitioned
  convolution, so the response no longer depends on the host block. A 100 Hz,
  Q 2, +12 dB bell gave +0.69 dB at 64-sample blocks; it is now within 0.1 dB.
  The latency is kernel/2 + kernel/16: 18432 samples at 48 kHz, instead of
  `maxBlockSize`. A band change is designed in slices and crossfaded, where
  the kernel swap stepped by 5x the steady change. Default audio changes in
  LinearPhase mode.
- `Equalizer` MinimumPhase fades a band switched on or off over 20 ms. It
  stepped by 4.2x the steady sample-to-sample change.
- `FilterEngine`, and so `Equalizer`: the first block after `prepare()` or
  `reset()` starts on the set parameters instead of gliding from 1 kHz and
  0 dB for about 150 ms. The cutoff glides in log frequency, critically damped:
  a 10 kHz to 500 Hz jump dipped a 150 Hz tone by 8.4 dB. The 16-sample
  refresh grid follows the stream, so nonlinearity and drift no longer depend
  on block sizes. `setState()` restores the resonance of the shelf and tilt
  shapes, audio passes through before `prepare()` (it was NaN), and the
  audio thread writes coefficients directly. Default audio changes in the
  first 150 ms and during cutoff glides.
- `Delay` stores its time in milliseconds. A time set before `prepare()` (it
  was stored as 0), a re-prepare at another rate (the echo fell to +3 samples),
  a state blob from another rate (100 ms became 50 ms at 96 kHz) and the
  feedback filter corners (a 2 kHz low-pass read -0.96 dB instead of -3 dB)
  now hold. The first block starts on the set delay; the start-up glide
  played an early copy of the input. Default audio changes in the first 13 ms
  after `prepare()` or `reset()`, in `Delay` and in the Binaural and Haas
  paths of `Panner`, which use it.
- `Delay` runs its feedback filters through silence and zero feedback. A
  frozen state played back as a peak of 0.15 to 0.32 when the feedback
  returned, and gated tails were cut short. Switching the smoother from None,
  or `reset()` followed by `setSmoothingTime()`, keeps the echo time; it fell
  to +3 samples.
- `getState()` compiles for `Delay<double>`, `Chorus<double>`,
  `Phaser<double>`, `Tremolo<double>`, `StereoWidth<double>`,
  `Panner<double>` and `NoiseGenerator<double>`.
- `FrequencyShifter` uses the allpass-pair Hilbert network. The image stays
  below -50 dB down to 60 Hz; it was -12 dB at 200 Hz. The latency is 0
  instead of 95 samples. Default audio changes.
- `Chorus`: a stereo spread of 1.0 puts the outer channels 180 degrees apart;
  it was 11.25 degrees. Spread and voice-count changes glide instead of
  resetting the LFO phases, which stepped by 5.6x the steady change. Blocks
  longer than `maxBlockSize` are mixed in full, and denormals are flushed.
  Default audio changes: the default spread of 0.5 is now 90 degrees.
- `Phaser` mixes blocks longer than `maxBlockSize` in full and flushes
  denormals.
- `Vibrato<float>` tracks the LFO phase and read position in double. A
  4-semitone sweep at 0.1 Hz strayed -54 dB from the double instance. Default
  audio changes at rounding level.
- `StereoWidth` and `DCBlocker` no longer latch to NaN after one NaN or Inf
  input sample.
- `NoiseGenerator` keeps the brown-noise corner at 38 Hz at every rate; it was
  77 Hz at 96 kHz. Default audio changes for brown noise away from 48 kHz.
- `Goertzel` runs its resonator in double. In float, a 27.5 Hz tone over
  65536 samples at 192 kHz read 3.96 dB low.
- `LevelFollower`: the float anti-denormal offset no longer lifts a long RMS
  window above the -100 dB floor in silence. A 3 s window at 192 kHz read
  -92.5 dB.

## [1.8.1] - 2026-10-13

### Added

- `Synthesizer`: a polyphonic synthesizer. Engines: analog (three
  `Oscillator`s with pulse-width modulation and hard sync), four-operator FM
  (eight algorithms, feedback, an envelope per operator) and wavetable (two
  morphing `WavetableBank` oscillators). Every engine adds a sub-oscillator,
  noise and a sample source that plays `SampleZoneSpec` zones through
  Sampler's band-limited reader. Per voice: ladder or state-variable filter
  with a per-sample cutoff, three ADSRs, two `Lfo`s, an eight-slot modulation
  matrix, unison (1..8), glide, poly/mono/legato, seeded analog drift.
  Sample-accurate events, Sampler's click-free stealing, a flat parameter
  table shaped like the plugin layer's, state blobs, and optional 2x/4x
  oversampling with an integer reported latency. Renders are bit-identical
  for any block size. Measured: pitch within 0.0001 cent, FM sidebands within
  1e-5 dB of the Bessel functions, the sample source within 2e-16 of
  `Sampler`. Voices are filtered in pairs, two independent ZDF recursions in
  flight, with an output identical to one voice at a time: 46 to 58 ns per
  voice and output sample on the 1x benchmark patches (the analog patch 16%
  faster than unpaired).
- `Lfo`: seven shapes (sample-and-hold and smooth random drawn per cycle from
  a seed), Hz or tempo divisions, delay and fade-in, rate modulation; every
  shape is a pure function of the cycle position, so a free-running LFO can
  be read at any instant.
- `WavetableBank`: mipmapped multi-frame wavetables with a morphing reader
  (Niemitalo's optimal 6-point interpolator over levels holding four samples
  per period of their top harmonic, its droop pre-compensated), four built-in
  banks and frames loaded from audio. Worst inharmonic component -151 dB with
  every harmonic at full scale, about -165 dB for ordinary spectra.
- `SampleZoneSpec` and `SampleLoopMode` (`Core/SampleZone.h`) describe a zone
  for both `Sampler` and `Synthesizer`; `Sampler::ZoneSpec` and
  `Sampler::LoopMode` remain as aliases.
- `Oscillator::setPulseWidth()`: pulse-width modulation of the square, both
  edges band-limited (also when they fall in one sample interval) and a
  width change across the current phase corrected as an edge.
- `Oversampling::getHighRateBlock()` and `getDecimationDelay()`: a
  decimation-only path for sources that render at the high rate.
- `MinBlepTable::addStep()`, `addDelayedStep()` and `addDelayedRamp()`: ring
  corrections over contiguous rows, and a minimum-phase band-limited ramp
  (minBLAMP) for slope changes against a delayed reference.

### Changed

- `MinBlepTable`: 128-tap Kaiser kernel and 2048 sub-sample rows refined from
  the cepstral step with Niemitalo's 32x interpolator. The worst inharmonic
  component of a saw or pulse from 110 Hz to 14 kHz goes from -89 dB
  (44.1 kHz) and -94 dB (48 kHz) to -143 dB or better; the 64-tap kernel let
  harmonics between 24.1 and 24.8 kHz fold below 20 kHz at 44.1 kHz. The rows
  also make a 5 kHz saw about 25% cheaper. `kTaps` is now 128 and
  `kOversample` 2048.
- `Oscillator`: phase, increments and edge times run in double for both sample
  types (a float accumulator detuned slow LFO rates); the hard-synced sine
  also corrects its slope change (-82 to -160 dB at 220 Hz x 1.5, -58 to
  -111 dB at 880 Hz x 1.5); the triangle's normalization follows a changing
  pitch without a `std::pow` per sample.
- `WavetableOscillator`: plays a `WavetableBank`, so its floor at 27.5 Hz goes
  from -72 dB to about -165 dB; `setBank()` and `setPosition()` play any bank
  with morphing.
- `LadderFilter` and `Sampler` move their per-sample arithmetic to shared
  kernels (`detail/LadderZdf.h`, `detail/SampleReader.h`); their output is
  bit-identical.
- `Sampler`: a mono zone's constant-power pan takes both gains from `cos`,
  so a centred zone is exactly mono and opposite pans mirror exactly
  (`sin` and `cos` of pi/4 round differently in glibc, leaving the centre one
  ulp off mono on Linux).

## [1.8.0] - 2026-10-05

### Added

- `wdf::ToneStackFMV::analogStateSpace()`: the continuous-time state space
  of the Bassman tone network (capacitor voltages as states), from nodal
  analysis of the same element values and topology as the R-type adaptor.
  With an ideal source and load it matches the published analog transfer
  function to 1e-8 relative.
  `wdf::Resistor::getResistance()` and `wdf::Capacitor::getCapacitance()`
  report element values.

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

- `OfflineStereoBalance`: source-bound side-only leveling from shared mid/side
  analysis, with an actual-render guard, exact exclusions and transactional
  output. It preserves the original mid without a master gain trim or limiter.

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
  The attack map also recovers upper-register strikes masked by falling
  low-frequency energy without extending bass holds. Corpus results include
  the measured precision/recall tradeoff in `docs/offline-processing.md`.
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
  spectral distance 4.12 dB (Standard engine 5.01), strike timing 0.59 ms median
  (4.16); pitch shift spectral distance 1.96 dB at the default frame and
  1.47 at 4096 (Standard engine 2.50), strike timing 0.68 ms (13.18).
- The Studio pitch shifter's latency is the same at every pitch (the Standard
  engine's real delay drifted 64 ms late at -12 semitones and 32 ms early
  at +12 against one reported latency), and it shifts a source panned hard
  to either side exactly (the Standard engine, deciding phases on the first
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

- `TubePreamp` tabulates the implicit Koren load line instead of iterating
  it per sample (the independent current reference stays within 5e-10 A
  over five internal rates), and at factors 2/4/8/16 solves the circuit in
  continuous time inside each internal sample interval: a Farrow polynomial
  reconstructs the grid input, intervals are split at the triode knees
  (closed form where a stage is saturated, Gauss-Legendre nodes elsewhere),
  the tone circuit is propagated exactly in its analog modal form and the
  plate voltage is band-limited once, at the output, with an exact
  compensation of the projection droop. At the 2x default the worst alias
  component below 20 kHz of a 23-tone sweep up to +36 dB drive is
  -80.6 dBc; the September 26 source (`d8a98a6`) measured -9.8 dBc for two
  stages at +36 dB on the same
  measurement. Against independent dense solutions of the same circuit
  equations the 2x waveform error is at most -73.5 dB. Reported latency is
  71/99/113/121 samples at 2/4/8/16x; 1x keeps the point circuit and zero
  latency. The shared per-instance table occupies 156672 bytes. Stereo
  processing at 2x costs 0.5-1.7 times that September 26 source at the same factor (less for
  clean and moderate settings, more for top-octave tones at +36 dB), with
  the measurement conditions in the header. The default remains 2x.
  Stage changes prime a prepared second circuit for 5 ms and then crossfade
  for 20 ms, preserving latency and avoiding callback allocations.
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
  level, the reading it moved from is `secondaryTempoBpm`. Rational period
  alternatives at 1/3, 2/3, 3/2 and 3 of the proposal now participate when
  range and pulse support allow, with phase-aware scoring and the original
  binary coefficients retained. These alternatives do not identify a time
  signature; the header documents grouped validation and training overlap.
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
  selects the earlier renderings (`Standard`, and `High` for the pitch
  shifter), bit-exact as before; state blobs without the quality field
  restore Standard. `PitchCorrector` keeps the Standard engine, whose
  retune dynamics it is tuned to. Crossing between Studio and Standard/High
  in `PitchShifter` restarts the stream at the next block.

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
- Updated mix, width, gain and shape controls glide in sample time instead of
  stepping once per block. Each class documents its transition duration.
- VST3 `restartComponent` and AU `Latency` listeners are called on the host's
  UI/main thread only: a latency change detected in the audio callback raises
  an atomic flag that a ~30 Hz UI-thread tick hands to the host, with no host
  call or allocation on the audio thread.
- The Conan recipe installs under `include/dspark` like the CMake package, so
  `#include <DSPark.h>` works with every package manager, and ships the plugin
  layer with `dspark_add_plugin()`. Both recipes pin the 1.8.0 source.

### Fixed

- Offline clipping reuses budgeted Core scratch storage when calibration changes
  the projection geometry. Repeatedly allocating equivalent maps could exhaust
  the cumulative job budget on long sources despite a smaller live workspace.
  The memory budget, numerical processing and target tolerances are unchanged.

- `OfflineBeatCompressor` compiles after the Windows SDK, whose `near` macro
  previously erased a local helper name. Include-order coverage uses the real
  SDK header; numerical processing is unchanged.

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
