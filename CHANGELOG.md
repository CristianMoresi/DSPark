# Changelog

All notable user-facing changes to DSPark are documented here.

## [Unreleased]

### Added

- `Limiter::setHoldMode(HoldMode::EveryPeak)` (opt-in, default `Deepest`
  unchanged and bit-identical): every peak's reduction is held for the hold
  time after it has passed (an exact sliding minimum over lookahead plus
  hold), so the gain no longer rises and falls between peaks of unequal
  height. A 60 Hz + 7 kHz two-tone driven 6 dB over the ceiling: worst
  sideband -87.1 dBc against -61.7 at the defaults, -84.3 against -69.4 at
  5 ms lookahead, 15 ms hold and 40 ms release; low-tone THD+N improves too
  (30 Hz at the defaults: -66.6 against -52.4 dB). The cost is loudness on
  dense material: 0.45 dB lower driven 12 dB over the ceiling at 5 ms / 15 ms
  / 40 ms, 0.10 dB at 6 dB over. Saved in the state (`holdMode`; blobs
  without it load `Deepest`); exposed in DSParkLab.
- `TruePeakDetector::getPhases()` and `getReversedPhase()`: the official
  Annex 2 phases for callers that run the interpolator as block FIRs.
- `FFTReal::inverseStep()` and `getInverseStepCount()`: the inverse transform
  split into its passes (step 0 packs the spectrum), for real-time callers
  that spread a long transform over several callbacks. All steps in order
  write exactly what `inverse()` writes.
- `Saturation` `Antialiasing::SecondOrder` now has kernels for the Exciter
  (exact for its clamped cubic) and the Tape curve (the anhysteretic
  magnetisation, from tabulated antiderivatives; its second antiderivative
  is not elementary), and covers every MultiStage stage from 2x. Worst alias
  over 1-19.25 kHz tones at -6/-18 dBFS, 2x, +24 dB drive: Exciter -27.5 dBc
  against -9.8, Tape -38.0 against -7.7, MultiStage -28.0 against -6.8
  (4x: -49.7 against -6.0; +12 dB at 2x: -37.1 against -16.0). Off and
  FirstOrder are bit-identical; the latency is still one base sample. Cost
  at 2x, stereo float: Tape 510 against 300 ns per frame, MultiStage 820
  against 410, Exciter 150 against 37.
- `OnsetDetector::setPicker(Picker)` with `Picker::Gated` (default) and
  `Picker::Plain`; `BeatTracker::getBeatTempoBpm()`, the tempo the causal
  beats are announced at.
- `LadderFilter::setSaturation(Saturation::Stages)` (opt-in, default
  `Feedback` unchanged and bit-identical): every stage saturates and the
  delay-free loop is solved implicitly (clipped Newton in double, 2.1 steps
  on average; below a 0.047 fs cutoff it starts at the integrator states). Against a 128x oversampled RK4 of the same ladder equations
  the harmonics land within 0.23 dB up to a 2 kHz cutoff (1.6 dB at 8 kHz),
  in float and double; self-oscillation at resonance 1.1 is within 0.1 cent
  and 0.07 dB at 1 kHz. Resonance accepts up to 1.2 (Feedback still uses at
  most 1). Costs 53-104 ns per sample and channel against 10-19 ns (56 ns
  at a 1 kHz cutoff with drive 3, where a fixed-pivot start costs 74). The
  filter gains `getState()`/`setState()` with the model in its state.
- `Smoothers::HermiteSmoother`: a cubic glide that lands exactly after the
  ramp time with a continuous slope, also when retargeted. At equal ramp
  time its zipper on a filter sweep is -68.3 dB against -53.8 dB for the
  linear ramp; about 1 ns per sample.
- `Biquad::processRampCore()`: filters a run while its coefficients move
  linearly per sample between two designs.
- `simd::firCorrelateBank()` and `simd::firCorrelateBankSum()`: banks of
  FIRs over one shared input, or summed over interleaved inputs, in one
  pass over tap-interleaved coefficients. The bank is bit-identical to
  separate `firCorrelate()` calls; both run on the AVX2 copy when the
  run-time dispatch is active.
- Plugin layer: `MidiEvent::choke`. A CLAP NOTE_CHOKE (a host choke group
  cutting a voice) arrives as a `NoteOff` with `choke` set, so a plugin can
  end the voice at once instead of running its release; a plugin that
  ignores the flag releases the voice as before. The synth example fades a
  choked voice in 1 ms: frames 128-511 of the choke block carried 53% of a
  held block's energy (its 150 ms release) and now carry none.
- `Clipper::setAntialiasing(bool)`: second-order antiderivative
  antialiasing (Bilbao, Esqueda, Parker and Valimaki 2017) for every mode,
  with exact closed forms where the curve has kinks. Hard clipping at 2x and
  +24 dB drive: worst alias over 1-19.25 kHz tones at -6/-18 dBFS -40.5 dBc
  against -14.5 off (4x: -60.6 against -21.0). The delay is padded to
  ceil(4 / factor) base samples for any stage count; getLatency() reports it
  and the dry path matches. Off by default; the default output is unchanged.
- `Saturation::Antialiasing::SecondOrder` (`setAntialiasing(Antialiasing)`):
  second-order kernels for SoftClip, Tube, HardClip, Wavefolder, Transformer
  and the MultiStage tube stage; the other algorithms are delayed to the same
  one-sample alignment. At the default 2x and +24 dB drive SoftClip reaches
  -42.1 dBc against -29.7 for first order and -15.2 off, Transformer -43.2
  against -15.7. Adds one sample of latency (reported, dry path aligned);
  -1.3 dB at 10 kHz and -5.9 dB at 20 kHz at 2x. State key "antialiasOrder"
  (the boolean "antialias" is still written and read). Off by default; the
  Off and FirstOrder outputs are bit-identical to before.
- `TubePreamp::setGridModel(GridModel::Conduction)`: grid current of the
  grid-to-cathode voltage (12AX7 fit of Dempwolf and Zoelzer, DAFx-11)
  through a 68 k ohm source, added to the cathode current. A 1 V source at
  rest reaches the grid at 0.936 V instead of the 0.624 V of the
  ground-referenced clamp. At 2x, one stage, 1 kHz at -6 dBFS and +12 dB
  drive the output differs from the clamp by -10.4 dB (second harmonic
  -33.9 against -15.0 dBc). Setup thread, re-runs the calibration; state key
  "gridModel". The Clamp default is bit-identical to before.
- `Compressor::setCleanBallistics(CleanBallistics::Decoupled)` (opt-in;
  Branching stays the default, state key `cleanBallistics`): the Clean
  character in feed-forward holds the gain target's peaks and smooths the held
  envelope. On a tone 14 dB over a 4:1 threshold at 5/100 ms, THD+N measures
  -102 dB at 1 kHz (Branching -65.5) and -42.8 dB at 30 Hz (-35.1), and the
  level sits on the peak-referenced static curve (Branching reads 0.75 dB
  less reduction). Attack and release t63 stay on the knobs. It follows the
  peaks of dense material, so the same settings compress more: 1.9 dB lower
  output on a drum programme at the defaults, 4.5 dB on a noise-like
  programme at -24 dB, 3:1.
- `ZeroLatencyConvolver::getNumTailLevels()`: the number of partition sizes in
  the tail (see the Changed entry).
- Run-time AVX2 dispatch: x86-64 builds for the generic baseline also compile
  AVX2+FMA copies of the vector kernels (the FFT passes, `firCorrelate`,
  `dotProduct`, `complexMulAccum`) and select them on CPUs that support them.
  Measured on MSVC x64 against the SSE2 build: FFTReal 2048 1.5x, uniform
  convolution 1.8-2.2x, 4x oversampling 1.5x. `simd::setRuntimeDispatch(false)`
  keeps renders bit-identical across CPUs; `DSPARK_NO_RUNTIME_DISPATCH` compiles
  the baseline only.
- `Mp3File` decodes MPEG-2 and MPEG-2.5 Layer III (16, 22.05, 24, 8, 11.025
  and 12 kHz, ISO 13818-3), with LSF scalefactors and intensity stereo.
  Encoded files at every rate null against a reference decoder at 131 dB or
  better.
- `WavFile` reads RF64 and BW64. `setRf64Mode()` writes RF64 always, or
  promotes the file only when it outgrows 4 GiB.
- `WavFile` reads G.711 A-law and mu-law (also inside WAVE_FORMAT_EXTENSIBLE),
  IMA/DVI ADPCM and Microsoft ADPCM (mono and stereo), and PCM whose depth is
  not a whole number of bytes (12-bit, 20-bit) in the container nBlockAlign
  defines. Every coding decodes bit for bit against a reference decoder on
  encoder-produced files. Microsoft ADPCM uses the fmt chunk's coefficient
  table and truncating prediction, the common decoder convention (flooring
  it moves predictors 3 to 6 by a few LSB). A fact chunk ending inside the
  last ADPCM block trims its padding. These codings report `bitsPerSample`
  16; `getEncoding()` tells them apart. A 5-minute stereo file reads in
  66 ms (IMA) and 115 ms (Microsoft ADPCM). Default audio of existing PCM
  and float files is unchanged.
- `Mp3File` decodes MPEG audio Layer I and Layer II (MPEG-1 and the MPEG-2
  low sampling frequencies; mono, stereo, dual channel and joint stereo; CRC
  protected or not) and free-format streams of all three layers. Encoder
  output and random streams that use every allocation class, scalefactor
  selection and joint-stereo bound decode at 130 dB or better against a
  reference decoder. A free-format frame length is accepted only when a third
  header confirms it and is never shorter than a frame at the lowest listed
  bitrate, so header patterns in other data cannot inflate the decoded size.
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
- `WavetableBank::readBlock()`: the block form of `read()` (an array of
  phases, one frame position for the block or one per sample), interpolating
  four float or two double samples at a time on SSE2 and NEON and eight or
  four with AVX2 (chosen at run time on x86-64). It returns `read()`'s values,
  operation for operation (bit for bit where the compiler does not contract
  multiply-adds). At 440 Hz with the level crossfade active, float, phase
  accumulation included: 2.3 ns per sample with AVX2 and 3.0 ns with SSE2,
  against 14 ns for `read()`; 4.1 ns with AVX2 when the frame position moves
  every sample. On one level (no crossfade) 1.9 and 2.4 ns, against 2.5 ns
  for a 4-point cubic reader of one table. The SSE2, NEON and AVX2 kernels
  share one source over each instruction set's lane operations.
  `Synthesizer`'s wavetable engine reads through it: 28.7 to 14.7 ns per voice
  and frame (one oscillator), 153 to 52 (two oscillators, unison 4), output
  unchanged.
- `PitchCorrector::setLowLatency()` (opt-in, default off; state key
  `lowLatency`): a pitch-synchronous splice engine instead of the phase
  vocoder. Latency 1024 samples at 44.1/48 kHz (21.3 ms at 48 kHz, the same
  span at every rate) against 4096. The read head refines the pitch where it
  plays, so a hard snap holds a +-40 cent vibrato on its note at 1.3 cents rms
  (the default engine 36 cents, a reference time-domain corrector 5.5), and a
  corrected sustained vowel keeps an 87-90 dB harmonic-to-residual ratio
  (vocoder 39-57 dB). At zero correction it is an exact delay. With
  `setFormantPreserve(true)` it renders by pitch-synchronous overlap-add and
  keeps the spectral envelope: a G3 vowel corrected five semitones up keeps
  its envelope centroid within 8% and its spectral centroid within 1% (the
  read head +20% and +25%), at a 59-69 dB harmonic-to-residual ratio and 49
  ns per sample. The default engine and its output are unchanged
  (bit-identical).
- `Resampler::setOfflineEngine(OfflineEngine::TwoStage)` (opt-in; `Direct`
  stays the default, so default audio is unchanged): `process()` and
  `processRange()` band-limit the input by FFT convolution (at the input rate,
  or at twice it when the lower Nyquist is above a quarter of the input rate)
  and change the rate with a 22-34 tap kernel. Every tier keeps Direct's
  passband edge, image and alias rejection and in-band tone residual, or
  improves them (Ultra, 96 to 44.1 kHz: alias of stopband tones -218 dB
  instead of -209). Double, ns per input sample once prepared, into caller
  storage, Direct to TwoStage: Ultra 26.2 to 3.3 (96 to 44.1 kHz), 26.9 to
  5.5 (48 to 44.1), 27.7 to 6.2 (44.1 to 48), 60.3 to 9.3 (44.1 to 96); High
  16.4 to 3.0, 16.6 to 5.0, 20.4 to 5.4, 42.1 to 7.9. Both instantiations
  compute in double, so float output is limited by its own rounding (48 to 96
  kHz High, 20 kHz tone: image residual -162.9 dB instead of -141.0). On
  in-band material the two engines agree to -152 dB (High) and -222 dB
  (Ultra); `processRange()` spans agree with each other and with `process()`
  bit for bit; `getReach()` follows the engine in force. New constant
  `Resampler::kMaxTwoStageOrder`.
- `Resampler::process(input, inputLength, output, outputCapacity)`: the
  offline conversion into caller storage, the vector form's samples bit for
  bit (truncated to the capacity). It allocates nothing once the offline
  engine is designed.
- `PitchDetector::setNoiseSuppression()` (opt-in, default off; the output is
  unchanged off): a steady window that reads unvoiced is read again with its
  noise floor removed by spectral subtraction (the floor estimated per band
  from the window itself). Harmonic tones of 70 to 1000 Hz at 0 dB SNR, 400
  tones, readings within 50 cents: white noise 77% right and 9% wrong (off:
  unvoiced; the frame stage of the probabilistic-threshold method 29% and
  71%), pink noise 75% and 3%; 5 and 10 dB unchanged or better. The reading
  keeps the input's own confidence (about 0.5 at 0 dB), so a confidence gate
  still ignores it. Band-limited noise alone reads voiced more often (a Q 2
  band at 300 Hz 0% to 11% of windows, at confidence up to 0.61). CPU on
  white noise 20 to 91 ns per sample; voiced windows are not read twice.
- `MidiFile::read(path, MidiReadOptions)`: opt-in, per-defect recovery of
  malformed files (`MidiReadRecovery`: redundant VLQ bytes, data after the
  end-of-track event, a wrong track length or count, a missing end-of-track
  event, unterminated SysEx, trailing bytes), each reported by
  `recoveries()`. Of 18 defect files it reads all 18, where a reference reader
  reads 12 (one without its second track) and the strict reader none. The
  default read stays strict and unchanged.
- `MidiFile` SMPTE time division: `MidiReadOptions::acceptSmpteDivision`,
  `createSmpte()`, `isSmpteDivision()`, `smpteFramesPerSecond()`,
  `ticksPerFrame()` and `division()`. Ticks are exact fractions of a second
  (30000/1001 frames per second for the 29 code) and tempo events do not
  affect timing. Off by default, so a default read still guarantees
  `ticksPerQuarter() > 0`.

- Plugin layer, CLAP: every parameter is modulatable
  (`CLAP_PARAM_IS_MODULATABLE`). A host modulator's global
  `CLAP_EVENT_PARAM_MOD` amount, in plain units, adds to the value
  `setParameter()` receives, clamped and snapped, at its position in the
  block like automation; the parameter's own value (host display, state,
  editor) stays unmodulated. Host modulation used to have no effect.

### Changed

- `Limiter` true-peak mode reads every sample interval with a 64-tap 4x
  reconstruction refined to its local maxima, maxed with the BS.1770 Annex 2
  estimate it used alone before (which under-reads peaks built from content
  near the top of the band). On an independent 8x, 128-tap reconstruction
  at the defaults the output now reaches +0.06 / +0.14 dB over the ceiling on
  dense mixes driven 12 / 6 dB over it (was +0.93 / +0.57), +0.13 dB on a
  hard-clipped 5 kHz tone (was +0.77), +0.27 dB on differentiated white noise
  (was +1.26); an Annex 2 meter still reads at or below the ceiling. True-peak
  output changes accordingly (the clipped 5 kHz tone comes out 0.64 dB lower,
  dense mixes 0.07 to 0.11 dB); sample-peak mode is bit-identical. A reading
  reaches 33 samples back instead of 11, so the attack ramp is 22 samples
  shorter (60 Hz + 7 kHz intermodulation at the 2 ms default: -58.9 against
  -61.6 dBc) and the lookahead is at least 49 samples in true-peak mode
  (`getLatency()` reports it). Cost: 35 against 18 ns per stereo frame.
- `CrossoverFilter` linear phase (default audio moves within rounding: at most
  2.7e-7, -128.6 dB, on noise): the 4096-tap FIR runs in eight partitions of
  512 instead of four of 1024, and the engine delays its output by two
  partitions, so the work for each output partition is spread over the
  partition before it. The latency is unchanged (3072 samples at 48 kHz).
  The last band is the delayed input minus the other bands (no convolution),
  and kernel partitions whose taps sum below 1e-6 (-120 dB) are not
  convolved: at the default splits two to four of eight remain per band. The
  bands match the ideal Linkwitz-Riley magnitudes within 7.2e-7 (2.0e-7
  before). Measured at 48 kHz stereo, default splits, LR24, median of nine
  heap layouts, us per block (mean / worst steady / worst with a move every
  37 blocks; two runs): 12 bands at 64 samples 2.8 / 75-89 / 47-52 became
  2.3 / 6 / 9-12, at 512 samples 22 / 46 / 55 became 18 / 20-22 / 36-38; 4
  bands at 64 samples 1.06 / 17.6 / 17.9 became 0.76 / 2.6-3.3 / 3.9-4.6,
  at 512 samples 8.3 / 18 / 18.6 became 5.6 / 6.3-7.5 / 12. A redesign is
  paced at a fixed share of the stereo work per sample (its long inverse
  transform split into passes), so a change is heard 85 ms later at 64 and
  512-sample blocks in mono, stereo or six channels (before: 21 ms at
  64-sample blocks, where the design rode on the blocks between partitions,
  85 ms at 512, 149 ms in mono at 512). In linear-phase mode the IIR
  coefficients are no longer rebuilt on every change (they are rebuilt when
  the mode switches back).
- `TubePreamp` `GridModel::Conduction` (opt-in) now models blocking
  distortion: each grid is fed through a 22 nF coupling capacitor and a
  1 M ohm grid leak, so grid current charges the capacitor and biases the
  stage colder after loud passages. After a 300 ms burst at +24 dB a quiet
  tone comes out 15.6 dB below its settled level 2 ms later and recovers
  within about 40 ms (before: 0.4 dB). At +24 dB and above the stage spends
  most of the cycle near cutoff: one stage, 1 kHz at -6 dBFS, the second
  harmonic rises to -3.9 dBc (previous Conduction model -26.8, Clamp -21.2)
  and the level falls 1.7 dB against the previous Conduction model. The
  default Clamp model is bit-identical.
- `Saturation` `Antialiasing::SecondOrder`: every algorithm is delayed by the
  same min(factor, 3) processing samples inside the pipeline (previously
  one, padded after the algorithms). The output of the algorithms whose
  kernel did not change is bit-identical except during an algorithm
  crossfade, where the fade now meets the delayed outputs.
- `wdf::DiodePairRoot` and `wdf::DiodeRoot`: at low drive (incident waves up
  to about 45-55 mV with the default 1N4148 constants) the root is an exact
  series reversion of the diode equation, without transcendental functions:
  14.7 against 30 ns per sample for the pair in an RC clipper, 11.9 against
  26.6 ns for the single diode. The output moves by rounding only (at most
  4e-16 relative against 50-digit roots).
- `OnsetDetector` default picker is gated, default delta 0.03 to 0.0035
  (analysis readout; no audio). The figures below are measured against an
  intermediate development default (plain picker, delta 0.015); the released
  delta 0.03 recalled fewer soft attacks still (30 ms attacks 0.428, 60 ms
  0.214 on an earlier version of the corpus). The threshold is twice the
  moving mean plus the delta, and a peak is kept only when the frame energy
  0.43 analysis frames after it has not fallen 2 dB below the energy 0.2
  frames before it
  (no fall at all within 0.0045 of the threshold), a decay already under way
  excused. On 24 held-out files per kind (F at +/-50 ms, false onsets per
  second): 30 ms attacks 0.788 to 0.966, 60 ms 0.603 to 0.909, note releases
  into silence 0.06/0.23 to 0.00/0.00 false per second, percussion with quiet
  hits inside loud decays 0.948 to 0.965, slow attacks over pink noise 0.471
  to 0.621, clicks over 0 dB pink noise 1.000 to 0.995 (0.02 per second).
  Causal latency unchanged. `setPicker(Picker::Plain)` with
  `setThreshold(0.03f)` restores the released output. A tremolo false-onset
  rate measured during development (1.85 per second) came from a corpus whose
  vibrato swept by up to 57%; with the phase integrated the tremolo kind
  reports none at either setting.
- `BeatTracker` causal beats follow the metrical level of a comb-filter
  stage (autocorrelation of the last 6 s of the envelope, mean of four
  multiples, Rayleigh prior at 0.45 s, Gaussian transition) folded onto a
  coherent subharmonic; the published tempo and confidence are unchanged.
  Causal beat F: syncopated grooves at 100/124/140 BPM 0.649/0.377/0.814 to
  0.980/0.984/0.986; ballroom 0.625 to 0.706 and 0.647 to 0.728 on the two
  halves of 698 recordings; Salsa 0.286 to 0.326; click trains 40 to 240 BPM
  unchanged. CPU 1.38 to 1.69 ms per second of audio.
- `FilterEngine` glides (default audio changes while a cutoff, Q or gain
  moves; settled output is bit-identical): the cutoff follows a monotonic
  cubic in log frequency that arrives exactly 165 ms after a change (10-90%
  in 100 ms) instead of a critically damped glide that took 331 ms to come
  within 1 cent, and the chunk-rate coefficient designs are interpolated
  sample by sample. On a 10 kHz to 500 Hz glide over noise the output is
  -94.5 dB from a per-sample design (-39.1 dB before); a tone under a glide
  carries chunk-rate sidebands of -120 dB at most (-55.7 dB before). The
  gliding path costs 6.2 instead of 9.4 ns per sample.
- `OfflineHardClipper`/`OfflineSoftClipper` at oversampling factor 1 (not the
  default 4; factors 2 and above are bit-identical): the clipping residual of
  a 32-sample windowed-sinc reconstruction is integrated against five
  moments per sample interval and band-limited by a continuous moment
  kernel, flat to 0.40 fs and below -65 dB from 0.47 fs on all of its
  images, so distortion folded across Nyquist is removed wherever it lands
  (a cubic B-spline average with inverse compensation left the third
  harmonic of a 10 kHz sine at 18 kHz). Sine clipped by 6 dB, full-band
  alias floor: 10 kHz at 48 kHz -95.0 dB (-24.3 dB before), 5 kHz at
  44.1 kHz -107.6 dB (-37.3), 5 kHz at 48 kHz -106.4 dB (-26.6 in 1.8.1),
  15 kHz at 44.1 kHz -91.4 dB (-36.6); Sine and Tanh curves at 10 kHz
  -97.0 and -99.0 dB (-25.3 and -26.9). Harmonic error below 20 kHz against
  the exact clipped sine: -84 to -111 dB (-28 to -69 before). Distortion
  above 0.40 fs is attenuated with the aliases it cannot be told apart from
  at the native rate. Factor-1 renders take about half the time (3 s of
  stereo drums, Hard: 0.23 instead of 0.43 s): the source-band projection,
  an identity at factor 1, is no longer computed.
- Rational-knee integration (the `StereoGenerator` color at every factor):
  beyond the knee the curve has a pole just outside each segment that starts
  at a knee crossing, and the bisection descended toward it. Its simple-pole
  part is now integrated in closed form and the smooth remainder by the same
  rules and tolerance. Color integration on +6 dB noise: 859 instead of
  2878 ns per sample; `StereoGenerator` at 48 kHz, factor 1: 1267 instead of
  2047 ns per stereo frame at +6 dB and 1456 instead of 3074 at +12 dB
  (factor 2: 1537 and 1852 instead of 2028 and 3045). Output moves by at most
  1.1e-15 (-282 dB re peak) on loud noise; moments match an independent
  graded reference to 5e-16. The GoldenRatio offline curve, with its wide
  knee, rarely meets the case: its default-factor renders of drums and noise
  are bit-identical.
- `StereoGenerator` at its default factor 1 runs its two FIR banks as single
  interleaved passes: 375 instead of 590 ns per stereo frame at low level
  with the AVX2 dispatch (728 instead of 1004 on SSE2). Float output is
  bit-identical over 10.75 M measured samples; double output differs by at
  most 4.4e-16.
- `SmoothedValue` Chase: 99% of a step now arrives at the ramp time at every
  sample rate (it took about 171 ms whatever the ramp time). No built-in
  effect uses Chase.
- `DspMath`: each fast approximation documents its measured error bound,
  range and speed; no code change.
- `wdf::DiodePairRoot` and `wdf::DiodeRoot` return the root of the diode
  node equation to rounding at every drive. Newton stopped at an absolute
  residual of 1e-10 V, so small signals came back with up to 2.4e-4 relative
  error (2.2 k ohm source, 1N4148). A table-seeded Wright-omega estimate,
  corrected by a Newton or Halley step, replaces it: 31.6 ns per sample at
  8 V peak (148 before), 29.7 ns at 0.01 V (64 before), error -313 dB re
  peak (-191 to -216 before). Output moves by at most that former error.
- `Compressor`: the feed-forward block path runs its stages as passes over
  32-frame chunks, with only the ballistics recursion serial. Stereo, 512-frame
  blocks: Clean Peak 37.9 to 7.9 ns/frame, RMS 48.7 to 15.7, Hilbert 77 to 34,
  Opto 45 to 9.1; FeedBack Peak 41 to 25 and FET 59 to 33. Output within
  rounding (max difference -144 dBFS on a drum programme).
- `ZeroLatencyConvolver`: the tail runs in growing partitions (1024 doubling
  up to 8192, the number of levels chosen per IR length for the least work
  per sample), each level's work spread over its block. A 20 s IR takes 122
  partitions in 4 levels instead of 936: mean cost per 64-sample block 10.4
  to 2.9 us, largest block 34 to 16 us. Latency stays zero; the output
  differs within rounding (null against an exact convolution -132 dB, was
  -126 dB with a 6 s IR).
- `OfflineSoftClipper<float>`: the analysis padding is sized for a 1e-9
  omitted tail instead of 1e-10 (`kTailTolerance`). The output differs by at
  most 3 ulp, at -149 dB re peak, and Sine/Tanh renders run 2.2x faster on
  bright material (1.4x on drums). Double output is unchanged.
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
- `SampleReader` (the zone reader of `Sampler` and `Synthesizer`'s sample
  source): at or below the recording's rate the two polyphase rows are
  blended and both output channels' dot products taken in one vector pass
  (AVX2 at run time on x86-64), with every sum formed in the order of the
  scalar form, so the output is unchanged. Per stereo float frame, the reader
  alone: 37.7 to 13.3 ns with AVX2, 17.4 ns with SSE2; double 41.8 to 17.7
  (AVX2). A `Sampler` voice two octaves down: 39.6 to 26.5 ns per frame.
- `GranularProcessor`: the grain cloud is loudness neutral at its defaults.
  Each grain's gain now compensates the Hann envelope's mean square (3/8) and
  the equal-power pan's (1/2), so the wet path is 7.27 dB louder than before
  (the same factor, sqrt(16/3), at every setting), and the defaults measure
  within 0.3 dB of the dry input on white and pink noise, a chord and a sine
  (they measured -7.4 dB). The overlap behind the gain now counts a grain's
  actual length and the 64-grain pool, so capped clouds also stay within
  0.7 dB (a 500 ms, 200 grains/s cloud measured -9.5 dB). With jitter, pitch
  and pitch jitter all at 0 the grains add in amplitude: +1.1 dB at the
  default density.
- `PitchDetector` reads pitch in noise. The period is chosen by a vote over a
  Beta distribution of thresholds (the frame stage of Mauch and Dixon 2014;
  `setThreshold()` sets its mean, default 0.1 as before), and a dip up to the
  voicing ceiling (0.30 at the default) reads voiced when the window is
  steady. Measured at 48 kHz on harmonic tones of 70 to 1000 Hz in white
  noise, readings within 50 cents: 87.4% at 5 dB SNR (0% before: every frame
  read unvoiced) and 92.0% at 10 dB (88.5%), with octave errors at 10 dB down
  from 6.8% to 0; a clean-room port of the method's frame stage scores 87.3%
  and 92.0%. At 0 dB SNR the detector still reads unvoiced by default (see
  `setNoiseSuppression()`). White and pink noise stay unvoiced; resonant
  noise (Q 5 to 10 band-pass) now reads its centre in 2% to 73% of frames,
  at confidence 0.94 or less.
  Default output change: steady clean tones read bit-identically; on clean
  material with sweeps and legato note changes 98% of readings are
  unchanged, confidently wrong readings fall from 127 to 34 of 66090 (an
  octave error at a high note whose period falls between lags is gone), and
  a window across a legato note change whose dip lies at a longer period the
  two notes share (four periods of one, five of the other) reads unvoiced
  rather than that period: wrong readings 184 to 52, every correct reading
  kept. A fundamental below the register no longer reads as the lowest lag
  (46.9 Hz at 48 kHz, 25 cents above F#1).
  Noisy readings carry their own confidence (about 0.76 at 5 dB SNR), so
  `PitchFollower` at its default gate of 0.85 keeps ignoring a noisy tone; a
  gate of 0.7 tracks a 220 Hz tone at 5 dB SNR within 15 cents.
  `PitchCorrector`, which acts on any voiced reading, now corrects a tone at
  5 dB SNR (in tune 98% of the time, against 0% before); on clean notes its
  steady stretches stay in tune as before and one frame of 180 at a note
  change differs. CPU: 6% less on clean tones (a redundant energy pass is
  removed and the candidate scan stops once the choice is settled), 1% to 3%
  more on noise.
- `Resampler::prepare()` designs its kernel tables 2.6x faster (Ultra, 96 to
  44.1 kHz: 4.1 to 1.6 ms; High 2.2 to 0.8 ms) from one symmetric
  fine-grid evaluation per table. `Resampler<float>` output is bit-identical;
  `Resampler<double>` moves at rounding level (-295 dB re peak at most).
- `ChordDetector` evaluates its note bank on a cascade of half-band
  decimators, one octave per rate: 304 to 2.1 us per frame (float, 44.1 kHz,
  4096), 614 to 2.9 us at 96 kHz, 1214 to 5.7 us with 16384. Chroma within
  0.008 dB of the direct evaluation on pitch classes within 30 dB of the
  loudest; held chord unchanged on every frame measured. The analysed span
  now ends `getAnalysisDelaySamples()` before the frame boundary (770 samples
  at 44.1 kHz). `KeyDetector`, which reads the same front end, drops from 6.65
  to 0.06 ms per second of audio with unchanged estimates at its default.
- `BeatTracker` causal beats are placed by a cumulative beat score instead of
  the resonator's phase turn, and a different metrical level must outscore
  the one in force for four of its beats before the beats follow it. On
  swung and dotted clicks with off-beats up to 0.7 of the beat, 659 of 660
  beats land within 70 ms (374 of 672 before) at a mean 3 ms late (11 to 105
  ms early before). Causal beat F-measure: 0.611 to 0.625 and 0.629 to 0.647
  on the two halves of 698 ballroom recordings, 0.257 to 0.286 on Salsa.
  Tempo and confidence readings are unchanged.
- `MidiFile::read()` counts each track's events in an allocation-free pass
  and then reserves exactly that many: a 1.9 M-event file parses in 33 ms
  instead of 65, with peak memory 83 MB instead of 132 (commit 80 MB instead
  of 160). No input needs more memory than before, and the parsed documents
  are identical.
- `FlacFile` decodes 2.7 to 3.1 times faster with bit-identical PCM and the
  same validation: Rice codes are taken from a 64-bit register and each
  sample is restored as its residual arrives, the frame CRC-16 runs eight
  bytes per step, and channel decorrelation drops overflow checks that could
  never fire. A 5-minute 16-bit stereo file reads in 93 ms (was 255 ms; a
  reference decoder takes 129 ms from memory), and opens in 110 ms with a zero
  MD5 (was 272 ms) or 191 ms when the MD5 is verified (was 395 ms). 24-bit,
  48 kHz: 120 ms (was 375 ms).
- `Mp3File` decodes Layer III 3.4 to 3.6 times faster: Huffman pairs and
  quads come from a 64-bit bit register through two-level code tables with
  branch-free signs, requantization stops at the last coded line, and the
  IMDCT and polyphase synthesis run on vectors in double precision (two time
  slots per DCT, or four in the AVX2+FMA copy selected at run time). A 30 s
  128 kbps stereo file opens in 17.7 ms instead of 60.6 ms, 320 kbps in 21.5
  ms instead of 76.0 ms. With the runtime dispatch off the output is
  bit-identical to before; the AVX2+FMA copy differs by at most one rounding
  of the last double bit (float output identical on encoder-produced files,
  within 6e-11 elsewhere).

- `Mp3File` reads stream: `openRead()` loads and indexes the compressed
  file and decodes nothing, and `readSamples()` decodes only the frames its
  range covers, so the decoded PCM of a whole file is never held (a 30 s
  stereo file needed 10.6 MB of it, five minutes 106 MB). A read that does
  not continue the previous one restarts two frames earlier with the bit
  reservoir rebuilt from the frames before, and returns exactly the samples
  of a decode from the start: output is bit-identical to before for every
  range, sequential, reversed or random, on a 90-file corpus of all three
  layers, both MPEG versions and both kernel copies. The synthesis windows
  a whole group of slots per load of the window and keeps the DCT in
  registers (25% less time per slot with AVX2, 27% on the SSE2 baseline).
  Opening a 30 s 128 kbps stereo file and reading it in 4096-frame chunks
  takes 14.2 ms instead of 17.2 ms (320 kbps: 17.5-17.7 ms instead of
  20.8-20.9 ms), 0.91-0.98 times a frame-by-frame float decode loop of a
  reference decoder over the same bytes in memory (stereo 128, 320 and
  VBR, 22.05 kHz; 0.63 mono, 0.67 Layer II), where it was 1.09-1.26 times;
  reading it whole into a new buffer takes 15.6 ms instead of 18.3 ms. No
  audio change.
- Plugin layer: processing start clears the processing history. VST3
  `setProcessing(true)` and CLAP `start_processing()` now call the
  plugin's `reset()`, as VST3 `setProcessing(false)` and CLAP `reset()`
  already did, so processing that starts again without a stop in between
  (or a CLAP device that sleeps and wakes) no longer resumes the tails and
  delay lines of the previous run.

### Fixed

- `Phaser`: a stereo spread change jumped the right LFO phase in one sample;
  its second difference reached 1311 times the effect's steady value. The
  offset now glides over 0.1 s (1.25 times at most). Default audio changes
  only while the spread moves.
- `Oscillator` with PolyBLEP (the LFO duty of `Phaser` and `Chorus`): a reset
  or re-phased triangle was seeded on straight lines, up to 0.12 of full
  scale off the leaky integrator's real cycle, and eased back over seconds
  at LFO rates. It is now seeded on that cycle (within 1.1e-4 at 0.5-30 Hz).
  Triangle LFO output changes after a reset or re-phase; sine and minBLEP
  oscillators are unchanged.
- VST3: a plugin that opts out of sample-accurate automation receives every
  MIDI controller value through `IMidiMapping`, each with its sample
  offset (1 of 4 CC points in a block reached it before; CLAP and AU raw
  MIDI delivered all 4).
- AU: while initialized, a different `MaximumFramesPerSlice` is refused
  with `kAudioUnitErr_Initialized` and a different stream format or sample
  rate with `kAudioUnitErr_PropertyNotWritable`, so the host re-initializes.
  The unit accepted them: a raised slice size then failed every longer
  slice (silence), a new rate halved the reported latency (64 samples
  reported as 0.67 ms instead of 1.33 ms) and a new width reached
  `processBlock` unprepared. Re-applying the current value still succeeds.
- `Saturation<double>::getState()` compiles: the slew sensitivity is written
  as float like every other parameter (the double overload was ambiguous).
- `CrossoverFilter` linear phase: the FIR length followed the host's
  maxBlockSize, so the low-frequency response did too (band 0 of a 200 Hz
  LR24 split read -12.6 dB at 100 Hz with 64-sample blocks, -1.7 dB at 512,
  against the ideal -0.527). The FIR now spans about 85 ms (4096 taps at
  48 kHz, scaled with the rate) at every block size as a partitioned
  convolution: the response is the ideal one, the bands sum to the delayed
  input within 3e-8 (was 1e-3 at 64-sample blocks; close splits at
  100/150/225 Hz summed with 0.87 dB of ripple, now flat), and the output no
  longer depends on how the host slices the stream. The latency is fixed at
  3072 samples at 48 kHz (was maxBlockSize / 2): re-read `getLatency()`,
  also for `MultibandCompressor` in linear-phase mode. CPU: see the
  `CrossoverFilter` entry under Changed.
- `CrossoverFilter` linear phase: a change rebuilt every kernel inside one
  block (150 us for that block at 12 bands, 512-sample blocks, against 21 us
  for a steady block). A change is now designed over the following blocks
  into a second kernel set and swapped in whole (it used to be heard from
  the next block; pacing and landing time: see the `CrossoverFilter` entry
  under Changed). A live band-count change also left the bands it added
  without output for a partition, so the bands stopped summing to the
  delayed input (0.155 of error on random live changes): it now lands as
  one complete design, and the sum stays exact (3e-7).
- `CrossoverFilter` IIR: a frequency glide updated its coefficients every 32
  samples of each block, so the glide depended on the host's block size (up to
  0.14 difference against 512-sample blocks). The update grid now runs across
  blocks; output is unchanged at block sizes that are multiples of 32.
- `Compressor` FeedBack with the RMS detector: the loop ran faster than its
  window could report the result, so a 12 dB drop overshot the settled gain by
  3.0 dB (10 ms window, 1 ms release) and a rise over-compressed by 4.5 dB
  (0.5 ms attack), ringing for about 40 direction changes. The loop advance is
  now capped at the critically damped limit for the window (twice that on the
  attack side): with the default 10 ms window, release settles no faster than
  22 ms t63 and attack 14 ms, with no overshoot. Slower settings are
  bit-identical; at the default 5/100 ms a drum programme comes out 0.5 dB
  louder (less over-compression).
- `StereoGenerator`: a silent tail reaches exact zero. The band, color and
  DC recursions settled into a subnormal limit cycle near 1e-321 after the
  input stopped, so the output never returned to digital silence and the
  CPU cost rose 15-30% in silence.
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
- `Resampler::process()` returns an empty vector for a negative input length
  (and for a null input) instead of sizing its output from it as a huge
  unsigned count: -5 at 48 to 44.1 kHz threw `std::length_error` or
  `std::bad_alloc`, or aborted with exceptions off.
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
  held to a 0.1 LU grid: it agrees with an exact block-mode reference to four
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
