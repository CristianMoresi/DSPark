# Changelog

All notable user-facing changes to DSPark are documented here.

## [Unreleased]

### Added

- `TonalBalance::setQuality(Quality::Render)`: half the hop and the bank
  refined three times at every hop, for a bounce: the bank's response
  misses the corrections asked by 0.055 dB in place of 0.086, at about
  three times the cost and the same latency. `setQualityFollowsRender()`
  with `setOfflineRendering()` takes it by itself at the `prepare()` of an
  offline render (`getQualityInUse()` tells which). State keys `quality`
  and `qualityFollowsRender`.
- `TonalBalance::setLocalSlope()`: what a band lacks or has in excess is
  turned into decibels with the slope of the loudness curve at the band's
  own level, in place of a fixed exponent. Near the threshold in quiet that
  slope is steeper (twice the exponent 10 dB over it), so the fixed
  exponent overstated the difference: on a programme 40 dB down, the five
  bands at the two ends were asked for 0.60 dB in all and are asked for
  0.10 dB, and the middle for 0.22 dB in place of 0.35. State key
  `localSlope`.
- `TonalBalance::setListeningCompensation()`, -30 to +30 dB: compensates a
  listening level that much lower than the one the programme is balanced
  at. The ear loses the two ends first as the level falls, so each band is
  raised by the level times one minus the ratio of the mid loudness exponent
  to its own, and the total loudness stays: with 20 dB, +2.1 dB at 60 Hz
  and +1.9 dB at 12 kHz against -0.1 dB at 1 kHz, the loudness unmoved.
  State key `listening`.
- `TonalBalance::setResponse(Response::Common)` and `setResponseTime()`: a
  short integration per band (four periods of its centre, 8 to 100 ms) and
  then one response time for every band, so the curve moves together. On a
  programme that turns from dark to bright the correction covers 63 % of
  its way in 0.16 s at 150 Hz and 0.06 s at 8 kHz with 60 ms, against 1.29 s
  and 0.23 s by band (the default). State keys `response` and `responseMs`.
- `TonalBalance::setReference(Reference::EvenPower)`: the programme is
  compared against the loudness, band by band, of a programme with the same
  power in every band of the ear's scale at its own level, read through the
  same model, in place of a flat loudness. Such a programme is then balanced
  at any level (corrections under 0.03 dB at -50, -30 and -10 dBFS over the
  72 bands, where the flat reference asked for 3.1, 1.6 and 0.6 dB). State
  key `reference`.
- `TonalBalance::setFloor()`, 10 to 120 dB (default 30): how far under its
  loudest band, in the model's level, a band may lie and still be given back
  by the balance (see the fix below). State key `floor`.
- `TonalBalance::setRangeKnee()`, 0 to 6 dB: rounds the approach to the
  range, so the correction of a dip that deepens bends into its limit with
  no corner (its sharpest bend falls from 0.63 to 0.42 dB with a knee of
  2 dB). State key `rangeKnee`.
- `TonalBalance::getLoudnessRaised()`, `getLoudnessLowered()`,
  `getLoudness()` and `getCentroidShift()`: what the corrections do, for a
  display: the share of the loudness the raised bands gain and the lowered
  ones lose, the total, and how far its centre moves along the ear's scale
  (a dark programme at a full balance: 0.045 raised, 0.045 lowered, 0.7 Bark
  up).
- `ResonanceSuppressor::setMix()`: how much of the filter is applied, taken
  inside it (its gain becomes 1 - mix + mix * gain before the minimum-phase
  design), so the part left alone and the part reduced never meet in two
  phases: a line taken 11.15 dB down at a mix of 1 sits 3.90 dB down at
  0.5, which is the mix of its two amplitudes. State key `mix`.
- `ResonanceSuppressor::setKnee()`, 0 to 12 dB: a soft knee at the
  threshold. A line that hovers there moves a quarter less from frame to
  frame with a knee of 6 dB (0.60 to 0.45 dB). State key `knee`.
- `ResonanceSuppressor::setPersistence()`, 0 to 500 ms: the excess of each
  bin is averaged over that time while it grows and followed at once when
  it falls, so only what lasts counts. White noise read with no threshold
  loses 1.43 dB counted at once, 0.77 with 20 ms, 0.25 with 60 ms and 0.04
  with 150; a click keeps its peak (0.83 dB down at once, 0.16 with 60 ms)
  and a line that rings is taken as before. State key `persistence`.
- `ResonanceSuppressor::setQuietLevel()`, -120 to 0 dBFS: each bin is read
  as its amplitude plus that level, so quiet passages, tails and the noise
  of a room show no resonance. With it at -60 dBFS a line taken 11.1 dB
  down at full level is taken 2.3 dB down forty decibels lower and left
  alone sixty lower (without it, 11.15 dB at any level). State key
  `quietLevel`.
- `ResonanceSuppressor::setAuditoryWeighting()`: the amount weighed by the
  threshold of hearing in quiet (a third of its distance in dB to its lowest
  point comes off the weight): the same excess weighs 0.44 at 150 Hz and as
  before at 3 kHz. State key `auditory`.
- `ResonanceSuppressor::setEnvelopePasses()` and `setWidthTaper()`, on
  `SpectralEnvelope::setPasses()` and `setTaper()`: two passes of the
  envelope's smoothing keep its width and lose its tails (a line of 20 dB
  stands half a decibel further out; wide rises are read alike), and the
  taper narrows the smoothing above a corner (a rise of 10 dB an octave wide
  stands 0.9 dB over an envelope a third of an octave wide, and 0.1 dB over
  one tapered with an exponent of 1). `SpectralEnvelope::excess()` gives the
  spectrum less its envelope with its sign. State keys `envelopePasses`,
  `taperHz` and `taperExponent`.
- `SpectralFilter::setShortKernel()` and `ResonanceSuppressor::setShortKernel()`:
  the filter's impulse response cut to half the transform and tapered over
  its whole length. Of the response of a cut of 24 dB, the energy past half
  a window falls from -33 to -43 dB, at the same depth. State key
  `shortKernel`.
- `ResonanceSuppressor::setQualityFollowsRender()` and
  `setOfflineRendering()`: with the first on, a `prepare()` made while the
  host renders offline takes the Offline quality whatever `setQuality()`
  says (`getQualityInUse()` tells which); the latency is the same in both.
  State key `qualityFollowsRender`.
- `ResonanceSuppressor::setRemovalBand()`: what `setRemovalListen()` plays,
  in one band alone (a line inside it is heard as in the whole removal, one
  outside it 34 dB lower).
- `SpectralFilter`: a key that is the programme itself is read as no key,
  with one transform per channel and hop instead of two (the suppressor
  keyed by its own programme: 378 to 357 ns per stereo frame, the cost of
  the unkeyed one).
- `TapeMachine::Speed::IPS_3_75`: 3.75 ips, with 90 us and the low-frequency
  constant in both standards. The head loses more of the top (at full losses
  10 kHz is 8.7 dB down, against 4.1 at 7.5 ips, 2.0 at 15 and 1.0 at 30),
  the head bump sits at half the frequency of 7.5 ips and the reference
  level stays where it is. The value comes last in the enumeration, so
  stored states keep their meaning.
- `TapeMachine::setRecordEmphasis()`, -6 to +6 dB: a high shelf around 4 kHz
  on the record side and its exact inverse on the reproduce side, as a
  record equaliser aligned against its reproduce twin. The round trip stays
  flat at low level (within 0.01 dB); the top end reaches the tape hotter or
  cooler (the third harmonic of 5 kHz at -6 dBFS: -29.1 dB at 0, -27.9 at
  +6, -33.1 at -6) and the tape's hiss between 8 and 16 kHz falls 5.4 dB at
  +6 and rises 5.5 at -6. State key `recordEmphasis`.
- `TapeMachine::setTrackFormat()` moves the head bump too: it sits where the
  wavelength meets the length of the head's contact with the tape (the speed
  over that length), and each format has its head: at 15 ips, 90 Hz with
  Half (as before), 72 Hz with Full and 112 Hz with Quarter.
- `TapeMachine::setOutputIron()` also takes the iron's low end, in a linear
  way: by the same share the output crosses a first-order high-pass at
  20 Hz (at the full amount 3.0 dB down at 20 Hz, 0.18 at 100 Hz), which
  removes by structure what DC the even term leaves.
- `TapeMachine` rests while it is not heard: with the mix settled on the dry
  side, or the source monitored, the machine does not run (3 ns per stereo
  frame against 3550 on the same host) and the output is the input, late by
  the latency, to the bit. Asked for again it starts from rest, runs unheard
  for its latency and comes in under the fade: no step (the largest between
  samples, 0.03266 against 0.03265 of the settled tone).
- `TapeMachine::prepare()` asked for another block size alone (same rate and
  channels) keeps the machine's memory: the output goes on to the bit, where
  it used to restart from silence. Any other call starts from rest.
- `BucketBrigade::delayFor()`: the delay to ask of the chain for the whole
  device (its two low-passes, its hold, the average over the step) to be
  late by a given time.
- `CrossoverFilter::setOrder()` takes 6 and 36 dB per octave as well, in the
  minimum-phase engine (both structures) and in linear phase. 36 is two
  third-order Butterworth filters per branch (6 dB down at the split, 36 dB
  an octave past it, the bands summing flat with a first-order and a
  second-order allpass per split above). 6 is a first-order low-pass and its
  complement: the bands add up to the input itself, with no phase
  correction. `MultibandCompressor` and `MultibandSaturation` take both.
- `MultibandSaturation::setMix()`: the sum of the bands against the programme
  as the crossover leaves it (the bands summed before their curves, as late
  as the slowest band), so any mix is as flat as the crossover, at the
  splits too (each band's own 20 Hz high-pass and DC filter still lead the
  phase at the bottom: 0.14 dB at 100 Hz at a mix of one half).
  `MultibandCompressor::setMix()` sets the mix of every band, which is the
  same thing there. State key `mix`.
- `MultibandSaturation::setOutputClip()` and `setOutputCeiling()`: a clipper
  four times oversampled after the sum, and no sample over its ceiling (six
  driven bands add up to several times full scale). While on it adds its
  latency, which `getLatency()` reports, and about a quarter of the cost.
  State keys `outputClip` and `outputCeiling`.
- `Saturation::setCombLoopClosing(true)`: the tuned feedback follows the
  level of the input and closes once there is no programme (under -70 dBFS),
  so a loop near unity does not ring on: with a loop of 0.95 at 200 Hz the
  ring 300 ms after the programme stops falls from -41 to under -120 dBFS.
  Off by default. State key `combLoopClosing`.
- `Compressor::setHoldMode(HoldMode::EveryPeak)`: the hold keeps every gain
  change for the hold time after it has passed (the deepest of a sliding
  window, exact to a 256th of the hold time). On a 50 Hz tone that dies away
  with a hold of 60 ms, the reduction of the default mode (`Deepest`, as
  before) goes deeper again 27 times, by up to 1.4 dB, each time a hold ends
  inside the wave; with `EveryPeak` it comes back in steps and never goes
  deeper again. It costs 5 ns per frame more. State key `holdMode`.
- `Compressor::setLookaheadShape(LookaheadShape::Ramp)`: the gain reaches
  each reduction exactly when its peak leaves the lookahead's delay, along an
  S-shaped ramp as long as the lookahead (the deepest gain change of the next
  lookahead, through two moving averages), and holds it for the lookahead
  after the peak. On a level that jumps 20 dB with an attack of 0.1 ms and a
  lookahead of 5 ms, the gain of the default shape (`Lead`, as before) falls
  2.8 dB per sample a whole lookahead before the jump arrives; the ramp moves
  0.12 dB per sample at most and has the settled reduction when the jump
  arrives (within 0.03 dB). State key `lookaheadShape`.
- `Compressor::setFeedbackKeyed(true)`: a feedback loop closed around the key
  ahead of the audio, so the lookahead (with its latency, its shape and the
  latency kept for it) and an external key act in `Topology::FeedBack` and in
  the characters that always detect in feedback. With the programme as the
  key and no lookahead it is the same compressor. Off by default (a feedback
  loop reads its output: no lookahead, an external key ignored). State key
  `feedbackKeyed`.
- `KeyMonitor` (in `Core/KeyEqualizer.h`): the key of a dynamics processor as
  it is listened to, late by the processor's latency and held under full
  scale.
- `SpectralFilter::getFramePower(channel)`: every hop reads the power of all
  the channels before the first filter is designed, so the design of one
  channel can look at the others of the same frame. `GraphicBank::setGains()`
  takes `kCarryRefinement` for the calls between two refined ones (the last
  refinement's correction is added). `TonalBalance::getLoudnessChange()`
  reads what the corrections leave of a change of the total loudness.
  `AuditoryScale::dominantPower()` takes a complex cross term, and
  `SpectralEnvelope::prepare()` the narrowest smoothing in bins.
- `BucketBrigade`: `prepare(rate, stages, corner, 2)` runs a line at twice
  the host's rate (the input interpolated half way between samples, the
  output read every other step after its low-pass), so the images of a clock
  near or below the host's rate are filtered as in the device instead of
  folding into the band; `processPair()` runs two lines on the same input
  and corner through one input low-pass, each giving to the bit what it
  gives alone; `setPrecision(Precision::High)` reads the input at the clock's
  ticks by a quintic through six steps and the held output through a
  quadratic B-spline over three steps (three steps more delay), which takes
  out the two artifacts of the model itself: the images of the input read
  by a straight line between steps, and those of the clock's harmonics that
  the step's average let fold back. Defaults are as before, to the bit.
- `Limiter::setPrecision(Precision::High)`: the sample-peak signal read at
  eight points per sample (a 16-tap Kaiser-windowed sinc gives the seven
  between the samples) and the gain computer's sliding minimum, hold,
  release and both moving averages run at eight steps per sample, so the
  corners of the gain fall where the crests are rather than on the sample
  grid; a crest within 0.01 dB of the reduction held restarts the hold.
  On a tone 6 dB over a -6 dB ceiling at 48 kHz (release 20 and 100 ms,
  with the default hold and without), the largest line that is no harmonic
  of the tone goes from -62 to -108 dBc to -145 to -146 at 100 Hz, from
  -75 to -85 to -117 to -126 at 1 kHz and to -123 to -128 at 5 kHz; the
  tone's level moves by 0.03 dB at most. Readings with only the steps, or
  with the eight points but no tolerance on the hold, left the lines at
  1 and 5 kHz, or at 100 Hz, where they were. The points concern the
  sample 7 frames back, so `getLatency()` is 7 samples longer; every sample
  still leaves at or below the ceiling; the per-sample path is the block
  path to the bit for mono; it can be switched during playback. With true
  peak on the true-peak reading feeds the gain computer and the precision
  does not apply. A stereo frame costs 105 to 159 ns (10 to 17 in
  Standard); `prepare()` sizes the gain computer's histories for eight
  steps per sample whatever the precision. State key `precision`. The
  expander and the gate were measured the same way and need neither part:
  the expander's analytic detector already leaves its lines at -111 to
  -152 dBc, and the gate's gain does not move on a steady tone.

- `FilterDesign` (new, `Core/FilterDesign.h`, with `AnalogSection` and
  `AnalogCascade`): filters described as analog sections and turned into the
  biquads whose magnitude stays on the analog curve up to Nyquist, for every
  shape (bells, shelves with a slope, low-pass, high-pass, band-pass, notch,
  first-order sections, tilt), not only the six that had a closed form.
  `FilterDesign::digital(section, band)` carries the poles over by impulse
  invariance, fits the numerator to the analog power by linear least squares
  (the squared magnitude of a biquad polynomial is a quadratic in 1 - cos w)
  and, where that is not within a hundredth of a decibel, moves numerator and
  denominator together in at most two more linear steps weighted for the
  relative error. It looks at sixteen frequencies across the band and at
  fourteen around the poles and zeros of the section. Zeros at DC and on the
  frequency axis stay exactly where the prototype has them and the gain at
  DC is held; an all-pass stays an all-pass. Worst distance to the analog
  magnitude from 20 Hz to 20 kHz over centres from 30 Hz to 20 kHz, Q from
  0.3 to 16 and gains to 24 dB (decibels, worst / mean; the same on three
  compilers): at 48 kHz bells 0.25 / 0.03 (closed-form matched design
  2.95 / 0.24, bilinear 11.5 / 1.7), shelves of any slope 0.10 / 0.01
  (0.44 / 0.08; 6.3 / 1.0), low-pass above -60 dB 0.16 / 0.07 (1.06 / 0.57),
  high-pass 1.27 / 0.18 (3.79 / 0.49), band-pass 0.37 / 0.20 (2.01 / 0.91),
  notch 0.42 / 0.09 (bilinear 13.7 / 3.3), first-order sections 0.28 at
  most (bilinear 9.1); at 44.1 kHz bells 0.52 / 0.06 (4.24 / 0.36); at
  96 kHz every shape within 0.06 (closed forms 0.03 to 0.43). The design is
  a smooth function of its parameters: a step of 0.01 dB of gain moves the
  response by 0.03 dB at most. A section costs 0.55 to 1.0 us where the
  numerator is enough and 1.4 to 2.6 us with the joint steps (the closed
  forms 0.06 to 0.10 us), with nothing allocated: it can run where the
  parameters change. `FilterDesign::Band` keeps the frequencies of the fit
  for one sample rate; above 20 kHz an error counts less and less.
  On top of the sections, the cascades of an equalizer as analog prototypes:
  `highCut()` and `lowCut()` of any slope from 0 to 96 dB per octave as a
  real number (whole orders are Butterworth to rounding; between them the
  corner stays at -3 dB and the cascade follows 1 / (1 + w^(2 order)) down
  to -36 dB within 0.8 dB between 6 and 12 dB per octave, 0.5 at 15 and
  under 0.1 from 21 up; under 6 dB per octave it is the fraction of a pole,
  within 0.15 dB; a step of 0.006 dB per octave moves the response by
  0.07 dB at most, across every whole order), with the resonance on the
  sharpest pair; `bell()`, `lowShelf()` and `highShelf()` of order 1 to 16
  (Butterworth: the gain at the centre, half of it at the edges at every
  order, no overshoot, a cut that is the exact inverse of the boost; the
  top within 0.1 dB over 7 % of the half width at order 1 and 84 % at
  order 16); `bandPass()`, `bandStop()` and `allPass()` of any order;
  `tilt()`, a straight line of -6 to +6 dB per octave across the whole band
  within 0.07 dB; and `ellipticHighCut()` / `ellipticLowCut()`, an elliptic
  wall (order 12, 0.05 dB of ripple and 100 dB of floor by default: the
  floor starts 1.17 times the corner away). As biquads at 48 kHz the
  cascades stay on their prototypes where a section by section bilinear
  transform is off by tens of decibels: a 30 dB per octave cut at 12 kHz
  0.12 dB (35), a 96 dB per octave cut at 16 kHz 0.36 (76), a bell of
  order 8 at 14 kHz 0.10 (6.9), the wall at 10 kHz 0.45 (40).
- `Equalizer::setAnalogResponse(true)` (off by default): every band is
  designed from its analog prototype by `FilterDesign` and runs in a
  `BiquadCascade` whose coefficients are ramped. The curve of a band no
  longer shrinks towards Nyquist in any type. Worst distance to the analog
  curve from 20 Hz to 20 kHz at 48 kHz, as played (cookbook; default;
  analog response, in decibels): a bell at 12 kHz, Q 2, +12 dB 3.04; 0.35;
  0.10. A bell at 15 kHz, Q 1, -18 dB 8.13; 2.14; 0.15. A high shelf of
  +12 dB at 12 kHz 1.88; 1.88; 0.03. A 24 dB per octave low-pass at 10 kHz
  30.9; 30.9; 0.17. A notch at 14 kHz 5.6; 5.6; 0.11. A band-pass at 9 kHz
  9.4; 9.4; 0.21. At 44.1 kHz the same bands stay within 0.43 dB (the
  default: up to 47.6), at 96 kHz within 0.01. In LinearPhase the kernel is
  built from the magnitude of the prototypes themselves: 0.00 dB in every
  case above (the default kernel: up to 5.6). The engine is also the cheaper
  one: eight bands on a stereo block cost 22 to 26 ns per frame against 77
  to 143 with the cookbook engine on three compiler and machine pairs (mono:
  14 to 20 against 39 to 67). A change of a band glides instead of stepping:
  40 ms for a small move, 10 ms for a jump, with nothing depending on how
  the host cuts the stream; a block that takes a moved bell costs 2 us more,
  one that takes a bell of order 16, 23 us. A gain band within 0.05 dB of
  0 dB leaves the path (the samples pass untouched and it costs nothing)
  and comes back without a step. With the analog response the slope of a
  cut is any value from 0 to 96 dB per octave (`BandConfig::slope` is now a
  real number; the cookbook engine keeps running whole slopes from 6 to
  48), and a bell, shelf, notch or band-pass takes an order from 1 to 16
  (`BandConfig::order`: flat top and 6 dB per octave of side per order).
  Four band types are new and always run this way: `AllPass` (180 degrees
  per order), `ConstantTilt` (gain in dB per octave, a straight line across the
  band), `LowPassWall` and `HighPassWall` (elliptic, order 12: within
  0.05 dB up to the corner and 100 dB down from 1.17 times it; as biquads
  they go through the bilinear transform with the corner held, which keeps
  ripple and floor exact). `buildBandSections()` gives an interface the
  sections of a band in either engine (up to 16). The state carries
  `analog`, and `order` and `slopeDb` per band; older states load as before.
  A band type that is not one now keeps the band's own type instead of
  clamping to the last.
- `Equalizer` bands take a scope in the stereo pair
  (`BandConfig::scope`: `Stereo`, the default, `Left`, `Right`, `Mid` or
  `Side`), in either engine and in both phase modes. The left, right and
  stereo bands run first; then the pair is turned into middle and sides
  for the bands placed there, and back. In LinearPhase the same is one
  matrix of four kernels between the two channels, built in the background
  like the single kernel, so the latency does not change and a set of bands
  with no scope costs what it did. A band that changes place comes in
  through the 20 ms band fade. With one channel the left and middle bands
  act on it and the right and side bands do nothing; `processSample()` runs
  the left and right bands and leaves the middle and side ones to
  `processBlock()`, which sees both channels of a frame.
  `getMagnitudeForFrequencyArray()` takes the scope to draw (the stereo
  bands and the ones placed there). The state carries `scope` per band.
- `Equalizer::FilterMode::PrototypePhase` (new mode): the phase of the analog
  curve, 32 samples late. A minimum-phase digital filter that follows the
  magnitude of an analog one does not have its phase, because the analog
  curve goes on above Nyquist and the digital one folds there. In this mode
  every band runs from its analog prototype and one 64-tap filter per
  channel takes the bands the rest of the way: it is designed, each time a
  band changes, from the analog response over the digital one of every
  section, reduced to nothing between the top of the audio band and Nyquist
  and faded in over 20 ms. Worst distance in phase to the analog curve from
  200 Hz to 20 kHz at 48 kHz (minimum phase with the analog response; this
  mode): a bell at 12 kHz, Q 2, +12 dB 17.1 degrees; 0.08. A bell at
  15 kHz, -18 dB 34.0; 0.15. A high shelf of +12 dB at 12 kHz 24.2; 0.11. A
  12 dB per octave low-pass at 16 kHz 88; 0.34. A 24 dB per octave low-pass
  at 10 kHz 178; 1.05. A bell at 1 kHz 2.7; 0.01. The level gets closer too
  (the bell at 12 kHz: 0.104 dB; 0.003). At 44.1 kHz, where there is less
  room for the correction to fade out, 21 to 38 degrees become 0.5 to 0.9
  and the steep cut keeps 5.4; at 96 kHz everything is within 0.01. Eight
  bands on a stereo block cost 31.9 ns per frame against 23.8 in minimum
  phase, and a block of 64 that takes a moved band 8.9 us against 3.4. With
  placed bands the correction is a matrix between the two channels, as in
  LinearPhase. A set of bands at 0 dB is the signal, 32 samples late, to the
  bit. `getLatency()` reports 32 at any rate; `processSample()` runs the
  bands of the mode without the correction.
- `DynamicEQ::setMixedBands(true)` (off by default): a band is the signal
  and one fixed filter per side, the end of its range, and what moves is how
  much of the filter is heard, with the weight taken from the gain in
  amplitude so that the centre of a bell (the plateau of a shelf) sits on
  the gain asked for at every sample. No recursion has its coefficients
  moved and nothing is designed while the gain moves. The default engine
  designs the band again every 16 samples, which is a gain that moves in
  steps: a steady 6 kHz tone under a band that a key opens and closes 25
  times a second (attack 5 ms, release 20 ms) grows lines 3 kHz away at
  -65.6 dBc, and mixed they are at -103.9 (with 20 and 100 ms: -69.2 and
  -119.6; with 1 and 5 ms, where the gain itself is that fast: -63.2 and
  -79.6). The cost is the same (eight bands on a stereo block: 410 to 438 ns
  per frame against 418 to 456 on three compilers). Two fixed curves cannot
  be the nominal band at every depth, so the end is chosen for the band to
  be exactly the bell or shelf of its Q half way through the range in
  decibels, fitted to the analog curve up to Nyquist; a 12 dB cut of Q 1 is
  1.63 octaves wide at half its depth when it is 3 dB deep, 1.39 (nominal)
  at 6 dB and 1.00 at 12, and a boost the other way round (1.18, 1.39,
  1.90). A band that rests at 0 dB and is not acting returns its input bit
  for bit.
- `Compressor`, four controls, all off by default and working in both
  topologies and in `processSample()`: `setAutoThreshold(true)`: the
  threshold counts from the key's own recent peak (a reference that rises at
  once and falls 1 dB per second) instead of from full scale, so the
  compressor does the same on a take recorded 40 dB lower (with an absolute
  threshold it would not act there at all). `getThresholdInUseDb()` reads the
  threshold it is using (the recent peak plus the one set) from any thread. `setMaxLookahead(ms)`: a latency
  kept for the lookahead; the audio is always that late, `getLatency()`
  reports it, and `setLookahead()` moves inside it with the gain held back
  by the difference, so the lookahead can be automated under one delay
  compensation (0 keeps the latency equal to the lookahead, as before).
  `setAmount(0..1)`: the share of the gain change that is applied, as a
  scale of its decibels, with the same timing and nothing of the dry signal
  added back; at 0 the output is the input bit for bit, and in FeedBack the
  loop keeps hearing the whole reduction. `setSoftRange(true)`: the gain
  change approaches the range without a corner, as g / (1 + (g / R)^4)^(1/4)
  (1.5 % short at half the range, 16 % short at the range, never over it).
  The hold already froze the release and left the attack free; a test now
  says so. The state carries `autoThreshold`, `maxLookahead`, `amount` and
  `softRange`.
- `Compressor::setAutoMakeup(AutoMakeupMode::Analytic)`: a makeup worked out
  from the settings, with nothing measured. It is minus the gain the
  ballistics come to rest on over dense programme: the levels of a Gaussian
  signal as the selected detector reads them, each with its target on the
  static curve, and the followers resting where their attack and release
  pulls cancel (the memory stage of Opto and FET, the decoupled hold and a
  feedback loop each in their own way). A constant between edits, so it never
  pumps, and it moves to its new value over a few tens of milliseconds after
  one. It takes the programme at -18 dBFS RMS; with `setAutoThreshold(true)`
  the threshold counts from the programme's own peak and the result holds at
  any level, and it follows `setAmount()`. Works in Downward and in Upward.
  Level left over the input's on pink noise at -18 dBFS (dB): Peak detector
  within 0.1 over thresholds from -30 to -10 dB, ratios from 2 to 10, attacks
  from 0.5 to 100 ms and releases from 20 ms to 1 s, where the Static mode
  leaves -2.4 to +6.1; decoupled and feedback 0.2; RMS, Hilbert, TruePeak and
  Opto 0.3; FET 0.5; Upward 0.03; with the automatic threshold the same at
  -36 and at -6 dBFS (0.14 at most). On programme that swells 5 dB up and
  down it stays within 0.7, where the Adaptive mode is 0.2 to 1.2 off. The
  RMS window reads the low end of the programme: on white noise it is off
  by up to 1.3.
- `Compressor::setPrecision(Precision::High)` (off by default): the control
  signal is worked out between the samples, the audio staying at its rate.
  The Peak and TruePeak detectors read the key at four points per sample
  (the sample and three points before it, from a 16-sample windowed-sinc
  interpolation) and the ballistics take eight steps per sample towards a
  target that moves along the line between those points (between samples for
  RMS and Hilbert, whose reading is already smooth); the gain is the mean of
  the eight. In `processBlock()`, in feed-forward and in the feedback loops
  solved against the input (the Peak detector, Downward: the FET character
  among them). The interpolated detectors read 7 samples late and the audio
  waits for them: `getLatency()` reports 7 samples more with Peak or
  TruePeak, none with RMS or Hilbert. Largest line that is no harmonic of a
  tone 12 dB over a 4:1 threshold, 48 kHz, Standard to High (dBc): Peak
  detector at 0.1 ms / 50 ms -77 to -111 at 1 kHz, -66 to -86 at 5 kHz, -51
  to -85 at 15 kHz, -38 to -66 at 18 kHz; at 5 ms / 100 ms -83 to -130, -83
  to -98, -79 to -98 and -46 to -76; FET at 20 us, ratio 20, 30 dB over -68
  to -91, -50 to -66, -49 to -70 and -37 to -66; Clean in feedback -71 to
  -104 at 1 kHz and -39 to -72 at 18 kHz; decoupled -70 to -94 and -39 to
  -66; Hilbert -124 to -153 at 1 kHz. The gain of the tone stays within
  0.1 dB of Standard in every case. A stereo frame costs 125 ns against 17 in
  feed-forward and 190 against 52 with FET (x64 server core). Two things
  were tried first and dropped on measurement: averaging the rectifier and
  the static curve over three consecutive values (second-order antiderivative
  antialiasing) read a tone of the top octave up to 10 dB low, since the
  average over three sorted values is not the average along the waveform;
  and the eight steps alone, with the target on a line between samples,
  gave 10 to 30 dB up to 5 kHz and nothing at 18 kHz. State key `precision`.
- `TapeMachine`, the machine around the tape, all off or at the classic
  setting by default; the defaults and every earlier control give the same
  output as before, to the bit:
  `setTransport(Transport::Speed)` keeps the tape speed as a state: the
  reproduce pitch is the speed of the moment, whose deviation is a smoothed
  noise, the capstan's rotation (5 Hz at 15 ips, with a third of its second
  harmonic) and the reel's (0.6 Hz at 15 ips), at rates that follow the
  speed and deeper at slower speeds; the delay is held at the head distance
  over the speed and read through a four-point Lagrange interpolator, and a
  change of `setSpeed()` glides the speed (0.45 s up, 0.7 s down). A 3 kHz
  tone at full depth moves by 0.69, 0.54 and 0.38 % peak to peak at 7.5, 15
  and 30 ips (0.08 % at the default depth at 15 ips); switching from 15 to
  7.5 ips moves its pitch by 0.09 % at most.
  `setHum(dBFS, mains)` adds mains hum after the tape (50 or 60 Hz with its
  second harmonic 6 dB down, in phase on every channel).
  `setChannelBleed(dB)` leaks each channel of a pair into the other through a
  first-order high-pass at 10 kHz: -50 dB at 1 kHz is -70.0 at 100 Hz, -40.6
  at 3 kHz and -31.6 at 10 kHz.
  `setEvenHarmonics(0..1)` adds an even term before the tape: a 1 kHz tone at
  -12 dBFS carries its second harmonic at -40.3 and -34.3 dBc at 0.5 and 1
  (-118 without), 12 dB lower at -24 dBFS.
  `setOutputIron(0..1)` adds an output transformer as a parallel branch (a
  one-pole low-pass at 60 Hz, a curve with an even and an odd term, the
  inverse filter capped at 4 kHz): at 1, a 40 Hz tone at -6 dBFS carries
  its second harmonic at -41 dBc (-116 without) and its third at -28.7
  (-33.4), while a 1 kHz tone keeps its level and its third harmonic within
  0.01 dB.
  `setHeadModel(HeadModel::Contour)` makes the reproduce head's low end a
  chain of four resonances, the bump and three ripples above it, each
  designed on its analog prototype.
  `setElectronics(Electronics::Circuit)` runs the record and reproduce
  networks from their analog transfer functions at the internal rate, as
  normalized ladders, with a record amplifier limit at 45 kHz and a
  reproduce amplifier pole at 50 kHz (a fifth-order reproduce chain with the
  output coupling): within 0.01 dB of the ideal electronics at 1 kHz and
  0.5 dB lower at 20 kHz.
  `setAlignment(hfDb, lfDb)` trims the reproduce equalization as a machine's
  HF and LF adjustments: +3 dB moves 16 kHz by 2.99 dB, -3 dB moves 31.5 Hz
  by -2.80 dB (NAB).
  `setFormulation()` (Classic, HighOutput, Vintage) changes the material: a
  1 kHz tone at -12 dBFS carries 0.16, 0.07 and 0.25 % THD, and the hiss
  moves by 0, -1.5 and +3 dB; the bias keeps 0.5 as the optimum of each.
  `setTrackFormat()` (Half, Full, Quarter) moves the hiss by 0, -5.0 and
  +3.0 dB (the track width).
  `setCalibrationLevel(0..9 dB)` drives the tape harder at the same output
  level: at -12 dBFS the THD is 0.16, 0.31, 0.60 and 1.14 % at 0, 3, 6 and
  9 dB (6 dB gives the 0.4 to 0.8 % tape machines are specified at their
  reference level), and the hiss falls by as much.
  `setMonitor()` (PlayHead, RecordHead, Source): RecordHead reads through the record head
  (0.66 dB lower at 16 kHz), Source is the input delayed like the tape path,
  to the bit. With everything on the machine costs 6 % more.
  State keys `transport`, `humDb`, `mains`, `channelBleed`, `evenHarmonics`,
  `outputIron`, `headModel`, `electronics`, `alignHf`, `alignLf`,
  `formulation`, `trackFormat`, `calibrationLevel` and `monitor`.
- `AlgorithmicReverb`, around the tank, all off by default; the defaults and
  every earlier control give the same output as before, to the bit (Room,
  Hall, Plate, Spring and Cathedral in Eco):
  `setDecayBand(index, band)`, six bands of a parametric equalizer of the
  decay time (bell, low shelf, high shelf, each a frequency, a Q and a
  multiplier of the decay there): each line carries the band as a filter in
  its loop whose gain is -60 x (round trip / T60) x (1 / multiplier - 1) dB,
  and bands that lengthen the decay are scaled down together until no loop
  passes 0.9995. A 1.5 s hall measures 2.71 s at 2 kHz with a bell there at
  x2 (1.48 without), 0.78 s at 8 kHz with a high shelf at 4 kHz at x0.5
  (1.47) and 2.58 s at 125 Hz with a low shelf at 250 Hz at x2 (1.63).
  `setTailHold(true)` fades the tank's input and the early field out over
  20 ms while every loop's absorption and decay band fade to unity: the tail
  holds its level within 0.01 dB from one second to six and the balance of
  its 6 kHz and 500 Hz thirds of an octave within 0.8 dB.
  `setDryDip(dB)`, `setGate(threshold)`, `setGateHold()` and
  `setGateRelease()` shape the wet signal from the dry one (`TailShaper`):
  the tail dips by the amount set while noise plays (-6.00, -11.99 and
  -23.98 dB at 6, 12 and 24) and comes back over a second and a half (-0.13,
  -0.27 and -0.53 dB 1.5 s after the noise stops); the gate closes once the
  dry signal has stayed under its recent peak by the threshold for the hold
  time (a 2 s hall under -70 dBFS 0.44 s after the noise stops, with -30 dB,
  50 ms and 40 ms).
  `setSourceDepth(0..1)` moves the balance from the early reflections to the
  tail, shortens the pre-delay, adds input diffusion and darkens the wet
  signal: the first 80 ms of a hall against the rest go from +0.72 to
  -3.71 dB and 8 kHz against 500 Hz from -7.5 to -22.8 dB.
  `setTankLeveling(0..1)` compresses the tank's input (ratio up to 4 over
  -30 dBFS, a 24 dB knee, gain compensated at -18 dBFS): at 1 a quiet
  passage at -40 dBFS feeds a tail 8.7 dB louder and a loud one at -12 dBFS
  one 2.0 dB quieter; `setTankSaturation(true)` adds a soft saturation
  after it.
  `setTempo()`, `setPreDelayNote()` and `setGateHoldNote()` set the
  pre-delay (up to 500 ms) and the gate hold (up to 4 s) as note values,
  straight, dotted or triplet (a sixteenth at 120 BPM is 125 ms).
  `setInputWidth(0..1)` narrows the input before the tank (an inverted
  stereo pair is more than 60 dB quieter at 0), and
  `setWidthLaw(WidthLaw::PowerKept)` raises the middle as `setWidth()`
  lowers the side: the tail keeps its power within 0.2 dB from a width of 0
  to 1.41, where the side law loses 1.1 dB at 0.
  `setPostEqBand(index, band)`: six bands of a parametric equalizer of the
  wet signal (cuts of 6 to 48 dB per octave, bells, shelves, tilt; 20 ms
  ramps).
  `setModulationStyle(ModulationStyle::Segments)` moves every other line in
  straight segments toward random positions at random speeds instead of the
  smooth random modulation.
  `setSpaceLayout(SpaceLayout::Geometric)` spaces the line lengths evenly in
  log over the same range (about 6.6 per octave, the two sides interleaved),
  then to the nearest prime; the lengths glide to the new layout.
  `setShimmer(0..1)` pitches the tail up an octave with two read heads of
  random window length (30 to 60 ms) and feeds it back into the tank: a
  500 Hz tone's octave sits 85 dB under it without, 47.5, 41.5 and 35.5 dB
  under it at 0.25, 0.5 and 1, and the tail stays bounded over 20 s.
  The bands of the decay equalizer cost 3 % each (six: 17 %), the others up to
  11 % of the tank. Processing the tank in blocks was tried and dropped on
  measurement (blocks of 64 cost 7 % more than sample by sample). State keys
  `db0.shape` to `db5.mult`, `tailHold`, `dryDip`, `gate`, `gateHold`,
  `gateRelease`, `sourceDepth`, `tankLeveling`, `tankSat`, `tempo`, `preNote`,
  `preFeel`, `gateNote`, `gateFeel`, `inWidth`, `widthLaw`, `modStyle`,
  `layout`, `shimmer` and the wet equalizer's bands.
- `Reverb` (convolution): `setDryDip()`, `setGate()`, `setGateHold()` and
  `setGateRelease()`, the same shaping of the wet signal from the dry one.
  State keys `dryDip`, `gate`, `gateHold` and `gateRelease`.
- `TailShaper` (new, `Core/TailShaper.h`): a dip under the dry signal and a relative gate for a
  wet signal, driven by the dry one. Each follower's reference is the peak
  of its own envelope (the dip's falls over ten seconds), so the
  dip and the threshold are relative to the programme's own level; the
  dip's gain is exp(-depth x min(1, level / reference)) with a 30 ms
  attack and a 400 ms release, the gate's gain is smoothed twice, with a
  hold.
- `BucketBrigade` (new, `Core/BucketBrigade.h`): a bucket-brigade delay
  line of 64 to 8192 stages whose clock the delay sets (delay = stages /
  (2 clock)), modelled at the host's rate: the clock is a phase that
  advances by clock / rate per sample; at each tick the input, interpolated
  to the tick's time, enters the chain and the oldest bucket leaves it; the
  output is the last bucket held and averaged over each host sample (so the
  steps' images do not fold back when the clock is faster than the host),
  between fourth-order Butterworth anti-alias and reconstruction filters at
  a corner the caller sets.
- `Chorus::setModel(Model::Analog)` (off by default): the channels summed to
  mono, high-passed at 5 Hz, into two bucket-brigade lines of 512 stages
  whose clocks a slightly asymmetric triangle moves in opposition (in phase
  in the Vibrato mode), with the device's filters at a fifth of the slowest
  clock and the line's hiss (`setAnalogNoise()`, -84 dBFS by default), which
  grows with the delay; the dry signal keeps its stereo. `setAnalogMode()`
  loads three presets of rate, depth and centre (Slow 0.5 Hz, Fast 0.85 Hz,
  Vibrato 7 Hz in phase). From 176.4 kHz the engine runs at half the rate.
  Other lines in the band stay 81 to 90 dB under a 1 kHz input at 48 kHz
  and 99 to 104 dB at 96 and 192 kHz; a 9 kHz input leaves no other line
  within 73 dB of it. It costs 56 ns per stereo frame against 81 for the
  digital model. The digital model gives the same output as before, to the
  bit. State keys `model`, `analogMode` and `analogNoise`.
- `Vibrato::setModel(Model::Analog)` (off by default): each channel through
  a bucket-brigade line of 1024 stages whose clock the sweep moves, with the
  filters at a fifth of the slowest clock of the sweep and the line's hiss
  (`setAnalogNoise()`). At the defaults a 500 Hz tone keeps its level and
  no other line comes within 114 dB of it; a deep slow sweep (4 semitones
  at 1 Hz, a 74 ms peak delay) is darker, as on the hardware: 52 dB under
  the digital model at 3 kHz. It costs 131 ns per stereo frame against 35.
  The digital model gives the same output as before, to the bit. State keys
  `model` and `analogNoise`.
- `Delay::setCharacter()` (Clean by default, the same output as before, to
  the bit): `Character::Tape` moves the read point with a tape transport
  (`TapeTransport`): a change of the delay is a change of tape speed reached
  with the motor's inertia, so the pitch bends while it glides (300 to
  150 ms: the pitch 1.43 times up 0.1 s later, 1.08 at 0.5 s, within 0.4 %
  at 1.2 s),
  wow and flutter of the capstan (`setWowFlutter(0..1)`, 0.5 by default:
  0.14 % peak to peak on 0.1 s windows), the playback head's loss and the
  tape's hiss. `Character::Bucket` runs the line through a bucket-brigade
  device of 4096 stages, with its filters at 0.3 of the clock, so a longer
  echo is darker (2 kHz is 0.4 dB down at 150 ms, 6.6 dB at 300 ms and 52 dB
  at 600 ms; 4 kHz 0.4 dB down at 50 ms), and its hiss; the feedback goes
  round the device (echoes at 300.0 and 600.0 ms with a 300 ms delay) and
  the clock's images stay 68 dB under a 1 kHz tone at 300 ms.
  `setCharacterNoise()` sets the hiss of both (-84 dBFS by default). A
  stereo frame costs 171 ns with Tape and 80 with Bucket against 53. State
  keys `character`, `wowFlutter` and `charNoise`.
- `TapeTransport` (new, `Core/TapeTransport.h`): a tape transport as a speed
  state, the one `TapeMachine` and `Delay` use: the speed glides with
  inertia (one time constant up, another down), deviates by a smoothed
  noise, the capstan's rotation and the reel's at rates that follow the
  speed, and a scrape band; the delay is held at the head distance over the
  speed. `setTargetDelay()` sets a delay as a speed (the head distance
  stays). `TapeMachine`'s speed model now runs on it (its own random
  sequence: the figures above are of this version).
- `AuditoryBands` (new, `Core/AuditoryBands.h`) and `AuditoryScale`: the
  published formulas of the ear's scale (Bark after Traunmuller, critical
  bandwidth after Zwicker and Terhardt, threshold in quiet after Terhardt,
  specific loudness as a power law over the threshold with an exponent that
  rises where the threshold is high), bands uniform in Bark over a power
  spectrum (each bin feeds the two bands around it with weights that sum to
  one; a band narrower than a bin reads the density at its centre), a
  smoothing across bands in both directions, total loudness, and
  `dominantPower()`: the level of the dominant component of a stereo pair,
  the larger eigenvalue of the channels' covariance.
- `SpectralEnvelope` (new, `Core/SpectralEnvelope.h`): the envelope of a
  spectrum by a one-pole run both ways whose width follows frequency (a
  fraction of an octave), and the resonances as what stands over it.
- `SpectralFilter` (new, `Core/SpectralFilter.h`): a time-varying filter
  designed every hop from a gain curve and applied as a minimum-phase FIR:
  the frame is analysed with a Hann window and zero-padded (2, 4 or 8
  times), the gain curve becomes a minimum-phase filter by the real
  cepstrum, cut to the room the padding leaves, and the frame is filtered
  by it (a linear convolution) before the overlap-add. A gain of one is the
  input late by the window; the energy that comes out before an impulse
  through a cut is more than 300 dB under the total (no pre-echo); a key
  can replace the power the design reads; one filter can serve linked
  channels; setRemovalListen() applies one minus the filter.
- `LaneCascade` (new, `Core/LaneCascade.h`): a long cascade of biquads on
  one channel spread over the lanes of a vector, each lane running its own
  stretch one sample behind the lane before it: the serial cascade, late by
  lanes - 1 samples, with coefficients that ramp per sample. In mono it costs
  0.82 to 0.92 of the serial cascade; on two channels the serial cascade,
  which already fills a vector with the pair, stays the faster (0.55 of it).
- `GraphicBank` (new, `Core/GraphicBank.h`): a graphic equalizer whose bands
  do not interact: the gains asked of the bands become the gains of
  overlapping sections (a bell per band, a shelf at each end, with the
  critical bandwidth as their width) by weighted, regularized least squares
  on the centres and the points between them, with refining steps on the
  real response kept only when they bring it nearer. Bands uniform in Bark
  meet the gains asked within 0.15 dB at their centres (0.22 dB for 72
  bands), where the plain sections would pile their skirts up; it runs on
  a LaneCascade in mono and on a BiquadCascade with more channels (no
  latency).
- `TonalBalance` (new, `Effects/TonalBalance.h`): an adaptive equalizer
  driven by an auditory model. Every hop (128 samples at 48 kHz, a
  transform of 2048 with an asymmetric window weighted toward the recent
  samples, both scaling by octaves with the rate) the programme's power is
  smoothed across frequency, read in 72 bands uniform in Bark, its stereo
  covariance smoothed in time (longer in the low end) and read as the power
  of the dominant component, and turned into specific loudness. A broad
  trend across the bands is compared with a reference shape (a tilt) and
  each band with the trend; soft-knee curves with a slope on each side turn
  those deviations into corrections (`setBalance()`, `setFillDips()`,
  `setTrimPeaks()`, `setRange()`, `setKnee()`, `setFrequencyRange()` with edges
  one Bark wide, 40 Hz to 16 kHz by default), offset so the total loudness
  stays, and drive a GraphicBank. `setLevelCompensation()` converts loudness
  to decibels with each band's exponent. A key can drive it (with a fade).
  On pink noise with a peak at 3 kHz the peak comes down 3.2 dB, 300 Hz
  moves 0.9 dB and the level 1.2 dB (what is kept is the loudness); the
  latency is 16.2 ms at 48 kHz; it costs 780 to 1060 ns per stereo frame.
- `ResonanceSuppressor` (new, `Effects/ResonanceSuppressor.h`): it finds
  what stands over the envelope of the spectrum (in decibels or in
  amplitude), frame by frame, and lowers it over a threshold by a share of
  the excess, limited, weighted by a sensitivity curve of six bands, with an
  attack and a release per bin that are slower in the low end, smoothed
  across bins and applied by a SpectralFilter (no pre-echo). The window is
  about 46 ms at any rate; stereo per bin with a link (one detector and one
  filter at full link) or mid and side with a side weight; a key can drive it;
  removal and focus listening; an offline quality (eight times padding,
  sixteen frames per window). A 2 kHz line in white noise comes down 11.2 dB
  while the noise at 600 Hz moves 0.2 dB; it costs 270 ns per stereo frame.
- `SpectralDynamicEQ` (new, `Effects/SpectralDynamicEQ.h`): four dynamic
  bands (bell or shelves) that act cell by cell inside themselves: a band
  is cut into up to 128 cells placed by its weight (equal steps of its shape
  integrated over log frequency: they crowd where the band acts); each
  cell's level is
  the largest power of the key in it weighted by the band's shape, the
  threshold is relative to the band's recent peak, each cell has an attack
  and a release (faster in the upper cells), and the actions are projected
  onto the bins with the band's shape, between its flat and its full
  response, and applied as a minimum-phase filter. A loud tone at 2.5 kHz in
  a bell at 2 kHz comes down 6.5 dB while a quiet one at 1.5 kHz in the
  same bell does not move; it costs 190 ns per stereo frame.
- `SpectralReadout` (in `Core/SpectralFilter.h`): a per-bin quantity of the
  last frame published at 128 points, evenly in log frequency from 20 Hz to
  20 kHz, each point keeping the largest magnitude of the bins within half a
  point of it (a notch one bin wide still shows), for a display read from
  any thread. `ResonanceSuppressor::getDisplayReduction()` (the reduction as
  it is applied), `getDisplayExcess()` (what the detector found over the
  threshold) and `getPeakReduction()`; `SpectralDynamicEQ::getDisplayGain()`
  and `getPeakGain()`.
- `SpectralDenoiser`: options, all off by default (the output is then the
  same as before, to the bit): `setSpectralSmoothing()` across bins,
  `setGainRelease()` per bin (slower in the low end), a sensitivity curve
  (`setSensitivityBand()`) weighting the reduction, `setRemovalListen()`,
  `setFilterMode()` (the gains applied as a minimum-phase filter, the same
  latency, the same reduction of learned noise: -18.0 dB, as the STFT)
  with `setLink()` between channels, and `prepare(spec, 0)` choosing the
  transform size by the rate. State keys `smoothing`, `release`,
  `filterMode`, `link` and the sensitivity bands (`sens*`).
- Removal listening (`setRemovalListen()`, a monitoring control outside the state):
  `Compressor` (the aligned input minus the compressed signal before the
  makeup gain), `DeEsser` and `DynamicEQ` (the input late by the latency
  minus the output).
- `PhaseCorrelation::getDominantLevelDb()`: the level of the dominant
  component of the pair (0 dB for a full-scale sine in both channels in or
  out of phase, -3 dB in one channel).
  `SpectrumAnalyzer::toAuditoryBands()`: a spectrum read in bands of the
  ear's scale. `WindowFunctions::asymmetricHann()`: a window for low-latency
  analysis.
- `Compressor::setLinkMode()` and `DynamicEQ::setLinkMode()`
  (`LinkMode::Dominant`, opt-in; state key `linkMode`): a stereo pair linked
  to the level of its dominant component, the larger eigenvalue of the
  covariance that the two channels' levels make with their correlation over
  20 ms (`dominantPairPower()` and `dominantPairLevel()`, new in
  `Core/DspMath.h`). A source moved by an equal-power pan reads the same
  wherever it sits, and uncorrelated channels read the louder 3 dB down. A
  1 kHz tone 21 dB over the threshold of a 4:1 compressor, centred and then
  panned hard, is reduced by 15.09 and 15.09 dB (Loudest: 15.09 and 17.34);
  in a dynamic band by 15.73 and 15.73 dB (Loudest: 15.73 and 17.99). With
  the peak detector the compressor costs 28 to 38 ns per stereo frame with
  it (17 to 22 with Loudest).
- Minimum-phase oversampling in `Clipper`, `TubePreamp` and `TapeMachine`
  (`setOversamplingMode()`, state key `oversamplingMode`), as `Saturation`
  has it: polyphase allpass half-bands, with the dry copy of the mix through
  the same filters (`OversamplingDryPath`), so a partial mix adds in phase at
  every frequency. Latency at 48 kHz in samples, linear phase against
  minimum phase: `Clipper` 64 and 3 at 2x, 70 and 4 at 4x; `TubePreamp` 71
  and 10; `TapeMachine` 197 and 131.
- `FilterEngine::setStructure(Structure::Ladder)` (opt-in; state key
  `structure`): every section as a normalized ladder, whose memory can only
  lose energy however fast the coefficients move, on the same 16-sample
  ramps; `NormalizedLadder::processSegment()` runs one channel along a ramp
  without moving it on. A resonant low-pass swept by the signal itself
  (nonlinearity 1, Q 18, noise at three times full scale) peaks at 9.2
  against 10.9 in direct form. Per stereo frame it costs 12 to 18 ns at rest
  and 21 to 28 ns moving (the direct form, now run in one pass, 4 to 10 ns
  at rest).
- `SpectrumAnalyzer::WindowType::AsymmetricHann`: the weight on the recent
  samples (the fall over the last eighth of the frame). A tone 1024 samples
  after it starts reads -8.3 dB against -12.0 under a Hann window, and a
  steady tone reads its level (-6.02 dB for 0.5).
- `setStructure(Structure::Ladder)` in `Equalizer` (its minimum-phase
  bands), `DynamicEQ` (the band filter of the default engine, designed again
  every 16 samples as the gain moves) and `DeEsser` (the cut bell), opt-in
  with the state key `structure`: the filters as normalized ladders, whose
  memory cannot gain energy at a change of coefficients. Where a dynamic
  band jumps in from rest the ladder starts from the signal's history
  instead of an empty memory, so it enters more smoothly. Per stereo frame,
  direct form against ladder: four bands of the equalizer 16 to 42 and 47 to
  77 ns, a dynamic band 63 to 143 and 65 to 148 ns, the de-esser 22 to 35
  and 26 to 42 ns.
- `CrossoverFilter::setStructure(Structure::Ladder)` (opt-in; state key
  `structure`), and `setCrossoverStructure()` in `MultibandCompressor` and
  `MultibandSaturation` (state key `xoverStructure`): the sections of the
  splits and of the phase correction as normalized ladders, allocated by
  `prepare()` on the heap so that the switch never allocates; during a glide
  their coefficients ramp across the 32-sample grid. The bands are the
  direct form's to 4e-9; four bands in stereo cost 2.4 to 6.3 times as much.
  The 5 ms glide of the splits already keeps the direct form's memory
  consistent: with a split jumping between 100 Hz and 10 kHz every 64
  samples, the band sum of 0.3 of noise peaks at 0.65 either way at LR24
  (0.70 against 0.64 at LR48).
- `Compressor::setOptoCell(OptoCell::ThreeStage)` (opt-in; state key
  `optoCell`): the Opto character with a medium memory between the fast
  stage and the deep one (charge over one release time, 30 ms to 1 s;
  release over five), shares of the gain 0.6, 0.25 and 0.15, and the fast
  stage at 0.725 release times so that a fully charged cell still recovers
  half its reduction in one release time (0.497 measured). With a release of
  100 ms and 4:1 at 0 dBFS, after a 150 ms phrase 1.82 dB are left at 0.3 s
  (two stages 1.28) and 0.65 at 1 s (0.92); after 4 s of compression 0.61 dB
  are left at 3 s (two stages 1.61). The analytic makeup models it too.
- `Core/CharacterStage.h` (`CharacterStage<T>`), and
  `Compressor::setCharacterStage(StagePosition)` with
  `setCharacterStageSettings()` (opt-in; state keys `stage`, `stageDrive`,
  `stageEmphasis`, `stageAsymmetry`, `stageDrift`, `stageSag`,
  `stageNoise`): the colour circuit around a gain element. An emphasis
  shelf (0 to 12 dB, centred on 3 kHz) and its exact inverse after two static
  saturators (the first biased for even harmonics), divided by the chain's
  small-signal gain, so a quiet signal passes at unity (within 0.05 dB from
  100 Hz to 18 kHz); slow random drift of drive and bias; a sag that raises
  the drive with a slow envelope of the input; a noise floor that follows a
  50 ms envelope (silence stays silent, the floor under a tone lands within
  0.1 dB of its setting). It runs at about 176 kHz (4x at 44.1 and 48 kHz,
  polyphase IIR half-bands, 4 samples of delay at 48 kHz). A 1 kHz tone at
  -6 dBFS carries its third harmonic at -26.8, -14.5 and -12.1 dBc with 0,
  12 and 24 dB of drive and its second at -39.4 dBc with an asymmetry of 0.6
  (none without); 12 dB of emphasis raises the third harmonic of a 6 kHz tone
  by 3.8 dB and leaves that of 200 Hz; full sag raises it by 7.3 dB at 0 dB
  of drive. In the Compressor the stage sits before the compression (the
  detector hears it) or after it; a change of place fades over 20 ms (the
  largest step of a 1 kHz tone through three changes is the tone's own) and
  the stage adds its delay to `getLatency()`. It costs 198 to 618 ns per
  stereo frame at 48 kHz.
- `DeEsser::setMixedBell()` (opt-in; state key `mixedBell`): the cut as one
  fixed filter at the end of the range mixed by the reduction sample by
  sample, `y = x + w (end(x) - x)`, with the end chosen as `DynamicEQ`'s
  mixed bands choose theirs, so the centre sits on the reduction of the
  moment and nothing is designed while it moves; at rest the output is the
  input bit for bit, and it does not depend on how the stream is cut. On a
  6 kHz tone that jumps 28 dB 25 times a second, with a release of 1 or 5 ms,
  the default bell (designed again every 16 samples) leaves lines 3 kHz away
  at -46 to -52 dBc and the mixed one at -64 to -96; with the default 20 ms
  both sit on the floor of the measurement. 41 to 81 ns per stereo frame
  against 18 to 34.
- `Chorus::Interpolation::Linear`: two-point reads of the digital lines,
  whose loss of highs belongs to the sound of many chorus units: half way
  between samples a 15 kHz tone at 48 kHz loses 5.1 dB (the cubic read
  2.5 dB), a 1 kHz tone 0.02 dB. 0.61 to 0.84 of the cubic read's cost with
  three voices (62 to 143 against 81 to 184 ns per stereo frame). The state
  key `interp` takes the value 2.
- `Saturation::Algorithm::Curve`: a soft clipper shaped by three controls
  (`setCurveShape(hardness, curvature, asymmetry)`) that keep the slope at
  the origin and the ceiling: how far it stays a straight line before it
  turns (0 to 0.95 of its ceiling), how fast it turns (an exponent, 0.5 to
  4, of tanh(u^p)^(1/p)) and how low one half-cycle's ceiling is (to
  2^-asymmetry; neither half goes past 1). At its defaults it is tanh, to
  0.1 dB of its Fourier series in the third harmonic; a 1 kHz tone at
  -6 dBFS with 12 dB of drive carries its third harmonic at -15.5 dBc, at
  -13.8 and -13.1 with a hardness of 0.5 and 0.8, at -16.7, -14.0 and -13.2
  with a curvature of 0.5, 2 and 4, and its second at -23.8 and -18.3 with
  an asymmetry of 0.5 and 1. With a curvature other than 1 it costs 112 ns
  per stereo frame at 2x against 61 for SoftClip.
- `Saturation::setOversamplingMode(MinimumPhase)`: elliptic allpass
  half-bands in place of the linear-phase ones, with 3 samples of latency at
  2x and 4 at 4x (64 and 70 with linear phase), and the dry path of the mix,
  of the Delta output and of the adaptive blend through the same filters
  with nothing in between: a half mix of a tone the curve leaves alone is
  within 0.002 dB of the wet signal from 500 Hz to 19 kHz, and its Delta
  output is 48 dB down (what is left is the phase of the input high-pass).
  81 ns per stereo frame at 2x against 61. State key `oversamplingMode`.
- `Saturation`, stages around the curve of any algorithm, all off by
  default and at the base rate:
  `setAsymmetry(-1..1)` adds a share of the square of the signal (less its
  mean) before the curve: a 1 kHz tone at -6 dBFS through SoftClip carries
  its second harmonic at -36.6, -30.6 and -24.6 dBc at 0.25, 0.5 and 1
  (-38.3, -32.3 and -26.2 with 12 dB of drive).
  `setProgramNoise(dB)` adds white noise counted from the recent peak of
  the programme (held 50 ms, released over 150 ms) before the curve, which
  shapes it: at -40 the noise under a tone measures 45.2 dB under its peak,
  is 5e-5 half a second after the tone stops and under 1e-11 after three
  seconds.
  `setLevelTracking(-1..1)` moves the level into the curve with a follower of the
  programme (10 ms up, 150 ms down, smoothed over 5 ms) and takes it back
  after the curve, in time with the wet path: at +1 a 1 kHz tone through
  SoftClip with 12 dB of drive carries its third harmonic at -32.8 dBc and
  comes out 11.40 dB up at -36, -24, -12 and -6 dBFS alike (without it,
  -69.6, -45.7, -23.5 and -15.5 dBc); at -1 the level moves twice as far
  (-107, -59, -16.2 and -10.0 dBc). `setTrackingCorner(hz)` has the follower
  hear only the programme under a frequency.
  `setCombLoop(amount, hz)` returns the output of the curve to its input,
  inverted, one period of the frequency later (the latency of the
  processor included): at 0.8 and 200 Hz the gain is -5.11 dB at 200 and
  400 Hz and +13.95 dB at 300 Hz, 1/(1 + a) and 1/(1 - a); while it is on
  the block is processed in pieces no longer than the delay, and the output
  does not depend on the host's block size. `getCombLoopFrequency()` gives
  the frequency the latency allows.
  `setDriveCompensation(true)` measures the gain of the curve on pink noise
  at -18 dBFS through a twin of the processor (against the same noise 40 dB
  down at no drive, so that the input filters cancel) whenever the
  algorithm, drive, character, shape, antialiasing or oversampling move, at
  most every 50 ms, and takes three quarters of it back at the output: with
  SoftClip the level of such noise goes from -1.13 to 9.72, 15.87 and
  17.39 dB at 0, 12, 24 and 36 dB of drive without it, and from -1.07 to
  1.56, 2.95 and 3.31 dB with it.
  State keys `curveHardness`, `curveCurvature`, `curveAsymmetry`,
  `asymmetry`, `programNoise`, `levelTracking`, `trackingCorner`, `combLoop`,
  `combLoopHz` and `driveCompensation`.
- `MultibandSaturation` (new, `Effects/MultibandSaturation.h`): a
  Linkwitz-Riley crossover of 2 to 6 bands, a `Saturation` per band, and a
  level, a pan, a mute and a solo per band before the bands are summed. The
  bands are summed in time whatever each one runs with: a band with less
  latency than the slowest is delayed by the difference (bands at 2x, 4x
  and 8x sum back to the input's level within 0.15 dB at 100 Hz to 10 kHz).
  Level and pan glide over 20 ms, mute and solo fade over 20 ms on a ramp
  of their own, and a band that is not heard is not computed until it is
  heard again. Oversampling, its filters and antialiasing are set for every
  band at once; each band's Saturation is reachable for the rest.
- `CabinetModel` (new, `Core/CabinetModel.h`): a loudspeaker cabinet with
  no latency and no convolution. Its colour is a 12 dB per octave low cut
  with the cone's resonance (140 Hz in a small box to 70 Hz in a large one),
  a shallow dip at 450 Hz, the breakup peak and a 24 dB per octave roll-off
  (3 to 7 kHz by the brightness), designed from their analog prototypes;
  its box is six reflections between 0.29 and 3.41 ms with alternating
  signs through two allpass diffusers, added by the distance of the
  microphone with the level kept (pink noise within 1.5 dB from no distance
  to full). Settings glide over 20 ms.
- `TubePreamp::setCabinet(true)` runs the wet signal through a
  `CabinetModel` (`getCabinet()`), after the circuit and before the mix,
  with no latency and a fade of 20 ms. State keys `cabinet`, `cab.size`,
  `cab.brightness` and `cab.distance`.
- `TransientFollower` (new, in `Analysis/EnvelopeFollower.h`): how sudden the
  signal is, 0 to 1, whatever its level, as a modulation source: the
  loudest channel's analytic magnitude through a follower with a fast
  attack (0.5 ms) and one with a slow attack (30 ms), the same release for
  both; their distance in decibels over a range (12 dB by default), held
  20 ms and released over 80 ms. The same bursts give the same values at
  -40 and -10 dBFS (to 1e-4), a held tone gives none and a step of 12 dB
  reaches three quarters of the range or more. `EnvelopeFollower::
  getEnvelopeNormalized(channel, floorDb)` gives the envelope in decibels
  mapped to 0..1.
- `LoFi` (new, `Effects/LoFi.h`): grains that smear (`setSmear()`) and move
  the pitch (`setPitch()`, -24 to +12 semitones), a fold (`setFold()`,
  sin(g x)) whose output keeps the level of a sine at -12 dBFS at any gain
  (within 0.1 dB from 6 to 36 dB), a hold at a lower sample rate
  (`setRate()`, with a 48 dB per octave low-pass in front at 0.45 of the
  rate with `setRateFilter(true)`) and fewer bits (`setBits()`, 1 to 24,
  with triangular dither by `setDither(true)`), then a mix and an output
  gain. Every stage is off at its default; then the output is the input to
  the bit. No latency.
- `KeyEqualizer` (new, `Core/KeyEqualizer.h`): a parametric equalizer for the
  key of a dynamics processor, the signal its detector reacts to. Six bands,
  each a low or a high cut from 6 to 48 dB per octave, a bell, a low or a
  high shelf (6 to 24 dB per octave of transition) or a tilt (up to 6 dB per
  octave about a pivot), designed from their analog prototypes with
  `FilterDesign` and run together in one `BiquadCascade`. `setBand()`
  publishes a band from any thread; the audio thread designs it in `sync()`
  and moves the coefficients there over 20 ms. A band that is Off, or a bell
  or a shelf at 0 dB, leaves through its own poles and is then no longer
  computed; with no band in the path `processSample()` hands the sample
  back. A 24 dB per octave low cut reads -3.0 dB at its corner and -48.2 dB
  two octaves under it, a 48 dB per octave high cut -48.2 dB one octave
  over, a bell its gain at the centre and half of it at the edges
  frequency / q apart, and six bands together play the curve
  `getResponse()` reports to 0.05 dB. `writeState()` and `readState()` keep
  the bands under a key prefix.
- `Compressor`: an equalizer on the key, and key listen.
  `setSidechainBand(band, settings)` sets one of the six bands of a
  `KeyEqualizer` that runs after the sidechain high-pass on whatever the
  detector reads (the programme, the external key or, in feedback, the
  compressed output), in `processBlock()`, in `processSample()` and at
  `Precision::High`; the audio is the one of an external key shaped by the
  same filter. With a 60 Hz tone 20 dB over a quiet 3 kHz one, a 24 dB per
  octave low cut at 500 Hz on the key takes the reduction from some 7 dB to
  none. `getSidechainResponse()` gives the curve for a display.
  `setSidechainListen(true)` plays what each detector hears instead of the
  compressed audio, ahead of the lookahead, while the compressor keeps
  working underneath: the meter still reads, and switching it off returns
  to the audio of a compressor that never listened. It is a monitoring
  switch and is not part of the state. State keys `key<band>.shape`,
  `.freq`, `.gain`, `.q` and `.slope`.
- `Compressor::Character::Layered`: two layers in feed-forward. One levels,
  with half the slope of the ratio from 6 dB under the threshold, eight
  times the attack (30 ms at least) and six times the release; over it, one
  on the knobs catches what rises, with the other half of the slope from
  6 dB over the threshold. Far over the threshold the curve is the ratio's
  own (at 4:1 and a threshold of -20 dB: 0.56 dB of reduction at -26 dBFS,
  2.25 at -20, 4.78 at -14 and 9.0 at -8, where the Clean character gives
  0, 0, 4.5 and 9.0), with a knee 18 dB wide or more. Both layers take
  longer the farther they are from their target (twice as long 12 dB away
  in attack, 6 dB away in release), so a reduction comes back at a steady
  pace: with a 100 ms release, a fifth of a 15 dB reduction is back in the
  first 100 ms and it is down to 1/e after 767 ms, where a one-pole gives
  back 63 % and takes 101 ms; a step 12 dB over the threshold is 63 %
  caught after 40 ms (5 ms attack; a one-pole, 5.3 ms) and settles on the
  same 9.0 dB. Hold and range act on the sum of the layers. Topology and
  `Precision::High` do not apply to it; in Upward mode it is one envelope
  on the knobs. Cost 40 ns per frame against 18 for Clean.
- `Compressor::setCharacterColor()` with the Varimu character: the gain is
  applied by the curve of a push-pull pair of remote-cutoff stages biased
  by it, so compression and colour come from one element. The harmonics are
  odd (the even ones stay under -200 dBc), a small signal gets the gain
  exactly, and they grow with the level and with the reduction: at 1.0 a
  1 kHz tone at -6 dBFS carries its third harmonic at -61.5 dBc at rest,
  -46.8 under 3 dB of reduction, -69.6 under 6 dB (the minimum of the
  pair), -37.0 under 12 dB and -31.2 under 20 dB; at 0.5, -73.9, -58.3,
  -80.8, -48.3 and -42.6. While it is on the block runs frame by frame:
  66 ns per frame mono and 94 stereo against 19 and 21.
- `Compressor::setGainSmoothing(ms)`: the gain change passes through two
  equal one-pole sections after the ballistics and before it becomes gain
  (0 to 50 ms; 0, the default, is off). A step of the gain reaches 63 % in
  the time set (0.33 ms without it, 10.0 ms at 10), and the ripple a quick
  release leaves on a low tone goes: a 100 Hz tone 14 dB over a 4:1
  threshold with 1 ms of attack and 20 ms of release rides 0.75 dB of gain
  ripple with its third harmonic at -31.9 dBc; at 5 ms, 0.07 dB and
  -53.2 dBc; at 20 ms, 0.005 dB and -76.6 dBc, on the same mean depth. A
  feedback loop reads the gain before the smoothing. State key
  `gainSmoothing`. Cost 2.5 ns per frame.
- `NoiseGate` and `Expander`: the same equalizer on the key and the same key
  listen as the compressor (`setSidechainBand()`, `getSidechainBand()`,
  `getSidechainResponse()`, `setSidechainListen()`), after their sidechain
  high-pass, on the programme or on the external key. A 60 Hz hum at
  -10 dBFS that holds a gate open at a -30 dB threshold stops doing so with
  a 24 dB per octave low cut at 500 Hz on its key, and the audio is the one
  of an external key through the same filter. State keys as in the
  compressor.
- `Compressor`, timing that follows the programme: four weights, all 0 by
  default, on the Clean and Varimu characters in feed-forward with Branching
  ballistics, in `processBlock()`, `processSample()` and at
  `Precision::High` (Opto and FET have a memory of their own; Decoupled,
  feedback and the SplitPolarity detector keep fixed times).
  `setProgramRelease(0..1)`: the release is the knob's after a brief
  reduction and up to eight times the knob's after a sustained one, told
  apart by a slow follower of the depth of the gain change against the depth
  itself; a 100 ms release measures 101 ms after 30 ms or 3 s of reduction
  with the weight at 0, and 105 ms and 777 ms at 1 (439 ms at 0.5), so the
  level does not pump back up between phrases while a stray peak is let go
  at once. `setAdaptiveAttack(0..1)`: the attack is the knob's for a target
  6 dB or more past the gain and up to four times longer for one that is
  only just past it; a 20 ms attack covers 63 % of a 12 dB step in 29 ms and
  of a 1 dB step in 76 ms (20 and 21 ms with the weight at 0).
  `setSpectralTiming(0..1)`: both times follow the balance of the key
  between two bands of the same width on a logarithmic axis, 30 to 200 Hz
  and 2.5 to 16 kHz (what lies under 30 Hz is left out); a 100 ms release
  measures 304 ms after a 100 Hz tone and 34 ms after a 6 kHz one (105 and
  100 with the weight at 0), and pink noise stays on the knob (144 ms where
  the fixed release gives 133). `setAttackEmphasis(-1..+1)`: the difference
  between the gain and a copy of it smoothed over 30 ms is added back while
  the gain falls; a 9.0 dB reduction reaches 13.1 dB at its leading edge at
  +1, and at -1 is 1.0 dB down 10 ms into the onset where the plain attack
  is 5.8 dB down; the settled gain is the same. State keys `programRelease`,
  `adaptiveAttack`, `spectralTiming` and `attackEmphasis`.
- `Compressor::DetectorType::Hybrid`: the larger of the root of twice the
  mean square (a one-pole mean over the time of `setRmsWindow()`) and the
  rectified sample. A steady tone reads its peak with next to no ripple and
  whatever rises over the running level is caught in the same sample: a
  1 kHz tone 12 dB over a 4:1 threshold with a 1 ms attack and a 5 ms
  release settles on 9.04 dB of reduction (the curve asks 9.03) with
  0.001 dB of ripple, where the Peak detector rides the waveform at 7.42 dB
  with 0.22 dB of ripple; the first samples of an onset come out exactly as
  with the Peak detector. The analytic makeup knows it.
- `fastDecay(x)` (`Core/DspMath.h`): e^(-x) for x >= 0 by a [3/3] rational
  approximation, for the coefficient of a one-pole whose time changes every
  sample (four multiplies and a division, no call into the math library).
  Relative error 1.1e-5 at x = 1, 1.5e-3 at x = 2 and 3.1 % at x = 3; the
  argument is held to [0, 4].
- `Limiter::setMaxLookahead(ms)`: a latency kept for the lookahead. The
  audio always waits that long, `getLatency()` reports it, and
  `setLookahead()` moves under it with the gain held back by the difference,
  so the lookahead can be automated under one delay compensation. The output
  is the one of the same lookahead with no latency kept, later by the
  difference and bit for bit. 0 (default) keeps the latency equal to the
  lookahead, as before. State key `maxLookahead`.
- `Expander::setAutoThreshold(true)`, `NoiseGate::setAutoThreshold(true)`
  and `DeEsser::setAutoThreshold(true)` (off by default): the threshold
  counts from the programme instead of from full scale, as it already could
  in `Compressor`. A reference rises at once and falls 1 dB per second; in
  the expander and the gate it follows the detected level (and falls a
  quarter as fast while the key is under the closing threshold, so a pause
  lowers the threshold by 1 dB every four seconds), in the de-esser the
  largest sample of the whole input, so the band is judged against the voice
  that carries it. The output of a signal 30 dB lower is then the same
  output 30 dB lower (to the rounding of the arithmetic: 2e-7 of its peak),
  where a threshold in dBFS treats the quieter signal otherwise or not at
  all. State key `autoThreshold` in the three.
- `DynamicEQ::setLinearPhase(true)` (off by default; a setup choice): the
  bands change the level of the signal and nothing of its phase. Each band is
  then zero-phase kernels (its rest, and the end of each side, chosen as in
  the mixed engine: half way through the range the band has exactly the
  magnitude of the nominal one), all of them run by one partitioned
  convolution, and the gain sets how much of each is added to the signal.
  200 Hz below the centre of a 6 dB cut at 1 kHz the mixed engine is 19
  degrees off its input; this mode is within 0.5 with the same level. The
  latency is half the kernel plus one partition (2560 samples at 44.1 and
  48 kHz with the default kernel of 4096 taps; `setLinearPhaseKernelLength()`
  takes 1024 to 65536) and `getLatency()` reports it; the gain is held back
  to meet the audio it was computed from, and the lookahead comes out of
  that latency instead of adding to it. The bands add up side by side, each
  heard from the input, so the latency does not grow with them. Eight bands
  on a stereo block cost 373 to 466 ns per frame on three compilers, as the
  other engines (354 to 424); with blocks of 64 samples the block that
  carries a hop takes 58 to 73 us, and a moved band is designed one kernel
  per block within that. With no band acting the output is the input, late,
  to the bit. All the band fields work in the mode (gain at rest, knee,
  scope, program timing, relative threshold); oversampling is not used
  with it. The state carries `linear` and `linearTaps`.
- `FirBank<T>` (new, `Core/FirBank.h`): many long FIR kernels over shared
  input lines, by one uniformly partitioned convolution in the frequency
  domain. Each line is transformed once per hop into its own delay line of
  spectra, and a kernel only multiplies over it and transforms back, for
  every line that hears it (a mask). `setKernel()` replaces a kernel while
  the audio runs: it is transformed on the calling thread, without
  allocating, and comes in at the next hop through a crossfade of one
  partition; `removeKernel()` fades one out the same way. `room()`, `tap()`
  and `push()` give the caller the outputs in place, one partition late,
  whatever way it cuts the stream.
- `DynamicEQ` bands, in either engine: `BandConfig::gain`, the gain the band
  rests at (the dynamic part moves from there); `knee`, a soft knee around
  the threshold in dB, or `kAutomaticKnee` for one that grows with the
  width of the band and with the root of its range (7.2 dB for a bell of
  Q 1 and 12 dB of range); `scope` (`Stereo`, `Left`, `Right`, `Mid`,
  `Side`): where the band acts and what part of the sidechain it listens
  to; `timing = Program`: the band listens through a filter 1.5 times
  narrower and three followers of the level replace the attack and release
  fields, timed from the band's own frequency and width (a fast one, a slow
  one that follows it, and a last one over the larger of the two), so a
  short event is let go at the fast rate and a held one at the slow rate (a
  band at 1 kHz, Q 2, gives back 90 % of a 6 dB cut 25 ms after a 15 ms
  event, 136 ms after one of 60 ms and 360 ms after 2 s; with manual times
  of 5 and 50 ms, 230 ms in every case); and `relativeThreshold`: the
  threshold counts from the band's own recent level (a reference that rises
  at once and falls 2 dB per second) instead of from full scale, so the band
  acts the same on a quiet take and on a loud one, and when the band is
  moved to another zone a second reference listens there for half a second
  and takes over. `getBandMagnitude()` gives an interface the curve of a
  band at any gain of its dynamic part, as the engine in use plays it. The
  state carries `mixed`, and `gain`, `knee`, `place`, `timing` and
  `relative` per band; older states load as before.
- `BiquadCascade<T, MaxSections, MaxChannels>` (new,
  `Core/BiquadCascade.h`): second-order sections in series, run up to four
  at a time inside one loop over the samples, in direct form I with the node
  history shared between neighbours (two values of double state per
  section) and pairs of channels in the lanes of one register. A pass of
  one `Biquad` per section is bound by the latency of each recursion; with
  the sections in the same loop their recursions overlap. Measured on float
  stereo blocks of 512, five compiler and machine pairs: eight sections cost
  0.36 to 0.72 of eight `Biquad::processBlock()` passes and 0.29 to 0.60 of
  running every section per sample through `processSampleCore()`;
  twenty-four cost 0.34 to 0.68 and 0.29 to 0.47 (21 to 51 ns per stereo
  frame). One section costs what a `Biquad` pass does (0.81 to 1.34). A mono
  block costs 0.33 to 0.61 of the passes and 0.6 to 1.3 of the per-sample
  chain, which already overlaps its sections: no gain there. The state is
  signal, not weighted sums, so the coefficients can move:
  `rampCoeffs(section, target, samples)` interpolates all five linearly and
  lands exactly (a block whose sections all ramp costs about twice a still
  one), `rampToBypass()` takes a section out of the path with its poles
  still running, and `setSectionEnabled()` stops computing a transparent
  section and brings it back from the history of its input. The output is
  the same for any partition of the stream, bit for bit, with or without a
  ramp in flight. Owner-managed: its setters belong to the processing
  thread.
- `BiquadCoeffs::bypassed()`: the set with the same poles whose numerator
  equals its denominator (unity at every frequency). A bell at 0 dB already
  is that set.
- `NormalizedLadder<T, MaxChannels>` (new, `Core/NormalizedLadder.h`): a
  second-order section as two plane rotations and three taps (Gray and
  Markel, 1975), with `fromBiquad()` / `toBiquad()` between the two forms
  and linear ramps of rotations and taps. With no input the energy of its
  memory cannot rise, whatever the coefficients do from one sample to the
  next, so it survives what a direct form does not. Measured at 48 kHz in
  double against the transposed form (`Biquad`), the direct form
  (`BiquadCascade`) and `StateVariableFilter`: a low-pass of Q 8 switched
  between 300 Hz and 3 kHz every seven samples under a tone of 0.5 leaves
  both direct forms non-finite, the ladder peaking at 0.65 and the
  state-variable filter at 0.50; its cutoff modulated at 5 kHz over 2.5
  octaves each way at Q 10 under noise, the same (ladder 1.09, state-variable
  0.98); with the input removed and a random cutoff every sample at Q 30 the
  direct forms diverge and the ladder decays. Under an occasional hard
  switch the direct form I is the quieter of the two (peak 0.59 against
  2.27 for the ladder's taps at Q 8, switched every 137 samples), so the
  ladder is for coefficients that move fast and is ramped where they jump.
  It costs 1.1 to 1.65 times a transposed biquad per mono frame (3.6 to
  3.8 ns against 2.7 to 3.5).
- Automatic oversampling factor. `Oversampling::factorForRate(sampleRate,
  minimumInternalRate = 176400, maxFactor = 16)` gives the smallest power of
  two that brings a host rate up to an internal rate (4 at 44.1 and 48 kHz,
  2 at 88.2 and 96 kHz, 1 from 176.4 kHz up with the default), and the
  processors that oversample inside can take it on their own:
  `setAutomaticOversampling(true)` on `Saturation`, `Clipper`,
  `TapeMachine`, `TubePreamp`, `TransformerModel`, `DynamicEQ` and
  `WaveshapeTable` makes every `prepare()` choose the factor that reaches the
  internal rate their default factor gives at 44.1 kHz (176.4 kHz for the
  tape, 88.2 kHz for the others). Off by default, stored in the state
  (`oversamplingAuto`), and turned off by an explicit `setOversampling()`.
  The nonlinear stage then runs at about the same rate in every session and
  a high-rate session stops paying for a factor it does not need. Measured
  with a 1 kHz tone at -12 dBFS, stereo, on two machines: `TapeMachine` at
  96 kHz gives 0.1547 % of harmonic distortion at 4x and at the automatic 2x
  for 0.51 of the cost, and at 192 kHz 0.1548 % at 4x and at the automatic
  1x for 0.27 of the cost (a second of stereo audio at 192 kHz took 874 ms
  at 4x and 237 ms with the automatic factor on one of the machines);
  `TubePreamp` at 96 kHz 11.903 % and 11.908 % for 0.23 of the cost and no
  latency instead of 71 samples; `TransformerModel` 1.2546 % and 1.2545 %
  for 0.50; `Saturation` 6.7065 % both ways for 0.64 (soft clip) and 0.52
  (tape curve), with no latency instead of 64 samples. `DynamicEQ` gains
  `getOversamplingFactor()`.
- `OversamplingDryPath<T>` (new, in `Core/Oversampling.h`): carries a dry
  signal through the same round trip as an `Oversampling` engine, so a
  dry/wet mix adds in phase at every frequency. With the linear-phase engine
  it is the engine's whole-sample delay. With the polyphase IIR engine,
  whose delay is fractional and rises towards the band edge, it is a twin of
  the engine with nothing between its two halves: an untouched wet signal
  nulls against the dry one exactly, where against a dry signal delayed by
  the rounded latency a tone near 18 kHz at 48 kHz (2x, High) leaves a
  difference of 1.98 times its amplitude, the two close to opposite.
- `SmoothedValue::setAdaptiveRamp(span, fastMs, slowMs)` (opt-in, Linear
  curve): the duration of each ramp follows the size of the step that starts
  it. A step of `span` or more lands in `fastMs`, a vanishing one glides
  over `slowMs`, and a step d in between takes
  fastMs + (slowMs - fastMs) (1 - d / span)^2, so fine automation hides its
  steps and a preset change does not drag. An edit in mid ramp starts from
  the value already reached.
- `DownbeatEstimator<T>` (new, `Analysis/DownbeatEstimator.h`; offline):
  names the beats that open a bar, from a signal and its beats
  (`BeatTracker`'s, a host's grid or a tap). `estimate(mono, sampleRate,
  beatSamples, beatsPerBar = 0)` returns the bar length (the caller's, or 3
  or 4 as estimated), the first beat that opens a bar, all of them, and a
  confidence. It reads six cues at every beat: how far the beat's spectrum
  is from the one before, at about 5 Hz of resolution; how far the two bars
  after a beat are from the two before it, in pitch classes, in spectrum and
  in bass spectrum; the low end from 30 ms before the beat to 120 ms after
  it; and how low the bass sits. Each is brought to zero mean and unit
  deviation over the beats, and the candidate first beat on which most of
  them falls is taken. The two cues of the low end count in proportion to
  how flat the spectra of the beats are: in full where there are drums, not
  at all in music of pitched lines alone, where a loud low note says nothing
  of the bar. The bar length is 3 when the beats resemble each other more
  three apart than two or four apart and a first beat stands out more in
  bars of three. Measured in one job per set on three annotated sets against
  a detector of the spectral-difference kind (Davies and Plumbley, 2006)
  given the same beats and bar length, as the mean F-measure of the first
  beats within 70 ms. On 698 excerpts of dance music (175 in three, 523 in
  four): with the annotated beats and bar length 0.872 against 0.747 (the
  right first beat in 87.1 % of the excerpts against 74.6 %); with the bar
  length estimated (right in 98.0 %) 0.868 against 0.740; with
  `BeatTracker`'s own beats 0.677 against 0.602, and 0.677 against 0.602
  with the bar length estimated (right in 93.0 %). On 360 excerpts of solo
  guitar, all in four: 0.628 against 0.550 (62.8 % against 55.0 %); 0.625
  against 0.561 with the bar length estimated (right in 93.1 %); with
  `BeatTracker`'s own beats 0.296 against 0.253, and 0.294 against 0.258. On
  243 takes of a drum kit alone, 240 of them in four: 0.770 against 0.379
  (77.0 % against 37.9 %); 0.759 against 0.385 with the bar length estimated
  (right in 95.9 %); with `BeatTracker`'s own beats 0.596 against 0.286, and
  0.597 against 0.285. With the beats given, the estimator is ahead or level
  in each of the twenty-eight styles of five excerpts or more; with the
  library's own beats it is ahead in twenty-one, level in one and behind in
  six, by 0.025 at most: beats on the off-beats or at another metrical level
  leave no first beat to be right. The confidence sorts the answers (98.9 %,
  95.4 % and 100 % right above 0.75 in the three sets; 57.6 %, 34.3 % and
  57.4 % below 0.25) and reads near zero where nothing marks the bar.
- `LadderFilter::setSaturation(Saturation::Input)` (opt-in; `Feedback` and
  `Stages` are unchanged and bit-identical): a third saturation model, with
  linear stages and one saturator on the loop input, run in four sub-steps
  per sample. Every sub-step closes the loop without delay: the loop gain
  past the saturator stays below 0.036, so one Newton step from the explicit
  value leaves the square of its error. The input is interpolated across the
  sub-steps and the output is their mean, each with a one-pole level
  correction. Measured at 48 kHz next to a fixed-cost ladder of the same
  class written into the harness (four sub-steps, a cubic rational saturator
  on the loop input, a sub-step of delay in the loop, polynomial tuning
  corrections). Harmonics above -80 dBc of a full-scale 110 Hz sine against
  each model's own equations solved 32x oversampled, over cutoffs from 250
  Hz to 20 kHz, drive 1 and 3 and resonance 0 to 0.9: within 0.05 dB
  (comparison 0.42; `Stages` 0.27 up to 2 kHz and 1.6 at 8 kHz). Resonance
  peak against the analog one: within 0.25 cents up to an 8 kHz cutoff and
  1.1 cents at 20 kHz (comparison -15.5 to +15.3; the other two models +3.8
  at 8 kHz and +18 at 20 kHz). Small-signal response against the analog
  four-pole with no resonance: within 0.5 dB up to 15 kHz at any cutoff and
  -0.6 to -2.1 dB at 20 kHz (comparison -1.0 to -1.4 dB at 5 kHz, -4.4 to
  -5.0 at 10 kHz and -11 to -13 at 15 kHz). Aliasing of a full-scale 5 kHz
  tone at drive 3 over cutoffs from 500 Hz to 20 kHz: -45 to -61 dBc
  (comparison -29 to -38; `Stages` -21 to -38 up to 12 kHz and -64 at 20
  kHz, where this model gives -60). Cost, the same at every cutoff and
  resonance: 53 to 74 ns per sample in mono depending on the machine (float;
  GCC, Clang and MSVC on x64 and GCC on arm64), which on each machine and
  setting is 0.47 to 0.85 times the comparison model, 0.30 to 0.95 times
  `Stages` and 1.5 to 4.0 times `Feedback`; channel pairs advance together
  at 35 to 48 ns per channel. Resonance above 1 (up to 1.2) self-oscillates
  at a level the saturator sets; the HP24 output is taken from the saturated
  loop input. The state carries the model, and the integrators carry over a
  switch between models.
- `Limiter::setTruePeakPrecision(TruePeakPrecision::High)` (opt-in; the
  default `Standard` is unchanged): true-peak mode reads with
  `TruePeakAnalyzer` (an 8x grid from a 512-tap first stage) instead of the
  64-tap reading that answers within 33 samples, and follows the content
  between 0.45 of the sample rate and Nyquist that the short reading cannot.
  The 262 samples the reading reaches back are added to `getLatency()`
  (5.5 ms more at 48 kHz) and the lookahead stays all attack ramp, as in
  sample-peak mode. Output true peak over a -1 dB ceiling at the defaults,
  read with a referee four times as long (a 2048-tap first stage), `High`
  against `Standard`: white noise +0.21 against +0.86 dB, its first
  difference +0.32 against +1.40, noise above 0.42 of the sample rate +0.40
  against +1.51, bursts of a tone at 0.47 of the sample rate +0.01 against
  +0.59, a hard-clipped 5 kHz tone +0.01 against +0.20, a low thump under
  noise +0.08 against +0.23, a 48-partial mix +0.001 against +0.004. Every
  reading waits for the slowest one of its sample, so none turns the gain
  down early: the dense mix comes out 0.05 dB louder than in `Standard`, and
  material `Standard` reads low comes out quieter by what it let through
  (white noise 0.25 dB). Intermodulation as in sample-peak mode at the same
  lookahead (60 Hz + 7 kHz at the default: -63.0 dBc, double -58.9, against
  -61.9 and -61.5). A frame costs 2.1 to 2.7 times what it costs in
  `Standard`, by compiler. `prepare()` now allocates the reading's kernels and
  histories and the gain histories of the longer lookahead whether or not
  the mode is used: 25 KB more for float stereo at 48 kHz, 41 KB in double.
  The state carries the choice (`truePeakPrecision`; older blobs restore
  `Standard`), and the reading can be switched during playback.
- `TruePeakAnalyzer::process()`: the analyzer's readings one per input
  sample (the largest of them is what `processMax()` keeps), with the BS.1770
  Annex 2 reading as a separate output, for a detector that can wait 262
  samples for them.
- `PitchFollower::track(mono, sampleRate[, settings])` (static, offline):
  reads a whole mono signal and returns one `TrackPoint` (time, pitch,
  confidence, voiced) per detection, decoded over the complete sequence. The
  real-time follower judges each detection with the past alone; with the
  whole sequence known, every detection offers its own reading, the octaves
  of it, its neighbours' readings and the line between them, and the track is
  the cheapest path through those and "unvoiced" (pitch changes cost their
  size, entering or leaving a note more than filling a short gap). Measured
  at 48 kHz on legato melodies, frames within 50 cents / frames with voicing
  and pitch right / voicing false alarms, against the follower at its best
  settings and an HMM note tracker decoded over the whole file: clean 95.2 /
  95.3 / 2.4% (follower 93.7 / 94.2 / 1.9, HMM 94.2 / 94.3 / 5.1); 100 cents
  of vibrato 95.7 / 95.7 / 4.0 (94.6 / 94.8 / 2.9, HMM 95.0 / 94.7 / 5.6);
  5 dB SNR white noise 96.7 / 96.3 / 0.0 (93.8 / 94.1 / 0.0, HMM 95.1 / 94.8
  / 4.7); 5 dB pink noise 96.3 / 95.9 / 0.0 (93.3 / 92.9 / 0.0, HMM 95.1 /
  95.1 / 4.3); 0 dB white noise 92.3 / 91.6 / 1.6 (86.2 / 87.3 / 0.0, the HMM
  reads none); a fundamental 15 dB below its partials 100 / 99.5 / 0.6 (97.8
  / 98.0 / 0.6, HMM 99.8 / 98.4 / 6.9). Octave errors at most 0.2% of voiced
  frames (follower up to 0.9%). The same holds on material generated from
  other seeds. It allocates and is not for the audio thread.
- `PitchFollower::isVoiced()`: whether the latest detection passed the
  confidence and range gates, a per-detection voicing flag without
  `isTracking()`'s 250 ms hold; and `PitchFollower::setNoiseSuppression()` /
  `getNoiseSuppression()`, which reach the wrapped detector's read through
  noise (off in the follower by default). Against an HMM note tracker (the
  second stage of Mauch & Dixon 2014,
  decoded causally) on 30 s legato melodies at 48 kHz, with `setGlide(0)`,
  `setConfidence(0.5)` and `isVoiced()`: readings within 50 cents 93.7%
  against 93.5% clean, 94.6% against 94.1% with 100 cents of vibrato, 93.8%
  against 71.0% at 5 dB SNR; voicing false alarms 1.9% against 5.1%; 22
  against 24 ms from a note change to a reading within 50 cents; with the
  noise suppression and `setConfidence(0.45)`, 86.2% at 0 dB SNR, where the
  HMM reads none.
- `simd::dotProductStrided(shared, first, stride, count, outputs, out)`:
  `outputs` double dot products of one array with windows `stride` elements
  apart. Every product keeps its own accumulator, so a result has the same
  bits whether it is computed alone or in a set of any size, and the shared
  operand is loaded once for up to eight products that run side by side. It
  runs on the AVX2+FMA kernels with the run-time dispatch.
- `simd::dotProduct4(shared, other, count, out)`: four double dot products of
  one array with four others, each equal to `dotProduct(shared, other[x],
  count)` bit for bit. The AVX kernels load the shared operand once for the
  four products and share their reductions; other builds run four
  `dotProduct()` calls.
- `TruePeakEstimator` (Core/TruePeakDetector.h): a block reader of the
  band-limited reconstruction peak (the BS.1770 Annex 2 procedure with
  64-tap Kaiser-windowed sinc phases and the crest of every 4x grid maximum
  refined), with a tail query for finite programmes and a running-maximum
  form. It is the long-kernel estimator the Limiter's true-peak mode already
  ran, moved to Core; the Limiter's output is bit-identical.
- `TruePeakAnalyzer` (Core/TruePeakDetector.h): the meter's reading of the
  band-limited reconstruction peak. The BS.1770 Annex 2 procedure carried to
  8x in three stages (a 512-tap Kaiser-windowed sinc for the midpoint of
  every sample interval, then 16 and 8 taps for the 4x and 8x points), the
  crest of every grid maximum refined, kept as a running maximum that is the
  same value however the stream is cut into calls, with a tail query for
  finite programmes. A reading reaches 261 samples back; no allocation.
- `Oversampling` stopband depth: a fourth constructor argument,
  `Depth::Deep`, keeps a quality preset's band (passband and stop edge) and
  makes every half-band stage at least 100 dB deep over the band it
  protects (`getDepth()`). The later stages of the presets are only as deep
  as their first stage (High: 80 dB), so harmonics a nonlinear stage leaves
  near the top of the oversampled band fold back at that level; with Deep a
  line at 80 kHz at 4x comes back below -100 dB. Taps per stage (first
  first): Low 91/35/23/31, Medium 115/31/23/31, High 163/31/23/31 (Preset
  31/11/7/15, 63/19/15/15, 127/23/23/15); High latency at 2x/4x/8x/16x
  82/90/93/95 base samples (Preset 64/70/73/74). Maximum is unchanged.
  The polyphase IIR mode gives every stage the coefficient count that
  reaches 100 dB (104 dB or deeper for every preset). The default
  (`Depth::Preset`) is unchanged.
- Plugin layer, VST3 note expressions: a class that declares
  `static constexpr uint32_t noteExpressions = dspark::plugin::expressions(...)`
  offers those expressions (volume, pan, tuning, vibrato, expression,
  brightness) through `INoteExpressionController`, and each
  `NoteExpressionValueEvent` reaches `handleMidiEvent` as a
  `MidiEvent::Type::NoteExpression` on the voice its note id names, in the
  same units as CLAP (volume a linear gain 0..4, tuning in semitones); a
  NoteOn's own tuning arrives as the voice's first Tuning expression. A CLAP
  build of a declaring class delivers only the declared types. Before, VST3
  hosts could send no expression at all.
- Plugin layer, CLAP per-note automation: a `perVoice()` parameter also
  declares per-note-id, per-key and per-channel automation; an addressed
  `CLAP_EVENT_PARAM_VALUE` sets the matching voices' own value (the
  parameter's value stays as it is), and each voice's `NoteModulation`
  carries its own value (else the parameter's) plus its own modulation
  (else the global one). Before, an addressed value moved the global value.
- AU: `MusicDeviceMIDIEventList` (Universal MIDI Packets, MIDI 1.0 and 2.0
  channel voice messages) and `kAudioUnitProperty_AudioUnitMIDIProtocol`
  (MIDI 2.0) on SDK 12 and later, so MIDI 2.0 data keeps its resolution
  (65536 velocity levels instead of 127; 32-bit controllers and bend);
  `kAudioUnitProperty_LastRenderError` (the first refused or failed render
  since the last read, listeners told on the main thread) and
  `kAudioUnitProperty_ShouldAllocateBuffer` per element (an input set to
  false is pulled without lent buffers). Before, the three properties and
  the selector were refused.
- Plugin layer: a state the host loads while the editor is open resizes it
  at once (VST3 `IPlugFrame::resizeView`, CLAP `request_resize`, the AU
  container view); before, the loaded size applied at the next opening.
- Plugin layer, CLAP per-voice signals: `perVoice(param(...))` marks a
  parameter the host may modulate voice by voice. It declares per-note-id,
  per-key, per-channel and per-port modulation, and each addressed
  modulation reaches `handleMidiEvent` as one `MidiEvent::Type::NoteModulation`
  per voice it matches, carrying that voice's plain value (re-sent when the
  parameter moves). Note expressions (volume, pan, tuning, vibrato,
  expression, brightness, pressure) arrive as `MidiEvent::Type::NoteExpression`
  per matching voice; `MidiEvent::noteId` carries the host's note id on
  NoteOn/NoteOff too. The optional `isVoiceActive(channel, key, noteId)`
  tells the host when a voice really ends (CLAP NOTE_END); without it a
  voice ends at its note-off. Before, every addressed modulation and every
  note expression was dropped.
- Plugin layer: a resizable editor (`EditorResize::Free` / `KeepAspect`)
  reopens at the size the user left it, and the size is saved with the
  session in every format (logical pixels in the shared state container,
  reserved entries `EDSW` / `EDSH`; older builds skip them). Before, every
  open used the declared `editorSize`.
- AU: render notifications (`AudioUnitAddRenderNotify`) run before and after
  every slice with the pre/post-render flags, and with the error flag on a
  refused call; the selectors used to accept and drop them. A change of
  `getTailSeconds()` after parameter motion notifies the `TailTime`
  listeners on the main thread, and an initialize that moves the latency or
  the tail notifies both properties.
- `OnsetDetector::setLinearFlux(bool)` / `getLinearFlux()` and
  `OdfFrame::linear`: the linear-magnitude spectral flux of the same frame as
  the envelope, for a consumer that wants the loudest sound as well as every
  register's share (default off; `BeatTracker` turns it on). The envelope and
  the onsets are bit-identical either way; about 0.1 ms of CPU per second of
  audio when on.
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
  accumulation included: 1.8 ns per sample with AVX2 and 2.4 ns with SSE2,
  against 14 ns for `read()` and 2.45 ns for a 4-point cubic reader of one
  table (no crossfade); 3.2 ns with AVX2 when the frame position moves every
  sample. On one level 1.5 and 1.9 ns. The SSE2, NEON and AVX2 kernels share
  one source over each instruction set's lane operations, and derive each
  group's table indices while the previous group is interpolated.
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
  and change the rate with a 24-36 tap kernel. Every tier keeps Direct's
  passband edge, image and alias rejection and in-band tone residual, or
  improves them (Ultra, 96 to 44.1 kHz: alias of stopband tones -218 dB
  instead of -209). Double, ns per input sample once prepared, into caller
  storage, Direct to TwoStage: Ultra 26.2 to 2.8 (96 to 44.1 kHz), 26.9 to
  4.7 (48 to 44.1), 27.7 to 5.5 (44.1 to 48), 60.3 to 8.5 (44.1 to 96); High
  16.4 to 2.7, 16.6 to 4.5, 20.4 to 4.7, 42.1 to 7.0 (stage 1 reads a double
  input in place and runs its filter product and inverse as one transform;
  at 2:1, 96 to 48 kHz, the inverse runs at half length on the folded
  spectrum: High 14.6 to 1.4, Ultra 23.9 to 1.6). Both instantiations
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
- `PitchDetector::setNoiseSuppression()` (on by default as of the entry
  under Changed; the figures here are from when it was added, switched off
  by default): a steady window that reads unvoiced is read again with its
  noise floor removed by spectral subtraction (the floor estimated per band
  from the window itself). The cleaned window locates the period, the
  input's own difference function next to its dip gives it (the subtraction
  shifts partials by a few hertz), and the reading is kept only when the input
  repeats at the period's multiples about as strongly as at the period, which
  a narrow band of noise does not. Harmonic tones of 70 to 1000 Hz at 0 dB
  SNR, 400 tones, readings within 50 cents: white noise 84.5% right and 0.2%
  wrong (off: unvoiced; the frame stage of the probabilistic-threshold method
  29% and 71%), pink noise 75.2% and 3.0%; 5 and 10 dB unchanged. The reading
  keeps the input's own confidence (about 0.5 at 0 dB), so a confidence gate
  still ignores it. Band-limited noise alone stays near what the first reading
  voices: a Q 5 band at 1 kHz 10.1% of windows (off 8.4%), a Q 2 band at
  300 Hz 0.6% (off 0%). Periods below 67 Hz (automatic window, 48 kHz) are
  not read through the noise. CPU on noise 20 to 95 ns per sample; voiced
  windows are not read twice.
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

- `TonalBalance` weighs each band by how audible it is (nothing under its
  threshold in quiet, in full from 10 dB over it), and reads a dip as a band
  under the audible bands both below and above it and a peak as a band over
  both. In the bands at the ends, where the threshold climbs, a programme's
  loudness is almost nothing: it read as a hole of tens of decibels, so Fill
  Dips raised 20 to 50 Hz and 12 to 20 kHz to the range (+3.3 to +6 dB on a
  mix-like programme), the loudness offset raised them after Trim Peaks,
  and the fall of the content at an end read as a dip. Those ends now move
  0.07 dB at most with Fill Dips or Trim Peaks alone and 0.8 dB at most at
  the defaults, on pink noise and on a mix-like programme at two levels. The same ends set the mean the reference is measured
  against, so the tilt hardly moved the curve (+0.08 dB/oct of correction
  between 200 Hz and 5 kHz for a tilt of +3); it is now -0.98 and +1.10
  dB/oct for -3 and +3 at Balance 0.5 (it acts through Balance). A peak is
  measured against its sides without itself, so Trim Peaks takes off more of
  it (a 3 kHz tone over noise: -5.2 dB at the default 0.5, -3.2 before); the
  auditory model's total loudness stays within 0.15 dB, while the RMS and
  the integrated loudness of that programme, carried by the tone, fall 1.9
  dB and 2.3 LU.
- `KeyDetector` reads recorded music with an analysis of its own, the new
  default `Profile::Recorded`: the spectral peaks of a ~186 ms window (8192
  samples at 44.1 and 48 kHz, the power of two nearest that length at other
  rates, a quarter of it between frames), each placed to a third of a
  semitone by parabolic interpolation between 26 Hz and 5 kHz; the tuning of
  the programme read from where those peaks fall within the semitone; every
  note crediting the peaks of its first five harmonics (0.6, 0.36 ... from
  the second); the pitch classes of MIDI 36 to 83 taken per frame relative
  to the strongest, classes under 0.3 of it left out; and the key named by
  the cosine of the cube root of their sum with degree weights
  (`kRecordedMajor`, `kRecordedMinor`) set on the measurements. On annotated
  recordings, a fifth, a relative or a parallel key counted 0.5, 0.3 and
  0.2: 70.9 in 100 on 604 electronic tracks (it read 44.2), 68.8 on 1159
  electronic tracks no weight was chosen on (42.3), 70.4 on 360 solo-guitar
  takes (55.3), 73.1 on 837 excerpts of ten genres (49.6) and 94.0 on 48 art
  songs (65.8), the same at the files' own 22.05 kHz. The best of three
  classical estimators run on the same audio reads 69.5, 67.7, 66.6, 71.2
  and 94.6 on those sets; the one that reads 94.6 on the art songs reads 10
  to 18 points under this analysis on the other four. A cadence 40 cents
  flat is named in all 24 keys (the published profiles over the note
  chroma: 5). `Profile::KrumhanslKessler` and `Profile::Temperley`
  are unchanged, on the note chroma as before; both front ends run on every
  stream, so a switch re-reads evidence from the start, and `chroma()`,
  `getFrameCount()` and `getWindowSize()` report the front end of the
  profile in effect. A stereo frame costs 16 to 24 ns with both front ends
  (3 to 4 with the note chroma alone): an FFT of the window every quarter
  window, 8192 points every 2048 samples at 48 kHz.
- `AlgorithmicReverb`: the wet signal at full mix has the dry signal's level.
  Its energy grows with the decay over the mean round trip of the loops (the
  law the early field was already matched to); a gain from those two takes
  that back, so on pink noise every type sits within -1.1 and +1.0 dB of the
  dry level at its preset (it sat 7 to 16.5 dB under it: Room -16.5, Hall
  -11.0, Cathedral -8.9) and a Hall stays within -1 to +3.5 dB as its decay
  goes from 12 s down to 0.3 s at any size (it moved over 10 dB).
- `AlgorithmicReverb`: a change of decay, size, bass or treble multiplier or
  decay band glides the loop absorption to its new design over the same
  ~20 ms as the other parameters while the tank holds signal (the loop gains
  stepped, a click on every line's output); an empty tank, after `prepare()`
  or once its content is under -120 dB, still takes the design at once.
- `AlgorithmicReverb` runs its input diffusers one stage at a time over
  each chunk, in pieces that read only samples older than themselves and
  never cross the end of a line, so each piece is array arithmetic on the
  line itself. The output is the same to the bit (and in blocks of one
  sample, where every piece is one sample, the same as in blocks of 64). Per
  frame it costs 0.93 to 1.05 of the sample-by-sample form across the
  machines measured (0.93 to 0.96 on one, within the spread of the others).
- `FilterEngine` and `CrossoverFilter` (and with it `MultibandCompressor`
  and `MultibandSaturation`) run their sections in one pass
  (`BiquadCascade`: every section and a pair of channels together, the
  memory in double), at rest and through their ramps and glides. Against the
  sections run sample by sample one after another the pass costs, per frame,
  0.26 to 0.58 of it in stereo and 0.39 to 0.99 in mono (1.14 for five
  sections with one compiler) for one to five sections. A 24 dB/oct low-pass
  in stereo costs 4.1 to 9.8 ns per frame (10.1 to 17.1 before) and a
  four-band equalizer 16 to 42 (38 to 86). The response is the same; the
  output moves at the level of rounding.
- `LoudnessMeter`: the two K-weighting stages run side by side, stage 1 of a
  frame (both channels in one register) next to stage 2 of the frame before
  it, and every call closes the pipeline on its last frame. Every frame goes
  through the same two-lane step, the first and last of a call included, so
  the readings do not depend on how the signal is split into calls (the step
  fuses its multiply-adds explicitly where the target has them, so the
  compiler has none left to fuse differently from one place to another).
  They are the same to the bit as before on targets without fused
  multiply-adds and move at the level of rounding on the others. The meter
  costs 0.93 to 0.996 of what it did per frame (the true-peak reader is most
  of the rest).
- `DCBlocker`: pairs of channels run in the two lanes of one register (an
  odd last channel alone, 32-bit x86 one at a time), each lane computing the
  single-channel operations. In stereo it costs 0.26 to 0.80 of what it did
  per frame at orders 2 to 10 and 0.53 to 0.84 at order 1 (the same on one
  compiler); the output is the same to the bit where the compiler does not
  fuse multiply-adds (3e-8 in float otherwise).
- `PitchShifter::prepare(spec, 0)`: every engine takes the power of two
  nearest 43 ms, as Studio did (Standard and High kept 2048 at any rate):
  2048 at 44.1 and 48 kHz as before, 4096 at 88.2 and 96 kHz and 8192 at
  176.4 and 192 kHz, so the analysis spans the same time at any rate (the
  latency of Standard and High is twice the frame).
- `StereoWidth`: the bass mono fades in and out over 20 ms with its filter
  running. Switching it off stepped the side by its low end (up to 0.45 of a
  0.5 side at 60 Hz) and switching it back on started from the filter's
  stale memory; settled, the output is the same to the bit.
- `TapeMachine::setNoise()`: the hiss is laid on the tape, before the
  reproduce head and the de-emphasis, so it takes the colour of the speed and
  the standard (at NAB 15 ips the band from 5 to 15 kHz is 11 dB under the
  band from 100 Hz to 1 kHz; it was white), at the level set (within
  0.5 dB), on independent channels, and its level glides over 50 ms.
- `Compressor`: the characters are data. What told Clean, Opto, FET and
  Varimu apart (the limits of their knobs, the times and the weight of
  their memory stage, the floor of their knee, the growth of their ratio,
  whether they close their own loop) is one table of numbers read by one
  engine, with the same arithmetic as before. `setCharacter()` now takes
  five values; a state written by an earlier version reads the same.
- `BeatTracker::analyze()` and the offline session, and `OfflineTempoAnalyzer`
  with them: the beat grid is laid in two passes, and the default tightness
  is 400 (was 25). A recursion loose enough to follow a tempo that moves is
  loose enough to take a beat off the grid onto a louder note beside it and
  bring it back at the next one (the anticipated sixteenth of a samba, the
  swung eighth of a shuffle), and that is what the single loose pass did. Now
  a first pass at 25 only follows; its intervals give the period the music
  keeps overall and the period it moves at around each beat; and the second
  pass is held at the tightness asked for to the overall period wherever the
  two agree, and follows the local one where they part: from a twelfth of
  the period away where the intervals around the beat are spread out, from a
  fiftieth where they all agree, which is a tempo of its own. Where the
  tempo changes at once, the period is read on the side that is steady; and
  a second tempo too far from the first for the loose pass to reach is
  looked for between the two, and taken where the grid laid there has a
  quarter of its intervals at each, in runs of eight or more. An interval of
  the held pass costs at most eight beats' worth of onset strength, so a
  change of phase by half a beat is taken in one interval where a tight grid
  would slide into it over eight; where the tempo swings faster than its
  period can be read, the held grid carries less than half of what the loose
  one does and the loose one is delivered. The beats are no longer moved to the
  largest envelope peak within an eighth of a period, which put them on the
  note beside the beat as readily as on the beat; a chain may start at any
  frame, where it used to open with an interval from the head of the signal
  to its first note; and the two ends of the grid are kept down to a quarter
  of what a beat carries (was a half), a half in the first frames of the
  signal. Among the candidates of the period, one within 0.35 of an octave of
  another that correlates more is ranked only if it correlates 0.6 of it: a
  lag on the shoulder of a peak used to outrank the peak. `setTightness()`
  keeps its meaning for values up to 25 (one loose grid); above, it is the
  tightness of the passages at a steady tempo. `defaultTightness()` is new.
  Measured against annotated beats, as the mean F-measure at 70 ms, on 698
  excerpts of dance music, 243 takes of a drum kit alone and 360 of a guitar
  alone, before / now / a reference tracker of the same family (a dynamic
  program over an onset function) run in the same job. Against the
  annotation as it is: 0.776 / 0.830 / 0.784, 0.760 / 0.805 / 0.763 and
  0.520 / 0.559 / 0.608. Against its best reading (as annotated, twice or
  half as fast, on the off-beats, in threes): 0.838 / 0.919 / 0.891, 0.860 /
  0.925 / 0.917 and 0.668 / 0.786 / 0.775. Beats named that are right: 0.873
  / 0.935 / 0.889, 0.896 / 0.955 / 0.930 and 0.801 / 0.874 / 0.843; annotated
  beats found: 0.825 / 0.921 / 0.904, 0.847 / 0.918 / 0.915 and 0.612 / 0.762
  / 0.755. Intervals within an eighth of the grid's median: 0.878 / 0.998,
  0.919 / 0.997 and 0.776 / 0.995 (reference 0.952, 0.966, 0.917). Behind
  the reference: the annotation as it is on the guitar alone, where the
  grid is at the annotated tempo in 53 % of the pieces against 61 %, and at
  half of it in 14 % against 5 %. On clicks: a bar whose loudest note is a
  sixteenth before a beat that carries nothing, at 80 to 140 BPM, 0.75 to
  0.93 before and 1.000 now at every tempo; one change of phase by half a
  beat, taken in one interval with every click on a beat (it was bridged
  with a beat on nothing, 0.98 to 0.99 at 15 ms); clicks fading to a
  twentieth over the last four, all four named (two before); a tempo that
  changes at once, between 90 and 140 BPM, by a twentieth or by two fifths,
  once or back and forth, with a third of the clicks or more at each tempo,
  every click a beat within 7.6 ms (7.7 on a steady train), where six of the
  thirteen cases tried read 0.67 to 0.86 at 15 ms before; a tempo
  swinging by 15 % every sixteen beats and every nine, and ramps between 100
  and 140 BPM, as before.
- `FFTReal` from 16 double or 32 float samples on, in builds with vectors: a
  new engine, faster on every machine measured, with output that changes at
  rounding level. The samples are taken as W sequences of N/W points, one in
  each lane of a W-lane vector (sample t of sequence j at x[W t + j]), so a
  vector loaded from the signal holds one point of each: the passes
  transform every sequence in its own lane, as a scalar transform works on
  numbers, and shuffle nothing. They are real radix-4 passes on packed half
  spectra (a first pass of four or eight samples without twiddles, then
  passes that join four spectra into one of four times the points), and one
  last step joins the W spectra, W bins at a time, into the interleaved
  spectrum; at the AVX width that step works by halves of a vector while the
  transform stays within the first cache (eight float lanes at every size,
  four double lanes up to 2048 samples), which keeps every shuffle within a
  128-bit half. The inverse runs the same steps backwards. Shorter
  transforms, and builds without vectors, keep the half-size complex engine
  and their output, bit for bit; `FFTComplex` is untouched. What a caller
  sees: `getInverseStepCount()` returns other counts than before and a step
  of `inverseStep()` may do nothing (the count is the same for every kernel
  set of a build); `allocationBytes()` reports the new memory (the two work
  buffers as before, and one table allocation per kernel set where the
  constructor made five or six allocations); a unit impulse and a flat
  spectrum still map to each other exactly, and the inverse still takes the
  imaginary parts of DC and Nyquist as zero whatever they hold. Measured
  before and after in one job per machine, forward plus inverse at 64 to
  65536 points, as the time after over the time before: on arm64 (a server
  core), GCC 0.71 to 0.87 in double and 0.77 to 0.83 in float, Clang 0.74 to
  0.86 and 0.72 to 0.86; on x64 with SSE2, GCC 0.68 to 0.95 and 0.73 to
  0.91, Clang 0.72 to 1.03 and 0.75 to 1.01, MSVC 0.70 to 0.94 and 0.63 to
  0.92; built for AVX2 and FMA, GCC 0.74 to 0.98 and 0.77 to 0.94, Clang
  0.71 to 0.89 and 0.63 to 0.94, MSVC 0.78 to 0.93 and 0.73 to 0.98; through
  the run-time dispatch, GCC 0.74 to 1.02 and 0.76 to 0.98, Clang 0.76 to
  0.95 and 0.70 to 0.90, MSVC 0.79 to 0.94 and 0.76 to 0.94. The rows at or
  above 1.0 are four: with SSE2 and Clang, 8192 and 32768 double points
  (1.00 and 1.03) and 65536 float points (1.01); through the dispatch with
  GCC, 2048 double points (1.02). Against a reference FFT's ordered real
  transform in the same jobs (it leaves its inverse unscaled), worst size,
  before -> after: on arm64, double 1.22 -> 0.99 and float 1.18 -> 0.94
  (GCC), 1.22 -> 0.98 and 1.11 -> 0.94 (Clang); on x64 with SSE2, double
  1.11 -> 0.95 and float 1.08 -> 0.93 (GCC), 1.06 -> 0.85 and 0.97 -> 0.82
  (Clang), 1.04 -> 0.89 and 1.08 -> 0.82 (MSVC); built for AVX2, double 0.91
  -> 0.89 and float 0.95 -> 0.73 (GCC), 0.96 -> 0.83 and 1.03 -> 0.64
  (Clang), 0.97 -> 0.83 and 0.90 -> 0.75 (MSVC). What decided the arm64
  figures with GCC: two neighbouring vectors stored as a pair take one
  vector operation on that core where two single stores take two, and that
  compiler addressed the streams of a pass in a form that has no store pair;
  giving each pair a base register of its own took a tenth off every pass.
  What decided the AVX figures in float: on one x64 server CPU family the
  step that joins eight lanes was bound by the one port that runs a shuffle
  across the halves of a vector, 24 of them for every eight bins; loading
  and storing half vectors takes them all away, at the price of loads and
  stores that the step only waits for once its data has left the first
  cache.
- `PitchDetector` (analysis readout; `PitchFollower`, `PitchFollower::track()`
  and `PitchCorrector` read through it): a dip above the threshold no longer
  reads as a note on its own. Such a dip is what a narrow band of noise or
  slow wander shows now and then, and it was all the first look asked for.
  It must now repeat at the multiples of its period inside the analysis
  window, as a tone does, and have a ground: the detections before it found
  the same period, a detection reported a note within the last six, or the
  input has just grown. While a note goes on, a period the window can test
  (two of its multiples fit: from about 67 Hz at the default window) reads
  wherever it lies, so a legato change is read at once, and its window test
  passes from a lower ratio, since a glide or a wide vibrato blurs the far
  multiples; a longer period, estimated over two or three periods, reads
  only near the note. The noise suppression no longer starts a note in a
  window with more than half its power below the lowest note it can read,
  and to start one from the input's own dip the period must repeat over all
  eight multiples almost as at the first. With the suppression off, a window
  that reads unvoiced now leaves its own dip as the candidate the next
  detections can agree with. Measured at 48 kHz at the defaults, window by
  window, before and after; in brackets, where it applies, a clean-room port
  of the frame stage of the probabilistic-threshold method followed by its
  HMM decoded causally.
  - Noise alone, 80 s each, windows read as a note. A band of noise of Q 5
    at 60, 120 and 300 Hz and 1 kHz: 24.6%, 18.7%, 9.3% and 3.9% to 15.8%,
    10.5%, 1.4% and none (35.9, 21.3, 5.4, 0.4). Of Q 2 at 60, 120 and 300
    Hz: 10.2%, 4.3% and 0.36% to 3.8%, 0.8% and 0.01% (6.8, 1.1, none). Of
    Q 10 at 60, 120 and 300 Hz and 1 kHz: 42.5%, 34.8%, 31.6% and 46.9% to
    34.6%, 30.1%, 18.2% and 9.8% (64.2, 54.8, 53.9, 70.8). Brown noise, over
    eight realisations of 80 s (59936 windows): 242 windows to 41 (38), and
    106 to 32 with the suppression off. White and pink noise: none before or
    after.
  - Notes with gaps between them in continuous noise, 7500 windows per row,
    right meaning within 50 cents. Notes of 47 to 67 Hz in pink noise at 10
    and 5 dB SNR: 97.1% to 97.7% and 87.6% to 89.8% right, 1.6% to 1.0% and
    4.3% to 1.6% wrong (at 0 dB 68.3% right either way, 1.4% to 1.2% wrong);
    in brown noise at 5 dB 87.8% to 88.9% right and 4.1% to 2.7% wrong, with
    the gaps read as a note in 0.2% of their windows before and under 0.1%
    now. Notes of 70 to 1000 Hz in pink noise at 5 dB: 98.7% to 98.8% right,
    0.7% to 0.5% wrong. The other rows read as before: white noise at 10, 5
    and 0 dB, pink noise at 10 and 0 dB from 70 Hz up, 50 cents of vibrato
    at 2 dB SNR and clean notes. With the suppression off: notes of 47 to 67
    Hz in pink noise at 0 and 5 dB 12.4% to 14.5% and 74.9% to 76.4% right
    (4.3% to 3.7% wrong at 5 dB), in brown noise at 5 dB 86.9% to 87.2%
    (4.1% to 3.8% wrong); notes of 70 to 1000 Hz in pink noise at 0 and 5 dB
    11.0% to 11.8% and 94.4% to 94.7%; the other rows as before.
  - Legato melodies of 80 to 700 Hz with 25 ms glides, 30 s each: clean,
    with 50 and 100 cents of vibrato, in white noise at 10, 5 and 0 dB SNR
    and in pink noise at 5 and 0 dB. `PitchFollower` with its gate open and
    no glide reads the same share of windows right and reaches each new note
    as soon as before in every row, with the suppression off and on, but
    two: pink noise at 0 dB with it off, 3.4% to 6.0%, and pink noise at 5
    dB with it on, 93.3% to 93.2%, where the detector itself reads one
    window more right and four fewer wrong (it no longer reads four windows
    an octave below the note, which the follower had been folding back).
    `PitchFollower::track()`: clean 95.2% to 95.6% right, 50 and 100 cents
    of vibrato 96.3% to 96.8% and 95.7% to 96.7%, white noise at 10 dB 95.6%
    to 95.9%, the other rows as before.
  - Cost per input sample, before and after in one job (GCC; Clang in
    brackets). A tone at 5 dB SNR, whose every detection now runs the window
    test: 12.3 to 16.1 ns (19.4 to 25.0). Brown noise alone with the
    suppression on: 90 to 47 ns (95 to 54); with it off, and pink noise
    alone with it off, 28 to 29 ns and 29 to 30 ns (36 to 38, 46 to 48).
    Clean tones, silence, white noise and a tone at 0 dB SNR: unchanged.
- `FFTReal` and `FFTComplex` on arm64 builds other than macOS and iOS:
  transforms of up to 4096 double or 8192 float complex points take the bits
  between their first and last passes three at a time, in a radix-8 pass
  with twiddles, where that saves a pass over the data: 256, 1024, 2048 and
  4096 complex points in double and 1024 and 4096 in float (real transforms
  of twice those sizes). The factors of such a pass sit in its table as
  whole vectors, fourteen per butterfly group, and are loaded where they are
  used. The output of those sizes changes at rounding level;
  `allocationBytes()` follows the larger table. Measured before and after in
  one job per compiler on an arm64 server core, forward plus inverse, as the
  time after over the time before at those sizes: real transforms 0.96 to
  0.97 in double and in float with GCC, 0.97 to 0.99 and 0.97 to 0.98 with
  Clang; complex ones 0.95 to 0.97 and 0.95 to 0.96 with GCC, 0.96 to 0.99
  and 0.96 to 0.97 with Clang; every other size within 1 %. The same plans
  measured 7 to 12 % slower at 512 real points on the arm64 cores of the
  macOS machines, and slower at most sizes on x86-64, whose sixteen vector
  registers do not hold the butterfly: those builds keep their plans. From
  8192 double complex points on the pass no longer paid on the server core
  either (0.96 to 0.97 with one compiler, 1.00 to 1.02 with the other).
- `FFTReal`, `FFTComplex` and what runs on them, faster wherever the kernels
  use 32-byte vectors (builds for AVX2, and x86-64 builds on a CPU with AVX2
  through the run-time dispatch). The engine's two work buffers share one
  allocation and each starts on a 64-byte boundary (the second a whole number
  of 4 KiB pages after the first once a buffer is that long). They were two
  vectors: the usual allocators hand a large block out 16 bytes past a page
  boundary, so every other 32-byte load and store of every pass lay across
  two cache lines. A copy of a transform object places its own buffers and
  carries their contents over (a transform that `inverseStep()` left half way
  goes on in the copy). A forward transform into interleaved output no longer
  multiplies its last pass by unit scales (the same values). On x86 a
  transform of 2048 double or 4096 float complex points (real: 4096 and 8192)
  closes with a radix-4 and a radix-2 pass in place of the radix-8 one: its
  eight rows sat exactly 4 KiB apart, a way of the level-1 data cache, and on
  8-way caches the pass then cost up to six times as much, by where the
  output lay; the output of those two sizes changes at rounding level, every
  other output is the same bit for bit. `allocationBytes()` follows: one
  request fewer per engine and up to 27 values more. Measured before and
  after in one job per machine, forward plus inverse over 64 to 65536 real
  points (complex: half as many points), as the time after over the time
  before. Built for AVX2 and FMA (GCC and Clang, one CPU family): real
  transforms 0.76 to 0.86 in double from 2048 points and 0.70 to 0.79 in
  float from 8192, 0.89 to 1.00 below those sizes; complex ones 0.67 to 0.80
  in double and 0.61 to 0.86 in float from 4096 points, 0.88 to 1.00 below,
  but for 1.10 at 128 float points, where the caller's buffer started on a
  32-byte boundary before the change and off one after (0.96 with it on one
  in both). Baseline x86-64 builds through the dispatch: 0.65 to 0.85 from
  those sizes and 0.90 to 0.99 below on that CPU family (GCC; 1.10 at the
  same 128 points), 0.89 to 1.05 over all sizes on another (Clang), 0.83 to
  1.02 with MSVC, whose standard library already aligns large blocks. On the
  SSE2 and NEON kernels, whose vectors never lay across two lines: 0.93 to
  1.03 from 512 points, except 1.05 (real) and 1.07 (complex) at the size
  whose closing passes change, on a CPU with a 12-way cache that the radix-8
  pass did not trouble, and 0.95 to 1.06 at 128 and 256 points; the complex
  transforms on arm64 0.93 to 0.99 (the unit scales). At 64 points (a
  transform of 70 ns) the same code moved by up to 30 % either way between
  builds and machines. Against a reference FFT in the same jobs (its inverse
  unscaled), worst size, before -> after: built for AVX2, real double 1.09 ->
  0.95 and complex double 1.27 -> 0.89 (float below 1.0 but for 1.04 at 64
  points with Clang, as before); with the SSE2 and NEON kernels the real
  transforms stay where they were (double up to 1.08 with GCC on x64 and
  1.22 on arm64), and the complex ones in double run at 0.73 to 0.89 of it
  on x64 and 0.80 to 1.01 on arm64.
- `TruePeakEstimator` and `TruePeakAnalyzer` (the true-peak modes of
  `Limiter`, `LoudnessMeter`'s true peak and what reads them): the reading
  loops (the largest grid point of every interval and the crests refined
  around the grid points) run on vectors, on the AVX2 kernels when the
  run-time dispatch is active, and give the same readings, bit for bit, on
  every path. A compiler vectorises the scalar loop only when it may run the
  vertex's division and comparisons for positions the source skips, so with
  those that honour floating-point traps the readings cost two to three times
  what they cost with the others. `processMax()` also skips, within a chunk,
  every vector whose grid points cannot reach the maximum it holds (a crest
  stands at most a quarter above the point it is centred on). Measured before
  and after in one job per machine (GCC at -O2 and -O3, Clang and MSVC on
  x64, GCC and Clang on arm64), the baseline kernels and the dispatched ones,
  float and double, as the time after over the time before. The estimator's
  per-sample readings with the Annex 2 guard: GCC 0.39 to 0.49 in float and
  0.58 to 0.67 in double with the dispatch, 0.62 to 0.70 and 0.81 to 0.87 on
  SSE2; MSVC 0.38 and 0.56, 0.57 and 0.75; Clang, which already vectorised
  the loop, 0.85 and 0.70, 0.99 and 0.86; arm64 0.65 and 0.86 (GCC), 0.98
  and 0.99 (Clang). Its running maximum: 0.37 to 0.68 and 0.52 to 0.83. The
  analyzer's per-sample readings: 0.61 to 0.85 in float and 0.78 to 1.00 in
  double (its 8x loops stay scalar at two NEON lanes, where the vector
  division cost more than the few crests of that grid); its running maximum
  0.49 to 0.84 and 0.65 to 1.00. A mono frame of `Limiter` in true-peak
  mode: GCC 0.57 to 0.76 in float and 0.67 to 0.89 in double (at -O2 on x64
  27.5 -> 16.1 ns and 34.1 -> 23.8), MSVC 0.63 and 0.70 (31.6 -> 19.9 and
  38.4 -> 26.8), Clang 0.92 and 0.76 on x64, arm64 0.73 and 0.89 (GCC,
  42.0 -> 30.9 and 57.2 -> 50.7) and 0.99 and 1.00 (Clang); with the
  high-precision reading 0.72 to 0.89 in float and 0.79 to 1.00 in double.
  No row of the 72 measured on the eight builds is slower outside the noise
  of the runners (the largest, 3 %, in 37-sample chunks on one of them).
- `FFTReal`, faster, with output that changes at rounding level;
  `FFTComplex` and the split engine read the same twiddles in another order
  and write the same output, bit for bit. A pass reads its twiddles as one
  stream: the six factors of a butterfly group side by side, or, for the
  first pass, the six rows of a block of groups. They were six arrays a
  quarter of the pass apart, which from 4096 double points on sat a multiple
  of 4 KiB from each other and met the four quarters of the pass's input in
  the same sets of an 8-way level-1 cache. The real transforms split the
  half-size spectrum, and pack it for the inverse, with one difference and
  one complex product per pair of bins k and N/2 - k, and no sum or halving:
  with d = Z[k] - conj(Z[N/2-k]) and the factor c_k = (1 - j w^k) / 2 in the
  table (its real part taken as sin^2(pi/4 - pi k / N), exact conjugates for
  k and N/2 - k), X[k] = conj(Z[N/2-k]) + c_k d and X[N/2-k] = conj(Z[k] -
  c_k d). Memory and `allocationBytes()` are unchanged. Measured before and
  after in one job per machine, forward plus inverse of the real transform
  at 64 to 65536 points, as the time after over the time before: GCC and
  Clang on x64 (SSE2) 0.93 to 1.03 in double (mean 0.96 to 0.97) and 0.93 to
  0.99 in float; built for AVX2 and FMA 0.94 to 1.03 (mean 0.96 to 0.99) and
  0.85 to 1.02; GCC and Clang on arm64 0.92 to 0.96 and 0.94 to 0.97; MSVC
  x64 0.96 to 1.11 in double (mean 1.00: its small sizes moved either way
  between runs) and 0.83 to 1.01 in float. Against a reference FFT's ordered
  real transform in the same jobs (it leaves its inverse unscaled), worst
  size, before -> after: on x64 with SSE2, double 1.12 -> 1.07 and float
  1.08 -> 1.03 (GCC), 1.09 -> 1.05 and 1.01 -> 0.98 (Clang), 1.08 -> 1.07
  and 1.07 -> 1.05 (MSVC); built for AVX2, double 1.16 -> 1.11 (GCC) and
  1.12 -> 1.10 (Clang), float below 1.0 but for 1.04 at 64 points (Clang);
  on arm64, double 1.30 -> 1.22 and float 1.21 -> 1.17 (GCC), 1.28 -> 1.22
  and 1.16 -> 1.11 (Clang).
- `wdf::DiodeRoot` for c > 0.04 (silicon diodes behind more than about
  0.7 M ohm with the default constants, germanium behind a few k ohm): the
  root is read from a series and tables for the constant of the instance,
  with no transcendental function and no division per sample, where it took
  an omega estimate and Halley steps with an exponential and a division
  each. At low drive a series to y^33 from a recurrence for the constant;
  outside the drives of the roots -1 and 1.5 the closed form from the omega
  table, with x + c, ln c and the constant term of the table each carried
  to twice the working precision and the table index formed from the wave;
  between them, on either side of the origin, local expansions of the root
  (`detail::DiodeLocal`), at most 24 cells of sixteen terms for 3000
  constants up to 1e5 in room for 32. Measured at 48 kHz on an RC circuit
  next to the closed form with a fast omega estimate (-43 to -80 dB of
  error), before and after in the same job on four machines (GCC, Clang and
  MSVC on x64, GCC on arm64), behind 1 M to 10 M ohm at 0.05 to 8 V peak (c
  from 0.05 to 0.5): the root went from 1.12 to 1.98 times the estimate's
  cost to 0.49 to 0.87 (39 to 58 ns per sample -> 17 to 27). Behind 2.2 k
  to 330 k ohm (c up to 0.018, the tabulated series and the table as
  before) it costs 0.48 to 0.81 times the estimate, within 1 % of before.
  Worst relative error against a long double reference over drives of
  either sign from 1e-6 to 1e6 times (1 + c): 3.6e-16 for c from 1e-9 to
  1e5 (3.3e-16 beyond c = 0.04) and, under reverse drive, 3.5e-16 up to
  c = 1e9, where the knee of the blocked diode (x near -c) was off by up to
  7e-14 from c = 300 on. `prepare()` builds the series and the cells
  for the circuit as it stands (2 to 24 microseconds); after a change of
  constants while running they are built again once the new constants have
  held for 256 samples, one cell per sample, and the table and the
  correction steps answer until then. An instance grows to 6.2 KB.
- `detail::DiodeOmegaTable` (both diode roots): the polynomials of -omega
  and ln(omega) fold the powers 10 to 13 of their series into the nine
  kept, each replaced by the polynomial of lower degree closest to it over
  the node, instead of being cut at degree 9. The cut left up to 4e-16 at
  the edges of the nodes nearest the branch points of omega; the fold
  leaves 1/512 of the first omitted power. For the single diode at c = 0.04
  the worst relative error where the table answers went from 3.4e-16 to no
  more than 3.0e-16; the pair's worst error over c from 1e-9 to 1e5 is
  unchanged (3.2e-16, set by its series) and its cost is the same on the
  four machines. The table also keeps what its constant terms leave out, to
  twice the working precision, and grows from 99 to 103 KB.
- `wdf::DiodePairRoot` at medium drive, and at every drive for c > 0.02
  (silicon diodes behind more than about 0.36 M ohm, germanium behind a few
  k ohm): the root comes from local expansions built for the constants of
  the instance (`detail::DiodeLocal`). Between the low-drive series and the
  drive from which the table's reverse-diode correction is short, up to 56
  cells of sixteen terms about points of the curve, found from the bits of
  the drive; beyond c = 0.02 also a low-drive series to y^35 for the
  constant, and the table root at high drive. No transcendental function
  and no division per sample is left on those paths, where the root took
  the table with its full correction (c up to 0.02) or repeated Halley
  steps with an exponential and a division each (beyond). Measured at 48
  kHz on an RC clipper next to the single-omega approximation that ignores
  the reverse diode, before and after in the same job on four machines
  (GCC, Clang and MSVC on x64, GCC on arm64), 0.05 to 8 V peak: behind 1 M
  to 10 M ohm with the default constants (c from 0.05 to 0.5) the root went
  from 1.3 to 3.1 times the approximation's cost to 0.55 to 0.81 (57 to
  124 ns per sample -> 18 to 39), where the approximation is off by -12 to
  -49 dB; behind 94 k to 330 k ohm at 0.3 to 0.5 V peak, its closest rows,
  from 0.78 to 0.96 to 0.61 to 0.86, and over every row from 2.2 k ohm up
  it now costs 0.54 to 0.88 times the approximation. Worst relative error
  against a long double reference over drives from 1e-6 to 1e6 times
  (1 + 2c): 3.2e-16 for c from 1e-9 to 1e5 (1.1e-15 before for c > 0.02);
  `c <= 1e-4` gives the same output bit for bit. `prepare()` builds the
  expansions for the circuit as it stands (10 to 45 microseconds); after a
  change of constants while running they are built again once the new
  constants have held for 256 samples, one cell per sample, and the
  previous paths answer until then, so a circuit whose values move every
  sample pays no more than before. An instance grows to 9.8 KB.
- `wdf::DiodePairRoot` for 1e-4 < c <= 0.02 (silicon diodes behind about
  1.8 k to 0.36 M ohm with the default constants) at low and medium drive.
  The exact reversion series is carried to y^35 and takes the drives above
  the short series, 1.2 to 2.5 units of nVt further (incident waves up to
  80 to 159 mV), where the root came from the table with its full
  correction or, where the table starts higher (c above 0.0062), from an
  omega estimate and a correction step with an exponential and a division.
  The longer series reaches the table start for every c up to 0.02, so that
  step is no longer used in the range. The table index is formed from the
  wave, one multiply less on the way to the table. Measured at 48 kHz on an
  RC clipper next to the single-omega approximation that ignores the
  reverse diode (-39 to -88 dB of error), before and after in the same job
  on four machines (GCC, Clang and MSVC on x64, GCC on arm64), behind 2.2 k
  to 330 k ohm at 0.05 to 8 V peak: the root cost up to 1.42 times the
  approximation (1.28, 1.19 and 1.09 on the other three machines; more than
  the approximation on 14 to 21 of the 41 rows with GCC and MSVC and on 2
  with Clang, all behind 10 k ohm or more at 0.5 V peak or less) and now
  costs 0.48 to 0.97 times it on every row, 15 to 33 ns per sample; behind
  330 k ohm at 0.1 V peak 33 to 44 ns -> 20 to 23. The roots stay within
  one unit in the last place of 50-digit references on the new path, and
  the output for c <= 1e-4 and for c > 0.02 is the same bit for bit. For
  c > 0.02 (silicon behind more than about 0.36 M ohm, germanium behind a
  few k ohm) nothing changes: the repeated Halley steps cost 1.7 to 3.1
  times the approximation, whose error there is -12 to -49 dB.
- `PitchDetector` (analysis readout) reads notes through noise at its
  defaults. Measured at 48 kHz window by window on sustained harmonic notes
  with gaps between them in continuous noise, 7500 windows per row; right
  means within 50 cents of the note.
  - A choice above the threshold yields to the first separate dip about as
    deep. In noise the dips at a period and at its multiples are equally
    deep, and the deepest was as often a multiple. Notes of 70 to 1000 Hz at
    5 dB SNR, suppression off: white noise 70.3% to 80.8% right and 17.5% to
    0.2% wrong, pink noise 77.9% to 94.4% right and 17.2% to 0.6% wrong.
  - A window that reads unvoiced and holds power below the lowest note the
    window can read is looked at again without that band (a sixth-order
    high-pass, run only over the samples such windows need). Rumble and the
    bottom of pink or brown noise fill the difference function at the lags
    where low notes dip. Notes of 47 to 67 Hz at 5 dB SNR, suppression off:
    brown noise 57.0% to 86.9% right (wrong 3.9% to 4.1%, gaps read as a
    note 0.4% to 0.1% of their windows), pink noise 61.9% to 74.9%. Brown
    noise alone reads as a note in 0.11% of windows (was 0.75%).
  - `setNoiseSuppression()` is on by default
    (`setNoiseSuppression(false)` restores the first looks alone) and no
    longer runs on a window whose own dip sits at the floor noise alone
    reaches. At the defaults, before and after: notes of 70 to 1000 Hz in
    white noise at 5 and 0 dB SNR 70.3% and 0.3% to 99.3% and 97.5%, in pink
    noise 77.9% and 7.6% to 98.7% and 85.3%; notes of 47 to 67 Hz in white
    noise 87.2% and 2.6% to 99.5% and 95.7%, in pink noise 61.9% and 8.0% to
    87.6% and 68.3%; notes at 2 dB with 50 cents of vibrato 12.0% to 96.5%.
    Against the suppression as it was when switched on: 70 to 1000 Hz at 5
    dB in white noise 80.4% to 99.3%, at 0 dB in pink noise 70.9% to 85.3%
    (wrong 4.2% to 0.2%), 47 to 67 Hz at 0 dB in pink noise 44.1% to 68.3%
    (wrong 0.8% to 1.4%). Windows read wrong: under 1% from 70 Hz up, 1.4 to
    4.3% below it in pink and brown noise.
  - Against the frame stage of the probabilistic-threshold method followed
    by its HMM decoded causally (a clean-room port), same notes: ahead in
    every row with noise (70 to 1000 Hz at 5 dB in white noise 99.3% against
    32.7%, in pink 98.7% against 80.5%; 47 to 67 Hz at 5 dB in pink 87.6%
    against 56.5%, in brown 87.8% against 85.7%), level on clean notes
    (100%). Behind it: gaps in brown noise read as a note in 0.2% of their
    windows against 0.1%, and bands of noise alone read as a note more often
    in some bands (Q 2 at 120 Hz 4.7% of windows against 1.2%, Q 5 at 300 Hz
    10.1% against 6.1%, at 1 kHz 4.6% against 0.2%) and less often in others
    (Q 5 at 60 Hz 25.1% against 36.2%, Q 10 from 60 Hz to 1 kHz 33 to 47%
    against 55 to 72%).
  - Cost, ns per sample, suppression off / on (before): a clean tone 15.4
    either way, unchanged; a tone in white noise at 0 dB 18 / 145 (18 /
    150); a low note in pink noise at 5 dB 27 / 30 (18 / 34); white noise
    alone 17 / 17 (17 / 118); pink noise alone 38 / 102 (17 / 107); brown
    noise alone 38 / 109 (17 / 55). `prepare()`-time heap 260 to 300 KB at
    44.1/48 kHz, 515 to 600 KB at 88.2/96 kHz, 1030 to 1190 KB at 176.4/192
    kHz (float). Steady clean tones read exactly as before.
  - `PitchFollower` and `PitchCorrector` keep the suppression off (the
    follower's confidence gate would discard its readings; the corrector's
    tracking was tuned on the first reading) and take the first two changes.
    On the legato melodies of the `PitchFollower` entries above, clean and
    vibrato rows read as before to the last frame. At 5 dB SNR in pink noise
    the follower at its best settings goes from 91.0% to 93.3% of frames
    within 50 cents (voicing and pitch right 87.8% to 92.9%, octave errors
    2.6% to 0.7%) and `track()` from 96.4% to 96.3% (voicing and pitch right
    96.0% to 95.9%, false alarms 0.3% to none; on a second set of seeds
    96.6% to 96.9%); at 0 dB in pink noise `track()` goes from 91.7% to
    93.2% (voicing and pitch right 85.2% to 89.3%).
- `Limiter` gain computer (no audio change): the gain a frame requires is
  taken as the ratio of the ceiling to the larger of peak and ceiling instead
  of a minimum against 1. The value is the same bit for bit (compared over
  160 million ceiling and peak pairs in float and double, zero, infinity and
  NaN included), and no minimum is left for a compiler to turn into a branch
  on whether the peak crosses the ceiling, which mispredicts on every other
  frame of a signal that is being limited. GCC did that in the float
  instantiation: `Limiter<float>` driven 6 dB over the ceiling goes from 13.8
  to 8.7 ns per mono frame in sample-peak mode there (stereo 13.0 to 10.3),
  level with `Limiter<double>` (7.9 and 9.6).
- `PitchDetector` (default path, analysis readout): each threshold of the
  vote picks the deepest minimum of the stretch of lags below it instead of
  the first minimum there, so the small minima noise leaves on a broad
  dip's walls no longer split its vote. 48 kHz, readings within 50 cents,
  harmonic tones in white noise: 47 to 67 Hz at 10 dB SNR 68.5% to 100%
  (31.5% had read 50 to 140 cents sharp at confidence 0.81 to 0.89), 70 to
  1000 Hz at 10 dB 92.0% to 100% and at 5 dB 85.2% to 97.0% (octave errors
  5.5% to 2.0%); in pink noise 47 to 67 Hz at 5 dB 70.0% to 79.5% right,
  14.0% to 4.5% wrong. Steady clean tones, sweeps, noise alone and
  band-limited noise read exactly as before; on clean stepped melodies at
  three rates the wrong readings fall from 52 to 46 of 66090, every
  correct reading kept. The prior's survival is read from a table (cubic
  Hermite, error below 1e-9 at the default threshold) and the vote's
  records are merged on a stack:
  CPU 16.8 to 17.0 ns per sample on a clean tone, 18.7 to 19.1 on a tone in
  noise.
- `PitchDetector::setNoiseSuppression()`: where the cleaned window
  does not dip below the threshold, the input's own dip is the candidate;
  either candidate is refined from the correlation peaks at its multiples
  over the two windows of history, and a reading is kept only when two of
  the three detections before it found the same period (within 25 cents
  per detection of age). At 0 dB SNR: 70 to 1000 Hz in white noise 84.5% to
  99.8% right (0 wrong), 80 to 1000 Hz in pink noise 74.8% to 80.8% (1.8%
  wrong both), 47 to 67 Hz 40.5% to 99.5% in white noise (0.5% to 0
  wrong) and 12.5% to 51.5% in pink (6.0% to 1.5% wrong); continuous notes
  at 2 dB 88.7% to 99.6%, and with 50 cents of vibrato 83.4% to 99.6%.
  Band-limited noise alone voices at most 0.1% of windows more than the
  first read (up to 2.0% before; a Q 2 band at 120 Hz 6.4% to 4.4%, off
  4.4%). A reading through the noise starts one or two detections after
  the tone. CPU with it on: 89 to 90 ns per sample on noise, 100 to 110 on
  a tone in white noise and 80 to 91 on a low note in pink noise, which it
  now reads.
- `PitchCorrector` (audio): its pitch readings follow the `PitchDetector`
  vote change above. Clean input is corrected exactly as before, bit for
  bit; on a 196 Hz melody at 10 dB SNR 4 of 265 detections read
  differently, on a 55 Hz melody 228 of 267 (by 14 cents on average, the
  earlier readings pulled off the period by noise), so its output on noisy
  low notes differs from the previous release's.
- `PitchFollower`: a large interval, an octave leap among them, arrives once
  two consecutive readings agree on it (three before); a single reading
  whole octaves away is still folded back. Before, every octave-away
  reading was folded, so a melody's octave leap never came through and a
  wrong octave read at a note's start stayed: on legato melodies with an
  octave leap in one step of eight, 21.9% of voiced frames an octave off,
  0.7% now; notes whose fundamental is 15 dB below its partials 28.6% to
  0.7%. Readings within 50 cents at the defaults 68.9% to 89.4% (clean
  legato), time to a note change 104 to 48 ms.
- `Resampler` two-stage engine (opt-in): the stage-2 kernel length is a
  multiple of four, so no product has a scalar tail at any vector width (High
  24 taps at every ratio, 22 before when downsampling; Ultra 36 instead of 34
  when upsampling and 32 as before when downsampling; Normal 20 instead of
  18), and stage 2 computes the outputs of up to eight phase cycles together
  with `simd::dotProductStrided()`, in groups whose windows stay in the
  first-level cache while every kernel row meets them. ns per input sample
  (MSVC x64, AVX2 dispatch, double, into caller storage) at 96 -> 44.1 /
  48 -> 44.1 / 44.1 -> 48 / 44.1 -> 96 kHz: Ultra 2.36 / 3.78 / 4.90 / 7.85
  -> 2.25 / 3.61 / 3.99 / 5.98, High 2.32 / 3.94 / 3.67 / 5.44 -> 2.06 /
  3.43 / 3.39 / 5.07; without the dispatch (SSE2) Ultra 4.31 / 7.39 / 8.22 /
  12.83 -> 3.64 / 6.05 / 6.64 / 9.66. Against a reference resampler of the
  same two-stage structure at its 32-bit setting, Ultra: 1.21 and 1.06 times
  its time at 44.1 -> 96 and 44.1 -> 48 kHz before, 0.92 and 0.87 now (96 ->
  44.1 kHz 0.99 -> 0.94). Output: where the kernel grew it moves by the
  kernel's own residual, far below the tier (-127 dB re peak at Normal, -167
  dB at High, -228 dB at Ultra); elsewhere the rational ratios move at
  rounding level (-309 dB), and the 2:1 and 1:2 paths, non-rational ratios
  whose kernel kept its length and the Direct engine are bit-identical. Tone
  residuals are unchanged or lower (19 kHz at 96 -> 44.1 kHz, High: -182.5
  -> -182.7 dB re the tone) and `processRange()` spans still agree bit for
  bit.
- FFT engine (`FFTReal`, `FFTComplex` and the split engine the two-stage
  resampler runs on), faster with the same output, bit for bit, in every
  build: the passes keep their pointers in registers and load each twiddle
  where it is used (no twiddle vector is spilled); a pass whose stride is one
  block broadcasts its twiddles per block; the radix-8 closing pass takes its
  sums and its differences in two sweeps over its inputs; at the AVX width the
  first pass reads interleaved input without crossing 128-bit halves (its
  twiddles follow the permuted lanes: `FFTReal` and `FFTComplex` keep a
  permuted copy of the first pass's twiddles in x86-64 builds, 0.75 N and
  1.5 N values), and a double transform that ends in interleaved output keeps
  its blocks in that lane order, so its closing pass interleaves within halves
  too; transforms of one pass (2 or 4 complex points) run inline. A transform
  runs all of its passes on one kernel set, and `FFTReal::inverseStep()` keeps
  the set its first engine step ran on, so a `simd::setRuntimeDispatch()`
  change between steps no longer mixes the two sets (the steps then equal
  `inverse()` on that set, bit for bit). Forward plus inverse, double, real:
  13-15% faster with the AVX2 dispatch from 256 to 4096 points (5% at 128,
  2-9% above 4096), 3-11% on SSE2; complex 15-22% and 4-16%; float real 1-10%
  and 5-12% (parity at 64 and 128 points with the dispatch).
  Against a reference FFT's ordered real transform, double, both built for
  AVX2: 1.17 / 1.12 / 1.17 times its time at 256 / 512 / 1024 points before,
  1.00-1.02 / 0.93-0.95 / 0.99-1.00 now (0.77-0.97 at the other sizes from 64
  to 16384); on SSE2 1.09 / 1.05 / 1.04 at 1024 / 2048 / 4096 (1.11 / 1.09 /
  1.07 before), 1.02 at 16384 and below 1.0 at the other sizes.
- `Resampler` two-stage engine (opt-in): its stage-2 interpolation computes
  outputs one phase cycle apart together, four at a time with
  `simd::dotProduct4()`, so each kernel row is read once for four windows, and
  the table mode reads its four phases in one call. Output unchanged, bit for
  bit. With the faster transforms, ns per input sample (MSVC x64, AVX2
  dispatch, double): Ultra 96 -> 44.1 kHz 2.60 -> 2.32, 48 -> 44.1 4.31 ->
  3.74, 44.1 -> 48 5.15 -> 4.88, 44.1 -> 96 8.11 -> 7.75, 96 -> 48 1.41 ->
  1.30; High 2.47 -> 2.30, 4.19 -> 3.87, 4.25 -> 3.58, 6.33 -> 5.33, 1.27 ->
  1.18.
- `LoudnessMeter` true peak (and `AudioIntervalAnalyzer`'s, which reads it):
  read with `TruePeakAnalyzer` instead of the BS.1770 Annex 2 example table.
  Against the exact band-limited reconstruction (the table in brackets):
  steady sines within 0.004 dB up to 0.49 fs (up to +0.22 dB near 0.09 and
  0.25 fs); tones that start abruptly within 0.01 dB up to 0.45 fs
  (-0.78 / +0.19), -0.09 / +0.05 dB up to 0.485 fs and -0.21 / +0.08 dB at
  0.49 fs (-2.8 / +0.23 above 0.45 fs); noise band-limited to 0.45 fs within
  0.003 dB (-0.37 / +0.04); white noise -0.19 / +0.13 dB (-1.08 / +0.40).
  Default readings move by up to -0.22 dB on steady tones and by up to
  +2.8 dB on content between 0.45 fs and Nyquist, which the table
  attenuates. A sample's own magnitude still counts at once; the
  reconstruction around it is read 261 samples later and at the end by
  `finalizeTruePeak()`, in single precision for every sample type. CPU 0.49
  to 1.85 ms per second of 48 kHz stereo with all measures (2.1 while the
  level keeps rising; 2.5 and 2.7 on SSE2) and 16 KB more per meter. The
  conformance program reads two abrupt tones (0.47 and 0.48 fs) against
  their exact reconstruction peak on every target.
- `LoudnessMeter` loudness range: every short-term value is held to 0.005 LU
  (was 0.1 LU) and the relative gate is set from the exact mean energy, so
  LRA matches the EBU Tech 3342 reference code run on the meter's own 10 Hz
  short-term series within 0.005 LU (was up to 0.1 LU: 0.056 to 0.097 on the
  test programmes). Default readings move by up to 0.1 LU. The short-term
  rate stays 10 Hz, as Tech 3342 has required since V3: a 1 Hz series reads
  a programme stepping every 3 s 1.0 LU higher and moves by up to 1.5 LU
  when it starts a fraction of a second later. 48 KB more per meter, still
  constant and held in the object.
- AU: with `kAudioUnitProperty_ShouldAllocateBuffer` false on the main input
  and on the output, the unit releases its main pull buffers (two channels
  of MaximumFramesPerSlice floats: 9.2 KB at 1156 frames); the source's
  buffers reach the host's directly, a missing source renders silence into
  them, and a render without a host output buffer is refused with
  `kAudioUnitErr_InvalidPropertyValue`, as the format's reference
  implementation does. Other combinations keep the buffers they can need.
- `OnsetDetector` gated picker (the default; analysis readout, no audio): a
  peak whose frame energy falls across the release test is also kept when
  bands it raised at least double their amplitude over anything they or
  their neighbours held in the two analysis frames before, and still hold
  that energy twice the test's look-ahead after it, by 0.3% of the energy
  after the peak: a quiet note starting while a louder one is released. The
  share is computed only for candidates (allocation free; CPU unchanged at
  0.88 ms per second of audio; 16 KB more held from prepare()). The envelope
  of `getLastOdfFrame()`, the BeatTracker and `Picker::Plain` are unchanged,
  bit for bit, and so is the latency. On 24 held-out files per kind at 44.1
  kHz, offline (F at +/-50 ms): 60 ms attacks 0.946 to 0.960 (0.952 to 0.964
  and 0.965 at 48 and 96 kHz; causal 0.951/0.952/0.955 to
  0.965/0.964/0.969), 30 ms attacks 0.981 to 0.987, 1 ms 0.999 to 1.000,
  slow attacks over pink noise 0.803 to 0.805; the false onsets of every
  kind are unchanged on the held-out files and on two further sets of 24
  files per kind. Quiet attacks 15 to 45 ms before a louder note's end:
  10/7/8 of 56 to 53 offline and 52 causal at 44.1/48/96 kHz.
- `PitchDetector::setNoiseSuppression()`: the multiples of a period
  are tested over the last two analysis windows of input instead of the
  window alone (the input buffer keeps two windows: 32 KB more at 48 kHz).
  Notes below 67 Hz at 48 kHz, which left no multiple inside the window and
  were never read through the noise, now are: 47-67 Hz tones at 0 dB SNR
  0.5% to 40.5% right in white noise (0.5% wrong both), 3.0% to 12.5% in
  pink (1.0% to 6.0% wrong, all within 85 cents, confidence below 0.75).
  Narrow bands of noise voice less: over 20 s, a Q 2 band at 120 Hz 9.8% to
  6.4% of windows (off 4.4%), at 300 Hz 2.0% to 1.3% (off 0.3%), a Q 5 band
  at 1 kHz 5.5% to 4.3% (off 4.0%). Tones of 70 to 1000 Hz at 0 dB in white
  noise: 84.5% right as before, wrong 0.2% to 0; 80 to 1000 Hz in pink 75.2%
  to 74.8% right, 3.0% to 1.8% wrong; continuous notes at 2 dB 88.7% as
  before, 86.8% to 83.4% with 50 cents of vibrato. Off, the output is
  unchanged, bit for bit.
- FFT engine (`FFTReal`, `FFTComplex` and the split engine the two-stage
  resampler runs on): between passes the data sits in blocks of 4 double or
  8 float real parts followed by the same elements' imaginary parts, so a
  pass keeps one pointer per stream; an odd log2 size closes with one
  twiddle-free radix-8 pass instead of a radix-4 and a radix-2 pass while the
  radix-8 rows stay under 4 KiB apart (complex sizes 32 or 128 to 2048); the
  real transforms take bins k and N/2 - k together in their pre and post
  passes, on a twiddle table that is an exact mirror (w^(N/2-k) =
  -conj(w^k)), with one shared product per pair. Double, forward plus
  inverse, ns per real point at 64 / 1024 / 4096 / 65536 points: SSE2
  2.0 / 2.4 / 2.8 / 3.7 -> 1.5 / 1.9 / 2.3 / 3.2, AVX2 dispatch
  1.5 / 1.3 / 1.5 / 2.7 -> 1.1 / 1.0 / 1.2 / 2.2; float SSE2
  1.4 / 1.3 / 1.5 / 1.9 -> 1.2 / 1.0 / 1.2 / 1.6, AVX2 dispatch
  1.4 / 0.85 / 0.88 / 1.3 -> 1.0 / 0.62 / 0.66 / 1.0. The two-stage
  resampler's transforms are 15% cheaper with the AVX2 dispatch and 4-9% on
  SSE2 (Ultra 96 -> 44.1 kHz 2.83 -> 2.66 ns per input sample). Output: the
  double real transforms move at rounding level (at most 1.7 ulp of the
  spectrum's peak) through the mirrored twiddles, and the sizes that close
  with the radix-8 pass (and the real transforms of twice those sizes) move
  by at most 3 ulp of the peak in either type; everything else is
  bit-identical where the arithmetic is not fused, and moves by at most
  1.2 ulp of the peak in a few bins where it is (the vectorised pre and post
  passes now cover every bin). The RMS error against an exact transform is
  unchanged or lower at every size (double round trip at 65536 points 2.41
  -> 2.36 epsilon); the two-stage resampler's double output moves by at
  most -303 dB re its peak, its float output not at all.
- `WavetableBank`: the two bass pairs whose level sizes differ (8192 and
  4096 samples, 4096 and 2048) keep a copy of the duller level on the
  brighter one's grid, so their crossfade blends the samples and
  interpolates once like every other pair. `readBlock()` float at 33 and
  60 Hz, SSE2: 2.99 -> 2.35-2.42 ns per sample (the 4-point cubic reader of
  one table: 2.38-2.42); AVX2: 2.10 -> 1.71. Memory 229 -> 279 KB per frame.
  Output moves only in those two pairs (27.4-38.7 and 54.7-77.4 Hz at
  48 kHz), by at most -138 dB re full scale; the worst inharmonic component
  stays at -165.6 dB for the saw, and for a flat spectrum it is -149.0 to
  -153.4 dB inside the pairs (before -148.2 to -151.2) and -141.1 dB at
  their tops (before -136.5).
- `Saturation` `Antialiasing::SecondOrder` at the default 2x adds two base
  samples instead of one (48 kHz: getLatency() 65 -> 66, reported as soon as
  the order is set; 1x and 4x and above unchanged). Every algorithm takes
  the same delay, so crossfades between algorithms, the dry path, Delta and
  the unprocessed M/S channel stay aligned; Off and FirstOrder are
  bit-identical, and the other algorithms' SecondOrder output is the
  previous one a sample later. The room lets MultiStage use the transformer
  kernel with a 1.5-sample linear-phase interpolator from +16 dB of drive,
  blended by drive from +14 dB (the previous two-sample cascade below, its
  output a sample later). Worst alias over 1-19.25 kHz tones at
  -6/-18 dBFS: +24 dB -28.0 -> -36.2 dBc, +18 dB -31.1 -> -33.1, +36 dB
  -28.4 -> -33.4, +16 dB -33.3 -> -34.1, unchanged to +14 dB (-37.1 at
  +12). Harmonic error at 1 kHz against the curve at 16x: +12 dB
  unchanged (-47.4 dB), +24 dB -38.4 -> -38.0. The fundamental at
  19.25 kHz under +24 dB drive is 1.3 dB lower (the kernel's droop). Both
  transformer forms run on every sample: MultiStage costs 860 -> 1000 ns
  per stereo frame at 2x.
- `TubePreamp` `GridModel::Conduction` (opt-in) at 2x and above: inside each
  internal sample the coupling capacitor's charge by grid current is
  integrated piece by piece from the grid currents at the integration nodes
  (predicted, then corrected once), the stage-1 leak toward the source is
  exact in the source polynomial, and stage 1 is split once more halfway
  between cutoff and 3 V below the cathode (where the colder-biased stage
  starts its plate-current pulses). Before, the capacitor ramped linearly
  across each sample at 2x and was held at its predicted midpoint from 4x.
  Worst alias over the documented sweep at 2x (2 s of settling), one stage
  +12/+24/+36 dB: -85.5/-74.0/-74.0 -> -102.8/-88.2/-83.2 dBc (Clamp
  -87.7/-80.6/-80.6); two stages -81.6/-73.1/-73.1 -> -90.5/-88.1/-84.1
  (Clamp -83.0/-83.0/-84.8); at 4x one stage +24/+36 -88.0/-84.9 ->
  -88.9/-86.6, two stages -84.0/-81.7 -> -89.3/-84.9. The core error
  against the point circuit at 32x is unchanged (worst -56.7 dB against
  -56.6). Cost, stereo float at 2x: +9..+33 % against the previous
  Conduction model (most for high tones at +24 dB); against Clamp +14..+49 %
  on a 1 kHz + 7.3 kHz program (before +3..+25 %) and +66 % on a 16 kHz
  tone at +24 dB, two stages (before +26 %). 1x and the Clamp default are
  bit-identical.
- `wdf::DiodePairRoot` for 1e-4 < c <= 0.02 (silicon diodes behind about
  1.8 k to 0.36 M ohm with the default constants): the table root starts
  where the reverse-diode correction q = c^2 / (omega (1 + omega)) reaches
  3.5e-3 instead of 3e-5,
  with that correction to seventh order (in q and c^2 / (1 + omega)^2,
  scaled by the instance) and the table's R and 1 / (1 + omega)^2 terms
  extended to degrees 9 and 6 (`detail::DiodeOmegaTable` 83 -> 99 KB); up
  to c of about 6e-3 (110 k ohm) the table now reaches the low-drive
  series, so the omega estimate and its correction step are no longer used
  there. Cost
  per sample on an RC clipper behind 94 k ohm: 28.6 -> 22.6 ns at 0.5 V
  peak, 27.2 -> 24.0 ns at 2 V, 24.9 -> 24.5 ns at 8 V. The roots stay
  within the previous error against 45-digit roots (worst 2.7e-16 relative
  for c from 1.5e-4 to 0.0195; 9.3e-16 before at c = 0.0195); c <= 1e-4
  (707 ohm with the default constants) is bit-identical.
- `TubePreamp` at 4x and above decimates through `Oversampling::Depth::Deep`
  (every stage at least 100 dB deep; 2x keeps the High preset): latency at
  4x/8x/16x 73/74/75 -> 93/94/96 base samples, the dry path follows. The
  later decimation stages were only 80 dB deep and set the alias floor
  there; worst alias over the documented sweep (2 s of settling), Clamp at
  4x, one stage +12/+24/+36 dB -101.4/-93.0/-92.1 -> -102.0/-96.4/-96.4 dBc
  and two stages -91.3/-89.4/-88.9 -> -93.9/-93.9/-105.8 (8x: one stage
  -101.4/-92.1/-91.2 -> -118.7/-109.9/-96.8, two stages
  -92.1/-90.3/-89.9 -> -109.6/-105.7/-105.7); `GridModel::Conduction` at
  4x one stage -102.8/-88.9/-86.6 -> -120.2/-114.1/-105.5 and two stages
  -94.4/-89.3/-84.9 -> -115.2/-103.1/-102.6. Cost at 4x within 1 % (Clamp)
  and 6 % lower (Conduction, with the changes below). 1x and 2x, and so the
  default, are bit-identical.
- `Saturation` from 8x and `Clipper` at 16x decimate through
  `Oversampling::Depth::Deep` (every stage at least 100 dB deep; the lower
  factors keep the High preset, where the folding of the curve itself is what
  is left): latency of `Saturation` at 8x/16x 73/74 -> 93/95 base samples and
  of `Clipper` at 16x 74 -> 95, the dry path follows. Worst alias over tones
  of 1 to 19.25 kHz at -6 dBFS with +24 dB of drive: the wavefolder at 8x
  -75.2 -> -104.3 dBc and at 16x -74.7 -> -105.3; with second-order
  antialiasing at 16x SoftClip -95.1 -> -119.3, Tape -86.4 -> -114.1,
  HardClip -94.9 -> -114.7 and the `Clipper` modes -94.9..-95.4 ->
  -114.7..-119.4; at +6 dB the floor goes from -111 to -128 dBc. Cost:
  `Saturation` within 1 %, `Clipper` at 16x 7 to 13 % more. `DynamicEQ`
  keeps the preset: the deeper stages change nothing there (-122.8 dBc
  either way). `Saturation` up to 4x and `Clipper` up to 8x, and so the
  defaults, are bit-identical.
- `TubePreamp` cost: the tone circuit's phi functions of a whole internal
  sample are kept per mode instead of recomputed (bit-identical, both grid
  models), the conduction model's grid table stores each cell's cubic
  (bit-identical), and the conduction model integrates stage-1 pieces
  shorter than 0.03 internal samples on two nodes and those shorter than
  0.1 on three, as stage 2 already did. Stereo float at 2x, % of one core,
  Conduction on 1 kHz + 7.3 kHz / 16 kHz / white noise at +0..+36 dB:
  3-15 % lower (16 kHz, one stage, +36 dB: 10.33 -> 8.82; previous
  conduction model 7.94), so its cost over the previous conduction model
  falls from up to +35 % to up to +24 %; Clamp 0-8 % lower. The
  conduction model's output at 2x moves at the level of its numerical
  error: worst alias over the documented sweep unchanged within 0.1 dB
  except two stages at +36 dB, -84.1 -> -84.5 dBc.
- `wdf::DiodePairRoot` for 1e-4 < c <= 0.02: on the table nodes where the
  reverse-diode correction q is below 4.05e-5 and c^2 / (1 + omega)^2
  below 1.47e-5 (from z = 0 on behind 94 k ohm) the correction keeps
  q^3, r q^2 and r^2 q, three multiply-adds after R instead of seven
  (omitted terms below 9e-18 of the root). Cost per sample on an RC
  clipper behind a 94 k ohm port: 24.7 -> 23.8 ns at 0.5 V peak, 25.0 ->
  23.3 at 2 V, 24.1 -> 23.7 at 8 V (single-omega approximation 23.8, 28.2,
  29.1); 22 k ohm 24.4 -> 23.0, 25.0 -> 23.2, 24.0 -> 23.6. The roots stay
  within the previous error against 45-digit roots (worst 2.9e-16 relative
  for c from 1.5e-4 to 0.0195, unchanged at every c); the other nodes are
  bit-identical.
- `wdf::DiodePairRoot` and `wdf::DiodeRoot` above the low-drive series
  evaluate the diode's closed form with a full-precision omega table
  (`detail::DiodeOmegaTable`, 83 KB, built once per process) instead of an
  omega estimate and a correction step with an exponential and a division;
  the pair's reverse diode is a correction read from the same table point.
  The roots stay within one unit in the last place of 50-digit roots (as
  before); RC-clipper output moves by at most -287 dB re its RMS. Cost per
  sample on an RC clipper at 0.5-8 V peak, default constants: pair 28-30 ->
  19-20 ns behind 707 ohm and 30-32 -> 23-25 ns behind 5-9 k ohm; single
  diode 26-27 -> 13-16 ns. Low drive (series) unchanged.
- `LadderFilter` `Saturation::Stages` (opt-in): above the state-start
  cutoff Newton now starts from the drive-free ladder solved exactly while
  its stages stay within two knee widths (the fixed-pivot start otherwise),
  and channel pairs are solved together. Mean cost over cutoff, resonance,
  drive and signal falls from 2.7 to 2.2 Newton-step equivalents, worst 8 ->
  7 steps; mono at a 15 kHz cutoff with drive 3: sines 76 -> 58 ns, a saw
  72 -> 54 ns, white noise 104 -> 98 ns per sample (at drive 10 white noise
  rises 103 -> 110 ns); stereo 10-20 % less per channel. Below the
  state-start cutoff mono output is bit-identical; above
  it the output moves within the solver tolerance.
- `TubePreamp` `GridModel::Conduction` (opt-in) at 2x: the coupling
  capacitor's voltage ramps across each internal sample instead of being
  held at its predicted midpoint and stepped at every boundary. Worst alias
  over the documented sweep: one stage -83.4/-72.7/-68.3 -> -85.5/-74.0/-74.0
  dBc at +12/+24/+36 dB, two stages -77.1/-68.2/-65.5 -> -81.6/-73.1/-73.1.
  The output moves by -88 to -54 dB RMS (0 to +36 dB drive), within the
  model's error against the dense point circuit, which is unchanged; the
  blocking figures are unchanged. 1x and 4x and above are bit-identical;
  the Clamp default is untouched.
- `OnsetDetector` gated picker (the default; analysis readout, no audio):
  it picks the SuperFlux compared with the frame a quarter of the analysis
  frame back (two hops under the automatic frame, as the published method
  derives it from the window) instead of the previous frame; the envelope of
  `getLastOdfFrame()` and `Picker::Plain` are unchanged, bit for bit. A peak
  within 0.0045 of its threshold must also show a new sound: its band rise
  across the release test's span, against the largest of the bands over the
  0.43 frames before, must be 2.1 times its moving mean, with a 2 dB rise of
  the frame energy unless the ratio reaches 3. The release test excuses a
  fall of the frame energy when one register doubles its energy (by 3% of the
  frame's), and the causal moving mean extends over the frames that test
  waits for (latency unchanged). On 24 held-out files per kind at 44.1 kHz,
  offline (F at +/-50 ms): 60 ms attacks 0.909 to 0.946 (0.952 at 48 and 96
  kHz; causal 0.918 to 0.951, 0.952 and 0.955), 30 ms 0.966 to 0.981, slow
  attacks over pink noise 0.621 to 0.803, sharp attacks over a quiet noise bed
  0.978 to 0.989, note releases 0.999 to 1.000 with no false onset, clicks
  over 0 dB pink noise 0.995 (0.02 per second) unchanged, percussion 0.965 to
  0.964. `getOnsetStrength()` reports the picked function's value. The
  OnsetDetector costs 0.82 instead of 0.78 ms of CPU per second of audio
  (its flux loops are now branch free).
- `BeatTracker` causal beats (the published tempo and confidence and the
  offline path are unchanged): the beat score reads the log-band envelope
  and the linear spectral flux, each over its own running peak, and the comb
  stage also reads the correlation at half the period, under a Rayleigh prior
  at 0.40 s (0.45 before). Beats are attributed one hop before the frame
  they are placed on, where the linear flux puts the transient, so
  `getLatencySamples()` grows by one hop (549 to 770 samples at 44.1 kHz).
  Causal beat F (+/-70 ms after 5 s): ballroom 0.706 to 0.740 and 0.728 to
  0.757 on the two halves of 698 recordings; first 30 s of the Salsa Dataset
  0.326 to 0.410 (0.399 and 0.420 on its halves; the four-thirds metrical
  error on a tresillo falls from 31 of 122 recordings to 3); the three
  syncopated grooves (0.980, 0.984, 0.986) and click trains from 40 to 240 BPM
  unchanged; the off-beat sweep keeps all 660 beats within 70 ms (659
  before), a mean 1.2 to 2.0 ms late (2.6 to 4.0); beat placement on clean
  clicks median 2.1 ms, worst 6.3 ms (3.6 and 6.7); mean offset from the
  annotated beat on 200 ballroom recordings 3.7 ms (4.3). CPU 1.69 to 1.78 ms
  per second of audio.
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
- `Limiter` figures measured again (no audio change). The true-peak overshoot
  is read with `TruePeakAnalyzer`, which also resolves content above 0.45 of
  the sample rate: at the defaults and a -1 dB ceiling a 48-partial mix
  driven 6 or 12 dB over the ceiling comes out +0.004 dB over it, a
  hard-clipped 5 kHz tone +0.19 dB, white noise +0.77 dB and differentiated
  white noise +1.10 dB, where the 128-tap reading above gave +0.27
  (sample-peak mode: +0.66, +2.56, +5.92 and +4.24 dB). The intermodulation
  figures now state what sets them, the length of the attack ramp: 60 Hz +
  7 kHz at the 2 ms default reads -59.1 dBc in true-peak mode and -61.9 in
  sample-peak mode, and -64.1 dBc in true-peak mode with `setLookahead(2.7)`,
  which gives the ramp its 96 samples back for 129 of latency.
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

- `TonalBalance`: what lay far under the programme was raised like a hole in
  it. A programme with nothing over 4 kHz, on a noise 50 dB under it, had
  that noise raised by the whole range (6 dB over 8 kHz). A band more than
  `setFloor()` dB under the loudest one no longer takes the balance nor the
  offset that keeps the loudness (0.0 dB there now), and the last 10 dB
  over the floor fade in; a dip between two sides that sound is filled as
  before, and a dark programme is still given its top back.
- `GraphicBank`: a sample that is not a number stayed in its sections for
  good. It now leaves as silence and the sections start again from rest:
  the next block comes out as from a bank that never saw it.
- `ResonanceSuppressor::setSensitivityBand()`: a new sensitivity curve was
  written over the one in use an eighth at a time, so for seven hops the low
  bins were weighed by the new curve and the high ones by the old. It is
  computed aside and taken whole.
- `TapeMachine`: DC at the input reached the tape and leaned its loop to one
  side, a second harmonic that no later stage removed (the output coupling
  only took the DC itself): a 1 kHz tone at -12 dBFS over a DC of 0.1
  carried it at -44 dB, and at -86 dB over a 0.5 Hz drift of 0.1. The input
  is now AC coupled, as a machine's line input is (two one-pole high-passes
  at 4 Hz in series, critically damped): -147 and -98 dB. The low end gives
  0.34 dB at 20 Hz, 0.15 at 30 and 0.08 at 40; nothing from 60 Hz up.
- `TapeMachine::setBias()`: under the optimum the top end did not follow the
  bias (10 kHz against 1 kHz stayed within 0.1 dB from 0.1 to 0.55), and
  over it the self-erasure came in at once, a step of 1.2 dB at 0.57. Under
  the optimum the top now rises as the bias falls (10 kHz 2.0 dB up at 0.2
  and 2.6 at 0.1, a shelf that follows the speed), and over it the top
  darkens from the optimum on, without a step (0.8 dB down at 0.55). At the
  optimum, and from 0.6 up, the response is what it was.
- `Delay` with `Character::Bucket`: the device's filters and its hold added
  their own delay to the chain's, so an echo of 300 ms sat at 300.49 ms and
  each repeat as much again. The chain's clock now takes that delay off:
  the echoes of a 300 ms delay sit at 299.995 and 600.014 ms (100 ms:
  99.993 and 200.006; 600 ms: 599.994 and 1200.014).
- `Saturation::setCombLoop()`: a new frequency moved the read point of the
  feedback line at once, by a whole number of samples: a cut in a signal
  that recirculates, a click each time (a tone under a loop of 0.9 taken
  from 200 to 150 Hz: a step 62 times the largest of the steady signal).
  The delay now glides to its new length, an octave in a tenth of a second,
  read between samples on the way: no step. A loop that is not moving runs
  as before, to the bit.
- `Compressor`, `NoiseGate`, `Expander`: the listened key (`setSidechainListen()`)
  left without the delay of the lookahead, so with a lookahead it played
  early by the whole latency the processor reports and the track moved when
  the listen was switched; and an equalized key went out as loud as its
  bands made it (a tone at half scale under a bell of +24 dB: 7.9 times full
  scale). It now leaves late by the latency and is held under full scale.
  Under full scale and without a latency it is the key itself, as before.
- `Compressor` with a latency kept for the lookahead (`setMaxLookahead()`):
  moving the lookahead jumped the gain to another point of its own past, a
  step of level whenever the gain was moving (a level swelling at 3 Hz, the
  lookahead taken from 8 to 0 ms: a step 370 times the largest of the steady
  signal). The delay of the gain now slides to its new value a quarter of a
  sample per sample: no step.
- `ResonanceSuppressor`: a link between 0 and 1 behaved as 0, because each
  channel designed its filter without the power of the other. With a line on
  the left only, the right channel took 1.3 dB at the line unlinked
  and 22.2 dB linked, and the same 1.3 dB at any link in
  between; it now takes 12.2 dB at a link of one half. The level each
  channel is read by goes that share of the way to the level of the pair's
  mean power, in decibels in `Mode::Smooth` and in amplitude in `Mode::Deep`.
- `ResonanceSuppressor` with `Quality::Offline`: the detector's smoothings
  were counted in hops and in bins, so with four times the hops and four
  times the bins of `Quality::Realtime` it smoothed a quarter as long and a
  quarter as wide, and found other resonances (pink noise with three lines:
  the reductions of the two qualities 1.3 dB apart on average over the
  curve, 4.5 dB at the worst point; a 500 Hz line taken 9.9 dB down in one
  and 14.1 dB in the other). They are now times and widths in hertz: 0.04 dB
  apart on average, 1.0 dB at the worst point, the line 9.9 and 9.4 dB down.
  `Quality::Realtime` is unchanged to the bit.
- `SpectralDenoiser` in filter mode with a link between 0 and 1: the gains of
  the two channels were mixed, the first channel reading the second one's
  gains of the frame before, so the channels were not treated alike
  (swapping the channels of the programme did not swap the output) and a
  tone on one channel lost 2.3 dB at a link of one half. The link now acts
  on the power each channel is read by, within the same frame: the output
  swaps with the input to the bit and the tone keeps its level (0.17 dB).
- `TonalBalance`: the cross term of the stereo covariance kept only its real
  part, so with one channel a delayed copy of the other the band powers took
  a comb and the corrections followed it (a delay of a quarter of a
  millisecond: corrections up to 0.78 dB away from those of two equal
  channels; now 0.19 dB). The bank's solution was refined one hop in eight
  and left plain on the other seven, so on a steady programme the response
  stepped with a period of eight hops (0.09 dB at 9 kHz with corrections of
  12 dB; now under 0.001 dB: the hops in between carry the correction). And
  the loudness offset was scaled by each band's weight and then limited,
  which left the total loudness up to 0.02 dB away; it is now solved with
  the limits and the weights in place. The analysis costs 9 % more.
- `DeEsser` and `DynamicEQ`: the detector's band-pass (and the low-pass and
  high-pass of the shelves) was a bilinear design, which narrows toward
  Nyquist, so near the top of the band the detector heard a narrower band
  than the one it moves. A tone half an octave under a 10 kHz band and one
  half an octave over it, at the same level and at 44.1 kHz, read 5.2 and
  3.2 dB of reduction in the de-esser and 4.2 and 2.7 dB in the equalizer;
  with the detector matched to its analog prototype they read 6.5 and 6.6,
  and 5.3 and 5.3. Low bands read as before (2 kHz: within 0.1 dB).
- `DynamicEQ` with mixed bands: changing the range of a band at work moved
  the weight of its end at once while the end's filter took its 20 ms, a
  step of level on the way (a tone held 6 dB down, its range set to 18 dB,
  came up to -3.5 dB first). The weight now follows the filter's glide: the
  tone goes down without rising.
- `SpectralDynamicEQ`: a band edited at work kept each cell's action by
  index, so after moving its frequency, Q or cell count the reduction sat
  at other frequencies. The actions are now carried to the new cells by
  frequency (nothing outside the span the band had), and a span that moved
  takes its level reference from the new span after half a second. A cell
  also looks half a cell into its neighbours with a fading weight, so a
  partial that crosses a border no longer changes hands (a tone with a slow
  vibrato across the borders of a bell of 8 cells: largest step of the
  reduction between 10 ms windows 1.56 dB before, 0.16 dB now, and the mean
  reduction 11.1 dB where it was 9.7). The band's shape comes from its
  analog prototype, so a band near the top of the audio band keeps its
  width.
- `Equalizer` with `BandType::ConstantTilt`: with the pivot at one end of
  the band the other end rose without limit (a pivot at 20 Hz and +6 dB per
  octave lifted 20 kHz by 60 dB). The end that rises is now held to 36 dB:
  the slope comes down just enough (`FilterDesign::tiltSlopeWithin()`); a
  pivot in the middle of the band keeps every slope, and the end that falls
  is not limited.
- `AlgorithmicReverb::setTankLeveling()`: its level rose over 20 ms and fell
  over 4 ms, so the reduction let go within each cycle of a low note (a
  50 Hz tone at -10 dBFS came out with 52 % of harmonics, 100 Hz with 6 %,
  200 Hz with 1.8 %) and the first 20 ms of a hit went through unreduced (a
  20 ms hit and one 20 dB under it left tails 10.3 dB apart where the curve
  gives 6.9). It now reads the peak of each sample; the reduction deepens at
  once, holds 30 ms and comes back with a time constant of half a second:
  0.0002 % on the 50 Hz tone, and the two hits 6.9 dB apart.
- `AlgorithmicReverb`: the wet signal's level followed the mid decay alone,
  so it moved with the damping, the bass multiplier and the decay bands (a
  Hall from 0.3 to 12 s of decay across those settings: 5 dB), it rose as
  the decay got short (the early field has its own energy) and a plate whose
  decay is short against its round trip read up to 9 dB low. It now takes
  back the energy of the tail as heard (the passes round the tank from the
  loop gains of every band, the first pass through each line exactly) plus
  that of the early field: on pink noise the six types sit within 0.7 dB of
  the dry level at their presets, a Hall within 0.5 dB at any decay, size,
  damping, bass multiplier or decay band, a plate within 0.6 dB and a
  spring within 1.2 dB from 0.3 to 12 s at any size.
- `AlgorithmicReverb::setSize()` (and a change of layout) slid the line
  lengths: a glide of pitch of up to 4 % for most of a second (a steady
  1 kHz tone lost most of its energy from the tone during the change). A
  change beyond 0.2 % of a line's length now opens a second read head at
  the new lengths (lines and in-loop allpasses, springs included) and
  cross-fades to it at constant power over a quarter of a second: the
  energy more than 20 Hz from the tone stays 47 dB under it, without
  clicks. The loop gains now ramp sample by sample toward a new design (they
  moved in steps of a chunk, which a resonating tail made audible).
- `AlgorithmicReverb` with the Segments modulation: the straight segments
  left a step of pitch at each corner. The lines now follow them through a
  critically damped second-order filter and the segments alternate sides:
  with a 3 kHz tone the energy more than 30 Hz from it goes from -50 to
  -69 dB.
- `Chorus` analog model: the right tap was the asymmetric triangle half a
  cycle on, not the left one inverted, so the two delays did not add up to
  a constant (in the Slow mode the sum swung by 15 to 17 samples at 44.1
  and 48 kHz, and twice a cycle both taps rose together); one phase now
  carries the asymmetry in its speed and both taps read a symmetric triangle
  of it half a turn apart (the sum holds within half a sample). The centre
  and the depth were read once a block and the mode's offset switched at
  once: automating the centre stepped the clock at the block rate (lines at
  the tone +- fs / 512, -39 dBc when swept over a second in blocks of 512;
  now -70) and a change of mode jumped (19 times the steady sweep's largest
  second difference; now 2.3): they now ramp across the block and glide
  (50 ms), and the filters' corner moves at most an octave in a quarter of a
  second. At 176.4 and 192 kHz the half-rate engine's output alternated
  between two delays a sample apart (the image of a 5 kHz tone at 91 kHz was
  -16 dBc; now -43, the linear interpolation's). And the lines' clock,
  slower than the host's rate in every mode, folded its images into the
  band: the lines now run at twice the rate below 88.2 kHz, with their high
  precision, so at the longest delay of the Slow mode and no depth a 1, 3 or
  5 kHz tone leaves no other line within 91 dB of it at 44.1, 48 and 96 kHz
  (before: 35 to 81 dB). The two lines share their input low-pass. Cost: 91
  ns per stereo frame at 48 kHz (46 before) and 57 at 96 kHz (40).
- `Compressor`: an envelope within 1e-4 dB of zero was put back on zero
  whatever its target, to end the tail of a release. Leaving zero in steps
  shorter than that, each step was undone and the gain never moved: a slow
  attack ignored a shallow target for good (200 ms of attack and 0.3 dB of
  reduction asked for: 0 dB, at 48 and at 96 kHz; 100 ms at 96 kHz ignored
  anything under 1 dB), and the memory stage of the Opto character, which
  charges over two seconds with a release of 500 ms, never charged under
  9.6 dB of reduction at 48 kHz, so on light compression the gain settled
  short of the curve and the long tail was missing (ten seconds 6 dB over a
  2:1 threshold now read 3.0 dB of reduction and, one second into the pause
  that follows, 1.05 dB: the memory). The envelope is now put on zero only when
  its target is there too. Deeper or faster settings are bit for bit as they
  were.
- `Synthesizer` with oversampling, built with one recent compiler: every event
  landed early, by 32 samples at 2x and 17 at 4x, and where it landed depended
  on where the block began. The two lines that place the content against the
  decimation delay (the delay rounded up to whole samples, and what the
  rounding added) were evaluated wrong by that compiler's optimizer (MSVC
  19.52.37029 for x64, at /O1 and /O2; the same two lines alone in a file give
  -63 where 0 is right); they are now written with the remainder, which it
  evaluates right. Builds with every other compiler tested, and with that one
  without optimization, were not affected.
- Offline transient analysis, the offline sinc stages, the offline clippers
  and StereoGenerator charged their FFTs from a fixed estimate (fewer than 4N
  scalars for a real transform of size N), which the faster transforms of
  this release outgrew (up to about 4.25N): StereoGenerator at 8 kHz with 16x
  oversampling prepared 4.4 KB more than `getPrepareMemoryBound()` reported,
  and a job given exactly its reported `memoryBytes` could hit its limit.
  `FFTReal::allocationBytes()` and `FFTComplex::allocationBytes()` now report
  an upper bound of what a constructor requests (including standard-library
  bookkeeping), every budget charges that, and the pass plan is one exact
  allocation.

- AU: MIDI from several threads at once (a host's MIDI thread, its render
  thread, an on-screen keyboard) is no longer lost: the MIDI queue takes any
  number of producers (one compare-exchange per event). The single-producer
  ring lost 132 of 32000 events pushed by two threads and 176 to 890 of
  64000 pushed by four (and duplicated one); none now. A push costs 5.5 ns
  instead of 2.5 ns.
- CLAP: every voice the host starts receives exactly one NOTE_END. Past the
  128 voices of the table a new voice retires the oldest (a released one
  first) with its NOTE_END; before, voices past 128 were not tracked and
  never ended for the host (128 NOTE_ENDs for 200 voices), which kept their
  modulators running.
- AU host contract: every parameter is flagged high-resolution (a host
  that limits control resolution now gives it the highest it has); a NaN or
  infinite value from `AudioUnitSetParameter` or `AudioUnitScheduleParameters`
  is refused with `kAudioUnitErr_InvalidParameterValue` instead of moving the
  parameter to an end of its range; a render of another output element is
  refused (it ran the DSP twice per cycle) and so is a host buffer shorter
  than the slice (512 samples were written into a 256-sample buffer); an
  immediate scheduled value reads back at once; the scheduled events of a
  slice that is not rendered no longer replay in the next one (a refused
  slice's ramp applied 8 stale points); `Reset` drops the MIDI queued before
  it; the saved state's `name` is the preset's (it was the plugin's), is
  restored with the state and announced to PresentPreset listeners; a preset
  of another unit (subtype or manufacturer) is refused instead of feeding
  this plugin its user section; a bypass moved through the bypass parameter
  (automation, a state load) notifies the `BypassEffect` listeners, and one
  set through the property notifies the parameter's listeners, so the host's
  button and its automation lane agree (neither was notified before).
- CLAP: a NaN `CLAP_EVENT_PARAM_VALUE` is ignored; it moved the parameter to
  its minimum.

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
