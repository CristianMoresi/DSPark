# DSPark Cookbook

Working recipes for common production tasks. Every snippet assumes:

```cpp
#include "DSPark.h"
using namespace dspark;

AudioSpec spec { 48000.0, 512, 2 };   // rate, max block, channels
```

All processors follow the same lifecycle: construct → `prepare(spec)` once
(allocates) → `processBlock(view)` in the callback (allocation-free,
lock-free parameter setters from any thread).

---

## 1. Sidechain ducking (music under a voice-over)

```cpp
Compressor<float> duck;
duck.prepare(spec);
duck.setThreshold(-30.0f);
duck.setRatio(6.0f);
duck.setAttack(5.0f);
duck.setRelease(250.0f);

// callback: music in `bus`, narration in `voice`
duck.processBlockWithSidechain(bus, voice);   // music ducks under the voice
```

## 2. De-esser for vocals

```cpp
DeEsser<float> deEsser;
deEsser.prepare(spec);
deEsser.setFrequency(6800.0f);   // sibilance center
deEsser.setThreshold(-28.0f);

deEsser.processBlock(vocals);
```

## 3. Pitch-tracking low-cut (clean lows that never thin the voice)

```cpp
PitchFollower<float> follower;
follower.prepare(spec);
follower.setRange(70.0f, 800.0f);
follower.setGlide(60.0f);                       // ms per octave

FilterEngine<float> lowCut;
lowCut.prepare(spec);
lowCut.setHighPass(70.0f, 0.707f, 12);

// callback:
follower.processBlock(buffer);                  // internal mono sum
if (follower.getSmoothedHz() > 0.0f)
    lowCut.setFrequency(follower.getSmoothedHz() * 0.9f);
lowCut.processBlock(buffer);
```

Generalize the same pattern to any parameter with `ModulationRouter`
(Core/ModulationRouter.h).

## 4. Mastering chain

```cpp
Equalizer<float> eq;            eq.prepare(spec);
eq.setMatchedBells(true);       // Orfanidis de-cramped bells near Nyquist
eq.setBand(0, 90.0f, -1.5f);
eq.setBand(1, 12000.0f, 1.0f);

MultibandCompressor<float> mb;  mb.prepare(spec);
mb.setNumBands(3);

Limiter<float> limiter;         limiter.prepare(spec);
limiter.setCeiling(-1.0f);      // dBTP for streaming delivery
limiter.setTruePeak(true);

LoudnessMeter<float> meter;     meter.prepare(spec.sampleRate, 2);

// callback:
eq.processBlock(buffer);
mb.processBlock(buffer);
limiter.processBlock(buffer);
meter.processBlock(buffer);
// meter.getIntegratedLUFS() / getTruePeakDb() -> GUI / delivery check
```

The LoudnessMeter passes the official EBU R128 vectors (Tech 3341/3342:
integrated, LRA and true peak) — see `conformance/`.

## 5. Analog console color (tape + transformer + tube)

```cpp
TubePreamp<float> pre;          pre.prepare(spec);
pre.setStages(1);               // single triode: even-harmonic warmth
pre.setDrive(6.0f);
pre.setTreble(0.6f); pre.setBass(0.5f); pre.setMiddle(0.5f);

TransformerModel<float> iron;   iron.prepare(spec);
iron.setDrive(3.0f);            // low-end bloom, LF-weighted harmonics

TapeMachine<float> tape;        tape.prepare(spec);
tape.setSpeed(TapeMachine<float>::Speed::IPS_15);
tape.setDrive(4.0f);
tape.setWowFlutter(0.1f);

// callback:
pre.processBlock(buffer);
iron.processBlock(buffer);
tape.processBlock(buffer);
```

All three are physical models (Koren triode + WDF FMV tone stack;
flux-domain and tape-calibrated Jiles-Atherton hysteresis), loudness-
compensated: drive moves saturation, not volume.

## 6. Synth voice (sync lead with granular air)

```cpp
Oscillator<float> osc;
osc.prepare(spec.sampleRate);
osc.setWaveform(Oscillator<float>::Waveform::Saw);
osc.setFrequency(110.0f);
osc.setSyncRatio(2.7f);          // band-limited hard sync

EnvelopeGenerator<float> env;    env.prepare(spec);
LadderFilter<float> ladder;      ladder.prepare(spec);
ladder.setCutoff(1200.0f);
ladder.setResonance(0.4f);

GranularProcessor<float> cloud;  cloud.prepare(spec);
cloud.setMix(0.25f);
cloud.setSpread(0.8f);
```

## 7. Restoration (denoise a location recording)

```cpp
SpectralDenoiser<float> dn;
dn.prepare(spec);
dn.setReduction(18.0f);

dn.setLearning(true);    // feed ~1 s of room tone / hiss only
// ... process the noise-only region ...
dn.setLearning(false);   // now process the programme
```

## 8. Zero-latency cabinet IR (monitoring path)

```cpp
ZeroLatencyConvolver<float> cab;   // Gardner partitioning: latency 0,
cab.prepare(ir.data(), irLength);  // flat CPU even for second-long IRs

// callback (any block size, even 1):
cab.processInPlace(buffer.getChannel(0), buffer.getNumSamples());
```

## 9. Stereo health on the master

```cpp
PhaseCorrelation<float> corr;
corr.prepare(spec);

ChordDetector<float> chords;       // optional: key/chord display
chords.prepare(spec);

// callback:
corr.processBlock(buffer);
chords.processBlock(buffer);
// corr.getCorrelation() in [-1, +1]; corr.getGonioPoints(...) -> vectorscope
```

## 10. Oversampling a nonlinear section

Oversampling belongs to the product, not to each module: wrap the whole
nonlinear section once instead of paying one resampler (latency + CPU +
band-limiting) per effect.

```cpp
Oversampling<float> os(4);         // factor 1, 2, 4, 8 or 16
os.prepare(spec);

// Stages inside the section are prepared at the oversampled rate. Time
// constants are in milliseconds, so their behaviour does not change.
AudioSpec spec4x { spec.sampleRate * 4, spec.maxBlockSize * 4, spec.numChannels };
myShaper.prepare(spec4x);          // e.g. a hot custom waveshaper
comp.prepare(spec4x);

// callback:
auto up = os.upsample(buffer);     // view at fs * factor
myShaper.processBlock(up);
comp.processBlock(up);
os.downsample(buffer);             // back to host rate, band-limited

// Report os.getLatency() (plus any in-section stage latency, scaled back
// by the factor) as plugin latency.
```

Measure before reaching for this: DSPark's own nonlinear stages
(Saturation, TapeMachine, Clipper) already oversample internally where the
algorithm needs it, and the Compressor's gain path stays at or below
-72 dBc of aliasing at 1x even in its worst case (FET character at minimum
attack). A section like the one above earns its resampler when you drive
custom waveshaping hard, not for dynamics alone.

Transparency: every processor that oversamples internally
exposes the factor via `setOversampling(int)` with **factor = 1 meaning OFF**
(no internal resampling, zero added latency), supports at least {1, 2, 4}, and
reports the added latency through `getLatency()` so the host can compensate.
`DynamicEQ` embeds such a stage and **defaults to 1x (off)**: turn it up to 2x
or 4x only when a band's detector needs the extra alias suppression, and leave
it at 1x when you already oversample the whole section (above) to avoid
cascaded resamplers. Cost scales roughly linearly with the factor; the exact
per-factor latency is whatever `getLatency()` returns for the active setting.

The two nonlinear-modelling stages that generate aliasing *inside* the model
default to internal oversampling and expose the same control (`setOversampling`
is setup-thread only on both - it reallocates and re-calibrates like
`prepare()`):

- `TapeMachine` **defaults to 4x**. The AC-bias carrier sits at 0.375x the
  internal rate, so 4x/48k puts it ultrasonic (72 kHz) with its even folds
  killed by the downsampler. CPU scales ~linearly with the factor (4x is the
  reference cost of physical AC bias). `setOversampling(1)` turns the resampler
  off (zero added latency) but then drops the carrier in-band (18 kHz at 48k) -
  use 1x only under a high host rate or an already-oversampled section.
- `TubePreamp` **defaults to 2x**. The triode + WDF tone solve runs `factor`x
  oversampled; 4x roughly doubles the 2x CPU, 1x is the cheapest and adds zero
  latency but lets the grid nonlinearity alias in-band. `getLatency()` reports
  0 at 1x for TubePreamp (its only latency source is the oversampler). For
  TapeMachine `getLatency()` at 1x is NOT zero: it still reports the loss-FIR
  centre + transport delay (127 at 48k); only the oversampler contribution
  drops to 0. Always query `getLatency()` for the active factor rather than
  assuming a value.

`Saturation`, `Clipper` and `Core/WaveshapeTable` also expose `setOversampling`
(all defaulting to 1x=off); `WaveshapeTable` is a memoryless table (no ADAA), so
its Hermite interpolation reduces table noise but not aliasing - oversample it
to suppress alias products.
The `Compressor` does NOT run an internal audio-path resampler: its optional
TruePeak mode oversamples only the *detector* (ITU-R BS.1770 inter-sample peak
measurement), which adds no audio-path latency and cannot cascade, so there is
no factor to configure there.

## 11. Hardware compressor recipes

The Compressor's characters are calibrated against the published hardware
figures, so classic units map to plain settings. Feedback operation lands
on the requested static curve exactly (the element's law is the closed-form
inverse of the user's curve, matching how hardware panels are marked with
observed ratios), and the loop is resolved semi-implicitly with the peak
detector, so it stays stable down to 20 us attacks.

**LA-2A style leveler** (Teletronix spec: 10 ms attack, ~50% release in
0.06 s, complete release 0.5 to 5 s depending on programme, ~3:1, gentle
knee):

```cpp
Compressor<float> comp;
comp.setCharacter(Compressor<float>::Character::Opto);
comp.setTopology(Compressor<float>::Topology::FeedBack);
comp.setRatio(3.0f);
comp.setAttack(10.0f);    // the Opto floor; lower requests clamp here
comp.setRelease(60.0f);   // the spec's "0.06 s to 50%"
comp.setKnee(0.0f);       // the photocell's 10 dB floor takes over
comp.setThreshold(-30.0f);// drive to taste (the hardware knob is "peak reduction")
```

Measured on this recipe: 50% release 64 ms after a long squeeze, complete
release ~2.1 s (and faster after brief peaks: the memory stage only charges
under sustained compression), knee floor engaging right at the threshold.

**1176 style FET limiter** (UREI/UA spec: 20-800 us attack, 50-1100 ms
release, panel ratios 4/8 compress and 12/20 limit, THD < 0.5% while
limiting):

```cpp
Compressor<float> comp;
comp.setCharacter(Compressor<float>::Character::FET); // forces feedback + peak detection
comp.setRatio(20.0f);        // panel ratios: 4, 8, 12, 20
comp.setAttack(0.02f);       // hardware knob range 0.02-0.8 ms (fastest = 7)
comp.setRelease(50.0f);      // hardware knob range 50-1100 ms
comp.setCharacterColor(1.0f);// the FET's 2nd-order signature, <0.5% THD calibrated
```

Measured: settled gain reduction lands on the panel curve (19 dB at 20:1
with the level 20 dB over threshold), observed attack t63 21 us at the
fastest setting with the loop stable, colour THD 0.42% at -6 dBFS programme
while limiting (0% with colour off). Driving low frequencies at the fastest
attack/release rides the waveform within the cycle exactly like the
hardware (several percent THD at 100 Hz: that is the 1176 grit, back off
attack or release to clean it up). All-buttons mode is not modeled.

**Fairchild 670 style vari-mu** (0.2-0.8 ms attack; release 0.3 to 25 s
across the six Time Constant positions, the slowest ones programme
dependent):

```cpp
Compressor<float> comp;
comp.setCharacter(Compressor<float>::Character::Varimu);
comp.setTopology(Compressor<float>::Topology::FeedBack);
comp.setRatio(2.0f);      // the effective ratio grows with level on its own
comp.setAttack(0.4f);
comp.setRelease(300.0f);  // TC 1; up to 25000 for the slow positions
comp.setKnee(0.0f);       // the remote-cutoff tube's 10 dB floor takes over
```

Notes that apply to all three: with a memory detector (RMS) in FeedBack
keep the attack at or above the detector window, or the loop hunts on the
window's lag (a 0.1 ms attack against a 10 ms RMS window pumps ~3 dB at
20:1; the hardware units above are all peak detected, which resolves
implicitly and does not hunt). Release knobs are t63 measured after the
signal drops, so the loop does not alter them.

## 12. Change tempo by +/-10 BPM (without moving the pitch)

`TimeStretch` changes how long a passage takes without touching what note it
is on. Tempo is a ratio, so a passage recorded at 120 BPM played back at
110 BPM lasts `120 / 110` times as long:

```cpp
dspark::TimeStretch<float> ts;
ts.prepare(spec);                       // 2048-sample frame by default
ts.setTimeRatio(120.0f / 110.0f);       // 120 BPM passage, played at 110

dspark::AudioBuffer<float> slower;
ts.process(loop.toView(), slower);      // slower.getNumSamples() == round(n * 120/110)
```

`setTempoChangePercent()` is the same control from the other end, in the
units a tempo knob is marked in: `-8.33f` is 120 BPM played at 110,
`+8.33f` is 120 played at 130. Either way the realised ratio is exact and
does not drift: measured over 30 s at 48 kHz the stretch lands within
0.003% of the target, and the output is exactly `round(inputLength * ratio)`
samples long, so a stretched loop still meets its own end.

**Drums and other strikes need no setting.** Attacks are found with spectral
flux over a log-frequency filterbank and the analysis hop is held across each
one, which is what keeps a strike from doubling when the stretch spreads it.
Measured on a click train, the stretched strike is as concentrated as the
input's own at every frame size from 256 to 4096 and at every ratio, and no
onset lands more than 1.71 ms from where the stretched timeline puts it - at
120, 480 and 960 strikes per minute alike. It is on by default;
`setTransientPreserve(false)` turns the whole transient path off, which is
useful for hearing what it does and is not otherwise a better setting.

The one limit worth knowing: the hold needs a gap between strikes to repay
the input it did not consume, and the density it can serve scales as
`sampleRate / fftSize`. At the default 2048-sample frame and 48 kHz that
holds to about 1040 strikes per minute; past roughly 1060 the timing
degrades to about 3.9 ms. A shorter frame moves the limit up in proportion.

**Streaming: use the rate-changing pair away from ratio 1.** A stretch
changes duration, so the honest streaming shape is one where the input count
and the output count are allowed to differ. That is `feedInput()` /
`pullOutput()`:

```cpp
ts.setTimeRatio(120.0f / 110.0f);
// per callback, in either order and in any sizes:
const int took = ts.feedInput(block.toView());        // <= getInputCapacity()
const int gave = ts.pullOutput(destination.toView()); // <= getAvailableOutput()
```

There is no latency to compensate on that path: what comes out is the
stretched timeline itself, output sample `k` being sample `k` of the
stretched signal. It primes first - `pullOutput()` returns 0 until about one
frame of input has been fed - and after that the output tracks
`ratio *` the input it has consumed, less a fixed offset of one frame minus
one analysis hop that is the overlap-add's own incomplete tail. Feeding
without pulling makes `getInputCapacity()` fall to 0, which is the back
pressure that keeps the two ends honest.

**The block path is a fixed-rate playback adaptor.** `processBlock()` hands
back exactly as many samples as it was given, latency `getLatency()` (one
frame: 2048 samples, 42.7 ms at 48 kHz). At ratio 1 it is exact
indefinitely. Away from unity it cannot be, and the cost is stated in
numbers rather than described: below unity exactly `1 - ratio` of the output
is silence (measured 20.00% at 0.8, 7.40% at 0.926, 5.00% at 0.95, worst
error 0.03 percentage points over durations of 5 to 30 s and blocks of 64 to
4096), delivered as gaps of at most one block; above unity the block
physically cannot carry the stretched stream, so the adaptor refuses the
fraction `1 - 1/ratio` of the input at the head, spread evenly, and counts
every refused sample in `getDiscardedInput()`. Nothing that survives is
displaced - measured worst 38 samples, 0.79 ms, at ratio 1.081 - which is
the whole point of refusing at the head rather than letting a queue fill and
splice the stream. What it cannot preserve is a strike's height: on the same
strike train the mean strike height, as a fraction of the same class's own
ratio-1 rendering, is 0.85 at ratio 1.01, 0.79 at 1.02, 0.38 at 1.05, 0.82 at
1.081, 0.89 at 1.25, 0.56 at 1.5 and 0.11 at 2 - not monotonic in the ratio,
so those figures cannot be interpolated between. Use the pair above, or
`process()`, unless the ratio is 1.

The two streaming paths own the same queue with different invariants, so
whichever one is used first after `prepare()` or `reset()` owns the instance
until the next `reset()`; the other one's calls do nothing in the meantime.

At ratio 1 the streaming path is transparent: measured residual against the
delayed input is -146 dBFS at 48 kHz on a 1 kHz tone, so leaving the effect
in a chain unengaged costs nothing but its latency.

---

## 13. Find the tempo and the beats

`BeatTracker` answers two different questions with two different methods, and
which one you want depends on whether you have the whole file.

**Offline, when you do.** `analyze()` sees the future, so it can place beats
using evidence that arrives after them:

```cpp
dspark::BeatTracker<float> bt;
bt.prepare(spec);                          // mono: channel 0 is read

auto beat = bt.analyze(loop.toView());
// beat.tempoBpm          fitted to the whole grid, not read off one lag
// beat.beatSamples       every beat position, in samples, ascending
// beat.confidence        [0,1] -- how much of the signal the grid explains
// beat.secondaryTempoBpm the metrical level that came second, or 0 if none
```

On a click track from 40 to 240 BPM the tempo lands within 0.001 BPM of the
truth at the correct metrical level across the whole range, and every beat is
found within 5.4 ms at worst. On a 100-to-140 BPM ramp the grid follows the
tempo being played rather than the average: worst local inter-beat error
0.18%.

**Read the confidence before you trust the grid.** It is the share of the
onset strength that falls in phase with the beats returned, so 1.0 means the
grid explains everything the signal did. It is not a probability that the
tempo is right, and it is deliberately reduced when a second metrical reading
explains the signal nearly as well: on a three-against-two polyrhythm, where
two pulses are genuinely present, it reports around 0.2 rather than pretending
one of them is the answer. Below about 0.5, look at `secondaryTempoBpm` before
acting.

**Half and double tempo are the error that matters,** so check the level, not
just the number. `secondaryTempoBpm` names the alternative a listener could
plausibly have tapped instead -- usually the half-tempo reading -- and is 0
when there is no distinct alternative inside the searched range. Narrow the
range if you know the material:

```cpp
bt.setTempoRange(70.0f, 140.0f);   // no allocation; safe while audio runs
```

**In the callback, when you do not have the future.** Feed blocks and read the
running estimate:

```cpp
void processBlock(dspark::AudioBufferView<float> io)
{
    bt.processBlock(io);                       // allocation-free, lock-free

    float bpm = 0.0f, confidence = 0.0f;
    bt.getTempoAndConfidence(bpm, confidence); // one load: they belong together

    if (bt.beatNow())
        scheduleClick(bt.getLastBeatSample()); // where the beat WAS
}
```

The running tempo reaches within 5% of the truth inside 2.6 s at worst on a
click track and holds to 0.75% after that, and beats are attributed to within
21 ms. `getLastBeatSample()` is where the beat happened in your own timeline;
`getLatencySamples()` is how far in the past that is by the time you are told
(549 samples, 12.4 ms, at 44.1 kHz). Use the former to align anything and the
latter only to state your own delay.

One behaviour to know: with no onset energy arriving, the pulse estimate and
the mass it is measured against decay together, so the confidence HOLDS its
last value through silence instead of falling. A caller that needs to tell
"steady" from "nothing playing" reads level separately.

`BeatTracker` consumes `OnsetDetector`'s onset-strength envelope, so if you
want onsets too, take them from `getOnsetDetector()` rather than running a
second copy of the same analysis.
