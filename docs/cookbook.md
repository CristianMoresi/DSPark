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
