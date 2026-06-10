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
