# Offline dynamics and analysis

`OfflineLeveler` automatically raises quieter sustained passages toward the
strongest sustained input RMS level. One `amount` control selects how much of the
measured difference to correct. It operates on a complete mono or stereo source;
it is not an audio-callback processor and does not satisfy `AudioProcessor`.

These worker APIs require C++ exceptions so allocation and host callback failures
can return explicit statuses. `DSPark.h` defines `DSPARK_HAS_OFFLINE` to 1 when
they are available, or 0 with exceptions disabled or `DSPARK_NO_OFFLINE` defined.
Direct offline headers follow the same rule. Real-time classes remain available
in the existing no-exceptions/no-RTTI embedded profile. Emscripten users must
enable exception handling for offline jobs, including catching allocation errors.

```cpp
#include "DSPark.h"

bool levelAudio(const dspark::AudioBuffer<float>& input,
                dspark::AudioBuffer<float>& output, double sampleRate)
{
    dspark::OfflineLeveler<float> leveler;
    dspark::OfflineLeveler<float>::Options options;
    options.amount = 0.5;
    const auto result = leveler.run(input, output, sampleRate, options);
    return result.succeeded();
}
```

`run` constructs a provisional output and replaces `output` only on success.
Input and output may be the same owning buffer. Invalid audio, cancellation and
allocation failure leave the original output intact. Zero amount is bit-exact,
including signed zero; non-finite input is rejected even with zero amount.

## Signal behavior

- The gain plan is stereo-linked and upward only. There is no limiter, hidden global trim,
  automatic makeup gain or change to channel count, length or sample alignment.
- The reference is the strongest approximately 3 s input RMS window on a 100 ms
  grid. Short sources use their full length. The final short bin uses its actual
  duration. This is RMS leveling, not integrated LUFS normalization; use
  `LoudnessNormalizer` for the latter.
- `amount` is in [0,1]. `speed`, normally left at 0.5, varies macro context from
  6 to 2 seconds and transition width from 1.2 to 0.2 seconds. It is not compressor
  attack or release. `maximumBoostDb` is in [0,24], with default 24 dB.
- An activity threshold and soft transition suppress correction of very quiet
  tails. This is a level heuristic, not recognition of musical intent or noise.
- Each analysis-grid window in the scalar gain plan is conservatively bounded by
  the strongest input window using the maximum gain within each bin. Bandlimited
  interpolation changes individual samples and can change those window energies.
  This planning bound does not impose a ceiling on the interpolated render.
- Output may exceed 0 dBFS. `outputSamplePeakDb` and `outputTruePeakDb` report the
  resulting headroom, using the shared `TruePeakDetector` and its final FIR tail.
  Silent peak readings are negative infinity. No corrective output limiter is
  inserted by the class.

`Report` includes the reference RMS, planned gain range, control-point count,
detected persistent boundaries and correction-cap diagnostics. `peaksMeasured`
distinguishes render measurements from a plan alone. `NoChange` is a successful
identity render, with a reason such as zero amount, no measurable activity or all
regions excluded. A successful status does not imply any specific loudness change.
For extremely large finite PCM, gain is reduced where multiplication would exceed
the sample type's representable range. `RepresentabilityLimited` and
`representabilityLimitedBins` report that lost correction explicitly; this does
not impose a 0 dBFS ceiling. Bandlimited rendering can additionally cap its
processing delta for floating-point representability; `renderInfo` identifies it.

## Gain interpolation and render measurements

The shared gain renderer uses complete-source bandlimited multiplication. It
reconstructs finite PCM by sinc interpolation, multiplies the reconstructed audio
and control, and projects the product into the source baseband. Control continues
independently with the first gain before the source and the last gain after it.
The two infinite tails are evaluated analytically, without a padding buffer.
Constant-gain processing remains an exact scalar operation. Source length
and sample alignment are retained. This does not enable audio oversampling.

Changing gain creates interpolation tails, including before a transient. A
planned gain of one at a frame does not imply identical output at that frame;
an upward control does not imply that every output sample increases in magnitude.
Zero amount, constant unity and explicitly protected PCM remain bit-exact.

Every processor report includes `OfflineGainRenderInfo`. `bandlimited` indicates
that the interpolation product was evaluated. `boundaryGain` states the affine
reference; `leftBoundaryGain` and `rightBoundaryGain` state the two exterior
controls before `deltaScale`. `boundaryCalibrated` reports an exterior-control
adjustment. `deltaScale` calibrates `output = input + deltaScale * processingDelta`;
it is not a master gain applied to the source. `representabilityLimited` reports
a cap needed to keep floating-point PCM finite. `minimumControlGain` and
`maximumControlGain` give the calibrated control range, which can differ from
the initial plan. They do not describe the pointwise output/input sample ratio.
`targetFeasible = false` with
`NumericalFailure` rejects an unattainable target before beginning a sink output.
No unmeasured candidate is silently published.

## Reusing complete-file analysis

`OfflineEnergyAnalyzer` produces an immutable, move-only 100 ms energy map. The
map stores linked RMS and sample peak per bin, source format/revision and a PCM
fingerprint, without retaining PCM. Gain plans can be rebuilt from that map when
amount, speed or exclusions change:

```cpp
#include "DSPark.h"

auto prepareLeveler(dspark::AudioBuffer<float>& input, double sampleRate)
{
    dspark::OfflineBufferSource<float> source(input.toView(), sampleRate, 17, 1);
    auto measured = dspark::OfflineEnergyAnalyzer<float>().analyze(source);
    dspark::OfflineLeveler<float> leveler;
    if (!measured.succeeded())
    {
        dspark::OfflineLeveler<float>::PlanResult failed;
        failed.status = measured.status;
        return failed;
    }
    dspark::OfflineLeveler<float>::Options options;
    options.amount = 0.75;
    return leveler.makePlan(measured.analysis, options);
}
```

The last two adapter arguments are host source identity and revision. A plan is
valid only for the exact stage input analyzed: change revision and reanalyze after
editing audio, changing its interpretation, resampling or modifying an upstream
effect. A noncryptographic, block-independent PCM fingerprint also detects
accidental stale reuse, even when a revision was not updated; it is not a security
identity or a substitute for immutable source ownership during a job.

The plan exposes read-only gain lookup at 64-bit source-relative frame positions.
Outside its source extent, lookup returns unity. Source `timelineOrigin` is
provenance metadata; it does not shift the offsets passed to `read` or `gainAt`.
Spans into a map or plan remain valid only while that object is alive and unmoved.

## Exclusions

Set `Options::exclusions` to source-relative `[begin,end)` intervals. Planning
copies, sorts and merges touching/overlapping intervals. Reversed, negative or
out-of-source intervals are rejected. Source edits require transforming intervals
in the host and building a new plan.

Gain is exactly one throughout protected intervals; feathers lie outside them.
The masks have three zero edge derivatives. When neighboring feathers overlap,
their masks multiply, retaining a smooth join instead of a cusp at the midpoint.
Exclusions do not alter the input RMS reference. Subsequent filters or resamplers
in an application can spread neighboring changes across an exclusion boundary;
their support must be accounted for separately. Within the shared renderer, the
additional interpolation correction is feathered outside the protected region
and is exactly zero inside it. This constrained result differs from unconstrained
bandlimited multiplication; exact PCM protection is not a claim of identical
spectral behavior at an exclusion boundary.

## Long sources, resource bounds and cancellation

Implement `OfflineAudioSource<T>` for a decoder or other rewindable source. Reads
use 64-bit offsets and bounded `AudioBufferView` blocks. No source sample is
silently substituted for an invalid or failed read. Mono and stereo are supported;
other layouts are rejected explicitly. Supported PCM types are float and double.

Implement `OfflineAudioSink<T>` to keep output provisional until `commit` succeeds.
`abort` must discard all provisional writes and must not throw, including after a
failed `begin` or `commit`. `render(source,plan,sink)` verifies the source before
starting the sink, then writes in bounded blocks and checks the source again
before commit. Sample/true-peak measurement shares the provisional write pass;
the interpolator tail and cancellation checks finish before publication. A failure
after `begin` aborts all provisional PCM. No filesystem format, implicit thread
pool or I/O library is added.

`OfflineJobOptions` controls block size, a payload-memory budget, a polled atomic
cancel flag and a synchronous progress callback. Returning false from progress
cancels the job. Progress phases are analysis, planning, verification and render;
their units are source frames, except planning which reports control points.
Complete-source transient analysis uses two analysis passes; a phase may recur
and progress restarts for each pass.
Callbacks and options remain alive and unmodified until the operation returns.

The budget covers newly requested dynamic payload, including a returned map or
plan. It excludes caller-owned input/existing plans/output, allocator metadata,
stack state and any allocations performed by host source/sink callbacks. The
owning convenience call additionally budgets its provisional output and retained
intermediate state. `memoryBytes` records requested payload, not process RSS.
The transient analyzer also charges a conservative setup allowance for its shared
FFT workspace, checked against ordinary allocation requests in the test suite.
Use a transactional streaming sink when a full output buffer is undesirable.

The standalone energy scan is O(source frames * channels). Maps and gain plans
grow with duration/100 ms, not with sample rate. The plan uses bounded-duration
neighborhoods and an order-statistic selection. Streaming render keeps bounded
PCM/control caches and compact interpolation maps, with O(source frames / B)
storage for internal block size B (128..4096). It creates no full-file PCM or gain
array. Source verification, map construction, calibration, measurement and output
make repeated reads; every cache reload checks its saved PCM fingerprint.
`blockFrames` bounds host read/write requests independently of the internal map
partition. Repeated reads may have decoding/I/O cost in the host.

## Attack and pulse maps

`OfflineTransientAnalyzer<T>` analyzes immutable mono/stereo sources at 8-384 kHz.
It first reuses `OfflineEnergyAnalyzer` to measure the source peak and fingerprint.
A second bounded scan normalizes analysis samples by that peak, pools spectral
channel power, and computes SuperFlux attacks and spectral-flux pulses from one
FFT per channel. Processing audio is never normalized or modified. Reversing one
channel's polarity cannot cancel events, and swapping channels leaves maps equal.

```cpp
#include "DSPark.h"

auto analyzeTransients(const dspark::AudioBuffer<float>& input, double rate)
{
    dspark::OfflineBufferSource<float> source(input.toView(), rate, 42, 1);
    dspark::OfflineTransientAnalyzer<float>::Options options;
    options.attacks = true;
    options.pulses = true;
    return dspark::OfflineTransientAnalyzer<float>().analyze(source, options);
}
```

The immutable result provides separate `attacks()` and `pulses()` spans. Each event
contains source-relative `begin`, `peak`, exclusive `end`, normalized novelty
strength, local power contrast, pooled peak amplitude and a sustain RMS reference.
Contrast is a signal measurement, not a calibrated probability. Events do not
constitute a beat grid or a tempo estimate. `energy()` provides the reusable macro
map, source specification and fingerprint.

Timing refinement uses approximately 1 ms linked power bins, positive local energy
growth and the source's low-band energy share. Bass attacks use longer context to
separate a new note from the cycles of a sustained waveform. Both ends of a source
receive full padded spectral support. Tests pin isolated first/final-frame impulses,
eight annotated 9-100 ms bursts at six rates, dense attacks and soft bass over a
sustained bed. The burst fixtures require starts within 2 ms and ends within 4 ms;
the soft bass fixture requires starts within 10 ms and no extra note-off events.

Event bodies follow local smoothed power into a sustain reference. The search ends
at the next onset, 250 ms, or the source end. `endLimited` marks an unresolved body
transition at a search boundary; `overlapsNext` identifies that boundary as the next
event. This supplies a bounded interval and explicit body state for gain planning.
Minimum spacing after refinement is approximately 30 ms for attacks and 50 ms for
pulses. There is no signal mutation, oversampling or hidden dynamics processing.

By default only sparse events and the energy map remain. The temporary power map
uses O(duration / 1 ms) storage and the shared spectral features use
O(duration / 5 ms). `Options::retainFeatures` keeps the latter for downstream
analysis without another FFT scan; a silent source returns empty event/features
spans. `featureCenter(i)` gives the Hann window center, including padding outside
the source; this is distinct from a refined event position. Features include both
novelty curves, four SuperFlux registers, window power and low-band energy share.
Disable both maps and feature retention to perform just the energy scan.

The same source identity, revision, fingerprint, cancellation and memory contracts
apply. Changed PCM between scans produces `SourceMismatch` and no analysis.
`algorithmRevision` identifies the detector policy for host cache metadata.

The attack map also recovers source-energy restarts after 8 ms of local quiet
with at least 30 dB contrast. This prevents a louder event elsewhere from masking
an isolated quiet hit. Such events have `energyRestart = true`; their `strength`
is normalized local RMS rather than normalized spectral novelty. A missing left
context is not treated as measured silence. Padded spectral frames beyond the
source ending cannot create attacks. Pulse selection remains a separate policy.

## Automatic peak compression

`OfflinePeakCompressor<T>` accepts the positive `reductionDb` control (0..18 dB,
default zero). It measures the complete source's linked sample peak and sustained
level using `EnvelopeFollower` at 60 ms attack and 300 ms release. Available
headroom caps the requested reduction; the report distinguishes a crest limit
from a full reduction. The algorithm targets sample peaks and reports true peaks;
it does not impose a separate true-peak ceiling or add makeup gain.
At the subnormal floor, a target may not be representable in the source sample
type. The ceiling rounds toward the input and reports `ResolutionLimited`, with
the achievable reduction. It never rounds the entire peak to silence while
claiming that a finite requested reduction was achieved.

```cpp
#include "DSPark.h"

auto compressOfflinePeaks(const dspark::AudioBuffer<float>& input,
                          dspark::AudioBuffer<float>& output, double rate)
{
    dspark::OfflinePeakCompressor<float>::Options options;
    options.reductionDb = 3;
    return dspark::OfflinePeakCompressor<float>().run(input, output, rate, options);
}
```

The source-clock attack map provides local hold intervals. Actual input peaks
are remeasured for the processing stage; above-ceiling excursions outside mapped
events also receive holds. A 30 ms quiet gap separates those additional excursions
without following individual cycles of sustained bass. These holds are reported
separately. Their strength is the gain required to bring their maximum sample to
the measured ceiling, so every above-ceiling source sample is covered.

The sparse gain computer takes the minimum gain of overlapping extended holds
and exponential recoveries. Four moving averages span the lookahead (2 ms by
default), smoothing changes between events as well as the initial attack. They
read future gain requirements, retaining the sample bound without delaying audio
or applying a final hard clip. Their arithmetic is shared with `Limiter`.
`releaseMs` is an amplitude recovery time constant, default 80 ms, adjustable
between 60 and 100 ms. The initial release transition also passes through the
lookahead smoother; it is not an instantaneous exponential corner. At 36 time
constants the remaining deficit is below 2.32e-16 and becomes exact unity.

`analyze()` builds the map and plan. `makePlan(source, analysis, options)` reuses an
existing `OfflineTransientAnalyzer::Analysis` and scans actual PCM for sustained
headroom, hold count and hold values. A nonzero reduction requires an attack map.
All three scans check its fingerprint. The resulting move-only plan retains
sparse holds, provenance and canonical exclusions, and supports `gainAt(frame)`
for inspection. Random lookup sums the finite smoothing support; rendering uses
a sequential cursor with bounded lookahead storage. No full PCM or per-sample
gain array is retained.

Rendering uses the same verified product maps and transaction as Leveler. It
reports requested, effective and achieved sample-peak reduction and output true
peak. Excluded intervals are exact PCM, including signed zero; smooth transitions
lie outside their bounds. An excluded peak can prevent the whole-file target,
which is reported as `ProtectedPeaks`. Amount zero and an entirely excluded source
are exact identity operations. These worker-only APIs provide no real-time audio
processing entry point and never enable oversampling implicitly.

`lookaheadMs` requests 0.1..10 ms of source-aligned anticipation. If the
bandlimited peak target cannot be attained by scaling the initial processing
delta, rendering doubles anticipation, up to 10 ms, and measures again. Event
holds and the release constant are retained. `requestedLookaheadMs`, `lookaheadMs`
and `renderAttempts` distinguish the request, actual anticipation and measurement
count. `Plan::gainAt` inspects the initial control; it is not a prediction of the
final interpolated sample ratio. If the target remains infeasible, no output is
published. This adaptation does not lower the source's global gain or hard-clip
an overshooting sample.

## Automatic transient boost

`OfflinePunch<T>` applies an equal `boostDb` (0..12 dB, default zero) to attack
events. It does not require a pulse map or tempo estimate, and does not scale
the boost by event strength. Processing is linked across mono/stereo channels.

```cpp
#include "DSPark.h"

auto emphasizeOfflineAttacks(const dspark::AudioBuffer<float>& input,
                            dspark::AudioBuffer<float>& output, double rate)
{
    dspark::OfflinePunch<float>::Options options;
    options.boostDb = 3;
    return dspark::OfflinePunch<float>().run(input, output, rate, options);
}
```

The envelope starts 5 ms before the refined attack, reaches full boost at that
attack, holds for its duration clamped to 15..50 ms, then returns to unity over
45 ms. It merges overlapping weights by maximum and computes gain as
`decibelsToGain(boostDb * weight)`. A positive four-box kernel smooths cosine
flanks and their intersections. Its 2 ms support is included in those timings:
there is no added audio delay or extension of the outer support. Hann window
generation and the box kernel reuse Core implementations. These operations
shape control data; they do not resample the audio.

`analyze`, `makePlan`, `render` and `run` follow the same worker, provenance,
budget and transactional contracts as Peak. The plan retains sparse events and
short rate-dependent window tables, not a complete PCM copy or gain array.
It accepts 8..384 kHz mono/stereo sources. `gainAt(frame)` inspects its linked
automation; rendering uses a bounded sequential cursor. Exact exclusions use
5 ms transitions outside protected intervals. Zero amount is exact PCM identity.

There is no full-scale limiter, normalization, makeup or hidden output trim.
Output may exceed 0 dBFS. If a boost would overflow the sample type, planning
reduces the common boost and reports `RepresentabilityLimited`; this considers
only affected samples, so an excluded large peak does not constrain unrelated
transients. `effectiveBoostDb` describes the envelope plateau, not an assertion
that the track's global peak increases by that amount. The result separately
measures sample and true peak, including floating-point PCM rounding.

## Shared tempo analysis

`OfflineTempoAnalyzer<T>` feeds retained pooled-channel spectral features to the
existing `BeatTracker` conditioning, metrical ranking and grid engine. It does
not run another FFT or duplicate the tempo algorithm. To reuse an existing
`OfflineTransientAnalyzer::Analysis`, enable both `pulses` and `retainFeatures`
when constructing it. The direct source overload builds those features with two
bounded source scans. Neither overload owns source PCM.

```cpp
#include "DSPark.h"

auto analyzeOfflineTempo(dspark::OfflineAudioSource<float>& source)
{
    dspark::OfflineTransientAnalyzer<float>::Options options;
    options.retainFeatures = true;
    auto features = dspark::OfflineTransientAnalyzer<float>().analyze(source, options);
    if (!features.succeeded())
        return dspark::OfflineTempoAnalyzer<float>::Result{features.status, {}, 0};
    return dspark::OfflineTempoAnalyzer<float>().analyze(features.analysis);
}
```

The immutable result preserves source identity, revision, fingerprint and timeline
origin. Beat `frame` values are relative int64 source positions. They locate
interpolated spectral-feature centers; event processors use the separate refined
pulse map for acoustic attack positions. The global report includes tempo, its
competing metrical interpretation and confidence. Search defaults to 40..240 BPM
and may be set within 20..480 BPM.

`tempoAt(frame)` uses the median of the current and adjacent beat intervals to
reject single-interval timing jitter; a missing neighbor repeats the current
interval. Original grid positions remain available. It returns that local tempo
only when confidence meets `minimumConfidence` (default 0.25) and both ends
have a nearby detected pulse. The search range constrains nominal metrical
candidates; local intervals of an expressive grid may depart from it and are
not clipped to it. The anchor tolerance is one eighth of that interval,
capped at 50 ms. It returns zero in unsupported gaps
and rejects a raw interval more than twice that anchor tolerance away from its
median, preventing a missing-beat gap from being hidden by the local smoothing.
It also returns zero outside anchored intervals, so a consumer can explicitly use its fallback
release. This is a deterministic evidence gate, not a probability of correctness
or a promise to resolve every half/double/ternary ambiguity. Uncertain analyses
still return their grid and diagnostics.

The shared engine charges actual requested allocation payload, including dynamic
programming and temporary grid vectors, before each request. It supports worker
cancellation during its principal loops; progress has `total == 0` when the
remaining iterative work is unknown. Memory-limit, cancellation and allocation
failures publish no partial analysis.

When the global confidence is low, overlapping feature windows ask the same
tempo engine for local evidence. Windows span at least eight seconds and four
beats at the slowest searched tempo. An interval is confirmed only if a local
result meets the unchanged confidence threshold and contains matching consecutive
grid points; the pulse anchors and timing-consistency checks still apply.
`localWindows` and `locallyConfirmedIntervals` report this work separately from
the global confidence. Original grid positions do not move. Scratch blocks are
reused between windows, with all backing allocations and bookkeeping budgeted.

## Beat compression

`OfflineBeatCompressor<T>` levels detected rhythmic pulses against their upper
median peak. It measures the maximum absolute sample across channels during the
first 30 ms of each pulse on the actual input to the stage. Only positive excess
above that reference is reduced, by at most `reductionDb` (0..18 dB). The default
is zero. The reference includes protected pulses; changing exclusions does not
change the level the remaining pulses are compared with.

```cpp
#include "DSPark.h"

bool levelOfflineBeats(const dspark::AudioBuffer<float>& input,
                       dspark::AudioBuffer<float>& output, double sampleRate)
{
    dspark::OfflineBeatCompressor<float>::Options options;
    options.reductionDb = 6;
    const auto result = dspark::OfflineBeatCompressor<float>().run(
        input, output, sampleRate, options);
    return result.succeeded();
}
```

The shared attenuation curve has a 4 ms preroll and 2 ms finite attack support,
and holds each measured pulse body. Overlaps select minimum gain. All recovery
tails follow one exponential amplitude-release clock; each pulse's local release
takes effect when its extended hold ends, including pulses needing no reduction.
The four-box attack replaces the source application's short attack pole with a
finite, source-aligned envelope. No audio delay or hidden makeup is introduced.

For a reduced pulse whose anticipation or body touches a source boundary, the renderer verifies its
30 ms peak window. If necessary, it adjusts the exterior gain continuation within
the nonnegative attenuation range. The source-frame curve, pulse hold and release
remain unchanged. Both boundaries are solved together; the closest feasible
continuation is selected, and the actual exterior gains are reported. Exclusions
and their feathers retain the scalar plan's admissible level. An infeasible
target returns `NumericalFailure` before publication; there is no output clip or
master gain trim.

Release defaults to one quarter note of reliable local tempo. `releaseNote` also
offers eighth, half and whole notes. `manualBpm` explicitly overrides analysis
within 20..480 BPM. Unsupported tempo uses 250 ms, independent of note selection;
each immutable `Pulse` exposes its release, reduction and `TempoSource`, and the
report counts fallback pulses. A pulse at a grid boundary can use the adjoining
reliable interval. A missing outer grid point can be bridged by one measured
quarter note, within 50 ms; the processor does not extrapolate through an interior
gap. An ambiguous global tempo remains visible even when local intervals are
independently confirmed.

`makePlan(source, features, tempo, options)` reuses source-bound analysis and makes
one verified PCM scan. Supplied maps must match the source identity and content.
The overload without a tempo map builds one from retained spectral features;
manual BPM skips that work. `options.tempo` configures newly constructed maps;
a supplied tempo analysis keeps its own settings. A weaker amount creates a new
plan and cannot inherit earlier stronger gain state.

`maximumReductionDb` describes the planned control. Bandlimited rendering can
change individual sample peaks and produce interpolation tails; output/input
sample ratios are not a gain meter. The renderer reports final sample and true
peak, control bounds and any representability adjustment. There is no limiter,
global trim or automatic normalization. Zero amount and explicit exclusions keep
the original PCM exactly. Plans are immutable, move-only and contain sparse
events and automation, never an owned copy of the audio.

## Automatic clipping

@ref dspark::OfflineSoftClipper and @ref dspark::OfflineHardClipper analyze the
complete source and calibrate a shared channel-linked curve ceiling. A positive
`reductionDb` requests 0..12 dB of **sample-peak reduction** relative to the full
input. There is no makeup gain, master trim or additional output limiter. Zero
amount and protected samples retain exact PCM, including signed zero. The soft
class defaults to `Curve::Sine`; `Tanh` and `GoldenRatio` reuse the same curves as
`Clipper`. These are mathematical waveshapers, not models of measured hardware.

```cpp
#include "DSPark.h"

bool softenPeaks(const dspark::AudioBuffer<float>& input,
                 dspark::AudioBuffer<float>& output, double sampleRate)
{
    dspark::OfflineSoftClipper<float> clipper;
    dspark::OfflineSoftClipper<float>::Options options;
    options.reductionDb = 3;
    options.curve = dspark::OfflineSoftClipper<float>::Curve::Sine;
    options.oversamplingFactor = 4;
    const auto result = clipper.run(input, output, sampleRate, options);
    return result.succeeded();
}
```

The internal correction integrates a degree-11 reconstructed signal over
continuous intervals, with cubic B-spline averaging, finite compensation and a
matched linear branch. Complete-source cardinal-sinc interpolation and projection
reuse Core FFT convolution and coarse Cauchy maps. They include the finite
source's reconstruction tails rather than imposing a finite FIR transition band.
`oversamplingFactor` explicitly accepts 1, 2, 4, 8 or 16, with default
4 for this offline algorithm. Lower factors reduce cost but have larger
high-frequency errors; they are not equivalent quality modes. This setting does
not change any other effect's oversampling. Bilateral offline evaluation produces
samples at their original indices, so `compensatedLatencyFrames` is zero. Output
keeps the original length and source clock. Samples outside the source are zero;
their bandlimited reconstruction can have nonzero tails. Adaptive exterior support
bounds the omitted nonlinear contribution; it does not append output frames.

`analyze(source, options)` returns an immutable, move-only plan. To reuse an
existing `OfflineEnergyAnalyzer` result, call
`makePlan(source, analysis, options)`. Plans own copied exclusions, format and
PCM fingerprint; they do not retain source samples or filtered PCM. Each render
reuses its DSP storage between calibration passes, rebuilding curve-dependent
fields. It reads bounded blocks of at most `min(job.blockFrames, 4096)` frames.
PCM caches and filter storage are bounded. Coarse interpolation maps grow with
source duration and exterior support; no complete upsampled PCM copy is retained.
The internal map block size is selected within 128..16384 source frames, with
at most 65536 frames on the oversampled grid. The temporary energy analysis
uses the existing 100 ms map. All allocation requests, including geometry changes
during calibration, remain subject to the payload budget.

Calibration measures complete, rounded output PCM and adjusts only the curve
ceiling. It permits at most 24 trial passes and requires a peak error within
0.005 dB of the representable target. The final render must reproduce the measured
peak before the sink can commit. `Report::calibrationPasses`, `normalizedCeiling`,
`compensatedLatencyFrames`, `achievedReductionDb`, `targetErrorDb` and `targetMet`
make this visible. Output sample and true peak are measured independently; a
sample-peak target is not a true-peak ceiling. True peak uses the Core detector
including its flushed tail.

Exclusions are exact source-frame constraints with external 5 ms feathers. A
protected global maximum can make a requested global reduction impossible.
`TargetUnreachable` with `Reason::ConstraintLimited` refuses that render without
publishing output. Other calibration failures also publish nothing. A fully
excluded source returns `NoChange` and reports whether the target was met.
Subnormal PCM may not represent the requested target: it is rounded toward the
input peak and reported through `effectiveReductionDb` and
`representabilityLimited`. There is no hidden gain adjustment to conceal this.

## Stereo generation and reusable delta

@ref dspark::OfflineStereoGenerator uses the same twenty-band processed copy,
rational color, original subtraction and delta add-back as
@ref dspark::StereoGenerator. It accepts complete sources at 8..384 kHz and
produces exactly two channels with the original frame count. A mono source
requires `duplicateMono = true`. Width is 0..1, with default zero. The local
color oversampling factor is explicitly 2, 4, 8 or 16 (default 4); other
effects' settings are unchanged. `lowCutHz` optionally filters only the delta.
See the [stereo recipe](cookbook.md) for the core's frequency
transition, source behavior and latency tradeoffs.

```cpp
#include "DSPark.h"

bool generateStereo(const dspark::AudioBuffer<float>& input,
                    dspark::AudioBuffer<float>& output, double sampleRate)
{
    dspark::OfflineStereoGenerator<float>::Options options;
    options.width = 0.4f;
    options.duplicateMono = true;
    const auto result = dspark::OfflineStereoGenerator<float>().run(
        input, output, sampleRate, options);
    return result.succeeded();
}
```

`analyze` scans complete PCM for its fingerprint and peak without building
energy, transient or tempo maps. `makePlan(source, energyAnalysis, options)`
reuses an existing energy analysis. `replan(source, previousPlan, options)`
updates controls and copies new exclusions without rereading PCM; rendering
still verifies the complete source. Zero width, silence and fully excluded
sources skip the DSP core. Exclusions retain original samples exactly, with
external 5 ms feathers applied to the delta. Outside them, the same change is
added to the left and subtracted from the right, preserving mid up to output
rounding. There is no limiter, side-energy normalization or master gain trim.

Rendering reads at most `min(job.blockFrames, 256)` frames at once. It starts
empty filter history at `timelineOrigin`, flushes zeros, removes the core's
alignment delay and truncates the tail to the original duration. Offsets and
exclusions remain source-relative. A trimmed input does not restore earlier
filter history. `compensatedLatencyFrames` reports the removed delay; skipped
generation reports zero. Output sample and true peak are measured from final
rounded PCM, including the meter's flushed tail. These readings report required
headroom, not a ceiling promise. An unrepresentable result fails transactionally.

For repeated widths, `buildDeltaCache(source, plan, deltaSink)` writes one
canonical, unit-width **mono double** delta and returns a move-only `DeltaCache`
certificate only after commit. The caller owns the sink and its storage, which
can be memory or a file. Generation ignores requested width and exclusions;
they apply later at composition. `cache.getSpec()` gives the exact format and
source metadata required of a source adapter for that stored delta.

`renderCached(source, plan, deltaSource, certificate, outputSink)` composes the
cached delta with original PCM, avoiding band/color generation. A different
width or exclusion list can reuse the certificate. Different source PCM,
format, identity, revision, timeline origin or generation settings cannot.
Both complete streams are fingerprinted even at zero width, and invalid cache
content aborts provisional output. The certificate owns metadata only and does
not retain references to host storage. Cache construction still processes the
unit-width path; its numerical representability is required even if a later
composition requests zero width. Worker memory is independent of file duration;
host cache storage requires eight bytes per source frame, excluding its metadata.
