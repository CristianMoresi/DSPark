# Streaming measurement of sample intervals

`AudioIntervalAnalyzer<T>` selects exactly `[begin,end)` in source frames (one
sample per channel), with nonnegative signed 64-bit coordinates. Feed contiguous
blocks and call `finish()` after the interval or at EOF. Neither block boundaries
nor the decoder's chunk size define a measurement boundary. The analyzer retains
no PCM and performs no allocation; storage is fixed, independent of duration.
Feed several instances the same blocks to measure overlapping regions in one pass.

See the compilable `examples/interval_analysis.cpp` example and the
`tests/TestAudioInterval.cpp` regression tests. Include
`Analysis/AudioIntervalAnalyzer.h`, or `DSPark.h` with Analysis enabled.

## Existing implementations reused

| Existing API | Capability and reuse |
| --- | --- |
| `LoudnessMeter<T>::process/processBlock` and loudness readouts | K-weighting, complete 400 ms gating windows, integrated energy gating, 3 s short-term windows and LRA histograms. The interval adapter supplies only the selected fragment to its regional meter. |
| `LoudnessMeter<T>::finalizeTruePeak()` | Completes the finite interpolation tail without advancing loudness duration, filters, windows or gates. |
| `LoudnessMeter<T>::getMeasurementInfo()` | Stream-owner snapshot of committed/pending frames, hop length and counts passing each absolute gate. Used to reject incomplete readouts. |
| `TruePeakDetector<T>::getTailPeak(channel)` | Evaluates eleven zero frames on a copy of one Core detector history, without allocation or live-state mutation. Shared by the meter. |
| `OfflineEnergyAnalyzer<T>::analyze()` | Full-source 100 ms RMS/peak bins. Bin summaries cannot recover arbitrary sub-bin intervals. Its overflow-safe energy accumulator was extracted to shared Core and is reused here. |
| `LevelFollower<T>` | Exponential peak/RMS envelopes. These ballistics are useful for live displays but do not provide rectangular interval RMS. |

There is one loudness implementation and one true-peak filter. Integrated
loudness is obtained from the meter's gated power histogram; LUFS readings are
never averaged to obtain regional loudness. Existing histogram quantization is
0.1 LU, so this API does not promise exact unquantized integrated/LRA results.

## Numeric and temporal contract

`prepare(spec, interval, context, streamOrigin)` resets the measurement. Layouts
are mono or stereo; other channel counts are rejected. `streamOrigin` defaults
to zero and must be at or before `begin`. `processBlock(view, firstFrame)` requires
contiguous positions starting there and no block larger than `spec.maxBlockSize`.
The API accepts the same finite positive sample rates as `AudioSpec`; numerical
failures of the underlying meter remain explicit, not valid measurements.

| Result | Definition |
| --- | --- |
| `requested` | The caller's exact source-frame interval. |
| `measured` | The actual selected frames received. Early EOF yields `Truncated`; a request beyond EOF yields an empty measured interval at `begin`. |
| `rms` | `sqrt(sum(x[c,n]^2) / (frames * channels))`, with scaled accumulation to avoid overflow/underflow. Linear full-scale units; stereo channels are pooled equally. |
| `samplePeak` | Maximum absolute input sample over the selected frames and channels, in linear full-scale units. |
| `truePeakDbtp` | Maximum of original samples and the existing BS.1770 four-phase FIR reconstruction of the finite selected fragment, including its zero-extended tail. The meter display floor is -100 dBTP. |
| `region` | Loudness of the fragment measured independently, with K filters, windows and gates reset at `begin`. |
| `continuous` | Optional continuous loudness observations from `streamOrigin` to the selected end. Each observation includes its own input/support interval. |

True peak always uses the **finite fragment convention**, in either context mode.
Samples before `begin` and at/after `end` do not enter its interpolation history.
Zero extension can produce ringing outside the fragment; that reconstruction tail
is included in the peak but never adds frames to `measured` or loudness. This is
not a crop of an interpolator running over the entire recording.

`Context::Independent` only measures the fragment. Blocks before it are skipped;
a caller with random access may start feeding at `begin` by setting `streamOrigin`
to `begin`. `Context::IncludeContinuous` **also** runs a separate instance of the
existing meter from `streamOrigin`. Its K-weighting state and windows carry prior
audio across the selected start. It reports cumulative integrated/LRA observations
from that origin, not regional integrated/LRA values. Regional results remain
available alongside them. No sample at or after the selected end is processed by
either meter, even if it occurs in the same supplied block.

For continuous observation matching a meter started at frame zero, feed every
frame from zero. Starting at a later origin deliberately defines a new continuous
measurement; it cannot recreate filter history that was never supplied.

### Window coverage

The existing meter commits a hop of `floor(sampleRate / 10)` frames, clamped to
`[1,1000000000]`. The integer hop is the timing authority. M and I require four
complete hops; S and LRA require thirty. At 48 kHz these are exactly 400 ms and
3 s. At unusual rates, use the returned hop/coverage rather than assuming a
fractional frame. This API preserves the meter's 10 Hz readout cadence.

`Loudness::input` is the supplied scope. `committedEnd` is the last completed hop's
source position; `pendingFrames` identifies the uncommitted suffix. M/S
`Measurement::support` is the actual four/thirty-hop window ending there. It may
start before the requested region in a continuous observation. Integrated/LRA
support bounds the complete windows submitted to their respective gates; gated-out
windows inside that span do not contribute. IIR filters can retain dependence on
earlier samples within `input`, even outside the displayed M/S window.

A trailing partial hop is retained internally but is not padded, committed or
included in loudness windows by `finish()`. It **is** included in RMS, sample peak
and true peak. This is exposed in the returned coverage, not silently represented
as a loudness measurement over extra silence or an invented partial gating block.

## Validity and errors

Every measurement has a `Validity`; `hasValue()` accepts only `Valid` and
`Silence`. Unavailable values are NaN, never a plausible incomplete LUFS reading.

| Validity | Meaning |
| --- | --- |
| `Valid` | Complete supported numeric measurement; normal meter flooring still applies. |
| `Silence` | Nonempty scope containing only exact zero PCM, with enough complete windows for this measure. RMS/peak are zero; the meter returns -100 LUFS/dBTP and LRA zero as explicit silence sentinels. |
| `Empty` | No selected frames, including a successfully requested `[n,n)`. |
| `TooShort` | Fewer than four complete hops for M/I, or thirty for S/LRA. A partially filled window is not treated as silence-padded valid audio. |
| `BelowGate` | Complete I/LRA windows exist but none passes the absolute gate; nonzero quiet audio is distinguished from exact silence. |
| `Incomplete` | EOF occurred before the requested end. The actual received interval is reported, but the requested measurement is not complete. |
| `InvalidInput` | Invalid configuration, a gap/overlap, layout/block violation, or non-finite PCM in a measured scope. |
| `NumericalError` | Finite input exceeded the existing meter's representable filter/power/histogram range. RMS/peak can remain valid independently. |
| `NotRequested` | Continuous observations were not requested. |

LRA with one complete accepted short-term window is mathematically zero, per the
existing estimator. That result does not establish a representative programme
distribution. A full window and useful programme statistics are different claims.

`finish()` is idempotent and seals the analyzer. Subsequent blocks after successful
completion are ignored. An early `finish()` requires a new `prepare()` to restart.
Non-finite samples in ignored prefixes/suffixes do not invalidate an independent
fragment; the same prefix is part of a requested continuous observation and must
therefore be finite. No call is safe concurrently with another on the same
instance. Copy the finished value result for publication to another thread.

## Expected regression results

| Signal / request | Expected result |
| --- | --- |
| Stereo four-frame fixture in `AudioInterval_exact_RMS_and_peaks_at_half_open_boundaries` | RMS `sqrt(0.46875/8)`, sample peak `0.5`; outside peaks of magnitude 10 excluded. |
| One mono sample `0.5` | RMS/peak `0.5`, true peak `-6.0205999133 dBTP`; M/I/S/LRA unavailable as too short. |
| Three seconds of digital silence at 48 kHz | Explicit silence, RMS/peak zero, I `-100 LUFS`, LRA zero. |
| One-second finite ending `...0.05,+0.9,-0.9` | Final true peak approximately `-0.244683381 dBTP` for float, including the interpolation tail. |
| Equal 16 s stereo 997 Hz sections at -20 and -30 LUFS | Integrated approximately `-22.5964 LUFS` within 0.1 LU, LRA 10 LU; not the LUFS arithmetic mean -25. |
| 19199/19200 and 143999/144000 frames at 48 kHz | M/I and S/LRA respectively cross their complete-window validity boundaries. |

Tests also compare full results with block sizes 1, 127, 512, 1009 and 4096,
float/double, nonaligned boundaries and a source origin above `2^53`. Continuous
observations are checked against a separately driven `LoudnessMeter` on identical
input, while regional energy has direct sample arithmetic oracles.

Window/gate conventions follow [ITU-R BS.1770-5](https://www.itu.int/rec/R-REC-BS.1770-5-202311-I/en),
[EBU Tech 3341](https://tech.ebu.ch/publications/tech3341) and the existing
LRA implementation in `Analysis/LoudnessMeter.h`.
