# Migrating from DSPark v1.6.1 to v1.7.0

Most v1.6.1 code remains source-compatible. The changes below affect explicit
types, latency assumptions or thread ownership and should be reviewed during an
upgrade.

## Analysis-window latency

Default analysis windows now preserve approximately the same time span across
sample rates instead of preserving a fixed sample count. This keeps frequency
register and spectral resolution consistent, but increases the resolved window
at higher rates.

Callers that relied on default-window latency at sample rates at or above
88.2 kHz may pass `windowSize=4096` explicitly and accept the documented
reduced register. Use `getWindowSize()` for latency accounting rather than
assuming a fixed default. Components that name the parameter `fftSize` expose
the equivalent resolved-size getter documented by that component.

Explicit positive window and hop values remain available when a fixed count is
part of an application's contract. Above the automatic window's documented
sample-rate ceiling, choose an explicit size if the application needs a lower
frequency floor.

## Biquad coefficients and publication ownership

`BiquadCoeffs<T>` is now the non-template `BiquadCoeffs`; its fields and the
biquad recursion use double precision regardless of the buffer sample type.
For example:

```cpp
// v1.6.1
auto coefficients = dspark::BiquadCoeffs<float>::makeLowPass(48000.0, 1200.0);

// v1.7.0
auto coefficients = dspark::BiquadCoeffs::makeLowPass(48000.0, 1200.0);
```

The former `BiquadCoeffs::makeDcBlocker()` convenience factory is no longer
part of the coefficient API. Use `DCBlocker<T>` for DC removal, or
`BiquadCoeffs::makeHighPass()` when a general biquad high-pass is specifically
required.

Choose the coefficient setter by origin:

- `setCoeffs()` publishes a set from a control thread to the processing thread.
  Adoption on the audio thread is bounded; if publication overlaps every read
  attempt, the processor keeps its previous complete set, re-arms the pending
  update and adopts it on a later call.
- `setCoeffsNow()` directly updates audio-thread-private state. Use it only for
  coefficients computed by the stream owner, or in single-threaded/offline
  processing. It is not a replacement for concurrent control-thread
  publication.

Designate one master for each embedded `Biquad`: either control-published or
stream-owner-modulated. Mixing both origins on one instance makes the last set
adopted by the audio thread win and obscures ownership.

## `AudioBufferView` type queries

The converting and pointer-array constructors now constrain the pointer
conversion instead of failing with a check inside the constructor body. A
mutable view still converts to a const view; the reverse now correctly fails
during overload resolution and in type traits:

```cpp
static_assert(std::is_convertible_v<
    dspark::AudioBufferView<float>,
    dspark::AudioBufferView<const float>>);

static_assert(!std::is_convertible_v<
    dspark::AudioBufferView<const float>,
    dspark::AudioBufferView<float>>);
```

No previously valid evaluated conversion changes meaning. Code that depended
on the old false-positive result in `std::is_convertible_v`,
`std::is_constructible_v`, `decltype`, SFINAE or a `requires` expression should
query the legal mutable-to-const direction instead.

`AudioBuffer<T, MaxChannels>::toView()` now preserves `MaxChannels` in the
returned view type. Prefer `auto` for the result, or spell the matching
`AudioBufferView<T, MaxChannels>` type when an exact type is required.

## `AudioProcessor` means in-place

`AudioProcessor` now accepts only processors whose `processBlock` contract is
unambiguously an in-place mutable-buffer operation. Read-only analysers such as
`BeatTracker`, `LoudnessMeter`, `OnsetDetector` and `PitchFollower` no longer
satisfy it; run them beside a `ProcessorChain` on a const view of the same
buffer.

A type exposing both mutable and const `processBlock` overloads is treated as
read-only by the concept. If the type is intended to be an insert, give its
analysis operation a distinct name.
