# Threading model

Every DSPark component that can be touched by more than one thread follows the
same contract, and states in its own header wherever it departs from it. The
contract is written down once here; each header's `Threading:` block then only
has to say which of its own methods belongs where.

## The two threads

DSPark assumes **one control thread and one audio thread** -- single producer,
single consumer.

| Role | What runs there |
|---|---|
| **Setup** | `prepare()`, `setState()`, construction, moves, and -- unless a header says otherwise -- `reset()`. Never concurrent with processing. |
| **Audio** | `processBlock()` / `processSample()` and anything else that runs inside the callback. Exactly one stream owner per object. |
| **Control** | Parameter setters. **One** non-audio writer. |
| **Any thread** | Lock-free readouts: getters that load a published atomic word, or hand back a private copy of one. These may run concurrently with processing. The `get` prefix is not the criterion -- a getter that returns a reference or a raw pointer needs the rule in "Pointers returned across threads" below. |

Two things fall outside the contract, and a component that needs them says so
in its own header:

- **More than one control writer.** Route the extra producers through
  `Core/SpscQueue.h` (one queue per producer) or design for it explicitly.
- **`prepare()` or `reset()` concurrent with processing.** Setup calls
  reallocate; nothing may be running in the callback while they do.

Thread-safety claims in DSPark are scoped -- to a method, or to a named group
of setters. A few class briefs do say "thread-safe" without qualification; read
that as shorthand for the per-method contract stated further down in the same
header, not as a class-wide guarantee, because there is no useful version of
that sentence.

## Which words must be atomic

Any word that both threads can reach concurrently is `std::atomic`, or it
travels through `Core/SpscQueue.h`. No plain scalar, struct or array is read on
one thread while another writes it -- including inside a seqlock's critical
section -- and no hand-off is read across threads *by reference*, which would
let the reader observe the writer's later stores. (A getter that returns a
reference to state owned by the processing thread is a different animal; see
the exception at the end of "Pointers returned across threads".)

The words are chosen so they cannot lock. `tests/TestCoreFoundation.cpp`
asserts `std::atomic<T>::is_always_lock_free` for nine word types -- `bool`,
`int`, `unsigned`, `std::size_t`, `std::uint32_t`, `std::int64_t`,
`std::uint64_t`, `float` and `double` -- so a target where one of them falls
back to a mutex fails the suite.

Those nine are what the suite pins, not every word the framework publishes. An
enum word inherits the property from its underlying integer type, and the
atomic enums here are backed by the integers above, so they hold without being
asserted one by one. One published word is neither an integer nor an enum:
`Effects/Saturation.h` hands its active algorithm across as an atomic pointer,
lock-free on every supported target and outside the census.

That is a run-time census of the word types. It is **not** a compile-time check
on your component, and the build will not stop you using a word the census
never saw. Where the word type is a template parameter the census cannot reach
it at all. Four headers pin it themselves -- `Analysis/SpectrumAnalyzer.h`,
`Effects/AutoGain.h`, `Effects/Equalizer.h` and `Effects/DynamicEQ.h` -- and
most of the headers that declare such an atomic do not. A new component with an
atomic word on the audio path should do the same:

```cpp
static_assert(std::atomic<T>::is_always_lock_free,
              "audio-thread stores must not lock");
```

Without it, a type that is not lock-free on some target quietly takes a mutex
inside the callback and nothing says a word.

## The two publication patterns

Parameter and coefficient hand-off uses one of exactly two shapes.

### Single independent word

Used when the value stands on its own and no invariant couples it to any other
published value: cutoff, mix, an enum, a readout of the current LFO value.

The writer validates, then `store(std::memory_order_relaxed)`. The reader does
`load(std::memory_order_relaxed)`. On x86 and ARM64 both compile to a plain
move, so the hot loop pays nothing.

### Multi-word set with an invariant: the seqlock

Used when several words must be adopted together, because half of one update
combined with half of the next is not a valid state -- a biquad's five
coefficients, an FIR's coefficient vector, a band configuration.

**The origin rule.** The staged channel exists for control-to-audio
publication only: a value computed on the audio thread for its own use is
written directly into the audio-thread-private active state (as
`Core/Biquad.h`'s `setCoeffsNow()` does), and the stream owner never
self-publishes -- self-publication would pay the cross-thread publish and then
re-adopt the same thread's own write inside the hot path, and it is what keeps
every adoption below cold: only the control thread can arm one.

Writer, on the control thread only:

```cpp
seq.fetch_add(1, std::memory_order_acq_rel);      // -> odd: write in progress
std::atomic_thread_fence(std::memory_order_release);
// relaxed stores of every data word
seq.fetch_add(1, std::memory_order_release);      // -> even: complete
dirty.store(true, std::memory_order_release);
```

Reader, on the audio thread. Every line of the comment marked `bounded seqlock
read` is load-bearing; `tools/verify_threading_doc.py` finds the framework's
audio-path readers by that marker and checks each one against this shape:

```cpp
if (!dirty.exchange(false, std::memory_order_acquire)) return;  // nothing new

// bounded seqlock read: at most kSeqlockMaxAttempts validation attempts
bool adopted = false;
for (int attempt = 0; attempt < kSeqlockMaxAttempts; ++attempt)
{
    const unsigned s0 = seq.load(std::memory_order_acquire);
    if ((s0 & 1u) != 0u) continue;   // writer mid-publish: do not copy at all
    // relaxed loads of every word into a THREAD-PRIVATE plain destination
    // that is NOT the set the hot path is currently reading
    std::atomic_thread_fence(std::memory_order_acquire);
    if (s0 == seq.load(std::memory_order_relaxed)) { adopted = true; break; }
}
if (!adopted) { dirty.store(true, std::memory_order_release); return; }
// commit the private copy -> active set (plain, audio-thread-private)
```

Both fences are mandatory, not decoration. Without them the relaxed data
accesses are not ordered against the counter, and a reader can observe a
mid-update word while still seeing an even counter -- the classic seqlock
mistake. The pairing is the one [atomics.fences]/2 defines; the analysis is
Boehm, *"Can Seqlocks Get Along With Programming Language Memory Models?"*,
MSPC 2012.

**The loop is bounded.** A reader that loses `kSeqlockMaxAttempts` attempts
adopts nothing, keeps the coefficient set it is already using, and re-arms the
dirty flag, so the update lands on a later call instead of holding the callback
open. Bounding cannot make a torn set adoptable: the accept test is unchanged,
and giving up adopts nothing. What it can do is deliver a parameter change one
block late under sustained contention, which is the trade being made. Without
the bound the reader's worst case is set by the writer's scheduling rather than
by its own instruction count: with an 8192-tap FIR published from a control
thread sharing one CPU with the audio thread -- the ordinary arrangement on the
single-core embedded and wasm targets -- one `processBlock()` call was measured
at 5.1 seconds.

Two details of the shape are what make the bound safe rather than merely short.
The oddness test sits **before** the copy, so meeting a writer mid-publish costs
one relaxed load instead of a whole coefficient set copied and then discarded;
that is what keeps three attempts affordable even from `processSample`, where
"a later call" means the next sample. And the copy goes to a destination the
hot path is **not** reading, committed only once it validates -- copying
straight into the live set was safe only while the loop could not exit before
the copy was valid, and a bounded loop that gave up would leave a torn mixture
in force.

Note what the reader does after accepting: it works from a **private plain
copy**, never from the shared words. The per-sample loop is then ordinary
non-atomic code and runs at the same speed it would with no hand-off at all.

The bound applies to readers **on the audio thread**, and only there.
Control- and GUI-thread readouts of the same published state keep the unbounded
retry: they are not real-time, and they have no previously adopted copy to fall
back on, so "try again" would be a worse answer than a microsecond of spinning.
A seqlock reachable from both exposes both entry points -- a bounded
`tryRead(T&)` for the audio path and an unbounded `read()` for the readouts --
as `Effects/Equalizer.h` and `Effects/DynamicEQ.h` do.

`Core/FIRFilter.h` (runtime-sized coefficient vector) and `Core/Biquad.h`
(five named coefficients) are the reference implementations. A new hand-off
should match one of them frame for frame rather than improvise.

## Pointers returned across threads

A getter that returns a pointer or reference to shared storage has to answer a
second question beyond "who may touch this now": **how long does the returned
pointer stay valid?** Ownership at the moment of the call says nothing about
the next call.

DSPark's answer is that no pointer into *published* (cross-thread) storage
escapes a component. `Analysis/SpectrumAnalyzer.h` is the worked case.
`getMagnitudesDb()` and `getPeakHoldDb()` each acquire the freshest published
slot, copy it into reader-private snapshot storage while they still own that
slot, and return a pointer to the copy. The audio thread cannot reach a
snapshot, so the caller may hold the pointer for as long as it likes without
another thread writing underneath it.

Naming both getters is the point, because one alone does not show why the copy
is necessary rather than merely tidy: they share one slot pool, so the next
acquisition by *either* of them hands the previously held slot back to the
writer. A pointer into the slot would therefore be invalidated by the *other*
getter's next call. A pointer into a snapshot is not: each snapshot is
rewritten only by its own getter, plus the setup-only `reset()` / `prepare()`.

**The exception, stated here because the headers state it.** A getter that
returns a reference to state owned by the *processing* thread is not a
cross-thread readout, and the rule above does not cover it. `Core/Biquad.h`'s
`getCoeffs()` returns a reference straight into the active coefficient set,
which the processing thread rewrites in `applyPendingCoeffs()`; a GUI thread
reading it concurrently with a promotion may observe a half-updated set. The
header says so at the method, and says what to do instead -- keep your own copy
of the coefficients you computed. `Core/ProcessorChain.h`'s `get()` has the
same shape: it hands back the sub-processor itself, so what you may do with the
reference is that processor's contract, not the chain's. The rule to carry
away: a getter that returns a reference or a raw pointer belongs to the
processing thread unless its own documentation says otherwise.

## Blocking

`Core/SpinLock.h` exists for mutual exclusion **off** the audio path. The audio
thread never calls its blocking `lock()`; only `tryLock()` / `ScopedTryLock`,
which cannot be made to wait on a control thread. An audio-thread reader that
can block on a control-thread writer is priority inversion by construction, and
the framework contains exactly one, named here rather than left to be found.

`Effects/Reverb.h` publishes its impulse-response bank behind a one-flag
spinlock instead of a lock-free swap. `loadBank()` runs from `processBlock()`
and spins on the flag `storeBank()` holds while `loadIR()` swaps a bank in from
the control thread. The critical section is a single shared-pointer copy, so the
audio thread normally waits nanoseconds; if the control thread is descheduled
inside it, the audio thread spins for a whole scheduling quantum inside the
callback, which is a dropout. The method states the trade where it is made, and
a wait-free reclaim is backlogged.

Two other things on the audio path repeat work, and neither is a wait. The
seqlock readers above retry at most `kSeqlockMaxAttempts` times and then give
up; the worst case is that many copies of one coefficient set -- measured at
about 5 us for an 8192-tap FIR set, and proportional to the set size -- and it
is set by DSPark's own instruction count, not by another thread's scheduling.
`Analysis/SpectrumAnalyzer.h`'s `acquireLatestSlot()` is a lock-free CAS loop
against a writer that holds nothing, so it cannot be made to wait either. Three
magic statics are reachable from a processing call (`Core/MinBlepTable.h`,
`Effects/AlgorithmicReverb.h`'s prime table, `Core/Hilbert.h`); each is forced
on the setup thread by `prepare()` or at construction, so the guard is already
resolved when the audio thread arrives.

That list is deliberately a list of named things rather than a sentence about
everything else. An unqualified negative claim over ninety headers is one no
reader can check and one commit can silently falsify, so this page does not make
one: what is claimed here is backed by `tools/verify_threading_doc.py`, which
enumerates the audio-path multi-word readers positively -- every conforming one
carries the `bounded seqlock read` marker -- and fails if a seqlock reader
exists that is neither marked bounded nor listed there as a readout.

## Verifying a change

CI runs ThreadSanitizer over the concurrency cases on every push. TSan is the
only oracle here that understands C++11 atomics: `valgrind --tool=drd` and
Helgrind model POSIX happens-before and are blind to relaxed atomics compiled
to plain moves, so they can prove the absence of plain-word conflicts and
nothing more. A hand-off that no concurrent test exercises is unverified no
matter how green the run looks, so a new hand-off ships with a concurrent case
that fails against the unfixed code.
