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
it at all. Two headers pin it themselves -- `Analysis/SpectrumAnalyzer.h` and
`Effects/AutoGain.h` -- and most of the headers that declare such an atomic do
not. A new component with an atomic word on the audio path should do the same:

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

Writer, on the control thread only:

```cpp
seq.fetch_add(1, std::memory_order_acq_rel);      // -> odd: write in progress
std::atomic_thread_fence(std::memory_order_release);
// relaxed stores of every data word
seq.fetch_add(1, std::memory_order_release);      // -> even: complete
dirty.store(true, std::memory_order_release);
```

Reader, on the audio thread:

```cpp
if (!dirty.exchange(false, std::memory_order_acquire)) return;  // nothing new
do {
    s0 = seq.load(std::memory_order_acquire);
    // relaxed loads of every word into a THREAD-PRIVATE plain copy
    std::atomic_thread_fence(std::memory_order_acquire);
    s1 = seq.load(std::memory_order_relaxed);
} while ((s0 & 1u) != 0u || s0 != s1);   // odd = writer was mid-update
```

Both fences are mandatory, not decoration. Without them the relaxed data
accesses are not ordered against the counter, and a reader can observe a
mid-update word while still seeing an even counter -- the classic seqlock
mistake. The pairing is the one [atomics.fences]/2 defines; the analysis is
Boehm, *"Can Seqlocks Get Along With Programming Language Memory Models?"*,
MSPC 2012.

Note what the reader does after accepting: it works from a **private plain
copy**, never from the shared words. The per-sample loop is then ordinary
non-atomic code and runs at the same speed it would with no hand-off at all.

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
the framework does not contain one.

## Verifying a change

CI runs ThreadSanitizer over the concurrency cases on every push. TSan is the
only oracle here that understands C++11 atomics: `valgrind --tool=drd` and
Helgrind model POSIX happens-before and are blind to relaxed atomics compiled
to plain moves, so they can prove the absence of plain-word conflicts and
nothing more. A hand-off that no concurrent test exercises is unverified no
matter how green the run looks, so a new hand-off ships with a concurrent case
that fails against the unfixed code.
