# Threading model

Every DSPark component that can be touched by more than one thread follows the
same contract. It is written down once here; each header's `Threading:` block
then only has to say which of its own methods belongs where.

## The two threads

DSPark assumes **one control thread and one audio thread** -- single producer,
single consumer.

| Role | What runs there |
|---|---|
| **Setup** | `prepare()`, `setState()`, construction, moves, and -- unless a header says otherwise -- `reset()`. Never concurrent with processing. |
| **Audio** | `processBlock()` / `processSample()` and anything else that runs inside the callback. Exactly one stream owner per object. |
| **Control** | Parameter setters. **One** non-audio writer. |
| **Any thread** | Lock-free readouts: `get*()` methods that return an already-published value. These may run concurrently with processing. |

Two things fall outside the contract, and a component that needs them says so
in its own header:

- **More than one control writer.** Route the extra producers through
  `Core/SpscQueue.h` (one queue per producer) or design for it explicitly.
- **`prepare()` or `reset()` concurrent with processing.** Setup calls
  reallocate; nothing may be running in the callback while they do.

Thread-safety claims in DSPark are always per method. There is no blanket
"this class is thread-safe" anywhere in the framework, because there is no
useful version of that sentence.

## Which words must be atomic

Any word that both threads can reach concurrently is `std::atomic`, or it
travels through `Core/SpscQueue.h`. No plain scalar, struct or array is read on
one thread while another writes it -- including inside a seqlock's critical
section -- and nothing is read across threads *by reference*, which would let
the reader observe the writer's later stores.

Every atomic type on an audio path is verified lock-free at compile time:

```cpp
static_assert(std::atomic<T>::is_always_lock_free,
              "audio-thread stores must not lock");
```

A type that is not lock-free on some target fails the build there rather than
quietly taking a mutex inside the callback.

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
    s1 = seq.load(std::memory_order_acquire);
    if (s1 & 1) continue;                          // writer mid-update
    // relaxed loads of every word into a THREAD-PRIVATE plain copy
    std::atomic_thread_fence(std::memory_order_acquire);
    s2 = seq.load(std::memory_order_relaxed);
} while (s1 != s2);
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

DSPark's answer is that no pointer into shared storage escapes a component.
Where a getter returns a buffer -- `SpectrumAnalyzer::getMagnitudes()`, for
instance -- it copies the published data into reader-private storage while it
still owns the source, and returns a pointer to that copy. The audio thread
cannot reach the copy, so the caller may hold the pointer for as long as it
likes without another thread writing underneath it.

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
