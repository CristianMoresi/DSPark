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
the stream-owner case at the end of "Pointers returned across threads".)

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
it at all. Ten headers pin it themselves -- `Analysis/SpectrumAnalyzer.h`,
`Analysis/BeatTracker.h`, `Effects/AutoGain.h`, `Effects/Equalizer.h`,
`Effects/DynamicEQ.h`, `Effects/SpectralFreeze.h`,
`Effects/PitchCorrector.h`, `Effects/detail/PhaseVocoderEngine.h`,
`Effects/TimeStretch.h` and `Music/KeyDetector.h` -- and
most of the headers that declare such an atomic do not.
Five of the ten pin a word whose
type is a template parameter, which is the case the census cannot reach, and
seven pin a concrete width the census already covers --
`Analysis/SpectrumAnalyzer.h` and `Effects/PitchCorrector.h` do both. They pin
the concrete ones anyway, because a compile-time assertion at the declaration
is a stronger statement than a run-time one in another file and
it is the declaration that a later edit changes. A new component with an atomic
word on the audio path should do the same:

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
re-adopt the same thread's own write inside the hot path. After this origin
rule, every adoption the audio thread performs is cold by construction: only
the control thread can arm one.

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

### The other direction: a set the audio thread publishes

A seqlock moves a set from the control thread to the audio thread. A set that
travels the other way -- computed inside the callback, read by a GUI -- does not
need one, and should not have one, because the cheaper answer is available: pack
the set into a single word and it cannot tear at all. `Music/ChordDetector.h`
does this with its whole result, and `Analysis/BeatTracker.h` does it with the
pair a caller is most likely to misuse if it can straddle an update. Its
`getRunningTempoBpm()` and `getConfidence()` describe the same instant -- the
confidence is measured *at* that tempo -- so a caller that gates on the
confidence before trusting the tempo must never be handed one of them refreshed
without the other. Both unpack the same 64-bit load, and
`getTempoAndConfidence()` returns them together for callers that want both.

Two of its readouts stay independent single words, and what keeps them
consistent is stated here because an argument, not a mechanism, is doing the
work. `beatNow()` and `getLastBeatSample()` are both written in the same
processing call: the position first, the latch last. That order is not enough
on its own. Two relaxed loads of two distinct objects may be satisfied in
either order, and a weakly ordered machine -- the 64-bit ARM this library also
ships to emits plain loads and plain stores here, with no barrier -- is free to
give a reader the latch from one call and the position from the one *before*
it, which is a stale position a full beat period old. So the pair is ordered:
the latch is stored with release and `beatNow()` loads it with acquire, which
costs nothing on x86 and one instruction on ARM.

That ordering makes exactly ONE reader order safe, and it is the order to use:
test the latch with `beatNow()` first, and read `getLastBeatSample()` second.
Then the position is the one written in the same processing call that set the
latch, or in a later one -- never an earlier one.

The reverse order is not safe, and the reason has nothing to do with the memory
model. `getLastBeatSample()` and `beatNow()` are two separate loads, and the
writer is free to run between them: a reader that takes the position first and
tests the latch second holds a position from before that pause and a latch from
after it, so it is told a beat just happened and handed the position of an
earlier one. The release/acquire pairing cannot help, because the two values
the reader ends up with never came from the same call in the first place.

The cost is a full beat, and it is not capped at one. Measured against this
implementation with a writer on real-time-paced blocks and a pause between the
reader's two loads: in the position-first order 179 of 181 latched observations
carried a stale position, worst 1022.3 ms at 60 BPM (1.02 beat periods) and
666.5 ms at 180 BPM (2.00 beat periods) -- the staleness is bounded by how long
the reader takes between its own two loads, not by the beat period. In the
latch-first order the same probe reports 0 stale positions out of 183. A caller
that cannot
guarantee the two loads stay adjacent -- one that reads the position in a
message-thread timer and tests the latch elsewhere, say -- should not use these
two readouts as a pair at all; it should treat the latch alone as the event and
the position alone as a timestamp, or ask for a packed publication.

The same class's `setTempoRange()` packs its two indices for the opposite
reason: they travel control-to-audio, and half of one range combined with half
of the next is a range nobody asked for, which the audio thread would then
search. Packing is what lets that setter stay allocation-free and callable while
the stream runs.

`Effects/SpectralFreeze.h` packs its two requested controls -- the freeze
request and the phase mode -- into one `std::atomic<std::uint32_t>` for that
same control-to-audio reason, plus one of its own: the word is a state latch,
not an event queue. `setFrozen()` and `setPhaseMode()` each perform one relaxed
read-modify-write on the packed word and may be called from any thread; the
stream owner loads the word exactly once per completed all-channel STFT frame,
so requests landing between frame boundaries coalesce to the latest word and
are adopted together at the next frame. `isFrozen()` and `getPhaseMode()`
report the requested bits, deliberately not transition progress -- there is no
cross-thread active-state readout to catch mid-glide. `prepare()` and `reset()`
keep the usual setup ownership and preserve the requested word, like any other
parameter.

`Effects/PitchCorrector.h` packs its scale the same way, and for the same
reason: `setScale()` publishes the mask, the root and the pre-rotated
pitch-class set in one `std::atomic<std::uint32_t>`, so the audio thread can
never pair a fresh mask with a stale root. It is also the framework's clearest
case of a processor that OWNS other processors. Its detector and its shifter
are private members, and their control entry points -- the ones a normal owner
would call from a message thread -- are called only by the stream owner from
inside `processBlock()` and `reset()`. That is stronger than the single-writer
rule those members are documented for, not weaker: their control-to-audio
hand-offs become same-thread sequential code, and the only genuinely
cross-thread words left are this class's own three relaxed publications. A
component that composes others should either do this, or leave the inner
setters to one external control writer -- never both at once, because two
writers are exactly what the single-producer contract does not cover. The
inner `setSemitones()` and `setFormantPreserve()` calls named here belong to
`Effects/PitchShifter.h`.

`Analysis/LoopFinder.h` sits outside the audio-thread contract altogether: the
loop search is an offline operation. `find()` may allocate bounded scratch and
take real time on a long file, so it belongs on a worker or loading thread, one
caller owning the instance for the duration of the search; `renderLoop()` is
allocation-free and validates everything before its first write, but it is
offline all the same. Neither has cross-thread setters, because there is
nothing to publish: every argument arrives in the call.

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

**The stream-owner case.** A getter that returns a reference to state owned by
the *processing* thread is not a cross-thread readout. It is valid on that
thread only, between that thread's own processing calls; reading it on another
thread while processing continues is a data race. A conforming method carries
the exact marker `stream-owner reference readout` both at the accessor and in
its class's `Threading:` block, and the class provides a separate atomic
publication for what another thread legitimately needs. `Music/ChordDetector.h`
does that with `getChroma()` / `getChord()`, and `Music/KeyDetector.h` with
`chroma()` / `getKey()`.

**The owner-managed offline case.** A reference into an offline document does
not need an atomic publication when the owning object forbids concurrent
access. `IO/MidiFile.h`'s `tracks()` is the exact current case. It carries the
marker `owner-thread reference view` both at the method and in the class's
`Threading:` block. The reference is valid only while its `MidiFile` owner
remains alive and until the next non-const operation that can replace or mutate
the document. No thread may read the view while another thread accesses the
same instance mutably. This is owner-managed use, not atomic publication, and
it does not satisfy the processing-thread rule above: a processing component
must still provide a separate atomic publication for foreign-thread readout.

`tools/verify_threading_doc.py` enumerates public const-reference accessors
that return member storage positively. Its dependency-free C++ tokenizer and
declaration parser builds one source-wide graph of lexical scopes, declaration
points and lifetimes, then resolves explicit and trailing return types,
const-reference aliases, multiline and parenthesized function names, attributes,
member qualifiers, direct member/subobject returns, reopened and nested named
namespace identity, qualified type aliases, direct/chained/nested namespace
aliases, public declarations with namespace-scope inline definitions, accessible
inherited storage through literal bases or visible non-dependent `using` and
`typedef` base aliases, and lexical alias shadowing across global, namespace,
nested-class and sibling-class scopes. Base aliases may be direct or chained;
direct return roots may also spell
`[this->] class-id::member` before a supported field, balanced index or
literal `std::get<I>` / `::std::get<I>` subobject. Only dot fields and balanced
built-in indexes may follow a storage root; a later `->` crosses an indirection
and does not prove member-subobject ownership. That exclusion does not affect
the supported `this->member` and `this->class-id::member` object prefixes.
Unqualified, ADL-selected, namespace-aliased and custom qualified `get` calls
remain ordinary call negatives. A relative `std::get` is standard only when
`std` is unshadowed and denotes the global standard namespace at that exact
use point; a visible type alias, class, union, namespace, namespace alias or
using-declaration named `std` makes the call an ordinary 0/0 negative. A block
declaration begins at its declaration point and stops at that block's end. A
local class is a block type fact only: it never becomes a class-member identity
or a census owner. Namespace fragments share one canonical namespace identity,
but a fact in a later reopened fragment cannot retroactively shadow an earlier
use. A later namespace- or non-member-class declaration likewise does not
retroactively shadow an earlier use in the same fragment. The absolute
`::std::get` spelling is not affected by such local shadows, while an alias to
`::std` is still not one of the two admitted literal spellings. The
graph-backed relative-`std` decision is solely the central qualified result,
including its inherited-type selection; no consumer-specific inherited veto
or fallback follows it. The graphless compatibility branch retains only its
lexical namespace-alias check. The expression's `class-id` is resolved
independently at that exact use point and must be the accessor owner or one
uniquely reachable accessible ancestor. Qualified lookup retains the exact
accessor owner, qualifier target, data-member owner and member name, so ordinary
hiding, ambiguous base subobjects, inheritance access and private members cannot
be replaced by a matching basename. Relative, absolute, namespace-qualified,
namespace-alias-qualified and visible direct or chained class aliases are
supported. For an unqualified class-id in a member context, direct owner names
and aliases are considered first, followed by one canonical injected base-class
name, and only then enclosing lexical scopes. A direct but unsupported or
imported type fact is still decisive: it blocks injected-base and enclosing
lookup instead of allowing a lower-precedence declaration to win. Tokens inside
a bounded `[[...]]` attribute never introduce declaration-graph facts. Every
supported qualified lookup starts with one source-positioned selected
declaration and retains that declaration, its kind, canonical target, remaining
components, access and alias provenance as one result. Each class-head
component also retains its identifier coordinates and whether its source form
was a template-id; template arguments remain opaque. A selected alias is
followed only from its own right-hand-side position. An inaccessible, imported,
dependent, conflicting or otherwise unsupported selection is terminal; lookup
does not retry an outer alias, basename or later aggregate fact. Every recursive
declaration-graph walk shares a 64-step limit in addition to its cycle guard:
one selected alias or one inheritance edge consumes one step. Attempting a 65th
step is a stable fail-closed `unsupported` result. This bound applies to
selected aliases, inherited named types and aliases, injected base names,
owner/ancestor paths, member lookup and inherited-member aggregation; ordinary
chains within the limit retain their normal result.
Qualified class definitions are joined by declaration ID in a lightweight
structural phase that uses this same resolver. Only after those identities
stabilize is the immutable source-wide graph materialized, exactly once; the
structural index is not a second graph or a consumer-specific lookup path.
Namespace and type aliases retain their selected declaration and terminal class
provenance through that join. Nested-template class-head joins retain each
component's template-id bit, but that metadata never instantiates an accessor
owner: explicit or partial class-template owner forms keep their existing
fail-closed census behavior. A qualified class definition or out-of-class member
definition may name its own private nested owner, but that purpose-bound
association does not relax access for an ordinary type or return expression.
Every `class`, `struct` and `union` token is classified before the graph can
record it. Definitions and genuine standalone forward declarations introduce
type facts. Elaborated uses in signatures, data declarations and alignment
expressions resolve existing types without inventing nested declarations;
template parameters, the `class` component of an enum, and attribute contents
introduce none. A compiler-valid form outside this bounded grammar is retained
as unsupported instead of being guessed. The injected candidate must name one
accessible, non-virtual ancestry path;
repeated, virtual or inaccessible paths stay fail-closed. A namespace-alias
prefix is canonicalized before a terminal type alias is resolved, so the two
alias kinds compose without losing either use point or RHS declaration point.
Direct and chained type aliases may
also qualify an out-of-class definition; association still uses their canonical
original owner and the exact overload, never the alias basename. Class-member
alias access is checked from the
owner and its enclosing/ancestor context: public aliases are nameable,
protected inherited aliases are supported, and own or enclosing private aliases
remain available, while private/protected access that works only through
friendship stays fail-closed. Inline member-function bodies use complete-class
lookup for the owner and its enclosing classes, including later aliases, and
may use an accessible unambiguous alias inherited from a base. Function return
types and namespace/global lookup keep their ordinary point-of-declaration
behavior. In an out-of-class member definition, an explicit leading return
type is resolved in the enclosing namespace/global context before the
qualified declarator. Parameters, supported member suffixes, a trailing return
type and the body are resolved after that boundary in the exact member-owner
context. The parser segments those token ranges before canonicalizing types; it
never reparses the whole signature under one convenient scope. The nearest
matching member signature supplies the owner for its parameters, trailing
return and direct body. Elaborated uses there, in an inline member signature,
in a function-local class declaration, or through an accessible protected base
type retain the existing canonical identity and original token coordinates;
none creates a phantom class fact. A nested callable or local-class method
starts its own owner range.
Every alias lookup is evaluated at its exact lexical use point, while an alias
right-hand side and each link in a chain are evaluated at that alias's own
declaration point. A later declaration therefore cannot retroactively change a
base or function signature, and a same-header class identity retained by a type
alias cannot be reinterpreted by a nearer class or namespace alias at a later
use. Reopened namespace fragments share only the aliases owned by that
namespace, and an identically named sibling alias cannot leak into owner lookup.
Multiple visible declarations of one alias coalesce only when they resolve to
the same canonical type, class or namespace; conflicting or cyclic targets stay
unsupported. A namespace-scope definition is associated only when exactly one
public declaration has the same
function name, return type and reference category, ordinary parameter-type
sequence, member `const`/`volatile` qualification, and `&`/`&&` ref qualifier.
Parameter names and declaration-only default arguments are ignored, visible
type aliases are expanded, and ordinary parameter types use the C++ function-
type adjustments: top-level cv on a value parameter is ignored, including cv
on the outermost pointer, while pointee and referred-to cv remain significant;
prefix/suffix cv spellings are equivalent; repeated `const` or `volatile`
introduced by direct or chained alias expansion is idempotent at that exact
base or pointer level; and an outer array parameter becomes a pointer while
element cv and nested extents remain significant. Qualification never moves
across a pointer, reference or array-element boundary. An arity,
parameter-type, cv/ref or value-return sibling therefore cannot lend its
documentation, source line or public access to the definition; zero or multiple
exact matches are rejected conservatively.
Return roots are bound against parameters and visible block locals before
member lookup; `this->` explicitly selects the member. Every return in a method
is retained as an ordered record with its disposition, exact qualifier/member
identity and return-token offset and line, and every record must bind positively
for enumeration. Data names declared by classic `for`, range `for`, `if` and
`switch` headers have graph-owned declaration points and exact recursively
computed controlled-statement ends. Their lifetimes cover the complete braced
or unbraced controlled statement, including nested controls and an associated
`else`, and end before the following statement. Parameters, ordinary block
locals and control declarations are all facts in that same graph. Every
declarator in a comma-separated init statement is retained; an `if` or
`switch` may contribute names from both its init-statement and declaration
condition. A structured binding is admitted only after a valid declaration
prefix containing `auto`. A syntactically unambiguous local declaration whose
qualified or template-id type cannot be resolved within this bounded grammar
is retained as a conservative, fail-closed shadow fact. It may suppress a
member-storage positive, but it can never create one. Subscript, logical,
conditional and assignment expressions in the same positions never become
declarations. Ordinary compound, branch, loop and switch blocks remain part of
that method. A nested lambda or local-class method owns its own returns, so
those bodies are skipped and traversal resumes after them; their returns can
neither create nor contaminate an outer accessor. Equal returns are not
deduplicated. When all records carry one exact identity, the legacy singular
identity fields project that value and an
explicit uniform flag is true. When identities differ, the complete collection
remains authoritative, the flag is false and all three singular identity fields
are cleared together rather than selecting a branch by source order. An
unsupported owner/ancestor return
diagnoses even beside a bound return, while a bound return mixed only with a
deliberate negative (an unrelated class, call, namespace object, parameter or
other non-member) remains an ordinary 0/0 exclusion. Its production-policy
mutation matrix checks every spelling, overload association, scope and binding,
isolated marker deletion, and value/temporary/parameter/local near miss on every
run. It currently finds nine: the two stream-owner marked sites above, one
owner-thread view and six explicit legacy warnings whose component audits must
document them. A new unmarked site fails the check, so the warning set cannot
grow silently. `Core/Biquad.h`'s `getCoeffs()` illustrates the hazard:
it returns a reference straight into the active coefficient set, which the
processing thread rewrites in `applyPendingCoeffs()`, so a GUI thread reading
it during a promotion may observe a half-updated set. `Core/ProcessorChain.h`'s
`get()` has both mutable- and const-reference forms; either hands back the
sub-processor itself, so what may be done with the reference is that processor's
contract. The rule to carry away is unchanged: a getter that returns a reference
or raw pointer belongs to the processing thread unless its own documentation
identifies a stricter exact category. An offline owner-thread view still
forbids concurrent mutable access and promises no publication.

The parser's deliberate boundary is named class, struct and union type facts and
named namespaces present in the same header, with direct return expressions and
ordinary parameter or block-local declarations. Class and struct definitions
are the ordinary accessor owners; local types remain shadow facts rather than
owners. Base definitions and aliases must also be visible in that header.
Overload matching removes ordinary
top-level parameter names, so those names may differ and a default may appear
only on the declaration; the resulting structural type identities must agree
after alias expansion and the parameter adjustments above. Names nested inside
parenthesized declarators remain in the key, so different spellings are rejected
rather than guessed. Ordinary and fail-closed definition recognition share one
backward name cursor. It crosses only complete adjacent post-name `[[...]]`
groups without changing source coordinates and recognizes ordinary, qualified,
parenthesized, `operator[]` and `operator()` names. A recognizable public
const-reference definition that cannot be associated inside this boundary
emits a line-specific diagnostic,
and the production Tier-G gate fails on every such diagnostic. The matrix pins
that fail-closed path with compiler-valid nested function-pointer declarators,
including qualified parenthesized names and `operator[]` / `operator()`, and
with a qualified class-template owner or a member-like return through an
unresolved base. A template-id owner is recognized but is not reported as a
successful census entry because the parser does not instantiate templates; it
diagnoses only when one exact normalized declaration is public, so an unrelated
public overload cannot lend access to a private target. For an unresolved base,
direct qualified member and subobject returns such as `Base<T>::storage_`,
including the supported `std::get<I>(...)` wrapper, diagnose instead of being
silently omitted. The controls separately prove that value-return, member-call,
free-object, private, private-template and qualified namespace free-function
forms do not raise that diagnostic. Macro-generated declarations, dependent
bases, imported members and coroutine returns remain outside the positive
dependency-free subset. Returns inside a nested lambda or local class are
structurally owned by that callable and deliberately excluded from the outer
method; this is distinct from treating lambda declarations as accessor owners.
Function-block type aliases are tracked
positionally only as shadow facts: a member-like return through one diagnoses,
the alias stops shadowing at the end of its block, and it is never promoted into
the positive alias subset. A template-id qualifier, including balanced
template arguments on nested components, is likewise recognized only far
enough to diagnose a proven owner/ancestor case; an unrelated template class or
a call remains 0/0. Namespace-alias-qualified namespace objects are
canonicalized as namespaces and also remain 0/0. Friend-only access,
virtual-base ownership and repeated base graphs that cannot prove one
accessible subobject remain fail-closed rather than being guessed.
Those forms are not associated by declaration order or basename; a
recognizable public reference accessor must diagnose until the gate is extended
with a sound positive fixture for it.

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
