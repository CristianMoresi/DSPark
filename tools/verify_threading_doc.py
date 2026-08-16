#!/usr/bin/env python3
"""Check docs/threading.md against the code it describes.

That page is the single statement of the threading contract, and nine headers
plus CONTRIBUTING.md send the reader to it. A page nobody checks decays into a
page that is wrong in the places nobody rereads, which is worse than no page at
all, because it is trusted. Everything the page asserts that a machine can
check, this checks.

Run from the repository root; exits non-zero and lists every failure.

    python3 tools/verify_threading_doc.py

Tiers
  A  every backticked `name()` must be DECLARED in a tracked source file (in
     non-comment context), or be a std:: / language name.
  B  paragraph association: every `name()` in a paragraph must be declared in
     one of the tracked headers that same paragraph names. A paragraph that
     names several headers is held to all of them together, not skipped.
  C  every backticked path must be a tracked file.
  D  quantifier claims. The word types the page says the suite asserts must be
     exactly the word types it does assert, in both directions, AND the numeral
     the sentence writes in front of them must be that many; likewise for the
     headers the page names as pinning the property locally, which must be
     exactly the headers that carry that static_assert, in the number the
     sentence claims. A count in prose and the list beside it can drift apart
     one at a time, and the prose is the half nobody rereads. The page must
     also not promise a compile-time guarantee it does not have, and its "most
     do not" must stay true of the headers whose atomic word type is a
     template parameter.
  E  the seqlock reader snippet must be semantically equal to Core/Biquad.h's,
     in the BOUNDED form: a counted loop, the oddness test before the copy, the
     acquire fence, the unchanged accept test, and a give-up that re-arms.
  F  every seqlock reader in the framework's headers, enumerated POSITIVELY and
     classified. An audio-thread reader carries the marker `bounded seqlock
     read` and is checked against the shape above; anything else must be named
     in READOUTS below with the reason it is allowed to be unbounded. A grep
     for loop syntax cannot do this job -- `} while (` matches the four readers
     that existed when this was written and would miss a `for (;;) { break; }`
     written next month -- so the tier keys on the accept guard every seqlock
     reader must contain, and requires each one to say which kind it is.
  G  public const-reference accessors that return member storage, enumerated
     POSITIVELY by declaration-aware tokenization. One source-wide lexical
     graph records global, namespace-fragment, class, function, lambda and
     ordinary-block scopes plus each type fact's declaration point and
     lifetime. Explicit/trailing returns, const-reference aliases, multiline/
     parenthesized declarators, attributes, qualifiers, reopened/qualified
     namespace aliases, namespace-scope inline definitions, exact overload
     association, accessible inherited storage and direct member/subobject
     returns are covered. One immutable qualified result retains the selected
     first declaration, canonical target, remaining components with exact
     coordinates and class-head template-id presence, access, exact use point
     and alias provenance; a selected unsupported fact is terminal. Selected
     aliases and every recursive inheritance consumer share one 64-step depth
     bound plus cycle guards; a 65th alias or base edge is fail-closed while
     ordinary shorter chains retain their result.
     Qualified class joins use that same resolver through a lightweight
     declaration-id index, followed by exactly one immutable graph
     materialization. Definition-owner privileges for private nested classes
     are purpose-bound and do not relax ordinary expression access.
     Roots qualified by a bounded owner or ancestor retain exact owner,
     qualifier and member-declaration identities across relative, absolute,
     namespace and visible class-alias spellings. Direct owner names and aliases
     precede one canonical injected base name, which precedes enclosing lexical
     lookup. Namespace- and type-alias owner components compose without losing
     canonical identity or exact overload association. Every class-key token is
     classified before graph insertion, so elaborated uses, template/enum
     components and attribute contents cannot become phantom declarations.
     Out-of-class leading returns use enclosing namespace/global lookup while
     parameters, trailing returns and direct bodies use the nearest exact
     member owner. Inline, out-of-class, body, local and accessible inherited
     elaborated uses retain their existing identity and token coordinates.
     Positive subobjects admit dot/index suffixes and literal std::get forms,
     while custom get calls and post-root pointer arrows remain ordinary
     negatives. Relative std provenance observes class, union, alias,
     using-declaration and namespace facts at the exact use point and block
     lifetime; its graph-backed consumer accepts only the central qualified
     result, with no inherited-type veto or fallback. The graphless compatibility
     path retains its namespace-alias check, and absolute ::std bypasses those
     shadows. Complete-class lookup is confined to genuine member contexts.
     Local classes never become census owners. Every return retains exact
     identity and source position; singular
     identity is projected only for a uniform complete collection. Nested
     lambda and local-class bodies own their returns and traversal resumes after
     them, while ordinary branch/loop/switch blocks remain in the outer method.
     Classic/range-for, if and switch data names use graph-owned declaration
     points and recursively exact braced or unbraced controlled-statement ends.
     Parameters, block locals and every comma-separated control declarator are
     facts in the same graph. Declaration conditions and valid auto structured
     bindings are distinguished from subscript, logical, conditional and
     assignment expressions, which introduce no data-name fact. A syntactically
     unambiguous local declaration with an unresolved qualified or template-id
     type remains a conservative fail-closed shadow fact and can never create a
     member-storage positive.
     Ordinary and fail-closed recognizers share a backward declarator-name
     cursor that crosses only complete adjacent post-name attribute groups.
     Value, temporary, shadowing parameter and shadowing local returns are
     excluded. Explicit `this->` remains member-bound. Out-of-class definitions
     match a unique declaration
     by parameter types, member cv/ref qualifiers and return type/category;
     ordinary parameter names and declaration-only defaults are ignored. A
     C++ function-type identity normalizes top-level parameter cv, equivalent
     cv spelling and outer array-to-pointer adjustment while preserving
     pointee/referred cv and nested array extents. Repeated cv introduced by
     aliases is idempotent at its own type level. Namespace and visible base-
     type aliases resolve by lexical direct/chained/nested identity without
     sibling leakage. Each use and alias RHS observes its own declaration
     point; equivalent semantic redeclarations coalesce, while conflicting,
     cyclic or over-depth targets fail closed. Nested-template class joins
     retain template-id metadata without instantiating template owners. A
     recognizable unsupported const-reference definition, exact public
     class-template target or direct/subobject return
     through an unresolved base emits a production-fatal diagnostic; value,
     member-call, free-object, private-template and qualified namespace free-
     function controls do not. A processing-thread-only readout carries the
     marker `stream-owner reference readout` in both its method documentation
     and the class's Threading block, and names its separate atomic publication.
     An owner-managed offline view carries the distinct marker `owner-thread
     reference view` in both places and must belong to the exact reviewed-site
     set; it does not borrow or broaden the processing-thread policy. Existing
     unmarked sites are an explicit, counted legacy set until their owning
     audits document them; any new unclassified site is a failure, so the
     warning set cannot grow silently. The production policy runs its mutation
     matrix on every invocation.
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threading_reference_parser import (  # noqa: E402
    analyze_public_const_reference_accessors,
    cpp_identity_association_cases,
    fail_closed_association_cases,
    fail_closed_diagnostic_expectations,
    find_public_const_reference_accessors,
    mutation_matrix_cases,
    overload_association_cases,
    qualified_owner_ancestor_association_cases,
)

DOC = "docs/threading.md"
SWEEP_SOURCE = "tests/TestCoreFoundation.cpp"
SWEEP_CASE = r"DSPARK_TEST\(\w*swept_atomic_word_types_are_lock_free\)\s*\{(.*?)\n\}"
# The page's scope: the framework's own headers.
FRAMEWORK_DIRS = ("Core/", "Effects/", "Analysis/", "Music/", "IO/")

# Names that resolve to the standard library or the language, not to DSPark.
STD_OK = {
    "std::atomic", "std::size_t", "std::memory_order_relaxed",
    "std::memory_order_acquire", "std::memory_order_release",
    "std::memory_order_acq_rel", "std::atomic_thread_fence",
    "bool", "int", "unsigned", "float", "double",
}


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], check=True,
                         stdout=subprocess.PIPE, text=True).stdout
    return [p for p in out.split("\0") if p]


TRACKED = set(tracked())
SOURCES = [p for p in TRACKED
           if p.endswith((".h", ".hpp", ".inl", ".c", ".cc", ".cpp"))
           and not p.startswith(("DSParkLab/vendor/", "plugin/clap/clap/",
                                 "plugin/vst3/", "plugin/webview/webview/"))]

BLOCK = re.compile(r"/\*.*?\*/", re.S)
LINE = re.compile(r"//[^\n]*")


def code_of(path):
    """File contents with comments removed, so a mention is not a declaration."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    return LINE.sub("", BLOCK.sub("", text))


CODE = {p: code_of(p) for p in SOURCES}


def declared_in(name, path):
    """True if `name` is used as a declared/defined entity in `path`.

    A call through an object -- `dirty.exchange(...)`, `p->reset()` -- is not
    a declaration, and counting it as one would let the page attribute a
    method to a header that merely uses it. A qualified definition
    (`Biquad<T>::getCoeffs(...)`) is one, so only member access is excluded."""
    body = CODE.get(path, "")
    if re.search(r"(?<![.\w>])" + re.escape(name) + r"\s*\(", body):
        return True
    if re.search(r"\b(class|struct|using|enum\s+class)\s+" + re.escape(name)
                 + r"\b", body):
        return True
    return False


def declared_anywhere(name):
    return [p for p in SOURCES if declared_in(name, p)]


failures = []

with open(DOC, "r", encoding="utf-8") as fh:
    doc = fh.read()

# Strip fenced code blocks for token extraction; they are checked by tier E.
fenced = re.findall(r"```cpp\n(.*?)```", doc, re.S)
prose = re.sub(r"```.*?```", "\n\n", doc, flags=re.S)


def blocks(text):
    """Paragraphs, with each markdown list item and table row its own block:
    two adjacent bullets are two separate contexts, not one paragraph."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        current = []
        for line in para.split("\n"):
            if re.match(r"\s*(?:[-*+]\s|\|)", line):
                if current:
                    out.append("\n".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            out.append("\n".join(current))
    return out


paragraphs = blocks(prose)

tick = re.compile(r"`([^`]+)`")
callish = re.compile(r"^(?:([A-Za-z_][A-Za-z_0-9]*)::)?([A-Za-z_][A-Za-z_0-9]*)"
                     r"(?:<[^>]*>)?\(\)$")
# The character set git actually uses for the paths in this tree: hyphens are
# in every documentation file name here, and a leading dot is how the CI
# directory is spelled. A path this cannot parse is a path tier C cannot check.
pathish = re.compile(r"^[.A-Za-z_][A-Za-z_0-9.-]*(?:/[A-Za-z_0-9.-]+)+$")
# A bare type name, as the page writes one: no template arguments, no path.
typeish = re.compile(r"^(?:std::)?[a-z_][A-Za-z_0-9]*$")

print("== tier C: paths named in the page ==")
for tok in sorted(set(tick.findall(prose))):
    if pathish.match(tok):
        ok = tok in TRACKED
        print("  {:40s} tracked={}".format(tok, ok))
        if not ok:
            failures.append("C: path not tracked: " + tok)

print("== tier A: every `name()` the page names ==")
tokens = sorted({t for t in tick.findall(prose) if callish.match(t)})
for tok in tokens:
    m = callish.match(tok)
    qual, name = m.group(1), m.group(2)
    if tok.split("(")[0] in STD_OK or (qual or "").startswith("std"):
        print("  {:40s} std/language".format(tok))
        continue
    where = declared_anywhere(name)
    print("  {:40s} declared in {} file(s){}".format(
        tok, len(where), "" if where else "   <== NOT FOUND"))
    if not where:
        failures.append("A: identifier not declared anywhere: " + tok)
        continue
    if qual:
        hosts = [p for p in where if declared_in(qual, p)]
        if not hosts:
            failures.append("A: {}::{} -- no file declares both".format(qual, name))
        else:
            print("      qualified host(s): {}".format(", ".join(hosts[:3])))

print("== tier B: paragraph association (every header the paragraph names) ==")
checked_b = 0
for para in paragraphs:
    toks = tick.findall(para)
    headers = sorted({t for t in toks if pathish.match(t) and t in TRACKED
                      and t.endswith((".h", ".hpp"))})
    calls = sorted({t for t in toks if callish.match(t)})
    if not headers or not calls:
        continue
    for tok in calls:
        m = callish.match(tok)
        qual, name = m.group(1), m.group(2)
        if tok.split("(")[0] in STD_OK or (qual or "").startswith("std"):
            continue
        hosts = [h for h in headers if declared_in(name, h)]
        checked_b += 1
        print("  {:28s} in {:52s} -> {}{}".format(
            tok, "|".join(h.split("/")[-1] for h in headers), bool(hosts),
            "   <== MISMATCH" if not hosts else ""))
        if not hosts:
            failures.append(
                "B: {} named in a paragraph about {} but declared in none of them"
                .format(tok, ", ".join(headers)))
print("  {} call/paragraph pair(s) checked".format(checked_b))

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}


def check_numeral(text, phrase, subject, expected):
    """The count the page writes in front of `phrase` must be `expected`.

    Written out or in digits: this page writes its small numbers as words, and
    a word is what a stale count looks like."""
    found = re.search(r"(\w+)\s+" + phrase, text)
    if not found:
        failures.append("D: the page no longer states how many {}; the count "
                        "cannot be checked".format(subject))
        return
    written = found.group(1)
    value = int(written) if written.isdigit() else NUMBER_WORDS.get(
        written.lower())
    if value is None:
        failures.append("D: the page says '{} {}' and '{}' is not a number "
                        "this can read".format(written, subject, written))
    elif value != expected:
        failures.append("D: the page says '{} {}'; disk says {}".format(
            written, subject, expected))
    else:
        print("  the page's count of {} reads '{}' = {}  OK".format(
            subject, written, expected))


print("== tier D: quantifier claims ==")
template_atomic = sorted(p for p in SOURCES
                         if p.startswith(FRAMEWORK_DIRS) and p.endswith(".h")
                         and re.search(r"std::atomic<\s*(?:T|Real|SampleType)\s*>",
                                       CODE[p]))
assert_headers = sorted(p for p in SOURCES
                        if re.search(r"static_assert\(\s*std::atomic", CODE[p]))
print("  headers whose atomic word type is a template parameter: {}".format(
    len(template_atomic)))
print("  headers with static_assert(std::atomic...)             : {} {}".format(
    len(assert_headers), assert_headers))

# The page must not promise a compile-time guarantee across the board.
for phrase in ["verified lock-free at compile time", "fails the build there"]:
    if phrase in doc:
        failures.append("D: page still claims '{}' while only {} of {} headers "
                        "with a template word type assert it"
                        .format(phrase, len(assert_headers), len(template_atomic)))
    else:
        print("  page does not claim: '{}'  OK".format(phrase))

# The claim about which headers pin the property locally is read from the ONE
# paragraph that makes it, not from the page as a whole: a header can be named
# elsewhere on the page for unrelated reasons, and counting that as "named as
# pinning it" would let a new static_assert land unnoticed in any header the
# page already mentions.
PIN_ANCHOR = "pin it themselves"
pin_para = [p for p in paragraphs if PIN_ANCHOR in p]
if not pin_para:
    failures.append("D: the sentence naming the headers that pin the property "
                    "locally is no longer in the page; tier D cannot check it")
else:
    para = pin_para[0]
    named_assert = sorted({t for t in tick.findall(para)
                           if pathish.match(t) and t in TRACKED})
    print("  headers named by that paragraph as pinning it: {}".format(named_assert))
    if set(named_assert) != set(assert_headers):
        failures.append("D: the page names {} as pinning it locally; disk says {}"
                        .format(named_assert, assert_headers))
    check_numeral(para, r"headers?\s+pin it themselves",
                  "headers that pin it themselves", len(assert_headers))
    # "most of the headers that declare such an atomic do not" must stay true.
    if re.search(r"most of the headers that declare such an atomic do\s+not",
                 para):
        if len(assert_headers) * 2 >= len(template_atomic):
            failures.append("D: the page says most such headers do not pin it, "
                            "but {} of {} do".format(len(assert_headers),
                                                     len(template_atomic)))
        else:
            print("  'most do not' holds: {} of {} pin it locally  OK".format(
                len(assert_headers), len(template_atomic)))
    else:
        failures.append("D: the sentence bounding how many headers pin it "
                        "locally is no longer in the page; tier D cannot check it")


def bare(word):
    return word[5:] if word.startswith("std::") else word


# The word types the page says are asserted must be exactly the ones asserted.
sweep_src = open(SWEEP_SOURCE, encoding="utf-8").read()
block = re.search(SWEEP_CASE, sweep_src, re.S)
sweep_para = [p for p in paragraphs if SWEEP_SOURCE in p]
if not block:
    failures.append("D: the atomic word-type sweep test was not found")
elif not sweep_para:
    failures.append("D: no paragraph of the page names " + SWEEP_SOURCE)
else:
    swept = {bare(w) for w in re.findall(
        r"std::atomic<([^>]+)>::is_always_lock_free", block.group(1))}
    claimed = {bare(t) for t in tick.findall(sweep_para[0]) if typeish.match(t)}
    print("  suite asserts : {}".format(sorted(swept)))
    print("  page claims   : {}".format(sorted(claimed)))
    for word in sorted(swept - claimed):
        failures.append("D: word type asserted by the suite but not named by "
                        "the page: " + word)
    for word in sorted(claimed - swept):
        failures.append("D: word type named by the page but NOT asserted by "
                        "the suite: " + word)
    if swept == claimed:
        print("  the two sets are equal ({} word types)  OK".format(len(swept)))
    check_numeral(sweep_para[0], r"word types", "word types", len(swept))

MARKER_BOUNDED = "bounded seqlock read"
ATTEMPT_CONST = "kSeqlockMaxAttempts"
ATTEMPT_VALUE = "3"

FUNC_DEF = re.compile(
    r"^\s*(?:\[\[nodiscard\]\]\s*)?(?:static\s+|inline\s+|constexpr\s+|explicit\s+"
    r"|DSPARK_NOINLINE\s+)*"
    r"[A-Za-z_][\w:<>,&*\s]*?\b(\w+)\s*\([^;]*\)\s*(?:const\s*)?(?:noexcept\s*)?$")


def enclosing_function(lines, index):
    """(name, first_line, last_line) of the function containing `index`.

    The region returned starts at the function's doc comment, so a marker
    written above the signature counts, and ends at its closing brace, so a
    marker belonging to the NEXT function never does."""
    start = None
    for i in range(index, -1, -1):
        m = FUNC_DEF.match(lines[i].rstrip())
        if m and i + 1 < len(lines) and lines[i + 1].strip().startswith("{"):
            start, name = i, m.group(1)
            break
    if start is None:
        return None, index, index
    doc = start
    while doc > 0 and (lines[doc - 1].lstrip().startswith(("*", "/**", "//", "/*"))):
        doc -= 1
    depth, end = 0, start
    for i in range(start + 1, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if depth <= 0 and "}" in lines[i]:
            end = i
            break
    return name, doc, end


def reader_regions(path):
    """Every seqlock reader in `path`: (name, region text, line number).

    A reader is recognised by the accept guard every one of them must contain --
    the oddness test on the sequence counter. Loop syntax is deliberately not
    used: `} while (` matches the readers that existed when this was written and
    would miss a `for (;;) { ... break; }` written next month."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    out = []
    for idx, line in enumerate(lines):
        if re.search(r"&\s*1u", line):
            name, first, last = enclosing_function(lines, idx)
            out.append((name, "\n".join(lines[first:last + 1]), idx + 1, lines,
                        first, last))
    return out


print("== tier E: seqlock reader snippet vs Core/Biquad.h ==")

# The five properties a bounded reader must have. Each is (label, regex), and
# the ORDER of the first three inside the text is checked as well: an oddness
# test after the copy is the defect this shape exists to remove.
BOUNDED_SHAPE = [
    ("counted loop over kSeqlockMaxAttempts",
     r"for\s*\([^)]*<\s*kSeqlockMaxAttempts\s*;"),
    ("oddness test that skips the copy",
     r"\(\s*s0\s*&\s*1u\s*\)\s*!=\s*0u\s*\)\s*continue"),
    ("mandatory acquire fence",
     r"std::atomic_thread_fence\(std::memory_order_acquire\)"),
    ("accept test s0 == counter.load(relaxed)",
     r"s0\s*==\s*\w+\.load\(std::memory_order_relaxed\)"),
    ("give-up re-arms the dirty flag with release",
     r"store\(true,\s*std::memory_order_release\)"),
]


REARM = BOUNDED_SHAPE[-1]


def check_bounded_shape(text, where, require_rearm=True):
    """Every property present, and the oddness test before the fence.

    `require_rearm` is off for the split form: a `tryRead()` reports the give-up
    to its caller instead of re-arming a flag it does not own, and the caller's
    re-arm is checked at the call sites instead."""
    pos = {}
    for label, pattern in BOUNDED_SHAPE:
        if label == REARM[0] and not require_rearm:
            continue
        m = re.search(pattern, text)
        print("    {:48s} {}".format(label, "OK" if m else "<== MISSING"))
        if not m:
            failures.append("{}: {} has no {}".format(where[0], where[1], label))
        else:
            pos[label] = m.start()
    odd = pos.get("oddness test that skips the copy")
    fence = pos.get("mandatory acquire fence")
    if odd is not None and fence is not None:
        ordered = odd < fence
        print("    {:48s} {}".format("oddness test precedes the copy/fence",
                                     "OK" if ordered else "<== OUT OF ORDER"))
        if not ordered:
            failures.append("{}: {} tests oddness only AFTER copying"
                            .format(where[0], where[1]))
    if re.search(r"\bdo\s*\{", text):
        failures.append("{}: {} still uses a do/while (unbounded) reader"
                        .format(where[0], where[1]))


reader = [b for b in fenced if "dirty.exchange" in b]
if not reader:
    failures.append("E: reader snippet not found in the page")
else:
    print("  the page's reader snippet:")
    check_bounded_shape(reader[0], ("E", "the page's reader snippet"))

BIQUAD = "Core/Biquad.h"
biq_bounded = [(n, r) for n, r, _, _, _, _ in reader_regions(BIQUAD)
               if MARKER_BOUNDED in r]
if not biq_bounded:
    failures.append("E: Core/Biquad.h no longer carries a reader marked '{}'"
                    .format(MARKER_BOUNDED))
else:
    name, region = biq_bounded[0]
    print("  the reference implementation, Core/Biquad.h {}():".format(name))
    check_bounded_shape(region, ("E", "Core/Biquad.h"))

print("== tier F: audio-path seqlock readers, enumerated positively ==")

# Seqlock readers that are allowed to be unbounded, each named with the reason.
# The direction rule is what makes them safe: their reader is the control or
# GUI thread, so nothing real-time is waiting. Adding a file here does NOT
# exempt the rest of it -- entries are (file, function).
READOUTS = {
    ("Effects/Equalizer.h", "read"):
        "control/GUI readout (getBandConfig, getMagnitudeForFrequencyArray); "
        "no previously adopted copy to keep, not real-time",
    ("Effects/DynamicEQ.h", "read"):
        "control/GUI readout (getState); no previously adopted copy to keep, "
        "not real-time",
    ("Analysis/SpectrumAnalyzer.h", "snapshotFrame"):
        "control/GUI readout (getMagnitudesDb, getPeakHoldDb); the audio thread "
        "is the WRITER here, so this reader is not on the audio path. It is "
        "bounded anyway, by kSnapshotAttempts, with its own previous-snapshot "
        "fallback",
    ("DSParkLab/AudioEngine.h", "getWaveform"):
        "interface-thread readout of the waveform snapshot; the audio thread is "
        "the WRITER here, so this reader is not on the audio path",
}

# The seqlock population is deliberately WIDER than the page's scope. The page
# documents the framework's headers; this tier exists so that no seqlock reader
# anywhere in the repository can be added without being classified, and a
# reader in the demo application is a reader all the same. Only this population
# is widened - every claim checked against the page stays framework-scoped.
SEQLOCK_DIRS = FRAMEWORK_DIRS + ("DSParkLab/",)
seq_headers = sorted(p for p in SOURCES
                     if p.startswith(SEQLOCK_DIRS) and p.endswith(".h")
                     and "atomic_thread_fence" in CODE[p])
print("  headers with a cross-thread fence (the seqlock population): {}".format(
    len(seq_headers)))

bounded_sites, readout_sites = [], []
for path in seq_headers:
    for name, region, idx, lines, first, last in reader_regions(path):
        idx -= 1
        marked = MARKER_BOUNDED in region
        listed = (path, name) in READOUTS
        label = "{}:{} {}()".format(path, idx + 1, name)
        if marked:
            bounded_sites.append(label)
            print("  BOUNDED  {}".format(label))
            selfrearm = re.search(REARM[1], region) is not None
            check_bounded_shape(region, ("F", label), require_rearm=selfrearm)
            if ATTEMPT_CONST not in region:
                failures.append("F: {} is marked bounded but does not reference {}"
                                .format(label, ATTEMPT_CONST))
            if not selfrearm:
                # Split form. The reader hands the give-up back as `false`, so
                # the re-arm lives at every call site and that is where a
                # forgotten one would silently drop a parameter update.
                sites = [i for i, ln in enumerate(lines)
                         if "." + name + "(" in ln and not (first <= i <= last)]
                print("    split form: {} call site(s) must re-arm".format(len(sites)))
                if not sites:
                    failures.append("F: {} never re-arms and is never called; it "
                                    "cannot be the audio-path reader it claims to be"
                                    .format(label))
                for i in sites:
                    window = "\n".join(lines[i:i + 15])
                    ok = re.search(REARM[1], window) is not None
                    print("      {}:{} {}".format(path, i + 1, "re-arms  OK" if ok
                                                  else "<== DOES NOT RE-ARM"))
                    if not ok:
                        failures.append(
                            "F: {}:{} calls {}() and does not re-arm its dirty flag "
                            "with release on the give-up path; the update would be "
                            "dropped, not deferred".format(path, i + 1, name))
        elif listed:
            readout_sites.append(label)
            print("  READOUT  {}  -- {}".format(label, READOUTS[(path, name)]))
        else:
            print("  {}   <== UNCLASSIFIED".format(label))
            failures.append(
                "F: {} is a seqlock reader that is neither marked '{}' nor listed "
                "as a readout. Bound it, or list it with its reason."
                .format(label, MARKER_BOUNDED))

print("  {} bounded audio-path reader(s), {} listed readout(s)".format(
    len(bounded_sites), len(readout_sites)))
if not bounded_sites:
    failures.append("F: no bounded audio-path reader found at all; the marker "
                    "or this tier has drifted from the code")

# The attempt bound is one constant, declared per site with one value.
decls = []
for path in seq_headers:
    for m in re.finditer(r"static constexpr int " + ATTEMPT_CONST + r"\s*=\s*(\w+)\s*;",
                         CODE[path]):
        decls.append((path, m.group(1)))
print("  {} declaration(s) of {}: {}".format(len(decls), ATTEMPT_CONST,
                                             sorted(set(v for _, v in decls))))
for path, value in decls:
    if value != ATTEMPT_VALUE:
        failures.append("F: {} declares {} = {}; every site must agree on {}"
                        .format(path, ATTEMPT_CONST, value, ATTEMPT_VALUE))
for label in bounded_sites:
    host = label.split(":")[0]
    if host not in [p for p, _ in decls]:
        failures.append("F: {} is a bounded reader but its header declares no {}"
                        .format(label, ATTEMPT_CONST))

# The page has to carry the marker too, or nothing sends a reader from the
# contract to the check that enforces it.
if MARKER_BOUNDED not in doc:
    failures.append("F: the page never writes the marker '{}' that this tier "
                    "enumerates by".format(MARKER_BOUNDED))
else:
    print("  the page names the marker '{}'  OK".format(MARKER_BOUNDED))


# A public method that returns a const reference to one of its object's data
# members exposes storage whose later mutation the caller can observe. A
# dependency-free tokenizer/parser resolves the declaration and direct return
# semantics instead of matching one spelling or today's two conforming names.
# Every candidate is printed for review.
MARKER_STREAM_OWNER_REF = "stream-owner reference readout"
MARKER_OWNER_THREAD_REF = "owner-thread reference view"

# These sites pre-date this tier and belong to their component audits. Keeping
# the set explicit makes the gate useful immediately: it exits green with a
# counted warning for this fixed debt, while any NEW unmarked site fails. An
# owning audit removes an entry only after adding the marker and the complete
# per-method contract.
LEGACY_REFERENCE_ACCESSORS = {
    ("Analysis/BeatTracker.h", "getOnsetDetector"),
    ("Core/Biquad.h", "getCoeffs"),
    ("Core/ProcessorChain.h", "get"),
    ("Core/StateBlob.h", "entries"),
    ("Effects/Equalizer.h", "getBandFilter"),
    ("Effects/MultibandCompressor.h", "getBandCompressor"),
}

# A marked reference is admitted only when the same class exposes the result a
# foreign thread legitimately needs through an atomic publication. The map is
# intentionally exact: a marker cannot be pasted onto an arbitrary reference
# accessor and turn it into a conforming one.
STREAM_OWNER_FOREIGN_READOUT = {
    ("Music/ChordDetector.h", "getChroma"): "getChord",
    ("Music/KeyDetector.h", "chroma"): "getKey",
}

# Offline document views have a different contract from processing-thread
# readouts: callers retain no cross-thread publication guarantee and must avoid
# every concurrent mutable access. Keep this category exact so its marker
# cannot be pasted onto an arbitrary reference accessor as an exemption.
OWNER_THREAD_REFERENCE_VIEWS = {
    ("IO/MidiFile.h", "tracks"),
}
EXPECTED_OWNER_THREAD_REFERENCE_VIEWS = {
    ("IO/MidiFile.h", "tracks"),
}

def reference_accessors(path):
    """Public const-reference functions plus fail-closed parser diagnostics."""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    accessors, diagnostics = analyze_public_const_reference_accessors(text)
    return accessors, diagnostics, text


def reference_binding_issues(label, item, source):
    """Validate the complete return collection and compatibility projection."""
    issues = []
    bindings = tuple(item.return_bindings)
    if not bindings:
        return [label + " exposes no return-binding records"]
    if any(binding.disposition != "bound" for binding in bindings):
        issues.append(label + " exposes a non-bound enumerated return")
    positions = tuple(binding.source_offset for binding in bindings)
    if (positions != tuple(sorted(positions))
            or len(set(positions)) != len(positions)):
        issues.append(label + " return positions are not unique source order")
    for binding in bindings:
        if (source[binding.source_offset:binding.source_offset + 6] != "return"
                or binding.source_line
                != source.count("\n", 0, binding.source_offset) + 1):
            issues.append(label + " has a detached return source position")
            break
    identities = tuple((
        tuple(binding.qualifier_target),
        tuple(binding.member_declaring_owner),
        binding.member_name,
    ) for binding in bindings)
    uniform = (
        all(binding.disposition == "bound" for binding in bindings)
        and all(identity == identities[0] for identity in identities)
    )
    projection = identities[0] if uniform else ((), (), "")
    observed = (
        tuple(item.qualifier_target),
        tuple(item.member_declaring_owner),
        item.member_name,
    )
    if item.binding_identity_is_uniform is not uniform:
        issues.append(label + " has an incorrect uniform-binding flag")
    if observed != projection:
        issues.append(label + " has a non-atomic singular binding projection")
    return issues


def reference_diagnostic_issues(label, diagnostics):
    """Make unsupported associations fatal in the production Tier-G gate."""
    return [
        "G: {}:{}: {}".format(label, diagnostic.line, diagnostic.message)
        for diagnostic in diagnostics
    ]


def reference_policy_issues(label, region, class_documentation, foreign,
                            foreign_declared):
    """Apply the production marker/foreign-readout policy to one candidate."""
    issues = []
    if MARKER_STREAM_OWNER_REF not in region:
        issues.append(
            "{} has no '{}' marker in its method documentation"
            .format(label, MARKER_STREAM_OWNER_REF))
        return issues
    if foreign is None:
        issues.append(
            "{} carries the stream-owner marker but has no declared "
            "foreign-thread publication".format(label))
        return issues
    if not foreign_declared:
        issues.append(
            "{} names {}() as its foreign-thread publication, but the "
            "header does not declare it".format(label, foreign))
    threading = re.search(r"Threading:.*?\*/", class_documentation, re.S)
    if not threading or MARKER_STREAM_OWNER_REF not in threading.group(0):
        issues.append(
            "{} has the marker at the accessor but not in its class "
            "Threading block".format(label))
    return issues


def owner_thread_reference_policy_issues(label, region, class_documentation):
    """Require the offline owner-thread marker at the method and class."""
    issues = []
    if MARKER_OWNER_THREAD_REF not in region:
        issues.append(
            "{} has no '{}' marker in its method documentation"
            .format(label, MARKER_OWNER_THREAD_REF))
    threading = re.search(r"Threading:.*?\*/", class_documentation, re.S)
    if not threading or MARKER_OWNER_THREAD_REF not in threading.group(0):
        issues.append(
            "{} has no '{}' marker in its class Threading block"
            .format(label, MARKER_OWNER_THREAD_REF))
    return issues


print("== tier G: reference readouts and views, enumerated positively ==")
print("  qualified owner/ancestor invariant matrix:")
qualified_cases = qualified_owner_ancestor_association_cases(
    MARKER_STREAM_OWNER_REF)
if len(qualified_cases) != 128 or len({case.label for case in qualified_cases}) != 128:
    failures.append(
        "G: qualified owner/ancestor matrix must contain 128 unique cells")
for qualified_case in qualified_cases:
    found_matrix, parse_diagnostics = \
        analyze_public_const_reference_accessors(qualified_case.source)
    metadata = [] if len(found_matrix) != 1 else [
        found_matrix[0].line,
        found_matrix[0].access,
        found_matrix[0].accessor_owner,
        found_matrix[0].qualifier_target,
        found_matrix[0].member_declaring_owner,
        found_matrix[0].member_name,
    ]
    expected_metadata = [
        qualified_case.expected_line,
        "public",
        qualified_case.accessor_owner,
        qualified_case.qualifier_target,
        qualified_case.member_declaring_owner,
        qualified_case.member_name,
    ]
    if (parse_diagnostics or len(found_matrix) != 1
            or metadata != expected_metadata):
        failures.append(
            "G: qualified matrix {} expected exact owner/qualifier/member "
            "metadata {}, found {} with {} diagnostics".format(
                qualified_case.label, expected_metadata, metadata,
                len(parse_diagnostics)))
        continue
    item = found_matrix[0]
    binding_issues = reference_binding_issues(
        "qualified matrix {} state()".format(qualified_case.label),
        item, qualified_case.source)
    exact_binding = (
        len(item.return_bindings) == 1
        and item.binding_identity_is_uniform
        and (
            tuple(item.return_bindings[0].qualifier_target),
            tuple(item.return_bindings[0].member_declaring_owner),
            item.return_bindings[0].member_name,
        ) == (
            qualified_case.qualifier_target,
            qualified_case.member_declaring_owner,
            qualified_case.member_name,
        )
    )
    if binding_issues or not exact_binding:
        failures.extend("G: " + issue for issue in binding_issues)
        if not exact_binding:
            failures.append(
                "G: qualified matrix {} did not retain its exact return binding"
                .format(qualified_case.label))
        continue
    policy_issues = reference_policy_issues(
        "qualified matrix {} state()".format(qualified_case.label),
        item.documentation, item.class_documentation, "getPublished", True)
    if policy_issues:
        failures.extend("G: qualified matrix: " + issue
                        for issue in policy_issues)
        continue
    if qualified_case.source.count(qualified_case.deletion_anchor) != 1:
        failures.append(
            "G: qualified matrix {} has no unique method marker".format(
                qualified_case.label))
        continue
    deleted_source = qualified_case.source.replace(
        qualified_case.deletion_anchor,
        "/** ordinary matched declaration after mutation */", 1)
    deleted, deleted_diagnostics = \
        analyze_public_const_reference_accessors(deleted_source)
    deleted_issues = [] if len(deleted) != 1 else reference_policy_issues(
        "qualified deletion {} state()".format(qualified_case.label),
        deleted[0].documentation, deleted[0].class_documentation,
        "getPublished", True)
    deletion_isolated = (
        not deleted_diagnostics
        and len(deleted) == 1
        and deleted[0].line == qualified_case.expected_line
        and deleted[0].accessor_owner == qualified_case.accessor_owner
        and deleted[0].qualifier_target == qualified_case.qualifier_target
        and deleted[0].member_declaring_owner
            == qualified_case.member_declaring_owner
        and deleted[0].member_name == qualified_case.member_name
        and MARKER_STREAM_OWNER_REF not in deleted[0].documentation
        and deleted_source.count(MARKER_STREAM_OWNER_REF) >= 2
        and any("has no" in issue and "marker" in issue
                for issue in deleted_issues)
    )
    if not deletion_isolated:
        failures.append(
            "G: qualified matrix {} method-marker deletion was not isolated"
            .format(qualified_case.label))
print("    {} cells: exact binding metadata and isolated marker deletion"
      .format(len(qualified_cases)))

print("  exact out-of-class overload association matrix:")
for (matrix_label, source, expected_count, expected_marker,
     expected_line) in overload_association_cases(MARKER_STREAM_OWNER_REF):
    found_matrix = find_public_const_reference_accessors(source)
    actual_marker = bool(
        found_matrix
        and MARKER_STREAM_OWNER_REF in found_matrix[0].documentation)
    actual_line = found_matrix[0].line if found_matrix else None
    if (len(found_matrix) != expected_count
            or actual_marker != expected_marker
            or actual_line != expected_line):
        failures.append(
            "G: overload matrix {} expected count/marker/line {}/{}/{}, "
            "found {}/{}/{}".format(
                matrix_label, expected_count, int(expected_marker),
                expected_line, len(found_matrix), int(actual_marker),
                actual_line))
        continue
    policy_issues = reference_policy_issues(
        "overload matrix {} state()".format(matrix_label),
        found_matrix[0].documentation, found_matrix[0].class_documentation,
        "getPublished", True)
    has_missing_marker = any(
        "has no" in issue and "marker" in issue for issue in policy_issues)
    if has_missing_marker == expected_marker:
        failures.append(
            "G: overload matrix {} did not make only the matched method "
            "marker load-bearing".format(matrix_label))
        continue
    print("    {}: count={}, marker={}, line={}, policy={}".format(
        matrix_label, len(found_matrix), int(actual_marker), actual_line,
        "classified" if expected_marker else "missing-marker rejection"))

print("  C++ parameter identity and namespace-alias association matrix:")
for identity_case in cpp_identity_association_cases(MARKER_STREAM_OWNER_REF):
    found_matrix, parse_diagnostics = \
        analyze_public_const_reference_accessors(identity_case.source)
    if parse_diagnostics:
        failures.extend(reference_diagnostic_issues(
            "identity matrix " + identity_case.label, parse_diagnostics))
        continue
    actual_marker = bool(
        found_matrix
        and MARKER_STREAM_OWNER_REF in found_matrix[0].documentation)
    metadata_matches = (
        len(found_matrix) == identity_case.expected_count
        and found_matrix[0].name == "state"
        and actual_marker == identity_case.expected_marker
        and found_matrix[0].line == identity_case.expected_line
        and found_matrix[0].access == identity_case.expected_access
    )
    if not metadata_matches:
        actual = [(item.name,
                   MARKER_STREAM_OWNER_REF in item.documentation,
                   item.line, item.access) for item in found_matrix]
        failures.append(
            "G: identity matrix {} expected count/name/marker/line/access "
            "{}/state/{}/{}/{}, found {}".format(
                identity_case.label, identity_case.expected_count,
                int(identity_case.expected_marker),
                identity_case.expected_line, identity_case.expected_access,
                actual))
        continue
    binding_issues = reference_binding_issues(
        "identity matrix {} state()".format(identity_case.label),
        found_matrix[0], identity_case.source)
    if binding_issues:
        failures.extend("G: " + issue for issue in binding_issues)
        continue
    if (identity_case.deletion_anchor
            and found_matrix[0].documentation.strip()
            != identity_case.deletion_anchor):
        failures.append(
            "G: identity matrix {} did not preserve the matched declaration's "
            "exact documentation".format(identity_case.label))
        continue

    policy_issues = reference_policy_issues(
        "identity matrix {} state()".format(identity_case.label),
        found_matrix[0].documentation, found_matrix[0].class_documentation,
        "getPublished", True)
    missing_marker = any("has no" in issue and "marker" in issue
                         for issue in policy_issues)
    if missing_marker == identity_case.expected_marker:
        failures.append(
            "G: identity matrix {} did not bind policy to the exact matched "
            "declaration".format(identity_case.label))
        continue

    if identity_case.deletion_anchor:
        if identity_case.source.count(identity_case.deletion_anchor) != 1:
            failures.append(
                "G: identity matrix {} has no unique matched-marker anchor"
                .format(identity_case.label))
            continue
        deleted_source = identity_case.source.replace(
            identity_case.deletion_anchor,
            "/** ordinary matched declaration after mutation */", 1)
        deleted, deleted_diagnostics = \
            analyze_public_const_reference_accessors(deleted_source)
        if deleted_diagnostics:
            failures.extend(reference_diagnostic_issues(
                "identity deletion " + identity_case.label,
                deleted_diagnostics))
            continue
        deleted_metadata = (
            len(deleted) == 1 and deleted[0].name == "state"
            and deleted[0].line == identity_case.expected_line
            and deleted[0].access == identity_case.expected_access
            and MARKER_STREAM_OWNER_REF not in deleted[0].documentation)
        deleted_issues = [] if not deleted else reference_policy_issues(
            "identity deletion {} state()".format(identity_case.label),
            deleted[0].documentation, deleted[0].class_documentation,
            "getPublished", True)
        if (not deleted_metadata
                or MARKER_STREAM_OWNER_REF not in deleted_source
                or not any("has no" in issue and "marker" in issue
                           for issue in deleted_issues)):
            failures.append(
                "G: identity matrix {} matched-marker deletion did not retain "
                "discovery and fail policy while sibling/class markers remained"
                .format(identity_case.label))
            continue
    print("    {}: exact identity/metadata{}".format(
        identity_case.label,
        ", marker deletion fails policy"
        if identity_case.deletion_anchor else ", marked sibling not borrowed"))

print("  fail-closed unsupported-association diagnostics:")
diagnostic_expectations = fail_closed_diagnostic_expectations()
for (case_label, source, expected_count,
     expected_diagnostics) in fail_closed_association_cases(
         MARKER_STREAM_OWNER_REF):
    found_case, parse_diagnostics = \
        analyze_public_const_reference_accessors(source)
    production_rejections = reference_diagnostic_issues(
        "fail-closed matrix " + case_label, parse_diagnostics)
    if (len(found_case) != expected_count
            or len(parse_diagnostics) != expected_diagnostics):
        failures.append(
            "G: fail-closed matrix {} expected census/diagnostics {}/{}, "
            "found {}/{}".format(
                case_label, expected_count, expected_diagnostics,
                len(found_case), len(parse_diagnostics)))
        continue
    if expected_diagnostics and not production_rejections:
        failures.append(
            "G: fail-closed matrix {} produced no production-gate rejection"
            .format(case_label))
        continue
    if not expected_diagnostics and production_rejections:
        failures.extend(production_rejections)
        continue
    if case_label in diagnostic_expectations:
        expected_line, message_fragment = diagnostic_expectations[case_label]
        exact_diagnostic = (
            len(parse_diagnostics) == 1
            and parse_diagnostics[0].line == expected_line
            and message_fragment in parse_diagnostics[0].message
        )
        if not exact_diagnostic:
            failures.append(
                "G: fail-closed matrix {} expected diagnostic line/text "
                "{}/{!r}, found {}".format(
                    case_label, expected_line, message_fragment,
                    [(item.line, item.message)
                     for item in parse_diagnostics]))
            continue
    print("    {}: census={}, diagnostics={}, production={}".format(
        case_label, len(found_case), len(parse_diagnostics),
        "rejected" if production_rejections else "not-applicable"))

print("  declaration spelling and near-miss mutation matrix (production policy):")
accepted_matrix, negative_matrix = mutation_matrix_cases(MARKER_STREAM_OWNER_REF)
for matrix_label, source in accepted_matrix:
    found_matrix = find_public_const_reference_accessors(source)
    if len(found_matrix) != 1:
        failures.append("G: matrix {} was not enumerated exactly once"
                        .format(matrix_label))
        continue
    matrix_item = found_matrix[0]
    matrix_site = "matrix {} {}()".format(matrix_label, matrix_item.name)
    binding_issues = reference_binding_issues(
        matrix_site, matrix_item, source)
    if binding_issues:
        failures.extend("G: matrix binding: " + issue
                        for issue in binding_issues)
        continue
    issues = reference_policy_issues(
        matrix_site, matrix_item.documentation, matrix_item.class_documentation,
        "getPublished",
        re.search(r"\bgetPublished\s*\(",
                  matrix_item.class_definition) is not None)
    if issues:
        failures.extend("G: matrix correct-marker integration: " + issue
                        for issue in issues)
        continue
    deleted_source = source.replace(
        "/** {} */".format(MARKER_STREAM_OWNER_REF),
        "/** same-thread reference */", 1)
    deleted = find_public_const_reference_accessors(deleted_source)
    deleted_issues = [] if len(deleted) != 1 else reference_policy_issues(
        matrix_site, deleted[0].documentation,
        deleted[0].class_documentation, "getPublished", True)
    if len(deleted) != 1 or not any("has no" in issue and "marker" in issue
                                    for issue in deleted_issues):
        failures.append("G: matrix {} marker deletion did not fail production policy"
                        .format(matrix_label))
        continue
    print("    {}: discovered, classified, marker deletion fails".format(matrix_label))

# A marker on a different class in the same header cannot classify the owner.
alien_source = accepted_matrix[0][1].replace(
    "/** Threading: the foreign readout is atomic; this is a {}. */".format(
        MARKER_STREAM_OWNER_REF),
    "/** Threading: stream-owner only. */", 1)
alien_source = ("/** Threading: {}. */\nstruct AlienMarkerOwner {{}};\n".format(
    MARKER_STREAM_OWNER_REF) + alien_source)
alien_found = find_public_const_reference_accessors(alien_source)
alien_issues = [] if len(alien_found) != 1 else reference_policy_issues(
    "matrix foreign-class-marker explicitAccessor()",
    alien_found[0].documentation, alien_found[0].class_documentation,
    "getPublished",
    re.search(r"\bgetPublished\s*\(",
              alien_found[0].class_definition) is not None)
if len(alien_found) != 1 or not any("class Threading block" in issue
                                    for issue in alien_issues):
    failures.append("G: a Threading marker on another class satisfied the owner")
else:
    print("    foreign-class-marker: rejected (marker belongs to AlienMarkerOwner)")

def check_scoped_alias_case(case_label, source, expected_names):
    """Require every scoped accessor and make each method marker load-bearing."""
    found_case = find_public_const_reference_accessors(source)
    found_by_name = {item.name: item for item in found_case}
    issues = []
    for item in found_case:
        issues.extend(reference_policy_issues(
            "matrix {} {}()".format(case_label, item.name),
            item.documentation, item.class_documentation, "getPublished",
            re.search(r"\bgetPublished\s*\(", item.class_definition) is not None))
    if (len(found_case) != len(expected_names)
            or set(found_by_name) != set(expected_names) or issues):
        failures.append(
            "G: {} lexical scopes were not all independently enumerated and "
            "classified (expected {}; found {})".format(
                case_label, sorted(expected_names), sorted(found_by_name)))
        return
    print("    {}: {}/{} independently discovered and classified".format(
        case_label, len(found_case), len(expected_names)))

    # Delete each accessor marker separately.  Both declarations must remain
    # in the positive census and only the mutated one may fail policy.  This
    # makes every scoped discovery, rather than merely the aggregate count,
    # load-bearing in the production gate.
    for name in expected_names:
        marked_comment = "/** {} method:{} */".format(
            MARKER_STREAM_OWNER_REF, name)
        if source.count(marked_comment) != 1:
            failures.append("G: {} has no unique mutation anchor for {}()"
                            .format(case_label, name))
            continue
        mutated_source = source.replace(
            marked_comment, "/** same-thread reference method:{} */".format(name), 1)
        mutated = find_public_const_reference_accessors(mutated_source)
        mutated_by_name = {item.name: item for item in mutated}
        if (len(mutated) != len(expected_names)
                or set(mutated_by_name) != set(expected_names)):
            failures.append("G: {} marker deletion hid a scoped declaration: {}()"
                            .format(case_label, name))
            continue
        target = mutated_by_name[name]
        target_issues = reference_policy_issues(
            "matrix {} {}()".format(case_label, name), target.documentation,
            target.class_documentation, "getPublished", True)
        other_issues = []
        for other_name, other in mutated_by_name.items():
            if other_name != name:
                other_issues.extend(reference_policy_issues(
                    "matrix {} {}()".format(case_label, other_name),
                    other.documentation, other.class_documentation,
                    "getPublished", True))
        if (not any("has no" in issue and "marker" in issue
                    for issue in target_issues) or other_issues):
            failures.append("G: {} marker deletion was not isolated to {}()"
                            .format(case_label, name))
        else:
            print("      marker deletion for {}() fails only that policy"
                  .format(name))


# Identically named aliases in sibling classes have independent class scopes.
sibling_alias_source = """
/** Threading: the publication is atomic; this is a %s. */
struct IntAliasOwner {
    using Ref = const int&;
    /** %s method:intValue */
    Ref intValue() const { return value_; }
    int getPublished() const { return 0; }
private:
    int value_ = 0;
};
/** Threading: the publication is atomic; this is a %s. */
struct FloatAliasOwner {
    using Ref = const float&;
    /** %s method:floatValue */
    Ref floatValue() const { return value_; }
    int getPublished() const { return 0; }
private:
    float value_ = 0.0f;
};
""" % ((MARKER_STREAM_OWNER_REF,) * 4)
check_scoped_alias_case(
    "sibling-class-alias", sibling_alias_source,
    ("intValue", "floatValue"))

# A nested class's same-spelled alias must not erase its outer class's alias.
nested_alias_source = """
/** Threading: the publication is atomic; this is a %s. */
struct OuterAliasOwner {
    using Ref = const int&;
    /** Threading: the publication is atomic; this is a %s. */
    struct InnerAliasOwner {
        using Ref = const float&;
        /** %s method:inner */
        Ref inner() const { return inner_; }
        int getPublished() const { return 0; }
    private:
        float inner_ = 0.0f;
    };
    /** %s method:outer */
    Ref outer() const { return outer_; }
    int getPublished() const { return 0; }
private:
    int outer_ = 0;
};
""" % ((MARKER_STREAM_OWNER_REF,) * 4)
check_scoped_alias_case(
    "nested-class-alias-shadow", nested_alias_source, ("inner", "outer"))

# Namespace aliases are lexical: siblings do not poison each other.
sibling_namespace_source = """
namespace IntNamespace {
using Ref = const int&;
/** Threading: the publication is atomic; this is a %s. */
struct IntOwner {
    /** %s method:namespaceInt */
    Ref namespaceInt() const { return value_; }
    int getPublished() const { return 0; }
private:
    int value_ = 0;
};
}
namespace FloatNamespace {
using Ref = const float&;
/** Threading: the publication is atomic; this is a %s. */
struct FloatOwner {
    /** %s method:namespaceFloat */
    Ref namespaceFloat() const { return value_; }
    int getPublished() const { return 0; }
private:
    float value_ = 0.0f;
};
}
""" % ((MARKER_STREAM_OWNER_REF,) * 4)
check_scoped_alias_case(
    "sibling-namespace-alias", sibling_namespace_source,
    ("namespaceInt", "namespaceFloat"))

# A namespace-local alias shadows a global alias only inside that namespace.
global_namespace_shadow_source = """
using Ref = const int&;
/** Threading: the publication is atomic; this is a %s. */
struct GlobalAliasOwner {
    /** %s method:globalValue */
    Ref globalValue() const { return value_; }
    int getPublished() const { return 0; }
private:
    int value_ = 0;
};
namespace ShadowNamespace {
using Ref = const float&;
/** Threading: the publication is atomic; this is a %s. */
struct NamespaceAliasOwner {
    /** %s method:namespaceValue */
    Ref namespaceValue() const { return value_; }
    int getPublished() const { return 0; }
private:
    float value_ = 0.0f;
};
}
""" % ((MARKER_STREAM_OWNER_REF,) * 4)
check_scoped_alias_case(
    "global-namespace-alias-shadow", global_namespace_shadow_source,
    ("globalValue", "namespaceValue"))
for matrix_label, source in negative_matrix:
    if find_public_const_reference_accessors(source):
        failures.append("G: matrix near miss entered the census: " + matrix_label)
    else:
        print("    {}: excluded".format(matrix_label))

# Bind the same unqualified token before falling back to a member. Method and
# class markers are present in all three cases, so comments cannot manufacture
# a candidate when a parameter or visible local owns the spelling.
def binding_source(parameters="", local=""):
    return """
/** Threading: the publication is atomic; this is a %s. */
struct BindingOwner {
    using Storage = int;
    /** %s */
    const Storage& bindingAccessor(%s) const
    {
        %s
        return storage_;
    }
    int getPublished() const { return 0; }
private:
    Storage storage_ = 0;
};
""" % (MARKER_STREAM_OWNER_REF, MARKER_STREAM_OWNER_REF, parameters, local)


binding_cases = (
    ("member", binding_source()),
    ("parameter-shadow", binding_source("const Storage& storage_")),
    ("block-local-shadow", binding_source(
        local="static const Storage storage_ = 1;")),
)
binding_counts = [len(find_public_const_reference_accessors(source))
                  for _, source in binding_cases]
print("    member-binding-discriminator: expected=1/0/0 found={}".format(
    "/".join(map(str, binding_counts))))
if binding_counts != [1, 0, 0]:
    failures.append(
        "G: member/parameter/local binding discriminator expected 1/0/0, got "
        + "/".join(map(str, binding_counts)))
reference_sites = []
marked_sites = set()
owner_thread_sites = set()
if OWNER_THREAD_REFERENCE_VIEWS != EXPECTED_OWNER_THREAD_REFERENCE_VIEWS:
    failures.append(
        "G: owner-thread reference-view category must be exactly {} but is {}"
        .format(sorted(EXPECTED_OWNER_THREAD_REFERENCE_VIEWS),
                sorted(OWNER_THREAD_REFERENCE_VIEWS)))
for path in sorted(p for p in SOURCES
                   if p.startswith(FRAMEWORK_DIRS) and p.endswith(".h")):
    parsed_accessors, parse_diagnostics, reference_source = \
        reference_accessors(path)
    if parse_diagnostics:
        failures.extend(reference_diagnostic_issues(path, parse_diagnostics))
        for diagnostic in parse_diagnostics:
            print("  UNSUPPORTED {}:{}  {}".format(
                path, diagnostic.line, diagnostic.message))
    for item in parsed_accessors:
        name = item.name
        region = item.documentation
        class_documentation = item.class_documentation
        class_definition = item.class_definition
        line_no = item.line
        site = (path, name)
        reference_sites.append(site)
        label = "{}:{} {}()".format(path, line_no, name)
        failures.extend(
            "G: " + issue for issue in reference_binding_issues(
                label, item, reference_source))
        marked = MARKER_STREAM_OWNER_REF in region
        owner_marked = MARKER_OWNER_THREAD_REF in region
        if site in OWNER_THREAD_REFERENCE_VIEWS:
            owner_thread_sites.add(site)
            print("  OWNER   {}".format(label))
            if marked:
                failures.append(
                    "G: {} carries the processing-thread marker but belongs to "
                    "the owner-thread reference-view category".format(label))
            failures.extend(
                "G: " + issue for issue in owner_thread_reference_policy_issues(
                    label, region, class_documentation))
        elif owner_marked:
            print("  {}  <== UNCLASSIFIED OWNER-THREAD MARKER".format(label))
            failures.append(
                "G: {} carries the '{}' marker but is not in the exact "
                "owner-thread reference-view category".format(
                    label, MARKER_OWNER_THREAD_REF))
        elif marked:
            marked_sites.add(site)
            print("  MARKED  {}".format(label))
            foreign = STREAM_OWNER_FOREIGN_READOUT.get(site)
            failures.extend(
                "G: " + issue for issue in reference_policy_issues(
                    label, region, class_documentation, foreign,
                    foreign is not None and
                    re.search(r"\b{}\s*\(".format(re.escape(foreign)),
                              class_definition) is not None))
        elif site in LEGACY_REFERENCE_ACCESSORS:
            print("  LEGACY  {}  <== counted warning: owner audit must mark it"
                  .format(label))
        else:
            print("  {}  <== UNCLASSIFIED".format(label))
            failures.append(
                "G: {} is a public const-reference accessor returning member "
                "storage without the '{}' marker. Mark and document it, or add "
                "a reviewed legacy classification.".format(
                    label, MARKER_STREAM_OWNER_REF))

found = set(reference_sites)
for site in sorted(LEGACY_REFERENCE_ACCESSORS - found):
    failures.append("G: stale legacy reference-accessor entry: {} {}()"
                    .format(site[0], site[1]))
for site in sorted(set(STREAM_OWNER_FOREIGN_READOUT) - found):
    failures.append("G: required marked reference accessor not found: {} {}()"
                    .format(site[0], site[1]))
for site in sorted(OWNER_THREAD_REFERENCE_VIEWS - found):
    failures.append("G: required owner-thread reference view not found: {} {}()"
                    .format(site[0], site[1]))

marked_count = len(marked_sites)
owner_thread_count = len(owner_thread_sites)
legacy_count = sum(1 for site in found if site in LEGACY_REFERENCE_ACCESSORS)
print("  {} stream-owner marked site(s), {} owner-thread view(s), "
      "{} explicit legacy warning(s), {} total".format(
          marked_count, owner_thread_count, legacy_count, len(reference_sites)))

ref_paragraphs = [p for p in paragraphs
                  if "enumerates public const-reference accessors" in p]
if not ref_paragraphs:
    failures.append("G: the page has no paragraph stating the reference-accessor "
                    "population and legacy count")
else:
    ref_para = ref_paragraphs[0]
    for pattern, subject, expected in [
            (r"finds\s+(\w+)", "total reference accessors", len(reference_sites)),
            (r"the\s+(\w+)\s+stream-owner marked sites",
             "stream-owner reference accessors", marked_count),
            (r"(?:and\s+)?(?:the\s+)?(\w+)\s+owner-thread view",
             "owner-thread reference views", owner_thread_count),
            (r"and\s+(\w+)\s+explicit legacy warnings", "legacy reference accessors",
             legacy_count)]:
        match = re.search(pattern, ref_para)
        if not match:
            failures.append("G: the page no longer states its {} count"
                            .format(subject))
            continue
        word = match.group(1)
        value = int(word) if word.isdigit() else NUMBER_WORDS.get(word.lower())
        print("  page count {:28s}: '{}' -> {}".format(subject, word, value))
        if value != expected:
            failures.append("G: the page says {} {} but disk says {}"
                            .format(word, subject, expected))
if MARKER_STREAM_OWNER_REF not in doc:
    failures.append("G: the page never writes the marker '{}' that this tier "
                    "enumerates by".format(MARKER_STREAM_OWNER_REF))
else:
    print("  the page names the marker '{}'  OK".format(MARKER_STREAM_OWNER_REF))
if MARKER_OWNER_THREAD_REF not in doc:
    failures.append("G: the page never writes the marker '{}' that this tier "
                    "enumerates by".format(MARKER_OWNER_THREAD_REF))
else:
    print("  the page names the marker '{}'  OK".format(MARKER_OWNER_THREAD_REF))

print("")
if failures:
    print("FAILURES ({}):".format(len(failures)))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("threading: every identifier, path, quantifier, snippet and reference "
      "readout in the page checks out against the tree.")
sys.exit(0)
