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
     exactly the word types it does assert, in both directions. The headers the
     page names as pinning the property locally must be exactly the headers
     that carry that static_assert. The page must not promise a compile-time
     guarantee it does not have, and its "most do not" must stay true of the
     headers whose atomic word type is a template parameter.
  E  the seqlock reader snippet must be semantically equal to Core/Biquad.h's.
"""

import re
import subprocess
import sys

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
    """True if `name` is used as a declared/defined entity in `path`."""
    body = CODE.get(path, "")
    if re.search(r"\b" + re.escape(name) + r"\s*\(", body):
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
pathish = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*(?:/[A-Za-z_0-9.]+)+$")
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

print("== tier E: seqlock reader snippet vs Core/Biquad.h ==")
reader = [b for b in fenced if "dirty.exchange" in b]
if not reader:
    failures.append("E: reader snippet not found in the page")
else:
    snippet = reader[0]
    if re.search(r"continue\s*;", snippet):
        failures.append("E: snippet still uses `continue` inside a do-while "
                        "(jumps to the condition, can accept a torn read)")
    else:
        print("  no `continue` inside the do-while  OK")
    cond = re.search(r"\}\s*while\s*\((.*?)\);", snippet)
    biq = open("Core/Biquad.h", encoding="utf-8").read()
    ref = re.search(r"\}\s*while\s*\(\(s0 & 1u\) != 0u \|\| s0 != s1\);", biq)
    print("  page condition : {}".format(cond.group(1) if cond else "<none>"))
    print("  Core/Biquad.h has the reference condition: {}".format(bool(ref)))
    if not cond or "& 1" not in cond.group(1) or "!=" not in cond.group(1):
        failures.append("E: the loop condition does not test oddness AND equality")
    if not ref:
        failures.append("E: Core/Biquad.h no longer carries the reference condition")

print("")
if failures:
    print("FAILURES ({}):".format(len(failures)))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("threading: every identifier, path, quantifier and snippet in the page "
      "checks out against the tree.")
sys.exit(0)
