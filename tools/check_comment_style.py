#!/usr/bin/env python3
"""Comment style check: comments and documentation must be self-contained.

A reader of this repository has the repository and nothing else. A comment that
cites a reference code -- an issue number, a ticket tag, a planning or design
document id -- sends that reader after something they cannot open, and the
reason the code is the way it is goes with it.

So the codes are rejected and the reasoning is written out instead:

    // Guard the recursion (T-412).            -> rejected
    // Guard the recursion: one non-finite     -> fine
    // sample would otherwise persist forever
    // in the feedback path.

Referring to a commit in this repository is fine; a reader can resolve it.

Run from the repository root:

    python3 tools/check_comment_style.py

Exits non-zero and lists every offending line if any tracked source or
documentation file carries one of the patterns below.
"""

import re
import subprocess
import sys

# Reference-code shapes that do not resolve inside this repository.
PATTERNS = [
    r"\bM-\d{3}[A-Za-z]?\b",
    r"\bM-R\d{2}\b",
    r"\bD-M\d{3}[A-Za-z]?-[A-Za-z]?\d+[a-z]?\b",
    r"\bADR-\d{3}\b",
    r"\bRF-\d{3}\b",
    r"\bRNF-\d{3}\b",
    r"\bF-\d{3}\b",
    r"\bUC-\d{3}\b",
    r"\bAC-\d{3}[A-Za-z]?-\d+\b",
    r"\bAG-\d{1,2}\b",
    r"\bD-\d\b",
    r"\baudit_ag\d+\b",
]

# Files the rule applies to: what we write, in the languages we write it in.
INCLUDE_SUFFIXES = (
    ".h", ".hpp", ".inl", ".c", ".cc", ".cpp",
    ".md", ".py", ".cmake", ".yml", ".yaml",
)
INCLUDE_NAMES = ("CMakeLists.txt",)

# Third-party code is vendored verbatim and is not ours to restyle.
EXCLUDE_PREFIXES = (
    "DSParkLab/vendor/",
    "plugin/clap/clap/",
    "plugin/vst3/vst3_c_api.h",
    "plugin/webview/webview/",
)

# This file quotes the shapes it looks for, so it exempts itself.
SELF = "tools/check_comment_style.py"


def tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout
    for path in out.split("\0"):
        if not path or path == SELF:
            continue
        if path.startswith(EXCLUDE_PREFIXES):
            continue
        if path.endswith(INCLUDE_SUFFIXES) or path.rsplit("/", 1)[-1] in INCLUDE_NAMES:
            yield path


def main():
    regex = re.compile("|".join(PATTERNS))
    hits = []
    for path in tracked_files():
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, 1):
                    for match in regex.finditer(line):
                        hits.append((path, number, match.group(0), line.rstrip()))
        except OSError as error:
            print("could not read {}: {}".format(path, error), file=sys.stderr)
            return 2

    if not hits:
        print("comment style: no unresolvable reference codes found")
        return 0

    for path, number, token, line in hits:
        print("{}:{}: {}".format(path, number, token))
        print("    {}".format(line.strip()))
    print("")
    print("{} unresolvable reference code(s) found.".format(len(hits)))
    print("Comments must stand on their own: write out the reason instead of")
    print("citing a code that only resolves somewhere else.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
