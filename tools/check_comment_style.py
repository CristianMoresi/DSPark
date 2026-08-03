#!/usr/bin/env python3
"""Comment style check: comments and documentation must be self-contained.

A reader of this repository has the repository and nothing else. A comment that
cites something only resolvable elsewhere -- an issue number, a ticket tag, a
planning document, a directory that exists only on the author's machine --
sends that reader after something they cannot open, and the reason the code is
the way it is goes with it.

Three rules, from the narrowest to the most general:

1. REFERENCE CODES are rejected outright, and the reasoning is written out
   instead:

       // Guard the recursion (T-412).            -> rejected
       // Guard the recursion: one non-finite     -> fine
       // sample would otherwise persist forever
       // in the feedback path.

2. PATHS must resolve in this repository. A path of three or more segments
   ending in a source-ish extension is checked against the tracked file list,
   both as written and relative to the citing file. Pointing a reader at a file
   that is not here is the same failure as citing a ticket number, and it is
   the form that survives longest unnoticed, because it looks like help.

3. LABELS a file invents for itself -- the short criterion tags an acceptance
   suite hangs its cases on -- are fine, but only in the file that defines
   them. Cite one from a second file and it has become a reference code again.
   Families are listed in LOCAL_LABEL_FAMILIES; a definition is a comment line
   of the form "LABEL-n: what it means".

Referring to a commit in this repository is fine; a reader can resolve it. So
is a published citation or a standard's number: those resolve in the world.

The rules are applied to every tracked text file that this project authors.
Vendored code is excluded because it is not ours to restyle, and binary files
are skipped by content. There is deliberately no extension allow-list: a check
that only looks at the file types someone thought of is a check with a door in
it.

Run from the repository root:

    python3 tools/check_comment_style.py

Exits non-zero and lists every offending line.
"""

import os
import re
import subprocess
import sys

# 1. Reference-code shapes that do not resolve inside this repository.
PATTERNS = [
    r"\bM-\d{3}[A-Za-z]?\b",
    r"\bM\d{3}[A-Za-z]?(?![0-9A-Za-z])",
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

# 3. Label families a file may define for its own use. Cited without a
#    definition in the same file, they are reference codes.
LOCAL_LABEL_FAMILIES = ("OA",)

# Third-party code is vendored verbatim and is not ours to restyle.
EXCLUDE_PREFIXES = (
    "DSParkLab/vendor/",
    "plugin/clap/clap/",
    "plugin/vst3/vst3_c_api.h",
    "plugin/webview/webview/",
)

# 2. Path rule: extensions that make a token a path claim rather than prose.
PATH_EXTENSIONS = (
    "h|hpp|inl|c|cc|cpp|md|py|json|jsonl|yml|yaml|txt|cmake|log|sh|bat|in|"
    "html|css|js|plist|xml|csv"
)
PATH_TOKEN = re.compile(
    r"(?<![\w/.$%-])((?:\.{1,2}/)?[A-Za-z][\w.+-]*(?:/[\w.+-]+){2,}\.(?:"
    + PATH_EXTENSIONS + r"))\b"
)
# The installation docs write include paths with the framework's own folder
# name in front, the way a user who copied it into their project would.
INSTALL_PREFIX = "DSPark/"
# Ignore files list patterns for things that are deliberately absent, so the
# path rule cannot apply to them.
NO_PATH_RULE = (".gitignore", ".gitattributes")


def run(*args):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          text=True).stdout


def tracked_paths():
    return [p for p in run("git", "ls-files", "-z").split("\0") if p]


def local_only_directories(tracked_set):
    """Top-level names that exist here but are hidden by a rule this
    repository does not carry -- a scratch tree, a private working area, an
    editor's cache. A reader cloning the repository has no way to know what
    they are, so a comment must not name one. The list is read from the
    working tree at run time and never written down: on a clean clone, where
    there is nothing local to leak, it comes out empty."""
    names = set()
    try:
        status = run("git", "status", "--ignored", "--porcelain")
    except subprocess.CalledProcessError:
        return names
    candidates = sorted({line[3:].split("/", 1)[0]
                         for line in status.split("\n")
                         if line.startswith("!! ")})
    for name in candidates:
        try:
            source = run("git", "check-ignore", "-v", name).split(":", 1)[0]
        except subprocess.CalledProcessError:
            continue
        # Hidden by a rule that ships with the repository: every reader sees
        # that rule, so the name is public and may be mentioned freely.
        if source in tracked_set:
            continue
        # Very short names collide with ordinary words too easily to ban.
        if len(name) >= 4:
            names.add(name)
    return names


def is_binary(path):
    try:
        with open(path, "rb") as handle:
            return b"\0" in handle.read(8192)
    except OSError:
        return True


def main():
    tracked = tracked_paths()
    tracked_set = set(tracked)
    directories = set()
    for path in tracked:
        parts = path.split("/")
        for index in range(1, len(parts)):
            directories.add("/".join(parts[:index]))

    regex = re.compile("|".join(PATTERNS))
    local_dirs = local_only_directories(tracked_set)
    local_regex = (re.compile("|".join(r"\b" + re.escape(n) + r"\b"
                                       for n in sorted(local_dirs)))
                   if local_dirs else None)
    label_regex = re.compile(
        r"\b(" + "|".join(LOCAL_LABEL_FAMILIES) + r")-(\d+)\b")

    hits = []
    for path in tracked:
        if path.startswith(EXCLUDE_PREFIXES) or is_binary(path):
            continue
        name = path.rsplit("/", 1)[-1]
        base = os.path.dirname(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
        except OSError as error:
            print("could not read {}: {}".format(path, error), file=sys.stderr)
            return 2

        body = "".join(lines)
        for number, line in enumerate(lines, 1):
            for match in regex.finditer(line):
                hits.append((path, number, match.group(0), line, "reference code"))
            if local_regex is not None:
                for match in local_regex.finditer(line):
                    hits.append((path, number, match.group(0), line,
                                 "names a directory that exists only here"))
            if name not in NO_PATH_RULE:
                stripped = re.sub(r"https?://\S+", "", line)
                stripped = re.sub(r"#\s*include\s*<[^>]*>", "", stripped)
                for match in PATH_TOKEN.finditer(stripped):
                    token = match.group(1)
                    candidates = {token, os.path.normpath(
                        os.path.join(base, token)) if base else token}
                    if token.startswith(INSTALL_PREFIX):
                        candidates.add(token[len(INSTALL_PREFIX):])
                    if any(c in tracked_set or c in directories
                           for c in candidates):
                        continue
                    hits.append((path, number, token, line,
                                 "path does not resolve in this repository"))
            for match in label_regex.finditer(line):
                label = match.group(0)
                defined = re.search(
                    r"(?m)^\s*(?://|#|\*)\s*" + re.escape(label) + r"\s*:", body)
                if not defined:
                    hits.append((path, number, label, line,
                                 "label is not defined in this file"))

    if not hits:
        print("comment style: no unresolvable references found")
        return 0

    for path, number, token, line, why in hits:
        print("{}:{}: {}  ({})".format(path, number, token, why))
        print("    {}".format(line.strip()))
    print("")
    print("{} unresolvable reference(s) found.".format(len(hits)))
    print("Comments must stand on their own: write out the reason instead of")
    print("citing something that only resolves somewhere else.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
