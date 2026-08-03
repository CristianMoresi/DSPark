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

2. PATHS must resolve in this repository. A token of three or more segments
   that claims to be a location is checked against the tracked file list, both
   as written and relative to the citing file. Pointing a reader at a file that
   is not here is the same failure as citing a ticket number, and it is the
   form that survives longest unnoticed, because it looks like help.

3. LABELS a file invents for itself -- the short criterion tags an acceptance
   suite hangs its cases on -- are fine, but only in the file that defines
   them. Cite one from a second file and it has become a reference code again.
   Families are listed in LOCAL_LABEL_FAMILIES; a definition is a comment line
   of the form "LABEL-n: what it means".

Referring to a commit in this repository is fine; a reader can resolve it. So
is a published citation or a standard's number: those resolve in the world.

All three rules are applied to every tracked text file that this project
authors, AND to the tracked file names themselves: a name is more public than
anything inside the file, since it shows in the file listing and in every
release archive without being opened. Vendored code is excluded because it is
not ours to restyle. Binary files are skipped by content, never by extension,
and the number skipped is reported; a file that cannot be READ is a different
thing from a file that is not text, and it stops the check with exit code 2
rather than passing quietly.

What rule 2 does and does not see, stated so the rule and the implementation
agree (each limit below was measured against this tree before being accepted):

  - A location claim is a token of three or more segments that either ends in
    a file extension, or ends in "/", or is rooted at "/", "~/" or a drive
    letter. Separators may be "/" or "\\".
  - Extensions are NOT an allow-list; every extension counts except those of
    build outputs and binary assets (ARTEFACT_EXTENSIONS), because a bundle
    layout in a build recipe is not a claim about this repository.
  - A rooted path needs four or more segments: three-segment absolute paths in
    this tree are system locations (/usr/bin/env) rather than project files.
  - An extension-less path with no trailing "/" is NOT seen. It cannot be told
    apart from ordinary prose alternation (HTML/CSS/JS, attack/release/ratio)
    by shape alone: measured on this tree, treating those as location claims
    produces between 22 and 114 false reports. Write such a path with its
    trailing "/" and the rule applies.
  - Two-segment paths are not seen either; too much prose is two words with a
    slash between them.
  - In Windows scripts the backslash form is skipped, since backslash paths
    are that language's native way of naming system directories.

Run from the repository root:

    python3 tools/check_comment_style.py

Exits 1 and lists every offending line, or 2 if a tracked file cannot be read.
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

# 2. Path rule. A location claim is three or more segments that end in an
#    extension, end in a slash, or start at a filesystem root.
SEGMENT = r"[\w.+-]+"
PATH_TOKEN = re.compile(
    r"(?<![\w/.$%~:-])((?:~/|\.{1,2}/|/)?[A-Za-z][\w.+-]*(?:/" + SEGMENT
    + r"){2,}/?)")
DRIVE_TOKEN = re.compile(
    r"(?<![\w/.$%~:-])([A-Za-z]:[\\/](?:" + SEGMENT + r"[\\/]){2,}" + SEGMENT
    + r"[\\/]?)")
BACKSLASH_TOKEN = re.compile(
    r"(?<![\w\\.$%:-])([A-Za-z][\w.+-]*(?:\\" + SEGMENT + r"){2,})")
ANY_EXTENSION = re.compile(r"\.([A-Za-z][A-Za-z0-9]{0,5})$")
# Extensions of things a build produces or of binary assets. A path ending in
# one of these describes an output layout, not a file this repository tracks.
ARTEFACT_EXTENSIONS = frozenset((
    "so", "dll", "dylib", "a", "lib", "o", "obj", "exe", "pdb", "ilk", "exp",
    "vst3", "clap", "component", "app", "framework", "bundle", "aaxplugin",
    "zip", "gz", "tar", "7z", "dmg", "pkg", "msi", "deb", "rpm",
    "png", "jpg", "jpeg", "gif", "ico", "bmp", "tiff", "icns",
    "wav", "mp3", "flac", "aiff", "aif", "ogg", "mid", "midi",
    "ttf", "otf", "woff", "woff2", "pyc",
))
# Windows scripts name system directories with backslashes as a matter of
# course; the backslash form of the rule would be noise there.
WINDOWS_SCRIPT_SUFFIXES = (".bat", ".cmd", ".ps1")
# The installation docs write include paths with the framework's own folder
# name in front, the way a user who copied it into their project would -- and
# they write where that folder goes in the user's project, which is the same
# claim seen from the other end.
INSTALL_PREFIX = "DSPark/"
INSTALL_FOLDER = "DSPark"
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
    they are, so a comment must not name one.

    A candidate comes from git's own list of ignored entries, and is kept when
    `git check-ignore` answers for it with a rule whose source file is not
    tracked here. The question is asked about the directory's CONTENTS -- the
    trailing slash matters -- because a directory is very often hidden by a
    .gitignore that sits inside it rather than by a rule that names it from
    outside, and asked about the bare name that case answers "no rule" and the
    directory would be missed. Both shapes must reach this list; they were
    measured side by side.

    The list is read from the working tree at run time and never written down.
    That has a consequence worth stating: on a clean clone, and on any machine
    whose working tree holds nothing locally ignored, it comes out empty and
    this rule checks nothing at all. A green run is not evidence that the rule
    fired."""
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
            source = run("git", "check-ignore", "-v",
                         name + "/").split(":", 1)[0]
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
    """True when the file's first block holds a NUL byte. Raises OSError if
    the file cannot be read: a scanner that cannot open a tracked file must
    say so, not assume it is a picture."""
    with open(path, "rb") as handle:
        return b"\0" in handle.read(8192)


def extension_of(segment):
    """The segment's extension in lower case, or "" if it has none."""
    match = ANY_EXTENSION.search(segment)
    return match.group(1).lower() if match else ""


def path_claims(line, in_windows_script):
    """Every token in `line` that claims to be a location, normalised to
    forward slashes."""
    tokens = []
    for match in DRIVE_TOKEN.finditer(line):
        tokens.append(match.group(1)[2:].replace("\\", "/"))
    if not in_windows_script:
        for match in BACKSLASH_TOKEN.finditer(line):
            tokens.append(match.group(1).replace("\\", "/"))
    for match in PATH_TOKEN.finditer(line):
        tokens.append(match.group(1))
    claims = []
    for token in tokens:
        token = token.rstrip(".,;:")
        rooted = token.startswith(("/", "~/"))
        body = token[2:] if token.startswith("~/") else token.lstrip("/")
        segments = [s for s in body.split("/") if s]
        if len(segments) < 3 or (rooted and len(segments) < 4):
            continue
        if any(extension_of(s) in ARTEFACT_EXTENSIONS for s in segments):
            continue
        extension = extension_of(segments[-1])
        if token.endswith("/"):
            # A directory claim. The install docs say where the framework's
            # own folder goes in someone else's project; that is not a claim
            # about a path here.
            if segments[-1] == INSTALL_FOLDER:
                continue
        elif not extension:
            # Extension-less and not marked as a directory: indistinguishable
            # from prose alternation by shape. See the module docstring.
            continue
        claims.append((token, "/".join(segments)))
    return claims


def resolves(token, base, tracked_set, directories):
    """A claim resolves when it names something tracked here, as written or
    read relative to the citing file. A tail of it matching is deliberately
    NOT enough: an outside path would then be excused whenever its last
    segments happened to coincide with a file here."""
    candidates = {token}
    if base:
        candidates.add(os.path.normpath(os.path.join(base, token)))
    if token.startswith(INSTALL_PREFIX):
        candidates.add(token[len(INSTALL_PREFIX):])
    return any(c in tracked_set or c in directories for c in candidates)


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
    skipped_binary = 0

    def scan_names(text, number, path, body, where):
        """Rules 1 and 3 over one string. `where` names what is being read."""
        for match in regex.finditer(text):
            hits.append((path, number, match.group(0), text,
                         "reference code" + where))
        if local_regex is not None:
            for match in local_regex.finditer(text):
                hits.append((path, number, match.group(0), text,
                             "names a directory that exists only here" + where))
        for match in label_regex.finditer(text):
            label = match.group(0)
            defined = re.search(
                r"(?m)^\s*(?://|#|\*)\s*" + re.escape(label) + r"\s*:", body)
            if not defined:
                hits.append((path, number, label, text,
                             "label is not defined in this file" + where))

    for path in tracked:
        if path.startswith(EXCLUDE_PREFIXES):
            continue
        try:
            binary = is_binary(path)
        except OSError as error:
            print("could not read {}: {}".format(path, error), file=sys.stderr)
            return 2
        lines = []
        if binary:
            skipped_binary += 1
        else:
            try:
                with open(path, "r", encoding="utf-8",
                          errors="replace") as handle:
                    lines = handle.readlines()
            except OSError as error:
                print("could not read {}: {}".format(path, error),
                      file=sys.stderr)
                return 2

        name = path.rsplit("/", 1)[-1]
        base = os.path.dirname(path)
        body = "".join(lines)

        # The file's own name is read first: it is public whether or not the
        # contents are text, so this runs for binary files too.
        scan_names(path, 0, path, body, " in the file's name")
        if binary:
            continue

        windows_script = name.lower().endswith(WINDOWS_SCRIPT_SUFFIXES)
        for number, line in enumerate(lines, 1):
            scan_names(line, number, path, body, "")
            if name in NO_PATH_RULE:
                continue
            stripped = re.sub(r"https?://\S+", "", line)
            stripped = re.sub(r"#\s*include\s*<[^>]*>", "", stripped)
            for token, normalised in path_claims(stripped, windows_script):
                if resolves(normalised, base, tracked_set, directories):
                    continue
                hits.append((path, number, token, line,
                             "path does not resolve in this repository"))

    if skipped_binary:
        print("comment style: {} tracked file(s) skipped as binary "
              "(names still checked).".format(skipped_binary))
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
