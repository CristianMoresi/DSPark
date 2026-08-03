#!/usr/bin/env python3
"""Comment style check: comments and documentation must be self-contained.

A reader of this repository has the repository and nothing else. A comment that
cites something only resolvable elsewhere -- an issue number, a ticket tag, a
planning document, a directory that exists only on the author's machine --
sends that reader after something they cannot open, and the reason the code is
the way it is goes with it.

Four rules, from the narrowest to the most general:

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

4. LOCAL DIRECTORIES: a top-level directory that exists on the machine running
   the check and is hidden by a rule this repository does not carry is not
   something a reader can look up, so its name must not appear in a comment
   either. See local_only_directories() for how the list is built and for the
   two tests that keep this rule off the vocabulary of the project itself.

Rules 1 and 4 ignore case; rule 4's list is read from the working tree, so on a
clean clone it is empty and that rule checks nothing. The run says how many
names it holds, because a green line for a check that did not run is worth
less than no line at all. It also says which commit it measured "a word this
repository already uses" against, since that answer decides which names the
rule holds: a name this repository already PUBLISHES is not banned, so a
mention that has already been pushed stops arming the rule for that name.

Referring to a commit in this repository is fine; a reader can resolve it. So
is a published citation or a standard's number: those resolve in the world.

All four rules are applied to every tracked text file that this project
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
    letter. Separators may be "/" or "\\", doubled separators count once, and
    a "file://" prefix is dropped, since such a URL is a local path wearing a
    scheme. An "https://" URL is not a location claim at all: it resolves in
    the world.
  - Extensions are NOT an allow-list; every extension counts except those of
    build outputs and binary assets (ARTEFACT_EXTENSIONS), because a bundle
    layout in a build recipe is not a claim about this repository.
  - A path rooted at "/" needs four or more segments: three-segment absolute
    paths in this tree are system locations (/usr/bin/env) rather than project
    files. A path rooted at a home directory takes the ordinary floor of
    three, since nothing under someone's home directory is a system location.
  - An extension-less path with no trailing "/" is NOT seen. It cannot be told
    apart from ordinary prose alternation (HTML/CSS/JS, attack/release/ratio)
    by shape alone: measured on this tree, treating those as location claims
    produces between 22 and 114 false reports. Write such a path with its
    trailing "/" and the rule applies.
  - Two-segment paths are not seen either; too much prose is two words with a
    slash between them.
  - In Windows scripts the backslash form is skipped, since backslash paths
    are that language's native way of naming system directories.
  - A path written through the home DIRECTORY VARIABLE -- "$HOME/...",
    "${HOME}/..." or "%USERPROFILE%\\..." -- is read as "~/", because it is
    the same claim about the same place; but inside a shell or CI recipe it
    is left alone, for the same reason the backslash form is left alone in a
    Windows script: there that spelling is how the language names a system
    directory, not prose about this repository. Measured on this tree, that
    condition is what makes the form free: read everywhere, it reports one
    line, the plugin install destination in this project's own CI recipe;
    read everywhere but in recipes, it reports none and still catches a
    private working tree named under the same home directory.

What NO rule sees, stated for the same reason. The scan is line oriented, so a
token split across two lines of a wrapped comment, across a backslash
continuation, or written as two adjacent string literals is not seen; and it
compares bytes, so a Cyrillic letter standing in for a Latin one, a U+2010
hyphen or a zero-width space is not seen either. Closing those needs a parser
and a Unicode normaliser. This check is for the honest mistake, which is the
one that actually happens; it is not a defence against an author working
around it, and nothing here should be read as one.

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

# 4. A local directory's name is banned only if it looks like a name someone
#    chose rather than a word the language or the field already owns: it
#    carries a hyphen, an underscore, or a capital that is not the first
#    letter. See local_only_directories().
NAME_SHAPED = re.compile(r"[-_]|(?<=.)[A-Z]")

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
# A "file://" URL is a local path with a scheme in front; the path underneath
# is the claim. Doubled separators are the same path written carelessly.
FILE_URL = re.compile(r"\bfile://")
DOUBLED_SEPARATOR = re.compile(r"(?<!:)/{2,}")
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
# The home directory written through its variable is the same claim as "~/".
HOME_VARIABLE = re.compile(r"(?:\$\{?HOME\}?|%USERPROFILE%)[\\/]")
# ... except in a recipe, where naming a system directory through that
# variable is the language's own spelling rather than a claim about this
# repository -- the same exemption Windows scripts get for backslashes.
RECIPE_SUFFIXES = (".sh", ".yml", ".yaml")
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


def run_quiet(*args):
    """run() with git's own diagnostics discarded. Used for the probes below,
    which ask git questions it is entitled to refuse: an unignored path, or a
    path behind a symlink. Those refusals are answers here, not errors, and a
    raw "fatal:" in this tool's output -- or in a CI log -- would say the
    check broke when it did not."""
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True).stdout


# A gitlink is a commit id, not a file: there is nothing in this worktree to
# open at that path, and opening it is how a submodule would turn this check
# red with a message about an unreadable file.
GITLINK_MODE = "160000"


def tracked_paths():
    paths = []
    for entry in run("git", "ls-files", "-s", "-z").split("\0"):
        if not entry or "\t" not in entry:
            continue
        header, path = entry.split("\t", 1)
        if header.split(" ", 1)[0] == GITLINK_MODE:
            continue
        paths.append(path)
    return paths


def published_ref():
    """The commit this repository has PUBLISHED, and whether that is what it
    really is. The branch's upstream if it has one, else the remote's default
    branch, else HEAD.

    See local_only_directories() for why the calibration corpus must not be
    the working tree or the index. HEAD is not right either, and for the same
    reason one step along: a commit that has not left this machine is not yet
    something a reader can resolve, so a mention of the private name inside
    one must not count as this repository "using" the word -- otherwise the
    very act the rule guards against, committing the name, switches the rule
    off for that name for good.

    With no upstream and no remote there is nothing published to measure
    against and the corpus falls back to HEAD, which is the old behaviour and
    the weaker one; the run says so rather than pretending otherwise."""
    for candidate in ("@{upstream}", "origin/HEAD"):
        try:
            name = run_quiet("git", "rev-parse", "--abbrev-ref", "--verify",
                             candidate).strip()
        except subprocess.CalledProcessError:
            continue
        if name:
            return name, True
    return "HEAD", False


def published_paths(ref):
    """The file list of the published commit."""
    try:
        out = run_quiet("git", "ls-tree", "-r", "-z", "--name-only", ref)
    except subprocess.CalledProcessError:
        return []
    return [p for p in out.split("\0") if p]


def used_as_a_word_here(name, published, ref):
    """True when `name` already occurs as a word in the published tree, in a
    file name or in the text of a file this check reads."""
    if any(re.search(r"\b" + re.escape(name) + r"\b", path, re.IGNORECASE)
           for path in published):
        return True
    # Vendored code is excluded here for the same reason it is excluded from
    # the scan: a word this repository never reads cannot be reported by it.
    probe = subprocess.run(
        ("git", "grep", "-I", "-i", "-w", "-F", "-q", "-e", name, ref, "--")
        + tuple(":(exclude)" + prefix for prefix in EXCLUDE_PREFIXES),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 0 found, 1 not found, anything else is git declining to answer (no
    # commit yet, a git too old for exclude pathspecs). An unanswered question
    # counts as "used", which drops the candidate: this rule reporting nothing
    # is visible in the status line, whereas this rule reporting the wrong
    # thing is what made it worth deleting.
    return probe.returncode != 1


def local_only_directories(tracked_set, ref=None):
    """Top-level names that exist here but are hidden by a rule this
    repository does not carry -- a scratch tree, a private working area, an
    editor's cache. A reader cloning the repository has no way to know what
    they are, so a comment must not name one.

    A candidate comes from git's own list of ignored entries, and is kept when
    `git check-ignore` answers for it with a rule whose source file is not
    tracked here. The question is asked BOTH ways, about the directory's
    contents and about the bare name, and either answer will do. A directory is
    very often hidden by a .gitignore sitting inside it rather than by a rule
    naming it from outside, and asked about the bare name that case answers "no
    rule"; but asked with the trailing slash about a name that is a symlink to
    a tree elsewhere, git refuses the question entirely ("pathspec is beyond a
    symbolic link") and the answer is only available without it. Each form is
    the only one that works in a configuration this project actually uses.

    A name is then banned only if it could not plausibly be anything else,
    because this rule reads a private machine and reports on shared code, and
    a false report here is worse than a miss: it turns the check red on files
    the contributor never touched, and a check that cries wolf gets deleted.
    Two tests, both cheap:

      - It must LOOK like a name rather than a word -- a hyphen, an underscore
        or an internal capital. "audio", "gain", "private" and "samples" are
        the names people give scratch directories in this field, and they are
        also this framework's own vocabulary.
      - It must not already be USED as a word by the PUBLISHED tree, in a file
        name or in the text of a file this check reads. A word the repository
        itself publishes is by definition not something only resolvable on one
        machine, so banning it can only produce false reports. The other side
        of that is the rule's one blind spot, and it is worth stating plainly:
        once a mention of the name has been pushed, the name is published
        vocabulary and this rule no longer holds it. See published_ref().

    A plain lowercase name stays reachable through rule 2 in its path form,
    which is unchanged and needs no local knowledge at all.

    The list is read from the working tree at run time and never written down.
    That has a consequence worth stating: on a clean clone, and on any machine
    whose working tree holds nothing locally ignored, it comes out empty and
    this rule checks nothing at all. A green run is not evidence that the rule
    fired, which is why the run prints how many names it holds."""
    names = set()
    try:
        status = run("git", "status", "--ignored", "--porcelain")
    except subprocess.CalledProcessError:
        return names
    candidates = sorted({line[3:].split("/", 1)[0]
                         for line in status.split("\n")
                         if line.startswith("!! ")})
    published = None
    for name in candidates:
        # Very short names collide with ordinary words too easily to ban.
        if len(name) < 4:
            continue
        if not NAME_SHAPED.search(name):
            continue
        source = None
        for probe in (name + "/", name):
            try:
                source = run_quiet("git", "check-ignore", "-v",
                                   probe).split(":", 1)[0]
                break
            except subprocess.CalledProcessError:
                continue
        if source is None:
            continue
        # Hidden by a rule that ships with the repository: every reader sees
        # that rule, so the name is public and may be mentioned freely.
        if source in tracked_set:
            continue
        if published is None:
            ref = ref or published_ref()[0]
            published = published_paths(ref)
        if used_as_a_word_here(name, published, ref):
            continue
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


def path_claims(line, in_windows_script, in_recipe):
    """Every token in `line` that claims to be a location, normalised to
    forward slashes."""
    if not in_recipe:
        line = HOME_VARIABLE.sub("~/", line)
    line = DOUBLED_SEPARATOR.sub("/", FILE_URL.sub("", line))
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
        absolute = token.startswith("/")
        body = token[2:] if token.startswith("~/") else token.lstrip("/")
        segments = [s for s in body.split("/") if s]
        # The four-segment floor answers /usr/bin/env; nothing under someone's
        # home directory is a system location, so "~/" keeps the usual three.
        if len(segments) < 3 or (absolute and len(segments) < 4):
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

    # Case is not part of either claim. A reference code lower-cased is the
    # same dangling reference, and this repository's own file names are all
    # lower case, which is the spelling a document migrating out of a private
    # tree would most naturally arrive with. Measured over this tree, reading
    # both rules without regard to case costs nothing: it adds no report.
    regex = re.compile("|".join(PATTERNS), re.IGNORECASE)
    ref, is_published = published_ref()
    local_dirs = local_only_directories(tracked_set, ref)
    local_regex = (re.compile("|".join(r"\b" + re.escape(n) + r"\b"
                                       for n in sorted(local_dirs)),
                              re.IGNORECASE)
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
        recipe = windows_script or name.lower().endswith(RECIPE_SUFFIXES)
        for number, line in enumerate(lines, 1):
            scan_names(line, number, path, body, "")
            if name in NO_PATH_RULE:
                continue
            stripped = re.sub(r"https?://\S+", "", line)
            stripped = re.sub(r"#\s*include\s*<[^>]*>", "", stripped)
            for token, normalised in path_claims(stripped, windows_script,
                                                 recipe):
                if resolves(normalised, base, tracked_set, directories):
                    continue
                hits.append((path, number, token, line,
                             "path does not resolve in this repository"))

    if skipped_binary:
        print("comment style: {} tracked file(s) skipped as binary "
              "(names still checked).".format(skipped_binary))
    # The count, never the names: it is 0 on a clean clone and on CI, and it
    # is the difference between "this rule found nothing" and "this rule was
    # not looking". The names themselves are the private thing and stay here.
    # The ref is public by construction, and which one it is decides which
    # names the rule holds, so it is said out loud rather than assumed.
    print("comment style: local-directory rule: {} name(s) in effect "
          "(calibrated against {})".format(
              len(local_dirs),
              ref if is_published
              else ref + ": nothing published to measure against"))
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
