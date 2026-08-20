#!/usr/bin/env python3
"""Reject dependent member comparisons that MSVC can parse as template-ids.

Inside a template, a name reached through an expression whose type is not yet
known cannot be resolved when the body is parsed.  MSVC can read

    value.last < value.first || value.last > limit

as a template argument list beginning at the less-than sign and ending at the
later greater-than sign.  Parenthesizing each comparison removes the ambiguity.

This is intentionally a conservative lexical superset rather than a C++
parser.  It fires only inside a template body, after ``.`` or ``->``, when a
later ``>`` balances the ``<`` and the span has a top-level comma, ``||`` or
``&&``.  A real member-template call caught by that shape needs the ``template``
keyword and is therefore actionable too.

Run from the repository root.  Exit 1 means at least one ambiguous span, and
exit 2 means a tracked source could not be enumerated or read.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


SOURCE_SUFFIXES = (".h", ".hpp", ".inl", ".cpp", ".cc", ".cxx")
EXCLUDE_PREFIXES = (
    "DSParkLab/vendor/",
    "plugin/clap/clap/",
    "plugin/vst3/vst3_c_api.h",
    "plugin/webview/webview/",
)
PUNCTUATION = sorted(
    (
        "<=>", "...", "->*", "<<=", ">>=", "->", "::", "<=", ">=", "<<",
        ">>", "&&", "||", "==", "!=", "++", "--", "+=", "-=", "*=", "/=",
        "%=", "&=", "|=", "^=", ".*",
    ),
    key=len,
    reverse=True,
)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
NUMBER = re.compile(r"\.?[0-9](?:[eEpP][+-]|[A-Za-z_0-9.'])*")
EXPRESSION_ENDS = (";", "{", "}")
AMBIGUITY_MARKERS = (",", "||", "&&")
GITLINK_MODE = "160000"


def tracked_sources(repo: Path) -> list[str]:
    completed = subprocess.run(
        ("git", "-C", str(repo), "ls-files", "-s", "-z"),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise OSError(f"cannot enumerate tracked files: {detail}")
    paths: list[str] = []
    for raw_entry in completed.stdout.split(b"\0"):
        if not raw_entry or b"\t" not in raw_entry:
            continue
        raw_header, raw_path = raw_entry.split(b"\t", 1)
        mode = raw_header.split(b" ", 1)[0].decode("ascii")
        path = raw_path.decode("utf-8", "surrogateescape")
        if mode == GITLINK_MODE:
            continue
        if not path.endswith(SOURCE_SUFFIXES):
            continue
        if path.startswith(EXCLUDE_PREFIXES):
            continue
        paths.append(path)
    return paths


def tokenize(text: str) -> list[tuple[int, str]]:
    """Return line/token pairs with comments, literals and directives hidden."""

    tokens: list[tuple[int, str]] = []
    index, end, line = 0, len(text), 1
    while index < end:
        char = text[index]
        if char == "\n":
            line += 1
            index += 1
        elif char in " \t\r":
            index += 1
        elif text.startswith("//", index):
            stop = text.find("\n", index)
            index = end if stop < 0 else stop
        elif text.startswith("/*", index):
            stop = text.find("*/", index + 2)
            stop = end if stop < 0 else stop + 2
            line += text.count("\n", index, stop)
            index = stop
        elif char == "#":
            stop = index
            while True:
                newline = text.find("\n", stop)
                if newline < 0:
                    stop = end
                    break
                if text[newline - 1] != "\\":
                    stop = newline
                    break
                stop = newline + 1
            line += text.count("\n", index, stop)
            index = stop
        elif char in "\"'":
            stop = index + 1
            while stop < end and text[stop] != char:
                stop += 2 if text[stop] == "\\" else 1
            stop = min(stop + 1, end)
            line += text.count("\n", index, stop)
            tokens.append((line, "LITERAL"))
            index = stop
        else:
            number = NUMBER.match(text, index)
            if number and (
                char.isdigit()
                or (char == "." and index + 1 < end and text[index + 1].isdigit())
            ):
                tokens.append((line, "LITERAL"))
                index = number.end()
                continue
            name = IDENTIFIER.match(text, index)
            if name:
                tokens.append((line, name.group()))
                index = name.end()
                continue
            for punctuation in PUNCTUATION:
                if text.startswith(punctuation, index):
                    tokens.append((line, punctuation))
                    index += len(punctuation)
                    break
            else:
                tokens.append((line, char))
                index += 1
    return tokens


def template_body_flags(tokens: list[tuple[int, str]]) -> list[bool]:
    flags = [False] * len(tokens)
    opened_by_template: list[bool] = []
    pending = False
    for position, (_, token) in enumerate(tokens):
        flags[position] = any(opened_by_template)
        if token == "template":
            pending = True
        elif token == "{":
            opened_by_template.append(pending)
            pending = False
        elif token == "}":
            if opened_by_template:
                opened_by_template.pop()
        elif token == ";":
            pending = False
    return flags


def balancing_angle(tokens: list[tuple[int, str]], opening: int) -> int | None:
    nesting, angles = 0, 1
    position = opening + 1
    while position < len(tokens):
        token = tokens[position][1]
        if token in ("(", "["):
            nesting += 1
        elif token in (")", "]"):
            if nesting == 0:
                return None
            nesting -= 1
        elif token in EXPRESSION_ENDS:
            return None
        elif nesting == 0 and token == "<":
            angles += 1
        elif nesting == 0 and token in (">", ">>"):
            angles -= 1 if token == ">" else 2
            if angles <= 0:
                return position
        position += 1
    return None


def is_ambiguous_span(tokens: list[tuple[int, str]], opening: int, closing: int) -> bool:
    nesting = 0
    for _, token in tokens[opening + 1:closing]:
        if token in ("(", "["):
            nesting += 1
        elif token in (")", "]"):
            nesting -= 1
        elif nesting == 0 and token in AMBIGUITY_MARKERS:
            return True
    return False


def offences(tokens: list[tuple[int, str]]):
    flags = template_body_flags(tokens)
    for position in range(2, len(tokens)):
        line, token = tokens[position]
        if token != "<" or not flags[position]:
            continue
        name = tokens[position - 1][1]
        if name == "LITERAL" or not IDENTIFIER.fullmatch(name):
            continue
        if tokens[position - 2][1] not in (".", "->"):
            continue
        closing = balancing_angle(tokens, position)
        if closing is None or not is_ambiguous_span(tokens, position, closing):
            continue
        span = " ".join(
            "..." if token == "LITERAL" else token
            for _, token in tokens[position - 2:closing + 1]
        )
        yield line, tokens[closing][0], span


def scan_repository(repo: Path) -> int:
    found = 0
    for relative in tracked_sources(repo):
        path = repo / relative
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except OSError as error:
            print(f"cannot read {relative}: {error}", file=sys.stderr)
            return 2
        except UnicodeDecodeError:
            continue
        for line, closing_line, span in offences(tokenize(text)):
            found += 1
            print(
                "%s:%d: `<` here is closed by the `>` on line %d, so MSVC "
                "can read this as a template argument list"
                % (relative, line, closing_line)
            )
            print("    %s" % span[:200])
            print("    parenthesise each comparison: (a.b < c) || (a.b > d)")
    if found:
        print("dependent comparisons: %d ambiguous span(s) found" % found)
        return 1
    print("dependent comparisons: no ambiguous span found")
    return 0


def self_test_bad() -> int:
    bad = "template<class T> bool bad(T x) { return x.last < x.first || x.last > 4; }"
    found = list(offences(tokenize(bad)))
    if len(found) != 1:
        print("dependent comparisons: bad-shape control was not detected", file=sys.stderr)
        return 2
    print("dependent comparisons: 1 ambiguous span(s) found in synthetic bad shape")
    return 1


def self_test() -> int:
    bad = "template<class T> bool bad(T x) { return x.last < x.first || x.last > 4; }"
    near = "template<class T> bool good(T x) { return (x.last < x.first) || (x.last > 4); }"
    bad_count = len(list(offences(tokenize(bad))))
    near_count = len(list(offences(tokenize(near))))
    if bad_count != 1 or near_count != 0:
        print(
            f"dependent comparisons: self-test failed bad={bad_count} near={near_count}",
            file=sys.stderr,
        )
        return 2
    print("dependent comparisons: self-test PASS bad=1 near-miss=0")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-bad", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.self_test_bad:
        return self_test_bad()
    if args.self_test:
        return self_test()
    return scan_repository(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    sys.exit(main())
