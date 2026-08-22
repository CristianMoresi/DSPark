#!/usr/bin/env python3
"""Fail-closed validation of every tracked DSPark public file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = ROOT / "tools" / "public_file_policy.json"
TEXT_KINDS = {"first_party_ascii_text", "third_party_utf8_exact"}
ALL_KINDS = TEXT_KINDS | {"binary_exact"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(root: Path, arguments: list[str], input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


def tracked_paths(root: Path) -> list[str]:
    raw = git(root, ["ls-files", "-z"])
    return [item.decode("utf-8") for item in raw.split(b"\0") if item]


def index_modes(root: Path) -> dict[str, str]:
    modes: dict[str, str] = {}
    raw = git(root, ["ls-files", "--stage", "-z"])
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, _object_id, stage = metadata.decode("ascii").split()
        if stage == "0":
            modes[path.decode("utf-8")] = mode
    return modes


def text_attributes(root: Path, paths: Iterable[str]) -> dict[str, str]:
    path_list = list(paths)
    request = b"".join(os.fsencode(path) + b"\0" for path in path_list)
    raw = git(root, ["check-attr", "-z", "--stdin", "text"], request)
    fields = raw.split(b"\0")
    attributes: dict[str, str] = {}
    for index in range(0, len(fields) - 2, 3):
        if not fields[index]:
            continue
        path = fields[index].decode("utf-8")
        name = fields[index + 1].decode("ascii")
        value = fields[index + 2].decode("ascii")
        if name != "text":
            raise RuntimeError(f"unexpected Git attribute {name!r}")
        attributes[path] = value
    return attributes


def validate_bytes(entry: dict[str, Any], data: bytes, attribute: str) -> list[str]:
    kind = entry["kind"]
    errors: list[str] = []
    if kind == "first_party_ascii_text":
        if b"\0" in data:
            errors.append("NUL_IN_FIRST_PARTY_TEXT")
        try:
            decoded = data.decode("utf-8", "strict")
        except UnicodeDecodeError:
            errors.append("INVALID_UTF8_TEXT")
        else:
            if any(ord(character) > 0x7F for character in decoded):
                errors.append("FIRST_PARTY_NON_ASCII")
    elif kind == "third_party_utf8_exact":
        try:
            data.decode("utf-8", "strict")
        except UnicodeDecodeError:
            errors.append("INVALID_UTF8_VENDOR_TEXT")
        if sha256(data) != entry.get("sha256"):
            errors.append("VENDOR_BYTE_DRIFT")
    elif kind == "binary_exact":
        if sha256(data) != entry.get("sha256"):
            errors.append("BINARY_BYTE_DRIFT")
    else:
        errors.append("UNKNOWN_POLICY_KIND")

    if kind in TEXT_KINDS and attribute == "unset":
        errors.append("TEXT_ATTRIBUTE_DRIFT")
    if kind == "binary_exact" and attribute != "unset":
        errors.append("BINARY_TEXT_ATTRIBUTE_DRIFT")
    return errors


def compare_inventory(policy_paths: set[str], actual_paths: set[str]) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    errors.extend(("UNKNOWN_PATH", path) for path in sorted(actual_paths - policy_paths))
    errors.extend(("MISSING_PATH", path) for path in sorted(policy_paths - actual_paths))
    return errors


def validate(root: Path, policy_path: Path) -> tuple[list[tuple[str, str]], dict[str, int]]:
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    entries = policy.get("paths")
    if not isinstance(entries, list):
        raise RuntimeError("policy paths must be a list")

    by_path: dict[str, dict[str, Any]] = {}
    errors: list[tuple[str, str]] = []
    for entry in entries:
        path = entry.get("path")
        kind = entry.get("kind")
        if not isinstance(path, str) or not path or path.startswith("/"):
            errors.append(("INVALID_POLICY_PATH", repr(path)))
            continue
        if path in by_path:
            errors.append(("DUPLICATE_POLICY_PATH", path))
            continue
        if kind not in ALL_KINDS:
            errors.append(("UNKNOWN_POLICY_KIND", path))
        by_path[path] = entry

    actual = set(tracked_paths(root))
    errors.extend(compare_inventory(set(by_path), actual))
    modes = index_modes(root)
    attributes = text_attributes(root, actual)

    counts = {kind: 0 for kind in ALL_KINDS}
    for path in sorted(actual & set(by_path)):
        entry = by_path[path]
        kind = entry["kind"]
        counts[kind] += 1
        full_path = root / path
        try:
            file_stat = full_path.lstat()
        except FileNotFoundError:
            errors.append(("MISSING_WORKTREE_PATH", path))
            continue
        if not stat.S_ISREG(file_stat.st_mode) or modes.get(path) not in {"100644", "100755"}:
            errors.append(("NONREGULAR_OR_SYMLINK_PATH", path))
            continue
        data = full_path.read_bytes()
        for code in validate_bytes(entry, data, attributes.get(path, "unspecified")):
            errors.append((code, path))

    declared_counts = policy.get("counts", {})
    expected_counts = dict(counts)
    expected_counts["total"] = sum(counts.values())
    if declared_counts != expected_counts:
        errors.append(("POLICY_COUNT_DRIFT", json.dumps(expected_counts, sort_keys=True)))

    required_lines = policy.get("gitattributes_required_lines", [])
    attribute_lines = set((root / ".gitattributes").read_text(encoding="ascii").splitlines())
    for line in required_lines:
        if line not in attribute_lines:
            errors.append(("MISSING_GITATTRIBUTES_RULE", line))
    return errors, expected_counts


def self_test(root: Path, policy_path: Path) -> int:
    checks: list[tuple[str, bool]] = []
    first = {"kind": "first_party_ascii_text"}
    checks.append(("first-party-nonascii", "FIRST_PARTY_NON_ASCII" in validate_bytes(first, b"x\xc3\xa9", "auto")))
    checks.append(("invalid-utf8-text", "INVALID_UTF8_TEXT" in validate_bytes(first, b"x\xff", "auto")))
    checks.append(("new-unclassified-path", compare_inventory({"a"}, {"a", "new"}) == [("UNKNOWN_PATH", "new")]))

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    by_path = {entry["path"]: entry for entry in policy["paths"]}
    vendor_mutants = (
        ("imgui-valid-ascii-byte-drift", "DSParkLab/vendor/imgui/imgui.h"),
        ("miniaudio-valid-ascii-byte-drift", "DSParkLab/vendor/miniaudio.h"),
        ("existing-webview-vendor-byte-drift", "plugin/webview/webview/webview.h"),
        ("existing-clap-vendor-byte-drift", "plugin/clap/clap/clap.h"),
        ("legal-file-byte-drift", "plugin/clap/LICENSE_CLAP.txt"),
    )
    for name, path in vendor_mutants:
        entry = by_path.get(path, {})
        data = (root / path).read_bytes()
        checks.append((
            name,
            entry.get("kind") == "third_party_utf8_exact"
            and "VENDOR_BYTE_DRIFT" in validate_bytes(entry, data + b" ", "auto"),
        ))

    binary = {"kind": "binary_exact", "sha256": sha256(b"\x00\x01")}
    checks.append(("binary-text-attribute-drift", "BINARY_TEXT_ATTRIBUTE_DRIFT" in validate_bytes(binary, b"\x00\x01", "auto")))
    failures = [name for name, passed in checks if not passed]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} mutant {name}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        try:
            return self_test(arguments.root.resolve(), arguments.policy.resolve())
        except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"ERROR HARNESS {error}", file=sys.stderr)
            return 2
    try:
        errors, counts = validate(arguments.root.resolve(), arguments.policy.resolve())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR HARNESS {error}", file=sys.stderr)
        return 2
    for code, subject in errors:
        print(f"ERROR {code} {subject}", file=sys.stderr)
    if errors:
        return 1
    print("PASS public text policy " + json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
