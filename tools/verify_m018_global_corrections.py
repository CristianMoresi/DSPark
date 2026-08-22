#!/usr/bin/env python3
"""Aggregate fail-closed gates for M-018 product-source correction P."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
INSTALLED_DIRECTORIES = ("Core", "Effects", "Analysis", "IO", "Music")
EXPECTED_INSTALLED_HEADERS = 102
EXPECTED_ORDINARY_TESTS = 885
EXPECTED_PACKAGE_HASHES = {
    "packaging/conan/conanfile.py": "796ddfeb9e73dbc6815bd4f5ab813116c8d484f3bf0b403a7cf5ec2811b0a0b3",
    "packaging/vcpkg/portfile.cmake": "24671f71c4e248f2922d114bc43951938eda12ef961e8d95b590d71d95dbe8ea",
    "packaging/vcpkg/vcpkg.json": "dc1cc44137d375143c531a06ae690e2c215f7b3a4ed0150d6d25dd254e833c0e",
}
EXPECTED_SEMANTIC_HASHES = {
    "packaging/conan/conanfile.py": "a5ca52184e29c51b093507a8855122ec38e766469bdb69207931f267cbfdaa17",
    "packaging/vcpkg/portfile.cmake": "7e8ad3376e35622215041bd7cb3b855cef1ef5f408485cd507a6a8f3bd64d1b5",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(command: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False
    )


def parse_doxyfile(text: str) -> dict[str, str]:
    logical: list[str] = []
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        continued = line.endswith("\\")
        if continued:
            line = line[:-1]
        pending += (" " if pending else "") + line.strip()
        if not continued:
            if pending and not pending.startswith("#"):
                logical.append(pending)
            pending = ""
    values: dict[str, str] = {}
    for line in logical:
        match = re.match(r"^([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def split_doxygen_words(value: str) -> list[str]:
    return re.findall(r'"[^"]*"|\S+', value)


def covered(path: str, inputs: Iterable[str], recursive: bool) -> bool:
    for item in inputs:
        item = item.strip('"').rstrip("/")
        if path == item:
            return True
        if recursive and path.startswith(item + "/"):
            return True
    return False


def doxyfile_errors(root: Path, text: str) -> list[str]:
    config = parse_doxyfile(text)
    errors: list[str] = []
    required = {
        "WARN_AS_ERROR": "YES",
        "WARN_IF_DOC_ERROR": "YES",
        "WARN_IF_INCOMPLETE_DOC": "YES",
        "WARN_NO_PARAMDOC": "YES",
        "RECURSIVE": "YES",
        "EXTRACT_ALL": "YES",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            errors.append(f"DOXYGEN_CONFIG {key} expected {expected}")
    # EXTRACT_ALL plus the exact header census is the independent enumeration
    # that prevents an undocumented declaration/header from being omitted.
    if config.get("WARN_IF_UNDOCUMENTED") != "NO":
        errors.append("DOXYGEN_CONFIG WARN_IF_UNDOCUMENTED expected NO with census")

    inputs = split_doxygen_words(config.get("INPUT", ""))
    excludes = split_doxygen_words(config.get("EXCLUDE_PATTERNS", ""))
    recursive = config.get("RECURSIVE") == "YES"
    headers = ["DSPark.h"]
    for directory in INSTALLED_DIRECTORIES:
        headers.extend(
            path.relative_to(root).as_posix()
            for path in sorted((root / directory).rglob("*.h"))
        )
    if len(headers) != EXPECTED_INSTALLED_HEADERS:
        errors.append(
            f"INSTALLED_HEADER_COUNT expected {EXPECTED_INSTALLED_HEADERS} got {len(headers)}"
        )
    for header in headers:
        if not covered(header, inputs, recursive):
            errors.append(f"DOXYGEN_HEADER_EXCLUDED {header}")
        if any(fnmatch.fnmatch("/" + header, pattern) for pattern in excludes):
            errors.append(f"DOXYGEN_HEADER_EXCLUDED {header}")

    tracked_markdown = run(["git", "ls-files", "*.md"], root).stdout.splitlines()
    for markdown in tracked_markdown:
        if not covered(markdown, inputs, recursive):
            errors.append(f"DOXYGEN_MARKDOWN_EXCLUDED {markdown}")
    if not any("docs/api" in pattern for pattern in excludes):
        errors.append("DOXYGEN_GENERATED_OUTPUT_NOT_EXCLUDED")
    return errors


def conan_semantics(data: bytes) -> bytes:
    tree = ast.parse(data.decode("utf-8"))
    return ast.dump(tree, annotate_fields=True, include_attributes=False).encode("ascii")


def cmake_noncomment_semantics(data: bytes) -> bytes:
    canonical: list[str] = []
    for source_line in data.decode("utf-8").splitlines():
        # This recipe has no quoted '#'; the exact-file hash below prevents a
        # future syntax change from silently broadening this canonicalizer.
        line = source_line.split("#", 1)[0]
        line = " ".join(line.split())
        if line:
            canonical.append(line)
    return "\n".join(canonical).encode("utf-8")


def package_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path, expected in EXPECTED_PACKAGE_HASHES.items():
        actual = digest((root / path).read_bytes())
        if actual != expected:
            errors.append(f"PACKAGE_P_BYTE_DRIFT {path} {actual}")
    semantic_functions = {
        "packaging/conan/conanfile.py": conan_semantics,
        "packaging/vcpkg/portfile.cmake": cmake_noncomment_semantics,
    }
    for path, function in semantic_functions.items():
        actual = digest(function((root / path).read_bytes()))
        if actual != EXPECTED_SEMANTIC_HASHES[path]:
            errors.append(f"PACKAGE_P_SEMANTIC_DRIFT {path} {actual}")

    conan = (root / "packaging/conan/conanfile.py").read_text(encoding="ascii")
    port = (root / "packaging/vcpkg/portfile.cmake").read_text(encoding="ascii")
    required_stale = (
        ('version = "1.2.2"', conan),
        ('"cmake_target_name", "DSPark::DSPark"', conan),
        ("REF v1.4.1", port),
    )
    for marker, content in required_stale:
        if marker not in content:
            errors.append(f"PACKAGE_R_SEMANTIC_CHANGED_PREMATURELY {marker}")
    return errors


STALE_PUBLIC_PHRASES = {
    "examples/plugin_saturator/README.md": ("editor layer is on the roadmap",),
    "examples/plugin_webview_editor/README.md": ("embedding path is not wired",),
    "examples/plugin_webview_editor/webview_saturator.cpp": ("generic host UI):",),
    ".github/workflows/ci.yml": ("stub on Linux", "grown to 694 cases"),
}


def stale_truth_errors_for_text(path: str, content: str) -> list[str]:
    return [
        f"STALE_PUBLIC_TRUTH {path} {phrase}"
        for phrase in STALE_PUBLIC_PHRASES.get(path, ())
        if phrase in content
    ]


def stale_truth_errors(root: Path) -> list[str]:
    errors: list[str] = []
    for path in STALE_PUBLIC_PHRASES:
        content = (root / path).read_text(encoding="ascii")
        errors.extend(stale_truth_errors_for_text(path, content))
    ci = (root / ".github/workflows/ci.yml").read_text(encoding="ascii")
    if "ordinary suite authority is currently 885" not in ci:
        errors.append("CURRENT_TEST_AUTHORITY_MISSING 885")
    return errors


def reverb_errors(root: Path) -> list[str]:
    source = (root / "Effects/Reverb.h").read_text(encoding="ascii")
    errors: list[str] = []
    required = (
        "ReverbBankPublisher", "std::array<BankSlot, 4>",
        "pendingToken_.exchange", "Phase::exhausted", "pinLatest",
        "publicationMetadata_", "std::unique_ptr<Bank>",
    )
    for marker in required:
        if marker not in source:
            errors.append(f"REVERB_PROTOCOL_MARKER_MISSING {marker}")
    for marker in ("std::shared_ptr", "atomic_flag", "while (bankLock_"):
        if marker in source:
            errors.append(f"REVERB_FORBIDDEN_OWNERSHIP {marker}")
    tests = (root / "tests/TestReverbPublication.cpp").read_text(encoding="ascii")
    for test_id in (
        "T-M018-reverb-slot-state-machine",
        "T-M018-reverb-fixed-audio-operation-bound",
        "T-M018-reverb-publisher-phase-parking",
        "T-M018-reverb-no-audio-allocation-or-final-release",
        "T-M018-reverb-premature-reuse-mutant",
        "T-M018-reverb-publisher-starvation-mutant",
        "T-M018-reverb-generation-aba",
        "T-M018-reverb-getconvolver-pin-lifetime",
        "T-M018-reverb-reset-metadata-exception-shutdown",
        "T-M018-reverb-retained-bank-memory-bound",
        "T-M018-reverb-race-sanitizer-matrix",
    ):
        if test_id not in tests:
            errors.append(f"REVERB_NAMED_TEST_MISSING {test_id}")
    cmake = (root / "tests/CMakeLists.txt").read_text(encoding="ascii")
    suite_block = cmake.split("add_executable(dspark_tests", 1)[1].split(")", 1)[0]
    if "TestReverbPublication.cpp" in suite_block:
        errors.append("ORDINARY_885_AUTHORITY_CHANGED")
    if "add_executable(dspark_reverb_publication TestReverbPublication.cpp)" not in cmake:
        errors.append("REVERB_DEDICATED_TARGET_MISSING")
    return errors


def run_doxygen_param_mutant(doxygen: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="dspark-doxygen-mutant-") as directory:
        root = Path(directory)
        (root / "subject.h").write_text(
            "/** @file subject.h */\n"
            "/** @brief Function with one intentionally undocumented argument.\n"
            " * @param documented The documented argument. */\n"
            "inline void subject(int documented, int missing) "
            "{ (void)documented; (void)missing; }\n",
            encoding="ascii",
        )
        (root / "Doxyfile").write_text(
            "PROJECT_NAME=X\nINPUT=subject.h\nGENERATE_HTML=NO\n"
            "GENERATE_LATEX=NO\nQUIET=YES\nEXTRACT_ALL=YES\n"
            "WARN_IF_DOC_ERROR=YES\nWARN_IF_INCOMPLETE_DOC=YES\n"
            "WARN_NO_PARAMDOC=YES\nWARN_AS_ERROR=YES\n",
            encoding="ascii",
        )
        result = run([doxygen, "Doxyfile"], root)
        return result.returncode != 0 and "not documented" in result.stdout


def self_test(root: Path, doxygen: str | None) -> int:
    config = (root / "Doxyfile").read_text(encoding="ascii")
    checks: list[tuple[str, bool]] = []
    checks.append(("warning-as-error-disabled", bool(doxyfile_errors(root, config.replace("WARN_AS_ERROR          = YES", "WARN_AS_ERROR          = NO")))))
    checks.append(("public-header-excluded", bool(doxyfile_errors(root, config.replace("DSPark.h Core", "DSPark.h")))))
    ci = (root / ".github/workflows/ci.yml").read_text(encoding="ascii")
    stale_mutants = (
        (
            "roadmap-phrase-restored",
            "examples/plugin_saturator/README.md",
            "editor layer is on the roadmap",
        ),
        (
            "linux-not-wired-restored",
            "examples/plugin_webview_editor/README.md",
            "embedding path is not wired",
        ),
        (
            "generic-host-ui-restored",
            "examples/plugin_webview_editor/webview_saturator.cpp",
            "generic host UI):",
        ),
        (
            "linux-stub-restored",
            ".github/workflows/ci.yml",
            "stub on Linux",
        ),
        (
            "suite-694-restored",
            ".github/workflows/ci.yml",
            "grown to 694 cases",
        ),
    )
    for name, path, phrase in stale_mutants:
        original = ci if path == ".github/workflows/ci.yml" else (
            root / path
        ).read_text(encoding="ascii")
        checks.append((
            name,
            bool(stale_truth_errors_for_text(path, original + "\n" + phrase + "\n")),
        ))
    mutated_conan = (root / "packaging/conan/conanfile.py").read_bytes().replace(b'1.2.2', b'1.7.0', 1)
    checks.append(("premature-package-semantics", digest(conan_semantics(mutated_conan)) != EXPECTED_SEMANTIC_HASHES["packaging/conan/conanfile.py"]))
    if doxygen:
        checks.append(("undocumented-public-parameter", run_doxygen_param_mutant(doxygen)))
    else:
        print("UNAVAILABLE mutant undocumented-public-parameter (no Doxygen)")
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} mutant {name}")
    return 0 if all(passed for _name, passed in checks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--skip-public-text", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--doxygen", default=shutil.which("doxygen"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.self_test:
        return self_test(root, arguments.doxygen)

    errors: list[str] = []
    errors.extend(doxyfile_errors(root, (root / "Doxyfile").read_text(encoding="ascii")))
    errors.extend(package_errors(root))
    errors.extend(stale_truth_errors(root))
    errors.extend(reverb_errors(root))
    if not arguments.skip_public_text:
        result = run([sys.executable, "tools/check_public_text.py"], root)
        if result.returncode != 0:
            errors.append("PUBLIC_TEXT_GATE\n" + result.stdout.rstrip())
    for error in errors:
        print(f"ERROR {error}", file=sys.stderr)
    if errors:
        return 1
    print(
        f"PASS M-018 P aggregate: {EXPECTED_INSTALLED_HEADERS} installed headers, "
        f"ordinary suite authority {EXPECTED_ORDINARY_TESTS}, package semantics held for R"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
