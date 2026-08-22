#!/usr/bin/env python3
"""Aggregate fail-closed gates for global product-source corrections."""

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
        "HIDE_UNDOC_CLASSES": "YES",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            errors.append(f"DOXYGEN_CONFIG {key} expected {expected}")
    # EXTRACT_ALL plus the exact header census is the independent enumeration
    # that prevents an undocumented declaration/header from being omitted.
    if config.get("WARN_IF_UNDOCUMENTED") != "NO":
        errors.append("DOXYGEN_CONFIG WARN_IF_UNDOCUMENTED expected NO with census")
    if config.get("USE_MDFILE_AS_MAINPAGE") != "./README.md":
        errors.append(
            "DOXYGEN_CONFIG USE_MDFILE_AS_MAINPAGE expected ./README.md"
        )

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
        "Resampler<T> resampler", "resampler.prepare(effIrRate",
        "resampler.getMaxOutputSamples(irLen)", "resampler.processBlock(",
    )
    for marker in required:
        if marker not in source:
            errors.append(f"REVERB_PROTOCOL_MARKER_MISSING {marker}")
    for marker in (
        "std::shared_ptr", "atomic_flag", "while (bankLock_",
        "resampleImpulseResponse", "reverbBesselI0",
        "same 32-tap/256-phase", "first-use history resize",
    ):
        if marker in source:
            errors.append(f"REVERB_FORBIDDEN_IMPLEMENTATION {marker}")
    anchor_count = source.count(LIVE_REVERB_MUTATION_ANCHOR)
    if anchor_count != 1:
        errors.append(
            f"REVERB_LIVE_MUTATION_ANCHOR expected 1 got {anchor_count}"
        )
    tests = (root / "tests/TestReverbPublication.cpp").read_text(encoding="ascii")
    for test_id in (
        "reverb-slot-state-machine",
        "reverb-fixed-audio-operation-bound",
        "reverb-publisher-phase-parking",
        "reverb-no-audio-allocation-or-final-release",
        "reverb-premature-reuse-mutant",
        "reverb-publisher-starvation-mutant",
        "reverb-generation-aba",
        "reverb-getconvolver-pin-lifetime",
        "reverb-reset-metadata-exception-shutdown",
        "reverb-retained-bank-memory-bound",
        "reverb-race-sanitizer-matrix",
    ):
        if test_id not in tests:
            errors.append(f"REVERB_NAMED_TEST_MISSING {test_id}")
    for control in (
        "generation-exhausted Reverb did not report no-capacity",
        "for (std::ptrdiff_t failurePoint = 0;; ++failurePoint)",
        "failed setState changed serialized state bytes",
        "failed setState changed fixed rendered PCM bytes",
    ):
        if control not in tests:
            errors.append(f"REVERB_TRANSACTION_CONTROL_MISSING {control}")
    for detached in (
        "TransactionMutantState", "earlyMixCommitMutantIsDetected",
        "earlyPreDelayCommitMutantIsDetected",
        "partialShapingPublicationMutantIsDetected",
    ):
        if detached in tests:
            errors.append(f"REVERB_DETACHED_MUTANT_HELPER {detached}")
    cmake = (root / "tests/CMakeLists.txt").read_text(encoding="ascii")
    suite_block = cmake.split("add_executable(dspark_tests", 1)[1].split(")", 1)[0]
    if "TestReverbPublication.cpp" in suite_block:
        errors.append("ORDINARY_885_AUTHORITY_CHANGED")
    if "add_executable(dspark_reverb_publication TestReverbPublication.cpp)" not in cmake:
        errors.append("REVERB_DEDICATED_TARGET_MISSING")
    return errors


def doxygen_param_config(generate_xml: bool = True,
                         warn_no_paramdoc: bool = True) -> str:
    return (
        "PROJECT_NAME=X\nINPUT=subject.h\nOUTPUT_DIRECTORY=out\n"
        "GENERATE_HTML=NO\n"
        f"GENERATE_XML={'YES' if generate_xml else 'NO'}\n"
        "XML_OUTPUT=xml\nGENERATE_LATEX=NO\nQUIET=YES\nEXTRACT_ALL=YES\n"
        "WARN_IF_UNDOCUMENTED=NO\nWARN_IF_DOC_ERROR=YES\n"
        "WARN_IF_INCOMPLETE_DOC=YES\n"
        f"WARN_NO_PARAMDOC={'YES' if warn_no_paramdoc else 'NO'}\n"
        "WARN_AS_ERROR=YES\n"
    )


def doxygen_param_contract_errors(config_text: str) -> list[str]:
    config = parse_doxyfile(config_text)
    required = {
        "GENERATE_HTML": "NO",
        "GENERATE_XML": "YES",
        "XML_OUTPUT": "xml",
        "WARN_IF_DOC_ERROR": "YES",
        "WARN_IF_INCOMPLETE_DOC": "YES",
        "WARN_NO_PARAMDOC": "YES",
        "WARN_AS_ERROR": "YES",
    }
    return [
        f"{key} expected {expected}"
        for key, expected in required.items()
        if config.get(key) != expected
    ]


def run_doxygen_param_subject(doxygen: str, documented: bool,
                              config_text: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="dspark-doxygen-mutant-") as directory:
        root = Path(directory)
        missing_doc = (
            " * @param missing The second documented argument.\n"
            if documented else ""
        )
        (root / "subject.h").write_text(
            "/** @file subject.h */\n"
            "/** @brief Function with two arguments.\n"
            " * @param documented The first documented argument.\n"
            + missing_doc
            + " */\n"
            "inline void subject(int documented, int missing) "
            "{ (void)documented; (void)missing; }\n",
            encoding="ascii",
        )
        (root / "Doxyfile").write_text(config_text, encoding="ascii")
        result = run([doxygen, "Doxyfile"], root)
        contract_ok = not doxygen_param_contract_errors(config_text)
        xml_exists = (root / "out" / "xml" / "index.xml").is_file()
        if documented:
            valid = contract_ok and result.returncode == 0 and xml_exists
        else:
            diagnostic = "parameter 'missing'" in result.stdout \
                and "not documented" in result.stdout
            # Newer Doxygen aborts before flushing index.xml when a warning is
            # fatal; 1.9.8 leaves a partial XML tree. The configured real-output
            # contract, intended diagnostic and nonzero exit are the portable
            # oracle. The documented positive independently proves XML emission.
            valid = contract_ok and result.returncode != 0 and diagnostic
        return valid, result.stdout


def run_doxygen_duplicate_mainpage_mutant(doxygen: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="dspark-doxygen-mainpage-") as directory:
        root = Path(directory)
        (root / "one.md").write_text(
            "\\mainpage First\n\nFirst page.\n", encoding="ascii"
        )
        (root / "two.md").write_text(
            "\\mainpage Second\n\nSecond page.\n", encoding="ascii"
        )
        (root / "Doxyfile").write_text(
            "PROJECT_NAME=X\nINPUT=one.md two.md\nOUTPUT_DIRECTORY=out\n"
            "GENERATE_HTML=YES\nGENERATE_LATEX=NO\nQUIET=YES\n"
            "WARN_IF_DOC_ERROR=YES\nWARN_AS_ERROR=YES\n",
            encoding="ascii",
        )
        result = run([doxygen, "Doxyfile"], root)
        return result.returncode != 0 \
            and "more than one" in result.stdout \
            and "mainpage" in result.stdout


LIVE_REVERB_MUTATION_ANCHOR = (
    "        // Candidate construction owns every throwing operation. Publisher\n"
    "        // capacity is resolved before its first slot/scalar mutation; once the\n"
    "        // commit begins, only unique_ptr moves and atomic/plain no-throw stores\n"
    "        // remain.\n"
    "        auto candidate = buildBank(irStorage_, irLength_, irChannels_,\n"
)

LIVE_REVERB_MUTATIONS = (
    (
        "early-mix-commit",
        "        mix_.store(mix, std::memory_order_relaxed);",
    ),
    (
        "early-pre-delay-commit",
        "        preDelayMs_.store(preDelay, std::memory_order_relaxed);\n"
        "        preDelaySamples_.store(preDelaySamples, std::memory_order_relaxed);",
    ),
    (
        "partial-shaping-publication-commit",
        "        decayScale_.store(ds, std::memory_order_relaxed);",
    ),
)

REQUIRED_COMPILER_CANDIDATES = {
    "gcc": ("g++-13", "g++"),
    "clang": ("clang++-18", "clang++"),
}


def compiler_family(version: str) -> str | None:
    lowered = version.lower()
    if "clang" in lowered:
        return "clang"
    if "g++" in lowered or "gcc" in lowered:
        return "gcc"
    return None


def discover_required_compilers() -> tuple[
    dict[str, tuple[str, str]], dict[str, str]
]:
    compilers: dict[str, tuple[str, str]] = {}
    failures: dict[str, str] = {}
    for required_family, candidates in REQUIRED_COMPILER_CANDIDATES.items():
        attempts: list[str] = []
        for candidate in candidates:
            executable = shutil.which(candidate)
            if executable is None:
                attempts.append(f"{candidate}: not found")
                continue
            version_result = run([executable, "--version"])
            first_line = version_result.stdout.splitlines()[0] \
                if version_result.stdout.splitlines() else "no version output"
            actual_family = compiler_family(first_line)
            attempts.append(
                f"{candidate}: {first_line} (classified {actual_family})"
            )
            if version_result.returncode == 0 and actual_family == required_family:
                compilers[required_family] = (executable, first_line)
                break
        if required_family not in compilers:
            failures[required_family] = "; ".join(attempts)
    return compilers, failures


def copy_reverb_subject(root: Path, destination: Path) -> None:
    shutil.copytree(root / "Core", destination / "Core")
    shutil.copytree(root / "IO", destination / "IO")
    (destination / "Effects").mkdir()
    (destination / "tests").mkdir()
    shutil.copy2(root / "Effects/Reverb.h", destination / "Effects/Reverb.h")
    shutil.copy2(
        root / "tests/TestReverbPublication.cpp",
        destination / "tests/TestReverbPublication.cpp",
    )


def apply_live_reverb_mutation(destination: Path, insertion: str) -> str | None:
    header = destination / "Effects/Reverb.h"
    source = header.read_text(encoding="ascii")
    count = source.count(LIVE_REVERB_MUTATION_ANCHOR)
    if count != 1:
        return f"production anchor count expected 1, got {count}"
    replacement = LIVE_REVERB_MUTATION_ANCHOR.replace(
        "        auto candidate = buildBank(irStorage_, irLength_, irChannels_,\n",
        insertion
        + "\n        auto candidate = buildBank(irStorage_, irLength_, irChannels_,\n",
    )
    header.write_text(
        source.replace(LIVE_REVERB_MUTATION_ANCHOR, replacement, 1),
        encoding="ascii",
    )
    return None


def compile_and_run_reverb_subject(
    compiler: str,
    family: str,
    source_root: Path,
    binary_name: str,
    expect_transaction_red: bool,
) -> tuple[bool, str]:
    binary = source_root / binary_name
    command = [
        compiler,
        "-std=c++20",
        "-O1",
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Werror",
        "-DDSPARK_REVERB_TEST_GENERATION_MAX=1",
        "-pthread",
        "-I",
        str(source_root),
        str(source_root / "tests/TestReverbPublication.cpp"),
        "-o",
        str(binary),
    ]
    if family == "gcc":
        command.insert(6, "-Wno-mismatched-new-delete")
    try:
        built = subprocess.run(
            command,
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=180,
        )
    except subprocess.TimeoutExpired as error:
        return False, f"build timed out: {error}"
    if built.returncode != 0:
        return False, "build failed:\n" + built.stdout[-8000:]
    try:
        executed = subprocess.run(
            [str(binary)],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        return False, f"execution timed out: {error}"
    if expect_transaction_red:
        expected = "failed setState changed serialized state bytes"
        passed = executed.returncode != 0 and expected in executed.stdout
        if passed:
            return True, (
                f"expected transaction RED rc={executed.returncode}: {expected}"
            )
        return False, (
            f"expected transaction RED missing; rc={executed.returncode}\n"
            + executed.stdout[-8000:]
        )
    passed = executed.returncode == 0 and "11 checks, 0 failures" in executed.stdout
    if passed:
        return True, "baseline dedicated Reverb subject passed 11 checks"
    return False, (
        f"baseline dedicated Reverb subject failed; rc={executed.returncode}\n"
        + executed.stdout[-8000:]
    )


def live_reverb_mutation_checks(
    root: Path,
) -> tuple[list[tuple[str, bool]], dict[str, str]]:
    checks: list[tuple[str, bool]] = []
    details: dict[str, str] = {}
    compilers, discovery_failures = discover_required_compilers()
    for family in REQUIRED_COMPILER_CANDIDATES:
        name = f"live-reverb-{family}-compiler-discovery"
        passed = family in compilers
        checks.append((name, passed))
        details[name] = compilers[family][1] if passed \
            else discovery_failures.get(family, "no discovery evidence")

    with tempfile.TemporaryDirectory(prefix="dspark-live-reverb-mutants-") as directory:
        scratch = Path(directory)
        baseline = scratch / "baseline"
        copy_reverb_subject(root, baseline)

        mutation_roots: dict[str, Path] = {}
        for mutation_name, insertion in LIVE_REVERB_MUTATIONS:
            mutation_root = scratch / mutation_name
            copy_reverb_subject(root, mutation_root)
            mutation_error = apply_live_reverb_mutation(mutation_root, insertion)
            if mutation_error is not None:
                details[f"mutation-source-{mutation_name}"] = mutation_error
            mutation_roots[mutation_name] = mutation_root

        for family in REQUIRED_COMPILER_CANDIDATES:
            if family not in compilers:
                continue
            compiler, _version = compilers[family]
            baseline_name = f"live-reverb-baseline-{family}"
            baseline_passed, baseline_detail = compile_and_run_reverb_subject(
                compiler, family, baseline, f"subject-{family}", False
            )
            checks.append((baseline_name, baseline_passed))
            details[baseline_name] = baseline_detail

            for mutation_name, _insertion in LIVE_REVERB_MUTATIONS:
                check_name = f"live-reverb-{mutation_name}-{family}"
                source_error = details.get(f"mutation-source-{mutation_name}")
                if source_error is not None:
                    checks.append((check_name, False))
                    details[check_name] = source_error
                    continue
                passed, detail = compile_and_run_reverb_subject(
                    compiler,
                    family,
                    mutation_roots[mutation_name],
                    f"subject-{family}",
                    True,
                )
                checks.append((check_name, passed))
                details[check_name] = detail

    expected_check_count = 10
    cardinality_name = "live-reverb-execution-matrix-cardinality"
    cardinality_passed = len(checks) == expected_check_count
    details[cardinality_name] = (
        f"expected {expected_check_count} discovery/build/run checks, got {len(checks)}"
    )
    checks.append((cardinality_name, cardinality_passed))
    return checks, details


def self_test(root: Path, doxygen: str | None) -> int:
    config = (root / "Doxyfile").read_text(encoding="ascii")
    checks: list[tuple[str, bool]] = []
    details: dict[str, str] = {}
    checks.append(("warning-as-error-disabled", bool(doxyfile_errors(root, config.replace("WARN_AS_ERROR          = YES", "WARN_AS_ERROR          = NO")))))
    checks.append(("public-header-excluded", bool(doxyfile_errors(root, config.replace("DSPark.h Core", "DSPark.h")))))
    checks.append(("required-public-markdown-excluded", bool(doxyfile_errors(root, config.replace(" examples/README.md ", " ")))))
    checks.append(("duplicate-mainpage-basename-restored", bool(doxyfile_errors(root, config.replace("USE_MDFILE_AS_MAINPAGE = ./README.md", "USE_MDFILE_AS_MAINPAGE = README.md")))))
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
        checks.append((
            "duplicate-mainpage-restored",
            run_doxygen_duplicate_mainpage_mutant(doxygen),
        ))
        parameter_config = doxygen_param_config()
        undocumented, _undocumented_log = run_doxygen_param_subject(
            doxygen, False, parameter_config
        )
        documented, _documented_log = run_doxygen_param_subject(
            doxygen, True, parameter_config
        )
        output_disabled, _output_disabled_log = run_doxygen_param_subject(
            doxygen, False, doxygen_param_config(generate_xml=False)
        )
        warning_disabled, _warning_disabled_log = run_doxygen_param_subject(
            doxygen, False, doxygen_param_config(warn_no_paramdoc=False)
        )
        checks.append(("undocumented-public-parameter", undocumented))
        checks.append(("documented-public-parameter-positive", documented))
        checks.append(("output-disabled-mutant-rejected", not output_disabled))
        checks.append(("parameter-warning-disabled-mutant-rejected", not warning_disabled))
    else:
        print("UNAVAILABLE mutant undocumented-public-parameter (no Doxygen)")
    live_checks, live_details = live_reverb_mutation_checks(root)
    checks.extend(live_checks)
    details.update(live_details)
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} mutant {name}")
        if name in details:
            print(details[name])
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
        f"PASS global product corrections: {EXPECTED_INSTALLED_HEADERS} installed headers, "
        f"ordinary suite authority {EXPECTED_ORDINARY_TESTS}, "
        "later package-revision semantics preserved"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
