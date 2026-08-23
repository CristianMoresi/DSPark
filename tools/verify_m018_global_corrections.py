#!/usr/bin/env python3
"""Aggregate fail-closed gates for global product-source corrections."""

from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
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
    "        auto candidate = buildBank(irStorage_, irLength_, irChannels_,\n"
    "                                   irSampleRate_, spec_, fftBlockSize_, ds, st);\n"
)
LIVE_REVERB_BASELINE_HASH = (
    "294a1053756adc86d84ab22240afdc4a55a706bef5e09a890ba6e084f4f6a80f"
)
LIVE_REVERB_SUBJECT_HASH = (
    "ad203837e0ca53a8da3af2868d1c349e2e340fac3baed5a46371295d350d3e2b"
)
LIVE_REVERB_ANCHOR_HASH = (
    "6cf8d8b452439b4a8b12e6a4e3cf68c962f315bb4fcc4060af68f4aad031c616"
)
LIVE_REVERB_VARIANTS = (
    {
        "id": "baseline",
        "insertion": "",
        "source_sha256": LIVE_REVERB_BASELINE_HASH,
        "patch_sha256": hashlib.sha256(b"").hexdigest(),
        "oracle": "GREEN",
    },
    {
        "id": "early-mix",
        "insertion": "        mix_.store(mix, std::memory_order_relaxed);\n",
        "source_sha256": "a92a17baa0dee3618acf0a0de08a2e70b74fe479c73fd0bbf53c7dde4ec6ccfa",
        "patch_sha256": "39400ede7f92730ec68e02599d8d0c125e066b571573422d036aaefaaa3222e9",
        "oracle": "EXPECTED_RED",
    },
    {
        "id": "early-predelay",
        "insertion": (
            "        preDelayMs_.store(preDelay, std::memory_order_relaxed);\n"
            "        preDelaySamples_.store(preDelaySamples, std::memory_order_relaxed);\n"
        ),
        "source_sha256": "2bfc9fc8137ea9c214c740e0e953b4c1f9a5e13b096c5fcd0a4967c64cca23b6",
        "patch_sha256": "42f1bb61861e677382493a2675f5d387959dab2af11aabca81bc1572ce977aff",
        "oracle": "EXPECTED_RED",
    },
    {
        "id": "partial-shaping-publication",
        "insertion": "        decayScale_.store(ds, std::memory_order_relaxed);\n",
        "source_sha256": "1f6af93ae51eeaead1cb4c929641ecaa7817d687b36ee7ff7fb6441cce6b2010",
        "patch_sha256": "557fbc4018cef7d9f91ffc8c9aee972a0b7218e24b210238f25f33834a8fdadc",
        "oracle": "EXPECTED_RED",
    },
)
LIVE_COMPILER_SPECS = (
    ("gcc13", "gcc", 13, ("g++-13", "g++")),
    ("clang18", "clang", 18, ("clang++-18", "clang++")),
)
LIVE_PROFILES = (
    ("normal", ()),
    (
        "sanitizer",
        (
            "-fno-omit-frame-pointer",
            "-fsanitize=address,undefined,float-cast-overflow",
            "-fno-sanitize-recover=all",
        ),
    ),
)
VERSION_TIMEOUT_SECONDS = 5
COMPILE_TIMEOUT_SECONDS = 240
RUN_TIMEOUT_SECONDS = 180
PROCESS_GRACE_SECONDS = 1
PROCESS_CLEANUP_SECONDS = 2
MAX_CAPTURE_BYTES = 1024 * 1024
BASELINE_STDOUT_HASH = (
    "eafa64837b60b5c6e9549451a31a60eae06c9ae7e1c5668be64317c799a004b3"
)
MUTANT_STDOUT_HASH = (
    "841c51be4ed070ceee4c4821c4fc60f89a273d7e30051bf8c7564f7282a3fd40"
)
MUTANT_STDERR = (
    "FAIL reverb-reset-metadata-exception-shutdown: "
    "failed setState changed serialized state bytes\n"
)
MUTANT_STDERR_HASH = (
    "0f05f2094aec4099a5a1d7bc0de2b35f6dfeffa1ec4c0bafc4aa88da1376900b"
)
EMPTY_HASH = hashlib.sha256(b"").hexdigest()
SANITIZER_DIAGNOSTICS = (
    "ERROR: AddressSanitizer",
    "runtime error:",
    "LeakSanitizer",
)


def process_group_members(process_group: int) -> list[int]:
    members: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return members
    for item in proc.iterdir():
        if not item.name.isdigit():
            continue
        try:
            stat_text = (item / "stat").read_text(encoding="ascii")
            fields = stat_text[stat_text.rfind(")") + 2:].split()
            if fields[0] != "Z" and int(fields[2]) == process_group:
                members.append(int(item.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return sorted(members)


def signal_process_group(process_group: int, value: signal.Signals) -> None:
    try:
        os.killpg(process_group, value)
    except ProcessLookupError:
        pass


def run_owned(command: list[str], cwd: Path, timeout: float, phase: str,
              environment: dict[str, str] | None = None) -> dict[str, object]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        shell=False,
    )
    process_group = process.pid
    timed_out = False
    cleanup_actions: list[str] = []
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_actions.append("SIGTERM_PROCESS_GROUP")
        signal_process_group(process_group, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(
                timeout=PROCESS_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup_actions.append("SIGKILL_PROCESS_GROUP")
            signal_process_group(process_group, signal.SIGKILL)
            stdout, stderr = process.communicate()

    residue_detected = process_group_members(process_group)
    if residue_detected:
        cleanup_actions.append("SIGTERM_RESIDUAL_PROCESS_GROUP")
        signal_process_group(process_group, signal.SIGTERM)
        time.sleep(PROCESS_GRACE_SECONDS)
        if process_group_members(process_group):
            cleanup_actions.append("SIGKILL_RESIDUAL_PROCESS_GROUP")
            signal_process_group(process_group, signal.SIGKILL)
    deadline = time.monotonic() + PROCESS_CLEANUP_SECONDS
    residue = process_group_members(process_group)
    while residue and time.monotonic() < deadline:
        time.sleep(0.02)
        residue = process_group_members(process_group)

    stdout_too_large = len(stdout) > MAX_CAPTURE_BYTES
    stderr_too_large = len(stderr) > MAX_CAPTURE_BYTES
    stdout = stdout[:MAX_CAPTURE_BYTES]
    stderr = stderr[:MAX_CAPTURE_BYTES]
    errors: list[str] = []
    if timed_out:
        errors.append(phase.upper() + "_TIMEOUT")
    if residue_detected or residue:
        errors.append("PROCESS_TREE_CLEANUP_FAILED")
    if stdout_too_large:
        errors.append("STDOUT_LIMIT_EXCEEDED")
    if stderr_too_large:
        errors.append("STDERR_LIMIT_EXCEEDED")
    return {
        "command": command,
        "phase": phase,
        "timeout_seconds": timeout,
        "duration_seconds": round(time.monotonic() - started, 6),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout.decode("utf-8", "replace"),
        "stderr": stderr.decode("utf-8", "replace"),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_size": len(stdout),
        "stderr_size": len(stderr),
        "cleanup_actions": cleanup_actions,
        "residue_detected": residue_detected,
        "cleanup_residue": residue,
        "errors": errors,
    }


GCC_DRIVER_TOKEN = (
    r"(?<![A-Za-z0-9_+.-])"
    r"(?:g\+\+(?:-\d+)?|gcc(?:-\d+)?)"
    r"(?![A-Za-z0-9_+.-])"
)


def parse_compiler_version(first_line: str) -> tuple[str | None, int | None]:
    clang = re.search(r"\bclang version\s+(\d+)(?:\.|\b)", first_line, re.I)
    if clang:
        return "clang", int(clang.group(1))
    if "clang" in first_line.lower():
        return "clang", None
    gcc = re.search(
        GCC_DRIVER_TOKEN + r".*?"
        r"(?:\)\s*|\bversion\s+)?(\d+)\.(\d+)",
        first_line,
        re.I,
    )
    if gcc:
        return "gcc", int(gcc.group(1))
    if re.search(GCC_DRIVER_TOKEN, first_line, re.I):
        return "gcc", None
    return None, None


def parse_dump_version(output: str) -> int | None:
    match = re.fullmatch(r"\s*(\d+)(?:\.\d+){0,3}\s*", output)
    return int(match.group(1)) if match else None


def discover_compiler(label: str, family: str, major: int,
                      candidates: tuple[str, ...], cwd: Path,
                      resolver: dict[str, str] | None = None) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    for candidate in candidates:
        executable = (resolver or {}).get(candidate) if resolver is not None \
            else shutil.which(candidate)
        if not executable:
            attempts.append({
                "family": family,
                "candidate": candidate,
                "reason": "MISSING_EXECUTABLE",
            })
            continue
        version = run_owned(
            [executable, "--version"], cwd, VERSION_TIMEOUT_SECONDS,
            "version_probe")
        if version["timed_out"] or version["cleanup_residue"] \
                or version["residue_detected"]:
            reason = "VERSION_PROBE_TIMEOUT" if version["timed_out"] \
                else "PROCESS_TREE_CLEANUP_FAILED"
        elif version["exit_code"] != 0:
            reason = "VERSION_PROBE_NONZERO:{}".format(version["exit_code"])
        elif not str(version["stdout"]).splitlines():
            reason = "VERSION_OUTPUT_EMPTY"
        else:
            first_line = str(version["stdout"]).splitlines()[0]
            parsed_family, parsed_major = parse_compiler_version(first_line)
            if parsed_family is None or parsed_major is None:
                reason = "VERSION_OUTPUT_MALFORMED"
            elif parsed_family != family:
                reason = "FAMILY_MISMATCH:{}".format(parsed_family)
            elif parsed_major != major:
                reason = "MAJOR_MISMATCH:{}".format(parsed_major)
            else:
                dumped = run_owned(
                    [executable, "-dumpfullversion", "-dumpversion"], cwd,
                    VERSION_TIMEOUT_SECONDS, "dump_version_probe")
                dump_major = parse_dump_version(str(dumped["stdout"])) \
                    if dumped["exit_code"] == 0 else None
                if dumped["timed_out"] or dumped["cleanup_residue"] \
                        or dumped["residue_detected"]:
                    reason = "DUMP_VERSION_TIMEOUT" if dumped["timed_out"] \
                        else "PROCESS_TREE_CLEANUP_FAILED"
                elif dumped["exit_code"] != 0:
                    reason = "DUMP_VERSION_NONZERO:{}".format(
                        dumped["exit_code"])
                elif dump_major is None:
                    reason = "DUMP_VERSION_MALFORMED"
                elif dump_major != major or dump_major != parsed_major:
                    reason = "DUMP_VERSION_MAJOR_MISMATCH:{}".format(dump_major)
                else:
                    return {
                        "label": label,
                        "status": "PASS",
                        # Preserve the invoked driver spelling. On common
                        # installations clang++ is a symlink to clang; resolving
                        # it before execution silently drops C++ linker-driver
                        # semantics even though both names reach one binary.
                        "path": executable,
                        "resolved_path": str(Path(executable).resolve()),
                        "candidate": candidate,
                        "family": family,
                        "major": major,
                        "version_first_line": first_line,
                        "dump_version": str(dumped["stdout"]).strip(),
                        "version_process": version,
                        "dump_process": dumped,
                        "attempts": attempts,
                    }
                attempts.append({
                    "family": family,
                    "candidate": candidate,
                    "reason": reason,
                    "process": dumped,
                })
                continue
        attempts.append({
            "family": family,
            "candidate": candidate,
            "reason": reason,
            "process": version,
        })
    terminal = (
        "COMPILER_DISCOVERY_FAILED family={} required_major={}; ".format(
            family, major)
        + "; ".join(
            "candidate={} reason={}".format(
                attempt["candidate"], attempt["reason"])
            for attempt in attempts)
    )
    return {
        "label": label,
        "status": "FAIL",
        "required_family": family,
        "required_major": major,
        "attempts": attempts,
        "terminal": terminal,
    }


def validate_copy_source(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.is_dir() or path.is_symlink():
        return ["SOURCE_COPY_ROOT_INVALID " + str(path)]
    for item in path.rglob("*"):
        if item.is_symlink():
            errors.append("SOURCE_COPY_SYMLINK " + str(item))
        elif not item.is_dir() and not item.is_file():
            errors.append("SOURCE_COPY_NONREGULAR " + str(item))
    return errors


def copy_reverb_subject(root: Path, destination: Path) -> list[str]:
    errors = validate_copy_source(root / "Core")
    errors.extend(validate_copy_source(root / "IO"))
    for relative in ("Effects/Reverb.h", "tests/TestReverbPublication.cpp"):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            errors.append("SOURCE_COPY_FILE_INVALID " + relative)
    if errors:
        return errors
    shutil.copytree(root / "Core", destination / "Core")
    shutil.copytree(root / "IO", destination / "IO")
    (destination / "Effects").mkdir()
    (destination / "tests").mkdir()
    shutil.copy2(root / "Effects/Reverb.h", destination / "Effects/Reverb.h")
    shutil.copy2(
        root / "tests/TestReverbPublication.cpp",
        destination / "tests/TestReverbPublication.cpp")
    return []


def prepare_reverb_variants(root: Path, scratch: Path) -> tuple[
    list[dict[str, object]], dict[str, Path], list[str]
]:
    errors: list[str] = []
    baseline_bytes = (root / "Effects/Reverb.h").read_bytes()
    subject_bytes = (root / "tests/TestReverbPublication.cpp").read_bytes()
    baseline = baseline_bytes.decode("ascii")
    if digest(baseline_bytes) != LIVE_REVERB_BASELINE_HASH:
        errors.append("BASELINE_SOURCE_HASH")
    if digest(subject_bytes) != LIVE_REVERB_SUBJECT_HASH:
        errors.append("DEDICATED_SUBJECT_HASH")
    if baseline.count(LIVE_REVERB_MUTATION_ANCHOR) != 1:
        errors.append("MUTATION_ANCHOR_CARDINALITY:{}".format(
            baseline.count(LIVE_REVERB_MUTATION_ANCHOR)))
    if digest(LIVE_REVERB_MUTATION_ANCHOR.encode("ascii")) \
            != LIVE_REVERB_ANCHOR_HASH:
        errors.append("MUTATION_ANCHOR_HASH")

    records: list[dict[str, object]] = []
    roots: dict[str, Path] = {}
    for variant in LIVE_REVERB_VARIANTS:
        variant_id = str(variant["id"])
        insertion = str(variant["insertion"])
        changed = baseline if not insertion else baseline.replace(
            LIVE_REVERB_MUTATION_ANCHOR,
            insertion + LIVE_REVERB_MUTATION_ANCHOR,
            1)
        patch = "".join(difflib.unified_diff(
            baseline.splitlines(keepends=True),
            changed.splitlines(keepends=True),
            fromfile="Effects/Reverb.h@P3",
            tofile="Effects/Reverb.h@{}".format(variant_id),
        ))
        insertion_delta = 0
        if insertion:
            insertion_delta = changed.count(insertion.rstrip()) \
                - baseline.count(insertion.rstrip())
        record = {
            "id": variant_id,
            "insertion_sha256": digest(insertion.encode("ascii")),
            "source_sha256": digest(changed.encode("ascii")),
            "patch_sha256": digest(patch.encode("ascii")),
            "anchor_count_before": baseline.count(LIVE_REVERB_MUTATION_ANCHOR),
            "anchor_count_after": changed.count(LIVE_REVERB_MUTATION_ANCHOR),
            "insertion_delta": insertion_delta,
        }
        records.append(record)
        expected_delta = 0 if variant_id == "baseline" else 1
        if record["source_sha256"] != variant["source_sha256"]:
            errors.append("MUTATED_SOURCE_HASH:" + variant_id)
        if record["patch_sha256"] != variant["patch_sha256"]:
            errors.append("MUTATION_PATCH_HASH:" + variant_id)
        if record["anchor_count_after"] != 1:
            errors.append("POST_MUTATION_ANCHOR_CARDINALITY:" + variant_id)
        if insertion_delta != expected_delta:
            errors.append("MUTATION_INSERTION_DELTA:" + variant_id)

        destination = scratch / variant_id
        errors.extend(copy_reverb_subject(root, destination))
        if destination.is_dir():
            header = destination / "Effects/Reverb.h"
            header.write_text(changed, encoding="ascii")
            if digest(header.read_bytes()) != record["source_sha256"]:
                errors.append("ISOLATED_SOURCE_HASH:" + variant_id)
            roots[variant_id] = destination
    return records, roots, errors


def reverb_row(compiler_record: dict[str, object], profile: str,
               extra_flags: tuple[str, ...], variant: dict[str, object],
               source_record: dict[str, object], source_root: Path) -> dict[str, object]:
    compiler_label = str(compiler_record["label"])
    family = str(compiler_record["family"])
    variant_id = str(variant["id"])
    identity = "reverb::{}::{}::{}".format(
        compiler_label, profile, variant_id)
    binary = source_root / "subject-{}-{}-{}".format(
        compiler_label, profile, variant_id)
    flags = [
        "-std=c++20", "-O1", "-g", *extra_flags,
        "-Wall", "-Wextra", "-Wpedantic", "-Werror",
    ]
    if family == "gcc":
        flags.append("-Wno-mismatched-new-delete")
    command = [
        str(compiler_record["path"]), *flags,
        "-DDSPARK_REVERB_TEST_GENERATION_MAX=1", "-pthread",
        "-I", str(source_root),
        str(source_root / "tests/TestReverbPublication.cpp"),
        "-o", str(binary),
    ]
    build = run_owned(
        command, source_root, COMPILE_TIMEOUT_SECONDS, "compile")
    build_ok = (
        build["exit_code"] == 0 and not build["errors"]
        and binary.is_file() and not binary.is_symlink()
    )
    removed_environment = ("ASAN_OPTIONS", "UBSAN_OPTIONS", "LSAN_OPTIONS")
    set_environment = {"LC_ALL": "C", "LANG": "C"}
    if profile == "sanitizer":
        set_environment.update({
            "ASAN_OPTIONS": "halt_on_error=1:abort_on_error=1:detect_leaks=1",
            "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
        })
    environment_contract = {
        "remove_first": list(removed_environment),
        "set": set_environment,
    }
    execution: dict[str, object] | None = None
    oracle_satisfied = False
    if build_ok:
        environment = dict(os.environ)
        for name in removed_environment:
            environment.pop(name, None)
        environment.update(set_environment)
        execution = run_owned(
            [str(binary)], source_root, RUN_TIMEOUT_SECONDS, "run",
            environment)
        combined = str(execution["stdout"]) + str(execution["stderr"])
        sanitizer_clean = not any(
            marker in combined for marker in SANITIZER_DIAGNOSTICS)
        if variant_id == "baseline":
            oracle_satisfied = (
                execution["exit_code"] == 0
                and execution["stdout_sha256"] == BASELINE_STDOUT_HASH
                and str(execution["stdout"]).endswith("11 checks, 0 failures\n")
                and execution["stderr_sha256"] == EMPTY_HASH
                and not execution["errors"]
                and sanitizer_clean
            )
        else:
            oracle_satisfied = (
                execution["exit_code"] == 1
                and execution["stdout_sha256"] == MUTANT_STDOUT_HASH
                and str(execution["stdout"]).endswith("11 checks, 1 failures\n")
                and execution["stderr_sha256"] == MUTANT_STDERR_HASH
                and execution["stderr"] == MUTANT_STDERR
                and not execution["errors"]
                and sanitizer_clean
            )
    return {
        "id": identity,
        "compiler": compiler_label,
        "profile": profile,
        "variant": variant_id,
        "oracle": variant["oracle"],
        "source_sha256": source_record["source_sha256"],
        "build_ok": build_ok,
        "run_timed_out": bool(execution and execution["timed_out"]),
        "cleanup_residue": [] if execution is None else execution["cleanup_residue"],
        "oracle_satisfied": oracle_satisfied,
        "environment_contract": environment_contract,
        "compile": build,
        "run": execution,
    }


def build_live_transcript(root: Path) -> dict[str, object]:
    transcript: dict[str, object] = {
        "schema": "dspark.global-validation-producer.v1",
        "invocation_count": 1,
        "compiler_discovery": [],
        "source_variants": [],
        "executions": [],
        "errors": [],
    }
    if os.name != "posix" or not Path("/proc").is_dir():
        transcript["errors"] = ["UNSUPPORTED_PROCESS_TREE_PLATFORM"]
        transcript["status"] = "FAIL"
        return transcript
    with tempfile.TemporaryDirectory(
            prefix="dspark-global-validation-") as directory:
        scratch = Path(directory)
        discovery = [
            discover_compiler(label, family, major, candidates, scratch)
            for label, family, major, candidates in LIVE_COMPILER_SPECS
        ]
        transcript["compiler_discovery"] = discovery
        source_records, source_roots, source_errors = prepare_reverb_variants(
            root, scratch / "variants")
        transcript["source_variants"] = source_records
        errors = list(source_errors)
        errors.extend(
            str(record["terminal"])
            for record in discovery if record["status"] != "PASS")
        rows: list[dict[str, object]] = []
        if not errors:
            source_by_id = {
                str(record["id"]): record for record in source_records}
            for compiler_record in discovery:
                for profile, extra_flags in LIVE_PROFILES:
                    for variant in LIVE_REVERB_VARIANTS:
                        variant_id = str(variant["id"])
                        rows.append(reverb_row(
                            compiler_record, profile, extra_flags, variant,
                            source_by_id[variant_id], source_roots[variant_id]))
        transcript["executions"] = rows
        if len(rows) != 16:
            errors.append("EXECUTION_ROW_CARDINALITY:{}".format(len(rows)))
        errors.extend(
            "ROW_ORACLE_FAILED:" + str(row["id"])
            for row in rows if not row["oracle_satisfied"])
        transcript["errors"] = errors
        transcript["status"] = "PASS" if not errors else "FAIL"
    return transcript


def write_machine_transcript(path: Path, transcript: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as handle:
        json.dump(transcript, handle, indent=2, sort_keys=True)
        handle.write("\n")


def self_test(root: Path, doxygen: str | None,
              machine_json: Path | None = None) -> int:
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
    transcript = build_live_transcript(root)
    discovery = transcript.get("compiler_discovery", [])
    sources = transcript.get("source_variants", [])
    rows = transcript.get("executions", [])
    checks.extend((
        ("live-reverb-compiler-discovery-cardinality", len(discovery) == 2),
        ("live-reverb-source-cardinality", len(sources) == 4),
        ("live-reverb-execution-cardinality", len(rows) == 16),
        ("live-reverb-complete-matrix", transcript.get("status") == "PASS"),
    ))
    details["live-reverb-complete-matrix"] = json.dumps(
        transcript.get("errors", []), sort_keys=True)
    if machine_json is not None:
        try:
            write_machine_transcript(machine_json, transcript)
        except (FileExistsError, OSError) as error:
            print(f"ERROR HARNESS MACHINE_TRANSCRIPT_WRITE {error}",
                  file=sys.stderr)
            return 2
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
    parser.add_argument("--machine-json", type=Path)
    parser.add_argument("--doxygen", default=shutil.which("doxygen"))
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.self_test:
        return self_test(root, arguments.doxygen, arguments.machine_json)
    if arguments.machine_json is not None:
        print("ERROR HARNESS --machine-json requires --self-test",
              file=sys.stderr)
        return 2

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
