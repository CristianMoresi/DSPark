#!/usr/bin/env python3
"""Independently bind global product gates to their complete live evidence."""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_m018_global_corrections as producer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXECUTION_IDS = (
    "reverb::gcc13::normal::baseline",
    "reverb::gcc13::normal::early-mix",
    "reverb::gcc13::normal::early-predelay",
    "reverb::gcc13::normal::partial-shaping-publication",
    "reverb::gcc13::sanitizer::baseline",
    "reverb::gcc13::sanitizer::early-mix",
    "reverb::gcc13::sanitizer::early-predelay",
    "reverb::gcc13::sanitizer::partial-shaping-publication",
    "reverb::clang18::normal::baseline",
    "reverb::clang18::normal::early-mix",
    "reverb::clang18::normal::early-predelay",
    "reverb::clang18::normal::partial-shaping-publication",
    "reverb::clang18::sanitizer::baseline",
    "reverb::clang18::sanitizer::early-mix",
    "reverb::clang18::sanitizer::early-predelay",
    "reverb::clang18::sanitizer::partial-shaping-publication",
)
EXPECTED_EXECUTION_CARDINALITY = 16
EXPECTED_ROWS = (
    ("reverb::gcc13::normal::baseline", "gcc13", "normal", "baseline", "GREEN"),
    ("reverb::gcc13::normal::early-mix", "gcc13", "normal", "early-mix", "EXPECTED_RED"),
    ("reverb::gcc13::normal::early-predelay", "gcc13", "normal", "early-predelay", "EXPECTED_RED"),
    ("reverb::gcc13::normal::partial-shaping-publication", "gcc13", "normal", "partial-shaping-publication", "EXPECTED_RED"),
    ("reverb::gcc13::sanitizer::baseline", "gcc13", "sanitizer", "baseline", "GREEN"),
    ("reverb::gcc13::sanitizer::early-mix", "gcc13", "sanitizer", "early-mix", "EXPECTED_RED"),
    ("reverb::gcc13::sanitizer::early-predelay", "gcc13", "sanitizer", "early-predelay", "EXPECTED_RED"),
    ("reverb::gcc13::sanitizer::partial-shaping-publication", "gcc13", "sanitizer", "partial-shaping-publication", "EXPECTED_RED"),
    ("reverb::clang18::normal::baseline", "clang18", "normal", "baseline", "GREEN"),
    ("reverb::clang18::normal::early-mix", "clang18", "normal", "early-mix", "EXPECTED_RED"),
    ("reverb::clang18::normal::early-predelay", "clang18", "normal", "early-predelay", "EXPECTED_RED"),
    ("reverb::clang18::normal::partial-shaping-publication", "clang18", "normal", "partial-shaping-publication", "EXPECTED_RED"),
    ("reverb::clang18::sanitizer::baseline", "clang18", "sanitizer", "baseline", "GREEN"),
    ("reverb::clang18::sanitizer::early-mix", "clang18", "sanitizer", "early-mix", "EXPECTED_RED"),
    ("reverb::clang18::sanitizer::early-predelay", "clang18", "sanitizer", "early-predelay", "EXPECTED_RED"),
    ("reverb::clang18::sanitizer::partial-shaping-publication", "clang18", "sanitizer", "partial-shaping-publication", "EXPECTED_RED"),
)
EXPECTED_SOURCE_HASHES = {
    "baseline": "294a1053756adc86d84ab22240afdc4a55a706bef5e09a890ba6e084f4f6a80f",
    "early-mix": "a92a17baa0dee3618acf0a0de08a2e70b74fe479c73fd0bbf53c7dde4ec6ccfa",
    "early-predelay": "2bfc9fc8137ea9c214c740e0e953b4c1f9a5e13b096c5fcd0a4967c64cca23b6",
    "partial-shaping-publication": "1f6af93ae51eeaead1cb4c929641ecaa7817d687b36ee7ff7fb6441cce6b2010",
}
EXPECTED_INSERTION_HASHES = {
    "baseline": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "early-mix": "198cbf40437bae2cf942bb530552687078544ce62a5a38ee3fb1b1977a256b57",
    "early-predelay": "6bbfb61f9aa7c7d7f9502d11ccbd4b3907318e1f9d1e4f8c0a9fc606389d6936",
    "partial-shaping-publication": "4d3adddfcbc82e7bd4e6ee7a4015289675870106876426bcea60fd400166e87f",
}
EXPECTED_PATCH_HASHES = {
    "baseline": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "early-mix": "39400ede7f92730ec68e02599d8d0c125e066b571573422d036aaefaaa3222e9",
    "early-predelay": "42f1bb61861e677382493a2675f5d387959dab2af11aabca81bc1572ce977aff",
    "partial-shaping-publication": "557fbc4018cef7d9f91ffc8c9aee972a0b7218e24b210238f25f33834a8fdadc",
}
BASELINE_STDOUT = (
    "PASS reverb-slot-state-machine\n"
    "PASS reverb-fixed-audio-operation-bound\n"
    "PASS reverb-publisher-phase-parking\n"
    "PASS reverb-no-audio-allocation-or-final-release\n"
    "PASS reverb-premature-reuse-mutant\n"
    "PASS reverb-publisher-starvation-mutant\n"
    "PASS reverb-generation-aba\n"
    "PASS reverb-getconvolver-pin-lifetime\n"
    "PASS reverb-reset-metadata-exception-shutdown\n"
    "PASS reverb-retained-bank-memory-bound\n"
    "PASS reverb-race-sanitizer-matrix\n"
    "11 checks, 0 failures\n"
)
MUTANT_STDOUT = (
    "PASS reverb-slot-state-machine\n"
    "PASS reverb-fixed-audio-operation-bound\n"
    "PASS reverb-publisher-phase-parking\n"
    "PASS reverb-no-audio-allocation-or-final-release\n"
    "PASS reverb-premature-reuse-mutant\n"
    "PASS reverb-publisher-starvation-mutant\n"
    "PASS reverb-generation-aba\n"
    "PASS reverb-getconvolver-pin-lifetime\n"
    "PASS reverb-retained-bank-memory-bound\n"
    "PASS reverb-race-sanitizer-matrix\n"
    "11 checks, 1 failures\n"
)
MUTANT_STDERR = (
    "FAIL reverb-reset-metadata-exception-shutdown: "
    "failed setState changed serialized state bytes\n"
)
EXPECTED_OUTPUT_CONTRACTS = {
    "GREEN": (0, BASELINE_STDOUT, ""),
    "EXPECTED_RED": (1, MUTANT_STDOUT, MUTANT_STDERR),
}
SANITIZER_DIAGNOSTICS = (
    "ERROR: AddressSanitizer",
    "runtime error:",
    "LeakSanitizer",
)
COMMON_COMPILE_FLAGS = (
    "-std=c++20", "-O1", "-g", "-Wall", "-Wextra", "-Wpedantic",
    "-Werror", "-DDSPARK_REVERB_TEST_GENERATION_MAX=1", "-pthread",
)
SANITIZER_FLAGS = (
    "-fno-omit-frame-pointer",
    "-fsanitize=address,undefined,float-cast-overflow",
    "-fno-sanitize-recover=all",
)
REMOVED_RUNTIME_ENVIRONMENT = (
    "ASAN_OPTIONS", "UBSAN_OPTIONS", "LSAN_OPTIONS",
)
NORMAL_RUNTIME_ENVIRONMENT = {"LC_ALL": "C", "LANG": "C"}
SANITIZER_RUNTIME_ENVIRONMENT = {
    "LC_ALL": "C",
    "LANG": "C",
    "ASAN_OPTIONS": "halt_on_error=1:abort_on_error=1:detect_leaks=1",
    "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1",
}
EXPECTED_OUTPUTS = (
    "c1-stationary-fidelity.csv",
    "c2-spurious-floor.csv",
    "c3-transient-preservation.csv",
    "c4-vertical-coherence.csv",
    "c5-stereo-integrity.csv",
    "c6-exact-ratio-drift.csv",
    "c7-unity-passthrough.csv",
    "c8-chopping-determinism.csv",
    "timestretch-metrics.md",
)
TIMESTRETCH_TRANSACTION_DIRECTORY = ".dspark-timestretch-transaction"
TIMESTRETCH_FAILURE_PHASES = (
    ("STAGE", "DSPARK_TIMESTRETCH_FAIL_STAGE_INDEX"),
    ("COMMIT", "DSPARK_TIMESTRETCH_FAIL_COMMIT_INDEX"),
)
EXPECTED_THREADING_EXTERNAL_IDS = (
    "count-total",
    "count-template",
    "count-concrete",
    "count-overlap",
    "member-total-01",
    "member-total-02",
    "member-total-03",
    "member-total-04",
    "member-total-05",
    "member-total-06",
    "member-total-07",
    "member-total-08",
    "member-total-09",
    "member-total-10",
    "member-total-11",
    "member-template-01",
    "member-template-02",
    "member-template-03",
    "member-template-04",
    "member-template-05",
    "member-concrete-01",
    "member-concrete-02",
    "member-concrete-03",
    "member-concrete-04",
    "member-concrete-05",
    "member-concrete-06",
    "member-concrete-07",
    "member-concrete-08",
    "member-overlap-01",
    "member-overlap-02",
    "stale-10-5-7-2",
    "block-absent",
    "block-duplicate",
)
LIVE_COMMAND = (
    "python3 -B tools/verify_global_validation_contract.py --live"
)
LIVE_PRODUCER_CALL = "        producer_result = run_live_producer(root, transcript_path)\n"
PRODUCER_MATRIX_CALL = "    transcript = build_live_transcript(root)\n"
MAX_CAPTURE_BYTES = 1024 * 1024


def terminate_child(process: subprocess.Popen[bytes]) -> list[str]:
    actions: list[str] = []
    if os.name == "posix":
        for value, name in ((signal.SIGTERM, "SIGTERM_PROCESS_GROUP"),
                            (signal.SIGKILL, "SIGKILL_PROCESS_GROUP")):
            actions.append(name)
            try:
                os.killpg(process.pid, value)
            except ProcessLookupError:
                return actions
            try:
                process.wait(timeout=1)
                return actions
            except subprocess.TimeoutExpired:
                continue
    elif os.name == "nt":
        actions.append("TASKKILL_PROCESS_TREE")
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        actions.append("UNSUPPORTED_PROCESS_TREE_PLATFORM")
    return actions


def run_child(command: list[str], cwd: Path, timeout: float) -> dict[str, object]:
    creationflags = 0
    keywords: dict[str, object] = {}
    if os.name == "posix":
        keywords["start_new_session"] = True
    elif os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        return {
            "command": command,
            "exit_code": None,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "errors": ["UNSUPPORTED_PROCESS_TREE_PLATFORM"],
        }
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=creationflags, **keywords)
    timed_out = False
    actions: list[str] = []
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        actions = terminate_child(process)
        stdout, stderr = process.communicate()
    stdout_large = len(stdout) > MAX_CAPTURE_BYTES
    stderr_large = len(stderr) > MAX_CAPTURE_BYTES
    stdout = stdout[:MAX_CAPTURE_BYTES]
    stderr = stderr[:MAX_CAPTURE_BYTES]
    errors: list[str] = []
    if timed_out:
        errors.append("CHILD_TIMEOUT")
    if "UNSUPPORTED_PROCESS_TREE_PLATFORM" in actions:
        errors.append("UNSUPPORTED_PROCESS_TREE_PLATFORM")
    if stdout_large:
        errors.append("STDOUT_LIMIT_EXCEEDED")
    if stderr_large:
        errors.append("STDERR_LIMIT_EXCEEDED")
    return {
        "command": command,
        "timeout_seconds": timeout,
        "duration_seconds": round(time.monotonic() - started, 6),
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout.decode("utf-8", "replace"),
        "stderr": stderr.decode("utf-8", "replace"),
        "cleanup_actions": actions,
        "errors": errors,
    }


def job_block(text: str, name: str) -> str:
    match = re.search(
        r"(?ms)^  {}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)".format(
            re.escape(name)), text)
    return match.group(1) if match else ""


def binding_errors(root: Path,
                   overrides: dict[str, str] | None = None) -> list[str]:
    overrides = overrides or {}
    paths = (
        ".github/workflows/ci.yml",
        ".github/workflows/docs.yml",
        "tests/CMakeLists.txt",
    )
    text = {
        path: overrides.get(
            path, (root / path).read_text(encoding="ascii"))
        for path in paths
    }
    errors: list[str] = []
    ci_job = job_block(text[paths[0]], "documentation-quality")
    docs_job = job_block(text[paths[1]], "build-docs")
    for label, block in (("CI_DOCUMENTATION", ci_job),
                         ("DOCS_BUILD", docs_job)):
        if not block:
            errors.append("VALIDATION_OUTER_BINDING_JOB_MISSING:" + label)
            continue
        if "runs-on: ubuntu-24.04" not in block:
            errors.append("VALIDATION_OUTER_BINDING_RUNNER:" + label)
        if "g++-13" not in block or "clang-18" not in block:
            errors.append("VALIDATION_OUTER_BINDING_COMPILERS:" + label)
        if block.count(LIVE_COMMAND) != 1:
            errors.append("VALIDATION_OUTER_BINDING_LIVE:{}:{}".format(
                label, block.count(LIVE_COMMAND)))
    cmake = text[paths[2]]
    test_match = re.search(
        r"add_test\(NAME\s+global_product_corrections\s+"
        r"COMMAND\s+\$\{Python3_EXECUTABLE\}\s+-B\s+"
        r"\$\{PROJECT_SOURCE_DIR\}/tools/verify_global_validation_contract\.py\s+"
        r"--ctest\s+WORKING_DIRECTORY\s+\$\{PROJECT_SOURCE_DIR\}\s*\)",
        cmake, re.S)
    if test_match is None:
        errors.append("VALIDATION_OUTER_BINDING_CTEST")
    if len(re.findall(r"add_test\(NAME", cmake)) != 13:
        errors.append("VALIDATION_OUTER_CTEST_LOCAL_CARDINALITY")
    root_cmake = (root / "CMakeLists.txt").read_text(encoding="ascii")
    if len(re.findall(r"add_test\(NAME", root_cmake)) != 2:
        errors.append("VALIDATION_OUTER_CTEST_ROOT_CARDINALITY")
    return errors


def function_definition(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    return matches[0] if len(matches) == 1 else None


def assignment_value(tree: ast.AST, name: str) -> ast.AST | None:
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            matches.append(node.value)
    return matches[0] if len(matches) == 1 else None


def direct_call_count(node: ast.AST, name: str) -> int:
    return sum(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
        for item in ast.walk(node)
    )


def structural_errors(
    source: str, producer_source: str | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["VALIDATION_OUTER_SOURCE_SYNTAX:" + str(error)]

    inventory_node = assignment_value(tree, "EXPECTED_EXECUTION_IDS")
    try:
        inventory_value = ast.literal_eval(inventory_node) \
            if inventory_node is not None else None
    except (ValueError, TypeError):
        inventory_value = None
    if inventory_value != EXPECTED_EXECUTION_IDS:
        errors.append("VALIDATION_OUTER_EXPECTED_INVENTORY_LITERAL")
    cardinality_node = assignment_value(
        tree, "EXPECTED_EXECUTION_CARDINALITY")
    try:
        cardinality_value = ast.literal_eval(cardinality_node) \
            if cardinality_node is not None else None
    except (ValueError, TypeError):
        cardinality_value = None
    if cardinality_value != 16:
        errors.append("VALIDATION_OUTER_EXPECTED_CARDINALITY_LITERAL")

    live = function_definition(tree, "live_mode")
    if live is None or direct_call_count(live, "run_live_producer") != 1:
        errors.append("VALIDATION_OUTER_PRODUCER_CALL_CARDINALITY")
    validator = function_definition(tree, "validate_transcript")
    if validator is None:
        errors.append("VALIDATION_OUTER_VALIDATOR_FUNCTION")
    else:
        inventory_comparisons = [
            node for node in ast.walk(validator)
            if isinstance(node, ast.Compare)
            and any(isinstance(operator, ast.NotEq) for operator in node.ops)
            and "set(identities)" in ast.unparse(node)
            and "set(expected_ids)" in ast.unparse(node)
        ]
        if len(inventory_comparisons) != 1:
            errors.append("VALIDATION_OUTER_INVENTORY_COMPARISON_STRUCTURE")
        oracle_blocks = [
            node for node in ast.walk(validator)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "row_oracle_enabled"
            and any(
                isinstance(item, ast.Constant)
                and item.value == "oracle_satisfied"
                for item in ast.walk(node)
            )
        ]
        if len(oracle_blocks) != 1:
            errors.append("VALIDATION_OUTER_ROW_ORACLE_STRUCTURE")

    if producer_source is None:
        producer_source = (
            ROOT / "tools/verify_m018_global_corrections.py"
        ).read_text(encoding="ascii")
    try:
        producer_tree = ast.parse(producer_source)
    except SyntaxError as error:
        errors.append("VALIDATION_OUTER_PRODUCER_SOURCE_SYNTAX:" + str(error))
        return errors
    producer_self_test = function_definition(producer_tree, "self_test")
    if producer_self_test is None or direct_call_count(
            producer_self_test, "build_live_transcript") != 1:
        errors.append("VALIDATION_OUTER_PRODUCER_MATRIX_CALL_CARDINALITY")

    phases_node = assignment_value(tree, "TIMESTRETCH_FAILURE_PHASES")
    try:
        phases_value = ast.literal_eval(phases_node) \
            if phases_node is not None else None
    except (ValueError, TypeError):
        phases_value = None
    if phases_value != TIMESTRETCH_FAILURE_PHASES:
        errors.append("VALIDATION_OUTER_TIMESTRETCH_PHASE_INVENTORY")
    timestretch = function_definition(tree, "timestretch_live")
    if timestretch is None:
        errors.append("VALIDATION_OUTER_TIMESTRETCH_LIVE_FUNCTION")
    else:
        if direct_call_count(timestretch, "validate_timestretch_rollback") != 1:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_ROLLBACK_CALL")
        phase_loops = [
            node for node in ast.walk(timestretch)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "TIMESTRETCH_FAILURE_PHASES"
            and ast.unparse(node.target) == "(phase, variable)"
        ]
        position_loops = [
            node for node in ast.walk(timestretch)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
            and any(isinstance(item, ast.Name)
                    and item.id == "EXPECTED_OUTPUTS"
                    for item in node.iter.args)
        ]
        if len(phase_loops) != 1 or len(position_loops) != 1:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_POSITION_LOOPS")
    rollback_validator = function_definition(
        tree, "validate_timestretch_rollback")
    if rollback_validator is None:
        errors.append("VALIDATION_OUTER_TIMESTRETCH_ROLLBACK_ORACLE")
    else:
        validator_text = ast.unparse(rollback_validator)
        if "after != before" not in validator_text:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_SENTINEL_ORACLE")
        if "timestretch_transaction_residue(directory)" \
                not in validator_text:
            errors.append("VALIDATION_OUTER_TIMESTRETCH_CLEANUP_ORACLE")
    return errors


def threading_control_structure_errors(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return ["THREADING_CONTROL_SOURCE_SYNTAX:" + str(error)]
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_census_controls"
    ]
    loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "census_controls"
    ]
    errors: list[str] = []
    if len(calls) != 1:
        errors.append("THREADING_INTERNAL_CONTROL_INVOCATION")
    if len(loops) != 1:
        errors.append("THREADING_INTERNAL_CONTROL_EXECUTION")
    baseline_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_pin_census"
        and any(isinstance(argument, ast.Name) and argument.id == "doc"
                for argument in node.args)
    ]
    if len(calls) == 1 and (len(baseline_calls) != 1
                            or baseline_calls[0].lineno >= calls[0].lineno):
        errors.append("THREADING_BASELINE_BEFORE_INTERNAL_CONTROLS")
    return errors


GCC_DRIVER_TOKEN = (
    r"(?<![A-Za-z0-9_+.-])"
    r"(?:g\+\+(?:-\d+)?|gcc(?:-\d+)?)"
    r"(?![A-Za-z0-9_+.-])"
)


def compiler_version_identity(first_line: str) -> tuple[str | None, int | None]:
    clang = re.search(r"\bclang version\s+(\d+)(?:\.|\b)", first_line, re.I)
    if clang:
        return "clang", int(clang.group(1))
    if "clang" in first_line.lower():
        return "clang", None
    gcc = re.search(
        GCC_DRIVER_TOKEN + r".*?"
        r"(?:\)\s*|\bversion\s+)?(\d+)\.(\d+)", first_line, re.I)
    if gcc:
        return "gcc", int(gcc.group(1))
    if re.search(GCC_DRIVER_TOKEN, first_line, re.I):
        return "gcc", None
    return None, None


def process_record_errors(
    record: object, prefix: str, expected_exit: int,
    expected_command: list[str] | None = None,
) -> list[str]:
    if not isinstance(record, dict):
        return [prefix + ":TYPE"]
    errors: list[str] = []
    if expected_command is not None and record.get("command") != expected_command:
        errors.append(prefix + ":COMMAND")
    if record.get("exit_code") != expected_exit:
        errors.append(prefix + ":EXIT")
    if record.get("timed_out") is not False:
        errors.append(prefix + ":TIMEOUT")
    if record.get("errors") != []:
        errors.append(prefix + ":ERRORS")
    if record.get("cleanup_residue"):
        errors.append(prefix + ":RESIDUE")
    if record.get("residue_detected"):
        errors.append(prefix + ":RESIDUE_DETECTED")
    for stream in ("stdout", "stderr"):
        value = record.get(stream)
        if not isinstance(value, str):
            errors.append(prefix + ":" + stream.upper() + "_TYPE")
            continue
        encoded = value.encode("utf-8")
        if len(encoded) > MAX_CAPTURE_BYTES:
            errors.append(prefix + ":" + stream.upper() + "_LIMIT")
        if record.get(stream + "_size") != len(encoded):
            errors.append(prefix + ":" + stream.upper() + "_SIZE")
        if record.get(stream + "_sha256") != hashlib.sha256(encoded).hexdigest():
            errors.append(prefix + ":" + stream.upper() + "_HASH")
    return errors


def validate_transcript(
    transcript: dict[str, object], producer_exit: int = 0,
    expected_ids: tuple[str, ...] = EXPECTED_EXECUTION_IDS,
    comparison_enabled: bool = True, row_oracle_enabled: bool = True,
) -> list[str]:
    errors: list[str] = []
    if producer_exit != 0:
        errors.append("VALIDATION_OUTER_PRODUCER_EXIT:{}".format(producer_exit))
    if transcript.get("schema") != "dspark.global-validation-producer.v1":
        errors.append("VALIDATION_OUTER_TRANSCRIPT_SCHEMA")
    if transcript.get("status") != "PASS" or transcript.get("errors") != []:
        errors.append("VALIDATION_OUTER_TRANSCRIPT_STATUS")
    if transcript.get("invocation_count") != 1:
        errors.append("VALIDATION_OUTER_MATRIX_INVOCATION_COUNT:{}".format(
            transcript.get("invocation_count")))
    if len(expected_ids) != EXPECTED_EXECUTION_CARDINALITY:
        errors.append("VALIDATION_OUTER_EXPECTED_INVENTORY_CARDINALITY:{}".format(
            len(expected_ids)))
    if len(set(expected_ids)) != len(expected_ids):
        errors.append("VALIDATION_OUTER_EXPECTED_INVENTORY_DUPLICATE")
    if not comparison_enabled:
        errors.append("VALIDATION_OUTER_INVENTORY_COMPARISON_DISABLED")
    if not row_oracle_enabled:
        errors.append("VALIDATION_OUTER_ROW_ORACLE_DISABLED")

    expected_discovery = {"gcc13": ("gcc", 13), "clang18": ("clang", 18)}
    discovery = transcript.get("compiler_discovery")
    observed: dict[object, dict[str, object]] = {}
    if not isinstance(discovery, list) or len(discovery) != 2 \
            or not all(isinstance(item, dict) for item in discovery):
        errors.append("VALIDATION_OUTER_DISCOVERY_CARDINALITY")
    else:
        observed = {item.get("label"): item for item in discovery}
        if set(observed) != set(expected_discovery):
            errors.append("VALIDATION_OUTER_DISCOVERY_IDENTITIES")
        for label, (family, major) in expected_discovery.items():
            item = observed.get(label, {})
            if (item.get("status"), item.get("family"), item.get("major")) \
                    != ("PASS", family, major):
                errors.append("VALIDATION_OUTER_DISCOVERY_ORACLE:" + label)
            path = item.get("path")
            resolved = item.get("resolved_path")
            if not isinstance(path, str) or not path \
                    or not isinstance(resolved, str) or not resolved:
                errors.append("VALIDATION_OUTER_DISCOVERY_PATH:" + label)
                continue
            version = item.get("version_process")
            dump = item.get("dump_process")
            errors.extend(process_record_errors(
                version, "VALIDATION_OUTER_VERSION_PROCESS:" + label, 0,
                [path, "--version"]))
            errors.extend(process_record_errors(
                dump, "VALIDATION_OUTER_DUMP_PROCESS:" + label, 0,
                [path, "-dumpfullversion", "-dumpversion"]))
            first_line = item.get("version_first_line")
            if not isinstance(first_line, str) \
                    or compiler_version_identity(first_line) != (family, major):
                errors.append("VALIDATION_OUTER_VERSION_IDENTITY:" + label)
            dumped = item.get("dump_version")
            match = re.fullmatch(r"\s*(\d+)(?:\.\d+){0,3}\s*", dumped) \
                if isinstance(dumped, str) else None
            if match is None or int(match.group(1)) != major:
                errors.append("VALIDATION_OUTER_DUMP_IDENTITY:" + label)

    sources = transcript.get("source_variants")
    if not isinstance(sources, list) or len(sources) != 4:
        errors.append("VALIDATION_OUTER_SOURCE_CARDINALITY")
        source_by_id: dict[str, dict[str, object]] = {}
    else:
        source_by_id = {
            str(item.get("id")): item for item in sources
            if isinstance(item, dict)
        }
        if len(source_by_id) != 4 or set(source_by_id) != set(EXPECTED_SOURCE_HASHES):
            errors.append("VALIDATION_OUTER_SOURCE_IDENTITIES")
        for variant, expected_hash in EXPECTED_SOURCE_HASHES.items():
            item = source_by_id.get(variant, {})
            if item.get("source_sha256") != expected_hash:
                errors.append("VALIDATION_OUTER_SOURCE_HASH:" + variant)
            if item.get("insertion_sha256") != EXPECTED_INSERTION_HASHES[variant]:
                errors.append("VALIDATION_OUTER_INSERTION_HASH:" + variant)
            if item.get("patch_sha256") != EXPECTED_PATCH_HASHES[variant]:
                errors.append("VALIDATION_OUTER_PATCH_HASH:" + variant)
            if item.get("anchor_count_before") != 1 \
                    or item.get("anchor_count_after") != 1:
                errors.append("VALIDATION_OUTER_SOURCE_ANCHOR:" + variant)
            expected_delta = 0 if variant == "baseline" else 1
            if item.get("insertion_delta") != expected_delta:
                errors.append("VALIDATION_OUTER_SOURCE_INSERTION:" + variant)

    rows = transcript.get("executions")
    if not isinstance(rows, list):
        rows = []
        errors.append("VALIDATION_OUTER_EXECUTIONS_TYPE")
    identities = [row.get("id") for row in rows if isinstance(row, dict)]
    if len(identities) != len(rows):
        errors.append("VALIDATION_OUTER_ROW_TYPE")
    if len(identities) != len(set(identities)):
        errors.append("VALIDATION_OUTER_OBSERVED_IDENTITY_DUPLICATE")
    if not comparison_enabled:
        pass
    elif set(identities) != set(expected_ids):
        errors.append("VALIDATION_OUTER_MATRIX_IDENTITY_SET")
    contracts = {
        identity: (compiler, profile, variant, oracle)
        for identity, compiler, profile, variant, oracle in EXPECTED_ROWS
    }
    if row_oracle_enabled:
        for row in rows:
            if not isinstance(row, dict):
                continue
            identity = row.get("id")
            expected = contracts.get(identity)
            if expected is None:
                errors.append("VALIDATION_OUTER_UNEXPECTED_ROW:{}".format(identity))
                continue
            fields = tuple(row.get(name) for name in
                           ("compiler", "profile", "variant", "oracle"))
            if fields != expected:
                errors.append("VALIDATION_OUTER_ROW_FIELDS:" + str(identity))
            compiler, profile, variant, oracle = expected
            if row.get("build_ok") is not True:
                errors.append("VALIDATION_OUTER_BUILD_NOT_OK:" + str(identity))
            if row.get("run_timed_out") is not False:
                errors.append("VALIDATION_OUTER_RUN_TIMEOUT:" + str(identity))
            if row.get("cleanup_residue") != []:
                errors.append("VALIDATION_OUTER_CLEANUP_RESIDUE:" + str(identity))
            if row.get("oracle_satisfied") is not True:
                errors.append("VALIDATION_OUTER_ROW_ORACLE:" + str(identity))
            if row.get("source_sha256") != EXPECTED_SOURCE_HASHES[variant]:
                errors.append("VALIDATION_OUTER_ROW_SOURCE:" + str(identity))

            compile_record = row.get("compile")
            errors.extend(process_record_errors(
                compile_record, "VALIDATION_OUTER_COMPILE:" + str(identity), 0))
            command = compile_record.get("command", []) \
                if isinstance(compile_record, dict) else []
            if not isinstance(command, list):
                command = []
            discovered_path = observed.get(compiler, {}).get("path")
            if not command or command[0] != discovered_path:
                errors.append("VALIDATION_OUTER_COMPILE_DRIVER:" + str(identity))
            for flag in COMMON_COMPILE_FLAGS:
                if command.count(flag) != 1:
                    errors.append("VALIDATION_OUTER_COMPILE_FLAG:{}:{}".format(
                        identity, flag))
            for flag in SANITIZER_FLAGS:
                count = command.count(flag)
                if (profile == "sanitizer" and count != 1) \
                        or (profile == "normal" and count != 0):
                    errors.append("VALIDATION_OUTER_PROFILE_FLAG:{}:{}".format(
                        identity, flag))
            gcc_flag_count = command.count("-Wno-mismatched-new-delete")
            if (compiler == "gcc13" and gcc_flag_count != 1) \
                    or (compiler == "clang18" and gcc_flag_count != 0):
                errors.append("VALIDATION_OUTER_COMPILER_FLAG:" + str(identity))
            if not any(str(item).endswith("/tests/TestReverbPublication.cpp")
                       for item in command) \
                    or command.count("-I") != 1 or command.count("-o") != 1:
                errors.append("VALIDATION_OUTER_COMPILE_SUBJECT:" + str(identity))
            if isinstance(compile_record, dict) \
                    and (compile_record.get("stdout") != ""
                         or compile_record.get("stderr") != ""):
                errors.append("VALIDATION_OUTER_COMPILE_OUTPUT:" + str(identity))

            expected_environment = NORMAL_RUNTIME_ENVIRONMENT \
                if profile == "normal" else SANITIZER_RUNTIME_ENVIRONMENT
            environment_contract = row.get("environment_contract")
            if environment_contract != {
                    "remove_first": list(REMOVED_RUNTIME_ENVIRONMENT),
                    "set": expected_environment}:
                errors.append("VALIDATION_OUTER_RUNTIME_ENVIRONMENT:" + str(identity))

            expected_exit, expected_stdout, expected_stderr = \
                EXPECTED_OUTPUT_CONTRACTS[oracle]
            run_record = row.get("run")
            errors.extend(process_record_errors(
                run_record, "VALIDATION_OUTER_RUN:" + str(identity),
                expected_exit))
            if isinstance(run_record, dict):
                stdout = run_record.get("stdout")
                stderr = run_record.get("stderr")
                if stdout != expected_stdout or stderr != expected_stderr:
                    errors.append("VALIDATION_OUTER_RUN_OUTPUT:" + str(identity))
                combined = str(stdout) + str(stderr)
                if any(marker in combined for marker in SANITIZER_DIAGNOSTICS):
                    errors.append("VALIDATION_OUTER_SANITIZER_DIAGNOSTIC:" + str(identity))
    dict_rows = [row for row in rows if isinstance(row, dict)]
    if len([row for row in dict_rows if row.get("oracle") == "GREEN"]) != 4:
        errors.append("VALIDATION_OUTER_BASELINE_CARDINALITY")
    if len([row for row in dict_rows
            if row.get("oracle") == "EXPECTED_RED"]) != 12:
        errors.append("VALIDATION_OUTER_EXPECTED_RED_CARDINALITY")
    return errors


def synthetic_process(
    command: list[str], exit_code: int, stdout: str = "", stderr: str = "",
) -> dict[str, object]:
    stdout_bytes = stdout.encode("utf-8")
    stderr_bytes = stderr.encode("utf-8")
    return {
        "command": command,
        "exit_code": exit_code,
        "timed_out": False,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stdout_size": len(stdout_bytes),
        "stderr_size": len(stderr_bytes),
        "cleanup_residue": [],
        "residue_detected": [],
        "errors": [],
    }


def synthetic_transcript() -> dict[str, object]:
    sources = [
        {
            "id": variant,
            "source_sha256": source_hash,
            "insertion_sha256": EXPECTED_INSERTION_HASHES[variant],
            "patch_sha256": EXPECTED_PATCH_HASHES[variant],
            "anchor_count_before": 1,
            "anchor_count_after": 1,
            "insertion_delta": 0 if variant == "baseline" else 1,
        }
        for variant, source_hash in EXPECTED_SOURCE_HASHES.items()
    ]
    rows = []
    for identity, compiler, profile, variant, oracle in EXPECTED_ROWS:
        driver = "/opt/g++-13" if compiler == "gcc13" \
            else "/opt/clang++-18"
        flags = list(COMMON_COMPILE_FLAGS)
        if profile == "sanitizer":
            flags.extend(SANITIZER_FLAGS)
        if compiler == "gcc13":
            flags.append("-Wno-mismatched-new-delete")
        source_root = "/tmp/" + variant
        command = [
            driver, *flags, "-I", source_root,
            source_root + "/tests/TestReverbPublication.cpp",
            "-o", source_root + "/subject",
        ]
        expected_exit, expected_stdout, expected_stderr = \
            EXPECTED_OUTPUT_CONTRACTS[oracle]
        environment = NORMAL_RUNTIME_ENVIRONMENT \
            if profile == "normal" else SANITIZER_RUNTIME_ENVIRONMENT
        rows.append({
            "id": identity,
            "compiler": compiler,
            "profile": profile,
            "variant": variant,
            "oracle": oracle,
            "source_sha256": EXPECTED_SOURCE_HASHES[variant],
            "build_ok": True,
            "run_timed_out": False,
            "cleanup_residue": [],
            "oracle_satisfied": True,
            "environment_contract": {
                "remove_first": list(REMOVED_RUNTIME_ENVIRONMENT),
                "set": environment,
            },
            "compile": synthetic_process(command, 0),
            "run": synthetic_process(
                [source_root + "/subject"], expected_exit,
                expected_stdout, expected_stderr),
        })
    return {
        "schema": "dspark.global-validation-producer.v1",
        "status": "PASS",
        "errors": [],
        "invocation_count": 1,
        "compiler_discovery": [
            {
                "label": "gcc13", "status": "PASS", "family": "gcc",
                "major": 13,
                "path": "/opt/g++-13",
                "resolved_path": "/opt/g++-13",
                "version_first_line": "g++-13 (control) 13.3.0",
                "dump_version": "13.3.0",
                "version_process": synthetic_process(
                    ["/opt/g++-13", "--version"], 0,
                    "g++-13 (control) 13.3.0\n"),
                "dump_process": synthetic_process(
                    ["/opt/g++-13", "-dumpfullversion", "-dumpversion"],
                    0, "13.3.0\n"),
            },
            {
                "label": "clang18", "status": "PASS", "family": "clang",
                "major": 18,
                "path": "/opt/clang++-18",
                "resolved_path": "/opt/clang++-18",
                "version_first_line": "clang version 18.1.3",
                "dump_version": "18.1.3",
                "version_process": synthetic_process(
                    ["/opt/clang++-18", "--version"], 0,
                    "clang version 18.1.3\n"),
                "dump_process": synthetic_process(
                    ["/opt/clang++-18", "-dumpfullversion", "-dumpversion"],
                    0, "18.1.3\n"),
            },
        ],
        "source_variants": sources,
        "executions": rows,
    }


def fake_compiler(path: Path, family: str = "gcc", major: int = 13,
                  mode: str = "ok", driver: str | None = None,
                  dump_major: int | None = None) -> None:
    driver = driver or "gcc"
    first = "{} (validation control) {}.2.0".format(driver, major) \
        if family == "gcc" else "clang version {}.0.1".format(major)
    body = """#!/usr/bin/env python3
import signal
import subprocess
import sys
import time
mode = {mode!r}
if mode == "hang-leader":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
if mode == "hang-descendant":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    time.sleep(60)
if mode == "residue":
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    raise SystemExit(0)
if mode == "nonzero":
    print({first!r})
    raise SystemExit(9)
if mode == "empty":
    raise SystemExit(0)
if mode == "malformed":
    print("compiler version unknown")
    raise SystemExit(0)
if "--version" in sys.argv:
    print({first!r})
elif "-dumpfullversion" in sys.argv:
    if mode == "dump-nonzero":
        raise SystemExit(9)
    if mode == "dump-empty":
        raise SystemExit(0)
    if mode == "dump-malformed":
        print("unknown")
        raise SystemExit(0)
    if mode == "dump-hang":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(60)
    print({dump!r})
else:
    time.sleep(60)
""".format(
        mode=mode,
        first=first,
        dump="{}.2.0".format(major if dump_major is None else dump_major),
    )
    path.write_text(body, encoding="ascii")
    path.chmod(0o755)


def compiler_process_controls() -> list[tuple[str, bool]]:
    controls: list[tuple[str, bool]] = []
    if os.name != "posix" or not Path("/proc").is_dir():
        return [("compiler-process-controls-platform-deferred", True)]
    with tempfile.TemporaryDirectory(
            prefix="dspark-compiler-controls-") as directory:
        scratch = Path(directory)
        old_timeout = producer.VERSION_TIMEOUT_SECONDS
        old_grace = producer.PROCESS_GRACE_SECONDS
        old_cleanup = producer.PROCESS_CLEANUP_SECONDS
        producer.VERSION_TIMEOUT_SECONDS = 0.25
        producer.PROCESS_GRACE_SECONDS = 0.1
        producer.PROCESS_CLEANUP_SECONDS = 0.5
        try:
            for name, driver in (("generic-gxx", "g++"),
                                 ("versioned-gxx", "g++-13")):
                path = scratch / ("compiler-" + name)
                fake_compiler(path, driver=driver)
                record = producer.discover_compiler(
                    name, "gcc", 13, (driver,), scratch,
                    {driver: str(path)})
                controls.append((
                    "compiler-" + name + "-positive",
                    record.get("status") == "PASS"
                    and record.get("candidate") == driver
                    and record.get("family") == "gcc"
                    and record.get("major") == 13))
            missing = producer.discover_compiler(
                "gcc13", "gcc", 13, ("candidate",), scratch, {})
            controls.append((
                "compiler-missing",
                "MISSING_EXECUTABLE" in str(missing.get("terminal"))))
            cases = (
                ("compiler-swapped", "gcc", 13, "clang", 18, "FAMILY_MISMATCH"),
                ("compiler-gcc12", "gcc", 13, "gcc", 12, "MAJOR_MISMATCH:12"),
                ("compiler-gcc14", "gcc", 13, "gcc", 14, "MAJOR_MISMATCH:14"),
                ("compiler-clang17", "clang", 18, "clang", 17, "MAJOR_MISMATCH:17"),
                ("compiler-clang19", "clang", 18, "clang", 19, "MAJOR_MISMATCH:19"),
            )
            for name, required_family, required_major, family, major, reason in cases:
                path = scratch / name
                fake_compiler(path, family, major)
                record = producer.discover_compiler(
                    name, required_family, required_major, ("candidate",),
                    scratch, {"candidate": str(path)})
                controls.append((name, reason in str(record.get("terminal"))))
            for mode, reason in (
                    ("empty", "VERSION_OUTPUT_EMPTY"),
                    ("malformed", "VERSION_OUTPUT_MALFORMED"),
                    ("nonzero", "VERSION_PROBE_NONZERO"),
                    ("hang-leader", "VERSION_PROBE_TIMEOUT"),
                    ("hang-descendant", "VERSION_PROBE_TIMEOUT")):
                path = scratch / ("compiler-" + mode)
                fake_compiler(path, mode=mode)
                record = producer.discover_compiler(
                    mode, "gcc", 13, ("candidate",), scratch,
                    {"candidate": str(path)})
                controls.append(("compiler-" + mode,
                                 reason in str(record.get("terminal"))))
            for mode, dump_major, reason in (
                    ("dump-empty", None, "DUMP_VERSION_MALFORMED"),
                    ("dump-malformed", None, "DUMP_VERSION_MALFORMED"),
                    ("dump-nonzero", None, "DUMP_VERSION_NONZERO:9"),
                    ("dump-hang", None, "DUMP_VERSION_TIMEOUT"),
                    ("ok", 14, "DUMP_VERSION_MAJOR_MISMATCH:14")):
                name = "compiler-wrong-dump-{}-{}".format(
                    mode, "default" if dump_major is None else dump_major)
                path = scratch / name
                fake_compiler(path, mode=mode, dump_major=dump_major)
                record = producer.discover_compiler(
                    name, "gcc", 13, ("candidate",), scratch,
                    {"candidate": str(path)})
                controls.append((name,
                                 reason in str(record.get("terminal"))))
            hanging = scratch / "phase-timeout"
            fake_compiler(hanging, mode="hang-leader")
            for phase in ("compile", "run"):
                result = producer.run_owned(
                    [str(hanging)], scratch, 0.1, phase)
                controls.append((
                    phase + "-timeout-cleanup",
                    result["timed_out"] and not result["cleanup_residue"]))
            residue = scratch / "phase-residue"
            fake_compiler(residue, mode="residue")
            result = producer.run_owned([str(residue)], scratch, 1, "run")
            controls.append((
                "process-residue-rejected",
                "PROCESS_TREE_CLEANUP_FAILED" in result["errors"]
                and not result["cleanup_residue"]))
        finally:
            producer.VERSION_TIMEOUT_SECONDS = old_timeout
            producer.PROCESS_GRACE_SECONDS = old_grace
            producer.PROCESS_CLEANUP_SECONDS = old_cleanup
    return controls


def compiler_banner_controls() -> list[tuple[str, bool]]:
    cases = (
        ("generic-gxx", "g++ (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("versioned-gxx", "g++-13 (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("generic-gcc", "gcc (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("versioned-gcc", "gcc-13 (Ubuntu 13.4.0) 13.4.0", ("gcc", 13)),
        ("path-versioned-gxx", "/usr/bin/g++-13 (control) 13.2.0", ("gcc", 13)),
        ("punctuated-generic-gxx", "(g++) (control) 13.2.0", ("gcc", 13)),
        ("clang18", "Ubuntu clang version 18.1.3", ("clang", 18)),
        ("generic-gxx-no-version", "g++ (unknown)", ("gcc", None)),
        ("substring-prefix-gxx", "myg++ (control) 13.2.0", (None, None)),
        ("substring-suffix-gxx", "g++driver (control) 13.2.0", (None, None)),
        ("substring-versioned-gxx", "g++-13driver (control) 13.2.0", (None, None)),
        ("substring-prefix-gcc", "mygcc (control) 13.2.0", (None, None)),
        ("substring-suffix-gcc", "gccdriver (control) 13.2.0", (None, None)),
        ("substring-versioned-gcc", "gcc-13driver (control) 13.2.0", (None, None)),
    )
    return [
        (
            "compiler-banner-" + name,
            compiler_version_identity(banner) == expected
            and producer.parse_compiler_version(banner) == expected,
        )
        for name, banner, expected in cases
    ]


def generic_gcc_live(root: Path) -> list[str]:
    with tempfile.TemporaryDirectory(
            prefix="dspark-generic-gcc-live-") as directory:
        record = producer.discover_compiler(
            "gcc13-generic", "gcc", 13, ("g++",), Path(directory))
    errors: list[str] = []
    if record.get("status") != "PASS":
        errors.append("GENERIC_GCC13_DISCOVERY:" + str(record.get("terminal")))
        return errors
    if record.get("candidate") != "g++" \
            or record.get("family") != "gcc" or record.get("major") != 13:
        errors.append("GENERIC_GCC13_IDENTITY")
    first_line = record.get("version_first_line")
    if not isinstance(first_line, str) \
            or compiler_version_identity(first_line) != ("gcc", 13) \
            or producer.parse_compiler_version(first_line) != ("gcc", 13):
        errors.append("GENERIC_GCC13_REAL_BANNER")
    for key in ("version_process", "dump_process"):
        value = record.get(key)
        if not isinstance(value, dict) or value.get("errors") != [] \
                or value.get("timed_out") is not False \
                or value.get("cleanup_residue"):
            errors.append("GENERIC_GCC13_PROCESS:" + key)
    return errors


def self_test(root: Path) -> int:
    baseline = synthetic_transcript()
    source = Path(__file__).read_text(encoding="ascii")
    producer_source = (
        root / "tools/verify_m018_global_corrections.py"
    ).read_text(encoding="ascii")
    threading_source = (
        root / "tools/verify_threading_doc.py"
    ).read_text(encoding="ascii")
    controls: list[tuple[str, bool]] = [
        ("outer-baseline", not validate_transcript(baseline)),
        ("outer-structure", not structural_errors(source, producer_source)),
        ("threading-control-structure",
         not threading_control_structure_errors(threading_source)),
    ]

    def catches(name: str, value: dict[str, object], prefix: str,
                **options: object) -> None:
        errors = validate_transcript(value, **options)
        controls.append((name, any(error.startswith(prefix) for error in errors)))

    mutant = copy.deepcopy(baseline)
    mutant["invocation_count"] = 0
    catches("whole-producer-invocation", mutant,
            "VALIDATION_OUTER_MATRIX_INVOCATION_COUNT")
    errors = validate_transcript(
        baseline, expected_ids=EXPECTED_EXECUTION_IDS[:-1])
    controls.append((
        "literal-expected-inventory",
        any(error.startswith(
            "VALIDATION_OUTER_EXPECTED_INVENTORY_CARDINALITY")
            for error in errors)))
    controls.append((
        "inventory-comparison",
        "VALIDATION_OUTER_INVENTORY_COMPARISON_DISABLED"
        in validate_transcript(baseline, comparison_enabled=False)))
    controls.append((
        "per-row-oracle",
        "VALIDATION_OUTER_ROW_ORACLE_DISABLED"
        in validate_transcript(baseline, row_oracle_enabled=False)))
    controls.append((
        "literal-inventory-source",
        "VALIDATION_OUTER_EXPECTED_INVENTORY_LITERAL"
        in structural_errors(source.replace(
            '    "reverb::gcc13::normal::baseline",\n', "", 1),
            producer_source)))
    controls.append((
        "inventory-comparison-source",
        "VALIDATION_OUTER_INVENTORY_COMPARISON_STRUCTURE"
        in structural_errors(source.replace(
            "    elif set(identities) != set(expected_ids):\n",
            "    elif False:\n", 1), producer_source)))
    controls.append((
        "row-oracle-source",
        "VALIDATION_OUTER_ROW_ORACLE_STRUCTURE"
        in structural_errors(source.replace(
            "    if row_oracle_enabled:\n", "    if False:\n", 1),
            producer_source)))
    for identity in EXPECTED_EXECUTION_IDS:
        mutant = copy.deepcopy(baseline)
        mutant["executions"] = [
            row for row in mutant["executions"] if row["id"] != identity]
        catches("identity-" + identity, mutant,
                "VALIDATION_OUTER_MATRIX_IDENTITY_SET")
    for dimension, values in (
        ("compiler", ("gcc13", "clang18")),
        ("profile", ("normal", "sanitizer")),
        ("variant", tuple(EXPECTED_SOURCE_HASHES)),
    ):
        for value in values:
            mutant = copy.deepcopy(baseline)
            mutant["executions"] = [
                row for row in mutant["executions"]
                if row[dimension] != value]
            catches("dimension-{}-{}".format(dimension, value), mutant,
                    "VALIDATION_OUTER_MATRIX_IDENTITY_SET")
    for variant in tuple(EXPECTED_SOURCE_HASHES)[1:]:
        mutant = copy.deepcopy(baseline)
        for source_record in mutant["source_variants"]:
            if source_record["id"] == variant:
                source_record["source_sha256"] = EXPECTED_SOURCE_HASHES["baseline"]
        catches("replacement-" + variant, mutant,
                "VALIDATION_OUTER_SOURCE_HASH")
    mutant = copy.deepcopy(baseline)
    next(row for row in mutant["executions"]
         if row["profile"] == "sanitizer")["oracle_satisfied"] = False
    catches("sanitizer-only-oracle", mutant,
            "VALIDATION_OUTER_ROW_ORACLE")
    mutant = copy.deepcopy(baseline)
    mutant["executions"][0]["run"]["stdout"] += "unexpected\n"
    catches("raw-row-output", mutant,
            "VALIDATION_OUTER_RUN:")
    mutant = copy.deepcopy(baseline)
    mutant["executions"][0]["environment_contract"]["set"]["LANG"] = ""
    catches("runtime-environment", mutant,
            "VALIDATION_OUTER_RUNTIME_ENVIRONMENT")
    controls.append((
        "producer-nonzero",
        "VALIDATION_OUTER_PRODUCER_EXIT:9"
        in validate_transcript(baseline, producer_exit=9)))
    mutant = copy.deepcopy(baseline)
    mutant["executions"][0]["cleanup_residue"] = [999]
    catches("leftover-process", mutant,
            "VALIDATION_OUTER_CLEANUP_RESIDUE")
    mutant = copy.deepcopy(baseline)
    mutant["executions"].append(copy.deepcopy(mutant["executions"][0]))
    catches("duplicate-identity", mutant,
            "VALIDATION_OUTER_OBSERVED_IDENTITY_DUPLICATE")
    mutant = copy.deepcopy(baseline)
    extra = copy.deepcopy(mutant["executions"][0])
    extra["id"] = "reverb::extra"
    mutant["executions"].append(extra)
    catches("extra-identity", mutant,
            "VALIDATION_OUTER_MATRIX_IDENTITY_SET")

    controls.append((
        "delete-live-producer-call",
        "VALIDATION_OUTER_PRODUCER_CALL_CARDINALITY"
        in structural_errors(
            source.replace(LIVE_PRODUCER_CALL, "", 1), producer_source)))
    controls.append((
        "delete-producer-matrix-call",
        "VALIDATION_OUTER_PRODUCER_MATRIX_CALL_CARDINALITY"
        in structural_errors(
            source, producer_source.replace(PRODUCER_MATRIX_CALL, "", 1))))
    controls.append((
        "delete-timestretch-rollback-call",
        "VALIDATION_OUTER_TIMESTRETCH_ROLLBACK_CALL"
        in structural_errors(source.replace(
            "                    errors.extend(validate_timestretch_rollback(\n"
            "                        failure_root, before, result, phase, output_name))\n",
            "", 1), producer_source)))
    controls.append((
        "neutralize-timestretch-sentinel-oracle",
        "VALIDATION_OUTER_TIMESTRETCH_SENTINEL_ORACLE"
        in structural_errors(source.replace(
            "    if after != before:\n", "    if False:\n", 1),
            producer_source)))
    controls.append((
        "neutralize-timestretch-cleanup-oracle",
        "VALIDATION_OUTER_TIMESTRETCH_CLEANUP_ORACLE"
        in structural_errors(source.replace(
            "    if timestretch_transaction_residue(directory):\n",
            "    if False:\n", 1), producer_source)))
    controls.append((
        "delete-timestretch-position-loop",
        "VALIDATION_OUTER_TIMESTRETCH_POSITION_LOOPS"
        in structural_errors(source.replace(
            "                for index, output_name in enumerate(EXPECTED_OUTPUTS):\n",
            "                for index, output_name in ():\n", 1),
            producer_source)))
    controls.append((
        "delete-threading-internal-controls",
        "THREADING_INTERNAL_CONTROL_INVOCATION"
        in threading_control_structure_errors(threading_source.replace(
            "build_census_controls(doc, pin_census)", "([], [])", 1))))
    controls.append((
        "threading-external-missing-row",
        "THREADING_EXTERNAL_CARDINALITY:32"
        in threading_external_inventory_errors(
            list(EXPECTED_THREADING_EXTERNAL_IDS[:-1]))))
    workflow_paths = (
        ".github/workflows/ci.yml",
        ".github/workflows/docs.yml",
        "tests/CMakeLists.txt",
    )
    for path in workflow_paths:
        original = (root / path).read_text(encoding="ascii")
        if path.endswith("CMakeLists.txt"):
            changed = original.replace(
                "${PROJECT_SOURCE_DIR}/tools/verify_global_validation_contract.py",
                "${PROJECT_SOURCE_DIR}/tools/verify_m018_global_corrections.py",
                1)
        else:
            changed = original.replace(LIVE_COMMAND, "", 1)
        controls.append((
            "binding-" + path,
            bool(binding_errors(root, {path: changed}))))
    controls.extend(compiler_banner_controls())
    controls.extend(compiler_process_controls())
    for name, passed in controls:
        print("{} outer mutant {}".format("PASS" if passed else "FAIL", name))
    return 0 if all(passed for _name, passed in controls) else 1


def normal_gate(root: Path) -> dict[str, object]:
    return run_child(
        [sys.executable, "-B", "tools/verify_m018_global_corrections.py"],
        root, 300)


def ctest_mode(root: Path) -> int:
    errors = structural_errors(Path(__file__).read_text(encoding="ascii"))
    errors.extend(threading_control_structure_errors(
        (root / "tools/verify_threading_doc.py").read_text(encoding="ascii")))
    errors.extend(binding_errors(root))
    if self_test(root) != 0:
        errors.append("VALIDATION_OUTER_META_CONTROLS")
    child = normal_gate(root)
    if child["exit_code"] != 0 or child["errors"]:
        errors.append("VALIDATION_OUTER_NORMAL_GATE\n" + str(child["stdout"])
                      + str(child["stderr"]))
    for error in errors:
        print("ERROR " + error, file=sys.stderr)
    if errors:
        return 1
    print("PASS global validation contract ctest")
    return 0


def run_live_producer(root: Path, transcript_path: Path) -> dict[str, object]:
    return run_child(
        [
            sys.executable, "-B",
            "tools/verify_m018_global_corrections.py", "--self-test",
            "--machine-json", str(transcript_path),
        ],
        root, 1200)


def csv_rows(path: Path) -> list[dict[str, str]]:
    lines = [line for line in path.read_text(encoding="ascii").splitlines()
             if line and not line.startswith("#")]
    return list(csv.DictReader(lines))


def measurement_errors(directory: Path) -> list[str]:
    errors: list[str] = []
    expected_counts = {
        "c1-stationary-fidelity.csv": 7,
        "c2-spurious-floor.csv": 7,
        "c3-transient-preservation.csv": 7,
        "c4-vertical-coherence.csv": 14,
        "c5-stereo-integrity.csv": 7,
        "c6-exact-ratio-drift.csv": 7,
        "c7-unity-passthrough.csv": 4,
        "c8-chopping-determinism.csv": 16,
    }
    expected_columns = {
        "c1-stationary-fidelity.csv": (
            "ratio", "lsd_db", "lsd_limit_db", "lsd_db_unclipped",
            "spectral_convergence_db"),
        "c2-spurious-floor.csv": (
            "ratio", "worst_spurious_db_re_fundamental", "worst_bin",
            "worst_hz"),
        "c3-transient-preservation.csv": (
            "ratio", "onsets", "mean_energy_ratio_vs_wsola",
            "worst_energy_ratio_vs_wsola", "source_attack_ms",
            "output_attack_ms", "attack_expansion_percent"),
        "c4-vertical-coherence.csv": (
            "ratio", "bed", "consistency_in", "consistency_out_locked",
            "consistency_out_plain", "ratio_out_over_in", "vcoh_in",
            "vcoh_locked", "vcoh_plain", "locked_better_than_plain"),
        "c5-stereo-integrity.csv": (
            "ratio", "ild_in_db", "ild_out_db", "ild_delta_db",
            "coherence_in", "coherence_out"),
        "c6-exact-ratio-drift.csv": (
            "target_ratio", "input_samples", "output_samples",
            "expected_samples", "length_error_samples", "one_hop",
            "realised_ratio_from_onsets", "ratio_error_percent",
            "onsets_used", "worst_onset_error_samples"),
        "c7-unity-passthrough.csv": (
            "signal", "path", "residual_dbfs", "thd_n_dbfs", "peak_dbfs"),
        "c8-chopping-determinism.csv": (
            "ratio", "pattern", "samples_compared", "differing_samples",
            "first_divergence", "max_abs_difference"),
    }
    parsed: dict[str, list[dict[str, str]]] = {}
    for name, count in expected_counts.items():
        try:
            parsed[name] = csv_rows(directory / name)
        except (OSError, csv.Error, UnicodeError) as error:
            errors.append("TIMESTRETCH_SCHEMA:{}:{}".format(name, error))
            continue
        if len(parsed[name]) != count:
            errors.append("TIMESTRETCH_ROW_COUNT:{}:{}".format(
                name, len(parsed[name])))
        if parsed[name] and tuple(parsed[name][0]) != expected_columns[name]:
            errors.append("TIMESTRETCH_SCHEMA_COLUMNS:" + name)
    try:
        for row in parsed["c1-stationary-fidelity.csv"]:
            if float(row["lsd_db"]) > float(row["lsd_limit_db"]):
                errors.append("TIMESTRETCH_STATIONARY_FIDELITY")
        for row in parsed["c2-spurious-floor.csv"]:
            if float(row["worst_spurious_db_re_fundamental"]) > -80.0:
                errors.append("TIMESTRETCH_SPURIOUS_FLOOR")
        for row in parsed["c3-transient-preservation.csv"]:
            if int(row["onsets"]) != 10 \
                    or float(row["worst_energy_ratio_vs_wsola"]) < 0.65:
                errors.append("TIMESTRETCH_TRANSIENT_PRESERVATION")
        for row in parsed["c4-vertical-coherence.csv"]:
            numeric = tuple(float(row[name]) for name in (
                "ratio", "consistency_in", "consistency_out_locked",
                "consistency_out_plain", "ratio_out_over_in", "vcoh_in",
                "vcoh_locked", "vcoh_plain"))
            if not all(math.isfinite(value) for value in numeric) \
                    or float(row["vcoh_locked"]) <= 0.98 \
                    or row["bed"] not in {"stationary", "vibrato"} \
                    or row["locked_better_than_plain"] not in {"yes", "NO"}:
                errors.append("TIMESTRETCH_VERTICAL_COHERENCE")
        for row in parsed["c5-stereo-integrity.csv"]:
            if abs(float(row["ild_delta_db"])) >= 0.1 \
                    or float(row["coherence_out"]) <= 0.98:
                errors.append("TIMESTRETCH_STEREO_INTEGRITY")
        for row in parsed["c6-exact-ratio-drift.csv"]:
            if int(row["length_error_samples"]) != 0 \
                    or abs(float(row["ratio_error_percent"])) >= 0.01 \
                    or float(row["worst_onset_error_samples"]) > float(row["one_hop"]):
                errors.append("TIMESTRETCH_RATIO_DRIFT")
        for row in parsed["c7-unity-passthrough.csv"]:
            if float(row["residual_dbfs"]) >= -130.0:
                errors.append("TIMESTRETCH_UNITY_PASSTHROUGH")
        for row in parsed["c8-chopping-determinism.csv"]:
            if int(row["differing_samples"]) != 0 \
                    or int(row["first_divergence"]) != -1 \
                    or float(row["max_abs_difference"]) != 0.0:
                errors.append("TIMESTRETCH_CHOPPING_DETERMINISM")
    except (KeyError, ValueError) as error:
        errors.append("TIMESTRETCH_MEASUREMENT_PARSE:" + str(error))
    summary = (directory / "timestretch-metrics.md").read_text(
        encoding="ascii")
    if summary.count("| r=") != 10 or summary.count("| 120->") != 4:
        errors.append("TIMESTRETCH_SUMMARY_ROWS")
    return sorted(set(errors))


def threading_external_inventory_errors(observed: list[str]) -> list[str]:
    errors: list[str] = []
    if len(observed) != len(EXPECTED_THREADING_EXTERNAL_IDS):
        errors.append("THREADING_EXTERNAL_CARDINALITY:{}".format(
            len(observed)))
    if len(observed) != len(set(observed)):
        errors.append("THREADING_EXTERNAL_DUPLICATE_ID")
    if set(observed) != set(EXPECTED_THREADING_EXTERNAL_IDS):
        errors.append("THREADING_EXTERNAL_IDENTITY_SET")
    return errors


def threading_census_mutations(
    text: str,
) -> list[tuple[str, str, str]]:
    begin = "<!-- THREADING_PIN_CENSUS_BEGIN -->"
    end = "<!-- THREADING_PIN_CENSUS_END -->"
    labels = (
        ("total", "All local pin headers"),
        ("template", "Template-parameter pin headers"),
        ("concrete", "Concrete-word pin headers"),
        ("overlap", "Overlap headers"),
    )

    def row(key: str, label: str, source: str) -> tuple[int, list[str], int]:
        lines = source.splitlines(keepends=True)
        matches = [
            index for index, line in enumerate(lines)
            if line.startswith("- {} (".format(label))
        ]
        if len(matches) != 1:
            raise ValueError("{} row cardinality {}".format(key, len(matches)))
        line = lines[matches[0]]
        count = re.search(r"\((\d+)\)", line)
        members = re.findall(r"`([^`]+\.h)`", line)
        if count is None or int(count.group(1)) != len(members):
            raise ValueError("{} row is not a valid baseline".format(key))
        return matches[0], members, int(count.group(1))

    def replace_line(source: str, index: int, replacement: str) -> str:
        lines = source.splitlines(keepends=True)
        newline = "\n" if lines[index].endswith("\n") else ""
        lines[index] = replacement.rstrip("\n") + newline
        return "".join(lines)

    mutations: list[tuple[str, str, str]] = []
    baseline_rows: dict[str, tuple[str, list[str], int]] = {}
    for key, label in labels:
        index, members, count = row(key, label, text)
        baseline_rows[key] = (label, members, count)
        count_mutant = replace_line(
            text, index,
            text.splitlines()[index].replace(
                "({})".format(count), "({})".format(count - 1), 1))
        mutations.append((
            "count-" + key, count_mutant,
            "DOC_COUNT_MISMATCH:" + key))

    for key, label in labels:
        index, members, _count = row(key, label, text)
        original_line = text.splitlines()[index]
        for member_index, member in enumerate(members, 1):
            needle = "`{}`".format(member)
            if original_line.count(needle) != 1:
                raise ValueError("{} member anchor is not unique".format(key))
            mutant = replace_line(
                text, index,
                original_line.replace(
                    needle, "`Core/AnalogRandom.h`", 1))
            mutations.append((
                "member-{}-{:02d}".format(key, member_index), mutant,
                "SOURCE_SET_MISMATCH:" + key))

    stale = text
    for key in ("total", "concrete"):
        label, _members, _count = baseline_rows[key]
        index, members, count = row(key, label, stale)
        members.remove("Analysis/BeatTracker.h")
        stale = replace_line(
            stale, index,
            "- {} ({}): {}.".format(
                label, count - 1,
                ", ".join("`{}`".format(member) for member in members)))
    mutations.append((
        "stale-10-5-7-2", stale, "SOURCE_SET_MISMATCH:"))
    mutations.append((
        "block-absent", text.replace(begin, "", 1),
        "DOC_CENSUS_BLOCK_CARDINALITY"))
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError("census block baseline cardinality changed")
    body = text.split(begin, 1)[1].split(end, 1)[0]
    mutations.append((
        "block-duplicate", text + "\n" + begin + body + end + "\n",
        "DOC_CENSUS_BLOCK_CARDINALITY"))
    return mutations


def threading_external_live(root: Path) -> list[str]:
    errors: list[str] = []
    baseline_doc = (root / "docs/threading.md").read_text(encoding="ascii")
    try:
        mutations = threading_census_mutations(baseline_doc)
    except (OSError, UnicodeError, ValueError) as error:
        return ["THREADING_EXTERNAL_GENERATOR:" + str(error)]
    generated_ids = [name for name, _text, _terminal in mutations]
    errors.extend(threading_external_inventory_errors(generated_ids))
    if errors:
        return errors

    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=False,
        capture_output=True, timeout=30).stdout.split(b"\0")
    tracked_paths = [item.decode("utf-8") for item in tracked if item]
    if len(tracked_paths) != 484 or len(tracked_paths) != len(set(tracked_paths)):
        return ["THREADING_EXTERNAL_TRACKED_CENSUS:{}".format(
            len(tracked_paths))]

    observed: list[str] = []
    with tempfile.TemporaryDirectory(
            prefix="dspark-threading-external-") as directory:
        scratch = Path(directory)
        base = scratch / "base"
        base.mkdir()
        for relative in tracked_paths:
            source = root / relative
            destination = base / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_symlink() or not source.is_file():
                return ["THREADING_EXTERNAL_NONREGULAR_SOURCE:" + relative]
            shutil.copy2(source, destination)

        for name, mutant_doc, expected_terminal in mutations:
            fixture = scratch / name
            shutil.copytree(base, fixture, copy_function=os.link)
            doc_path = fixture / "docs/threading.md"
            doc_path.unlink()
            doc_path.write_text(mutant_doc, encoding="ascii", newline="")
            setup = subprocess.run(
                ["git", "init", "-q"], cwd=fixture, check=False,
                capture_output=True, timeout=30)
            indexed = subprocess.run(
                ["git", "add", "-f", "-A"], cwd=fixture, check=False,
                capture_output=True, timeout=60)
            indexed_paths = subprocess.run(
                ["git", "ls-files", "-z"], cwd=fixture, check=False,
                capture_output=True, timeout=30)
            indexed_count = len([
                item for item in indexed_paths.stdout.split(b"\0") if item
            ])
            if setup.returncode != 0 or indexed.returncode != 0 \
                    or indexed_paths.returncode != 0 or indexed_count != 484:
                errors.append("THREADING_EXTERNAL_FIXTURE:{}".format(name))
                observed.append(name)
                continue
            result = producer.run_owned(
                [sys.executable, "-B", "tools/verify_threading_doc.py"],
                fixture, 120, "threading_external")
            combined = str(result.get("stdout")) + str(result.get("stderr"))
            prefix = "THREADING_EXTERNAL_MUTANT:" + name
            if result.get("exit_code") == 0 or result.get("errors") != []:
                errors.append(prefix + ":PROCESS")
            if expected_terminal not in combined:
                errors.append(prefix + ":NAMED_TERMINAL")
            if "Traceback" in combined or "RuntimeError" in combined:
                errors.append(prefix + ":UNCAUGHT_EXCEPTION")
            if "SKIP threading census internal controls: external baseline invalid" \
                    not in combined:
                errors.append(prefix + ":BASELINE_FIRST")
            if "threading census mutant" in combined:
                errors.append(prefix + ":LATE_INTERNAL_CONTROL")
            observed.append(name)
    errors.extend(threading_external_inventory_errors(observed))
    return errors


def timestretch_snapshot(directory: Path) -> dict[str, str] | None:
    snapshot: dict[str, str] = {}
    for name in EXPECTED_OUTPUTS:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return None
        snapshot[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def write_timestretch_sentinels(directory: Path, seed: str) -> dict[str, str]:
    directory.mkdir()
    for index, name in enumerate(EXPECTED_OUTPUTS):
        (directory / name).write_bytes(
            "DSPark transaction sentinel {} {} {}\n".format(
                seed, index, name).encode("ascii"))
    snapshot = timestretch_snapshot(directory)
    if snapshot is None:
        raise RuntimeError("sentinel construction failed")
    return snapshot


def timestretch_transaction_residue(directory: Path) -> list[str]:
    return sorted(
        path.name for path in directory.iterdir()
        if path.name == TIMESTRETCH_TRANSACTION_DIRECTORY
        or path.name.startswith(".dspark-timestretch-transaction-")
    )


def validate_timestretch_rollback(
    directory: Path,
    before: dict[str, str],
    result: dict[str, object],
    phase: str,
    output_name: str,
) -> list[str]:
    errors: list[str] = []
    prefix = "TIMESTRETCH_{}_ROLLBACK:{}".format(phase, output_name)
    expected_stderr = "ERROR TIMESTRETCH_TRANSACTION_{} {}\n".format(
        phase, output_name)
    if result.get("exit_code") != 3 or result.get("errors") != []:
        errors.append(prefix + ":PROCESS")
    if result.get("stderr") != expected_stderr:
        errors.append(prefix + ":TYPED_TERMINAL")
    if "characterisation written" in str(result.get("stdout")):
        errors.append(prefix + ":FALSE_SUCCESS")
    if sorted(path.name for path in directory.iterdir()) \
            != sorted(EXPECTED_OUTPUTS):
        errors.append(prefix + ":FINAL_CENSUS")
    after = timestretch_snapshot(directory)
    if after != before:
        errors.append(prefix + ":SENTINEL_HASH_DRIFT")
    if timestretch_transaction_residue(directory):
        errors.append(prefix + ":TRANSACTION_RESIDUE")
    return errors


def timestretch_live(root: Path,
                     discovery: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(
            prefix="dspark-timestretch-live-") as directory:
        scratch = Path(directory)
        for compiler in discovery:
            label = str(compiler["label"])
            binary = scratch / ("characterize-" + label)
            instrumented = scratch / ("characterize-transaction-" + label)
            base_command = [
                str(compiler["path"]), "-std=c++20", "-O2", "-Wall",
                "-Wextra", "-Wpedantic", "-Werror", "-I", str(root),
                str(root / "tools/characterize_timestretch.cpp"),
            ]
            build = producer.run_owned(
                [*base_command, "-o", str(binary)], scratch, 240, "compile")
            test_build = producer.run_owned(
                [*base_command,
                 "-DDSPARK_TIMESTRETCH_TRANSACTION_TESTING=1",
                 "-o", str(instrumented)], scratch, 240, "compile")
            if build["exit_code"] != 0 or build["errors"]:
                errors.append("TIMESTRETCH_BUILD:" + label)
                continue
            if test_build["exit_code"] != 0 or test_build["errors"]:
                errors.append("TIMESTRETCH_INSTRUMENTED_BUILD:" + label)
                continue

            missing = scratch / ("missing-" + label)
            result = producer.run_owned(
                [str(binary), str(missing)], scratch, 180, "run")
            if result["exit_code"] != 2 \
                    or "ERROR TIMESTRETCH_OUTPUT_DIRECTORY" not in result["stderr"] \
                    or "characterisation written" in result["stdout"] \
                    or missing.exists():
                errors.append("TIMESTRETCH_MISSING_DIRECTORY:" + label)
            regular = scratch / ("regular-" + label)
            regular.write_bytes(b"not a directory\n")
            result = producer.run_owned(
                [str(binary), str(regular)], scratch, 180, "run")
            if result["exit_code"] != 2 \
                    or "ERROR TIMESTRETCH_OUTPUT_DIRECTORY" not in result["stderr"] \
                    or "characterisation written" in result["stdout"]:
                errors.append("TIMESTRETCH_REGULAR_FILE:" + label)

            nonregular = scratch / ("nonregular-final-" + label)
            nonregular.mkdir()
            (nonregular / EXPECTED_OUTPUTS[0]).mkdir()
            result = producer.run_owned(
                [str(binary), str(nonregular)], scratch, 180, "run")
            if result["exit_code"] != 3 \
                    or result["stderr"] != (
                        "ERROR TIMESTRETCH_TRANSACTION_PRECHECK {}\n".format(
                            EXPECTED_OUTPUTS[0])) \
                    or not (nonregular / EXPECTED_OUTPUTS[0]).is_dir() \
                    or len(list(nonregular.iterdir())) != 1:
                errors.append("TIMESTRETCH_NONREGULAR_FINAL:" + label)

            symlink_root = scratch / ("symlink-final-" + label)
            symlink_root.mkdir()
            symlink_target = scratch / ("symlink-target-" + label)
            symlink_target.write_bytes(b"caller-owned symlink target\n")
            (symlink_root / EXPECTED_OUTPUTS[0]).symlink_to(symlink_target)
            target_before = hashlib.sha256(symlink_target.read_bytes()).hexdigest()
            result = producer.run_owned(
                [str(binary), str(symlink_root)], scratch, 180, "run")
            if result["exit_code"] != 3 \
                    or result["stderr"] != (
                        "ERROR TIMESTRETCH_TRANSACTION_PRECHECK {}\n".format(
                            EXPECTED_OUTPUTS[0])) \
                    or not (symlink_root / EXPECTED_OUTPUTS[0]).is_symlink() \
                    or hashlib.sha256(symlink_target.read_bytes()).hexdigest() \
                    != target_before:
                errors.append("TIMESTRETCH_SYMLINK_FINAL:" + label)

            collision = scratch / ("collision-" + label)
            collision_before = write_timestretch_sentinels(
                collision, label + "-collision")
            (collision / TIMESTRETCH_TRANSACTION_DIRECTORY).mkdir()
            result = producer.run_owned(
                [str(binary), str(collision)], scratch, 180, "run")
            (collision / TIMESTRETCH_TRANSACTION_DIRECTORY).rmdir()
            if result["exit_code"] != 3 \
                    or result["stderr"] != (
                        "ERROR TIMESTRETCH_TRANSACTION_COLLISION {}\n".format(
                            TIMESTRETCH_TRANSACTION_DIRECTORY)) \
                    or timestretch_snapshot(collision) != collision_before:
                errors.append("TIMESTRETCH_STAGE_COLLISION:" + label)

            for phase, variable in TIMESTRETCH_FAILURE_PHASES:
                for index, output_name in enumerate(EXPECTED_OUTPUTS):
                    failure_root = scratch / (
                        "{}-failure-{}-{}".format(
                            phase.lower(), label, index))
                    before = write_timestretch_sentinels(
                        failure_root,
                        "{}-{}-{}".format(label, phase.lower(), index))
                    environment = dict(os.environ)
                    for _other_phase, other_variable \
                            in TIMESTRETCH_FAILURE_PHASES:
                        environment.pop(other_variable, None)
                    environment[variable] = str(index)
                    result = producer.run_owned(
                        [str(instrumented), str(failure_root)],
                        scratch, 180, "run", environment)
                    errors.extend(validate_timestretch_rollback(
                        failure_root, before, result, phase, output_name))

            output = scratch / ("output-" + label)
            output.mkdir()
            production_environment = dict(os.environ)
            production_environment.update({
                "DSPARK_TIMESTRETCH_FAIL_STAGE_INDEX": "0",
                "DSPARK_TIMESTRETCH_FAIL_COMMIT_INDEX": "0",
            })
            result = producer.run_owned(
                [str(binary), str(output)], scratch, 180, "run",
                production_environment)
            names = sorted(path.name for path in output.iterdir())
            if result["exit_code"] != 0 or result["errors"] \
                    or result["stderr"] != "" \
                    or result["stdout"] != (
                        "characterisation written to {}\n".format(output)) \
                    or names != sorted(EXPECTED_OUTPUTS) \
                    or timestretch_transaction_residue(output):
                errors.append("TIMESTRETCH_SUCCESS_TRANSACTION:" + label)
                continue
            first_success = timestretch_snapshot(output)
            if first_success is None:
                errors.append("TIMESTRETCH_OUTPUT_ARTIFACT:" + label)
                continue
            rerun = producer.run_owned(
                [str(binary), str(output)], scratch, 180, "run")
            if rerun["exit_code"] != 0 or rerun["errors"] \
                    or rerun["stderr"] != "" \
                    or timestretch_snapshot(output) != first_success \
                    or timestretch_transaction_residue(output):
                errors.append("TIMESTRETCH_SUCCESSFUL_REPLACEMENT:" + label)
            errors.extend(
                error + ":" + label for error in measurement_errors(output))
    return errors


def flac_live(root: Path,
              discovery: list[dict[str, object]]) -> list[str]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="dspark-flac-live-") as directory:
        scratch = Path(directory)
        for compiler in discovery:
            label = str(compiler["label"])
            command = [
                str(compiler["path"]), "-std=c++20", "-O3", "-Wall",
                "-Wextra", "-Wpedantic", "-Werror",
                "-fno-omit-frame-pointer",
                "-fsanitize=address,undefined,float-cast-overflow",
                "-fno-sanitize-recover=all",
                '-DDSPARK_TEST_FIXTURE_DIR="{}"'.format(
                    root / "tests/fixtures"),
                "-I", str(root), "-c", str(root / "tests/TestIO.cpp"),
                "-o", str(scratch / ("TestIO-" + label + ".o")),
            ]
            result = producer.run_owned(command, scratch, 240, "compile")
            if result["exit_code"] != 0 or result["errors"]:
                errors.append("FLAC_STRICT_COMPILE:" + label)
    return errors


def live_mode(root: Path) -> int:
    if os.name != "posix" or not Path("/proc").is_dir():
        print("ERROR UNSUPPORTED_PROCESS_TREE_PLATFORM", file=sys.stderr)
        return 1
    if ctest_mode(root) != 0:
        return 1
    with tempfile.TemporaryDirectory(
            prefix="dspark-global-outer-") as directory:
        transcript_path = Path(directory) / "producer-transcript.json"
        producer_result = run_live_producer(root, transcript_path)
        if not transcript_path.is_file():
            print("ERROR VALIDATION_OUTER_TRANSCRIPT_MISSING", file=sys.stderr)
            return 1
        try:
            transcript = json.loads(transcript_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            print("ERROR VALIDATION_OUTER_TRANSCRIPT_PARSE " + str(error),
                  file=sys.stderr)
            return 1
        producer_exit_value = producer_result.get("exit_code")
        producer_exit = producer_exit_value \
            if isinstance(producer_exit_value, int) else -1
        errors = validate_transcript(transcript, producer_exit)
        if producer_result.get("errors"):
            errors.append("VALIDATION_OUTER_PRODUCER_PROCESS:" + str(
                producer_result["errors"]))
        discovery = transcript.get("compiler_discovery", [])
        if not errors:
            errors.extend(generic_gcc_live(root))
        if not errors:
            errors.extend(threading_external_live(root))
        if not errors:
            errors.extend(timestretch_live(root, discovery))
            errors.extend(flac_live(root, discovery))
        for error in errors:
            print("ERROR " + error, file=sys.stderr)
        if errors:
            return 1
    print("PASS global validation contract live: 2 primary discovery, "
          "1 generic GCC13 fallback, 4 source, 16 execution rows, "
          "33 threading mutations, 36 TimeStretch rollback injections, "
          "4 green baselines, 12 exact expected REDs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--ctest", action="store_true")
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    if arguments.self_test:
        return self_test(root)
    if arguments.ctest:
        return ctest_mode(root)
    return live_mode(root)


if __name__ == "__main__":
    raise SystemExit(main())
