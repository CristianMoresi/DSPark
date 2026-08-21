#!/usr/bin/env python3
"""Run complete DSPark test-suite processes concurrently from one directory."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time


class HarnessFailure(RuntimeError):
    def __init__(self, message: str, *, classification: str = "runner_internal_error",
                 phase: str = "runner",
                 details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.classification = classification
        self.phase = phase
        self.details = details or {}


_SYNC_ROOT = "DSPARK_TESTIO_COLLISION_SYNC_ROOT"
_SYNC_IDENTITY = "DSPARK_TESTIO_COLLISION_SYNC_IDENTITY"
_SYNC_PARTICIPANTS = "DSPARK_TESTIO_COLLISION_SYNC_PARTICIPANTS"
_SYNC_TIMEOUT = "DSPARK_TESTIO_COLLISION_SYNC_TIMEOUT_MS"
_SYNC_MUTANT = "DSPARK_TESTIO_COLLISION_SYNC_MUTANT"
_SYNC_DIAGNOSTICS = {
    "missing-readiness": "DSPARK_TESTIO_SYNC_TIMEOUT: readiness",
    "duplicate-identity": "DSPARK_TESTIO_SYNC_ERROR: duplicate process identity",
    "missing-attempted": "DSPARK_TESTIO_SYNC_TIMEOUT: attempted",
}
_TAIL_LINES = 12
_TAIL_LINE_CHARACTERS = 320
_MAX_COPIES = 64
_TERMINATE_GRACE_SECONDS = 5.0
_PR_SET_CHILD_SUBREAPER = 36


def _bounded_line(line: str, limit: int = _TAIL_LINE_CHARACTERS) -> str:
    if len(line) <= limit:
        return line
    marker = f"...<truncated {len(line) - limit} chars>..."
    side = max(1, (limit - len(marker)) // 2)
    return line[:side] + marker + line[-side:]


def _summary_tail(output: bytes, lines: int = _TAIL_LINES) -> list[str]:
    decoded = output.decode("utf-8", errors="replace")
    return [_bounded_line(line) for line in decoded.splitlines()[-lines:]]


def _stream_result(output: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(output).hexdigest(),
        "size": len(output),
        "tail": _summary_tail(output),
    }


def _run(command: list[str], display_command: list[str], cwd: Path,
         phase: str, history: list[dict[str, object]]) -> None:
    completed = subprocess.run(
        command, cwd=cwd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    result: dict[str, object] = {
        "phase": phase, "command": display_command, "exit": completed.returncode
    }
    if completed.returncode != 0:
        result.update(
            stdout=_stream_result(completed.stdout),
            stderr=_stream_result(completed.stderr),
        )
        history.append(result)
        detail = _summary_tail(completed.stderr) or _summary_tail(completed.stdout)
        raise HarnessFailure(
            "command failed (%d) during %s: %s; tail=%s"
            % (completed.returncode, phase, " ".join(display_command), detail or ["no diagnostic"]),
            classification="build_configuration_failure",
            phase=phase,
            details={"command_failure": result},
        )
    history.append(result)


def _build(repo: Path, root: Path, mutant: str | None,
           history: list[dict[str, object]]) -> Path:
    build = root / "build"
    compiler = os.environ.get("CXX", "c++")
    configure = [
        "cmake",
        "-S",
        os.fspath(repo),
        "-B",
        os.fspath(build),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CXX_COMPILER={compiler}",
        "-DCMAKE_COMPILE_WARNING_AS_ERROR=ON",
        "-DDSPARK_BUILD_TESTS=ON",
        "-DDSPARK_BUILD_CONFORMANCE=OFF",
    ]
    if shutil.which("ninja"):
        configure.extend(("-G", "Ninja"))
    if mutant == "fixed-testio-names":
        configure.append("-DCMAKE_CXX_FLAGS=-DDSPARK_TESTIO_FIXED_NAME_MUTANT=1")
    display = lambda item: (
        "$BUILD" if item == os.fspath(build) else
        "$REPO" if item == os.fspath(repo) else item
    )
    _run(configure, [display(item) for item in configure], repo, "configure", history)
    build_command = ["cmake", "--build", os.fspath(build), "--target", "dspark_tests", "--parallel"]
    _run(build_command, [display(item) for item in build_command], repo, "build", history)
    names = ("dspark_tests", "dspark_tests.exe")
    candidates = [path for name in names for path in build.rglob(name) if path.is_file()]
    if len(candidates) != 1:
        _fail(f"expected one dspark_tests executable, found {len(candidates)}",
              "build_configuration_failure", "executable-discovery")
    return candidates[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_alive(process: subprocess.Popen[bytes]) -> bool:
    if os.name != "posix":
        return process.poll() is None
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _enable_child_subreaper() -> None:
    """Adopt descendants on Linux so killed process trees can be reaped."""
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _reap_process_group(process: subprocess.Popen[bytes]) -> None:
    while sys.platform.startswith("linux"):
        try:
            if os.waitpid(-process.pid, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def _signal_process_tree(process: subprocess.Popen[bytes], force: bool) -> None:
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    else:
        (process.kill if force else process.terminate)()


def _stop_processes(processes: list[subprocess.Popen[bytes]], reason: str
                    ) -> dict[int, dict[str, object]]:
    targeted = [process for process in processes if _group_alive(process)]
    states: dict[int, dict[str, object]] = {
        process.pid: {"reason": reason, "terminate_requested": True,
                      "kill_requested": False} for process in targeted
    }
    for process in targeted:
        try:
            _signal_process_tree(process, False)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while any(_group_alive(process) for process in targeted) and time.monotonic() < deadline:
        time.sleep(0.05)
    for process in targeted:
        if _group_alive(process):
            states[process.pid]["kill_requested"] = True
            try:
                _signal_process_tree(process, True)
            except ProcessLookupError:
                pass
    for process in targeted:
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            states[process.pid]["wait_failed_after_kill"] = True
    group_deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while any(_group_alive(process) for process in targeted) and time.monotonic() < group_deadline:
        for process in targeted:
            _reap_process_group(process)
        time.sleep(0.01)
    for process in targeted:
        _reap_process_group(process)
        states[process.pid]["group_alive_after_stop"] = _group_alive(process)
    return states


def _launch(executable: Path, copies: int, shared: Path,
            temporary_parent: Path, log_directory: Path,
            timeout_seconds: int, synchronization_root: Path | None,
            synchronization_mutant: str | None
            ) -> dict[str, object]:
    base_environment = dict(os.environ)
    base_environment.update(
        {variable: os.fspath(temporary_parent) for variable in ("TMPDIR", "TMP", "TEMP")}
    )
    for variable in (
        _SYNC_ROOT, _SYNC_IDENTITY, _SYNC_PARTICIPANTS, _SYNC_TIMEOUT, _SYNC_MUTANT
    ):
        base_environment.pop(variable, None)

    processes: list[subprocess.Popen[bytes]] = []
    log_paths: list[tuple[Path, Path]] = []
    identities: list[int] = []
    observed_roots: set[str] = set()
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    early_diagnostic: str | None = None
    termination: dict[int, dict[str, object]] = {}
    launch_error: str | None = None
    def observe_roots() -> None:
        observed_roots.update(
            path.name for path in temporary_parent.iterdir()
            if path.name.startswith("dspark-testio-")
        )
    try:
        _enable_child_subreaper()
        for index in range(copies):
            identity = 0 if synchronization_mutant == "duplicate-identity" and index == 1 else index
            identities.append(identity)
            environment = dict(base_environment)
            if synchronization_root is not None:
                environment[_SYNC_ROOT] = os.fspath(synchronization_root)
                environment[_SYNC_IDENTITY] = str(identity)
                environment[_SYNC_PARTICIPANTS] = str(copies)
                environment[_SYNC_TIMEOUT] = "1000" if synchronization_mutant else "30000"
                if (synchronization_mutant in ("missing-readiness", "missing-attempted")
                        and index == copies - 1):
                    environment[_SYNC_MUTANT] = synchronization_mutant
            stdout_path = log_directory / f"copy-{index}.stdout"
            stderr_path = log_directory / f"copy-{index}.stderr"
            with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
                process = subprocess.Popen(
                    [os.fspath(executable)],
                    cwd=shared,
                    env=environment,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    start_new_session=(os.name == "posix"),
                )
            processes.append(process)
            log_paths.append((stdout_path, stderr_path))
        while any(process.poll() is None for process in processes):
            observe_roots()
            if synchronization_mutant is not None:
                expected = _SYNC_DIAGNOSTICS[synchronization_mutant]
                for _, stderr_path in log_paths:
                    stderr = stderr_path.read_text(
                        encoding="utf-8", errors="replace")
                    matching = next(
                        (line.strip() for line in stderr.splitlines()
                         if expected in line),
                        None,
                    )
                    if matching is not None:
                        early_diagnostic = matching
                        break
                if early_diagnostic is not None:
                    termination = _stop_processes(
                        processes, "synchronization_mutant_diagnostic"
                    )
                    break
            if time.monotonic() >= deadline:
                timed_out = True
                termination = _stop_processes(processes, "timeout")
                break
            time.sleep(0.01)
    except BaseException as error:
        launch_error = f"{type(error).__name__}: {error}"
        termination = _stop_processes(processes, "launch_or_monitor_failure")
    if not termination:
        termination = _stop_processes(processes, "post_exit_descendant_cleanup")
    observe_roots()

    results: list[dict[str, object]] = []
    for index, (process, paths) in enumerate(zip(processes, log_paths)):
        process.wait()
        stdout = paths[0].read_bytes()
        stderr = paths[1].read_bytes()
        stop_state = termination.get(process.pid, {})
        results.append(
            {
                "copy": index, "identity": identities[index], "exit": process.returncode,
                "timed_out": stop_state.get("reason") == "timeout",
                "termination": stop_state or None,
                "reported_atomic_claim_failure": (
                    b"FAIL: TestIO_process_temporary_root_is_exclusive" in stdout
                    or b"FAIL: TestIO_process_temporary_root_is_exclusive" in stderr
                ),
                "stdout_size": len(stdout), "stderr_size": len(stderr),
                "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
                "stdout_tail": _summary_tail(stdout), "stderr_tail": _summary_tail(stderr),
            }
        )
    return {
        "results": results,
        "observed_roots": sorted(observed_roots),
        "early_diagnostic": early_diagnostic,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "log_paths": log_paths,
    }


def _synchronization_outcomes(root: Path, copies: int) -> dict[int, str]:
    pattern = re.compile(r"attempted-(\d+)-(winner|loser)\Z")
    outcomes: dict[int, str] = {}
    for path in root.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        identity = int(match.group(1))
        outcome = match.group(2)
        if identity in outcomes:
            _fail(f"duplicate synchronization outcome for identity {identity}",
                  "collision_oracle_failure", "collision-validation")
        outcomes[identity] = outcome
    if set(outcomes) != set(range(copies)):
        _fail("incomplete synchronization outcomes: " + repr(outcomes),
              "collision_oracle_failure", "collision-validation")
    return outcomes


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_logs(target: Path, log_paths: list[tuple[Path, Path]],
                  results: list[dict[str, object]]) -> list[dict[str, object]]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _reject(f"diagnostic directory already exists: {target}", "diagnostic-preflight")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    manifest: list[dict[str, object]] = []
    try:
        for copy, sources in enumerate(log_paths):
            for stream_name, source in zip(("stdout", "stderr"), sources):
                name = f"copy-{copy}.{stream_name}"
                destination = staging / name
                shutil.copyfile(source, destination)
                size = destination.stat().st_size
                digest = _sha256(destination)
                expected_size = results[copy][f"{stream_name}_size"]
                expected_digest = results[copy][f"{stream_name}_sha256"]
                if size != expected_size or digest != expected_digest:
                    raise HarnessFailure(
                        f"diagnostic stream integrity mismatch: {name}",
                        classification="diagnostic_persistence_failure",
                        phase="diagnostic-persistence",
                        details={
                            "name": name, "expected_size": expected_size,
                            "actual_size": size, "expected_sha256": expected_digest,
                            "actual_sha256": digest,
                        },
                    )
                manifest.append({
                    "copy": copy, "stream": stream_name, "name": name,
                    "size": size, "sha256": digest,
                })
        _write_json(staging / "streams.json", {
            "schema": "dspark.concurrent-test-streams.v1", "files": manifest,
        })
        if target.exists() or target.is_symlink():
            _fail(f"diagnostic directory appeared during publication: {target}",
                  "diagnostic_persistence_failure", "diagnostic-persistence")
        os.rename(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _fit_tail(lines: list[str], budget: int) -> list[str]:
    selected: list[str] = []
    remaining = max(0, budget)
    for line in reversed(lines):
        if remaining <= 0:
            break
        line = _bounded_line(line, min(_TAIL_LINE_CHARACTERS, remaining))
        selected.append(line)
        remaining -= len(line)
    return list(reversed(selected))


def _failure_diagnostic(record: dict[str, object]) -> str:
    results = record.get("results", [])
    assert isinstance(results, list)
    failing = [
        result for result in results
        if isinstance(result, dict)
        and (
            int(result.get("exit", 0)) != 0
            or bool(result.get("timed_out"))
            or result.get("termination") is not None
        )
    ]
    per_stream_budget = max(96, min(1280, 24000 // max(2, len(failing) * 2)))
    children = [{
        "copy": item["copy"], "exit": item["exit"],
        "timed_out": item["timed_out"], "termination": item["termination"],
        "stdout_sha256": item["stdout_sha256"],
        "stderr_sha256": item["stderr_sha256"],
        "stdout_tail": _fit_tail(item["stdout_tail"], per_stream_budget),  # type: ignore[arg-type]
        "stderr_tail": _fit_tail(item["stderr_tail"], per_stream_budget),  # type: ignore[arg-type]
    } for item in failing]
    failure = record.get("failure")
    command_failure: dict[str, object] | None = None
    if isinstance(failure, dict):
        details = failure.get("details")
        if isinstance(details, dict) and isinstance(details.get("command_failure"), dict):
            command_failure = details["command_failure"]
    return json.dumps({
        "classification": record.get("classification"),
        "terminal": record.get("terminal"),
        "children": children,
        "command_failure": command_failure,
    }, sort_keys=True, separators=(",", ":"))


def _mark_failure(record: dict[str, object], error: HarnessFailure) -> None:
    record.update(status="FAIL", classification=error.classification, terminal=str(error))
    record["failure"] = {
        "phase": error.phase,
        "message": str(error),
        "details": error.details,
    }


def _finalize(args: argparse.Namespace, record: dict[str, object]) -> int:
    if args.output is not None:
        try:
            _write_json(args.output, record)
        except BaseException as error:
            _mark_failure(record, HarnessFailure(
                f"could not atomically persist --output: {type(error).__name__}: {error}",
                classification="output_persistence_failure",
                phase="output-persistence",
            ))
    if record["status"] == "PASS":
        results = record.get("results", [])
        exits = [result["exit"] for result in results]  # type: ignore[index]
        print(
            "concurrent test suites: PASS copies=%d exits=%s roots=%d mutant=%s"
            % (args.copies, exits, len(record.get("observed_testio_roots", [])),
               args.mutant or "none")
        )
        return 0
    print("concurrent test suites: FAIL " + _failure_diagnostic(record), file=sys.stderr)
    return 1


def _base_record(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema": "dspark.concurrent-test-suites.v1", "status": "FAIL",
        "classification": "runner_internal_error",
        "terminal": "runner did not reach classification",
        "copies": args.copies, "copies_launched": 0,
        "expectation": "collision" if args.expect_collision else "all_pass",
        "mutant": args.mutant, "synchronization_mutant": args.sync_mutant,
        "timeout_seconds": args.timeout_seconds, "timeout_triggered": False,
        "executable_sha256": None, "build_commands": [], "results": [],
        "observed_testio_roots": [], "shared_residue": [], "temporary_residue": [],
        "synchronization_artifacts": [], "synchronization_outcomes": None,
        "diagnostic_streams": [], "failure": None,
    }


def _reject(message: str, phase: str = "argument-validation") -> None:
    raise HarnessFailure(message, classification="configuration_failure", phase=phase)


def _fail(message: str, classification: str, phase: str) -> None:
    raise HarnessFailure(message, classification=classification, phase=phase)


def _validate_args(args: argparse.Namespace) -> None:
    if args.copies < 3 or args.copies > _MAX_COPIES:
        _reject(f"copies must be in [3, {_MAX_COPIES}]")
    if args.timeout_seconds <= 0:
        _reject("timeout-seconds must be positive")
    if args.expect_collision != (args.mutant == "fixed-testio-names"):
        _reject("the collision expectation requires the fixed-name mutant")
    if args.sync_mutant is not None and args.mutant != "fixed-testio-names":
        _reject("a synchronization mutant requires the fixed-name mutant")
    if args.executable is not None and args.mutant is not None:
        _reject("an external executable cannot establish the requested mutant")
    if args.diagnostic_directory is not None:
        target = args.diagnostic_directory
        if target.exists() or target.is_symlink():
            _reject(f"diagnostic directory already exists: {target}", "diagnostic-preflight")
        if args.output is not None and target.absolute() == args.output.absolute():
            _reject("--output and --diagnostic-directory must be distinct")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--working-directory", choices=("shared",), default="shared")
    parser.add_argument("--mutant", choices=("fixed-testio-names",))
    parser.add_argument("--sync-mutant", choices=tuple(_SYNC_DIAGNOSTICS))
    expectation = parser.add_mutually_exclusive_group(required=True)
    expectation.add_argument("--expect-all-pass", action="store_true")
    expectation.add_argument("--expect-collision", action="store_true")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostic-directory", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def _executable(args: argparse.Namespace, repo: Path, root: Path,
                record: dict[str, object]) -> Path:
    if args.executable is None:
        history = record["build_commands"]
        assert isinstance(history, list)
        return _build(repo, root, args.mutant, history)
    try:
        executable = args.executable.resolve(strict=True)
    except OSError as error:
        _reject(f"external executable unavailable: {type(error).__name__}: {error}",
                "executable-resolution")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        _reject(f"external executable is not an executable file: {executable}",
                "executable-resolution")
    return executable


def _evaluate(args: argparse.Namespace, record: dict[str, object],
              outcome: dict[str, object], synchronization_root: Path | None) -> None:
    if outcome["launch_error"] is not None:
        _fail("child launch or monitoring failed: " + str(outcome["launch_error"]),
              "launch_failure", "child-launch-or-monitor")
    if outcome["timed_out"]:
        _fail(f"concurrent suites exceeded {args.timeout_seconds} seconds",
              "timeout", "child-monitor")
    if args.sync_mutant is not None:
        expected = _SYNC_DIAGNOSTICS[args.sync_mutant]
        diagnostic = outcome["early_diagnostic"]
        if diagnostic is None or expected not in str(diagnostic):
            _fail(f"synchronization mutant {args.sync_mutant} lacked {expected}",
                  "synchronization_mutant_oracle_failure", "synchronization-validation")
        _fail(f"synchronization mutant {args.sync_mutant} live red: {diagnostic}",
              "synchronization_mutant_red", "synchronization-validation")

    results = record["results"]
    assert isinstance(results, list)
    exits = [int(result["exit"]) for result in results]
    if args.expect_all_pass:
        if any(exits):
            _fail(f"candidate concurrent exits were {exits}",
                  "child_nonzero", "candidate-validation")
        roots = record["observed_testio_roots"]
        if not isinstance(roots, list) or len(roots) < args.copies:
            _fail("did not observe one process-unique TestIO root per copy: " + repr(roots),
                  "candidate_oracle_failure", "candidate-validation")
        if record["shared_residue"] or record["temporary_residue"]:
            _fail("candidate left residue shared=%s temporary=%s"
                  % (record["shared_residue"], record["temporary_residue"]),
                  "candidate_oracle_failure", "candidate-validation")
        record.update(
            status="PASS", classification="all_pass",
            terminal="all concurrent suites passed with unique cleaned roots",
            failure=None,
        )
        return

    assert synchronization_root is not None
    outcomes = _synchronization_outcomes(synchronization_root, args.copies)
    record["synchronization_outcomes"] = outcomes
    winners = [identity for identity, value in outcomes.items() if value == "winner"]
    losers = [identity for identity, value in outcomes.items() if value == "loser"]
    if len(winners) != 1 or not losers:
        _fail(f"atomic claim lacked one winner and at least one loser: {outcomes}",
              "collision_oracle_failure", "collision-validation")
    by_identity = {int(result["identity"]): result for result in results}
    for identity in losers:
        result = by_identity[identity]
        if int(result["exit"]) == 0 or not result["reported_atomic_claim_failure"]:
            _fail(f"losing identity {identity} lacked the exact atomic-claim failure",
                  "collision_oracle_failure", "collision-validation")
    if by_identity[winners[0]]["reported_atomic_claim_failure"]:
        _fail("atomic-claim winner reported the loser failure",
              "collision_oracle_failure", "collision-validation")
    record.update(
        status="PASS", classification="expected_fixed_name_collision",
        terminal=("fixed-name mutant produced one synchronized atomic winner and "
                  f"{len(losers)} losers"), failure=None,
    )


def main() -> int:
    args = _parse_args()
    record = _base_record(args)
    try:
        _validate_args(args)
    except HarnessFailure as error:
        _mark_failure(record, error)
        return _finalize(args, record)
    try:
        temporary_context = tempfile.TemporaryDirectory(prefix="dspark-concurrent-suites-")
    except BaseException as error:
        _mark_failure(record, HarnessFailure(
            f"could not create scratch root: {type(error).__name__}: {error}",
            classification="runner_internal_error", phase="scratch-creation",
        ))
        return _finalize(args, record)

    with temporary_context as directory:
        root = Path(directory)
        shared = root / "shared"
        temporary_parent = root / "temporary"
        log_directory = root / "logs"
        synchronization_root = root / "synchronization" if args.mutant else None
        log_paths: list[tuple[Path, Path]] = []
        try:
            for path in (shared, temporary_parent, log_directory):
                path.mkdir()
            if synchronization_root is not None:
                synchronization_root.mkdir()

            executable = _executable(args, Path(__file__).resolve().parents[1], root, record)
            record["executable_sha256"] = _sha256(executable)
            outcome = _launch(
                executable, args.copies, shared, temporary_parent, log_directory,
                args.timeout_seconds, synchronization_root, args.sync_mutant,
            )
            log_paths = outcome["log_paths"]  # type: ignore[assignment]
            results = outcome["results"]
            record.update(
                results=results, copies_launched=len(results),  # type: ignore[arg-type]
                observed_testio_roots=outcome["observed_roots"],
                timeout_triggered=bool(outcome["timed_out"]),
            )
            record["shared_residue"] = sorted(path.name for path in shared.iterdir())
            record["temporary_residue"] = sorted(path.name for path in temporary_parent.iterdir())
            if synchronization_root is not None:
                record["synchronization_artifacts"] = sorted(
                    path.name for path in synchronization_root.iterdir())
            _evaluate(args, record, outcome, synchronization_root)
        except HarnessFailure as error:
            _mark_failure(record, error)
        except BaseException as error:
            _mark_failure(record, HarnessFailure(
                f"unexpected runner error: {type(error).__name__}: {error}",
                classification="runner_internal_error", phase="runner",
            ))

        if args.diagnostic_directory is not None and log_paths:
            try:
                record["diagnostic_streams"] = _publish_logs(
                    args.diagnostic_directory, log_paths, record["results"]  # type: ignore[arg-type]
                )
            except BaseException as error:
                prior_failure = record.get("failure")
                if isinstance(error, HarnessFailure):
                    error.details["prior_failure"] = prior_failure
                else:
                    error = HarnessFailure(
                        "could not atomically publish diagnostic streams: "
                        f"{type(error).__name__}: {error}",
                        classification="diagnostic_persistence_failure",
                        phase="diagnostic-persistence", details={"prior_failure": prior_failure},
                    )
                _mark_failure(record, error)
        return _finalize(args, record)


if __name__ == "__main__":
    raise SystemExit(main())
