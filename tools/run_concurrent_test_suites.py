#!/usr/bin/env python3
"""Run complete DSPark test-suite processes concurrently from one directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


class HarnessFailure(RuntimeError):
    pass


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


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise HarnessFailure(
            "command failed (%d): %s\n%s"
            % (completed.returncode, " ".join(command), detail)
        )
    return completed


def _build(repo: Path, root: Path, mutant: str | None) -> tuple[Path, list[list[str]]]:
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
    _run(configure, repo)
    build_command = ["cmake", "--build", os.fspath(build), "--target", "dspark_tests", "--parallel"]
    _run(build_command, repo)
    names = ("dspark_tests", "dspark_tests.exe")
    candidates = [path for name in names for path in build.rglob(name) if path.is_file()]
    if len(candidates) != 1:
        raise HarnessFailure(f"expected one dspark_tests executable, found {candidates}")
    return candidates[0], [configure, build_command]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_tail(output: str, lines: int = 12) -> list[str]:
    return output.splitlines()[-lines:]


def _stop_processes(processes: list[subprocess.Popen[str]]) -> list[int]:
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        process.terminate()
    terminate_deadline = time.monotonic() + 5.0
    while any(process.poll() is None for process in alive) and time.monotonic() < terminate_deadline:
        time.sleep(0.05)
    killed: list[int] = []
    for process in alive:
        if process.poll() is None:
            killed.append(process.pid)
            process.kill()
    for process in alive:
        process.wait()
    return killed


def _launch(executable: Path, copies: int, shared: Path,
            temporary_parent: Path, log_directory: Path,
            timeout_seconds: int, synchronization_root: Path | None,
            synchronization_mutant: str | None
            ) -> tuple[list[dict[str, object]], list[str], str | None]:
    base_environment = dict(os.environ)
    for variable in ("TMPDIR", "TMP", "TEMP"):
        base_environment[variable] = os.fspath(temporary_parent)
    for variable in (
        _SYNC_ROOT, _SYNC_IDENTITY, _SYNC_PARTICIPANTS, _SYNC_TIMEOUT, _SYNC_MUTANT
    ):
        base_environment.pop(variable, None)

    processes: list[subprocess.Popen[str]] = []
    log_paths: list[tuple[Path, Path]] = []
    log_streams: list[object] = []
    identities: list[int] = []
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
        stdout_stream = stdout_path.open("w", encoding="utf-8", newline="\n")
        stderr_stream = stderr_path.open("w", encoding="utf-8", newline="\n")
        log_paths.append((stdout_path, stderr_path))
        log_streams.extend((stdout_stream, stderr_stream))
        processes.append(
            subprocess.Popen(
                [os.fspath(executable)],
                cwd=shared,
                env=environment,
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
            )
        )

    observed_roots: set[str] = set()
    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    killed: list[int] = []
    early_diagnostic: str | None = None
    try:
        while any(process.poll() is None for process in processes):
            observed_roots.update(
                path.name
                for path in temporary_parent.iterdir()
                if path.name.startswith("dspark-testio-")
            )
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
                        killed = _stop_processes(processes)
                        break
                if early_diagnostic is not None:
                    break
            if time.monotonic() >= deadline:
                timed_out = True
                killed = _stop_processes(processes)
                break
            time.sleep(0.01)
    except BaseException:
        _stop_processes(processes)
        raise
    finally:
        for stream in log_streams:
            stream.close()
    observed_roots.update(
        path.name
        for path in temporary_parent.iterdir()
        if path.name.startswith("dspark-testio-")
    )

    results: list[dict[str, object]] = []
    for index, (process, paths) in enumerate(zip(processes, log_paths)):
        process.wait()
        stdout = paths[0].read_text(encoding="utf-8", errors="replace")
        stderr = paths[1].read_text(encoding="utf-8", errors="replace")
        results.append(
            {
                "copy": index,
                "identity": identities[index],
                "exit": process.returncode,
                "reported_atomic_claim_failure": (
                    "FAIL: TestIO_process_temporary_root_is_exclusive" in stdout
                    or "FAIL: TestIO_process_temporary_root_is_exclusive" in stderr
                ),
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
                "stdout_tail": _summary_tail(stdout),
                "stderr_tail": _summary_tail(stderr),
            }
        )
    if timed_out:
        exits = [process.returncode for process in processes]
        tails = {
            int(result["copy"]): {
                "stdout": result["stdout_tail"][-3:],
                "stderr": result["stderr_tail"][-3:],
            }
            for result in results
        }
        raise HarnessFailure(
            "concurrent suites exceeded %d seconds; terminate exits=%s "
            "kill_pids=%s tails=%s"
            % (timeout_seconds, exits, killed, tails)
        )
    return results, sorted(observed_roots), early_diagnostic


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
            raise HarnessFailure(
                f"duplicate synchronization outcome for identity {identity}"
            )
        outcomes[identity] = outcome
    if set(outcomes) != set(range(copies)):
        raise HarnessFailure(
            "incomplete synchronization outcomes: " + repr(outcomes)
        )
    return outcomes


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.copies < 3:
        raise HarnessFailure("at least three concurrent copies are required")
    if args.timeout_seconds <= 0:
        raise HarnessFailure("timeout-seconds must be positive")
    if args.expect_collision != (args.mutant == "fixed-testio-names"):
        raise HarnessFailure("the collision expectation requires the fixed-name mutant")
    if args.sync_mutant is not None and args.mutant != "fixed-testio-names":
        raise HarnessFailure("a synchronization mutant requires the fixed-name mutant")

    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="dspark-concurrent-suites-") as directory:
        root = Path(directory)
        shared = root / "shared"
        temporary_parent = root / "temporary"
        log_directory = root / "logs"
        synchronization_root = (
            root / "synchronization" if args.mutant == "fixed-testio-names" else None
        )
        shared.mkdir()
        temporary_parent.mkdir()
        log_directory.mkdir()
        if synchronization_root is not None:
            synchronization_root.mkdir()
        build_commands: list[list[str]] = []
        if args.executable is None:
            executable, build_commands = _build(repo, root, args.mutant)
        else:
            executable = args.executable.resolve(strict=True)
            if args.mutant is not None:
                raise HarnessFailure("an external executable cannot establish the requested mutant")

        results, observed_roots, early_diagnostic = _launch(
            executable, args.copies, shared, temporary_parent, log_directory,
            args.timeout_seconds, synchronization_root, args.sync_mutant
        )
        if args.sync_mutant is not None:
            expected = _SYNC_DIAGNOSTICS[args.sync_mutant]
            if early_diagnostic is None or expected not in early_diagnostic:
                raise HarnessFailure(
                    f"synchronization mutant {args.sync_mutant} lacked {expected}"
                )
            raise HarnessFailure(
                f"synchronization mutant {args.sync_mutant} live red: "
                + early_diagnostic
            )
        exits = [int(result["exit"]) for result in results]
        shared_residue = sorted(path.name for path in shared.iterdir())
        temporary_residue = sorted(path.name for path in temporary_parent.iterdir())

        if args.expect_all_pass:
            if any(exits):
                raise HarnessFailure(f"candidate concurrent exits were {exits}")
            if len(observed_roots) < args.copies:
                raise HarnessFailure(
                    "did not observe one process-unique TestIO root per copy: "
                    + repr(observed_roots)
                )
            if shared_residue or temporary_residue:
                raise HarnessFailure(
                    "candidate left residue shared=%s temporary=%s"
                    % (shared_residue, temporary_residue)
                )
            status = "PASS"
            terminal = "all concurrent suites passed with unique cleaned roots"
        else:
            assert synchronization_root is not None
            outcomes = _synchronization_outcomes(
                synchronization_root, args.copies)
            winners = sorted(
                identity for identity, outcome in outcomes.items()
                if outcome == "winner"
            )
            losers = sorted(
                identity for identity, outcome in outcomes.items()
                if outcome == "loser"
            )
            if len(winners) != 1 or not losers:
                raise HarnessFailure(
                    f"atomic claim lacked one winner and at least one loser: {outcomes}"
            )
            by_identity = {int(result["identity"]): result for result in results}
            for identity in losers:
                result = by_identity[identity]
                if (int(result["exit"]) == 0
                        or not result["reported_atomic_claim_failure"]):
                    raise HarnessFailure(
                        f"losing identity {identity} lacked the exact atomic-claim failure"
                    )
            if by_identity[winners[0]]["reported_atomic_claim_failure"]:
                raise HarnessFailure("atomic-claim winner reported the loser failure")
            status = "PASS"
            terminal = (
                "fixed-name mutant produced one synchronized atomic winner and "
                f"{len(losers)} losers"
            )

        record: dict[str, object] = {
            "schema": "dspark.concurrent-test-suites.v1",
            "status": status,
            "terminal": terminal,
            "copies": args.copies,
            "mutant": args.mutant,
            "executable_sha256": _sha256(executable),
            "build_commands": build_commands,
            "results": results,
            "observed_testio_roots": observed_roots,
            "shared_residue": shared_residue,
            "temporary_residue": temporary_residue,
            "synchronization_outcomes": (
                outcomes if args.expect_collision else None
            ),
        }
        if args.output is not None:
            _write_json(args.output, record)
        print(
            "concurrent test suites: PASS copies=%d exits=%s roots=%d mutant=%s"
            % (args.copies, exits, len(observed_roots), args.mutant or "none")
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessFailure as error:
        print(f"concurrent test suites: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from None
