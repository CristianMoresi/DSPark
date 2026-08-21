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
_EXPECTED_SUITE_TESTS = 885
_ATOMIC_CLAIM_TEST = "TestIO_process_temporary_root_is_exclusive"
_WINDOWS_TRAMPOLINE = "--windows-trampoline"
_WINDOWS_SELF_TEST_SUBJECT = "--windows-self-test-subject"
_WINDOWS_SELF_TEST_DESCENDANT = "--windows-self-test-descendant"
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_JOB_TERMINATION_EXIT = 0xC000013A

_PROGRESS_PATTERN = re.compile(r"^\[ (\d+)/(\d+) \] ([^\r\n]+)\r?$", re.MULTILINE)
_SUMMARY_PATTERN = re.compile(
    r"^[ \t]*(\d+) tests \| (\d+) passed \| (\d+) failed \| (\d+) ms[ \t]*\r?$",
    re.MULTILINE,
)
_FAILURE_PATTERN = re.compile(r"^[ \t]*FAIL: ([^\r\n]+?)[ \t]*\r?$", re.MULTILINE)
_EXCEPTION_PATTERN = re.compile(
    r"^[ \t]*EXCEPTION \[([^\]\r\n]+)\](?: [^\r\n]*)?[ \t]*\r?$",
    re.MULTILINE,
)


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JobObjectBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


def _last_windows_error(operation: str) -> OSError:
    error_number = ctypes.get_last_error()
    return OSError(error_number, f"{operation}: {ctypes.FormatError(error_number)}")


class _WindowsJob:
    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows Job Objects are unavailable on this platform")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        self._kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        self._kernel32.SetInformationJobObject.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32
        )
        self._kernel32.SetInformationJobObject.restype = ctypes.c_int
        self._kernel32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        self._kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        self._kernel32.QueryInformationJobObject.argtypes = (
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self._kernel32.QueryInformationJobObject.restype = ctypes.c_int
        self._kernel32.TerminateJobObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        self._kernel32.TerminateJobObject.restype = ctypes.c_int
        self._kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        self._kernel32.CloseHandle.restype = ctypes.c_int
        handle = self._kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise _last_windows_error("CreateJobObjectW")
        self._handle: int | None = int(handle)
        limits = _JobObjectExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._kernel32.SetInformationJobObject(
            ctypes.c_void_p(self._handle), _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = _last_windows_error("SetInformationJobObject")
            close_error: OSError | None = None
            try:
                self.close()
            except OSError as close_failure:
                close_error = close_failure
            if close_error is not None:
                raise OSError(f"{error}; cleanup also failed: {close_error}") from error
            raise error

    def _required_handle(self) -> int:
        if self._handle is None:
            raise OSError("Job Object handle is already closed")
        return self._handle

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        process_handle = int(process._handle)  # type: ignore[attr-defined]
        if not self._kernel32.AssignProcessToJobObject(
            ctypes.c_void_p(self._required_handle()), ctypes.c_void_p(process_handle)
        ):
            raise _last_windows_error("AssignProcessToJobObject")

    def active_processes(self) -> int:
        accounting = _JobObjectBasicAccountingInformation()
        if not self._kernel32.QueryInformationJobObject(
            ctypes.c_void_p(self._required_handle()),
            _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting), ctypes.sizeof(accounting), None
        ):
            raise _last_windows_error("QueryInformationJobObject")
        return int(accounting.ActiveProcesses)

    def terminate(self) -> None:
        if not self._kernel32.TerminateJobObject(
            ctypes.c_void_p(self._required_handle()), _WINDOWS_JOB_TERMINATION_EXIT
        ):
            raise _last_windows_error("TerminateJobObject")

    def close(self) -> None:
        handle = self._required_handle()
        if not self._kernel32.CloseHandle(ctypes.c_void_p(handle)):
            raise _last_windows_error("CloseHandle(JobObject)")
        self._handle = None


class _OwnedProcess:
    def __init__(self, process: subprocess.Popen[bytes], job: _WindowsJob | None = None,
                 assignment_before_release: bool = False,
                 release_completed: bool = False) -> None:
        self.process = process
        self.job = job
        self.assignment_before_release = assignment_before_release
        self.release_completed = release_completed

    @property
    def pid(self) -> int:
        return self.process.pid

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.process.wait(timeout=timeout)


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


def _parse_suite_output(stdout: bytes, stderr: bytes) -> dict[str, object]:
    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")
    progress = [
        (int(index), int(total), name)
        for index, total, name in _PROGRESS_PATTERN.findall(stdout_text)
    ]
    summaries = [
        {
            "total": int(total), "passed": int(passed),
            "failed": int(failed), "milliseconds": int(milliseconds),
        }
        for total, passed, failed, milliseconds in _SUMMARY_PATTERN.findall(stdout_text)
    ]
    failed_tests = _FAILURE_PATTERN.findall(stdout_text + "\n" + stderr_text)
    exception_tests = _EXCEPTION_PATTERN.findall(stdout_text + "\n" + stderr_text)
    indices = [item[0] for item in progress]
    totals = [item[1] for item in progress]
    names = [item[2] for item in progress]
    summary = summaries[0] if len(summaries) == 1 else None
    complete = bool(
        summary is not None
        and summary["total"] == _EXPECTED_SUITE_TESTS
        and summary["passed"] + summary["failed"] == _EXPECTED_SUITE_TESTS
        and summary["failed"] == len(failed_tests)
        and len(progress) == _EXPECTED_SUITE_TESTS
        and indices == list(range(1, _EXPECTED_SUITE_TESTS + 1))
        and all(total == _EXPECTED_SUITE_TESTS for total in totals)
        and len(set(names)) == _EXPECTED_SUITE_TESTS
    )
    names_payload = "\0".join(names).encode("utf-8", errors="surrogatepass")
    return {
        "complete": complete,
        "expected_total": _EXPECTED_SUITE_TESTS,
        "progress_count": len(progress),
        "progress_indices_exact": indices == list(range(1, len(indices) + 1)),
        "progress_totals": sorted(set(totals)),
        "progress_test_names_sha256": hashlib.sha256(names_payload).hexdigest(),
        "progress_unique_test_names": len(set(names)),
        "summary_count": len(summaries),
        "summary": summary,
        "failed_tests": failed_tests,
        "exception_tests": exception_tests,
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


def _leader(process: _OwnedProcess | subprocess.Popen[bytes]) -> subprocess.Popen[bytes]:
    return process.process if isinstance(process, _OwnedProcess) else process


def _group_alive(process: _OwnedProcess | subprocess.Popen[bytes]) -> bool:
    if os.name == "nt":
        if not isinstance(process, _OwnedProcess) or process.job is None:
            raise OSError("Windows process tree has no owned Job Object")
        return process.job.active_processes() != 0
    leader = _leader(process)
    leader.poll()
    try:
        os.killpg(leader.pid, 0)
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


def _reap_process_group(process: _OwnedProcess | subprocess.Popen[bytes]) -> None:
    leader = _leader(process)
    while sys.platform.startswith("linux"):
        try:
            if os.waitpid(-leader.pid, os.WNOHANG)[0] == 0:
                return
        except ChildProcessError:
            return


def _signal_process_tree(process: _OwnedProcess | subprocess.Popen[bytes],
                         force: bool) -> None:
    leader = _leader(process)
    if os.name == "posix":
        os.killpg(leader.pid, signal.SIGKILL if force else signal.SIGTERM)
        return
    if not isinstance(process, _OwnedProcess) or process.job is None:
        raise OSError("refusing leader-only Windows termination without a Job Object")
    process.job.terminate()


def _stop_posix_processes(
    processes: list[_OwnedProcess | subprocess.Popen[bytes]], reason: str
) -> dict[int, dict[str, object]]:
    targeted: list[_OwnedProcess | subprocess.Popen[bytes]] = []
    states: dict[int, dict[str, object]] = {}
    for process in processes:
        try:
            if _group_alive(process):
                targeted.append(process)
                states[process.pid] = {
                    "reason": reason,
                    "ownership_mode": "posix_process_group",
                    "terminate_requested": True,
                    "kill_requested": False,
                }
        except OSError as error:
            states[process.pid] = {
                "reason": reason,
                "ownership_mode": "posix_process_group",
                "liveness_query_failed": f"{type(error).__name__}: {error}",
                "terminate_requested": False,
                "kill_requested": False,
                "group_alive_after_stop": True,
            }
    for process in targeted:
        try:
            _signal_process_tree(process, False)
        except ProcessLookupError:
            pass
        except OSError as error:
            states[process.pid]["terminate_failed"] = f"{type(error).__name__}: {error}"
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        try:
            if not any(_group_alive(process) for process in targeted):
                break
        except OSError:
            break
        time.sleep(0.05)
    for process in targeted:
        try:
            alive = _group_alive(process)
        except OSError as error:
            states[process.pid]["liveness_query_failed"] = f"{type(error).__name__}: {error}"
            alive = True
        if alive:
            states[process.pid]["kill_requested"] = True
            try:
                _signal_process_tree(process, True)
            except ProcessLookupError:
                pass
            except OSError as error:
                states[process.pid]["kill_failed"] = f"{type(error).__name__}: {error}"
    for process in targeted:
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            states[process.pid]["wait_failed_after_kill"] = True
    group_deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while time.monotonic() < group_deadline:
        try:
            if not any(_group_alive(process) for process in targeted):
                break
        except OSError:
            break
        for process in targeted:
            _reap_process_group(process)
        time.sleep(0.01)
    for process in targeted:
        _reap_process_group(process)
        try:
            states[process.pid]["group_alive_after_stop"] = _group_alive(process)
        except OSError as error:
            states[process.pid]["liveness_query_failed"] = f"{type(error).__name__}: {error}"
            states[process.pid]["group_alive_after_stop"] = True
    return states


def _stop_windows_processes(
    processes: list[_OwnedProcess | subprocess.Popen[bytes]], reason: str
) -> dict[int, dict[str, object]]:
    states: dict[int, dict[str, object]] = {}
    for candidate in processes:
        state: dict[str, object] = {
            "reason": reason,
            "ownership_mode": "windows_job_object",
            "terminate_requested": False,
            "kill_requested": False,
            "assignment_before_release": False,
            "release_completed": False,
            "job_zero_active_proven": False,
            "job_closed": False,
        }
        states[candidate.pid] = state
        if not isinstance(candidate, _OwnedProcess) or candidate.job is None:
            state["job_ownership_missing"] = True
            state["group_alive_after_stop"] = True
            continue
        state["assignment_before_release"] = candidate.assignment_before_release
        state["release_completed"] = candidate.release_completed
        active: int | None = None
        try:
            active = candidate.job.active_processes()
            state["active_processes_before_stop"] = active
        except OSError as error:
            state["job_query_failed"] = f"{type(error).__name__}: {error}"

        if active is None or active != 0:
            state["terminate_requested"] = True
            state["kill_requested"] = True
            try:
                candidate.job.terminate()
                state["job_terminate_succeeded"] = True
            except OSError as error:
                state["job_terminate_failed"] = f"{type(error).__name__}: {error}"

        deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
        while time.monotonic() < deadline:
            try:
                active = candidate.job.active_processes()
            except OSError as error:
                state["job_query_failed"] = f"{type(error).__name__}: {error}"
                active = None
                break
            if active == 0:
                state["active_processes_after_stop"] = 0
                state["job_zero_active_proven"] = True
                break
            time.sleep(0.01)
        if not state["job_zero_active_proven"] and active is not None:
            state["active_processes_after_stop"] = active
            state["job_zero_wait_timed_out"] = True

        try:
            candidate.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            state["wait_failed_after_kill"] = True
            state["leader_wait_failed"] = True

        try:
            candidate.job.close()
            state["job_closed"] = True
        except OSError as error:
            state["job_close_failed"] = f"{type(error).__name__}: {error}"
        state["group_alive_after_stop"] = not bool(state["job_zero_active_proven"])
    return states


def _stop_processes(
    processes: list[_OwnedProcess | subprocess.Popen[bytes]], reason: str
) -> dict[int, dict[str, object]]:
    if os.name == "nt":
        return _stop_windows_processes(processes, reason)
    return _stop_posix_processes(processes, reason)


def _write_release_token(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, b"release\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_unassigned_windows_wrapper(
    process: subprocess.Popen[bytes], job: _WindowsJob
) -> list[str]:
    errors: list[str] = []
    try:
        process.terminate()
        process.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except BaseException as error:
        errors.append(f"blocked-wrapper-terminate: {type(error).__name__}: {error}")
    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except BaseException as error:
            errors.append(f"blocked-wrapper-wait: {type(error).__name__}: {error}")
    try:
        if job.active_processes() != 0:
            errors.append("unassigned Job Object unexpectedly has active processes")
    except OSError as error:
        errors.append(f"unassigned-job-query: {type(error).__name__}: {error}")
    try:
        job.close()
    except OSError as error:
        errors.append(f"unassigned-job-close: {type(error).__name__}: {error}")
    return errors


def _start_owned_process(
    command: list[str], cwd: Path, environment: dict[str, str],
    stdout_stream: object, stderr_stream: object, release_path: Path
) -> _OwnedProcess:
    if os.name != "nt":
        return _OwnedProcess(subprocess.Popen(
            command, cwd=cwd, env=environment,
            stdout=stdout_stream, stderr=stderr_stream,
            start_new_session=True,
        ))

    release_path.unlink(missing_ok=True)
    job = _WindowsJob()
    wrapper_command = [
        sys.executable, os.fspath(Path(__file__).resolve()),
        _WINDOWS_TRAMPOLINE, os.fspath(release_path), "--", *command,
    ]
    try:
        process = subprocess.Popen(
            wrapper_command, cwd=cwd, env=environment,
            stdout=stdout_stream, stderr=stderr_stream,
        )
    except BaseException as error:
        close_error: OSError | None = None
        try:
            job.close()
        except OSError as failure:
            close_error = failure
        if close_error is not None:
            raise OSError(
                f"blocked-wrapper launch failed: {error}; Job Object close failed: "
                f"{close_error}"
            ) from error
        raise
    owned = _OwnedProcess(process=process, job=job)
    try:
        job.assign(process)
        owned.assignment_before_release = True
    except BaseException as error:
        cleanup_errors = _cleanup_unassigned_windows_wrapper(process, job)
        suffix = f"; cleanup_errors={cleanup_errors}" if cleanup_errors else ""
        raise OSError(f"assignment-before-release failed: {error}{suffix}") from error
    try:
        _write_release_token(release_path)
        owned.release_completed = True
    except BaseException as error:
        state = _stop_windows_processes([owned], "handshake_release_failure")[owned.pid]
        raise OSError(f"pre-execution release failed: {error}; cleanup={state}") from error
    return owned


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

    processes: list[_OwnedProcess] = []
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
                process = _start_owned_process(
                    [os.fspath(executable)], shared, environment,
                    stdout_stream, stderr_stream,
                    log_directory / f"copy-{index}.release",
                )
            processes.append(process)
            log_paths.append((stdout_path, stderr_path))
        while any(_group_alive(process) for process in processes):
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
            # Leader completion is a cleanup trigger, never a whole-tree proof.
            # A surviving group/job is still queried, terminated and proved empty.
            if all(process.poll() is not None for process in processes):
                termination = _stop_processes(
                    processes, "post_exit_descendant_cleanup"
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
        stop_state = termination.get(process.pid, {})
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            if not stop_state:
                stop_state = {
                    "reason": "result_collection",
                    "ownership_mode": (
                        "windows_job_object" if os.name == "nt"
                        else "posix_process_group"
                    ),
                }
                termination[process.pid] = stop_state
            stop_state["leader_wait_failed"] = True
            stop_state["wait_failed_after_kill"] = True
        stdout = paths[0].read_bytes()
        stderr = paths[1].read_bytes()
        exit_code = process.returncode if process.returncode is not None else -1
        results.append(
            {
                "copy": index, "identity": identities[index], "exit": exit_code,
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
                "suite": _parse_suite_output(stdout, stderr),
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


def _windows_trampoline(arguments: list[str]) -> int:
    if os.name != "nt":
        print("Windows trampoline invoked on a non-Windows host", file=sys.stderr)
        return 125
    if len(arguments) < 3 or arguments[1] != "--":
        print("invalid Windows trampoline arguments", file=sys.stderr)
        return 125
    release_path = Path(arguments[0])
    command = arguments[2:]
    deadline = time.monotonic() + 60.0
    while not release_path.is_file():
        if time.monotonic() >= deadline:
            print("Windows trampoline release timed out", file=sys.stderr)
            return 124
        time.sleep(0.005)
    try:
        child = subprocess.Popen(command)
        return child.wait()
    except BaseException as error:
        print(f"Windows trampoline child launch failed: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 125


def _wait_for_path(path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path.name}")
        time.sleep(0.01)


def _windows_self_test_descendant(arguments: list[str]) -> int:
    if os.name != "nt" or len(arguments) != 1:
        return 125
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (AttributeError, OSError, ValueError):
        pass
    marker = Path(arguments[0])
    descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    while True:
        time.sleep(1.0)


def _windows_self_test_subject(arguments: list[str]) -> int:
    if os.name != "nt" or len(arguments) != 2:
        return 125
    mode, marker_name = arguments
    if mode not in ("timeout-resistant-descendant", "leader-exit-descendant"):
        return 125
    if mode == "timeout-resistant-descendant":
        try:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        except (AttributeError, OSError, ValueError):
            pass
    child = subprocess.Popen([
        sys.executable, os.fspath(Path(__file__).resolve()),
        _WINDOWS_SELF_TEST_DESCENDANT, marker_name,
    ])
    _wait_for_path(Path(marker_name), 10.0)
    if mode == "leader-exit-descendant":
        return 0
    while child.poll() is None:
        time.sleep(1.0)
    return child.returncode if child.returncode is not None else 125


def _windows_self_test_state_is_green(state: dict[str, object]) -> bool:
    failures = (
        "wait_failed_after_kill", "leader_wait_failed", "group_alive_after_stop",
        "job_ownership_missing", "job_query_failed", "job_terminate_failed",
        "job_zero_wait_timed_out", "job_close_failed",
    )
    return bool(
        state.get("assignment_before_release") is True
        and state.get("release_completed") is True
        and state.get("terminate_requested") is True
        and state.get("job_terminate_succeeded") is True
        and state.get("job_zero_active_proven") is True
        and state.get("active_processes_after_stop") == 0
        and state.get("job_closed") is True
        and not any(state.get(key) for key in failures)
    )


def _run_windows_self_test_case(root: Path, mode: str) -> dict[str, object]:
    marker = root / f"{mode}.descendant-pid"
    stdout_path = root / f"{mode}.stdout"
    stderr_path = root / f"{mode}.stderr"
    release_path = root / f"{mode}.release"
    command = [
        sys.executable, os.fspath(Path(__file__).resolve()),
        _WINDOWS_SELF_TEST_SUBJECT, mode, os.fspath(marker),
    ]
    owned: _OwnedProcess | None = None
    stopped = False
    try:
        with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
            owned = _start_owned_process(
                command, root, dict(os.environ), stdout_stream, stderr_stream, release_path
            )
        _wait_for_path(marker, 10.0)
        assert owned.job is not None
        active_with_descendant = owned.job.active_processes()
        minimum_active = 2 if mode == "timeout-resistant-descendant" else 1
        if active_with_descendant < minimum_active:
            raise RuntimeError(
                f"Job Object did not observe the descendant tree: active={active_with_descendant}"
            )
        leader_exited_while_descendant_live = False
        if mode == "timeout-resistant-descendant":
            timeout_deadline = time.monotonic() + 0.25
            while time.monotonic() < timeout_deadline:
                time.sleep(0.01)
        else:
            deadline = time.monotonic() + 10.0
            while owned.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            if owned.poll() is None:
                raise TimeoutError("leader-exit trampoline did not exit boundedly")
            active_after_leader_exit = owned.job.active_processes()
            leader_exited_while_descendant_live = active_after_leader_exit > 0
            if not leader_exited_while_descendant_live:
                raise RuntimeError("leader exit was mistaken for whole-job completion")
        state = _stop_processes([owned], mode)[owned.pid]
        stopped = True
        if not _windows_self_test_state_is_green(state):
            raise RuntimeError(f"Windows Job Object cleanup proof failed: {state}")
        return {
            "mode": mode,
            "status": "PASS",
            "active_processes_with_descendant": active_with_descendant,
            "leader_exited_while_descendant_live": leader_exited_while_descendant_live,
            "termination": state,
        }
    finally:
        if owned is not None and not stopped:
            _stop_processes([owned], "self_test_exception_cleanup")


def _windows_process_tree_self_test() -> int:
    if os.name != "nt":
        print(json.dumps({
            "schema": "dspark.windows-process-tree-self-test.v1",
            "status": "PENDING_REMOTE",
            "platform": sys.platform,
            "terminal": "native Windows Job Object runtime is unavailable locally",
        }, sort_keys=True))
        return 0
    record: dict[str, object] = {
        "schema": "dspark.windows-process-tree-self-test.v1",
        "status": "FAIL",
        "platform": sys.platform,
        "cases": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="dspark-windows-job-self-test-") as directory:
            root = Path(directory)
            cases = record["cases"]
            assert isinstance(cases, list)
            cases.append(_run_windows_self_test_case(root, "timeout-resistant-descendant"))
            cases.append(_run_windows_self_test_case(root, "leader-exit-descendant"))
        record.update(
            status="PASS",
            terminal="native Windows Job Object tree ownership and cleanup proved",
        )
    except BaseException as error:
        record["terminal"] = f"{type(error).__name__}: {error}"
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0 if record["status"] == "PASS" else 1


def _validate_args(args: argparse.Namespace) -> None:
    if args.copies < 3 or args.copies > _MAX_COPIES:
        _reject(f"copies must be in [3, {_MAX_COPIES}]")
    if args.timeout_seconds <= 0:
        _reject("timeout-seconds must be positive")
    if not args.expect_all_pass and not args.expect_collision:
        _reject("one expectation is required")
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
    parser.add_argument(
        "--self-test-windows-process-tree", action="store_true",
        help="run the bounded native-Windows Job Object ownership control",
    )
    parser.add_argument("--copies", type=int, default=3)
    parser.add_argument("--working-directory", choices=("shared",), default="shared")
    parser.add_argument("--mutant", choices=("fixed-testio-names",))
    parser.add_argument("--sync-mutant", choices=tuple(_SYNC_DIAGNOSTICS))
    expectation = parser.add_mutually_exclusive_group()
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


def _validate_process_cleanup(record: dict[str, object]) -> None:
    results = record.get("results", [])
    assert isinstance(results, list)
    issues: list[dict[str, object]] = []
    failure_keys = (
        "wait_failed_after_kill", "leader_wait_failed", "group_alive_after_stop",
        "liveness_query_failed", "terminate_failed", "kill_failed",
        "job_ownership_missing", "job_query_failed", "job_terminate_failed",
        "job_zero_wait_timed_out", "job_close_failed",
    )
    for result in results:
        if not isinstance(result, dict):
            issues.append({"copy": None, "issue": "malformed result"})
            continue
        termination = result.get("termination")
        if termination is None:
            if os.name == "nt":
                issues.append({
                    "copy": result.get("copy"),
                    "issue": "missing Windows Job Object cleanup proof",
                })
            continue
        if not isinstance(termination, dict):
            issues.append({"copy": result.get("copy"), "issue": "malformed termination"})
            continue
        present = {
            key: termination[key] for key in failure_keys
            if termination.get(key)
        }
        windows_proof_present = (
            termination.get("ownership_mode") == "windows_job_object"
            or "job_zero_active_proven" in termination
            or "active_processes_after_stop" in termination
            or "job_closed" in termination
        )
        if windows_proof_present:
            requirements = {
                "ownership_mode": termination.get("ownership_mode") == "windows_job_object",
                "assignment_before_release": termination.get("assignment_before_release") is True,
                "release_completed": termination.get("release_completed") is True,
                "job_zero_active_proven": termination.get("job_zero_active_proven") is True,
                "active_processes_after_stop": termination.get("active_processes_after_stop") == 0,
                "job_closed": termination.get("job_closed") is True,
            }
            missing = [key for key, satisfied in requirements.items() if not satisfied]
            if missing:
                present["missing_windows_proofs"] = missing
        if present:
            issues.append({"copy": result.get("copy"), "uncertainty": present})
    if issues:
        _fail(
            "process-tree cleanup was not proven: " + repr(issues),
            "process_cleanup_failure", "process-cleanup-validation",
        )


def _validate_complete_suite(result: dict[str, object], classification: str,
                             phase: str) -> dict[str, object]:
    suite = result.get("suite")
    if not isinstance(suite, dict) or suite.get("complete") is not True:
        _fail(
            f"copy {result.get('copy')} did not complete the exact "
            f"{_EXPECTED_SUITE_TESTS}-test suite: {suite!r}",
            classification, phase,
        )
    summary = suite.get("summary")
    if not isinstance(summary, dict):
        _fail(
            f"copy {result.get('copy')} lacked one exact suite summary",
            classification, phase,
        )
    return suite


def _evaluate(args: argparse.Namespace, record: dict[str, object],
              outcome: dict[str, object], synchronization_root: Path | None) -> None:
    _validate_process_cleanup(record)
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
        for result in results:
            assert isinstance(result, dict)
            suite = _validate_complete_suite(
                result, "candidate_oracle_failure", "candidate-validation"
            )
            summary = suite["summary"]
            assert isinstance(summary, dict)
            if (
                summary.get("passed") != _EXPECTED_SUITE_TESTS
                or summary.get("failed") != 0
                or suite.get("failed_tests") != []
                or suite.get("exception_tests") != []
            ):
                _fail(
                    f"copy {result.get('copy')} was not an exact green suite: {suite!r}",
                    "candidate_oracle_failure", "candidate-validation",
                )
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
    if len(by_identity) != args.copies or set(by_identity) != set(range(args.copies)):
        _fail(f"result identities did not exactly cover every copy: {sorted(by_identity)}",
              "collision_oracle_failure", "collision-validation")
    winner = by_identity[winners[0]]
    winner_suite = _validate_complete_suite(
        winner, "collision_oracle_failure", "collision-validation"
    )
    winner_summary = winner_suite["summary"]
    assert isinstance(winner_summary, dict)
    if (
        int(winner["exit"]) != 0
        or winner_summary.get("passed") != _EXPECTED_SUITE_TESTS
        or winner_summary.get("failed") != 0
        or winner_suite.get("failed_tests") != []
        or winner_suite.get("exception_tests") != []
    ):
        _fail(
            f"atomic-claim winner was not an exact green suite: {winner!r}",
            "collision_oracle_failure", "collision-validation",
        )
    for identity in losers:
        result = by_identity[identity]
        suite = _validate_complete_suite(
            result, "collision_oracle_failure", "collision-validation"
        )
        summary = suite["summary"]
        assert isinstance(summary, dict)
        if (
            int(result["exit"]) == 0
            or summary.get("passed") != _EXPECTED_SUITE_TESTS - 1
            or summary.get("failed") != 1
            or suite.get("failed_tests") != [_ATOMIC_CLAIM_TEST]
            or suite.get("exception_tests") != []
            or not result["reported_atomic_claim_failure"]
        ):
            _fail(f"losing identity {identity} lacked only the exact atomic-claim failure",
                  "collision_oracle_failure", "collision-validation")
    if winner["reported_atomic_claim_failure"]:
        _fail("atomic-claim winner reported the loser failure",
              "collision_oracle_failure", "collision-validation")
    record.update(
        status="PASS", classification="expected_fixed_name_collision",
        terminal=("fixed-name mutant produced one synchronized atomic winner and "
                  f"{len(losers)} losers"), failure=None,
    )


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == _WINDOWS_TRAMPOLINE:
        return _windows_trampoline(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == _WINDOWS_SELF_TEST_SUBJECT:
        return _windows_self_test_subject(sys.argv[2:])
    if len(sys.argv) >= 2 and sys.argv[1] == _WINDOWS_SELF_TEST_DESCENDANT:
        return _windows_self_test_descendant(sys.argv[2:])
    args = _parse_args()
    if args.self_test_windows_process_tree:
        if len(sys.argv) != 2:
            print(json.dumps({
                "schema": "dspark.windows-process-tree-self-test.v1",
                "status": "FAIL",
                "classification": "configuration_failure",
                "terminal": "--self-test-windows-process-tree accepts no other options",
            }, sort_keys=True), file=sys.stderr)
            return 1
        return _windows_process_tree_self_test()
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
