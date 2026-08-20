#!/usr/bin/env python3
"""Prove that the float-cast-overflow sanitizer is present and load-bearing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


WARNING_FLAGS = ("-Wall", "-Wextra", "-Wpedantic", "-Werror")
SANITIZER_FLAGS = (
    "-fsanitize=float-cast-overflow",
    "-fno-sanitize-recover=float-cast-overflow",
)
DIAGNOSTIC_MARKERS = (
    "outside the range of representable values",
    "cannot be represented in type",
)


class HarnessFailure(RuntimeError):
    pass


def _compile(compiler: str, source: Path, output: Path,
             instrument: bool) -> subprocess.CompletedProcess[str]:
    resolved = shutil.which(compiler) if not os.path.isabs(compiler) else compiler
    if not resolved or not Path(resolved).exists():
        raise HarnessFailure(f"compiler is unavailable: {compiler}")
    command = [
        resolved,
        "-std=c++20",
        "-O1",
        "-g",
        *WARNING_FLAGS,
        *(SANITIZER_FLAGS if instrument else ()),
        os.fspath(source),
        "-o",
        os.fspath(output),
    ]
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise HarnessFailure(
            "negative-control compilation failed\ncommand: %s\n%s"
            % (shlex.join(command), detail)
        )
    return completed


def _run(subject: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [os.fspath(subject)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _diagnostic_is_float_cast(stderr: str) -> bool:
    lowered = stderr.lower()
    return "runtime error" in lowered and any(
        marker in lowered for marker in DIAGNOSTIC_MARKERS
    )


def verify_subject(subject: Path) -> None:
    completed = _run(subject)
    if completed.returncode == 0:
        raise HarnessFailure("instrumented child exited zero; sanitizer did not trigger")
    if not _diagnostic_is_float_cast(completed.stderr):
        detail = completed.stderr.strip() or "no stderr"
        raise HarnessFailure(
            "child failed without the expected float-cast-overflow class: " + detail
        )
    print(
        "float-cast-overflow sanitizer: PASS child_exit=%d diagnostic=runtime-error-outside-range"
        % completed.returncode
    )


def compile_and_verify(repo: Path, compiler: str) -> None:
    source = repo / "tests" / "float_cast_overflow_subject.cpp"
    with tempfile.TemporaryDirectory(prefix="dspark-float-cast-") as directory:
        subject = Path(directory) / "float_cast_overflow_subject"
        _compile(compiler, source, subject, instrument=True)
        verify_subject(subject)


def flag_removal_control(repo: Path, compiler: str) -> None:
    source = repo / "tests" / "float_cast_overflow_subject.cpp"
    with tempfile.TemporaryDirectory(prefix="dspark-float-cast-red-") as directory:
        subject = Path(directory) / "float_cast_overflow_subject"
        compile_result = _compile(compiler, source, subject, instrument=False)
        child = _run(subject)
        if child.returncode != 0:
            raise HarnessFailure(
                "flag-removal child failed independently of the sanitizer flag"
            )
        try:
            verify_subject(subject)
        except HarnessFailure as error:
            if str(error) != "instrumented child exited zero; sanitizer did not trigger":
                raise HarnessFailure(
                    "flag-removal meta-test failed for the wrong reason: " + str(error)
                ) from error
            nested_exit = 1
        else:
            raise HarnessFailure("flag-removal meta-test unexpectedly stayed green")
        print(
            "float-cast-overflow sanitizer: PASS flag-removal-control "
            "compile_exit=%d child_exit=%d nested_gate_exit=%d"
            % (compile_result.returncode, child.returncode, nested_exit)
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--subject", type=Path)
    parser.add_argument("--self-test-flag-removal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.self_test_flag_removal:
        flag_removal_control(repo, args.compiler)
    elif args.subject is not None:
        verify_subject(args.subject.resolve(strict=True))
    else:
        compile_and_verify(repo, args.compiler)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessFailure as error:
        print(f"float-cast-overflow sanitizer: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from None
