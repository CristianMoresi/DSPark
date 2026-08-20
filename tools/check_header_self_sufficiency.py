#!/usr/bin/env python3
"""Compile DSPark's installed headers and umbrella profiles in isolation.

The inventory is derived from the install directories in CMakeLists.txt and
the includes in DSPark.h.  No second list of public headers is maintained by
this tool.  The release counters are scalar assertions over that derived
inventory: 100 umbrella-facing headers and 101 installed library headers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile


EXPECTED_UMBRELLA_HEADERS = 100
EXPECTED_INSTALLED_HEADERS = 101
SHUFFLE_SEED = 0xD5A170
WARNING_FLAGS = ("-Wall", "-Wextra", "-Wpedantic", "-Werror")


class GateFailure(RuntimeError):
    pass


def _install_directories(repo: Path) -> tuple[str, ...]:
    cmake = (repo / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(
        r"install\s*\(\s*DIRECTORY\s+(.*?)\s+DESTINATION\b",
        cmake,
        flags=re.DOTALL,
    )
    if match is None:
        raise GateFailure("cannot derive installed header directories from CMakeLists.txt")
    without_comments = re.sub(r"#[^\n]*", "", match.group(1))
    directories = tuple(shlex.split(without_comments))
    if not directories:
        raise GateFailure("the CMake install directory set is empty")
    for directory in directories:
        if not (repo / directory).is_dir():
            raise GateFailure(f"installed header directory does not exist: {directory}")
    return directories


def derive_inventory(repo: Path) -> tuple[list[str], list[str]]:
    directories = _install_directories(repo)
    installed = sorted(
        path.relative_to(repo).as_posix()
        for directory in directories
        for path in (repo / directory).rglob("*.h")
    )
    top_level = sorted(
        path.relative_to(repo).as_posix()
        for directory in directories
        for path in (repo / directory).glob("*.h")
    )

    umbrella_text = (repo / "DSPark.h").read_text(encoding="utf-8")
    include_pattern = re.compile(r'^\s*#\s*include\s+"([^"]+\.h)"', re.MULTILINE)
    installed_set = set(installed)
    umbrella = sorted(
        include
        for include in include_pattern.findall(umbrella_text)
        if include in installed_set
    )

    if len(installed) != len(set(installed)):
        raise GateFailure("derived installed inventory contains a duplicate")
    if len(umbrella) != len(set(umbrella)):
        raise GateFailure("DSPark.h contains a duplicate public include")
    if umbrella != top_level:
        missing = sorted(set(top_level) - set(umbrella))
        extra = sorted(set(umbrella) - set(top_level))
        raise GateFailure(
            "umbrella inventory mismatch: missing=%s extra=%s" % (missing, extra)
        )
    if len(umbrella) != EXPECTED_UMBRELLA_HEADERS:
        raise GateFailure(
            "umbrella-facing header count is %d, expected %d"
            % (len(umbrella), EXPECTED_UMBRELLA_HEADERS)
        )
    if len(installed) != EXPECTED_INSTALLED_HEADERS:
        raise GateFailure(
            "installed library header count is %d, expected %d"
            % (len(installed), EXPECTED_INSTALLED_HEADERS)
        )
    return umbrella, installed


def _compiler_command(compiler: str, repo: Path, source: Path,
                      extra_flags: tuple[str, ...] = ()) -> list[str]:
    resolved = shutil.which(compiler) if not os.path.isabs(compiler) else compiler
    if not resolved or not Path(resolved).exists():
        raise GateFailure(f"compiler is unavailable: {compiler}")
    return [
        resolved,
        "-std=c++20",
        *WARNING_FLAGS,
        *extra_flags,
        "-I",
        os.fspath(repo),
        "-fsyntax-only",
        os.fspath(source),
    ]


def _compile(compiler: str, repo: Path, scratch: Path, label: str, source: str,
             extra_flags: tuple[str, ...] = (), expect_success: bool = True,
             expected_diagnostic: str | None = None) -> subprocess.CompletedProcess[str]:
    source_path = scratch / f"{label}.cpp"
    source_path.write_text(source, encoding="utf-8", newline="\n")
    command = _compiler_command(compiler, repo, source_path, extra_flags)
    completed = subprocess.run(
        command,
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    succeeded = completed.returncode == 0
    if succeeded != expect_success:
        expectation = "succeed" if expect_success else "fail"
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise GateFailure(
            "%s was expected to %s\ncommand: %s\n%s"
            % (label, expectation, shlex.join(command), detail)
        )
    if expected_diagnostic is not None:
        combined = completed.stdout + completed.stderr
        if expected_diagnostic not in combined:
            raise GateFailure(
                f"{label} failed without the expected {expected_diagnostic!r} diagnostic"
            )
    return completed


def _include_source(headers: list[str]) -> str:
    return "".join(f'#include "{header}"\n' for header in headers) + "int main() {}\n"


def _umbrella_source(no_file_io: bool = False) -> str:
    prefix = "#define DSPARK_NO_FILE_IO\n" if no_file_io else ""
    return prefix + r'''#include "DSPark.h"
int main()
{
    dspark::AudioSpec spec { 48000.0, 64, 2 };
    dspark::AudioBuffer<float> floats;
    dspark::AudioBuffer<double> doubles;
    floats.resize(2, 64);
    doubles.resize(2, 64);
    dspark::Compressor<float> floatProcessor;
    dspark::Compressor<double> doubleProcessor;
    floatProcessor.prepare(spec);
    doubleProcessor.prepare(spec);
    floatProcessor.processBlock(floats.toView());
    doubleProcessor.processBlock(doubles.toView());
    return 0;
}
'''


def _no_file_io_source(enabled: bool) -> str:
    prefix = "#define DSPARK_NO_FILE_IO\n" if enabled else ""
    return prefix + r'''#include "DSPark.h"
int main()
{
    dspark::AudioFileInfo* info = nullptr;
    dspark::AudioFile* audio = nullptr;
    dspark::WavFile* wav = nullptr;
    dspark::Mp3File* mp3 = nullptr;
    dspark::MidiFile* midi = nullptr;
    dspark::FlacFile* flac = nullptr;
    return info == nullptr && audio == nullptr && wav == nullptr
        && mp3 == nullptr && midi == nullptr && flac == nullptr ? 0 : 1;
}
'''


def run_full(repo: Path, compiler: str) -> None:
    umbrella, installed = derive_inventory(repo)
    with tempfile.TemporaryDirectory(prefix="dspark-header-audit-") as directory:
        scratch = Path(directory)
        for index, header in enumerate(installed):
            _compile(
                compiler,
                repo,
                scratch,
                f"header_{index:03d}",
                f'#include "{header}"\nint main() {{}}\n',
            )

        orders: list[tuple[str, list[str]]] = [
            ("forward", list(installed)),
            ("reverse", list(reversed(installed))),
        ]
        shuffled = list(installed)
        random.Random(SHUFFLE_SEED).shuffle(shuffled)
        orders.append(("seeded_shuffle", shuffled))
        for label, headers in orders:
            _compile(compiler, repo, scratch, label, _include_source(headers))

        _compile(compiler, repo, scratch, "umbrella_float_double", _umbrella_source())
        _compile(
            compiler,
            repo,
            scratch,
            "umbrella_no_file_io",
            _umbrella_source(no_file_io=True),
        )
        _compile(
            compiler,
            repo,
            scratch,
            "umbrella_embedded",
            _umbrella_source(no_file_io=True),
            extra_flags=("-fno-exceptions", "-fno-rtti"),
        )
    print(
        "header self-sufficiency: PASS compiler=%s umbrella=%d installed=%d "
        "standalone=%d include_orders=3 profiles=3"
        % (compiler, len(umbrella), len(installed), len(installed))
    )


def run_embedded(repo: Path, compiler: str) -> None:
    derive_inventory(repo)
    with tempfile.TemporaryDirectory(prefix="dspark-header-embedded-") as directory:
        _compile(
            compiler,
            repo,
            Path(directory),
            "umbrella_embedded",
            _umbrella_source(no_file_io=True),
            extra_flags=("-fno-exceptions", "-fno-rtti"),
        )
    print(f"header self-sufficiency: PASS compiler={compiler} profile=embedded")


def run_no_file_io_exactness(repo: Path, compiler: str) -> None:
    derive_inventory(repo)
    with tempfile.TemporaryDirectory(prefix="dspark-header-no-io-") as directory:
        scratch = Path(directory)
        _compile(
            compiler,
            repo,
            scratch,
            "io_names_present",
            _no_file_io_source(enabled=False),
        )
        _compile(
            compiler,
            repo,
            scratch,
            "io_names_absent",
            _no_file_io_source(enabled=True),
            expect_success=False,
        )
        _compile(
            compiler,
            repo,
            scratch,
            "non_io_profile_present",
            _umbrella_source(no_file_io=True),
        )
    print(f"header self-sufficiency: PASS compiler={compiler} profile=no-file-io-exactness")


def run_negative_self_test(repo: Path) -> None:
    compilers = [compiler for compiler in ("g++", "clang++") if shutil.which(compiler)]
    if not compilers:
        raise GateFailure("negative control requires g++ or clang++")
    for compiler in compilers:
        with tempfile.TemporaryDirectory(prefix="dspark-header-negative-") as directory:
            scratch = Path(directory)
            header = scratch / "synthetic_missing_include.h"
            header.write_text(
                "#pragma once\nstruct MissingInclude { std::vector<int> values; };\n",
                encoding="utf-8",
                newline="\n",
            )
            source = '#include "synthetic_missing_include.h"\nint main() {}\n'
            source_path = scratch / "missing_include.cpp"
            source_path.write_text(source, encoding="utf-8", newline="\n")
            command = [
                shutil.which(compiler) or compiler,
                "-std=c++20",
                *WARNING_FLAGS,
                "-I",
                os.fspath(scratch),
                "-fsyntax-only",
                os.fspath(source_path),
            ]
            completed = subprocess.run(
                command,
                cwd=repo,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode == 0 or "vector" not in completed.stderr:
                raise GateFailure(
                    f"synthetic missing-include control did not fail correctly under {compiler}"
                )
    print(
        "header self-sufficiency: PASS negative-control=missing-vector "
        f"compilers={','.join(compilers)}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--std", choices=("c++20", "20"), default="c++20")
    parser.add_argument("--warnings-as-errors", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("full", "embedded", "no-file-io-exactness"),
        default="full",
    )
    parser.add_argument("--self-test-negative-control", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.self_test_negative_control:
        run_negative_self_test(repo)
        return 0
    if args.inventory_only:
        umbrella, installed = derive_inventory(repo)
        detail = sorted(set(installed) - set(umbrella))
        print(
            "header inventory: PASS umbrella=%d installed=%d detail=%s"
            % (len(umbrella), len(installed), ",".join(detail))
        )
        return 0
    if args.profile == "embedded":
        run_embedded(repo, args.compiler)
    elif args.profile == "no-file-io-exactness":
        run_no_file_io_exactness(repo, args.compiler)
    else:
        run_full(repo, args.compiler)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as error:
        print(f"header self-sufficiency: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from None
