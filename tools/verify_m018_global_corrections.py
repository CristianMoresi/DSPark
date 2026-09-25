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

from check_public_text import canonical_text_bytes


ROOT = Path(__file__).resolve().parents[1]
INSTALLED_DIRECTORIES = ("Core", "Effects", "Analysis", "IO", "Music")
EXPECTED_INSTALLED_HEADERS = 102
EXPECTED_ORDINARY_TESTS = 902
PRODUCT_P7_COMMIT = "5a47d959de4b3d48445a8850960f74377999faf9"
PRODUCT_P7_PARENT = "ff56759f0d12e9bfad20a77b5b8c80b642ffe5f2"
PACKAGE_SOURCE_URL_PREFIX = (
    "https://codeload.github.com/CristianMoresi/DSPark/tar.gz/"
)
PACKAGE_SOURCE_SHA256 = (
    "3b6d44a863ab97749f1b4131c255689c2fbdf00b891bdaa9329aadc1a1e00ce2"
)
PACKAGE_SOURCE_FILENAME = "dspark-1.7.0.tar.gz"
PACKAGE_SOURCE_SHA512 = (
    "b7382dc3e0247fbf93e0a5555750deda74798809d4c3d8154a08bb517faf9cf0"
    "46cabdd04d88fc6e7895a7a6666e7017eeed719ea2582edca5363bba8ef9383f"
)
EXPECTED_PACKAGE_HASHES = {
    "packaging/conan/conanfile.py": "fe14207493ecdecc7fcefcc4eccda4cdd09f88c9755db576133bfff3bdfa74f0",
    "packaging/vcpkg/portfile.cmake": "c2b5f8c0c30518e220c1f85abed9cbe7bdcaa26608f153821be99f5785d10815",
    "packaging/vcpkg/vcpkg.json": "08fd9782597218b0f6ddbfc4720ebe83a4a710cff6758f76752833f491e99bdd",
}
EXPECTED_SEMANTIC_HASHES = {
    "packaging/conan/conanfile.py": "4eabfe4a38cd593eb73789313dea018e2a9ba022339ec566125c2aeb178a832d",
    "packaging/vcpkg/portfile.cmake": "9dfe0b00f07741b1308546e76f0985d277952ed3f3aaa6864d31a2ef9e2a9048",
    "packaging/vcpkg/vcpkg.json": "987800079ee1d7556da43c2ff8cea3fb08553eb3e1bd3e7142e8fb342255093e",
}
PACKAGE_FIELD_MUTANT_IDS = (
    "MUT-R-CONAN-COMMENT",
    "MUT-R-CONAN-NAME",
    "MUT-R-CONAN-VERSION",
    "MUT-R-CONAN-PACKAGE-TYPE",
    "MUT-R-CONAN-HOMEPAGE",
    "MUT-R-CONAN-SOURCE-URL",
    "MUT-R-CONAN-FILENAME",
    "MUT-R-CONAN-REF",
    "MUT-R-CONAN-SHA256",
    "MUT-R-CONAN-STRIP-ROOT",
    "MUT-R-CONAN-MODULES",
    "MUT-R-CONAN-INCLUDE-DESTINATION",
    "MUT-R-CONAN-CMAKE-FILE",
    "MUT-R-CONAN-CMAKE-TARGET",
    "MUT-R-CONAN-LIB-BIN-DIRS",
    "MUT-R-CONAN-PACKAGE-ID",
    "MUT-R-VCPKG-COMMENT",
    "MUT-R-VCPKG-REPO",
    "MUT-R-VCPKG-REF",
    "MUT-R-VCPKG-SHA512",
    "MUT-R-VCPKG-HEAD-REF",
    "MUT-R-VCPKG-CONFIGURE",
    "MUT-R-VCPKG-FIXUP",
    "MUT-R-VCPKG-CLEANUP",
    "MUT-R-VCPKG-COPYRIGHT",
    "MUT-R-MANIFEST-BYTES",
    "MUT-R-MANIFEST-NAME",
    "MUT-R-MANIFEST-VERSION",
    "MUT-R-MANIFEST-HOMEPAGE-LICENSE",
    "MUT-R-MANIFEST-DEPENDENCIES",
    "MUT-R-MANIFEST-DUPLICATE-OR-EXTRA",
)
PACKAGE_FIELD_MUTANT_INVENTORY_SHA256 = (
    "1434ee1784aa0dd7e16e2d04c4dbfd95d5e14af0278c68bbf65c50c2fe430227"
)
PACKAGE_FIELD_MUTANT_SUBCASE_COUNTS = (
    1, 1, 1, 1, 1, 1, 12, 5, 1, 2,
    3, 1, 1, 1, 2, 1, 1, 2, 5, 1,
    1, 3, 2, 2, 1, 2, 1, 1, 2, 5,
    2,
)
EXPECTED_PACKAGE_FIELD_MUTANT_SUBCASES = 66


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
    tree = ast.parse(data.decode("ascii"))
    options = {"annotate_fields": True, "include_attributes": False}
    if sys.version_info >= (3, 13):
        options["show_empty"] = True
    return ast.dump(tree, **options).encode("ascii")


def cmake_noncomment_semantics(data: bytes) -> bytes:
    canonical: list[str] = []
    for source_line in data.decode("ascii").splitlines():
        line = source_line.split("#", 1)[0]
        line = " ".join(line.split())
        if line:
            canonical.append(line)
    return "\n".join(canonical).encode("ascii")


class DuplicateJsonKey(ValueError):
    """Raised when strict package-manifest parsing sees a duplicate key."""


def reject_json_duplicates(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(key)
        result[key] = value
    return result


def strict_json_value(data: bytes) -> object:
    return json.loads(
        data.decode("ascii"), object_pairs_hook=reject_json_duplicates)


def manifest_semantics(data: bytes) -> bytes:
    return json.dumps(
        strict_json_value(data), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def unique_errors(errors: list[str]) -> list[str]:
    return list(dict.fromkeys(errors))



def package_r_executable_bootstrap_errors(source: str) -> list[str]:
    """Audit the producer binding before any protected helper is invoked."""
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            "PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_PRODUCER_POSITION",
        ]

    def top(name: str) -> ast.FunctionDef | None:
        matches = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        return matches[0] if len(matches) == 1 else None

    def called(call: ast.Call, name: str) -> bool:
        return isinstance(call.func, ast.Name) and call.func.id == name

    def extend_argument(statement: ast.stmt) -> ast.AST | None:
        if not isinstance(statement, ast.Expr) \
                or not isinstance(statement.value, ast.Call):
            return None
        call = statement.value
        if call.keywords or len(call.args) != 1 \
                or not isinstance(call.func, ast.Attribute) \
                or call.func.attr != "extend" \
                or not isinstance(call.func.value, ast.Name) \
                or call.func.value.id != "errors":
            return None
        return call.args[0]

    def exact(statement: ast.stmt, callee: str,
              arguments: tuple[str, ...]) -> bool:
        value = extend_argument(statement)
        return isinstance(value, ast.Call) and called(value, callee) \
            and not value.keywords and len(value.args) == len(arguments) \
            and all(isinstance(argument, ast.Name)
                    and argument.id == expected
                    for argument, expected in zip(value.args, arguments))

    checker = top("package_oracle_binding_errors")
    current = top("current_package_oracle_binding_errors")
    if checker is None or current is None:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_HELPER_MISSING")
    else:
        checker_returns = [
            node for node in checker.body
            if isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and called(node.value, "unique_errors")
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "errors"
            and not node.value.keywords
        ]
        checker_parses = [
            node for node in ast.walk(checker)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parse"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ast"
        ]
        current_calls = [
            node for node in ast.walk(current)
            if isinstance(node, ast.Call)
            and called(node, "package_oracle_binding_errors")
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "source"
            and not node.keywords
        ]
        source_reads = [
            node for node in ast.walk(current)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_text"
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "Path"
            and len(node.func.value.args) == 1
            and isinstance(node.func.value.args[0], ast.Name)
            and node.func.value.args[0].id == "__file__"
            and len(node.keywords) == 1
            and node.keywords[0].arg == "encoding"
            and isinstance(node.keywords[0].value, ast.Constant)
            and node.keywords[0].value.value == "ascii"
        ]
        if [item.arg for item in checker.args.args] != ["source"] \
                or current.args.args or len(checker_returns) != 1 \
                or len(checker_parses) != 1 or len(current_calls) != 1 \
                or len(source_reads) != 1:
            errors.append("PACKAGE_R_ORACLE_PRODUCER_HELPER_NEUTRALIZED")

    main_function = top("main")
    if main_function is None:
        return list(dict.fromkeys(errors + [
            "PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_PRODUCER_POSITION",
        ]))
    package_indexes = [
        index for index, statement in enumerate(main_function.body)
        if exact(statement, "package_errors", ("root",))
    ]
    invariant_indexes = [
        index for index, statement in enumerate(main_function.body)
        if exact(statement, "current_package_oracle_binding_errors", ())
    ]
    package_calls = sum(
        isinstance(node, ast.Call) and called(node, "package_errors")
        for node in ast.walk(main_function)
    )
    invariant_calls = sum(
        isinstance(node, ast.Call)
        and called(node, "current_package_oracle_binding_errors")
        for node in ast.walk(main_function)
    )
    assignments: list[int] = []
    empty: list[int] = []
    for index, statement in enumerate(main_function.body):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.AnnAssign):
            target, value = statement.target, statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        if isinstance(target, ast.Name) and target.id == "errors":
            assignments.append(index)
            if isinstance(value, ast.List) and not value.elts:
                empty.append(index)
    boundary = next((
        index for index, statement in enumerate(main_function.body)
        if (isinstance(statement, ast.For)
            and isinstance(statement.iter, ast.Name)
            and statement.iter.id == "errors")
        or (isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "errors")
        or isinstance(statement, ast.Return)
    ), None)
    if package_calls == 0:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_MISSING")
    if package_calls > 1 or len(package_indexes) > 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_DUPLICATE")
    if package_calls != 1 or len(package_indexes) != 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT")
    if invariant_calls != 1 or len(invariant_indexes) != 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_INVARIANT_MISSING")
    if len(assignments) != 1 or len(empty) != 1 \
            or len(invariant_indexes) != 1 or len(package_indexes) != 1 \
            or boundary is None \
            or not empty[0] < invariant_indexes[0] < package_indexes[0] < boundary:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_POSITION")
    return list(dict.fromkeys(errors))


def _m018_top_level_function(
    tree: ast.Module, name: str,
) -> ast.FunctionDef | None:
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return matches[0] if len(matches) == 1 else None


def _m018_called_name(call: ast.Call, name: str) -> bool:
    return isinstance(call.func, ast.Name) and call.func.id == name


def _m018_errors_extend_argument(statement: ast.stmt) -> ast.AST | None:
    if not isinstance(statement, ast.Expr) \
            or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if call.keywords or len(call.args) != 1:
        return None
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "extend":
        return None
    if not isinstance(call.func.value, ast.Name) \
            or call.func.value.id != "errors":
        return None
    return call.args[0]


def _m018_exact_package_aggregation(statement: ast.stmt) -> bool:
    value = _m018_errors_extend_argument(statement)
    return (
        isinstance(value, ast.Call)
        and _m018_called_name(value, "package_errors")
        and not value.keywords
        and len(value.args) == 1
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "root"
    )


def _m018_exact_invariant_aggregation(statement: ast.stmt) -> bool:
    value = _m018_errors_extend_argument(statement)
    return (
        isinstance(value, ast.Call)
        and _m018_called_name(
            value, "current_package_oracle_binding_errors")
        and not value.args
        and not value.keywords
    )


def _m018_direct_indexes(
    function: ast.FunctionDef, predicate: object,
) -> list[int]:
    return [
        index for index, statement in enumerate(function.body)
        if callable(predicate) and predicate(statement)
    ]


def _m018_errors_initialization_indexes(
    function: ast.FunctionDef,
) -> tuple[list[int], list[int]]:
    assignments: list[int] = []
    empty_lists: list[int] = []
    for index, statement in enumerate(function.body):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        if isinstance(target, ast.Name) and target.id == "errors":
            assignments.append(index)
            if isinstance(value, ast.List) and not value.elts:
                empty_lists.append(index)
    return assignments, empty_lists


def _m018_terminal_boundary(function: ast.FunctionDef) -> int | None:
    for index, statement in enumerate(function.body):
        if isinstance(statement, ast.For) \
                and isinstance(statement.iter, ast.Name) \
                and statement.iter.id == "errors":
            return index
        if isinstance(statement, ast.If) \
                and isinstance(statement.test, ast.Name) \
                and statement.test.id == "errors":
            return index
        if isinstance(statement, ast.Return):
            return index
    return None


def _m018_named_call_count(function: ast.FunctionDef, name: str) -> int:
    return sum(
        isinstance(node, ast.Call) and _m018_called_name(node, name)
        for node in ast.walk(function)
    )


def _m018_helper_surface_errors(tree: ast.Module) -> list[str]:
    errors: list[str] = []
    checker = _m018_top_level_function(tree, "package_oracle_binding_errors")
    current = _m018_top_level_function(
        tree, "current_package_oracle_binding_errors")
    if checker is None or current is None:
        return ["PACKAGE_R_ORACLE_PRODUCER_HELPER_MISSING"]
    checker_return = [
        node for node in checker.body
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Call)
        and _m018_called_name(node.value, "unique_errors")
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "errors"
        and not node.value.keywords
    ]
    checker_parse_calls = [
        node for node in ast.walk(checker)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ast"
    ]
    if len(checker_return) != 1 or len(checker_parse_calls) != 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_HELPER_NEUTRALIZED")
    current_calls = _m018_named_call_count(
        current, "package_oracle_binding_errors")
    source_reads = [
        node for node in ast.walk(current)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "Path"
        and len(node.func.value.args) == 1
        and isinstance(node.func.value.args[0], ast.Name)
        and node.func.value.args[0].id == "__file__"
        and len(node.keywords) == 1
        and node.keywords[0].arg == "encoding"
        and isinstance(node.keywords[0].value, ast.Constant)
        and node.keywords[0].value.value == "ascii"
    ]
    if current_calls != 1 or len(source_reads) != 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_HELPER_NEUTRALIZED")
    return errors



def _m018_bootstrap_surface_errors(tree: ast.Module) -> list[str]:
    errors: list[str] = []
    bootstrap = _m018_top_level_function(
        tree, "package_r_executable_bootstrap_errors")
    if bootstrap is None:
        return ["PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_HELPER_MISSING"]
    constants = {
        node.value for node in ast.walk(bootstrap)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {
        "PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT",
        "PACKAGE_R_ORACLE_PRODUCER_POSITION",
        "PACKAGE_R_ORACLE_PRODUCER_HELPER_MISSING",
        "PACKAGE_R_ORACLE_PRODUCER_HELPER_NEUTRALIZED",
        "PACKAGE_R_ORACLE_PRODUCER_INVARIANT_MISSING",
    }
    parse_calls = [
        node for node in ast.walk(bootstrap)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parse"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ast"
    ]
    if [item.arg for item in bootstrap.args.args] != ["source"] \
            or len(parse_calls) != 1 or not required.issubset(constants) \
            or not any(isinstance(node, ast.Return)
                       and isinstance(node.value, ast.Call)
                       and isinstance(node.value.func, ast.Name)
                       and node.value.func.id == "list"
                       for node in ast.walk(bootstrap)):
        errors.append("PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_NEUTRALIZED")

    main_function = _m018_top_level_function(tree, "main")
    if main_function is None:
        return errors + ["PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_BOUNDARY"]
    get_calls = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "globals"
        and node.func.attr == "get"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "package_r_executable_bootstrap_errors"
    ]
    bootstrap_calls = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "package_r_bootstrap"
    ]
    source_reads = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_text"
        and any(isinstance(item, ast.Name) and item.id == "__file__"
                for item in ast.walk(node.func.value))
    ]
    caught: list[str] = []
    try_nodes = [
        node for node in ast.walk(main_function)
        if isinstance(node, ast.Try)
        and any(isinstance(item, ast.Call)
                and isinstance(item.func, ast.Name)
                and item.func.id == "package_r_bootstrap"
                for item in ast.walk(node))
    ]
    for node in try_nodes:
        for handler in node.handlers:
            if isinstance(handler.type, ast.Tuple):
                caught.extend(
                    item.id for item in handler.type.elts
                    if isinstance(item, ast.Name))
    expected = [
        "AttributeError", "NameError", "OSError", "SyntaxError",
        "TypeError", "UnicodeError", "ValueError",
    ]
    if len(get_calls) != 1 or len(bootstrap_calls) != 1 \
            or len(try_nodes) != 1 \
            or len(source_reads) < 1 or sorted(caught) != sorted(expected):
        errors.append("PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_BOUNDARY")
    return errors


def package_oracle_binding_errors(source: str) -> list[str]:
    """Validate the normal Package-R oracle binding from Python AST only."""
    errors: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            "PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_PRODUCER_POSITION",
        ]
    errors.extend(_m018_bootstrap_surface_errors(tree))
    main_function = _m018_top_level_function(tree, "main")
    if main_function is None:
        return unique_errors([
            "PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT",
            "PACKAGE_R_ORACLE_PRODUCER_POSITION",
            *_m018_helper_surface_errors(tree),
        ])
    package_indexes = _m018_direct_indexes(
        main_function, _m018_exact_package_aggregation)
    invariant_indexes = _m018_direct_indexes(
        main_function, _m018_exact_invariant_aggregation)
    package_calls = _m018_named_call_count(main_function, "package_errors")
    invariant_calls = _m018_named_call_count(
        main_function, "current_package_oracle_binding_errors")
    assignments, empty_initializations = \
        _m018_errors_initialization_indexes(main_function)
    boundary = _m018_terminal_boundary(main_function)

    if package_calls == 0:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_MISSING")
    if package_calls > 1 or len(package_indexes) > 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_DUPLICATE")
    if package_calls != 1 or len(package_indexes) != 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_CALL_DRIFT")
    if invariant_calls != 1 or len(invariant_indexes) != 1:
        errors.append("PACKAGE_R_ORACLE_PRODUCER_INVARIANT_MISSING")
    if (
        len(assignments) != 1
        or len(empty_initializations) != 1
        or len(package_indexes) != 1
        or len(invariant_indexes) != 1
        or boundary is None
        or not (
            empty_initializations[0]
            < invariant_indexes[0]
            < package_indexes[0]
            < boundary
        )
    ):
        errors.append("PACKAGE_R_ORACLE_PRODUCER_POSITION")
    errors.extend(_m018_helper_surface_errors(tree))
    return unique_errors(errors)


def current_package_oracle_binding_errors() -> list[str]:
    """Read and validate this producer's current source fail-closed."""
    try:
        source = Path(__file__).read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        return ["PACKAGE_R_ORACLE_PRODUCER_SOURCE_READ:" + str(error)]
    return package_oracle_binding_errors(source)


def class_literal(class_node: ast.ClassDef, name: str) -> object | None:
    values: list[ast.AST] = []
    for node in class_node.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name
               for target in targets):
            values.append(node.value)
    if len(values) != 1:
        return None
    try:
        return ast.literal_eval(values[0])
    except (TypeError, ValueError):
        return None


def class_method(
    class_node: ast.ClassDef, name: str,
) -> ast.FunctionDef | None:
    matches = [
        node for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    return matches[0] if len(matches) == 1 else None


def conan_field_errors(data: bytes) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(data.decode("ascii"))
    except (UnicodeError, SyntaxError):
        return ["PACKAGE_R_CONAN_FIELD:syntax"]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(classes) != 1 or classes[0].name != "DSParkConan":
        return ["PACKAGE_R_CONAN_FIELD:class"]
    recipe = classes[0]
    for field, expected in (
        ("name", "dspark"),
        ("version", "1.7.0"),
        ("package_type", "header-library"),
        ("homepage", "https://github.com/CristianMoresi/DSPark"),
    ):
        if class_literal(recipe, field) != expected:
            errors.append("PACKAGE_R_CONAN_FIELD:" + field)

    source = class_method(recipe, "source")
    get_calls = [] if source is None else [
        node for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "get"
    ]
    if len(get_calls) != 1:
        errors.append("PACKAGE_R_CONAN_FIELD:source_url")
    else:
        call = get_calls[0]
        url = call.args[1].value if len(call.args) == 2 \
            and isinstance(call.args[1], ast.Constant) \
            and isinstance(call.args[1].value, str) else None
        if len(call.args) != 2 or ast.unparse(call.args[0]) != "self":
            errors.append("PACKAGE_R_CONAN_FIELD:source_url")
        if not isinstance(url, str) or not url.startswith(
                PACKAGE_SOURCE_URL_PREFIX):
            errors.append("PACKAGE_R_CONAN_FIELD:source_url")
        elif url[len(PACKAGE_SOURCE_URL_PREFIX):] != PRODUCT_P7_COMMIT:
            errors.append("PACKAGE_R_CONAN_FIELD:source_ref")
        keywords = {
            keyword.arg: keyword.value for keyword in call.keywords
            if keyword.arg is not None
        }
        filename_nodes = [
            keyword.value for keyword in call.keywords
            if keyword.arg == "filename"
        ]
        if len(filename_nodes) != 1 \
                or not isinstance(filename_nodes[0], ast.Constant) \
                or filename_nodes[0].value != PACKAGE_SOURCE_FILENAME:
            errors.append("PACKAGE_R_CONAN_FIELD:filename")
        if len(keywords) != len(call.keywords) or set(keywords) != {
                "filename", "sha256", "strip_root"}:
            errors.append("PACKAGE_R_CONAN_FIELD:source_url")
        sha_node = keywords.get("sha256")
        if not isinstance(sha_node, ast.Constant) \
                or sha_node.value != PACKAGE_SOURCE_SHA256:
            errors.append("PACKAGE_R_CONAN_FIELD:source_sha256")
        strip_node = keywords.get("strip_root")
        if not isinstance(strip_node, ast.Constant) or strip_node.value is not True:
            errors.append("PACKAGE_R_CONAN_FIELD:strip_root")

    package = class_method(recipe, "package")
    module_loops = [] if package is None else [
        node for node in ast.walk(package)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name) and node.target.id == "module"
    ]
    modules: object | None = None
    if len(module_loops) == 1:
        try:
            modules = ast.literal_eval(module_loops[0].iter)
        except (TypeError, ValueError):
            modules = None
    if modules != ("Core", "Effects", "Analysis", "IO", "Music"):
        errors.append("PACKAGE_R_CONAN_FIELD:modules")
    copy_calls = [] if package is None else [
        node for node in ast.walk(package)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "copy"
    ]
    module_destinations = [
        ast.unparse(call.args[3]) for call in copy_calls
        if len(call.args) == 4 and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "*.h"
    ]
    umbrella_destinations = [
        ast.unparse(call.args[3]) for call in copy_calls
        if len(call.args) == 4 and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "DSPark.h"
    ]
    if module_destinations != [
            "os.path.join(self.package_folder, 'include', 'DSPark', module)"
    ] or umbrella_destinations != [
            "os.path.join(self.package_folder, 'include', 'DSPark')"
    ]:
        errors.append("PACKAGE_R_CONAN_FIELD:include_destination")

    package_info = class_method(recipe, "package_info")
    property_calls = [] if package_info is None else [
        node for node in ast.walk(package_info)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self.cpp_info.set_property"
        and len(node.args) == 2
        and all(isinstance(argument, ast.Constant) for argument in node.args)
    ]
    properties = {
        str(call.args[0].value): call.args[1].value for call in property_calls
    }
    if len(properties) != len(property_calls) \
            or properties.get("cmake_file_name") != "dspark":
        errors.append("PACKAGE_R_CONAN_FIELD:cmake_file_name")
    if properties.get("cmake_target_name") != "dspark::dspark":
        errors.append("PACKAGE_R_CONAN_FIELD:cmake_target_name")
    directory_values: dict[str, object] = {}
    if package_info is not None:
        for node in package_info.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = ast.unparse(node.targets[0])
            if target in ("self.cpp_info.bindirs", "self.cpp_info.libdirs"):
                try:
                    directory_values[target] = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    directory_values[target] = None
    if directory_values != {
        "self.cpp_info.bindirs": [],
        "self.cpp_info.libdirs": [],
    }:
        errors.append("PACKAGE_R_CONAN_FIELD:header_only_dirs")

    package_id = class_method(recipe, "package_id")
    clear_calls = [] if package_id is None else [
        node for node in ast.walk(package_id)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self.info.clear"
        and not node.args and not node.keywords
    ]
    if len(clear_calls) != 1:
        errors.append("PACKAGE_R_CONAN_FIELD:package_id")
    return unique_errors(errors)


def strip_cmake_comments(text: str) -> str:
    result: list[str] = []
    quoted = False
    escaped = False
    comment = False
    for character in text:
        if comment:
            if character == "\n":
                result.append(character)
                comment = False
            continue
        if quoted:
            result.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == "#":
            comment = True
        else:
            result.append(character)
            if character == '"':
                quoted = True
    if quoted or escaped:
        raise ValueError("unterminated quoted CMake argument")
    return "".join(result)


def cmake_argument_tokens(body: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(body):
        while index < len(body) and body[index].isspace():
            index += 1
        if index == len(body):
            break
        if body[index] == '"':
            index += 1
            token: list[str] = []
            escaped = False
            while index < len(body):
                character = body[index]
                index += 1
                if escaped:
                    token.append(character)
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    break
                else:
                    token.append(character)
            else:
                raise ValueError("unterminated CMake argument")
            if escaped:
                raise ValueError("unterminated CMake escape")
            if index < len(body) and not body[index].isspace():
                raise ValueError("concatenated CMake argument")
            tokens.append("".join(token))
            continue
        begin = index
        while index < len(body) and not body[index].isspace():
            if body[index] in '()"':
                raise ValueError("unsupported CMake token boundary")
            index += 1
        tokens.append(body[begin:index])
    return tokens


def cmake_commands(data: bytes) -> list[tuple[str, list[str]]]:
    text = strip_cmake_comments(data.decode("ascii"))
    commands: list[tuple[str, list[str]]] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text):
            break
        match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[index:])
        if match is None:
            raise ValueError("CMake command name expected")
        name = match.group(0)
        index += len(name)
        while index < len(text) and text[index].isspace():
            index += 1
        if index == len(text) or text[index] != "(":
            raise ValueError("CMake command opening parenthesis expected")
        index += 1
        begin = index
        quoted = False
        escaped = False
        depth = 1
        while index < len(text) and depth:
            character = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        if depth != 0 or quoted or escaped:
            raise ValueError("unbalanced CMake command")
        body = text[begin:index]
        index += 1
        commands.append((name, cmake_argument_tokens(body)))
    return commands


def cmake_keyword(args: list[str], keyword: str) -> str | None:
    positions = [index for index, value in enumerate(args) if value == keyword]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        return None
    return args[positions[0] + 1]


def vcpkg_field_errors(data: bytes) -> list[str]:
    try:
        commands = cmake_commands(data)
    except (UnicodeError, ValueError):
        return ["PACKAGE_R_VCPKG_FIELD:syntax"]
    errors: list[str] = []
    expected_names = [
        "vcpkg_from_github",
        "vcpkg_cmake_configure",
        "vcpkg_cmake_install",
        "vcpkg_cmake_config_fixup",
        "file",
        "vcpkg_install_copyright",
    ]
    if [name for name, _args in commands] != expected_names:
        errors.append("PACKAGE_R_VCPKG_FIELD:command_inventory")
    by_name: dict[str, list[list[str]]] = {}
    for name, args in commands:
        by_name.setdefault(name, []).append(args)
    source_rows = by_name.get("vcpkg_from_github", [])
    if len(source_rows) != 1:
        return unique_errors(errors + ["PACKAGE_R_VCPKG_FIELD:source_helper"])
    source = source_rows[0]
    if cmake_keyword(source, "REPO") != "CristianMoresi/DSPark":
        errors.append("PACKAGE_R_VCPKG_FIELD:repo")
    if cmake_keyword(source, "REF") != PRODUCT_P7_COMMIT:
        errors.append("PACKAGE_R_VCPKG_FIELD:ref")
    if cmake_keyword(source, "SHA512") != PACKAGE_SOURCE_SHA512:
        errors.append("PACKAGE_R_VCPKG_FIELD:sha512")
    if "HEAD_REF" in source:
        errors.append("PACKAGE_R_VCPKG_FIELD:head_ref_forbidden")
    if source != [
        "OUT_SOURCE_PATH", "SOURCE_PATH",
        "REPO", "CristianMoresi/DSPark",
        "REF", PRODUCT_P7_COMMIT,
        "SHA512", PACKAGE_SOURCE_SHA512,
    ]:
        errors.append("PACKAGE_R_VCPKG_FIELD:source_shape")
    configure = by_name.get("vcpkg_cmake_configure", [])
    if configure != [[
        "SOURCE_PATH", "${SOURCE_PATH}", "OPTIONS",
        "-DDSPARK_BUILD_CONFORMANCE=OFF", "-DDSPARK_BUILD_TESTS=OFF",
    ]]:
        errors.append("PACKAGE_R_VCPKG_FIELD:configure")
    if by_name.get("vcpkg_cmake_install", []) != [[]]:
        errors.append("PACKAGE_R_VCPKG_FIELD:install")
    if by_name.get("vcpkg_cmake_config_fixup", []) != [[
            "PACKAGE_NAME", "dspark", "CONFIG_PATH", "lib/cmake/dspark"]]:
        errors.append("PACKAGE_R_VCPKG_FIELD:config_fixup")
    if by_name.get("file", []) != [[
        "REMOVE_RECURSE", "${CURRENT_PACKAGES_DIR}/debug",
        "${CURRENT_PACKAGES_DIR}/lib",
    ]]:
        errors.append("PACKAGE_R_VCPKG_FIELD:cleanup")
    if by_name.get("vcpkg_install_copyright", []) != [[
            "FILE_LIST", "${SOURCE_PATH}/LICENSE"]]:
        errors.append("PACKAGE_R_VCPKG_FIELD:copyright")
    return unique_errors(errors)


def manifest_field_errors(data: bytes) -> list[str]:
    try:
        value = strict_json_value(data)
    except (UnicodeError, json.JSONDecodeError, DuplicateJsonKey):
        return ["PACKAGE_R_MANIFEST_FIELD:object_shape"]
    if not isinstance(value, dict):
        return ["PACKAGE_R_MANIFEST_FIELD:object_shape"]
    errors: list[str] = []
    expected_keys = {
        "name", "version", "description", "homepage", "license",
        "dependencies",
    }
    if set(value) != expected_keys:
        errors.append("PACKAGE_R_MANIFEST_FIELD:object_shape")
    if value.get("name") != "dspark":
        errors.append("PACKAGE_R_MANIFEST_FIELD:name")
    if value.get("version") != "1.7.0":
        errors.append("PACKAGE_R_MANIFEST_FIELD:version")
    if value.get("homepage") != "https://github.com/CristianMoresi/DSPark" \
            or value.get("license") != "MIT":
        errors.append("PACKAGE_R_MANIFEST_FIELD:metadata")
    dependencies = value.get("dependencies")
    if dependencies != [
        {"name": "vcpkg-cmake", "host": True},
        {"name": "vcpkg-cmake-config", "host": True},
    ]:
        errors.append("PACKAGE_R_MANIFEST_FIELD:dependencies")
    return unique_errors(errors)


def validate_package_data(data_by_path: dict[str, bytes]) -> list[str]:
    errors: list[str] = []
    semantic_functions = {
        "packaging/conan/conanfile.py": conan_semantics,
        "packaging/vcpkg/portfile.cmake": cmake_noncomment_semantics,
        "packaging/vcpkg/vcpkg.json": manifest_semantics,
    }
    field_functions = {
        "packaging/conan/conanfile.py": conan_field_errors,
        "packaging/vcpkg/portfile.cmake": vcpkg_field_errors,
        "packaging/vcpkg/vcpkg.json": manifest_field_errors,
    }
    for path, expected in EXPECTED_PACKAGE_HASHES.items():
        data = data_by_path.get(path)
        if not isinstance(data, bytes):
            errors.append("PACKAGE_R_MISSING:" + path)
            continue
        actual = digest(data)
        if actual != expected:
            errors.append("PACKAGE_R_BYTE_DRIFT:{}:{}".format(path, actual))
        try:
            semantic = semantic_functions[path](data)
        except (UnicodeError, SyntaxError, ValueError, json.JSONDecodeError):
            semantic = None
        if semantic is not None:
            semantic_hash = digest(semantic)
            if semantic_hash != EXPECTED_SEMANTIC_HASHES[path]:
                errors.append("PACKAGE_R_SEMANTIC_DRIFT:{}:{}".format(
                    path, semantic_hash))
        errors.extend(field_functions[path](data))
    if set(data_by_path) != set(EXPECTED_PACKAGE_HASHES):
        errors.append("PACKAGE_R_RECIPE_PATH_SET")
    return unique_errors(errors)


def replace_package_bytes(
    data: bytes, old: bytes, new: bytes, *, expected_count: int = 1,
) -> bytes:
    if data.count(old) != expected_count:
        raise ValueError("package mutant source anchor cardinality")
    return data.replace(old, new)


def dumped_manifest(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode(
        "ascii")


def package_mutant_cases(
    baseline: dict[str, bytes],
) -> list[dict[str, object]]:
    conan_path = "packaging/conan/conanfile.py"
    port_path = "packaging/vcpkg/portfile.cmake"
    manifest_path = "packaging/vcpkg/vcpkg.json"
    conan = baseline[conan_path]
    port = baseline[port_path]
    manifest = baseline[manifest_path]
    cases: list[dict[str, object]] = []

    def add(mutant_id: str, case: str, path: str, data: bytes,
            terminal: str) -> None:
        cases.append({
            "case": case,
            "data": data,
            "id": mutant_id,
            "path": path,
            "required_terminal": terminal,
        })

    add("MUT-R-CONAN-COMMENT", "one-comment-byte", conan_path,
        replace_package_bytes(conan, b"# DSPark Conan", b"# DSpark Conan"),
        "PACKAGE_R_BYTE_DRIFT:" + conan_path)
    add("MUT-R-CONAN-NAME", "uppercase", conan_path,
        replace_package_bytes(conan, b'name = "dspark"', b'name = "DSPark"'),
        "PACKAGE_R_CONAN_FIELD:name")
    add("MUT-R-CONAN-VERSION", "next-patch", conan_path,
        replace_package_bytes(conan, b'version = "1.7.0"', b'version = "1.7.1"'),
        "PACKAGE_R_CONAN_FIELD:version")
    add("MUT-R-CONAN-PACKAGE-TYPE", "static-library", conan_path,
        replace_package_bytes(
            conan, b'package_type = "header-library"',
            b'package_type = "static-library"'),
        "PACKAGE_R_CONAN_FIELD:package_type")
    add("MUT-R-CONAN-HOMEPAGE", "wrong-repository", conan_path,
        replace_package_bytes(
            conan, b'homepage = "https://github.com/CristianMoresi/DSPark"',
            b'homepage = "https://github.com/CristianMoresi/Other"'),
        "PACKAGE_R_CONAN_FIELD:homepage")
    add("MUT-R-CONAN-SOURCE-URL", "tag-url", conan_path,
        replace_package_bytes(
            conan,
            (PACKAGE_SOURCE_URL_PREFIX + PRODUCT_P7_COMMIT).encode("ascii"),
            b"https://github.com/CristianMoresi/DSPark/archive/refs/tags/v1.7.0.tar.gz"),
        "PACKAGE_R_CONAN_FIELD:source_url")
    filename_anchor = (
        b'            filename="' + PACKAGE_SOURCE_FILENAME.encode("ascii")
        + b'",\n'
    )
    add("MUT-R-CONAN-FILENAME", "absent", conan_path,
        replace_package_bytes(conan, filename_anchor, b""),
        "PACKAGE_R_CONAN_FIELD:filename")
    for case, replacement in (
        ("extensionless", "dspark-1.7.0"),
        ("zip", "dspark-1.7.0.zip"),
        ("posix-path", "archive/dspark-1.7.0.tar.gz"),
        ("windows-path", r"archive\\dspark-1.7.0.tar.gz"),
        ("url", "https://example.invalid/dspark-1.7.0.tar.gz"),
        ("version-drift", "dspark-1.7.1.tar.gz"),
        ("tag-ref", "dspark-v1.7.0.tar.gz"),
        ("branch-ref", "dspark-main.tar.gz"),
        ("short-ref", "dspark-" + PRODUCT_P7_COMMIT[:12] + ".tar.gz"),
        ("r-placeholder", "dspark-" + "R" * 40 + ".tar.gz"),
    ):
        replacement_line = (
            b'            filename="' + replacement.encode("ascii") + b'",\n'
        )
        add("MUT-R-CONAN-FILENAME", case, conan_path,
            replace_package_bytes(conan, filename_anchor, replacement_line),
            "PACKAGE_R_CONAN_FIELD:filename")
    add("MUT-R-CONAN-FILENAME", "duplicate", conan_path,
        replace_package_bytes(conan, filename_anchor,
                              filename_anchor + filename_anchor),
        "PACKAGE_R_CONAN_FIELD:filename")

    for case, reference in (
        ("parent", PRODUCT_P7_PARENT),
        ("short", PRODUCT_P7_COMMIT[:12]),
        ("tag", "v1.7.0"),
        ("branch", "main"),
        ("r-placeholder", "R" * 40),
    ):
        add("MUT-R-CONAN-REF", case, conan_path,
            replace_package_bytes(
                conan,
                (PACKAGE_SOURCE_URL_PREFIX + PRODUCT_P7_COMMIT).encode("ascii"),
                (PACKAGE_SOURCE_URL_PREFIX + reference).encode("ascii")),
            "PACKAGE_R_CONAN_FIELD:source_ref")
    changed_sha256 = ("4" + PACKAGE_SOURCE_SHA256[1:]).encode("ascii")
    add("MUT-R-CONAN-SHA256", "nibble", conan_path,
        replace_package_bytes(
            conan, PACKAGE_SOURCE_SHA256.encode("ascii"), changed_sha256),
        "PACKAGE_R_CONAN_FIELD:source_sha256")
    add("MUT-R-CONAN-STRIP-ROOT", "false", conan_path,
        replace_package_bytes(conan, b"strip_root=True", b"strip_root=False"),
        "PACKAGE_R_CONAN_FIELD:strip_root")
    add("MUT-R-CONAN-STRIP-ROOT", "absent", conan_path,
        replace_package_bytes(conan, b", strip_root=True", b""),
        "PACKAGE_R_CONAN_FIELD:strip_root")
    module_tuple = b'("Core", "Effects", "Analysis", "IO", "Music")'
    for case, replacement in (
        ("remove", b'("Core", "Effects", "Analysis", "IO")'),
        ("add", b'("Core", "Effects", "Analysis", "IO", "Music", "Extra")'),
        ("reorder", b'("Effects", "Core", "Analysis", "IO", "Music")'),
    ):
        add("MUT-R-CONAN-MODULES", case, conan_path,
            replace_package_bytes(conan, module_tuple, replacement),
            "PACKAGE_R_CONAN_FIELD:modules")
    add("MUT-R-CONAN-INCLUDE-DESTINATION", "lowercase", conan_path,
        replace_package_bytes(
            conan, b'"include", "DSPark"', b'"include", "dspark"',
            expected_count=2),
        "PACKAGE_R_CONAN_FIELD:include_destination")
    add("MUT-R-CONAN-CMAKE-FILE", "uppercase", conan_path,
        replace_package_bytes(
            conan, b'("cmake_file_name", "dspark")',
            b'("cmake_file_name", "DSPark")'),
        "PACKAGE_R_CONAN_FIELD:cmake_file_name")
    add("MUT-R-CONAN-CMAKE-TARGET", "uppercase", conan_path,
        replace_package_bytes(
            conan, b'("cmake_target_name", "dspark::dspark")',
            b'("cmake_target_name", "DSPark::DSPark")'),
        "PACKAGE_R_CONAN_FIELD:cmake_target_name")
    for case, old, new in (
        ("bindirs", b"self.cpp_info.bindirs = []",
         b'self.cpp_info.bindirs = ["bin"]'),
        ("libdirs", b"self.cpp_info.libdirs = []",
         b'self.cpp_info.libdirs = ["lib"]'),
    ):
        add("MUT-R-CONAN-LIB-BIN-DIRS", case, conan_path,
            replace_package_bytes(conan, old, new),
            "PACKAGE_R_CONAN_FIELD:header_only_dirs")
    add("MUT-R-CONAN-PACKAGE-ID", "clear-removed", conan_path,
        replace_package_bytes(
            conan, b"        self.info.clear()\n", b"        pass\n"),
        "PACKAGE_R_CONAN_FIELD:package_id")

    add("MUT-R-VCPKG-COMMENT", "one-comment-byte", port_path,
        replace_package_bytes(port, b"# DSPark vcpkg", b"# DSpark vcpkg"),
        "PACKAGE_R_BYTE_DRIFT:" + port_path)
    for case, replacement in (
        ("owner", b"Other/DSPark"),
        ("repository", b"CristianMoresi/Other"),
    ):
        add("MUT-R-VCPKG-REPO", case, port_path,
            replace_package_bytes(
                port, b"CristianMoresi/DSPark", replacement),
            "PACKAGE_R_VCPKG_FIELD:repo")
    for case, reference in (
        ("parent", PRODUCT_P7_PARENT),
        ("tag", "v1.7.0"),
        ("branch", "main"),
        ("short", PRODUCT_P7_COMMIT[:12]),
        ("r-placeholder", "R" * 40),
    ):
        add("MUT-R-VCPKG-REF", case, port_path,
            replace_package_bytes(
                port, ("REF " + PRODUCT_P7_COMMIT).encode("ascii"),
                ("REF " + reference).encode("ascii")),
            "PACKAGE_R_VCPKG_FIELD:ref")
    changed_sha512 = ("7" + PACKAGE_SOURCE_SHA512[1:]).encode("ascii")
    add("MUT-R-VCPKG-SHA512", "nibble", port_path,
        replace_package_bytes(
            port, PACKAGE_SOURCE_SHA512.encode("ascii"), changed_sha512),
        "PACKAGE_R_VCPKG_FIELD:sha512")
    add("MUT-R-VCPKG-HEAD-REF", "main", port_path,
        replace_package_bytes(
            port, b"    SHA512 " + PACKAGE_SOURCE_SHA512.encode("ascii") + b"\n",
            b"    SHA512 " + PACKAGE_SOURCE_SHA512.encode("ascii")
            + b"\n    HEAD_REF main\n"),
        "PACKAGE_R_VCPKG_FIELD:head_ref_forbidden")
    for case, old, new in (
        ("source", b'SOURCE_PATH "${SOURCE_PATH}"',
         b'SOURCE_PATH "${SOURCE_PATH}/wrong"'),
        ("conformance", b"-DDSPARK_BUILD_CONFORMANCE=OFF",
         b"-DDSPARK_BUILD_CONFORMANCE=ON"),
        ("tests", b"-DDSPARK_BUILD_TESTS=OFF",
         b"-DDSPARK_BUILD_TESTS=ON"),
    ):
        add("MUT-R-VCPKG-CONFIGURE", case, port_path,
            replace_package_bytes(port, old, new),
            "PACKAGE_R_VCPKG_FIELD:configure")
    for case, old, new in (
        ("name", b"PACKAGE_NAME dspark", b"PACKAGE_NAME DSPark"),
        ("path", b"CONFIG_PATH lib/cmake/dspark",
         b"CONFIG_PATH lib/cmake/DSPark"),
    ):
        add("MUT-R-VCPKG-FIXUP", case, port_path,
            replace_package_bytes(port, old, new),
            "PACKAGE_R_VCPKG_FIELD:config_fixup")
    for case, old in (
        ("debug", b'    "${CURRENT_PACKAGES_DIR}/debug"\n'),
        ("lib", b'    "${CURRENT_PACKAGES_DIR}/lib"'),
    ):
        add("MUT-R-VCPKG-CLEANUP", case, port_path,
            replace_package_bytes(port, old, b""),
            "PACKAGE_R_VCPKG_FIELD:cleanup")
    add("MUT-R-VCPKG-COPYRIGHT", "license-path", port_path,
        replace_package_bytes(
            port, b'"${SOURCE_PATH}/LICENSE"', b'"${SOURCE_PATH}/COPYING"'),
        "PACKAGE_R_VCPKG_FIELD:copyright")

    add("MUT-R-MANIFEST-BYTES", "whitespace", manifest_path,
        replace_package_bytes(
            manifest, b'{\n  "name"', b'{\n\n  "name"'),
        "PACKAGE_R_BYTE_DRIFT:" + manifest_path)
    manifest_value = strict_json_value(manifest)
    if not isinstance(manifest_value, dict):
        raise ValueError("manifest baseline object")
    reordered = {
        key: manifest_value[key] for key in (
            "version", "name", "description", "homepage", "license",
            "dependencies",
        )
    }
    add("MUT-R-MANIFEST-BYTES", "ordering", manifest_path,
        dumped_manifest(reordered), "PACKAGE_R_BYTE_DRIFT:" + manifest_path)
    changed = dict(manifest_value)
    changed["name"] = "DSPark"
    add("MUT-R-MANIFEST-NAME", "uppercase", manifest_path,
        dumped_manifest(changed), "PACKAGE_R_MANIFEST_FIELD:name")
    changed = dict(manifest_value)
    changed["version"] = "1.7.1"
    add("MUT-R-MANIFEST-VERSION", "next-patch", manifest_path,
        dumped_manifest(changed), "PACKAGE_R_MANIFEST_FIELD:version")
    for case, key, value in (
        ("homepage", "homepage", "https://github.com/Other/DSPark"),
        ("license", "license", "Apache-2.0"),
    ):
        changed = dict(manifest_value)
        changed[key] = value
        add("MUT-R-MANIFEST-HOMEPAGE-LICENSE", case, manifest_path,
            dumped_manifest(changed), "PACKAGE_R_MANIFEST_FIELD:metadata")
    dependency_mutants: list[tuple[str, list[dict[str, object]]]] = []
    dependencies = manifest_value.get("dependencies")
    if not isinstance(dependencies, list):
        raise ValueError("manifest baseline dependencies")
    dependency_mutants.append(("remove", [dict(dependencies[0])]))
    dependency_mutants.append((
        "add", [*(dict(item) for item in dependencies),
                {"name": "extra", "host": True}]))
    dependency_mutants.append((
        "reorder", [dict(dependencies[1]), dict(dependencies[0])]))
    renamed = [dict(item) for item in dependencies]
    renamed[0]["name"] = "vcpkg-cmake-wrong"
    dependency_mutants.append(("rename", renamed))
    host_cleared = [dict(item) for item in dependencies]
    host_cleared[0]["host"] = False
    dependency_mutants.append(("host-flag", host_cleared))
    for case, mutant_dependencies in dependency_mutants:
        changed = dict(manifest_value)
        changed["dependencies"] = mutant_dependencies
        add("MUT-R-MANIFEST-DEPENDENCIES", case, manifest_path,
            dumped_manifest(changed), "PACKAGE_R_MANIFEST_FIELD:dependencies")
    duplicate = replace_package_bytes(
        manifest, b'  "version": "1.7.0",\n',
        b'  "version": "1.7.0",\n  "version": "1.7.0",\n')
    add("MUT-R-MANIFEST-DUPLICATE-OR-EXTRA", "duplicate", manifest_path,
        duplicate, "PACKAGE_R_MANIFEST_FIELD:object_shape")
    changed = dict(manifest_value)
    changed["extra"] = True
    add("MUT-R-MANIFEST-DUPLICATE-OR-EXTRA", "extra", manifest_path,
        dumped_manifest(changed), "PACKAGE_R_MANIFEST_FIELD:object_shape")
    return cases


def terminal_seen(errors: list[str], terminal: str) -> bool:
    return any(error == terminal or error.startswith(terminal + ":")
               for error in errors)


def package_mutant_control_results(
    baseline: dict[str, bytes],
) -> tuple[list[dict[str, object]], list[str]]:
    integrity_errors: list[str] = []
    inventory_hash = digest("\n".join(
        PACKAGE_FIELD_MUTANT_IDS).encode("ascii"))
    if len(PACKAGE_FIELD_MUTANT_IDS) != 31 \
            or len(set(PACKAGE_FIELD_MUTANT_IDS)) != 31 \
            or inventory_hash != PACKAGE_FIELD_MUTANT_INVENTORY_SHA256:
        integrity_errors.append("PACKAGE_R_MUTANT_INVENTORY_LITERAL")
    try:
        cases = package_mutant_cases(baseline)
    except (KeyError, TypeError, ValueError) as error:
        return [], integrity_errors + [
            "PACKAGE_R_MUTANT_GENERATOR:" + str(error)]
    observed_ids = list(dict.fromkeys(str(case["id"]) for case in cases))
    if observed_ids != list(PACKAGE_FIELD_MUTANT_IDS):
        integrity_errors.append("PACKAGE_R_MUTANT_INVENTORY_ORDER")
    counts = tuple(sum(case["id"] == mutant_id for case in cases)
                   for mutant_id in PACKAGE_FIELD_MUTANT_IDS)
    if counts != PACKAGE_FIELD_MUTANT_SUBCASE_COUNTS \
            or len(cases) != EXPECTED_PACKAGE_FIELD_MUTANT_SUBCASES:
        integrity_errors.append("PACKAGE_R_MUTANT_SUBCASE_INVENTORY")

    grouped: dict[str, list[dict[str, object]]] = {
        mutant_id: [] for mutant_id in PACKAGE_FIELD_MUTANT_IDS
    }
    for case in cases:
        path = str(case["path"])
        mutated = dict(baseline)
        mutated[path] = bytes(case["data"])
        errors = validate_package_data(mutated)
        required = str(case["required_terminal"])
        passed = bool(errors) and terminal_seen(errors, required)
        grouped.setdefault(str(case["id"]), []).append({
            "case": case["case"],
            "errors": errors,
            "passed": passed,
            "required_terminal": required,
        })
    results = []
    for mutant_id, expected_count in zip(
            PACKAGE_FIELD_MUTANT_IDS, PACKAGE_FIELD_MUTANT_SUBCASE_COUNTS):
        subcases = grouped.get(mutant_id, [])
        results.append({
            "case_count": len(subcases),
            "expected_case_count": expected_count,
            "id": mutant_id,
            "status": "PASS" if len(subcases) == expected_count
            and all(bool(item["passed"]) for item in subcases) else "FAIL",
            "subcases": subcases,
        })
    return results, integrity_errors


def package_mutant_control_errors(
    baseline: dict[str, bytes],
) -> list[str]:
    results, errors = package_mutant_control_results(baseline)
    errors.extend(
        "PACKAGE_R_MUTANT_CONTROL:" + str(result["id"])
        for result in results if result["status"] != "PASS"
    )
    if len(results) != 31:
        errors.append("PACKAGE_R_MUTANT_RESULT_CARDINALITY")
    return unique_errors(errors)


def package_errors(root: Path) -> list[str]:
    data_by_path: dict[str, bytes] = {}
    errors: list[str] = []
    for path in EXPECTED_PACKAGE_HASHES:
        try:
            data_by_path[path] = canonical_text_bytes((root / path).read_bytes())
        except ValueError as error:
            errors.append("PACKAGE_R_EOL_DRIFT:{}:{}".format(path, error))
        except OSError as error:
            errors.append("PACKAGE_R_READ:{}:{}".format(path, error))
    errors.extend(validate_package_data(data_by_path))
    # Every normal invocation executes exactly the same production oracle and
    # literal field-mutant inventory used by --self-test.
    errors.extend(package_mutant_control_errors(data_by_path))
    return unique_errors(errors)


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
    if "ordinary suite authority is currently 902" not in ci:
        errors.append("CURRENT_TEST_AUTHORITY_MISSING 902")
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
    "ebf743a9e67d0236682043d679dc624cce2f82cc6977d0f0ee81da4bb8cefca4"
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
        except (FileNotFoundError, ProcessLookupError, PermissionError,
                ValueError, IndexError):
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
    package_data = {
        path: canonical_text_bytes((root / path).read_bytes())
        for path in EXPECTED_PACKAGE_HASHES
    }
    package_positive_errors = validate_package_data(package_data)
    package_results, package_inventory_errors = \
        package_mutant_control_results(package_data)
    checks.append(("package-r-positive", not package_positive_errors))
    checks.append((
        "package-r-inventory-integrity",
        not package_inventory_errors and len(package_results) == 31,
    ))
    details["package-r-positive"] = json.dumps(
        package_positive_errors, sort_keys=True)
    details["package-r-inventory-integrity"] = json.dumps(
        package_inventory_errors, sort_keys=True)
    for result in package_results:
        name = str(result["id"])
        checks.append((name, result["status"] == "PASS"))
        details[name] = json.dumps(result, sort_keys=True)
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
    package_r_bootstrap = globals().get(
        "package_r_executable_bootstrap_errors")
    if not callable(package_r_bootstrap):
        package_r_bootstrap_errors = [
            "PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_HELPER_MISSING"]
    else:
        try:
            package_r_bootstrap_source = Path(__file__).read_text(
                encoding="ascii")
            package_r_bootstrap_errors = package_r_bootstrap(
                package_r_bootstrap_source)
        except (AttributeError, NameError, OSError, SyntaxError, TypeError,
                UnicodeError, ValueError) as error:
            package_r_bootstrap_errors = [
                "PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_INVOCATION:"
                + type(error).__name__]
    if not isinstance(package_r_bootstrap_errors, list) \
            or not all(isinstance(item, str)
                       for item in package_r_bootstrap_errors):
        package_r_bootstrap_errors = [
            "PACKAGE_R_ORACLE_PRODUCER_BOOTSTRAP_RESULT"]
    if package_r_bootstrap_errors:
        for error in package_r_bootstrap_errors:
            print("ERROR " + error, file=sys.stderr)
        return 1

    if arguments.self_test:
        return self_test(root, arguments.doxygen, arguments.machine_json)
    if arguments.machine_json is not None:
        print("ERROR HARNESS --machine-json requires --self-test",
              file=sys.stderr)
        return 2

    errors: list[str] = []
    errors.extend(current_package_oracle_binding_errors())
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
        "exact Package R and 31 field-mutant oracles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
