#!/usr/bin/env python3
"""Validate every local file and fragment link in generated Doxygen HTML."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        values = dict(attributes)
        for name in ("id", "name"):
            if values.get(name):
                self.anchors.add(values[name] or "")
        for name in ("href", "src"):
            if values.get(name):
                self.links.append(values[name] or "")


def parse_html(path: Path) -> LinkParser:
    parser = LinkParser()
    parser.feed(path.read_text(encoding="utf-8", errors="strict"))
    return parser


def local_target(root: Path, source: Path, link: str) -> tuple[Path | None, str]:
    split = urlsplit(link)
    schemes = {"http", "https", "mailto", "ftp", "javascript", "data"}
    if split.scheme.lower() in schemes or split.netloc or link.startswith("mai'+'lto"):
        return None, ""
    raw_path = unquote(split.path)
    if raw_path.startswith("/"):
        target = root / raw_path.lstrip("/")
    elif raw_path:
        target = source.parent / raw_path
    else:
        target = source
    return target.resolve(), unquote(split.fragment)


def repair_doxygen_indexes(root: Path) -> int:
    """Add aliases for Doxygen 1.18 qindex links whose anchors it omits."""
    root = root.resolve()
    html_files = sorted(root.rglob("*.html"))
    parsed = {path: parse_html(path) for path in html_files}
    aliases: dict[Path, set[str]] = {}
    for source, page in parsed.items():
        for link in page.links:
            target, fragment = local_target(root, source, link)
            if target is None or target not in parsed or not fragment:
                continue
            if fragment in parsed[target].anchors:
                continue
            if re.fullmatch(r"index_[A-Za-z~]+", fragment):
                aliases.setdefault(target, set()).add(fragment)
    for target, fragments in aliases.items():
        content = target.read_text(encoding="utf-8")
        marker = '<div class="contents">'
        injection = "".join(
            f'<span id="{fragment}" class="dspark-doxygen-index-anchor"></span>'
            for fragment in sorted(fragments)
        )
        if marker not in content:
            raise RuntimeError(f"Doxygen contents marker missing in {target}")
        target.write_text(content.replace(marker, marker + injection, 1), encoding="utf-8")
    return sum(len(fragments) for fragments in aliases.values())


def validate(root: Path) -> list[str]:
    root = root.resolve()
    index = root / "index.html"
    if not index.is_file():
        return ["NO_GENERATED_HTML"]
    parsed: dict[Path, LinkParser] = {}
    errors: list[str] = []
    pending = [index]
    visited: set[Path] = set()
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        page = parsed.setdefault(source, parse_html(source))
        for link in page.links:
            target, fragment = local_target(root, source, link)
            if target is None:
                continue
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"LINK_ESCAPES_OUTPUT {source.relative_to(root)} {link}")
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"MISSING_LOCAL_TARGET {source.relative_to(root)} {link}")
                continue
            if target.suffix.lower() in {".html", ".htm"}:
                target_page = parsed.get(target)
                if target_page is None:
                    target_page = parse_html(target)
                    parsed[target] = target_page
                # SHOW_FILES=NO keeps Doxygen's auto-generated directory
                # inventories out of the public navigation. Doxygen 1.18 still
                # emits orphan dir_*.html pages (and lists non-input source
                # files in them), so validate that a breadcrumb target exists
                # but do not treat those hidden inventories as public pages.
                if not target.name.startswith("dir_") and target.name != "doxygen_crawl.html":
                    pending.append(target)
                if fragment and fragment not in target_page.anchors:
                    errors.append(
                        f"MISSING_LOCAL_FRAGMENT {source.relative_to(root)} {link}"
                    )
    return sorted(set(errors))


def self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="dspark-doc-link-test-") as directory:
        root = Path(directory)
        (root / "index.html").write_text(
            '<html><body><a href="other.html#present">ok</a></body></html>',
            encoding="utf-8",
        )
        # Doxygen versions use both HTML4 name anchors and HTML5 id anchors.
        # Either is a real semantic destination, rather than a guessed slug.
        (root / "other.html").write_text(
            '<html><body><a name="present"></a></body></html>', encoding="utf-8"
        )
        good = not validate(root)
        (root / "index.html").write_text(
            '<html><body><a href="missing.html#absent">bad</a></body></html>',
            encoding="utf-8",
        )
        bad = any(error.startswith("MISSING_LOCAL_TARGET") for error in validate(root))
        (root / "index.html").write_text(
            '<html><body><a href="other.html#absent">bad fragment</a></body></html>',
            encoding="utf-8",
        )
        missing_fragment = any(
            error.startswith("MISSING_LOCAL_FRAGMENT") for error in validate(root)
        )
    print(f"{'PASS' if good else 'FAIL'} generated-link-positive-control")
    print(f"{'PASS' if bad else 'FAIL'} mutant unresolved-local-link")
    print(
        f"{'PASS' if missing_fragment else 'FAIL'} "
        "mutant missing-generated-fragment"
    )
    return 0 if good and bad and missing_fragment else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-root", type=Path, default=Path("docs/api"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repair-doxygen-indexes", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        return self_test()
    if arguments.repair_doxygen_indexes:
        repaired = repair_doxygen_indexes(arguments.html_root)
        print(f"Repaired {repaired} omitted Doxygen qindex anchors")
    errors = validate(arguments.html_root)
    for error in errors:
        print(f"ERROR {error}")
    if errors:
        return 1
    print("PASS generated documentation local links and fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
