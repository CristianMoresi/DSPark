#!/usr/bin/env python3
"""Regenerate the compact Standard MIDI File interoperability fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent

FORMAT_0 = bytes.fromhex(
    "4d54686400000006000000010060"
    "4d54726b0000003b"
    "00ff580404021808"
    "00ff510307a120"
    "00c005"
    "00c12e"
    "00c246"
    "00923060"
    "003c60"
    "60914340"
    "60904c20"
    "8140823040"
    "003c40"
    "00814340"
    "00804c40"
    "00ff2f00"
)

FORMAT_1 = bytes.fromhex(
    "4d54686400000006000100040060"
    "4d54726b00000014"
    "00ff580404021808"
    "00ff510307a120"
    "8300ff2f00"
    "4d54726b00000010"
    "00c005"
    "8140904c20"
    "81404c00"
    "00ff2f00"
    "4d54726b0000000f"
    "00c12e"
    "60914340"
    "82204300"
    "00ff2f00"
    "4d54726b00000015"
    "00c246"
    "00923060"
    "003c60"
    "83003000"
    "003c00"
    "00ff2f00"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output_root: Path = ROOT) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    fixtures = {
        "rp001-format0.mid": {
            "data": FORMAT_0,
            "page": "12-13",
            "format": 0,
            "tracks": 1,
            "events": 14,
        },
        "rp001-format1.mid": {
            "data": FORMAT_1,
            "page": "13",
            "format": 1,
            "tracks": 4,
            "events": 17,
        },
    }

    entries = []
    for name, fixture in fixtures.items():
        data = fixture["data"]
        (output_root / name).write_bytes(data)
        entries.append(
            {
                "file": name,
                "sha256": sha256(data),
                "bytes": len(data),
                "format": fixture["format"],
                "tracks": fixture["tracks"],
                "semantic_events_including_eot": fixture["events"],
                "origin": (
                    "MIDI Manufacturers Association, Standard MIDI Files 1.0, "
                    f"RP-001 revised February 1996, pages {fixture['page']}"
                ),
                "license": (
                    "MIDI Manufacturers Association copyright; minimal factual "
                    "interoperability example bytes reproduced for conformance"
                ),
            }
        )

    source = Path(__file__).read_bytes()
    manifest = {
        "schema_version": "1",
        "generator": "generate_fixtures.py",
        "generator_sha256": sha256(source),
        "command": "python3 tests/fixtures/midi/generate_fixtures.py",
        "source_locator": "https://midi.org/standard-midi-files-specification",
        "fixtures": entries,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return manifest


def fixture_hashes(root: Path) -> dict[str, bytes]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).digest()
        for path in root.iterdir()
        if path.is_file() and (path.suffix == ".mid" or path.name == "manifest.json")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        committed = fixture_hashes(ROOT)
        with tempfile.TemporaryDirectory(prefix="dspark-midi-fixtures-") as temporary:
            temporary_root = Path(temporary)
            manifest = generate(temporary_root)
            regenerated = fixture_hashes(temporary_root)
        if committed != regenerated:
            changed = sorted(committed.keys() | regenerated.keys())
            raise SystemExit("MIDI fixture regeneration changed: " + ", ".join(changed))
        print(f"verified {len(manifest['fixtures'])} RP-001 MIDI fixtures")
    else:
        manifest = generate()
        print(f"generated {len(manifest['fixtures'])} RP-001 MIDI fixtures")


if __name__ == "__main__":
    main()
