#!/usr/bin/env python3
"""Regenerate the deterministic DSPark FLAC conformance fixtures.

The declarative encoder below only emits explicitly named RFC 9639 fields. It
is intentionally small and is not used by DSPark itself. Every complete stream
is independently decoded by libFLAC before its hashes enter the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
TOOL_SOURCE = ROOT / "libflac_fixture_tool.cpp"
GENERATOR_COMMAND = "python3 tests/fixtures/flac/generate_fixtures.py"
VERIFICATION_COMMAND = "python3 tests/fixtures/flac/generate_fixtures.py --verify"
COMPILE_COMMAND = (
    "g++ -std=c++20 -Wall -Wextra -Wpedantic -Werror "
    "tests/fixtures/flac/libflac_fixture_tool.cpp "
    "$(pkg-config --cflags --libs flac) "
    "-o tests/fixtures/flac/.libflac_fixture_tool"
)


class Bits:
    def __init__(self) -> None:
        self.values: list[int] = []

    def unsigned(self, value: int, width: int) -> None:
        if width < 0 or value < 0 or (width != 0 and value >= 1 << width):
            raise ValueError(f"unsigned value {value} does not fit {width} bits")
        if width == 0:
            if value != 0:
                raise ValueError("nonzero zero-width value")
            return
        self.values.extend((value >> bit) & 1 for bit in range(width - 1, -1, -1))

    def signed(self, value: int, width: int) -> None:
        if width == 0:
            if value != 0:
                raise ValueError("nonzero zero-width signed value")
            return
        low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if not low <= value <= high:
            raise ValueError(f"signed value {value} does not fit {width} bits")
        self.unsigned(value & ((1 << width) - 1), width)

    def unary(self, zeros: int) -> None:
        if zeros < 0:
            raise ValueError("negative unary count")
        self.values.extend([0] * zeros)
        self.values.append(1)

    def bytes(self) -> bytes:
        values = list(self.values)
        values.extend([0] * ((-len(values)) % 8))
        output = bytearray()
        for offset in range(0, len(values), 8):
            value = 0
            for bit in values[offset:offset + 8]:
                value = (value << 1) | bit
            output.append(value)
        return bytes(output)


def crc8(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = ((value << 1) ^ (0x07 if value & 0x80 else 0)) & 0xFF
    return value


def crc16(data: bytes) -> int:
    value = 0
    for byte in data:
        value ^= byte << 8
        for _ in range(8):
            value = ((value << 1) ^ (0x8005 if value & 0x8000 else 0)) & 0xFFFF
    return value


def coded_number(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFFF:
        raise ValueError("coded number out of range")
    if value < 0x80:
        return bytes([value])
    limits = [(2, 0x800), (3, 0x10000), (4, 0x200000),
              (5, 0x4000000), (6, 0x80000000), (7, 0x1000000000)]
    for length, limit in limits:
        if value < limit:
            break
    payload = [0] * length
    remaining = value
    for index in range(length - 1, 0, -1):
        payload[index] = 0x80 | (remaining & 0x3F)
        remaining >>= 6
    prefixes = {2: 0xC0, 3: 0xE0, 4: 0xF0, 5: 0xF8, 6: 0xFC, 7: 0xFE}
    payload[0] = prefixes[length] | remaining
    return bytes(payload)


def pcm_bytes(samples: list[list[int]], bits_per_sample: int) -> bytes:
    width = (bits_per_sample + 7) // 8
    output = bytearray()
    for frame in zip(*samples):
        for value in frame:
            output.extend(int(value).to_bytes(width, "little", signed=True))
    return bytes(output)


def pcm32_bytes(samples: list[list[int]]) -> bytes:
    return b"".join(struct.pack("<i", value) for frame in zip(*samples) for value in frame)


def fixed_prediction(values: list[int], index: int, order: int) -> int:
    coefficients = ((), (1,), (2, -1), (3, -3, 1), (4, -6, 4, -1))
    return sum(coefficients[order][j] * values[index - j - 1] for j in range(order))


def residual_values(values: list[int], spec: dict[str, Any]) -> tuple[int, list[int]]:
    kind = spec["kind"]
    if kind == "fixed":
        order = int(spec["order"])
        residual = [values[index] - fixed_prediction(values, index, order)
                    for index in range(order, len(values))]
        return order, residual
    if kind == "lpc":
        coefficients = list(spec["coefficients"])
        order = len(coefficients)
        shift = int(spec.get("shift", 0))
        residual = []
        for index in range(order, len(values)):
            prediction_sum = sum(coefficients[j] * values[index - j - 1]
                                 for j in range(order))
            prediction = (prediction_sum // (1 << shift) if shift >= 0
                          else prediction_sum * (1 << -shift))
            residual.append(values[index] - prediction)
        return order, residual
    return 0, []


def write_residual(bits: Bits, values: list[int], predictor_order: int,
                   spec: dict[str, Any], block_size: int) -> None:
    method = int(spec.get("method", 0))
    partition_order = int(spec.get("partition_order", 0))
    if method not in (0, 1):
        raise ValueError("bad Rice method")
    partitions = 1 << partition_order
    if block_size % partitions:
        raise ValueError("partition does not divide block")
    bits.unsigned(method, 2)
    bits.unsigned(partition_order, 4)
    parameter_width = 4 if method == 0 else 5
    parameters = spec.get("parameters", [spec.get("parameter", 2)] * partitions)
    escapes = spec.get("escape_widths", [None] * partitions)
    cursor = 0
    partition_size = block_size // partitions
    for partition in range(partitions):
        count = partition_size - (predictor_order if partition == 0 else 0)
        part = values[cursor:cursor + count]
        cursor += count
        escape_width = escapes[partition]
        if escape_width is not None:
            bits.unsigned((1 << parameter_width) - 1, parameter_width)
            bits.unsigned(int(escape_width), 5)
            for value in part:
                bits.signed(value, int(escape_width))
        else:
            parameter = int(parameters[partition])
            bits.unsigned(parameter, parameter_width)
            for value in part:
                folded = value * 2 if value >= 0 else (-value * 2) - 1
                bits.unary(folded >> parameter)
                bits.unsigned(folded & ((1 << parameter) - 1), parameter)
    if cursor != len(values):
        raise ValueError("residual partition accounting mismatch")


def subframe(samples: list[int], stored_depth: int, spec: dict[str, Any]) -> bytes:
    wasted = int(spec.get("wasted", 0))
    if wasted and any(value % (1 << wasted) for value in samples):
        raise ValueError("wasted-bit samples are not divisible")
    values = [value // (1 << wasted) for value in samples]
    depth = stored_depth - wasted
    kind = spec["kind"]
    if kind == "constant":
        type_code = 0
    elif kind == "verbatim":
        type_code = 1
    elif kind == "fixed":
        type_code = 8 + int(spec["order"])
    elif kind == "lpc":
        type_code = 31 + len(spec["coefficients"])
    else:
        raise ValueError(f"unknown subframe {kind}")

    bits = Bits()
    bits.unsigned(0, 1)
    bits.unsigned(type_code, 6)
    bits.unsigned(1 if wasted else 0, 1)
    if wasted:
        bits.unary(wasted - 1)
    if kind == "constant":
        if any(value != values[0] for value in values):
            raise ValueError("non-constant constant subframe")
        bits.signed(values[0], depth)
    elif kind == "verbatim":
        for value in values:
            bits.signed(value, depth)
    else:
        predictor_order, residual = residual_values(values, spec)
        for value in values[:predictor_order]:
            bits.signed(value, depth)
        if kind == "lpc":
            precision = int(spec.get("precision", 15))
            bits.unsigned(precision - 1, 4)
            bits.signed(int(spec.get("shift", 0)), 5)
            for coefficient in spec["coefficients"]:
                bits.signed(int(coefficient), precision)
        write_residual(bits, residual, predictor_order, spec, len(values))
    return bits.bytes()


def transform_channels(samples: list[list[int]], mode: int) -> list[list[int]]:
    if mode < 8:
        return [list(channel) for channel in samples]
    left, right = samples
    side = [a - b for a, b in zip(left, right)]
    if mode == 8:
        return [list(left), side]
    if mode == 9:
        return [side, list(right)]
    return ([(a + b) // 2 for a, b in zip(left, right)], side)


def make_frame(samples: list[list[int]], bits_per_sample: int, sample_rate: int,
               channel_mode: int, specs: list[dict[str, Any]], variable: bool,
               coded: int, rate_code: int = 0) -> tuple[bytes, dict[str, Any]]:
    block_size = len(samples[0])
    if any(len(channel) != block_size for channel in samples):
        raise ValueError("channel sizes differ")
    if block_size <= 256:
        block_code, block_extra = 6, bytes([block_size - 1])
    else:
        block_code, block_extra = 7, struct.pack(">H", block_size - 1)
    rate_extra = b""
    if rate_code == 12:
        if sample_rate % 1000:
            raise ValueError("rate code 12 needs kHz")
        rate_extra = bytes([sample_rate // 1000])
    elif rate_code == 13:
        rate_extra = struct.pack(">H", sample_rate)
    elif rate_code == 14:
        if sample_rate % 10:
            raise ValueError("rate code 14 needs 10-Hz units")
        rate_extra = struct.pack(">H", sample_rate // 10)
    elif rate_code != 0:
        raise ValueError("generator only emits explicit or STREAMINFO rate")

    header = bytearray([0xFF, 0xF8 | int(variable),
                        (block_code << 4) | rate_code, channel_mode << 4])
    header.extend(coded_number(coded))
    header.extend(block_extra)
    header.extend(rate_extra)
    header.append(crc8(header))
    transformed = transform_channels(samples, channel_mode)
    if len(transformed) != len(specs):
        raise ValueError("subframe spec count mismatch")
    payload = bytearray()
    kinds: list[str] = []
    residuals: list[str] = []
    for channel, (values, spec) in enumerate(zip(transformed, specs)):
        depth = bits_per_sample + int(
            (channel_mode == 8 and channel == 1)
            or (channel_mode == 9 and channel == 0)
            or (channel_mode == 10 and channel == 1))
        payload.extend(subframe(values, depth, spec))
        kind = spec["kind"].upper()
        if kind in ("FIXED", "LPC"):
            kind += str(spec.get("order", len(spec.get("coefficients", []))))
            method = int(spec.get("method", 0))
            residuals.append("RICE5" if method else "RICE4")
            for width in spec.get("escape_widths", []):
                if width is not None:
                    residuals.append(f"ESCAPE{width}")
        if spec.get("wasted"):
            kind += f"_WASTED{spec['wasted']}"
        kinds.append(kind)
    frame_without_crc = bytes(header + payload)
    frame = frame_without_crc + struct.pack(">H", crc16(frame_without_crc))
    return frame, {
        "blocking": "variable" if variable else "fixed",
        "block_size": block_size,
        "sample_rate_code": rate_code,
        "channel_mode": {8: "left_side", 9: "side_right", 10: "mid_side"}.get(
            channel_mode, "independent"),
        "subframes": kinds,
        "residual_modes": residuals,
    }


def make_stream(frames: list[list[list[int]]], bits_per_sample: int,
                sample_rate: int, channel_mode: int,
                frame_specs: list[list[dict[str, Any]]], variable: bool = False,
                rate_codes: list[int] | None = None, zero_md5: bool = False,
                unknown_total: bool = False) -> tuple[bytes, list[list[int]], list[dict[str, Any]]]:
    channels = len(frames[0])
    complete = [[] for _ in range(channels)]
    encoded_frames: list[bytes] = []
    inventory: list[dict[str, Any]] = []
    first_sample = 0
    for number, block in enumerate(frames):
        if len(block) != channels:
            raise ValueError("frame channel count changed")
        coded = first_sample if variable else number
        encoded, measured = make_frame(
            block, bits_per_sample, sample_rate, channel_mode,
            frame_specs[number], variable, coded,
            (rate_codes or [0] * len(frames))[number])
        encoded_frames.append(encoded)
        inventory.append(measured)
        for channel in range(channels):
            complete[channel].extend(block[channel])
        first_sample += len(block[0])
    sizes = [len(block[0]) for block in frames]
    packed = ((sample_rate << 44) | ((channels - 1) << 41)
              | ((bits_per_sample - 1) << 36)
              | (0 if unknown_total else first_sample))
    digest = bytes(16) if zero_md5 else hashlib.md5(
        pcm_bytes(complete, bits_per_sample)).digest()
    streaminfo = (struct.pack(">HH", min(sizes), max(sizes)) + bytes(6)
                  + packed.to_bytes(8, "big") + digest)
    stream = b"fLaC" + bytes([0x80, 0, 0, 34]) + streaminfo + b"".join(encoded_frames)
    return stream, complete, inventory


RFC_D1 = bytes.fromhex(
    "664c6143800000221000100000000f00000f0ac442f0000000013e84b41807dc69"
    "0307586a3dad1a2e0ffff869180000bf0358fd03128baa9a")
RFC_D2 = bytes.fromhex(
    "664c614300000022001000100000170000440ac442f000000013d5b0564975e98b8d8b93"
    "0422757b8103030000120000000000000000000000000000000000100400003a20000000"
    "7265666572656e6365206c6962464c414320312e332e3320323031393038303401000000"
    "0e0000005449544c453dd7a9d79cd795d79d81000006000000000000fff86998000f9912"
    "086701623d1442998f5df70d6fe00c17caeb21000ee7a77a24a1590c1217b603097b784f"
    "aa9a33d285e070ad5b1b4851b4010d99d2cd1a68f1e6b810fff869180102a402c382c40b"
    "c14a03ee48dd03b67c1330")
RFC_D3 = bytes.fromhex(
    "664c6143800000221000100000001f00001f07d0007000000018f8f9e396f5cbcfc6dc807f"
    "9977906b32fff868020017e944004f6f313d1047d227cb6d090831452bdc2822228057a3")

RFC_PCM: dict[str, tuple[int, int, list[list[int]]]] = {
    "rfc9639-d1": (44100, 16, [[25588], [10416]]),
    "rfc9639-d2": (44100, 16, [
        [10372, 18041, 14942, 17876, 15627, 17899, 16242, 18077, 16824,
         18263, 17295, -14418, -15201, -14508, -15195, -14818, -15486,
         -15349, -16054],
        [6070, 10545, 8743, 10449, 9143, 10463, 9502, 10569, 9840,
         10680, 10113, -8428, -8895, -8476, -8896, -8653, -9072,
         -8958, -9410]]),
    "rfc9639-d3": (32000, 8, [[0, 79, 111, 78, 8, -61, -90, -68, -13,
                               42, 67, 53, 13, -27, -46, -38, -12, 14, 24,
                               19, 6, -4, -5, 0]]),
}


def declarative_vectors() -> list[dict[str, Any]]:
    ramp = [index - 16 for index in range(32)]
    cubic = [index ** 3 - 15000 for index in range(32)]
    quartic = [(index - 8) ** 4 - 20000 for index in range(32)]
    lpc32_coefficients = [16383] * 16 + [-16384] * 16
    side_max = (1 << 32) - 1
    lpc_side = [side_max] * 32
    lpc_side.append(sum(lpc32_coefficients[j] * lpc_side[-j - 1]
                        for j in range(32)) // (1 << 15))
    lpc_mid = [-1] * 32 + [0]
    lpc_left = [(mid * 2 + (side & 1) + side) // 2
                for mid, side in zip(lpc_mid, lpc_side)]
    lpc_right = [(mid * 2 + (side & 1) - side) // 2
                 for mid, side in zip(lpc_mid, lpc_side)]
    stereo_left = [(-1200 + index * 71) for index in range(32)]
    stereo_right = [(900 - index * 53) for index in range(32)]
    negative_side_left = [-3 + (index % 3) for index in range(32)]
    negative_side_right = [2 + (index % 2) for index in range(32)]

    common: list[dict[str, Any]] = [
        {"name": "decl-constant", "bits": 16, "rate": 48000,
         "frames": [[[-7] * 32]], "mode": 0,
         "specs": [[{"kind": "constant"}]]},
        {"name": "decl-verbatim-wasted", "bits": 16, "rate": 48000,
         "frames": [[[value * 4 for value in ramp]]], "mode": 0,
         "specs": [[{"kind": "verbatim", "wasted": 2}]]},
        {"name": "decl-fixed0-rice4", "bits": 16, "rate": 48000,
         "frames": [[[(-1 if i % 3 else 2) for i in range(32)]]], "mode": 0,
         "specs": [[{"kind": "fixed", "order": 0, "method": 0, "parameter": 2}]]},
        {"name": "decl-fixed1-rice5", "bits": 16, "rate": 48000,
         "frames": [[ramp]], "mode": 0,
         "specs": [[{"kind": "fixed", "order": 1, "method": 1, "parameter": 0}]]},
        {"name": "decl-fixed2-escape0", "bits": 16, "rate": 48000,
         "frames": [[ramp]], "mode": 0,
         "specs": [[{"kind": "fixed", "order": 2, "method": 0,
                      "escape_widths": [0]}]]},
        {"name": "decl-fixed3-escape31", "bits": 24, "rate": 48000,
         "frames": [[cubic]], "mode": 0,
         "specs": [[{"kind": "fixed", "order": 3, "method": 0,
                      "escape_widths": [31]}]]},
        {"name": "decl-fixed4-partitioned", "bits": 24, "rate": 48000,
         "frames": [[quartic]], "mode": 0,
         "specs": [[{"kind": "fixed", "order": 4, "method": 0,
                      "partition_order": 1, "parameters": [2, 2]}]]},
        {"name": "decl-lpc1", "bits": 16, "rate": 48000,
         "frames": [[[123] * 32]], "mode": 0,
         "specs": [[{"kind": "lpc", "coefficients": [1], "precision": 2,
                      "method": 0, "parameter": 0}]]},
        {"name": "decl-lpc12", "bits": 24, "rate": 48000,
         "frames": [[ramp]], "mode": 0,
         "specs": [[{"kind": "lpc", "coefficients": [1] + [0] * 11,
                      "precision": 2, "method": 1, "parameter": 0}]]},
        {"name": "decl-lpc32-mid-side", "bits": 32, "rate": 96000,
         "frames": [[lpc_left, lpc_right]], "mode": 10,
         "specs": [[{"kind": "verbatim"},
                     {"kind": "lpc", "coefficients": lpc32_coefficients,
                      "precision": 15, "shift": 15, "method": 0,
                      "escape_widths": [0]}]]},
        {"name": "decl-left-side", "bits": 16, "rate": 44100,
         "frames": [[stereo_left, stereo_right]], "mode": 8,
         "specs": [[{"kind": "verbatim"}, {"kind": "verbatim"}]]},
        {"name": "decl-side-right", "bits": 16, "rate": 44100,
         "frames": [[stereo_left, stereo_right]], "mode": 9,
         "specs": [[{"kind": "verbatim"}, {"kind": "verbatim"}]]},
        {"name": "decl-mid-side-negative-odd", "bits": 16, "rate": 44100,
         "frames": [[negative_side_left, negative_side_right]], "mode": 10,
         "specs": [[{"kind": "verbatim"}, {"kind": "verbatim"}]]},
        {"name": "decl-variable-uncommon-rate", "bits": 16, "rate": 12345,
         "frames": [[[value * 3 for value in range(16)]],
                    [[100 - value * 2 for value in range(17)]]], "mode": 0,
         "specs": [[{"kind": "verbatim"}], [{"kind": "verbatim"}]],
         "variable": True, "rate_codes": [13, 13]},
        {"name": "decl-rice-unary-cap", "bits": 32, "rate": 48000,
         "frames": [[[524288] + [0] * 15]], "mode": 0,
         "specs": [[{"kind": "fixed", "order": 0, "method": 0,
                      "parameter": 0}]]},
        {"name": "decl-zero-md5-unknown-total", "bits": 16, "rate": 48000,
         "frames": [[[19] * 16]], "mode": 0,
         "specs": [[{"kind": "constant"}]],
         "zero_md5": True, "unknown_total": True},
        {"name": "decl-depth4-rate-code12", "bits": 4, "rate": 48000,
         "frames": [[[(index % 16) - 8 for index in range(32)]]], "mode": 0,
         "specs": [[{"kind": "verbatim"}]], "rate_codes": [12]},
        {"name": "decl-depth12-rate-code14", "bits": 12, "rate": 44100,
         "frames": [[[(index * 127 % 4096) - 2048 for index in range(32)]]], "mode": 0,
         "specs": [[{"kind": "verbatim"}]], "rate_codes": [14]},
        {"name": "decl-depth20", "bits": 20, "rate": 88200,
         "frames": [[[(index * 8191 % (1 << 20)) - (1 << 19)
                       for index in range(32)]]], "mode": 0,
         "specs": [[{"kind": "verbatim"}]]},
        {"name": "decl-eight-channel", "bits": 16, "rate": 48000,
         "frames": [[[channel * 100 + index for index in range(16)]
                     for channel in range(8)]], "mode": 7,
         "specs": [[{"kind": "verbatim"} for _ in range(8)]]},
    ]
    return common


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def oracle_decode(tool: Path, flac_path: Path, reference_path: Path) -> None:
    subprocess.run([str(tool), "decode", str(flac_path), str(reference_path)], check=True)


def fixture_entry(name: str, origin: str, license_name: str,
                  rate: int, bits: int, samples: list[list[int]],
                  inventory: list[dict[str, Any]], root: Path,
                  tool: Path) -> dict[str, Any]:
    flac_path = root / f"{name}.flac"
    source_path = root / f"{name}.source.pcm"
    reference_path = root / f"{name}.reference.pcm"
    expected = pcm32_bytes(samples)
    write_file(source_path, expected)
    oracle_decode(tool, flac_path, reference_path)
    if reference_path.read_bytes() != expected:
        raise RuntimeError(f"libFLAC PCM mismatch for {name}")
    return {
        "name": name,
        "origin": origin,
        "license": license_name,
        "sample_rate": rate,
        "channels": len(samples),
        "bits_per_sample": bits,
        "sample_frames": len(samples[0]),
        "source_pcm_sha256": sha(source_path),
        "flac_sha256": sha(flac_path),
        "reference_pcm_sha256": sha(reference_path),
        "actual_inventory": inventory,
        "oracle_command": (
            "tests/fixtures/flac/.libflac_fixture_tool decode "
            f"tests/fixtures/flac/{name}.flac "
            f"tests/fixtures/flac/{name}.reference.pcm"),
    }


def make_malformed(base: bytes) -> list[tuple[str, bytes, str]]:
    cases: list[tuple[str, bytes, str]] = []
    def add(name: str, data: bytes | bytearray, reason: str) -> None:
        cases.append((name, bytes(data), reason))
    for index in range(4):
        changed = bytearray(base); changed[index] ^= 0x20
        add(f"marker-byte-{index}", changed, "native marker mismatch")
    for length in range(4):
        add(f"truncated-marker-{length}", base[:length], "truncated native marker")
    for length in range(4, 8):
        add(f"truncated-metadata-header-{length}", base[:length], "truncated metadata header")
    for encoded_length in (0, 1, 33, 35, 0xFFFFFF):
        changed = bytearray(base)
        changed[5:8] = encoded_length.to_bytes(3, "big")
        add(f"streaminfo-length-{encoded_length}", changed, "STREAMINFO length is not 34")
    edits = [
        ("min-block-zero", 8, b"\x00\x00", "minimum block size zero"),
        ("min-block-fifteen", 8, b"\x00\x0f", "minimum block size below 16"),
        ("min-block-over-max", 8, b"\xff\xff", "minimum block exceeds maximum"),
        ("max-block-fifteen", 10, b"\x00\x0f", "maximum block size below 16"),
        ("sample-rate-zero", 18, b"\x00\x00\x00", "sample rate zero"),
    ]
    for name, offset, value, reason in edits:
        changed = bytearray(base); changed[offset:offset + len(value)] = value
        add(name, changed, reason)
    changed = bytearray(base); changed[4] = 0xFF
    add("metadata-type-127", changed, "reserved metadata type")
    changed = bytearray(base); changed[4] = 0x81
    add("streaminfo-not-first", changed, "first metadata block is not STREAMINFO")
    changed = bytearray(base); changed[4] = 0
    add("unterminated-metadata", changed, "metadata last flag missing")
    add("metadata-without-frame", base[:42], "no audio frame")
    duplicate = base[:42] + bytes([0x80, 0, 0, 34]) + base[8:42] + base[42:]
    add("duplicate-streaminfo", duplicate, "duplicate STREAMINFO")
    changed = bytearray(base); changed[22] ^= 1
    add("declared-sample-mismatch", changed, "declared sample coverage mismatch")
    changed = bytearray(base); changed[26] ^= 1
    add("nonzero-md5-mismatch", changed, "STREAMINFO MD5 mismatch")
    frame = 42
    header_mutations = [
        ("frame-sync", 0, 0x01, "frame sync mismatch"),
        ("frame-reserved-bit", 1, 0x02, "reserved frame bit"),
        ("frame-block-code-zero", 2, 0x00, "reserved block-size code"),
        ("frame-rate-code-15", 2, 0x0F, "reserved sample-rate code"),
        ("frame-channel-code-11", 3, 0xB0, "reserved channel assignment"),
        ("frame-depth-code-3", 3, 0x06, "reserved sample-depth code"),
        ("frame-fourth-reserved", 3, 0x01, "reserved frame-header bit"),
    ]
    for name, offset, mask, reason in header_mutations:
        changed = bytearray(base)
        if name in ("frame-block-code-zero", "frame-rate-code-15"):
            changed[frame + offset] = ((changed[frame + offset] & 0x0F) if "block" in name
                                       else (changed[frame + offset] & 0xF0)) | mask
        elif name in ("frame-channel-code-11", "frame-depth-code-3"):
            changed[frame + offset] = ((changed[frame + offset] & 0x0F) | mask
                                       if "channel" in name else
                                       (changed[frame + offset] & 0xF1) | mask)
        else:
            changed[frame + offset] ^= mask
        add(name, changed, reason)
    changed = bytearray(base); changed[frame + 6] ^= 1
    repaired_footer = crc16(changed[frame:-2])
    changed[-2:] = repaired_footer.to_bytes(2, "big")
    add("frame-crc8", changed, "frame-header CRC-8 mismatch")
    changed = bytearray(base); changed[-1] ^= 1
    add("frame-crc16", changed, "frame CRC-16 mismatch")
    for removed in range(1, 13):
        add(f"truncated-frame-tail-{removed}", base[:-removed], "truncated frame field")
    add("trailing-zero", base + b"\x00", "trailing byte")
    add("trailing-frame-prefix", base + b"\xff\xf8", "truncated extra frame")
    for index in range(8):
        changed = bytearray(base); changed[frame + 7 + index] ^= 1 << (index % 8)
        add(f"subframe-bit-corruption-{index}", changed, "subframe or payload corruption")
    if len(cases) < 48:
        raise RuntimeError("malformed corpus too small")
    return cases


def deep_malformed(declarative: dict[str, bytes]) -> list[tuple[str, bytes, str]]:
    """Create field-specific invalid streams with all outer CRCs repaired."""
    cases: list[tuple[str, bytes, str]] = []
    base = declarative["decl-constant"]
    frame = 42
    header_crc = frame + 6

    def add(name: str, data: bytes | bytearray, reason: str) -> None:
        cases.append((name, bytes(data), reason))

    def repair(data: bytearray, frame_offset: int = frame,
               crc_position: int = header_crc, footer_position: int | None = None) -> None:
        data[crc_position] = crc8(data[frame_offset:crc_position])
        footer = len(data) - 2 if footer_position is None else footer_position
        data[footer:footer + 2] = crc16(data[frame_offset:footer]).to_bytes(2, "big")

    header_edits = [
        ("frame-reserved-sync-bit-repaired", 43, 0xFA, "reserved frame-header sync bit"),
        ("frame-channel-property-change", 45, 0x10, "channel count differs from STREAMINFO"),
        ("frame-depth-property-change", 45, 0x02, "bit depth differs from STREAMINFO"),
        ("frame-number-gap", 46, 0x01, "fixed-block frame number is not sequential"),
        ("frame-block-over-stream-max", 47, 0x20, "block exceeds STREAMINFO maximum"),
        ("frame-rate-property-change", 44, 0x69, "sample rate differs from STREAMINFO"),
    ]
    for name, offset, value, reason in header_edits:
        changed = bytearray(base); changed[offset] = value; repair(changed)
        add(name, changed, reason)

    for name, value, reason in (
        ("subframe-zero-padding-bit", 0x80, "subframe header zero bit is one"),
        ("subframe-reserved-type-2", 0x04, "reserved subframe type 2"),
        ("subframe-reserved-type-7", 0x0E, "reserved subframe type 7"),
        ("subframe-reserved-type-13", 0x1A, "reserved subframe type 13"),
    ):
        changed = bytearray(base); changed[49] = value; repair(changed)
        add(name, changed, reason)

    residual_base = declarative["decl-fixed0-rice4"]
    changed = bytearray(residual_base); changed[50] = (changed[50] & 0x3F) | 0x80
    repair(changed)
    add("residual-method-reserved-2", changed, "reserved residual coding method 2")
    changed = bytearray(residual_base); changed[50] = (changed[50] & 0x3F) | 0xC0
    repair(changed)
    add("residual-method-reserved-3", changed, "reserved residual coding method 3")
    changed = bytearray(residual_base); changed[50] = (changed[50] & 0xC3) | (6 << 2)
    repair(changed)
    add("residual-partition-not-divisible", changed,
        "Rice partition count does not divide block size")

    overlong = bytearray(base)
    overlong[46] = 0xC0
    overlong.insert(47, 0x80)
    repair(overlong, crc_position=49)
    add("coded-number-overlong-repaired", overlong,
        "nonminimal two-byte coded frame number")
    continuation = bytearray(overlong)
    continuation[47] = 0x00
    repair(continuation, crc_position=49)
    add("coded-number-bad-continuation-repaired", continuation,
        "coded frame number continuation byte lacks 10 prefix")

    precision16, _, _ = make_stream(
        [[[1] * 32]], 16, 48000, 0,
        [[{"kind": "lpc", "coefficients": [1], "precision": 16,
           "method": 0, "parameter": 0}]])
    add("lpc-precision-16", precision16, "reserved LPC precision field")
    negative_shift, _, _ = make_stream(
        [[[1] * 32]], 16, 48000, 0,
        [[{"kind": "lpc", "coefficients": [1], "precision": 2, "shift": -1,
           "method": 0, "parameter": 0}]])
    add("lpc-negative-shift", negative_shift,
        "negative LPC shift rejected by the DSPark arithmetic contract")

    pad_stream, _, _ = make_stream(
        [[[-3] * 16]], 4, 48000, 0, [[{"kind": "constant"}]])
    nonzero_pad = bytearray(pad_stream)
    nonzero_pad[-3] |= 1
    repair(nonzero_pad)
    add("subframe-nonzero-byte-padding", nonzero_pad,
        "nonzero frame-alignment padding bit")

    variable = declarative["decl-variable-uncommon-rate"]
    second = variable.find(b"\xff\xf9", frame + 2)
    if second < 0:
        raise RuntimeError("second variable frame not found")
    second_crc = second + 8
    for name, coded, reason in (
        ("variable-sample-overlap", 15, "variable frame overlaps prior coverage"),
        ("variable-sample-gap", 17, "variable frame leaves a coverage gap"),
    ):
        changed = bytearray(variable); changed[second + 4] = coded
        repair(changed, second, second_crc)
        add(name, changed, reason)
    changed = bytearray(variable)
    changed[second + 1] = 0xF8
    changed[second + 4] = 1
    repair(changed, second, second_crc)
    add("mixed-blocking-strategy", changed, "blocking strategy changes between frames")
    changed = bytearray(variable)
    changed[second + 6:second + 8] = (12346).to_bytes(2, "big")
    repair(changed, second, second_crc)
    add("changing-frame-sample-rate", changed, "second frame changes sample rate")

    changed = bytearray(base); changed[12:15] = (255).to_bytes(3, "big")
    add("streaminfo-min-frame-too-large", changed,
        "actual frame is smaller than STREAMINFO minimum")
    changed = bytearray(base); changed[15:18] = (1).to_bytes(3, "big")
    add("streaminfo-max-frame-too-small", changed,
        "actual frame is larger than STREAMINFO maximum")
    changed = bytearray(base)
    changed[12:15] = (100).to_bytes(3, "big")
    changed[15:18] = (10).to_bytes(3, "big")
    add("streaminfo-frame-min-over-max", changed,
        "STREAMINFO minimum frame size exceeds maximum")
    changed = bytearray(base)
    packed = int.from_bytes(changed[18:26], "big")
    packed = (packed & ~(0x1F << 36)) | (2 << 36)
    changed[18:26] = packed.to_bytes(8, "big")
    add("streaminfo-bit-depth-three", changed, "STREAMINFO bit depth is below four")
    changed = bytearray(base)
    packed = int.from_bytes(changed[18:26], "big")
    packed = (packed & ~0xFFFFFFFFF) | ((1 << 31) + 1)
    changed[18:26] = packed.to_bytes(8, "big")
    add("streaminfo-sample-policy-over", changed,
        "declared sample count exceeds the public policy")
    return cases


def compile_tool(tool: Path) -> str:
    flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "flac"], text=True).split()
    command = ["g++", "-std=c++20", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
               str(TOOL_SOURCE), *flags, "-o", str(tool)]
    subprocess.run(command, check=True)
    return subprocess.check_output([str(tool), "--version"], text=True).strip()


def generate(output_root: Path = ROOT) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    tool = output_root / ".libflac_fixture_tool"
    version = compile_tool(tool)
    entries: list[dict[str, Any]] = []

    for name, data in (("rfc9639-d1", RFC_D1), ("rfc9639-d2", RFC_D2),
                       ("rfc9639-d3", RFC_D3)):
        rate, bits, samples = RFC_PCM[name]
        write_file(output_root / f"{name}.flac", data)
        entries.append(fixture_entry(
            name, "RFC 9639 Appendix D exact hexadecimal listing",
            "IETF Trust Legal Provisions / Code Components", rate, bits, samples,
            [{"source": f"RFC 9639 Appendix {name[-2:].upper()}"}],
            output_root, tool))

    for channels in (1, 2):
        for bits in (8, 16, 24, 32):
            name = f"libflac-{channels}ch-{bits}bit"
            limit = 1 << (bits - 1)
            samples = []
            for channel in range(channels):
                values = []
                for index in range(64):
                    raw = ((index * 1103515245 + channel * 12345 + bits * 97) & 0xFFFFFFFF)
                    value = (raw % (2 * limit)) - limit
                    if index == 0: value = -limit
                    if index == 1: value = limit - 1
                    values.append(value)
                samples.append(values)
            source = output_root / f"{name}.source.pcm"
            write_file(source, pcm32_bytes(samples))
            subprocess.run([str(tool), "encode", str(source),
                            str(output_root / f"{name}.flac"),
                            "48000", str(channels), str(bits), "64"], check=True)
            entry = fixture_entry(
                name, "deterministic source PCM encoded by libFLAC 1.4.3",
                "MIT fixture data; libFLAC is BSD-3-Clause", 48000, bits, samples,
                [{"encoder": "libFLAC 1.4.3", "channels": channels,
                  "bits_per_sample": bits}], output_root, tool)
            entry["encoder_command"] = (
                "tests/fixtures/flac/.libflac_fixture_tool encode "
                f"tests/fixtures/flac/{name}.source.pcm "
                f"tests/fixtures/flac/{name}.flac 48000 {channels} {bits} 64")
            entries.append(entry)

    declarative_data: dict[str, bytes] = {}
    for vector in declarative_vectors():
        stream, samples, inventory = make_stream(
            vector["frames"], vector["bits"], vector["rate"], vector["mode"],
            vector["specs"], bool(vector.get("variable", False)),
            vector.get("rate_codes"), bool(vector.get("zero_md5", False)),
            bool(vector.get("unknown_total", False)))
        name = vector["name"]
        declarative_data[name] = stream
        write_file(output_root / f"{name}.flac", stream)
        entries.append(fixture_entry(
            name, "DSPark declarative RFC 9639 field generator",
            "MIT", vector["rate"], vector["bits"], samples, inventory,
            output_root, tool))

    malformed_root = output_root / "malformed"
    malformed_root.mkdir(exist_ok=True)
    for old in malformed_root.glob("*.flac"):
        old.unlink()
    malformed_manifest = []
    malformed_base = declarative_data["decl-verbatim-wasted"]
    for name, data, reason in make_malformed(malformed_base):
        path = malformed_root / f"{name}.flac"
        write_file(path, data)
        malformed_manifest.append({"name": name, "reason": reason,
                                   "sha256": sha(path)})
    for name, data, reason in deep_malformed(declarative_data):
        path = malformed_root / f"{name}.flac"
        write_file(path, data)
        malformed_manifest.append({"name": name, "reason": reason,
                                   "sha256": sha(path)})
    unary_over, _, _ = make_stream(
        [[[-524289] + [0] * 15]], 32, 48000, 0,
        [[{"kind": "fixed", "order": 0, "method": 0, "parameter": 0}]])
    unary_over_path = malformed_root / "rice-unary-cap-over.flac"
    write_file(unary_over_path, unary_over)
    malformed_manifest.append({
        "name": "rice-unary-cap-over",
        "reason": "Rice unary quotient exceeds the public parser policy by one",
        "sha256": sha(unary_over_path),
    })

    measured = {"blocking": set(), "channel_modes": set(),
                "subframes": set(), "residual_modes": set()}
    for entry in entries:
        for frame in entry["actual_inventory"]:
            if "blocking" in frame: measured["blocking"].add(frame["blocking"])
            if "channel_mode" in frame: measured["channel_modes"].add(frame["channel_mode"])
            measured["subframes"].update(frame.get("subframes", []))
            measured["residual_modes"].update(frame.get("residual_modes", []))

    manifest = {
        "schema_version": 1,
        "description": "Deterministic positive and malformed native FLAC fixture corpus",
        "generator_command": GENERATOR_COMMAND,
        "verification_command": VERIFICATION_COMMAND,
        "oracle_compile_command": COMPILE_COMMAND,
        "oracle_version": version,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_source_sha256": sha(TOOL_SOURCE),
        "positive_count": len(entries),
        "malformed_count": len(malformed_manifest),
        "actual_branch_inventory": {
            key: sorted(values) for key, values in measured.items()
        },
        "fixtures": entries,
        "malformed": malformed_manifest,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    tool.unlink(missing_ok=True)
    return manifest


def fixture_hashes(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).digest()
        for path in root.rglob("*")
        if path.is_file()
        and (path.suffix in {".flac", ".pcm"} or path.name == "manifest.json")
    }


def verify() -> None:
    committed = fixture_hashes(ROOT)
    with tempfile.TemporaryDirectory(prefix="dspark-flac-fixtures-") as temporary:
        temporary_root = Path(temporary)
        manifest = generate(temporary_root)
        regenerated = fixture_hashes(temporary_root)
    if committed != regenerated:
        changed = sorted(str(path) for path in committed.keys() | regenerated.keys()
                         if committed.get(path) != regenerated.get(path))
        raise SystemExit("fixture regeneration changed: " + ", ".join(changed))
    print(f"verified {manifest['positive_count']} positive and "
          f"{manifest['malformed_count']} malformed FLAC fixtures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        manifest = generate()
        print(f"generated {manifest['positive_count']} positive and "
              f"{manifest['malformed_count']} malformed FLAC fixtures")


if __name__ == "__main__":
    main()
