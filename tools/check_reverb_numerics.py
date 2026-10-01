#!/usr/bin/env python3
"""Compare full reverb fixtures from baseline x64 MSVC, GCC and Clang builds.

Requires Python's standard library and tests/reverb_numeric_fixture.cpp output.
Rejects missing/extra cases, mismatched inputs, truncation and non-finite data.
The tolerance concerns this fixed numerical fixture, not audible quality or a
bit-identity promise across other architectures, SIMD modes or fast-math builds.
"""

from array import array
import argparse
import itertools
import math
from pathlib import Path
import sys


CASES = tuple((kind, rate, mode) for kind in ("float", "double")
              for rate in (44100, 48000, 96000) for mode in (0, 1, 2))


def samples(path, kind, rate):
    values = array("f" if kind == "float" else "d")
    raw = path.read_bytes()
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != 8 * rate or not all(math.isfinite(x) for x in values):
        raise ValueError(f"{path}: expected {8 * rate} finite samples")
    return raw, values


def residual(first, second):
    differences = [float(a) - float(b) for a, b in zip(first, second)]
    return (max(map(abs, differences)),
            math.sqrt(math.fsum(x * x for x in differences) / len(differences)))


def acceptable(kind, peak, rms):
    # Frozen bounds, comfortably rejecting the original ~-92 dBFS float error.
    # Double builds retain ordinary libm roundoff rather than requiring hashes.
    return (peak <= 1e-6 and rms <= 1e-7) if kind == "float" else (
        peak <= 1e-12 and rms <= 1e-13)


def db(value):
    return f"{20 * math.log10(value):.6f}" if value else "-inf"


def compare(directories):
    expected = {f"{kind}-{rate}-{mode}.{suffix}" for kind, rate, mode in CASES
                for suffix in ("input", "output")}
    for directory in directories:
        if {p.name for p in directory.iterdir()} != expected:
            raise ValueError(f"{directory}: incomplete or unexpected fixture inventory")
    passed = True
    for kind, rate, mode in CASES:
        name = f"{kind}-{rate}-{mode}"
        sources, outputs = [], []
        for directory in directories:
            source, _ = samples(directory / (name + ".input"), kind, rate)
            _, output = samples(directory / (name + ".output"), kind, rate)
            sources.append(source)
            outputs.append(output)
        if any(source != sources[0] for source in sources[1:]):
            raise ValueError(f"{name}: input bytes differ")
        for a, b in itertools.combinations(range(len(directories)), 2):
            peak, rms = residual(outputs[a], outputs[b])
            success = acceptable(kind, peak, rms)
            passed &= success
            print(f"{'PASS' if success else 'FAIL'} {name} "
                  f"{directories[a].name}/{directories[b].name}: "
                  f"peak={db(peak)} dBFS rms={db(rms)} dBFS")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        # A narrow peak and a distributed sub-peak error must each fail their
        # own criterion; zero residual is accepted in both sample formats.
        for kind, limit in (("float", 1e-6), ("double", 1e-12)):
            if not acceptable(kind, 0, 0): return 1
            if acceptable(kind, 2 * limit, 0): return 1
            if acceptable(kind, 0.5 * limit, 0.2 * limit): return 1
        print("PASS reverb residual threshold controls")
        return 0
    if len(args.directories) < 2:
        parser.error("at least two fixture directories are required")
    try:
        return 0 if compare(args.directories) else 1
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
