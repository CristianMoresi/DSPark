# Contributing to DSPark

DSPark is header-only and dependency-free, so working on the framework itself
needs nothing beyond a C++20 compiler and CMake. Everything below works
identically on Windows, Linux and macOS.

## Prerequisites

| | Minimum |
|---|---|
| Compiler | MSVC 19.50+, GCC 12+, Clang 15+ |
| CMake | 3.21 |

No package manager, no vendored SDK to fetch, no environment to prepare. The
framework has no dependencies; the only third-party code in the repository is
bundled inside `DSParkLab/vendor/` and `plugin/`, and neither is needed to build
or test the framework.

## Build and test

```bash
git clone https://github.com/CristianMoresi/DSPark.git
cd DSPark
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On Windows the Visual Studio generator is multi-config, so name the
configuration on the last two commands:

```bash
cmake --build build --config Release --parallel
ctest --test-dir build --build-config Release --output-on-failure
```

There are also CMake presets:

```bash
cmake --workflow --preset ninja    # configure, build and test in one step
```

That builds and runs four things:

| Test | What it covers |
|---|---|
| `suite` | The test suite proper: 630+ cases across every module |
| `smoke` | Every effect at default settings, checked for NaN/Inf and runaway output |
| `standalone` | Compile gate: the umbrella header alone, every template at float **and** double |
| `conformance` | The public conformance suite, including EBU R128 validation |

Both suites are on by default when DSPark is the top-level project and off when
it is consumed through `add_subdirectory` or `FetchContent`. Use
`-DDSPARK_BUILD_TESTS=OFF` or `-DDSPARK_BUILD_CONFORMANCE=OFF` to opt out.

### Assertions stay live

The test targets are built with optimisations but **without** `NDEBUG`, even in
Release. A number of cases exercise the framework's debug-time contracts
(bounds, ranges, preconditions), so `tests/CMakeLists.txt` strips `NDEBUG` from
the optimised configurations. This is deliberate: the suite is optimised for
speed, not to switch off its own checks.

If you are measuring performance rather than correctness, use `bench/` instead,
which builds the way an application would.

## What CI checks

Every push and pull request runs, across Windows (MSVC x64 and ARM64), Linux
(GCC and Clang, x64 and ARM64), macOS (ARM64) and WebAssembly:

- the test suite and the conformance suite on all six native targets
- AddressSanitizer and UndefinedBehaviorSanitizer over both suites
- the conformance suite a second time on the AVX2/FMA kernel paths, which the
  SSE2 baseline build would never reach
- an exceptions-free, RTTI-free, file-IO-free embedded profile
- the single-header amalgamation, compiled after generation
- EBU R128 conformance against the official Tech 3341/3342 test vectors
- the example plugins in all three formats, through `pluginval`,
  `clap-validator` and Apple's `auval`

A change is ready when all of it is green.

## Conventions

- **English only**, in code, comments, commit messages and documentation.
- **ASCII only** in source files. No em-dashes, no typographic symbols: they
  provoke MSVC C4819 under non-Latin code pages. Write `->` not an arrow,
  `+/-` not a plus-minus sign, `~` not an approximation sign.
- **Include what you use.** MSVC's standard library pulls in far more
  transitively than libstdc++ or libc++ do, so a header that "works" on Windows
  can fail to compile elsewhere. Include `<algorithm>`, `<vector>`, `<cmath>`
  and friends explicitly.
- **No allocation on the audio thread.** Everything is pre-allocated in
  `prepare()`. Anything shared between the control thread and the audio thread
  follows [the threading model](docs/threading.md), which states the contract
  once so each header only has to say which of its methods belongs where.
- **No dependencies.** The C++ standard library only.
- **Comments stand on their own.** A reader has this repository and nothing
  else, so a comment must not cite a reference code -- an issue number, a
  ticket tag, a design or planning document id -- that only resolves somewhere
  they cannot reach. Write the reason out instead; that reason is the valuable
  half, and it is the half a bare code throws away. Pointing at a commit in
  this repository is fine. `python3 tools/check_comment_style.py` checks this
  and runs in CI.

### Tests

Add cases next to the module they cover, using the harness in
`tests/dspark_test.h` (auto-registering, zero dependencies, DSP-specific
assertions such as `EXPECT_SILENT`, `EXPECT_BOUNDED` and `EXPECT_NO_NAN`):

```cpp
DSPARK_TEST(MyEffect_does_the_thing)
{
    MyEffect<float> fx;
    fx.prepare(defaultSpec());
    // ...
    EXPECT_NEAR(measured, expected, tolerance);
}
```

A test that pins a measured DSP property is worth more than one that only
checks the code runs. Where a case exists because of a specific defect, say so
in a comment and give the number the old behaviour produced, so the next reader
knows what the tolerance is protecting.

Any change to a public API must update the tests in the same commit. The suite
is part of the repository precisely so this cannot drift.

## DSParkLab

The interactive testing app is Windows-only: it is built on Win32 and Direct3D
11. Build it with `DSParkLab/build.bat`. The framework, the test suite and the
plugin layer are fully cross-platform; only the Lab is not.

## Licence

By contributing you agree that your contributions are licensed under the MIT
Licence, the same terms that cover the project.
