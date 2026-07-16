// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DenormalGuard.h
 * @brief RAII scope guard that disables denormal (subnormal) float arithmetic.
 *
 * Part of the DSPark framework. Standalone: depends only on <cstdint> and the
 * control-register intrinsics of the active architecture.
 */

#include <cstdint>

// Capability gates: a branch is enabled only when the architecture AND the
// toolchain/FPU features it needs are all present; every other target
// compiles as a safe no-op (denormals stay enabled - a performance matter,
// never a correctness one).
//
// - x86/x64 needs SSE for LDMXCSR/STMXCSR. Always present on x64; assumed on
//   32-bit MSVC (every Windows-supported x86 CPU has SSE); on 32-bit
//   GCC/Clang only with -msse (__SSE__) - without it the SSE builtins do not
//   even compile.
// - AArch64 GCC/Clang (including clang-cl) uses mrs/msr inline assembly.
//   MSVC has no inline assembly on ARM64 and uses the status-register
//   intrinsics instead.
// - 32-bit ARM vmrs/vmsr are VFP instructions: only available with a
//   hardware FPU (__ARM_FP, per ACLE). Soft-float targets (e.g. Cortex-M0/M3)
//   get the no-op.
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) \
    || (defined(__i386__) && defined(__SSE__))
    #define DSPARK_DENORMAL_X86 1
    #include <immintrin.h>
#elif (defined(__aarch64__) || defined(_M_ARM64)) && (defined(__GNUC__) || defined(__clang__))
    #define DSPARK_DENORMAL_ARM64_ASM 1
#elif defined(_M_ARM64) && defined(_MSC_VER)
    #define DSPARK_DENORMAL_ARM64_MSVC 1
    #include <intrin.h>   // _ReadStatusReg / _WriteStatusReg
#elif defined(__arm__) && defined(__ARM_FP) && (defined(__GNUC__) || defined(__clang__))
    #define DSPARK_DENORMAL_ARM32 1
#endif

namespace dspark {

/**
 * @class DenormalGuard
 * @brief RAII scope guard to disable denormalised (subnormal) floating-point numbers.
 *
 * Denormals cause severe performance penalties (CPU spikes) in recursive DSP
 * calculations (IIR filters, delays, reverbs) when signals decay towards zero.
 * This guard configures the CPU to flush denormal results to zero (FTZ) and,
 * on x86, to also treat denormal inputs as zero (DAZ) for the lifetime of the
 * object, preventing the degradation without audible effect on sound quality.
 *
 * The floating-point mode is a per-thread CPU register: construct the guard
 * on the thread whose maths it must protect (typically at the top of the
 * audio callback), and give it a name - a discarded temporary is destroyed
 * immediately and protects nothing (the class is marked [[nodiscard]] so
 * compilers flag that mistake). Guards nest safely: each one restores the
 * exact state it captured, so the host's own floating-point configuration
 * survives. Construction and destruction cost a handful of cycles.
 *
 * **Architecture support:**
 * - x86/x64: FTZ (bit 15) + DAZ (bit 6) of MXCSR via `_mm_getcsr`/`_mm_setcsr`.
 *   32-bit GCC/Clang builds need `-msse`; without it the guard is a no-op.
 * - AArch64 (GCC/Clang): FZ (bit 24) of FPCR via `mrs`/`msr` inline assembly.
 * - AArch64 (MSVC): the same FZ bit via `_ReadStatusReg`/`_WriteStatusReg`.
 * - 32-bit ARM (GCC/Clang, hardware FP): FZ (bit 24) of FPSCR via `vmrs`/`vmsr`.
 * - WebAssembly and any other target: safe no-op (Wasm has no FTZ mode).
 *
 * Use isActive() to ask at compile time whether this build has a real
 * implementation or the no-op.
 *
 * @code
 * void processBlock(float* data, int numSamples) noexcept
 * {
 *     dspark::DenormalGuard guard; // denormals disabled for this scope
 *
 *     for (int i = 0; i < numSamples; ++i)
 *         data[i] = filter_.processSample(data[i]);
 * } // previous FP state restored here
 * @endcode
 */
class [[nodiscard]] DenormalGuard
{
public:
    /**
     * @brief Constructs the guard, saving the current FP state and enabling FTZ (plus DAZ on x86).
     */
    DenormalGuard() noexcept
    {
#if defined(DSPARK_DENORMAL_X86)
        previousState_ = _mm_getcsr();
        // 0x8040u = FTZ (bit 15) | DAZ (bit 6)
        _mm_setcsr(static_cast<unsigned int>(previousState_) | 0x8040u);

#elif defined(DSPARK_DENORMAL_ARM64_ASM)
        unsigned long long fpcr;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        previousState_ = fpcr;
        fpcr |= (1ULL << 24); // FZ bit
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));

#elif defined(DSPARK_DENORMAL_ARM64_MSVC)
        const long long fpcr = _ReadStatusReg(ARM64_FPCR);
        previousState_ = static_cast<uint64_t>(fpcr);
        _WriteStatusReg(ARM64_FPCR, fpcr | (1LL << 24)); // FZ bit

#elif defined(DSPARK_DENORMAL_ARM32)
        unsigned int fpscr;
        __asm__ __volatile__("vmrs %0, fpscr" : "=r"(fpscr));
        previousState_ = fpscr;
        fpscr |= (1u << 24); // FZ bit
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr));

#else
        // No-op fallback: WebAssembly (no FTZ mode exists), 32-bit MSVC ARM,
        // x86/ARM builds without the required SSE/FPU features, and any
        // architecture not listed above.
        previousState_ = 0;
#endif
    }

    /**
     * @brief Destroys the guard, restoring the exact FP state captured at construction.
     */
    ~DenormalGuard() noexcept
    {
#if defined(DSPARK_DENORMAL_X86)
        _mm_setcsr(static_cast<unsigned int>(previousState_));

#elif defined(DSPARK_DENORMAL_ARM64_ASM)
        const unsigned long long fpcr = static_cast<unsigned long long>(previousState_);
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));

#elif defined(DSPARK_DENORMAL_ARM64_MSVC)
        _WriteStatusReg(ARM64_FPCR, static_cast<long long>(previousState_));

#elif defined(DSPARK_DENORMAL_ARM32)
        const unsigned int fpscr = static_cast<unsigned int>(previousState_);
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr));
#endif
    }

    /**
     * @brief True when this build has a real flush-to-zero implementation.
     *
     * False where the guard compiles as a documented no-op (WebAssembly,
     * soft-float ARM, 32-bit x86 without SSE, unknown architectures). Lets
     * tests and diagnostics assert flushing only where the hardware can do it.
     */
    [[nodiscard]] static constexpr bool isActive() noexcept
    {
#if defined(DSPARK_DENORMAL_X86) || defined(DSPARK_DENORMAL_ARM64_ASM) \
    || defined(DSPARK_DENORMAL_ARM64_MSVC) || defined(DSPARK_DENORMAL_ARM32)
        return true;
#else
        return false;
#endif
    }

    // A scope guard is tied to its constructing scope and thread: neither
    // copyable nor movable.
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;
    DenormalGuard(DenormalGuard&&) = delete;
    DenormalGuard& operator=(DenormalGuard&&) = delete;

private:
    uint64_t previousState_ = 0;
};

} // namespace dspark

#undef DSPARK_DENORMAL_X86
#undef DSPARK_DENORMAL_ARM64_ASM
#undef DSPARK_DENORMAL_ARM64_MSVC
#undef DSPARK_DENORMAL_ARM32
