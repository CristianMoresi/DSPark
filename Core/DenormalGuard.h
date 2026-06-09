// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include <cstdint>

// Correct architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define DSPARK_ARCH_X86 1
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DSPARK_ARCH_ARM64 1
    #if defined(_MSC_VER) && !defined(__clang__)
        #include <intrin.h>   // _ReadStatusReg / _WriteStatusReg (MSVC on ARM64)
    #endif
#elif defined(__arm__) || defined(_M_ARM)
    #define DSPARK_ARCH_ARM32 1
#endif

namespace dspark {

/**
 * @class DenormalGuard
 * @brief RAII scope guard to disable denormalised (subnormal) floating-point numbers.
 *
 * Denormals cause severe performance penalties (CPU spikes) in modern processors 
 * during recursive DSP calculations (e.g., IIR filters, delays, reverbs) when signals
 * decay towards zero. This guard configures the CPU to flush denormals to zero (FTZ) 
 * and treat denormal inputs as zero (DAZ) for the lifetime of the object, completely
 * preventing performance degradation without affecting audible sound quality.
 *
 * @note This uses hardware-level intrinsic registers and executes in a few clock cycles.
 * The previous floating-point state is strictly restored upon destruction, ensuring
 * host DAW stability.
 *
 * **Architecture Support:**
 * - x86/x64: Sets FTZ (bit 15) and DAZ (bit 6) via `_mm_setcsr`.
 * - ARM/AArch64 (GCC/Clang): Sets FZ (bit 24) via inline assembly.
 * - WebAssembly / Unknown: Safe no-op.
 *
 * @code
 * void processBlock(float* data, int numSamples) noexcept
 * {
 * dspark::DenormalGuard guard; // Denormals disabled for this scope
 *
 * for (int i = 0; i < numSamples; ++i)
 * data[i] = filter_.processSample(data[i]);
 * } // Original FP state is automatically restored here
 * @endcode
 */
class DenormalGuard
{
public:
    /**
     * @brief Constructs the guard, saving the current FPU state and enabling FTZ/DAZ.
     */
    DenormalGuard() noexcept
    {
#if defined(DSPARK_ARCH_X86)
        previousState_ = _mm_getcsr();
        // 0x8040u = FTZ (bit 15) | DAZ (bit 6)
        _mm_setcsr(static_cast<unsigned int>(previousState_) | 0x8040u);

#elif defined(DSPARK_ARCH_ARM64) && (defined(__GNUC__) || defined(__clang__))
        unsigned long long fpcr;
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(fpcr));
        previousState_ = fpcr;
        fpcr |= (1ULL << 24); // FZ bit
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));

#elif defined(DSPARK_ARCH_ARM64) && defined(_MSC_VER)
        // Windows on ARM (MSVC): no inline asm, use the status-register intrinsics.
        const long long fpcr = _ReadStatusReg(ARM64_FPCR);
        previousState_ = static_cast<uint64_t>(fpcr);
        _WriteStatusReg(ARM64_FPCR, fpcr | (1LL << 24)); // FZ bit

#elif defined(DSPARK_ARCH_ARM32) && (defined(__GNUC__) || defined(__clang__))
        unsigned int fpscr;
        __asm__ __volatile__("vmrs %0, fpscr" : "=r"(fpscr));
        previousState_ = fpscr;
        fpscr |= (1u << 24); // FZ bit
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr));

#else
        // Fallback for WASM, MSVC on ARM (no inline asm support), or unsupported archs.
        previousState_ = 0;
#endif
    }

    /**
     * @brief Destructs the guard, restoring the exact FPU state captured during construction.
     */
    ~DenormalGuard() noexcept
    {
#if defined(DSPARK_ARCH_X86)
        _mm_setcsr(static_cast<unsigned int>(previousState_));

#elif defined(DSPARK_ARCH_ARM64) && (defined(__GNUC__) || defined(__clang__))
        unsigned long long fpcr = static_cast<unsigned long long>(previousState_);
        __asm__ __volatile__("msr fpcr, %0" : : "r"(fpcr));

#elif defined(DSPARK_ARCH_ARM64) && defined(_MSC_VER)
        _WriteStatusReg(ARM64_FPCR, static_cast<long long>(previousState_));

#elif defined(DSPARK_ARCH_ARM32) && (defined(__GNUC__) || defined(__clang__))
        unsigned int fpscr = static_cast<unsigned int>(previousState_);
        __asm__ __volatile__("vmsr fpscr, %0" : : "r"(fpscr));
#endif
    }

    // Delete copy and move semantics strictly
    DenormalGuard(const DenormalGuard&) = delete;
    DenormalGuard& operator=(const DenormalGuard&) = delete;
    DenormalGuard(DenormalGuard&&) = delete;
    DenormalGuard& operator=(DenormalGuard&&) = delete;

private:
    uint64_t previousState_ = 0;
};

} // namespace dspark

#undef DSPARK_ARCH_X86
#undef DSPARK_ARCH_ARM64
#undef DSPARK_ARCH_ARM32