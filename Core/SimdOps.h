// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file SimdOps.h
 * @brief SIMD-accelerated buffer operations for real-time audio processing.
 *
 * Kernels use FMA (fused multiply-add) where the target enables it, and
 * restrict-qualified pointers so compilers can unroll aggressively.
 *
 * Build flags: an x86-64 build for the generic baseline also carries AVX2+FMA
 * copies of the kernels that gain from them (the Vec abstraction, firCorrelate,
 * dotProduct, complexMulAccum and the FFT passes) and selects them at run time
 * on CPUs that support them (see setRuntimeDispatch()). /arch:AVX2 (MSVC) or
 * -march=x86-64-v3 (GCC/Clang) still widens the rest of the code when you
 * control the deployment target. WebAssembly: compile with -msimd128 -msse2 so
 * Emscripten lowers the SSE2 paths to Wasm SIMD128; otherwise the scalar
 * fallbacks are used.
 *
 * Measured on MSVC x64 (SSE2 baseline -> AVX2 copies): FFTReal 2048 1.5x,
 * 4x oversampling 1.5x, uniform convolution 1.8-2.2x faster.
 */

// --- Platform SIMD detection ------------------------------------------------
#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__) \
    || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || (defined(__i386__) && defined(__SSE2__))
    #define DSPARK_SIMD_SSE2 1
    #include <emmintrin.h>           // SSE2
    #if defined(__AVX__)
        #define DSPARK_SIMD_AVX 1
        #include <immintrin.h>       // AVX
    #endif
    // Detect FMA3 support. __FMA__ is the authoritative flag on GCC/Clang:
    // AVX2 alone does not licence FMA codegen there, and calling _mm*_fmadd_*
    // without the fma target feature is a hard compile error (e.g. building
    // with -mavx2 but not -mfma). MSVC never defines __FMA__ but permits FMA
    // intrinsics under /arch:AVX2; clang-cl defines _MSC_VER yet enforces
    // target features like clang, so it must take the __FMA__ route only.
    #if defined(__FMA__) || (defined(__AVX2__) && defined(_MSC_VER) && !defined(__clang__))
        #define DSPARK_SIMD_FMA 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DSPARK_SIMD_NEON 1
    #include <arm_neon.h>
#elif defined(__EMSCRIPTEN__) && defined(__wasm_simd128__) && defined(__SSE2__)
    // Emscripten lowers the SSE2 intrinsic set to Wasm SIMD128 when compiled
    // with -msimd128 -msse2 (both flags required). Anything less stays scalar.
    #define DSPARK_SIMD_SSE2 1
    #include <emmintrin.h>
#endif

// --- Run-time AVX2 dispatch -------------------------------------------------
// x86-64 builds below AVX compile a second copy of the wide kernels inside an
// AVX2+FMA target region and pick it at run time after a CPU check. Define
// DSPARK_NO_RUNTIME_DISPATCH to compile the baseline paths only. clang-cl is
// left out: its target-attribute and CPU-query support differ from both
// clang and MSVC.
#if defined(DSPARK_SIMD_SSE2) && !defined(DSPARK_SIMD_AVX) && !defined(DSPARK_NO_RUNTIME_DISPATCH) \
    && (defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__))     \
    && !(defined(_MSC_VER) && defined(__clang__))
    #define DSPARK_SIMD_DISPATCH_AVX2 1
    #include <immintrin.h>
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
    #endif
    #if defined(__clang__)
        #define DSPARK_SIMD_AVX2_BEGIN \
            _Pragma("clang attribute push (__attribute__((target(\"avx2,fma\"))), apply_to = function)")
        #define DSPARK_SIMD_AVX2_END _Pragma("clang attribute pop")
    #elif defined(__GNUC__)
        #define DSPARK_SIMD_AVX2_BEGIN _Pragma("GCC push_options") _Pragma("GCC target(\"avx2,fma\")")
        #define DSPARK_SIMD_AVX2_END _Pragma("GCC pop_options")
    #else
        // MSVC accepts AVX2/FMA intrinsics in any function.
        #define DSPARK_SIMD_AVX2_BEGIN
        #define DSPARK_SIMD_AVX2_END
    #endif
#endif

// --- AVX2 without FMA --------------------------------------------------------
// For kernels that must round exactly as the baseline code does (the block
// forms of a scalar reader, bit for bit): a region that enables AVX2 only, so
// the compiler has no fused multiply-add to contract the arithmetic into.
// DSPARK_SIMD_EXACT_AVX2 is defined when such kernels can be built: always in
// AVX2 builds, and in x86-64 baseline builds that dispatch at run time
// (DSPARK_SIMD_EXACT_AVX2_DISPATCHED), where callers check
// simd::isRuntimeDispatchActive() before using them.
#if defined(DSPARK_SIMD_SSE2) && defined(__AVX2__)
    #define DSPARK_SIMD_EXACT_AVX2 1
    #define DSPARK_SIMD_EXACT_AVX2_BEGIN
    #define DSPARK_SIMD_EXACT_AVX2_END
#elif defined(DSPARK_SIMD_DISPATCH_AVX2)
    #define DSPARK_SIMD_EXACT_AVX2 1
    #define DSPARK_SIMD_EXACT_AVX2_DISPATCHED 1
    #if defined(__clang__)
        #define DSPARK_SIMD_EXACT_AVX2_BEGIN \
            _Pragma("clang attribute push (__attribute__((target(\"avx2\"))), apply_to = function)")
        #define DSPARK_SIMD_EXACT_AVX2_END _Pragma("clang attribute pop")
    #elif defined(__GNUC__)
        #define DSPARK_SIMD_EXACT_AVX2_BEGIN _Pragma("GCC push_options") _Pragma("GCC target(\"avx2\")")
        #define DSPARK_SIMD_EXACT_AVX2_END _Pragma("GCC pop_options")
    #else
        // MSVC accepts AVX2 intrinsics in any function and never contracts.
        #define DSPARK_SIMD_EXACT_AVX2_BEGIN
        #define DSPARK_SIMD_EXACT_AVX2_END
    #endif
#endif

// --- Restrict Macro for Pointer Aliasing Optimization -----------------------
// NOTE: restrict constrains WRITTEN ranges only. A range written through one
// DSPARK_RESTRICT pointer must not overlap any other range accessed in the
// same call (so dst != src, no partial overlaps). Read-only pointers may
// alias each other freely: dotProduct(data, data, n) is well defined, and is
// exactly what sumOfSquares() does. Defined idempotently (#ifndef) because a
// few headers historically declared the same macro.
#ifndef DSPARK_RESTRICT
  #if defined(_MSC_VER)
    #define DSPARK_RESTRICT __restrict
  #elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_RESTRICT __restrict__
  #else
    #define DSPARK_RESTRICT
  #endif
#endif

#include <atomic>
#include <cstdint>
#include <type_traits>

// --- Vec<T, W> and the kernels written on it --------------------------------
// detail/SimdVec.inl defines Vec<T, W> (W lanes of T; W = 1 is scalar code),
// kVecWidth / kVecNarrowWidth and the wide kernels, for the feature set named
// by the DSPARK_VEC_* macros. The baseline copy lives in dspark::simd and is
// what FFT.h, Oversampling.h and the effects use directly.
#if defined(DSPARK_SIMD_SSE2)
    #define DSPARK_VEC_SSE2 1
#endif
#if defined(DSPARK_SIMD_NEON)
    #define DSPARK_VEC_NEON 1
#endif
#if defined(DSPARK_SIMD_AVX)
    #define DSPARK_VEC_AVX 1
#endif
#if defined(DSPARK_SIMD_FMA)
    #define DSPARK_VEC_FMA 1
#endif
namespace dspark {
namespace simd {
#include "detail/SimdVec.inl"
} // namespace simd
} // namespace dspark
#undef DSPARK_VEC_SSE2
#undef DSPARK_VEC_NEON
#undef DSPARK_VEC_AVX
#undef DSPARK_VEC_FMA

#if defined(DSPARK_SIMD_DISPATCH_AVX2)
// The AVX2+FMA copy, compiled for that target and called only after the check.
DSPARK_SIMD_AVX2_BEGIN
#define DSPARK_VEC_SSE2 1
#define DSPARK_VEC_AVX 1
#define DSPARK_VEC_FMA 1
namespace dspark {
namespace simd {
namespace avx2 {
#include "detail/SimdVec.inl"
} // namespace avx2
} // namespace simd
} // namespace dspark
#undef DSPARK_VEC_SSE2
#undef DSPARK_VEC_AVX
#undef DSPARK_VEC_FMA
DSPARK_SIMD_AVX2_END

namespace dspark {
namespace simd {
namespace dispatch {

/// True when the CPU executes AVX2 and FMA and the OS saves the YMM state.
inline bool cpuSupportsAvx2Fma() noexcept
{
#if defined(_MSC_VER)
    int r[4] = {};
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const unsigned ecx = static_cast<unsigned>(r[2]);
    const bool fma = (ecx & (1u << 12)) != 0, osxsave = (ecx & (1u << 27)) != 0,
               avx = (ecx & (1u << 28)) != 0;
    if (!fma || !osxsave || !avx) return false;
    if ((_xgetbv(0) & 6u) != 6u) return false;   // XMM and YMM state enabled
    __cpuidex(r, 7, 0);
    return (static_cast<unsigned>(r[1]) & (1u << 5)) != 0;   // AVX2
#else
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool fma = (c & (1u << 12)) != 0, osxsave = (c & (1u << 27)) != 0,
               avx = (c & (1u << 28)) != 0;
    if (!fma || !osxsave || !avx) return false;
    unsigned xcr0 = 0, xcr0High = 0;
    // xgetbv, as bytes: the mnemonic needs the xsave target feature in clang.
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr0), "=d"(xcr0High) : "c"(0));
    (void)xcr0High;
    if ((xcr0 & 6u) != 6u) return false;
    __cpuid_count(7, 0, a, b, c, d);
    return (b & (1u << 5)) != 0;   // AVX2
#endif
}

/// Whether the AVX2 copies are in use. Zero-initialised (false) until the
/// dynamic initialiser runs, so code that runs earlier takes the baseline.
inline std::atomic<bool> avx2Active { cpuSupportsAvx2Fma() };

} // namespace dispatch
} // namespace simd
} // namespace dspark
#endif

namespace dspark {
namespace simd {

/**
 * @brief Whether this build carries AVX2+FMA kernels and the CPU can run them.
 *
 * True only in x86-64 builds below AVX (see DSPARK_NO_RUNTIME_DISPATCH) on a
 * CPU with AVX2, FMA and OS support for the YMM state. Builds that target AVX2
 * at compile time use those kernels everywhere and return false here.
 */
[[nodiscard]] inline bool runtimeDispatchSupported() noexcept
{
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
    return dispatch::cpuSupportsAvx2Fma();
#else
    return false;
#endif
}

/** @brief Whether the AVX2+FMA kernels are selected now (see setRuntimeDispatch()). */
[[nodiscard]] inline bool isRuntimeDispatchActive() noexcept
{
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
    return dispatch::avx2Active.load(std::memory_order_relaxed);
#else
    return false;
#endif
}

/**
 * @brief Selects or deselects the AVX2+FMA kernels (on by default where
 *        runtimeDispatchSupported()).
 *
 * The two kernel sets round differently (FMA, wider sums), so output can differ
 * between machines in the last bits. Disable it before processing when renders
 * must be bit-identical on every CPU. Enabling it on a CPU without AVX2 and FMA
 * has no effect. Thread-safe; calls in progress finish on the set they began.
 */
inline void setRuntimeDispatch(bool enabled) noexcept
{
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
    dispatch::avx2Active.store(enabled && dispatch::cpuSupportsAvx2Fma(), std::memory_order_relaxed);
#else
    (void)enabled;
#endif
}

} // namespace simd
} // namespace dspark

namespace dspark {
namespace simd {

// ============================================================================
// addWithGain -- dst[i] += src[i] * gain
// ============================================================================

/**
 * @brief Adds source samples scaled by a gain factor into a destination buffer.
 */
inline void addWithGain(float* DSPARK_RESTRICT dst, const float* DSPARK_RESTRICT src, float gain, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256 vGain = _mm256_set1_ps(gain);
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        __m256 vDst = _mm256_loadu_ps(dst + i);
        __m256 vSrc = _mm256_loadu_ps(src + i);
        
        #if defined(DSPARK_SIMD_FMA)
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(vSrc, vGain, vDst));
        #else
            _mm256_storeu_ps(dst + i, _mm256_add_ps(vDst, _mm256_mul_ps(vSrc, vGain)));
        #endif
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;

#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vGain = _mm_set1_ps(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        __m128 vDst = _mm_loadu_ps(dst + i);
        __m128 vSrc = _mm_loadu_ps(src + i);
        
        #if defined(DSPARK_SIMD_FMA)
            _mm_storeu_ps(dst + i, _mm_fmadd_ps(vSrc, vGain, vDst));
        #else
            _mm_storeu_ps(dst + i, _mm_add_ps(vDst, _mm_mul_ps(vSrc, vGain)));
        #endif
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;

#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vGain = vdupq_n_f32(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        float32x4_t vDst = vld1q_f32(dst + i);
        float32x4_t vSrc = vld1q_f32(src + i);
        // vfmaq guarantees a fused FMLA; vmlaq has separate mul+add semantics
        // in ACLE and GCC lowers it to fmul+fadd.
        vst1q_f32(dst + i, vfmaq_f32(vDst, vSrc, vGain));
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;

#else
    for (int i = 0; i < count; ++i) dst[i] += src[i] * gain;
#endif
}

/** @brief Double overload */
inline void addWithGain(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT src, double gain, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256d vGain = _mm256_set1_pd(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        __m256d vDst = _mm256_loadu_pd(dst + i);
        __m256d vSrc = _mm256_loadu_pd(src + i);

        #if defined(DSPARK_SIMD_FMA)
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd(vSrc, vGain, vDst));
        #else
            _mm256_storeu_pd(dst + i, _mm256_add_pd(vDst, _mm256_mul_pd(vSrc, vGain)));
        #endif
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;

#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vGain = _mm_set1_pd(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d vDst = _mm_loadu_pd(dst + i);
        __m128d vSrc = _mm_loadu_pd(src + i);
        
        #if defined(DSPARK_SIMD_FMA)
            _mm_storeu_pd(dst + i, _mm_fmadd_pd(vSrc, vGain, vDst));
        #else
            _mm_storeu_pd(dst + i, _mm_add_pd(vDst, _mm_mul_pd(vSrc, vGain)));
        #endif
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;

#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vGain = vdupq_n_f64(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        float64x2_t vDst = vld1q_f64(dst + i);
        float64x2_t vSrc = vld1q_f64(src + i);
        vst1q_f64(dst + i, vfmaq_f64(vDst, vSrc, vGain));
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;
#else
    for (int i = 0; i < count; ++i) dst[i] += src[i] * gain;
#endif
}

// ============================================================================
// applyGain -- data[i] *= gain
// ============================================================================

/**
 * @brief Multiplies all samples in a buffer by a gain factor.
 */
inline void applyGain(float* DSPARK_RESTRICT data, float gain, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256 vGain = _mm256_set1_ps(gain);
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, _mm256_mul_ps(v, vGain));
    }
    for (; i < count; ++i) data[i] *= gain;

#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vGain = _mm_set1_ps(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        __m128 v = _mm_loadu_ps(data + i);
        _mm_storeu_ps(data + i, _mm_mul_ps(v, vGain));
    }
    for (; i < count; ++i) data[i] *= gain;

#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vGain = vdupq_n_f32(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        vst1q_f32(data + i, vmulq_f32(v, vGain));
    }
    for (; i < count; ++i) data[i] *= gain;

#else
    for (int i = 0; i < count; ++i) data[i] *= gain;
#endif
}

/** @brief Double overload. */
inline void applyGain(double* DSPARK_RESTRICT data, double gain, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256d vGain = _mm256_set1_pd(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
        _mm256_storeu_pd(data + i, _mm256_mul_pd(_mm256_loadu_pd(data + i), vGain));
    for (; i < count; ++i) data[i] *= gain;

#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vGain = _mm_set1_pd(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d v = _mm_loadu_pd(data + i);
        _mm_storeu_pd(data + i, _mm_mul_pd(v, vGain));
    }
    for (; i < count; ++i) data[i] *= gain;

#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vGain = vdupq_n_f64(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
        vst1q_f64(data + i, vmulq_f64(vld1q_f64(data + i), vGain));
    for (; i < count; ++i) data[i] *= gain;
#else
    for (int i = 0; i < count; ++i) data[i] *= gain;
#endif
}

// ============================================================================
// peakLevel -- max(abs(data[i]))
// ============================================================================

/**
 * @brief Returns the peak absolute sample value in a buffer.
 *
 * NaN samples are ignored on every code path (SIMD and scalar): the result is
 * the peak of the finite samples, 0 if there are none.
 */
inline float peakLevel(const float* DSPARK_RESTRICT data, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256 absMask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 vMax = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_and_ps(v, absMask);
        // Data operand first: MAXPS returns the SECOND operand on NaN, so a
        // NaN sample is skipped and the accumulator keeps its history
        // (matching the scalar tail, whose comparison also skips NaNs).
        vMax = _mm256_max_ps(v, vMax);
    }
    __m128 hi = _mm256_extractf128_ps(vMax, 1);
    __m128 lo = _mm256_castps256_ps128(vMax);
    __m128 m4 = _mm_max_ps(lo, hi);
    __m128 m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    __m128 m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    float peak = _mm_cvtss_f32(m1);
    
    for (; i < count; ++i)
    {
        float a = data[i] < 0.0f ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;

#elif defined(DSPARK_SIMD_SSE2)
    const __m128 absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    __m128 vMax = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        __m128 v = _mm_loadu_ps(data + i);
        v = _mm_and_ps(v, absMask);
        vMax = _mm_max_ps(v, vMax); // data first: NaN skipped (see AVX note)
    }
    __m128 shuf = _mm_movehl_ps(vMax, vMax);
    __m128 maxPair = _mm_max_ps(vMax, shuf);
    __m128 maxSingle = _mm_max_ss(maxPair, _mm_shuffle_ps(maxPair, maxPair, 1));
    float peak = _mm_cvtss_f32(maxSingle);
    
    for (; i < count; ++i)
    {
        float a = data[i] < 0.0f ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;

#elif defined(DSPARK_SIMD_NEON)
    float32x4_t vMax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        float32x4_t v = vld1q_f32(data + i);
        v = vabsq_f32(v);
        vMax = vmaxnmq_f32(vMax, v); // FMAXNM: NaN operands are skipped
    }
    float peak = vmaxvq_f32(vMax);
    
    for (; i < count; ++i)
    {
        float a = data[i] < 0.0f ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;

#else
    float peak = 0.0f;
    for (int i = 0; i < count; ++i)
    {
        float a = data[i] < 0.0f ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;
#endif
}

/** @brief Double overload. NaN samples are ignored, as in the float version. */
inline double peakLevel(const double* DSPARK_RESTRICT data, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256d absMask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFll));
    __m256d vMax = _mm256_setzero_pd();
    int i = 0;
    for (; i + 3 < count; i += 4)
        vMax = _mm256_max_pd(_mm256_and_pd(_mm256_loadu_pd(data + i), absMask), vMax);
    __m128d lo = _mm256_castpd256_pd128(vMax);
    __m128d hi = _mm256_extractf128_pd(vMax, 1);
    __m128d m2 = _mm_max_pd(lo, hi);
    double peak = _mm_cvtsd_f64(_mm_max_sd(m2, _mm_unpackhi_pd(m2, m2)));

    for (; i < count; ++i)
    {
        double a = data[i] < 0.0 ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;

#elif defined(DSPARK_SIMD_SSE2)
    const __m128d absMask = _mm_castsi128_pd(_mm_set_epi64x(
        static_cast<int64_t>(0x7FFFFFFFFFFFFFFF),
        static_cast<int64_t>(0x7FFFFFFFFFFFFFFF)));
    __m128d vMax = _mm_setzero_pd();
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d v = _mm_loadu_pd(data + i);
        v = _mm_and_pd(v, absMask);
        vMax = _mm_max_pd(v, vMax); // data first: NaN skipped
    }
    __m128d hi = _mm_unpackhi_pd(vMax, vMax);
    __m128d maxVal = _mm_max_sd(vMax, hi);
    double peak = _mm_cvtsd_f64(maxVal);

    for (; i < count; ++i)
    {
        double a = data[i] < 0.0 ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;

#elif defined(DSPARK_SIMD_NEON)
    float64x2_t vMax = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 1 < count; i += 2)
        vMax = vmaxnmq_f64(vMax, vabsq_f64(vld1q_f64(data + i)));
    double peak = vmaxvq_f64(vMax);

    for (; i < count; ++i)
    {
        double a = data[i] < 0.0 ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;
#else
    double peak = 0.0;
    for (int i = 0; i < count; ++i)
    {
        double a = data[i] < 0.0 ? -data[i] : data[i];
        if (a > peak) peak = a;
    }
    return peak;
#endif
}

// ============================================================================
// dotProduct -- sum(a[i] * b[i])
// ============================================================================

/**
 * @brief Computes the dot product of two arrays.
 */
inline float dotProduct(const float* DSPARK_RESTRICT a, const float* DSPARK_RESTRICT b, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    return dotProductWide(a, b, count);
#elif defined(DSPARK_SIMD_SSE2)
  #if defined(DSPARK_SIMD_DISPATCH_AVX2)
    if (isRuntimeDispatchActive()) return avx2::dotProductWide(a, b, count);
  #endif
    // Two accumulators hide the add/FMA latency on SSE-class hardware.
    __m128 vSum0 = _mm_setzero_ps();
    __m128 vSum1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm_fmadd_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i),     vSum0);
            vSum1 = _mm_fmadd_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4), vSum1);
        #else
            vSum0 = _mm_add_ps(vSum0, _mm_mul_ps(_mm_loadu_ps(a + i),     _mm_loadu_ps(b + i)));
            vSum1 = _mm_add_ps(vSum1, _mm_mul_ps(_mm_loadu_ps(a + i + 4), _mm_loadu_ps(b + i + 4)));
        #endif
    }
    for (; i + 3 < count; i += 4)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm_fmadd_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i), vSum0);
        #else
            vSum0 = _mm_add_ps(vSum0, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        #endif
    }
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, _mm_add_ps(vSum0, vSum1));
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;

#elif defined(DSPARK_SIMD_NEON)
    float32x4_t vSum0 = vdupq_n_f32(0.0f);
    float32x4_t vSum1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        vSum0 = vfmaq_f32(vSum0, vld1q_f32(a + i),     vld1q_f32(b + i));
        vSum1 = vfmaq_f32(vSum1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    for (; i + 3 < count; i += 4)
        vSum0 = vfmaq_f32(vSum0, vld1q_f32(a + i), vld1q_f32(b + i));
    float sum = vaddvq_f32(vaddq_f32(vSum0, vSum1));

    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;

#else
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

/** @brief Double overload. */
inline double dotProduct(const double* DSPARK_RESTRICT a, const double* DSPARK_RESTRICT b, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    return dotProductWide(a, b, count);
#elif defined(DSPARK_SIMD_SSE2)
  #if defined(DSPARK_SIMD_DISPATCH_AVX2)
    if (isRuntimeDispatchActive()) return avx2::dotProductWide(a, b, count);
  #endif
    __m128d vSum0 = _mm_setzero_pd();
    __m128d vSum1 = _mm_setzero_pd();
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm_fmadd_pd(_mm_loadu_pd(a + i),     _mm_loadu_pd(b + i),     vSum0);
            vSum1 = _mm_fmadd_pd(_mm_loadu_pd(a + i + 2), _mm_loadu_pd(b + i + 2), vSum1);
        #else
            vSum0 = _mm_add_pd(vSum0, _mm_mul_pd(_mm_loadu_pd(a + i),     _mm_loadu_pd(b + i)));
            vSum1 = _mm_add_pd(vSum1, _mm_mul_pd(_mm_loadu_pd(a + i + 2), _mm_loadu_pd(b + i + 2)));
        #endif
    }
    for (; i + 1 < count; i += 2)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm_fmadd_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i), vSum0);
        #else
            vSum0 = _mm_add_pd(vSum0, _mm_mul_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i)));
        #endif
    }
    __m128d vSum = _mm_add_pd(vSum0, vSum1);
    __m128d hi = _mm_unpackhi_pd(vSum, vSum);
    double sum = _mm_cvtsd_f64(_mm_add_sd(vSum, hi));

    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;

#elif defined(DSPARK_SIMD_NEON)
    float64x2_t vSum0 = vdupq_n_f64(0.0);
    float64x2_t vSum1 = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        vSum0 = vfmaq_f64(vSum0, vld1q_f64(a + i),     vld1q_f64(b + i));
        vSum1 = vfmaq_f64(vSum1, vld1q_f64(a + i + 2), vld1q_f64(b + i + 2));
    }
    for (; i + 1 < count; i += 2)
        vSum0 = vfmaq_f64(vSum0, vld1q_f64(a + i), vld1q_f64(b + i));
    double sum = vaddvq_f64(vaddq_f64(vSum0, vSum1));

    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;
#else
    double sum = 0.0;
    for (int i = 0; i < count; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

// ============================================================================
// add -- dst[i] += src[i]
// ============================================================================

/**
 * @brief Adds source samples into a destination buffer (no scaling).
 */
inline void add(float* DSPARK_RESTRICT dst, const float* DSPARK_RESTRICT src, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        __m256 vDst = _mm256_loadu_ps(dst + i);
        __m256 vSrc = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, _mm256_add_ps(vDst, vSrc));
    }
    for (; i < count; ++i) dst[i] += src[i];

#elif defined(DSPARK_SIMD_SSE2)
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        __m128 vDst = _mm_loadu_ps(dst + i);
        __m128 vSrc = _mm_loadu_ps(src + i);
        _mm_storeu_ps(dst + i, _mm_add_ps(vDst, vSrc));
    }
    for (; i < count; ++i) dst[i] += src[i];

#elif defined(DSPARK_SIMD_NEON)
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        float32x4_t vDst = vld1q_f32(dst + i);
        float32x4_t vSrc = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(vDst, vSrc));
    }
    for (; i < count; ++i) dst[i] += src[i];

#else
    for (int i = 0; i < count; ++i) dst[i] += src[i];
#endif
}

/** @brief Double overload. */
inline void add(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT src, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    int i = 0;
    for (; i + 3 < count; i += 4)
        _mm256_storeu_pd(dst + i, _mm256_add_pd(_mm256_loadu_pd(dst + i), _mm256_loadu_pd(src + i)));
    for (; i < count; ++i) dst[i] += src[i];

#elif defined(DSPARK_SIMD_SSE2)
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d vDst = _mm_loadu_pd(dst + i);
        __m128d vSrc = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, _mm_add_pd(vDst, vSrc));
    }
    for (; i < count; ++i) dst[i] += src[i];

#elif defined(DSPARK_SIMD_NEON)
    int i = 0;
    for (; i + 1 < count; i += 2)
        vst1q_f64(dst + i, vaddq_f64(vld1q_f64(dst + i), vld1q_f64(src + i)));
    for (; i < count; ++i) dst[i] += src[i];
#else
    for (int i = 0; i < count; ++i) dst[i] += src[i];
#endif
}

// ============================================================================
// multiply -- dst[i] = a[i] * b[i]
// ============================================================================

/** @brief Element-wise product of two buffers into a destination. */
inline void multiply(float* DSPARK_RESTRICT dst, const float* DSPARK_RESTRICT a, const float* DSPARK_RESTRICT b, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    int i = 0;
    for (; i + 7 < count; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    for (; i < count; ++i) dst[i] = a[i] * b[i];
#elif defined(DSPARK_SIMD_SSE2)
    int i = 0;
    for (; i + 3 < count; i += 4)
        _mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
    for (; i < count; ++i) dst[i] = a[i] * b[i];
#elif defined(DSPARK_SIMD_NEON)
    int i = 0;
    for (; i + 3 < count; i += 4)
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    for (; i < count; ++i) dst[i] = a[i] * b[i];
#else
    for (int i = 0; i < count; ++i) dst[i] = a[i] * b[i];
#endif
}

/** @brief Double overload. */
inline void multiply(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT a, const double* DSPARK_RESTRICT b, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    int i = 0;
    for (; i + 3 < count; i += 4)
        _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
    for (; i < count; ++i) dst[i] = a[i] * b[i];
#elif defined(DSPARK_SIMD_SSE2)
    int i = 0;
    for (; i + 1 < count; i += 2)
        _mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(a + i), _mm_loadu_pd(b + i)));
    for (; i < count; ++i) dst[i] = a[i] * b[i];

#elif defined(DSPARK_SIMD_NEON)
    int i = 0;
    for (; i + 1 < count; i += 2)
        vst1q_f64(dst + i, vmulq_f64(vld1q_f64(a + i), vld1q_f64(b + i)));
    for (; i < count; ++i) dst[i] = a[i] * b[i];
#else
    for (int i = 0; i < count; ++i) dst[i] = a[i] * b[i];
#endif
}

// ============================================================================
// copyWithGain -- dst[i] = src[i] * gain
// ============================================================================

/** @brief Copies a buffer applying a gain factor (out-of-place). */
inline void copyWithGain(float* DSPARK_RESTRICT dst, const float* DSPARK_RESTRICT src, float gain, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256 vGain = _mm256_set1_ps(gain);
    int i = 0;
    for (; i + 7 < count; i += 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(src + i), vGain));
    for (; i < count; ++i) dst[i] = src[i] * gain;
#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vGain = _mm_set1_ps(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
        _mm_storeu_ps(dst + i, _mm_mul_ps(_mm_loadu_ps(src + i), vGain));
    for (; i < count; ++i) dst[i] = src[i] * gain;
#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vGain = vdupq_n_f32(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
        vst1q_f32(dst + i, vmulq_f32(vld1q_f32(src + i), vGain));
    for (; i < count; ++i) dst[i] = src[i] * gain;
#else
    for (int i = 0; i < count; ++i) dst[i] = src[i] * gain;
#endif
}

/** @brief Double overload. */
inline void copyWithGain(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT src, double gain, int count) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    const __m256d vGain = _mm256_set1_pd(gain);
    int i = 0;
    for (; i + 3 < count; i += 4)
        _mm256_storeu_pd(dst + i, _mm256_mul_pd(_mm256_loadu_pd(src + i), vGain));
    for (; i < count; ++i) dst[i] = src[i] * gain;
#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vGain = _mm_set1_pd(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
        _mm_storeu_pd(dst + i, _mm_mul_pd(_mm_loadu_pd(src + i), vGain));
    for (; i < count; ++i) dst[i] = src[i] * gain;

#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vGain = vdupq_n_f64(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
        vst1q_f64(dst + i, vmulq_f64(vld1q_f64(src + i), vGain));
    for (; i < count; ++i) dst[i] = src[i] * gain;
#else
    for (int i = 0; i < count; ++i) dst[i] = src[i] * gain;
#endif
}

// ============================================================================
// applyGainRamp -- data[i] *= gainStart + i * step   (linear fade)
// ============================================================================

/**
 * @brief In-place linear gain ramp (the canonical click-free fade primitive).
 *
 * Sample i is scaled by `gainStart + i * (gainEnd - gainStart) / count`, so
 * the ramp covers [gainStart, gainEnd) across the block: the last sample sits
 * one step below gainEnd. For a continuous fade across consecutive blocks,
 * start the next block at gainEnd. count <= 0 is a no-op.
 *
 * Every gain is computed from its own index, never accumulated: an
 * accumulated ramp drifted by 1.3e-3 over 4.8M samples and stepped at the
 * seam between the vector body and the scalar tail. Here the vector lanes
 * and the tail evaluate the same expression and agree exactly.
 */
inline void applyGainRamp(float* DSPARK_RESTRICT data, float gainStart, float gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const float step = (gainEnd - gainStart) / static_cast<float>(count);
    int i = 0;
#if defined(DSPARK_SIMD_AVX)
    const __m256 vStart = _mm256_set1_ps(gainStart), vStep = _mm256_set1_ps(step), vInc = _mm256_set1_ps(8.0f);
    __m256 vIdx = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    for (; i + 7 < count; i += 8)
    {
        const __m256 vGain = _mm256_add_ps(vStart, _mm256_mul_ps(vStep, vIdx));
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vGain));
        vIdx = _mm256_add_ps(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vStart = _mm_set1_ps(gainStart), vStep = _mm_set1_ps(step), vInc = _mm_set1_ps(4.0f);
    __m128 vIdx = _mm_setr_ps(0.0f, 1.0f, 2.0f, 3.0f);
    for (; i + 3 < count; i += 4)
    {
        const __m128 vGain = _mm_add_ps(vStart, _mm_mul_ps(vStep, vIdx));
        _mm_storeu_ps(data + i, _mm_mul_ps(_mm_loadu_ps(data + i), vGain));
        vIdx = _mm_add_ps(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vStart = vdupq_n_f32(gainStart), vStep = vdupq_n_f32(step), vInc = vdupq_n_f32(4.0f);
    const float init[4] = { 0.0f, 1.0f, 2.0f, 3.0f };
    float32x4_t vIdx = vld1q_f32(init);
    for (; i + 3 < count; i += 4)
    {
        const float32x4_t vGain = vaddq_f32(vStart, vmulq_f32(vStep, vIdx));
        vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), vGain));
        vIdx = vaddq_f32(vIdx, vInc);
    }
#endif
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<float>(i);
}

/** @brief Double overload. */
inline void applyGainRamp(double* DSPARK_RESTRICT data, double gainStart, double gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const double step = (gainEnd - gainStart) / static_cast<double>(count);
    int i = 0;
#if defined(DSPARK_SIMD_AVX)
    const __m256d vStart = _mm256_set1_pd(gainStart), vStep = _mm256_set1_pd(step), vInc = _mm256_set1_pd(4.0);
    __m256d vIdx = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    for (; i + 3 < count; i += 4)
    {
        const __m256d vGain = _mm256_add_pd(vStart, _mm256_mul_pd(vStep, vIdx));
        _mm256_storeu_pd(data + i, _mm256_mul_pd(_mm256_loadu_pd(data + i), vGain));
        vIdx = _mm256_add_pd(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vStart = _mm_set1_pd(gainStart), vStep = _mm_set1_pd(step), vInc = _mm_set1_pd(2.0);
    __m128d vIdx = _mm_setr_pd(0.0, 1.0);
    for (; i + 1 < count; i += 2)
    {
        const __m128d vGain = _mm_add_pd(vStart, _mm_mul_pd(vStep, vIdx));
        _mm_storeu_pd(data + i, _mm_mul_pd(_mm_loadu_pd(data + i), vGain));
        vIdx = _mm_add_pd(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vStart = vdupq_n_f64(gainStart), vStep = vdupq_n_f64(step), vInc = vdupq_n_f64(2.0);
    const double init[2] = { 0.0, 1.0 };
    float64x2_t vIdx = vld1q_f64(init);
    for (; i + 1 < count; i += 2)
    {
        const float64x2_t vGain = vaddq_f64(vStart, vmulq_f64(vStep, vIdx));
        vst1q_f64(data + i, vmulq_f64(vld1q_f64(data + i), vGain));
        vIdx = vaddq_f64(vIdx, vInc);
    }
#endif
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<double>(i);
}

// ============================================================================
// addWithGainRamp -- dst[i] += src[i] * (gainStart + i * step)
// ============================================================================

/**
 * @brief Adds source samples scaled by a linear gain ramp into a destination.
 *
 * Same ramp convention as applyGainRamp: sample i uses
 * `gainStart + i * (gainEnd - gainStart) / count`, spanning [gainStart,
 * gainEnd) across the block, each gain computed from its index (no drift).
 * count <= 0 is a no-op.
 */
inline void addWithGainRamp(float* DSPARK_RESTRICT dst, const float* DSPARK_RESTRICT src,
                            float gainStart, float gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const float step = (gainEnd - gainStart) / static_cast<float>(count);
    int i = 0;
#if defined(DSPARK_SIMD_AVX)
    const __m256 vStart = _mm256_set1_ps(gainStart), vStep = _mm256_set1_ps(step), vInc = _mm256_set1_ps(8.0f);
    __m256 vIdx = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    for (; i + 7 < count; i += 8)
    {
        const __m256 vGain = _mm256_add_ps(vStart, _mm256_mul_ps(vStep, vIdx));
        const __m256 vDst = _mm256_loadu_ps(dst + i);
        const __m256 vSrc = _mm256_loadu_ps(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(vSrc, vGain, vDst));
        #else
            _mm256_storeu_ps(dst + i, _mm256_add_ps(vDst, _mm256_mul_ps(vSrc, vGain)));
        #endif
        vIdx = _mm256_add_ps(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vStart = _mm_set1_ps(gainStart), vStep = _mm_set1_ps(step), vInc = _mm_set1_ps(4.0f);
    __m128 vIdx = _mm_setr_ps(0.0f, 1.0f, 2.0f, 3.0f);
    for (; i + 3 < count; i += 4)
    {
        const __m128 vGain = _mm_add_ps(vStart, _mm_mul_ps(vStep, vIdx));
        const __m128 vDst = _mm_loadu_ps(dst + i);
        const __m128 vSrc = _mm_loadu_ps(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm_storeu_ps(dst + i, _mm_fmadd_ps(vSrc, vGain, vDst));
        #else
            _mm_storeu_ps(dst + i, _mm_add_ps(vDst, _mm_mul_ps(vSrc, vGain)));
        #endif
        vIdx = _mm_add_ps(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vStart = vdupq_n_f32(gainStart), vStep = vdupq_n_f32(step), vInc = vdupq_n_f32(4.0f);
    const float init[4] = { 0.0f, 1.0f, 2.0f, 3.0f };
    float32x4_t vIdx = vld1q_f32(init);
    for (; i + 3 < count; i += 4)
    {
        const float32x4_t vGain = vaddq_f32(vStart, vmulq_f32(vStep, vIdx));
        vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vld1q_f32(src + i), vGain));
        vIdx = vaddq_f32(vIdx, vInc);
    }
#endif
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<float>(i));
}

/** @brief Double overload. */
inline void addWithGainRamp(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT src,
                            double gainStart, double gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const double step = (gainEnd - gainStart) / static_cast<double>(count);
    int i = 0;
#if defined(DSPARK_SIMD_AVX)
    const __m256d vStart = _mm256_set1_pd(gainStart), vStep = _mm256_set1_pd(step), vInc = _mm256_set1_pd(4.0);
    __m256d vIdx = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
    for (; i + 3 < count; i += 4)
    {
        const __m256d vGain = _mm256_add_pd(vStart, _mm256_mul_pd(vStep, vIdx));
        const __m256d vDst = _mm256_loadu_pd(dst + i);
        const __m256d vSrc = _mm256_loadu_pd(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd(vSrc, vGain, vDst));
        #else
            _mm256_storeu_pd(dst + i, _mm256_add_pd(vDst, _mm256_mul_pd(vSrc, vGain)));
        #endif
        vIdx = _mm256_add_pd(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vStart = _mm_set1_pd(gainStart), vStep = _mm_set1_pd(step), vInc = _mm_set1_pd(2.0);
    __m128d vIdx = _mm_setr_pd(0.0, 1.0);
    for (; i + 1 < count; i += 2)
    {
        const __m128d vGain = _mm_add_pd(vStart, _mm_mul_pd(vStep, vIdx));
        const __m128d vDst = _mm_loadu_pd(dst + i);
        const __m128d vSrc = _mm_loadu_pd(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm_storeu_pd(dst + i, _mm_fmadd_pd(vSrc, vGain, vDst));
        #else
            _mm_storeu_pd(dst + i, _mm_add_pd(vDst, _mm_mul_pd(vSrc, vGain)));
        #endif
        vIdx = _mm_add_pd(vIdx, vInc);
    }
#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vStart = vdupq_n_f64(gainStart), vStep = vdupq_n_f64(step), vInc = vdupq_n_f64(2.0);
    const double init[2] = { 0.0, 1.0 };
    float64x2_t vIdx = vld1q_f64(init);
    for (; i + 1 < count; i += 2)
    {
        const float64x2_t vGain = vaddq_f64(vStart, vmulq_f64(vStep, vIdx));
        vst1q_f64(dst + i, vfmaq_f64(vld1q_f64(dst + i), vld1q_f64(src + i), vGain));
        vIdx = vaddq_f64(vIdx, vInc);
    }
#endif
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<double>(i));
}

// ============================================================================
// sumOfSquares -- sum(data[i]^2)   (RMS / energy metering)
// ============================================================================

/** @brief Returns the sum of squared samples (energy). */
inline float sumOfSquares(const float* DSPARK_RESTRICT data, int count) noexcept
{
    return dotProduct(data, data, count);
}

/** @brief Double overload. */
inline double sumOfSquares(const double* DSPARK_RESTRICT data, int count) noexcept
{
    return dotProduct(data, data, count);
}

// ============================================================================
// complexMulAccum -- accum[k] += a[k] * b[k] over interleaved complex bins
// ============================================================================

/**
 * @brief Complex multiply-accumulate over interleaved [re, im, ...] spectra.
 *
 * The workhorse of frequency-domain (partitioned) convolution:
 * `accum += a * b` evaluated as complex numbers, for `bins` complex bins.
 */
inline void complexMulAccum(float* DSPARK_RESTRICT accum, const float* DSPARK_RESTRICT a,
                            const float* DSPARK_RESTRICT b, int bins) noexcept
{
#if defined(DSPARK_SIMD_AVX)
    complexMulAccumWide(accum, a, b, bins);
#elif defined(DSPARK_SIMD_SSE2)
  #if defined(DSPARK_SIMD_DISPATCH_AVX2)
    if (isRuntimeDispatchActive()) { avx2::complexMulAccumWide(accum, a, b, bins); return; }
  #endif
    // Sign mask: negate real positions for (re1*re2 - im1*im2)
    const __m128 negMask = _mm_castsi128_ps(_mm_setr_epi32(
        static_cast<int>(0x80000000u), 0,
        static_cast<int>(0x80000000u), 0));

    int k = 0;
    for (; k + 1 < bins; k += 2)
    {
        const __m128 va   = _mm_loadu_ps(a + 2 * k);
        const __m128 vb   = _mm_loadu_ps(b + 2 * k);
        __m128 vacc       = _mm_loadu_ps(accum + 2 * k);

        const __m128 aRe   = _mm_shuffle_ps(va, va, _MM_SHUFFLE(2, 2, 0, 0));
        const __m128 aIm   = _mm_shuffle_ps(va, va, _MM_SHUFFLE(3, 3, 1, 1));
        const __m128 bSwap = _mm_shuffle_ps(vb, vb, _MM_SHUFFLE(2, 3, 0, 1));

        const __m128 p1 = _mm_mul_ps(aRe, vb);
        const __m128 p2 = _mm_xor_ps(_mm_mul_ps(aIm, bSwap), negMask);

        vacc = _mm_add_ps(vacc, _mm_add_ps(p1, p2));
        _mm_storeu_ps(accum + 2 * k, vacc);
    }
    for (; k < bins; ++k)
    {
        const float re1 = a[2 * k], im1 = a[2 * k + 1];
        const float re2 = b[2 * k], im2 = b[2 * k + 1];
        accum[2 * k]     += re1 * re2 - im1 * im2;
        accum[2 * k + 1] += re1 * im2 + im1 * re2;
    }
#elif defined(DSPARK_SIMD_NEON)
    alignas(16) static constexpr uint32_t kNegRe[4] = { 0x80000000u, 0u, 0x80000000u, 0u };
    const uint32x4_t negMask = vld1q_u32(kNegRe);

    int k = 0;
    for (; k + 1 < bins; k += 2)
    {
        const float32x4_t va   = vld1q_f32(a + 2 * k);
        const float32x4_t vb   = vld1q_f32(b + 2 * k);
        float32x4_t vacc       = vld1q_f32(accum + 2 * k);

        const float32x4_t aRe   = vtrn1q_f32(va, va);
        const float32x4_t aIm   = vtrn2q_f32(va, va);
        const float32x4_t bSwap = vrev64q_f32(vb);

        const float32x4_t p1 = vmulq_f32(aRe, vb);
        const float32x4_t p2 = vreinterpretq_f32_u32(
            veorq_u32(vreinterpretq_u32_f32(vmulq_f32(aIm, bSwap)), negMask));

        vacc = vaddq_f32(vacc, vaddq_f32(p1, p2));
        vst1q_f32(accum + 2 * k, vacc);
    }
    for (; k < bins; ++k)
    {
        const float re1 = a[2 * k], im1 = a[2 * k + 1];
        const float re2 = b[2 * k], im2 = b[2 * k + 1];
        accum[2 * k]     += re1 * re2 - im1 * im2;
        accum[2 * k + 1] += re1 * im2 + im1 * re2;
    }
#else
    for (int k = 0; k < bins; ++k)
    {
        const float re1 = a[2 * k], im1 = a[2 * k + 1];
        const float re2 = b[2 * k], im2 = b[2 * k + 1];
        accum[2 * k]     += re1 * re2 - im1 * im2;
        accum[2 * k + 1] += re1 * im2 + im1 * re2;
    }
#endif
}

/** @brief Double overload (scalar on purpose: double spectra are an offline path). */
inline void complexMulAccum(double* DSPARK_RESTRICT accum, const double* DSPARK_RESTRICT a,
                            const double* DSPARK_RESTRICT b, int bins) noexcept
{
    for (int k = 0; k < bins; ++k)
    {
        const double re1 = a[2 * k], im1 = a[2 * k + 1];
        const double re2 = b[2 * k], im2 = b[2 * k + 1];
        accum[2 * k]     += re1 * re2 - im1 * im2;
        accum[2 * k + 1] += re1 * im2 + im1 * re2;
    }
}

// ============================================================================
// Template dispatchers
// ============================================================================

template <typename T>
void addWithGainT(T* DSPARK_RESTRICT dst, const T* DSPARK_RESTRICT src, T gain, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    addWithGain(dst, src, gain, count);
}

template <typename T>
void applyGainT(T* DSPARK_RESTRICT data, T gain, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    applyGain(data, gain, count);
}

template <typename T>
T peakLevelT(const T* DSPARK_RESTRICT data, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    return peakLevel(data, count);
}

template <typename T>
T dotProductT(const T* DSPARK_RESTRICT a, const T* DSPARK_RESTRICT b, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    return dotProduct(a, b, count);
}

template <typename T>
void addT(T* DSPARK_RESTRICT dst, const T* DSPARK_RESTRICT src, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    add(dst, src, count);
}

template <typename T>
void multiplyT(T* DSPARK_RESTRICT dst, const T* DSPARK_RESTRICT a, const T* DSPARK_RESTRICT b, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    multiply(dst, a, b, count);
}

template <typename T>
void copyWithGainT(T* DSPARK_RESTRICT dst, const T* DSPARK_RESTRICT src, T gain, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    copyWithGain(dst, src, gain, count);
}

template <typename T>
T sumOfSquaresT(const T* DSPARK_RESTRICT data, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    return sumOfSquares(data, count);
}

template <typename T>
void applyGainRampT(T* DSPARK_RESTRICT data, T gainStart, T gainEnd, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    applyGainRamp(data, gainStart, gainEnd, count);
}

template <typename T>
void addWithGainRampT(T* DSPARK_RESTRICT dst, const T* DSPARK_RESTRICT src, T gainStart, T gainEnd, int count) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    addWithGainRamp(dst, src, gainStart, gainEnd, count);
}

// ============================================================================
// firCorrelate -- block FIR on the vector abstraction (detail/SimdVec.inl)
// ============================================================================

/**
 * @brief Block FIR as a valid correlation: y[i] = gain * sum_j h[j] * x[i + j]
 *        for i in [0, n), reading x[0 .. n + taps - 1).
 *
 * Register-blocked over outputs (four vectors at a time, the taps in the
 * inner loop), so no output pays a horizontal sum: for the 16..256-tap
 * kernels of oversamplers and resamplers it runs several times faster than
 * one dotProduct() per output. y must not overlap x or h. Runs on the AVX2+FMA
 * copy when isRuntimeDispatchActive().
 *
 * @param x     Input, n + taps - 1 samples.
 * @param h     Kernel, taps coefficients (h[0] meets x[i]).
 * @param taps  Kernel length.
 * @param y     Output, n samples.
 * @param n     Number of outputs.
 * @param gain  Scale applied to every output.
 */
template <typename T>
void firCorrelate(const T* x, const T* h, int taps, T* DSPARK_RESTRICT y, int n, T gain) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
    if (isRuntimeDispatchActive())
    {
        avx2::firCorrelateKernel(x, h, taps, y, n, gain);
        return;
    }
#endif
    firCorrelateKernel(x, h, taps, y, n, gain);
}

} // namespace simd
} // namespace dspark
