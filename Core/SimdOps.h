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
 * Build flags: binaries compiled for a generic x64 baseline stay on the SSE2
 * paths. Enable the wider kernels with /arch:AVX2 (MSVC) or -march=x86-64-v3
 * (GCC/Clang) when you control the deployment target. WebAssembly: compile
 * with -msimd128 -msse2 so Emscripten lowers the SSE2 paths to Wasm SIMD128;
 * otherwise the scalar fallbacks are used.
 */

// --- Platform SIMD detection ------------------------------------------------
#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
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

#include <cstdint>
#include <type_traits>

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
    // Four independent accumulators break the FMA latency chain (4-5 cycles),
    // letting long FIR kernels run at full multiply throughput (~2-4x faster).
    __m256 vSum0 = _mm256_setzero_ps();
    __m256 vSum1 = _mm256_setzero_ps();
    __m256 vSum2 = _mm256_setzero_ps();
    __m256 vSum3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < count; i += 32)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      vSum0);
            vSum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  vSum1);
            vSum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), vSum2);
            vSum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), vSum3);
        #else
            vSum0 = _mm256_add_ps(vSum0, _mm256_mul_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i)));
            vSum1 = _mm256_add_ps(vSum1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8)));
            vSum2 = _mm256_add_ps(vSum2, _mm256_mul_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16)));
            vSum3 = _mm256_add_ps(vSum3, _mm256_mul_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24)));
        #endif
    }
    for (; i + 7 < count; i += 8)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), vSum0);
        #else
            vSum0 = _mm256_add_ps(vSum0, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        #endif
    }
    __m256 vSum = _mm256_add_ps(_mm256_add_ps(vSum0, vSum1), _mm256_add_ps(vSum2, vSum3));
    __m128 hi = _mm256_extractf128_ps(vSum, 1);
    __m128 lo = _mm256_castps256_ps128(vSum);
    __m128 sum4 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(sum4, sum4);
    __m128 sum2 = _mm_add_ps(sum4, shuf);
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    float sum = _mm_cvtss_f32(sum1);

    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;

#elif defined(DSPARK_SIMD_SSE2)
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
    __m256d vSum0 = _mm256_setzero_pd();
    __m256d vSum1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i),     vSum0);
            vSum1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), vSum1);
        #else
            vSum0 = _mm256_add_pd(vSum0, _mm256_mul_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i)));
            vSum1 = _mm256_add_pd(vSum1, _mm256_mul_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
        #endif
    }
    for (; i + 3 < count; i += 4)
    {
        #if defined(DSPARK_SIMD_FMA)
            vSum0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i), vSum0);
        #else
            vSum0 = _mm256_add_pd(vSum0, _mm256_mul_pd(_mm256_loadu_pd(a + i), _mm256_loadu_pd(b + i)));
        #endif
    }
    __m256d vSum = _mm256_add_pd(vSum0, vSum1);
    __m128d lo = _mm256_castpd256_pd128(vSum);
    __m128d hi = _mm256_extractf128_pd(vSum, 1);
    __m128d s2 = _mm_add_pd(lo, hi);
    double sum = _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));

    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;

#elif defined(DSPARK_SIMD_SSE2)
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
 */
inline void applyGainRamp(float* DSPARK_RESTRICT data, float gainStart, float gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const float step = (gainEnd - gainStart) / static_cast<float>(count);
#if defined(DSPARK_SIMD_AVX)
    const __m256 vStep = _mm256_set1_ps(step * 8.0f);
    __m256 vGain = _mm256_setr_ps(gainStart,            gainStart + step,
                                  gainStart + 2 * step, gainStart + 3 * step,
                                  gainStart + 4 * step, gainStart + 5 * step,
                                  gainStart + 6 * step, gainStart + 7 * step);
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vGain));
        vGain = _mm256_add_ps(vGain, vStep);
    }
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<float>(i);

#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vStep = _mm_set1_ps(step * 4.0f);
    __m128 vGain = _mm_setr_ps(gainStart, gainStart + step, gainStart + 2 * step, gainStart + 3 * step);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        _mm_storeu_ps(data + i, _mm_mul_ps(_mm_loadu_ps(data + i), vGain));
        vGain = _mm_add_ps(vGain, vStep);
    }
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<float>(i);

#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vStep = vdupq_n_f32(step * 4.0f);
    const float init[4] = { gainStart, gainStart + step, gainStart + 2 * step, gainStart + 3 * step };
    float32x4_t vGain = vld1q_f32(init);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), vGain));
        vGain = vaddq_f32(vGain, vStep);
    }
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<float>(i);

#else
    for (int i = 0; i < count; ++i) data[i] *= gainStart + step * static_cast<float>(i);
#endif
}

/** @brief Double overload. */
inline void applyGainRamp(double* DSPARK_RESTRICT data, double gainStart, double gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const double step = (gainEnd - gainStart) / static_cast<double>(count);
#if defined(DSPARK_SIMD_AVX)
    const __m256d vStep = _mm256_set1_pd(step * 4.0);
    __m256d vGain = _mm256_setr_pd(gainStart,            gainStart + step,
                                   gainStart + 2 * step, gainStart + 3 * step);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        _mm256_storeu_pd(data + i, _mm256_mul_pd(_mm256_loadu_pd(data + i), vGain));
        vGain = _mm256_add_pd(vGain, vStep);
    }
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<double>(i);

#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vStep = _mm_set1_pd(step * 2.0);
    __m128d vGain = _mm_setr_pd(gainStart, gainStart + step);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        _mm_storeu_pd(data + i, _mm_mul_pd(_mm_loadu_pd(data + i), vGain));
        vGain = _mm_add_pd(vGain, vStep);
    }
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<double>(i);

#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vStep = vdupq_n_f64(step * 2.0);
    const double init[2] = { gainStart, gainStart + step };
    float64x2_t vGain = vld1q_f64(init);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        vst1q_f64(data + i, vmulq_f64(vld1q_f64(data + i), vGain));
        vGain = vaddq_f64(vGain, vStep);
    }
    for (; i < count; ++i) data[i] *= gainStart + step * static_cast<double>(i);

#else
    for (int i = 0; i < count; ++i) data[i] *= gainStart + step * static_cast<double>(i);
#endif
}

// ============================================================================
// addWithGainRamp -- dst[i] += src[i] * (gainStart + i * step)
// ============================================================================

/**
 * @brief Adds source samples scaled by a linear gain ramp into a destination.
 *
 * Same ramp convention as applyGainRamp: sample i uses
 * `gainStart + i * (gainEnd - gainStart) / count`, spanning [gainStart,
 * gainEnd) across the block. count <= 0 is a no-op.
 */
inline void addWithGainRamp(float* DSPARK_RESTRICT dst, const float* DSPARK_RESTRICT src,
                            float gainStart, float gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const float step = (gainEnd - gainStart) / static_cast<float>(count);
#if defined(DSPARK_SIMD_AVX)
    const __m256 vStep = _mm256_set1_ps(step * 8.0f);
    __m256 vGain = _mm256_setr_ps(gainStart,            gainStart + step,
                                  gainStart + 2 * step, gainStart + 3 * step,
                                  gainStart + 4 * step, gainStart + 5 * step,
                                  gainStart + 6 * step, gainStart + 7 * step);
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        const __m256 vDst = _mm256_loadu_ps(dst + i);
        const __m256 vSrc = _mm256_loadu_ps(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(vSrc, vGain, vDst));
        #else
            _mm256_storeu_ps(dst + i, _mm256_add_ps(vDst, _mm256_mul_ps(vSrc, vGain)));
        #endif
        vGain = _mm256_add_ps(vGain, vStep);
    }
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<float>(i));

#elif defined(DSPARK_SIMD_SSE2)
    const __m128 vStep = _mm_set1_ps(step * 4.0f);
    __m128 vGain = _mm_setr_ps(gainStart, gainStart + step, gainStart + 2 * step, gainStart + 3 * step);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        const __m128 vDst = _mm_loadu_ps(dst + i);
        const __m128 vSrc = _mm_loadu_ps(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm_storeu_ps(dst + i, _mm_fmadd_ps(vSrc, vGain, vDst));
        #else
            _mm_storeu_ps(dst + i, _mm_add_ps(vDst, _mm_mul_ps(vSrc, vGain)));
        #endif
        vGain = _mm_add_ps(vGain, vStep);
    }
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<float>(i));

#elif defined(DSPARK_SIMD_NEON)
    const float32x4_t vStep = vdupq_n_f32(step * 4.0f);
    const float init[4] = { gainStart, gainStart + step, gainStart + 2 * step, gainStart + 3 * step };
    float32x4_t vGain = vld1q_f32(init);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        vst1q_f32(dst + i, vfmaq_f32(vld1q_f32(dst + i), vld1q_f32(src + i), vGain));
        vGain = vaddq_f32(vGain, vStep);
    }
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<float>(i));

#else
    for (int i = 0; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<float>(i));
#endif
}

/** @brief Double overload. */
inline void addWithGainRamp(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT src,
                            double gainStart, double gainEnd, int count) noexcept
{
    if (count <= 0) return;
    const double step = (gainEnd - gainStart) / static_cast<double>(count);
#if defined(DSPARK_SIMD_AVX)
    const __m256d vStep = _mm256_set1_pd(step * 4.0);
    __m256d vGain = _mm256_setr_pd(gainStart,            gainStart + step,
                                   gainStart + 2 * step, gainStart + 3 * step);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        const __m256d vDst = _mm256_loadu_pd(dst + i);
        const __m256d vSrc = _mm256_loadu_pd(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm256_storeu_pd(dst + i, _mm256_fmadd_pd(vSrc, vGain, vDst));
        #else
            _mm256_storeu_pd(dst + i, _mm256_add_pd(vDst, _mm256_mul_pd(vSrc, vGain)));
        #endif
        vGain = _mm256_add_pd(vGain, vStep);
    }
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<double>(i));

#elif defined(DSPARK_SIMD_SSE2)
    const __m128d vStep = _mm_set1_pd(step * 2.0);
    __m128d vGain = _mm_setr_pd(gainStart, gainStart + step);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        const __m128d vDst = _mm_loadu_pd(dst + i);
        const __m128d vSrc = _mm_loadu_pd(src + i);
        #if defined(DSPARK_SIMD_FMA)
            _mm_storeu_pd(dst + i, _mm_fmadd_pd(vSrc, vGain, vDst));
        #else
            _mm_storeu_pd(dst + i, _mm_add_pd(vDst, _mm_mul_pd(vSrc, vGain)));
        #endif
        vGain = _mm_add_pd(vGain, vStep);
    }
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<double>(i));

#elif defined(DSPARK_SIMD_NEON)
    const float64x2_t vStep = vdupq_n_f64(step * 2.0);
    const double init[2] = { gainStart, gainStart + step };
    float64x2_t vGain = vld1q_f64(init);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        vst1q_f64(dst + i, vfmaq_f64(vld1q_f64(dst + i), vld1q_f64(src + i), vGain));
        vGain = vaddq_f64(vGain, vStep);
    }
    for (; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<double>(i));

#else
    for (int i = 0; i < count; ++i) dst[i] += src[i] * (gainStart + step * static_cast<double>(i));
#endif
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
    // 4 complex bins per iteration. vpermilps builds the [re,re] / [im,im]
    // duplicates and the [im,re] swap within each 128-bit lane; the
    // alternating subtract/add of the two partial products comes straight
    // from (fm)addsub, so no sign mask is needed on this path.
    int k = 0;
    for (; k + 3 < bins; k += 4)
    {
        const __m256 va   = _mm256_loadu_ps(a + 2 * k);
        const __m256 vb   = _mm256_loadu_ps(b + 2 * k);
        const __m256 vacc = _mm256_loadu_ps(accum + 2 * k);

        const __m256 aRe   = _mm256_permute_ps(va, _MM_SHUFFLE(2, 2, 0, 0));
        const __m256 aIm   = _mm256_permute_ps(va, _MM_SHUFFLE(3, 3, 1, 1));
        const __m256 bSwap = _mm256_permute_ps(vb, _MM_SHUFFLE(2, 3, 0, 1));

        #if defined(DSPARK_SIMD_FMA)
            const __m256 prod = _mm256_fmaddsub_ps(aRe, vb, _mm256_mul_ps(aIm, bSwap));
        #else
            const __m256 prod = _mm256_addsub_ps(_mm256_mul_ps(aRe, vb), _mm256_mul_ps(aIm, bSwap));
        #endif

        _mm256_storeu_ps(accum + 2 * k, _mm256_add_ps(vacc, prod));
    }
    for (; k < bins; ++k)
    {
        const float re1 = a[2 * k], im1 = a[2 * k + 1];
        const float re2 = b[2 * k], im2 = b[2 * k + 1];
        accum[2 * k]     += re1 * re2 - im1 * im2;
        accum[2 * k + 1] += re1 * im2 + im1 * re2;
    }
#elif defined(DSPARK_SIMD_SSE2)
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

} // namespace simd
} // namespace dspark