// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file SimdOps.h
 * @brief SIMD-accelerated buffer operations for real-time audio processing.
 *
 * Optimized with FMA (Fused Multiply-Add) detection and Strict Aliasing
 * declarations (__restrict) for aggressive compiler unrolling.
 */

// --- Platform SIMD detection ------------------------------------------------
#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    #define DSPARK_SIMD_SSE2 1
    #include <emmintrin.h>           // SSE2
    #if defined(__AVX__)
        #define DSPARK_SIMD_AVX 1
        #include <immintrin.h>       // AVX
    #endif
    // Detect FMA3 support
    #if defined(__FMA__) || defined(__AVX2__)
        #define DSPARK_SIMD_FMA 1
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define DSPARK_SIMD_NEON 1
    #include <arm_neon.h>
#endif

// --- Restrict Macro for Pointer Aliasing Optimization -----------------------
// NOTE: functions taking DSPARK_RESTRICT pointers assume the arguments do NOT
// alias (no overlapping ranges, dst != src). Passing overlapping/identical
// buffers is undefined behaviour. Defined idempotently (#ifndef) because a few
// headers historically declared the same macro.
#ifndef DSPARK_RESTRICT
  #if defined(_MSC_VER)
    #define DSPARK_RESTRICT __restrict
  #elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_RESTRICT __restrict__
  #else
    #define DSPARK_RESTRICT
  #endif
#endif

#include <cmath>
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
        vst1q_f32(dst + i, vmlaq_f32(vDst, vSrc, vGain)); // vmlaq is native FMA
    }
    for (; i < count; ++i) dst[i] += src[i] * gain;

#else
    for (int i = 0; i < count; ++i) dst[i] += src[i] * gain;
#endif
}

/** @brief Double overload */
inline void addWithGain(double* DSPARK_RESTRICT dst, const double* DSPARK_RESTRICT src, double gain, int count) noexcept
{
#if defined(DSPARK_SIMD_SSE2)
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
#if defined(DSPARK_SIMD_SSE2)
    const __m128d vGain = _mm_set1_pd(gain);
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d v = _mm_loadu_pd(data + i);
        _mm_storeu_pd(data + i, _mm_mul_pd(v, vGain));
    }
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
        vMax = _mm256_max_ps(vMax, v);
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
        vMax = _mm_max_ps(vMax, v);
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
        vMax = vmaxq_f32(vMax, v);
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

/** @brief Double overload. */
inline double peakLevel(const double* DSPARK_RESTRICT data, int count) noexcept
{
#if defined(DSPARK_SIMD_SSE2)
    const __m128d absMask = _mm_castsi128_pd(_mm_set_epi64x(
        static_cast<int64_t>(0x7FFFFFFFFFFFFFFF),
        static_cast<int64_t>(0x7FFFFFFFFFFFFFFF)));
    __m128d vMax = _mm_setzero_pd();
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d v = _mm_loadu_pd(data + i);
        v = _mm_and_pd(v, absMask);
        vMax = _mm_max_pd(vMax, v);
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
    __m256 vSum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        
        #if defined(DSPARK_SIMD_FMA)
            vSum = _mm256_fmadd_ps(va, vb, vSum);
        #else
            vSum = _mm256_add_ps(vSum, _mm256_mul_ps(va, vb));
        #endif
    }
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
    __m128 vSum = _mm_setzero_ps();
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        
        #if defined(DSPARK_SIMD_FMA)
            vSum = _mm_fmadd_ps(va, vb, vSum);
        #else
            vSum = _mm_add_ps(vSum, _mm_mul_ps(va, vb));
        #endif
    }
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, vSum);
    float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    
    for (; i < count; ++i) sum += a[i] * b[i];
    return sum;

#elif defined(DSPARK_SIMD_NEON)
    float32x4_t vSum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < count; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vSum = vmlaq_f32(vSum, va, vb);
    }
    float sum = vaddvq_f32(vSum);
    
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
#if defined(DSPARK_SIMD_SSE2)
    __m128d vSum = _mm_setzero_pd();
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d va = _mm_loadu_pd(a + i);
        __m128d vb = _mm_loadu_pd(b + i);
        
        #if defined(DSPARK_SIMD_FMA)
            vSum = _mm_fmadd_pd(va, vb, vSum);
        #else
            vSum = _mm_add_pd(vSum, _mm_mul_pd(va, vb));
        #endif
    }
    __m128d hi = _mm_unpackhi_pd(vSum, vSum);
    __m128d s = _mm_add_sd(vSum, hi);
    double sum = _mm_cvtsd_f64(s);
    
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
#if defined(DSPARK_SIMD_SSE2)
    int i = 0;
    for (; i + 1 < count; i += 2)
    {
        __m128d vDst = _mm_loadu_pd(dst + i);
        __m128d vSrc = _mm_loadu_pd(src + i);
        _mm_storeu_pd(dst + i, _mm_add_pd(vDst, vSrc));
    }
    for (; i < count; ++i) dst[i] += src[i];
#else
    for (int i = 0; i < count; ++i) dst[i] += src[i];
#endif
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

} // namespace simd
} // namespace dspark