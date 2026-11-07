// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WavetableBank.h
 * @brief Band-limited, mipmapped multi-frame wavetables with a morphing reader.
 *
 * A bank holds one or more single-cycle frames. Each frame is stored as 20
 * mip levels half an octave apart, every level synthesised exactly from the
 * frame's harmonics by inverse FFT, so a level holds no harmonic above its
 * budget. The reader blends two adjacent frames (the morph position) and two
 * adjacent levels (the anti-aliasing crossfade) and interpolates the result.
 *
 * Interpolation: Niemitalo's optimal 6-point, 5th-order polynomial for 4x
 * oversampled data ("Polynomial Interpolators for High-Quality Resampling of
 * Oversampled Audio", 2001). Every level therefore keeps at least four
 * table samples per period of its top harmonic, twice Nyquist: a level of H
 * harmonics is max(2048, 8 H) samples rounded up to a power of two. The
 * interpolator is not flat at the edge of that band (up to 0.4 dB of droop),
 * so each harmonic is pre-emphasised by the inverse of the interpolator's
 * own response at its table frequency, computed when the bank is built: the
 * played amplitudes are the frame's amplitudes. The interpolator is linear in
 * the samples, so frames, and levels of equal size, are blended before one
 * polynomial evaluation.
 *
 * Measured at 48 kHz, worst inharmonic component below 20 kHz relative to the
 * strongest harmonic, 27.5 Hz to 5 kHz: about -165 dB for 1/h and 1/sqrt(h)
 * spectra and -152 dB for a flat spectrum (every harmonic at full scale), the
 * float storage being the limit. The 4-point cubic Hermite reader of 2048
 * sample levels measured -72 dB (1/h) and -74 dB (flat) at 27.5 Hz.
 *
 * Anti-aliasing contract (as WavetableOscillator's): a level is used at a
 * frequency f only while its highest harmonic stays below
 * max(fs/2, fs - 20 kHz), so anything that folds lands above 20 kHz, and the
 * crossfade pairs the brightest such level with the next duller one, so a
 * frequency sweep changes the timbre continuously.
 *
 * readBlock() is the block form of read(): the phases of a block (and one
 * frame position for it, or one per sample), interpolated four float or two
 * double samples at a time on SSE2 and NEON and eight or four with AVX2
 * (chosen at run time on x86-64). It returns read()'s values, operation for
 * operation. Measured on x86-64 at 440 Hz with the level crossfade active,
 * float: read() 14 ns per sample, readBlock() 2.4 ns with AVX2 and 3.5 ns
 * with SSE2, phase accumulation included; 4.1 ns with AVX2 when the frame
 * position moves every sample.
 *
 * Storage is float whatever the reader's sample type, with guard samples
 * around every table so a read never wraps. A bank costs 229 KB per frame.
 * The four built-in banks (builtin()) are built once per process, on first
 * use, and shared.
 *
 * Threading: building (buildFromHarmonics, loadFrames) allocates and belongs
 * to a setup thread, never concurrent with reads of the same bank. A built
 * bank is immutable: any number of threads may read it concurrently.
 * builtin() is safe to call from several threads (function-local statics),
 * but its first call per bank builds it; call it during setup.
 *
 * Dependencies: DspMath.h, FFT.h, SimdOps.h.
 */

#include "DspMath.h"
#include "FFT.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace dspark {

namespace detail {

/// Niemitalo's 4x 6-point coefficients, three per polynomial coefficient
/// c0..c5, in the order WavetableBank's interpolator uses them.
inline constexpr std::array<double, 18> kWavetableNiemitalo = {
     0.41496902959240894,  0.08343081932889224,  0.00160015038681571,
     0.31625515004859783,  0.21197848565176958,  0.00956166668408054,
    -0.203271896548875371, 0.17989908432249280,  0.02337283412161328,
    -0.20209241069835732,  0.01760734419526000,  0.02985927012435252,
     0.04100948858761910, -0.06147760875085254,  0.02046802954581191,
     0.06607747864416924, -0.03255079211953620,  0.00628989632244913 };

/// One block's level pair, as WavetableBank's block readers see it.
struct WavetableRows
{
    bool two = false;                ///< Two levels are blended.
    bool same = true;                ///< Both levels share a grid.
    int numFrames = 0;
    double sizeA = 0.0, sizeB = 0.0;
    size_t strideA = 0, strideB = 0;
    const float* baseA = nullptr;    ///< Frame 0 at table offset -2, level a.
    const float* baseB = nullptr;
};

/// Frame row and blend of one position, exactly as WavetableBank::read()
/// derives them.
struct WavetableFrame
{
    size_t f0 = 0;      ///< Lower frame.
    size_t step = 0;    ///< 1 to the upper frame, 0 on the last one.
    double t = 0.0;     ///< Blend toward the upper frame.
};

[[nodiscard]] inline WavetableFrame wavetableFrame(double position, int numFrames) noexcept
{
    const double maxPos = static_cast<double>(numFrames - 1);
    if (!(position > 0.0)) position = 0.0;
    else if (position > maxPos) position = maxPos;
    const int f0 = static_cast<int>(position);
    WavetableFrame p;
    p.f0 = static_cast<size_t>(f0);
    p.step = (f0 < numFrames - 1) ? 1u : 0u;
    p.t = position - static_cast<double>(f0);
    return p;
}

/// Level layouts of a block: one level, two levels on one grid, two grids.
inline constexpr int kWavetableOneLevel = 0, kWavetableSameGrid = 1, kWavetableTwoGrids = 2;

} // namespace detail

// Narrow block kernels: four float or two double lanes, on SSE2 and NEON.
// Every operation is the one read() performs on the same values, in the
// same order, so the block reader returns read()'s results.
#if defined(DSPARK_SIMD_SSE2) || defined(DSPARK_SIMD_NEON)
    #define DSPARK_WAVETABLE_NARROW 1
namespace detail {
namespace wavetable_narrow {

#if defined(DSPARK_SIMD_SSE2)
using F4 = __m128;
using D2 = __m128d;

inline F4 add(F4 a, F4 b) noexcept { return _mm_add_ps(a, b); }
inline F4 sub(F4 a, F4 b) noexcept { return _mm_sub_ps(a, b); }
inline F4 mul(F4 a, F4 b) noexcept { return _mm_mul_ps(a, b); }
inline F4 set1(float x) noexcept { return _mm_set1_ps(x); }
inline F4 load(const float* p) noexcept { return _mm_loadu_ps(p); }
inline void store(float* p, F4 v) noexcept { _mm_storeu_ps(p, v); }
inline D2 add(D2 a, D2 b) noexcept { return _mm_add_pd(a, b); }
inline D2 sub(D2 a, D2 b) noexcept { return _mm_sub_pd(a, b); }
inline D2 mul(D2 a, D2 b) noexcept { return _mm_mul_pd(a, b); }
inline D2 set1(double x) noexcept { return _mm_set1_pd(x); }
inline void store(double* p, D2 v) noexcept { _mm_storeu_pd(p, v); }

/// Two floats widened to doubles.
inline D2 pair(const float* p) noexcept
{
    return _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}

/// True when both phases lie in [0, 1) (NaN fails).
inline bool inRange2(const double* ph) noexcept
{
    const __m128d v = _mm_loadu_pd(ph);
    const __m128d ok = _mm_and_pd(_mm_cmpge_pd(v, _mm_setzero_pd()), _mm_cmplt_pd(v, _mm_set1_pd(1.0)));
    return _mm_movemask_pd(ok) == 3;
}

/// Index (truncated) and fraction of two phases on a grid of `size`.
/// phase * size is exact (sizes are powers of two), so the index stays
/// below the size, as read()'s min() guarantees.
inline D2 grid2(const double* ph, double size, int32_t* idx) noexcept
{
    const __m128d p = _mm_mul_pd(_mm_loadu_pd(ph), _mm_set1_pd(size));
    const __m128i i = _mm_cvttpd_epi32(p);
    idx[0] = _mm_cvtsi128_si32(i);
    idx[1] = _mm_cvtsi128_si32(_mm_srli_si128(i, 4));
    return _mm_sub_pd(p, _mm_cvtepi32_pd(i));
}

/// Two fractions narrowed to float, as two lanes of a float vector's half.
inline F4 narrow4(D2 lo, D2 hi) noexcept
{
    return _mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi));
}

/// Frame index and blend of two positions, clamped as read() clamps them.
inline D2 frames2(const double* pos, double maxPos, int32_t* f0) noexcept
{
    const __m128d p = _mm_loadu_pd(pos);
    const __m128d c = _mm_and_pd(_mm_cmpgt_pd(p, _mm_setzero_pd()), _mm_min_pd(p, _mm_set1_pd(maxPos)));
    const __m128i i = _mm_cvttpd_epi32(c);
    f0[0] = _mm_cvtsi128_si32(i);
    f0[1] = _mm_cvtsi128_si32(_mm_srli_si128(i, 4));
    return _mm_sub_pd(c, _mm_cvtepi32_pd(i));
}

/// Four lanes' rows (lo: offsets -2..+1, hi: +2..+5) into six columns.
inline void columns4(const F4* lo, const F4* hi, F4* y) noexcept
{
    __m128 a = lo[0], b = lo[1], c = lo[2], d = lo[3];
    _MM_TRANSPOSE4_PS(a, b, c, d);
    y[0] = a;
    y[1] = b;
    y[2] = c;
    y[3] = d;
    const __m128 u = _mm_unpacklo_ps(hi[0], hi[1]);   // h0[0] h1[0] h0[1] h1[1]
    const __m128 v = _mm_unpacklo_ps(hi[2], hi[3]);   // h2[0] h3[0] h2[1] h3[1]
    y[4] = _mm_movelh_ps(u, v);
    y[5] = _mm_movehl_ps(v, u);
}

/// Two lanes' double rows (three pairs each) into six columns.
inline void columns2(const D2* r0, const D2* r1, D2* y) noexcept
{
    for (int k = 0; k < 3; ++k)
    {
        y[2 * k]     = _mm_unpacklo_pd(r0[k], r1[k]);
        y[2 * k + 1] = _mm_unpackhi_pd(r0[k], r1[k]);
    }
}
#else // NEON
using F4 = float32x4_t;
#if defined(_MSC_VER) && !defined(__clang__)
// MSVC declares float32x4_t and float64x2_t as one type (__n128), so the
// double lanes get a type of their own to keep the overloads apart.
struct D2 { float64x2_t v; };
inline D2 d2(float64x2_t x) noexcept { return D2{ x }; }
inline float64x2_t raw(D2 x) noexcept { return x.v; }
#else
using D2 = float64x2_t;
inline D2 d2(float64x2_t x) noexcept { return x; }
inline float64x2_t raw(D2 x) noexcept { return x; }
#endif

inline F4 add(F4 a, F4 b) noexcept { return vaddq_f32(a, b); }
inline F4 sub(F4 a, F4 b) noexcept { return vsubq_f32(a, b); }
inline F4 mul(F4 a, F4 b) noexcept { return vmulq_f32(a, b); }
inline F4 set1(float x) noexcept { return vdupq_n_f32(x); }
inline F4 load(const float* p) noexcept { return vld1q_f32(p); }
inline void store(float* p, F4 v) noexcept { vst1q_f32(p, v); }
inline D2 add(D2 a, D2 b) noexcept { return d2(vaddq_f64(raw(a), raw(b))); }
inline D2 sub(D2 a, D2 b) noexcept { return d2(vsubq_f64(raw(a), raw(b))); }
inline D2 mul(D2 a, D2 b) noexcept { return d2(vmulq_f64(raw(a), raw(b))); }
inline D2 set1(double x) noexcept { return d2(vdupq_n_f64(x)); }
inline void store(double* p, D2 v) noexcept { vst1q_f64(p, raw(v)); }

inline D2 pair(const float* p) noexcept { return d2(vcvt_f64_f32(vld1_f32(p))); }

inline bool inRange2(const double* ph) noexcept
{
    const float64x2_t v = vld1q_f64(ph);
    const uint64x2_t ok = vandq_u64(vcgeq_f64(v, vdupq_n_f64(0.0)), vcltq_f64(v, vdupq_n_f64(1.0)));
    return vgetq_lane_u64(ok, 0) != 0 && vgetq_lane_u64(ok, 1) != 0;
}

inline D2 grid2(const double* ph, double size, int32_t* idx) noexcept
{
    const float64x2_t p = vmulq_f64(vld1q_f64(ph), vdupq_n_f64(size));
    const int64x2_t i = vcvtq_s64_f64(p);   // rounds toward zero, as the cast
    idx[0] = static_cast<int32_t>(vgetq_lane_s64(i, 0));
    idx[1] = static_cast<int32_t>(vgetq_lane_s64(i, 1));
    return d2(vsubq_f64(p, vcvtq_f64_s64(i)));
}

inline F4 narrow4(D2 lo, D2 hi) noexcept
{
    return vcombine_f32(vcvt_f32_f64(raw(lo)), vcvt_f32_f64(raw(hi)));
}

inline D2 frames2(const double* pos, double maxPos, int32_t* f0) noexcept
{
    const float64x2_t p = vld1q_f64(pos);
    // FMIN returns NaN for a NaN input; the mask clears it to 0, as read().
    const uint64x2_t c = vandq_u64(vcgtq_f64(p, vdupq_n_f64(0.0)),
                                   vreinterpretq_u64_f64(vminq_f64(p, vdupq_n_f64(maxPos))));
    const float64x2_t cd = vreinterpretq_f64_u64(c);
    const int64x2_t i = vcvtq_s64_f64(cd);
    f0[0] = static_cast<int32_t>(vgetq_lane_s64(i, 0));
    f0[1] = static_cast<int32_t>(vgetq_lane_s64(i, 1));
    return d2(vsubq_f64(cd, vcvtq_f64_s64(i)));
}

inline void columns4(const F4* lo, const F4* hi, F4* y) noexcept
{
    const float32x4x2_t t01 = vtrnq_f32(lo[0], lo[1]);   // a0 b0 a2 b2 | a1 b1 a3 b3
    const float32x4x2_t t23 = vtrnq_f32(lo[2], lo[3]);   // c0 d0 c2 d2 | c1 d1 c3 d3
    y[0] = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    y[1] = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    y[2] = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    y[3] = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
    const float32x4x2_t h01 = vtrnq_f32(hi[0], hi[1]);
    const float32x4x2_t h23 = vtrnq_f32(hi[2], hi[3]);
    y[4] = vcombine_f32(vget_low_f32(h01.val[0]), vget_low_f32(h23.val[0]));
    y[5] = vcombine_f32(vget_low_f32(h01.val[1]), vget_low_f32(h23.val[1]));
}

inline void columns2(const D2* r0, const D2* r1, D2* y) noexcept
{
    for (int k = 0; k < 3; ++k)
    {
        y[2 * k]     = d2(vzip1q_f64(raw(r0[k]), raw(r1[k])));
        y[2 * k + 1] = d2(vzip2q_f64(raw(r0[k]), raw(r1[k])));
    }
}
#endif

/// Index and fraction (narrowed to float) of four phases on a grid.
inline F4 grid4(const double* ph, double size, int32_t* idx) noexcept
{
    const D2 lo = grid2(ph, size, idx);
    const D2 hi = grid2(ph + 2, size, idx + 2);
    return narrow4(lo, hi);
}

/// WavetableBank's interpolator on vector lanes of S, operation for operation.
template <typename S, typename V>
inline V interpolateLanes(const V* y, V x) noexcept
{
    const auto& k = kWavetableNiemitalo;
    const auto c = [&](size_t i) { return set1(static_cast<S>(k[i])); };
    const V z = sub(x, set1(S(0.5)));
    const V even1 = add(y[3], y[2]), odd1 = sub(y[3], y[2]);
    const V even2 = add(y[4], y[1]), odd2 = sub(y[4], y[1]);
    const V even3 = add(y[5], y[0]), odd3 = sub(y[5], y[0]);
    const auto sum3 = [&](V p, V q, V r, size_t i) {
        return add(add(mul(p, c(i)), mul(q, c(i + 1))), mul(r, c(i + 2)));
    };
    const V c0 = sum3(even1, even2, even3, 0), c1 = sum3(odd1, odd2, odd3, 3);
    const V c2 = sum3(even1, even2, even3, 6), c3 = sum3(odd1, odd2, odd3, 9);
    const V c4 = sum3(even1, even2, even3, 12), c5 = sum3(odd1, odd2, odd3, 15);
    V r = add(mul(c5, z), c4);
    r = add(mul(r, z), c3);
    r = add(mul(r, z), c2);
    r = add(mul(r, z), c1);
    return add(mul(r, z), c0);
}

/// read()'s frame blend a + t (b - a), per lane.
template <typename V, typename S>
inline V lerp(V a, V b, S t) noexcept
{
    return add(a, mul(set1(t), sub(b, a)));
}

} // namespace wavetable_narrow
} // namespace detail
#endif

// AVX2 block kernels, used when the CPU has AVX2 (selected at run time in
// x86-64 baseline builds; always in AVX2 builds). They are built in
// SimdOps.h's exact AVX2 region, without FMA, so they compute what read()
// computes, operation for operation.
#if defined(DSPARK_SIMD_EXACT_AVX2)
DSPARK_SIMD_EXACT_AVX2_BEGIN
namespace detail {
namespace wavetable_avx2 {

/// WavetableBank's interpolator on eight float lanes, operation for operation.
inline __m256 interpolate8(const __m256* y, __m256 x) noexcept
{
    const auto& k = kWavetableNiemitalo;
    const __m256 z = _mm256_sub_ps(x, _mm256_set1_ps(0.5f));
    const __m256 even1 = _mm256_add_ps(y[3], y[2]), odd1 = _mm256_sub_ps(y[3], y[2]);
    const __m256 even2 = _mm256_add_ps(y[4], y[1]), odd2 = _mm256_sub_ps(y[4], y[1]);
    const __m256 even3 = _mm256_add_ps(y[5], y[0]), odd3 = _mm256_sub_ps(y[5], y[0]);
    __m256 c[6];
    for (int i = 0; i < 6; ++i)
    {
        const __m256 p = (i % 2 == 0) ? even1 : odd1;
        const __m256 q = (i % 2 == 0) ? even2 : odd2;
        const __m256 r = (i % 2 == 0) ? even3 : odd3;
        const auto b = static_cast<size_t>(3 * i);
        c[i] = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(p, _mm256_set1_ps(static_cast<float>(k[b]))),
                                           _mm256_mul_ps(q, _mm256_set1_ps(static_cast<float>(k[b + 1])))),
                             _mm256_mul_ps(r, _mm256_set1_ps(static_cast<float>(k[b + 2]))));
    }
    __m256 v = _mm256_add_ps(_mm256_mul_ps(c[5], z), c[4]);
    v = _mm256_add_ps(_mm256_mul_ps(v, z), c[3]);
    v = _mm256_add_ps(_mm256_mul_ps(v, z), c[2]);
    v = _mm256_add_ps(_mm256_mul_ps(v, z), c[1]);
    return _mm256_add_ps(_mm256_mul_ps(v, z), c[0]);
}

/// The same on four double lanes.
inline __m256d interpolate4d(const __m256d* y, __m256d x) noexcept
{
    const auto& k = kWavetableNiemitalo;
    const __m256d z = _mm256_sub_pd(x, _mm256_set1_pd(0.5));
    const __m256d even1 = _mm256_add_pd(y[3], y[2]), odd1 = _mm256_sub_pd(y[3], y[2]);
    const __m256d even2 = _mm256_add_pd(y[4], y[1]), odd2 = _mm256_sub_pd(y[4], y[1]);
    const __m256d even3 = _mm256_add_pd(y[5], y[0]), odd3 = _mm256_sub_pd(y[5], y[0]);
    __m256d c[6];
    for (int i = 0; i < 6; ++i)
    {
        const __m256d p = (i % 2 == 0) ? even1 : odd1;
        const __m256d q = (i % 2 == 0) ? even2 : odd2;
        const __m256d r = (i % 2 == 0) ? even3 : odd3;
        const auto b = static_cast<size_t>(3 * i);
        c[i] = _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(p, _mm256_set1_pd(k[b])),
                                           _mm256_mul_pd(q, _mm256_set1_pd(k[b + 1]))),
                             _mm256_mul_pd(r, _mm256_set1_pd(k[b + 2])));
    }
    __m256d v = _mm256_add_pd(_mm256_mul_pd(c[5], z), c[4]);
    v = _mm256_add_pd(_mm256_mul_pd(v, z), c[3]);
    v = _mm256_add_pd(_mm256_mul_pd(v, z), c[2]);
    v = _mm256_add_pd(_mm256_mul_pd(v, z), c[1]);
    return _mm256_add_pd(_mm256_mul_pd(v, z), c[0]);
}

/// Eight lanes' rows (offsets -2..+5 of each interval) into six columns.
inline void columns8(const __m256* r, __m256* y) noexcept
{
    const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
    const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
    const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
    const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
    const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
    y[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    y[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    y[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    y[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    y[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    y[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
}

/// Indices and fractions of eight phases on a grid of `size` samples.
inline __m256 grid8(const double* phases, __m256d size, int32_t* idx) noexcept
{
    const __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(phases), size);
    const __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(phases + 4), size);
    const __m128i i0 = _mm256_cvttpd_epi32(p0), i1 = _mm256_cvttpd_epi32(p1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(idx + 4), i1);
    const __m128 x0 = _mm256_cvtpd_ps(_mm256_sub_pd(p0, _mm256_cvtepi32_pd(i0)));
    const __m128 x1 = _mm256_cvtpd_ps(_mm256_sub_pd(p1, _mm256_cvtepi32_pd(i1)));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(x0), x1, 1);
}

/// Frame rows and blends of eight positions, as wavetableFrame() derives them.
inline void frames8(const double* positions, int numFrames, int32_t* f0, float* t) noexcept
{
    const __m256d zero = _mm256_setzero_pd(), maxPos = _mm256_set1_pd(static_cast<double>(numFrames - 1));
    for (int h = 0; h < 2; ++h)
    {
        const __m256d p = _mm256_loadu_pd(positions + 4 * h);
        const __m256d c = _mm256_and_pd(_mm256_cmp_pd(p, zero, _CMP_GT_OQ), _mm256_min_pd(p, maxPos));
        const __m128i i = _mm256_cvttpd_epi32(c);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(f0 + 4 * h), i);
        _mm_storeu_ps(t + 4 * h, _mm256_cvtpd_ps(_mm256_sub_pd(c, _mm256_cvtepi32_pd(i))));
    }
}

/// The same for four positions, with the blends in double.
inline void frames4d(const double* positions, int numFrames, int32_t* f0, double* t) noexcept
{
    const __m256d zero = _mm256_setzero_pd(), maxPos = _mm256_set1_pd(static_cast<double>(numFrames - 1));
    const __m256d p = _mm256_loadu_pd(positions);
    const __m256d c = _mm256_and_pd(_mm256_cmp_pd(p, zero, _CMP_GT_OQ), _mm256_min_pd(p, maxPos));
    const __m128i i = _mm256_cvttpd_epi32(c);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(f0), i);
    _mm256_storeu_pd(t, _mm256_sub_pd(c, _mm256_cvtepi32_pd(i)));
}

/// One lane's eight table samples, blended toward the next frame by t.
inline __m256 row8(const float* p, size_t step, float t, bool blend) noexcept
{
    const __m256 a = _mm256_loadu_ps(p);
    if (!blend) return a;
    return _mm256_add_ps(a, _mm256_mul_ps(_mm256_set1_ps(t), _mm256_sub_ps(_mm256_loadu_ps(p + step), a)));
}

/**
 * @brief Float samples, eight at a time; every phase must lie in [0, 1).
 * @return Samples written (a multiple of eight).
 */
template <int Layout, bool PerSample, bool Blend>
int read8(const WavetableRows& s, const double* phases, const double* positions,
          const WavetableFrame& fixed, float w, float* out, int n) noexcept
{
    const __m256 wv = _mm256_set1_ps(w);
    const __m256d sizeA = _mm256_set1_pd(s.sizeA), sizeB = _mm256_set1_pd(s.sizeB);
    const float* fixedA = s.baseA + fixed.f0 * s.strideA;
    const float* fixedB = s.baseB + fixed.f0 * s.strideB;
    const auto fixedT = static_cast<float>(fixed.t);
    int j = 0;
    for (; j + 8 <= n; j += 8)
    {
        alignas(32) int32_t ia[8], ib[8], fa[8];
        alignas(32) float ta[8];
        if constexpr (PerSample) frames8(positions + j, s.numFrames, fa, ta);
        const __m256 xa = grid8(phases + j, sizeA, ia);
        __m256 xb = xa;
        if constexpr (Layout == kWavetableTwoGrids) xb = grid8(phases + j, sizeB, ib);
        __m256 ra[8], rb[8];
        for (int l = 0; l < 8; ++l)
        {
            const float* rowA = fixedA;
            const float* rowB = fixedB;
            size_t stepA = fixed.step * s.strideA, stepB = fixed.step * s.strideB;
            float t = fixedT;
            bool blend = Blend;
            if constexpr (PerSample)
            {
                const auto f0 = static_cast<size_t>(fa[l]);
                const bool upper = fa[l] < s.numFrames - 1;
                rowA = s.baseA + f0 * s.strideA;
                rowB = s.baseB + f0 * s.strideB;
                stepA = upper ? s.strideA : 0;
                stepB = upper ? s.strideB : 0;
                t = ta[l];
                blend = t != 0.0f;
            }
            ra[l] = row8(rowA + static_cast<size_t>(ia[l]), stepA, t, blend);
            if constexpr (Layout != kWavetableOneLevel)
            {
                const auto offB = static_cast<size_t>(Layout == kWavetableSameGrid ? ia[l] : ib[l]);
                rb[l] = row8(rowB + offB, stepB, t, blend);
            }
            if constexpr (Layout == kWavetableSameGrid)
                ra[l] = _mm256_add_ps(rb[l], _mm256_mul_ps(wv, _mm256_sub_ps(ra[l], rb[l])));
        }
        __m256 y[6];
        columns8(ra, y);
        __m256 r = interpolate8(y, xa);
        if constexpr (Layout == kWavetableTwoGrids)
        {
            columns8(rb, y);
            const __m256 sb = interpolate8(y, xb);
            r = _mm256_add_ps(sb, _mm256_mul_ps(wv, _mm256_sub_ps(r, sb)));
        }
        _mm256_storeu_ps(out + j, r);
    }
    return j;
}

/// One lane's six samples as doubles: y0..y3 and the pair y4, y5.
inline void row6d(const float* p, size_t step, double t, bool blend, __m256d& lo, __m128d& hi) noexcept
{
    lo = _mm256_cvtps_pd(_mm_loadu_ps(p));
    hi = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 4))));
    if (!blend) return;
    const __m256d lo1 = _mm256_cvtps_pd(_mm_loadu_ps(p + step));
    const __m128d hi1 = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + step + 4))));
    lo = _mm256_add_pd(lo, _mm256_mul_pd(_mm256_set1_pd(t), _mm256_sub_pd(lo1, lo)));
    hi = _mm_add_pd(hi, _mm_mul_pd(_mm_set1_pd(t), _mm_sub_pd(hi1, hi)));
}

/// Four lanes' double rows into six columns.
inline void columns4d(const __m256d* lo, const __m128d* hi, __m256d* y) noexcept
{
    const __m256d t0 = _mm256_unpacklo_pd(lo[0], lo[1]), t1 = _mm256_unpackhi_pd(lo[0], lo[1]);
    const __m256d t2 = _mm256_unpacklo_pd(lo[2], lo[3]), t3 = _mm256_unpackhi_pd(lo[2], lo[3]);
    y[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
    y[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
    y[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
    y[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
    const __m128d u0 = _mm_unpacklo_pd(hi[0], hi[1]), v0 = _mm_unpacklo_pd(hi[2], hi[3]);
    const __m128d u1 = _mm_unpackhi_pd(hi[0], hi[1]), v1 = _mm_unpackhi_pd(hi[2], hi[3]);
    y[4] = _mm256_insertf128_pd(_mm256_castpd128_pd256(u0), v0, 1);
    y[5] = _mm256_insertf128_pd(_mm256_castpd128_pd256(u1), v1, 1);
}

/// Indices and fractions of four phases on a grid of `size` samples.
inline __m256d grid4d(const double* phases, __m256d size, int32_t* idx) noexcept
{
    const __m256d p = _mm256_mul_pd(_mm256_loadu_pd(phases), size);
    const __m128i i = _mm256_cvttpd_epi32(p);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
    return _mm256_sub_pd(p, _mm256_cvtepi32_pd(i));
}

/**
 * @brief Double samples, four at a time; every phase must lie in [0, 1).
 * @return Samples written (a multiple of four).
 */
template <int Layout, bool PerSample, bool Blend>
int read4d(const WavetableRows& s, const double* phases, const double* positions,
           const WavetableFrame& fixed, double w, double* out, int n) noexcept
{
    const __m256d wv = _mm256_set1_pd(w);
    const __m128d wv2 = _mm_set1_pd(w);
    const __m256d sizeA = _mm256_set1_pd(s.sizeA), sizeB = _mm256_set1_pd(s.sizeB);
    const float* fixedA = s.baseA + fixed.f0 * s.strideA;
    const float* fixedB = s.baseB + fixed.f0 * s.strideB;
    int j = 0;
    for (; j + 4 <= n; j += 4)
    {
        alignas(16) int32_t ia[4], ib[4], fa[4];
        alignas(32) double ta[4];
        if constexpr (PerSample) frames4d(positions + j, s.numFrames, fa, ta);
        const __m256d xa = grid4d(phases + j, sizeA, ia);
        __m256d xb = xa;
        if constexpr (Layout == kWavetableTwoGrids) xb = grid4d(phases + j, sizeB, ib);
        __m256d loA[4], loB[4];
        __m128d hiA[4], hiB[4];
        for (int l = 0; l < 4; ++l)
        {
            const float* rowA = fixedA;
            const float* rowB = fixedB;
            size_t stepA = fixed.step * s.strideA, stepB = fixed.step * s.strideB;
            double t = fixed.t;
            bool blend = Blend;
            if constexpr (PerSample)
            {
                const auto f0 = static_cast<size_t>(fa[l]);
                const bool upper = fa[l] < s.numFrames - 1;
                rowA = s.baseA + f0 * s.strideA;
                rowB = s.baseB + f0 * s.strideB;
                stepA = upper ? s.strideA : 0;
                stepB = upper ? s.strideB : 0;
                t = ta[l];
                blend = t != 0.0;
            }
            row6d(rowA + static_cast<size_t>(ia[l]), stepA, t, blend, loA[l], hiA[l]);
            if constexpr (Layout != kWavetableOneLevel)
            {
                const auto offB = static_cast<size_t>(Layout == kWavetableSameGrid ? ia[l] : ib[l]);
                row6d(rowB + offB, stepB, t, blend, loB[l], hiB[l]);
            }
            if constexpr (Layout == kWavetableSameGrid)
            {
                loA[l] = _mm256_add_pd(loB[l], _mm256_mul_pd(wv, _mm256_sub_pd(loA[l], loB[l])));
                hiA[l] = _mm_add_pd(hiB[l], _mm_mul_pd(wv2, _mm_sub_pd(hiA[l], hiB[l])));
            }
        }
        __m256d y[6];
        columns4d(loA, hiA, y);
        __m256d r = interpolate4d(y, xa);
        if constexpr (Layout == kWavetableTwoGrids)
        {
            columns4d(loB, hiB, y);
            const __m256d sb = interpolate4d(y, xb);
            r = _mm256_add_pd(sb, _mm256_mul_pd(wv, _mm256_sub_pd(r, sb)));
        }
        _mm256_storeu_pd(out + j, r);
    }
    return j;
}

} // namespace wavetable_avx2
} // namespace detail
DSPARK_SIMD_EXACT_AVX2_END
#endif

/**
 * @class WavetableBank
 * @brief Mipmapped multi-frame wavetable storage and its morphing reader.
 *
 * @code
 *   const auto& bank = dspark::WavetableBank::builtin(dspark::WavetableBank::Builtin::Basic);
 *   auto mip = bank.selectMip(frequencyHz, sampleRate);    // on pitch changes
 *   double phase = 0.0;
 *   double phases[64];                                     // n <= 64
 *   for (int i = 0; i < n; ++i)
 *   {
 *       phases[i] = phase;
 *       phase += frequencyHz / sampleRate;
 *       if (phase >= 1.0) phase -= 1.0;
 *   }
 *   bank.readBlock<float>(phases, position, mip, out, n);  // position 0..frames-1
 * @endcode
 */
class WavetableBank
{
public:
    /// Mip levels, half an octave apart.
    static constexpr int kMipLevels = 20;
    /// Harmonics of the richest level.
    static constexpr int kMaxHarmonics = 1023;
    /// Most frames a bank holds.
    static constexpr int kMaxFrames = 256;
    /// Largest single-cycle frame loadFrames() analyses.
    static constexpr int kMaxFrameSize = 1 << 20;

    /** @brief Built-in banks (see builtin()). */
    enum class Builtin
    {
        Basic,     ///< 6 frames: sine, triangle, saw, square, 25% and 12.5% pulse.
        Harmonic,  ///< 8 frames: 1/h^p spectra from p = 3 (near sine) to p = 0.5 (bright).
        Formant,   ///< 9 frames: vowels A, E, I, O, U and the four transitions between them.
        Fm         ///< 8 frames: sin(x + b sin x) with the index b rising from 0 to 6.
    };
    static constexpr int kNumBuiltins = 4;

    /** @brief How a bank's frames are scaled. */
    enum class Normalization
    {
        PerFrame,  ///< Every frame peaks at 1 (equal peak level across a morph).
        Bank       ///< One factor for the whole bank: the frames keep their relative levels.
    };

    /** @brief One harmonic's cosine and sine amplitudes. */
    struct Harmonic
    {
        double cosine = 0.0;
        double sine = 0.0;
    };

    /** @brief The level pair and crossfade weight for one pitch (selectMip()). */
    struct Mip
    {
        int level = 0;        ///< Brighter level.
        int next = 0;         ///< Duller level it fades into.
        float weight = 1.0f;  ///< Weight of `level`; 1 reads it alone.
    };

    WavetableBank() = default;

    // -- Building (setup thread; allocates) ------------------------------------

    /**
     * @brief Builds the bank from per-frame harmonic amplitudes.
     *
     * @param numFrames    Frame count, 1..kMaxFrames.
     * @param numHarmonics Highest harmonic the frames hold, 1..kMaxHarmonics.
     * @param coefficients Callable `Harmonic(int frame, int harmonic)` for
     *                     harmonic 1..numHarmonics (the DC is not stored).
     * @param normalization Frame scaling.
     * @return False (bank unchanged) on an invalid count or a non-finite
     *         coefficient, or when every frame is silent.
     */
    template <typename Coefficients>
    bool buildFromHarmonics(int numFrames, int numHarmonics, Coefficients&& coefficients,
                            Normalization normalization = Normalization::PerFrame)
    {
        if (numFrames < 1 || numFrames > kMaxFrames || numHarmonics < 1)
            return false;
        numHarmonics = std::min(numHarmonics, kMaxHarmonics);

        std::vector<Harmonic> coeffs(static_cast<size_t>(numFrames) * static_cast<size_t>(numHarmonics));
        for (int f = 0; f < numFrames; ++f)
            for (int h = 1; h <= numHarmonics; ++h)
            {
                const Harmonic c = coefficients(f, h);
                if (!std::isfinite(c.cosine) || !std::isfinite(c.sine)) return false;
                coeffs[static_cast<size_t>(f) * static_cast<size_t>(numHarmonics)
                       + static_cast<size_t>(h - 1)] = c;
            }
        return build(numFrames, numHarmonics, coeffs, normalization);
    }

    /**
     * @brief Builds the bank from consecutive single-cycle frames.
     *
     * Each frame is analysed exactly (an FFT for power-of-two sizes, a direct
     * transform otherwise); its DC is discarded, and so is the Nyquist bin of
     * an even size. A frame of N samples yields (N - 1) / 2 harmonics, at
     * most kMaxHarmonics. Non-finite samples count as 0.
     *
     * @param data          numFrames * frameSize samples, frame after frame.
     * @param frameSize     Samples per cycle, 3..kMaxFrameSize.
     * @param numFrames     Frame count, 1..kMaxFrames.
     * @param normalization Frame scaling (Bank keeps the frames' relative levels).
     * @return False (bank unchanged) on invalid arguments or silent input.
     */
    template <FloatType S>
    bool loadFrames(const S* data, int frameSize, int numFrames,
                    Normalization normalization = Normalization::Bank)
    {
        if (data == nullptr || frameSize < 3 || frameSize > kMaxFrameSize
            || numFrames < 1 || numFrames > kMaxFrames)
            return false;
        const int numHarmonics = std::min((frameSize - 1) / 2, kMaxHarmonics);
        std::vector<Harmonic> coeffs(static_cast<size_t>(numFrames) * static_cast<size_t>(numHarmonics));
        std::vector<double> cycle(static_cast<size_t>(frameSize));
        const bool pow2 = (frameSize & (frameSize - 1)) == 0 && frameSize >= 4;
        std::vector<double> spec(pow2 ? static_cast<size_t>(frameSize + 2) : 0);
        FFTReal<double> fft(pow2 ? static_cast<size_t>(frameSize) : 4);

        for (int f = 0; f < numFrames; ++f)
        {
            const S* src = data + static_cast<size_t>(f) * static_cast<size_t>(frameSize);
            for (int i = 0; i < frameSize; ++i)
                cycle[static_cast<size_t>(i)] = std::isfinite(src[i]) ? static_cast<double>(src[i]) : 0.0;
            Harmonic* dst = coeffs.data() + static_cast<size_t>(f) * static_cast<size_t>(numHarmonics);
            const double scale = 2.0 / static_cast<double>(frameSize);
            if (pow2)
            {
                // X[h] = (N/2) (a_h - j b_h) for x = sum a_h cos + b_h sin.
                fft.forward(cycle.data(), spec.data());
                for (int h = 1; h <= numHarmonics; ++h)
                    dst[h - 1] = { spec[static_cast<size_t>(2 * h)] * scale,
                                   -spec[static_cast<size_t>(2 * h + 1)] * scale };
            }
            else
            {
                for (int h = 1; h <= numHarmonics; ++h)
                {
                    double sumCos = 0.0, sumSin = 0.0;
                    for (int i = 0; i < frameSize; ++i)
                    {
                        // Integer phase reduction keeps the argument exact.
                        const auto k = static_cast<long long>(h) * i % frameSize;
                        const double ph = twoPi<double> * static_cast<double>(k)
                                        / static_cast<double>(frameSize);
                        sumCos += cycle[static_cast<size_t>(i)] * std::cos(ph);
                        sumSin += cycle[static_cast<size_t>(i)] * std::sin(ph);
                    }
                    dst[h - 1] = { sumCos * scale, sumSin * scale };
                }
            }
        }
        return build(numFrames, numHarmonics, coeffs, normalization);
    }

    /**
     * @brief A built-in bank, built on the first call and shared by the process.
     * @note The first call per bank allocates about 1.4 to 2.1 MB and runs the
     *       synthesis: make it from a setup thread.
     */
    [[nodiscard]] static const WavetableBank& builtin(Builtin which)
    {
        switch (which)
        {
            case Builtin::Basic:    { static const WavetableBank b = makeBasic();    return b; }
            case Builtin::Harmonic: { static const WavetableBank b = makeHarmonic(); return b; }
            case Builtin::Formant:  { static const WavetableBank b = makeFormant();  return b; }
            case Builtin::Fm:       { static const WavetableBank b = makeFm();       return b; }
        }
        static const WavetableBank empty;
        return empty;
    }

    // -- Queries -----------------------------------------------------------------

    /** @brief Frames in the bank (0 before a successful build). */
    [[nodiscard]] int getNumFrames() const noexcept { return numFrames_; }

    /** @brief Highest harmonic the frames hold (0 before a build). */
    [[nodiscard]] int getNumHarmonics() const noexcept { return levelHarmonics_[0]; }

    /** @brief Bytes of table storage. */
    [[nodiscard]] size_t getMemoryBytes() const noexcept { return data_.size() * sizeof(float); }

    // -- Playback (real-time safe, any thread) ----------------------------------

    /**
     * @brief Chooses the level pair and crossfade weight for a pitch.
     *
     * Call it when the frequency changes (a synthesizer voice does at its
     * control rate), not per sample. A negative frequency selects as its
     * magnitude; a non-finite one selects the dullest level.
     */
    [[nodiscard]] Mip selectMip(double frequencyHz, double sampleRate) const noexcept
    {
        Mip m;
        if (numFrames_ == 0 || !(sampleRate > 0.0)) return m;
        const double f = std::isfinite(frequencyHz) ? std::abs(frequencyHz) : 1e300;
        // Highest harmonic frequency whose alias still lands above 20 kHz
        // (never below Nyquist itself: at low rates there is no free band).
        const double fold = std::max(sampleRate * 0.5, sampleRate - 20000.0);
        const int last = kMipLevels - 1;
        int level = 0;
        while (level < last && f > fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level)]))
            ++level;
        m.level = level;
        m.next = std::min(level + 1, last);
        if (m.next == level) return m;

        // Bottom of this level's range: the previous level's limit, or, for
        // level 0, one level-spacing below its own limit.
        const double hi = fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level)]);
        const double lo = (level > 0)
            ? fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level - 1)])
            : hi * static_cast<double>(levelHarmonics_[1]) / static_cast<double>(levelHarmonics_[0]);
        const double span = hi - lo;
        m.weight = (span > 0.0) ? static_cast<float>(std::clamp((hi - f) / span, 0.0, 1.0)) : 1.0f;
        return m;
    }

    /**
     * @brief One sample of the bank.
     * @param phase    Cycle phase in [0, 1) (wrapped if outside).
     * @param position Frame position in [0, frames - 1], blended linearly
     *                 between adjacent frames (clamped).
     * @param mip      Selection from selectMip() for the current pitch.
     * @return 0 for an empty bank.
     */
    template <FloatType T>
    [[nodiscard]] T read(double phase, double position, const Mip& mip) const noexcept
    {
        if (numFrames_ == 0) return T(0);
        if (!(phase >= 0.0 && phase < 1.0))
        {
            phase -= std::floor(phase);
            if (!(phase >= 0.0 && phase < 1.0)) phase = 0.0;
        }
        const double maxPos = static_cast<double>(numFrames_ - 1);
        if (!(position > 0.0)) position = 0.0;          // NaN included
        else if (position > maxPos) position = maxPos;
        const int f0 = static_cast<int>(position);
        const int f1 = (f0 < numFrames_ - 1) ? f0 + 1 : f0;
        const T t = static_cast<T>(position - static_cast<double>(f0));

        const int a = std::clamp(mip.level, 0, kMipLevels - 1);
        const int b = std::clamp(mip.next, 0, kMipLevels - 1);
        const T w = static_cast<T>(std::clamp(mip.weight, 0.0f, 1.0f));

        T ya[6];
        const T xa = gather<T>(a, f0, f1, t, phase, ya);
        if (a == b || w >= T(1))
            return interpolate(ya, xa);

        T yb[6];
        const T xb = gather<T>(b, f0, f1, t, phase, yb);
        if (levelSize_[static_cast<size_t>(a)] == levelSize_[static_cast<size_t>(b)])
        {
            // Same grid: blend the samples, interpolate once.
            for (int k = 0; k < 6; ++k) ya[k] = yb[k] + w * (ya[k] - yb[k]);
            return interpolate(ya, xa);
        }
        const T sa = interpolate(ya, xa);
        const T sb = interpolate(yb, xb);
        return sb + w * (sa - sb);
    }

    /**
     * @brief A block of samples at one frame position: read() at each phase.
     *
     * The values of read() sample by sample (bit for bit, unless the
     * compiler contracts the scalar arithmetic into fused multiply-adds) at
     * a fraction of its cost: the level and frame setup is done once per
     * block and, on SSE2 targets, four float or two double samples are
     * interpolated at once. Real-time safe.
     *
     * @param phases   `n` cycle phases in [0, 1) (wrapped if outside).
     * @param position Frame position of every sample (clamped as by read()).
     * @param mip      Selection from selectMip() for the current pitch.
     * @param out      `n` output samples (0 for an empty bank).
     * @param n        Sample count; nothing happens for n <= 0.
     */
    template <FloatType T>
    void readBlock(const double* phases, double position, const Mip& mip, T* out, int n) const noexcept
    {
        readBlockImpl<T>(phases, nullptr, position, mip, out, n);
    }

    /**
     * @brief A block of samples with one frame position per sample.
     * @param phases    `n` cycle phases in [0, 1) (wrapped if outside).
     * @param positions `n` frame positions (clamped as by read()).
     * @param mip       Selection from selectMip() for the current pitch.
     * @param out       `n` output samples (0 for an empty bank).
     * @param n         Sample count; nothing happens for n <= 0.
     */
    template <FloatType T>
    void readBlock(const double* phases, const double* positions, const Mip& mip,
                   T* out, int n) const noexcept
    {
        readBlockImpl<T>(phases, positions, 0.0, mip, out, n);
    }

    /**
     * @brief Amplitude response of the reader's interpolator at a frequency
     *        in cycles per table sample (the value pre-emphasis divides by).
     */
    [[nodiscard]] static double interpolatorResponse(double cyclesPerSample) noexcept
    {
        // The kernel is even, so H(v) = sum_k int_0^1 f_k(x) cos(2 pi v (x - k)) dx,
        // f_k being the weight of sample k (offsets -2..3) at fraction x.
        // 16-point Gauss-Legendre on each unit piece: the integrand is a
        // degree-5 polynomial times a slow cosine.
        static constexpr std::array<double, 8> node = {
            0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
            0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499 };
        static constexpr std::array<double, 8> weight = {
            0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
            0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541 };
        double sum = 0.0;
        for (int k = -2; k <= 3; ++k)
        {
            double y[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            y[k + 2] = 1.0;
            for (size_t q = 0; q < node.size(); ++q)
                for (const double s : { -1.0, 1.0 })
                {
                    const double x = 0.5 + 0.5 * s * node[q];
                    sum += 0.5 * weight[q] * interpolate(y, x)
                         * std::cos(twoPi<double> * cyclesPerSample * (x - static_cast<double>(k)));
                }
        }
        return sum;
    }

private:
    /// Fewest samples a level table holds.
    static constexpr int kMinTableSize = 2048;
    /// Guard samples on each side of every table (the reader spans -2..+3).
    static constexpr int kGuard = 3;

    /// Harmonic budget per level, half an octave apart from the top down to
    /// a single sine (the same budgets as WavetableOscillator).
    static constexpr std::array<int, kMipLevels> kLevelHarmonics = {
        kMaxHarmonics, 724, 512, 362, 256, 181, 128, 90, 64, 45,
        32, 22, 16, 11, 8, 5, 4, 3, 2, 1 };

    /**
     * @brief Niemitalo's optimal 4x 6-point 5th-order interpolator (z-form).
     * @param y Samples at offsets -2..3 around the interval.
     * @param x Fraction in [0, 1) between y[2] and y[3].
     */
    template <FloatType T>
    [[nodiscard]] static T interpolate(const T* y, T x) noexcept
    {
        const auto& k = kNiemitalo;
        const T z = x - T(0.5);
        const T even1 = y[3] + y[2], odd1 = y[3] - y[2];
        const T even2 = y[4] + y[1], odd2 = y[4] - y[1];
        const T even3 = y[5] + y[0], odd3 = y[5] - y[0];
        const T c0 = even1 * T(k[0]) + even2 * T(k[1]) + even3 * T(k[2]);
        const T c1 = odd1 * T(k[3]) + odd2 * T(k[4]) + odd3 * T(k[5]);
        const T c2 = even1 * T(k[6]) + even2 * T(k[7]) + even3 * T(k[8]);
        const T c3 = odd1 * T(k[9]) + odd2 * T(k[10]) + odd3 * T(k[11]);
        const T c4 = even1 * T(k[12]) + even2 * T(k[13]) + even3 * T(k[14]);
        const T c5 = odd1 * T(k[15]) + odd2 * T(k[16]) + odd3 * T(k[17]);
        return ((((c5 * z + c4) * z + c3) * z + c2) * z + c1) * z + c0;
    }

    /// Niemitalo's coefficients in the order interpolate() uses them.
    static constexpr const std::array<double, 18>& kNiemitalo = detail::kWavetableNiemitalo;

    /// Floats past the last table that the block reader may load: its float
    /// path reads eight samples per interval (offsets -2..+5; interpolation
    /// uses -2..+3), which reaches into the next row's guard, and past the end
    /// of the storage for the very last row.
    static constexpr size_t kTailPad = 2;

    /** Block reader: setup once, then vector or scalar samples. */
    template <FloatType T>
    void readBlockImpl(const double* phases, const double* positions, double position,
                       const Mip& mip, T* out, int n) const noexcept
    {
        if (n <= 0 || out == nullptr) return;
        if (numFrames_ == 0 || phases == nullptr)
        {
            std::fill_n(out, n, T(0));
            return;
        }
        int done = 0;
#if defined(DSPARK_SIMD_EXACT_AVX2)
    #if defined(DSPARK_SIMD_EXACT_AVX2_DISPATCHED)
        if (simd::isRuntimeDispatchActive())
    #endif
            done = readBlockAvx2(phases, positions, position, mip, out, n);
#endif
#if defined(DSPARK_WAVETABLE_NARROW)
        done += readBlockNarrow(phases + done, positions != nullptr ? positions + done : nullptr,
                                position, mip, out + done, n - done);
#endif
        for (int j = done; j < n; ++j)
            out[j] = read<T>(phases[j], positions != nullptr ? positions[j] : position, mip);
    }

#if defined(DSPARK_WAVETABLE_NARROW)
    using BlockSetup = detail::WavetableRows;
    using FramePick = detail::WavetableFrame;
    static constexpr int kOneLevel = detail::kWavetableOneLevel;
    static constexpr int kSameGrid = detail::kWavetableSameGrid;
    static constexpr int kTwoGrids = detail::kWavetableTwoGrids;

    template <FloatType T>
    [[nodiscard]] BlockSetup blockSetup(const Mip& mip, T w) const noexcept
    {
        BlockSetup s;
        s.numFrames = numFrames_;
        const int a = std::clamp(mip.level, 0, kMipLevels - 1);
        const int b = std::clamp(mip.next, 0, kMipLevels - 1);
        s.two = !(a == b || w >= T(1));
        const auto la = static_cast<size_t>(a), lb = static_cast<size_t>(b);
        s.same = levelSize_[la] == levelSize_[lb];
        s.sizeA = static_cast<double>(levelSize_[la]);
        s.sizeB = static_cast<double>(levelSize_[lb]);
        s.strideA = static_cast<size_t>(levelSize_[la] + 2 * kGuard);
        s.strideB = static_cast<size_t>(levelSize_[lb] + 2 * kGuard);
        s.baseA = data_.data() + levelOffset_[la] + static_cast<size_t>(kGuard - 2);
        s.baseB = data_.data() + levelOffset_[lb] + static_cast<size_t>(kGuard - 2);
        return s;
    }

    [[nodiscard]] FramePick framePick(double position) const noexcept
    {
        return detail::wavetableFrame(position, numFrames_);
    }

#if defined(DSPARK_SIMD_EXACT_AVX2)
    /// True when every phase lies in [0, 1) (NaN fails).
    [[nodiscard]] static bool allPhasesInRange(const double* phases, int n) noexcept
    {
        __m128d ok = _mm_castsi128_pd(_mm_set1_epi32(-1));
        const __m128d zero = _mm_setzero_pd(), one = _mm_set1_pd(1.0);
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            const __m128d ph = _mm_loadu_pd(phases + j);
            ok = _mm_and_pd(ok, _mm_and_pd(_mm_cmpge_pd(ph, zero), _mm_cmplt_pd(ph, one)));
        }
        bool inRange = _mm_movemask_pd(ok) == 3;
        for (; j < n; ++j) inRange = inRange && phases[j] >= 0.0 && phases[j] < 1.0;
        return inRange;
    }

    /// The AVX2 kernels over the block's whole multiple of their width, when
    /// every phase is in range (else 0: the SSE2 path handles the block).
    template <FloatType T>
    int readBlockAvx2(const double* phases, const double* positions, double position,
                      const Mip& mip, T* out, int n) const noexcept
    {
        namespace k = detail::wavetable_avx2;
        if (!allPhasesInRange(phases, n)) return 0;
        const float wf = std::clamp(mip.weight, 0.0f, 1.0f);
        const auto w = static_cast<T>(wf);
        const BlockSetup s = blockSetup<T>(mip, w);
        const int layout = !s.two ? kOneLevel : (s.same ? kSameGrid : kTwoGrids);
        const FramePick f = framePick(position);
        const bool blend = static_cast<T>(f.t) != T(0);
        const auto run = [&](auto layoutTag, auto perSampleTag, auto blendTag) {
            constexpr int L = decltype(layoutTag)::value;
            constexpr bool P = decltype(perSampleTag)::value;
            constexpr bool B = decltype(blendTag)::value;
            if constexpr (std::is_same_v<T, float>)
                return k::read8<L, P, B>(s, phases, positions, f, w, out, n);
            else
                return k::read4d<L, P, B>(s, phases, positions, f, w, out, n);
        };
        using One = std::integral_constant<int, kOneLevel>;
        using Same = std::integral_constant<int, kSameGrid>;
        using Two = std::integral_constant<int, kTwoGrids>;
        using Yes = std::true_type;
        using No = std::false_type;
        if (positions != nullptr)
        {
            if (layout == kOneLevel) return run(One {}, Yes {}, Yes {});
            if (layout == kSameGrid) return run(Same {}, Yes {}, Yes {});
            return run(Two {}, Yes {}, Yes {});
        }
        if (blend)
        {
            if (layout == kOneLevel) return run(One {}, No {}, Yes {});
            if (layout == kSameGrid) return run(Same {}, No {}, Yes {});
            return run(Two {}, No {}, Yes {});
        }
        if (layout == kOneLevel) return run(One {}, No {}, No {});
        if (layout == kSameGrid) return run(Same {}, No {}, No {});
        return run(Two {}, No {}, No {});
    }
#endif

    /**
     * @brief The narrow kernels for one block: picks the loop specialised for
     *        the level layout, per-sample positions and the frame blend.
     * @return Samples written (a multiple of the lane count).
     */
    template <FloatType T>
    int readBlockNarrow(const double* phases, const double* positions, double position,
                        const Mip& mip, T* out, int n) const noexcept
    {
        const auto w = static_cast<T>(std::clamp(mip.weight, 0.0f, 1.0f));
        const BlockSetup s = blockSetup<T>(mip, w);
        const int layout = !s.two ? kOneLevel : (s.same ? kSameGrid : kTwoGrids);
        const FramePick f = framePick(position);
        const auto run = [&](auto layoutTag, auto perSampleTag, auto blendTag) {
            constexpr int L = decltype(layoutTag)::value;
            constexpr bool P = decltype(perSampleTag)::value;
            constexpr bool B = decltype(blendTag)::value;
            if constexpr (std::is_same_v<T, float>)
                return this->template readFloat4<L, P, B>(phases, positions, position, s, f, w, mip, out, n);
            else
                return this->template readDouble2<L, P, B>(phases, positions, position, s, f, w, mip, out, n);
        };
        using One = std::integral_constant<int, kOneLevel>;
        using Same = std::integral_constant<int, kSameGrid>;
        using Two = std::integral_constant<int, kTwoGrids>;
        using Yes = std::true_type;
        using No = std::false_type;
        if (positions != nullptr)
        {
            if (layout == kOneLevel) return run(One {}, Yes {}, Yes {});
            if (layout == kSameGrid) return run(Same {}, Yes {}, Yes {});
            return run(Two {}, Yes {}, Yes {});
        }
        if (static_cast<T>(f.t) != T(0))
        {
            if (layout == kOneLevel) return run(One {}, No {}, Yes {});
            if (layout == kSameGrid) return run(Same {}, No {}, Yes {});
            return run(Two {}, No {}, Yes {});
        }
        if (layout == kOneLevel) return run(One {}, No {}, No {});
        if (layout == kSameGrid) return run(Same {}, No {}, No {});
        return run(Two {}, No {}, No {});
    }

    /**
     * @brief Four float samples per step for one level layout.
     * @tparam Layout    kOneLevel, kSameGrid or kTwoGrids.
     * @tparam PerSample Positions per sample (else `fixed` for the block).
     * @tparam Blend     Frame blend of the fixed position (t not 0).
     */
    template <int Layout, bool PerSample, bool Blend>
    int readFloat4(const double* phases, const double* positions, double position,
                   const BlockSetup& s, const FramePick& fixed, float w, const Mip& mip,
                   float* out, int n) const noexcept
    {
        namespace v = detail::wavetable_narrow;
        const v::F4 wv = v::set1(w);
        const double maxPos = static_cast<double>(numFrames_ - 1);
        const float* fixedA = s.baseA + fixed.f0 * s.strideA;
        const float* fixedB = s.baseB + fixed.f0 * s.strideB;
        const auto fixedT = static_cast<float>(fixed.t);
        int j = 0;
        for (; j + 4 <= n; j += 4)
        {
            if (!v::inRange2(phases + j) || !v::inRange2(phases + j + 2))
            {
                for (int l = 0; l < 4; ++l)
                    out[j + l] = read<float>(phases[j + l], PerSample ? positions[j + l] : position, mip);
                continue;
            }
            alignas(16) int32_t ia[4], ib[4], fa[4];
            alignas(16) float ta[4];
            const v::F4 xa = v::grid4(phases + j, s.sizeA, ia);
            v::F4 xb = xa;
            if constexpr (Layout == kTwoGrids) xb = v::grid4(phases + j, s.sizeB, ib);
            if constexpr (PerSample)
                v::store(ta, v::narrow4(v::frames2(positions + j, maxPos, fa),
                                        v::frames2(positions + j + 2, maxPos, fa + 2)));

            v::F4 loA[4], hiA[4], loB[4], hiB[4];
            for (int l = 0; l < 4; ++l)
            {
                const float* rowA = fixedA;
                const float* rowB = fixedB;
                size_t stepA = fixed.step * s.strideA, stepB = fixed.step * s.strideB;
                float t = fixedT;
                bool blend = Blend;
                if constexpr (PerSample)
                {
                    const auto f0 = static_cast<size_t>(fa[l]);
                    const bool upper = fa[l] < numFrames_ - 1;
                    rowA = s.baseA + f0 * s.strideA;
                    rowB = s.baseB + f0 * s.strideB;
                    stepA = upper ? s.strideA : 0;
                    stepB = upper ? s.strideB : 0;
                    t = ta[l];
                    blend = t != 0.0f;
                }
                const float* pa = rowA + static_cast<size_t>(ia[l]);
                loA[l] = v::load(pa);
                hiA[l] = v::load(pa + 4);
                if (blend)
                {
                    loA[l] = v::lerp(loA[l], v::load(pa + stepA), t);
                    hiA[l] = v::lerp(hiA[l], v::load(pa + stepA + 4), t);
                }
                if constexpr (Layout != kOneLevel)
                {
                    const float* pb = rowB + static_cast<size_t>(Layout == kSameGrid ? ia[l] : ib[l]);
                    loB[l] = v::load(pb);
                    hiB[l] = v::load(pb + 4);
                    if (blend)
                    {
                        loB[l] = v::lerp(loB[l], v::load(pb + stepB), t);
                        hiB[l] = v::lerp(hiB[l], v::load(pb + stepB + 4), t);
                    }
                }
                if constexpr (Layout == kSameGrid)
                {
                    // Same grid: blend the samples, interpolate once.
                    loA[l] = v::add(loB[l], v::mul(wv, v::sub(loA[l], loB[l])));
                    hiA[l] = v::add(hiB[l], v::mul(wv, v::sub(hiA[l], hiB[l])));
                }
            }
            v::F4 y[6];
            v::columns4(loA, hiA, y);
            v::F4 r = v::interpolateLanes<float>(y, xa);
            if constexpr (Layout == kTwoGrids)
            {
                v::columns4(loB, hiB, y);
                const v::F4 rb = v::interpolateLanes<float>(y, xb);
                r = v::add(rb, v::mul(wv, v::sub(r, rb)));
            }
            v::store(out + j, r);
        }
        return j;
    }

    /** Two double samples per step for one level layout (see readFloat4()). */
    template <int Layout, bool PerSample, bool Blend>
    int readDouble2(const double* phases, const double* positions, double position,
                    const BlockSetup& s, const FramePick& fixed, double w, const Mip& mip,
                    double* out, int n) const noexcept
    {
        namespace v = detail::wavetable_narrow;
        const v::D2 wv = v::set1(w);
        const double maxPos = static_cast<double>(numFrames_ - 1);
        const float* fixedA = s.baseA + fixed.f0 * s.strideA;
        const float* fixedB = s.baseB + fixed.f0 * s.strideB;
        int j = 0;
        for (; j + 2 <= n; j += 2)
        {
            if (!v::inRange2(phases + j))
            {
                for (int l = 0; l < 2; ++l)
                    out[j + l] = read<double>(phases[j + l], PerSample ? positions[j + l] : position, mip);
                continue;
            }
            alignas(16) int32_t ia[2], ib[2], fa[2];
            alignas(16) double ta[2];
            const v::D2 xa = v::grid2(phases + j, s.sizeA, ia);
            v::D2 xb = xa;
            if constexpr (Layout == kTwoGrids) xb = v::grid2(phases + j, s.sizeB, ib);
            if constexpr (PerSample) v::store(ta, v::frames2(positions + j, maxPos, fa));

            v::D2 rowA[2][3], rowB[2][3];
            for (int l = 0; l < 2; ++l)
            {
                const float* baseRowA = fixedA;
                const float* baseRowB = fixedB;
                size_t stepA = fixed.step * s.strideA, stepB = fixed.step * s.strideB;
                double t = fixed.t;
                bool blend = Blend;
                if constexpr (PerSample)
                {
                    const auto f0 = static_cast<size_t>(fa[l]);
                    const bool upper = fa[l] < numFrames_ - 1;
                    baseRowA = s.baseA + f0 * s.strideA;
                    baseRowB = s.baseB + f0 * s.strideB;
                    stepA = upper ? s.strideA : 0;
                    stepB = upper ? s.strideB : 0;
                    t = ta[l];
                    blend = t != 0.0;
                }
                const float* pa = baseRowA + static_cast<size_t>(ia[l]);
                for (int k = 0; k < 3; ++k)
                {
                    rowA[l][k] = v::pair(pa + 2 * k);
                    if (blend) rowA[l][k] = v::lerp(rowA[l][k], v::pair(pa + stepA + 2 * k), t);
                }
                if constexpr (Layout != kOneLevel)
                {
                    const float* pb = baseRowB + static_cast<size_t>(Layout == kSameGrid ? ia[l] : ib[l]);
                    for (int k = 0; k < 3; ++k)
                    {
                        rowB[l][k] = v::pair(pb + 2 * k);
                        if (blend) rowB[l][k] = v::lerp(rowB[l][k], v::pair(pb + stepB + 2 * k), t);
                    }
                }
                if constexpr (Layout == kSameGrid)
                    for (int k = 0; k < 3; ++k)
                        rowA[l][k] = v::add(rowB[l][k], v::mul(wv, v::sub(rowA[l][k], rowB[l][k])));
            }
            v::D2 y[6];
            v::columns2(rowA[0], rowA[1], y);
            v::D2 r = v::interpolateLanes<double>(y, xa);
            if constexpr (Layout == kTwoGrids)
            {
                v::columns2(rowB[0], rowB[1], y);
                const v::D2 rb = v::interpolateLanes<double>(y, xb);
                r = v::add(rb, v::mul(wv, v::sub(r, rb)));
            }
            v::store(out + j, r);
        }
        return j;
    }
#endif

    /** Six frame-blended samples around the phase on one level; returns the fraction. */
    template <FloatType T>
    T gather(int level, int f0, int f1, T t, double phase, T* y) const noexcept
    {
        const auto L = static_cast<size_t>(level);
        const int size = levelSize_[L];
        const double p = phase * static_cast<double>(size);
        const int i = std::min(static_cast<int>(p), size - 1);
        const size_t stride = static_cast<size_t>(size + 2 * kGuard);
        const float* ta = data_.data() + levelOffset_[L] + static_cast<size_t>(f0) * stride
                        + static_cast<size_t>(kGuard + i - 2);
        if (t == T(0))
        {
            for (int k = 0; k < 6; ++k) y[k] = static_cast<T>(ta[k]);
        }
        else
        {
            const float* tb = ta + static_cast<size_t>(f1 - f0) * stride;
            for (int k = 0; k < 6; ++k)
                y[k] = static_cast<T>(ta[k]) + t * (static_cast<T>(tb[k]) - static_cast<T>(ta[k]));
        }
        return static_cast<T>(p - static_cast<double>(i));
    }

    /** Synthesises, pre-emphasises and normalises every level of every frame. */
    bool build(int numFrames, int numHarmonics, const std::vector<Harmonic>& coeffs,
               Normalization normalization)
    {
        std::array<int, kMipLevels> harmonics {};
        std::array<int, kMipLevels> sizes {};
        std::array<size_t, kMipLevels> offsets {};
        size_t total = 0;
        for (int L = 0; L < kMipLevels; ++L)
        {
            const auto l = static_cast<size_t>(L);
            harmonics[l] = std::max(1, std::min(kLevelHarmonics[l], numHarmonics));
            int size = kMinTableSize;
            while (size < 8 * harmonics[l]) size *= 2;
            sizes[l] = size;
            offsets[l] = total;
            total += static_cast<size_t>(numFrames) * static_cast<size_t>(size + 2 * kGuard);
        }

        // Pre-emphasis per (table size, harmonic): the inverse interpolator
        // response at h / size cycles per sample.
        std::vector<std::vector<double>> emphasis(static_cast<size_t>(kMipLevels));
        for (int L = 0; L < kMipLevels; ++L)
        {
            const auto l = static_cast<size_t>(L);
            int same = -1;
            for (int j = 0; j < L; ++j)
                if (sizes[static_cast<size_t>(j)] == sizes[l]
                    && harmonics[static_cast<size_t>(j)] >= harmonics[l]) { same = j; break; }
            if (same >= 0) { emphasis[l] = emphasis[static_cast<size_t>(same)]; continue; }
            emphasis[l].resize(static_cast<size_t>(harmonics[l] + 1), 1.0);
            for (int h = 1; h <= harmonics[l]; ++h)
                emphasis[l][static_cast<size_t>(h)] =
                    1.0 / interpolatorResponse(static_cast<double>(h) / static_cast<double>(sizes[l]));
        }

        std::vector<double> spec, cycle;
        std::vector<FFTReal<double>> ffts;
        std::vector<int> fftSizes;
        auto fftFor = [&](int size) -> FFTReal<double>& {
            for (size_t j = 0; j < fftSizes.size(); ++j)
                if (fftSizes[j] == size) return ffts[j];
            fftSizes.push_back(size);
            ffts.emplace_back(static_cast<size_t>(size));
            return ffts.back();
        };
        for (int L = 0; L < kMipLevels; ++L) (void)fftFor(sizes[static_cast<size_t>(L)]);

        // One level of one frame into `cycle`, scaled by `gain`.
        auto synthesize = [&](int frame, int L) {
            const auto l = static_cast<size_t>(L);
            const int size = sizes[l];
            spec.assign(static_cast<size_t>(size + 2), 0.0);
            cycle.resize(static_cast<size_t>(size));
            const double half = 0.5 * static_cast<double>(size);
            const Harmonic* c = coeffs.data() + static_cast<size_t>(frame) * static_cast<size_t>(numHarmonics);
            for (int h = 1; h <= harmonics[l] && h <= numHarmonics; ++h)
            {
                // x = sum a cos + b sin has X[h] = (N/2)(a - j b) under the
                // inverse transform's 1/N scaling.
                const double e = emphasis[l][static_cast<size_t>(h)];
                spec[static_cast<size_t>(2 * h)]     =  half * c[h - 1].cosine * e;
                spec[static_cast<size_t>(2 * h + 1)] = -half * c[h - 1].sine * e;
            }
            fftFor(size).inverse(spec.data(), cycle.data());
        };

        // Pass 1: peak per frame over all its levels (in practice the
        // richest). Pass 2: write the scaled tables with their guards.
        std::vector<double> peak(static_cast<size_t>(numFrames), 0.0);
        for (int f = 0; f < numFrames; ++f)
            for (int L = 0; L < kMipLevels; ++L)
            {
                synthesize(f, L);
                for (const double v : cycle)
                    peak[static_cast<size_t>(f)] = std::max(peak[static_cast<size_t>(f)], std::abs(v));
            }
        double bankPeak = 0.0;
        for (const double p : peak) bankPeak = std::max(bankPeak, p);
        if (!(bankPeak > 0.0) || !std::isfinite(bankPeak)) return false;

        std::vector<float> data(total + kTailPad, 0.0f);
        for (int f = 0; f < numFrames; ++f)
        {
            const double framePeak = peak[static_cast<size_t>(f)];
            const double gain = (normalization == Normalization::PerFrame)
                ? (framePeak > 0.0 ? 1.0 / framePeak : 0.0)
                : 1.0 / bankPeak;
            for (int L = 0; L < kMipLevels; ++L)
            {
                synthesize(f, L);
                const auto l = static_cast<size_t>(L);
                const int size = sizes[l];
                float* dst = data.data() + offsets[l]
                           + static_cast<size_t>(f) * static_cast<size_t>(size + 2 * kGuard);
                for (int i = 0; i < size; ++i)
                    dst[kGuard + i] = static_cast<float>(cycle[static_cast<size_t>(i)] * gain);
                for (int g = 0; g < kGuard; ++g)
                {
                    dst[g] = dst[size + g];                     // wraps of the cycle's end
                    dst[kGuard + size + g] = dst[kGuard + g];   // and of its start
                }
            }
        }

        data_ = std::move(data);
        levelHarmonics_ = harmonics;
        levelSize_ = sizes;
        levelOffset_ = offsets;
        numFrames_ = numFrames;
        return true;
    }

    // -- Built-in recipes ----------------------------------------------------------

    /** Fourier coefficients of a +-1 pulse that is high for the first `w` of the cycle. */
    [[nodiscard]] static Harmonic pulse(double w, int h) noexcept
    {
        const double theta = twoPi<double> * static_cast<double>(h) * w;
        const double k = 2.0 / (pi<double> * static_cast<double>(h));
        return { k * std::sin(theta), k * (1.0 - std::cos(theta)) };
    }

    [[nodiscard]] static WavetableBank makeBasic()
    {
        WavetableBank b;
        b.buildFromHarmonics(6, kMaxHarmonics, [](int frame, int h) -> Harmonic {
            const double hd = static_cast<double>(h);
            switch (frame)
            {
                case 0: return { 0.0, h == 1 ? 1.0 : 0.0 };
                case 1:
                {
                    if (h % 2 == 0) return {};
                    const double sign = ((h / 2) % 2 == 0) ? 1.0 : -1.0;
                    return { 0.0, sign / (hd * hd) };
                }
                case 2: return { 0.0, (h % 2 == 0) ? -1.0 / hd : 1.0 / hd };
                case 3: return pulse(0.5, h);
                case 4: return pulse(0.25, h);
                default: return pulse(0.125, h);
            }
        });
        return b;
    }

    [[nodiscard]] static WavetableBank makeHarmonic()
    {
        WavetableBank b;
        b.buildFromHarmonics(8, kMaxHarmonics, [](int frame, int h) -> Harmonic {
            const double p = 3.0 - 2.5 * static_cast<double>(frame) / 7.0;
            return { 0.0, std::pow(static_cast<double>(h), -p) };
        });
        return b;
    }

    [[nodiscard]] static WavetableBank makeFormant()
    {
        // Three formants (Hz) of the vowels A E I O U, placed on the harmonic
        // series of a 110 Hz cycle; odd frames sit halfway between vowels.
        static constexpr double vowels[5][3] = {
            { 730.0, 1090.0, 2440.0 }, { 530.0, 1840.0, 2480.0 }, { 270.0, 2290.0, 3010.0 },
            { 570.0, 840.0, 2410.0 },  { 300.0, 870.0, 2240.0 } };
        static constexpr double gains[3] = { 1.0, 0.5, 0.25 };
        static constexpr double widths[3] = { 90.0, 120.0, 160.0 };
        WavetableBank b;
        b.buildFromHarmonics(9, kMaxHarmonics, [](int frame, int h) -> Harmonic {
            const int v0 = frame / 2;
            const int v1 = std::min(v0 + (frame % 2), 4);
            const double f = 110.0 * static_cast<double>(h);
            double env = 0.01;
            for (int i = 0; i < 3; ++i)
            {
                const double fc = 0.5 * (vowels[v0][i] + vowels[v1][i]);
                const double d = (f - fc) / widths[i];
                env += gains[i] / (1.0 + d * d);
            }
            return { 0.0, env / static_cast<double>(h) };
        });
        return b;
    }

    [[nodiscard]] static WavetableBank makeFm()
    {
        // sin(x + b sin x): a smooth cycle whose harmonics above about b + 12
        // are below -150 dB for b <= 6, so a dense numeric cycle analysed by
        // FFT gives its coefficients exactly.
        constexpr int n = 4096;
        std::vector<double> cycles(static_cast<size_t>(8 * n));
        for (int frame = 0; frame < 8; ++frame)
        {
            const double b = 6.0 * std::pow(static_cast<double>(frame) / 7.0, 1.5);
            for (int i = 0; i < n; ++i)
            {
                const double x = twoPi<double> * static_cast<double>(i) / static_cast<double>(n);
                cycles[static_cast<size_t>(frame * n + i)] = std::sin(x + b * std::sin(x));
            }
        }
        WavetableBank bank;
        bank.loadFrames(cycles.data(), n, 8, Normalization::PerFrame);
        return bank;
    }

    int numFrames_ = 0;
    std::array<int, kMipLevels> levelHarmonics_ {};   ///< Harmonic budget per level.
    std::array<int, kMipLevels> levelSize_ {};        ///< Table samples per level.
    std::array<size_t, kMipLevels> levelOffset_ {};   ///< Start of each level's frame block.
    std::vector<float> data_;                         ///< [level][frame][guard, table, guard].
};

} // namespace dspark
