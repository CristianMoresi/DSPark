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
 * float: read() 14 ns per sample, readBlock() 2.3 ns with AVX2 and 3.0 ns
 * with SSE2, phase accumulation included; 4.1 ns with AVX2 when the frame
 * position moves every sample. On one level (no crossfade) 1.9 and 2.4 ns,
 * against 2.5 ns for a 4-point cubic reader of one table in the same loop.
 * The SSE2, NEON and AVX2 kernels are one source, detail/WavetableKernels.inl,
 * over each instruction set's lane operations.
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
 * Dependencies: DspMath.h, FFT.h, SimdOps.h, detail/WavetableKernels.inl.
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

// Lane operations for the block kernels of detail/WavetableKernels.inl, one
// pair of structs per instruction set: four float or two double lanes on SSE2
// and NEON, eight or four with AVX2. The kernels are included once per set.
#if defined(DSPARK_SIMD_SSE2) || defined(DSPARK_SIMD_NEON)
    #define DSPARK_WAVETABLE_NARROW 1
namespace detail {
namespace wavetable_narrow {

#if defined(DSPARK_SIMD_SSE2)
/// Four float lanes on SSE2.
struct FloatLanes
{
    using S = float;
    using V = __m128;
    static constexpr int N = 4;

    struct Rows
    {
        V lo[4];   ///< Lane l: offsets -2..+1.
        V hi[2];   ///< Lanes 2k and 2k + 1: offsets +2, +3 of each.
    };

    static V add(V a, V b) noexcept { return _mm_add_ps(a, b); }
    static V sub(V a, V b) noexcept { return _mm_sub_ps(a, b); }
    static V mul(V a, V b) noexcept { return _mm_mul_ps(a, b); }
    static V set1(float x) noexcept { return _mm_set1_ps(x); }
    static void store(float* p, V v) noexcept { _mm_storeu_ps(p, v); }

    /// [0, 1) holds exactly the doubles whose high word, as an unsigned
    /// number, lies below that of 1.0 (sign clear; NaN and infinities fail).
    static bool inRange(const double* ph) noexcept
    {
        const __m128 a = _mm_castpd_ps(_mm_loadu_pd(ph));
        const __m128 b = _mm_castpd_ps(_mm_loadu_pd(ph + 2));
        const __m128i hi = _mm_castps_si128(_mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1)));
        const __m128i flip = _mm_set1_epi32(static_cast<int>(0x80000000u));
        const __m128i bad = _mm_cmpgt_epi32(_mm_xor_si128(hi, flip),
                                            _mm_set1_epi32(static_cast<int>(0x3FEFFFFFu ^ 0x80000000u)));
        return _mm_movemask_epi8(bad) == 0;
    }

    /// phase * size is exact (sizes are powers of two), so the truncated
    /// index stays below the size, as read()'s min() guarantees.
    static V grid(const double* ph, double size, uint32_t* idx) noexcept
    {
        const __m128d sz = _mm_set1_pd(size);
        const __m128d p0 = _mm_mul_pd(_mm_loadu_pd(ph), sz);
        const __m128d p1 = _mm_mul_pd(_mm_loadu_pd(ph + 2), sz);
        const __m128i i0 = _mm_cvttpd_epi32(p0), i1 = _mm_cvttpd_epi32(p1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), _mm_unpacklo_epi64(i0, i1));
        return _mm_movelh_ps(_mm_cvtpd_ps(_mm_sub_pd(p0, _mm_cvtepi32_pd(i0))),
                             _mm_cvtpd_ps(_mm_sub_pd(p1, _mm_cvtepi32_pd(i1))));
    }

    static void frames(const double* pos, double maxPos, int32_t* f0, float* t) noexcept
    {
        const __m128d mx = _mm_set1_pd(maxPos), zero = _mm_setzero_pd();
        __m128 frac[2];
        for (int h = 0; h < 2; ++h)
        {
            const __m128d p = _mm_loadu_pd(pos + 2 * h);
            const __m128d c = _mm_and_pd(_mm_cmpgt_pd(p, zero), _mm_min_pd(p, mx));
            const __m128i i = _mm_cvttpd_epi32(c);
            f0[2 * h] = _mm_cvtsi128_si32(i);
            f0[2 * h + 1] = _mm_cvtsi128_si32(_mm_srli_si128(i, 4));
            frac[h] = _mm_cvtpd_ps(_mm_sub_pd(c, _mm_cvtepi32_pd(i)));
        }
        _mm_storeu_ps(t, _mm_movelh_ps(frac[0], frac[1]));
    }

    /// Offsets +2, +3 of two lanes in one vector.
    static V pair(const float* a, const float* b) noexcept
    {
        return _mm_castsi128_ps(_mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a)),
                                                   _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b))));
    }

    static V lerp(V a, V b, V t) noexcept { return add(a, mul(t, sub(b, a))); }

    // Each pair of lanes reads its row pointers and blends before its first
    // store, so the stores cannot force them to be recomputed.
    template <bool Blend, class At, class Step, class Tl>
    static void gather(Rows& r, const At& at, const Step& step, const Tl& t) noexcept
    {
        for (int k = 0; k < 2; ++k)
        {
            const int l = 2 * k;
            const float* const p0 = at(l);
            const float* const p1 = at(l + 1);
            if constexpr (Blend)
            {
                const float* const q0 = p0 + step(l);
                const float* const q1 = p1 + step(l + 1);
                const float t0 = t(l), t1 = t(l + 1);
                r.lo[l] = lerp(_mm_loadu_ps(p0), _mm_loadu_ps(q0), _mm_set1_ps(t0));
                r.lo[l + 1] = lerp(_mm_loadu_ps(p1), _mm_loadu_ps(q1), _mm_set1_ps(t1));
                r.hi[k] = lerp(pair(p0 + 4, p1 + 4), pair(q0 + 4, q1 + 4), _mm_setr_ps(t0, t0, t1, t1));
            }
            else
            {
                r.lo[l] = _mm_loadu_ps(p0);
                r.lo[l + 1] = _mm_loadu_ps(p1);
                r.hi[k] = pair(p0 + 4, p1 + 4);
            }
        }
    }

    template <bool Blend, class AtA, class StepA, class AtB, class StepB, class Tl>
    static void gatherMix(Rows& r, const AtA& atA, const StepA& stepA, const AtB& atB, const StepB& stepB,
                          const Tl& t, V w) noexcept
    {
        for (int k = 0; k < 2; ++k)
        {
            const int l = 2 * k;
            const float* const pa0 = atA(l);
            const float* const pa1 = atA(l + 1);
            const float* const pb0 = atB(l);
            const float* const pb1 = atB(l + 1);
            V a0 = _mm_loadu_ps(pa0), a1 = _mm_loadu_ps(pa1), ah = pair(pa0 + 4, pa1 + 4);
            V b0 = _mm_loadu_ps(pb0), b1 = _mm_loadu_ps(pb1), bh = pair(pb0 + 4, pb1 + 4);
            if constexpr (Blend)
            {
                const float* const qa0 = pa0 + stepA(l);
                const float* const qa1 = pa1 + stepA(l + 1);
                const float* const qb0 = pb0 + stepB(l);
                const float* const qb1 = pb1 + stepB(l + 1);
                const float s0 = t(l), s1 = t(l + 1);
                const V t0 = _mm_set1_ps(s0), t1 = _mm_set1_ps(s1);
                const V th = _mm_setr_ps(s0, s0, s1, s1);
                a0 = lerp(a0, _mm_loadu_ps(qa0), t0);
                a1 = lerp(a1, _mm_loadu_ps(qa1), t1);
                ah = lerp(ah, pair(qa0 + 4, qa1 + 4), th);
                b0 = lerp(b0, _mm_loadu_ps(qb0), t0);
                b1 = lerp(b1, _mm_loadu_ps(qb1), t1);
                bh = lerp(bh, pair(qb0 + 4, qb1 + 4), th);
            }
            r.lo[l] = lerp(b0, a0, w);
            r.lo[l + 1] = lerp(b1, a1, w);
            r.hi[k] = lerp(bh, ah, w);
        }
    }

    static void columns(const Rows& r, V* y) noexcept
    {
        __m128 a = r.lo[0], b = r.lo[1], c = r.lo[2], d = r.lo[3];
        _MM_TRANSPOSE4_PS(a, b, c, d);
        y[0] = a;
        y[1] = b;
        y[2] = c;
        y[3] = d;
        y[4] = _mm_shuffle_ps(r.hi[0], r.hi[1], _MM_SHUFFLE(2, 0, 2, 0));
        y[5] = _mm_shuffle_ps(r.hi[0], r.hi[1], _MM_SHUFFLE(3, 1, 3, 1));
    }
};

/// Two double lanes on SSE2.
struct DoubleLanes
{
    using S = double;
    using V = __m128d;
    static constexpr int N = 2;

    struct Rows
    {
        V r[2][3];   ///< Lane l: offsets -2..+3 as three pairs.
    };

    static V add(V a, V b) noexcept { return _mm_add_pd(a, b); }
    static V sub(V a, V b) noexcept { return _mm_sub_pd(a, b); }
    static V mul(V a, V b) noexcept { return _mm_mul_pd(a, b); }
    static V set1(double x) noexcept { return _mm_set1_pd(x); }
    static void store(double* p, V v) noexcept { _mm_storeu_pd(p, v); }

    static bool inRange(const double* ph) noexcept
    {
        const __m128d v = _mm_loadu_pd(ph);
        const __m128d ok = _mm_and_pd(_mm_cmpge_pd(v, _mm_setzero_pd()), _mm_cmplt_pd(v, _mm_set1_pd(1.0)));
        return _mm_movemask_pd(ok) == 3;
    }

    static V grid(const double* ph, double size, uint32_t* idx) noexcept
    {
        const __m128d p = _mm_mul_pd(_mm_loadu_pd(ph), _mm_set1_pd(size));
        const __m128i i = _mm_cvttpd_epi32(p);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(idx), i);
        return _mm_sub_pd(p, _mm_cvtepi32_pd(i));
    }

    static void frames(const double* pos, double maxPos, int32_t* f0, double* t) noexcept
    {
        const __m128d p = _mm_loadu_pd(pos);
        const __m128d c = _mm_and_pd(_mm_cmpgt_pd(p, _mm_setzero_pd()), _mm_min_pd(p, _mm_set1_pd(maxPos)));
        const __m128i i = _mm_cvttpd_epi32(c);
        f0[0] = _mm_cvtsi128_si32(i);
        f0[1] = _mm_cvtsi128_si32(_mm_srli_si128(i, 4));
        _mm_storeu_pd(t, _mm_sub_pd(c, _mm_cvtepi32_pd(i)));
    }

    /// Two floats widened to doubles.
    static V pair(const float* p) noexcept
    {
        return _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
    }

    static V lerp(V a, V b, V t) noexcept { return add(a, mul(t, sub(b, a))); }

    // Each lane's row pointers and blend are read before its first store, so
    // the stores cannot force them to be recomputed.
    template <bool Blend, class At, class Step, class Tl>
    static void gather(Rows& r, const At& at, const Step& step, const Tl& t) noexcept
    {
        for (int l = 0; l < 2; ++l)
        {
            const float* const p = at(l);
            if constexpr (Blend)
            {
                const float* const q = p + step(l);
                const V tl = _mm_set1_pd(t(l));
                for (int k = 0; k < 3; ++k) r.r[l][k] = lerp(pair(p + 2 * k), pair(q + 2 * k), tl);
            }
            else
            {
                for (int k = 0; k < 3; ++k) r.r[l][k] = pair(p + 2 * k);
            }
        }
    }

    template <bool Blend, class AtA, class StepA, class AtB, class StepB, class Tl>
    static void gatherMix(Rows& r, const AtA& atA, const StepA& stepA, const AtB& atB, const StepB& stepB,
                          const Tl& t, V w) noexcept
    {
        for (int l = 0; l < 2; ++l)
        {
            const float* const pa = atA(l);
            const float* const pb = atB(l);
            if constexpr (Blend)
            {
                const float* const qa = pa + stepA(l);
                const float* const qb = pb + stepB(l);
                const V tl = _mm_set1_pd(t(l));
                for (int k = 0; k < 3; ++k)
                {
                    const V a = lerp(pair(pa + 2 * k), pair(qa + 2 * k), tl);
                    const V b = lerp(pair(pb + 2 * k), pair(qb + 2 * k), tl);
                    r.r[l][k] = lerp(b, a, w);
                }
            }
            else
            {
                for (int k = 0; k < 3; ++k) r.r[l][k] = lerp(pair(pb + 2 * k), pair(pa + 2 * k), w);
            }
        }
    }

    static void columns(const Rows& r, V* y) noexcept
    {
        for (int k = 0; k < 3; ++k)
        {
            y[2 * k]     = _mm_unpacklo_pd(r.r[0][k], r.r[1][k]);
            y[2 * k + 1] = _mm_unpackhi_pd(r.r[0][k], r.r[1][k]);
        }
    }
};
#else // NEON
/// Four float lanes on NEON.
struct FloatLanes
{
    using S = float;
    using V = float32x4_t;
    static constexpr int N = 4;

    struct Rows
    {
        V lo[4];   ///< Lane l: offsets -2..+1.
        V hi[2];   ///< Lanes 2k and 2k + 1: offsets +2, +3 of each.
    };

    static V add(V a, V b) noexcept { return vaddq_f32(a, b); }
    static V sub(V a, V b) noexcept { return vsubq_f32(a, b); }
    static V mul(V a, V b) noexcept { return vmulq_f32(a, b); }
    static V set1(float x) noexcept { return vdupq_n_f32(x); }
    static void store(float* p, V v) noexcept { vst1q_f32(p, v); }

    static bool inRange(const double* ph) noexcept
    {
        const float64x2_t one = vdupq_n_f64(1.0), zero = vdupq_n_f64(0.0);
        const float64x2_t a = vld1q_f64(ph), b = vld1q_f64(ph + 2);
        const uint64x2_t ok = vandq_u64(vandq_u64(vcgeq_f64(a, zero), vcltq_f64(a, one)),
                                        vandq_u64(vcgeq_f64(b, zero), vcltq_f64(b, one)));
        return vgetq_lane_u64(ok, 0) != 0 && vgetq_lane_u64(ok, 1) != 0;
    }

    static V grid(const double* ph, double size, uint32_t* idx) noexcept
    {
        const float64x2_t sz = vdupq_n_f64(size);
        const float64x2_t p0 = vmulq_f64(vld1q_f64(ph), sz), p1 = vmulq_f64(vld1q_f64(ph + 2), sz);
        const int64x2_t i0 = vcvtq_s64_f64(p0), i1 = vcvtq_s64_f64(p1);   // toward zero, as the cast
        idx[0] = static_cast<uint32_t>(vgetq_lane_s64(i0, 0));
        idx[1] = static_cast<uint32_t>(vgetq_lane_s64(i0, 1));
        idx[2] = static_cast<uint32_t>(vgetq_lane_s64(i1, 0));
        idx[3] = static_cast<uint32_t>(vgetq_lane_s64(i1, 1));
        return vcombine_f32(vcvt_f32_f64(vsubq_f64(p0, vcvtq_f64_s64(i0))),
                            vcvt_f32_f64(vsubq_f64(p1, vcvtq_f64_s64(i1))));
    }

    static void frames(const double* pos, double maxPos, int32_t* f0, float* t) noexcept
    {
        float32x2_t frac[2];
        for (int h = 0; h < 2; ++h)
        {
            const float64x2_t p = vld1q_f64(pos + 2 * h);
            // FMIN returns NaN for a NaN input; the mask clears it to 0, as read().
            const uint64x2_t c = vandq_u64(vcgtq_f64(p, vdupq_n_f64(0.0)),
                                           vreinterpretq_u64_f64(vminq_f64(p, vdupq_n_f64(maxPos))));
            const float64x2_t cd = vreinterpretq_f64_u64(c);
            const int64x2_t i = vcvtq_s64_f64(cd);
            f0[2 * h] = static_cast<int32_t>(vgetq_lane_s64(i, 0));
            f0[2 * h + 1] = static_cast<int32_t>(vgetq_lane_s64(i, 1));
            frac[h] = vcvt_f32_f64(vsubq_f64(cd, vcvtq_f64_s64(i)));
        }
        vst1q_f32(t, vcombine_f32(frac[0], frac[1]));
    }

    static V pair(const float* a, const float* b) noexcept { return vcombine_f32(vld1_f32(a), vld1_f32(b)); }

    static V lerp(V a, V b, V t) noexcept { return add(a, mul(t, sub(b, a))); }

    template <bool Blend, class At, class Step, class Tl>
    static void gather(Rows& r, const At& at, const Step& step, const Tl& t) noexcept
    {
        for (int k = 0; k < 2; ++k)
        {
            const int l = 2 * k;
            r.lo[l] = vld1q_f32(at(l));
            r.lo[l + 1] = vld1q_f32(at(l + 1));
            r.hi[k] = pair(at(l) + 4, at(l + 1) + 4);
            if constexpr (Blend)
            {
                r.lo[l] = lerp(r.lo[l], vld1q_f32((at(l) + step(l))), vdupq_n_f32(t(l)));
                r.lo[l + 1] = lerp(r.lo[l + 1], vld1q_f32((at(l + 1) + step(l + 1))), vdupq_n_f32(t(l + 1)));
                r.hi[k] = lerp(r.hi[k], pair((at(l) + step(l)) + 4, (at(l + 1) + step(l + 1)) + 4),
                               vcombine_f32(vdup_n_f32(t(l)), vdup_n_f32(t(l + 1))));
            }
        }
    }

    template <bool Blend, class AtA, class StepA, class AtB, class StepB, class Tl>
    static void gatherMix(Rows& r, const AtA& atA, const StepA& stepA, const AtB& atB, const StepB& stepB,
                          const Tl& t, V w) noexcept
    {
        for (int k = 0; k < 2; ++k)
        {
            const int l = 2 * k;
            V a0 = vld1q_f32(atA(l)), a1 = vld1q_f32(atA(l + 1)), ah = pair(atA(l) + 4, atA(l + 1) + 4);
            V b0 = vld1q_f32(atB(l)), b1 = vld1q_f32(atB(l + 1)), bh = pair(atB(l) + 4, atB(l + 1) + 4);
            if constexpr (Blend)
            {
                const V t0 = vdupq_n_f32(t(l)), t1 = vdupq_n_f32(t(l + 1));
                const V th = vcombine_f32(vdup_n_f32(t(l)), vdup_n_f32(t(l + 1)));
                a0 = lerp(a0, vld1q_f32((atA(l) + stepA(l))), t0);
                a1 = lerp(a1, vld1q_f32((atA(l + 1) + stepA(l + 1))), t1);
                ah = lerp(ah, pair((atA(l) + stepA(l)) + 4, (atA(l + 1) + stepA(l + 1)) + 4), th);
                b0 = lerp(b0, vld1q_f32((atB(l) + stepB(l))), t0);
                b1 = lerp(b1, vld1q_f32((atB(l + 1) + stepB(l + 1))), t1);
                bh = lerp(bh, pair((atB(l) + stepB(l)) + 4, (atB(l + 1) + stepB(l + 1)) + 4), th);
            }
            r.lo[l] = lerp(b0, a0, w);
            r.lo[l + 1] = lerp(b1, a1, w);
            r.hi[k] = lerp(bh, ah, w);
        }
    }

    static void columns(const Rows& r, V* y) noexcept
    {
        const float32x4x2_t t01 = vtrnq_f32(r.lo[0], r.lo[1]);   // a0 b0 a2 b2 | a1 b1 a3 b3
        const float32x4x2_t t23 = vtrnq_f32(r.lo[2], r.lo[3]);   // c0 d0 c2 d2 | c1 d1 c3 d3
        y[0] = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
        y[1] = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
        y[2] = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
        y[3] = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
        const float32x4x2_t h = vuzpq_f32(r.hi[0], r.hi[1]);    // even | odd elements
        y[4] = h.val[0];
        y[5] = h.val[1];
    }
};

/// Two double lanes on NEON. The struct keeps the double lanes' operations
/// apart from the float ones: MSVC declares float32x4_t and float64x2_t as
/// one type, so they could not be overloads of one name.
struct DoubleLanes
{
    using S = double;
    using V = float64x2_t;
    static constexpr int N = 2;

    struct Rows
    {
        V r[2][3];   ///< Lane l: offsets -2..+3 as three pairs.
    };

    static V add(V a, V b) noexcept { return vaddq_f64(a, b); }
    static V sub(V a, V b) noexcept { return vsubq_f64(a, b); }
    static V mul(V a, V b) noexcept { return vmulq_f64(a, b); }
    static V set1(double x) noexcept { return vdupq_n_f64(x); }
    static void store(double* p, V v) noexcept { vst1q_f64(p, v); }

    static bool inRange(const double* ph) noexcept
    {
        const float64x2_t v = vld1q_f64(ph);
        const uint64x2_t ok = vandq_u64(vcgeq_f64(v, vdupq_n_f64(0.0)), vcltq_f64(v, vdupq_n_f64(1.0)));
        return vgetq_lane_u64(ok, 0) != 0 && vgetq_lane_u64(ok, 1) != 0;
    }

    static V grid(const double* ph, double size, uint32_t* idx) noexcept
    {
        const float64x2_t p = vmulq_f64(vld1q_f64(ph), vdupq_n_f64(size));
        const int64x2_t i = vcvtq_s64_f64(p);
        idx[0] = static_cast<uint32_t>(vgetq_lane_s64(i, 0));
        idx[1] = static_cast<uint32_t>(vgetq_lane_s64(i, 1));
        return vsubq_f64(p, vcvtq_f64_s64(i));
    }

    static void frames(const double* pos, double maxPos, int32_t* f0, double* t) noexcept
    {
        const float64x2_t p = vld1q_f64(pos);
        const uint64x2_t c = vandq_u64(vcgtq_f64(p, vdupq_n_f64(0.0)),
                                       vreinterpretq_u64_f64(vminq_f64(p, vdupq_n_f64(maxPos))));
        const float64x2_t cd = vreinterpretq_f64_u64(c);
        const int64x2_t i = vcvtq_s64_f64(cd);
        f0[0] = static_cast<int32_t>(vgetq_lane_s64(i, 0));
        f0[1] = static_cast<int32_t>(vgetq_lane_s64(i, 1));
        vst1q_f64(t, vsubq_f64(cd, vcvtq_f64_s64(i)));
    }

    static V pair(const float* p) noexcept { return vcvt_f64_f32(vld1_f32(p)); }

    static V lerp(V a, V b, V t) noexcept { return add(a, mul(t, sub(b, a))); }

    template <bool Blend, class At, class Step, class Tl>
    static void gather(Rows& r, const At& at, const Step& step, const Tl& t) noexcept
    {
        for (int l = 0; l < 2; ++l)
            for (int k = 0; k < 3; ++k)
            {
                r.r[l][k] = pair(at(l) + 2 * k);
                if constexpr (Blend) r.r[l][k] = lerp(r.r[l][k], pair((at(l) + step(l)) + 2 * k), vdupq_n_f64(t(l)));
            }
    }

    template <bool Blend, class AtA, class StepA, class AtB, class StepB, class Tl>
    static void gatherMix(Rows& r, const AtA& atA, const StepA& stepA, const AtB& atB, const StepB& stepB,
                          const Tl& t, V w) noexcept
    {
        for (int l = 0; l < 2; ++l)
            for (int k = 0; k < 3; ++k)
            {
                V a = pair(atA(l) + 2 * k), b = pair(atB(l) + 2 * k);
                if constexpr (Blend)
                {
                    const V tl = vdupq_n_f64(t(l));
                    a = lerp(a, pair((atA(l) + stepA(l)) + 2 * k), tl);
                    b = lerp(b, pair((atB(l) + stepB(l)) + 2 * k), tl);
                }
                r.r[l][k] = lerp(b, a, w);
            }
    }

    static void columns(const Rows& r, V* y) noexcept
    {
        for (int k = 0; k < 3; ++k)
        {
            y[2 * k]     = vzip1q_f64(r.r[0][k], r.r[1][k]);
            y[2 * k + 1] = vzip2q_f64(r.r[0][k], r.r[1][k]);
        }
    }
};
#endif

#include "detail/WavetableKernels.inl"

} // namespace wavetable_narrow
} // namespace detail
#endif

// AVX2 lanes, used when the CPU has AVX2 (selected at run time in x86-64
// baseline builds; always in AVX2 builds). They are built in SimdOps.h's
// exact AVX2 region, without FMA, so they compute what read() computes,
// operation for operation.
#if defined(DSPARK_SIMD_EXACT_AVX2)
DSPARK_SIMD_EXACT_AVX2_BEGIN
namespace detail {
namespace wavetable_avx2 {

/// Eight float lanes on AVX2.
struct FloatLanes
{
    using S = float;
    using V = __m256;
    static constexpr int N = 8;

    struct Rows
    {
        V r[8];   ///< Lane l: offsets -2..+5 (the kernel uses -2..+3).
    };

    static V add(V a, V b) noexcept { return _mm256_add_ps(a, b); }
    static V sub(V a, V b) noexcept { return _mm256_sub_ps(a, b); }
    static V mul(V a, V b) noexcept { return _mm256_mul_ps(a, b); }
    static V set1(float x) noexcept { return _mm256_set1_ps(x); }
    static void store(float* p, V v) noexcept { _mm256_storeu_ps(p, v); }

    static bool inRange(const double* ph) noexcept
    {
        const __m256d zero = _mm256_setzero_pd(), one = _mm256_set1_pd(1.0);
        const __m256d a = _mm256_loadu_pd(ph), b = _mm256_loadu_pd(ph + 4);
        const __m256d ok = _mm256_and_pd(_mm256_and_pd(_mm256_cmp_pd(a, zero, _CMP_GE_OQ), _mm256_cmp_pd(a, one, _CMP_LT_OQ)),
                                         _mm256_and_pd(_mm256_cmp_pd(b, zero, _CMP_GE_OQ), _mm256_cmp_pd(b, one, _CMP_LT_OQ)));
        return _mm256_movemask_pd(ok) == 0xF;
    }

    static V grid(const double* ph, double size, uint32_t* idx) noexcept
    {
        const __m256d sz = _mm256_set1_pd(size);
        const __m256d p0 = _mm256_mul_pd(_mm256_loadu_pd(ph), sz);
        const __m256d p1 = _mm256_mul_pd(_mm256_loadu_pd(ph + 4), sz);
        const __m128i i0 = _mm256_cvttpd_epi32(p0), i1 = _mm256_cvttpd_epi32(p1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx + 4), i1);
        const __m128 x0 = _mm256_cvtpd_ps(_mm256_sub_pd(p0, _mm256_cvtepi32_pd(i0)));
        const __m128 x1 = _mm256_cvtpd_ps(_mm256_sub_pd(p1, _mm256_cvtepi32_pd(i1)));
        return _mm256_insertf128_ps(_mm256_castps128_ps256(x0), x1, 1);
    }

    static void frames(const double* pos, double maxPos, int32_t* f0, float* t) noexcept
    {
        const __m256d zero = _mm256_setzero_pd(), mx = _mm256_set1_pd(maxPos);
        for (int h = 0; h < 2; ++h)
        {
            const __m256d p = _mm256_loadu_pd(pos + 4 * h);
            const __m256d c = _mm256_and_pd(_mm256_cmp_pd(p, zero, _CMP_GT_OQ), _mm256_min_pd(p, mx));
            const __m128i i = _mm256_cvttpd_epi32(c);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(f0 + 4 * h), i);
            _mm_storeu_ps(t + 4 * h, _mm256_cvtpd_ps(_mm256_sub_pd(c, _mm256_cvtepi32_pd(i))));
        }
    }

    static V lerp(V a, V b, V t) noexcept { return add(a, mul(t, sub(b, a))); }

    template <bool Blend, class At, class Step, class Tl>
    static void gather(Rows& r, const At& at, const Step& step, const Tl& t) noexcept
    {
        for (int l = 0; l < 8; ++l)
        {
            r.r[l] = _mm256_loadu_ps(at(l));
            if constexpr (Blend) r.r[l] = lerp(r.r[l], _mm256_loadu_ps((at(l) + step(l))), _mm256_set1_ps(t(l)));
        }
    }

    template <bool Blend, class AtA, class StepA, class AtB, class StepB, class Tl>
    static void gatherMix(Rows& r, const AtA& atA, const StepA& stepA, const AtB& atB, const StepB& stepB,
                          const Tl& t, V w) noexcept
    {
        for (int l = 0; l < 8; ++l)
        {
            V a = _mm256_loadu_ps(atA(l)), b = _mm256_loadu_ps(atB(l));
            if constexpr (Blend)
            {
                const V tl = _mm256_set1_ps(t(l));
                a = lerp(a, _mm256_loadu_ps((atA(l) + stepA(l))), tl);
                b = lerp(b, _mm256_loadu_ps((atB(l) + stepB(l))), tl);
            }
            r.r[l] = lerp(b, a, w);
        }
    }

    static void columns(const Rows& rows, V* y) noexcept
    {
        const V* r = rows.r;
        const V t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
        const V t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
        const V t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
        const V t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
        const V s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        const V s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        const V s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        const V s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
        y[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
        y[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
        y[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
        y[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
        y[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
        y[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    }
};

/// Four double lanes on AVX2.
struct DoubleLanes
{
    using S = double;
    using V = __m256d;
    static constexpr int N = 4;

    struct Rows
    {
        V lo[4];        ///< Lane l: offsets -2..+1.
        __m128d hi[4];  ///< Lane l: offsets +2, +3.
    };

    static V add(V a, V b) noexcept { return _mm256_add_pd(a, b); }
    static V sub(V a, V b) noexcept { return _mm256_sub_pd(a, b); }
    static V mul(V a, V b) noexcept { return _mm256_mul_pd(a, b); }
    static V set1(double x) noexcept { return _mm256_set1_pd(x); }
    static void store(double* p, V v) noexcept { _mm256_storeu_pd(p, v); }

    static bool inRange(const double* ph) noexcept
    {
        const __m256d a = _mm256_loadu_pd(ph);
        const __m256d ok = _mm256_and_pd(_mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_GE_OQ),
                                         _mm256_cmp_pd(a, _mm256_set1_pd(1.0), _CMP_LT_OQ));
        return _mm256_movemask_pd(ok) == 0xF;
    }

    static V grid(const double* ph, double size, uint32_t* idx) noexcept
    {
        const __m256d p = _mm256_mul_pd(_mm256_loadu_pd(ph), _mm256_set1_pd(size));
        const __m128i i = _mm256_cvttpd_epi32(p);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(idx), i);
        return _mm256_sub_pd(p, _mm256_cvtepi32_pd(i));
    }

    static void frames(const double* pos, double maxPos, int32_t* f0, double* t) noexcept
    {
        const __m256d p = _mm256_loadu_pd(pos);
        const __m256d c = _mm256_and_pd(_mm256_cmp_pd(p, _mm256_setzero_pd(), _CMP_GT_OQ),
                                        _mm256_min_pd(p, _mm256_set1_pd(maxPos)));
        const __m128i i = _mm256_cvttpd_epi32(c);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(f0), i);
        _mm256_storeu_pd(t, _mm256_sub_pd(c, _mm256_cvtepi32_pd(i)));
    }

    static V lo4(const float* p) noexcept { return _mm256_cvtps_pd(_mm_loadu_ps(p)); }
    static __m128d hi2(const float* p) noexcept
    {
        return _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + 4))));
    }
    static V lerp(V a, V b, V t) noexcept { return add(a, mul(t, sub(b, a))); }
    static __m128d lerp(__m128d a, __m128d b, __m128d t) noexcept
    {
        return _mm_add_pd(a, _mm_mul_pd(t, _mm_sub_pd(b, a)));
    }

    template <bool Blend, class At, class Step, class Tl>
    static void gather(Rows& r, const At& at, const Step& step, const Tl& t) noexcept
    {
        for (int l = 0; l < 4; ++l)
        {
            r.lo[l] = lo4(at(l));
            r.hi[l] = hi2(at(l));
            if constexpr (Blend)
            {
                r.lo[l] = lerp(r.lo[l], lo4((at(l) + step(l))), _mm256_set1_pd(t(l)));
                r.hi[l] = lerp(r.hi[l], hi2((at(l) + step(l))), _mm_set1_pd(t(l)));
            }
        }
    }

    template <bool Blend, class AtA, class StepA, class AtB, class StepB, class Tl>
    static void gatherMix(Rows& r, const AtA& atA, const StepA& stepA, const AtB& atB, const StepB& stepB,
                          const Tl& t, V w) noexcept
    {
        const __m128d w2 = _mm256_castpd256_pd128(w);
        for (int l = 0; l < 4; ++l)
        {
            V a = lo4(atA(l)), b = lo4(atB(l));
            __m128d ah = hi2(atA(l)), bh = hi2(atB(l));
            if constexpr (Blend)
            {
                const V tl = _mm256_set1_pd(t(l));
                const __m128d th = _mm_set1_pd(t(l));
                a = lerp(a, lo4((atA(l) + stepA(l))), tl);
                ah = lerp(ah, hi2((atA(l) + stepA(l))), th);
                b = lerp(b, lo4((atB(l) + stepB(l))), tl);
                bh = lerp(bh, hi2((atB(l) + stepB(l))), th);
            }
            r.lo[l] = lerp(b, a, w);
            r.hi[l] = lerp(bh, ah, w2);
        }
    }

    static void columns(const Rows& r, V* y) noexcept
    {
        const V t0 = _mm256_unpacklo_pd(r.lo[0], r.lo[1]), t1 = _mm256_unpackhi_pd(r.lo[0], r.lo[1]);
        const V t2 = _mm256_unpacklo_pd(r.lo[2], r.lo[3]), t3 = _mm256_unpackhi_pd(r.lo[2], r.lo[3]);
        y[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
        y[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
        y[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
        y[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
        const __m128d u0 = _mm_unpacklo_pd(r.hi[0], r.hi[1]), v0 = _mm_unpacklo_pd(r.hi[2], r.hi[3]);
        const __m128d u1 = _mm_unpackhi_pd(r.hi[0], r.hi[1]), v1 = _mm_unpackhi_pd(r.hi[2], r.hi[3]);
        y[4] = _mm256_insertf128_pd(_mm256_castpd128_pd256(u0), v0, 1);
        y[5] = _mm256_insertf128_pd(_mm256_castpd128_pd256(u1), v1, 1);
    }
};

#include "detail/WavetableKernels.inl"

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

    /// Floats past the last table that the block reader may load: its AVX2
    /// float lanes read eight samples per interval (offsets -2..+5;
    /// interpolation uses -2..+3), which reaches into the next row's guard,
    /// and past the end of the storage for the very last row.
    static constexpr size_t kTailPad = 2;

    /**
     * Block reader: the level and frame setup once, then the widest kernels
     * the CPU runs, the narrow ones for what they leave, and read() for a
     * group holding an out-of-range phase and for the tail.
     */
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
#if defined(DSPARK_WAVETABLE_NARROW)
        const auto w = static_cast<T>(std::clamp(mip.weight, 0.0f, 1.0f));
        const detail::WavetableRows s = blockSetup<T>(mip, w);
        const int layout = !s.two ? detail::kWavetableOneLevel
                         : (s.same ? detail::kWavetableSameGrid : detail::kWavetableTwoGrids);
        const detail::WavetableFrame f = detail::wavetableFrame(position, numFrames_);
    #if defined(DSPARK_SIMD_EXACT_AVX2) && defined(DSPARK_SIMD_EXACT_AVX2_DISPATCHED)
        const bool wide = simd::isRuntimeDispatchActive();
    #elif defined(DSPARK_SIMD_EXACT_AVX2)
        constexpr bool wide = true;
    #endif
        // One narrow group: the most read() may have to take over at once.
        constexpr int kGroup = std::is_same_v<T, float> ? 4 : 2;
        int j = 0;
        while (j < n)
        {
            const double* pos = positions != nullptr ? positions + j : nullptr;
    #if defined(DSPARK_SIMD_EXACT_AVX2)
            if (wide)
            {
                j += detail::wavetable_avx2::readBlockKernel(s, layout, phases + j, pos, f, w, out + j, n - j);
                pos = positions != nullptr ? positions + j : nullptr;
            }
    #endif
            j += detail::wavetable_narrow::readBlockKernel(s, layout, phases + j, pos, f, w, out + j, n - j);
            for (const int stop = std::min(n, j + kGroup); j < stop; ++j)
                out[j] = read<T>(phases[j], positions != nullptr ? positions[j] : position, mip);
        }
#else
        for (int j = 0; j < n; ++j)
            out[j] = read<T>(phases[j], positions != nullptr ? positions[j] : position, mip);
#endif
    }

#if defined(DSPARK_WAVETABLE_NARROW)
    /** @brief The block's level pair as the kernels see it. */
    template <FloatType T>
    [[nodiscard]] detail::WavetableRows blockSetup(const Mip& mip, T w) const noexcept
    {
        detail::WavetableRows s;
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
                {
                    // No negative zeros: the block kernels blend frames with t = 0
                    // arithmetically, and -0 + 0 would read back as +0.
                    const float v = static_cast<float>(cycle[static_cast<size_t>(i)] * gain);
                    dst[kGuard + i] = (v == 0.0f) ? 0.0f : v;
                }
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
