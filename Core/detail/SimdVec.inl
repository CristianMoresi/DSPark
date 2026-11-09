// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// SimdVec.inl - the vector abstraction and the kernels written on it, included
// once per instruction set into its own namespace (no include guard on
// purpose). SimdOps.h includes it into dspark::simd for the build's baseline
// and, in x86-64 builds below AVX, again into dspark::simd::avx2 inside an
// AVX2+FMA target region, so each copy has its own symbols and the AVX2 code
// runs only after the run-time CPU check. The includer defines the features of
// the copy: DSPARK_VEC_SSE2, DSPARK_VEC_NEON, DSPARK_VEC_AVX, DSPARK_VEC_AVX2
// and DSPARK_VEC_FMA. This file includes nothing; the intrinsic headers come from
// SimdOps.h.

// ============================================================================
// Vec<T, W>: W lanes of T for kernels written once for every target. W = 1 is
// plain scalar code and exists everywhere; 4/2 lanes (float/double) exist on
// SSE2 and NEON, 8/4 lanes with AVX. Used by FFT.h, Oversampling.h and
// firCorrelateKernel() below.
// ============================================================================

template <typename T, int W> struct Vec;

template <typename T>
struct Vec<T, 1>
{
    using V = T;
    static V load(const T* p) noexcept { return *p; }
    static void store(T* p, V v) noexcept { *p = v; }
    static V set1(T x) noexcept { return x; }
    static V add(V a, V b) noexcept { return a + b; }
    static V sub(V a, V b) noexcept { return a - b; }
    static V mul(V a, V b) noexcept { return a * b; }
    static V mulSub(V a, V b, V c, V d) noexcept { return a * b - c * d; }
    static V mulAdd(V a, V b, V c, V d) noexcept { return a * b + c * d; }
    static V madd(V a, V b, V c) noexcept { return a * b + c; }
    static V reverse(V v) noexcept { return v; }
    static void loadDeinterleave(const T* p, V& re, V& im) noexcept { re = p[0]; im = p[1]; }
    static void loadDeinterleaveReversed(const T* p, V& re, V& im) noexcept { re = p[0]; im = p[1]; }
    static void storeInterleave2(T* p, V re, V im) noexcept { p[0] = re; p[1] = im; }
    static void storeInterleave2Reversed(T* p, V re, V im) noexcept { p[0] = re; p[1] = im; }
    static void storeInterleave4(T* p, V a, V b, V c, V d) noexcept
    {
        p[0] = a; p[1] = b; p[2] = c; p[3] = d;
    }
    static void storeInterleave4To(T* p0, T* p1, T* p2, T* p3, V a, V b, V c, V d) noexcept
    {
        *p0 = a; *p1 = b; *p2 = c; *p3 = d;
    }
};

#if defined(DSPARK_VEC_SSE2)
inline constexpr int kVecFloatNarrow = 4;
inline constexpr int kVecDoubleNarrow = 2;

template <>
struct Vec<float, 4>
{
    using V = __m128;
    static V load(const float* p) noexcept { return _mm_loadu_ps(p); }
    static void store(float* p, V v) noexcept { _mm_storeu_ps(p, v); }
    static V set1(float x) noexcept { return _mm_set1_ps(x); }
    static V add(V a, V b) noexcept { return _mm_add_ps(a, b); }
    static V sub(V a, V b) noexcept { return _mm_sub_ps(a, b); }
    static V mul(V a, V b) noexcept { return _mm_mul_ps(a, b); }
#if defined(DSPARK_VEC_FMA)
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm_fmsub_ps(a, b, _mm_mul_ps(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm_fmadd_ps(a, b, _mm_mul_ps(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm_fmadd_ps(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm_sub_ps(_mm_mul_ps(a, b), _mm_mul_ps(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm_add_ps(_mm_mul_ps(a, b), _mm_mul_ps(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm_add_ps(_mm_mul_ps(a, b), c); }
#endif
    static V reverse(V v) noexcept { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3)); }
    static void loadDeinterleave(const float* p, V& re, V& im) noexcept
    {
        const V a = _mm_loadu_ps(p), b = _mm_loadu_ps(p + 4);
        re = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
        im = _mm_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    }
    static void storeInterleave2(float* p, V re, V im) noexcept
    {
        _mm_storeu_ps(p, _mm_unpacklo_ps(re, im));
        _mm_storeu_ps(p + 4, _mm_unpackhi_ps(re, im));
    }
    static void loadDeinterleaveReversed(const float* p, V& re, V& im) noexcept
    {
        loadDeinterleave(p, re, im);
        re = reverse(re);
        im = reverse(im);
    }
    static void storeInterleave2Reversed(float* p, V re, V im) noexcept { storeInterleave2(p, reverse(re), reverse(im)); }
    static void storeInterleave4(float* p, V a, V b, V c, V d) noexcept
    {
        _MM_TRANSPOSE4_PS(a, b, c, d);
        _mm_storeu_ps(p, a); _mm_storeu_ps(p + 4, b);
        _mm_storeu_ps(p + 8, c); _mm_storeu_ps(p + 12, d);
    }
    static void storeInterleave4To(float* p0, float* p1, float* p2, float* p3, V a, V b, V c, V d) noexcept
    {
        _MM_TRANSPOSE4_PS(a, b, c, d);
        _mm_storeu_ps(p0, a); _mm_storeu_ps(p1, b);
        _mm_storeu_ps(p2, c); _mm_storeu_ps(p3, d);
    }
};

template <>
struct Vec<double, 2>
{
    using V = __m128d;
    static V load(const double* p) noexcept { return _mm_loadu_pd(p); }
    static void store(double* p, V v) noexcept { _mm_storeu_pd(p, v); }
    static V set1(double x) noexcept { return _mm_set1_pd(x); }
    static V add(V a, V b) noexcept { return _mm_add_pd(a, b); }
    static V sub(V a, V b) noexcept { return _mm_sub_pd(a, b); }
    static V mul(V a, V b) noexcept { return _mm_mul_pd(a, b); }
#if defined(DSPARK_VEC_FMA)
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm_fmsub_pd(a, b, _mm_mul_pd(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm_fmadd_pd(a, b, _mm_mul_pd(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm_fmadd_pd(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm_sub_pd(_mm_mul_pd(a, b), _mm_mul_pd(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm_add_pd(_mm_mul_pd(a, b), _mm_mul_pd(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm_add_pd(_mm_mul_pd(a, b), c); }
#endif
    static V reverse(V v) noexcept { return _mm_shuffle_pd(v, v, 1); }
    static V set2(double lane0Value, double lane1Value) noexcept { return _mm_set_pd(lane1Value, lane0Value); }
    static double lane0(V v) noexcept { return _mm_cvtsd_f64(v); }
    static double lane1(V v) noexcept { return _mm_cvtsd_f64(_mm_unpackhi_pd(v, v)); }
    static void loadDeinterleave(const double* p, V& re, V& im) noexcept
    {
        const V a = _mm_loadu_pd(p), b = _mm_loadu_pd(p + 2);
        re = _mm_unpacklo_pd(a, b);
        im = _mm_unpackhi_pd(a, b);
    }
    static void storeInterleave2(double* p, V re, V im) noexcept
    {
        _mm_storeu_pd(p, _mm_unpacklo_pd(re, im));
        _mm_storeu_pd(p + 2, _mm_unpackhi_pd(re, im));
    }
    /// loadDeinterleave() with the lanes reversed, in the same two shuffles.
    static void loadDeinterleaveReversed(const double* p, V& re, V& im) noexcept
    {
        const V a = _mm_loadu_pd(p), b = _mm_loadu_pd(p + 2);
        re = _mm_unpacklo_pd(b, a);
        im = _mm_unpackhi_pd(b, a);
    }
    /// storeInterleave2() of the lanes reversed, in the same two shuffles.
    static void storeInterleave2Reversed(double* p, V re, V im) noexcept
    {
        _mm_storeu_pd(p, _mm_unpackhi_pd(re, im));
        _mm_storeu_pd(p + 2, _mm_unpacklo_pd(re, im));
    }
    static void storeInterleave4(double* p, V a, V b, V c, V d) noexcept
    {
        _mm_storeu_pd(p,     _mm_unpacklo_pd(a, b));
        _mm_storeu_pd(p + 2, _mm_unpacklo_pd(c, d));
        _mm_storeu_pd(p + 4, _mm_unpackhi_pd(a, b));
        _mm_storeu_pd(p + 6, _mm_unpackhi_pd(c, d));
    }
    static void storeInterleave4To(double* p0, double* p1, double* p2, double* p3, V a, V b, V c, V d) noexcept
    {
        _mm_storeu_pd(p0, _mm_unpacklo_pd(a, b));
        _mm_storeu_pd(p1, _mm_unpacklo_pd(c, d));
        _mm_storeu_pd(p2, _mm_unpackhi_pd(a, b));
        _mm_storeu_pd(p3, _mm_unpackhi_pd(c, d));
    }
};
#elif defined(DSPARK_VEC_NEON)
inline constexpr int kVecFloatNarrow = 4;
inline constexpr int kVecDoubleNarrow = 2;

template <>
struct Vec<float, 4>
{
    using V = float32x4_t;
    static V load(const float* p) noexcept { return vld1q_f32(p); }
    static void store(float* p, V v) noexcept { vst1q_f32(p, v); }
    static V set1(float x) noexcept { return vdupq_n_f32(x); }
    static V add(V a, V b) noexcept { return vaddq_f32(a, b); }
    static V sub(V a, V b) noexcept { return vsubq_f32(a, b); }
    static V mul(V a, V b) noexcept { return vmulq_f32(a, b); }
    static V mulSub(V a, V b, V c, V d) noexcept { return vfmsq_f32(vmulq_f32(a, b), c, d); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return vfmaq_f32(vmulq_f32(a, b), c, d); }
    // Explicit fusion keeps vector and broadcast tails identical even when
    // GCC otherwise contracts their multiply/add expressions differently.
    static V madd(V a, V b, V c) noexcept { return vfmaq_f32(c, a, b); }
    static V reverse(V v) noexcept { const V r = vrev64q_f32(v); return vextq_f32(r, r, 2); }
    static void loadDeinterleave(const float* p, V& re, V& im) noexcept
    {
        const float32x4x2_t v = vld2q_f32(p);
        re = v.val[0]; im = v.val[1];
    }
    static void storeInterleave2(float* p, V re, V im) noexcept
    {
        float32x4x2_t v; v.val[0] = re; v.val[1] = im;
        vst2q_f32(p, v);
    }
    static void loadDeinterleaveReversed(const float* p, V& re, V& im) noexcept
    {
        loadDeinterleave(p, re, im);
        re = reverse(re);
        im = reverse(im);
    }
    static void storeInterleave2Reversed(float* p, V re, V im) noexcept { storeInterleave2(p, reverse(re), reverse(im)); }
    static void storeInterleave4(float* p, V a, V b, V c, V d) noexcept
    {
        float32x4x4_t v; v.val[0] = a; v.val[1] = b; v.val[2] = c; v.val[3] = d;
        vst4q_f32(p, v);
    }
    static void storeInterleave4To(float* p0, float* p1, float* p2, float* p3, V a, V b, V c, V d) noexcept
    {
        const float32x4x2_t ab = vtrnq_f32(a, b), cd = vtrnq_f32(c, d);
        vst1q_f32(p0, vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0])));
        vst1q_f32(p1, vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1])));
        vst1q_f32(p2, vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0])));
        vst1q_f32(p3, vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1])));
    }
};

template <>
struct Vec<double, 2>
{
    using V = float64x2_t;
    static V load(const double* p) noexcept { return vld1q_f64(p); }
    static void store(double* p, V v) noexcept { vst1q_f64(p, v); }
    static V set1(double x) noexcept { return vdupq_n_f64(x); }
    static V add(V a, V b) noexcept { return vaddq_f64(a, b); }
    static V sub(V a, V b) noexcept { return vsubq_f64(a, b); }
    static V mul(V a, V b) noexcept { return vmulq_f64(a, b); }
    static V mulSub(V a, V b, V c, V d) noexcept { return vfmsq_f64(vmulq_f64(a, b), c, d); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return vfmaq_f64(vmulq_f64(a, b), c, d); }
    static V madd(V a, V b, V c) noexcept { return vfmaq_f64(c, a, b); }
    static V reverse(V v) noexcept { return vextq_f64(v, v, 1); }
    static V set2(double lane0Value, double lane1Value) noexcept { return vsetq_lane_f64(lane1Value, vdupq_n_f64(lane0Value), 1); }
    static double lane0(V v) noexcept { return vgetq_lane_f64(v, 0); }
    static double lane1(V v) noexcept { return vgetq_lane_f64(v, 1); }
    static void loadDeinterleave(const double* p, V& re, V& im) noexcept
    {
        const float64x2x2_t v = vld2q_f64(p);
        re = v.val[0]; im = v.val[1];
    }
    static void storeInterleave2(double* p, V re, V im) noexcept
    {
        float64x2x2_t v; v.val[0] = re; v.val[1] = im;
        vst2q_f64(p, v);
    }
    static void loadDeinterleaveReversed(const double* p, V& re, V& im) noexcept
    {
        const V a = vld1q_f64(p), b = vld1q_f64(p + 2);
        re = vzip1q_f64(b, a);
        im = vzip2q_f64(b, a);
    }
    static void storeInterleave2Reversed(double* p, V re, V im) noexcept
    {
        vst1q_f64(p, vzip2q_f64(re, im));
        vst1q_f64(p + 2, vzip1q_f64(re, im));
    }
    static void storeInterleave4(double* p, V a, V b, V c, V d) noexcept
    {
        float64x2x4_t v; v.val[0] = a; v.val[1] = b; v.val[2] = c; v.val[3] = d;
        vst4q_f64(p, v);
    }
    static void storeInterleave4To(double* p0, double* p1, double* p2, double* p3, V a, V b, V c, V d) noexcept
    {
        vst1q_f64(p0, vzip1q_f64(a, b));
        vst1q_f64(p1, vzip1q_f64(c, d));
        vst1q_f64(p2, vzip2q_f64(a, b));
        vst1q_f64(p3, vzip2q_f64(c, d));
    }
};
#else
inline constexpr int kVecFloatNarrow = 1;
inline constexpr int kVecDoubleNarrow = 1;
#endif

#if defined(DSPARK_VEC_AVX)
inline constexpr int kVecFloatWide = 8;
inline constexpr int kVecDoubleWide = 4;

template <>
struct Vec<float, 8>
{
    using V = __m256;
    static V load(const float* p) noexcept { return _mm256_loadu_ps(p); }
    static void store(float* p, V v) noexcept { _mm256_storeu_ps(p, v); }
    static V set1(float x) noexcept { return _mm256_set1_ps(x); }
    static V add(V a, V b) noexcept { return _mm256_add_ps(a, b); }
    static V sub(V a, V b) noexcept { return _mm256_sub_ps(a, b); }
    static V mul(V a, V b) noexcept { return _mm256_mul_ps(a, b); }
#if defined(DSPARK_VEC_FMA)
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm256_fmsub_ps(a, b, _mm256_mul_ps(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm256_fmadd_ps(a, b, _mm256_mul_ps(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm256_fmadd_ps(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm256_sub_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm256_add_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
#endif
    static V reverse(V v) noexcept
    {
#if defined(DSPARK_VEC_AVX2)
        return _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
        const V swapped = _mm256_permute2f128_ps(v, v, 1);
        return _mm256_permute_ps(swapped, _MM_SHUFFLE(0, 1, 2, 3));
#endif
    }
    static void loadDeinterleave(const float* p, V& re, V& im) noexcept
    {
        const V a = _mm256_loadu_ps(p), b = _mm256_loadu_ps(p + 8);
        const V lo = _mm256_permute2f128_ps(a, b, 0x20);
        const V hi = _mm256_permute2f128_ps(a, b, 0x31);
        re = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0));
        im = _mm256_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1));
    }
    static void storeInterleave2(float* p, V re, V im) noexcept
    {
        const V lo = _mm256_unpacklo_ps(re, im);
        const V hi = _mm256_unpackhi_ps(re, im);
        _mm256_storeu_ps(p,     _mm256_permute2f128_ps(lo, hi, 0x20));
        _mm256_storeu_ps(p + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
    }
    static void loadDeinterleaveReversed(const float* p, V& re, V& im) noexcept
    {
        loadDeinterleave(p, re, im);
        re = reverse(re);
        im = reverse(im);
    }
    static void storeInterleave2Reversed(float* p, V re, V im) noexcept { storeInterleave2(p, reverse(re), reverse(im)); }
    static void storeInterleave4(float* p, V a, V b, V c, V d) noexcept
    {
        const V t0 = _mm256_unpacklo_ps(a, b), t1 = _mm256_unpackhi_ps(a, b);
        const V t2 = _mm256_unpacklo_ps(c, d), t3 = _mm256_unpackhi_ps(c, d);
        const V u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const V u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const V u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const V u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        _mm256_storeu_ps(p,      _mm256_permute2f128_ps(u0, u1, 0x20));
        _mm256_storeu_ps(p + 8,  _mm256_permute2f128_ps(u2, u3, 0x20));
        _mm256_storeu_ps(p + 16, _mm256_permute2f128_ps(u0, u1, 0x31));
        _mm256_storeu_ps(p + 24, _mm256_permute2f128_ps(u2, u3, 0x31));
    }
    static void storeInterleave4To(float* p0, float* p1, float* p2, float* p3, V a, V b, V c, V d) noexcept
    {
        const V t0 = _mm256_unpacklo_ps(a, b), t1 = _mm256_unpackhi_ps(a, b);
        const V t2 = _mm256_unpacklo_ps(c, d), t3 = _mm256_unpackhi_ps(c, d);
        const V u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
        const V u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
        const V u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
        const V u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
        _mm256_storeu_ps(p0, _mm256_permute2f128_ps(u0, u1, 0x20));
        _mm256_storeu_ps(p1, _mm256_permute2f128_ps(u2, u3, 0x20));
        _mm256_storeu_ps(p2, _mm256_permute2f128_ps(u0, u1, 0x31));
        _mm256_storeu_ps(p3, _mm256_permute2f128_ps(u2, u3, 0x31));
    }
};

template <>
struct Vec<double, 4>
{
    using V = __m256d;
    static V load(const double* p) noexcept { return _mm256_loadu_pd(p); }
    static void store(double* p, V v) noexcept { _mm256_storeu_pd(p, v); }
    static V set1(double x) noexcept { return _mm256_set1_pd(x); }
    static V add(V a, V b) noexcept { return _mm256_add_pd(a, b); }
    static V sub(V a, V b) noexcept { return _mm256_sub_pd(a, b); }
    static V mul(V a, V b) noexcept { return _mm256_mul_pd(a, b); }
#if defined(DSPARK_VEC_FMA)
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm256_fmsub_pd(a, b, _mm256_mul_pd(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm256_fmadd_pd(a, b, _mm256_mul_pd(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm256_fmadd_pd(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm256_sub_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm256_add_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }
#endif
    static V reverse(V v) noexcept
    {
#if defined(DSPARK_VEC_AVX2)
        return _mm256_permute4x64_pd(v, _MM_SHUFFLE(0, 1, 2, 3));
#else
        const V swapped = _mm256_permute2f128_pd(v, v, 1);
        return _mm256_permute_pd(swapped, 0x5);
#endif
    }
    static void loadDeinterleave(const double* p, V& re, V& im) noexcept
    {
        const V a = _mm256_loadu_pd(p), b = _mm256_loadu_pd(p + 4);
        const V lo = _mm256_permute2f128_pd(a, b, 0x20);
        const V hi = _mm256_permute2f128_pd(a, b, 0x31);
        re = _mm256_unpacklo_pd(lo, hi);
        im = _mm256_unpackhi_pd(lo, hi);
    }
    static void storeInterleave2(double* p, V re, V im) noexcept
    {
        const V lo = _mm256_unpacklo_pd(re, im);
        const V hi = _mm256_unpackhi_pd(re, im);
        _mm256_storeu_pd(p,     _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(p + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
    /// loadDeinterleave() with the lanes reversed, in the same four shuffles.
    static void loadDeinterleaveReversed(const double* p, V& re, V& im) noexcept
    {
        const V a = _mm256_loadu_pd(p), b = _mm256_loadu_pd(p + 4);
        const V lo = _mm256_permute2f128_pd(b, a, 0x31);   // x3 y3 | x1 y1
        const V hi = _mm256_permute2f128_pd(b, a, 0x20);   // x2 y2 | x0 y0
        re = _mm256_unpacklo_pd(lo, hi);
        im = _mm256_unpackhi_pd(lo, hi);
    }
    /// storeInterleave2() of the lanes reversed, in the same four shuffles.
    static void storeInterleave2Reversed(double* p, V re, V im) noexcept
    {
        const V lo = _mm256_unpacklo_pd(re, im);   // r0 i0 | r2 i2
        const V hi = _mm256_unpackhi_pd(re, im);   // r1 i1 | r3 i3
        _mm256_storeu_pd(p,     _mm256_permute2f128_pd(hi, lo, 0x31));
        _mm256_storeu_pd(p + 4, _mm256_permute2f128_pd(hi, lo, 0x20));
    }
    static void storeInterleave4(double* p, V a, V b, V c, V d) noexcept
    {
        const V t0 = _mm256_unpacklo_pd(a, b), t1 = _mm256_unpackhi_pd(a, b);
        const V t2 = _mm256_unpacklo_pd(c, d), t3 = _mm256_unpackhi_pd(c, d);
        _mm256_storeu_pd(p,      _mm256_permute2f128_pd(t0, t2, 0x20));
        _mm256_storeu_pd(p + 4,  _mm256_permute2f128_pd(t1, t3, 0x20));
        _mm256_storeu_pd(p + 8,  _mm256_permute2f128_pd(t0, t2, 0x31));
        _mm256_storeu_pd(p + 12, _mm256_permute2f128_pd(t1, t3, 0x31));
    }
    static void storeInterleave4To(double* p0, double* p1, double* p2, double* p3, V a, V b, V c, V d) noexcept
    {
        const V t0 = _mm256_unpacklo_pd(a, b), t1 = _mm256_unpackhi_pd(a, b);
        const V t2 = _mm256_unpacklo_pd(c, d), t3 = _mm256_unpackhi_pd(c, d);
        _mm256_storeu_pd(p0, _mm256_permute2f128_pd(t0, t2, 0x20));
        _mm256_storeu_pd(p1, _mm256_permute2f128_pd(t1, t3, 0x20));
        _mm256_storeu_pd(p2, _mm256_permute2f128_pd(t0, t2, 0x31));
        _mm256_storeu_pd(p3, _mm256_permute2f128_pd(t1, t3, 0x31));
    }
};
#else
inline constexpr int kVecFloatWide = kVecFloatNarrow;
inline constexpr int kVecDoubleWide = kVecDoubleNarrow;
#endif

/// Widest vector width for T on this target (1 = scalar).
template <typename T> inline constexpr int kVecWidth = std::is_same_v<T, float> ? kVecFloatWide : kVecDoubleWide;
/// The next width down (the SSE2/NEON width in AVX builds).
template <typename T> inline constexpr int kVecNarrowWidth = std::is_same_v<T, float> ? kVecFloatNarrow : kVecDoubleNarrow;


/// firCorrelate() at this copy's widest vector (see SimdOps.h).
template <typename T>
void firCorrelateKernel(const T* x, const T* h, int taps, T* DSPARK_RESTRICT y, int n, T gain) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    constexpr int W = kVecWidth<T>;
    using O = Vec<T, W>;
    using V = typename O::V;
    const V g = O::set1(gain);
    int i = 0;
    for (; i + 4 * W <= n; i += 4 * W)
    {
        V a0 = O::set1(T(0)), a1 = a0, a2 = a0, a3 = a0;
        const T* xp = x + i;
        for (int j = 0; j < taps; ++j)
        {
            const V hj = O::set1(h[j]);
            a0 = O::madd(hj, O::load(xp + j), a0);
            a1 = O::madd(hj, O::load(xp + j + W), a1);
            a2 = O::madd(hj, O::load(xp + j + 2 * W), a2);
            a3 = O::madd(hj, O::load(xp + j + 3 * W), a3);
        }
        O::store(y + i, O::mul(a0, g));
        O::store(y + i + W, O::mul(a1, g));
        O::store(y + i + 2 * W, O::mul(a2, g));
        O::store(y + i + 3 * W, O::mul(a3, g));
    }
    for (; i + W <= n; i += W)
    {
        V a = O::set1(T(0));
        for (int j = 0; j < taps; ++j)
            a = O::madd(O::set1(h[j]), O::load(x + i + j), a);
        O::store(y + i, O::mul(a, g));
    }
    for (; i < n; ++i)
    {
        // Use the same multiply-add policy as the full vectors. A scalar
        // expression may contract differently, making the result depend on
        // which samples happen to fall in the tail of a host block.
        V a = O::set1(T(0));
        for (int j = 0; j < taps; ++j)
            a = O::madd(O::set1(h[j]), O::set1(x[i + j]), a);
        T lanes[W];
        O::store(lanes, O::mul(a, g));
        y[i] = lanes[0];
    }
}

/// firCorrelateBank() at this copy's widest vector (see SimdOps.h). The lanes
/// hold banks, so each output of each bank is the same multiply-add chain over
/// the taps as in firCorrelateKernel() with unit gain.
template <typename T, int Banks>
void firCorrelateBankKernel(const T* x, const T* h, int taps, T* const* y, int n) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    constexpr int W = Banks % kVecWidth<T> == 0 ? kVecWidth<T> : 1;
    constexpr int Vecs = Banks / W;
    constexpr int Outputs = Vecs >= 8 ? 1 : 8 / Vecs;   // keep eight accumulators live
    using O = Vec<T, W>;
    using V = typename O::V;
    alignas(64) T lanes[Banks];
    int i = 0;
    const auto block = [&](auto outputs) {
        constexpr int P = decltype(outputs)::value;
        V acc[P][Vecs];
        for (int p = 0; p < P; ++p)
            for (int v = 0; v < Vecs; ++v) acc[p][v] = O::set1(T(0));
        const T* xp = x + i;
        for (int j = 0; j < taps; ++j)
        {
            V c[Vecs];
            for (int v = 0; v < Vecs; ++v) c[v] = O::load(h + j * Banks + v * W);
            for (int p = 0; p < P; ++p)
            {
                const V s = O::set1(xp[p + j]);
                for (int v = 0; v < Vecs; ++v) acc[p][v] = O::madd(c[v], s, acc[p][v]);
            }
        }
        for (int p = 0; p < P; ++p)
        {
            for (int v = 0; v < Vecs; ++v) O::store(lanes + v * W, acc[p][v]);
            for (int k = 0; k < Banks; ++k) y[k][i + p] = lanes[k];
        }
    };
    for (; i + Outputs <= n; i += Outputs) block(std::integral_constant<int, Outputs>{});
    for (; i < n; ++i) block(std::integral_constant<int, 1>{});
}

/// firCorrelateBankSum() at this copy's widest vector (see SimdOps.h).
template <typename T, int Banks>
void firCorrelateBankSumKernel(const T* m, const T* h, int taps, T* DSPARK_RESTRICT y, int n) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "SimdOps: only float and double are supported");
    constexpr int W = Banks % kVecWidth<T> == 0 ? kVecWidth<T> : 1;
    constexpr int Vecs = Banks / W;
    constexpr int Outputs = Vecs >= 8 ? 1 : 8 / Vecs;
    using O = Vec<T, W>;
    using V = typename O::V;
    alignas(64) T lanes[Banks];
    int i = 0;
    const auto block = [&](auto outputs) {
        constexpr int P = decltype(outputs)::value;
        V acc[P][Vecs];
        for (int p = 0; p < P; ++p)
            for (int v = 0; v < Vecs; ++v) acc[p][v] = O::set1(T(0));
        const T* mp = m + static_cast<std::ptrdiff_t>(i) * Banks;
        for (int j = 0; j < taps; ++j)
        {
            V c[Vecs];
            for (int v = 0; v < Vecs; ++v) c[v] = O::load(h + j * Banks + v * W);
            for (int p = 0; p < P; ++p)
                for (int v = 0; v < Vecs; ++v)
                    acc[p][v] = O::madd(c[v], O::load(mp + (p + j) * Banks + v * W), acc[p][v]);
        }
        for (int p = 0; p < P; ++p)
        {
            for (int v = 0; v < Vecs; ++v) O::store(lanes + v * W, acc[p][v]);
            T sum = T(0);
            for (int k = 0; k < Banks; ++k) sum += lanes[k];
            y[i + p] = sum;
        }
    };
    for (; i + Outputs <= n; i += Outputs) block(std::integral_constant<int, Outputs>{});
    for (; i < n; ++i) block(std::integral_constant<int, 1>{});
}

#if defined(DSPARK_VEC_AVX)
/// dotProduct() at 8 float lanes.
inline float dotProductWide(const float* DSPARK_RESTRICT a, const float* DSPARK_RESTRICT b, int count) noexcept
{
    // Four independent accumulators break the FMA latency chain (4-5 cycles),
    // letting long FIR kernels run at full multiply throughput (~2-4x faster).
    __m256 vSum0 = _mm256_setzero_ps();
    __m256 vSum1 = _mm256_setzero_ps();
    __m256 vSum2 = _mm256_setzero_ps();
    __m256 vSum3 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 31 < count; i += 32)
    {
        #if defined(DSPARK_VEC_FMA)
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
        #if defined(DSPARK_VEC_FMA)
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
}

/// dotProduct() at 4 double lanes.
inline double dotProductWide(const double* DSPARK_RESTRICT a, const double* DSPARK_RESTRICT b, int count) noexcept
{
    __m256d vSum0 = _mm256_setzero_pd();
    __m256d vSum1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        #if defined(DSPARK_VEC_FMA)
            vSum0 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i),     vSum0);
            vSum1 = _mm256_fmadd_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4), vSum1);
        #else
            vSum0 = _mm256_add_pd(vSum0, _mm256_mul_pd(_mm256_loadu_pd(a + i),     _mm256_loadu_pd(b + i)));
            vSum1 = _mm256_add_pd(vSum1, _mm256_mul_pd(_mm256_loadu_pd(a + i + 4), _mm256_loadu_pd(b + i + 4)));
        #endif
    }
    for (; i + 3 < count; i += 4)
    {
        #if defined(DSPARK_VEC_FMA)
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
}

/// complexMulAccum() at 4 complex float bins per iteration.
inline void complexMulAccumWide(float* DSPARK_RESTRICT accum, const float* DSPARK_RESTRICT a,
                                const float* DSPARK_RESTRICT b, int bins) noexcept
{
    // vpermilps builds the [re,re] / [im,im] duplicates and the [im,re] swap
    // within each 128-bit lane; the alternating subtract/add of the two
    // partial products comes straight from (fm)addsub, so no sign mask is
    // needed on this path.
    int k = 0;
    for (; k + 3 < bins; k += 4)
    {
        const __m256 va   = _mm256_loadu_ps(a + 2 * k);
        const __m256 vb   = _mm256_loadu_ps(b + 2 * k);
        const __m256 vacc = _mm256_loadu_ps(accum + 2 * k);

        const __m256 aRe   = _mm256_permute_ps(va, _MM_SHUFFLE(2, 2, 0, 0));
        const __m256 aIm   = _mm256_permute_ps(va, _MM_SHUFFLE(3, 3, 1, 1));
        const __m256 bSwap = _mm256_permute_ps(vb, _MM_SHUFFLE(2, 3, 0, 1));

        #if defined(DSPARK_VEC_FMA)
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
}
#endif
