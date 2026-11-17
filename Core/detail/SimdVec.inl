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
// SSE2 and NEON, 8/4 lanes with AVX. Used by FFT.h, Oversampling.h,
// TruePeakDetector.h and firCorrelateKernel() below. The vector widths also
// compare, select and take magnitudes and quotients (see Vec<float, 4>).
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
    static V nmadd(V a, V b, V c) noexcept { return c - a * b; }
    static void transpose(V*) noexcept {}
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
    /// The sum of the lanes, halves added pairwise ((l0 + l2) + (l1 + l3) at
    /// four lanes); sumLanes4() writes four of them in the same order.
    static T sumLanes(V v) noexcept { return v; }
    static void sumLanes4(T* out, V a, V b, V c, V d) noexcept
    {
        out[0] = a; out[1] = b; out[2] = c; out[3] = d;
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
    static V nmadd(V a, V b, V c) noexcept { return _mm_fnmadd_ps(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm_sub_ps(_mm_mul_ps(a, b), _mm_mul_ps(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm_add_ps(_mm_mul_ps(a, b), _mm_mul_ps(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm_add_ps(_mm_mul_ps(a, b), c); }
    static V nmadd(V a, V b, V c) noexcept { return _mm_sub_ps(c, _mm_mul_ps(a, b)); }
#endif
    /// Comparisons give a mask M (all bits of a lane set where it holds; false
    /// where an operand is not a number); both() and either() combine two,
    /// any() tells whether a lane is set and select() takes a where the mask
    /// is set, b elsewhere. maxOf(a, b) is std::max(a, b): a unless a < b, so
    /// a value that is not a number is ignored in b and kept in a.
    /// maxNum(a, b) is the larger of two numbers in one instruction; with a
    /// value that is not a number its result depends on the target.
    using M = __m128;
    static V abs(V v) noexcept { return _mm_andnot_ps(_mm_set1_ps(-0.0f), v); }
    static V div(V a, V b) noexcept { return _mm_div_ps(a, b); }
    static V maxOf(V a, V b) noexcept { return _mm_max_ps(b, a); }
    static V maxNum(V a, V b) noexcept { return _mm_max_ps(a, b); }
    static M lt(V a, V b) noexcept { return _mm_cmplt_ps(a, b); }
    static M le(V a, V b) noexcept { return _mm_cmple_ps(a, b); }
    static M both(M a, M b) noexcept { return _mm_and_ps(a, b); }
    static M either(M a, M b) noexcept { return _mm_or_ps(a, b); }
    static bool any(M m) noexcept { return _mm_movemask_ps(m) != 0; }
    static V select(M m, V a, V b) noexcept { return _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b)); }
    static V reverse(V v) noexcept { return _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 1, 2, 3)); }
    /// Transposes W vectors in place: v[k] receives lane k of each, in order.
    static void transpose(V* v) noexcept
    {
        const V a = _mm_unpacklo_ps(v[0], v[1]), b = _mm_unpacklo_ps(v[2], v[3]);
        const V c = _mm_unpackhi_ps(v[0], v[1]), d = _mm_unpackhi_ps(v[2], v[3]);
        v[0] = _mm_movelh_ps(a, b);
        v[1] = _mm_movehl_ps(b, a);
        v[2] = _mm_movelh_ps(c, d);
        v[3] = _mm_movehl_ps(d, c);
    }
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
    static V nmadd(V a, V b, V c) noexcept { return _mm_fnmadd_pd(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm_sub_pd(_mm_mul_pd(a, b), _mm_mul_pd(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm_add_pd(_mm_mul_pd(a, b), _mm_mul_pd(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm_add_pd(_mm_mul_pd(a, b), c); }
    static V nmadd(V a, V b, V c) noexcept { return _mm_sub_pd(c, _mm_mul_pd(a, b)); }
#endif
    /// Masks, select() and maxOf() as in Vec<float, 4>.
    using M = __m128d;
    static V abs(V v) noexcept { return _mm_andnot_pd(_mm_set1_pd(-0.0), v); }
    static V div(V a, V b) noexcept { return _mm_div_pd(a, b); }
    static V maxOf(V a, V b) noexcept { return _mm_max_pd(b, a); }
    static V maxNum(V a, V b) noexcept { return _mm_max_pd(a, b); }
    static M lt(V a, V b) noexcept { return _mm_cmplt_pd(a, b); }
    static M le(V a, V b) noexcept { return _mm_cmple_pd(a, b); }
    static M both(M a, M b) noexcept { return _mm_and_pd(a, b); }
    static M either(M a, M b) noexcept { return _mm_or_pd(a, b); }
    static bool any(M m) noexcept { return _mm_movemask_pd(m) != 0; }
    static V select(M m, V a, V b) noexcept { return _mm_or_pd(_mm_and_pd(m, a), _mm_andnot_pd(m, b)); }
    static V reverse(V v) noexcept { return _mm_shuffle_pd(v, v, 1); }
    /// transpose() as in Vec<float, 4>.
    static void transpose(V* v) noexcept
    {
        const V a = _mm_unpacklo_pd(v[0], v[1]);
        v[1] = _mm_unpackhi_pd(v[0], v[1]);
        v[0] = a;
    }
    static V set2(double lane0Value, double lane1Value) noexcept { return _mm_set_pd(lane1Value, lane0Value); }
    static double lane0(V v) noexcept { return _mm_cvtsd_f64(v); }
    static double lane1(V v) noexcept { return _mm_cvtsd_f64(_mm_unpackhi_pd(v, v)); }
    static double sumLanes(V v) noexcept { return _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v))); }
    static void sumLanes4(double* out, V a, V b, V c, V d) noexcept
    {
        _mm_storeu_pd(out, _mm_add_pd(_mm_unpacklo_pd(a, b), _mm_unpackhi_pd(a, b)));
        _mm_storeu_pd(out + 2, _mm_add_pd(_mm_unpacklo_pd(c, d), _mm_unpackhi_pd(c, d)));
    }
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
    static V nmadd(V a, V b, V c) noexcept { return vfmsq_f32(c, a, b); }
    /// Comparisons give a mask M (all bits of a lane set where it holds; false
    /// where an operand is not a number); both() and either() combine two,
    /// any() tells whether a lane is set and select() takes a where the mask
    /// is set, b elsewhere. maxOf(a, b) is std::max(a, b): a unless a < b, so
    /// a value that is not a number is ignored in b and kept in a.
    /// maxNum(a, b) is the larger of two numbers in one instruction; with a
    /// value that is not a number its result depends on the target.
    using M = uint32x4_t;
    static V abs(V v) noexcept { return vabsq_f32(v); }
    static V div(V a, V b) noexcept { return vdivq_f32(a, b); }
    static V maxOf(V a, V b) noexcept { return vbslq_f32(vcltq_f32(a, b), b, a); }
    static V maxNum(V a, V b) noexcept { return vmaxnmq_f32(a, b); }
    static M lt(V a, V b) noexcept { return vcltq_f32(a, b); }
    static M le(V a, V b) noexcept { return vcleq_f32(a, b); }
    static M both(M a, M b) noexcept { return vandq_u32(a, b); }
    static M either(M a, M b) noexcept { return vorrq_u32(a, b); }
    static bool any(M m) noexcept { return vmaxvq_u32(m) != 0; }
    static V select(M m, V a, V b) noexcept { return vbslq_f32(m, a, b); }
    static V reverse(V v) noexcept { const V r = vrev64q_f32(v); return vextq_f32(r, r, 2); }
    /// Transposes W vectors in place: v[k] receives lane k of each, in order.
    static void transpose(V* v) noexcept
    {
        const float32x4x2_t ab = vtrnq_f32(v[0], v[1]), cd = vtrnq_f32(v[2], v[3]);
        v[0] = vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0]));
        v[1] = vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1]));
        v[2] = vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0]));
        v[3] = vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]));
    }
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
    static V nmadd(V a, V b, V c) noexcept { return vfmsq_f64(c, a, b); }
    /// Masks, select() and maxOf() as in Vec<float, 4>.
    using M = uint64x2_t;
    static V abs(V v) noexcept { return vabsq_f64(v); }
    static V div(V a, V b) noexcept { return vdivq_f64(a, b); }
    static V maxOf(V a, V b) noexcept { return vbslq_f64(vcltq_f64(a, b), b, a); }
    static V maxNum(V a, V b) noexcept { return vmaxnmq_f64(a, b); }
    static M lt(V a, V b) noexcept { return vcltq_f64(a, b); }
    static M le(V a, V b) noexcept { return vcleq_f64(a, b); }
    static M both(M a, M b) noexcept { return vandq_u64(a, b); }
    static M either(M a, M b) noexcept { return vorrq_u64(a, b); }
    static bool any(M m) noexcept { return (vgetq_lane_u64(m, 0) | vgetq_lane_u64(m, 1)) != 0; }
    static V select(M m, V a, V b) noexcept { return vbslq_f64(m, a, b); }
    static V reverse(V v) noexcept { return vextq_f64(v, v, 1); }
    /// transpose() as in Vec<float, 4>.
    static void transpose(V* v) noexcept
    {
        const V a = vzip1q_f64(v[0], v[1]);
        v[1] = vzip2q_f64(v[0], v[1]);
        v[0] = a;
    }
    static V set2(double lane0Value, double lane1Value) noexcept { return vsetq_lane_f64(lane1Value, vdupq_n_f64(lane0Value), 1); }
    static double lane0(V v) noexcept { return vgetq_lane_f64(v, 0); }
    static double lane1(V v) noexcept { return vgetq_lane_f64(v, 1); }
    static double sumLanes(V v) noexcept { return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1); }
    static void sumLanes4(double* out, V a, V b, V c, V d) noexcept
    {
        vst1q_f64(out, vpaddq_f64(a, b));
        vst1q_f64(out + 2, vpaddq_f64(c, d));
    }
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
    static V nmadd(V a, V b, V c) noexcept { return _mm256_fnmadd_ps(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm256_sub_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm256_add_ps(_mm256_mul_ps(a, b), _mm256_mul_ps(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
    static V nmadd(V a, V b, V c) noexcept { return _mm256_sub_ps(c, _mm256_mul_ps(a, b)); }
#endif
    /// Masks, select() and maxOf() as in Vec<float, 4>.
    using M = __m256;
    static V abs(V v) noexcept { return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), v); }
    static V div(V a, V b) noexcept { return _mm256_div_ps(a, b); }
    static V maxOf(V a, V b) noexcept { return _mm256_max_ps(b, a); }
    static V maxNum(V a, V b) noexcept { return _mm256_max_ps(a, b); }
    static M lt(V a, V b) noexcept { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
    static M le(V a, V b) noexcept { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
    static M both(M a, M b) noexcept { return _mm256_and_ps(a, b); }
    static M either(M a, M b) noexcept { return _mm256_or_ps(a, b); }
    static bool any(M m) noexcept { return _mm256_movemask_ps(m) != 0; }
    static V select(M m, V a, V b) noexcept { return _mm256_blendv_ps(b, a, m); }
    static V reverse(V v) noexcept
    {
#if defined(DSPARK_VEC_AVX2)
        return _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
        const V swapped = _mm256_permute2f128_ps(v, v, 1);
        return _mm256_permute_ps(swapped, _MM_SHUFFLE(0, 1, 2, 3));
#endif
    }
    /// transpose() as in Vec<float, 4>.
    static void transpose(V* v) noexcept
    {
        const V a0 = _mm256_unpacklo_ps(v[0], v[1]), a1 = _mm256_unpackhi_ps(v[0], v[1]);
        const V a2 = _mm256_unpacklo_ps(v[2], v[3]), a3 = _mm256_unpackhi_ps(v[2], v[3]);
        const V a4 = _mm256_unpacklo_ps(v[4], v[5]), a5 = _mm256_unpackhi_ps(v[4], v[5]);
        const V a6 = _mm256_unpacklo_ps(v[6], v[7]), a7 = _mm256_unpackhi_ps(v[6], v[7]);
        const V b0 = _mm256_shuffle_ps(a0, a2, 0x44), b1 = _mm256_shuffle_ps(a0, a2, 0xEE);
        const V b2 = _mm256_shuffle_ps(a1, a3, 0x44), b3 = _mm256_shuffle_ps(a1, a3, 0xEE);
        const V b4 = _mm256_shuffle_ps(a4, a6, 0x44), b5 = _mm256_shuffle_ps(a4, a6, 0xEE);
        const V b6 = _mm256_shuffle_ps(a5, a7, 0x44), b7 = _mm256_shuffle_ps(a5, a7, 0xEE);
        v[0] = _mm256_permute2f128_ps(b0, b4, 0x20);
        v[1] = _mm256_permute2f128_ps(b1, b5, 0x20);
        v[2] = _mm256_permute2f128_ps(b2, b6, 0x20);
        v[3] = _mm256_permute2f128_ps(b3, b7, 0x20);
        v[4] = _mm256_permute2f128_ps(b0, b4, 0x31);
        v[5] = _mm256_permute2f128_ps(b1, b5, 0x31);
        v[6] = _mm256_permute2f128_ps(b2, b6, 0x31);
        v[7] = _mm256_permute2f128_ps(b3, b7, 0x31);
    }
    /// Transposes v[0 .. 3] and v[4 .. 7] within each 128-bit half: v[g + a]
    /// receives lanes a and a + 4 of the four vectors of its group, the
    /// first in its low half and the second in its high half. No shuffle
    /// crosses the halves; storeHalves() and loadHalves() move the halves
    /// to and from where a full transposition would put them.
    static void transposeHalves(V* v) noexcept
    {
        // Written out, as transpose() is: every compiler then keeps the
        // vectors in registers.
        const V a0 = _mm256_unpacklo_ps(v[0], v[1]), b0 = _mm256_unpacklo_ps(v[2], v[3]);
        const V c0 = _mm256_unpackhi_ps(v[0], v[1]), d0 = _mm256_unpackhi_ps(v[2], v[3]);
        const V a1 = _mm256_unpacklo_ps(v[4], v[5]), b1 = _mm256_unpacklo_ps(v[6], v[7]);
        const V c1 = _mm256_unpackhi_ps(v[4], v[5]), d1 = _mm256_unpackhi_ps(v[6], v[7]);
        v[0] = _mm256_shuffle_ps(a0, b0, 0x44);
        v[1] = _mm256_shuffle_ps(a0, b0, 0xEE);
        v[2] = _mm256_shuffle_ps(c0, d0, 0x44);
        v[3] = _mm256_shuffle_ps(c0, d0, 0xEE);
        v[4] = _mm256_shuffle_ps(a1, b1, 0x44);
        v[5] = _mm256_shuffle_ps(a1, b1, 0xEE);
        v[6] = _mm256_shuffle_ps(c1, d1, 0x44);
        v[7] = _mm256_shuffle_ps(c1, d1, 0xEE);
    }
    /// A vector from two runs of four values: its low and its high half.
    static V loadHalves(const float* low, const float* high) noexcept
    {
        return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(low)), _mm_loadu_ps(high), 1);
    }
    static void storeHalves(float* low, float* high, V v) noexcept
    {
        _mm_storeu_ps(low, _mm256_castps256_ps128(v));
        _mm_storeu_ps(high, _mm256_extractf128_ps(v, 1));
    }
    /// loadDeinterleaveInLane() of eight pairs taken backwards: lane l
    /// holds pair 7 - b, b the pair that lane holds in the forward order
    /// (0, 1, 4, 5, 2, 3, 6, 7). The halves are loaded where they go.
    static void loadDeinterleaveReversedInLane(const float* p, V& re, V& im) noexcept
    {
        const V x = loadHalves(p + 12, p + 8), y = loadHalves(p + 4, p);
        re = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(0, 2, 0, 2));
        im = _mm256_shuffle_ps(x, y, _MM_SHUFFLE(1, 3, 1, 3));
    }
    /// storeInterleave2() of lanes in the order (0, 1, 4, 5, 2, 3, 6, 7):
    /// each half then interleaves into one run of the output.
    static void storeInterleave2InLane(float* p, V re, V im) noexcept
    {
        _mm256_storeu_ps(p,     _mm256_unpacklo_ps(re, im));
        _mm256_storeu_ps(p + 8, _mm256_unpackhi_ps(re, im));
    }
    /// storeInterleave2InLane() with the pairs written backwards.
    static void storeInterleave2ReversedInLane(float* p, V re, V im) noexcept
    {
        const V lo = _mm256_permute_ps(_mm256_unpacklo_ps(re, im), _MM_SHUFFLE(1, 0, 3, 2));
        const V hi = _mm256_permute_ps(_mm256_unpackhi_ps(re, im), _MM_SHUFFLE(1, 0, 3, 2));
        storeHalves(p + 4, p, hi);
        storeHalves(p + 12, p + 8, lo);
    }
    /// loadDeinterleave() without crossing the 128-bit halves: the lanes come
    /// out in the order (0, 1, 4, 5, 2, 3, 6, 7).
    static void loadDeinterleaveInLane(const float* p, V& re, V& im) noexcept
    {
        const V a = _mm256_loadu_ps(p), b = _mm256_loadu_ps(p + 8);
        re = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
        im = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
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
    static V nmadd(V a, V b, V c) noexcept { return _mm256_fnmadd_pd(a, b, c); }
#else
    static V mulSub(V a, V b, V c, V d) noexcept { return _mm256_sub_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d)); }
    static V mulAdd(V a, V b, V c, V d) noexcept { return _mm256_add_pd(_mm256_mul_pd(a, b), _mm256_mul_pd(c, d)); }
    static V madd(V a, V b, V c) noexcept { return _mm256_add_pd(_mm256_mul_pd(a, b), c); }
    static V nmadd(V a, V b, V c) noexcept { return _mm256_sub_pd(c, _mm256_mul_pd(a, b)); }
#endif
    /// The sum of the lanes, (l0 + l2) + (l1 + l3); sumLanes4() writes four
    /// of them in the same order.
    static double sumLanes(V v) noexcept
    {
        const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
        return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
    }
    static void sumLanes4(double* out, V a, V b, V c, V d) noexcept
    {
        const V ac = _mm256_add_pd(_mm256_permute2f128_pd(a, c, 0x20), _mm256_permute2f128_pd(a, c, 0x31));
        const V bd = _mm256_add_pd(_mm256_permute2f128_pd(b, d, 0x20), _mm256_permute2f128_pd(b, d, 0x31));
        _mm256_storeu_pd(out, _mm256_hadd_pd(ac, bd));
    }
    /// Masks, select() and maxOf() as in Vec<float, 4>.
    using M = __m256d;
    static V abs(V v) noexcept { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), v); }
    static V div(V a, V b) noexcept { return _mm256_div_pd(a, b); }
    static V maxOf(V a, V b) noexcept { return _mm256_max_pd(b, a); }
    static V maxNum(V a, V b) noexcept { return _mm256_max_pd(a, b); }
    static M lt(V a, V b) noexcept { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
    static M le(V a, V b) noexcept { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
    static M both(M a, M b) noexcept { return _mm256_and_pd(a, b); }
    static M either(M a, M b) noexcept { return _mm256_or_pd(a, b); }
    static bool any(M m) noexcept { return _mm256_movemask_pd(m) != 0; }
    static V select(M m, V a, V b) noexcept { return _mm256_blendv_pd(b, a, m); }
    static V reverse(V v) noexcept
    {
#if defined(DSPARK_VEC_AVX2)
        return _mm256_permute4x64_pd(v, _MM_SHUFFLE(0, 1, 2, 3));
#else
        const V swapped = _mm256_permute2f128_pd(v, v, 1);
        return _mm256_permute_pd(swapped, 0x5);
#endif
    }
    /// transpose() as in Vec<float, 4>.
    static void transpose(V* v) noexcept
    {
        const V a = _mm256_unpacklo_pd(v[0], v[1]), b = _mm256_unpackhi_pd(v[0], v[1]);
        const V c = _mm256_unpacklo_pd(v[2], v[3]), d = _mm256_unpackhi_pd(v[2], v[3]);
        v[0] = _mm256_permute2f128_pd(a, c, 0x20);
        v[1] = _mm256_permute2f128_pd(b, d, 0x20);
        v[2] = _mm256_permute2f128_pd(a, c, 0x31);
        v[3] = _mm256_permute2f128_pd(b, d, 0x31);
    }
    /// transposeHalves(), loadHalves() and storeHalves() as in
    /// Vec<float, 8>, with groups of two vectors and halves of two values.
    static void transposeHalves(V* v) noexcept
    {
        const V a = _mm256_unpacklo_pd(v[0], v[1]), b = _mm256_unpackhi_pd(v[0], v[1]);
        const V c = _mm256_unpacklo_pd(v[2], v[3]), d = _mm256_unpackhi_pd(v[2], v[3]);
        v[0] = a;
        v[1] = b;
        v[2] = c;
        v[3] = d;
    }
    static V loadHalves(const double* low, const double* high) noexcept
    {
        return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(low)), _mm_loadu_pd(high), 1);
    }
    static void storeHalves(double* low, double* high, V v) noexcept
    {
        _mm_storeu_pd(low, _mm256_castpd256_pd128(v));
        _mm_storeu_pd(high, _mm256_extractf128_pd(v, 1));
    }
    /// loadDeinterleaveInLane() of four pairs taken backwards: lane l holds
    /// pair 3 - b, b the pair that lane holds in the forward order (0, 2,
    /// 1, 3).
    static void loadDeinterleaveReversedInLane(const double* p, V& re, V& im) noexcept
    {
        const V x = loadHalves(p + 6, p + 4), y = loadHalves(p + 2, p);
        re = _mm256_unpacklo_pd(x, y);
        im = _mm256_unpackhi_pd(x, y);
    }
    /// storeInterleave2InLane() with the pairs written backwards.
    static void storeInterleave2ReversedInLane(double* p, V re, V im) noexcept
    {
        storeHalves(p + 2, p, _mm256_unpackhi_pd(re, im));
        storeHalves(p + 6, p + 4, _mm256_unpacklo_pd(re, im));
    }
    /// loadDeinterleave() without crossing the 128-bit halves: the lanes come
    /// out in the order (0, 2, 1, 3).
    static void loadDeinterleaveInLane(const double* p, V& re, V& im) noexcept
    {
        const V a = _mm256_loadu_pd(p), b = _mm256_loadu_pd(p + 4);
        re = _mm256_unpacklo_pd(a, b);
        im = _mm256_unpackhi_pd(a, b);
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
    /// storeInterleave2() of lanes held in the order (0, 2, 1, 3): no shuffle
    /// crosses the 128-bit halves.
    static void storeInterleave2InLane(double* p, V re, V im) noexcept
    {
        _mm256_storeu_pd(p,     _mm256_unpacklo_pd(re, im));
        _mm256_storeu_pd(p + 4, _mm256_unpackhi_pd(re, im));
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

/// Outputs products of dotProductStrided() (see SimdOps.h) at this copy's
/// widest double vector: one accumulator each, so a result is the same
/// multiply-add chain whatever the number of products in the call. The
/// accumulators are named, not an array, so they stay in registers, and the
/// block is inlined into dotProductStridedKernel().
template <int Outputs>
DSPARK_VEC_INLINE void dotProductStridedBlock(const double* shared, const double* first, std::ptrdiff_t stride,
                                              int count, double* out) noexcept
{
    static_assert(Outputs >= 1 && Outputs <= 8, "SimdOps: one to eight products per block");
    constexpr int W = kVecWidth<double>;
    using O = Vec<double, W>;
    using V = typename O::V;
    V a0 = O::set1(0.0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const int vectorEnd = count - count % W;
    int i = 0;
    for (; i < vectorEnd; i += W)
    {
        const V s = O::load(shared + i);
        const double* const q = first + i;
        a0 = O::madd(s, O::load(q), a0);
        if constexpr (Outputs > 1) a1 = O::madd(s, O::load(q + stride), a1);
        if constexpr (Outputs > 2) a2 = O::madd(s, O::load(q + 2 * stride), a2);
        if constexpr (Outputs > 3) a3 = O::madd(s, O::load(q + 3 * stride), a3);
        if constexpr (Outputs > 4) a4 = O::madd(s, O::load(q + 4 * stride), a4);
        if constexpr (Outputs > 5) a5 = O::madd(s, O::load(q + 5 * stride), a5);
        if constexpr (Outputs > 6) a6 = O::madd(s, O::load(q + 6 * stride), a6);
        if constexpr (Outputs > 7) a7 = O::madd(s, O::load(q + 7 * stride), a7);
    }
    if constexpr (Outputs >= 4)
        O::sumLanes4(out, a0, a1, a2, a3);
    else
    {
        out[0] = O::sumLanes(a0);
        if constexpr (Outputs > 1) out[1] = O::sumLanes(a1);
        if constexpr (Outputs > 2) out[2] = O::sumLanes(a2);
    }
    if constexpr (Outputs == 8)
        O::sumLanes4(out + 4, a4, a5, a6, a7);
    else
    {
        if constexpr (Outputs > 4) out[4] = O::sumLanes(a4);
        if constexpr (Outputs > 5) out[5] = O::sumLanes(a5);
        if constexpr (Outputs > 6) out[6] = O::sumLanes(a6);
    }
    // The last count % W terms one at a time, after the lanes.
    for (int j = i; j < count; ++j)
        for (int x = 0; x < Outputs; ++x)
            out[x] += shared[j] * first[x * stride + j];
}

/// dotProductStrided() on this copy (see SimdOps.h): the products in blocks
/// of at most eight, of nearly equal sizes (a block of one or two would wait
/// on its chain).
inline void dotProductStridedKernel(const double* shared, const double* first, std::ptrdiff_t stride, int count,
                                    int outputs, double* out) noexcept
{
    while (outputs > 0)
    {
        const int blocks = (outputs + 7) / 8;
        const int n = (outputs + blocks - 1) / blocks;
        switch (n)
        {
            case 8: dotProductStridedBlock<8>(shared, first, stride, count, out); break;
            case 7: dotProductStridedBlock<7>(shared, first, stride, count, out); break;
            case 6: dotProductStridedBlock<6>(shared, first, stride, count, out); break;
            case 5: dotProductStridedBlock<5>(shared, first, stride, count, out); break;
            case 4: dotProductStridedBlock<4>(shared, first, stride, count, out); break;
            case 3: dotProductStridedBlock<3>(shared, first, stride, count, out); break;
            case 2: dotProductStridedBlock<2>(shared, first, stride, count, out); break;
            default: dotProductStridedBlock<1>(shared, first, stride, count, out); break;
        }
        outputs -= n;
        first += n * stride;
        out += n;
    }
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

/// dotProductWide() of one operand with four others, each bit for bit as
/// on its own: every product keeps its two accumulators and its reduction,
/// and the shared operand is loaded once for all four (see dotProduct4() in
/// SimdOps.h).
inline void dotProduct4Wide(const double* shared, const double* const* other, int count, double* out) noexcept
{
    __m256d s0[4], s1[4];
    for (int x = 0; x < 4; ++x) s0[x] = s1[x] = _mm256_setzero_pd();
    const double* const b0 = other[0];
    const double* const b1 = other[1];
    const double* const b2 = other[2];
    const double* const b3 = other[3];
    int i = 0;
    for (; i + 7 < count; i += 8)
    {
        const __m256d u = _mm256_loadu_pd(shared + i), v = _mm256_loadu_pd(shared + i + 4);
    #if defined(DSPARK_VEC_FMA)
        s0[0] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b0 + i), s0[0]);
        s1[0] = _mm256_fmadd_pd(v, _mm256_loadu_pd(b0 + i + 4), s1[0]);
        s0[1] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b1 + i), s0[1]);
        s1[1] = _mm256_fmadd_pd(v, _mm256_loadu_pd(b1 + i + 4), s1[1]);
        s0[2] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b2 + i), s0[2]);
        s1[2] = _mm256_fmadd_pd(v, _mm256_loadu_pd(b2 + i + 4), s1[2]);
        s0[3] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b3 + i), s0[3]);
        s1[3] = _mm256_fmadd_pd(v, _mm256_loadu_pd(b3 + i + 4), s1[3]);
    #else
        s0[0] = _mm256_add_pd(s0[0], _mm256_mul_pd(u, _mm256_loadu_pd(b0 + i)));
        s1[0] = _mm256_add_pd(s1[0], _mm256_mul_pd(v, _mm256_loadu_pd(b0 + i + 4)));
        s0[1] = _mm256_add_pd(s0[1], _mm256_mul_pd(u, _mm256_loadu_pd(b1 + i)));
        s1[1] = _mm256_add_pd(s1[1], _mm256_mul_pd(v, _mm256_loadu_pd(b1 + i + 4)));
        s0[2] = _mm256_add_pd(s0[2], _mm256_mul_pd(u, _mm256_loadu_pd(b2 + i)));
        s1[2] = _mm256_add_pd(s1[2], _mm256_mul_pd(v, _mm256_loadu_pd(b2 + i + 4)));
        s0[3] = _mm256_add_pd(s0[3], _mm256_mul_pd(u, _mm256_loadu_pd(b3 + i)));
        s1[3] = _mm256_add_pd(s1[3], _mm256_mul_pd(v, _mm256_loadu_pd(b3 + i + 4)));
    #endif
    }
    for (; i + 3 < count; i += 4)
    {
        const __m256d u = _mm256_loadu_pd(shared + i);
    #if defined(DSPARK_VEC_FMA)
        s0[0] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b0 + i), s0[0]);
        s0[1] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b1 + i), s0[1]);
        s0[2] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b2 + i), s0[2]);
        s0[3] = _mm256_fmadd_pd(u, _mm256_loadu_pd(b3 + i), s0[3]);
    #else
        s0[0] = _mm256_add_pd(s0[0], _mm256_mul_pd(u, _mm256_loadu_pd(b0 + i)));
        s0[1] = _mm256_add_pd(s0[1], _mm256_mul_pd(u, _mm256_loadu_pd(b1 + i)));
        s0[2] = _mm256_add_pd(s0[2], _mm256_mul_pd(u, _mm256_loadu_pd(b2 + i)));
        s0[3] = _mm256_add_pd(s0[3], _mm256_mul_pd(u, _mm256_loadu_pd(b3 + i)));
    #endif
    }
    const __m256d v0 = _mm256_add_pd(s0[0], s1[0]), v1 = _mm256_add_pd(s0[1], s1[1]);
    const __m256d v2 = _mm256_add_pd(s0[2], s1[2]), v3 = _mm256_add_pd(s0[3], s1[3]);
    // Low half plus high half of each (outputs 0 and 2, then 1 and 3), then
    // the two lanes left: (x0 + x2) + (x1 + x3), as dotProductWide().
    const __m256d ac = _mm256_add_pd(_mm256_permute2f128_pd(v0, v2, 0x20), _mm256_permute2f128_pd(v0, v2, 0x31));
    const __m256d bd = _mm256_add_pd(_mm256_permute2f128_pd(v1, v3, 0x20), _mm256_permute2f128_pd(v1, v3, 0x31));
    const __m256d r = _mm256_hadd_pd(ac, bd);
    if (i == count)
    {
        _mm256_storeu_pd(out, r);
        return;
    }
    // The last count % 4 terms one at a time, as dotProductWide() adds them.
    const __m128d lo = _mm256_castpd256_pd128(r), hi = _mm256_extractf128_pd(r, 1);
    double sum[4] = { _mm_cvtsd_f64(lo), _mm_cvtsd_f64(_mm_unpackhi_pd(lo, lo)),
                      _mm_cvtsd_f64(hi), _mm_cvtsd_f64(_mm_unpackhi_pd(hi, hi)) };
    for (int x = 0; x < 4; ++x)
    {
        for (int j = i; j < count; ++j) sum[x] += shared[j] * other[x][j];
        out[x] = sum[x];
    }
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
