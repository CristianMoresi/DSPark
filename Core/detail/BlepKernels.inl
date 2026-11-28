// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// BlepKernels.inl - the block kernels of MinBlepTable and Oscillator, written
// once and included once per instruction set into its own namespace (no
// include guard on purpose). MinBlepTable.h includes it into
// detail::blep_narrow (SSE2, NEON or scalar) and, in x86-64 builds, again
// into detail::blep_avx2 inside SimdOps.h's exact AVX2 region (AVX2 without
// FMA, so nothing is contracted). The includer defines the copy's features:
// DSPARK_BLEP_SSE2, DSPARK_BLEP_AVX2 (with DSPARK_BLEP_SSE2) or
// DSPARK_BLEP_NEON; none of them leaves the scalar loops.
//
// Every kernel performs, lane by lane, the operations of its scalar loop in
// the same order, so the vector and the scalar forms agree bit for bit. The
// ramp kernels step their lanes by addition: their callers pass values on a
// grid where every sum involved is exact (see Oscillator), so a stepped lane
// holds exactly v0 + j * dv. This file includes nothing.

/// r[j] += scale * (a[j] + w * (b[j] - a[j])) for j < n.
inline void blepAddRows(const float* a, const float* b, float* r, float scale, float w, int n) noexcept
{
    int j = 0;
#if defined(DSPARK_BLEP_AVX2)
    {
        const __m256 s8 = _mm256_set1_ps(scale), w8 = _mm256_set1_ps(w);
        for (; j + 8 <= n; j += 8)
        {
            const __m256 ra = _mm256_loadu_ps(a + j);
            const __m256 v = _mm256_add_ps(ra, _mm256_mul_ps(w8, _mm256_sub_ps(_mm256_loadu_ps(b + j), ra)));
            _mm256_storeu_ps(r + j, _mm256_add_ps(_mm256_loadu_ps(r + j), _mm256_mul_ps(s8, v)));
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    {
        const __m128 s4 = _mm_set1_ps(scale), w4 = _mm_set1_ps(w);
        for (; j + 4 <= n; j += 4)
        {
            const __m128 ra = _mm_loadu_ps(a + j);
            const __m128 v = _mm_add_ps(ra, _mm_mul_ps(w4, _mm_sub_ps(_mm_loadu_ps(b + j), ra)));
            _mm_storeu_ps(r + j, _mm_add_ps(_mm_loadu_ps(r + j), _mm_mul_ps(s4, v)));
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    {
        const float32x4_t s4 = vdupq_n_f32(scale), w4 = vdupq_n_f32(w);
        for (; j + 4 <= n; j += 4)
        {
            const float32x4_t ra = vld1q_f32(a + j);
            const float32x4_t v = vaddq_f32(ra, vmulq_f32(w4, vsubq_f32(vld1q_f32(b + j), ra)));
            vst1q_f32(r + j, vaddq_f32(vld1q_f32(r + j), vmulq_f32(s4, v)));
        }
    }
#endif
    // One operation per statement, as the lanes above perform them: a
    // compiler that fuses a product and a sum inside one expression must not
    // round the remainder differently from the vectors.
    for (; j < n; ++j)
    {
        const float ra = a[j];
        const float slope = w * (b[j] - ra);
        const float value = ra + slope;
        const float scaled = scale * value;
        r[j] += scaled;
    }
}

/// The same into a double ring (the rows stay float).
inline void blepAddRows(const float* a, const float* b, double* r, double scale, double w, int n) noexcept
{
    int j = 0;
#if defined(DSPARK_BLEP_AVX2)
    {
        const __m256d s4 = _mm256_set1_pd(scale), w4 = _mm256_set1_pd(w);
        for (; j + 4 <= n; j += 4)
        {
            const __m256d ra = _mm256_cvtps_pd(_mm_loadu_ps(a + j));
            const __m256d rb = _mm256_cvtps_pd(_mm_loadu_ps(b + j));
            const __m256d v = _mm256_add_pd(ra, _mm256_mul_pd(w4, _mm256_sub_pd(rb, ra)));
            _mm256_storeu_pd(r + j, _mm256_add_pd(_mm256_loadu_pd(r + j), _mm256_mul_pd(s4, v)));
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    {
        const __m128d s2 = _mm_set1_pd(scale), w2 = _mm_set1_pd(w);
        for (; j + 2 <= n; j += 2)
        {
            const __m128d ra = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + j))));
            const __m128d rb = _mm_cvtps_pd(_mm_castsi128_ps(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + j))));
            const __m128d v = _mm_add_pd(ra, _mm_mul_pd(w2, _mm_sub_pd(rb, ra)));
            _mm_storeu_pd(r + j, _mm_add_pd(_mm_loadu_pd(r + j), _mm_mul_pd(s2, v)));
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    {
        const float64x2_t s2 = vdupq_n_f64(scale), w2 = vdupq_n_f64(w);
        for (; j + 2 <= n; j += 2)
        {
            const float64x2_t ra = vcvt_f64_f32(vld1_f32(a + j));
            const float64x2_t rb = vcvt_f64_f32(vld1_f32(b + j));
            const float64x2_t v = vaddq_f64(ra, vmulq_f64(w2, vsubq_f64(rb, ra)));
            vst1q_f64(r + j, vaddq_f64(vld1q_f64(r + j), vmulq_f64(s2, v)));
        }
    }
#endif
    for (; j < n; ++j)
    {
        const double ra = static_cast<double>(a[j]);
        const double slope = w * (static_cast<double>(b[j]) - ra);
        const double value = ra + slope;
        const double scaled = scale * value;
        r[j] += scaled;
    }
}

/// out[j] = (float(v0 + j * dv) + ring[j]) - off, then ring[j] = 0, for j < n.
inline void blepRampRing(float* out, float* ring, int n, double v0, double dv, float off) noexcept
{
    int j = 0;
    [[maybe_unused]] const auto at = [v0, dv](int k) noexcept { return v0 + static_cast<double>(k) * dv; };
#if defined(DSPARK_BLEP_AVX2)
    if (n >= 8)
    {
        __m256d lo = _mm256_set_pd(at(3), at(2), at(1), at(0));
        __m256d hi = _mm256_set_pd(at(7), at(6), at(5), at(4));
        const __m256d step = _mm256_set1_pd(8.0 * dv);
        const __m256 o = _mm256_set1_ps(off), zero = _mm256_setzero_ps();
        for (; j + 8 <= n; j += 8)
        {
            const __m256 v = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(lo)), _mm256_cvtpd_ps(hi), 1);
            _mm256_storeu_ps(out + j, _mm256_sub_ps(_mm256_add_ps(v, _mm256_loadu_ps(ring + j)), o));
            _mm256_storeu_ps(ring + j, zero);
            lo = _mm256_add_pd(lo, step);
            hi = _mm256_add_pd(hi, step);
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    if (n - j >= 4)
    {
        __m128d lo = _mm_set_pd(at(j + 1), at(j));
        __m128d hi = _mm_set_pd(at(j + 3), at(j + 2));
        const __m128d step = _mm_set1_pd(4.0 * dv);
        const __m128 o = _mm_set1_ps(off), zero = _mm_setzero_ps();
        for (; j + 4 <= n; j += 4)
        {
            const __m128 v = _mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi));
            _mm_storeu_ps(out + j, _mm_sub_ps(_mm_add_ps(v, _mm_loadu_ps(ring + j)), o));
            _mm_storeu_ps(ring + j, zero);
            lo = _mm_add_pd(lo, step);
            hi = _mm_add_pd(hi, step);
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    if (n >= 4)
    {
        const double first[4] = { at(0), at(1), at(2), at(3) };
        float64x2_t lo = vld1q_f64(first), hi = vld1q_f64(first + 2);
        const float64x2_t step = vdupq_n_f64(4.0 * dv);
        const float32x4_t o = vdupq_n_f32(off), zero = vdupq_n_f32(0.0f);
        for (; j + 4 <= n; j += 4)
        {
            const float32x4_t v = vcombine_f32(vcvt_f32_f64(lo), vcvt_f32_f64(hi));
            vst1q_f32(out + j, vsubq_f32(vaddq_f32(v, vld1q_f32(ring + j)), o));
            vst1q_f32(ring + j, zero);
            lo = vaddq_f64(lo, step);
            hi = vaddq_f64(hi, step);
        }
    }
#endif
    for (; j < n; ++j)
    {
        out[j] = (static_cast<float>(v0 + static_cast<double>(j) * dv) + ring[j]) - off;
        ring[j] = 0.0f;
    }
}

/// The double form: out[j] = ((v0 + j * dv) + ring[j]) - off, then ring[j] = 0.
inline void blepRampRing(double* out, double* ring, int n, double v0, double dv, double off) noexcept
{
    int j = 0;
    [[maybe_unused]] const auto at = [v0, dv](int k) noexcept { return v0 + static_cast<double>(k) * dv; };
#if defined(DSPARK_BLEP_AVX2)
    if (n >= 4)
    {
        __m256d v = _mm256_set_pd(at(3), at(2), at(1), at(0));
        const __m256d step = _mm256_set1_pd(4.0 * dv), o = _mm256_set1_pd(off), zero = _mm256_setzero_pd();
        for (; j + 4 <= n; j += 4)
        {
            _mm256_storeu_pd(out + j, _mm256_sub_pd(_mm256_add_pd(v, _mm256_loadu_pd(ring + j)), o));
            _mm256_storeu_pd(ring + j, zero);
            v = _mm256_add_pd(v, step);
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    if (n - j >= 2)
    {
        __m128d v = _mm_set_pd(at(j + 1), at(j));
        const __m128d step = _mm_set1_pd(2.0 * dv), o = _mm_set1_pd(off), zero = _mm_setzero_pd();
        for (; j + 2 <= n; j += 2)
        {
            _mm_storeu_pd(out + j, _mm_sub_pd(_mm_add_pd(v, _mm_loadu_pd(ring + j)), o));
            _mm_storeu_pd(ring + j, zero);
            v = _mm_add_pd(v, step);
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    if (n >= 2)
    {
        const double first[2] = { at(0), at(1) };
        float64x2_t v = vld1q_f64(first);
        const float64x2_t step = vdupq_n_f64(2.0 * dv), o = vdupq_n_f64(off), zero = vdupq_n_f64(0.0);
        for (; j + 2 <= n; j += 2)
        {
            vst1q_f64(out + j, vsubq_f64(vaddq_f64(v, vld1q_f64(ring + j)), o));
            vst1q_f64(ring + j, zero);
            v = vaddq_f64(v, step);
        }
    }
#endif
    for (; j < n; ++j)
    {
        out[j] = ((v0 + static_cast<double>(j) * dv) + ring[j]) - off;
        ring[j] = 0.0;
    }
}

/// out[j] = float(v0 + j * dv) for j < n.
inline void blepRamp(float* out, int n, double v0, double dv) noexcept
{
    int j = 0;
    [[maybe_unused]] const auto at = [v0, dv](int k) noexcept { return v0 + static_cast<double>(k) * dv; };
#if defined(DSPARK_BLEP_AVX2)
    if (n >= 8)
    {
        __m256d lo = _mm256_set_pd(at(3), at(2), at(1), at(0));
        __m256d hi = _mm256_set_pd(at(7), at(6), at(5), at(4));
        const __m256d step = _mm256_set1_pd(8.0 * dv);
        for (; j + 8 <= n; j += 8)
        {
            _mm256_storeu_ps(out + j,
                             _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(lo)), _mm256_cvtpd_ps(hi), 1));
            lo = _mm256_add_pd(lo, step);
            hi = _mm256_add_pd(hi, step);
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    if (n - j >= 4)
    {
        __m128d lo = _mm_set_pd(at(j + 1), at(j));
        __m128d hi = _mm_set_pd(at(j + 3), at(j + 2));
        const __m128d step = _mm_set1_pd(4.0 * dv);
        for (; j + 4 <= n; j += 4)
        {
            _mm_storeu_ps(out + j, _mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi)));
            lo = _mm_add_pd(lo, step);
            hi = _mm_add_pd(hi, step);
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    if (n >= 4)
    {
        const double first[4] = { at(0), at(1), at(2), at(3) };
        float64x2_t lo = vld1q_f64(first), hi = vld1q_f64(first + 2);
        const float64x2_t step = vdupq_n_f64(4.0 * dv);
        for (; j + 4 <= n; j += 4)
        {
            vst1q_f32(out + j, vcombine_f32(vcvt_f32_f64(lo), vcvt_f32_f64(hi)));
            lo = vaddq_f64(lo, step);
            hi = vaddq_f64(hi, step);
        }
    }
#endif
    for (; j < n; ++j)
        out[j] = static_cast<float>(v0 + static_cast<double>(j) * dv);
}

/// The double form: out[j] = v0 + j * dv.
inline void blepRamp(double* out, int n, double v0, double dv) noexcept
{
    for (int j = 0; j < n; ++j)
        out[j] = v0 + static_cast<double>(j) * dv;
}

/// out[j] = float(v0 + j * dv) - off for j < n.
inline void blepRampOffset(float* out, int n, double v0, double dv, float off) noexcept
{
    int j = 0;
    [[maybe_unused]] const auto at = [v0, dv](int k) noexcept { return v0 + static_cast<double>(k) * dv; };
#if defined(DSPARK_BLEP_AVX2)
    if (n >= 8)
    {
        __m256d lo = _mm256_set_pd(at(3), at(2), at(1), at(0));
        __m256d hi = _mm256_set_pd(at(7), at(6), at(5), at(4));
        const __m256d step = _mm256_set1_pd(8.0 * dv);
        const __m256 o = _mm256_set1_ps(off);
        for (; j + 8 <= n; j += 8)
        {
            const __m256 v = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(lo)), _mm256_cvtpd_ps(hi), 1);
            _mm256_storeu_ps(out + j, _mm256_sub_ps(v, o));
            lo = _mm256_add_pd(lo, step);
            hi = _mm256_add_pd(hi, step);
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    if (n - j >= 4)
    {
        __m128d lo = _mm_set_pd(at(j + 1), at(j));
        __m128d hi = _mm_set_pd(at(j + 3), at(j + 2));
        const __m128d step = _mm_set1_pd(4.0 * dv);
        const __m128 o = _mm_set1_ps(off);
        for (; j + 4 <= n; j += 4)
        {
            _mm_storeu_ps(out + j, _mm_sub_ps(_mm_movelh_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi)), o));
            lo = _mm_add_pd(lo, step);
            hi = _mm_add_pd(hi, step);
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    if (n >= 4)
    {
        const double first[4] = { at(0), at(1), at(2), at(3) };
        float64x2_t lo = vld1q_f64(first), hi = vld1q_f64(first + 2);
        const float64x2_t step = vdupq_n_f64(4.0 * dv);
        const float32x4_t o = vdupq_n_f32(off);
        for (; j + 4 <= n; j += 4)
        {
            vst1q_f32(out + j, vsubq_f32(vcombine_f32(vcvt_f32_f64(lo), vcvt_f32_f64(hi)), o));
            lo = vaddq_f64(lo, step);
            hi = vaddq_f64(hi, step);
        }
    }
#endif
    for (; j < n; ++j)
        out[j] = static_cast<float>(v0 + static_cast<double>(j) * dv) - off;
}

/// The double form: out[j] = (v0 + j * dv) - off.
inline void blepRampOffset(double* out, int n, double v0, double dv, double off) noexcept
{
    for (int j = 0; j < n; ++j)
        out[j] = (v0 + static_cast<double>(j) * dv) - off;
}

/// out[j] = (level + ring[j]) - dc, then ring[j] = 0, for j < n.
inline void blepLevelRing(float* out, float* ring, int n, float level, float dc) noexcept
{
    int j = 0;
#if defined(DSPARK_BLEP_AVX2)
    {
        const __m256 l = _mm256_set1_ps(level), d = _mm256_set1_ps(dc), zero = _mm256_setzero_ps();
        for (; j + 8 <= n; j += 8)
        {
            _mm256_storeu_ps(out + j, _mm256_sub_ps(_mm256_add_ps(l, _mm256_loadu_ps(ring + j)), d));
            _mm256_storeu_ps(ring + j, zero);
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    {
        const __m128 l = _mm_set1_ps(level), d = _mm_set1_ps(dc), zero = _mm_setzero_ps();
        for (; j + 4 <= n; j += 4)
        {
            _mm_storeu_ps(out + j, _mm_sub_ps(_mm_add_ps(l, _mm_loadu_ps(ring + j)), d));
            _mm_storeu_ps(ring + j, zero);
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    {
        const float32x4_t l = vdupq_n_f32(level), d = vdupq_n_f32(dc), zero = vdupq_n_f32(0.0f);
        for (; j + 4 <= n; j += 4)
        {
            vst1q_f32(out + j, vsubq_f32(vaddq_f32(l, vld1q_f32(ring + j)), d));
            vst1q_f32(ring + j, zero);
        }
    }
#endif
    for (; j < n; ++j)
    {
        out[j] = (level + ring[j]) - dc;
        ring[j] = 0.0f;
    }
}

/// The double form.
inline void blepLevelRing(double* out, double* ring, int n, double level, double dc) noexcept
{
    int j = 0;
#if defined(DSPARK_BLEP_AVX2)
    {
        const __m256d l = _mm256_set1_pd(level), d = _mm256_set1_pd(dc), zero = _mm256_setzero_pd();
        for (; j + 4 <= n; j += 4)
        {
            _mm256_storeu_pd(out + j, _mm256_sub_pd(_mm256_add_pd(l, _mm256_loadu_pd(ring + j)), d));
            _mm256_storeu_pd(ring + j, zero);
        }
    }
#endif
#if defined(DSPARK_BLEP_SSE2)
    {
        const __m128d l = _mm_set1_pd(level), d = _mm_set1_pd(dc), zero = _mm_setzero_pd();
        for (; j + 2 <= n; j += 2)
        {
            _mm_storeu_pd(out + j, _mm_sub_pd(_mm_add_pd(l, _mm_loadu_pd(ring + j)), d));
            _mm_storeu_pd(ring + j, zero);
        }
    }
#elif defined(DSPARK_BLEP_NEON)
    {
        const float64x2_t l = vdupq_n_f64(level), d = vdupq_n_f64(dc), zero = vdupq_n_f64(0.0);
        for (; j + 2 <= n; j += 2)
        {
            vst1q_f64(out + j, vsubq_f64(vaddq_f64(l, vld1q_f64(ring + j)), d));
            vst1q_f64(ring + j, zero);
        }
    }
#endif
    for (; j < n; ++j)
    {
        out[j] = (level + ring[j]) - dc;
        ring[j] = 0.0;
    }
}
