// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// FftKernels.inl - the FFT passes written on the vector abstraction, included
// once per instruction set into its own namespace (no include guard on
// purpose). FFT.h includes it into dspark::detail::fft for the baseline and,
// in x86-64 builds below AVX, again into dspark::detail::fft::avx2 inside an
// AVX2+FMA target region. The includer first declares Vec, kWide and kNarrow
// for the copy; TwiddleSpan, Pass, Buffers and kBlock come from the enclosing
// namespace. This file includes nothing.
//
// Work layout: between the first and the closing pass the data is blocked,
// kBlock<T> real parts followed by the same elements' imaginary parts (see
// blockedOffset()). Every vector width a copy uses divides the block, so a
// vector is one contiguous load in either part, and the imaginary part of
// anything sits at a fixed distance from its real part: a pass keeps one
// pointer per stream instead of two. The block is the same in every copy, so
// the steps of one transform may run on different copies.

/// Offset of element i's real part in the blocked work layout; its imaginary
/// part is kBlock<T> further.
template <typename T>
inline size_t blockedOffset(size_t i) noexcept
{
    constexpr size_t B = kBlock<T>;
    return 2 * i - (i & (B - 1));
}

/// Where the first pass reads its input from, and where the last pass writes.
/// Reading interleaved data directly in the first pass, and writing it directly
/// in the last, saves one full pass over memory in each direction.
template <typename T, int W>
struct SplitSource
{
    using O = Vec<T, W>;
    const T* re;
    const T* im;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        r = O::load(re + i);
        m = O::load(im + i);
    }
};

/// The blocked work layout as a source (the closing pass after the others).
template <typename T, int W>
struct BlockedSource
{
    using O = Vec<T, W>;
    const T* base;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        const T* p = base + blockedOffset<T>(i);
        r = O::load(p);
        m = O::load(p + kBlock<T>);
    }
};

template <typename T, int W>
struct InterleavedSource
{
    using O = Vec<T, W>;
    const T* data;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        O::loadDeinterleave(data + 2 * i, r, m);
    }
};

template <typename T, int W>
struct ConjugatedSource
{
    using O = Vec<T, W>;
    const T* data;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        O::loadDeinterleave(data + 2 * i, r, m);
        m = O::sub(O::set1(T(0)), m);
    }
};

/// The product of a spectrum with a real gain, its parts swapped: bin i reads
/// (im[i & mask] gain[i], re[i & mask] gain[i]). Transforming a product's
/// swapped parts forward is its inverse transform with the parts swapped; the
/// mask repeats a spectrum of mask + 1 bins over the transform.
template <typename T, int W>
struct ScaledSwapSource
{
    using O = Vec<T, W>;
    const T* re;
    const T* im;
    const T* gain;
    size_t mask;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        const typename O::V g = O::load(gain + i);
        r = O::mul(O::load(im + (i & mask)), g);
        m = O::mul(O::load(re + (i & mask)), g);
    }
};

/// ScaledSwapSource folded: bin i sums the products of bins i + f * stride
/// for f < fold. A forward transform of that is the inverse of the product,
/// read at every fold-th sample (the spectrum must hold nothing that the fold
/// would alias).
template <typename T, int W>
struct FoldedScaledSwapSource
{
    using O = Vec<T, W>;
    const T* re;
    const T* im;
    const T* gain;
    size_t stride;
    size_t fold;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        typename O::V g = O::load(gain + i);
        r = O::mul(O::load(im + i), g);
        m = O::mul(O::load(re + i), g);
        for (size_t f = 1; f < fold; ++f)
        {
            const size_t j = i + f * stride;
            g = O::load(gain + j);
            r = O::add(r, O::mul(O::load(im + j), g));
            m = O::add(m, O::mul(O::load(re + j), g));
        }
    }
};

template <typename T, int W>
struct SplitSink
{
    using O = Vec<T, W>;
    T* re;
    T* im;
    void write(size_t i, typename O::V r, typename O::V m) const noexcept
    {
        O::store(re + i, r);
        O::store(im + i, m);
    }
};

template <typename T, int W>
struct InterleavedSink
{
    using O = Vec<T, W>;
    T* data;
    typename O::V scaleRe;
    typename O::V scaleIm;
    void write(size_t i, typename O::V r, typename O::V m) const noexcept
    {
        O::storeInterleave2(data + 2 * i, O::mul(r, scaleRe), O::mul(m, scaleIm));
    }
};

/**
 * @brief One Stockham radix-4 DIF pass with stride s >= kBlock, blocked
 *        layout in and out, vectorised over the stride (twiddles broadcast
 *        per butterfly group).
 *
 * y[q + s(4p + k)] = w^(kp) * (radix-4 butterfly of x[q + s(p + k n/4)]).
 */
template <typename T, int W>
void radix4Blocked(const T* x, T* y, size_t n, size_t s, const TwiddleSpan<T>& tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    const size_t n1 = n / 4;
    // s is a multiple of B, so every row starts a block: a row of s elements
    // spans 2 s values, and one index walks all eight rows of a group.
    const size_t rowIn = 2 * s * n1, rowOut = 2 * s;
    const T* const w1rp = tw.w1r; const T* const w1ip = tw.w1i;
    const T* const w2rp = tw.w2r; const T* const w2ip = tw.w2i;
    const T* const w3rp = tw.w3r; const T* const w3ip = tw.w3i;
    for (size_t p = 0; p < n1; ++p)
    {
        const V w1r = O::set1(w1rp[p]), w1i = O::set1(w1ip[p]);
        const V w2r = O::set1(w2rp[p]), w2i = O::set1(w2ip[p]);
        const V w3r = O::set1(w3rp[p]), w3i = O::set1(w3ip[p]);
        const T* const a = x + 2 * s * p;
        const T* const b = a + rowIn;
        const T* const c = b + rowIn;
        const T* const d = c + rowIn;
        T* const y0 = y + 8 * s * p;
        T* const y1 = y0 + rowOut;
        T* const y2 = y1 + rowOut;
        T* const y3 = y2 + rowOut;
        for (size_t q = 0; q < 2 * s; q += 2 * B)
        {
            for (size_t h = q; h < q + B; h += W)
            {
                const V a_r = O::load(a + h), a_i = O::load(a + B + h);
                const V b_r = O::load(b + h), b_i = O::load(b + B + h);
                const V c_r = O::load(c + h), c_i = O::load(c + B + h);
                const V d_r = O::load(d + h), d_i = O::load(d + B + h);
                const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
                const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
                const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
                const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
                // t1 = amc - j bmd, t2 = apc - bpd, t3 = amc + j bmd
                const V t1R = O::add(amcR, bmdI), t1I = O::sub(amcI, bmdR);
                const V t2R = O::sub(apcR, bpdR), t2I = O::sub(apcI, bpdI);
                const V t3R = O::sub(amcR, bmdI), t3I = O::add(amcI, bmdR);
                O::store(y0 + h,     O::add(apcR, bpdR));
                O::store(y0 + B + h, O::add(apcI, bpdI));
                O::store(y1 + h,     O::mulSub(t1R, w1r, t1I, w1i));
                O::store(y1 + B + h, O::mulAdd(t1R, w1i, t1I, w1r));
                O::store(y2 + h,     O::mulSub(t2R, w2r, t2I, w2i));
                O::store(y2 + B + h, O::mulAdd(t2R, w2i, t2I, w2r));
                O::store(y3 + h,     O::mulSub(t3R, w3r, t3I, w3i));
                O::store(y3 + B + h, O::mulAdd(t3R, w3i, t3I, w3r));
            }
        }
    }
}

/**
 * @brief radix4Blocked() for a stride of half a block: two butterfly groups
 *        share each block, p even in its first half and p + 1 in its second,
 *        so the groups run in pairs at fixed offsets.
 */
template <typename T, int W>
void radix4HalfBlock(const T* x, T* y, size_t n, const TwiddleSpan<T>& tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    constexpr size_t s = B / 2;
    const size_t n1 = n / 4;   // even: the pass is at least 8 long
    const size_t i1 = 2 * s * n1, i2 = 2 * i1, i3 = 3 * i1;
    // Output k of group p is element s (4p + k), at offsets 8 s p plus
    // 0, s, 4 s and 5 s.
    constexpr size_t o1 = s, o2 = 4 * s, o3 = 5 * s;
    const T* const w1rp = tw.w1r; const T* const w1ip = tw.w1i;
    const T* const w2rp = tw.w2r; const T* const w2ip = tw.w2i;
    const T* const w3rp = tw.w3r; const T* const w3ip = tw.w3i;
    for (size_t p = 0; p < n1; ++p)
    {
        const V w1r = O::set1(w1rp[p]), w1i = O::set1(w1ip[p]);
        const V w2r = O::set1(w2rp[p]), w2i = O::set1(w2ip[p]);
        const V w3r = O::set1(w3rp[p]), w3i = O::set1(w3ip[p]);
        const T* const xp = x + 2 * s * p - s * (p & 1);
        T* const yp = y + 8 * s * p;
        for (size_t h = 0; h < s; h += W)
        {
            const V a_r = O::load(xp + h),      a_i = O::load(xp + B + h);
            const V b_r = O::load(xp + i1 + h), b_i = O::load(xp + i1 + B + h);
            const V c_r = O::load(xp + i2 + h), c_i = O::load(xp + i2 + B + h);
            const V d_r = O::load(xp + i3 + h), d_i = O::load(xp + i3 + B + h);
            const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
            const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
            const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
            const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
            const V t1R = O::add(amcR, bmdI), t1I = O::sub(amcI, bmdR);
            const V t2R = O::sub(apcR, bpdR), t2I = O::sub(apcI, bpdI);
            const V t3R = O::sub(amcR, bmdI), t3I = O::add(amcI, bmdR);
            O::store(yp + h,          O::add(apcR, bpdR));
            O::store(yp + B + h,      O::add(apcI, bpdI));
            O::store(yp + o1 + h,     O::mulSub(t1R, w1r, t1I, w1i));
            O::store(yp + o1 + B + h, O::mulAdd(t1R, w1i, t1I, w1r));
            O::store(yp + o2 + h,     O::mulSub(t2R, w2r, t2I, w2i));
            O::store(yp + o2 + B + h, O::mulAdd(t2R, w2i, t2I, w2r));
            O::store(yp + o3 + h,     O::mulSub(t3R, w3r, t3I, w3i));
            O::store(yp + o3 + B + h, O::mulAdd(t3R, w3i, t3I, w3r));
        }
    }
}

/**
 * @brief The first (stride-1) radix-4 pass, vectorised across W butterflies
 *        (n/4 must be a multiple of W): the inputs and twiddles are
 *        contiguous in p, and the four outputs of each butterfly are adjacent,
 *        written through a 4-way interleave into the blocked layout.
 */
template <typename T, int W, typename Source>
void radix4First(const Source& input, T* y, size_t n, const TwiddleSpan<T>& tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    const size_t n1 = n / 4;
    for (size_t p = 0; p < n1; p += W)
    {
        const V w1r = O::load(tw.w1r + p), w1i = O::load(tw.w1i + p);
        const V w2r = O::load(tw.w2r + p), w2i = O::load(tw.w2i + p);
        const V w3r = O::load(tw.w3r + p), w3i = O::load(tw.w3i + p);
        V a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i;
        input.read(p, a_r, a_i);
        input.read(p + n1, b_r, b_i);
        input.read(p + 2 * n1, c_r, c_i);
        input.read(p + 3 * n1, d_r, d_i);
        const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
        const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
        const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
        const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
        const V t1R = O::add(amcR, bmdI), t1I = O::sub(amcI, bmdR);
        const V t2R = O::sub(apcR, bpdR), t2I = O::sub(apcI, bpdI);
        const V t3R = O::sub(amcR, bmdI), t3I = O::add(amcI, bmdR);
        // The 4 W outputs, elements 4p .. 4p + 4W - 1, as four runs of W.
        T* const r0 = y + blockedOffset<T>(4 * p);
        T* const r1 = y + blockedOffset<T>(4 * p + W);
        T* const r2 = y + blockedOffset<T>(4 * p + 2 * W);
        T* const r3 = y + blockedOffset<T>(4 * p + 3 * W);
        O::storeInterleave4To(r0, r1, r2, r3, O::add(apcR, bpdR),
                              O::mulSub(t1R, w1r, t1I, w1i),
                              O::mulSub(t2R, w2r, t2I, w2i),
                              O::mulSub(t3R, w3r, t3I, w3i));
        O::storeInterleave4To(r0 + B, r1 + B, r2 + B, r3 + B, O::add(apcI, bpdI),
                              O::mulAdd(t1R, w1i, t1I, w1r),
                              O::mulAdd(t2R, w2i, t2I, w2r),
                              O::mulAdd(t3R, w3i, t3I, w3r));
    }
}

/// The closing radix-4 pass (n = 4, stride s = N / 4): its only twiddles are 1.
template <typename T, int W, typename Source, typename Sink>
void radix4Last(const Source& input, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    for (size_t q = 0; q < s; q += W)
    {
        V a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i;
        input.read(q, a_r, a_i);
        input.read(q + s, b_r, b_i);
        input.read(q + 2 * s, c_r, c_i);
        input.read(q + 3 * s, d_r, d_i);
        const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
        const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
        const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
        const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
        output.write(q,         O::add(apcR, bpdR), O::add(apcI, bpdI));
        output.write(q + s,     O::add(amcR, bmdI), O::sub(amcI, bmdR));
        output.write(q + 2 * s, O::sub(apcR, bpdR), O::sub(apcI, bpdI));
        output.write(q + 3 * s, O::sub(amcR, bmdI), O::add(amcI, bmdR));
    }
}

/// The closing radix-2 pass (n = 2, stride s = N / 2): no twiddles.
template <typename T, int W, typename Source, typename Sink>
void radix2Last(const Source& input, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    for (size_t q = 0; q < s; q += W)
    {
        typename O::V a_r, a_i, b_r, b_i;
        input.read(q, a_r, a_i);
        input.read(q + s, b_r, b_i);
        output.write(q, O::add(a_r, b_r), O::add(a_i, b_i));
        output.write(q + s, O::sub(a_r, b_r), O::sub(a_i, b_i));
    }
}

/// radix4Last() from the blocked layout with s >= kBlock: rows start blocks.
template <typename T, int W, typename Sink>
void radix4LastBlocked(const T* x, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    for (size_t qb = 0; qb < s; qb += B)
    {
        const T* const r0 = x + 2 * qb;
        const T* const r1 = r0 + 2 * s;
        const T* const r2 = r0 + 4 * s;
        const T* const r3 = r0 + 6 * s;
        for (size_t h = 0; h < B; h += W)
        {
            const size_t q = qb + h;
            const V a_r = O::load(r0 + h), a_i = O::load(r0 + B + h);
            const V b_r = O::load(r1 + h), b_i = O::load(r1 + B + h);
            const V c_r = O::load(r2 + h), c_i = O::load(r2 + B + h);
            const V d_r = O::load(r3 + h), d_i = O::load(r3 + B + h);
            const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
            const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
            const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
            const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
            output.write(q,         O::add(apcR, bpdR), O::add(apcI, bpdI));
            output.write(q + s,     O::add(amcR, bmdI), O::sub(amcI, bmdR));
            output.write(q + 2 * s, O::sub(apcR, bpdR), O::sub(apcI, bpdI));
            output.write(q + 3 * s, O::sub(amcR, bmdI), O::add(amcI, bmdR));
        }
    }
}

/// radix2Last() from the blocked layout with s >= kBlock.
template <typename T, int W, typename Sink>
void radix2LastBlocked(const T* x, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    for (size_t qb = 0; qb < s; qb += B)
    {
        const T* const r0 = x + 2 * qb;
        const T* const r1 = r0 + 2 * s;
        for (size_t h = 0; h < B; h += W)
        {
            const size_t q = qb + h;
            const V a_r = O::load(r0 + h), a_i = O::load(r0 + B + h);
            const V b_r = O::load(r1 + h), b_i = O::load(r1 + B + h);
            output.write(q, O::add(a_r, b_r), O::add(a_i, b_i));
            output.write(q + s, O::sub(a_r, b_r), O::sub(a_i, b_i));
        }
    }
}

/**
 * @brief The closing radix-8 pass (n = 8, stride s = N / 8 >= kBlock) from the
 *        blocked layout: it replaces a radix-4 pass of length 8 and the
 *        closing radix-2 pass that would follow it, one pass over the data
 *        fewer. Twiddle-free: the eighth roots enter as the exact sqrt(1/2).
 *
 * y[q + s k] = sum_r x[q + s r] w8^(r k), split as two 4-point transforms of
 * the sums x_r + x_(r+4) (even k) and the twiddled differences (odd k).
 */
template <typename T, int W, typename Sink>
void radix8LastBlocked(const T* x, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    const V h = O::set1(static_cast<T>(0.70710678118654752440));
    const V zero = O::set1(T(0));
    // 4-point transform of z0..z3 into rows k0, k0 + 2, k0 + 4, k0 + 6.
    const auto dft4 = [&](size_t q, size_t k0, const V* zr, const V* zi) noexcept {
        const V apcR = O::add(zr[0], zr[2]), apcI = O::add(zi[0], zi[2]);
        const V amcR = O::sub(zr[0], zr[2]), amcI = O::sub(zi[0], zi[2]);
        const V bpdR = O::add(zr[1], zr[3]), bpdI = O::add(zi[1], zi[3]);
        const V bmdR = O::sub(zr[1], zr[3]), bmdI = O::sub(zi[1], zi[3]);
        output.write(q + s * k0,       O::add(apcR, bpdR), O::add(apcI, bpdI));
        output.write(q + s * (k0 + 2), O::add(amcR, bmdI), O::sub(amcI, bmdR));
        output.write(q + s * (k0 + 4), O::sub(apcR, bpdR), O::sub(apcI, bpdI));
        output.write(q + s * (k0 + 6), O::sub(amcR, bmdI), O::add(amcI, bmdR));
    };
    for (size_t qb = 0; qb < s; qb += B)
    {
        const T* const r0 = x + 2 * qb;
        for (size_t hh = 0; hh < B; hh += W)
        {
            V ar[4], ai[4], br[4], bi[4];
            for (size_t r = 0; r < 4; ++r)
            {
                const T* const lo = r0 + 2 * s * r + hh;
                const T* const hi = lo + 8 * s;
                const V xr = O::load(lo), xi = O::load(lo + B);
                const V yr = O::load(hi), yi = O::load(hi + B);
                ar[r] = O::add(xr, yr); ai[r] = O::add(xi, yi);
                br[r] = O::sub(xr, yr); bi[r] = O::sub(xi, yi);
            }
            // Differences times w8^r: (1 - j) / sqrt 2, -j, -(1 + j) / sqrt 2.
            const V b1r = O::mul(O::add(br[1], bi[1]), h), b1i = O::mul(O::sub(bi[1], br[1]), h);
            const V b2r = bi[2], b2i = O::sub(zero, br[2]);
            const V b3r = O::mul(O::sub(bi[3], br[3]), h), b3i = O::sub(zero, O::mul(O::add(br[3], bi[3]), h));
            br[1] = b1r; bi[1] = b1i;
            br[2] = b2r; bi[2] = b2i;
            br[3] = b3r; bi[3] = b3i;
            dft4(qb + hh, 0, ar, ai);
            dft4(qb + hh, 1, br, bi);
        }
    }
}

/// The twiddles of one radix-4 pass.
template <typename T>
TwiddleSpan<T> twiddleSpanOf(const Pass& pass, const T* twiddles) noexcept
{
    const size_t n1 = pass.len / 4;
    const T* w = twiddles + pass.twOffset;
    return { w, w + n1, w + 2 * n1, w + 3 * n1, w + 4 * n1, w + 5 * n1 };
}

/// Runs the first pass at the widest vector its length allows, into the
/// blocked layout at y.
template <typename T, template <typename, int> class Source, typename... Args>
void firstPass(const Pass& pass, const T* twiddles, T* y, Args... input) noexcept
{
    const size_t n1 = pass.len / 4;
    const TwiddleSpan<T> tw = twiddleSpanOf(pass, twiddles);
    constexpr int wide = kWide<T>;
    constexpr int narrow = kNarrow<T>;
    if (n1 % wide == 0)
        radix4First<T, wide>(Source<T, wide> { input... }, y, pass.len, tw);
    else if (n1 % narrow == 0)
        radix4First<T, narrow>(Source<T, narrow> { input... }, y, pass.len, tw);
    else
        radix4First<T, 1>(Source<T, 1> { input... }, y, pass.len, tw);
}

/// Runs a radix-4 pass of stride > 1 at the widest vector its stride allows,
/// blocked layout in and out. Strides are powers of 4, so the only one below
/// the block is 4, half a float block.
template <typename T>
void stridedPass(const Pass& pass, const T* twiddles, const Buffers<T>& b) noexcept
{
    const TwiddleSpan<T> tw = twiddleSpanOf(pass, twiddles);
    constexpr int narrow = kNarrow<T>;
    if (pass.stride >= kBlock<T>)
    {
        radix4Blocked<T, kWide<T>>(b.xr, b.yr, pass.len, pass.stride, tw);
        return;
    }
    assert(2 * pass.stride == kBlock<T>);
    if (pass.stride % narrow == 0)
        radix4HalfBlock<T, narrow>(b.xr, b.yr, pass.len, tw);
    else
        radix4HalfBlock<T, 1>(b.xr, b.yr, pass.len, tw);
}

template <typename T, int W, template <typename, int> class Sink>
Sink<T, W> makeSink(T* re, T* im) noexcept
{
    return { re, im };
}

template <typename T, int W, template <typename, int> class Sink>
Sink<T, W> makeSink(T* data, T scaleRe, T scaleIm) noexcept
{
    return { data, Vec<T, W>::set1(scaleRe), Vec<T, W>::set1(scaleIm) };
}

template <typename T, int W, template <typename, int> class Sink, typename... Args>
void closingAt(const Pass& pass, const T* xr, const T* xi, bool blocked, Args... output) noexcept
{
    const Sink<T, W> sink = makeSink<T, W, Sink>(output...);
    if (pass.radix8)
        radix8LastBlocked<T, W>(xr, sink, pass.stride);
    else if (blocked && pass.stride >= kBlock<T>)
    {
        if (pass.radix2)
            radix2LastBlocked<T, W>(xr, sink, pass.stride);
        else
            radix4LastBlocked<T, W>(xr, sink, pass.stride);
    }
    else if (blocked)
    {
        const BlockedSource<T, W> in { xr };
        if (pass.radix2)
            radix2Last<T, W>(in, sink, pass.stride);
        else
            radix4Last<T, W>(in, sink, pass.stride);
    }
    else
    {
        const SplitSource<T, W> in { xr, xi };
        if (pass.radix2)
            radix2Last<T, W>(in, sink, pass.stride);
        else
            radix4Last<T, W>(in, sink, pass.stride);
    }
}

/// Runs a pass that needs no twiddles, at the widest vector its stride allows.
/// Its input is in the blocked layout after other passes, and split when it
/// is a transform's only pass.
template <typename T, template <typename, int> class Sink, typename... Args>
void closingPass(const Pass& pass, const T* xr, const T* xi, bool blocked, Args... output) noexcept
{
    constexpr int wide = kWide<T>;
    constexpr int narrow = kNarrow<T>;
    if (pass.stride % wide == 0)
        closingAt<T, wide, Sink>(pass, xr, xi, blocked, output...);
    else if (pass.stride % narrow == 0)
        closingAt<T, narrow, Sink>(pass, xr, xi, blocked, output...);
    else
        closingAt<T, 1, Sink>(pass, xr, xi, blocked, output...);
}

/// Runs passes [begin, end) between the two work buffers.
template <typename T>
void runPasses(const Pass* passes, size_t begin, size_t end, const T* twiddles,
               Buffers<T>& b) noexcept
{
    for (size_t i = begin; i < end; ++i)
    {
        const Pass& pass = passes[i];
        if (pass.radix2 || pass.radix8 || pass.len == 4)
            closingPass<T, SplitSink>(pass, b.xr, b.xi, i > 0, b.yr, b.yi);
        else if (pass.stride == 1)
            firstPass<T, SplitSource>(pass, twiddles, b.yr, b.xr, b.xi);
        else
            stridedPass(pass, twiddles, b);
        T* r = b.xr; b.xr = b.yr; b.yr = r;
        T* m = b.xi; b.xi = b.yi; b.yi = m;
    }
}

/// Reads interleaved input, conjugated if asked; returns the passes it ran.
template <typename T>
size_t readInput(const Pass* passes, size_t passCount, size_t n, const T* twiddles,
                 const T* data, bool conjugate, Buffers<T>& b) noexcept
{
    if (passCount < 2)
    {
        for (size_t i = 0; i < n; ++i)
        {
            b.xr[i] = data[2 * i];
            b.xi[i] = conjugate ? -data[2 * i + 1] : data[2 * i + 1];
        }
        return 0;
    }
    if (conjugate)
        firstPass<T, ConjugatedSource>(passes[0], twiddles, b.yr, data);
    else
        firstPass<T, InterleavedSource>(passes[0], twiddles, b.yr, data);
    T* r = b.xr; b.xr = b.yr; b.yr = r;
    T* m = b.xi; b.xi = b.yi; b.yi = m;
    return 1;
}

/// Reads split input from outside the work buffers; returns the passes it ran.
template <typename T>
size_t readSplit(const Pass* passes, size_t passCount, size_t n, const T* twiddles,
                 const T* re, const T* im, Buffers<T>& b) noexcept
{
    if (passCount < 2)
    {
        for (size_t i = 0; i < n; ++i)
        {
            b.xr[i] = re[i];
            b.xi[i] = im[i];
        }
        return 0;
    }
    firstPass<T, SplitSource>(passes[0], twiddles, b.yr, re, im);
    T* r = b.xr; b.xr = b.yr; b.yr = r;
    T* m = b.xi; b.xi = b.yi; b.yi = m;
    return 1;
}

/// Reads a scaled, swapped spectrum product (ScaledSwapSource); returns the
/// passes it ran.
template <typename T>
size_t readScaledSwap(const Pass* passes, size_t passCount, size_t n, const T* twiddles,
                      const T* re, const T* im, const T* gain, size_t mask, Buffers<T>& b) noexcept
{
    if (passCount < 2)
    {
        for (size_t i = 0; i < n; ++i)
        {
            const T g = gain[i];
            const T a = im[i & mask] * g, c = re[i & mask] * g;
            b.xr[i] = a;
            b.xi[i] = c;
        }
        return 0;
    }
    firstPass<T, ScaledSwapSource>(passes[0], twiddles, b.yr, re, im, gain, mask);
    T* r = b.xr; b.xr = b.yr; b.yr = r;
    T* m = b.xi; b.xi = b.yi; b.yi = m;
    return 1;
}

/// Reads a folded, scaled, swapped spectrum product (FoldedScaledSwapSource);
/// returns the passes it ran.
template <typename T>
size_t readScaledSwapFolded(const Pass* passes, size_t passCount, size_t n, const T* twiddles,
                            const T* re, const T* im, const T* gain, size_t fold, Buffers<T>& b) noexcept
{
    if (passCount < 2)
    {
        for (size_t i = 0; i < n; ++i)
        {
            T a = im[i] * gain[i], c = re[i] * gain[i];
            for (size_t f = 1; f < fold; ++f)
            {
                const size_t j = i + f * n;
                const T pa = im[j] * gain[j], pc = re[j] * gain[j];
                a = a + pa;
                c = c + pc;
            }
            b.xr[i] = a;
            b.xi[i] = c;
        }
        return 0;
    }
    firstPass<T, FoldedScaledSwapSource>(passes[0], twiddles, b.yr, re, im, gain, n, fold);
    T* r = b.xr; b.xr = b.yr; b.yr = r;
    T* m = b.xi; b.xi = b.yi; b.yi = m;
    return 1;
}

/**
 * @brief Packs the spectrum of a real signal of 2M samples (bins 0 .. M,
 *        interleaved) into the conjugated half-size spectrum conj(Z) the
 *        inverse transforms, W bins at a time with their mirror images:
 *        Z[k] = Xe[k] + j conj(w^k) Xd[k], with Xe and Xd the even and odd
 *        parts (X[k] +- conj(X[M-k])) / 2.
 *
 * The 1/2 of Xe and Xd is left out (the inverse's scale takes it: a power
 * of two, so the result is the same to the bit). Bins k and M - k share their
 * sums, differences and twiddle product (w^(M-k) = -conj(w^k) exactly, see
 * FFTReal), so one pass over k <= M/2 writes both. The imaginary parts of
 * bins 0 and M are ignored, as a real signal has none.
 */
template <typename T, int W>
void packRealSpectrumAt(const T* in, T* zr, T* zi, size_t m, const T* twRe, const T* twIm) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const T dc = in[0];
    const T ny = in[2 * m];
    zr[0] = dc + ny;
    zi[0] = -(dc - ny);
    const V zero = O::set1(T(0));
    for (size_t k = 1; k <= m / 2; k += W)   // k .. k+W-1 against M-k .. M-k-W+1
    {
        const size_t c = m - k - (W - 1);
        V xkR, xkI, xcR, xcI;
        O::loadDeinterleave(in + 2 * k, xkR, xkI);
        O::loadDeinterleaveReversed(in + 2 * c, xcR, xcI);
        const V eR = O::add(xkR, xcR), eI = O::sub(xkI, xcI);
        const V dR = O::sub(xkR, xcR), dI = O::add(xkI, xcI);
        // conj(w) * d, then j * that: (-(tI), tR)
        const V wr = O::load(twRe + k), wi = O::load(twIm + k);
        const V tR = O::mulAdd(wr, dR, wi, dI);    // wr*dR - (-wi)*dI
        const V tI = O::mulSub(wr, dI, wi, dR);    // wr*dI + (-wi)*dR
        O::store(zr + k, O::sub(eR, tI));
        O::store(zi + k, O::sub(zero, O::add(eI, tR)));   // conjugated
        // Bin M - k: its e and d are (eR, -eI) and (-dR, dI), its twiddle
        // product (tR, -tI).
        O::store(zr + c, O::reverse(O::add(eR, tI)));
        O::store(zi + c, O::reverse(O::sub(eI, tR)));
    }
}

/// packRealSpectrumAt() at the widest vector whose bin pairs fit in M.
template <typename T>
void packRealSpectrum(const T* in, T* zr, T* zi, size_t m, const T* twRe, const T* twIm) noexcept
{
    if (m >= 2 * static_cast<size_t>(kWide<T>))
        packRealSpectrumAt<T, kWide<T>>(in, zr, zi, m, twRe, twIm);
    else if (m >= 2 * static_cast<size_t>(kNarrow<T>))
        packRealSpectrumAt<T, kNarrow<T>>(in, zr, zi, m, twRe, twIm);
    else
        packRealSpectrumAt<T, 1>(in, zr, zi, m, twRe, twIm);
}

/// Runs the closing pass with scaled interleaved output.
template <typename T>
void closeInterleaved(const Pass& pass, const T* xr, const T* xi, bool blocked, T* out, T scaleRe,
                      T scaleIm) noexcept
{
    closingPass<T, InterleavedSink>(pass, xr, xi, blocked, out, scaleRe, scaleIm);
}

/**
 * @brief Splits the half-size spectrum Z into the real spectrum X, W bins at a
 *        time with their mirror images: X[k] = (Z[k] + conj(Z[M-k])) / 2 -
 *        j w^k (Z[k] - conj(Z[M-k])) / 2.
 *
 * Bins k and M - k share their sums and differences (the second's are the
 * first's with signs flipped, exactly) and their twiddle product (w^(M-k) =
 * -conj(w^k) exactly, see FFTReal), so one pass over k <= M/2 writes both and
 * every value is what the formula gives for that bin alone. The halving is
 * done once, at the end (a power of two, so the result is unchanged).
 */
template <typename T, int W>
void unpackRealForwardAt(const T* zr, const T* zi, T* out, size_t m, const T* twRe,
                         const T* twIm) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const T dc = zr[0] + zi[0];
    const T ny = zr[0] - zi[0];
    const V half = O::set1(T(0.5));
    for (size_t k = 1; k <= m / 2; k += W)   // k .. k+W-1 against M-k .. M-k-W+1
    {
        const size_t c = m - k - (W - 1);
        const V hkR = O::load(zr + k), hkI = O::load(zi + k);
        const V hcR = O::reverse(O::load(zr + c)), hcI = O::reverse(O::load(zi + c));
        const V eR = O::add(hkR, hcR), eI = O::sub(hkI, hcI);
        const V oR = O::sub(hkR, hcR), oI = O::add(hkI, hcI);
        // Bin k: -j * o = (oI, -oR), times w = (wr, wi).
        const V wr = O::load(twRe + k), wi = O::load(twIm + k);
        const V tR = O::mulAdd(wr, oI, wi, oR);   // wr*oI - wi*(-oR)
        const V tI = O::mulSub(wi, oI, wr, oR);   // wr*(-oR) + wi*oI
        O::storeInterleave2(out + 2 * k, O::mul(half, O::add(eR, tR)), O::mul(half, O::add(eI, tI)));
        // Bin M - k: its e and o are (eR, -eI) and (-oR, oI), its twiddle
        // product (-tR, tI).
        O::storeInterleave2Reversed(out + 2 * c, O::mul(half, O::sub(eR, tR)), O::mul(half, O::sub(tI, eI)));
    }
    // Written last: in-place use shares the buffer with the input pairs,
    // which the engine has already consumed.
    out[0] = dc;
    out[1] = T(0);
    out[2 * m] = ny;
    out[2 * m + 1] = T(0);
}

/// unpackRealForwardAt() at the widest vector whose bin pairs fit in M.
template <typename T>
void unpackRealForward(const T* zr, const T* zi, T* out, size_t m, const T* twRe,
                       const T* twIm) noexcept
{
    if (m >= 2 * static_cast<size_t>(kWide<T>))
        unpackRealForwardAt<T, kWide<T>>(zr, zi, out, m, twRe, twIm);
    else if (m >= 2 * static_cast<size_t>(kNarrow<T>))
        unpackRealForwardAt<T, kNarrow<T>>(zr, zi, out, m, twRe, twIm);
    else
        unpackRealForwardAt<T, 1>(zr, zi, out, m, twRe, twIm);
}
