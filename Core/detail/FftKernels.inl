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

/// Whether the copy's W-lane vector deinterleaves without crossing its 128-bit
/// halves (the AVX widths): its lanes then come out as four runs of W / 4 in
/// the order (0, 2, 1, 3), which the first pass follows in its twiddles and
/// in where it stores each butterfly (see FFT.h, SplitFFT).
template <typename T, int W>
inline constexpr bool kInLaneDeinterleave = requires { &Vec<T, W>::loadDeinterleaveInLane; };

/// Whether this copy can hold double blocks with their lanes in the order
/// (0, 2, 1, 3) (the AVX width): a transform that closes into interleaved
/// output then writes that order from its first pass, so the closing pass
/// interleaves without crossing 128-bit halves. Every other pass is
/// lane-wise and does not see the order. Float keeps the natural order: its
/// half-block pass needs each half of a block to hold one butterfly group.
template <typename T>
inline constexpr bool kInLaneLayout = std::is_same_v<T, double> && kInLaneDeinterleave<T, kWide<T>>;

/// Where the first pass reads its input from, and where the last pass writes.
/// Reading interleaved data directly in the first pass, and writing it directly
/// in the last, saves one full pass over memory in each direction.
template <typename T, int W>
struct SplitSource
{
    using O = Vec<T, W>;
    static constexpr bool inLane = false;
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
    static constexpr bool inLane = kInLaneDeinterleave<T, W>;
    const T* data;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        if constexpr (inLane)
            O::loadDeinterleaveInLane(data + 2 * i, r, m);
        else
            O::loadDeinterleave(data + 2 * i, r, m);
    }
};

template <typename T, int W>
struct ConjugatedSource
{
    using O = Vec<T, W>;
    static constexpr bool inLane = kInLaneDeinterleave<T, W>;
    const T* data;
    void read(size_t i, typename O::V& r, typename O::V& m) const noexcept
    {
        if constexpr (inLane)
            O::loadDeinterleaveInLane(data + 2 * i, r, m);
        else
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
    static constexpr bool inLane = false;
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
    static constexpr bool inLane = false;
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

/// InterleavedSink for blocks in the in-lane order (kInLaneLayout); only
/// instantiated at the width that has it.
template <typename T, int W>
struct InterleavedInLaneSink
{
    using O = Vec<T, W>;
    T* data;
    typename O::V scaleRe;
    typename O::V scaleIm;
    void write(size_t i, typename O::V r, typename O::V m) const noexcept
    {
        O::storeInterleave2InLane(data + 2 * i, O::mul(r, scaleRe), O::mul(m, scaleIm));
    }
};

/// InterleavedSink with unit scales: the outputs as they are, with no
/// multiplication (a forward transform to interleaved output).
template <typename T, int W>
struct PlainInterleavedSink
{
    using O = Vec<T, W>;
    T* data;
    void write(size_t i, typename O::V r, typename O::V m) const noexcept
    {
        O::storeInterleave2(data + 2 * i, r, m);
    }
};

/// PlainInterleavedSink for blocks in the in-lane order (kInLaneLayout); only
/// instantiated at the width that has it.
template <typename T, int W>
struct PlainInterleavedInLaneSink
{
    using O = Vec<T, W>;
    T* data;
    void write(size_t i, typename O::V r, typename O::V m) const noexcept
    {
        O::storeInterleave2InLane(data + 2 * i, r, m);
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
    const T* const wp = tw.w;   // six factors per group, together
    if (s == B)
    {
        // One block per row: a group's twiddles serve one block, so they are
        // broadcast where they are used rather than kept live, and the four
        // output rows are adjacent blocks.
        for (size_t p = 0; p < n1; ++p)
        {
            const T* const a = x + 2 * B * p;
            const T* const b = a + rowIn;
            const T* const c = b + rowIn;
            const T* const d = c + rowIn;
            T* const y0 = y + 8 * B * p;
            const T* const t = wp + 6 * p;
            for (size_t h = 0; h < B; h += W)
            {
                const V a_r = O::load(a + h), a_i = O::load(a + B + h);
                const V b_r = O::load(b + h), b_i = O::load(b + B + h);
                const V c_r = O::load(c + h), c_i = O::load(c + B + h);
                const V d_r = O::load(d + h), d_i = O::load(d + B + h);
                const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
                const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
                const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
                const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
                const V t1R = O::add(amcR, bmdI), t1I = O::sub(amcI, bmdR);
                const V t2R = O::sub(apcR, bpdR), t2I = O::sub(apcI, bpdI);
                const V t3R = O::sub(amcR, bmdI), t3I = O::add(amcI, bmdR);
                O::store(y0 + h,         O::add(apcR, bpdR));
                O::store(y0 + B + h,     O::add(apcI, bpdI));
                V w = O::set1(t[0]), u = O::set1(t[1]);
                O::store(y0 + 2 * B + h, O::mulSub(t1R, w, t1I, u));
                O::store(y0 + 3 * B + h, O::mulAdd(t1R, u, t1I, w));
                w = O::set1(t[2]); u = O::set1(t[3]);
                O::store(y0 + 4 * B + h, O::mulSub(t2R, w, t2I, u));
                O::store(y0 + 5 * B + h, O::mulAdd(t2R, u, t2I, w));
                w = O::set1(t[4]); u = O::set1(t[5]);
                O::store(y0 + 6 * B + h, O::mulSub(t3R, w, t3I, u));
                O::store(y0 + 7 * B + h, O::mulAdd(t3R, u, t3I, w));
            }
        }
        return;
    }
    for (size_t p = 0; p < n1; ++p)
    {
        const T* const t = wp + 6 * p;
        const V w1r = O::set1(t[0]), w1i = O::set1(t[1]);
        const V w2r = O::set1(t[2]), w2i = O::set1(t[3]);
        const V w3r = O::set1(t[4]), w3i = O::set1(t[5]);
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
 * @brief One Stockham radix-8 DIF pass with stride s >= kBlock, blocked
 *        layout in and out, vectorised over the stride: three bits of the
 *        transform in one pass over the data.
 *
 * y[q + s(8p + k)] = w^(kp) * sum_r x[q + s(p + r n/8)] w8^(r k), the sum
 * split as radix8LastBlocked() splits it: two 4-point transforms, of the sums
 * x_r + x_(r+4) (even k) and of the differences times w8^r (odd k). tw holds
 * the fourteen factors of each group together, w^(kp) for k = 1 .. 7, real
 * part then imaginary part, each as a whole vector of W equal values: the
 * butterfly's sixteen vectors and its temporaries leave no register to keep
 * the factors in, and loading them where they are used takes no vector
 * operation, which a broadcast would.
 */
template <typename T, int W>
void radix8Blocked(const T* x, T* y, size_t n, size_t s, const T* tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    const size_t n1 = n / 8;
    const size_t rowIn = 2 * s * n1, rowOut = 2 * s;
    const V h = O::set1(static_cast<T>(0.70710678118654752440));
    const V nh = O::set1(static_cast<T>(-0.70710678118654752440));
    constexpr size_t L = static_cast<size_t>(W);
    // Row k of a group: the value times w^(kp), w at the factor's two vectors.
    const auto twisted = [](T* dst, V r, V m, const T* w) noexcept {
        const V wr = O::load(w), wi = O::load(w + static_cast<size_t>(W));
        O::store(dst, O::mulSub(r, wr, m, wi));
        O::store(dst + kBlock<T>, O::mulAdd(r, wi, m, wr));
    };
    for (size_t p = 0; p < n1; ++p)
    {
        const T* const t = tw + 14 * L * p;
        const T* const a = x + 2 * s * p;
        T* const y0 = y + 16 * s * p;
        for (size_t q = 0; q < 2 * s; q += 2 * B)
        {
            for (size_t hh = q; hh < q + B; hh += W)
            {
                const T* const lo = a + hh;
                const T* const hi = lo + 4 * rowIn;
                T* const o = y0 + hh;
                // Sums and differences of rows r and r + 4.
                const V x0r = O::load(lo), x0i = O::load(lo + B);
                const V x4r = O::load(hi), x4i = O::load(hi + B);
                const V u0r = O::add(x0r, x4r), u0i = O::add(x0i, x4i);
                const V v0r = O::sub(x0r, x4r), v0i = O::sub(x0i, x4i);
                const V x1r = O::load(lo + rowIn), x1i = O::load(lo + rowIn + B);
                const V x5r = O::load(hi + rowIn), x5i = O::load(hi + rowIn + B);
                const V u1r = O::add(x1r, x5r), u1i = O::add(x1i, x5i);
                const V v1r = O::sub(x1r, x5r), v1i = O::sub(x1i, x5i);
                const V x2r = O::load(lo + 2 * rowIn), x2i = O::load(lo + 2 * rowIn + B);
                const V x6r = O::load(hi + 2 * rowIn), x6i = O::load(hi + 2 * rowIn + B);
                const V u2r = O::add(x2r, x6r), u2i = O::add(x2i, x6i);
                const V v2r = O::sub(x2r, x6r), v2i = O::sub(x2i, x6i);
                const V x3r = O::load(lo + 3 * rowIn), x3i = O::load(lo + 3 * rowIn + B);
                const V x7r = O::load(hi + 3 * rowIn), x7i = O::load(hi + 3 * rowIn + B);
                const V u3r = O::add(x3r, x7r), u3i = O::add(x3i, x7i);
                const V v3r = O::sub(x3r, x7r), v3i = O::sub(x3i, x7i);
                // Even rows: the 4-point transform of the sums.
                {
                    const V apcR = O::add(u0r, u2r), apcI = O::add(u0i, u2i);
                    const V amcR = O::sub(u0r, u2r), amcI = O::sub(u0i, u2i);
                    const V bpdR = O::add(u1r, u3r), bpdI = O::add(u1i, u3i);
                    const V bmdR = O::sub(u1r, u3r), bmdI = O::sub(u1i, u3i);
                    O::store(o,     O::add(apcR, bpdR));
                    O::store(o + B, O::add(apcI, bpdI));
                    twisted(o + 2 * rowOut, O::add(amcR, bmdI), O::sub(amcI, bmdR), t + 2 * L);
                    twisted(o + 4 * rowOut, O::sub(apcR, bpdR), O::sub(apcI, bpdI), t + 6 * L);
                    twisted(o + 6 * rowOut, O::sub(amcR, bmdI), O::add(amcI, bmdR), t + 10 * L);
                }
                // Odd rows: the 4-point transform of the differences times
                // w8^r, that is 1, (1 - j) / sqrt 2, -j and -(1 + j) / sqrt 2.
                // The products by sqrt(1/2) join the sums they feed.
                {
                    const V apcR = O::add(v0r, v2i), apcI = O::sub(v0i, v2r);
                    const V amcR = O::sub(v0r, v2i), amcI = O::add(v0i, v2r);
                    const V m1r = O::add(v1r, v1i), m1i = O::sub(v1i, v1r);
                    const V n3r = O::sub(v3i, v3r), g3 = O::add(v3r, v3i);
                    const V sr = O::add(m1r, n3r), si = O::sub(m1i, g3);
                    const V dr = O::sub(m1r, n3r), di = O::add(m1i, g3);
                    twisted(o + rowOut,     O::madd(sr, h, apcR),  O::madd(si, h, apcI),  t);
                    twisted(o + 3 * rowOut, O::madd(di, h, amcR),  O::madd(dr, nh, amcI), t + 4 * L);
                    twisted(o + 5 * rowOut, O::madd(sr, nh, apcR), O::madd(si, nh, apcI), t + 8 * L);
                    twisted(o + 7 * rowOut, O::madd(di, nh, amcR), O::madd(dr, h, amcI),  t + 12 * L);
                }
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
    const T* const wp = tw.w;   // six factors per group, together
    for (size_t p = 0; p < n1; ++p)
    {
        const T* const t = wp + 6 * p;
        const V w1r = O::set1(t[0]), w1i = O::set1(t[1]);
        const V w2r = O::set1(t[2]), w2i = O::set1(t[3]);
        const V w3r = O::set1(t[4]), w3i = O::set1(t[5]);
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
template <typename T, int W, typename Source, bool InLaneOut = false>
void radix4First(const Source& input, T* y, size_t n, const TwiddleSpan<T>& tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    const size_t n1 = n / 4;
    // Locals, not the structs: a store through y may not alias them, so the
    // pointers stay in registers, and each twiddle is loaded where it is used.
    const Source in = input;
    // The twiddles sit in blocks of bf groups, six rows of bf each.
    const T* const wp = tw.w;
    const size_t bf = TwiddleSpan<T>::firstBlock(n1, B);
    for (size_t p = 0; p < n1; p += W)
    {
        const size_t l = p & (bf - 1);
        const T* const t = wp + 6 * (p - l) + l;
        V a_r, a_i, b_r, b_i, c_r, c_i, d_r, d_i;
        in.read(p, a_r, a_i);
        in.read(p + n1, b_r, b_i);
        in.read(p + 2 * n1, c_r, c_i);
        in.read(p + 3 * n1, d_r, d_i);
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
        const V y0r = O::add(apcR, bpdR), y0i = O::add(apcI, bpdI);
        V w = O::load(t), u = O::load(t + bf);
        const V y1r = O::mulSub(t1R, w, t1I, u), y1i = O::mulAdd(t1R, u, t1I, w);
        w = O::load(t + 2 * bf); u = O::load(t + 3 * bf);
        const V y2r = O::mulSub(t2R, w, t2I, u), y2i = O::mulAdd(t2R, u, t2I, w);
        w = O::load(t + 4 * bf); u = O::load(t + 5 * bf);
        const V y3r = O::mulSub(t3R, w, t3I, u), y3i = O::mulAdd(t3R, u, t3I, w);
        // In-lane input holds the butterflies' runs in the order (0, 2, 1, 3).
        T* const s1 = Source::inLane ? r2 : r1;
        T* const s2 = Source::inLane ? r1 : r2;
        // The in-lane layout holds a block's four outputs as (0, 2, 1, 3).
        if constexpr (InLaneOut)
        {
            O::storeInterleave4To(r0, s1, s2, r3, y0r, y2r, y1r, y3r);
            O::storeInterleave4To(r0 + B, s1 + B, s2 + B, r3 + B, y0i, y2i, y1i, y3i);
        }
        else
        {
            O::storeInterleave4To(r0, s1, s2, r3, y0r, y1r, y2r, y3r);
            O::storeInterleave4To(r0 + B, s1 + B, s2 + B, r3 + B, y0i, y1i, y2i, y3i);
        }
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
    const Sink out = output;   // a local: the stores cannot alias its pointers
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
            out.write(q,         O::add(apcR, bpdR), O::add(apcI, bpdI));
            out.write(q + s,     O::add(amcR, bmdI), O::sub(amcI, bmdR));
            out.write(q + 2 * s, O::sub(apcR, bpdR), O::sub(apcI, bpdI));
            out.write(q + 3 * s, O::sub(amcR, bmdI), O::add(amcI, bmdR));
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
    const Sink out = output;   // a local: the stores cannot alias its pointers
    for (size_t qb = 0; qb < s; qb += B)
    {
        const T* const r0 = x + 2 * qb;
        const T* const r1 = r0 + 2 * s;
        for (size_t h = 0; h < B; h += W)
        {
            const size_t q = qb + h;
            const V a_r = O::load(r0 + h), a_i = O::load(r0 + B + h);
            const V b_r = O::load(r1 + h), b_i = O::load(r1 + B + h);
            out.write(q, O::add(a_r, b_r), O::add(a_i, b_i));
            out.write(q + s, O::sub(a_r, b_r), O::sub(a_i, b_i));
        }
    }
}

/// radix8LastBlocked() of a stride of one block: a single column, whose
/// sums and differences are all taken at once (one sweep over the inputs).
template <typename T, int W, typename Sink>
void radix8LastOneBlock(const T* x, const Sink& output) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>, s = B;
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
    for (size_t hh = 0; hh < B; hh += W)
    {
        V ar[4], ai[4], br[4], bi[4];
        for (size_t r = 0; r < 4; ++r)
        {
            const T* const lo = x + 2 * s * r + hh;
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
        dft4(hh, 0, ar, ai);
        dft4(hh, 1, br, bi);
    }
}

/**
 * @brief The closing radix-8 pass (n = 8, stride s = N / 8 >= kBlock) from the
 *        blocked layout: it replaces a radix-4 pass of length 8 and the
 *        closing radix-2 pass that would follow it, one pass over the data
 *        fewer. Twiddle-free: the eighth roots enter as the exact sqrt(1/2).
 *
 * y[q + s k] = sum_r x[q + s r] w8^(r k), split as two 4-point transforms of
 * the sums x_r + x_(r+4) (even k) and the twiddled differences (odd k). Past
 * one block of stride the two are taken in two sweeps over the inputs, so
 * only eight sums or differences are live at a time (sixteen do not fit the
 * vector registers).
 */
template <typename T, int W, typename Sink>
void radix8LastBlocked(const T* x, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t B = kBlock<T>;
    if (s == B)
    {
        radix8LastOneBlock<T, W>(x, output);
        return;
    }
    const Sink out = output;   // a local: the stores cannot alias its pointers
    // 4-point transform of z0..z3 into rows k0, k0 + 2, k0 + 4, k0 + 6.
    const auto dft4 = [s, &out](size_t q, size_t k0, V z0r, V z0i, V z1r, V z1i, V z2r, V z2i,
                                V z3r, V z3i) noexcept {
        const V apcR = O::add(z0r, z2r), apcI = O::add(z0i, z2i);
        const V amcR = O::sub(z0r, z2r), amcI = O::sub(z0i, z2i);
        const V bpdR = O::add(z1r, z3r), bpdI = O::add(z1i, z3i);
        const V bmdR = O::sub(z1r, z3r), bmdI = O::sub(z1i, z3i);
        out.write(q + s * k0,       O::add(apcR, bpdR), O::add(apcI, bpdI));
        out.write(q + s * (k0 + 2), O::add(amcR, bmdI), O::sub(amcI, bmdR));
        out.write(q + s * (k0 + 4), O::sub(apcR, bpdR), O::sub(apcI, bpdI));
        out.write(q + s * (k0 + 6), O::sub(amcR, bmdI), O::add(amcI, bmdR));
    };
    for (size_t qb = 0; qb < s; qb += B)
    {
        const T* const r0 = x + 2 * qb;
        for (size_t hh = 0; hh < B; hh += W)
        {
            const T* const lo0 = r0 + hh;
            const T* const lo1 = lo0 + 2 * s;
            const T* const lo2 = lo0 + 4 * s;
            const T* const lo3 = lo0 + 6 * s;
            const size_t hi = 8 * s;
            dft4(qb + hh, 0,
                 O::add(O::load(lo0), O::load(lo0 + hi)), O::add(O::load(lo0 + B), O::load(lo0 + hi + B)),
                 O::add(O::load(lo1), O::load(lo1 + hi)), O::add(O::load(lo1 + B), O::load(lo1 + hi + B)),
                 O::add(O::load(lo2), O::load(lo2 + hi)), O::add(O::load(lo2 + B), O::load(lo2 + hi + B)),
                 O::add(O::load(lo3), O::load(lo3 + hi)), O::add(O::load(lo3 + B), O::load(lo3 + hi + B)));
            // Differences times w8^r: (1 - j) / sqrt 2, -j, -(1 + j) / sqrt 2.
            const V h = O::set1(static_cast<T>(0.70710678118654752440));
            const V zero = O::set1(T(0));
            const V b0r = O::sub(O::load(lo0), O::load(lo0 + hi));
            const V b0i = O::sub(O::load(lo0 + B), O::load(lo0 + hi + B));
            const V d1r = O::sub(O::load(lo1), O::load(lo1 + hi));
            const V d1i = O::sub(O::load(lo1 + B), O::load(lo1 + hi + B));
            const V d2r = O::sub(O::load(lo2), O::load(lo2 + hi));
            const V d2i = O::sub(O::load(lo2 + B), O::load(lo2 + hi + B));
            const V d3r = O::sub(O::load(lo3), O::load(lo3 + hi));
            const V d3i = O::sub(O::load(lo3 + B), O::load(lo3 + hi + B));
            dft4(qb + hh, 1, b0r, b0i,
                 O::mul(O::add(d1r, d1i), h), O::mul(O::sub(d1i, d1r), h),
                 d2i, O::sub(zero, d2r),
                 O::mul(O::sub(d3i, d3r), h), O::sub(zero, O::mul(O::add(d3r, d3i), h)));
        }
    }
}

/// The twiddles of one radix-4 pass.
template <typename T>
TwiddleSpan<T> twiddleSpanOf(const Pass& pass, const T* twiddles) noexcept
{
    return { twiddles + pass.twOffset };
}

/// Runs the first pass at the widest vector its length allows, into the
/// blocked layout at y.
template <typename T, template <typename, int> class Source, bool InLaneOut, typename... Args>
void firstPassAs(const Pass& pass, const T* twiddles, const T* inLaneTwiddles, T* y, Args... input) noexcept
{
    const size_t n1 = pass.len / 4;
    const TwiddleSpan<T> tw = twiddleSpanOf(pass, twiddles);
    constexpr int wide = kWide<T>;
    constexpr int narrow = kNarrow<T>;
    // In-lane input needs its permuted twiddles; without them it is read at
    // the narrow width.
    if (n1 % wide == 0 && (!Source<T, wide>::inLane || inLaneTwiddles != nullptr))
    {
        if constexpr (Source<T, wide>::inLane)
        {
            // The same twiddles, permuted as the in-lane input (see SplitFFT).
            const Pass permuted { pass.len, pass.stride, 0, false, false, false };
            radix4First<T, wide, Source<T, wide>, InLaneOut>(Source<T, wide> { input... }, y, pass.len,
                                                              twiddleSpanOf(permuted, inLaneTwiddles));
        }
        else
            radix4First<T, wide, Source<T, wide>, InLaneOut>(Source<T, wide> { input... }, y, pass.len, tw);
    }
    else if (n1 % narrow == 0)
        radix4First<T, narrow, Source<T, narrow>, InLaneOut>(Source<T, narrow> { input... }, y, pass.len, tw);
    else
        radix4First<T, 1, Source<T, 1>, InLaneOut>(Source<T, 1> { input... }, y, pass.len, tw);
}

/// firstPassAs() writing the in-lane layout when asked (and this copy has it).
template <typename T, template <typename, int> class Source, typename... Args>
void firstPass(const Pass& pass, const T* twiddles, const T* inLaneTwiddles, bool inLaneOut, T* y,
               Args... input) noexcept
{
    if constexpr (kInLaneLayout<T>)
    {
        if (inLaneOut)
        {
            firstPassAs<T, Source, true>(pass, twiddles, inLaneTwiddles, y, input...);
            return;
        }
    }
    firstPassAs<T, Source, false>(pass, twiddles, inLaneTwiddles, y, input...);
}

/// Whether a transform of these passes that closes into interleaved output
/// uses the in-lane layout on this copy: it needs a blocked closing pass at
/// the full width.
template <typename T>
bool inLaneLayoutFor(const Pass* passes, size_t passCount) noexcept
{
    if constexpr (kInLaneLayout<T>)
        return passCount > 1 && passes[passCount - 1].stride % static_cast<size_t>(kWide<T>) == 0;
    else
        return false;
}

/// Runs a twiddled pass of stride > 1 at the widest vector its stride allows,
/// blocked layout in and out. The radix-8 ones come with whole blocks of
/// stride; of the radix-4 ones, the only stride below the block is 4, half a
/// float block.
template <typename T>
void stridedPass(const Pass& pass, const T* twiddles, const Buffers<T>& b) noexcept
{
    if (pass.twiddled8)
    {
        radix8Blocked<T, kWide<T>>(b.xr, b.yr, pass.len, pass.stride, twiddles + pass.twOffset);
        return;
    }
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

template <typename T, int W, template <typename, int> class Sink>
Sink<T, W> makeSink(T* data) noexcept
{
    return { data };
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
void runPasses(const Pass* passes, size_t begin, size_t end, const T* twiddles, bool inLaneOut,
               Buffers<T>& b) noexcept
{
    for (size_t i = begin; i < end; ++i)
    {
        const Pass& pass = passes[i];
        if (pass.radix2 || pass.radix8 || pass.len == 4)
            closingPass<T, SplitSink>(pass, b.xr, b.xi, i > 0, b.yr, b.yi);
        else if (pass.stride == 1)
            firstPass<T, SplitSource>(pass, twiddles, nullptr, inLaneOut, b.yr, b.xr, b.xi);
        else
            stridedPass(pass, twiddles, b);
        T* r = b.xr; b.xr = b.yr; b.yr = r;
        T* m = b.xi; b.xi = b.yi; b.yi = m;
    }
}

/// Reads interleaved input, conjugated if asked; returns the passes it ran.
template <typename T>
size_t readInput(const Pass* passes, size_t passCount, size_t n, const T* twiddles,
                 const T* inLaneTwiddles, bool inLaneOut, const T* data, bool conjugate,
                 Buffers<T>& b) noexcept
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
        firstPass<T, ConjugatedSource>(passes[0], twiddles, inLaneTwiddles, inLaneOut, b.yr, data);
    else
        firstPass<T, InterleavedSource>(passes[0], twiddles, inLaneTwiddles, inLaneOut, b.yr, data);
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
    firstPass<T, SplitSource>(passes[0], twiddles, nullptr, false, b.yr, re, im);
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
    firstPass<T, ScaledSwapSource>(passes[0], twiddles, nullptr, false, b.yr, re, im, gain, mask);
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
    firstPass<T, FoldedScaledSwapSource>(passes[0], twiddles, nullptr, false, b.yr, re, im, gain, n, fold);
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
 * Taken as Z[k] = conj(X[M-k]) + conj(c_k) d and Z[M-k] = conj(X[k] -
 * conj(c_k) d), with d = X[k] - conj(X[M-k]) and c_k = (1 - j w^k) / 2 (the
 * split factors, see FFTReal): one difference and one complex product serve
 * both bins, and nothing is halved. c_(M-k) = conj(c_k) exactly in the table
 * and c_(M/2) = 0, so one pass over k <= M/2 writes both bins and every value
 * is what the formula gives for that bin alone. The imaginary parts of bins 0
 * and M are ignored, as a real signal has none.
 */
template <typename T, int W>
void packRealSpectrumAt(const T* in, T* zr, T* zi, size_t m, const T* cRe, const T* cIm) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const T dc = in[0];
    const T ny = in[2 * m];
    zr[0] = T(0.5) * (dc + ny);
    zi[0] = T(0.5) * (ny - dc);   // conjugated
    for (size_t k = 1; k <= m / 2; k += W)   // k .. k+W-1 against M-k .. M-k-W+1
    {
        const size_t c = m - k - (W - 1);
        V pR, pI, rR, rI;
        O::loadDeinterleave(in + 2 * k, pR, pI);
        O::loadDeinterleaveReversed(in + 2 * c, rR, rI);
        const V dR = O::sub(pR, rR), dI = O::add(pI, rI);
        // u = conj(c) * d
        const V cr = O::load(cRe + k), ci = O::load(cIm + k);
        const V uR = O::mulAdd(cr, dR, ci, dI);
        const V uI = O::mulSub(cr, dI, ci, dR);
        // conj(Z[k]) = X[M-k] + conj(u), conj(Z[M-k]) = X[k] - u.
        O::store(zr + k, O::add(rR, uR));
        O::store(zi + k, O::sub(rI, uI));
        O::store(zr + c, O::reverse(O::sub(pR, uR)));
        O::store(zi + c, O::reverse(O::sub(pI, uI)));
    }
}

/// packRealSpectrumAt() at the widest vector whose bin pairs fit in M.
template <typename T>
void packRealSpectrum(const T* in, T* zr, T* zi, size_t m, const T* cRe, const T* cIm) noexcept
{
    if (m >= 2 * static_cast<size_t>(kWide<T>))
        packRealSpectrumAt<T, kWide<T>>(in, zr, zi, m, cRe, cIm);
    else if (m >= 2 * static_cast<size_t>(kNarrow<T>))
        packRealSpectrumAt<T, kNarrow<T>>(in, zr, zi, m, cRe, cIm);
    else
        packRealSpectrumAt<T, 1>(in, zr, zi, m, cRe, cIm);
}

/// Runs the closing pass with scaled interleaved output. Unit scales (a
/// forward transform) are not multiplied in: the same values, since a
/// product by one is exact.
template <typename T>
void closeInterleaved(const Pass& pass, const T* xr, const T* xi, bool blocked, bool inLane, T* out,
                      T scaleRe, T scaleIm) noexcept
{
    const bool unit = scaleRe == T(1) && scaleIm == T(1);
    if constexpr (kInLaneLayout<T>)
    {
        // inLaneLayoutFor() guarantees a blocked pass at the full width.
        if (inLane)
        {
            if (unit)
                closingAt<T, kWide<T>, PlainInterleavedInLaneSink>(pass, xr, xi, true, out);
            else
                closingAt<T, kWide<T>, InterleavedInLaneSink>(pass, xr, xi, true, out, scaleRe, scaleIm);
            return;
        }
    }
    if (unit)
        closingPass<T, PlainInterleavedSink>(pass, xr, xi, blocked, out);
    else
        closingPass<T, InterleavedSink>(pass, xr, xi, blocked, out, scaleRe, scaleIm);
}

/**
 * @brief Splits the half-size spectrum Z into the real spectrum X, W bins at a
 *        time with their mirror images: X[k] = (Z[k] + conj(Z[M-k])) / 2 -
 *        j w^k (Z[k] - conj(Z[M-k])) / 2.
 *
 * Taken as X[k] = conj(Z[M-k]) + c_k d and X[M-k] = conj(Z[k] - c_k d), with
 * d = Z[k] - conj(Z[M-k]) and c_k = (1 - j w^k) / 2 (the split factors, see
 * FFTReal): one difference and one complex product serve both bins, and
 * nothing is halved. c_(M-k) = conj(c_k) exactly in the table and c_(M/2) =
 * 0, so one pass over k <= M/2 writes both bins and every value is what the
 * formula gives for that bin alone.
 */
template <typename T, int W>
void unpackRealForwardAt(const T* zr, const T* zi, T* out, size_t m, const T* cRe,
                         const T* cIm) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const T dc = zr[0] + zi[0];
    const T ny = zr[0] - zi[0];
    for (size_t k = 1; k <= m / 2; k += W)   // k .. k+W-1 against M-k .. M-k-W+1
    {
        const size_t c = m - k - (W - 1);
        const V aR = O::load(zr + k), aI = O::load(zi + k);
        const V hR = O::reverse(O::load(zr + c)), hI = O::reverse(O::load(zi + c));
        const V dR = O::sub(aR, hR), dI = O::add(aI, hI);
        // u = c * d
        const V cr = O::load(cRe + k), ci = O::load(cIm + k);
        const V uR = O::mulSub(cr, dR, ci, dI);
        const V uI = O::mulAdd(cr, dI, ci, dR);
        // X[k] = conj(Z[M-k]) + u, X[M-k] = conj(Z[k] - u).
        O::storeInterleave2(out + 2 * k, O::add(hR, uR), O::sub(uI, hI));
        O::storeInterleave2Reversed(out + 2 * c, O::sub(aR, uR), O::sub(uI, aI));
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
void unpackRealForward(const T* zr, const T* zi, T* out, size_t m, const T* cRe,
                       const T* cIm) noexcept
{
    if (m >= 2 * static_cast<size_t>(kWide<T>))
        unpackRealForwardAt<T, kWide<T>>(zr, zi, out, m, cRe, cIm);
    else if (m >= 2 * static_cast<size_t>(kNarrow<T>))
        unpackRealForwardAt<T, kNarrow<T>>(zr, zi, out, m, cRe, cIm);
    else
        unpackRealForwardAt<T, 1>(zr, zi, out, m, cRe, cIm);
}

// ---------------------------------------------------------------------------
// Whole transforms. SplitFFT (FFT.h) chooses the kernel copy once per
// transform and calls one of these, so every pass of a transform runs on one
// copy and they all agree on the work layout.
// ---------------------------------------------------------------------------

/// The passes on the input buffers, split output (SplitFFT::run()).
template <typename T>
void transformRun(const Plan<T>& p, Buffers<T>& b) noexcept
{
    runPasses(p.passes, 0, p.passCount, p.twiddles, false, b);
}

/// Interleaved input, conjugated if asked, split output (SplitFFT::runFrom()).
template <typename T>
void transformFrom(const Plan<T>& p, Buffers<T>& b, const T* data, bool conjugate) noexcept
{
    const size_t done = readInput(p.passes, p.passCount, p.n, p.twiddles, p.inLaneTwiddles, false,
                                  data, conjugate, b);
    runPasses(p.passes, done, p.passCount, p.twiddles, false, b);
}

/// Split input from outside the work buffers (SplitFFT::runFromSplit()).
template <typename T>
void transformFromSplit(const Plan<T>& p, Buffers<T>& b, const T* re, const T* im) noexcept
{
    const size_t done = readSplit(p.passes, p.passCount, p.n, p.twiddles, re, im, b);
    runPasses(p.passes, done, p.passCount, p.twiddles, false, b);
}

/// A scaled, swapped spectrum product (SplitFFT::runScaledSwap()).
template <typename T>
void transformScaledSwap(const Plan<T>& p, Buffers<T>& b, const T* re, const T* im, const T* gain,
                         size_t mask) noexcept
{
    const size_t done = readScaledSwap(p.passes, p.passCount, p.n, p.twiddles, re, im, gain, mask, b);
    runPasses(p.passes, done, p.passCount, p.twiddles, false, b);
}

/// Its folded form (SplitFFT::runScaledSwapFolded()).
template <typename T>
void transformScaledSwapFolded(const Plan<T>& p, Buffers<T>& b, const T* re, const T* im, const T* gain,
                               size_t fold) noexcept
{
    const size_t done = readScaledSwapFolded(p.passes, p.passCount, p.n, p.twiddles, re, im, gain, fold, b);
    runPasses(p.passes, done, p.passCount, p.twiddles, false, b);
}

/// The input buffers to interleaved output (SplitFFT::runTo()).
template <typename T>
void transformTo(const Plan<T>& p, Buffers<T>& b, const Closing<T>& c) noexcept
{
    const bool inLane = inLaneLayoutFor<T>(p.passes, p.passCount);
    runPasses(p.passes, 0, p.passCount - 1, p.twiddles, inLane, b);
    closeInterleaved(p.passes[p.passCount - 1], b.xr, b.xi, p.passCount > 1, inLane, c.out, c.scaleRe,
                     c.scaleIm);
}

/// Interleaved input to interleaved output (SplitFFT::runFromTo()).
template <typename T>
void transformFromTo(const Plan<T>& p, Buffers<T>& b, const T* data, bool conjugate, const Closing<T>& c) noexcept
{
    const bool inLane = inLaneLayoutFor<T>(p.passes, p.passCount);
    const size_t done = readInput(p.passes, p.passCount, p.n, p.twiddles, p.inLaneTwiddles, inLane,
                                  data, conjugate, b);
    runPasses(p.passes, done, p.passCount - 1, p.twiddles, inLane, b);
    closeInterleaved(p.passes[p.passCount - 1], b.xr, b.xi, p.passCount > 1, inLane, c.out, c.scaleRe,
                     c.scaleIm);
}

/// One step of transformTo() (SplitFFT::runToStep()).
template <typename T>
void transformToStep(const Plan<T>& p, Buffers<T>& b, size_t step, const Closing<T>& c) noexcept
{
    const bool inLane = inLaneLayoutFor<T>(p.passes, p.passCount);
    if (step + 1 < p.passCount)
        runPasses(p.passes, step, step + 1, p.twiddles, inLane, b);
    else
        closeInterleaved(p.passes[p.passCount - 1], b.xr, b.xi, p.passCount > 1, inLane, c.out, c.scaleRe,
                         c.scaleIm);
}

// ---------------------------------------------------------------------------
// The real transform by lanes (FFTReal).
//
// The N samples are W sequences of n = N / W points, point t of sequence j
// at x[W t + j]: a vector loaded at a multiple of W holds one point of each.
// The passes below work on such vectors as a scalar transform works on
// numbers, so each sequence is transformed in its own lane and no pass
// shuffles anything; one last step joins the W spectra into the spectrum of
// the N samples, W bins of each at a time (the only place where lanes mix).
//
// A spectrum of m real points is kept packed in m elements: X[0] and X[m/2]
// (both real) in elements 0 and 1, then the real and imaginary parts of
// X[1] .. X[m/2 - 1] in elements 2i and 2i + 1. A buffer holds the spectra of
// the sequences t0 + (n / m) t, t0 = 0 .. n/m - 1, one after another: the
// samples themselves are that with m = 1. A pass joins the spectra t0,
// t0 + q, t0 + 2q and t0 + 3q (q a quarter of the spectra) into the one of
// t0 with four times the points; its twiddles depend on the bin alone, so
// every group and every lane shares them.
// ---------------------------------------------------------------------------

/// One twiddle of a lane pass: a whole vector where the table repeats each
/// value across the lanes, a broadcast otherwise.
template <typename T, int W, bool Whole>
inline typename Vec<T, W>::V laneTwiddle(const T* at) noexcept
{
    if constexpr (Whole) return Vec<T, W>::load(at);
    else return Vec<T, W>::set1(*at);
}

/// The first pass of a lane transform whose log2(n) is even: the four-point
/// spectra of the samples t0, t0 + n/4, t0 + n/2 and t0 + 3n/4.
template <typename T, int W>
void realFirst4(const T* in, T* out, size_t n) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    const size_t quarter = w * (n / 4);
    for (const T* const end = in + quarter; in != end; in += w, out += 4 * w)
    {
        const V a0 = O::load(in), a1 = O::load(in + quarter);
        const V a2 = O::load(in + 2 * quarter), a3 = O::load(in + 3 * quarter);
        const V s02 = O::add(a0, a2), s13 = O::add(a1, a3);
        O::store(out, O::add(s02, s13));           // X[0]
        O::store(out + w, O::sub(s02, s13));       // X[2]
        O::store(out + 2 * w, O::sub(a0, a2));     // X[1] = (a0 - a2) + j (a3 - a1)
        O::store(out + 3 * w, O::sub(a3, a1));
    }
}

/// The first pass where log2(n) is odd: the eight-point spectra of the
/// samples t0 + k n/8, in one sweep (a radix-2 pass and a radix-4 pass over
/// the samples would cost the same additions and one more sweep).
template <typename T, int W>
void realFirst8(const T* in, T* out, size_t n) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    const size_t eighth = w * (n / 8);
    const V h = O::set1(static_cast<T>(0.70710678118654752440));
    const V nh = O::set1(static_cast<T>(-0.70710678118654752440));
    for (const T* const end = in + eighth; in != end; in += w, out += 8 * w)
    {
        const V x0 = O::load(in), x4 = O::load(in + 4 * eighth);
        const V x1 = O::load(in + eighth), x5 = O::load(in + 5 * eighth);
        const V x2 = O::load(in + 2 * eighth), x6 = O::load(in + 6 * eighth);
        const V x3 = O::load(in + 3 * eighth), x7 = O::load(in + 7 * eighth);
        const V s0 = O::add(x0, x4), s1 = O::add(x1, x5), s2 = O::add(x2, x6), s3 = O::add(x3, x7);
        const V d0 = O::sub(x0, x4), d1 = O::sub(x1, x5), d2 = O::sub(x2, x6), d3 = O::sub(x3, x7);
        const V s02 = O::add(s0, s2), s13 = O::add(s1, s3);
        // The odd bins take the differences turned by w8^k: X[1] = d0 +
        // h (d1 - d3) - j (d2 + h (d1 + d3)), X[3] = d0 - h (d1 - d3) +
        // j (d2 - h (d1 + d3)), h = sqrt(1/2).
        const V p = O::mul(h, O::sub(d1, d3)), r = O::mul(nh, O::add(d1, d3));
        O::store(out, O::add(s02, s13));           // X[0]
        O::store(out + w, O::sub(s02, s13));       // X[4]
        O::store(out + 2 * w, O::add(d0, p));      // X[1]
        O::store(out + 3 * w, O::sub(r, d2));
        O::store(out + 4 * w, O::sub(s0, s2));     // X[2]
        O::store(out + 5 * w, O::sub(s3, s1));
        O::store(out + 6 * w, O::sub(d0, p));      // X[3]
        O::store(out + 7 * w, O::add(d2, r));
    }
}

/**
 * @brief Joins four spectra of m points (m >= 4) into one of 4m, for every
 *        group of the buffer: Z[k] = sum_j w^(jk) X_j[k mod m], w =
 *        exp(-2 pi i / 4m).
 *
 * With t_j = w^(ji) X_j[i] for a bin 0 < i < m/2, u0 = X_0 + t2, u1 = X_0 -
 * t2, u2 = t1 + t3 and u3 = t3 - t1, the four bins that bin i reaches are
 * Z[i] = u0 + u2, Z[2m - i] = conj(u0 - u2), Z[m + i] = u1 + j u3 and
 * Z[m - i] = conj(u1 - j u3). Bins 0 and m/2 of the inputs are real and
 * take a few additions each.
 *
 * @param in  n / m spectra of m points.
 * @param out Receives n / 4m spectra of 4m points.
 * @param tw  (re, im) of w^i, w^2i, w^3i for i = 1 .. m/2 - 1.
 */
template <typename T, int W, bool Whole>
void realRadix4(const T* in, T* out, size_t n, size_t m, const T* tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    constexpr size_t tl = Whole ? w : 1;
    const size_t quarter = w * (n / 4);   // between the four spectra that join
    const size_t run = w * m;             // one of them
    const V h = O::set1(static_cast<T>(0.70710678118654752440));
    const V nh = O::set1(static_cast<T>(-0.70710678118654752440));
    for (const T* const end = in + quarter; in != end; in += run, out += 4 * run)
    {
        const T* const x0 = in;
        const T* const x1 = in + quarter;
        const T* const x2 = in + 2 * quarter;
        const T* const x3 = in + 3 * quarter;
        {
            // Bin 0 of each: bins 0, 2m (both real) and m.
            const V a0 = O::load(x0), a1 = O::load(x1), a2 = O::load(x2), a3 = O::load(x3);
            const V s02 = O::add(a0, a2), s13 = O::add(a1, a3);
            O::store(out, O::add(s02, s13));
            O::store(out + w, O::sub(s02, s13));
            O::store(out + 2 * run, O::sub(a0, a2));
            O::store(out + 2 * run + w, O::sub(a3, a1));
        }
        {
            // Bin m/2 of each: bins m/2 and 3m/2, turned by the eighth roots.
            const V b0 = O::load(x0 + w), b1 = O::load(x1 + w), b2 = O::load(x2 + w), b3 = O::load(x3 + w);
            const V p = O::mul(h, O::sub(b1, b3)), r = O::mul(nh, O::add(b1, b3));
            O::store(out + run, O::add(b0, p));
            O::store(out + run + w, O::sub(r, b2));
            O::store(out + 3 * run, O::sub(b0, p));
            O::store(out + 3 * run + w, O::add(b2, r));
        }
        const T* t = tw;
        for (size_t e = 2 * w; e < run; e += 2 * w, t += 6 * tl)
        {
            const V ar = O::load(x0 + e), ai = O::load(x0 + e + w);
            const V br = O::load(x1 + e), bi = O::load(x1 + e + w);
            const V cr = O::load(x2 + e), ci = O::load(x2 + e + w);
            const V dr = O::load(x3 + e), di = O::load(x3 + e + w);
            const V w1r = laneTwiddle<T, W, Whole>(t), w1i = laneTwiddle<T, W, Whole>(t + tl);
            const V w2r = laneTwiddle<T, W, Whole>(t + 2 * tl), w2i = laneTwiddle<T, W, Whole>(t + 3 * tl);
            const V w3r = laneTwiddle<T, W, Whole>(t + 4 * tl), w3i = laneTwiddle<T, W, Whole>(t + 5 * tl);
            const V t1r = O::mulSub(w1r, br, w1i, bi), t1i = O::mulAdd(w1r, bi, w1i, br);
            const V t2r = O::mulSub(w2r, cr, w2i, ci), t2i = O::mulAdd(w2r, ci, w2i, cr);
            const V t3r = O::mulSub(w3r, dr, w3i, di), t3i = O::mulAdd(w3r, di, w3i, dr);
            const V u0r = O::add(ar, t2r), u0i = O::add(ai, t2i);
            const V u1r = O::sub(ar, t2r), u1i = O::sub(ai, t2i);
            const V u2r = O::add(t1r, t3r), u2i = O::add(t1i, t3i);
            const V u3r = O::sub(t3r, t1r), u3i = O::sub(t3i, t1i);
            T* const z0 = out + e;               // bin i
            T* const z1 = out + 2 * run - e;     // bin m - i
            T* const z2 = out + 2 * run + e;     // bin m + i
            T* const z3 = out + 4 * run - e;     // bin 2m - i
            O::store(z0, O::add(u0r, u2r));
            O::store(z0 + w, O::add(u0i, u2i));
            O::store(z1, O::add(u1r, u3i));
            O::store(z1 + w, O::sub(u3r, u1i));
            O::store(z2, O::sub(u1r, u3i));
            O::store(z2 + w, O::add(u1i, u3r));
            O::store(z3, O::sub(u0r, u2r));
            O::store(z3 + w, O::sub(u2i, u0i));
        }
    }
}

/// realRadix4() undone, four times over: splits the spectra of 4m points
/// into four of m (the same twiddles, conjugated).
template <typename T, int W, bool Whole>
void realInverseRadix4(const T* in, T* out, size_t n, size_t m, const T* tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    constexpr size_t tl = Whole ? w : 1;
    const size_t quarter = w * (n / 4);
    const size_t run = w * m;
    const V k = O::set1(static_cast<T>(1.41421356237309504880));
    const V nk = O::set1(static_cast<T>(-1.41421356237309504880));
    for (T* const end = out + quarter; out != end; out += run, in += 4 * run)
    {
        T* const x0 = out;
        T* const x1 = out + quarter;
        T* const x2 = out + 2 * quarter;
        T* const x3 = out + 3 * quarter;
        {
            const V z0 = O::load(in), z2 = O::load(in + w);
            const V zr = O::load(in + 2 * run), zi = O::load(in + 2 * run + w);
            const V s = O::add(z0, z2), d = O::sub(z0, z2);
            const V zr2 = O::add(zr, zr), zi2 = O::add(zi, zi);
            O::store(x0, O::add(s, zr2));
            O::store(x1, O::sub(d, zi2));
            O::store(x2, O::sub(s, zr2));
            O::store(x3, O::add(d, zi2));
        }
        {
            const V pr = O::load(in + run), pi = O::load(in + run + w);
            const V qr = O::load(in + 3 * run), qi = O::load(in + 3 * run + w);
            const V a = O::add(pr, qr), b = O::sub(pr, qr), c = O::sub(qi, pi), d = O::add(pi, qi);
            O::store(x0 + w, O::add(a, a));
            O::store(x1 + w, O::mul(k, O::sub(b, d)));
            O::store(x2 + w, O::add(c, c));
            O::store(x3 + w, O::mul(nk, O::add(b, d)));
        }
        const T* t = tw;
        for (size_t e = 2 * w; e < run; e += 2 * w, t += 6 * tl)
        {
            const T* const z0 = in + e;               // bin i
            const T* const z1 = in + 2 * run - e;     // bin m - i
            const T* const z2 = in + 2 * run + e;     // bin m + i
            const T* const z3 = in + 4 * run - e;     // bin 2m - i
            const V pr = O::load(z0), pi = O::load(z0 + w);
            const V sr = O::load(z1), si = O::load(z1 + w);
            const V rr = O::load(z2), ri = O::load(z2 + w);
            const V qr = O::load(z3), qi = O::load(z3 + w);
            // 2 u0 = Z[i] + conj(Z[2m - i]), 2 u2 their difference; 2 u1 =
            // Z[m + i] + conj(Z[m - i]) and 2 u3 = (f, -g).
            const V e0r = O::add(pr, qr), e0i = O::sub(pi, qi);
            const V e2r = O::sub(pr, qr), e2i = O::add(pi, qi);
            const V e1r = O::add(rr, sr), e1i = O::sub(ri, si);
            const V f = O::add(ri, si), g = O::sub(rr, sr);
            const V t1r = O::sub(e2r, f), t1i = O::add(e2i, g);
            const V t2r = O::sub(e0r, e1r), t2i = O::sub(e0i, e1i);
            const V t3r = O::add(e2r, f), t3i = O::sub(e2i, g);
            const V w1r = laneTwiddle<T, W, Whole>(t), w1i = laneTwiddle<T, W, Whole>(t + tl);
            const V w2r = laneTwiddle<T, W, Whole>(t + 2 * tl), w2i = laneTwiddle<T, W, Whole>(t + 3 * tl);
            const V w3r = laneTwiddle<T, W, Whole>(t + 4 * tl), w3i = laneTwiddle<T, W, Whole>(t + 5 * tl);
            O::store(x0 + e, O::add(e0r, e1r));
            O::store(x0 + e + w, O::add(e0i, e1i));
            O::store(x1 + e, O::mulAdd(w1r, t1r, w1i, t1i));
            O::store(x1 + e + w, O::mulSub(w1r, t1i, w1i, t1r));
            O::store(x2 + e, O::mulAdd(w2r, t2r, w2i, t2i));
            O::store(x2 + e + w, O::mulSub(w2r, t2i, w2i, t2r));
            O::store(x3 + e, O::mulAdd(w3r, t3r, w3i, t3i));
            O::store(x3 + e + w, O::mulSub(w3r, t3i, w3i, t3r));
        }
    }
}

/// realFirst4() undone and scaled: the samples from the four-point spectra.
template <typename T, int W>
void realLast4(const T* in, T* out, size_t n, T scale) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    const size_t quarter = w * (n / 4);
    const V c = O::set1(scale), c2 = O::set1(scale + scale);
    for (T* const end = out + quarter; out != end; out += w, in += 4 * w)
    {
        const V z0 = O::load(in), z2 = O::load(in + w);
        const V zr = O::load(in + 2 * w), zi = O::load(in + 3 * w);
        const V s = O::mul(O::add(z0, z2), c), d = O::mul(O::sub(z0, z2), c);
        O::store(out, O::madd(zr, c2, s));
        O::store(out + quarter, O::nmadd(zi, c2, d));
        O::store(out + 2 * quarter, O::nmadd(zr, c2, s));
        O::store(out + 3 * quarter, O::madd(zi, c2, d));
    }
}

/// realFirst8() undone and scaled.
template <typename T, int W>
void realLast8(const T* in, T* out, size_t n, T scale) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    const size_t eighth = w * (n / 8);
    const V c = O::set1(scale), c2 = O::set1(scale + scale);
    const V kc = O::set1(static_cast<T>(1.41421356237309504880) * scale);
    const V nkc = O::set1(static_cast<T>(-1.41421356237309504880) * scale);
    for (T* const end = out + eighth; out != end; out += w, in += 8 * w)
    {
        const V z0 = O::load(in), z4 = O::load(in + w);
        const V pr = O::load(in + 2 * w), pi = O::load(in + 3 * w);
        const V z2r = O::load(in + 4 * w), z2i = O::load(in + 5 * w);
        const V qr = O::load(in + 6 * w), qi = O::load(in + 7 * w);
        // The sums of the sample pairs four apart, from the even bins ...
        const V s = O::mul(O::add(z0, z4), c), d = O::mul(O::sub(z0, z4), c);
        const V s0 = O::madd(z2r, c2, s), s2 = O::nmadd(z2r, c2, s);
        const V s1 = O::nmadd(z2i, c2, d), s3 = O::madd(z2i, c2, d);
        // ... and their differences, from the odd ones.
        const V a = O::add(pr, qr), b = O::sub(pr, qr), g = O::sub(qi, pi), e = O::add(pi, qi);
        const V d0 = O::mul(a, c2), d2 = O::mul(g, c2);
        const V d1 = O::mul(kc, O::sub(b, e)), d3 = O::mul(nkc, O::add(b, e));
        O::store(out, O::add(s0, d0));
        O::store(out + eighth, O::add(s1, d1));
        O::store(out + 2 * eighth, O::add(s2, d2));
        O::store(out + 3 * eighth, O::add(s3, d3));
        O::store(out + 4 * eighth, O::sub(s0, d0));
        O::store(out + 5 * eighth, O::sub(s1, d1));
        O::store(out + 6 * eighth, O::sub(s2, d2));
        O::store(out + 7 * eighth, O::sub(s3, d3));
    }
}

/**
 * @brief The last step of the forward transform: joins the W lanes' spectra
 *        of n points into the spectrum of the N = W n samples, as interleaved
 *        (re, im) bins 0 .. N/2.
 *
 * It is the pass that joins W spectra, with the spectra in the lanes: a_j =
 * w^(ji) Y_j[i] (w = exp(-2 pi i / N)) and A = the W-point transform of a
 * give X[i + r n] = A[r] and X[(r + 1) n - i] = conj(A[W - 1 - r]) for r <
 * W/2. W bins i go at a time: their 2W vectors are transposed, so that each
 * lane's values lie along a vector, and the results leave as W/2 runs of W
 * rising bins and W/2 runs of W falling ones. Bin 0 goes with the others (its
 * imaginary part taken as zero; what it writes twice, it writes equal), bin
 * n/2 on its own after them.
 *
 * @param in   The lanes' packed spectra (n elements).
 * @param out  Receives N + 2 values.
 * @param tw   For every group of W bins and every lane j = 1 .. W - 1, the
 *             real and the imaginary parts of w^(ji) over the group's bins.
 * @param edge cos then sin of pi j (2r + 1) / W at [j W/2 + r].
 */
template <typename T, int W>
void realFinalize(const T* in, T* out, size_t n, const T* tw, const T* edge) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    constexpr size_t half = w / 2;
    T middle[W];
    O::store(middle, O::load(in + w));
    for (size_t i0 = 0; i0 < n / 2; i0 += w, in += 2 * w * w, tw += 2 * (w - 1) * w)
    {
        V re[W], im[W];
        for (size_t t = 0; t < w; ++t)
        {
            re[t] = O::load(in + 2 * t * w);
            im[t] = O::load(in + (2 * t + 1) * w);
        }
        if (i0 == 0) im[0] = O::set1(T(0));   // that element holds bin n/2
        O::transpose(re);
        O::transpose(im);
        for (size_t j = 1; j < w; ++j)
        {
            const V wr = O::load(tw + 2 * (j - 1) * w), wi = O::load(tw + (2 * j - 1) * w);
            const V r = O::mulSub(wr, re[j], wi, im[j]);
            im[j] = O::mulAdd(wr, im[j], wi, re[j]);
            re[j] = r;
        }
        // Rising bins i + r n and falling bins (r + 1) n - i; the falling
        // ones go first, so that bin 0's doubles end as the rising values.
        const auto rising = [&](size_t r) noexcept { return out + 2 * (i0 + r * n); };
        const auto falling = [&](size_t r) noexcept { return out + 2 * ((r + 1) * n - i0 - (w - 1)); };
        if constexpr (W == 2)
        {
            O::storeInterleave2Reversed(falling(0), O::sub(re[0], re[1]), O::sub(im[1], im[0]));
            O::storeInterleave2(rising(0), O::add(re[0], re[1]), O::add(im[0], im[1]));
        }
        else if constexpr (W == 4)
        {
            const V u0r = O::add(re[0], re[2]), u0i = O::add(im[0], im[2]);
            const V u1r = O::sub(re[0], re[2]), u1i = O::sub(im[0], im[2]);
            const V u2r = O::add(re[1], re[3]), u2i = O::add(im[1], im[3]);
            const V u3r = O::sub(re[3], re[1]), u3i = O::sub(im[3], im[1]);
            O::storeInterleave2Reversed(falling(0), O::add(u1r, u3i), O::sub(u3r, u1i));
            O::storeInterleave2Reversed(falling(1), O::sub(u0r, u2r), O::sub(u2i, u0i));
            O::storeInterleave2(rising(0), O::add(u0r, u2r), O::add(u0i, u2i));
            O::storeInterleave2(rising(1), O::sub(u1r, u3i), O::add(u1i, u3r));
        }
        else
        {
            // Eight lanes: the even outputs are the four-point transform of
            // p_j = a_j + a_(j+4), the odd ones that of q_j = a_j - a_(j+4)
            // turned by w8^j.
            const V h = O::set1(static_cast<T>(0.70710678118654752440));
            const V nh = O::set1(static_cast<T>(-0.70710678118654752440));
            const V p0r = O::add(re[0], re[4]), p0i = O::add(im[0], im[4]);
            const V p1r = O::add(re[1], re[5]), p1i = O::add(im[1], im[5]);
            const V p2r = O::add(re[2], re[6]), p2i = O::add(im[2], im[6]);
            const V p3r = O::add(re[3], re[7]), p3i = O::add(im[3], im[7]);
            const V q0r = O::sub(re[0], re[4]), q0i = O::sub(im[0], im[4]);
            const V q1r = O::sub(re[1], re[5]), q1i = O::sub(im[1], im[5]);
            const V q2r = O::sub(re[2], re[6]), q2i = O::sub(im[2], im[6]);
            const V q3r = O::sub(re[3], re[7]), q3i = O::sub(im[3], im[7]);
            const V u0r = O::add(p0r, p2r), u0i = O::add(p0i, p2i);
            const V u1r = O::sub(p0r, p2r), u1i = O::sub(p0i, p2i);
            const V u2r = O::add(p1r, p3r), u2i = O::add(p1i, p3i);
            const V u3r = O::sub(p3r, p1r), u3i = O::sub(p3i, p1i);
            const V v0r = O::add(q0r, q2i), v0i = O::sub(q0i, q2r);
            const V v1r = O::sub(q0r, q2i), v1i = O::add(q0i, q2r);
            const V m1 = O::sub(q1r, q3r), m2 = O::add(q1i, q3i);
            const V m3 = O::sub(q1i, q3i), m4 = O::add(q1r, q3r);
            const V k1 = O::mul(h, O::add(m1, m2)), k2 = O::mul(h, O::sub(m3, m4));
            const V k3 = O::mul(nh, O::add(m3, m4)), k4 = O::mul(h, O::sub(m1, m2));
            O::storeInterleave2Reversed(falling(0), O::add(v1r, k4), O::sub(k3, v1i));   // conj(A[7])
            O::storeInterleave2Reversed(falling(1), O::add(u1r, u3i), O::sub(u3r, u1i)); // conj(A[6])
            O::storeInterleave2Reversed(falling(2), O::sub(v0r, k1), O::sub(k2, v0i));   // conj(A[5])
            O::storeInterleave2Reversed(falling(3), O::sub(u0r, u2r), O::sub(u2i, u0i)); // conj(A[4])
            O::storeInterleave2(rising(0), O::add(u0r, u2r), O::add(u0i, u2i));           // A[0]
            O::storeInterleave2(rising(1), O::add(v0r, k1), O::add(v0i, k2));             // A[1]
            O::storeInterleave2(rising(2), O::sub(u1r, u3i), O::add(u1i, u3r));           // A[2]
            O::storeInterleave2(rising(3), O::sub(v1r, k4), O::add(v1i, k3));             // A[3]
        }
    }
    // Bin n/2 of the lanes is real: X[n/2 + r n] = sum_j Y_j[n/2] exp(-pi i j (2r + 1) / W).
    for (size_t r = 0; r < half; ++r)
    {
        T sumRe = T(0), sumIm = T(0);
        for (size_t j = 0; j < w; ++j)
        {
            sumRe += middle[j] * edge[j * half + r];
            sumIm -= middle[j] * edge[w * half + j * half + r];
        }
        out[2 * (n / 2 + r * n)] = sumRe;
        out[2 * (n / 2 + r * n) + 1] = sumIm;
    }
    out[1] = T(0);
    out[w * n + 1] = T(0);
}

/**
 * @brief realFinalize() undone: the lanes' packed spectra, W times too large,
 *        from the N/2 + 1 interleaved bins.
 *
 * A[r] = X[i + r n] and A[W - 1 - r] = conj(X[(r + 1) n - i]) are the
 * W-point transform that a_j = sum_r A[r] exp(2 pi i j r / W) undoes, and
 * Y_j[i] = conj(w^(ji)) a_j. The imaginary parts of bins 0 and N/2 are not
 * read (a real signal has none).
 */
template <typename T, int W>
void realPrepare(const T* in, T* out, size_t n, const T* tw, const T* edge) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    constexpr size_t w = static_cast<size_t>(W);
    constexpr size_t half = w / 2;
    T* const first = out;
    for (size_t i0 = 0; i0 < n / 2; i0 += w, out += 2 * w * w, tw += 2 * (w - 1) * w)
    {
        V dr[W / 2], di[W / 2], mr[W / 2], mi[W / 2];   // rising and falling bins
        for (size_t r = 0; r < half; ++r)
        {
            O::loadDeinterleave(in + 2 * (i0 + r * n), dr[r], di[r]);
            O::loadDeinterleaveReversed(in + 2 * ((r + 1) * n - i0 - (w - 1)), mr[r], mi[r]);
        }
        if (i0 == 0)
        {
            // Bins 0 and N/2 with their imaginary parts as zero.
            T head[2 * W], tail[2 * W];
            std::copy_n(in, 2 * w, head);
            head[1] = T(0);
            std::copy_n(in + 2 * (half * n - (w - 1)), 2 * w, tail);
            tail[2 * w - 1] = T(0);
            O::loadDeinterleave(head, dr[0], di[0]);
            O::loadDeinterleaveReversed(tail, mr[half - 1], mi[half - 1]);
        }
        V re[W], im[W];
        if constexpr (W == 2)
        {
            re[0] = O::add(dr[0], mr[0]);
            im[0] = O::sub(di[0], mi[0]);
            re[1] = O::sub(dr[0], mr[0]);
            im[1] = O::add(di[0], mi[0]);
        }
        else if constexpr (W == 4)
        {
            // A[0] = X[i], A[1] = X[n + i], A[2] = conj(X[2n - i]), A[3] = conj(X[n - i]).
            const V e0r = O::add(dr[0], mr[1]), e0i = O::sub(di[0], mi[1]);
            const V e1r = O::sub(dr[0], mr[1]), e1i = O::add(di[0], mi[1]);
            const V e2r = O::add(dr[1], mr[0]), e2i = O::sub(di[1], mi[0]);
            const V e3r = O::sub(dr[1], mr[0]), e3i = O::add(di[1], mi[0]);
            re[0] = O::add(e0r, e2r);
            im[0] = O::add(e0i, e2i);
            re[1] = O::sub(e1r, e3i);
            im[1] = O::add(e1i, e3r);
            re[2] = O::sub(e0r, e2r);
            im[2] = O::sub(e0i, e2i);
            re[3] = O::add(e1r, e3i);
            im[3] = O::sub(e1i, e3r);
        }
        else
        {
            // Even A: X[i], X[2n + i], conj(X[4n - i]), conj(X[2n - i]); odd
            // A: X[n + i], X[3n + i], conj(X[3n - i]), conj(X[n - i]). a_j =
            // P_j + conj(w8^j) Q_j and a_(j+4) = P_j - conj(w8^j) Q_j, P and
            // Q their four-point inverse transforms.
            const V h = O::set1(static_cast<T>(0.70710678118654752440));
            const V nh = O::set1(static_cast<T>(-0.70710678118654752440));
            const V e0r = O::add(dr[0], mr[3]), e0i = O::sub(di[0], mi[3]);
            const V e1r = O::sub(dr[0], mr[3]), e1i = O::add(di[0], mi[3]);
            const V e2r = O::add(dr[2], mr[1]), e2i = O::sub(di[2], mi[1]);
            const V e3r = O::sub(dr[2], mr[1]), e3i = O::add(di[2], mi[1]);
            const V f0r = O::add(dr[1], mr[2]), f0i = O::sub(di[1], mi[2]);
            const V f1r = O::sub(dr[1], mr[2]), f1i = O::add(di[1], mi[2]);
            const V f2r = O::add(dr[3], mr[0]), f2i = O::sub(di[3], mi[0]);
            const V f3r = O::sub(dr[3], mr[0]), f3i = O::add(di[3], mi[0]);
            const V p0r = O::add(e0r, e2r), p0i = O::add(e0i, e2i);
            const V p1r = O::sub(e1r, e3i), p1i = O::add(e1i, e3r);
            const V p2r = O::sub(e0r, e2r), p2i = O::sub(e0i, e2i);
            const V p3r = O::add(e1r, e3i), p3i = O::sub(e1i, e3r);
            const V q0r = O::add(f0r, f2r), q0i = O::add(f0i, f2i);
            const V q1r = O::sub(f1r, f3i), q1i = O::add(f1i, f3r);
            const V q2r = O::sub(f0r, f2r), q2i = O::sub(f0i, f2i);
            const V q3r = O::add(f1r, f3i), q3i = O::sub(f1i, f3r);
            const V t1r = O::mul(h, O::sub(q1r, q1i)), t1i = O::mul(h, O::add(q1r, q1i));
            const V t3r = O::mul(nh, O::add(q3r, q3i)), t3i = O::mul(h, O::sub(q3r, q3i));
            re[0] = O::add(p0r, q0r);
            im[0] = O::add(p0i, q0i);
            re[4] = O::sub(p0r, q0r);
            im[4] = O::sub(p0i, q0i);
            re[1] = O::add(p1r, t1r);
            im[1] = O::add(p1i, t1i);
            re[5] = O::sub(p1r, t1r);
            im[5] = O::sub(p1i, t1i);
            re[2] = O::sub(p2r, q2i);
            im[2] = O::add(p2i, q2r);
            re[6] = O::add(p2r, q2i);
            im[6] = O::sub(p2i, q2r);
            re[3] = O::add(p3r, t3r);
            im[3] = O::add(p3i, t3i);
            re[7] = O::sub(p3r, t3r);
            im[7] = O::sub(p3i, t3i);
        }
        for (size_t j = 1; j < w; ++j)
        {
            const V wr = O::load(tw + 2 * (j - 1) * w), wi = O::load(tw + (2 * j - 1) * w);
            const V r = O::mulAdd(wr, re[j], wi, im[j]);
            im[j] = O::mulSub(wr, im[j], wi, re[j]);
            re[j] = r;
        }
        O::transpose(re);
        O::transpose(im);
        for (size_t t = 0; t < w; ++t)
        {
            O::store(out + 2 * t * w, re[t]);
            O::store(out + (2 * t + 1) * w, im[t]);
        }
    }
    // Bin n/2 of the lanes: W Y_j[n/2] = 2 Re sum_r X[n/2 + r n] exp(pi i j (2r + 1) / W).
    T middle[W];
    for (size_t j = 0; j < w; ++j)
    {
        T sum = T(0);
        for (size_t r = 0; r < half; ++r)
        {
            const T* const x = in + 2 * (n / 2 + r * n);
            sum += x[0] * edge[j * half + r] - x[1] * edge[w * half + j * half + r];
        }
        middle[j] = sum + sum;
    }
    O::store(first + w, O::load(middle));
}

/// The forward transform at W lanes: the passes between the two work buffers
/// a and b (N values each), the first one reading the samples.
template <typename T, int W, bool Whole>
void realForwardAt(const RealPlan<T>& p, const T* time, T* freq, T* a, T* b) noexcept
{
    const size_t n = p.n;
    if (p.first8)
        realFirst8<T, W>(time, a, n);
    else
        realFirst4<T, W>(time, a, n);
    const T* tw = p.twiddles;
    T* src = a;
    T* dst = b;
    for (size_t m = p.firstLength(); m < n; m *= 4)
    {
        realRadix4<T, W, Whole>(src, dst, n, m, tw);
        tw += RealPlan<T>::passValues(m, p.twiddleRepeat);
        std::swap(src, dst);
    }
    realFinalize<T, W>(src, freq, n, p.turn, p.edge);
}

/// One step of the inverse at W lanes: step 0 reads the spectrum into a,
/// each later step is one pass (they alternate between a and b, so the
/// buffer a step reads follows from its index) and the last one writes the
/// scaled samples.
template <typename T, int W, bool Whole>
void realInverseStepAt(const RealPlan<T>& p, size_t step, const T* freq, T* time, T* a, T* b) noexcept
{
    const size_t n = p.n;
    if (step == 0)
    {
        realPrepare<T, W>(freq, a, n, p.turn, p.edge);
        return;
    }
    const size_t pass = step - 1;
    const T* const src = (pass & 1) == 0 ? a : b;
    T* const dst = (pass & 1) == 0 ? b : a;
    const size_t m = (n / 4) >> (2 * pass);   // points of the spectra this pass leaves
    if (m >= p.firstLength())
    {
        const T* tw = p.twiddles;
        for (size_t below = p.firstLength(); below < m; below *= 4)
            tw += RealPlan<T>::passValues(below, p.twiddleRepeat);
        realInverseRadix4<T, W, Whole>(src, dst, n, m, tw);
    }
    else if (p.first8)
        realLast8<T, W>(src, time, n, p.scale);
    else
        realLast4<T, W>(src, time, n, p.scale);
}

/// The forward real transform of a plan, at the plan's lanes (FFTReal).
template <typename T>
void realForward([[maybe_unused]] const RealPlan<T>& p, [[maybe_unused]] const T* time,
                 [[maybe_unused]] T* freq, [[maybe_unused]] T* a, [[maybe_unused]] T* b) noexcept
{
    if constexpr (kNarrow<T> > 1)
    {
        const bool whole = p.twiddleRepeat > 1;
        if (p.lanes == static_cast<size_t>(kWide<T>))
        {
            if (whole) realForwardAt<T, kWide<T>, true>(p, time, freq, a, b);
            else realForwardAt<T, kWide<T>, false>(p, time, freq, a, b);
        }
        else
        {
            if (whole) realForwardAt<T, kNarrow<T>, true>(p, time, freq, a, b);
            else realForwardAt<T, kNarrow<T>, false>(p, time, freq, a, b);
        }
    }
}

/// One step of the inverse real transform of a plan (FFTReal::inverseStep()).
template <typename T>
void realInverseStep([[maybe_unused]] const RealPlan<T>& p, [[maybe_unused]] size_t step,
                     [[maybe_unused]] const T* freq, [[maybe_unused]] T* time, [[maybe_unused]] T* a,
                     [[maybe_unused]] T* b) noexcept
{
    if constexpr (kNarrow<T> > 1)
    {
        const bool whole = p.twiddleRepeat > 1;
        if (p.lanes == static_cast<size_t>(kWide<T>))
        {
            if (whole) realInverseStepAt<T, kWide<T>, true>(p, step, freq, time, a, b);
            else realInverseStepAt<T, kWide<T>, false>(p, step, freq, time, a, b);
        }
        else
        {
            if (whole) realInverseStepAt<T, kNarrow<T>, true>(p, step, freq, time, a, b);
            else realInverseStepAt<T, kNarrow<T>, false>(p, step, freq, time, a, b);
        }
    }
}

/// The inverse real transform of a plan: its steps in order.
template <typename T>
void realInverse(const RealPlan<T>& p, const T* freq, T* time, T* a, T* b) noexcept
{
    const size_t steps = RealPlan<T>::passCount(p.n) + 1;
    for (size_t step = 0; step < steps; ++step)
        realInverseStep(p, step, freq, time, a, b);
}
