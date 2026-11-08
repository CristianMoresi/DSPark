// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

// FftKernels.inl - the FFT passes written on the vector abstraction, included
// once per instruction set into its own namespace (no include guard on
// purpose). FFT.h includes it into dspark::detail::fft for the baseline and,
// in x86-64 builds below AVX, again into dspark::detail::fft::avx2 inside an
// AVX2+FMA target region. The includer first declares Vec, kWide and kNarrow
// for the copy; TwiddleSpan, Pass and Buffers come from the enclosing
// namespace. This file includes nothing.

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
 * @brief One Stockham radix-4 DIF pass with stride s >= W, vectorised over
 *        the stride (twiddles broadcast per butterfly group).
 *
 * y[q + s(4p + k)] = w^(kp) * (radix-4 butterfly of x[q + s(p + k n/4)]).
 */
template <typename T, int W>
void radix4Strided(const T* xr, const T* xi, T* yr, T* yi,
                   size_t n, size_t s, const TwiddleSpan<T>& tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    const size_t n1 = n / 4;
    for (size_t p = 0; p < n1; ++p)
    {
        const V w1r = O::set1(tw.w1r[p]), w1i = O::set1(tw.w1i[p]);
        const V w2r = O::set1(tw.w2r[p]), w2i = O::set1(tw.w2i[p]);
        const V w3r = O::set1(tw.w3r[p]), w3i = O::set1(tw.w3i[p]);
        const T* ar = xr + s * p;            const T* ai = xi + s * p;
        const T* br = xr + s * (p + n1);     const T* bi = xi + s * (p + n1);
        const T* cr = xr + s * (p + 2 * n1); const T* ci = xi + s * (p + 2 * n1);
        const T* dr = xr + s * (p + 3 * n1); const T* di = xi + s * (p + 3 * n1);
        T* y0r = yr + s * (4 * p);     T* y0i = yi + s * (4 * p);
        T* y1r = y0r + s;              T* y1i = y0i + s;
        T* y2r = y0r + 2 * s;          T* y2i = y0i + 2 * s;
        T* y3r = y0r + 3 * s;          T* y3i = y0i + 3 * s;
        for (size_t q = 0; q < s; q += W)
        {
            const V a_r = O::load(ar + q), a_i = O::load(ai + q);
            const V b_r = O::load(br + q), b_i = O::load(bi + q);
            const V c_r = O::load(cr + q), c_i = O::load(ci + q);
            const V d_r = O::load(dr + q), d_i = O::load(di + q);
            const V apcR = O::add(a_r, c_r), apcI = O::add(a_i, c_i);
            const V amcR = O::sub(a_r, c_r), amcI = O::sub(a_i, c_i);
            const V bpdR = O::add(b_r, d_r), bpdI = O::add(b_i, d_i);
            const V bmdR = O::sub(b_r, d_r), bmdI = O::sub(b_i, d_i);
            // t1 = amc - j bmd, t2 = apc - bpd, t3 = amc + j bmd
            const V t1R = O::add(amcR, bmdI), t1I = O::sub(amcI, bmdR);
            const V t2R = O::sub(apcR, bpdR), t2I = O::sub(apcI, bpdI);
            const V t3R = O::sub(amcR, bmdI), t3I = O::add(amcI, bmdR);
            O::store(y0r + q, O::add(apcR, bpdR));
            O::store(y0i + q, O::add(apcI, bpdI));
            O::store(y1r + q, O::mulSub(t1R, w1r, t1I, w1i));
            O::store(y1i + q, O::mulAdd(t1R, w1i, t1I, w1r));
            O::store(y2r + q, O::mulSub(t2R, w2r, t2I, w2i));
            O::store(y2i + q, O::mulAdd(t2R, w2i, t2I, w2r));
            O::store(y3r + q, O::mulSub(t3R, w3r, t3I, w3i));
            O::store(y3i + q, O::mulAdd(t3R, w3i, t3I, w3r));
        }
    }
}

/**
 * @brief The first (stride-1) radix-4 pass, vectorised across W butterflies
 *        (n/4 must be a multiple of W): the inputs and twiddles are
 *        contiguous in p, and the four outputs of each butterfly are adjacent,
 *        written through a 4-way interleave.
 */
template <typename T, int W, typename Source>
void radix4First(const Source& input, T* yr, T* yi,
                 size_t n, const TwiddleSpan<T>& tw) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
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
        O::storeInterleave4(yr + 4 * p, O::add(apcR, bpdR),
                            O::mulSub(t1R, w1r, t1I, w1i),
                            O::mulSub(t2R, w2r, t2I, w2i),
                            O::mulSub(t3R, w3r, t3I, w3i));
        O::storeInterleave4(yi + 4 * p, O::add(apcI, bpdI),
                            O::mulAdd(t1R, w1i, t1I, w1r),
                            O::mulAdd(t2R, w2i, t2I, w2r),
                            O::mulAdd(t3R, w3i, t3I, w3r));
    }
}

/// The closing radix-4 pass (n = 4, stride s = N / 4): its only twiddles are 1.
template <typename T, int W, typename Sink>
void radix4Last(const T* xr, const T* xi, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    using V = typename O::V;
    for (size_t q = 0; q < s; q += W)
    {
        const V a_r = O::load(xr + q),         a_i = O::load(xi + q);
        const V b_r = O::load(xr + q + s),     b_i = O::load(xi + q + s);
        const V c_r = O::load(xr + q + 2 * s), c_i = O::load(xi + q + 2 * s);
        const V d_r = O::load(xr + q + 3 * s), d_i = O::load(xi + q + 3 * s);
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
template <typename T, int W, typename Sink>
void radix2Last(const T* xr, const T* xi, const Sink& output, size_t s) noexcept
{
    using O = Vec<T, W>;
    for (size_t q = 0; q < s; q += W)
    {
        const auto a_r = O::load(xr + q), a_i = O::load(xi + q);
        const auto b_r = O::load(xr + q + s), b_i = O::load(xi + q + s);
        output.write(q, O::add(a_r, b_r), O::add(a_i, b_i));
        output.write(q + s, O::sub(a_r, b_r), O::sub(a_i, b_i));
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

/// Runs the first pass at the widest vector its length allows.
template <typename T, template <typename, int> class Source, typename... Args>
void firstPass(const Pass& pass, const T* twiddles, T* yr, T* yi, Args... input) noexcept
{
    const size_t n1 = pass.len / 4;
    const TwiddleSpan<T> tw = twiddleSpanOf(pass, twiddles);
    constexpr int wide = kWide<T>;
    constexpr int narrow = kNarrow<T>;
    if (n1 % wide == 0)
        radix4First<T, wide>(Source<T, wide> { input... }, yr, yi, pass.len, tw);
    else if (n1 % narrow == 0)
        radix4First<T, narrow>(Source<T, narrow> { input... }, yr, yi, pass.len, tw);
    else
        radix4First<T, 1>(Source<T, 1> { input... }, yr, yi, pass.len, tw);
}

/// Runs a radix-4 pass of stride > 1 at the widest vector its stride allows.
template <typename T>
void stridedPass(const Pass& pass, const T* twiddles, const Buffers<T>& b) noexcept
{
    const TwiddleSpan<T> tw = twiddleSpanOf(pass, twiddles);
    constexpr int wide = kWide<T>;
    constexpr int narrow = kNarrow<T>;
    if (pass.stride % wide == 0)
        radix4Strided<T, wide>(b.xr, b.xi, b.yr, b.yi, pass.len, pass.stride, tw);
    else if (pass.stride % narrow == 0)
        radix4Strided<T, narrow>(b.xr, b.xi, b.yr, b.yi, pass.len, pass.stride, tw);
    else
        radix4Strided<T, 1>(b.xr, b.xi, b.yr, b.yi, pass.len, pass.stride, tw);
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
void closingAt(const Pass& pass, const T* xr, const T* xi, Args... output) noexcept
{
    const Sink<T, W> sink = makeSink<T, W, Sink>(output...);
    if (pass.radix2)
        radix2Last<T, W>(xr, xi, sink, pass.stride);
    else
        radix4Last<T, W>(xr, xi, sink, pass.stride);
}

/// Runs a pass that needs no twiddles, at the widest vector its stride allows.
template <typename T, template <typename, int> class Sink, typename... Args>
void closingPass(const Pass& pass, const T* xr, const T* xi, Args... output) noexcept
{
    constexpr int wide = kWide<T>;
    constexpr int narrow = kNarrow<T>;
    if (pass.stride % wide == 0)
        closingAt<T, wide, Sink>(pass, xr, xi, output...);
    else if (pass.stride % narrow == 0)
        closingAt<T, narrow, Sink>(pass, xr, xi, output...);
    else
        closingAt<T, 1, Sink>(pass, xr, xi, output...);
}

/// Runs passes [begin, end) between the two work buffers.
template <typename T>
void runPasses(const Pass* passes, size_t begin, size_t end, const T* twiddles,
               Buffers<T>& b) noexcept
{
    for (size_t i = begin; i < end; ++i)
    {
        const Pass& pass = passes[i];
        if (pass.radix2 || pass.len == 4)
            closingPass<T, SplitSink>(pass, b.xr, b.xi, b.yr, b.yi);
        else if (pass.stride == 1)
            firstPass<T, SplitSource>(pass, twiddles, b.yr, b.yi, b.xr, b.xi);
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
        firstPass<T, ConjugatedSource>(passes[0], twiddles, b.yr, b.yi, data);
    else
        firstPass<T, InterleavedSource>(passes[0], twiddles, b.yr, b.yi, data);
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
    firstPass<T, SplitSource>(passes[0], twiddles, b.yr, b.yi, re, im);
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
    firstPass<T, ScaledSwapSource>(passes[0], twiddles, b.yr, b.yi, re, im, gain, mask);
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
    firstPass<T, FoldedScaledSwapSource>(passes[0], twiddles, b.yr, b.yi, re, im, gain, n, fold);
    T* r = b.xr; b.xr = b.yr; b.yr = r;
    T* m = b.xi; b.xi = b.yi; b.yi = m;
    return 1;
}

/// Runs the closing pass with scaled interleaved output.
template <typename T>
void closeInterleaved(const Pass& pass, const T* xr, const T* xi, T* out, T scaleRe,
                      T scaleIm) noexcept
{
    closingPass<T, InterleavedSink>(pass, xr, xi, out, scaleRe, scaleIm);
}

/**
 * @brief Splits the half-size spectrum Z into the real spectrum X:
 *        X[k] = (Z[k] + conj(Z[M-k])) / 2 - j w^k (Z[k] - conj(Z[M-k])) / 2.
 */
template <typename T>
void unpackRealForward(const T* zr, const T* zi, T* out, size_t m, const T* twRe,
                       const T* twIm) noexcept
{
    constexpr int W = kWide<T>;
    using O = Vec<T, W>;

    const T dc = zr[0] + zi[0];
    const T ny = zr[0] - zi[0];

    const auto half = O::set1(T(0.5));
    size_t k = 1;
    for (; k + W <= m; k += W)   // k .. k+W-1 against M-k .. M-k-W+1
    {
        const auto hkR = O::load(zr + k), hkI = O::load(zi + k);
        const auto hcR = O::reverse(O::load(zr + (m - k - (W - 1))));
        const auto hcI = O::reverse(O::load(zi + (m - k - (W - 1))));
        const auto xeR = O::mul(half, O::add(hkR, hcR));
        const auto xeI = O::mul(half, O::sub(hkI, hcI));
        const auto xoR = O::mul(half, O::sub(hkR, hcR));
        const auto xoI = O::mul(half, O::add(hkI, hcI));
        // -j * xo = (xoI, -xoR), times w = (wr, wi)
        const auto wr = O::load(twRe + k), wi = O::load(twIm + k);
        const auto tR = O::mulAdd(wr, xoI, wi, xoR);   // wr*xoI - wi*(-xoR)
        const auto tI = O::mulSub(wi, xoI, wr, xoR);   // wr*(-xoR) + wi*xoI
        O::storeInterleave2(out + 2 * k, O::add(xeR, tR), O::add(xeI, tI));
    }
    for (; k < m; ++k)
    {
        const size_t c = m - k;
        const T xeR = T(0.5) * (zr[k] + zr[c]);
        const T xeI = T(0.5) * (zi[k] - zi[c]);
        const T xoR = T(0.5) * (zr[k] - zr[c]);
        const T xoI = T(0.5) * (zi[k] + zi[c]);
        const T wr = twRe[k], wi = twIm[k];
        out[2 * k]     = xeR + (wr * xoI + wi * xoR);
        out[2 * k + 1] = xeI + (wi * xoI - wr * xoR);
    }
    // Written last: in-place use shares the buffer with the input pairs,
    // which the engine has already consumed.
    out[0] = dc;
    out[1] = T(0);
    out[2 * m] = ny;
    out[2 * m + 1] = T(0);
}

/**
 * @brief Packs the real spectrum X into the CONJUGATED half-size spectrum
 *        conj(Z), Z[k] = Xe[k] + j conj(w^k) Xd[k] (see unpackRealForward()).
 */
template <typename T>
void packRealInverseConj(const T* in, T* zr, T* zi, size_t m, const T* twRe,
                         const T* twIm) noexcept
{
    constexpr int W = kWide<T>;
    using O = Vec<T, W>;

    const T dc = in[0];
    const T ny = in[2 * m];
    zr[0] = T(0.5) * (dc + ny);
    zi[0] = -T(0.5) * (dc - ny);

    const auto half = O::set1(T(0.5));
    const auto zero = O::set1(T(0));
    size_t k = 1;
    for (; k + W <= m; k += W)
    {
        typename O::V xkR, xkI, xcR, xcI;
        O::loadDeinterleave(in + 2 * k, xkR, xkI);
        O::loadDeinterleave(in + 2 * (m - k - (W - 1)), xcR, xcI);
        xcR = O::reverse(xcR);
        xcI = O::reverse(xcI);
        const auto xeR = O::mul(half, O::add(xkR, xcR));
        const auto xeI = O::mul(half, O::sub(xkI, xcI));
        const auto dR = O::mul(half, O::sub(xkR, xcR));
        const auto dI = O::mul(half, O::add(xkI, xcI));
        // conj(w) * d, then j * that: (-(tI), tR)
        const auto wr = O::load(twRe + k), wi = O::load(twIm + k);
        const auto tR = O::mulAdd(wr, dR, wi, dI);    // wr*dR - (-wi)*dI
        const auto tI = O::mulSub(wr, dI, wi, dR);    // wr*dI + (-wi)*dR
        O::store(zr + k, O::sub(xeR, tI));
        O::store(zi + k, O::sub(zero, O::add(xeI, tR)));   // conjugated
    }
    for (; k < m; ++k)
    {
        const size_t c = m - k;
        const T xeR = T(0.5) * (in[2 * k] + in[2 * c]);
        const T xeI = T(0.5) * (in[2 * k + 1] - in[2 * c + 1]);
        const T dR = T(0.5) * (in[2 * k] - in[2 * c]);
        const T dI = T(0.5) * (in[2 * k + 1] + in[2 * c + 1]);
        const T wr = twRe[k], wi = twIm[k];
        const T tR = wr * dR + wi * dI;
        const T tI = wr * dI - wi * dR;
        zr[k] = xeR - tI;
        zi[k] = -(xeI + tR);
    }
}
