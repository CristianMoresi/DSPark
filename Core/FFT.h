// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file FFT.h
 * @brief Fast Fourier Transform: Stockham radix-4 on split complex data, SIMD.
 *
 * Provides forward and inverse FFT for both complex and real-valued signals,
 * for power-of-two sizes.
 *
 * Engine:
 * - **Stockham autosort, radix 4**: log4(N) passes (plus one radix-2 pass
 *   when log2(N) is odd) with no bit-reversal permutation; each pass reads
 *   one buffer and writes the other in natural order.
 * - **Split complex layout inside**: the real and imaginary parts live in
 *   separate arrays while the passes run, so every butterfly is plain vector
 *   arithmetic (no shuffles in the complex multiplies). The interleaved API
 *   layout is converted once on the way in and once on the way out.
 * - **SIMD at the widest width a pass allows**: AVX (8 float / 4 double
 *   lanes, with FMA when enabled) where the build targets it, SSE2 on every
 *   x86-64, NEON on ARM64, scalar otherwise. The first pass, whose stride is
 *   1, runs across butterflies and writes its outputs through an in-register
 *   4-way interleave.
 * - **Twiddles in double**: every factor is computed directly with cos/sin in
 *   double precision (no recurrences), so the error stays at the level of the
 *   arithmetic itself.
 * - **Inverse by conjugation**: ifft(x) = conj(fft(conj(x))) / N, folded into
 *   the conversion passes, so both directions share one set of kernels.
 * - **Real transforms** run a half-size complex FFT on the (even, odd) sample
 *   pairs and split its spectrum with a vectorised post-pass.
 * - **Zero allocations** after construction; forward()/inverse() are
 *   noexcept and accept pointers of any alignment.
 *
 * Measured on one core of a 2.1 GHz Xeon, float: a 1024-point real FFT takes
 * about 2.0 us on the SSE2 baseline and 1.0 us with AVX2/FMA enabled
 * (-march=x86-64-v3 or /arch:AVX2), against 3.9 us and 4.6 us for the
 * radix-2 engine this replaced; sizes past the L2 cache gain about 2.5x.
 *
 * Threading: an instance keeps work buffers, so one instance serves one
 * thread at a time; give each thread its own instance.
 *
 * Dependencies: SimdOps.h (platform SIMD detection and the vector abstraction).
 */

#include "SimdOps.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

// --- Exception policy --------------------------------------------------------
// By default invalid FFT sizes throw std::invalid_argument. For embedded and
// plugin builds compiled without exception support, define DSPARK_NO_EXCEPTIONS
// (auto-detected from -fno-exceptions / /EHs-c-) -- invalid sizes then assert in
// debug and degrade to a safe minimal size in release instead of throwing.
#if !defined(DSPARK_NO_EXCEPTIONS)
  #if (defined(__GNUC__) || defined(__clang__)) && !defined(__EXCEPTIONS)
    #define DSPARK_NO_EXCEPTIONS 1
  #elif defined(_MSC_VER) && !defined(_CPPUNWIND)
    #define DSPARK_NO_EXCEPTIONS 1
  #endif
#endif

#if !defined(DSPARK_NO_EXCEPTIONS)
  #include <stdexcept>
#endif

namespace dspark {
namespace detail {
namespace fft {

/// The shared SIMD vector abstraction (SimdOps.h) and its widths.
template <typename T, int W> using Vec = simd::Vec<T, W>;
template <typename T> inline constexpr int kWide = simd::kVecWidth<T>;
template <typename T> inline constexpr int kNarrow = simd::kVecNarrowWidth<T>;

// ============================================================================
// Kernels
// ============================================================================

/// Twiddles of one radix-4 pass, split by factor: w1, w2, w3 (re, im) for
/// p = 0 .. n/4 - 1.
template <typename T>
struct TwiddleSpan
{
    const T* w1r; const T* w1i;
    const T* w2r; const T* w2i;
    const T* w3r; const T* w3i;
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
template <typename T, int W>
void radix4First(const T* xr, const T* xi, T* yr, T* yi,
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
        const V a_r = O::load(xr + p),          a_i = O::load(xi + p);
        const V b_r = O::load(xr + p + n1),     b_i = O::load(xi + p + n1);
        const V c_r = O::load(xr + p + 2 * n1), c_i = O::load(xi + p + 2 * n1);
        const V d_r = O::load(xr + p + 3 * n1), d_i = O::load(xi + p + 3 * n1);
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

/// The closing radix-2 pass (n = 2, stride s = N / 2): no twiddles.
template <typename T, int W>
void radix2Last(const T* xr, const T* xi, T* yr, T* yi, size_t s) noexcept
{
    using O = Vec<T, W>;
    for (size_t q = 0; q < s; q += W)
    {
        const auto a_r = O::load(xr + q), a_i = O::load(xi + q);
        const auto b_r = O::load(xr + q + s), b_i = O::load(xi + q + s);
        O::store(yr + q, O::add(a_r, b_r));
        O::store(yi + q, O::add(a_i, b_i));
        O::store(yr + q + s, O::sub(a_r, b_r));
        O::store(yi + q + s, O::sub(a_i, b_i));
    }
}

/**
 * @class SplitFFT
 * @brief The shared Stockham engine: a forward complex FFT of size N on
 *        split (re, im) arrays, ping-ponging between two work buffers.
 */
template <typename T>
class SplitFFT
{
public:
    explicit SplitFFT(size_t n) : n_(n)
    {
        bufA_.assign(2 * n_, T(0));
        bufB_.assign(2 * n_, T(0));

        // Plan: radix-4 passes of length n, 4n', ... with stride 1, 4, ...;
        // a radix-2 pass closes an odd log2(N).
        size_t len = n_, stride = 1, offset = 0;
        while (len >= 4)
        {
            passes_.push_back({ len, stride, offset, false });
            offset += 6 * (len / 4);
            len /= 4;
            stride *= 4;
        }
        if (len == 2)
            passes_.push_back({ 2, stride, offset, true });

        twiddles_.assign(offset, T(0));
        for (const Pass& pass : passes_)
        {
            if (pass.radix2) continue;
            const size_t n1 = pass.len / 4;
            T* w = twiddles_.data() + pass.twOffset;
            for (size_t p = 0; p < n1; ++p)
            {
                const double a = -2.0 * std::numbers::pi_v<double> * static_cast<double>(p)
                               / static_cast<double>(pass.len);
                w[p]          = static_cast<T>(std::cos(a));
                w[n1 + p]     = static_cast<T>(std::sin(a));
                w[2 * n1 + p] = static_cast<T>(std::cos(2.0 * a));
                w[3 * n1 + p] = static_cast<T>(std::sin(2.0 * a));
                w[4 * n1 + p] = static_cast<T>(std::cos(3.0 * a));
                w[5 * n1 + p] = static_cast<T>(std::sin(3.0 * a));
            }
        }
    }

    [[nodiscard]] size_t size() const noexcept { return n_; }

    /// Input buffers: fill inRe() / inIm() with N values each before run().
    [[nodiscard]] T* inRe() noexcept { return bufA_.data(); }
    [[nodiscard]] T* inIm() noexcept { return bufA_.data() + n_; }

    /**
     * @brief Runs the forward transform of the input buffers.
     * @param outRe Receives the pointer to the N output real parts.
     * @param outIm Receives the pointer to the N output imaginary parts.
     */
    void run(const T*& outRe, const T*& outIm) noexcept
    {
        T* xr = bufA_.data(); T* xi = xr + n_;
        T* yr = bufB_.data(); T* yi = yr + n_;
        for (const Pass& pass : passes_)
        {
            if (pass.radix2)
                runRadix2(xr, xi, yr, yi, pass.stride);
            else
                runRadix4(xr, xi, yr, yi, pass);
            std::swap(xr, yr);
            std::swap(xi, yi);
        }
        outRe = xr;
        outIm = xi;
    }

private:
    struct Pass { size_t len; size_t stride; size_t twOffset; bool radix2; };

    void runRadix4(const T* xr, const T* xi, T* yr, T* yi, const Pass& pass) const noexcept
    {
        const size_t n1 = pass.len / 4;
        const T* w = twiddles_.data() + pass.twOffset;
        const TwiddleSpan<T> tw { w, w + n1, w + 2 * n1, w + 3 * n1, w + 4 * n1, w + 5 * n1 };
        constexpr int wide = kWide<T>;
        constexpr int narrow = kNarrow<T>;
        if (pass.stride == 1)
        {
            if (n1 % wide == 0)        radix4First<T, wide>(xr, xi, yr, yi, pass.len, tw);
            else if (n1 % narrow == 0) radix4First<T, narrow>(xr, xi, yr, yi, pass.len, tw);
            else                       radix4First<T, 1>(xr, xi, yr, yi, pass.len, tw);
        }
        else if (pass.stride % wide == 0)   radix4Strided<T, wide>(xr, xi, yr, yi, pass.len, pass.stride, tw);
        else if (pass.stride % narrow == 0) radix4Strided<T, narrow>(xr, xi, yr, yi, pass.len, pass.stride, tw);
        else                                radix4Strided<T, 1>(xr, xi, yr, yi, pass.len, pass.stride, tw);
    }

    static void runRadix2(const T* xr, const T* xi, T* yr, T* yi, size_t s) noexcept
    {
        constexpr int wide = kWide<T>;
        constexpr int narrow = kNarrow<T>;
        if (s % wide == 0)        radix2Last<T, wide>(xr, xi, yr, yi, s);
        else if (s % narrow == 0) radix2Last<T, narrow>(xr, xi, yr, yi, s);
        else                      radix2Last<T, 1>(xr, xi, yr, yi, s);
    }

    size_t n_;
    std::vector<T> bufA_, bufB_;   ///< Ping-pong buffers, [re | im] each.
    std::vector<T> twiddles_;
    std::vector<Pass> passes_;
};

} // namespace fft
} // namespace detail

// ============================================================================
// FFTComplex
// ============================================================================

/**
 * @class FFTComplex
 * @brief Complex FFT on interleaved data (Stockham radix-4 engine, SIMD).
 *
 * Data layout: interleaved [re0, im0, re1, im1, ...], total 2*N elements.
 * The transform is in place from the caller's view (the engine works in its
 * own buffers).
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FFTComplex
{
public:
    /**
     * @brief Constructs an FFT processor for the given size.
     * @param size Number of complex samples. Must be a power of two and >= 2.
     * @throw std::invalid_argument if size is invalid (with DSPARK_NO_EXCEPTIONS:
     *        asserts and degrades to the minimal valid size instead).
     */
    explicit FFTComplex(size_t size)
        : engine_(validateSize(size))
    {
    }

    /**
     * @brief Returns the FFT size (number of complex points).
     * @return The size provided at construction.
     */
    [[nodiscard]] size_t getSize() const noexcept { return engine_.size(); }

    /**
     * @brief Performs a forward (time->frequency) FFT in-place.
     * @param data Interleaved complex data [re, im, ...], 2*N elements, any
     *             alignment.
     */
    void forward(T* data) noexcept { transform(data, false); }

    /**
     * @brief Performs an inverse (frequency->time) FFT in-place.
     * @param data Interleaved complex data, overwritten with time-domain result.
     * @note The output is automatically scaled by 1/N. Unaligned-safe.
     */
    void inverse(T* data) noexcept { transform(data, true); }

private:
    static size_t validateSize(size_t size)
    {
        if (size < 2 || (size & (size - 1)) != 0)
        {
#if defined(DSPARK_NO_EXCEPTIONS)
            assert(false && "FFTComplex size must be a power of two >= 2");
            return 2; // degrade to the minimal valid size
#else
            throw std::invalid_argument("FFTComplex size must be a power of two >= 2");
#endif
        }
        return size;
    }

    void transform(T* data, bool inverse) noexcept
    {
        constexpr int W = detail::fft::kWide<T>;
        using O = detail::fft::Vec<T, W>;
        using O1 = detail::fft::Vec<T, 1>;
        const size_t n = engine_.size();
        T* re = engine_.inRe();
        T* im = engine_.inIm();

        // Split the interleaved input; the inverse conjugates on the way in.
        size_t i = 0;
        for (; i + W <= n; i += W)
        {
            typename O::V r, m;
            O::loadDeinterleave(data + 2 * i, r, m);
            O::store(re + i, r);
            O::store(im + i, inverse ? O::sub(O::set1(T(0)), m) : m);
        }
        for (; i < n; ++i)
        {
            re[i] = data[2 * i];
            im[i] = inverse ? -data[2 * i + 1] : data[2 * i + 1];
        }

        const T* outRe = nullptr;
        const T* outIm = nullptr;
        engine_.run(outRe, outIm);

        // Interleave back; the inverse conjugates again and scales by 1/N.
        const T scale = inverse ? T(1) / static_cast<T>(n) : T(1);
        const T imScale = inverse ? -scale : scale;
        const auto vs = O::set1(scale), vis = O::set1(imScale);
        i = 0;
        for (; i + W <= n; i += W)
            O::storeInterleave2(data + 2 * i, O::mul(O::load(outRe + i), vs),
                                O::mul(O::load(outIm + i), vis));
        for (; i < n; ++i)
            O1::storeInterleave2(data + 2 * i, outRe[i] * scale, outIm[i] * imScale);
    }

    detail::fft::SplitFFT<T> engine_;
};

// ============================================================================
// FFTReal
// ============================================================================

/**
 * @class FFTReal
 * @brief FFT optimised for real-valued input signals (the common audio case).
 *
 * Runs a half-size complex FFT on the (even, odd) sample pairs and splits its
 * spectrum with a vectorised post-pass, saving about half the work.
 *
 * **Frequency domain layout**: N+2 elements (interleaved complex).
 * - Bins 0 to N/2 inclusive -> (N/2 + 1) complex values -> (N + 2) floats.
 * - `data[2*k]` = real part of bin k.
 * - `data[2*k+1]` = imaginary part of bin k.
 * - Bin 0 = DC, bin N/2 = Nyquist.
 *
 * In-place use is supported: timeData and freqData may point to the same
 * buffer as long as it holds getFrequencyDomainSize() = N+2 elements.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FFTReal
{
public:
    /**
     * @brief Constructs a real FFT processor.
     * @param size Number of real samples. Must be a power of two and >= 4.
     * @throw std::invalid_argument if size is invalid (with DSPARK_NO_EXCEPTIONS:
     *        asserts and degrades to the minimal valid size instead).
     */
    explicit FFTReal(size_t size)
        : realSize_(validateSize(size))   // validates BEFORE the engine is built
        , halfSize_(realSize_ / 2)        // derived from the VALIDATED size, so the
        , engine_(realSize_ / 2)          // trio stays coherent when size degrades
    {
        computePostTwiddles();
    }

    /** @brief Returns the number of real input samples (N). */
    [[nodiscard]] size_t getSize() const noexcept { return realSize_; }

    /** @brief Returns the frequency-domain buffer size in elements (N + 2). */
    [[nodiscard]] size_t getFrequencyDomainSize() const noexcept { return realSize_ + 2; }

    /** @brief Returns the number of frequency bins (N/2 + 1) including DC and Nyquist. */
    [[nodiscard]] size_t getNumBins() const noexcept { return halfSize_ + 1; }

    /**
     * @brief Forward transform: real time-domain -> complex frequency-domain.
     * @param timeData Input: N real samples.
     * @param freqData Output: N+2 elements (interleaved complex, N/2+1 bins).
     */
    void forward(const T* timeData, T* freqData) noexcept
    {
        constexpr int W = detail::fft::kWide<T>;
        using O = detail::fft::Vec<T, W>;
        const size_t m = halfSize_;
        T* re = engine_.inRe();
        T* im = engine_.inIm();

        // (even, odd) sample pairs are the complex input: a deinterleave.
        size_t i = 0;
        for (; i + W <= m; i += W)
        {
            typename O::V r, q;
            O::loadDeinterleave(timeData + 2 * i, r, q);
            O::store(re + i, r);
            O::store(im + i, q);
        }
        for (; i < m; ++i)
        {
            re[i] = timeData[2 * i];
            im[i] = timeData[2 * i + 1];
        }

        const T* zr = nullptr;
        const T* zi = nullptr;
        engine_.run(zr, zi);
        unpackForward(zr, zi, freqData);
    }

    /**
     * @brief Inverse transform: complex frequency-domain -> real time-domain.
     * @param freqData Input: N+2 elements (interleaved complex, N/2+1 bins).
     * @param timeData Output: N real samples.
     */
    void inverse(const T* freqData, T* timeData) noexcept
    {
        constexpr int W = detail::fft::kWide<T>;
        using O = detail::fft::Vec<T, W>;
        const size_t m = halfSize_;

        // Pack into the conjugated half-size spectrum, run the forward engine,
        // conjugate back and scale by 1/(N/2): ifft = conj(fft(conj)) / M.
        packInverseConj(freqData, engine_.inRe(), engine_.inIm());
        const T* zr = nullptr;
        const T* zi = nullptr;
        engine_.run(zr, zi);

        const T scale = T(1) / static_cast<T>(m);
        const auto vs = O::set1(scale), vis = O::set1(-scale);
        size_t i = 0;
        for (; i + W <= m; i += W)
            O::storeInterleave2(timeData + 2 * i, O::mul(O::load(zr + i), vs),
                                O::mul(O::load(zi + i), vis));
        for (; i < m; ++i)
        {
            timeData[2 * i]     = zr[i] * scale;
            timeData[2 * i + 1] = -zi[i] * scale;
        }
    }

    /**
     * @brief Computes the magnitude of each frequency bin.
     * @param freqData   Input: frequency-domain data (N+2 elements).
     * @param magnitudes Output: N/2+1 magnitude values. Must be pre-allocated.
     */
    void computeMagnitudes(const T* freqData, T* magnitudes) const noexcept
    {
        for (size_t k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            magnitudes[k] = std::sqrt(re * re + im * im);
        }
    }

    /**
     * @brief Computes the phase angle of each frequency bin.
     * @param freqData Input: frequency-domain data (N+2 elements).
     * @param phases   Output: N/2+1 phase values in radians (-pi to +pi).
     */
    void computePhases(const T* freqData, T* phases) const noexcept
    {
        for (size_t k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            phases[k] = std::atan2(im, re);
        }
    }

    /**
     * @brief Computes the power spectrum (magnitude squared) of each bin.
     * @param freqData Input: frequency-domain data (N+2 elements).
     * @param power    Output: N/2+1 power values.
     */
    void computePowerSpectrum(const T* freqData, T* power) const noexcept
    {
        for (size_t k = 0; k <= halfSize_; ++k)
        {
            T re = freqData[2 * k];
            T im = freqData[2 * k + 1];
            power[k] = re * re + im * im;
        }
    }

    /**
     * @brief Returns the frequency in Hz corresponding to a given bin index.
     * @param binIndex   Bin index (0 to N/2).
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (N).
     * @return Center frequency of the requested bin.
     */
    [[nodiscard]] static T binToFrequency(size_t binIndex, double sampleRate, size_t fftSize) noexcept
    {
        return static_cast<T>(static_cast<double>(binIndex) * sampleRate / static_cast<double>(fftSize));
    }

    /**
     * @brief Returns the bin index closest to a given frequency.
     * @param frequency  Frequency in Hz.
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (N).
     * @return Integer index of the nearest bin.
     */
    [[nodiscard]] static size_t frequencyToBin(double frequency, double sampleRate, size_t fftSize) noexcept
    {
        return static_cast<size_t>(std::round(frequency * static_cast<double>(fftSize) / sampleRate));
    }

private:
    /** @brief Validates the real-FFT size and returns it (used in the init list
     *  so the FFTReal-specific message fires before the inner engine).
     *  Under DSPARK_NO_EXCEPTIONS, invalid sizes assert and degrade to the
     *  minimal valid size (4) instead of throwing. */
    static size_t validateSize(size_t size)
    {
        if (size < 4 || (size & (size - 1)) != 0)
        {
#if defined(DSPARK_NO_EXCEPTIONS)
            assert(false && "FFTReal size must be a power of two >= 4");
            return 4;
#else
            throw std::invalid_argument("FFTReal size must be a power of two >= 4");
#endif
        }
        return size;
    }

    void computePostTwiddles()
    {
        twRe_.resize(halfSize_);
        twIm_.resize(halfSize_);
        for (size_t k = 0; k < halfSize_; ++k)
        {
            const double angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k)
                               / static_cast<double>(realSize_);
            twRe_[k] = static_cast<T>(std::cos(angle));
            twIm_[k] = static_cast<T>(std::sin(angle));
        }
    }

    /**
     * @brief Splits the half-size spectrum Z into the real spectrum X:
     *        X[k] = (Z[k] + conj(Z[M-k])) / 2 - j w^k (Z[k] - conj(Z[M-k])) / 2.
     */
    void unpackForward(const T* zr, const T* zi, T* out) const noexcept
    {
        constexpr int W = detail::fft::kWide<T>;
        using O = detail::fft::Vec<T, W>;
        const size_t m = halfSize_;

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
            const auto wr = O::load(twRe_.data() + k), wi = O::load(twIm_.data() + k);
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
            const T wr = twRe_[k], wi = twIm_[k];
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
     *        conj(Z), Z[k] = Xe[k] + j conj(w^k) Xd[k] (see unpackForward()).
     */
    void packInverseConj(const T* in, T* zr, T* zi) const noexcept
    {
        constexpr int W = detail::fft::kWide<T>;
        using O = detail::fft::Vec<T, W>;
        const size_t m = halfSize_;

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
            const auto wr = O::load(twRe_.data() + k), wi = O::load(twIm_.data() + k);
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
            const T wr = twRe_[k], wi = twIm_[k];
            const T tR = wr * dR + wi * dI;
            const T tI = wr * dI - wi * dR;
            zr[k] = xeR - tI;
            zi[k] = -(xeI + tR);
        }
    }

    size_t realSize_;
    size_t halfSize_;
    detail::fft::SplitFFT<T> engine_;
    std::vector<T> twRe_, twIm_;   ///< Post-pass twiddles w^k = exp(-2 pi i k / N).
};

} // namespace dspark
