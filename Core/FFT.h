// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file FFT.h
 * @brief Fast Fourier Transform: Stockham radix-4 on blocked complex data, SIMD.
 *
 * Provides forward and inverse FFT for both complex and real-valued signals,
 * for power-of-two sizes.
 *
 * Engine:
 * - **Stockham autosort, radix 4**: log4(N) passes with no bit-reversal
 *   permutation; each pass reads one buffer and writes the other in natural
 *   order. An odd log2(N) closes with one radix-8 pass (twiddle-free, the
 *   eighth roots entering as sqrt(1/2)) in place of a radix-4 and a radix-2
 *   pass, or with that pair where the radix-8 pass's rows would sit 4 KiB
 *   apart or more and share cache sets.
 * - **Blocked complex layout inside**: between passes the data is stored in
 *   blocks of 4 double or 8 float real parts followed by the same elements'
 *   imaginary parts, so every butterfly is plain vector arithmetic (no
 *   shuffles in the complex multiplies) and a pass keeps one pointer per
 *   stream. The interleaved or split API layouts are read by the first pass
 *   and written by the last, so converting them costs no pass over memory of
 *   its own. The closing pass needs no table twiddles.
 * - **SIMD at the widest width a pass allows**: AVX (8 float / 4 double
 *   lanes, with FMA when enabled) where the build targets it, and in x86-64
 *   builds below AVX on CPUs with AVX2 and FMA through the run-time dispatch
 *   (simd::setRuntimeDispatch()); SSE2 on every other x86-64, NEON on ARM64,
 *   scalar otherwise. The first pass, whose stride is 1, runs across
 *   butterflies and writes its outputs through an in-register 4-way
 *   interleave.
 * - **Twiddles in double**: every factor is computed directly with cos/sin in
 *   double precision (no recurrences), so the error stays at the level of the
 *   arithmetic itself.
 * - **Inverse by conjugation**: ifft(x) = conj(fft(conj(x))) / N, folded into
 *   the conversion passes, so both directions share one set of kernels.
 * - **Real transforms** run a half-size complex FFT on the (even, odd) sample
 *   pairs and split its spectrum with a vectorised post-pass that takes bins
 *   k and N/2 - k together (their twiddles are exact mirrors, so they share
 *   one product); the inverse packs its input the same way.
 * - **Zero allocations** after construction; forward()/inverse() are
 *   noexcept and accept pointers of any alignment.
 *
 * Measured on one desktop x86-64 core, double, forward plus inverse, ns
 * per real point, 64 / 1024 / 4096 / 65536 points: 1.5 / 1.9 / 2.3 / 3.2 on
 * SSE2 and 1.1 / 1.0 / 1.2 / 2.2 with the AVX2 dispatch (the previous engine:
 * 2.0 / 2.4 / 2.8 / 3.7 and 1.5 / 1.3 / 1.5 / 2.7). Float: 1.2 / 1.0 / 1.2 /
 * 1.6 and 1.0 / 0.6 / 0.7 / 1.0.
 *
 * Threading: an instance keeps work buffers, so one instance serves one
 * thread at a time; give each thread its own instance.
 *
 * Dependencies: SimdOps.h (platform SIMD detection and the vector abstraction).
 */

#include "SimdOps.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
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

// ============================================================================
// Kernels
// ============================================================================

// The plan types live in a namespace with no functions, so argument-dependent
// lookup on them cannot reach the baseline passes from the AVX2 copy.
namespace plan {

/// Twiddles of one radix-4 pass, split by factor: w1, w2, w3 (re, im) for
/// p = 0 .. n/4 - 1.
template <typename T>
struct TwiddleSpan
{
    const T* w1r; const T* w1i;
    const T* w2r; const T* w2i;
    const T* w3r; const T* w3i;
};

/// One pass of the plan: its length, stride and twiddle offset. radix2 and
/// radix8 mark the closing passes of length 2 and 8 (radix 4 otherwise).
struct Pass { size_t len; size_t stride; size_t twOffset; bool radix2; bool radix8; };

/// The two ping-pong buffers, [re | im] each: x is read, y is written.
template <typename T>
struct Buffers { T* xr; T* xi; T* yr; T* yi; };

} // namespace plan

using plan::TwiddleSpan;
using plan::Pass;
using plan::Buffers;

/// Block of the work layout between passes (detail/FftKernels.inl): the
/// widest vector any copy of the kernels uses, the same in every copy.
template <typename T> inline constexpr size_t kBlock = std::is_same_v<T, double> ? 4 : 8;

// The passes, compiled for the build's baseline (detail/FftKernels.inl).
template <typename T, int W> using Vec = simd::Vec<T, W>;
template <typename T> inline constexpr int kWide = simd::kVecWidth<T>;
template <typename T> inline constexpr int kNarrow = simd::kVecNarrowWidth<T>;
#include "detail/FftKernels.inl"

} // namespace fft
} // namespace detail
} // namespace dspark

#if defined(DSPARK_SIMD_DISPATCH_AVX2)
// The same passes compiled for AVX2+FMA, run when isRuntimeDispatchActive().
DSPARK_SIMD_AVX2_BEGIN
namespace dspark {
namespace detail {
namespace fft {
namespace avx2 {
template <typename T, int W> using Vec = simd::avx2::Vec<T, W>;
template <typename T> inline constexpr int kWide = simd::avx2::kVecWidth<T>;
template <typename T> inline constexpr int kNarrow = simd::avx2::kVecNarrowWidth<T>;
#include "detail/FftKernels.inl"
} // namespace avx2
} // namespace fft
} // namespace detail
} // namespace dspark
DSPARK_SIMD_AVX2_END
#endif

namespace dspark {
namespace detail {
namespace fft {

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

        // Plan: radix-4 passes of length n, n/4, ... with stride 1, 4, ...;
        // an odd log2(N) closes with a radix-8 pass in place of a length-8
        // radix-4 pass and a radix-2 one, or with that pair where the
        // radix-8 pass's eight output rows would sit 4 KiB apart or more
        // (they then share cache sets and the pass slows down several
        // times), where its rows would not fill a block of the work layout,
        // and where it would be the first pass.
        const bool odd = (std::countr_zero(n_) & 1) != 0;
        const bool radix8 = odd && n_ >= 32 && n_ / 8 >= kBlock<T>
                         && (n_ / 8) * sizeof(T) < 4096;
        size_t len = n_, stride = 1, offset = 0;
        while (len >= 4 && !(radix8 && len == 8))
        {
            passes_.push_back({ len, stride, offset, false, false });
            offset += 6 * (len / 4);
            len /= 4;
            stride *= 4;
        }
        if (len == 8)
            passes_.push_back({ 8, stride, offset, false, true });
        else if (len == 2)
            passes_.push_back({ 2, stride, offset, true, false });

        twiddles_.assign(offset, T(0));
        for (const Pass& pass : passes_)
        {
            if (pass.radix2 || pass.radix8) continue;
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
        Buffers<T> b = buffers();
        passes(b, 0, passes_.size());
        outRe = b.xr;
        outIm = b.xi;
    }

    /**
     * @brief Transforms interleaved (re, im) pairs, conjugated if asked, without
     *        first copying them into the input buffers.
     * @param data      N interleaved complex values.
     * @param conjugate Conjugates the input.
     * @param outRe     Receives the pointer to the N output real parts.
     * @param outIm     Receives the pointer to the N output imaginary parts.
     */
    void runFrom(const T* data, bool conjugate, const T*& outRe, const T*& outIm) noexcept
    {
        Buffers<T> b = buffers();
        const size_t done = input(data, conjugate, b);
        passes(b, done, passes_.size());
        outRe = b.xr;
        outIm = b.xi;
    }

    /**
     * @brief Transforms N values read from two arrays of real and imaginary
     *        parts, without first copying them into the input buffers.
     * @param re     N real parts (outside this engine's buffers).
     * @param im     N imaginary parts (outside this engine's buffers).
     * @param outRe  Receives the pointer to the N output real parts.
     * @param outIm  Receives the pointer to the N output imaginary parts.
     */
    void runFromSplit(const T* re, const T* im, const T*& outRe, const T*& outIm) noexcept
    {
        Buffers<T> b = buffers();
        const size_t done = inputSplit(re, im, b);
        passes(b, done, passes_.size());
        outRe = b.xr;
        outIm = b.xi;
    }

    /**
     * @brief Transforms the spectrum product whose bin k is
     *        (im[k & mask] gain[k], re[k & mask] gain[k]), read straight from
     *        the arrays. Run on a product's swapped parts, a forward transform
     *        is its inverse with the parts swapped back on the way out, so
     *        this is a filter's multiply and inverse in one go. The product
     *        is never stored, so a build that contracts multiply-adds may
     *        fuse it into the first pass: one rounding fewer than writing it
     *        to the input buffers first and calling run().
     * @param re    Real parts, mask + 1 bins (may be this engine's last output).
     * @param im    Imaginary parts, mask + 1 bins.
     * @param gain  N real gains.
     * @param mask  One less than a power of two that divides N: the spectrum
     *              repeats every mask + 1 bins (N - 1 for none).
     * @param outRe Receives the pointer to the N output real parts.
     * @param outIm Receives the pointer to the N output imaginary parts.
     */
    void runScaledSwap(const T* re, const T* im, const T* gain, size_t mask,
                       const T*& outRe, const T*& outIm) noexcept
    {
        Buffers<T> b = buffers();
        // The first pass must not write over its own input: when that is the
        // second work buffer, the passes start from it.
        const std::less<const T*> before;
        const T* const second = bufB_.data();
        if (!before(re, second) && before(re, second + 2 * n_))
        {
            std::swap(b.xr, b.yr);
            std::swap(b.xi, b.yi);
        }
        const size_t done = inputScaledSwap(re, im, gain, mask, b);
        passes(b, done, passes_.size());
        outRe = b.xr;
        outIm = b.xi;
    }

    /**
     * @brief runScaledSwap() of a spectrum fold times this engine's size,
     *        folded: bin k sums the products of bins k + f N, f < fold. The
     *        result is the inverse of the product at every fold-th sample,
     *        exact when the product holds nothing the fold aliases (a filter
     *        whose stopband starts at the decimated Nyquist).
     * @param re    Real parts, fold * N bins (outside this engine's buffers).
     * @param im    Imaginary parts, fold * N bins.
     * @param gain  fold * N real gains.
     * @param fold  Decimation factor, at least 1.
     * @param outRe Receives the pointer to the N output real parts.
     * @param outIm Receives the pointer to the N output imaginary parts.
     */
    void runScaledSwapFolded(const T* re, const T* im, const T* gain, size_t fold,
                             const T*& outRe, const T*& outIm) noexcept
    {
        Buffers<T> b = buffers();
        size_t done = 0;
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            done = avx2::readScaledSwapFolded(passes_.data(), passes_.size(), n_, twiddles_.data(), re, im,
                                              gain, fold, b);
        else
#endif
            done = fft::readScaledSwapFolded(passes_.data(), passes_.size(), n_, twiddles_.data(), re, im,
                                             gain, fold, b);
        passes(b, done, passes_.size());
        outRe = b.xr;
        outIm = b.xi;
    }

    /**
     * @brief Transforms the input buffers and writes interleaved output directly.
     * @param out     Receives N interleaved complex values.
     * @param scaleRe Factor applied to the real parts.
     * @param scaleIm Factor applied to the imaginary parts.
     */
    void runTo(T* out, T scaleRe, T scaleIm) noexcept
    {
        Buffers<T> b = buffers();
        passes(b, 0, passes_.size() - 1);
        close(b, out, scaleRe, scaleIm);
    }

    /// runFrom() and runTo() together. A transform of one pass cannot do both
    /// in that pass, so it reads through the input buffers.
    void runFromTo(const T* data, bool conjugate, T* out, T scaleRe, T scaleIm) noexcept
    {
        Buffers<T> b = buffers();
        const size_t done = input(data, conjugate, b);
        passes(b, done, passes_.size() - 1);
        close(b, out, scaleRe, scaleIm);
    }

    /// Steps of runTo() as runToStep() splits it: one per pass before the
    /// closing one, then the closing pass.
    [[nodiscard]] size_t runToStepCount() const noexcept { return passes_.size(); }

    /**
     * @brief Runs one step of runTo(): step s below runToStepCount() - 1 is
     *        pass s, the last step the closing pass into `out`. All steps in
     *        order, with nothing else run in between, equal runTo() bit for
     *        bit (the passes alternate between the two work buffers, so the
     *        buffer a step reads follows from its index).
     * @param step    Step index, 0 .. runToStepCount() - 1.
     * @param out     Receives N interleaved complex values (last step only).
     * @param scaleRe Factor applied to the real parts (last step only).
     * @param scaleIm Factor applied to the imaginary parts (last step only).
     */
    void runToStep(size_t step, T* out, T scaleRe, T scaleIm) noexcept
    {
        Buffers<T> b = buffers();
        if ((step & 1u) != 0)
        {
            std::swap(b.xr, b.yr);
            std::swap(b.xi, b.yi);
        }
        if (step + 1 < passes_.size())
            passes(b, step, step + 1);
        else
            close(b, out, scaleRe, scaleIm);
    }

private:
    [[nodiscard]] Buffers<T> buffers() noexcept
    {
        return { bufA_.data(), bufA_.data() + n_, bufB_.data(), bufB_.data() + n_ };
    }

    // Each step runs on the AVX2+FMA copy of the passes when it is active.
    void passes(Buffers<T>& b, size_t begin, size_t end) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            avx2::runPasses(passes_.data(), begin, end, twiddles_.data(), b);
            return;
        }
#endif
        fft::runPasses(passes_.data(), begin, end, twiddles_.data(), b);
    }

    size_t input(const T* data, bool conjugate, Buffers<T>& b) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            return avx2::readInput(passes_.data(), passes_.size(), n_, twiddles_.data(), data,
                                   conjugate, b);
#endif
        return fft::readInput(passes_.data(), passes_.size(), n_, twiddles_.data(), data,
                              conjugate, b);
    }

    size_t inputSplit(const T* re, const T* im, Buffers<T>& b) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            return avx2::readSplit(passes_.data(), passes_.size(), n_, twiddles_.data(), re, im, b);
#endif
        return fft::readSplit(passes_.data(), passes_.size(), n_, twiddles_.data(), re, im, b);
    }

    size_t inputScaledSwap(const T* re, const T* im, const T* gain, size_t mask, Buffers<T>& b) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            return avx2::readScaledSwap(passes_.data(), passes_.size(), n_, twiddles_.data(), re, im, gain,
                                        mask, b);
#endif
        return fft::readScaledSwap(passes_.data(), passes_.size(), n_, twiddles_.data(), re, im, gain, mask, b);
    }

    // The closing pass reads the blocked layout after other passes, and the
    // split input buffers when it is the only one.
    void close(const Buffers<T>& b, T* out, T scaleRe, T scaleIm) const noexcept
    {
        const bool blocked = passes_.size() > 1;
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            avx2::closeInterleaved(passes_.back(), b.xr, b.xi, blocked, out, scaleRe, scaleIm);
            return;
        }
#endif
        fft::closeInterleaved(passes_.back(), b.xr, b.xi, blocked, out, scaleRe, scaleIm);
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
        // The inverse conjugates on the way in, and on the way out scales by 1/N
        // and conjugates again.
        const T scale = inverse ? T(1) / static_cast<T>(engine_.size()) : T(1);
        engine_.runFromTo(data, inverse, data, scale, inverse ? -scale : scale);
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
        // (even, odd) sample pairs are the complex input.
        const T* zr = nullptr;
        const T* zi = nullptr;
        engine_.runFrom(timeData, false, zr, zi);
        unpackForward(zr, zi, freqData);
    }

    /**
     * @brief Inverse transform: complex frequency-domain -> real time-domain.
     * @param freqData Input: N+2 elements (interleaved complex, N/2+1 bins).
     * @param timeData Output: N real samples.
     */
    void inverse(const T* freqData, T* timeData) noexcept
    {
        // Pack into the conjugated half-size spectrum, run the forward engine,
        // conjugate back and scale by 1/(N/2): ifft = conj(fft(conj)) / M,
        // and by the pack's factor 2 (see inverseScale()).
        packInverseConj(freqData, engine_.inRe(), engine_.inIm());
        const T scale = inverseScale();
        engine_.runTo(timeData, scale, -scale);
    }

    /** @brief Returns the number of steps inverseStep() splits inverse() into. */
    [[nodiscard]] int getInverseStepCount() const noexcept
    {
        return static_cast<int>(engine_.runToStepCount()) + 1;
    }

    /**
     * @brief Runs one step of inverse(), for callers that spread a large
     *        transform over several calls (a real-time thread that must not
     *        spend the whole transform in one callback).
     *
     * Step 0 packs the spectrum, each later step is one pass over the data.
     * Running steps 0 .. getInverseStepCount() - 1 in order, with no other
     * transform on this instance in between, writes exactly what inverse()
     * writes, bit for bit.
     *
     * @param step     Step index, 0 .. getInverseStepCount() - 1.
     * @param freqData Input spectrum, N+2 elements (read by step 0 only).
     * @param timeData Output: N real samples (written by the last step only).
     */
    void inverseStep(int step, const T* freqData, T* timeData) noexcept
    {
        if (step <= 0)
        {
            packInverseConj(freqData, engine_.inRe(), engine_.inIm());
            return;
        }
        const T scale = inverseScale();
        engine_.runToStep(static_cast<size_t>(step - 1), timeData, scale, -scale);
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
        // The pre and post passes pair bin k <= M/2 with M - k and take
        // w^(M-k) = -conj(w^k) from w^k, so the table is built to satisfy it
        // exactly: computed up to M/2 (where the real part is exactly 0) and
        // mirrored past it, as far as a vector starting at or below M/2 reads.
        const size_t m = halfSize_;
        const size_t used = std::min(m, m / 2 + detail::fft::kBlock<T>);
        twRe_.resize(used);
        twIm_.resize(used);
        for (size_t k = 0; k <= m / 2; ++k)
        {
            const double angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k)
                               / static_cast<double>(realSize_);
            twRe_[k] = static_cast<T>(std::cos(angle));
            twIm_[k] = static_cast<T>(std::sin(angle));
        }
        twRe_[m / 2] = T(0);
        twIm_[m / 2] = T(-1);
        for (size_t k = m / 2 + 1; k < used; ++k)
        {
            twRe_[k] = -twRe_[m - k];
            twIm_[k] = twIm_[m - k];
        }
    }

    /// Splits the half-size spectrum into the real one (FftKernels.inl).
    void unpackForward(const T* zr, const T* zi, T* out) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            detail::fft::avx2::unpackRealForward(zr, zi, out, halfSize_, twRe_.data(), twIm_.data());
            return;
        }
#endif
        detail::fft::unpackRealForward(zr, zi, out, halfSize_, twRe_.data(), twIm_.data());
    }

    /// Packs the real spectrum into the conjugated half-size one, at twice its
    /// value (FftKernels.inl).
    void packInverseConj(const T* in, T* zr, T* zi) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            detail::fft::avx2::packRealSpectrum(in, zr, zi, halfSize_, twRe_.data(), twIm_.data());
            return;
        }
#endif
        detail::fft::packRealSpectrum(in, zr, zi, halfSize_, twRe_.data(), twIm_.data());
    }

    /// The inverse's 1/M, halved for the half-size spectrum packed at twice
    /// its value (a power of two: the result is exact either way).
    [[nodiscard]] T inverseScale() const noexcept { return T(0.5) / static_cast<T>(halfSize_); }

    size_t realSize_;
    size_t halfSize_;
    detail::fft::SplitFFT<T> engine_;
    std::vector<T> twRe_, twIm_;   ///< Post-pass twiddles w^k = exp(-2 pi i k / N).
};

} // namespace dspark
