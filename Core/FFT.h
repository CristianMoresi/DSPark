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
 *   interleave. A transform runs all of its passes on one kernel set, chosen
 *   when it starts.
 * - **No shuffle across 128-bit halves for interleaved data** at the AVX
 *   width: the first pass reads interleaved input with its lanes permuted
 *   (its twiddles follow), and a double transform that ends in interleaved
 *   output keeps its blocks in that order, so its closing pass interleaves
 *   within halves. The lane-wise passes in between do not see the order.
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
 * per real point, 64 / 1024 / 4096 / 65536 points: 1.4 / 1.8 / 2.1 / 2.9 on
 * SSE2 and 1.1 / 0.87 / 1.1 / 1.9 with the AVX2 dispatch (before the
 * register-resident passes and the in-lane layout: 1.5 / 1.9 / 2.2 / 3.0 and
 * 1.1 / 1.0 / 1.3 / 2.0). Float: 1.1 / 0.97 / 1.1 / 1.5 and 0.97 / 0.59 /
 * 0.63 / 0.93.
 *
 * Threading: an instance keeps work buffers, so one instance serves one
 * thread at a time; give each thread its own instance.
 *
 * Dependencies: SimdOps.h (platform SIMD detection and the vector abstraction).
 */

#include "SimdOps.h"

#include <algorithm>
#include <array>
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

/// What a transform reads of its engine: the passes, their twiddles, and the
/// first pass's twiddles permuted for in-lane input (null when not built).
template <typename T>
struct Plan
{
    const Pass* passes;
    size_t passCount;
    size_t n;
    const T* twiddles;
    const T* inLaneTwiddles;
};

/// Where a closing pass into interleaved output writes, and its scales.
template <typename T>
struct Closing { T* out; T scaleRe; T scaleIm; };

} // namespace plan

using plan::TwiddleSpan;
using plan::Pass;
using plan::Buffers;
using plan::Plan;
using plan::Closing;

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
    explicit SplitFFT(size_t n, bool interleavedInput = false) : n_(n)
    {
        bufA_.assign(2 * n_, T(0));
        bufB_.assign(2 * n_, T(0));

        PlanPasses steps;
        const size_t passCount = buildPlan(n_, steps);
        passes_.assign(steps.begin(), steps.begin() + static_cast<std::ptrdiff_t>(passCount));
        const Pass& last = passes_.back();
        const size_t offset = last.radix2 || last.radix8 ? last.twOffset
                                                         : last.twOffset + 6 * (last.len / 4);

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

        // The first pass's twiddles once more, for interleaved input read in
        // lane (see FftKernels.inl, kInLaneDeinterleave): within each block of
        // butterflies, its four runs of kBlock / 4 in the order (0, 2, 1, 3).
        // Only builds with a kernel set that reads that way keep them.
        if (const size_t inLane = inLaneTwiddleCount(steps, passCount, interleavedInput); inLane > 0)
        {
            const Pass& first = passes_.front();
            constexpr size_t B = kBlock<T>, q = B / 4;
            const size_t n1 = first.len / 4;
            inLaneTwiddles_.assign(inLane, T(0));
            const T* w = twiddles_.data() + first.twOffset;
            for (size_t row = 0; row < 6; ++row)
                for (size_t p = 0; p < n1; ++p)
                {
                    const size_t l = p % B, j = l / q;
                    const size_t from = p - l + q * ((j & 1) * 2 + (j >> 1)) + l % q;
                    inLaneTwiddles_[row * n1 + p] = w[row * n1 + from];
                }
        }
    }

    /// Room a standard library may add to one allocation request for its own
    /// alignment bookkeeping (one library requests 39 more bytes for every
    /// vector of 4 KiB or more).
    static constexpr size_t kAllocationOverhead = 64;

    /**
     * @brief Upper bound of the bytes the constructor requests for a
     *        power-of-two size n: the two work buffers, the twiddle tables and
     *        the pass plan, one allocation each, plus kAllocationOverhead per
     *        allocation. Allocates nothing.
     */
    [[nodiscard]] static size_t allocationBytes(size_t n, bool interleavedInput) noexcept
    {
        PlanPasses steps;
        const size_t passCount = buildPlan(n, steps);
        const Pass& last = steps[passCount - 1];
        const size_t twiddles = last.radix2 || last.radix8 ? last.twOffset
                                                           : last.twOffset + 6 * (last.len / 4);
        const size_t inLane = inLaneTwiddleCount(steps, passCount, interleavedInput);
        const size_t allocations = 3 + (twiddles > 0 ? 1 : 0) + (inLane > 0 ? 1 : 0);
        return (4 * n + twiddles + inLane) * sizeof(T) + passCount * sizeof(Pass)
             + allocations * kAllocationOverhead;
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
        const Plan<T> p = plan();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformRun(p, b);
        else
#endif
            fft::transformRun(p, b);
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
        if (passes_.size() == 1)
        {
            T* const re = bufB_.data();
            T* const im = re + n_;
            onePass([&](size_t i, T& r, T& m) { readPair(data, conjugate, i, r, m); },
                    [&](size_t i, T r, T m) { re[i] = r; im[i] = m; });
            outRe = re;
            outIm = im;
            return;
        }
        Buffers<T> b = buffers();
        const Plan<T> p = plan();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformFrom(p, b, data, conjugate);
        else
#endif
            fft::transformFrom(p, b, data, conjugate);
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
        const Plan<T> p = plan();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformFromSplit(p, b, re, im);
        else
#endif
            fft::transformFromSplit(p, b, re, im);
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
        const Plan<T> p = plan();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformScaledSwap(p, b, re, im, gain, mask);
        else
#endif
            fft::transformScaledSwap(p, b, re, im, gain, mask);
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
        const Plan<T> p = plan();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformScaledSwapFolded(p, b, re, im, gain, fold);
        else
#endif
            fft::transformScaledSwapFolded(p, b, re, im, gain, fold);
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
        if (passes_.size() == 1)
        {
            const T* const re = bufA_.data();
            const T* const im = re + n_;
            onePass([&](size_t i, T& r, T& m) { r = re[i]; m = im[i]; },
                    [&](size_t i, T r, T m) { out[2 * i] = r * scaleRe; out[2 * i + 1] = m * scaleIm; });
            return;
        }
        Buffers<T> b = buffers();
        const Plan<T> p = plan();
        const Closing<T> c { out, scaleRe, scaleIm };
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformTo(p, b, c);
        else
#endif
            fft::transformTo(p, b, c);
    }

    /// runFrom() and runTo() together. A transform of one pass cannot do both
    /// in that pass, so it reads through the input buffers.
    void runFromTo(const T* data, bool conjugate, T* out, T scaleRe, T scaleIm) noexcept
    {
        if (passes_.size() == 1)
        {
            onePass([&](size_t i, T& r, T& m) { readPair(data, conjugate, i, r, m); },
                    [&](size_t i, T r, T m) { out[2 * i] = r * scaleRe; out[2 * i + 1] = m * scaleIm; });
            return;
        }
        Buffers<T> b = buffers();
        const Plan<T> p = plan();
        const Closing<T> c { out, scaleRe, scaleIm };
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            avx2::transformFromTo(p, b, data, conjugate, c);
        else
#endif
            fft::transformFromTo(p, b, data, conjugate, c);
    }

    /// Steps of runTo() as runToStep() splits it: one per pass before the
    /// closing one, then the closing pass.
    [[nodiscard]] size_t runToStepCount() const noexcept { return passes_.size(); }

    /**
     * @brief Runs one step of runTo(): step s below runToStepCount() - 1 is
     *        pass s, the last step the closing pass into `out`. All steps in
     *        order, with nothing else run in between, equal runTo() bit for
     *        bit (the passes alternate between the two work buffers, so the
     *        buffer a step reads follows from its index). Step 0 chooses the
     *        kernel set and the later steps keep it, as runTo() does.
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
        const Plan<T> p = plan();
        const Closing<T> c { out, scaleRe, scaleIm };
        if (step == 0) stepAvx2_ = simd::isRuntimeDispatchActive();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (stepAvx2_)
            avx2::transformToStep(p, b, step, c);
        else
#endif
            fft::transformToStep(p, b, step, c);
    }

private:
    [[nodiscard]] Buffers<T> buffers() noexcept
    {
        return { bufA_.data(), bufA_.data() + n_, bufB_.data(), bufB_.data() + n_ };
    }

    /// Interleaved pair i, conjugated if asked (by negation, as readInput()).
    static void readPair(const T* data, bool conjugate, size_t i, T& r, T& m) noexcept
    {
        r = data[2 * i];
        m = conjugate ? -data[2 * i + 1] : data[2 * i + 1];
    }

    // A transform of one pass (2 or 4 points) runs here: its few additions
    // cost less than reaching the kernels. The arithmetic is the closing
    // pass's (FftKernels.inl), so the result is the same to the bit. Every
    // input is read before the first output is written (in place is safe).
    template <typename Load, typename Store>
    void onePass(Load load, Store store) const noexcept
    {
        if (n_ == 2)
        {
            T ar, ai, br, bi;
            load(0, ar, ai);
            load(1, br, bi);
            store(0, ar + br, ai + bi);
            store(1, ar - br, ai - bi);
            return;
        }
        T ar, ai, br, bi, cr, ci, dr, di;
        load(0, ar, ai);
        load(1, br, bi);
        load(2, cr, ci);
        load(3, dr, di);
        const T apcR = ar + cr, apcI = ai + ci, amcR = ar - cr, amcI = ai - ci;
        const T bpdR = br + dr, bpdI = bi + di, bmdR = br - dr, bmdI = bi - di;
        store(0, apcR + bpdR, apcI + bpdI);
        store(1, amcR + bmdI, amcI - bmdR);
        store(2, apcR - bpdR, apcI - bpdI);
        store(3, amcR - bmdI, amcI + bmdR);
    }

    // Each transform makes one call into one kernel set (the AVX2+FMA copy
    // when the dispatch is active), so all of its passes agree on the work
    // layout even if the dispatch flag changes meanwhile.
    [[nodiscard]] Plan<T> plan() const noexcept
    {
        return { passes_.data(), passes_.size(), n_, twiddles_.data(),
                 inLaneTwiddles_.empty() ? nullptr : inLaneTwiddles_.data() };
    }

    /// Room for every plan up to 2^63 points (one pass per factor of four).
    using PlanPasses = std::array<Pass, 40>;

    // Plan: radix-4 passes of length n, n/4, ... with stride 1, 4, ...; an
    // odd log2(N) closes with a radix-8 pass in place of a length-8 radix-4
    // pass and a radix-2 one, or with that pair where the radix-8 pass's
    // eight output rows would sit 4 KiB apart or more (they then share cache
    // sets and the pass slows down several times), where its rows would not
    // fill a block of the work layout, and where it would be the first pass.
    // The constructor and allocationBytes() both build it here.
    static size_t buildPlan(size_t n, PlanPasses& out) noexcept
    {
        const bool odd = (std::countr_zero(n) & 1) != 0;
        const bool radix8 = odd && n >= 32 && n / 8 >= kBlock<T>
                         && (n / 8) * sizeof(T) < 4096;
        size_t len = n, stride = 1, offset = 0, count = 0;
        while (len >= 4 && !(radix8 && len == 8))
        {
            out[count++] = { len, stride, offset, false, false };
            offset += 6 * (len / 4);
            len /= 4;
            stride *= 4;
        }
        if (len == 8)
            out[count++] = { 8, stride, offset, false, true };
        else if (len == 2)
            out[count++] = { 2, stride, offset, true, false };
        return count;
    }

    /// Size of the first pass's in-lane twiddle copy (zero when not built).
    static size_t inLaneTwiddleCount(const PlanPasses& steps, size_t count,
                                     bool interleavedInput) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        constexpr bool inLaneReader = true;
#else
        constexpr bool inLaneReader = kInLaneDeinterleave<T, kWide<T>>;
#endif
        const Pass& first = steps[0];
        if (inLaneReader && interleavedInput && count > 1 && !first.radix2 && !first.radix8
            && (first.len / 4) % kBlock<T> == 0)
            return 6 * (first.len / 4);
        return 0;
    }

    size_t n_;
    std::vector<T> bufA_, bufB_;   ///< Ping-pong buffers, [re | im] each.
    std::vector<T> twiddles_;
    std::vector<T> inLaneTwiddles_;   ///< The first pass's, permuted for in-lane interleaved input.
    std::vector<Pass> passes_;
    bool stepAvx2_ = false;           ///< The kernel set runToStep() chose at step 0.
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
        : engine_(validateSize(size), true)
    {
    }

    /**
     * @brief Returns the FFT size (number of complex points).
     * @return The size provided at construction.
     */
    [[nodiscard]] size_t getSize() const noexcept { return engine_.size(); }

    /**
     * @brief Upper bound of the heap bytes a constructor of this size
     *        requests, excluding the object itself, with up to 64 bytes per
     *        allocation for standard-library bookkeeping. Allocates nothing;
     *        use it to budget memory before constructing. The count depends on
     *        the build's kernel sets.
     * @param size Number of complex samples: a power of two >= 2 (other
     *             sizes return 0).
     * @return At least the sum of the constructor's allocation requests.
     */
    [[nodiscard]] static size_t allocationBytes(size_t size) noexcept
    {
        if (size < 2 || (size & (size - 1)) != 0)
            return 0;
        return detail::fft::SplitFFT<T>::allocationBytes(size, true);
    }

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
        , engine_(realSize_ / 2, true)    // trio stays coherent when size degrades
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
     * @brief Upper bound of the heap bytes a constructor of this size
     *        requests, excluding the object itself, with up to 64 bytes per
     *        allocation for standard-library bookkeeping. Allocates nothing;
     *        use it to budget memory before constructing. The count depends on
     *        the build's kernel sets.
     * @param size Number of real samples: a power of two >= 4 (other sizes
     *             return 0).
     * @return At least the sum of the constructor's allocation requests.
     */
    [[nodiscard]] static size_t allocationBytes(size_t size) noexcept
    {
        if (size < 4 || (size & (size - 1)) != 0)
            return 0;
        return detail::fft::SplitFFT<T>::allocationBytes(size / 2, true)
             + 2 * (postTwiddleCount(size / 2) * sizeof(T)
                    + detail::fft::SplitFFT<T>::kAllocationOverhead);
    }

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

    /// Post-pass twiddles kept for a half size m (see computePostTwiddles()).
    [[nodiscard]] static size_t postTwiddleCount(size_t m) noexcept
    {
        return std::min(m, m / 2 + detail::fft::kBlock<T>);
    }

    void computePostTwiddles()
    {
        // The pre and post passes pair bin k <= M/2 with M - k and take
        // w^(M-k) = -conj(w^k) from w^k, so the table is built to satisfy it
        // exactly: computed up to M/2 (where the real part is exactly 0) and
        // mirrored past it, as far as a vector starting at or below M/2 reads.
        const size_t m = halfSize_;
        const size_t used = postTwiddleCount(m);
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
