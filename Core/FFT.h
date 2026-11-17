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
 *   order. On arm64 other than macOS and iOS builds, transforms of up to
 *   4096 double or 8192 float complex points take the bits between the first
 *   pass and the last three at a time (a radix-8 pass with twiddles) where
 *   that saves a pass over the data. An odd log2(N) closes with one radix-8
 *   pass (twiddle-free, the eighth roots entering as sqrt(1/2)) in place of
 *   a radix-4 and a radix-2 pass, or with that pair where the radix-8 pass's
 *   eight rows would sit a way of the level-1 data cache apart or more (4
 *   KiB on x86) and fill one of its sets.
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
 * - **Work buffers on cache lines**: the two buffers the passes alternate
 *   between share one allocation and each starts on a 64-byte boundary (the
 *   second a whole number of 4 KiB pages after the first once a buffer is
 *   that long), so no vector of a pass lies across two cache lines.
 * - **Twiddles in double**: every factor is computed directly with cos/sin in
 *   double precision (no recurrences), so the error stays at the level of the
 *   arithmetic itself. A pass reads its factors as one stream: the six of a
 *   butterfly group side by side, or, for the first pass, the six rows of a
 *   block of groups.
 * - **Inverse by conjugation** (complex transforms): ifft(x) =
 *   conj(fft(conj(x))) / N, folded into the conversion passes, so both
 *   directions share one set of kernels.
 * - **Real transforms by lanes**: the N samples are W sequences of N/W
 *   points, sample t of sequence j at x[W t + j], so a W-lane vector loaded
 *   from the signal holds one point of each: the passes transform every
 *   sequence in its own lane, as a scalar transform works on numbers, and
 *   shuffle nothing. They are real radix-4 passes on packed half spectra (a
 *   first pass of four or eight samples with no twiddles, then passes that
 *   join four spectra into one of four times the points, whose two real
 *   bins take a few additions each), and one last step joins the W spectra,
 *   W bins at a time, into the interleaved spectrum: the only place where
 *   lanes mix. At the AVX width that step works by halves of a vector where
 *   that is faster (eight float lanes at every size, four double lanes up to
 *   2048 samples): it then loads and stores half vectors and shuffles
 *   nothing across the 128-bit halves, with the same bits either way. The
 *   inverse runs the same steps backwards and scales in its last pass. Every
 *   factor comes from the values of one eighth of a turn, so the factors of
 *   mirrored bins are equal or opposite to the last bit, and a unit impulse
 *   and a flat spectrum map to each other exactly.
 *   Transforms shorter than 16 double or 32 float samples, and builds
 *   without vectors, run a half-size complex FFT on the (even, odd) sample
 *   pairs and split its spectrum with a post-pass that takes bins k and
 *   N/2 - k together (one difference and one complex product for the two).
 * - **Zero allocations** after construction; forward()/inverse() are
 *   noexcept and accept pointers of any alignment. With the 32-byte kernels,
 *   interleaved data that starts on a 32-byte boundary costs a complex
 *   transform up to 17 % less (measured at 4096 double points): a vector's
 *   storage is only sure of 16.
 *
 * Measured on hosted CI runners, real transforms, forward plus inverse, ns
 * per real point at 64 / 1024 / 4096 / 65536 points, double and then float,
 * each row on one machine (the machines differ between rows). SSE2 (GCC):
 * 1.3 / 2.0 / 2.4 / 3.4 and 0.93 / 1.0 / 1.2 / 1.7. Built for AVX2 and FMA
 * (GCC): 1.2 / 1.2 / 2.3 / 3.8 and 1.1 / 0.65 / 0.95 / 1.6. NEON on an arm64
 * server core (GCC): 2.2 / 3.4 / 4.2 / 6.2 and 1.6 / 1.9 / 2.3 / 3.1.
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

/// Twiddles of one radix-4 pass: the factors w1, w2, w3 (re, im; rows 0 .. 5)
/// of the groups p = 0 .. n/4 - 1, laid out so that a pass reads them as one
/// stream. Kept as six arrays, the rows of a first pass sat n/4 values apart:
/// a multiple of 4 KiB from 4096 double points on, where they met the four
/// quarters of the input in the same sets of an 8-way level-1 cache.
template <typename T>
struct TwiddleSpan
{
    const T* w;

    /// Groups per block of the first pass's layout, for n/4 = n1 groups and
    /// the work layout's block.
    [[nodiscard]] static constexpr size_t firstBlock(size_t n1, size_t block) noexcept
    {
        return n1 < block ? n1 : block;
    }

    /// Where row `row` of group p sits. The first pass (stride 1), which
    /// loads vectors across p, keeps blocks of `block` groups, six rows of
    /// `block` values each; a strided pass, which broadcasts a group's
    /// factors, keeps the six of each group together.
    [[nodiscard]] static constexpr size_t index(size_t stride, size_t block, size_t row, size_t p) noexcept
    {
        if (stride != 1) return 6 * p + row;
        const size_t l = p & (block - 1);
        return 6 * (p - l) + row * block + l;
    }
};

/// One pass of the plan: its length, stride and twiddle offset. radix2 and
/// radix8 mark the closing passes of length 2 and 8, twiddled8 a radix-8 pass
/// with twiddles (fourteen factors per butterfly group, each repeated across
/// a vector); radix 4 otherwise.
struct Pass { size_t len; size_t stride; size_t twOffset; bool radix2; bool radix8; bool twiddled8; };

/// Bytes between the rows of a closing radix-8 pass from which the plan
/// closes with a radix-4 and a radix-2 pass instead. The pass reads its eight
/// rows together; spaced a way of the level-1 data cache apart they fill one
/// set of an 8-way cache, and the pass then costs from the same to six times
/// as much, by where the output lies. A way holds 4 KiB on x86 (32 KiB in 8
/// ways, 48 KiB in 12); the arm64 cores measured hold 16 KiB, and rows 4 KiB
/// apart cost them less than the two passes.
#if defined(_M_AMD64) || defined(_M_X64) || defined(__x86_64__) || defined(__amd64__) \
    || defined(_M_IX86) || defined(__i386__) || defined(__EMSCRIPTEN__)
inline constexpr size_t kClosingRowReach = 4096;
#else
inline constexpr size_t kClosingRowReach = 8192;
#endif

/// Whether plans take three bits per pass, with twiddles, between the first
/// pass and the closing one where that saves a pass over the data. It needs
/// registers for the sixteen vectors of a radix-8 butterfly and its
/// temporaries (32 on arm64; with the 16 of x86-64 such plans measured slower
/// at most sizes), and it pays where vector operations are what bounds a
/// pass: 2 to 5 % of a transform on an arm64 server core with two vector
/// pipes. On the arm64 cores of the macOS machines measured, which have four,
/// the same plans ran 7 to 12 % slower at 512 real points, so macOS and iOS
/// builds keep to radix 4.
#if defined(DSPARK_SIMD_NEON) && !defined(__APPLE__)
inline constexpr bool kTwiddledRadix8 = true;
#else
inline constexpr bool kTwiddledRadix8 = false;
#endif

/// Bytes between the eight input rows of a twiddled radix-8 pass (an eighth
/// of the transform) from which a plan keeps to radix-4 passes: from 16 KiB
/// on, the pass no longer measured faster than the two it replaces (its
/// eight input rows and eight output rows then compete for the same sets of
/// the level-1 data cache).
inline constexpr size_t kTwiddled8RowReach = 16384;

/// Longest spectra (in points) whose joining pass, in a real transform by
/// lanes, keeps each twiddle as a whole vector for the baseline kernels to
/// load; longer ones keep it once and the pass broadcasts it. A broadcast
/// from memory takes a shuffle on SSE2 and a vector operation on arm64, six
/// per butterfly, where the load takes none; the tables of the long passes
/// would then be as large as the data they turn, and cost more than they
/// save once the transform leaves the level-2 cache. AVX broadcasts for free.
#if (defined(DSPARK_SIMD_SSE2) && !defined(DSPARK_SIMD_AVX)) || defined(DSPARK_SIMD_NEON)
inline constexpr size_t kRealWholeTwiddleReach = 256;
#else
inline constexpr size_t kRealWholeTwiddleReach = 0;
#endif

/// Longest real transforms by lanes (in samples) whose joining step, at
/// vectors of 32 bytes, works by halves (RealPlan::halves): no shuffle then
/// crosses the 128-bit halves, and the step loads and stores half vectors.
/// Those shuffles have one port on the x64 cores that bound this step; the
/// loads and stores are free while the transform works within the level-1
/// data cache and are what the step waits for beyond it. Four double lanes
/// load and store twice as often per value as eight float ones: they gain up
/// to this size, the float lanes at every size measured.
template <typename T>
inline constexpr size_t kRealHalvesReach = sizeof(T) == sizeof(float) ? ~size_t{ 0 } : size_t{ 2048 };

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

/// What a real transform by lanes reads (detail/FftKernels.inl): `lanes`
/// sequences of n points each, one per lane of a vector.
template <typename T>
struct RealPlan
{
    size_t n;               ///< Points of each lane's transform.
    size_t lanes;           ///< Lanes: the transform has n * lanes samples.
    bool first8;            ///< The first pass takes eight samples (log2 n odd), not four.
    bool halves;            ///< The joining step works by halves, its tables in the order of laneBin().
    size_t wholeReach;      ///< Passes joining spectra of up to this many points hold each twiddle `lanes` times.
    const T* twiddles;      ///< The pass tables, by rising length of the spectra they join.
    const T* turn;          ///< The factors of the step that joins the lanes.
    const T* edge;          ///< The factors of the lanes' middle bin.
    T scale;                ///< 1 / (n * lanes), for the inverse.

    /// Points of the spectra the first pass leaves.
    [[nodiscard]] constexpr size_t firstLength() const noexcept { return first8 ? 8 : 4; }

    /// Values in the table of the pass that joins spectra of m points:
    /// (re, im) of three factors for each bin 1 .. m/2 - 1, each one
    /// `repeat` times.
    [[nodiscard]] static constexpr size_t passValues(size_t m, size_t repeat) noexcept
    {
        return 6 * repeat * (m / 2 - 1);
    }

    /// Times the table of the pass that joins spectra of m points holds
    /// each value.
    [[nodiscard]] static constexpr size_t repeatFor(size_t m, size_t lanes, size_t wholeReach) noexcept
    {
        return m <= wholeReach ? lanes : 1;
    }

    /// Whether the joining step of a transform of `size` samples works by
    /// halves: only vectors of 32 bytes have halves, and only up to `reach`
    /// samples does it pay (kRealHalvesReach).
    [[nodiscard]] static constexpr bool halvesFor(size_t size, size_t lanes, size_t reach) noexcept
    {
        return lanes * sizeof(T) == 32 && size <= reach;
    }

    /// The bin (counted from the first of its group) that position p of a
    /// vector holds in the step that joins the lanes: p itself, or, where
    /// the step works by halves, four runs of lanes / 4 bins in the order
    /// (0, 2, 1, 3): what taking interleaved pairs apart gives when no value
    /// leaves its 128-bit half. The tables of that step follow this order.
    [[nodiscard]] static constexpr size_t laneBin(size_t lanes, size_t p, bool byHalves) noexcept
    {
        if (!byHalves) return p;
        const size_t run = lanes / 4;
        const size_t order[4] = { 0, 2, 1, 3 };
        return run * order[p / run] + p % run;
    }

    /// Passes of a lane transform of n points, the first one included: one
    /// per two bits, the first taking three where their number is odd.
    [[nodiscard]] static constexpr size_t passCount(size_t n) noexcept
    {
        return static_cast<size_t>(std::countr_zero(n)) / 2;
    }
};

} // namespace plan

using plan::TwiddleSpan;
using plan::Pass;
using plan::Buffers;
using plan::Plan;
using plan::Closing;
using plan::RealPlan;

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
 * @class WorkBuffers
 * @brief Two buffers of the same length in one allocation, each starting on
 *        a 64-byte boundary, the second a whole number of 64-byte lines after
 *        the first (of 4 KiB pages from 4 KiB per buffer).
 *
 * A pass reads one buffer and writes the other with vectors of up to 32
 * bytes. A buffer that starts wherever its allocation does puts some of those
 * vectors across two cache lines (with 32-byte vectors, every other one where
 * the start is 16 bytes off, which is where the usual allocators put a large
 * block): measured with the 32-byte kernels, transforms of a few thousand
 * points and more then cost 1.2 to 1.6 times as much on one CPU family and
 * 1.05 to 1.1 times on another, and shorter ones up to 1.15 times.
 *
 * Copies place their own buffers and carry the contents over; a moved-from
 * object holds nothing.
 */
template <typename T>
class WorkBuffers
{
public:
    WorkBuffers() = default;

    /// Two zeroed buffers of count values each.
    explicit WorkBuffers(size_t count)
        : count_(count), storage_(values(count), T(0)), lead_(leadOf(storage_.data()))
    {
    }

    WorkBuffers(const WorkBuffers& other)
        : count_(other.count_), storage_(other.storage_.size(), T(0)), lead_(leadOf(storage_.data()))
    {
        if (count_ > 0)
            std::copy_n(other.first(), stride(count_) + count_, first());
    }

    WorkBuffers(WorkBuffers&& other) noexcept
        : count_(std::exchange(other.count_, size_t{ 0 })), storage_(std::move(other.storage_)),
          lead_(std::exchange(other.lead_, size_t{ 0 }))
    {
    }

    WorkBuffers& operator=(WorkBuffers other) noexcept
    {
        std::swap(count_, other.count_);
        storage_.swap(other.storage_);
        std::swap(lead_, other.lead_);
        return *this;
    }

    [[nodiscard]] T* first() noexcept { return storage_.data() + lead_; }
    [[nodiscard]] const T* first() const noexcept { return storage_.data() + lead_; }
    [[nodiscard]] T* second() noexcept { return first() + stride(count_); }
    [[nodiscard]] const T* second() const noexcept { return first() + stride(count_); }

    /// Values the constructor requests for two buffers of count values: the
    /// first buffer rounded up to whole lines, the second, and room to reach
    /// the first boundary.
    [[nodiscard]] static constexpr size_t values(size_t count) noexcept
    {
        return (kLine - 1) + stride(count) + count;
    }

private:
    static constexpr size_t kLine = 64 / sizeof(T);   ///< Values in 64 bytes.

    static constexpr size_t stride(size_t count) noexcept
    {
        return (count + kLine - 1) / kLine * kLine;
    }

    /// Values from data to the next 64-byte boundary.
    static size_t leadOf(const T* data) noexcept
    {
        const auto address = reinterpret_cast<std::uintptr_t>(data);
        return static_cast<size_t>((64 - address % 64) % 64) / sizeof(T);
    }

    size_t count_ = 0;
    std::vector<T> storage_;
    size_t lead_ = 0;
};

/**
 * @class SplitFFT
 * @brief The shared Stockham engine: a forward complex FFT of size N on
 *        split (re, im) arrays, ping-ponging between two work buffers.
 */
template <typename T>
class SplitFFT
{
public:
    /// An engine that holds nothing and must not be run (a real transform of a
    /// size that runs by lanes keeps one of these).
    SplitFFT() = default;

    explicit SplitFFT(size_t n, bool interleavedInput = false) : n_(n), work_(2 * n)
    {
        PlanPasses steps;
        size_t offset = 0;
        const size_t passCount = buildPlan(n_, steps, offset);
        passes_.assign(steps.begin(), steps.begin() + static_cast<std::ptrdiff_t>(passCount));

        twiddles_.assign(offset, T(0));
        for (const Pass& pass : passes_)
        {
            if (pass.radix2 || pass.radix8) continue;
            if (pass.twiddled8)
            {
                // w^(kp), k = 1 .. 7: the fourteen values of a group together,
                // each one a whole vector of the width the pass runs at (it
                // loads them: a broadcast would take a vector operation).
                constexpr size_t lanes = static_cast<size_t>(kWide<T>);
                T* const w = twiddles_.data() + pass.twOffset;
                for (size_t p = 0; p < pass.len / 8; ++p)
                {
                    const double a = -2.0 * std::numbers::pi_v<double> * static_cast<double>(p)
                                   / static_cast<double>(pass.len);
                    for (size_t k = 1; k <= 7; ++k)
                    {
                        T* const at = w + lanes * (14 * p + 2 * (k - 1));
                        std::fill_n(at, lanes, static_cast<T>(std::cos(static_cast<double>(k) * a)));
                        std::fill_n(at + lanes, lanes, static_cast<T>(std::sin(static_cast<double>(k) * a)));
                    }
                }
                continue;
            }
            const size_t n1 = pass.len / 4;
            const size_t block = TwiddleSpan<T>::firstBlock(n1, kBlock<T>);
            T* w = twiddles_.data() + pass.twOffset;
            for (size_t p = 0; p < n1; ++p)
            {
                const double a = -2.0 * std::numbers::pi_v<double> * static_cast<double>(p)
                               / static_cast<double>(pass.len);
                const double factors[6] = { std::cos(a), std::sin(a), std::cos(2.0 * a), std::sin(2.0 * a),
                                            std::cos(3.0 * a), std::sin(3.0 * a) };
                for (size_t row = 0; row < 6; ++row)
                    w[TwiddleSpan<T>::index(pass.stride, block, row, p)] = static_cast<T>(factors[row]);
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
                    inLaneTwiddles_[TwiddleSpan<T>::index(1, B, row, p)] = w[TwiddleSpan<T>::index(1, B, row, from)];
                }
        }
    }

    /// Room a standard library may add to one allocation request for its own
    /// alignment bookkeeping (one library requests 39 more bytes for every
    /// vector of 4 KiB or more).
    static constexpr size_t kAllocationOverhead = 64;

    /**
     * @brief Upper bound of the bytes the constructor requests for a
     *        power-of-two size n: the two work buffers (one allocation), the
     *        twiddle tables and the pass plan (one each), plus
     *        kAllocationOverhead per allocation. Allocates nothing.
     */
    [[nodiscard]] static size_t allocationBytes(size_t n, bool interleavedInput) noexcept
    {
        PlanPasses steps;
        size_t twiddles = 0;
        const size_t passCount = buildPlan(n, steps, twiddles);
        const size_t inLane = inLaneTwiddleCount(steps, passCount, interleavedInput);
        const size_t allocations = 2 + (twiddles > 0 ? 1 : 0) + (inLane > 0 ? 1 : 0);
        return (WorkBuffers<T>::values(2 * n) + twiddles + inLane) * sizeof(T)
             + passCount * sizeof(Pass) + allocations * kAllocationOverhead;
    }

    [[nodiscard]] size_t size() const noexcept { return n_; }

    /// Input buffers: fill inRe() / inIm() with N values each before run().
    [[nodiscard]] T* inRe() noexcept { return work_.first(); }
    [[nodiscard]] T* inIm() noexcept { return work_.first() + n_; }

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
            T* const re = work_.second();
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
        const T* const second = work_.second();
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
            const T* const re = work_.first();
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
        return { work_.first(), work_.first() + n_, work_.second(), work_.second() + n_ };
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

    // Plan: a first radix-4 pass, twiddled passes and a closing pass without
    // twiddles. With radix-4 passes alone, an odd log2(N) closes with a
    // radix-8 pass in place of a length-8 radix-4 pass and a radix-2 one, or
    // with that pair where the radix-8 pass's eight rows (n/8 elements of the
    // blocked layout each, re and im) would sit kClosingRowReach apart or
    // more, where its rows would not fill a block of the work layout, and
    // where it would be the first pass. Where kTwiddledRadix8, and up to the
    // size whose eighths sit kTwiddled8RowReach apart, the bits between the
    // first pass and the closing one go three at a time (a twiddled radix-8
    // pass), after the one or two radix-4 passes that the count leaves over,
    // and the closing pass is the radix-8 one whenever it may be (per bit it
    // is the cheapest pass) and the radix-4 one otherwise.
    // The constructor and allocationBytes() both build it here; twiddleValues
    // receives the size of the twiddle table.
    static size_t buildPlan(size_t n, PlanPasses& out, size_t& twiddleValues) noexcept
    {
        const size_t bits = static_cast<size_t>(std::countr_zero(n));
        size_t len = n, stride = 1, offset = 0, count = 0;
        const auto radix4 = [&]() noexcept {
            out[count++] = { len, stride, offset, false, false, false };
            offset += 6 * (len / 4);
            len /= 4;
            stride *= 4;
        };
        const auto twiddled8 = [&]() noexcept {
            out[count++] = { len, stride, offset, false, false, true };
            offset += 14 * static_cast<size_t>(kWide<T>) * (len / 8);
            len /= 8;
            stride *= 8;
        };
        if (bits >= 2)
        {
            const size_t rest = bits - 2;   // bits after the first pass
            const bool closing8 = n >= 32 && n / 8 >= kBlock<T>
                               && 2 * (n / 8) * sizeof(T) < plan::kClosingRowReach;
            // Bits of the closing pass (0: the first pass is the only one),
            // and the radix-4 and twiddled radix-8 passes before it.
            size_t closing = 0, fours = 0, eights = 0;
            if (rest > 0)
            {
                closing = (rest & 1) == 0 ? 2 : closing8 ? 3 : 1;
                fours = (rest - closing) / 2;
                if constexpr (plan::kTwiddledRadix8)
                {
                    const size_t c = closing8 && rest >= 3 && rest != 4 ? 3 : 2;
                    if (rest >= c + 3 && 2 * (n / 8) * sizeof(T) < plan::kTwiddled8RowReach)
                    {
                        const size_t between = rest - c;
                        // Remainders 0, 1 and 2 of three leave 0, 2 and 1 radix-4 passes.
                        const size_t f = (3 - between % 3) % 3;
                        const size_t e = (between - 2 * f) / 3;
                        // A twiddled radix-8 pass needs whole blocks of stride.
                        if (between >= 2 * f && e > 0 && (size_t{ 4 } << (2 * f)) >= kBlock<T>)
                        {
                            closing = c;
                            fours = f;
                            eights = e;
                        }
                    }
                }
            }
            radix4();
            for (size_t i = 0; i < fours; ++i) radix4();
            for (size_t i = 0; i < eights; ++i) twiddled8();
            if (closing == 2)
                radix4();
            else if (closing == 3)
                out[count++] = { 8, stride, offset, false, true, false };
            else if (closing == 1)
                out[count++] = { 2, stride, offset, true, false, false };
        }
        else
            out[count++] = { 2, stride, offset, true, false, false };
        twiddleValues = offset;
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

    size_t n_ = 0;
    WorkBuffers<T> work_;          ///< Ping-pong buffers, [re | im] each.
    std::vector<T> twiddles_;
    std::vector<T> inLaneTwiddles_;   ///< The first pass's, permuted for in-lane interleaved input.
    std::vector<Pass> passes_;
    bool stepAvx2_ = false;           ///< The kernel set runToStep() chose at step 0.
};

/**
 * @class RealLaneTables
 * @brief The tables of a real transform by lanes (detail/FftKernels.inl) for
 *        one lane count, in one allocation: the twiddles of its passes, the
 *        factors of the step that joins the lanes and those of the lanes'
 *        middle bin.
 */
template <typename T>
class RealLaneTables
{
public:
    RealLaneTables() = default;

    /**
     * @brief Builds the tables of a transform of `size` real samples.
     * @param size       Samples: a power of two, at least 4 * lanes and 2 * lanes^2.
     * @param lanes      Lanes of the kernels that will run it (2, 4 or 8).
     * @param wholeReach Passes that join spectra of up to this many points
     *                   keep each twiddle `lanes` times, for kernels that
     *                   load it as a whole vector; the others keep it once.
     * @param halvesReach Transforms of up to this many samples, at vectors
     *                   of 32 bytes, join their lanes by halves
     *                   (RealPlan::halves): the joining step's factors then
     *                   follow RealPlan::laneBin().
     */
    RealLaneTables(size_t size, size_t lanes, size_t wholeReach, size_t halvesReach)
        : n_(size / lanes), lanes_(lanes), wholeReach_(wholeReach),
          halves_(RealPlan<T>::halvesFor(size, lanes, halvesReach)), values_(valueCount(size, lanes, wholeReach))
    {
        T* w = values_.data();
        double c = 0.0, s = 0.0;
        // Passes: w^(ji), j = 1 .. 3, w = exp(-2 pi i / 4m), for the bins i
        // of the spectra of m points that the pass joins.
        for (size_t m = firstLength(n_); m < n_; m *= 4)
        {
            const size_t repeat = RealPlan<T>::repeatFor(m, lanes_, wholeReach_);
            for (size_t i = 1; i < m / 2; ++i)
                for (size_t j = 1; j <= 3; ++j)
                {
                    unit(j * i, 4 * m, c, s);
                    w = std::fill_n(w, repeat, static_cast<T>(c));
                    w = std::fill_n(w, repeat, static_cast<T>(0.0 - s));
                }
        }
        // The joining step: exp(-2 pi i j k / size) for the lanes j >= 1,
        // the bins k of a group along a vector, in the order the kernels
        // hold them (RealPlan::laneBin()).
        turn_ = static_cast<size_t>(w - values_.data());
        for (size_t first = 0; first < n_ / 2; first += lanes_)
            for (size_t j = 1; j < lanes_; ++j)
            {
                for (size_t t = 0; t < lanes_; ++t)
                {
                    unit(j * (first + RealPlan<T>::laneBin(lanes_, t, halves_)), size, c, s);
                    w[t] = static_cast<T>(c);
                    w[lanes_ + t] = static_cast<T>(0.0 - s);
                }
                w += 2 * lanes_;
            }
        // The middle bin: cos and sin of pi j (2r + 1) / lanes.
        edge_ = static_cast<size_t>(w - values_.data());
        const size_t half = lanes_ / 2;
        for (size_t j = 0; j < lanes_; ++j)
            for (size_t r = 0; r < half; ++r)
            {
                unit(j * (2 * r + 1), 2 * lanes_, c, s);
                w[j * half + r] = static_cast<T>(c);
                w[lanes_ * half + j * half + r] = static_cast<T>(s);
            }
    }

    /// Values the constructor requests (one allocation).
    [[nodiscard]] static constexpr size_t valueCount(size_t size, size_t lanes, size_t wholeReach) noexcept
    {
        const size_t n = size / lanes;
        size_t count = (lanes - 1) * n + lanes * lanes;
        for (size_t m = firstLength(n); m < n; m *= 4)
            count += RealPlan<T>::passValues(m, RealPlan<T>::repeatFor(m, lanes, wholeReach));
        return count;
    }

    [[nodiscard]] bool empty() const noexcept { return lanes_ == 0; }

    /// Passes of each lane's transform (0 when empty).
    [[nodiscard]] size_t passes() const noexcept { return empty() ? 0 : RealPlan<T>::passCount(n_); }

    [[nodiscard]] RealPlan<T> plan() const noexcept
    {
        return { n_, lanes_, firstLength(n_) == 8, halves_, wholeReach_, values_.data(), values_.data() + turn_,
                 values_.data() + edge_, T(1) / static_cast<T>(n_ * lanes_) };
    }

private:
    /// Points of the spectra the first pass leaves: eight where log2(n) is odd.
    [[nodiscard]] static constexpr size_t firstLength(size_t n) noexcept
    {
        return (std::countr_zero(n) & 1) != 0 ? 8 : 4;
    }

    /// cos and sin of 2 pi k / count (count a power of two, at least 4) with
    /// the symmetries of the circle exact: the values of an eighth of a turn
    /// serve the whole of it, so that the factors of mirrored bins are
    /// equal or opposite to the last bit and what must cancel does.
    static void unit(size_t k, size_t count, double& c, double& s) noexcept
    {
        const size_t quarter = count / 4;
        const size_t rest = k % quarter;
        const double step = 2.0 * std::numbers::pi_v<double> / static_cast<double>(count);
        double x = 1.0, y = 0.0;
        if (rest != 0)
        {
            if (2 * rest == quarter)
                x = y = 0.70710678118654752440;
            else if (2 * rest < quarter)
            {
                x = std::cos(step * static_cast<double>(rest));
                y = std::sin(step * static_cast<double>(rest));
            }
            else
            {
                x = std::sin(step * static_cast<double>(quarter - rest));
                y = std::cos(step * static_cast<double>(quarter - rest));
            }
        }
        switch ((k / quarter) & 3)   // quarter turns
        {
            case 0: c = x; s = y; break;
            case 1: c = 0.0 - y; s = x; break;   // zeros stay positive
            case 2: c = 0.0 - x; s = 0.0 - y; break;
            default: c = y; s = 0.0 - x; break;
        }
    }

    size_t n_ = 0;        ///< Points per lane.
    size_t lanes_ = 0;
    size_t wholeReach_ = 0;
    bool halves_ = false; ///< The joining step works by halves.
    size_t turn_ = 0;     ///< Where the joining step's factors start.
    size_t edge_ = 0;     ///< Where the middle bin's start.
    std::vector<T> values_;
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
 * Transforms the samples as W interleaved sequences, one in each lane of a
 * vector, with real radix-4 passes, and joins their spectra in one last step
 * (see the file comment): about half the work of a complex transform of the
 * same length. The shortest sizes run a half-size complex FFT on the (even,
 * odd) sample pairs and split its spectrum with a post-pass.
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
        : realSize_(validateSize(size))   // validates BEFORE anything is built
        , halfSize_(realSize_ / 2)        // derived from the VALIDATED size, so
        , engine_(makeEngine(realSize_))  // everything stays coherent when size degrades
    {
        const size_t lanes = baseLanes(realSize_);
        if (lanes == 0)
        {
            computeSplitFactors();
            return;
        }
        laneWork_ = detail::fft::WorkBuffers<T>(realSize_);
        constexpr size_t halvesReach = detail::fft::plan::kRealHalvesReach<T>;
        lanes_ = detail::fft::RealLaneTables<T>(realSize_, lanes, detail::fft::plan::kRealWholeTwiddleReach,
                                                halvesReach);
        if (const size_t other = dispatchedLanes(realSize_); other != lanes)
            wideLanes_ = detail::fft::RealLaneTables<T>(realSize_, other, 0, halvesReach);
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
        constexpr size_t overhead = detail::fft::SplitFFT<T>::kAllocationOverhead;
        const size_t lanes = baseLanes(size);
        if (lanes == 0)
            return detail::fft::SplitFFT<T>::allocationBytes(size / 2, true)
                 + 2 * (splitFactorCount(size / 2) * sizeof(T) + overhead);
        using Tables = detail::fft::RealLaneTables<T>;
        size_t values = detail::fft::WorkBuffers<T>::values(size)
                      + Tables::valueCount(size, lanes, detail::fft::plan::kRealWholeTwiddleReach);
        size_t allocations = 2;
        if (const size_t other = dispatchedLanes(size); other != lanes)
        {
            values += Tables::valueCount(size, other, 0);
            ++allocations;
        }
        return values * sizeof(T) + allocations * overhead;
    }

    /**
     * @brief Forward transform: real time-domain -> complex frequency-domain.
     * @param timeData Input: N real samples.
     * @param freqData Output: N+2 elements (interleaved complex, N/2+1 bins).
     */
    void forward(const T* timeData, T* freqData) noexcept
    {
        if (!lanes_.empty())
        {
            T* const a = laneWork_.first();
            T* const b = laneWork_.second();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
            if (simd::isRuntimeDispatchActive())
            {
                detail::fft::avx2::realForward(dispatchedTables().plan(), timeData, freqData, a, b);
                return;
            }
#endif
            detail::fft::realForward(lanes_.plan(), timeData, freqData, a, b);
            return;
        }
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
        if (!lanes_.empty())
        {
            T* const a = laneWork_.first();
            T* const b = laneWork_.second();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
            if (simd::isRuntimeDispatchActive())
            {
                detail::fft::avx2::realInverse(dispatchedTables().plan(), freqData, timeData, a, b);
                return;
            }
#endif
            detail::fft::realInverse(lanes_.plan(), freqData, timeData, a, b);
            return;
        }
        // Pack into the conjugated half-size spectrum, run the forward engine,
        // conjugate back and scale by 1/(N/2): ifft = conj(fft(conj)) / M.
        packInverseConj(freqData, engine_.inRe(), engine_.inIm());
        const T scale = inverseScale();
        engine_.runTo(timeData, scale, -scale);
    }

    /** @brief Returns the number of steps inverseStep() splits inverse() into
     *  (the same for the life of the object). */
    [[nodiscard]] int getInverseStepCount() const noexcept
    {
        if (!lanes_.empty())
            return static_cast<int>(std::max(lanes_.passes(), wideLanes_.passes())) + 1;
        return static_cast<int>(engine_.runToStepCount()) + 1;
    }

    /**
     * @brief Runs one step of inverse(), for callers that spread a large
     *        transform over several calls (a real-time thread that must not
     *        spend the whole transform in one callback).
     *
     * Step 0 reads the spectrum and each later step is at most one pass over
     * the data: the count is the same for every kernel set of the build, and
     * a set whose transform has fewer passes does nothing in the steps right
     * after the first. Running steps 0 .. getInverseStepCount() - 1 in order,
     * with no other transform on this instance in between, writes exactly
     * what inverse() writes, bit for bit.
     *
     * @param step     Step index, 0 .. getInverseStepCount() - 1.
     * @param freqData Input spectrum, N+2 elements (read by step 0 only).
     * @param timeData Output: N real samples (written by the last step only).
     */
    void inverseStep(int step, const T* freqData, T* timeData) noexcept
    {
        if (!lanes_.empty())
        {
            // Step 0 chooses the kernel set and the later steps keep it. A
            // set whose transform has fewer passes than the count leaves its
            // spare steps right after the first, so the last step is always
            // the one that writes the samples.
            if (step <= 0) stepDispatched_ = simd::isRuntimeDispatchActive();
            const detail::fft::RealLaneTables<T>& tables = stepDispatched_ ? dispatchedTables() : lanes_;
            const int spare = getInverseStepCount() - 1 - static_cast<int>(tables.passes());
            if (step > 0 && step <= spare) return;
            const size_t own = step <= 0 ? size_t{ 0 } : static_cast<size_t>(step - spare);
            T* const a = laneWork_.first();
            T* const b = laneWork_.second();
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
            if (stepDispatched_)
            {
                detail::fft::avx2::realInverseStep(tables.plan(), own, freqData, timeData, a, b);
                return;
            }
#endif
            detail::fft::realInverseStep(tables.plan(), own, freqData, timeData, a, b);
            return;
        }
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

    /// Split factors kept for a half size m (see computeSplitFactors()).
    [[nodiscard]] static size_t splitFactorCount(size_t m) noexcept
    {
        return std::min(m, m / 2 + detail::fft::kBlock<T>);
    }

    void computeSplitFactors()
    {
        // The pre and post passes pair bin k <= M/2 with M - k through the
        // split factor c_k = (1 - j w^k) / 2, w = exp(-2 pi i / N): with t =
        // 2 pi k / N, c_k = ((1 - sin t) / 2, -cos(t) / 2), its real part
        // taken as sin^2(pi / 4 - t / 2), which keeps its relative accuracy
        // where 1 - sin t cancels. c_(M-k) = conj(c_k), so the table is built
        // to satisfy it exactly: computed up to M/2 (where c is exactly 0)
        // and mirrored past it, as far as a vector starting at or below M/2
        // reads.
        const size_t m = halfSize_;
        const size_t used = splitFactorCount(m);
        splitRe_.resize(used);
        splitIm_.resize(used);
        for (size_t k = 0; k <= m / 2; ++k)
        {
            const double half = std::numbers::pi_v<double> * static_cast<double>(k)
                              / static_cast<double>(realSize_);
            const double s = std::sin(0.25 * std::numbers::pi_v<double> - half);
            splitRe_[k] = static_cast<T>(s * s);
            splitIm_[k] = static_cast<T>(-0.5 * std::cos(2.0 * half));
        }
        splitRe_[m / 2] = T(0);
        splitIm_[m / 2] = T(0);
        for (size_t k = m / 2 + 1; k < used; ++k)
        {
            splitRe_[k] = splitRe_[m - k];
            splitIm_[k] = -splitIm_[m - k];
        }
    }

    /// Splits the half-size spectrum into the real one (FftKernels.inl).
    void unpackForward(const T* zr, const T* zi, T* out) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            detail::fft::avx2::unpackRealForward(zr, zi, out, halfSize_, splitRe_.data(), splitIm_.data());
            return;
        }
#endif
        detail::fft::unpackRealForward(zr, zi, out, halfSize_, splitRe_.data(), splitIm_.data());
    }

    /// Packs the real spectrum into the conjugated half-size one
    /// (FftKernels.inl).
    void packInverseConj(const T* in, T* zr, T* zi) const noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            detail::fft::avx2::packRealSpectrum(in, zr, zi, halfSize_, splitRe_.data(), splitIm_.data());
            return;
        }
#endif
        detail::fft::packRealSpectrum(in, zr, zi, halfSize_, splitRe_.data(), splitIm_.data());
    }

    /// The inverse's 1/M for the half-size spectrum.
    [[nodiscard]] T inverseScale() const noexcept { return T(1) / static_cast<T>(halfSize_); }

    /// Lanes that kernels with vectors of `wide` and `narrow` values give a
    /// transform of `size` samples: the widest vector whose lanes each hold
    /// at least eight points and half the square of the lane count (16
    /// samples at two lanes, 32 at four, 256 at eight); 0 where the transform
    /// runs on the half-size complex engine (sizes below that, and builds
    /// without vectors). The step that joins the lanes costs more the wider
    /// the vector, and a shorter transform has too few passes to make up for
    /// it: below those sizes the lanes measured the same as the other engine
    /// or slower with one compiler or another.
    [[nodiscard]] static constexpr size_t lanesFor(size_t size, int wide, int narrow) noexcept
    {
        const auto fits = [size](int width) noexcept {
            const size_t lanes = static_cast<size_t>(width);
            return width > 1 && size >= 8 * lanes && 2 * size >= lanes * lanes * lanes;
        };
        return fits(wide) ? static_cast<size_t>(wide) : fits(narrow) ? static_cast<size_t>(narrow) : size_t{ 0 };
    }

    /// Lanes of the build's baseline kernels for `size` samples.
    [[nodiscard]] static constexpr size_t baseLanes(size_t size) noexcept
    {
        return lanesFor(size, detail::fft::kWide<T>, detail::fft::kNarrow<T>);
    }

    /// Lanes of the kernels that run when the run-time dispatch is active
    /// (the baseline's where the build has no dispatch).
    [[nodiscard]] static constexpr size_t dispatchedLanes(size_t size) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        return lanesFor(size, detail::fft::avx2::kWide<T>, detail::fft::avx2::kNarrow<T>);
#else
        return baseLanes(size);
#endif
    }

    /// The half-size complex engine, for the sizes that run on it.
    [[nodiscard]] static detail::fft::SplitFFT<T> makeEngine(size_t size)
    {
        if (baseLanes(size) != 0)
            return detail::fft::SplitFFT<T>();
        return detail::fft::SplitFFT<T>(size / 2, true);
    }

    /// The tables of the dispatched kernels: their own where their lane count
    /// differs from the baseline's.
    [[nodiscard]] const detail::fft::RealLaneTables<T>& dispatchedTables() const noexcept
    {
        return wideLanes_.empty() ? lanes_ : wideLanes_;
    }

    size_t realSize_;
    size_t halfSize_;
    detail::fft::SplitFFT<T> engine_;    ///< Runs the sizes that do not go by lanes (empty otherwise).
    std::vector<T> splitRe_, splitIm_;   ///< Split factors c_k = (1 - j exp(-2 pi i k / N)) / 2.
    detail::fft::WorkBuffers<T> laneWork_;        ///< The two buffers of the transform by lanes.
    detail::fft::RealLaneTables<T> lanes_;        ///< The baseline kernels' tables (empty: engine_ runs).
    detail::fft::RealLaneTables<T> wideLanes_;    ///< The dispatched kernels' own tables.
    bool stepDispatched_ = false;                 ///< The kernel set inverseStep() chose at step 0.
};

} // namespace dspark
