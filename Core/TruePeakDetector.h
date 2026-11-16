// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file TruePeakDetector.h
 * @brief Shared ITU-R BS.1770 true-peak (inter-sample peak) detectors.
 *
 * Three readings of the analog reconstruction peak of a digital signal, all
 * following the BS.1770 Annex 2 procedure (over-sampling by at least 4,
 * low-pass interpolation, absolute value):
 *
 * - TruePeakDetector runs the example interpolator printed in BS.1770-5
 *   Annex 2: the 48-tap, 4-phase polyphase FIR (12 taps per phase). It reads
 *   what a meter built on that table reads, including its pass-band ripple:
 *   steady sines read up to 0.22 dB above their true peak near 0.09 fs and
 *   0.25 fs, and the pass band ends near 0.41 fs. Use it to stay at or below
 *   a ceiling as such a meter judges it (Compressor, Limiter guard, the
 *   offline gain stages).
 * - TruePeakEstimator reads the reconstruction itself within 33 samples:
 *   64-tap Kaiser-windowed sinc phases at the quarter points of every sample
 *   interval, and the vertex of the parabola through each local maximum of
 *   that 4x grid. Against the exact band-limited reconstruction it reads
 *   steady sines up to 0.49 fs within 0.003 dB, and tones up to 0.45 fs that
 *   start abruptly within -0.24 / +0.07 dB (the 12-tap table: -0.78 / +0.19).
 *   It is the detector's reading (the Limiter's true-peak mode): above
 *   0.45 fs a tone that starts abruptly reads up to 0.75 dB high or 1.1 dB
 *   low, because the crest there is built by samples further away than a
 *   detector can wait for.
 * - TruePeakAnalyzer is the meter's reading: an 8x grid from a 512-tap first
 *   stage and two short ones, each crest refined, kept as a running maximum.
 *   Tones that start abruptly read within 0.01 dB up to 0.45 fs and within
 *   -0.09 / +0.05 dB up to 0.485 fs. LoudnessMeter's true-peak readout uses
 *   it, and so does the Limiter's high-precision true-peak mode, which pays
 *   for its 262-sample reach in latency.
 *
 * The EBU Tech 3341 true-peak tolerances (+0.2/-0.4 dB, cases 15-23) hold for
 * all three.
 *
 * The reading loops of the estimator and of the analyzer (the largest grid
 * point of every interval and the crests refined around the grid points) run
 * on vectors, on the AVX2 kernels when the run-time dispatch is active, and
 * give the values of their scalar form on every path. A compiler vectorises
 * the scalar loop only when it may run the vertex's division and comparisons
 * for positions the source skips, which those that honour floating-point
 * traps do not: with those the readings cost two to three times as much.
 *
 * Dependencies: DspMath.h, SimdOps.h, detail/TruePeakKernels.inl.
 */

#include "DspMath.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

namespace dspark {

/**
 * @class TruePeakDetector
 * @brief Per-channel 4x-oversampled inter-sample peak estimator.
 *
 * Real-time safe: the polyphase coefficients are compile-time constants and
 * processing is a fixed 48-multiply routine per sample (for float on SSE2 and
 * NEON, one 4-lane pass with the phases as lanes; elsewhere the scalar
 * per-phase dot products). The reading is the
 * maximum of the four official interpolation phases and the raw |sample|
 * (the raw sample can only pull the estimate toward the true analog peak,
 * never above it).
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Number of independent channel histories.
 */
template <FloatType T, int MaxChannels = 16>
class TruePeakDetector
{
public:
    /** @brief Clears all channel histories. Safe on the audio thread. */
    void reset() noexcept
    {
        for (auto& s : states_)
            s = {};
    }

    /**
     * @brief Feeds one sample and returns the local true-peak estimate.
     *
     * The result is the maximum of |sample| and the absolute values of the
     * four official Annex 2 interpolation phases evaluated at this position
     * (together they tile the 4x-oversampled reconstruction grid).
     *
     * @param sample  Input sample.
     * @param channel Channel index (0-based; clamped into range).
     * @return True-peak magnitude (>= |sample|).
     */
    [[nodiscard]] T processSample(T sample, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels);
        // Release-safe clamp to the nearest valid channel. (Redirecting an
        // out-of-range index to channel 0 would corrupt that channel's
        // history with a foreign stream.)
        channel = std::clamp(channel, 0, MaxChannels - 1);
        return processState(sample, states_[static_cast<size_t>(channel)]);
    }

    /** @brief Measures the zero-extended interpolation tail of one channel.
     * Evaluates getTaps()-1 zero frames on a stack copy of that channel's state.
     * Does not advance or clear live history. Call from the stream owner, not
     * concurrently with processing/reset. Allocation-free and bounded work.
     * Returns only the tail peak; combine with the maximum already observed.
     */
    [[nodiscard]] T getTailPeak(int channel) const noexcept
    {
        assert(channel >= 0 && channel < MaxChannels);
        channel = std::clamp(channel, 0, MaxChannels - 1);
        auto state = states_[static_cast<size_t>(channel)];
        T peak = T(0);
        for (int i = 0; i < kTaps - 1; ++i)
            peak = std::max(peak, processState(T(0), state));
        return peak;
    }

    /** @brief Group delay of the interpolation FIR in samples (per phase). */
    [[nodiscard]] static constexpr int getLatency() noexcept { return kTaps / 2; }

    /**
     * @brief Input samples one estimate is computed from (taps per phase).
     *
     * The reading returned for sample n was formed from x[n-getTaps()+1] .. x[n].
     * A caller that wants to change a reported peak has to change all of them:
     * a gain applied to only part of that span does not scale the estimate by
     * that gain.
     */
    [[nodiscard]] static constexpr int getTaps() noexcept { return kTaps; }

    /** @brief Number of interpolation phases (4x over-sampling). */
    [[nodiscard]] static constexpr int getPhases() noexcept { return kPhases; }

    /**
     * @brief One official interpolation phase, row reversed, for callers that
     *        run the interpolator as a block FIR.
     *
     * Element j multiplies x[n - getTaps() + 1 + j] (the window oldest first).
     * The maximum of the absolute values of the four phases and of |x[n]| is
     * the reading processSample() returns, up to summation rounding.
     *
     * @param phase Phase index, clamped to [0, getPhases() - 1].
     * @return getTaps() coefficients in read-only storage.
     */
    [[nodiscard]] static constexpr const T* getReversedPhase(int phase) noexcept
    {
        return kReversedCoeffs[static_cast<size_t>(std::clamp(phase, 0, kPhases - 1))].data();
    }

private:
    static constexpr int kTaps     = 12;  ///< Taps per polyphase branch.
    static constexpr int kPhases   = 4;   ///< Interpolation phases (4x oversampling).
    static constexpr int kHistSize = 16;  ///< Power-of-two ring length per channel.
    static constexpr int kHistMask = kHistSize - 1;

    struct State
    {
        /// Mirrored ring (2x length): sample i is stored at i and i + 16 so
        /// any 12-sample window ending inside the ring reads contiguously.
        T history[kHistSize * 2] = {};
        int writePos = 0;
    };

    [[nodiscard]] static T processState(T sample, State& tp) noexcept
    {
        // Mirroring makes the latest FIR window contiguous for the shared dot product.
        tp.history[static_cast<size_t>(tp.writePos)] = sample;
        tp.history[static_cast<size_t>(tp.writePos + kHistSize)] = sample;
        const int newest = tp.writePos;
        tp.writePos = (tp.writePos + 1) & kHistMask;
        const T* window = &tp.history[static_cast<size_t>(newest + kHistSize - (kTaps - 1))];
        T peak = std::abs(sample);
#if defined(DSPARK_SIMD_SSE2) || defined(DSPARK_SIMD_NEON)
        if constexpr (std::is_same_v<T, float>)
        {
            // The four phases are four lanes: one pass over the 12 taps with
            // the transposed table, then one horizontal max (3.6x faster than
            // four horizontal dot products). The sums run in tap order, so the
            // reading differs from the scalar path only by rounding (< 2e-7).
#if defined(DSPARK_SIMD_SSE2)
            __m128 acc = _mm_setzero_ps();
            for (int j = 0; j < kTaps; ++j)
                acc = _mm_add_ps(acc, _mm_mul_ps(_mm_load_ps(kVertical.c[j]), _mm_set1_ps(window[j])));
            acc = _mm_andnot_ps(_mm_set1_ps(-0.0f), acc);
            __m128 m = _mm_max_ps(acc, _mm_movehl_ps(acc, acc));
            m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
            const float a = _mm_cvtss_f32(m);
#else
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int j = 0; j < kTaps; ++j)
                acc = vaddq_f32(acc, vmulq_f32(vld1q_f32(kVertical.c[j]), vdupq_n_f32(window[j])));
            acc = vabsq_f32(acc);
            float32x2_t m = vpmax_f32(vget_low_f32(acc), vget_high_f32(acc));
            m = vpmax_f32(m, m);
            const float a = vget_lane_f32(m, 0);
#endif
            return (a > peak) ? a : peak;
        }
#endif
        for (int phase = 0; phase < kPhases; ++phase)
        {
            const T interp = simd::dotProduct(
                kReversedCoeffs[static_cast<size_t>(phase)].data(), window, kTaps);
            const T a = std::abs(interp);
            if (a > peak) peak = a;
        }
        return peak;
    }

    /** @brief Official ITU-R BS.1770-5 Annex 2 polyphase interpolator.
     *
     * Verbatim from the Annex 2 table ("one set of filter coefficients (for
     * the order 48, 4-phase, FIR interpolating) that would satisfy the
     * requirements"). Row k of phase p is h[4k+p] of the symmetric 48-tap
     * low-pass and multiplies x[n-k] (k growing into the past).
     *
     * Stored with each row REVERSED so the convolution runs as a forward
     * dot product over the oldest-first contiguous window:
     * sum_j window[j] * row[j] with window[j] = x[n-11+j]. */
    [[nodiscard]] static constexpr std::array<std::array<T, kTaps>, kPhases>
    buildReversedPhaseTable() noexcept
    {
        constexpr double kAnnex2[kPhases][kTaps] = {
            {  0.0017089843750,  0.0109863281250, -0.0196533203125,
               0.0332031250000, -0.0594482421875,  0.1373291015625,
               0.9721679687500, -0.1022949218750,  0.0476074218750,
              -0.0266113281250,  0.0148925781250, -0.0083007812500 },
            { -0.0291748046875,  0.0292968750000, -0.0517578125000,
               0.0891113281250, -0.1665039062500,  0.4650878906250,
               0.7797851562500, -0.2003173828125,  0.1015625000000,
              -0.0582275390625,  0.0330810546875, -0.0189208984375 },
            { -0.0189208984375,  0.0330810546875, -0.0582275390625,
               0.1015625000000, -0.2003173828125,  0.7797851562500,
               0.4650878906250, -0.1665039062500,  0.0891113281250,
              -0.0517578125000,  0.0292968750000, -0.0291748046875 },
            { -0.0083007812500,  0.0148925781250, -0.0266113281250,
               0.0476074218750, -0.1022949218750,  0.9721679687500,
               0.1373291015625, -0.0594482421875,  0.0332031250000,
              -0.0196533203125,  0.0109863281250,  0.0017089843750 },
        };

        std::array<std::array<T, kTaps>, kPhases> result {};
        for (int phase = 0; phase < kPhases; ++phase)
            for (int k = 0; k < kTaps; ++k)
                result[static_cast<size_t>(phase)][static_cast<size_t>(kTaps - 1 - k)]
                    = static_cast<T>(kAnnex2[phase][k]);
        return result;
    }

    /// Compile-time table in read-only storage: no magic-static guard on the
    /// audio thread, nothing to initialise at run time.
    static constexpr std::array<std::array<T, kTaps>, kPhases> kReversedCoeffs =
        buildReversedPhaseTable();

    /// The same table transposed for the 4-lane pass: c[j][p] multiplies
    /// window[j] for phase p.
    struct VerticalTable { alignas(16) float c[kTaps][kPhases]; };
    [[nodiscard]] static constexpr VerticalTable buildVerticalTable() noexcept
    {
        VerticalTable v {};
        for (int j = 0; j < kTaps; ++j)
            for (int p = 0; p < kPhases; ++p)
                v.c[j][p] = static_cast<float>(kReversedCoeffs[static_cast<size_t>(p)][static_cast<size_t>(j)]);
        return v;
    }
    static constexpr VerticalTable kVertical = buildVerticalTable();

    std::array<State, MaxChannels> states_ {};
};

namespace detail {

// Kaiser-windowed sinc interpolation phase (beta `beta`, cut at Nyquist),
// normalised to unity at DC: `out[j]` multiplies window sample j of a
// `Count`-sample window and the phase reads the point `frac` (0 < frac < 1)
// past window sample Count / 2 - 1. Designed in double.
template <int Count, typename T>
inline void designSincPhase(T* out, double beta, double frac) noexcept
{
    constexpr double kPi = std::numbers::pi_v<double>;
    const auto besselI0 = [](double x) noexcept {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k < 64; ++k)
        {
            const double h = x / (2.0 * k);
            term *= h * h;
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        return sum;
    };
    const double norm = besselI0(beta);
    constexpr double halfTaps = Count / 2;
    double taps[Count];
    double sum = 0.0;
    for (int j = 0; j < Count; ++j)
    {
        // Distance from tap j to the interpolation point.
        const double d = static_cast<double>(j) - (halfTaps - 1.0) - frac;
        const double sinc = std::sin(kPi * d) / (kPi * d);
        const double u = d / halfTaps;
        const double w = besselI0(beta * std::sqrt(std::max(0.0, 1.0 - u * u))) / norm;
        taps[j] = sinc * w;
        sum += taps[j];
    }
    for (int j = 0; j < Count; ++j)
        out[j] = static_cast<T>(taps[j] / sum);
}

// Raises m to the vertex of the parabola through (a, b, c) when b is a
// local maximum of the magnitude there (never by more than a quarter of
// |b|). No multiply feeds an add, so the result does not depend on
// floating-point contraction.
template <typename T>
[[nodiscard]] inline T refineVertex(T a, T b, T c, T m) noexcept
{
    const T ab = std::abs(b);
    const T d = (a + c) - (b + b);
    if (ab >= std::abs(a) && ab >= std::abs(c) && b * d < T(0))
    {
        const T q = c - a;
        const T v = std::abs(b - (q * q) / (T(8) * d));
        if (v > m) m = v;
    }
    return m;
}

// The reading of position i of a 4x grid: the largest magnitude among the
// five grid points of the interval from sample s[i] to s[i + 1] (its quarter,
// half and three-quarter points at q[i], h[i] and t[i + 1]) and among the
// crests refined around the four inner ones, the first with the
// three-quarter point before the interval, t[i].
template <typename T>
[[nodiscard]] inline T gridReading(const T* s, const T* q, const T* h, const T* t, int i) noexcept
{
    const T a0 = s[i], a4 = s[i + 1];
    const T y0 = t[i], y1 = q[i], y2 = h[i], y3 = t[i + 1];
    T m = std::max(std::max(std::abs(a0), std::abs(a4)),
                   std::max(std::abs(y2), std::max(std::abs(y1), std::abs(y3))));
    m = refineVertex(y0, a0, y1, m);
    m = refineVertex(a0, y1, y2, m);
    m = refineVertex(y1, y2, y3, m);
    m = refineVertex(y2, y3, a4, m);
    return m;
}

// Raises m to the two crests that point u of an 8x grid completes. The grid
// alternates the points a[u] and e[u + 1] (e[0] is the point before a[0], and
// a[-1] the one before that): the crests are refined around e[u], between
// a[u - 1] and a[u], and around a[u], between e[u] and e[u + 1].
template <typename T>
[[nodiscard]] inline T eighthCrests(const T* a, const T* e, int u, T m) noexcept
{
    m = refineVertex(a[u - 1], e[u], a[u], m);
    return refineVertex(e[u], a[u], e[u + 1], m);
}

// Raises peak[i] to the reading of sample i of an 8x grid, for i in
// [first, n): the largest magnitude among the sample's four pairs of points
// (4 i .. 4 i + 3) and the crests they complete (eighthCrests()), the two
// points before each pair carried from one to the next. Magnitudes that are
// not numbers are ignored.
template <typename T>
inline void raiseToEighthSamples(const T* a, const T* e, T* peak, int first, int n) noexcept
{
    if (first >= n) return;
    T p0 = a[4 * first - 1], p1 = e[4 * first];
    for (int i = first; i < n; ++i)
    {
        T m = T(0);
        for (int k = 0; k < 4; ++k)
        {
            const T x = a[4 * i + k];
            const T y = e[4 * i + k + 1];
            m = std::max(m, std::abs(x));
            m = std::max(m, std::abs(y));
            m = refineVertex(p0, p1, x, m);
            m = refineVertex(p1, x, y, m);
            p0 = x;
            p1 = y;
        }
        peak[i] = std::max(peak[i], m);
    }
}

// The reading loops on vectors, compiled for the build's baseline
// (detail/TruePeakKernels.inl). A compiler vectorises the scalar loop only
// when it may run the vertex's division and comparisons for positions the
// source skips, which those that honour floating-point traps do not.
namespace truepeak {
template <typename T, int W> using Vec = simd::Vec<T, W>;
template <typename T> inline constexpr int kWide = simd::kVecWidth<T>;
template <typename T> inline constexpr int kNarrow = simd::kVecNarrowWidth<T>;
#include "detail/TruePeakKernels.inl"
} // namespace truepeak

} // namespace detail
} // namespace dspark

#if defined(DSPARK_SIMD_DISPATCH_AVX2)
// The same loops compiled for AVX2+FMA, run when isRuntimeDispatchActive().
DSPARK_SIMD_AVX2_BEGIN
namespace dspark {
namespace detail {
namespace truepeak {
namespace avx2 {
template <typename T, int W> using Vec = simd::avx2::Vec<T, W>;
template <typename T> inline constexpr int kWide = simd::avx2::kVecWidth<T>;
template <typename T> inline constexpr int kNarrow = simd::avx2::kVecNarrowWidth<T>;
#include "detail/TruePeakKernels.inl"
} // namespace avx2
} // namespace truepeak
} // namespace detail
} // namespace dspark
DSPARK_SIMD_AVX2_END
#endif

namespace dspark {
namespace detail {

// The reading loops (detail/TruePeakKernels.inl) on the kernel copy in use:
// the AVX2+FMA one when the run-time dispatch is active. Every copy returns
// the same values, bit for bit, so a stream may change copy between calls.
template <typename T>
struct TruePeakLoops
{
    /// Raises peak[i] to gridReading(s, q, h, t, i) for i in [0, n).
    static void raiseToReadings(const T* s, const T* q, const T* h, const T* t, T* peak, int n) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            truepeak::avx2::raiseToReadings(s, q, h, t, peak, n);
            return;
        }
#endif
        truepeak::raiseToReadings(s, q, h, t, peak, n);
    }

    /// The largest finite gridReading() of the positions [0, n) and start.
    [[nodiscard]] static T largestFiniteReading(const T* s, const T* q, const T* h, const T* t, int n,
                                                T start) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            return truepeak::avx2::largestFiniteReading(s, q, h, t, n, start);
#endif
        return truepeak::largestFiniteReading(s, q, h, t, n, start);
    }

    /// Raises dst[i] to |src[i]| for i in [0, n).
    static void raiseToMagnitudes(T* dst, const T* src, int n) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            truepeak::avx2::raiseToMagnitudes(dst, src, n);
            return;
        }
#endif
        truepeak::raiseToMagnitudes(dst, src, n);
    }

    /// The largest finite magnitude among count values and start.
    [[nodiscard]] static T largestFiniteMagnitude(const T* v, int count, T start) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            return truepeak::avx2::largestFiniteMagnitude(v, count, start);
#endif
        return truepeak::largestFiniteMagnitude(v, count, start);
    }

    /// Raises peak[i] to the reading of sample i of an 8x grid (see
    /// raiseToEighthSamples()) for i in [0, n).
    static void raiseToEighthReadings(const T* a, const T* e, T* peak, int n) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
        {
            truepeak::avx2::raiseToEighthReadings(a, e, peak, n);
            return;
        }
#endif
        truepeak::raiseToEighthReadings(a, e, peak, n);
    }

    /// The largest finite crest of the points [0, points) of an 8x grid and
    /// start.
    [[nodiscard]] static T largestFiniteEighthCrest(const T* a, const T* e, int points, T start) noexcept
    {
#if defined(DSPARK_SIMD_DISPATCH_AVX2)
        if (simd::isRuntimeDispatchActive())
            return truepeak::avx2::largestFiniteEighthCrest(a, e, points, start);
#endif
        return truepeak::largestFiniteEighthCrest(a, e, points, start);
    }
};

} // namespace detail

/**
 * @class TruePeakEstimator
 * @brief Block-based reading of the band-limited reconstruction peak.
 *
 * The BS.1770 Annex 2 procedure (4x over-sampling, low-pass interpolation,
 * absolute value) with a longer interpolator than the Annex 2 example table:
 * the reconstruction is evaluated at the quarter, half and three-quarter
 * points of every sample interval with a 64-tap Kaiser-windowed sinc per
 * phase (beta 8, cut at Nyquist, each phase normalised to unity at DC), and
 * every local maximum of that 4x grid is refined with the vertex of the
 * parabola through it and its two neighbours, which removes most of the grid
 * error (a 4x grid alone misses a crest at 0.45 fs by up to 0.55 dB).
 *
 * Measured against the exact reconstruction sum of the zero-extended signal
 * (the Annex 2 table in brackets): steady sines read within 0.003 dB up to
 * 0.49 fs (+0.22 dB); tones up to 0.45 fs that start abruptly within
 * -0.24 / +0.07 dB (-0.78 / +0.19); noise band-limited to 0.45 fs within
 * -0.03 / +0.01 dB (-0.37 / +0.04). Content between 0.45 fs and Nyquist is
 * beyond any short kernel: tones there that start abruptly read
 * -1.1 / +0.75 dB (-2.8 / +0.23) and white noise -0.55 / +0.47 dB
 * (-1.08 / +0.40); the table reads that band low because it attenuates it.
 * A crest there is built by samples hundreds of positions away, so a reading
 * that must answer within getDelay() samples cannot follow it: meters, which
 * can wait, read with TruePeakAnalyzer.
 *
 * Each reading at input sample n describes the interval x[n - 32] .. x[n - 31]
 * and the half grid step before it, so it reaches back getDelay() samples; a
 * finite programme is completed with getTailPeak().
 *
 * Cost per sample of one channel in 128-sample chunks of white noise, with
 * the Annex 2 guard, measured on hosted CI runners (before the reading loops
 * ran on vectors in brackets): 7 to 10 ns in float and 14 to 20 in double on
 * x86-64 with the run-time dispatch (GCC, Clang and MSVC; 10 to 26 and 24 to
 * 34), 13 to 16 and 25 to 33 on SSE2 (16 to 26 and 33 to 40), and 16 to 20
 * and 33 to 40 on an arm64 server core (17 to 31 and 33 to 46). processMax()
 * also skips every vector whose grid points cannot reach the maximum it
 * already holds, and costs 0.37 to 0.83 of what it did.
 *
 * Threading: a channel's state and the estimator's scratch belong to the
 * stream owner; process() and getTailPeak() are not re-entrant. No
 * allocation; the kernel is designed by the constructor.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TruePeakEstimator
{
public:
    static constexpr int kTaps = 64;   ///< Taps per interpolation phase.
    static constexpr int kPhases = 3;  ///< Quarter, half and three-quarter points.
    static constexpr int kChunk = 128; ///< Most samples one process() call takes.

    /** @brief History of one channel (kTaps - 1 samples plus one chunk). */
    struct Channel
    {
        T history[kTaps - 1 + kChunk] = {}; ///< Oldest first; see process().
        int fill = 0;                       ///< Samples past the history prefix.
        T last = T(0);                      ///< Last three-quarter point.
    };

    /** @brief Designs the interpolation phases. */
    TruePeakEstimator() noexcept
    {
        for (int p = 0; p < kPhases; ++p)
            detail::designSincPhase<kTaps>(coeffs_[p], 8.0, static_cast<double>(p + 1) / 4.0);
    }

    /** @brief Input samples one reading reaches back. */
    [[nodiscard]] static constexpr int getDelay() noexcept { return kTaps / 2 + 1; }

    /**
     * @brief Appends samples to a channel and raises each peak[i] to the
     *        reading of x[i]'s position.
     *
     * NaN readings are ignored (peak[i] keeps its value).
     *
     * @param channel       The channel's history (owned by the caller).
     * @param x             Input samples.
     * @param n             Sample count, 1 .. kChunk.
     * @param peak          Readings to raise, n values.
     * @param annex2Guard   Also raises peak[i] to the BS.1770 Annex 2 table's
     *                      phases over the same window, so a meter built on
     *                      that table never reads above the result either.
     */
    void process(Channel& channel, const T* x, int n, T* peak, bool annex2Guard = false) noexcept
    {
        assert(n >= 1 && n <= kChunk);
        n = std::clamp(n, 0, kChunk);
        const T* base = appendAndInterpolate(channel, x, n);

        if (annex2Guard)
        {
            // BS.1770 Annex 2 phases over the newest kAnnexTaps samples.
            const T* annexBase = base + (kTaps - kAnnexTaps);
            for (int p = 0; p < kAnnexPhases; ++p)
            {
                simd::firCorrelate(annexBase, Annex::getReversedPhase(p), kAnnexTaps, annex_, n, T(1));
                Loops::raiseToMagnitudes(peak, annex_, n); // NaN is ignored
            }
        }
        Loops::raiseToReadings(base + (kTaps / 2 - 1), quarter_, half_, threeQuarter(), peak, n); // NaN is ignored
        channel.fill += n;
    }

    /**
     * @brief Appends samples to a channel and returns the largest finite
     *        reading among them (0 if none), as process() would read them.
     *
     * For a running maximum: when no grid point of the chunk exceeds
     * atLeast / 1.25 (a crest refinement raises a grid point by at most a
     * quarter), the refinement is skipped and the grid maximum is returned;
     * otherwise every reading is evaluated exactly as process() does. The
     * history advances the same way in both cases.
     *
     * @param channel The channel's history (owned by the caller).
     * @param x       Input samples.
     * @param n       Sample count, 1 .. kChunk.
     * @param atLeast The running maximum the caller already holds.
     * @return The chunk's largest finite reading, exact whenever it can
     *         exceed atLeast.
     */
    [[nodiscard]] T processMax(Channel& channel, const T* x, int n, T atLeast) noexcept
    {
        assert(n >= 1 && n <= kChunk);
        n = std::clamp(n, 0, kChunk);
        const T* base = appendAndInterpolate(channel, x, n);
        channel.fill += n;
        // Grid points of the chunk: the samples x[i - 32] .. x[i - 31] and the
        // three interpolated points between them.
        const T* const samples = base + (kTaps / 2 - 1);
        T gridMax = Loops::largestFiniteMagnitude(samples, n + 1, T(0));
        gridMax = Loops::largestFiniteMagnitude(quarter_, n, gridMax);
        gridMax = Loops::largestFiniteMagnitude(half_, n, gridMax);
        gridMax = Loops::largestFiniteMagnitude(threeQuarter() + 1, n, gridMax);
        if (!(gridMax * T(1.25) > atLeast))
            return gridMax;
        return Loops::largestFiniteReading(samples, quarter_, half_, threeQuarter(), n, gridMax);
    }

    /**
     * @brief Peak of the zero-extended tail of a finite programme.
     *
     * Reads kTaps zero samples on a copy of the channel, so every interval
     * the last samples still reach is measured; the live history is not
     * changed. Returns only the tail; combine it with the readings already
     * taken. Same threading as process().
     *
     * @param channel The channel's history.
     */
    [[nodiscard]] T getTailPeak(const Channel& channel) noexcept
    {
        Channel copy = channel;
        T zeros[kTaps] = {};
        T peak[kTaps] = {};
        process(copy, zeros, kTaps, peak);
        T result = T(0);
        for (const T v : peak) result = std::max(result, v);
        return result;
    }

private:
    using Annex = TruePeakDetector<T, 1>;
    using Loops = detail::TruePeakLoops<T>;
    static constexpr int kAnnexTaps = Annex::getTaps();
    static constexpr int kAnnexPhases = Annex::getPhases();
    static_assert(kAnnexTaps <= kTaps, "the BS.1770 window must fit the estimator history");
    static_assert(kTaps <= kChunk, "the tail is measured in one call");

    /** Appends n (<= kChunk) samples to the channel history and evaluates the
     *  three interpolation phases for them; returns the window base (the
     *  oldest of the kTaps - 1 + n samples the phases read). */
    const T* appendAndInterpolate(Channel& channel, const T* x, int n) noexcept
    {
        T* h = channel.history;
        if (channel.fill + n > kChunk)
        {
            std::copy(h + channel.fill, h + channel.fill + (kTaps - 1), h); // keep the last kTaps - 1 samples
            channel.fill = 0;
        }
        std::copy(x, x + n, h + (kTaps - 1) + channel.fill);
        const T* base = h + channel.fill;
        simd::firCorrelate(base, coeffs_[0], kTaps, quarter_, n, T(1));
        simd::firCorrelate(base, coeffs_[1], kTaps, half_, n, T(1));
        // The three-quarter point before the chunk sits ahead of the chunk's,
        // so every interval reads the one before it from the same array.
        T* const t = threeQuarter();
        t[0] = channel.last;
        simd::firCorrelate(base, coeffs_[2], kTaps, t + 1, n, T(1));
        channel.last = t[n];
        return base;
    }

    /// The chunk's three-quarter points start 32 bytes into their array, as
    /// aligned as the arrays beside it, and the point before the chunk takes
    /// the slot before them: threeQuarter()[0] is that point and
    /// threeQuarter()[1 + i] the chunk's.
    static constexpr int kLead = 32 / static_cast<int>(sizeof(T));
    [[nodiscard]] T* threeQuarter() noexcept { return threeQuarter_ + (kLead - 1); }

    T coeffs_[kPhases][kTaps] = {};
    T quarter_[kChunk] = {};          ///< The chunk's quarter points.
    T half_[kChunk] = {};             ///< The chunk's half points.
    T threeQuarter_[kLead + kChunk] = {}; ///< The chunk's three-quarter points, from [kLead] on.
    T annex_[kChunk] = {};            ///< One Annex 2 phase.
};

/**
 * @class TruePeakAnalyzer
 * @brief Running-maximum reading of the band-limited reconstruction peak, for
 *        meters.
 *
 * The BS.1770 Annex 2 procedure (over-sampling, low-pass interpolation,
 * absolute value) carried to 8x in three stages: a 512-tap Kaiser-windowed
 * sinc (beta 7) reads the midpoint of every sample interval, a 16-tap one
 * (beta 9) the midpoints of that 2x stream and an 8-tap one (beta 9) the
 * midpoints of the 4x stream (the 2x stream fills only the lower half of
 * its band and the 4x stream the lower quarter, so short kernels are exact
 * there). The vertex of the parabola through every local maximum of the 8x
 * grid refines the crest. The long first stage is what reads content near
 * Nyquist: the transition band of an interpolator is as wide as its kernel
 * is short, and a tone inside it is reconstructed with the wrong phase
 * between samples, so a kernel that reaches 32 samples (TruePeakEstimator)
 * misplaces every crest a tone above 0.45 fs builds where it starts or
 * stops.
 *
 * Measured against the exact reconstruction sum of the zero-extended signal
 * (TruePeakEstimator in brackets): steady sines up to 0.49 fs read within
 * 0.004 dB (0.002); tones that start abruptly within 0.01 dB up to 0.45 fs
 * (-0.22 / +0.06), -0.09 / +0.05 dB up to 0.485 fs (-0.82 / +0.75) and
 * -0.21 / +0.08 dB at 0.49 fs (-1.2 / +0.11); noise band-limited to 0.45 fs
 * within 0.003 dB (0.02); white noise -0.19 / +0.13 dB (-0.55 / +0.47);
 * noise with all its energy above 0.40 fs -0.27 / +0.20 dB (-0.98 / +0.63).
 * What is left belongs to the last 0.01 fs below Nyquist, where halving the
 * error again doubles the kernel.
 *
 * Readings are taken in chunks and only the largest is kept (processMax()),
 * so a meter's max-hold is exact while chunks that cannot raise it skip the
 * crest refinement, and within a chunk that can, so does every vector of
 * points that cannot. A reading reaches getDelay() samples back; a finite
 * programme is completed with getTailPeak(). process() gives the same
 * readings one per input sample to a detector that can wait that long for
 * them (the Limiter's high-precision true-peak mode); the detectors that
 * must answer within a few dozen samples use TruePeakEstimator instead.
 *
 * Threading: a channel's state and the analyzer's scratch belong to the
 * stream owner; processMax(), process() and getTailPeak() are not
 * re-entrant. No allocation; the kernels are designed by the constructor.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TruePeakAnalyzer
{
    static constexpr int kTaps2 = 16;  ///< Taps of the 2x-to-4x stage.
    static constexpr int kTaps3 = 8;   ///< Taps of the 4x-to-8x stage.

public:
    static constexpr int kTaps = 512;  ///< Taps of the first (interval midpoint) stage.
    static constexpr int kChunk = 128; ///< Most samples one processMax() call takes.

    /** @brief History of one channel (each stage's kernel span plus one chunk). */
    struct Channel
    {
        T input[kTaps - 1 + kChunk] = {};             ///< Input samples, oldest first.
        T twice[kTaps2 - 1 + 2 * kChunk] = {};        ///< 2x stream, oldest first.
        T fourTimes[kTaps3 - 1 + 4 * kChunk] = {};    ///< 4x stream, oldest first.
        int fill = 0;                                 ///< Input samples past the history prefix.
        T previous[2] = {};                           ///< Last two points of the 8x grid.
    };

    /** @brief Designs the three interpolation stages. */
    TruePeakAnalyzer() noexcept
    {
        detail::designSincPhase<kTaps>(coeffs1_, 7.0, 0.5);
        detail::designSincPhase<kTaps2>(coeffs2_, 9.0, 0.5);
        detail::designSincPhase<kTaps3>(coeffs3_, 9.0, 0.5);
    }

    /** @brief Input samples one reading reaches back: the newest 8x grid
     *         point a call reads lies this many samples before the newest
     *         input sample, at most. */
    [[nodiscard]] static constexpr int getDelay() noexcept
    {
        return kDelay;
    }

    /**
     * @brief Appends samples to a channel and returns the largest finite
     *        reading among the 8x grid points they complete (0 if none).
     *
     * Every crest refined by a call is centred on one of its grid points or
     * on the last point before them, and a refinement raises its centre by at
     * most a quarter. When none of those points exceeds atLeast / 1.25 the
     * refinement is therefore skipped and the grid maximum is returned;
     * otherwise every crest is refined. Either way the history advances the
     * same way, so the running maximum of a stream does not depend on how it
     * is cut into calls.
     *
     * @param channel The channel's history (owned by the caller).
     * @param x       Input samples.
     * @param n       Sample count, 1 .. kChunk.
     * @param atLeast The running maximum the caller already holds.
     * @return The chunk's largest finite reading, exact whenever it can
     *         exceed atLeast.
     */
    [[nodiscard]] T processMax(Channel& channel, const T* x, int n, T atLeast) noexcept
    {
        assert(n >= 1 && n <= kChunk);
        n = std::clamp(n, 0, kChunk);
        if (n == 0) return T(0);
        const T* fourth = advance(channel, x, n); // 4n points of the 4x stream
        const int points = 4 * n;
        constexpr T kFinite = std::numeric_limits<T>::max();
        T gridMax = Loops::largestFiniteMagnitude(fourth, points, T(0));
        gridMax = Loops::largestFiniteMagnitude(eighth() + 1, points, gridMax);
        // The first crest of the call is centred on the point carried over.
        const T carried = std::abs(eighth()[0]);
        const T centres = carried <= kFinite ? std::max(gridMax, carried) : gridMax;
        if (!(centres * T(1.25) > atLeast))
            return gridMax;
        return Loops::largestFiniteEighthCrest(fourth, eighth(), points, gridMax);
    }

    /**
     * @brief Appends samples to a channel and raises each peak[i] to the
     *        reading of the 8x grid points x[i] completes.
     *
     * The per-sample form, for a detector that can wait for it: the reading
     * for input sample n covers the reconstruction from x[n - getDelay()] to
     * the last eighth point before x[n - getDelay() + 1], plus the crest
     * centred on the eighth point before x[n - getDelay()], so it concerns
     * the samples n - getDelay() - 1 .. n - getDelay() + 1. Every reading is
     * the same arithmetic however the stream is cut into calls, and for
     * finite input the largest of them is what processMax() returns. NaN
     * readings are ignored (peak[i] keeps its value).
     *
     * @param channel     The channel's history (owned by the caller).
     * @param x           Input samples.
     * @param n           Sample count, 1 .. kChunk.
     * @param peak        Readings to raise, n values.
     * @param annex2      When not null, n values raised to the BS.1770 Annex 2
     *                    table's phases over x[i - 11] .. x[i]: what a meter
     *                    built on that table reads at x[i] between samples.
     *                    Kept apart because that reading sits 6 samples
     *                    back, not getDelay().
     */
    void process(Channel& channel, const T* x, int n, T* peak, T* annex2 = nullptr) noexcept
    {
        assert(n >= 1 && n <= kChunk);
        n = std::clamp(n, 0, kChunk);
        if (n == 0) return;
        const T* fourth = advance(channel, x, n); // 4n points of the 4x stream

        if (annex2 != nullptr)
        {
            // BS.1770 Annex 2 phases over the newest kAnnexTaps input samples.
            const T* annexBase = channel.input + (channel.fill - n) + (kTaps - kAnnexTaps);
            for (int p = 0; p < kAnnexPhases; ++p)
            {
                simd::firCorrelate(annexBase, Annex::getReversedPhase(p), kAnnexTaps, scratch_, n, T(1));
                Loops::raiseToMagnitudes(annex2, scratch_, n); // NaN is ignored
            }
        }
        Loops::raiseToEighthReadings(fourth, eighth(), peak, n); // NaN is ignored
    }

    /**
     * @brief Peak of the zero-extended tail of a finite programme.
     *
     * Reads enough zero samples on a copy of the channel that every grid
     * point the last samples still reach is measured; the live history is
     * not changed. Returns only the tail; combine it with the readings
     * already taken. Same threading as processMax().
     *
     * @param channel The channel's history.
     */
    [[nodiscard]] T getTailPeak(const Channel& channel) noexcept
    {
        Channel copy = channel;
        constexpr T zeros[kChunk] = {};
        T result = T(0);
        for (int left = kTailSamples; left > 0; left -= kChunk)
        {
            const T peak = processMax(copy, zeros, std::min(left, kChunk), result);
            if (peak > result) result = peak;
        }
        return result;
    }

private:
    using Annex = TruePeakDetector<T, 1>;
    using Loops = detail::TruePeakLoops<T>;
    static constexpr int kAnnexTaps = Annex::getTaps();
    static constexpr int kAnnexPhases = Annex::getPhases();
    static_assert(kAnnexTaps <= kTaps, "the BS.1770 window must fit the first stage history");

    /// Half of each stage's kernel, in input samples.
    static constexpr int kDelay = kTaps / 2 + kTaps2 / 4 + kTaps3 / 8;
    /// Zero samples after which the last input sample no longer reaches any
    /// grid point: its first-stage reach plus the reading delay.
    static constexpr int kTailSamples = kTaps / 2 + kDelay + 1;
    static_assert(kTaps % 2 == 0 && kTaps2 % 2 == 0 && kTaps3 % 2 == 0, "midpoint kernels are even");

    /** Appends n (<= kChunk) samples and runs the three stages: leaves the
     *  4n eighth points in eighth() from [1] on, after the last one of the
     *  call before, and returns the 4n points of the 4x stream they follow
     *  (4x point u of this call at [u], the last one before them at [-1]). */
    const T* advance(Channel& channel, const T* x, int n) noexcept
    {
        if (channel.fill + n > kChunk)
        {
            // Keep the last kernel span of each stage.
            const int f = channel.fill;
            std::copy(channel.input + f, channel.input + f + (kTaps - 1), channel.input);
            std::copy(channel.twice + 2 * f, channel.twice + 2 * f + (kTaps2 - 1), channel.twice);
            std::copy(channel.fourTimes + 4 * f, channel.fourTimes + 4 * f + (kTaps3 - 1), channel.fourTimes);
            channel.fill = 0;
        }
        const int f = channel.fill;

        // Stage 1: the midpoint of every interval, from the input.
        std::copy(x, x + n, channel.input + (kTaps - 1) + f);
        const T* in = channel.input + f;
        simd::firCorrelate(in, coeffs1_, kTaps, scratch_, n, T(1));
        T* twice = channel.twice + (kTaps2 - 1) + 2 * f;
        for (int i = 0; i < n; ++i)
        {
            twice[2 * i] = in[i + kTaps / 2 - 1];
            twice[2 * i + 1] = scratch_[i];
        }

        // Stage 2: the midpoints of the 2x stream.
        const T* two = channel.twice + 2 * f;
        simd::firCorrelate(two, coeffs2_, kTaps2, scratch_, 2 * n, T(1));
        T* four = channel.fourTimes + (kTaps3 - 1) + 4 * f;
        for (int t = 0; t < 2 * n; ++t)
        {
            four[2 * t] = two[t + kTaps2 / 2 - 1];
            four[2 * t + 1] = scratch_[t];
        }

        // Stage 3: the midpoints of the 4x stream. The eighth point before
        // the chunk sits ahead of the chunk's, so every crest reads the
        // points before it from the same arrays (the 4x point before the
        // chunk is in the history).
        const T* base = channel.fourTimes + 4 * f;
        T* const e = eighth();
        e[0] = channel.previous[1];
        simd::firCorrelate(base, coeffs3_, kTaps3, e + 1, 4 * n, T(1));
        const T* fourth = base + (kTaps3 / 2 - 1);
        channel.previous[0] = fourth[4 * n - 1];
        channel.previous[1] = e[4 * n];
        channel.fill += n;
        return fourth;
    }

    /// Stage 3's outputs start 32 bytes into their array, as aligned as the
    /// arrays beside it, and the eighth point before the chunk takes the slot
    /// before them: eighth()[0] is that point and eighth()[1 + u] the
    /// chunk's.
    static constexpr int kLead = 32 / static_cast<int>(sizeof(T));
    [[nodiscard]] T* eighth() noexcept { return eighth_ + (kLead - 1); }

    T coeffs1_[kTaps] = {};
    T coeffs2_[kTaps2] = {};
    T coeffs3_[kTaps3] = {};
    T scratch_[2 * kChunk] = {};  ///< Stage 1 and stage 2 outputs.
    T eighth_[kLead + 4 * kChunk] = {}; ///< Stage 3's outputs, from [kLead] on.
};

} // namespace dspark
