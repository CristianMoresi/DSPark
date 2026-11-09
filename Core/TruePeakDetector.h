// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file TruePeakDetector.h
 * @brief Shared ITU-R BS.1770 true-peak (inter-sample peak) detectors.
 *
 * Two readings of the analog reconstruction peak of a digital signal, both
 * following the BS.1770 Annex 2 procedure (4x over-sampling, low-pass
 * interpolation, absolute value):
 *
 * - TruePeakDetector runs the example interpolator printed in BS.1770-5
 *   Annex 2: the 48-tap, 4-phase polyphase FIR (12 taps per phase). It reads
 *   what a meter built on that table reads, including its pass-band ripple:
 *   steady sines read up to 0.22 dB above their true peak near 0.09 fs and
 *   0.25 fs, and the pass band ends near 0.41 fs. Use it to stay at or below
 *   a ceiling as such a meter judges it (Compressor, Limiter guard, the
 *   offline gain stages).
 * - TruePeakEstimator reads the reconstruction itself: 64-tap Kaiser-windowed
 *   sinc phases at the quarter points of every sample interval, and the
 *   vertex of the parabola through each local maximum of that 4x grid.
 *   Against the exact band-limited reconstruction it reads steady sines up to
 *   0.49 fs within 0.003 dB, and tones up to 0.45 fs that start abruptly
 *   within -0.24 / +0.07 dB (the 12-tap table: -0.78 / +0.19). LoudnessMeter's
 *   true-peak readout and the Limiter's true-peak mode use it.
 *
 * The EBU Tech 3341 true-peak tolerances (+0.2/-0.4 dB, cases 15-23) hold for
 * both.
 *
 * Dependencies: DspMath.h, SimdOps.h.
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
 *
 * Each reading at input sample n describes the interval x[n - 32] .. x[n - 31]
 * and the half grid step before it, so it reaches back getDelay() samples; a
 * finite programme is completed with getTailPeak().
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
        constexpr double kBeta = 8.0;
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
        const double norm = besselI0(kBeta);
        constexpr double halfTaps = kTaps / 2;
        for (int p = 0; p < kPhases; ++p)
        {
            const double frac = static_cast<double>(p + 1) / 4.0;
            double taps[kTaps];
            double sum = 0.0;
            for (int j = 0; j < kTaps; ++j)
            {
                // Distance from tap j to the interpolation point, which sits
                // frac past the window's sample halfTaps - 1.
                const double d = static_cast<double>(j) - (halfTaps - 1.0) - frac;
                const double sinc = std::sin(kPi * d) / (kPi * d);
                const double u = d / halfTaps;
                const double w = besselI0(kBeta * std::sqrt(std::max(0.0, 1.0 - u * u))) / norm;
                taps[j] = sinc * w;
                sum += taps[j];
            }
            for (int j = 0; j < kTaps; ++j)
                coeffs_[p][j] = static_cast<T>(taps[j] / sum);
        }
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
                simd::firCorrelate(annexBase, Annex::getReversedPhase(p), kAnnexTaps,
                                   phase_[kPhases], n, T(1));
                for (int i = 0; i < n; ++i)
                    peak[i] = std::max(peak[i], std::abs(phase_[kPhases][i])); // NaN is ignored
            }
        }
        T last = channel.last;
        for (int i = 0; i < n; ++i)
        {
            const T a0 = base[i + kTaps / 2 - 1];
            const T a4 = base[i + kTaps / 2];
            const T y1 = phase_[0][i], y2 = phase_[1][i], y3 = phase_[2][i];
            T m = std::max(std::max(std::abs(a0), std::abs(a4)),
                           std::max(std::abs(y2), std::max(std::abs(y1), std::abs(y3))));
            m = refineVertex(last, a0, y1, m);
            m = refineVertex(a0, y1, y2, m);
            m = refineVertex(y1, y2, y3, m);
            m = refineVertex(y2, y3, a4, m);
            last = y3;
            peak[i] = std::max(peak[i], m); // NaN is ignored
        }
        channel.last = last;
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
        constexpr T kFinite = std::numeric_limits<T>::max();
        // Grid points of the chunk: the samples x[i - 32] .. x[i - 31] and the
        // three interpolated points between them.
        T gridMax = T(0);
        for (int i = 0; i <= n; ++i)
        {
            const T a = std::abs(base[i + kTaps / 2 - 1]);
            if (a > gridMax && a <= kFinite) gridMax = a;
        }
        for (int p = 0; p < kPhases; ++p)
            for (int i = 0; i < n; ++i)
            {
                const T a = std::abs(phase_[p][i]);
                if (a > gridMax && a <= kFinite) gridMax = a;
            }
        if (!(gridMax * T(1.25) > atLeast))
        {
            channel.last = n > 0 ? phase_[2][n - 1] : channel.last;
            channel.fill += n;
            return gridMax;
        }
        T result = gridMax;
        T last = channel.last;
        for (int i = 0; i < n; ++i)
        {
            const T a0 = base[i + kTaps / 2 - 1];
            const T a4 = base[i + kTaps / 2];
            const T y1 = phase_[0][i], y2 = phase_[1][i], y3 = phase_[2][i];
            T m = std::max(std::max(std::abs(a0), std::abs(a4)),
                           std::max(std::abs(y2), std::max(std::abs(y1), std::abs(y3))));
            m = refineVertex(last, a0, y1, m);
            m = refineVertex(a0, y1, y2, m);
            m = refineVertex(y1, y2, y3, m);
            m = refineVertex(y2, y3, a4, m);
            last = y3;
            if (m > result && m <= kFinite) result = m;
        }
        channel.last = last;
        channel.fill += n;
        return result;
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
        for (int p = 0; p < kPhases; ++p)
            simd::firCorrelate(base, coeffs_[p], kTaps, phase_[p], n, T(1));
        return base;
    }

    /** Raises m to the vertex of the parabola through (a, b, c) when b is a
     *  local maximum of the magnitude there. No multiply feeds an add, so the
     *  result does not depend on floating-point contraction. */
    [[nodiscard]] static inline T refineVertex(T a, T b, T c, T m) noexcept
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

    T coeffs_[kPhases][kTaps] = {};
    T phase_[kPhases + 1][kChunk] = {}; ///< Phase outputs (+ one Annex 2 phase).
};

} // namespace dspark
