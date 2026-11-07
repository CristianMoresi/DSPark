// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file TruePeakDetector.h
 * @brief Shared ITU-R BS.1770 true-peak (inter-sample peak) detector.
 *
 * Estimates the analog reconstruction peak of a digital signal with the
 * official ITU-R BS.1770-5 Annex 2 4x over-sampling interpolator: the
 * reference 48-tap, 4-phase polyphase FIR (12 taps per phase). The EBU Tech
 * 3341 true-peak tolerances (+0.2/-0.4 dB, cases 15-23) are defined around
 * exactly this filter, and DSPark passes them with it.
 *
 * One shared implementation backs every true-peak consumer in the framework
 * (Compressor, Limiter, LoudnessMeter), so the filter design exists in exactly
 * one place.
 *
 * Dependencies: DspMath.h, SimdOps.h.
 */

#include "DspMath.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
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

} // namespace dspark
