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

namespace dspark {

/**
 * @class TruePeakDetector
 * @brief Per-channel 4x-oversampled inter-sample peak estimator.
 *
 * Real-time safe: the polyphase coefficients are compile-time constants and
 * processing is a fixed 48-multiply routine per sample. The reading is the
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
        auto& tp = states_[static_cast<size_t>(channel)];

        // Mirrored write: every sample lands at writePos and writePos + 16,
        // so the latest 12-tap window is always contiguous in memory and each
        // phase collapses to one linear dot product (SIMD) instead of a
        // masked ring walk per tap.
        tp.history[static_cast<size_t>(tp.writePos)] = sample;
        tp.history[static_cast<size_t>(tp.writePos + kHistSize)] = sample;
        const int newest = tp.writePos;
        tp.writePos = (tp.writePos + 1) & kHistMask;

        // Window holding x[n-11] .. x[n] in forward (oldest-first) order.
        const T* window = &tp.history[static_cast<size_t>(newest + kHistSize - (kTaps - 1))];

        T peak = std::abs(sample);
        for (int phase = 0; phase < kPhases; ++phase)
        {
            const T interp = simd::dotProduct(
                kReversedCoeffs[static_cast<size_t>(phase)].data(), window, kTaps);
            const T a = std::abs(interp);
            if (a > peak) peak = a;
        }
        return peak;
    }

    /** @brief Group delay of the interpolation FIR in samples (per phase). */
    [[nodiscard]] static constexpr int getLatency() noexcept { return kTaps / 2; }

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

    std::array<State, MaxChannels> states_ {};
};

} // namespace dspark
