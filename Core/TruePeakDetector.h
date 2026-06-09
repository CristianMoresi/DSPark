// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file TruePeakDetector.h
 * @brief Shared ITU-R BS.1770-4 true-peak (inter-sample peak) detector.
 *
 * Estimates the analog reconstruction peak of a digital signal by evaluating
 * three additional 4x-oversampled phases between consecutive samples with a
 * 48-tap polyphase FIR (12 taps per phase, Kaiser beta = 8).
 *
 * One shared implementation backs every true-peak consumer in the framework
 * (Compressor, Limiter, LoudnessMeter), so the filter design exists in exactly
 * one place.
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <array>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class TruePeakDetector
 * @brief Per-channel 4x-oversampled inter-sample peak estimator.
 *
 * Real-time safe: the polyphase coefficients are built once (lazily, on first
 * use — thread-safe C++11 static initialisation) and processing is a fixed
 * 36-multiply routine per sample.
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
     * three interpolated inter-sample points preceding it.
     *
     * @param sample  Input sample.
     * @param channel Channel index (0-based; clamped into range).
     * @return True-peak magnitude (>= |sample|).
     */
    [[nodiscard]] T processSample(T sample, int channel) noexcept
    {
        if (channel < 0 || channel >= MaxChannels) channel = 0;
        auto& tp = states_[static_cast<size_t>(channel)];

        tp.history[tp.writePos] = sample;
        tp.writePos = (tp.writePos + 1) & kHistMask;

        const auto& coeffs = phaseCoeffs();

        T peak = std::abs(sample);
        const int newest = (tp.writePos - 1) & kHistMask;

        for (int phase = 0; phase < kPhases; ++phase)
        {
            T interp = T(0);
            int idx = newest;
            for (int k = 0; k < kTaps; ++k)
            {
                interp += tp.history[idx] * coeffs[static_cast<size_t>(phase)][static_cast<size_t>(k)];
                idx = (idx - 1) & kHistMask;
            }
            const T a = std::abs(interp);
            if (a > peak) peak = a;
        }
        return peak;
    }

    /** @brief Group delay of the interpolation FIR in samples (per phase). */
    [[nodiscard]] static constexpr int getLatency() noexcept { return kTaps / 2; }

private:
    static constexpr int kTaps     = 12;  ///< Taps per polyphase branch.
    static constexpr int kPhases   = 3;   ///< Inter-sample phases (4x oversampling).
    static constexpr int kHistSize = 16;  ///< Power-of-two history per channel.
    static constexpr int kHistMask = kHistSize - 1;

    struct State
    {
        T history[kHistSize] = {};
        int writePos = 0;
    };

    /** @brief Builds the 4x polyphase interpolation coefficients once. */
    [[nodiscard]] static const std::array<std::array<T, kTaps>, kPhases>& phaseCoeffs() noexcept
    {
        static const std::array<std::array<T, kTaps>, kPhases> table = []
        {
            constexpr int N = kTaps * 4;
            constexpr double M = (N - 1) / 2.0;
            constexpr double fc = 0.25;
            constexpr double beta = 8.0;
            constexpr double pi = std::numbers::pi;

            auto besselI0 = [](double x) -> double {
                double sum = 1.0, term = 1.0;
                for (int k = 1; k <= 25; ++k)
                {
                    const double half = x / (2.0 * k);
                    term *= half * half;
                    sum += term;
                    if (term < 1e-15 * sum) break;
                }
                return sum;
            };

            const double i0Beta = besselI0(beta);
            double h[N];
            for (int n = 0; n < N; ++n)
            {
                const double x = static_cast<double>(n) - M;
                const double sincArg = 2.0 * fc * x;
                const double sincVal = (std::abs(sincArg) < 1e-10)
                    ? 1.0
                    : std::sin(pi * sincArg) / (pi * sincArg);
                const double t = x / M;
                const double kaiserVal = (std::abs(t) > 1.0)
                    ? 0.0
                    : besselI0(beta * std::sqrt(1.0 - t * t)) / i0Beta;
                h[n] = sincVal * kaiserVal;
            }

            std::array<std::array<T, kTaps>, kPhases> result {};
            for (int phase = 0; phase < kPhases; ++phase)
            {
                const int p = phase + 1;
                for (int k = 0; k < kTaps; ++k)
                    result[static_cast<size_t>(phase)][static_cast<size_t>(k)]
                        = static_cast<T>(h[4 * k + p]);
            }
            return result;
        }();
        return table;
    }

    std::array<State, MaxChannels> states_ {};
};

} // namespace dspark
