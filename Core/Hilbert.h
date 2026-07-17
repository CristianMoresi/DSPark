// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Hilbert.h
 * @brief FIR Hilbert transformer producing sample-aligned analytic signals.
 *
 * Used by FrequencyShifter (single-sideband shift) and Compressor (analytic
 * envelope detector). The kernel is sample-rate independent and shared per
 * process (lazily built, thread-safe); each instance owns only its delay
 * line.
 *
 * Threading: owner-managed. prepare/reset are setup-time; process and
 * processBlock belong to the owning (audio) thread. Not internally
 * thread-safe. processBlock installs a DenormalGuard; per-sample callers are
 * expected to guard their own callback (framework convention) - the FIR
 * itself cannot generate denormals, but denormal INPUTS would make all
 * kTaps multiplies slow without one.
 *
 * Dependencies: DspMath.h, DenormalGuard.h, SimdOps.h.
 */

#include "DspMath.h"
#include "DenormalGuard.h"
#include "SimdOps.h" // SIMD dot product for the FIR convolution

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <span>

namespace dspark {

/**
 * @class Hilbert
 * @brief 90-degree phase-differencing network (analytic-signal generator).
 *
 * Implemented as a windowed-sinc **FIR Hilbert transformer**: the quadrature
 * (imaginary) output is the input convolved with the ideal Hilbert kernel,
 * windowed with a Blackman window; the in-phase (real) output is the input
 * delayed by the FIR group delay (kCenter samples) so the two outputs stay
 * sample-aligned and form a true analytic pair.
 *
 * This FIR design is correct by construction. With the default 191-tap kernel
 * at 48 kHz the measured analytic-signal magnitude ripple is < 0.1% above
 * ~1 kHz and < 2.5% down to ~500 Hz (the kernel is rate-independent, so these
 * corner frequencies scale with the sample rate). Like every Hilbert
 * transformer it cannot produce exact quadrature at DC, and the odd-length
 * antisymmetric (Type III) design also rolls off approaching Nyquist - both
 * inherent, not defects (lengthen the kernel for more sub-500 Hz accuracy).
 *
 * @note Introduces a latency of getLatencySamples() = kCenter samples on BOTH
 *       outputs. The real branch is the delayed input, so callers that mix the
 *       "dry" path from `real` stay phase-aligned with the shifted/wet path.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Hilbert
{
public:
    struct Result
    {
        T real; ///< In-phase component (input delayed by the FIR group delay).
        T imag; ///< Quadrature component (90-deg shifted).
    };

    /**
     * @brief Prepares the transformer. The FIR kernel is sample-rate independent;
     *        sampleRate is accepted for API symmetry.
     * @param sampleRate The system sample rate. Must be > 0 (invalid values,
     *        including NaN, are ignored).
     */
    void prepare(double sampleRate) noexcept
    {
        assert(sampleRate > 0.0);
        if (!(sampleRate > 0.0)) return;

        sampleRate_ = sampleRate;
        reset();
        isPrepared_ = true;
    }

    /**
     * @brief Processes one sample, returning the analytic signal {real, imag}.
     *
     * Uses a mirrored (double-write) delay line so the convolution window is
     * one contiguous span, dispatched to the SIMD dot product - an order of
     * magnitude faster than a wrapped per-tap loop for 191 taps. (Half of the
     * kernel taps are exact zeros - the ideal Hilbert kernel vanishes on even
     * indices - but a contiguous dot product still beats a strided read that
     * would skip them; same trade-off as the framework's half-band FIRs.)
     *
     * @param input Real-valued input sample.
     */
    [[nodiscard]] inline Result process(T input) noexcept
    {
        delay_[static_cast<size_t>(writePos_)]         = input;
        delay_[static_cast<size_t>(writePos_ + kTaps)] = input;

        // Contiguous window of the last kTaps samples, oldest first:
        // window[j] = x[n - (kTaps - 1 - j)].
        const T* window = delay_.data() + writePos_ + 1;

        // imag = sum_k h[k] * x[n-k]: with an oldest-first window this is the
        // dot product against the REVERSED kernel. The pointer is cached in a
        // member so the hot path never touches the magic-static init guard.
        const T imag = simd::dotProduct(kernelData_, window, kTaps);

        // real = x[n - kCenter]: index kTaps-1-kCenter == kCenter (odd length).
        const T re = window[kCenter];

        writePos_ = (writePos_ + 1 == kTaps) ? 0 : (writePos_ + 1);
        return { re, imag };
    }

    /**
     * @brief Processes a block of samples. Optimized for CPU cache.
     *
     * @pre Both output spans must be at least input.size() long. Violations
     * assert in debug builds; release builds clamp to the shortest span.
     */
    void processBlock(std::span<const T> input,
                      std::span<T> outReal,
                      std::span<T> outImag) noexcept
    {
        assert(isPrepared_);
        assert(outReal.size() >= input.size() && outImag.size() >= input.size());
        DenormalGuard dg;

        const size_t numSamples =
            std::min(input.size(), std::min(outReal.size(), outImag.size()));
        for (size_t i = 0; i < numSamples; ++i)
        {
            const auto res = process(input[i]);
            outReal[i] = res.real;
            outImag[i] = res.imag;
        }
    }

    /** @brief Resets the delay line. Mandatory when seeking or starting playback. */
    void reset() noexcept
    {
        delay_.fill(T(0));
        writePos_ = 0;
    }

    /** @brief FIR group-delay latency applied to both outputs, in samples. */
    [[nodiscard]] static constexpr int getLatencySamples() noexcept { return kCenter; }

private:
    // 191-tap FIR: < 0.1% analytic ripple above ~1 kHz at 48 kHz (< 2.5% at
    // 500 Hz), 95-sample group delay. Odd length keeps a true integer center tap.
    static constexpr int kTaps   = 191;
    static constexpr int kCenter = kTaps / 2;

    /** @brief Builds the REVERSED windowed Hilbert kernel once per process
     *  (thread-safe C++11 static init). Reversed so the oldest-first mirrored
     *  window can be convolved with a straight SIMD dot product. */
    [[nodiscard]] static const std::array<T, kTaps>& reversedKernel() noexcept
    {
        static const std::array<T, kTaps> kernel = []
        {
            std::array<T, kTaps> k {};
            constexpr double kPi = 3.14159265358979323846;
            for (int i = 0; i < kTaps; ++i)
            {
                // Ideal Hilbert kernel h[n] = 2/(pi*n) for odd n, 0 for even n,
                // windowed with Blackman to control ripple / sideband leakage.
                const int n = i - kCenter;
                double ideal = ((n == 0) || (n % 2 == 0)) ? 0.0 : (2.0 / (kPi * static_cast<double>(n)));
                double w = 0.42
                         - 0.5  * std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(kTaps - 1))
                         + 0.08 * std::cos(4.0 * kPi * static_cast<double>(i) / static_cast<double>(kTaps - 1));
                k[static_cast<size_t>(kTaps - 1 - i)] = static_cast<T>(ideal * w);
            }
            return k;
        }();
        return kernel;
    }

    double sampleRate_ = 0.0;
    bool isPrepared_ = false;
    int writePos_ = 0;

    // Cached at construction so process() skips the magic-static guard check
    // (the static outlives every instance; copies stay valid).
    const T* kernelData_ = reversedKernel().data();

    // Mirrored delay line (double-write) for contiguous SIMD reads. No
    // over-alignment: the window starts at a variable offset every sample, so
    // the SIMD dot product uses unaligned loads regardless.
    std::array<T, kTaps * 2> delay_{};
};

} // namespace dspark
