// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Hilbert.h
 * @brief Hilbert transformers: a sample-aligned FIR and a zero-latency IIR pair.
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

/**
 * @class HilbertIIR
 * @brief Zero-latency analytic pair from two allpass chains (a 90-degree
 *        phase-difference network).
 *
 * Two cascades of seven allpass sections in z^-2, H(z) = (c - z^-2) /
 * (1 - c z^-2), one of them followed by a one-sample delay: the classic
 * polyphase-allpass Hilbert network (cf. O. Niemitalo's 90-degree phase
 * difference IIR). The coefficients were designed for this header by minimax
 * optimisation of the phase difference over 20 Hz at 192 kHz up to 20 kHz at
 * 44.1 kHz (normalised 1.04e-4 .. 0.4535): the outputs stay within 0.065
 * degrees of quadrature over that whole band, so at any rate from 44.1 to
 * 192 kHz the analytic magnitude of a tone ripples by +-0.06% (measured
 * 0.11% peak to peak).
 *
 * Unlike the FIR Hilbert above the pair has no latency and keeps its accuracy
 * down to 20 Hz, but neither output is the undelayed input: both carry the
 * same frequency-dependent allpass phase. Use it for envelopes (magnitude)
 * and quadrature pairs, not for sample-aligned processing against a dry path.
 * The recursion runs in double regardless of T (poles up to |z| = 0.9998).
 *
 * @tparam T Sample type of the interface (float or double).
 */
template <FloatType T>
class HilbertIIR
{
public:
    struct Result
    {
        T real; ///< In-phase component (allpass-filtered input).
        T imag; ///< Quadrature component, 90 degrees behind real (like the Hilbert transform of a cosine).
    };

    /** @brief Clears the allpass states. RT-safe. */
    void reset() noexcept
    {
        for (auto& s : sections_) s = {};
        delayedImag_ = 0.0;
    }

    /** @brief Processes one sample into the analytic pair. RT-safe. */
    [[nodiscard]] inline Result process(T input) noexcept
    {
        double re = 0.0, im = 0.0;
        step(static_cast<double>(input), re, im);
        return { static_cast<T>(re), static_cast<T>(im) };
    }

    /** @brief Processes one sample and returns the analytic magnitude
     *  sqrt(real^2 + imag^2): a ripple-free envelope for tones. RT-safe. */
    [[nodiscard]] inline T magnitude(T input) noexcept
    {
        double re = 0.0, im = 0.0;
        step(static_cast<double>(input), re, im);
        return static_cast<T>(std::sqrt(re * re + im * im));
    }

    /**
     * @brief Analytic magnitude of a block: out[i] = |analytic(in[i])|.
     *
     * Same result as calling magnitude() per sample, computed section by
     * section over short chunks so each allpass recursion keeps its state in
     * registers (the cascade is latency bound when run sample by sample).
     * in and out may alias. RT-safe.
     */
    void magnitudeBlock(const T* in, T* out, int numSamples) noexcept
    {
        constexpr int kChunk = 64;
        double re[kChunk];
        double im[kChunk];
        for (int start = 0; start < numSamples; start += kChunk)
        {
            const int n = std::min(kChunk, numSamples - start);
            for (int i = 0; i < n; ++i)
                re[i] = im[i] = static_cast<double>(in[start + i]);
            for (int k = 0; k < kSections; ++k)
                Section::runPair(sections_[static_cast<size_t>(k)], re, kReal[k],
                                 sections_[static_cast<size_t>(kSections + k)], im, kImag[k], n);
            for (int i = 0; i < n; ++i)
            {
                const double imDelayed = delayedImag_;
                delayedImag_ = im[i];
                out[start + i] = static_cast<T>(std::sqrt(re[i] * re[i] + imDelayed * imDelayed));
            }
        }
    }

private:
    static constexpr int kSections = 7;

    inline void step(double input, double& re, double& im) noexcept
    {
        re = input;
        double quad = input;
        for (int k = 0; k < kSections; ++k)
        {
            re = sections_[static_cast<size_t>(k)].process(re, kReal[k]);
            quad = sections_[static_cast<size_t>(kSections + k)].process(quad, kImag[k]);
        }
        im = delayedImag_;
        delayedImag_ = quad;
    }

    /** One allpass section in z^-2: y[n] = c (x[n] + y[n-2]) - x[n-2]. */
    struct Section
    {
        double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;

        // y = (c x[n] - x[n-2]) + c y[n-2]: the recursive term enters last,
        // so the loop-carried path is one multiply and one add.
        inline double process(double x, double c) noexcept
        {
            const double y = (c * x - x2) + c * y2;
            x2 = x1; x1 = x;
            y2 = y1; y1 = y;
            return y;
        }

        /** Two independent sections over a chunk each, in place and in one
         *  loop: the two recursions overlap in the pipeline (each alone is
         *  latency bound). Bit-identical to process(). */
        static inline void runPair(Section& p, double* dp, double cp,
                                   Section& q, double* dq, double cq, int n) noexcept
        {
            double px1 = p.x1, px2 = p.x2, py1 = p.y1, py2 = p.y2;
            double qx1 = q.x1, qx2 = q.x2, qy1 = q.y1, qy2 = q.y2;
            for (int i = 0; i < n; ++i)
            {
                const double xp = dp[i];
                const double xq = dq[i];
                const double yp = (cp * xp - px2) + cp * py2;
                const double yq = (cq * xq - qx2) + cq * qy2;
                px2 = px1; px1 = xp; py2 = py1; py1 = yp;
                qx2 = qx1; qx1 = xq; qy2 = qy1; qy1 = yq;
                dp[i] = yp;
                dq[i] = yq;
            }
            p.x1 = px1; p.x2 = px2; p.y1 = py1; p.y2 = py2;
            q.x1 = qx1; q.x2 = qx2; q.y1 = qy1; q.y2 = qy2;
        }
    };

    static constexpr double kReal[kSections] = {
        0.0849750635881296, 0.5135957829646070, 0.8197816632773615, 0.9420612863757530,
        0.9822486171863649, 0.9947076541867014, 0.9986500580725346 };
    static constexpr double kImag[kSections] = {  // followed by the one-sample delay
        0.2887645620083060, 0.6955314021190576, 0.8968265992273647, 0.9678214299445055,
        0.9902622348692460, 0.9971984927047717, 0.9995996402566010 };

    std::array<Section, 2 * kSections> sections_ {};
    double delayedImag_ = 0.0;
};

} // namespace dspark
