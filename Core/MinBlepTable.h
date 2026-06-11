// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include "FFT.h"
#include "WindowFunctions.h"

#include <array>
#include <vector>
#include <cmath>
#include <cstddef>

namespace dspark {

/**
 * @class MinBlepTable
 * @brief Shared minimum-phase band-limited step (minBLEP) residual table.
 *
 * A discontinuity rendered naively into a sample stream aliases. The classic
 * fix is to replace the ideal step with a band-limited one; the *minimum-phase*
 * variant (Brandt, ICMC 2001) concentrates all of its energy at and after the
 * discontinuity, which has two practical consequences:
 *
 * - **Causal correction** — the residual is zero before the event, so no
 *   look-ahead, no pre-ringing, and no kernel halves that must be predicted
 *   or un-queued when events collide (the failure mode that makes 2-point
 *   PolyBLEP hard-sync bookkeeping delicate).
 * - **Long kernel for free** — since nothing precedes the event, the kernel
 *   can be dozens of samples long without adding latency, pushing alias
 *   rejection from the ~-40 dB envelope of a 2-point polynomial kernel to
 *   the stopband of a properly windowed sinc (~-90 dB).
 *
 * The table is built once per process from first principles (no baked-in
 * magic data): a Blackman-Harris windowed sinc cutting at the base-rate
 * Nyquist is converted to minimum phase via the real cepstrum, integrated
 * into a step, and stored as the residual `minBlepStep(t) - unitStep(t)`
 * at @ref kOversample sub-sample positions per output sample.
 *
 * **Usage** — for a discontinuity of amplitude `jump` occurring `frac`
 * samples before output sample `n` (`frac` in [0, 1)), add
 * `jump * residual(j + frac)` to output sample `n + j` for
 * `j = 0 .. kTaps - 1`. The naive signal must already contain the raw step.
 *
 * @note instance() builds the table on first call (FFT work and temporary
 * heap allocations). Call it once from a control thread — e.g. inside
 * `prepare()` — never from the audio thread. After construction the table
 * is immutable and lock-free to read from any thread.
 *
 * @tparam T Sample type (float or double). Table generation always runs in
 *           double precision internally.
 */
template <typename T>
class MinBlepTable
{
    static_assert(std::is_floating_point_v<T>, "MinBlepTable requires float or double");

public:
    /// Correction span in base-rate samples (power of two, ring-buffer friendly).
    static constexpr int kTaps = 64;
    /// Sub-sample table resolution. Linear interpolation between entries keeps
    /// the sampling error near -80 dB, matching the window's stopband class.
    static constexpr int kOversample = 64;
    /// Total table entries (one guard point at the end for interpolation).
    static constexpr int kTableSize = kTaps * kOversample + 1;

    /**
     * @brief Returns the process-wide shared table, building it on first call.
     *
     * Thread-safe (C++11 magic static). The one-time build performs FFTs and
     * temporary allocations — touch it from `prepare()`, not the audio thread.
     */
    static const MinBlepTable& instance() noexcept
    {
        static const MinBlepTable table;
        return table;
    }

    /**
     * @brief Residual of the minimum-phase band-limited step at position @p t.
     *
     * @param t Time since the discontinuity, in samples (fractional). Must be
     *          >= 0; values beyond the table return 0 (the step has settled).
     * @return `minBlepStep(t) - 1` — starts near -1 right at the event (the
     *         band-limited transition has barely begun) and decays to exactly
     *         0 at the end of the table.
     */
    [[nodiscard]] T residual(T t) const noexcept
    {
        const T pos = t * static_cast<T>(kOversample);
        const auto idx = static_cast<int>(pos);
        if (idx < 0 || idx >= kTableSize - 1)
            return T(0);
        const T frac = pos - static_cast<T>(idx);
        const T a = residual_[static_cast<size_t>(idx)];
        const T b = residual_[static_cast<size_t>(idx) + 1];
        return a + frac * (b - a);
    }

private:
    MinBlepTable() noexcept { build(); }

    void build() noexcept
    {
        constexpr int    n       = kTableSize;     // 4097 oversampled points
        constexpr size_t fftSize = 65536;          // ~16x n keeps cepstral aliasing low
        constexpr double pi      = 3.14159265358979323846;

        // -- 1. Linear-phase BLIT: sinc cutting at the base-rate Nyquist
        //       (zeros every kOversample points), Blackman-Harris windowed
        //       (-92 dB stopband), normalised to unit DC gain so the
        //       integrated step settles at 1.
        std::vector<double> x(fftSize, 0.0);
        std::vector<double> win(n);
        WindowFunctions<double>::blackmanHarris(win.data(), n, false);
        const double center = 0.5 * (n - 1);
        double dc = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double t = (static_cast<double>(i) - center) / kOversample;
            const double s = (std::abs(t) < 1e-9) ? 1.0 : std::sin(pi * t) / (pi * t);
            x[static_cast<size_t>(i)] = s * win[static_cast<size_t>(i)];
            dc += s * win[static_cast<size_t>(i)];
        }
        for (int i = 0; i < n; ++i)
            x[static_cast<size_t>(i)] /= dc;

        // -- 2. Real cepstrum. log|X| is a real, even function of frequency,
        //       so running it through the real inverse FFT (imaginary parts
        //       zero) yields the even cepstrum directly.
        FFTReal<double> fft(fftSize);
        std::vector<double> spec(fftSize + 2);
        std::vector<double> cep(fftSize);
        fft.forward(x.data(), spec.data());

        const size_t bins = fftSize / 2 + 1;
        for (size_t k = 0; k < bins; ++k)
        {
            const double re  = spec[2 * k];
            const double im  = spec[2 * k + 1];
            // Soft floor at -100 dB, blended in power. The floor must sit just
            // below the window's -92 dB stopband, not far below it: hard, deep
            // corners in log|X| make the cepstrum decay like 1/q^2 and alias
            // through the causal fold, which comes back as percent-level error
            // in the reconstructed step (measured, not hypothetical).
            constexpr double floorMag = 1e-5;
            const double mag = std::sqrt(re * re + im * im + floorMag * floorMag);
            spec[2 * k]     = std::log(mag);
            spec[2 * k + 1] = 0.0;
        }
        fft.inverse(spec.data(), cep.data());

        // -- 3. Fold onto the causal part (Hilbert relation between log
        //       magnitude and minimum phase): keep q=0 and q=N/2, double
        //       1..N/2-1, zero the anticausal half.
        for (size_t q = 1; q < fftSize / 2; ++q)
        {
            cep[q]           *= 2.0;
            cep[fftSize - q]  = 0.0;
        }

        // -- 4. Back to the spectrum, exponentiate, back to time: the
        //       minimum-phase BLIT, energy packed at the front.
        fft.forward(cep.data(), spec.data());
        for (size_t k = 0; k < bins; ++k)
        {
            const double m  = std::exp(spec[2 * k]);
            const double ph = spec[2 * k + 1];
            spec[2 * k]     = m * std::cos(ph);
            spec[2 * k + 1] = m * std::sin(ph);
        }
        fft.inverse(spec.data(), x.data());

        // -- 5. Integrate into a step and store the residual. Pinning the
        //       table end exactly at step==1 closes the truncated residual
        //       at zero (no leftover micro-step when the kernel expires).
        double acc = 0.0;
        std::vector<double> step(n);
        for (int i = 0; i < n; ++i)
        {
            acc += x[static_cast<size_t>(i)];
            step[static_cast<size_t>(i)] = acc;
        }
        const double settle = step[static_cast<size_t>(n - 1)];
        for (int i = 0; i < n; ++i)
            residual_[static_cast<size_t>(i)] =
                static_cast<T>(step[static_cast<size_t>(i)] / settle - 1.0);
    }

    std::array<T, kTableSize> residual_{};
};

} // namespace dspark
