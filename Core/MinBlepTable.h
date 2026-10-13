// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file MinBlepTable.h
 * @brief Shared minimum-phase band-limited step (minBLEP) residual table.
 *
 * Built once per process from first principles (no baked-in magic data): a
 * Kaiser-windowed sinc cutting at the base-rate Nyquist is converted to
 * minimum phase via the real cepstrum, integrated into a step, refined to a
 * fine sub-sample grid and stored as polyphase rows. Oscillator uses it to
 * correct every discontinuity causally (no pre-ring, no look-ahead).
 *
 * Dependencies: FFT.h.
 */

#include "FFT.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <vector>

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
 * - **Causal correction** -- the residual is zero before the event, so no
 *   look-ahead, no pre-ringing, and no kernel halves that must be predicted
 *   or un-queued when events collide (the failure mode that makes 2-point
 *   PolyBLEP hard-sync bookkeeping delicate).
 * - **Long kernel for free** -- since nothing precedes the event, the kernel
 *   can be long without adding latency. Its cost is per discontinuity, not
 *   per sample.
 *
 * Design, each choice measured on free-running saws and pulses at 44.1 and
 * 48 kHz (worst inharmonic component below 20 kHz, 110 Hz to 14 kHz):
 *
 * - **128 taps, Kaiser window (beta 13, about -127 dB stopband), cut at
 *   Nyquist.** The transition band then ends near 23.4 kHz at 44.1 kHz,
 *   below fs - 20 kHz, so nothing a sweep leaves partially attenuated
 *   folds below 20 kHz. The previous 64-tap Blackman-Harris kernel let the
 *   harmonics between 24.1 and 24.8 kHz through partially: -95 dB at
 *   440 Hz / 44.1 kHz, landing at 19.5 kHz.
 * - **2048 sub-sample phases.** Linear interpolation between adjacent rows
 *   leaves an error that falls 12 dB per doubling; the previous 64 positions
 *   capped the floor at -94 dB (9 kHz / 48 kHz), and 2048 reach -145 dB.
 *   The minimum-phase step is built on a 64-position grid (a 2^17 FFT) and
 *   refined to the fine grid with Niemitalo's optimal 6-point, 5th-order
 *   interpolator for 32x oversampled data: the step is band-limited to the
 *   base Nyquist, 64 times below the coarse grid's, so the refinement adds
 *   no measurable error and costs milliseconds instead of a 2^22 FFT.
 * - **Polyphase rows in float.** Row q holds the residual at j + q / kPhases
 *   for every tap j, so a discontinuity reads two contiguous rows: cache
 *   friendly and vectorisable. Float storage keeps the table at 1 MB; its
 *   rounding (-150 dB) is below the interpolation floor.
 *
 * Measured floors: -145 dB or better from 110 Hz to 14 kHz at 48 kHz and
 * 44.1 kHz, against -89 to -121 dB for the previous table.
 *
 * **Usage** -- for a discontinuity of amplitude `jump` occurring `frac`
 * samples before output sample `n` (`frac` >= 0), add
 * `jump * residual(j + frac)` to output sample `n + j` for
 * `j = 0 .. kTaps - 1`; addStep() does exactly that into a ring buffer. The
 * naive signal must already contain the raw step.
 *
 * @note instance() builds the table on first call (FFT work, about 1 MB of
 * table and transient buffers). Call it once from a control thread -- e.g.
 * inside `prepare()` -- never from the audio thread. After construction the
 * table is immutable and lock-free to read from any thread. The build is
 * noexcept by design: an allocation failure during it terminates (fail-fast;
 * the only caller path is a noexcept prepare() anyway).
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
    static constexpr int kTaps = 128;
    /// Sub-sample resolution: rows per sample, linearly interpolated.
    static constexpr int kPhases = 2048;
    /// Sub-sample positions per output sample (the same resolution).
    static constexpr int kOversample = kPhases;
    /// Points of the residual's domain [0, kTaps] at that resolution.
    static constexpr int kTableSize = kTaps * kOversample + 1;

    /**
     * @brief Returns the process-wide shared table, building it on first call.
     *
     * Thread-safe (C++11 magic static). The one-time build performs FFTs and
     * temporary allocations -- touch it from `prepare()`, not the audio thread.
     */
    static const MinBlepTable& instance() noexcept
    {
        static const MinBlepTable table;
        return table;
    }

    /**
     * @brief Residual of the minimum-phase band-limited step at position @p t.
     *
     * @param t Time since the discontinuity, in samples (fractional). Values
     *          below 0 or at and beyond kTaps return 0 (the step has settled).
     * @return `minBlepStep(t) - 1` -- starts near -1 right at the event (the
     *         band-limited transition has barely begun) and decays to exactly
     *         0 at the end of the table.
     */
    [[nodiscard]] T residual(T t) const noexcept
    {
        if (!(t >= T(0)) || !(t < static_cast<T>(kTaps)))
            return T(0);
        const auto whole = static_cast<int>(t);
        const T pos = (t - static_cast<T>(whole)) * static_cast<T>(kPhases);
        const auto q = std::min(static_cast<int>(pos), kPhases - 1);
        const T w = pos - static_cast<T>(q);
        const float* a = rows_.data() + static_cast<size_t>(q) * kTaps;
        const T ra = static_cast<T>(a[whole]);
        const T rb = static_cast<T>(a[kTaps + whole]);
        return ra + w * (rb - ra);
    }

    /**
     * @brief Adds `jump * residual(j + frac)` to `ring[(head + j) & (kTaps - 1)]`
     *        for every tap j.
     *
     * @param ring A kTaps-long ring of pending corrections.
     * @param head Ring slot of the output sample the correction starts at.
     * @param jump Signed discontinuity amplitude (new value minus old).
     * @param frac Time from the event to that output sample, in samples
     *             (>= 0; ages of kTaps or more add nothing).
     */
    void addStep(T* ring, int head, T jump, T frac) const noexcept
    {
        addRows(rows_, ring, head, jump, frac);
    }

    /**
     * @brief Step correction for a waveform rendered dcDelay() samples late.
     *
     * A generator that evaluates its whole naive waveform dcDelay() samples
     * in the past (so its smooth segments line up with the band-limited
     * steps and ramps) corrects a jump of the CURRENT waveform with the
     * band-limited step minus the step delayed by dcDelay(): the residual
     * plus 1 before dcDelay(). The jump is still timed at the event.
     */
    void addDelayedStep(T* ring, int head, T jump, T frac) const noexcept
    {
        addRows(rows_, ring, head, jump, frac);
        addEarly(ring, head, frac, [jump](T) noexcept { return jump; });
    }

    /**
     * @brief Slope-change (kink) correction for a waveform rendered
     *        dcDelay() samples late: the minimum-phase band-limited ramp
     *        (minBLAMP) minus the ideal ramp delayed by dcDelay().
     *
     * @param ring      A kTaps-long ring of pending corrections.
     * @param head      Ring slot of the output sample the correction starts at.
     * @param slopeJump Change of the derivative at the event, per sample
     *                  (new slope minus old slope).
     * @param frac      Time from the event to the ring's head sample (>= 0).
     *
     * Unlike a ramp correction against the undelayed ideal ramp, which tends
     * to -dcDelay() * slopeJump and cannot be truncated, this one settles at
     * exactly 0 at the kernel's end.
     */
    void addDelayedRamp(T* ring, int head, T slopeJump, T frac) const noexcept
    {
        addRows(rampRows_, ring, head, slopeJump, frac);
        const T delay = dcDelay_;
        addEarly(ring, head, frac, [slopeJump, delay](T t) noexcept { return slopeJump * (t - delay); });
    }

    /**
     * @brief Low-frequency delay of the minimum-phase step, in samples.
     *
     * Equals minus the integral of the residual: the band-limited step
     * completes its rise about this much later than the ideal step it
     * replaces. A generator that adds the residuals to a naive waveform must
     * delay the waveform's continuous (sloped) segments by the same amount,
     * i.e. subtract slope * dcDelay(): otherwise every jump contributes
     * jump * -dcDelay() of area, a DC offset of 2 * dcDelay() * f0 / fs on a
     * sawtooth.
     */
    [[nodiscard]] T dcDelay() const noexcept { return dcDelay_; }

private:
    /// Positions per sample of the grid the minimum-phase step is built on.
    static constexpr int kCoarse = 64;
    /// Kaiser window shape: about -127 dB of stopband.
    static constexpr double kBeta = 13.0;

    MinBlepTable() noexcept { build(); }

    /** Niemitalo's optimal 32x 6-point, 5th-order interpolator (z-form);
     *  y holds the samples at offsets -2..3, x the fraction after y[2]. */
    [[nodiscard]] static double refine(const double* y, double x) noexcept
    {
        const double z = x - 0.5;
        const double even1 = y[3] + y[2], odd1 = y[3] - y[2];
        const double even2 = y[4] + y[1], odd2 = y[4] - y[1];
        const double even3 = y[5] + y[0], odd3 = y[5] - y[0];
        const double c0 = even1 * 0.42685983409379380 + even2 * 0.07238123511170030
                        + even3 * 0.00075893079450573;
        const double c1 = odd1 * 0.35831772348893259 + odd2 * 0.20451644554758297
                        + odd3 * 0.00562658797241955;
        const double c2 = even1 * -0.217009177221292431 + even2 * 0.20051376594086157
                        + even3 * 0.01649541128040211;
        const double c3 = odd1 * -0.25112715343740988 + odd2 * 0.04223025992200458
                        + odd3 * 0.02488727472995134;
        const double c4 = even1 * 0.04166946673533273 + even2 * -0.06250420114356986
                        + even3 * 0.02083473440841799;
        const double c5 = odd1 * 0.08349799235675044 + odd2 * -0.04174912841630993
                        + odd3 * 0.00834987866042734;
        return ((((c5 * z + c4) * z + c3) * z + c2) * z + c1) * z + c0;
    }

    void build() noexcept
    {
        constexpr int    n       = kTaps * kCoarse + 1;   // 8193 coarse points
        constexpr size_t fftSize = 131072;                // 16x n keeps cepstral aliasing low
        constexpr double pi      = std::numbers::pi_v<double>;

        // -- 1. Linear-phase BLIT: sinc cutting at the base-rate Nyquist
        //       (zeros every kCoarse points), Kaiser windowed, normalised to
        //       unit DC gain so the integrated step settles at 1.
        auto besselI0 = [](double x) noexcept {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k < 200; ++k)
            {
                term *= (x / (2.0 * k)) * (x / (2.0 * k));
                sum += term;
                if (term < sum * 1e-18) break;
            }
            return sum;
        };
        std::vector<double> x(fftSize, 0.0);
        const double center = 0.5 * (n - 1);
        const double norm = besselI0(kBeta);
        double dc = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double t = (static_cast<double>(i) - center) / kCoarse;
            const double s = (std::abs(t) < 1e-9) ? 1.0 : std::sin(pi * t) / (pi * t);
            const double r = (static_cast<double>(i) - center) / center;
            const double w = besselI0(kBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) / norm;
            x[static_cast<size_t>(i)] = s * w;
            dc += s * w;
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
            // Soft floor just below the window's stopband, blended in power:
            // hard, deep corners in log|X| make the cepstrum decay like 1/q^2
            // and alias through the causal fold, which comes back as
            // percent-level error in the reconstructed step.
            constexpr double floorMag = 1e-7;
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

        // -- 5. Integrate into a step on the coarse grid, settling exactly
        //       at 1 at the end of the kernel.
        std::vector<double> step(static_cast<size_t>(n));
        double acc = 0.0;
        for (int i = 0; i < n; ++i)
        {
            acc += x[static_cast<size_t>(i)];
            step[static_cast<size_t>(i)] = acc;
        }
        const double settle = step[static_cast<size_t>(n - 1)];
        for (double& s : step) s /= settle;

        // -- 6. Refine the step to the fine grid: the residual at every fine
        //       point of [0, kTaps], pinned to exactly 0 at the kernel's end.
        //       The step is 0 before the event and 1 after the kernel.
        auto stepAt = [&](long i) noexcept {
            if (i < 0) return 0.0;
            if (i >= n) return 1.0;
            return step[static_cast<size_t>(i)];
        };
        constexpr long fineEnd = static_cast<long>(kTaps) * kPhases;
        constexpr int ratio = kPhases / kCoarse;
        std::vector<double> r1(static_cast<size_t>(fineEnd + 1), 0.0);
        for (long fine = 0; fine < fineEnd; ++fine)
        {
            const long i = fine / ratio;
            const double frac = static_cast<double>(fine % ratio) / ratio;
            double y[6];
            for (int k = 0; k < 6; ++k) y[k] = stepAt(i - 2 + k);
            r1[static_cast<size_t>(fine)] = refine(y, frac) - 1.0;
        }

        // -- 7. The ramp residual: the running trapezoidal integral of the
        //       step residual. It tends to -dcDelay (the band-limited ramp
        //       lags the ideal one), so it is stored plus dcDelay, which
        //       settles exactly at 0 at the kernel's end.
        std::vector<double> r2(static_cast<size_t>(fineEnd + 1), 0.0);
        for (long fine = 1; fine <= fineEnd; ++fine)
            r2[static_cast<size_t>(fine)] = r2[static_cast<size_t>(fine - 1)]
                + 0.5 * (r1[static_cast<size_t>(fine - 1)] + r1[static_cast<size_t>(fine)]) / kPhases;
        const double delay = -r2[static_cast<size_t>(fineEnd)];
        dcDelay_ = static_cast<T>(delay);

        // -- 8. Polyphase rows: row q, tap j holds the value at j + q / kPhases.
        rows_.assign(static_cast<size_t>(kPhases + 1) * kTaps, 0.0f);
        rampRows_.assign(static_cast<size_t>(kPhases + 1) * kTaps, 0.0f);
        for (int q = 0; q <= kPhases; ++q)
            for (int j = 0; j < kTaps; ++j)
            {
                const long fine = static_cast<long>(j) * kPhases + q;
                const auto at = static_cast<size_t>(q) * kTaps + static_cast<size_t>(j);
                if (fine >= fineEnd) continue;   // both settle at exactly 0
                rows_[at] = static_cast<float>(r1[static_cast<size_t>(fine)]);
                rampRows_[at] = static_cast<float>(r2[static_cast<size_t>(fine)] + delay);
            }
    }

    /** Adds `term(j + frac)` to every tap j whose time j + frac precedes
     *  dcDelay(): the part of a delayed-reference correction before the
     *  delayed naive waveform reaches the event. Exact, not tabulated. */
    template <typename Term>
    void addEarly(T* ring, int head, T frac, Term term) const noexcept
    {
        if (!(frac >= T(0))) return;
        for (int j = 0; j < kTaps; ++j)
        {
            const T t = static_cast<T>(j) + frac;
            if (!(t < dcDelay_)) break;
            ring[(head + j) & (kTaps - 1)] += term(t);
        }
    }

    /** Adds `scale * rows(j + frac)` into the ring for every tap j. */
    static void addRows(const std::vector<float>& rows, T* ring, int head, T scale, T frac) noexcept
    {
        if (!(frac >= T(0)) || !(frac < static_cast<T>(kTaps)))
            return;
        const auto whole = static_cast<int>(frac);
        const T pos = (frac - static_cast<T>(whole)) * static_cast<T>(kPhases);
        const auto q = std::min(static_cast<int>(pos), kPhases - 1);
        const T w = pos - static_cast<T>(q);
        const float* a = rows.data() + static_cast<size_t>(q) * kTaps + whole;
        const float* b = a + kTaps;
        const int n = kTaps - whole;
        // Two contiguous runs around the ring's wrap.
        const int first = std::min(n, kTaps - (head & (kTaps - 1)));
        T* r = ring + (head & (kTaps - 1));
        for (int j = 0; j < first; ++j)
        {
            const T ra = static_cast<T>(a[j]);
            r[j] += scale * (ra + w * (static_cast<T>(b[j]) - ra));
        }
        for (int j = first; j < n; ++j)
        {
            const T ra = static_cast<T>(a[j]);
            ring[j - first] += scale * (ra + w * (static_cast<T>(b[j]) - ra));
        }
    }

    std::vector<float> rows_;       ///< (kPhases + 1) rows of kTaps step residuals.
    std::vector<float> rampRows_;   ///< The same for the ramp residual plus dcDelay.
    T dcDelay_ = T(0);
};

} // namespace dspark
