// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LoudnessNormalizer.h
 * @brief OFFLINE loudness normalization to a target LUFS under a true-peak
 *        ceiling (ITU-R BS.1770-5 / EBU R 128).
 *
 * OFFLINE ONLY. normalize() walks the whole buffer several times, allocates a
 * gain envelope and takes time proportional to the programme; it must never be
 * called from an audio callback. There is no streaming entry point here on
 * purpose: gated integrated loudness is defined over a finished measurement
 * interval, so a normalizer that had not yet seen the end of the programme
 * would be normalizing to a number that is still moving.
 *
 * What the standards fix, and where:
 *
 * - The measurement is ITU-R BS.1770-5, Annex 1: K-weighting (two-stage
 *   pre-filter, Tables 1 and 2), mean square per channel, channel-weighted
 *   sum (Table 3), and the two-stage gate of equations (6) and (7) -- an
 *   absolute threshold at -70 LKFS and a relative threshold 10 LU below the
 *   absolute-gated result. The gating block is 400 ms long and overlaps its
 *   neighbour by 75%; the Annex states that as normative prose, not as a
 *   numbered equation, and equation (3) is the block mean square z_ij written
 *   in terms of that duration and that overlap.
 *   `LoudnessMeter` implements all of it and this class calls it; the
 *   measurement is not re-derived here.
 * - The true-peak estimate is ITU-R BS.1770-5, Annex 2: 4x over-sampling with
 *   the published order-48, 4-phase FIR interpolator. `TruePeakDetector`
 *   carries that coefficient table and this class uses it both to drive the
 *   ceiling and to report the result.
 * - The default target of -23.0 LUFS is EBU R 128 recommendation (h), and the
 *   default ceiling of -1.0 dBTP is EBU R 128 recommendation (m). Both are
 *   defaults, not limits: streaming platforms publish their own targets (-16
 *   to -13 LUFS are common) and those are house rules, not standardised
 *   values, so setTargetLUFS() takes any finite number.
 *
 * Two passes, and a third only when it is earned:
 *
 * 1. Measure the integrated loudness of the input.
 * 2. Apply the constant gain `target - measured` to every channel, then
 *    true-peak limit down to the ceiling.
 * 3. Re-measure. Limiting can only remove loudness, so if it removed more than
 *    kIterationThresholdLU the gain is corrected once by the shortfall and the
 *    limiter is run again on the result. It is done once, not to convergence:
 *    the two constraints can be genuinely incompatible (see below) and a loop
 *    chasing an unreachable target would only add gain reduction.
 *
 * The ceiling wins. Where the material cannot reach the target without
 * exceeding the ceiling, the ceiling is honoured and the target is missed;
 * Result::outLUFS reports what was actually achieved, and a caller that needs
 * the target met exactly must check it. This is not a defect to be tuned away,
 * it is what the two constraints mean together.
 *
 * The ceiling is met against the 4x estimator, which is what the standard
 * specifies and what every compliance meter reads -- not against the analog
 * reconstruction. Annex 2 prefers an over-sampled rate of at least 192 kHz;
 * 4x reaches that at 48 kHz and above but only 176.4 kHz at 44.1 kHz, so at
 * 44.1 kHz the estimate (and therefore the guarantee built on it) is the
 * slightly optimistic one the standard's own tolerance allows for.
 *
 * Dependencies: LoudnessMeter.h, AudioBuffer.h, DspMath.h, DenormalGuard.h,
 * TruePeakDetector.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/TruePeakDetector.h"
#include "LoudnessMeter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dspark {

/**
 * @class LoudnessNormalizer
 * @brief Offline two-pass LUFS normalizer with a true-peak ceiling.
 *
 * Threading: offline and single-threaded. normalize() owns the buffer it is
 * given for the duration of the call, allocates, and is NOT audio-thread
 * safe -- nothing published here is read by an audio thread, so no cross-thread
 * hand-off exists in this class. The two setters are lock-free atomic
 * publications so a control thread may configure the object between runs;
 * calling them while normalize() is running is not defined, because normalize()
 * reads each of them once and the run would silently mix two configurations.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LoudnessNormalizer final
{
public:
    /** @brief What one normalize() run did. */
    struct Result
    {
        T measuredLUFS   = T(-100);  ///< Integrated loudness of the input (LUFS).
        T appliedGainDb  = T(0);     ///< Total constant gain applied before limiting (dB).
        T outLUFS        = T(-100);  ///< Integrated loudness of the output (LUFS).
        T outTruePeakDb  = T(-100);  ///< True peak of the output (dBTP, BS.1770-5 Annex 2).
        int limiterPasses = 0;       ///< Refinement passes the ceiling needed (0 = never engaged).
        bool targetReached = false;  ///< outLUFS within kTargetToleranceLU of the target.
    };

    // -- Parameters (control thread) -------------------------------------------

    /**
     * @brief Target integrated loudness in LUFS (default -23.0).
     *
     * -23.0 LUFS is the EBU R 128 Programme Loudness Level. Non-finite values
     * are ignored, as everywhere else in the framework: a NaN target would
     * turn the gain into a NaN and destroy the programme.
     */
    void setTargetLUFS(T target) noexcept
    {
        if (!std::isfinite(target)) return;
        targetLUFS_.store(target, std::memory_order_relaxed);
    }

    /** @return The configured target in LUFS. */
    [[nodiscard]] T getTargetLUFS() const noexcept
    {
        return targetLUFS_.load(std::memory_order_relaxed);
    }

    /**
     * @brief True-peak ceiling in dBTP (default -1.0).
     *
     * -1.0 dBTP is the EBU R 128 maximum permitted true-peak level for
     * production. Non-finite values are ignored; the value is clamped to
     * [-40, 0] dBTP, since a ceiling above 0 dBTP would ask the limiter to
     * permit clipping in every downstream converter and a ceiling below -40
     * would be a fader, not a ceiling.
     */
    void setTruePeakCeilingDb(T ceilingDb) noexcept
    {
        if (!std::isfinite(ceilingDb)) return;
        ceilingDb_.store(std::clamp(ceilingDb, T(-40), T(0)), std::memory_order_relaxed);
    }

    /** @return The configured true-peak ceiling in dBTP. */
    [[nodiscard]] T getTruePeakCeilingDb() const noexcept
    {
        return ceilingDb_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Backward reach of the gain envelope, in milliseconds (default 1.0).
     *
     * The gain reduction a peak demands is spread BACKWARDS from it, so the
     * level is already down by the time the peak arrives. Exactly: this is the
     * time the envelope takes to traverse a factor of ten in amplitude (20 dB),
     * which is the natural unit of the log-amplitude domain the envelope is
     * built in and not a tuning constant of its own; a peak needing 6 dB of
     * reduction therefore starts coming down 0.3 * lookahead before it.
     *
     * Two lower bounds are structural rather than a matter of taste: the
     * reduction must cover the whole 12-sample support of the Annex 2
     * interpolator that reported the peak (0.25 ms at 48 kHz), and the gain
     * must be near-constant across that support for the peak to scale with it.
     * The value is clamped to [0.05, 50] ms and, independently, never falls
     * below that support in samples. Larger values trade loudness for
     * smoothness. The ceiling does NOT depend on this: the envelope is a
     * maximum over the requirement, so it dominates it at every sample whatever
     * the slope, and the verification pass below catches whatever is left.
     */
    void setLookaheadMs(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        lookaheadMs_.store(std::clamp(ms, T(0.05), T(50)), std::memory_order_relaxed);
    }

    /** @return The configured backward reach of the gain envelope, in ms. */
    [[nodiscard]] T getLookaheadMs() const noexcept
    {
        return lookaheadMs_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Forward decay of the gain envelope, in milliseconds (default 1).
     *
     * How long the gain takes to come back after a peak has passed, in the same
     * unit as the lookahead: the time to traverse a factor of ten in amplitude
     * (20 dB). Clamped to [1, 2000] ms.
     *
     * The default was selected on a steady-state 1 kHz carrier with periodic
     * transient trains at 4, 6, 10, 20, 40 and 80 Hz. Nine candidates from 1
     * to 2000 ms were ranked independently at every rate by output sideband
     * energy more than 20 Hz from the carrier and by absolute loudness miss;
     * the predeclared equal-rank sum selected 1 ms (24, against 39 for 5 ms and
     * 47 for 20 ms). Its worst far-sideband result was -29.49 dBc and its worst
     * absolute miss was 0.2397 LU. All 54 points met a -1 dBTP ceiling at
     * -1.00087 dBTP, so release controls transparency and loudness rather than
     * the ceiling guarantee. The bed is synthetic and intentionally separating,
     * not a claim that one release is perceptually optimal for every programme;
     * callers may choose any value in the documented range.
     */
    void setReleaseMs(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::clamp(ms, T(1), T(2000)), std::memory_order_relaxed);
    }

    /** @return The configured forward decay of the gain envelope, in ms. */
    [[nodiscard]] T getReleaseMs() const noexcept
    {
        return releaseMs_.load(std::memory_order_relaxed);
    }

    // -- Offline processing -----------------------------------------------------

    /**
     * @brief Normalizes `audio` in place and reports what it did.
     *
     * @param audio      Buffer to normalize in place. Untouched if it is empty
     *                   or if `sampleRate` is not a usable rate.
     * @param sampleRate Sample rate of `audio` in Hz.
     * @return The measurements and the gain applied.
     *
     * Loudness is measured over the first two channels, which is the extent of
     * `LoudnessMeter`'s channel-weighted sum; the gain and the ceiling are
     * applied to EVERY channel. On a programme with more than two channels the
     * reading is therefore a stereo reading of a multichannel programme, not
     * the BS.1770 Table 3 surround-weighted one, and the class does not pretend
     * otherwise.
     *
     * Silence, or anything whose gated loudness sits at the meter's floor, is
     * left alone: there is no gain that makes silence measure -23 LUFS, and
     * applying an enormous one would only normalize the noise floor.
     *
     * WHAT IT COSTS, so a caller can size the job before starting it. The
     * ceiling stage keeps two `double` buffers as long as the programme --
     * the gain envelope and the double buffer its support dilation writes
     * through -- which is **16 bytes per sample, independent of the channel
     * count**: about 5.5 GB for a 60-minute programme at 96 kHz, mono or 5.1
     * alike. Time is proportional to the programme too, and the constant is
     * not small: each measurement walks every sample through a 48-tap
     * interpolator per channel, and a run performs two or three loudness
     * measurements plus one true-peak pass per ceiling refinement. A caller
     * whose material will not fit should normalize in sections.
     */
    template <int MaxChannels>
    Result normalize(AudioBuffer<T, MaxChannels>& audio, double sampleRate)
    {
        DenormalGuard guard;

        Result out;
        const int nCh = audio.getNumChannels();
        const int nS  = audio.getNumSamples();
        if (nCh <= 0 || nS <= 0 || !std::isfinite(sampleRate) || sampleRate <= 0.0)
            return out;

        const T target  = getTargetLUFS();
        const T ceiling = decibelsToGain(getTruePeakCeilingDb(), T(-200));

        out.measuredLUFS = measureIntegrated(audio, sampleRate);
        // The meter reports its floor for silence and for anything with no
        // gating block above -70 LKFS. There is no meaningful gain to derive
        // from that, so the programme is returned untouched and reported as it
        // was measured.
        if (out.measuredLUFS <= kMeterFloorLUFS)
        {
            out.outLUFS       = out.measuredLUFS;
            out.outTruePeakDb = measureTruePeak(audio);
            return out;
        }

        T gainDb = target - out.measuredLUFS;
        applyConstantGain(audio, decibelsToGain(gainDb, T(-200)));
        out.limiterPasses = limitToTruePeakCeiling(audio, sampleRate, ceiling);
        out.outLUFS = measureIntegrated(audio, sampleRate);

        // Limiting can only take loudness away. If it took more than the
        // policy allows, give the shortfall back once and limit again: the
        // second run removes less than the first because most of the peaks are
        // already down. Once, not to convergence -- where the ceiling is the
        // binding constraint the shortfall never closes.
        if (std::isfinite(out.outLUFS) && (target - out.outLUFS) > kIterationThresholdLU)
        {
            const T extraDb = target - out.outLUFS;
            applyConstantGain(audio, decibelsToGain(extraDb, T(-200)));
            out.limiterPasses += limitToTruePeakCeiling(audio, sampleRate, ceiling);
            gainDb += extraDb;
            out.outLUFS = measureIntegrated(audio, sampleRate);
        }

        out.appliedGainDb  = gainDb;
        out.outTruePeakDb  = measureTruePeak(audio);
        out.targetReached  = std::isfinite(out.outLUFS)
                          && std::abs(out.outLUFS - target) <= kTargetToleranceLU;
        return out;
    }

    /** @brief Integrated loudness of `audio` in LUFS, without modifying it. */
    template <int MaxChannels>
    [[nodiscard]] T measureLUFS(const AudioBuffer<T, MaxChannels>& audio, double sampleRate)
    {
        if (audio.getNumChannels() <= 0 || audio.getNumSamples() <= 0
            || !std::isfinite(sampleRate) || sampleRate <= 0.0)
            return T(-100);
        return measureIntegrated(audio, sampleRate);
    }

    /** @brief True peak of `audio` in dBTP (BS.1770-5 Annex 2 estimator). */
    template <int MaxChannels>
    [[nodiscard]] T measureTruePeakDb(const AudioBuffer<T, MaxChannels>& audio)
    {
        return measureTruePeak(audio);
    }

    /// Loudness shortfall (LU) beyond which the gain is corrected once and the
    /// limiter re-run. Policy, not a standard: it is the point at which a
    /// programme that limited hard is worth one more pass rather than being
    /// delivered quiet.
    static constexpr T kIterationThresholdLU = T(0.3);

    /// How close outLUFS must land for Result::targetReached. EBU R 128
    /// recommendation (i) allows +/-0.2 LU for measurement error in a loudness
    /// workflow; this is tighter, because both measurements here come from the
    /// same meter and the meter's own error cancels.
    static constexpr T kTargetToleranceLU = T(0.1);

private:
    /// Meter reading at or below this is the meter's silence floor, not a level.
    static constexpr T kMeterFloorLUFS = T(-99);

    /// Ceiling refinement passes. The gain envelope is built so that every
    /// sample of a violating peak's support is turned down by at least the
    /// amount that peak demands, but the interpolator has negative taps and the
    /// envelope is not constant across that support, so the first pass is a
    /// very good estimate and not a proof. Each further pass measures what is
    /// actually left and removes it. One local pass preserves nearly all the
    /// loudness this stage can recover on the declared programme beds; the
    /// mandatory constant trim below removes the small residual exactly. The
    /// bound therefore keeps the offline cost predictable and makes the
    /// guarantee visibly independent of iterative convergence.
    static constexpr int kMaxCeilingPasses = 1;

    /// The passes aim this far BELOW the ceiling, as a fraction of it
    /// (1e-4 = 0.00087 dB). Aiming exactly at the boundary makes each pass land
    /// on it to within floating-point rounding and the next pass find a
    /// last-bit excess to remove, so the loop crawls; aiming a thousandth of a
    /// decibel under ends it. It is a rounding allowance, not a tuning: it is
    /// three orders of magnitude below the +0.2/-0.4 dB tolerance the true-peak
    /// measurement itself is specified to.
    static constexpr double kCeilingAimMargin = 1.0e-4;

    /// Channels the true-peak stage tracks independently.
    static constexpr int kMaxTpChannels = 16;

    // -- Measurement ------------------------------------------------------------

    template <int MaxChannels>
    T measureIntegrated(const AudioBuffer<T, MaxChannels>& audio, double sampleRate)
    {
        const int nCh = audio.getNumChannels();
        const int nS  = audio.getNumSamples();
        meter_.prepare(sampleRate, std::min(nCh, 2));
        meter_.reset();
        if (nCh >= 2)
            meter_.process(audio.getChannel(0), audio.getChannel(1), nS);
        else
            meter_.process(audio.getChannel(0), nS);
        return meter_.getIntegratedLUFS();
    }

    template <int MaxChannels>
    T measureTruePeak(const AudioBuffer<T, MaxChannels>& audio)
    {
        return gainToDecibels(truePeakLinear(audio), T(-100));
    }

    /// Largest Annex 2 true-peak estimate over every channel, linear.
    template <int MaxChannels>
    T truePeakLinear(const AudioBuffer<T, MaxChannels>& audio)
    {
        const int nCh = std::min(audio.getNumChannels(), kMaxTpChannels);
        const int nS  = audio.getNumSamples();
        if (nCh <= 0 || nS <= 0) return T(0);

        tp_.reset();
        T peak = T(0);
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* d = audio.getChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                const T v = tp_.processSample(d[i], ch);
                if (std::isfinite(v) && v > peak) peak = v;
            }
        }
        return peak;
    }

    // -- Gain -------------------------------------------------------------------

    template <int MaxChannels>
    static void applyConstantGain(AudioBuffer<T, MaxChannels>& audio, T gain) noexcept
    {
        if (!(gain > T(0)) || gain == T(1)) return;
        const int nCh = audio.getNumChannels();
        const int nS  = audio.getNumSamples();
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* d = audio.getChannel(ch);
            for (int i = 0; i < nS; ++i) d[i] *= gain;
        }
    }

    /**
     * @brief Brings the true peak of `audio` down to `ceiling` (linear).
     * @return Number of passes actually run (0 if the ceiling was already met).
     *
     * One pass is: measure the per-sample true-peak envelope over all channels;
     * turn the requirement into a reduction in dB; widen it over the
     * interpolator's own support so every sample that CREATES a reported peak
     * is covered, not only the sample the estimate was reported at; spread it
     * backwards over the lookahead and forwards over the release; apply. The
     * spreading is a maximum, so the reduction applied at any sample is never
     * less than the reduction that sample demanded.
     */
    template <int MaxChannels>
    int limitToTruePeakCeiling(AudioBuffer<T, MaxChannels>& audio, double sampleRate, T ceiling)
    {
        const int nCh = std::min(audio.getNumChannels(), kMaxTpChannels);
        const int nS  = audio.getNumSamples();
        if (nCh <= 0 || nS <= 0 || !(ceiling > T(0))) return 0;

        const T aim = static_cast<T>(static_cast<double>(ceiling) * (1.0 - kCeilingAimMargin));
        const int support = TruePeakDetector<T, kMaxTpChannels>::getTaps();
        const auto reachSamples = std::max<int>(
            support,
            static_cast<int>(static_cast<double>(getLookaheadMs()) * sampleRate / 1000.0));
        const auto relSamples = std::max<int>(
            1, static_cast<int>(static_cast<double>(getReleaseMs()) * sampleRate / 1000.0));

        // Reduction in dB per sample of distance: a straight line in dB, which
        // is what a limiter's attack and release curves are.
        const double perSampleUp   = 1.0 / static_cast<double>(reachSamples);
        const double perSampleDown = 1.0 / static_cast<double>(relSamples);

        reduction_.assign(static_cast<std::size_t>(nS), 0.0);

        int passes = 0;
        for (int pass = 0; pass < kMaxCeilingPasses; ++pass)
        {
            // 1. Per-sample true-peak envelope, maximum over channels.
            tp_.reset();
            for (auto& v : reduction_) v = 0.0;
            bool over = false;
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* d = audio.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                {
                    const T v = tp_.processSample(d[i], ch);
                    if (!std::isfinite(v) || v <= aim) continue;
                    // Excess as a NORMALIZED reduction in [0, 1]: 1 means the
                    // full dynamic range of the scale below, which never
                    // happens; the linear ramps below live in this unit so a
                    // dB slope is one subtraction per sample.
                    const double need = std::log10(static_cast<double>(v)
                                                   / static_cast<double>(aim));
                    if (need > reduction_[static_cast<std::size_t>(i)])
                    {
                        reduction_[static_cast<std::size_t>(i)] = need;
                        over = true;
                    }
                }
            }
            if (!over) break;
            ++passes;

            // 2. Widen backwards over the interpolator's support: the estimate
            //    reported at i was computed from samples i-support+1 .. i, and
            //    all of them have to come down for it to come down.
            dilateSupport(support, nS);

            // 3. Spread: backwards at the lookahead slope, forwards at the
            //    release slope. Both are running maxima of a decayed value, so
            //    the envelope dominates the requirement everywhere.
            for (int i = nS - 2; i >= 0; --i)
            {
                const double cand = reduction_[static_cast<std::size_t>(i + 1)] - perSampleUp;
                if (cand > reduction_[static_cast<std::size_t>(i)])
                    reduction_[static_cast<std::size_t>(i)] = cand;
            }
            for (int i = 1; i < nS; ++i)
            {
                const double cand = reduction_[static_cast<std::size_t>(i - 1)] - perSampleDown;
                if (cand > reduction_[static_cast<std::size_t>(i)])
                    reduction_[static_cast<std::size_t>(i)] = cand;
            }

            // 4. Apply.
            for (int ch = 0; ch < audio.getNumChannels(); ++ch)
            {
                T* d = audio.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                {
                    const double r = reduction_[static_cast<std::size_t>(i)];
                    if (r > 0.0)
                        d[i] = static_cast<T>(static_cast<double>(d[i]) * std::pow(10.0, -r));
                }
            }
        }

        // THIS is where the ceiling comes from, and it is not optional.
        //
        // The loop above is bounded, and a run that ends at the bound can still
        // leave the estimate above the ceiling: the envelope is not constant
        // across a peak's support and the interpolator has negative taps.
        // Measured on a pink programme targeted to -5 LUFS, deleting this step
        // leaves the output at -0.95050 dBTP for a -1 dBTP ceiling; the two
        // outputs differ at every sample, so the trim is not a no-op on that
        // operating point. A constant gain, by contrast, scales the true-peak
        // estimate EXACTLY -- the estimator is linear and its input is scaled
        // uniformly -- so one multiplication puts the peak at the aim by
        // construction. That makes the ceiling a property of the function
        // rather than of the loop's convergence, which is the only form in
        // which a bounded loop can carry a guarantee at all.
        //
        // It costs loudness across the whole programme. That cost is the
        // reason the loop above exists -- to leave this step as little as
        // possible to take -- and not a reason to treat this step as a
        // fallback that a later optimisation may remove.
        const T left = truePeakLinear(audio);
        if (std::isfinite(left) && left > ceiling)
        {
            applyConstantGain(audio, aim / left);
            ++passes;
        }
        return passes;
    }

    /**
     * @brief Replaces each entry with the maximum of the `support` entries
     *        starting at it, so the demand of a peak reaches every sample the
     *        estimator formed it from.
     *
     * Twelve entries wide, so the direct scan is shorter and faster than the
     * monotone deque an arbitrary window would need.
     */
    void dilateSupport(int support, int nS)
    {
        if (support <= 1) return;
        scratch_.assign(static_cast<std::size_t>(nS), 0.0);
        for (int i = 0; i < nS; ++i)
        {
            double running = 0.0;
            const int last = std::min(nS - 1, i + support - 1);
            for (int j = i; j <= last; ++j)
                running = std::max(running, reduction_[static_cast<std::size_t>(j)]);
            scratch_[static_cast<std::size_t>(i)] = running;
        }
        reduction_.swap(scratch_);
    }

    std::atomic<T> targetLUFS_  { T(-23) };
    std::atomic<T> ceilingDb_   { T(-1) };
    std::atomic<T> lookaheadMs_ { T(1) };
    std::atomic<T> releaseMs_   { T(1) };

    LoudnessMeter<T> meter_;
    TruePeakDetector<T, kMaxTpChannels> tp_;

    /// Per-sample gain reduction under construction, in units of log10 of the
    /// linear excess (one unit = 20 dB), and its double buffer.
    std::vector<double> reduction_, scratch_;
};

} // namespace dspark
