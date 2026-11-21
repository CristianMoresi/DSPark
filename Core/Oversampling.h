// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Oversampling.h
 * @brief Power-of-two oversampling with polyphase half-band filters.
 *
 * Cascaded 2x stages in one of two filter modes:
 *
 * - **Linear-phase FIR (default).** Kaiser-windowed symmetric half-band
 *   filters, both directions as polyphase pairs. Upsampling: the even outputs
 *   are the FIR phase over the non-zero half-band taps, the odd outputs a pure
 *   delay through the centre tap, so zero-stuffed buffers are never
 *   materialised. Decimation: the high-rate stream is split into its even and
 *   odd samples; the even ones meet the non-zero taps and the odd ones the
 *   centre tap, at exactly the positions of the full high-rate convolution.
 *   The FIR phases run through simd::firCorrelate(), register-blocked over
 *   outputs. Each stage after the first is shorter: its transition band only
 *   has to protect what the first stage keeps, so it is designed to be no
 *   worse than the first stage over that band, never to a fixed tap count.
 *   The round trip is an exact integer delay at every frequency.
 *   Depth::Deep (opt-in) keeps a preset's band and makes every stage at
 *   least 100 dB deep, for nonlinear processing whose harmonics fill the
 *   oversampled band. Band::Audible (opt-in) fits the first stage to the
 *   host rate: it keeps 20 kHz whole and stops the image of 20 kHz, so the
 *   higher the rate the shorter the stage.
 * - **Polyphase IIR (opt-in, Mode::PolyphaseIir).** Elliptic half-band
 *   filters realised as two first-order allpass cascades in z^-2
 *   (Valenzuela and Constantinides, IEE Proc. 1983), the coefficients
 *   designed here from the elliptic filter's theta series. Same or deeper
 *   stopband than the FIR preset for 2-6 samples of round-trip delay instead
 *   of 16-141. CPU is close to the FIR at 2x (High: 3.9 vs 4.7 ns per base
 *   sample, mono, SSE2) but higher from 4x up, where every recursive stage
 *   runs once per high-rate sample and cannot be register-blocked like the
 *   FIR (High 4x: 10.6 vs 7.2 ns). The phase is NOT linear: the round-trip delay is a
 *   fractional, frequency-dependent number of samples (getRoundTripDelay()),
 *   so a parallel dry path cannot be aligned by an integer delay above the
 *   low audio range. getLatency() reports it rounded at low frequencies.
 *
 * Features:
 * - Factors 1/2/4/8/16 with four quality presets, two stopband depths and
 *   two bands.
 * - Block-based streaming; variable block sizes are supported and give
 *   bit-identical output for any partition of the stream.
 * - Zero allocations after prepare(); upsample()/downsample() are noexcept.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DenormalGuard.h, FIRFilter.h,
 * SimdOps.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "DenormalGuard.h"
#include "FIRFilter.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class Oversampling
 * @brief Power-of-two oversampling processor with polyphase anti-aliasing.
 *
 * Usage is a paired streaming operation per audio callback:
 * @code
 * auto up = os.upsample(block);   // 1. base rate -> high rate
 * process(up);                    // 2. nonlinear processing at the high rate
 * os.downsample(block);           // 3. high rate -> base rate, in place
 * @endcode
 *
 * @note upsample() and downsample() form a pair: downsample() consumes the
 *       high-rate samples the most recent upsample() left in the internal
 *       buffer, so call them once each per block, with the same block length.
 *
 * A source that generates at the high rate (a synthesizer voice bank) skips
 * the upsampling: it fills getHighRateBlock() and calls downsample(), and its
 * delay is getDecimationDelay() high-rate samples.
 *
 * Threading: prepare() and reset() belong to the setup thread (prepare
 * allocates). upsample() and downsample() belong to the audio thread and are
 * noexcept and allocation-free. There are no runtime setters -- factor,
 * quality, mode, depth and band are fixed at construction and all filter
 * state is only touched by the audio thread -- so no cross-thread
 * synchronisation is needed.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels supported.
 */
template <typename T, int MaxChannels = 16>
class Oversampling
{
public:
    /**
     * @brief Quality presets defining the steepness and rejection of the anti-aliasing filter.
     *
     * The figures are the worst image/alias level of the whole cascade, which
     * the first stage sets (its stop edge); later stages are designed to be
     * no worse over the band that folds back into the first stage's passband
     * and transition. Linear-phase FIR taps of the first stage are 31/63/127/255.
     * In Mode::PolyphaseIir the same transition band gets at least the same
     * depth (worst image or alias over every factor, measured: Low -55,
     * Medium -83, High -91, Maximum -107 dB).
     */
    enum class Quality
    {
        Low,      ///< ~-40 dB stopband. For fast previewing.
        Medium,   ///< ~-60 dB stopband. General purpose.
        High,     ///< ~-80 dB stopband. Professional standard.
        Maximum   ///< ~-100 dB stopband. Mastering grade.
    };

    /** @brief Half-band filter family (see the file notes). */
    enum class Mode
    {
        LinearPhaseFir, ///< Exact integer delay at every frequency (default).
        PolyphaseIir    ///< Elliptic allpass pairs: a few samples of nonlinear-phase delay.
    };

    /**
     * @brief Stopband depth of the cascade (see the constructor).
     *
     * - Preset (default): the depth of the quality preset, as above.
     * - Deep: every stage at least 100 dB deep over the band it protects,
     *   with the preset's transition band (the same passband edge and stop
     *   edge). A nonlinear process at the high rate leaves harmonics across
     *   the whole band above the base Nyquist at close to their full level,
     *   and the preset's depth is then the floor of what folds back.
     *   Linear-phase FIR taps per stage (first stage first): Low 91/35/23/31,
     *   Medium 115/31/23/31, High 163/31/23/31 (with Preset 31/11/7/15,
     *   63/19/15/15 and 127/23/23/15). Round-trip latency at High:
     *   82/90/93/95 base samples at 2x/4x/8x/16x (Preset 64/70/73/74).
     *   Maximum already has that depth and is unchanged. In
     *   Mode::PolyphaseIir every stage gets the coefficient count that
     *   reaches 100 dB over its band (104 dB or deeper for every preset).
     */
    enum class Depth
    {
        Preset, ///< The quality preset's depth (default).
        Deep    ///< At least 100 dB in every stage, with the preset's band.
    };

    /**
     * @brief The band the first stage keeps (see the constructor).
     *
     * - Preset (default): the quality preset's own transition band, a fixed
     *   share of the host rate around its Nyquist frequency (High: from 0.460
     *   to 0.540 of the rate), whatever the rate.
     * - Audible: the first stage keeps everything up to 20 kHz and stops
     *   everything from the image of 20 kHz (the host rate less 20 kHz) up,
     *   to the depth of the preset; what lies between, which nobody hears,
     *   is neither kept whole nor stopped. The transition is then
     *   1/2 - 20 kHz / rate of the 2x rate: 0.046 at 44.1 kHz, 0.083 at
     *   48 kHz, 0.292 at 96 kHz and 0.396 at 192 kHz, against a fixed 0.040
     *   for High. prepare() takes the shortest kernel that measures the
     *   preset's depth over that stopband, so the stage gets shorter as the
     *   rate rises (and where the preset's own band is the wider one, as Low
     *   and Medium are at 44.1 kHz, it stays). Later stages keep their
     *   designs: they pass 20 kHz and stop its images with room to spare.
     *   The latency is known after prepare() and stays a whole number of
     *   samples. In Mode::PolyphaseIir the first stage gets the same band
     *   and the coefficient count that reaches the depth over it (at 96 kHz
     *   High delays 2.27 samples at 1 kHz in place of 3.07).
     *
     * First stage of High, in taps, at 44.1, 48, 88.2, 96, 176.4 and 192 kHz:
     * 111, 63, 23, 19, 15 and 15 against 127, which at 2x is a latency of 56,
     * 32, 12, 10, 8 and 8 samples against 64. At 96 kHz Low, Medium and
     * Maximum take 7, 15 and 27 taps against 31, 63 and 255, and Depth::Deep
     * 27 against 163. What folds under 20 kHz at 96 kHz measures -82.6 dB or
     * less for High (the preset's first stage: -79.2) and a 19.9 kHz tone
     * comes back within 0.0006 dB.
     *
     * What Audible gives up: content of the input between 20 kHz and its
     * Nyquist frequency is not carried whole to the high rate, and its images
     * are not removed whole; on the way down, what a process leaves between
     * the Nyquist frequency and the image of 20 kHz folds back above 20 kHz.
     * Nothing folds below 20 kHz that the preset would have stopped. At
     * 96 kHz with High a 30 kHz tone comes back at 0.948 of its level and a
     * 40 kHz one at 0.672, where the preset returns both whole.
     */
    enum class Band
    {
        Preset,  ///< The preset's transition band (default).
        Audible  ///< The first stage keeps 20 kHz and stops its image, by the host rate.
    };

    /**
     * @brief Constructs the oversampling engine.
     * @param factor  Oversampling factor. Must be a power of two (1, 2, 4, 8, 16).
     * @param quality Filter quality preset. Default is High (-80 dB).
     * @param mode    Filter family. Default is the linear-phase FIR.
     * @param depth   Stopband depth. Default is the preset's (Depth::Preset).
     * @param band    Band of the first stage. Default is the preset's
     *                (Band::Preset); Band::Audible fits it to the host rate
     *                at prepare().
     * @pre factor >= 1 and is a power of 2.
     * @note Release builds normalise an invalid factor into range and round it
     *       down to a power of two (debug builds assert).
     */
    explicit Oversampling(int factor = 2, Quality quality = Quality::High,
                          Mode mode = Mode::LinearPhaseFir, Depth depth = Depth::Preset,
                          Band band = Band::Preset)
        : quality_(quality), mode_(mode), depth_(depth), band_(band),
          firstStage_(firStageDesign(quality, depth, 0)), firstTransition_(firTransition(quality))
    {
        assert(factor >= 1 && (factor & (factor - 1)) == 0 && "Factor must be a power of 2");
        assert(factor <= (1 << kMaxStages) && "Factor must be <= 16 (kMaxStages stages)");
        // Release-safe normalisation: clamp into [1, 2^kMaxStages], then derive
        // the stage count and re-derive factor_ = 2^numStages_ so the pair stays
        // coherent even for a non-power-of-two request. An incoherent pair
        // (e.g. factor_ = 3 driving one 2x stage) would make downsample() emit
        // factor_/2 samples per base sample -- overrunning the caller's output
        // buffer -- and overflow the per-stage histories sized for 2^numStages_.
        factor = std::clamp(factor, 1, 1 << kMaxStages);
        numStages_ = 0;
        while ((1 << (numStages_ + 1)) <= factor)
            ++numStages_;
        factor_ = 1 << numStages_;

        if (mode_ == Mode::PolyphaseIir)
        {
            for (int s = 0; s < numStages_; ++s)
                iir_[static_cast<size_t>(s)].design(iirCoefficientCount(quality_, depth_, s),
                                                    iirTransition(quality_, s));
        }
        computeDelays();
    }

    /**
     * @brief Pre-allocates buffers and computes FIR filter coefficients.
     * @param spec The audio specification at the base (1x) sample rate.
     * @post The processor is ready to handle up to `spec.maxBlockSize` base samples.
     * @note With Band::Audible the first stage is fitted to spec.sampleRate
     *       here, so getLatency() and getDecimationDelay() are those of this
     *       rate from now on (before the first prepare() they are the
     *       preset's). The fit tries a few lengths and windows and measures
     *       each, which takes longer the longer the stage: for High, 0.2 to
     *       0.4 ms at 96 kHz, 1 to 1.7 ms at 48 kHz and 3 to 7 ms at
     *       44.1 kHz; for Maximum at 44.1 kHz, 5 to 8 ms (six builds).
     */
    void prepare(const AudioSpec& spec)
    {
        baseSpec_ = spec;
        upBuffer_.resize(spec.numChannels, spec.maxBlockSize * factor_);
        fitFirstStage(spec.sampleRate);

        for (int stage = 0; stage < numStages_; ++stage)
        {
            // The max block size entering a stage is baseSize * 2^stage
            const int stageMaxSamples = spec.maxBlockSize * (1 << stage);
            if (mode_ == Mode::PolyphaseIir)
            {
                iir_[static_cast<size_t>(stage)].allocate(spec.numChannels, stageMaxSamples);
            }
            else
            {
                const StageDesign d = stageDesign(stage);
                filters_[static_cast<size_t>(stage)].design(d.taps, static_cast<T>(d.beta),
                                                            spec.numChannels, stageMaxSamples);
            }
        }

        reset();
    }

    /**
     * @brief Flushes all internal delay lines and history buffers.
     * Call this when resetting the transport or to clear audio tails.
     */
    void reset() noexcept
    {
        upBuffer_.clear();
        for (int i = 0; i < numStages_; ++i)
        {
            filters_[static_cast<size_t>(i)].reset();
            iir_[static_cast<size_t>(i)].reset();
        }
    }

    /**
     * @brief The smallest factor that brings a host rate up to an internal rate.
     *
     * A nonlinear stage that always runs at about the same internal rate
     * sounds the same at every host rate and stops paying for oversampling
     * the host already provides: with the default minimum the factor is 4 at
     * 44.1 and 48 kHz, 2 at 88.2 and 96 kHz and 1 from 176.4 kHz up.
     *
     * @param sampleRate          Host rate in Hz. A rate that is not a positive
     *                            finite number gives 1.
     * @param minimumInternalRate Internal rate to reach, in Hz.
     * @param maxFactor           Largest factor returned (at most 16).
     * @return A power of two between 1 and maxFactor.
     */
    [[nodiscard]] static constexpr int factorForRate(double sampleRate,
                                                     double minimumInternalRate = 176400.0,
                                                     int maxFactor = 1 << kMaxStages) noexcept
    {
        if (!(sampleRate > 0.0) || !(sampleRate < 1e12)) return 1;
        const int limit = maxFactor < 1 ? 1 : (maxFactor > (1 << kMaxStages) ? (1 << kMaxStages) : maxFactor);
        int factor = 1;
        // A rate a rounding error short of an exact multiple still counts.
        while (factor * 2 <= limit && sampleRate * factor < minimumInternalRate * (1.0 - 1e-9))
            factor *= 2;
        return factor;
    }

    /** @brief Returns the current oversampling factor. */
    [[nodiscard]] int getFactor() const noexcept { return factor_; }

    /** @brief Returns the current quality preset. */
    [[nodiscard]] Quality getQuality() const noexcept { return quality_; }

    /** @brief Returns the half-band filter family. */
    [[nodiscard]] Mode getMode() const noexcept { return mode_; }

    /** @brief Returns the stopband depth (Depth::Preset or Depth::Deep). */
    [[nodiscard]] Depth getDepth() const noexcept { return depth_; }

    /** @brief Returns the band of the first stage (Band::Preset or Band::Audible). */
    [[nodiscard]] Band getBand() const noexcept { return band_; }

    /**
     * @brief Length of a linear-phase stage's kernel, in taps at its high rate.
     *
     * Half of them (the odd ones from the centre) are the half-band's zeros
     * and cost nothing. With Band::Audible the first stage's length is that
     * of the rate of the last prepare().
     *
     * @param stage Stage, 0 the first (the one at the lowest rate).
     * @return The length, or 0 for a stage the engine does not have or in
     *         Mode::PolyphaseIir.
     */
    [[nodiscard]] int getStageLength(int stage) const noexcept
    {
        if (mode_ != Mode::LinearPhaseFir || stage < 0 || stage >= numStages_) return 0;
        return stageDesign(stage).taps;
    }

    /**
     * @brief Transition band of the first stage, as a fraction of the 2x rate.
     *
     * The stage keeps everything under (1/4 - half of this) of the 2x rate
     * and stops everything over (1/4 + half of it): the preset's own figure,
     * or 1/2 - 20 kHz / rate with Band::Audible where that is wider.
     */
    [[nodiscard]] double getFirstStageTransition() const noexcept { return firstTransition_; }

    /**
     * @brief Round-trip (upsample + downsample) delay at the BASE rate, in samples.
     *
     * Linear-phase FIR: exact at every frequency. Each stage s contributes its
     * centre-tap delay on the way up and down, (halfOrder_s + 1) samples at
     * its high rate, i.e. (halfOrder_s + 1) / 2^s base samples; the stage
     * lengths are chosen so every term is an integer. Verified sample-exact
     * against an impulse round trip for every factor and preset.
     *
     * Polyphase IIR: the low-frequency round-trip delay rounded to the nearest
     * sample. The true delay is fractional and rises toward the band edge;
     * query it with getRoundTripDelay().
     */
    [[nodiscard]] int getLatency() const noexcept { return latency_; }

    /**
     * @brief Exact round-trip delay at a frequency, in base-rate samples.
     *
     * Constant (getLatency()) in the linear-phase mode. In Mode::PolyphaseIir
     * it is the phase delay of the elliptic allpass pairs, evaluated from the
     * designed coefficients: for example 3.07 samples at 1 kHz rising to 4.23
     * samples at 18 kHz for 2x High at 48 kHz. Not real-time (it evaluates
     * the transfer functions); call it from a setup or UI thread.
     *
     * @param frequencyHz Frequency in Hz, at the base rate of prepare()
     *        (48 kHz is assumed before prepare()).
     */
    [[nodiscard]] double getRoundTripDelay(double frequencyHz) const noexcept
    {
        if (mode_ != Mode::PolyphaseIir || numStages_ == 0)
            return static_cast<double>(latency_);
        const double fs = baseSpec_.sampleRate > 0.0 ? baseSpec_.sampleRate : 48000.0;
        const double w = 2.0 * 3.14159265358979323846 * std::clamp(frequencyHz / fs, 1e-6, 0.4999);
        return iirDelay(w, true);
    }

    /**
     * @brief Group delay of the decimation path alone, in HIGH-RATE samples.
     *
     * The delay of a source rendered directly at the high rate through
     * getHighRateBlock() and downsample(). Linear-phase FIR: every half-band
     * stage delays by its centre tap, halfOrder_s samples at its input rate,
     * which is exact and generally fractional at the base rate (31.5 samples
     * at 2x High); a source that must report an integer base-rate latency L
     * starts its content L * getFactor() minus this value high-rate samples
     * later. Polyphase IIR: the low-frequency delay, rounded (the true delay
     * is fractional and frequency dependent).
     */
    [[nodiscard]] int getDecimationDelay() const noexcept { return decimationDelay_; }

    /**
     * @brief Writable high-rate block for a source that renders directly at
     *        the oversampled rate (a synthesizer, for example).
     *
     * The decimation-only counterpart of upsample(): no upsampling filter
     * runs. Fill every sample of the returned view, then call downsample()
     * with the same base block length. The path delays by
     * getDecimationDelay() high-rate samples. Real-time safe.
     *
     * @param numSamples Base-rate block length, clamped to prepare()'s
     *        maxBlockSize.
     * @return View of the internal buffer, numSamples * getFactor() long,
     *         with the prepared channel count.
     */
    [[nodiscard]] AudioBufferView<T> getHighRateBlock(int numSamples) noexcept
    {
        const int nS = std::clamp(numSamples, 0, baseSpec_.maxBlockSize);
        return upBuffer_.toView().getSubView(0, nS * factor_);
    }

    /**
     * @brief Upsamples the input base-rate signal into the internal high-rate buffer.
     *
     * @param input View of the base-rate audio buffer. Blocks longer than
     *        prepare()'s maxBlockSize are truncated (release-safe); channels
     *        beyond the prepared channel count are ignored.
     * @return A view of the internal oversampled buffer (length = input samples * factor).
     * @note Real-time safe. Pair every call with one downsample() of the same
     *       block length before the next upsample().
     */
    [[nodiscard]] AudioBufferView<T> upsample(AudioBufferView<const T> input) noexcept
    {
        const int nCh = std::min(input.getNumChannels(), upBuffer_.getNumChannels());
        // Release-safe clamp: a block larger than prepare()'s maxBlockSize would
        // overflow the per-stage histories and the internal high-rate buffer.
        const int nS  = std::min(input.getNumSamples(), baseSpec_.maxBlockSize);

        if (numStages_ == 0)
        {
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(upBuffer_.getChannel(ch), input.getChannel(ch), static_cast<std::size_t>(nS) * sizeof(T));
            return upBuffer_.toView().getSubView(0, nS);
        }

        if (mode_ == Mode::PolyphaseIir)
        {
            DenormalGuard guard; // recursive state: keep decaying tails out of the denormal range
            int len = nS;
            for (int stage = 0; stage < numStages_; ++stage)
            {
                for (int ch = 0; ch < nCh; ++ch)
                    iir_[static_cast<size_t>(stage)].upsampleChannel(
                        stage == 0 ? input.getChannel(ch) : upBuffer_.getChannel(ch),
                        upBuffer_.getChannel(ch), ch, len);
                len *= 2;
            }
            return upBuffer_.toView().getSubView(0, nS * factor_);
        }

        // Stage 0: Read directly from input view
        filters_[0].processUpsample(input, upBuffer_.toView(), nCh, nS);

        // Subsequent stages: in-place upsampling inside upBuffer_
        int currentLen = nS * 2;
        for (int stage = 1; stage < numStages_; ++stage)
        {
            filters_[static_cast<size_t>(stage)].processUpsampleInPlace(upBuffer_.toView(), nCh, currentLen);
            currentLen *= 2;
        }

        return upBuffer_.toView().getSubView(0, nS * factor_);
    }

    /**
     * @brief Downsamples the internal high-rate buffer back to base-rate.
     *
     * Consumes the high-rate samples written by the most recent upsample()
     * (and modified in place by the caller in between).
     *
     * @param output Buffer view where the base-rate samples will be written.
     *        Use the same block length as the paired upsample() input.
     * @note Real-time safe.
     */
    void downsample(AudioBufferView<T> output) noexcept
    {
        const int nCh = std::min(output.getNumChannels(), upBuffer_.getNumChannels());
        const int nS  = std::min(output.getNumSamples(), baseSpec_.maxBlockSize);

        if (numStages_ == 0)
        {
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(output.getChannel(ch), upBuffer_.getChannel(ch), static_cast<std::size_t>(nS) * sizeof(T));
            return;
        }

        if (mode_ == Mode::PolyphaseIir)
        {
            DenormalGuard guard;
            int len = nS * factor_;
            for (int stage = numStages_ - 1; stage >= 0; --stage)
            {
                for (int ch = 0; ch < nCh; ++ch)
                    iir_[static_cast<size_t>(stage)].downsampleChannel(
                        upBuffer_.getChannel(ch),
                        stage == 0 ? output.getChannel(ch) : upBuffer_.getChannel(ch), ch, len);
                len /= 2;
            }
            return;
        }

        int currentLen = nS * factor_;

        for (int stage = numStages_ - 1; stage > 0; --stage)
        {
            filters_[static_cast<size_t>(stage)].processDownsampleInPlace(upBuffer_.toView(), nCh, currentLen);
            currentLen /= 2;
        }

        // Stage 0: Write directly to final output
        filters_[0].processDownsample(upBuffer_.toView(), output, nCh, currentLen);

        // Level transparency note: no gain compensation is applied (or needed).
        // An impulse round-trip peaks slightly below 1.0 only because the
        // impulse contains supra-Nyquist energy the anti-alias filters must
        // remove; in-band sine-wave gain is already ~1.000 (verified to better
        // than 0.01 dB across all quality presets).
    }

private:
    static constexpr int kMaxStages = 4;

    // ========================================================================
    // Polyphase Block Half-Band FIR
    // ========================================================================
    /**
     * @struct PolyphaseHalfBand
     * @brief Inner 2x oversampling engine utilizing polyphase half-band decomposition.
     */
    struct PolyphaseHalfBand
    {
        std::vector<T> evenTaps;   ///< Non-zero half-band taps (the even array indices).
        T centerTap = T(0);
        int halfOrder = 0;
        int delaySamples = 0;

        /**
         * @brief Per-channel FIR state.
         */
        struct ChannelState {
            std::vector<T> history;    ///< Upsampler: base-rate input with its FIR prefix.
            std::vector<T> evenHist;   ///< Decimator: even high-rate samples with prefix.
            std::vector<T> oddHist;    ///< Decimator: odd high-rate samples with prefix.
        };
        std::vector<ChannelState> upChannels;
        std::vector<ChannelState> downChannels;
        std::vector<T> scratch;        ///< FIR-phase outputs of one call.

        /**
         * @brief Designs the half-band filter and allocates history buffers.
         * @param taps Total filter length.
         * @param beta Kaiser window beta parameter.
         * @param numChannels Number of audio channels.
         * @param maxBlockSamples Maximum number of base samples to process in one call.
         */
        void design(int taps, T beta, int numChannels, int maxBlockSamples)
        {
            halfOrder = (taps - 1) / 2;
            // INVARIANT: the polyphase split below collects the array-even taps,
            // which coincide with the non-zero (odd-from-centre) half-band taps
            // ONLY when halfOrder is odd. Every stage design satisfies this
            // (firStageDesign); keep (taps-1)/2 odd or the filter degenerates.
            assert((halfOrder & 1) == 1 && "half-band polyphase requires odd halfOrder");
            // Pure-delay alignment for the centre-tap (odd) phase. With the full
            // even branch (halfOrder+1 taps, group delay (halfOrder)/2 = numTaps/2
            // base samples) the centre sample must sit numTaps/2 = (halfOrder+1)/2
            // samples into the window to stay phase-aligned with the even phase.
            delaySamples = (halfOrder + 1) / 2;

            auto fullCoeffs = FIRDesign<T>::lowPass(1.0, 0.25, taps, beta);
            centerTap = fullCoeffs[static_cast<std::size_t>(halfOrder)];

            // The non-zero taps: all even array indices (the centre sits at the
            // odd index halfOrder, and the other odd indices are the half-band
            // zeros).
            evenTaps.clear();
            for (int i = 0; i < taps; i += 2)
                evenTaps.push_back(fullCoeffs[static_cast<std::size_t>(i)]);

            // Up history: base-rate (numTaps + block). Down histories: halfOrder
            // samples of prefix plus up to maxBlockSamples new samples per phase
            // (a stage's decimator input is 2 * maxBlockSamples long).
            const int upHistorySize   = static_cast<int>(evenTaps.size()) + maxBlockSamples;
            const int downHistorySize = halfOrder + maxBlockSamples;

            upChannels.resize(static_cast<std::size_t>(numChannels));
            downChannels.resize(static_cast<std::size_t>(numChannels));

            for (int ch = 0; ch < numChannels; ++ch)
            {
                upChannels[ch].history.assign(static_cast<std::size_t>(upHistorySize), T(0));
                downChannels[ch].evenHist.assign(static_cast<std::size_t>(downHistorySize), T(0));
                downChannels[ch].oddHist.assign(static_cast<std::size_t>(downHistorySize), T(0));
            }
            scratch.assign(static_cast<std::size_t>(maxBlockSamples), T(0));
        }

        /**
         * @brief Clears the history buffers to prevent audio clicks on transport reset.
         */
        void reset() noexcept
        {
            for (auto& ch : upChannels)
                std::fill(ch.history.begin(), ch.history.end(), T(0));
            for (auto& ch : downChannels)
            {
                std::fill(ch.evenHist.begin(), ch.evenHist.end(), T(0));
                std::fill(ch.oddHist.begin(), ch.oddHist.end(), T(0));
            }
        }

        /**
         * @brief One channel of the 2x upsampler: dst[2i] is the FIR phase,
         *        dst[2i + 1] the centre-tap delay. src may be dst itself (the
         *        history copy is what the phases read).
         */
        void upsampleChannel(const T* src, T* dst, ChannelState& state, int n) noexcept
        {
            constexpr int W = simd::kVecWidth<T>;
            using O = simd::Vec<T, W>;
            const int numTaps = static_cast<int>(evenTaps.size());
            auto& hist = state.history;

            // Append the new block after the fixed numTaps-sample history prefix.
            std::memmove(hist.data() + numTaps, src, static_cast<std::size_t>(n) * sizeof(T));

            simd::firCorrelate(hist.data(), evenTaps.data(), numTaps, scratch.data(), n, T(2));

            const T* centre = hist.data() + numTaps - delaySamples;
            const T centre2 = centerTap * T(2);
            const auto c2 = O::set1(centre2);
            int i = 0;
            for (; i + W <= n; i += W)
                O::storeInterleave2(dst + 2 * i, O::load(scratch.data() + i),
                                    O::mul(O::load(centre + i), c2));
            for (; i < n; ++i)
            {
                dst[2 * i] = scratch[static_cast<std::size_t>(i)];
                dst[2 * i + 1] = centre[i] * centre2;
            }

            // Save the trailing numTaps input samples for the next block. Using
            // the current n keeps this correct under variable block sizes.
            std::memmove(hist.data(), hist.data() + n, static_cast<std::size_t>(numTaps) * sizeof(T));
        }

        /**
         * @brief One channel of the 2x decimator over len high-rate samples:
         *        dst[k] = sum_i h[i] x[2k + i] of the full kernel, computed as
         *        the non-zero taps over the even samples plus the centre tap
         *        on the odd ones. dst may be src itself.
         */
        void downsampleChannel(const T* src, T* dst, ChannelState& state, int len) noexcept
        {
            constexpr int W = simd::kVecWidth<T>;
            using O = simd::Vec<T, W>;
            const int outLen = len / 2;
            const int numTaps = static_cast<int>(evenTaps.size());
            T* even = state.evenHist.data();
            T* odd = state.oddHist.data();

            // Split the new samples after the halfOrder-sample prefixes.
            int m = 0;
            for (; m + W <= outLen; m += W)
            {
                typename O::V e, o;
                O::loadDeinterleave(src + 2 * m, e, o);
                O::store(even + halfOrder + m, e);
                O::store(odd + halfOrder + m, o);
            }
            for (; m < outLen; ++m)
            {
                even[halfOrder + m] = src[2 * m];
                odd[halfOrder + m] = src[2 * m + 1];
            }

            simd::firCorrelate(even, evenTaps.data(), numTaps, dst, outLen, T(1));

            // The centre tap meets x[2k + halfOrder] = odd[k + (halfOrder - 1) / 2].
            const T* centre = odd + (halfOrder - 1) / 2;
            const auto c = O::set1(centerTap);
            int k = 0;
            for (; k + W <= outLen; k += W)
                O::store(dst + k, O::madd(c, O::load(centre + k), O::load(dst + k)));
            for (; k < outLen; ++k)
            {
                // Keep the vector multiply-add rounding at block tails too.
                T lanes[W];
                O::store(lanes, O::madd(c, O::set1(centre[k]), O::set1(dst[k])));
                dst[k] = lanes[0];
            }

            std::memmove(even, even + outLen, static_cast<std::size_t>(halfOrder) * sizeof(T));
            std::memmove(odd, odd + outLen, static_cast<std::size_t>(halfOrder) * sizeof(T));
        }

        void processUpsample(AudioBufferView<const T> input, AudioBufferView<T> output, int nCh, int nS) noexcept
        {
            for (int ch = 0; ch < nCh; ++ch)
                upsampleChannel(input.getChannel(ch), output.getChannel(ch),
                                upChannels[static_cast<std::size_t>(ch)], nS);
        }

        void processUpsampleInPlace(AudioBufferView<T> buffer, int nCh, int currentLen) noexcept
        {
            for (int ch = 0; ch < nCh; ++ch)
                upsampleChannel(buffer.getChannel(ch), buffer.getChannel(ch),
                                upChannels[static_cast<std::size_t>(ch)], currentLen);
        }

        void processDownsample(AudioBufferView<const T> input, AudioBufferView<T> output, int nCh, int currentLen) noexcept
        {
            for (int ch = 0; ch < nCh; ++ch)
                downsampleChannel(input.getChannel(ch), output.getChannel(ch),
                                  downChannels[static_cast<std::size_t>(ch)], currentLen);
        }

        void processDownsampleInPlace(AudioBufferView<T> buffer, int nCh, int currentLen) noexcept
        {
            for (int ch = 0; ch < nCh; ++ch)
                downsampleChannel(buffer.getChannel(ch), buffer.getChannel(ch),
                                  downChannels[static_cast<std::size_t>(ch)], currentLen);
        }
    };

    // ========================================================================
    // Polyphase IIR half-band (elliptic allpass pair)
    // ========================================================================

    /// Two allpass branches advanced in lock step: one SIMD register of two
    /// doubles where the target has one, two scalars otherwise. The arithmetic
    /// is the same three operations either way, so both give the same bits.
    template <int W, typename D = double> struct BranchPair
    {
        struct V { D a, b; };
        static V set(D a, D b) noexcept { return { a, b }; }
        static V load(const D* p) noexcept { return { p[0], p[1] }; }
        static void store(D* p, V v) noexcept { p[0] = v.a; p[1] = v.b; }
        static V step(V c, V x, V x1, V y1) noexcept
        {
            return { c.a * (x.a - y1.a) + x1.a, c.b * (x.b - y1.b) + x1.b };
        }
        static D first(V v) noexcept { return v.a; }
        static D second(V v) noexcept { return v.b; }
    };
    template <typename D> struct BranchPair<2, D>
    {
        using O = simd::Vec<D, 2>;
        using V = typename O::V;
        static V set(D a, D b) noexcept { return O::set2(a, b); }
        static V load(const D* p) noexcept { return O::load(p); }
        static void store(D* p, V v) noexcept { O::store(p, v); }
        static V step(V c, V x, V x1, V y1) noexcept
        {
            return O::add(O::mul(c, O::sub(x, y1)), x1);
        }
        static D first(V v) noexcept { return O::lane0(v); }
        static D second(V v) noexcept { return O::lane1(v); }
    };
    using Pair = BranchPair<simd::kVecDoubleNarrow>;

    /**
     * @struct PolyphaseIirHalfBand
     * @brief 2x elliptic half-band as H(z) = (A0(z^2) + z^-1 A1(z^2)) / 2.
     *
     * Each branch is a cascade of first-order sections (a + z^-1)/(1 + a z^-1)
     * at the low rate. Upsampling feeds both branches the same input and
     * interleaves their outputs (A0 to the even sample); decimation feeds the
     * odd input sample to A0 and the even one to A1 and averages, which keeps
     * the full-filter output at odd high-rate instants: one high-rate sample
     * earlier than the even instants, with the same alias rejection. The
     * recursion runs in double regardless of T.
     */
    struct PolyphaseIirHalfBand
    {
        static constexpr int kMaxPairs = 8; ///< Up to 16 coefficients per stage.

        int numPairs = 0;
        alignas(16) std::array<double, 2 * kMaxPairs> coef {}; ///< [A0 k, A1 k] interleaved.

        /// The previous input pair of the cascade followed by each section's
        /// previous output pair. A section's previous input is the previous
        /// output of the section before it, so no separate copy is kept.
        struct ChannelState
        {
            alignas(16) std::array<double, 2 * (kMaxPairs + 1)> s {};
        };
        std::vector<ChannelState> up, down;
        std::vector<T> scratch;

        /**
         * @brief Designs numCoefs (even) allpass coefficients of an elliptic
         *        half-band with the given transition width.
         *
         * transition is the full width of the transition band as a fraction of
         * this stage's OUTPUT rate (the band is centred on a quarter of it).
         * From the passband edge fp = 1/4 - transition/2 the bilinear
         * selectivity is k = tan^2(pi fp) and the nome q follows from the
         * modulus k' = sqrt(1 - k^2) by its standard rapidly converging series.
         * The half-band elliptic filter of odd order n = 2N + 1 has a real pole
         * at s = -1 and N pole pairs on the unit circle of the analog plane;
         * with the pole frequencies Omega_i from the theta-function series
         * (Antoniou, "Digital Filters", sec. 5.5) the real part of pole i is
         * sigma_i = sqrt((1 - k Omega_i^2)(1 - Omega_i^2 / k)) / (1 + Omega_i^2),
         * and the bilinear transform maps it to the z^-2 allpass coefficient
         * a_i = (1 - sigma_i) / (1 + sigma_i). Sorted ascending, the
         * coefficients alternate between the two branches. Stopband depth:
         * 10 log10(1 + q^(-n/2) / 4) dB (the order inequality of the same
         * reference with power-complementary ripples).
         */
        void design(int numCoefs, double transition) noexcept
        {
            constexpr double kPi = 3.14159265358979323846;
            numCoefs = std::clamp(numCoefs & ~1, 2, 2 * kMaxPairs);
            numPairs = numCoefs / 2;
            const double fp = 0.25 - 0.5 * std::clamp(transition, 1e-4, 0.4999);
            const double k = std::tan(kPi * fp) * std::tan(kPi * fp);
            const double q = nome(k);
            const int order = 2 * numCoefs + 1;
            std::array<double, 2 * kMaxPairs> a {};
            for (int i = 1; i <= numCoefs; ++i)
            {
                double num = 0.0, den = 0.0;
                for (int m = 0; m < 20; ++m)
                {
                    const double sign = (m & 1) ? -1.0 : 1.0;
                    num += sign * std::pow(q, m * (m + 1.0)) * std::sin((2 * m + 1) * kPi * i / order);
                    if (m > 0) den += sign * std::pow(q, static_cast<double>(m) * m) * std::cos(2 * m * kPi * i / order);
                }
                const double w = 2.0 * std::pow(q, 0.25) * num / (1.0 + 2.0 * den);
                const double w2 = w * w;
                const double sigma = std::sqrt(std::max(0.0, (1.0 - k * w2) * (1.0 - w2 / k))) / (1.0 + w2);
                a[static_cast<size_t>(i - 1)] = (1.0 - sigma) / (1.0 + sigma);
            }
            std::sort(a.begin(), a.begin() + numCoefs);
            for (int i = 0; i < numCoefs; ++i)
                coef[static_cast<size_t>(i)] = a[static_cast<size_t>(i)]; // even -> A0, odd -> A1
        }

        /** @brief Nome of the elliptic modulus k (k < 1). */
        static double nome(double k) noexcept
        {
            const double kp = std::sqrt(1.0 - k * k);
            const double e = 0.5 * (1.0 - std::sqrt(kp)) / (1.0 + std::sqrt(kp));
            const double e4 = e * e * e * e;
            return e * (1.0 + e4 * (2.0 + e4 * (15.0 + 150.0 * e4)));
        }

        /** @brief Stopband depth in dB of numCoefs coefficients at a transition width. */
        static double attenuationDb(int numCoefs, double transition) noexcept
        {
            constexpr double kPi = 3.14159265358979323846;
            const double fp = 0.25 - 0.5 * transition;
            const double q = nome(std::tan(kPi * fp) * std::tan(kPi * fp));
            return 10.0 * std::log10(1.0 + std::pow(q, -(2.0 * numCoefs + 1.0) / 2.0) / 4.0);
        }

        /** @brief Frequency response H at w radians per sample of the stage output rate. */
        std::complex<double> response(double w) const noexcept
        {
            const std::complex<double> z2 = std::polar(1.0, -2.0 * w);
            std::complex<double> a0 = 1.0, a1 = 1.0;
            for (int k = 0; k < numPairs; ++k)
            {
                const double c0 = coef[static_cast<size_t>(2 * k)];
                const double c1 = coef[static_cast<size_t>(2 * k + 1)];
                a0 *= (c0 + z2) / (1.0 + c0 * z2);
                a1 *= (c1 + z2) / (1.0 + c1 * z2);
            }
            return 0.5 * (a0 + std::polar(1.0, -w) * a1);
        }

        void allocate(int numChannels, int maxInputSamples)
        {
            const auto n = static_cast<size_t>(std::max(0, numChannels));
            up.assign(n, ChannelState {});
            down.assign(n, ChannelState {});
            scratch.assign(static_cast<size_t>(std::max(0, maxInputSamples)), T(0));
        }

        void reset() noexcept
        {
            for (auto& s : up) s = ChannelState {};
            for (auto& s : down) s = ChannelState {};
        }

        /// A block through both branches, P sections each, with the state in
        /// registers for the whole block (P is a compile-time count so the
        /// loops unroll). Up: both branches take x[i] and dst[2i], dst[2i + 1]
        /// receive A0 and A1. Down: A0 takes x[2k + 1], A1 x[2k], dst[k] their
        /// mean. A NaN or Inf that reached the recursion is cleared at the end
        /// of the block, so a single bad sample cannot silence the stream.
        template <int P, bool Up>
        static void runBlock(const double* coefs, ChannelState& st, const T* src, T* dst, int n) noexcept
        {
            using V = typename Pair::V;
            V prev = Pair::load(st.s.data());
            V y1[P];
            V c[P];
            for (int k = 0; k < P; ++k)
            {
                y1[k] = Pair::load(st.s.data() + 2 * (k + 1));
                c[k] = Pair::load(coefs + 2 * k);
            }
            for (int i = 0; i < n; ++i)
            {
                V in;
                if constexpr (Up)
                {
                    const double x = static_cast<double>(src[i]);
                    in = Pair::set(x, x);
                }
                else
                {
                    in = Pair::set(static_cast<double>(src[2 * i + 1]), static_cast<double>(src[2 * i]));
                }
                V xPrev = prev;
                prev = in;
                for (int k = 0; k < P; ++k)
                {
                    const V yOld = y1[k];
                    const V y = Pair::step(c[k], in, xPrev, yOld);
                    xPrev = yOld;
                    y1[k] = y;
                    in = y;
                }
                if constexpr (Up)
                {
                    dst[2 * i] = static_cast<T>(Pair::first(in));
                    dst[2 * i + 1] = static_cast<T>(Pair::second(in));
                }
                else
                {
                    dst[i] = static_cast<T>(0.5 * (Pair::first(in) + Pair::second(in)));
                }
            }
            Pair::store(st.s.data(), prev);
            double sum = Pair::first(prev) + Pair::second(prev);
            for (int k = 0; k < P; ++k)
            {
                Pair::store(st.s.data() + 2 * (k + 1), y1[k]);
                sum += Pair::first(y1[k]) + Pair::second(y1[k]);
            }
            if (!std::isfinite(sum)) st = ChannelState {};
        }

        template <bool Up>
        void dispatch(ChannelState& st, const T* src, T* dst, int n) noexcept
        {
            const double* c = coef.data();
            switch (numPairs)
            {
                case 1: runBlock<1, Up>(c, st, src, dst, n); break;
                case 2: runBlock<2, Up>(c, st, src, dst, n); break;
                case 3: runBlock<3, Up>(c, st, src, dst, n); break;
                case 4: runBlock<4, Up>(c, st, src, dst, n); break;
                case 5: runBlock<5, Up>(c, st, src, dst, n); break;
                case 6: runBlock<6, Up>(c, st, src, dst, n); break;
                case 7: runBlock<7, Up>(c, st, src, dst, n); break;
                default: runBlock<8, Up>(c, st, src, dst, n); break;
            }
        }

        /// dst[2i] = A0(x)[i], dst[2i + 1] = A1(x)[i]. src may be dst.
        void upsampleChannel(const T* src, T* dst, int ch, int n) noexcept
        {
            if (static_cast<size_t>(ch) >= up.size()) return;
            if (src == dst)
            {
                std::memcpy(scratch.data(), src, static_cast<size_t>(n) * sizeof(T));
                src = scratch.data();
            }
            dispatch<true>(up[static_cast<size_t>(ch)], src, dst, n);
        }

        /// dst[k] = (A0(x[2k + 1]) + A1(x[2k])) / 2. dst may be src.
        void downsampleChannel(const T* src, T* dst, int ch, int len) noexcept
        {
            if (static_cast<size_t>(ch) >= down.size()) return;
            dispatch<false>(down[static_cast<size_t>(ch)], src, dst, len / 2);
        }
    };

    struct StageDesign { int taps; double beta; };

    /**
     * Linear-phase stage designs. Stage 0 sets the preset's stop edge and
     * depth. A later stage s only has to stop what folds back into the band
     * stage 0 keeps (up to its stop edge), so it gets the shortest kernel and
     * Kaiser beta whose stopband over that band is no worse than stage 0's
     * own (searched exhaustively over beta), subject to (halfOrder + 1) being
     * a multiple of 2^s so the round-trip delay stays an exact integer.
     * Stopband peaks (stage 0 / 1 / 2 / 3), dB: Low -38.9/-43.7/-44.8/-91.9,
     * Medium -59.5/-64.0/-78.0/-92.5, High -79.2/-80.3/-122.9/-93.1,
     * Maximum -100.1/-109.4/-124.1/-159.3. Depth::Deep keeps each preset's
     * stop edge and takes, per stage, the shortest kernel (same delay rule)
     * reaching -100.1 dB over the same band: Low -104.0/-111.3/-117.7/-159.0,
     * Medium -103.6/-102.2/-119.8/-159.4, High -100.7/-106.3/-122.3/-160.1.
     */
    static constexpr StageDesign firStageDesign(Quality q, Depth depth, int stage) noexcept
    {
        constexpr StageDesign kLow[kMaxStages]  = { { 31, 3.395 },  { 11, 2.90 },  { 7, 2.35 },   { 15, 9.45 } };
        constexpr StageDesign kMed[kMaxStages]  = { { 63, 5.653 },  { 19, 6.25 },  { 15, 7.90 },  { 15, 9.50 } };
        constexpr StageDesign kHigh[kMaxStages] = { { 127, 7.857 }, { 23, 7.95 },  { 23, 12.55 }, { 15, 9.50 } };
        constexpr StageDesign kMax[kMaxStages]  = { { 255, 10.056 },{ 31, 11.20 }, { 23, 12.70 }, { 31, 15.65 } };
        constexpr StageDesign kLowDeep[kMaxStages]  = { { 91, 10.45 },  { 35, 11.35 }, { 23, 12.30 }, { 31, 15.65 } };
        constexpr StageDesign kMedDeep[kMaxStages]  = { { 115, 10.45 }, { 31, 10.40 }, { 23, 12.40 }, { 31, 15.65 } };
        constexpr StageDesign kHighDeep[kMaxStages] = { { 163, 10.10 }, { 31, 10.85 }, { 23, 12.55 }, { 31, 15.65 } };
        const int s = std::clamp(stage, 0, kMaxStages - 1);
        const bool deep = depth == Depth::Deep;
        switch (q)
        {
            case Quality::Low:     return deep ? kLowDeep[s] : kLow[s];
            case Quality::Medium:  return deep ? kMedDeep[s] : kMed[s];
            case Quality::High:    return deep ? kHighDeep[s] : kHigh[s];
            case Quality::Maximum: return kMax[s];
        }
        return kHigh[s];
    }

    /// The design a linear-phase stage runs: the table's, or for the first
    /// stage the one fitted to the host rate (Band::Audible).
    [[nodiscard]] StageDesign stageDesign(int stage) const noexcept
    {
        return stage == 0 ? firstStage_ : firStageDesign(quality_, depth_, stage);
    }

    /// Depth of a preset's first stage, measured, in dB (Depth::Deep: Maximum's).
    static constexpr double firstStageDepthDb(Quality q, Depth depth) noexcept
    {
        if (depth == Depth::Deep) return 100.1;
        switch (q)
        {
            case Quality::Low:     return 38.9;
            case Quality::Medium:  return 59.5;
            case Quality::High:    return 79.2;
            case Quality::Maximum: return 100.1;
        }
        return 79.2;
    }

    /// Top of the band Band::Audible keeps whole, in Hz.
    static constexpr double kAudibleEdgeHz = 20000.0;
    /// Shortest kernel the fit gives (the shortest of the tables).
    static constexpr int kShortestFit = 7;

    /// Kaiser's window parameter for a stopband depth in dB (his estimate).
    static double kaiserBeta(double depthDb) noexcept
    {
        if (depthDb > 50.0) return 0.1102 * (depthDb - 8.7);
        if (depthDb > 21.0) return 0.5842 * std::pow(depthDb - 21.0, 0.4) + 0.07886 * (depthDb - 21.0);
        return 0.0;
    }

    /**
     * Largest response of a half-band kernel over its stopband, in dB: from
     * `edge` (a fraction of the 2x rate) to its Nyquist frequency, on a grid
     * of sixteen points to the lobe. It reads the kernel the stage would run:
     * the centre tap and the taps at odd distances from it (the others are
     * the half-band's zeros and are left out), which pair up as cosines; the
     * cosines of the odd multiples come from one another, without a
     * trigonometric call per tap.
     */
    static double halfBandStopPeakDb(int taps, double beta, double edge)
    {
        const std::vector<double> full = FIRDesign<double>::lowPass(1.0, 0.25, taps, beta);
        const int half = (taps - 1) / 2;
        const int points = std::max(64, 4 * taps);
        double peak = 0.0;
        for (int p = 0; p <= points; ++p)
        {
            const double f = edge + (0.5 - edge) * static_cast<double>(p) / static_cast<double>(points);
            const double theta = 2.0 * 3.14159265358979323846 * f;
            const double twice = 2.0 * std::cos(2.0 * theta);
            double before = std::cos(theta), now = before;   // cos(-theta), cos(theta)
            double sum = full[static_cast<size_t>(half)];
            for (int k = 1; k <= half; k += 2)
            {
                sum += (full[static_cast<size_t>(half + k)] + full[static_cast<size_t>(half - k)]) * now;
                const double next = twice * now - before;       // cos((k + 2) theta)
                before = now;
                now = next;
            }
            peak = std::max(peak, std::abs(sum));
        }
        return 20.0 * std::log10(std::max(peak, 1e-12));
    }

    /**
     * The shortest half-band kernel (half order odd, so the stage's delay is
     * a whole number of base samples) whose stopband over a transition
     * measures `depthDb` or deeper, with the Kaiser window that goes deepest
     * at that length: lengths from two steps under Kaiser's estimate, and
     * for each the window parameter in steps of 0.25 around his and then of
     * 0.05 around the best of those. `otherwise` if none shorter than it does.
     */
    static StageDesign fitHalfBand(double transition, double depthDb, StageDesign otherwise)
    {
        const double edge = 0.25 + 0.5 * transition;
        const double estimate = (depthDb - 7.95) / (14.36 * transition) + 1.0;
        int taps = 4 * static_cast<int>(estimate / 4.0) + 3 - 8;
        if (taps < kShortestFit) taps = kShortestFit;
        const double centre = kaiserBeta(depthDb);
        for (; taps < otherwise.taps; taps += 4)
        {
            double bestBeta = centre, bestPeak = halfBandStopPeakDb(taps, centre, edge);
            for (int i = -8; i <= 16; ++i)
            {
                const double beta = centre + 0.25 * static_cast<double>(i);
                if (i == 0 || beta < 0.5) continue;
                const double peak = halfBandStopPeakDb(taps, beta, edge);
                if (peak < bestPeak) { bestPeak = peak; bestBeta = beta; }
            }
            const double around = bestBeta;
            for (int i = -4; i <= 4; ++i)
            {
                const double beta = around + 0.05 * static_cast<double>(i);
                if (i == 0 || beta < 0.5) continue;
                const double peak = halfBandStopPeakDb(taps, beta, edge);
                if (peak < bestPeak) { bestPeak = peak; bestBeta = beta; }
            }
            if (bestPeak <= -depthDb) return { taps, bestBeta };
        }
        return otherwise;
    }

    /**
     * The first stage for a host rate. Band::Preset: the table's. Band::Audible:
     * the transition from 20 kHz to its image, rate - 20 kHz, is
     * 1/2 - 20 kHz / rate of the 2x rate; where that is wider than the
     * preset's the stage is fitted to it (a rate that is not a positive
     * finite number keeps the preset). Sets the delays the engine reports.
     */
    void fitFirstStage(double sampleRate)
    {
        firstStage_ = firStageDesign(quality_, depth_, 0);
        firstTransition_ = firTransition(quality_);
        if (band_ != Band::Audible || numStages_ == 0) return;

        if (sampleRate > 2.0 * kAudibleEdgeHz && sampleRate < 1e12)
        {
            const double wide = 0.5 - kAudibleEdgeHz / sampleRate;
            if (wide > firstTransition_)
            {
                firstTransition_ = wide;
                if (mode_ == Mode::LinearPhaseFir)
                    firstStage_ = fitHalfBand(wide, firstStageDepthDb(quality_, depth_), firstStage_);
            }
        }
        if (mode_ == Mode::PolyphaseIir)
            iir_[0].design(iirCoefficientCountOver(firstStageDepthDb(quality_, depth_), firstTransition_),
                           firstTransition_);
        computeDelays();
    }

    /// Stage-0 transition of the FIR preset (Kaiser's estimate), as a fraction
    /// of the 2x rate; the IIR stages reuse it so both modes keep the same band.
    static constexpr double firTransition(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Low:     return (40.0 - 7.95) / (14.36 * 30.0);
            case Quality::Medium:  return (60.0 - 7.95) / (14.36 * 62.0);
            case Quality::High:    return (80.0 - 7.95) / (14.36 * 126.0);
            case Quality::Maximum: return (100.0 - 7.95) / (14.36 * 254.0);
        }
        return (80.0 - 7.95) / (14.36 * 126.0);
    }

    /// IIR stage s transition: stage 0 as the FIR preset; later stages pass up
    /// to stage 0's stop edge, (1/2 + t0) of the base rate, and stop its image.
    static double iirTransition(Quality q, int stage) noexcept
    {
        const double t0 = firTransition(q);
        return stage == 0 ? t0 : 0.5 - (1.0 + 2.0 * t0) / static_cast<double>(1 << (stage + 1));
    }

    /// Smallest even coefficient count at least as deep as the FIR preset's
    /// first stage (stage-0 depths: Low 62, Medium 84, High 99, Maximum 109 dB),
    /// or as Maximum's with Depth::Deep.
    static int iirCoefficientCount(Quality q, Depth depth, int stage) noexcept
    {
        return iirCoefficientCountOver(firstStageDepthDb(q, depth), iirTransition(q, stage));
    }

    /// Smallest even coefficient count that reaches a depth over a transition.
    static int iirCoefficientCountOver(double depthDb, double transition) noexcept
    {
        int n = 2;
        while (n < 2 * PolyphaseIirHalfBand::kMaxPairs
               && PolyphaseIirHalfBand::attenuationDb(n, transition) < depthDb)
            n += 2;
        return n;
    }

    /// Low-frequency or round-trip phase delay of the IIR cascade, base samples.
    /// The phase is unwrapped along a frequency sweep from near DC.
    double iirDelay(double w, bool roundTrip) const noexcept
    {
        constexpr int kSteps = 512;
        double total = 0.0;
        for (int s = 0; s < numStages_; ++s)
        {
            const double scale = 1.0 / static_cast<double>(1 << (s + 1));
            double prev = 0.0, unwrapped = 0.0;
            for (int i = 1; i <= kSteps; ++i)
            {
                const double wi = w * scale * static_cast<double>(i) / kSteps;
                const double ph = std::arg(iir_[static_cast<size_t>(s)].response(wi));
                double d = ph - prev;
                while (d > 3.14159265358979323846) d -= 2.0 * 3.14159265358979323846;
                while (d < -3.14159265358979323846) d += 2.0 * 3.14159265358979323846;
                unwrapped += d;
                prev = ph;
            }
            // Down path keeps the odd instant: one high-rate sample of advance.
            const double down = unwrapped + w * scale;
            total += roundTrip ? (unwrapped + down) : down;
        }
        return -total / w;
    }

    void computeDelays() noexcept
    {
        if (numStages_ == 0) { latency_ = 0; decimationDelay_ = 0; return; }
        if (mode_ == Mode::PolyphaseIir)
        {
            const double w = 1e-4;
            latency_ = static_cast<int>(std::lround(iirDelay(w, true)));
            decimationDelay_ = static_cast<int>(std::lround(iirDelay(w, false) * factor_));
            return;
        }
        int latency = 0, decimation = 0;
        for (int s = 0; s < numStages_; ++s)
        {
            const int halfOrder = (stageDesign(s).taps - 1) / 2;
            latency += (halfOrder + 1) / (1 << s);                  // exact by design
            decimation += halfOrder * (factor_ >> (s + 1));
        }
        latency_ = latency;
        decimationDelay_ = decimation;
    }

    int factor_;
    int numStages_;
    Quality quality_;
    Mode mode_;
    Depth depth_;
    Band band_;
    StageDesign firstStage_;          ///< The first stage's design in use (Band::Audible: fitted at prepare()).
    double firstTransition_;          ///< Its transition, as a fraction of the 2x rate.
    int latency_ = 0;
    int decimationDelay_ = 0;
    AudioSpec baseSpec_ {};
    AudioBuffer<T, MaxChannels> upBuffer_;

    std::array<PolyphaseHalfBand, kMaxStages> filters_;
    std::array<PolyphaseIirHalfBand, kMaxStages> iir_;
};

/**
 * @class OversamplingDryPath
 * @brief Carries a dry signal through the same round trip as an Oversampling
 *        engine, so that a dry/wet mix adds in phase at every frequency.
 *
 * A linear-phase engine delays by a whole number of samples at every
 * frequency, and the dry path is that delay. A polyphase IIR engine does not:
 * its round-trip delay is fractional and rises towards the band edge (3.07
 * samples at 1 kHz and 4.23 at 18 kHz for 2x High at 48 kHz), so against a dry
 * signal delayed by the rounded figure a 50 % mix of an untouched wet signal
 * measures a notch above the middle of the band. Here the dry signal runs
 * through a twin of the engine with nothing between its two halves, which is
 * the same linear filter the wet signal crosses: the two add in phase, and a
 * wet path that does nothing nulls against the dry one exactly.
 *
 * The twin costs what the engine's filters cost (at 2x High, about as much
 * again as the round trip itself); the linear-phase case costs a delay.
 *
 * Threading: prepare() and reset() on the setup thread (prepare allocates);
 * process() on the audio thread, noexcept and allocation-free.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels supported.
 */
template <typename T, int MaxChannels = 16>
class OversamplingDryPath
{
public:
    using Engine = Oversampling<T, MaxChannels>;

    /**
     * @brief Sizes the path for an engine.
     * @param spec   The specification the engine was prepared with (base rate).
     * @param engine The engine whose round trip the dry signal has to match,
     *               already prepared with the same specification (with
     *               Oversampling::Band::Audible its delay is known only then).
     *               Its factor, quality, mode, depth, band and latency are read.
     */
    void prepare(const AudioSpec& spec, const Engine& engine)
    {
        twin_.reset();
        delay_.clear();
        delaySamples_ = 0;
        writePos_ = 0;
        numChannels_ = std::clamp(spec.numChannels, 0, MaxChannels);
        if (engine.getFactor() <= 1 || numChannels_ == 0 || spec.maxBlockSize <= 0)
            return;

        if (engine.getMode() == Engine::Mode::PolyphaseIir)
        {
            twin_ = std::make_unique<Engine>(engine.getFactor(), engine.getQuality(),
                                             engine.getMode(), engine.getDepth(), engine.getBand());
            twin_->prepare(spec);
        }
        else
        {
            delaySamples_ = engine.getLatency();
            delay_.assign(static_cast<size_t>(numChannels_) * static_cast<size_t>(std::max(delaySamples_, 1)), T(0));
        }
    }

    /** @brief Clears the path's memory. */
    void reset() noexcept
    {
        if (twin_) twin_->reset();
        std::fill(delay_.begin(), delay_.end(), T(0));
        writePos_ = 0;
    }

    /** @brief The delay of the path in base-rate samples, as the engine reports its own. */
    [[nodiscard]] int getLatency() const noexcept { return twin_ ? twin_->getLatency() : delaySamples_; }

    /**
     * @brief Passes a block of the dry signal through the path, in place.
     * @param dry Base-rate block. Blocks longer than prepare()'s maxBlockSize
     *            keep their tail untouched; channels beyond the prepared
     *            count are left alone.
     */
    void process(AudioBufferView<T> dry) noexcept
    {
        if (twin_)
        {
            [[maybe_unused]] const auto high = twin_->upsample(dry);
            twin_->downsample(dry);
            return;
        }
        if (delaySamples_ <= 0) return;

        const int nCh = std::min(dry.getNumChannels(), numChannels_);
        const int nS = dry.getNumSamples();
        if (nS <= 0) return;
        const size_t length = static_cast<size_t>(delaySamples_);
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* x = dry.getChannel(ch);
            T* line = delay_.data() + static_cast<size_t>(ch) * length;
            size_t pos = writePos_;
            for (int i = 0; i < nS; ++i)
            {
                const T out = line[pos];
                line[pos] = x[i];
                x[i] = out;
                if (++pos == length) pos = 0;
            }
        }
        // Every channel advanced by the same count from the same position.
        writePos_ = (writePos_ + static_cast<size_t>(nS)) % length;
    }

private:
    std::unique_ptr<Engine> twin_;   ///< Polyphase IIR: the engine again, with nothing in between.
    std::vector<T> delay_;           ///< Linear phase: one delay line per channel, side by side.
    int delaySamples_ = 0;
    int numChannels_ = 0;
    size_t writePos_ = 0;
};

} // namespace dspark
