// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Oversampling.h
 * @brief Power-of-two oversampling with polyphase half-band FIR filters.
 *
 * Cascaded 2x stages built on Kaiser-windowed symmetric half-band filters,
 * both directions as polyphase pairs. Upsampling: the even outputs are the
 * FIR phase over the non-zero half-band taps, the odd outputs a pure delay
 * through the centre tap, so zero-stuffed buffers are never materialised.
 * Decimation: the high-rate stream is split into its even and odd samples;
 * the even ones meet the non-zero taps and the odd ones the centre tap, at
 * exactly the positions of the full high-rate convolution (every other tap
 * of a half-band is zero, and it is never multiplied). The FIR phases run
 * through simd::firCorrelate(), register-blocked over outputs
 * (SSE2/AVX/NEON with scalar fallback): about 2x faster than one
 * dot product per output over the full kernel.
 *
 * Features:
 * - Factors 1/2/4/8/16 with four quality presets (31..255 taps per stage).
 * - Block-based streaming; variable block sizes are supported (the per-stage
 *   histories are block-size agnostic).
 * - Exact integer group delay reported by getLatency() for plugin delay
 *   compensation.
 * - Zero allocations after prepare(); upsample()/downsample() are noexcept.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, FIRFilter.h, SimdOps.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "FIRFilter.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
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
 * Threading: prepare() and reset() belong to the setup thread (prepare
 * allocates). upsample() and downsample() belong to the audio thread and are
 * noexcept and allocation-free. There are no runtime setters -- factor and
 * quality are fixed at construction and all filter state is only touched by
 * the audio thread -- so no cross-thread synchronisation is needed.
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
     */
    enum class Quality
    {
        Low,      ///< 31 taps/stage, ~-40 dB stopband. For fast previewing.
        Medium,   ///< 63 taps/stage, ~-60 dB stopband. General purpose.
        High,     ///< 127 taps/stage, ~-80 dB stopband. Professional standard.
        Maximum   ///< 255 taps/stage, ~-100 dB stopband. Mastering grade.
    };

    /**
     * @brief Constructs the oversampling engine.
     * @param factor  Oversampling factor. Must be a power of two (1, 2, 4, 8, 16).
     * @param quality Filter quality preset. Default is High (-80 dB).
     * @pre factor >= 1 and is a power of 2.
     * @note Release builds normalise an invalid factor into range and round it
     *       down to a power of two (debug builds assert).
     */
    explicit Oversampling(int factor = 2, Quality quality = Quality::High)
        : quality_(quality)
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
    }

    /**
     * @brief Pre-allocates buffers and computes FIR filter coefficients.
     * @param spec The audio specification at the base (1x) sample rate.
     * @post The processor is ready to handle up to `spec.maxBlockSize` base samples.
     */
    void prepare(const AudioSpec& spec)
    {
        baseSpec_ = spec;
        upBuffer_.resize(spec.numChannels, spec.maxBlockSize * factor_);

        const int taps = tapsForQuality(quality_);
        const T beta = betaForQuality(quality_);

        for (int stage = 0; stage < numStages_; ++stage)
        {
            // The max block size entering a stage is baseSize * 2^stage
            int stageMaxSamples = spec.maxBlockSize * (1 << stage);
            filters_[stage].design(taps, beta, spec.numChannels, stageMaxSamples);
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
            filters_[i].reset();
    }

    /** @brief Returns the current oversampling factor. */
    [[nodiscard]] int getFactor() const noexcept { return factor_; }

    /** @brief Returns the current quality preset. */
    [[nodiscard]] Quality getQuality() const noexcept { return quality_; }

    /**
     * @brief Calculates the exact group delay latency of the oversampling chain.
     * @return Latency in samples at the BASE (1x) sample rate.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        if (numStages_ == 0) return 0;
        const int halfOrder = (tapsForQuality(quality_) - 1) / 2;
        // Exact up->down round-trip group delay at the base rate. Each 2x half-band
        // stage contributes (halfOrder+1) high-rate samples on the way up and the
        // same on the way down; cascaded over numStages_ and referred to the base
        // rate this telescopes to 2*(halfOrder+1)*(factor-1)/factor. Verified
        // sample-exact against an impulse round-trip for factors 2/4/8/16 and all
        // quality presets. (halfOrder+1 and factor are powers of two -> exact.)
        return 2 * (halfOrder + 1) * (factor_ - 1) / factor_;
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

        // Stage 0: Read directly from input view
        filters_[0].processUpsample(input, upBuffer_.toView(), nCh, nS);

        // Subsequent stages: in-place upsampling inside upBuffer_
        int currentLen = nS * 2;
        for (int stage = 1; stage < numStages_; ++stage)
        {
            filters_[stage].processUpsampleInPlace(upBuffer_.toView(), nCh, currentLen);
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

        int currentLen = nS * factor_;

        for (int stage = numStages_ - 1; stage > 0; --stage)
        {
            filters_[stage].processDownsampleInPlace(upBuffer_.toView(), nCh, currentLen);
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
            // ONLY when halfOrder is odd. All quality presets (taps 31/63/127/255
            // -> halfOrder 15/31/63/127) satisfy this. If you add a preset, keep
            // (taps-1)/2 odd or the filter degenerates to silence.
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
                dst[k] += centerTap * centre[k];

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

    static constexpr int tapsForQuality(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Low:     return 31;
            case Quality::Medium:  return 63;
            case Quality::High:    return 127;
            case Quality::Maximum: return 255;
        }
        return 127;
    }

    static constexpr T betaForQuality(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Low:     return T(3.395);
            case Quality::Medium:  return T(5.653);
            case Quality::High:    return T(7.857);
            case Quality::Maximum: return T(10.056);
        }
        return T(7.857);
    }

    int factor_;
    int numStages_;
    Quality quality_;
    AudioSpec baseSpec_ {};
    AudioBuffer<T, MaxChannels> upBuffer_;

    std::array<PolyphaseHalfBand, kMaxStages> filters_;
};

} // namespace dspark
