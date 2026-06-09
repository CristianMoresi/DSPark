// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Oversampling.h
 * @brief Real-time oversampling using Polyphase Half-Band FIR filters.
 *
 * Provides CPU-efficient upsampling and downsampling to reduce aliasing in nonlinear 
 * processing. This implementation uses a polyphase decomposition of Kaiser-windowed 
 * symmetric half-band filters. 
 *
 * By exploiting the polyphase structure, zero-stuffing and discarding samples are 
 * avoided entirely. The process is block-based and linear in memory, allowing 
 * modern compilers to auto-vectorize (SIMD) the convolution loops natively. All
 * in-place operations are memory-safe and optimized for CPU cache.
 *
 * @note Memory is strictly pre-allocated in `prepare()`. No allocations occur 
 * during the `process()` thread.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "FIRFilter.h"

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
     */
    explicit Oversampling(int factor = 2, Quality quality = Quality::High)
        : factor_(factor), quality_(quality)
    {
        assert(factor >= 1 && (factor & (factor - 1)) == 0 && "Factor must be a power of 2");
        assert(factor <= (1 << kMaxStages) && "Factor must be <= 16 (kMaxStages stages)");
        // Clamp for Release safety: only kMaxStages polyphase stages exist, so a
        // factor above 2^kMaxStages would index filters_ out of bounds.
        factor_ = std::clamp(factor, 1, 1 << kMaxStages);
        numStages_ = 0;
        int f = factor_;
        while (f > 1) { ++numStages_; f >>= 1; }
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
     * @param input View of the base-rate audio buffer.
     * @return A view of the internal oversampled buffer (length = input.samples * factor).
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
     * @param output Buffer view where the base-rate samples will be written.
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
        std::vector<T> evenTaps;   ///< Even-index polyphase taps (upsample FIR phase).
        std::vector<T> fullTaps;   ///< Full symmetric half-band (downsample, exact alignment).
        T centerTap = T(0);
        int halfOrder = 0;
        int delaySamples = 0;

        /**
         * @brief State tracking per channel, optimizing memory alignment for SIMD.
         */
        struct ChannelState {
            std::vector<T> history;    ///< Contiguous sliding-window history for the FIR convolution.
        };
        std::vector<ChannelState> upChannels;
        std::vector<ChannelState> downChannels;

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
            // → halfOrder 15/31/63/127) satisfy this. If you add a preset, keep
            // (taps-1)/2 odd or the filter degenerates to silence.
            assert((halfOrder & 1) == 1 && "half-band polyphase requires odd halfOrder");
            // Pure-delay alignment for the centre-tap (odd) phase. With the full
            // even branch (halfOrder+1 taps, group delay (halfOrder)/2 = numTaps/2
            // base samples) the centre sample must sit numTaps/2 = (halfOrder+1)/2
            // samples into the window to stay phase-aligned with the even phase.
            delaySamples = (halfOrder + 1) / 2;

            auto fullCoeffs = FIRDesign<T>::lowPass(1.0, 0.25, taps, beta);
            centerTap = fullCoeffs[static_cast<std::size_t>(halfOrder)];

            // Even polyphase branch = ALL even-array-index coefficients of the
            // symmetric half-band (these are the non-zero off-centre taps, since
            // the centre sits at the odd index halfOrder). Used by the upsample
            // FIR phase. (halfOrder is odd, so i never equals halfOrder.)
            evenTaps.clear();
            for (int i = 0; i < taps; i += 2)
                evenTaps.push_back(fullCoeffs[static_cast<std::size_t>(i)]);

            // Full symmetric kernel for the downsampler. Decimation is done as a
            // single, unambiguous high-rate convolution (filter then pick every
            // other sample), which keeps every tap aligned to its true high-rate
            // position — unlike an even/odd polyphase split with independent
            // delays, which left a 0.5-sample mismatch between the FIR and centre
            // branches and capped the round-trip SNR.
            fullTaps.assign(fullCoeffs.begin(), fullCoeffs.end());

            // Up history: base-rate (numTaps + block). Down history: high-rate
            // ((taps-1) + up to 2*maxBlockSamples high-rate input for this stage).
            const int upHistorySize   = static_cast<int>(evenTaps.size()) + maxBlockSamples;
            const int downHistorySize = (taps - 1) + 2 * maxBlockSamples;

            upChannels.resize(static_cast<std::size_t>(numChannels));
            downChannels.resize(static_cast<std::size_t>(numChannels));

            for (int ch = 0; ch < numChannels; ++ch)
            {
                upChannels[ch].history.assign(static_cast<std::size_t>(upHistorySize), T(0));
                downChannels[ch].history.assign(static_cast<std::size_t>(downHistorySize), T(0));
            }
        }

        /**
         * @brief Clears the history buffers to prevent audio clicks on transport reset.
         */
        void reset() noexcept
        {
            for (auto& ch : upChannels)
                std::fill(ch.history.begin(), ch.history.end(), T(0));
            for (auto& ch : downChannels)
                std::fill(ch.history.begin(), ch.history.end(), T(0));
        }

        void processUpsample(AudioBufferView<const T> input, AudioBufferView<T> output, int nCh, int nS) noexcept
        {
            const int numTaps = static_cast<int>(evenTaps.size());
            
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* src = input.getChannel(ch);
                T* dst = output.getChannel(ch);
                auto& hist = upChannels[ch].history;

                // Append the new block after the fixed numTaps-sample history prefix.
                std::memcpy(hist.data() + numTaps, src, nS * sizeof(T));

                // Hint for compiler auto-vectorization
                T* __restrict dstR = dst;
                const T* __restrict histR = hist.data();
                const T* __restrict tapsR = evenTaps.data();

                for (int i = 0; i < nS; ++i)
                {
                    T sum = T(0);
                    // This inner loop will now safely auto-vectorize
                    for (int k = 0; k < numTaps; ++k)
                        sum += tapsR[k] * histR[i + k];
                    
                    dstR[i * 2] = sum * T(2); 
                    dstR[i * 2 + 1] = histR[i + numTaps - delaySamples] * centerTap * T(2);
                }

                // Save the trailing numTaps input samples for the next block. Using
                // the current nS keeps this correct under variable block sizes.
                std::memmove(hist.data(), hist.data() + nS, numTaps * sizeof(T));
            }
        }

        void processUpsampleInPlace(AudioBufferView<T> buffer, int nCh, int currentLen) noexcept
        {
            const int numTaps = static_cast<int>(evenTaps.size());
            
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch); 
                auto& hist = upChannels[ch].history;

                // Append the new block after the fixed numTaps-sample history prefix.
                std::memcpy(hist.data() + numTaps, data, currentLen * sizeof(T));

                T* __restrict dataR = data;
                const T* __restrict histR = hist.data();
                const T* __restrict tapsR = evenTaps.data();

                for (int i = 0; i < currentLen; ++i)
                {
                    T sum = T(0);
                    for (int k = 0; k < numTaps; ++k)
                        sum += tapsR[k] * histR[i + k];
                    
                    dataR[i * 2] = sum * T(2); 
                    dataR[i * 2 + 1] = histR[i + numTaps - delaySamples] * centerTap * T(2);
                }

                // Save the trailing numTaps input samples for the next block. Using
                // the current length keeps this correct under variable block sizes.
                std::memmove(hist.data(), hist.data() + currentLen, numTaps * sizeof(T));
            }
        }

        // Decimating half-band as a single high-rate convolution: filter the
        // incoming high-rate stream with the full symmetric kernel, then keep one
        // sample in two. Every tap multiplies its true high-rate neighbour, so the
        // FIR and centre contributions stay perfectly aligned (no polyphase
        // 0.5-sample mismatch). The forward form histR[2n+k] exploits kernel
        // symmetry (h[k]==h[nTaps-1-k]) for clean auto-vectorisation.
        //
        // hist layout (per channel, block-size agnostic): hist[0..hLen-1] carries
        // the previous block's trailing hLen high-rate samples; the new block is
        // copied to hist[hLen..]. The trailing hLen samples are re-saved afterwards.
        void processDownsample(AudioBufferView<const T> input, AudioBufferView<T> output, int nCh, int currentLen) noexcept
        {
            const int outLen = currentLen / 2;
            const int nTaps  = static_cast<int>(fullTaps.size());
            const int hLen   = nTaps - 1;

            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* src = input.getChannel(ch);
                T* dst = output.getChannel(ch);
                auto& hist = downChannels[ch].history;

                std::memcpy(hist.data() + hLen, src, static_cast<std::size_t>(currentLen) * sizeof(T));

                const T* __restrict h     = fullTaps.data();
                const T* __restrict histR = hist.data();
                T* __restrict dstR        = dst;

                for (int n = 0; n < outLen; ++n)
                {
                    const int base = 2 * n;
                    T sum = T(0);
                    for (int k = 0; k < nTaps; ++k)
                        sum += h[k] * histR[base + k];
                    dstR[n] = sum;
                }

                std::memmove(hist.data(), hist.data() + currentLen, static_cast<std::size_t>(hLen) * sizeof(T));
            }
        }

        void processDownsampleInPlace(AudioBufferView<T> buffer, int nCh, int currentLen) noexcept
        {
            const int outLen = currentLen / 2;
            const int nTaps  = static_cast<int>(fullTaps.size());
            const int hLen   = nTaps - 1;

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                auto& hist = downChannels[ch].history;

                std::memcpy(hist.data() + hLen, data, static_cast<std::size_t>(currentLen) * sizeof(T));

                const T* __restrict h     = fullTaps.data();
                const T* __restrict histR = hist.data();
                T* __restrict dataR       = data;

                for (int n = 0; n < outLen; ++n)
                {
                    const int base = 2 * n;
                    T sum = T(0);
                    for (int k = 0; k < nTaps; ++k)
                        sum += h[k] * histR[base + k];
                    dataR[n] = sum;
                }

                std::memmove(hist.data(), hist.data() + currentLen, static_cast<std::size_t>(hLen) * sizeof(T));
            }
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