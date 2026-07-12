// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Resampler.h
 * @brief High-quality sample rate converter for audio signals.
 *
 * Converts audio between different sample rates (e.g., 44100 <-> 48000 <->
 * 96000 Hz) with configurable quality. Uses polyphase windowed-sinc
 * interpolation (Kaiser window, 256 tabulated phases + linear phase
 * interpolation, SIMD dot-product kernels) -- the method used in
 * professional DAWs and mastering tools.
 *
 * Two modes:
 *
 * - **Offline (batch)**: process() converts an entire buffer at once and is
 *   time-aligned (zero latency -- the symmetric kernel reads future samples).
 *   Best for file processing.
 * - **Streaming**: processBlock() converts chunks incrementally and is
 *   causal: the output is delayed by getLatency() samples. Best for
 *   real-time applications where audio arrives in blocks.
 *
 * Threading: prepare() allocates and must run on a setup thread. After
 * prepare(), processBlock() and reset() are allocation-free and belong to
 * the single stream owner; there are no cross-thread parameter setters.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, SimdOps.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "SimdOps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class Resampler
 * @brief Windowed-sinc sample rate converter optimized for real-time DSP.
 *
 * Quality tiers trade CPU for image/alias rejection and passband flatness.
 * Measured on the float instantiation (each column states its test):
 *
 * | Quality | Taps | 20 kHz image (48->96) | 26 kHz alias (96->44.1) | 20 kHz gain (44.1->48) |
 * |---------|------|-----------------------|-------------------------|------------------------|
 * | Draft   | 8    | -12 dB                | -10 dB                  | -3.3 dB                |
 * | Normal  | 32   | -56 dB                | -27 dB                  | -0.6 dB                |
 * | High    | 64   | -140 dB               | -69 dB                  | -0.02 dB               |
 * | Ultra   | 128  | -139 dB               | -143 dB                 | flat                   |
 *
 * The middle column probes the transition band right at the target Nyquist:
 * only Ultra's kernel is long enough to keep it in the deep stopband. For
 * extreme downsampling ratios (beyond ~8:1) no fixed tap count can hold the
 * anti-alias transition band; cascade two resamplers instead.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class Resampler
{
public:
    enum class Quality
    {
        Draft,   ///< 8-point sinc, fastest (previews; HF response drops early)
        Normal,  ///< 32-point sinc, balanced (~-56 dB worst-case images)
        High,    ///< 64-point sinc, high quality (~-140 dB beyond the transition band)
        Ultra    ///< 128-point sinc, mastering grade (deep stopband even at the band edge)
    };

    /**
     * @brief Prepares the resampler for a given rate conversion.
     *
     * Allocates internal buffers. MUST be called outside the audio thread.
     *
     * @param sourceRate      Source sample rate in Hz.
     * @param targetRate      Target sample rate in Hz.
     * @param quality         Interpolation quality (default: Normal).
     */
    void prepare(double sourceRate, double targetRate,
                 Quality quality = Quality::Normal)
    {
        // Guard against non-positive rates (would make ratio_ = inf/NaN and
        // poison every downstream division).
        sourceRate_ = std::max(sourceRate, 1.0);
        targetRate_ = std::max(targetRate, 1.0);
        ratio_ = targetRate_ / sourceRate_;
        srcStep_ = 1.0 / ratio_; // source samples advanced per output sample

        sincPoints_ = qualityToSincPoints(quality);
        kaiserBeta_ = qualityToKaiserBeta(quality);

        buildSincTable();
        reset();
    }

    /**
     * @brief Prepares the resampler using AudioSpec (unified API).
     *
     * Pre-allocates multi-channel states. MUST be called outside the audio thread.
     *
     * @param spec       Audio environment (sampleRate, numChannels used).
     * @param targetRate Target sample rate in Hz.
     * @param quality    Interpolation quality (default: Normal).
     */
    void prepare(const AudioSpec& spec, double targetRate,
                 Quality quality = Quality::Normal)
    {
        prepare(spec.sampleRate, targetRate, quality);
        ensureChannelStates(spec.numChannels);
    }

    /**
     * @brief Resets the internal state (delay lines, all channels) to zero.
     *
     * Safe to call from the audio thread after prepare(): the assigns below
     * only refill storage that prepare() already sized (no reallocation).
     */
    void reset() noexcept
    {
        // assign (not fill): after a re-prepare with a higher quality the
        // histories must GROW to the new sincPoints_, otherwise the mirror
        // write at [writePos + sincPoints_] lands past the end of the
        // allocation (confirmed heap overflow under ASan).
        resetChannelState(mono_);
        for (auto& cs : channelStates_)
            resetChannelState(cs);
    }

    /**
     * @brief Resamples an entire buffer (offline batch processing).
     *
     * Stateless and time-aligned: output sample k interpolates the input at
     * position k / getRatio() exactly (no latency; the buffer edges are
     * zero-padded). Allocates the output vector -- offline use only.
     *
     * @param input       Source audio samples.
     * @param inputLength Number of input samples.
     * @return Vector of resampled output samples (ceil(inputLength * ratio)).
     */
    [[nodiscard]] std::vector<T> process(const T* input, int inputLength)
    {
        if (sincTable_.empty()) return {}; // not prepared

        // Clamp through double before the int cast: huge ratios could
        // otherwise overflow the cast itself (undefined behaviour).
        const double outLenD =
            std::ceil(static_cast<double>(inputLength) * ratio_);
        const int outputLength = static_cast<int>(
            std::min(outLenD, static_cast<double>(std::numeric_limits<int>::max())));

        std::vector<T> output(static_cast<size_t>(outputLength));
        const int halfSinc = sincPoints_ / 2;

        for (int outIdx = 0; outIdx < outputLength; ++outIdx)
        {
            // Calculate absolute source position to avoid float drift accumulation
            double srcPos = static_cast<double>(outIdx) / ratio_;
            if (srcPos >= inputLength) break;

            int intPos = static_cast<int>(srcPos);
            double frac = srcPos - static_cast<double>(intPos);

            output[static_cast<size_t>(outIdx)] = interpolateOffline(input, inputLength, intPos, frac, halfSinc);
        }

        return output;
    }

    /**
     * @brief Resamples a block of audio in single-channel streaming mode.
     *
     * @param input       Input audio samples.
     * @param inputLength Number of input samples.
     * @param output      Output buffer (must hold at least getMaxOutputSamples()).
     * @return Number of output samples produced.
     */
    int processBlock(const T* input, int inputLength, T* output) noexcept
    {
        return processChannel(input, inputLength, output, mono_);
    }

    /**
     * @brief Resamples multi-channel audio using AudioBufferView (streaming).
     *
     * Processes each channel sequentially with its independent state.
     *
     * @param input  Input audio buffer view.
     * @param output Output audio buffer view (pre-allocated).
     * @return Number of output samples produced per channel.
     */
    int processBlock(AudioBufferView<T> input, AudioBufferView<T> output) noexcept
    {
        int numCh = std::min(input.getNumChannels(), output.getNumChannels());
        int inLen = input.getNumSamples();

        assert(static_cast<int>(channelStates_.size()) >= numCh &&
               "Resampler channels not allocated! Call prepare(spec...) first.");
        // Release-safe: never index channelStates_ past what prepare() allocated
        // (the assert above flags the missing prepare(spec) in debug builds).
        numCh = std::min(numCh, static_cast<int>(channelStates_.size()));

        int outCount = 0;
        for (int ch = 0; ch < numCh; ++ch)
        {
            outCount = processChannel(input.getChannel(ch), inLen,
                                      output.getChannel(ch),
                                      channelStates_[static_cast<size_t>(ch)]);
        }

        return outCount;
    }

    /**
     * @brief Returns the maximum number of output samples for a given input length.
     * @param inputLength Number of input samples.
     * @return Maximum possible output samples.
     */
    [[nodiscard]] int getMaxOutputSamples(int inputLength) const noexcept
    {
        const double maxOut =
            std::ceil(static_cast<double>(inputLength) * ratio_) + 2.0;
        return static_cast<int>(
            std::min(maxOut, static_cast<double>(std::numeric_limits<int>::max())));
    }

    /** @brief Returns the conversion ratio (targetRate / sourceRate). */
    [[nodiscard]] double getRatio() const noexcept { return ratio_; }

    /**
     * @brief Returns the streaming latency in output samples.
     *
     * The streaming path (processBlock) delays the signal by exactly
     * sincPoints/2 input samples -- the sinc kernel's group delay -- which
     * this getter reports rounded to the nearest output sample. The offline
     * process() path is already time-aligned and has zero latency.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return static_cast<int>(std::round(static_cast<double>(sincPoints_ / 2) * ratio_));
    }

private:
    static constexpr int kOversample = 256;

    static int qualityToSincPoints(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Draft:  return 8;
            case Quality::Normal: return 32;
            case Quality::High:   return 64;
            case Quality::Ultra:  return 128;
        }
        return 32;
    }

    /**
     * @brief Kaiser beta per quality tier.
     *
     * The window's sidelobe level caps the achievable image rejection no
     * matter how many taps the kernel has (beta 10 caps near -100 dB), so
     * beta must scale with the tap budget: the longer kernels spend their
     * extra taps on a deeper stopband, while the 8-tap Draft trades stopband
     * depth for less passband droop. A single beta for all tiers would leave
     * Ultra performing exactly like High, and Draft 4 dB down at 20 kHz.
     */
    static double qualityToKaiserBeta(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Draft:  return 6.0;
            case Quality::Normal: return 10.0;
            case Quality::High:   return 12.5;
            case Quality::Ultra:  return 14.5;
        }
        return 10.0;
    }

    /** @brief Modified Bessel I0 for the continuous Kaiser window. */
    [[nodiscard]] static double besselI0(double x) noexcept
    {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k <= 50; ++k)
        {
            const double half = x / (2.0 * k);
            term *= half * half;
            sum += term;
            if (term < 1e-15 * sum) break;
        }
        return sum;
    }

    void buildSincTable()
    {
        // kOversample + 1 phases: the extra phase holds the frac = 1.0 kernel,
        // so the 2-point phase interpolation in the read path never has to
        // clamp or wrap (exact at both ends of the fractional range).
        sincTable_.resize(static_cast<size_t>((kOversample + 1) * sincPoints_));

        const int halfSinc = sincPoints_ / 2;
        constexpr double kPi = std::numbers::pi;
        const double beta = kaiserBeta_;
        const double i0Beta = besselI0(beta);

        // Apply 0.95 margin on downsampling to prevent transition-band aliasing.
        double cutoff = (ratio_ < 1.0) ? (ratio_ * 0.95) : 1.0;

        for (int phase = 0; phase <= kOversample; ++phase)
        {
            const double frac = static_cast<double>(phase) / static_cast<double>(kOversample);
            const int base = phase * sincPoints_;
            double sum = 0.0;

            for (int tap = 0; tap < sincPoints_; ++tap)
            {
                // Tap alignment: tap j weighs source sample intPos-halfSinc+1+j,
                // so its position relative to the interpolation point
                // intPos + frac is t = (j - halfSinc + 1) - frac. This is the
                // symmetric placement for an even-length kernel: every tap
                // stays inside the open window support (-halfSinc, halfSinc)
                // for frac in (0, 1). Shifting the window one tap earlier
                // would pin the first tap at |t| >= halfSinc where the Kaiser
                // window is exactly zero (a dead tap in every phase) and
                // shorten the group delay to halfSinc-1, breaking getLatency().
                // The MINUS sign on frac is essential: `+ frac` would sample
                // the kernel time-reversed (heavy zipper distortion).
                const double t = static_cast<double>(tap - halfSinc + 1) - frac;
                const double x = t * cutoff;

                double sincVal = (std::abs(x) < 1e-10)
                    ? cutoff
                    : cutoff * std::sin(kPi * x) / (kPi * x);

                // Continuous Kaiser window evaluated at the SAME shifted
                // position as the sinc (a per-tap fixed window leaves a small
                // phase-dependent ripple in the passband).
                const double wx = t / static_cast<double>(halfSinc);
                const double win = (std::abs(wx) >= 1.0)
                    ? 0.0
                    : besselI0(beta * std::sqrt(1.0 - wx * wx)) / i0Beta;

                const double v = sincVal * win;
                sincTable_[static_cast<size_t>(base + tap)] = static_cast<T>(v);
                sum += v;
            }

            // Normalise each polyphase branch to unity DC gain so the passband
            // is flat (otherwise the windowed sinc leaves a small gain ripple).
            if (std::abs(sum) > 1e-12)
            {
                const T inv = static_cast<T>(1.0 / sum);
                for (int tap = 0; tap < sincPoints_; ++tap)
                    sincTable_[static_cast<size_t>(base + tap)] *= inv;
            }
        }
    }

    /**
     * @brief Offline interpolation with boundary checks.
     *
     * Linear interpolation between two adjacent table phases: with 256 phases
     * (plus the explicit frac=1 phase) the phase-quantisation images sit
     * below -90 dB. The in-range fast path dispatches both kernels to the
     * SIMD dot product.
     */
    T interpolateOffline(const T* data, int length, int intPos, double frac, int halfSinc) const noexcept
    {
        const double exactPhase = frac * static_cast<double>(kOversample);
        int p0 = static_cast<int>(exactPhase);
        if (p0 > kOversample - 1) p0 = kOversample - 1; // frac is < 1 by contract
        const T pf = static_cast<T>(exactPhase - static_cast<double>(p0));

        const T* k0 = sincTable_.data() + static_cast<size_t>(p0) * sincPoints_;
        const T* k1 = k0 + sincPoints_; // safe: table holds kOversample+1 phases

        // Tap j weighs data[intPos - halfSinc + 1 + j] (see buildSincTable).
        const int firstSrc = intPos - halfSinc + 1;
        if (firstSrc >= 0 && firstSrc + sincPoints_ <= length)
        {
            // Fully in range: two SIMD dot products + one lerp.
            const T* src = data + firstSrc;
            const T s0 = simd::dotProduct(k0, src, sincPoints_);
            const T s1 = simd::dotProduct(k1, src, sincPoints_);
            return s0 + pf * (s1 - s0);
        }

        // Edge path: zero-pad outside the buffer.
        T result = T(0);
        for (int tap = 0; tap < sincPoints_; ++tap)
        {
            const int srcIdx = firstSrc + tap;
            if (srcIdx < 0 || srcIdx >= length) continue;
            const T kernel = k0[tap] + pf * (k1[tap] - k0[tap]);
            result += data[srcIdx] * kernel;
        }
        return result;
    }

    /**
     * @brief High-performance contiguous memory interpolator.
     *
     * Relies on the double-buffered history allowing linear SIMD loading:
     * two SIMD dot products against adjacent table phases plus one lerp.
     * The window holds the last sincPoints_ input samples (oldest first);
     * the kernel peaks at tap halfSinc-1+frac, so the output stream is
     * delayed by exactly halfSinc input samples (see getLatency()).
     */
    T interpolateFromHistory(const T* historyPtr, int readPos, double frac) const noexcept
    {
        const double exactPhase = frac * static_cast<double>(kOversample);
        int p0 = static_cast<int>(exactPhase);
        if (p0 > kOversample - 1) p0 = kOversample - 1; // frac is < 1 by contract
        const T pf = static_cast<T>(exactPhase - static_cast<double>(p0));

        const T* k0 = sincTable_.data() + static_cast<size_t>(p0) * sincPoints_;
        const T* k1 = k0 + sincPoints_; // safe: table holds kOversample+1 phases

        // Linear memory read from 'readPos'. No modulo required.
        const T* histPtr = &historyPtr[readPos];

        const T s0 = simd::dotProduct(k0, histPtr, sincPoints_);
        const T s1 = simd::dotProduct(k1, histPtr, sincPoints_);
        return s0 + pf * (s1 - s0);
    }

    struct ChannelState
    {
        std::vector<T> history;
        int writePos = 0;
        double fractionalPos = 0.0;
    };

    void resetChannelState(ChannelState& cs)
    {
        cs.history.assign(static_cast<size_t>(sincPoints_ * 2), T(0));
        cs.writePos = 0;
        cs.fractionalPos = 0.0;
    }

    void ensureChannelStates(int numChannels)
    {
        if (static_cast<int>(channelStates_.size()) < numChannels)
            channelStates_.resize(static_cast<size_t>(numChannels));

        // Double size supports the modulo-free mirrored convolution. The size
        // check is against the CURRENT sincPoints_: quality may have changed
        // between prepares, and a stale (smaller) history would overflow.
        const size_t needed = static_cast<size_t>(sincPoints_ * 2);
        for (auto& cs : channelStates_)
        {
            if (cs.history.size() != needed)
            {
                cs.history.assign(needed, T(0));
                cs.writePos = 0;
                cs.fractionalPos = 0.0;
            }
        }
    }

    int processChannel(const T* input, int inputLength, T* output,
                       ChannelState& state) noexcept
    {
        if (state.history.empty()) return 0; // not prepared

        // Hot state lives in locals for the duration of the block: this both
        // tells the compiler the fields cannot alias the output writes (no
        // per-sample reloads) and keeps them in registers.
        T* const hist = state.history.data();
        const int n = sincPoints_;
        const double step = srcStep_;
        int writePos = state.writePos;
        double frac = state.fractionalPos;
        int outIdx = 0;

        for (int i = 0; i < inputLength; ++i)
        {
            // Double-buffered push: mirror write keeps the read window
            // [writePos, writePos + n) contiguous (no modulo).
            const T x = input[i];
            hist[writePos] = x;
            hist[writePos + n] = x;
            if (++writePos >= n)
                writePos = 0;

            while (frac < 1.0)
            {
                output[outIdx++] = interpolateFromHistory(hist, writePos, frac);
                frac += step;
            }

            frac -= 1.0;
        }

        state.writePos = writePos;
        state.fractionalPos = frac;
        return outIdx;
    }

    double sourceRate_ = 44100.0;
    double targetRate_ = 48000.0;
    double ratio_ = 1.0;
    double srcStep_ = 1.0;
    double kaiserBeta_ = 10.0;

    int sincPoints_ = 32;

    std::vector<T> sincTable_;
    ChannelState mono_;
    std::vector<ChannelState> channelStates_;
};

} // namespace dspark
