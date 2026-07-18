// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralProcessor.h
 * @brief Zero-allocation streaming STFT framework for spectral processing.
 *
 * Provides a WOLA (Weighted Overlap-Add) STFT pipeline: input ring ->
 * analysis window -> FFT -> user inline callback -> IFFT -> synthesis
 * window -> overlap-add -> output. Analysis and synthesis both use a
 * periodic sqrt-Hann window, so the round-trip window product is Hann and
 * the constant-overlap-add condition holds exactly for every divisor hop.
 *
 * The processing callback is a template parameter, so the compiler inlines
 * the frequency-domain modification directly into the hop loop: no
 * std::function allocation and no virtual dispatch on the audio thread.
 *
 * Latency is exactly fftSize samples and is independent of the caller's
 * block sizes: every STFT frame is anchored to its absolute position in
 * the sample stream, so chopping the stream into arbitrary (even 1-sample)
 * blocks produces bit-identical output.
 *
 * Threading: owner-managed. prepare()/reset() from a setup thread while
 * idle; processBlock() from the audio thread. No cross-thread setters.
 *
 * Dependencies: FFT.h, WindowFunctions.h, AudioSpec.h, AudioBuffer.h,
 * DenormalGuard.h.
 *
 * @code
 *   dspark::SpectralProcessor<float> sp;
 *   sp.prepare(spec, 2048);
 *
 *   // Real-time processing in the audio thread using a lambda:
 *   sp.processBlock(buffer, [](float* freqData, int numBins) {
 *       // freqData is interleaved [re0, im0, re1, im1, ...]
 *       // Modify magnitudes/phases here directly.
 *   });
 * @endcode
 */

#include "FFT.h"
#include "WindowFunctions.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class SpectralProcessor
 * @brief High-performance STFT analysis-modification-synthesis pipeline.
 *
 * Callback contract:
 * - The callback receives the spectrum of one channel as numBins complex
 *   bins in interleaved CCS layout [re0, im0, re1, im1, ...] with
 *   numBins = fftSize/2 + 1. Bin 0 is DC and bin numBins-1 is Nyquist;
 *   their imaginary parts are zero on entry and are ignored by the
 *   inverse transform.
 * - Bins carry the raw transform scale: the forward FFT is unnormalized
 *   and the inverse applies 1/N, so an unmodified spectrum reconstructs
 *   the input exactly. A full-scale sine of amplitude A appears with
 *   magnitude about A*fftSize/2 (times the window's coherent gain).
 * - The callback runs once per channel per hop, in channel order. It is
 *   invoked inside a noexcept audio path and must not throw.
 *
 * Channels beyond the prepared channel count pass through dry. The first
 * fftSize output samples after prepare()/reset() are the windowed ramp-in
 * of the zero-primed analysis ring (silence before input arrives).
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class SpectralProcessor
{
public:
    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Allocates buffers and prepares the WOLA processing state.
     *
     * Not real-time safe (allocates). Any previous stream state is cleared.
     * An invalid spec (non-positive or NaN sample rate, no channels) is
     * ignored, preserving the previous state.
     *
     * @param spec    Audio environment specification.
     * @param fftSize FFT size (default 2048). Must be a power of two; in
     *                release builds any other value is rounded up to the
     *                next power of two within [4, 1<<20].
     * @param hopSize Hop size in samples. Values <= 0 (default) select
     *                fftSize/2 (50% overlap). Clamped to [1, fftSize/2];
     *                non-divisors of fftSize fall back to fftSize/2 (a
     *                non-divisor hop would break the COLA condition and
     *                cause audible amplitude modulation).
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048, int hopSize = 0)
    {
        if (!(spec.sampleRate > 0.0) || spec.numChannels < 1) return;

        // Disarm first: if an allocation below throws, the processor is a
        // safe no-op instead of running half-mutated state (basic guarantee).
        prepared_ = false;
        spec_ = spec;

        assert((fftSize & (fftSize - 1)) == 0 && fftSize >= kMinFftSize
               && "FFT size must be a power of 2");
        // Release-safe: clamp, then round UP to a power of two (never down,
        // so a requested resolution is always met). The clamp bounds the
        // loop; kMaxFftSize is itself a power of two, so no overflow.
        fftSize = std::clamp(fftSize, kMinFftSize, kMaxFftSize);
        int pow2 = kMinFftSize;
        while (pow2 < fftSize) pow2 <<= 1;
        fftSize_ = pow2;
        mask_ = fftSize_ - 1;

        hopSize_ = (hopSize > 0) ? hopSize : fftSize_ / 2;
        hopSize_ = std::clamp(hopSize_, 1, fftSize_ / 2);
        if (fftSize_ % hopSize_ != 0) hopSize_ = fftSize_ / 2;
        numBins_ = fftSize_ / 2 + 1;

        fft_ = std::make_unique<FFTReal<T>>(fftSize_);

        // WOLA window: periodic sqrt-Hann (analysis and synthesis).
        window_.resize(static_cast<std::size_t>(fftSize_));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        for (auto& w : window_) w = std::sqrt(w);

        computeWolaNorm();

        const int nCh = spec.numChannels;
        inputRing_.resize(static_cast<std::size_t>(nCh));
        outputAccum_.resize(static_cast<std::size_t>(nCh));
        inputPos_.assign(static_cast<std::size_t>(nCh), 0);
        outputReadPos_.assign(static_cast<std::size_t>(nCh), 0);

        // The overlap-add accumulator holds twice the FFT size (also pow2).
        accumMask_ = (fftSize_ * 2) - 1;

        for (int ch = 0; ch < nCh; ++ch)
        {
            inputRing_[static_cast<std::size_t>(ch)]
                .assign(static_cast<std::size_t>(fftSize_), T(0));
            outputAccum_[static_cast<std::size_t>(ch)]
                .assign(static_cast<std::size_t>(fftSize_ * 2), T(0));
        }

        // Plain vectors are fine here: the FFT kernels are unaligned-safe
        // by design, so no aligned allocator is required.
        fftIn_.resize(static_cast<std::size_t>(fftSize_));
        fftOut_.resize(static_cast<std::size_t>(fftSize_ + 2));
        fftResult_.resize(static_cast<std::size_t>(fftSize_));

        hopCounter_ = 0;
        prepared_ = true;
    }

    /**
     * @brief Processes a block of audio through the STFT pipeline.
     *
     * In-place; arbitrary block sizes are supported and produce
     * bit-identical output regardless of how the stream is chopped.
     * Inlines the provided callback to manipulate frequency bins without
     * function call overhead. Safe no-op before prepare().
     *
     * @tparam Func Callback type, compatible with
     *              `void(T* freqData, int numBins)`. Must not throw.
     * @param buffer      Audio block to process in-place.
     * @param processFunc User-defined spectral modification function.
     */
    template <typename Func>
    void processBlock(AudioBufferView<T> buffer, Func&& processFunc) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(),
                                 static_cast<int>(inputRing_.size()));
        const int nS  = buffer.getNumSamples();

        // Chunked processing between hop boundaries. Order per chunk:
        // push input, drain output, then fire the hop. Draining BEFORE the
        // hop is what anchors every frame to its absolute stream position
        // (the accumulator write position equals sample index m*hop at the
        // m-th hop no matter how the caller chops the stream); it is also
        // what makes the pipeline in-place safe (input is read into the
        // ring before the drain overwrites it).
        int i = 0;
        while (i < nS)
        {
            const int chunk = std::min(nS - i, hopSize_ - hopCounter_);

            // 1. Push the input chunk into the ring buffers (two contiguous
            //    spans instead of a per-sample masked loop).
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* data = buffer.getChannel(ch) + i;
                auto& ring = inputRing_[static_cast<std::size_t>(ch)];
                const int wp = inputPos_[static_cast<std::size_t>(ch)];
                const int first = std::min(chunk, fftSize_ - wp);
                std::copy_n(data, first, ring.data() + wp);
                std::copy_n(data + first, chunk - first, ring.data());
                inputPos_[static_cast<std::size_t>(ch)] = (wp + chunk) & mask_;
            }

            // 2. Drain the overlap-add accumulator for the chunk (read and
            //    clear in one pass, two contiguous spans).
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch) + i;
                T* acc  = outputAccum_[static_cast<std::size_t>(ch)].data();
                const int rp = outputReadPos_[static_cast<std::size_t>(ch)];
                const int first = std::min(chunk, fftSize_ * 2 - rp);
                for (int k = 0; k < first; ++k)
                {
                    data[k] = acc[rp + k];
                    acc[rp + k] = T(0);
                }
                for (int k = first; k < chunk; ++k)
                {
                    data[k] = acc[k - first];
                    acc[k - first] = T(0);
                }
                outputReadPos_[static_cast<std::size_t>(ch)] = (rp + chunk) & accumMask_;
            }

            // 3. Fire the STFT hop at the boundary.
            hopCounter_ += chunk;
            if (hopCounter_ >= hopSize_)
            {
                hopCounter_ = 0;
                for (int ch = 0; ch < nCh; ++ch)
                    processHop(ch, processFunc);
            }

            i += chunk;
        }
    }

    // -- Queries -------------------------------------------------------------

    /**
     * @brief I/O latency in samples: exactly one FFT frame.
     *
     * Constant and independent of the caller's block sizes.
     */
    [[nodiscard]] int getLatency() const noexcept { return fftSize_; }
    [[nodiscard]] int getFFTSize() const noexcept { return fftSize_; }
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }
    [[nodiscard]] int getHopSize() const noexcept { return hopSize_; }

    /** @brief Clears all internal delay lines and state variables. */
    void reset() noexcept
    {
        for (auto& ring : inputRing_) std::fill(ring.begin(), ring.end(), T(0));
        for (auto& acc : outputAccum_) std::fill(acc.begin(), acc.end(), T(0));
        std::fill(inputPos_.begin(), inputPos_.end(), 0);
        std::fill(outputReadPos_.begin(), outputReadPos_.end(), 0);
        hopCounter_ = 0;
    }

private:
    /**
     * @brief Processes a single channel STFT overlap-add iteration.
     *
     * Called at a hop boundary, after the boundary chunk has been pushed
     * and drained: inputPos_ is the oldest ring sample (ring size equals
     * the frame size) and outputReadPos_ is the accumulator slot of the
     * next output sample, which anchors the frame in stream time.
     */
    template <typename Func>
    void processHop(int ch, Func& processFunc) noexcept
    {
        // 1. Copy from ring buffer with the analysis window, as two
        //    contiguous multiply spans (auto-vectorizable).
        const T* ring = inputRing_[static_cast<std::size_t>(ch)].data();
        const T* win  = window_.data();
        const int readPos = inputPos_[static_cast<std::size_t>(ch)];
        const int head = fftSize_ - readPos;
        for (int k = 0; k < head; ++k)
            fftIn_[static_cast<std::size_t>(k)] = ring[readPos + k] * win[k];
        for (int k = head; k < fftSize_; ++k)
            fftIn_[static_cast<std::size_t>(k)] = ring[k - head] * win[k];

        // 2. Forward FFT
        fft_->forward(fftIn_.data(), fftOut_.data());

        // 3. User callback (inlined directly here)
        processFunc(fftOut_.data(), numBins_);

        // 4. Inverse FFT (FFTReal::inverse already applies the 1/N
        //    normalization, so the analysis->synthesis round-trip is unity
        //    before windowing).
        fft_->inverse(fftOut_.data(), fftResult_.data());

        // 5. Synthesis window and overlap-add, two contiguous spans.
        T* acc = outputAccum_[static_cast<std::size_t>(ch)].data();
        const T* res = fftResult_.data();
        const int writePos = outputReadPos_[static_cast<std::size_t>(ch)];
        const int headOut = std::min(fftSize_, fftSize_ * 2 - writePos);
        for (int k = 0; k < headOut; ++k)
            acc[writePos + k] += res[k] * win[k] * wolaNorm_;
        for (int k = headOut; k < fftSize_; ++k)
            acc[k - headOut] += res[k] * win[k] * wolaNorm_;
    }

    /**
     * @brief Computes the WOLA normalization.
     *
     * For the sqrt-Hann window and a divisor hop the squared-window
     * overlap sum is the same constant at every position (the Hann
     * partition-of-unity property), so dividing by its maximum gives
     * exact unity gain for the STFT round-trip.
     */
    void computeWolaNorm() noexcept
    {
        const int numOverlaps = fftSize_ / hopSize_;
        T maxSum = T(0);
        for (int pos = 0; pos < hopSize_; ++pos)
        {
            T sumSq = T(0);
            for (int hop = 0; hop < numOverlaps; ++hop)
            {
                const int idx = pos + hop * hopSize_;
                if (idx < fftSize_)
                {
                    const T w = window_[static_cast<std::size_t>(idx)];
                    sumSq += w * w;
                }
            }
            if (sumSq > maxSum) maxSum = sumSq;
        }
        wolaNorm_ = (maxSum > T(1e-10)) ? (T(1) / maxSum) : T(1);
    }

    // -- Members -------------------------------------------------------------
    static constexpr int kMinFftSize = 4;       ///< FFTReal's own minimum.
    static constexpr int kMaxFftSize = 1 << 20; ///< Sanity cap for release-safe sizing.

    AudioSpec spec_ {};
    bool prepared_ = false;

    int fftSize_ = 2048;
    int hopSize_ = 1024;
    int numBins_ = 1025;
    int mask_ = 2047;      // Bitwise mask for fftSize_ bounds
    int accumMask_ = 4095; // Bitwise mask for fftSize_*2 bounds

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_;
    T wolaNorm_ = T(1);

    std::vector<std::vector<T>> inputRing_;
    std::vector<std::vector<T>> outputAccum_;
    std::vector<int> inputPos_;
    std::vector<int> outputReadPos_;
    int hopCounter_ = 0;

    // Shared work buffers for sequential channel processing
    std::vector<T> fftIn_, fftOut_, fftResult_;
};

} // namespace dspark
