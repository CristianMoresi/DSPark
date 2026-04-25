// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpectralProcessor.h
 * @brief STFT analysis-modification-synthesis framework for spectral processing.
 *
 * Provides a complete STFT pipeline: input ring → window → FFT → user callback →
 * IFFT → window → overlap-add → output. The user supplies a callback function
 * that operates on the frequency-domain data, enabling spectral compression,
 * resonance suppression, spectral gating, pitch shifting, auto-EQ, and more.
 *
 * Uses Weighted Overlap-Add (WOLA) with Hann window and 50% overlap for
 * perfect reconstruction when the callback is identity.
 *
 * Dependencies: FFT.h, WindowFunctions.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::SpectralProcessor<float> sp;
 *   sp.prepare(spec, 2048);  // 2048-point FFT
 *
 *   // Identity callback (passthrough):
 *   sp.setCallback([](float* freqData, int numBins) {
 *       // freqData is interleaved [re0, im0, re1, im1, ...] with numBins complex bins
 *       // Modify magnitudes/phases here
 *   });
 *
 *   sp.processBlock(buffer);
 *
 *   // Spectral gate example:
 *   sp.setCallback([](float* freqData, int numBins) {
 *       for (int k = 0; k < numBins; ++k) {
 *           float re = freqData[2*k], im = freqData[2*k+1];
 *           float mag = std::sqrt(re*re + im*im);
 *           if (mag < 0.01f) { freqData[2*k] = 0; freqData[2*k+1] = 0; }
 *       }
 *   });
 * @endcode
 */

#include "FFT.h"
#include "WindowFunctions.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class SpectralProcessor
 * @brief STFT analysis-modification-synthesis with user callback.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class SpectralProcessor
{
public:
    /** @brief Callback signature: (freqData, numBins) where freqData is interleaved complex. */
    using SpectralCallback = std::function<void(T* freqData, int numBins)>;

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the processor.
     *
     * @param spec    Audio environment.
     * @param fftSize FFT size (must be power of two, default 2048).
     * @param hopSize Hop size in samples (default 0 = fftSize/2 for 50% overlap).
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048, int hopSize = 0)
    {
        spec_ = spec;

        // Round fftSize to power of 2
        fftSize_ = 4;
        while (fftSize_ < fftSize) fftSize_ <<= 1;

        hopSize_ = (hopSize > 0) ? hopSize : fftSize_ / 2;
        numBins_ = fftSize_ / 2 + 1;

        fft_ = std::make_unique<FFTReal<T>>(fftSize_);

        // sqrt(Hann) window (periodic for WOLA)
        // Analysis sqrt(Hann) * Synthesis sqrt(Hann) = Hann, which is COLA at 50% overlap
        window_.resize(static_cast<size_t>(fftSize_));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        for (auto& w : window_) w = std::sqrt(w);

        // Compute WOLA normalization factor
        // For 50% overlap with Hann: sum of squared windows = constant
        // Normalization = 1 / (sum of window^2 at each output position)
        computeWolaNorm();

        // Per-channel state
        int nCh = spec.numChannels;
        inputRing_.resize(static_cast<size_t>(nCh));
        outputAccum_.resize(static_cast<size_t>(nCh));
        inputPos_.resize(static_cast<size_t>(nCh), 0);
        outputReadPos_.resize(static_cast<size_t>(nCh), 0);

        for (int ch = 0; ch < nCh; ++ch)
        {
            inputRing_[ch].assign(static_cast<size_t>(fftSize_), T(0));
            outputAccum_[ch].assign(static_cast<size_t>(fftSize_ * 2), T(0));
        }

        // Work buffers
        fftIn_.resize(static_cast<size_t>(fftSize_));
        fftOut_.resize(static_cast<size_t>(fftSize_ + 2));
        fftResult_.resize(static_cast<size_t>(fftSize_));

        hopCounter_ = 0;

        prepared_ = true;
    }

    /**
     * @brief Processes audio through the STFT pipeline.
     *
     * Accumulates input samples into a ring buffer, applies the STFT when
     * enough samples are collected (hop size), and writes results to output
     * via overlap-add.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(),
                                 static_cast<int>(inputRing_.size()));
        const int nS  = buffer.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            // Push input samples into ring buffer (all channels)
            for (int ch = 0; ch < nCh; ++ch)
            {
                inputRing_[ch][inputPos_[ch]] = buffer.getChannel(ch)[i];
                inputPos_[ch] = (inputPos_[ch] + 1) % fftSize_;
            }

            ++hopCounter_;

            // Process when we have accumulated hopSize samples
            if (hopCounter_ >= hopSize_)
            {
                hopCounter_ = 0;

                for (int ch = 0; ch < nCh; ++ch)
                    processHop(ch);
            }

            // Read from output accumulator
            for (int ch = 0; ch < nCh; ++ch)
            {
                int& rp = outputReadPos_[ch];
                buffer.getChannel(ch)[i] = outputAccum_[ch][rp];
                outputAccum_[ch][rp] = T(0); // Clear after reading
                rp = (rp + 1) % static_cast<int>(outputAccum_[ch].size());
            }
        }
    }

    /**
     * @brief Sets the spectral modification callback.
     *
     * The callback receives interleaved complex data [re0, im0, re1, im1, ...]
     * with numBins complex bins (fftSize/2 + 1). Modify in-place.
     *
     * @note M10: `setCallback` must be called from a **single thread** at a
     *       time. The double-buffer swap is lock-free versus the audio
     *       thread, but two concurrent writers could race on the same slot
     *       (`std::function::operator=` is not atomic). If you cannot
     *       guarantee this externally, wrap your calls in a mutex.
     */
    void setCallback(SpectralCallback cb)
    {
        int inactive = 1 - activeCallback_.load(std::memory_order_relaxed);
        if (inactive == 0)
            callbackA_ = std::move(cb);
        else
            callbackB_ = std::move(cb);
        callbackPending_.store(true, std::memory_order_release);
    }

    // -- Queries -------------------------------------------------------------

    /** @brief Returns the latency in samples (= fftSize). */
    [[nodiscard]] int getLatency() const noexcept { return fftSize_; }

    [[nodiscard]] int getFFTSize() const noexcept { return fftSize_; }
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }
    [[nodiscard]] int getHopSize() const noexcept { return hopSize_; }

    void reset() noexcept
    {
        for (auto& ring : inputRing_)
            std::fill(ring.begin(), ring.end(), T(0));
        for (auto& acc : outputAccum_)
            std::fill(acc.begin(), acc.end(), T(0));
        std::fill(inputPos_.begin(), inputPos_.end(), 0);
        std::fill(outputReadPos_.begin(), outputReadPos_.end(), 0);
        hopCounter_ = 0;
    }

private:
    /**
     * @brief Processes one STFT hop for a single channel.
     *
     * Steps:
     * 1. Copy fftSize samples from ring buffer (with wrap)
     * 2. Apply analysis window
     * 3. Forward FFT
     * 4. User callback (spectral modification)
     * 5. Inverse FFT
     * 6. Apply synthesis window
     * 7. Overlap-add into output accumulator
     */
    void processHop(int ch) noexcept
    {
        // 1. Copy from ring buffer with windowing
        int readPos = (inputPos_[ch] - fftSize_ + static_cast<int>(inputRing_[ch].size()))
                      % static_cast<int>(inputRing_[ch].size());

        for (int k = 0; k < fftSize_; ++k)
        {
            int idx = (readPos + k) % fftSize_;
            fftIn_[k] = inputRing_[ch][idx] * window_[k]; // analysis window
        }

        // 2. Forward FFT
        fft_->forward(fftIn_.data(), fftOut_.data());

        // 3. User callback (double-buffered for thread safety)
        if (callbackPending_.load(std::memory_order_acquire))
        {
            activeCallback_.store(1 - activeCallback_.load(std::memory_order_relaxed),
                                  std::memory_order_relaxed);
            callbackPending_.store(false, std::memory_order_relaxed);
        }
        {
            int active = activeCallback_.load(std::memory_order_relaxed);
            auto& cb = (active == 0) ? callbackA_ : callbackB_;
            if (cb)
                cb(fftOut_.data(), numBins_);
        }

        // 4. Inverse FFT
        fft_->inverse(fftOut_.data(), fftResult_.data());

        // 5. Synthesis window + overlap-add
        int writePos = outputReadPos_[ch];
        int accumSize = static_cast<int>(outputAccum_[ch].size());

        for (int k = 0; k < fftSize_; ++k)
        {
            int idx = (writePos + k) % accumSize;
            outputAccum_[ch][idx] += fftResult_[k] * window_[k] * wolaNorm_;
        }
    }

    /**
     * @brief Computes the WOLA normalization factor.
     *
     * For Hann window with 50% overlap, the sum of squared windows
     * at each position is constant. The normalization factor ensures
     * unity gain through the analysis-synthesis round trip.
     */
    void computeWolaNorm() noexcept
    {
        int numOverlaps = fftSize_ / hopSize_;
        T maxSum = T(0);
        for (int pos = 0; pos < hopSize_; ++pos)
        {
            T sumSq = T(0);
            for (int hop = 0; hop < numOverlaps; ++hop)
            {
                int idx = pos + hop * hopSize_;
                if (idx < fftSize_)
                {
                    T w = window_[static_cast<size_t>(idx)];
                    sumSq += w * w;
                }
            }
            if (sumSq > maxSum) maxSum = sumSq;
        }
        wolaNorm_ = (maxSum > T(1e-10)) ? T(1) / maxSum : T(1);
    }

    // -- Members -------------------------------------------------------------

    AudioSpec spec_ {};
    bool prepared_ = false;
    int fftSize_ = 2048;
    int hopSize_ = 1024;
    int numBins_ = 1025;

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_;
    T wolaNorm_ = T(1);

    SpectralCallback callbackA_;
    SpectralCallback callbackB_;
    std::atomic<int> activeCallback_ { 0 };
    std::atomic<bool> callbackPending_ { false };

    // Per-channel state
    std::vector<std::vector<T>> inputRing_;
    std::vector<std::vector<T>> outputAccum_;
    std::vector<int> inputPos_;
    std::vector<int> outputReadPos_;

    int hopCounter_ = 0;

    // Work buffers (shared, not per-channel — processed sequentially)
    std::vector<T> fftIn_, fftOut_, fftResult_;
};

} // namespace dspark
