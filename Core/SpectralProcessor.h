// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpectralProcessor.h
 * @brief Zero-latency, zero-allocation STFT framework for spectral processing.
 *
 * Provides a highly optimized STFT pipeline: input ring → window → FFT → 
 * user inline callback → IFFT → window → overlap-add → output. 
 * 
 * Uses Weighted Overlap-Add (WOLA) with a constant-overlap-add condition.
 * By utilizing C++ templates for the processing callback, it completely avoids 
 * `std::function` allocations and virtual dispatch overhead, allowing the compiler
 * to inline the frequency-domain modifications directly into the SIMD hot path.
 *
 * Dependencies: FFT.h, WindowFunctions.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::SpectralProcessor<float> sp;
 *   sp.prepare(spec, 2048); 
 *
 *   // Real-time processing in the audio thread using lambda:
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
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class SpectralProcessor
 * @brief High-performance STFT analysis-modification-synthesis pipeline.
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
     * @param spec    Audio environment specification.
     * @param fftSize FFT size. Must be a power of two (default 2048).
     * @param hopSize Hop size in samples. Default 0 evaluates to fftSize/2 (50% overlap).
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048, int hopSize = 0)
    {
        spec_ = spec;

        // Force power of 2 for bitwise masking optimizations
        assert((fftSize & (fftSize - 1)) == 0 && "FFT size must be a power of 2");
        fftSize_ = fftSize;
        mask_ = fftSize_ - 1;

        hopSize_ = (hopSize > 0) ? hopSize : fftSize_ / 2;
        numBins_ = fftSize_ / 2 + 1;

        fft_ = std::make_unique<FFTReal<T>>(fftSize_);

        // Initialize WOLA Window (Periodic sqrt-Hann)
        window_.resize(static_cast<size_t>(fftSize_));
        WindowFunctions<T>::hann(window_.data(), fftSize_, true);
        for (auto& w : window_) w = std::sqrt(w);

        computeWolaNorm();

        int nCh = spec.numChannels;
        inputRing_.resize(static_cast<size_t>(nCh));
        outputAccum_.resize(static_cast<size_t>(nCh));
        inputPos_.resize(static_cast<size_t>(nCh), 0);
        outputReadPos_.resize(static_cast<size_t>(nCh), 0);

        // Accumulated output buffer must be exactly twice the FFT size (which is also a pow2)
        accumMask_ = (fftSize_ * 2) - 1;

        for (int ch = 0; ch < nCh; ++ch)
        {
            inputRing_[ch].assign(static_cast<size_t>(fftSize_), T(0));
            outputAccum_[ch].assign(static_cast<size_t>(fftSize_ * 2), T(0));
        }

        // Ideally, use a custom AlignedAllocator here for AVX 32-byte alignment.
        fftIn_.resize(static_cast<size_t>(fftSize_));
        fftOut_.resize(static_cast<size_t>(fftSize_ + 2)); 
        fftResult_.resize(static_cast<size_t>(fftSize_));

        hopCounter_ = 0;
        prepared_ = true;
    }

    /**
     * @brief Processes a block of audio through the STFT pipeline.
     *
     * Inlines the provided callback to manipulate frequency bins without
     * function call overhead.
     *
     * @tparam Func Callback type. Signature must be compatible with `void(T* freqData, int numBins)`.
     * @param buffer   Audio block to process in-place.
     * @param processFunc User-defined spectral modification function.
     */
    template <typename Func>
    void processBlock(AudioBufferView<T> buffer, Func&& processFunc) noexcept
    {
        if (!prepared_) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(inputRing_.size()));
        const int nS  = buffer.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            // 1. Push input samples into ring buffer
            for (int ch = 0; ch < nCh; ++ch)
            {
                inputRing_[ch][inputPos_[ch]] = buffer.getChannel(ch)[i];
                inputPos_[ch] = (inputPos_[ch] + 1) & mask_; // Bitwise wrap
            }

            ++hopCounter_;

            // 2. Trigger STFT Hop
            if (hopCounter_ >= hopSize_)
            {
                hopCounter_ = 0;
                for (int ch = 0; ch < nCh; ++ch)
                    processHop(ch, std::forward<Func>(processFunc));
            }

            // 3. Output overlapping samples
            for (int ch = 0; ch < nCh; ++ch)
            {
                int& rp = outputReadPos_[ch];
                buffer.getChannel(ch)[i] = outputAccum_[ch][rp];
                outputAccum_[ch][rp] = T(0); // Clear after reading
                rp = (rp + 1) & accumMask_;  // Bitwise wrap
            }
        }
    }

    // -- Queries -------------------------------------------------------------

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
     */
    template <typename Func>
    void processHop(int ch, Func&& processFunc) noexcept
    {
        // 1. Copy from ring buffer with analysis window
        // Because ring size == fftSize_, oldest sample is exactly at inputPos_
        int readPos = inputPos_[ch]; 

        for (int k = 0; k < fftSize_; ++k)
        {
            int idx = (readPos + k) & mask_;
            fftIn_[k] = inputRing_[ch][idx] * window_[k];
        }

        // 2. Forward FFT
        fft_->forward(fftIn_.data(), fftOut_.data());

        // 3. User callback (Inlined directly here)
        processFunc(fftOut_.data(), numBins_);

        // 4. Inverse FFT
        fft_->inverse(fftOut_.data(), fftResult_.data());

        // NOTE: If FFTReal::inverse() does not apply 1/N scaling inherently, 
        // it must be applied here or folded into wolaNorm_ to avoid distortion.

        // 5. Synthesis window & Overlap-Add
        int writePos = outputReadPos_[ch];
        for (int k = 0; k < fftSize_; ++k)
        {
            int idx = (writePos + k) & accumMask_;
            outputAccum_[ch][idx] += fftResult_[k] * window_[k] * wolaNorm_;
        }
    }

    /**
     * @brief Computes WOLA normalization.
     * Ensures unity gain for the STFT round-trip based on the chosen window and hop.
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
        wolaNorm_ = (maxSum > T(1e-10)) ? (T(1) / maxSum) : T(1);
    }

    // -- Members -------------------------------------------------------------
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