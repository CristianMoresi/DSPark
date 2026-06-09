// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpectrumAnalyzer.h
 * @brief Real-time FFT-based spectrum analyser for audio visualisation.
 *
 * Provides a highly optimised spectrum analyser that handles the complete pipeline:
 * windowing → FFT → magnitude → smoothing → dB conversion.
 * * @note **Performance Warning:** This class performs FFT and transcendental math 
 * synchronously when enough samples are accumulated. For large FFT sizes (>2048) 
 * on tight audio callbacks, consider a lock-free queue architecture where the 
 * GUI thread pulls samples and computes the FFT independently to avoid audio dropouts.
 *
 * @code
 * dspark::SpectrumAnalyzer<float> analyzer;
 * analyzer.prepare(48000.0, 2048);  // 48 kHz, 2048-point FFT
 *
 * // In audio callback:
 * analyzer.pushSamples(buffer.getChannel(0), numSamples);
 *
 * // In GUI paint (Timer at 60Hz):
 * if (analyzer.isNewDataReady()) {
 * const float* spectrum = analyzer.getMagnitudesDb();
 * // Draw spectrum...
 * }
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <vector>
#include <memory>

namespace dspark {

/**
 * @class SpectrumAnalyzer
 * @brief Real-time FFT spectrum analyser with unified state smoothing and SIMD-friendly loops.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class SpectrumAnalyzer
{
public:
    /** @brief Available window types for the FFT analysis. */
    enum class WindowType
    {
        Hann,           ///< Default. Good general-purpose choice.
        Hamming,        ///< Slightly better side lobe rejection.
        Blackman,       ///< High dynamic range.
        BlackmanHarris, ///< Highest side lobe rejection.
        FlatTop,        ///< Amplitude-accurate measurement.
        Rectangular     ///< No windowing (transient analysis).
    };

    /**
     * @brief Prepares the analyser and allocates all necessary buffers.
     * * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (must be power of two, 256–16384).
     * @param windowType Window function to use (default: Hann).
     */
    void prepare(double sampleRate, int fftSize = 2048, WindowType windowType = WindowType::Hann)
    {
        assert(fftSize >= 256 && (fftSize & (fftSize - 1)) == 0);

        sampleRate_ = sampleRate;
        fftSize_ = fftSize;
        numBins_ = fftSize / 2 + 1;
        hopSize_ = fftSize / 2; // 50% overlap

        fft_ = std::make_unique<FFTReal<T>>(fftSize);

        window_.resize(static_cast<size_t>(fftSize));
        generateWindow(windowType);

        windowGain_ = WindowFunctions<T>::coherentGain(window_.data(), fftSize);
        if (windowGain_ < T(0.001)) windowGain_ = T(1);
        invGain_ = T(2) / (static_cast<T>(fftSize_) * windowGain_);

        inputRing_.assign(static_cast<size_t>(fftSize), T(0));
        fftBuffer_.resize(static_cast<size_t>(fftSize));
        freqBuffer_.resize(static_cast<size_t>(fftSize + 2));

        // Unified DSP State
        magnitudesState_.assign(static_cast<size_t>(numBins_), T(0));
        peakState_.assign(static_cast<size_t>(numBins_), T(-100));

        // Triple buffer for tear-free cross-thread reading
        for (auto& slot : outSlots_)
        {
            slot.magnitudesDb.assign(static_cast<size_t>(numBins_), T(-100));
            slot.peakDb.assign(static_cast<size_t>(numBins_), T(-100));
        }
        writeSlot_ = 0;
        readSlot_  = 1;
        pendingSlot_.store(2, std::memory_order_relaxed);

        ringWritePos_ = 0;
        ringMask_ = fftSize_ - 1; // power of two guaranteed by the assert above
        samplesUntilFFT_ = hopSize_;
        newDataReady_.store(false, std::memory_order_relaxed);
    }

    /** @brief Resets all internal buffers and state to zero/floor values. */
    void reset() noexcept
    {
        std::fill(inputRing_.begin(), inputRing_.end(), T(0));
        std::fill(magnitudesState_.begin(), magnitudesState_.end(), T(0));
        std::fill(peakState_.begin(), peakState_.end(), floorDb_);

        for (auto& slot : outSlots_)
        {
            std::fill(slot.magnitudesDb.begin(), slot.magnitudesDb.end(), floorDb_);
            std::fill(slot.peakDb.begin(), slot.peakDb.end(), floorDb_);
        }

        ringWritePos_ = 0;
        samplesUntilFFT_ = hopSize_;
    }

    void setSmoothing(T factor) noexcept { smoothing_ = std::clamp(factor, T(0), T(0.99)); }
    void setPeakDecay(T decayDbPerSecond) noexcept { peakDecayRate_ = decayDbPerSecond; }
    void setPeakHoldEnabled(bool enabled) noexcept { peakHoldEnabled_ = enabled; }
    void setFloorDb(T floorDb) noexcept { floorDb_ = floorDb; }

    /**
     * @brief Pushes audio samples into the analyser's internal ring buffer.
     *
     * Computes the FFT synchronously when the internal hop size (50% overlap) is met.
     *
     * @param samples Pointer to the continuous audio data.
     * @param numSamples Number of samples to process.
     */
    void pushSamples(const T* samples, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            inputRing_[static_cast<size_t>(ringWritePos_)] = samples[i];
            ringWritePos_ = (ringWritePos_ + 1) & ringMask_; // pow2 mask, no division

            if (--samplesUntilFFT_ <= 0)
            {
                computeSpectrum();
                samplesUntilFFT_ = hopSize_; // Restart countdown with 50% overlap
            }
        }
    }

    /**
     * @brief Returns the current magnitude spectrum in decibels.
     * @return Pointer to an array of size getNumBins().
     */
    [[nodiscard]] const T* getMagnitudesDb() const noexcept
    {
        acquireLatestSlot();
        return outSlots_[static_cast<size_t>(readSlot_)].magnitudesDb.data();
    }

    /**
     * @brief Returns the peak-hold spectrum in decibels.
     * @return Pointer to an array of size getNumBins().
     */
    [[nodiscard]] const T* getPeakHoldDb() const noexcept
    {
        acquireLatestSlot();
        return outSlots_[static_cast<size_t>(readSlot_)].peakDb.data();
    }

    /** @brief Consumes and returns the new data flag. True if updated since last call. */
    [[nodiscard]] bool isNewDataReady() noexcept
    {
        return newDataReady_.exchange(false, std::memory_order_relaxed);
    }

    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }
    [[nodiscard]] int getFFTSize() const noexcept { return fftSize_; }
    [[nodiscard]] T binToFrequency(int bin) const noexcept { return static_cast<T>(bin) * static_cast<T>(sampleRate_) / static_cast<T>(fftSize_); }

private:
    void generateWindow(WindowType type)
    {
        switch (type)
        {
            case WindowType::Hann: WindowFunctions<T>::hann(window_.data(), fftSize_); break;
            case WindowType::Hamming: WindowFunctions<T>::hamming(window_.data(), fftSize_); break;
            case WindowType::Blackman: WindowFunctions<T>::blackman(window_.data(), fftSize_); break;
            case WindowType::BlackmanHarris: WindowFunctions<T>::blackmanHarris(window_.data(), fftSize_); break;
            case WindowType::FlatTop: WindowFunctions<T>::flatTop(window_.data(), fftSize_); break;
            case WindowType::Rectangular: WindowFunctions<T>::rectangular(window_.data(), fftSize_); break;
        }
    }

    void computeSpectrum() noexcept
    {
        // 1. Copy & Window (Auto-vectorizable; pow2 mask instead of division)
        for (int i = 0; i < fftSize_; ++i)
        {
            int ringIdx = (ringWritePos_ + i) & ringMask_;
            fftBuffer_[static_cast<size_t>(i)] = inputRing_[static_cast<size_t>(ringIdx)] * window_[static_cast<size_t>(i)];
        }

        // 2. Forward FFT
        fft_->forward(fftBuffer_.data(), freqBuffer_.data());

        // 3. Magnitude Calculation & Smoothing (SIMD friendly logic)
        const T oneMinusSmooth = T(1) - smoothing_;

        for (int k = 0; k < numBins_; ++k)
        {
            T re = freqBuffer_[static_cast<size_t>(2 * k)];
            T im = freqBuffer_[static_cast<size_t>(2 * k + 1)];
            T mag = std::sqrt(re * re + im * im) * invGain_;

            // DC and Nyquist carry no mirrored bin, so the single-sided 2x in
            // invGain_ must be undone HERE, on the fresh magnitude. (Scaling
            // the smoothed STATE compounded 0.5x every frame and parked those
            // bins ~9.5 dB low in steady state.)
            if (k == 0 || k == numBins_ - 1) mag *= T(0.5);

            magnitudesState_[static_cast<size_t>(k)] =
                smoothing_ * magnitudesState_[static_cast<size_t>(k)] + oneMinusSmooth * mag;
        }

        // 4. Time-Domain Peak Decay Calculation (Using real hopSize_)
        const T peakDecayDb = peakDecayRate_ * static_cast<T>(hopSize_) / static_cast<T>(sampleRate_);

        // 5. Write into the writer-owned slot of the triple buffer
        auto& slot = outSlots_[static_cast<size_t>(writeSlot_)];

        // 6. DB Conversion and Peak Hold (SIMD friendly logic)
        for (int k = 0; k < numBins_; ++k)
        {
            T dB = gainToDecibels(magnitudesState_[static_cast<size_t>(k)], floorDb_);
            slot.magnitudesDb[static_cast<size_t>(k)] = dB;

            if (peakHoldEnabled_)
            {
                T& peak = peakState_[static_cast<size_t>(k)];
                peak = (dB > peak) ? dB : std::max(floorDb_, peak - peakDecayDb);
                slot.peakDb[static_cast<size_t>(k)] = peak;
            }
        }

        // 7. Publish: swap the finished slot into 'pending' (with the fresh
        // bit) and adopt whatever slot was there as the next write target.
        // Wait-free; the reader can never observe a slot mid-write.
        const int old = pendingSlot_.exchange(writeSlot_ | kFreshBit, std::memory_order_acq_rel);
        writeSlot_ = old & kSlotMask;
        newDataReady_.store(true, std::memory_order_release);
    }

    double sampleRate_ = 48000.0;
    int fftSize_ = 2048;
    int numBins_ = 1025;
    int hopSize_ = 1024;

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_;
    T windowGain_ = T(1);
    T invGain_ = T(1);

    std::vector<T> inputRing_;
    int ringWritePos_ = 0;
    int samplesUntilFFT_ = 0;

    std::vector<T> fftBuffer_;
    std::vector<T> freqBuffer_;

    // Single source of truth for DSP state
    std::vector<T> magnitudesState_;
    std::vector<T> peakState_;

    // Tear-free triple buffer: writer owns writeSlot_, the reader owns
    // readSlot_, and pendingSlot_ (atomic, with a freshness bit) carries the
    // hand-off. Classic wait-free GUI metering scheme.
    static constexpr int kFreshBit = 4;
    static constexpr int kSlotMask = 3;

    struct OutSlot
    {
        std::vector<T> magnitudesDb;
        std::vector<T> peakDb;
    };
    std::array<OutSlot, 3> outSlots_;
    int writeSlot_ = 0;                       // writer-thread private
    mutable int readSlot_ = 1;                // reader-thread private
    mutable std::atomic<int> pendingSlot_{ 2 };

    /** @brief Reader side: adopt the freshest published slot, if any. */
    void acquireLatestSlot() const noexcept
    {
        int expected = pendingSlot_.load(std::memory_order_acquire);
        while (expected & kFreshBit)
        {
            if (pendingSlot_.compare_exchange_weak(expected, readSlot_,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire))
            {
                readSlot_ = expected & kSlotMask;
                return;
            }
        }
    }

    int ringMask_ = 2047;

    T smoothing_ = T(0.8);
    T peakDecayRate_ = T(10);
    T floorDb_ = T(-100);
    bool peakHoldEnabled_ = false;

    std::atomic<bool> newDataReady_{ false };
};

} // namespace dspark