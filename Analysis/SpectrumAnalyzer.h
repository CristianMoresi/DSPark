// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpectrumAnalyzer.h
 * @brief Real-time FFT-based spectrum analyser for audio visualisation.
 *
 * Provides a ready-to-use spectrum analyser that handles the complete pipeline:
 * windowing → FFT → magnitude → smoothing → dB conversion. Designed to feed
 * GUI spectrum displays with minimal setup.
 *
 * Features:
 * - Configurable FFT size (256 to 16384)
 * - Configurable window function
 * - Peak hold with decay
 * - Exponential smoothing for stable display
 * - Thread-safe: push audio from the audio thread, read spectrum from the GUI thread
 *
 * Dependencies: FFT.h, WindowFunctions.h, DspMath.h.
 *
 * @code
 *   dspark::SpectrumAnalyzer<float> analyzer;
 *   analyzer.prepare(48000.0, 2048);  // 48 kHz, 2048-point FFT
 *
 *   // In audio callback:
 *   analyzer.pushSamples(buffer.getChannel(0), numSamples);
 *
 *   // In GUI paint:
 *   const float* spectrum = analyzer.getMagnitudesDb();
 *   int numBins = analyzer.getNumBins();
 *   for (int k = 0; k < numBins; ++k)
 *   {
 *       float freq = analyzer.binToFrequency(k);
 *       float dB = spectrum[k];  // Already in dB
 *       // Draw...
 *   }
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class SpectrumAnalyzer
 * @brief Real-time FFT spectrum analyser with smoothing and peak hold.
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
     * @brief Prepares the analyser for use.
     *
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (must be power of two, 256–16384).
     * @param windowType Window function to use (default: Hann).
     */
    void prepare(double sampleRate, int fftSize = 2048,
                 WindowType windowType = WindowType::Hann)
    {
        assert(fftSize >= 256 && (fftSize & (fftSize - 1)) == 0);

        sampleRate_ = sampleRate;
        fftSize_ = fftSize;
        numBins_ = fftSize / 2 + 1;

        // Create FFT
        fft_ = std::make_unique<FFTReal<T>>(fftSize);

        // Generate window
        window_.resize(static_cast<size_t>(fftSize));
        generateWindow(windowType);

        // Compute window compensation
        windowGain_ = WindowFunctions<T>::coherentGain(window_.data(), fftSize);
        if (windowGain_ < T(0.001)) windowGain_ = T(1);

        // Allocate buffers
        inputRing_.assign(static_cast<size_t>(fftSize), T(0));
        fftBuffer_.resize(static_cast<size_t>(fftSize));
        freqBuffer_.resize(static_cast<size_t>(fftSize + 2));
        magnitudesA_.assign(static_cast<size_t>(numBins_), T(0));
        magnitudesB_.assign(static_cast<size_t>(numBins_), T(0));
        // Double-buffered output: write to one, read from the other (thread-safe)
        magnitudesDbA_.assign(static_cast<size_t>(numBins_), T(-100));
        magnitudesDbB_.assign(static_cast<size_t>(numBins_), T(-100));
        peakHoldDbA_.assign(static_cast<size_t>(numBins_), T(-100));
        peakHoldDbB_.assign(static_cast<size_t>(numBins_), T(-100));
        writeBuffer_.store(0, std::memory_order_relaxed);

        ringWritePos_ = 0;
        samplesUntilFFT_ = fftSize;
        newDataReady_.store(false, std::memory_order_relaxed);
    }

    /** @brief Resets all analysis state. */
    void reset() noexcept
    {
        std::fill(inputRing_.begin(), inputRing_.end(), T(0));
        std::fill(magnitudesA_.begin(), magnitudesA_.end(), T(0));
        std::fill(magnitudesB_.begin(), magnitudesB_.end(), T(0));
        std::fill(magnitudesDbA_.begin(), magnitudesDbA_.end(), T(-100));
        std::fill(magnitudesDbB_.begin(), magnitudesDbB_.end(), T(-100));
        std::fill(peakHoldDbA_.begin(), peakHoldDbA_.end(), T(-100));
        std::fill(peakHoldDbB_.begin(), peakHoldDbB_.end(), T(-100));
        ringWritePos_ = 0;
        samplesUntilFFT_ = fftSize_;
    }

    // -- Configuration ---------------------------------------------------------

    /** @brief Sets the smoothing factor (0 = no smoothing, 0.95 = very smooth). */
    void setSmoothing(T factor) noexcept { smoothing_ = std::clamp(factor, T(0), T(0.99)); }

    /**
     * @brief Sets the peak hold decay rate.
     * @param decayDbPerSecond How fast peaks decay in dB per second (default: 10).
     */
    void setPeakDecay(T decayDbPerSecond) noexcept { peakDecayRate_ = decayDbPerSecond; }

    /** @brief Enables or disables peak hold. */
    void setPeakHoldEnabled(bool enabled) noexcept { peakHoldEnabled_ = enabled; }

    /** @brief Sets the minimum dB value for the display (default: -100 dB). */
    void setFloorDb(T floorDb) noexcept { floorDb_ = floorDb; }

    // -- Audio thread: push samples --------------------------------------------

    /**
     * @brief Pushes audio samples into the analyser.
     *
     * Call this from the audio callback. When enough samples have been
     * accumulated (one FFT frame), the spectrum is computed automatically.
     * Thread-safe with respect to getMagnitudesDb().
     *
     * @param samples Audio sample data.
     * @param numSamples Number of samples.
     */
    void pushSamples(const T* samples, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            inputRing_[static_cast<size_t>(ringWritePos_)] = samples[i];
            ringWritePos_ = (ringWritePos_ + 1) % fftSize_;

            --samplesUntilFFT_;
            if (samplesUntilFFT_ <= 0)
            {
                computeSpectrum();
                // Overlap: next FFT starts at 50% of the frame
                samplesUntilFFT_ = fftSize_ / 2;
            }
        }
    }

    // -- GUI thread: read spectrum ---------------------------------------------

    /**
     * @brief Returns the current magnitude spectrum in decibels.
     *
     * @return Pointer to an array of getNumBins() dB values.
     *         DC is at index 0, Nyquist at index getNumBins()-1.
     */
    [[nodiscard]] const T* getMagnitudesDb() const noexcept
    {
        // Read from the buffer NOT being written to (double-buffer thread safety)
        int readBuf = 1 - writeBuffer_.load(std::memory_order_acquire);
        return (readBuf == 0) ? magnitudesDbA_.data() : magnitudesDbB_.data();
    }

    /**
     * @brief Returns the peak-hold spectrum in decibels.
     * @return Pointer to an array of getNumBins() dB values.
     */
    [[nodiscard]] const T* getPeakHoldDb() const noexcept
    {
        int readBuf = 1 - writeBuffer_.load(std::memory_order_acquire);
        return (readBuf == 0) ? peakHoldDbA_.data() : peakHoldDbB_.data();
    }

    /**
     * @brief Returns the linear (non-dB) magnitude spectrum.
     * @return Pointer to an array of getNumBins() linear magnitude values.
     */
    [[nodiscard]] const T* getMagnitudes() const noexcept
    {
        int readBuf = 1 - writeBuffer_.load(std::memory_order_acquire);
        return (readBuf == 0) ? magnitudesA_.data() : magnitudesB_.data();
    }

    /** @brief Returns true if new spectrum data is available since last check. */
    [[nodiscard]] bool isNewDataReady() noexcept
    {
        return newDataReady_.exchange(false, std::memory_order_relaxed);
    }

    // -- Utility ---------------------------------------------------------------

    /** @brief Returns the number of frequency bins. */
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }

    /** @brief Returns the FFT size. */
    [[nodiscard]] int getFFTSize() const noexcept { return fftSize_; }

    /**
     * @brief Returns the frequency in Hz for a given bin index.
     * @param bin Bin index (0 = DC, numBins-1 = Nyquist).
     */
    [[nodiscard]] T binToFrequency(int bin) const noexcept
    {
        return static_cast<T>(static_cast<double>(bin) * sampleRate_
                            / static_cast<double>(fftSize_));
    }

    /**
     * @brief Returns the bin index closest to a given frequency.
     * @param freqHz Frequency in Hz.
     */
    [[nodiscard]] int frequencyToBin(T freqHz) const noexcept
    {
        return static_cast<int>(std::round(
            static_cast<double>(freqHz) * static_cast<double>(fftSize_) / sampleRate_));
    }

    /**
     * @brief Returns the frequency resolution (Hz per bin).
     */
    [[nodiscard]] T getFrequencyResolution() const noexcept
    {
        return static_cast<T>(sampleRate_ / static_cast<double>(fftSize_));
    }

private:
    void generateWindow(WindowType type)
    {
        switch (type)
        {
            case WindowType::Hann:
                WindowFunctions<T>::hann(window_.data(), fftSize_);
                break;
            case WindowType::Hamming:
                WindowFunctions<T>::hamming(window_.data(), fftSize_);
                break;
            case WindowType::Blackman:
                WindowFunctions<T>::blackman(window_.data(), fftSize_);
                break;
            case WindowType::BlackmanHarris:
                WindowFunctions<T>::blackmanHarris(window_.data(), fftSize_);
                break;
            case WindowType::FlatTop:
                WindowFunctions<T>::flatTop(window_.data(), fftSize_);
                break;
            case WindowType::Rectangular:
                WindowFunctions<T>::rectangular(window_.data(), fftSize_);
                break;
        }
    }

    void computeSpectrum() noexcept
    {
        // Copy from ring buffer to FFT input (with windowing)
        for (int i = 0; i < fftSize_; ++i)
        {
            int ringIdx = (ringWritePos_ + i) % fftSize_;
            fftBuffer_[static_cast<size_t>(i)] =
                inputRing_[static_cast<size_t>(ringIdx)]
                * window_[static_cast<size_t>(i)];
        }

        // Forward FFT
        fft_->forward(fftBuffer_.data(), freqBuffer_.data());

        // Compute magnitudes with smoothing
        const T invGain = T(2) / (static_cast<T>(fftSize_) * windowGain_);
        const T oneMinusSmooth = T(1) - smoothing_;

        // Write to the current write buffer (double-buffered for thread safety)
        int wb = writeBuffer_.load(std::memory_order_relaxed);
        auto& magOut  = (wb == 0) ? magnitudesA_     : magnitudesB_;
        auto& dbOut   = (wb == 0) ? magnitudesDbA_   : magnitudesDbB_;
        auto& peakOut = (wb == 0) ? peakHoldDbA_     : peakHoldDbB_;

        // M9 fix: peak-hold buffers alternate writes (A one hop, B the next).
        // If we leave each buffer to evolve independently the reader sees two
        // diverging histories (flicker) and the effective decay rate is
        // halved. Seed the active buffer from the other one at the start of
        // every hop so both agree on a single, monotonically-updated state.
        if (peakHoldEnabled_)
        {
            const auto& peakIn = (wb == 0) ? peakHoldDbB_ : peakHoldDbA_;
            std::copy(peakIn.begin(), peakIn.end(), peakOut.begin());
        }

        // Precompute the per-FFT peak-hold decay once (same for every bin).
        // The previous code divided by (sampleRate * 2) which produces half
        // the requested dB/s rate. Correct expression: dBPerSecond * hopSeconds
        // with hop ≈ fftSize samples (no overlap in this analyzer).
        const T peakDecayDb = peakDecayRate_ * static_cast<T>(fftSize_)
                            / static_cast<T>(sampleRate_);

        for (int k = 0; k < numBins_; ++k)
        {
            T re = freqBuffer_[static_cast<size_t>(2 * k)];
            T im = freqBuffer_[static_cast<size_t>(2 * k + 1)];
            T mag = std::sqrt(re * re + im * im) * invGain;

            // DC and Nyquist have half the energy
            if (k == 0 || k == numBins_ - 1) mag *= T(0.5);

            // Exponential smoothing
            magOut[static_cast<size_t>(k)] =
                smoothing_ * magOut[static_cast<size_t>(k)]
                + oneMinusSmooth * mag;

            // Convert to dB
            T dB = gainToDecibels(magOut[static_cast<size_t>(k)], floorDb_);
            dbOut[static_cast<size_t>(k)] = dB;

            // Peak hold
            if (peakHoldEnabled_)
            {
                T& peak = peakOut[static_cast<size_t>(k)];
                if (dB > peak)
                    peak = dB;
                else
                {
                    peak -= peakDecayDb;
                    if (peak < floorDb_) peak = floorDb_;
                }
            }
        }

        // Swap buffers: readers now see the freshly written data
        writeBuffer_.store(1 - wb, std::memory_order_release);
        newDataReady_.store(true, std::memory_order_relaxed);
    }

    double sampleRate_ = 48000.0;
    int fftSize_ = 2048;
    int numBins_ = 1025;

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_;
    T windowGain_ = T(1);

    // Ring buffer for incoming audio
    std::vector<T> inputRing_;
    int ringWritePos_ = 0;
    int samplesUntilFFT_ = 0;

    // FFT working buffers
    std::vector<T> fftBuffer_;
    std::vector<T> freqBuffer_;

    // Output spectrum (double-buffered: A/B swap for thread-safe read/write)
    std::vector<T> magnitudesA_;        // smoothed magnitudes buffer A
    std::vector<T> magnitudesB_;        // smoothed magnitudes buffer B
    std::vector<T> magnitudesDbA_;      // dB buffer A
    std::vector<T> magnitudesDbB_;      // dB buffer B
    std::vector<T> peakHoldDbA_;        // peak hold buffer A
    std::vector<T> peakHoldDbB_;        // peak hold buffer B
    std::atomic<int> writeBuffer_ { 0 }; // which buffer (0=A, 1=B) is being written

    // Configuration
    T smoothing_ = T(0.8);
    T peakDecayRate_ = T(10);
    T floorDb_ = T(-100);
    bool peakHoldEnabled_ = false;

    std::atomic<bool> newDataReady_ { false };
};

} // namespace dspark
