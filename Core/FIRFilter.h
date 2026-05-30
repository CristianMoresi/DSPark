// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file FIRFilter.h
 * @brief FIR (Finite Impulse Response) filter with windowed-sinc coefficient design.
 *
 * FIR filters provide **linear phase** response — they do not distort the phase
 * of the signal. This makes them essential for:
 * - Mastering-grade equalisation
 * - Linear-phase crossovers
 * - Sample rate conversion (anti-aliasing)
 * - Any application where phase accuracy matters
 *
 * Trade-offs vs IIR (Biquad):
 *
 * |              | IIR (Biquad)        | FIR                     |
 * |--------------|---------------------|-------------------------|
 * | Phase        | Non-linear          | **Linear** (symmetric)  |
 * | Efficiency   | Very few coefficients| Many coefficients needed|
 * | Latency      | Minimal             | N/2 samples             |
 * | Stability    | Always stable       | Always stable           |
 *
 * For short FIR filters (<= ~256 taps), direct convolution is used via the 
 * FIRFilter class, which features a lock-free Ping-Pong buffer for safe 
 * real-time coefficient updates and strictly aligned memory for SIMD vectorization.
 * * Dependencies: WindowFunctions.h, DspMath.h, SimdOps.h, AudioBuffer.h
 */

#include "AudioBuffer.h"
#include "DspMath.h"
#include "SimdOps.h"
#include "WindowFunctions.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <numbers>
#include <span>
#include <vector>

namespace dspark {

// ============================================================================
// FIRDesign — Coefficient design via windowed-sinc method
// ============================================================================

/**
 * @class FIRDesign
 * @brief Static methods for designing FIR filter coefficients.
 *
 * Uses the windowed-sinc method: an ideal (sinc) impulse response is
 * multiplied by a window function to produce a realisable FIR filter.
 *
 * All methods return a vector of filter coefficients (taps). The number
 * of taps determines the filter's steepness and stopband attenuation.
 * More taps = steeper transition but more latency and computation.
 *
 * Rule of thumb for number of taps:
 * - `N ≈ 4 / (transitionWidth / sampleRate)` for a Kaiser window
 * - For a 1 kHz transition band at 48 kHz: N ≈ 4 / (1000/48000) ≈ 192 taps
 * - Always use an **odd** number for symmetric (Type I) FIR filters
 *
 * @tparam T Coefficient type (float or double).
 */
template <typename T>
class FIRDesign
{
public:
    /**
     * @brief Designs a low-pass FIR filter.
     *
     * @param sampleRate Sample rate in Hz.
     * @param cutoffHz   Cutoff frequency in Hz (−6 dB point).
     * @param numTaps    Number of filter taps (must be odd, ≥ 3).
     * @param beta       Kaiser window beta (default: 5.0). Higher = more attenuation.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> lowPass(double sampleRate, double cutoffHz,
                                                 int numTaps, T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        return designSinc(cutoffHz / sampleRate, numTaps, beta, false);
    }

    /**
     * @brief Designs a high-pass FIR filter.
     *
     * Created by spectrally inverting a low-pass filter.
     *
     * @param sampleRate Sample rate in Hz.
     * @param cutoffHz   Cutoff frequency in Hz.
     * @param numTaps    Number of filter taps (must be odd, ≥ 3).
     * @param beta       Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> highPass(double sampleRate, double cutoffHz,
                                                  int numTaps, T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        return designSinc(cutoffHz / sampleRate, numTaps, beta, true);
    }

    /**
     * @brief Designs a band-pass FIR filter.
     *
     * Created by subtracting a low-pass at lowCutoff from a low-pass at highCutoff.
     *
     * @param sampleRate  Sample rate in Hz.
     * @param lowCutoffHz Lower cutoff frequency in Hz.
     * @param highCutoffHz Upper cutoff frequency in Hz.
     * @param numTaps     Number of filter taps (must be odd, ≥ 3).
     * @param beta        Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> bandPass(double sampleRate, double lowCutoffHz,
                                                  double highCutoffHz, int numTaps,
                                                  T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        assert(lowCutoffHz < highCutoffHz);

        auto lp1 = designSinc(highCutoffHz / sampleRate, numTaps, beta, false);
        auto lp2 = designSinc(lowCutoffHz / sampleRate, numTaps, beta, false);

        for (int i = 0; i < numTaps; ++i)
            lp1[static_cast<size_t>(i)] -= lp2[static_cast<size_t>(i)];

        return lp1;
    }

    /**
     * @brief Designs a band-stop (notch) FIR filter.
     *
     * @param sampleRate  Sample rate in Hz.
     * @param lowCutoffHz Lower cutoff frequency in Hz.
     * @param highCutoffHz Upper cutoff frequency in Hz.
     * @param numTaps     Number of filter taps (must be odd, ≥ 3).
     * @param beta        Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> bandStop(double sampleRate, double lowCutoffHz,
                                                  double highCutoffHz, int numTaps,
                                                  T beta = T(5)) noexcept
    {
        auto bp = bandPass(sampleRate, lowCutoffHz, highCutoffHz, numTaps, beta);

        // Spectral inversion: negate all and add 1 to centre tap
        int centre = numTaps / 2;
        for (int i = 0; i < numTaps; ++i)
            bp[static_cast<size_t>(i)] = -bp[static_cast<size_t>(i)];
        bp[static_cast<size_t>(centre)] += T(1);

        return bp;
    }

    /**
     * @brief Estimates the required number of taps for a given specification.
     *
     * Uses the Kaiser formula to estimate the minimum number of odd taps needed
     * to achieve the desired stopband attenuation with a given transition bandwidth.
     *
     * @param sampleRate      Sample rate in Hz.
     * @param transitionHz    Width of the transition band in Hz.
     * @param attenuationDb   Desired stopband attenuation in dB (positive value, e.g., 60).
     * @return Estimated number of taps (always odd).
     */
    [[nodiscard]] static int estimateTaps(double sampleRate, double transitionHz,
                                           double attenuationDb) noexcept
    {
        double normTransition = transitionHz / sampleRate;
        int n = static_cast<int>(std::ceil((attenuationDb - 7.95) / (14.36 * normTransition)));
        if (n < 3) n = 3;
        if (n % 2 == 0) ++n; // Ensure odd
        return n;
    }

    /**
     * @brief Estimates the Kaiser beta parameter for a desired attenuation.
     *
     * @param attenuationDb Desired stopband attenuation in dB (positive).
     * @return Kaiser beta parameter.
     */
    [[nodiscard]] static T estimateKaiserBeta(double attenuationDb) noexcept
    {
        if (attenuationDb > 50.0)
            return static_cast<T>(0.1102 * (attenuationDb - 8.7));
        else if (attenuationDb >= 21.0)
            return static_cast<T>(0.5842 * std::pow(attenuationDb - 21.0, 0.4)
                                + 0.07886 * (attenuationDb - 21.0));
        else
            return T(0);
    }

private:
    /**
     * @brief Core windowed-sinc FIR design.
     *
     * @param normFreq Normalised cutoff frequency (cutoff / sampleRate), range [0, 0.5].
     * @param numTaps  Number of taps (odd).
     * @param beta     Kaiser window parameter.
     * @param invert   If true, spectrally invert for high-pass.
     * @return Coefficient vector.
     */
    [[nodiscard]] static std::vector<T> designSinc(double normFreq, int numTaps,
                                                    T beta, bool invert) noexcept
    {
        std::vector<T> coeffs(static_cast<size_t>(numTaps));
        std::vector<T> window(static_cast<size_t>(numTaps));

        // Generate Kaiser window
        WindowFunctions<T>::kaiser(window.data(), numTaps, beta, false);

        const int centre = numTaps / 2;
        const double fc = normFreq * 2.0; // Normalised to [0, 1] for sinc
        constexpr double kPi = std::numbers::pi;

        // Compute windowed sinc
        for (int i = 0; i < numTaps; ++i)
        {
            int n = i - centre;
            if (n == 0)
            {
                coeffs[static_cast<size_t>(i)] = static_cast<T>(fc);
            }
            else
            {
                double x = static_cast<double>(n) * kPi;
                coeffs[static_cast<size_t>(i)] = static_cast<T>(
                    std::sin(fc * x) / x);
            }

            // Apply window
            coeffs[static_cast<size_t>(i)] *= window[static_cast<size_t>(i)];
        }

        // Normalise for unity gain at DC (low-pass) or Nyquist (high-pass)
        T sum = T(0);
        for (auto c : coeffs) sum += c;
        if (std::abs(sum) > T(1e-10))
        {
            T invSum = T(1) / sum;
            for (auto& c : coeffs) c *= invSum;
        }

        // Spectral inversion for high-pass
        if (invert)
        {
            for (auto& c : coeffs) c = -c;
            coeffs[static_cast<size_t>(centre)] += T(1);
        }

        return coeffs;
    }
};

// ============================================================================
// FIRFilter — FIR filter processor (SIMD direct convolution, Thread-Safe)
// ============================================================================

/**
 * @class FIRFilter
 * @brief FIR filter using direct-form convolution with a mirrored delay line.
 *
 * Optimized for CPU cache and explicit SIMD vectorization.
 * Implements a lock-free Ping-Pong buffer for safe asynchronous coefficient 
 * updates from the UI thread without reallocating memory on the audio thread.
 * * @warning To guarantee 32-byte alignment for AVX/SIMD instructions, the internal 
 * `std::vector` instances should ideally use a custom AlignedAllocator. 
 * Currently relies on the OS default heap alignment and unaligned SIMD loads.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class alignas(32) FIRFilter
{
public:
    FIRFilter() = default;
    ~FIRFilter() = default;

    /**
     * @brief Pre-allocates memory and initializes the delay lines. 
     * @note MUST be called offline (e.g., in prepareToPlay) before any processing.
     *
     * @param maxTaps     Maximum number of filter coefficients supported.
     * @param numChannels Number of concurrent audio channels to process.
     */
    void prepare(int maxTaps, int numChannels)
    {
        assert(maxTaps > 0 && numChannels > 0);

        maxTaps_ = maxTaps;
        numChannels_ = numChannels;

        // SPSC coefficient publication: a shared staging buffer (written by the
        // control thread under a seqlock) and an audio-thread-private active
        // buffer (copied once per update, never overwritten mid-block).
        stagingCoeffs_.assign(static_cast<size_t>(maxTaps_), T(0));
        activeCoeffs_.assign(static_cast<size_t>(maxTaps_), T(0));
        stagingTaps_  = 0;
        activeTaps_   = 0;
        coeffSeq_.store(0, std::memory_order_release);
        coeffDirty_.store(false, std::memory_order_release);
        reportedTaps_.store(0, std::memory_order_release);

        // Flattened delay line buffer using the "mirror" technique.
        // Size: numChannels * (maxTaps * 2)
        delayLineBuffer_.assign(static_cast<size_t>(numChannels_ * maxTaps_ * 2), T(0));
        writePositions_.assign(static_cast<size_t>(numChannels_), 0);

        isPrepared_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the filter coefficients asynchronously.
     * * Reverses coefficients for direct SIMD dot product alignment.
     * @note Thread-Safe: Writes to the inactive buffer, then atomically swaps pointers.
     * Rapid successive calls without audio thread consumption may result in data races 
     * on the inactive buffer. A real-time crossfade is recommended for smooth morphing.
     *
     * @param coeffs Span of coefficients. Size must be <= maxTaps passed to prepare().
     * @note Thread-safe single-producer publish via a seqlock. The audio thread
     *       copies the staged set into its private buffer atomically, so neither
     *       a (ptr, count) mismatch nor a mid-block overwrite can occur.
     */
    void setCoefficients(std::span<const T> coeffs) noexcept
    {
        if (!isPrepared_.load(std::memory_order_acquire)) return;

        const int numTaps = static_cast<int>(coeffs.size());
        if (numTaps == 0 || numTaps > maxTaps_) return;

        // Seqlock publish: odd sequence = write in progress.
        coeffSeq_.fetch_add(1, std::memory_order_acq_rel);
        // Write REVERSED coefficients to the staging buffer for SIMD dot product.
        for (int k = 0; k < numTaps; ++k)
            stagingCoeffs_[static_cast<size_t>(k)] = coeffs[static_cast<size_t>(numTaps - 1 - k)];
        stagingTaps_ = numTaps;
        coeffSeq_.fetch_add(1, std::memory_order_release); // even = consistent
        coeffDirty_.store(true, std::memory_order_release);
        reportedTaps_.store(numTaps, std::memory_order_release);
    }

    /** * @brief Resets all delay lines to zero, clearing the filter's memory. 
     */
    void reset() noexcept
    {
        std::fill(delayLineBuffer_.begin(), delayLineBuffer_.end(), T(0));
        std::fill(writePositions_.begin(), writePositions_.end(), 0);
    }

    /**
     * @brief Processes a full audio buffer in-place.
     * * Standardized to match the rest of the DSPark framework. Loads atomic 
     * state once per block to avoid intra-block tearing and pipeline stalls.
     *
     * @param buffer Audio buffer view to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!isPrepared_.load(std::memory_order_acquire)) return;

        // Pick up any pending coefficient update once per block (seqlock copy
        // into the audio-thread-private active buffer).
        pullCoeffsIfDirty();

        const int currentTaps = activeTaps_;
        if (currentTaps == 0) return; // Bypass if uninitialized

        const T* currentCoeffs = activeCoeffs_.data();
        const int numChannels = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples  = buffer.getNumSamples();

        for (int ch = 0; ch < numChannels; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& wp = writePositions_[static_cast<size_t>(ch)];
            const int channelOffset = ch * (maxTaps_ * 2);
            T* dl = delayLineBuffer_.data() + channelOffset;

            for (int i = 0; i < numSamples; ++i)
            {
                // Write into mirror buffer based on fixed maxTaps_ bound
                dl[wp] = data[i];
                dl[wp + maxTaps_] = data[i];

                // Calculate oldest sample position for the contiguous read
                const T* readPtr = dl + wp + maxTaps_ - currentTaps + 1;

                // Execute SIMD convolution (unaligned load assumed for readPtr)
                data[i] = simd::dotProduct(currentCoeffs, readPtr, currentTaps);

                // Advance write position and wrap fixed bound
                if (++wp >= maxTaps_) wp = 0;
            }
        }
    }

    /**
     * @brief Processes a single sample through the FIR filter.
     * * @warning Use `processBlock` instead when possible to avoid atomic load 
     * overhead on a per-sample basis.
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Filtered output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (!isPrepared_.load(std::memory_order_acquire)) return input;

        // Cheap relaxed check; the seqlock copy only runs when an update landed.
        if (coeffDirty_.load(std::memory_order_relaxed)) [[unlikely]]
            pullCoeffsIfDirty();

        const int currentTaps = activeTaps_;
        if (currentTaps == 0) return input;

        const T* currentCoeffs = activeCoeffs_.data();

        auto& wp = writePositions_[static_cast<size_t>(channel)];
        const int channelOffset = channel * (maxTaps_ * 2);
        T* dl = delayLineBuffer_.data() + channelOffset;

        dl[wp] = input;
        dl[wp + maxTaps_] = input;

        const T* readPtr = dl + wp + maxTaps_ - currentTaps + 1;
        const T output = simd::dotProduct(currentCoeffs, readPtr, currentTaps);

        if (++wp >= maxTaps_) wp = 0;

        return output;
    }

    /** * @brief Returns the filter's group delay (latency) in samples. 
     * @return Latency in samples based on currently active taps.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        const int taps = reportedTaps_.load(std::memory_order_relaxed);
        return taps > 0 ? (taps - 1) / 2 : 0;
    }

private:
    /** @brief Audio-thread: copy the staged coefficient set into the private
     *  active buffer if an update was published (seqlock read with retry). */
    void pullCoeffsIfDirty() noexcept
    {
        if (!coeffDirty_.exchange(false, std::memory_order_acquire)) return;
        unsigned s0, s1;
        int n;
        do {
            s0 = coeffSeq_.load(std::memory_order_acquire);
            n  = stagingTaps_;
            for (int k = 0; k < n; ++k)
                activeCoeffs_[static_cast<size_t>(k)] = stagingCoeffs_[static_cast<size_t>(k)];
            s1 = coeffSeq_.load(std::memory_order_acquire);
        } while ((s0 & 1u) != 0u || s0 != s1);
        activeTaps_ = n;
    }

    int maxTaps_{0};
    int numChannels_{0};
    std::atomic<bool> isPrepared_{false};

    // SPSC seqlock coefficient publication (writer: setCoefficients).
    std::vector<T> stagingCoeffs_;          ///< Shared staging (reversed coeffs).
    int stagingTaps_{0};                    ///< Guarded by coeffSeq_.
    std::atomic<unsigned> coeffSeq_{0};     ///< Seqlock counter (odd = writing).
    std::atomic<bool> coeffDirty_{false};   ///< Pending-update flag (fast path).
    std::atomic<int> reportedTaps_{0};      ///< Latest set tap count for getLatency().

    // Audio-thread-private active coefficient set (never overwritten mid-block).
    std::vector<T> activeCoeffs_;
    int activeTaps_{0};

    // Flattened, cache-contiguous delay lines
    std::vector<T> delayLineBuffer_;
    std::vector<int> writePositions_;
};

} // namespace dspark