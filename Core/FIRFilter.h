// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file FIRFilter.h
 * @brief FIR (Finite Impulse Response) filter with windowed-sinc coefficient design.
 *
 * FIR filters provide **linear phase** response -- they do not distort the phase
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
 * FIRFilter class, which features a lock-free ping-pong coefficient publish
 * for safe real-time updates and SIMD (simd::dotProduct) convolution.
 *
 * Coefficient design (FIRDesign) always runs its maths in double precision
 * and casts to T on return: design is setup-time work, so float builds get
 * full float accuracy for free.
 *
 * Dependencies: WindowFunctions.h, SimdOps.h, AudioBuffer.h.
 */

#include "AudioBuffer.h"
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
// FIRDesign -- Coefficient design via windowed-sinc method
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
 * - `N ~ 4 / (transitionWidth / sampleRate)` for a Kaiser window
 * - For a 1 kHz transition band at 48 kHz: N ~ 4 / (1000/48000) ~ 192 taps
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
     * @param cutoffHz   Cutoff frequency in Hz (-6 dB point).
     * @param numTaps    Number of filter taps (must be odd, >= 3).
     * @param beta       Kaiser window beta (default: 5.0). Higher = more attenuation.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> lowPass(double sampleRate, double cutoffHz,
                                                 int numTaps, T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        numTaps = sanitizeTaps(numTaps);
        return designSinc(cutoffHz / sampleRate, numTaps, beta, false);
    }

    /**
     * @brief Designs a high-pass FIR filter.
     *
     * Created by spectrally inverting a low-pass filter.
     *
     * @param sampleRate Sample rate in Hz.
     * @param cutoffHz   Cutoff frequency in Hz.
     * @param numTaps    Number of filter taps (must be odd, >= 3).
     * @param beta       Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> highPass(double sampleRate, double cutoffHz,
                                                  int numTaps, T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        numTaps = sanitizeTaps(numTaps);
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
     * @param numTaps     Number of filter taps (must be odd, >= 3).
     * @param beta        Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> bandPass(double sampleRate, double lowCutoffHz,
                                                  double highCutoffHz, int numTaps,
                                                  T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        assert(lowCutoffHz < highCutoffHz);
        numTaps = sanitizeTaps(numTaps);

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
     * @param numTaps     Number of filter taps (must be odd, >= 3).
     * @param beta        Kaiser window beta.
     * @return Vector of filter coefficients.
     */
    [[nodiscard]] static std::vector<T> bandStop(double sampleRate, double lowCutoffHz,
                                                  double highCutoffHz, int numTaps,
                                                  T beta = T(5)) noexcept
    {
        assert(numTaps >= 3 && (numTaps % 2 == 1));
        // Sanitise HERE too: bandPass() sanitises its own copy, so an even
        // request would return numTaps+1 coefficients while the loops below
        // still ran over the caller's numTaps -- leaving the last tap
        // un-negated (a broken, asymmetric filter) in release builds.
        numTaps = sanitizeTaps(numTaps);

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
     * @param transitionHz    Width of the transition band in Hz. Must be > 0.
     * @param attenuationDb   Desired stopband attenuation in dB (positive value, e.g., 60).
     * @return Estimated number of taps (always odd).
     */
    [[nodiscard]] static int estimateTaps(double sampleRate, double transitionHz,
                                           double attenuationDb) noexcept
    {
        assert(sampleRate > 0.0 && transitionHz > 0.0);
        // Release-safe floor: a zero/negative transition width would push the
        // estimate through ceil(inf) into undefined integer conversion.
        double normTransition = std::max(transitionHz / sampleRate, 1e-6);
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
    /** @brief Release-safe tap sanitiser: spectral inversion (high-pass /
     *  band-stop) silently breaks for even tap counts (Type II FIR has a
     *  forced zero at Nyquist), so force the next odd count >= 3. */
    [[nodiscard]] static int sanitizeTaps(int numTaps) noexcept
    {
        if (numTaps < 3) numTaps = 3;
        return numTaps | 1;
    }

    /**
     * @brief Core windowed-sinc FIR design. All maths in double; cast on store.
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
        std::vector<double> design(static_cast<size_t>(numTaps));
        std::vector<double> window(static_cast<size_t>(numTaps));

        // Generate Kaiser window (double engine regardless of T)
        WindowFunctions<double>::kaiser(window.data(), numTaps,
                                        static_cast<double>(beta), false);

        const int centre = numTaps / 2;
        const double fc = normFreq * 2.0; // Normalised to [0, 1] for sinc
        constexpr double kPi = std::numbers::pi;

        // Compute windowed sinc and its DC sum in one pass
        double sum = 0.0;
        for (int i = 0; i < numTaps; ++i)
        {
            const int n = i - centre;
            const double x = static_cast<double>(n) * kPi;
            const double s = (n == 0) ? fc : std::sin(fc * x) / x;
            const double c = s * window[static_cast<size_t>(i)];
            design[static_cast<size_t>(i)] = c;
            sum += c;
        }

        // Normalise for unity gain at DC (low-pass) or Nyquist (high-pass)
        if (std::abs(sum) > 1e-10)
        {
            const double invSum = 1.0 / sum;
            for (auto& c : design) c *= invSum;
        }

        // Spectral inversion for high-pass
        if (invert)
        {
            for (auto& c : design) c = -c;
            design[static_cast<size_t>(centre)] += 1.0;
        }

        std::vector<T> coeffs(static_cast<size_t>(numTaps));
        for (int i = 0; i < numTaps; ++i)
            coeffs[static_cast<size_t>(i)] = static_cast<T>(design[static_cast<size_t>(i)]);
        return coeffs;
    }
};

// ============================================================================
// FIRFilter -- FIR filter processor (SIMD direct convolution, thread-safe)
// ============================================================================

/**
 * @class FIRFilter
 * @brief FIR filter using direct-form convolution with a mirrored delay line.
 *
 * Implements a lock-free coefficient publish (seqlock) for safe asynchronous
 * updates from the control thread without reallocating memory on the audio
 * thread; the convolution itself runs through simd::dotProduct.
 *
 * @note Buffers use the default heap alignment; the SIMD kernels use
 *       unaligned loads, which cost the same as aligned on modern CPUs.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class FIRFilter
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
        stagingTaps_.store(0, std::memory_order_relaxed);
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
     *
     * Coefficients are stored reversed for direct SIMD dot-product alignment.
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
        stagingTaps_.store(numTaps, std::memory_order_relaxed);
        coeffSeq_.fetch_add(1, std::memory_order_release); // even = consistent
        coeffDirty_.store(true, std::memory_order_release);
        reportedTaps_.store(numTaps, std::memory_order_release);
    }

    /**
     * @brief Resets all delay lines to zero, clearing the filter's memory.
     */
    void reset() noexcept
    {
        std::fill(delayLineBuffer_.begin(), delayLineBuffer_.end(), T(0));
        std::fill(writePositions_.begin(), writePositions_.end(), 0);
    }

    /**
     * @brief Processes a full audio buffer in-place.
     *
     * Loads atomic state once per block to avoid intra-block tearing and
     * pipeline stalls.
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
     *
     * @warning Use `processBlock` instead when possible to avoid atomic load
     * overhead on a per-sample basis.
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Filtered output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (!isPrepared_.load(std::memory_order_acquire)) return input;

        // Release-safe channel bound (processBlock clamps; this entry point
        // must too, or an out-of-range channel indexes the flat delay line OOB).
        assert(channel >= 0 && channel < numChannels_);
        if (channel < 0 || channel >= numChannels_) return input;

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

    /**
     * @brief Returns the filter's group delay (latency) in samples.
     * @return Latency based on the most recently PUBLISHED tap count (the
     *         audio thread may adopt it one block later).
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
            // Defensive clamp: the loop bound must never exceed the buffers
            // even if this speculative read races a concurrent publish.
            n = std::min(stagingTaps_.load(std::memory_order_relaxed), maxTaps_);
            for (int k = 0; k < n; ++k)
                activeCoeffs_[static_cast<size_t>(k)] = stagingCoeffs_[static_cast<size_t>(k)];
            // The fence orders the copy above BEFORE the re-read below. A plain
            // load-acquire is not enough: acquire only orders LATER accesses,
            // so on weakly-ordered CPUs (ARM) the copy could sink below the
            // second read and a torn copy would pass the s0 == s1 check.
            std::atomic_thread_fence(std::memory_order_acquire);
            s1 = coeffSeq_.load(std::memory_order_relaxed);
        } while ((s0 & 1u) != 0u || s0 != s1);
        activeTaps_ = n;
    }

    int maxTaps_{0};
    int numChannels_{0};
    std::atomic<bool> isPrepared_{false};

    // SPSC seqlock coefficient publication (writer: setCoefficients).
    std::vector<T> stagingCoeffs_;          ///< Shared staging (reversed coeffs).
    std::atomic<int> stagingTaps_{0};       ///< Guarded by coeffSeq_ (relaxed slot).
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
