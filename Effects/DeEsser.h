// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DeEsser.h
 * @brief Frequency-selective dynamic processor for sibilance reduction.
 *
 * A split-band de-esser that detects energy in a configurable frequency band
 * (typically 4–10 kHz) and applies dynamic gain reduction only in that band.
 * More transparent than wideband de-essing because non-sibilant content
 * passes through unaffected.
 *
 * Architecture:
 * 1. Bandpass sidechain (Biquad) isolates the sibilant band.
 * 2. Envelope follower (peak with attack/release) tracks sibilance level.
 * 3. When level exceeds threshold, a parametric bell cut is applied.
 *
 * Dependencies: Biquad.h, DspMath.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::DeEsser<float> deesser;
 *   deesser.prepare(spec);
 *   deesser.setFrequency(7000.0f);   // centre of sibilance band
 *   deesser.setThreshold(-20.0f);     // dB threshold
 *   deesser.setReduction(12.0f);      // max reduction in dB
 *
 *   // In audio callback:
 *   deesser.processBlock(buffer);
 *   float gr = deesser.getGainReductionDb();  // for metering
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class DeEsser
 * @brief Split-band de-esser with dynamic sibilance detection.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DeEsser
{
public:
    /** @brief Detection mode for sibilance identification. */
    enum class DetectionMode
    {
        Bandpass,    ///< Standard bandpass filter detection (default).
        Derivative   ///< Multi-derivative cascade (DeBess-style). Only sustained sibilance triggers.
    };

    /**
     * @brief Prepares the de-esser.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        // Envelope attack/release (configurable via setAttack/setRelease;
        // defaults match the previous hardcoded 0.5 ms / 20 ms values).
        updateEnvelopeCoeffs();

        // 1 ms one-pole smoother on grDb so the 16-sample coefficient refresh
        // does not introduce zipper artefacts in the bell filter.
        coefSmoothCoeff_ = static_cast<T>(
            1.0 - std::exp(-1.0 / (sampleRate_ * 0.001)));

        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            detector_[ch].reset();
            reduction_[ch].reset();
            envelope_[ch] = T(0);
            smoothedGrDb_[ch] = T(0);
            derivShift_[ch].fill(T(0));
        }

        updateCoefficients();
        gainReduction_.store(T(0), std::memory_order_relaxed);
    }

    /**
     * @brief Processes audio in-place (applies de-essing).
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(),
                             std::min(numChannels_, kMaxChannels));
        int numSamples = buffer.getNumSamples();

        // Sync atomics
        T freq    = frequency_.load(std::memory_order_relaxed);
        T bw      = bandwidth_.load(std::memory_order_relaxed);
        T thresh  = threshold_.load(std::memory_order_relaxed);
        T maxRed  = maxReduction_.load(std::memory_order_relaxed);
        auto mode = detectionMode_.load(std::memory_order_relaxed);

        // Update detector filter coefficients
        auto bpCoeffs = BiquadCoeffs<T>::makeBandPass(
            sampleRate_, static_cast<double>(freq), static_cast<double>(bw));
        for (int ch = 0; ch < numCh; ++ch)
            detector_[ch].setCoeffs(bpCoeffs);

        // Envelope coefficients are cached in attackCoeff_/releaseCoeff_.
        // If setAttack/setRelease changed them, refresh.
        if (envCoefsDirty_.exchange(false, std::memory_order_relaxed))
            updateEnvelopeCoeffs();

        T maxGr = T(0);

        // Coefficient refresh interval: the dynamic bell filter rebuilds its
        // coefficients every kCoefRefreshInterval samples instead of per sample.
        // At 48 kHz, 16 samples = 333 µs — far below perceptual thresholds
        // while the 1 ms smoother on grDb keeps the parameter path continuous.
        // Net cost reduction: ~16× fewer trig/sqrt/pow calls.
        constexpr int kCoefRefreshInterval = 16;
        const T coefSmooth = coefSmoothCoeff_;

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            T& smoothedGr = smoothedGrDb_[ch];

            for (int i = 0; i < numSamples; ++i)
            {
                T level;

                if (mode == DetectionMode::Derivative)
                {
                    // DeBess-style multi-derivative cascade:
                    // Shift register + products of adjacent first differences.
                    // Only sustained high-frequency content survives the cascade.
                    auto& sr = derivShift_[ch];
                    for (int k = kDerivLen - 1; k > 0; --k)
                        sr[k] = sr[k - 1];
                    sr[0] = detector_[ch].processSample(data[i], 0);

                    // Product of adjacent differences
                    T product = T(1);
                    for (int k = 0; k < kDerivLen - 1; ++k)
                    {
                        T diff = (sr[k] - sr[k + 1]) * (k > 0 ? (sr[k - 1] - sr[k]) : sr[k]);
                        product *= std::clamp(diff, T(-1), T(1));
                    }
                    level = std::abs(product);
                }
                else
                {
                    // Standard bandpass detection
                    T sidechain = detector_[ch].processSample(data[i], 0);
                    level = std::abs(sidechain);
                }

                // Envelope follower (peak)
                T coeff = (level > envelope_[ch]) ? attackCoeff_ : releaseCoeff_;
                envelope_[ch] += coeff * (level - envelope_[ch]);

                // Compute gain reduction
                T envDb = gainToDecibels(envelope_[ch] + T(1e-30));
                T overDb = envDb - thresh;

                T grDb = T(0);
                if (overDb > T(0))
                    grDb = -std::min(overDb, maxRed);

                // Smooth grDb with a 1 ms one-pole so the bell filter coefficient
                // path is continuous between refreshes.
                T targetGr = (grDb < T(-0.1)) ? grDb : T(0);
                smoothedGr += coefSmooth * (targetGr - smoothedGr);

                // Refresh bell filter coefficients only every N samples.
                if ((i & (kCoefRefreshInterval - 1)) == 0)
                {
                    auto peakCoeffs = BiquadCoeffs<T>::makePeak(
                        sampleRate_, static_cast<double>(freq),
                        static_cast<double>(bw),
                        static_cast<double>(smoothedGr));
                    reduction_[ch].setCoeffs(peakCoeffs);
                }

                data[i] = reduction_[ch].processSample(data[i], 0);

                if (-grDb > maxGr)
                    maxGr = -grDb;
            }
        }

        gainReduction_.store(maxGr, std::memory_order_relaxed);
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            detector_[ch].reset();
            reduction_[ch].reset();
            envelope_[ch] = T(0);
            smoothedGrDb_[ch] = T(0);
            derivShift_[ch].fill(T(0));
        }
        gainReduction_.store(T(0), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the envelope attack time in milliseconds.
     *
     * Faster attack reacts sooner to sibilance but can sound grainy;
     * slower attack lets some sibilance through but is more transparent.
     *
     * @param ms Attack time (default: 0.5 ms, range: 0.1 – 20 ms).
     */
    void setAttack(T ms) noexcept
    {
        attackMs_.store(std::clamp(ms, T(0.1), T(20)), std::memory_order_relaxed);
        envCoefsDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the envelope release time in milliseconds.
     *
     * Faster release lets non-sibilant content through sooner; slower release
     * produces smoother, more transparent reduction but can affect adjacent syllables.
     *
     * @param ms Release time (default: 20 ms, range: 1 – 500 ms).
     */
    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::clamp(ms, T(1), T(500)), std::memory_order_relaxed);
        envCoefsDirty_.store(true, std::memory_order_relaxed);
    }

    [[nodiscard]] T getAttack() const noexcept { return attackMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets the centre frequency of the sibilance band.
     * @param hz Typically 4000 – 10000 Hz (default: 7000).
     */
    void setFrequency(T hz) noexcept
    {
        frequency_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the detection bandwidth.
     * @param octaves Width in octaves (default: 1.5).
     */
    void setBandwidth(T octaves) noexcept
    {
        T bw = std::max(octaves, T(0.1));
        T q = T(1) / (T(2) * std::sinh(T(0.34657359) * bw));
        bandwidth_.store(q, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the detection threshold.
     * @param db Threshold in dB (typically -30 to -10).
     */
    void setThreshold(T db) noexcept { threshold_.store(db, std::memory_order_relaxed); }

    /**
     * @brief Sets the maximum gain reduction.
     * @param db Maximum cut in dB (positive value, e.g. 12.0).
     */
    void setReduction(T db) noexcept { maxReduction_.store(std::abs(db), std::memory_order_relaxed); }

    /**
     * @brief Sets the detection mode.
     *
     * - **Bandpass** (default): Standard bandpass-filtered envelope detection.
     * - **Derivative**: Multi-derivative cascade (DeBess/Airwindows-style).
     *   Only sustained sibilance survives the cascade of difference products,
     *   making it more selective than simple bandpass detection.
     */
    void setDetectionMode(DetectionMode mode) noexcept
    {
        detectionMode_.store(mode, std::memory_order_relaxed);
    }

    /** @brief Returns the current gain reduction in dB (positive value). */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReduction_.load(std::memory_order_relaxed); }

    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }
    [[nodiscard]] DetectionMode getDetectionMode() const noexcept { return detectionMode_.load(std::memory_order_relaxed); }

private:
    void updateCoefficients() noexcept
    {
        T freq = frequency_.load(std::memory_order_relaxed);
        T bw = bandwidth_.load(std::memory_order_relaxed);
        auto c = BiquadCoeffs<T>::makeBandPass(
            sampleRate_, static_cast<double>(freq), static_cast<double>(bw));
        for (auto& d : detector_)
            d.setCoeffs(c);
    }

    /** @brief Refreshes envelope attack/release coefficients from the atomic ms values. */
    void updateEnvelopeCoeffs() noexcept
    {
        if (sampleRate_ <= 0) return;
        const double atkSec = static_cast<double>(attackMs_.load(std::memory_order_relaxed)) * 0.001;
        const double relSec = static_cast<double>(releaseMs_.load(std::memory_order_relaxed)) * 0.001;
        attackCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * std::max(atkSec, 1e-6))));
        releaseCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * std::max(relSec, 1e-6))));
    }

    static constexpr int kMaxChannels = 2;
    static constexpr int kDerivLen = 8;  ///< Shift register length for derivative detection

    double sampleRate_ = 44100.0;
    int numChannels_ = 2;

    // Atomic parameters
    std::atomic<T> frequency_ { T(7000) };
    std::atomic<T> bandwidth_ { T(2) };       // Q value
    std::atomic<T> threshold_ { T(-20) };
    std::atomic<T> maxReduction_ { T(12) };
    std::atomic<T> attackMs_ { T(0.5) };      // Configurable attack time
    std::atomic<T> releaseMs_ { T(20) };      // Configurable release time
    std::atomic<DetectionMode> detectionMode_ { DetectionMode::Bandpass };
    std::atomic<bool> envCoefsDirty_ { false };

    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    T coefSmoothCoeff_ = T(0);   ///< 1 ms one-pole coefficient for smoothing grDb between refreshes
    std::atomic<T> gainReduction_ { T(0) };

    Biquad<T, 1> detector_[kMaxChannels]{};
    Biquad<T, 1> reduction_[kMaxChannels]{};
    T envelope_[kMaxChannels]{};
    T smoothedGrDb_[kMaxChannels]{};   ///< Smoothed grDb fed to refreshed coefficients

    // Derivative detection shift registers
    std::array<std::array<T, kDerivLen>, kMaxChannels> derivShift_ {};
};

} // namespace dspark
