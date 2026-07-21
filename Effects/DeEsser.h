// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DeEsser.h
 * @brief Frequency-selective dynamic processor for sibilance reduction.
 *
 * A dynamic-EQ style de-esser: a bandpass sidechain detects energy in a
 * configurable frequency band and a dynamic peak-cut bell applies the gain
 * reduction at that band. Optimized for zero inner-loop allocations and
 * trigonometric eliminations during dynamic EQ modulation.
 *
 * Architecture:
 * 1. Bandpass sidechain (Biquad) isolates the sibilant band.
 * 2. Envelope follower (peak with attack/release) tracks maximum stereo sibilance.
 * 3. Fast-math Biquad Peak cut applies gain reduction with precomputed trig terms.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Biquad.h, DspMath.h, AudioSpec.h, AudioBuffer.h, StateBlob.h.
 *
 * @code
 *   dspark::DeEsser<float> deesser;
 *   deesser.prepare(spec);
 *   deesser.setFrequency(7000.0f);
 *   deesser.setThreshold(-20.0f);
 *   deesser.setReduction(12.0f);
 *   deesser.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class DeEsser
 * @brief Stereo-linked, CPU-optimized dynamic-EQ de-esser.
 *
 * Detection is stereo-linked (one shared envelope) to preserve imaging.
 * Channels beyond the first two are left untouched (pass-through), as is the
 * whole buffer before prepare().
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DeEsser
{
public:
    enum class DetectionMode
    {
        Bandpass,    ///< Standard bandpass filter detection.
        Derivative   ///< Sum of absolute derivatives. Sensitive to sustained high frequencies.
    };

    /**
     * @brief Prepares the de-esser state and filters.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        updateEnvelopeCoeffs();
        coefSmoothCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * 0.001)));

        reset();

        filterParamsDirty_.store(true, std::memory_order_release);
        updateFiltersIfDirty();
    }

    /**
     * @brief Processes audio in-place. Stereo-linked to preserve imaging.
     * @param buffer Audio data.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        int numCh = std::min(buffer.getNumChannels(), std::min(numChannels_, kMaxChannels));
        int numSamples = buffer.getNumSamples();
        if (numSamples == 0 || numCh == 0) return;

        updateFiltersIfDirty();

        if (envCoefsDirty_.exchange(false, std::memory_order_acquire))
            updateEnvelopeCoeffs();

        T thresh = threshold_.load(std::memory_order_relaxed);
        T maxRed = maxReduction_.load(std::memory_order_relaxed);
        auto mode = detectionMode_.load(std::memory_order_relaxed);

        constexpr int kCoefRefreshInterval = 16;
        const T coefSmooth = coefSmoothCoeff_;
        T blockMaxGr = T(0);

        // Branch hoisting: Process samples depending on the mode to avoid inner-loop conditionals.
        if (mode == DetectionMode::Derivative)
        {
            processInternal<true>(buffer, numCh, numSamples, thresh, maxRed, coefSmooth, kCoefRefreshInterval, blockMaxGr);
        }
        else
        {
            processInternal<false>(buffer, numCh, numSamples, thresh, maxRed, coefSmooth, kCoefRefreshInterval, blockMaxGr);
        }

        gainReduction_.store(blockMaxGr, std::memory_order_relaxed);
    }

    /** @brief Resets internal DSP state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            detector_[ch].reset();
            reduction_[ch].reset();
            smoothedGrDb_[ch] = T(0);
            derivShift_[ch].fill(T(0));
        }
        envelope_ = T(0);
        gainReduction_.store(T(0), std::memory_order_relaxed);
    }

    // Setters

    /** @brief Sets the detector attack, clamped to [0.1, 20] ms. Non-finite values are ignored. */
    void setAttack(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        attackMs_.store(std::clamp(ms, T(0.1), T(20)), std::memory_order_relaxed);
        envCoefsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the detector release, clamped to [1, 500] ms. Non-finite values are ignored. */
    void setRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::clamp(ms, T(1), T(500)), std::memory_order_relaxed);
        envCoefsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the sibilance centre frequency in Hz.
     *
     * Clamped against the prepared sample rate when the filters rebuild
     * (effective range [1, 0.499 * fs]). Non-finite values are ignored.
     */
    void setFrequency(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        frequency_.store(hz, std::memory_order_relaxed);
        filterParamsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the detection/cut bandwidth in octaves (floored at 0.1). Non-finite values are ignored. */
    void setBandwidth(T octaves) noexcept
    {
        if (!std::isfinite(octaves)) return;
        T bw = std::max(octaves, T(0.1));
        T q = T(1) / (T(2) * std::sinh(T(0.34657359) * bw));
        bandwidth_.store(q, std::memory_order_relaxed);
        filterParamsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the detection threshold in dBFS. Non-finite values are ignored. */
    void setThreshold(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        threshold_.store(db, std::memory_order_relaxed);
    }

    /** @brief Sets the maximum gain reduction in dB (sign is ignored). Non-finite values are ignored. */
    void setReduction(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        maxReduction_.store(std::abs(db), std::memory_order_relaxed);
    }

    /** @brief Sets the detection mode. Out-of-range values are clamped so the getter stays honest. */
    void setDetectionMode(DetectionMode mode) noexcept
    {
        const int v = std::clamp(static_cast<int>(mode),
                                 static_cast<int>(DetectionMode::Bandpass),
                                 static_cast<int>(DetectionMode::Derivative));
        detectionMode_.store(static_cast<DetectionMode>(v), std::memory_order_relaxed);
    }

    // Getters
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReduction_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getFrequency() const noexcept { return frequency_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }
    [[nodiscard]] DetectionMode getDetectionMode() const noexcept { return detectionMode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getAttack() const noexcept { return attackMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }

    /** @return The detection/cut bandwidth in octaves (inverse of the stored Q). */
    [[nodiscard]] T getBandwidth() const noexcept
    {
        return static_cast<T>(std::asinh(1.0 / (2.0 * static_cast<double>(
            bandwidth_.load(std::memory_order_relaxed)))) / 0.34657359);
    }

    /** @return The maximum gain reduction in dB (positive). */
    [[nodiscard]] T getReduction() const noexcept { return maxReduction_.load(std::memory_order_relaxed); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float (setState reads float back); the explicit
        // casts also keep this overload resolvable when T is double.
        StateWriter w(stateId("DEES"), 1);
        w.write("frequency", static_cast<float>(frequency_.load(std::memory_order_relaxed)));
        // bandwidth_ stores Q = 1/(2 sinh(ln2/2 * oct)); serialize back in octaves.
        w.write("bandwidth", static_cast<float>(getBandwidth()));
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        w.write("reduction", static_cast<float>(maxReduction_.load(std::memory_order_relaxed)));
        w.write("attack", static_cast<float>(attackMs_.load(std::memory_order_relaxed)));
        w.write("release", static_cast<float>(releaseMs_.load(std::memory_order_relaxed)));
        w.write("detection", static_cast<int32_t>(detectionMode_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DEES")) return false;
        setFrequency(static_cast<T>(r.read("frequency", 7000.0f)));
        setBandwidth(static_cast<T>(r.read("bandwidth", 2.0f)));
        setThreshold(static_cast<T>(r.read("threshold", -20.0f)));
        setReduction(static_cast<T>(r.read("reduction", 12.0f)));
        setAttack(static_cast<T>(r.read("attack", 0.5f)));
        setRelease(static_cast<T>(r.read("release", 20.0f)));
        setDetectionMode(static_cast<DetectionMode>(r.read("detection", 0)));
        return true;
    }

private:
    template <bool IsDerivative>
    void processInternal(AudioBufferView<T>& buffer, int numCh, int numSamples,
                         T thresh, T maxRed, T coefSmooth, int refreshInterval, T& blockMaxGr) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
        {
            T maxLevel = T(0);

            // 1. Stereo-linked detection
            for (int ch = 0; ch < numCh; ++ch)
            {
                T dataIn = buffer.getChannel(ch)[i];
                T level = T(0);

                if constexpr (IsDerivative)
                {
                    auto& sr = derivShift_[ch];
                    for (int k = kDerivLen - 1; k > 0; --k) sr[k] = sr[k - 1];
                    sr[0] = detector_[ch].processSample(dataIn, 0);

                    T sumDiffs = T(0);
                    for (int k = 0; k < kDerivLen - 1; ++k)
                    {
                        sumDiffs += std::abs(sr[k] - sr[k + 1]);
                    }
                    level = sumDiffs / T(kDerivLen - 1);
                }
                else
                {
                    level = std::abs(detector_[ch].processSample(dataIn, 0));
                }

                if (level > maxLevel) maxLevel = level;
            }

            // 2. Single linked envelope
            T coeff = (maxLevel > envelope_) ? attackCoeff_ : releaseCoeff_;
            envelope_ += coeff * (maxLevel - envelope_);

            // 3. Compute GR
            T envDb = gainToDecibels(envelope_ + T(1e-30));
            T overDb = envDb - thresh;
            T grDb = (overDb > T(0)) ? -std::min(overDb, maxRed) : T(0);

            if (-grDb > blockMaxGr) blockMaxGr = -grDb;

            // 4. Apply processing per channel
            for (int ch = 0; ch < numCh; ++ch)
            {
                T& smoothedGr = smoothedGrDb_[ch];
                smoothedGr += coefSmooth * (grDb - smoothedGr);

                // Refresh coefficients without trig functions
                if ((i & (refreshInterval - 1)) == 0)
                {
                    updateDynamicPeakCoeffs(ch, smoothedGr);
                }

                T* channelData = buffer.getChannel(ch);
                channelData[i] = reduction_[ch].processSample(channelData[i], 0);
            }
        }
    }

    /**
     * @brief Updates Biquad filters if frequency or bandwidth changed.
     * Precomputes trigonometric constants for the inner-loop Peak filter.
     */
    void updateFiltersIfDirty() noexcept
    {
        if (!filterParamsDirty_.exchange(false, std::memory_order_acquire))
            return;

        T freq = frequency_.load(std::memory_order_relaxed);
        T bw = bandwidth_.load(std::memory_order_relaxed);

        // One shared effective frequency for BOTH filters. The bandpass
        // factory clamps internally, but the precomputed peak terms did not:
        // beyond Nyquist sin(w0) goes negative, alpha flips sign and the
        // dynamic bell turns unstable under gain reduction.
        const double fEff = std::clamp(static_cast<double>(freq), 1.0,
                                       std::max(1.0, sampleRate_ * 0.499));

        auto c = BiquadCoeffs<T>::makeBandPass(sampleRate_, fEff, static_cast<double>(bw));
        for (int ch = 0; ch < kMaxChannels; ++ch)
            detector_[ch].setCoeffs(c);

        // Precompute trig terms for the Peak filter to avoid math in the inner loop
        double w0 = twoPi<double> * fEff / sampleRate_;
        precomputedCos_ = static_cast<T>(std::cos(w0));
        precomputedAlpha_ = static_cast<T>(std::sin(w0) / (2.0 * static_cast<double>(bw)));
    }

    /**
     * @brief Calculates fast Peak EQ coefficients dynamically using precomputed trig.
     * @param ch Channel index.
     * @param gainDb Gain to apply (negative).
     */
    void updateDynamicPeakCoeffs(int ch, T gainDb) noexcept
    {
        // A = 10^(gainDb / 40)
        T A = std::pow(T(10), gainDb / T(40));

        T b0 = T(1) + precomputedAlpha_ * A;
        T b1 = T(-2) * precomputedCos_;
        T b2 = T(1) - precomputedAlpha_ * A;
        T a0 = T(1) + precomputedAlpha_ / A;
        T a1 = T(-2) * precomputedCos_;
        T a2 = T(1) - precomputedAlpha_ / A;

        // Normalization
        T a0Inv = T(1) / a0;

        BiquadCoeffs<T> coeffs;
        coeffs.b0 = b0 * a0Inv;
        coeffs.b1 = b1 * a0Inv;
        coeffs.b2 = b2 * a0Inv;
        coeffs.a1 = a1 * a0Inv;
        coeffs.a2 = a2 * a0Inv;

        reduction_[ch].setCoeffs(coeffs);
    }

    void updateEnvelopeCoeffs() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        const double atkSec = static_cast<double>(attackMs_.load(std::memory_order_relaxed)) * 0.001;
        const double relSec = static_cast<double>(releaseMs_.load(std::memory_order_relaxed)) * 0.001;
        attackCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * std::max(atkSec, 1e-6))));
        releaseCoeff_ = static_cast<T>(1.0 - std::exp(-1.0 / (sampleRate_ * std::max(relSec, 1e-6))));
    }

    static constexpr int kMaxChannels = 2;
    static constexpr int kDerivLen = 8;

    double sampleRate_ = 44100.0;
    int numChannels_ = 0;

    std::atomic<T> frequency_ { T(7000) };
    std::atomic<T> bandwidth_ { T(2) };
    std::atomic<T> threshold_ { T(-20) };
    std::atomic<T> maxReduction_ { T(12) };
    std::atomic<T> attackMs_ { T(0.5) };
    std::atomic<T> releaseMs_ { T(20) };
    std::atomic<DetectionMode> detectionMode_ { DetectionMode::Bandpass };

    std::atomic<bool> envCoefsDirty_ { false };
    std::atomic<bool> filterParamsDirty_ { true };

    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    T coefSmoothCoeff_ = T(0);
    std::atomic<T> gainReduction_ { T(0) };

    Biquad<T, 1> detector_[kMaxChannels]{};
    Biquad<T, 1> reduction_[kMaxChannels]{};

    T envelope_{T(0)}; // Stereo-linked envelope
    T smoothedGrDb_[kMaxChannels]{};

    std::array<std::array<T, kDerivLen>, kMaxChannels> derivShift_ {};

    // Precomputed components for fast Biquad Peak modulation
    T precomputedCos_ = T(0);
    T precomputedAlpha_ = T(0);
};

} // namespace dspark
