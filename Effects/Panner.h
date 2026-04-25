// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Panner.h
 * @brief Stereo panning toolkit with multiple algorithms.
 *
 * Ported from the existing JUCE-dependent Panner. Provides six panning methods,
 * each with a distinct perceptual character:
 *
 * - **Equal Power**: Standard -3 dB constant-power pan (no center dip).
 * - **Binaural**: Combined cross-feeding + ITD delay for headphone spatialisation.
 * - **Mid Pan**: Pans only the centre (mid) image, preserving stereo width.
 * - **Side Pan**: Pans only the stereo (side) image toward one side.
 * - **Haas**: Precedence effect via inter-channel delay.
 * - **Spectral**: Frequency-dependent panning via high-shelf filter.
 *
 * Dependencies: Delay.h, Biquad.h, AudioBuffer.h, AudioSpec.h, DspMath.h, Smoothers.h.
 *
 * @code
 *   dspark::Panner<float> panner;
 *   panner.prepare(spec);
 *   panner.applyEqualPower(buffer.toView(), 0.5f);  // pan right
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/Smoothers.h"
#include "Delay.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>

namespace dspark {

template <typename T = float>
class Panner
{
public:
    virtual ~Panner() = default;
    /** @brief Panning algorithm selection. */
    enum class Algorithm
    {
        EqualPower,  ///< Standard -3 dB constant-power pan.
        Binaural,    ///< Combined cross-feeding + ITD delay.
        MidPan,      ///< Pans only the centre (mid) image.
        SidePan,     ///< Pans only the stereo (side) image.
        Haas,        ///< Precedence effect via inter-channel delay.
        Spectral     ///< Frequency-dependent panning via high-shelf.
    };

    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        panSmoother_.reset(sampleRate_, smoothingTime_.load(std::memory_order_relaxed));

        // Prepare delays for binaural and Haas
        float maxMs = std::max(binauralMaxITD_.load(std::memory_order_relaxed),
                               haasMaxDelay_.load(std::memory_order_relaxed));
        AudioSpec monoSpec { sampleRate_, spec.maxBlockSize, 1 };
        delayL_.prepareMs(monoSpec, static_cast<double>(maxMs));
        delayR_.prepareMs(monoSpec, static_cast<double>(maxMs));
        delayL_.setSmoother(Delay<T>::SmootherType::CriticallyDamped);
        delayR_.setSmoother(Delay<T>::SmootherType::CriticallyDamped);
        delayL_.setSmoothingTime(smoothingTime_);
        delayR_.setSmoothingTime(smoothingTime_);

        // Spectral filters (high shelf, one per channel)
        updateSpectralFilters(T(0)); // neutral
    }

    // -- Unified API (satisfies AudioProcessor concept) ----------------------

    /**
     * @brief Sets the panning algorithm.
     * @param algo Algorithm to use.
     */
    void setAlgorithm(Algorithm algo) noexcept { algorithm_.store(algo, std::memory_order_relaxed); }

    /**
     * @brief Sets the pan position.
     * @param position -1.0 (left) to +1.0 (right). 0 = center.
     */
    void setPan(T position) noexcept
    {
        T p = std::clamp(position, T(-1), T(1));
        pan_.store(p, std::memory_order_relaxed);
        panSmoother_.setTargetValue(static_cast<float>(p));
    }

    /**
     * @brief Processes the buffer using the current algorithm and pan position.
     *
     * This is the unified API that satisfies the AudioProcessor concept.
     * Dispatches to the appropriate apply*() method based on algorithm_.
     *
     * @param buffer Stereo audio buffer to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumChannels() < 2) return;
        float p = static_cast<float>(pan_.load(std::memory_order_relaxed));

        switch (algorithm_.load(std::memory_order_relaxed))
        {
            case Algorithm::EqualPower: applyEqualPower(buffer, p); break;
            case Algorithm::Binaural:   applyCombinedBinaural(buffer, p); break;
            case Algorithm::MidPan:     applyMidPan(buffer, p); break;
            case Algorithm::SidePan:    applySidePan(buffer, p); break;
            case Algorithm::Haas:       applyHaas(buffer, p); break;
            case Algorithm::Spectral:   applySpectral(buffer, p); break;
        }
    }

    /**
     * @brief Resets internal state (delays, filters, smoothers).
     */
    void reset() noexcept
    {
        delayL_.reset();
        delayR_.reset();
        spectralL_.reset();
        spectralR_.reset();
        panSmoother_.skip();
    }

    // -- Configuration -------------------------------------------------------

    void setBinauralMaxITD(float ms)    { binauralMaxITD_.store(ms, std::memory_order_relaxed); }
    void setHaasMaxDelay(float ms)      { haasMaxDelay_.store(ms, std::memory_order_relaxed); }
    void setSpectralFrequency(float hz) { spectralFreq_.store(std::clamp(hz, 20.0f, 20000.0f), std::memory_order_relaxed); }
    void setSpectralMaxGain(float dB)   { spectralMaxGain_.store(dB, std::memory_order_relaxed); }
    void setSmoothingTime(float ms)     { smoothingTime_.store(ms, std::memory_order_relaxed); }

    // -- Panning algorithms --------------------------------------------------

    /**
     * @brief Standard -3 dB constant-power pan.
     * @param buffer Stereo buffer.
     * @param pan    -1.0 (left) to +1.0 (right).
     */
    void applyEqualPower(AudioBufferView<T> buffer, float /*pan*/)
    {
        assert(buffer.getNumChannels() >= 2);

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();
        constexpr T halfPi = pi<T> / T(2);

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());
            T angle = (p * T(0.5) + T(0.5)) * halfPi;
            L[i] *= std::cos(angle);
            R[i] *= std::sin(angle);
        }
    }

    /**
     * @brief Combined binaural pan: cross-feeding + ITD delay.
     * @param buffer Stereo buffer.
     * @param pan    -1.0 (left) to +1.0 (right).
     */
    void applyCombinedBinaural(AudioBufferView<T> buffer, float pan)
    {
        assert(buffer.getNumChannels() >= 2);

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        // A12 optimisation: set the ITD target once per block, not once per
        // sample. The Delay has its own critically-damped smoother that
        // handles the per-sample interpolation — re-targeting it on every
        // sample was both redundant and slower than letting the smoother run.
        // Cross-feeding still reads the sample-rate smoothed pan so the
        // amplitude law remains click-free.
        T itdMax = T(binauralMaxITD_.load(std::memory_order_relaxed));
        T panTarget = static_cast<T>(pan);
        T delayLms = itdMax * std::max(T(0), panTarget);
        T delayRms = itdMax * std::max(T(0), -panTarget);
        delayL_.setDelayMs(delayLms);
        delayR_.setDelayMs(delayRms);

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());

            // Cross-feeding based on pan position (uses sample-rate smoothed p)
            if (p < T(0))
            {
                L[i] += R[i] * (-p);
                R[i] *= (T(1) + p);
            }
            else if (p > T(0))
            {
                R[i] += L[i] * p;
                L[i] *= (T(1) - p);
            }

            L[i] = delayL_.processSample(0, L[i]);  // ch=0 on mono delay
            delayL_.advanceWriteIndex();
            R[i] = delayR_.processSample(0, R[i]);  // ch=0 on mono delay
            delayR_.advanceWriteIndex();
        }
    }

    /**
     * @brief Pans only the Mid (centre) component.
     */
    void applyMidPan(AudioBufferView<T> buffer, float /*pan*/)
    {
        assert(buffer.getNumChannels() >= 2);

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());
            T mid  = (L[i] + R[i]) * T(0.5);
            T side = (L[i] - R[i]) * T(0.5);
            T midGainL = T(1) - p;
            T midGainR = T(1) + p;
            L[i] = mid * midGainL + side;
            R[i] = mid * midGainR - side;
        }
    }

    /**
     * @brief Pans only the Side (stereo width) component.
     */
    void applySidePan(AudioBufferView<T> buffer, float /*pan*/)
    {
        assert(buffer.getNumChannels() >= 2);

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();
        constexpr T halfPi = pi<T> / T(2);

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());
            T mid  = (L[i] + R[i]) * T(0.5);
            T side = (L[i] - R[i]) * T(0.5);
            T angle = (p * T(0.5) + T(0.5)) * halfPi;
            T sideGainL = std::cos(angle);
            T sideGainR = std::sin(angle);
            L[i] = mid + side * sideGainL;
            R[i] = mid - side * sideGainR;
        }
    }

    /**
     * @brief Haas (precedence) effect panning via inter-channel delay.
     */
    void applyHaas(AudioBufferView<T> buffer, float pan)
    {
        assert(buffer.getNumChannels() >= 2);
        if (pan == 0.0f) return;

        T haasMax = T(haasMaxDelay_.load(std::memory_order_relaxed));
        T delayLms = haasMax * std::max(T(0), static_cast<T>(pan));
        T delayRms = haasMax * std::max(T(0), static_cast<T>(-pan));

        // Both delayL_ and delayR_ are mono (numChannels=1), so use ch=0
        // and process per-sample to advance the write index correctly
        delayL_.setDelayMs(delayLms);
        delayR_.setDelayMs(delayRms);

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        for (int i = 0; i < n; ++i)
        {
            L[i] = delayL_.processSample(0, L[i]);
            delayL_.advanceWriteIndex();
            R[i] = delayR_.processSample(0, R[i]);
            delayR_.advanceWriteIndex();
        }
    }

    /**
     * @brief Spectral (frequency-dependent) panning via high-shelf filter.
     *
     * Pan is smoothed via panSmoother_ to avoid zipper noise on automation.
     * Shelf coefficients are refreshed every kRefresh samples (~333µs @48kHz)
     * to keep per-sample cost low while remaining sonically continuous.
     */
    void applySpectral(AudioBufferView<T> buffer, float pan)
    {
        assert(buffer.getNumChannels() >= 2);
        pan = std::clamp(pan, -1.0f, 1.0f);
        panSmoother_.setTargetValue(pan);

        const int n = buffer.getNumSamples();
        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);

        constexpr int kRefresh = 16;
        for (int i = 0; i < n; ++i)
        {
            float p = panSmoother_.getNextValue();
            if ((i & (kRefresh - 1)) == 0)
                updateSpectralFilters(static_cast<T>(p));

            L[i] = spectralL_.processSample(L[i], 0);
            R[i] = spectralR_.processSample(R[i], 0);
        }
    }

protected:
    void updateSpectralFilters(T pan) noexcept
    {
        float sMaxGain = spectralMaxGain_.load(std::memory_order_relaxed);
        float sFreq = spectralFreq_.load(std::memory_order_relaxed);
        T gainLdB = -pan * static_cast<T>(sMaxGain);
        T gainRdB =  pan * static_cast<T>(sMaxGain);
        spectralL_.setCoeffs(BiquadCoeffs<T>::makeHighShelf(
            sampleRate_, static_cast<double>(sFreq), static_cast<double>(gainLdB)));
        spectralR_.setCoeffs(BiquadCoeffs<T>::makeHighShelf(
            sampleRate_, static_cast<double>(sFreq), static_cast<double>(gainRdB)));
    }

    double sampleRate_ = 48000.0;
    std::atomic<Algorithm> algorithm_ { Algorithm::EqualPower };
    std::atomic<T> pan_ { T(0) };

    Delay<T> delayL_, delayR_;
    Smoothers::LinearSmoother panSmoother_;
    Biquad<T, 1> spectralL_, spectralR_;

    std::atomic<float> smoothingTime_   { 50.0f };
    std::atomic<float> binauralMaxITD_  { 0.66f };
    std::atomic<float> haasMaxDelay_    { 30.0f };
    std::atomic<float> spectralFreq_    { 4000.0f };
    std::atomic<float> spectralMaxGain_ { 6.0f };
};

} // namespace dspark
