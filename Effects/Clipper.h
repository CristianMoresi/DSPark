// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Clipper.h
 * @brief Multi-mode clipping processor with oversampling and slew limiting.
 *
 * Four clipping modes:
 *
 * - **Hard**: Digital clamp — transparent until ceiling, brickwall at ceiling.
 * - **Soft**: Hyperbolic tangent — smooth saturation with no hard discontinuity.
 * - **Analog**: Sine-based — models the soft clipping of transformer saturation.
 * - **GoldenRatio**: Hard clip with phi-weighted interpolated reconstruction —
 *   golden-ratio blend between clipped and unclipped produces harmonically
 *   pleasing transition at the clipping point.
 *
 * Features:
 * - Input gain (drive) up to +48 dB
 * - Ceiling from -60 to 0 dBFS
 * - Multi-stage clipping (1–4 stages, splits gain across stages)
 * - Optional slew limiter (anti-alias at the clip discontinuity)
 * - Optional 2x or 4x oversampling (uses Core/Oversampling.h)
 * - Dry/wet mix via DryWetMixer
 *
 * All parameters are std::atomic for thread-safe real-time control.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h, DryWetMixer.h, Oversampling.h.
 *
 * @code
 *   dspark::Clipper<float> clip;
 *   clip.prepare(spec);
 *   clip.setMode(dspark::Clipper<float>::Mode::Soft);
 *   clip.setCeiling(-1.0f);      // -1 dBFS
 *   clip.setInputGain(12.0f);    // +12 dB drive
 *   clip.processBlock(buffer);
 *
 *   // Multi-stage analog clipping:
 *   clip.setMode(dspark::Clipper<float>::Mode::Analog);
 *   clip.setStages(3);           // 3 cascaded stages
 *   clip.setInputGain(24.0f);    // split as 8 dB per stage
 *
 *   // With oversampling for maximum quality:
 *   clip.setOversampling(4);     // 4x oversampling
 *   clip.prepare(spec);          // re-prepare to allocate OS buffers
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/DryWetMixer.h"
#include "../Core/Oversampling.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <numbers>

namespace dspark {

/**
 * @class Clipper
 * @brief Multi-mode clipper with oversampling and slew limiting.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Clipper
{
public:
    /** @brief Clipping algorithm. */
    enum class Mode
    {
        Hard,         ///< Digital brickwall clamp.
        Soft,         ///< tanh soft clipping.
        Analog,       ///< Sine-based analog-style soft clip.
        GoldenRatio   ///< Hard clip with phi-weighted interpolated reconstruction.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the clipper for processing.
     *
     * Allocates oversampling buffers if oversampling > 1. Must be called
     * before processBlock(), and again if oversampling factor changes.
     *
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        int osFactor = osFactor_.load(std::memory_order_relaxed);
        oversampler_ = std::make_unique<Oversampling<T>>(
            osFactor, Oversampling<T>::Quality::High);
        oversampler_->prepare(spec);

        for (int ch = 0; ch < kMaxChannels; ++ch)
            slewPrev_[ch] = T(0);
    }

    /** @brief Resets internal state. */
    void reset() noexcept
    {
        mixer_.reset();
        if (oversampler_) oversampler_->reset();
        for (int ch = 0; ch < kMaxChannels; ++ch)
            slewPrev_[ch] = T(0);
        gainReductionDb_.store(T(0), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------

    /**
     * @brief Processes audio through the clipper.
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        T mixVal = mix_.load(std::memory_order_relaxed);
        mixer_.pushDry(buffer);

        if (oversampler_ && oversampler_->getFactor() > 1)
        {
            auto upView = oversampler_->upsample(buffer);
            processClipping(upView);
            oversampler_->downsample(buffer);
        }
        else
        {
            processClipping(buffer);
        }

        mixer_.mixWet(buffer, mixVal);
    }

    // -- Parameters (all thread-safe) -------------------------------------------

    /**
     * @brief Sets the clipping mode.
     * @param mode Clipping algorithm.
     */
    void setMode(Mode mode) noexcept
    {
        mode_.store(mode, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the ceiling level.
     * @param dB Ceiling in dBFS (-60 to 0).
     */
    void setCeiling(T dB) noexcept
    {
        ceilingDb_.store(std::clamp(dB, T(-60), T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the input gain (drive).
     * @param dB Input gain in dB (0 to 48).
     */
    void setInputGain(T dB) noexcept
    {
        inputGainDb_.store(std::clamp(dB, T(0), T(48)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the number of clipping stages.
     *
     * Multi-stage clipping divides the input gain across stages, applying
     * the clip algorithm N times. Produces different harmonic content than
     * single-stage at the same total gain.
     *
     * @param count Number of stages (1–4).
     */
    void setStages(int count) noexcept
    {
        stages_.store(std::clamp(count, 1, kMaxStages), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dry/wet mix.
     * @param amount 0 = dry, 1 = wet.
     */
    void setMix(T amount) noexcept
    {
        mix_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Enables slew limiting and sets the rate.
     *
     * Limits the maximum rate of change per sample at the clipping point,
     * reducing aliasing from hard clip discontinuities. Higher values = more
     * limiting (smoother but less bright).
     *
     * @param amount 0 = off, 0.01–1.0 = increasing smoothness.
     */
    void setSlewLimit(T amount) noexcept
    {
        slewLimit_.store(std::max(amount, T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the oversampling factor.
     *
     * Clipping always uses at least 2x oversampling to reduce aliasing
     * from the nonlinear waveshaping. Call prepare() again after changing.
     *
     * @param factor 2, 4, 8, or 16.
     */
    void setOversampling(int factor) noexcept
    {
        // Round to nearest valid power of two (minimum 2x)
        if (factor <= 2) factor = 2;
        else if (factor <= 4) factor = 4;
        else if (factor <= 8) factor = 8;
        else factor = 16;
        osFactor_.store(factor, std::memory_order_relaxed);
    }

    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getCeiling() const noexcept { return ceilingDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getInputGain() const noexcept { return inputGainDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getStages() const noexcept { return stages_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getOversampling() const noexcept { return osFactor_.load(std::memory_order_relaxed); }

    /**
     * @brief Returns the latency in samples introduced by oversampling.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return oversampler_ ? oversampler_->getLatency() : 0;
    }

    /** @brief Returns the current gain reduction in dB (for metering). */
    [[nodiscard]] T getGainReductionDb() const noexcept
    {
        return gainReductionDb_.load(std::memory_order_relaxed);
    }

protected:
    static constexpr int kMaxStages = 4;
    static constexpr int kMaxChannels = 16;

    /// Golden ratio (phi) for GoldenRatio mode interpolation.
    static constexpr T kPhi = static_cast<T>(1.6180339887498948482);
    static constexpr T kInvPhiPlusOne = T(1) / (kPhi + T(1));

    /**
     * @brief Core clipping process (operates at whatever sample rate the buffer is at).
     */
    void processClipping(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        auto modeVal     = mode_.load(std::memory_order_relaxed);
        T ceilDb         = ceilingDb_.load(std::memory_order_relaxed);
        T gainDb         = inputGainDb_.load(std::memory_order_relaxed);
        int numStages    = stages_.load(std::memory_order_relaxed);
        T slewLim        = slewLimit_.load(std::memory_order_relaxed);

        T ceiling        = dbToLinear(ceilDb);
        T totalGainLin   = dbToLinear(gainDb);
        // Split gain across stages: each stage gets totalGain^(1/N)
        T stageGain      = (numStages > 1)
                           ? std::pow(totalGainLin, T(1) / static_cast<T>(numStages))
                           : totalGainLin;

        T peakIn  = T(0);
        T peakOut = T(0);

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                T sample = data[i];
                T driven = sample * totalGainLin;

                for (int s = 0; s < numStages; ++s)
                {
                    sample *= stageGain;
                    sample = clipSample(sample, ceiling, modeVal);
                }

                // Slew limiter
                if (slewLim > T(0))
                {
                    T maxDelta = ceiling * slewLim;
                    T delta = sample - slewPrev_[ch];
                    if (std::abs(delta) > maxDelta)
                        sample = slewPrev_[ch] + std::copysign(maxDelta, delta);
                }
                slewPrev_[ch] = sample;

                peakIn  = std::max(peakIn,  std::abs(driven));
                peakOut = std::max(peakOut, std::abs(sample));

                data[i] = sample;
            }
        }

        // Gain reduction metering
        if (peakIn > T(1e-6))
        {
            auto ratio = std::min(peakOut / peakIn, T(1));
            gainReductionDb_.store(gainToDecibels(ratio, T(-100)), std::memory_order_relaxed);
        }
        else
        {
            gainReductionDb_.store(T(0), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Clips a single sample using the selected mode.
     */
    [[nodiscard]] static T clipSample(T sample, T ceiling, Mode mode) noexcept
    {
        switch (mode)
        {
            case Mode::Hard:
                return std::clamp(sample, -ceiling, ceiling);

            case Mode::Soft:
                return ceiling * std::tanh(sample / ceiling);

            case Mode::Analog:
            {
                T normalized = std::clamp(sample / ceiling, T(-1), T(1));
                return ceiling * std::sin(normalized * static_cast<T>(std::numbers::pi * 0.5));
            }

            case Mode::GoldenRatio:
            {
                // Hard clip + phi-weighted interpolation at clipping boundary
                T clipped = std::clamp(sample, -ceiling, ceiling);
                // Blend: phi * clipped + 1 * sample, normalized by (phi + 1)
                // Only blends near the clipping point; transparent below ceiling
                T blended = (kPhi * clipped + sample) * kInvPhiPlusOne;
                // Final safety clamp
                return std::clamp(blended, -ceiling, ceiling);
            }
        }
        return sample;
    }

    /** @brief Converts dB to linear gain. */
    [[nodiscard]] static T dbToLinear(T dB) noexcept
    {
        return std::pow(T(10), dB / T(20));
    }

    AudioSpec spec_ {};
    DryWetMixer<T> mixer_;
    std::unique_ptr<Oversampling<T>> oversampler_;

    // Atomic parameters
    std::atomic<Mode> mode_ { Mode::Hard };
    std::atomic<T> ceilingDb_ { T(0) };
    std::atomic<T> inputGainDb_ { T(0) };
    std::atomic<int> stages_ { 1 };
    std::atomic<T> mix_ { T(1) };
    std::atomic<T> slewLimit_ { T(0) };
    std::atomic<int> osFactor_ { 2 };

    // Per-channel slew state
    T slewPrev_[kMaxChannels] {};

    // Gain reduction tracking
    std::atomic<T> gainReductionDb_ { T(0) };
};

} // namespace dspark
