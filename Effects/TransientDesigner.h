// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file TransientDesigner.h
 * @brief Transient shaping via dual-envelope analysis (VCA Logarithmic Model).
 *
 * Emulates classic analog transient shapers by calculating the difference
 * between a fast (peak) and slow (RMS) envelope in the logarithmic domain.
 * This effectively separates the attack and release phases, applying
 * independent gain adjustments to the transient and the body.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 *
 * @code
 *   dspark::TransientDesigner<float> td;
 *   td.prepare(spec);
 *   td.setAttack(0.5f);    // +50% attack emphasis
 *   td.setSustain(-0.3f);  // -30% sustain reduction
 *   td.processBlock(buffer);
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class TransientDesigner
 * @brief Zero-allocation, thread-safe, SIMD-friendly transient shaper.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TransientDesigner
{
public:
    ~TransientDesigner() = default; // non-virtual: leaf class (no virtual dispatch)

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the processor with the current audio specification.
     * @param spec Audio specification including sample rate.
     */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate);
    }

    /**
     * @brief Prepares the processor with a specific sample rate.
     * @param sampleRate The operating sample rate in Hz.
     */
    void prepare(double sampleRate) noexcept
    {
        sampleRate_ = sampleRate;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place.
     * @param buffer View of the audio buffer to process.
     * @note Loop order optimized for SoA (Structure of Arrays) SIMD execution.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        const T attAmt = attackAmount_.load(std::memory_order_relaxed);
        const T susAmt = sustainAmount_.load(std::memory_order_relaxed);
        const bool odr = outputDepRecovery_.load(std::memory_order_relaxed);

        // Constants for VCA log-domain emulation
        constexpr T noiseFloor = T(1e-5);     // -100 dB floor avoids log(0) and denormals
        constexpr T maxGainLog = T(2.77258);  // approx +24dB max gain change

        // Outer loop: Channels (Cache & SIMD friendly)
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* const channelData = buffer.getChannel(ch);
            T fast = envFast_[ch];
            T slow = envSlow_[ch];
            T lastOut = lastOutput_[ch];

            // Inner loop: Samples (Auto-vectorization target)
            for (int i = 0; i < nS; ++i)
            {
                T sample = channelData[i];
                T absSample = std::abs(sample) + noiseFloor;

                // 1. Fast envelope (Peak)
                T fastCoeff = (absSample > fast) ? fastAttackCoeff_ : fastReleaseCoeff_;
                fast += fastCoeff * (absSample - fast);

                // 2. Slow envelope (Sustain/RMS tracker)
                T currentSlowRelCoeff = slowReleaseCoeff_;
                
                if (odr) 
                {
                    // Corrected ODR: Higher output = LARGER coefficient = Faster release
                    T modifier = T(1) + std::abs(lastOut) * T(2.0); 
                    currentSlowRelCoeff = std::min(fastReleaseCoeff_, currentSlowRelCoeff * modifier);
                }

                T slowCoeff = (absSample > slow) ? slowAttackCoeff_ : currentSlowRelCoeff;
                slow += slowCoeff * (absSample - slow);

                // 3. Log-domain VCA computation (fastLog/fastExp: relative
                // error < 2e-7 — far below audibility, ~3x faster than libm)
                T diffLog = fastLog(fast) - fastLog(slow);

                // Attack kicks in when fast > slow (diffLog > 0). Sustain when slow > fast (diffLog < 0)
                T gainLog = (diffLog > T(0)) ? (diffLog * attAmt) : (-diffLog * susAmt);

                gainLog = std::clamp(gainLog, -maxGainLog, maxGainLog);

                T gain = fastExp(gainLog);
                
                lastOut = sample * gain;
                channelData[i] = lastOut;
            }

            // Save state
            envFast_[ch] = fast;
            envSlow_[ch] = slow;
            lastOutput_[ch] = lastOut;
        }
    }

    // -- Parameters ----------------------------------------------------------

    /**
     * @brief Sets attack (transient) emphasis.
     * @param amount -100 to +100 (%). Positive = boost transients, negative = soften.
     */
    void setAttack(T amount) noexcept
    {
        attackAmount_.store(std::clamp(amount, T(-100), T(100)) / T(100), std::memory_order_relaxed);
    }

    /**
     * @brief Sets sustain (body) emphasis.
     * @param amount -100 to +100 (%). Positive = boost sustain, negative = reduce.
     */
    void setSustain(T amount) noexcept
    {
        sustainAmount_.store(std::clamp(amount, T(-100), T(100)) / T(100), std::memory_order_relaxed);
    }

    /**
     * @brief Enables output-dependent recovery (ODR).
     * @param enabled If true, slow envelope release speed scales with output level.
     */
    void setOutputDepRecovery(bool enabled) noexcept
    {
        outputDepRecovery_.store(enabled, std::memory_order_relaxed);
    }

    /**
     * @brief Sets character as a single macro-knob.
     * @param amount Range [-1.0, 1.0]. -1 = soften transients/boost sustain, +1 = boost transients/reduce sustain.
     */
    void setCharacter(T amount) noexcept
    {
        T c = std::clamp(amount, T(-1), T(1));
        attackAmount_.store(c, std::memory_order_relaxed);
        sustainAmount_.store(-c * T(0.5), std::memory_order_relaxed);
    }

    /**
     * @brief Clears all internal buffers and state. Must be lock-free.
     */
    void reset() noexcept
    {
        envFast_.fill(T(1e-5)); // Init to noise floor
        envSlow_.fill(T(1e-5));
        lastOutput_.fill(T(0));
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("TDES"), 1);
        w.write("attack", attackAmount_.load(std::memory_order_relaxed) * 100.0f);   // percent
        w.write("sustain", sustainAmount_.load(std::memory_order_relaxed) * 100.0f); // percent
        w.write("outputDep", outputDepRecovery_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("TDES")) return false;
        setAttack(static_cast<T>(r.read("attack", 0.0f)));
        setSustain(static_cast<T>(r.read("sustain", 0.0f)));
        setOutputDepRecovery(r.read("outputDep", false));
        return true;
    }

private:
    static constexpr int kMaxChannels = 16;

    void updateCoefficients() noexcept
    {
        if (sampleRate_ <= 0.0) return;
        T fs = static_cast<T>(sampleRate_);
        fastAttackCoeff_  = T(1) - std::exp(T(-1) / (fs * T(0.0001)));
        fastReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.005)));
        slowAttackCoeff_  = T(1) - std::exp(T(-1) / (fs * T(0.020)));
        slowReleaseCoeff_ = T(1) - std::exp(T(-1) / (fs * T(0.200)));
    }

    double sampleRate_ = 48000.0;

    // Atomic parameters (Lock-free thread safety)
    std::atomic<T> attackAmount_ { T(0) };
    std::atomic<T> sustainAmount_ { T(0) };
    std::atomic<bool> outputDepRecovery_ { false };

    // Envelope coefficients
    T fastAttackCoeff_ = T(0), fastReleaseCoeff_ = T(0);
    T slowAttackCoeff_ = T(0), slowReleaseCoeff_ = T(0);

    // Per-channel state (Aligned for SIMD)
    alignas(32) std::array<T, kMaxChannels> envFast_ {};
    alignas(32) std::array<T, kMaxChannels> envSlow_ {};
    alignas(32) std::array<T, kMaxChannels> lastOutput_ {};
};

} // namespace dspark
