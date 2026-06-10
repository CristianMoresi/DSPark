// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file DCBlocker.h
 * @brief Removes DC offset from audio signals with configurable filter order.
 *
 * Order 1 uses a lightweight 1-pole high-pass filter (2 multiplies, 2 additions
 * per sample). Orders 2–10 use cascaded Butterworth biquad high-pass stages
 * with predefined Q values for maximally-flat passband response.
 *
 * Dependencies: DspMath.h, Biquad.h, AudioSpec.h, AudioBuffer.h.
 */

#include "../Core/DspMath.h"
#include "../Core/Biquad.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class DCBlocker
 * @brief DC blocking filter with configurable Butterworth order (1–10).
 *
 * Designed for strict real-time processing. Zero allocations, cache-friendly
 * memory layout (32-byte aligned), and branchless per-sample inner loops.
 * 
 * @warning Do not modulate the cutoff frequency at audio rates, as updating 
 * biquad coefficients triggers trigonometric functions.
 *
 * @tparam T Sample type (float or double). internal state precision should ideally 
 * be double if processing very low cutoffs at extremely high sample rates.
 */
template <FloatType T>
class alignas(32) DCBlocker
{
public:
    ~DCBlocker() = default; // Non-virtual to avoid vtable injection

    static constexpr int kMaxBiquadStages = 5;
    static constexpr int kMaxChannels = 16;

    /**
     * @brief Prepares the DC blocker, resetting internal states and precalculating coefficients.
     *
     * @param sampleRate Sample rate in Hz.
     * @param numChannels Number of audio channels (max 16, default: 2).
     * @param cutoffHz    Cutoff frequency in Hz (clamped to min 1 Hz, default: 5.0).
     */
    void prepare(double sampleRate, int numChannels = 2, double cutoffHz = 5.0)
    {
        sampleRate_ = sampleRate;
        numChannels_ = std::clamp(numChannels, 1, kMaxChannels);
        
        reset();
        
        // Use the thread-safe setter to clamp and initialize safely
        setCutoff(static_cast<T>(cutoffHz));
        forceUpdateCoefficients();
    }

    /** @brief Prepares from AudioSpec (unified API). */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Sets the filter order (1–10). Thread-safe.
     *
     * - Order 1: efficient 1-pole filter (6 dB/oct).
     * - Order 2–10: cascaded Butterworth biquad HPFs (12*N dB/oct for even orders).
     * 
     * @warning Changing order during playback may result in audio clicks, as filter
     * states are not dynamically crossfaded. Best used during prepare() or silence.
     *
     * @note Order 1 = 1-pole (6 dB/oct). Orders >= 2 use floor(order/2) cascaded
     *       Butterworth biquads, so ODD orders behave like the next lower even
     *       order (e.g. 3 == 2, 5 == 4). Prefer 1 or even values for predictable slopes.
     *
     * @param order Filter order (1–10).
     */
    void setOrder(int order) noexcept
    {
        order_.store(std::clamp(order, 1, 10), std::memory_order_relaxed);
    }

    /** @brief Returns the current filter order. */
    [[nodiscard]] int getOrder() const noexcept 
    { 
        return order_.load(std::memory_order_relaxed); 
    }

    /**
     * @brief Requests a cutoff frequency update.
     * 
     * Coefficients are recomputed lazily on the next processBlock() call to 
     * ensure thread safety, but beware that calculating high-order biquads involves 
     * trigonometric operations.
     * 
     * @param hz Cutoff frequency in Hz (clamped to min 1.0).
     */
    void setCutoff(T hz) noexcept
    {
        cutoffHz_.store(std::max(hz, T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Processes an AudioBufferView in-place.
     *
     * Checks for parameter updates and applies the appropriate filter topology
     * across all channels using branchless inner loops for auto-vectorization.
     *
     * @param buffer Audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        updateCoefficientsIfNeeded();

        const int currentOrder = order_.load(std::memory_order_relaxed);
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();

        if (currentOrder <= 1)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                // SIMD-friendly loop (no branches)
                for (int i = 0; i < nS; ++i)
                {
                    T input = data[i];
                    T output = input - xPrev_[ch] + R_ * yPrev_[ch];
                    xPrev_[ch] = input;
                    yPrev_[ch] = output;
                    data[i] = output;
                }
            }
        }
        else
        {
            const int numStages = std::clamp(currentOrder / 2, 1, kMaxBiquadStages);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                {
                    T sample = data[i];
                    for (int s = 0; s < numStages; ++s)
                        sample = biquadStages_[s].processSample(sample, ch);
                    data[i] = sample;
                }
            }
        }
    }

    /**
     * @brief Processes a block of samples for one channel in-place.
     *
     * @param channel Channel index.
     * @param data Audio samples (modified in-place).
     * @param numSamples Number of samples to process.
     */
    void processBlock(int channel, T* data, int numSamples) noexcept
    {
        updateCoefficientsIfNeeded();
        const int currentOrder = order_.load(std::memory_order_relaxed);
        
        if (currentOrder <= 1)
        {
            auto ch = static_cast<size_t>(channel);
            for (int i = 0; i < numSamples; ++i)
            {
                T input = data[i];
                T output = input - xPrev_[ch] + R_ * yPrev_[ch];
                xPrev_[ch] = input;
                yPrev_[ch] = output;
                data[i] = output;
            }
        }
        else
        {
            const int numStages = std::clamp(currentOrder / 2, 1, kMaxBiquadStages);
            for (int i = 0; i < numSamples; ++i)
            {
                T sample = data[i];
                for (int s = 0; s < numStages; ++s)
                    sample = biquadStages_[s].processSample(sample, channel);
                data[i] = sample;
            }
        }
    }

    /**
     * @brief Processes a single sample for a given channel.
     *
     * @note Only use this if external parameter polling is handled manually. 
     * For processing multiple samples, prefer processBlock() to avoid overhead.
     *
     * @param channel Channel index.
     * @param input Input sample.
     * @return Sample with DC offset removed.
     */
    [[nodiscard]] T processSample(int channel, T input) noexcept
    {
        if (order_.load(std::memory_order_relaxed) <= 1)
        {
            auto ch = static_cast<size_t>(channel);
            T output = input - xPrev_[ch] + R_ * yPrev_[ch];
            xPrev_[ch] = input;
            yPrev_[ch] = output;
            return output;
        }

        T sample = input;
        const int numStages = std::clamp(order_.load(std::memory_order_relaxed) / 2, 1, kMaxBiquadStages);
        for (int s = 0; s < numStages; ++s)
            sample = biquadStages_[s].processSample(sample, channel);
            
        return sample;
    }

    /**
     * @brief Clears the internal history states to zero.
     */
    void reset() noexcept
    {
        xPrev_.fill(T(0));
        yPrev_.fill(T(0));
        for (auto& stage : biquadStages_)
            stage.reset();
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DCBL"), 1);
        w.write("order", order_.load(std::memory_order_relaxed));
        w.write("cutoff", cutoffHz_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DCBL")) return false;
        setOrder(r.read("order", 1));
        setCutoff(static_cast<T>(r.read("cutoff", 5.0f)));
        return true;
    }

protected:
    void updateCoefficientsIfNeeded() noexcept
    {
        T cutoff = cutoffHz_.load(std::memory_order_relaxed);
        if (cutoff != lastCutoff_)
        {
            forceUpdateCoefficients(cutoff);
        }
    }

    void forceUpdateCoefficients(T explicitCutoff = T(0)) noexcept
    {
        T cutoff = explicitCutoff > T(0) ? explicitCutoff : cutoffHz_.load(std::memory_order_relaxed);
        
        // 1-Pole update
        R_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0 * static_cast<double>(cutoff) / sampleRate_));
        
        // Biquad updates
        static constexpr float qTable[6][kMaxBiquadStages] = {
            {},                                                // index 0 (unused)
            { 0.7071f },                                       // order 2
            { 0.5412f, 1.3066f },                              // order 4
            { 0.5177f, 0.7071f, 1.9319f },                     // order 6
            { 0.5098f, 0.6013f, 0.8999f, 2.5628f },            // order 8
            { 0.5062f, 0.5612f, 0.7071f, 1.1013f, 3.1962f }    // order 10
        };

        const int tableIdx = std::clamp(order_.load(std::memory_order_relaxed) / 2, 1, 5);
        for (int s = 0; s < tableIdx; ++s)
        {
            auto c = BiquadCoeffs<T>::makeHighPass(
                sampleRate_, static_cast<double>(cutoff),
                static_cast<double>(qTable[tableIdx][s]));
            biquadStages_[s].setCoeffs(c);
        }

        lastCutoff_ = cutoff;
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 2;
    T R_ = T(0.995);
    
    std::atomic<int> order_ { 1 };
    std::atomic<T> cutoffHz_ { T(5) };
    T lastCutoff_ = T(-1); 

    // Cache-friendly fixed size states (aligned to 32 bytes via class alignment)
    std::array<T, kMaxChannels> xPrev_{};
    std::array<T, kMaxChannels> yPrev_{};

    // Biquad cascade state
    std::array<Biquad<T, kMaxChannels>, kMaxBiquadStages> biquadStages_ {};
};

} // namespace dspark
