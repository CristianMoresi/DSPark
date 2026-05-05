// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file LadderFilter.h
 * @brief Moog-style 4-pole resonant ladder filter with TPT discretization.
 *
 * Classic analog-modelled filter using the Topology-Preserving Transform
 * (Zavalishin) for accurate analog behaviour without zero-delay feedback delay.
 * Includes non-linear drive for harmonic saturation and self-oscillation support.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cassert>

namespace dspark {

/**
 * @class LadderFilter
 * @brief 4-pole resonant ladder filter optimized for real-time SIMD and cache coherency.
 *
 * Thread-safe implementation utilizing atomic parameter loading. To maximize
 * performance, the filter mode is dispatched at the block level, avoiding branch
 * penalties in the sample-processing hot path.
 *
 * @tparam T Sample type (float or double). Must satisfy FloatType concept.
 */
template <FloatType T>
class LadderFilter
{
public:
    /** @brief Defines the frequency response mode of the filter output. */
    enum class Mode
    {
        LP6,    ///< 1-pole lowpass (6 dB/oct).
        LP12,   ///< 2-pole lowpass (12 dB/oct).
        LP18,   ///< 3-pole lowpass (18 dB/oct).
        LP24,   ///< 4-pole lowpass (24 dB/oct) — Classic Moog style.
        BP12,   ///< Bandpass (12 dB/oct).
        HP24    ///< 4-pole highpass (24 dB/oct).
    };

    ~LadderFilter() = default;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the filter with the current audio environment.
     * @param spec Audio specification including sample rate.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place (Thread-Safe).
     * * Creates an atomic snapshot of current parameters and dispatches to
     * a branchless inner loop via templates for maximum CPU efficiency.
     * * @param buffer View of the multi-channel audio data to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        // Local snapshot to guarantee thread-safety and avoid mid-block changes
        const T currentG = g_.load(std::memory_order_relaxed);
        const T currentRes = resonance_.load(std::memory_order_relaxed);
        const T currentDrive = drive_.load(std::memory_order_relaxed);
        const Mode currentMode = mode_.load(std::memory_order_relaxed);

        // Template dispatch to eliminate per-sample branching
        switch (currentMode)
        {
            case Mode::LP6:  processBlockInternal<Mode::LP6>(buffer, nCh, nS, currentG, currentRes, currentDrive); break;
            case Mode::LP12: processBlockInternal<Mode::LP12>(buffer, nCh, nS, currentG, currentRes, currentDrive); break;
            case Mode::LP18: processBlockInternal<Mode::LP18>(buffer, nCh, nS, currentG, currentRes, currentDrive); break;
            case Mode::LP24: processBlockInternal<Mode::LP24>(buffer, nCh, nS, currentG, currentRes, currentDrive); break;
            case Mode::BP12: processBlockInternal<Mode::BP12>(buffer, nCh, nS, currentG, currentRes, currentDrive); break;
            case Mode::HP24: processBlockInternal<Mode::HP24>(buffer, nCh, nS, currentG, currentRes, currentDrive); break;
        }
    }

    /**
     * @brief Clears the internal integrators state.
     * * Should be called when playback stops or continuity is broken to prevent clicks.
     */
    void reset() noexcept
    {
        for (auto& s : state_)
        {
            s.z.fill(T(0));
            s.stage.fill(T(0));
        }
    }

    // -- Parameters -------------------------------------------------------------

    /**
     * @brief Sets the cutoff frequency (Thread-Safe).
     * @param hz Cutoff frequency in Hz. Automatically clamped to [20, Nyquist].
     */
    void setCutoff(T hz) noexcept
    {
        cutoff_.store(std::clamp(hz, T(20), static_cast<T>(spec_.sampleRate) * T(0.499)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /**
     * @brief Sets resonance amount (Thread-Safe).
     * @param amount Range [0.0, 1.0]. Values near 1.0 induce self-oscillation.
     */
    void setResonance(T amount) noexcept
    {
        resonance_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the nonlinear drive amount (Thread-Safe).
     * @param amount Drive multiplier. 1.0 = linear. >1.0 = analog saturation.
     */
    void setDrive(T amount) noexcept 
    { 
        drive_.store(std::max(amount, T(0.1)), std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the filter output mode (Thread-Safe).
     * @param mode Target frequency response type.
     */
    void setMode(Mode mode) noexcept 
    { 
        mode_.store(mode, std::memory_order_relaxed); 
    }

    [[nodiscard]] T getCutoff() const noexcept { return cutoff_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getResonance() const noexcept { return resonance_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }

protected:
    static constexpr int kMaxChannels = 16;

    // Cache-line alignment (32 bytes) for SIMD vectorization requirements
    struct alignas(32) ChannelState
    {
        std::array<T, 4> z {};      // TPT Integrator states (z^-1 equivalents)
        std::array<T, 4> stage {};  // Tap outputs for multimode mixing
    };

    std::array<ChannelState, kMaxChannels> state_ {};
    AudioSpec spec_ {};

    // Atomic variables for thread-safety between Audio Thread and UI/Main Thread
    std::atomic<T> cutoff_ {T(1000)};
    std::atomic<T> resonance_ {T(0)};
    std::atomic<T> drive_ {T(1)};
    std::atomic<T> g_ {T(0)};
    std::atomic<Mode> mode_ {Mode::LP24};

private:
    /**
     * @brief Recalculates the analog pre-warped integrator gain.
     */
    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate > 0)
        {
            const T currentCutoff = cutoff_.load(std::memory_order_relaxed);
            const T preWarpedGain = static_cast<T>(std::tan(pi<double> * static_cast<double>(currentCutoff) / spec_.sampleRate));
            g_.store(preWarpedGain, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Core block processing routine isolated per mode.
     */
    template <Mode FilterMode>
    void processBlockInternal(AudioBufferView<T>& buffer, int nCh, int nS, T g, T res, T drive) noexcept
    {
        for (int ch = 0; ch < nCh; ++ch)
        {
            auto* channelData = buffer.getChannel(ch);
            auto& state = state_[ch];
            
            for (int i = 0; i < nS; ++i)
            {
                channelData[i] = processSampleInternal<FilterMode>(channelData[i], state, g, res, drive);
            }
        }
    }

    /**
     * @brief Core sample-processing DSP algorithm.
     * * Uses template parameters for output mode to ensure the compiler completely
     * inlines and eliminates branches inside the audio processing loop.
     */
    template <Mode FilterMode>
    [[nodiscard]] inline T processSampleInternal(T input, ChannelState& s, T g, T res, T drive) noexcept
    {
        // TPT integrator coefficients
        const T G = g / (T(1) + g);
        const T G2 = G * G;
        const T G3 = G2 * G;
        const T G4 = G3 * G;
        const T ig = T(1) / (T(1) + g);

        // Estimate LP24 output from integrators (Zero-Delay logic)
        T Sest = G3 * ig * s.z[0]
               + G2 * ig * s.z[1]
               + G  * ig * s.z[2]
               +      ig * s.z[3];

        // Resonance feedback coefficient (k = 4 -> self oscillation)
        const T k = res * T(4);

        // Apply drive to estimated feedback (ZDF non-linear approximation)
        T fbSignal = Sest;
        if (drive > T(1))
            fbSignal = fastTanh(fbSignal * drive) / drive;

        // Resolve zero-delay loop
        const T u = (input - k * fbSignal) / (T(1) + k * G4);

        // Process 4 cascading integrators
        T x = u;
        for (int i = 0; i < 4; ++i)
        {
            T v = (x - s.z[i]) * G;
            T y = v + s.z[i];
            s.z[i] = y + v;
            s.stage[i] = y;
            x = y;
        }

        // Output logic (resolved at compile time per block instantiation)
        if constexpr (FilterMode == Mode::LP6)  return s.stage[0];
        if constexpr (FilterMode == Mode::LP12) return s.stage[1];
        if constexpr (FilterMode == Mode::LP18) return s.stage[2];
        if constexpr (FilterMode == Mode::LP24) return s.stage[3];
        if constexpr (FilterMode == Mode::BP12) return s.stage[0] - s.stage[2];
        if constexpr (FilterMode == Mode::HP24) return input - T(4)*s.stage[0] + T(6)*s.stage[1] - T(4)*s.stage[2] + s.stage[3];
        
        return s.stage[3]; // Fallback, shouldn't reach here
    }
};

} // namespace dspark