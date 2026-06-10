// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file FrequencyShifter.h
 * @brief Frequency shifting (not pitch shifting) using Hilbert transform.
 *
 * Shifts all frequency components by a fixed amount in Hz using Single Sideband 
 * (SSB) modulation. Unlike pitch shifting, this does NOT preserve harmonic 
 * relationships.
 *
 * Implementation: 
 * Applies a Hilbert transform to extract the analytic signal (I + jQ).
 * The complex signal is modulated by a quadrature oscillator e^(j·2π·f·t).
 * To ensure phase coherency when mixing Dry/Wet, the "Dry" signal is taken 
 * directly from the real branch of the Hilbert transformer, avoiding comb-filtering.
 *
 * Dependencies: DspMath.h, Hilbert.h, AudioSpec.h, AudioBuffer.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Hilbert.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class FrequencyShifter
 * @brief Constant-Hz frequency shift optimized via Quadrature Oscillator.
 *
 * Uses a recursive rotation matrix to generate the complex carrier, avoiding 
 * expensive per-sample trigonometric calls. Supports arbitrary channel counts.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class FrequencyShifter
{
public:
    /**
     * @brief Prepares the frequency shifter state and allocates internal buffers.
     * @param spec Audio environment specification (sample rate, num channels).
     */
    void prepare(const AudioSpec& spec)
    {
        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;
        
        // Zero-allocations on audio thread: allocate all Hilberts during prepare.
        hilberts_.resize(numChannels_);
        for (auto& h : hilberts_) {
            h.prepare(spec.sampleRate);
        }
        
        reset();
    }

    /**
     * @brief Processes audio in-place applying the frequency shift.
     * @param buffer Audio data view.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();
        if (numSamples == 0 || numCh == 0) return;

        const T mixVal = mix_.load(std::memory_order_relaxed);
        const T shiftHz = shift_.load(std::memory_order_relaxed);

        // 1. Compute rotation matrix coefficients once per block
        const double w = (shiftHz * 2.0 * std::numbers::pi) / sampleRate_;
        const T cos_w = static_cast<T>(std::cos(w));
        const T sin_w = static_cast<T>(std::sin(w));

        // Cache the starting phase components for this block
        const T startCos = static_cast<T>(std::cos(phase_));
        const T startSin = static_cast<T>(std::sin(phase_));

        // 2. Process planar channels to maximize cache locality
        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& hilbert = hilberts_[ch];

            // Local quadrature oscillator state
            T u = startCos;
            T v = startSin;

            for (int i = 0; i < numSamples; ++i)
            {
                // Hilbert processing (I + jQ)
                auto h = hilbert.process(data[i]);

                // Modulate analytic signal: real part of (I+jQ) * (u+jv)
                T shifted = h.real * u - h.imag * v;

                // Mix correctly: Use h.real as the phase-aligned dry signal
                data[i] = h.real + (shifted - h.real) * mixVal;

                // Advance quadrature oscillator: rotation matrix
                T next_u = u * cos_w - v * sin_w;
                T next_v = u * sin_w + v * cos_w;
                u = next_u;
                v = next_v;
            }
        }

        // 3. Advance absolute phase once per block to prevent float drift
        phase_ += w * numSamples;
        
        // Wrap phase precisely
        constexpr double kTwoPi = 2.0 * std::numbers::pi;
        phase_ = std::fmod(phase_, kTwoPi);
        if (phase_ < 0.0) phase_ += kTwoPi;
    }

    /** 
     * @brief Resets internal filter states and phase accumulator. 
     */
    void reset() noexcept
    {
        phase_ = 0.0;
        for (auto& h : hilberts_) {
            h.reset();
        }
    }

    /**
     * @brief Sets the frequency shift amount in Hz.
     * @param hz Shift in Hz (Negative shifts frequencies down).
     */
    void setShift(T hz) noexcept
    {
        shift_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dry/wet mix.
     * @param mix Range [0.0, 1.0]. 0.0 = pure dry (phase-aligned), 1.0 = fully shifted.
     */
    void setMix(T mix) noexcept
    {
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @return The current frequency shift amount in Hz. */
    [[nodiscard]] T getShift() const noexcept { return shift_.load(std::memory_order_relaxed); }

    /** @return The current dry/wet mix ratio. */
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /**
     * @brief Reports the processing latency in samples.
     *
     * The Hilbert transformer delays BOTH the dry (real branch) and shifted
     * paths by its FIR group delay; report this for plugin delay compensation.
     */
    [[nodiscard]] static constexpr int getLatency() noexcept
    {
        return Hilbert<T>::getLatencySamples();
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("FSHF"), 1);
        w.write("shift", shift_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("FSHF")) return false;
        setShift(static_cast<T>(r.read("shift", 0.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        return true;
    }

private:
    double sampleRate_ = 44100.0;
    int numChannels_ = 0;
    
    // Using double for phase accumulation to prevent drift over long sessions
    double phase_ = 0.0; 

    std::atomic<T> shift_{ T(0) };
    std::atomic<T> mix_{ T(1) };

    std::vector<Hilbert<T>> hilberts_;
};

} // namespace dspark
