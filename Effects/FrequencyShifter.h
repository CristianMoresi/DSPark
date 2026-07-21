// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file FrequencyShifter.h
 * @brief Frequency shifting (not pitch shifting) using Hilbert transform.
 *
 * Shifts all frequency components by a fixed amount in Hz using Single
 * Sideband (SSB) modulation. Unlike pitch shifting, this does NOT preserve
 * harmonic relationships.
 *
 * Implementation:
 * Applies a Hilbert transform to extract the analytic signal (I + jQ).
 * The complex signal is modulated by a quadrature oscillator e^(j*2*pi*f*t).
 * To ensure phase coherency when mixing Dry/Wet, the "Dry" signal is taken
 * directly from the real branch of the Hilbert transformer, avoiding
 * comb-filtering.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Hilbert.h, DspMath.h, AudioSpec.h, AudioBuffer.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Hilbert.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class FrequencyShifter
 * @brief Constant-Hz frequency shift optimized via Quadrature Oscillator.
 *
 * Uses a recursive rotation matrix to generate the complex carrier, avoiding
 * expensive per-sample trigonometric calls; the oscillator state is re-anchored
 * from the double-precision phase accumulator at every block, so rounding
 * drift never accumulates past one block.
 *
 * Channels beyond those passed to prepare() are left untouched
 * (pass-through), as is the whole buffer before prepare().
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class FrequencyShifter
{
public:
    /**
     * @brief Prepares the frequency shifter state and allocates internal buffers.
     *
     * Settles the mix smoothing state on the published target. An invalid
     * spec (non-positive or non-finite fields) is a no-op that keeps the
     * previous state.
     *
     * @param spec Audio environment specification (sample rate, num channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        // Zero-allocations on audio thread: allocate all Hilberts during prepare.
        hilberts_.resize(numChannels_);
        for (auto& h : hilberts_) {
            h.prepare(spec.sampleRate);
        }

        currentMix_ = mix_.load(std::memory_order_relaxed);
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

        const T targetMix = mix_.load(std::memory_order_relaxed);
        const T shiftHz = shift_.load(std::memory_order_relaxed);

        // The mix is smoothed with a linear per-block ramp: an unsmoothed
        // step would jump the dry/wet crossfade audibly. The shift needs no
        // smoothing - it only changes the speed of the quadrature carrier,
        // whose phase stays continuous across blocks.
        const T mixInc = (targetMix - currentMix_) / static_cast<T>(numSamples);

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

            // Local quadrature oscillator state (identical carrier and mix
            // ramp on every channel)
            T u = startCos;
            T v = startSin;
            T smoothMix = currentMix_;

            for (int i = 0; i < numSamples; ++i)
            {
                smoothMix += mixInc;

                // Hilbert processing (I + jQ)
                auto h = hilbert.process(data[i]);

                // Modulate analytic signal: real part of (I+jQ) * (u+jv)
                T shifted = h.real * u - h.imag * v;

                // Mix correctly: Use h.real as the phase-aligned dry signal
                data[i] = h.real + (shifted - h.real) * smoothMix;

                // Advance quadrature oscillator: rotation matrix
                T next_u = u * cos_w - v * sin_w;
                T next_v = u * sin_w + v * cos_w;
                u = next_u;
                v = next_v;
            }
        }

        // Land the mix ramp exactly on the published target (the next block
        // must start settled - the framework's smoothing convention).
        currentMix_ = targetMix;

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
     * @param hz Shift in Hz (negative shifts frequencies down). There is no
     * range clamp: carriers beyond Nyquist alias by design. Non-finite
     * values are ignored (a NaN here would poison the phase accumulator
     * permanently).
     */
    void setShift(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        shift_.store(hz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dry/wet mix. Smoothed internally.
     * @param mix Range [0.0, 1.0], clamped. 0.0 = pure dry (phase-aligned),
     * 1.0 = fully shifted. Non-finite values are ignored.
     */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
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
        // The blob stores float (setState reads float back); the explicit
        // casts also keep this overload resolvable when T is double.
        StateWriter w(stateId("FSHF"), 1);
        w.write("shift", static_cast<float>(shift_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
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

    // Smoothed state for the audio thread
    T currentMix_{ T(1) };

    std::vector<Hilbert<T>> hilberts_;
};

} // namespace dspark
