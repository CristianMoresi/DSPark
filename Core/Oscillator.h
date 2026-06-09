// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"

#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstddef>

namespace dspark {

/**
 * @class Oscillator
 * @brief Band-limited oscillator featuring PolyBLEP anti-aliasing and analog-modeled integration.
 *
 * This oscillator provides high-quality waveform generation suitable for both 
 * audio-rate synthesis and low-frequency modulation (LFO). It utilizes PolyBLEP
 * (Polynomial Band-Limited Step) to drastically reduce aliasing artifacts in 
 * discontinuous waveforms (Saw, Square). The Triangle wave is generated via a 
 * leaky integrator driven by a PolyBLEP square, providing an analog-style curve.
 *
 * @note To guarantee thread-safety in real-time contexts, parameter changes 
 * (e.g., setFrequency) should be applied before calling processBlock() or 
 * getNextSample(), ideally using a parameter smoothing queue.
 *
 * @tparam T Sample type (must be float or double).
 */
template <typename T>
class Oscillator
{
    static_assert(std::is_floating_point_v<T>, "Oscillator requires float or double");

public:
    enum class Waveform { Sine, Saw, Square, Triangle };

    /**
     * @brief Prepares the oscillator with the system sample rate.
     * @param sampleRate The operating sample rate in Hz. Must be > 0.
     */
    void prepare(double sampleRate) noexcept
    {
        assert(sampleRate > 0.0);
        sampleRate_ = sampleRate;
        updatePhaseInc();
    }

    /**
     * @brief Prepares the oscillator from an AudioSpec configuration.
     * @param spec The structural audio specification of the processing chain.
     */
    void prepare(const AudioSpec& spec) noexcept 
    { 
        prepare(spec.sampleRate); 
    }

    /**
     * @brief Sets the oscillator's fundamental frequency.
     * @param freq Frequency in Hz. Will be clamped between 0 and Nyquist.
     */
    void setFrequency(T freq) noexcept
    {
        // Clamp frequency to valid bounds [0, Nyquist] to prevent aliasing breakdown
        T nyquist = static_cast<T>(sampleRate_ * 0.5);
        frequency_ = std::clamp(freq, T(0), nyquist);
        updatePhaseInc();
    }

    /**
     * @brief Changes the active waveform.
     * @param w The desired waveform type.
     */
    void setWaveform(Waveform w) noexcept { waveform_ = w; }

    /**
     * @brief Forces the oscillator phase to a specific value.
     * @param phase Normalized phase in the range [0.0, 1.0).
     */
    void setPhase(T phase) noexcept
    {
        // Wrap (not clamp): a phase of exactly 1.0 must land on 0.0 so the
        // first sample is not a one-off discontinuity.
        phase_ = phase - std::floor(phase);
        if (phase_ >= T(1)) phase_ -= T(1);
    }

    /** @brief Hard-resets the oscillator phase and integrator state to zero. */
    void reset() noexcept 
    { 
        phase_ = T(0); 
        triState_ = T(0);
    }

    /**
     * @brief Computes and returns the next single audio sample.
     * @return A band-limited sample, nominally in the range [-1.0, 1.0].
     */
    [[nodiscard]] inline T getNextSample() noexcept
    {
        T out = T(0);

        switch (waveform_)
        {
            case Waveform::Sine:
                // fastSin: error < ~4e-6 in float (over 100 dB down), 3-6x
                // faster than std::sin — inaudible even for direct synthesis.
                out = fastSin(phase_ * twoPi<T>);
                break;

            case Waveform::Saw:
                out = T(2) * phase_ - T(1);
                out -= polyBlep(phase_, phaseInc_);
                break;

            case Waveform::Square:
            {
                T raw = (phase_ < T(0.5)) ? T(1) : T(-1);
                raw += polyBlep(phase_, phaseInc_);
                
                // Optimized phase shift without std::fmod
                T halfPhase = phase_ + T(0.5);
                if (halfPhase >= T(1)) halfPhase -= T(1);
                
                raw -= polyBlep(halfPhase, phaseInc_);
                out = raw;
                break;
            }

            case Waveform::Triangle:
            {
                T raw = (phase_ < T(0.5)) ? T(1) : T(-1);
                raw += polyBlep(phase_, phaseInc_);
                
                T halfPhase = phase_ + T(0.5);
                if (halfPhase >= T(1)) halfPhase -= T(1);
                raw -= polyBlep(halfPhase, phaseInc_);
                
                // Leaky integration for analog-style triangle
                triState_ = phaseInc_ * raw + (T(1) - phaseInc_) * triState_;
                out = triState_ * triNorm_;
                break;
            }
        }

        // Fast phase wrap assuming positive frequencies only (enforced in setFrequency)
        phase_ += phaseInc_;
        if (phase_ >= T(1)) phase_ -= T(1);

        return out;
    }

    /**
     * @brief Fills a buffer with generated samples. Optimized for SIMD vectorization.
     * @param buffer Pointer to the audio buffer. Must be 32-byte aligned if required by SIMD.
     * @param numSamples Number of samples to generate.
     */
    void processBlock(T* buffer, size_t numSamples) noexcept
    {
        for (size_t i = 0; i < numSamples; ++i)
        {
            buffer[i] = getNextSample();
        }
    }

    /** @brief Generator contract alias for getNextSample() (GeneratorProcessor). */
    [[nodiscard]] inline T getSample() noexcept { return getNextSample(); }

    /**
     * @brief Fills every channel of the view with the generated waveform.
     * Satisfies the GeneratorProcessor concept. The oscillator is mono, so all
     * channels receive the same signal.
     */
    void generateBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        for (int i = 0; i < nS; ++i)
        {
            const T s = getNextSample();
            for (int ch = 0; ch < nCh; ++ch)
                buffer.getChannel(ch)[i] = s;
        }
    }

    [[nodiscard]] T getPhase() const noexcept { return phase_; }
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

private:
    /**
     * @brief Computes the PolyBLEP residual.
     * @param phase The current oscillator phase [0, 1).
     * @param inc The phase increment per sample.
     * @return The polynomial correction value.
     */
    static inline T polyBlep(T phase, T inc) noexcept
    {
        if (inc < T(1e-10)) return T(0);

        if (phase < inc)
        {
            T t = phase / inc;
            return t + t - t * t - T(1);
        }
        else if (phase > T(1) - inc)
        {
            T t = (phase - T(1)) / inc;
            return t * t + t + t + T(1);
        }
        return T(0);
    }

    /** @brief Centralized method to update phase increment and normalization. */
    void updatePhaseInc() noexcept
    {
        phaseInc_ = frequency_ / static_cast<T>(sampleRate_);
        updateTriNorm();
    }

    /**
     * @brief Precomputes the normalization factor for the leaky integrator.
     */
    void updateTriNorm() noexcept
    {
        if (phaseInc_ > T(0) && phaseInc_ < T(1))
        {
            // Steady-state peak of the leaky integrator driven by a +-1 square:
            // over a half period of n samples starting at -p, the state reaches
            // p = q*(-p) + (1 - q) with q = leak^n, so p = (1 - q) / (1 + q).
            // (Using just 1 - q under-normalised the triangle by ~4 dB.)
            T leakCoeff = T(1) - phaseInc_;
            T halfPeriodSamples = T(0.5) / phaseInc_;
            T q = std::pow(leakCoeff, halfPeriodSamples);
            T expectedPeak = (T(1) - q) / (T(1) + q);
            triNorm_ = (expectedPeak > T(0.001)) ? T(1) / expectedPeak : T(4);
        }
        else
        {
            triNorm_ = T(4);
        }
    }

    double   sampleRate_ = 48000.0;
    T        frequency_  = T(440);
    T        phase_      = T(0);
    T        phaseInc_   = T(0);
    T        triState_   = T(0);
    T        triNorm_    = T(4);
    Waveform waveform_   = Waveform::Sine;
};

} // namespace dspark