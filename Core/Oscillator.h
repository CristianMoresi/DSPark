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
     * @brief Enables band-limited hard sync.
     *
     * The oscillator's frequency becomes the sync MASTER; an internal slave
     * runs at `ratio` times that frequency and is phase-reset every master
     * cycle — the classic ripping sync timbre. The reset discontinuity is
     * corrected with a PolyBLEP scaled to the actual jump amplitude, using
     * the master phase as the band-limiting clock, so the result stays as
     * clean as the free-running waveforms.
     *
     * @param ratio Slave/master frequency ratio. Values <= 1 disable sync.
     */
    void setSyncRatio(T ratio) noexcept
    {
        syncRatio_ = std::max(T(0), ratio);
        updatePhaseInc();
    }

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

    /** @brief Hard-resets the oscillator phase and integrator state.
     *
     * The triangle integrator is seeded at the negative steady-state peak
     * (phase 0 drives the underlying square positive, so the steady cycle
     * starts at -peak). Seeding at zero made the first half-cycle overshoot
     * to (1+q)x the nominal level — an audible +4 dB pop on note retrigger. */
    void reset() noexcept
    {
        phase_ = T(0);
        triState_ = -triExpectedPeak_;
        slavePhase_ = T(0);
        pendingSyncJump_ = T(0);
        pendAmp0_ = pendAmp1_ = T(0);
        justSynced_ = false;
    }

    /**
     * @brief Computes and returns the next single audio sample.
     * @return A band-limited sample, nominally in the range [-1.0, 1.0].
     */
    [[nodiscard]] inline T getNextSample() noexcept
    {
        if (syncOn_)
            return nextSyncSample();

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
    /** @brief Raw (pre-BLEP) slave waveform value at a given phase. */
    [[nodiscard]] T rawSlaveValue(T ph) const noexcept
    {
        switch (waveform_)
        {
            case Waveform::Sine:     return fastSin(ph * twoPi<T>);
            case Waveform::Saw:      return T(2) * ph - T(1);
            case Waveform::Square:
            case Waveform::Triangle: return (ph < T(0.5)) ? T(1) : T(-1);
        }
        return T(0);
    }

    /** @brief Rising/after half of the 2-point BLEP kernel, t in [0, 1). */
    static inline T blepAfter(T t) noexcept { return t + t - t * t - T(1); }

    /**
     * @brief One sample of the hard-synced slave with full BLEP correction.
     *
     * Event bookkeeping: a sync reset both creates a jump of its own and can
     * abort an upcoming natural slave edge — while a natural edge that lands
     * just BEFORE the reset still needs its trailing kernel half on the next
     * sample, where the slave phase (re-seeded by the reset) no longer
     * encodes it. Those trailing halves are queued explicitly.
     */
    [[nodiscard]] T nextSyncSample() noexcept
    {
        const bool masterBefore = phase_ > T(1) - phaseInc_;
        const T uSync = masterBefore ? (T(1) - phase_) / phaseInc_ : T(2);
        const bool squareLike =
            waveform_ == Waveform::Square || waveform_ == Waveform::Triangle;

        // --- slave's own edges, gated against the reset --------------------
        T ownBlep = polyBlep(slavePhase_, slaveInc_);
        if (ownBlep != T(0))
        {
            const bool afterBranch = slavePhase_ < slaveInc_;
            if (afterBranch)
            {
                if (justSynced_) ownBlep = T(0);          // reset artifact
            }
            else if ((T(1) - slavePhase_) / slaveInc_ > uSync)
            {
                ownBlep = T(0);                            // aborted by reset
            }
        }

        T halfBlep = T(0);
        T half = slavePhase_ + T(0.5);
        if (half >= T(1)) half -= T(1);
        if (squareLike)
        {
            halfBlep = polyBlep(half, slaveInc_);
            if (halfBlep != T(0))
            {
                const bool afterBranch = half < slaveInc_;
                if (afterBranch)
                {
                    if (justSynced_) halfBlep = T(0);
                }
                else if ((T(1) - half) / slaveInc_ > uSync)
                {
                    halfBlep = T(0);
                }
            }
        }

        T raw = rawSlaveValue(slavePhase_);
        if (waveform_ == Waveform::Saw)
            raw -= ownBlep;
        else if (squareLike)
            raw += ownBlep - halfBlep;

        // Queued trailing halves of natural edges that preceded a reset.
        raw += pendAmp0_ * blepAfter(pendT0_);
        raw += pendAmp1_ * blepAfter(pendT1_);
        pendAmp0_ = pendAmp1_ = T(0);

        // --- the reset jump itself, clocked by the master phase -------------
        const T blep = polyBlep(phase_, phaseInc_);
        if (blep != T(0))
        {
            if (phase_ > T(0.5))   // before the wrap: predict the jump
            {
                T atJump = slavePhase_ + uSync * slaveInc_;
                atJump -= std::floor(atJump);
                pendingSyncJump_ = rawSlaveValue(atJump) - rawSlaveValue(T(0));
            }
            raw -= pendingSyncJump_ * T(0.5) * blep;
        }

        T out = raw;
        if (waveform_ == Waveform::Triangle)
        {
            triState_ = slaveInc_ * raw + (T(1) - slaveInc_) * triState_;
            out = triState_ * triNorm_;
        }

        // --- advance; a master wrap re-seeds the slave ----------------------
        justSynced_ = false;
        const T slaveOld = slavePhase_;
        phase_ += phaseInc_;
        slavePhase_ += slaveInc_;
        if (slavePhase_ >= T(1)) slavePhase_ -= T(1);
        if (phase_ >= T(1))
        {
            // Natural edges that genuinely happened before the reset lose
            // their trailing kernel half (the re-seeded phase forgets them):
            // queue it for the next sample.
            const T alphaSync = uSync;
            const T alphaWrap = (T(1) - slaveOld) / slaveInc_;
            if (alphaWrap <= alphaSync && alphaWrap <= T(1))
            {
                pendAmp0_ = (waveform_ == Waveform::Saw) ? T(-1) : (squareLike ? T(1) : T(0));
                pendT0_ = T(1) - alphaWrap;
            }
            if (squareLike)
            {
                const T halfOld = (slaveOld < T(0.5)) ? (T(0.5) - slaveOld)
                                                      : (T(1.5) - slaveOld);
                const T alphaHalf = halfOld / slaveInc_;
                if (alphaHalf <= alphaSync && alphaHalf <= T(1))
                {
                    pendAmp1_ = T(-1);
                    pendT1_ = T(1) - alphaHalf;
                }
            }

            phase_ -= T(1);
            slavePhase_ = (phase_ / std::max(phaseInc_, T(1e-12))) * slaveInc_;
            slavePhase_ -= std::floor(slavePhase_);
            justSynced_ = true;
        }
        return out;
    }

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
        syncOn_ = syncRatio_ > T(1.001) && phaseInc_ > T(0);
        // Clamp the slave below Nyquist like any oscillator frequency.
        slaveInc_ = std::min(phaseInc_ * syncRatio_, T(0.5));
        updateTriNorm();
    }

    /**
     * @brief Precomputes the normalization factor for the leaky integrator.
     */
    void updateTriNorm() noexcept
    {
        // Under hard sync the integrator runs at the slave rate.
        const T inc = syncOn_ ? slaveInc_ : phaseInc_;
        if (inc > T(0) && inc < T(1))
        {
            // Steady-state peak of the leaky integrator driven by a +-1 square:
            // over a half period of n samples starting at -p, the state reaches
            // p = q*(-p) + (1 - q) with q = leak^n, so p = (1 - q) / (1 + q).
            // (Using just 1 - q under-normalised the triangle by ~4 dB.)
            T leakCoeff = T(1) - inc;
            T halfPeriodSamples = T(0.5) / inc;
            T q = std::pow(leakCoeff, halfPeriodSamples);
            T expectedPeak = (T(1) - q) / (T(1) + q);
            triExpectedPeak_ = expectedPeak;
            triNorm_ = (expectedPeak > T(0.001)) ? T(1) / expectedPeak : T(4);
        }
        else
        {
            triExpectedPeak_ = T(0.25);
            triNorm_ = T(4);
        }
    }

    double   sampleRate_ = 48000.0;
    T        frequency_  = T(440);
    T        phase_      = T(0);
    T        phaseInc_   = T(0);
    T        triState_   = T(0);
    T        triNorm_    = T(4);
    T        triExpectedPeak_ = T(0.25); ///< Steady-state integrator peak (reset seed).
    Waveform waveform_   = Waveform::Sine;

    // Hard sync (slave) state.
    bool     syncOn_     = false;
    bool     justSynced_ = false;
    T        syncRatio_  = T(0);
    T        slavePhase_ = T(0);
    T        slaveInc_   = T(0);
    T        pendingSyncJump_ = T(0);
    T        pendAmp0_ = T(0), pendT0_ = T(0);   ///< Queued trailing edge halves.
    T        pendAmp1_ = T(0), pendT1_ = T(0);
};

} // namespace dspark