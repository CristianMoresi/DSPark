// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Oscillator.h
 * @brief Band-limited oscillator: table-minBLEP waveforms (PolyBLEP optional).
 *
 * Header-only C++20 oscillator for audio-rate synthesis and LFO duty. Saw and
 * square discontinuities are corrected with the shared minimum-phase
 * band-limited step table (MinBlepTable) by default, or with a 2-point
 * PolyBLEP when setAntiAliasing() selects it (LFO duty, where the minBLEP's
 * band-limited overshoot is unwanted). The triangle is a leaky integration of
 * the band-limited square (analog-style curve). Hard sync always uses the
 * minBLEP, whose causal kernel survives the arbitrary jump amplitudes sync
 * creates.
 *
 * Features:
 * - Sine / Saw / Square / Triangle, band-limited. Worst inharmonic component
 *   below 20 kHz, saw and pulse from 110 Hz to 14 kHz at 44.1 and 48 kHz:
 *   minBLEP -143 dB or better (double; measured on exact-bin pitches),
 *   against PolyBLEP's -59 dB at 110 Hz falling to -23 dB at 7 kHz.
 * - Pulse-width modulation of the Square (setPulseWidth()): both edges of
 *   every cycle are band-limited, including the case where they fall inside
 *   one sample interval, and a width change that moves the falling edge
 *   across the current phase is itself corrected as a band-limited edge. The
 *   pulse is AC-coupled (zero mean at every width), like an analog
 *   oscillator's output stage.
 * - Hard sync with table-minBLEP correction: saw, square and triangle -139 dB
 *   or better (220 Hz .. 2.5 kHz masters, ratios 1.5 .. 4.3). The synced
 *   sine's slope change is corrected too (minimum-phase band-limited ramp):
 *   -160 dB at 220 Hz x 1.5, -111 dB at 880 Hz x 1.5, and -47 dB at
 *   2.5 kHz x 4.3, where its higher-order discontinuities remain.
 * - Phase, increments and edge times in double whatever T, and the triangle
 *   integrator too (framework rule for recursive state).
 * - Pitch can change every sample: setFrequency() is cheap for every
 *   waveform (the triangle's normalization follows its increment by a
 *   first-order update, re-solved exactly beyond 0.1% of change).
 * - Zero allocations after prepare(); the shared minBLEP table builds once there.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, MinBlepTable.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "MinBlepTable.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace dspark {

/**
 * @class Oscillator
 * @brief Band-limited oscillator featuring PolyBLEP anti-aliasing and analog-modeled integration.
 *
 * This oscillator provides high-quality waveform generation suitable for both
 * audio-rate synthesis and low-frequency modulation (LFO). Every discontinuity
 * of the Saw and Square waves is corrected with a table minBLEP (MinBlepTable)
 * -- a causal minimum-phase band-limited step whose alias rejection (-143 dB
 * or better, measured) is some 80-120 dB better than a 2-point PolyBLEP
 * across the keyboard. The Triangle wave is generated via a leaky integrator driven by
 * the band-limited square, providing an analog-style curve. Under hard sync
 * (setSyncRatio) the same kernel corrects the slave edges and the reset jumps.
 * A hard-synced Sine also changes slope at every reset (a kink); it is
 * rendered dcDelay() samples late and both its jump and its kink are
 * corrected against that delayed reference (MinBlepTable's delayed step and
 * minimum-phase band-limited ramp), so it is band-limited too.
 *
 * A band-limited step rings (Gibbs), and a minimum-phase one puts all of that
 * ringing after the edge: minBLEP saw and square peaks reach about 1.45x full
 * scale, so leave ~3.5 dB of headroom. For modulation duty, where that
 * overshoot is unwanted and aliasing is irrelevant, select
 * AntiAliasing::PolyBLEP (the waveforms then stay within [-1, 1]).
 *
 * @note This class is not internally thread-safe: apply parameter changes
 * (e.g. setFrequency) from the audio thread between process calls, publishing
 * values from other threads via atomics, as the framework effects do.
 *
 * @tparam T Sample type (must be float or double).
 */
template <typename T>
class Oscillator
{
    static_assert(std::is_floating_point_v<T>, "Oscillator requires float or double");

    /// Phase-domain arithmetic (phases, increments, edge times): double
    /// whatever T, the framework rule for recursive state. A float phase
    /// accumulator jitters every edge by its rounding and capped a float
    /// oscillator's alias floor near -100 dB.
    using Phase = double;

public:
    enum class Waveform { Sine, Saw, Square, Triangle };

    /** @brief Discontinuity correction used when hard sync is off. */
    enum class AntiAliasing
    {
        MinBLEP,  ///< Minimum-phase band-limited step table (default; audio duty).
        PolyBLEP  ///< 2-point polynomial step (no overshoot; LFO duty).
    };

    /**
     * @brief Prepares the oscillator with the system sample rate.
     *
     * Also re-clamps the stored frequency against the new Nyquist bound, so
     * calling prepare() again with a LOWER sample rate leaves the oscillator
     * in a valid state (the PolyBLEP correction assumes increment <= 0.5).
     *
     * @param sampleRate The operating sample rate in Hz. Must be > 0.
     */
    void prepare(double sampleRate) noexcept
    {
        assert(sampleRate > 0.0);
        sampleRate_ = sampleRate;
        // Touch the shared minBLEP table here so its one-time FFT build runs
        // on the control thread, never inside the audio callback.
        blepDelay_ = static_cast<Phase>(MinBlepTable<T>::instance().dcDelay());
        setFrequency(frequency_);
        if (!syncOn_) primeMinBlep();
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
        // Reject NaN before clamp: std::clamp(NaN, ...) returns NaN (all
        // comparisons false), which would poison phaseInc_ and the phase
        // accumulator forever. Matches Phasor/WavetableOscillator guards.
        if (freq != freq) return;
        // Clamp frequency to valid bounds [0, Nyquist] to prevent aliasing breakdown
        T nyquist = static_cast<T>(sampleRate_ * 0.5);
        frequency_ = std::clamp(freq, T(0), nyquist);
        updatePhaseInc();
    }

    /**
     * @brief Changes the active waveform.
     * @param w The desired waveform type.
     */
    void setWaveform(Waveform w) noexcept
    {
        if (w == waveform_) return;
        waveform_ = w;
        // The triangle's normalization is only kept current while it plays.
        if (w == Waveform::Triangle) updateTriNorm();
        // The pending minBLEP tails belong to the previous waveform's edges.
        if (!syncOn_) primeMinBlep();
    }

    /**
     * @brief Selects the discontinuity correction for the non-synced waveforms.
     *
     * MinBLEP (the default) is the audio-rate choice; its band-limited edges
     * ring to ~1.45x full scale. PolyBLEP keeps every waveform inside [-1, 1]
     * at the cost of far more aliasing: use it for LFOs and control signals. Pending minBLEP corrections are discarded on a
     * change. Hard sync always uses the minBLEP.
     *
     * @param mode Correction method.
     */
    void setAntiAliasing(AntiAliasing mode) noexcept
    {
        if (mode == antiAliasing_) return;
        antiAliasing_ = mode;
        if (!syncOn_) primeMinBlep();
    }

    /** @brief Returns the discontinuity correction in use (see setAntiAliasing()). */
    [[nodiscard]] AntiAliasing getAntiAliasing() const noexcept { return antiAliasing_; }

    /**
     * @brief Enables band-limited hard sync.
     *
     * The oscillator's frequency becomes the sync MASTER; an internal slave
     * runs at `ratio` times that frequency and is phase-reset every master
     * cycle -- the classic ripping sync timbre. Every discontinuity (the
     * slave's own edges and the reset jump, scaled to its actual amplitude)
     * is corrected with a table minBLEP (see MinBlepTable): a minimum-phase
     * band-limited step whose correction is fully causal, pushing aliasing
     * to the windowed-sinc stopband instead of the ~-40 dB envelope of a
     * 2-point polynomial kernel.
     *
     * @note A band-limited (brickwall) rendition of the synced waveform
     * legitimately overshoots the naive one: peaks may reach ~1.5x full
     * scale (Gibbs), unlike the sync-off waveforms which stay within
     * [-1, 1]. Leave headroom or follow with a gain stage.
     *
     * @param ratio Slave/master frequency ratio. Values <= 1 disable sync
     *              (a small guard band just above 1 also disables it -- a
     *              1:1 slave adds nothing but correction noise).
     */
    void setSyncRatio(T ratio) noexcept
    {
        syncRatio_ = std::max(T(0), ratio);
        updatePhaseInc();
    }

    /**
     * @brief Sets the Square's pulse width (duty cycle).
     *
     * The waveform is high for the first `width` of each cycle and low for the
     * rest; 0.5 (the default) is the symmetric square and leaves the output
     * unchanged from a waveform without pulse-width control. The pulse is
     * AC-coupled: its DC, 2 * width - 1, is removed, so its mean is zero at
     * every width and its peak rises toward 2 (minus the narrow side's share)
     * as the pulse narrows -- leave headroom for narrow pulses.
     *
     * Only the Square is affected; the Triangle stays the integral of the
     * symmetric square. Both edges are corrected with the minBLEP (or
     * PolyBLEP, see setAntiAliasing()). A change that moves the falling edge
     * across the current phase flips the output immediately, and on the
     * minBLEP paths that flip is band-limited like any other edge, so the
     * width can be modulated every sample.
     *
     * @param width Pulse width in [kMinPulseWidth, 1 - kMinPulseWidth];
     *              values outside are clamped and NaN is ignored.
     */
    void setPulseWidth(T width) noexcept
    {
        if (width != width) return;
        const T w = std::clamp(width, kMinPulseWidth, T(1) - kMinPulseWidth);
        if (w == pulseWidth_) return;
        const T old = pulseWidth_;
        pulseWidth_ = w;
        pulseDc_ = T(2) * w - T(1);
        if (waveform_ != Waveform::Square) return;

        // The falling edge moved across the current phase: the naive value
        // flips at this very sample. Queue that flip as a minBLEP edge whose
        // event time is the sample about to be emitted (age 0), so it is
        // band-limited instead of an instantaneous step.
        const Phase ph = syncOn_ ? slavePhase_ : phase_;
        const bool wasHigh = ph < old;
        const bool isHigh = ph < w;
        if (wasHigh != isHigh && (syncOn_ || antiAliasing_ == AntiAliasing::MinBLEP))
            scheduleMinBlep(isHigh ? T(2) : T(-2), T(0));
    }

    /** @brief Narrowest pulse width setPulseWidth() accepts (and 1 minus it the widest). */
    static constexpr T kMinPulseWidth = T(0.01);

    /**
     * @brief Forces the oscillator phase to a specific value.
     *
     * The triangle integrator is re-seeded at its steady-state value for the
     * requested phase, so a Triangle LFO lands on the expected waveform point
     * immediately instead of easing in over ~1/increment samples (the
     * stereo-spread re-phasing pattern used by Chorus relies on this). Under
     * hard sync the slave is re-phased coherently and pending minBLEP
     * corrections are discarded.
     *
     * @param phase Normalized phase in the range [0.0, 1.0).
     */
    void setPhase(T phase) noexcept
    {
        // Wrap (not clamp): a phase of exactly 1.0 must land on 0.0 so the
        // first sample is not a one-off discontinuity.
        phase_ = static_cast<Phase>(phase) - std::floor(static_cast<Phase>(phase));
        if (phase_ >= 1.0) phase_ -= 1.0;

        if (syncOn_)
        {
            reseedSlave();
        }
        else
        {
            // A forced phase is a new stream: rebuild the pending minBLEP
            // tails of the edges that stream would already have produced.
            primeMinBlep();
        }

        seedTriangle();
    }

    /** @brief Hard-resets the oscillator phase and integrator state.
     *
     * The triangle integrator is seeded at the negative steady-state peak
     * (phase 0 drives the underlying square positive, so the steady cycle
     * starts at -peak). Seeding at zero made the first half-cycle overshoot
     * to (1+q)x the nominal level -- an audible +4 dB pop on note retrigger. */
    void reset() noexcept
    {
        phase_ = 0.0;
        slavePhase_ = 0.0;
        corr_.fill(T(0));
        corrHead_ = 0;
        resetAge_ = { kNoReset, kNoReset };
        if (!syncOn_) primeMinBlep();
        seedTriangle();
    }

    /**
     * @brief Computes and returns the next single audio sample.
     * @return A band-limited sample. PolyBLEP output stays in [-1.0, 1.0];
     *         minBLEP edges (the default, and hard sync) ring to ~1.45x full
     *         scale -- Gibbs, see setAntiAliasing().
     */
    [[nodiscard]] inline T getNextSample() noexcept
    {
        if (syncOn_)
            return nextSyncSample();
        if (antiAliasing_ == AntiAliasing::MinBLEP && waveform_ != Waveform::Sine)
            return nextMinBlepSample();

        T out = T(0);

        switch (waveform_)
        {
            case Waveform::Sine:
                // fastSin: minimax error ~2e-7 in float / 4e-9 in double
                // (-135 / -168 dB), 3-6x faster than std::sin.
                out = static_cast<T>(fastSin(phase_ * twoPi<Phase>));
                break;

            case Waveform::Saw:
                out = static_cast<T>(2.0 * phase_ - 1.0 - polyBlep(phase_, phaseInc_));
                break;

            case Waveform::Square:
            {
                Phase raw = (phase_ < pulseWidth_) ? 1.0 : -1.0;
                raw += polyBlep(phase_, phaseInc_);

                // Phase relative to the falling edge at pulseWidth_ (optimized
                // shift without std::fmod; 0.5 for the symmetric square).
                Phase halfPhase = phase_ + (1.0 - static_cast<Phase>(pulseWidth_));
                if (halfPhase >= 1.0) halfPhase -= 1.0;

                raw -= polyBlep(halfPhase, phaseInc_);
                out = static_cast<T>(raw - static_cast<Phase>(pulseDc_));
                break;
            }

            case Waveform::Triangle:
            {
                Phase raw = (phase_ < 0.5) ? 1.0 : -1.0;
                raw += polyBlep(phase_, phaseInc_);

                Phase halfPhase = phase_ + 0.5;
                if (halfPhase >= 1.0) halfPhase -= 1.0;
                raw -= polyBlep(halfPhase, phaseInc_);

                // Leaky integration for analog-style triangle. The recursion
                // runs in double (framework rule for recursive state): with a
                // float accumulator, slow LFO rates quantise near the peaks.
                const double inc = static_cast<double>(phaseInc_);
                triState_ = inc * static_cast<double>(raw) + (1.0 - inc) * triState_;
                out = static_cast<T>(triState_) * triNorm_;
                break;
            }
        }

        // Fast phase wrap assuming positive frequencies only (enforced in setFrequency)
        phase_ += phaseInc_;
        if (phase_ >= 1.0) phase_ -= 1.0;

        return out;
    }

    /**
     * @brief Fills a buffer with generated samples.
     *
     * Plain per-sample loop over getNextSample(): waveform generation is
     * inherently serial (recursive phase/integrator state), so there is no
     * SIMD variant and no alignment requirement on @p buffer.
     *
     * @param buffer Destination pointer (any alignment).
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
     * channels receive the same signal (channel 0 is generated, the rest copy it).
     */
    void generateBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        if (nCh <= 0)
        {
            for (int i = 0; i < nS; ++i)
                (void) getNextSample();   // keep the phase advancing
            return;
        }
        T* ch0 = buffer.getChannel(0);
        for (int i = 0; i < nS; ++i)
            ch0[i] = getNextSample();
        for (int ch = 1; ch < nCh; ++ch)
            std::copy_n(ch0, static_cast<size_t>(nS), buffer.getChannel(ch));
    }

    [[nodiscard]] T getPhase() const noexcept { return static_cast<T>(phase_); }
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }
    [[nodiscard]] Waveform getWaveform() const noexcept { return waveform_; }
    [[nodiscard]] T getSyncRatio() const noexcept { return syncRatio_; }
    [[nodiscard]] T getPulseWidth() const noexcept { return pulseWidth_; }

private:
    /** @brief Raw (pre-BLEP) slave waveform value at a given phase. */
    [[nodiscard]] T rawSlaveValue(Phase ph) const noexcept
    {
        switch (waveform_)
        {
            case Waveform::Sine:     return static_cast<T>(fastSin(ph * twoPi<Phase>));
            case Waveform::Saw:      return static_cast<T>(2.0 * ph - 1.0);
            case Waveform::Square:   return (ph < pulseWidth_) ? T(1) : T(-1);
            case Waveform::Triangle: return (ph < 0.5) ? T(1) : T(-1);
        }
        return T(0);
    }

    /**
     * @brief Seeds the triangle integrator at its steady-state value.
     *
     * Piecewise-linear ideal at the current phase (the leaky integrator's
     * true cycle deviates by O(increment), so any remaining transient is
     * negligible). Phase 0 drives the underlying square positive, so the
     * triangle rises from -peak over [0, 0.5). The free-running minBLEP square
     * lags the naive one by MinBlepTable::dcDelay() samples, so its triangle
     * is seeded that much earlier in phase: seeding at the naive phase opened
     * a 5 kHz / 44.1 kHz triangle at 1.62x full scale.
     */
    void seedTriangle() noexcept
    {
        updateTriNorm();   // the seed needs the peak of the current increment
        Phase ph = syncOn_ ? slavePhase_ : phase_;
        if (!syncOn_ && antiAliasing_ == AntiAliasing::MinBLEP)
        {
            ph -= phaseInc_ * blepDelay_;
            ph -= std::floor(ph);
        }
        const Phase ideal = (ph < 0.5) ? (4.0 * ph - 1.0) : (3.0 - 4.0 * ph);
        triState_ = ideal * static_cast<Phase>(triExpectedPeak_);
    }

    /**
     * @brief Rebuilds the minBLEP ring as if the oscillator had been running.
     *
     * The saw's ramp is delayed by the kernel's dcDelay() and every edge's
     * band-limited tail lasts kCorrLen samples, so a stream that starts cold
     * (reset(), setPhase(), a waveform or mode change) would open with the
     * ramp compensation but none of the tails it pairs with: a saw at
     * 15 kHz / 44.1 kHz started at -2.47 instead of its steady-state -0.48.
     * Queueing the tails of the edges the free-running waveform produced
     * during the last kCorrLen samples makes the first sample the
     * steady-state continuation.
     */
    void primeMinBlep() noexcept
    {
        corr_.fill(T(0));
        corrHead_ = 0;
        if (antiAliasing_ != AntiAliasing::MinBLEP || waveform_ == Waveform::Sine
            || !(phaseInc_ > 0.0))
            return;

        const Phase period = 1.0 / phaseInc_;
        const auto primeEdge = [&](Phase edgePhase, T jump) noexcept {
            // Samples elapsed since the most recent edge at edgePhase, taken
            // relative to the sample about to be emitted.
            Phase age = (phase_ - edgePhase) * period;
            if (age < 0.0) age += period;
            for (; age < static_cast<Phase>(kCorrLen); age += period)
                scheduleMinBlep(jump, age);
        };
        if (waveform_ == Waveform::Saw)
        {
            primeEdge(0.0, T(-2));
        }
        else
        {
            primeEdge(0.0, T(2));
            primeEdge(fallingEdge(), T(-2));
        }
    }

    /** @brief Phase of the falling edge: the pulse width for the Square, the
     *  symmetric 0.5 for the Triangle's internal square. */
    [[nodiscard]] Phase fallingEdge() const noexcept
    {
        return waveform_ == Waveform::Square ? static_cast<Phase>(pulseWidth_) : 0.5;
    }

    /**
     * @brief One free-running Saw/Square/Triangle sample with table-minBLEP
     *        correction of every discontinuity.
     *
     * Same causal bookkeeping as the synced path: emit the raw value plus the
     * pending correction, advance the phase, and queue the band-limiting
     * residual of each edge crossed inside the interval (at most one: the
     * increment is clamped to 0.5).
     */
    [[nodiscard]] T nextMinBlepSample() noexcept
    {
        T raw = rawSlaveValue(phase_) + corr_[static_cast<size_t>(corrHead_)];
        corr_[static_cast<size_t>(corrHead_)] = T(0);
        corrHead_ = (corrHead_ + 1) & kCorrMask;   // now the NEXT sample's slot

        T out = raw;
        if (waveform_ == Waveform::Saw)
        {
            // Delay the ramp by the minBLEP's own low-frequency delay so the
            // band-limited jumps and the slope stay time-aligned (see
            // MinBlepTable::dcDelay): without this the saw carries a DC offset
            // of 2 * dcDelay * f0 / fs.
            out -= static_cast<T>(2.0 * phaseInc_ * blepDelay_);
        }
        else if (waveform_ == Waveform::Square)
        {
            out -= pulseDc_;   // AC-coupled pulse (zero at the 0.5 default)
        }
        else if (waveform_ == Waveform::Triangle)
        {
            // Leaky integration of the band-limited square (double state:
            // framework rule for recursive state).
            const double inc = static_cast<double>(phaseInc_);
            triState_ = inc * static_cast<double>(raw) + (1.0 - inc) * triState_;
            out = static_cast<T>(triState_) * triNorm_;
        }

        const Phase old = phase_;
        phase_ += phaseInc_;
        if (phaseInc_ > 0.0)
        {
            if (waveform_ == Waveform::Saw)
            {
                if (phase_ >= 1.0)
                {
                    // Wrap: the saw falls by 2.
                    const Phase alpha = (1.0 - old) / phaseInc_;
                    scheduleMinBlep(T(-2), 1.0 - alpha);
                }
            }
            else
            {
                // The square (or the triangle's drive) rises by 2 at the wrap
                // and falls by 2 at its falling edge. A narrow or wide pulse
                // can put both inside one interval -- the falling edge before
                // the wrap (width near 1) or after it (width near 0) -- so the
                // candidates are taken in time order. The symmetric square
                // (increment <= 0.5) never reaches a second one.
                const Phase edge = fallingEdge();
                if (old < edge && phase_ >= edge)
                {
                    const Phase alpha = (edge - old) / phaseInc_;
                    scheduleMinBlep(T(-2), 1.0 - alpha);
                }
                if (phase_ >= 1.0)
                {
                    const Phase alpha = (1.0 - old) / phaseInc_;
                    scheduleMinBlep(T(2), 1.0 - alpha);
                    if (phase_ >= 1.0 + edge)
                    {
                        const Phase alphaFall = (1.0 + edge - old) / phaseInc_;
                        scheduleMinBlep(T(-2), 1.0 - alphaFall);
                    }
                }
            }
        }
        if (phase_ >= 1.0) phase_ -= 1.0;
        return out;
    }

    /**
     * @brief One sample of the hard-synced slave with table-minBLEP correction.
     *
     * The minBLEP kernel is causal (minimum phase): every correction lands at
     * or after its discontinuity, queued into a small ring buffer. That makes
     * event handling purely chronological -- within one sample interval, a
     * natural slave edge happens only if it precedes the master reset, and the
     * reset jump is measured from the slave value at the exact reset instant
     * (which already includes any earlier edge). Nothing has to be predicted
     * before an event or un-queued after one, the failure modes that made the
     * 2-point kernel's bookkeeping delicate.
     */
    [[nodiscard]] T nextSyncSample() noexcept
    {
        const bool squareLike =
            waveform_ == Waveform::Square || waveform_ == Waveform::Triangle;

        // --- emit: raw value plus the pending band-limiting correction ------
        // The sine is evaluated dcDelay() samples late (see delayedSinePhase).
        const T naive = (waveform_ == Waveform::Sine)
            ? static_cast<T>(fastSin(delayedSinePhase() * twoPi<Phase>))
            : rawSlaveValue(slavePhase_);
        T raw = naive + corr_[static_cast<size_t>(corrHead_)];
        corr_[static_cast<size_t>(corrHead_)] = T(0);
        corrHead_ = (corrHead_ + 1) & kCorrMask;   // now the NEXT sample's slot
        resetAge_[0] += 1.0;
        resetAge_[1] += 1.0;

        T out = raw;
        if (waveform_ == Waveform::Saw)
        {
            // Ramp aligned with the minBLEP delay (see nextMinBlepSample()).
            out -= static_cast<T>(2.0 * slaveInc_ * blepDelay_);
        }
        else if (waveform_ == Waveform::Square)
        {
            out -= pulseDc_;   // the free-running pulse's DC (see setPulseWidth())
        }
        else if (waveform_ == Waveform::Triangle)
        {
            // Feed the integrator with the duty-compensated square: the synced
            // square has inherent DC (see updatePhaseInc) and the integrator's
            // unity DC gain times triNorm_ would turn it into a large offset
            // (+0.45 measured at ratio 2.7 without compensation).
            const double inc = static_cast<double>(slaveInc_);
            triState_ = inc * (static_cast<double>(raw) - static_cast<double>(syncSquareDc_))
                      + (1.0 - inc) * triState_;
            out = static_cast<T>(triState_) * triNorm_;
        }

        // --- advance both phases; schedule corrections in time order --------
        const Phase masterOld = phase_;
        const Phase slaveOld  = slavePhase_;
        phase_      += phaseInc_;
        slavePhase_ += slaveInc_;

        const bool masterWrap = phase_ >= 1.0;
        const Phase alphaSync = masterWrap ? (1.0 - masterOld) / phaseInc_ : 2.0;

        // Natural slave edge inside this interval (at most one: slaveInc_ is
        // clamped to 0.5). It only happens if the reset does not pre-empt it;
        // on an exact tie the edge wins and the reset jump measures zero.
        if (waveform_ != Waveform::Sine)
        {
            const auto naturalEdge = [&](Phase at, T jump) noexcept {
                const Phase alphaEdge = (at - slaveOld) / slaveInc_;
                if (alphaEdge <= alphaSync && alphaEdge <= 1.0)
                    scheduleMinBlep(jump, 1.0 - alphaEdge);
            };
            if (squareLike)
            {
                // Falling edge, wrap, and (narrow pulse) the next falling edge,
                // in time order; the symmetric square crosses at most one.
                const Phase edge = fallingEdge();
                if (slaveOld < edge && slavePhase_ >= edge)
                    naturalEdge(edge, T(-2));
                if (slavePhase_ >= 1.0)
                {
                    naturalEdge(1.0, T(2));
                    if (slavePhase_ >= 1.0 + edge)
                        naturalEdge(1.0 + edge, T(-2));
                }
            }
            else if (slavePhase_ >= 1.0)
            {
                naturalEdge(1.0, T(-2));   // the saw falls by 2 at its wrap
            }
        }
        if (slavePhase_ >= 1.0) slavePhase_ -= 1.0;

        if (masterWrap)
        {
            // Slave value the instant before the reset (wrap folded in); the
            // jump lands on the freshly seeded phase-0 value.
            Phase atJump = slaveOld + alphaSync * slaveInc_;
            atJump -= std::floor(atJump);
            const T jump = rawSlaveValue(0.0) - rawSlaveValue(atJump);
            if (waveform_ == Waveform::Sine)
            {
                // Jump and kink, against the delayed naive sine: the slope
                // per sample goes from 2 pi inc cos(2 pi atJump) to 2 pi inc.
                const auto& table = MinBlepTable<T>::instance();
                const Phase age = 1.0 - alphaSync;
                const auto slopeJump = static_cast<T>(
                    twoPi<Phase> * slaveInc_ * (1.0 - std::cos(twoPi<Phase> * atJump)));
                table.addDelayedStep(corr_.data(), corrHead_, jump, static_cast<T>(age));
                table.addDelayedRamp(corr_.data(), corrHead_, slopeJump, static_cast<T>(age));
                resetAge_[1] = resetAge_[0];
                prePhase_[1] = prePhase_[0];
                resetAge_[0] = age;
                prePhase_[0] = atJump;
            }
            else if (jump != T(0))
            {
                scheduleMinBlep(jump, 1.0 - alphaSync);
            }

            phase_ -= 1.0;
            slavePhase_ = (phase_ / std::max(phaseInc_, 1e-12)) * slaveInc_;
            slavePhase_ -= std::floor(slavePhase_);
        }
        return out;
    }

    /**
     * @brief Queues the minBLEP residual of one discontinuity into the ring.
     * @param jump Signed discontinuity amplitude (new value minus old value).
     * @param frac Time from the event to the next output sample, in samples:
     *             `1 - alpha` of an event inside the just-finished interval
     *             (in [0, 1)), or an older event's age when priming the ring.
     */
    void scheduleMinBlep(T jump, Phase frac) noexcept
    {
        MinBlepTable<T>::instance().addStep(corr_.data(), corrHead_, jump, static_cast<T>(frac));
    }

    /**
     * @brief Re-phases the sync slave coherently from the master phase and
     *        discards any pending minBLEP corrections.
     *
     * Called when sync (re-)engages or the phase is forced: leftover ring
     * corrections belong to a stream that no longer exists, and the slave
     * phase must match where it would be had sync been running since the
     * last master reset.
     */
    void reseedSlave() noexcept
    {
        const Phase r = slaveInc_ / std::max(phaseInc_, 1e-12);
        slavePhase_ = phase_ * r;
        slavePhase_ -= std::floor(slavePhase_);
        corr_.fill(T(0));
        corrHead_ = 0;
        resetAge_ = { kNoReset, kNoReset };
    }

    /**
     * @brief Phase of the synced sine dcDelay() samples ago.
     *
     * The minimum-phase ramp correction settles only against a waveform that
     * is that much late, so the synced sine is read on the segment the
     * delayed instant falls in: the current one, or one of the two before it
     * (resets are at least two samples apart: the master increment is at
     * most 0.5).
     */
    [[nodiscard]] Phase delayedSinePhase() const noexcept
    {
        const Phase d = blepDelay_;
        Phase ph;
        if (resetAge_[0] >= d)      ph = slavePhase_ - slaveInc_ * d;
        else if (resetAge_[1] >= d) ph = prePhase_[0] - slaveInc_ * (d - resetAge_[0]);
        else                        ph = prePhase_[1] - slaveInc_ * (d - resetAge_[1]);
        return ph - std::floor(ph);
    }

    /**
     * @brief Computes the PolyBLEP residual.
     * @param phase The current oscillator phase [0, 1).
     * @param inc The phase increment per sample. Must be <= 0.5 (enforced by
     *            updatePhaseInc) or the two correction branches would overlap.
     * @return The polynomial correction value.
     */
    static inline Phase polyBlep(Phase phase, Phase inc) noexcept
    {
        if (inc < 1e-10) return 0.0;

        if (phase < inc)
        {
            Phase t = phase / inc;
            return t + t - t * t - 1.0;
        }
        else if (phase > 1.0 - inc)
        {
            Phase t = (phase - 1.0) / inc;
            return t * t + t + t + 1.0;
        }
        return 0.0;
    }

    /** @brief Centralized method to update phase increment and normalization. */
    void updatePhaseInc() noexcept
    {
        // Defensive Nyquist clamp. setFrequency() already clamps, but prepare()
        // can lower the sample rate afterwards (it re-clamps too), and the
        // T-cast Nyquist bound may round up by an ulp; polyBlep() and the
        // sync event logic both assume inc <= 0.5.
        phaseInc_ = std::min(static_cast<Phase>(frequency_) / sampleRate_, 0.5);
        const bool wasOn = syncOn_;
        syncOn_ = syncRatio_ > T(1.001) && phaseInc_ > 0.0;
        // Clamp the slave below Nyquist like any oscillator frequency.
        slaveInc_ = std::min(phaseInc_ * static_cast<Phase>(syncRatio_), 0.5);

        // DC of the hard-synced square (feeds the triangle integrator). The
        // slave always restarts in its +1 half, so its duty cycle is
        // asymmetric by construction: for an effective ratio r with
        // fractional part f, the +1 time per master cycle exceeds the -1
        // time by min(f, 0.5) - max(f - 0.5, 0) slave cycles.
        syncSquareDc_ = 0.0;
        if (syncOn_ && phaseInc_ > 0.0)
        {
            const Phase r = slaveInc_ / phaseInc_;   // effective (clamped) ratio
            const Phase f = r - std::floor(r);
            syncSquareDc_ = (std::min(f, 0.5) - std::max(f - 0.5, 0.0)) / r;
        }

        // Sync just (re-)engaged: the ring may hold corrections from a
        // previous sync run and the slave phase is stale -- re-seed both.
        if (syncOn_ && !wasOn)
            reseedSlave();

        // Only the Triangle reads the normalization, and its std::pow would
        // otherwise run on every frequency change of every waveform: a voice
        // modulating its pitch per sample calls setFrequency() per sample.
        // setWaveform() and seedTriangle() refresh it when it is next needed,
        // from the same pure function of the increment.
        if (waveform_ == Waveform::Triangle) updateTriNorm();
    }

    /**
     * @brief Precomputes the normalization factor for the leaky integrator.
     */
    void updateTriNorm() noexcept
    {
        // Under hard sync the integrator runs at the slave rate.
        const Phase inc = syncOn_ ? slaveInc_ : phaseInc_;
        if (inc == triNormInc_)
            return;   // unchanged increment
        triNormInc_ = inc;
        if (inc > 0.0 && inc < 1.0)
        {
            // Steady-state peak of the leaky integrator driven by a +-1 square:
            // over a half period of n samples starting at -p, the state reaches
            // p = q*(-p) + (1 - q) with q = leak^n, so p = (1 - q) / (1 + q).
            // (Using just 1 - q under-normalised the triangle by ~4 dB.)
            //
            // A pitch modulated every sample (vibrato, glide, a synthesizer
            // voice) moves the increment every sample, and the exact form
            // costs a std::pow each time. Within 0.1% of the increment the
            // peak was last solved at, the first-order expansion around that
            // anchor is used instead: its error is about 1e-7 of the peak.
            Phase expectedPeak;
            if (std::abs(inc - triAnchorInc_) <= 1e-3 * triAnchorInc_)
            {
                expectedPeak = triAnchorPeak_ + triAnchorSlope_ * (inc - triAnchorInc_);
            }
            else
            {
                const Phase lnQ = 0.5 * std::log1p(-inc) / inc;
                const Phase q = std::exp(lnQ);
                expectedPeak = (1.0 - q) / (1.0 + q);
                // dp/dinc = dp/dq * q * dlnQ/dinc
                const Phase dLnQ = 0.5 * (-inc / (1.0 - inc) - std::log1p(-inc)) / (inc * inc);
                triAnchorInc_ = inc;
                triAnchorPeak_ = expectedPeak;
                triAnchorSlope_ = -2.0 / ((1.0 + q) * (1.0 + q)) * q * dLnQ;
            }
            triExpectedPeak_ = static_cast<T>(expectedPeak);
            triNorm_ = (expectedPeak > 0.001) ? static_cast<T>(1.0 / expectedPeak) : T(4);
        }
        else
        {
            triExpectedPeak_ = T(0.25);
            triNorm_ = T(4);
        }
    }

    double   sampleRate_ = 48000.0;
    T        frequency_  = T(440);
    Phase    phase_      = 0.0;
    Phase    phaseInc_   = 0.0;
    double   triState_   = 0.0;      ///< Leaky-integrator state (double: recursive state rule).
    T        triNorm_    = T(4);
    T        triExpectedPeak_ = T(0.25); ///< Steady-state integrator peak (reset seed).
    Phase    triNormInc_ = -1.0;     ///< Increment the tri norm was computed for (cache key).
    Phase    triAnchorInc_ = -1.0;   ///< Increment the peak was last solved exactly at.
    Phase    triAnchorPeak_ = 0.25;  ///< That exact peak.
    Phase    triAnchorSlope_ = 0.0;  ///< Its derivative with respect to the increment.
    Waveform waveform_   = Waveform::Sine;
    AntiAliasing antiAliasing_ = AntiAliasing::MinBLEP;
    Phase    blepDelay_  = 0.0;          ///< MinBlepTable::dcDelay(), cached by prepare().

    // Hard sync (slave) and minBLEP correction state.
    static constexpr int kCorrLen  = MinBlepTable<T>::kTaps;
    static constexpr int kCorrMask = kCorrLen - 1;
    static_assert((kCorrLen & kCorrMask) == 0, "minBLEP ring needs a power-of-two span");

    bool     syncOn_     = false;
    T        syncRatio_  = T(0);
    Phase    slavePhase_ = 0.0;
    Phase    slaveInc_   = 0.0;
    Phase    syncSquareDc_ = 0.0;      ///< Analytic duty-cycle DC of the synced square.
    static constexpr Phase kNoReset = 1e9;
    std::array<Phase, 2> resetAge_{ kNoReset, kNoReset }; ///< Synced sine: samples since the last two resets.
    std::array<Phase, 2> prePhase_{};  ///< Synced sine: slave phase just before those resets.
    T        pulseWidth_ = T(0.5);     ///< Square duty cycle (setPulseWidth()).
    T        pulseDc_    = T(0);       ///< 2 * pulseWidth_ - 1, removed from the Square.
    std::array<T, kCorrLen> corr_{};   ///< Pending causal minBLEP corrections.
    int      corrHead_   = 0;          ///< Ring slot of the next output sample.
};

} // namespace dspark
