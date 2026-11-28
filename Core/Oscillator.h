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
 *   integrator too (framework rule for recursive state). The phase and its
 *   increment are kept on a grid of 2^-52 of a cycle (1.1e-11 Hz at 48 kHz),
 *   on which the accumulator is exact: it carries no rounding error however
 *   long it runs.
 * - processBlock() writes the stretches between discontinuities in closed
 *   form with vector kernels and returns what getNextSample() returns, bit
 *   for bit.
 * - Pitch can change every sample: setFrequency() is cheap for every
 *   waveform (the triangle's normalization follows its increment by a
 *   first-order update, re-solved exactly beyond 0.1% of change).
 * - Zero allocations after prepare(); the shared minBLEP table builds once there.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, MinBlepTable.h
 * (and through it SimdOps.h and detail/BlepKernels.inl).
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
#include <cstdint>
#include <type_traits>

// Keeps the per-sample loop of processBlock() out of line, so that
// getNextSample() has one call site in this header and a caller's own loop
// over it is inlined as before. Locally defined and guarded, the same way
// Core/Biquad.h spells it.
#ifndef DSPARK_NOINLINE
  #if defined(_MSC_VER)
    #define DSPARK_NOINLINE __declspec(noinline)
  #elif defined(__GNUC__) || defined(__clang__)
    #define DSPARK_NOINLINE __attribute__((noinline))
  #else
    #define DSPARK_NOINLINE
  #endif
#endif

// The two-line accessors of the correction store, always inside their callers.
#if defined(_MSC_VER)
  #define DSPARK_OSC_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
  #define DSPARK_OSC_INLINE inline __attribute__((always_inline))
#else
  #define DSPARK_OSC_INLINE inline
#endif

// MSVC decides about getNextSample() by its size and leaves polyBlep() as a
// call (two per sample of a PolyBLEP pulse or triangle). There both are
// forced into their callers, and the minBLEP path stays a function of its
// own so that what is inlined into a caller's loop remains small. The other
// compilers already inline all three.
#if defined(_MSC_VER) && !defined(__clang__)
  #define DSPARK_OSC_OUTLINE DSPARK_NOINLINE
  #define DSPARK_OSC_MSVC_INLINE __forceinline
#else
  #define DSPARK_OSC_OUTLINE
  #define DSPARK_OSC_MSVC_INLINE inline
#endif

// Marks a value as rounded here: the compiler may not fuse the product that
// made it with a later sum (see Oscillator::integrateTriangle()). An empty
// statement on the value's register where the compiler takes one, a pass
// through memory on other GCC targets, nothing where products and sums are
// only fused inside one expression.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER) \
    && (defined(__x86_64__) || defined(__i386__)) && defined(__SSE2__)
  #define DSPARK_OSC_ROUNDED(value) __asm__ volatile("" : "+x"(value))
#elif (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER) && defined(__aarch64__)
  #define DSPARK_OSC_ROUNDED(value) __asm__ volatile("" : "+w"(value))
#elif defined(__GNUC__) && !defined(__clang__)
  #define DSPARK_OSC_ROUNDED(value) do { volatile double held = (value); (value) = held; } while (false)
#else
  #define DSPARK_OSC_ROUNDED(value) static_cast<void>(value)
#endif

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
        if (!(sampleRate > 0.0) || !std::isfinite(sampleRate)) return;
        sampleRate_ = sampleRate;
        invSampleRate_ = 1.0 / sampleRate;
        nyquist_ = static_cast<T>(sampleRate * 0.5);
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
        frequency_ = std::clamp(freq, T(0), nyquist_);
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
     * @param phase Normalized phase in the range [0.0, 1.0); a non-finite
     *              value is ignored.
     */
    void setPhase(T phase) noexcept
    {
        if (!std::isfinite(phase)) return;
        // Wrap (not clamp): a phase of exactly 1.0 must land on 0.0 so the
        // first sample is not a one-off discontinuity. The phase is kept on
        // the phase grid (see toGrid()).
        phase_ = toGrid(static_cast<Phase>(phase) - std::floor(static_cast<Phase>(phase)));
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
        clearCorrections();
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
    [[nodiscard]] DSPARK_OSC_MSVC_INLINE T getNextSample() noexcept
    {
        // The branches taken once or twice a cycle (a correction, a wrap) are
        // marked unlikely here and in polyBlep(): the compiler then lays the
        // run of every other sample out straight, wherever this is inlined.
        if (syncOn_) [[unlikely]]
            return nextSyncSample();
        if (antiAliasing_ == AntiAliasing::MinBLEP && waveform_ != Waveform::Sine)
            return nextMinBlepSample();

        // The phase is advanced and stored first, and the waveform is worked
        // out from the value it had: the next sample's read of the phase then
        // never waits behind this sample's waveform. (Fast wrap: positive
        // frequencies only, enforced in setFrequency().)
        const Phase ph = phase_;
        const Phase inc = phaseInc_;
        Phase next = ph + inc;
        if (next >= 1.0) [[unlikely]] next -= 1.0;
        phase_ = next;

        switch (waveform_)
        {
            case Waveform::Sine:
                // fastSin: minimax error ~2e-7 in float / 4e-9 in double
                // (-135 / -168 dB), 3-6x faster than std::sin.
                return static_cast<T>(fastSin(ph * twoPi<Phase>));

            case Waveform::Saw:
                return static_cast<T>(2.0 * ph - 1.0 - polyBlep(ph, inc));

            case Waveform::Square:
            {
                Phase raw = (ph < pulseWidth_) ? 1.0 : -1.0;
                raw += polyBlep(ph, inc);

                // Phase relative to the falling edge at pulseWidth_ (optimized
                // shift without std::fmod; 0.5 for the symmetric square).
                Phase halfPhase = ph + (1.0 - static_cast<Phase>(pulseWidth_));
                if (halfPhase >= 1.0) halfPhase -= 1.0;

                raw -= polyBlep(halfPhase, inc);
                return static_cast<T>(raw - static_cast<Phase>(pulseDc_));
            }

            case Waveform::Triangle:
            {
                Phase raw = (ph < 0.5) ? 1.0 : -1.0;
                raw += polyBlep(ph, inc);

                Phase halfPhase = ph + 0.5;
                if (halfPhase >= 1.0) halfPhase -= 1.0;
                raw -= polyBlep(halfPhase, inc);

                // Leaky integration for analog-style triangle. The recursion
                // runs in double (framework rule for recursive state): with a
                // float accumulator, slow LFO rates quantise near the peaks.
                triState_ = integrateTriangle(triState_, inc, raw);
                return static_cast<T>(triState_) * triNorm_;
            }
        }
        return T(0);
    }

    /**
     * @brief Fills a buffer with generated samples.
     *
     * Returns exactly what the same number of getNextSample() calls would,
     * bit for bit, and leaves the oscillator in the same state, so the two
     * can be mixed freely. Between two discontinuities the waveform is a
     * closed form of the phase: those stretches are written by vector
     * kernels (the ramp, the pending minBLEP corrections and the pulse
     * level, several samples at a time), and only the sample whose interval
     * holds a discontinuity goes through the per-sample path. Hard sync, the
     * sine and pitches whose stretches are a few samples long (above a
     * sixteenth of the sample rate, an eighth for the saw) are rendered
     * sample by sample: there the closed form saves nothing.
     *
     * @param buffer Destination pointer (any alignment).
     * @param numSamples Number of samples to generate.
     */
    void processBlock(T* buffer, size_t numSamples) noexcept
    {
        const Phase longest = waveform_ == Waveform::Saw ? kBlockSawInc : kBlockPulseInc;
        if (syncOn_ || waveform_ == Waveform::Sine || phaseInc_ > longest)
        {
            renderSamples(buffer, numSamples);
            return;
        }
        size_t i = 0;
        while (i < numSamples)
        {
            i += renderQuiet(buffer + i, numSamples - i);
            if (i < numSamples)
            {
                renderSamples(buffer + i, 1);
                ++i;
            }
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
        processBlock(ch0, static_cast<size_t>(std::max(nS, 0)));
        for (int ch = 1; ch < nCh; ++ch)
            std::copy_n(ch0, static_cast<size_t>(nS), buffer.getChannel(ch));
    }

    [[nodiscard]] T getPhase() const noexcept { return static_cast<T>(phase_); }
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }
    [[nodiscard]] Waveform getWaveform() const noexcept { return waveform_; }
    [[nodiscard]] T getSyncRatio() const noexcept { return syncRatio_; }
    [[nodiscard]] T getPulseWidth() const noexcept { return pulseWidth_; }

private:
    /// Largest increments processBlock() renders in closed form: beyond them
    /// a stretch between discontinuities is too short to gain from it.
    static constexpr Phase kBlockSawInc = 1.0 / 8.0;
    static constexpr Phase kBlockPulseInc = 1.0 / 16.0;

    /** @brief The per-sample loop of processBlock(), out of line. */
    DSPARK_NOINLINE void renderSamples(T* out, size_t n) noexcept
    {
        for (size_t i = 0; i < n; ++i)
            out[i] = getNextSample();
    }

    /**
     * @brief One step of the triangle's leaky integrator: the state decays
     *        by 1 - inc and takes inc times the drive.
     *
     * The drive term is rounded on its own before the sum, so the only
     * product a compiler can fuse with the addition is the decay's: the same
     * one in getNextSample() and in processBlock(), which therefore agree
     * bit for bit on targets with a fused multiply-add too. (Left to itself
     * a compiler fuses whichever product it meets first, and that changes
     * with the code around the expression.)
     */
    [[nodiscard]] static DSPARK_OSC_INLINE double integrateTriangle(double state, double inc, double drive) noexcept
    {
        double term = inc * drive;
        DSPARK_OSC_ROUNDED(term);
        return (1.0 - inc) * state + term;
    }

    /**
     * @brief Rounds a phase or an increment in [0, 1] to the phase grid.
     *
     * The grid is 2^-52 of a cycle, the spacing of the doubles in [1, 2).
     * Every multiple of it below 2 is a double, so with the phase and its
     * increment on the grid each `phase += increment` and each wrap is
     * exact: the accumulator carries no rounding error at all, and the phase
     * n samples ahead is exactly `phase + n * increment` while it stays
     * below the wrap. processBlock() relies on that to write a stretch in
     * closed form and still match getNextSample() bit for bit. The grid is
     * 1.1e-11 Hz wide at 48 kHz.
     */
    [[nodiscard]] static Phase toGrid(Phase x) noexcept
    {
        constexpr Phase kGrid = 4503599627370496.0;   // 2^52
        return static_cast<Phase>(static_cast<std::int64_t>(x * kGrid + 0.5)) * (1.0 / kGrid);
    }

    /**
     * @brief How many of the next samples (at most @p cap) advance without
     *        reaching @p limit: the count of j >= 0 with
     *        ph + (j + 1) * inc < limit.
     *
     * Exact: ph and inc lie on the phase grid, so every sum compared here is
     * the phase the per-sample recursion reaches.
     */
    [[nodiscard]] static size_t quietAdvances(Phase ph, Phase inc, Phase limit, size_t cap) noexcept
    {
        if (cap == 0) return 0;
        if (!(inc > 0.0)) return ph < limit ? cap : 0;
        // The usual case of a short block: all of it fits.
        if (ph + static_cast<Phase>(cap) * inc < limit) return cap;
        if (!(ph + inc < limit)) return 0;
        const Phase estimate = (limit - ph) / inc;
        size_t k = estimate >= static_cast<Phase>(cap) ? cap : static_cast<size_t>(estimate);
        if (k < 1) k = 1;
        while (k > 1 && !(ph + static_cast<Phase>(k) * inc < limit)) --k;
        while (k < cap && ph + static_cast<Phase>(k + 1) * inc < limit) ++k;
        return k;
    }

    /// Longest stretch renderQuiet() writes in one call.
    static constexpr size_t kQuietCap = 4096;
    /// Guard band around the PolyBLEP corrections' ends, in cycles: wider
    /// than any rounding of the phases they test, far below an increment.
    static constexpr Phase kPolyGuard = 1e-9;

    /**
     * @brief Writes the samples ahead whose intervals hold no discontinuity
     *        (and, for PolyBLEP, no correction), in closed form.
     *
     * Each of them is what getNextSample() returns for it, bit for bit, and
     * the state is left as that many calls leave it. Stops before the first
     * sample that needs the per-sample path; never called under hard sync.
     *
     * @return Samples written (0 when the next one is not such a sample).
     */
    [[nodiscard]] size_t renderQuiet(T* out, size_t n) noexcept
    {
        if (n == 0) return 0;
        n = std::min(n, kQuietCap);
        const Phase ph = phase_;
        const Phase inc = phaseInc_;
        const bool minBlep = antiAliasing_ == AntiAliasing::MinBLEP;
        size_t q = 0;

        // Renders the next `count` samples in two parts: while corrections
        // are pending, `withStore(offset, slots, length)` over their slots,
        // in pieces that end where the store is rebased or emptied; then
        // `without(offset, length)` for the rest, which have none to add.
        const auto overStore = [this](size_t count, auto&& withStore, auto&& without) noexcept {
            size_t done = 0;
            while (done < count && corrEnd_ != 0)
            {
                const int span = std::min(corrEnd_, kCorrLen) - corrHead_;
                const auto len = static_cast<int>(std::min<size_t>(count - done, static_cast<size_t>(span)));
                withStore(done, corr_.data() + corrHead_, len);
                done += static_cast<size_t>(len);
                advanceCorrections(len);
            }
            if (done < count) without(done, count - done);
        };
        // The stretch of a PolyBLEP square whose corrections are all zero:
        // phases at least one increment (and the guard) from both edges.
        const auto polyZone = [&](Phase edge, Phase& level) noexcept -> size_t {
            const Phase margin = inc + kPolyGuard;
            Phase limit;
            if (ph >= margin && ph < edge - margin)            { limit = edge - margin; level = 1.0; }
            else if (ph >= edge + margin && ph < 1.0 - margin) { limit = 1.0 - margin;  level = -1.0; }
            else return 0;
            return 1 + quietAdvances(ph, inc, limit, n - 1);
        };

        switch (waveform_)
        {
            case Waveform::Sine:
                return 0;   // processBlock() renders the sine sample by sample

            case Waveform::Saw:
                if (minBlep)
                {
                    q = quietAdvances(ph, inc, 1.0, n);
                    const T offset = sawDelay_;
                    overStore(q,
                        [&](size_t at, T* ring, int len) noexcept {
                            const Phase start = ph + static_cast<Phase>(at) * inc;
                            detail::rampOverBlepRing(out + at, ring, len, 2.0 * start - 1.0, 2.0 * inc, offset);
                        },
                        [&](size_t at, size_t len) noexcept {
                            const Phase start = ph + static_cast<Phase>(at) * inc;
                            detail::rampMinusOffset(out + at, static_cast<int>(len), 2.0 * start - 1.0, 2.0 * inc, offset);
                        });
                }
                else
                {
                    // Past the rising correction (phase < increment); the
                    // falling one begins where the advance would wrap.
                    if (!(inc < 1e-10) && ph < inc) return 0;
                    q = quietAdvances(ph, inc, 1.0, n);
                    detail::rampBlock(out, static_cast<int>(q), 2.0 * ph - 1.0, 2.0 * inc);
                }
                break;

            case Waveform::Square:
                if (minBlep)
                {
                    const Phase edge = fallingEdge();
                    const bool high = ph < edge;
                    const T level = high ? T(1) : T(-1);
                    q = quietAdvances(ph, inc, high ? edge : 1.0, n);
                    overStore(q,
                        [&](size_t at, T* ring, int len) noexcept {
                            detail::levelOverBlepRing(out + at, ring, len, level, pulseDc_);
                        },
                        [&](size_t at, size_t len) noexcept { std::fill_n(out + at, len, level - pulseDc_); });
                }
                else
                {
                    Phase level = 0.0;
                    q = polyZone(fallingEdge(), level);
                    std::fill_n(out, q, static_cast<T>(level - static_cast<Phase>(pulseDc_)));
                }
                break;

            case Waveform::Triangle:
            {
                double tri = triState_;
                const T norm = triNorm_;
                if (minBlep)
                {
                    const bool high = ph < 0.5;
                    const T level = high ? T(1) : T(-1);
                    q = quietAdvances(ph, inc, high ? 0.5 : 1.0, n);
                    overStore(q,
                        [&](size_t at, T* ring, int len) noexcept {
                            for (int k = 0; k < len; ++k)
                            {
                                const T raw = level + ring[k];
                                ring[k] = T(0);
                                tri = integrateTriangle(tri, inc, static_cast<double>(raw));
                                out[at + static_cast<size_t>(k)] = static_cast<T>(tri) * norm;
                            }
                        },
                        [&](size_t at, size_t len) noexcept {
                            for (size_t k = 0; k < len; ++k)
                            {
                                tri = integrateTriangle(tri, inc, static_cast<double>(level));
                                out[at + k] = static_cast<T>(tri) * norm;
                            }
                        });
                }
                else
                {
                    Phase level = 0.0;
                    q = polyZone(0.5, level);
                    for (size_t j = 0; j < q; ++j)
                    {
                        tri = integrateTriangle(tri, inc, level);
                        out[j] = static_cast<T>(tri) * norm;
                    }
                }
                triState_ = tri;
                break;
            }
        }
        phase_ = ph + static_cast<Phase>(q) * inc;
        return q;
    }

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
     * Phase 0 drives the underlying square positive, so the triangle rises
     * from -peak over [0, 0.5).
     *
     * PolyBLEP (the LFO duty): the free-running triangle is a leaky
     * integrator of the +-1 square (leak = increment), so its steady cycle is
     * two exponential arcs, not straight lines: over [0, 0.5) the state is
     * 1 - (1 + peak) * (1 - inc)^(phase / inc), mirrored over [0.5, 1), and
     * the per-sample recursion runs half a sample behind the phase. The seed
     * is that value, so a re-phased oscillator continues the free-running
     * waveform: re-phasing every sample from getPhase() stays within 1.1e-4
     * of it at 0.5 to 30 Hz and 3.7e-3 at 3 kHz / 48 kHz. (The
     * piecewise-linear seed sat 0.12 of full scale off it mid-slope and eased
     * back over about 1/increment samples: two seconds of modulation error at
     * a 0.5 Hz rate, and a jump wherever an owner re-phases while running.)
     *
     * MinBLEP: the free-running minBLEP square lags the naive one by
     * MinBlepTable::dcDelay() samples, so its triangle is seeded that much
     * earlier in phase on the piecewise-linear shape: seeding at the naive
     * phase opened a 5 kHz / 44.1 kHz triangle at 1.62x full scale. Under hard
     * sync the slave restarts every master cycle and the same linear seed is
     * used.
     */
    void seedTriangle() noexcept
    {
        updateTriNorm();   // the seed needs the peak of the current increment
        Phase ph = syncOn_ ? slavePhase_ : phase_;
        const Phase peak = static_cast<Phase>(triExpectedPeak_);
        if (!syncOn_ && antiAliasing_ == AntiAliasing::PolyBLEP && phaseInc_ > 0.0 && phaseInc_ < 1.0)
        {
            ph -= 0.5 * phaseInc_;
            ph -= std::floor(ph);
            // Offsets from -peak and +peak keep the ends of each arc exact.
            const Phase decay = std::log1p(-phaseInc_) / phaseInc_;   // ln of the leak per unit phase
            triState_ = (ph < 0.5) ? -peak - (1.0 + peak) * std::expm1(ph * decay)
                                   :  peak + (1.0 + peak) * std::expm1((ph - 0.5) * decay);
            return;
        }
        if (!syncOn_ && antiAliasing_ == AntiAliasing::MinBLEP)
        {
            ph -= phaseInc_ * blepDelay_;
            ph -= std::floor(ph);
        }
        const Phase ideal = (ph < 0.5) ? (4.0 * ph - 1.0) : (3.0 - 4.0 * ph);
        triState_ = ideal * peak;
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
        clearCorrections();
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
    [[nodiscard]] DSPARK_OSC_OUTLINE T nextMinBlepSample() noexcept
    {
        // The waveform is decided once, and each one is a straight run:
        // this is the path of every sample. The interval's discontinuities
        // are handled out of line (an edge comes once or twice a cycle; an
        // increment of zero crosses nothing, the phase does not move).
        const T pending = takeCorrection();
        const Phase old = phase_;
        const Phase next = old + phaseInc_;
        phase_ = next;
        switch (waveform_)
        {
            case Waveform::Saw:
            {
                // The ramp is delayed by the minBLEP's own low-frequency
                // delay so the band-limited jumps and the slope stay
                // time-aligned (see MinBlepTable::dcDelay): without this the
                // saw carries a DC offset of 2 * dcDelay * f0 / fs. The term
                // is kept ready (sawDelay_), one value for this path and for
                // renderQuiet().
                const T out = (static_cast<T>(2.0 * old - 1.0) + pending) - sawDelay_;
                if (next >= 1.0) [[unlikely]] crossEdges(old);
                return out;
            }
            case Waveform::Square:
            {
                const Phase edge = static_cast<Phase>(pulseWidth_);
                const bool high = old < edge;
                // AC-coupled pulse (pulseDc_ is zero at the 0.5 default).
                const T out = ((high ? T(1) : T(-1)) + pending) - pulseDc_;
                if ((high && next >= edge) || next >= 1.0) [[unlikely]] crossEdges(old);
                return out;
            }
            case Waveform::Triangle:
            {
                // Leaky integration of the band-limited square (double state:
                // framework rule for recursive state).
                const bool high = old < 0.5;
                const T raw = (high ? T(1) : T(-1)) + pending;
                triState_ = integrateTriangle(triState_, static_cast<double>(phaseInc_), static_cast<double>(raw));
                const T out = static_cast<T>(triState_) * triNorm_;
                if ((high && next >= 0.5) || next >= 1.0) [[unlikely]] crossEdges(old);
                return out;
            }
            case Waveform::Sine:
                break;   // the sine has no discontinuity: getNextSample() never sends it here
        }
        return T(0);
    }

    /**
     * @brief Queues the corrections of the discontinuities inside the
     *        interval just advanced over (from @p old to the phase), and
     *        wraps the phase.
     *
     * The time from an edge to the next sample is the phase past it over
     * the increment. The saw falls by 2 at its wrap. The square (or the
     * triangle's drive) rises by 2 at the wrap and falls by 2 at its falling
     * edge; a narrow or wide pulse can put both inside one interval -- the
     * falling edge before the wrap (width near 1) or after it (width near 0)
     * -- so the candidates are taken in time order. The symmetric square
     * (increment <= 0.5) never reaches a second one.
     */
    DSPARK_NOINLINE void crossEdges(Phase old) noexcept
    {
        const Phase perCycle = samplesPerCycle();
        if (waveform_ == Waveform::Saw)
        {
            scheduleMinBlep(T(-2), (phase_ - 1.0) * perCycle);
        }
        else
        {
            const Phase edge = fallingEdge();
            if (old < edge && phase_ >= edge)
                scheduleMinBlep(T(-2), (phase_ - edge) * perCycle);
            if (phase_ >= 1.0)
            {
                scheduleMinBlep(T(2), (phase_ - 1.0) * perCycle);
                if (phase_ >= 1.0 + edge)
                    scheduleMinBlep(T(-2), (phase_ - (1.0 + edge)) * perCycle);
            }
        }
        if (phase_ >= 1.0) phase_ -= 1.0;
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
        T raw = naive + takeCorrection();
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
                table.addDelayedStep(corr_.data() + corrHead_, 0, jump, static_cast<T>(age));
                table.addDelayedRamp(corr_.data() + corrHead_, 0, slopeJump, static_cast<T>(age));
                corrEnd_ = corrHead_ + kCorrLen;
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
        // The store is linear from its head (see corr_): one contiguous run.
        MinBlepTable<T>::instance().addStep(corr_.data() + corrHead_, 0, jump, static_cast<T>(frac));
        corrEnd_ = corrHead_ + kCorrLen;
    }

    /** @brief The correction pending for the sample being emitted (0 when
     *         none is); frees its slot and moves on to the next sample's. */
    [[nodiscard]] DSPARK_OSC_INLINE T takeCorrection() noexcept
    {
        if (corrEnd_ == 0) return T(0);
        const auto slot = static_cast<size_t>(corrHead_);
        const T value = corr_[slot];
        corr_[slot] = T(0);
        advanceCorrections(1);
        return value;
    }

    /** @brief Moves the head past @p count emitted (and cleared) slots. */
    DSPARK_OSC_INLINE void advanceCorrections(int count) noexcept
    {
        corrHead_ += count;
        if (corrHead_ >= std::min(corrEnd_, kCorrLen)) [[unlikely]] settleCorrections();
    }

    /**
     * @brief Brings the store back to its start once the head has reached
     *        the end of the pending corrections (none left: nothing to move)
     *        or the second half of the store (the pending ones move back
     *        with it).
     */
    DSPARK_NOINLINE void settleCorrections() noexcept
    {
        const int left = corrEnd_ - corrHead_;
        if (left > 0)
        {
            const auto live = static_cast<size_t>(left);
            std::copy_n(corr_.data() + corrHead_, live, corr_.data());
            std::fill_n(corr_.data() + corrHead_, live, T(0));
        }
        corrHead_ = 0;
        corrEnd_ = std::max(left, 0);
    }

    void clearCorrections() noexcept
    {
        if (corrEnd_ != 0)
            std::fill_n(corr_.data() + corrHead_, static_cast<size_t>(corrEnd_ - corrHead_), T(0));
        corrHead_ = 0;
        corrEnd_ = 0;
    }

    /**
     * @brief Samples per cycle, 1 / increment, for the edge times.
     *
     * A division, kept until the increment changes: a steady pitch pays it
     * once, and a pitch that moves every sample pays it only at its edges.
     */
    [[nodiscard]] Phase samplesPerCycle() noexcept
    {
        if (perCycleInc_ != phaseInc_)
        {
            perCycleInc_ = phaseInc_;
            perCycle_ = 1.0 / phaseInc_;
        }
        return perCycle_;
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
        clearCorrections();
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
    static DSPARK_OSC_MSVC_INLINE Phase polyBlep(Phase phase, Phase inc) noexcept
    {
        if (inc < 1e-10) [[unlikely]] return 0.0;

        if (phase < inc) [[unlikely]]
        {
            Phase t = phase / inc;
            return t + t - t * t - 1.0;
        }
        else if (phase > 1.0 - inc) [[unlikely]]
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
        // sync event logic both assume inc <= 0.5. The increment is kept on
        // the phase grid (see toGrid()), which makes the accumulator exact.
        phaseInc_ = toGrid(std::min(static_cast<Phase>(frequency_) * invSampleRate_, 0.5));
        // The saw's ramp delay (see nextMinBlepSample()), ready for both
        // render paths.
        sawDelay_ = static_cast<T>(2.0 * phaseInc_ * blepDelay_);
        // The slave's values are only read under sync and are worked out,
        // out of line, when it is asked for: without it (the usual case) a
        // pitch that changes every sample pays for none of that.
        if (syncRatio_ > T(1.001)) updateSync();
        else syncOn_ = false;

        // Only the Triangle reads the normalization, and its solution would
        // otherwise run on every frequency change of every waveform: a voice
        // modulating its pitch per sample calls setFrequency() per sample.
        // setWaveform() and seedTriangle() refresh it when it is next needed,
        // from the same pure function of the increment.
        if (waveform_ == Waveform::Triangle) updateTriNorm();
    }

    /** @brief The slave's increment, the synced square's DC and the sync state
     *         for the current increment and ratio (see setSyncRatio()). */
    DSPARK_NOINLINE void updateSync() noexcept
    {
        const bool wasOn = syncOn_;
        syncOn_ = phaseInc_ > 0.0;
        // Clamp the slave below Nyquist like any oscillator frequency.
        slaveInc_ = std::min(phaseInc_ * static_cast<Phase>(syncRatio_), 0.5);

        // DC of the hard-synced square (feeds the triangle integrator). The
        // slave always restarts in its +1 half, so its duty cycle is
        // asymmetric by construction: for an effective ratio r with
        // fractional part f, the +1 time per master cycle exceeds the -1
        // time by min(f, 0.5) - max(f - 0.5, 0) slave cycles.
        syncSquareDc_ = 0.0;
        if (syncOn_)
        {
            const Phase r = slaveInc_ / phaseInc_;   // effective (clamped) ratio
            const Phase f = r - std::floor(r);
            syncSquareDc_ = (std::min(f, 0.5) - std::max(f - 0.5, 0.0)) / r;
        }

        // Sync just (re-)engaged: the store may hold corrections from a
        // previous sync run and the slave phase is stale -- re-seed both.
        if (syncOn_ && !wasOn)
            reseedSlave();
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
                expectedPeak = solveTrianglePeak(inc);
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

    /** @brief The integrator's exact steady-state peak at @p inc, which
     *         becomes the anchor of updateTriNorm()'s first-order update. */
    DSPARK_NOINLINE Phase solveTrianglePeak(Phase inc) noexcept
    {
        const Phase lnQ = 0.5 * std::log1p(-inc) / inc;
        const Phase q = std::exp(lnQ);
        const Phase expectedPeak = (1.0 - q) / (1.0 + q);
        // dp/dinc = dp/dq * q * dlnQ/dinc
        const Phase dLnQ = 0.5 * (-inc / (1.0 - inc) - std::log1p(-inc)) / (inc * inc);
        triAnchorInc_ = inc;
        triAnchorPeak_ = expectedPeak;
        triAnchorSlope_ = -2.0 / ((1.0 + q) * (1.0 + q)) * q * dLnQ;
        return expectedPeak;
    }

    // Read or written on every sample: together at the front of the object.
    Phase    phase_      = 0.0;
    Phase    phaseInc_   = 0.0;
    double   triState_   = 0.0;      ///< Leaky-integrator state (double: recursive state rule).
    T        triNorm_    = T(4);
    T        sawDelay_   = T(0);     ///< 2 * increment * blepDelay_: the saw's ramp delay term.
    T        pulseWidth_ = T(0.5);   ///< Square duty cycle (setPulseWidth()).
    T        pulseDc_    = T(0);     ///< 2 * pulseWidth_ - 1, removed from the Square.
    int      corrHead_   = 0;        ///< Slot of the next output sample (see corr_).
    int      corrEnd_    = 0;        ///< One past the last slot that may be non-zero (0: none pending).
    Waveform waveform_   = Waveform::Sine;
    AntiAliasing antiAliasing_ = AntiAliasing::MinBLEP;
    bool     syncOn_     = false;

    // Set up by prepare() and the setters.
    double   sampleRate_ = 48000.0;
    double   invSampleRate_ = 1.0 / 48000.0;
    T        nyquist_    = T(24000);
    T        frequency_  = T(440);
    T        triExpectedPeak_ = T(0.25); ///< Steady-state integrator peak (reset seed).
    Phase    triNormInc_ = -1.0;     ///< Increment the tri norm was computed for (cache key).
    Phase    triAnchorInc_ = -1.0;   ///< Increment the peak was last solved exactly at.
    Phase    triAnchorPeak_ = 0.25;  ///< That exact peak.
    Phase    triAnchorSlope_ = 0.0;  ///< Its derivative with respect to the increment.
    Phase    blepDelay_  = 0.0;      ///< MinBlepTable::dcDelay(), cached by prepare().
    Phase    perCycleInc_ = -1.0;    ///< Increment perCycle_ was computed for (cache key).
    Phase    perCycle_   = 0.0;      ///< 1 / increment (see samplesPerCycle()).

    // Hard sync (slave) and minBLEP correction state.
    static constexpr int kCorrLen  = MinBlepTable<T>::kTaps;

    T        syncRatio_  = T(0);
    Phase    slavePhase_ = 0.0;
    Phase    slaveInc_   = 0.0;
    Phase    syncSquareDc_ = 0.0;      ///< Analytic duty-cycle DC of the synced square.
    static constexpr Phase kNoReset = 1e9;
    std::array<Phase, 2> resetAge_{ kNoReset, kNoReset }; ///< Synced sine: samples since the last two resets.
    std::array<Phase, 2> prePhase_{};  ///< Synced sine: slave phase just before those resets.
    /// Pending causal minBLEP corrections, linear from the head: the one
    /// for the sample j ahead is corr_[corrHead_ + j], j < kCorrLen, so a
    /// discontinuity adds its whole kernel as one contiguous run. They end
    /// before corrEnd_, and every slot outside [corrHead_, corrEnd_) is
    /// zero; with none pending both are 0 and nothing is read or cleared.
    /// The head stays below kCorrLen (see settleCorrections()).
    std::array<T, 2 * kCorrLen> corr_{};
};

} // namespace dspark

#undef DSPARK_OSC_INLINE
#undef DSPARK_OSC_OUTLINE
#undef DSPARK_OSC_MSVC_INLINE
#undef DSPARK_OSC_ROUNDED
