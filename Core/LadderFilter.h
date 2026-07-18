// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LadderFilter.h
 * @brief Moog-style 4-pole resonant ladder filter (TPT, zero-delay feedback).
 *
 * Four cascaded TPT one-pole stages (Zavalishin) with the global feedback
 * loop resolved analytically per sample - no unit delay in the loop, so
 * cutoff and resonance match the analog prototype at the prewarped
 * frequency.
 *
 * Threading: parameter setters/getters are lock-free atomics (any thread).
 * prepare/reset are setup-time; processBlock/processSample belong to the
 * owning audio thread. processBlock installs a DenormalGuard; per-sample
 * callers are expected to guard their own callback (framework convention) -
 * resonant tails decay through the denormal range otherwise.
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

namespace dspark {

/**
 * @class LadderFilter
 * @brief 4-pole resonant ladder filter (Moog topology, TPT discretization).
 *
 * Four cascaded one-pole TPT lowpass stages with global feedback. The
 * zero-delay feedback loop is resolved analytically each sample, and the
 * multimode outputs are tap mixes of the cascade. processBlock() snapshots
 * the parameters once per block and dispatches to a per-mode template
 * instantiation, so the sample loop carries no branches.
 *
 * Contract notes:
 * - Like the analog original, the passband gain drops by 1/(1 + 4*resonance)
 *   as resonance rises (about -14 dB at resonance 1). Compensate externally
 *   if loudness-matched sweeps are needed.
 * - Drive saturates the resonance feedback path only: with resonance at 0 it
 *   has no effect on the signal. With drive > 1 the bounded tanh stage keeps
 *   self-oscillation (resonance = 1) at a finite level; at drive <= 1 the
 *   loop is exactly marginal at resonance = 1, so sustained input at the
 *   cutoff accumulates without decay - back off resonance slightly or engage
 *   drive for stable self-oscillation.
 * - Channels beyond kMaxChannels (16) pass through unprocessed.
 * - Mode can be switched while running; the integrator state is continuous.
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
        LP24,   ///< 4-pole lowpass (24 dB/oct) - classic Moog style.
        BP12,   ///< Bandpass (6 dB/oct per side).
        HP24    ///< 4-pole highpass (24 dB/oct).
    };

    ~LadderFilter() = default;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the filter with the current audio environment.
     *
     * Invalid specs (sample rate not > 0, NaN included) are ignored and the
     * previous state is kept. Clears the integrator state.
     *
     * @param spec Audio specification including sample rate.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!(spec.sampleRate > 0.0)) return;
        spec_ = spec;
        updateCoefficients();
        reset();
    }

    /**
     * @brief Processes an audio buffer in-place (thread-safe vs. setters).
     *
     * Takes an atomic snapshot of the current parameters and dispatches to a
     * branchless inner loop instantiated per mode.
     *
     * @param buffer View of the multi-channel audio data to process.
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
     * @brief Processes a single sample on one channel.
     *
     * Convenience per-sample entry point mirroring Biquad. For bulk
     * processing prefer processBlock(), which snapshots the parameters and
     * derives the loop coefficients once per block instead of per sample.
     *
     * @note Unlike processBlock(), no DenormalGuard is installed here (the
     *       per-sample FP-environment writes would cost more than the filter
     *       itself); per-sample callers should place one DenormalGuard
     *       around their processing loop.
     *
     * @param input   Input sample.
     * @param channel Channel index in [0, kMaxChannels).
     * @return Filtered sample (returns @p input unchanged if @p channel is out of range).
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (channel < 0 || channel >= kMaxChannels) return input;

        const T g     = g_.load(std::memory_order_relaxed);
        const T res   = resonance_.load(std::memory_order_relaxed);
        const T drive = drive_.load(std::memory_order_relaxed);
        const BlockCoeffs c = makeBlockCoeffs(g, res);
        ChannelState& s = state_[channel];

        switch (mode_.load(std::memory_order_relaxed))
        {
            case Mode::LP6:  return processSampleInternal<Mode::LP6>(input, s, c, drive);
            case Mode::LP12: return processSampleInternal<Mode::LP12>(input, s, c, drive);
            case Mode::LP18: return processSampleInternal<Mode::LP18>(input, s, c, drive);
            case Mode::LP24: return processSampleInternal<Mode::LP24>(input, s, c, drive);
            case Mode::BP12: return processSampleInternal<Mode::BP12>(input, s, c, drive);
            case Mode::HP24: return processSampleInternal<Mode::HP24>(input, s, c, drive);
        }
        return input;
    }

    /**
     * @brief Clears the internal integrator state.
     *
     * Should be called when playback stops or continuity is broken to
     * prevent clicks.
     */
    void reset() noexcept
    {
        for (auto& s : state_)
            s.z.fill(T(0));
    }

    // -- Parameters -------------------------------------------------------------

    /**
     * @brief Sets the cutoff frequency (thread-safe, lock-free).
     *
     * Non-finite values (NaN/Inf) are ignored - they would poison the
     * integrator gain permanently otherwise. Before prepare() the sample
     * rate is unknown, so the raw request is stored and clamped to
     * [20, sampleRate * 0.499] once prepare() runs (getCutoff() then
     * reports the clamped value).
     *
     * @param hz Cutoff frequency in Hz.
     */
    void setCutoff(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        cutoff_.store(std::max(hz, T(0)), std::memory_order_relaxed);
        updateCoefficients();
    }

    /**
     * @brief Sets resonance amount (thread-safe, lock-free).
     * @param amount Range [0.0, 1.0]. Values near 1.0 induce self-oscillation
     *               (see the class notes on drive for loop stability).
     *               Non-finite values are ignored.
     */
    void setResonance(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;  // NaN would poison the loop via k
        resonance_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the nonlinear drive amount (thread-safe, lock-free).
     * @param amount Drive multiplier, floored at 0.1. Values <= 1.0 keep the
     *               feedback path linear; > 1.0 saturates it (analog-style
     *               resonance compression). Non-finite values are ignored.
     */
    void setDrive(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;  // Inf turns 0 * drive into NaN
        drive_.store(std::max(amount, T(0.1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the filter output mode (thread-safe, lock-free).
     * @param mode Target frequency response type.
     */
    void setMode(Mode mode) noexcept
    {
        mode_.store(mode, std::memory_order_relaxed);
    }

    [[nodiscard]] T getCutoff() const noexcept { return cutoff_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getResonance() const noexcept { return resonance_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDrive() const noexcept { return drive_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }

protected:
    static constexpr int kMaxChannels = 16;

    // Scalar per-channel state - no SIMD kernel ever touches it, and adjacent
    // channels sharing a cache line is better locality than padded isolation,
    // so it is deliberately not over-aligned. The stage tap outputs are NOT
    // state (each sample overwrites all of them), so they live in registers
    // inside processSampleInternal().
    struct ChannelState
    {
        std::array<T, 4> z {};  // TPT integrator states (z^-1 equivalents)
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
     * @brief Clamps the stored cutoff and derives the prewarped integrator gain.
     *
     * No-op until prepare() provides a valid sample rate. Re-storing the
     * clamped cutoff keeps getCutoff() honest for pre-prepare requests.
     */
    void updateCoefficients() noexcept
    {
        if (!(spec_.sampleRate > 0.0)) return;
        const T clamped = std::clamp(cutoff_.load(std::memory_order_relaxed),
                                     T(20), static_cast<T>(spec_.sampleRate) * T(0.499));
        cutoff_.store(clamped, std::memory_order_relaxed);
        const T preWarpedGain = static_cast<T>(std::tan(pi<double> * static_cast<double>(clamped) / spec_.sampleRate));
        g_.store(preWarpedGain, std::memory_order_relaxed);
    }

    /** @brief g/res-derived coefficients, constant for the whole block. */
    struct BlockCoeffs
    {
        T G, G2, G3, G4, ig, k, invFbDen;
    };

    /** @brief Derives the block-constant coefficients from g and resonance. */
    [[nodiscard]] static BlockCoeffs makeBlockCoeffs(T g, T res) noexcept
    {
        BlockCoeffs c;
        c.G   = g / (T(1) + g);
        c.G2  = c.G * c.G;
        c.G3  = c.G2 * c.G;
        c.G4  = c.G3 * c.G;
        c.ig  = T(1) / (T(1) + g);
        c.k   = res * T(4);                         // k = 4 -> self-oscillation
        c.invFbDen = T(1) / (T(1) + c.k * c.G4);    // zero-delay loop denominator
        return c;
    }

    /** @brief Per-mode block loop; the mode template kills per-sample branching. */
    template <Mode FilterMode>
    void processBlockInternal(AudioBufferView<T> buffer, int nCh, int nS, T g, T res, T drive) noexcept
    {
        // Precompute the block-constant coefficients ONCE (previously these two
        // divisions ran for every sample of every channel).
        const BlockCoeffs c = makeBlockCoeffs(g, res);

        for (int ch = 0; ch < nCh; ++ch)
        {
            auto* channelData = buffer.getChannel(ch);
            auto& state = state_[ch];

            for (int i = 0; i < nS; ++i)
            {
                channelData[i] = processSampleInternal<FilterMode>(channelData[i], state, c, drive);
            }
        }
    }

    /**
     * @brief Core sample-processing DSP algorithm.
     *
     * The mode template parameter lets the compiler inline the tap mix and
     * remove every branch from the audio processing loop.
     */
    template <Mode FilterMode>
    [[nodiscard]] T processSampleInternal(T input, ChannelState& s, const BlockCoeffs& c, T drive) noexcept
    {
        // Estimate the LP24 output from the integrator states (zero-delay logic)
        const T Sest = c.G3 * c.ig * s.z[0]
                     + c.G2 * c.ig * s.z[1]
                     + c.G  * c.ig * s.z[2]
                     +        c.ig * s.z[3];

        // Saturate the estimated feedback (cheap ZDF nonlinearity: the state
        // contribution is shaped, the direct G4*u term stays linear).
        // fastTanh clamps its input, so the saturated loop stays bounded.
        T fbSignal = Sest;
        if (drive > T(1))
            fbSignal = fastTanh(fbSignal * drive) / drive;

        // Resolve zero-delay loop (denominator precomputed once per block)
        const T u = (input - c.k * fbSignal) * c.invFbDen;

        // Four cascaded TPT one-pole stages. Tap outputs carry no state
        // between samples, so they stay in registers ({} avoids a spurious
        // uninitialised-variable warning; the stores are dead after unroll).
        std::array<T, 4> y {};
        T x = u;
        for (int i = 0; i < 4; ++i)
        {
            const T v = (x - s.z[i]) * c.G;
            y[i] = v + s.z[i];
            s.z[i] = y[i] + v;
            x = y[i];
        }

        // Output tap mix (resolved at compile time per mode instantiation)
        if constexpr (FilterMode == Mode::LP6)       return y[0];
        else if constexpr (FilterMode == Mode::LP12) return y[1];
        else if constexpr (FilterMode == Mode::LP18) return y[2];
        else if constexpr (FilterMode == Mode::LP24) return y[3];
        else if constexpr (FilterMode == Mode::BP12) return y[0] - y[2];
        else
        {
            // HP24: apply the binomial (1-L)^4 to `u` (the ladder's true input,
            // post feedback), NOT to `input`. With resonance, input = u*(1+k) at
            // DC, so the input-based form leaked DC with gain k/(1+k) - e.g. 67%
            // of the DC passed straight through a "high-pass" at resonance 0.5.
            // With u the DC gain is exactly 0 and the resonant peak is preserved.
            return u - T(4)*y[0] + T(6)*y[1] - T(4)*y[2] + y[3];
        }
    }
};

} // namespace dspark
