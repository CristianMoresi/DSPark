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
 * frequency. Two opt-in saturation models (setSaturation()): Stages
 * saturates every stage as well and solves the resulting nonlinear
 * zero-delay loop implicitly each sample; Input saturates the loop input and
 * runs four sub-steps per sample, each with its loop closed without delay.
 *
 * Threading (SPSC model, see docs/threading.md): parameter setters belong to
 * the control thread (ONE non-audio writer; independent single-word relaxed
 * atomics -- setCutoff() also derives and republishes the g_ word, which is
 * why a single writer is required); getters are lock-free readouts callable
 * from any thread. prepare/reset are setup-time; processBlock/processSample
 * belong to the owning audio thread. processBlock installs a DenormalGuard;
 * per-sample callers are expected to guard their own callback (framework
 * convention) - resonant tails decay through the denormal range otherwise.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, DenormalGuard.h,
 * StateBlob.h, detail/LadderZdf.h (the per-sample arithmetic, shared with
 * owners that modulate the cutoff every sample).
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "DenormalGuard.h"
#include "StateBlob.h"
#include "detail/LadderZdf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

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
 * Three saturation models (setSaturation()):
 * - **Feedback** (default): linear stages, with drive > 1 shaping only the
 *   resonance feedback through a bounded tanh. Cheapest (no iteration).
 * - **Stages**: the transistor-ladder model in which every stage saturates,
 *   dy_i/dt = wc [phi(y_{i-1}) - phi(y_i)], phi(v) = tanh(drive v) / drive,
 *   solved implicitly per sample (detail::ladderStepStages()). Against a
 *   128x oversampled reference of the same equations (48 kHz, 110 Hz
 *   full-scale sine, drive 1 and 3, resonance 0 to 0.9) every harmonic above
 *   -80 dBc lands within 0.23 dB up to a 2 kHz cutoff and within 1.7 dB at
 *   8 kHz (the frequency-warping limit of any trapezoidal solve at the base
 *   rate). It self-oscillates above resonance 1 at a level the stages set:
 *   at resonance 1.1 the pitch is within 0.1 cents of the reference up to a
 *   1 kHz cutoff, 1.5 cents at 4 kHz and 7.9 cents at 10 kHz, and the level
 *   within 0.07 dB. It changes the character, so it is opt-in. It costs
 *   several times the Feedback model (see setSaturation()).
 * - **Input**: linear stages and one saturator on the loop input,
 *   y_0 = phi(x - k y_4), run in four sub-steps per sample with the loop
 *   closed without delay in each (detail::ladderStepInput()). The sub-steps
 *   put the frequency warping of the integrators and the aliasing of the
 *   saturator two octaves further up. At 48 kHz, against a 32x oversampled
 *   solve of the same equations (110 Hz full-scale sine, drive 1 and 3,
 *   resonance 0 to 0.9) every harmonic above -80 dBc lands within 0.05 dB at
 *   cutoffs from 250 Hz to 20 kHz. The resonance peak is within 0.25 cents
 *   of the analog one up to an 8 kHz cutoff, 0.9 cents at 16 kHz and 1.1 at
 *   20 kHz (the other two models: 3.8, 13 and 18 cents sharp), and with the
 *   resonance at 0 the small-signal response is within 0.5 dB of the analog
 *   four-pole up to 15 kHz at any cutoff (-0.6 to -2.1 dB at 20 kHz). A
 *   full-scale 5 kHz tone at drive 3 and resonance 0.5 aliases at -45 to
 *   -61 dBc over cutoffs from 500 Hz to 20 kHz. It costs the same at every
 *   cutoff and resonance, below Stages at every setting measured (see
 *   setSaturation()). It is a third character, so it is opt-in as well.
 *
 * Contract notes:
 * - Like the analog original, the passband gain drops by 1/(1 + 4*resonance)
 *   as resonance rises (about -14 dB at resonance 1). Compensate externally
 *   if loudness-matched sweeps are needed.
 * - Feedback model: drive saturates the resonance feedback path only: with
 *   resonance at 0 it has no effect on the signal. With drive > 1 the
 *   bounded tanh stage keeps self-oscillation (resonance = 1) at a finite
 *   level; at drive <= 1 the loop is exactly marginal at resonance = 1, so
 *   sustained input at the cutoff accumulates without decay - back off
 *   resonance slightly or engage drive for stable self-oscillation.
 * - Stages model: drive is the level at which the stages saturate (a
 *   full-scale signal at drive 1 sits at the tanh(1) knee), at any drive.
 *   Its cutoff is limited to 0.45 * fs (detail::kLadderStagesMaxGain).
 * - Input model: drive is the level at which the loop input (the input
 *   minus the resonance feedback) saturates, at any drive and resonance:
 *   with the resonance at 0, a full-scale signal at drive 1 sits at the
 *   tanh(1) knee. Above resonance 1 it self-oscillates at a level the
 *   saturator sets. Its cutoff reaches 0.499 * fs, like the Feedback
 *   model's. The HP24 output is taken from the saturated loop input. The
 *   input is interpolated across the sub-steps and the output is their
 *   mean, each with a one-pole level correction; the first sample after
 *   another model or a reset starts both histories.
 * - Channels beyond kMaxChannels (16) pass through unprocessed.
 * - Mode and saturation model can be switched while running; the integrator
 *   state is continuous.
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

    /** @brief Where the filter saturates (see the class notes). */
    enum class Saturation
    {
        Feedback,   ///< Linear stages, tanh on the resonance feedback (default).
        Stages,     ///< Every stage saturates; implicit per-sample solve.
        Input       ///< Linear stages, tanh on the loop input; sub-stepped.
    };

    /** @brief Highest resonance setResonance() accepts (Feedback uses at most 1). */
    static constexpr double kMaxResonance = 1.2;

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
     * branchless inner loop instantiated per mode. A non-finite integrator
     * state is cleared at the end of the block, so one NaN input cannot
     * silence the channel until reset().
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
        const T storedRes = resonance_.load(std::memory_order_relaxed);
        const T currentDrive = drive_.load(std::memory_order_relaxed);
        const Mode currentMode = mode_.load(std::memory_order_relaxed);
        const Saturation model = saturation_.load(std::memory_order_relaxed);

        if (model == Saturation::Input)
        {
            processBlockInput(buffer, nCh, nS,
                              inputCoeffs(cutoff_.load(std::memory_order_relaxed), storedRes, currentDrive),
                              currentMode);
            return;
        }
        // The other models take the integrator states back from the Input model.
        for (int ch = 0; ch < nCh; ++ch) leaveInput(state_[ch]);

        if (model == Saturation::Stages)
        {
            processBlockStages(buffer, nCh, nS,
                               detail::makeLadderStageCoeffs(currentG, storedRes, currentDrive),
                               currentMode);
            return;
        }

        // The Feedback model stops at the self-oscillation boundary.
        const T currentRes = std::min(storedRes, T(1));

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
        ChannelState& s = state_[channel];
        const Saturation model = saturation_.load(std::memory_order_relaxed);

        if (model == Saturation::Input)
        {
            const auto& ic = inputCoeffs(cutoff_.load(std::memory_order_relaxed), res, drive);
            switch (mode_.load(std::memory_order_relaxed))
            {
                case Mode::LP6:  return inputSample<detail::LadderTap::LP6>(input, s, ic);
                case Mode::LP12: return inputSample<detail::LadderTap::LP12>(input, s, ic);
                case Mode::LP18: return inputSample<detail::LadderTap::LP18>(input, s, ic);
                case Mode::LP24: return inputSample<detail::LadderTap::LP24>(input, s, ic);
                case Mode::BP12: return inputSample<detail::LadderTap::BP12>(input, s, ic);
                case Mode::HP24: return inputSample<detail::LadderTap::HP24>(input, s, ic);
            }
            return input;
        }
        leaveInput(s);

        if (model == Saturation::Stages)
        {
            const auto sc = detail::makeLadderStageCoeffs(g, res, drive);
            switch (mode_.load(std::memory_order_relaxed))
            {
                case Mode::LP6:  return detail::ladderStepStages<detail::LadderTap::LP6>(input, s.z, sc);
                case Mode::LP12: return detail::ladderStepStages<detail::LadderTap::LP12>(input, s.z, sc);
                case Mode::LP18: return detail::ladderStepStages<detail::LadderTap::LP18>(input, s.z, sc);
                case Mode::LP24: return detail::ladderStepStages<detail::LadderTap::LP24>(input, s.z, sc);
                case Mode::BP12: return detail::ladderStepStages<detail::LadderTap::BP12>(input, s.z, sc);
                case Mode::HP24: return detail::ladderStepStages<detail::LadderTap::HP24>(input, s.z, sc);
            }
            return input;
        }

        const BlockCoeffs c = makeBlockCoeffs(g, std::min(res, T(1)));

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
        {
            s.z.fill(T(0));
            s.input = {};
            s.inputLive = false;
        }
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
     * @param amount Range [0.0, 1.0] for the Feedback model; values near 1.0
     *               induce self-oscillation (see the class notes on drive
     *               for loop stability). The Stages and Input models also
     *               take values up to kMaxResonance (1.2), where they
     *               self-oscillate at a level their saturation sets; the
     *               Feedback model treats them as 1. Clamped to
     *               [0, kMaxResonance]; non-finite values are ignored.
     */
    void setResonance(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;  // NaN would poison the loop via k
        resonance_.store(std::clamp(amount, T(0), static_cast<T>(kMaxResonance)),
                         std::memory_order_relaxed);
    }

    /**
     * @brief Sets the nonlinear drive amount (thread-safe, lock-free).
     * @param amount Drive multiplier, floored at 0.1. Feedback model: values
     *               <= 1.0 keep the feedback path linear; > 1.0 saturates it
     *               (analog-style resonance compression). Stages model: the
     *               level scale of every stage, tanh(drive * v) / drive.
     *               Input model: the same scale on the loop input.
     *               Non-finite values are ignored.
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

    /**
     * @brief Selects the saturation model (thread-safe, lock-free).
     *
     * Feedback (the default) keeps the stages linear. Stages saturates every
     * stage and solves the loop implicitly: a clipped Newton iteration in
     * double, 2.1 steps on average and at most 7 measured (cap 16). It costs
     * 38 to 100 ns per sample (mono) against 10 to 19 ns for Feedback (MSVC
     * x64, float), rising with cutoff, drive and the noisiness of the input:
     * 54 ns at a 1 kHz cutoff with drive 3; at 15 kHz with drive 3, 54-58 ns
     * for sines and a saw and 98 ns for white noise. Channel pairs are solved
     * together, which lowers the cost per channel by 10-20 % in stereo.
     *
     * Input saturates the loop input and runs four sub-steps per sample, the
     * same work at every cutoff and resonance: 53 to 74 ns per sample in
     * mono depending on the machine (float; GCC, Clang and MSVC on x64 and
     * GCC on arm64) and 35 to 48 ns per channel for channel pairs, which
     * advance together. A saturator driven past the limit of its tanh skips
     * its division (at drive 10, from 16% less to 4% more). On the same
     * machine and settings that is 1.5 to 4.0 times Feedback and 0.30 to
     * 0.95 times Stages.
     *
     * The integrator state carries over a switch. Out-of-range values (a wild
     * cast or a corrupt state blob) select Feedback.
     *
     * @param model Saturation model.
     */
    void setSaturation(Saturation model) noexcept
    {
        const bool known = model == Saturation::Stages || model == Saturation::Input;
        saturation_.store(known ? model : Saturation::Feedback, std::memory_order_relaxed);
    }

    // -- State (setup/UI thread) ------------------------------------------------

    /** @brief Serializes the parameters (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float: the casts keep the double instantiation
        // compiling (a double matches no StateWriter overload unambiguously).
        StateWriter w(stateId("LADR"), 1);
        w.write("cutoff", static_cast<float>(cutoff_.load(std::memory_order_relaxed)));
        w.write("resonance", static_cast<float>(resonance_.load(std::memory_order_relaxed)));
        w.write("drive", static_cast<float>(drive_.load(std::memory_order_relaxed)));
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("saturation", static_cast<int32_t>(saturation_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /**
     * @brief Restores parameters from a blob (tolerant; rejects foreign ids).
     *
     * Missing keys keep their defaults: a blob without "saturation" selects
     * the Feedback model.
     */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("LADR")) return false;
        setCutoff(static_cast<T>(r.read("cutoff", 1000.0f)));
        setResonance(static_cast<T>(r.read("resonance", 0.0f)));
        setDrive(static_cast<T>(r.read("drive", 1.0f)));
        const int mode = std::clamp(r.read("mode", static_cast<int32_t>(Mode::LP24)),
                                    0, static_cast<int32_t>(Mode::HP24));
        setMode(static_cast<Mode>(mode));
        setSaturation(static_cast<Saturation>(r.read("saturation", 0)));
        return true;
    }

    [[nodiscard]] T getCutoff() const noexcept { return cutoff_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getResonance() const noexcept { return resonance_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDrive() const noexcept { return drive_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] Saturation getSaturation() const noexcept { return saturation_.load(std::memory_order_relaxed); }

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
        detail::LadderInputState input {};  // Input model: its own state, in double
        bool inputLive = false;             // whether the Input model holds the stream
    };

    std::array<ChannelState, kMaxChannels> state_ {};
    AudioSpec spec_ {};

    // Coefficients of the Input model, rebuilt on the audio thread when a
    // parameter it was derived from changes.
    struct InputCache
    {
        T cutoff = T(-1), resonance = T(-1), drive = T(-1);
        double sampleRate = 0.0;
        detail::LadderInputCoeffs coeffs {};
    };
    InputCache inputCache_ {};

    // Atomic variables for thread-safety between Audio Thread and UI/Main Thread
    std::atomic<T> cutoff_ {T(1000)};
    std::atomic<T> resonance_ {T(0)};
    std::atomic<T> drive_ {T(1)};
    std::atomic<T> g_ {T(0)};
    std::atomic<Mode> mode_ {Mode::LP24};
    std::atomic<Saturation> saturation_ {Saturation::Feedback};

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
    using BlockCoeffs = detail::LadderCoeffs<T>;

    /** @brief Derives the block-constant coefficients from g and resonance. */
    [[nodiscard]] static BlockCoeffs makeBlockCoeffs(T g, T res) noexcept
    {
        return detail::makeLadderCoeffs(g, res);
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
            // Integrators in a local: the sample pointer cannot alias it, so it
            // stays in registers (11.3 -> 9.8 ns/sample measured, same output).
            std::array<T, 4> z = state_[ch].z;

            for (int i = 0; i < nS; ++i)
                channelData[i] = detail::ladderStep<tapOf(FilterMode)>(channelData[i], z, c, drive);

            // A NaN or Inf sample that reached the recursion is cleared once
            // per block, so the channel recovers instead of staying NaN.
            if (!std::isfinite(z[0] + z[1] + z[2] + z[3])) z.fill(T(0));
            state_[ch].z = z;
        }
    }

    /**
     * @brief The Input model's coefficients for these parameters (audio thread).
     *
     * Rebuilt when a parameter they were derived from changed; the channels
     * the model is running on then get their carried sums recomputed.
     */
    [[nodiscard]] const detail::LadderInputCoeffs& inputCoeffs(T cutoff, T res, T drive) noexcept
    {
        InputCache& cache = inputCache_;
        const bool same = cutoff == cache.cutoff && res == cache.resonance && drive == cache.drive
                       && spec_.sampleRate == cache.sampleRate;
        if (!same)
        {
            cache.coeffs = detail::makeLadderInputCoeffs(static_cast<double>(cutoff) / spec_.sampleRate, res, drive);
            for (auto& s : state_)
                if (s.inputLive) detail::ladderInputRefresh(s.input, cache.coeffs);
            cache.cutoff = cutoff;
            cache.resonance = res;
            cache.drive = drive;
            cache.sampleRate = spec_.sampleRate;
        }
        return cache.coeffs;
    }

    /** @brief Hands a channel the Input model was running on back to the other models. */
    static void leaveInput(ChannelState& s) noexcept
    {
        if (!s.inputLive) return;
        for (size_t i = 0; i < 4; ++i) s.z[i] = static_cast<T>(s.input.q[i]);
        s.inputLive = false;
    }

    /**
     * @brief One sample of the Input model on a channel.
     *
     * The first sample after another model (or a reset) takes the integrator
     * states over and starts the model's history from that sample: no
     * interpolation from a stale input and no level correction against a
     * stale output. The kernel is inlined here once, so that sample runs it
     * with its own value as the previous input (nothing to interpolate) and
     * no previous output, and the gain that leaves on its output is taken
     * back.
     */
    template <detail::LadderTap Tap>
    [[nodiscard]] T inputSample(T input, ChannelState& s, const detail::LadderInputCoeffs& c) noexcept
    {
        const bool first = !s.inputLive;
        if (first) [[unlikely]]
        {
            s.input = {};
            for (size_t i = 0; i < 4; ++i) s.input.q[i] = static_cast<double>(s.z[i]);
            detail::ladderInputRefresh(s.input, c);
            s.input.input = static_cast<double>(input);
            s.inputLive = true;
        }
        T y = detail::ladderStepInput<Tap>(input, s.input, c);
        if (first) [[unlikely]]
        {
            const double mean = s.input.output / (1.0 + c.post);
            s.input.output = mean;
            y = static_cast<T>(mean);
        }
        return y;
    }

    /** @brief Block loop of the Input model, one mode instantiation each. */
    void processBlockInput(AudioBufferView<T> buffer, int nCh, int nS,
                           const detail::LadderInputCoeffs& c, Mode mode) noexcept
    {
        switch (mode)
        {
            case Mode::LP6:  inputLoop<detail::LadderTap::LP6>(buffer, nCh, nS, c); break;
            case Mode::LP12: inputLoop<detail::LadderTap::LP12>(buffer, nCh, nS, c); break;
            case Mode::LP18: inputLoop<detail::LadderTap::LP18>(buffer, nCh, nS, c); break;
            case Mode::LP24: inputLoop<detail::LadderTap::LP24>(buffer, nCh, nS, c); break;
            case Mode::BP12: inputLoop<detail::LadderTap::BP12>(buffer, nCh, nS, c); break;
            case Mode::HP24: inputLoop<detail::LadderTap::HP24>(buffer, nCh, nS, c); break;
        }
    }

    /** @brief Clears a state of the Input model that a NaN or Inf sample reached. */
    static void recoverInput(detail::LadderInputState& h) noexcept
    {
        if (!std::isfinite(((h.q[0] + h.q[1]) + (h.q[2] + h.q[3])) + ((h.input + h.output) + h.ahead)
                           + (h.loopStates + h.loopDrive)))
            h = {};
    }

    template <detail::LadderTap Tap>
    void inputLoop(AudioBufferView<T> buffer, int nCh, int nS, const detail::LadderInputCoeffs& c) noexcept
    {
        if (nS <= 0) return;
        int ch = 0;
        // Channel pairs advance together, sub-step by sub-step: the two
        // solves are independent, so their dependency chains overlap (same
        // arithmetic per channel, identical output).
        for (; ch + 1 < nCh; ch += 2)
        {
            auto* a = buffer.getChannel(ch);
            auto* b = buffer.getChannel(ch + 1);
            ChannelState& sa = state_[ch];
            ChannelState& sb = state_[ch + 1];
            int i = 0;
            if (!sa.inputLive || !sb.inputLive)
            {
                a[0] = inputSample<Tap>(a[0], sa, c);
                b[0] = inputSample<Tap>(b[0], sb, c);
                i = 1;
            }
            // The states in locals: the sample pointers cannot alias them.
            detail::LadderInputState ha = sa.input, hb = sb.input;
            for (; i < nS; ++i)
                detail::ladderStepInputPair<Tap>(a[i], b[i], ha, hb, c);
            // Same per-block NaN recovery as the other models.
            recoverInput(ha);
            recoverInput(hb);
            sa.input = ha;
            sb.input = hb;
        }
        for (; ch < nCh; ++ch)
        {
            auto* channelData = buffer.getChannel(ch);
            ChannelState& s = state_[ch];
            int i = 0;
            if (!s.inputLive)
            {
                channelData[0] = inputSample<Tap>(channelData[0], s, c);
                i = 1;
            }
            detail::LadderInputState h = s.input;
            for (; i < nS; ++i)
                channelData[i] = detail::ladderStepInput<Tap>(channelData[i], h, c);
            recoverInput(h);
            s.input = h;
        }
    }

    /** @brief Block loop of the Stages model, one mode instantiation each. */
    void processBlockStages(AudioBufferView<T> buffer, int nCh, int nS,
                            const detail::LadderStageCoeffs& c, Mode mode) noexcept
    {
        switch (mode)
        {
            case Mode::LP6:  stagesLoop<detail::LadderTap::LP6>(buffer, nCh, nS, c); break;
            case Mode::LP12: stagesLoop<detail::LadderTap::LP12>(buffer, nCh, nS, c); break;
            case Mode::LP18: stagesLoop<detail::LadderTap::LP18>(buffer, nCh, nS, c); break;
            case Mode::LP24: stagesLoop<detail::LadderTap::LP24>(buffer, nCh, nS, c); break;
            case Mode::BP12: stagesLoop<detail::LadderTap::BP12>(buffer, nCh, nS, c); break;
            case Mode::HP24: stagesLoop<detail::LadderTap::HP24>(buffer, nCh, nS, c); break;
        }
    }

    template <detail::LadderTap Tap>
    void stagesLoop(AudioBufferView<T> buffer, int nCh, int nS, const detail::LadderStageCoeffs& c) noexcept
    {
        int ch = 0;
        // Channel pairs advance together, one sample of each per iteration:
        // the two solves are independent, so their dependency chains overlap
        // (same arithmetic per channel, identical output).
        for (; ch + 1 < nCh; ch += 2)
        {
            auto* a = buffer.getChannel(ch);
            auto* b = buffer.getChannel(ch + 1);
            std::array<T, 4> za = state_[ch].z, zb = state_[ch + 1].z;
            for (int i = 0; i < nS; ++i)
            {
                a[i] = detail::ladderStepStages<Tap>(a[i], za, c);
                b[i] = detail::ladderStepStages<Tap>(b[i], zb, c);
            }
            if (!std::isfinite(za[0] + za[1] + za[2] + za[3])) za.fill(T(0));
            if (!std::isfinite(zb[0] + zb[1] + zb[2] + zb[3])) zb.fill(T(0));
            state_[ch].z = za;
            state_[ch + 1].z = zb;
        }
        for (; ch < nCh; ++ch)
        {
            auto* channelData = buffer.getChannel(ch);
            std::array<T, 4> z = state_[ch].z;
            for (int i = 0; i < nS; ++i)
                channelData[i] = detail::ladderStepStages<Tap>(channelData[i], z, c);
            // Same per-block NaN recovery as the Feedback loop.
            if (!std::isfinite(z[0] + z[1] + z[2] + z[3])) z.fill(T(0));
            state_[ch].z = z;
        }
    }

    /** @brief The tap mix of a mode, for the shared ladder step. */
    [[nodiscard]] static constexpr detail::LadderTap tapOf(Mode mode) noexcept
    {
        switch (mode)
        {
            case Mode::LP6:  return detail::LadderTap::LP6;
            case Mode::LP12: return detail::LadderTap::LP12;
            case Mode::LP18: return detail::LadderTap::LP18;
            case Mode::LP24: return detail::LadderTap::LP24;
            case Mode::BP12: return detail::LadderTap::BP12;
            case Mode::HP24: return detail::LadderTap::HP24;
        }
        return detail::LadderTap::LP24;
    }

    /**
     * @brief Core sample-processing DSP algorithm (detail::ladderStep()).
     *
     * The mode template parameter lets the compiler inline the tap mix and
     * remove every branch from the audio processing loop.
     */
    template <Mode FilterMode>
    [[nodiscard]] T processSampleInternal(T input, ChannelState& s, const BlockCoeffs& c, T drive) noexcept
    {
        return detail::ladderStep<tapOf(FilterMode)>(input, s.z, c, drive);
    }
};

} // namespace dspark
