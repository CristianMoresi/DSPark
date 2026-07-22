// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file EnvelopeFollower.h
 * @brief Peak/RMS envelope follower with asymmetric attack/release.
 *
 * The detector every dynamics processor carries internally, exposed as a
 * public building block: a one-pole follower with independent attack and
 * release time constants, in Peak mode (rectified signal) or RMS mode
 * (the ballistics run on the squared signal and readouts return the square
 * root, so the time constants act in the power domain - the standard
 * detector convention). Use it as a sidechain detector, a modulation
 * source (with ModulationRouter), or a meter.
 *
 * Non-finite input samples: the poisoned channel restarts from zero at the
 * next publish and measurement resumes immediately (attack-time recovery);
 * the readouts never serve non-finite values.
 *
 * Threading:
 * - prepare(): setup thread (allocation-free; not concurrent with
 *   processing).
 * - processBlock() / processSample() / reset(): audio thread (stream
 *   owner).
 * - setAttack() / setRelease() / setMode(): any thread (relaxed atomics;
 *   non-finite values are ignored).
 * - getEnvelope() / getEnvelopeMax() / getEnvelopeDb() and the parameter
 *   getters: any thread, lock-free.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>

namespace dspark {

/**
 * @class EnvelopeFollower
 * @brief Attack/release envelope detector (Peak or RMS).
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum tracked channels.
 */
template <FloatType T, int MaxChannels = 16>
class EnvelopeFollower
{
public:
    /** @brief Detection law. */
    enum class Mode { Peak, RMS };

    EnvelopeFollower() noexcept { updateCoeffs(); }

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the follower. Allocation-free.
     *
     * An invalid or non-finite specification is ignored (conservative
     * no-op: the previous state is kept). Channels beyond MaxChannels are
     * clamped.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid() || !std::isfinite(spec.sampleRate)) return;
        sampleRate_.store(spec.sampleRate, std::memory_order_relaxed);
        numChannels_.store(std::clamp(spec.numChannels, 1, MaxChannels),
                           std::memory_order_relaxed);
        updateCoeffs();
        reset();
    }

    /**
     * @brief Clears all envelopes.
     *
     * Allocation-free, but it writes the same plain per-channel state the
     * processing calls own: call it from the stream owner (or with the
     * stream stopped).
     */
    void reset() noexcept
    {
        for (auto& s : state_) s = 0.0;
        for (auto& e : published_) e.store(T(0), std::memory_order_relaxed);
    }

    // -- Parameters ----------------------------------------------------------------

    /** @brief Attack time in milliseconds (default 10, floor 0.01;
     *         non-finite values are ignored). */
    void setAttack(T ms) noexcept
    {
        const double v = static_cast<double>(ms);
        if (!std::isfinite(v)) return;
        attackMs_.store(std::max(v, 0.01), std::memory_order_relaxed);
        updateCoeffs();
    }

    /** @brief Release time in milliseconds (default 150, floor 0.01;
     *         non-finite values are ignored). */
    void setRelease(T ms) noexcept
    {
        const double v = static_cast<double>(ms);
        if (!std::isfinite(v)) return;
        releaseMs_.store(std::max(v, 0.01), std::memory_order_relaxed);
        updateCoeffs();
    }

    /** @brief Peak (default) or RMS detection. Out-of-range values clamp. */
    void setMode(Mode m) noexcept
    {
        const int v = std::clamp(static_cast<int>(m), 0, 1);
        mode_.store(static_cast<Mode>(v), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------------

    /**
     * @brief Analyzes a block (read-only) and updates per-channel envelopes.
     *
     * Channels beyond the prepared count are ignored.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(),
                                 numChannels_.load(std::memory_order_relaxed));
        const int nS = buffer.getNumSamples();
        const double aAtt = attackA_.load(std::memory_order_relaxed);
        const double aRel = releaseA_.load(std::memory_order_relaxed);
        const bool peak = mode_.load(std::memory_order_relaxed) == Mode::Peak;

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* d = buffer.getChannel(ch);
            double env = state_[static_cast<size_t>(ch)];
            if (peak)
            {
                for (int i = 0; i < nS; ++i)
                {
                    const double x = std::abs(static_cast<double>(d[i]));
                    const double a = (x > env) ? aAtt : aRel;
                    env = a * env + (1.0 - a) * x;
                }
            }
            else
            {
                for (int i = 0; i < nS; ++i)
                {
                    const double x = static_cast<double>(d[i]) * d[i];
                    const double a = (x > env) ? aAtt : aRel;
                    env = a * env + (1.0 - a) * x;
                }
            }
            // Publish-time guard: a non-finite sample would park NaN in the
            // recursion for good (the branch comparison goes false forever),
            // so the poisoned channel restarts and re-measures on the next
            // block. The same test flushes fully decayed envelopes to true
            // zero (~-1000 dBFS) before they walk into double denormals.
            if (!std::isfinite(env) || env < 1e-100) env = 0.0;
            state_[static_cast<size_t>(ch)] = env;
            published_[static_cast<size_t>(ch)].store(readout(env, peak),
                                                      std::memory_order_relaxed);
        }
    }

    /**
     * @brief Single-sample path on channel 0 (for embedding in processors).
     *
     * Also publishes to the lock-free readout, so getEnvelope() stays
     * coherent with this path.
     *
     * @return The current envelope (linear amplitude).
     */
    [[nodiscard]] T processSample(T input) noexcept
    {
        const double aAtt = attackA_.load(std::memory_order_relaxed);
        const double aRel = releaseA_.load(std::memory_order_relaxed);
        const bool peak = mode_.load(std::memory_order_relaxed) == Mode::Peak;
        double env = state_[0];
        const double x = peak ? std::abs(static_cast<double>(input))
                              : static_cast<double>(input) * input;
        const double a = (x > env) ? aAtt : aRel;
        env = a * env + (1.0 - a) * x;
        if (!std::isfinite(env) || env < 1e-100) env = 0.0;
        state_[0] = env;
        const T out = readout(env, peak);
        published_[0].store(out, std::memory_order_relaxed);
        return out;
    }

    // -- Readout (lock-free, any thread) ------------------------------------------------

    /** @return Envelope of `channel` as linear amplitude. */
    [[nodiscard]] T getEnvelope(int channel = 0) const noexcept
    {
        const int n = numChannels_.load(std::memory_order_relaxed);
        channel = std::clamp(channel, 0, n - 1);
        return published_[static_cast<size_t>(channel)].load(std::memory_order_relaxed);
    }

    /** @return Loudest channel envelope (useful as a mono modulation source). */
    [[nodiscard]] T getEnvelopeMax() const noexcept
    {
        const int n = numChannels_.load(std::memory_order_relaxed);
        T m = T(0);
        for (int ch = 0; ch < n; ++ch)
            m = std::max(m, published_[static_cast<size_t>(ch)].load(std::memory_order_relaxed));
        return m;
    }

    /** @return Envelope of `channel` in dBFS (floor -120 dB). */
    [[nodiscard]] T getEnvelopeDb(int channel = 0) const noexcept
    {
        const double e = static_cast<double>(getEnvelope(channel));
        return static_cast<T>(20.0 * std::log10(std::max(e, 1e-6)));
    }

    /** @return Attack time in milliseconds. */
    [[nodiscard]] T getAttack() const noexcept
    {
        return static_cast<T>(attackMs_.load(std::memory_order_relaxed));
    }

    /** @return Release time in milliseconds. */
    [[nodiscard]] T getRelease() const noexcept
    {
        return static_cast<T>(releaseMs_.load(std::memory_order_relaxed));
    }

    /** @return Current detection law. */
    [[nodiscard]] Mode getMode() const noexcept
    {
        return mode_.load(std::memory_order_relaxed);
    }

private:
    static_assert(MaxChannels >= 1, "EnvelopeFollower needs at least one channel");

    [[nodiscard]] T readout(double env, bool peak) const noexcept
    {
        return static_cast<T>(peak ? env : std::sqrt(std::max(env, 0.0)));
    }

    void updateCoeffs() noexcept
    {
        const double fs = sampleRate_.load(std::memory_order_relaxed);
        attackA_.store(std::exp(-1.0 / (attackMs_.load(std::memory_order_relaxed) * 0.001 * fs)),
                       std::memory_order_relaxed);
        releaseA_.store(std::exp(-1.0 / (releaseMs_.load(std::memory_order_relaxed) * 0.001 * fs)),
                        std::memory_order_relaxed);
    }

    std::atomic<double> sampleRate_ { 48000.0 };
    std::atomic<int> numChannels_ { 1 };
    std::atomic<double> attackMs_ { 10.0 };
    std::atomic<double> releaseMs_ { 150.0 };
    std::atomic<double> attackA_ { 0.99 };
    std::atomic<double> releaseA_ { 0.999 };
    std::atomic<Mode> mode_ { Mode::Peak };

    std::array<double, MaxChannels> state_ {};
    std::array<std::atomic<T>, MaxChannels> published_ {};
};

} // namespace dspark
