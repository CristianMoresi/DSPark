// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file EnvelopeFollower.h
 * @brief Peak/RMS envelope follower with asymmetric attack/release.
 *
 * The detector every dynamics processor carries internally, exposed as a
 * public building block: a one-pole follower with independent attack and
 * release time constants, in Peak mode (rectified signal) or RMS mode
 * (squared signal, square-rooted on readout). Use it as a sidechain
 * detector, a modulation source (with ModulationRouter), or a meter.
 *
 * Per-channel envelopes are tracked; readouts are lock-free from any
 * thread. processSample() offers the single-value path for embedding in
 * other processors.
 *
 * Dependencies: AudioSpec.h, AudioBuffer.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

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

    // -- Lifecycle ---------------------------------------------------------------

    /** @brief Prepares the follower (no allocation). */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (spec.sampleRate <= 0.0) return;
        sampleRate_ = spec.sampleRate;
        numChannels_ = std::clamp(spec.numChannels, 1, MaxChannels);
        updateCoeffs();
        reset();
    }

    /** @brief Clears all envelopes. RT-safe. */
    void reset() noexcept
    {
        for (auto& s : state_) s = 0.0;
        for (auto& e : published_) e.store(T(0), std::memory_order_relaxed);
    }

    // -- Parameters ----------------------------------------------------------------

    /** @brief Attack time in milliseconds (default 10). */
    void setAttack(T ms) noexcept
    {
        attackMs_ = std::max(static_cast<double>(ms), 0.01);
        updateCoeffs();
    }

    /** @brief Release time in milliseconds (default 150). */
    void setRelease(T ms) noexcept
    {
        releaseMs_ = std::max(static_cast<double>(ms), 0.01);
        updateCoeffs();
    }

    /** @brief Peak (default) or RMS detection. */
    void setMode(Mode m) noexcept { mode_ = m; }

    // -- Processing -------------------------------------------------------------------

    /** @brief Analyzes a block (read-only) and updates per-channel envelopes. */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* d = buffer.getChannel(ch);
            double env = state_[static_cast<size_t>(ch)];
            if (mode_ == Mode::Peak)
            {
                for (int i = 0; i < nS; ++i)
                {
                    const double x = std::abs(static_cast<double>(d[i]));
                    const double a = (x > env) ? attackA_ : releaseA_;
                    env = a * env + (1.0 - a) * x;
                }
            }
            else
            {
                for (int i = 0; i < nS; ++i)
                {
                    const double x = static_cast<double>(d[i]) * d[i];
                    const double a = (x > env) ? attackA_ : releaseA_;
                    env = a * env + (1.0 - a) * x;
                }
            }
            state_[static_cast<size_t>(ch)] = env;
            published_[static_cast<size_t>(ch)].store(readout(env), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Single-sample path on channel 0 (for embedding in processors).
     * @return The current envelope (linear amplitude).
     */
    [[nodiscard]] T processSample(T input) noexcept
    {
        double& env = state_[0];
        const double x = (mode_ == Mode::Peak)
            ? std::abs(static_cast<double>(input))
            : static_cast<double>(input) * input;
        const double a = (x > env) ? attackA_ : releaseA_;
        env = a * env + (1.0 - a) * x;
        return readout(env);
    }

    // -- Readout (lock-free, any thread) ------------------------------------------------

    /** @return Envelope of `channel` as linear amplitude. */
    [[nodiscard]] T getEnvelope(int channel = 0) const noexcept
    {
        channel = std::clamp(channel, 0, numChannels_ - 1);
        return published_[static_cast<size_t>(channel)].load(std::memory_order_relaxed);
    }

    /** @return Loudest channel envelope (useful as a mono modulation source). */
    [[nodiscard]] T getEnvelopeMax() const noexcept
    {
        T m = T(0);
        for (int ch = 0; ch < numChannels_; ++ch)
            m = std::max(m, published_[static_cast<size_t>(ch)].load(std::memory_order_relaxed));
        return m;
    }

    /** @return Envelope of `channel` in dBFS (floor -120 dB). */
    [[nodiscard]] T getEnvelopeDb(int channel = 0) const noexcept
    {
        const double e = static_cast<double>(getEnvelope(channel));
        return static_cast<T>(20.0 * std::log10(std::max(e, 1e-6)));
    }

private:
    [[nodiscard]] T readout(double env) const noexcept
    {
        return static_cast<T>((mode_ == Mode::RMS) ? std::sqrt(std::max(env, 0.0)) : env);
    }

    void updateCoeffs() noexcept
    {
        attackA_ = std::exp(-1.0 / (attackMs_ * 0.001 * sampleRate_));
        releaseA_ = std::exp(-1.0 / (releaseMs_ * 0.001 * sampleRate_));
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 1;
    double attackMs_ = 10.0;
    double releaseMs_ = 150.0;
    double attackA_ = 0.99;
    double releaseA_ = 0.999;
    Mode mode_ = Mode::Peak;

    std::array<double, MaxChannels> state_ {};
    std::array<std::atomic<T>, MaxChannels> published_ {};
};

} // namespace dspark
