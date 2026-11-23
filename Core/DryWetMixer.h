// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file DryWetMixer.h
 * @brief Real-time safe dry/wet mixer with parameter smoothing and mix laws.
 *
 * Captures a copy of the dry (unprocessed) signal before an effect runs,
 * then blends it with the wet (processed) signal. Includes block-level
 * parameter smoothing to prevent zipper noise during DAW automation, and
 * supports both Linear and Equal Power mixing laws.
 *
 * @note If your effect introduces algorithmic latency (e.g., FIR filters),
 * call setLatencyCompensation(effectLatency) during setup. Then pass the original,
 * undelayed input to pushDry(); the mixer delays its own copy to align both paths
 * and prevent phase cancellation (comb filtering).
 *
 * Threading (SPSC model, see docs/threading.md): pushDry() / mixWet() /
 * reset() belong to the processing thread (single stream owner), and so do
 * the dry-snapshot readouts getDryChannel() / getDryCapturedSamples(), which
 * expose plain processing-thread state. setMixRule() belongs to the control
 * thread (ONE non-audio writer; single relaxed word read once per mixWet()).
 * getMixRule() reads it back from any thread; prepare() and
 * setLatencyCompensation() (and getLatencyCompensation()) allocate/touch
 * setup state and belong to the setup thread. Moves are setup-time only.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, and the C++20 standard-library
 * `algorithm`, `array`, `atomic`, `cmath`, `cstddef`, `numbers` and `vector`
 * headers.
 *
 * @code
 * dspark::DryWetMixer<float> mixer;
 *
 * void prepare(const dspark::AudioSpec& spec) {
 *     mixer.prepare(spec);
 *     mixer.setMixRule(dspark::DryWetMixer<float>::MixRule::EqualPower);
 * }
 *
 * void process(dspark::AudioBufferView<float> buffer) {
 *     mixer.pushDry(buffer);
 *     applyEffect(buffer);        // Modifies buffer in-place (now wet)
 *     mixer.mixWet(buffer, 0.5f); // Smoothly transitions to 50% wet
 * }
 * @endcode
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

// DSPARK_RESTRICT is normally provided by SimdOps.h (pulled in via AudioBuffer.h).
// Guard against redefinition so this header stays self-contained if included alone.
#ifndef DSPARK_RESTRICT
  #if defined(__clang__) || defined(__GNUC__)
    #define DSPARK_RESTRICT __restrict__
  #elif defined(_MSC_VER)
    #define DSPARK_RESTRICT __restrict
  #else
    #define DSPARK_RESTRICT
  #endif
#endif

namespace dspark {

/**
 * @class DryWetMixer
 * @brief Pre-allocated, SIMD-friendly dry/wet blender for real-time audio.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of channels (compile-time bound).
 */
template <typename T, int MaxChannels = 16>
class DryWetMixer
{
public:
    /** @brief Defines the mathematical curve used for mixing. */
    enum class MixRule {
        /** Linear crossfade. Best for highly correlated signals (EQ, saturation). */
        Linear,
        /** Constant power crossfade. Best for uncorrelated signals (Reverb, Delay). */
        EqualPower,
        /** The dry stays whole up to half and then falls; the wet rises to whole at half:
            at 0.5 the output is dry plus wet, as in hardware that adds the two. */
        FullDryToHalf
    };

    DryWetMixer() noexcept = default;

    // The atomic mix rule makes the implicit move operations vanish. Provide
    // manual moves so the mixer keeps working inside movable owners; moves
    // are single-threaded setup-time relocations, so plain relaxed transfers
    // of the atomic are sufficient. Copying is deleted by the owned buffers.
    DryWetMixer(DryWetMixer&& other) noexcept
        : dryBuffer_(std::move(other.dryBuffer_)),
          capturedSamples_(other.capturedSamples_),
          delayHist_(std::move(other.delayHist_)),
          latencySamples_(other.latencySamples_),
          histPos_(other.histPos_),
          mixRule_(other.mixRule_.load(std::memory_order_relaxed)),
          currentMix_(other.currentMix_),
          maxMixStep_(other.maxMixStep_)
    {}

    DryWetMixer& operator=(DryWetMixer&& other) noexcept
    {
        if (this == &other) return *this;
        dryBuffer_       = std::move(other.dryBuffer_);
        capturedSamples_ = other.capturedSamples_;
        delayHist_       = std::move(other.delayHist_);
        latencySamples_  = other.latencySamples_;
        histPos_         = other.histPos_;
        mixRule_.store(other.mixRule_.load(std::memory_order_relaxed),
                       std::memory_order_relaxed);
        currentMix_      = other.currentMix_;
        maxMixStep_      = other.maxMixStep_;
        return *this;
    }

    DryWetMixer(const DryWetMixer&)            = delete;
    DryWetMixer& operator=(const DryWetMixer&) = delete;

    /**
     * @brief Allocates the internal dry buffer for the given audio spec.
     * @param spec Audio environment specification.
     */
    void prepare(const AudioSpec& spec)
    {
        dryBuffer_.resize(spec.numChannels, spec.maxBlockSize);

        // Mix changes ramp over at least kMinRampSeconds for a full-scale
        // move, whatever the host block size (an invalid rate keeps the
        // one-block ramp).
        maxMixStep_ = spec.isValid()
            ? static_cast<T>(1.0 / std::max(1.0, spec.sampleRate * kMinRampSeconds))
            : T(1);

        // If latency compensation was configured before prepare(), the delay
        // history was sized with 0 channels; rebuild it now that the channel
        // count is known (otherwise pushDry would index a 0-channel buffer).
        if (latencySamples_ > 0)
        {
            delayHist_.resize(spec.numChannels, latencySamples_);
            histPos_ = 0;
        }

        reset();
    }

    /** @brief Resets the internal buffer and smoothing states to zero. */
    void reset() noexcept
    {
        dryBuffer_.clear();
        if (delayHist_.getNumSamples() > 0) delayHist_.clear();
        histPos_ = 0;
        capturedSamples_ = 0;  // No valid dry snapshot until the next pushDry().
        currentMix_ = T(-1); // Forces immediate jump on first use
    }

    /**
     * @brief Sets the mathematical curve to use during the mix phase.
     *
     * Control thread (single-word relaxed atomic). mixWet() reads the rule
     * once per block, so a change lands at the next block boundary.
     *
     * @param rule The chosen MixRule (out-of-range values read as Linear).
     */
    void setMixRule(MixRule rule) noexcept
    {
        const int r = static_cast<int>(rule);
        mixRule_.store(r >= 0 && r <= 2 ? rule : MixRule::Linear, std::memory_order_relaxed);
    }

    /** @return The rule of the mix (any thread). */
    [[nodiscard]] MixRule getMixRule() const noexcept { return mixRule_.load(std::memory_order_relaxed); }

    /**
     * @brief Delays the captured dry signal to compensate for an effect's
     *        internal latency (e.g. an oversampler's FIR group delay).
     *
     * When the wet path introduces @p samples of delay, the dry copy must be
     * delayed by the same amount; otherwise blending dry+wet (mix < 1, Delta or
     * adaptive modes) produces comb filtering. Pass 0 to disable (default).
     *
     * @note **NOT Real-Time Safe.** Allocates the internal delay history.
     *       Call from the setup thread (typically right after prepare()).
     * @param samples Latency in samples at this mixer's sample rate (>= 0).
     */
    void setLatencyCompensation(int samples)
    {
        samples = std::max(0, samples);
        if (samples == latencySamples_) return;
        latencySamples_ = samples;
        delayHist_.resize(dryBuffer_.getNumChannels(), samples);
        histPos_ = 0;
    }

    /** @brief Returns the configured dry-path latency compensation in samples. */
    [[nodiscard]] int getLatencyCompensation() const noexcept { return latencySamples_; }

    /**
     * @brief Captures a snapshot of the dry (unprocessed) signal.
     *
     * Channels beyond the prepared channel count are ignored; a shorter
     * input than the prepared block size shortens the valid snapshot (and
     * therefore the region the next mixWet() call blends).
     *
     * @param input The unprocessed audio buffer. Accepts const or mutable
     * views implicitly (AudioBufferView converts T to const T).
     */
    void pushDry(const AudioBufferView<const T>& input) noexcept
    {
        if (dryBuffer_.getNumSamples() == 0) return; // Prevent ops if unprepared

        const int chCount  = std::min(input.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(input.getNumSamples(), dryBuffer_.getNumSamples());

        if (latencySamples_ <= 0)
        {
            for (int ch = 0; ch < chCount; ++ch)
            {
                const T* DSPARK_RESTRICT src = input.getChannel(ch);
                T* DSPARK_RESTRICT dst       = dryBuffer_.getChannel(ch);
                std::copy_n(src, nSamples, dst);
            }
        }
        else
        {
            // Latency-compensated capture: route the dry through a per-channel
            // circular delay of exactly latencySamples_ so it stays time-aligned
            // with a wet path that incurs the same delay.
            const int D = latencySamples_;
            for (int ch = 0; ch < chCount; ++ch)
            {
                const T* DSPARK_RESTRICT src = input.getChannel(ch);
                T* DSPARK_RESTRICT dst       = dryBuffer_.getChannel(ch);
                T* DSPARK_RESTRICT hist      = delayHist_.getChannel(ch);
                int pos = histPos_;
                for (int i = 0; i < nSamples; ++i)
                {
                    dst[i]    = hist[pos];   // sample written D steps ago
                    hist[pos] = src[i];
                    if (++pos == D) pos = 0;
                }
            }
            histPos_ = (histPos_ + nSamples) % D;
        }

        capturedSamples_ = nSamples;
    }

    /**
     * @brief Blends the stored dry signal with the current (wet) buffer in-place.
     *
     * Automatically applies sample-accurate linear interpolation to the mix
     * proportion to prevent zipper noise when the parameter changes. A change
     * ramps across the block, but never faster than a full-scale move in
     * 20 ms: with small host blocks (32 samples is under a millisecond) a
     * one-block ramp would click on uncorrelated dry/wet pairs, so the ramp
     * then continues into the following blocks.
     *
     * Only the region covered by the last pushDry() snapshot is blended:
     * without a captured snapshot (e.g. right after reset()) this is a no-op
     * and the buffer stays fully wet. Channels beyond the prepared channel
     * count also pass through untouched.
     *
     * @param wetBuffer     The processed buffer (modified in-place).
     * @param targetMix     Target mix amount: 0.0 = fully dry, 1.0 = fully wet.
     */
    void mixWet(AudioBufferView<T> wetBuffer, T targetMix) noexcept
    {
        // Handle invalid floating point inputs (NaN) safely
        if (std::isnan(targetMix)) targetMix = T(0);
        targetMix = std::clamp(targetMix, T(0), T(1));

        const int chCount  = std::min(wetBuffer.getNumChannels(), dryBuffer_.getNumChannels());
        const int nSamples = std::min(wetBuffer.getNumSamples(), capturedSamples_);

        if (nSamples == 0 || chCount == 0) return;

        // Initialize smoothing state on first run
        if (currentMix_ < T(0)) currentMix_ = targetMix;

        const bool needsSmoothing = std::abs(currentMix_ - targetMix) > T(1e-5);
        T mixStep = T(0);
        bool rateLimited = false;
        if (needsSmoothing)
        {
            mixStep = (targetMix - currentMix_) / T(nSamples);
            if (std::abs(mixStep) > maxMixStep_)
            {
                mixStep = mixStep > T(0) ? maxMixStep_ : -maxMixStep_;
                rateLimited = true;
            }
        }
        const MixRule rule = mixRule_.load(std::memory_order_relaxed);

        // Every channel ramps from the same smoothed origin. The ramps below
        // are indexed (mix0 + mixStep * i) rather than accumulated: one
        // rounding per sample instead of a drift that grows with the block
        // (a serial accumulator can overshoot the [0, 1] range on long
        // blocks), and no loop-carried dependency in the way of the
        // auto-vectoriser. A settled mix within the 1e-5 dead band renders
        // at the target itself, so no residual offset is ever held.
        const T mix0 = needsSmoothing ? currentMix_ : targetMix;

        for (int ch = 0; ch < chCount; ++ch)
        {
            T* DSPARK_RESTRICT wetData       = wetBuffer.getChannel(ch);
            const T* DSPARK_RESTRICT dryData = dryBuffer_.getChannel(ch);

            if (rule == MixRule::EqualPower)
            {
                if (!needsSmoothing)
                {
                    // Static mix: hoist both square roots out of the loop;
                    // the body becomes a single FMA per sample (vectorizable).
                    const T w = std::sqrt(mix0);
                    const T d = std::sqrt(T(1) - mix0);
                    for (int i = 0; i < nSamples; ++i)
                        wetData[i] = dryData[i] * d + wetData[i] * w;
                }
                else
                {
                    for (int i = 0; i < nSamples; ++i)
                    {
                        // Exact constant-power law: w^2 + d^2 = 1. The max()
                        // guards keep float rounding at the ramp ends from
                        // pushing a radicand a few ulp negative, where sqrt()
                        // would inject NaN into the output.
                        const T m = mix0 + mixStep * T(i);
                        const T w = std::sqrt(std::max(T(0), m));
                        const T d = std::sqrt(std::max(T(0), T(1) - m));
                        wetData[i] = dryData[i] * d + wetData[i] * w;
                    }
                }
            }
            else if (rule == MixRule::FullDryToHalf)
            {
                // dry min(1, 2 (1 - m)), wet min(1, 2 m).
                for (int i = 0; i < nSamples; ++i)
                {
                    const T m = needsSmoothing ? mix0 + mixStep * T(i) : mix0;
                    const T w = std::min(T(1), T(2) * m);
                    const T d = std::min(T(1), T(2) * (T(1) - m));
                    wetData[i] = dryData[i] * d + wetData[i] * w;
                }
            }
            else // MixRule::Linear
            {
                if (!needsSmoothing)
                {
                    const T w = mix0;
                    const T d = T(1) - w;
                    for (int i = 0; i < nSamples; ++i)
                        wetData[i] = dryData[i] * d + wetData[i] * w;
                }
                else
                {
                    for (int i = 0; i < nSamples; ++i)
                    {
                        const T w = mix0 + mixStep * T(i);
                        const T d = T(1) - w;
                        wetData[i] = dryData[i] * d + wetData[i] * w;
                    }
                }
            }
        }

        // Update the persistent state after the block is processed: a
        // rate-limited ramp resumes from where this block left it.
        currentMix_ = rateLimited
            ? std::clamp(mix0 + mixStep * T(nSamples), T(0), T(1))
            : targetMix;
    }

    /**
     * @brief Retrieves a read-only pointer to the captured dry channel data.
     * @param ch Channel index (0-based).
     * @return Pointer to dry samples, or nullptr if out of bounds.
     */
    [[nodiscard]] const T* getDryChannel(int ch) const noexcept
    {
        if (ch < 0 || ch >= dryBuffer_.getNumChannels()) return nullptr;
        return dryBuffer_.getChannel(ch);
    }

    /** @brief Returns the internal capacity of channels in the dry buffer. */
    [[nodiscard]] int getDryNumChannels() const noexcept { return dryBuffer_.getNumChannels(); }

    /** @brief Returns the number of samples valid from the last pushDry() call. */
    [[nodiscard]] int getDryCapturedSamples() const noexcept { return capturedSamples_; }

private:
    AudioBuffer<T, MaxChannels> dryBuffer_;
    int capturedSamples_ = 0;

    // Optional dry-path latency compensation (circular delay). Disabled (0) by default.
    AudioBuffer<T, MaxChannels> delayHist_;
    int latencySamples_ = 0;
    int histPos_        = 0;

    std::atomic<MixRule> mixRule_ { MixRule::Linear };
    T currentMix_ = T(-1); // Internal state for parameter smoothing
    T maxMixStep_ = T(1);  // Per-sample mix slew limit (set in prepare())

    static constexpr double kMinRampSeconds = 0.02;
};

namespace detail {

/**
 * @class DryPhase
 * @brief Gives the dry side of a mix the phase that a high-pass leaves in
 *        the wet side, and none of its magnitude.
 *
 * A first-order all-pass that leads: two one-pole high-passes at its corner
 * turn the phase exactly as much, and a second-order Butterworth high-pass
 * within a few degrees of it with the corner at kButterworthShare of its
 * own. A wet side behind such a filter and a dry side without it are a
 * quarter of a turn or more apart at the bottom, and their mix loses there
 * what neither side loses by itself.
 *
 * It fades in and out over 20 ms (aim()). Faded out it returns its input to
 * the bit, and it goes on following it, so that it starts from where the
 * programme is.
 *
 * Processing thread, except prepare() (setup thread; allocates).
 */
class DryPhase
{
public:
    /** The corner for a second-order Butterworth high-pass, over that of the filter. */
    static constexpr double kButterworthShare = 0.8;

    /**
     * @brief Sizes the state (allocates).
     * @param sampleRate The rate it runs at.
     * @param channels   The channels it keeps.
     */
    void prepare(double sampleRate, int channels)
    {
        sampleRate_ = sampleRate;
        step_ = 1.0 / std::max(1.0, sampleRate * 0.02);
        state_.assign(static_cast<std::size_t>(std::max(channels, 0)), {});
        corner_ = -1.0;
    }

    /** @brief Clears the memory (the amount stays). */
    void reset() noexcept { std::fill(state_.begin(), state_.end(), std::array<double, 2> {}); }

    /**
     * @brief Moves the corner (a tangent, when it changes).
     * @param hz The corner in hertz.
     */
    void setCorner(double hz) noexcept
    {
        if (hz == corner_) return;
        corner_ = hz;
        const double t = std::tan(std::numbers::pi * std::clamp(hz / sampleRate_, 0.0, 0.45));
        pole_ = (1.0 - t) / (1.0 + t);
    }

    /**
     * @brief Where the amount goes.
     * @param on Whether the dry side is turned.
     */
    void aim(bool on) noexcept { target_ = on ? 1.0 : 0.0; }

    /** @brief Puts the amount where it goes, without the fade. */
    void land() noexcept { amount_ = target_; }

    /** @brief Moves the amount by a frame and returns it. */
    [[nodiscard]] double advance() noexcept
    {
        if (amount_ != target_)
            amount_ = (amount_ < target_) ? std::min(target_, amount_ + step_) : std::max(target_, amount_ - step_);
        return amount_;
    }

    /**
     * @brief A sample of a channel, with the amount that advance() returned.
     * @param channel The channel (one that was not prepared passes).
     * @param x       The dry sample.
     * @param amount  The amount of this frame.
     */
    template <typename T>
    [[nodiscard]] T processSample(int channel, T x, double amount) noexcept
    {
        if (channel < 0 || static_cast<std::size_t>(channel) >= state_.size()) return x;
        auto& s = state_[static_cast<std::size_t>(channel)];
        const double in = static_cast<double>(x);
        double y = pole_ * (in + s[1]) - s[0];
        if (!std::isfinite(y))
        {
            s = {};
            return x;
        }
        if (std::abs(y) < 1e-30) y = 0.0;
        s[0] = in;
        s[1] = y;
        if (amount == 0.0) return x;
        return static_cast<T>(amount == 1.0 ? y : in + amount * (y - in));
    }

    /**
     * @brief A stretch of a buffer, in place.
     * @param buffer The dry side.
     */
    template <typename T>
    void process(AudioBufferView<T> buffer) noexcept
    {
        const int channels = std::min(buffer.getNumChannels(), static_cast<int>(state_.size()));
        const int samples = buffer.getNumSamples();
        for (int i = 0; i < samples; ++i)
        {
            const double amount = advance();
            for (int ch = 0; ch < channels; ++ch)
            {
                T* data = buffer.getChannel(ch);
                data[i] = processSample(ch, data[i], amount);
            }
        }
    }

private:
    std::vector<std::array<double, 2>> state_;   ///< Per channel: the last input and the last output.
    double sampleRate_ = 48000.0, step_ = 1.0 / 960.0;
    double corner_ = -1.0, pole_ = 0.0;
    double amount_ = 0.0, target_ = 0.0;
};

} // namespace detail

} // namespace dspark
