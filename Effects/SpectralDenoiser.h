// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralDenoiser.h
 * @brief Spectral noise reduction with a learnable noise profile.
 *
 * Broadcast-style noise reduction on the SpectralProcessor STFT pipeline:
 *
 * 1. **Learn**: while learning is enabled, the per-bin power of the incoming
 *    signal (noise only - e.g. a second of room tone) accumulates into the
 *    noise profile: the running mean noise power per bin.
 * 2. **Reduce**: each bin gets the Wiener gain xi / (1 + xi), where the a
 *    priori SNR xi is the decision-directed estimate of Ephraim and Malah
 *    (weight 0.98 per hop) against the profile scaled by `threshold^2`.
 *    Gains never fall below the `reduction` floor. The decision-directed
 *    estimate follows onsets at once but ignores random noise flicker, so
 *    the residual noise stays a smooth, attenuated copy of the noise instead
 *    of the "musical noise" (isolated tones) a hard per-bin gate leaves.
 *
 * Latency is the STFT's (fftSize samples). All per-bin state is per channel;
 * the profile is shared. The decision-directed weight is per STFT hop, so its
 * memory in milliseconds scales with fftSize/sampleRate.
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored). prepare() is setup-thread
 * only (allocates; invalid specs are ignored and an unprepared instance
 * passes audio through). reset() and clearProfile() belong to the stream
 * owner (the profile is written by the audio thread while learning).
 * getState()/setState() are setup/UI threads. Channels beyond the prepared
 * count pass through dry (SpectralProcessor behaviour).
 *
 * Dependencies: Core/SpectralProcessor.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/DspMath.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SpectralProcessor.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class SpectralDenoiser
 * @brief Learn-a-profile spectral noise reduction (hiss/hum/room-tone).
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SpectralDenoiser
{
public:
    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the STFT pipeline and the per-channel bin state.
     *
     * Invalid specs (non-positive/non-finite rate, block size or channel
     * count) are ignored: the previous state is kept and an unprepared
     * instance stays pass-through. fftSize is sanitized by the STFT engine
     * (power of two in [4, 1 << 20]).
     *
     * @param spec    Audio environment specification.
     * @param fftSize STFT size (default 2048; larger = finer hum notching).
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        numChannels_ = spec.numChannels;
        stft_.prepare(spec, fftSize, fftSize / 4);
        numBins_ = stft_.getNumBins();

        profile_.assign(static_cast<size_t>(numBins_), 0.0f);
        learnFrames_ = 0;
        cleanPower_.assign(static_cast<size_t>(numChannels_),
                           std::vector<float>(static_cast<size_t>(numBins_), 0.0f));

        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears signal state and per-bin gain memories (keeps profile). */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        stft_.reset();
        for (auto& c : cleanPower_)
            std::fill(c.begin(), c.end(), 0.0f);
        callCounter_ = 0;
    }

    /** @brief Forgets the learned noise profile (stream-owner thread). */
    void clearProfile() noexcept
    {
        std::fill(profile_.begin(), profile_.end(), 0.0f);
        learnFrames_ = 0;
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief While true, incoming audio trains the noise profile. */
    void setLearning(bool learning) noexcept
    {
        learning_.store(learning, std::memory_order_relaxed);
    }

    /** @brief Maximum attenuation of noise bins in dB [0, 40] (default 18):
     *  the floor of the per-bin gain. Non-finite values are ignored. */
    void setReduction(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        reduction_.store(std::clamp(db, T(0), T(40)), std::memory_order_relaxed);
    }

    /** @brief Noise over-subtraction factor over the learned profile, in
     *  magnitude [1, 8] (default 2): the gain rule sees the learned noise
     *  scaled by it, so higher values remove more noise and more low-level
     *  signal. Non-finite values are ignored. */
    void setThreshold(T factor) noexcept
    {
        if (!std::isfinite(factor)) return;
        threshold_.store(std::clamp(factor, T(1), T(8)), std::memory_order_relaxed);
    }

    [[nodiscard]] bool getLearning() const noexcept { return learning_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getReduction() const noexcept { return reduction_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }

    /** @brief Latency in samples (the STFT pipeline's). */
    [[nodiscard]] int getLatency() const noexcept { return stft_.getLatency(); }

    /** @brief Serializes the parameter state (the learned profile is material-
     *  dependent content, not a preset, and is intentionally not included). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DNSE"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("reduction", static_cast<float>(reduction_.load(std::memory_order_relaxed)));
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DNSE")) return false;
        setReduction(static_cast<T>(r.read("reduction", 18.0f)));
        setThreshold(static_cast<T>(r.read("threshold", 2.0f)));
        return true;
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Processes a block in-place. Pass-through until prepare() succeeds. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;

        const bool learning = learning_.load(std::memory_order_relaxed);
        const float floorGain = std::pow(
            10.0f, -static_cast<float>(reduction_.load(std::memory_order_relaxed)) / 20.0f);
        const float thresh = static_cast<float>(threshold_.load(std::memory_order_relaxed));

        // The STFT invokes the callback once per PROCESSED channel per hop
        // (in channel order) - and it processes min(buffer, prepared)
        // channels. Modulo by that same effective count, reset per block, or
        // a narrow buffer over a wider spec would rotate the per-channel
        // gain memories between hops (channel 0 alternating onto channel 1's
        // release state - measured 0.004 divergence versus a mono-prepared
        // twin before this fix).
        const int nChEff = std::max(1, std::min(buffer.getNumChannels(), numChannels_));
        callCounter_ = 0;

        // Over-subtraction: the learned noise power is scaled by threshold^2.
        const float overSub = thresh * thresh;

        stft_.processBlock(buffer, [this, learning, floorGain, overSub, nChEff](T* bins, int numBins)
        {
            auto& clean = cleanPower_[static_cast<size_t>(callCounter_ % nChEff)];
            ++callCounter_;

            // The profile is the running MEAN noise power per bin (the
            // cumulative average over every learned frame, then an
            // exponential average once kMaxLearnFrames are in).
            float learnRate = 0.0f;
            if (learning)
            {
                learnFrames_ = std::min(learnFrames_ + 1, kMaxLearnFrames);
                learnRate = 1.0f / static_cast<float>(learnFrames_);
            }

            for (int k = 0; k < numBins; ++k)
            {
                const float re = static_cast<float>(bins[2 * k]);
                const float im = static_cast<float>(bins[2 * k + 1]);
                const float power = re * re + im * im;

                auto& noise = profile_[static_cast<size_t>(k)];
                if (learning)
                    noise += learnRate * (power - noise);

                // Decision-directed Wiener gain (Ephraim-Malah a priori SNR):
                // the a priori SNR mixes the previous frame's clean estimate
                // with this frame's excess power, so it follows speech and
                // music onsets at once while random noise flicker cannot open
                // a bin (the hard per-bin gate did exactly that: musical noise).
                float gain = 1.0f;
                const float lambda = overSub * noise;
                auto& prevClean = clean[static_cast<size_t>(k)];
                if (lambda > 0.0f)
                {
                    const float post = power / lambda;
                    const float prio = std::max(kDDAlpha * prevClean / lambda
                                                + (1.0f - kDDAlpha) * std::max(post - 1.0f, 0.0f),
                                                kMinPrioriSnr);
                    gain = std::max(prio / (1.0f + prio), floorGain);
                }
                prevClean = gain * gain * power;

                bins[2 * k] = static_cast<T>(re * gain);
                bins[2 * k + 1] = static_cast<T>(im * gain);
            }
        });
    }

private:
    SpectralProcessor<T> stft_;
    int numChannels_ = 0;
    int numBins_ = 0;
    std::atomic<bool> prepared_ { false };

    static constexpr float kDDAlpha = 0.98f;           ///< Decision-directed weight per hop.
    static constexpr float kMinPrioriSnr = 0.003162f;  ///< -25 dB a priori SNR floor.
    static constexpr int kMaxLearnFrames = 4096;       ///< Averaging horizon of the profile.

    std::vector<float> profile_;                  ///< Learned mean noise power per bin.
    int learnFrames_ = 0;                         ///< Frames in the running mean.
    std::vector<std::vector<float>> cleanPower_;  ///< Per-channel previous clean power estimate.
    int callCounter_ = 0;

    std::atomic<bool> learning_ { false };
    std::atomic<T> reduction_ { T(18) };
    std::atomic<T> threshold_ { T(2) };
};

} // namespace dspark
