// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralDenoiser.h
 * @brief Spectral gating denoiser with a learnable noise profile.
 *
 * Classic broadcast noise reduction on the SpectralProcessor STFT pipeline:
 *
 * 1. **Learn**: while learning is enabled, the per-bin magnitude of the
 *    incoming signal (noise only - e.g. a second of room tone) accumulates
 *    into a noise profile (running maximum with a soft average).
 * 2. **Gate**: each bin whose magnitude falls below `threshold *` profile is
 *    attenuated by up to `reduction` dB. Gains are smoothed over time per
 *    bin (fast attack, slow release) and across frequency (3-bin averaging),
 *    the standard defenses against musical noise.
 *
 * Latency is the STFT's (fftSize samples). All per-bin state is per channel.
 * The temporal release coefficient is per STFT hop, so the release time in
 * milliseconds scales with fftSize/sampleRate (about 70 ms at the 2048
 * default and 48 kHz).
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
 * @brief Learn-a-profile spectral gate (hiss/hum/room-tone reduction).
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
        gains_.assign(static_cast<size_t>(numChannels_),
                      std::vector<float>(static_cast<size_t>(numBins_), 1.0f));
        smooth_.assign(static_cast<size_t>(numBins_), 1.0f);

        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears signal state and per-bin gain memories (keeps profile). */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        stft_.reset();
        for (auto& g : gains_)
            std::fill(g.begin(), g.end(), 1.0f);
        callCounter_ = 0;
    }

    /** @brief Forgets the learned noise profile (stream-owner thread). */
    void clearProfile() noexcept
    {
        std::fill(profile_.begin(), profile_.end(), 0.0f);
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief While true, incoming audio trains the noise profile. */
    void setLearning(bool learning) noexcept
    {
        learning_.store(learning, std::memory_order_relaxed);
    }

    /** @brief Maximum attenuation of gated bins in dB [0, 40] (default 18).
     *  Non-finite values are ignored. */
    void setReduction(T db) noexcept
    {
        if (!std::isfinite(db)) return;
        reduction_.store(std::clamp(db, T(0), T(40)), std::memory_order_relaxed);
    }

    /** @brief Gate threshold over the learned profile [1, 8] (default 2).
     *  Non-finite values are ignored. */
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

        stft_.processBlock(buffer, [this, learning, floorGain, thresh, nChEff](T* bins, int numBins)
        {
            auto& chGain = gains_[static_cast<size_t>(callCounter_ % nChEff)];
            ++callCounter_;

            // Per-bin raw gate decision.
            for (int k = 0; k < numBins; ++k)
            {
                const float re = static_cast<float>(bins[2 * k]);
                const float im = static_cast<float>(bins[2 * k + 1]);
                const float mag = std::sqrt(re * re + im * im);

                if (learning)
                {
                    // Profile: peak-hold with a gentle average pull-up.
                    auto& p = profile_[static_cast<size_t>(k)];
                    p = std::max(p * 0.995f + mag * 0.005f, std::max(p, mag * 0.8f));
                }

                const float open = profile_[static_cast<size_t>(k)] * thresh;
                smooth_[static_cast<size_t>(k)] = (mag > open) ? 1.0f : floorGain;
            }

            // Frequency smoothing (3-bin average) defends against isolated
            // flickering bins - the source of musical noise.
            for (int k = 0; k < numBins; ++k)
            {
                const float a = smooth_[static_cast<size_t>(std::max(k - 1, 0))];
                const float b = smooth_[static_cast<size_t>(k)];
                const float c = smooth_[static_cast<size_t>(std::min(k + 1, numBins - 1))];
                float target = (a + b + c) * (1.0f / 3.0f);

                // Temporal smoothing: instant attack (open fast), slow release.
                auto& g = chGain[static_cast<size_t>(k)];
                g = (target > g) ? target : (g * 0.85f + target * 0.15f);

                bins[2 * k] = static_cast<T>(static_cast<float>(bins[2 * k]) * g);
                bins[2 * k + 1] = static_cast<T>(static_cast<float>(bins[2 * k + 1]) * g);
            }
        });
    }

private:
    SpectralProcessor<T> stft_;
    int numChannels_ = 0;
    int numBins_ = 0;
    std::atomic<bool> prepared_ { false };

    std::vector<float> profile_;               ///< Learned noise magnitude per bin.
    std::vector<std::vector<float>> gains_;    ///< Per-channel smoothed gains.
    std::vector<float> smooth_;                ///< Scratch raw/frequency-smoothed gains.
    int callCounter_ = 0;

    std::atomic<bool> learning_ { false };
    std::atomic<T> reduction_ { T(18) };
    std::atomic<T> threshold_ { T(2) };
};

} // namespace dspark
