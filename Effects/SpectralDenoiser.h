// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpectralDenoiser.h
 * @brief Spectral gating denoiser with a learnable noise profile.
 *
 * Classic broadcast noise reduction on the SpectralProcessor STFT pipeline:
 *
 * 1. **Learn**: while learning is enabled, the per-bin magnitude of the
 *    incoming signal (noise only — e.g. a second of room tone) accumulates
 *    into a noise profile (running maximum with a soft average).
 * 2. **Gate**: each bin whose magnitude falls below `threshold ×` profile is
 *    attenuated by up to `reduction` dB. Gains are smoothed over time per
 *    bin (fast attack, slow release) and across frequency (3-bin averaging),
 *    the standard defenses against musical noise.
 *
 * Latency is the STFT's (fftSize samples). All per-bin state is per channel.
 *
 * Dependencies: SpectralProcessor.h, AudioSpec.h, AudioBuffer.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SpectralProcessor.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
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
     * @param spec    Audio environment specification.
     * @param fftSize STFT size (default 2048; larger = finer hum notching).
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048)
    {
        if (spec.sampleRate <= 0.0 || spec.numChannels < 1) return;
        numChannels_ = spec.numChannels;
        stft_.prepare(spec, fftSize, fftSize / 4);
        numBins_ = stft_.getNumBins();

        profile_.assign(static_cast<size_t>(numBins_), 0.0f);
        gains_.assign(static_cast<size_t>(numChannels_),
                      std::vector<float>(static_cast<size_t>(numBins_), 1.0f));
        smooth_.assign(static_cast<size_t>(numBins_), 1.0f);

        prepared_ = true;
        reset();
    }

    /** @brief Clears signal state and per-bin gain memories (keeps profile). */
    void reset() noexcept
    {
        if (!prepared_) return;
        stft_.reset();
        for (auto& g : gains_)
            std::fill(g.begin(), g.end(), 1.0f);
        callCounter_ = 0;
    }

    /** @brief Forgets the learned noise profile. */
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

    /** @brief Maximum attenuation of gated bins in dB [0, 40] (default 18). */
    void setReduction(T db) noexcept
    {
        reduction_.store(std::clamp(db, T(0), T(40)), std::memory_order_relaxed);
    }

    /** @brief Gate threshold over the learned profile [1, 8] (default 2). */
    void setThreshold(T factor) noexcept
    {
        threshold_.store(std::clamp(factor, T(1), T(8)), std::memory_order_relaxed);
    }

    /** @brief Latency in samples (the STFT pipeline's). */
    [[nodiscard]] int getLatency() const noexcept { return stft_.getLatency(); }

    /** @brief Serializes the parameter state (the learned profile is material-
     *  dependent content, not a preset, and is intentionally not included). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DNSE"), 1);
        w.write("reduction", reduction_.load(std::memory_order_relaxed));
        w.write("threshold", threshold_.load(std::memory_order_relaxed));
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

    /** @brief Processes a block in-place. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;

        const bool learning = learning_.load(std::memory_order_relaxed);
        const float floorGain = std::pow(
            10.0f, -static_cast<float>(reduction_.load(std::memory_order_relaxed)) / 20.0f);
        const float thresh = static_cast<float>(threshold_.load(std::memory_order_relaxed));

        stft_.processBlock(buffer, [this, learning, floorGain, thresh](T* bins, int numBins)
        {
            // SpectralProcessor invokes the callback once per channel, in
            // channel order, every hop.
            auto& chGain = gains_[static_cast<size_t>(callCounter_ % numChannels_)];
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
            // flickering bins — the source of musical noise.
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
    bool prepared_ = false;

    std::vector<float> profile_;               ///< Learned noise magnitude per bin.
    std::vector<std::vector<float>> gains_;    ///< Per-channel smoothed gains.
    std::vector<float> smooth_;                ///< Scratch raw/frequency-smoothed gains.
    int callCounter_ = 0;

    std::atomic<bool> learning_ { false };
    std::atomic<T> reduction_ { T(18) };
    std::atomic<T> threshold_ { T(2) };
};

} // namespace dspark
