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
 * A non-finite input reaches neither the profile nor the gain memory: the
 * frames that contain it come out silent and the next clean frame continues.
 *
 * Options, all off by default (the output is then the same as before, to the
 * bit): setSpectralSmoothing() smooths the gain curve across bins;
 * setGainRelease() gives the gains of each bin a release, slower in the low
 * end (by the square root of the frequency against 1 kHz); a sensitivity
 * curve (setSensitivityBand(), six bands: cuts, bells, shelves, tilt) weights
 * the reduction frequency by frequency; setDelta(true) outputs what is taken
 * out; prepare(spec, 0) picks the transform size by the rate (2048 at 44.1
 * and 48 kHz, doubling by octaves); setFilterMode(true) applies the gains as
 * a minimum-phase filter designed every hop (SpectralFilter: no pre-echo, a
 * transform padded twice, the same latency), where setLink() mixes the gains
 * of the channels bin by bin (1: one filter for both).
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
 * Dependencies: Core/SpectralProcessor.h, Core/SpectralFilter.h,
 * Core/KeyEqualizer.h, Core/AudioSpec.h, Core/AudioBuffer.h, Core/DspMath.h,
 * Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/KeyEqualizer.h"
#include "../Core/SpectralFilter.h"
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
     * @param fftSize STFT size (default 2048; larger = finer hum notching;
     *                0 = by the rate: 2048 at 44.1 and 48 kHz, doubling by
     *                octaves).
     */
    void prepare(const AudioSpec& spec, int fftSize = 2048)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        numChannels_ = spec.numChannels;
        sampleRate_ = spec.sampleRate;
        if (fftSize <= 0)
        {
            const int octaves = std::max(0, static_cast<int>(std::lround(std::log2(std::max(spec.sampleRate, 44100.0) / 44100.0))));
            fftSize = std::min(2048 << octaves, 16384);
        }
        stft_.prepare(spec, fftSize, fftSize / 4);
        numBins_ = stft_.getNumBins();
        filter_.prepare(spec, stft_.getFFTSize(), 2, 4);
        filterBins_ = filter_.getNumBins();

        profile_.assign(static_cast<size_t>(numBins_), 0.0f);
        learnFrames_ = 0;
        cleanPower_.assign(static_cast<size_t>(numChannels_),
                           std::vector<float>(static_cast<size_t>(numBins_), 0.0f));
        filterClean_.assign(static_cast<size_t>(numChannels_ + 1),
                            std::vector<float>(static_cast<size_t>(filterBins_), 0.0f));
        heldGain_.assign(static_cast<size_t>(numChannels_ + 1),
                         std::vector<float>(static_cast<size_t>(std::max(numBins_, filterBins_)), 1.0f));
        gains_.assign(static_cast<size_t>(std::max(numBins_, filterBins_)), 1.0f);
        linkedGains_.assign(static_cast<size_t>(numChannels_ + 1),
                            std::vector<float>(static_cast<size_t>(filterBins_), 1.0f));
        weight_.assign(static_cast<size_t>(numBins_), 1.0f);
        release_.assign(static_cast<size_t>(numBins_), 0.0f);
        releaseFilter_.assign(static_cast<size_t>(filterBins_), 0.0f);
        sensitivity_.prepare(spec.sampleRate);
        weightDirty_.store(true, std::memory_order_release);
        releaseFor_ = -1.0f;

        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Clears signal state and per-bin gain memories (keeps profile). */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        stft_.reset();
        filter_.reset();
        for (auto& c : cleanPower_)
            std::fill(c.begin(), c.end(), 0.0f);
        for (auto& c : filterClean_) std::fill(c.begin(), c.end(), 0.0f);
        for (auto& c : heldGain_) std::fill(c.begin(), c.end(), 1.0f);
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

    /** @brief Smoothing of the gain curve across bins, 0 (off, default) to 0.95. */
    void setSpectralSmoothing(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        smoothing_.store(std::clamp(amount, T(0), T(0.95)), std::memory_order_relaxed);
    }
    /** @brief Release of each bin's gain at 1 kHz, 0 (off, default) to 2000 ms; slower below, faster above. */
    void setGainRelease(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::clamp(ms, T(0), T(2000)), std::memory_order_relaxed);
    }
    /** @brief One band of the sensitivity curve: the reduction is scaled by its gain in dB / 12 + 1, clamped 0 to 2. */
    void setSensitivityBand(int index, const typename KeyEqualizer<T, 2>::Band& band) noexcept
    {
        sensitivity_.setBand(index, band);
        weightDirty_.store(true, std::memory_order_release);
    }
    /** @return One band of the sensitivity curve. */
    [[nodiscard]] typename KeyEqualizer<T, 2>::Band getSensitivityBand(int index) const noexcept { return sensitivity_.getBand(index); }
    /** @brief Outputs what is taken out instead of what is kept. */
    void setDelta(bool on) noexcept { delta_.store(on, std::memory_order_relaxed); }
    /** @brief Applies the gains as a minimum-phase filter (see the file comment). */
    void setFilterMode(bool on) noexcept { filterMode_.store(on, std::memory_order_relaxed); }
    /** @brief In filter mode, the gains of the channels mixed bin by bin, 0 to 1 (default 0). */
    void setLink(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        link_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getSpectralSmoothing() const noexcept { return smoothing_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getGainRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getDelta() const noexcept { return delta_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getFilterMode() const noexcept { return filterMode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLink() const noexcept { return link_.load(std::memory_order_relaxed); }

    [[nodiscard]] bool getLearning() const noexcept { return learning_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getReduction() const noexcept { return reduction_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }

    /** @brief Latency in samples (the STFT pipeline's); 0 before prepare()
     *  succeeds, while the instance passes audio through. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return prepared_.load(std::memory_order_relaxed) ? stft_.getLatency() : 0;
    }

    /** @brief Serializes the parameter state (the learned profile is material-
     *  dependent content, not a preset, and is intentionally not included). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DNSE"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("reduction", static_cast<float>(reduction_.load(std::memory_order_relaxed)));
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        w.write("smoothing", static_cast<float>(getSpectralSmoothing()));
        w.write("release", static_cast<float>(getGainRelease()));
        w.write("filterMode", getFilterMode());
        w.write("link", static_cast<float>(getLink()));
        sensitivity_.writeState(w, "sens");
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DNSE")) return false;
        setReduction(static_cast<T>(r.read("reduction", 18.0f)));
        setThreshold(static_cast<T>(r.read("threshold", 2.0f)));
        setSpectralSmoothing(static_cast<T>(r.read("smoothing", 0.0f)));
        setGainRelease(static_cast<T>(r.read("release", 0.0f)));
        setFilterMode(r.read("filterMode", false));
        setLink(static_cast<T>(r.read("link", 0.0f)));
        sensitivity_.readState(r, "sens");
        weightDirty_.store(true, std::memory_order_release);
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

        refreshOptions();
        if (filterMode_.load(std::memory_order_relaxed))
        {
            processFiltered(buffer, learning, floorGain, overSub, nChEff);
            return;
        }
        const bool extras = weighted_ || smoothNow_ > 0.0f || releaseNow_ > 0.0f || delta_.load(std::memory_order_relaxed);
        if (extras)
        {
            processWithOptions(buffer, learning, floorGain, overSub, nChEff);
            return;
        }

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

                // The gain memory is recursive and the profile is a running
                // mean: one NaN in either would stay for the whole stream.
                if (!std::isfinite(power))
                {
                    clean[static_cast<size_t>(k)] = 0.0f;
                    bins[2 * k] = T(0);
                    bins[2 * k + 1] = T(0);
                    continue;
                }

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
    /** @brief Picks up the options and rebuilds what depends on them. */
    void refreshOptions() noexcept
    {
        smoothNow_ = static_cast<float>(smoothing_.load(std::memory_order_relaxed));
        releaseNow_ = static_cast<float>(releaseMs_.load(std::memory_order_relaxed));
        if (weightDirty_.exchange(false, std::memory_order_acquire))
        {
            weighted_ = sensitivity_.hasBands();
            const auto response = sensitivity_.getResponse();
            const double binHz = sampleRate_ / static_cast<double>(stft_.getFFTSize());
            for (int k = 0; k < numBins_; ++k)
                weight_[static_cast<size_t>(k)] = weighted_
                    ? static_cast<float>(std::clamp(1.0 + response.magnitudeDb(std::max(1.0, k * binHz)) / 12.0, 0.0, 2.0))
                    : 1.0f;
        }
        if (releaseNow_ != releaseFor_)
        {
            releaseFor_ = releaseNow_;
            // One hop of either path: a quarter of the transform.
            const double hop = static_cast<double>(stft_.getFFTSize()) / 4.0 / sampleRate_;
            const auto fill = [&](std::vector<float>& r, double binHz) {
                for (size_t k = 0; k < r.size(); ++k)
                {
                    const double f = std::max(20.0, static_cast<double>(k) * binHz);
                    const double tau = 0.001 * static_cast<double>(releaseNow_) * std::sqrt(1000.0 / f);
                    r[k] = releaseNow_ > 0.0f ? static_cast<float>(std::exp(-hop / std::max(tau, 1e-6))) : 0.0f;
                }
            };
            fill(release_, sampleRate_ / static_cast<double>(stft_.getFFTSize()));
            fill(releaseFilter_, filter_.getBinHz());
        }
    }

    /**
     * @brief The gain of one bin (the decision-directed Wiener rule) with
     *        its floor; updates the bin's clean power memory.
     */
    static float binGain(float power, float noise, float& prevClean, float overSub, float floorGain) noexcept
    {
        float gain = 1.0f;
        const float lambda = overSub * noise;
        if (lambda > 0.0f)
        {
            const float post = power / lambda;
            const float prio = std::max(kDDAlpha * prevClean / lambda
                                        + (1.0f - kDDAlpha) * std::max(post - 1.0f, 0.0f),
                                        kMinPrioriSnr);
            gain = std::max(prio / (1.0f + prio), floorGain);
        }
        return gain;
    }

    /** @brief The options on a gain curve: smoothing across bins, the release. */
    void shapeGains(float* g, int n, std::vector<float>& held, const std::vector<float>& release) noexcept
    {
        if (smoothNow_ > 0.0f)
        {
            for (int k = 1; k < n; ++k) g[k] += smoothNow_ * (g[k - 1] - g[k]);
            for (int k = n - 2; k >= 0; --k) g[k] += smoothNow_ * (g[k + 1] - g[k]);
        }
        if (releaseNow_ > 0.0f)
            for (int k = 0; k < n; ++k)
            {
                float& h = held[static_cast<size_t>(k)];
                h = g[k] < h ? g[k] : g[k] + release[static_cast<size_t>(k)] * (h - g[k]);
                g[k] = h;
            }
    }

    /** @brief The STFT path with the options. */
    void processWithOptions(AudioBufferView<T> buffer, bool learning, float floorGain, float overSub, int nChEff) noexcept
    {
        const bool delta = delta_.load(std::memory_order_relaxed);
        const float reductionDb = static_cast<float>(reduction_.load(std::memory_order_relaxed));
        stft_.processBlock(buffer, [&](T* bins, int numBins)
        {
            const int ch = callCounter_ % nChEff;
            auto& clean = cleanPower_[static_cast<size_t>(ch)];
            ++callCounter_;
            float learnRate = 0.0f;
            if (learning)
            {
                learnFrames_ = std::min(learnFrames_ + 1, kMaxLearnFrames);
                learnRate = 1.0f / static_cast<float>(learnFrames_);
            }
            for (int k = 0; k < numBins; ++k)
            {
                const float re = static_cast<float>(bins[2 * k]), im = static_cast<float>(bins[2 * k + 1]);
                const float power = re * re + im * im;
                if (!std::isfinite(power)) { gains_[static_cast<size_t>(k)] = 0.0f; clean[static_cast<size_t>(k)] = 0.0f; continue; }
                auto& noise = profile_[static_cast<size_t>(k)];
                if (learning) noise += learnRate * (power - noise);
                const float fl = weighted_ ? std::pow(10.0f, -reductionDb * weight_[static_cast<size_t>(k)] / 20.0f) : floorGain;
                gains_[static_cast<size_t>(k)] = binGain(power, noise, clean[static_cast<size_t>(k)], overSub, fl);
            }
            shapeGains(gains_.data(), numBins, heldGain_[static_cast<size_t>(ch)], release_);
            for (int k = 0; k < numBins; ++k)
            {
                const float g = gains_[static_cast<size_t>(k)];
                const float re = static_cast<float>(bins[2 * k]), im = static_cast<float>(bins[2 * k + 1]);
                const float power = re * re + im * im;
                if (!std::isfinite(power)) { bins[2 * k] = T(0); bins[2 * k + 1] = T(0); continue; }
                clean[static_cast<size_t>(k)] = g * g * power;
                const float applied = delta ? 1.0f - g : g;
                bins[2 * k] = static_cast<T>(re * applied);
                bins[2 * k + 1] = static_cast<T>(im * applied);
            }
        });
    }

    /**
     * @brief The filter path: the same rule on the filter's power (read
     *        against the profile in the STFT's bins and scale), applied as a
     *        minimum-phase filter.
     */
    void processFiltered(AudioBufferView<T> buffer, bool learning, float floorGain, float overSub, int nChEff) noexcept
    {
        const float link = static_cast<float>(link_.load(std::memory_order_relaxed));
        const float reductionDb = static_cast<float>(reduction_.load(std::memory_order_relaxed));
        filter_.setLinked(link >= 1.0f && nChEff == 2);
        filter_.setDelta(delta_.load(std::memory_order_relaxed));
        // The profile is noise power in the STFT's bins and window: noise
        // reads 12 / N^2 of it through the filter's Hann window and power
        // scale, at filter bin 2k for STFT bin k.
        const float n = static_cast<float>(stft_.getFFTSize());
        const float toFilter = 12.0f / (n * n);
        filter_.processBlock(buffer, [&](int ch, const T* power, T* gain, int bins) noexcept
        {
            const int c = ch < 0 ? nChEff : ch;
            float learnRate = 0.0f;
            if (learning)
            {
                learnFrames_ = std::min(learnFrames_ + 1, kMaxLearnFrames);
                learnRate = 1.0f / static_cast<float>(learnFrames_);
            }
            auto& clean = filterClean_[static_cast<size_t>(c)];
            for (int k = 0; k < bins; ++k)
            {
                const float p = static_cast<float>(power[k]);
                const int sk = std::min(k / 2, numBins_ - 1);
                if (learning && (k & 1) == 0 && std::isfinite(p))
                    profile_[static_cast<size_t>(sk)] += learnRate * (p / toFilter - profile_[static_cast<size_t>(sk)]);
                const float noise = 0.5f * (profile_[static_cast<size_t>(sk)] + profile_[static_cast<size_t>(std::min(sk + (k & 1), numBins_ - 1))]) * toFilter;
                const float fl = weighted_ ? std::pow(10.0f, -reductionDb * weight_[static_cast<size_t>(sk)] / 20.0f) : floorGain;
                gains_[static_cast<size_t>(k)] = std::isfinite(p) ? binGain(p, noise, clean[static_cast<size_t>(k)], overSub, fl) : 0.0f;
            }
            shapeGains(gains_.data(), bins, heldGain_[static_cast<size_t>(c)], releaseFilter_);
            // The link mixes this channel's gains with the others' last ones.
            auto& mine = linkedGains_[static_cast<size_t>(c)];
            for (int k = 0; k < bins; ++k) mine[static_cast<size_t>(k)] = gains_[static_cast<size_t>(k)];
            if (link > 0.0f && ch >= 0 && nChEff == 2)
            {
                const auto& other = linkedGains_[static_cast<size_t>(1 - ch)];
                for (int k = 0; k < bins; ++k)
                    gains_[static_cast<size_t>(k)] += link * 0.5f * (other[static_cast<size_t>(k)] - gains_[static_cast<size_t>(k)]);
            }
            for (int k = 0; k < bins; ++k)
            {
                const float g = gains_[static_cast<size_t>(k)];
                clean[static_cast<size_t>(k)] = g * g * static_cast<float>(power[k]);
                gain[k] = static_cast<T>(g);
            }
        });
    }

    SpectralProcessor<T> stft_;
    SpectralFilter<T> filter_;
    KeyEqualizer<T, 2> sensitivity_;
    int numChannels_ = 0;
    int numBins_ = 0, filterBins_ = 0;
    double sampleRate_ = 48000.0;
    std::atomic<bool> prepared_ { false };
    std::atomic<T> smoothing_ { T(0) }, releaseMs_ { T(0) }, link_ { T(0) };
    std::atomic<bool> delta_ { false }, filterMode_ { false }, weightDirty_ { true };
    bool weighted_ = false;
    float smoothNow_ = 0.0f, releaseNow_ = 0.0f, releaseFor_ = -1.0f;
    std::vector<std::vector<float>> filterClean_, heldGain_, linkedGains_;
    std::vector<float> gains_, weight_, release_, releaseFilter_;

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
