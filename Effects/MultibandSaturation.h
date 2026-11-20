// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file MultibandSaturation.h
 * @brief Multi-band saturation: a crossover split, a Saturation per band,
 *        and the bands summed back in time with each other.
 *
 * Splits the signal into 2..MaxBands bands with a Linkwitz-Riley crossover
 * (IIR or linear-phase), runs each band through its own Saturation (any
 * algorithm, drive, character, curve shape, the stages around the curve, its
 * own mix), then applies a level, a pan, a mute and a solo per band and sums
 * them. The bands are summed in time whatever each one's latency: a band
 * that runs with less latency than the slowest one is delayed by the
 * difference, so a band with another oversampling factor or antialiasing
 * order does not comb against its neighbours at the crossover.
 *
 * Level and pan glide over 20 ms; mute and solo fade over 20 ms on a ramp of
 * their own. A band that is not heard (muted, or left out by a solo, once
 * its fade is over) is not computed: its Saturation is skipped and starts
 * from silence when the band is heard again, under its fade in.
 *
 * Threading model:
 *  - prepare() / reset() / setOversampling() / setOversamplingMode() /
 *    setAntialiasing() / getState() / setState(): setup thread only.
 *  - setNumBands() / setCrossoverFrequency() / setOrder() /
 *    setCrossoverMode() / setCrossoverStructure() and the band setters
 *    (level, pan, mute, solo, and the convenience drive and algorithm
 *    setters): safe from any thread.
 *  - getBandSaturation(): the returned Saturation's parameter setters are
 *    safe from any thread; its prepare(), setOversampling() and setState()
 *    are setup-thread.
 *  - getLatency() and the getters: safe from any thread.
 *
 * Dependencies: CrossoverFilter.h, Saturation.h, Core/AudioBuffer.h,
 * Core/AudioSpec.h, Core/DspMath.h, Core/StateBlob.h.
 */

#include "CrossoverFilter.h"
#include "Saturation.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class MultibandSaturation
 * @brief Multi-band saturation: crossover split, a Saturation per band, sum.
 *
 * @tparam T        Sample type (float or double).
 * @tparam MaxBands Maximum number of bands (compile-time, default 6).
 */
template <FloatType T, int MaxBands = 6>
class MultibandSaturation
{
    static_assert(MaxBands >= 2, "MultibandSaturation needs at least 2 bands (one split)");

public:
    using Crossover = CrossoverFilter<T, MaxBands>;

    MultibandSaturation()
    {
        for (auto& band : saturation_) band = std::make_unique<Saturation<T>>();
    }

    MultibandSaturation(const MultibandSaturation&) = delete;
    MultibandSaturation& operator=(const MultibandSaturation&) = delete;

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the crossover, every band and the buffers.
     *
     * Invalid specs are ignored. May allocate; setup thread only.
     *
     * @param spec The audio specification (sample rate, max block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_.store(false, std::memory_order_relaxed);
        spec_ = spec;
        crossover_.prepare(spec);
        for (int b = 0; b < MaxBands; ++b)
        {
            bandBuffers_[static_cast<size_t>(b)].resize(spec.numChannels, spec.maxBlockSize);
            saturation_[static_cast<size_t>(b)]->prepare(spec);
        }
        padLines_.assign(static_cast<size_t>(MaxBands) * static_cast<size_t>(std::max(spec.numChannels, 1))
                         * static_cast<size_t>(kPadSize), T(0));
        fadeStep_ = 1.0 / std::max(1.0, kFadeSeconds * spec.sampleRate);
        reset();
        prepared_.store(true, std::memory_order_relaxed);
    }

    /** @brief Clears the crossover, the bands and the alignment delays. */
    void reset() noexcept
    {
        crossover_.reset();
        for (auto& band : saturation_) band->reset();
        std::fill(padLines_.begin(), padLines_.end(), T(0));
        padPos_ = 0;
        for (int b = 0; b < MaxBands; ++b)
        {
            BandRamp& ramp = ramps_[static_cast<size_t>(b)];
            const Targets t = targets(b, anySolo());
            ramp.heard = t.heard;
            ramp.left = t.left;
            ramp.right = t.right;
            ramp.computed = t.heard > 0.0;
        }
    }

    /**
     * @brief Processes audio through the bands, in place.
     *
     * Pass-through until prepare() succeeds. A block longer than the
     * prepared maxBlockSize is processed in pieces of that size; channels
     * beyond the prepared count pass through.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        const int total = buffer.getNumSamples();
        const int chunk = bandBuffers_[0].getNumSamples();
        if (chunk <= 0) return;
        for (int offset = 0; offset < total; offset += chunk)
            processChunk(buffer.getSubView(offset, std::min(chunk, total - offset)));
    }

    // -- Crossover -------------------------------------------------------------

    /** @brief Number of bands (2..MaxBands; any thread; re-spaces the split frequencies). */
    void setNumBands(int n) noexcept { crossover_.setNumBands(std::clamp(n, 2, MaxBands)); }
    /** @brief Sets a split frequency (any thread). */
    void setCrossoverFrequency(int index, T hz) noexcept { crossover_.setCrossoverFrequency(index, hz); }
    /** @brief Slope of the crossover in dB per octave (any thread; see CrossoverFilter::setOrder()). */
    void setOrder(int order) noexcept { crossover_.setOrder(order); }
    /** @brief Minimum-phase (IIR) or linear-phase crossover (any thread; see CrossoverFilter). */
    void setCrossoverMode(typename Crossover::FilterMode mode) noexcept { crossover_.setFilterMode(mode); }
    /** @brief Direct-form or ladder sections in the minimum-phase crossover (any thread; see CrossoverFilter::setStructure()). */
    void setCrossoverStructure(typename Crossover::Structure structure) noexcept { crossover_.setStructure(structure); }

    [[nodiscard]] int getNumBands() const noexcept { return crossover_.getNumBands(); }
    [[nodiscard]] T getCrossoverFrequency(int index) const noexcept { return crossover_.getCrossoverFrequency(index); }
    [[nodiscard]] int getOrder() const noexcept { return crossover_.getOrder(); }
    [[nodiscard]] typename Crossover::FilterMode getCrossoverMode() const noexcept { return crossover_.getFilterMode(); }
    [[nodiscard]] typename Crossover::Structure getCrossoverStructure() const noexcept { return crossover_.getStructure(); }

    // -- Bands ---------------------------------------------------------------

    /** @brief The Saturation of a band (the index is clamped). */
    [[nodiscard]] Saturation<T>& getBandSaturation(int band) noexcept
    {
        return *saturation_[static_cast<size_t>(std::clamp(band, 0, MaxBands - 1))];
    }
    /** @brief The Saturation of a band (the index is clamped). */
    [[nodiscard]] const Saturation<T>& getBandSaturation(int band) const noexcept
    {
        return *saturation_[static_cast<size_t>(std::clamp(band, 0, MaxBands - 1))];
    }

    /** @brief The algorithm of a band (any thread). */
    void setBandAlgorithm(int band, typename Saturation<T>::Algorithm algo) { getBandSaturation(band).setAlgorithm(algo); }
    /** @brief The drive of a band in dB (any thread). */
    void setBandDrive(int band, T dB) { getBandSaturation(band).setDrive(dB); }

    /** @brief Level of a band in dB, -60 to +24 (any thread; glides over 20 ms). */
    void setBandLevel(int band, T dB) noexcept
    {
        if (band < 0 || band >= MaxBands || !std::isfinite(dB)) return;
        params_[static_cast<size_t>(band)].levelDb.store(std::clamp(dB, T(-60), T(24)), std::memory_order_relaxed);
    }
    /** @brief Pan of a band, -1 (left) to +1 (right); a balance that never lifts a side (any thread). */
    void setBandPan(int band, T pan) noexcept
    {
        if (band < 0 || band >= MaxBands || !std::isfinite(pan)) return;
        params_[static_cast<size_t>(band)].pan.store(std::clamp(pan, T(-1), T(1)), std::memory_order_relaxed);
    }
    /** @brief Mutes a band (any thread; fades over 20 ms). */
    void setBandMute(int band, bool on) noexcept
    {
        if (band < 0 || band >= MaxBands) return;
        params_[static_cast<size_t>(band)].mute.store(on, std::memory_order_relaxed);
    }
    /** @brief Solos a band: while any band is soloed only the soloed ones are heard (any thread; fades over 20 ms). */
    void setBandSolo(int band, bool on) noexcept
    {
        if (band < 0 || band >= MaxBands) return;
        params_[static_cast<size_t>(band)].solo.store(on, std::memory_order_relaxed);
    }

    [[nodiscard]] T getBandLevel(int band) const noexcept { return param(band).levelDb.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBandPan(int band) const noexcept { return param(band).pan.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isBandMuted(int band) const noexcept { return param(band).mute.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isBandSoloed(int band) const noexcept { return param(band).solo.load(std::memory_order_relaxed); }

    /** @brief Whether a band is computed now (false while it is not heard; audio-thread state). */
    [[nodiscard]] bool isBandComputed(int band) const noexcept
    {
        return band >= 0 && band < MaxBands && ramps_[static_cast<size_t>(band)].computed;
    }

    // -- Settings shared by every band (setup thread) ----------------------------

    /** @brief The oversampling factor of every band (NOT real-time safe; see Saturation::setOversampling()). */
    void setOversampling(int factor)
    {
        for (auto& band : saturation_) band->setOversampling(factor);
    }
    /** @brief The oversampler filters of every band (NOT real-time safe). */
    void setOversamplingMode(typename Saturation<T>::OversamplingMode mode)
    {
        for (auto& band : saturation_) band->setOversamplingMode(mode);
    }
    /** @brief The antialiasing order of every band. */
    void setAntialiasing(typename Saturation<T>::Antialiasing mode) noexcept
    {
        for (auto& band : saturation_) band->setAntialiasing(mode);
    }

    /**
     * @brief Latency in samples: the crossover's and the slowest band's.
     *
     * Every band is delayed to the slowest band's latency, so this is the
     * latency of the sum whatever each band runs with.
     */
    [[nodiscard]] int getLatency() const noexcept { return crossover_.getLatency() + slowestBand(crossover_.getNumBands()); }

    // -- State -----------------------------------------------------------------

    /** @brief Serializes the crossover, each band's Saturation and its level, pan, mute and solo. */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("MBST"), 1);
        const int n = crossover_.getNumBands();
        w.write("numBands", n);
        w.write("order", crossover_.getOrder());
        w.write("xoverMode", static_cast<int32_t>(crossover_.getFilterMode()));
        w.write("xoverStructure", static_cast<int32_t>(crossover_.getStructure()));
        char key[32];
        for (int i = 0; i < n - 1; ++i)
        {
            std::snprintf(key, sizeof(key), "x%d", i);
            w.write(key, static_cast<float>(crossover_.getCrossoverFrequency(i)));
        }
        for (int i = 0; i < n; ++i)
        {
            std::snprintf(key, sizeof(key), "band%d", i);
            w.write(key, getBandSaturation(i).getState());
            std::snprintf(key, sizeof(key), "b%d.level", i);
            w.write(key, static_cast<float>(getBandLevel(i)));
            std::snprintf(key, sizeof(key), "b%d.pan", i);
            w.write(key, static_cast<float>(getBandPan(i)));
            std::snprintf(key, sizeof(key), "b%d.mute", i);
            w.write(key, isBandMuted(i));
            std::snprintf(key, sizeof(key), "b%d.solo", i);
            w.write(key, isBandSoloed(i));
        }
        return w.blob();
    }

    /** @brief Restores what getState() wrote (setup thread: the bands may reallocate). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("MBST")) return false;
        const int n = std::clamp(static_cast<int>(r.read("numBands", int32_t(3))), 2, MaxBands);
        setNumBands(n);
        setOrder(r.read("order", int32_t(24)));
        setCrossoverMode(static_cast<typename Crossover::FilterMode>(std::clamp(static_cast<int>(r.read("xoverMode", int32_t(0))), 0, 1)));
        setCrossoverStructure(static_cast<typename Crossover::Structure>(std::clamp(static_cast<int>(r.read("xoverStructure", int32_t(0))), 0, 1)));
        char key[32];
        for (int i = 0; i < n - 1; ++i)
        {
            std::snprintf(key, sizeof(key), "x%d", i);
            const float f = r.read(key, -1.0f);
            if (f > 0.0f) setCrossoverFrequency(i, static_cast<T>(f));
        }
        for (int i = 0; i < n; ++i)
        {
            std::snprintf(key, sizeof(key), "band%d", i);
            const auto nested = r.readBlob(key);
            if (!nested.empty()) getBandSaturation(i).setState(nested.data(), nested.size());
            std::snprintf(key, sizeof(key), "b%d.level", i);
            setBandLevel(i, static_cast<T>(r.read(key, 0.0f)));
            std::snprintf(key, sizeof(key), "b%d.pan", i);
            setBandPan(i, static_cast<T>(r.read(key, 0.0f)));
            std::snprintf(key, sizeof(key), "b%d.mute", i);
            setBandMute(i, r.read(key, false));
            std::snprintf(key, sizeof(key), "b%d.solo", i);
            setBandSolo(i, r.read(key, false));
        }
        return true;
    }

private:
    static constexpr double kFadeSeconds = 0.02;    ///< Glide of level and pan, fade of mute and solo.
    static constexpr int kPadSize = 512;            ///< Longest latency a band can be delayed by, plus room.

    struct BandParams
    {
        std::atomic<T> levelDb { T(0) };
        std::atomic<T> pan { T(0) };
        std::atomic<bool> mute { false };
        std::atomic<bool> solo { false };
    };

    /** Where each of a band's two ramps is (audio thread). */
    struct BandRamp
    {
        double heard = 1.0;             ///< Mute and solo: 0 silent, 1 heard.
        double left = 1.0, right = 1.0; ///< Level and pan, per side.
        bool computed = true;
    };

    struct Targets
    {
        double heard, left, right;
    };

    [[nodiscard]] const BandParams& param(int band) const noexcept
    {
        return params_[static_cast<size_t>(std::clamp(band, 0, MaxBands - 1))];
    }

    [[nodiscard]] bool anySolo() const noexcept
    {
        const int n = crossover_.getNumBands();
        for (int b = 0; b < n; ++b)
            if (params_[static_cast<size_t>(b)].solo.load(std::memory_order_relaxed)) return true;
        return false;
    }

    /** Where a band's ramps are going. */
    [[nodiscard]] Targets targets(int band, bool soloing) const noexcept
    {
        const BandParams& p = params_[static_cast<size_t>(band)];
        const bool heard = !p.mute.load(std::memory_order_relaxed)
                        && (!soloing || p.solo.load(std::memory_order_relaxed));
        const double level = std::pow(10.0, static_cast<double>(p.levelDb.load(std::memory_order_relaxed)) / 20.0);
        const double pan = static_cast<double>(p.pan.load(std::memory_order_relaxed));
        const double left = (pan > 0.0) ? std::cos(0.5 * std::numbers::pi * pan) : 1.0;
        const double right = (pan < 0.0) ? std::cos(-0.5 * std::numbers::pi * pan) : 1.0;
        return { heard ? 1.0 : 0.0, level * left, level * right };
    }

    /** The latency of the slowest of the first n bands. */
    [[nodiscard]] int slowestBand(int n) const noexcept
    {
        int slowest = 0;
        for (int b = 0; b < std::clamp(n, 0, MaxBands); ++b)
            slowest = std::max(slowest, saturation_[static_cast<size_t>(b)]->getLatency());
        return std::min(slowest, kPadSize - 1);
    }

    void processChunk(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), bandBuffers_[0].getNumChannels());
        const int nS = std::min(buffer.getNumSamples(), bandBuffers_[0].getNumSamples());
        if (nCh <= 0 || nS <= 0) return;

        for (int b = 0; b < MaxBands; ++b)
            views_[static_cast<size_t>(b)] = bandBuffers_[static_cast<size_t>(b)].toView().getSubView(0, nS);
        const int nb = crossover_.processBlock(buffer, views_.data(), MaxBands);
        if (nb < 2) return;

        const bool soloing = anySolo();
        const int slowest = slowestBand(nb);
        for (int ch = 0; ch < nCh; ++ch) std::fill_n(buffer.getChannel(ch), nS, T(0));

        for (int b = 0; b < nb; ++b)
        {
            BandRamp& ramp = ramps_[static_cast<size_t>(b)];
            const Targets t = targets(b, soloing);

            // A band that is not heard, and has faded out, is not computed.
            const bool silent = t.heard == 0.0 && ramp.heard == 0.0;
            if (silent)
            {
                ramp.computed = false;
                ramp.left = t.left;
                ramp.right = t.right;
                continue;
            }
            Saturation<T>& sat = *saturation_[static_cast<size_t>(b)];
            if (!ramp.computed)
            {
                // Back from silence: its history is of another time.
                sat.reset();
                clearPad(b);
                ramp.computed = true;
            }
            auto view = views_[static_cast<size_t>(b)];
            sat.processBlock(view);
            const int pad = slowest - sat.getLatency();

            const double heardStep = std::clamp(t.heard - ramp.heard, -fadeStep_, fadeStep_);
            const bool fading = ramp.heard != t.heard;
            const double leftStep = (t.left - ramp.left) * fadeStep_;
            const double rightStep = (t.right - ramp.right) * fadeStep_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* in = view.getChannel(ch);
                T* out = buffer.getChannel(ch);
                T* line = padLines_.data() + (static_cast<size_t>(b) * static_cast<size_t>(bandBuffers_[0].getNumChannels())
                                              + static_cast<size_t>(ch)) * static_cast<size_t>(kPadSize);
                double heard = ramp.heard, left = ramp.left, right = ramp.right;
                const bool isRight = (ch == 1);
                int pos = padPos_;
                for (int i = 0; i < nS; ++i)
                {
                    // Every band is delayed to the slowest one's latency.
                    line[pos] = in[i];
                    const T aligned = line[(pos - pad) & (kPadSize - 1)];
                    pos = (pos + 1) & (kPadSize - 1);
                    if (fading)
                    {
                        heard += heardStep;
                        if ((heardStep > 0.0 && heard > t.heard) || (heardStep < 0.0 && heard < t.heard)) heard = t.heard;
                    }
                    left += leftStep;
                    right += rightStep;
                    const double side = (nCh == 2) ? (isRight ? right : left) : 0.5 * (left + right);
                    out[i] += static_cast<T>(static_cast<double>(aligned) * heard * side);
                }
                if (ch == nCh - 1)
                {
                    ramp.heard = heard;
                    ramp.left = left;
                    ramp.right = right;
                }
            }
        }
        // Bands past the count (after a change of it) are left as they were.
        padPos_ = (padPos_ + nS) & (kPadSize - 1);
        for (int b = nb; b < MaxBands; ++b) ramps_[static_cast<size_t>(b)].computed = false;
    }

    void clearPad(int band) noexcept
    {
        const size_t per = static_cast<size_t>(bandBuffers_[0].getNumChannels()) * static_cast<size_t>(kPadSize);
        std::fill_n(padLines_.data() + static_cast<size_t>(band) * per, per, T(0));
    }

    std::atomic<bool> prepared_ { false };
    AudioSpec spec_ {};
    Crossover crossover_;
    std::array<std::unique_ptr<Saturation<T>>, MaxBands> saturation_ {};
    std::array<AudioBuffer<T>, MaxBands> bandBuffers_ {};
    std::array<AudioBufferView<T>, MaxBands> views_ {};
    std::array<BandParams, MaxBands> params_ {};
    std::array<BandRamp, MaxBands> ramps_ {};
    std::vector<T> padLines_;   ///< Per band and channel, kPadSize samples each.
    int padPos_ = 0;
    double fadeStep_ = 1.0 / 960.0;
};

} // namespace dspark
