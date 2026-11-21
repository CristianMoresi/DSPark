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
 * setMix() blends the sum with the programme as the crossover leaves it (the
 * bands summed without their curves, as late as the slowest band): the dry
 * side carries the crossover's phase, so any mix is as flat as the crossover
 * is, at the splits too. Each band of the dry side is the dry side of that
 * band's Saturation: through the same oversampling filters when they are
 * minimum phase, and in the phase of its high-passes while the mix is
 * between its ends. setOutputClip() puts an oversampled clipper after the sum, with a
 * ceiling (setOutputCeiling()): many driven bands can add up to far more
 * than any one of them.
 *
 * Level and pan glide over 20 ms; mute and solo fade over 20 ms on a ramp of
 * their own. A band that is not heard (muted, or left out by a solo, once
 * its fade is over) is not computed unless the mix needs its dry side: its
 * Saturation is skipped and starts from silence when the band is heard
 * again, under its fade in.
 *
 * Threading model:
 *  - prepare() / reset() / setOversampling() / setOversamplingMode() /
 *    setOversamplingBand() /
 *    setAntialiasing() / getState() / setState(): setup thread only.
 *  - setNumBands() / setCrossoverFrequency() / setOrder() /
 *    setCrossoverMode() / setCrossoverStructure() / setMix() /
 *    setOutputClip() / setOutputCeiling() and the band setters
 *    (level, pan, mute, solo, and the convenience drive and algorithm
 *    setters): safe from any thread.
 *  - getBandSaturation(): the returned Saturation's parameter setters are
 *    safe from any thread; its prepare(), setOversampling() and setState()
 *    are setup-thread.
 *  - getLatency() and the getters: safe from any thread.
 *
 * Dependencies: Clipper.h, CrossoverFilter.h, Saturation.h,
 * Core/AudioBuffer.h, Core/AudioSpec.h, Core/DspMath.h, Core/StateBlob.h.
 */

#include "Clipper.h"
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
        // Two lines a band: its dry side waits for the slowest band too.
        padLines_.assign(static_cast<size_t>(2 * MaxBands) * static_cast<size_t>(std::max(spec.numChannels, 1))
                         * static_cast<size_t>(kPadSize), T(0));
        dryBuffer_.resize(spec.numChannels, spec.maxBlockSize);
        bandDry_.resize(spec.numChannels, spec.maxBlockSize);
        clip_.setMode(Clipper<T>::Mode::Hard);
        clip_.setOversampling(4);
        clip_.setCeiling(outputCeilingDb_.load(std::memory_order_relaxed));
        clip_.prepare(spec);
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
        clip_.reset();
        clipping_ = outputClip_.load(std::memory_order_relaxed);
        mixNow_ = static_cast<double>(mix_.load(std::memory_order_relaxed));
        dryRunning_ = mixNow_ < 1.0;
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

    // -- The sum ---------------------------------------------------------------

    /**
     * @brief Blends the sum of the bands with the programme, 0 (dry) to 1
     *        (the default: the bands alone; any thread; glides over 20 ms).
     *
     * The dry side is the programme as the crossover leaves it (the bands
     * summed without their curves) and as late as the slowest band, so it
     * meets the processed bands in the crossover's phase: with no drive a
     * mix of one half is as flat as the crossover is, at the splits too.
     * Each band's dry side crosses what that band's does: the filters of
     * minimum-phase oversampling, and between the two ends the phase (not
     * the magnitude) of its Saturation's high-passes, so a mix of one half
     * is the mean of the two levels at the bottom and at the top as well.
     * Each band's own Saturation mix still acts inside its band.
     */
    void setMix(T amount) noexcept
    {
        if (std::isfinite(amount)) mix_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /**
     * @brief An oversampled clipper after the sum (off by default; any thread).
     *
     * Driven bands add up to more than any one of them (six bands at their
     * ceilings sum to several times full scale). With this on, the sum is
     * clipped at setOutputCeiling() four times oversampled, and no sample
     * leaves over the ceiling; under it nothing changes but the latency,
     * which grows by the clipper's (getLatency() reports it).
     */
    void setOutputClip(bool on) noexcept { outputClip_.store(on, std::memory_order_relaxed); }
    [[nodiscard]] bool getOutputClip() const noexcept { return outputClip_.load(std::memory_order_relaxed); }

    /** @brief Ceiling of the output clipper, -24 to 0 dBFS (default 0; any thread). */
    void setOutputCeiling(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        outputCeilingDb_.store(std::clamp(dB, T(-24), T(0)), std::memory_order_relaxed);
        clip_.setCeiling(outputCeilingDb_.load(std::memory_order_relaxed));
    }
    [[nodiscard]] T getOutputCeiling() const noexcept { return outputCeilingDb_.load(std::memory_order_relaxed); }

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
    /** @brief The band the first stage of every band's oversampler keeps (NOT
     *  real-time safe; see Saturation::setOversamplingBand()). */
    void setOversamplingBand(typename Saturation<T>::OversamplingBand band)
    {
        for (auto& each : saturation_) each->setOversamplingBand(band);
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
     * latency of the sum whatever each band runs with; the output clipper
     * adds its own while it is on.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return crossover_.getLatency() + slowestBand(crossover_.getNumBands())
             + (outputClip_.load(std::memory_order_relaxed) ? clip_.getLatency() : 0);
    }

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
        w.write("mix", static_cast<float>(getMix()));
        w.write("outputClip", getOutputClip());
        w.write("outputCeiling", static_cast<float>(getOutputCeiling()));
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
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setOutputClip(r.read("outputClip", false));
        setOutputCeiling(static_cast<T>(r.read("outputCeiling", 0.0f)));
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

        // The dry side of the mix: the dry side of each band, waiting for the
        // slowest band as the band itself does, summed without level, pan or
        // mute. It runs while the mix is, or is on its way, under one.
        const double mixTarget = static_cast<double>(mix_.load(std::memory_order_relaxed));
        const bool dry = mixTarget < 1.0 || mixNow_ < 1.0;
        if (dry && !dryRunning_)
            for (int b = 0; b < MaxBands; ++b) clearPad(MaxBands + b);
        dryRunning_ = dry;
        auto dryBand = bandDry_.toView().getSubView(0, nS);
        if (dry)
            for (int ch = 0; ch < nCh; ++ch) std::fill_n(dryBuffer_.getChannel(ch), nS, T(0));
        for (int ch = 0; ch < nCh; ++ch) std::fill_n(buffer.getChannel(ch), nS, T(0));

        for (int b = 0; b < nb; ++b)
        {
            BandRamp& ramp = ramps_[static_cast<size_t>(b)];
            const Targets t = targets(b, soloing);

            // A band that is not heard, and has faded out, is not computed,
            // unless the mix wants its dry side.
            const bool silent = t.heard == 0.0 && ramp.heard == 0.0;
            if (silent && !dry)
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
            if (dry) sat.processPieces(view, &dryBand, mixTarget > 0.0);
            else sat.processBlock(view);
            const int pad = slowest - sat.getLatency();
            if (dry) addDryBand(b, nCh, nS, pad);

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

        // The mix, gliding like the levels do.
        if (dry)
        {
            const double step = std::clamp(mixTarget - mixNow_, -fadeStep_, fadeStep_);
            double mix = mixNow_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* d = dryBuffer_.getChannel(ch);
                T* out = buffer.getChannel(ch);
                mix = mixNow_;
                for (int i = 0; i < nS; ++i)
                {
                    if (mix != mixTarget)
                    {
                        mix += step;
                        if ((step > 0.0 && mix > mixTarget) || (step < 0.0 && mix < mixTarget)) mix = mixTarget;
                    }
                    out[i] = static_cast<T>(static_cast<double>(d[i]) + mix * (static_cast<double>(out[i]) - static_cast<double>(d[i])));
                }
            }
            mixNow_ = mix;
        }

        // The clipper after the sum, and no sample over its ceiling (its
        // filters ring a little past it).
        const bool clip = outputClip_.load(std::memory_order_relaxed);
        if (clip && !clipping_) clip_.reset();
        clipping_ = clip;
        if (clip)
        {
            clip_.processBlock(buffer.getSubView(0, nS));
            const T top = decibelsToGain(outputCeilingDb_.load(std::memory_order_relaxed));
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i) out[i] = std::clamp(out[i], -top, top);
            }
        }
    }

    /** Adds the dry side of a band, as late as the slowest band. */
    void addDryBand(int band, int channels, int samples, int pad) noexcept
    {
        for (int ch = 0; ch < channels; ++ch)
        {
            const T* in = bandDry_.getChannel(ch);
            T* sum = dryBuffer_.getChannel(ch);
            T* line = padLines_.data() + (static_cast<size_t>(MaxBands + band) * static_cast<size_t>(bandBuffers_[0].getNumChannels())
                                          + static_cast<size_t>(ch)) * static_cast<size_t>(kPadSize);
            int pos = padPos_;
            for (int i = 0; i < samples; ++i)
            {
                line[pos] = in[i];
                sum[i] += line[(pos - pad) & (kPadSize - 1)];
                pos = (pos + 1) & (kPadSize - 1);
            }
        }
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
    std::vector<T> padLines_;   ///< Per band (then per band's dry side) and channel, kPadSize samples each.
    int padPos_ = 0;
    AudioBuffer<T> dryBuffer_, bandDry_;            ///< The dry side of the mix, and that of the band at hand.
    Clipper<T> clip_;                               ///< The clipper after the sum.
    std::atomic<T> mix_ { T(1) }, outputCeilingDb_ { T(0) };
    std::atomic<bool> outputClip_ { false };
    double mixNow_ = 1.0;                           ///< The mix as it glides (audio thread).
    bool dryRunning_ = false, clipping_ = false;    ///< The dry line and the clipper ran in the last chunk.
    double fadeStep_ = 1.0 / 960.0;
};

} // namespace dspark
