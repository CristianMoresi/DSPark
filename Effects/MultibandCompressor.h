// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file MultibandCompressor.h
 * @brief Multi-band compressor using CrossoverFilter + per-band Compressor.
 *
 * Splits the signal into 2..MaxBands frequency bands with a Linkwitz-Riley
 * crossover (IIR or linear-phase), compresses each band independently with a
 * full Compressor instance, then sums the bands back together. The band sum
 * relies on every band sharing the same latency (see processBlock()).
 *
 * Threading model:
 *  - prepare() / reset() / getState() / setState(): setup thread only (never
 *    concurrent with processBlock()).
 *  - setNumBands() / setCrossoverFrequency() / setOrder() / setCrossoverMode()
 *    and the per-band convenience setters: safe from any thread (they delegate
 *    to the atomic publications of CrossoverFilter / Compressor).
 *  - getBandCompressor(): the returned Compressor's parameter setters are
 *    safe from any thread; its prepare()/reset()/setState() are setup-thread
 *    (this class already prepares and resets every band for you).
 *  - getBandGainReductionDb() and the other getters: metering-style reads,
 *    safe from any thread (approximate while audio is running).
 *
 * Dependencies: CrossoverFilter.h, Compressor.h, Core/AudioBuffer.h,
 * Core/AudioSpec.h, Core/DspMath.h, Core/SimdOps.h, Core/StateBlob.h.
 */

#include "CrossoverFilter.h"
#include "Compressor.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/SimdOps.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dspark {

/**
 * @class MultibandCompressor
 * @brief Multi-band compressor: crossover split, per-band compression, sum.
 *
 * @tparam T        Sample type (float or double).
 * @tparam MaxBands Maximum number of bands (compile-time, default 12).
 */
template <FloatType T, int MaxBands = 12>
class MultibandCompressor
{
    static_assert(MaxBands >= 2, "MultibandCompressor needs at least 2 bands (one split)");

public:
    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the multiband compressor and internal buffers for processing.
     *
     * Invalid specs (non-positive/non-finite rate, block size or channel count)
     * are ignored: the previous state is kept and an unprepared instance stays
     * pass-through. May allocate; setup thread only.
     *
     * @param spec The audio specification (sample rate, max block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;

        prepared_.store(false, std::memory_order_relaxed);

        crossover_.prepare(spec);
        for (int b = 0; b < MaxBands; ++b)
        {
            bandBuffers_[b].resize(spec.numChannels, spec.maxBlockSize);
            compressors_[b].prepare(spec);
        }
        lastNumBands_ = MaxBands; // freshly prepared bands are all clean

        prepared_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Processes audio through the multi-band compressor.
     *
     * Pass-through until prepare() succeeds. The processed span is clamped to
     * the prepared maxBlockSize (trailing samples of an oversized block are
     * left untouched); channels beyond the prepared count pass through.
     *
     * @param buffer In/Out audio buffer view.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;

        // Clamp to the per-band buffers' geometry (allocated for the prepared
        // spec): a wider or longer caller buffer must never index the band
        // buffers out of bounds below.
        const int nCh = std::min(buffer.getNumChannels(), bandBuffers_[0].getNumChannels());
        const int nS  = std::min(buffer.getNumSamples(), bandBuffers_[0].getNumSamples());
        if (nCh <= 0 || nS <= 0) return;

        // 1. Hand the crossover every band slot, truncated to this block. It
        // returns how many bands it actually wrote THIS block (its band count
        // is an atomic a concurrent setNumBands() may move between our read
        // and its own); summing only what was written keeps stale band
        // buffers out of the output.
        for (int b = 0; b < MaxBands; ++b)
            views_[b] = bandBuffers_[b].toView().getSubView(0, nS);

        // 2. Split into bands
        const int nb = crossover_.processBlock(buffer, views_.data(), MaxBands);
        if (nb < 2) return; // crossover inactive: input left untouched

        // 3. Bands (re-)enabled by a live band-count increase start clean:
        // their compressors would otherwise replay the gain reduction from
        // when they were last active (arbitrarily stale).
        if (nb > lastNumBands_)
            for (int b = lastNumBands_; b < nb; ++b)
                compressors_[b].reset();
        lastNumBands_ = nb;

        // 4. Compress each band independently.
        // IMPORTANT: bands are summed directly, so every band must share the
        // SAME latency. Compressors default to 0 lookahead (latency 0) and a
        // latency-free detector -> coherent sum. If you enable lookahead or
        // the Hilbert detector (feed-forward adds its compensation delay),
        // apply the SAME setting to every band via getBandCompressor(b); a
        // divergent per-band latency would phase-cancel at the crossover
        // regions (this class does not auto-delay-compensate).
        for (int b = 0; b < nb; ++b)
            compressors_[b].processBlock(views_[b]);

        // 5. Sum bands back into the output buffer (SIMD add kernels)
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* const __restrict out = buffer.getChannel(ch);
            const T* const __restrict src0 = bandBuffers_[0].getChannel(ch);

            // Base copy (band 0)
            std::copy(src0, src0 + nS, out);

            // Accumulate remaining active bands
            for (int b = 1; b < nb; ++b)
                simd::add(out, bandBuffers_[b].getChannel(ch), nS);
        }
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Sets the number of active frequency bands.
     *
     * Safe from any thread. A live change applies instantly (audible click)
     * and re-initialises ALL split frequencies to the crossover's log-spaced
     * defaults - reapply custom frequencies afterwards (see
     * CrossoverFilter::setNumBands). Compressors of newly enabled bands are
     * reset on the audio thread so they start clean.
     *
     * @param n Number of bands (clamped between 2 and MaxBands - a crossover
     *          needs at least one split point).
     */
    void setNumBands(int n) noexcept
    {
        crossover_.setNumBands(std::clamp(n, 2, MaxBands));
    }

    /**
     * @brief Sets the crossover frequency for a specific split point.
     * @param index Split point index (0 is between band 0 and 1).
     * @param freqHz Target frequency in Hertz (non-finite values are ignored).
     */
    void setCrossoverFrequency(int index, T freqHz) noexcept
    {
        crossover_.setCrossoverFrequency(index, freqHz);
    }

    /**
     * @brief Sets the crossover slope in dB/oct: 12, 24 or 48.
     * @param order Slope in dB/oct; other values are ignored (see CrossoverFilter::setOrder).
     */
    void setOrder(int order) noexcept
    {
        crossover_.setOrder(order);
    }

    /**
     * @brief Sets the phase/processing mode of the crossover (IIR or linear-phase).
     * @param mode Selected filter mode from the CrossoverFilter enum.
     */
    void setCrossoverMode(typename CrossoverFilter<T, MaxBands>::FilterMode mode) noexcept
    {
        crossover_.setFilterMode(mode);
    }

    // -- Per-band compressor access ------------------------------------------

    /**
     * @brief Direct access to a band's compressor for full configuration.
     * @param band Band index. Clamped safely to valid range in release mode.
     * @return Reference to the requested Compressor.
     */
    [[nodiscard]] Compressor<T>& getBandCompressor(int band) noexcept
    {
        assert(band >= 0 && band < MaxBands);
        int safeBand = std::clamp(band, 0, MaxBands - 1);
        return compressors_[safeBand];
    }

    /**
     * @brief Direct constant access to a band's compressor for state queries.
     * @param band Band index. Clamped safely to valid range in release mode.
     * @return Const reference to the requested Compressor.
     */
    [[nodiscard]] const Compressor<T>& getBandCompressor(int band) const noexcept
    {
        assert(band >= 0 && band < MaxBands);
        int safeBand = std::clamp(band, 0, MaxBands - 1);
        return compressors_[safeBand];
    }

    // -- Convenience per-band setters ----------------------------------------

    /**
     * @brief Sets the threshold for a specific band.
     * @param band Target band index.
     * @param dB Threshold in decibels.
     */
    void setBandThreshold(int band, T dB) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setThreshold(dB);
    }

    /**
     * @brief Sets the ratio for a specific band.
     * @param band Target band index.
     * @param ratio Compression ratio (e.g., 4.0 for 4:1).
     */
    void setBandRatio(int band, T ratio) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setRatio(ratio);
    }

    /**
     * @brief Sets the attack time for a specific band.
     * @param band Target band index.
     * @param ms Attack time in milliseconds.
     */
    void setBandAttack(int band, T ms) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setAttack(ms);
    }

    /**
     * @brief Sets the release time for a specific band.
     * @param band Target band index.
     * @param ms Release time in milliseconds.
     */
    void setBandRelease(int band, T ms) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setRelease(ms);
    }

    // -- Queries -------------------------------------------------------------

    /**
     * @brief Gets the current gain reduction applied to a specific band.
     *
     * Metering read, safe from any thread. Bands that are currently disabled
     * report the value frozen from when they were last active.
     *
     * @param band Target band index.
     * @return Gain reduction in decibels (typically <= 0.0).
     */
    [[nodiscard]] T getBandGainReductionDb(int band) const noexcept
    {
        if (band < 0 || band >= MaxBands) return T(0);
        return compressors_[band].getGainReductionDb();
    }

    /** @brief Returns the current number of active bands. */
    [[nodiscard]] int getNumBands() const noexcept { return crossover_.getNumBands(); }

    /** @brief Returns the target frequency of split point `index` in Hz. */
    [[nodiscard]] T getCrossoverFrequency(int index) const noexcept
    {
        return crossover_.getCrossoverFrequency(index);
    }

    /** @brief Returns the crossover slope in dB/oct (12, 24 or 48). */
    [[nodiscard]] int getOrder() const noexcept { return crossover_.getOrder(); }

    /** @brief Returns the crossover processing mode (IIR or linear-phase). */
    [[nodiscard]] typename CrossoverFilter<T, MaxBands>::FilterMode getCrossoverMode() const noexcept
    {
        return crossover_.getFilterMode();
    }

    /** @brief Returns the total latency of the multi-band system. */
    [[nodiscard]] int getLatency() const noexcept
    {
        // Crossover latency + the largest per-band compressor latency (bands
        // are expected to share the same lookahead/detector; see processBlock()).
        int maxBand = 0;
        const int nb = crossover_.getNumBands();
        for (int b = 0; b < nb && b < MaxBands; ++b)
            maxBand = std::max(maxBand, compressors_[b].getLatency());
        return crossover_.getLatency() + maxBand;
    }

    /** @brief Resets all internal states (envelopes, delay lines, etc.). */
    void reset() noexcept
    {
        crossover_.reset();
        for (auto& c : compressors_) c.reset();
        lastNumBands_ = MaxBands;
    }

    /** @brief Serializes crossover topology and per-band compressor states. */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("MBCP"), 1);
        const int n = crossover_.getNumBands();
        w.write("numBands", n);
        w.write("order", crossover_.getOrder());
        w.write("xoverMode", static_cast<int32_t>(crossover_.getFilterMode()));
        char key[24];
        for (int i = 0; i < n - 1; ++i)
        {
            std::snprintf(key, sizeof(key), "x%d", i);
            w.write(key, static_cast<float>(crossover_.getCrossoverFrequency(i)));
        }
        for (int i = 0; i < n; ++i)
        {
            std::snprintf(key, sizeof(key), "band%d", i);
            w.write(key, getBandCompressor(i).getState());
        }
        return w.blob();
    }

    /** @brief Restores topology and band compressors from a blob. */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("MBCP")) return false;
        const int n = std::clamp(r.read("numBands", 3), 2, MaxBands);
        setNumBands(n);
        setOrder(r.read("order", 24));
        setCrossoverMode(static_cast<typename CrossoverFilter<T, MaxBands>::FilterMode>(
            r.read("xoverMode", 0)));
        char key[24];
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
            if (!nested.empty())
                getBandCompressor(i).setState(nested.data(), nested.size());
        }
        return true;
    }

private:
    std::atomic<bool> prepared_ { false };
    int lastNumBands_ = MaxBands; ///< Audio-thread tracker for band re-enable resets.
    CrossoverFilter<T, MaxBands> crossover_;
    std::array<Compressor<T>, MaxBands> compressors_ {};
    std::array<AudioBuffer<T>, MaxBands> bandBuffers_ {};
    std::array<AudioBufferView<T>, MaxBands> views_ {}; ///< Audio-thread scratch.
};

} // namespace dspark
