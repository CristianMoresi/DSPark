// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file MultibandCompressor.h
 * @brief Multi-band compressor using CrossoverFilter + per-band Compressor.
 *
 * Splits the signal into 2–12 frequency bands using a Linkwitz-Riley crossover,
 * compresses each band independently, then sums the bands back together.
 * Each band's compressor is fully configurable with all Compressor features
 * (threshold, ratio, attack, release, knee, character, etc.).
 *
 * Dependencies: CrossoverFilter.h, Compressor.h, AudioBuffer.h.
 *
 * @code
 *   dspark::MultibandCompressor<float> mbc;
 *   mbc.setNumBands(3);
 *   mbc.setCrossoverFrequency(0, 200.0f);
 *   mbc.setCrossoverFrequency(1, 2000.0f);
 *   mbc.prepare(spec);
 *
 *   // Configure per-band compression
 *   mbc.getBandCompressor(0).setThreshold(-20.0f);  // low band
 *   mbc.getBandCompressor(0).setRatio(4.0f);
 *   mbc.getBandCompressor(1).setThreshold(-15.0f);  // mid band
 *   mbc.getBandCompressor(2).setThreshold(-10.0f);  // high band
 *
 *   mbc.processBlock(buffer);
 * @endcode
 */

#include "CrossoverFilter.h"
#include "Compressor.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"

#include <algorithm>
#include <array>

namespace dspark {

/**
 * @class MultibandCompressor
 * @brief Multi-band compressor: crossover split → per-band compression → sum.
 *
 * @tparam T        Sample type (float or double).
 * @tparam MaxBands Maximum number of bands (compile-time, default 12).
 */
template <FloatType T, int MaxBands = 12>
class MultibandCompressor
{
public:
    // -- Lifecycle -----------------------------------------------------------

    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;

        // Prepare crossover
        crossover_.prepare(spec);

        // Allocate per-band buffers
        for (int b = 0; b < MaxBands; ++b)
        {
            bandBuffers_[b].resize(spec.numChannels, spec.maxBlockSize);
            compressors_[b].prepare(spec);
        }

        // Build views array
        updateViews();

        prepared_ = true;
    }

    /**
     * @brief Processes audio through the multi-band compressor.
     *
     * Splits → compresses each band → sums back into the buffer.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;

        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        const int nb  = crossover_.getNumBands();

        // Build truncated views for this block (A11 fix).
        //
        // bandBuffers_ are sized to spec.maxBlockSize, but a given block may
        // be smaller (the host is allowed to call with any nS ≤ maxBlockSize).
        // Passing the full-capacity view would cause the crossover and each
        // compressor to process stale samples beyond nS, corrupting state and
        // burning CPU on data the host will not read. Truncating to nS keeps
        // processing and the envelope detector tied to the actual block size.
        for (int b = 0; b < nb; ++b)
            views_[b] = bandBuffers_[b].toView().getSubView(0, nS);

        // Split into bands
        crossover_.processBlock(buffer, views_.data(), nb);

        // Compress each band independently
        for (int b = 0; b < nb; ++b)
            compressors_[b].processBlock(views_[b]);

        // Sum bands back into output buffer
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* out = buffer.getChannel(ch);
            // Start with band 0
            const T* src0 = bandBuffers_[0].getChannel(ch);
            std::copy(src0, src0 + nS, out);

            // Add remaining bands
            for (int b = 1; b < nb; ++b)
            {
                const T* src = bandBuffers_[b].getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    out[i] += src[i];
            }
        }
    }

    // -- Configuration -------------------------------------------------------

    void setNumBands(int n) noexcept { crossover_.setNumBands(n); }

    void setCrossoverFrequency(int index, T freqHz) noexcept
    {
        crossover_.setCrossoverFrequency(index, freqHz);
    }

    void setOrder(int order) noexcept { crossover_.setOrder(order); }

    void setCrossoverMode(typename CrossoverFilter<T, MaxBands>::FilterMode mode) noexcept
    {
        crossover_.setFilterMode(mode);
    }

    // -- Per-band compressor access ------------------------------------------

    /**
     * @brief Direct access to a band's compressor for full configuration.
     * @param band Band index (0 to numBands-1).
     */
    [[nodiscard]] Compressor<T>& getBandCompressor(int band) noexcept
    {
        return compressors_[band];
    }

    [[nodiscard]] const Compressor<T>& getBandCompressor(int band) const noexcept
    {
        return compressors_[band];
    }

    // -- Convenience per-band setters ----------------------------------------

    void setBandThreshold(int band, T dB) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setThreshold(dB);
    }

    void setBandRatio(int band, T ratio) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setRatio(ratio);
    }

    void setBandAttack(int band, T ms) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setAttack(ms);
    }

    void setBandRelease(int band, T ms) noexcept
    {
        if (band >= 0 && band < MaxBands)
            compressors_[band].setRelease(ms);
    }

    // -- Queries -------------------------------------------------------------

    [[nodiscard]] T getBandGainReductionDb(int band) const noexcept
    {
        if (band < 0 || band >= MaxBands) return T(0);
        return compressors_[band].getGainReductionDb();
    }

    [[nodiscard]] int getNumBands() const noexcept { return crossover_.getNumBands(); }
    [[nodiscard]] int getLatency() const noexcept { return crossover_.getLatency(); }

    /** @brief Resets all internal state. */
    void reset() noexcept
    {
        crossover_.reset();
        for (auto& c : compressors_) c.reset();
    }

private:
    void updateViews() noexcept
    {
        for (int b = 0; b < MaxBands; ++b)
            views_[b] = bandBuffers_[b].toView();
    }

    AudioSpec spec_ {};
    bool prepared_ = false;
    CrossoverFilter<T, MaxBands> crossover_;
    std::array<Compressor<T>, MaxBands> compressors_ {};
    std::array<AudioBuffer<T>, MaxBands> bandBuffers_ {};
    std::array<AudioBufferView<T>, MaxBands> views_ {};
};

} // namespace dspark
