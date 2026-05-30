// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file MultibandCompressor.h
 * @brief Multi-band compressor using CrossoverFilter + per-band Compressor.
 *
 * Splits the signal into 2–12 frequency bands using a Linkwitz-Riley crossover,
 * compresses each band independently, then sums the bands back together.
 * Includes safety bounds and phase-alignment awareness.
 *
 * Dependencies: CrossoverFilter.h, Compressor.h, AudioBuffer.h.
 */

#include "CrossoverFilter.h"
#include "Compressor.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <cassert>

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

    /**
     * @brief Prepares the multiband compressor and internal buffers for processing.
     * @param spec The audio specification (sample rate, max block size, channels).
     */
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

        updateViews();
        prepared_ = true;
    }

    /**
     * @brief Processes audio through the multi-band compressor.
     * @param buffer In/Out audio buffer view.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;

        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        const int nb  = crossover_.getNumBands();

        // 1. Truncate views to actual block size (A11 fix)
        for (int b = 0; b < nb; ++b)
            views_[b] = bandBuffers_[b].toView().getSubView(0, nS);

        // 2. Split into bands
        crossover_.processBlock(buffer, views_.data(), nb);

        // 3. Compress each band independently.
        // IMPORTANT: bands are summed directly, so every band must share the SAME
        // latency. Compressors default to 0 lookahead (latency 0) → coherent sum.
        // If you enable lookahead, set the SAME value on every band via
        // getBandCompressor(b).setLookahead(); a non-uniform per-band lookahead
        // would phase-cancel at the crossover regions (this class does not
        // auto-delay-compensate divergent per-band latencies).
        for (int b = 0; b < nb; ++b)
            compressors_[b].processBlock(views_[b]);

        // 4. Sum bands back into output buffer (Optimized for SIMD)
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* const __restrict out = buffer.getChannel(ch);
            const T* const __restrict src0 = bandBuffers_[0].getChannel(ch);
            
            // Base copy (band 0)
            std::copy(src0, src0 + nS, out);

            // Accumulate remaining active bands
            for (int b = 1; b < nb; ++b)
            {
                const T* const __restrict src = bandBuffers_[b].getChannel(ch);
                
                // Explicit auto-vectorization hint pattern
                #pragma omp simd
                for (int i = 0; i < nS; ++i)
                {
                    out[i] += src[i];
                }
            }
        }
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Sets the number of active frequency bands.
     * @warning Must not be called concurrently with processBlock() unless external locking is used.
     * @param n Number of bands (clamped between 1 and MaxBands).
     */
    void setNumBands(int n) noexcept 
    { 
        crossover_.setNumBands(std::clamp(n, 1, MaxBands)); 
    }

    /**
     * @brief Sets the crossover frequency for a specific split point.
     * @param index Split point index (0 is between band 0 and 1).
     * @param freqHz Target frequency in Hertz.
     */
    void setCrossoverFrequency(int index, T freqHz) noexcept
    {
        crossover_.setCrossoverFrequency(index, freqHz);
    }

    /**
     * @brief Sets the filter order for the Linkwitz-Riley crossover (e.g., 2, 4, 8).
     * @param order Filter order (even numbers typically).
     */
    void setOrder(int order) noexcept 
    { 
        crossover_.setOrder(order); 
    }

    /**
     * @brief Sets the phase/processing mode of the crossover (e.g., IIR, FIR).
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
    
    /** @brief Returns the total latency of the multi-band system. */
    [[nodiscard]] int getLatency() const noexcept
    {
        // Crossover latency + the largest per-band compressor lookahead (bands are
        // expected to share the same lookahead; see processBlock()).
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
