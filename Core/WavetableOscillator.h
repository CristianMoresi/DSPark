// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WavetableOscillator.h
 * @brief Bandlimited wavetable oscillator with mipmap anti-aliasing.
 *
 * A wavetable oscillator using a flattened contiguous memory layout for
 * cache-friendly access, bitwise phase masking, and 3rd-order Hermite
 * interpolation. Fractional mipmapping with a quadratic crossfade prevents
 * aliasing at all frequencies and keeps frequency sweeps timbre-continuous.
 *
 * Table content is sample-rate independent (each mip level stores a fixed
 * harmonic count); prepare() re-derives the per-level cutoff frequencies, so
 * the build and load methods may be called before or after prepare().
 *
 * @note Setup methods (build*, load*) allocate memory and perform heavy
 * mathematical operations (a direct DFT: loadWavetable costs
 * O(size * harmonics)). They MUST NOT be called on the real-time audio
 * thread, nor while the audio thread is generating.
 *
 * Threading: owner-managed. Setters and generation belong to the owning
 * (audio) thread; build/load/prepare are setup-time.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, Phasor.h,
 * Interpolation.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "Phasor.h"
#include "Interpolation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace dspark {

/**
 * @class WavetableOscillator
 * @brief Professional mipmapped wavetable oscillator for bandlimited synthesis.
 *
 * @tparam T Sample type (float or double). Must satisfy std::floating_point.
 */
template <FloatType T>
class WavetableOscillator
{
public:
    static constexpr int kTableSize = 2048;
    static constexpr int kTableMask = kTableSize - 1; // Used for ultra-fast wrapping
    static constexpr int kMaxMipLevels = 12;

    WavetableOscillator() = default;

    /**
     * @brief Prepares the oscillator for a given sample rate.
     *
     * Re-derives the mip cutoff frequencies for tables that were already
     * built, so a rate change after build*() keeps the anti-aliasing exact.
     *
     * @param sampleRate Sample rate in Hz. Must be > 0 (invalid values,
     * including NaN, are ignored).
     */
    void prepare(double sampleRate)
    {
        assert(sampleRate > 0.0);
        if (!(sampleRate > 0.0)) return;

        sampleRate_ = sampleRate;
        phasor_.prepare(sampleRate);
        updateMipCutoffs();
    }

    /** 
     * @brief Prepares from AudioSpec (unified API). 
     * @param spec Framework AudioSpec instance containing sample rate.
     */
    void prepare(const AudioSpec& spec) 
    { 
        prepare(spec.sampleRate); 
    }

    // -- Built-in waveform generators (OFFLINE ONLY) ----------------------------

    /**
     * @brief Builds a bandlimited sawtooth wavetable with mipmaps.
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void buildSaw()
    {
        buildFromHarmonics([](int harmonic) -> T {
            return (harmonic % 2 == 0) ? T(-1.0 / harmonic) : T(1.0 / harmonic);
        });
    }

    /**
     * @brief Builds a bandlimited square wavetable with mipmaps.
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void buildSquare()
    {
        buildFromHarmonics([](int harmonic) -> T {
            return (harmonic % 2 == 0) ? T(0) : T(1.0 / harmonic);
        });
    }

    /**
     * @brief Builds a bandlimited triangle wavetable with mipmaps.
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void buildTriangle()
    {
        buildFromHarmonics([](int harmonic) -> T {
            if (harmonic % 2 == 0) return T(0);
            T sign = ((harmonic / 2) % 2 == 0) ? T(1) : T(-1);
            return sign / static_cast<T>(harmonic * harmonic);
        });
    }

    /**
     * @brief Builds a pure sine wavetable. 
     * Requires only 1 level since it contains only the fundamental.
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void buildSine()
    {
        numMipLevels_ = 1;
        mipData_.assign(kTableSize, T(0)); // Contiguous layout

        for (int i = 0; i < kTableSize; ++i)
        {
            T phase = twoPi<T> * static_cast<T>(i) / static_cast<T>(kTableSize);
            mipData_[static_cast<size_t>(i)] = std::sin(phase);
        }

        updateMipCutoffs();
    }

    /**
     * @brief Builds contiguous mipmapped tables from a harmonic amplitude function.
     *
     * @tparam HarmonicFunc Callable signature: `T(int harmonic)`
     * @param harmonicFunc Function returning the amplitude for harmonic N (starts at 1).
     * @note Allocates memory. Call only during initialization/preparation.
     */
    template <typename HarmonicFunc>
    void buildFromHarmonics(HarmonicFunc harmonicFunc)
    {
        numMipLevels_ = kMaxMipLevels;
        mipData_.assign(static_cast<size_t>(numMipLevels_ * kTableSize), T(0));

        const int maxTableHarmonics = kTableSize / 2; // Absolute Nyquist limit of the table itself

        for (int level = 0; level < numMipLevels_; ++level)
        {
            // Fixed harmonic budget per level (2^(levels-1-level)), computed in
            // exact integer arithmetic. This is what makes the table content
            // sample-rate independent: prepare() only re-derives the cutoffs.
            int maxHarmonics = 1 << (numMipLevels_ - 1 - level);
            maxHarmonics = std::clamp(maxHarmonics, 1, maxTableHarmonics);

            size_t offset = static_cast<size_t>(level * kTableSize);

            for (int h = 1; h <= maxHarmonics; ++h)
            {
                T amplitude = harmonicFunc(h);
                if (amplitude == T(0)) continue;

                for (int i = 0; i < kTableSize; ++i)
                {
                    T phase = twoPi<T> * static_cast<T>(h) * static_cast<T>(i) / static_cast<T>(kTableSize);
                    mipData_[offset + static_cast<size_t>(i)] += amplitude * std::sin(phase);
                }
            }
        }

        updateMipCutoffs();

        // Normalise ALL levels with ONE global factor (the peak of the richest
        // level). Per-level peak normalisation made the level/timbre jump
        // slightly at every mip crossfade, because the Gibbs overshoot varies
        // with the harmonic count.
        normalizeAllLevelsGlobally();
    }

    /**
     * @brief Loads and analyzes a custom single-cycle wavetable using DFT.
     *
     * Performs a Discrete Fourier Transform strictly on the original data length
     * to avoid low-pass smearing, then reconstructs bandlimited mipmaps.
     * 
     * @param data Pointer to wavetable samples.
     * @param size Number of samples in the input data.
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void loadWavetable(const T* data, int size)
    {
        if (!data || size <= 0) return;

        // Perform DFT on raw input data to extract true harmonics without
        // interpolation loss. The DC term is discarded (an oscillator should
        // not reproduce offset), and for even sizes the Nyquist bin is
        // excluded too: the 2/N single-sided scaling below would count it
        // twice ((size - 1) / 2 stops one bin short of it).
        int rawHarmonicLimit = (size - 1) / 2;
        int targetNyquistLimit = kTableSize / 2;
        int maxHarmonics = std::min(rawHarmonicLimit, targetNyquistLimit);
        if (maxHarmonics < 1) return; // size < 3: no extractable harmonics

        std::vector<T> cosCoeffs(static_cast<size_t>(maxHarmonics + 1), T(0));
        std::vector<T> sinCoeffs(static_cast<size_t>(maxHarmonics + 1), T(0));

        for (int h = 1; h <= maxHarmonics; ++h)
        {
            T sumCos = T(0), sumSin = T(0);
            for (int i = 0; i < size; ++i)
            {
                T phase = twoPi<T> * static_cast<T>(h) * static_cast<T>(i) / static_cast<T>(size);
                sumCos += data[i] * std::cos(phase);
                sumSin += data[i] * std::sin(phase);
            }
            cosCoeffs[static_cast<size_t>(h)] = sumCos * T(2) / static_cast<T>(size);
            sinCoeffs[static_cast<size_t>(h)] = sumSin * T(2) / static_cast<T>(size);
        }

        numMipLevels_ = kMaxMipLevels;
        mipData_.assign(static_cast<size_t>(numMipLevels_ * kTableSize), T(0));

        for (int level = 0; level < numMipLevels_; ++level)
        {
            // Same exact integer harmonic budget as buildFromHarmonics,
            // additionally capped by what the source material contains.
            int maxH = 1 << (numMipLevels_ - 1 - level);
            maxH = std::clamp(maxH, 1, maxHarmonics);

            size_t offset = static_cast<size_t>(level * kTableSize);

            for (int h = 1; h <= maxH; ++h)
            {
                T a = cosCoeffs[static_cast<size_t>(h)];
                T b = sinCoeffs[static_cast<size_t>(h)];
                if (a == T(0) && b == T(0)) continue;

                for (int i = 0; i < kTableSize; ++i)
                {
                    T phase = twoPi<T> * static_cast<T>(h) * static_cast<T>(i) / static_cast<T>(kTableSize);
                    mipData_[offset + static_cast<size_t>(i)] += a * std::cos(phase) + b * std::sin(phase);
                }
            }
        }

        updateMipCutoffs();

        // Single global normalisation factor across all mip levels (see
        // buildFromHarmonics for the rationale).
        normalizeAllLevelsGlobally();
    }

    // -- Playback (REAL-TIME SAFE) ----------------------------------------------

    /**
     * @brief Sets the fundamental oscillation frequency.
     * @param frequencyHz Frequency in Hz.
     * @note Not inherently thread-safe. Synchronization should be managed externally.
     */
    void setFrequency(T frequencyHz) noexcept
    {
        // NaN is ignored, mirroring the internal Phasor: otherwise the pitch
        // would keep the old value while the mip selection went to the
        // dullest level (inconsistent timbre).
        if (frequencyHz != frequencyHz) return;
        frequency_ = frequencyHz;
        phasor_.setFrequency(frequencyHz);
    }

    /**
     * @brief Returns the current internal frequency.
     * @return Frequency in Hz.
     */
    [[nodiscard]] T getFrequency() const noexcept { return frequency_; }

    /**
     * @brief Generates the next sample using Hermite interpolation.
     * @return Output sample, nominally in [-1.0, 1.0] (tables are peak
     * normalised; interpolation may overshoot slightly between table points).
     */
    [[nodiscard]] inline T getSample() noexcept
    {
        T phase = phasor_.advance();
        return readTable(phase);
    }

    /**
     * @brief Generates a block of samples.
     * @param output Pointer to output buffer.
     * @param numSamples Number of samples to process.
     */
    void processBlock(T* output, int numSamples) noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            output[i] = getSample();
    }

    /**
     * @brief Fills every channel of the view with the generated waveform.
     * Satisfies the GeneratorProcessor concept (mono source, all channels equal).
     */
    void generateBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        for (int i = 0; i < nS; ++i)
        {
            const T s = getSample();
            for (int ch = 0; ch < nCh; ++ch)
                buffer.getChannel(ch)[i] = s;
        }
    }

    /**
     * @brief Hard resets the oscillator phase.
     * @param phase Initial phase normalized [0.0, 1.0].
     */
    void reset(T phase = T(0)) noexcept
    {
        phasor_.reset(phase);
    }

private:

    /**
     * @brief Reads a sample from a specific mipmap level using Hermite
     * interpolation (the shared Catmull-Rom kernel from Interpolation.h).
     * Employs bitwise masking for zero-branching index wrapping.
     *
     * @param phase Phase in [0, 1).
     * @param level Mipmap level index.
     * @return Interpolated sample.
     */
    [[nodiscard]] inline T readFromLevel(T phase, int level) const noexcept
    {
        T pos = phase * static_cast<T>(kTableSize);
        int i1 = static_cast<int>(pos); // Fast truncation replacing std::floor
        T frac = pos - static_cast<T>(i1);

        // Bitwise mask wrapping ensures bounds without modulo operator penalty
        int i0 = (i1 - 1) & kTableMask;
        i1 = i1 & kTableMask;
        int i2 = (i1 + 1) & kTableMask;
        int i3 = (i1 + 2) & kTableMask;

        size_t offset = static_cast<size_t>(level * kTableSize);
        const T* table = &mipData_[offset];

        return interpolateHermite(table[i0], table[i1], table[i2], table[i3], frac);
    }

    /**
     * @brief Reads and crossfades between adjacent mipmaps to prevent timbre stepping.
     */
    [[nodiscard]] inline T readTable(T phase) const noexcept
    {
        if (mipData_.empty()) return T(0);

        T levelF = selectMipLevelFloat();
        int level0 = static_cast<int>(levelF);
        
        // Optimize crossfade edge-case
        if (levelF == static_cast<T>(level0)) 
        {
            return readFromLevel(phase, level0);
        }

        int level1 = std::min(level0 + 1, numMipLevels_ - 1);
        T frac = levelF - static_cast<T>(level0);

        T s0 = readFromLevel(phase, level0);
        T s1 = readFromLevel(phase, level1);

        // Quadratic weighting of the brighter (lower) level: its topmost
        // harmonics start aliasing as soon as the frequency moves past its
        // design point, so its weight must die off much faster than linear.
        // (1-t)^2 keeps the worst-case alias contribution ~12 dB lower than a
        // linear crossfade at mid-octave while preserving a smooth timbre.
        const T w0 = (T(1) - frac) * (T(1) - frac);
        return s1 + w0 * (s0 - s1);
    }

    /**
     * @brief Calculates fractional mipmap level based on current frequency.
     */
    [[nodiscard]] inline T selectMipLevelFloat() const noexcept
    {
        T absFreq = std::abs(frequency_);

        if (numMipLevels_ <= 1 || absFreq <= mipMaxFreq_[0])
            return T(0);

        for (int i = 1; i < numMipLevels_; ++i)
        {
            if (absFreq <= mipMaxFreq_[static_cast<size_t>(i)])
            {
                T freqLow  = mipMaxFreq_[static_cast<size_t>(i - 1)];
                T freqHigh = mipMaxFreq_[static_cast<size_t>(i)];
                T t = (absFreq - freqLow) / (freqHigh - freqLow + T(1e-9)); 
                return static_cast<T>(i - 1) + t;
            }
        }

        return static_cast<T>(numMipLevels_ - 1);
    }

    /**
     * @brief Derives the per-level cutoff frequencies from the current sample
     * rate. Level content is rate-independent (fixed harmonic budgets), so
     * this is all prepare() needs to redo after a rate change.
     */
    void updateMipCutoffs()
    {
        if (numMipLevels_ <= 0) return;
        mipMaxFreq_.resize(static_cast<size_t>(numMipLevels_));

        const T nyquist = static_cast<T>(sampleRate_ / 2.0);
        for (int level = 0; level < numMipLevels_; ++level)
            mipMaxFreq_[static_cast<size_t>(level)] =
                nyquist / static_cast<T>(1 << (numMipLevels_ - 1 - level));
    }

    /**
     * @brief Normalises every mip level with one shared factor.
     *
     * The factor is the global peak across all levels (in practice the
     * richest level: lower levels are subsets of its harmonics and cannot
     * exceed it). This preserves the exact relative energy between levels,
     * so mip crossfades are level- and timbre-continuous.
     */
    void normalizeAllLevelsGlobally()
    {
        T maxVal = T(0);
        for (const T v : mipData_)
            maxVal = std::max(maxVal, std::abs(v));

        if (maxVal > T(0))
        {
            const T invMax = T(1) / maxVal;
            for (T& v : mipData_)
                v *= invMax;
        }
    }

    double sampleRate_ = 48000.0;
    T frequency_ = T(440);

    Phasor<T> phasor_;

    int numMipLevels_ = 0;
    // Cache-friendly contiguous flat layout (Level0 + Level1 + ...)
    std::vector<T> mipData_; 
    std::vector<T> mipMaxFreq_;
};

} // namespace dspark