// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WavetableOscillator.h
 * @brief Bandlimited wavetable oscillator with mipmap anti-aliasing.
 *
 * A wavetable oscillator using a flattened contiguous memory layout for
 * cache-friendly access, bitwise phase masking, and 3rd-order Hermite
 * interpolation. Half-octave mipmaps are crossfaded ONLY between levels whose
 * harmonics are audibly alias-free at the current pitch, so a frequency sweep
 * is both timbre-continuous and clean.
 *
 * Anti-aliasing contract: a level is used at a frequency f only while its
 * highest harmonic stays below max(fs/2, fs - 20 kHz). Harmonics between
 * Nyquist and fs - 20 kHz fold to between 20 kHz and Nyquist, above the
 * audible band, which buys brightness at no audible cost. The crossfade
 * always pairs the brightest such level with the next duller one, so the
 * spectrum is complete up to at least (fs - 20 kHz) / 2 (14 kHz at 48 kHz)
 * and partially present above it. Measured on a saw at 48 kHz, worst alias
 * below 20 kHz: the previous octave mipmaps with a crossfade into the next
 * BRIGHTER level reached -39 dB at 440 Hz and -16 dB at 7 kHz.
 *
 * Table content is sample-rate independent (each mip level stores a fixed
 * harmonic count); prepare() re-derives the per-level cutoff frequencies, so
 * the build and load methods may be called before or after prepare().
 *
 * @note Setup methods (build*, load*) allocate memory and run FFT synthesis
 * (loadWavetable also runs an O(size * harmonics) analysis DFT). They MUST
 * NOT be called on the real-time audio thread, nor while the audio thread is
 * generating.
 *
 * Threading: owner-managed. Setters and generation belong to the owning
 * (audio) thread; build/load/prepare are setup-time.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, Phasor.h,
 * Interpolation.h, FFT.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "Phasor.h"
#include "Interpolation.h"
#include "FFT.h"

#include <array>
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
    /// Mip levels, half an octave apart (harmonic budgets in kLevelHarmonics).
    static constexpr int kMaxMipLevels = 20;

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
        updateMipSelection();
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
        levelHarmonics_.assign(1, 1);

        for (int i = 0; i < kTableSize; ++i)
        {
            const double phase = twoPi<double> * static_cast<double>(i) / static_cast<double>(kTableSize);
            mipData_[static_cast<size_t>(i)] = static_cast<T>(std::sin(phase));
        }

        updateMipCutoffs();
        updateMipSelection();
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
        std::vector<double> cosAmp(kMaxHarmonics + 1, 0.0);
        std::vector<double> sinAmp(kMaxHarmonics + 1, 0.0);
        for (int h = 1; h <= kMaxHarmonics; ++h)
            sinAmp[static_cast<size_t>(h)] = static_cast<double>(harmonicFunc(h));
        buildLevels(cosAmp, sinAmp, kMaxHarmonics);
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
        int maxHarmonics = std::min(rawHarmonicLimit, kMaxHarmonics);
        if (maxHarmonics < 1) return; // size < 3: no extractable harmonics

        // Analysis in double whatever T is (setup-time work).
        std::vector<double> cosCoeffs(static_cast<size_t>(maxHarmonics + 1), 0.0);
        std::vector<double> sinCoeffs(static_cast<size_t>(maxHarmonics + 1), 0.0);

        for (int h = 1; h <= maxHarmonics; ++h)
        {
            double sumCos = 0.0, sumSin = 0.0;
            for (int i = 0; i < size; ++i)
            {
                // Integer phase reduction keeps the argument small (exact).
                const auto k = static_cast<long long>(h) * i % size;
                const double phase = twoPi<double> * static_cast<double>(k) / static_cast<double>(size);
                sumCos += static_cast<double>(data[i]) * std::cos(phase);
                sumSin += static_cast<double>(data[i]) * std::sin(phase);
            }
            cosCoeffs[static_cast<size_t>(h)] = sumCos * 2.0 / static_cast<double>(size);
            sinCoeffs[static_cast<size_t>(h)] = sumSin * 2.0 / static_cast<double>(size);
        }

        buildLevels(cosCoeffs, sinCoeffs, maxHarmonics);
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
        const bool changed = frequencyHz != frequency_;
        frequency_ = frequencyHz;
        phasor_.setFrequency(frequencyHz);
        if (changed) updateMipSelection();
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
     * @brief Reads the selected level, crossfaded with the next duller one.
     *
     * Both levels are audibly alias-free at the current pitch (see
     * updateMipSelection()), and the duller level's harmonics are a subset
     * of the brighter one's with identical amplitudes and phases, so the
     * crossfade only scales the extra top harmonics: no comb or phase
     * artefacts, and a sweep changes the timbre continuously.
     */
    [[nodiscard]] inline T readTable(T phase) const noexcept
    {
        if (mipData_.empty()) return T(0);

        const T s0 = readFromLevel(phase, selLevel_);
        if (selNext_ == selLevel_ || selWeight_ >= T(1))
            return s0;
        const T s1 = readFromLevel(phase, selNext_);
        return s1 + selWeight_ * (s0 - s1);
    }

    /**
     * @brief Chooses the level pair and crossfade weight for the current pitch.
     *
     * Level L is alias-safe up to safeFreq_[L] (its top harmonic reaches the
     * fold limit there). For a pitch in (safeFreq_[L-1], safeFreq_[L]] the
     * brightest safe level is L: it is blended with L+1, with L's weight
     * falling from 1 at the bottom of the range to 0 at safeFreq_[L], so the
     * output is continuous where the choice moves on to the next pair. Runs
     * on frequency changes only, never per sample.
     */
    void updateMipSelection() noexcept
    {
        selLevel_ = 0;
        selNext_ = 0;
        selWeight_ = T(1);
        if (numMipLevels_ <= 1 || safeFreq_.size() < static_cast<size_t>(numMipLevels_))
            return;

        const T f = std::abs(frequency_);
        const int last = numMipLevels_ - 1;
        int level = 0;
        while (level < last && f > safeFreq_[static_cast<size_t>(level)])
            ++level;

        selLevel_ = level;
        selNext_ = std::min(level + 1, last);
        if (selNext_ == level) return;

        // Bottom of this level's range: the previous level's safe limit, or,
        // for level 0, the point one level-spacing below its own limit.
        const T hi = safeFreq_[static_cast<size_t>(level)];
        const T lo = (level > 0)
            ? safeFreq_[static_cast<size_t>(level - 1)]
            : hi * static_cast<T>(levelHarmonics_[1]) / static_cast<T>(levelHarmonics_[0]);
        const T span = hi - lo;
        selWeight_ = (span > T(0)) ? std::clamp((hi - f) / span, T(0), T(1)) : T(1);
    }

    /**
     * @brief Derives the per-level alias-safe frequencies from the sample
     * rate. Level content is rate-independent (fixed harmonic budgets), so
     * this is all prepare() needs to redo after a rate change.
     */
    void updateMipCutoffs()
    {
        if (numMipLevels_ <= 0) return;
        safeFreq_.resize(static_cast<size_t>(numMipLevels_));

        // Highest harmonic frequency whose alias still lands above 20 kHz
        // (never below Nyquist itself: at low rates there is no free band).
        const double fold = std::max(sampleRate_ * 0.5, sampleRate_ - 20000.0);
        for (int level = 0; level < numMipLevels_; ++level)
            safeFreq_[static_cast<size_t>(level)] = static_cast<T>(
                fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level)]));
    }

    /**
     * @brief Synthesises every mip level from harmonic amplitudes by inverse
     *        FFT (exact band-limited content, O(levels * N log N)).
     *
     * @param cosAmp    Cosine amplitude per harmonic (index 1..available).
     * @param sinAmp    Sine amplitude per harmonic (index 1..available).
     * @param available Highest harmonic present in the source.
     */
    void buildLevels(const std::vector<double>& cosAmp, const std::vector<double>& sinAmp,
                     int available)
    {
        numMipLevels_ = kMaxMipLevels;
        mipData_.assign(static_cast<size_t>(numMipLevels_ * kTableSize), T(0));
        levelHarmonics_.assign(static_cast<size_t>(numMipLevels_), 1);

        FFTReal<double> fft(static_cast<size_t>(kTableSize));
        std::vector<double> spec(static_cast<size_t>(kTableSize + 2));
        std::vector<double> cycle(static_cast<size_t>(kTableSize));
        const double half = 0.5 * static_cast<double>(kTableSize);

        for (int level = 0; level < numMipLevels_; ++level)
        {
            const int budget = std::min(kLevelHarmonics[static_cast<size_t>(level)], available);
            levelHarmonics_[static_cast<size_t>(level)] = std::max(budget, 1);

            // x[n] = sum a_h cos(2 pi h n / N) + b_h sin(2 pi h n / N) has the
            // one-sided spectrum X[h] = (N/2) (a_h - j b_h) under the inverse
            // FFT's 1/N scaling.
            std::fill(spec.begin(), spec.end(), 0.0);
            for (int h = 1; h <= budget; ++h)
            {
                spec[static_cast<size_t>(2 * h)]     =  half * cosAmp[static_cast<size_t>(h)];
                spec[static_cast<size_t>(2 * h + 1)] = -half * sinAmp[static_cast<size_t>(h)];
            }
            fft.inverse(spec.data(), cycle.data());

            T* dst = &mipData_[static_cast<size_t>(level * kTableSize)];
            for (int i = 0; i < kTableSize; ++i)
                dst[i] = static_cast<T>(cycle[static_cast<size_t>(i)]);
        }

        updateMipCutoffs();

        // Normalise ALL levels with ONE global factor (the peak of the richest
        // level). Per-level peak normalisation made the level/timbre jump
        // slightly at every mip crossfade, because the Gibbs overshoot varies
        // with the harmonic count.
        normalizeAllLevelsGlobally();
        updateMipSelection();
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

    /// Highest harmonic a level may hold: one below the table's own Nyquist
    /// (a Nyquist-bin cosine would need a different scaling, and no pitch
    /// could use it anyway).
    static constexpr int kMaxHarmonics = kTableSize / 2 - 1;

    /// Harmonic budget per level, half an octave apart from the table limit
    /// down to a single sine.
    static constexpr std::array<int, kMaxMipLevels> kLevelHarmonics = {
        kMaxHarmonics, 724, 512, 362, 256, 181, 128, 90, 64, 45,
        32, 22, 16, 11, 8, 5, 4, 3, 2, 1 };

    double sampleRate_ = 48000.0;
    T frequency_ = T(440);

    Phasor<T> phasor_;

    int numMipLevels_ = 0;
    // Cache-friendly contiguous flat layout (Level0 + Level1 + ...)
    std::vector<T> mipData_;
    std::vector<int> levelHarmonics_;   ///< Actual top harmonic per level.
    std::vector<T> safeFreq_;           ///< Alias-safe pitch limit per level.

    // Level pair and crossfade weight for the current pitch
    // (updateMipSelection(); read per sample by readTable()).
    int selLevel_ = 0;
    int selNext_ = 0;
    T selWeight_ = T(1);
};

} // namespace dspark