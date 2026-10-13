// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WavetableOscillator.h
 * @brief Band-limited wavetable oscillator over a mipmapped WavetableBank.
 *
 * Plays its own single-cycle table (built from harmonics or loaded from a
 * cycle) or any external WavetableBank, morphing across the bank's frames
 * with setPosition(). The tables, the mip selection and the interpolation are
 * WavetableBank's: half-octave levels crossfaded only between levels whose
 * harmonics are audibly alias-free at the current pitch, and an optimal
 * 6-point polynomial interpolator over tables that keep four samples per
 * period of their top harmonic, with the interpolator's droop pre-compensated.
 *
 * Anti-aliasing contract: a level is used at a frequency f only while its
 * highest harmonic stays below max(fs/2, fs - 20 kHz). Harmonics between
 * Nyquist and fs - 20 kHz fold to between 20 kHz and Nyquist, above the
 * audible band, which buys brightness at no audible cost. The crossfade
 * always pairs the brightest such level with the next duller one, so the
 * spectrum is complete up to at least (fs - 20 kHz) / 2 (14 kHz at 48 kHz)
 * and partially present above it.
 *
 * Measured on a saw at 48 kHz, worst inharmonic component below 20 kHz
 * relative to the strongest harmonic: -163 dB or better from 27.5 Hz to
 * 5 kHz in double. The previous reader (4-point Hermite over 2048-sample
 * levels) measured -72 dB at 27.5 Hz and -85 dB at 55 Hz: its interpolation
 * images, not the mip levels, set that floor.
 *
 * Table content is sample-rate independent; prepare() re-derives the mip
 * selection, so the build and load methods may be called before or after it.
 *
 * @note Setup methods (build*, load*) allocate memory and run FFT synthesis
 * (loadWavetable also analyses the cycle). They MUST NOT be called on the
 * real-time audio thread, nor while the audio thread is generating.
 *
 * Threading: owner-managed. Setters and generation belong to the owning
 * (audio) thread; build/load/prepare/setBank are setup-time.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, Phasor.h,
 * WavetableBank.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "Phasor.h"
#include "WavetableBank.h"

#include <algorithm>
#include <cassert>
#include <cmath>

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
    /// Samples of the shortest level table (brighter levels hold more; see WavetableBank).
    static constexpr int kTableSize = 2048;
    static constexpr int kTableMask = kTableSize - 1;
    /// Mip levels, half an octave apart.
    static constexpr int kMaxMipLevels = WavetableBank::kMipLevels;

    WavetableOscillator() = default;

    /**
     * @brief Prepares the oscillator for a given sample rate.
     *
     * Re-derives the mip selection for tables that were already built, so a
     * rate change after build*() keeps the anti-aliasing exact.
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
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void buildSine()
    {
        own_.buildFromHarmonics(1, 1, [](int, int) {
            return WavetableBank::Harmonic { 0.0, 1.0 };
        });
        updateMipSelection();
    }

    /**
     * @brief Builds contiguous mipmapped tables from a harmonic amplitude function.
     *
     * @tparam HarmonicFunc Callable signature: `T(int harmonic)`
     * @param harmonicFunc Function returning the sine amplitude for harmonic N
     *        (starts at 1, up to WavetableBank::kMaxHarmonics).
     * @note Allocates memory. Call only during initialization/preparation.
     */
    template <typename HarmonicFunc>
    void buildFromHarmonics(HarmonicFunc harmonicFunc)
    {
        own_.buildFromHarmonics(1, WavetableBank::kMaxHarmonics, [&](int, int h) {
            return WavetableBank::Harmonic { 0.0, static_cast<double>(harmonicFunc(h)) };
        });
        updateMipSelection();
    }

    /**
     * @brief Loads and analyzes a custom single-cycle wavetable.
     *
     * The cycle is analysed exactly on its own length (no resampling), its
     * DC discarded, and the mip levels rebuilt from its harmonics.
     *
     * @param data Pointer to wavetable samples.
     * @param size Number of samples in the input data (3 or more;
     *        smaller sizes and null data are ignored).
     * @note Allocates memory. Call only during initialization/preparation.
     */
    void loadWavetable(const T* data, int size)
    {
        if (!data || size < 3) return;
        own_.loadFrames(data, size, 1, WavetableBank::Normalization::PerFrame);
        updateMipSelection();
    }

    /**
     * @brief Plays an external bank instead of the oscillator's own table.
     *
     * The bank is not owned and must outlive its use; it must not be rebuilt
     * while this oscillator generates. nullptr returns to the own table.
     * Setup-time (same contract as the build methods).
     */
    void setBank(const WavetableBank* bank) noexcept
    {
        external_ = bank;
        updateMipSelection();
    }

    /**
     * @brief Morph position across the bank's frames, 0 .. frames - 1
     *        (fractional positions blend adjacent frames; clamped on read).
     */
    void setPosition(T position) noexcept
    {
        if (position != position) return;
        position_ = position;
    }

    [[nodiscard]] T getPosition() const noexcept { return position_; }

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
     * @brief Generates the next sample.
     * @return Output sample, nominally in [-1.0, 1.0] (tables are peak
     * normalised; interpolation may overshoot slightly between table points).
     */
    [[nodiscard]] inline T getSample() noexcept
    {
        const T phase = phasor_.advance();
        return bank().template read<T>(static_cast<double>(phase),
                                       static_cast<double>(position_), mip_);
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
    [[nodiscard]] const WavetableBank& bank() const noexcept
    {
        return external_ != nullptr ? *external_ : own_;
    }

    /** @brief Level pair and crossfade weight for the current pitch (on changes only). */
    void updateMipSelection() noexcept
    {
        mip_ = bank().selectMip(static_cast<double>(frequency_), sampleRate_);
    }

    double sampleRate_ = 48000.0;
    T frequency_ = T(440);
    T position_ = T(0);

    Phasor<T> phasor_;

    WavetableBank own_;
    const WavetableBank* external_ = nullptr;
    WavetableBank::Mip mip_;
};

} // namespace dspark
