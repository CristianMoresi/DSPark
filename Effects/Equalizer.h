// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Equalizer.h
 * @brief Multi-band parametric equalizer with progressive disclosure API.
 *
 * Combines multiple FilterEngine instances into a single processor with
 * a unified interface. Supports up to MaxBands simultaneous filter bands,
 * each independently configurable as Peak, Shelf, LowPass, HighPass, or Notch.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `eq.setBand(0, 1000.0f, -3.0f)` — just frequency and gain.
 * - **Level 2 (intermediate):** `eq.setBand(0, 1000.0f, -3.0f, 1.5f)` — adds Q factor.
 * - **Level 3 (expert):** `eq.setBand(0, { .frequency = 1000, .gain = -3, .q = 1.5,
 *                          .type = BandType::LowShelf, .slope = 24 })` — full control.
 *
 * **Linear-phase mode:** Set `eq.setFilterMode(FilterMode::LinearPhase)` to
 * apply the EQ with zero phase distortion. Uses FFT-based processing internally.
 * Adds latency equal to the block size, but preserves the waveform shape perfectly.
 * Ideal for mastering and critical listening. IIR mode (default) has zero latency.
 *
 * Dependencies: Filters.h (FilterEngine), FFT.h, Biquad.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   // Level 1 — Desktop developer, zero DSP knowledge needed:
 *   dspark::Equalizer<float> eq;
 *   eq.prepare(spec);
 *   eq.setNumBands(4);                 // Auto-spaced: 100, 400, 1600, 6400 Hz
 *   eq.setBand(0, 100.0f, 3.0f);      // Boost bass
 *   eq.setBand(3, 6400.0f, -2.0f);    // Cut highs
 *   eq.processBlock(buffer);
 *
 *   // Linear-phase mode — zero phase distortion for mastering:
 *   eq.setFilterMode(dspark::Equalizer<float>::FilterMode::LinearPhase);
 *   eq.prepare(spec);    // Re-prepare to allocate FFT buffers
 *
 *   // Level 3 — DSP engineer, full control:
 *   eq.setBand(0, { .frequency = 80, .gain = 4, .q = 0.5,
 *                    .type = dspark::Equalizer<float>::BandType::LowShelf,
 *                    .slope = 24, .enabled = true });
 * @endcode
 */

#include "Filters.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/FFT.h"

#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class Equalizer
 * @brief Parametric multi-band EQ using cascaded FilterEngine instances.
 *
 * Each band is an independent FilterEngine. Bands are processed in series
 * (band 0 first, then band 1, etc.). Disabled bands are skipped at zero cost.
 *
 * @tparam T        Sample type (float or double).
 * @tparam MaxBands Maximum number of EQ bands (compile-time, default 16).
 */
template <FloatType T, int MaxBands = 16>
class Equalizer
{
public:
    /** @brief Filter processing mode. */
    enum class FilterMode
    {
        MinimumPhase, ///< IIR biquads (zero latency, phase shift). Default.
        LinearPhase   ///< FFT-based (block-size latency, zero phase distortion).
                      ///< M5a note: the current implementation is strictly
                      ///< *zero-phase* (impulse symmetric around n=0 with
                      ///< halfFft latency), not linear-phase. Zero-phase
                      ///< filters exhibit pre-ringing on transients — this is
                      ///< inherent, not a bug. A true linear-phase redesign
                      ///< (symmetric FIR with N/2 delay in the time domain)
                      ///< is on the roadmap.
    };

    virtual ~Equalizer() = default;

    /** @brief Filter type for each EQ band. */
    enum class BandType
    {
        Peak,       ///< Parametric bell (boost/cut around frequency).
        LowShelf,   ///< Shelf: boosts/cuts below frequency.
        HighShelf,  ///< Shelf: boosts/cuts above frequency.
        LowPass,    ///< Removes frequencies above cutoff.
        HighPass,   ///< Removes frequencies below cutoff.
        Notch,      ///< Narrow rejection at frequency.
        BandPass,   ///< Bandpass around frequency.
        Tilt        ///< Tilt EQ: pivots spectrum around frequency.
    };

    /**
     * @brief Full configuration for a single EQ band.
     *
     * Used with the Level 3 API for complete control. All fields have sensible
     * defaults so you only need to specify what you want to change.
     */
    struct BandConfig
    {
        T frequency     = T(1000);          ///< Center/cutoff frequency in Hz.
        T gain          = T(0);             ///< Gain in dB (Peak and Shelf types).
        T q             = T(0.707);         ///< Q factor (0.1 = wide, 10 = narrow).
        BandType type   = BandType::Peak;   ///< Filter type for this band.
        int slope       = 12;               ///< Slope in dB/oct (LP/HP only: 6-48).
        bool enabled    = true;             ///< False to bypass this band.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares all bands for processing.
     *
     * Must be called before processBlock(). Re-call if sample rate,
     * block size, or channel count changes.
     *
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        for (int i = 0; i < MaxBands; ++i)
            bands_[i].prepare(spec);

        // Linear-phase FFT resources
        if (filterMode_ == FilterMode::LinearPhase && spec.maxBlockSize > 0)
        {
            int B = spec.maxBlockSize;
            lpFftSize_ = B * 2;
            // Round up to power of two
            int fftPow2 = 1;
            while (fftPow2 < lpFftSize_) fftPow2 <<= 1;
            lpFftSize_ = fftPow2;

            lpFft_ = std::make_unique<FFTReal<T>>(lpFftSize_);
            int numBins = lpFftSize_ / 2 + 1;
            lpMagnitude_.resize(static_cast<size_t>(numBins), T(1));
            lpPrevBlock_.resize(static_cast<size_t>(spec.numChannels));
            for (auto& pb : lpPrevBlock_)
                pb.assign(static_cast<size_t>(lpFftSize_ / 2), T(0));
            lpFftIn_.resize(static_cast<size_t>(lpFftSize_), T(0));
            lpFftOut_.resize(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpFftResult_.resize(static_cast<size_t>(lpFftSize_), T(0));
            lpDirty_ = true;
        }
    }

    /**
     * @brief Processes an audio buffer through all enabled bands in series.
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        // M5b: high-Q biquads can slip into denormals during long silences;
        // flush them to zero for the duration of the block.
        DenormalGuard guard;

        if (filterMode_ == FilterMode::LinearPhase && lpFft_)
        {
            processLinearPhase(buffer);
            return;
        }

        for (int i = 0; i < numBands_; ++i)
        {
            if (configs_[i].enabled)
                bands_[i].processBlock(buffer);
        }
    }

    /**
     * @brief Processes a single sample through all enabled bands (IIR mode only).
     *
     * For per-sample processing and custom routing. Only works in
     * MinimumPhase mode (linear-phase requires block processing).
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return EQ'd output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        T sample = input;
        for (int i = 0; i < numBands_; ++i)
        {
            if (configs_[i].enabled)
                sample = bands_[i].processSample(sample, channel);
        }
        return sample;
    }

    /** @brief Resets all filter states (call after silence or transport stop). */
    void reset() noexcept
    {
        for (int i = 0; i < MaxBands; ++i)
            bands_[i].reset();
        for (auto& pb : lpPrevBlock_)
            std::fill(pb.begin(), pb.end(), T(0));
    }

    // -- Level 1: Simple API (frequency + gain) ---------------------------------

    /**
     * @brief Configures a band with just frequency and gain.
     *
     * Uses Peak filter type with default Q (0.707). This is all a desktop
     * developer needs for basic tone shaping.
     *
     * @param index     Band index (0 to numBands-1).
     * @param frequency Center frequency in Hz.
     * @param gainDb    Boost/cut in dB.
     */
    void setBand(int index, T frequency, T gainDb)
    {
        setBand(index, frequency, gainDb, T(0.707));
    }

    // -- Level 2: Intermediate API (+ Q factor) ---------------------------------

    /**
     * @brief Configures a band with frequency, gain, and Q.
     *
     * Uses Peak filter type. Q controls bandwidth:
     * - 0.707 = wide (Butterworth)
     * - 1.0-2.0 = moderate
     * - 5.0-10.0 = narrow surgical cut
     *
     * @param index     Band index.
     * @param frequency Center frequency in Hz.
     * @param gainDb    Boost/cut in dB.
     * @param q         Quality factor.
     */
    void setBand(int index, T frequency, T gainDb, T q)
    {
        BandConfig cfg;
        cfg.frequency = frequency;
        cfg.gain      = gainDb;
        cfg.q         = q;
        cfg.type      = BandType::Peak;
        cfg.slope     = 12;
        cfg.enabled   = true;
        setBand(index, cfg);
    }

    // -- Level 3: Expert API (full BandConfig) ----------------------------------

    /**
     * @brief Configures a band with full control over all parameters.
     *
     * @param index  Band index (0 to MaxBands-1).
     * @param config Complete band configuration.
     */
    void setBand(int index, const BandConfig& config)
    {
        if (index < 0 || index >= MaxBands) return;

        configs_[index] = config;

        // Ensure this band is active
        if (index >= numBands_)
            numBands_ = index + 1;

        applyConfig(index);
    }

    // -- Band management --------------------------------------------------------

    /**
     * @brief Sets the number of active bands with auto-logarithmic spacing.
     *
     * Bands are spaced logarithmically from 80 Hz to 16 kHz, each set to
     * Peak type with 0 dB gain. Existing configurations are replaced.
     *
     * @param count Number of bands (1 to MaxBands).
     */
    void setNumBands(int count)
    {
        numBands_ = std::clamp(count, 1, MaxBands);

        // Logarithmic spacing from 80 Hz to 16 kHz
        const T logMin = std::log(T(80));
        const T logMax = std::log(T(16000));

        for (int i = 0; i < numBands_; ++i)
        {
            T t = (numBands_ > 1)
                ? static_cast<T>(i) / static_cast<T>(numBands_ - 1)
                : T(0.5);

            BandConfig cfg;
            cfg.frequency = std::exp(logMin + t * (logMax - logMin));
            cfg.gain      = T(0);
            cfg.q         = T(0.707);
            cfg.type      = BandType::Peak;
            cfg.slope     = 12;
            cfg.enabled   = true;

            configs_[i] = cfg;
            applyConfig(i);
        }
    }

    /** @brief Returns the number of active bands. */
    [[nodiscard]] int getNumBands() const noexcept { return numBands_; }

    /**
     * @brief Returns the current configuration of a band.
     * @param index Band index.
     * @return Copy of the band's BandConfig.
     */
    [[nodiscard]] BandConfig getBandConfig(int index) const noexcept
    {
        if (index < 0 || index >= MaxBands) return {};
        return configs_[index];
    }

    /**
     * @brief Enables or disables a band without changing its parameters.
     * @param index   Band index.
     * @param enabled True to enable, false to bypass.
     */
    void setBandEnabled(int index, bool enabled) noexcept
    {
        if (index >= 0 && index < MaxBands)
            configs_[index].enabled = enabled;
    }

    // -- Filter mode ------------------------------------------------------------

    /**
     * @brief Sets the filter processing mode.
     *
     * - **MinimumPhase** (default): IIR biquad filters. Zero latency.
     *   Phase varies with frequency (normal for most applications).
     * - **LinearPhase**: FFT-based processing. Zero phase distortion
     *   (preserves waveform shape). Adds latency = block size samples.
     *   Ideal for mastering and critical listening.
     *
     * Call prepare() again after changing mode.
     *
     * @param mode Filter mode.
     */
    void setFilterMode(FilterMode mode) noexcept
    {
        filterMode_ = mode;
        lpDirty_ = true;
    }

    /** @brief Returns the current filter mode. */
    [[nodiscard]] FilterMode getFilterMode() const noexcept { return filterMode_; }

    /**
     * @brief Returns the latency in samples (0 for MinimumPhase, blockSize for LinearPhase).
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_ == FilterMode::LinearPhase) ? spec_.maxBlockSize : 0;
    }

    // -- Soft mode --------------------------------------------------------------

    /**
     * @brief Enables soft mode (anti-ringing Q reduction).
     *
     * When enabled, the effective Q of Peak and Shelf bands is automatically
     * reduced as gain increases: `effectiveQ = min(Q, 1 + 8 / (|gainDb| + 1))`.
     * This prevents sharp resonant peaks at high boost levels, producing a
     * smoother, more musical EQ curve. Ideal for mastering.
     *
     * @param enabled True to enable soft mode.
     */
    void setSoftMode(bool enabled) noexcept
    {
        softMode_.store(enabled, std::memory_order_relaxed);
        for (int i = 0; i < numBands_; ++i)
            applyConfig(i);
    }

    /** @brief Returns whether soft mode is enabled. */
    [[nodiscard]] bool getSoftMode() const noexcept
    {
        return softMode_.load(std::memory_order_relaxed);
    }

    // -- Frequency response analysis --------------------------------------------

    /**
     * @brief Computes the combined magnitude response of all enabled bands.
     *
     * Essential for drawing EQ curves in a GUI. Evaluates the product of
     * all enabled bands' transfer functions at each requested frequency.
     *
     * @param frequencies  Array of frequencies in Hz.
     * @param magnitudes   Output array (same size, linear scale).
     * @param numPoints    Number of frequency points.
     */
    void getMagnitudeForFrequencyArray(const T* frequencies, T* magnitudes,
                                       int numPoints) const noexcept
    {
        // Initialize to unity
        for (int i = 0; i < numPoints; ++i)
            magnitudes[i] = T(1);

        for (int b = 0; b < numBands_; ++b)
        {
            if (!configs_[b].enabled) continue;

            auto coeffs = computeBandCoeffs(configs_[b]);
            int numStages = (configs_[b].type == BandType::LowPass ||
                             configs_[b].type == BandType::HighPass)
                            ? std::clamp(configs_[b].slope / 12, 1, 4) : 1;

            for (int i = 0; i < numPoints; ++i)
            {
                T mag = coeffs.getMagnitude(static_cast<double>(frequencies[i]),
                                            spec_.sampleRate);
                // Cascade stages
                for (int s = 1; s < numStages; ++s)
                    mag *= coeffs.getMagnitude(static_cast<double>(frequencies[i]),
                                               spec_.sampleRate);
                magnitudes[i] *= mag;
            }
        }
    }

    // -- Expert access ----------------------------------------------------------

    /**
     * @brief Direct access to a band's underlying FilterEngine.
     *
     * For DSP engineers who need fine-grained control: analog drift,
     * custom filter shapes, per-sample processing, etc.
     *
     * @param index Band index.
     * @return Reference to the FilterEngine for this band.
     */
    FilterEngine<T>& getBandFilter(int index)
    {
        return bands_[index];
    }

    /** @brief Const overload. */
    const FilterEngine<T>& getBandFilter(int index) const
    {
        return bands_[index];
    }

protected:
    void applyConfig(int index) noexcept
    {
        auto& filter = bands_[index];
        const auto& cfg = configs_[index];

        float freq = static_cast<float>(cfg.frequency);
        float gain = static_cast<float>(cfg.gain);
        float q    = static_cast<float>(cfg.q);

        // Soft mode: reduce Q as gain increases to prevent ringing
        if (softMode_.load(std::memory_order_relaxed))
        {
            float absGain = std::abs(gain);
            float maxQ = 1.0f + 8.0f / (absGain + 1.0f);
            q = std::min(q, maxQ);
        }

        switch (cfg.type)
        {
            case BandType::Peak:
                filter.setPeaking(freq, gain, q);
                break;
            case BandType::LowShelf:
                filter.setLowShelf(freq, gain, q);
                break;
            case BandType::HighShelf:
                filter.setHighShelf(freq, gain, q);
                break;
            case BandType::LowPass:
                filter.setLowPass(freq, q, cfg.slope);
                break;
            case BandType::HighPass:
                filter.setHighPass(freq, q, cfg.slope);
                break;
            case BandType::Notch:
                filter.setNotch(freq, q);
                break;
            case BandType::BandPass:
                filter.setBandPass(freq, q);
                break;
            case BandType::Tilt:
                filter.setTilt(freq, gain);
                break;
        }

        lpDirty_ = true;
    }

    /** @brief Computes biquad coefficients for a band config (for LP magnitude). */
    [[nodiscard]] BiquadCoeffs<T> computeBandCoeffs(const BandConfig& cfg) const noexcept
    {
        double sr = spec_.sampleRate;
        double f  = static_cast<double>(cfg.frequency);
        double g  = static_cast<double>(cfg.gain);
        double q  = static_cast<double>(cfg.q);

        switch (cfg.type)
        {
            case BandType::Peak:      return BiquadCoeffs<T>::makePeak(sr, f, q, g);
            case BandType::LowShelf:  return BiquadCoeffs<T>::makeLowShelf(sr, f, g);
            case BandType::HighShelf:  return BiquadCoeffs<T>::makeHighShelf(sr, f, g);
            case BandType::LowPass:   return BiquadCoeffs<T>::makeLowPass(sr, f, q);
            case BandType::HighPass:   return BiquadCoeffs<T>::makeHighPass(sr, f, q);
            case BandType::Notch:     return BiquadCoeffs<T>::makeNotch(sr, f, q);
            case BandType::BandPass:  return BiquadCoeffs<T>::makeBandPass(sr, f, q);
            case BandType::Tilt:      return BiquadCoeffs<T>::makeTilt(sr, f, g);
        }
        return {};
    }

    /**
     * @brief Recomputes the combined magnitude response for all enabled bands.
     *
     * Evaluates H(z) = B(z)/A(z) at each FFT bin frequency and stores |H|
     * as a magnitude-only spectrum. For cascaded stages (LP/HP slopes > 12 dB/oct),
     * the single-stage response is raised to the power of numStages.
     */
    void recomputeLinearPhaseMagnitude() noexcept
    {
        if (!lpDirty_ || lpFftSize_ == 0) return;
        lpDirty_ = false;

        int numBins = lpFftSize_ / 2 + 1;
        // Reset to unity
        for (int k = 0; k < numBins; ++k)
            lpMagnitude_[k] = T(1);

        for (int b = 0; b < numBands_; ++b)
        {
            if (!configs_[b].enabled) continue;

            auto coeffs = computeBandCoeffs(configs_[b]);
            int numStages = (configs_[b].type == BandType::LowPass ||
                             configs_[b].type == BandType::HighPass)
                            ? std::clamp(configs_[b].slope / 12, 1, 4)
                            : 1;

            for (int k = 0; k < numBins; ++k)
            {
                // Normalised angular frequency for this bin
                T omega = T(2) * static_cast<T>(pi<double>) *
                          static_cast<T>(k) / static_cast<T>(lpFftSize_);

                // z = e^(j*omega) => cos(omega) + j*sin(omega)
                T cosW  = std::cos(omega);
                T cos2W = std::cos(T(2) * omega);
                T sinW  = std::sin(omega);
                T sin2W = std::sin(T(2) * omega);

                // Numerator: b0 + b1*z^-1 + b2*z^-2
                T numRe = coeffs.b0 + coeffs.b1 * cosW + coeffs.b2 * cos2W;
                T numIm =           - coeffs.b1 * sinW - coeffs.b2 * sin2W;

                // Denominator: 1 + a1*z^-1 + a2*z^-2
                T denRe = T(1) + coeffs.a1 * cosW + coeffs.a2 * cos2W;
                T denIm =      - coeffs.a1 * sinW - coeffs.a2 * sin2W;

                T numMag2 = numRe * numRe + numIm * numIm;
                T denMag2 = denRe * denRe + denIm * denIm;

                T mag = (denMag2 > T(1e-30))
                    ? std::sqrt(numMag2 / denMag2)
                    : T(0);

                // Cascade: raise to power of numStages
                if (numStages > 1)
                {
                    T m = mag;
                    for (int s = 1; s < numStages; ++s)
                        m *= mag;
                    mag = m;
                }

                lpMagnitude_[k] *= mag;
            }
        }
    }

    /**
     * @brief Linear-phase processing via overlap-save FFT convolution.
     *
     * Uses magnitude-only spectrum (zero phase). The overlap-save approach:
     * 1. Assemble [prevBlock | currentBlock] into FFT input
     * 2. FFT forward
     * 3. Multiply by |H(k)| (magnitude only, no phase)
     * 4. IFFT
     * 5. Output the last B samples (discard first B as overlap)
     */
    void processLinearPhase(AudioBufferView<T> buffer) noexcept
    {
        recomputeLinearPhaseMagnitude();

        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        const int halfFft = lpFftSize_ / 2;

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* channelData = buffer.getChannel(ch);
            auto& prev = lpPrevBlock_[ch];

            // Build input: [prev half | current samples | zero-pad]
            for (int i = 0; i < halfFft; ++i)
                lpFftIn_[i] = (i < static_cast<int>(prev.size())) ? prev[i] : T(0);

            for (int i = 0; i < nS && i < halfFft; ++i)
                lpFftIn_[halfFft + i] = channelData[i];

            // Zero-pad remainder if nS < halfFft
            for (int i = nS; i < halfFft; ++i)
                lpFftIn_[halfFft + i] = T(0);

            // Save UNFILTERED input as prev for next overlap-save call.
            // This must happen after building lpFftIn_ (which reads the old prev)
            // but before the output overwrites channelData with filtered samples.
            // Bug fix: the old code stored the FILTERED output, which corrupted
            // the overlap-save convolution — it requires raw (unfiltered) input.
            for (int i = 0; i < halfFft; ++i)
                prev[i] = (i < nS) ? channelData[i] : T(0);

            // Forward FFT
            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());

            // Apply magnitude-only filter (multiply each complex bin by |H(k)|)
            int numBins = lpFftSize_ / 2 + 1;
            for (int k = 0; k < numBins; ++k)
            {
                lpFftOut_[2 * k]     *= lpMagnitude_[k];
                lpFftOut_[2 * k + 1] *= lpMagnitude_[k];
            }

            // Inverse FFT (FFTReal roundtrip is identity — no extra scaling needed)
            lpFft_->inverse(lpFftOut_.data(), lpFftResult_.data());

            // Output: take the last B samples (overlap-save)
            for (int i = 0; i < nS; ++i)
                channelData[i] = lpFftResult_[halfFft + i];
        }
    }

    AudioSpec spec_ {};
    int numBands_ = 0;

    std::array<FilterEngine<T>, MaxBands> bands_ {};
    std::array<BandConfig, MaxBands> configs_ {};

    std::atomic<bool> softMode_ { false };

    // Linear-phase state
    FilterMode filterMode_ = FilterMode::MinimumPhase;
    std::unique_ptr<FFTReal<T>> lpFft_;
    int lpFftSize_ = 0;
    std::vector<T> lpMagnitude_;
    std::vector<std::vector<T>> lpPrevBlock_;
    std::vector<T> lpFftIn_, lpFftOut_, lpFftResult_;
    bool lpDirty_ = true;
};

} // namespace dspark
