// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Equalizer.h
 * @brief Multi-band parametric equalizer with Minimum and Linear Phase modes.
 *
 * Combines multiple FilterEngine instances into a single processor. Supports up to 
 * MaxBands simultaneous filter bands. Completely thread-safe, lock-free parameter 
 * updates, and zero memory allocations in the audio thread.
 *
 * Dependencies: Filters.h (FilterEngine), FFT.h, Biquad.h, AudioSpec.h, AudioBuffer.h.
 */

#include "Filters.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"

#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

namespace dspark {

/**
 * @class Equalizer
 * @brief Parametric multi-band EQ using cascaded biquads or FFT overlap-save convolution.
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
        MinimumPhase, ///< IIR biquads (zero latency, minimum phase shift). Default.
        LinearPhase   ///< FFT-based overlap-save (block-size latency, zero phase distortion).
    };

    /** @brief Non-virtual destructor to prevent vtable instantiation (zero virtual dispatch). */
    ~Equalizer() = default;

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
     */
    struct BandConfig
    {
        T frequency     = T(1000);          ///< Center/cutoff frequency in Hz.
        T gain          = T(0);             ///< Gain in dB (Peak, Shelf, Tilt).
        T q             = T(0.707);         ///< Q factor (0.1 = wide, 10 = narrow).
        BandType type   = BandType::Peak;   ///< Filter type for this band.
        int slope       = 12;               ///< Slope in dB/oct (LP/HP only: 6-48).
        bool enabled    = true;             ///< False to bypass this band.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares all bands and allocates necessary resources for processing.
     *
     * Must be called from the host's prepareToPlay/setup method. 
     * Performs all memory allocations.
     *
     * @param spec Audio environment (sample rate, max block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        for (int i = 0; i < MaxBands; ++i)
            bands_[i].prepare(spec);

        if (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && spec.maxBlockSize > 0)
        {
            // Overlap-save requires FFT size N >= L + M - 1
            // Where L is maxBlockSize, M is impulse response length.
            // We use M = 2 * L, so N must be >= 3 * L. We round up to next power of 2.
            int targetFftSize = spec.maxBlockSize * 4;
            int fftPow2 = 1;
            while (fftPow2 < targetFftSize) fftPow2 <<= 1;
            lpFftSize_ = fftPow2;

            lpFft_ = std::make_unique<FFTReal<T>>(lpFftSize_);
            
            // Complex kernel representation (Real, Imaginary interleaved)
            lpKernel_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            
            // Overlap-save needs (M-1) samples of history where M = 2*maxBlockSize
            // is the FIR kernel length. Size the history buffer accordingly.
            lpPrevBlock_.resize(static_cast<size_t>(spec.numChannels));
            for (auto& pb : lpPrevBlock_)
                pb.assign(static_cast<size_t>(spec.maxBlockSize * 2), T(0));
                
            lpFftIn_.assign(static_cast<size_t>(lpFftSize_), T(0));
            lpFftOut_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));

            // Pre-allocate recompute scratch so a live band change never allocates on
            // the audio thread (recomputeLinearPhaseKernel runs there via processBlock).
            lpMagScratch_.assign(static_cast<size_t>(lpFftSize_ / 2 + 1), T(1));
            lpTempFreq_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpImpulse_.assign(static_cast<size_t>(lpFftSize_), T(0));
            lpKernelSpace_.assign(static_cast<size_t>(lpFftSize_), T(0));

            lpDirty_.store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Processes an audio buffer in-place.
     * 
     * Guarantees zero allocations and uses SIMD-friendly contiguous arrays.
     * Checks atomic flags lock-free to update DSP state if the host changed parameters.
     *
     * @param buffer Audio data to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;

        // Check if config was updated from the UI thread (Lock-free acquire)
        if (configDirty_.exchange(false, std::memory_order_acquire))
        {
            updateActiveFilters();
            lpDirty_.store(true, std::memory_order_release);
        }

        FilterMode currentMode = filterMode_.load(std::memory_order_acquire);

        if (currentMode == FilterMode::LinearPhase && lpFft_)
        {
            // If kernel needs recalculation, do it once.
            // In a production environment, this should ideally be deferred to a background thread.
            if (lpDirty_.exchange(false, std::memory_order_acquire))
                recomputeLinearPhaseKernel();
                
            processLinearPhase(buffer);
            return;
        }

        // Minimum Phase (IIR) Processing
        const int activeBands = numBands_.load(std::memory_order_relaxed);
        for (int i = 0; i < activeBands; ++i)
        {
            if (bandEnabled_[i].load(std::memory_order_relaxed))
                bands_[i].processBlock(buffer);
        }
    }

    /**
     * @brief Processes a single sample through all enabled bands (IIR mode only).
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return EQ'd output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        T sample = input;
        const int activeBands = numBands_.load(std::memory_order_relaxed);
        
        for (int i = 0; i < activeBands; ++i)
        {
            if (bandEnabled_[i].load(std::memory_order_relaxed))
                sample = bands_[i].processSample(sample, channel);
        }
        return sample;
    }

    /** 
     * @brief Resets all filter states to zero to prevent ringing on playback start. 
     */
    void reset() noexcept
    {
        for (int i = 0; i < MaxBands; ++i)
            bands_[i].reset();
            
        for (auto& pb : lpPrevBlock_)
            std::fill(pb.begin(), pb.end(), T(0));
    }

    // -- API --------------------------------------------------------------------

    /**
     * @brief Configures a band with frequency and gain (Peak filter).
     * @param index     Band index (0 to MaxBands-1).
     * @param frequency Center frequency in Hz.
     * @param gainDb    Boost/cut in dB.
     */
    void setBand(int index, T frequency, T gainDb)
    {
        setBand(index, frequency, gainDb, T(0.707));
    }

    /**
     * @brief Configures a band with frequency, gain, and Q.
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

    /**
     * @brief Configures a band with full control over all parameters. Thread-safe.
     * @param index  Band index.
     * @param config Complete band configuration.
     */
    void setBand(int index, const BandConfig& config)
    {
        if (index < 0 || index >= MaxBands) return;

        configs_[index] = config;
        bandEnabled_[index].store(config.enabled, std::memory_order_release);

        int currentBands = numBands_.load(std::memory_order_relaxed);
        if (index >= currentBands)
            numBands_.store(index + 1, std::memory_order_relaxed);

        // Signal audio thread to update filters
        configDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the number of active bands with auto-logarithmic spacing.
     * @param count Number of bands (1 to MaxBands).
     */
    void setNumBands(int count)
    {
        int validCount = std::clamp(count, 1, MaxBands);
        numBands_.store(validCount, std::memory_order_relaxed);

        const T logMin = std::log(T(80));
        const T logMax = std::log(T(16000));

        for (int i = 0; i < validCount; ++i)
        {
            T t = (validCount > 1) ? static_cast<T>(i) / static_cast<T>(validCount - 1) : T(0.5);

            BandConfig cfg;
            cfg.frequency = std::exp(logMin + t * (logMax - logMin));
            cfg.gain      = T(0);
            cfg.q         = T(0.707);
            cfg.type      = BandType::Peak;
            cfg.slope     = 12;
            cfg.enabled   = true;

            configs_[i] = cfg;
            bandEnabled_[i].store(true, std::memory_order_release);
        }
        configDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns the number of active bands. */
    [[nodiscard]] int getNumBands() const noexcept
    {
        return numBands_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Switches Peak bands to the Orfanidis matched (de-cramped) design.
     *
     * Bilinear bells cramp near Nyquist (narrower, response pinned at fs/2);
     * the matched design prescribes the analog prototype's Nyquist gain so
     * high bells keep their analog shape — the state-of-the-art digital EQ
     * behaviour. Off by default for bit-compatibility with previous output.
     */
    void setMatchedBells(bool enabled) noexcept
    {
        matchedBells_.store(enabled, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Returns the current configuration of a band.
     * @param index Band index.
     * @return Copy of the band's BandConfig.
     */
    [[nodiscard]] BandConfig getBandConfig(int index) const noexcept
    {
        if (index < 0 || index >= MaxBands) return {};
        return configs_[index]; // Note: Struct copy is safe if mostly read from GUI
    }

    /**
     * @brief Enables or disables a band without changing its parameters.
     * @param index   Band index.
     * @param enabled True to enable, false to bypass.
     */
    void setBandEnabled(int index, bool enabled) noexcept
    {
        if (index >= 0 && index < MaxBands)
        {
            configs_[index].enabled = enabled;
            bandEnabled_[index].store(enabled, std::memory_order_release);
            configDirty_.store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Sets the filter processing mode (Minimum Phase or Linear Phase).
     * @param mode Filter mode.
     */
    void setFilterMode(FilterMode mode) noexcept
    {
        filterMode_.store(mode, std::memory_order_release);
        if (mode == FilterMode::LinearPhase)
            lpDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns the current filter mode. */
    [[nodiscard]] FilterMode getFilterMode() const noexcept 
    { 
        return filterMode_.load(std::memory_order_relaxed); 
    }

    /**
     * @brief Returns the latency in samples.
     * @return 0 for MinimumPhase, maxBlockSize for LinearPhase.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase) 
                ? spec_.maxBlockSize : 0;
    }

    /**
     * @brief Enables soft mode (anti-ringing Q reduction dynamically based on gain).
     * @param enabled True to enable soft mode.
     */
    void setSoftMode(bool enabled) noexcept
    {
        softMode_.store(enabled, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns whether soft mode is enabled. */
    [[nodiscard]] bool getSoftMode() const noexcept
    {
        return softMode_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Computes the combined magnitude response of all enabled bands.
     *
     * @param frequencies   Array of frequencies in Hz.
     * @param magnitudes    Output array (same size, linear scale).
     * @param numPoints     Number of frequency points.
     */
    void getMagnitudeForFrequencyArray(const T* frequencies, T* magnitudes, int numPoints) const noexcept
    {
        for (int i = 0; i < numPoints; ++i)
            magnitudes[i] = T(1);

        const int activeBands = numBands_.load(std::memory_order_relaxed);
        for (int b = 0; b < activeBands; ++b)
        {
            if (!configs_[b].enabled) continue;

            // Evaluate the TRUE per-stage cascade (built once per band) — the old
            // code used a single-Q biquad with `mag *= mag` (which squares the
            // exponent each step instead of incrementing it).
            BiquadCoeffs<T> st[5];
            const int ns = buildBandStages(configs_[b], st);

            for (int i = 0; i < numPoints; ++i)
            {
                T mag = T(1);
                for (int s = 0; s < ns; ++s)
                    mag *= st[s].getMagnitude(static_cast<double>(frequencies[i]), spec_.sampleRate);
                magnitudes[i] *= mag;
            }
        }
    }

    /**
     * @brief Direct access to a band's underlying FilterEngine.
     * @param index Band index.
     * @return Reference to the FilterEngine for this band.
     */
    FilterEngine<T>& getBandFilter(int index) { return bands_[index]; }

    /** @brief Const overload. */
    const FilterEngine<T>& getBandFilter(int index) const { return bands_[index]; }

protected:

    /**
     * @brief Translates BandConfigs into internal FilterEngine parameters safely.
     * Called by processBlock when configDirty_ is true.
     */
    void updateActiveFilters() noexcept
    {
        bool soft = softMode_.load(std::memory_order_relaxed);
        int activeBands = numBands_.load(std::memory_order_relaxed);

        for (int i = 0; i < activeBands; ++i)
        {
            auto& filter = bands_[i];
            const auto& cfg = configs_[i]; // Thread-safe copy from atomic boundaries

            float freq = static_cast<float>(cfg.frequency);
            float gain = static_cast<float>(cfg.gain);
            float q    = static_cast<float>(cfg.q);

            if (soft)
            {
                float absGain = std::abs(gain);
                float maxQ = 1.0f + 8.0f / (absGain + 1.0f);
                q = std::min(q, maxQ);
            }

            switch (cfg.type)
            {
                case BandType::Peak:      filter.setPeaking(freq, gain, q); break;
                // Shelves take a SLOPE (0..1], not a Q: convert with the RBJ
                // S<->Q relation so the user's Q behaves consistently here and
                // in the linear-phase kernel (which uses the same conversion).
                case BandType::LowShelf:  filter.setLowShelf(freq, gain,
                                              static_cast<float>(shelfSlopeFromQ(q, gain))); break;
                case BandType::HighShelf: filter.setHighShelf(freq, gain,
                                              static_cast<float>(shelfSlopeFromQ(q, gain))); break;
                case BandType::LowPass:   filter.setLowPass(freq, q, cfg.slope); break;
                case BandType::HighPass:  filter.setHighPass(freq, q, cfg.slope); break;
                case BandType::Notch:     filter.setNotch(freq, q); break;
                case BandType::BandPass:  filter.setBandPass(freq, q); break;
                case BandType::Tilt:      filter.setTilt(freq, gain); break;
            }
        }
    }

    /**
     * @brief Computes raw biquad coefficients for magnitude analysis.
     * @param cfg Band configuration.
     * @return BiquadCoeffs structure.
     */
    /**
     * @brief Converts a user-facing shelf Q into the RBJ shelf slope S.
     *
     * RBJ relation: 1/Q^2 = (A + 1/A) * (1/S - 1) + 2 with A = 10^(dB/40).
     * Q = 0.7071 maps to S = 1 (the standard shelf). Out-of-domain values are
     * clamped to the stable (0, 1] range that the coefficient factory accepts.
     */
    [[nodiscard]] static double shelfSlopeFromQ(double q, double gainDb) noexcept
    {
        q = std::max(q, 0.05);
        const double A = std::pow(10.0, std::abs(gainDb) / 40.0);
        const double denom = A + 1.0 / A;
        const double invS = (1.0 / (q * q) - 2.0) / denom + 1.0;
        if (invS <= 1.0) return 1.0;          // steeper-than-standard requests clamp to S = 1
        return std::clamp(1.0 / invS, 0.0001, 1.0);
    }

    [[nodiscard]] BiquadCoeffs<T> computeBandCoeffs(const BandConfig& cfg) const noexcept
    {
        double sr = spec_.sampleRate;
        double f  = static_cast<double>(cfg.frequency);
        double g  = static_cast<double>(cfg.gain);
        double q  = static_cast<double>(cfg.q);

        switch (cfg.type)
        {
            case BandType::Peak:
                return matchedBells_.load(std::memory_order_relaxed)
                    ? BiquadCoeffs<T>::makePeakMatched(sr, f, q, g)
                    : BiquadCoeffs<T>::makePeak(sr, f, q, g);
            case BandType::LowShelf:  return BiquadCoeffs<T>::makeLowShelf(sr, f, g, shelfSlopeFromQ(q, g));
            case BandType::HighShelf: return BiquadCoeffs<T>::makeHighShelf(sr, f, g, shelfSlopeFromQ(q, g));
            case BandType::LowPass:   return BiquadCoeffs<T>::makeLowPass(sr, f, q);
            case BandType::HighPass:  return BiquadCoeffs<T>::makeHighPass(sr, f, q);
            case BandType::Notch:     return BiquadCoeffs<T>::makeNotch(sr, f, q);
            case BandType::BandPass:  return BiquadCoeffs<T>::makeBandPass(sr, f, q);
            case BandType::Tilt:      return BiquadCoeffs<T>::makeTilt(sr, f, g);
        }
        return {};
    }

    /**
     * @brief Fills `stages` with the ACTUAL biquad cascade for a band (per-stage
     * Butterworth Q for multi-stage LP/HP, single biquad otherwise) and returns
     * the stage count. Used for an accurate magnitude response that matches what
     * the IIR FilterEngine really applies (instead of a single-Q^N approximation).
     * @param stages Output buffer (capacity >= 5).
     */
    [[nodiscard]] int buildBandStages(const BandConfig& cfg, BiquadCoeffs<T>* stages) const noexcept
    {
        const double sr = spec_.sampleRate;
        const double f  = static_cast<double>(cfg.frequency);

        if (cfg.type == BandType::LowPass || cfg.type == BandType::HighPass)
        {
            const bool lp = (cfg.type == BandType::LowPass);
            auto casc = FilterEngine<T>::cascadeForSlope(cfg.slope);
            int n = 0;
            if (casc.hasFirstOrder)
                stages[n++] = lp ? BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f)
                                 : BiquadCoeffs<T>::makeFirstOrderHighPass(sr, f);
            for (int s = 0; s < casc.numSecondOrder; ++s)
            {
                const double q = static_cast<double>(casc.qValues[s]);
                stages[n++] = lp ? BiquadCoeffs<T>::makeLowPass(sr, f, q)
                                 : BiquadCoeffs<T>::makeHighPass(sr, f, q);
            }
            return n;
        }

        stages[0] = computeBandCoeffs(cfg);
        return 1;
    }

    /**
     * @brief Mathematically robust Linear Phase kernel computation.
     *
     * Constructs a true zero-phase impulse response by evaluating the H(k)
     * magnitude, performing IFFT, shifting by M/2 to make it causal, 
     * windowing it with Blackman-Harris to reduce truncation artifacts, 
     * zero-padding to N, and converting back to FFT for overlap-save.
     */
    void recomputeLinearPhaseKernel() noexcept
    {
        if (lpFftSize_ == 0) return;

        const int numBins = lpFftSize_ / 2 + 1;
        T* const mag = lpMagScratch_.data();   // pre-allocated scratch (no audio-thread alloc)
        std::fill_n(mag, numBins, T(1));
        const int activeBands = numBands_.load(std::memory_order_relaxed);

        // 1. Accumulate the TRUE per-stage cascade magnitude of all active bands
        //    (matches what the IIR FilterEngine applies in MinimumPhase mode, so
        //    switching modes keeps the same magnitude response).
        const double sr = spec_.sampleRate;
        for (int b = 0; b < activeBands; ++b)
        {
            if (!configs_[b].enabled) continue;

            BiquadCoeffs<T> st[5];
            const int ns = buildBandStages(configs_[b], st);

            for (int k = 0; k < numBins; ++k)
            {
                const double freq = sr * static_cast<double>(k) / static_cast<double>(lpFftSize_);
                T m = T(1);
                for (int s = 0; s < ns; ++s)
                    m *= st[s].getMagnitude(freq, sr);
                mag[k] *= m;
            }
        }

        // 2. Prepare Zero-Phase Frequency buffer (Real = Mag, Imag = 0)
        std::fill(lpTempFreq_.begin(), lpTempFreq_.end(), T(0));
        for (int k = 0; k < numBins; ++k)
            lpTempFreq_[2 * k] = mag[k];

        // 3. IFFT to get temporal impulse response (wrapped around t=0)
        lpFft_->inverse(lpTempFreq_.data(), lpImpulse_.data());

        // 4. Shift, Windowing and Zero-pad for Overlap-Save
        const int M = spec_.maxBlockSize * 2; // Desired Kernel length
        const int halfM = M / 2;
        std::fill(lpKernelSpace_.begin(), lpKernelSpace_.end(), T(0)); // zero-pad scratch

        for (int i = 0; i < M; ++i)
        {
            // Circular read from center 0
            int readIdx = (i - halfM + lpFftSize_) % lpFftSize_;

            // Blackman-Harris window formulation
            double t = static_cast<double>(i) / static_cast<double>(M - 1);
            double window = 0.35875
                          - 0.48829 * std::cos(2.0 * pi<double> * t)
                          + 0.14128 * std::cos(4.0 * pi<double> * t)
                          - 0.01168 * std::cos(6.0 * pi<double> * t);

            // No extra scaling: FFTReal::inverse already applies the full 1/N
            // normalisation (verified by exact round-trip). The previous extra
            // division by lpFftSize_ attenuated the whole linear-phase path by
            // 20*log10(N) dB — about -60 dB with the default sizes.
            lpKernelSpace_[i] = lpImpulse_[readIdx] * static_cast<T>(window);
        }

        // 5. Transform finalized zero-padded causal kernel to frequency domain
        lpFft_->forward(lpKernelSpace_.data(), lpKernel_.data());
    }

    /**
     * @brief Linear-phase processing via mathematically correct overlap-save FFT convolution.
     * @param buffer Audio buffer.
     */
    void processLinearPhase(AudioBufferView<T> buffer) noexcept
    {
        // Safety bound check: Prevent out-of-bounds if host pushes dynamic channel counts
        const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(lpPrevBlock_.size()));
        const int L   = buffer.getNumSamples();
        const int N   = lpFftSize_;
        const int M   = spec_.maxBlockSize * 2; // Kernel length

        // Safety check: block size cannot exceed pre-allocated maxBlockSize
        if (L > spec_.maxBlockSize) return; 

        const int overlapSize = M - 1;  // FIR history overlap-save needs (kernel len - 1)

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* channelData = buffer.getChannel(ch);
            auto& prev = lpPrevBlock_[ch];

            // 1. Build overlap-save input: [history (overlapSize) | current (L) | zeros].
            //    The full (M-1)-sample history is required for a length-M FIR;
            //    the previous code only kept L samples, corrupting the output
            //    (especially for blocks smaller than maxBlockSize).
            for (int i = 0; i < overlapSize; ++i)
                lpFftIn_[i] = prev[i];
            for (int i = 0; i < L; ++i)
                lpFftIn_[overlapSize + i] = channelData[i];
            for (int i = overlapSize + L; i < N; ++i)
                lpFftIn_[i] = T(0);

            // 2. Save the LAST overlapSize samples of [history | current] as the next
            //    block's history (read from lpFftIn_ before the inverse FFT reuses it).
            for (int i = 0; i < overlapSize; ++i)
                prev[i] = lpFftIn_[L + i];

            // 3. Forward FFT
            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());

            // 3. Complex multiplication: H(k) * X(k)
            int numBins = N / 2 + 1;
            for (int k = 0; k < numBins; ++k)
            {
                T realX = lpFftOut_[2 * k];
                T imagX = lpFftOut_[2 * k + 1];
                T realH = lpKernel_[2 * k];
                T imagH = lpKernel_[2 * k + 1];

                lpFftOut_[2 * k]     = realX * realH - imagX * imagH;
                lpFftOut_[2 * k + 1] = realX * imagH + imagX * realH;
            }

            // 4. Inverse FFT
            lpFft_->inverse(lpFftOut_.data(), lpFftIn_.data()); // Reusing lpFftIn_ to save memory

            // 5. Overlap-save output extraction: valid data starts at index M - 1
            int offset = M - 1;
            for (int i = 0; i < L; ++i)
            {
                // Assign filtered output back to the channel
                channelData[i] = lpFftIn_[offset + i]; 
            }
        }
    }

    AudioSpec spec_ {};
    std::atomic<int> numBands_ { 0 };

    std::array<FilterEngine<T>, MaxBands> bands_ {};
    std::array<BandConfig, MaxBands> configs_ {};
    // Per-band enable flag read lock-free every block. The full BandConfig is only
    // read under the configDirty_ acquire gate; `enabled` is toggled often and read
    // on the hot path, so it gets its own atomic to avoid a torn/unsynchronized read.
    std::array<std::atomic<bool>, MaxBands> bandEnabled_ {};

    std::atomic<bool> softMode_ { false };
    std::atomic<bool> configDirty_ { false };
    std::atomic<bool> matchedBells_ { false };

    // Linear-phase state
    std::atomic<FilterMode> filterMode_ { FilterMode::MinimumPhase };
    std::atomic<bool> lpDirty_ { true };

    std::unique_ptr<FFTReal<T>> lpFft_;
    int lpFftSize_ = 0;
    
    std::vector<T> lpKernel_;
    std::vector<std::vector<T>> lpPrevBlock_;
    std::vector<T> lpFftIn_, lpFftOut_;
    // Pre-allocated recompute scratch (no audio-thread allocation on band changes).
    std::vector<T> lpMagScratch_, lpTempFreq_, lpImpulse_, lpKernelSpace_;
};

} // namespace dspark
