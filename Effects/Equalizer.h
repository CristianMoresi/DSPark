// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Equalizer.h
 * @brief Multi-band parametric equalizer with Minimum and Linear Phase modes.
 *
 * Combines multiple FilterEngine instances into a single processor. Supports up
 * to MaxBands simultaneous filter bands. Lock-free parameter updates and zero
 * memory allocations in the audio thread (the linear-phase engine, including
 * its recompute scratch, is fully pre-allocated in prepare()).
 *
 * Threading (per method; the model is docs/threading.md):
 * - Setup thread only, never concurrent with processing: prepare(), setState().
 * - Audio thread (the one stream owner): processBlock(), processSample(),
 *   reset().
 * - Control thread (one writer): setBand(), setNumBands(), setBandEnabled(),
 *   setFilterMode(), setSoftMode(), setMatchedBells(). Each band is published
 *   through its own seqlock over std::atomic words, so the audio thread adopts
 *   a whole band configuration or none of it; a change is consumed at the next
 *   block (or the next processSample()). Non-finite band fields are replaced
 *   with the band's current values.
 * - Any thread: getBandConfig(), getMagnitudeForFrequencyArray(), getState(),
 *   getNumBands(), getFilterMode(), getSoftMode(), getLatency(). The first two
 *   read the same seqlock and hand back a private copy, so they never observe a
 *   half-written band.
 * - Processing thread only: getBandFilter(), which returns a reference to a
 *   band's live FilterEngine (its contract, not the equalizer's).
 *
 * Dependencies: Filters.h (FilterEngine), AudioSpec.h, AudioBuffer.h,
 *               Biquad.h, DenormalGuard.h, DspMath.h, FFT.h, StateBlob.h.
 */

#include "Filters.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Biquad.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

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

    static_assert(std::atomic<T>::is_always_lock_free,
                  "audio-thread stores must not lock");

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares all bands and allocates necessary resources for processing.
     *
     * Must be called from the host's prepareToPlay/setup method. Performs all
     * memory allocations, including the linear-phase engine, so
     * setFilterMode(LinearPhase) works even when called after prepare().
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment (sample rate, max block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        spec_ = spec;
        for (int i = 0; i < MaxBands; ++i)
            bands_[i].prepare(spec);

        // Rebuild the linear-phase engine. lpFft_ gates the LP audio path, so
        // it is torn down first and re-created last (basic guarantee if an
        // allocation throws mid-way).
        lpFft_.reset();
        lpBlock_ = spec.maxBlockSize;

        // The FFT size is 4x the block size; cap it so the pow2 round-up can
        // never overflow int with an absurd host block size (LP mode is then
        // unavailable and getLatency() honestly reports 0).
        if (lpBlock_ > kLpMaxBlockSize)
        {
            lpBlock_ = 0;
            lpFftSize_ = 0;
            return;
        }

        // Overlap-save requires FFT size N >= L + M - 1
        // Where L is maxBlockSize, M is impulse response length.
        // We use M = 2 * L, so N must be >= 3 * L. We round up to next power of 2.
        int targetFftSize = lpBlock_ * 4;
        int fftPow2 = 1;
        while (fftPow2 < targetFftSize) fftPow2 <<= 1;
        lpFftSize_ = fftPow2;

        // Complex kernel representation (Real, Imaginary interleaved)
        lpKernel_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));

        // Overlap-save needs (M-1) samples of history where M = 2*maxBlockSize
        // is the FIR kernel length. Size the history buffer accordingly.
        lpPrevBlock_.resize(static_cast<size_t>(spec.numChannels));
        for (auto& pb : lpPrevBlock_)
            pb.assign(static_cast<size_t>(lpBlock_ * 2), T(0));

        lpFftIn_.assign(static_cast<size_t>(lpFftSize_), T(0));
        lpFftOut_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));

        // Pre-allocate recompute scratch so a live band change never allocates on
        // the audio thread (recomputeLinearPhaseKernel runs there via processBlock).
        lpMagScratch_.assign(static_cast<size_t>(lpFftSize_ / 2 + 1), T(1));
        lpTempFreq_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
        lpImpulse_.assign(static_cast<size_t>(lpFftSize_), T(0));
        lpKernelSpace_.assign(static_cast<size_t>(lpFftSize_), T(0));

        lpDirty_.store(true, std::memory_order_release);
        lpFft_ = std::make_unique<FFTReal<T>>(lpFftSize_); // gate opens last
    }

    /**
     * @brief Processes an audio buffer in-place.
     *
     * Guarantees zero allocations. Checks atomic flags lock-free to update DSP
     * state if the host changed parameters. In LinearPhase mode a pending band
     * change recomputes the FIR kernel on this thread (a bounded, allocation-
     * free spike); switching modes changes getLatency(), so notify the host
     * and consider reset() to clear the other path's tail.
     *
     * @param buffer Audio data to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;

        // Front-door non-finite guard: the IIR bands (and the FFT overlap-save
        // history in LinearPhase mode) latch a NaN/Inf input permanently.
        // Scrub non-finite input to 0 before either path runs.
        // No-op on finite input, so conformance metrics stay byte-identical.
        {
            const int gN = buffer.getNumSamples();
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            {
                T* d = buffer.getChannel(ch);
                for (int i = 0; i < gN; ++i)
                    if (!std::isfinite(d[i])) d[i] = T(0);
            }
        }

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
     * Pending band changes are consumed here too (applied immediately, without
     * the block path's parameter smoothing), so per-sample streams that never
     * call processBlock() still pick up setBand() and friends.
     *
     * @param input   Input sample.
     * @param channel Channel index (out-of-range channels pass through).
     * @return EQ'd output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        // Plain load first: the exchange RMW is only paid when a publication
        // is actually pending (this runs per sample).
        if (configDirty_.load(std::memory_order_acquire)
            && configDirty_.exchange(false, std::memory_order_acquire))
        {
            updateActiveFilters();
            const int n = numBands_.load(std::memory_order_relaxed);
            for (int i = 0; i < n; ++i)
                bands_[i].applyParametersNow();
            lpDirty_.store(true, std::memory_order_release);
        }

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
     *
     * Non-finite frequency/gain/Q fields fall back to the band's current
     * values (they would otherwise poison the serialized state, the analysis
     * curve, and the linear-phase kernel); wild type/slope values clamp.
     *
     * @param index  Band index.
     * @param config Complete band configuration.
     */
    void setBand(int index, const BandConfig& config)
    {
        if (index < 0 || index >= MaxBands) return;

        BandConfig cfg = config;
        // The fallback reads the control thread's own private master, never the
        // published words: a setter must not race the audio thread to decide
        // what "the band's current value" is.
        const BandConfig& prev = masterConfigs_[index];
        if (!std::isfinite(cfg.frequency)) cfg.frequency = prev.frequency;
        if (!std::isfinite(cfg.gain))      cfg.gain      = prev.gain;
        if (!std::isfinite(cfg.q))         cfg.q         = prev.q;
        cfg.type = static_cast<BandType>(std::clamp(static_cast<int>(cfg.type), 0,
                                                    static_cast<int>(BandType::Tilt)));
        cfg.slope = std::clamp(cfg.slope, 6, 48);

        masterConfigs_[index] = cfg;
        staged_[index].publish(cfg);
        bandEnabled_[index].store(cfg.enabled, std::memory_order_release);

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

            masterConfigs_[i] = cfg;
            staged_[i].publish(cfg);
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
     * high bells keep their analog shape, the state-of-the-art digital EQ
     * behaviour. Applies to the IIR engines, the linear-phase kernel, and the
     * analysis curve alike. Off by default for bit-compatibility with
     * previous output.
     */
    void setMatchedBells(bool enabled) noexcept
    {
        matchedBells_.store(enabled, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Returns the current configuration of a band.
     *
     * Safe from any thread: the band is read through its seqlock into a private
     * copy, so a configuration being published concurrently is either fully
     * seen or fully missed, never mixed with the previous one.
     *
     * @param index Band index.
     * @return Copy of the band's BandConfig.
     */
    [[nodiscard]] BandConfig getBandConfig(int index) const noexcept
    {
        if (index < 0 || index >= MaxBands) return {};
        return staged_[index].read();
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
            masterConfigs_[index].enabled = enabled;
            staged_[index].publish(masterConfigs_[index]);
            bandEnabled_[index].store(enabled, std::memory_order_release);
            configDirty_.store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Sets the filter processing mode (Minimum Phase or Linear Phase).
     *
     * Works before or after prepare() (the linear-phase engine is always
     * pre-allocated by prepare()). Wild enum values clamp. Switching modes
     * changes getLatency(): hosts must be notified.
     *
     * @param mode Filter mode.
     */
    void setFilterMode(FilterMode mode) noexcept
    {
        const int m = std::clamp(static_cast<int>(mode), 0,
                                 static_cast<int>(FilterMode::LinearPhase));
        filterMode_.store(static_cast<FilterMode>(m), std::memory_order_release);
        if (static_cast<FilterMode>(m) == FilterMode::LinearPhase)
            lpDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns the current filter mode. */
    [[nodiscard]] FilterMode getFilterMode() const noexcept
    {
        return filterMode_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the latency in samples.
     *
     * 0 for MinimumPhase; the prepared max block size for LinearPhase. Reports
     * 0 when the linear-phase engine is unavailable (not prepared yet), so the
     * value always matches the path that actually runs.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && lpFft_)
                ? lpBlock_ : 0;
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
     * Evaluates the same per-stage cascade the audio path applies (including
     * soft mode's Q cap and the matched-bell design), so the drawn curve
     * matches what is heard in both filter modes.
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
            const BandConfig cfg = staged_[b].read();
            if (!cfg.enabled) continue;

            BiquadCoeffs st[5];
            const int ns = buildBandStages(cfg, st);

            for (int i = 0; i < numPoints; ++i)
            {
                double mag = 1.0;
                for (int s = 0; s < ns; ++s)
                    mag *= st[s].getMagnitude(static_cast<double>(frequencies[i]), spec_.sampleRate);
                magnitudes[i] = static_cast<T>(static_cast<double>(magnitudes[i]) * mag);
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


    /** @brief Serializes bands and modes (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("PEQZ"), 1);
        const int n = numBands_.load(std::memory_order_relaxed);
        w.write("numBands", n);
        w.write("matchedBells", matchedBells_.load(std::memory_order_relaxed));
        w.write("filterMode", static_cast<int32_t>(filterMode_.load(std::memory_order_relaxed)));
        w.write("softMode", softMode_.load(std::memory_order_relaxed));
        char key[24];
        for (int i = 0; i < n; ++i)
        {
            const BandConfig cfg = getBandConfig(i);
            std::snprintf(key, sizeof(key), "b%d.freq", i);
            w.write(key, static_cast<float>(cfg.frequency));
            std::snprintf(key, sizeof(key), "b%d.gain", i);
            w.write(key, static_cast<float>(cfg.gain));
            std::snprintf(key, sizeof(key), "b%d.q", i);
            w.write(key, static_cast<float>(cfg.q));
            std::snprintf(key, sizeof(key), "b%d.type", i);
            w.write(key, static_cast<int32_t>(cfg.type));
            std::snprintf(key, sizeof(key), "b%d.slope", i);
            w.write(key, cfg.slope);
            std::snprintf(key, sizeof(key), "b%d.on", i);
            w.write(key, cfg.enabled);
        }
        return w.blob();
    }

    /** @brief Restores bands and modes from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("PEQZ")) return false;
        const int n = std::clamp(r.read("numBands", 0), 0, MaxBands);
        setMatchedBells(r.read("matchedBells", false));
        // Older blobs carry no mode keys: keep the instance's current modes.
        setFilterMode(static_cast<FilterMode>(
            r.read("filterMode", static_cast<int32_t>(filterMode_.load(std::memory_order_relaxed)))));
        setSoftMode(r.read("softMode", softMode_.load(std::memory_order_relaxed)));
        char key[24];
        for (int i = 0; i < n; ++i)
        {
            BandConfig cfg;
            std::snprintf(key, sizeof(key), "b%d.freq", i);
            cfg.frequency = static_cast<T>(r.read(key, 1000.0f));
            std::snprintf(key, sizeof(key), "b%d.gain", i);
            cfg.gain = static_cast<T>(r.read(key, 0.0f));
            std::snprintf(key, sizeof(key), "b%d.q", i);
            cfg.q = static_cast<T>(r.read(key, 0.707f));
            std::snprintf(key, sizeof(key), "b%d.type", i);
            cfg.type = static_cast<BandType>(r.read(key, 0));
            std::snprintf(key, sizeof(key), "b%d.slope", i);
            cfg.slope = r.read(key, 12);
            std::snprintf(key, sizeof(key), "b%d.on", i);
            cfg.enabled = r.read(key, true);
            setBand(i, cfg);
        }
        numBands_.store(n, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
        return true;
    }

protected:

    /**
     * @brief The band Q the audio path actually uses (soft mode caps it by gain).
     */
    [[nodiscard]] T effectiveQ(const BandConfig& cfg) const noexcept
    {
        T q = cfg.q;
        if (softMode_.load(std::memory_order_relaxed))
        {
            const T absGain = std::abs(cfg.gain);
            const T maxQ = T(1) + T(8) / (absGain + T(1));
            q = std::min(q, maxQ);
        }
        return q;
    }

    /**
     * @brief Translates BandConfigs into internal FilterEngine parameters safely.
     *
     * Audio thread; called by processBlock()/processSample() when configDirty_
     * is true. Each band is pulled through its seqlock into a plain local, so
     * the filter is configured from ONE publication. Reading the band by
     * reference instead would let a concurrent setBand() land between two field
     * reads and configure a filter with a frequency from one call and a slope
     * from the next. That is not a theoretical hazard: before this was a
     * seqlock, roughly one adoption in thirty was such a mixture.
     */
    void updateActiveFilters() noexcept
    {
        const bool matched = matchedBells_.load(std::memory_order_relaxed);
        int activeBands = numBands_.load(std::memory_order_relaxed);

        for (int i = 0; i < activeBands; ++i)
        {
            auto& filter = bands_[i];
            const BandConfig cfg = staged_[i].read();

            float freq = static_cast<float>(cfg.frequency);
            float gain = static_cast<float>(cfg.gain);
            float q    = static_cast<float>(effectiveQ(cfg));

            filter.setMatchedPeak(matched);
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

    /**
     * @brief Computes single-biquad coefficients for a band (analysis/kernel).
     * @param cfg Band configuration (its q must already be the effective one).
     * @return BiquadCoeffs structure.
     */
    [[nodiscard]] BiquadCoeffs computeBandCoeffs(const BandConfig& cfg) const noexcept
    {
        double sr = spec_.sampleRate;
        double f  = static_cast<double>(cfg.frequency);
        double g  = static_cast<double>(cfg.gain);
        double q  = static_cast<double>(cfg.q);

        switch (cfg.type)
        {
            case BandType::Peak:
                return matchedBells_.load(std::memory_order_relaxed)
                    ? BiquadCoeffs::makePeakMatched(sr, f, q, g)
                    : BiquadCoeffs::makePeak(sr, f, q, g);
            case BandType::LowShelf:  return BiquadCoeffs::makeLowShelf(sr, f, g, shelfSlopeFromQ(q, g));
            case BandType::HighShelf: return BiquadCoeffs::makeHighShelf(sr, f, g, shelfSlopeFromQ(q, g));
            case BandType::LowPass:   return BiquadCoeffs::makeLowPass(sr, f, q);
            case BandType::HighPass:  return BiquadCoeffs::makeHighPass(sr, f, q);
            case BandType::Notch:     return BiquadCoeffs::makeNotch(sr, f, q);
            case BandType::BandPass:  return BiquadCoeffs::makeBandPass(sr, f, q);
            case BandType::Tilt:      return BiquadCoeffs::makeTilt(sr, f, g);
        }
        return {};
    }

public:
    /**
     * @brief Fills `stages` with the ACTUAL biquad cascade for a band (per-stage
     * Butterworth Q for multi-stage LP/HP, single biquad otherwise) and returns
     * the stage count. Public analysis API: UIs use it to draw a magnitude
     * response that matches what the IIR FilterEngine really applies,
     * including the user's resonance on LP/HP cascades, soft mode's Q cap and
     * the matched-bell design. Mirrors the engine's own parameter
     * sanitization (frequency and Q floors). Requires prepare().
     * @param stages Output buffer (capacity >= 5).
     */
    [[nodiscard]] int buildBandStages(const BandConfig& cfg, BiquadCoeffs* stages) const noexcept
    {
        const double sr = spec_.sampleRate;
        // Mirror the FilterEngine's own clamps so the analysis matches the audio.
        const double f = std::clamp(static_cast<double>(cfg.frequency), 10.0, sr * 0.499);
        const T q = std::max(effectiveQ(cfg), T(0.1));

        if (cfg.type == BandType::LowPass || cfg.type == BandType::HighPass)
        {
            const bool lp = (cfg.type == BandType::LowPass);
            // The user Q scales the final cascade stage exactly like the engine.
            auto casc = FilterEngine<T>::cascadeForSlope(cfg.slope, static_cast<float>(q));
            int n = 0;
            if (casc.hasFirstOrder)
                stages[n++] = lp ? BiquadCoeffs::makeFirstOrderLowPass(sr, f)
                                 : BiquadCoeffs::makeFirstOrderHighPass(sr, f);
            for (int s = 0; s < casc.numSecondOrder; ++s)
            {
                const double stageQ = static_cast<double>(casc.qValues[s]);
                stages[n++] = lp ? BiquadCoeffs::makeLowPass(sr, f, stageQ)
                                 : BiquadCoeffs::makeHighPass(sr, f, stageQ);
            }
            return n;
        }

        BandConfig eff = cfg;
        eff.frequency = static_cast<T>(f);
        eff.q = q;
        stages[0] = computeBandCoeffs(eff);
        return 1;
    }

protected:
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
            const BandConfig cfg = staged_[b].read();
            if (!cfg.enabled) continue;

            BiquadCoeffs st[5];
            const int ns = buildBandStages(cfg, st);

            for (int k = 0; k < numBins; ++k)
            {
                const double freq = sr * static_cast<double>(k) / static_cast<double>(lpFftSize_);
                double m = 1.0;
                for (int s = 0; s < ns; ++s)
                    m *= st[s].getMagnitude(freq, sr);
                mag[k] = static_cast<T>(static_cast<double>(mag[k]) * m);
            }
        }

        // 2. Prepare Zero-Phase Frequency buffer (Real = Mag, Imag = 0)
        std::fill(lpTempFreq_.begin(), lpTempFreq_.end(), T(0));
        for (int k = 0; k < numBins; ++k)
            lpTempFreq_[2 * k] = mag[k];

        // 3. IFFT to get temporal impulse response (wrapped around t=0)
        lpFft_->inverse(lpTempFreq_.data(), lpImpulse_.data());

        // 4. Shift, Windowing and Zero-pad for Overlap-Save
        const int M = lpBlock_ * 2; // Desired Kernel length
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
            // 20*log10(N) dB - about -60 dB with the default sizes.
            lpKernelSpace_[i] = lpImpulse_[readIdx] * static_cast<T>(window);
        }

        // 5. Transform finalized zero-padded causal kernel to frequency domain
        lpFft_->forward(lpKernelSpace_.data(), lpKernel_.data());
    }

    /**
     * @brief Linear-phase processing via overlap-save FFT convolution.
     *
     * Blocks larger than the prepared max block size pass through unprocessed
     * (the engine's buffers are sized in prepare(); hosts honour maxBlockSize).
     *
     * @param buffer Audio buffer.
     */
    void processLinearPhase(AudioBufferView<T> buffer) noexcept
    {
        // Safety bound check: Prevent out-of-bounds if host pushes dynamic channel
        // counts. Channels beyond the prepared count pass through untouched.
        const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(lpPrevBlock_.size()));
        const int L   = buffer.getNumSamples();
        const int N   = lpFftSize_;
        const int M   = lpBlock_ * 2; // Kernel length

        // Safety check: block size cannot exceed pre-allocated maxBlockSize
        if (L > lpBlock_) return;

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

            // 4. Complex multiplication: H(k) * X(k)
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

            // 5. Inverse FFT
            lpFft_->inverse(lpFftOut_.data(), lpFftIn_.data()); // Reusing lpFftIn_ to save memory

            // 6. Overlap-save output extraction: valid data starts at index M - 1
            int offset = M - 1;
            for (int i = 0; i < L; ++i)
            {
                // Assign filtered output back to the channel
                channelData[i] = lpFftIn_[offset + i];
            }
        }
    }

    // The pow2 round-up computes lpBlock_ * 4: cap the block size so that can
    // never overflow int (hosts never get close; LP mode disables above it).
    static constexpr int kLpMaxBlockSize = 1 << 18;

    AudioSpec spec_ {};
    std::atomic<int> numBands_ { 0 };

    std::array<FilterEngine<T>, MaxBands> bands_ {};

    /**
     * @brief One band's published configuration: a seqlock over atomic words.
     *
     * A BandConfig is a multi-word set with an invariant -- half of one
     * publication combined with half of the next is not a configuration the
     * caller ever asked for -- so it uses the canonical seqlock of
     * docs/threading.md, matching Core/Biquad.h and Core/FIRFilter.h frame for
     * frame. Every word crossing threads is std::atomic; nothing here is a
     * plain scalar, and no reader ever holds a reference into this storage.
     */
    struct StagedBand
    {
        std::atomic<T>    frequency { T(1000) };
        std::atomic<T>    gain      { T(0) };
        std::atomic<T>    q         { T(0.707) };
        std::atomic<int>  type      { static_cast<int>(BandType::Peak) };
        std::atomic<int>  slope     { 12 };
        std::atomic<bool> enabled   { true };
        std::atomic<unsigned> seq   { 0 };

        /**
         * @brief Publishes a whole band configuration (control thread only).
         *
         * An odd sequence number marks a write in progress; a reader that
         * observes one retries. The release fence is mandatory, not decoration:
         * an acquire/release counter alone does not order the relaxed word
         * stores against it, so without the fence a reader could see a
         * mid-publish word while the counter still looked even
         * ([atomics.fences]/2; Boehm, "Can Seqlocks Get Along With Programming
         * Language Memory Models?", MSPC 2012). Control-thread only, so the
         * audio path pays nothing for it.
         */
        void publish(const BandConfig& c) noexcept
        {
            seq.fetch_add(1, std::memory_order_acq_rel);   // -> odd
            std::atomic_thread_fence(std::memory_order_release);
            frequency.store(c.frequency, std::memory_order_relaxed);
            gain.store(c.gain, std::memory_order_relaxed);
            q.store(c.q, std::memory_order_relaxed);
            type.store(static_cast<int>(c.type), std::memory_order_relaxed);
            slope.store(c.slope, std::memory_order_relaxed);
            enabled.store(c.enabled, std::memory_order_relaxed);
            seq.fetch_add(1, std::memory_order_release);   // -> even
        }

        /**
         * @brief Reads the band into a caller-private plain copy (any thread).
         *
         * The acquire fence keeps the word copy from sinking below the re-read
         * of the counter: an acquire LOAD orders only later accesses, so
         * without the fence a torn copy could pass the s0 == s1 check on a
         * weakly ordered target or after compiler reordering. It also pairs
         * with the writer's release fence, so a copy that read any mid-publish
         * word cannot validate. The writer's critical section is six word
         * stores, so this converges in at most a couple of iterations.
         */
        [[nodiscard]] BandConfig read() const noexcept
        {
            BandConfig c;
            unsigned s0, s1;
            do {
                s0 = seq.load(std::memory_order_acquire);
                c.frequency = frequency.load(std::memory_order_relaxed);
                c.gain      = gain.load(std::memory_order_relaxed);
                c.q         = q.load(std::memory_order_relaxed);
                c.type      = static_cast<BandType>(type.load(std::memory_order_relaxed));
                c.slope     = slope.load(std::memory_order_relaxed);
                c.enabled   = enabled.load(std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_acquire);
                s1 = seq.load(std::memory_order_relaxed);
            } while ((s0 & 1u) != 0u || s0 != s1);
            return c;
        }
    };

    /// Published band configurations. The ONLY band storage both threads reach.
    std::array<StagedBand, MaxBands> staged_ {};

    /// Control-thread-private master copy. Never read by the audio thread or by
    /// any readout: it exists so a setter can fall back to the band's current
    /// value for a non-finite field without reading published storage. Single
    /// writer by contract, so it needs no synchronization of its own.
    std::array<BandConfig, MaxBands> masterConfigs_ {};

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
    int lpBlock_ = 0; ///< Max block size the LP engine was sized for (= its latency).

    std::vector<T> lpKernel_;
    std::vector<std::vector<T>> lpPrevBlock_;
    std::vector<T> lpFftIn_, lpFftOut_;
    // Pre-allocated recompute scratch (no audio-thread allocation on band changes).
    std::vector<T> lpMagScratch_, lpTempFreq_, lpImpulse_, lpKernelSpace_;
};

} // namespace dspark
