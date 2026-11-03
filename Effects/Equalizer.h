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
 * MinimumPhase: zero latency. A band switched on or off (setBandEnabled(),
 * setNumBands(), setBand()'s enabled flag) crossfades in or out over 20 ms
 * instead of stepping.
 *
 * LinearPhase: one zero-phase FIR of a fixed length, independent of the host
 * block (setLinearPhaseKernelLength(); default 32768 taps at 44.1/48 kHz,
 * scaled with the sample rate to the nearest power of two), run by a
 * uniformly partitioned frequency-domain convolution with kernel/16-sample
 * partitions. Latency = kernel/2 + kernel/16 samples: 18432 samples (384 ms)
 * at 48 kHz with the default kernel; getLatency() reports it exactly. The
 * kernel resolves the low end: at 48 kHz the default realises a 100 Hz,
 * Q 2, +12 dB bell within 0.1 dB of the drawn curve. A band change is
 * designed on the audio thread in bounded slices, one per partition hop
 * (under 1 ms of work each), then crossfaded over one hop: one bell is heard
 * 4 hops after the change (0.17 s at 48 kHz), 8 bells 6 hops (0.26 s). The
 * first LinearPhase block after prepare(), reset() or a switch into the mode
 * completes a pending design at once (a one-off CPU cost of a few ms).
 * Memory: about 2 MB per stereo instance at 48 kHz, proportional to the
 * kernel length.
 *
 * Threading (per method; the model is docs/threading.md):
 * - Setup thread only, never concurrent with processing: prepare(), setState(),
 *   setLinearPhaseKernelLength() (applied at the next prepare()),
 *   getLinearPhaseKernelLength().
 * - Audio thread (the one stream owner): processBlock(), processSample(),
 *   reset().
 * - Control thread (one writer): setBand(), setNumBands(), setBandEnabled(),
 *   setFilterMode(), setSoftMode(), setMatchedBells(). Each band is published
 *   through its own seqlock over std::atomic words, so the audio thread adopts
 *   a whole band configuration or none of it; a change is consumed at the next
 *   block (or the next processSample()). Non-finite band fields are replaced
 *   with the band's current values. The audio thread's read of that seqlock is
 *   BOUNDED (StagedBand::tryRead): it never waits for the control thread to
 *   finish publishing, and a band whose read gives up keeps its current
 *   coefficients while the dirty flag is re-armed, so the change lands one call
 *   later instead of holding the callback open.
 * - Any thread: getBandConfig(), getMagnitudeForFrequencyArray(), getState(),
 *   getNumBands(), getFilterMode(), getSoftMode(), getLatency(). The first two
 *   read the same seqlock and hand back a private copy, so they never observe a
 *   half-written band. They use the UNBOUNDED StagedBand::read(): a readout has
 *   no previously adopted copy to keep, and it is not real-time.
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
#include <cassert>
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
        LinearPhase   ///< Partitioned FFT convolution (fixed latency, see getLatency()).
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
    static_assert(MaxBands > 0, "an equalizer needs at least one band");

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

        // Band crossfade scratch: one block of dry signal per channel the
        // FilterEngines process (they hold at most 16 channels).
        bandDryChannels_ = std::min(spec.numChannels, 16);
        bandDryStride_ = spec.maxBlockSize;
        bandDry_.assign(static_cast<size_t>(bandDryChannels_) * static_cast<size_t>(bandDryStride_), T(0));
        bandFadeStep_ = static_cast<T>(1.0 / std::max(1.0, spec.sampleRate * kBandFadeSeconds));
        bandSnap_ = true;

        // Rebuild the linear-phase engine. lpFft_ gates the LP audio path, so
        // it is torn down first and re-created last (basic guarantee if an
        // allocation throws mid-way).
        lpFft_.reset();
        lpDesignFft_.reset();
        lpKernelLength_ = 0;
        lpPartition_ = 0;
        lpPartitions_ = 0;
        lpChannels_ = spec.numChannels;

        const int kernel = resolveKernelLength(spec.sampleRate);
        const int part = std::max(kLpMinPartition, kernel / kLpPartitionCount);
        const int parts = kernel / part;
        const size_t specLen = static_cast<size_t>(2 * part + 2);
        const size_t channels = static_cast<size_t>(lpChannels_);

        for (auto& set : lpKernelSet_)
            set.assign(static_cast<size_t>(parts) * specLen, T(0));
        lpFdl_.assign(channels * static_cast<size_t>(parts) * specLen, T(0));
        lpInput_.assign(channels * static_cast<size_t>(2 * part), T(0));
        lpOutput_.assign(channels * static_cast<size_t>(part), T(0));
        lpAccum_.assign(specLen, T(0));
        lpBlockTime_.assign(static_cast<size_t>(2 * part), T(0));
        lpBlockTimeOld_.assign(static_cast<size_t>(2 * part), T(0));

        const int design = 2 * kernel;
        lpWindow_.resize(static_cast<size_t>(kernel));
        for (int i = 0; i < kernel; ++i)
        {
            // Periodic Blackman-Harris: symmetric about i = kernel/2, the
            // kernel's centre, so the FIR is exactly linear phase.
            const double t = 2.0 * pi<double> * static_cast<double>(i) / static_cast<double>(kernel);
            lpWindow_[static_cast<size_t>(i)] = static_cast<T>(
                0.35875 - 0.48829 * std::cos(t) + 0.14128 * std::cos(2.0 * t) - 0.01168 * std::cos(3.0 * t));
        }
        lpMag_.assign(static_cast<size_t>(kernel + 1), T(1));
        lpDesignSpec_.assign(static_cast<size_t>(design + 2), T(0));
        lpDesignTime_.assign(static_cast<size_t>(design), T(0));
        lpKernelTime_.assign(static_cast<size_t>(kernel), T(0));

        lpDesignFft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(design));
        auto partitionFft = std::make_unique<FFTReal<T>>(static_cast<size_t>(2 * part));

        lpKernelLength_ = kernel;
        lpPartition_ = part;
        lpPartitions_ = parts;
        lpFft_ = std::move(partitionFft); // gate opens last

        // Design the kernel of the current configuration now, on the setup
        // thread, so the engine never starts on an empty kernel.
        lpDirty_.store(false, std::memory_order_relaxed);
        lpStage_ = LpStage::Snapshot;
        lpRestartPending_ = false;
        lpSwapPending_ = false;
        lpActive_ = 0;
        for (int guard = 0; lpStage_ != LpStage::Idle && guard < kLpSyncSteps; ++guard)
            advanceDesign(DesignPace::Setup);
        if (lpSwapPending_) { lpActive_ = 1 - lpActive_; lpSwapPending_ = false; }
        clearLinearPhaseStream();
        lpSnap_ = true;
        lpWasActive_ = false;
    }

    /**
     * @brief Processes an audio buffer in-place.
     *
     * Guarantees zero allocations. Checks atomic flags lock-free to update DSP
     * state if the host changed parameters. In LinearPhase mode a pending band
     * change is designed on this thread in bounded slices and crossfaded in
     * (see the file comment); any block length works and the output does not
     * depend on how the host splits the stream. Switching modes changes
     * getLatency(), so notify the host and consider reset() to clear the other
     * path's tail.
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
            if (!lpWasActive_)
            {
                // Entering the mode: the engine's history belongs to an
                // earlier stretch of audio, and any pending design completes
                // on this block.
                clearLinearPhaseStream();
                lpSnap_ = true;
                lpWasActive_ = true;
            }
            processLinearPhase(buffer);
            return;
        }
        lpWasActive_ = false;
        processMinimumPhase(buffer);
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

        bandSnap_ = true;
        clearLinearPhaseStream();
        lpSnap_ = true;
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
     * @brief Switches Peak bands to the analog-matched (de-cramped) design.
     *
     * Bilinear bells cramp near Nyquist (narrower, response pinned at fs/2);
     * the matched design (BiquadCoeffs::makePeakMatched, impulse-invariant
     * poles with a magnitude-matched numerator) keeps high bells on their
     * analog shape, the state-of-the-art digital EQ behaviour. Applies to the
     * IIR engines, the linear-phase kernel, and the analysis curve alike.
     * On by default (worst audible-band deviation from the analog bell at
     * 48 kHz: 2.9 dB matched vs 11 dB bilinear); set false for the classic
     * bilinear (cookbook) bells.
     */
    void setMatchedBells(bool enabled) noexcept
    {
        matchedBells_.store(enabled, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns whether Peak bands use the matched design (setMatchedBells()). */
    [[nodiscard]] bool isMatchedBells() const noexcept
    {
        return matchedBells_.load(std::memory_order_relaxed);
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
     * changes getLatency(): hosts must be notified. The first LinearPhase
     * block after a switch completes any pending kernel design at once.
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
     * 0 for MinimumPhase; kernel/2 + kernel/16 for LinearPhase (the zero-phase
     * kernel's centre plus one partition of buffering: 18432 samples at the
     * default 48 kHz kernel), independent of the host block size. Reports 0
     * when the linear-phase engine is unavailable (not prepared yet), so the
     * value always matches the path that actually runs.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && lpFft_)
                ? lpKernelLength_ / 2 + lpPartition_ : 0;
    }

    /**
     * @brief Requests the linear-phase kernel length in taps.
     *
     * Applied at the next prepare() (it sizes the engine). 0 or a negative
     * value selects the automatic length: 32768 taps at 44.1/48 kHz, scaled
     * with the sample rate to the nearest power of two (65536 at 96 kHz).
     * Explicit lengths clamp to [256, 262144] and round up to a power of two.
     * Longer kernels resolve lower frequencies and cost latency
     * (kernel/2 + kernel/16) and memory; CPU per sample barely changes.
     */
    void setLinearPhaseKernelLength(int taps) noexcept
    {
        lpKernelRequest_.store(taps <= 0 ? 0 : std::clamp(taps, kLpMinKernel, kLpMaxKernel),
                               std::memory_order_relaxed);
    }

    /** @brief The prepared linear-phase kernel length in taps (0 before
     *  prepare()). Read it from the setup thread or once prepared. */
    [[nodiscard]] int getLinearPhaseKernelLength() const noexcept { return lpKernelLength_; }

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
     * matches what is heard in both filter modes (in LinearPhase within the
     * resolution of the kernel length).
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
     *
     * The index is clamped, so an out-of-range one returns band 0 or the last
     * band instead of binding a reference past the end of the band array. A
     * debug build asserts first: clamping keeps a release build defined, it
     * does not make a wrong index mean anything.
     *
     * @param index Band index. Clamped, not validated -- see the note above.
     * @return Reference to the FilterEngine for this band.
     */
    [[nodiscard]] FilterEngine<T>& getBandFilter(int index) noexcept
    {
        assert(index >= 0 && index < MaxBands);
        return bands_[static_cast<std::size_t>(std::clamp(index, 0, MaxBands - 1))];
    }

    /** @brief Const overload. Same clamping contract. */
    [[nodiscard]] const FilterEngine<T>& getBandFilter(int index) const noexcept
    {
        assert(index >= 0 && index < MaxBands);
        return bands_[static_cast<std::size_t>(std::clamp(index, 0, MaxBands - 1))];
    }


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
        setMatchedBells(r.read("matchedBells", true));
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
     *
     * The pull is bounded (tryRead): a band whose read gives up is left with
     * the coefficients it already has, and configDirty_ is re-armed so the whole
     * set is retried on a later call. Reconfiguring a band from its published
     * config is idempotent, so retrying all of them costs nothing but the work.
     */
    void updateActiveFilters() noexcept
    {
        const bool matched = matchedBells_.load(std::memory_order_relaxed);
        int activeBands = numBands_.load(std::memory_order_relaxed);

        for (int i = 0; i < activeBands; ++i)
        {
            auto& filter = bands_[i];
            BandConfig cfg;
            if (!staged_[i].tryRead(cfg))
            {
                // Bounded read gave up: keep this band's current coefficients
                // and pick the publication up on a later call.
                configDirty_.store(true, std::memory_order_release);
                continue;
            }

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
     * @param cfg Band configuration to translate into processing stages.
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
    static constexpr double kBandFadeSeconds = 0.02;

    [[nodiscard]] static T moveTowards(T from, T to, T delta) noexcept
    {
        return from < to ? std::min(to, from + delta) : std::max(to, from - delta);
    }

    /**
     * @brief MinimumPhase processing, with each band's on/off state
     *        crossfaded over kBandFadeSeconds.
     *
     * A settled enabled band runs exactly as before (no extra work); a band
     * fading in or out processes a copy and blends it with the dry block. A
     * band fading in restarts from cleared filter state, settled on its
     * current parameters. Blocks longer than the prepared maximum run in
     * prepared-size slices so the dry scratch always holds them.
     */
    void processMinimumPhase(AudioBufferView<T> buffer) noexcept
    {
        const int activeBands = numBands_.load(std::memory_order_relaxed);
        const int total = buffer.getNumSamples();
        const int slice = bandDryStride_ > 0 ? bandDryStride_ : std::max(total, 1);

        for (int offset = 0; offset < total; offset += slice)
        {
            const int n = std::min(slice, total - offset);
            AudioBufferView<T> view = buffer.getSubView(offset, n);
            for (int i = 0; i < MaxBands; ++i)
            {
                const bool on = i < activeBands && bandEnabled_[i].load(std::memory_order_relaxed);
                const T target = on ? T(1) : T(0);
                T& mix = bandMix_[static_cast<size_t>(i)];
                if (bandSnap_ || bandDry_.empty()) mix = target;
                if (mix == target)
                {
                    if (on) bands_[i].processBlock(view);
                    continue;
                }

                if (mix == T(0)) bands_[i].reset();   // re-entering band: clean state

                const int nCh = std::min(view.getNumChannels(), bandDryChannels_);
                for (int ch = 0; ch < nCh; ++ch)
                    std::copy_n(view.getChannel(ch), n,
                                bandDry_.data() + static_cast<size_t>(ch) * static_cast<size_t>(bandDryStride_));
                bands_[i].processBlock(view);
                const T start = mix;
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* wet = view.getChannel(ch);
                    const T* dry = bandDry_.data() + static_cast<size_t>(ch) * static_cast<size_t>(bandDryStride_);
                    for (int k = 0; k < n; ++k)
                    {
                        const T m = moveTowards(start, target, bandFadeStep_ * static_cast<T>(k + 1));
                        wet[k] = dry[k] + (wet[k] - dry[k]) * m;
                    }
                }
                mix = moveTowards(start, target, bandFadeStep_ * static_cast<T>(n));
            }
            bandSnap_ = false;
        }
    }

    // -- Linear-phase engine ----------------------------------------------------

    enum class LpStage { Idle, Snapshot, Magnitude, Transform, Partition };

    static constexpr int kLpMinKernel = 256;
    static constexpr int kLpMaxKernel = 1 << 18;
    static constexpr int kLpPartitionCount = 16;   ///< Partitions per kernel.
    static constexpr int kLpMinPartition = 64;
    static constexpr int kLpBinBudget = 131072;    ///< Stage evaluations per hop (< 1 ms).
    static constexpr int kLpPartitionSlices = 1;   ///< Hops spent on partition FFTs.
    static constexpr int kLpSyncSteps = 16;        ///< Bound of a synchronous design.

    /** @brief How much of a design one advanceDesign() call may do. */
    enum class DesignPace
    {
        Sliced,     ///< Audio thread, one bounded slice per partition hop.
        Immediate,  ///< Audio thread, whole stages at once (bounded seqlock reads).
        Setup       ///< Setup thread, whole stages, unbounded seqlock reads.
    };

    /** @brief Kernel length for a sample rate (see setLinearPhaseKernelLength()). */
    [[nodiscard]] int resolveKernelLength(double sampleRate) const noexcept
    {
        int n = kLpMinKernel;
        const int requested = lpKernelRequest_.load(std::memory_order_relaxed);
        if (requested > 0)
        {
            while (n < requested && n < kLpMaxKernel) n <<= 1;
            return n;
        }
        // Nearest power of two (in octaves) to 32768 * fs / 48 kHz.
        const double target = 32768.0 * sampleRate / 48000.0;
        while (n < kLpMaxKernel && static_cast<double>(n) * 1.4142135623730951 < target) n <<= 1;
        return n;
    }

    /** @brief Clears the engine's audio history (not the kernels). */
    void clearLinearPhaseStream() noexcept
    {
        std::fill(lpFdl_.begin(), lpFdl_.end(), T(0));
        std::fill(lpInput_.begin(), lpInput_.end(), T(0));
        std::fill(lpOutput_.begin(), lpOutput_.end(), T(0));
        lpFifoPos_ = 0;
        lpFdlPos_ = 0;
    }

    /**
     * @brief Runs one bounded slice of a kernel design into the inactive set.
     *
     * Snapshot: pulls every band through its bounded seqlock read (a give-up
     * retries on the next hop). Magnitude: evaluates the cascade magnitude on
     * at most kLpBinBudget stage-bins of the design grid. Transform: one inverse FFT,
     * the shift to the kernel centre and the window. Partition: forward FFTs
     * of 1/kLpPartitionSlices of the partitions. The finished set is swapped
     * in by the caller (crossfaded during processing).
     */
    void advanceDesign(DesignPace pace) noexcept
    {
        const int kernel = lpKernelLength_;
        const int design = 2 * kernel;
        switch (lpStage_)
        {
            case LpStage::Idle:
                return;

            case LpStage::Snapshot:
            {
                const int activeBands = numBands_.load(std::memory_order_relaxed);
                for (int b = 0; b < activeBands; ++b)
                {
                    BandConfig cfg;
                    if (pace == DesignPace::Setup)
                    {
                        cfg = staged_[b].read();
                    }
                    else if (!staged_[b].tryRead(cfg))
                    {
                        // Bounded read gave up: abandon this design (the active
                        // kernel keeps playing) and re-arm, so the next hop
                        // starts it again from a complete band set.
                        lpDirty_.store(true, std::memory_order_release);
                        lpStage_ = LpStage::Idle;
                        return;
                    }
                    lpStageCount_[static_cast<size_t>(b)] = cfg.enabled
                        ? buildBandStages(cfg, lpStages_.data() + static_cast<size_t>(b) * 5)
                        : 0;
                }
                lpDesignBands_ = activeBands;
                lpDesignStages_ = 0;
                for (int b = 0; b < activeBands; ++b) lpDesignStages_ += lpStageCount_[static_cast<size_t>(b)];
                lpStep_ = 0;
                lpStage_ = LpStage::Magnitude;
            }
                // The snapshot is cheap: start the magnitude in the same hop.
                [[fallthrough]];

            case LpStage::Magnitude:
            {
                const int bins = kernel + 1;
                const int perSlice = pace == DesignPace::Sliced
                    ? std::max(256, kLpBinBudget / std::max(1, lpDesignStages_))
                    : bins;
                const int first = lpStep_;
                const int last = std::min(bins, first + perSlice);
                // Bin k sits at w = pi * k / kernel. The unit phasor is rotated
                // bin to bin in double (error grows ~1e-16 per step), seeded
                // exactly at the slice start.
                const double dw = pi<double> / static_cast<double>(kernel);
                const double rc = std::cos(dw), rs = std::sin(dw);
                double c1 = std::cos(dw * static_cast<double>(first));
                double s1 = std::sin(dw * static_cast<double>(first));
                for (int k = first; k < last; ++k)
                {
                    const double c2 = 2.0 * c1 * c1 - 1.0;
                    const double s2 = 2.0 * s1 * c1;
                    double power = 1.0;
                    for (int b = 0; b < lpDesignBands_; ++b)
                    {
                        const BiquadCoeffs* st = lpStages_.data() + static_cast<size_t>(b) * 5;
                        for (int s = 0; s < lpStageCount_[static_cast<size_t>(b)]; ++s)
                        {
                            const BiquadCoeffs& c = st[s];
                            const double nRe = c.b0 + c.b1 * c1 + c.b2 * c2;
                            const double nIm = -(c.b1 * s1 + c.b2 * s2);
                            const double dRe = 1.0 + c.a1 * c1 + c.a2 * c2;
                            const double dIm = -(c.a1 * s1 + c.a2 * s2);
                            const double den = dRe * dRe + dIm * dIm;
                            power *= (den > 1e-300) ? (nRe * nRe + nIm * nIm) / den : 0.0;
                        }
                    }
                    lpMag_[static_cast<size_t>(k)] = static_cast<T>(std::sqrt(power));
                    const double cn = c1 * rc - s1 * rs;
                    s1 = s1 * rc + c1 * rs;
                    c1 = cn;
                }
                if (last >= bins) lpStage_ = LpStage::Transform;
                else lpStep_ = last;
                return;
            }

            case LpStage::Transform:
            {
                // Zero-phase spectrum -> impulse centred on t = 0 (circular).
                std::fill(lpDesignSpec_.begin(), lpDesignSpec_.end(), T(0));
                for (int k = 0; k <= kernel; ++k)
                    lpDesignSpec_[static_cast<size_t>(2 * k)] = lpMag_[static_cast<size_t>(k)];
                lpDesignFft_->inverse(lpDesignSpec_.data(), lpDesignTime_.data());

                // Centre at kernel/2 and window. Tap 0 pairs with the absent
                // tap `kernel` and is zeroed, so the FIR is symmetric.
                lpKernelTime_[0] = T(0);
                for (int i = 1; i < kernel; ++i)
                {
                    const int src = (i - kernel / 2 + design) % design;
                    lpKernelTime_[static_cast<size_t>(i)] =
                        lpDesignTime_[static_cast<size_t>(src)] * lpWindow_[static_cast<size_t>(i)];
                }
                lpStep_ = 0;
                lpStage_ = LpStage::Partition;
                return;
            }

            case LpStage::Partition:
            {
                const int part = lpPartition_;
                const size_t specLen = static_cast<size_t>(2 * part + 2);
                auto& set = lpKernelSet_[static_cast<size_t>(1 - lpActive_)];
                const int perSlice = pace == DesignPace::Sliced
                    ? (lpPartitions_ + kLpPartitionSlices - 1) / kLpPartitionSlices
                    : lpPartitions_;
                const int first = lpStep_ * perSlice;
                const int last = std::min(lpPartitions_, first + perSlice);
                for (int p = first; p < last; ++p)
                {
                    std::copy_n(lpKernelTime_.data() + static_cast<size_t>(p) * static_cast<size_t>(part),
                                part, lpBlockTime_.data());
                    std::fill(lpBlockTime_.begin() + part, lpBlockTime_.end(), T(0));
                    lpFft_->forward(lpBlockTime_.data(), set.data() + static_cast<size_t>(p) * specLen);
                }
                if (last >= lpPartitions_)
                {
                    lpSwapPending_ = true;
                    lpStage_ = lpRestartPending_ ? LpStage::Snapshot : LpStage::Idle;
                    lpRestartPending_ = false;
                }
                else
                {
                    ++lpStep_;
                }
                return;
            }
        }
    }

    /** @brief Completes a pending design now and adopts it without a crossfade
     *  (first block after prepare/reset/mode entry). */
    void finishDesignNow() noexcept
    {
        if (lpDirty_.exchange(false, std::memory_order_acquire))
        {
            lpStage_ = LpStage::Snapshot;
            lpRestartPending_ = false;
        }
        for (int guard = 0; lpStage_ != LpStage::Idle && guard < kLpSyncSteps; ++guard)
            advanceDesign(DesignPace::Immediate);
        if (lpSwapPending_)
        {
            lpActive_ = 1 - lpActive_;
            lpSwapPending_ = false;
        }
    }

    /**
     * @brief One partition hop: schedule the design, then convolve every
     *        prepared channel (uniformly partitioned overlap-save).
     */
    void runLinearPhaseHop() noexcept
    {
        if (lpDirty_.load(std::memory_order_relaxed)
            && lpDirty_.exchange(false, std::memory_order_acquire))
        {
            if (lpStage_ == LpStage::Idle) lpStage_ = LpStage::Snapshot;
            else lpRestartPending_ = true;
        }

        bool crossfade = false;
        if (lpSwapPending_)
        {
            crossfade = true;
            lpSwapPending_ = false;
        }
        else if (lpStage_ != LpStage::Idle)
        {
            advanceDesign(DesignPace::Sliced);
        }

        const int part = lpPartition_;
        const int parts = lpPartitions_;
        const size_t specLen = static_cast<size_t>(2 * part + 2);
        const int bins = part + 1;
        const int newSet = crossfade ? 1 - lpActive_ : lpActive_;
        const T invPart = T(1) / static_cast<T>(part);

        for (int ch = 0; ch < lpChannels_; ++ch)
        {
            T* input = lpInput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(2 * part);
            T* output = lpOutput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(part);
            T* fdl = lpFdl_.data() + static_cast<size_t>(ch) * static_cast<size_t>(parts) * specLen;

            lpFft_->forward(input, fdl + static_cast<size_t>(lpFdlPos_) * specLen);

            auto convolve = [&](int setIndex, T* time) noexcept {
                std::fill(lpAccum_.begin(), lpAccum_.end(), T(0));
                const T* kernel = lpKernelSet_[static_cast<size_t>(setIndex)].data();
                for (int p = 0; p < parts; ++p)
                {
                    const int slot = (lpFdlPos_ - p + parts) % parts;
                    simd::complexMulAccum(lpAccum_.data(), fdl + static_cast<size_t>(slot) * specLen,
                                          kernel + static_cast<size_t>(p) * specLen, bins);
                }
                lpFft_->inverse(lpAccum_.data(), time);
            };

            convolve(newSet, lpBlockTime_.data());
            if (crossfade)
            {
                convolve(lpActive_, lpBlockTimeOld_.data());
                for (int i = 0; i < part; ++i)
                {
                    const T oldY = lpBlockTimeOld_[static_cast<size_t>(part + i)];
                    const T newY = lpBlockTime_[static_cast<size_t>(part + i)];
                    output[i] = oldY + (newY - oldY) * (static_cast<T>(i + 1) * invPart);
                }
            }
            else
            {
                std::copy_n(lpBlockTime_.data() + part, part, output);
            }

            // The current half becomes the next hop's history.
            std::copy_n(input + part, part, input);
        }

        lpFdlPos_ = (lpFdlPos_ + 1) % parts;
        if (crossfade) lpActive_ = 1 - lpActive_;
    }

    /**
     * @brief LinearPhase processing: any block length streams through a
     *        partition-sized FIFO, so the output is identical whatever the
     *        host's block partition. Channels beyond the prepared count pass
     *        through unprocessed (not delayed).
     */
    void processLinearPhase(AudioBufferView<T> buffer) noexcept
    {
        if (lpSnap_)
        {
            finishDesignNow();
            lpSnap_ = false;
        }

        const int nCh = std::min(buffer.getNumChannels(), lpChannels_);
        const int total = buffer.getNumSamples();
        const int part = lpPartition_;
        int done = 0;
        while (done < total)
        {
            const int seg = std::min(total - done, part - lpFifoPos_);
            for (int ch = 0; ch < lpChannels_; ++ch)
            {
                T* in = lpInput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(2 * part)
                      + static_cast<size_t>(part + lpFifoPos_);
                const T* out = lpOutput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(part)
                             + static_cast<size_t>(lpFifoPos_);
                if (ch < nCh)
                {
                    T* data = buffer.getChannel(ch) + done;
                    std::copy_n(data, seg, in);      // read the input before
                    std::copy_n(out, seg, data);     // overwriting it in place
                }
                else
                {
                    std::fill_n(in, seg, T(0));
                }
            }
            lpFifoPos_ += seg;
            done += seg;
            if (lpFifoPos_ == part)
            {
                runLinearPhaseHop();
                lpFifoPos_ = 0;
            }
        }
    }

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
        // Initialised FROM BandConfig's own defaults, never from a second copy
        // of the literals: a staged band that starts life disagreeing with the
        // BandConfig a caller reads back is a bug no test would obviously
        // catch, and two hand-written default sets drift one field at a time.
        std::atomic<T>    frequency { BandConfig{}.frequency };
        std::atomic<T>    gain      { BandConfig{}.gain };
        std::atomic<T>    q         { BandConfig{}.q };
        std::atomic<int>  type      { static_cast<int>(BandConfig{}.type) };
        std::atomic<int>  slope     { BandConfig{}.slope };
        std::atomic<bool> enabled   { BandConfig{}.enabled };
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

        /// Attempt bound for the audio-thread read (tryRead). Three is enough
        /// that attempt 1 succeeds whenever the reader does not collide with a
        /// publish, with two spare attempts to absorb one unlucky collision;
        /// past that, deferring the update by one call costs less than spending
        /// more of the block budget on it.
        static constexpr int kSeqlockMaxAttempts = 3;

        /** @brief Relaxed copy of the six published words (no ordering of its
         *  own; the caller's fences and counter checks provide that).
         *
         *  Shared by both readers below so the word list exists exactly once: a
         *  field added to the band cannot be picked up by one reader and missed
         *  by the other. */
        void loadWordsRelaxed(BandConfig& c) const noexcept
        {
            c.frequency = frequency.load(std::memory_order_relaxed);
            c.gain      = gain.load(std::memory_order_relaxed);
            c.q         = q.load(std::memory_order_relaxed);
            c.type      = static_cast<BandType>(type.load(std::memory_order_relaxed));
            c.slope     = slope.load(std::memory_order_relaxed);
            c.enabled   = enabled.load(std::memory_order_relaxed);
        }

        /**
         * @brief The bounded seqlock read for the AUDIO thread.
         *
         * Makes at most kSeqlockMaxAttempts validation attempts and then gives
         * up rather than spin until the control thread finishes publishing --
         * an audio-thread loop whose exit depends on another thread's progress
         * is a stall of that thread's scheduling quantum, inside the callback.
         * The accept test is the same one read() uses, so bounding can never
         * make a torn set adoptable; it can only defer an adoption. `out` is
         * written only once a copy has validated, so a caller may pass storage
         * it is still using and a give-up leaves it intact.
         *
         * An odd counter means the writer is mid-publish: skip the copy
         * entirely instead of copying six words that would be discarded.
         *
         * @param out Receives the band configuration if and only if true is returned.
         * @return true if a consistent configuration was adopted; false if the
         *         attempt bound was reached (the caller keeps what it has and
         *         should re-arm its dirty flag to retry on a later call).
         */
        [[nodiscard]] bool tryRead(BandConfig& out) const noexcept
        {
            for (int attempt = 0; attempt < kSeqlockMaxAttempts; ++attempt)
            {
                const unsigned s0 = seq.load(std::memory_order_acquire);
                if ((s0 & 1u) != 0u) continue;   // writer mid-publish: do not copy
                BandConfig c;
                loadWordsRelaxed(c);
                std::atomic_thread_fence(std::memory_order_acquire);
                if (s0 == seq.load(std::memory_order_relaxed))
                {
                    out = c;                     // commit only on validation
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Reads the band into a caller-private plain copy (readouts).
         *
         * Control- and GUI-thread entry point, and NOT for the audio thread:
         * this retry is unbounded on purpose. A readout has no previously
         * adopted copy to fall back on, so "sorry, try again" would be a worse
         * API than a microsecond of spinning; the audio path uses tryRead().
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
                loadWordsRelaxed(c);
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
    std::atomic<bool> matchedBells_ { true };  ///< Matched (de-cramped) bells by default.

    // Linear-phase state
    std::atomic<FilterMode> filterMode_ { FilterMode::MinimumPhase };
    std::atomic<bool> lpDirty_ { true };
    std::atomic<int> lpKernelRequest_ { 0 };   ///< 0 = automatic length.

    // MinimumPhase band crossfades (audio thread).
    std::array<T, MaxBands> bandMix_ {};
    std::vector<T> bandDry_;
    int bandDryChannels_ = 0;
    int bandDryStride_ = 0;
    T bandFadeStep_ = T(1);
    bool bandSnap_ = true;

    // Engine geometry (written by prepare(), read by the audio thread).
    std::unique_ptr<FFTReal<T>> lpFft_;        ///< Partition FFT (2 x partition); gates the engine.
    std::unique_ptr<FFTReal<T>> lpDesignFft_;  ///< Design FFT (2 x kernel).
    int lpKernelLength_ = 0;
    int lpPartition_ = 0;
    int lpPartitions_ = 0;
    int lpChannels_ = 0;

    // Streaming state (audio thread).
    std::array<std::vector<T>, 2> lpKernelSet_;   ///< Partition spectra, active + incoming.
    int lpActive_ = 0;
    std::vector<T> lpFdl_;                        ///< Per channel: partitions x spectrum.
    std::vector<T> lpInput_, lpOutput_;           ///< Per channel FIFOs.
    std::vector<T> lpAccum_, lpBlockTime_, lpBlockTimeOld_;
    int lpFifoPos_ = 0;
    int lpFdlPos_ = 0;
    bool lpSnap_ = true;
    bool lpWasActive_ = false;

    // Design state (audio thread, sliced).
    LpStage lpStage_ = LpStage::Idle;
    int lpStep_ = 0;
    bool lpRestartPending_ = false;
    bool lpSwapPending_ = false;
    int lpDesignBands_ = 0;
    int lpDesignStages_ = 0;
    std::array<BiquadCoeffs, MaxBands * 5> lpStages_ {};
    std::array<int, MaxBands> lpStageCount_ {};
    std::vector<T> lpWindow_, lpMag_;
    std::vector<T> lpDesignSpec_, lpDesignTime_, lpKernelTime_;
};

} // namespace dspark
