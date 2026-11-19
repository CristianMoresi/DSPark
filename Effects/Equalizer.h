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
 * Analog response (setAnalogResponse(), off by default): every band is
 * designed from its analog prototype by FilterDesign, so its curve does not
 * shrink towards Nyquist, and runs in a BiquadCascade whose coefficients are
 * ramped: a parameter change glides (slowly for a small move, fast for a
 * jump) and a band at 0 dB leaves the path and costs nothing. It opens cuts
 * of any slope from 0 to 96 dB per octave and bells, shelves, notches and
 * band-passes of order 1 to 16. The all-pass, flat tilt and wall bands, and
 * any band of order above 1, always run this way.
 *
 * Placement (BandConfig::placement): a band acts on every channel, on the
 * left or the right one alone, or on the middle or the sides of the first
 * pair. The left, right and stereo bands run first, then the pair is turned
 * into middle and sides for the bands placed there, and back. In LinearPhase
 * the same is one matrix of four kernels between the two channels, so the
 * latency does not change. With a single channel the left and middle bands
 * act on it and the right and side bands do nothing.
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
 *   setFilterMode(), setSoftMode(), setMatchedBells(), setMatchedFilters(),
 *   setAnalogResponse(). Each band is published
 *   through its own seqlock over std::atomic words, so the audio thread adopts
 *   a whole band configuration or none of it; a change is consumed at the next
 *   block (or the next processSample()). Non-finite band fields are replaced
 *   with the band's current values. The audio thread's read of that seqlock is
 *   BOUNDED (StagedBand::tryRead): it never waits for the control thread to
 *   finish publishing, and a band whose read gives up keeps its current
 *   coefficients while the dirty flag is re-armed, so the change lands one call
 *   later instead of holding the callback open.
 * - Any thread: getBandConfig(), getMagnitudeForFrequencyArray(), getState(),
 *   getNumBands(), getFilterMode(), getSoftMode(), getLatency(),
 *   isAnalogResponse(), buildBandSections(). The first two
 *   read the same seqlock and hand back a private copy, so they never observe a
 *   half-written band. They use the UNBOUNDED StagedBand::read(): a readout has
 *   no previously adopted copy to keep, and it is not real-time.
 * - Processing thread only: getBandFilter(), which returns a reference to a
 *   band's live FilterEngine (its contract, not the equalizer's).
 *
 * Dependencies: Filters.h (FilterEngine), MidSide.h, AudioSpec.h, AudioBuffer.h,
 *               Biquad.h, BiquadCascade.h, DenormalGuard.h, DspMath.h, FFT.h,
 *               FilterDesign.h, StateBlob.h.
 */

#include "Filters.h"
#include "MidSide.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Biquad.h"
#include "../Core/BiquadCascade.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/FilterDesign.h"
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
        Tilt,       ///< Tilt EQ: pivots spectrum around frequency.
        AllPass,      ///< Turns the phase around frequency, 180 degrees per order; the level stays.
        FlatTilt,     ///< The same slope across the whole band: gain is in dB per octave (-6 to +6), 0 dB at frequency.
        LowPassWall,  ///< Elliptic cut above frequency: flat to it, 100 dB down a quarter of an octave later.
        HighPassWall  ///< Elliptic cut below frequency.
    };

    /** @brief Where in a stereo pair a band acts. */
    enum class Placement
    {
        Stereo, ///< Every channel. Default.
        Left,   ///< The first channel only.
        Right,  ///< The second channel only.
        Mid,    ///< The middle of the first two channels: what they share.
        Side    ///< Their sides: what sets them apart.
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
        T slope         = T(12);            ///< Slope in dB/oct (LP/HP only): 6 to 48 in steps of 6, or any value from 0 to 96 with the analog response.
        int order       = 1;                ///< Order of a bell, shelf, notch, band-pass or all-pass (1 to 16; 6 dB/oct each). Above 1 the band runs with the analog response.
        bool enabled    = true;             ///< False to bypass this band.
        Placement placement = Placement::Stereo;   ///< The channels the band acts on.
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

        // The engine of the analog response: one cascade per band, settled on
        // the published configuration.
        designBand_.prepare(spec.sampleRate);
        analog_ = std::make_unique<std::array<AnalogBand, MaxBands>>();
        for (int i = 0; i < MaxBands; ++i)
        {
            const BandConfig cfg = staged_[i].read();
            bandAnalog_[static_cast<size_t>(i)] = usesAnalog(cfg);
            bandPlacement_[static_cast<size_t>(i)] = static_cast<int>(cfg.placement);
            if (bandAnalog_[static_cast<size_t>(i)]) designAnalogBand(i, cfg, true);
        }

        // Rebuild the linear-phase engine. lpFft_ gates the LP audio path, so
        // it is torn down first and re-created last (basic guarantee if an
        // allocation throws mid-way).
        lpFft_.reset();
        lpDesignFft_.reset();
        lpKernelLength_ = 0;
        lpPartition_ = 0;
        lpPartitions_ = 0;
        lpChannels_ = spec.numChannels;
        lpSlots_ = lpChannels_ >= 2 ? (lpChannels_ > 2 ? 5 : 4) : 1;

        const int kernel = resolveKernelLength(spec.sampleRate);
        const int part = std::max(kLpMinPartition, kernel / kLpPartitionCount);
        const int parts = kernel / part;
        const size_t specLen = static_cast<size_t>(2 * part + 2);
        const size_t channels = static_cast<size_t>(lpChannels_);

        for (auto& set : lpKernelSet_)
            set.assign(static_cast<size_t>(lpSlots_) * static_cast<size_t>(parts) * specLen, T(0));
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
        lpMag_.assign(static_cast<size_t>(lpSlots_) * static_cast<size_t>(kernel + 1), T(1));
        lpPlaced_[0] = lpPlaced_[1] = false;
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
     * A band placed on the left or the right acts on channel 0 or 1 alone.
     * The bands of the middle and the sides need both channels of a frame
     * and only run in processBlock().
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
            if (!bandEnabled_[i].load(std::memory_order_relaxed)) continue;
            // A band of one channel keeps its memory in the first slot of its
            // engine, as in the block form. Middle and sides need both
            // channels of a frame: those bands only run in processBlock().
            const int placement = bandPlacement_[static_cast<size_t>(i)];
            if (placement >= static_cast<int>(Placement::Mid)) continue;
            if (placement != static_cast<int>(Placement::Stereo)
                && channel != (placement == static_cast<int>(Placement::Left) ? 0 : 1)) continue;
            const int slot = placement == static_cast<int>(Placement::Stereo) ? channel : 0;
            if (bandAnalog_[static_cast<size_t>(i)] && analog_)
            {
                // Ramps move once per frame, on the first channel the band plays.
                AnalogBand& band = (*analog_)[static_cast<size_t>(i)];
                if (slot == 0)
                {
                    band.cascade.advanceRamps(1);
                    settleAnalogBand(band);
                }
                if (slot >= 0 && slot < kEngineChannels && !band.idle)
                    sample = static_cast<T>(band.cascade.processSample(static_cast<double>(sample), slot));
            }
            else
            {
                sample = bands_[i].processSample(sample, slot);
            }
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
        if (analog_)
            for (auto& band : *analog_) band.cascade.reset();

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
     * Non-finite frequency/gain/Q/slope fields and a type that is not one
     * fall back to the band's current values (they would otherwise poison the
     * serialized state, the analysis curve, and the linear-phase kernel); a
     * slope or an order out of range clamps.
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
        if (!std::isfinite(cfg.slope))     cfg.slope     = prev.slope;
        if (static_cast<int>(cfg.type) < 0 || static_cast<int>(cfg.type) > static_cast<int>(BandType::HighPassWall))
            cfg.type = prev.type;
        if (static_cast<int>(cfg.placement) < 0 || static_cast<int>(cfg.placement) > static_cast<int>(Placement::Side))
            cfg.placement = prev.placement;
        cfg.slope = std::clamp(cfg.slope, T(0), T(96));
        cfg.order = std::clamp(cfg.order, 1, kMaxBandSections);

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
     * @brief Switches LowPass/HighPass, BandPass and slope-1 shelf bands to
     *        the analog-matched designs.
     *
     * The counterpart of setMatchedBells() for the other band types
     * (BiquadCoeffs::make*Matched, after Vicanek): near Nyquist they keep the
     * analog prototype's shape (48 kHz worst deviation, 20 Hz - 20 kHz: LP at
     * 15 kHz 0.58 dB vs 9.4 dB bilinear, shelves 0.40 dB vs 3.9 dB). LP/HP
     * first-order sections, shelves whose Q maps to a slope below 1, Notch
     * and Tilt stay bilinear. Applies to the IIR engines, the linear-phase
     * kernel and the analysis curve alike. Off by default (bit-compatible).
     */
    void setMatchedFilters(bool enabled) noexcept
    {
        matchedFilters_.store(enabled, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns whether non-bell bands use the matched designs (setMatchedFilters()). */
    [[nodiscard]] bool isMatchedFilters() const noexcept
    {
        return matchedFilters_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Designs every band from its analog prototype (see the file notes).
     *
     * Off by default. With it on, the curve of each band stays on its analog
     * shape up to Nyquist in every type (FilterDesign), a parameter change
     * ramps the coefficients of the band instead of stepping them, and the
     * slope of a cut is any value from 0 to 96 dB per octave. Applies to the
     * minimum-phase engine, the linear-phase kernel and the analysis curve.
     * A band switching engine fades in over 20 ms.
     */
    void setAnalogResponse(bool enabled) noexcept
    {
        analogResponse_.store(enabled, std::memory_order_relaxed);
        configDirty_.store(true, std::memory_order_release);
    }

    /** @brief Returns whether bands are designed from their analog prototypes (setAnalogResponse()). */
    [[nodiscard]] bool isAnalogResponse() const noexcept
    {
        return analogResponse_.load(std::memory_order_relaxed);
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
        magnitudeOfBands(frequencies, magnitudes, numPoints, -1);
    }

    /**
     * @brief The magnitude response at one place of the stereo pair: the
     *        stereo bands and the bands placed there.
     *
     * @param frequencies Array of frequencies in Hz.
     * @param magnitudes  Output array (same size, linear scale).
     * @param numPoints   Number of frequency points.
     * @param where       Left, Right, Mid or Side (Stereo gives the stereo bands alone).
     */
    void getMagnitudeForFrequencyArray(const T* frequencies, T* magnitudes, int numPoints, Placement where) const noexcept
    {
        magnitudeOfBands(frequencies, magnitudes, numPoints, static_cast<int>(where));
    }

protected:
    /** @brief The product of the enabled bands: all of them (where < 0), or the stereo ones and those placed at `where`. */
    void magnitudeOfBands(const T* frequencies, T* magnitudes, int numPoints, int where) const noexcept
    {
        for (int i = 0; i < numPoints; ++i)
            magnitudes[i] = T(1);

        const int activeBands = numBands_.load(std::memory_order_relaxed);
        for (int b = 0; b < activeBands; ++b)
        {
            const BandConfig cfg = staged_[b].read();
            if (!cfg.enabled) continue;
            if (where >= 0 && cfg.placement != Placement::Stereo && static_cast<int>(cfg.placement) != where) continue;

            BiquadCoeffs st[kMaxBandSections];
            const int ns = buildBandSections(cfg, st, kMaxBandSections);

            for (int i = 0; i < numPoints; ++i)
            {
                double mag = 1.0;
                for (int s = 0; s < ns; ++s)
                    mag *= st[s].getMagnitude(static_cast<double>(frequencies[i]), spec_.sampleRate);
                magnitudes[i] = static_cast<T>(static_cast<double>(magnitudes[i]) * mag);
            }
        }
    }

public:

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
        w.write("matchedFilters", matchedFilters_.load(std::memory_order_relaxed));
        w.write("filterMode", static_cast<int32_t>(filterMode_.load(std::memory_order_relaxed)));
        w.write("softMode", softMode_.load(std::memory_order_relaxed));
        w.write("analog", analogResponse_.load(std::memory_order_relaxed));
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
            w.write(key, static_cast<int32_t>(std::lround(static_cast<double>(cfg.slope))));
            std::snprintf(key, sizeof(key), "b%d.slopeDb", i);
            w.write(key, static_cast<float>(cfg.slope));
            std::snprintf(key, sizeof(key), "b%d.order", i);
            w.write(key, static_cast<int32_t>(cfg.order));
            std::snprintf(key, sizeof(key), "b%d.on", i);
            w.write(key, cfg.enabled);
            std::snprintf(key, sizeof(key), "b%d.place", i);
            w.write(key, static_cast<int32_t>(cfg.placement));
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
        setMatchedFilters(r.read("matchedFilters", false));   // older blobs: bilinear, as before
        // Older blobs carry no mode keys: keep the instance's current modes.
        setFilterMode(static_cast<FilterMode>(
            r.read("filterMode", static_cast<int32_t>(filterMode_.load(std::memory_order_relaxed)))));
        setSoftMode(r.read("softMode", softMode_.load(std::memory_order_relaxed)));
        setAnalogResponse(r.read("analog", analogResponse_.load(std::memory_order_relaxed)));
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
            const int wholeSlope = r.read(key, 12);
            std::snprintf(key, sizeof(key), "b%d.slopeDb", i);
            cfg.slope = static_cast<T>(r.read(key, static_cast<float>(wholeSlope)));   // older blobs: the whole slope
            std::snprintf(key, sizeof(key), "b%d.order", i);
            cfg.order = r.read(key, 1);
            std::snprintf(key, sizeof(key), "b%d.on", i);
            cfg.enabled = r.read(key, true);
            std::snprintf(key, sizeof(key), "b%d.place", i);
            cfg.placement = static_cast<Placement>(r.read(key, 0));
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
        const bool matchedOthers = matchedFilters_.load(std::memory_order_relaxed);
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

            // A band that moves to other channels comes in through the band fade.
            if (static_cast<int>(cfg.placement) != bandPlacement_[static_cast<size_t>(i)])
            {
                bandPlacement_[static_cast<size_t>(i)] = static_cast<int>(cfg.placement);
                bandMix_[static_cast<size_t>(i)] = T(0);
            }

            // A band of the analog response runs in its cascade; one that
            // changes engine comes in through the band fade.
            const bool analog = analog_ != nullptr && usesAnalog(cfg);
            if (analog != bandAnalog_[static_cast<size_t>(i)])
            {
                bandAnalog_[static_cast<size_t>(i)] = analog;
                bandMix_[static_cast<size_t>(i)] = T(0);
                if (analog) (*analog_)[static_cast<size_t>(i)].configured = false;
            }
            if (analog)
            {
                designAnalogBand(i, cfg, false);
                continue;
            }

            float freq = static_cast<float>(cfg.frequency);
            float gain = static_cast<float>(cfg.gain);
            float q    = static_cast<float>(effectiveQ(cfg));

            filter.setMatchedPeak(matched);
            filter.setMatchedFilters(matchedOthers);
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
                case BandType::LowPass:   filter.setLowPass(freq, q, legacySlope(cfg)); break;
                case BandType::HighPass:  filter.setHighPass(freq, q, legacySlope(cfg)); break;
                case BandType::Notch:     filter.setNotch(freq, q); break;
                case BandType::BandPass:  filter.setBandPass(freq, q); break;
                case BandType::Tilt:      filter.setTilt(freq, gain); break;
                default:                  break;      // the other types run in the analog engine
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
        const bool mf = matchedFilters_.load(std::memory_order_relaxed);

        switch (cfg.type)
        {
            case BandType::Peak:
                return matchedBells_.load(std::memory_order_relaxed)
                    ? BiquadCoeffs::makePeakMatched(sr, f, q, g)
                    : BiquadCoeffs::makePeak(sr, f, q, g);
            case BandType::LowShelf:
            case BandType::HighShelf:
            {
                const double slope = shelfSlopeFromQ(q, g);
                const bool lowShelf = cfg.type == BandType::LowShelf;
                if (mf && slope >= 1.0)
                    return lowShelf ? BiquadCoeffs::makeLowShelfMatched(sr, f, g)
                                    : BiquadCoeffs::makeHighShelfMatched(sr, f, g);
                return lowShelf ? BiquadCoeffs::makeLowShelf(sr, f, g, slope)
                                : BiquadCoeffs::makeHighShelf(sr, f, g, slope);
            }
            case BandType::LowPass:   return mf ? BiquadCoeffs::makeLowPassMatched(sr, f, q)
                                                : BiquadCoeffs::makeLowPass(sr, f, q);
            case BandType::HighPass:  return mf ? BiquadCoeffs::makeHighPassMatched(sr, f, q)
                                                : BiquadCoeffs::makeHighPass(sr, f, q);
            case BandType::Notch:     return BiquadCoeffs::makeNotch(sr, f, q);
            case BandType::BandPass:  return mf ? BiquadCoeffs::makeBandPassMatched(sr, f, q)
                                                : BiquadCoeffs::makeBandPass(sr, f, q);
            case BandType::Tilt:      return BiquadCoeffs::makeTilt(sr, f, g);
            default:                  break;      // the other types have no single section
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
            auto casc = FilterEngine<T>::cascadeForSlope(legacySlope(cfg), static_cast<float>(q));
            int n = 0;
            if (casc.hasFirstOrder)
                stages[n++] = lp ? BiquadCoeffs::makeFirstOrderLowPass(sr, f)
                                 : BiquadCoeffs::makeFirstOrderHighPass(sr, f);
            const bool mf = matchedFilters_.load(std::memory_order_relaxed);
            for (int s = 0; s < casc.numSecondOrder; ++s)
            {
                const double stageQ = static_cast<double>(casc.qValues[s]);
                if (mf)
                    stages[n++] = lp ? BiquadCoeffs::makeLowPassMatched(sr, f, stageQ)
                                     : BiquadCoeffs::makeHighPassMatched(sr, f, stageQ);
                else
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

public:
    /**
     * @brief Fills `sections` with the biquads a band runs, whichever engine
     *        runs it, and returns how many (up to 16).
     *
     * The analysis API for user interfaces: with the analog response a band
     * can have more sections than buildBandStages() has room for. Requires
     * prepare().
     *
     * @param cfg      Band configuration.
     * @param sections Receives the sections.
     * @param capacity Room in sections (16 holds any band).
     * @return Number of sections written.
     */
    [[nodiscard]] int buildBandSections(const BandConfig& cfg, BiquadCoeffs* sections, int capacity) const noexcept
    {
        if (usesAnalog(cfg)) return designSections(cfg, sections, capacity);
        BiquadCoeffs stages[5];
        const int n = std::min(buildBandStages(cfg, stages), capacity);
        for (int i = 0; i < n; ++i) sections[i] = stages[i];
        return n;
    }

protected:
    static constexpr int kMaxBandSections = AnalogCascade::kMaxSections;   ///< Most sections a band has.
    static constexpr int kEngineChannels = 16;                             ///< Channels the band engines hold.
    static constexpr double kRampSlowMs = 40.0;    ///< Glide of a small move of a band of the analog response.
    static constexpr double kRampFastMs = 10.0;    ///< Glide of a jump.
    static constexpr double kRestDb = 0.05;        ///< A gain band closer to 0 dB than this leaves the path.

    /// One band of the analog response.
    struct AnalogBand
    {
        BiquadCascade<T, kMaxBandSections, kEngineChannels> cascade;
        BandConfig config {};       ///< What the cascade is on, or gliding to.
        bool soft = false;          ///< Soft mode when it was designed (it changes the Q).
        int wanted = 0;             ///< Sections of that design (a glide can still carry more).
        bool configured = false;
        bool resting = false;       ///< The design is transparent (a gain band at 0 dB).
        bool idle = false;          ///< Resting and landed: no section is computed.
    };

    /** @brief The slope the cookbook engine runs: 6 to 48 dB per octave, whole. */
    [[nodiscard]] static int legacySlope(const BandConfig& cfg) noexcept
    {
        return static_cast<int>(std::lround(std::clamp(static_cast<double>(cfg.slope), 6.0, 48.0)));
    }

    /** @brief Whether a band runs with the analog response (the switch, or a band only it can make). */
    [[nodiscard]] bool usesAnalog(const BandConfig& cfg) const noexcept
    {
        return analogResponse_.load(std::memory_order_relaxed)
            || static_cast<int>(cfg.type) > static_cast<int>(BandType::Tilt)
            || cfg.order > 1;
    }

    /** @brief The analog prototype of a band. */
    [[nodiscard]] AnalogCascade bandPrototype(const BandConfig& cfg) const noexcept
    {
        const double f = std::clamp(static_cast<double>(cfg.frequency), 1.0, 80000.0);
        const double g = std::clamp(static_cast<double>(cfg.gain), -96.0, 96.0);
        const double q = std::max(static_cast<double>(effectiveQ(cfg)), 0.025);
        const int order = std::clamp(cfg.order, 1, kMaxBandSections);
        const double slope = static_cast<double>(cfg.slope);
        AnalogCascade one;
        switch (cfg.type)
        {
            case BandType::Peak:
                if (order > 1) return FilterDesign::bell(f, q, g, order);
                one.add(AnalogSection::peak(f, q, g));
                break;
            case BandType::LowShelf:
                if (order > 1) return FilterDesign::lowShelf(f, g, order);
                one.add(AnalogSection::lowShelf(f, g, shelfSlopeFromQ(q, g)));
                break;
            case BandType::HighShelf:
                if (order > 1) return FilterDesign::highShelf(f, g, order);
                one.add(AnalogSection::highShelf(f, g, shelfSlopeFromQ(q, g)));
                break;
            case BandType::LowPass:      return FilterDesign::highCut(f, slope, q);
            case BandType::HighPass:     return FilterDesign::lowCut(f, slope, q);
            case BandType::Notch:
                if (order > 1) return FilterDesign::bandStop(f, q, order);
                one.add(AnalogSection::notch(f, q));
                break;
            case BandType::BandPass:
                if (order > 1) return FilterDesign::bandPass(f, q, order);
                one.add(AnalogSection::bandPass(f, q));
                break;
            case BandType::Tilt:         one.add(AnalogSection::tilt(f, g)); break;
            case BandType::AllPass:      return FilterDesign::allPass(f, q, order);
            case BandType::FlatTilt:     return FilterDesign::tilt(f, g);
            case BandType::LowPassWall:  return FilterDesign::ellipticHighCut(f);
            case BandType::HighPassWall: return FilterDesign::ellipticLowCut(f);
        }
        return one;
    }

    /**
     * @brief The biquads of a band of the analog response.
     *
     * Every type follows its prototype (FilterDesign::digital()) except the
     * walls: an elliptic cut through the bilinear transform, with its corner
     * held in place, is an elliptic filter again, with its ripple and its
     * floor exact, and only the wall a little steeper.
     */
    [[nodiscard]] int designSections(const BandConfig& cfg, BiquadCoeffs* sections, int capacity) const noexcept
    {
        const AnalogCascade proto = bandPrototype(cfg);
        if (cfg.type != BandType::LowPassWall && cfg.type != BandType::HighPassWall)
            return FilterDesign::digital(proto, designBand_, sections, capacity);
        const int n = std::min(proto.count, capacity);
        const double corner = std::clamp(static_cast<double>(cfg.frequency), 1.0, 80000.0);
        for (int i = 0; i < n; ++i)
            sections[i] = FilterDesign::bilinear(proto.sections[static_cast<size_t>(i)], spec_.sampleRate, corner);
        return n;
    }

    /** @brief Whether a band does nothing as configured: a gain band within kRestDb of 0 dB. */
    [[nodiscard]] static bool restsAtUnity(const BandConfig& cfg) noexcept
    {
        const double gain = std::abs(static_cast<double>(cfg.gain));
        switch (cfg.type)
        {
            case BandType::Peak:
            case BandType::LowShelf:
            case BandType::HighShelf:
            case BandType::Tilt:      return gain < kRestDb;
            case BandType::FlatTilt:  return gain < 0.1 * kRestDb;
            default:                  return false;
        }
    }

    /**
     * @brief How long the coefficients of a band take to reach a new design:
     *        long for a small move, short for a jump.
     */
    [[nodiscard]] int rampSamples(const BandConfig& from, const BandConfig& to) const noexcept
    {
        double distance = 1.0;
        if (from.type == to.type && from.order == to.order)
        {
            const double f0 = std::max(static_cast<double>(from.frequency), 1.0), f1 = std::max(static_cast<double>(to.frequency), 1.0);
            const double q0 = std::max(static_cast<double>(from.q), 0.025), q1 = std::max(static_cast<double>(to.q), 0.025);
            distance = std::max({ std::abs(static_cast<double>(to.gain) - static_cast<double>(from.gain)) / 24.0,
                                  std::abs(std::log2(f1 / f0)) / 4.0,
                                  std::abs(std::log2(q1 / q0)) / 4.0,
                                  std::abs(static_cast<double>(to.slope) - static_cast<double>(from.slope)) / 48.0 });
        }
        const double closeness = std::max(0.0, 1.0 - distance);
        const double ms = kRampFastMs + (kRampSlowMs - kRampFastMs) * closeness * closeness;
        return std::max(1, static_cast<int>(ms * 0.001 * spec_.sampleRate));
    }

    /**
     * @brief Designs a band of the analog response and starts its cascade
     *        towards the design (audio thread, or prepare()).
     *
     * Sections the design adds come in transparent, with their poles, and
     * open; sections it drops close and leave when the glide lands; a band
     * that was resting comes back from the history of its input.
     *
     * @param index Band index.
     * @param cfg   Its configuration.
     * @param snap  True to land at once (prepare(), or a band with no design yet).
     */
    void designAnalogBand(int index, const BandConfig& cfg, bool snap) noexcept
    {
        AnalogBand& band = (*analog_)[static_cast<size_t>(index)];
        const bool soft = softMode_.load(std::memory_order_relaxed);
        if (!snap && band.configured && band.soft == soft && sameDesign(band.config, cfg)) return;
        band.soft = soft;
        BiquadCoeffs target[kMaxBandSections];
        const int n = designSections(cfg, target, kMaxBandSections);
        const bool rest = restsAtUnity(cfg);
        auto& cascade = band.cascade;
        const bool fresh = snap || !band.configured;
        const int had = band.configured ? cascade.getNumSections() : 0;
        const int count = std::max(had, n);
        const int ramp = fresh ? 0 : rampSamples(band.config, cfg);

        if (!band.configured) cascade.setNumSections(0);
        for (int s = had; s < count; ++s)
        {
            cascade.setSectionEnabled(s, true);
            cascade.setCoeffs(s, target[s].bypassed());
        }
        cascade.setNumSections(count);
        if (band.idle)
            for (int s = 0; s < count; ++s) cascade.setSectionEnabled(s, true);
        for (int s = 0; s < count; ++s)
        {
            if (s >= n) cascade.rampToBypass(s, ramp);
            else cascade.rampCoeffs(s, rest ? target[s].bypassed() : target[s], ramp);
        }
        band.config = cfg;
        band.wanted = n;
        band.resting = rest;
        band.idle = false;
        band.configured = true;
        settleAnalogBand(band);
    }

    /** @brief Whether two configurations give the same sections (the enabled flag is not part of the design). */
    [[nodiscard]] static bool sameDesign(const BandConfig& a, const BandConfig& b) noexcept
    {
        return a.frequency == b.frequency && a.gain == b.gain && a.q == b.q && a.type == b.type
            && a.slope == b.slope && a.order == b.order;
    }

    /** @brief Once a glide has landed: dropped sections leave, and a resting band stops computing. */
    void settleAnalogBand(AnalogBand& band) noexcept
    {
        auto& cascade = band.cascade;
        if (cascade.isRamping()) return;
        if (cascade.getNumSections() > band.wanted) cascade.setNumSections(band.wanted);
        if (band.resting && !band.idle)
        {
            for (int s = 0; s < band.wanted; ++s) cascade.setSectionEnabled(s, false);
            band.idle = true;
        }
    }

    /** @brief One block through a band, in the engine that runs it. */
    void runBand(int index, AudioBufferView<T> view) noexcept
    {
        if (bandAnalog_[static_cast<size_t>(index)] && analog_)
        {
            AnalogBand& band = (*analog_)[static_cast<size_t>(index)];
            if (!band.idle) band.cascade.processBlock(view);
            settleAnalogBand(band);
            return;
        }
        bands_[index].processBlock(view);
    }

    void resetBand(int index) noexcept
    {
        if (bandAnalog_[static_cast<size_t>(index)] && analog_)
            (*analog_)[static_cast<size_t>(index)].cascade.reset();
        else
            bands_[index].reset();
    }

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
        const bool pair = buffer.getNumChannels() >= 2;
        const int firstOfThePair = static_cast<int>(Placement::Mid);

        // Whether a band of the middle or the sides plays, or is still fading out.
        bool second = false;
        for (int i = 0; i < MaxBands; ++i)
            if (bandPlacement_[static_cast<size_t>(i)] >= firstOfThePair
                && ((i < activeBands && bandEnabled_[i].load(std::memory_order_relaxed))
                    || bandMix_[static_cast<size_t>(i)] != T(0)))
                second = true;

        for (int offset = 0; offset < total; offset += slice)
        {
            const int n = std::min(slice, total - offset);
            AudioBufferView<T> view = buffer.getSubView(offset, n);
            for (int i = 0; i < MaxBands; ++i)
                if (bandPlacement_[static_cast<size_t>(i)] < firstOfThePair) processBandSlice(i, view, n, activeBands);
            if (second)
            {
                if (pair) MidSide<T>::encode(view);
                for (int i = 0; i < MaxBands; ++i)
                    if (bandPlacement_[static_cast<size_t>(i)] >= firstOfThePair) processBandSlice(i, view, n, activeBands);
                if (pair) MidSide<T>::decode(view);
            }
            bandSnap_ = false;
        }
    }

    /**
     * @brief One band over one slice, on the channels of its placement, with
     *        its on/off fade. For a band of the middle or the sides the view
     *        is already in that form.
     */
    void processBandSlice(int i, AudioBufferView<T> view, int n, int activeBands) noexcept
    {
        const bool on = i < activeBands && bandEnabled_[i].load(std::memory_order_relaxed);
        const T target = on ? T(1) : T(0);
        T& mix = bandMix_[static_cast<size_t>(i)];
        if (bandSnap_ || bandDry_.empty()) mix = target;

        T* single[1] = { nullptr };
        AudioBufferView<T> where = view;
        const int placement = bandPlacement_[static_cast<size_t>(i)];
        if (placement != static_cast<int>(Placement::Stereo))
        {
            const int channel = (placement == static_cast<int>(Placement::Left)
                                 || placement == static_cast<int>(Placement::Mid)) ? 0 : 1;
            if (channel >= view.getNumChannels())
            {
                mix = target;      // nothing to act on: a single channel has no right and no sides
                return;
            }
            single[0] = view.getChannel(channel);
            where = AudioBufferView<T>(single, 1, n);
        }

        if (mix == target)
        {
            if (on) runBand(i, where);
            return;
        }

        if (mix == T(0)) resetBand(i);   // re-entering band: clean state

        const int nCh = std::min(where.getNumChannels(), bandDryChannels_);
        for (int ch = 0; ch < nCh; ++ch)
            std::copy_n(where.getChannel(ch), n,
                        bandDry_.data() + static_cast<size_t>(ch) * static_cast<size_t>(bandDryStride_));
        runBand(i, where);
        const T start = mix;
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* wet = where.getChannel(ch);
            const T* dry = bandDry_.data() + static_cast<size_t>(ch) * static_cast<size_t>(bandDryStride_);
            for (int k = 0; k < n; ++k)
            {
                const T m = moveTowards(start, target, bandFadeStep_ * static_cast<T>(k + 1));
                wet[k] = dry[k] + (wet[k] - dry[k]) * m;
            }
        }
        mix = moveTowards(start, target, bandFadeStep_ * static_cast<T>(n));
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
                    lpStageCount_[static_cast<size_t>(b)] = 0;
                    lpAnalogCount_[static_cast<size_t>(b)] = 0;
                    lpBandGroup_[static_cast<size_t>(b)] = 0;
                    if (!cfg.enabled) continue;
                    if (lpSlots_ > 1)
                    {
                        lpBandGroup_[static_cast<size_t>(b)] = static_cast<int>(cfg.placement);
                    }
                    else if (cfg.placement == Placement::Right || cfg.placement == Placement::Side)
                    {
                        continue;       // a single channel has no right and no sides
                    }
                    if (usesAnalog(cfg))
                    {
                        // The magnitude of the prototype itself: nothing to shrink.
                        const AnalogCascade proto = bandPrototype(cfg);
                        lpAnalogCount_[static_cast<size_t>(b)] = proto.count;
                        for (int s = 0; s < proto.count; ++s)
                            lpAnalog_[static_cast<size_t>(b * kMaxBandSections + s)] = proto.sections[static_cast<size_t>(s)];
                    }
                    else
                    {
                        lpStageCount_[static_cast<size_t>(b)] = buildBandStages(cfg, lpStages_.data() + static_cast<size_t>(b) * 5);
                    }
                }
                lpDesignBands_ = activeBands;
                lpDesignStages_ = 0;
                for (int b = 0; b < activeBands; ++b)
                    lpDesignStages_ += lpStageCount_[static_cast<size_t>(b)] + lpAnalogCount_[static_cast<size_t>(b)];
                lpDesignPlaced_ = false;
                for (int b = 0; b < activeBands; ++b)
                    if (lpBandGroup_[static_cast<size_t>(b)] != 0
                        && lpStageCount_[static_cast<size_t>(b)] + lpAnalogCount_[static_cast<size_t>(b)] > 0)
                        lpDesignPlaced_ = true;
                lpSlot_ = 0;
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
                    // One product per place of the pair (all in the first when no band is placed).
                    double powers[5] = { 1.0, 1.0, 1.0, 1.0, 1.0 };
                    for (int b = 0; b < lpDesignBands_; ++b)
                    {
                        double& power = powers[lpDesignPlaced_ ? lpBandGroup_[static_cast<size_t>(b)] : 0];
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
                        const AnalogSection* proto = lpAnalog_.data() + static_cast<size_t>(b * kMaxBandSections);
                        const double omega = dw * static_cast<double>(k) * spec_.sampleRate;
                        for (int s = 0; s < lpAnalogCount_[static_cast<size_t>(b)]; ++s)
                            power *= proto[s].magnitudeSquared(omega);
                    }
                    if (!lpDesignPlaced_)
                    {
                        lpMag_[static_cast<size_t>(k)] = static_cast<T>(std::sqrt(powers[0]));
                    }
                    else
                    {
                        // Left and right bands, then middle and sides, then back, as one matrix:
                        // each output is its own channel times one response plus the other times another.
                        const double both = std::sqrt(powers[0]), left = std::sqrt(powers[1]), right = std::sqrt(powers[2]);
                        const double sum = 0.5 * (std::sqrt(powers[3]) + std::sqrt(powers[4]));
                        const double difference = 0.5 * (std::sqrt(powers[3]) - std::sqrt(powers[4]));
                        const size_t stride = static_cast<size_t>(kernel + 1), at = static_cast<size_t>(k);
                        lpMag_[at] = static_cast<T>(both * left * sum);                          // left to left
                        lpMag_[stride + at] = static_cast<T>(both * right * sum);                // right to right
                        lpMag_[2 * stride + at] = static_cast<T>(both * right * difference);     // right to left
                        lpMag_[3 * stride + at] = static_cast<T>(both * left * difference);      // left to right
                        if (lpSlots_ > 4) lpMag_[4 * stride + at] = static_cast<T>(both);        // further channels
                    }
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
                const T* response = lpMag_.data() + static_cast<size_t>(lpSlot_) * static_cast<size_t>(kernel + 1);
                for (int k = 0; k <= kernel; ++k)
                    lpDesignSpec_[static_cast<size_t>(2 * k)] = response[static_cast<size_t>(k)];
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
                    lpFft_->forward(lpBlockTime_.data(),
                                    set.data() + static_cast<size_t>(lpSlot_ * lpPartitions_ + p) * specLen);
                }
                if (last >= lpPartitions_ && lpDesignPlaced_ && lpSlot_ + 1 < lpSlots_)
                {
                    // The next kernel of the matrix.
                    ++lpSlot_;
                    lpStage_ = LpStage::Transform;
                }
                else if (last >= lpPartitions_)
                {
                    lpPlaced_[static_cast<size_t>(1 - lpActive_)] = lpDesignPlaced_;
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

        // Every channel's block into its delay line first: a matrix of
        // kernels reads the other channel's too.
        const size_t lineLen = static_cast<size_t>(parts) * specLen;
        for (int ch = 0; ch < lpChannels_; ++ch)
            lpFft_->forward(lpInput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(2 * part),
                            lpFdl_.data() + static_cast<size_t>(ch) * lineLen + static_cast<size_t>(lpFdlPos_) * specLen);

        for (int ch = 0; ch < lpChannels_; ++ch)
        {
            T* input = lpInput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(2 * part);
            T* output = lpOutput_.data() + static_cast<size_t>(ch) * static_cast<size_t>(part);

            auto convolve = [&](int setIndex, T* time) noexcept {
                std::fill(lpAccum_.begin(), lpAccum_.end(), T(0));
                const bool placed = lpPlaced_[static_cast<size_t>(setIndex)];
                const T* kernels = lpKernelSet_[static_cast<size_t>(setIndex)].data();
                const auto add = [&](int slot, int from) noexcept {
                    const T* kernel = kernels + static_cast<size_t>(slot) * lineLen;
                    const T* line = lpFdl_.data() + static_cast<size_t>(from) * lineLen;
                    for (int p = 0; p < parts; ++p)
                    {
                        const int at = (lpFdlPos_ - p + parts) % parts;
                        simd::complexMulAccum(lpAccum_.data(), line + static_cast<size_t>(at) * specLen,
                                              kernel + static_cast<size_t>(p) * specLen, bins);
                    }
                };
                // Its own channel through its own kernel; with placed bands, the
                // other channel of the pair through the kernel that crosses.
                add(!placed ? 0 : (ch == 0 ? 0 : (ch == 1 ? 1 : 4)), ch);
                if (placed && ch < 2) add(ch == 0 ? 2 : 3, 1 - ch);
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
        std::atomic<T>    slope     { BandConfig{}.slope };
        std::atomic<int>  order     { BandConfig{}.order };
        std::atomic<bool> enabled   { BandConfig{}.enabled };
        std::atomic<int>  placement { static_cast<int>(BandConfig{}.placement) };
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
            order.store(c.order, std::memory_order_relaxed);
            enabled.store(c.enabled, std::memory_order_relaxed);
            placement.store(static_cast<int>(c.placement), std::memory_order_relaxed);
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
            c.order     = order.load(std::memory_order_relaxed);
            c.enabled   = enabled.load(std::memory_order_relaxed);
            c.placement = static_cast<Placement>(placement.load(std::memory_order_relaxed));
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
    std::atomic<bool> matchedFilters_ { false }; ///< Matched LP/HP/BP/shelves: opt-in.
    std::atomic<bool> analogResponse_ { false }; ///< Every band from its analog prototype: opt-in.

    // The engine of the analog response (allocated by prepare(), run by the audio thread).
    std::unique_ptr<std::array<AnalogBand, MaxBands>> analog_;
    FilterDesign::Band designBand_;
    std::array<bool, MaxBands> bandAnalog_ {};   ///< Which engine runs each band.
    std::array<int, MaxBands> bandPlacement_ {}; ///< Where each band acts (audio thread's copy).

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
    std::array<AnalogSection, MaxBands * AnalogCascade::kMaxSections> lpAnalog_ {};   ///< Prototypes of the analog bands.
    std::array<int, MaxBands> lpAnalogCount_ {};
    std::array<int, MaxBands> lpBandGroup_ {};   ///< Where each band of the design acts (0 when it acts on every channel).
    int lpSlots_ = 1;                            ///< Kernels a set holds: 1, or the matrix of a pair (and one more for further channels).
    int lpSlot_ = 0;                             ///< Kernel of the design being transformed.
    bool lpDesignPlaced_ = false;                ///< The design in flight is a matrix.
    std::array<bool, 2> lpPlaced_ {};            ///< Whether each kernel set is one.
    std::vector<T> lpWindow_, lpMag_;
    std::vector<T> lpDesignSpec_, lpDesignTime_, lpKernelTime_;
};

} // namespace dspark
