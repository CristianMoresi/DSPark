// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file CrossoverFilter.h
 * @brief Linkwitz-Riley crossover filter for multi-band audio processing.
 *
 * Splits an audio signal into 2-12 frequency bands using Linkwitz-Riley
 * crossover filters. Supports five slopes (6 dB per octave, LR12, LR24, LR36
 * and LR48) and two processing modes: minimum-phase IIR with allpass phase correction,
 * and linear-phase FFT-based processing with zero phase distortion.
 * Its processBlock(input, bandOutputs, count) writes separate caller-owned
 * band buffers, rather than replacing input with one band. Non-finite input
 * samples enter the filters as silence (they would latch in the IIR state). It therefore
 * needs explicit routing and is not an in-place ProcessorChain slot. Sum
 * getLatency() once for the split, not once for each parallel band.
 *
 * Linkwitz-Riley crossovers guarantee that all bands sum to an allpass
 * (flat magnitude response) at every frequency, making them the standard
 * for professional multi-band processing. In the IIR tree, band b passes
 * through one allpass section per split above it (LR12/LR24) or two
 * sections (LR48) so the phase of every branch matches at the summing
 * point. LR12 additionally bakes a polarity inversion into the high-pass
 * branch of each split (the textbook LR2 convention): odd-numbered bands
 * come out polarity-inverted so the tree sums flat instead of notching
 * at every crossover. LR36 (two third-order Butterworth filters per branch)
 * does the same, with a first-order and a second-order allpass section per
 * split above. The 6 dB per octave split is a first-order low-pass and its
 * complement: the two branches add up to the input itself, so its bands sum
 * to the input (not to an allpass of it) and need no phase correction. The
 * linear-phase path needs neither (zero-phase magnitudes sum to unity by
 * construction).
 *
 * The linear-phase FIR has a fixed length of about 85 ms (4096 taps at 44.1
 * and 48 kHz, scaling with the rate), whatever the host block size. Its
 * latency is fixed: three quarters of the FIR, 3072 samples at 48 kHz. The
 * FIR runs as a uniformly partitioned convolution in eight partitions of an
 * eighth of its length (512 taps at 48 kHz), with one shared input transform
 * per channel for all bands. The engine delays its output by two partitions,
 * so the work for an output partition can start one partition before it is
 * heard: it is spread evenly over the samples of that partition, and no
 * host block pays for a whole partition (after W. G. Gardner, "Efficient
 * Convolution without Input-Output Delay", JAES 43(3), 1995). The last band
 * is the delayed input minus the other bands, so it costs no convolution. A
 * kernel partition whose taps sum below 1e-6 is not convolved (it moves its
 * band by less than -120 dB at any frequency): away from the centre most
 * partitions hold only the design's rounding noise, so at the default
 * splits two to four of the eight remain per band. The kernels are the ideal
 * Linkwitz-Riley magnitudes with a flat-top window (cosine tapers on the
 * outer quarters), which keeps the passband and slopes exact: band 0 of a
 * 200 Hz LR24 split at 48 kHz measures -0.527 dB at 100 Hz and -24.61 dB at
 * 400 Hz (ideal -0.527 / -24.61); LR24 stays within 0.03 dB of ideal down to
 * a 50 Hz split, LR48 down to 100 Hz. The band magnitudes telescope like the
 * IIR tree (each band carries the high-pass of every split below it), so the
 * bands sum to the delayed input exactly.
 *
 * A frequency, order or band-count change is designed on the audio thread,
 * from scratch allocated in prepare(), over the blocks that follow: each
 * band's kernel goes into a second kernel set, and the set is swapped in
 * whole, so the bands always come from one design and keep summing to the
 * delayed input (until a band-count change lands, the playing design's
 * bands are kept, missing ones silent and extra ones folded into the last
 * requested band). The first partition on a new design also runs the
 * previous one, and each band goes from the previous design to the new
 * along it (a linear ramp of one partition, 10.7 ms at 48 kHz; a band the
 * previous design did not have rises from silence): no band steps from one
 * design to the other. With a 1 kHz tone and a split moved from 500 Hz to
 * 2 kHz, band 0's largest step between samples is the one of its new steady
 * state (0.061 at 0.5 of amplitude; a swap in one sample jumped 0.380). That
 * partition does the convolution twice: with a split moving every 200 ms,
 * 12 bands in stereo at 64-sample blocks cost 4 % more on average and no
 * more in the worst block. The design runs in small steps (the long inverse
 * transform is split into its passes) at a fixed share of the stereo steady
 * work per sample, so the time a change takes to land depends neither on the
 * host block size nor on the channel count. A change that arrives during a
 * design lets it finish, then the next design starts from the latest
 * targets, so continuous automation keeps landing. Measured at 48 kHz: the
 * new response is heard 85 ms after a change at 64 and 512-sample blocks,
 * in mono, stereo or six channels alike, and the worst block during a
 * redesign is about 2x the worst steady block in stereo (the design costs
 * the same in mono, where the steady work is half).
 *
 * CPU, 48 kHz stereo, default splits, LR24: with 12 bands, 2.4 us per
 * 64-sample block on average and under 7 us in the worst block (the
 * previous engine of four 1024-tap partitions averaged 2.9 us and spent
 * 40-90 us in the block that completed a partition); 18.7 us per 512-sample
 * block, every block alike. The cost follows the kept partitions: low
 * splits keep all eight (about 1.3x the default work). prepare() runs the
 * first design whole.
 *
 * The minimum-phase engine arranges its splits in a cascade (default) or a
 * balanced tree (setTopology()), which sums to the same allpass with fewer
 * phase corrections: 48 kHz stereo, default splits, 6 bands 38.7 against
 * 31.9 ns per frame at LR24 (54.6 against 46.6 at LR48), 12 bands 139.8
 * against 84.5 (186.5 against 119.5).
 *
 * Threading model:
 * - prepare() / reset() / getState() / setState(): setup or UI threads only
 *   (prepare and getState allocate), never concurrently with processBlock().
 * - setNumBands / setNumBandsKeepingSplits / setCrossoverFrequency / setOrder /
 *   setFilterMode / setStructure / setTopology: RT-safe
 *   atomic publications from any thread; the audio thread applies them at
 *   the next block. Frequency changes are smoothed (5 ms) in IIR mode, with
 *   the coefficients refreshed on a 32-sample grid that runs across blocks
 *   (the glide does not depend on the host block size), and designed over
 *   the following blocks in linear-phase mode (see above).
 * - getLatency() and the getters are safe from any thread.
 *
 * Dependencies: Biquad.h, BiquadCascade.h, NormalizedLadder.h, AudioBuffer.h,
 *               AudioSpec.h, DspMath.h, FFT.h, SimdOps.h, DenormalGuard.h,
 *               SmoothedValue.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/BiquadCascade.h"
#include "../Core/DspMath.h"
#include "../Core/DenormalGuard.h"
#include "../Core/FFT.h"
#include "../Core/NormalizedLadder.h"
#include "../Core/SimdOps.h"
#include "../Core/SmoothedValue.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class CrossoverFilter
 * @brief Linkwitz-Riley crossover with 2-12 bands: 6, 12, 24, 36 or 48 dB per octave.
 *
 * @tparam T      Sample type (float or double).
 * @tparam MaxBands Maximum number of output bands (compile-time, default 12).
 */
template <FloatType T, int MaxBands = 12>
class CrossoverFilter
{
public:
    /** @brief Filter processing mode. */
    enum class FilterMode
    {
        MinimumPhase, ///< IIR biquads with allpass phase correction (zero latency).
        LinearPhase   ///< FFT-based FIR crossover (introduces latency, zero phase distortion, introduces pre-ringing).
    };

    CrossoverFilter()
    {
        initDefaultFrequencies();
    }

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the crossover for processing.
     *
     * Configured split frequencies, band count, order and mode are preserved
     * across prepare() calls (configure-then-prepare is fully supported).
     * The linear-phase engine is always allocated (so a later mode switch
     * works); its size depends on the sample rate and channel count only.
     *
     * @param spec Audio environment (sample rate, block size, channels).
     *             Invalid specs are ignored (previous state is kept).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false; // basic guarantee while (re)allocating
        spec_ = spec;
        // Per-split filters are Biquad<T, 16>; clamp so processing more than 16
        // channels can't index their fixed per-channel state out of bounds.
        numChannels_ = std::min(static_cast<int>(spec.numChannels), 16);

        // Allocate flat IIR work buffer (Channels * MaxBlockSize)
        workBuf_.assign(static_cast<size_t>(spec.numChannels) * static_cast<size_t>(spec.maxBlockSize), T(0));

        // Linear-phase resources. Allocated regardless of the current mode so
        // setFilterMode(LinearPhase) after prepare() actually engages. The
        // FIR length follows the sample rate (about 85 ms), never the host
        // block. lpFft_ acts as the gate and is created last: if any
        // allocation throws, the engine stays unavailable but coherent.
        lpFft_.reset();
        lpDesignFft_.reset();
        firLength_ = 0;
        lpPart_ = 0;
        lpLatency_ = 0;
        {
            int firLen = 256;
            const double wanted = spec.sampleRate * kLpFirSeconds;
            while (firLen < wanted && firLen < kLpMaxFirLength) firLen <<= 1;
            firLength_ = firLen;
            lpPart_ = firLen / kLpPartitions;
            // The kernel is centred at firLength/2; the engine delays its
            // output by kLpLeadBlocks partitions (the scheduling slack).
            lpLatency_ = firLength_ / 2 + kLpLeadBlocks * lpPart_;

            const size_t specLen = static_cast<size_t>(2 * lpPart_ + 2);
            const size_t chans = static_cast<size_t>(numChannels_);
            const size_t designBins = static_cast<size_t>(firLength_ + 1); // 2L-point design grid

            // Two kernel sets: one plays while the next design is built. The
            // last band of a set is the residual and has no kernel.
            lpKernelSpec_.assign(2 * static_cast<size_t>(kMaxSplits * kLpPartitions) * specLen, T(0));
            lpPartOn_.assign(2 * static_cast<size_t>(kMaxSplits * kLpPartitions), uint8_t(0));
            lpFdl_.assign(chans * static_cast<size_t>(kLpPartitions) * specLen, T(0));
            lpRing_.assign(chans * static_cast<size_t>(kLpRingBlocks * lpPart_), T(0));
            lpOut_.assign(chans * 2 * static_cast<size_t>(MaxBands * lpPart_), T(0));
            lpAccum_.assign(specLen, T(0));
            lpResult_.assign(static_cast<size_t>(2 * lpPart_), T(0));

            // Cost model of the steps (units: one complex multiply-accumulate
            // bin), used to spread a partition's work and a design's steps.
            const double part = static_cast<double>(lpPart_);
            lpCostFft_ = (kLpFftUnitsPerPoint + kLpCopyUnitsPerSample) * 2.0 * part;
            lpCostCma_ = part + 1.0;
            lpCostIfft_ = kLpFftUnitsPerPoint * 2.0 * part + kLpCopyUnitsPerSample * part;
            lpCostResidualPerBand_ = kLpResidualUnitsPerSample * part;
            lpCostMagChunk_ = kLpMagUnitsPerBin * static_cast<double>((designBins + kLpMagChunks - 1) / kLpMagChunks);
            lpCostPartitionStep_ = lpCostIfft_ + kLpCopyUnitsPerSample * part;

            // Design scratch, so a live crossover/order change never allocates
            // on the audio thread (the design steps run there).
            lpCarry_.assign(designBins, 1.0);
            lpBinFreq_.assign(designBins, 0.0);
            lpMag_.assign(designBins, T(0));
            for (size_t k = 0; k < designBins; ++k)
                lpBinFreq_[k] = spec.sampleRate * static_cast<double>(k) / static_cast<double>(2 * firLength_);
            // Flat-top window: a cosine taper over the outer quarter on each side.
            lpWindow_.assign(static_cast<size_t>(firLength_), T(1));
            const int taper = firLength_ / 4;
            for (int i = 0; i < firLength_; ++i)
            {
                const int edge = std::min(i, firLength_ - 1 - i);
                if (edge < taper)
                    lpWindow_[static_cast<size_t>(i)] = static_cast<T>(
                        0.5 - 0.5 * std::cos(std::numbers::pi * (static_cast<double>(edge) + 0.5)
                                             / static_cast<double>(taper)));
            }
            lpTimeResponse_.assign(static_cast<size_t>(2 * firLength_ + 2), T(0));
            lpFirKernel_.assign(static_cast<size_t>(2 * firLength_), T(0));
            lpScratch_.assign(static_cast<size_t>(2 * lpPart_), T(0));

            lpDesignFft_ = std::make_unique<FFTReal<T>>(2 * firLength_);
            lpDesignFftSteps_ = lpDesignFft_->getInverseStepCount();
            lpCostDesignFftStep_ = kLpDesignFftUnitsPerPoint * 2.0 * static_cast<double>(firLength_)
                                   / static_cast<double>(lpDesignFftSteps_);
            lpFft_ = std::make_unique<FFTReal<T>>(2 * lpPart_);
        }

        // The two banks of sections: the correction chains [band (or tree
        // node) * kMaxSplits + split], and the ladders of setStructure(Ladder),
        // allocated whatever the current structure and topology so a later
        // switch never allocates.
        for (auto& k : banks_)
        {
            if (!k) k = std::make_unique<IirBank>();
            k->allpass.resize(static_cast<size_t>(MaxBands * kMaxSplits));
            k->ladders.assign(static_cast<size_t>(kLadderCount), NormalizedLadder<T, 16> {});
        }
        bank_ = 0;
        fadeBank_ = -1;
        ladderTmp_.assign(static_cast<size_t>(spec.maxBlockSize), 0.0);
        ladderActive_ = false;

        // A change of the band count in the minimum-phase engine: both
        // topologies run for kFadeSeconds and the bands go from one to the
        // other, a chunk at a time through this scratch.
        fadeLength_ = std::max(1, static_cast<int>(std::lround(kFadeSeconds * spec.sampleRate)));
        fadeBands_.assign(static_cast<size_t>(MaxBands * numChannels_ * kFadeChunk), T(0));
        fadeLower_.assign(static_cast<size_t>(MaxBands * numChannels_ * kFadeChunk), T(0));
        fadeTop_.assign(static_cast<size_t>(numChannels_ * kFadeChunk), T(0));
        lpFadeOld_.assign(static_cast<size_t>(MaxBands) * static_cast<size_t>(lpPart_), T(0));

        // Re-sync smoothers to the CURRENT targets. prepare() used to reset
        // all split frequencies to the log-spaced defaults, silently discarding
        // any configuration made before prepare().
        for (int i = 0; i < kMaxSplits; ++i)
        {
            freqSmoothers_[i].prepare(spec.sampleRate, 5.0);
            freqSmoothers_[i].setSmoothingType(SmoothedValue<T>::SmoothingType::Exponential);
            T target = targetFrequencies_[i].load(std::memory_order_relaxed);
            freqSmoothers_[i].reset(target);
            frequencies_[i] = target;
        }

        lastMode_ = filterMode_.load(std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
        // The first design is built whole here (setup thread); later changes
        // are spread over blocks by the audio thread.
        lpActiveSet_ = 0;
        lpSetBands_ = { 0, 0 };
        lpSetParts_ = { 0, 0 };
        lpDesignActive_ = false;
        lpDesignRedo_ = false;
        if (lpFft_ != nullptr) designLinearPhaseNow();
        reset();

        prepared_ = true;
    }

    /**
     * @brief Splits input into separate band outputs.
     *
     * The processed span is clamped to the prepared maxBlockSize and to the
     * shortest output view. The linear-phase engine buffers internally, so
     * any block size up to maxBlockSize gives the same output. Input channels beyond the prepared/16-channel
     * limit are passed through on band 0 (remaining bands silent) so the
     * band sum still reconstructs the input.
     *
     * @param input        Input audio buffer.
     * @param bandOutputs  Array of AudioBufferView, one per band.
     * @param numOutputBands Number of output bands (must match getNumBands();
     *                       fewer bands re-purpose the first splits only in
     *                       IIR mode and is not a supported configuration).
     * @return The number of bands actually written THIS block (0 when nothing
     *         was written: unprepared, degenerate span, or null outputs).
     *         Callers that sum the bands back together should trust this value
     *         rather than a separately read getNumBands(): the band count is
     *         an atomic that a concurrent setNumBands() may move between the
     *         two reads, and views beyond the returned count are untouched.
     */
    int processBlock(AudioBufferView<T> input,
                     AudioBufferView<T>* bandOutputs, int numOutputBands) noexcept
    {
        if (!prepared_ || bandOutputs == nullptr) return 0;

        const FilterMode mode = filterMode_.load(std::memory_order_relaxed);
        const bool lpActive = (mode == FilterMode::LinearPhase) && lpFft_ != nullptr;
        if (mode != lastMode_)
        {
            // Live mode switch: clear filter/overlap state so the incoming
            // engine does not replay stale history (audible transient is
            // documented behaviour of an engine switch).
            lastMode_ = mode;
            reset();
            if (lpActive) syncFrequenciesToTargets();
        }

        // Check if UI requested a frequency update
        if (freqUpdatePending_.exchange(false, std::memory_order_acquire))
        {
            if (lpActive)
            {
                // FIR kernels are rebuilt per change (per-sample smoothing
                // cannot apply to a kernel), from the targets synced here.
                syncFrequenciesToTargets();
                lpMagDirty_.store(true, std::memory_order_relaxed);
            }
            else
            {
                for (int i = 0; i < kMaxSplits; ++i)
                    freqSmoothers_[i].setTargetValue(targetFrequencies_[i].load(std::memory_order_relaxed));
            }
        }

        // A change of structure restarts the memory of the sections; the
        // ladders take the coefficients in use.
        const bool ladder = structure_.load(std::memory_order_relaxed) == Structure::Ladder && !bank(0).ladders.empty();
        if (!lpActive && ladder != ladderActive_)
        {
            ladderActive_ = ladder;
            resetBank(bank(0));
            resetBank(bank(1));
            if (ladder) updateCoefficients();
        }

        // The IIR coefficients are only rebuilt while the IIR engine runs; in
        // linear-phase mode the flag stays set for a later switch back.
        if (!lpActive && dirty_.load(std::memory_order_relaxed) &&
            dirty_.exchange(false, std::memory_order_acquire))
        {
            updateCoefficients();
        }

        const int n = std::min(numOutputBands, numBands_.load(std::memory_order_relaxed));
        if (n < 2) return 0;

        // Defensive span clamp: never index past the prepared work buffers
        // (fixed maxBlockSize stride) or past any output view. The channel
        // count is clamped to the narrowest band view too: a view with fewer
        // channels used to be written through a null channel pointer.
        int nS = std::min(input.getNumSamples(), spec_.maxBlockSize);
        int nCh = std::min(input.getNumChannels(), numChannels_);
        for (int b = 0; b < n; ++b)
        {
            nS = std::min(nS, bandOutputs[b].getNumSamples());
            nCh = std::min(nCh, bandOutputs[b].getNumChannels());
        }
        if (nS <= 0) return 0;

        if (lpActive)
        {
            processLinearPhase(input, bandOutputs, n, nS, nCh);
            passExtraChannels(input, bandOutputs, n, nS, nCh);
            return n;
        }

        // The minimum-phase engine changes its band count or its topology
        // over a fade (a change that arrives during one waits for it to end).
        const Topology topology = topology_.load(std::memory_order_relaxed);
        if (iirBands_ == 0)
        {
            iirBands_ = n;
            iirTopo_ = topology;
        }
        else if (fadeLeft_ == 0 && (n != iirBands_ || topology != iirTopo_)) startBandFade(n, topology);
        const int written = std::min(iirBands_, numOutputBands);
        processIIR(input, bandOutputs, written, nS, nCh);
        passExtraChannels(input, bandOutputs, written, nS, nCh);
        return written;
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Sets the number of output bands (2..MaxBands).
     *
     * @note Changing the band count re-initialises ALL split frequencies to
     *       log-spaced defaults (100 Hz .. 10 kHz); reapply custom frequencies
     *       afterwards, or use setNumBandsKeepingSplits(). A live change goes
     *       over without a step: in the minimum-phase engine the old and the
     *       new topology run together for 20 ms and the bands move from one
     *       to the other (a band that leaves folds into the new top band, one
     *       that comes in rises from silence), so the band sum moves from one
     *       allpass to the other; in the linear-phase engine the new design
     *       comes in over a partition (see the file notes). Measured at LR24
     *       with tones at 100 Hz, 3 kHz and 9 kHz, 3 to 4 bands: band 0's
     *       largest step 0.0050 against 0.0049 steady (0.072 when the change
     *       took one sample), the new top band's 0.187 against 0.184.
     */
    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 2, MaxBands), std::memory_order_relaxed);
        initDefaultFrequencies();
        dirty_.store(true, std::memory_order_release);
        lpMagDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the number of output bands (2..MaxBands) and keeps the splits.
     *
     * The split frequencies do not depend on the band count: the count only
     * says how many of the splits are used. A split that stays keeps its
     * frequency (and, in the minimum-phase engine, its memory); a split
     * that comes back takes the frequency it had, or, when that is not
     * above the split below it, the point halfway in octaves between that
     * split and 20 kHz. The change itself goes over as setNumBands()
     * describes. Thread-safe.
     */
    void setNumBandsKeepingSplits(int n) noexcept
    {
        const int bands = std::clamp(n, 2, MaxBands);
        for (int s = 1; s < bands - 1; ++s)
        {
            const T below = targetFrequencies_[static_cast<size_t>(s - 1)].load(std::memory_order_relaxed);
            const T at = targetFrequencies_[static_cast<size_t>(s)].load(std::memory_order_relaxed);
            if (!(at > below))
                targetFrequencies_[static_cast<size_t>(s)].store(
                    std::sqrt(std::max(below, T(1)) * T(20000)), std::memory_order_relaxed);
        }
        numBands_.store(bands, std::memory_order_relaxed);
        freqUpdatePending_.store(true, std::memory_order_release);
        dirty_.store(true, std::memory_order_release);
        lpMagDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets a crossover frequency. Automatically maintains sorting.
     *
     * Non-finite frequencies are ignored; values are floored to 1 Hz here and
     * clamped to [20, 0.499 * sampleRate] at design time.
     */
    void setCrossoverFrequency(int index, T freqHz) noexcept
    {
        if (!std::isfinite(freqHz)) return;
        if (index >= 0 && index < numBands_.load(std::memory_order_relaxed) - 1)
        {
            freqHz = std::max(freqHz, T(1));

            // Read all current targets into a local array to sort them
            std::array<T, kMaxSplits> localTargets;
            for (int i = 0; i < kMaxSplits; ++i)
                localTargets[i] = targetFrequencies_[i].load(std::memory_order_relaxed);

            localTargets[static_cast<size_t>(index)] = freqHz;

            // Sort on the UI thread to avoid Audio Thread priority inversion/CPU
            // spikes. activeSplits is in [0, kMaxSplits] (numBands_ is clamped to
            // [2, MaxBands]); an explicit insertion sort over this tiny fixed
            // array is exact for the <=11 elements and avoids libstdc++'s generic
            // std::sort, whose internal threshold constant trips a spurious GCC
            // -O2 -Warray-bounds on the small std::array.
            const int activeSplits = std::clamp(numBands_.load(std::memory_order_relaxed) - 1, 0, kMaxSplits);
            for (int i = 1; i < activeSplits; ++i)
            {
                const T key = localTargets[static_cast<size_t>(i)];
                int j = i - 1;
                while (j >= 0 && localTargets[static_cast<size_t>(j)] > key)
                {
                    localTargets[static_cast<size_t>(j + 1)] = localTargets[static_cast<size_t>(j)];
                    --j;
                }
                localTargets[static_cast<size_t>(j + 1)] = key;
            }

            for (int i = 0; i < kMaxSplits; ++i)
                targetFrequencies_[i].store(localTargets[static_cast<size_t>(i)], std::memory_order_relaxed);

            freqUpdatePending_.store(true, std::memory_order_release);
            dirty_.store(true, std::memory_order_release);
            lpMagDirty_.store(true, std::memory_order_release);
        }
    }

    /**
     * @brief Sets the crossover slope: 6, 12, 24, 36 or 48 (dB/oct). Other values are ignored.
     *
     * 12 to 48 are Linkwitz-Riley splits (the bands sum to an allpass of the
     * input; each branch is 6 dB down at the split). 6 is a first-order
     * low-pass and its complement (the bands sum to the input itself; each
     * branch is 3 dB down at the split in the minimum-phase engine and 6 dB
     * in the linear-phase one).
     */
    void setOrder(int order) noexcept
    {
        if (order == 6 || order == 12 || order == 24 || order == 36 || order == 48)
        {
            order_.store(order, std::memory_order_relaxed);
            dirty_.store(true, std::memory_order_release);
            lpMagDirty_.store(true, std::memory_order_release);
        }
    }

    /** @brief Sets the processing mode. A live switch resets filter state (transient). */
    void setFilterMode(FilterMode mode) noexcept
    {
        const int m = std::clamp(static_cast<int>(mode), 0, 1);
        filterMode_.store(static_cast<FilterMode>(m), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
        lpMagDirty_.store(true, std::memory_order_release);
    }

    /** @brief How the minimum-phase sections are realised (see setStructure()). */
    enum class Structure
    {
        DirectForm,  ///< Direct form with a double memory, all sections in one pass (default).
        Ladder       ///< Normalized ladders: passive under any movement of the splits.
    };

    /**
     * @brief Chooses how the sections of the minimum-phase engine are realised.
     *
     * Ladder runs every section of the splits and of the phase correction as
     * a normalized ladder (NormalizedLadder): the same band responses (to
     * 4e-9), and a memory that can only lose energy however its coefficients
     * move. During a glide the ladders' coefficients ramp linearly from one
     * refresh of the 32-sample grid to the next instead of stepping. Measured
     * on four bands in stereo it costs 2.4 to 6.3 times the direct form. The
     * 5 ms glide of the splits already keeps the direct form's memory
     * consistent with its coefficients: with a split jumping between 100 Hz
     * and 10 kHz every 64 samples, the band sum of 0.3 of noise peaks at 0.65
     * either way at LR24 (0.70 ladder, 0.64 direct at LR48), so the ladder is
     * for those who want the guarantee rather than a lower peak. The ladders are allocated by prepare() on the
     * heap, so the switch itself never allocates. Thread-safe; the switch
     * restarts the sections' memory at the next block, which may click:
     * choose it before playback. The linear-phase engine is unaffected.
     *
     * @param structure DirectForm (default) or Ladder.
     */
    void setStructure(Structure structure) noexcept
    {
        structure_.store(static_cast<Structure>(std::clamp(static_cast<int>(structure), 0, 1)),
                         std::memory_order_relaxed);
    }

    /** @brief The structure set by setStructure(). */
    [[nodiscard]] Structure getStructure() const noexcept { return structure_.load(std::memory_order_relaxed); }

    /** @brief How the splits of the minimum-phase engine are arranged (see setTopology()). */
    enum class Topology
    {
        Cascade,  ///< Each split takes what the one below left above it (default).
        Balanced  ///< The middle split first, each half divided the same way.
    };

    /**
     * @brief Arranges the splits of the minimum-phase engine.
     *
     * Cascade splits the lowest band off first and passes what is left on;
     * each band then takes the phase correction of every split above it,
     * S(S-1)/2 chains for S splits (10 with 6 bands, 55 with 12). Balanced
     * splits at the middle split first and divides each half the same way;
     * each branch takes the corrections of the splits of the other half once,
     * before it is divided (6 chains with 6 bands, 22 with 12). The bands
     * still sum to the same allpass, but their skirts differ: a band carries
     * the slopes of the splits on its way down the tree rather than every
     * split below it, so some skirts fall faster and others slower than in
     * the cascade. A live change goes over without a step (both arrangements
     * run for 20 ms on separate memories and the bands move from one to the
     * other). Thread-safe. The linear-phase engine is unaffected.
     *
     * @param topology Cascade (default) or Balanced.
     */
    void setTopology(Topology topology) noexcept
    {
        topology_.store(static_cast<Topology>(std::clamp(static_cast<int>(topology), 0, 1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief The arrangement set by setTopology(). */
    [[nodiscard]] Topology getTopology() const noexcept { return topology_.load(std::memory_order_relaxed); }

    // -- Queries -------------------------------------------------------------

    [[nodiscard]] int getNumBands() const noexcept { return numBands_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getOrder() const noexcept { return order_.load(std::memory_order_relaxed); }
    [[nodiscard]] FilterMode getFilterMode() const noexcept { return filterMode_.load(std::memory_order_relaxed); }

    /** @brief Returns the target frequency of split point `index`. */
    [[nodiscard]] T getCrossoverFrequency(int index) const noexcept
    {
        if (index < 0 || index >= kMaxSplits) return T(0);
        return targetFrequencies_[static_cast<size_t>(index)].load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns latency in samples: 0 for IIR; for linear-phase, three
     *        quarters of the FIR (3072 at 48 kHz), whatever the block size.
     *
     * Reports the linear-phase latency only when the engine actually exists
     * (see prepare()); if linear-phase mode fell back to IIR, this returns 0.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && lpFft_ != nullptr)
                   ? lpLatency_ : 0;
    }

    void reset() noexcept
    {
        for (auto& k : banks_)
            if (k) resetBank(*k);
        fadeBank_ = -1;

        std::fill(lpRing_.begin(), lpRing_.end(), T(0));
        std::fill(lpFdl_.begin(), lpFdl_.end(), T(0));
        std::fill(lpOut_.begin(), lpOut_.end(), T(0));
        lpPos_ = 0;
        lpStageBlock_ = 0;
        lpPlay_ = 0;
        lpOutBands_ = lpSetBands_[static_cast<size_t>(lpActiveSet_)];
        // No partition task is in flight: the first one starts when the
        // first partition of input is in.
        lpTaskSet_ = -1;
        lpTaskFdl_ = 0;
        lpTaskUnit_ = 0;
        lpTaskUnits_ = 0;
        lpTaskDone_ = 0.0;
        lpTaskTotal_ = 0.0;
        lpLaunchedSet_ = -1;
        lpTaskFadeSet_ = -1;
        glideRemain_ = 0;
        iirBands_ = 0;
        fadeLeft_ = 0;
    }


    /** @brief Serializes the split topology (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("XOVR"), 1);
        const int n = getNumBands();
        w.write("numBands", n);
        w.write("order", getOrder());
        w.write("mode", static_cast<int32_t>(getFilterMode()));
        w.write("structure", static_cast<int32_t>(getStructure()));
        w.write("topology", static_cast<int32_t>(getTopology()));
        char key[16];
        for (int i = 0; i < n - 1; ++i)
        {
            std::snprintf(key, sizeof(key), "x%d", i);
            w.write(key, static_cast<float>(getCrossoverFrequency(i)));
        }
        return w.blob();
    }

    /** @brief Restores the split topology from a blob. */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("XOVR")) return false;
        setNumBands(std::clamp(r.read("numBands", 2), 2, MaxBands));
        setOrder(r.read("order", 24));
        setFilterMode(static_cast<FilterMode>(std::clamp(r.read("mode", 0), 0, 1)));
        setStructure(static_cast<Structure>(std::clamp(r.read("structure", 0), 0, 1)));
        setTopology(static_cast<Topology>(std::clamp(r.read("topology", 0), 0, 1)));
        const int n = getNumBands();
        char key[16];
        for (int i = 0; i < n - 1; ++i)
        {
            std::snprintf(key, sizeof(key), "x%d", i);
            const float f = r.read(key, -1.0f);
            if (f > 0.0f) setCrossoverFrequency(i, static_cast<T>(f));
        }
        return true;
    }

private:
    static constexpr int kMaxSplits = MaxBands - 1;
    static constexpr int kMaxStagesPerFilter = 4;
    static constexpr double kLpFirSeconds = 4096.0 / 48000.0; ///< Linear-phase FIR span (rounded up to a power of two).
    static constexpr int kLpMaxFirLength = 1 << 16; ///< FIR length cap (768 kHz).
    static constexpr int kLpPartitions = 8;        ///< Uniform partitions of the FIR.
    static_assert(kLpPartitions % 2 == 0, "the FIR centre must fall on a partition boundary");
    /// Output delay of the engine, in partitions. Two leave one partition of
    /// slack: the work for an output partition can be spread over the
    /// partition before it. Latency: half the FIR plus these partitions.
    static constexpr int kLpLeadBlocks = 2;
    /// Input history, in partitions (power of two). The work for input
    /// partition i runs while partition i + 1 is staged and reads partitions
    /// i - 1 and i (the transform frame) and i - kLpPartitions / 2 (the
    /// delayed input of the residual band).
    static constexpr int kLpRingBlocks = 8;
    static_assert(kLpRingBlocks >= kLpPartitions / 2 + 2
                      && (kLpRingBlocks & (kLpRingBlocks - 1)) == 0,
                  "the input history must hold the residual delay and the staged block");
    static constexpr int kGlideGrid = 32;           ///< IIR coefficient refresh grid during a glide (samples).
    static constexpr double kFadeSeconds = 0.02;    ///< A band-count change in the minimum-phase engine.
    static constexpr int kFadeChunk = 64;           ///< Samples per pass of that fade.
    // Linear-phase cost model, in units of one complex multiply-accumulate
    // bin (about 0.2 ns on a desktop core with AVX2). It weighs the steps of
    // a partition's work (spread over the partition) and of a design (paced
    // at a share of the steady work); only the ratios matter.
    static constexpr double kLpFftUnitsPerPoint = 2.4;       ///< Partition-size transform, per real point.
    static constexpr double kLpDesignFftUnitsPerPoint = 2.7; ///< Long design transform, per real point.
    static constexpr double kLpMagUnitsPerBin = 3.4;         ///< Ideal magnitude, per design bin.
    static constexpr double kLpCopyUnitsPerSample = 0.1;     ///< Copy or window, per sample.
    static constexpr double kLpResidualUnitsPerSample = 0.35; ///< Residual band, per sample and band.
    static constexpr int kLpMagChunks = 4;                   ///< Design steps per band's magnitudes.
    /// A kernel partition whose taps sum (in magnitude) below this is not
    /// convolved: it moves the band's response by less than -120 dB.
    static constexpr double kLpSkipL1 = 1e-6;
    /// Design work per sample, as a share of the steady work of a stereo
    /// stream with the design's band count (whatever the channel count, so a
    /// change lands in the same time in mono).
    static constexpr double kLpDesignShare = 0.45;

    struct SplitPoint
    {
        BiquadCascade<T, kMaxStagesPerFilter, 16> lp;   ///< The low branch's sections, run in one pass.
        BiquadCascade<T, kMaxStagesPerFilter, 16> hp;   ///< The high branch's sections.
    };

    struct AllPassChain
    {
        BiquadCascade<T, kMaxStagesPerFilter, 16> stages;   ///< The phase correction's sections.
    };

    /**
     * First-order allpass, bilinear transform of (omega - s)/(omega + s):
     * H(z) = (a + z^-1)/(1 + a z^-1) with a = (w - 1)/(w + 1), w = tan(pi*f/fs).
     * Phase is -90 degrees exactly at `freq` and equals LP1 - HP1 of the
     * matching first-order pair bit-exactly (the LR12 sum identity). The
     * previous version used a = (1 - w)/(1 + w) (sign flipped), which placed
     * the -90 degree point at the mirrored frequency fs/2 - f, so the LR12
     * phase correction was wrong across the whole band.
     */
    [[nodiscard]] static BiquadCoeffs makeFirstOrderAllPass(double sampleRate, double freq) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        const double w = std::tan(std::numbers::pi * freq / sampleRate);
        const double a = (w - 1.0) / (w + 1.0);
        BiquadCoeffs coeffs;
        coeffs.b0 = a;
        coeffs.b1 = 1.0;
        coeffs.b2 = 0.0;
        coeffs.a1 = a;
        coeffs.a2 = 0.0;
        return coeffs;
    }

    void initDefaultFrequencies() noexcept
    {
        int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        const T logMin = std::log(T(100));
        const T logMax = std::log(T(10000));

        for (int s = 0; s < numSplits; ++s)
        {
            T t = static_cast<T>(s + 1) / static_cast<T>(numSplits + 1);
            targetFrequencies_[s].store(std::exp(logMin + t * (logMax - logMin)), std::memory_order_relaxed);
        }
        freqUpdatePending_.store(true, std::memory_order_release);
    }

    /// Applies the published targets immediately (linear-phase path and mode
    /// switches) and parks the smoothers on them so a later switch back to
    /// IIR does not replay a stale ramp. Audio thread only.
    void syncFrequenciesToTargets() noexcept
    {
        for (int i = 0; i < kMaxSplits; ++i)
        {
            const T t = targetFrequencies_[i].load(std::memory_order_relaxed);
            frequencies_[i] = t;
            freqSmoothers_[i].reset(t);
        }
    }

    [[nodiscard]] static double clampSplitFreq(double f, double sr) noexcept
    {
        return std::clamp(f, 20.0, sr * 0.499);
    }

    void updateCoefficients(int ladderRamp = 0) noexcept
    {
        if (spec_.sampleRate <= 0 || !banks_[0]) return;

        int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        double sr = spec_.sampleRate;

        // Splits (and their allpass chains) that a band count increase brings
        // back are reset where they come into the path (startBandFade()):
        // a split still fading out keeps playing until then.

        // The phase-correction allpass per band/split must equal the allpass
        // that the split's LP/HP branches sum to: ONE first-order section for
        // LR12 (LP1^2 - HP1^2 = AP1 exactly), ONE second-order section for
        // LR24 (LP2^2 + HP2^2 = AP2(Q=0.7071)), TWO sections (q1, q2) for
        // LR48. Applying numStagesPerFilter_ sections (the old behaviour)
        // doubled the correction phase and carved up to -3.5 dB (LR24) /
        // -13 dB (LR48) holes into the band sum with octave-spaced splits.
        // Note: Frequencies are already guaranteed to be sorted by the setter thread.
        switch (order_.load(std::memory_order_relaxed))
        {
            case 6:
                // A first-order low-pass and the high-pass on the same pole:
                // they add up to the input itself (their numerators sum to
                // the denominator), so no band needs a phase correction.
                numStagesPerFilter_ = 1;
                numStagesAllpass_   = 0;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                    setSplitSection(s, 0, BiquadCoeffs::makeFirstOrderLowPass(sr, f),
                                    BiquadCoeffs::makeFirstOrderHighPass(sr, f), ladderRamp);
                }
                break;

            case 36:
                // Two third-order Butterworth filters in series per branch:
                // a first-order section and a second-order one of Q = 1,
                // twice. The two branches meet half a turn apart (three
                // times 180 degrees), as in LR12, so the high branch is
                // inverted and the pair sums to the allpass of one
                // third-order filter: a first-order and a second-order
                // section.
                numStagesPerFilter_ = 4;
                numStagesAllpass_   = 2;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                    const auto lp1 = BiquadCoeffs::makeFirstOrderLowPass(sr, f);
                    const auto hp1 = BiquadCoeffs::makeFirstOrderHighPass(sr, f);
                    const auto lp2 = BiquadCoeffs::makeLowPass(sr, f, 1.0);
                    const auto hp2 = BiquadCoeffs::makeHighPass(sr, f, 1.0);
                    auto hp1Inverted = hp1;
                    hp1Inverted.b0 = -hp1Inverted.b0;
                    hp1Inverted.b1 = -hp1Inverted.b1;
                    hp1Inverted.b2 = -hp1Inverted.b2;
                    setSplitSection(s, 0, lp1, hp1Inverted, ladderRamp);
                    setSplitSection(s, 1, lp2, hp2, ladderRamp);
                    setSplitSection(s, 2, lp1, hp1, ladderRamp);
                    setSplitSection(s, 3, lp2, hp2, ladderRamp);
                    const auto ap1 = makeFirstOrderAllPass(sr, f);
                    const auto ap2 = BiquadCoeffs::makeAllPass(sr, f, 1.0);
                    setCorrection(s, numSplits, 0, ap1, ladderRamp);
                    setCorrection(s, numSplits, 1, ap2, ladderRamp);
                }
                break;

            case 12:
                numStagesPerFilter_ = 2;
                numStagesAllpass_   = 1;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                    auto lpC = BiquadCoeffs::makeFirstOrderLowPass(sr, f);
                    auto hpC = BiquadCoeffs::makeFirstOrderHighPass(sr, f);
                    auto apC = makeFirstOrderAllPass(sr, f);

                    // LR12 sums flat only with the high branch polarity
                    // inverted (LP^2 - HP^2 = allpass; LP^2 + HP^2 notches at
                    // fc). Bake the inversion into the first HP stage: odd
                    // bands come out inverted, the sum is allpass-flat.
                    auto hpC0 = hpC;
                    hpC0.b0 = -hpC0.b0;
                    hpC0.b1 = -hpC0.b1;
                    hpC0.b2 = -hpC0.b2;

                    setSplitSection(s, 0, lpC, hpC0, ladderRamp);
                    setSplitSection(s, 1, lpC, hpC, ladderRamp);
                    setCorrection(s, numSplits, 0, apC, ladderRamp);
                }
                break;

            case 24:
                numStagesPerFilter_ = 2;
                numStagesAllpass_   = 1;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                    // Exact Butterworth Q: the rounded 0.7071 left the band sum
                    // 3.4e-4 dB off flat (now 1e-6).
                    constexpr double qB = 0.70710678118654752440;
                    auto lpC = BiquadCoeffs::makeLowPass(sr, f, qB);
                    auto hpC = BiquadCoeffs::makeHighPass(sr, f, qB);
                    auto apC = BiquadCoeffs::makeAllPass(sr, f, qB);

                    for (int st = 0; st < 2; ++st)
                        setSplitSection(s, st, lpC, hpC, ladderRamp);
                    setCorrection(s, numSplits, 0, apC, ladderRamp);
                }
                break;

            case 48:
            {
                numStagesPerFilter_ = 4;
                numStagesAllpass_   = 2;
                // Exact 4th-order Butterworth section Qs, 1/(2 cos(pi/8)) and
                // 1/(2 cos(3 pi/8)): the rounded values left the band sum
                // 1.1e-3 dB off flat (now 1.5e-6).
                constexpr double q1 = 0.54119610014619698440;
                constexpr double q2 = 1.30656296487637652786;
                const double qArr[4] = { q1, q2, q1, q2 };

                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);

                    for (int st = 0; st < 4; ++st)
                        setSplitSection(s, st, BiquadCoeffs::makeLowPass(sr, f, qArr[st]),
                                        BiquadCoeffs::makeHighPass(sr, f, qArr[st]), ladderRamp);
                    const auto ap1 = BiquadCoeffs::makeAllPass(sr, f, q1);
                    const auto ap2 = BiquadCoeffs::makeAllPass(sr, f, q2);
                    setCorrection(s, numSplits, 0, ap1, ladderRamp);
                    setCorrection(s, numSplits, 1, ap2, ladderRamp);
                }
                break;
            }

            default: break;
        }

        const bool tree = treeInUse();
        for (int i = 0; i < 2; ++i)
        {
            if (i != bank_ && i != fadeBank_) continue;
            IirBank& k = bank(i);
            for (int s = 0; s < numSplits; ++s)
            {
                k.splits[static_cast<size_t>(s)].lp.setNumSections(numStagesPerFilter_);
                k.splits[static_cast<size_t>(s)].hp.setNumSections(numStagesPerFilter_);
                for (int b = 0; b < (tree ? numSplits : s); ++b)
                    if (b != s) k.allpass[static_cast<size_t>(b * kMaxSplits + s)].stages.setNumSections(numStagesAllpass_);
            }
        }
    }

    static constexpr int kLadderSplitStride = 2 * kMaxStagesPerFilter;
    static constexpr int kLadderAllpassBase = kMaxSplits * kLadderSplitStride;
    static constexpr int kLadderCount = kLadderAllpassBase + MaxBands * kMaxSplits * 2;

    /// Ladder index of section `st` of split `s`, low branch.
    [[nodiscard]] static constexpr int ladderLow(int s, int st) noexcept { return s * kLadderSplitStride + st; }
    /// Ladder index of section `st` of split `s`, high branch.
    [[nodiscard]] static constexpr int ladderHigh(int s, int st) noexcept
    {
        return s * kLadderSplitStride + kMaxStagesPerFilter + st;
    }
    /// Ladder index of section `st` of the correction of band `b` for split `s`.
    [[nodiscard]] static constexpr int ladderAllpass(int b, int s, int st) noexcept
    {
        return kLadderAllpassBase + (b * kMaxSplits + s) * 2 + st;
    }

    /// The sections of the minimum-phase engine: the splits, the phase
    /// corrections and the same as ladders. Two banks: a change that the
    /// band sum cannot follow in one memory (one involving the balanced
    /// topology) runs the old configuration on one and the new on the other.
    struct IirBank
    {
        std::array<SplitPoint, kMaxSplits> splits {};
        std::vector<AllPassChain> allpass;             ///< [band (or tree node) * kMaxSplits + split]
        std::vector<NormalizedLadder<T, 16>> ladders;  ///< See ladderLow(), ladderHigh(), ladderAllpass().
    };

    /// Channel pointers of every band, at the offset of a range.
    struct BandPtrs
    {
        T* p[MaxBands][16] {};
    };

    [[nodiscard]] IirBank& bank(int index) noexcept { return *banks_[static_cast<size_t>(index)]; }

    /// Whether some configuration in play uses the balanced tree (whose
    /// corrections also run for splits below a band).
    [[nodiscard]] bool treeInUse() const noexcept
    {
        return topology_.load(std::memory_order_relaxed) == Topology::Balanced || iirTopo_ == Topology::Balanced
               || (fadeBank_ >= 0 && fadeFromTopo_ == Topology::Balanced);
    }

    /// One section of the two branches of a split: to the one-pass cascades
    /// and, while the ladders run, to the ladders (ramping over `ramp` samples),
    /// in the bank that plays and in the one a fade brings in.
    void setSplitSection(int s, int st, const BiquadCoeffs& low, const BiquadCoeffs& high, int ramp) noexcept
    {
        for (int i = 0; i < 2; ++i)
        {
            if (i != bank_ && i != fadeBank_) continue;
            IirBank& k = bank(i);
            k.splits[static_cast<size_t>(s)].lp.setCoeffs(st, low);
            k.splits[static_cast<size_t>(s)].hp.setCoeffs(st, high);
            if (ladderActive_)
            {
                k.ladders[static_cast<size_t>(ladderLow(s, st))].rampCoeffs(low, ramp);
                k.ladders[static_cast<size_t>(ladderHigh(s, st))].rampCoeffs(high, ramp);
            }
        }
    }

    /// One section of the phase correction of band `b` (or tree node `b`) for split `s`.
    void setAllpassSection(int b, int s, int st, const BiquadCoeffs& c, int ramp) noexcept
    {
        for (int i = 0; i < 2; ++i)
        {
            if (i != bank_ && i != fadeBank_) continue;
            IirBank& k = bank(i);
            k.allpass[static_cast<size_t>(b * kMaxSplits + s)].stages.setCoeffs(st, c);
            if (ladderActive_)
                k.ladders[static_cast<size_t>(ladderAllpass(b, s, st))].rampCoeffs(c, ramp);
        }
    }

    /// The corrections for split `s`: of every band below it, and with the
    /// tree in play also of every node above it.
    void setCorrection(int s, int numSplits, int st, const BiquadCoeffs& c, int ramp) noexcept
    {
        const int end = treeInUse() ? numSplits : s;
        for (int b = 0; b < end; ++b)
            if (b != s) setAllpassSection(b, s, st, c, ramp);
    }

    /// `count` ladders from `first` over one channel: src to dst (may be the
    /// same), in double between the sections. advanceLadders() moves the ramps on.
    void runLadders(IirBank& k, const T* src, T* dst, int length, int first, int count, int channel) noexcept
    {
        double* tmp = ladderTmp_.data();
        for (int i = 0; i < length; ++i) tmp[i] = static_cast<double>(src[i]);
        for (int j = 0; j < count; ++j)
            k.ladders[static_cast<size_t>(first + j)].processSegment(tmp, length, channel);
        for (int i = 0; i < length; ++i) dst[i] = static_cast<T>(tmp[i]);
    }

    /// Moves the ramps of `count` ladders from `first` on, once every channel has run.
    void advanceLadders(IirBank& k, int first, int count, int length) noexcept
    {
        for (int j = 0; j < count; ++j)
            k.ladders[static_cast<size_t>(first + j)].advanceRamp(length);
    }

    void processIIR(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nS, int nCh) noexcept
    {
        DenormalGuard guard;
        const int numSplits = numBands - 1;

        // During a glide the coefficients are refreshed every kGlideGrid
        // samples on a grid that runs across blocks: a segment started in one
        // block keeps its coefficients into the next, so the glide is the
        // same whatever the host block size.
        int offset = 0;
        while (offset < nS)
        {
            if (glideRemain_ == 0)
            {
                bool anySmoothing = false;
                for (int i = 0; i < numSplits; ++i)
                    anySmoothing = anySmoothing || freqSmoothers_[i].isSmoothing();
                if (!anySmoothing) break;
                for (int i = 0; i < numSplits; ++i)
                {
                    for (int s = 0; s < kGlideGrid; ++s)
                        (void)freqSmoothers_[i].getNextValue();
                    frequencies_[i] = freqSmoothers_[i].getCurrentValue();
                }
                // The ladders ramp to the new coefficients over the segment.
                updateCoefficients(kGlideGrid);
                glideRemain_ = kGlideGrid;
            }
            const int len = std::min(glideRemain_, nS - offset);
            processRange(input, outputs, numBands, nCh, offset, len);
            offset += len;
            glideRemain_ -= len;
        }
        if (offset < nS)
            processRange(input, outputs, numBands, nCh, offset, nS - offset);
    }

    [[nodiscard]] static BandPtrs bandsOf(AudioBufferView<T>* outputs, int numBands, int nCh, int offset) noexcept
    {
        BandPtrs out;
        for (int b = 0; b < numBands; ++b)
            for (int ch = 0; ch < nCh; ++ch) out.p[b][ch] = outputs[b].getChannel(ch) + offset;
        return out;
    }

    /// A range of the minimum-phase engine: through the fade while one runs.
    void processRange(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nCh, int offset,
                      int length) noexcept
    {
        while (fadeLeft_ > 0 && length > 0)
        {
            const int len = std::min({ length, fadeLeft_, kFadeChunk });
            if (fadeBank_ >= 0) processBankFadeRange(input, outputs, numBands, nCh, offset, len);
            else processFadeRange(input, outputs, numBands, nCh, offset, len);
            fadeLeft_ -= len;
            offset += len;
            length -= len;
            if (fadeLeft_ == 0 && fadeBank_ >= 0)
            {
                bank_ = fadeBank_;
                fadeBank_ = -1;
            }
        }
        if (length > 0)
        {
            BandPtrs out = bandsOf(outputs, numBands, nCh, offset);
            runTopology(bank(bank_), iirTopo_, numBands, input, offset, out, nCh, length);
        }
    }

    /// Clears the memory of every section of a bank.
    static void resetBank(IirBank& k) noexcept
    {
        for (auto& sp : k.splits)
        {
            sp.lp.reset();
            sp.hp.reset();
        }
        for (auto& apChain : k.allpass)
            apChain.stages.reset();
        for (auto& l : k.ladders)
            l.reset();
    }

    /// Starts the fade from the configuration in play to `bands` bands on `topology`.
    void startBandFade(int bands, Topology topology) noexcept
    {
        const bool shared = iirTopo_ == Topology::Cascade && topology == Topology::Cascade;
        fadeFrom_ = iirBands_;
        fadeFromTopo_ = iirTopo_;
        iirBands_ = bands;
        iirTopo_ = topology;
        fadeLeft_ = fadeLength_;
        if (!shared)
        {
            // The new configuration on the other bank: the coefficients of
            // the bank in play (the containers have the same sizes: copies,
            // no allocation), its memory from silence.
            fadeBank_ = 1 - bank_;
            IirBank& to = bank(fadeBank_);
            const IirBank& from = bank(bank_);
            to.splits = from.splits;
            std::copy(from.allpass.begin(), from.allpass.end(), to.allpass.begin());
            std::copy(from.ladders.begin(), from.ladders.end(), to.ladders.begin());
            resetBank(to);
            return;
        }
        // The splits that come in, and the corrections of the bands below
        // them, start from silence (their memory may be arbitrarily old).
        IirBank& k = bank(bank_);
        for (int s = fadeFrom_ - 1; s < bands - 1; ++s)
        {
            k.splits[static_cast<size_t>(s)].lp.reset();
            k.splits[static_cast<size_t>(s)].hp.reset();
            for (int b = 0; b < s; ++b)
                k.allpass[static_cast<size_t>(b * kMaxSplits + s)].stages.reset();
            if (!k.ladders.empty())
            {
                for (int st = 0; st < kMaxStagesPerFilter; ++st)
                {
                    k.ladders[static_cast<size_t>(ladderLow(s, st))].reset();
                    k.ladders[static_cast<size_t>(ladderHigh(s, st))].reset();
                }
                for (int b = 0; b < s; ++b)
                    for (int st = 0; st < 2; ++st)
                        k.ladders[static_cast<size_t>(ladderAllpass(b, s, st))].reset();
            }
        }
    }

    /// Split `s` over `len` samples: its low branch from `work` into `low`, and `work` to its high branch.
    void runSplit(IirBank& k, int s, T* const* work, T* const* low, int nCh, int len) noexcept
    {
        if (ladderActive_)
        {
            const int perFilter = numStagesPerFilter_;
            for (int ch = 0; ch < nCh; ++ch)
            {
                runLadders(k, work[ch], low[ch], len, ladderLow(s, 0), perFilter, ch);
                runLadders(k, work[ch], work[ch], len, ladderHigh(s, 0), perFilter, ch);
            }
            advanceLadders(k, ladderLow(s, 0), perFilter, len);
            advanceLadders(k, ladderHigh(s, 0), perFilter, len);
            return;
        }
        for (int ch = 0; ch < nCh; ++ch) std::copy(work[ch], work[ch] + len, low[ch]);
        k.splits[static_cast<size_t>(s)].lp.processBlock(AudioBufferView<T>(low, nCh, len));
        k.splits[static_cast<size_t>(s)].hp.processBlock(AudioBufferView<T>(work, nCh, len));
    }

    /// The correction of band `b` (or tree node `b`) for split `s`, in place over `len` samples.
    void runAllpass(IirBank& k, int b, int s, T* const* band, int nCh, int len) noexcept
    {
        if (ladderActive_)
        {
            for (int ch = 0; ch < nCh; ++ch)
                runLadders(k, band[ch], band[ch], len, ladderAllpass(b, s, 0), numStagesAllpass_, ch);
            advanceLadders(k, ladderAllpass(b, s, 0), numStagesAllpass_, len);
            return;
        }
        k.allpass[static_cast<size_t>(b * kMaxSplits + s)].stages.processBlock(AudioBufferView<T>(band, nCh, len));
    }

    /**
     * The balanced tree over bands lo..hi, whose signal is in `buf` (the
     * buffer of band hi). The middle split r divides it; the low branch goes
     * to band r's buffer and takes the corrections of the splits above r in
     * this range, the high branch those of the splits below r; each half is
     * then divided the same way. The corrections of node r are chains
     * [r * kMaxSplits + s].
     */
    void runTree(IirBank& k, int lo, int hi, T* const* buf, BandPtrs& out, int nCh, int len) noexcept
    {
        if (lo >= hi) return;
        const int r = lo + (hi - lo - 1) / 2;
        T* const* low = out.p[r];
        runSplit(k, r, buf, low, nCh, len);
        if (numStagesAllpass_ > 0)
        {
            for (int s = r + 1; s < hi; ++s) runAllpass(k, r, s, low, nCh, len);
            for (int s = lo; s < r; ++s) runAllpass(k, r, s, buf, nCh, len);
        }
        runTree(k, lo, r, low, out, nCh, len);
        runTree(k, r + 1, hi, buf, out, nCh, len);
    }

    /// One configuration of the minimum-phase engine over a range: the input
    /// at `offset` into the bands of `out`.
    void runTopology(IirBank& k, Topology topology, int numBands, AudioBufferView<T> input, int offset, BandPtrs& out,
                     int nCh, int len) noexcept
    {
        const int numSplits = numBands - 1;

        // Non-finite input samples enter as silence: in the recursive
        // sections they would latch for good.
        T* work[16] = {};
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = input.getChannel(ch) + offset;
            work[ch] = getWorkBufChannel(ch);
            for (int i = 0; i < len; ++i)
                work[ch][i] = std::isfinite(src[i]) ? src[i] : T(0);
        }

        if (topology == Topology::Balanced)
        {
            runTree(k, 0, numSplits, work, out, nCh, len);
            for (int ch = 0; ch < nCh; ++ch) std::copy(work[ch], work[ch] + len, out.p[numBands - 1][ch]);
            return;
        }

        // Each split: its low band from what is left, and what is left above
        // it, every section of a filter and every channel in one pass (the
        // memory in double: re-quantising to T between sections would put
        // the reconstruction error back where the double memory removed it).
        for (int s = 0; s < numSplits; ++s)
            runSplit(k, s, work, out.p[s], nCh, len);

        for (int ch = 0; ch < nCh; ++ch) std::copy(work[ch], work[ch] + len, out.p[numBands - 1][ch]);

        for (int s = 1; s < numSplits; ++s)
            for (int b = 0; b < s; ++b)
                runAllpass(k, b, s, out.p[b], nCh, len);
    }

    [[nodiscard]] T* fadeBandOf(std::vector<T>& scratch, int band, int ch) noexcept
    {
        return scratch.data()
               + (static_cast<size_t>(band) * static_cast<size_t>(numChannels_) + static_cast<size_t>(ch))
                     * static_cast<size_t>(kFadeChunk);
    }

    /**
     * A chunk of a band-count fade of the cascade. The larger topology runs
     * whole; on the way the smaller one is read off it: its top band is what
     * is left above its last split, its other bands are the larger
     * topology's before the corrections of the splits it does not have. Each
     * band then goes from the old topology's version to the new one's (a band
     * beyond the smaller topology is silent in it; leaving, it folds into the
     * top band).
     */
    void processFadeRange(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nCh,
                          int offset, int len) noexcept
    {
        IirBank& k = bank(bank_);
        const int large = std::max(fadeFrom_, iirBands_);
        const int small = std::min(fadeFrom_, iirBands_);
        const bool adding = iirBands_ > fadeFrom_;
        T* work[16] = {};
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = input.getChannel(ch) + offset;
            work[ch] = getWorkBufChannel(ch);
            for (int i = 0; i < len; ++i) work[ch][i] = std::isfinite(src[i]) ? src[i] : T(0);
        }
        T* top[16] = {};
        for (int ch = 0; ch < nCh; ++ch) top[ch] = fadeTop_.data() + static_cast<size_t>(ch) * kFadeChunk;
        for (int s = 0; s < large - 1; ++s)
        {
            if (s == small - 1)
                for (int ch = 0; ch < nCh; ++ch) std::copy(work[ch], work[ch] + len, top[ch]);
            T* low[16] = {};
            for (int ch = 0; ch < nCh; ++ch) low[ch] = fadeBandOf(fadeBands_, s, ch);
            runSplit(k, s, work, low, nCh, len);
        }
        for (int ch = 0; ch < nCh; ++ch)
            std::copy(work[ch], work[ch] + len, fadeBandOf(fadeBands_, large - 1, ch));
        // The corrections, the smaller topology's lower bands read off
        // before the first split it does not have (small - 1 <= large - 2).
        for (int s = 1; s < large - 1; ++s)
        {
            if (s == small - 1)
                for (int b = 0; b < small - 1; ++b)
                    for (int ch = 0; ch < nCh; ++ch)
                        std::copy_n(fadeBandOf(fadeBands_, b, ch), len, fadeBandOf(fadeLower_, b, ch));
            for (int b = 0; b < s; ++b)
            {
                T* band[16] = {};
                for (int ch = 0; ch < nCh; ++ch) band[ch] = fadeBandOf(fadeBands_, b, ch);
                runAllpass(k, b, s, band, nCh, len);
            }
        }

        // Each written band from the old version to the new one.
        const T step = T(1) / static_cast<T>(fadeLength_);
        const T start = static_cast<T>(fadeLength_ - fadeLeft_);
        for (int b = 0; b < numBands; ++b)
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = outputs[b].getChannel(ch) + offset;
                const T* lg = fadeBandOf(fadeBands_, b, ch);
                const T* sm = b < small - 1 ? fadeBandOf(fadeLower_, b, ch) : b == small - 1 ? top[ch] : nullptr;
                for (int i = 0; i < len; ++i)
                {
                    const T w = (start + static_cast<T>(i + 1)) * step;   // the new topology's share
                    T inLarge = lg[i];
                    if (!adding && b == small - 1)
                        for (int j = b + 1; j < large; ++j) inLarge += fadeBandOf(fadeBands_, j, ch)[i];
                    const T inSmall = sm != nullptr ? sm[i] : T(0);
                    out[i] = adding ? inSmall + (inLarge - inSmall) * w : inLarge + (inSmall - inLarge) * w;
                }
            }
    }

    /**
     * A chunk of a two-bank fade: the old configuration on the bank in play
     * into scratch, the new one on the other bank into the outputs, then each
     * band from the old version (read in the new band count: missing bands
     * silent, extra ones folded into the top band) to the new one.
     */
    void processBankFadeRange(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nCh,
                              int offset, int len) noexcept
    {
        BandPtrs before;
        for (int b = 0; b < fadeFrom_; ++b)
            for (int ch = 0; ch < nCh; ++ch) before.p[b][ch] = fadeBandOf(fadeBands_, b, ch);
        runTopology(bank(bank_), fadeFromTopo_, fadeFrom_, input, offset, before, nCh, len);
        BandPtrs now = bandsOf(outputs, numBands, nCh, offset);
        runTopology(bank(fadeBank_), iirTopo_, numBands, input, offset, now, nCh, len);

        const T step = T(1) / static_cast<T>(fadeLength_);
        const T start = static_cast<T>(fadeLength_ - fadeLeft_);
        for (int b = 0; b < numBands; ++b)
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* out = now.p[b][ch];
                for (int i = 0; i < len; ++i)
                {
                    T was = T(0);
                    if (b < numBands - 1) was = b < fadeFrom_ ? before.p[b][ch][i] : T(0);
                    else
                        for (int j = b; j < fadeFrom_; ++j) was += before.p[j][ch][i];
                    const T w = (start + static_cast<T>(i + 1)) * step;
                    out[i] = was + (out[i] - was) * w;
                }
            }
    }

    inline T* getWorkBufChannel(int ch) noexcept
    {
        return workBuf_.data() + static_cast<size_t>(ch) * static_cast<size_t>(spec_.maxBlockSize);
    }

    /// Input channels beyond the processed count get band 0 = input,
    /// remaining bands = silence, so the band sum still equals the input
    /// (leaving caller buffers untouched would hand back garbage).
    void passExtraChannels(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nS,
                           int firstCh) noexcept
    {
        const int inCh = input.getNumChannels();
        for (int ch = firstCh; ch < inCh; ++ch)
        {
            for (int b = 0; b < numBands; ++b)
            {
                if (ch >= outputs[b].getNumChannels()) continue;
                T* dst = outputs[b].getChannel(ch);
                if (b == 0)
                {
                    const T* src = input.getChannel(ch);
                    std::copy(src, src + nS, dst);
                }
                else
                {
                    std::fill(dst, dst + nS, T(0));
                }
            }
        }
    }

    // -- Linear-phase kernel design -------------------------------------------
    //
    // A design builds every band's kernel into the kernel set that is not
    // playing, one small step at a time, then swaps sets in one assignment:
    // the bands always come from one consistent design, so they keep summing
    // to the delayed input. Steps per band: the band's ideal magnitudes (in
    // kLpMagChunks pieces), the long inverse FFT (one step per pass), then one
    // windowed partition transform per partition. The last band is the
    // residual and needs no kernel. Everything runs on the audio thread from
    // scratch allocated in prepare().

    [[nodiscard]] T* lpKernelSet(int set) noexcept
    {
        const size_t specLen = static_cast<size_t>(2 * lpPart_ + 2);
        return lpKernelSpec_.data() + static_cast<size_t>(set) * static_cast<size_t>(kMaxSplits * kLpPartitions) * specLen;
    }

    /// Steady cost of one partition (units), for `bands` bands with `parts`
    /// convolved kernel partitions in all and `chans` channels.
    [[nodiscard]] double lpPartitionCost(int bands, int chans, int parts) const noexcept
    {
        return static_cast<double>(chans)
               * (lpCostFft_ + static_cast<double>(parts) * lpCostCma_
                  + static_cast<double>(bands - 1) * lpCostIfft_
                  + static_cast<double>(bands) * lpCostResidualPerBand_);
    }

    /// Captures the design targets and arms the step machine.
    void startLinearPhaseDesign() noexcept
    {
        lpDesignBands_ = std::clamp(numBands_.load(std::memory_order_relaxed), 2, MaxBands);
        const double sr = spec_.sampleRate;
        for (int sp = 0; sp < lpDesignBands_ - 1; ++sp)
            lpDesignFc_[static_cast<size_t>(sp)] = clampSplitFreq(static_cast<double>(frequencies_[static_cast<size_t>(sp)]), sr);
        switch (order_.load(std::memory_order_relaxed))
        {
            case 6: lpDesignExpo_ = 1; break;
            case 12: lpDesignExpo_ = 2; break;
            case 36: lpDesignExpo_ = 6; break;
            case 48: lpDesignExpo_ = 8; break;
            default: lpDesignExpo_ = 4; break;
        }
        lpDesignSet_ = 1 - lpActiveSet_;
        lpDesignBand_ = 0;
        lpDesignStep_ = 0;
        lpDesignParts_ = 0;
        lpDesignRedo_ = false;
        lpDesignActive_ = true;
        // The pace is a share of the steady work of a stereo stream with the
        // design's band count and every partition convolved, whatever the
        // channel count and however many partitions the design keeps.
        lpDesignRate_ = kLpDesignShare * lpPartitionCost(lpDesignBands_, 2, (lpDesignBands_ - 1) * kLpPartitions)
                        / static_cast<double>(lpPart_);
    }

    /// Band magnitudes for a fixed slope over bins [begin, end):
    /// |LP| = 1/(1 + r^Expo), r = f/fc, written as the real parts of the
    /// design spectrum (through lpMag_); the running high-pass product goes
    /// to lpCarry_.
    template <int Expo>
    void designBandMagnitudes(double fc, int begin, int end) noexcept
    {
        const double* freq = lpBinFreq_.data();
        double* carry = lpCarry_.data();
        T* mag = lpMag_.data();
        const double invFc = 1.0 / fc;
        for (int k = begin; k < end; ++k)
        {
            const double r = freq[k] * invFc;
            double rPow = r;
            if constexpr (Expo >= 2) rPow = r * r;
            if constexpr (Expo == 6) rPow *= rPow * rPow;
            else
            {
                if constexpr (Expo >= 4) rPow *= rPow;
                if constexpr (Expo >= 8) rPow *= rPow;
            }
            const double lp = 1.0 / (1.0 + rPow);
            const double c = carry[k];
            mag[k] = static_cast<T>(c * lp);
            carry[k] = c * (rPow * lp); // the high-pass, 1 - lp
        }
        // A separate interleave keeps the loop above contiguous (vectorised).
        T* re = lpTimeResponse_.data();
        for (int k = begin; k < end; ++k) re[2 * k] = mag[k];
    }

    /// Cost of the design step the machine runs next (units; an upper bound:
    /// a partition the design drops costs less).
    [[nodiscard]] double linearPhaseDesignStepCost() const noexcept
    {
        if (lpDesignStep_ < kLpMagChunks) return lpCostMagChunk_;
        if (lpDesignStep_ < kLpMagChunks + lpDesignFftSteps_) return lpCostDesignFftStep_;
        return lpCostPartitionStep_;
    }

    /// Runs one design step and returns its cost (units). The last step of
    /// the last convolved band swaps the kernel sets.
    double linearPhaseDesignStep() noexcept
    {
        const int designSize = 2 * firLength_;
        const int numBins = firLength_ + 1;
        const int b = lpDesignBand_;

        if (lpDesignStep_ < kLpMagChunks)
        {
            // Ideal zero-phase magnitudes of band b. Each band carries the
            // high-pass of every split below it (lpCarry_, the running
            // product), like the IIR tree, so the bands telescope to exactly 1
            // (LP + HP = 1 per split). Ideal Linkwitz-Riley
            // |LP| = 1/(1 + (f/fc)^N) with N = order/6 (1/2/4/6/8 ->
            // 6/12/24/36/48 dB/oct), matching the IIR path's slope; integer
            // powers by repeated squaring. The imaginary parts stay zero (set in
            // prepare(); the inverse FFT does not write its input).
            const int chunk = (numBins + kLpMagChunks - 1) / kLpMagChunks;
            const int begin = std::min(numBins, lpDesignStep_ * chunk);
            const int end = std::min(numBins, begin + chunk);
            if (b == 0) std::fill(lpCarry_.begin() + begin, lpCarry_.begin() + end, 1.0);
            // One straight loop per slope keeps the divisions vectorised.
            const double fc = lpDesignFc_[static_cast<size_t>(b)];
            switch (lpDesignExpo_)
            {
                case 1: designBandMagnitudes<1>(fc, begin, end); break;
                case 2: designBandMagnitudes<2>(fc, begin, end); break;
                case 6: designBandMagnitudes<6>(fc, begin, end); break;
                case 8: designBandMagnitudes<8>(fc, begin, end); break;
                default: designBandMagnitudes<4>(fc, begin, end); break;
            }
            ++lpDesignStep_;
            return lpCostMagChunk_;
        }

        if (lpDesignStep_ < kLpMagChunks + lpDesignFftSteps_)
        {
            // The long inverse FFT, one pass per step (bit-identical to a
            // single inverse()).
            lpDesignFft_->inverseStep(lpDesignStep_ - kLpMagChunks, lpTimeResponse_.data(), lpFirKernel_.data());
            ++lpDesignStep_;
            return lpCostDesignFftStep_;
        }

        // One partition of band b: the zero-phase response centred at
        // firLength/2 under the flat-top window (a cosine taper over the
        // outer quarter on each side: it leaves the response's core
        // untouched, so the passband and slopes stay exact where a
        // full-length window would smear them), then its spectrum.
        const int part = lpDesignStep_ - kLpMagChunks - lpDesignFftSteps_;
        const int halfLen = firLength_ / 2;
        const size_t specLen = static_cast<size_t>(2 * lpPart_ + 2);
        const int first = part * lpPart_; // tap index of the partition start (the
                                          // centre is a partition boundary)
        // Tap i reads the inverse FFT at i - halfLen, wrapped into [0, 2L).
        const int shift = (first >= halfLen) ? -halfLen : designSize - halfLen;
        const T* src = lpFirKernel_.data() + (first + shift);
        const T* win = lpWindow_.data() + first;
        simd::multiplyT(lpScratch_.data(), src, win, lpPart_);
        // A partition whose taps sum below kLpSkipL1 is dropped (not
        // convolved): it changes the band by less than that at any frequency.
        // Away from the centre most partitions hold only the design's rounding
        // noise, so this skips most of the products of the higher bands.
        double l1 = 0.0;
        for (int j = 0; j < lpPart_; ++j) l1 += std::abs(static_cast<double>(lpScratch_[static_cast<size_t>(j)]));
        const size_t slot = static_cast<size_t>(lpDesignSet_ * kMaxSplits * kLpPartitions + b * kLpPartitions + part);
        const bool keep = l1 >= kLpSkipL1;
        lpPartOn_[slot] = keep ? 1 : 0;
        double cost = 2.0 * kLpCopyUnitsPerSample * static_cast<double>(lpPart_);
        if (keep)
        {
            ++lpDesignParts_;
            std::fill(lpScratch_.begin() + lpPart_, lpScratch_.end(), T(0));
            lpFft_->forward(lpScratch_.data(),
                            lpKernelSet(lpDesignSet_) + static_cast<size_t>(b * kLpPartitions + part) * specLen);
            cost = lpCostPartitionStep_;
        }

        if (++lpDesignStep_ == kLpMagChunks + lpDesignFftSteps_ + kLpPartitions)
        {
            lpDesignStep_ = 0;
            if (++lpDesignBand_ == lpDesignBands_ - 1)
            {
                // Every convolved band of the new design is ready: swap.
                // Partition work started from here on uses it; the work in
                // flight finishes on the previous design.
                lpActiveSet_ = lpDesignSet_;
                lpSetBands_[static_cast<size_t>(lpActiveSet_)] = lpDesignBands_;
                lpSetParts_[static_cast<size_t>(lpActiveSet_)] = lpDesignParts_;
                lpDesignActive_ = false;
            }
        }
        return cost;
    }

    /// Runs a whole design at once (prepare(): setup thread).
    void designLinearPhaseNow() noexcept
    {
        lpMagDirty_.store(false, std::memory_order_relaxed);
        startLinearPhaseDesign();
        while (lpDesignActive_) (void)linearPhaseDesignStep();
        lpDesignCredit_ = 0.0;
    }

    /// Advances a pending design within this block's budget. A change that
    /// arrives during a design lets the design finish, then starts the next
    /// one from the latest targets: under continuous automation the kernels
    /// keep landing (a restart would starve them until the automation stops).
    void advanceLinearPhaseDesign(int nS) noexcept
    {
        if (lpMagDirty_.load(std::memory_order_relaxed) &&
            lpMagDirty_.exchange(false, std::memory_order_acquire))
        {
            if (lpDesignActive_) lpDesignRedo_ = true;
            else { startLinearPhaseDesign(); lpDesignCredit_ = 0.0; }
        }
        if (!lpDesignActive_) return;

        // A step runs only when the credit covers it, so a block adds at most
        // its budget plus the credit a short block left (under one step), or
        // one step when a step is larger than a block's budget (the credit
        // then builds up over a few blocks). The cap keeps a wait from
        // turning into a burst.
        const double budget = lpDesignRate_ * static_cast<double>(nS);
        const double maxStep = std::max(lpCostMagChunk_, std::max(lpCostDesignFftStep_, lpCostPartitionStep_));
        lpDesignCredit_ = std::min(lpDesignCredit_ + budget, budget + maxStep);
        while (lpDesignActive_)
        {
            // The partition work in flight may still read the set the design
            // writes (right after a swap); the design then waits for it.
            if (lpDesignSet_ == lpTaskSet_ && lpTaskUnit_ < lpTaskUnits_) break;
            // Nor the previous design while a fade from it is in flight or still to start.
            if (lpDesignSet_ == lpTaskFadeSet_ && lpTaskUnit_ < lpTaskUnits_) break;
            if (lpLaunchedSet_ >= 0 && lpLaunchedSet_ != lpActiveSet_ && lpDesignSet_ == lpLaunchedSet_) break;
            if (linearPhaseDesignStepCost() > lpDesignCredit_) break;
            lpDesignCredit_ -= linearPhaseDesignStep();
            if (!lpDesignActive_ && lpDesignRedo_) startLinearPhaseDesign();
        }
        if (!lpDesignActive_) lpDesignCredit_ = 0.0;
    }

    // -- Linear-phase engine ---------------------------------------------------
    //
    // Input partition i (lpPart_ samples) is staged into a history ring. When
    // it is complete, the work for it starts: per channel, the transform of
    // the frame [partition i - 1, partition i] into the frequency-domain delay
    // line; per convolved band, the partition products and the inverse
    // transform, whose last half is the band's output partition y_i; then the
    // residual band, the input delayed by the latency minus the other bands.
    // y_i is heard kLpLeadBlocks partitions later, so the work has the whole
    // of partition i + 1 to run: it is split into units and run in
    // proportion to the samples of partition i + 1 staged so far.

    inline T* lpRingOf(int ch) noexcept
    {
        return lpRing_.data() + static_cast<size_t>(ch) * static_cast<size_t>(kLpRingBlocks * lpPart_);
    }
    inline T* lpOutOf(int ch, int buf, int band) noexcept
    {
        return lpOut_.data()
               + ((static_cast<size_t>(ch) * 2 + static_cast<size_t>(buf)) * MaxBands + static_cast<size_t>(band))
                     * static_cast<size_t>(lpPart_);
    }

    /// Units of the partition work for one channel: the forward transform,
    /// kLpPartitions products and an inverse transform per convolved band,
    /// then the residual band.
    [[nodiscard]] int lpUnitsPerChannel() const noexcept
    {
        const int fading = lpTaskFadeSet_ >= 0 ? lpTaskFadeBands_ - 1 : 0;
        return 2 + (lpTaskBands_ - 1 + fading) * (kLpPartitions + 1);
    }

    /// Starts the work for the input partition just staged (ring block `block`).
    void launchLinearPhaseTask(int block) noexcept
    {
        // The first partition on a new design also runs the previous one,
        // and each band goes from that design to the new over the partition.
        lpTaskFadeSet_ = (lpLaunchedSet_ >= 0 && lpLaunchedSet_ != lpActiveSet_) ? lpLaunchedSet_ : -1;
        lpTaskFadeBands_ = lpTaskFadeSet_ >= 0 ? std::max(2, lpSetBands_[static_cast<size_t>(lpTaskFadeSet_)]) : 0;
        lpLaunchedSet_ = lpActiveSet_;
        lpTaskSet_ = lpActiveSet_;
        lpTaskBands_ = std::max(2, lpSetBands_[static_cast<size_t>(lpActiveSet_)]);
        lpTaskBlock_ = block;
        lpTaskFdl_ = (lpTaskFdl_ + 1) % kLpPartitions;
        lpTaskUnit_ = 0;
        lpTaskUnits_ = numChannels_ * lpUnitsPerChannel();
        lpTaskDone_ = 0.0;
        lpTaskTotal_ = lpPartitionCost(lpTaskBands_, numChannels_, lpSetParts_[static_cast<size_t>(lpActiveSet_)]);
        if (lpTaskFadeSet_ >= 0)
            lpTaskTotal_ += static_cast<double>(numChannels_)
                            * (static_cast<double>(lpSetParts_[static_cast<size_t>(lpTaskFadeSet_)]) * lpCostCma_
                               + static_cast<double>(lpTaskFadeBands_ - 1) * lpCostIfft_
                               + static_cast<double>(lpTaskFadeBands_ + lpTaskBands_) * lpCostResidualPerBand_);
    }

    /// Runs one unit of the partition work and returns its cost (units).
    double runLinearPhaseUnit(int unit) noexcept
    {
        const int perCh = lpUnitsPerChannel();
        const int ch = unit / perCh;
        const int u = unit - ch * perCh;
        const int N = lpPart_;
        const size_t specLen = static_cast<size_t>(2 * N + 2);
        T* fdl = lpFdl_.data() + static_cast<size_t>(ch) * static_cast<size_t>(kLpPartitions) * specLen;
        const int next = 1 - lpPlay_;

        if (u == 0)
        {
            // Overlap-save frame [partition i - 1, partition i].
            const T* ring = lpRingOf(ch);
            const int prev = (lpTaskBlock_ + kLpRingBlocks - 1) & (kLpRingBlocks - 1);
            std::copy_n(ring + static_cast<size_t>(prev) * static_cast<size_t>(N), N, lpScratch_.begin());
            std::copy_n(ring + static_cast<size_t>(lpTaskBlock_) * static_cast<size_t>(N), N, lpScratch_.begin() + N);
            lpFft_->forward(lpScratch_.data(), fdl + static_cast<size_t>(lpTaskFdl_) * specLen);
            return lpCostFft_;
        }

        const int conv = lpTaskBands_ - 1;
        const int delayed = (lpTaskBlock_ + kLpRingBlocks - kLpPartitions / 2) & (kLpRingBlocks - 1);
        const T* delayedInput = lpRingOf(ch) + static_cast<size_t>(delayed) * static_cast<size_t>(N);
        if (u == perCh - 1)
        {
            if (lpTaskFadeSet_ >= 0) blendLinearPhaseFade(ch, next, delayedInput);
            // Residual band: the input delayed by the latency, minus the
            // convolved bands, so the bands sum to the delayed input exactly.
            T* dst = lpOutOf(ch, next, conv);
            std::copy_n(delayedInput, N, dst);
            for (int b = 0; b < conv; ++b)
                simd::addWithGainT(dst, lpOutOf(ch, next, b), T(-1), N); // exact: dst - src
            return static_cast<double>(lpTaskBands_) * lpCostResidualPerBand_;
        }

        // The bands of the design (then, in a fade, those of the previous one).
        int band = (u - 1) / (kLpPartitions + 1);
        const int step = (u - 1) - band * (kLpPartitions + 1);
        const bool previous = band >= conv;
        const int set = previous ? lpTaskFadeSet_ : lpTaskSet_;
        if (previous) band -= conv;
        if (step < kLpPartitions)
        {
            if (step == 0) std::fill(lpAccum_.begin(), lpAccum_.end(), T(0));
            if (lpPartOn_[static_cast<size_t>(set * kMaxSplits * kLpPartitions + band * kLpPartitions + step)] == 0)
                return 0.0; // a dropped (negligible) partition
            const T* kernel = lpKernelSet(set) + static_cast<size_t>(band * kLpPartitions + step) * specLen;
            const int slot = (lpTaskFdl_ - step + kLpPartitions) % kLpPartitions;
            simd::complexMulAccum(lpAccum_.data(), fdl + static_cast<size_t>(slot) * specLen, kernel, N + 1);
            return lpCostCma_;
        }
        lpFft_->inverse(lpAccum_.data(), lpResult_.data());
        T* to = previous ? lpFadeOld_.data() + static_cast<size_t>(band) * static_cast<size_t>(N) : lpOutOf(ch, next, band);
        std::copy_n(lpResult_.begin() + N, N, to);
        return lpCostIfft_;
    }

    /**
     * The fade of one channel's partition from the previous design to the
     * new one. The previous design's bands (its residual made here) are
     * read in the new band count: missing bands silent, extra ones folded
     * into the last band. Each convolved band of the new design then moves
     * from that version to its own along the partition; the residual made
     * after this keeps the sum on the delayed input, exactly.
     */
    void blendLinearPhaseFade(int ch, int next, const T* delayedInput) noexcept
    {
        const int N = lpPart_;
        const int before = lpTaskFadeBands_;
        const int conv = lpTaskBands_ - 1;
        auto old = [&](int b) { return lpFadeOld_.data() + static_cast<size_t>(b) * static_cast<size_t>(N); };
        T* oldResidual = old(before - 1);
        std::copy_n(delayedInput, N, oldResidual);
        for (int b = 0; b < before - 1; ++b) simd::addWithGainT(oldResidual, old(b), T(-1), N);
        const T invN = T(1) / static_cast<T>(N);
        for (int b = 0; b < conv; ++b)
        {
            T* out = lpOutOf(ch, next, b);
            const T* from = b < before ? old(b) : nullptr;
            for (int i = 0; i < N; ++i)
            {
                const T was = from != nullptr ? from[i] : T(0);
                out[i] = was + (out[i] - was) * (static_cast<T>(i + 1) * invN);
            }
        }
    }

    /// Runs the partition work in flight up to `staged` samples of its
    /// window (lpPart_ runs it whole).
    void runLinearPhaseTask(int staged) noexcept
    {
        if (lpTaskSet_ < 0) return;
        const double target = (staged >= lpPart_)
                                  ? lpTaskTotal_ + 1.0
                                  : lpTaskTotal_ * static_cast<double>(staged) / static_cast<double>(lpPart_);
        while (lpTaskUnit_ < lpTaskUnits_ && lpTaskDone_ < target)
            lpTaskDone_ += runLinearPhaseUnit(lpTaskUnit_++);
    }

    void processLinearPhase(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nS,
                            int nCh) noexcept
    {
        advanceLinearPhaseDesign(nS);

        const int N = lpPart_;
        int done = 0;
        while (done < nS)
        {
            const int run = std::min(nS - done, N - lpPos_);
            // The output partition was computed with lpOutBands_ bands. Until
            // a band-count change lands, missing bands are silent and extra
            // ones fold into the last requested band, so the sum stays exact.
            const int have = lpOutBands_;
            const size_t stage = static_cast<size_t>(lpStageBlock_) * static_cast<size_t>(N)
                                 + static_cast<size_t>(lpPos_);
            for (int ch = 0; ch < numChannels_; ++ch)
            {
                // Stage the input first (the outputs may alias it). Every
                // prepared channel advances, so the shared delay-line slot
                // stays coherent when a later block has more channels.
                T* dst = lpRingOf(ch) + stage;
                if (ch >= nCh)
                {
                    std::fill_n(dst, run, T(0));
                    continue;
                }
                const T* src = input.getChannel(ch) + done;
                for (int i = 0; i < run; ++i)
                    dst[i] = std::isfinite(src[i]) ? src[i] : T(0); // never latch a NaN
                for (int b = 0; b < numBands; ++b)
                {
                    T* out = outputs[b].getChannel(ch) + done;
                    if (b < have) std::copy_n(lpOutOf(ch, lpPlay_, b) + lpPos_, run, out);
                    else std::fill_n(out, run, T(0));
                }
                for (int b = numBands; b < have; ++b)
                {
                    simd::addT(outputs[numBands - 1].getChannel(ch) + done, lpOutOf(ch, lpPlay_, b) + lpPos_, run);
                }
            }
            lpPos_ += run;
            done += run;
            runLinearPhaseTask(lpPos_);

            if (lpPos_ == N)
            {
                // The work in flight is complete (its window has ended): its
                // partition plays next, and the work for the partition just
                // staged starts.
                if (lpTaskSet_ >= 0)
                {
                    lpPlay_ = 1 - lpPlay_;
                    lpOutBands_ = lpTaskBands_;
                }
                launchLinearPhaseTask(lpStageBlock_);
                lpStageBlock_ = (lpStageBlock_ + 1) & (kLpRingBlocks - 1);
                lpPos_ = 0;
            }
        }
    }

    // -- Members -------------------------------------------------------------

    AudioSpec spec_ {};
    bool prepared_ = false;
    std::atomic<int> numBands_ { 2 };
    std::atomic<int> order_ { 24 };
    int numChannels_ = 2;
    std::atomic<FilterMode> filterMode_ { FilterMode::MinimumPhase };
    std::atomic<bool> dirty_ { true };

    std::array<std::atomic<T>, kMaxSplits> targetFrequencies_ {};
    std::atomic<bool> freqUpdatePending_ { false };

    std::array<T, kMaxSplits> frequencies_ {};
    std::array<SmoothedValue<T>, kMaxSplits> freqSmoothers_;

    int numStagesPerFilter_ = 2;
    int numStagesAllpass_ = 1;
    FilterMode lastMode_ = FilterMode::MinimumPhase;

    std::array<std::unique_ptr<IirBank>, 2> banks_;   ///< The sections (heap, from prepare()).
    int bank_ = 0;        ///< The bank that plays.
    int fadeBank_ = -1;   ///< The bank a two-bank fade brings in (-1: none).
    std::vector<T> workBuf_;

    // The same sections as normalized ladders (setStructure()): per split the
    // low then the high branch (kMaxStagesPerFilter each), then two per
    // phase-correction chain [band * kMaxSplits + split]. Heap, from prepare().
    std::vector<double> ladderTmp_;   ///< One channel of a range, in double.
    std::atomic<Structure> structure_ { Structure::DirectForm };
    std::atomic<Topology> topology_ { Topology::Cascade };
    bool ladderActive_ = false;       ///< The audio thread runs the ladders.

    // Linear-phase state (uniformly partitioned convolution)
    std::unique_ptr<FFTReal<T>> lpFft_;        // partition transforms (2 x lpPart_); the engine gate
    std::unique_ptr<FFTReal<T>> lpDesignFft_;  // kernel design (2 x firLength_)
    int firLength_ = 0;
    int lpPart_ = 0;
    int lpLatency_ = 0;
    int lpPos_ = 0;        // samples staged in the current partition
    int lpStageBlock_ = 0; // history-ring block being staged
    int lpPlay_ = 0;       // output buffer that plays (the other is being computed)
    int lpOutBands_ = 0;   // bands in the output partition being emitted
    int lpActiveSet_ = 0;  // kernel set that new partition work uses (audio thread)
    std::array<int, 2> lpSetBands_ { 0, 0 }; // band count of each kernel set
    std::array<int, 2> lpSetParts_ { 0, 0 }; // convolved partitions of each kernel set (all bands)
    // Partition work in flight (audio thread).
    int lpTaskSet_ = -1;   // kernel set it reads (-1: none in flight)
    int lpTaskBands_ = 2;
    int lpTaskBlock_ = 0;  // history-ring block of its input partition
    int lpTaskFdl_ = 0;    // delay-line slot of its input spectrum
    int lpTaskUnit_ = 0;
    int lpTaskUnits_ = 0;
    double lpTaskDone_ = 0.0;
    double lpTaskTotal_ = 0.0;
    int lpLaunchedSet_ = -1;   // kernel set of the last partition work launched
    int lpTaskFadeSet_ = -1;   // previous set the work in flight fades from (-1: none)
    int lpTaskFadeBands_ = 0;
    // Cost model (units), set in prepare().
    double lpCostFft_ = 0.0, lpCostCma_ = 0.0, lpCostIfft_ = 0.0, lpCostResidualPerBand_ = 0.0;
    double lpCostMagChunk_ = 0.0, lpCostDesignFftStep_ = 0.0, lpCostPartitionStep_ = 0.0;
    // Design step machine (audio thread; prepare() runs it whole).
    bool lpDesignActive_ = false;
    bool lpDesignRedo_ = false;   // a change arrived during the design
    int lpDesignSet_ = 1;         // kernel set the design writes
    int lpDesignBand_ = 0;
    int lpDesignStep_ = 0;        // magnitude chunks, inverse FFT passes, partitions
    int lpDesignBands_ = 0;
    int lpDesignParts_ = 0;       // partitions the design keeps so far
    int lpDesignExpo_ = 4;
    int lpDesignFftSteps_ = 1;
    std::array<double, kMaxSplits> lpDesignFc_ {};
    double lpDesignCredit_ = 0.0;
    double lpDesignRate_ = 0.0;   // design units per sample
    int glideRemain_ = 0;  // samples left in the current IIR glide segment
    // The minimum-phase engine's band count and the fade of a change.
    int iirBands_ = 0;     // bands of the topology heading to (0: none yet since a reset)
    int fadeFrom_ = 0;     // bands of the topology fading out
    Topology iirTopo_ = Topology::Cascade;      // the topology heading to
    Topology fadeFromTopo_ = Topology::Cascade; // the topology fading out
    int fadeLeft_ = 0;     // samples left of the fade
    int fadeLength_ = 960;
    std::vector<T> fadeBands_, fadeLower_, fadeTop_;   // [band][ch][kFadeChunk] scratch of the fade
    std::atomic<bool> lpMagDirty_ { true };

    std::vector<T> lpKernelSpec_;     // [set][(band * partitions + part) * specLen] kernel partition spectra
    std::vector<uint8_t> lpPartOn_;   // [set][band * partitions + part] 1 = convolved, 0 = negligible
    std::vector<T> lpFdl_;            // [(ch * partitions + slot) * specLen] input spectra
    std::vector<T> lpRing_;           // [ch][ring block][lpPart_] input history
    std::vector<T> lpOut_;            // [((ch * 2 + buffer) * MaxBands + band) * lpPart_] output partitions
    std::vector<T> lpAccum_, lpResult_, lpScratch_;
    std::vector<T> lpFadeOld_;        // [band][lpPart_] one channel's previous-design bands in a fade
    std::vector<double> lpCarry_;     // [bin] product of the high-passes below the band being designed
    std::vector<double> lpBinFreq_;   // [bin] design grid frequencies
    std::vector<T> lpMag_;            // [bin] magnitudes of the band being designed
    std::vector<T> lpWindow_;         // [tap] flat-top window
    std::vector<T> lpTimeResponse_;   // design IFFT input scratch (2 x firLength_ + 2)
    std::vector<T> lpFirKernel_;      // design IFFT output scratch (2 x firLength_)
};

} // namespace dspark
