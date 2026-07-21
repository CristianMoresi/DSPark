// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file CrossoverFilter.h
 * @brief Linkwitz-Riley crossover filter for multi-band audio processing.
 *
 * Splits an audio signal into 2-12 frequency bands using Linkwitz-Riley
 * crossover filters. Supports three slope options (LR12, LR24, LR48) and
 * two processing modes: minimum-phase IIR with allpass phase correction,
 * and linear-phase FFT-based processing with zero phase distortion.
 *
 * Linkwitz-Riley crossovers guarantee that all bands sum to an allpass
 * (flat magnitude response) at every frequency, making them the standard
 * for professional multi-band processing. In the IIR tree, band b passes
 * through one allpass section per split above it (LR12/LR24) or two
 * sections (LR48) so the phase of every branch matches at the summing
 * point. LR12 additionally bakes a polarity inversion into the high-pass
 * branch of each split (the textbook LR2 convention): odd-numbered bands
 * come out polarity-inverted so the tree sums flat instead of notching
 * at every crossover. The linear-phase path needs neither (zero-phase
 * magnitudes sum to unity by construction).
 *
 * The linear-phase FIR length equals the prepared maxBlockSize, so its
 * low-frequency resolution scales with the block size: very low crossover
 * frequencies need large blocks to be resolved. Kernel recomputation after
 * a frequency/order/band-count change runs on the audio thread from
 * pre-allocated scratch (no allocation, but it is a CPU spike of
 * 2 x numBands FFTs for that block).
 *
 * Threading model:
 * - prepare() / reset() / getState() / setState(): setup or UI threads only
 *   (prepare and getState allocate), never concurrently with processBlock().
 * - setNumBands / setCrossoverFrequency / setOrder / setFilterMode: RT-safe
 *   atomic publications from any thread; the audio thread applies them at
 *   the next block. Frequency changes are smoothed (5 ms) in IIR mode and
 *   applied instantly in linear-phase mode (kernels are rebuilt per change).
 * - getLatency() and the getters are safe from any thread.
 *
 * Dependencies: Biquad.h, AudioBuffer.h, AudioSpec.h, DspMath.h, FFT.h,
 *               DenormalGuard.h, SmoothedValue.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/DenormalGuard.h"
#include "../Core/FFT.h"
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
 * @brief Linkwitz-Riley crossover with 2-12 bands, LR12/LR24/LR48.
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
     * works) unless maxBlockSize exceeds 2^18, in which case linear-phase
     * mode falls back to the IIR path and getLatency() reports 0.
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

        // Linear-phase FFT resources. Allocated regardless of the current mode
        // so setFilterMode(LinearPhase) after prepare() actually engages
        // (mode-dependent allocation silently fell back to IIR). lpFft_ acts
        // as the gate and is created last: if any allocation throws, the
        // engine stays unavailable but coherent.
        lpFft_.reset();
        lpFftSize_ = 0;
        firLength_ = 0;
        lpLatency_ = 0;
        if (spec.maxBlockSize <= kLpMaxBlockSize)
        {
            // For Overlap-Save, FFT size must be >= BlockSize + FIR_Length - 1
            // We choose FIR_Length = maxBlockSize. Thus FFT >= 2 * maxBlockSize - 1.
            // Floor at 4: FFTReal requires a power of two >= 4 (it throws
            // below that, which a prepare with maxBlockSize 1 used to hit).
            int fftPow2 = 4;
            while (fftPow2 < spec.maxBlockSize * 2) fftPow2 <<= 1;
            lpFftSize_ = fftPow2;
            firLength_ = spec.maxBlockSize; // Use block size as FIR length for good resolution
            // The kernel is centred at firLength/2 (the circular shift uses
            // i - halfLen), so the exact group delay is firLength/2 - the old
            // (firLength-1)/2 under-reported PDC by one sample for even sizes.
            lpLatency_ = firLength_ / 2;

            const int numBins = lpFftSize_ / 2 + 1;

            lpMagnitudesFlat_.assign(static_cast<size_t>(MaxBands) * static_cast<size_t>(numBins) * 2, T(0));
            lpPrevBlockFlat_.assign(static_cast<size_t>(spec.numChannels) * static_cast<size_t>(firLength_), T(0));

            lpFftIn_.assign(static_cast<size_t>(lpFftSize_), T(0));
            lpFftOut_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpBandFft_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpFftResult_.assign(static_cast<size_t>(lpFftSize_), T(0));

            // Pre-allocate recompute scratch so a live crossover/order change never
            // allocates on the audio thread (recomputeLinearPhaseMagnitudes runs there).
            lpIdealMagsFlat_.assign(static_cast<size_t>(MaxBands) * static_cast<size_t>(numBins), T(0));
            lpTimeResponse_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpFirKernel_.assign(static_cast<size_t>(lpFftSize_), T(0));

            lpFft_ = std::make_unique<FFTReal<T>>(lpFftSize_);
        }

        // Allocate flat allpass correction chains
        // Access via: band * kMaxSplits + split
        allpassFlat_.resize(static_cast<size_t>(MaxBands * kMaxSplits));

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

        lastNumSplits_ = 0;
        lastMode_ = filterMode_.load(std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
        lpMagDirty_.store(true, std::memory_order_relaxed);
        reset();

        prepared_ = true;
    }

    /**
     * @brief Splits input into separate band outputs.
     *
     * The processed span is clamped to the prepared maxBlockSize and to the
     * shortest output view. Input channels beyond the prepared/16-channel
     * limit are passed through on band 0 (remaining bands silent) so the
     * band sum still reconstructs the input.
     *
     * @param input        Input audio buffer.
     * @param bandOutputs  Array of AudioBufferView, one per band.
     * @param numOutputBands Number of output bands (must match getNumBands();
     *                       fewer bands re-purpose the first splits only in
     *                       IIR mode and is not a supported configuration).
     */
    void processBlock(AudioBufferView<T> input,
                      AudioBufferView<T>* bandOutputs, int numOutputBands) noexcept
    {
        if (!prepared_ || bandOutputs == nullptr) return;

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
                // FIR kernels are rebuilt whole per change: apply instantly
                // (per-sample smoothing cannot apply to a kernel rebuild).
                syncFrequenciesToTargets();
            }
            else
            {
                for (int i = 0; i < kMaxSplits; ++i)
                    freqSmoothers_[i].setTargetValue(targetFrequencies_[i].load(std::memory_order_relaxed));
            }
        }

        if (dirty_.load(std::memory_order_relaxed) &&
            dirty_.exchange(false, std::memory_order_acquire))
        {
            updateCoefficients();
        }

        const int n = std::min(numOutputBands, numBands_.load(std::memory_order_relaxed));
        if (n < 2) return;

        // Defensive span clamp: never index past the prepared work buffers
        // (fixed maxBlockSize stride) or past any output view.
        int nS = std::min(input.getNumSamples(), spec_.maxBlockSize);
        for (int b = 0; b < n; ++b)
            nS = std::min(nS, bandOutputs[b].getNumSamples());
        if (nS <= 0) return;

        if (lpActive)
            processLinearPhase(input, bandOutputs, n, nS);
        else
            processIIR(input, bandOutputs, n, nS);

        passExtraChannels(input, bandOutputs, n, nS);
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Sets the number of output bands (2..MaxBands).
     *
     * @note Changing the band count re-initialises ALL split frequencies to
     *       log-spaced defaults (100 Hz .. 10 kHz); reapply custom frequencies
     *       afterwards. A live topology change is applied instantly (click).
     */
    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 2, MaxBands), std::memory_order_relaxed);
        initDefaultFrequencies();
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

            // Sort on the UI thread to avoid Audio Thread priority inversion/CPU spikes
            int activeSplits = numBands_.load(std::memory_order_relaxed) - 1;
            std::sort(localTargets.begin(), localTargets.begin() + activeSplits);

            for (int i = 0; i < kMaxSplits; ++i)
                targetFrequencies_[i].store(localTargets[static_cast<size_t>(i)], std::memory_order_relaxed);

            freqUpdatePending_.store(true, std::memory_order_release);
            dirty_.store(true, std::memory_order_release);
            lpMagDirty_.store(true, std::memory_order_release);
        }
    }

    /** @brief Sets the crossover slope: 12, 24 or 48 (dB/oct). Other values are ignored. */
    void setOrder(int order) noexcept
    {
        if (order == 12 || order == 24 || order == 48)
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
     * @brief Returns latency in samples (0 for IIR, FIR group delay for linear-phase).
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
        for (auto& sp : splits_)
        {
            for (auto& b : sp.lp) b.reset();
            for (auto& b : sp.hp) b.reset();
        }
        for (auto& apChain : allpassFlat_)
            for (auto& b : apChain.stages) b.reset();

        std::fill(lpPrevBlockFlat_.begin(), lpPrevBlockFlat_.end(), T(0));
    }


    /** @brief Serializes the split topology (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("XOVR"), 1);
        const int n = getNumBands();
        w.write("numBands", n);
        w.write("order", getOrder());
        w.write("mode", static_cast<int32_t>(getFilterMode()));
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
    static constexpr int kLpMaxBlockSize = 1 << 18; ///< Above this, no linear-phase engine (IIR fallback).

    struct SplitPoint
    {
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> lp;
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> hp;
    };

    struct AllPassChain
    {
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> stages;
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
    [[nodiscard]] static BiquadCoeffs<T> makeFirstOrderAllPass(double sampleRate, double freq) noexcept
    {
        freq = std::clamp(freq, 1.0, std::max(1.0, sampleRate * 0.499));
        const double w = std::tan(std::numbers::pi * freq / sampleRate);
        const double a = (w - 1.0) / (w + 1.0);
        BiquadCoeffs<T> coeffs;
        coeffs.b0 = static_cast<T>(a);
        coeffs.b1 = static_cast<T>(1.0);
        coeffs.b2 = T(0);
        coeffs.a1 = static_cast<T>(a);
        coeffs.a2 = T(0);
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

    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate <= 0) return;

        int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        double sr = spec_.sampleRate;

        // Splits (and their allpass chains) re-activated by a band count
        // increase would otherwise replay arbitrarily old filter history.
        if (numSplits > lastNumSplits_)
        {
            for (int s = lastNumSplits_; s < numSplits; ++s)
            {
                for (auto& b : splits_[s].lp) b.reset();
                for (auto& b : splits_[s].hp) b.reset();
                for (int b = 0; b < s; ++b)
                    for (auto& st : allpassFlat_[static_cast<size_t>(b * kMaxSplits + s)].stages) st.reset();
            }
        }
        lastNumSplits_ = numSplits;

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
            case 12:
                numStagesPerFilter_ = 2;
                numStagesAllpass_   = 1;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                    auto lpC = BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f);
                    auto hpC = BiquadCoeffs<T>::makeFirstOrderHighPass(sr, f);
                    auto apC = makeFirstOrderAllPass(sr, f);

                    // LR12 sums flat only with the high branch polarity
                    // inverted (LP^2 - HP^2 = allpass; LP^2 + HP^2 notches at
                    // fc). Bake the inversion into the first HP stage: odd
                    // bands come out inverted, the sum is allpass-flat.
                    auto hpC0 = hpC;
                    hpC0.b0 = -hpC0.b0;
                    hpC0.b1 = -hpC0.b1;
                    hpC0.b2 = -hpC0.b2;

                    splits_[s].lp[0].setCoeffs(lpC);
                    splits_[s].lp[1].setCoeffs(lpC);
                    splits_[s].hp[0].setCoeffs(hpC0);
                    splits_[s].hp[1].setCoeffs(hpC);
                    for (int b = 0; b < s; ++b)
                        allpassFlat_[static_cast<size_t>(b * kMaxSplits + s)].stages[0].setCoeffs(apC);
                }
                break;

            case 24:
                numStagesPerFilter_ = 2;
                numStagesAllpass_   = 1;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                    auto lpC = BiquadCoeffs<T>::makeLowPass(sr, f, 0.7071);
                    auto hpC = BiquadCoeffs<T>::makeHighPass(sr, f, 0.7071);
                    auto apC = BiquadCoeffs<T>::makeAllPass(sr, f, 0.7071);

                    for (int st = 0; st < 2; ++st)
                    {
                        splits_[s].lp[st].setCoeffs(lpC);
                        splits_[s].hp[st].setCoeffs(hpC);
                    }
                    for (int b = 0; b < s; ++b)
                        allpassFlat_[static_cast<size_t>(b * kMaxSplits + s)].stages[0].setCoeffs(apC);
                }
                break;

            case 48:
            {
                numStagesPerFilter_ = 4;
                numStagesAllpass_   = 2;
                constexpr double q1 = 0.5412;
                constexpr double q2 = 1.3066;
                const double qArr[4] = { q1, q2, q1, q2 };

                for (int s = 0; s < numSplits; ++s)
                {
                    double f = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);

                    for (int st = 0; st < 4; ++st)
                    {
                        splits_[s].lp[st].setCoeffs(BiquadCoeffs<T>::makeLowPass(sr, f, qArr[st]));
                        splits_[s].hp[st].setCoeffs(BiquadCoeffs<T>::makeHighPass(sr, f, qArr[st]));
                    }
                    for (int b = 0; b < s; ++b)
                    {
                        auto& chain = allpassFlat_[static_cast<size_t>(b * kMaxSplits + s)];
                        chain.stages[0].setCoeffs(BiquadCoeffs<T>::makeAllPass(sr, f, q1));
                        chain.stages[1].setCoeffs(BiquadCoeffs<T>::makeAllPass(sr, f, q2));
                    }
                }
                break;
            }

            default: break;
        }
    }

    void processIIR(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nS) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(input.getNumChannels(), numChannels_);
        const int numSplits = numBands - 1;

        bool anySmoothing = false;
        for (int i = 0; i < numSplits; ++i)
            anySmoothing = anySmoothing || freqSmoothers_[i].isSmoothing();

        if (anySmoothing)
        {
            constexpr int kSubBlockSize = 32;
            int offset = 0;
            while (offset < nS)
            {
                int blockLen = std::min(kSubBlockSize, nS - offset);
                for (int i = 0; i < numSplits; ++i)
                {
                    for (int s = 0; s < blockLen; ++s)
                        (void)freqSmoothers_[i].getNextValue();
                    frequencies_[i] = freqSmoothers_[i].getCurrentValue();
                }
                updateCoefficients();
                processIIRRange(input, outputs, numBands, nCh, offset, blockLen);
                offset += blockLen;
            }
        }
        else
        {
            processIIRRange(input, outputs, numBands, nCh, 0, nS);
        }
    }

    inline T* getWorkBufChannel(int ch) noexcept
    {
        return workBuf_.data() + static_cast<size_t>(ch) * static_cast<size_t>(spec_.maxBlockSize);
    }

    void processIIRRange(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nCh, int offset, int blockLen) noexcept
    {
        const int numSplits = numBands - 1;

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = input.getChannel(ch) + offset;
            T* dst = getWorkBufChannel(ch);
            std::copy(src, src + blockLen, dst);
        }

        for (int s = 0; s < numSplits; ++s)
        {
            for (int i = 0; i < blockLen; ++i)
            {
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* workCh = getWorkBufChannel(ch);
                    T sample = workCh[i];

                    T lpSample = sample;
                    for (int st = 0; st < numStagesPerFilter_; ++st)
                        lpSample = splits_[s].lp[st].processSample(lpSample, ch);
                    outputs[s].getChannel(ch)[offset + i] = lpSample;

                    T hpSample = sample;
                    for (int st = 0; st < numStagesPerFilter_; ++st)
                        hpSample = splits_[s].hp[st].processSample(hpSample, ch);
                    workCh[i] = hpSample;
                }
            }
        }

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* dst = outputs[numBands - 1].getChannel(ch) + offset;
            const T* src = getWorkBufChannel(ch);
            std::copy(src, src + blockLen, dst);
        }

        for (int s = 1; s < numSplits; ++s)
        {
            for (int b = 0; b < s; ++b)
            {
                for (int i = 0; i < blockLen; ++i)
                {
                    for (int ch = 0; ch < nCh; ++ch)
                    {
                        T sample = outputs[b].getChannel(ch)[offset + i];
                        for (int st = 0; st < numStagesAllpass_; ++st)
                            sample = allpassFlat_[static_cast<size_t>(b * kMaxSplits + s)].stages[st].processSample(sample, ch);
                        outputs[b].getChannel(ch)[offset + i] = sample;
                    }
                }
            }
        }
    }

    /// Input channels beyond the processed count get band 0 = input,
    /// remaining bands = silence, so the band sum still equals the input
    /// (leaving caller buffers untouched would hand back garbage).
    void passExtraChannels(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nS) noexcept
    {
        const int inCh = input.getNumChannels();
        for (int ch = numChannels_; ch < inCh; ++ch)
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

    void recomputeLinearPhaseMagnitudes() noexcept
    {
        if (lpFftSize_ == 0) return;
        if (!lpMagDirty_.load(std::memory_order_relaxed) ||
            !lpMagDirty_.exchange(false, std::memory_order_acquire))
            return;

        const int numBins = lpFftSize_ / 2 + 1;
        const int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        const double sr = spec_.sampleRate;

        int expo = 0;
        switch (order_.load(std::memory_order_relaxed))
        {
            case 12: expo = 2; break;
            case 24: expo = 4; break;
            case 48: expo = 8; break;
        }

        // Ideal zero-phase magnitudes go into the pre-allocated flat scratch
        // lpIdealMagsFlat_[band * numBins + bin] - no audio-thread allocation.
        const int nBands = numBands_.load(std::memory_order_relaxed);

        for (int k = 0; k < numBins; ++k)
        {
            double freq = sr * static_cast<double>(k) / static_cast<double>(lpFftSize_);
            T lpMag[kMaxSplits], hpMag[kMaxSplits];

            for (int s = 0; s < numSplits; ++s)
            {
                double fc = clampSplitFreq(static_cast<double>(frequencies_[s]), sr);
                double ratio = freq / fc;
                // Ideal Linkwitz-Riley magnitude |LP| = 1/(1 + (f/fc)^N) with
                // N = expo = order/6 (2/4/8 -> 12/24/48 dB/oct). This must match the
                // IIR path's slope; the exponent was previously expo*2, which made the
                // linear-phase crossover twice as steep as the selected order.
                double rPow = std::pow(ratio, static_cast<double>(expo));

                double denom = 1.0 + rPow;
                lpMag[s] = static_cast<T>(1.0 / denom);
                hpMag[s] = static_cast<T>(rPow / denom);
            }

            lpIdealMagsFlat_[k] = lpMag[0];                              // band 0 (lowpass)
            for (int b = 1; b < nBands - 1; ++b)
                lpIdealMagsFlat_[static_cast<size_t>(b * numBins + k)] = hpMag[b - 1] * lpMag[b];
            lpIdealMagsFlat_[static_cast<size_t>((nBands - 1) * numBins + k)] = hpMag[numSplits - 1];
        }

        // Window Design & FIR Kernel generation (pre-allocated scratch)
        for (int b = 0; b < nBands; ++b)
        {
            // Prepare complex bins for inverse FFT (zero phase)
            for (int k = 0; k < numBins; ++k)
            {
                lpTimeResponse_[static_cast<size_t>(2 * k)] = lpIdealMagsFlat_[static_cast<size_t>(b * numBins + k)];
                lpTimeResponse_[static_cast<size_t>(2 * k + 1)] = T(0);
            }

            lpFft_->inverse(lpTimeResponse_.data(), lpFirKernel_.data());

            // Circular shift, windowing (Blackman), and zero-padding
            std::fill(lpFftIn_.begin(), lpFftIn_.end(), T(0));
            int halfLen = firLength_ / 2;
            const double N = static_cast<double>(firLength_ - 1);

            for (int i = 0; i < firLength_; ++i)
            {
                // Un-wrap circular time response to causal shifted response
                int srcIdx = (i - halfLen + lpFftSize_) % lpFftSize_;

                // Blackman Window (degenerate 1-tap kernel: unity)
                double window = 1.0;
                if (N >= 1.0)
                {
                    double n = static_cast<double>(i);
                    window = 0.42 - 0.5 * std::cos(2.0 * std::numbers::pi * n / N) + 0.08 * std::cos(4.0 * std::numbers::pi * n / N);
                }

                lpFftIn_[static_cast<size_t>(i)] = lpFirKernel_[static_cast<size_t>(srcIdx)] * static_cast<T>(window);
            }

            // Forward FFT to get the usable overlap-save kernel
            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());

            // Store complex spectrum contiguously
            T* bandMagData = lpMagnitudesFlat_.data() + static_cast<size_t>(b * numBins * 2);
            std::copy(lpFftOut_.begin(), lpFftOut_.begin() + (numBins * 2), bandMagData);
        }
    }

    inline T* getPrevBlockChannel(int ch) noexcept
    {
        return lpPrevBlockFlat_.data() + static_cast<size_t>(ch) * static_cast<size_t>(firLength_);
    }

    void processLinearPhase(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands, int nS) noexcept
    {
        recomputeLinearPhaseMagnitudes();

        const int nCh = std::min(input.getNumChannels(), numChannels_);
        const int numBins = lpFftSize_ / 2 + 1;
        const int overlapSize = firLength_ - 1;

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* channelData = input.getChannel(ch);
            T* prev = getPrevBlockChannel(ch);

            // Overlap-Save: [overlap block | new samples | zero pad]
            for (int i = 0; i < overlapSize; ++i)
                lpFftIn_[static_cast<size_t>(i)] = prev[i];

            for (int i = 0; i < nS; ++i)
                lpFftIn_[static_cast<size_t>(overlapSize + i)] = channelData[i];

            for (int i = overlapSize + nS; i < lpFftSize_; ++i)
                lpFftIn_[static_cast<size_t>(i)] = T(0);

            // Save tail for next block
            for (int i = 0; i < overlapSize; ++i)
            {
                if (nS - overlapSize + i >= 0)
                    prev[i] = channelData[nS - overlapSize + i];
                else
                    prev[i] = lpFftIn_[static_cast<size_t>(nS + i)]; // Handle block size < overlap
            }

            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());

            for (int b = 0; b < numBands; ++b)
            {
                const T* kernelSpectrum = lpMagnitudesFlat_.data() + static_cast<size_t>(b * numBins * 2);

                // Complex multiplication
                for (int k = 0; k < numBins; ++k)
                {
                    T r1 = lpFftOut_[static_cast<size_t>(2 * k)];
                    T i1 = lpFftOut_[static_cast<size_t>(2 * k + 1)];
                    T r2 = kernelSpectrum[2 * k];
                    T i2 = kernelSpectrum[2 * k + 1];

                    lpBandFft_[static_cast<size_t>(2 * k)]     = r1 * r2 - i1 * i2;
                    lpBandFft_[static_cast<size_t>(2 * k + 1)] = r1 * i2 + i1 * r2;
                }

                lpFft_->inverse(lpBandFft_.data(), lpFftResult_.data());

                // Discard garbage and copy valid output
                T* outCh = outputs[b].getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    outCh[i] = lpFftResult_[static_cast<size_t>(overlapSize + i)];
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
    int lastNumSplits_ = 0;
    FilterMode lastMode_ = FilterMode::MinimumPhase;
    std::array<SplitPoint, kMaxSplits> splits_ {};

    // Flattened data structures for cache locality
    std::vector<AllPassChain> allpassFlat_;
    std::vector<T> workBuf_;

    // Linear-phase state
    std::unique_ptr<FFTReal<T>> lpFft_;
    int lpFftSize_ = 0;
    int firLength_ = 0;
    int lpLatency_ = 0;
    std::atomic<bool> lpMagDirty_ { true };

    std::vector<T> lpMagnitudesFlat_; // [band * numBins * 2 + complexIdx]
    std::vector<T> lpPrevBlockFlat_;  // [ch * firLength_ + sampleIdx]
    std::vector<T> lpFftIn_, lpFftOut_, lpBandFft_, lpFftResult_;
    std::vector<T> lpIdealMagsFlat_;  // [band * numBins + bin] ideal zero-phase mags (recompute scratch)
    std::vector<T> lpTimeResponse_;   // IFFT input scratch (lpFftSize_ + 2)
    std::vector<T> lpFirKernel_;      // IFFT output scratch (lpFftSize_)
};

} // namespace dspark
