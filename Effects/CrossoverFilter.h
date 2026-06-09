// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file CrossoverFilter.h
 * @brief Linkwitz-Riley crossover filter for multi-band audio processing.
 *
 * Splits an audio signal into 2–12 frequency bands using Linkwitz-Riley
 * crossover filters. Supports three slope options (LR12, LR24, LR48) and
 * two processing modes: minimum-phase IIR with allpass phase correction,
 * and linear-phase FFT-based processing with zero phase distortion.
 *
 * Linkwitz-Riley crossovers guarantee that all bands sum to unity gain
 * (flat magnitude response) at every frequency, making them the standard
 * for professional multi-band processing.
 *
 * Dependencies: Biquad.h, AudioBuffer.h, AudioSpec.h, DspMath.h, FFT.h,
 *               DenormalGuard.h, SmoothedValue.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/DenormalGuard.h"
#include "../Core/FFT.h"
#include "../Core/SmoothedValue.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class CrossoverFilter
 * @brief Linkwitz-Riley crossover with 2–12 bands, LR12/LR24/LR48.
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
        for (auto& f : targetFrequencies_) f.store(1000.0f, std::memory_order_relaxed);
    }

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the crossover for processing.
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        // Per-split filters are Biquad<T, 16>; clamp so processing more than 16
        // channels can't index their fixed per-channel state out of bounds.
        numChannels_ = std::min(static_cast<int>(spec.numChannels), 16);

        // Allocate flat IIR work buffer (Channels * MaxBlockSize)
        workBuf_.assign(static_cast<size_t>(spec.numChannels * spec.maxBlockSize), T(0));

        // Linear-phase FFT resources
        if (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && spec.maxBlockSize > 0)
        {
            // For Overlap-Save, FFT size must be >= BlockSize + FIR_Length - 1
            // We choose FIR_Length = maxBlockSize. Thus FFT >= 2 * maxBlockSize - 1.
            int fftPow2 = 1;
            while (fftPow2 < spec.maxBlockSize * 2) fftPow2 <<= 1;
            lpFftSize_ = fftPow2;
            firLength_ = spec.maxBlockSize; // Use block size as FIR length for good resolution
            // The kernel is centred at firLength/2 (the circular shift uses
            // i - halfLen), so the exact group delay is firLength/2 — the old
            // (firLength-1)/2 under-reported PDC by one sample for even sizes.
            lpLatency_ = firLength_ / 2;

            lpFft_ = std::make_unique<FFTReal<T>>(lpFftSize_);
            int numBins = lpFftSize_ / 2 + 1;

            lpMagnitudesFlat_.assign(static_cast<size_t>(MaxBands * numBins * 2), T(0));
            lpPrevBlockFlat_.assign(static_cast<size_t>(spec.numChannels * firLength_), T(0));

            lpFftIn_.assign(static_cast<size_t>(lpFftSize_), T(0));
            lpFftOut_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpBandFft_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpFftResult_.assign(static_cast<size_t>(lpFftSize_), T(0));

            // Pre-allocate recompute scratch so a live crossover/order change never
            // allocates on the audio thread (recomputeLinearPhaseMagnitudes runs there).
            lpIdealMagsFlat_.assign(static_cast<size_t>(MaxBands * numBins), T(0));
            lpTimeResponse_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpFirKernel_.assign(static_cast<size_t>(lpFftSize_), T(0));
        }

        // Allocate flat allpass correction chains
        // Access via: band * kMaxSplits + split
        allpassFlat_.resize(static_cast<size_t>(MaxBands * kMaxSplits));

        // Initialize frequency smoothers
        initDefaultFrequencies();
        
        for (int i = 0; i < kMaxSplits; ++i)
        {
            freqSmoothers_[i].prepare(spec.sampleRate, 5.0);
            freqSmoothers_[i].setSmoothingType(SmoothedValue<T>::SmoothingType::Exponential);
            T target = targetFrequencies_[i].load(std::memory_order_relaxed);
            freqSmoothers_[i].reset(target);
            frequencies_[i] = target;
        }

        dirty_.store(true, std::memory_order_relaxed);
        lpMagDirty_.store(true, std::memory_order_relaxed);
        reset();

        prepared_ = true;
    }

    /**
     * @brief Splits input into separate band outputs.
     *
     * @param input        Input audio buffer.
     * @param bandOutputs Array of AudioBufferView, one per band.
     * @param numOutputBands Number of output bands (must match getNumBands()).
     */
    void processBlock(AudioBufferView<T> input,
                      AudioBufferView<T>* bandOutputs, int numOutputBands) noexcept
    {
        if (!prepared_) return;

        // Check if UI requested a frequency update
        if (freqUpdatePending_.exchange(false, std::memory_order_relaxed))
        {
            for (int i = 0; i < kMaxSplits; ++i)
            {
                freqSmoothers_[i].setTargetValue(targetFrequencies_[i].load(std::memory_order_relaxed));
            }
        }

        if (dirty_.load(std::memory_order_relaxed)) updateCoefficients();

        int n = std::min(numOutputBands, numBands_.load(std::memory_order_relaxed));
        if (n < 2) return;

        if (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && lpFft_)
            processLinearPhase(input, bandOutputs, n);
        else
            processIIR(input, bandOutputs, n);
    }

    // -- Configuration -------------------------------------------------------

    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 2, MaxBands), std::memory_order_relaxed);
        initDefaultFrequencies();
        dirty_.store(true, std::memory_order_relaxed);
        lpMagDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Sets a crossover frequency. Automatically maintains sorting.
     */
    void setCrossoverFrequency(int index, T freqHz) noexcept
    {
        if (index >= 0 && index < numBands_.load(std::memory_order_relaxed) - 1)
        {
            // Read all current targets into a local array to sort them
            std::array<T, kMaxSplits> localTargets;
            for (int i = 0; i < kMaxSplits; ++i)
                localTargets[i] = targetFrequencies_[i].load(std::memory_order_relaxed);
            
            localTargets[index] = freqHz;
            
            // Sort on the UI thread to avoid Audio Thread priority inversion/CPU spikes
            int activeSplits = numBands_.load(std::memory_order_relaxed) - 1;
            std::sort(localTargets.begin(), localTargets.begin() + activeSplits);

            for (int i = 0; i < kMaxSplits; ++i)
                targetFrequencies_[i].store(localTargets[i], std::memory_order_relaxed);

            freqUpdatePending_.store(true, std::memory_order_relaxed);
            dirty_.store(true, std::memory_order_relaxed);
            lpMagDirty_.store(true, std::memory_order_relaxed);
        }
    }

    void setOrder(int order) noexcept
    {
        if (order == 12 || order == 24 || order == 48)
        {
            order_.store(order, std::memory_order_relaxed);
            dirty_.store(true, std::memory_order_relaxed);
            lpMagDirty_.store(true, std::memory_order_relaxed);
        }
    }

    void setFilterMode(FilterMode mode) noexcept
    {
        filterMode_.store(mode, std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_relaxed);
        lpMagDirty_.store(true, std::memory_order_relaxed);
    }

    // -- Queries -------------------------------------------------------------

    [[nodiscard]] int getNumBands() const noexcept { return numBands_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getOrder() const noexcept { return order_.load(std::memory_order_relaxed); }
    [[nodiscard]] FilterMode getFilterMode() const noexcept { return filterMode_.load(std::memory_order_relaxed); }

    /** @brief Returns latency in samples (0 for IIR, FIR group delay for linear-phase). */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase) ? lpLatency_ : 0;
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

private:
    static constexpr int kMaxSplits = MaxBands - 1;
    static constexpr int kMaxStagesPerFilter = 4;

    struct SplitPoint
    {
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> lp;
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> hp;
    };

    struct AllPassChain
    {
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> stages;
    };

    [[nodiscard]] static BiquadCoeffs<T> makeFirstOrderAllPass(double sampleRate, double freq) noexcept
    {
        double w = std::tan(std::numbers::pi * freq / sampleRate);
        double c = (1.0 - w) / (1.0 + w);
        BiquadCoeffs<T> coeffs;
        coeffs.b0 = static_cast<T>(c);
        coeffs.b1 = static_cast<T>(1.0);
        coeffs.b2 = T(0);
        coeffs.a1 = static_cast<T>(c);
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
        freqUpdatePending_.store(true, std::memory_order_relaxed);
    }

    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate <= 0) return;
        dirty_.store(false, std::memory_order_relaxed);

        int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        double sr = spec_.sampleRate;

        // Note: Frequencies are already guaranteed to be sorted by the setter thread.
        switch (order_.load(std::memory_order_relaxed))
        {
            case 12:
                numStagesPerFilter_ = 2;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = std::clamp(static_cast<double>(frequencies_[s]), 20.0, sr * 0.499);
                    auto lpC = BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f);
                    auto hpC = BiquadCoeffs<T>::makeFirstOrderHighPass(sr, f);
                    auto apC = makeFirstOrderAllPass(sr, f);

                    for (int st = 0; st < 2; ++st)
                    {
                        splits_[s].lp[st].setCoeffs(lpC);
                        splits_[s].hp[st].setCoeffs(hpC);
                    }
                    for (int b = 0; b < s; ++b)
                        for (int st = 0; st < 2; ++st)
                            allpassFlat_[b * kMaxSplits + s].stages[st].setCoeffs(apC);
                }
                break;

            case 24:
                numStagesPerFilter_ = 2;
                for (int s = 0; s < numSplits; ++s)
                {
                    double f = std::clamp(static_cast<double>(frequencies_[s]), 20.0, sr * 0.499);
                    auto lpC = BiquadCoeffs<T>::makeLowPass(sr, f, 0.7071);
                    auto hpC = BiquadCoeffs<T>::makeHighPass(sr, f, 0.7071);
                    auto apC = BiquadCoeffs<T>::makeAllPass(sr, f, 0.7071);

                    for (int st = 0; st < 2; ++st)
                    {
                        splits_[s].lp[st].setCoeffs(lpC);
                        splits_[s].hp[st].setCoeffs(hpC);
                    }
                    for (int b = 0; b < s; ++b)
                        for (int st = 0; st < 2; ++st)
                            allpassFlat_[b * kMaxSplits + s].stages[st].setCoeffs(apC);
                }
                break;

            case 48:
            {
                numStagesPerFilter_ = 4;
                constexpr double q1 = 0.5412;
                constexpr double q2 = 1.3066;
                const double qArr[4] = { q1, q2, q1, q2 };

                for (int s = 0; s < numSplits; ++s)
                {
                    double f = std::clamp(static_cast<double>(frequencies_[s]), 20.0, sr * 0.499);

                    for (int st = 0; st < 4; ++st)
                    {
                        splits_[s].lp[st].setCoeffs(BiquadCoeffs<T>::makeLowPass(sr, f, qArr[st]));
                        splits_[s].hp[st].setCoeffs(BiquadCoeffs<T>::makeHighPass(sr, f, qArr[st]));
                    }
                    for (int b = 0; b < s; ++b)
                        for (int st = 0; st < 4; ++st)
                            allpassFlat_[b * kMaxSplits + s].stages[st].setCoeffs(BiquadCoeffs<T>::makeAllPass(sr, f, qArr[st]));
                }
                break;
            }

            default: break;
        }
    }

    void processIIR(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(input.getNumChannels(), numChannels_);
        const int nS  = input.getNumSamples();
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
        return workBuf_.data() + ch * spec_.maxBlockSize;
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
                        for (int st = 0; st < numStagesPerFilter_; ++st)
                            sample = allpassFlat_[b * kMaxSplits + s].stages[st].processSample(sample, ch);
                        outputs[b].getChannel(ch)[offset + i] = sample;
                    }
                }
            }
        }
    }

    void recomputeLinearPhaseMagnitudes() noexcept
    {
        if (!lpMagDirty_.load(std::memory_order_relaxed) || lpFftSize_ == 0) return;
        lpMagDirty_.store(false, std::memory_order_relaxed);

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
        // lpIdealMagsFlat_[band * numBins + bin] — no audio-thread allocation.
        const int nBands = numBands_.load(std::memory_order_relaxed);

        for (int k = 0; k < numBins; ++k)
        {
            double freq = sr * static_cast<double>(k) / static_cast<double>(lpFftSize_);
            T lpMag[kMaxSplits], hpMag[kMaxSplits];

            for (int s = 0; s < numSplits; ++s)
            {
                double fc = std::max(static_cast<double>(frequencies_[s]), 1.0);
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
                lpIdealMagsFlat_[b * numBins + k] = hpMag[b - 1] * lpMag[b];
            lpIdealMagsFlat_[(nBands - 1) * numBins + k] = hpMag[numSplits - 1];
        }

        // Window Design & FIR Kernel generation (pre-allocated scratch)
        for (int b = 0; b < nBands; ++b)
        {
            // Prepare complex bins for inverse FFT (zero phase)
            for(int k = 0; k < numBins; ++k) {
                lpTimeResponse_[2 * k] = lpIdealMagsFlat_[b * numBins + k];
                lpTimeResponse_[2 * k + 1] = T(0);
            }

            lpFft_->inverse(lpTimeResponse_.data(), lpFirKernel_.data());
            
            // Circular shift, windowing (Blackman), and zero-padding
            std::fill(lpFftIn_.begin(), lpFftIn_.end(), T(0));
            int halfLen = firLength_ / 2;
            
            for (int i = 0; i < firLength_; ++i)
            {
                // Un-wrap circular time response to causal shifted response
                int srcIdx = (i - halfLen + lpFftSize_) % lpFftSize_;
                
                // Blackman Window
                double n = static_cast<double>(i);
                double N = static_cast<double>(firLength_ - 1);
                double window = 0.42 - 0.5 * std::cos(2.0 * std::numbers::pi * n / N) + 0.08 * std::cos(4.0 * std::numbers::pi * n / N);
                
                lpFftIn_[i] = lpFirKernel_[srcIdx] * static_cast<T>(window);
            }
            
            // Forward FFT to get the usable overlap-save kernel
            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());
            
            // Store complex spectrum contiguously
            T* bandMagData = lpMagnitudesFlat_.data() + (b * numBins * 2);
            std::copy(lpFftOut_.begin(), lpFftOut_.begin() + (numBins * 2), bandMagData);
        }
    }

    inline T* getPrevBlockChannel(int ch) noexcept
    {
        return lpPrevBlockFlat_.data() + ch * firLength_;
    }

    void processLinearPhase(AudioBufferView<T> input, AudioBufferView<T>* outputs, int numBands) noexcept
    {
        recomputeLinearPhaseMagnitudes();

        const int nCh = std::min(input.getNumChannels(), numChannels_);
        const int nS  = input.getNumSamples();
        const int numBins = lpFftSize_ / 2 + 1;
        const int overlapSize = firLength_ - 1;

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* channelData = input.getChannel(ch);
            T* prev = getPrevBlockChannel(ch);

            // Overlap-Save: [overlap block | new samples | zero pad]
            for (int i = 0; i < overlapSize; ++i)
                lpFftIn_[i] = prev[i];
                
            for (int i = 0; i < nS; ++i)
                lpFftIn_[overlapSize + i] = channelData[i];
                
            for (int i = overlapSize + nS; i < lpFftSize_; ++i)
                lpFftIn_[i] = T(0);

            // Save tail for next block
            for (int i = 0; i < overlapSize; ++i)
            {
                if (nS - overlapSize + i >= 0)
                    prev[i] = channelData[nS - overlapSize + i];
                else
                    prev[i] = lpFftIn_[overlapSize + nS - overlapSize + i]; // Handle block size < overlap
            }

            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());

            for (int b = 0; b < numBands; ++b)
            {
                const T* kernelSpectrum = lpMagnitudesFlat_.data() + (b * numBins * 2);
                
                // Complex multiplication
                for (int k = 0; k < numBins; ++k)
                {
                    T r1 = lpFftOut_[2 * k];
                    T i1 = lpFftOut_[2 * k + 1];
                    T r2 = kernelSpectrum[2 * k];
                    T i2 = kernelSpectrum[2 * k + 1];

                    lpBandFft_[2 * k]     = r1 * r2 - i1 * i2;
                    lpBandFft_[2 * k + 1] = r1 * i2 + i1 * r2;
                }

                lpFft_->inverse(lpBandFft_.data(), lpFftResult_.data());

                // Discard garbage and copy valid output
                T* outCh = outputs[b].getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    outCh[i] = lpFftResult_[overlapSize + i];
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
