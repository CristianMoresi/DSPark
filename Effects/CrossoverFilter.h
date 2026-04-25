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
 *               DenormalGuard.h.
 *
 * @code
 *   dspark::CrossoverFilter<float> xover;
 *   xover.setNumBands(3);
 *   xover.setCrossoverFrequency(0, 200.0f);   // low/mid split
 *   xover.setCrossoverFrequency(1, 2000.0f);  // mid/high split
 *   xover.setOrder(24);                        // LR24 (24 dB/oct)
 *   xover.prepare(spec);
 *
 *   // Allocate output buffers (one per band)
 *   dspark::AudioBuffer<float> bands[3];
 *   for (auto& b : bands) b.resize(spec.numChannels, spec.maxBlockSize);
 *   dspark::AudioBufferView<float> views[3] = { bands[0].toView(), bands[1].toView(), bands[2].toView() };
 *
 *   xover.processBlock(inputBuffer, views, 3);
 *   // views[0] = low, views[1] = mid, views[2] = high
 * @endcode
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
 * @tparam T        Sample type (float or double).
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
        LinearPhase   ///< FFT-based magnitude-only (block-size latency, zero
                      ///< phase distortion). M5a note: strictly *zero-phase*,
                      ///< so transients get pre-ringing (same caveat as
                      ///< Equalizer::FilterMode::LinearPhase).
    };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the crossover for processing.
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        numChannels_ = spec.numChannels;

        // Allocate IIR work buffer
        workBuf_.resize(static_cast<size_t>(spec.numChannels));
        for (auto& ch : workBuf_)
            ch.assign(static_cast<size_t>(spec.maxBlockSize), T(0));

        // Linear-phase FFT resources
        if (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && spec.maxBlockSize > 0)
        {
            int fftPow2 = 1;
            while (fftPow2 < spec.maxBlockSize * 2) fftPow2 <<= 1;
            lpFftSize_ = fftPow2;

            lpFft_ = std::make_unique<FFTReal<T>>(lpFftSize_);
            int numBins = lpFftSize_ / 2 + 1;

            lpMagnitudes_.resize(static_cast<size_t>(MaxBands));
            for (auto& m : lpMagnitudes_)
                m.assign(static_cast<size_t>(numBins), T(0));

            lpPrevBlock_.resize(static_cast<size_t>(spec.numChannels));
            for (auto& pb : lpPrevBlock_)
                pb.assign(static_cast<size_t>(lpFftSize_ / 2), T(0));

            lpFftIn_.assign(static_cast<size_t>(lpFftSize_), T(0));
            lpFftOut_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpBandFft_.assign(static_cast<size_t>(lpFftSize_ + 2), T(0));
            lpFftResult_.assign(static_cast<size_t>(lpFftSize_), T(0));
        }

        // Allocate allpass correction chains (heap to avoid large inline arrays)
        allpass_.resize(static_cast<size_t>(MaxBands));
        for (auto& v : allpass_)
            v.resize(static_cast<size_t>(kMaxSplits));

        // Initialize frequency smoothers
        int nb = numBands_.load(std::memory_order_relaxed);
        for (int i = 0; i < kMaxSplits; ++i)
        {
            freqSmoothers_[i].prepare(spec.sampleRate, 5.0);
            freqSmoothers_[i].setSmoothingType(SmoothedValue<T>::SmoothingType::Exponential);
            if (i < nb - 1)
                freqSmoothers_[i].setCurrentAndTarget(frequencies_[i]);
        }

        dirty_.store(true, std::memory_order_relaxed);
        lpMagDirty_.store(true, std::memory_order_relaxed);
        reset();

        prepared_ = true;
    }

    /**
     * @brief Splits input into separate band outputs.
     *
     * @param input       Input audio buffer.
     * @param bandOutputs Array of AudioBufferView, one per band.
     * @param numOutputBands Number of output bands (must match getNumBands()).
     */
    void processBlock(AudioBufferView<T> input,
                      AudioBufferView<T>* bandOutputs, int numOutputBands) noexcept
    {
        if (!prepared_) return;
        if (dirty_.load(std::memory_order_relaxed)) updateCoefficients();

        int n = std::min(numOutputBands, numBands_.load(std::memory_order_relaxed));
        if (n < 2) return;

        if (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase && lpFft_)
            processLinearPhase(input, bandOutputs, n);
        else
            processIIR(input, bandOutputs, n);
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Sets the number of output bands (2–MaxBands).
     *
     * Default crossover frequencies are logarithmically spaced from 100 Hz to 10 kHz.
     */
    void setNumBands(int n) noexcept
    {
        numBands_.store(std::clamp(n, 2, MaxBands), std::memory_order_relaxed);
        initDefaultFrequencies();
        dirty_.store(true, std::memory_order_relaxed);
        lpMagDirty_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Sets a crossover frequency.
     * @param index Split index (0 to numBands-2).
     * @param freqHz Frequency in Hz.
     */
    void setCrossoverFrequency(int index, T freqHz) noexcept
    {
        if (index >= 0 && index < kMaxSplits)
        {
            frequencies_[index] = freqHz;
            freqSmoothers_[index].setTargetValue(freqHz);
            dirty_.store(true, std::memory_order_relaxed);
            lpMagDirty_.store(true, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Sets the Linkwitz-Riley order (12, 24, or 48 dB/oct).
     * @param order 12 (LR12), 24 (LR24, default), or 48 (LR48).
     */
    void setOrder(int order) noexcept
    {
        if (order == 12 || order == 24 || order == 48)
        {
            order_.store(order, std::memory_order_relaxed);
            dirty_.store(true, std::memory_order_relaxed);
            lpMagDirty_.store(true, std::memory_order_relaxed);
        }
    }

    /**
     * @brief Sets the filter processing mode.
     *
     * Call prepare() again after changing mode.
     */
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

    /** @brief Returns latency in samples (0 for IIR, blockSize for linear-phase). */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (filterMode_.load(std::memory_order_relaxed) == FilterMode::LinearPhase) ? spec_.maxBlockSize : 0;
    }

    /** @brief Resets all internal filter state. */
    void reset() noexcept
    {
        for (auto& sp : splits_)
        {
            for (auto& b : sp.lp) b.reset();
            for (auto& b : sp.hp) b.reset();
        }
        for (auto& bandAps : allpass_)
            for (auto& apChain : bandAps)
                for (auto& b : apChain.stages) b.reset();
        for (auto& pb : lpPrevBlock_)
            std::fill(pb.begin(), pb.end(), T(0));
    }

private:
    static constexpr int kMaxSplits = MaxBands - 1;
    static constexpr int kMaxStagesPerFilter = 4; // LR48 max

    // -- IIR filter structures -----------------------------------------------

    struct SplitPoint
    {
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> lp;
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> hp;
    };

    struct AllPassChain
    {
        std::array<Biquad<T, 16>, kMaxStagesPerFilter> stages;
    };

    // -- First-order allpass (for LR12) --------------------------------------

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

    // -- Default frequencies -------------------------------------------------

    void initDefaultFrequencies() noexcept
    {
        int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        const T logMin = std::log(T(100));
        const T logMax = std::log(T(10000));

        for (int s = 0; s < numSplits; ++s)
        {
            T t = static_cast<T>(s + 1) / static_cast<T>(numSplits + 1);
            frequencies_[s] = std::exp(logMin + t * (logMax - logMin));
        }
    }

    // -- Coefficient updates -------------------------------------------------

    void updateCoefficients() noexcept
    {
        if (spec_.sampleRate <= 0) return;
        dirty_.store(false, std::memory_order_relaxed);

        int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        double sr = spec_.sampleRate;

        // Sort frequencies ascending
        std::sort(frequencies_.begin(), frequencies_.begin() + numSplits);

        switch (order_.load(std::memory_order_relaxed))
        {
            case 12: // LR12: two first-order Butterworth stages
            {
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
                            allpass_[b][s].stages[st].setCoeffs(apC);
                }
                break;
            }

            case 24: // LR24: two second-order Butterworth stages (Q = 0.7071)
            {
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
                            allpass_[b][s].stages[st].setCoeffs(apC);
                }
                break;
            }

            case 48: // LR48: four second-order stages (BW4 Q = {0.5412, 1.3066})
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
                            allpass_[b][s].stages[st].setCoeffs(
                                BiquadCoeffs<T>::makeAllPass(sr, f, qArr[st]));
                }
                break;
            }

            default: break;
        }
    }

    // -- IIR processing (minimum-phase) --------------------------------------

    void processIIR(AudioBufferView<T> input,
                    AudioBufferView<T>* outputs, int numBands) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(input.getNumChannels(), numChannels_);
        const int nS  = input.getNumSamples();
        const int numSplits = numBands - 1;

        // Check if any frequency smoother is still active
        bool anySmoothing = false;
        for (int i = 0; i < numSplits; ++i)
            anySmoothing = anySmoothing || freqSmoothers_[i].isSmoothing();

        if (anySmoothing)
        {
            // Process in sub-blocks of 32 samples with coefficient updates
            constexpr int kSubBlockSize = 32;
            int offset = 0;
            while (offset < nS)
            {
                int blockLen = std::min(kSubBlockSize, nS - offset);

                // Advance smoothers and update coefficients with smoothed frequencies
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

    /**
     * @brief Processes a sub-range [offset, offset+blockLen) through the IIR crossover.
     */
    void processIIRRange(AudioBufferView<T> input,
                         AudioBufferView<T>* outputs, int numBands,
                         int nCh, int offset, int blockLen) noexcept
    {
        const int numSplits = numBands - 1;

        // Copy input to work buffer
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* src = input.getChannel(ch) + offset;
            T* dst = workBuf_[ch].data();
            std::copy(src, src + blockLen, dst);
        }

        // Cascaded splitting: LP → band output, HP → residual work buffer
        for (int s = 0; s < numSplits; ++s)
        {
            for (int i = 0; i < blockLen; ++i)
            {
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T sample = workBuf_[ch][i];

                    // LP cascade → band output
                    T lpSample = sample;
                    for (int st = 0; st < numStagesPerFilter_; ++st)
                        lpSample = splits_[s].lp[st].processSample(lpSample, ch);
                    outputs[s].getChannel(ch)[offset + i] = lpSample;

                    // HP cascade → work buffer (in-place)
                    T hpSample = sample;
                    for (int st = 0; st < numStagesPerFilter_; ++st)
                        hpSample = splits_[s].hp[st].processSample(hpSample, ch);
                    workBuf_[ch][i] = hpSample;
                }
            }
        }

        // Last band = remaining work buffer (highest frequencies)
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* dst = outputs[numBands - 1].getChannel(ch) + offset;
            const T* src = workBuf_[ch].data();
            std::copy(src, src + blockLen, dst);
        }

        // Allpass phase correction for lower bands
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
                            sample = allpass_[b][s].stages[st].processSample(sample, ch);
                        outputs[b].getChannel(ch)[offset + i] = sample;
                    }
                }
            }
        }
    }

    // -- Linear-phase processing (FFT-based) ---------------------------------

    /**
     * @brief Recomputes per-band LR magnitude curves for FFT processing.
     *
     * Uses the analytic Butterworth magnitude formula:
     *   |LP(f)|  = 1 / (1 + (f/fc)^(2N))
     *   |HP(f)|  = (f/fc)^(2N) / (1 + (f/fc)^(2N))
     * where N = LR order (2 for LR12, 4 for LR24, 8 for LR48).
     *
     * Bands sum to unity at every frequency (LR property).
     */
    void recomputeLinearPhaseMagnitudes() noexcept
    {
        if (!lpMagDirty_.load(std::memory_order_relaxed) || lpFftSize_ == 0) return;
        lpMagDirty_.store(false, std::memory_order_relaxed);

        const int numBins = lpFftSize_ / 2 + 1;
        const int numSplits = numBands_.load(std::memory_order_relaxed) - 1;
        const double sr = spec_.sampleRate;

        // Exponent for the Butterworth magnitude formula
        int expo = 0;
        switch (order_.load(std::memory_order_relaxed))
        {
            case 12: expo = 2; break;  // (f/fc)^2
            case 24: expo = 4; break;  // (f/fc)^4
            case 48: expo = 8; break;  // (f/fc)^8
        }

        for (int k = 0; k < numBins; ++k)
        {
            double freq = sr * static_cast<double>(k) / static_cast<double>(lpFftSize_);

            // Compute LP and HP magnitude at each split point
            T lpMag[kMaxSplits], hpMag[kMaxSplits];

            for (int s = 0; s < numSplits; ++s)
            {
                double fc = std::max(static_cast<double>(frequencies_[s]), 1.0);
                double ratio = freq / fc;

                // (f/fc)^expo via repeated squaring
                double rPow = 1.0;
                double r2 = ratio * ratio;
                if (expo >= 2) rPow = r2;
                if (expo >= 4) rPow = r2 * r2;
                if (expo >= 8) rPow = rPow * rPow;

                double denom = 1.0 + rPow;
                lpMag[s] = static_cast<T>(1.0 / denom);
                hpMag[s] = static_cast<T>(rPow / denom);
            }

            // Per-band magnitude (cascaded split)
            lpMagnitudes_[0][k] = lpMag[0];
            for (int b = 1; b < numBands_ - 1; ++b)
                lpMagnitudes_[b][k] = hpMag[b - 1] * lpMag[b];
            lpMagnitudes_[numBands_ - 1][k] = hpMag[numSplits - 1];
        }
    }

    void processLinearPhase(AudioBufferView<T> input,
                            AudioBufferView<T>* outputs, int numBands) noexcept
    {
        recomputeLinearPhaseMagnitudes();

        const int nCh = std::min(input.getNumChannels(), numChannels_);
        const int nS  = input.getNumSamples();
        const int halfFft = lpFftSize_ / 2;
        const int numBins = lpFftSize_ / 2 + 1;

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* channelData = input.getChannel(ch);
            auto& prev = lpPrevBlock_[ch];

            // Build FFT input: [prev | current | zero-pad]
            for (int i = 0; i < halfFft; ++i)
                lpFftIn_[i] = (i < static_cast<int>(prev.size())) ? prev[i] : T(0);
            for (int i = 0; i < nS && i < halfFft; ++i)
                lpFftIn_[halfFft + i] = channelData[i];
            for (int i = nS; i < halfFft; ++i)
                lpFftIn_[halfFft + i] = T(0);

            // Save raw input as prev for next overlap-save call
            for (int i = 0; i < halfFft; ++i)
                prev[i] = (i < nS) ? channelData[i] : T(0);

            // Forward FFT (shared across bands)
            lpFft_->forward(lpFftIn_.data(), lpFftOut_.data());

            // For each band: apply magnitude, IFFT, extract overlap-save output
            for (int b = 0; b < numBands; ++b)
            {
                // Multiply complex spectrum by band magnitude
                for (int k = 0; k < numBins; ++k)
                {
                    T mag = lpMagnitudes_[b][k];
                    lpBandFft_[2 * k]     = lpFftOut_[2 * k]     * mag;
                    lpBandFft_[2 * k + 1] = lpFftOut_[2 * k + 1] * mag;
                }

                // Inverse FFT
                lpFft_->inverse(lpBandFft_.data(), lpFftResult_.data());

                // Overlap-save: take last B samples
                T* outCh = outputs[b].getChannel(ch);
                for (int i = 0; i < nS; ++i)
                    outCh[i] = lpFftResult_[halfFft + i];
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

    // Crossover frequencies (sorted ascending)
    std::array<T, kMaxSplits> frequencies_ {};
    std::array<SmoothedValue<T>, kMaxSplits> freqSmoothers_;

    // IIR state
    int numStagesPerFilter_ = 2;
    std::array<SplitPoint, kMaxSplits> splits_ {};
    // Dynamically allocated to avoid large inline arrays (MaxBands * kMaxSplits * 4 Biquads)
    std::vector<std::vector<AllPassChain>> allpass_; // [band][split]
    std::vector<std::vector<T>> workBuf_; // [channel][sample]

    // Linear-phase state
    std::unique_ptr<FFTReal<T>> lpFft_;
    int lpFftSize_ = 0;
    std::atomic<bool> lpMagDirty_ { true };
    std::vector<std::vector<T>> lpMagnitudes_; // [band][bin]
    std::vector<std::vector<T>> lpPrevBlock_;  // [channel][sample]
    std::vector<T> lpFftIn_, lpFftOut_, lpBandFft_, lpFftResult_;
};

} // namespace dspark
