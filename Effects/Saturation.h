// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Saturation.h
 * @brief Professional multi-algorithm saturation processor with analog simulation.
 *
 * This class provides a zero-latency, high-quality saturation pipeline featuring 
 * 10 distinct algorithms ranging from soft clipping to complex magnetic tape and 
 * transformer hysteresis modeling. 
 *
 * @details
 * **Architecture & Performance:**
 * - 100% Real-Time Safe: No memory allocations, locks, or blocking calls in the `process` path.
 * - Zero Virtual Dispatch in Hot Path: Algorithm execution is resolved statically per block.
 * - Lock-Free Parameters: All parameter changes are pushed via a Single-Producer/Single-Consumer (SPSC) queue.
 * - SIMD/Cache Friendly: Internal states (drift, slew) are pre-calculated in contiguous arrays.
 *
 * Dependencies: DSP/Core headers only (C++20 STL).
 *
 * @tparam SampleType The floating-point precision to use (must be `float` or `double`).
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/Oversampling.h"
#include "../Core/Smoothers.h"
#include "../Core/AnalogRandom.h"
#include "../Core/SpscQueue.h"
#include "../Core/SpinLock.h"
#include "MidSide.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <cstring>

namespace dspark {

template <typename SampleType> class Saturation;

namespace detail {

template <typename T>
class SaturationAlgorithm
{
public:
    virtual ~SaturationAlgorithm() = default;

    /** @brief Prepares the algorithm with the current audio specification. */
    virtual void prepare(const AudioSpec& spec) noexcept = 0;

    /** @brief Resets internal states (filters, phase, memory). */
    virtual void reset() noexcept = 0;

    /** @brief Updates internal coefficients dependent on block-rate parameters. */
    virtual void update(T /*driveGain*/, T /*character*/, const AudioSpec& /*spec*/) noexcept {}

    /** @brief Identifies the exact algorithm type for CRTP static dispatch. */
    virtual typename Saturation<T>::Algorithm getType() const noexcept = 0;
};

// -- SoftClip (tanh) ---------------------------------------------------------
template <typename T>
class TanhAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::SoftClip; }

    inline T processSample(T sample, T drive, T character, int) noexcept
    {
        T x    = sample * drive;
        T bias = character * T(0.3);
        return fastTanh(x + bias) - fastTanh(bias);
    }
};

// -- Tube (12AX7-style asymmetric triode model) -----------------------------
template <typename T>
class TubeAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Tube; }

    inline T processSample(T sample, T drive, T character, int) noexcept
    {
        T x = sample * drive;
        T asym = T(1.15) + character * T(0.5);
        return (x >= T(0)) ? fastTanh(x) : fastTanh(x * asym);
    }
};

// -- HardClip ----------------------------------------------------------------
template <typename T>
class HardClipAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::HardClip; }

    inline T processSample(T sample, T drive, T character, int) noexcept
    {
        T bias = character * T(0.3);
        T x = sample * drive + bias;
        T clipped = std::clamp(x, T(-1), T(1));
        return clipped - std::clamp(bias, T(-1), T(1));
    }
};

// -- Exciter (polynomial waveshaper) -----------------------------------------
template <typename T>
class ExciterAlgorithm final : public SaturationAlgorithm<T>
{
public:
    void prepare(const AudioSpec&) noexcept override {}
    void reset() noexcept override {}
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Exciter; }

    inline T processSample(T sample, T drive, T character, int) noexcept
    {
        T x      = std::clamp(sample * drive, T(-10), T(10));
        T x2     = x * x;
        T x3     = x2 * x;
        T result = x + character * T(0.25) * x2 - T(0.15) * x3;
        return std::clamp(result, T(-1), T(1));
    }
};

// -- Wavefolder (sin + first-order ADAA) -------------------------------------
template <typename T>
class WavefolderAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<T, kMaxCh> lastX_ {};
    std::array<T, kMaxCh> lastF_ {};

public:
    void prepare(const AudioSpec&) noexcept override { reset(); }
    void reset() noexcept override
    {
        lastX_.fill(T(0));
        lastF_.fill(T(-1)); 
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Wavefolder; }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        T x   = sample * drive * twoPi<T> + character * pi<T>;
        T F_x = -std::cos(x);
        T diff = x - lastX_[ch];

        T result;
        if (std::abs(diff) > T(1e-5))
            result = (F_x - lastF_[ch]) / diff;
        else
            result = std::sin(x);

        lastX_[ch] = x;
        lastF_[ch] = F_x;
        return result;
    }
};

// -- Bitcrusher (TPDF dither) ------------------------------------------------
template <typename T>
class BitcrusherAlgorithm final : public SaturationAlgorithm<T>
{
    AnalogRandom::Generator<T> ditherGen_;
    T steps_ = T(1);
    T invSteps_ = T(1);

public:
    void prepare(const AudioSpec& spec) noexcept override { ditherGen_.prepare(spec.sampleRate); }
    void reset() noexcept override {}
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Bitcrusher; }

    void update(T drive, T /*character*/, const AudioSpec&) noexcept override
    {
        T clamped  = std::clamp(drive, T(1), T(100));
        T bitDepth = mapRange(clamped, T(1), T(100), T(16), T(2));
        steps_     = std::pow(T(2), bitDepth);
        invSteps_  = T(1) / steps_;
    }

    inline T processSample(T sample, T, T, int) noexcept
    {
        T dither = (ditherGen_.getNextSample() - ditherGen_.getNextSample()) * invSteps_;
        return invSteps_ * std::round((sample + dither) * steps_);
    }
};

// -- Tape (hysteresis model + head bump + HF rolloff) ------------------------
template <typename T>
class TapeAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<Biquad<T, 1>, kMaxCh> preFilters_;  
    std::array<Biquad<T, 1>, kMaxCh> postFilters_; 
    std::array<T, kMaxCh>            M_ {};        
    std::array<T, kMaxCh>            lastH_ {};    
    int numChannels_ = 0;

    static inline T langevin(T x) noexcept
    {
        T ax = std::abs(x);
        if (ax < T(0.01)) return x / T(3);
        return T(1) / std::tanh(x) - T(1) / x;
    }

    static inline T langevinDeriv(T x) noexcept
    {
        T ax = std::abs(x);
        if (ax < T(0.01)) return T(1) / T(3);
        T csch = T(1) / std::sinh(x);
        return -csch * csch + T(1) / (x * x);
    }

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = std::min(spec.numChannels, kMaxCh); // clamp per-channel state
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : preFilters_)  f.reset();
        for (auto& f : postFilters_) f.reset();
        M_.fill(T(0));
        lastH_.fill(T(0));
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Tape; }

    void update(T drive, T /*character*/, const AudioSpec& spec) noexcept override
    {
        auto driveDb = gainToDecibels(drive, T(-100));
        T bumpGain = T(1.5) + std::min(driveDb * T(0.05), T(3.0));
        auto peakCoeffs = BiquadCoeffs<T>::makePeak(spec.sampleRate, 80.0, 0.6, static_cast<double>(bumpGain));
        auto lpFreq = std::max(2000.0, 18000.0 - static_cast<double>(driveDb) * 250.0);
        auto lpCoeffs = BiquadCoeffs<T>::makeLowPass(spec.sampleRate, lpFreq, 0.55);

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            preFilters_[ch].setCoeffs(peakCoeffs);
            postFilters_[ch].setCoeffs(lpCoeffs);
        }
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        T filtered = preFilters_[ch].processSample(sample, 0);

        T a  = T(1.0) - character * T(0.25);
        T Ms = T(3) * a;
        T c  = T(0.6) + character * T(0.2);

        T H = filtered * drive;
        T dH = H - lastH_[ch];
        T He = H + c * M_[ch];

        T Man = Ms * langevin(He / a);
        T delta = (dH >= T(0)) ? T(1) : T(-1);
        T ManDiff = Man - M_[ch];
        T dManDH = Ms * langevinDeriv(He / a) / a;

        T denominator = T(1) - c * dManDH;
        if (std::abs(denominator) < T(1e-10))
            denominator = std::copysign(T(1e-10), denominator);

        T dM;
        if (delta * ManDiff > T(0))
            dM = (ManDiff / denominator) * std::abs(dH) / (a + std::abs(dH));
        else
            dM = T(0);

        M_[ch] = std::clamp(M_[ch] + dM, -Ms, Ms);
        lastH_[ch] = H;

        return postFilters_[ch].processSample(M_[ch], 0);
    }
};

// -- Transformer (frequency-dependent: heavy LF, light HF) ------------------
template <typename T>
class TransformerAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<Biquad<T, 1>, kMaxCh> lpFilters_;
    int numChannels_ = 0;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = std::min(spec.numChannels, kMaxCh); // clamp per-channel state
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : lpFilters_) f.reset();
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Transformer; }

    void update(T /*drive*/, T /*character*/, const AudioSpec& spec) noexcept override
    {
        auto c = BiquadCoeffs<T>::makeLowPass(spec.sampleRate, 250.0, 0.707);
        for (int ch = 0; ch < numChannels_; ++ch)
            lpFilters_[ch].setCoeffs(c);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        T low  = lpFilters_[ch].processSample(sample, 0);
        T high = sample - low;
        T bias = character * T(0.2);
        T satLow  = fastTanh((low  + bias) * drive * T(1.3)) - fastTanh(bias);
        T satHigh = fastTanh((high + bias) * drive * T(0.7)) - fastTanh(bias);
        return satLow + satHigh;
    }
};

// -- Downsample (sample rate reduction) --------------------------------------
template <typename T>
class DownsampleAlgorithm final : public SaturationAlgorithm<T>
{
    static constexpr int kMaxCh = 16;
    std::array<Biquad<T, 1>, kMaxCh> aaFilters_;
    std::array<T, kMaxCh>            lastSample_ {};
    std::array<int, kMaxCh>          counter_    {};
    int numChannels_ = 0;
    int reduction_   = 1;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        numChannels_ = std::min(spec.numChannels, kMaxCh); // clamp per-channel state
        reset();
    }
    void reset() noexcept override
    {
        for (auto& f : aaFilters_) f.reset();
        lastSample_.fill(T(0));
        counter_.fill(0);
    }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::Downsample; }

    void update(T drive, T, const AudioSpec& spec) noexcept override
    {
        T clamped = std::clamp(drive, T(1), T(100));
        reduction_ = std::max(1, static_cast<int>(mapRange(clamped, T(1), T(100), T(1), T(50))));
        
        auto c = BiquadCoeffs<T>::makeLowPass(spec.sampleRate, spec.sampleRate / (2.5 * reduction_), 0.707);
        for (int ch = 0; ch < numChannels_; ++ch)
            aaFilters_[ch].setCoeffs(c);
    }

    inline T processSample(T sample, T, T, int ch) noexcept
    {
        T filtered = aaFilters_[ch].processSample(sample, 0);
        if (++counter_[ch] >= reduction_)
        {
            counter_[ch] = 0;
            lastSample_[ch] = filtered;
        }
        return lastSample_[ch];
    }
};

// -- MultiStage (Tube → Tape → Transformer cascade) -------------------------
template <typename T>
class MultiStageAlgorithm final : public SaturationAlgorithm<T>
{
    TubeAlgorithm<T>        tube_;
    TapeAlgorithm<T>        tape_;
    TransformerAlgorithm<T> xfmr_;

public:
    void prepare(const AudioSpec& spec) noexcept override
    {
        tube_.prepare(spec);
        tape_.prepare(spec);
        xfmr_.prepare(spec);
    }
    void reset() noexcept override { tube_.reset(); tape_.reset(); xfmr_.reset(); }
    typename Saturation<T>::Algorithm getType() const noexcept override { return Saturation<T>::Algorithm::MultiStage; }

    void update(T drive, T character, const AudioSpec& spec) noexcept override
    {
        tape_.update(drive * T(0.6), character, spec);
        xfmr_.update(drive * T(0.8), character, spec);
    }

    inline T processSample(T sample, T drive, T character, int ch) noexcept
    {
        T tubeOut  = tube_.processSample(sample,  drive * T(0.5), character, ch);
        T tapeOut  = tape_.processSample(tubeOut, drive * T(0.6), character, ch);
        return       xfmr_.processSample(tapeOut, drive * T(0.8), character, ch);
    }
};

} // namespace detail

// ============================================================================
// Saturation — Public API
// ============================================================================

/**
 * @class Saturation
 * @brief Professional multi-algorithm saturation processor with analog simulation.
 *
 * This class provides a zero-latency, high-quality saturation pipeline featuring 
 * 10 distinct algorithms ranging from soft clipping to complex magnetic tape and 
 * transformer hysteresis modeling. 
 *
 * @tparam SampleType The floating-point precision to use (must be `float` or `double`).
 */
template <typename SampleType>
class Saturation
{
    static_assert(std::is_floating_point_v<SampleType>,
                  "Saturation: SampleType must be float or double.");

public:
    // -- Enums ---------------------------------------------------------------

    /**
     * @brief Defines the harmonic generation topology.
     */
    enum class Algorithm
    {
        Tube,        /**< Asymmetric triode emulation. Dominant 2nd harmonic, compresses negative excursions. */
        Tape,        /**< Magnetic hysteresis model (Langevin function) with dynamic head-bump and HF roll-off. */
        Transformer, /**< Core saturation model. LF saturates before HF. */
        SoftClip,    /**< Symmetric Tanh curve. Odd harmonics only (3rd, 5th, 7th). */
        HardClip,    /**< Digital hard-clipping. Extreme odd harmonics, highly aliasing without oversampling. */
        Exciter,     /**< Polynomial waveshaper designed to synthesize high-frequency harmonic content. */
        Wavefolder,  /**< First-order ADAA (Antiderivative Anti-Aliasing) sine wavefolder. */
        Bitcrusher,  /**< Quantization noise generator with TPDF dithering. */
        Downsample,  /**< Sample-and-hold rate reduction with anti-aliasing pre-filtering. */
        MultiStage   /**< Serial cascade: Tube -> Tape -> Transformer. Gain-staged internally. */
    };

    /** @brief Determines how the stereo field is processed. */
    enum class ProcessingMode { Stereo, MidOnly, SideOnly, MidSide };

    /** @brief Determines the final output signal routing. */
    enum class OutputMode { Normal, WetOnly, Delta };

    // -- Lifecycle -----------------------------------------------------------

    Saturation()
    {
        pool_[0]  = std::make_unique<detail::TubeAlgorithm<SampleType>>();
        pool_[1]  = std::make_unique<detail::TapeAlgorithm<SampleType>>();
        pool_[2]  = std::make_unique<detail::TransformerAlgorithm<SampleType>>();
        pool_[3]  = std::make_unique<detail::TanhAlgorithm<SampleType>>();
        pool_[4]  = std::make_unique<detail::HardClipAlgorithm<SampleType>>();
        pool_[5]  = std::make_unique<detail::ExciterAlgorithm<SampleType>>();
        pool_[6]  = std::make_unique<detail::WavefolderAlgorithm<SampleType>>();
        pool_[7]  = std::make_unique<detail::BitcrusherAlgorithm<SampleType>>();
        pool_[8]  = std::make_unique<detail::DownsampleAlgorithm<SampleType>>();
        pool_[9]  = std::make_unique<detail::MultiStageAlgorithm<SampleType>>();

        active_.store(pool_[static_cast<int>(Algorithm::SoftClip)].get());
        next_.store(nullptr);
    }

    ~Saturation() = default;
    Saturation(const Saturation&)            = delete;
    Saturation& operator=(const Saturation&) = delete;

    /**
     * @brief Prepares all internal resources, filters, and buffers.
     * 
     * @note **NOT Real-Time Safe.** This method performs memory allocations (`std::vector::resize`) 
     * and filter coefficient calculations. It must be called from the main/setup thread before 
     * audio processing begins.
     * 
     * @param spec The audio environment specifications (sample rate, max block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        for (auto& algo : pool_)
            if (algo) algo->prepare(spec);

        preFilter_.reset();
        postFilter_.reset();
        dcBlocker_.reset();
        dcBlocker_.setCoeffs(BiquadCoeffs<SampleType>::makeDcBlocker(spec.sampleRate));
        
        lastPreHpFreq_    = -1.0f;
        lastPostTiltFreq_ = -1.0f;
        lastPostTiltGain_ = std::numeric_limits<float>::quiet_NaN();
        dryWetMixer_.prepare(spec);
        
        if (oversampler_) oversampler_->prepare(spec);

        tempBuffer_.resize(spec.numChannels, spec.maxBlockSize * std::max(1, oversamplingFactor_));
        driftBuffer_.resize(spec.numChannels, spec.maxBlockSize * std::max(1, oversamplingFactor_));

        // Keep the dry path aligned with the (latent) oversampled wet path so the
        // dry/wet, Delta and adaptive-blend mixes do not comb-filter.
        dryWetMixer_.setLatencyCompensation(
            (oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0);

        auto sr = spec.sampleRate;
        driveSmoother_.reset(sr, 20.0f, 0.707f, 0.0f);
        mixSmoother_.reset(sr, 20.0f, 1.0f);
        characterSmoother_.reset(sr, 20.0f, 0.0f);
        driftSmoother_.reset(sr, 500.0f, 0.0f);
        preHpSmoother_.reset(sr, 30.0f, 0.707f, 20.0f);
        postTiltFreqSmoother_.reset(sr, 30.0f, 0.707f, 1000.0f);
        postTiltGainSmoother_.reset(sr, 30.0f, 0.0f);
        outputGainSmoother_.reset(sr, 20.0f, 0.0f);
        crossfader_.reset(sr, 10.0f, 1.0f);

        leftDrift_.prepare(sr);
        rightDrift_.prepare(sr);
        leftDrift_.reseed(0x9E3779B97F4A7C15ULL);
        rightDrift_.reseed(0xBF58476D1CE4E5B9ULL);

        reset();
        prepared_ = true;
    }

    /**
     * @brief Clears all internal states, phase memory, and history buffers.
     * 
     * @note **Real-Time Safe.** Can be called safely from the audio thread to prevent clicks 
     * when the transport stops or loops.
     */
    void reset()
    {
        for (auto& algo : pool_)
            if (algo) algo->reset();

        preFilter_.reset();
        postFilter_.reset();
        dcBlocker_.reset();
        dryWetMixer_.reset();
        if (oversampler_) oversampler_->reset();

        driveSmoother_.skip();
        mixSmoother_.skip();
        characterSmoother_.skip();
        driftSmoother_.skip();
        preHpSmoother_.skip();
        postTiltFreqSmoother_.skip();
        postTiltGainSmoother_.skip();
        outputGainSmoother_.skip();
        crossfader_.setCurrentAndTargetValue(1.0f);

        prevBlendSample_.fill(SampleType(0));
        prevSlewSample_.fill(SampleType(0));
    }

    // -- Audio Processing ----------------------------------------------------

    /**
     * @brief Processes an audio block in-place (AudioProcessor standard contract).
     * @param buffer Mutable view of the audio data.
     */
    void processBlock(AudioBufferView<SampleType> buffer) noexcept { if (!prepared_) return; process(buffer); }

    /**
     * @brief The core audio processing pipeline.
     *
     * Executes the following sequence: parameter updates (lock-free) -> Mid/Side Encoding ->
     * Pre-filtering -> Slew detection -> Drift generation -> Saturation -> Adaptive Blend ->
     * Mid/Side Decoding -> Post-filtering -> DC Blocking -> Dry/Wet Mix.
     *
     * @note **Real-Time Safe.** Call this inside your audio callback.
     * @pre `prepare()` must have been called successfully prior to execution.
     * 
     * @param buffer Mutable view of the audio data. Will be modified in-place.
     */
    void process(AudioBufferView<SampleType> buffer) noexcept
    {
        if (!prepared_) return;
        handleParameterChanges();

        dryWetMixer_.pushDry(buffer);

        // Pre-filter
        {
            const int numSamples = buffer.getNumSamples();
            constexpr int kCoefRefresh = 16;
            for (int i = 0; i < numSamples; i += kCoefRefresh)
            {
                const int chunk = std::min(kCoefRefresh, numSamples - i);
                float curFreq = lastPreHpFreq_;
                for (int k = 0; k < chunk; ++k)
                    curFreq = preHpSmoother_.getNextValue();

                if (curFreq != lastPreHpFreq_)
                {
                    auto c = BiquadCoeffs<SampleType>::makeHighPass(spec_.sampleRate, static_cast<double>(curFreq));
                    preFilter_.setCoeffs(c);
                    lastPreHpFreq_ = curFreq;
                }
                auto subView = buffer.getSubView(i, chunk);
                preFilter_.processBlock(subView);
            }
        }

        const bool isMidSide = (procMode_ == ProcessingMode::MidOnly || procMode_ == ProcessingMode::SideOnly || procMode_ == ProcessingMode::MidSide) && buffer.getNumChannels() == 2;
        if (isMidSide) MidSide<SampleType>::encode(buffer);

        if (oversamplingFactor_ > 1 && oversampler_)
        {
            auto upView = oversampler_->upsample(buffer);
            processSaturationPipeline(upView);
            oversampler_->downsample(buffer);
        }
        else
        {
            processSaturationPipeline(buffer);
        }

        if (isMidSide) MidSide<SampleType>::decode(buffer);

        // Post-filter
        {
            const int numSamples = buffer.getNumSamples();
            constexpr int kCoefRefresh = 16;
            for (int i = 0; i < numSamples; i += kCoefRefresh)
            {
                const int chunk = std::min(kCoefRefresh, numSamples - i);
                float curFreq = lastPostTiltFreq_;
                float curGain = lastPostTiltGain_;
                for (int k = 0; k < chunk; ++k)
                {
                    curFreq = postTiltFreqSmoother_.getNextValue();
                    curGain = postTiltGainSmoother_.getNextValue();
                }

                if (curFreq != lastPostTiltFreq_ || curGain != lastPostTiltGain_)
                {
                    auto c = BiquadCoeffs<SampleType>::makePeak(spec_.sampleRate, static_cast<double>(curFreq), 0.707, static_cast<double>(curGain));
                    postFilter_.setCoeffs(c);
                    lastPostTiltFreq_ = curFreq;
                    lastPostTiltGain_ = curGain;
                }
                auto subView = buffer.getSubView(i, chunk);
                postFilter_.processBlock(subView);
            }
        }

        if (dcBlockingEnabled_) dcBlocker_.processBlock(buffer);
        applyOutputGain(buffer);

        // Mix Output
        if (outputMode_ == OutputMode::WetOnly) {} 
        else if (outputMode_ == OutputMode::Delta)
        {
            const int nCh = std::min(buffer.getNumChannels(), dryWetMixer_.getDryNumChannels());
            const int nS  = std::min(buffer.getNumSamples(), dryWetMixer_.getDryCapturedSamples());
            for (int ch = 0; ch < nCh; ++ch)
            {
                SampleType*       wet = buffer.getChannel(ch);
                const SampleType* dry = dryWetMixer_.getDryChannel(ch);
                for (int i = 0; i < nS; ++i) wet[i] -= dry[i];
            }
        }
        else 
        {
            dryWetMixer_.mixWet(buffer, static_cast<SampleType>(mixSmoother_.getTargetValue()));
        }
    }

    // -- Thread-Safe Setters (GUI / Automation Thread) -----------------------

    /**
     * @brief Sets the saturation algorithm topology.
     * @note Thread-Safe. Changes are crossfaded internally over 10ms to prevent clicks.
     * @param algo The desired algorithm (e.g., Algorithm::Tube).
     */
    void setAlgorithm(Algorithm algo)       { pushParam([&](auto& p){ p.algorithm = algo; }); }

    /**
     * @brief Sets the input drive gain.
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param dB Drive level in decibels. Range: [-24.0, +48.0]. 0 dB = unity gain.
     */
    void setDrive(SampleType dB)            { pushParam([&](auto& p){ p.driveDb = dB; }); }

    /**
     * @brief Sets the global Dry/Wet blend.
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param mix01 Mix ratio. Range: [0.0 (fully dry), 1.0 (fully wet)].
     */
    void setMix(SampleType mix01)           { pushParam([&](auto& p){ p.mix = mix01; }); }

    /**
     * @brief Adjusts the specific character/bias of the selected algorithm.
     * 
     * Behavior varies per algorithm:
     * - Tube: Controls waveform asymmetry.
     * - Tape: Controls hysteresis width and knee hardness.
     * - Transformer: Adjusts bias voltage.
     * 
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param c Character amount. Range: [-1.0, 1.0]. 0.0 represents the neutral state.
     */
    void setCharacter(SampleType c)         { pushParam([&](auto& p){ p.character = c; }); }

    /**
     * @brief Sets the routing configuration for multi-channel processing.
     * @note Thread-Safe. Applies instantly on the next block.
     * @param m The processing mode (Stereo, MidOnly, SideOnly, MidSide).
     */
    void setProcessingMode(ProcessingMode m){ pushParam([&](auto& p){ p.processingMode = m; }); }

    /**
     * @brief Sets the output signal path.
     * @note Thread-Safe. 
     * @param m The output mode (Normal, WetOnly, Delta). Delta mode outputs ONLY the generated harmonics.
     */
    void setOutputMode(OutputMode m)        { pushParam([&](auto& p){ p.outputMode = m; }); }

    /**
     * @brief Injects true-stereo pseudo-random low-frequency modulation (drift) into the saturation drive.
     * @note Thread-Safe. Smoothed internally over 500ms.
     * @param i Intensity of the drift. Range: [0.0, 1.0].
     */
    void setAnalogDrift(SampleType i)       { pushParam([&](auto& p){ p.analogDrift = i; }); }

    /**
     * @brief Configures a pre-saturation high-pass filter.
     * @note Thread-Safe. Smoothed internally over 30ms.
     * @param hz Cutoff frequency in Hertz. Range: [10.0, Nyquist].
     */
    void setPreFilterHpFrequency(SampleType hz) { pushParam([&](auto& p){ p.preFilterHpFreq = hz; }); }

    /**
     * @brief Sets the post-saturation make-up or trim gain.
     * @note Thread-Safe. Smoothed internally over 20ms.
     * @param dB Gain in decibels.
     */
    void setOutputGain(SampleType dB)       { pushParam([&](auto& p){ p.outputGain = dB; }); }

    /**
     * @brief Enables or disables the fixed 10Hz DC Blocker.
     * @note Thread-Safe.
     * @param on True to enable DC blocking (default).
     */
    void setDcBlocking(bool on)             { pushParam([&](auto& p){ p.dcBlocking = on; }); }

    /**
     * @brief Enables program-dependent saturation density.
     * @note Thread-Safe. Uses memory_order_relaxed atomic assignment.
     * @param on True to enable adaptive dynamic blending.
     */
    void setAdaptiveBlend(bool on) noexcept { adaptiveBlend_.store(on, std::memory_order_relaxed); }

    /**
     * @brief Sets a derivative-based (slew rate) saturation multiplier.
     * @note Thread-Safe. Uses memory_order_relaxed atomic assignment.
     * @param amount Sensitivity multiplier. Range: [0.0 (off), 1.0 (max)].
     */
    void setSlewSensitivity(SampleType amount) noexcept { slewSensitivity_.store(std::clamp(amount, SampleType(0), SampleType(1)), std::memory_order_relaxed); }
    
    /**
     * @brief Configures a post-saturation proportional-Q tilt EQ.
     * @note Thread-Safe. Both parameters smoothed internally over 30ms.
     * @param centerHz Pivot frequency in Hertz. Range: [100.0, Nyquist].
     * @param amountDb Gain at the extremes. Positive values brighten the signal; negative values darken it.
     */
    void setPostFilterTilt(SampleType centerHz, SampleType amountDb)
    {
        pushParam([&](auto& p){ p.postFilterTiltFreq = centerHz; p.postFilterTiltGain = amountDb; });
    }

    /**
     * @brief Configures internal polyphase oversampling to reduce aliasing.
     * 
     * @note **NOT Real-Time Safe.** Must be called before `prepare()`, or `prepare()` 
     * must be called immediately after.
     * 
     * @param factor The oversampling multiplier. Must be a power of 2 (1, 2, 4, 8, 16). 1 = Off.
     */
    void setOversampling(int factor)
    {
        if (factor < 1 || (factor & (factor - 1)) != 0) return;
        oversamplingFactor_ = factor;
        if (factor > 1)
        {
            oversampler_ = std::make_unique<Oversampling<SampleType>>(factor);
            if (spec_.sampleRate > 0) oversampler_->prepare(spec_);
        }
        else
            oversampler_.reset();

        // If prepare() already ran, grow the scratch buffers to hold the
        // upsampled block and realign the dry path; otherwise prepare() does it.
        if (prepared_)
        {
            const int upBlock = spec_.maxBlockSize * std::max(1, oversamplingFactor_);
            if (tempBuffer_.getNumSamples() < upBlock)
                tempBuffer_.resize(spec_.numChannels, upBlock);
            if (driftBuffer_.getNumSamples() < upBlock)
                driftBuffer_.resize(spec_.numChannels, upBlock);

            dryWetMixer_.setLatencyCompensation(
                (oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0);
        }
    }

    // -- Thread-Safe Getters (GUI / Metering) --------------------------------

    /** 
     * @brief Retrieves the current oversampling factor. 
     * @return Multiplier (1, 2, 4, 8, 16).
     */
    [[nodiscard]] int getOversamplingFactor() const noexcept { return oversamplingFactor_; }

    /**
     * @brief Reports the processor's algorithmic latency in samples.
     * @note Equals the oversampler group delay (0 when oversampling is off).
     *       Report this to the host for plugin delay compensation (PDC).
     */
    [[nodiscard]] int getLatencySamples() const noexcept
    {
        return (oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0;
    }

    /** 
     * @brief Retrieves the currently active underlying algorithm.
     * @note Real-Time Safe reading via relaxed atomics.
     * @return The active Algorithm enum.
     */
    [[nodiscard]] Algorithm getCurrentAlgorithm() const noexcept { return currentAlgoType_.load(std::memory_order_relaxed); }

    /** 
     * @brief Calculates the peak gain reduction (clipping amount) for metering.
     * @note Real-Time Safe reading via relaxed atomics.
     * @return Gain reduction in decibels (always <= 0.0).
     */
    [[nodiscard]] SampleType getGainReductionDb() const noexcept { return gainReductionDb_.load(std::memory_order_relaxed); }

protected:
    struct Params
    {
        Algorithm      algorithm      = Algorithm::SoftClip;
        ProcessingMode processingMode = ProcessingMode::Stereo;
        OutputMode     outputMode     = OutputMode::Normal;
        SampleType     driveDb        = SampleType(0);
        SampleType     mix            = SampleType(1);
        SampleType     character      = SampleType(0);
        SampleType     analogDrift    = SampleType(0);
        SampleType     preFilterHpFreq    = SampleType(20);
        SampleType     postFilterTiltFreq = SampleType(1000);
        SampleType     postFilterTiltGain = SampleType(0);
        SampleType     outputGain         = SampleType(0);
        bool           dcBlocking         = true;
    };

    template <typename Fn>
    void pushParam(Fn&& mutate)
    {
        SpinLock::ScopedLock guard(paramsLock_);
        mutate(lastParams_);
        paramQueue_.push(lastParams_);
    }

    void handleParameterChanges()
    {
        Params p;
        while (paramQueue_.pop(p))
        {
            driveSmoother_.setTargetValue(std::clamp(static_cast<float>(p.driveDb), -24.0f, 48.0f));
            mixSmoother_.setTargetValue(std::clamp(static_cast<float>(p.mix), 0.0f, 1.0f));
            characterSmoother_.setTargetValue(std::clamp(static_cast<float>(p.character), -1.0f, 1.0f));
            driftSmoother_.setTargetValue(std::clamp(static_cast<float>(p.analogDrift), 0.0f, 1.0f));
            outputGainSmoother_.setTargetValue(static_cast<float>(p.outputGain));

            float nyquist = static_cast<float>(spec_.sampleRate) / 2.0f;
            preHpSmoother_.setTargetValue(std::clamp(static_cast<float>(p.preFilterHpFreq), 10.0f, nyquist));
            postTiltFreqSmoother_.setTargetValue(std::clamp(static_cast<float>(p.postFilterTiltFreq), 100.0f, nyquist));
            postTiltGainSmoother_.setTargetValue(std::clamp(static_cast<float>(p.postFilterTiltGain), -12.0f, 12.0f));

            dcBlockingEnabled_ = p.dcBlocking;
            procMode_   = p.processingMode;
            outputMode_ = p.outputMode;
            currentAlgoType_.store(p.algorithm, std::memory_order_relaxed);

            auto* requested = pool_[static_cast<int>(p.algorithm)].get();
            if (active_.load() != requested && next_.load() != requested)
            {
                next_.store(requested);
                crossfader_.setTargetValue(0.0f);
            }
        }
    }

    // Data-Oriented Pipeline to eliminate per-sample virtual dispatch
    void processSaturationPipeline(AudioBufferView<SampleType> buffer)
    {
        auto* primary   = active_.load();
        auto* secondary = next_.load();
        bool  xfading   = secondary != nullptr && crossfader_.isSmoothing();

        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();

        auto driveDbTarget = static_cast<SampleType>(driveSmoother_.getTargetValue());
        auto driveGainTarget = decibelsToGain(driveDbTarget);
        auto characterTarget = static_cast<SampleType>(characterSmoother_.getTargetValue());

        AudioSpec updateSpec = spec_;
        if (oversamplingFactor_ > 1) updateSpec.sampleRate *= oversamplingFactor_;

        if (primary)   primary->update(driveGainTarget, characterTarget, updateSpec);
        if (secondary) secondary->update(driveGainTarget, characterTarget, updateSpec);

        SampleType peakInOriginal = SampleType(0);
        for (int ch = 0; ch < nCh; ++ch) {
            const SampleType* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i) peakInOriginal = std::max(peakInOriginal, std::abs(d[i]));
        }

        // 1. Slew (In-place)
        auto slewAmt = slewSensitivity_.load(std::memory_order_relaxed);
        if (slewAmt > SampleType(0))
        {
            for (int ch = 0; ch < nCh; ++ch) {
                SampleType* data = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i) {
                    SampleType dry = data[i];
                    SampleType delta = dry - prevSlewSample_[ch];
                    data[i] += std::tanh(std::abs(delta)) * slewAmt * dry;
                    prevSlewSample_[ch] = dry;
                }
            }
        }

        // 2. Pre-generate Drift
        auto driftIntensity = driftSmoother_.getTargetValue();
        bool useDrift = driftIntensity > 0.01f;
        if (useDrift)
        {
            auto driftView = driftBuffer_.toView();
            for (int i = 0; i < nS; ++i) {
                auto driftS = driftSmoother_.getNextValue();
                for (int ch = 0; ch < nCh; ++ch) {
                    SampleType noise = (ch == 0) ? leftDrift_.getNextSample() : rightDrift_.getNextSample();
                    driftView.getChannel(ch)[i] = SampleType(1) + static_cast<SampleType>(driftS) * noise;
                }
            }
        }

        // 3. Primary Saturation (and secondary if xfade)
        if (xfading)
        {
            auto tempView = tempBuffer_.toView().getSubView(0, nS);
            for (int ch = 0; ch < nCh; ++ch)
                std::memcpy(tempView.getChannel(ch), buffer.getChannel(ch), static_cast<std::size_t>(nS) * sizeof(SampleType));
            
            dispatchSaturator(primary, buffer, useDrift);
            dispatchSaturator(secondary, tempView, useDrift);

            for (int i = 0; i < nS; ++i) {
                auto fade = static_cast<SampleType>(crossfader_.getNextValue());
                for (int ch = 0; ch < nCh; ++ch) {
                    SampleType* out = buffer.getChannel(ch);
                    const SampleType* alt = tempView.getChannel(ch);
                    out[i] = out[i] * fade + alt[i] * (SampleType(1) - fade);
                }
            }

            if (!crossfader_.isSmoothing()) {
                active_.store(secondary);
                next_.store(nullptr);
                if (primary) primary->reset();
                crossfader_.setCurrentAndTargetValue(1.0f);
            }
        }
        else
        {
            dispatchSaturator(primary, buffer, useDrift);
        }

        // 4. Adaptive Blend
        if (adaptiveBlend_.load(std::memory_order_relaxed))
        {
            for (int ch = 0; ch < nCh; ++ch) {
                SampleType* wetData = buffer.getChannel(ch);
                const SampleType* dryData = dryWetMixer_.getDryChannel(ch);
                for (int i = 0; i < nS; ++i) {
                    SampleType dry = dryData[i];
                    SampleType wet = wetData[i];
                    SampleType avg = (std::abs(prevBlendSample_[ch]) + std::abs(dry)) * SampleType(0.5);
                    SampleType apply = std::clamp(avg, SampleType(0), SampleType(1));
                    wetData[i] = dry * (SampleType(1) - apply) + wet * apply;
                    prevBlendSample_[ch] = dry;
                }
            }
        }

        // M/S Routing
        for (int ch = 0; ch < nCh; ++ch) {
            if ((procMode_ == ProcessingMode::MidOnly && ch == 1) || (procMode_ == ProcessingMode::SideOnly && ch == 0)) {
                SampleType* wetData = buffer.getChannel(ch);
                const SampleType* dryData = dryWetMixer_.getDryChannel(ch);
                std::memcpy(wetData, dryData, static_cast<std::size_t>(nS) * sizeof(SampleType));
            }
        }

        // Gain Reduction Tracking
        SampleType peakOut = SampleType(0);
        for (int ch = 0; ch < nCh; ++ch) {
            const SampleType* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i) peakOut = std::max(peakOut, std::abs(d[i]));
        }

        SampleType peakInDriven = peakInOriginal * driveGainTarget;
        if (peakInDriven > SampleType(1e-6)) {
            SampleType ratio = std::min(peakOut / peakInDriven, SampleType(1));
            gainReductionDb_.store(gainToDecibels(ratio, SampleType(-100)), std::memory_order_relaxed);
        } else {
            gainReductionDb_.store(SampleType(0), std::memory_order_relaxed);
        }
    }

    // Compile-time resolver (Zero virtual dispatch in hot path)
    void dispatchSaturator(detail::SaturationAlgorithm<SampleType>* baseAlgo, AudioBufferView<SampleType> buffer, bool useDrift)
    {
        if (!baseAlgo) return;
        switch (baseAlgo->getType())
        {
            case Algorithm::Tube:        processCore(static_cast<detail::TubeAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Tape:        processCore(static_cast<detail::TapeAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Transformer: processCore(static_cast<detail::TransformerAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::SoftClip:    processCore(static_cast<detail::TanhAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::HardClip:    processCore(static_cast<detail::HardClipAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Exciter:     processCore(static_cast<detail::ExciterAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Wavefolder:  processCore(static_cast<detail::WavefolderAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Bitcrusher:  processCore(static_cast<detail::BitcrusherAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::Downsample:  processCore(static_cast<detail::DownsampleAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
            case Algorithm::MultiStage:  processCore(static_cast<detail::MultiStageAlgorithm<SampleType>*>(baseAlgo), buffer, useDrift); break;
        }
    }

    template <typename ExactAlgo>
    void processCore(ExactAlgo* algo, AudioBufferView<SampleType> buffer, bool useDrift)
    {
        const int nCh = buffer.getNumChannels();
        const int nS  = buffer.getNumSamples();
        auto driftView = driftBuffer_.toView();

        for (int i = 0; i < nS; ++i)
        {
            auto driveGainS = decibelsToGain(static_cast<SampleType>(driveSmoother_.getNextValue()));
            auto charS      = static_cast<SampleType>(characterSmoother_.getNextValue());

            for (int ch = 0; ch < nCh; ++ch)
            {
                SampleType drift = useDrift ? driftView.getChannel(ch)[i] : SampleType(1);
                SampleType* data = buffer.getChannel(ch);
                data[i] = algo->processSample(data[i], driveGainS * drift, charS * drift, ch);
            }
        }
    }

    void applyOutputGain(AudioBufferView<SampleType> buffer)
    {
        auto targetGainDb = outputGainSmoother_.getTargetValue();
        if (std::abs(targetGainDb - outputGainSmoother_.getCurrentValue()) < 0.001f)
        {
            buffer.applyGain(decibelsToGain(static_cast<SampleType>(targetGainDb)));
        }
        else
        {
            const int nCh = buffer.getNumChannels();
            const int nS  = buffer.getNumSamples();
            for (int i = 0; i < nS; ++i)
            {
                auto gain = decibelsToGain(static_cast<SampleType>(outputGainSmoother_.getNextValue()));
                for (int ch = 0; ch < nCh; ++ch) buffer.getChannel(ch)[i] *= gain;
            }
        }
    }

    AudioSpec spec_ {};
    bool prepared_ = false;

    static constexpr int kNumAlgorithms = 10;
    std::array<std::unique_ptr<detail::SaturationAlgorithm<SampleType>>, kNumAlgorithms> pool_;
    std::atomic<detail::SaturationAlgorithm<SampleType>*> active_ { nullptr };
    std::atomic<detail::SaturationAlgorithm<SampleType>*> next_   { nullptr };

    SpscQueue<Params>  paramQueue_;
    Params             lastParams_;
    SpinLock           paramsLock_;

    ProcessingMode procMode_   = ProcessingMode::Stereo;
    OutputMode     outputMode_ = OutputMode::Normal;
    bool           dcBlockingEnabled_ = true;
    std::atomic<Algorithm> currentAlgoType_ { Algorithm::SoftClip };

    Smoothers::StateVariableSmoother driveSmoother_, preHpSmoother_, postTiltFreqSmoother_;
    Smoothers::LinearSmoother        mixSmoother_, characterSmoother_, driftSmoother_,
                                     outputGainSmoother_, postTiltGainSmoother_;
    Smoothers::LinearSmoother        crossfader_;

    Biquad<SampleType> preFilter_, postFilter_, dcBlocker_;
    DryWetMixer<SampleType> dryWetMixer_;

    AnalogRandom::Generator<SampleType> leftDrift_, rightDrift_;

    AudioBuffer<SampleType> tempBuffer_;
    AudioBuffer<SampleType> driftBuffer_;

    std::unique_ptr<Oversampling<SampleType>> oversampler_;
    int oversamplingFactor_ = 1;

    std::atomic<SampleType> gainReductionDb_ { SampleType(0) };

    std::atomic<bool> adaptiveBlend_ { false };
    static constexpr int kMaxCh = 16;
    std::array<SampleType, kMaxCh> prevBlendSample_ {};

    std::atomic<SampleType> slewSensitivity_ { SampleType(0) };
    std::array<SampleType, kMaxCh> prevSlewSample_ {};

    float lastPreHpFreq_    = -1.0f;
    float lastPostTiltFreq_ = -1.0f;
    float lastPostTiltGain_ = std::numeric_limits<float>::quiet_NaN();
};

} // namespace dspark