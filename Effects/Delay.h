// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Delay.h
 * @brief Professional delay line with Hermite interpolation, analog feedback saturation, and stereo processing.
 *
 * Features:
 * - Power-of-two circular buffer with bitmask wrapping.
 * - 3rd-order Hermite interpolation for artifact-free fractional delays.
 * - Analog-modeled feedback loop with soft clipping and one-pole filtering.
 * - Multiple smoother choices for glitch-free parameter modulation.
 * - Fully independent channel write indices for safe interleaved or sequential processing.
 * - Zero-allocation, lock-free, cache-friendly architecture.
 *
 * @tparam SampleType float or double.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Smoothers.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>

namespace dspark {

template <typename SampleType>
class Delay final
{
public:
    enum class SmootherType
    {
        None, Linear, Exponential, OnePole, MultiPole2,
        Asymmetric, SlewLimiter, StateVariable, Butterworth, CriticallyDamped
    };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the delay structures and allocates memory. Must be called before processing.
     * @param spec The audio specification (sample rate, block size, etc.).
     * @param maxDelaySeconds The maximum theoretical delay time required.
     */
    void prepare(const AudioSpec& spec, double maxDelaySeconds)
    {
        sampleRate_ = static_cast<SampleType>(spec.sampleRate);
        numChannels_ = spec.numChannels;
        blockSize_   = spec.maxBlockSize;

        int required = static_cast<int>(std::ceil(maxDelaySeconds * spec.sampleRate));
        maxDelaySamples_ = nextPow2(required + blockSize_);
        bufferMask_ = maxDelaySamples_ - 1;

        delayBuffer_.resize(numChannels_, maxDelaySamples_);
        wetBuffer_.resize(numChannels_, blockSize_);

        float timeMs = smoothingTimeMs_.load(std::memory_order_relaxed);
        for (int ch = 0; ch < std::min(numChannels_, kMaxChannels); ++ch)
        {
            resetChannelSmoothers(states_[ch], timeMs, 0.0f);
            writeIndices_[ch] = 0;
        }

        mixSmoother_.reset(spec.sampleRate, timeMs, 1.0f);
        reset();
        
        prepared_.store(true, std::memory_order_release);
    }

    /**
     * @brief Prepares the delay with a maximum capacity in milliseconds.
     * @param spec The audio specification.
     * @param maxDelayMs Maximum delay capacity in milliseconds.
     */
    void prepareMs(const AudioSpec& spec, double maxDelayMs)
    {
        prepare(spec, maxDelayMs / 1000.0);
    }

    /**
     * @brief Clears the delay buffers and resets all filter and feedback states.
     */
    void reset() noexcept
    {
        delayBuffer_.clear();
        wetBuffer_.clear();
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            writeIndices_[ch] = 0;
            resetChannelState(states_[ch]);
        }
        mixSmoother_.skip();
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Sets the smoothing algorithm used for delay time changes.
     * @param type The desired SmootherType.
     */
    void setSmoother(SmootherType type) noexcept 
    { 
        smootherType_.store(type, std::memory_order_relaxed); 
        smootherDirty_.store(true, std::memory_order_relaxed); 
    }

    /**
     * @brief Sets the smoothing transition time.
     * @param ms Time in milliseconds.
     */
    void setSmoothingTime(float ms) noexcept
    {
        smoothingTimeMs_.store(std::max(0.0f, ms), std::memory_order_relaxed);
        smootherDirty_.store(true, std::memory_order_relaxed);
    }

    // -- Parameters ----------------------------------------------------------

    /**
     * @brief Sets the delay time in samples.
     * @param samples Target delay time (can be fractional).
     */
    void setDelaySamples(SampleType samples) noexcept
    {
        samples = std::clamp(samples, SampleType(0), static_cast<SampleType>(maxDelaySamples_ - 1));
        globalDelay_.store(samples, std::memory_order_relaxed);
        updateSmootherTargets(samples);
    }

    /** @brief Sets the delay time in milliseconds. */
    void setDelayMs(SampleType ms) noexcept { setDelaySamples(ms * sampleRate_ / SampleType(1000)); }

    /** @brief Sets the delay time in seconds. */
    void setDelaySeconds(SampleType secs) noexcept { setDelaySamples(secs * sampleRate_); }

    /** @brief Returns the current target delay time in samples. */
    [[nodiscard]] SampleType getCurrentDelaySamples() const noexcept { return globalDelay_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets the global feedback amount.
     * @param gain Feedback gain multiplier [-1.0, 1.0]. Will be saturated internally if exceeded.
     */
    void setFeedback(SampleType gain) noexcept
    {
        feedbackGain_.store(gain, std::memory_order_relaxed);
    }

    /** @brief Sets the cutoff frequency for the feedback low-pass filter (0 to disable). */
    void setFeedbackLpHz(SampleType freq) noexcept
    {
        fbLpCoef_.store((freq > 0) ? calcLpCoef(freq) : SampleType(0), std::memory_order_relaxed);
    }

    /** @brief Sets the cutoff frequency for the feedback high-pass filter (0 to disable). */
    void setFeedbackHpHz(SampleType freq) noexcept
    {
        fbHpCoef_.store((freq > 0) ? calcHpCoef(freq) : SampleType(0), std::memory_order_relaxed);
    }

    // -- Sample processing ---------------------------------------------------

    /**
     * @brief Processes a single sample for a specific channel.
     * @param ch Channel index.
     * @param input Input sample.
     * @return Processed delayed sample (wet only).
     */
    SampleType processSample(int ch, SampleType input) noexcept
    {
        if (ch < 0 || ch >= numChannels_ || !prepared_.load(std::memory_order_acquire)) return input;
        maybeUpdateSmoothers();
        
        auto& s = states_[ch];
        auto st = smootherType_.load(std::memory_order_relaxed);
        SampleType delay = (st == SmootherType::None) 
            ? globalDelay_.load(std::memory_order_relaxed) : advanceSmoother(s, st);
            
        SampleType out = processSampleInternal(ch, input, delay, s, 
                                               feedbackGain_.load(std::memory_order_relaxed), 
                                               fbLpCoef_.load(std::memory_order_relaxed), 
                                               fbHpCoef_.load(std::memory_order_relaxed));
        advanceWriteIndex(ch);
        return out;
    }

    // -- Block processing ----------------------------------------------------

    /**
     * @brief Processes a multi-channel block in-place.
     * @param buffer The audio buffer to process.
     * @param delayMs Delay time in milliseconds.
     * @param feedback Feedback gain multiplier.
     * @param lpHz Low-pass filter cutoff in Hz.
     * @param hpHz High-pass filter cutoff in Hz.
     */
    void processBlock(AudioBufferView<SampleType> buffer, SampleType delayMs,
                      SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        if (!prepared_.load(std::memory_order_acquire)) return;

        setDelayMs(delayMs);
        setFeedback(feedback);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        const int nS = buffer.getNumSamples();
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        
        // Cache atomic parameters to local stack for auto-vectorization & performance
        const auto smoothType = smootherType_.load(std::memory_order_relaxed);
        const SampleType targetDelay = globalDelay_.load(std::memory_order_relaxed);
        const SampleType fb = feedbackGain_.load(std::memory_order_relaxed);
        const SampleType lpC = fbLpCoef_.load(std::memory_order_relaxed);
        const SampleType hpC = fbHpCoef_.load(std::memory_order_relaxed);

        for (int ch = 0; ch < nCh; ++ch)
        {
            SampleType* data = buffer.getChannel(ch);
            auto& s = states_[ch];
            
            for (int i = 0; i < nS; ++i)
            {
                SampleType currentDelay = (smoothType == SmootherType::None) ? targetDelay : advanceSmoother(s, smoothType);
                data[i] = processSampleInternal(ch, data[i], currentDelay, s, fb, lpC, hpC);
                advanceWriteIndex(ch);
            }
        }
    }

    /**
     * @brief Processes a single channel block in-place. Safe to call sequentially for different channels.
     */
    void processChannel(AudioBufferView<SampleType> buffer, int ch, SampleType delayMs,
                        SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        if (ch < 0 || ch >= numChannels_ || !prepared_.load(std::memory_order_acquire)) return;

        setDelayMs(delayMs);
        setFeedback(feedback);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        SampleType* data = buffer.getChannel(ch);
        const int nS = buffer.getNumSamples();
        
        const auto smoothType = smootherType_.load(std::memory_order_relaxed);
        const SampleType targetDelay = globalDelay_.load(std::memory_order_relaxed);
        const SampleType fb = feedbackGain_.load(std::memory_order_relaxed);
        const SampleType lpC = fbLpCoef_.load(std::memory_order_relaxed);
        const SampleType hpC = fbHpCoef_.load(std::memory_order_relaxed);
        
        auto& s = states_[ch];

        for (int i = 0; i < nS; ++i)
        {
            SampleType currentDelay = (smoothType == SmootherType::None) ? targetDelay : advanceSmoother(s, smoothType);
            data[i] = processSampleInternal(ch, data[i], currentDelay, s, fb, lpC, hpC);
            advanceWriteIndex(ch);
        }
    }

    // -- Wet buffer processing -----------------------------------------------

    /** @brief Copies a dry buffer into the internal wet buffer. */
    void pushDryToWet(AudioBufferView<const SampleType> dry) noexcept
    {
        pushDryToWetImpl([&](int ch) { return dry.getChannel(ch); }, dry.getNumChannels(), dry.getNumSamples());
    }

    /** @brief Copies a dry buffer into the internal wet buffer. */
    void pushDryToWet(AudioBufferView<SampleType> dry) noexcept
    {
        pushDryToWetImpl([&](int ch) { return dry.getChannel(ch); }, dry.getNumChannels(), dry.getNumSamples());
    }

    /** @brief Processes the wet buffer in place with current settings. */
    void processWet(SampleType delayMs, SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        processBlock(wetBuffer_.toView(), delayMs, feedback, lpHz, hpHz);
    }

    /**
     * @brief Processes a true ping-pong delay (L feeds R, R feeds L).
     */
    void processPingPong(SampleType delayMs, SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        if (wetBuffer_.getNumChannels() < 2 || !prepared_.load(std::memory_order_acquire)) return;
        
        setDelayMs(delayMs);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        SampleType* L = wetBuffer_.getChannel(0);
        SampleType* R = wetBuffer_.getChannel(1);
        const int nS = wetBuffer_.getNumSamples();

        const auto smoothType = smootherType_.load(std::memory_order_relaxed);
        const SampleType targetDelay = globalDelay_.load(std::memory_order_relaxed);
        const SampleType lpC = fbLpCoef_.load(std::memory_order_relaxed);
        const SampleType hpC = fbHpCoef_.load(std::memory_order_relaxed);

        for (int i = 0; i < nS; ++i)
        {
            auto& sL = states_[0];
            auto& sR = states_[1];
            
            SampleType currentDelay = (smoothType == SmootherType::None) ? targetDelay : advanceSmoother(sL, smoothType);
            advanceSmoother(sR, smoothType); // Keep smoothers in sync

            SampleType inL = L[i] + sR.pingPongFb;
            SampleType inR = R[i] + sL.pingPongFb;

            SampleType outL = processSampleInternal(0, inL, currentDelay, sL, SampleType(0), lpC, hpC);
            SampleType outR = processSampleInternal(1, inR, currentDelay, sR, SampleType(0), lpC, hpC);

            sL.pingPongFb = processFbFilters(0, outR * feedback, lpC, hpC);
            sR.pingPongFb = processFbFilters(1, outL * feedback, lpC, hpC);

            L[i] = outL;
            R[i] = outR;
            advanceWriteIndex(0);
            advanceWriteIndex(1);
        }
    }

    /**
     * @brief Mixes the processed wet buffer back into the provided dry buffer.
     * @param dry Buffer containing the dry signal.
     * @param mix Mix ratio (0.0 = 100% dry, 1.0 = 100% wet).
     */
    void mixWetToDry(AudioBufferView<SampleType> dry, SampleType mix) noexcept
    {
        mix = std::clamp(mix, SampleType(0), SampleType(1));
        auto st = smootherType_.load(std::memory_order_relaxed);
        if (st != SmootherType::None)
            mixSmoother_.setTargetValue(static_cast<float>(mix));

        const int nS = std::min(dry.getNumSamples(), wetBuffer_.getNumSamples());
        const int nCh = std::min(dry.getNumChannels(), wetBuffer_.getNumChannels());

        for (int ch = 0; ch < nCh; ++ch)
        {
            SampleType* d = dry.getChannel(ch);
            const SampleType* w = wetBuffer_.getChannel(ch);
            for (int i = 0; i < nS; ++i)
            {
                SampleType m = (st != SmootherType::None) 
                    ? static_cast<SampleType>(mixSmoother_.getNextValue()) : mix;
                d[i] = d[i] * (SampleType(1) - m) + w[i] * m;
            }
        }
    }

    /** @brief Exposes the internal wet buffer view. */
    AudioBufferView<SampleType> getWetView() noexcept { return wetBuffer_.toView(); }
    
    /** @brief Returns maximum capacity in samples. */
    int getMaxDelaySamples() const noexcept { return maxDelaySamples_; }

protected:
    static constexpr int kMaxChannels = 16;

    struct ChannelState
    {
        SampleType fbLpZ1 = 0, fbHpZ1 = 0;
        SampleType lastFeedback = 0;
        SampleType pingPongFb = 0;
        SampleType currentDelay = 0, targetDelay = 0;

        Smoothers::LinearSmoother lin;
        Smoothers::ExponentialSmoother exp;
        Smoothers::OnePoleSmoother onePole;
        Smoothers::MultiPoleSmoother<2> multi2;
        Smoothers::AsymmetricSmoother asym;
        Smoothers::SlewLimiter slew;
        Smoothers::StateVariableSmoother svf;
        Smoothers::ButterworthSmoother butter;
        Smoothers::CriticallyDampedSmoother crit;
    };

public:
    /** 
     * @brief Advances the write index for a specific channel. 
     * @param ch Channel index to advance.
     */
    void advanceWriteIndex(int ch) noexcept { writeIndices_[ch] = (writeIndices_[ch] + 1) & bufferMask_; }

private:

    template <typename ChannelFn>
    void pushDryToWetImpl(ChannelFn&& getCh, int dryCh, int drySamples) noexcept
    {
        const int nCh = std::min(dryCh, wetBuffer_.getNumChannels());
        const int nS  = std::min(drySamples, wetBuffer_.getNumSamples());
        for (int ch = 0; ch < nCh; ++ch)
            std::memcpy(wetBuffer_.getChannel(ch), getCh(ch), static_cast<std::size_t>(nS) * sizeof(SampleType));
    }

    void resetChannelState(ChannelState& s) noexcept
    {
        s.fbLpZ1 = s.fbHpZ1 = s.lastFeedback = s.pingPongFb = 0;
        s.currentDelay = s.targetDelay = 0;
    }

    void resetChannelSmoothers(ChannelState& s, float timeMs, float init) noexcept
    {
        double sr = static_cast<double>(sampleRate_);
        s.lin.reset(sr, timeMs, init);
        s.exp.reset(sr, timeMs, std::max(init, 1e-6f));
        s.onePole.reset(sr, timeMs, init);
        s.multi2.reset(sr, timeMs, init);
        s.asym.reset(sr, timeMs / 5.0f, timeMs, init);
        float timeSec = timeMs / 1000.0f;
        float maxRate = static_cast<float>(maxDelaySamples_) / std::max(timeSec, 1e-6f);
        s.slew.reset(sr, maxRate, init);
        s.svf.reset(sr, timeMs, 0.707f, init);
        s.butter.reset(sr, timeMs, init);
        s.crit.reset(sr, timeMs, init);
    }

    void maybeUpdateSmoothers() noexcept
    {
        if (!smootherDirty_.load(std::memory_order_relaxed)) return;
        smootherDirty_.store(false, std::memory_order_relaxed);
        float timeMs = smoothingTimeMs_.load(std::memory_order_relaxed);
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            auto& s = states_[ch];
            float cur = static_cast<float>(s.currentDelay);
            float tgt = static_cast<float>(s.targetDelay);
            resetChannelSmoothers(s, timeMs, cur);
            setSmootherTarget(s, static_cast<SampleType>(tgt));
        }
        mixSmoother_.reset(static_cast<double>(sampleRate_), timeMs, mixSmoother_.getCurrentValue());
    }

    void updateSmootherTargets(SampleType target) noexcept
    {
        if (smootherType_.load(std::memory_order_relaxed) == SmootherType::None) return;
        for (int ch = 0; ch < numChannels_; ++ch)
            setSmootherTarget(states_[ch], target);
    }

    void setSmootherTarget(ChannelState& s, SampleType target) noexcept
    {
        s.targetDelay = target;
        float t = static_cast<float>(target);
        switch (smootherType_.load(std::memory_order_relaxed))
        {
            case SmootherType::Linear:          s.lin.setTargetValue(t); break;
            case SmootherType::Exponential:     s.exp.setTargetValue(t); break;
            case SmootherType::OnePole:         s.onePole.setTargetValue(t); break;
            case SmootherType::MultiPole2:      s.multi2.setTargetValue(t); break;
            case SmootherType::Asymmetric:      s.asym.setTargetValue(t); break;
            case SmootherType::SlewLimiter:     s.slew.setTargetValue(t); break;
            case SmootherType::StateVariable:   s.svf.setTargetValue(t); break;
            case SmootherType::Butterworth:     s.butter.setTargetValue(t); break;
            case SmootherType::CriticallyDamped:s.crit.setTargetValue(t); break;
            default: break;
        }
    }

    inline SampleType advanceSmoother(ChannelState& s, SmootherType st) noexcept
    {
        float val;
        switch (st)
        {
            case SmootherType::Linear:          val = s.lin.getNextValue(); break;
            case SmootherType::Exponential:     val = s.exp.getNextValue(); break;
            case SmootherType::OnePole:         val = s.onePole.getNextValue(); break;
            case SmootherType::MultiPole2:      val = s.multi2.getNextValue(); break;
            case SmootherType::Asymmetric:      val = s.asym.getNextValue(); break;
            case SmootherType::SlewLimiter:     val = s.slew.getNextValue(); break;
            case SmootherType::StateVariable:   val = s.svf.getNextValue(); break;
            case SmootherType::Butterworth:     val = s.butter.getNextValue(); break;
            case SmootherType::CriticallyDamped:val = s.crit.getNextValue(); break;
            default: val = static_cast<float>(s.targetDelay);
        }
        s.currentDelay = static_cast<SampleType>(val);
        return s.currentDelay;
    }

    inline SampleType processSampleInternal(int ch, SampleType input, SampleType delaySamples,
                                            ChannelState& s, SampleType fbMult,
                                            SampleType lpC, SampleType hpC) noexcept
    {
        // 4-point Hermite needs read positions at idx0-1, idx0, idx0+1, idx0+2.
        // The write index points to the slot we are about to write THIS sample.
        // Slots writeIdx, writeIdx+1, writeIdx+2 still hold stale data from
        // the previous ring-buffer wrap. To keep all 4 interpolation taps
        // inside the already-written history we need delaySamples >= 3.
        // Below that, samples idx1 / idx2 leak data ~bufferSize samples old
        // which manifests as audible clicks (notably when a binaural / Haas
        // panner crosses the centre). 3 samples ≈ 62 µs at 48 kHz —
        // imperceptible and the same on both channels, so spatial cues
        // (centre image, ITD ratios) are preserved.
        constexpr SampleType kMinHermiteDelay = SampleType(3);
        delaySamples = std::clamp(delaySamples, kMinHermiteDelay,
                                  static_cast<SampleType>(maxDelaySamples_ - 1));

        SampleType* data = delayBuffer_.getChannel(ch);
        int writeIdx = writeIndices_[ch];

        SampleType readPos = static_cast<SampleType>(writeIdx) - delaySamples;
        if (readPos < SampleType(0)) readPos += static_cast<SampleType>(maxDelaySamples_);

        // 3rd-order Hermite Interpolation (4-point)
        int idx0 = static_cast<int>(readPos) & bufferMask_;
        SampleType frac = readPos - static_cast<SampleType>(idx0);

        int idxM1 = (idx0 - 1 + maxDelaySamples_) & bufferMask_;
        int idx1  = (idx0 + 1) & bufferMask_;
        int idx2  = (idx0 + 2) & bufferMask_;

        SampleType ym1 = data[idxM1];
        SampleType y0  = data[idx0];
        SampleType y1  = data[idx1];
        SampleType y2  = data[idx2];

        // Hermite polynomial logic
        SampleType c = (y1 - ym1) * SampleType(0.5);
        SampleType v = y0 - y1;
        SampleType w = c + v;
        SampleType a = w + v + (y2 - y0) * SampleType(0.5);
        SampleType b = w + a;
        SampleType delayed = ((((a * frac) - b) * frac + c) * frac + y0);

        // Feedback processing with Soft Clipper
        SampleType fbInput = delayed * fbMult;
        if (fbInput != SampleType(0))
        {
            fbInput = processFbFilters(ch, fbInput, lpC, hpC);
            // Analog-style soft saturation to prevent clipping explosions
            fbInput = std::tanh(fbInput); 
        }
        s.lastFeedback = fbInput;

        data[writeIdx] = input + s.lastFeedback;
        return delayed;
    }

    inline SampleType processFbFilters(int ch, SampleType sample, SampleType lpC, SampleType hpC) noexcept
    {
        auto& s = states_[ch];
        SampleType out = sample;

        // Apply denormal prevention (1e-18f bias)
        out += SampleType(1e-18); 

        if (hpC > SampleType(0))
        {
            s.fbHpZ1 = out * (SampleType(1) - hpC) + s.fbHpZ1 * hpC;
            out -= s.fbHpZ1;
        }
        if (lpC > SampleType(0))
        {
            s.fbLpZ1 = out * (SampleType(1) - lpC) + s.fbLpZ1 * lpC;
            out = s.fbLpZ1;
        }
        
        out -= SampleType(1e-18);
        return out;
    }

    SampleType calcLpCoef(SampleType freq) const noexcept { return std::exp(-twoPi<SampleType> * freq / sampleRate_); }
    SampleType calcHpCoef(SampleType freq) const noexcept { return std::exp(-twoPi<SampleType> * freq / sampleRate_); }

    static int nextPow2(int v) noexcept
    {
        int r = 1;
        while (r < v) r <<= 1;
        return r;
    }

    // -- Members -------------------------------------------------------------
    std::atomic<bool> prepared_{ false };
    AudioBuffer<SampleType> delayBuffer_, wetBuffer_;
    std::array<ChannelState, kMaxChannels> states_ {};
    std::array<int, kMaxChannels> writeIndices_ {}; 

    int maxDelaySamples_ = 0, bufferMask_ = 0;
    int numChannels_ = 0, blockSize_ = 0;
    SampleType sampleRate_ = SampleType(48000);
    
    std::atomic<SampleType> globalDelay_ { SampleType(0) };
    std::atomic<SampleType> feedbackGain_ { SampleType(0) };
    std::atomic<SampleType> fbLpCoef_ { SampleType(0) };
    std::atomic<SampleType> fbHpCoef_ { SampleType(0) };

    std::atomic<SmootherType> smootherType_ { SmootherType::Exponential };
    std::atomic<float> smoothingTimeMs_ { 20.0f };
    std::atomic<bool> smootherDirty_ { false };

    Smoothers::LinearSmoother mixSmoother_;
};

} // namespace dspark
