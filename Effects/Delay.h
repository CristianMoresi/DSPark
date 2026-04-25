// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Delay.h
 * @brief Professional delay line with smoothing, feedback, and stereo processing.
 *
 * Ported from the existing JUCE-dependent Delay class. Features:
 * - Power-of-two circular buffer with bitmask wrapping
 * - Linear interpolation for fractional delay
 * - One-pole feedback filters (LP + HP) for vintage/dub character
 * - Multiple smoother choices for artifact-free parameter changes
 * - In-place and wet-buffer processing modes
 * - Ping-pong stereo delay
 * - Fully standalone (C++20 STL only)
 *
 * @tparam SampleType float or double.
 *
 * @code
 *   dspark::Delay<float> delay;
 *   delay.prepare(spec, 2.0);  // 2 seconds max
 *   delay.setDelayMs(250.0f);
 *   delay.setFeedback(0.4f);
 *
 *   // In process():
 *   delay.processBlock(buffer);
 * @endcode
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
class Delay
{
public:
    virtual ~Delay() = default;
    enum class SmootherType
    {
        None, Linear, Exponential, OnePole, MultiPole2,
        Asymmetric, SlewLimiter, StateVariable, Butterworth, CriticallyDamped
    };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the delay with a maximum delay time in seconds.
     * @param spec Audio spec.
     * @param maxDelaySeconds Maximum delay capacity.
     */
    void prepare(const AudioSpec& spec, double maxDelaySeconds)
    {
        sampleRate_ = static_cast<SampleType>(spec.sampleRate);
        numChannels_ = spec.numChannels;
        blockSize_   = spec.maxBlockSize;

        int required = static_cast<int>(std::ceil(maxDelaySeconds * spec.sampleRate));
        maxDelaySamples_ = nextPow2(required + blockSize_);
        bufferMask_ = maxDelaySamples_ - 1;

        // Allocate circular delay buffer
        delayBuffer_.resize(numChannels_, maxDelaySamples_);
        wetBuffer_.resize(numChannels_, blockSize_);

        float timeMs = smoothingTimeMs_;
        for (int ch = 0; ch < std::min(numChannels_, kMaxChannels); ++ch)
            resetChannelSmoothers(states_[ch], timeMs, 0.0f);

        mixSmoother_.reset(spec.sampleRate, timeMs, 1.0f);
        writeIndex_ = 0;
        reset();
    }

    /** @brief Prepare with max delay in milliseconds. */
    void prepareMs(const AudioSpec& spec, double maxDelayMs)
    {
        prepare(spec, maxDelayMs / 1000.0);
    }

    /** @brief Clears all delay buffers and resets state. */
    void reset() noexcept
    {
        delayBuffer_.clear();
        wetBuffer_.clear();
        writeIndex_ = 0;
        for (auto& s : states_) resetChannelState(s);
        mixSmoother_.skip();
    }

    // -- Smoother configuration ----------------------------------------------

    void setSmoother(SmootherType type) noexcept { smootherType_.store(type, std::memory_order_relaxed); smootherDirty_.store(true, std::memory_order_relaxed); }

    void setSmoothingTime(float ms) noexcept
    {
        smoothingTimeMs_ = std::max(0.0f, ms);
        smootherDirty_.store(true, std::memory_order_relaxed);
    }

    // -- Delay time ----------------------------------------------------------

    void setDelaySamples(SampleType samples) noexcept
    {
        samples = std::clamp(samples, SampleType(0), SampleType(maxDelaySamples_ - 1));
        globalDelay_.store(samples, std::memory_order_relaxed);
        updateSmootherTargets(samples);
    }

    void setDelayMs(SampleType ms) noexcept
    {
        setDelaySamples(ms * sampleRate_ / SampleType(1000));
    }

    void setDelaySeconds(SampleType secs) noexcept { setDelaySamples(secs * sampleRate_); }

    [[nodiscard]] SampleType getCurrentDelaySamples() const noexcept { return globalDelay_.load(std::memory_order_relaxed); }

    // -- Feedback ------------------------------------------------------------

    void setFeedback(SampleType gain) noexcept
    {
        feedbackGain_.store(std::clamp(gain, SampleType(-0.999), SampleType(0.999)), std::memory_order_relaxed);
    }

    void setFeedbackLpHz(SampleType freq) noexcept
    {
        fbLpCoef_.store((freq > 0) ? calcLpCoef(freq) : SampleType(0), std::memory_order_relaxed);
    }

    void setFeedbackHpHz(SampleType freq) noexcept
    {
        fbHpCoef_.store((freq > 0) ? calcHpCoef(freq) : SampleType(0), std::memory_order_relaxed);
    }

    // -- Sample processing ---------------------------------------------------

    SampleType processSample(int ch, SampleType input) noexcept
    {
        if (ch < 0 || ch >= numChannels_) return input;
        maybeUpdateSmoothers();
        auto& s = states_[ch];
        auto st = smootherType_.load(std::memory_order_relaxed);
        SampleType delay = (st == SmootherType::None)
            ? globalDelay_.load(std::memory_order_relaxed) : advanceSmoother(s);
        return processSampleInternal(ch, input, delay, s);
    }

    SampleType processSample(int ch, SampleType input, SampleType delaySamples) noexcept
    {
        if (ch < 0 || ch >= numChannels_) return input;
        maybeUpdateSmoothers();
        auto& s = states_[ch];
        auto st = smootherType_.load(std::memory_order_relaxed);
        if (st != SmootherType::None)
        {
            setSmootherTarget(s, delaySamples);
            delaySamples = advanceSmoother(s);
        }
        return processSampleInternal(ch, input, delaySamples, s);
    }

    // -- Block processing (in-place) -----------------------------------------

    void processBlock(AudioBufferView<SampleType> buffer, SampleType delayMs,
                      SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        setDelayMs(delayMs);
        setFeedback(feedback);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        const int nS = buffer.getNumSamples();
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);

        for (int i = 0; i < nS; ++i)
        {
            for (int ch = 0; ch < nCh; ++ch)
                buffer.getChannel(ch)[i] = processSample(ch, buffer.getChannel(ch)[i]);
            advanceWriteIndex();
        }
    }

    /**
     * @brief Processes a single channel through the delay.
     *
     * @warning This method advances the write index per sample. Do NOT call
     * this method sequentially for different channels of the same Delay instance —
     * use processBlock() for multi-channel processing, or use processSample() +
     * advanceWriteIndex() manually for interleaved multi-channel scenarios.
     */
    void processChannel(AudioBufferView<SampleType> buffer, int ch, SampleType delayMs,
                        SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        setDelayMs(delayMs);
        setFeedback(feedback);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        SampleType* data = buffer.getChannel(ch);
        const int nS = buffer.getNumSamples();
        for (int i = 0; i < nS; ++i)
        {
            data[i] = processSample(ch, data[i]);
            advanceWriteIndex();
        }
    }

    // -- Wet buffer processing -----------------------------------------------

    // M3: both overloads share the same memcpy implementation. The mutable
    // overload forwards to a shared helper rather than duplicating the loop.
    void pushDryToWet(AudioBufferView<const SampleType> dry) noexcept
    {
        pushDryToWetImpl(
            [&](int ch) { return dry.getChannel(ch); },
            dry.getNumChannels(), dry.getNumSamples());
    }

    void pushDryToWet(AudioBufferView<SampleType> dry) noexcept
    {
        pushDryToWetImpl(
            [&](int ch) { return dry.getChannel(ch); },
            dry.getNumChannels(), dry.getNumSamples());
    }

private:
    template <typename ChannelFn>
    void pushDryToWetImpl(ChannelFn&& getCh, int dryCh, int drySamples) noexcept
    {
        const int nCh = std::min(dryCh, wetBuffer_.getNumChannels());
        const int nS  = std::min(drySamples, wetBuffer_.getNumSamples());
        for (int ch = 0; ch < nCh; ++ch)
            std::memcpy(wetBuffer_.getChannel(ch), getCh(ch),
                        static_cast<std::size_t>(nS) * sizeof(SampleType));
    }

public:

    void processWet(SampleType delayMs, SampleType feedback = 0,
                    SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        processBlock(wetBuffer_.toView(), delayMs, feedback, lpHz, hpHz);
    }

    void processWetStereo(SampleType delayMsL, SampleType delayMsR,
                          SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        if (wetBuffer_.getNumChannels() < 2) return;
        setFeedback(feedback);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        SampleType* L = wetBuffer_.getChannel(0);
        SampleType* R = wetBuffer_.getChannel(1);
        SampleType delL = delayMsL * sampleRate_ / SampleType(1000);
        SampleType delR = delayMsR * sampleRate_ / SampleType(1000);
        const int nS = wetBuffer_.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            L[i] = processSample(0, L[i], delL);
            R[i] = processSample(1, R[i], delR);
            advanceWriteIndex();
        }
    }

    void processPingPong(SampleType delayMs, SampleType feedback = 0,
                         SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        if (wetBuffer_.getNumChannels() < 2) return;
        setFeedback(SampleType(0)); // manual cross-feedback
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        SampleType* L = wetBuffer_.getChannel(0);
        SampleType* R = wetBuffer_.getChannel(1);
        SampleType del = delayMs * sampleRate_ / SampleType(1000);
        const int nS = wetBuffer_.getNumSamples();

        for (int i = 0; i < nS; ++i)
        {
            auto& sL = states_[0];
            auto& sR = states_[1];

            SampleType inL = L[i] + sR.pingPongFb;
            SampleType inR = R[i] + sL.pingPongFb;

            SampleType outL = processSampleInternal(0, inL, del, sL);
            SampleType outR = processSampleInternal(1, inR, del, sR);

            sL.pingPongFb = processFbFilters(0, outR * feedback);
            sR.pingPongFb = processFbFilters(1, outL * feedback);

            L[i] = outL;
            R[i] = outR;
            advanceWriteIndex();
        }
    }

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

    AudioBufferView<SampleType> getWetView() noexcept { return wetBuffer_.toView(); }
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
    /** @brief Advances the shared write index by one sample.
     *  Call once per sample frame after processing all channels with processSample(). */
    void advanceWriteIndex() noexcept { writeIndex_ = (writeIndex_ + 1) & bufferMask_; }

private:
    // -- Internal helpers ----------------------------------------------------

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
        float timeMs = smoothingTimeMs_;
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
            case SmootherType::Exponential:      s.exp.setTargetValue(t); break;
            case SmootherType::OnePole:          s.onePole.setTargetValue(t); break;
            case SmootherType::MultiPole2:       s.multi2.setTargetValue(t); break;
            case SmootherType::Asymmetric:       s.asym.setTargetValue(t); break;
            case SmootherType::SlewLimiter:      s.slew.setTargetValue(t); break;
            case SmootherType::StateVariable:    s.svf.setTargetValue(t); break;
            case SmootherType::Butterworth:      s.butter.setTargetValue(t); break;
            case SmootherType::CriticallyDamped: s.crit.setTargetValue(t); break;
            default: break;
        }
    }

    SampleType advanceSmoother(ChannelState& s) noexcept
    {
        float val;
        switch (smootherType_.load(std::memory_order_relaxed))
        {
            case SmootherType::Linear:          val = s.lin.getNextValue(); break;
            case SmootherType::Exponential:      val = s.exp.getNextValue(); break;
            case SmootherType::OnePole:          val = s.onePole.getNextValue(); break;
            case SmootherType::MultiPole2:       val = s.multi2.getNextValue(); break;
            case SmootherType::Asymmetric:       val = s.asym.getNextValue(); break;
            case SmootherType::SlewLimiter:      val = s.slew.getNextValue(); break;
            case SmootherType::StateVariable:    val = s.svf.getNextValue(); break;
            case SmootherType::Butterworth:      val = s.butter.getNextValue(); break;
            case SmootherType::CriticallyDamped: val = s.crit.getNextValue(); break;
            default: val = static_cast<float>(s.targetDelay);
        }
        s.currentDelay = static_cast<SampleType>(val);
        return s.currentDelay;
    }

    SampleType processSampleInternal(int ch, SampleType input, SampleType delaySamples,
                                     ChannelState& s) noexcept
    {
        delaySamples = std::clamp(delaySamples, SampleType(0), SampleType(maxDelaySamples_ - 1));
        SampleType* data = delayBuffer_.getChannel(ch);

        SampleType readPos = static_cast<SampleType>(writeIndex_) - delaySamples;
        if (readPos < SampleType(0)) readPos += static_cast<SampleType>(maxDelaySamples_);

        // Linear interpolation
        int idx0 = static_cast<int>(readPos) & bufferMask_;
        int idx1 = (idx0 + 1) & bufferMask_;
        SampleType frac = readPos - static_cast<SampleType>(static_cast<int>(readPos));
        SampleType delayed = data[idx0] + frac * (data[idx1] - data[idx0]);

        SampleType fbInput = delayed * feedbackGain_.load(std::memory_order_relaxed);
        if (fbInput != SampleType(0))
            fbInput = processFbFilters(ch, fbInput);
        s.lastFeedback = fbInput;

        data[writeIndex_] = input + s.lastFeedback;
        return delayed;
    }

    SampleType processFbFilters(int ch, SampleType sample) noexcept
    {
        auto& s = states_[ch];
        SampleType out = sample;

        SampleType hpC = fbHpCoef_.load(std::memory_order_relaxed);
        SampleType lpC = fbLpCoef_.load(std::memory_order_relaxed);

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
        return out;
    }

    SampleType calcLpCoef(SampleType freq) const noexcept
    {
        return std::exp(-twoPi<SampleType> * freq / sampleRate_);
    }

    SampleType calcHpCoef(SampleType freq) const noexcept
    {
        return std::exp(-twoPi<SampleType> * freq / sampleRate_);
    }

    static int nextPow2(int v) noexcept
    {
        int r = 1;
        while (r < v) r <<= 1;
        return r;
    }

    // -- Members -------------------------------------------------------------
    AudioBuffer<SampleType> delayBuffer_, wetBuffer_;
    std::array<ChannelState, kMaxChannels> states_ {};

    int writeIndex_ = 0, maxDelaySamples_ = 0, bufferMask_ = 0;
    int numChannels_ = 0, blockSize_ = 0;
    SampleType sampleRate_ = SampleType(48000);
    std::atomic<SampleType> globalDelay_ { SampleType(0) };
    std::atomic<SampleType> feedbackGain_ { SampleType(0) };
    std::atomic<SampleType> fbLpCoef_ { SampleType(0) };
    std::atomic<SampleType> fbHpCoef_ { SampleType(0) };

    std::atomic<SmootherType> smootherType_ { SmootherType::Exponential };
    float smoothingTimeMs_ = 20.0f;
    std::atomic<bool> smootherDirty_ { false };

    Smoothers::LinearSmoother mixSmoother_;
};

} // namespace dspark
