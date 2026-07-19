// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

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
 * Dependencies: Core/AudioBuffer.h, Core/AudioSpec.h, Core/DspMath.h,
 * Core/Smoothers.h, Core/StateBlob.h.
 *
 * Threading: prepare() belongs to the setup thread. The processing calls and
 * reset() belong to the audio thread. Parameter setters are safe from any
 * thread (atomics with a publish/apply handoff consumed at the top of the
 * next processing call); non-finite setter values are ignored.
 *
 * @tparam SampleType float or double.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Smoothers.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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

    /** @brief Feedback path colour. */
    enum class FeedbackMode
    {
        Clean,  ///< Pristine digital regeneration: no per-pass saturation, the
                ///< decay time matches the feedback gain exactly. A hard
                ///< safety clamp at +-2.0 guards against runaway (|fb| >= 1).
        Analog  ///< Tape/BBD-style tanh soft saturation on every pass (default,
                ///< matches the original DSPark behaviour). Slightly shortens
                ///< the decay versus the nominal feedback and adds warmth.
    };

    /** @brief Selects clean or analog feedback regeneration. Thread-safe. */
    void setFeedbackMode(FeedbackMode mode) noexcept
    {
        feedbackMode_.store(mode, std::memory_order_relaxed);
    }

    /** @brief Returns the active feedback mode. */
    [[nodiscard]] FeedbackMode getFeedbackMode() const noexcept
    {
        return feedbackMode_.load(std::memory_order_relaxed);
    }

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Prepares the delay structures and allocates memory. Must be called before processing.
     * @param spec The audio specification (sample rate, block size, etc.).
     *             An invalid spec (non-positive or NaN fields) is ignored.
     * @param maxDelaySeconds The maximum theoretical delay time required.
     *                        Negative or NaN values are ignored; the capacity
     *                        is capped at 2^29 samples.
     */
    void prepare(const AudioSpec& spec, double maxDelaySeconds)
    {
        if (!spec.isValid() || !(maxDelaySeconds >= 0.0)) return;
        sampleRate_ = static_cast<SampleType>(spec.sampleRate);
        // Clamp to the fixed per-channel state arrays (states_/writeIndices_ are
        // kMaxChannels): maybeUpdateSmoothers()/updateSmootherTargets() iterate
        // [0, numChannels_) and would otherwise read/write out of bounds.
        numChannels_ = std::min(static_cast<int>(spec.numChannels), kMaxChannels);
        blockSize_   = spec.maxBlockSize;

        // Cap in double BEFORE the int cast: a huge request would otherwise be
        // undefined behaviour in the cast, and nextPow2 past 2^29 overflows
        // its shift. 2^29 samples is over 3 hours at 48 kHz.
        const double capacityD =
            std::min(std::ceil(maxDelaySeconds * spec.sampleRate) + blockSize_,
                     536870912.0); // 2^29
        maxDelaySamples_ = nextPow2(static_cast<int>(capacityD));
        bufferMask_ = maxDelaySamples_ - 1;

        delayBuffer_.resize(numChannels_, maxDelaySamples_);
        wetBuffer_.resize(numChannels_, blockSize_);
        lastWetSamples_ = blockSize_; // full capacity until the first push

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
        smootherDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the smoothing transition time.
     * @param ms Time in milliseconds. Non-finite values are ignored.
     */
    void setSmoothingTime(float ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        smoothingTimeMs_.store(std::max(0.0f, ms), std::memory_order_relaxed);
        smootherDirty_.store(true, std::memory_order_release);
    }

    // -- Parameters ----------------------------------------------------------

    /**
     * @brief Sets the delay time in samples.
     * @param samples Target delay time (can be fractional). Non-finite values
     *                are ignored (a NaN here would poison the read position
     *                and, through the feedback path, the whole buffer).
     */
    void setDelaySamples(SampleType samples) noexcept
    {
        if (!std::isfinite(samples)) return;
        samples = std::clamp(samples, SampleType(0),
                             static_cast<SampleType>(std::max(0, maxDelaySamples_ - 1)));
        globalDelay_.store(samples, std::memory_order_relaxed);
        // Publish/apply pattern: the smoothers are NOT touched here (they are
        // non-atomic audio-thread state - mutating them from a control thread
        // raced against advanceSmoother). The audio thread applies the new
        // target in maybeUpdateSmoothers() at the top of its next process call.
        delayTargetDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets the delay time in milliseconds. */
    void setDelayMs(SampleType ms) noexcept { setDelaySamples(ms * sampleRate_ / SampleType(1000)); }

    /** @brief Sets the delay time in seconds. */
    void setDelaySeconds(SampleType secs) noexcept { setDelaySamples(secs * sampleRate_); }

    /** @brief Returns the current target delay time in samples. */
    [[nodiscard]] SampleType getCurrentDelaySamples() const noexcept { return globalDelay_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets the global feedback amount.
     * @param gain Feedback gain multiplier [-1.0, 1.0]. Will be saturated
     *             internally if exceeded. Non-finite values are ignored (a NaN
     *             would recirculate through the delay buffer forever).
     */
    void setFeedback(SampleType gain) noexcept
    {
        if (!std::isfinite(gain)) return;
        feedbackGain_.store(gain, std::memory_order_relaxed);
    }

    /** @brief Sets the cutoff frequency for the feedback low-pass filter
     *  (0 to disable). Non-finite values are ignored. */
    void setFeedbackLpHz(SampleType freq) noexcept
    {
        if (!std::isfinite(freq)) return;
        fbLpCoef_.store((freq > 0) ? calcLpCoef(freq) : SampleType(0), std::memory_order_relaxed);
        fbLpHzShadow_.store(freq, std::memory_order_relaxed);   // serialization readback
    }

    /** @brief Sets the cutoff frequency for the feedback high-pass filter
     *  (0 to disable). Non-finite values are ignored. */
    void setFeedbackHpHz(SampleType freq) noexcept
    {
        if (!std::isfinite(freq)) return;
        fbHpCoef_.store((freq > 0) ? calcHpCoef(freq) : SampleType(0), std::memory_order_relaxed);
        fbHpHzShadow_.store(freq, std::memory_order_relaxed);   // serialization readback
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
        advanceWriteIndexUnchecked(ch);
        return out;
    }

    // -- Block processing ----------------------------------------------------

    /**
     * @brief Processes a multi-channel block in-place.
     *
     * @note The per-call parameters are published to the same state the
     *       individual setters write, so every call overwrites previously set
     *       values (including the defaults of the omitted arguments).
     *
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
                advanceWriteIndexUnchecked(ch);
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
            advanceWriteIndexUnchecked(ch);
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

    /** @brief Processes the wet buffer in place with current settings.
     *  Processes exactly the samples of the most recent pushDryToWet() call:
     *  running the full capacity regardless of the pushed length would advance
     *  the delay line past the real stream on short blocks, skewing every
     *  subsequent echo's timing. */
    void processWet(SampleType delayMs, SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        processBlock(wetBuffer_.toView().getSubView(0, lastWetSamples_), delayMs, feedback, lpHz, hpHz);
    }

    /**
     * @brief Processes a true ping-pong delay (L feeds R, R feeds L) on the
     *        samples of the most recent pushDryToWet() call.
     *
     * The cross-feed obeys the active FeedbackMode exactly like the straight
     * feedback path: tanh soft saturation in Analog mode, a +-2.0 safety
     * clamp in Clean mode (without it, |feedback| >= 1 grew without bound).
     */
    void processPingPong(SampleType delayMs, SampleType feedback = 0, SampleType lpHz = 0, SampleType hpHz = 0) noexcept
    {
        if (wetBuffer_.getNumChannels() < 2 || !prepared_.load(std::memory_order_acquire)) return;

        setDelayMs(delayMs);
        setFeedback(feedback);
        setFeedbackLpHz(lpHz);
        setFeedbackHpHz(hpHz);
        maybeUpdateSmoothers();

        SampleType* L = wetBuffer_.getChannel(0);
        SampleType* R = wetBuffer_.getChannel(1);
        const int nS = std::min(lastWetSamples_, wetBuffer_.getNumSamples());

        const auto smoothType = smootherType_.load(std::memory_order_relaxed);
        const SampleType targetDelay = globalDelay_.load(std::memory_order_relaxed);
        const SampleType fb  = feedbackGain_.load(std::memory_order_relaxed);
        const SampleType lpC = fbLpCoef_.load(std::memory_order_relaxed);
        const SampleType hpC = fbHpCoef_.load(std::memory_order_relaxed);
        const bool analogFb  = feedbackMode_.load(std::memory_order_relaxed) == FeedbackMode::Analog;

        for (int i = 0; i < nS; ++i)
        {
            auto& sL = states_[0];
            auto& sR = states_[1];

            SampleType currentDelay = (smoothType == SmootherType::None) ? targetDelay : advanceSmoother(sL, smoothType);
            advanceSmoother(sR, smoothType); // Keep smoothers in sync

            // s.pingPongFb holds what THIS channel receives next sample, i.e.
            // the OTHER channel's previous output. (The original wiring
            // consumed each channel's own stored value while storing f(other)
            // into it: every echo regenerated into its own channel and the
            // signal never crossed the stereo field at all.)
            SampleType inL = L[i] + sL.pingPongFb;
            SampleType inR = R[i] + sR.pingPongFb;

            SampleType outL = processSampleInternal(0, inL, currentDelay, sL, SampleType(0), lpC, hpC);
            SampleType outR = processSampleInternal(1, inR, currentDelay, sR, SampleType(0), lpC, hpC);

            sR.pingPongFb = saturateFeedback(processFbFilters(1, outL * fb, lpC, hpC), analogFb); // L crosses to R
            sL.pingPongFb = saturateFeedback(processFbFilters(0, outR * fb, lpC, hpC), analogFb); // R crosses to L

            L[i] = outL;
            R[i] = outR;
            advanceWriteIndexUnchecked(0);
            advanceWriteIndexUnchecked(1);
        }
    }

    /**
     * @brief Mixes the processed wet buffer back into the provided dry buffer.
     * @param dry Buffer containing the dry signal.
     * @param mix Mix ratio (0.0 = 100% dry, 1.0 = 100% wet). A non-finite
     *            value resolves to 0 (dry).
     */
    void mixWetToDry(AudioBufferView<SampleType> dry, SampleType mix) noexcept
    {
        // min/max with this argument order also resolves NaN to the dry bound.
        mix = std::min(SampleType(1), std::max(SampleType(0), mix));
        auto st = smootherType_.load(std::memory_order_relaxed);
        if (st != SmootherType::None)
            mixSmoother_.setTargetValue(static_cast<float>(mix));

        const int nS = std::min(dry.getNumSamples(), wetBuffer_.getNumSamples());
        const int nCh = std::min(dry.getNumChannels(), wetBuffer_.getNumChannels());

        // Sample-outer loop: ONE smoother value per sample, shared by every
        // channel. Advancing the smoother inside a channel-outer loop ran the
        // ramp numChannels times too fast and, worse, gave each channel a
        // different segment of it (channel 1 got the ramp a whole block ahead
        // of channel 0), audibly tilting the stereo image on every mix change.
        std::array<SampleType*, kMaxChannels> d {};
        std::array<const SampleType*, kMaxChannels> w {};
        for (int ch = 0; ch < nCh; ++ch)
        {
            d[ch] = dry.getChannel(ch);
            w[ch] = wetBuffer_.getChannel(ch);
        }
        for (int i = 0; i < nS; ++i)
        {
            const SampleType m = (st != SmootherType::None)
                ? static_cast<SampleType>(mixSmoother_.getNextValue()) : mix;
            for (int ch = 0; ch < nCh; ++ch)
                d[ch][i] = d[ch][i] * (SampleType(1) - m) + w[ch][i] * m;
        }
    }

    /** @brief Exposes the internal wet buffer view. */
    AudioBufferView<SampleType> getWetView() noexcept { return wetBuffer_.toView(); }
    
    /** @brief Returns maximum capacity in samples. */
    [[nodiscard]] int getMaxDelaySamples() const noexcept { return maxDelaySamples_; }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DLAY"), 1);
        w.write("delaySamples", globalDelay_.load(std::memory_order_relaxed));
        w.write("feedback", feedbackGain_.load(std::memory_order_relaxed));
        w.write("fbLpHz", fbLpHzShadow_.load(std::memory_order_relaxed));
        w.write("fbHpHz", fbHpHzShadow_.load(std::memory_order_relaxed));
        w.write("smoother", static_cast<int32_t>(smootherType_.load(std::memory_order_relaxed)));
        w.write("smoothingMs", smoothingTimeMs_.load(std::memory_order_relaxed));
        w.write("fbMode", static_cast<int32_t>(feedbackMode_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DLAY")) return false;
        setDelaySamples(static_cast<SampleType>(r.read("delaySamples", 0.0f)));
        setFeedback(static_cast<SampleType>(r.read("feedback", 0.0f)));
        setFeedbackLpHz(static_cast<SampleType>(r.read("fbLpHz", 0.0f)));
        setFeedbackHpHz(static_cast<SampleType>(r.read("fbHpHz", 0.0f)));
        // Enum fields clamped so a corrupt/foreign blob cannot install ids
        // outside the ranges the switches expect.
        setSmoother(static_cast<SmootherType>(std::clamp(r.read("smoother", 0), 0,
                    static_cast<int>(SmootherType::CriticallyDamped))));
        setSmoothingTime(r.read("smoothingMs", 20.0f));
        setFeedbackMode(static_cast<FeedbackMode>(std::clamp(r.read("fbMode", 1), 0,
                        static_cast<int>(FeedbackMode::Analog))));
        return true;
    }

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
     * @param ch Channel index to advance. Out-of-range indices are ignored
     *           (an unchecked index would write past the index array).
     */
    void advanceWriteIndex(int ch) noexcept
    {
        if (ch < 0 || ch >= kMaxChannels) return;
        advanceWriteIndexUnchecked(ch);
    }

private:
    /** Internal hot-path variant: every internal caller has already bounded ch. */
    void advanceWriteIndexUnchecked(int ch) noexcept { writeIndices_[ch] = (writeIndices_[ch] + 1) & bufferMask_; }

    /** The FeedbackMode nonlinearity, shared by the straight and ping-pong
     *  feedback paths: tanh warmth in Analog, a hard +-2.0 safety bound in
     *  Clean (exact regeneration below it). */
    [[nodiscard]] SampleType saturateFeedback(SampleType x, bool analog) const noexcept
    {
        return analog ? std::tanh(x) : std::clamp(x, SampleType(-2), SampleType(2));
    }

    template <typename ChannelFn>
    void pushDryToWetImpl(ChannelFn&& getCh, int dryCh, int drySamples) noexcept
    {
        const int nCh = std::min(dryCh, wetBuffer_.getNumChannels());
        const int nS  = std::min(drySamples, wetBuffer_.getNumSamples());
        for (int ch = 0; ch < nCh; ++ch)
            std::memcpy(wetBuffer_.getChannel(ch), getCh(ch), static_cast<std::size_t>(nS) * sizeof(SampleType));
        lastWetSamples_ = nS; // processWet()/processPingPong() run exactly this many
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
        // Audio-thread application point for control-thread publications.
        if (delayTargetDirty_.exchange(false, std::memory_order_acquire))
            updateSmootherTargets(globalDelay_.load(std::memory_order_relaxed));

        // exchange (not load+store: a publication between the two would be
        // lost) with acquire, pairing with the setters' release stores so the
        // freshly written type/time values are visible here.
        if (!smootherDirty_.exchange(false, std::memory_order_acquire)) return;
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
        // panner crosses the centre). 3 samples ~ 62 us at 48 kHz -
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

        // Feedback processing
        SampleType fbInput = delayed * fbMult;
        if (fbInput != SampleType(0))
        {
            fbInput = processFbFilters(ch, fbInput, lpC, hpC);
            fbInput = saturateFeedback(fbInput,
                feedbackMode_.load(std::memory_order_relaxed) == FeedbackMode::Analog);
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

    /** @brief One-pole coefficient shared by the feedback LP and HP filters. */
    SampleType calcOnePoleCoef(SampleType freq) const noexcept { return std::exp(-twoPi<SampleType> * freq / sampleRate_); }
    SampleType calcLpCoef(SampleType freq) const noexcept { return calcOnePoleCoef(freq); }
    SampleType calcHpCoef(SampleType freq) const noexcept { return calcOnePoleCoef(freq); }

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
    int lastWetSamples_ = 0;   ///< Length of the most recent pushDryToWet().
    SampleType sampleRate_ = SampleType(48000);
    
    std::atomic<SampleType> globalDelay_ { SampleType(0) };
    std::atomic<SampleType> feedbackGain_ { SampleType(0) };
    std::atomic<SampleType> fbLpCoef_ { SampleType(0) };
    std::atomic<SampleType> fbLpHzShadow_ { SampleType(0) };  ///< Hz mirror for getState.
    std::atomic<SampleType> fbHpHzShadow_ { SampleType(0) };  ///< Hz mirror for getState.
    std::atomic<SampleType> fbHpCoef_ { SampleType(0) };

    std::atomic<SmootherType> smootherType_ { SmootherType::Exponential };
    std::atomic<float> smoothingTimeMs_ { 20.0f };
    std::atomic<bool> smootherDirty_ { false };
    std::atomic<bool> delayTargetDirty_ { false };
    std::atomic<FeedbackMode> feedbackMode_ { FeedbackMode::Analog };

    Smoothers::LinearSmoother mixSmoother_;
};

} // namespace dspark
