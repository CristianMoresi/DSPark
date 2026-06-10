// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Compressor.h
 * @brief Modular dynamic range compressor with analog-modeled ballistics and Hilbert detection.
 *
 * Professional compressor with interchangeable detector, topology, and
 * ballistics character. Operates with linear-domain smoothing for authentic
 * analog hardware emulation (Opto, FET, Varimu).
 *
 * Features:
 * - Zero-allocation audio thread processing.
 * - Lock-free parameter updates via atomics and smoothers.
 * - Analytic signal envelope detection via Hilbert transform.
 * - ITU-R BS.1770-4 compliant True-Peak detection.
 * - Topologies: FeedForward and vintage FeedBack.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, SmoothedValue.h,
 *               DryWetMixer.h, RingBuffer.h, DenormalGuard.h, Hilbert.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/SmoothedValue.h"
#include "../Core/DryWetMixer.h"
#include "../Core/RingBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Hilbert.h"
#include "../Core/TruePeakDetector.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class Compressor
 * @brief High-fidelity modular compressor designed for real-time applications.
 *
 * This class avoids virtual dispatch entirely by utilizing branch-predicted
 * enum switches on pre-cached atomic states. Parameter changes are internally
 * smoothed to prevent audio artifacts. Processing is strictly allocation-free
 * after the initial prepare() call.
 *
 * @tparam T Floating-point sample type (float or double).
 */
template <FloatType T>
class Compressor
{
public:
    ~Compressor() = default;

    /** @brief Level detection methodologies. */
    enum class DetectorType
    {
        Peak,           ///< Instantaneous absolute value tracking. Fast and standard.
        Rms,            ///< Sliding-window Root-Mean-Square. Smoother, responds to average energy.
        TruePeak,       ///< 4x oversampled peak detection (ITU-R BS.1770-4 compliant).
        SplitPolarity,  ///< Asymmetric positive/negative half-wave tracking (ButterComp2 style).
        Hilbert         ///< Analytic signal magnitude. Instantaneous energy without RMS delay.
    };

    /** @brief Signal routing topology for the detector sidechain. */
    enum class Topology
    {
        FeedForward,  ///< Detector reads uncompressed input (modern, precise, transparent).
        FeedBack      ///< Detector reads compressed output (vintage, self-regulating, colored).
    };

    /** @brief Time-constant behavior and release curve shape. */
    enum class Character
    {
        Clean,    ///< Standard linear-domain RC smoothing. Transparent.
        Opto,     ///< Program-dependent release (slower at higher gain reductions). Emulates LA-2A.
        FET,      ///< Ultra-fast attack (<0.1ms) with dual-stage release tail. Emulates 1176.
        Varimu    ///< Program-dependent ratio increasing dynamically with signal level. Emulates Fairchild.
    };

    /** @brief Processing direction. */
    enum class Mode
    {
        Downward,  ///< Standard: Reduces dynamic range by attenuating signals above the threshold.
        Upward     ///< Upward: Reduces dynamic range by boosting signals below the threshold.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Allocates buffers and initializes internal DSP state.
     * @warning Must not be called from the audio thread. Allocates memory.
     * @param spec Audio environment specification (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        sampleRate_ = spec.sampleRate;

        T fs = static_cast<T>(sampleRate_);
        updateTimeConstants(fs);

        // Smoothed parameters initialization
        T thresh = threshold_.load(std::memory_order_relaxed);
        T rat    = ratio_.load(std::memory_order_relaxed);
        T knee   = kneeWidth_.load(std::memory_order_relaxed);
        
        thresholdSmooth_.prepare(sampleRate_, 30.0);
        thresholdSmooth_.reset(thresh);
        ratioSmooth_.prepare(sampleRate_, 30.0);
        ratioSmooth_.reset(std::max(rat, T(1)));
        kneeSmooth_.prepare(sampleRate_, 30.0);
        kneeSmooth_.reset(std::max(knee, T(0)));

        // Pre-allocate and initialize per-channel instances
        int maxLaSamples = static_cast<int>(sampleRate_ * 0.01) + 1; 
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            if (ch < spec.numChannels)
                lookaheadBuffers_[ch].prepare(maxLaSamples);
                
            hilbertDetectors_[ch].prepare(sampleRate_);
        }
        
        lookaheadSamples_ = static_cast<int>(fs * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));

        updateHpfCoefficients();

        // RMS Buffer pre-allocation (Max 500ms to ensure zero real-time allocation)
        int maxRmsSamples = static_cast<int>(sampleRate_ * 0.5) + 1;
        for (auto& buf : rmsBuffers_)
        {
            buf.assign(static_cast<size_t>(maxRmsSamples), T(0));
        }
        updateRmsWindow();

        reset();
    }

    /**
     * @brief Prepares the compressor using sample rate only (backward compatibility).
     * @param sampleRate The operating sample rate.
     */
    void prepare(double sampleRate) noexcept
    {
        AudioSpec spec { sampleRate, 512, 2 };
        prepare(spec);
    }

    /**
     * @brief Processes an audio buffer in-place using its own signal as the sidechain.
     * @param buffer Mutable reference to the audio block.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlockImpl(buffer, buffer);
    }

    /**
     * @brief Processes audio with an independent external sidechain.
     * 
     * The detector analyzes the `sidechain` buffer, but gain reduction is applied
     * to the `audio` buffer.
     * 
     * @param audio Audio buffer to be compressed (modified in-place).
     * @param sidechain Key/Sidechain signal (read-only).
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        processBlockImpl(audio, sidechain);
    }

    /**
     * @brief Processes a single sample on one channel.
     *
     * Bypasses block-level features like stereo linking, parallel mix, and lookahead.
     * Ideal for modular routing, synth voices, or per-sample environments.
     *
     * @param input Audio sample to process.
     * @param channel Index of the audio channel (used for state tracking).
     * @return Compressed output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        // Sync atomic parameters into the smoothers here too: processBlock does this
        // per block, but a processSample-only workflow would otherwise never see
        // setThreshold()/setRatio()/setKnee() changes (the smoothers stayed frozen).
        thresholdSmooth_.setTargetValue(threshold_.load(std::memory_order_relaxed));
        ratioSmooth_.setTargetValue(std::max(ratio_.load(std::memory_order_relaxed), T(1)));
        kneeSmooth_.setTargetValue(std::max(kneeWidth_.load(std::memory_order_relaxed), T(0)));

        T thresh = thresholdSmooth_.getNextValue();
        T ratio  = ratioSmooth_.getNextValue();
        T knee   = kneeSmooth_.getNextValue();
        
        auto detType   = detectorType_.load(std::memory_order_relaxed);
        auto topo      = topology_.load(std::memory_order_relaxed);
        auto charType  = character_.load(std::memory_order_relaxed);
        auto modeType  = mode_.load(std::memory_order_relaxed);
        bool scHpf     = scHpfEnabled_.load(std::memory_order_relaxed);
        bool autoMkup  = autoMakeup_.load(std::memory_order_relaxed);
        T relMs        = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));

        T sidechain = scHpf ? applySidechainHPF(input, channel) : input;
        
        T levelDb = (topo == Topology::FeedBack)
            ? detectLevel(fbLastOutput_[channel], channel, detType)
            : detectLevel(sidechain, channel, detType);

        T targetGR_Db = computeGain(levelDb, thresh, ratio, knee, charType, modeType);
        targetGR_Db = applyHoldAndRange(targetGR_Db, channel);

        T targetGainLinear = decibelsToGain(targetGR_Db);
        T smoothedGainLinear = applyBallisticsLinear(targetGainLinear, channel, charType, relMs);
        T smoothedGR_Db = gainToDecibels(smoothedGainLinear);

        autoMakeupEnv_ = smoothedGR_Db + autoMakeupCoeff_ * (autoMakeupEnv_ - smoothedGR_Db);

        T makeup = makeupGain_.load(std::memory_order_relaxed);
        if (autoMkup && modeType == Mode::Downward)
            makeup += -autoMakeupEnv_;
            
        T outputGain = smoothedGainLinear * decibelsToGain(makeup);
        T output = input * outputGain;
        
        fbLastOutput_[channel] = output;
        gainReductionDb_.store(smoothedGR_Db, std::memory_order_relaxed);
        return output;
    }

    /** 
     * @brief Resets all internal DSP history, states, and buffers to neutral.
     */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            envState_[ch] = T(1); // Linear gain neutral state is 1.0 (no compression)
            fbLastOutput_[ch] = T(0);
            scHpfState_[ch] = T(0);
            scHpfPrev_[ch] = T(0);
            channelLevelDb_[ch] = T(-200);
            lookaheadBuffers_[ch].reset();
            hilbertDetectors_[ch].reset();
        }
        for (auto& buf : rmsBuffers_)
            std::fill(buf.begin(), buf.end(), T(0));
        for (auto& sum : rmsSums_) sum = T(0);
        for (auto& idx : rmsIndices_) idx = 0;
        for (auto& cnt : rmsRecomputeCounters_) cnt = 0;

        truePeak_.reset();
        holdCounters_.fill(0);
        heldGrDb_.fill(T(0));
        gainReductionDb_.store(T(0), std::memory_order_relaxed);
        autoMakeupEnv_ = T(0);
        splitPosEnv_.fill(T(0));
        splitNegEnv_.fill(T(0));
        
        thresholdSmooth_.skip();
        ratioSmooth_.skip();
        kneeSmooth_.skip();
    }

    // =========================================================================
    // Parameter Setters
    // =========================================================================

    /** @brief Sets the compression threshold in dB. */
    void setThreshold(T dB) noexcept { threshold_.store(dB, std::memory_order_relaxed); }

    /** @brief Sets the compression ratio (1.0 = off, >20.0 = limiting). */
    void setRatio(T ratio) noexcept { ratio_.store(std::max(ratio, T(1)), std::memory_order_relaxed); }

    /** @brief Sets the attack time in milliseconds. */
    void setAttack(T ms) noexcept { attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed); }

    /** @brief Sets the release time in milliseconds. */
    void setRelease(T ms) noexcept { releaseMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed); }

    /** @brief Sets knee width in dB (0 = hard knee, >0 = soft knee). */
    void setKnee(T dB) noexcept { kneeWidth_.store(std::max(dB, T(0)), std::memory_order_relaxed); }

    /** @brief Sets manual static makeup gain in dB. */
    void setMakeupGain(T dB) noexcept { makeupGain_.store(dB, std::memory_order_relaxed); }

    /** @brief Toggles automatic volume compensation tracking. */
    void setAutoMakeup(bool on) noexcept { autoMakeup_.store(on, std::memory_order_relaxed); }

    /** @brief Sets processing mode (Downward or Upward compression). */
    void setMode(Mode mode) noexcept { mode_.store(mode, std::memory_order_relaxed); }

    /** @brief Sets stereo linking amount (0.0 = unlinked dual mono, 1.0 = fully linked). */
    void setStereoLink(T amount) noexcept { stereoLink_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed); }

    /** @brief Sets dry/wet balance for parallel (New York) compression (1.0 = fully wet). */
    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    /** 
     * @brief Sets lookahead time in ms (0 = off, max = 10ms). 
     * @warning Automatically disabled internally if topology is set to FeedBack.
     */
    void setLookahead(T ms) noexcept { lookaheadMs_.store(std::clamp(ms, T(0), T(10)), std::memory_order_relaxed); }

    /** @brief Changes the level detection algorithm (Peak, RMS, TruePeak, Hilbert). */
    void setDetector(DetectorType type) noexcept
    {
        // The shared TruePeakDetector builds its coefficients lazily on first
        // use (thread-safe static), so this is a pure atomic publication.
        detectorType_.store(type, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the gain-reduction hold time.
     *
     * The deepest gain reduction is held for this long before the release
     * stage may recover — the classic tool against pumping on dense material.
     *
     * @param ms Hold time in milliseconds (0 = off, mastering range 0-500).
     */
    void setHoldTime(T ms) noexcept
    {
        holdMs_.store(std::clamp(ms, T(0), T(500)), std::memory_order_relaxed);
    }

    /**
     * @brief Limits the maximum gain change the compressor may apply.
     *
     * Classic "range" control: gain reduction (or upward boost) never exceeds
     * this many dB regardless of how far the signal passes the threshold.
     *
     * @param dB Maximum |gain change| in dB (default 100 = unlimited in practice).
     */
    void setRange(T dB) noexcept
    {
        rangeDb_.store(std::max(dB, T(0)), std::memory_order_relaxed);
    }

    /** @brief Changes signal routing topology (FeedForward or FeedBack). */
    void setTopology(Topology topo) noexcept { topology_.store(topo, std::memory_order_relaxed); }

    /** @brief Changes ballistics and envelope behavior character. */
    void setCharacter(Character type) noexcept { character_.store(type, std::memory_order_relaxed); }

    /**
     * @brief Toggles the internal sidechain high-pass filter.
     * @param enabled True to activate HPF.
     * @param cutoffHz Cutoff frequency in Hz (Default: 80Hz).
     */
    void setSidechainHPF(bool enabled, T cutoffHz = T(80)) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        scHpfFreq_.store(cutoffHz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the RMS analysis window size in milliseconds.
     *
     * Published atomically; the audio thread applies the new window (and
     * resets the sliding state) at its next block, so no control-thread write
     * ever races the per-sample RMS accumulators.
     *
     * @warning Capable up to 500ms maximum based on prepare() pre-allocation.
     */
    void setRmsWindow(T ms) noexcept
    {
        rmsWindowMsAtomic_.store(std::max(ms, T(1)), std::memory_order_relaxed);
        rmsWindowDirty_.store(true, std::memory_order_release);
    }

    // =========================================================================
    // Metering & Getters
    // =========================================================================

    /** @brief Returns current active gain reduction in dB (negative value). */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReductionDb_.load(std::memory_order_relaxed); }

    /** @brief Returns the currently active detector type. */
    [[nodiscard]] DetectorType getDetector() const noexcept { return detectorType_.load(std::memory_order_relaxed); }

    /** @brief Returns the currently active topology. */
    [[nodiscard]] Topology getTopology() const noexcept { return topology_.load(std::memory_order_relaxed); }

    /** @brief Returns the currently active character. */
    [[nodiscard]] Character getCharacter() const noexcept { return character_.load(std::memory_order_relaxed); }

    /** @brief Returns total processing latency in samples (caused by lookahead). */
    [[nodiscard]] int getLatency() const noexcept { return lookaheadSamples_; }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("COMP"), 1);
        w.write("threshold", threshold_.load(std::memory_order_relaxed));
        w.write("ratio", ratio_.load(std::memory_order_relaxed));
        w.write("attack", attackMs_.load(std::memory_order_relaxed));
        w.write("release", releaseMs_.load(std::memory_order_relaxed));
        w.write("knee", kneeWidth_.load(std::memory_order_relaxed));
        w.write("makeup", makeupGain_.load(std::memory_order_relaxed));
        w.write("autoMakeup", autoMakeup_.load(std::memory_order_relaxed));
        w.write("stereoLink", stereoLink_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("lookahead", lookaheadMs_.load(std::memory_order_relaxed));
        w.write("hold", holdMs_.load(std::memory_order_relaxed));
        w.write("range", rangeDb_.load(std::memory_order_relaxed));
        w.write("detector", static_cast<int32_t>(detectorType_.load(std::memory_order_relaxed)));
        w.write("topology", static_cast<int32_t>(topology_.load(std::memory_order_relaxed)));
        w.write("character", static_cast<int32_t>(character_.load(std::memory_order_relaxed)));
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("scHpf", scHpfEnabled_.load(std::memory_order_relaxed));
        w.write("scHpfFreq", scHpfFreq_.load(std::memory_order_relaxed));
        w.write("rmsWindow", rmsWindowMsAtomic_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("COMP")) return false;
        setThreshold(static_cast<T>(r.read("threshold", -20.0f)));
        setRatio(static_cast<T>(r.read("ratio", 4.0f)));
        setAttack(static_cast<T>(r.read("attack", 5.0f)));
        setRelease(static_cast<T>(r.read("release", 100.0f)));
        setKnee(static_cast<T>(r.read("knee", 0.0f)));
        setMakeupGain(static_cast<T>(r.read("makeup", 0.0f)));
        setAutoMakeup(r.read("autoMakeup", true));
        setStereoLink(static_cast<T>(r.read("stereoLink", 1.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setLookahead(static_cast<T>(r.read("lookahead", 0.0f)));
        setHoldTime(static_cast<T>(r.read("hold", 0.0f)));
        setRange(static_cast<T>(r.read("range", 100.0f)));
        setDetector(static_cast<DetectorType>(r.read("detector", 0)));
        setTopology(static_cast<Topology>(r.read("topology", 0)));
        setCharacter(static_cast<Character>(r.read("character", 0)));
        setMode(static_cast<Mode>(r.read("mode", 0)));
        setSidechainHPF(r.read("scHpf", false), static_cast<T>(r.read("scHpfFreq", 80.0f)));
        setRmsWindow(static_cast<T>(r.read("rmsWindow", 10.0f)));
        return true;
    }

protected:
    static constexpr int kMaxChannels = 16;

    /** 
     * @brief Core DSP loop executing sidechain detection, linking, ballistics, and gain application.
     * @param audio Buffer to modify.
     * @param sidechain Buffer to read for level detection.
     */
    void processBlockImpl(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        DenormalGuard guard;
        const int nCh  = std::min(audio.getNumChannels(), kMaxChannels);
        const int scCh = sidechain.getNumChannels();
        const int nS   = audio.getNumSamples();

        // Sync atomic parameters to block-local smoothed state
        thresholdSmooth_.setTargetValue(threshold_.load(std::memory_order_relaxed));
        ratioSmooth_.setTargetValue(std::max(ratio_.load(std::memory_order_relaxed), T(1)));
        kneeSmooth_.setTargetValue(std::max(kneeWidth_.load(std::memory_order_relaxed), T(0)));

        T fs = static_cast<T>(sampleRate_);
        if (fs > T(0)) updateTimeConstants(fs);
        updateHpfCoefficients();

        // Apply a pending RMS window change here, on the audio thread.
        if (rmsWindowDirty_.exchange(false, std::memory_order_acquire))
        {
            rmsWindowMs_ = rmsWindowMsAtomic_.load(std::memory_order_relaxed);
            updateRmsWindow();
        }

        // Cache enum/bool params locally to prevent atomic stalls inside the tight DSP loop
        auto detType   = detectorType_.load(std::memory_order_relaxed);
        auto topo      = topology_.load(std::memory_order_relaxed);
        auto charType  = character_.load(std::memory_order_relaxed);
        auto modeType  = mode_.load(std::memory_order_relaxed);
        bool scHpf     = scHpfEnabled_.load(std::memory_order_relaxed);
        bool autoMkup  = autoMakeup_.load(std::memory_order_relaxed);
        T mkupGain     = makeupGain_.load(std::memory_order_relaxed);
        T sLink        = stereoLink_.load(std::memory_order_relaxed);
        T mixVal       = mix_.load(std::memory_order_relaxed);
        T relMs        = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));

        // Lookahead breaks causal logic in Feedback mode, so it is strictly disabled
        int activeLookahead = (topo == Topology::FeedBack) ? 0 : lookaheadSamples_;

        for (int i = 0; i < nS; ++i)
        {
            T thresh = thresholdSmooth_.getNextValue();
            T ratio  = ratioSmooth_.getNextValue();
            T knee   = kneeSmooth_.getNextValue();
            T linkedLevel = T(-200);

            // 1. Detection Path (Per-Channel)
            for (int ch = 0; ch < nCh; ++ch)
            {
                // Fallback to internal channel if sidechain buffer lacks channels
                T sample = (scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))[i] 
                                      : audio.getChannel(ch)[i];

                if (scHpf) sample = applySidechainHPF(sample, ch);

                T levelDb = (topo == Topology::FeedBack) 
                    ? detectLevel(fbLastOutput_[ch], ch, detType)
                    : detectLevel(sample, ch, detType);

                channelLevelDb_[ch] = levelDb;
                if (levelDb > linkedLevel) linkedLevel = levelDb;
            }

            // 2. Stereo Linking & Gain Application
            T blockGR = T(0);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T chLevel = channelLevelDb_[ch];
                T inputDb = chLevel + sLink * (linkedLevel - chLevel);

                T targetGR_Db = computeGain(inputDb, thresh, ratio, knee, charType, modeType);
                targetGR_Db = applyHoldAndRange(targetGR_Db, ch);

                // 3. Analog-Modeled Ballistics (Linear Gain Domain)
                T targetGainLinear = decibelsToGain(targetGR_Db);
                T smoothedGainLinear = applyBallisticsLinear(targetGainLinear, ch, charType, relMs);
                T smoothedGR_Db = gainToDecibels(smoothedGainLinear);

                // 4. Makeup & Mix
                T makeup = mkupGain;
                if (autoMkup && modeType == Mode::Downward)
                    makeup += -autoMakeupEnv_;

                T outputGain = smoothedGainLinear * decibelsToGain(makeup);

                T input;
                if (activeLookahead > 0)
                {
                    lookaheadBuffers_[ch].push(audio.getChannel(ch)[i]);
                    input = lookaheadBuffers_[ch].read(activeLookahead);
                }
                else
                {
                    input = audio.getChannel(ch)[i];
                }

                T wet = input * outputGain;
                fbLastOutput_[ch] = wet; // feedback detector reads the compressed signal

                // Parallel (New York) mix done inline: the dry reference is `input`,
                // which carries the SAME lookahead delay as the wet, so dry and wet
                // stay phase-aligned (the previous DryWetMixer captured the UNDELAYED
                // input and comb-filtered when lookahead + mix<1 were combined).
                audio.getChannel(ch)[i] = (mixVal < T(1))
                    ? (input * (T(1) - mixVal) + wet * mixVal)
                    : wet;

                if (ch == 0 || smoothedGR_Db < blockGR)
                    blockGR = smoothedGR_Db; // Track worst-case GR for metering
            }

            gainReductionDb_.store(blockGR, std::memory_order_relaxed);

            // Auto-makeup envelope tracks slowly (~300ms)
            autoMakeupEnv_ = blockGR + autoMakeupCoeff_ * (autoMakeupEnv_ - blockGR);
        }
    }

    /** 
     * @brief Pre-calculates recursive exponential decay coefficients.
     * @param fs Current sample rate.
     */
    void updateTimeConstants(T fs) noexcept
    {
        T attMs = std::max(attackMs_.load(std::memory_order_relaxed), T(0.01));
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));
        attackCoeff_  = std::exp(T(-1) / (fs * attMs / T(1000)));
        releaseCoeff_ = std::exp(T(-1) / (fs * relMs / T(1000)));
        autoMakeupCoeff_ = std::exp(T(-1) / (fs * T(0.3)));

        lookaheadSamples_ = static_cast<int>(fs * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));

        holdSamples_ = static_cast<int>(fs * holdMs_.load(std::memory_order_relaxed) / T(1000));
    }

    /**
     * @brief Applies the hold and range stages to the static gain target (dB).
     *
     * Hold keeps the deepest recent gain change for holdSamples_ before the
     * ballistics may relax; range bounds |gain change| to the configured dB.
     */
    [[nodiscard]] T applyHoldAndRange(T targetGR_Db, int ch) noexcept
    {
        const T range = rangeDb_.load(std::memory_order_relaxed);
        targetGR_Db = std::clamp(targetGR_Db, -range, range);

        if (holdSamples_ > 0)
        {
            T& held = heldGrDb_[ch];
            int& counter = holdCounters_[ch];
            if (std::abs(targetGR_Db) >= std::abs(held))
            {
                held = targetGR_Db;          // deeper action: re-arm the hold
                counter = holdSamples_;
            }
            else if (counter > 0)
            {
                --counter;                   // shallower: freeze at held depth
                targetGR_Db = held;
            }
            else
            {
                held = targetGR_Db;          // hold elapsed: track normally
            }
        }
        return targetGR_Db;
    }

    // ---- Detectors ----

    /**
     * @brief Computes level detection in Decibels.
     * @param sample Input sample.
     * @param ch Channel index (for state tracking).
     * @param detType Selected detector methodology.
     * @return Decibel representation of detected level.
     */
    [[nodiscard]] T detectLevel(T sample, int ch, DetectorType detType) noexcept
    {
        T level = std::abs(sample);
        switch (detType)
        {
            case DetectorType::Peak:
                break; // level already holds abs(sample)

            case DetectorType::Rms:
            {
                T sq = sample * sample;
                auto& buf = rmsBuffers_[ch];
                auto& sum = rmsSums_[ch];
                auto& idx = rmsIndices_[ch];
                auto& recomputeCount = rmsRecomputeCounters_[ch];
                int len = rmsWindowSamples_;

                if (len > 0 && len <= static_cast<int>(buf.capacity()))
                {
                    sum -= buf[idx];
                    buf[idx] = sq;
                    sum += sq;
                    if (++idx >= len) idx = 0; // branch beats an integer division per sample

                    // Periodic full re-summation to prevent floating-point drift
                    if (++recomputeCount >= kRmsRecomputePeriod)
                    {
                        sum = T(0);
                        for (int j = 0; j < len; ++j) sum += buf[j];
                        recomputeCount = 0;
                    }

                    level = std::sqrt(std::max(sum / static_cast<T>(len), T(0)));
                }
                break;
            }

            case DetectorType::TruePeak:
                level = truePeak_.processSample(sample, ch);
                break;

            case DetectorType::SplitPolarity:
            {
                T pos = std::max(sample, T(0));
                T neg = std::max(-sample, T(0));

                T posCoeff = (pos > splitPosEnv_[ch]) ? T(0.6) : T(0.99);
                T negCoeff = (neg > splitNegEnv_[ch]) ? T(0.6) : T(0.99);

                splitPosEnv_[ch] = pos + posCoeff * (splitPosEnv_[ch] - pos);
                splitNegEnv_[ch] = neg + negCoeff * (splitNegEnv_[ch] - neg);

                level = std::max(splitPosEnv_[ch], splitNegEnv_[ch]);
                break;
            }

            case DetectorType::Hilbert:
            {
                auto res = hilbertDetectors_[ch].process(sample);
                // Direct euclidean distance for magnitude calculation.
                // Assuming bounded audio signal [-1.0, 1.0], std::sqrt is safe and faster than std::hypot.
                level = std::sqrt(res.real * res.real + res.imag * res.imag);
                break;
            }
        }
        return gainToDecibels(level);
    }

    // ---- Gain curves ----

    /**
     * @brief Calculates static target gain reduction based on knee and ratio.
     * @param inputDb Detected level in decibels.
     * @param thresh Threshold parameter in dB.
     * @param ratio Ratio parameter.
     * @param knee Soft-knee width parameter in dB.
     * @param charType Modifies ratio dynamically if Varimu character is selected.
     * @param modeType Upward or Downward processing mode.
     * @return Target gain reduction in Decibels.
     */
    [[nodiscard]] T computeGain(T inputDb, T thresh, T ratio, T knee, Character charType, Mode modeType) const noexcept
    {
        if (modeType == Mode::Upward)
            return computeGainUpward(inputDb, thresh, ratio, knee);

        T effectiveRatio = ratio;
        if (charType == Character::Varimu && inputDb > thresh)
        {
            // Ratio increases smoothly depending on depth of compression
            T excess = inputDb - thresh;
            effectiveRatio = ratio * (T(1) + excess / T(40));
        }

        if (knee <= T(0))
        {
            // Hard knee
            if (inputDb <= thresh) return T(0);
            return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
        }
        else
        {
            // Soft knee interpolation
            T halfKnee = knee / T(2);
            T lower = thresh - halfKnee;
            T upper = thresh + halfKnee;

            if (inputDb <= lower) return T(0);
            if (inputDb >= upper) return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);

            T x = inputDb - lower;
            return (T(1) - T(1) / effectiveRatio) * x * x / (T(2) * knee) * T(-1);
        }
    }

    /**
     * @brief Upward compression curve calculation.
     */
    [[nodiscard]] T computeGainUpward(T inputDb, T thresh, T ratio, T knee) const noexcept
    {
        T effectiveRatio = ratio;
        if (knee <= T(0))
        {
            if (inputDb >= thresh) return T(0);
            return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
        }
        else
        {
            T halfKnee = knee / T(2);
            T lower = thresh - halfKnee;
            T upper = thresh + halfKnee;

            if (inputDb >= upper) return T(0);
            if (inputDb <= lower) return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);

            T x = upper - inputDb;
            return (T(1) - T(1) / effectiveRatio) * x * x / (T(2) * knee);
        }
    }

    // ---- Linear Domain Ballistics ----

    /**
     * @brief Applies hardware-style Attack and Release times in the Linear Amplitude domain.
     * @param targetGain The instantaneous calculated gain reduction (linear, 0.0 to 1.0).
     * @param ch Channel index.
     * @param charType Ballistics character style.
     * @param relMs Release time parameter (used directly for Opto math).
     * @return The smoothed, envelope-controlled linear gain to apply.
     */
    [[nodiscard]] T applyBallisticsLinear(T targetGain, int ch, Character charType, T relMs) noexcept
    {
        T& env = envState_[ch]; // Current state (linear gain)
        T fs = static_cast<T>(sampleRate_);

        T effectiveRelCoeff = releaseCoeff_;
        if (detectorType_.load(std::memory_order_relaxed) == DetectorType::SplitPolarity)
        {
            // Release speed dynamically reacts to output volume
            T outputLevel = std::abs(fbLastOutput_[ch]);
            effectiveRelCoeff = releaseCoeff_ / (T(1) + outputLevel);
        }

        switch (charType)
        {
            case Character::Clean:
            {
                // Standard digital one-pole filter mapped to linear amplitude
                T coeff = (targetGain < env) ? attackCoeff_ : effectiveRelCoeff;
                env = targetGain + coeff * (env - targetGain);
                break;
            }

            case Character::Opto:
            {
                T coeff;
                if (targetGain < env)
                {
                    coeff = attackCoeff_;
                }
                else
                {
                    // Opto Memory Effect: Deep compression yields slower recovery
                    T grDepth = T(1) - env;
                    T relMultiplier = T(1) + grDepth * T(0.5);
                    coeff = std::exp(T(-1) / (fs * (relMs * relMultiplier) / T(1000)));
                }
                env = targetGain + coeff * (env - targetGain);
                break;
            }

            case Character::FET:
            {
                if (targetGain < env)
                {
                    // Clamp attack to hardware minimums
                    T attMs = attackMs_.load(std::memory_order_relaxed);
                    T fetAttackMs = std::max(attMs, T(0.02));
                    T aCoeff = std::exp(T(-1) / (fs * fetAttackMs / T(1000)));
                    env = targetGain + aCoeff * (env - targetGain);
                }
                else
                {
                    // Dual-stage release: Initial snap-back is fast, tail is slow
                    T fastRel = effectiveRelCoeff;
                    T slowRel = std::exp(T(-1) / (fs * relMs * T(3) / T(1000)));
                    T normalizedGR = std::min((T(1) - env) / decibelsToGain(T(-20)), T(1));
                    T coeff = fastRel + (T(1) - normalizedGR) * (slowRel - fastRel);
                    env = targetGain + coeff * (env - targetGain);
                }
                break;
            }

            case Character::Varimu:
            {
                // Attack and release are both intrinsically bound to the time constants with non-linear mapping
                T attMs = attackMs_.load(std::memory_order_relaxed);
                T coeff;
                if (targetGain < env)
                    coeff = std::exp(T(-1) / (fs * attMs * T(1.5) / T(1000)));
                else
                    coeff = std::exp(T(-1) / (fs * relMs * T(2) / T(1000)));
                env = targetGain + coeff * (env - targetGain);
                break;
            }
        }
        return env;
    }

    // ---- Sidechain HPF ----

    /** @brief Updates the normalized DC-blocker / High-pass filter coefficients. */
    void updateHpfCoefficients() noexcept
    {
        T scFreq = scHpfFreq_.load(std::memory_order_relaxed);
        scHpfB1_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0 * static_cast<double>(scFreq) / sampleRate_));
        scHpfA0_ = (T(1) + scHpfB1_) / T(2); // Normalization to prevent high-frequency boost
    }

    /** 
     * @brief Applies sidechain filtering for a specific channel.
     * @param input Raw sidechain sample.
     * @param ch Channel index.
     * @return High-pass filtered sample.
     */
    [[nodiscard]] T applySidechainHPF(T input, int ch) noexcept
    {
        T& xp = scHpfPrev_[ch];
        T& yp = scHpfState_[ch];
        T output = scHpfA0_ * (input - xp) + scHpfB1_ * yp;
        xp = input;
        yp = output;
        return output;
    }

    // ---- RMS Configuration ----

    /** @brief Safely recalculates the RMS window size ensuring zero heap allocations. */
    void updateRmsWindow() noexcept
    {
        if (sampleRate_ > 0)
        {
            int requestedSamples = static_cast<int>(sampleRate_ * static_cast<double>(rmsWindowMs_) / 1000.0);

            // Limit logical window size to the pre-allocated physical capacity
            if (!rmsBuffers_[0].empty())
            {
                int maxCapacity = static_cast<int>(rmsBuffers_[0].capacity());
                rmsWindowSamples_ = std::clamp(requestedSamples, 1, maxCapacity);
            }
            else
            {
                rmsWindowSamples_ = std::max(1, requestedSamples);
            }

            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                // Clear the window contents too: stale squares from a previous
                // window length would otherwise be subtracted from the fresh
                // running sum (transient negative-sum glitch).
                std::fill(rmsBuffers_[ch].begin(), rmsBuffers_[ch].end(), T(0));
                rmsSums_[ch] = T(0);
                rmsIndices_[ch] = 0;
                rmsRecomputeCounters_[ch] = 0;
            }
        }
    }

    // =========================================================================
    // Members & State
    // =========================================================================

    AudioSpec spec_ {};     ///< Active audio environment specification.
    double sampleRate_ = 0; ///< Cached operating sample rate.

    // User Parameters (Atomic for thread safety)
    std::atomic<T> threshold_ { T(-20) };   ///< Threshold in dB.
    std::atomic<T> ratio_ { T(4) };         ///< Ratio (e.g., 4 = 4:1).
    std::atomic<T> attackMs_ { T(5) };      ///< Attack time in milliseconds.
    std::atomic<T> releaseMs_ { T(100) };   ///< Release time in milliseconds.
    std::atomic<T> kneeWidth_ { T(0) };     ///< Knee width in dB.
    std::atomic<T> makeupGain_ { T(0) };    ///< Static makeup gain in dB.
    std::atomic<T> stereoLink_ { T(1) };    ///< Stereo linking amount (0 to 1).
    std::atomic<T> mix_ { T(1) };           ///< Wet/Dry mix (1 = full wet).
    std::atomic<T> lookaheadMs_ { T(0) };   ///< Lookahead latency target in ms.
    std::atomic<bool> autoMakeup_ { true }; ///< Auto-gain compensation toggle.

    std::atomic<DetectorType> detectorType_ { DetectorType::Peak }; ///< Selected detection method.
    std::atomic<Topology> topology_ { Topology::FeedForward };      ///< Selected topology.
    std::atomic<Character> character_ { Character::Clean };         ///< Selected ballistics.
    std::atomic<Mode> mode_ { Mode::Downward };                     ///< Selected compression mode.

    // Internal DSP Coefficients & State
    T attackCoeff_ = T(0);              ///< Calculated exponential attack factor.
    T releaseCoeff_ = T(0);             ///< Calculated exponential release factor.
    T autoMakeupCoeff_ = T(0.9995);     ///< Auto-makeup tracking factor.
    T autoMakeupEnv_ = T(0);            ///< Smoothed internal auto-makeup envelope.

    SmoothedValue<T> thresholdSmooth_;  ///< De-zippered threshold.
    SmoothedValue<T> ratioSmooth_;      ///< De-zippered ratio.
    SmoothedValue<T> kneeSmooth_;       ///< De-zippered knee.

    std::array<T, kMaxChannels> envState_ {};       ///< Linear gain tracking per channel.
    std::array<T, kMaxChannels> fbLastOutput_ {};   ///< Feedback topology history buffer.
    std::array<T, kMaxChannels> channelLevelDb_ {}; ///< Raw detected level buffer.

    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuffers_ {}; ///< Lookahead delay lines.
    int lookaheadSamples_ = 0; ///< Active lookahead latency in samples.

    std::array<Hilbert<T>, kMaxChannels> hilbertDetectors_ {}; ///< Analytic signal generators for detection.

    // Sidechain Filtering
    std::atomic<bool> scHpfEnabled_ { false }; ///< HPF toggle.
    std::atomic<T> scHpfFreq_ { T(80) };       ///< HPF Cutoff frequency.
    T scHpfB1_ = T(0);                         ///< HPF internal feedback coefficient.
    T scHpfA0_ = T(0);                         ///< HPF normalized feedforward coefficient.
    std::array<T, kMaxChannels> scHpfState_ {};///< HPF y[n-1] state.
    std::array<T, kMaxChannels> scHpfPrev_ {}; ///< HPF x[n-1] state.

    // Split-Polarity Detector State
    std::array<T, kMaxChannels> splitPosEnv_ {}; ///< Positive half-wave tracking.
    std::array<T, kMaxChannels> splitNegEnv_ {}; ///< Negative half-wave tracking.

    // RMS Detector
    T rmsWindowMs_ = T(10); ///< Active RMS length in ms (audio-thread copy).
    std::atomic<T> rmsWindowMsAtomic_ { T(10) }; ///< Control-thread published target.
    std::atomic<bool> rmsWindowDirty_ { false }; ///< Applied at the next block.
    int rmsWindowSamples_ = 0; ///< Active RMS length in samples.
    std::array<std::vector<T>, kMaxChannels> rmsBuffers_; ///< Pre-allocated RMS sliding windows.
    std::array<T, kMaxChannels> rmsSums_ {}; ///< Running sums for RMS.
    std::array<int, kMaxChannels> rmsIndices_ {}; ///< Write heads for RMS buffers.
    static constexpr int kRmsRecomputePeriod = 4096; ///< Re-summation interval to halt drift.
    std::array<int, kMaxChannels> rmsRecomputeCounters_ {}; ///< Re-summation counters.

    // Shared ITU-R BS.1770-4 true-peak detector (Core/TruePeakDetector.h).
    TruePeakDetector<T, kMaxChannels> truePeak_;

    // Hold & Range
    std::atomic<T> holdMs_  { T(0) };   ///< Gain-hold time in ms (0 = off).
    std::atomic<T> rangeDb_ { T(100) }; ///< Max |gain change| in dB.
    int holdSamples_ = 0;
    std::array<int, kMaxChannels> holdCounters_ {};
    std::array<T, kMaxChannels> heldGrDb_ {};

    std::atomic<T> gainReductionDb_ { T(0) }; ///< Publicly readable Gain Reduction meter.
};

} // namespace dspark
