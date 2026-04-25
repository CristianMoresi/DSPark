// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Compressor.h
 * @brief Modular dynamic range compressor with progressive disclosure API.
 *
 * Professional compressor with interchangeable detector, topology, and
 * ballistics character — all controlled via enums (zero virtual dispatch).
 * Operates in the log (dB) domain for mathematically correct curves.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** Set threshold, ratio, attack, release — works.
 * - **Level 2 (intermediate):** Knee, makeup, stereo link, dry/wet, lookahead.
 * - **Level 3 (expert):** Detector type, topology, character, oversampling,
 *   sidechain HPF, RMS window, all internal modules.
 *
 * Architecture:
 * ```
 *   Input → [Sidechain HPF] → detectLevel() → computeGain() →
 *   applyBallistics() → applyGain() → [DryWetMix] → Output
 * ```
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, SmoothedValue.h,
 *               DryWetMixer.h, RingBuffer.h, Oversampling.h, DenormalGuard.h.
 *
 * @code
 *   // Level 1 — Desktop developer:
 *   dspark::Compressor<float> comp;
 *   comp.prepare(spec);
 *   comp.setThreshold(-20.0f);
 *   comp.setRatio(4.0f);
 *   comp.processBlock(buffer);
 *
 *   // Level 3 — DSP engineer:
 *   comp.setDetector(dspark::Compressor<float>::DetectorType::Rms);
 *   comp.setTopology(dspark::Compressor<float>::Topology::FeedBack);
 *   comp.setCharacter(dspark::Compressor<float>::Character::Opto);
 *   comp.setLookahead(5.0f);
 *   comp.setOversampling(4);
 *   comp.setSidechainHPF(true, 80.0f);
 *   comp.setMix(0.5f);   // Parallel compression
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/SmoothedValue.h"
#include "../Core/DryWetMixer.h"
#include "../Core/RingBuffer.h"
#include "../Core/Oversampling.h"
#include "../Core/DenormalGuard.h"

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
 * @brief Modular compressor with detector/topology/character enums.
 *
 * All module selection is via enum + switch — branch prediction handles
 * performance, no virtual dispatch, no template metaprogramming for the user.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Compressor
{
public:
    ~Compressor() = default;

    /** @brief Level detection mode. */
    enum class DetectorType
    {
        Peak,           ///< Instantaneous absolute value (fast, standard).
        Rms,            ///< Sliding-window RMS (smoother, musical).
        TruePeak,       ///< 4x oversampled peak detection (ISP-aware).
        SplitPolarity   ///< Independent pos/neg half-wave tracking (ButterComp2-style).
    };

    /** @brief Compression topology. */
    enum class Topology
    {
        FeedForward,  ///< Standard: detector reads input (predictable, transparent).
        FeedBack      ///< Vintage: detector reads output (auto-regulating, colored).
    };

    /** @brief Ballistics character. */
    enum class Character
    {
        Clean,    ///< Standard one-pole attack/release (transparent).
        Opto,     ///< Program-dependent: release slows with more compression.
        FET,      ///< Fast attack (min 0.02ms), snap-back release.
        Varimu    ///< Variable ratio: increases with input level (ultra-smooth).
    };

    /** @brief Compression mode. */
    enum class Mode
    {
        Downward,  ///< Standard: reduce gain above threshold (default).
        Upward     ///< Boost gain below threshold (amplify quiet signals).
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the compressor for processing.
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        sampleRate_ = spec.sampleRate;

        T fs = static_cast<T>(sampleRate_);
        T attMs = std::max(attackMs_.load(std::memory_order_relaxed), T(0.01));
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));
        if (fs > T(0))
        {
            attackCoeff_  = std::exp(T(-1) / (fs * attMs / T(1000)));
            releaseCoeff_ = std::exp(T(-1) / (fs * relMs / T(1000)));
            autoMakeupCoeff_ = std::exp(T(-1) / (fs * T(0.3)));
        }

        // Smoothed parameters
        T thresh = threshold_.load(std::memory_order_relaxed);
        T rat    = ratio_.load(std::memory_order_relaxed);
        T knee   = kneeWidth_.load(std::memory_order_relaxed);
        thresholdSmooth_.prepare(sampleRate_, 30.0);
        thresholdSmooth_.setCurrentAndTarget(thresh);
        ratioSmooth_.prepare(sampleRate_, 30.0);
        ratioSmooth_.setCurrentAndTarget(std::max(rat, T(1)));
        kneeSmooth_.prepare(sampleRate_, 30.0);
        kneeSmooth_.setCurrentAndTarget(std::max(knee, T(0)));

        // Lookahead ring buffers
        int maxLaSamples = static_cast<int>(sampleRate_ * 0.01) + 1; // max 10ms
        for (int ch = 0; ch < spec.numChannels && ch < kMaxChannels; ++ch)
            lookaheadBuffers_[ch].prepare(maxLaSamples);
        lookaheadSamples_ = static_cast<int>(fs * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));

        // Sidechain HPF coefficient
        T scFreq = scHpfFreq_.load(std::memory_order_relaxed);
        scHpfCoeff_ = static_cast<T>(
            std::exp(-std::numbers::pi * 2.0 * static_cast<double>(scFreq) / sampleRate_));

        // RMS window
        updateRmsWindow();

        // Dry/wet mixer for parallel compression
        mixer_.prepare(spec);

        // True-peak oversampler (4x, detection path only)
        if (detectorType_.load(std::memory_order_relaxed) == DetectorType::TruePeak)
            prepareTruePeakDetector();

        reset();
    }

    /** @brief Prepares from sample rate only (backward compat). */
    void prepare(double sampleRate) noexcept
    {
        AudioSpec spec { sampleRate, 512, 2 };
        prepare(spec);
    }

    /**
     * @brief Processes an audio buffer in-place (self-sidechain).
     * @param buffer Audio data (mono or multi-channel).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlockImpl(buffer, buffer);
    }

    /**
     * @brief Processes audio with an external sidechain signal.
     *
     * Level detection reads from the sidechain buffer; gain is applied to
     * the audio buffer. Mono sidechain with stereo audio is supported
     * (sidechain channel is clamped to available channels).
     *
     * @param audio     Audio buffer to compress (modified in-place).
     * @param sidechain External sidechain signal (read-only).
     */
    void processBlock(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        processBlockImpl(audio, sidechain);
    }

    /**
     * @brief Core processing: detection from sidechain, gain applied to audio.
     */
    void processBlockImpl(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        DenormalGuard guard;
        const int nCh   = std::min(audio.getNumChannels(), kMaxChannels);
        const int scCh  = sidechain.getNumChannels();
        const int nS    = audio.getNumSamples();

        // Sync atomic params to audio-thread state
        thresholdSmooth_.setTargetValue(threshold_.load(std::memory_order_relaxed));
        ratioSmooth_.setTargetValue(std::max(ratio_.load(std::memory_order_relaxed), T(1)));
        kneeSmooth_.setTargetValue(std::max(kneeWidth_.load(std::memory_order_relaxed), T(0)));

        T attMs = std::max(attackMs_.load(std::memory_order_relaxed), T(0.01));
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(0.01));
        T fs = static_cast<T>(sampleRate_);
        if (fs > T(0))
        {
            attackCoeff_  = std::exp(T(-1) / (fs * attMs / T(1000)));
            releaseCoeff_ = std::exp(T(-1) / (fs * relMs / T(1000)));
            autoMakeupCoeff_ = std::exp(T(-1) / (fs * T(0.3)));
            lookaheadSamples_ = static_cast<int>(fs * std::clamp(
                lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));

            T scFreq = scHpfFreq_.load(std::memory_order_relaxed);
            scHpfCoeff_ = static_cast<T>(
                std::exp(-std::numbers::pi * 2.0 * static_cast<double>(scFreq) / sampleRate_));
        }

        // Cache enum/bool params for this block
        auto detType   = detectorType_.load(std::memory_order_relaxed);
        auto topo      = topology_.load(std::memory_order_relaxed);
        auto charType  = character_.load(std::memory_order_relaxed);
        auto modeType  = mode_.load(std::memory_order_relaxed);
        bool scHpf     = scHpfEnabled_.load(std::memory_order_relaxed);
        bool autoMkup  = autoMakeup_.load(std::memory_order_relaxed);
        T mkupGain     = makeupGain_.load(std::memory_order_relaxed);
        T sLink        = stereoLink_.load(std::memory_order_relaxed);
        T mixVal       = mix_.load(std::memory_order_relaxed);

        if (mixVal < T(1))
            mixer_.pushDry(audio);

        for (int i = 0; i < nS; ++i)
        {
            // Smoothed parameters
            T thresh = thresholdSmooth_.getNextValue();
            T ratio  = ratioSmooth_.getNextValue();
            T knee   = kneeSmooth_.getNextValue();

            // --- Detect level (per-channel, then link) ---
            T linkedLevel = T(-200); // dB, will be maxed

            for (int ch = 0; ch < nCh; ++ch)
            {
                // Read from sidechain (clamp channel index)
                int sc = std::min(ch, scCh - 1);
                T sample = sidechain.getChannel(sc)[i];

                // Sidechain HPF
                if (scHpf)
                    sample = applySidechainHPF(sample, ch);

                T levelDb;
                if (topo == Topology::FeedBack)
                    levelDb = detectLevel(fbLastOutput_[ch], ch, detType);
                else
                    levelDb = detectLevel(sample, ch, detType);

                channelLevelDb_[ch] = levelDb;

                if (levelDb > linkedLevel)
                    linkedLevel = levelDb;
            }

            // Stereo linking
            for (int ch = 0; ch < nCh; ++ch)
            {
                T chLevel = channelLevelDb_[ch];
                channelLevelDb_[ch] = chLevel + sLink * (linkedLevel - chLevel);
            }

            // --- Compute gain, apply ballistics, apply gain per channel ---
            T blockGR = T(0);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T inputDb = channelLevelDb_[ch];

                // Gain computer (static curve)
                T gainReduction = computeGain(inputDb, thresh, ratio, knee, charType, modeType);

                // Ballistics (character-dependent)
                T smoothedGR = applyBallistics(gainReduction, ch, charType, relMs);

                // Makeup
                T makeup = mkupGain;
                if (autoMkup && modeType == Mode::Downward)
                    makeup += -autoMakeupEnv_;

                T outputGain = decibelsToGain(smoothedGR + makeup);

                // Apply gain (with lookahead delay if enabled)
                T input;
                if (lookaheadSamples_ > 0)
                {
                    auto& ring = lookaheadBuffers_[ch];
                    ring.push(audio.getChannel(ch)[i]);
                    input = ring.read(lookaheadSamples_);
                }
                else
                {
                    input = audio.getChannel(ch)[i];
                }

                T output = input * outputGain;
                fbLastOutput_[ch] = output;
                audio.getChannel(ch)[i] = output;

                // Metering: track worst GR across channels
                if (ch == 0 || smoothedGR < blockGR)
                    blockGR = smoothedGR;
            }

            gainReductionDb_.store(blockGR, std::memory_order_relaxed);

            // Adaptive auto-makeup: slow-track average GR (use ch0 representative)
            autoMakeupEnv_ = blockGR + autoMakeupCoeff_ * (autoMakeupEnv_ - blockGR);
        }

        // Parallel compression mix
        if (mixVal < T(1))
            mixer_.mixWet(audio, mixVal);
    }

    /**
     * @brief Processes a single sample on one channel.
     *
     * For per-sample modulation and custom feedback loops.
     * Does NOT apply stereo link, parallel mix, or lookahead
     * (those are block-level features). Uses current smoothed parameters.
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Compressed output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        T thresh = thresholdSmooth_.getCurrentValue();
        T ratio  = ratioSmooth_.getCurrentValue();
        T knee   = kneeSmooth_.getCurrentValue();
        auto detType  = detectorType_.load(std::memory_order_relaxed);
        auto topo     = topology_.load(std::memory_order_relaxed);
        auto charType = character_.load(std::memory_order_relaxed);
        auto modeType = mode_.load(std::memory_order_relaxed);
        bool scHpf    = scHpfEnabled_.load(std::memory_order_relaxed);
        bool autoMkup = autoMakeup_.load(std::memory_order_relaxed);
        T relMs       = releaseMs_.load(std::memory_order_relaxed);

        T sidechain = scHpf ? applySidechainHPF(input, channel) : input;
        T levelDb = (topo == Topology::FeedBack)
            ? detectLevel(fbLastOutput_[channel], channel, detType)
            : detectLevel(sidechain, channel, detType);

        T gainReduction = computeGain(levelDb, thresh, ratio, knee, charType, modeType);
        T smoothedGR = applyBallistics(gainReduction, channel, charType, relMs);

        autoMakeupEnv_ = smoothedGR + autoMakeupCoeff_
                       * (autoMakeupEnv_ - smoothedGR);

        T makeup = makeupGain_.load(std::memory_order_relaxed);
        if (autoMkup && modeType == Mode::Downward)
            makeup += -autoMakeupEnv_;
        T outputGain = decibelsToGain(smoothedGR + makeup);

        T output = input * outputGain;
        fbLastOutput_[channel] = output;
        gainReductionDb_.store(smoothedGR, std::memory_order_relaxed);
        return output;
    }

    /** @brief Resets all internal state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            envState_[ch] = T(0);
            fbLastOutput_[ch] = T(0);
            scHpfState_[ch] = T(0);
            scHpfPrev_[ch] = T(0);
            channelLevelDb_[ch] = T(-200);
            lookaheadBuffers_[ch].reset();
        }
        for (auto& buf : rmsBuffers_)
            std::fill(buf.begin(), buf.end(), T(0));
        for (auto& sum : rmsSums_)
            sum = T(0);
        for (auto& idx : rmsIndices_)
            idx = 0;
        for (auto& cnt : rmsRecomputeCounters_)
            cnt = 0;
        tpState_ = {};
        gainReductionDb_.store(T(0), std::memory_order_relaxed);
        autoMakeupEnv_ = T(0);
        splitPosEnv_.fill(T(0));
        splitNegEnv_.fill(T(0));
        thresholdSmooth_.skip();
        ratioSmooth_.skip();
        kneeSmooth_.skip();
        mixer_.reset();
    }

    // =========================================================================
    // Level 1: Simple API — Just threshold/ratio/attack/release
    // =========================================================================

    /** @brief Sets the compression threshold in dB. */
    void setThreshold(T dB) noexcept
    {
        threshold_.store(dB, std::memory_order_relaxed);
    }

    /** @brief Sets the compression ratio (1.0 = off, 4.0 = 4:1, >20 = limiter). */
    void setRatio(T ratio) noexcept
    {
        ratio_.store(std::max(ratio, T(1)), std::memory_order_relaxed);
    }

    /** @brief Sets the attack time in milliseconds. */
    void setAttack(T ms) noexcept
    {
        attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
    }

    /** @brief Sets the release time in milliseconds. */
    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
    }

    // =========================================================================
    // Level 2: Intermediate API
    // =========================================================================

    /** @brief Sets knee width in dB (0 = hard, 6 = soft, 12 = very soft). */
    void setKnee(T dB) noexcept
    {
        kneeWidth_.store(std::max(dB, T(0)), std::memory_order_relaxed);
    }

    /** @brief Sets manual makeup gain in dB. */
    void setMakeupGain(T dB) noexcept { makeupGain_.store(dB, std::memory_order_relaxed); }

    /** @brief Enables auto makeup gain. */
    void setAutoMakeup(bool on) noexcept { autoMakeup_.store(on, std::memory_order_relaxed); }

    /**
     * @brief Sets compression mode (downward or upward).
     *
     * - **Downward** (default): Reduces gain when signal is above threshold.
     * - **Upward**: Boosts gain when signal is below threshold, amplifying quiet
     *   signals while leaving loud signals untouched.
     */
    void setMode(Mode mode) noexcept { mode_.store(mode, std::memory_order_relaxed); }

    /** @brief Sets stereo link amount (0 = independent, 1 = fully linked). */
    void setStereoLink(T amount) noexcept
    {
        stereoLink_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets dry/wet mix for parallel compression.
     * @param dryWet 1.0 = fully compressed, 0.5 = NY-style parallel.
     */
    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    /**
     * @brief Sets lookahead time in milliseconds.
     *
     * Lookahead lets the compressor "see" transients before they arrive,
     * enabling transparent compression of fast attacks. Adds latency.
     *
     * @param ms Lookahead in ms (0 = off, max 10 ms).
     */
    void setLookahead(T ms) noexcept
    {
        lookaheadMs_.store(std::clamp(ms, T(0), T(10)), std::memory_order_relaxed);
    }

    // =========================================================================
    // Level 3: Expert API — Full modular control
    // =========================================================================

    /**
     * @brief Sets the level detector type.
     *
     * - **Peak**: Standard absolute value (fast, standard in most compressors).
     * - **Rms**: Sliding-window RMS (smoother, responds to average level).
     * - **TruePeak**: 4x oversampled peak detection (ISP-aware, broadcast).
     *
     * @param type Detector type.
     */
    void setDetector(DetectorType type) noexcept
    {
        detectorType_.store(type, std::memory_order_relaxed);
        if (type == DetectorType::TruePeak && sampleRate_ > 0)
            prepareTruePeakDetector();
    }

    /**
     * @brief Sets the compression topology.
     *
     * - **FeedForward**: Detector reads input signal. Predictable, transparent.
     * - **FeedBack**: Detector reads output signal. Self-regulating, vintage color.
     *   Classic hardware (LA-2A, 1176) used feedback topology.
     *
     * @param topo Topology.
     */
    void setTopology(Topology topo) noexcept { topology_.store(topo, std::memory_order_relaxed); }

    /**
     * @brief Sets the ballistics character.
     *
     * - **Clean**: Standard one-pole smoothing. Transparent.
     * - **Opto**: Program-dependent release (slows with more compression).
     *   Emulates optical compressors (LA-2A). Smooth on vocals/bass.
     * - **FET**: Ultra-fast attack (min 0.02ms). Release has fast initial +
     *   slow tail. Aggressive on drums/transients. Emulates FET compressors (1176).
     * - **Varimu**: Ratio increases with input level (not fixed). Ultra-smooth
     *   bus compression. Emulates variable-mu tubes (Fairchild 670).
     *
     * @param type Character type.
     */
    void setCharacter(Character type) noexcept { character_.store(type, std::memory_order_relaxed); }

    /**
     * @brief Enables oversampling in the detection path.
     *
     * Higher oversampling improves detection accuracy for transients.
     * Only affects the sidechain, not the audio signal path.
     *
     * @param factor Oversampling factor (1 = off, 2, 4).
     */
    void setOversampling(int factor) noexcept
    {
        oversamplingFactor_.store(std::clamp(factor, 1, 4), std::memory_order_relaxed);
    }

    /**
     * @brief Enables the sidechain high-pass filter.
     *
     * Removes low-frequency content from the sidechain signal so bass
     * doesn't trigger unwanted compression (pumping).
     *
     * @param enabled True to enable.
     * @param cutoffHz Cutoff frequency (default: 80 Hz).
     */
    void setSidechainHPF(bool enabled, T cutoffHz = T(80)) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        scHpfFreq_.store(cutoffHz, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the RMS detector window size.
     * @param ms Window in milliseconds (default: 10 ms).
     */
    void setRmsWindow(T ms) noexcept
    {
        rmsWindowMs_ = std::max(ms, T(1));
        updateRmsWindow();
    }

    // -- Metering ---------------------------------------------------------------

    /**
     * @brief Returns the current gain reduction in dB (negative value).
     * @return e.g. -6.0 means 6 dB of gain reduction.
     */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReductionDb_.load(std::memory_order_relaxed); }

    /** @brief Returns the current detector type. */
    [[nodiscard]] DetectorType getDetector() const noexcept { return detectorType_.load(std::memory_order_relaxed); }

    /** @brief Returns the current topology. */
    [[nodiscard]] Topology getTopology() const noexcept { return topology_.load(std::memory_order_relaxed); }

    /** @brief Returns the current character. */
    [[nodiscard]] Character getCharacter() const noexcept { return character_.load(std::memory_order_relaxed); }

    /** @brief Returns lookahead latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return lookaheadSamples_; }

protected:
    static constexpr int kMaxChannels = 16;

    // ---- Detector implementations ----

    [[nodiscard]] T detectLevel(T sample, int ch, DetectorType detType) noexcept
    {
        T level = std::abs(sample);
        switch (detType)
        {
            case DetectorType::Peak:
                // Already computed above as the default.
                break;

            case DetectorType::Rms:
            {
                T sq = sample * sample;
                auto& buf = rmsBuffers_[ch];
                auto& sum = rmsSums_[ch];
                auto& idx = rmsIndices_[ch];
                auto& recomputeCount = rmsRecomputeCounters_[ch];
                int len = rmsWindowSamples_;

                if (len > 0 && len <= static_cast<int>(buf.size()))
                {
                    sum -= buf[idx];
                    buf[idx] = sq;
                    sum += sq;
                    idx = (idx + 1) % len;

                    if (++recomputeCount >= kRmsRecomputePeriod)
                    {
                        sum = T(0);
                        for (int j = 0; j < len; ++j)
                            sum += buf[j];
                        recomputeCount = 0;
                    }

                    level = std::sqrt(std::max(sum / static_cast<T>(len), T(0)));
                }
                else
                {
                    level = std::abs(sample);
                }
                break;
            }

            case DetectorType::TruePeak:
            {
                level = detectTruePeakSample(sample, ch);
                break;
            }

            case DetectorType::SplitPolarity:
            {
                // ButterComp2-style: independent pos/neg half-wave tracking
                T pos = std::max(sample, T(0));
                T neg = std::max(-sample, T(0));

                T posCoeff = (pos > splitPosEnv_[ch]) ? T(0.6) : T(0.99);
                T negCoeff = (neg > splitNegEnv_[ch]) ? T(0.6) : T(0.99);

                splitPosEnv_[ch] = pos + posCoeff * (splitPosEnv_[ch] - pos);
                splitNegEnv_[ch] = neg + negCoeff * (splitNegEnv_[ch] - neg);

                // Combine: max of both envelopes captures asymmetric content
                level = std::max(splitPosEnv_[ch], splitNegEnv_[ch]);
                break;
            }
        }

        return gainToDecibels(level);
    }

    // ---- Gain computer ----

    [[nodiscard]] T computeGain(T inputDb, T thresh, T ratio, T knee,
                                Character charType, Mode modeType) const noexcept
    {
        if (modeType == Mode::Upward)
            return computeGainUpward(inputDb, thresh, ratio, knee);

        // --- Downward compression ---

        // Varimu: ratio increases with level
        T effectiveRatio = ratio;
        if (charType == Character::Varimu && inputDb > thresh)
        {
            T excess = inputDb - thresh;
            effectiveRatio = ratio * (T(1) + excess / T(40));
        }

        if (knee <= T(0))
        {
            // Hard knee
            if (inputDb <= thresh)
                return T(0);
            return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
        }
        else
        {
            // Soft knee (parabolic)
            T halfKnee = knee / T(2);
            T lower = thresh - halfKnee;
            T upper = thresh + halfKnee;

            if (inputDb <= lower)
                return T(0);

            if (inputDb >= upper)
                return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);

            T x = inputDb - lower;
            return (T(1) - T(1) / effectiveRatio) * x * x / (T(2) * knee) * T(-1);
        }
    }

    /// Upward compression: boosts signals below threshold.
    [[nodiscard]] T computeGainUpward(T inputDb, T thresh, T ratio, T knee) const noexcept
    {
        T effectiveRatio = ratio;

        if (knee <= T(0))
        {
            // Hard knee
            if (inputDb >= thresh)
                return T(0);
            // Positive gain (boost): the further below threshold, the more boost
            return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
        }
        else
        {
            // Soft knee (parabolic, mirrored)
            T halfKnee = knee / T(2);
            T lower = thresh - halfKnee;
            T upper = thresh + halfKnee;

            if (inputDb >= upper)
                return T(0);

            if (inputDb <= lower)
                return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);

            // Transition region: parabolic interpolation
            T x = upper - inputDb;
            return (T(1) - T(1) / effectiveRatio) * x * x / (T(2) * knee);
        }
    }

    // ---- Ballistics (character-dependent) ----

    [[nodiscard]] T applyBallistics(T gainReduction, int ch, Character charType, T relMs) noexcept
    {
        T& env = envState_[ch];
        T fs = static_cast<T>(sampleRate_);

        // Output-dependent recovery for SplitPolarity detector:
        // release gets faster when output is louder (ButterComp2-style)
        T effectiveRelCoeff = releaseCoeff_;
        if (detectorType_.load(std::memory_order_relaxed) == DetectorType::SplitPolarity)
        {
            T outputLevel = std::abs(fbLastOutput_[ch]);
            effectiveRelCoeff = releaseCoeff_ / (T(1) + outputLevel);
        }

        switch (charType)
        {
            case Character::Clean:
            {
                T coeff = (gainReduction < env) ? attackCoeff_ : effectiveRelCoeff;
                env = gainReduction + coeff * (env - gainReduction);
                break;
            }

            case Character::Opto:
            {
                T coeff;
                if (gainReduction < env)
                {
                    coeff = attackCoeff_;
                }
                else
                {
                    T grDepth = std::abs(env);
                    T relMultiplier = T(1) + grDepth * T(0.05);
                    T adjustedRelease = relMs * relMultiplier;
                    coeff = std::exp(T(-1) / (fs * adjustedRelease / T(1000)));
                }
                env = gainReduction + coeff * (env - gainReduction);
                break;
            }

            case Character::FET:
            {
                if (gainReduction < env)
                {
                    T attMs = attackMs_.load(std::memory_order_relaxed);
                    T fetAttackMs = std::max(attMs, T(0.02));
                    T aCoeff = std::exp(T(-1) / (fs * fetAttackMs / T(1000)));
                    env = gainReduction + aCoeff * (env - gainReduction);
                }
                else
                {
                    T fastRel = effectiveRelCoeff;
                    T slowRel = std::exp(T(-1) / (fs * relMs * T(3) / T(1000)));

                    T normalizedGR = std::min(std::abs(env) / T(20), T(1));
                    T coeff = fastRel + (T(1) - normalizedGR) * (slowRel - fastRel);
                    env = gainReduction + coeff * (env - gainReduction);
                }
                break;
            }

            case Character::Varimu:
            {
                T attMs = attackMs_.load(std::memory_order_relaxed);
                T coeff;
                if (gainReduction < env)
                    coeff = std::exp(T(-1) / (fs * attMs * T(1.5) / T(1000)));
                else
                    coeff = std::exp(T(-1) / (fs * relMs * T(2) / T(1000)));
                env = gainReduction + coeff * (env - gainReduction);
                break;
            }
        }

        return env;
    }

    // ---- Sidechain HPF ----

    [[nodiscard]] T applySidechainHPF(T input, int ch) noexcept
    {
        T& xp = scHpfPrev_[ch];
        T& yp = scHpfState_[ch];
        T output = input - xp + scHpfCoeff_ * yp;
        xp = input;
        yp = output;
        return output;
    }

    // ---- Makeup ----

    void updateRmsWindow() noexcept
    {
        if (sampleRate_ > 0)
        {
            rmsWindowSamples_ = std::max(1, static_cast<int>(
                sampleRate_ * static_cast<double>(rmsWindowMs_) / 1000.0));
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                rmsBuffers_[ch].assign(static_cast<size_t>(rmsWindowSamples_), T(0));
                rmsSums_[ch] = T(0);
                rmsIndices_[ch] = 0;
            }
        }
    }

    void prepareTruePeakDetector() noexcept
    {
        buildTruePeakFilter();
        tpState_ = {};
    }

    // ---- Members ----

    AudioSpec spec_ {};
    double sampleRate_ = 0;

    // Atomic parameters (thread-safe setters from any thread)
    std::atomic<T> threshold_ { T(-20) };
    std::atomic<T> ratio_ { T(4) };
    std::atomic<T> attackMs_ { T(5) };
    std::atomic<T> releaseMs_ { T(100) };
    std::atomic<T> kneeWidth_ { T(0) };
    std::atomic<T> makeupGain_ { T(0) };
    std::atomic<T> stereoLink_ { T(1) };
    std::atomic<T> mix_ { T(1) };
    std::atomic<T> lookaheadMs_ { T(0) };
    std::atomic<bool> autoMakeup_ { true };

    // Module selection (atomic enums)
    std::atomic<DetectorType> detectorType_ { DetectorType::Peak };
    std::atomic<Topology> topology_ { Topology::FeedForward };
    std::atomic<Character> character_ { Character::Clean };
    std::atomic<Mode> mode_ { Mode::Downward };
    std::atomic<int> oversamplingFactor_ { 1 };

    // Coefficients
    T attackCoeff_ = T(0);
    T releaseCoeff_ = T(0);
    T autoMakeupCoeff_ = T(0.9995);  // ~300ms at 44.1kHz
    T autoMakeupEnv_ = T(0);         // Smoothed average GR in dB

    // Parameter smoothing
    SmoothedValue<T> thresholdSmooth_;
    SmoothedValue<T> ratioSmooth_;
    SmoothedValue<T> kneeSmooth_;

    // Per-channel state
    std::array<T, kMaxChannels> envState_ {};
    std::array<T, kMaxChannels> fbLastOutput_ {};
    std::array<T, kMaxChannels> channelLevelDb_ {};

    // Lookahead
    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuffers_ {};
    int lookaheadSamples_ = 0;

    // Sidechain HPF
    std::atomic<bool> scHpfEnabled_ { false };
    std::atomic<T> scHpfFreq_ { T(80) };
    T scHpfCoeff_ = T(0.995);
    std::array<T, kMaxChannels> scHpfState_ {};
    std::array<T, kMaxChannels> scHpfPrev_ {};

    // Split-polarity detector state (ButterComp2-style)
    std::array<T, kMaxChannels> splitPosEnv_ {};
    std::array<T, kMaxChannels> splitNegEnv_ {};

    // RMS detector
    T rmsWindowMs_ = T(10);
    int rmsWindowSamples_ = 0;
    std::array<std::vector<T>, kMaxChannels> rmsBuffers_;
    std::array<T, kMaxChannels> rmsSums_ {};
    std::array<int, kMaxChannels> rmsIndices_ {};
    static constexpr int kRmsRecomputePeriod = 4096;
    std::array<int, kMaxChannels> rmsRecomputeCounters_ {};

    // True-peak detector: ITU-R BS.1770-4 compliant FIR 4x oversampling
    static constexpr int kTpTaps = 12;
    static constexpr int kTpPhases = 3;

    struct TruePeakState {
        T history[kTpTaps] = {};  ///< history[0] = oldest, history[kTpTaps-1] = newest
    };
    std::array<TruePeakState, kMaxChannels> tpState_{};
    std::array<std::array<T, kTpTaps>, kTpPhases> tpCoeffs_{};

    void buildTruePeakFilter() noexcept
    {
        constexpr int N = kTpTaps * 4;
        constexpr double M = (N - 1) / 2.0;
        constexpr double fc = 0.25;
        constexpr double beta = 8.0;
        constexpr double pi = std::numbers::pi;

        auto besselI0 = [](double x) -> double {
            double sum = 1.0, term = 1.0;
            for (int k = 1; k <= 25; ++k)
            {
                double half = x / (2.0 * k);
                term *= half * half;
                sum += term;
                if (term < 1e-15 * sum) break;
            }
            return sum;
        };

        const double i0Beta = besselI0(beta);
        double h[N];
        for (int n = 0; n < N; ++n)
        {
            double x = static_cast<double>(n) - M;
            double sincArg = 2.0 * fc * x;
            double sincVal = (std::abs(sincArg) < 1e-10)
                ? 1.0
                : std::sin(pi * sincArg) / (pi * sincArg);
            double t = x / M;
            double kaiserVal = (std::abs(t) > 1.0)
                ? 0.0
                : besselI0(beta * std::sqrt(1.0 - t * t)) / i0Beta;
            h[n] = sincVal * kaiserVal;
        }

        for (int phase = 0; phase < kTpPhases; ++phase)
        {
            int p = phase + 1;
            for (int k = 0; k < kTpTaps; ++k)
                tpCoeffs_[phase][k] = static_cast<T>(h[4 * k + p]);
        }
    }

    [[nodiscard]] T detectTruePeakSample(T sample, int ch) noexcept
    {
        auto& tp = tpState_[ch];
        for (int k = 0; k < kTpTaps - 1; ++k)
            tp.history[k] = tp.history[k + 1];
        tp.history[kTpTaps - 1] = sample;

        T peak = std::abs(sample);
        for (int phase = 0; phase < kTpPhases; ++phase)
        {
            T interp = T(0);
            for (int k = 0; k < kTpTaps; ++k)
                interp += tp.history[kTpTaps - 1 - k] * tpCoeffs_[phase][k];
            T absInterp = std::abs(interp);
            if (absInterp > peak)
                peak = absInterp;
        }
        return peak;
    }

    // Dry/wet (parallel compression)
    DryWetMixer<T> mixer_;

    // Metering
    std::atomic<T> gainReductionDb_ { T(0) };
};

} // namespace dspark
