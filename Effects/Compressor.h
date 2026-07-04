// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Compressor.h
 * @brief Modular dynamic range compressor with analog-modeled ballistics and Hilbert detection.
 *
 * Professional compressor with interchangeable detector, topology, and
 * ballistics character. Gain smoothing runs in the log (dB) domain: the
 * transparent characters use a smoothed branching one-pole (the design
 * recommended by Giannoulis/Massberg/Reiss, JAES 2012), and the analog
 * characters (Opto, FET) run a dual-envelope model with compression-history
 * memory that reproduces the two-stage program-dependent release of the
 * hardware they are named after.
 *
 * FeedBack operation is calibrated to the observed curve: the gain element's
 * law is the closed-form inverse of the user's static curve, so the settled
 * loop lands exactly on the requested ratio and knee (hardware front panels
 * are marked with observed ratios; a raw feed-forward law inside a feedback
 * loop can never compress past 2:1). With the peak detector the loop is
 * resolved semi-implicitly per sample, which keeps it unconditionally stable
 * down to the FET's 20 us attack.
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
 * @note The gain path runs at the host rate on purpose: worst-case
 *       gain-modulation aliasing measures <= -72 dBc (FET character at its
 *       minimum attack), inaudible on real material. Products that oversample
 *       a nonlinear section can wrap this compressor in that section like any
 *       other stage with Core/Oversampling: one resampler for the whole
 *       section instead of one per module (see docs/cookbook.md,
 *       "Oversampling a nonlinear section").
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
        Hilbert         ///< Analytic-signal magnitude: ripple-free envelope (lowest THD on
                        ///< sustained material). The 191-tap FIR detects ~95 samples late,
                        ///< so the audio path is delayed by the same amount to stay aligned
                        ///< with the gain (transients are caught; the delay is reported by
                        ///< getLatency()). Feedback operation keeps its loop causal, so the
                        ///< alignment delay is disabled there; processSample() (documented
                        ///< as lookahead-free) does not apply it either.
    };

    /** @brief Signal routing topology for the detector sidechain. */
    enum class Topology
    {
        FeedForward,  ///< Detector reads uncompressed input (modern, precise, transparent).
        FeedBack      ///< Detector reads compressed output (vintage, self-regulating,
                      ///< colored). The static curve still honours the panel values: the
                      ///< element's law is solved so the closed loop settles exactly on
                      ///< the requested ratio and knee, like the hardware whose panel
                      ///< markings reflect measured curves. External sidechain keys are
                      ///< ignored here (the detector is wired to the output).
    };

    /**
     * @brief Time-constant behavior and release curve shape.
     *
     * All characters smooth the gain reduction in dB (log domain), so attack
     * and release knobs read as the t63 time constant of the dB envelope.
     */
    enum class Character
    {
        Clean,    ///< Smoothed branching one-pole in the log domain (modern transparent
                  ///< design): constant dB-per-second ballistics, knobs mean what they say.

        Opto,     ///< T4 optical cell model (LA-2A lineage), calibrated to the
                  ///< Teletronix spec: ~50% recovers in one release time (the spec's
                  ///< "0.06 s to 50%" is release = 60 ms), complete release 0.5-5 s
                  ///< depending on how long the compressor was engaged (the slow
                  ///< memory stage only charges under sustained compression), and a
                  ///< 10 ms attack floor (the cell's published attack time). The knee
                  ///< has the same 10 dB physical floor as Varimu: a photocell's
                  ///< resistance curve is gradual and cannot form a hard corner.

        FET,      ///< 1176 lineage. Always detects in FeedBack topology with the
                  ///< hardware's peak rectifier (the Topology and Detector settings
                  ///< are ignored and lookahead is unavailable while selected).
                  ///< Attack clamps to the hardware range 0.02-0.8 ms and release to
                  ///< 50-1100 ms; release is two-stage with a program-dependent slow
                  ///< tail (~t63 at the knob value). The panel ratio is the observed
                  ///< curve, like the hardware: 4/8 compress, 12/20 limit.
                  ///< setCharacterColor() adds the FET's 2nd-order channel-modulation
                  ///< harmonics, calibrated at 1.0 to the 1176 THD spec (< 0.5%
                  ///< while limiting).

        Varimu    ///< Variable-mu tube lineage (Fairchild). The effective ratio grows
                  ///< with level above the threshold and the knee has a 10 dB floor
                  ///< (a remote-cutoff tube cannot produce a hard corner). Log-domain
                  ///< ballistics with the user's time constants.
    };

    /** @brief Processing direction. */
    enum class Mode
    {
        Downward,  ///< Standard: Reduces dynamic range by attenuating signals above the threshold.
        Upward     ///< Upward: Reduces dynamic range by boosting signals below the threshold.
                   ///< The boost fades out when the SUSTAINED level (peak-held, falling
                   ///< 60 dB/s) drops 40 to 60 dB below the threshold, so pauses and the
                   ///< noise floor are not amplified; setRange() bounds the boost.
    };

    /** @brief Automatic makeup-gain behavior (applies in Downward mode). */
    enum class AutoMakeupMode
    {
        Off,      ///< Manual makeup only (setMakeupGain).
        Static,   ///< Textbook auto makeup: a constant offset equal to half the static
                  ///< gain reduction of a 0 dBFS signal, derived from the (smoothed)
                  ///< threshold/ratio/knee. No program dependence.
        Adaptive  ///< Loudness matching: tracks the smoothed gain reduction (~300 ms)
                  ///< and compensates it in full, keeping the average output level
                  ///< matched to the input (quiet passages are lifted).
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
        timeConstantsDirty_.store(false, std::memory_order_relaxed);

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
        makeupSmooth_.prepare(sampleRate_, 30.0);
        makeupSmooth_.reset(makeupGain_.load(std::memory_order_relaxed));
        mixSmooth_.prepare(sampleRate_, 30.0);
        mixSmooth_.reset(mix_.load(std::memory_order_relaxed));
        colorSmooth_.prepare(sampleRate_, 30.0);
        colorSmooth_.reset(characterColor_.load(std::memory_order_relaxed));

        // Pre-allocate and initialize per-channel instances. Capacity covers
        // the 10 ms user lookahead plus the Hilbert detector's group delay
        // (the audio is delayed by it to stay aligned with the envelope).
        int maxLaSamples = static_cast<int>(sampleRate_ * 0.01) + 1
                         + Hilbert<T>::getLatencySamples();
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
     * @note Shared state (parameter smoothers, auto-makeup envelope) advances
     *       once per sample FRAME and is driven by channel 0. Multi-channel
     *       per-sample workflows must therefore include channel 0; other
     *       channels read the current smoothed values without advancing them.
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
        makeupSmooth_.setTargetValue(makeupGain_.load(std::memory_order_relaxed));
        colorSmooth_.setTargetValue(characterColor_.load(std::memory_order_relaxed));

        // Channel 0 advances the shared smoothers; advancing on every call
        // would scale their time constants with the caller's channel count.
        const bool advanceShared = (channel == 0);
        T thresh, ratio, knee, makeupDb, colorAmt;
        if (advanceShared)
        {
            thresh   = thresholdSmooth_.getNextValue();
            ratio    = ratioSmooth_.getNextValue();
            knee     = kneeSmooth_.getNextValue();
            makeupDb = makeupSmooth_.getNextValue();
            colorAmt = colorSmooth_.getNextValue();
        }
        else
        {
            thresh   = thresholdSmooth_.getCurrentValue();
            ratio    = ratioSmooth_.getCurrentValue();
            knee     = kneeSmooth_.getCurrentValue();
            makeupDb = makeupSmooth_.getCurrentValue();
            colorAmt = colorSmooth_.getCurrentValue();
        }

        auto detType   = detectorType_.load(std::memory_order_relaxed);
        auto topo      = topology_.load(std::memory_order_relaxed);
        auto charType  = character_.load(std::memory_order_relaxed);
        auto modeType  = mode_.load(std::memory_order_relaxed);
        bool scHpf     = scHpfEnabled_.load(std::memory_order_relaxed);
        auto autoMkup  = autoMakeupMode_.load(std::memory_order_relaxed);

        if (timeConstantsDirty_.exchange(false, std::memory_order_acquire))
        {
            T fs = static_cast<T>(sampleRate_);
            if (fs > T(0)) updateTimeConstants(fs);
        }

        // The FET character is a feedback design like the 1176 it models: it
        // always detects on the compressed output with the hardware's peak
        // rectifier, whatever Topology and Detector say.
        const Topology topoEff = (charType == Character::FET) ? Topology::FeedBack : topo;
        const DetectorType detTypeEff = (charType == Character::FET) ? DetectorType::Peak : detType;
        const bool fbImplicit = topoEff == Topology::FeedBack
                             && modeType == Mode::Downward
                             && detTypeEff == DetectorType::Peak;

        // The sidechain filter must process whatever the detector consumes:
        // in FeedBack that is the compressed output (previous sample on the
        // explicit path; on the semi-implicit path the loop equation supplies
        // the gain, so the current input feeds the filter directly).
        T detectorIn = (topoEff == Topology::FeedBack && !fbImplicit)
                     ? fbLastOutput_[channel] : input;
        if (scHpf) detectorIn = applySidechainHPF(detectorIn, channel);

        T levelDb = detectLevel(detectorIn, channel, detTypeEff);

        // Sustained-level guard envelope for Upward mode: instant rise,
        // constant 60 dB/s fall (peak hold with linear decay).
        T guardDb = levelDb;
        if (modeType == Mode::Upward)
        {
            T& g = upwardGuardDb_[channel];
            g = std::max(levelDb, g - upwardGuardDecay_);
            guardDb = g;
        }

        T smoothedGR_Db;
        if (fbImplicit)
        {
            smoothedGR_Db = applyBallisticsFeedbackImplicit(levelDb, channel, charType,
                                                            thresh, ratio, knee);
        }
        else if (topoEff == Topology::FeedBack && modeType == Mode::Downward)
        {
            const auto law = computeGainFeedback(levelDb, thresh, ratio, knee, charType);
            const T targetGR_Db = applyHoldAndRange(law.target, channel);
            smoothedGR_Db = applyBallistics(targetGR_Db, channel, law.slope);
        }
        else
        {
            T targetGR_Db = computeGain(levelDb, thresh, ratio, knee, charType, modeType, guardDb);
            targetGR_Db = applyHoldAndRange(targetGR_Db, channel);
            smoothedGR_Db = applyBallistics(targetGR_Db, channel);
        }
        T smoothedGainLinear = decibelsToGain(smoothedGR_Db);

        if (advanceShared)
            autoMakeupEnv_ = smoothedGR_Db + autoMakeupCoeff_ * (autoMakeupEnv_ - smoothedGR_Db);

        T makeup = makeupDb;
        if (modeType == Mode::Downward)
        {
            if (autoMkup == AutoMakeupMode::Adaptive)
                makeup += -autoMakeupEnv_;
            else if (autoMkup == AutoMakeupMode::Static)
                makeup += computeGain(T(0), thresh, ratio, knee, charType,
                                      Mode::Downward, T(0)) * T(-0.5);
        }

        T output = input * smoothedGainLinear;
        if (colorAmt > T(0) && charType == Character::FET)
        {
            // Same 2nd-order FET channel modulation as the block path.
            const T sq = output * output;
            T& dc = fetDcState_[channel];
            dc += fetDcCoeff_ * (sq - dc);
            const T grDepth = std::clamp(-smoothedGR_Db * T(0.1), T(0), T(1));
            output += colorAmt * kFetColorH2 * grDepth * (sq - dc);
        }
        output *= decibelsToGain(makeup);

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
            envFastDb_[ch] = T(0); // 0 dB gain change = neutral
            envSlowDb_[ch] = T(0);
            upwardGuardDb_[ch] = T(-200);
            fbLastOutput_[ch] = T(0);
            fetDcState_[ch] = T(0);
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
        makeupSmooth_.skip();
        mixSmooth_.skip();
        colorSmooth_.skip();
    }

    // =========================================================================
    // Parameter Setters
    // =========================================================================

    /** @brief Sets the compression threshold in dB. */
    void setThreshold(T dB) noexcept { threshold_.store(dB, std::memory_order_relaxed); }

    /** @brief Sets the compression ratio (1.0 = off, >20.0 = limiting). */
    void setRatio(T ratio) noexcept
    {
        ratio_.store(std::max(ratio, T(1)), std::memory_order_relaxed);
        // The feedback attack compensation depends on the loop gain (ratio).
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the attack time in milliseconds.
     *
     * Reads as the t63 time constant of the OBSERVED gain-reduction envelope
     * in dB. Character models clamp it to their hardware's physical range:
     * Opto floors it at 10 ms (the T4 cell's published attack time), FET
     * clamps it to 0.02-0.8 ms. In FeedBack operation the loop multiplies the
     * raw ballistics speed by its gain; the coefficient is re-derived so the
     * observed t63 stays on the knob (the 1176's 20-800 us figures are
     * measured results, not raw RC values).
     */
    void setAttack(T ms) noexcept
    {
        attackMs_.store(std::max(ms, T(0.01)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the release time in milliseconds (clamped to >= 1 ms: below
     *  that the envelope stops smoothing at all and the gain path degenerates
     *  into a waveshaper).
     *
     * Reads as the t63 time constant of the dB envelope for Clean/Varimu.
     * For Opto it is the time to ~50% recovery (the slow memory tail extends
     * up to ~35x longer); for FET it clamps to the 1176 range 50-1100 ms.
     */
    void setRelease(T ms) noexcept
    {
        releaseMs_.store(std::max(ms, T(1)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets knee width in dB (0 = hard knee, >0 = soft knee). */
    void setKnee(T dB) noexcept { kneeWidth_.store(std::max(dB, T(0)), std::memory_order_relaxed); }

    /** @brief Sets manual static makeup gain in dB. */
    void setMakeupGain(T dB) noexcept { makeupGain_.store(dB, std::memory_order_relaxed); }

    /**
     * @brief Selects the automatic makeup behavior (default: Off).
     *
     * Static is the textbook auto makeup: a constant, program-independent
     * offset of half the static gain reduction of a full-scale signal.
     * Adaptive is loudness matching: it tracks the smoothed gain reduction
     * (~300 ms) and compensates it in full, so the average output level
     * stays matched to the input and quiet passages are lifted accordingly
     * (macro-dynamics are reduced). For a fixed offset use setMakeupGain().
     */
    void setAutoMakeup(AutoMakeupMode mode) noexcept
    {
        autoMakeupMode_.store(mode, std::memory_order_relaxed);
    }

    /** @brief Convenience overload: true selects Adaptive, false turns auto makeup off. */
    void setAutoMakeup(bool on) noexcept
    {
        setAutoMakeup(on ? AutoMakeupMode::Adaptive : AutoMakeupMode::Off);
    }

    /** @brief Sets processing mode (Downward or Upward compression). */
    void setMode(Mode mode) noexcept
    {
        mode_.store(mode, std::memory_order_relaxed);
        // Feedback attack compensation only applies to Downward loops.
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets stereo linking amount (0.0 = unlinked dual mono, 1.0 = fully linked). */
    void setStereoLink(T amount) noexcept { stereoLink_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed); }

    /** @brief Sets dry/wet balance for parallel (New York) compression (1.0 = fully wet). */
    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    /** 
     * @brief Sets lookahead time in ms (0 = off, max = 10ms). 
     * @warning Automatically disabled internally if topology is set to FeedBack.
     */
    void setLookahead(T ms) noexcept
    {
        lookaheadMs_.store(std::clamp(ms, T(0), T(10)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

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
        timeConstantsDirty_.store(true, std::memory_order_release);
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
    void setTopology(Topology topo) noexcept
    {
        topology_.store(topo, std::memory_order_relaxed);
        // Feedback loops re-derive the attack coefficient (loop speed-up).
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Changes ballistics and envelope behavior character. */
    void setCharacter(Character type) noexcept
    {
        character_.store(type, std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the amount of the character's harmonic signature (0 to 1).
     *
     * Currently shapes the FET character: 2nd-order channel modulation of the
     * 1176's gain-reduction FET (the drain-source resistance bends with the
     * signal across it), scaled with the active gain reduction and AC-coupled
     * like the hardware's output stage. At 1.0 it is calibrated to the
     * published 1176 THD spec (< 0.5% while limiting, measured at -6 dBFS
     * program level); it only distorts while the FET conducts, so no gain
     * reduction means no added colour. Other characters ignore it for now.
     * Default 0 (clean). The squared term doubles the signal bandwidth: on
     * synthetic full-scale HF material wrap the compressor in an oversampled
     * section (see docs/cookbook.md) if the folded 2nd harmonic matters.
     */
    void setCharacterColor(T amount) noexcept
    {
        characterColor_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Returns the character colour amount (see setCharacterColor). */
    [[nodiscard]] T getCharacterColor() const noexcept
    {
        return characterColor_.load(std::memory_order_relaxed);
    }

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

    /**
     * @brief Returns total processing latency in samples.
     *
     * Lookahead plus the Hilbert detector's alignment delay when that
     * detector is active. Feedback operation (Topology::FeedBack, or the
     * FET character, which always detects in feedback) disables both, so
     * latency is 0 there. The plugin layer re-reads this after parameter
     * changes and re-notifies the host when it moves.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        const bool feedback =
            topology_.load(std::memory_order_relaxed) == Topology::FeedBack
            || character_.load(std::memory_order_relaxed) == Character::FET;
        if (feedback) return 0;
        const int hilbertComp =
            (detectorType_.load(std::memory_order_relaxed) == DetectorType::Hilbert)
            ? Hilbert<T>::getLatencySamples() : 0;
        // Derive the lookahead from the published parameter rather than the
        // audio-thread cache: hosts re-read the latency right after a setter,
        // before the next block has consumed the coefficient-update flag.
        const int lookNow = static_cast<int>(static_cast<T>(sampleRate_) * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        return lookNow + hilbertComp;
    }


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
        const auto amMode = autoMakeupMode_.load(std::memory_order_relaxed);
        w.write("autoMakeup", amMode != AutoMakeupMode::Off); // legacy bool key
        w.write("autoMakeupMode", static_cast<int32_t>(amMode));
        w.write("stereoLink", stereoLink_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("lookahead", lookaheadMs_.load(std::memory_order_relaxed));
        w.write("hold", holdMs_.load(std::memory_order_relaxed));
        w.write("range", rangeDb_.load(std::memory_order_relaxed));
        w.write("detector", static_cast<int32_t>(detectorType_.load(std::memory_order_relaxed)));
        w.write("topology", static_cast<int32_t>(topology_.load(std::memory_order_relaxed)));
        w.write("character", static_cast<int32_t>(character_.load(std::memory_order_relaxed)));
        w.write("characterColor", characterColor_.load(std::memory_order_relaxed));
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
        // The mode key wins; legacy blobs only carry the bool (true = Adaptive).
        const int amLegacy = r.read("autoMakeup", false) ? 2 : 0;
        setAutoMakeup(static_cast<AutoMakeupMode>(
            std::clamp(r.read("autoMakeupMode", amLegacy), 0, 2)));
        setStereoLink(static_cast<T>(r.read("stereoLink", 1.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setLookahead(static_cast<T>(r.read("lookahead", 0.0f)));
        setHoldTime(static_cast<T>(r.read("hold", 0.0f)));
        setRange(static_cast<T>(r.read("range", 100.0f)));
        setDetector(static_cast<DetectorType>(r.read("detector", 0)));
        setTopology(static_cast<Topology>(r.read("topology", 0)));
        setCharacter(static_cast<Character>(r.read("character", 0)));
        setCharacterColor(static_cast<T>(r.read("characterColor", 0.0f)));
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
        makeupSmooth_.setTargetValue(makeupGain_.load(std::memory_order_relaxed));
        mixSmooth_.setTargetValue(mix_.load(std::memory_order_relaxed));
        colorSmooth_.setTargetValue(characterColor_.load(std::memory_order_relaxed));

        if (timeConstantsDirty_.exchange(false, std::memory_order_acquire))
        {
            T fs = static_cast<T>(sampleRate_);
            if (fs > T(0)) updateTimeConstants(fs);
        }
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
        auto autoMkup  = autoMakeupMode_.load(std::memory_order_relaxed);
        T sLink        = stereoLink_.load(std::memory_order_relaxed);

        // The FET character is a feedback design like the 1176 it models: it
        // always detects on the compressed output with the hardware's peak
        // rectifier, whatever Topology and Detector say.
        const Topology topoEff = (charType == Character::FET) ? Topology::FeedBack : topo;
        const DetectorType detTypeEff = (charType == Character::FET) ? DetectorType::Peak : detType;

        // Downward feedback with the peak detector resolves the loop
        // semi-implicitly (the level the loop will read is a known function
        // of the gain): stable at any attack, and the observed static curve
        // lands exactly on the requested ratio/knee. Detectors with memory
        // (RMS/TruePeak/Hilbert/Split) keep the explicit one-sample loop.
        const bool fbImplicit = topoEff == Topology::FeedBack
                             && modeType == Mode::Downward
                             && detTypeEff == DetectorType::Peak;

        // The Hilbert detector reports the envelope kCenter samples late;
        // delaying the audio by the same amount re-aligns gain and signal.
        const int hilbertComp = (detTypeEff == DetectorType::Hilbert)
                              ? Hilbert<T>::getLatencySamples() : 0;

        // Lookahead (and the Hilbert alignment delay) break causal logic in
        // Feedback mode, so both are strictly disabled there.
        int activeLookahead = (topoEff == Topology::FeedBack) ? 0
                            : lookaheadSamples_ + hilbertComp;

        for (int i = 0; i < nS; ++i)
        {
            T thresh   = thresholdSmooth_.getNextValue();
            T ratio    = ratioSmooth_.getNextValue();
            T knee     = kneeSmooth_.getNextValue();
            T mkupGain = makeupSmooth_.getNextValue();
            T mixVal   = mixSmooth_.getNextValue();
            T colorAmt = colorSmooth_.getNextValue();
            T linkedLevel = T(-200);

            // Static auto makeup follows the smoothed curve parameters, so it
            // stays click-free through threshold/ratio/knee automation.
            if (autoMkup == AutoMakeupMode::Static && modeType == Mode::Downward)
                mkupGain += computeGain(T(0), thresh, ratio, knee, charType,
                                        Mode::Downward, T(0)) * T(-0.5);

            // 1. Detection Path (Per-Channel)
            for (int ch = 0; ch < nCh; ++ch)
            {
                // Fallback to internal channel if sidechain buffer lacks channels
                T sample = (scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))[i]
                                      : audio.getChannel(ch)[i];

                // The sidechain filter must process whatever the detector
                // consumes. In FeedBack that is the compressed output (the
                // external key is ignored): the previous sample on the
                // explicit path, or the channel's own current input on the
                // semi-implicit path (the loop equation supplies the gain).
                T detectorIn;
                if (topoEff == Topology::FeedBack)
                    detectorIn = fbImplicit ? audio.getChannel(ch)[i] : fbLastOutput_[ch];
                else
                    detectorIn = sample;
                if (scHpf) detectorIn = applySidechainHPF(detectorIn, ch);

                T levelDb = detectLevel(detectorIn, ch, detTypeEff);

                channelLevelDb_[ch] = levelDb;
                if (levelDb > linkedLevel) linkedLevel = levelDb;
            }

            // 2. Stereo Linking & Gain Application
            T blockGR = T(0);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T chLevel = channelLevelDb_[ch];
                T inputDb = chLevel + sLink * (linkedLevel - chLevel);

                // Sustained-level guard envelope for Upward mode: instant
                // rise, constant 60 dB/s fall (peak hold with linear decay).
                T guardDb = inputDb;
                if (modeType == Mode::Upward)
                {
                    T& g = upwardGuardDb_[ch];
                    g = std::max(inputDb, g - upwardGuardDecay_);
                    guardDb = g;
                }

                // 3. Static curve + character ballistics (log domain)
                T smoothedGR_Db;
                if (fbImplicit)
                {
                    smoothedGR_Db = applyBallisticsFeedbackImplicit(inputDb, ch, charType,
                                                                    thresh, ratio, knee);
                }
                else if (topoEff == Topology::FeedBack && modeType == Mode::Downward)
                {
                    // Explicit loop (detector with memory): calibrated element
                    // law plus a per-sample stability floor on the ballistics.
                    const auto law = computeGainFeedback(inputDb, thresh, ratio, knee, charType);
                    const T targetGR_Db = applyHoldAndRange(law.target, ch);
                    smoothedGR_Db = applyBallistics(targetGR_Db, ch, law.slope);
                }
                else
                {
                    T targetGR_Db = computeGain(inputDb, thresh, ratio, knee, charType, modeType, guardDb);
                    targetGR_Db = applyHoldAndRange(targetGR_Db, ch);
                    smoothedGR_Db = applyBallistics(targetGR_Db, ch);
                }
                T smoothedGainLinear = decibelsToGain(smoothedGR_Db);

                // 4. Makeup & Mix
                T makeup = mkupGain;
                if (autoMkup == AutoMakeupMode::Adaptive && modeType == Mode::Downward)
                    makeup += -autoMakeupEnv_;

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

                T wet = input * smoothedGainLinear;
                if (colorAmt > T(0) && charType == Character::FET)
                {
                    // 2nd-order FET channel modulation: the drain-source
                    // resistance bends with the signal across it, so colour
                    // only appears while the FET conducts (gain reduction
                    // active); the squared term is AC-coupled like the
                    // hardware's output transformer. Applied pre-makeup: the
                    // line amp after the element is clean.
                    const T sq = wet * wet;
                    T& dc = fetDcState_[ch];
                    dc += fetDcCoeff_ * (sq - dc);
                    const T grDepth = std::clamp(-smoothedGR_Db * T(0.1), T(0), T(1));
                    wet += colorAmt * kFetColorH2 * grDepth * (sq - dc);
                }
                wet *= decibelsToGain(makeup);
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
        T relMs = std::max(releaseMs_.load(std::memory_order_relaxed), T(1));
        autoMakeupCoeff_ = std::exp(T(-1) / (fs * T(0.3)));

        // One-pole coefficient for a time constant given in milliseconds.
        auto tc = [fs](T ms) { return std::exp(T(-1) / (fs * ms / T(1000))); };

        // Character ballistics: time constants of the dB-domain envelopes.
        switch (character_.load(std::memory_order_relaxed))
        {
            case Character::Clean:
            case Character::Varimu:
                charAttCoeff_     = tc(attMs);
                charRelCoeff_     = tc(relMs);
                charSlowRelCoeff_ = charRelCoeff_;
                charChargeCoeff_  = charAttCoeff_;
                charFastWeight_   = T(1); // single envelope
                break;

            case Character::Opto:
                // T4 cell: the release knob is the ~50% recovery time. With a
                // 0.6/0.4 fast/slow split, tau_fast = 0.61 x release puts the
                // half-recovery point on the knob value while the memory tail
                // decays ~35x slower, capped at the physical 5 s of the cell
                // (LA-2A spec: 50% in ~0.06 s, complete in 0.5-5 s). The slow
                // stage charges over ~4 release times (capped at 2 s), so only
                // sustained compression builds the long tail. The 10 ms attack
                // floor is the cell's published attack time.
                charAttCoeff_     = tc(std::max(attMs, T(10)));
                charRelCoeff_     = tc(relMs * T(0.61));
                charSlowRelCoeff_ = tc(std::min(relMs * T(25), T(5000)));
                charChargeCoeff_  = tc(std::clamp(relMs * T(4), T(100), T(2000)));
                charFastWeight_   = T(0.6);
                break;

            case Character::FET:
            {
                // 1176: knob ranges are 20-800 us attack and 50-1100 ms
                // release. With a 0.75/0.25 split the compound release passes
                // ~t63 at the knob value, and a 150 ms history charge gives
                // the program-dependent tail.
                const T rel = std::clamp(relMs, T(50), T(1100));
                charAttCoeff_     = tc(std::clamp(attMs, T(0.02), T(0.8)));
                charRelCoeff_     = tc(rel * T(0.7));
                charSlowRelCoeff_ = tc(rel * T(2.5));
                charChargeCoeff_  = tc(T(150));
                charFastWeight_   = T(0.75);
                break;
            }
        }

        // Downward feedback closes its loop around the attack stage, which
        // multiplies the raw ballistics speed by (1 + loop gain). Re-derive
        // the coefficient from rho = coeff/(1 + w beta A) so the OBSERVED
        // attack t63 stays on the knob, exactly like the hardware panels
        // (the 1176's 20-800 us figures are measured results, not RC values).
        // A is the linear-region loop gain (R - 1); Varimu uses its base
        // ratio (the progressive part is level-dependent).
        {
            const auto charNow = character_.load(std::memory_order_relaxed);
            const bool fbLoop = (charNow == Character::FET
                              || topology_.load(std::memory_order_relaxed) == Topology::FeedBack)
                             && mode_.load(std::memory_order_relaxed) == Mode::Downward;
            if (fbLoop)
            {
                const T loopGain = charFastWeight_
                                 * (std::max(ratio_.load(std::memory_order_relaxed), T(1)) - T(1));
                const T rho = charAttCoeff_;
                charAttCoeff_ = T(1) - (T(1) - rho) / (T(1) + rho * loopGain);
            }
        }

        // SplitPolarity detector ballistics. The time constants reproduce the
        // original per-sample coefficients (0.6 attack / 0.99 release) at
        // 44.1 kHz, but are now sample-rate invariant.
        splitDetAttCoeff_ = std::exp(T(-1) / (fs * T(44.39e-6)));
        splitDetRelCoeff_ = std::exp(T(-1) / (fs * T(2.2562e-3)));

        // Upward silence-guard envelope: 60 dB/s linear decay.
        upwardGuardDecay_ = T(60) / fs;

        // AC coupling (~10 Hz) of the FET color's 2nd-order term.
        fetDcCoeff_ = T(1) - std::exp(T(-2) * std::numbers::pi_v<T> * T(10) / fs);

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

                T posCoeff = (pos > splitPosEnv_[ch]) ? splitDetAttCoeff_ : splitDetRelCoeff_;
                T negCoeff = (neg > splitNegEnv_[ch]) ? splitDetAttCoeff_ : splitDetRelCoeff_;

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
     * @brief Level-dependent effective ratio of the character (Varimu grows).
     * @param excessDb Input level above the threshold in dB.
     */
    [[nodiscard]] static T effectiveRatioFor(Character charType, T ratio, T excessDb) noexcept
    {
        // Fairchild-style progressive compression: a remote-cutoff tube's
        // effective ratio grows with level above the threshold.
        if (charType == Character::Varimu && excessDb > T(0))
            return ratio * (T(1) + excessDb / T(40));
        return ratio;
    }

    /** @brief Physical knee floor of the character (dB). */
    [[nodiscard]] static T effectiveKneeFor(Character charType, T knee) noexcept
    {
        // Neither a remote-cutoff tube (Varimu) nor a photocell (Opto) can
        // form a hard corner: both transfers bend gradually over their whole
        // operating region, so the knee has a 10 dB physical floor.
        return (charType == Character::Varimu || charType == Character::Opto)
             ? std::max(knee, T(10)) : knee;
    }

    /**
     * @brief Calculates static target gain reduction based on knee and ratio.
     * @param inputDb Detected level in decibels.
     * @param thresh Threshold parameter in dB.
     * @param ratio Ratio parameter.
     * @param knee Soft-knee width parameter in dB.
     * @param charType Applies the character's knee floor and ratio law.
     * @param modeType Upward or Downward processing mode.
     * @return Target gain reduction in Decibels.
     */
    [[nodiscard]] T computeGain(T inputDb, T thresh, T ratio, T knee, Character charType, Mode modeType,
                                T guardDb) const noexcept
    {
        if (modeType == Mode::Upward)
            return computeGainUpward(inputDb, thresh, ratio, knee, guardDb);

        const T effectiveRatio = effectiveRatioFor(charType, ratio, inputDb - thresh);
        const T effectiveKnee = effectiveKneeFor(charType, knee);

        if (effectiveKnee <= T(0))
        {
            // Hard knee
            if (inputDb <= thresh) return T(0);
            return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
        }
        else
        {
            // Soft knee interpolation
            T halfKnee = effectiveKnee / T(2);
            T lower = thresh - halfKnee;
            T upper = thresh + halfKnee;

            if (inputDb <= lower) return T(0);
            if (inputDb >= upper) return (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);

            T x = inputDb - lower;
            return (T(1) - T(1) / effectiveRatio) * x * x / (T(2) * effectiveKnee) * T(-1);
        }
    }

    /**
     * @brief Upward compression curve calculation.
     *
     * The boost fades to zero when guardDb falls 40 to 60 dB below the
     * threshold: without that guard, pauses and the noise floor (which the
     * detector reads at its -100 dB floor) would receive the largest boost
     * of all. guardDb is the peak-held sustained level, NOT the instantaneous
     * one: gating on the instantaneous level would punch boost holes at every
     * zero crossing of the waveform.
     */
    [[nodiscard]] T computeGainUpward(T inputDb, T thresh, T ratio, T knee, T guardDb) const noexcept
    {
        T effectiveRatio = ratio;
        T boost;
        if (knee <= T(0))
        {
            if (inputDb >= thresh) return T(0);
            boost = (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
        }
        else
        {
            T halfKnee = knee / T(2);
            T lower = thresh - halfKnee;
            T upper = thresh + halfKnee;

            if (inputDb >= upper) return T(0);
            if (inputDb <= lower)
            {
                boost = (thresh - inputDb) * (T(1) - T(1) / effectiveRatio);
            }
            else
            {
                T x = upper - inputDb;
                boost = (T(1) - T(1) / effectiveRatio) * x * x / (T(2) * knee);
            }
        }

        const T silenceGuard = std::clamp((guardDb - (thresh - T(60))) / T(20), T(0), T(1));
        return boost * silenceGuard;
    }

    // ---- Feedback law (element transfer solved from the observed curve) ----

    /** @brief Element law result: static target (dB, <= 0) and its |slope|. */
    struct FeedbackLaw
    {
        T target;  ///< Gain change the element commands at this output level.
        T slope;   ///< d|target| / d(level in dB): the incremental loop gain.
    };

    /**
     * @brief Downward element law for feedback detection, per output level.
     *
     * The detector reads the compressed output, so a raw feed-forward law in
     * the loop settles far off the requested curve (its observed ratio can
     * never pass 2:1). This is the closed-form inverse instead:
     * gel(out) = grFF(in(out)), where in(out) inverts out = in + grFF(in).
     * The settled loop then reproduces the user's static curve exactly.
     * Piecewise: silence, quadratic knee (cancellation-free root), and a
     * linear region of slope (R - 1), the classic loop gain that makes a
     * feedback compressor observe ratio R.
     *
     * @param eOut Detected output level relative to the threshold (dB).
     * @param r Effective ratio (>= 1).
     * @param w Effective knee width (dB, >= 0).
     */
    [[nodiscard]] static FeedbackLaw elementLaw(T eOut, T r, T w) noexcept
    {
        const T a = r - T(1);
        if (w <= T(0))
        {
            if (eOut <= T(0)) return { T(0), T(0) };
            return { -a * eOut, a };
        }
        const T halfW = w / T(2);
        if (eOut <= -halfW) return { T(0), T(0) };
        if (eOut >= halfW / r) return { -a * eOut, a };
        // Knee: invert eOut = u - W/2 - s u^2 / (2W) for the input-side
        // excess u in (0, W). s = 1 - 1/R is the feed-forward slope factor;
        // the slope denominator is analytically >= 1/R.
        const T s = a / r;
        const T b = eOut + halfW;
        const T z = std::min(T(2) * s * b / w, T(1));
        const T u = std::min(T(2) * b / (T(1) + std::sqrt(T(1) - z)), w);
        const T su = s * u / w;
        return { -s * u * u / (T(2) * w), su / std::max(T(1) - su, T(1) / r) };
    }

    /**
     * @brief Feedback target for detectors with memory (explicit path).
     *
     * Varimu's level-dependent ratio is a function of the INPUT level, which
     * a feedback detector does not see; two fixed-point passes reconstruct it
     * from the output level (the law is smooth, so this lands well within
     * 0.1 dB of the exact curve).
     *
     * @param outDb Detected output level (dB).
     */
    [[nodiscard]] FeedbackLaw computeGainFeedback(T outDb, T thresh, T ratio, T knee,
                                                  Character charType) const noexcept
    {
        const T eOut = outDb - thresh;
        const T w = effectiveKneeFor(charType, knee);
        T r = std::max(effectiveRatioFor(charType, ratio, eOut), T(1));
        FeedbackLaw law = elementLaw(eOut, r, w);
        if (charType == Character::Varimu)
        {
            for (int pass = 0; pass < 2; ++pass)
            {
                r = std::max(effectiveRatioFor(charType, ratio, eOut - law.target), T(1));
                law = elementLaw(eOut, r, w);
            }
        }
        return law;
    }

    /** @brief Semi-implicit solve result: stepped gain and the law's target. */
    struct FeedbackSolve
    {
        T gain;    ///< Blended gain (dB) after this sample's ballistics step.
        T target;  ///< The element law's static target at the solved level.
    };

    /**
     * @brief Solves one feedback ballistics step against the current input.
     *
     * With the peak detector the level the loop will read is a known function
     * of the gain (out = in + G), so the one-pole step G = k + effB * gel(in + G)
     * is solved simultaneously with it, piece by piece: the element law is
     * linear or quadratic per piece and F(G) = G - k - effB * gel(in + G) is
     * strictly monotonic, so a piece whose solution verifies its own range IS
     * the unique solution. This is the backward-Euler view of the analog loop
     * (a first-order system): unconditionally stable at any attack, unlike
     * the explicit iteration whose one-sample delay rings once the loop gain
     * (R - 1) exceeds the ballistics speed.
     *
     * @param inDb Input level in dB (post link and sidechain filter).
     * @param k Explicit part of the ballistics step (coeff-weighted history).
     * @param effB Implicit weight: (1 - coeff) times the fast-stage blend.
     */
    [[nodiscard]] FeedbackSolve solveFeedbackGain(T inDb, T k, T effB, T thresh,
                                                  T ratio, T knee, Character charType) const noexcept
    {
        const T r = std::max(effectiveRatioFor(charType, ratio, inDb - thresh), T(1));
        const T w = effectiveKneeFor(charType, knee);
        const T halfW = w / T(2);
        const T a = r - T(1);

        // Piece 1: at or below the knee start the law is zero and G = k.
        T e = inDb + k - thresh;
        if (e <= ((w > T(0)) ? -halfW : T(0)))
            return { k, T(0) };

        // Piece 3: linear region, target = -a * e at the solved level. With a
        // hard knee this piece always resolves (e scales by 1/(1 + effB a)).
        const T g3 = (k - effB * a * (inDb - thresh)) / (T(1) + effB * a);
        e = inDb + g3 - thresh;
        if (e >= ((w > T(0)) ? halfW / r : T(0)))
            return { g3, -a * e };

        // Piece 2: quadratic knee. Substituting G = k + effB * gel into the
        // knee relation gives q u^2 - u + (d + W/2) = 0 with
        // q = (1 - effB) s / (2W); the cancellation-free smaller root is used.
        const T s = a / r;
        const T q = (T(1) - effB) * s / (T(2) * w);
        const T b = (inDb - thresh + k) + halfW;
        const T disc = std::max(T(1) - T(4) * q * b, T(0));
        T u = T(2) * b / (T(1) + std::sqrt(disc));
        u = std::clamp(u, T(0), w);
        const T target = -s * u * u / (T(2) * w);
        return { k + effB * target, target };
    }

    /**
     * @brief Feedback ballistics with the loop resolved semi-implicitly.
     *
     * Runs the same branching envelopes as applyBallistics, but the step
     * toward the target is solved simultaneously with the level the detector
     * will read. Hold and range act on the OBSERVED gain after the step (see
     * the inline note), and the memory stage tracks the real static curve.
     *
     * @param inDb Input level in dB (post link and sidechain filter).
     */
    [[nodiscard]] T applyBallisticsFeedbackImplicit(T inDb, int ch, Character charType,
                                                    T thresh, T ratio, T knee) noexcept
    {
        T& fast = envFastDb_[ch];
        T& slow = envSlowDb_[ch];
        const T wFast = charFastWeight_;
        const bool dual = wFast < T(1);

        if (dual)
        {
            // The memory stage tracks the REAL sustained gain reduction (the
            // static curve at the current input), never the element's
            // internal target: the loop gain scales that one by up to
            // (R - 1), and a memory charging toward the amplified signal
            // overshoots, drags the settled blend off the curve and smears
            // the observed attack. Advancing it first keeps the solve below
            // consistent with the state it blends against.
            const T range = rangeDb_.load(std::memory_order_relaxed);
            const T slowTarget = std::clamp(
                computeGain(inDb, thresh, ratio, knee, charType, Mode::Downward, T(0)),
                -range, range);
            const bool charging = slowTarget < slow;
            slow = slowTarget + (charging ? charChargeCoeff_ : charSlowRelCoeff_) * (slow - slowTarget);
            if (std::abs(slow) < T(1e-4)) slow = T(0);
        }

        // Branch on the direction the loop is about to move: attack while
        // the reduction deepens at the previous gain, release otherwise.
        const T shownPrev = dual ? wFast * fast + (T(1) - wFast) * slow : fast;
        const T probe = computeGainFeedback(inDb + shownPrev, thresh, ratio, knee, charType).target;
        const bool engaging = probe < fast;
        const T coeff = engaging ? charAttCoeff_ : charRelCoeff_;
        const T beta = T(1) - coeff;

        const T k = dual ? wFast * coeff * fast + (T(1) - wFast) * slow : coeff * fast;
        const T effB = dual ? wFast * beta : beta;
        const auto sol = solveFeedbackGain(inDb, k, effB, thresh, ratio, knee, charType);

        fast = sol.target + coeff * (fast - sol.target);
        if (std::abs(fast) < T(1e-4)) fast = T(0);
        T shown = dual ? wFast * fast + (T(1) - wFast) * slow : fast;

        // Hold and range act on the OBSERVED gain: the element's internal
        // target is loop-amplified by up to (R - 1) during transients, so
        // clamping THAT to the user's range would throttle the attack, and
        // holding it would compare mismatched domains. When they override,
        // the fast stage is rewritten so the blend lands on the bound.
        const T bounded = applyHoldAndRange(shown, ch);
        if (bounded != shown)
        {
            shown = bounded;
            fast = dual ? (shown - (T(1) - wFast) * slow) / wFast : shown;
        }
        return shown;
    }

    // ---- Ballistics (log domain) ----

    /**
     * @brief Smooths the static gain target with the active character's ballistics.
     *
     * Operates on the gain change in dB. Clean/Varimu run a smoothed branching
     * one-pole (attack while the gain moves away from unity, release while it
     * recovers). Opto/FET blend a fast envelope with a slow one that charges
     * with compression history, producing the two-stage program-dependent
     * release of the modeled hardware: brief peaks release fast because the
     * slow stage never charged, sustained compression leaves a long tail.
     *
     * @param targetGrDb Static gain change target in dB (negative = reduction).
     * @param ch Channel index.
     * @param loopSlope Incremental loop gain when driven from a feedback
     *        detector with memory (explicit loop): both coefficients get a
     *        stability floor, which caps the per-sample loop advance at 0.5
     *        and keeps the one-sample-delay iteration from ringing at extreme
     *        ratio/time combinations. 0 = no loop (default).
     * @return The smoothed gain change in dB to apply.
     */
    [[nodiscard]] T applyBallistics(T targetGrDb, int ch, T loopSlope = T(0)) noexcept
    {
        // Level-adaptive release for the SplitPolarity detector: full-scale
        // output halves the release time constant. Halving tau squares the
        // one-pole coefficient, so blending between the two valid endpoints
        // keeps the modulation on the exponential curve.
        T attCoeff = charAttCoeff_;
        T relCoeff = charRelCoeff_;
        if (detectorType_.load(std::memory_order_relaxed) == DetectorType::SplitPolarity)
        {
            const T outputLevel = std::min(std::abs(fbLastOutput_[ch]), T(1));
            relCoeff += outputLevel * (relCoeff * relCoeff - relCoeff);
        }
        if (loopSlope > T(0))
        {
            // Cap the per-sample loop advance at 0.5 (half-deadbeat): the
            // one-sample delay then converges monotonically instead of
            // sustaining a marginal dance on the detector's residual ripple.
            const T floorCoeff = T(1) - T(0.5) / (T(1) + loopSlope);
            attCoeff = std::max(attCoeff, floorCoeff);
            relCoeff = std::max(relCoeff, floorCoeff);
        }

        // Attack acts while the gain FALLS (louder signal: deeper reduction,
        // or less upward boost); release acts while it recovers upward. A
        // magnitude comparison would invert the two in Upward mode and make
        // the envelope chase the huge boosts of the waveform's zero crossings.
        T& fast = envFastDb_[ch];
        const bool engaging = targetGrDb < fast;
        fast = targetGrDb + (engaging ? attCoeff : relCoeff) * (fast - targetGrDb);
        if (std::abs(fast) < T(1e-4)) fast = T(0); // kill the asymptotic dB tail

        if (charFastWeight_ >= T(1)) // Clean / Varimu: single envelope
            return fast;

        // Opto / FET memory stage: charges toward the target over the history
        // time constant and discharges with the slow release, so its level
        // encodes how long (and how deep) the compressor has been working.
        T& slow = envSlowDb_[ch];
        const bool charging = targetGrDb < slow;
        slow = targetGrDb + (charging ? charChargeCoeff_ : charSlowRelCoeff_) * (slow - targetGrDb);
        if (std::abs(slow) < T(1e-4)) slow = T(0);

        return charFastWeight_ * fast + (T(1) - charFastWeight_) * slow;
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
    std::atomic<T> lookaheadMs_ { T(0) };    ///< Lookahead latency target in ms.
    std::atomic<T> characterColor_ { T(0) }; ///< Character harmonic amount (0 to 1).
    std::atomic<AutoMakeupMode> autoMakeupMode_ { AutoMakeupMode::Off }; ///< Auto-makeup behavior.

    std::atomic<DetectorType> detectorType_ { DetectorType::Peak }; ///< Selected detection method.
    std::atomic<Topology> topology_ { Topology::FeedForward };      ///< Selected topology.
    std::atomic<Character> character_ { Character::Clean };         ///< Selected ballistics.
    std::atomic<Mode> mode_ { Mode::Downward };                     ///< Selected compression mode.

    // Internal DSP Coefficients & State
    T autoMakeupCoeff_ = T(0.9995);     ///< Auto-makeup tracking factor.
    T autoMakeupEnv_ = T(0);            ///< Smoothed internal auto-makeup envelope.

    // Character ballistics coefficients (dB-domain one-poles, see
    // updateTimeConstants). charFastWeight_ == 1 selects the single-envelope
    // path (Clean/Varimu); Opto/FET blend fast and slow memory envelopes.
    std::atomic<bool> timeConstantsDirty_ { true }; ///< Coefficients need a recompute.
    T charAttCoeff_     = T(0); ///< Attack coefficient of the fast envelope.
    T charRelCoeff_     = T(0); ///< Release coefficient (fast stage).
    T charSlowRelCoeff_ = T(0); ///< Release coefficient of the memory stage.
    T charChargeCoeff_  = T(0); ///< History charge coefficient of the memory stage.
    T charFastWeight_   = T(1); ///< Fast/slow blend (1 = single envelope).

    SmoothedValue<T> thresholdSmooth_;  ///< De-zippered threshold.
    SmoothedValue<T> ratioSmooth_;      ///< De-zippered ratio.
    SmoothedValue<T> kneeSmooth_;       ///< De-zippered knee.
    SmoothedValue<T> makeupSmooth_;     ///< De-zippered makeup gain (dB).
    SmoothedValue<T> mixSmooth_;        ///< De-zippered parallel mix.
    SmoothedValue<T> colorSmooth_;      ///< De-zippered character colour amount.

    // FET colour: 2nd-order term gain at full colour, calibrated so limiting
    // at -6 dBFS program level measures within the 1176's published THD spec
    // (< 0.5%); the one-pole DC estimate AC-couples the squared term.
    static constexpr T kFetColorH2 = T(0.028);
    T fetDcCoeff_ = T(0);                        ///< ~10 Hz DC-tracking coefficient.
    std::array<T, kMaxChannels> fetDcState_ {};  ///< Per-channel DC estimate of wet^2.

    std::array<T, kMaxChannels> envFastDb_ {};      ///< Fast gain envelope per channel (dB).
    std::array<T, kMaxChannels> envSlowDb_ {};      ///< Slow memory envelope per channel (dB).
    std::array<T, kMaxChannels> upwardGuardDb_ {};  ///< Peak-held sustained level (Upward guard).
    T upwardGuardDecay_ = T(0.00125);               ///< Guard decay per sample (60 dB/s).
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
    T splitDetAttCoeff_ = T(0.6);  ///< Detector attack coefficient (sample-rate derived).
    T splitDetRelCoeff_ = T(0.99); ///< Detector release coefficient (sample-rate derived).

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
