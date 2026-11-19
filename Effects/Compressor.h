// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Compressor.h
 * @brief Modular dynamic range compressor with analog-modeled ballistics and Hilbert detection.
 *
 * Professional compressor with interchangeable detector, topology, and
 * ballistics character. Gain smoothing runs in the log (dB) domain: Clean
 * and Varimu use a branching one-pole (Clean can opt into decoupled
 * ballistics in feed-forward: a peak hold followed by a smoothing stage,
 * after the smooth decoupled detector of Giannoulis, Massberg and Reiss,
 * JAES 2012), and the analog characters (Opto, FET)
 * run a dual-envelope model with compression-history memory that
 * reproduces the two-stage program-dependent release of the hardware they
 * are named after.
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
 * Threading: prepare()/reset() from a setup thread only (never concurrent
 * with audio). processBlock()/processSample() from the audio thread that
 * owns the stream. All setters and getState() are safe from any thread
 * (atomic publication; the audio thread consumes changes at block/sample
 * boundaries). Invalid setter values (NaN/Inf, wild enums) are ignored or
 * clamped, so no control-thread input can poison the audio path. A
 * non-finite input sample reads as silence in the detector and cannot latch
 * in its state; it still passes to the output for that sample.
 * getGainReductionDb()/getLatency() are metering/host reads (relaxed).
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, SmoothedValue.h,
 *               RingBuffer.h, DenormalGuard.h, Hilbert.h,
 *               TruePeakDetector.h, StateBlob.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/SmoothedValue.h"
#include "../Core/RingBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Hilbert.h"
#include "../Core/TruePeakDetector.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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
 * @note The gain path runs at the host rate on purpose. Its gain-modulation
 *       aliasing grows with speed and depth: with the Hilbert or RMS detector
 *       it stays near -94 dBc, the Peak detector at a 0.1 ms attack reaches
 *       -53 dBc, and the FET character at its 20 us minimum attack measures
 *       -55 dBc (ratio 4, 6 dB over the threshold) to -41 dBc (ratio 20,
 *       30 dB over), worst on tones near 18 kHz. Products that need less can
 *       wrap this compressor in an oversampled section like any other stage
 *       with Core/Oversampling: one resampler for the whole section instead
 *       of one per module (see docs/cookbook.md, "Oversampling a nonlinear
 *       section"). setPrecision(High) lowers those lines by 20 to 40 dB
 *       at the host rate, working out only the control signal between the
 *       samples (the Peak detector at a 0.1 ms attack then reaches -66 dBc
 *       at 18 kHz and the FET character -66 dBc, at 7 samples of latency).
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
        SplitPolarity,  ///< Separate envelopes for the positive and negative half-waves.
        Hilbert         ///< Analytic-signal magnitude from a zero-latency allpass pair
                        ///< (HilbertIIR): a ripple-free envelope on sustained tones from
                        ///< 20 Hz up, so the static curve holds exactly and low tones are
                        ///< not modulated at their own frequency (THD+N of a 30 Hz tone
                        ///< 14 dB into 4:1 at 5/100 ms: -96 dB, against -35 dB with Peak).
                        ///< The larger of that magnitude and the rectified input is
                        ///< used, so an onset is caught as fast as with Peak. Adds no
                        ///< latency.
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
        Clean,    ///< Modern transparent design: a branching one-pole on the gain
                  ///< target. In feed-forward, setCleanBallistics(Decoupled) selects an
                  ///< instant peak hold that releases, then a smoothing stage, so the
                  ///< gain does not ride the waveform of a tone (THD+N of a 1 kHz tone
                  ///< 14 dB into 4:1 at 5/100 ms: -102 dB, against -65.5 dB for the
                  ///< one-pole); attack and release t63 stay on the knobs in both.

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

    /** @brief Ballistics of the Clean character in feed-forward (see setCleanBallistics). */
    enum class CleanBallistics
    {
        Branching, ///< A single branching one-pole on the gain target (default):
                   ///< release is exponential from its first sample, the gain
                   ///< rides the waveform of low-frequency tones.
        Decoupled  ///< Peak hold plus smoothing stage (opt-in): low gain-riding
                   ///< distortion, exact peak-referenced level on tones, more
                   ///< reduction on dense programme (see setCleanBallistics).
    };

    /** @brief Automatic makeup-gain behavior (Static and Adaptive apply in Downward mode). */
    enum class AutoMakeupMode
    {
        Off,      ///< Manual makeup only (setMakeupGain).
        Static,   ///< Textbook auto makeup (Giannoulis, Massberg & Reiss, JAES 2012):
                  ///< a constant offset equal to half the static gain reduction of a
                  ///< 0 dBFS signal, derived from the (smoothed) threshold/ratio/knee.
                  ///< Being constant it never pumps and follows curve automation
                  ///< without clicks - and, by the same token, it lifts material that
                  ///< never reaches the threshold (+7.5 dB at -20 dB / 4:1 / hard knee,
                  ///< even on a -40 dBFS passage). That is the defined behaviour, not a
                  ///< side effect: to add makeup only where gain is actually reduced,
                  ///< choose Adaptive.
        Adaptive, ///< Loudness matching: tracks the smoothed gain reduction (~300 ms)
                  ///< and compensates it in full, keeping the average output level
                  ///< matched to the input (quiet passages are lifted).
        Analytic  ///< Worked out, not measured: the gain the ballistics settle on over dense
                  ///< programme (the levels of a Gaussian signal as the selected detector
                  ///< reads them, weighed by attack and release as the follower weighs
                  ///< them), taken back in full. A constant, so it never pumps, and it
                  ///< follows the curve, the times, the character, the detector and the
                  ///< amount. The programme is taken at -18 dBFS RMS; with
                  ///< setAutoThreshold() the threshold counts from the programme's own
                  ///< peak and the result holds at any level. Both modes (in Upward it
                  ///< takes the boost back). See setAutoMakeup().
    };

    /** @brief How finely the control signal is computed (see setPrecision()). */
    enum class Precision
    {
        Standard, ///< One value of the gain per sample (default).
        High      ///< The control signal is worked out between the samples too: the
                  ///< detectors that rectify read the key at four points per sample and
                  ///< the ballistics take eight steps per sample. See setPrecision().
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Allocates buffers and initializes internal DSP state.
     *
     * An invalid spec (non-positive/NaN sample rate or no channels) is
     * ignored: the previous prepared state, if any, stays intact.
     *
     * @warning Must not be called from the audio thread. Allocates memory.
     * @param spec Audio environment specification (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
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
        amountSmooth_.prepare(sampleRate_, 30.0);
        amountSmooth_.reset(amount_.load(std::memory_order_relaxed));
        analyticSmooth_.prepare(sampleRate_, 30.0);
        makeupKey_.valid = false;
        analyticSmooth_.reset(analyticTarget());

        // The detector's points between samples at Precision::High: a
        // Kaiser-windowed sinc per quarter of a sample.
        for (int p = 0; p < kDetectorPoints - 1; ++p)
            detail::designSincPhase<kInterpTaps>(interpPhase_[p], kInterpBeta,
                                                 static_cast<double>(p + 1) / kDetectorPoints);

        // Pre-allocate and initialize per-channel instances. Capacity covers
        // the 10 ms user lookahead. Every channel slot gets a delay line, not just spec.numChannels:
        // a view wider than the spec would otherwise read an unprepared ring
        // (silence) on the extra channels whenever lookahead is active.
        int maxLaSamples = static_cast<int>(sampleRate_ * 0.01) + 1 + kDetectorDelay;
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            lookaheadBuffers_[ch].prepare(maxLaSamples);
            gainDelay_[ch].prepare(maxLaSamples);
            hilbertDetectors_[ch].reset();
        }
        
        lookaheadSamples_ = static_cast<int>(fs * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        latencySamples_ = std::max(lookaheadSamples_, static_cast<int>(fs * std::clamp(
            maxLookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000)));

        updateHpfCoefficients();
        scHpfDirty_.store(false, std::memory_order_relaxed);

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
     * @note No DenormalGuard is installed on this per-sample hot path (installing
     *       and tearing down the FTZ/DAZ guard every sample would dominate its
     *       cost); per-sample callers are expected to guard their own processing
     *       loop with a single DenormalGuard, exactly as the block path does once
     *       per block. This matches Limiter/NoiseGate processSample().
     * @note getLatency() reports the processBlock() path (its lookahead).
     *       This per-sample path BYPASSES the lookahead delay line,
     *       so it adds no lookahead latency regardless of setLookahead().
     *
     * @note This path runs at Precision::Standard whatever setPrecision()
     *       says: the finer control signal waits for a detector that reads
     *       late, and there is no delay line here for the audio to wait in.
     *
     * @note Shared state (parameter smoothers, auto-makeup envelope) advances
     *       once per sample FRAME and is driven by channel 0. Multi-channel
     *       per-sample workflows must therefore include channel 0; other
     *       channels read the current smoothed values without advancing them.
     *
     * @param input Audio sample to process. Returned unchanged (with no state
     *        touched) before prepare() or for a channel outside [0, 16).
     * @param channel Index of the audio channel (used for state tracking).
     * @return Compressed output sample.
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        // Release-safe guards: a wild channel would index every per-channel
        // state array out of bounds (a negative one is a giant size_t).
        if (channel < 0 || channel >= kMaxChannels) return input;
        if (!(sampleRate_ > 0)) return input;

        // Sync atomic parameters into the smoothers here too: processBlock does this
        // per block, but a processSample-only workflow would otherwise never see
        // setThreshold()/setRatio()/setKnee() changes (the smoothers stayed frozen).
        thresholdSmooth_.setTargetValue(threshold_.load(std::memory_order_relaxed));
        ratioSmooth_.setTargetValue(std::max(ratio_.load(std::memory_order_relaxed), T(1)));
        kneeSmooth_.setTargetValue(std::max(kneeWidth_.load(std::memory_order_relaxed), T(0)));
        makeupSmooth_.setTargetValue(makeupGain_.load(std::memory_order_relaxed));
        colorSmooth_.setTargetValue(characterColor_.load(std::memory_order_relaxed));
        amountSmooth_.setTargetValue(amount_.load(std::memory_order_relaxed));

        // Channel 0 advances the shared smoothers; advancing on every call
        // would scale their time constants with the caller's channel count.
        const bool advanceShared = (channel == 0);
        T thresh, ratio, knee, makeupDb, colorAmt, amount;
        if (advanceShared)
        {
            thresh   = thresholdSmooth_.getNextValue();
            ratio    = ratioSmooth_.getNextValue();
            knee     = kneeSmooth_.getNextValue();
            makeupDb = makeupSmooth_.getNextValue();
            colorAmt = colorSmooth_.getNextValue();
            amount   = amountSmooth_.getNextValue();
        }
        else
        {
            thresh   = thresholdSmooth_.getCurrentValue();
            ratio    = ratioSmooth_.getCurrentValue();
            knee     = kneeSmooth_.getCurrentValue();
            makeupDb = makeupSmooth_.getCurrentValue();
            colorAmt = colorSmooth_.getCurrentValue();
            amount   = amountSmooth_.getCurrentValue();
        }

        auto detType   = detectorType_.load(std::memory_order_relaxed);
        auto topo      = topology_.load(std::memory_order_relaxed);
        auto charType  = character_.load(std::memory_order_relaxed);
        auto modeType  = mode_.load(std::memory_order_relaxed);
        bool scHpf     = scHpfEnabled_.load(std::memory_order_relaxed);
        auto autoMkup  = autoMakeupMode_.load(std::memory_order_relaxed);

        // Cheap relaxed pre-check keeps the per-sample cost to one load when
        // nothing changed; the exchange claims the flag only when it is set.
        if (timeConstantsDirty_.load(std::memory_order_relaxed)
            && timeConstantsDirty_.exchange(false, std::memory_order_acquire))
        {
            updateTimeConstants(static_cast<T>(sampleRate_));
        }
        if (channel == 0
            && rmsWindowDirty_.load(std::memory_order_relaxed)
            && rmsWindowDirty_.exchange(false, std::memory_order_acquire))
        {
            rmsWindowMs_ = rmsWindowMsAtomic_.load(std::memory_order_relaxed);
            updateRmsWindow();
        }

        // The FET character is a feedback design like the 1176 it models: it
        // always detects on the compressed output with the hardware's peak
        // rectifier, whatever Topology and Detector say.
        const Topology topoEff = (charType == Character::FET) ? Topology::FeedBack : topo;
        const DetectorType detTypeEff = (charType == Character::FET) ? DetectorType::Peak : detType;
        const bool fbImplicit = topoEff == Topology::FeedBack
                             && modeType == Mode::Downward
                             && detTypeEff == DetectorType::Peak;

        if (channel == 0
            && scHpfDirty_.load(std::memory_order_relaxed)
            && scHpfDirty_.exchange(false, std::memory_order_acquire))
            updateHpfCoefficients();

        // The sidechain filter must process whatever the detector consumes:
        // in FeedBack that is the compressed output (previous sample on the
        // explicit path; on the semi-implicit path the loop equation supplies
        // the gain, so the current input feeds the filter directly).
        T detectorIn = detectorSafe((topoEff == Topology::FeedBack && !fbImplicit)
                     ? fbLastOutput_[channel] : input);
        if (scHpf) detectorIn = applySidechainHPF(detectorIn, channel);

        T levelDb = detectLevel(detectorIn, channel, detTypeEff);

        const bool splitAdaptive = detTypeEff == DetectorType::SplitPolarity;
        const bool decoupled = useDecoupled(charType, topoEff);
        syncDecoupledState(decoupled);
        const GainPathConfig cfg { charType, modeType, topoEff, fbImplicit, splitAdaptive, decoupled,
                                   feedbackDetectorDelay(detTypeEff) };
        // The automatic threshold counts from the recent peak of what this
        // channel's detector reads (of its input, in FeedBack).
        T threshNow = thresh;
        if (autoThreshold_.load(std::memory_order_relaxed))
        {
            const T heard = (topoEff == Topology::FeedBack) ? gainToDecibels(std::abs(detectorSafe(input))) : levelDb;
            double& ref = autoRefDb_[static_cast<size_t>(channel)];
            ref = std::max(static_cast<double>(heard), ref - autoRefFall_);
            threshNow = thresh + static_cast<T>(ref);
        }
        T smoothedGR_Db = gainPath(levelDb, channel, cfg, threshNow, ratio, knee);
        // The gain the loop reads back is the whole one; the amount scales
        // what is heard (and metered).
        const T loopGainLinear = decibelsToGain(smoothedGR_Db);
        if (amount < T(1)) smoothedGR_Db *= amount;
        T smoothedGainLinear = (amount < T(1)) ? decibelsToGain(smoothedGR_Db) : loopGainLinear;

        const bool adaptiveMakeup = autoMkup == AutoMakeupMode::Adaptive && modeType == Mode::Downward;
        if (advanceShared)
        {
            syncAdaptiveMakeup(adaptiveMakeup);
            if (adaptiveMakeup)
                autoMakeupEnv_ = smoothedGR_Db + autoMakeupCoeff_ * (autoMakeupEnv_ - smoothedGR_Db);
        }

        T makeup = makeupDb;
        if (modeType == Mode::Downward)
        {
            if (autoMkup == AutoMakeupMode::Adaptive)
                makeup += -autoMakeupEnv_;
            else if (autoMkup == AutoMakeupMode::Static)
                makeup += computeGain(T(0), thresh, ratio, knee, charType,
                                      Mode::Downward, T(0)) * T(-0.5);
        }
        // The analytic makeup has a smoother of its own, advanced with the
        // shared ones.
        if (advanceShared)
        {
            analyticSmooth_.setTargetValue(autoMkup == AutoMakeupMode::Analytic
                ? analyticMakeupDb(charType, modeType, detTypeEff, decoupled,
                                   topoEff == Topology::FeedBack) : T(0));
            makeup += analyticSmooth_.getNextValue();
        }
        else
        {
            makeup += analyticSmooth_.getCurrentValue();
        }

        T output = input * smoothedGainLinear;
        if (colorAmt > T(0) && charType == Character::FET)
        {
            // Same 2nd-order FET channel modulation as the block path.
            const T sq = output * output;
            T& dc = fetDcState_[channel];
            if (std::isfinite(sq)) dc += fetDcCoeff_ * (sq - dc); // a non-finite sample would latch
            const T grDepth = std::clamp(-smoothedGR_Db * T(0.1), T(0), T(1));
            output += colorAmt * kFetColorH2 * grDepth * (sq - dc);
        }
        // The feedback detector reads the compressed signal BEFORE makeup:
        // makeup is applied after the element, so feeding it back would shift
        // the static curve by the makeup (and run away with auto makeup).
        fbLastOutput_[channel] = (amount < T(1)) ? input * loopGainLinear : output;
        output *= makeupToGain(makeup);
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
            envFastDb_[ch] = 0.0; // 0 dB gain change = neutral
            envSlowDb_[ch] = 0.0;
            envHoldDb_[ch] = 0.0;
            upwardGuardDb_[ch] = T(-200);
            fbLastOutput_[ch] = T(0);
            fetDcState_[ch] = T(0);
            scHpfState_[ch] = T(0);
            scHpfPrev_[ch] = T(0);
            pointLevelDb_[ch].fill(T(-200));
            precLevel_[ch] = -100.0;
            precTarget_[ch] = 0.0;
            interp_[ch] = {};
            lookaheadBuffers_[ch].reset();
            gainDelay_[ch].reset();
            autoRefDb_[ch] = -100.0;
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
        amountSmooth_.skip();
        analyticSmooth_.skip();
    }

    // =========================================================================
    // Parameter Setters
    // =========================================================================

    // All numeric setters ignore non-finite values (NaN/Inf) and keep the
    // previous parameter: a poisoned time constant, level or coefficient
    // would otherwise contaminate the gain path permanently.

    /** @brief Sets the compression threshold in dB. */
    void setThreshold(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        threshold_.store(dB, std::memory_order_relaxed);
    }

    /** @brief Sets the compression ratio (1.0 = off, >20.0 = limiting). */
    void setRatio(T ratio) noexcept
    {
        if (!std::isfinite(ratio)) return;
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
        if (!std::isfinite(ms)) return;
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
        if (!std::isfinite(ms)) return;
        releaseMs_.store(std::max(ms, T(1)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets knee width in dB (0 = hard knee, >0 = soft knee). */
    void setKnee(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        kneeWidth_.store(std::max(dB, T(0)), std::memory_order_relaxed);
    }

    /** @brief Sets manual static makeup gain in dB. */
    void setMakeupGain(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        makeupGain_.store(dB, std::memory_order_relaxed);
    }

    /**
     * @brief Selects the automatic makeup behavior (default: Off).
     *
     * Static is the textbook auto makeup: a constant, program-independent
     * offset of half the static gain reduction of a full-scale signal.
     * Adaptive is loudness matching: it tracks the smoothed gain reduction
     * (~300 ms) and compensates it in full, so the average output level
     * stays matched to the input and quiet passages are lifted accordingly
     * (macro-dynamics are reduced). Analytic works the gain out of the
     * settings instead of measuring it: where the ballistics come to rest
     * over dense programme, for the curve, the times, the character and the
     * detector selected; it is a constant between edits of those, and moves
     * to its new value over a few tens of milliseconds after one. It takes
     * the programme at -18 dBFS RMS unless setAutoThreshold() is on, which
     * makes it hold at any level. Measured on pink noise at that level, as
     * the level left over the input's: within 0.1 dB with the Peak detector
     * over thresholds from -30 to -10 dB, ratios from 2 to 10, attacks from
     * 0.5 to 100 ms and releases from 20 ms to 1 s (the Static mode leaves
     * -2.4 to +6.1 dB there), 0.2 dB decoupled and in feedback, 0.3 dB with
     * the RMS, Hilbert and TruePeak detectors and with Opto, 0.5 dB with FET;
     * the same figures at -36 and -6 dBFS with the automatic threshold. The
     * RMS window reads the low end of the programme: on white noise it is
     * off by up to 1.3 dB. For a fixed offset use setMakeupGain().
     */
    void setAutoMakeup(AutoMakeupMode mode) noexcept
    {
        autoMakeupMode_.store(clampEnum(mode, AutoMakeupMode::Analytic),
                              std::memory_order_relaxed);
    }

    /** @brief Convenience overload: true selects Adaptive, false turns auto makeup off. */
    void setAutoMakeup(bool on) noexcept
    {
        setAutoMakeup(on ? AutoMakeupMode::Adaptive : AutoMakeupMode::Off);
    }

    /** @brief Sets processing mode (Downward or Upward compression). */
    void setMode(Mode mode) noexcept
    {
        mode_.store(clampEnum(mode, Mode::Upward), std::memory_order_relaxed);
        // Feedback attack compensation only applies to Downward loops.
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Sets stereo linking amount (0.0 = unlinked dual mono, 1.0 = fully linked). */
    void setStereoLink(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        stereoLink_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Sets dry/wet balance for parallel (New York) compression (1.0 = fully wet). */
    void setMix(T dryWet) noexcept
    {
        if (!std::isfinite(dryWet)) return;
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets lookahead time in ms (0 = off, max = 10ms).
     * @warning Automatically disabled internally if topology is set to FeedBack.
     */
    void setLookahead(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        lookaheadMs_.store(std::clamp(ms, T(0), T(10)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Keeps a latency of its own for the lookahead, so the lookahead can move without the latency changing.
     *
     * With a maximum above 0 the audio is always that late (getLatency()
     * reports it) and setLookahead() moves inside it: the gain is held back
     * by the difference, so the lookahead can be automated while the host
     * keeps one delay compensation. 0 (the default) is the behaviour of
     * before: the latency is the lookahead. FeedBack runs without lookahead
     * and without latency either way.
     *
     * @param ms Latency kept for the lookahead, 0 to 10 ms.
     */
    void setMaxLookahead(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        maxLookaheadMs_.store(std::clamp(ms, T(0), T(10)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief The latency kept for the lookahead, in ms (0: none; see setMaxLookahead()). */
    [[nodiscard]] T getMaxLookahead() const noexcept { return maxLookaheadMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Makes the threshold count from the key's own recent peak instead of from full scale.
     *
     * A reference follows the level the detector reads: it rises at once
     * and falls 1 dB per second, and the threshold is added to it. The
     * compressor then acts the same on a take recorded 20 dB lower: what it
     * hears is how far the level is from its own recent maximum. In
     * FeedBack the reference follows the input. Off by default.
     */
    void setAutoThreshold(bool on) noexcept { autoThreshold_.store(on, std::memory_order_relaxed); }

    /** @brief Whether the threshold counts from the key's recent peak (setAutoThreshold()). */
    [[nodiscard]] bool isAutoThreshold() const noexcept { return autoThreshold_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets how much of the gain change is applied, as a scale of its decibels.
     *
     * 1 (the default) applies it all; 0.5 turns 8 dB of reduction into 4 at
     * every instant, with the same timing. Unlike the parallel mix it does
     * not add the unprocessed signal back, so there is nothing to cancel
     * against: it is the same compressor at a gentler ratio. In FeedBack the
     * loop keeps hearing the full reduction. Smoothed.
     *
     * @param amount Share of the gain change, 0 to 1.
     */
    void setAmount(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        amount_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief The share of the gain change that is applied (setAmount()). */
    [[nodiscard]] T getAmount() const noexcept { return amount_.load(std::memory_order_relaxed); }

    /**
     * @brief Makes the gain change approach the range without a corner.
     *
     * With it on, a target of g dB under a range of R becomes
     * g / (1 + (g / R)^4)^(1/4): the same as g while it is small (1.5 % less
     * at half the range), 16 % less at the range, and never more than the
     * range. Off (the default) the target is cut at the range. The
     * semi-implicit FeedBack loop (Peak detector) keeps the corner.
     */
    void setSoftRange(bool on) noexcept { softRange_.store(on, std::memory_order_relaxed); }

    /** @brief Whether the gain change approaches the range without a corner (setSoftRange()). */
    [[nodiscard]] bool isSoftRange() const noexcept { return softRange_.load(std::memory_order_relaxed); }

    /**
     * @brief Selects how finely the control signal is computed (default: Standard).
     *
     * A gain that moves within a cycle of the signal has a spectrum far wider
     * than the audio band; computed once per sample, what lies above half the
     * rate folds back and ends as lines that are not harmonics of the signal.
     * High works the control signal out between the samples too, the audio
     * being left at its rate:
     *
     * - the Peak and TruePeak detectors read the key at four points per
     *   sample (the sample and three points between it and the last one,
     *   from a windowed-sinc interpolation), so the rectifier works on the
     *   waveform and not on its samples;
     * - the ballistics take eight steps per sample, with the coefficients of
     *   an eighth of a sample, towards a target that moves along the line
     *   between those points (between samples for the RMS and Hilbert
     *   detectors, whose reading is already smooth); the gain is the mean of
     *   the eight.
     *
     * It applies in processBlock(), in feed-forward and in the feedback loops
     * solved against the input (the Peak detector, Downward: the FET
     * character among them); the SplitPolarity detector and the other
     * feedback loops stay as they are. The interpolated detectors read 7
     * samples late and the audio waits for them: getLatency() reports those
     * 7 samples more with Peak or TruePeak, none with RMS or Hilbert. Change
     * it while stopped (the delay of the audio moves at once).
     *
     * Measured at 48 kHz on a tone 12 dB over the threshold of a 4:1 curve,
     * as the largest line that is no harmonic of the tone (Standard to
     * High). Peak detector, attack 0.1 ms, release 50 ms: -77 to -111 dBc at
     * 1 kHz, -66 to -86 at 5 kHz, -51 to -85 at 15 kHz, -38 to -66 at
     * 18 kHz. FET character at its 20 us attack, ratio 20, 30 dB over: -68
     * to -91, -50 to -66, -49 to -70 and -37 to -66. Hilbert detector: -124
     * to -153 at 1 kHz. The level the detector reads does not move: the gain
     * of the tone stays within 0.1 dB in every case. The cost of a stereo
     * frame goes from 17 to 125 ns in feed-forward and from 52 to 190 with
     * the FET character (x64 server core).
     *
     * State key "precision" (absent = Standard).
     */
    void setPrecision(Precision precision) noexcept
    {
        precision_.store(clampEnum(precision, Precision::High), std::memory_order_relaxed);
    }

    /** @brief How finely the control signal is computed (see setPrecision()). */
    [[nodiscard]] Precision getPrecision() const noexcept { return precision_.load(std::memory_order_relaxed); }

    /** @brief Changes the level detection algorithm (Peak, RMS, TruePeak, Hilbert). */
    void setDetector(DetectorType type) noexcept
    {
        // The shared TruePeakDetector builds its coefficients lazily on first
        // use (thread-safe static), so this is a pure atomic publication.
        detectorType_.store(clampEnum(type, DetectorType::Hilbert), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the gain-reduction hold time.
     *
     * The deepest gain reduction is held for this long before the release
     * stage may recover, the classic tool against pumping on dense material.
     *
     * @param ms Hold time in milliseconds (0 = off, mastering range 0-500).
     */
    void setHoldTime(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
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
        if (!std::isfinite(dB)) return;
        rangeDb_.store(std::max(dB, T(0)), std::memory_order_relaxed);
    }

    /** @brief Changes signal routing topology (FeedForward or FeedBack). */
    void setTopology(Topology topo) noexcept
    {
        topology_.store(clampEnum(topo, Topology::FeedBack), std::memory_order_relaxed);
        // Feedback loops re-derive the attack coefficient (loop speed-up).
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Changes ballistics and envelope behavior character. */
    void setCharacter(Character type) noexcept
    {
        character_.store(clampEnum(type, Character::Varimu), std::memory_order_relaxed);
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
        if (!std::isfinite(amount)) return;
        characterColor_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Selects the Clean character's feed-forward ballistics.
     *
     * Branching (default) is a single branching one-pole on the gain target.
     * Decoupled holds the gain target's peaks and smooths the held envelope,
     * so a sustained tone is compressed without its waveform riding the
     * gain: THD+N of a 1 kHz tone 14 dB into 4:1 at 5/100 ms measures
     * -102 dB, against -65.5 dB for Branching, and the gain lands on the
     * peak-referenced static curve (Branching reads up to 0.8 dB less
     * reduction on tones). Both honour the attack and release knobs as t63;
     * Decoupled rounds the first part of the release (within 0.8 dB of the
     * exponential on a 12 dB release at 5/100 ms).
     *
     * Decoupled follows the peaks of dense material, so the same settings
     * compress more: on a drum programme at the default settings the output
     * sits 1.9 dB lower than with Branching, on a noise-like programme at
     * -24 dB, 3:1 it sits 4.5 dB lower. Compare by ear (and retune the
     * threshold) before switching a preset. Other characters and feedback
     * operation are not affected; the state key is "cleanBallistics"
     * (absent = Branching).
     */
    void setCleanBallistics(CleanBallistics mode) noexcept
    {
        cleanBallistics_.store(clampEnum(mode, CleanBallistics::Decoupled), std::memory_order_relaxed);
    }

    /** @brief Returns the Clean character's feed-forward ballistics. */
    [[nodiscard]] CleanBallistics getCleanBallistics() const noexcept
    {
        return cleanBallistics_.load(std::memory_order_relaxed);
    }

    /** @brief Returns the character colour amount (see setCharacterColor). */
    [[nodiscard]] T getCharacterColor() const noexcept
    {
        return characterColor_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Toggles the internal sidechain high-pass filter.
     *
     * An invalid cutoff (non-positive or non-finite) keeps the previous
     * frequency and only applies the toggle: a negative cutoff would make
     * the one-pole coefficient exceed 1 (an unstable, runaway filter).
     *
     * @param enabled True to activate HPF.
     * @param cutoffHz Cutoff frequency in Hz (Default: 80Hz).
     */
    void setSidechainHPF(bool enabled, T cutoffHz = T(80)) noexcept
    {
        scHpfEnabled_.store(enabled, std::memory_order_relaxed);
        if (std::isfinite(cutoffHz) && cutoffHz > T(0))
            scHpfFreq_.store(cutoffHz, std::memory_order_relaxed);
        // Consumed by processBlock() and by processSample() (channel 0).
        scHpfDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the RMS analysis window size in milliseconds.
     *
     * Published atomically; the audio thread applies the new window (and
     * resets the sliding state) at its next block, so no control-thread write
     * ever races the per-sample RMS accumulators.
     *
     * In FeedBack the RMS window sits inside the loop, which cannot settle
     * faster than its detector reports: release faster than about
     * e x (1 + window) samples, and attack faster than half of that, settle
     * at those bounds, the critically damped limit (observed t63 with the
     * default 10 ms window: release 22 ms, attack 14 ms). Faster settings
     * would overshoot and ring.
     *
     * @warning Clamped to the 1-500 ms range of the prepare() pre-allocation.
     */
    void setRmsWindow(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        rmsWindowMsAtomic_.store(std::clamp(ms, T(1), T(500)), std::memory_order_relaxed);
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
     * The lookahead. Feedback operation (Topology::FeedBack, or the FET
     * character, which always detects in feedback) disables it, so latency
     * is 0 there. With setPrecision(High), the Peak and TruePeak detectors
     * read 7 samples late and the audio waits for them, in feed-forward and
     * in the feedback loops that mode covers. The plugin layer re-reads this
     * after parameter changes and re-notifies the host when it moves.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        const bool fet = character_.load(std::memory_order_relaxed) == Character::FET;
        const bool feedback = fet || topology_.load(std::memory_order_relaxed) == Topology::FeedBack;
        const DetectorType detType = fet ? DetectorType::Peak : detectorType_.load(std::memory_order_relaxed);
        // The same routing as processBlock(): see usePrecision().
        const bool solved = feedback && detType == DetectorType::Peak
                         && mode_.load(std::memory_order_relaxed) == Mode::Downward;
        const int reading = (precision_.load(std::memory_order_relaxed) == Precision::High
                             && readsBetweenSamples(detType) && (!feedback || solved)) ? kDetectorDelay : 0;
        if (feedback) return reading;
        // Derive the lookahead from the published parameter rather than the
        // audio-thread cache: hosts re-read the latency right after a setter,
        // before the next block has consumed the coefficient-update flag.
        const int lookNow = static_cast<int>(static_cast<T>(sampleRate_) * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        const int kept = static_cast<int>(static_cast<T>(sampleRate_) * std::clamp(
            maxLookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        return std::max(lookNow, kept) + reading;
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("COMP"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("threshold", static_cast<float>(threshold_.load(std::memory_order_relaxed)));
        w.write("ratio", static_cast<float>(ratio_.load(std::memory_order_relaxed)));
        w.write("attack", static_cast<float>(attackMs_.load(std::memory_order_relaxed)));
        w.write("release", static_cast<float>(releaseMs_.load(std::memory_order_relaxed)));
        w.write("knee", static_cast<float>(kneeWidth_.load(std::memory_order_relaxed)));
        w.write("makeup", static_cast<float>(makeupGain_.load(std::memory_order_relaxed)));
        const auto amMode = autoMakeupMode_.load(std::memory_order_relaxed);
        w.write("autoMakeup", amMode != AutoMakeupMode::Off); // legacy bool key
        w.write("autoMakeupMode", static_cast<int32_t>(amMode));
        w.write("stereoLink", static_cast<float>(stereoLink_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("lookahead", static_cast<float>(lookaheadMs_.load(std::memory_order_relaxed)));
        w.write("hold", static_cast<float>(holdMs_.load(std::memory_order_relaxed)));
        w.write("range", static_cast<float>(rangeDb_.load(std::memory_order_relaxed)));
        w.write("detector", static_cast<int32_t>(detectorType_.load(std::memory_order_relaxed)));
        w.write("topology", static_cast<int32_t>(topology_.load(std::memory_order_relaxed)));
        w.write("character", static_cast<int32_t>(character_.load(std::memory_order_relaxed)));
        w.write("characterColor", static_cast<float>(characterColor_.load(std::memory_order_relaxed)));
        w.write("cleanBallistics", static_cast<int32_t>(cleanBallistics_.load(std::memory_order_relaxed)));
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("scHpf", scHpfEnabled_.load(std::memory_order_relaxed));
        w.write("scHpfFreq", static_cast<float>(scHpfFreq_.load(std::memory_order_relaxed)));
        w.write("rmsWindow", static_cast<float>(rmsWindowMsAtomic_.load(std::memory_order_relaxed)));
        w.write("autoThreshold", autoThreshold_.load(std::memory_order_relaxed));
        w.write("maxLookahead", static_cast<float>(maxLookaheadMs_.load(std::memory_order_relaxed)));
        w.write("amount", static_cast<float>(amount_.load(std::memory_order_relaxed)));
        w.write("softRange", softRange_.load(std::memory_order_relaxed));
        w.write("precision", static_cast<int32_t>(precision_.load(std::memory_order_relaxed)));
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
            std::clamp(r.read("autoMakeupMode", amLegacy), 0, 3)));
        setStereoLink(static_cast<T>(r.read("stereoLink", 1.0f)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setLookahead(static_cast<T>(r.read("lookahead", 0.0f)));
        setHoldTime(static_cast<T>(r.read("hold", 0.0f)));
        setRange(static_cast<T>(r.read("range", 100.0f)));
        setDetector(static_cast<DetectorType>(r.read("detector", 0)));
        setTopology(static_cast<Topology>(r.read("topology", 0)));
        setCharacter(static_cast<Character>(r.read("character", 0)));
        setCharacterColor(static_cast<T>(r.read("characterColor", 0.0f)));
        setCleanBallistics(static_cast<CleanBallistics>(r.read("cleanBallistics", 0)));
        setMode(static_cast<Mode>(r.read("mode", 0)));
        setSidechainHPF(r.read("scHpf", false), static_cast<T>(r.read("scHpfFreq", 80.0f)));
        setRmsWindow(static_cast<T>(r.read("rmsWindow", 10.0f)));
        setAutoThreshold(r.read("autoThreshold", false));
        setMaxLookahead(static_cast<T>(r.read("maxLookahead", 0.0f)));
        setAmount(static_cast<T>(r.read("amount", 1.0f)));
        setSoftRange(r.read("softRange", false));
        setPrecision(static_cast<Precision>(r.read("precision", 0)));
        return true;
    }

protected:
    static constexpr int kMaxChannels = 16;

    /**
     * @brief Clamps an enum to its valid [first, last] range.
     *
     * Wild values (a cast integer, a corrupted state blob) would otherwise
     * fall through every switch case: updateTimeConstants() would keep its
     * zero-initialized coefficients (instant ballistics, i.e. a waveshaper)
     * and the getters would report an impossible value.
     */
    template <typename E>
    [[nodiscard]] static E clampEnum(E value, E last) noexcept
    {
        const int v = std::clamp(static_cast<int>(value), 0, static_cast<int>(last));
        return static_cast<E>(v);
    }

    /**
     * @brief Core DSP loop executing sidechain detection, linking, ballistics, and gain application.
     *
     * No-op before prepare(). A sidechain view with fewer samples than the
     * audio view is ignored (the detector falls back to the audio itself)
     * rather than read out of bounds.
     *
     * @param audio Buffer to modify.
     * @param sidechain Buffer to read for level detection.
     */
    void processBlockImpl(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        if (!(sampleRate_ > 0)) return; // not prepared: leave the audio untouched
        DenormalGuard guard;
        const int nCh  = std::min(audio.getNumChannels(), kMaxChannels);
        const int nS   = audio.getNumSamples();
        // A short external sidechain would be read past its end; fall back
        // to the internal key instead (release-safe, documented above).
        const int scCh = (sidechain.getNumSamples() >= nS)
                       ? sidechain.getNumChannels() : 0;

        // Sync atomic parameters to block-local smoothed state
        thresholdSmooth_.setTargetValue(threshold_.load(std::memory_order_relaxed));
        ratioSmooth_.setTargetValue(std::max(ratio_.load(std::memory_order_relaxed), T(1)));
        kneeSmooth_.setTargetValue(std::max(kneeWidth_.load(std::memory_order_relaxed), T(0)));
        makeupSmooth_.setTargetValue(makeupGain_.load(std::memory_order_relaxed));
        mixSmooth_.setTargetValue(mix_.load(std::memory_order_relaxed));
        colorSmooth_.setTargetValue(characterColor_.load(std::memory_order_relaxed));
        amountSmooth_.setTargetValue(amount_.load(std::memory_order_relaxed));

        if (timeConstantsDirty_.exchange(false, std::memory_order_acquire))
            updateTimeConstants(static_cast<T>(sampleRate_));
        if (scHpfDirty_.load(std::memory_order_relaxed)
            && scHpfDirty_.exchange(false, std::memory_order_acquire))
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

        const bool splitAdaptive = detTypeEff == DetectorType::SplitPolarity;
        const bool decoupled = useDecoupled(charType, topoEff);
        syncDecoupledState(decoupled);
        const bool precise = usePrecision(topoEff, fbImplicit, splitAdaptive);
        syncPrecisionState(precise);

        // Lookahead breaks causal logic in Feedback mode, so it is strictly
        // disabled there.
        int activeLookahead = (topoEff == Topology::FeedBack) ? 0 : latencySamples_;
        const bool autoThr = autoThreshold_.load(std::memory_order_relaxed);

        // Fully linked channels share one gain computer: every channel reads
        // the same linked level into the same state, so computing it once per
        // frame gives the same gain (the per-channel path differs only in the
        // rounding of level + 1 * (linked - level)). SplitPolarity's adaptive
        // release reads each channel's own output, so it keeps the
        // per-channel path.
        const bool shared = (sLink >= T(1)) && !splitAdaptive && nCh > 1;

        // Settled parameter smoothers: read once per block instead of a call
        // per sample (a settled smoother returns its target either way).
        analyticSmooth_.setTargetValue(autoMkup == AutoMakeupMode::Analytic
            ? analyticMakeupDb(charType, modeType, detTypeEff, decoupled,
                               topoEff == Topology::FeedBack) : T(0));
        const bool paramsMoving = thresholdSmooth_.isSmoothing() || ratioSmooth_.isSmoothing()
                               || kneeSmooth_.isSmoothing() || makeupSmooth_.isSmoothing()
                               || mixSmooth_.isSmoothing() || colorSmooth_.isSmoothing()
                               || amountSmooth_.isSmoothing() || analyticSmooth_.isSmoothing();
        T thresh   = thresholdSmooth_.getCurrentValue();
        T ratio    = ratioSmooth_.getCurrentValue();
        T knee     = kneeSmooth_.getCurrentValue();
        T mkupBase = makeupSmooth_.getCurrentValue();
        T mixVal   = mixSmooth_.getCurrentValue();
        T colorAmt = colorSmooth_.getCurrentValue();
        T amount   = amountSmooth_.getCurrentValue();
        const bool staticMakeup = autoMkup == AutoMakeupMode::Static && modeType == Mode::Downward;
        const bool adaptiveMakeup = autoMkup == AutoMakeupMode::Adaptive && modeType == Mode::Downward;
        syncAdaptiveMakeup(adaptiveMakeup);
        T staticMakeupDb = staticMakeup
            ? computeGain(T(0), thresh, ratio, knee, charType, Mode::Downward, T(0)) * T(-0.5)
            : T(0);
        T analyticDb = analyticSmooth_.getCurrentValue();

        const GainPathConfig cfg { charType, modeType, topoEff, fbImplicit, splitAdaptive, decoupled,
                                   feedbackDetectorDelay(detTypeEff) };

        // Feed-forward: the detector never reads the output, so the block
        // runs in passes (see processFeedForward). Feedback loops stay per
        // sample, and so does Precision::High, whose control signal is
        // worked out several times per sample.
        const bool feedForward = topoEff == Topology::FeedForward;
        if (feedForward && !precise)
        {
            const FeedForwardConfig ff { detTypeEff, charType, modeType, autoMkup, sLink,
                                         activeLookahead, scHpf, shared, decoupled,
                                         splitAdaptive, staticMakeup, adaptiveMakeup,
                                         paramsMoving, autoThr, activeLookahead - lookaheadSamples_ };
            const ChunkParams constants { thresh, ratio, knee, mkupBase, mixVal,
                                          staticMakeupDb + analyticDb, amount };
            processFeedForward(audio, sidechain, nCh, nS, scCh, ff, constants);
            return;
        }

        // At Precision::High the detectors that rectify read the key at
        // kDetectorPoints points per sample, kDetectorDelay samples late, and
        // the audio waits for them. The others keep one level per sample.
        const bool points = precise && readsBetweenSamples(detTypeEff);
        const int count = points ? kDetectorPoints : 1;
        const int audioDelay = activeLookahead + (points ? kDetectorDelay : 0);
        const int gainDelay = feedForward ? activeLookahead - lookaheadSamples_ : 0;

        for (int i = 0; i < nS; ++i)
        {
            if (paramsMoving)
            {
                thresh   = thresholdSmooth_.getNextValue();
                ratio    = ratioSmooth_.getNextValue();
                knee     = kneeSmooth_.getNextValue();
                mkupBase = makeupSmooth_.getNextValue();
                mixVal   = mixSmooth_.getNextValue();
                colorAmt = colorSmooth_.getNextValue();
                amount   = amountSmooth_.getNextValue();
                // Static auto makeup follows the smoothed curve parameters, so
                // it stays click-free through threshold/ratio/knee automation.
                if (staticMakeup)
                    staticMakeupDb = computeGain(T(0), thresh, ratio, knee, charType,
                                                 Mode::Downward, T(0)) * T(-0.5);
                analyticDb = analyticSmooth_.getNextValue();
            }

            // Makeup: shared by every channel of the frame, and usually
            // unchanged from the previous frame (the conversion is cached).
            // The adaptive envelope is the one left by the previous frame.
            T makeupDb = mkupBase + staticMakeupDb + analyticDb;
            if (adaptiveMakeup)
                makeupDb += -autoMakeupEnv_;
            const T makeupLin = makeupToGain(makeupDb);

            // 1. Detection path (per channel: every detector keeps its own state)
            T linkedLin[kDetectorPoints] = {};
            for (int ch = 0; ch < nCh; ++ch)
            {
                // The sidechain filter must process whatever the detector
                // consumes. In feed-forward that is the key (the audio when
                // the sidechain buffer lacks channels). In FeedBack it is the
                // compressed output (the external key is ignored): the
                // previous sample on the explicit path, or the channel's own
                // current input on the semi-implicit path (the loop equation
                // supplies the gain).
                T detectorIn;
                if (feedForward)
                    detectorIn = (scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))[i]
                                            : audio.getChannel(ch)[i];
                else
                    detectorIn = fbImplicit ? audio.getChannel(ch)[i] : fbLastOutput_[ch];
                detectorIn = detectorSafe(detectorIn);
                if (scHpf) detectorIn = applySidechainHPF(detectorIn, ch);

                // dB is monotonic in the linear level, so the linked level is
                // the dB of the largest linear level (one log per frame and
                // point when the channels share the gain computer).
                T level[kDetectorPoints] = {};
                if (points) detectorPoints(detectorIn, ch, level);
                else level[0] = detectLevelLinear(detectorIn, ch, detTypeEff);
                for (int j = 0; j < count; ++j)
                {
                    linkedLin[j] = std::max(linkedLin[j], level[j]);
                    if (!shared) pointLevelDb_[ch][j] = gainToDecibels(level[j]);
                }
            }

            // The automatic threshold: its reference follows the largest
            // level the gain computers read in feed-forward, and the linked
            // input in feedback, where the detector reads the output.
            T threshNow = thresh;
            if (autoThr)
            {
                T top = T(0);
                if (feedForward)
                    for (int j = 0; j < count; ++j) top = std::max(top, linkedLin[j]);
                else
                    for (int ch = 0; ch < nCh; ++ch) top = std::max(top, std::abs(detectorSafe(audio.getChannel(ch)[i])));
                autoRefDb_[0] = std::max(static_cast<double>(gainToDecibels(top)), autoRefDb_[0] - autoRefFall_);
                threshNow = thresh + static_cast<T>(autoRefDb_[0]);
            }

            // 2. Stereo linking, static curve and character ballistics (log
            //    domain). What is heard of the gain is held back inside a
            //    latency kept for the lookahead and scaled by the amount; the
            //    loop keeps the whole.
            T blockGR = T(0);
            if (shared)
            {
                T levelDb[kDetectorPoints] = {};
                for (int j = 0; j < count; ++j) levelDb[j] = gainToDecibels(linkedLin[j]);
                const T loopGR = precise ? gainPathPrecise(levelDb, count, 0, cfg, threshNow, ratio, knee)
                                         : gainPath(levelDb[0], 0, cfg, threshNow, ratio, knee);
                // (0 dB is the common case below threshold: skip the exp.)
                const T gainLin = (loopGR == T(0)) ? T(1) : decibelsToGain(loopGR);
                blockGR = heldBack(loopGR, 0, gainDelay);
                if (amount < T(1)) blockGR *= amount;
                const T outLin = (blockGR == loopGR) ? gainLin : decibelsToGain(blockGR);
                for (int ch = 0; ch < nCh; ++ch)
                    applyGain(audio, ch, i, gainLin, blockGR, audioDelay, charType,
                              colorAmt, makeupLin, mixVal, outLin);
            }
            else
            {
                T linkedDb[kDetectorPoints] = {};
                for (int j = 0; j < count; ++j)
                {
                    linkedDb[j] = pointLevelDb_[0][j];
                    for (int ch = 1; ch < nCh; ++ch) linkedDb[j] = std::max(linkedDb[j], pointLevelDb_[ch][j]);
                }
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T levelDb[kDetectorPoints] = {};
                    for (int j = 0; j < count; ++j)
                    {
                        const T own = pointLevelDb_[ch][j];
                        levelDb[j] = own + sLink * (linkedDb[j] - own);
                    }
                    const T loopGR = precise ? gainPathPrecise(levelDb, count, ch, cfg, threshNow, ratio, knee)
                                             : gainPath(levelDb[0], ch, cfg, threshNow, ratio, knee);
                    const T gainLin = (loopGR == T(0)) ? T(1) : decibelsToGain(loopGR);
                    T heardGR = heldBack(loopGR, ch, gainDelay);
                    if (amount < T(1)) heardGR *= amount;
                    const T outLin = (heardGR == loopGR) ? gainLin : decibelsToGain(heardGR);
                    applyGain(audio, ch, i, gainLin, heardGR, audioDelay, charType,
                              colorAmt, makeupLin, mixVal, outLin);
                    if (ch == 0 || heardGR < blockGR)
                        blockGR = heardGR; // Track worst-case GR for metering
                }
            }

            gainReductionDb_.store(blockGR, std::memory_order_relaxed);

            // Auto-makeup envelope tracks slowly (~300ms)
            if (adaptiveMakeup)
                autoMakeupEnv_ = blockGR + autoMakeupCoeff_ * (autoMakeupEnv_ - blockGR);
        }

        // The shared gain computer ran on channel 0's state: hand it to the
        // other channels so a later unlinked block continues from it.
        if (shared && nS > 0) copyGainStateFromChannel0(nCh);
    }

    /** @brief Per-block routing of the gain path (see gainPath()). */
    struct GainPathConfig
    {
        Character charType;
        Mode modeType;
        Topology topoEff;
        bool fbImplicit;
        bool splitAdaptive;
        bool decoupled;
        double loopDelay; ///< Detector delay seen by an explicit feedback loop (samples).
    };

    /** @brief Per-block routing of the feed-forward block path. */
    struct FeedForwardConfig
    {
        DetectorType detType;
        Character charType;
        Mode modeType;
        AutoMakeupMode autoMkup;
        T sLink;
        int lookahead;
        bool scHpf;
        bool shared;
        bool decoupled;
        bool splitAdaptive;
        bool staticMakeup;
        bool adaptiveMakeup;
        bool paramsMoving;
        bool autoThreshold;     ///< The threshold counts from the key's recent peak.
        int gainDelay;          ///< Samples the gain is held back: the latency kept minus the lookahead.
    };

    /** @brief Parameter values of a block whose smoothers are settled. */
    struct ChunkParams
    {
        T thresh, ratio, knee, makeup, mix, staticMakeup, amount;
    };

    static constexpr int kChunk = 32; ///< Frames per pass of the feed-forward path.

    /**
     * @brief Feed-forward block path, run in passes over chunks of frames.
     *
     * In feed-forward the detector never reads the output, so detection, the
     * level conversion, the gain recursion, the gain conversion and its
     * application can run as separate passes. Only the ballistics recursion
     * is then serial: the log, static curve and exp of different frames
     * overlap freely, and the envelope state stays in registers. Each frame
     * performs the same arithmetic as the per-sample path (processSample()
     * on a mono stream is bit-identical).
     */
    void processFeedForward(AudioBufferView<T>& audio, AudioBufferView<T>& sidechain,
                            int nCh, int nS, int scCh, const FeedForwardConfig& ff,
                            const ChunkParams& constants) noexcept
    {
        T thr[kChunk], rat[kChunk], kn[kChunk], mk[kChunk], mixA[kChunk], stMk[kChunk];
        T amt[kChunk], thrNow[kChunk];
        T mkLin[kChunk];
        T level[kMaxChannels][kChunk];   // linear level, then dB level, then linear gain
        T gain[kMaxChannels][kChunk];    // smoothed gain change (dB)
        T delayed[kMaxChannels][kChunk]; // lookahead-delayed input (the dry reference)
        const int nGain = ff.shared ? 1 : nCh;

        if (!ff.paramsMoving)
        {
            std::fill_n(thr, kChunk, constants.thresh);
            std::fill_n(rat, kChunk, constants.ratio);
            std::fill_n(kn, kChunk, constants.knee);
            std::fill_n(mk, kChunk, constants.makeup);
            std::fill_n(mixA, kChunk, constants.mix);
            std::fill_n(stMk, kChunk, constants.staticMakeup);
            std::fill_n(amt, kChunk, constants.amount);
        }

        T lastGr = T(0);
        for (int o = 0; o < nS; o += kChunk)
        {
            const int m = std::min(kChunk, nS - o);

            // A. Parameters (the colour smoother advances too: it only
            //    applies to the FET character, which is always feedback).
            if (ff.paramsMoving)
            {
                for (int k = 0; k < m; ++k)
                {
                    thr[k]  = thresholdSmooth_.getNextValue();
                    rat[k]  = ratioSmooth_.getNextValue();
                    kn[k]   = kneeSmooth_.getNextValue();
                    mk[k]   = makeupSmooth_.getNextValue();
                    mixA[k] = mixSmooth_.getNextValue();
                    (void)colorSmooth_.getNextValue();
                    amt[k]  = amountSmooth_.getNextValue();
                    stMk[k] = (ff.staticMakeup
                        ? computeGain(T(0), thr[k], rat[k], kn[k], ff.charType, Mode::Downward, T(0)) * T(-0.5)
                        : T(0)) + analyticSmooth_.getNextValue();
                }
            }

            // B. Detection (each channel's detector keeps its own state).
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* src = ((scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))
                                           : audio.getChannel(ch)) + o;
                T* lv = level[ch];
                if (ff.detType == DetectorType::Peak && !ff.scHpf)
                {
                    for (int k = 0; k < m; ++k) lv[k] = std::abs(detectorSafe(src[k]));
                }
                else
                {
                    for (int k = 0; k < m; ++k)
                    {
                        T d = detectorSafe(src[k]);
                        if (ff.scHpf) d = applySidechainHPF(d, ch);
                        lv[k] = detectLevelLinear(d, ch, ff.detType);
                    }
                }
            }

            // Lookahead: the delayed audio is the input the gain applies to.
            if (ff.lookahead > 0)
            {
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* src = audio.getChannel(ch) + o;
                    for (int k = 0; k < m; ++k)
                    {
                        lookaheadBuffers_[ch].push(src[k]);
                        delayed[ch][k] = lookaheadBuffers_[ch].read(ff.lookahead);
                    }
                }
            }

            // C. Levels in dB and stereo linking. dB is monotonic in the
            //    linear level, so the linked level is the largest channel's.
            if (ff.shared)
            {
                for (int k = 0; k < m; ++k)
                {
                    T mx = level[0][k];
                    for (int ch = 1; ch < nCh; ++ch) mx = std::max(mx, level[ch][k]);
                    level[0][k] = gainToDecibels(mx);
                }
            }
            else
            {
                for (int ch = 0; ch < nCh; ++ch)
                    for (int k = 0; k < m; ++k) level[ch][k] = gainToDecibels(level[ch][k]);
                if (nCh > 1 && ff.sLink > T(0))
                {
                    for (int k = 0; k < m; ++k)
                    {
                        T linked = level[0][k];
                        for (int ch = 1; ch < nCh; ++ch) linked = std::max(linked, level[ch][k]);
                        for (int ch = 0; ch < nCh; ++ch)
                            level[ch][k] = level[ch][k] + ff.sLink * (linked - level[ch][k]);
                    }
                }
            }

            // The automatic threshold: the reference follows the largest
            // level the gain computers read, and the threshold counts from it.
            const T* thrUse = thr;
            if (ff.autoThreshold)
            {
                double ref = autoRefDb_[0];
                for (int k = 0; k < m; ++k)
                {
                    T top = level[0][k];
                    for (int g = 1; g < nGain; ++g) top = std::max(top, level[g][k]);
                    ref = std::max(static_cast<double>(top), ref - autoRefFall_);
                    thrNow[k] = thr[k] + static_cast<T>(ref);
                }
                autoRefDb_[0] = ref;
                thrUse = thrNow;
            }

            // D. Static curve, hold/range and ballistics: the serial part.
            //    SplitPolarity's level-adaptive release reads the previous
            //    frame's output, so its recursion also forms the linear gain
            //    (written over the consumed dB level).
            for (int g = 0; g < nGain; ++g)
            {
                const T* dryIn = (ff.lookahead > 0) ? delayed[g] : audio.getChannel(g) + o;
                runGainRecursion(level[g], gain[g], m, g, ff, thrUse, rat, kn,
                                 ff.splitAdaptive ? dryIn : nullptr);
            }

            // What is heard of the gain: held back while the lookahead is
            // shorter than the latency kept for it, and scaled by the amount.
            const bool scaled = ff.paramsMoving || constants.amount < T(1);
            if (ff.gainDelay > 0 || scaled)
            {
                for (int g = 0; g < nGain; ++g)
                {
                    T* gd = gain[g];
                    if (ff.gainDelay > 0)
                        for (int k = 0; k < m; ++k)
                        {
                            gainDelay_[g].push(gd[k]);
                            gd[k] = gainDelay_[g].read(ff.gainDelay);
                        }
                    if (scaled)
                        for (int k = 0; k < m; ++k)
                            if (amt[k] < T(1)) gd[k] *= amt[k];
                    // SplitPolarity formed its linear gain inside the recursion.
                    if (ff.splitAdaptive)
                        for (int k = 0; k < m; ++k)
                            level[g][k] = (gd[k] == T(0)) ? T(1) : decibelsToGain(gd[k]);
                }
            }

            // E. Metering, adaptive makeup (frame by frame, like the
            //    per-sample path) and the linear gains.
            if (ff.adaptiveMakeup)
            {
                T env = autoMakeupEnv_; // a local: the audio pointers may alias members
                for (int k = 0; k < m; ++k)
                {
                    T blockGR = gain[0][k];
                    for (int g = 1; g < nGain; ++g) blockGR = std::min(blockGR, gain[g][k]);
                    mkLin[k] = makeupToGain(mk[k] + stMk[k] + -env);
                    env = blockGR + autoMakeupCoeff_ * (env - blockGR);
                }
                autoMakeupEnv_ = env;
            }
            else
            {
                for (int k = 0; k < m; ++k) mkLin[k] = makeupToGain(mk[k] + stMk[k]);
            }
            lastGr = gain[0][m - 1];
            for (int g = 1; g < nGain; ++g) lastGr = std::min(lastGr, gain[g][m - 1]);
            if (!ff.splitAdaptive)
            {
                for (int g = 0; g < nGain; ++g)
                {
                    const T* gd = gain[g];
                    T* gl = level[g];
                    // (0 dB is the common case below threshold: skip the exp.)
                    for (int k = 0; k < m; ++k) gl[k] = (gd[k] == T(0)) ? T(1) : decibelsToGain(gd[k]);
                }
            }

            // F. Application: lookahead, makeup and the parallel mix. The dry
            //    reference carries the same lookahead delay as the wet.
            const bool plain = ff.lookahead == 0 && !ff.paramsMoving && constants.mix >= T(1);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = audio.getChannel(ch) + o;
                const T* gl = level[ff.shared ? 0 : ch];
                if (plain)
                {
                    // No lookahead and fully wet: one multiply chain per sample.
                    fbLastOutput_[ch] = data[m - 1] * gl[m - 1]; // pre-makeup history
                    for (int k = 0; k < m; ++k) data[k] = (data[k] * gl[k]) * mkLin[k];
                    continue;
                }
                const T* dry = (ff.lookahead > 0) ? delayed[ch] : data;
                T wet = T(0);
                for (int k = 0; k < m; ++k)
                {
                    const T input = dry[k];
                    wet = input * gl[k];
                    const T out = wet * mkLin[k];
                    data[k] = (mixA[k] < T(1)) ? (input * (T(1) - mixA[k]) + out * mixA[k]) : out;
                }
                // Feedback history (pre-makeup) for a later topology switch.
                fbLastOutput_[ch] = wet;
            }
        }
        if (nS > 0) gainReductionDb_.store(lastGr, std::memory_order_relaxed);

        // The shared gain computer ran on channel 0's state: hand it to the
        // other channels so a later unlinked block continues from it.
        if (ff.shared && nS > 0) copyGainStateFromChannel0(nCh);
    }

    /** @brief Copies channel 0's gain-computer state to channels [1, nCh). */
    void copyGainStateFromChannel0(int nCh) noexcept
    {
        for (int ch = 1; ch < nCh; ++ch)
        {
            envFastDb_[ch] = envFastDb_[0];
            envSlowDb_[ch] = envSlowDb_[0];
            envHoldDb_[ch] = envHoldDb_[0];
            upwardGuardDb_[ch] = upwardGuardDb_[0];
            holdCounters_[ch] = holdCounters_[0];
            heldGrDb_[ch] = heldGrDb_[0];
            precLevel_[ch] = precLevel_[0];
            precTarget_[ch] = precTarget_[0];
        }
    }

    /**
     * @brief The serial part of the feed-forward path for one gain channel:
     *        Upward guard, static curve, hold/range and ballistics over a
     *        chunk, with the channel's state held in locals.
     *
     * @param inDb   Linked level per frame (dB).
     * @param out    Smoothed gain change per frame (dB).
     * @param m      Frames in the chunk.
     * @param ch     Channel whose state is advanced.
     * @param ff     Block routing.
     * @param thr    Threshold per frame.
     * @param rat    Ratio per frame.
     * @param kn     Knee per frame.
     * @param dryIn  SplitPolarity only: the (delayed) input the gain applies
     *               to. The recursion then also writes the linear gain into
     *               inDb (each frame's level is consumed before it is
     *               overwritten). nullptr otherwise.
     */
    void runGainRecursion(T* inDb, T* out, int m, int ch, const FeedForwardConfig& ff,
                          const T* thr, const T* rat, const T* kn, const T* dryIn) noexcept
    {
        double fast = envFastDb_[ch];
        double slow = envSlowDb_[ch];
        double hold = envHoldDb_[ch];
        T guard = upwardGuardDb_[ch];
        T held = heldGrDb_[ch];
        int counter = holdCounters_[ch];
        const T range = rangeDb_.load(std::memory_order_relaxed);
        const bool soft = softRange_.load(std::memory_order_relaxed);
        const int holdN = holdSamples_;
        const bool upward = ff.modeType == Mode::Upward;

        // Static curve first: no state, so the frames overlap freely (the
        // Upward guard is a cheap recursion of its own).
        T tgtA[kChunk];
        if (upward)
        {
            for (int k = 0; k < m; ++k)
            {
                guard = std::max(inDb[k], guard - upwardGuardDecay_);
                tgtA[k] = computeGainUpward(inDb[k], thr[k], rat[k], kn[k], guard);
            }
        }
        else
        {
            for (int k = 0; k < m; ++k)
                tgtA[k] = downwardCurve(inDb[k], thr[k], rat[k], kn[k], ff.charType);
        }

        if (dryIn != nullptr)
        {
            // Level-adaptive release (SplitPolarity): full-scale output
            // halves the release time constant, as in applyBallistics().
            T lastWet = fbLastOutput_[ch];
            for (int k = 0; k < m; ++k)
            {
                const T tgt = holdAndRange(tgtA[k], range, holdN, held, counter, soft);
                const double lastOut = std::abs(static_cast<double>(lastWet));
                const double outputLevel = (lastOut < 1.0) ? lastOut : 1.0;
                double y;
                if (ff.decoupled)
                {
                    const double c = decHoldCoeff_;
                    y = decoupledStep(static_cast<double>(tgt), hold, fast, c + outputLevel * (c * c - c));
                }
                else
                {
                    const double c = charRelCoeff_;
                    y = ballisticsStep(static_cast<double>(tgt), fast, slow, charAttCoeff_,
                                       c + outputLevel * (c * c - c));
                }
                const T gr = static_cast<T>(y);
                out[k] = gr;
                const T gl = (gr == T(0)) ? T(1) : decibelsToGain(gr);
                inDb[k] = gl;
                lastWet = dryIn[k] * gl;
            }
        }
        else
        {
            for (int k = 0; k < m; ++k)
            {
                const T tgt = holdAndRange(tgtA[k], range, holdN, held, counter, soft);
                const double y = ff.decoupled
                    ? decoupledStep(static_cast<double>(tgt), hold, fast, decHoldCoeff_)
                    : ballisticsStep(static_cast<double>(tgt), fast, slow, charAttCoeff_, charRelCoeff_);
                out[k] = static_cast<T>(y);
            }
        }

        envFastDb_[ch] = fast;
        envSlowDb_[ch] = slow;
        envHoldDb_[ch] = hold;
        upwardGuardDb_[ch] = guard;
        heldGrDb_[ch] = held;
        holdCounters_[ch] = counter;
    }

    /**
     * @brief Static curve plus character ballistics for one channel's state.
     *
     * Runs the Upward guard, the topology's gain law, hold/range and the
     * ballistics on channel `ch`'s state, and returns the smoothed gain
     * change in dB.
     *
     * @param inputDb Detected (linked) level in dB.
     * @param ch      Channel whose state is advanced.
     * @param cfg     Routing derived once per block.
     * @param thresh  Threshold in dB.
     * @param ratio   Ratio.
     * @param knee    Knee width in dB.
     */
    [[nodiscard]] T gainPath(T inputDb, int ch, const GainPathConfig& cfg,
                             T thresh, T ratio, T knee) noexcept
    {
        // Sustained-level guard envelope for Upward mode: instant rise,
        // constant 60 dB/s fall (peak hold with linear decay).
        T guardDb = inputDb;
        if (cfg.modeType == Mode::Upward)
        {
            T& g = upwardGuardDb_[ch];
            g = std::max(inputDb, g - upwardGuardDecay_);
            guardDb = g;
        }

        if (cfg.fbImplicit)
            return applyBallisticsFeedbackImplicit(inputDb, ch, cfg.charType, thresh, ratio, knee);

        if (cfg.topoEff == Topology::FeedBack)
        {
            // Explicit loop (detector with memory): calibrated element law
            // plus a per-sample stability floor on the ballistics. Upward
            // feedback inverts the law from the output level, so the loop
            // settles on the panel ratio too.
            const auto law = (cfg.modeType == Mode::Downward)
                ? computeGainFeedback(inputDb, thresh, ratio, knee, cfg.charType)
                : computeGainFeedbackUpward(inputDb, thresh, ratio, knee, guardDb);
            const T targetGR_Db = applyHoldAndRange(law.target, ch);
            return applyBallistics(targetGR_Db, ch, cfg.splitAdaptive, law.slope, cfg.loopDelay);
        }

        T targetGR_Db = computeGain(inputDb, thresh, ratio, knee, cfg.charType, cfg.modeType, guardDb);
        targetGR_Db = applyHoldAndRange(targetGR_Db, ch);
        return cfg.decoupled ? applyBallisticsDecoupled(targetGR_Db, ch, cfg.splitAdaptive)
                             : applyBallistics(targetGR_Db, ch, cfg.splitAdaptive);
    }

    /**
     * @brief Applies the frame's gain to one channel: lookahead, FET colour,
     *        feedback history, makeup and the parallel mix.
     */
    void applyGain(AudioBufferView<T>& audio, int ch, int i, T gainLin, T grDb,
                   int activeLookahead, Character charType, T colorAmt, T makeupLin,
                   T mixVal, T outLin) noexcept
    {
        T* data = audio.getChannel(ch);
        T input;
        if (activeLookahead > 0)
        {
            lookaheadBuffers_[ch].push(data[i]);
            input = lookaheadBuffers_[ch].read(activeLookahead);
        }
        else
        {
            input = data[i];
        }

        // outLin is the gain that is heard (gainLin scaled by the amount);
        // gainLin is the one the loop reads back.
        T wet = input * outLin;
        if (colorAmt > T(0) && charType == Character::FET)
        {
            // 2nd-order FET channel modulation: the drain-source resistance
            // bends with the signal across it, so colour only appears while
            // the FET conducts (gain reduction active); the squared term is
            // AC-coupled like the hardware's output transformer. Applied
            // pre-makeup: the line amp after the element is clean.
            const T sq = wet * wet;
            T& dc = fetDcState_[ch];
            if (std::isfinite(sq)) dc += fetDcCoeff_ * (sq - dc); // a non-finite sample would latch
            const T grDepth = std::clamp(-grDb * T(0.1), T(0), T(1));
            wet += colorAmt * kFetColorH2 * grDepth * (sq - dc);
        }
        // The feedback detector reads the compressed signal before makeup
        // (makeup sits after the element; feeding it back would shift the
        // static curve and run away with auto makeup).
        fbLastOutput_[ch] = (outLin == gainLin) ? wet : input * gainLin;
        wet *= makeupLin;

        // Parallel (New York) mix done inline: the dry reference is `input`,
        // which carries the SAME lookahead delay as the wet, so dry and wet
        // stay phase-aligned (the previous DryWetMixer captured the UNDELAYED
        // input and comb-filtered when lookahead + mix<1 were combined).
        data[i] = (mixVal < T(1)) ? (input * (T(1) - mixVal) + wet * mixVal) : wet;
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

        // One-pole coefficient for a time constant given in milliseconds
        // (double, like the envelopes it drives: a float one-pole stalls short
        // of its target once the per-sample step drops below an ulp).
        auto tc = [fs](T ms) { return std::exp(-1.0 / (static_cast<double>(fs) * static_cast<double>(ms) / 1000.0)); };
        const T prevWeight = charFastWeight_;

        // Character ballistics: time constants of the dB-domain envelopes.
        // The default arm backs up the setter's enum clamp: falling through
        // with no case would leave the coefficients zero-initialized
        // (instant ballistics, i.e. a waveshaper).
        switch (character_.load(std::memory_order_relaxed))
        {
            default:
            case Character::Clean:
            case Character::Varimu:
                charAttCoeff_     = tc(attMs);
                charRelCoeff_     = tc(relMs);
                charSlowRelCoeff_ = charRelCoeff_;
                charChargeCoeff_  = charAttCoeff_;
                charFastWeight_   = T(1); // single envelope
                {
                    // Decoupled detector (Clean, feed-forward): the smoothing
                    // pole's release side is a tenth of the release (never
                    // faster than the attack, never slower than half the
                    // release); the hold stage takes the rest of the release
                    // so the series t63 lands on the knob.
                    const T smoothMs = std::min(std::max(relMs * T(0.1), attMs), relMs * T(0.5));
                    decSmoothCoeff_ = tc(smoothMs);
                    decHoldCoeff_ = tc(static_cast<T>(decoupledHoldTau(
                        static_cast<double>(relMs), static_cast<double>(smoothMs))));
                }
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

        // The attack as it is observed (a feedback loop, below, changes the
        // coefficient and not the result), and the coefficients of an eighth
        // of a sample: the eight steps per sample of Precision::High.
        rawAttCoeff_ = charAttCoeff_;
        {
            const auto eighth = [](double c) { return std::pow(c, 1.0 / kSubsteps); };
            subAttCoeff_       = eighth(charAttCoeff_);
            subRelCoeff_       = eighth(charRelCoeff_);
            subSlowRelCoeff_   = eighth(charSlowRelCoeff_);
            subChargeCoeff_    = eighth(charChargeCoeff_);
            subDecHoldCoeff_   = eighth(decHoldCoeff_);
            subDecSmoothCoeff_ = eighth(decSmoothCoeff_);
        }

        // A live character change moves the fast/slow blend weight. Re-seat
        // the envelopes so the gain shown before and after the change is the
        // same (otherwise the blend jumps by up to the full reduction).
        if (charFastWeight_ != prevWeight)
        {
            const double w0 = static_cast<double>(prevWeight);
            const double w1 = static_cast<double>(charFastWeight_);
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                double& f = envFastDb_[static_cast<size_t>(ch)];
                double& s = envSlowDb_[static_cast<size_t>(ch)];
                const double shown = (w0 < 1.0) ? w0 * f + (1.0 - w0) * s : f;
                if (w1 >= 1.0) { f = shown; continue; }
                if (w0 >= 1.0) s = shown;
                f = (shown - (1.0 - w1) * s) / w1;
            }
        }

        // A feedback loop closes around the attack stage, which multiplies the
        // raw ballistics speed by (1 + loop gain). Re-derive the coefficient
        // from rho = coeff/(1 + w beta A) so the OBSERVED attack t63 stays on
        // the knob, exactly like the hardware panels (the 1176's 20-800 us
        // figures are measured results, not RC values). A is the
        // linear-region loop gain (R - 1) in both directions; Varimu uses its
        // base ratio (the progressive part is level-dependent).
        {
            const auto charNow = character_.load(std::memory_order_relaxed);
            const bool fbLoop = charNow == Character::FET
                             || topology_.load(std::memory_order_relaxed) == Topology::FeedBack;
            if (fbLoop)
            {
                const double loopGain = static_cast<double>(charFastWeight_)
                                      * (static_cast<double>(std::max(ratio_.load(std::memory_order_relaxed), T(1))) - 1.0);
                const double rho = charAttCoeff_;
                charAttCoeff_ = 1.0 - (1.0 - rho) / (1.0 + rho * loopGain);
                // The same for a step of an eighth of a sample: eight of
                // them then observe the knob's attack too.
                const double rhoSub = subAttCoeff_;
                subAttCoeff_ = 1.0 - (1.0 - rhoSub) / (1.0 + rhoSub * loopGain);
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
        latencySamples_ = std::max(lookaheadSamples_, static_cast<int>(fs * std::clamp(
            maxLookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000)));
        autoRefFall_ = kAutoRefFallDb / static_cast<double>(fs);

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
        return holdAndRange(targetGR_Db, rangeDb_.load(std::memory_order_relaxed), holdSamples_,
                            heldGrDb_[ch], holdCounters_[ch], softRange_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Hold and range on explicit state (shared by both block paths).
     * @param targetGR_Db Static gain change target in dB.
     * @param range       Maximum |gain change| in dB.
     * @param holdSamples Hold length in samples (0 = off).
     * @param held        Held gain change (state).
     * @param counter     Remaining hold samples (state).
     * @param soft        True to approach the range without a corner (setSoftRange()).
     */
    [[nodiscard]] static T holdAndRange(T targetGR_Db, T range, int holdSamples,
                                        T& held, int& counter, bool soft = false) noexcept
    {
        if (soft && range > T(0))
        {
            // g / (1 + (g / R)^4)^(1/4): g while it is small, R at most.
            const T x = targetGR_Db / range, x2 = x * x;
            targetGR_Db /= std::sqrt(std::sqrt(T(1) + x2 * x2));
        }
        else
        {
            targetGR_Db = std::clamp(targetGR_Db, -range, range);
        }

        if (holdSamples > 0)
        {
            if (std::abs(targetGR_Db) >= std::abs(held))
            {
                held = targetGR_Db;          // deeper action: re-arm the hold
                counter = holdSamples;
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

    /**
     * @brief Makeup dB to linear, cached: the makeup is shared by every
     * channel of a frame and constant unless automated or adaptive. Same
     * value as a direct decibelsToGain() call.
     * @param makeupDb Makeup gain in dB.
     * @return Linear makeup gain.
     */
    [[nodiscard]] T makeupToGain(T makeupDb) noexcept
    {
        if (makeupDb != cachedMakeupDb_)
        {
            cachedMakeupDb_ = makeupDb;
            cachedMakeupLin_ = decibelsToGain(makeupDb);
        }
        return cachedMakeupLin_;
    }

    // ---- Detectors ----

    /** @brief A non-finite sample reads as silence in the detector: it would
     *  otherwise latch in the recursive detector, filter and envelope state. */
    [[nodiscard]] static T detectorSafe(T x) noexcept
    {
        // |x| <= max is false for NaN and both infinities: the same test as
        // std::isfinite, as one compare (MSVC calls a library routine for it).
        return (std::abs(x) <= std::numeric_limits<T>::max()) ? x : T(0);
    }

    /**
     * @brief Computes level detection in Decibels.
     * @param sample Input sample.
     * @param ch Channel index (for state tracking).
     * @param detType Selected detector methodology.
     * @return Decibel representation of detected level.
     */
    [[nodiscard]] T detectLevel(T sample, int ch, DetectorType detType) noexcept
    {
        return gainToDecibels(detectLevelLinear(sample, ch, detType));
    }

    /**
     * @brief Computes the detected level as a linear magnitude (>= 0).
     * @param sample Input sample.
     * @param ch Channel index (for state tracking).
     * @param detType Selected detector methodology.
     * @return Linear detected level.
     */
    [[nodiscard]] T detectLevelLinear(T sample, int ch, DetectorType detType) noexcept
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

                if (len > 0 && len <= static_cast<int>(buf.size()))
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
                // The analytic magnitude is the envelope of a sustained tone,
                // but the allpass pair takes part of a cycle to form it after
                // an onset; the rectified input is never above the envelope of
                // a steady tone, so the larger of the two keeps the ripple-free
                // level and catches a step as fast as the peak detector.
                const double mag = hilbertDetectors_[ch].magnitude(static_cast<double>(sample));
                level = static_cast<T>(std::max(mag, std::abs(static_cast<double>(sample))));
                break;
            }
        }
        return level;
    }

    // ---- Gain curves ----

    /**
     * @brief Level-dependent effective ratio of the character (Varimu grows).
     * @param charType Compressor character whose ratio law is applied.
     * @param ratio Configured compression ratio.
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
     * @param guardDb Sustained level used to suppress upward gain in silence.
     * @return Target gain reduction in Decibels.
     */
    [[nodiscard]] T computeGain(T inputDb, T thresh, T ratio, T knee, Character charType, Mode modeType,
                                T guardDb) const noexcept
    {
        if (modeType == Mode::Upward)
            return computeGainUpward(inputDb, thresh, ratio, knee, guardDb);
        return downwardCurve(inputDb, thresh, ratio, knee, charType);
    }

    /**
     * @brief Downward static curve with the character's knee floor and ratio
     *        law (see computeGain).
     */
    [[nodiscard]] static T downwardCurve(T inputDb, T thresh, T ratio, T knee, Character charType) noexcept
    {
        const T effectiveRatio = effectiveRatioFor(charType, ratio, inputDb - thresh);
        const T effectiveKnee = effectiveKneeFor(charType, knee);
        const T slope = T(1) - T(1) / effectiveRatio;

        // Branch-free pieces (the level crosses the threshold at random on
        // dense programme, where a branch per sample mispredicts): below the
        // threshold the linear piece is positive and min() returns 0.
        if (effectiveKnee <= T(0))
            return std::min(thresh - inputDb, T(0)) * slope;

        // Soft knee: the quadratic below the knee top never exceeds the
        // linear piece, and with x clamped to the knee width it is the larger
        // one above it, so min() picks the right piece on every side.
        const T halfKnee = effectiveKnee / T(2);
        const T x = std::clamp(inputDb - (thresh - halfKnee), T(0), effectiveKnee);
        const T quad = T(0) - slope * x * x / (T(2) * effectiveKnee);
        return std::min((thresh - inputDb) * slope, quad);
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
     * @param thresh Threshold parameter in dB.
     * @param ratio Configured compression ratio.
     * @param knee Soft-knee width parameter in dB.
     * @param charType Compressor character whose static law is applied.
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

    /**
     * @brief Upward element law for feedback detection, per output level.
     *
     * The inverse of the feed-forward upward curve, as elementLaw() is for
     * the downward one: with the detector on the output, a raw feed-forward
     * law could never observe more than 2:1. Below the knee the boost is
     * (R - 1) * (threshold - out), so the settled loop lands on the panel
     * ratio; the soft knee is inverted in closed form. The silence guard
     * reads the (output) level the detector sees.
     *
     * @param outDb   Detected output level (dB).
     * @param thresh  Threshold in dB.
     * @param ratio   Configured ratio (>= 1).
     * @param knee    Knee width in dB (>= 0).
     * @param guardDb Sustained level for the silence guard.
     */
    [[nodiscard]] static FeedbackLaw computeGainFeedbackUpward(T outDb, T thresh, T ratio, T knee,
                                                               T guardDb) noexcept
    {
        const T r = std::max(ratio, T(1));
        const T a = r - T(1);
        const T guard = std::clamp((guardDb - (thresh - T(60))) / T(20), T(0), T(1));
        if (knee <= T(0))
        {
            const T e = thresh - outDb;
            if (e <= T(0)) return { T(0), T(0) };
            return { a * e * guard, a * guard };
        }
        const T s = a / r;
        const T eu = thresh + knee / T(2) - outDb;   // distance below the knee top
        if (eu <= T(0)) return { T(0), T(0) };
        if (eu >= knee * (T(1) - s / T(2)))
            return { a * (thresh - outDb) * guard, a * guard };
        // Knee: invert upper - out = v - s v^2 / (2W) for the input-side
        // depth v in (0, W) (cancellation-free root).
        const T z = std::min(T(2) * s * eu / knee, T(1));
        const T v = std::min(T(2) * eu / (T(1) + std::sqrt(T(1) - z)), knee);
        const T sv = s * v / knee;
        return { s * v * v / (T(2) * knee) * guard, sv / std::max(T(1) - sv, T(1) / r) * guard };
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
     * @param thresh Threshold parameter in dB.
     * @param ratio Configured compression ratio.
     * @param knee Soft-knee width parameter in dB.
     * @param charType Compressor character whose static law is applied.
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
     * @param ch Processing channel index.
     * @param charType Compressor character whose ballistics are applied.
     * @param thresh Threshold parameter in dB.
     * @param ratio Configured compression ratio.
     * @param knee Soft-knee width parameter in dB.
     */
    [[nodiscard]] T applyBallisticsFeedbackImplicit(T inDb, int ch, Character charType,
                                                    T thresh, T ratio, T knee) noexcept
    {
        const LoopCoeffs coeffs { charAttCoeff_, charRelCoeff_, charChargeCoeff_, charSlowRelCoeff_ };
        return boundImplicit(stepImplicit(inDb, ch, charType, thresh, ratio, knee, coeffs), ch);
    }

    /** @brief Coefficients of one step of the semi-implicit loop. */
    struct LoopCoeffs
    {
        double attack;       ///< Fast stage while the reduction deepens.
        double release;      ///< Fast stage while it recovers.
        double charge;       ///< Memory stage while it charges.
        double slowRelease;  ///< Memory stage while it discharges.
    };

    /** @brief One step of the semi-implicit loop on the channel's envelopes:
     *  the blended gain it leaves, before hold and range. */
    [[nodiscard]] T stepImplicit(T inDb, int ch, Character charType, T thresh, T ratio, T knee,
                                 const LoopCoeffs& coeffs) noexcept
    {
        double& fast = envFastDb_[ch];
        double& slow = envSlowDb_[ch];
        const double wFast = static_cast<double>(charFastWeight_);
        const bool dual = wFast < 1.0;

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
            const double st = static_cast<double>(slowTarget);
            const bool charging = st < slow;
            slow = st + (charging ? coeffs.charge : coeffs.slowRelease) * (slow - st);
            if (atRest(slow, st)) slow = 0.0;
        }

        // Branch on the direction the loop is about to move: attack while
        // the reduction deepens at the previous gain, release otherwise.
        const T shownPrev = static_cast<T>(dual ? wFast * fast + (1.0 - wFast) * slow : fast);
        const T probe = computeGainFeedback(inDb + shownPrev, thresh, ratio, knee, charType).target;
        const bool engaging = static_cast<double>(probe) < fast;
        const double coeff = engaging ? coeffs.attack : coeffs.release;
        const double beta = 1.0 - coeff;

        const T k = static_cast<T>(dual ? wFast * coeff * fast + (1.0 - wFast) * slow : coeff * fast);
        const T effB = static_cast<T>(dual ? wFast * beta : beta);
        const auto sol = solveFeedbackGain(inDb, k, effB, thresh, ratio, knee, charType);

        const double solT = static_cast<double>(sol.target);
        fast = solT + coeff * (fast - solT);
        if (atRest(fast, solT)) fast = 0.0;
        return static_cast<T>(dual ? wFast * fast + (1.0 - wFast) * slow : fast);
    }

    /**
     * @brief Hold and range of the semi-implicit loop.
     *
     * They act on the OBSERVED gain: the element's internal target is
     * loop-amplified by up to (R - 1) during transients, so clamping THAT to
     * the user's range would throttle the attack, and holding it would
     * compare mismatched domains. When they override, the fast stage is
     * rewritten so the blend lands on the bound.
     */
    [[nodiscard]] T boundImplicit(T shown, int ch) noexcept
    {
        const T bounded = applyHoldAndRange(shown, ch);
        if (bounded != shown)
        {
            const double wFast = static_cast<double>(charFastWeight_);
            const double sh = static_cast<double>(bounded);
            envFastDb_[ch] = (wFast < 1.0) ? (sh - (1.0 - wFast) * envSlowDb_[ch]) / wFast : sh;
        }
        return bounded;
    }

    /**
     * @brief An envelope within 1e-4 dB of no gain change, heading there: it
     *        is put on it, to end the asymptotic tail of a release.
     *
     * The target has to be at rest too. Asking only the envelope (as this
     * did) also caught an envelope leaving zero in steps shorter than that:
     * with a slow attack or a slow charge of the memory stage and a shallow
     * target, each step was undone and the gain never moved (100 ms of
     * attack at 96 kHz ignored any target under 1 dB; the memory stage of
     * Opto with a release of 500 ms, any under 9.6 dB at 48 kHz).
     */
    [[nodiscard]] static bool atRest(double envelope, double target) noexcept
    {
        return std::abs(envelope) < 1e-4 && std::abs(target) < 1e-4;
    }

    // ---- Ballistics (log domain) ----

    /**
     * @brief Smooths the static gain target with the active character's ballistics.
     *
     * Operates on the gain change in dB. Varimu, and Clean with Branching
     * ballistics or in feedback, run a branching one-pole (attack while the
     * gain moves away from unity, release while it recovers). Opto/FET blend
     * a fast envelope with a slow one that charges with compression history,
     * producing the two-stage program-dependent release of the modeled
     * hardware: brief peaks release fast because the slow stage never
     * charged, sustained compression leaves a long tail.
     *
     * @param targetGrDb Static gain change target in dB (negative = reduction).
     * @param ch Channel index.
     * @param splitAdaptive True when the EFFECTIVE detector is SplitPolarity
     *        (the callers derive it once per block/sample: it must follow the
     *        detector actually in use, which the FET character overrides).
     * @param loopSlope Incremental loop gain when driven from a feedback
     *        detector with memory (explicit loop): both coefficients get a
     *        stability floor, which caps the per-sample loop advance at 0.5
     *        and keeps the one-sample-delay iteration from ringing at extreme
     *        ratio/time combinations. It applies at any slope, 0 included:
     *        below the knee the loop is flat, but a release faster than the
     *        detector's delay still overshoots the settled gain. Negative =
     *        no loop (default).
     * @param loopDelay Delay of the detector inside that loop, in samples
     *        (see feedbackDetectorDelay()). A loop that advances faster than
     *        its detector can report the result overshoots and rings, so the
     *        floor also caps the release advance at 1 / (e (1 + delay)), the
     *        critically damped limit of a first-order loop around a delay,
     *        and the attack advance at twice that.
     * @return The smoothed gain change in dB to apply.
     */
    [[nodiscard]] T applyBallistics(T targetGrDb, int ch, bool splitAdaptive,
                                    T loopSlope = T(-1), double loopDelay = 0.0) noexcept
    {
        // Level-adaptive release for the SplitPolarity detector: full-scale
        // output halves the release time constant. Halving tau squares the
        // one-pole coefficient, so blending between the two valid endpoints
        // keeps the modulation on the exponential curve.
        double attCoeff = charAttCoeff_;
        double relCoeff = charRelCoeff_;
        if (splitAdaptive)
        {
            // Written so a non-finite previous output reads as full scale
            // instead of turning the coefficient (and the envelope) into NaN.
            const double lastOut = std::abs(static_cast<double>(fbLastOutput_[ch]));
            const double outputLevel = (lastOut < 1.0) ? lastOut : 1.0;
            relCoeff += outputLevel * (relCoeff * relCoeff - relCoeff);
        }
        if (loopSlope >= T(0))
        {
            // Cap the per-sample loop advance at 0.5 (half-deadbeat): the
            // one-sample delay then converges monotonically instead of
            // sustaining a marginal dance on the detector's residual ripple.
            // A detector with a delay (the RMS window) lowers the cap to the
            // critically damped advance of a loop around that delay. On a
            // level fall the window's mean square stays up until the old
            // loud samples leave it (a full window of delay); on a rise the
            // new loud samples dominate it within a fraction of the window,
            // so the attack side tolerates twice the advance (measured: no
            // over-compression at 2x, 0.07 dB at 2.5x).
            const double lim = (loopDelay > 0.0) ? 1.0 / (std::numbers::e * (1.0 + loopDelay)) : 0.5;
            const double attAdvance = std::min(0.5, 2.0 * lim);
            const double relAdvance = std::min(0.5, lim);
            const double loop = 1.0 + static_cast<double>(loopSlope);
            attCoeff = std::max(attCoeff, 1.0 - attAdvance / loop);
            relCoeff = std::max(relCoeff, 1.0 - relAdvance / loop);
        }

        return static_cast<T>(ballisticsStep(static_cast<double>(targetGrDb), envFastDb_[ch],
                                             envSlowDb_[ch], attCoeff, relCoeff));
    }

    /**
     * @brief Branch-free branching one-pole step toward `tgt`: coefficient
     *        `cDown` while the state is above the target (the gain falls),
     *        `cUp` otherwise.
     *
     * The two candidate steps are ordered by the coefficients: with
     * cDown <= cUp the falling step is the smaller one when the state is
     * above the target and the rising step the smaller one when it is below,
     * so min() selects the right branch (max() when cDown > cUp). Same value
     * as the branch, without a data-dependent jump per sample.
     */
    [[nodiscard]] static double branchingStep(double y, double tgt, double cDown, double cUp) noexcept
    {
        const double d = y - tgt;
        const double down = tgt + cDown * d;
        const double up = tgt + cUp * d;
        return (cDown <= cUp) ? std::min(down, up) : std::max(down, up);
    }

    /**
     * @brief Branching ballistics on explicit state (shared by both block
     *        paths): the fast envelope, plus the Opto/FET memory stage.
     *
     * Attack acts while the gain FALLS (louder signal: deeper reduction, or
     * less upward boost); release acts while it recovers upward. A magnitude
     * comparison would invert the two in Upward mode and make the envelope
     * chase the huge boosts of the waveform's zero crossings.
     */
    [[nodiscard]] double ballisticsStep(double tgt, double& fast, double& slow,
                                        double attCoeff, double relCoeff) const noexcept
    {
        return ballisticsCore(tgt, fast, slow, attCoeff, relCoeff, charChargeCoeff_, charSlowRelCoeff_);
    }

    /** @brief ballisticsStep() with the coefficients of the memory stage given too. */
    [[nodiscard]] double ballisticsCore(double tgt, double& fast, double& slow, double attCoeff,
                                        double relCoeff, double chargeCoeff,
                                        double slowRelCoeff) const noexcept
    {
        fast = branchingStep(fast, tgt, attCoeff, relCoeff);
        if (atRest(fast, tgt)) fast = 0.0; // end the asymptotic dB tail

        if (charFastWeight_ >= T(1)) // Clean / Varimu: single envelope
            return fast;

        // Opto / FET memory stage: charges toward the target over the history
        // time constant and discharges with the slow release, so its level
        // encodes how long (and how deep) the compressor has been working.
        slow = branchingStep(slow, tgt, chargeCoeff, slowRelCoeff);
        if (atRest(slow, tgt)) slow = 0.0;

        const double w = static_cast<double>(charFastWeight_);
        return w * fast + (1.0 - w) * slow;
    }

    /**
     * @brief The adaptive makeup envelope only runs while adaptive makeup is
     *        active; when it engages it starts from the current gain change
     *        (the settled value of a tracker that had kept running). Audio
     *        thread only.
     */
    void syncAdaptiveMakeup(bool adaptive) noexcept
    {
        if (adaptive && !adaptiveMakeupActive_)
            autoMakeupEnv_ = gainReductionDb_.load(std::memory_order_relaxed);
        adaptiveMakeupActive_ = adaptive;
    }

    /**
     * @brief Delay of the detector as an explicit feedback loop sees it, in
     *        samples: the RMS window (its reading of a level step completes
     *        one window later). The other detectors respond within a few
     *        samples, which the base cap of the loop advance already covers.
     */
    [[nodiscard]] double feedbackDetectorDelay(DetectorType detType) const noexcept
    {
        return (detType == DetectorType::Rms) ? static_cast<double>(rmsWindowSamples_) : 0.0;
    }

    /** @brief The Clean character in feed-forward runs the decoupled detector
     *  when setCleanBallistics() selected it. */
    [[nodiscard]] bool useDecoupled(Character charType, Topology topoEff) const noexcept
    {
        return charType == Character::Clean && topoEff == Topology::FeedForward
            && cleanBallistics_.load(std::memory_order_relaxed) == CleanBallistics::Decoupled;
    }

    /**
     * @brief Seats the decoupled hold stage on the current gain when the
     *        decoupled path engages (character or topology change), so the
     *        switch does not replay a stale hold value. Audio thread only.
     */
    void syncDecoupledState(bool decoupled) noexcept
    {
        if (decoupled && !decoupledActive_)
            for (int ch = 0; ch < kMaxChannels; ++ch)
                envHoldDb_[static_cast<size_t>(ch)] = envFastDb_[static_cast<size_t>(ch)];
        decoupledActive_ = decoupled;
    }

    /**
     * @brief Decoupled ballistics for the Clean character in feed-forward
     *        operation (after the smooth decoupled peak detector of
     *        Giannoulis, Massberg and Reiss, JAES 2012).
     *
     * A hold stage follows a deeper target at once and releases toward a
     * shallower one; a smoothing stage then follows the held envelope with
     * the attack pole while it deepens and a release-side pole (a tenth of
     * the release, see updateTimeConstants) while it recovers. Branching
     * ballistics release toward every dip of a rectified waveform (each zero
     * crossing of a tone), so the gain rides the waveform; here the dips
     * only shave the held level by its slow release, and the smoothing stage
     * filters that small sawtooth in both directions. A 1 kHz tone 14 dB
     * over a 4:1 threshold: THD+N -65.5 dB branching against -102 dB here
     * at 5/100 ms (-54 against -80 dB at 1/30 ms, -57 against -84 dB at
     * 0.1/50 ms), and the gain sits on the peak-referenced static curve
     * instead of up to 0.8 dB above it.
     *
     * Timing contract: the attack is the attack pole alone (the hold stage
     * is instant), so its t63 is the knob. On release the two stages are in
     * series; updateTimeConstants() solves the hold stage's time constant so
     * the series release still passes t63 at the release knob. The series
     * rounds the first part of the release (at 5/100 ms the curve stays
     * within 0.8 dB of the exponential on a 12 dB release). When the attack
     * is longer than half the release, the release-side pole is capped at
     * half the release, so a release can still be shorter than the attack.
     *
     * @param targetGrDb Static gain change target in dB (negative = reduction).
     * @param ch Channel index.
     * @param splitAdaptive True when the effective detector is SplitPolarity
     *        (its level-adaptive release then shortens the hold stage).
     * @return The smoothed gain change in dB to apply.
     */
    [[nodiscard]] T applyBallisticsDecoupled(T targetGrDb, int ch, bool splitAdaptive) noexcept
    {
        double holdCoeff = decHoldCoeff_;
        if (splitAdaptive)
        {
            const double lastOut = std::abs(static_cast<double>(fbLastOutput_[ch]));
            const double outputLevel = (lastOut < 1.0) ? lastOut : 1.0;
            holdCoeff += outputLevel * (holdCoeff * holdCoeff - holdCoeff);
        }

        return static_cast<T>(decoupledStep(static_cast<double>(targetGrDb), envHoldDb_[ch],
                                            envFastDb_[ch], holdCoeff));
    }

    /** @brief One decoupled step on explicit state (see applyBallisticsDecoupled). */
    [[nodiscard]] double decoupledStep(double tgt, double& hold, double& fast,
                                       double holdCoeff) const noexcept
    {
        return decoupledCore(tgt, hold, fast, holdCoeff, charAttCoeff_, decSmoothCoeff_);
    }

    /** @brief decoupledStep() with the coefficients of the smoothing stage given too. */
    [[nodiscard]] static double decoupledCore(double tgt, double& hold, double& fast, double holdCoeff,
                                              double attCoeff, double smoothCoeff) noexcept
    {
        // Hold stage: a deeper (lower) target is taken at once; otherwise
        // release toward the target. Same direction rule as the branching
        // ballistics: a falling gain is the attack side in both modes.
        hold = std::min(tgt, tgt + holdCoeff * (hold - tgt));
        if (atRest(hold, tgt)) hold = 0.0; // end the asymptotic dB tail

        // Smoothing stage: the attack pole while the held gain is below the
        // output (deepening), the release-side pole otherwise (never slower
        // than the attack, see updateTimeConstants).
        fast = branchingStep(fast, hold, attCoeff, smoothCoeff);
        if (atRest(fast, hold)) fast = 0.0;
        return fast;
    }

    /**
     * @brief Time constant of the decoupled hold stage (ms) so that the series
     *        release of the hold stage and the smoothing pole passes t63 at
     *        `releaseMs`.
     *
     * Series step-down response of two one-poles tau1, tau2:
     * (tau1 e^(-t/tau1) - tau2 e^(-t/tau2)) / (tau1 - tau2), increasing in
     * tau1; bisected (setup rate, not per sample).
     */
    [[nodiscard]] static double decoupledHoldTau(double releaseMs, double smoothMs) noexcept
    {
        const double target = std::exp(-1.0);
        auto remaining = [&](double tau1) {
            const double t = releaseMs, t2 = smoothMs;
            if (std::abs(tau1 - t2) < 1e-9 * t)
                return (1.0 + t / t2) * std::exp(-t / t2);
            return (tau1 * std::exp(-t / tau1) - t2 * std::exp(-t / t2)) / (tau1 - t2);
        };
        double lo = 1e-6 * releaseMs, hi = releaseMs;
        for (int it = 0; it < 48; ++it)
        {
            const double mid = 0.5 * (lo + hi);
            if (remaining(mid) < target) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    // ---- Precision of the control signal (setPrecision) ----

    /** @brief Whether the control signal runs at Precision::High with this
     *  routing: feed-forward, and feedback solved against the input. */
    [[nodiscard]] bool usePrecision(Topology topoEff, bool fbImplicit, bool splitAdaptive) const noexcept
    {
        return precision_.load(std::memory_order_relaxed) == Precision::High && !splitAdaptive
            && (topoEff == Topology::FeedForward || fbImplicit);
    }

    /** @brief The detectors that rectify: at Precision::High they read the
     *  key between the samples too. */
    [[nodiscard]] static bool readsBetweenSamples(DetectorType detType) noexcept
    {
        return detType == DetectorType::Peak || detType == DetectorType::TruePeak;
    }

    /**
     * @brief Seats the history of the precise path when it engages: it starts
     *        from the gain that is sounding and from silence at the detector.
     *        Audio thread only.
     */
    void syncPrecisionState(bool precise) noexcept
    {
        if (precise && !precisionActive_)
        {
            const double w = static_cast<double>(charFastWeight_);
            for (size_t ch = 0; ch < static_cast<size_t>(kMaxChannels); ++ch)
            {
                interp_[ch] = {};
                precLevel_[ch] = -100.0;
                precTarget_[ch] = (w < 1.0) ? w * envFastDb_[ch] + (1.0 - w) * envSlowDb_[ch]
                                            : envFastDb_[ch];
            }
        }
        precisionActive_ = precise;
    }

    /**
     * @brief The rectified key at kDetectorPoints points per sample.
     *
     * Feeds the channel's history and writes, in time order, the magnitude
     * of the key at a quarter, a half and three quarters of the way from one
     * sample to the next (a windowed-sinc interpolation of kInterpTaps
     * samples) and at that next sample itself, which is kDetectorDelay
     * samples old.
     *
     * @param sample The key (after the sidechain filter).
     * @param ch     Channel whose history is advanced.
     * @param points Receives kDetectorPoints magnitudes.
     */
    void detectorPoints(T sample, int ch, T* points) noexcept
    {
        auto& h = interp_[static_cast<size_t>(ch)];
        // Mirrored ring: the window of the last kInterpTaps samples is
        // contiguous, oldest first.
        h.x[h.at] = sample;
        h.x[h.at + kInterpTaps] = sample;
        const T* window = &h.x[h.at + 1];
        h.at = (h.at + 1) & (kInterpTaps - 1);
        for (int p = 0; p < kDetectorPoints - 1; ++p)
            points[p] = std::abs(simd::dotProduct(interpPhase_[p], window, kInterpTaps));
        points[kDetectorPoints - 1] = std::abs(window[kInterpTaps / 2]);
    }

    /** @brief The gain as it is heard with a latency kept for the lookahead:
     *  older by the difference (see setMaxLookahead()). */
    [[nodiscard]] T heldBack(T grDb, int g, int delay) noexcept
    {
        if (delay <= 0) return grDb;
        auto& line = gainDelay_[static_cast<size_t>(g)];
        line.push(grDb);
        return line.read(delay);
    }

    /**
     * @brief The gain of one frame at Precision::High, for one gain computer.
     *
     * The frame brings `count` levels in time order (kDetectorPoints from a
     * detector read between the samples, one otherwise). The ballistics take
     * kSubsteps steps in all, with the coefficients of that share of a
     * sample: in feed-forward towards a target that moves along the line
     * from one level's target to the next; in the loop solved against the
     * input, with the level moving along that line and the loop solved at
     * each step. The gain is the mean of the steps. Hold and range act once
     * per level in feed-forward and once per frame, on the gain the last
     * step leaves, in the loop.
     *
     * @param levelDb Levels in dB, oldest first.
     * @param count   How many (1 or kDetectorPoints).
     * @param ch      Channel whose state is advanced.
     * @param cfg     Routing derived once per block.
     * @param thresh  Threshold in dB.
     * @param ratio   Ratio.
     * @param knee    Knee width in dB.
     */
    [[nodiscard]] T gainPathPrecise(const T* levelDb, int count, int ch, const GainPathConfig& cfg,
                                    T thresh, T ratio, T knee) noexcept
    {
        const int stepsEach = kSubsteps / count;
        const double share = 1.0 / stepsEach;
        const size_t c = static_cast<size_t>(ch);
        double sum = 0.0;

        if (cfg.fbImplicit)
        {
            const LoopCoeffs coeffs { subAttCoeff_, subRelCoeff_, subChargeCoeff_, subSlowRelCoeff_ };
            double last = precLevel_[c];
            T shown = T(0);
            for (int j = 0; j < count; ++j)
            {
                const double to = static_cast<double>(levelDb[j]);
                for (int s = 1; s <= stepsEach; ++s)
                {
                    const double at = last + (to - last) * (s * share);
                    shown = stepImplicit(static_cast<T>(at), ch, cfg.charType, thresh, ratio, knee, coeffs);
                    sum += static_cast<double>(shown);
                }
                last = to;
            }
            precLevel_[c] = last;
            const T bounded = boundImplicit(shown, ch);
            return (bounded != shown) ? bounded : static_cast<T>(sum * (1.0 / kSubsteps));
        }

        // Sustained-level guard of the Upward curve, on the frame's top level.
        T guardDb = levelDb[0];
        for (int j = 1; j < count; ++j) guardDb = std::max(guardDb, levelDb[j]);
        if (cfg.modeType == Mode::Upward)
        {
            T& guard = upwardGuardDb_[c];
            guard = std::max(guardDb, guard - upwardGuardDecay_);
            guardDb = guard;
        }

        const T range = rangeDb_.load(std::memory_order_relaxed);
        const bool soft = softRange_.load(std::memory_order_relaxed);
        const int holdLength = holdSamples_ * count; // the hold counts levels here
        double last = precTarget_[c];
        double& hold = envHoldDb_[c];
        double& fast = envFastDb_[c];
        double& slow = envSlowDb_[c];
        for (int j = 0; j < count; ++j)
        {
            const T target = holdAndRange(
                computeGain(levelDb[j], thresh, ratio, knee, cfg.charType, cfg.modeType, guardDb),
                range, holdLength, heldGrDb_[c], holdCounters_[c], soft);
            const double to = static_cast<double>(target);
            for (int s = 1; s <= stepsEach; ++s)
            {
                const double at = last + (to - last) * (s * share);
                sum += cfg.decoupled
                    ? decoupledCore(at, hold, fast, subDecHoldCoeff_, subAttCoeff_, subDecSmoothCoeff_)
                    : ballisticsCore(at, fast, slow, subAttCoeff_, subRelCoeff_, subChargeCoeff_,
                                     subSlowRelCoeff_);
            }
            last = to;
        }
        precTarget_[c] = last;
        return static_cast<T>(sum * (1.0 / kSubsteps));
    }

    // ---- Analytic makeup ----

    /** @brief Everything the analytic makeup depends on: it is worked out
     *  again only when one of these changes. */
    struct MakeupModelKey
    {
        T thresh = T(0);            ///< Threshold (dB).
        T ratio = T(1);             ///< Ratio.
        T knee = T(0);              ///< Knee width (dB).
        T amount = T(1);            ///< Share of the gain change that is applied.
        T range = T(0);             ///< Largest gain change (dB).
        T fastWeight = T(1);        ///< Blend of the fast and memory stages.
        T window = T(0);            ///< Length of the RMS window (ms).
        double attack = 0.0;        ///< Observed attack coefficient.
        double release = 0.0;       ///< Release coefficient.
        double charge = 0.0;        ///< Charge coefficient of the memory stage.
        double slowRelease = 0.0;   ///< Release coefficient of the memory stage.
        double hold = 0.0;          ///< Release coefficient of the decoupled hold stage.
        Character charType = Character::Clean;       ///< Character.
        Mode modeType = Mode::Downward;              ///< Direction.
        DetectorType detType = DetectorType::Peak;   ///< Effective detector.
        bool decoupled = false;     ///< Decoupled ballistics.
        bool loop = false;          ///< The detector reads the output (feedback).
        bool autoThreshold = false; ///< The threshold counts from the key's peak.
        bool soft = false;          ///< Range with no corner.
        bool valid = false;         ///< False until the first solve.

        bool operator==(const MakeupModelKey&) const = default;
    };

    /**
     * @brief Target of the analytic makeup, in dB, for a routing that is
     *        already known (the solve only runs when a setting changed).
     */
    [[nodiscard]] T analyticMakeupDb(Character charType, Mode modeType, DetectorType detType,
                                     bool decoupled, bool loop) noexcept
    {
        MakeupModelKey key;
        key.thresh = threshold_.load(std::memory_order_relaxed);
        key.ratio = std::max(ratio_.load(std::memory_order_relaxed), T(1));
        key.knee = std::max(kneeWidth_.load(std::memory_order_relaxed), T(0));
        key.amount = amount_.load(std::memory_order_relaxed);
        key.range = rangeDb_.load(std::memory_order_relaxed);
        key.fastWeight = charFastWeight_;
        key.window = rmsWindowMsAtomic_.load(std::memory_order_relaxed);
        key.attack = rawAttCoeff_;
        key.release = charRelCoeff_;
        key.charge = charChargeCoeff_;
        key.slowRelease = charSlowRelCoeff_;
        key.hold = decHoldCoeff_;
        key.charType = charType;
        key.modeType = modeType;
        key.detType = detType;
        key.decoupled = decoupled;
        key.loop = loop;
        key.autoThreshold = autoThreshold_.load(std::memory_order_relaxed);
        key.soft = softRange_.load(std::memory_order_relaxed);
        key.valid = true;
        if (!(key == makeupKey_))
        {
            makeupKey_ = key;
            makeupModelDb_ = solveMakeupModel(key);
        }
        return makeupModelDb_;
    }

    /** @brief Target of the analytic makeup for the settings as they stand
     *  (0 dB when another makeup mode is selected). */
    [[nodiscard]] T analyticTarget() noexcept
    {
        if (autoMakeupMode_.load(std::memory_order_relaxed) != AutoMakeupMode::Analytic) return T(0);
        const Character charType = character_.load(std::memory_order_relaxed);
        const bool fet = charType == Character::FET;
        const Topology topoEff = fet ? Topology::FeedBack : topology_.load(std::memory_order_relaxed);
        const DetectorType detType = fet ? DetectorType::Peak : detectorType_.load(std::memory_order_relaxed);
        return analyticMakeupDb(charType, mode_.load(std::memory_order_relaxed), detType,
                                useDecoupled(charType, topoEff), topoEff == Topology::FeedBack);
    }

    /**
     * @brief Where a branching follower rests over a set of weighted targets.
     *
     * Each sample it moves towards the target by a share of the distance:
     * `attackRate` while the target is below it, `releaseRate` while it is
     * above. Over targets that change much faster than it moves, it rests
     * where the two pulls cancel: attackRate x mean distance to the targets
     * below = releaseRate x mean distance to those above. That point (an
     * expectile of the targets) is found by bisection.
     *
     * @param target      Gain changes (dB).
     * @param weight      How often each one is met.
     * @param count       How many there are.
     * @param attackRate  1 - attack coefficient.
     * @param releaseRate 1 - release coefficient.
     */
    [[nodiscard]] static double restingGain(const double* target, const double* weight, int count,
                                            double attackRate, double releaseRate) noexcept
    {
        double lo = target[0], hi = target[0];
        for (int i = 1; i < count; ++i)
        {
            lo = std::min(lo, target[i]);
            hi = std::max(hi, target[i]);
        }
        for (int pass = 0; pass < 48 && hi - lo > 1e-6; ++pass)
        {
            const double at = 0.5 * (lo + hi);
            double below = 0.0, above = 0.0;
            for (int i = 0; i < count; ++i)
            {
                const double d = target[i] - at;
                if (d < 0.0) below -= weight[i] * d;
                else above += weight[i] * d;
            }
            if (attackRate * below > releaseRate * above) hi = at;
            else lo = at;
        }
        return 0.5 * (lo + hi);
    }

    /**
     * @brief Spread of what an RMS window reads over dense programme: the
     *        standard deviation of its level, in dB.
     *
     * A fit to pink noise (equal power per octave), whose low end a short
     * window cannot average: 1.6 dB for 10 ms, falling with the cube root of
     * the length (measured 2.1, 1.6, 1.1, 0.7 and 0.5 dB at 3, 10, 30, 100
     * and 300 ms).
     */
    [[nodiscard]] static double windowSpreadDb(double windowMs) noexcept
    {
        return 1.6 * std::cbrt(10.0 / std::clamp(windowMs, 1.0, 500.0));
    }

    /**
     * @brief How far over the RMS of dense programme the reference of the
     *        automatic threshold rests, for each detector (dB).
     *
     * Measured on pink noise with the reference's own law (it rises at once
     * and falls 1 dB per second). The RMS window's rests about three of its
     * spreads up while the window is short; a long window moves its level as
     * slowly as the reference falls, and the reference then rests lower.
     */
    [[nodiscard]] static double referenceOverRmsDb(const MakeupModelKey& key) noexcept
    {
        switch (key.detType)
        {
            case DetectorType::Rms:
            {
                const double ms = std::clamp(static_cast<double>(key.window), 1.0, 500.0);
                return windowSpreadDb(ms) * 3.2 / (1.0 + ms / 130.0);
            }
            case DetectorType::Hilbert:  return kEnvelopeReferenceDb;
            case DetectorType::TruePeak: return kTruePeakReferenceDb;
            default:                     return kPeakReferenceDb;
        }
    }

    /**
     * @brief The analytic makeup: minus the gain the ballistics settle on
     *        over dense programme.
     *
     * The programme is a Gaussian signal as the detector reads it. For the
     * detectors that rectify, the magnitudes of its samples (a half normal);
     * for TruePeak the largest of kTruePeakSamples of them (the four points
     * between samples are not independent); for Hilbert its envelope (a
     * Rayleigh, kEnvelopeLiftDb higher since the detector also takes the
     * rectified sample); for the RMS window a log-normal about the RMS with
     * the spread of windowSpreadDb(). Each level gives a target on the static
     * curve (with the range), and the followers rest among those targets as
     * restingGain() says: the fast stage with attack and release, the memory
     * stage of Opto and FET with its charge and its release, the decoupled
     * hold with an attack of one sample. A feedback loop works against the
     * lag of the memory stage with its own gain, w (R - 1), so there that
     * stage only counts for (1 - w) / (1 + w (R - 1)). With the automatic
     * threshold the levels count from the peak its reference rests on;
     * without it the programme is at kNominalRmsDb.
     */
    [[nodiscard]] T solveMakeupModel(const MakeupModelKey& key) const noexcept
    {
        const double rmsDb = key.autoThreshold ? -referenceOverRmsDb(key) : kNominalRmsDb;
        const bool window = key.detType == DetectorType::Rms;
        const bool envelope = key.detType == DetectorType::Hilbert;
        const bool between = key.detType == DetectorType::TruePeak;
        const double spread = window ? windowSpreadDb(static_cast<double>(key.window)) : 0.0;
        double target[kMakeupNodes], weight[kMakeupNodes];
        T held = T(0);
        int counter = 0;
        for (int i = 0; i < kMakeupNodes; ++i)
        {
            double levelDb = rmsDb;
            if (window)
            {
                // Equal steps of the level from -4 to +4 spreads; the mean of the squares stays the RMS.
                const double z = ((static_cast<double>(i) + 0.5) / kMakeupNodes - 0.5) * 8.0;
                weight[i] = std::exp(-0.5 * z * z);
                levelDb += spread * z - 0.1151 * spread * spread;
            }
            else
            {
                // In units of the RMS: the middle of each of the equal steps up to kMakeupReach.
                const double t = (static_cast<double>(i) + 0.5) * (kMakeupReach / kMakeupNodes);
                weight[i] = std::exp(-0.5 * t * t);
                if (envelope) weight[i] *= t;
                if (between)
                    weight[i] *= std::pow(std::erf(t * 0.70710678118654752), kTruePeakSamples - 1.0);
                levelDb += 20.0 * std::log10(t) + (envelope ? kEnvelopeLiftDb : 0.0);
            }
            const T gain = computeGain(static_cast<T>(levelDb), key.thresh, key.ratio, key.knee, key.charType,
                                       key.modeType, static_cast<T>(rmsDb + kSustainedOverRmsDb));
            target[i] = static_cast<double>(holdAndRange(gain, key.range, 0, held, counter, key.soft));
        }

        const int count = kMakeupNodes;
        double settled = 0.0;
        if (key.decoupled)
        {
            settled = restingGain(target, weight, count, 1.0, 1.0 - key.hold);
        }
        else
        {
            settled = restingGain(target, weight, count, 1.0 - key.attack, 1.0 - key.release);
            const double w = static_cast<double>(key.fastWeight);
            if (w < 1.0)
            {
                const double memory = key.loop
                    ? (1.0 - w) / (1.0 + w * (static_cast<double>(key.ratio) - 1.0)) : 1.0 - w;
                settled = (1.0 - memory) * settled
                        + memory * restingGain(target, weight, count, 1.0 - key.charge, 1.0 - key.slowRelease);
            }
        }
        const double makeup = -static_cast<double>(key.amount) * settled;
        return static_cast<T>(std::clamp(makeup, -kMakeupLimitDb, kMakeupLimitDb));
    }

    // ---- Sidechain HPF ----

    /** @brief Updates the normalized DC-blocker / High-pass filter coefficients. */
    void updateHpfCoefficients() noexcept
    {
        // The setter rejects invalid cutoffs; the clamp keeps the coefficient
        // in the stable range even against a hostile in-memory value.
        const double scFreq = std::clamp(
            static_cast<double>(scHpfFreq_.load(std::memory_order_relaxed)),
            1.0, sampleRate_ * 0.45);
        scHpfB1_ = static_cast<T>(std::exp(-std::numbers::pi * 2.0 * scFreq / sampleRate_));
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

            // Limit the logical window to the LIVE element count. capacity()
            // can exceed size() after a re-prepare to a lower rate, and a
            // window reaching into the dead tail reads stale squares from the
            // previous stream (measured: audible gain reduction on silence).
            if (!rmsBuffers_[0].empty())
            {
                int maxLen = static_cast<int>(rmsBuffers_[0].size());
                rmsWindowSamples_ = std::clamp(requestedSamples, 1, maxLen);
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
    std::atomic<CleanBallistics> cleanBallistics_ { CleanBallistics::Branching }; ///< Clean feed-forward ballistics.
    std::atomic<Mode> mode_ { Mode::Downward };                     ///< Selected compression mode.

    // Internal DSP Coefficients & State
    T autoMakeupCoeff_ = T(0.9995);     ///< Auto-makeup tracking factor.
    T autoMakeupEnv_ = T(0);            ///< Smoothed internal auto-makeup envelope.
    T cachedMakeupDb_ = T(0);           ///< Last makeup converted by makeupToGain().
    T cachedMakeupLin_ = T(1);          ///< Its linear gain.

    // Character ballistics coefficients (dB-domain one-poles, see
    // updateTimeConstants). charFastWeight_ == 1 selects the single-envelope
    // path (Clean/Varimu); Opto/FET blend fast and slow memory envelopes.
    std::atomic<bool> timeConstantsDirty_ { true }; ///< Coefficients need a recompute.
    double charAttCoeff_     = 0.0; ///< Attack coefficient of the fast envelope.
    double charRelCoeff_     = 0.0; ///< Release coefficient (fast stage).
    double charSlowRelCoeff_ = 0.0; ///< Release coefficient of the memory stage.
    double charChargeCoeff_  = 0.0; ///< History charge coefficient of the memory stage.
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

    // Recursive dB envelopes are double for every T: in float a 3 s release
    // at 192 kHz parked 0.57 dB short of its target.
    std::array<double, kMaxChannels> envFastDb_ {}; ///< Fast gain envelope per channel (dB).
    std::array<double, kMaxChannels> envSlowDb_ {}; ///< Slow memory envelope per channel (dB).
    std::array<double, kMaxChannels> envHoldDb_ {}; ///< Decoupled hold stage per channel (dB).
    double decHoldCoeff_   = 0.0;  ///< Decoupled hold-stage release coefficient.
    double decSmoothCoeff_ = 0.0;  ///< Decoupled smoothing pole, release side.
    bool decoupledActive_ = false; ///< The decoupled path ran last (audio thread).
    bool adaptiveMakeupActive_ = false; ///< Adaptive makeup ran last (audio thread).
    std::array<T, kMaxChannels> upwardGuardDb_ {};  ///< Peak-held sustained level (Upward guard).
    T upwardGuardDecay_ = T(0.00125);               ///< Guard decay per sample (60 dB/s).
    std::array<T, kMaxChannels> fbLastOutput_ {};   ///< Feedback topology history buffer.
    /// Detected level per channel and point of the frame (dB).
    std::array<std::array<T, 4>, kMaxChannels> pointLevelDb_ {};

    std::array<RingBuffer<T>, kMaxChannels> lookaheadBuffers_ {}; ///< Lookahead delay lines.
    int lookaheadSamples_ = 0; ///< Active lookahead latency in samples.

    std::array<HilbertIIR<double>, kMaxChannels> hilbertDetectors_ {}; ///< Zero-latency analytic pairs for detection.

    // Sidechain Filtering
    std::atomic<bool> scHpfEnabled_ { false }; ///< HPF toggle.
    std::atomic<T> scHpfFreq_ { T(80) };       ///< HPF Cutoff frequency.
    std::atomic<bool> scHpfDirty_ { true };    ///< Coefficients need a recompute.
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

    // Automatic threshold, lookahead inside a fixed latency, amount, soft range.
    static constexpr double kAutoRefFallDb = 1.0;   ///< dB per second the reference of the automatic threshold falls.
    std::atomic<bool> autoThreshold_ { false };     ///< The threshold counts from the key's own recent peak.
    std::array<double, kMaxChannels> autoRefDb_ {}; ///< That reference per channel (the linked one in slot 0), in dB.
    double autoRefFall_ = 0.0;                      ///< Its fall per sample, in dB.
    std::atomic<T> maxLookaheadMs_ { T(0) };        ///< Latency kept for the lookahead (0: the lookahead itself).
    int latencySamples_ = 0;                        ///< Delay of the audio in feed-forward (audio thread).
    std::array<RingBuffer<T>, kMaxChannels> gainDelay_ {}; ///< Holds the gain back while the lookahead is shorter than the latency.
    std::atomic<T> amount_ { T(1) };                ///< Share of the gain change that is applied, in dB.
    SmoothedValue<T> amountSmooth_;                 ///< De-zippered amount.
    std::atomic<bool> softRange_ { false };         ///< The gain change approaches the range without a corner.

    // Precision of the control signal (setPrecision): the detector's points
    // between samples, the history of the steps and their coefficients.
    static constexpr int kSubsteps = 8;                 ///< Steps of the ballistics per sample at Precision::High.
    static constexpr int kDetectorPoints = 4;           ///< Points the rectifying detectors read per sample there.
    static constexpr int kInterpTaps = 16;              ///< Samples each point between samples is read from.
    static constexpr int kDetectorDelay = kInterpTaps / 2 - 1; ///< Samples those points are late.
    static constexpr double kInterpBeta = 6.5;          ///< Kaiser window of the interpolation.
    static_assert(kDetectorPoints == 4 && kSubsteps % kDetectorPoints == 0, "the steps divide among the points");
    /// History of the key of one channel, mirrored so its window is contiguous.
    struct InterpHistory
    {
        T x[2 * kInterpTaps] = {}; ///< Each sample twice, kInterpTaps apart.
        int at = 0;                ///< Where the next one goes.
    };
    std::atomic<Precision> precision_ { Precision::Standard }; ///< How finely the control signal is computed.
    bool precisionActive_ = false;                      ///< The precise path ran last (audio thread).
    T interpPhase_[kDetectorPoints - 1][kInterpTaps] = {};     ///< Interpolation phases (prepare()).
    std::array<InterpHistory, kMaxChannels> interp_ {}; ///< Their history per channel.
    std::array<double, kMaxChannels> precLevel_ {};     ///< Last level of the solved loop (dB).
    std::array<double, kMaxChannels> precTarget_ {};    ///< Last gain target, after hold and range (dB).
    double rawAttCoeff_ = 0.0;       ///< Attack coefficient as it is observed (before a loop's own speed).
    double subAttCoeff_ = 0.0;       ///< Attack coefficient of an eighth of a sample.
    double subRelCoeff_ = 0.0;       ///< Release coefficient of an eighth of a sample.
    double subSlowRelCoeff_ = 0.0;   ///< Memory-stage release coefficient of an eighth of a sample.
    double subChargeCoeff_ = 0.0;    ///< Memory-stage charge coefficient of an eighth of a sample.
    double subDecHoldCoeff_ = 0.0;   ///< Decoupled hold coefficient of an eighth of a sample.
    double subDecSmoothCoeff_ = 0.0; ///< Decoupled smoothing coefficient of an eighth of a sample.

    // Analytic makeup: the programme it assumes and its cached result.
    static constexpr int kMakeupNodes = 48;             ///< Levels of the programme that are weighed.
    static constexpr double kMakeupReach = 4.8;         ///< They reach this many times its RMS.
    static constexpr double kNominalRmsDb = -18.0;      ///< RMS the programme is taken at (dBFS).
    static constexpr double kSustainedOverRmsDb = 10.0; ///< Its sustained level over the RMS (Upward guard, dB).
    static constexpr double kPeakReferenceDb = 12.35;   ///< Automatic-threshold reference over the RMS, rectified.
    static constexpr double kTruePeakReferenceDb = 12.5;///< The same for TruePeak.
    static constexpr double kEnvelopeReferenceDb = 13.2;///< The same for the Hilbert detector.
    static constexpr double kTruePeakSamples = 2.6;     ///< Independent samples the TruePeak reading is the largest of.
    static constexpr double kEnvelopeLiftDb = 0.6;      ///< What the Hilbert detector reads over the envelope (dB).
    static constexpr double kMakeupLimitDb = 40.0;      ///< Bound of the result (dB).
    SmoothedValue<T> analyticSmooth_;                   ///< De-zippered analytic makeup (dB).
    MakeupModelKey makeupKey_ {};                       ///< Settings of the cached result.
    T makeupModelDb_ = T(0);                            ///< The cached result (dB).

    std::atomic<T> gainReductionDb_ { T(0) }; ///< Publicly readable Gain Reduction meter.
};

} // namespace dspark
