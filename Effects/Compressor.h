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
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, Biquad.h, CharacterStage.h,
 *               SmoothedValue.h, RingBuffer.h, DenormalGuard.h, Hilbert.h,
 *               KeyEqualizer.h, TruePeakDetector.h, StateBlob.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Biquad.h"
#include "../Core/CharacterStage.h"
#include "../Core/SmoothedValue.h"
#include "../Core/RingBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Hilbert.h"
#include "../Core/KeyEqualizer.h"
#include "../Core/TruePeakDetector.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
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
        Hilbert,        ///< Analytic-signal magnitude from a zero-latency allpass pair
                        ///< (HilbertIIR): a ripple-free envelope on sustained tones from
                        ///< 20 Hz up, so the static curve holds exactly and low tones are
                        ///< not modulated at their own frequency (THD+N of a 30 Hz tone
                        ///< 14 dB into 4:1 at 5/100 ms: -96 dB, against -35 dB with Peak).
                        ///< The larger of that magnitude and the rectified input is
                        ///< used, so an onset is caught as fast as with Peak. Adds no
                        ///< latency.
        Hybrid          ///< RMS for what is steady and peak for what jumps: the larger of
                        ///< the root of twice the mean square (a one-pole mean over the time
                        ///< of setRmsWindow()) and the rectified sample. A steady tone
                        ///< reads its peak with next to no ripple, like Hilbert at a
                        ///< fraction of its cost, and whatever rises over the running
                        ///< level is caught in the same sample, like Peak.
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

        Varimu,   ///< Variable-mu tube lineage (Fairchild). The effective ratio grows
                  ///< with level above the threshold and the knee has a 10 dB floor
                  ///< (a remote-cutoff tube cannot produce a hard corner). Log-domain
                  ///< ballistics with the user's time constants. setCharacterColor()
                  ///< applies the gain through the curve of a push-pull pair biased
                  ///< by it, so the harmonics grow with the reduction.

        Layered    ///< Two layers in feed-forward (Topology is ignored): one that levels,
                  ///< with half the slope of the ratio from 6 dB under the threshold and
                  ///< times several times the knobs', under one that catches what rises
                  ///< over it, with the other half from 6 dB over the threshold and the
                  ///< knobs' times. Far over the threshold the curve is the ratio's own;
                  ///< around it, a knee 18 dB wide or more. Both layers take longer the
                  ///< farther they are from their target, so a deep reduction comes back
                  ///< at a steady pace instead of rushing. In Upward mode it is one
                  ///< envelope on the knobs, like Clean; Precision::High does not apply.
    };

    /** @brief Processing direction. */
    enum class Mode
    {
        Downward,  ///< Standard: Reduces dynamic range by attenuating signals above the threshold.
        Upward,    ///< Upward: Reduces dynamic range by boosting signals below the threshold.
                   ///< The boost fades out when the SUSTAINED level (peak-held, falling
                   ///< 60 dB/s) drops 40 to 60 dB below the threshold, so pauses and the
                   ///< noise floor are not amplified; setRange() bounds the boost.
        Both,      ///< Downward over the threshold and upward under a second, lower one, from
                   ///< the same detector: what is loud comes down and what is quiet comes
                   ///< up (see setLift()). The rise fades in silence as in Upward. A
                   ///< feed-forward curve: in FeedBack (and so with the FET character) the
                   ///< mode is Downward. A layered character runs one envelope, as in
                   ///< Upward, and the automatic makeups other than Analytic are off.
        Leveler    ///< No ratio and no knee: the gain is 1 / (1 + level / threshold). It
                   ///< takes 6 dB at the threshold and hardens by itself, until the output
                   ///< stays at the threshold however loud the input gets; setRange()
                   ///< bounds it. A feed-forward curve, like Both; the automatic makeups
                   ///< count it as Downward does its own curve.
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
        linkCoeff_ = 1.0 - std::exp(-1.0 / (0.02 * sampleRate_));
        linkLL_ = linkRR_ = linkLR_ = 0.0;

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
        keyMonitor_.prepare(sampleRate_, maxLaSamples + kStageKeyRoom);
        // The windows of HoldMode::EveryPeak (without memory that mode holds as Deepest does).
        holdWindows_.reset(new (std::nothrow) SlidingDeepest[kMaxChannels]);
        if (holdWindows_)
            for (int ch = 0; ch < kMaxChannels; ++ch) holdWindows_[static_cast<size_t>(ch)].configure(holdSamples_);
        holdWindowFor_ = holdSamples_;
        // The ramps of LookaheadShape::Ramp, for the channels of the layout.
        rampCount_ = std::clamp(spec.numChannels, 1, kMaxChannels);
        lookaheadRamps_.resize(static_cast<size_t>(rampCount_));
        for (auto& ramp : lookaheadRamps_) ramp.prepare(maxLaSamples);
        
        lookaheadSamples_ = static_cast<int>(fs * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        latencySamples_ = std::max(lookaheadSamples_, static_cast<int>(fs * std::clamp(
            maxLookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000)));
        latencyKept_ = maxLookaheadMs_.load(std::memory_order_relaxed) > T(0);

        updateHpfCoefficients();
        scHpfDirty_.store(false, std::memory_order_relaxed);
        keyEq_.prepare(sampleRate_);

        // RMS Buffer pre-allocation (Max 500ms to ensure zero real-time allocation)
        int maxRmsSamples = static_cast<int>(sampleRate_ * 0.5) + 1;
        for (auto& buf : rmsBuffers_)
        {
            buf.assign(static_cast<size_t>(maxRmsSamples), T(0));
        }
        updateRmsWindow();

        // The colour stage, both places; a change of place fades over 20 ms.
        stageBefore_.prepare(spec);
        stageAfter_.prepare(spec);
        stageScratch_.resize(std::min(spec.numChannels, kMaxChannels), spec.maxBlockSize);
        // The routes of a pair: the key as its middle and its sides, and the
        // line the part that is not treated waits in.
        routeBlock_ = std::max(1, spec.maxBlockSize);
        routeKey_.resize(2, routeBlock_);
        routeReach_ = maxLaSamples;
        routeDelay_.prepare(routeReach_ + 1);
        stageFade_.assign(static_cast<size_t>(std::max(1, spec.maxBlockSize)), 0.0);
        stageFadeStep_ = 1.0 / (0.020 * sampleRate_);
        {
            const StagePosition stage = stagePosition_.load(std::memory_order_relaxed);
            stageWeightBefore_ = stage == StagePosition::Before ? 1.0 : 0.0;
            stageWeightAfter_ = stage == StagePosition::After ? 1.0 : 0.0;
        }

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
        const StagePosition stage = stagePosition_.load(std::memory_order_relaxed);
        runStage(stageBefore_, stageWeightBefore_, stage == StagePosition::Before, buffer);
        processRouted(buffer, buffer);
        runStage(stageAfter_, stageWeightAfter_, stage == StagePosition::After, buffer);
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
        const StagePosition stage = stagePosition_.load(std::memory_order_relaxed);
        runStage(stageBefore_, stageWeightBefore_, stage == StagePosition::Before, audio);
        processRouted(audio, sidechain);
        runStage(stageAfter_, stageWeightAfter_, stage == StagePosition::After, audio);
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
        const CharacterProfile& profile = profileOf(charType);
        const Topology topoEff = topologyFor(profile, topo);
        const DetectorType detTypeEff = detectorFor(profile, detType);
        modeType = modeIn(modeType, topoEff);
        const bool fbImplicit = topoEff == Topology::FeedBack
                             && modeType == Mode::Downward
                             && detTypeEff == DetectorType::Peak;

        if (channel == 0
            && scHpfDirty_.load(std::memory_order_relaxed)
            && scHpfDirty_.exchange(false, std::memory_order_acquire))
            updateHpfCoefficients();
        // The equalizer of the key takes its settings and moves its ramps
        // once per frame, on channel 0.
        if (channel == 0)
        {
            (void)keyEq_.sync();
            keyEq_.advance(1);
        }

        // The sidechain filter must process whatever the detector consumes:
        // in FeedBack that is the compressed output (previous sample on the
        // explicit path; on the semi-implicit path the loop equation supplies
        // the gain, so the current input feeds the filter directly).
        T detectorIn = detectorSafe((topoEff == Topology::FeedBack && !fbImplicit)
                     ? fbLastOutput_[channel] : input);
        if (scHpf) detectorIn = applySidechainHPF(detectorIn, channel);
        detectorIn = keyEq_.processSample(detectorIn, channel);
        // The settings of the audio path that are not the plain gain, once per frame.
        if (channel == 0) (void)syncAudioPath();
        const T detectorRead = (asymAmount_ != T(0) && rectifies(detTypeEff)) ? asymmetric(detectorIn, channel)
                                                                              : detectorIn;

        const bool layered = isLayered(profile, modeType);
        syncLayerState(layered);
        const bool splitAdaptive = detTypeEff == DetectorType::SplitPolarity && !layered;
        const bool decoupled = useDecoupled(charType, topoEff);
        syncDecoupledState(decoupled);
        // Programme-dependent timing: the settings are taken on channel 0,
        // which also advances the balance of the key.
        if (advanceShared)
        {
            program_ = programTiming(charType, topoEff, decoupled, splitAdaptive);
            syncProgramState(program_.active);
            if (program_.active && program_.spectral > 0.0)
            {
                double lowSq = 0.0, highSq = 0.0;
                balanceBands(detectorSafe(input), channel, lowSq, highSq);
                (void)balanceAdvance(lowSq, highSq);
            }
        }

        T levelDb = detectLevel(detectorRead, channel, detTypeEff);

        const GainPathConfig cfg { charType, modeType, topoEff, fbImplicit, splitAdaptive, decoupled,
                                   feedbackDetectorDelay(detTypeEff), program_.active, layered };
        // The automatic threshold counts from the recent peak of what this
        // channel's detector reads (of its input, in FeedBack).
        T threshNow = thresh;
        if (autoThreshold_.load(std::memory_order_relaxed))
        {
            const T heard = (topoEff == Topology::FeedBack) ? gainToDecibels(std::abs(detectorSafe(input))) : levelDb;
            double& ref = autoRefDb_[static_cast<size_t>(channel)];
            ref = std::max(static_cast<double>(heard), ref - autoRefFall_);
            threshNow = thresh + static_cast<T>(ref);
            if (channel == 0) thresholdInUseDb_.store(threshNow, std::memory_order_relaxed);
        }
        T smoothedGR_Db = gainPath(levelDb, channel, cfg, threshNow, ratio, knee);
        // The gain the loop reads back is the whole one; the amount scales
        // what is heard (and metered).
        const T loopGainLinear = decibelsToGain(smoothedGR_Db);
        if (amount < T(1)) smoothedGR_Db *= amount;
        const bool gainSmooth = shapesGain();
        if (advanceShared) syncGainSmoothing(gainSmooth);
        if (gainSmooth) smoothedGR_Db = smoothedGain(smoothedGR_Db, channel);
        const bool heardApart = amount < T(1) || gainSmooth;
        T smoothedGainLinear = heardApart ? decibelsToGain(smoothedGR_Db) : loopGainLinear;

        const bool adaptiveMakeup = autoMkup == AutoMakeupMode::Adaptive && reducesOnly(modeType);
        if (advanceShared)
        {
            syncAdaptiveMakeup(adaptiveMakeup);
            if (adaptiveMakeup)
                autoMakeupEnv_ = smoothedGR_Db + autoMakeupCoeff_ * (autoMakeupEnv_ - smoothedGR_Db);
        }

        T makeup = makeupDb;
        if (reducesOnly(modeType))
        {
            if (autoMkup == AutoMakeupMode::Adaptive)
                makeup += -autoMakeupEnv_;
            else if (autoMkup == AutoMakeupMode::Static)
                makeup += computeGain(T(0), thresh, ratio, knee, charType,
                                      modeType, T(0)) * T(-0.5);
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

        const bool pushPull = colorAmt > T(0) && profile.colour == Colour::PushPull;
        const bool extras = audioExtras_ && !pushPull;
        T output = pushPull ? pushPullElement(input, smoothedGainLinear, colorAmt)
                 : (extras ? gained(input, smoothedGainLinear, channel) : input * smoothedGainLinear);
        if (colorAmt > T(0) && profile.colour == Colour::SecondOrder)
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
        fbLastOutput_[channel] = (heardApart || extras) ? input * loopGainLinear : output;
        output *= makeupToGain(makeup);
        gainReductionDb_.store(smoothedGR_Db, std::memory_order_relaxed);
        // Key listen: what the detector heard goes out instead.
        return scListen_.load(std::memory_order_relaxed) ? detectorIn : output;
    }

    /** 
     * @brief Resets all internal DSP history, states, and buffers to neutral.
     */
    void reset() noexcept
    {
        keyEq_.reset();
        routeDelay_.reset();
        toneLow_.reset();
        toneHigh_.reset();
        toneActive_ = false;       // designed again, for the rate in use, at the next block
        linkLL_ = linkRR_ = linkLR_ = 0.0;
        stageBefore_.reset();
        stageAfter_.reset();
        keyMonitor_.reset();
        gainDelayNow_ = -1.0;
        for (auto& ramp : lookaheadRamps_) ramp.clear();
        if (holdWindows_)
            for (int ch = 0; ch < kMaxChannels; ++ch) holdWindows_[static_cast<size_t>(ch)].clear();
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            envFastDb_[ch] = 0.0; // 0 dB gain change = neutral
            envSlowDb_[ch] = 0.0;
            envMidDb_[ch] = 0.0;
            envDeepDb_[ch] = 0.0;
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
            programState_[ch] = {};
            layerLevelDb_[ch] = 0.0;
            layerPeakDb_[ch] = 0.0;
            gainSmoothA_[ch] = 0.0;
            gainSmoothB_[ch] = 0.0;
            landingDb_[ch] = 0.0;
            networkLow_[ch] = 0.0;
            leakBase_[ch] = T(1);
            asymCrest_[ch] = T(0);
            balanceBands_[ch] = {};
            hybridMeanSq_[ch] = 0.0;
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
        balanceLow_ = balanceHigh_ = balance_ = 0.0;
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

    /** @brief Sets the curve of the gain computer (Downward, Upward, Both or Leveler). */
    void setMode(Mode mode) noexcept
    {
        mode_.store(clampEnum(mode, Mode::Leveler), std::memory_order_relaxed);
        // Feedback attack compensation only applies to Downward loops.
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Sets the upward half of Mode::Both: what lies under a second,
     *        lower threshold is raised.
     *
     * The lower threshold sits `belowDb` under the main one and moves with
     * it; under it the level is raised with `ratio` (2: half the distance)
     * up to `rangeDb`, with the knee of the main curve. Between the two
     * thresholds the gain is 0 dB. A range of 0 leaves Mode::Both the same
     * as Downward, bit for bit.
     *
     * State keys "liftBelow", "liftRatio" and "liftRange" (absent = 30, 2 and 6).
     *
     * @param belowDb Distance of the lower threshold under the main one, 0 to 80 dB.
     * @param ratio   Ratio of the rise, 1 (none) to 20.
     * @param rangeDb The most it raises, 0 to 40 dB.
     *                Non-finite values keep the one in use.
     */
    void setLift(T belowDb, T ratio, T rangeDb) noexcept
    {
        if (std::isfinite(belowDb)) liftBelow_.store(std::clamp(belowDb, T(0), T(80)), std::memory_order_relaxed);
        if (std::isfinite(ratio)) liftRatio_.store(std::clamp(ratio, T(1), T(20)), std::memory_order_relaxed);
        if (std::isfinite(rangeDb)) liftRange_.store(std::clamp(rangeDb, T(0), T(40)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Distance of the lower threshold of Mode::Both under the main one, in dB (setLift()). */
    [[nodiscard]] T getLiftBelow() const noexcept { return liftBelow_.load(std::memory_order_relaxed); }

    /** @brief Ratio of the rise of Mode::Both (setLift()). */
    [[nodiscard]] T getLiftRatio() const noexcept { return liftRatio_.load(std::memory_order_relaxed); }

    /** @brief The most Mode::Both raises, in dB (setLift()). */
    [[nodiscard]] T getLiftRange() const noexcept { return liftRange_.load(std::memory_order_relaxed); }

    /** @brief Sets stereo linking amount (0.0 = unlinked dual mono, 1.0 = fully linked). */
    void setStereoLink(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        stereoLink_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief What the linked level of a stereo pair reads (see setLinkMode()). */
    enum class LinkMode
    {
        Loudest,  ///< The louder channel's level (the default).
        Dominant  ///< The level of the pair's dominant component, wherever it sits in the image.
    };

    /**
     * @brief Chooses how a stereo pair is linked.
     *
     * Loudest links to the louder channel. Dominant links to the level of
     * the pair's dominant component: the larger eigenvalue of the
     * covariance of the two detector levels, the correlation taken from the
     * key over 20 ms. A source panned anywhere with an equal-power law
     * reads the same level (a centred source reads its level in each
     * channel, as Loudest does; panned hard it reads 3 dB less than
     * Loudest), and two uncorrelated channels read the louder one 3 dB
     * down, so wide ambience pushes less than a solid source of the same
     * level. It applies to two-channel blocks with some link; other
     * channel counts link to the loudest. Setup or any thread.
     *
     * @param mode Loudest (default) or Dominant.
     */
    void setLinkMode(LinkMode mode) noexcept
    {
        linkMode_.store(clampEnum(mode, LinkMode::Dominant), std::memory_order_relaxed);
    }

    /** @return How a stereo pair is linked. */
    [[nodiscard]] LinkMode getLinkMode() const noexcept { return linkMode_.load(std::memory_order_relaxed); }

    /** @brief What of a stereo pair the compressor works on (see setRoute()). */
    enum class Route
    {
        Stereo,      ///< Left and right (default).
        MidSide,     ///< The middle and the sides, each with its gain; setStereoLink() links the two.
        Mid,         ///< The middle only: the sides pass untouched.
        Side,        ///< The sides only: the middle passes untouched.
        MidKeysSide, ///< The sides, with the gain the middle of the key asks for.
        SideKeysMid  ///< The middle, with the gain the sides of the key ask for.
    };

    /**
     * @brief Chooses what of a stereo pair is compressed, and from which part of the key.
     *
     * The pair is taken as its middle, (L + R) / 2, and its sides,
     * (L - R) / 2, compressed as the route says and put back as left and
     * right. A key given apart is split the same way. In the routes that
     * treat one of the two, the other waits for it (it comes out late by
     * the latency of the gain path, so the pair stays in time) and is
     * otherwise untouched, and the makeup acts on the part that is treated.
     * In the two crossed routes the detector reads one part and the gain is
     * applied to the other, as with a key given apart: a feedback loop,
     * which reads its own output, treats that part by itself. While the key
     * or what is removed is listened to, the part that is not treated is
     * left out.
     *
     * It acts in processBlock() on buffers of exactly two channels; any
     * other width, and processSample(), run as Stereo. The audio thread
     * takes a change at the next block and starts clean there (it may
     * click: choose the route before playback).
     *
     * State key "route" (absent = Stereo).
     */
    void setRoute(Route route) noexcept { route_.store(clampEnum(route, Route::SideKeysMid), std::memory_order_relaxed); }

    /** @brief What of a stereo pair is compressed (setRoute()). */
    [[nodiscard]] Route getRoute() const noexcept { return route_.load(std::memory_order_relaxed); }

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
     * by the difference (which slides to its new value a quarter of a sample
     * per sample, with no step in the gain), so the lookahead can be
     * automated while the host keeps one delay compensation. 0 (the default)
     * is the behaviour of before: the latency is the lookahead. FeedBack runs
     * without lookahead and without latency either way.
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

    /** @brief How the lookahead brings a reduction in (see setLookaheadShape()). */
    enum class LookaheadShape
    {
        /// The gain runs ahead of the audio by the lookahead with its own
        /// attack: a fast attack is complete long before the peak arrives
        /// (the default).
        Lead,
        /// The gain reaches each reduction exactly when its peak leaves the
        /// delay, along an S-shaped ramp as long as the lookahead: the
        /// deepest gain change of the next lookahead, through two moving
        /// averages. Nothing is turned down earlier than it must be, and
        /// the corner of a fast attack is not heard ahead of a transient.
        /// A reduction also stays for the lookahead after its peak.
        Ramp
    };

    /**
     * @brief Chooses how the lookahead brings a reduction in (Lead by default).
     *
     * Feed-forward, with a lookahead above 0; the ramp follows a lookahead
     * that moves inside a latency kept for it (setMaxLookahead()). The ramp
     * runs for the channels given to prepare().
     */
    void setLookaheadShape(LookaheadShape shape) noexcept
    {
        lookaheadShape_.store(clampEnum(shape, LookaheadShape::Ramp), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @return How the lookahead brings a reduction in. */
    [[nodiscard]] LookaheadShape getLookaheadShape() const noexcept { return lookaheadShape_.load(std::memory_order_relaxed); }

    /**
     * @brief In FeedBack, runs the loop on the key ahead of the audio.
     *
     * A feedback compressor reads its own output, which leaves no room for
     * a lookahead or for a key of its own: by default FeedBack (and the
     * characters that always detect in feedback) runs without lookahead and
     * ignores an external key. With this on, the loop is closed around the
     * key instead (the external key when one is given, the programme
     * otherwise, before any delay): the detector reads that key times the
     * gain, as it would read the output, and the gain the loop settles on
     * is applied to the audio. With the programme as the key and no
     * lookahead that is the same compressor; and the lookahead (with its
     * latency, its shape and the latency kept for it) and an external key
     * now act in feedback too. Off by default.
     */
    void setFeedbackKeyed(bool on) noexcept
    {
        feedbackKeyed_.store(on, std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @return Whether a feedback loop runs on the key ahead of the audio. */
    [[nodiscard]] bool isFeedbackKeyed() const noexcept { return feedbackKeyed_.load(std::memory_order_relaxed); }

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
     * @brief The threshold in use, in dBFS (metering; any thread): the set
     *        one, or with setAutoThreshold() the key's recent peak plus it.
     */
    [[nodiscard]] T getThresholdInUseDb() const noexcept
    {
        return isAutoThreshold() ? thresholdInUseDb_.load(std::memory_order_relaxed) : threshold_.load(std::memory_order_relaxed);
    }

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
    void setSoftRange(bool on) noexcept
    {
        rangeShape_.store(static_cast<int>(on ? RangeShape::Asymptotic : RangeShape::Corner), std::memory_order_relaxed);
    }

    /** @brief Whether the gain change meets the range without a corner (setSoftRange(), setRangeShape()). */
    [[nodiscard]] bool isSoftRange() const noexcept
    {
        return rangeShape_.load(std::memory_order_relaxed) != static_cast<int>(RangeShape::Corner);
    }

    /** @brief How the gain change meets the range (see setRangeShape()). */
    enum class RangeShape
    {
        Corner,      ///< Cut at the range (default).
        Asymptotic,  ///< It bends from afar and never quite arrives: the curve of setSoftRange().
        Local        ///< A bend in a stretch around the range; exact under it and over it.
    };

    /**
     * @brief Selects how the gain change meets the range.
     *
     * Local bends the target over a stretch centred on the range, 6 dB wide
     * (or as wide as the range, when the range is less than that): under the
     * stretch the target is left as it is, over it it is the range exactly,
     * and the slope is continuous at both ends. A target on the range comes
     * out an eighth of the stretch short of it. Asymptotic, the curve of
     * setSoftRange(), bends from much farther away: 16 % short at the range
     * and never on it (a target of 12 dB under a range of 6 gives 5.9).
     * The semi-implicit FeedBack loop (Peak detector) keeps the corner.
     *
     * State key "rangeShape" (absent: Asymptotic or Corner, as "softRange" says).
     */
    void setRangeShape(RangeShape shape) noexcept
    {
        rangeShape_.store(static_cast<int>(clampEnum(shape, RangeShape::Local)), std::memory_order_relaxed);
    }

    /** @brief How the gain change meets the range (setRangeShape()). */
    [[nodiscard]] RangeShape getRangeShape() const noexcept
    {
        return static_cast<RangeShape>(rangeShape_.load(std::memory_order_relaxed));
    }

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


    /** @brief Where the analog colour stage sits (see setCharacterStage()). */
    enum class StagePosition
    {
        Off,     ///< No colour stage (the default).
        Before,  ///< The input amplifier: coloured before it is compressed (and detected).
        After    ///< The output amplifier: the compressed signal is coloured.
    };

    /** @brief The settings of the colour stage (Core/CharacterStage.h). */
    using StageSettings = typename CharacterStage<T>::Settings;

    /**
     * @brief Puts an analog colour stage before or after the compression.
     *
     * The stage (Core/CharacterStage.h) is the circuit around the gain
     * element: an emphasis shelf, two static saturators and the de-emphasis
     * that undoes the shelf, with slow random drift, a sag that lowers the
     * clipping point with the level and a noise floor that follows the
     * signal; it runs oversampled to about 176 kHz and has unity gain at low
     * levels. Before, the compressor (and its detector, unless an external
     * key is given) hears the coloured input; After, the compressed output
     * is coloured. A change of position crossfades over 20 ms, both ways and
     * between the two positions. The stage adds its delay (a few samples,
     * the polyphase IIR half-bands', 4 samples at 48 kHz) to getLatency()
     * while it is in. With it the compressor costs 172 to 534 ns per stereo
     * frame at 48 kHz (14 to 30 without). It runs in processBlock();
     * processSample() leaves it out. Safe from any thread.
     *
     * State keys "stage", "stageDrive", "stageEmphasis", "stageAsymmetry",
     * "stageDrift", "stageSag" and "stageNoise" (absent = off and the
     * defaults of StageSettings).
     */
    void setCharacterStage(StagePosition position) noexcept
    {
        stagePosition_.store(clampEnum(position, StagePosition::After), std::memory_order_relaxed);
    }

    /** @brief Where the colour stage sits (see setCharacterStage()). */
    [[nodiscard]] StagePosition getCharacterStage() const noexcept { return stagePosition_.load(std::memory_order_relaxed); }

    /** @brief Sets drive, emphasis, asymmetry, drift, sag and noise of the colour stage. Any thread. */
    void setCharacterStageSettings(const StageSettings& settings) noexcept
    {
        stageBefore_.setSettings(settings);
        stageAfter_.setSettings(settings);
    }

    /** @brief The settings of the colour stage. */
    [[nodiscard]] StageSettings getCharacterStageSettings() const noexcept { return stageBefore_.settings(); }

    /**
     * @brief Makes the release depend on how long the compressor has been working.
     *
     * The release time becomes the knob's after a brief reduction and up to
     * eight times the knob's after a sustained one. What tells them apart is
     * a slow follower of the depth of the gain change against the depth
     * itself, fifteen times as slow as the release (1.5 s at the default
     * release of 100 ms), so that "sustained" means the same at any release:
     * a peak that comes and goes leaves it near zero and releases at the
     * knob's time; a passage that has kept the compressor down for ten
     * release times or more releases slowly, so the level does not pump
     * back up between phrases. `amount` weighs it (0, the default, is the
     * fixed release).
     *
     * This and the three settings that follow act on the Clean and Varimu
     * characters in feed-forward with Branching ballistics, in every path
     * (processBlock(), processSample() and Precision::High). Opto and FET
     * have a memory of their own; Decoupled, feedback and the SplitPolarity
     * detector keep fixed times.
     *
     * State key "programRelease" (absent = 0).
     *
     * @param amount 0 to 1. Non-finite values are ignored.
     */
    void setProgramRelease(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        programRelease_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Weight of the programme-dependent release (see setProgramRelease()). */
    [[nodiscard]] T getProgramRelease() const noexcept { return programRelease_.load(std::memory_order_relaxed); }

    /**
     * @brief Makes the attack depend on how far the target is.
     *
     * A target 6 dB or more past the gain is met with the knob's attack; one
     * that is only just past it, with an attack up to four times longer. A
     * transient is still caught at the nominal speed, while the last
     * decibels of every attack, and the small swells of dense programme, are
     * taken gently (less of the waveform ends up in the gain). `amount`
     * weighs it (0, the default, is the fixed attack). Same scope as
     * setProgramRelease().
     *
     * State key "adaptiveAttack" (absent = 0).
     *
     * @param amount 0 to 1. Non-finite values are ignored.
     */
    void setAdaptiveAttack(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        adaptiveAttack_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Weight of the target-dependent attack (see setAdaptiveAttack()). */
    [[nodiscard]] T getAdaptiveAttack() const noexcept { return adaptiveAttack_.load(std::memory_order_relaxed); }

    /**
     * @brief Makes attack and release follow the spectral balance of the key.
     *
     * The key is measured in two bands of the same width on a logarithmic
     * axis, 30 to 200 Hz and 2.5 to 16 kHz (what lies under 30 Hz is left
     * out, so rumble and DC do not count), and their balance, between -1
     * (all low) and +1 (all high), moves both times: at full weight a bright
     * key is followed three times faster and a dark one three times slower,
     * and programme with the same power in every octave stays on the knobs.
     * Short times suit a snare and long ones a bass; this sets them from
     * what is playing. The key is the one the detector reads, before the
     * sidechain filter. `amount` weighs it (0, the default, is off). Same
     * scope as setProgramRelease().
     *
     * State key "spectralTiming" (absent = 0).
     *
     * @param amount 0 to 1. Non-finite values are ignored.
     */
    void setSpectralTiming(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        spectralTiming_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Weight of the spectral balance on the times (see setSpectralTiming()). */
    [[nodiscard]] T getSpectralTiming() const noexcept { return spectralTiming_.load(std::memory_order_relaxed); }

    /**
     * @brief Deepens or softens the gain change while it attacks.
     *
     * After the ballistics, the difference between the gain and a copy of it
     * smoothed over six times the attack (30 ms at the default attack of
     * 5 ms) is added back while the gain is falling: at +1 the
     * leading edge of every reduction goes as far again past its smoothed
     * course (more bite on the transient), at -1 the attack follows the
     * smoothed copy (the transient is let through and the body is held). It
     * only acts while the gain falls; a settled gain and a release are left
     * as they are. 0 (default) is off. Same scope as setProgramRelease().
     *
     * State key "attackEmphasis" (absent = 0).
     *
     * @param amount -1 to +1. Non-finite values are ignored.
     */
    void setAttackEmphasis(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        attackEmphasis_.store(std::clamp(amount, T(-1), T(1)), std::memory_order_relaxed);
    }

    /** @brief Emphasis of the attack (see setAttackEmphasis()). */
    [[nodiscard]] T getAttackEmphasis() const noexcept { return attackEmphasis_.load(std::memory_order_relaxed); }

    /** @brief Changes the level detection algorithm (Peak, RMS, TruePeak, SplitPolarity, Hilbert, Hybrid). */
    void setDetector(DetectorType type) noexcept
    {
        // The shared TruePeakDetector builds its coefficients lazily on first
        // use (thread-safe static), so this is a pure atomic publication.
        detectorType_.store(clampEnum(type, DetectorType::Hybrid), std::memory_order_relaxed);
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

    /** @brief What the hold keeps (see setHoldMode()). */
    enum class HoldMode
    {
        /// The hold starts again only when the gain change goes deeper (or
        /// as deep); when it ends, the target is whatever the curve asks at
        /// that moment, so a shallower peak inside the hold is not held
        /// after it (the default).
        Deepest,
        /// Every gain change is held for the hold time after it has passed:
        /// the deepest of a sliding window. On a tone that dies away with a
        /// hold longer than its period the gain follows the crests in steps
        /// instead of dropping into the wave between them. The window is
        /// exact to a 256th of the hold time.
        EveryPeak
    };

    /** @brief Chooses what the hold keeps (Deepest by default). */
    void setHoldMode(HoldMode mode) noexcept
    {
        holdMode_.store(clampEnum(mode, HoldMode::EveryPeak), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @return What the hold keeps. */
    [[nodiscard]] HoldMode getHoldMode() const noexcept { return holdMode_.load(std::memory_order_relaxed); }

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
        character_.store(clampEnum(type, Character::Layered), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief The optical cell of the Opto character (see setOptoCell()). */
    enum class OptoCell
    {
        TwoStage,   ///< A fast stage and one memory stage (default).
        ThreeStage  ///< A fast stage, a medium memory and a deep memory.
    };

    /**
     * @brief Chooses how the cell of the Opto character remembers.
     *
     * ThreeStage puts a medium memory between the fast stage and the deep
     * one: it charges over one release time (30 ms to 1 s) and lets go over
     * five, so a phrase of a second or two leaves a tail between the quick
     * recovery of a short peak and the long one of sustained compression.
     * The fast stage then releases over 0.725 release times so that, fully
     * charged, half the reduction still recovers in one release time, and
     * the deep stage keeps the complete release of 0.5 to 5 s. The three
     * shares of the gain are 0.6 (fast), 0.25 (medium) and 0.15 (deep).
     * Only Opto uses it. Thread-safe; applied at the next block.
     *
     * @param cell TwoStage (default) or ThreeStage.
     */
    void setOptoCell(OptoCell cell) noexcept
    {
        optoCell_.store(clampEnum(cell, OptoCell::ThreeStage), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @return The cell set by setOptoCell(). */
    [[nodiscard]] OptoCell getOptoCell() const noexcept { return optoCell_.load(std::memory_order_relaxed); }

    /**
     * @brief Sets the amount of the character's harmonic signature (0 to 1).
     *
     * Currently shapes the FET character: 2nd-order channel modulation of the
     * 1176's gain-reduction FET (the drain-source resistance bends with the
     * signal across it), scaled with the active gain reduction and AC-coupled
     * like the hardware's output stage. At 1.0 it is calibrated to the
     * published 1176 THD spec (< 0.5% while limiting, measured at -6 dBFS
     * program level); it only distorts while the FET conducts, so no gain
     * reduction means no added colour. The squared term doubles the signal
     * bandwidth: on synthetic full-scale HF material wrap the compressor in
     * an oversampled section (see docs/cookbook.md) if the folded 2nd
     * harmonic matters.
     *
     * With the Varimu character the gain is applied by the curve of a
     * push-pull pair of remote-cutoff stages: the bias that sets the gain
     * also bends the curve, so compression and colour come from the same
     * element. The harmonics are odd, grow with the level and with the
     * reduction (through a minimum near 6 dB of it), and at 1.0 a tone at
     * -6 dBFS carries 0.08 % of them at rest, 1.4 % under 12 dB of reduction
     * and 2.8 % under 20 dB. While it is on, the block runs frame by frame.
     *
     * Other characters ignore it. Default 0 (clean).
     */
    void setCharacterColor(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        characterColor_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Smooths the gain after the ballistics (milliseconds; 0, the
     *        default, is off).
     *
     * The gain change in decibels passes through two equal one-pole sections
     * before it becomes gain: a second-order smoothing that rounds the
     * corners the ballistics leave (the start of an attack, the turn into a
     * release) and takes the ripple of the detector off the gain. The step
     * of the two sections reaches 63 % in about the time given, so the gain
     * arrives that much later (a lookahead of the same length puts it back).
     * A feedback loop reads the gain before the smoothing. Values are held
     * to [0, 50] ms; a value that is not finite is ignored.
     */
    void setGainSmoothing(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        gainSmoothMs_.store(std::clamp(ms, T(0), T(50)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief The smoothing of the gain after the ballistics, in milliseconds. */
    [[nodiscard]] T getGainSmoothing() const noexcept { return gainSmoothMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Makes the last decibels of a release come back slowly
     *        (milliseconds; 0, the default, is off).
     *
     * After the ballistics, the gain that is heard follows them at once
     * while the gain change deepens, and on the way back with a time that
     * grows as it gets near: `ms` once it is within tenths of a decibel of
     * the gain it follows, e times shorter for each half decibel it is
     * behind, so that several decibels behind it is the ballistics
     * themselves. A release starts as it did and lands softly instead of
     * arriving with the slope of its exponential, which is where a fast
     * release is heard to pump. A feedback loop reads the gain before it.
     * Values are held to [0, 2000] ms; a value that is not finite is ignored.
     *
     * State key "releaseLanding" (absent = 0).
     */
    void setReleaseLanding(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        landingMs_.store(std::clamp(ms, T(0), T(2000)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief The landing of the release, in milliseconds (setReleaseLanding()). */
    [[nodiscard]] T getReleaseLanding() const noexcept { return landingMs_.load(std::memory_order_relaxed); }

    /**
     * @brief A lead-lag network in the control path: only a part of each
     *        fast move of the gain is applied at once.
     *
     * The gain change in decibels, after the ballistics, passes through a
     * first-order shelf that takes `depthDb` off what moves faster than
     * `cornerHz` and leaves what is slower as it is. Its gain at 0 Hz is 1
     * by construction, so the static curve does not move. Of a sudden move
     * of the gain, 10^(-depthDb / 20) arrives at once and the rest with the
     * time constant of the corner: an attack bites less, a release goes in
     * two stretches, and the ripple a fast release leaves on a low tone
     * falls by the depth, with none of the delay a low-pass adds
     * (setGainSmoothing()). A feedback loop reads the gain before the
     * network. A depth of 0 (the default) is off.
     *
     * State keys "networkDepth" and "networkCorner" (absent = 0 and 10).
     *
     * @param depthDb  0 to 12 dB. Non-finite values are ignored.
     * @param cornerHz 1 to 100 Hz. Non-finite values keep the corner in use.
     */
    void setControlNetwork(T depthDb, T cornerHz = T(10)) noexcept
    {
        if (!std::isfinite(depthDb)) return;
        networkDepthDb_.store(std::clamp(depthDb, T(0), T(12)), std::memory_order_relaxed);
        if (std::isfinite(cornerHz))
            networkCornerHz_.store(std::clamp(cornerHz, T(1), T(100)), std::memory_order_relaxed);
        timeConstantsDirty_.store(true, std::memory_order_release);
    }

    /** @brief Depth of the network in the control path, in dB (setControlNetwork()). */
    [[nodiscard]] T getControlNetworkDepth() const noexcept { return networkDepthDb_.load(std::memory_order_relaxed); }

    /** @brief Corner of the network in the control path, in Hz (setControlNetwork()). */
    [[nodiscard]] T getControlNetworkCorner() const noexcept { return networkCornerHz_.load(std::memory_order_relaxed); }

    /** @brief Corner of the low shelf of the tone that follows the reduction, in Hz (setReductionTone()). */
    static constexpr double kReductionToneLowHz = 200.0;
    /** @brief Corner of its high shelf, in Hz. */
    static constexpr double kReductionToneHighHz = 4000.0;

    /**
     * @brief Makes the tone change with the reduction.
     *
     * The gain then acts on a copy of the signal that has passed a nearly
     * flat filter F, and what that copy loses is taken off the signal:
     * y = x - (1 - g) F(x). With no reduction the output is the input, bit
     * for bit; as the gain falls the response leans in proportion to what is
     * taken off, as it does with a gain element that sits inside a network
     * that depends on frequency. F is a low shelf at 200 Hz and a high shelf
     * at 4 kHz, each with its own gain. Where F is over 0 dB the reduction is
     * deeper than the gain asked for, and where it is under, shallower: with
     * 12 dB of reduction, +1 dB makes it 3.9 dB deeper and -1 dB 2.4 dB
     * shallower. A negative `highDb` keeps the top while the compressor
     * works, and a negative `lowDb` the weight. Both at 0 (the default): off.
     *
     * It does not act with the push-pull element (setCharacterColor() on
     * Varimu), which applies the gain through its own curve, and a feedback
     * loop keeps reading the plain gain. While it is on, processBlock() runs
     * frame by frame instead of in the passes of the feed-forward path.
     *
     * State keys "toneLow" and "toneHigh" (absent = 0).
     *
     * @param lowDb  Gain of F under 200 Hz, -3 to +3 dB. Non-finite values are ignored.
     * @param highDb Gain of F over 4 kHz, -3 to +3 dB. Non-finite values are ignored.
     */
    void setReductionTone(T lowDb, T highDb) noexcept
    {
        if (std::isfinite(lowDb)) toneLowDb_.store(std::clamp(lowDb, T(-3), T(3)), std::memory_order_relaxed);
        if (std::isfinite(highDb)) toneHighDb_.store(std::clamp(highDb, T(-3), T(3)), std::memory_order_relaxed);
    }

    /** @brief Gain under 200 Hz of the tone that follows the reduction, in dB (setReductionTone()). */
    [[nodiscard]] T getReductionToneLow() const noexcept { return toneLowDb_.load(std::memory_order_relaxed); }

    /** @brief Its gain over 4 kHz, in dB. */
    [[nodiscard]] T getReductionToneHigh() const noexcept { return toneHighDb_.load(std::memory_order_relaxed); }

    /**
     * @brief Lets the moves of the gain leak into the audio as low thumps
     *        (0, the default, is off).
     *
     * What a gain element lets through of its own control: the gain, through
     * a high-pass of 5 Hz, is added to the output of the element. A settled
     * gain adds nothing; every attack and every release leaves a thump under
     * the programme, in proportion to how far the gain moved. At 1 a gain
     * that falls at once from 1 to a half leaves a thump that starts at
     * -32 dBFS. A feedback loop keeps reading the plain gain. While it is
     * on, processBlock() runs frame by frame.
     *
     * State key "controlLeak" (absent = 0).
     *
     * @param amount 0 to 1. Non-finite values are ignored.
     */
    void setControlLeak(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        controlLeak_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief How much of the moves of the gain leaks into the audio (setControlLeak()). */
    [[nodiscard]] T getControlLeak() const noexcept { return controlLeak_.load(std::memory_order_relaxed); }

    /**
     * @brief Makes the detector read the two halves of the wave differently
     *        (0, the default, is off).
     *
     * A share of the key's own crest (its peak, held for 50 ms) is added to
     * the key before it is rectified, and the sum is scaled back so that the
     * half the share favours still reads the crest: the other half reads
     * (1 - |a|) / (1 + |a|) of it. The gain then moves at the frequency of
     * the signal and not at twice it, which gives the modulation the colour
     * of a second harmonic, as an unbalanced rectifier does, and a signal
     * that is not symmetric is read differently from its inverse. The level
     * a held tone settles on is the same. It acts with the detectors that
     * rectify the key (Peak, TruePeak and Hybrid); positive values favour
     * the positive half. While it is on, processBlock() runs frame by frame.
     *
     * State key "detectorAsymmetry" (absent = 0).
     *
     * @param amount -1 to +1. Non-finite values are ignored.
     */
    void setDetectorAsymmetry(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        detectorAsymmetry_.store(std::clamp(amount, T(-1), T(1)), std::memory_order_relaxed);
    }

    /** @brief How differently the detector reads the two halves of the wave (setDetectorAsymmetry()). */
    [[nodiscard]] T getDetectorAsymmetry() const noexcept { return detectorAsymmetry_.load(std::memory_order_relaxed); }

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

    /** @brief The equalizer of the key (see setSidechainBand()). */
    using KeyEq = KeyEqualizer<T, 16>;
    /** @brief The settings of one band of the key equalizer. */
    using KeyBand = typename KeyEq::Band;
    /** @brief What a band of the key equalizer does. */
    using KeyShape = typename KeyEq::Shape;
    /** @brief Bands of the key equalizer. */
    static constexpr int kKeyBands = KeyEq::kBands;

    /**
     * @brief Sets a band of the equalizer of the key.
     *
     * Six bands shape what the detector weighs, after the sidechain
     * high-pass: low and high cuts from 6 to 48 dB per octave, bells,
     * shelves and a tilt, each designed from its analog curve. They act on
     * whatever the detector reads (the programme, the external key or, in
     * FeedBack, the compressed output) and never on the audio. A change
     * glides over 20 ms; a band that is Off costs nothing. The timing that
     * follows the spectral balance (setSpectralTiming()) and the analytic
     * makeup read the key before the equalizer.
     *
     * @param band     Band index in [0, kKeyBands); others are ignored.
     * @param settings Shape, frequency, gain, q and slope of the band.
     */
    void setSidechainBand(int band, const KeyBand& settings) noexcept { keyEq_.setBand(band, settings); }

    /** @brief The settings of a band of the key equalizer. */
    [[nodiscard]] KeyBand getSidechainBand(int band) const noexcept { return keyEq_.getBand(band); }

    /**
     * @brief The curve of the key equalizer, for a display: build it when the
     *        bands change and read each frequency from it.
     */
    [[nodiscard]] typename KeyEq::Response getSidechainResponse() const noexcept { return keyEq_.getResponse(); }

    /**
     * @brief Plays the key instead of the compressed audio.
     *
     * The output of each channel is what its detector hears, after the
     * sidechain high-pass and the equalizer: the programme, the external key
     * or, in FeedBack with a detector that keeps a memory, the compressed
     * output of the previous sample. Through processBlock() it leaves late by
     * the latency the compressor reports, so the track keeps its place when
     * the listen is switched, and it is held under full scale (an equalized
     * key can be far louder than the programme: KeyMonitor). The compressor
     * keeps working underneath (the meter still reads). It is a monitoring
     * switch: it is not part of the state.
     */
    void setSidechainListen(bool on) noexcept { scListen_.store(on, std::memory_order_relaxed); }

    /** @brief Whether the key is being played instead of the audio. */
    [[nodiscard]] bool isSidechainListen() const noexcept { return scListen_.load(std::memory_order_relaxed); }

    /**
     * @brief Removal listen: the output is what the compression takes out, the
     *        aligned input minus the compressed signal before the makeup gain
     *        (the mix does not apply). A monitoring control, not in the state.
     */
    void setRemovalListen(bool on) noexcept { removalListen_.store(on, std::memory_order_relaxed); }
    /** @return Whether the output is what the compression takes out. */
    [[nodiscard]] bool isRemovalListen() const noexcept { return removalListen_.load(std::memory_order_relaxed); }

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
     * is 0 there, unless the loop runs on the key (setFeedbackKeyed()). With setPrecision(High), the Peak and TruePeak detectors
     * read 7 samples late and the audio waits for them, in feed-forward and
     * in the feedback loops that mode covers. A colour stage in place
     * (setCharacterStage()) adds its few samples. The plugin layer re-reads
     * this after parameter changes and re-notifies the host when it moves.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        const int coloured = (stagePosition_.load(std::memory_order_relaxed) != StagePosition::Off)
                           ? stageBefore_.getLatency() : 0;
        return gainPathLatency() + coloured;
    }

private:
    /** @brief The latency of the gain path alone: the lookahead and the detector's reading (no colour stage). */
    [[nodiscard]] int gainPathLatency() const noexcept
    {
        const CharacterProfile& profile = profileOf(character_.load(std::memory_order_relaxed));
        const bool feedback = topologyFor(profile, topology_.load(std::memory_order_relaxed)) == Topology::FeedBack;
        const DetectorType detType = detectorFor(profile, detectorType_.load(std::memory_order_relaxed));
        // The same routing as processBlock(): see usePrecision().
        const Mode mode = modeIn(mode_.load(std::memory_order_relaxed),
                                 feedback ? Topology::FeedBack : Topology::FeedForward);
        const bool solved = feedback && detType == DetectorType::Peak && mode == Mode::Downward;
        const bool layered = isLayered(profile, mode);
        const int reading = (precision_.load(std::memory_order_relaxed) == Precision::High && !layered
                             && readsBetweenSamples(detType) && (!feedback || solved)) ? kDetectorDelay : 0;
        if (feedback && !feedbackKeyed_.load(std::memory_order_relaxed)) return reading;
        // Derive the lookahead from the published parameter rather than the
        // audio-thread cache: hosts re-read the latency right after a setter,
        // before the next block has consumed the coefficient-update flag.
        const int lookNow = static_cast<int>(static_cast<T>(sampleRate_) * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        const int kept = static_cast<int>(static_cast<T>(sampleRate_) * std::clamp(
            maxLookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        return std::max(lookNow, kept) + reading;
    }

    /**
     * @brief The compressor on a pair as the route says: its middle and its
     *        sides in place of left and right, one of the two, or one with
     *        the key of the other. Anything but two channels runs as Stereo.
     */
    void processRouted(AudioBufferView<T> audio, AudioBufferView<T> sidechain) noexcept
    {
        const Route route = route_.load(std::memory_order_relaxed);
        const bool routed = route != Route::Stereo && audio.getNumChannels() == 2 && routeBlock_ > 0;
        if (routed != routeWas_ || (routed && route != routeIn_))
        {
            // Every state would carry another signal: the route starts clean.
            if (sampleRate_ > 0) reset();
            routeWas_ = routed;
            routeIn_ = route;
        }
        if (!routed)
        {
            processBlockImpl(audio, sidechain);
            return;
        }
        if (!(sampleRate_ > 0)) return;

        const int total = audio.getNumSamples();
        const bool external = sidechain.getNumChannels() > 0 && sidechain.getNumSamples() >= total
                           && sidechain.getChannel(0) != audio.getChannel(0);
        for (int off = 0; off < total; off += routeBlock_)
        {
            const int m = std::min(routeBlock_, total - off);
            T* const left = audio.getChannel(0) + off;
            T* const right = audio.getChannel(1) + off;
            // The pair as its middle and its sides, in place.
            for (int i = 0; i < m; ++i)
            {
                const T l = left[i], r = right[i];
                left[i] = (l + r) * T(0.5);
                right[i] = (l - r) * T(0.5);
            }
            T* keyMid = left;
            T* keySide = right;
            if (external)
            {
                const T* const kl = sidechain.getChannel(0) + off;
                const T* const kr = (sidechain.getNumChannels() > 1 ? sidechain.getChannel(1) : sidechain.getChannel(0)) + off;
                keyMid = routeKey_.getChannel(0);
                keySide = routeKey_.getChannel(1);
                for (int i = 0; i < m; ++i)
                {
                    keyMid[i] = (kl[i] + kr[i]) * T(0.5);
                    keySide[i] = (kl[i] - kr[i]) * T(0.5);
                }
            }

            if (route == Route::MidSide)
            {
                T* pair[2] = { left, right };
                T* keys[2] = { keyMid, keySide };
                processBlockImpl(AudioBufferView<T>(pair, 2, m), AudioBufferView<T>(keys, 2, m));
            }
            else
            {
                const bool sides = route == Route::Side || route == Route::MidKeysSide;
                const bool keyOfSides = route == Route::Side || route == Route::SideKeysMid;
                T* treated[1] = { sides ? right : left };
                T* key[1] = { keyOfSides ? keySide : keyMid };
                T* const waiting = sides ? left : right;
                processBlockImpl(AudioBufferView<T>(treated, 1, m), AudioBufferView<T>(key, 1, m));
                // The other part waits for it; listened to, it is left out.
                const int late = std::min(gainPathLatency(), routeReach_);
                const bool listening = scListen_.load(std::memory_order_relaxed)
                                    || removalListen_.load(std::memory_order_relaxed);
                for (int i = 0; i < m; ++i)
                {
                    routeDelay_.push(waiting[i]);
                    waiting[i] = listening ? T(0) : routeDelay_.read(late);
                }
            }

            // Back to left and right.
            for (int i = 0; i < m; ++i)
            {
                const T mid = left[i], side = right[i];
                left[i] = mid + side;
                right[i] = mid - side;
            }
        }
    }

public:


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
        w.write("linkMode", static_cast<int32_t>(linkMode_.load(std::memory_order_relaxed)));
        w.write("route", static_cast<int32_t>(getRoute()));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("lookahead", static_cast<float>(lookaheadMs_.load(std::memory_order_relaxed)));
        w.write("hold", static_cast<float>(holdMs_.load(std::memory_order_relaxed)));
        w.write("holdMode", static_cast<int32_t>(getHoldMode()));
        w.write("range", static_cast<float>(rangeDb_.load(std::memory_order_relaxed)));
        w.write("detector", static_cast<int32_t>(detectorType_.load(std::memory_order_relaxed)));
        w.write("topology", static_cast<int32_t>(topology_.load(std::memory_order_relaxed)));
        w.write("character", static_cast<int32_t>(character_.load(std::memory_order_relaxed)));
        w.write("optoCell", static_cast<int32_t>(optoCell_.load(std::memory_order_relaxed)));
        w.write("characterColor", static_cast<float>(characterColor_.load(std::memory_order_relaxed)));
        w.write("cleanBallistics", static_cast<int32_t>(cleanBallistics_.load(std::memory_order_relaxed)));
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("scHpf", scHpfEnabled_.load(std::memory_order_relaxed));
        w.write("scHpfFreq", static_cast<float>(scHpfFreq_.load(std::memory_order_relaxed)));
        w.write("rmsWindow", static_cast<float>(rmsWindowMsAtomic_.load(std::memory_order_relaxed)));
        w.write("autoThreshold", autoThreshold_.load(std::memory_order_relaxed));
        w.write("maxLookahead", static_cast<float>(maxLookaheadMs_.load(std::memory_order_relaxed)));
        w.write("lookaheadShape", static_cast<int32_t>(getLookaheadShape()));
        w.write("feedbackKeyed", isFeedbackKeyed());
        w.write("amount", static_cast<float>(amount_.load(std::memory_order_relaxed)));
        w.write("softRange", isSoftRange());   // what an older build understands
        w.write("rangeShape", static_cast<int32_t>(getRangeShape()));
        w.write("precision", static_cast<int32_t>(precision_.load(std::memory_order_relaxed)));
        {
            const StageSettings st = getCharacterStageSettings();
            w.write("stage", static_cast<int32_t>(getCharacterStage()));
            w.write("stageDrive", static_cast<float>(st.driveDb));
            w.write("stageEmphasis", static_cast<float>(st.emphasisDb));
            w.write("stageAsymmetry", static_cast<float>(st.asymmetry));
            w.write("stageDrift", static_cast<float>(st.drift));
            w.write("stageSag", static_cast<float>(st.sag));
            w.write("stageNoise", static_cast<float>(st.noiseDb));
        }
        w.write("programRelease", static_cast<float>(programRelease_.load(std::memory_order_relaxed)));
        w.write("adaptiveAttack", static_cast<float>(adaptiveAttack_.load(std::memory_order_relaxed)));
        w.write("spectralTiming", static_cast<float>(spectralTiming_.load(std::memory_order_relaxed)));
        w.write("attackEmphasis", static_cast<float>(attackEmphasis_.load(std::memory_order_relaxed)));
        keyEq_.writeState(w, "key");
        w.write("gainSmoothing", static_cast<float>(gainSmoothMs_.load(std::memory_order_relaxed)));
        w.write("releaseLanding", static_cast<float>(landingMs_.load(std::memory_order_relaxed)));
        w.write("networkDepth", static_cast<float>(networkDepthDb_.load(std::memory_order_relaxed)));
        w.write("networkCorner", static_cast<float>(networkCornerHz_.load(std::memory_order_relaxed)));
        w.write("toneLow", static_cast<float>(toneLowDb_.load(std::memory_order_relaxed)));
        w.write("toneHigh", static_cast<float>(toneHighDb_.load(std::memory_order_relaxed)));
        w.write("controlLeak", static_cast<float>(controlLeak_.load(std::memory_order_relaxed)));
        w.write("detectorAsymmetry", static_cast<float>(detectorAsymmetry_.load(std::memory_order_relaxed)));
        w.write("liftBelow", static_cast<float>(liftBelow_.load(std::memory_order_relaxed)));
        w.write("liftRatio", static_cast<float>(liftRatio_.load(std::memory_order_relaxed)));
        w.write("liftRange", static_cast<float>(liftRange_.load(std::memory_order_relaxed)));
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
        setLinkMode(static_cast<LinkMode>(r.read("linkMode", 0)));
        setRoute(static_cast<Route>(r.read("route", 0)));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setLookahead(static_cast<T>(r.read("lookahead", 0.0f)));
        setHoldTime(static_cast<T>(r.read("hold", 0.0f)));
        setHoldMode(static_cast<HoldMode>(r.read("holdMode", 0)));
        setRange(static_cast<T>(r.read("range", 100.0f)));
        setDetector(static_cast<DetectorType>(r.read("detector", 0)));
        setTopology(static_cast<Topology>(r.read("topology", 0)));
        setCharacter(static_cast<Character>(r.read("character", 0)));
        setOptoCell(static_cast<OptoCell>(r.read("optoCell", 0)));
        setCharacterColor(static_cast<T>(r.read("characterColor", 0.0f)));
        setCleanBallistics(static_cast<CleanBallistics>(r.read("cleanBallistics", 0)));
        setMode(static_cast<Mode>(r.read("mode", 0)));
        setSidechainHPF(r.read("scHpf", false), static_cast<T>(r.read("scHpfFreq", 80.0f)));
        setRmsWindow(static_cast<T>(r.read("rmsWindow", 10.0f)));
        setAutoThreshold(r.read("autoThreshold", false));
        setMaxLookahead(static_cast<T>(r.read("maxLookahead", 0.0f)));
        setLookaheadShape(static_cast<LookaheadShape>(r.read("lookaheadShape", 0)));
        setFeedbackKeyed(r.read("feedbackKeyed", false));
        setAmount(static_cast<T>(r.read("amount", 1.0f)));
        setRangeShape(static_cast<RangeShape>(r.read("rangeShape", r.read("softRange", false) ? 1 : 0)));
        setPrecision(static_cast<Precision>(r.read("precision", 0)));
        {
            const StageSettings defaults {};
            StageSettings st;
            st.driveDb = static_cast<T>(r.read("stageDrive", static_cast<float>(defaults.driveDb)));
            st.emphasisDb = static_cast<T>(r.read("stageEmphasis", static_cast<float>(defaults.emphasisDb)));
            st.asymmetry = static_cast<T>(r.read("stageAsymmetry", static_cast<float>(defaults.asymmetry)));
            st.drift = static_cast<T>(r.read("stageDrift", static_cast<float>(defaults.drift)));
            st.sag = static_cast<T>(r.read("stageSag", static_cast<float>(defaults.sag)));
            st.noiseDb = static_cast<T>(r.read("stageNoise", static_cast<float>(defaults.noiseDb)));
            setCharacterStageSettings(st);
            setCharacterStage(static_cast<StagePosition>(r.read("stage", 0)));
        }
        setProgramRelease(static_cast<T>(r.read("programRelease", 0.0f)));
        setAdaptiveAttack(static_cast<T>(r.read("adaptiveAttack", 0.0f)));
        setSpectralTiming(static_cast<T>(r.read("spectralTiming", 0.0f)));
        setAttackEmphasis(static_cast<T>(r.read("attackEmphasis", 0.0f)));
        keyEq_.readState(r, "key");
        setGainSmoothing(static_cast<T>(r.read("gainSmoothing", 0.0f)));
        setReleaseLanding(static_cast<T>(r.read("releaseLanding", 0.0f)));
        setControlNetwork(static_cast<T>(r.read("networkDepth", 0.0f)), static_cast<T>(r.read("networkCorner", 10.0f)));
        setReductionTone(static_cast<T>(r.read("toneLow", 0.0f)), static_cast<T>(r.read("toneHigh", 0.0f)));
        setControlLeak(static_cast<T>(r.read("controlLeak", 0.0f)));
        setDetectorAsymmetry(static_cast<T>(r.read("detectorAsymmetry", 0.0f)));
        setLift(static_cast<T>(r.read("liftBelow", 30.0f)), static_cast<T>(r.read("liftRatio", 2.0f)),
                static_cast<T>(r.read("liftRange", 6.0f)));
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
        const bool keyEqOn = keyEq_.sync();
        const bool listen = scListen_.load(std::memory_order_relaxed);
        // What is not the plain gain (a tone that follows the reduction, the
        // leak of the control, a detector that is not symmetric) runs frame by frame.
        const bool sampleWise = syncAudioPath();

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
        const LinkMode linkMode = linkMode_.load(std::memory_order_relaxed);

        // The FET character is a feedback design like the 1176 it models: it
        // always detects on the compressed output with the hardware's peak
        // rectifier, whatever Topology and Detector say.
        const CharacterProfile& profile = profileOf(charType);
        const Topology topoEff = topologyFor(profile, topo);
        const DetectorType detTypeEff = detectorFor(profile, detType);
        modeType = modeIn(modeType, topoEff);

        // Downward feedback with the peak detector resolves the loop
        // semi-implicitly (the level the loop will read is a known function
        // of the gain): stable at any attack, and the observed static curve
        // lands exactly on the requested ratio/knee. Detectors with memory
        // (RMS/TruePeak/Hilbert/Split) keep the explicit one-sample loop.
        const bool fbImplicit = topoEff == Topology::FeedBack
                             && modeType == Mode::Downward
                             && detTypeEff == DetectorType::Peak;

        const bool layered = isLayered(profile, modeType);
        syncLayerState(layered);
        const bool splitAdaptive = detTypeEff == DetectorType::SplitPolarity && !layered;
        const bool decoupled = useDecoupled(charType, topoEff);
        syncDecoupledState(decoupled);
        const bool precise = !layered && usePrecision(topoEff, fbImplicit, splitAdaptive);
        const bool gainSmooth = shapesGain();
        syncGainSmoothing(gainSmooth);
        syncPrecisionState(precise);
        program_ = programTiming(charType, topoEff, decoupled, splitAdaptive);
        syncProgramState(program_.active);
        const bool keyBalance = program_.active && program_.spectral > 0.0;

        // A loop that reads its own output has no room for a lookahead: in
        // FeedBack it is off, unless the loop is closed around the key
        // (setFeedbackKeyed()).
        const bool keyedLoop = topoEff == Topology::FeedBack && feedbackKeyed_.load(std::memory_order_relaxed);
        int activeLookahead = (topoEff == Topology::FeedBack && !keyedLoop) ? 0 : latencySamples_;
        const bool autoThr = autoThreshold_.load(std::memory_order_relaxed);
        // An external key does not pass the colour stage placed before the
        // compressor: listened to, it waits for that stage too.
        // The lookahead as a ramp: its state starts clean when it comes in.
        const bool ramp = rampShape_ && activeLookahead > 0;
        if (ramp != rampActive_)
        {
            if (ramp)
                for (auto& r : lookaheadRamps_) r.clear();
            rampActive_ = ramp;
        }
        const bool externalKey = scCh > 0 && nCh > 0 && sidechain.getChannel(0) != audio.getChannel(0);
        const int stageKeyDelay = (externalKey && stagePosition_.load(std::memory_order_relaxed) == StagePosition::Before)
                                ? std::min(stageBefore_.getLatency(), kStageKeyRoom) : 0;

        // Fully linked channels share one gain computer: every channel reads
        // the same linked level into the same state, so computing it once per
        // frame gives the same gain (the per-channel path differs only in the
        // rounding of level + 1 * (linked - level)). SplitPolarity's adaptive
        // release reads each channel's own output, so it keeps the
        // per-channel path.
        const bool shared = (sLink >= T(1)) && !splitAdaptive && nCh > 1;
        // The dominant link reads the pair as one (two channels, some link).
        const bool dominantLink = linkMode == LinkMode::Dominant && nCh == 2 && sLink > T(0);
        if (dominantLink) flushLinkCovariance();

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
        const bool staticMakeup = autoMkup == AutoMakeupMode::Static && reducesOnly(modeType);
        const bool adaptiveMakeup = autoMkup == AutoMakeupMode::Adaptive && reducesOnly(modeType);
        syncAdaptiveMakeup(adaptiveMakeup);
        T staticMakeupDb = staticMakeup
            ? computeGain(T(0), thresh, ratio, knee, charType, modeType, T(0)) * T(-0.5)
            : T(0);
        T analyticDb = analyticSmooth_.getCurrentValue();

        const GainPathConfig cfg { charType, modeType, topoEff, fbImplicit, splitAdaptive, decoupled,
                                   feedbackDetectorDelay(detTypeEff), program_.active, layered };

        // Feed-forward: the detector never reads the output, so the block
        // runs in passes (see processFeedForward). Feedback loops stay per
        // sample, and so do Precision::High, whose control signal is worked
        // out several times per sample, and the push-pull element, which
        // applies the gain through a curve.
        const bool feedForward = topoEff == Topology::FeedForward;
        const bool element = profile.colour == Colour::PushPull
                          && (colorAmt > T(0) || colorSmooth_.isSmoothing());
        if (feedForward && !precise && !element && !sampleWise)
        {
            const FeedForwardConfig ff { detTypeEff, charType, modeType, autoMkup, sLink,
                                         activeLookahead, scHpf, shared, decoupled,
                                         splitAdaptive, staticMakeup, adaptiveMakeup,
                                         paramsMoving, autoThr, activeLookahead - lookaheadSamples_,
                                         program_.active, keyBalance, keyEqOn, listen, layered, gainSmooth,
                                         dominantLink, activeLookahead + stageKeyDelay, latencyKept_, ramp };
            const ChunkParams constants { thresh, ratio, knee, mkupBase, mixVal,
                                          staticMakeupDb + analyticDb, amount };
            processFeedForward(audio, sidechain, nCh, nS, scCh, ff, constants);
            return;
        }

        // At Precision::High the detectors that rectify read the key at
        // kDetectorPoints points per sample, kDetectorDelay samples late, and
        // the audio waits for them. The others keep one level per sample.
        const bool points = precise && readsBetweenSamples(detTypeEff);
        const bool lopsided = asymAmount_ != T(0) && rectifies(detTypeEff);
        const int count = points ? kDetectorPoints : 1;
        const int audioDelay = activeLookahead + (points ? kDetectorDelay : 0);
        const int gainDelay = (feedForward || keyedLoop) ? activeLookahead - lookaheadSamples_ : 0;
        const bool gainLine = (feedForward || keyedLoop) && latencyKept_;
        const int listenDelay = audioDelay + stageKeyDelay;

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
                                                 modeType, T(0)) * T(-0.5);
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
            T pairLevel[2][kDetectorPoints] = {};
            T heard[kMaxChannels] = {};
            T loopSource[kMaxChannels] = {};   // a keyed loop: the key before the gain
            double lowSq = 0.0, highSq = 0.0;
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
                else if (keyedLoop)
                {
                    // The loop around the key: the key itself where the loop
                    // equation supplies the gain, the key times the last
                    // gain where the detector keeps a memory.
                    loopSource[ch] = detectorSafe((scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))[i]
                                                             : audio.getChannel(ch)[i]);
                    detectorIn = fbImplicit ? loopSource[ch] : fbLastOutput_[ch];
                }
                else
                    detectorIn = fbImplicit ? audio.getChannel(ch)[i] : fbLastOutput_[ch];
                detectorIn = detectorSafe(detectorIn);
                if (keyBalance) balanceBands(detectorIn, ch, lowSq, highSq);
                if (scHpf) detectorIn = applySidechainHPF(detectorIn, ch);
                if (keyEqOn) detectorIn = keyEq_.processSample(detectorIn, ch);
                heard[ch] = detectorIn;
                if (lopsided) detectorIn = asymmetric(detectorIn, ch);

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
                if (dominantLink && ch < 2)
                    for (int j = 0; j < count; ++j) pairLevel[ch][j] = level[j];
            }
            if (dominantLink)
            {
                const double rho2 = linkCorrelation(heard[0], heard[1]);
                for (int j = 0; j < count; ++j)
                    linkedLin[j] = dominantLevel(pairLevel[0][j], pairLevel[1][j], rho2);
            }

            if (keyBalance) (void)balanceAdvance(lowSq, highSq);

            // The automatic threshold: its reference follows the largest
            // level the gain computers read in feed-forward, and the linked
            // input in feedback, where the detector reads the output.
            T threshNow = thresh;
            if (autoThr)
            {
                T top = T(0);
                if (feedForward)
                    for (int j = 0; j < count; ++j) top = std::max(top, linkedLin[j]);
                else if (keyedLoop)
                    for (int ch = 0; ch < nCh; ++ch) top = std::max(top, std::abs(loopSource[ch]));
                else
                    for (int ch = 0; ch < nCh; ++ch) top = std::max(top, std::abs(detectorSafe(audio.getChannel(ch)[i])));
                autoRefDb_[0] = std::max(static_cast<double>(gainToDecibels(top)), autoRefDb_[0] - autoRefFall_);
                threshNow = thresh + static_cast<T>(autoRefDb_[0]);
                thresholdInUseDb_.store(threshNow, std::memory_order_relaxed);
            }

            // 2. Stereo linking, static curve and character ballistics (log
            //    domain). What is heard of the gain is held back inside a
            //    latency kept for the lookahead and scaled by the amount; the
            //    loop keeps the whole.
            const double slide = gainLine ? slideGainDelay(gainDelay) : 0.0;
            const int rampLength = gainLine ? activeLookahead - static_cast<int>(std::ceil(slide)) : activeLookahead;
            T blockGR = T(0);
            if (shared)
            {
                T levelDb[kDetectorPoints] = {};
                for (int j = 0; j < count; ++j) levelDb[j] = gainToDecibels(linkedLin[j]);
                const T loopGR = precise ? gainPathPrecise(levelDb, count, 0, cfg, threshNow, ratio, knee)
                                         : gainPath(levelDb[0], 0, cfg, threshNow, ratio, knee);
                // (0 dB is the common case below threshold: skip the exp.)
                const T gainLin = (loopGR == T(0)) ? T(1) : decibelsToGain(loopGR);
                blockGR = heldBack(rampedIn(loopGR, 0, rampLength, ramp), 0, slide, gainLine);
                if (amount < T(1)) blockGR *= amount;
                if (gainSmooth) blockGR = smoothedGain(blockGR, 0);
                const T outLin = (blockGR == loopGR) ? gainLin : decibelsToGain(blockGR);
                for (int ch = 0; ch < nCh; ++ch)
                    applyGain(audio, ch, i, gainLin, blockGR, audioDelay, charType,
                              colorAmt, makeupLin, mixVal, outLin);
                if (keyedLoop)
                    for (int ch = 0; ch < nCh; ++ch) fbLastOutput_[ch] = loopSource[ch] * gainLin;
            }
            else
            {
                T linkedDb[kDetectorPoints] = {};
                for (int j = 0; j < count; ++j)
                {
                    if (dominantLink)
                    {
                        linkedDb[j] = gainToDecibels(linkedLin[j]);
                        continue;
                    }
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
                    T heardGR = heldBack(rampedIn(loopGR, ch, rampLength, ramp), ch, slide, gainLine);
                    if (amount < T(1)) heardGR *= amount;
                    if (gainSmooth) heardGR = smoothedGain(heardGR, ch);
                    const T outLin = (heardGR == loopGR) ? gainLin : decibelsToGain(heardGR);
                    applyGain(audio, ch, i, gainLin, heardGR, audioDelay, charType,
                              colorAmt, makeupLin, mixVal, outLin);
                    if (keyedLoop) fbLastOutput_[ch] = loopSource[ch] * gainLin;
                    if (ch == 0 || heardGR < blockGR)
                        blockGR = heardGR; // Track worst-case GR for metering
                }
            }

            if (keyEqOn) keyEq_.advance(1);
            // Key listen: what the detectors heard goes out instead, late by
            // the latency and under full scale.
            if (listen)
            {
                keyMonitor_.processFrame(heard, nCh, listenDelay);
                for (int ch = 0; ch < nCh; ++ch) audio.getChannel(ch)[i] = heard[ch];
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
        bool program;     ///< Programme-dependent timing acts (see programTiming()).
        bool layered;     ///< The character runs its two layers (see layeredStep()).
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
        bool program;           ///< Programme-dependent timing acts.
        bool keyBalance;        ///< It reads the spectral balance of the key.
        bool keyEq;             ///< The equalizer of the key has bands in the path.
        bool listen;            ///< The key goes out instead of the audio.
        bool layered;           ///< The character runs its two layers.
        bool gainSmooth;        ///< The gain is smoothed after the ballistics.
        bool dominant;          ///< The pair links to its dominant component.
        int listenDelay;        ///< Samples the listened key waits: the latency of the audio path.
        bool gainLine;          ///< A latency is kept for the lookahead: the gain runs through its line.
        bool ramp;              ///< The lookahead brings the gain in along a ramp.
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
        T heard[kMaxChannels][kChunk];   // the key as the detectors read it (key listen)
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
                        ? computeGain(T(0), thr[k], rat[k], kn[k], ff.charType, ff.modeType, T(0)) * T(-0.5)
                        : T(0)) + analyticSmooth_.getNextValue();
                }
            }

            // B. Detection (each channel's detector keeps its own state).
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T* src = ((scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))
                                           : audio.getChannel(ch)) + o;
                T* lv = level[ch];
                if (ff.detType == DetectorType::Peak && !ff.scHpf && !ff.keyEq && !ff.listen && !ff.dominant)
                {
                    for (int k = 0; k < m; ++k) lv[k] = std::abs(detectorSafe(src[k]));
                }
                else
                {
                    T* key = heard[ch];
                    for (int k = 0; k < m; ++k)
                    {
                        T d = detectorSafe(src[k]);
                        if (ff.scHpf) d = applySidechainHPF(d, ch);
                        if (ff.keyEq) d = keyEq_.processSample(d, ch);
                        key[k] = d;
                        lv[k] = detectLevelLinear(d, ch, ff.detType);
                    }
                }
            }
            if (ff.keyEq) keyEq_.advance(m);

            // The correlation of the key, frame by frame, for the dominant link.
            double rho2[kChunk] = {};
            if (ff.dominant)
                for (int k = 0; k < m; ++k) rho2[k] = linkCorrelation(heard[0][k], heard[1][k]);

            // The spectral balance of the key, for the timing that follows it
            // (the key as the detector reads it, before the sidechain filter).
            double balance[kChunk];
            if (ff.keyBalance)
            {
                double lowSq[kChunk] = {}, highSq[kChunk] = {};
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* src = ((scCh > 0) ? sidechain.getChannel(std::min(ch, scCh - 1))
                                               : audio.getChannel(ch)) + o;
                    for (int k = 0; k < m; ++k) balanceBands(detectorSafe(src[k]), ch, lowSq[k], highSq[k]);
                }
                for (int k = 0; k < m; ++k) balance[k] = balanceAdvance(lowSq[k], highSq[k]);
            }
            else
            {
                std::fill_n(balance, m, balance_);
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
                    if (ff.dominant) mx = dominantLevel(level[0][k], level[1][k], rho2[k]);
                    else for (int ch = 1; ch < nCh; ++ch) mx = std::max(mx, level[ch][k]);
                    level[0][k] = gainToDecibels(mx);
                }
            }
            else
            {
                T dominantDb[kChunk] = {};
                if (ff.dominant)
                    for (int k = 0; k < m; ++k)
                        dominantDb[k] = gainToDecibels(dominantLevel(level[0][k], level[1][k], rho2[k]));
                for (int ch = 0; ch < nCh; ++ch)
                    for (int k = 0; k < m; ++k) level[ch][k] = gainToDecibels(level[ch][k]);
                if (nCh > 1 && ff.sLink > T(0))
                {
                    for (int k = 0; k < m; ++k)
                    {
                        T linked = level[0][k];
                        if (ff.dominant) linked = dominantDb[k];
                        else for (int ch = 1; ch < nCh; ++ch) linked = std::max(linked, level[ch][k]);
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
                if (m > 0) thresholdInUseDb_.store(thrNow[m - 1], std::memory_order_relaxed);
            }

            // D. Static curve, hold/range and ballistics: the serial part.
            //    SplitPolarity's level-adaptive release reads the previous
            //    frame's output, so its recursion also forms the linear gain
            //    (written over the consumed dB level).
            for (int g = 0; g < nGain; ++g)
            {
                const T* dryIn = (ff.lookahead > 0) ? delayed[g] : audio.getChannel(g) + o;
                runGainRecursion(level[g], gain[g], m, g, ff, thrUse, rat, kn,
                                 ff.splitAdaptive ? dryIn : nullptr, balance);
            }

            // What is heard of the gain: held back while the lookahead is
            // shorter than the latency kept for it, and scaled by the amount.
            const bool scaled = ff.paramsMoving || constants.amount < T(1);
            if (ff.gainLine || scaled || ff.gainSmooth || ff.ramp)
            {
                // The gain line at rest reads a whole delay; while the lookahead
                // moves its delay slides, and the ramp's length with it.
                const bool atRest = !ff.gainLine || gainDelayNow_ == static_cast<double>(ff.gainDelay);
                double slide[kChunk] = {};
                if (!atRest)
                    for (int k = 0; k < m; ++k) slide[k] = slideGainDelay(ff.gainDelay);
                for (int g = 0; g < nGain; ++g)
                {
                    T* gd = gain[g];
                    if (ff.ramp)
                        for (int k = 0; k < m; ++k)
                            gd[k] = rampedIn(gd[k], g, atRest ? ff.lookahead - ff.gainDelay
                                                              : ff.lookahead - static_cast<int>(std::ceil(slide[k])), true);
                    if (ff.gainLine && atRest)
                    {
                        auto& line = gainDelay_[static_cast<size_t>(g)];
                        for (int k = 0; k < m; ++k)
                        {
                            line.push(gd[k]);
                            gd[k] = line.read(ff.gainDelay);
                        }
                    }
                    else if (ff.gainLine)
                        for (int k = 0; k < m; ++k) gd[k] = heldBack(gd[k], g, slide[k], true);
                    if (scaled)
                        for (int k = 0; k < m; ++k)
                            if (amt[k] < T(1)) gd[k] *= amt[k];
                    if (ff.gainSmooth)
                        for (int k = 0; k < m; ++k) gd[k] = smoothedGain(gd[k], g);
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
            const bool removal = removalListen_.load(std::memory_order_relaxed);
            const bool plain = ff.lookahead == 0 && !ff.paramsMoving && constants.mix >= T(1) && !removal;
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
                    data[k] = removal ? input - wet : ((mixA[k] < T(1)) ? (input * (T(1) - mixA[k]) + out * mixA[k]) : out);
                }
                // Feedback history (pre-makeup) for a later topology switch.
                fbLastOutput_[ch] = wet;
            }

            // Key listen: what the detectors heard goes out instead, late by
            // the latency and under full scale.
            if (ff.listen)
                for (int k = 0; k < m; ++k)
                {
                    T frame[kMaxChannels];
                    for (int ch = 0; ch < nCh; ++ch) frame[ch] = heard[ch][k];
                    keyMonitor_.processFrame(frame, nCh, ff.listenDelay);
                    for (int ch = 0; ch < nCh; ++ch) audio.getChannel(ch)[o + k] = frame[ch];
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
            envMidDb_[ch] = envMidDb_[0];
            envDeepDb_[ch] = envDeepDb_[0];
            envHoldDb_[ch] = envHoldDb_[0];
            upwardGuardDb_[ch] = upwardGuardDb_[0];
            holdCounters_[ch] = holdCounters_[0];
            heldGrDb_[ch] = heldGrDb_[0];
            precLevel_[ch] = precLevel_[0];
            precTarget_[ch] = precTarget_[0];
            programState_[ch] = programState_[0];
            layerLevelDb_[ch] = layerLevelDb_[0];
            layerPeakDb_[ch] = layerPeakDb_[0];
            gainSmoothA_[ch] = gainSmoothA_[0];
            gainSmoothB_[ch] = gainSmoothB_[0];
            landingDb_[ch] = landingDb_[0];
            networkLow_[ch] = networkLow_[0];
        }
        if (holdEveryPeak_ && holdWindows_ && holdSamples_ > 0)
            for (int ch = 1; ch < nCh; ++ch) holdWindows_[static_cast<size_t>(ch)] = holdWindows_[0];
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
     * @param balance Spectral balance of the key per frame (see
     *               setSpectralTiming()), read by programme-dependent timing.
     */
    void runGainRecursion(T* inDb, T* out, int m, int ch, const FeedForwardConfig& ff,
                          const T* thr, const T* rat, const T* kn, const T* dryIn,
                          const double* balance) noexcept
    {
        double fast = envFastDb_[ch];
        double slow = envSlowDb_[ch];
        double mid = envMidDb_[ch];
        double deep = envDeepDb_[ch];
        double hold = envHoldDb_[ch];
        T guard = upwardGuardDb_[ch];
        T held = heldGrDb_[ch];
        int counter = holdCounters_[ch];
        const T range = rangeDb_.load(std::memory_order_relaxed);
        const int soft = rangeShape_.load(std::memory_order_relaxed);
        const int holdN = holdSamples_;
        SlidingDeepest* const window = holdWindow(ch);
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
        else if (ff.modeType == Mode::Both)
        {
            for (int k = 0; k < m; ++k)
            {
                guard = std::max(inDb[k], guard - upwardGuardDecay_);
                tgtA[k] = downwardCurve(inDb[k], thr[k], rat[k], kn[k], ff.charType)
                        + lift(inDb[k], thr[k], kn[k], guard);
            }
        }
        else if (ff.modeType == Mode::Leveler)
        {
            for (int k = 0; k < m; ++k) tgtA[k] = levelerCurve(inDb[k], thr[k]);
        }
        else
        {
            for (int k = 0; k < m; ++k)
                tgtA[k] = downwardCurve(inDb[k], thr[k], rat[k], kn[k], ff.charType);
        }

        if (ff.layered)
        {
            // Two layers, each towards its own curve; the gain is their sum.
            const CharacterProfile profile = profileOf(ff.charType);
            double level = layerLevelDb_[ch], peak = layerPeakDb_[ch];
            for (int k = 0; k < m; ++k)
            {
                fast = layeredStep(inDb[k], thr[k], rat[k], kn[k], profile, range, holdN, soft,
                                   held, counter, level, peak, window);
                out[k] = static_cast<T>(fast);
            }
            layerLevelDb_[ch] = level;
            layerPeakDb_[ch] = peak;
        }
        else if (dryIn != nullptr)
        {
            // Level-adaptive release (SplitPolarity): full-scale output
            // halves the release time constant, as in applyBallistics().
            T lastWet = fbLastOutput_[ch];
            for (int k = 0; k < m; ++k)
            {
                const T tgt = holdAndRange(tgtA[k], range, holdN, held, counter, soft, window);
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
                    y = ballisticsStep(static_cast<double>(tgt), fast, slow, mid, deep, charAttCoeff_,
                                       c + outputLevel * (c * c - c));
                }
                const T gr = static_cast<T>(y);
                out[k] = gr;
                const T gl = (gr == T(0)) ? T(1) : decibelsToGain(gr);
                inDb[k] = gl;
                lastWet = dryIn[k] * gl;
            }
        }
        else if (ff.program)
        {
            ProgramState state = programState_[ch];
            for (int k = 0; k < m; ++k)
            {
                const T tgt = holdAndRange(tgtA[k], range, holdN, held, counter, soft, window);
                out[k] = static_cast<T>(programStep(static_cast<double>(tgt), fast, state, balance[k]));
            }
            programState_[ch] = state;
        }
        else
        {
            for (int k = 0; k < m; ++k)
            {
                const T tgt = holdAndRange(tgtA[k], range, holdN, held, counter, soft, window);
                const double y = ff.decoupled
                    ? decoupledStep(static_cast<double>(tgt), hold, fast, decHoldCoeff_)
                    : ballisticsStep(static_cast<double>(tgt), fast, slow, mid, deep, charAttCoeff_, charRelCoeff_);
                out[k] = static_cast<T>(y);
            }
        }

        envFastDb_[ch] = fast;
        envSlowDb_[ch] = slow;
        envMidDb_[ch] = mid;
        envDeepDb_[ch] = deep;
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
        if (cfg.modeType == Mode::Upward || cfg.modeType == Mode::Both)
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

        if (cfg.layered)
        {
            const double shown = layeredStep(inputDb, thresh, ratio, knee, profileOf(cfg.charType),
                                             rangeDb_.load(std::memory_order_relaxed), holdSamples_,
                                             rangeShape_.load(std::memory_order_relaxed),
                                             heldGrDb_[ch], holdCounters_[ch],
                                             layerLevelDb_[ch], layerPeakDb_[ch], holdWindow(ch));
            envFastDb_[ch] = shown;
            return static_cast<T>(shown);
        }

        T targetGR_Db = computeGain(inputDb, thresh, ratio, knee, cfg.charType, cfg.modeType, guardDb);
        targetGR_Db = applyHoldAndRange(targetGR_Db, ch);
        if (cfg.program)
            return static_cast<T>(programStep(static_cast<double>(targetGR_Db), envFastDb_[ch],
                                              programState_[static_cast<size_t>(ch)], balance_));
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
        const Colour colour = profileOf(charType).colour;
        const bool pushPull = colorAmt > T(0) && colour == Colour::PushPull;
        const bool extras = audioExtras_ && !pushPull;
        T wet = pushPull ? pushPullElement(input, outLin, colorAmt)
                         : (extras ? gained(input, outLin, ch) : input * outLin);
        if (colorAmt > T(0) && colour == Colour::SecondOrder)
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
        fbLastOutput_[ch] = (outLin == gainLin && !extras) ? wet : input * gainLin;
        const T kept = wet;
        wet *= makeupLin;

        // Parallel (New York) mix done inline: the dry reference is `input`,
        // which carries the SAME lookahead delay as the wet, so dry and wet
        // stay phase-aligned (the previous DryWetMixer captured the UNDELAYED
        // input and comb-filtered when lookahead + mix<1 were combined).
        data[i] = removalListen_.load(std::memory_order_relaxed) ? input - kept
                : ((mixVal < T(1)) ? (input * (T(1) - mixVal) + wet * mixVal) : wet);
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

        // Character ballistics: time constants of the dB-domain envelopes,
        // from the numbers of the character (see CharacterProfile). The knobs
        // are held to its limits first; a character without a memory stage
        // has the fast one stand in for it.
        const CharacterProfile& profile = profileOf(character_.load(std::memory_order_relaxed));
        bool threeStageWanted = false;
        {
            const T att = std::clamp(attMs, profile.attackFloorMs, profile.attackCeilMs);
            const T rel = std::clamp(relMs, profile.releaseFloorMs, profile.releaseCeilMs);
            charAttCoeff_   = tc(att);
            charRelCoeff_   = tc(rel * profile.fastRelease);
            charFastWeight_ = profile.fastWeight;
            if (profile.fastWeight >= T(1))
            {
                charSlowRelCoeff_ = charRelCoeff_;
                charChargeCoeff_  = charAttCoeff_;
            }
            else
            {
                charSlowRelCoeff_ = tc(std::min(rel * profile.slowRelease, profile.slowReleaseCeilMs));
                charChargeCoeff_  = tc(std::clamp(rel * profile.chargeScale, profile.chargeFloorMs,
                                                  profile.chargeCeilMs));
            }

            // The three-stage cell of Opto (setOptoCell()): a faster fast
            // stage and the medium memory; the deep memory is the one above.
            threeStageWanted = character_.load(std::memory_order_relaxed) == Character::Opto
                            && optoCell_.load(std::memory_order_relaxed) == OptoCell::ThreeStage;
            if (threeStageWanted)
            {
                charRelCoeff_       = tc(rel * static_cast<T>(kOptoThreeFastRelease));
                charMidRelCoeff_    = tc(rel * static_cast<T>(kOptoMidRelease));
                charMidChargeCoeff_ = tc(std::clamp(rel * static_cast<T>(kOptoMidCharge),
                                                    static_cast<T>(kOptoMidChargeFloorMs),
                                                    static_cast<T>(kOptoMidChargeCeilMs)));
            }

            // Decoupled detector (Clean, feed-forward): the smoothing pole's
            // release side is a tenth of the release (never faster than the
            // attack, never slower than half the release); the hold stage
            // takes the rest of the release so the series t63 lands on the
            // knob.
            const T smoothMs = std::min(std::max(relMs * T(0.1), attMs), relMs * T(0.5));
            decSmoothCoeff_ = tc(smoothMs);
            decHoldCoeff_ = tc(static_cast<T>(decoupledHoldTau(
                static_cast<double>(relMs), static_cast<double>(smoothMs))));

            // The layers of a layered character run on rates (the share of
            // the distance covered per sample), which their pace then divides.
            if (profile.levelShare > T(0))
            {
                const double perMs = 1000.0 / static_cast<double>(fs);
                peakAttRate_ = perMs / static_cast<double>(att);
                peakRelRate_ = perMs / static_cast<double>(rel);
                layerAttRate_ = perMs / static_cast<double>(std::max(att * profile.levelAttack,
                                                                      profile.levelAttackFloorMs));
                layerRelRate_ = perMs / static_cast<double>(rel * profile.levelRelease);
            }
        }

        // The smoothing of the gain after the ballistics: two sections of
        // half its time each.
        {
            const double ms = static_cast<double>(gainSmoothMs_.load(std::memory_order_relaxed));
            gainSmoothRate_ = (ms > 0.0) ? 1.0 - std::exp(-2000.0 / (static_cast<double>(fs) * ms)) : 0.0;
        }

        // The landing of the release and the network of the control path.
        {
            const double rate = static_cast<double>(fs);
            const double landing = static_cast<double>(landingMs_.load(std::memory_order_relaxed));
            landingRate_ = (landing > 0.0) ? 1.0 - std::exp(-1000.0 / (rate * landing)) : 0.0;
            const double depth = static_cast<double>(networkDepthDb_.load(std::memory_order_relaxed));
            const double corner = static_cast<double>(networkCornerHz_.load(std::memory_order_relaxed));
            networkShare_ = (depth > 0.0) ? std::pow(10.0, -depth / 20.0) : 1.0;
            networkRate_ = 1.0 - std::exp(-2.0 * std::numbers::pi * corner / rate);
            leakCoeff_ = static_cast<T>(1.0 - std::exp(-2.0 * std::numbers::pi * kLeakCornerHz / rate));
            liftBelowDb_ = liftBelow_.load(std::memory_order_relaxed);
            liftRatioIn_ = std::max(liftRatio_.load(std::memory_order_relaxed), T(1));
            liftRangeDb_ = liftRange_.load(std::memory_order_relaxed);
            asymDecay_ = static_cast<T>(std::exp(-1.0 / (rate * kAsymCrestSeconds)));
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
            subMidRelCoeff_    = eighth(charMidRelCoeff_);
            subMidChargeCoeff_ = eighth(charMidChargeCoeff_);
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
                if (w0 >= 1.0)
                {
                    s = shown;
                    envMidDb_[static_cast<size_t>(ch)] = shown;
                    envDeepDb_[static_cast<size_t>(ch)] = shown;
                }
                f = (shown - (1.0 - w1) * s) / w1;
            }
        }

        // The three-stage cell starts with both memories where the memory
        // stage stands, so the gain does not jump when it engages.
        if (threeStageWanted && !threeStage_)
        {
            envMidDb_ = envSlowDb_;
            envDeepDb_ = envSlowDb_;
        }
        threeStage_ = threeStageWanted;

        // A feedback loop closes around the attack stage, which multiplies the
        // raw ballistics speed by (1 + loop gain). Re-derive the coefficient
        // from rho = coeff/(1 + w beta A) so the OBSERVED attack t63 stays on
        // the knob, exactly like the hardware panels (the 1176's 20-800 us
        // figures are measured results, not RC values). A is the
        // linear-region loop gain (R - 1) in both directions; Varimu uses its
        // base ratio (the progressive part is level-dependent).
        {
            const bool fbLoop = topologyFor(profile, topology_.load(std::memory_order_relaxed))
                             == Topology::FeedBack;
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

        // Programme-dependent timing: the two times as decays per sample
        // (the coefficient is fastDecay() of them, scaled), and the poles of
        // its measures.
        {
            const double rate = static_cast<double>(fs);
            const auto pole = [rate](double hz) { return 1.0 - std::exp(-2.0 * std::numbers::pi * hz / rate); };
            const auto over = [rate](double seconds) { return 1.0 - std::exp(-1.0 / (rate * seconds)); };
            attackX_ = -std::log(std::max(charAttCoeff_, 1e-300));
            releaseX_ = -std::log(std::max(charRelCoeff_, 1e-300));
            subRate_ = pole(kBalanceFloorHz);
            lowRate_ = pole(kBalanceLowHz);
            midRate_ = pole(kBalanceHighHz);
            highRate_ = pole(std::min(kBalanceCeilingHz, 0.45 * rate));
            balanceRate_ = over(kBalanceSeconds);
            // The windows of the two measures count in the knobs' own times, so
            // "sustained" and "the leading edge" mean the same at any setting.
            const double attackSeconds = static_cast<double>(std::clamp(attMs, profile.attackFloorMs, profile.attackCeilMs)) / 1000.0;
            const double releaseSeconds = static_cast<double>(std::clamp(relMs, profile.releaseFloorMs, profile.releaseCeilMs)) / 1000.0;
            sustainRate_ = over(std::clamp(kSustainReleases * releaseSeconds, kSustainFloorSeconds, kSustainCeilSeconds));
            emphasisRate_ = over(std::clamp(kEmphasisAttacks * attackSeconds, kEmphasisFloorSeconds, kEmphasisCeilSeconds));
        }

        // Upward silence-guard envelope: 60 dB/s linear decay.
        upwardGuardDecay_ = T(60) / fs;

        // AC coupling (~10 Hz) of the FET color's 2nd-order term.
        fetDcCoeff_ = T(1) - std::exp(T(-2) * std::numbers::pi_v<T> * T(10) / fs);

        lookaheadSamples_ = static_cast<int>(fs * std::clamp(
            lookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000));
        latencySamples_ = std::max(lookaheadSamples_, static_cast<int>(fs * std::clamp(
            maxLookaheadMs_.load(std::memory_order_relaxed), T(0), T(10)) / T(1000)));
        latencyKept_ = maxLookaheadMs_.load(std::memory_order_relaxed) > T(0);
        autoRefFall_ = kAutoRefFallDb / static_cast<double>(fs);

        holdSamples_ = static_cast<int>(fs * holdMs_.load(std::memory_order_relaxed) / T(1000));
        rampShape_ = lookaheadShape_.load(std::memory_order_relaxed) == LookaheadShape::Ramp;
        // The hold's windows follow its length and its mode; a change of
        // either starts the hold again.
        const bool everyPeak = holdMode_.load(std::memory_order_relaxed) == HoldMode::EveryPeak;
        if (everyPeak != holdEveryPeak_ || (everyPeak && holdSamples_ != holdWindowFor_))
        {
            if (holdWindows_)
                for (int ch = 0; ch < kMaxChannels; ++ch) holdWindows_[static_cast<size_t>(ch)].configure(holdSamples_);
            holdWindowFor_ = holdSamples_;
            holdCounters_.fill(0);
            heldGrDb_.fill(T(0));
            holdEveryPeak_ = everyPeak;
        }
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
                            heldGrDb_[ch], holdCounters_[ch], rangeShape_.load(std::memory_order_relaxed),
                            holdWindow(ch));
    }

    /**
     * @brief The deepest gain change of a sliding window (HoldMode::EveryPeak).
     *
     * The window is cut in at most kBlocks blocks. The deepest value of each
     * finished block waits in a queue kept in falling order of depth (a
     * block that a later, deeper one covers is dropped), so the deepest of
     * the window is the front of the queue or the block in progress. A value
     * is held for the window and at most one block more.
     */
    struct SlidingDeepest
    {
        static constexpr int kBlocks = 256;         ///< Blocks the window is cut in, at most.
        static constexpr int kSlots = 2 * kBlocks;  ///< Room of the queue (a power of two, over kBlocks + 1).
        std::array<T, kSlots> depth {};
        std::array<uint32_t, kSlots> block {};
        int head = 0, count = 0, inBlock = 0, blockLength = 1, blocks = 1;
        uint32_t index = 0;
        T current = T(0);

        /** @brief Sets the window in samples and empties it. */
        void configure(int windowSamples) noexcept
        {
            const int w = std::max(1, windowSamples);
            blockLength = (w + kBlocks - 1) / kBlocks;
            blocks = (w + blockLength - 1) / blockLength;
            clear();
        }

        void clear() noexcept
        {
            head = count = inBlock = 0;
            index = 0;
            current = T(0);
        }

        /**
         * @brief One value in, the deepest of the window out.
         * @param gr        Gain change in dB (either sign: the deepest is the largest magnitude).
         * @param perSample Values pushed per sample (the window is in samples).
         */
        [[nodiscard]] T push(T gr, int perSample = 1) noexcept
        {
            if (inBlock == 0 || std::abs(gr) >= std::abs(current)) current = gr;
            T deepest = current;
            if (count > 0 && std::abs(depth[static_cast<size_t>(head)]) > std::abs(deepest)) deepest = depth[static_cast<size_t>(head)];
            if (++inBlock >= blockLength * perSample)
            {
                // The block is done: it covers every queued block that is no deeper.
                while (count > 0 && std::abs(depth[static_cast<size_t>((head + count - 1) & (kSlots - 1))]) <= std::abs(current)) --count;
                const auto at = static_cast<size_t>((head + count) & (kSlots - 1));
                depth[at] = current;
                block[at] = index;
                ++count;
                ++index;
                inBlock = 0;
                // A block leaves when the window has passed it.
                while (count > 0 && index - block[static_cast<size_t>(head)] > static_cast<uint32_t>(blocks))
                {
                    head = (head + 1) & (kSlots - 1);
                    --count;
                }
            }
            return deepest;
        }
    };

    /**
     * @brief The lookahead as a ramp (LookaheadShape::Ramp).
     *
     * Each gain change is replaced by the deepest of the last `length` + 1
     * (an exact sliding window: a queue in falling order of depth), and that
     * runs through two moving averages whose lengths add up to `length` + 2.
     * A reduction that the gain computer asks for at one sample is so
     * reached, exactly, `length` samples later, when that sample leaves the
     * lookahead's delay, along an S-shaped ramp; and it stays for `length`
     * samples after the last sample that asked for it. With no gain change
     * for as long as the window the output is exactly 0 dB.
     */
    struct LookaheadRamp
    {
        std::vector<T> deepVal, boxA, boxB;   ///< The queue of the sliding window; the histories of the two averages.
        std::vector<uint32_t> deepPos;        ///< When each queued value came.
        uint32_t t = 0;                       ///< Values taken so far (wraps).
        int mask = 0, head = 0, count = 0, length = 0, lenA = 1, lenB = 1, zerosA = 0, zerosB = 0;
        double sumA = 0.0, sumB = 0.0;

        /** @brief Allocates for a longest ramp. */
        void prepare(int maxLength)
        {
            size_t n = 8;
            while (n < static_cast<size_t>(std::max(0, maxLength)) + 4) n <<= 1;
            deepVal.assign(n, T(0));
            boxA.assign(n, T(0));
            boxB.assign(n, T(0));
            deepPos.assign(n, 0u);
            mask = static_cast<int>(n) - 1;
            clear();
        }

        void clear() noexcept
        {
            std::fill(deepVal.begin(), deepVal.end(), T(0));
            std::fill(boxA.begin(), boxA.end(), T(0));
            std::fill(boxB.begin(), boxB.end(), T(0));
            t = 0;
            head = count = length = zerosA = zerosB = 0;
            lenA = lenB = 1;
            sumA = sumB = 0.0;
        }

        /** @brief Moves the ramp's length to `wanted`, a sample at a time (the sums follow, each step in constant time). */
        void setLength(int wanted) noexcept
        {
            wanted = std::clamp(wanted, 0, mask - 3);
            while (length != wanted)
            {
                const int next = length + (wanted > length ? 1 : -1);
                const int a = (next + 2) / 2, b = next + 2 - a;
                // The averages hold the last lenA and lenB values (the last one came at t - 1).
                if (a > lenA) sumA += static_cast<double>(boxA[(t - 1u - static_cast<uint32_t>(lenA)) & static_cast<uint32_t>(mask)]);
                else if (a < lenA) sumA -= static_cast<double>(boxA[(t - static_cast<uint32_t>(lenA)) & static_cast<uint32_t>(mask)]);
                if (b > lenB) sumB += static_cast<double>(boxB[(t - 1u - static_cast<uint32_t>(lenB)) & static_cast<uint32_t>(mask)]);
                else if (b < lenB) sumB -= static_cast<double>(boxB[(t - static_cast<uint32_t>(lenB)) & static_cast<uint32_t>(mask)]);
                lenA = a;
                lenB = b;
                length = next;
                zerosA = zerosB = 0;
            }
        }

        /** @brief One gain change in dB in, the ramped one out. */
        [[nodiscard]] T process(T gr) noexcept
        {
            const auto m = static_cast<uint32_t>(mask);
            // The deepest of the last length + 1 (the gain changes share a sign: the largest magnitude).
            const T depth = std::abs(gr);
            while (count > 0 && std::abs(deepVal[static_cast<size_t>((head + count - 1) & mask)]) <= depth) --count;
            const auto at = static_cast<size_t>((head + count) & mask);
            deepVal[at] = gr;
            deepPos[at] = t;
            ++count;
            while (t - deepPos[static_cast<size_t>(head)] > static_cast<uint32_t>(length))
            {
                head = (head + 1) & mask;
                --count;
            }
            const T deep = deepVal[static_cast<size_t>(head)];
            // Two moving averages; a sum that has seen nothing but zeros for its whole length is zero.
            sumA += static_cast<double>(deep) - static_cast<double>(boxA[(t - static_cast<uint32_t>(lenA)) & m]);
            boxA[t & m] = deep;
            zerosA = deep == T(0) ? std::min(zerosA + 1, lenA) : 0;
            if (zerosA >= lenA) sumA = 0.0;
            const T mid = static_cast<T>(sumA / lenA);
            sumB += static_cast<double>(mid) - static_cast<double>(boxB[(t - static_cast<uint32_t>(lenB)) & m]);
            boxB[t & m] = mid;
            zerosB = mid == T(0) ? std::min(zerosB + 1, lenB) : 0;
            if (zerosB >= lenB) sumB = 0.0;
            ++t;
            return static_cast<T>(sumB / lenB);
        }
    };

    /** @brief A gain change through the lookahead's ramp of its gain computer, or as it came. */
    [[nodiscard]] T rampedIn(T grDb, int g, int length, bool run) noexcept
    {
        if (!run || g >= rampCount_) return grDb;
        auto& ramp = lookaheadRamps_[static_cast<size_t>(g)];
        ramp.setLength(length);
        return ramp.process(grDb);
    }

    /** @return The window of a gain computer's hold in HoldMode::EveryPeak (nullptr in Deepest). */
    [[nodiscard]] SlidingDeepest* holdWindow(int ch) const noexcept
    {
        return (holdEveryPeak_ && holdWindows_) ? &holdWindows_[static_cast<size_t>(ch)] : nullptr;
    }

    /**
     * @brief Hold and range on explicit state (shared by both block paths).
     * @param targetGR_Db Static gain change target in dB.
     * @param range       Maximum |gain change| in dB.
     * @param holdSamples Hold length in samples (0 = off).
     * @param held        Held gain change (state).
     * @param counter     Remaining hold samples (state).
     * @param soft        How the target meets the range: a RangeShape, as its number.
     * @param window      The sliding window that holds instead (HoldMode::EveryPeak), or nullptr.
     * @param perSample   Targets given per sample (the window counts samples).
     */
    [[nodiscard]] static T holdAndRange(T targetGR_Db, T range, int holdSamples,
                                        T& held, int& counter, int soft = 0,
                                        SlidingDeepest* window = nullptr, int perSample = 1) noexcept
    {
        if (soft == static_cast<int>(RangeShape::Asymptotic) && range > T(0))
        {
            // g / (1 + (g / R)^4)^(1/4): g while it is small, R at most.
            const T x = targetGR_Db / range, x2 = x * x;
            targetGR_Db /= std::sqrt(std::sqrt(T(1) + x2 * x2));
        }
        else if (soft == static_cast<int>(RangeShape::Local) && range > T(0))
        {
            // A quadratic bend over a stretch centred on the range: the target
            // under it, the range over it, and the slope continuous at both ends.
            const T size = std::abs(targetGR_Db);
            const T stretch = std::min(static_cast<T>(kLocalRangeStretchDb), range);
            const T past = size - (range - stretch / T(2));
            if (past >= stretch) targetGR_Db = std::copysign(range, targetGR_Db);
            else if (past > T(0)) targetGR_Db = std::copysign(size - past * past / (T(2) * stretch), targetGR_Db);
        }
        else
        {
            targetGR_Db = std::clamp(targetGR_Db, -range, range);
        }

        if (holdSamples > 0 && window != nullptr) return window->push(targetGR_Db, perSample);
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

    /** One frame of the key's covariance for the dominant link; returns the
     *  squared correlation of the pair (0 when either side is silent). */
    double linkCorrelation(T keyL, T keyR) noexcept
    {
        const double l = static_cast<double>(keyL);
        const double r = static_cast<double>(keyR);
        linkLL_ += linkCoeff_ * (l * l - linkLL_);
        linkRR_ += linkCoeff_ * (r * r - linkRR_);
        linkLR_ += linkCoeff_ * (l * r - linkLR_);
        const double den = linkLL_ * linkRR_;
        return den > 1e-60 ? std::min(1.0, linkLR_ * linkLR_ / den) : 0.0;
    }

    /** The covariance decays toward zero in silence: flushed before it
     *  reaches the subnormal range (once per block). */
    void flushLinkCovariance() noexcept
    {
        if (linkLL_ < 1e-30) linkLL_ = 0.0;
        if (linkRR_ < 1e-30) linkRR_ = 0.0;
        if (std::abs(linkLR_) < 1e-30) linkLR_ = 0.0;
    }

    /** The level of the pair's dominant component from the two detector
     *  levels and the squared correlation of the key: the larger eigenvalue
     *  of their covariance, halved, so a source in both channels reads its
     *  level in each and an equal-power pan reads alike anywhere. */
    [[nodiscard]] static T dominantLevel(T levelL, T levelR, double rho2) noexcept
    {
        return dominantPairLevel(levelL, levelR, rho2);
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

            case DetectorType::Hybrid:
            {
                // The root of twice the mean square is the peak of a steady
                // tone; the rectified sample is over it only while the level
                // rises faster than the mean can.
                double& meanSq = hybridMeanSq_[static_cast<size_t>(ch)];
                const double x = static_cast<double>(sample);
                meanSq += hybridRate_ * (x * x - meanSq);
                level = static_cast<T>(std::max(std::sqrt(2.0 * meanSq), std::abs(x)));
                break;
            }
        }
        return level;
    }

    // ---- The colour stage (setCharacterStage()) ----

    /**
     * @brief Runs one place of the colour stage over a block, fading its share
     *        towards whether it is wanted (20 ms). Idle, it costs nothing; it
     *        starts from clean memory when it comes in.
     */
    void runStage(CharacterStage<T>& stage, double& weight, bool wanted, AudioBufferView<T> buffer) noexcept
    {
        if (!(sampleRate_ > 0) || stageFade_.empty()) return;
        const double target = wanted ? 1.0 : 0.0;
        if (weight == 0.0 && target == 0.0) return;
        if (weight == 0.0) stage.reset();
        if (weight == 1.0 && target == 1.0)
        {
            stage.process(buffer);
            return;
        }
        const int nCh = std::min(buffer.getNumChannels(), stageScratch_.getNumChannels());
        const int step = stageScratch_.getNumSamples();
        for (int off = 0; off < buffer.getNumSamples(); off += step)
        {
            const int m = std::min(step, buffer.getNumSamples() - off);
            auto part = buffer.getSubView(off, m);
            auto copy = stageScratch_.toView().getSubView(0, m);
            for (int ch = 0; ch < nCh; ++ch) std::copy_n(part.getChannel(ch), m, copy.getChannel(ch));
            stage.process(AudioBufferView<T>(copy));
            for (int i = 0; i < m; ++i)
            {
                weight = target > weight ? std::min(target, weight + stageFadeStep_)
                                         : std::max(target, weight - stageFadeStep_);
                stageFade_[static_cast<size_t>(i)] = weight;
            }
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* y = part.getChannel(ch);
                const T* s = copy.getChannel(ch);
                for (int i = 0; i < m; ++i)
                    y[i] += static_cast<T>(stageFade_[static_cast<size_t>(i)]) * (s[i] - y[i]);
            }
        }
    }

    // ---- Characters as data ----

    /** @brief The harmonic signature a character can add (setCharacterColor()). */
    enum class Colour
    {
        None,
        SecondOrder,    ///< Second-order modulation of the element by the signal across it.
        PushPull        ///< The gain applied by the curve of a push-pull pair biased by it.
    };

    /**
     * @struct CharacterProfile
     * @brief What makes a character: the numbers one engine reads.
     *
     * Every character has a fast stage; where fastWeight is under 1 a memory
     * stage runs beside it, charging while the reduction lasts and letting
     * go slowly. The knobs are held to the limits of the profile before any
     * stage takes its share of them.
     */
    struct CharacterProfile
    {
        T attackFloorMs, attackCeilMs;      ///< Limits of the attack knob.
        T releaseFloorMs, releaseCeilMs;    ///< Limits of the release knob.
        T fastRelease;                      ///< Release of the fast stage, as a multiple of the knob.
        T slowRelease, slowReleaseCeilMs;   ///< Release of the memory stage: a multiple of the knob, up to a ceiling.
        T chargeScale;                      ///< Time the memory stage takes to charge, as a multiple of the release knob...
        T chargeFloorMs, chargeCeilMs;      ///< ...held to these.
        T fastWeight;                       ///< Share of the fast stage in the gain (1: no memory stage).
        T kneeFloorDb;                      ///< Narrowest knee the element can form (0: any).
        T ratioGrowthDb;                    ///< Decibels over the threshold that double the ratio (0: a fixed ratio).
        bool ownLoop;                       ///< It detects on its own output with a peak rectifier, whatever Topology and Detector say.
        Colour colour;                      ///< Its harmonic signature.
        bool decoupled;                     ///< setCleanBallistics() applies to it.
        bool programTiming;                 ///< Programme-dependent timing applies to it.
        bool forwardOnly;                   ///< It runs in feed-forward whatever Topology says.
        T levelShare;                       ///< Share of the slope a levelling layer takes (0: one layer).
        T levelLeadDb;                      ///< Decibels under the threshold where the levelling layer starts.
        T levelAttack, levelAttackFloorMs;  ///< Its attack: a multiple of the knob, and the least it is.
        T levelRelease;                     ///< Its release, as a multiple of the knob.
        T attackPaceDb, releasePaceDb;      ///< Decibels of distance to the target that double the time of a layer.
    };

    static constexpr T kNoLimit = std::numeric_limits<T>::infinity();

    // The three-stage cell of Opto (setOptoCell()). The memory share of 0.4
    // splits 0.625 / 0.375 between the medium stage (charge over one release
    // time, 30 ms to 1 s; release over five) and the deep one; with the fast
    // stage at 0.725 release times, a fully charged cell recovers half its
    // reduction in one release time: 0.6 e^(-1/0.725) + 0.4 (0.625 e^(-1/5)
    // + 0.375 e^(-1/25)) = 0.500.
    static constexpr double kOptoThreeFastRelease = 0.725;
    static constexpr double kOptoMidShare = 0.625;
    static constexpr double kOptoMidRelease = 5.0;
    static constexpr double kOptoMidCharge = 1.0;
    static constexpr double kOptoMidChargeFloorMs = 30.0;
    static constexpr double kOptoMidChargeCeilMs = 1000.0;

    /** @brief The characters, in the order of Character. */
    static constexpr std::array<CharacterProfile, 5> kProfiles { {
        // Clean: one envelope on the knobs.
        { T(0.01), kNoLimit, T(1), kNoLimit, T(1), T(1), kNoLimit, T(0), T(0), T(0), T(1),
          T(0), T(0), false, Colour::None, true, true,
          false, T(0), T(0), T(0), T(0), T(0), T(0), T(0) },
        // Opto: the release knob is the 50 % recovery time. With a 0.6/0.4
        // split, a fast stage of 0.61 times the release puts the half
        // recovery on the knob while the memory tail decays some 35 times
        // slower, up to the 5 s of the cell; the memory charges over four
        // release times (0.1 to 2 s), so only sustained compression builds
        // the long tail. The 10 ms attack floor is the cell's.
        { T(10), kNoLimit, T(1), kNoLimit, T(0.61), T(25), T(5000), T(4), T(100), T(2000), T(0.6),
          T(10), T(0), false, Colour::None, false, false,
          false, T(0), T(0), T(0), T(0), T(0), T(0), T(0) },
        // FET: knobs of 20 to 800 us and 50 to 1100 ms. With a 0.75/0.25
        // split the compound release passes its t63 at the knob, and a
        // 150 ms charge gives the tail that depends on the programme.
        { T(0.02), T(0.8), T(50), T(1100), T(0.7), T(2.5), kNoLimit, T(0), T(150), T(150), T(0.75),
          T(0), T(0), true, Colour::SecondOrder, false, false,
          false, T(0), T(0), T(0), T(0), T(0), T(0), T(0) },
        // Varimu: one envelope, a ratio that grows with the level, a knee
        // that cannot be hard, and the gain through a push-pull pair.
        { T(0.01), kNoLimit, T(1), kNoLimit, T(1), T(1), kNoLimit, T(0), T(0), T(0), T(1),
          T(10), T(40), false, Colour::PushPull, false, true,
          false, T(0), T(0), T(0), T(0), T(0), T(0), T(0) },
        // Layered: a levelling layer with half the slope from 6 dB under the
        // threshold, eight times the attack (30 ms at least) and six times
        // the release, under a layer on the knobs; both take twice as long
        // 12 dB (attack) or 6 dB (release) away from their target.
        { T(0.01), kNoLimit, T(1), kNoLimit, T(1), T(1), kNoLimit, T(0), T(0), T(0), T(1),
          T(6), T(0), false, Colour::None, false, false,
          true, T(0.5), T(6), T(8), T(30), T(6), T(12), T(6) },
    } };

    /** @brief The profile of a character. */
    [[nodiscard]] static constexpr CharacterProfile profileOf(Character type) noexcept
    {
        const int last = static_cast<int>(kProfiles.size()) - 1;
        return kProfiles[static_cast<size_t>(std::clamp(static_cast<int>(type), 0, last))];
    }

    /** @brief The topology a character runs in when the user asks for one. */
    [[nodiscard]] static constexpr Topology topologyFor(const CharacterProfile& profile, Topology asked) noexcept
    {
        return profile.ownLoop ? Topology::FeedBack : (profile.forwardOnly ? Topology::FeedForward : asked);
    }

    /** @brief Whether a character runs its two layers: it has them, and the mode is Downward. */
    [[nodiscard]] static constexpr bool isLayered(const CharacterProfile& profile, Mode modeType) noexcept
    {
        return profile.levelShare > T(0) && modeType == Mode::Downward;
    }

    /** @brief The detector a character reads with when the user asks for one. */
    [[nodiscard]] static constexpr DetectorType detectorFor(const CharacterProfile& profile, DetectorType asked) noexcept
    {
        return profile.ownLoop ? DetectorType::Peak : asked;
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
        // Progressive compression: a remote-cutoff tube's effective ratio
        // grows with the level above the threshold.
        const T growth = profileOf(charType).ratioGrowthDb;
        if (growth > T(0) && excessDb > T(0))
            return ratio * (T(1) + excessDb / growth);
        return ratio;
    }

    /** @brief Physical knee floor of the character (dB). */
    [[nodiscard]] static T effectiveKneeFor(Character charType, T knee) noexcept
    {
        // Neither a remote-cutoff tube (Varimu) nor a photocell (Opto) can
        // form a hard corner: both transfers bend gradually over their whole
        // operating region, so the knee has a 10 dB physical floor.
        const T floor = profileOf(charType).kneeFloorDb;
        return (floor > T(0)) ? std::max(knee, floor) : knee;
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
        if (modeType == Mode::Leveler)
            return levelerCurve(inputDb, thresh);
        const T down = downwardCurve(inputDb, thresh, ratio, knee, charType);
        return modeType == Mode::Both ? down + lift(inputDb, thresh, knee, guardDb) : down;
    }

    /** @brief The mode a routing runs: the two-way and the levelling curves are
     *  feed-forward ones, and in a loop the curve is the ratio's. */
    [[nodiscard]] static constexpr Mode modeIn(Mode asked, Topology topoEff) noexcept
    {
        return (topoEff == Topology::FeedBack && asked != Mode::Upward) ? Mode::Downward : asked;
    }

    /** @brief Whether a mode only ever takes level off (the automatic makeups that add it back apply). */
    [[nodiscard]] static constexpr bool reducesOnly(Mode modeType) noexcept
    {
        return modeType == Mode::Downward || modeType == Mode::Leveler;
    }

    /**
     * @brief The levelling curve: the gain 1 / (1 + level / threshold), in
     *        decibels. Nothing far under the threshold, 6 dB at it, and all
     *        that the level is over it far above.
     */
    [[nodiscard]] static T levelerCurve(T inputDb, T thresh) noexcept
    {
        const T over = inputDb - thresh;
        if (over <= T(-120)) return T(0);
        return static_cast<T>(-20.0 / std::numbers::ln10) * std::log1p(decibelsToGain(over));
    }

    /** @brief The upward half of Mode::Both: the rise under the lower threshold, up to its range. */
    [[nodiscard]] T lift(T inputDb, T thresh, T knee, T guardDb) const noexcept
    {
        return std::min(computeGainUpward(inputDb, thresh - liftBelowDb_, liftRatioIn_, knee, guardDb), liftRangeDb_);
    }

    /**
     * @brief Downward static curve with the character's knee floor and ratio
     *        law (see computeGain).
     */
    [[nodiscard]] static T downwardCurve(T inputDb, T thresh, T ratio, T knee, Character charType) noexcept
    {
        const CharacterProfile profile = profileOf(charType);
        if (profile.levelShare > T(0))
        {
            const Layers layers = layeredCurve(inputDb, thresh, ratio, knee, profile);
            return layers.level + layers.peak;
        }
        const T effectiveRatio = effectiveRatioFor(charType, ratio, inputDb - thresh);
        const T effectiveKnee = effectiveKneeFor(charType, knee);
        return kneeCurve(inputDb, thresh, T(1) - T(1) / effectiveRatio, effectiveKnee);
    }

    /**
     * @brief A threshold, a slope (1 - 1/ratio) and a quadratic knee: the
     *        gain change at a level, in decibels.
     */
    [[nodiscard]] static T kneeCurve(T inputDb, T thresh, T slope, T knee) noexcept
    {
        // Branch-free pieces (the level crosses the threshold at random on
        // dense programme, where a branch per sample mispredicts): below the
        // threshold the linear piece is positive and min() returns 0.
        if (knee <= T(0))
            return std::min(thresh - inputDb, T(0)) * slope;

        // Soft knee: the quadratic below the knee top never exceeds the
        // linear piece, and with x clamped to the knee width it is the larger
        // one above it, so min() picks the right piece on every side.
        const T halfKnee = knee / T(2);
        const T x = std::clamp(inputDb - (thresh - halfKnee), T(0), knee);
        const T quad = T(0) - slope * x * x / (T(2) * knee);
        return std::min((thresh - inputDb) * slope, quad);
    }

    // ---- Layered characters ----

    /** @brief What the two layers of a layered character ask for at a level (dB). */
    struct Layers
    {
        T level;    ///< The levelling layer.
        T peak;     ///< The layer over it.
    };

    /**
     * @brief The curves of the two layers.
     *
     * The levelling layer takes its share of the slope from levelLeadDb
     * under the threshold, over a knee twice that wide; the other layer
     * takes the rest from as far over the threshold as leaves the sum, far
     * above, on the curve of the ratio itself.
     */
    [[nodiscard]] static Layers layeredCurve(T inputDb, T thresh, T ratio, T knee,
                                             const CharacterProfile& profile) noexcept
    {
        const T slope = T(1) - T(1) / ratio;
        const T levelSlope = profile.levelShare * slope;
        const T lag = profile.levelLeadDb * profile.levelShare / (T(1) - profile.levelShare);
        return { kneeCurve(inputDb, thresh - profile.levelLeadDb, levelSlope, T(2) * profile.levelLeadDb),
                 kneeCurve(inputDb, thresh + lag, slope - levelSlope, std::max(knee, profile.kneeFloorDb)) };
    }

    /**
     * @brief One step of a follower whose time grows with the distance to
     *        its target: far away it moves at a steady pace, close by like a
     *        one-pole.
     *
     * @param envelope     The follower (dB).
     * @param target       Where it is going (dB).
     * @param attackRate   Share of the distance per sample while the gain falls.
     * @param releaseRate  The same while it recovers.
     * @param attackPace   Decibels of distance that double the attack time.
     * @param releasePace  Decibels of distance that double the release time.
     */
    [[nodiscard]] static double pacedStep(double envelope, double target, double attackRate, double releaseRate,
                                          double attackPace, double releasePace) noexcept
    {
        const double distance = envelope - target;      // over zero: the gain has to fall
        const double gap = std::abs(distance);
        const double rate = (distance > 0.0) ? attackRate / (1.0 + gap / attackPace)
                                             : releaseRate / (1.0 + gap / releasePace);
        return target + fastDecay(rate) * distance;
    }

    /**
     * @brief One frame of a layered character: hold and range on the whole
     *        target, each layer towards its part of it, and their sum.
     *
     * The levelling layer never asks for more than the bounded whole; the
     * other layer takes what is left of it, so a held or a limited target
     * is shared out without a division.
     */
    [[nodiscard]] double layeredStep(T inputDb, T thresh, T ratio, T knee, const CharacterProfile& profile,
                                     T range, int holdN, int soft, T& held, int& counter,
                                     double& level, double& peak, SlidingDeepest* window = nullptr) const noexcept
    {
        const Layers want = layeredCurve(inputDb, thresh, ratio, knee, profile);
        const T bounded = holdAndRange(want.level + want.peak, range, holdN, held, counter, soft, window);
        const double levelTarget = static_cast<double>(std::max(want.level, bounded));
        const double peakTarget = static_cast<double>(bounded) - levelTarget;
        level = pacedStep(level, levelTarget, layerAttRate_, layerRelRate_,
                          static_cast<double>(profile.attackPaceDb), static_cast<double>(profile.releasePaceDb));
        peak = pacedStep(peak, peakTarget, peakAttRate_, peakRelRate_,
                         static_cast<double>(profile.attackPaceDb), static_cast<double>(profile.releasePaceDb));
        if (atRest(level, levelTarget)) level = 0.0;
        if (atRest(peak, peakTarget)) peak = 0.0;
        // (While the layers trade places their sum could pass the range.)
        return std::max(level + peak, -static_cast<double>(range));
    }

    /**
     * @brief Seats the two layers on the gain as it stands when a layered
     *        character engages. Audio thread only.
     */
    void syncLayerState(bool layered) noexcept
    {
        if (layered && !layeredActive_)
        {
            const double share = static_cast<double>(profileOf(character_.load(std::memory_order_relaxed)).levelShare);
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                const double shown = envFastDb_[static_cast<size_t>(ch)];
                layerLevelDb_[static_cast<size_t>(ch)] = share * shown;
                layerPeakDb_[static_cast<size_t>(ch)] = shown - share * shown;
            }
        }
        layeredActive_ = layered;
    }

    // ---- The gain after the ballistics ----

    /**
     * @brief The gain of a variable-mu stage applied by its own curve: two
     *        halves in push-pull, biased so that the slope of the pair at
     *        rest is the gain asked for.
     *
     * Each half follows a softplus (the smooth cutoff of a remote-cutoff
     * stage); the pair, half their difference, has the slope of a logistic
     * of the bias. The bias is the one whose slope is the gain (times the
     * slope the pair has at no reduction, kPushPullRest, taken back at the
     * output), so a small signal gets the gain exactly and a large one gets
     * the curve: compression where the pair is biased open, expansion once
     * it nears cutoff, and odd harmonics from both.
     *
     * @param input  Sample the gain applies to.
     * @param gain   Linear gain asked for (over 1 it is applied past the pair).
     * @param amount Level into the pair: at 1 full scale is kPushPullDrive
     *               times the scale of the curve.
     */
    [[nodiscard]] static T pushPullElement(T input, T gain, T amount) noexcept
    {
        if (!(gain > T(0)) || !(amount > T(0)) || !std::isfinite(input)) return input * gain;
        const double g = static_cast<double>(gain);
        const double slope = std::min(g, 1.0) * kPushPullRest;
        const double drive = kPushPullDrive * static_cast<double>(amount);
        // With e the exponential of the bias (the logistic of the bias is
        // the slope) and u that of the input, the difference of the two
        // halves is the logarithm of (1 + e u) / (1 + e / u): one
        // exponential and one logarithm per sample, exact for a small e too.
        const double e = slope / (1.0 - slope);
        const double u = std::exp(std::clamp(drive * static_cast<double>(input), -30.0, 30.0));
        const double pair = 0.5 * std::log1p(e * (u - 1.0 / u) / (1.0 + e / u));
        return static_cast<T>(pair * (g / slope) / drive);
    }

    // ---- The audio path beyond the plain gain ----

    /** @brief Whether a detector rectifies the key (the ones an asymmetry makes sense for). */
    [[nodiscard]] static bool rectifies(DetectorType detType) noexcept
    {
        return detType == DetectorType::Peak || detType == DetectorType::TruePeak || detType == DetectorType::Hybrid;
    }

    /**
     * @brief Takes the settings of the audio path that are not the plain gain
     *        (the tone that follows the reduction, the leak of the control,
     *        the asymmetry of the detector). Audio thread only.
     * @return Whether the block has to run frame by frame for any of them.
     */
    bool syncAudioPath() noexcept
    {
        const T low = toneLowDb_.load(std::memory_order_relaxed), high = toneHighDb_.load(std::memory_order_relaxed);
        const bool tone = low != T(0) || high != T(0);
        if (tone && (!toneActive_ || low != toneLowIn_ || high != toneHighIn_))
        {
            toneLow_.setCoeffsNow(BiquadCoeffs::makeLowShelfMatched(sampleRate_, kReductionToneLowHz,
                                                                   static_cast<double>(low)));
            toneHigh_.setCoeffsNow(BiquadCoeffs::makeHighShelfMatched(sampleRate_, kReductionToneHighHz,
                                                                     static_cast<double>(high)));
            if (!toneActive_)
            {
                toneLow_.reset();
                toneHigh_.reset();
            }
            toneLowIn_ = low;
            toneHighIn_ = high;
        }
        toneActive_ = tone;

        // The leak starts from the gain that is sounding: switching it on leaves no thump.
        const T leak = controlLeak_.load(std::memory_order_relaxed);
        if (leak > T(0) && !(leakAmount_ > T(0)))
            leakBase_.fill(decibelsToGain(gainReductionDb_.load(std::memory_order_relaxed)));
        leakAmount_ = leak;

        asymAmount_ = detectorAsymmetry_.load(std::memory_order_relaxed);
        audioExtras_ = toneActive_ || leakAmount_ > T(0);
        return audioExtras_ || asymAmount_ != T(0);
    }

    /**
     * @brief The gain on one sample of a channel when it is not a plain
     *        product: through the tone that follows the reduction, and with
     *        what leaks of the control.
     */
    [[nodiscard]] T gained(T input, T gain, int ch) noexcept
    {
        T wet = input * gain;
        if (toneActive_)
        {
            // y = x - (1 - g) F(x): at no reduction, the input itself.
            const T shaped = toneHigh_.processSample(
                toneLow_.processSample(std::isfinite(input) ? input : T(0), ch), ch);
            wet = input - (T(1) - gain) * shaped;
        }
        if (leakAmount_ > T(0))
        {
            // The gain through a high-pass: what it is over its own slow copy.
            T& base = leakBase_[static_cast<size_t>(ch)];
            const T moved = gain - base;
            if (std::abs(moved) < T(1e-7)) base = gain;     // (end the tail: a settled gain adds exactly nothing)
            else
            {
                base += leakCoeff_ * moved;
                wet += leakAmount_ * static_cast<T>(kLeakLevel) * moved;
            }
        }
        return wet;
    }

    /**
     * @brief The key with a share of its own crest added, scaled so that the
     *        half the share favours still reads the crest.
     */
    [[nodiscard]] T asymmetric(T key, int ch) noexcept
    {
        T& crest = asymCrest_[static_cast<size_t>(ch)];
        crest = std::max(std::abs(key), crest * asymDecay_);
        return (key + asymAmount_ * crest) / (T(1) + std::abs(asymAmount_));
    }

    /** @brief Whether anything stands between the ballistics and the gain: the landing, the network or the smoothing. */
    [[nodiscard]] bool shapesGain() const noexcept
    {
        return gainSmoothRate_ > 0.0 || landingRate_ > 0.0 || networkShare_ < 1.0;
    }

    /**
     * @brief One frame of what stands between the ballistics and the gain on
     *        a channel (dB): the landing of the release, the network of the
     *        control path and the two smoothing sections, those that are on.
     */
    [[nodiscard]] T smoothedGain(T grDb, int ch) noexcept
    {
        const size_t c = static_cast<size_t>(ch);
        double gain = static_cast<double>(grDb);
        if (landingRate_ > 0.0)
        {
            // At once while the gain change deepens or changes side; on the
            // way back, a follower that is slow near the gain it follows.
            double& heard = landingDb_[c];
            const double behind = std::abs(heard) - std::abs(gain);
            if (behind <= 0.0 || (heard > 0.0) != (gain > 0.0))
                heard = gain;
            else
                heard += std::min(1.0, landingRate_ * std::exp(std::min(behind * (1.0 / kLandingPaceDb), 40.0)))
                         * (gain - heard);
            if (gain == 0.0 && std::abs(heard) < 1e-6) heard = 0.0;
            gain = heard;
        }
        if (networkShare_ < 1.0)
        {
            // A shelf: what is slow, whole; of what is fast, a share.
            double& low = networkLow_[c];
            low += networkRate_ * (gain - low);
            if (gain == 0.0 && std::abs(low) < 1e-6) low = 0.0;
            gain = low + networkShare_ * (gain - low);
        }
        if (gainSmoothRate_ > 0.0)
        {
            double& first = gainSmoothA_[c];
            double& second = gainSmoothB_[c];
            first += gainSmoothRate_ * (gain - first);
            second += gainSmoothRate_ * (first - second);
            // (End the asymptotic tail: 0 dB is exact again once it is reached.)
            if (gain == 0.0 && std::abs(first) < 1e-6 && std::abs(second) < 1e-6)
            {
                first = 0.0;
                second = 0.0;
            }
            gain = second;
        }
        return static_cast<T>(gain);
    }

    /**
     * @brief Seats each of those stages on the gain as it stands when it
     *        engages. Audio thread only.
     */
    void syncGainSmoothing(bool) noexcept
    {
        const bool smooth = gainSmoothRate_ > 0.0, landing = landingRate_ > 0.0, network = networkShare_ < 1.0;
        if ((smooth && !gainSmoothActive_) || (landing && !landingActive_) || (network && !networkActive_))
        {
            const double shown = static_cast<double>(gainReductionDb_.load(std::memory_order_relaxed));
            if (smooth && !gainSmoothActive_)
            {
                gainSmoothA_.fill(shown);
                gainSmoothB_.fill(shown);
            }
            if (landing && !landingActive_) landingDb_.fill(shown);
            if (network && !networkActive_) networkLow_.fill(shown);
        }
        gainSmoothActive_ = smooth;
        landingActive_ = landing;
        networkActive_ = network;
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
        if (profileOf(charType).ratioGrowthDb > T(0))
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
        const LoopCoeffs coeffs { charAttCoeff_, charRelCoeff_, charChargeCoeff_, charSlowRelCoeff_,
                                  charMidChargeCoeff_, charMidRelCoeff_ };
        return boundImplicit(stepImplicit(inDb, ch, charType, thresh, ratio, knee, coeffs), ch);
    }

    /** @brief Coefficients of one step of the semi-implicit loop. */
    struct LoopCoeffs
    {
        double attack;       ///< Fast stage while the reduction deepens.
        double release;      ///< Fast stage while it recovers.
        double charge;       ///< Memory stage while it charges.
        double slowRelease;  ///< Memory stage while it discharges.
        double midCharge;    ///< Medium memory of the three-stage cell while it charges.
        double midRelease;   ///< Medium memory while it discharges.
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
            if (threeStage_)
            {
                double& mid = envMidDb_[ch];
                double& deep = envDeepDb_[ch];
                mid = st + ((st < mid) ? coeffs.midCharge : coeffs.midRelease) * (mid - st);
                if (atRest(mid, st)) mid = 0.0;
                deep = st + ((st < deep) ? coeffs.charge : coeffs.slowRelease) * (deep - st);
                if (atRest(deep, st)) deep = 0.0;
                slow = kOptoMidShare * mid + (1.0 - kOptoMidShare) * deep;
            }
            else
            {
                const bool charging = st < slow;
                slow = st + (charging ? coeffs.charge : coeffs.slowRelease) * (slow - st);
                if (atRest(slow, st)) slow = 0.0;
            }
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

        return static_cast<T>(ballisticsStep(static_cast<double>(targetGrDb), envFastDb_[ch], envSlowDb_[ch],
                                             envMidDb_[ch], envDeepDb_[ch], attCoeff, relCoeff));
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
    [[nodiscard]] double ballisticsStep(double tgt, double& fast, double& slow, double& mid, double& deep,
                                        double attCoeff, double relCoeff) const noexcept
    {
        return ballisticsCore(tgt, fast, slow, mid, deep, attCoeff, relCoeff, charChargeCoeff_, charSlowRelCoeff_,
                              charMidChargeCoeff_, charMidRelCoeff_);
    }

    /** @brief ballisticsStep() with the coefficients of the memory stage given too. */
    [[nodiscard]] double ballisticsCore(double tgt, double& fast, double& slow, double& mid, double& deep,
                                        double attCoeff, double relCoeff, double chargeCoeff,
                                        double slowRelCoeff, double midChargeCoeff,
                                        double midRelCoeff) const noexcept
    {
        fast = branchingStep(fast, tgt, attCoeff, relCoeff);
        if (atRest(fast, tgt)) fast = 0.0; // end the asymptotic dB tail

        if (charFastWeight_ >= T(1)) // Clean / Varimu: single envelope
            return fast;

        // Opto / FET memory stage: charges toward the target over the history
        // time constant and discharges with the slow release, so its level
        // encodes how long (and how deep) the compressor has been working.
        // The three-stage cell blends a medium and a deep memory there.
        if (threeStage_)
        {
            mid = branchingStep(mid, tgt, midChargeCoeff, midRelCoeff);
            if (atRest(mid, tgt)) mid = 0.0;
            deep = branchingStep(deep, tgt, chargeCoeff, slowRelCoeff);
            if (atRest(deep, tgt)) deep = 0.0;
            slow = kOptoMidShare * mid + (1.0 - kOptoMidShare) * deep;
        }
        else
        {
            slow = branchingStep(slow, tgt, chargeCoeff, slowRelCoeff);
            if (atRest(slow, tgt)) slow = 0.0;
        }

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
        return (detType == DetectorType::Rms || detType == DetectorType::Hybrid)
             ? static_cast<double>(rmsWindowSamples_) : 0.0;
    }

    /** @brief The Clean character in feed-forward runs the decoupled detector
     *  when setCleanBallistics() selected it. */
    [[nodiscard]] bool useDecoupled(Character charType, Topology topoEff) const noexcept
    {
        return profileOf(charType).decoupled && topoEff == Topology::FeedForward
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
    [[nodiscard]] T heldBack(T grDb, int g, double delay, bool run) noexcept
    {
        if (!run) return grDb;
        auto& line = gainDelay_[static_cast<size_t>(g)];
        line.push(grDb);
        const int whole = static_cast<int>(delay);
        const T part = static_cast<T>(delay - static_cast<double>(whole));
        const T at = line.read(whole);
        return part > T(0) ? at + part * (line.read(whole + 1) - at) : at;
    }

    /**
     * @brief The delay of the gain line this frame. It slides to its target
     *        kGainDelaySlide of a sample per sample, so a lookahead that
     *        moves inside the latency kept for it leaves no step in the gain
     *        (the first frame after prepare() or reset() starts at the target).
     */
    [[nodiscard]] double slideGainDelay(int target) noexcept
    {
        const double goal = static_cast<double>(target);
        if (gainDelayNow_ < 0.0) gainDelayNow_ = goal;
        else if (gainDelayNow_ != goal)
            gainDelayNow_ += std::clamp(goal - gainDelayNow_, -kGainDelaySlide, kGainDelaySlide);
        return gainDelayNow_;
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
            const LoopCoeffs coeffs { subAttCoeff_, subRelCoeff_, subChargeCoeff_, subSlowRelCoeff_,
                                      subMidChargeCoeff_, subMidRelCoeff_ };
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
        if (cfg.modeType == Mode::Upward || cfg.modeType == Mode::Both)
        {
            T& guard = upwardGuardDb_[c];
            guard = std::max(guardDb, guard - upwardGuardDecay_);
            guardDb = guard;
        }

        const T range = rangeDb_.load(std::memory_order_relaxed);
        const int soft = rangeShape_.load(std::memory_order_relaxed);
        const int holdLength = holdSamples_ * count; // the hold counts levels here
        double last = precTarget_[c];
        double& hold = envHoldDb_[c];
        double& fast = envFastDb_[c];
        double& slow = envSlowDb_[c];
        double& mid = envMidDb_[c];
        double& deep = envDeepDb_[c];
        double targets[kDetectorPoints] = {};
        double deepest = 0.0;
        for (int j = 0; j < count; ++j)
        {
            targets[j] = static_cast<double>(holdAndRange(
                computeGain(levelDb[j], thresh, ratio, knee, cfg.charType, cfg.modeType, guardDb),
                range, holdLength, heldGrDb_[c], holdCounters_[c], soft, holdWindow(static_cast<int>(c)), count));
            deepest = (j == 0) ? targets[j] : std::min(deepest, targets[j]);
        }
        // Programme-dependent timing sets the two coefficients once per
        // frame, for steps of an eighth of a sample.
        double attack = subAttCoeff_, release = subRelCoeff_;
        if (cfg.program)
        {
            const ProgramCoeffs timed = programCoeffs(deepest, fast, programState_[c], balance_, 1.0 / kSubsteps);
            attack = timed.attack;
            release = timed.release;
        }
        for (int j = 0; j < count; ++j)
        {
            const double to = targets[j];
            for (int s = 1; s <= stepsEach; ++s)
            {
                const double at = last + (to - last) * (s * share);
                sum += cfg.decoupled
                    ? decoupledCore(at, hold, fast, subDecHoldCoeff_, subAttCoeff_, subDecSmoothCoeff_)
                    : ballisticsCore(at, fast, slow, mid, deep, attack, release, subChargeCoeff_, subSlowRelCoeff_,
                                     subMidChargeCoeff_, subMidRelCoeff_);
            }
            last = to;
        }
        precTarget_[c] = last;
        const double mean = sum * (1.0 / kSubsteps);
        return static_cast<T>(cfg.program ? programShape(mean, programState_[c]) : mean);
    }

    // ---- Programme-dependent timing (setProgramRelease and the three after it) ----

    /** @brief The weights of a block (zero where the routing leaves the times fixed). */
    struct ProgramTiming
    {
        double release = 0.0;   ///< Weight of the sustained reduction on the release.
        double attack = 0.0;    ///< Weight of the closeness of the target on the attack.
        double spectral = 0.0;  ///< Weight of the balance of the key on both.
        double emphasis = 0.0;  ///< Emphasis of the attack, -1 to +1.
        bool active = false;    ///< Any of them acts.
    };

    /** @brief State of one gain computer under programme-dependent timing. */
    struct ProgramState
    {
        double sustained = 0.0; ///< Slow follower of the depth of the gain change (dB).
        double smoothed = 0.0;  ///< The gain smoothed over kEmphasisAttacks times the attack (dB).
    };

    /** @brief State of the two bands the balance of one channel's key is read from. */
    struct BalanceBands
    {
        double sub = 0.0;  ///< Under kBalanceFloorHz.
        double low = 0.0;  ///< From there to kBalanceLowHz: the low band.
        double mid = 0.0;  ///< Under kBalanceHighHz.
        double high = 0.0; ///< From there to kBalanceCeilingHz: the high band.
    };

    /** @brief The two coefficients of a frame. */
    struct ProgramCoeffs
    {
        double attack, release;
    };

    /** @brief The weights for this routing: the Clean and Varimu characters
     *  in feed-forward with Branching ballistics. */
    [[nodiscard]] ProgramTiming programTiming(Character charType, Topology topoEff, bool decoupled,
                                              bool splitAdaptive) const noexcept
    {
        ProgramTiming p;
        if (topoEff != Topology::FeedForward || decoupled || splitAdaptive
            || !profileOf(charType).programTiming)
            return p;
        p.release = static_cast<double>(programRelease_.load(std::memory_order_relaxed));
        p.attack = static_cast<double>(adaptiveAttack_.load(std::memory_order_relaxed));
        p.spectral = static_cast<double>(spectralTiming_.load(std::memory_order_relaxed));
        p.emphasis = static_cast<double>(attackEmphasis_.load(std::memory_order_relaxed));
        p.active = p.release > 0.0 || p.attack > 0.0 || p.spectral > 0.0 || p.emphasis != 0.0;
        return p;
    }

    /** @brief Seats the measures on the gain that is sounding when the
     *  programme-dependent timing engages. Audio thread only. */
    void syncProgramState(bool active) noexcept
    {
        if (active && !programActive_)
            for (size_t ch = 0; ch < static_cast<size_t>(kMaxChannels); ++ch)
                programState_[ch] = { std::abs(envFastDb_[ch]), envFastDb_[ch] };
        programActive_ = active;
    }

    /**
     * @brief Feeds one channel's key to the two bands of the balance and adds
     *        their squares to the frame's.
     *
     * One-pole splits: what a low-pass at kBalanceFloorHz keeps is taken out,
     * and the rest goes through a low-pass at kBalanceLowHz (the low band);
     * what a low-pass at kBalanceHighHz leaves out goes through one at
     * kBalanceCeilingHz (the high band). The bands are equally wide on a
     * logarithmic axis, so a key with the same power in every octave gives
     * the two the same.
     */
    void balanceBands(T key, int ch, double& lowSq, double& highSq) noexcept
    {
        BalanceBands& b = balanceBands_[static_cast<size_t>(ch)];
        const double x = static_cast<double>(key);
        b.sub += subRate_ * (x - b.sub);
        b.low += lowRate_ * ((x - b.sub) - b.low);
        b.mid += midRate_ * (x - b.mid);
        b.high += highRate_ * ((x - b.mid) - b.high);
        lowSq += b.low * b.low;
        highSq += b.high * b.high;
    }

    /** @brief Smooths the frame's two powers and returns the balance of the
     *  key: -1 all low, +1 all high, 0 the same in both (or silence). */
    [[nodiscard]] double balanceAdvance(double lowSq, double highSq) noexcept
    {
        balanceLow_ += balanceRate_ * (lowSq - balanceLow_);
        balanceHigh_ += balanceRate_ * (highSq - balanceHigh_);
        const double sum = balanceLow_ + balanceHigh_;
        balance_ = (sum > kBalanceSilence) ? (balanceHigh_ - balanceLow_) / sum : 0.0;
        return balance_;
    }

    /**
     * @brief The attack and release coefficients of a frame under
     *        programme-dependent timing, and the advance of its measures.
     *
     * @param target  Gain target of the frame (dB).
     * @param fast    The gain now (dB).
     * @param state   Measures of this gain computer.
     * @param balance Spectral balance of the key.
     * @param share   Share of a sample each step of the follower covers.
     */
    [[nodiscard]] ProgramCoeffs programCoeffs(double target, double fast, ProgramState& state,
                                              double balance, double share) const noexcept
    {
        // How sustained the gain change is: a slow follower of its depth
        // against the depth itself.
        const double depth = std::abs(fast);
        state.sustained += sustainRate_ * (depth - state.sustained);
        // (Squared: a peak that came and went must count for next to nothing.)
        const double held = std::min(state.sustained / (depth + kSustainFloorDb), 1.0);
        const double sustain = held * held;
        // How close the target is: full at none, nothing from kFullAttackDb.
        const double excess = std::max(fast - target, 0.0);
        const double close = 1.0 - std::min(excess * (1.0 / kFullAttackDb), 1.0);
        // A bright key shortens both times and a dark one lengthens them.
        const double tilt = program_.spectral * balance;
        const double speed = share * (2.0 + tilt) / (2.0 - tilt);
        return { fastDecay(attackX_ * speed / (1.0 + kAttackStretch * program_.attack * close)),
                 fastDecay(releaseX_ * speed / (1.0 + kReleaseStretch * program_.release * sustain)) };
    }

    /** @brief The gain with the emphasis of the attack: what it is under
     *  its own smoothed course while it falls, added again (or taken off). */
    [[nodiscard]] double programShape(double gain, ProgramState& state) const noexcept
    {
        state.smoothed += emphasisRate_ * (gain - state.smoothed);
        return gain + program_.emphasis * std::min(gain - state.smoothed, 0.0);
    }

    /** @brief One step of the branching follower with programme-dependent
     *  timing (the single envelope of Clean and Varimu). */
    [[nodiscard]] double programStep(double target, double& fast, ProgramState& state,
                                     double balance) const noexcept
    {
        const ProgramCoeffs timed = programCoeffs(target, fast, state, balance, 1.0);
        fast = branchingStep(fast, target, timed.attack, timed.release);
        if (atRest(fast, target)) fast = 0.0;
        return programShape(fast, state);
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
        double midCharge = 0.0;     ///< Charge coefficient of the medium memory (three-stage cell).
        double midRelease = 0.0;    ///< Release coefficient of the medium memory.
        bool threeStage = false;    ///< The memory stage is the three-stage cell.
        double hold = 0.0;          ///< Release coefficient of the decoupled hold stage.
        Character charType = Character::Clean;       ///< Character.
        Mode modeType = Mode::Downward;              ///< Direction.
        DetectorType detType = DetectorType::Peak;   ///< Effective detector.
        bool decoupled = false;     ///< Decoupled ballistics.
        bool loop = false;          ///< The detector reads the output (feedback).
        bool autoThreshold = false; ///< The threshold counts from the key's peak.
        int soft = 0;               ///< How the target meets the range (a RangeShape).
        T liftBelow = T(0), liftRatio = T(1), liftRange = T(0);   ///< The upward half of Mode::Both.
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
        key.midCharge = charMidChargeCoeff_;
        key.midRelease = charMidRelCoeff_;
        key.threeStage = threeStage_;
        key.hold = decHoldCoeff_;
        key.charType = charType;
        key.modeType = modeType;
        key.detType = detType;
        key.decoupled = decoupled;
        key.loop = loop;
        key.autoThreshold = autoThreshold_.load(std::memory_order_relaxed);
        key.soft = rangeShape_.load(std::memory_order_relaxed);
        key.liftBelow = liftBelowDb_;
        key.liftRatio = liftRatioIn_;
        key.liftRange = liftRangeDb_;
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
        const CharacterProfile& profile = profileOf(charType);
        const Topology topoEff = topologyFor(profile, topology_.load(std::memory_order_relaxed));
        const DetectorType detType = detectorFor(profile, detectorType_.load(std::memory_order_relaxed));
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
     * between samples are not independent); for Hybrid those magnitudes held
     * over the root of twice the mean square; for Hilbert its envelope (a
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
        const bool hybrid = key.detType == DetectorType::Hybrid;
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
                // Hybrid never reads under the root of twice the mean square.
                levelDb += 20.0 * std::log10(hybrid ? std::max(t, std::numbers::sqrt2) : t)
                         + (envelope ? kEnvelopeLiftDb : 0.0);
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
                const double deep = restingGain(target, weight, count, 1.0 - key.charge, 1.0 - key.slowRelease);
                const double rest = key.threeStage
                    ? kOptoMidShare * restingGain(target, weight, count, 1.0 - key.midCharge, 1.0 - key.midRelease)
                      + (1.0 - kOptoMidShare) * deep
                    : deep;
                settled = (1.0 - memory) * settled + memory * rest;
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

            // The Hybrid detector's mean of the squares takes the same time.
            hybridRate_ = 1.0 - std::exp(-1.0 / (sampleRate_ * std::max(static_cast<double>(rmsWindowMs_), 1.0)
                                                 / 1000.0));

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
    std::atomic<LinkMode> linkMode_ { LinkMode::Loudest }; ///< What the linked level reads.
    double linkCoeff_ = 0.0;                ///< One-pole coefficient of the key's covariance (20 ms).
    double linkLL_ = 0.0, linkRR_ = 0.0, linkLR_ = 0.0; ///< The key's covariance (dominant link).
    std::atomic<T> mix_ { T(1) };           ///< Wet/Dry mix (1 = full wet).
    std::atomic<T> lookaheadMs_ { T(0) };    ///< Lookahead latency target in ms.
    std::atomic<T> characterColor_ { T(0) }; ///< Character harmonic amount (0 to 1).
    std::atomic<AutoMakeupMode> autoMakeupMode_ { AutoMakeupMode::Off }; ///< Auto-makeup behavior.

    std::atomic<DetectorType> detectorType_ { DetectorType::Peak }; ///< Selected detection method.
    std::atomic<Topology> topology_ { Topology::FeedForward };      ///< Selected topology.
    std::atomic<Character> character_ { Character::Clean };         ///< Selected ballistics.
    std::atomic<OptoCell> optoCell_ { OptoCell::TwoStage };         ///< Cell of the Opto character.
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
    double charMidRelCoeff_ = 0.0;  ///< Release coefficient of the medium memory (three-stage cell).
    double charMidChargeCoeff_ = 0.0; ///< Charge coefficient of the medium memory.
    bool threeStage_ = false;       ///< The audio thread runs the three-stage cell.
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
    std::array<double, kMaxChannels> envMidDb_ {};  ///< Medium memory of the three-stage cell per channel (dB).
    std::array<double, kMaxChannels> envDeepDb_ {}; ///< Deep memory of the three-stage cell per channel (dB).
    std::array<double, kMaxChannels> envHoldDb_ {}; ///< Decoupled hold stage per channel (dB).
    double decHoldCoeff_   = 0.0;  ///< Decoupled hold-stage release coefficient.
    double decSmoothCoeff_ = 0.0;  ///< Decoupled smoothing pole, release side.
    bool decoupledActive_ = false; ///< The decoupled path ran last (audio thread).

    // Layered characters: the two layers and their rates (share of the
    // distance to the target per sample).
    std::array<double, kMaxChannels> layerLevelDb_ {};
    std::array<double, kMaxChannels> layerPeakDb_ {};
    double layerAttRate_ = 0.0, layerRelRate_ = 0.0, peakAttRate_ = 0.0, peakRelRate_ = 0.0;
    bool layeredActive_ = false;    ///< The layers ran last (audio thread).

    // The push-pull element of Varimu: the slope of the pair at no reduction
    // and the level of full scale in units of its curve at amount 1.
    static constexpr double kPushPullRest = 0.98;
    static constexpr double kPushPullDrive = 2.0;

    // The gain after the ballistics.
    std::atomic<T> gainSmoothMs_ { T(0) };          ///< Smoothing of the gain (0: off).
    double gainSmoothRate_ = 0.0;                   ///< Coefficient of each of its two sections (0: off).
    std::array<double, kMaxChannels> gainSmoothA_ {};
    std::array<double, kMaxChannels> gainSmoothB_ {};
    bool gainSmoothActive_ = false;                 ///< The smoothing ran last (audio thread).
    static constexpr double kLandingPaceDb = 0.5;   ///< Decibels behind that make the landing e times faster.
    std::atomic<T> landingMs_ { T(0) };             ///< Landing of the release (0: off).
    double landingRate_ = 0.0;                      ///< Its coefficient when it is on the gain it follows (0: off).
    std::array<double, kMaxChannels> landingDb_ {};
    bool landingActive_ = false;
    std::atomic<T> networkDepthDb_ { T(0) };        ///< Depth of the network of the control path (0: off).
    std::atomic<T> networkCornerHz_ { T(10) };      ///< Its corner.
    double networkShare_ = 1.0;                     ///< Share of a fast move applied at once (1: off).
    double networkRate_ = 0.0;                      ///< Coefficient of its low-pass.
    std::array<double, kMaxChannels> networkLow_ {};
    bool networkActive_ = false;

    // The audio path beyond the plain gain.
    static constexpr double kLeakCornerHz = 5.0;    ///< High-pass the gain leaks through.
    static constexpr double kLeakLevel = 0.05;      ///< What a gain that moves by 1 leaks, at an amount of 1.
    static constexpr double kAsymCrestSeconds = 0.05;   ///< Fall of the crest the asymmetric detector adds.
    std::atomic<T> toneLowDb_ { T(0) }, toneHighDb_ { T(0) };   ///< setReductionTone().
    std::atomic<T> controlLeak_ { T(0) };           ///< setControlLeak().
    std::atomic<T> detectorAsymmetry_ { T(0) };     ///< setDetectorAsymmetry().
    Biquad<T, kMaxChannels> toneLow_ {}, toneHigh_ {};   ///< The filter F of the tone (audio thread).
    T toneLowIn_ = T(0), toneHighIn_ = T(0);        ///< The gains those were designed for.
    bool toneActive_ = false;                       ///< The tone is in the path.
    T leakAmount_ = T(0);                           ///< The leak as the audio thread took it.
    T leakCoeff_ = T(0);
    std::array<T, kMaxChannels> leakBase_ {};       ///< The slow copy of each channel's gain.
    T asymAmount_ = T(0);                           ///< The asymmetry as the audio thread took it.
    T asymDecay_ = T(0);
    std::array<T, kMaxChannels> asymCrest_ {};      ///< The crest of each channel's key.
    bool audioExtras_ = false;                      ///< The gain is not a plain product (tone or leak).

    // The routes of a pair (setRoute()).
    std::atomic<Route> route_ { Route::Stereo };
    Route routeIn_ = Route::Stereo;                 ///< The route the states carry (audio thread).
    bool routeWas_ = false;                         ///< The last block ran a route.
    int routeBlock_ = 0, routeReach_ = 0;
    AudioBuffer<T> routeKey_;                       ///< A key given apart, as its middle and its sides.
    RingBuffer<T> routeDelay_;                      ///< The part that is not treated waits here.

    // The upward half of Mode::Both (setLift()), and as the audio thread took it.
    std::atomic<T> liftBelow_ { T(30) }, liftRatio_ { T(2) }, liftRange_ { T(6) };
    T liftBelowDb_ = T(30), liftRatioIn_ = T(2), liftRangeDb_ = T(6);
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
    KeyEq keyEq_;                              ///< Equalizer of the key, after the high-pass.
    KeyMonitor<T, 16> keyMonitor_;             ///< The listened key: late by the latency, under full scale.
    static constexpr int kStageKeyRoom = 64;   ///< Samples kept for the colour stage's latency in the listened key.
    std::atomic<bool> scListen_ { false };     ///< The key goes out instead of the audio.
    std::atomic<bool> removalListen_ { false };        ///< The output is what the compression takes out.

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
    std::atomic<HoldMode> holdMode_ { HoldMode::Deepest };
    std::unique_ptr<SlidingDeepest[]> holdWindows_;  ///< One window per gain computer (HoldMode::EveryPeak).
    int holdWindowFor_ = -1;                         ///< The hold length the windows are set for.
    bool holdEveryPeak_ = false;                     ///< The hold mode as the audio thread took it.

    // Automatic threshold, lookahead inside a fixed latency, amount, soft range.
    static constexpr double kAutoRefFallDb = 1.0;   ///< dB per second the reference of the automatic threshold falls.
    std::atomic<bool> autoThreshold_ { false };     ///< The threshold counts from the key's own recent peak.
    std::atomic<T> thresholdInUseDb_ { T(0) };      ///< With the automatic threshold, the threshold of the last sample (metering).
    std::array<double, kMaxChannels> autoRefDb_ {}; ///< That reference per channel (the linked one in slot 0), in dB.
    double autoRefFall_ = 0.0;                      ///< Its fall per sample, in dB.
    std::atomic<T> maxLookaheadMs_ { T(0) };        ///< Latency kept for the lookahead (0: the lookahead itself).
    int latencySamples_ = 0;                        ///< Delay of the audio in feed-forward (audio thread).
    std::array<RingBuffer<T>, kMaxChannels> gainDelay_ {}; ///< Holds the gain back while the lookahead is shorter than the latency.
    static constexpr double kGainDelaySlide = 0.25;  ///< Samples per sample the gain line's delay moves to a new lookahead.
    double gainDelayNow_ = -1.0;                     ///< The gain line's delay now, in samples (negative: not started).
    bool latencyKept_ = false;                       ///< A latency is kept for the lookahead (audio thread).
    std::atomic<LookaheadShape> lookaheadShape_ { LookaheadShape::Lead };
    std::atomic<bool> feedbackKeyed_ { false };       ///< A feedback loop runs on the key ahead of the audio.
    std::vector<LookaheadRamp> lookaheadRamps_;      ///< One ramp per gain computer of the layout (LookaheadShape::Ramp).
    int rampCount_ = 0;                              ///< Ramps prepared.
    bool rampShape_ = false;                         ///< The shape as the audio thread took it.
    bool rampActive_ = false;                        ///< The ramp ran in the last block.
    std::atomic<T> amount_ { T(1) };                ///< Share of the gain change that is applied, in dB.
    SmoothedValue<T> amountSmooth_;                 ///< De-zippered amount.
    std::atomic<int> rangeShape_ { 0 };             ///< How the gain change meets the range (a RangeShape).
    static constexpr double kLocalRangeStretchDb = 6.0; ///< Width of the bend of RangeShape::Local, at most.

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

    // The colour stage (setCharacterStage()).
    std::atomic<StagePosition> stagePosition_ { StagePosition::Off };
    CharacterStage<T> stageBefore_;                     ///< The input amplifier.
    CharacterStage<T> stageAfter_;                      ///< The output amplifier.
    AudioBuffer<T> stageScratch_;                       ///< The coloured copy while a place fades.
    std::vector<double> stageFade_;                     ///< The share of the colour, per sample of the fade.
    double stageFadeStep_ = 0.0;                        ///< Its change per sample (20 ms end to end).
    double stageWeightBefore_ = 0.0;                    ///< Share of the input stage (audio thread).
    double stageWeightAfter_ = 0.0;                     ///< Share of the output stage (audio thread).
    T interpPhase_[kDetectorPoints - 1][kInterpTaps] = {};     ///< Interpolation phases (prepare()).
    std::array<InterpHistory, kMaxChannels> interp_ {}; ///< Their history per channel.
    std::array<double, kMaxChannels> precLevel_ {};     ///< Last level of the solved loop (dB).
    std::array<double, kMaxChannels> precTarget_ {};    ///< Last gain target, after hold and range (dB).
    double rawAttCoeff_ = 0.0;       ///< Attack coefficient as it is observed (before a loop's own speed).
    double subAttCoeff_ = 0.0;       ///< Attack coefficient of an eighth of a sample.
    double subRelCoeff_ = 0.0;       ///< Release coefficient of an eighth of a sample.
    double subSlowRelCoeff_ = 0.0;   ///< Memory-stage release coefficient of an eighth of a sample.
    double subChargeCoeff_ = 0.0;    ///< Memory-stage charge coefficient of an eighth of a sample.
    double subMidRelCoeff_ = 0.0;    ///< Medium-memory release coefficient of an eighth of a sample.
    double subMidChargeCoeff_ = 0.0; ///< Medium-memory charge coefficient of an eighth of a sample.
    double subDecHoldCoeff_ = 0.0;   ///< Decoupled hold coefficient of an eighth of a sample.
    double subDecSmoothCoeff_ = 0.0; ///< Decoupled smoothing coefficient of an eighth of a sample.

    // Programme-dependent timing (setProgramRelease() and the three after it).
    static constexpr double kSustainReleases = 15.0;    ///< Follower of the depth of the gain change, in release times.
    static constexpr double kSustainFloorSeconds = 0.1; ///< It is never shorter than this,
    static constexpr double kSustainCeilSeconds = 60.0; ///< nor longer than this.
    static constexpr double kSustainFloorDb = 0.5;      ///< Depth under which a reduction does not count as sustained.
    static constexpr double kReleaseStretch = 7.0;      ///< The release grows up to 1 + this.
    static constexpr double kAttackStretch = 3.0;       ///< The attack grows up to 1 + this.
    static constexpr double kFullAttackDb = 6.0;        ///< Distance to the target from which the attack is the knob's.
    static constexpr double kEmphasisAttacks = 6.0;     ///< Smoothing the emphasis of the attack measures against, in attack times.
    static constexpr double kEmphasisFloorSeconds = 0.002;  ///< It is never shorter than this,
    static constexpr double kEmphasisCeilSeconds = 0.5;     ///< nor longer than this.
    static constexpr double kBalanceFloorHz = 30.0;     ///< The low band of the balance starts here.
    static constexpr double kBalanceLowHz = 200.0;      ///< And ends here.
    static constexpr double kBalanceHighHz = 2500.0;    ///< The high band starts here.
    static constexpr double kBalanceCeilingHz = 16000.0;///< And ends here.
    static constexpr double kBalanceSeconds = 0.05;     ///< Smoothing of the two powers.
    static constexpr double kBalanceSilence = 1e-24;    ///< Summed power under which the balance is 0.
    std::atomic<T> programRelease_ { T(0) };            ///< Weight of the sustained reduction on the release.
    std::atomic<T> adaptiveAttack_ { T(0) };            ///< Weight of the closeness of the target on the attack.
    std::atomic<T> spectralTiming_ { T(0) };            ///< Weight of the balance of the key on both times.
    std::atomic<T> attackEmphasis_ { T(0) };            ///< Emphasis of the attack.
    ProgramTiming program_ {};                          ///< The weights of the block (audio thread).
    bool programActive_ = false;                        ///< They acted last (audio thread).
    std::array<ProgramState, kMaxChannels> programState_ {};  ///< Measures per gain computer.
    std::array<BalanceBands, kMaxChannels> balanceBands_ {};  ///< Bands of the key per channel.
    double balanceLow_ = 0.0;        ///< Smoothed power of the low band.
    double balanceHigh_ = 0.0;       ///< Smoothed power of the high band.
    double balance_ = 0.0;           ///< Their balance, -1 to +1.
    double attackX_ = 0.0;           ///< The attack as a decay per sample (-ln of its coefficient).
    double releaseX_ = 0.0;          ///< The release as a decay per sample.
    double subRate_ = 0.0;           ///< Pole of the split at kBalanceFloorHz.
    double lowRate_ = 0.0;           ///< Pole of the split at kBalanceLowHz.
    double midRate_ = 0.0;           ///< Pole of the split at kBalanceHighHz.
    double highRate_ = 0.0;          ///< Pole of the split at kBalanceCeilingHz.
    double balanceRate_ = 0.0;       ///< Pole of the smoothing of the two powers.
    double sustainRate_ = 0.0;       ///< Pole of the follower of the depth.
    double emphasisRate_ = 0.0;      ///< Pole of the smoothing the emphasis measures against.

    // Hybrid detector: a one-pole mean of the squares per channel.
    std::array<double, kMaxChannels> hybridMeanSq_ {};  ///< Mean square of the key.
    double hybridRate_ = 0.0;                           ///< Its pole (the time of the RMS window).

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
