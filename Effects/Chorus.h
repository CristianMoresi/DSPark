// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Chorus.h
 * @brief High-performance Chorus / Flanger effect with LFO-modulated delay lines.
 *
 * Creates the classic chorus effect by mixing the dry signal with delayed
 * copies whose delay time is modulated by LFOs. Multiple voices with
 * phase-offset LFOs create a rich, wide stereo field.
 *
 * Engineered for real-time performance:
 * - No transcendental functions (sin, cos) in the audio hot path (the LFOs
 *   and the feedback saturation use polynomial approximations). The inner
 *   loop itself is a serial recursion (feedback plus parameter smoothing),
 *   so it does not vectorize; the per-sample work is branch-light instead.
 * - Sample-rate-invariant one-pole parameter smoothing for zipper-free
 *   automation.
 * - Tape-style feedback saturation prevents digital clipping.
 * - Stereo spread is the LFO phase offset between the first and last prepared
 *   channel: 1.0 puts a stereo pair 180 degrees apart (default 0.5: 90).
 * - Spread and voice-count changes are click-free: each LFO glides to its new
 *   phase by a bounded frequency offset (no phase jump), and voices fade in and
 *   out over 50 ms with the level normalisation following the fade.
 * - Delay reads use 4-point Hermite interpolation by default, or optionally the
 *   framework's 32-tap Kaiser sinc reader, or 2-point linear reads whose
 *   loss of highs belongs to the sound of many chorus units
 *   (setInterpolation()).
 * - setModel(Model::Analog) (off by default) runs a bucket-brigade chorus:
 *   the channels summed to mono, high-passed at 5 Hz, into two bucket-brigade
 *   lines of 512 stages (Core/BucketBrigade.h) whose clocks a triangle wave
 *   moves in opposition (the right tap is the left one inverted, so the two
 *   delays always add up to twice the centre; in phase in the Vibrato mode;
 *   setAnalogTapPhase() sets any offset between, gliding), each with the
 *   device's anti-alias and reconstruction filters (fourth-order
 *   Butterworth at 7 kHz, one input low-pass shared by the two lines), and
 *   the line's hiss, high-passed at 200 Hz, which grows with the delay (a
 *   slower clock). The corners are fixed in hertz and the same in the three
 *   modes, lower only when a setting makes the clock too slow for them (a
 *   fifth of the slowest clock the sweep needs); setAnalogTone() moves the
 *   output's; setAnalogCircuitFilters() takes the pair of a chorus circuit
 *   instead (BucketBrigade::Shape::Chorus: two slightly resonant sections
 *   at 8 kHz before the line and a damped one at 5.6 kHz after it, a darker
 *   wet that lets more of the clock's images through). The dry
 *   signal keeps its stereo; there is no feedback and the voices, feedback
 *   and interpolation settings do not apply. setAnalogMode() loads three
 *   presets of rate, depth and centre; the centre, the depth and the taps'
 *   offset glide (50 ms) instead of stepping at the block rate, and the
 *   filters' corners move at most an octave in a quarter of a second. The
 *   lines run with their high precision (BucketBrigade::Precision::High) and,
 *   below 88.2 kHz, at twice the rate, so the images of their clock are
 *   filtered as in the device instead of folding into the band; from
 *   176.4 kHz the analog engine runs at half the rate.
 *
 * A change of engine (setModel()) does not cut: the incoming engine runs
 * from the change, unheard, until its line is full, and then the two cross
 * over 10 ms. setMixRule() offers the mixer's rules, among them the one of
 * the hardware (DryWetMixer::MixRule::FullDryToHalf: at half, dry plus wet).
 *
 * Threading: prepare() belongs to the setup thread (allocates; never call it
 * concurrently with processing). processBlock() and reset() belong to the
 * audio thread. All setters are lock-free atomic publications, safe from any
 * thread; LFO reconfiguration (voices, waveform, spread) is deferred to the
 * next processBlock() on the audio thread. Non-finite setter arguments are
 * ignored.
 *
 * Dependencies: RingBuffer.h, Oscillator.h, DryWetMixer.h, DspMath.h,
 *               AudioSpec.h, AudioBuffer.h, DenormalGuard.h, Interpolation.h,
 *               StateBlob.h, BucketBrigade.h, Biquad.h.
 */

#include "../Core/RingBuffer.h"
#include "../Core/Oscillator.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DenormalGuard.h"
#include "../Core/Interpolation.h"
#include "../Core/StateBlob.h"
#include "../Core/BucketBrigade.h"
#include "../Core/Biquad.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class Chorus
 * @brief Multi-voice chorus/flanger with true stereo spread and smooth parameter handling.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Chorus
{
public:
    ~Chorus() = default; // non-virtual: leaf class, no virtual dispatch (framework policy)

    static constexpr int kMaxVoices = 4;
    static constexpr int kMaxChannels = 16;

    /** @brief Fractional read kernel of the modulated delay lines. */
    enum class Interpolation
    {
        Hermite,  ///< 4-point cubic (default).
        Sinc,     ///< 32-tap Kaiser sinc: transparent to 15 kHz at 44.1/48 kHz,
                  ///< about 5x the read cost; the shortest delay is 16 samples.
        Linear    ///< 2-point linear: its loss of highs (a gentle low-pass that
                  ///< moves with the fraction, -6 dB at Nyquist half way between
                  ///< samples) is part of the sound of many chorus units.
    };

    /** @brief The engine (setModel()). */
    enum class Model
    {
        Digital,  ///< Delay lines read by interpolation, voices and feedback (default).
        Analog    ///< Two bucket-brigade lines on the mono sum (see the file notes).
    };

    /** @brief Presets of the analog model (setAnalogMode()). */
    enum class AnalogMode
    {
        Slow,     ///< 0.5 Hz, 1.6 ms of depth about 5 ms, taps in opposition.
        Fast,     ///< 0.85 Hz, the same depth and centre.
        Vibrato   ///< 7 Hz, 0.25 ms about 7 ms, taps in phase.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the chorus for processing, allocating delay lines.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification (sample rate, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        spec_ = spec;
        mixer_.prepare(spec);

        // Max delay: 30ms center + 20ms depth = 50ms total safety margin.
        // The +4 pad keeps the interpolator's read window (intDelay + 2)
        // strictly inside the ring even when the requested size lands exactly
        // on a power of two (RingBuffer rounds its capacity up to one).
        // The sinc reader also needs SincInterpolator::kBefore + 1 samples
        // behind its read point: a +20 pad covers both readers.
        maxDelaySamples_ = static_cast<int>(spec.sampleRate * 0.05) + 1;

        for (int ch = 0; ch < spec.numChannels && ch < kMaxChannels; ++ch)
            delayLines_[ch].prepare(maxDelaySamples_ + 20);
        sinc_ = &sincTable();   // built on this setup thread, once per process
        voiceFadeStep_ = static_cast<T>(1.0 / std::max(1.0, spec.sampleRate * 0.05));

        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            for (int v = 0; v < kMaxVoices; ++v)
            {
                lfos_[ch][v].prepare(spec.sampleRate);
                // LFO duty: keep the waveforms inside [-1, 1] (no minBLEP overshoot).
                lfos_[ch][v].setAntiAliasing(Oscillator<T>::AntiAliasing::PolyBLEP);
                lfos_[ch][v].setWaveform(lfoWaveform_.load(std::memory_order_relaxed));
            }
        }

        // Initialize smoothing states
        smoothedCenterSamples_ = centerDelayMs_.load(std::memory_order_relaxed) * static_cast<T>(spec.sampleRate) / T(1000);
        smoothedDepthSamples_  = depthMs_.load(std::memory_order_relaxed) * static_cast<T>(spec.sampleRate) / T(1000);

        // The analog engine: at half the rate from 176.4 kHz; below 88.2 kHz
        // its lines run at twice the rate, so the images of their clock are
        // filtered as in the device instead of folding into the band.
        halfRate_ = spec.sampleRate >= 176400.0;
        engineRate_ = halfRate_ ? 0.5 * spec.sampleRate : spec.sampleRate;
        const int lineFactor = engineRate_ < 88200.0 ? 2 : 1;
        for (auto& b : bbd_)
        {
            b.setPrecision(BucketBrigade<T>::Precision::High);
            b.prepare(engineRate_, kAnalogStages, 4000.0, lineFactor);
        }
        hissHpCoeff_ = std::exp(-2.0 * 3.141592653589793 * kHissHighPassHz / engineRate_);
        switchBuffer_.resize(spec.numChannels, spec.maxBlockSize);
        switchFade_ = std::max(1, static_cast<int>(kSwitchFadeSeconds * spec.sampleRate));
        analogCornerFor_ = -1.0;
        analogGlide_ = 1.0 - std::exp(-1.0 / (kAnalogGlideSeconds * engineRate_));
        {
            static constexpr double kQ4[2] = { 0.5411961001461970, 1.3065629648763764 };
            for (int k = 0; k < 2; ++k)
                decimate_[static_cast<size_t>(k)] = BiquadCoeffs::makeLowPass(spec.sampleRate, std::min(20000.0, 0.2 * spec.sampleRate), kQ4[k]);
        }
        hpCoeff_ = std::exp(-2.0 * 3.141592653589793 * 5.0 / spec.sampleRate);
        noiseCoeff_ = 0.0;

        updateLfoPhases();
        reset();
    }

    /**
     * @brief Processes an audio block in-place.
     *
     * Channels beyond the prepared count pass through untouched.
     *
     * @param buffer Audio buffer view to process (planar layout expected).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (maxDelaySamples_ <= 0) return;   // not prepared: pass through
        DenormalGuard denormalGuard;

        // Front-door non-finite guard: a single NaN/Inf input sample would
        // recirculate through the recursive fbState_ path (fastTanh passes
        // NaN) and mute the voice forever. Replace non-finite input with 0 before
        // it reaches the dry snapshot or any delay/feedback state. No-op on finite
        // input, so conformance metrics stay byte-identical.
        const int nS = buffer.getNumSamples();
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            T* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                if (!std::isfinite(d[i])) d[i] = T(0);
        }

        // The dry/wet mixer holds one prepared block: a longer host block runs
        // in prepared-size slices so its tail is mixed too.
        for (int offset = 0; offset < nS; offset += spec_.maxBlockSize)
            processChunk(buffer.getSubView(offset, std::min(spec_.maxBlockSize, nS - offset)));
    }

    /**
     * @brief Resets delay lines and LFO phases.
     */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            delayLines_[ch].reset();
            fbState_[ch] = T(0);
        }
        mixer_.reset();
        for (auto& b : bbd_) b.reset();
        analogPhase_ = 0.0;
        analogPrimed_ = false;
        hpX1_ = hpY1_ = 0.0;
        for (auto& s : decimateState_) s = {};
        engineOut_ = {};
        engineLast_ = {};
        noiseLp_ = {};
        hissHpX_ = {};
        hissHpY_ = {};
        decimatePhase_ = 0;
        switching_ = false;
        playing_ = model_.load(std::memory_order_relaxed);
        // A reset starts on the target layout: phases set, voices at level.
        const int nv = numVoices_.load(std::memory_order_relaxed);
        for (int v = 0; v < kMaxVoices; ++v)
            voiceGain_[static_cast<size_t>(v)] = v < nv ? T(1) : T(0);
        updateLfoPhases();
    }

    /** @brief Selects the fractional read kernel. Thread-safe; an out-of-range value reads as Hermite. */
    void setInterpolation(Interpolation mode) noexcept
    {
        interpolation_.store(mode == Interpolation::Sinc || mode == Interpolation::Linear ? mode : Interpolation::Hermite,
                             std::memory_order_relaxed);
    }

    /** @brief Returns the fractional read kernel. */
    [[nodiscard]] Interpolation getInterpolation() const noexcept
    {
        return interpolation_.load(std::memory_order_relaxed);
    }

    // -- Parameters -------------------------------------------------------------    // -- Parameters -------------------------------------------------------------

    /**
     * @brief Sets the LFO modulation rate.
     * @param hz Frequency in Hertz [0.01 - 20.0]. Non-finite values are ignored.
     */
    void setRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        rate_.store(std::clamp(hz, T(0.01), T(20)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the LFO modulation depth in milliseconds.
     * @param ms Depth in milliseconds [0.0 - 20.0]. Non-finite values are ignored.
     */
    void setDepthMs(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        depthMs_.store(std::clamp(ms, T(0), T(20)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the wet/dry mix ratio.
     * @param dryWet Ratio from 0.0 (fully dry) to 1.0 (fully wet).
     *               Non-finite values are ignored.
     */
    void setMix(T dryWet) noexcept
    {
        if (!std::isfinite(dryWet)) return;
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the number of active voices per channel.
     * @param count Voices from 1 to 4.
     */
    void setVoices(int count) noexcept
    {
        numVoices_.store(std::clamp(count, 1, kMaxVoices), std::memory_order_relaxed);
        lfoPhaseDirty_.store(true, std::memory_order_release); // recomputed on audio thread
    }

    /**
     * @brief Sets the feedback gain.
     * @param amount Feedback from -0.99 to 0.99. Negative values yield flanger
     *               effects. Non-finite values are ignored.
     */
    void setFeedback(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        feedback_.store(std::clamp(amount, T(-0.99), T(0.99)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the base delay time.
     * @param ms Time in milliseconds [0.1 - 30.0]. Non-finite values are ignored.
     */
    void setCenterDelay(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        centerDelayMs_.store(std::clamp(ms, T(0.1), T(30)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the stereo spread amount.
     * @param amount Offset ratio from 0.0 to 1.0. Non-finite values are ignored.
     */
    void setStereoSpread(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        stereoSpread_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Couples depth to rate automatically to prevent pitch artifacts at high speeds.
     * @param enabled True to activate automatic depth attenuation.
     */
    void setAutoDepth(bool enabled) noexcept { autoDepth_.store(enabled, std::memory_order_relaxed); }

    /**
     * @brief Sets the oscillator waveform for all voices.
     * @warning Using Saw or Square waves for delay modulation without additional
     * lowpass filtering will cause audible clicks. Sine or Triangle are recommended.
     * @param wf The desired waveform type (wild enum values clamp).
     */
    void setModWaveform(typename Oscillator<T>::Waveform wf) noexcept
    {
        // Publish only; the audio thread reapplies it (the oscillators are not
        // thread-safe, so we must not write their state from the control thread).
        const int w = std::clamp(static_cast<int>(wf), 0,
                                 static_cast<int>(Oscillator<T>::Waveform::Triangle));
        lfoWaveform_.store(static_cast<typename Oscillator<T>::Waveform>(w),
                           std::memory_order_relaxed);
        waveformDirty_.store(true, std::memory_order_release);
    }


    /** @brief Selects the engine (see Model). Thread-safe; the change fills the incoming engine, then crosses over (see the file notes). */
    void setModel(Model m) noexcept
    {
        model_.store(m == Model::Analog ? Model::Analog : Model::Digital, std::memory_order_relaxed);
    }
    /** @return The engine. */
    [[nodiscard]] Model getModel() const noexcept { return model_.load(std::memory_order_relaxed); }

    /** @return The LFO rate in hertz. */
    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    /** @return The modulation depth in milliseconds. */
    [[nodiscard]] T getDepthMs() const noexcept { return depthMs_.load(std::memory_order_relaxed); }
    /** @return The centre delay in milliseconds. */
    [[nodiscard]] T getCenterDelay() const noexcept { return centerDelayMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Loads a preset of the analog model: rate, depth and centre (through
     *        their setters) and whether the taps move in opposition or in
     *        phase (setAnalogTapPhase()). In the analog model the centre, the depth and the taps'
     *        offset glide to a new setting with a 50 ms time constant (the
     *        rate only changes the sweep's speed, so it needs none).
     *        Out-of-range values are clamped.
     */
    void setAnalogMode(AnalogMode mode) noexcept
    {
        const int m = std::clamp(static_cast<int>(mode), 0, 2);
        analogMode_.store(m, std::memory_order_relaxed);
        static constexpr float kRate[3] = { 0.5f, 0.85f, 7.0f };
        static constexpr float kDepth[3] = { 1.6f, 1.6f, 0.25f };
        static constexpr float kCentre[3] = { 5.0f, 5.0f, 7.0f };
        setRate(static_cast<T>(kRate[m]));
        setDepthMs(static_cast<T>(kDepth[m]));
        setCenterDelay(static_cast<T>(kCentre[m]));
        setAnalogTapPhase(m == 2 ? T(0) : T(180));
    }
    /** @return The analog preset last loaded. */
    [[nodiscard]] AnalogMode getAnalogMode() const noexcept { return static_cast<AnalogMode>(analogMode_.load(std::memory_order_relaxed)); }

    /**
     * @brief Hiss of the analog lines: its RMS level in dBFS at the centre
     *        delay (it grows in proportion to the delay), the same at any rate;
     *        -120 and below turn it off (default -84). Non-finite values are
     *        ignored.
     */
    void setAnalogNoise(T dB) noexcept
    {
        if (std::isfinite(dB)) analogNoiseDb_.store(std::clamp(dB, T(-200), T(-40)), std::memory_order_relaxed);
    }
    /** @return The hiss level of the analog lines. */
    [[nodiscard]] T getAnalogNoise() const noexcept { return analogNoiseDb_.load(std::memory_order_relaxed); }

    /**
     * @brief The offset between the analog model's two taps, 0 to 180
     *        degrees of the sweep (180 in opposition, 0 in phase; the
     *        presets set 180, and 0 for Vibrato). It glides with the 50 ms
     *        of the analog settings. Non-finite values are ignored.
     */
    void setAnalogTapPhase(T degrees) noexcept
    {
        if (std::isfinite(degrees)) analogTapPhase_.store(std::clamp(degrees, T(0), T(180)), std::memory_order_relaxed);
    }
    /** @return The offset between the analog taps, in degrees. */
    [[nodiscard]] T getAnalogTapPhase() const noexcept { return analogTapPhase_.load(std::memory_order_relaxed); }

    /**
     * @brief The tone of the analog wet, -1 to 1: the corner of the filter
     *        after the lines moves by up to an octave down or up (0, the
     *        default, leaves it at 7 kHz; 5.6 kHz with the circuit pair).
     *        Non-finite values are ignored.
     */
    void setAnalogTone(T tone) noexcept
    {
        if (std::isfinite(tone)) analogTone_.store(std::clamp(tone, T(-1), T(1)), std::memory_order_relaxed);
    }
    /** @return The tone of the analog wet. */
    [[nodiscard]] T getAnalogTone() const noexcept { return analogTone_.load(std::memory_order_relaxed); }

    /**
     * @brief The analog lines with the filter pair of a chorus circuit (see
     *        the file notes) instead of the Butterworth pair. Thread-safe;
     *        applied at the next block.
     */
    void setAnalogCircuitFilters(bool on) noexcept { analogCircuit_.store(on, std::memory_order_relaxed); }
    /** @return Whether the analog lines use the circuit pair. */
    [[nodiscard]] bool getAnalogCircuitFilters() const noexcept { return analogCircuit_.load(std::memory_order_relaxed); }

    /** @brief The rule of the dry/wet mix (Linear by default; see DryWetMixer::MixRule). Thread-safe. */
    void setMixRule(typename DryWetMixer<T>::MixRule rule) noexcept { mixer_.setMixRule(rule); }
    /** @return The rule of the dry/wet mix. */
    [[nodiscard]] typename DryWetMixer<T>::MixRule getMixRule() const noexcept { return mixer_.getMixRule(); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float: the casts keep the double instantiation
        // compiling (a double matches no StateWriter overload unambiguously).
        StateWriter w(stateId("CHOR"), 1);
        w.write("rate", static_cast<float>(rate_.load(std::memory_order_relaxed)));
        w.write("depthMs", static_cast<float>(depthMs_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("feedback", static_cast<float>(feedback_.load(std::memory_order_relaxed)));
        w.write("centerDelay", static_cast<float>(centerDelayMs_.load(std::memory_order_relaxed)));
        w.write("spread", static_cast<float>(stereoSpread_.load(std::memory_order_relaxed)));
        w.write("voices", numVoices_.load(std::memory_order_relaxed));
        w.write("autoDepth", autoDepth_.load(std::memory_order_relaxed));
        w.write("waveform",
                static_cast<int32_t>(lfoWaveform_.load(std::memory_order_relaxed)));
        w.write("interp", static_cast<int32_t>(interpolation_.load(std::memory_order_relaxed)));
        w.write("model", static_cast<int32_t>(getModel()));
        w.write("analogMode", analogMode_.load(std::memory_order_relaxed));
        w.write("analogNoise", static_cast<float>(getAnalogNoise()));
        w.write("analogTapPhase", static_cast<float>(getAnalogTapPhase()));
        w.write("analogTone", static_cast<float>(getAnalogTone()));
        w.write("analogCircuit", getAnalogCircuitFilters());
        w.write("mixRule", static_cast<int32_t>(getMixRule()));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("CHOR")) return false;
        setRate(static_cast<T>(r.read("rate", 1.0f)));
        setDepthMs(static_cast<T>(r.read("depthMs", 3.5f)));
        setMix(static_cast<T>(r.read("mix", 0.5f)));
        setFeedback(static_cast<T>(r.read("feedback", 0.0f)));
        setCenterDelay(static_cast<T>(r.read("centerDelay", 7.0f)));
        setStereoSpread(static_cast<T>(r.read("spread", 0.5f)));
        setVoices(r.read("voices", 2));
        setAutoDepth(r.read("autoDepth", false));
        setModWaveform(static_cast<typename Oscillator<T>::Waveform>(
            r.read("waveform", 0)));
        setInterpolation(static_cast<Interpolation>(r.read("interp", 0)));
        setModel(r.read("model", 0) == 1 ? Model::Analog : Model::Digital);
        // The preset index only (rate, depth and centre were read above).
        analogMode_.store(std::clamp(static_cast<int>(r.read("analogMode", 0)), 0, 2), std::memory_order_relaxed);
        setAnalogNoise(static_cast<T>(r.read("analogNoise", -84.0f)));
        // A state without the offset takes the one of its preset.
        setAnalogTapPhase(static_cast<T>(r.read("analogTapPhase", analogMode_.load(std::memory_order_relaxed) == 2 ? 0.0f : 180.0f)));
        setAnalogTone(static_cast<T>(r.read("analogTone", 0.0f)));
        setAnalogCircuitFilters(r.read("analogCircuit", false));
        setMixRule(static_cast<typename DryWetMixer<T>::MixRule>(std::clamp(static_cast<int>(r.read("mixRule", 0)), 0, 2)));
        return true;
    }

protected:

    /** @brief Phase offset of a channel/voice LFO from channel 0, voice 0. */
    [[nodiscard]] T targetPhase(int ch, int v, int nv, T spread) const noexcept
    {
        // Spread spans half a cycle across the prepared channels: 1.0 puts the
        // first and last channel 180 degrees apart (a stereo pair).
        const int lastCh = std::max(1, std::min(spec_.numChannels, kMaxChannels) - 1);
        const T chOffset = static_cast<T>(ch) * spread / (T(2) * static_cast<T>(lastCh));
        const T phase = static_cast<T>(v) / static_cast<T>(std::max(1, nv)) + chOffset;
        return phase - std::floor(phase);
    }

    /**
     * @brief Sets every LFO phase to its target outright (prepare and reset;
     *        while running, the LFOs glide instead).
     */
    void updateLfoPhases() noexcept
    {
        const int nv = numVoices_.load(std::memory_order_relaxed);
        const T spreadVal = stereoSpread_.load(std::memory_order_relaxed);
        lastSpread_ = spreadVal;
        for (int ch = 0; ch < kMaxChannels; ++ch)
            for (int v = 0; v < kMaxVoices; ++v)
                lfos_[ch][v].setPhase(targetPhase(ch, v, nv, spreadVal));
    }

    /** @brief The shared sinc table (built on first use: prepare() calls it). */
    static const SincInterpolator<T>& sincTable()
    {
        static const SincInterpolator<T> table;
        return table;
    }

    /** @brief Sinc read `delay` samples behind the newest sample of `ring`
     *  (the RingBuffer::read() convention; needs delay >= 16). */
    [[nodiscard]] T readSinc(const RingBuffer<T>& ring, T delay) const noexcept
    {
        const int dInt = static_cast<int>(delay);
        const double dFrac = static_cast<double>(delay) - static_cast<double>(dInt);
        const int64_t newest = static_cast<int64_t>(ring.getWritePosition()) - 1;
        const int64_t intPos = newest - dInt - (dFrac > 0.0 ? 1 : 0);
        return sinc_->readRing(ring.data(), static_cast<int64_t>(ring.getCapacity() - 1), intPos,
                               dFrac > 0.0 ? 1.0 - dFrac : 0.0);
    }

    /** @brief One modulated read with the selected kernel. */
    [[nodiscard]] T readTap(const RingBuffer<T>& ring, T delay, Interpolation interp) const noexcept
    {
        if (interp == Interpolation::Sinc) return readSinc(ring, delay);
        if (interp == Interpolation::Linear) return ring.template readInterpolated<InterpMethod::Linear>(delay);
        return ring.readInterpolated(delay);
    }

    /** @brief One block of at most the prepared size. */
    void processChunk(AudioBufferView<T> buffer) noexcept
    {
        // Clamp to the PREPARED channel count: delay lines beyond
        // spec_.numChannels were never allocated and would read as silence.
        const int nCh = std::min({ buffer.getNumChannels(), spec_.numChannels, kMaxChannels });
        const int nS  = buffer.getNumSamples();

        const T mixVal = mix_.load(std::memory_order_relaxed);

        mixer_.pushDry(buffer);
        const Model wanted = model_.load(std::memory_order_relaxed);
        if (!switching_ && wanted != playing_) startSwitch(wanted);
        if (switching_) processSwitch(buffer, nCh, nS);
        else runEngine(playing_, buffer, nCh, nS);
        mixer_.mixWet(buffer, mixVal);
    }

    void runEngine(Model engine, AudioBufferView<T> buffer, int nCh, int nS) noexcept
    {
        if (engine == Model::Analog) processAnalog(buffer, nCh, nS);
        else processDigital(buffer, nCh, nS);
    }

    /// Starts a change of engine: the incoming one from silence, unheard until its line is full.
    void startSwitch(Model to) noexcept
    {
        switching_ = true;
        incoming_ = to;
        const double reach = static_cast<double>(centerDelayMs_.load(std::memory_order_relaxed) + depthMs_.load(std::memory_order_relaxed)) * 1e-3;
        if (to == Model::Analog)
        {
            for (auto& b : bbd_) b.reset();
            analogPrimed_ = false;
            analogCornerFor_ = -1.0;
            hpX1_ = hpY1_ = 0.0;
            for (auto& s : decimateState_) s = {};
            engineOut_ = {};
            engineLast_ = {};
            noiseLp_ = {};
            hissHpX_ = {};
            hissHpY_ = {};
            // The line, its filters and their settling.
            fillLeft_ = static_cast<int>((reach + kSwitchSettleSeconds) * spec_.sampleRate);
        }
        else
        {
            for (int ch = 0; ch < kMaxChannels; ++ch) fbState_[ch] = T(0);
            fillLeft_ = static_cast<int>(reach * spec_.sampleRate) + 4;
        }
        fadeLeft_ = switchFade_;
    }

    /// A block of a change of engine: both run; the incoming one is heard once its line is full.
    void processSwitch(AudioBufferView<T> buffer, int nCh, int nS) noexcept
    {
        auto incoming = switchBuffer_.toView().getSubView(0, nS);
        for (int ch = 0; ch < nCh; ++ch) std::copy_n(buffer.getChannel(ch), nS, incoming.getChannel(ch));
        runEngine(playing_, buffer, nCh, nS);
        T* scratch[kMaxChannels] = {};
        for (int ch = 0; ch < nCh; ++ch) scratch[ch] = incoming.getChannel(ch);
        runEngine(incoming_, AudioBufferView<T>(scratch, nCh, nS), nCh, nS);
        const int fill = fillLeft_, fade = fadeLeft_;
        for (int ch = 0; ch < nCh; ++ch)
        {
            T* out = buffer.getChannel(ch);
            const T* in = incoming.getChannel(ch);
            int f = fill, g = fade;
            for (int i = 0; i < nS; ++i)
            {
                if (f > 0) { --f; continue; }
                if (g > 0) --g;
                const T w = T(1) - static_cast<T>(g) / static_cast<T>(switchFade_);
                out[i] += (in[i] - out[i]) * w;
            }
        }
        const int filled = std::min(fillLeft_, nS);
        fillLeft_ -= filled;
        fadeLeft_ = std::max(0, fadeLeft_ - (nS - filled));
        if (fillLeft_ == 0 && fadeLeft_ == 0)
        {
            playing_ = incoming_;
            switching_ = false;
        }
    }

    /** @brief The digital engine on one block (wet only). */
    void processDigital(AudioBufferView<T> buffer, int nCh, int nS) noexcept
    {
        T targetRate   = rate_.load(std::memory_order_relaxed);
        T targetDepth  = depthMs_.load(std::memory_order_relaxed);
        T targetCenter = centerDelayMs_.load(std::memory_order_relaxed);
        T fbVal        = feedback_.load(std::memory_order_relaxed);
        int nVoices    = numVoices_.load(std::memory_order_relaxed);
        bool autoD     = autoDepth_.load(std::memory_order_relaxed);
        Interpolation interp = interpolation_.load(std::memory_order_relaxed);
        if (interp == Interpolation::Sinc && !sinc_) interp = Interpolation::Hermite;

        // Deferred LFO config on the AUDIO thread (setters only publish). A
        // voice-count change needs no phase reset: the glide below moves the
        // LFOs to the new layout.
        (void)lfoPhaseDirty_.exchange(false, std::memory_order_acquire);
        if (waveformDirty_.exchange(false, std::memory_order_acquire))
        {
            const auto wf = lfoWaveform_.load(std::memory_order_relaxed);
            for (int ch = 0; ch < kMaxChannels; ++ch)
                for (int v = 0; v < kMaxVoices; ++v)
                    lfos_[ch][v].setWaveform(wf);
        }
        const T spreadVal = stereoSpread_.load(std::memory_order_relaxed);
        lastSpread_ = spreadVal;

        // Target calculations
        T targetCenterSamples = targetCenter * static_cast<T>(spec_.sampleRate) / T(1000);
        T targetDepthSamples  = targetDepth * static_cast<T>(spec_.sampleRate) / T(1000);

        if (autoD && targetRate > T(0.01))
            targetDepthSamples /= std::sqrt(targetRate);

        // Keep depth <= center - 2 samples: a deeper sweep would flatten
        // against the 1-sample read floor, squaring off the modulation shape.
        targetDepthSamples = std::min(targetDepthSamples,
                                      std::max(T(0), targetCenterSamples - T(2)));

        // Voices: a voice joining starts silent on its target phase; a voice
        // leaving keeps running until its fade reaches zero.
        std::array<T, kMaxVoices> gainStart {};
        std::array<T, kMaxVoices> gainTarget {};
        int nRun = 0;
        bool fading = false;
        for (int v = 0; v < kMaxVoices; ++v)
        {
            const size_t vi = static_cast<size_t>(v);
            gainTarget[vi] = v < nVoices ? T(1) : T(0);
            gainStart[vi] = voiceGain_[vi];
            if (gainStart[vi] == T(0) && gainTarget[vi] > T(0))
                for (int ch = 0; ch < kMaxChannels; ++ch)
                    lfos_[ch][v].setPhase(T(lfos_[0][0].getPhase()) + targetPhase(ch, v, nVoices, spreadVal));
            if (gainStart[vi] > T(0) || gainTarget[vi] > T(0)) nRun = v + 1;
            fading = fading || gainStart[vi] != gainTarget[vi];
        }
        const T voiceNorm = T(1) / std::sqrt(static_cast<T>(nVoices));

        // Update LFO frequencies. Channel 0 / voice 0 is the phase reference;
        // every other LFO that sits off its target phase runs slightly faster
        // or slower until it arrives (a continuous phase glide: no jump in the
        // delay time, so no click). The offset is bounded to half a cycle per
        // second and never overshoots within one block.
        const T blockSeconds = static_cast<T>(nS) / static_cast<T>(spec_.sampleRate);
        const T glideTime = std::max(T(0.1), blockSeconds);
        const T refPhase = lfos_[0][0].getPhase();
        for (int ch = 0; ch < nCh; ++ch)
            for (int v = 0; v < nRun; ++v)
            {
                T freq = targetRate;
                if (ch != 0 || v != 0)
                {
                    T err = targetPhase(ch, v, nVoices, spreadVal)
                          - (lfos_[ch][v].getPhase() - refPhase);
                    err -= std::floor(err + T(0.5));   // signed, in [-0.5, 0.5)
                    if (std::abs(err) > T(1e-5))      // settled LFOs run untouched
                        freq += std::clamp(err / glideTime, -kMaxGlideHz, kMaxGlideHz);
                }
                lfos_[ch][v].setFrequency(freq);
            }

        // One-pole smoothing coefficient, sample-rate invariant: 240/fs gives
        // the same ~4.2 ms time constant at any rate (and is bit-identical to
        // the historical 0.005 at 48 kHz; the old fixed value made parameter
        // glides twice as fast at 96 kHz as at 48 kHz).
        T smoothCoeff = std::min(T(1), T(240) / static_cast<T>(spec_.sampleRate));
        const T minDelay = interp == Interpolation::Sinc ? T(SincInterpolator<T>::kAfter) : T(1);

        // 2. Channel outer, sample inner (cache-friendly per-channel state)
        for (int ch = 0; ch < nCh; ++ch)
        {
            auto* channelData = buffer.getChannel(ch);
            auto& ring = delayLines_[ch];

            // Local channel copies of the smoothing state (kept in registers)
            T centerSamples = smoothedCenterSamples_;
            T depthSamples  = smoothedDepthSamples_;

            for (int i = 0; i < nS; ++i)
            {
                // Smooth parameters per sample
                centerSamples += (targetCenterSamples - centerSamples) * smoothCoeff;
                depthSamples  += (targetDepthSamples - depthSamples) * smoothCoeff;

                T input = channelData[i];

                // Saturating analog-style feedback to prevent harsh digital
                // clipping. fastTanh is internally clamped; at typical levels
                // |argument| < 1, where its error is < 0.05%.
                T feedbackSig = fastTanh(fbState_[ch] * fbVal);
                ring.push(input + feedbackSig);

                T wetRaw = T(0);

                if (!fading)
                {
                    // Voice accumulation (steady voice count)
                    for (int v = 0; v < nVoices; ++v)
                    {
                        T lfoVal = lfos_[ch][v].getNextSample();
                        T rawDelay = centerSamples + lfoVal * depthSamples;

                        // Safety clamp against ring buffer limits
                        T delay = std::clamp(rawDelay, minDelay, static_cast<T>(maxDelaySamples_ - 2));
                        wetRaw += readTap(ring, delay, interp);
                    }
                    fbState_[ch] = wetRaw / static_cast<T>(nVoices);
                    channelData[i] = wetRaw * voiceNorm;
                    continue;
                }

                T gainSum = T(0);
                T powerSum = T(0);

                // Voice accumulation while voices fade in or out
                for (int v = 0; v < nRun; ++v)
                {
                    const size_t vi = static_cast<size_t>(v);
                    const T g = moveTowards(gainStart[vi], gainTarget[vi],
                                            voiceFadeStep_ * static_cast<T>(i + 1));
                    T lfoVal = lfos_[ch][v].getNextSample();
                    T rawDelay = centerSamples + lfoVal * depthSamples;

                    // Safety clamp against ring buffer limits
                    T delay = std::clamp(rawDelay, minDelay, static_cast<T>(maxDelaySamples_ - 2));

                    const T tap = readTap(ring, delay, interp);
                    wetRaw += g == T(1) ? tap : g * tap;
                    gainSum += g;
                    powerSum += g * g;
                }

                // Equal-power sum, normalised by the voices actually sounding
                // (1/sqrt(N) once every fade is complete).
                fbState_[ch] = wetRaw / std::max(gainSum, T(1e-6));
                channelData[i] = wetRaw / std::sqrt(std::max(powerSum, T(1e-12)));
            }

            // Save state back for the next block (only done by channel 0 to keep channels synced)
            if (ch == 0)
            {
                smoothedCenterSamples_ = centerSamples;
                smoothedDepthSamples_  = depthSamples;
            }
        }

        for (int v = 0; v < kMaxVoices; ++v)
        {
            const size_t vi = static_cast<size_t>(v);
            voiceGain_[vi] = moveTowards(gainStart[vi], gainTarget[vi],
                                         voiceFadeStep_ * static_cast<T>(nS));
        }
    }

    [[nodiscard]] static T moveTowards(T from, T to, T delta) noexcept
    {
        return from < to ? std::min(to, from + delta) : std::max(to, from - delta);
    }

    /**
     * @brief The analog model on one block: the mono sum, high-passed, into
     *        two bucket-brigade lines (sharing their input low-pass) whose
     *        delays a slightly asymmetric triangle moves, each with its hiss;
     *        the even channels take the first line, the odd ones the second.
     *
     * The asymmetry lives in the speed of one phase (slower on the way up),
     * and both taps read the same symmetric triangle of that phase, half a
     * turn apart in opposition: the second is then the first inverted, so the
     * two delays always add up to twice the centre, as with a triangle and an
     * inverter; in phase (Vibrato) they are the same. The centre and the depth
     * ramp across the block toward the values set and then glide with the
     * taps' offset (kAnalogGlideSeconds). At half rate the sum is low-passed
     * and every other sample runs the engine; the sample after a run is that
     * run's output and the run's own sample the mean of it and the one
     * before, so the output keeps one sample's delay throughout.
     */
    void processAnalog(AudioBufferView<T> buffer, int nCh, int nS) noexcept
    {
        const double rate = static_cast<double>(rate_.load(std::memory_order_relaxed));
        const double centreSet = static_cast<double>(centerDelayMs_.load(std::memory_order_relaxed)) * 1e-3;
        const double depthSet = std::min(static_cast<double>(depthMs_.load(std::memory_order_relaxed)) * 1e-3, 0.9 * centreSet);
        const double offsetSet = static_cast<double>(analogTapPhase_.load(std::memory_order_relaxed)) / 360.0;
        if (!analogPrimed_)
        {
            centreFrom_ = centreNow_ = centreSet;
            depthFrom_ = depthNow_ = depthSet;
            offsetNow_ = offsetSet;
            analogPrimed_ = true;
        }
        // The filters sit at their fixed corners, lower only when the slowest
        // clock the sweep needs (where it is or where it is going, whichever
        // is longer) would leave the input filter too high for it.
        // The corner moves at most an octave in a quarter of a second: a jump
        // of the filters' coefficients would leave a step in their output.
        const double reach = std::max(centreNow_ + std::min(depthNow_, 0.9 * centreNow_), centreSet + depthSet);
        double slowest = reach > 0.0 ? static_cast<double>(kAnalogStages) / (2.0 * reach) : 1e5;
        if (analogCornerFor_ > 0.0)
        {
            const double most = std::exp2(static_cast<double>(nS) / (kCornerOctaveSeconds * spec_.sampleRate));
            slowest = std::clamp(slowest, analogCornerFor_ / most, analogCornerFor_ * most);
        }
        const double tone = static_cast<double>(analogTone_.load(std::memory_order_relaxed));
        if (std::abs(slowest - analogCornerFor_) > 1e-3 * slowest || tone != analogToneFor_
            || analogCircuit_.load(std::memory_order_relaxed) != circuitFor_)
        {
            const bool circuit = analogCircuit_.load(std::memory_order_relaxed);
            if (circuit != circuitFor_)
            {
                for (auto& b : bbd_) b.setShape(circuit ? BucketBrigade<T>::Shape::Chorus : BucketBrigade<T>::Shape::Butterworth);
                circuitFor_ = circuit;
            }
            const double inputHz = circuit ? kCircuitInputHz : kAnalogCornerHz;
            const double outputHz = circuit ? kCircuitOutputHz : kAnalogCornerHz;
            const double input = std::min(inputHz, (circuit ? kCircuitClockShare : kAnalogClockShare) * slowest);
            const double output = input * (outputHz / inputHz) * std::exp2(tone);
            for (auto& b : bbd_) b.setCorners(input, output);
            const double heard = bbd_[0].getOutputCorner();
            noiseCoeff_ = 1.0 - std::exp(-2.0 * 3.141592653589793 * std::min(heard, 0.45 * engineRate_) / engineRate_);
            analogCornerFor_ = slowest;
            analogToneFor_ = tone;
        }
        const T noiseDb = analogNoiseDb_.load(std::memory_order_relaxed);
        // White noise through a one-pole at the corner, scaled by that
        // filter's own gain: the same level (and density) at any rate.
        const double noiseAmp = noiseDb > T(-120)
            ? std::pow(10.0, static_cast<double>(noiseDb) / 20.0)
                  * 1.7320508075688772 / std::sqrt(std::max(1e-9, noiseCoeff_ / (2.0 - noiseCoeff_)))
            : 0.0;
        const double step = rate / engineRate_;
        const auto triangle = [](double p) noexcept {
            p -= std::floor(p);
            return p < 0.5 ? 4.0 * p - 1.0 : 3.0 - 4.0 * p;
        };
        const auto glide = [this](double& now, double goal) noexcept {
            now += analogGlide_ * (goal - now);
            if (std::abs(goal - now) < 1e-12) now = goal;
        };
        const double perSample = 1.0 / static_cast<double>(std::max(1, nS));
        const int channels = std::max(1, nCh);
        for (int i = 0; i < nS; ++i)
        {
            double mono = 0.0;
            for (int ch = 0; ch < nCh; ++ch) mono += static_cast<double>(buffer.getChannel(ch)[i]);
            mono /= channels;
            const double hp = mono - hpX1_ + hpCoeff_ * hpY1_;
            hpX1_ = mono;
            hpY1_ = hp;
            double x = hp;
            bool run = true;
            if (halfRate_)
            {
                for (size_t k = 0; k < 2; ++k)
                {
                    auto& s = decimateState_[k];
                    const auto& c = decimate_[k];
                    const double y = c.b0 * x + s[0];
                    s[0] = c.b1 * x - c.a1 * y + s[1];
                    s[1] = c.b2 * x - c.a2 * y;
                    x = y;
                }
                decimatePhase_ ^= 1;
                run = decimatePhase_ == 1;
            }
            if (run)
            {
                const double u = static_cast<double>(i + 1) * perSample;
                glide(centreNow_, centreFrom_ + u * (centreSet - centreFrom_));
                glide(depthNow_, depthFrom_ + u * (depthSet - depthFrom_));
                glide(offsetNow_, offsetSet);
                advanceAnalogPhase(step);
                const double centre = centreNow_;
                const double depth = std::min(depthNow_, 0.9 * centre);
                const double d[2] = { centre + depth * triangle(analogPhase_),
                                      centre + depth * triangle(analogPhase_ + offsetNow_) };
                double y[2];
                bbd_[0].processPair(bbd_[1], static_cast<T>(x), d[0], d[1], y[0], y[1]);
                for (size_t k = 0; k < 2; ++k)
                {
                    if (noiseAmp > 0.0)
                    {
                        noiseRng_ = noiseRng_ * 1664525u + 1013904223u;
                        const double n = static_cast<double>(noiseRng_ >> 8) / 8388608.0 - 1.0;
                        // A hiss without lows: a high-pass, then the low-pass at the corner.
                        const double hissHp = hissHpCoeff_ * (hissHpY_[k] + n - hissHpX_[k]);
                        hissHpX_[k] = n;
                        hissHpY_[k] = hissHp;
                        noiseLp_[k] += noiseCoeff_ * (hissHp - noiseLp_[k]);
                        y[k] += noiseAmp * (d[k] / std::max(centre, 1e-4)) * noiseLp_[k];
                    }
                    engineLast_[k] = engineOut_[k];
                    engineOut_[k] = y[k];
                }
            }
            for (int ch = 0; ch < nCh; ++ch)
            {
                const size_t k = static_cast<size_t>(ch & 1);
                const double out = (halfRate_ && run) ? 0.5 * (engineOut_[k] + engineLast_[k]) : engineOut_[k];
                buffer.getChannel(ch)[i] = static_cast<T>(out);
            }
        }
        centreFrom_ = centreSet;
        depthFrom_ = depthSet;
    }

    /**
     * @brief Moves the sweep's phase by @p step periods of time: the phase
     *        runs at 0.5 / kRise per period on its rising half and at
     *        0.5 / (1 - kRise) on its falling half, so the rise takes kRise of
     *        the period.
     */
    void advanceAnalogPhase(double step) noexcept
    {
        double t = step;
        for (int edge = 0; edge < 4 && t > 0.0; ++edge)
        {
            const bool rising = analogPhase_ < 0.5;
            const double speed = rising ? 0.5 / kRise : 0.5 / (1.0 - kRise);
            const double need = ((rising ? 0.5 : 1.0) - analogPhase_) / speed;
            if (t < need)
            {
                analogPhase_ += t * speed;
                return;
            }
            t -= need;
            analogPhase_ = rising ? 0.5 : 0.0;
        }
    }

    static constexpr int kAnalogStages = 512;   ///< Stages of each analog line.
    static constexpr double kRise = 0.53;       ///< The sweep's rising share of its period.
    static constexpr double kAnalogGlideSeconds = 0.05;   ///< Glide of the analog centre, depth and offset.
    static constexpr double kCornerOctaveSeconds = 0.25;  ///< The analog filters' corner moves an octave in this time at most.
    static constexpr double kAnalogCornerHz = 7000.0;     ///< The analog filters' corner (tone 0).
    static constexpr double kAnalogClockShare = 0.2;      ///< The corner at most this share of the slowest clock.
    static constexpr double kCircuitInputHz = 8000.0;     ///< The circuit pair: the input filter's corner.
    static constexpr double kCircuitOutputHz = 5600.0;    ///< The circuit pair: the output filter's corner (tone 0).
    static constexpr double kCircuitClockShare = 0.23;    ///< The circuit pair's input corner at most this share of the slowest clock.
    static constexpr double kHissHighPassHz = 200.0;      ///< The hiss has no lows under this.
    static constexpr double kSwitchFadeSeconds = 0.01;    ///< A change of engine crosses over in this.
    static constexpr double kSwitchSettleSeconds = 0.003; ///< The analog line's filters settle in this after it fills.

    static constexpr T kMaxGlideHz = T(0.5);   ///< Phase glide: half a cycle per second.

    AudioSpec spec_ {};
    int maxDelaySamples_ { 0 };

    // Atomic parameters (updated via UI thread)
    std::atomic<T> rate_ { T(1) };
    std::atomic<T> depthMs_ { T(3.5) };
    std::atomic<T> mix_ { T(0.5) };
    std::atomic<T> feedback_ { T(0) };
    std::atomic<T> centerDelayMs_ { T(7) };
    std::atomic<T> stereoSpread_ { T(0.5) };
    std::atomic<int> numVoices_ { 2 };
    std::atomic<bool> autoDepth_ { false };

    T lastSpread_ { T(0.5) };
    std::atomic<typename Oscillator<T>::Waveform> lfoWaveform_ { Oscillator<T>::Waveform::Sine };
    std::atomic<bool> lfoPhaseDirty_ { false };  // voices changed -> recompute phases (audio thread)
    std::atomic<bool> waveformDirty_ { false };  // waveform changed -> reapply (audio thread)

    // Smoothed internal state variables (Audio thread only)
    T smoothedCenterSamples_ { T(0) };
    T smoothedDepthSamples_ { T(0) };

    // Processing arrays
    std::array<RingBuffer<T>, kMaxChannels> delayLines_ {};
    // 2D Array: Avoids dynamic phase calculations in the hot loop
    std::array<std::array<Oscillator<T>, kMaxVoices>, kMaxChannels> lfos_ {};
    std::array<T, kMaxChannels> fbState_ {};
    std::array<T, kMaxVoices> voiceGain_ {};        ///< Voice fades (audio thread).
    T voiceFadeStep_ = T(1);
    std::atomic<Interpolation> interpolation_ { Interpolation::Hermite };
    const SincInterpolator<T>* sinc_ = nullptr;     ///< Set by prepare().

    DryWetMixer<T> mixer_;

    // The analog model.
    std::atomic<Model> model_ { Model::Digital };
    std::atomic<int> analogMode_ { 0 };
    std::atomic<T> analogNoiseDb_ { T(-84) };
    std::atomic<T> analogTapPhase_ { T(180) };
    std::atomic<T> analogTone_ { T(0) };
    std::atomic<bool> analogCircuit_ { false };
    bool circuitFor_ = false;
    double analogToneFor_ = 0.0, hissHpCoeff_ = 0.0;
    std::array<double, 2> hissHpX_ {}, hissHpY_ {};
    // A change of engine.
    Model playing_ = Model::Digital, incoming_ = Model::Digital;
    bool switching_ = false;
    int fillLeft_ = 0, fadeLeft_ = 0, switchFade_ = 480;
    AudioBuffer<T> switchBuffer_;
    std::array<BucketBrigade<T>, 2> bbd_ {};
    bool halfRate_ = false;
    double engineRate_ = 48000.0, analogCornerFor_ = -1.0, hpCoeff_ = 0.0, noiseCoeff_ = 0.0;
    double analogPhase_ = 0.0, hpX1_ = 0.0, hpY1_ = 0.0;
    double analogGlide_ = 1.0, centreNow_ = 0.0, depthNow_ = 0.0, offsetNow_ = 0.5;
    double centreFrom_ = 0.0, depthFrom_ = 0.0;
    bool analogPrimed_ = false;
    std::array<BiquadCoeffs, 2> decimate_ {};
    std::array<std::array<double, 2>, 2> decimateState_ {};
    std::array<double, 2> engineOut_ {}, engineLast_ {}, noiseLp_ {};
    int decimatePhase_ = 0;
    uint32_t noiseRng_ = 0x6B43A9u;
};

} // namespace dspark
