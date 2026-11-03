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
 *   framework's 32-tap Kaiser sinc reader (setInterpolation()).
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
 *               StateBlob.h.
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
        Sinc      ///< 32-tap Kaiser sinc: transparent to 15 kHz at 44.1/48 kHz,
                  ///< about 5x the read cost; the shortest delay is 16 samples.
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
        // A reset starts on the target layout: phases set, voices at level.
        const int nv = numVoices_.load(std::memory_order_relaxed);
        for (int v = 0; v < kMaxVoices; ++v)
            voiceGain_[static_cast<size_t>(v)] = v < nv ? T(1) : T(0);
        updateLfoPhases();
    }

    /** @brief Selects the fractional read kernel. Thread-safe. */
    void setInterpolation(Interpolation mode) noexcept
    {
        interpolation_.store(mode == Interpolation::Sinc ? Interpolation::Sinc : Interpolation::Hermite,
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
        setInterpolation(r.read("interp", 0) == 1 ? Interpolation::Sinc : Interpolation::Hermite);
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

    /** @brief One block of at most the prepared size. */
    void processChunk(AudioBufferView<T> buffer) noexcept
    {
        // Clamp to the PREPARED channel count: delay lines beyond
        // spec_.numChannels were never allocated and would read as silence.
        const int nCh = std::min({ buffer.getNumChannels(), spec_.numChannels, kMaxChannels });
        const int nS  = buffer.getNumSamples();

        // 1. Read atomics once per block
        T targetRate   = rate_.load(std::memory_order_relaxed);
        T targetDepth  = depthMs_.load(std::memory_order_relaxed);
        T targetCenter = centerDelayMs_.load(std::memory_order_relaxed);
        T fbVal        = feedback_.load(std::memory_order_relaxed);
        T mixVal       = mix_.load(std::memory_order_relaxed);
        int nVoices    = numVoices_.load(std::memory_order_relaxed);
        bool autoD     = autoDepth_.load(std::memory_order_relaxed);
        const bool sinc = interpolation_.load(std::memory_order_relaxed) == Interpolation::Sinc && sinc_;

        mixer_.pushDry(buffer);

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
        const T minDelay = sinc ? T(SincInterpolator<T>::kAfter) : T(1);

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
                        wetRaw += sinc ? readSinc(ring, delay) : ring.readInterpolated(delay);
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

                    const T tap = sinc ? readSinc(ring, delay) : ring.readInterpolated(delay);
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

        mixer_.mixWet(buffer, mixVal);
    }

    [[nodiscard]] static T moveTowards(T from, T to, T delta) noexcept
    {
        return from < to ? std::min(to, from + delta) : std::max(to, from - delta);
    }

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
};

} // namespace dspark
