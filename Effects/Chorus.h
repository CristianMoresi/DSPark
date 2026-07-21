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
 *
 * Threading: prepare() belongs to the setup thread (allocates; never call it
 * concurrently with processing). processBlock() and reset() belong to the
 * audio thread. All setters are lock-free atomic publications, safe from any
 * thread; LFO reconfiguration (voices, waveform, spread) is deferred to the
 * next processBlock() on the audio thread. Non-finite setter arguments are
 * ignored.
 *
 * Dependencies: RingBuffer.h, Oscillator.h, DryWetMixer.h, DspMath.h,
 *               AudioSpec.h, AudioBuffer.h, StateBlob.h.
 */

#include "../Core/RingBuffer.h"
#include "../Core/Oscillator.h"
#include "../Core/DryWetMixer.h"
#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
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
        maxDelaySamples_ = static_cast<int>(spec.sampleRate * 0.05) + 1;

        for (int ch = 0; ch < spec.numChannels && ch < kMaxChannels; ++ch)
            delayLines_[ch].prepare(maxDelaySamples_ + 4);

        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            for (int v = 0; v < kMaxVoices; ++v)
            {
                lfos_[ch][v].prepare(spec.sampleRate);
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

        mixer_.pushDry(buffer);

        // Apply deferred LFO config on the AUDIO thread (setters only publish):
        // voices change -> recompute phases; waveform change -> reapply.
        if (lfoPhaseDirty_.exchange(false, std::memory_order_acquire))
            updateLfoPhases();
        if (waveformDirty_.exchange(false, std::memory_order_acquire))
        {
            const auto wf = lfoWaveform_.load(std::memory_order_relaxed);
            for (int ch = 0; ch < kMaxChannels; ++ch)
                for (int v = 0; v < kMaxVoices; ++v)
                    lfos_[ch][v].setWaveform(wf);
        }

        // Check if stereo spread changed (requires phase recalculation)
        T currentSpread = stereoSpread_.load(std::memory_order_relaxed);
        if (std::abs(currentSpread - lastSpread_) > T(0.001))
        {
            lastSpread_ = currentSpread;
            updateLfoPhases();
        }

        // Target calculations
        T targetCenterSamples = targetCenter * static_cast<T>(spec_.sampleRate) / T(1000);
        T targetDepthSamples  = targetDepth * static_cast<T>(spec_.sampleRate) / T(1000);

        if (autoD && targetRate > T(0.01))
            targetDepthSamples /= std::sqrt(targetRate);

        // Keep depth <= center - 2 samples: a deeper sweep would flatten
        // against the 1-sample read floor, squaring off the modulation shape.
        targetDepthSamples = std::min(targetDepthSamples,
                                      std::max(T(0), targetCenterSamples - T(2)));

        // Update LFO frequencies
        for (int ch = 0; ch < nCh; ++ch)
            for (int v = 0; v < nVoices; ++v)
                lfos_[ch][v].setFrequency(targetRate);

        T voiceNorm = T(1) / std::sqrt(static_cast<T>(nVoices));

        // One-pole smoothing coefficient, sample-rate invariant: 240/fs gives
        // the same ~4.2 ms time constant at any rate (and is bit-identical to
        // the historical 0.005 at 48 kHz; the old fixed value made parameter
        // glides twice as fast at 96 kHz as at 48 kHz).
        T smoothCoeff = std::min(T(1), T(240) / static_cast<T>(spec_.sampleRate));

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

                // Voice accumulation
                for (int v = 0; v < nVoices; ++v)
                {
                    T lfoVal = lfos_[ch][v].getNextSample();
                    T rawDelay = centerSamples + lfoVal * depthSamples;

                    // Safety clamp against ring buffer limits
                    T delay = std::clamp(rawDelay, T(1), static_cast<T>(maxDelaySamples_ - 2));

                    wetRaw += ring.readInterpolated(delay);
                }

                fbState_[ch] = wetRaw / static_cast<T>(nVoices);
                channelData[i] = wetRaw * voiceNorm;
            }

            // Save state back for the next block (only done by channel 0 to keep channels synced)
            if (ch == 0)
            {
                smoothedCenterSamples_ = centerSamples;
                smoothedDepthSamples_  = depthSamples;
            }
        }

        mixer_.mixWet(buffer, mixVal);
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
        updateLfoPhases();
    }

    // -- Parameters -------------------------------------------------------------

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
        StateWriter w(stateId("CHOR"), 1);
        w.write("rate", rate_.load(std::memory_order_relaxed));
        w.write("depthMs", depthMs_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("feedback", feedback_.load(std::memory_order_relaxed));
        w.write("centerDelay", centerDelayMs_.load(std::memory_order_relaxed));
        w.write("spread", stereoSpread_.load(std::memory_order_relaxed));
        w.write("voices", numVoices_.load(std::memory_order_relaxed));
        w.write("autoDepth", autoDepth_.load(std::memory_order_relaxed));
        w.write("waveform",
                static_cast<int32_t>(lfoWaveform_.load(std::memory_order_relaxed)));
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
        return true;
    }

protected:

    /**
     * @brief Recalculates LFO phases for all channels and voices based on stereo spread.
     */
    void updateLfoPhases() noexcept
    {
        int nv = numVoices_.load(std::memory_order_relaxed);
        T spreadVal = stereoSpread_.load(std::memory_order_relaxed);

        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            // Calculate base channel offset
            T chOffset = (ch > 0 && spreadVal > T(0))
                ? static_cast<T>(ch) * spreadVal / (T(2) * static_cast<T>(kMaxChannels))
                : T(0);

            for (int v = 0; v < kMaxVoices; ++v)
            {
                T basePhase = static_cast<T>(v) / static_cast<T>(nv);
                T finalPhase = basePhase + chOffset;
                finalPhase -= std::floor(finalPhase); // Wrap to [0, 1)
                lfos_[ch][v].setPhase(finalPhase);
            }
        }
    }

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

    DryWetMixer<T> mixer_;
};

} // namespace dspark
