// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Chorus.h
 * @brief Chorus / Flanger effect with LFO-modulated delay lines.
 *
 * Creates the classic chorus effect by mixing the dry signal with delayed
 * copies whose delay time is modulated by LFOs. Multiple voices with
 * phase-offset LFOs create a rich, wide sound. Negative feedback values
 * produce flanger-style comb filtering.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `chorus.setRate(1.5f); chorus.setDepth(0.5f);`
 * - **Level 2 (intermediate):** Add voices, feedback (flanger), center delay.
 * - **Level 3 (expert):** LFO waveform, stereo spread, interpolation method.
 *
 * Dependencies: RingBuffer.h, Oscillator.h, DryWetMixer.h, AudioSpec.h, AudioBuffer.h.
 *
 * @code
 *   dspark::Chorus<float> chorus;
 *   chorus.prepare(spec);
 *   chorus.setRate(1.5f);    // LFO at 1.5 Hz
 *   chorus.setDepth(0.5f);   // Moderate depth
 *   chorus.setMix(0.5f);     // 50/50
 *   chorus.processBlock(buffer);
 *
 *   // Flanger mode:
 *   chorus.setCenterDelay(1.0f);    // 1 ms base (flanger)
 *   chorus.setFeedback(-0.7f);      // Negative = metallic flanger
 * @endcode
 */

#include "../Core/RingBuffer.h"
#include "../Core/Oscillator.h"
#include "../Core/DryWetMixer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class Chorus
 * @brief Multi-voice chorus/flanger with stereo spread.
 *
 * Each voice has its own LFO with a phase offset for spread. The modulated
 * delay varies between (centerDelay - depth) and (centerDelay + depth)
 * milliseconds. RingBuffer with cubic interpolation ensures smooth
 * fractional-sample reads.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Chorus
{
public:
    virtual ~Chorus() = default;

    static constexpr int kMaxVoices = 4;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the chorus for processing.
     *
     * Allocates delay ring buffers (50ms max) and initializes LFOs.
     *
     * @param spec Audio environment.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        // Max delay: center (up to 30ms) + depth modulation (up to 20ms)
        int maxDelaySamples = static_cast<int>(spec.sampleRate * 0.05) + 1;

        for (int ch = 0; ch < spec.numChannels && ch < kMaxChannels; ++ch)
            delayLines_[ch].prepare(maxDelaySamples);

        for (int v = 0; v < kMaxVoices; ++v)
        {
            lfos_[v].prepare(spec.sampleRate);
            lfos_[v].setWaveform(lfoWaveform_);
            lfos_[v].setFrequency(rate_);
        }

        updateLfoPhases();
        reset();
    }

    /**
     * @brief Processes audio through the chorus effect.
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(kMaxChannels));
        const int nS  = buffer.getNumSamples();

        // Sync atomic params
        T rateVal    = rate_.load(std::memory_order_relaxed);
        T depthVal   = depthMs_.load(std::memory_order_relaxed);
        T mixVal     = mix_.load(std::memory_order_relaxed);
        T fbVal      = feedback_.load(std::memory_order_relaxed);
        T centerVal  = centerDelayMs_.load(std::memory_order_relaxed);
        T spreadVal  = stereoSpread_.load(std::memory_order_relaxed);
        int nVoices  = numVoices_.load(std::memory_order_relaxed);
        bool autoD   = autoDepth_.load(std::memory_order_relaxed);

        mixer_.pushDry(buffer);

        T centerSamples = centerVal * static_cast<T>(spec_.sampleRate) / T(1000);
        T depthSamples  = depthVal * static_cast<T>(spec_.sampleRate) / T(1000);

        // Auto depth coupling: reduce depth as rate increases
        if (autoD && rateVal > T(0.01))
            depthSamples /= std::sqrt(rateVal);

        // sqrt(N) normalization: correct energy preservation
        T voiceNorm = T(1) / std::sqrt(static_cast<T>(nVoices));

        for (int v = 0; v < nVoices; ++v)
            lfos_[v].setFrequency(rateVal);

        for (int i = 0; i < nS; ++i)
        {
            // Cache raw LFO phases before advancing (for per-channel offset)
            T voicePhases[kMaxVoices];
            for (int v = 0; v < nVoices; ++v)
                voicePhases[v] = lfos_[v].getPhase();

            // Advance LFOs (once per sample, shared across channels)
            T voiceLfoBase[kMaxVoices];
            for (int v = 0; v < nVoices; ++v)
                voiceLfoBase[v] = lfos_[v].getNextSample();

            for (int ch = 0; ch < nCh; ++ch)
            {
                T input = buffer.getChannel(ch)[i];
                auto& ring = delayLines_[ch];

                ring.push(input + fbState_[ch] * fbVal);

                T wetRaw = T(0);
                for (int v = 0; v < nVoices; ++v)
                {
                    T lfoVal;
                    if (nCh >= 2 && ch > 0 && spreadVal > T(0))
                    {
                        // Per-channel LFO phase offset for true stereo
                        // decorrelation. M5c fix: previously hardcoded to sine,
                        // which left triangle/saw/square voices with a sine on
                        // ch > 0 — shape mismatch between channels. Evaluate
                        // the currently-selected waveform at the offset phase.
                        T phaseOffset01 = static_cast<T>(ch) * spreadVal
                                        / (T(2) * static_cast<T>(std::max(nCh, 1)));
                        T offPhase = voicePhases[v] + phaseOffset01;
                        offPhase -= std::floor(offPhase);  // wrap to [0, 1)
                        lfoVal = evalLfoAtPhase(lfoWaveform_, offPhase);
                    }
                    else
                    {
                        lfoVal = voiceLfoBase[v];
                    }
                    T delay = std::max(centerSamples + lfoVal * depthSamples, T(1));
                    wetRaw += ring.readInterpolated(delay);
                }

                fbState_[ch] = wetRaw / static_cast<T>(nVoices);
                T wet = wetRaw * voiceNorm;
                buffer.getChannel(ch)[i] = wet;
            }
        }

        mixer_.mixWet(buffer, mixVal);
    }

    /**
     * @brief Processes a single sample on one channel.
     *
     * Advances LFOs and produces the wet-only chorus output.
     * Call once per channel per sample. LFOs advance on channel 0 only.
     *
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Chorused wet output (mix with dry externally).
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        T centerSamples = centerDelayMs_.load(std::memory_order_relaxed) * static_cast<T>(spec_.sampleRate) / T(1000);
        T depthSamples  = depthMs_.load(std::memory_order_relaxed) * static_cast<T>(spec_.sampleRate) / T(1000);
        int nVoices = numVoices_.load(std::memory_order_relaxed);
        T fb = feedback_.load(std::memory_order_relaxed);

        if (channel == 0)
        {
            for (int v = 0; v < nVoices; ++v)
                lfoCache_[v] = lfos_[v].getNextSample();
        }

        auto& ring = delayLines_[channel];
        ring.push(input + fbState_[channel] * fb);

        T wet = T(0);
        for (int v = 0; v < nVoices; ++v)
        {
            T delay = std::max(centerSamples + lfoCache_[v] * depthSamples, T(1));
            wet += ring.readInterpolated(delay);
        }
        wet *= T(1) / std::sqrt(static_cast<T>(nVoices));
        fbState_[channel] = wet;
        return wet;
    }

    /** @brief Resets all internal state. */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            delayLines_[ch].reset();
            fbState_[ch] = T(0);
        }
        for (int v = 0; v < kMaxVoices; ++v)
            lfos_[v].reset();
        mixer_.reset();
        updateLfoPhases();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the LFO rate (modulation speed).
     * @param hz LFO frequency in Hz (typical: 0.1 - 10 Hz).
     */
    void setRate(T hz) noexcept
    {
        rate_.store(std::clamp(hz, T(0.01), T(20)), std::memory_order_relaxed);
    }

    /** @brief Sets the modulation depth. @param amount 0.0–1.0. */
    void setDepth(T amount) noexcept
    {
        depthMs_.store(std::clamp(amount, T(0), T(1)) * T(7), std::memory_order_relaxed);
    }

    /** @brief Sets the dry/wet mix. */
    void setMix(T dryWet) noexcept { mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); }

    /** @brief Sets the number of chorus voices (1–4). */
    void setVoices(int count) noexcept
    {
        numVoices_.store(std::clamp(count, 1, kMaxVoices), std::memory_order_relaxed);
        updateLfoPhases();
    }

    /** @brief Sets the feedback amount (-1 to 1). Negative = flanger. */
    void setFeedback(T amount) noexcept
    {
        feedback_.store(std::clamp(amount, T(-0.99), T(0.99)), std::memory_order_relaxed);
    }

    /** @brief Sets the base delay time in milliseconds. */
    void setCenterDelay(T ms) noexcept
    {
        centerDelayMs_.store(std::clamp(ms, T(0.1), T(30)), std::memory_order_relaxed);
    }

    /** @brief Sets the LFO waveform. */
    void setModWaveform(typename Oscillator<T>::Waveform wf) noexcept
    {
        lfoWaveform_ = wf;
        for (int v = 0; v < kMaxVoices; ++v)
            lfos_[v].setWaveform(wf);
    }

    /** @brief Sets stereo spread (0–1). */
    void setStereoSpread(T amount) noexcept
    {
        stereoSpread_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Enables auto depth coupling.
     *
     * When enabled, depth automatically decreases as rate increases
     * (depthSamples /= sqrt(rate)), preventing excessive modulation
     * at high LFO speeds.
     */
    void setAutoDepth(bool enabled) noexcept
    {
        autoDepth_.store(enabled, std::memory_order_relaxed);
    }

    [[nodiscard]] int getVoices() const noexcept { return numVoices_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }

protected:
    static constexpr int kMaxChannels = 16;

    void updateLfoPhases() noexcept
    {
        int nv = numVoices_.load(std::memory_order_relaxed);
        for (int v = 0; v < nv; ++v)
        {
            T phase = static_cast<T>(v) / static_cast<T>(nv);
            lfos_[v].setPhase(phase);
        }
    }

    // M5c helper: evaluate an LFO waveform at an explicit normalised phase
    // without advancing any state. Aliasing correction (PolyBLEP, integrated
    // triangle) is skipped on purpose — LFO rates (< 20 Hz) are far below the
    // audio band so band-limiting brings no audible benefit and would need
    // per-voice state we don't want to duplicate here.
    static T evalLfoAtPhase(typename Oscillator<T>::Waveform wf, T phase) noexcept
    {
        switch (wf)
        {
            case Oscillator<T>::Waveform::Sine:
                return std::sin(phase * twoPi<T>);
            case Oscillator<T>::Waveform::Saw:
                return T(2) * phase - T(1);
            case Oscillator<T>::Waveform::Square:
                return (phase < T(0.5)) ? T(1) : T(-1);
            case Oscillator<T>::Waveform::Triangle:
                return T(4) * std::abs(phase - T(0.5)) - T(1);
        }
        return T(0);
    }

    AudioSpec spec_ {};

    // Atomic parameters
    std::atomic<T> rate_ { T(1) };
    std::atomic<T> depthMs_ { T(3.5) };
    std::atomic<T> mix_ { T(0.5) };
    std::atomic<T> feedback_ { T(0) };
    std::atomic<T> centerDelayMs_ { T(7) };
    std::atomic<T> stereoSpread_ { T(0.5) };
    std::atomic<int> numVoices_ { 2 };
    std::atomic<bool> autoDepth_ { false };
    typename Oscillator<T>::Waveform lfoWaveform_ = Oscillator<T>::Waveform::Sine;

    // Processing
    std::array<RingBuffer<T>, kMaxChannels> delayLines_ {};
    std::array<Oscillator<T>, kMaxVoices> lfos_ {};
    std::array<T, kMaxChannels> fbState_ {};
    std::array<T, kMaxVoices> lfoCache_ {};
    DryWetMixer<T> mixer_;
};

} // namespace dspark
