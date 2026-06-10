// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file Phaser.h
 * @brief Classic analog-modeled phaser effect using LFO-modulated allpass filter stages.
 *
 * Creates the sweeping "jet" sound by passing the signal through a series of
 * first-order allpass filters whose cutoff frequencies are modulated by an LFO. 
 * Includes an analog-style saturation stage in the feedback loop to emulate
 * classic hardware behavior and prevent harsh digital clipping.
 *
 * Three levels of API complexity are provided:
 * - **Level 1 (Simple):** `setRate()`, `setDepth()`, `setMix()`
 * - **Level 2 (Intermediate):** `setStages()`, `setFeedback()`, `setFrequencyRange()`
 * - **Level 3 (Expert):** `setLfoWaveform()`, `setCenterFrequency()`
 *
 * Dependencies: Biquad.h, Oscillator.h, DryWetMixer.h, AudioSpec.h, AudioBuffer.h, DspMath.h.
 *
 * @code
 *   dspark::Phaser<float> phaser;
 *   phaser.prepare(spec);
 *   phaser.setRate(0.5f);     // 0.5 Hz sweep
 *   phaser.setFeedback(0.7f); // High resonance
 *   phaser.processBlock(buffer);
 * @endcode
 */

#include "../Core/Oscillator.h"
#include "../Core/DryWetMixer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DspMath.h" // Must provide fast_exp and fast_tan
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numbers>

namespace dspark {

/**
 * @class Phaser
 * @brief Zero-latency, highly optimized allpass-based phaser.
 *
 * Employs parameter smoothing to prevent zipper noise and utilizes 
 * fast math approximations for real-time performance.
 *
 * @tparam T Sample type (must satisfy FloatType concept, e.g., float or double).
 */
template <FloatType T>
class Phaser
{
public:
    ~Phaser() = default; // non-virtual: leaf class (no virtual dispatch)

    static constexpr int kMaxStages = 12;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the phaser and allocates internal state blocks.
     * @param spec Hardware audio specifications (sample rate, block size).
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        mixer_.prepare(spec);

        lfo_.prepare(spec.sampleRate);
        lfo_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfo_.setWaveform(lfoWaveform_.load(std::memory_order_relaxed));

        lfoR_.prepare(spec.sampleRate);
        lfoR_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfoR_.setWaveform(lfoWaveform_.load(std::memory_order_relaxed));

        // One-pole parameter smoothing at a 20 Hz corner (~8 ms time constant)
        smoothCoef_ = T(1) - std::exp(static_cast<T>(-2.0 * std::numbers::pi * 20.0) / static_cast<T>(spec_.sampleRate));

        prepared_ = true;
        reset();
    }

    /**
     * @brief Processes an audio block in-place.
     * @param buffer The AudioBufferView containing channels to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return; // without prepare(), fsInv below would be 1/0 -> NaN chain

        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        // Target parameters loaded once per block
        const T targetDepth = depth_.load(std::memory_order_relaxed);
        const T targetMix   = mix_.load(std::memory_order_relaxed);
        const T targetFb    = feedback_.load(std::memory_order_relaxed);
        const int stagesVal = numStages_.load(std::memory_order_relaxed);
        const T minF        = minFreq_.load(std::memory_order_relaxed);
        const T maxF        = maxFreq_.load(std::memory_order_relaxed);
        const T spreadVal   = stereoSpread_.load(std::memory_order_relaxed);

        // Apply LFO rate/waveform on the AUDIO thread (the setters only publish
        // atomics) so the non-atomic Oscillator state is never written concurrently.
        lfo_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfo_.setWaveform(lfoWaveform_.load(std::memory_order_relaxed));
        lfoR_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfoR_.setWaveform(lfoWaveform_.load(std::memory_order_relaxed));

        // Re-phase the right-channel LFO when the spread changes (audio thread
        // only — Oscillator state is not thread-safe).
        if (std::abs(spreadVal - lastSpread_) > T(0.0001))
        {
            lastSpread_ = spreadVal;
            T ph = lfo_.getPhase() + spreadVal * T(0.5);
            ph -= std::floor(ph);
            lfoR_.setPhase(ph);
        }
        const bool useSpread = (spreadVal > T(0.0001)) && (nCh >= 2);

        mixer_.pushDry(buffer);

        const T logMin = std::log(minF);
        const T logMax = std::log(maxF);
        const T fsInv  = T(1) / static_cast<T>(spec_.sampleRate);
        const T nyquist = static_cast<T>(spec_.sampleRate) * T(0.499);
        constexpr T kPi = static_cast<T>(std::numbers::pi);

        auto coeffFromLfo = [&](T lfoVal) noexcept -> T
        {
            T modAmount = (lfoVal + T(1)) * T(0.5) * currentDepth_;
            T logFreq = logMin + modAmount * (logMax - logMin);
            // Convert log-frequency back to linear Hz (fastExp ≈ 2× std::exp)
            T cutoff = std::min(fastExp(logFreq), nyquist);
            // Bilinear-transform coefficient: tan(π·f/Fs). fastTan is safe —
            // `cutoff` stays below Nyquist thanks to the clamp above.
            T tanVal = fastTan(kPi * cutoff * fsInv);
            return (tanVal - T(1)) / (tanVal + T(1));
        };

        for (int i = 0; i < nS; ++i)
        {
            // Parameter smoothing (1-pole lowpass) to prevent zipper noise
            currentDepth_ += smoothCoef_ * (targetDepth - currentDepth_);
            currentFb_    += smoothCoef_ * (targetFb - currentFb_);

            const T cL = coeffFromLfo(lfo_.getNextSample());
            const T lfoRVal = lfoR_.getNextSample(); // keep phase advancing always
            const T cR = useSpread ? coeffFromLfo(lfoRVal) : cL;

            // Process active channels (odd channels follow the spread LFO)
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T c = (ch & 1) ? cR : cL;
                T sample = buffer.getChannel(ch)[i];

                // Analog-modeled feedback with soft clipping to prevent digital
                // blowups. fastTanh: argument bounded by |fb| < 1.
                T fbSignal = fastTanh(fbState_[ch] * currentFb_);
                sample += fbSignal;

                // Series first-order allpass chain
                for (int s = 0; s < stagesVal; ++s)
                {
                    auto& st = stages_[s];
                    T y = c * sample + st.xPrev[ch] - c * st.yPrev[ch];
                    st.xPrev[ch] = sample;
                    st.yPrev[ch] = y;
                    sample = y;
                }

                fbState_[ch] = sample; // 1-sample delay for next iteration
                buffer.getChannel(ch)[i] = sample;
            }
        }

        // DryWetMixer smooths the mix parameter internally.
        mixer_.mixWet(buffer, targetMix);
    }

    /** 
     * @brief Clears internal DSP state and history buffers. 
     * Call this when transport stops or playback jumps.
     */
    void reset() noexcept
    {
        for (auto& stage : stages_)
        {
            stage.xPrev.fill(T(0));
            stage.yPrev.fill(T(0));
        }
        for (auto& fb : fbState_) fb = T(0);

        currentDepth_ = depth_.load(std::memory_order_relaxed);
        currentFb_    = feedback_.load(std::memory_order_relaxed);

        lfo_.reset();
        lfoR_.reset();
        lastSpread_ = T(-1); // force re-phase of the spread LFO on next block
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the speed of the phaser sweep.
     * @param hz LFO frequency in Hertz. Range: [0.01, 20.0].
     */
    void setRate(T hz) noexcept
    {
        hz = std::clamp(hz, T(0.01), T(20));
        rate_.store(hz, std::memory_order_relaxed); // applied on the audio thread
    }

    /**
     * @brief Sets how wide the frequency sweep is.
     * @param amount Range: [0.0 (static), 1.0 (full spectrum)].
     */
    void setDepth(T amount) noexcept
    {
        depth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Balances the unprocessed and processed signal.
     * @param dryWet Range: [0.0 (fully dry), 1.0 (fully wet)].
     */
    void setMix(T dryWet) noexcept 
    { 
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed); 
    }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Configures the intensity/color of the phaser via filter stages.
     * @param count Number of allpass stages. Range: [1, 12]. Evens (2,4,6) are standard.
     */
    void setStages(int count) noexcept
    {
        numStages_.store(std::clamp(count, 1, kMaxStages), std::memory_order_relaxed);
    }

    /**
     * @brief Injects phase-shifted signal back into the input for resonance.
     * @param amount Feedback gain. Range: [-0.99, 0.99]. Negative values alter the vocal formant shape.
     */
    void setFeedback(T amount) noexcept
    {
        feedback_.store(std::clamp(amount, T(-0.99), T(0.99)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the stereo phase spread of the sweep LFO.
     *
     * Offsets the right channel's LFO phase against the left, like classic
     * hardware stereo phasers. 0 = mono-coherent sweep (both channels move
     * together), 1 = fully opposed (180-degree) sweep.
     *
     * @param amount Spread amount [0.0, 1.0]. Applied on the audio thread.
     */
    void setStereoSpread(T amount) noexcept
    {
        stereoSpread_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Defines the midpoint of the logarithmic frequency sweep.
     * @param hz Center frequency in Hertz.
     */
    void setCenterFrequency(T hz) noexcept
    {
        T curMin = minFreq_.load(std::memory_order_relaxed);
        T curMax = maxFreq_.load(std::memory_order_relaxed);
        T halfRange = std::sqrt(curMax / (curMin > T(0) ? curMin : T(1))); 
        
        minFreq_.store(std::max(hz / halfRange, T(20)), std::memory_order_relaxed);
        maxFreq_.store(std::min(hz * halfRange, static_cast<T>(spec_.sampleRate) * T(0.499)), std::memory_order_relaxed);
    }

    /**
     * @brief Explicitly defines the sweep boundaries.
     * @param minHz Lower limit of the allpass cutoff sweep.
     * @param maxHz Upper limit of the allpass cutoff sweep.
     */
    void setFrequencyRange(T minHz, T maxHz) noexcept
    {
        T mn = std::max(minHz, T(20));
        minFreq_.store(mn, std::memory_order_relaxed);
        maxFreq_.store(std::max(maxHz, mn + T(1)), std::memory_order_relaxed);
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Changes the geometric shape of the LFO modulation.
     * @param wf Target waveform (e.g., Sine, Triangle).
     */
    void setLfoWaveform(typename Oscillator<T>::Waveform wf) noexcept
    {
        lfoWaveform_.store(wf, std::memory_order_relaxed); // applied on the audio thread
    }

    /** @brief Retrieves the active stage count. @return Number of stages. */
    [[nodiscard]] int getStages() const noexcept { return numStages_.load(std::memory_order_relaxed); }

    /** @brief Retrieves the current sweep rate. @return Rate in Hz. */
    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("PHSR"), 1);
        w.write("rate", rate_.load(std::memory_order_relaxed));
        w.write("depth", depth_.load(std::memory_order_relaxed));
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("feedback", feedback_.load(std::memory_order_relaxed));
        w.write("minFreq", minFreq_.load(std::memory_order_relaxed));
        w.write("maxFreq", maxFreq_.load(std::memory_order_relaxed));
        w.write("stages", numStages_.load(std::memory_order_relaxed));
        w.write("spread", stereoSpread_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("PHSR")) return false;
        setRate(static_cast<T>(r.read("rate", 0.5f)));
        setDepth(static_cast<T>(r.read("depth", 0.8f)));
        setMix(static_cast<T>(r.read("mix", 0.5f)));
        setFeedback(static_cast<T>(r.read("feedback", 0.0f)));
        setFrequencyRange(static_cast<T>(r.read("minFreq", 200.0f)),
                          static_cast<T>(r.read("maxFreq", 6000.0f)));
        setStages(r.read("stages", 4));
        setStereoSpread(static_cast<T>(r.read("spread", 0.0f)));
        return true;
    }

protected:
    static constexpr int kMaxChannels = 16;

    AudioSpec spec_ {};

    // Parameters
    std::atomic<T> rate_     { T(0.5) };
    std::atomic<T> depth_    { T(0.8) };
    std::atomic<T> mix_      { T(0.5) };
    std::atomic<T> feedback_ { T(0) };
    std::atomic<T> minFreq_  { T(200) };
    std::atomic<T> maxFreq_  { T(6000) };
    std::atomic<int> numStages_ { 4 };
    std::atomic<typename Oscillator<T>::Waveform> lfoWaveform_ { Oscillator<T>::Waveform::Sine };
    std::atomic<T> stereoSpread_ { T(0) };

    // Smoothed state parameters
    bool prepared_  { false };
    T currentDepth_ { T(0.8) };
    T currentFb_    { T(0) };
    T smoothCoef_   { T(0.01) };
    T lastSpread_   { T(-1) };

    struct FirstOrderAllpass
    {
        std::array<T, kMaxChannels> xPrev {};
        std::array<T, kMaxChannels> yPrev {};
    };

    // Processing state
    std::array<FirstOrderAllpass, kMaxStages> stages_ {};
    std::array<T, kMaxChannels> fbState_ {};
    Oscillator<T> lfo_;
    Oscillator<T> lfoR_;   // right-channel LFO for stereo spread
    DryWetMixer<T> mixer_;
};

} // namespace dspark
