// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

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
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. All setters are lock-free atomic publications,
 * safe from any thread; the LFO rate/waveform/spread are applied on the audio
 * thread at the next block (the oscillators are not thread-safe). Non-finite
 * setter arguments are ignored.
 *
 * Dependencies: Oscillator.h, DryWetMixer.h, AudioSpec.h, AudioBuffer.h,
 *               DenormalGuard.h,
 *               DspMath.h, StateBlob.h.
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
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h" // fastExp, fastTan, fastTanh
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

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

    /** @brief Duration of the right-LFO phase glide after a spread change. */
    static constexpr double kSpreadGlideSeconds = 0.1;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the phaser and allocates internal state blocks.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Hardware audio specifications (sample rate, block size).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        spec_ = spec;
        mixer_.prepare(spec);

        lfo_.prepare(spec.sampleRate);
        // LFO duty: keep the waveforms inside [-1, 1] (no minBLEP overshoot).
        lfo_.setAntiAliasing(Oscillator<T>::AntiAliasing::PolyBLEP);
        lfo_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfo_.setWaveform(lfoWaveform_.load(std::memory_order_relaxed));

        lfoR_.prepare(spec.sampleRate);
        lfoR_.setAntiAliasing(Oscillator<T>::AntiAliasing::PolyBLEP);
        lfoR_.setFrequency(rate_.load(std::memory_order_relaxed));
        lfoR_.setWaveform(lfoWaveform_.load(std::memory_order_relaxed));

        // One-pole parameter smoothing at a 20 Hz corner (~8 ms time constant)
        smoothCoef_ = T(1) - std::exp(static_cast<T>(-2.0 * std::numbers::pi * 20.0) / static_cast<T>(spec_.sampleRate));

        prepared_ = true;
        reset();
    }

    /**
     * @brief Processes an audio block in-place.
     *
     * Any length works: a block longer than the prepared maximum runs in
     * prepared-size slices (the dry/wet mixer holds one prepared block).
     *
     * @param buffer The AudioBufferView containing channels to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return; // without prepare(), fsInv below would be 1/0 -> NaN chain
        // The allpass chain's decaying tails reach denormals within seconds
        // of silence: flush them in hardware for the whole callback.
        DenormalGuard denormalGuard;
        const int total = buffer.getNumSamples();
        const int slice = std::max(1, spec_.maxBlockSize);
        for (int offset = 0; offset < total; offset += slice)
            processChunk(buffer.getSubView(offset, std::min(slice, total - offset)));
    }

protected:
    /** @brief One block of at most the prepared size. */
    void processChunk(AudioBufferView<T> buffer) noexcept
    {

        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        // Front-door non-finite guard: the allpass recursion (yPrev/xPrev)
        // latches a NaN/Inf input permanently -- even with feedback
        // off -- so a single glitch would silence the effect forever. Replace
        // non-finite input with 0 before it reaches the dry snapshot or any
        // allpass/feedback state. No-op on finite input (metrics byte-identical).
        for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
        {
            T* d = buffer.getChannel(ch);
            for (int i = 0; i < nS; ++i)
                if (!std::isfinite(d[i])) d[i] = T(0);
        }

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

        // A spread change moves the right-channel LFO's phase offset on the
        // audio thread (Oscillator state is not thread-safe). The first block
        // after prepare()/reset() starts on the spread; later changes glide
        // the offset linearly over kSpreadGlideSeconds, sample by sample, so
        // the right channel's allpass coefficients never step. (Re-phasing in
        // one go stepped them: a second difference up to 455 times the
        // effect's own on a 60 Hz tone.)
        if (std::abs(spreadVal - lastSpread_) > T(0.0001))
        {
            spreadTarget_ = 0.5 * static_cast<double>(spreadVal);
            if (lastSpread_ < T(0))
            {
                spreadOffset_ = spreadTarget_;
                spreadGlideLeft_ = 0;
                T ph = lfo_.getPhase() + static_cast<T>(spreadOffset_);
                ph -= std::floor(ph);
                lfoR_.setPhase(ph);
            }
            else
            {
                spreadGlideLeft_ = std::max(1, static_cast<int>(kSpreadGlideSeconds * spec_.sampleRate + 0.5));
                spreadStep_ = (spreadTarget_ - spreadOffset_) / spreadGlideLeft_;
            }
            lastSpread_ = spreadVal;
        }
        const bool useSpread = (spreadVal > T(0.0001) || spreadGlideLeft_ > 0 || spreadOffset_ > 0.0)
                            && (nCh >= 2);

        // Stages activated by a live increase start from cleared state: their
        // history is from whenever they were last active (arbitrarily old),
        // and replaying it injects an unrelated step into the chain.
        if (stagesVal > lastStages_)
            for (int s = lastStages_; s < stagesVal; ++s)
            {
                stages_[s].xPrev.fill(T(0));
                stages_[s].yPrev.fill(T(0));
            }
        lastStages_ = stagesVal;

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
            // Convert log-frequency back to linear Hz (fastExp, ~2x std::exp)
            T cutoff = std::min(fastExp(logFreq), nyquist);
            // Bilinear-transform coefficient: tan(pi * f / Fs). fastTan is safe:
            // `cutoff` stays below Nyquist thanks to the clamp above.
            T tanVal = fastTan(kPi * cutoff * fsInv);
            return (tanVal - T(1)) / (tanVal + T(1));
        };

        for (int i = 0; i < nS; ++i)
        {
            // Parameter smoothing (1-pole lowpass) to prevent zipper noise
            currentDepth_ += smoothCoef_ * (targetDepth - currentDepth_);
            currentFb_    += smoothCoef_ * (targetFb - currentFb_);

            if (spreadGlideLeft_ > 0)
            {
                // Lands exactly on the target on the glide's last sample.
                spreadOffset_ = (--spreadGlideLeft_ == 0) ? spreadTarget_ : spreadOffset_ + spreadStep_;
                double ph = static_cast<double>(lfo_.getPhase()) + spreadOffset_;
                ph -= std::floor(ph);
                lfoR_.setPhase(static_cast<T>(ph));
            }

            const T cL = coeffFromLfo(lfo_.getNextSample());
            const T lfoRVal = lfoR_.getNextSample(); // keep phase advancing always
            const T cR = useSpread ? coeffFromLfo(lfoRVal) : cL;

            // Process active channels (odd channels follow the spread LFO)
            for (int ch = 0; ch < nCh; ++ch)
            {
                const T c = (ch & 1) ? cR : cL;
                T sample = buffer.getChannel(ch)[i];

                // Analog-modeled feedback with soft clipping to prevent digital
                // blowups. fastTanh is internally clamped; at typical levels
                // the argument stays below 1, where its error is < 0.05%.
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

public:

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
        lastSpread_ = T(-1); // the next block starts on the spread (no glide)
        spreadOffset_ = 0.0;
        spreadGlideLeft_ = 0;
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

    /**
     * @brief Sets the speed of the phaser sweep.
     * @param hz LFO frequency in Hertz. Range: [0.01, 20.0].
     *           Non-finite values are ignored.
     */
    void setRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        rate_.store(std::clamp(hz, T(0.01), T(20)), std::memory_order_relaxed); // applied on the audio thread
    }

    /**
     * @brief Sets how wide the frequency sweep is.
     * @param amount Range: [0.0 (static), 1.0 (full spectrum)].
     *               Non-finite values are ignored.
     */
    void setDepth(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        depth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Balances the unprocessed and processed signal.
     * @param dryWet Range: [0.0 (fully dry), 1.0 (fully wet)].
     *               Non-finite values are ignored.
     */
    void setMix(T dryWet) noexcept
    {
        if (!std::isfinite(dryWet)) return;
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
     * @param amount Feedback gain. Range: [-0.99, 0.99]. Negative values alter
     *               the vocal formant shape. Non-finite values are ignored.
     */
    void setFeedback(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        feedback_.store(std::clamp(amount, T(-0.99), T(0.99)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the stereo phase spread of the sweep LFO.
     *
     * Offsets the right channel's LFO phase against the left, like classic
     * hardware stereo phasers. 0 = mono-coherent sweep (both channels move
     * together), 1 = fully opposed (180-degree) sweep. A change while
     * running glides the offset over kSpreadGlideSeconds (0.1 s), so the
     * sweep never jumps; after prepare() or reset() the first block starts
     * on the set spread.
     *
     * @param amount Spread amount [0.0, 1.0]. Applied on the audio thread.
     *               Non-finite values are ignored.
     */
    void setStereoSpread(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        stereoSpread_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Defines the midpoint of the logarithmic frequency sweep.
     *
     * Keeps the current sweep RATIO and recenters it geometrically. Callable
     * before prepare() (a 20 kHz ceiling stands in for Nyquist until the rate
     * is known; the per-sample cutoff clamp covers later rate changes anyway).
     *
     * @param hz Center frequency in Hertz (non-positive or non-finite values
     *           are ignored).
     */
    void setCenterFrequency(T hz) noexcept
    {
        if (!std::isfinite(hz) || !(hz > T(0))) return;
        T curMin = minFreq_.load(std::memory_order_relaxed);
        T curMax = maxFreq_.load(std::memory_order_relaxed);
        T halfRange = std::sqrt(curMax / (curMin > T(0) ? curMin : T(1)));

        const T ceiling = (spec_.sampleRate > 0)
            ? static_cast<T>(spec_.sampleRate) * T(0.499) : T(20000);
        const T mn = std::clamp(hz / halfRange, T(20), ceiling - T(1));
        minFreq_.store(mn, std::memory_order_relaxed);
        maxFreq_.store(std::clamp(hz * halfRange, mn + T(1), ceiling),
                       std::memory_order_relaxed);
    }

    /**
     * @brief Explicitly defines the sweep boundaries.
     * @param minHz Lower limit of the allpass cutoff sweep.
     * @param maxHz Upper limit of the allpass cutoff sweep.
     *              Non-finite values are ignored.
     */
    void setFrequencyRange(T minHz, T maxHz) noexcept
    {
        if (!std::isfinite(minHz) || !std::isfinite(maxHz)) return;
        T mn = std::max(minHz, T(20));
        minFreq_.store(mn, std::memory_order_relaxed);
        maxFreq_.store(std::max(maxHz, mn + T(1)), std::memory_order_relaxed);
    }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Changes the geometric shape of the LFO modulation.
     * @param wf Target waveform (e.g., Sine, Triangle; wild enum values clamp).
     */
    void setLfoWaveform(typename Oscillator<T>::Waveform wf) noexcept
    {
        const int w = std::clamp(static_cast<int>(wf), 0,
                                 static_cast<int>(Oscillator<T>::Waveform::Triangle));
        lfoWaveform_.store(static_cast<typename Oscillator<T>::Waveform>(w),
                           std::memory_order_relaxed); // applied on the audio thread
    }

    /** @brief Retrieves the active stage count. @return Number of stages. */
    [[nodiscard]] int getStages() const noexcept { return numStages_.load(std::memory_order_relaxed); }

    /** @brief Retrieves the current sweep rate. @return Rate in Hz. */
    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float: the casts keep the double instantiation
        // compiling (a double matches no StateWriter overload unambiguously).
        StateWriter w(stateId("PHSR"), 1);
        w.write("rate", static_cast<float>(rate_.load(std::memory_order_relaxed)));
        w.write("depth", static_cast<float>(depth_.load(std::memory_order_relaxed)));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("feedback", static_cast<float>(feedback_.load(std::memory_order_relaxed)));
        w.write("minFreq", static_cast<float>(minFreq_.load(std::memory_order_relaxed)));
        w.write("maxFreq", static_cast<float>(maxFreq_.load(std::memory_order_relaxed)));
        w.write("stages", numStages_.load(std::memory_order_relaxed));
        w.write("spread", static_cast<float>(stereoSpread_.load(std::memory_order_relaxed)));
        w.write("waveform",
                static_cast<int32_t>(lfoWaveform_.load(std::memory_order_relaxed)));
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
        setLfoWaveform(static_cast<typename Oscillator<T>::Waveform>(
            r.read("waveform", 0)));
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
    double spreadOffset_ { 0.0 };   ///< Current right-LFO phase offset (cycles).
    double spreadTarget_ { 0.0 };   ///< Offset the glide lands on (cycles).
    double spreadStep_   { 0.0 };   ///< Per-sample offset increment of the glide.
    int spreadGlideLeft_ { 0 };     ///< Samples left in the glide (0 = settled).
    int lastStages_ { kMaxStages }; // audio-thread: clears newly activated stages

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
