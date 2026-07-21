// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Filters.h
 * @brief Multi-mode cascaded biquad filter engine with real-time parameter smoothing.
 *
 * Supports all standard filter shapes (LP, HP, BP, Peak, Shelves, Notch, Tilt,
 * AllPass) with configurable slopes from 6 to 48 dB/oct via cascaded biquad stages.
 * Parameters are smoothed per-sample to prevent zipper noise.
 *
 * @warning Biquad filters are not suited for audio-rate cutoff modulation.
 * Nonlinearity and analog drift update the coefficients at chunk rate
 * (every 16 samples) to maintain CPU stability and prevent severe phase
 * distortion.
 *
 * Dependencies: Core/AudioBuffer.h, Core/AudioSpec.h, Core/Biquad.h,
 * Core/DspMath.h, Core/Smoothers.h, Core/AnalogRandom.h, Core/DenormalGuard.h,
 * Core/StateBlob.h.
 *
 * Threading: prepare(), enableAnalogDrift() and setState() belong to the setup
 * thread; processBlock()/processSample()/reset() to the audio thread. The
 * remaining parameter setters are safe from any thread (relaxed atomics read
 * once per block); non-finite setter values are ignored. The shape helpers
 * (setLowPass()/setShape()/...) write several atomics non-atomically as a
 * group: the audio thread may see the new topology with the previous targets
 * for one block (smoothed, click-free).
 *
 * @code
 *   dspark::FilterEngine<float> filter;
 *   filter.prepare(spec);
 *   filter.setLowPass(2000.0f, 0.707f, 24);  // 2kHz, Butterworth Q, 24dB/oct
 *   filter.processBlock(buffer);
 *
 *   // Real-time parameter changes from UI thread (thread-safe):
 *   filter.setFrequency(4000.0f);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/Smoothers.h"
#include "../Core/AnalogRandom.h"
#include "../Core/DenormalGuard.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class FilterEngine
 * @brief Professional multi-mode filter with cascaded biquad stages.
 *
 * Utilizes SIMD-friendly branching and atomic parameter states for thread-safe
 * real-time manipulation. 
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum number of audio channels allowed.
 */
template <typename T, int MaxChannels = 16>
class FilterEngine
{
public:
    // Removed virtual destructor to avoid vptr overhead and maintain cache alignment.
    ~FilterEngine() = default;

    /**
     * @brief Supported filter shapes.
     */
    enum class Shape
    {
        LowPass, HighPass, BandPass, Peak,
        LowShelf, HighShelf, Notch, AllPass, Tilt
    };

    // -- Lifecycle -----------------------------------------------------------

    /**
     * @brief Initializes the filter engine with the current audio specification.
     * @param spec The audio specification containing sample rate and block size.
     *             An invalid spec (non-positive or NaN fields) is ignored.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        spec_ = spec;
        freqSmoother_.reset(spec.sampleRate, 30.0f, 0.707f, 1000.0f);
        resSmoother_.reset(spec.sampleRate, 20.0f, 0.707f);
        gainSmoother_.reset(spec.sampleRate, 20.0f, 0.0f);
        // Coefficients depend on the sample rate: force a rebuild on the next
        // block. (Without this, re-preparing at a new rate with unchanged
        // parameters kept the OLD rate's coefficients: the cutoff landed at
        // freq * newRate / oldRate.)
        lastFreq_ = -1.0f;
        if (driftEnabled_.load(std::memory_order_relaxed))
            driftGen_.prepare(spec.sampleRate); // re-sync the drift LFO clock
        reset();
    }

    /**
     * @brief Resets the internal state of all cascaded biquads and smoothers.
     * Prevents clicks when relocating playback or enabling the effect.
     */
    void reset() noexcept
    {
        for (auto& stage : stages_) stage.reset();
        freqSmoother_.skip();
        resSmoother_.skip();
        gainSmoother_.skip();
    }

    // -- Configuration -------------------------------------------------------

    /**
     * @brief Configures a low-pass filter.
     * @param freq Target cutoff frequency in Hz.
     * @param Q    Quality factor (0.707 = Butterworth).
     * @param slopeDb Slope in dB/octave (6, 12, 18, 24, 30, 36, 42, 48).
     */
    void setLowPass(float freq, float Q = 0.707f, int slopeDb = 12)
    {
        shape_ = Shape::LowPass;
        slopeDb_ = slopeDb;
        numStages_ = slopeToStages(slopeDb);
        setFrequency(freq);
        setResonance(Q);
    }

    /**
     * @brief Configures a high-pass filter.
     * @param freq Target cutoff frequency in Hz.
     * @param Q    Quality factor (0.707 = Butterworth).
     * @param slopeDb Slope in dB/octave (6, 12, 18, 24, 30, 36, 42, 48).
     */
    void setHighPass(float freq, float Q = 0.707f, int slopeDb = 12)
    {
        shape_ = Shape::HighPass;
        slopeDb_ = slopeDb;
        numStages_ = slopeToStages(slopeDb);
        setFrequency(freq);
        setResonance(Q);
    }

    /**
     * @brief Configures a band-pass filter (fixed 12 dB/oct).
     * @param freq Target center frequency in Hz.
     * @param Q    Quality factor (bandwidth control).
     */
    void setBandPass(float freq, float Q = 0.707f)
    {
        shape_ = Shape::BandPass;
        slopeDb_ = 12;
        numStages_ = 1;
        setFrequency(freq);
        setResonance(Q);
    }

    /**
     * @brief Configures a peaking / bell EQ filter.
     * @param freq Target center frequency in Hz.
     * @param gainDb Gain in dB to boost or cut.
     * @param Q    Quality factor.
     */
    void setPeaking(float freq, float gainDb, float Q = 1.0f)
    {
        shape_ = Shape::Peak;
        slopeDb_ = 12;
        numStages_ = 1;
        setFrequency(freq);
        setResonance(Q);
        setGain(gainDb);
    }

    /**
     * @brief Configures a low-shelf EQ filter.
     * @param freq Target transition frequency in Hz.
     * @param gainDb Gain in dB to boost or cut.
     * @param slope Shelf transition slope (1.0 = standard).
     */
    void setLowShelf(float freq, float gainDb, float slope = 1.0f)
    {
        shape_ = Shape::LowShelf;
        slopeDb_ = 12;
        numStages_ = 1;
        shelfSlope_ = slope;
        setFrequency(freq);
        setGain(gainDb);
    }

    /**
     * @brief Configures a high-shelf EQ filter.
     * @param freq Target transition frequency in Hz.
     * @param gainDb Gain in dB to boost or cut.
     * @param slope Shelf transition slope (1.0 = standard).
     */
    void setHighShelf(float freq, float gainDb, float slope = 1.0f)
    {
        shape_ = Shape::HighShelf;
        slopeDb_ = 12;
        numStages_ = 1;
        shelfSlope_ = slope;
        setFrequency(freq);
        setGain(gainDb);
    }

    /**
     * @brief Configures a notch (band-reject) filter.
     * @param freq Target center frequency in Hz.
     * @param Q    Quality factor (bandwidth of the cut).
     */
    void setNotch(float freq, float Q = 10.0f)
    {
        shape_ = Shape::Notch;
        slopeDb_ = 12;
        numStages_ = 1;
        setFrequency(freq);
        setResonance(Q);
    }

    /**
     * @brief Configures an all-pass filter (shifts phase, flat frequency response).
     * @param freq Phase transition center frequency in Hz.
     * @param Q    Quality factor.
     */
    void setAllPass(float freq, float Q = 0.707f)
    {
        shape_ = Shape::AllPass;
        slopeDb_ = 12;
        numStages_ = 1;
        setFrequency(freq);
        setResonance(Q);
    }

    /**
     * @brief Configures a tilt EQ filter (boost highs/cut lows or vice versa).
     * @param centerFreq Pivot frequency in Hz.
     * @param gainDb Positive = bright, negative = dark.
     */
    void setTilt(float centerFreq, float gainDb)
    {
        shape_ = Shape::Tilt;
        slopeDb_ = 12;
        numStages_ = 1;
        setFrequency(centerFreq);
        setGain(gainDb);
    }

    /**
     * @brief Selects the Orfanidis matched (de-cramped) design for Peak bells.
     *
     * Bilinear bells cramp near Nyquist; the matched design prescribes the
     * analog prototype's Nyquist gain so high bells keep their analog shape.
     * Off by default (bit-compatible with previous output). Thread-safe.
     * Not serialized by getState()/setState(); owners that persist it must
     * re-apply it on restore (Equalizer does).
     */
    void setMatchedPeak(bool enabled) noexcept
    {
        matchedPeak_.store(enabled, std::memory_order_relaxed);
    }

    /** @brief Returns whether Peak bells use the matched (de-cramped) design. */
    [[nodiscard]] bool isMatchedPeak() const noexcept
    {
        return matchedPeak_.load(std::memory_order_relaxed);
    }

    /** @brief Returns the active filter shape. */
    [[nodiscard]] Shape getShape() const noexcept { return shape_.load(std::memory_order_relaxed); }

    /** @brief Returns the active slope in dB/oct (LP/HP only - others = 12). */
    [[nodiscard]] int getSlopeDb() const noexcept { return slopeDb_.load(std::memory_order_relaxed); }

    /** @brief Per-stage Butterworth cascade layout for an LP/HP slope (for analysis). */
    struct CascadeInfo { bool hasFirstOrder = false; int numSecondOrder = 0; float qValues[4] = {}; };

    /**
     * @brief Returns the exact Butterworth cascade (first-order flag + per-stage Q
     * values) used internally for a given LP/HP slope. Lets callers reproduce the
     * true cascade magnitude instead of approximating with a single Q.
     *
     * @param slopeDb Slope in dB/octave (6..48).
     * @param userQ   The engine's resonance parameter: scales the final stage
     *                exactly as updateCoefficients() does (neutral at the
     *                0.707 default, resonant peak above it).
     */
    [[nodiscard]] static CascadeInfo cascadeForSlope(int slopeDb, float userQ = 0.707f) noexcept
    {
        auto c = computeCascade(slopeDb);
        CascadeInfo info;
        info.hasFirstOrder  = c.hasFirstOrder;
        info.numSecondOrder = c.numSecondOrder;
        for (int i = 0; i < c.numSecondOrder && i < 4; ++i) info.qValues[i] = c.qValues[i];
        if (std::isfinite(userQ) && info.numSecondOrder > 0)
            info.qValues[info.numSecondOrder - 1] *= userQ / 0.707f;
        return info;
    }

    /**
     * @brief Switches the filter topology while keeping the current frequency,
     *        resonance, and gain unchanged.
     *
     * Useful for UI dropdowns where the user expects the existing slider values
     * to carry over when they pick a different filter type. For one-shot setup
     * with explicit parameters, the setLowPass()/setPeaking()/etc. helpers are
     * still preferred.
     *
     * @param newShape Target filter shape. Out-of-range values (a wild cast or
     *                  corrupt state blob) are clamped into the enum range.
     * @param slopeDb  Slope in dB/octave (only used by LP/HP, default 12).
     */
    void setShape(Shape newShape, int slopeDb = 12) noexcept
    {
        newShape = static_cast<Shape>(std::clamp(static_cast<int>(newShape), 0,
                                                 static_cast<int>(Shape::Tilt)));
        shape_ = newShape;
        slopeDb_ = slopeDb;
        switch (newShape)
        {
            case Shape::LowPass:
            case Shape::HighPass:
                numStages_ = slopeToStages(slopeDb);
                break;
            default:
                slopeDb_ = 12;
                numStages_ = 1;
                break;
        }
    }

    // -- Real-time parameter changes (Thread-safe) ---------------------------

    /**
     * @brief Sets the target cutoff/center frequency. Thread-safe.
     * @param freq Frequency in Hz. Non-finite values are ignored.
     */
    void setFrequency(float freq) noexcept
    {
        if (!std::isfinite(freq)) return;
        targetFreq_.store(freq, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target resonance/Q. Thread-safe.
     *
     * For LowPass/HighPass the cascade stays an exact Butterworth at the
     * default Q (0.707) and larger values scale the final (most resonant)
     * stage, adding the classic resonant peak at the cutoff. The 6 dB/oct
     * slope is first-order and has no Q.
     *
     * @param Q Quality factor. Non-finite values are ignored.
     */
    void setResonance(float Q)    noexcept
    {
        if (!std::isfinite(Q)) return;
        targetRes_.store(Q, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target gain in dB. Thread-safe.
     * @param dB Gain in decibels. Non-finite values are ignored.
     */
    void setGain(float dB)        noexcept
    {
        if (!std::isfinite(dB)) return;
        targetGain_.store(dB, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the nonlinearity amount. Thread-safe.
     * @param amount 0 = linear (default), 1 = full nonlinearity.
     *               Non-finite values are ignored.
     */
    void setNonlinearity(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        targetNonlinearity_.store(static_cast<float>(std::clamp(amount, T(0), T(1))), std::memory_order_relaxed);
    }

    // -- Analog drift --------------------------------------------------------

    /**
     * @brief Enables analog-style low-frequency modulation of the cutoff.
     * @note Setup thread only (it re-prepares the internal generator, which is
     *       not safe concurrently with processBlock). Call prepare() first so
     *       the generator receives the real sample rate.
     * @param component The analog component profile to simulate.
     * @param intensity Modulation depth (0.0 to 1.0). Non-finite values are
     *                  ignored.
     */
    void enableAnalogDrift(AnalogRandom::AnalogComponent component, float intensity = 0.5f)
    {
        if (!std::isfinite(intensity)) return;
        driftIntensity_.store(intensity, std::memory_order_relaxed);
        driftGen_.setAnalogDefault(component);
        driftGen_.prepare(spec_.sampleRate);
        driftEnabled_.store(true, std::memory_order_release);
    }

    /**
     * @brief Disables analog-style drift modulation. Thread-safe.
     */
    void disableAnalogDrift() noexcept { driftEnabled_.store(false, std::memory_order_relaxed); }

    // -- Processing ----------------------------------------------------------

    /**
     * @brief Processes an entire block of audio data.
     * Internally branches into static, smoothed, or nonlinear processing paths
     * to optimize CPU usage and allow auto-vectorization when possible.
     * 
     * @param buffer View of the audio buffer to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        DenormalGuard guard;
        const int nCh = std::min(buffer.getNumChannels(), MaxChannels);
        const int nS  = buffer.getNumSamples();

        // Update smoothers with latest atomic targets
        freqSmoother_.setTargetValue(targetFreq_.load(std::memory_order_relaxed));
        resSmoother_.setTargetValue(targetRes_.load(std::memory_order_relaxed));
        gainSmoother_.setTargetValue(targetGain_.load(std::memory_order_relaxed));
        T nonLin = static_cast<T>(targetNonlinearity_.load(std::memory_order_relaxed));
        const bool  drift    = driftEnabled_.load(std::memory_order_relaxed);
        const float driftInt = driftIntensity_.load(std::memory_order_relaxed);

        const bool needSmoothing = freqSmoother_.isSmoothing() || resSmoother_.isSmoothing() || gainSmoother_.isSmoothing();
        const bool dynamicPath = needSmoothing || drift || (nonLin > T(0));
        
        constexpr int kChunkSize = 16; // Optimized chunk size for Biquad coefficient updates

        if (!dynamicPath)
        {
            // -- Fast Path: SIMD Friendly, Outer Channel Loop --
            if (nS > 0)
            {
                float f = freqSmoother_.getCurrentValue(); // Value is static
                float q = resSmoother_.getCurrentValue();
                float g = gainSmoother_.getCurrentValue();

                float nyquist = static_cast<float>(spec_.sampleRate) * 0.499f;
                f = std::clamp(f, 10.0f, nyquist);
                q = std::max(q, 0.1f);

                // Skip the trig-heavy coefficient rebuild when absolutely
                // nothing changed since the last block (the common case).
                const Shape sh = shape_.load(std::memory_order_relaxed);
                const int sdb  = slopeDb_.load(std::memory_order_relaxed);
                const bool mp  = matchedPeak_.load(std::memory_order_relaxed);
                if (f != lastFreq_ || q != lastQ_ || g != lastGain_ ||
                    sh != lastShape_ || sdb != lastSlopeDb_ || mp != lastMatched_)
                {
                    updateCoefficients(f, q, g);
                    lastFreq_ = f; lastQ_ = q; lastGain_ = g;
                    lastShape_ = sh; lastSlopeDb_ = sdb; lastMatched_ = mp;
                }

                const int ns = numStages_.load(std::memory_order_relaxed);
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T* channelData = buffer.getChannel(ch);
                    for (int i = 0; i < nS; ++i)
                    {
                        T sample = channelData[i];
                        for (int s = 0; s < ns; ++s)
                            sample = stages_[s].processSample(sample, ch);
                        channelData[i] = sample;
                    }
                }
            }
        }
        else
        {
            // -- Dynamic Path: Modulated / Smoothed (Chunked updates) --
            // Processes audio in small chunks to avoid per-sample trigonometric calculations.
            for (int i = 0; i < nS; ++i)
            {
                float freq = freqSmoother_.getNextValue();
                float res  = resSmoother_.getNextValue();
                float gain = gainSmoother_.getNextValue();

                // Only calculate trig/coefficients every kChunkSize samples to save CPU
                if (i % kChunkSize == 0)
                {
                    float driftValue = drift ? (driftGen_.getNextSample() * driftInt) : 0.0f;
                    freq *= (1.0f + driftValue);

                    if (nonLin > T(0))
                    {
                        // Signal-dependent cutoff modulation ("dielectric" FM,
                        // chunk-rate). Reformulated as a bounded depth control:
                        //   freq *= 1 + depth * 2 * g(level),  g = x/(1+x) in [0,1)
                        // so depth -> 0 is exactly neutral and depth = 1 sweeps
                        // up to one octave on loud material. (The previous
                        // |2-(x+n)/n| form DIVERGED as the knob approached 0.)
                        T avgAbs = T(0);
                        for (int ch = 0; ch < nCh; ++ch)
                            avgAbs += std::abs(buffer.getChannel(ch)[i]);
                        avgAbs /= static_cast<T>(nCh);

                        const T bounded = avgAbs / (T(1) + avgAbs);
                        freq *= static_cast<float>(T(1) + nonLin * T(2) * bounded);
                    }

                    float nyquist = static_cast<float>(spec_.sampleRate) * 0.499f;
                    updateCoefficients(std::clamp(freq, 10.0f, nyquist), std::max(res, 0.1f), gain);
                    lastFreq_ = -1.0f; // invalidate the static-path cache
                }
                else if (drift)
                {
                    (void)driftGen_.getNextSample(); // Advance LFO phase to keep sync
                }

                // Process inner loops
                const int ns = numStages_.load(std::memory_order_relaxed);
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T sample = buffer.getChannel(ch)[i];
                    for (int s = 0; s < ns; ++s)
                        sample = stages_[s].processSample(sample, ch);
                    buffer.getChannel(ch)[i] = sample;
                }
            }
        }
    }

    /**
     * @brief Processes a single sample without parameter smoothing or coefficient updates.
     * @warning Must only be used when parameter changes are managed externally:
     *          call applyParametersNow() after changing parameters, or rely on
     *          processBlock() to consume them (it smooths).
     * @param input Input sample value.
     * @param channel Index of the audio channel being processed. Out-of-range
     *                channels return the input unprocessed (the per-channel
     *                biquad state is bounded to MaxChannels).
     * @return T Processed sample value.
     */
    T processSample(T input, int channel) noexcept
    {
        if (channel < 0 || channel >= MaxChannels) return input;
        T sample = input;
        const int ns = numStages_.load(std::memory_order_relaxed);
        for (int s = 0; s < ns; ++s)
            sample = stages_[s].processSample(sample, channel);
        return sample;
    }

    /**
     * @brief Applies pending parameter targets immediately and rebuilds the
     *        coefficients (no smoothing: values jump).
     *
     * For per-sample workflows where processBlock() never runs, so the atomic
     * parameter targets would otherwise never reach the biquad stages.
     * Owner (audio) thread only, like processSample().
     */
    void applyParametersNow() noexcept
    {
        freqSmoother_.setTargetValue(targetFreq_.load(std::memory_order_relaxed));
        resSmoother_.setTargetValue(targetRes_.load(std::memory_order_relaxed));
        gainSmoother_.setTargetValue(targetGain_.load(std::memory_order_relaxed));
        freqSmoother_.skip();
        resSmoother_.skip();
        gainSmoother_.skip();

        float f = freqSmoother_.getCurrentValue();
        float q = resSmoother_.getCurrentValue();
        float g = gainSmoother_.getCurrentValue();
        const float nyquist = static_cast<float>(spec_.sampleRate) * 0.499f;
        f = std::clamp(f, 10.0f, nyquist);
        q = std::max(q, 0.1f);

        updateCoefficients(f, q, g);
        lastFreq_ = f; lastQ_ = q; lastGain_ = g;
        lastShape_ = shape_.load(std::memory_order_relaxed);
        lastSlopeDb_ = slopeDb_.load(std::memory_order_relaxed);
        lastMatched_ = matchedPeak_.load(std::memory_order_relaxed);
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("FENG"), 1);
        w.write("shape", static_cast<int32_t>(shape_.load(std::memory_order_relaxed)));
        w.write("slope", slopeDb_.load(std::memory_order_relaxed));
        w.write("shelfSlope", shelfSlope_.load(std::memory_order_relaxed));
        w.write("freq", targetFreq_.load(std::memory_order_relaxed));
        w.write("res", targetRes_.load(std::memory_order_relaxed));
        w.write("gain", targetGain_.load(std::memory_order_relaxed));
        w.write("nonlin", static_cast<float>(targetNonlinearity_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("FENG")) return false;
        const auto shape = static_cast<Shape>(r.read("shape", 0));
        const float freq = r.read("freq", 1000.0f);
        const float res = r.read("res", 0.707f);
        const float gain = r.read("gain", 0.0f);
        switch (shape)
        {
            case Shape::LowShelf:
                setLowShelf(freq, gain, r.read("shelfSlope", 1.0f));
                break;
            case Shape::HighShelf:
                setHighShelf(freq, gain, r.read("shelfSlope", 1.0f));
                break;
            case Shape::Peak:
                setPeaking(freq, gain, res);
                break;
            case Shape::Tilt:
                setTilt(freq, gain);
                break;
            default:
                setShape(shape, r.read("slope", 12));
                setFrequency(freq);
                setResonance(res);
                setGain(gain);
                break;
        }
        setNonlinearity(static_cast<T>(r.read("nonlin", 0.0f)));
        return true;
    }

protected:
    static constexpr int kMaxStages = 4;
    static constexpr int kMaxOrder = 8; 

    struct ButterworthCascade
    {
        int order = 0;
        bool hasFirstOrder = false;   
        int numSecondOrder = 0;       
        float qValues[kMaxStages] {}; 
    };

    static ButterworthCascade computeCascade(int slopeDb) noexcept
    {
        static constexpr float qTable[kMaxOrder + 1][kMaxStages] = {
            {}, {}, 
            { 0.7071f }, 
            { 1.0f }, 
            { 0.5412f, 1.3066f }, 
            { 0.6180f, 1.6180f }, 
            { 0.5176f, 0.7071f, 1.9319f }, 
            { 0.5549f, 0.8019f, 2.2470f }, 
            { 0.5098f, 0.6013f, 0.9000f, 2.5628f }  
        };

        ButterworthCascade result {};
        result.order = std::clamp(slopeDb / 6, 1, kMaxOrder);
        result.hasFirstOrder = (result.order % 2 != 0);
        result.numSecondOrder = result.order / 2;
        for (int i = 0; i < result.numSecondOrder; ++i)
            result.qValues[i] = qTable[result.order][i];
        return result;
    }

    static int slopeToStages(int slopeDb) noexcept
    {
        int order = std::clamp(slopeDb / 6, 1, kMaxOrder);
        return (order + 1) / 2;
    }

    void updateCoefficients(float freq, float Q, float gainDb) noexcept
    {
        double sr = spec_.sampleRate;
        double f  = static_cast<double>(freq);

        // Snapshot atomic topology once.
        const Shape sh = shape_.load(std::memory_order_relaxed);
        const int sdb  = slopeDb_.load(std::memory_order_relaxed);
        const double ssl = static_cast<double>(shelfSlope_.load(std::memory_order_relaxed));

        auto cascade = computeCascade(sdb);
        int stageIdx = 0;

        if (cascade.hasFirstOrder)
        {
            BiquadCoeffs<T> c;
            switch (sh)
            {
                case Shape::LowPass:   c = BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f); break;
                case Shape::HighPass:  c = BiquadCoeffs<T>::makeFirstOrderHighPass(sr, f); break;
                default:               c = BiquadCoeffs<T>::makeFirstOrderLowPass(sr, f); break;
            }
            stages_[stageIdx++].setCoeffs(c);
        }

        for (int s = 0; s < cascade.numSecondOrder; ++s)
        {
            float stageQ = cascade.qValues[s];

            if (sh == Shape::Peak || sh == Shape::BandPass ||
                sh == Shape::Notch || sh == Shape::AllPass)
                stageQ = Q;
            else if ((sh == Shape::LowPass || sh == Shape::HighPass) &&
                     s == cascade.numSecondOrder - 1)
            {
                // The user Q scales the FINAL (most resonant) stage of the
                // Butterworth cascade: at the 0.707 default the ratio is
                // exactly 1 (bit-identical Butterworth), larger values add the
                // classic resonant peak at the cutoff. Before this, the table
                // silently overrode the user Q for every LP/HP slope, so
                // setLowPass(f, 5.0f) sounded identical to Butterworth and
                // setResonance() was dead for LP/HP.
                stageQ *= Q / 0.707f;
            }

            BiquadCoeffs<T> c;
            switch (sh)
            {
                case Shape::LowPass:   c = BiquadCoeffs<T>::makeLowPass(sr, f, stageQ);  break;
                case Shape::HighPass:  c = BiquadCoeffs<T>::makeHighPass(sr, f, stageQ); break;
                case Shape::BandPass:  c = BiquadCoeffs<T>::makeBandPass(sr, f, stageQ); break;
                case Shape::Peak:
                    c = matchedPeak_.load(std::memory_order_relaxed)
                        ? BiquadCoeffs<T>::makePeakMatched(sr, f, stageQ, gainDb)
                        : BiquadCoeffs<T>::makePeak(sr, f, stageQ, gainDb);
                    break;
                case Shape::LowShelf:  c = BiquadCoeffs<T>::makeLowShelf(sr, f, gainDb, ssl); break;
                case Shape::HighShelf: c = BiquadCoeffs<T>::makeHighShelf(sr, f, gainDb, ssl); break;
                case Shape::Notch:     c = BiquadCoeffs<T>::makeNotch(sr, f, stageQ);  break;
                case Shape::AllPass:   c = BiquadCoeffs<T>::makeAllPass(sr, f, stageQ); break;
                case Shape::Tilt:      c = BiquadCoeffs<T>::makeTilt(sr, f, gainDb); break;
            }
            stages_[stageIdx++].setCoeffs(c);
        }

        numStages_ = stageIdx;
    }

    AudioSpec spec_ {};
    // Topology state is atomic: setLowPass()/setShape()/... may be called from a
    // control thread while processBlock() reads these on the audio thread.
    // (Reads happen once per block, so atomics impose no hot-path cost.)
    std::atomic<Shape> shape_ { Shape::LowPass };
    std::atomic<int> numStages_ { 1 };
    std::atomic<int> slopeDb_ { 12 };
    std::atomic<float> shelfSlope_ { 1.0f };
    std::atomic<bool> matchedPeak_ { false };

    std::array<Biquad<T, MaxChannels>, kMaxStages> stages_ {};

    Smoothers::StateVariableSmoother freqSmoother_;
    Smoothers::LinearSmoother resSmoother_, gainSmoother_;

    // Atomics for thread-safe UI->Audio communication
    std::atomic<float> targetFreq_{1000.0f};
    std::atomic<float> targetRes_{0.707f};
    std::atomic<float> targetGain_{0.0f};
    std::atomic<float> targetNonlinearity_{0.0f};

    // Atomic: enable/disable may come from a control thread while the audio
    // thread reads them each block (the generator itself is only reconfigured
    // from the setup thread, as enableAnalogDrift documents).
    std::atomic<bool> driftEnabled_ { false };
    std::atomic<float> driftIntensity_ { 0.0f };
    AnalogRandom::Generator<float> driftGen_;

    // Static-path coefficient cache (skip rebuilds when nothing changed).
    float lastFreq_ = -1.0f;
    float lastQ_ = -1.0f;
    float lastGain_ = -1e9f;
    Shape lastShape_ = Shape::LowPass;
    int lastSlopeDb_ = -1;
    bool lastMatched_ = false;
};

} // namespace dspark
