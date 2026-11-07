// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Clipper.h
 * @brief Multi-mode clipping processor with oversampling, multi-stage architecture, and time-aware slew limiting.
 *
 * Zero-allocation in the audio thread and entirely lock-free. Four clipping
 * modes are supported:
 * - **Hard**: Digital clamp. Transparent until the ceiling is hit, then applies a brickwall limit.
 * - **Soft**: Hyperbolic tangent (tanh). Smooth mathematical saturation with no hard discontinuities.
 * - **Analog**: Sine-based waveshaping. Models the soft clipping characteristics of magnetic/transformer saturation.
 * - **GoldenRatio**: True soft-knee using golden ratio thresholds. Operates linearly up to `ceiling/phi`,
 *   then transitions using a smooth rational asymptotic curve to the ceiling.
 *
 * Threading model:
 *  - prepare(): setup thread only (allocates when oversampling is enabled);
 *    never concurrently with processBlock(). Invalid specs are ignored and an
 *    unprepared instance passes audio through untouched.
 *  - Parameter setters/getters (setMode, setCeiling, ...): std::atomic based,
 *    safe from any thread during playback. Non-finite values are ignored.
 *    Input gain and ceiling glide linearly over 20 ms at the processing rate
 *    (they used to step once per block, zippering under automation); values
 *    set before prepare() apply from the first sample.
 *  - getState()/setState(): setup/UI threads (getState allocates).
 *  - getGainReductionDb(): metering-style read, safe from any thread.
 *
 * Antialiasing (setAntialiasing(true), off by default): every mode then runs
 * second-order antiderivative antialiasing (Bilbao, Esqueda, Parker and
 * Valimaki 2017; kernels in Core/detail/Adaa.h), per stage. Measured over
 * tones of 1-19.25 kHz at -6 and -18 dBFS (worst non-harmonic component
 * below 20 kHz relative to the tone, 48 kHz): hard clip at 2x and +12 dB
 * -46.1 dBc against -27.0 dBc without it; at 2x and +24 dB -40.5 against
 * -14.5; at 4x and +24 dB -60.6 against -21.0. Each stage delays by one
 * processing-rate sample; the wet path is padded to a whole number of base
 * samples, independent of the stage count, so getLatency() grows by
 * ceil(4 / factor) base samples (4 at 1x, 2 at 2x, 1 from 4x) and the dry
 * path is delayed to match. The kernel attenuates a linear signal by
 * (1 + 2 cos w) / 3 at the processing rate: -1.3 dB at 10 kHz and -5.9 dB at
 * 20 kHz at 2x, -1.3 dB at 20 kHz at 4x, so use it with oversampling (at 1x
 * it is -12 dB at 20 kHz). Switching it changes the latency (notify the
 * host) and steps the output once.
 *
 * Channel handling: without oversampling up to 16 channels are clipped; the
 * dry/wet blend and the oversampled path cover only the prepared channel
 * count, so keep the view within the prepared spec for consistent behaviour.
 * Blocks longer than the prepared maximum are processed in pieces.
 *
 * @note The ceiling is enforced sample-exactly in the processing (high) rate.
 * With oversampling, the downsampling filter can reconstruct small
 * inter-sample overshoots above the ceiling; follow with a true-peak limiter
 * when broadcast compliance is required.
 *
 * Dependencies: Core/AudioBuffer.h, Core/AudioSpec.h, Core/DspMath.h,
 * Core/DryWetMixer.h, Core/Oversampling.h, Core/SmoothedValue.h, Core/StateBlob.h,
 * Core/detail/ClipperShape.h, Core/detail/Adaa.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/DryWetMixer.h"
#include "../Core/Oversampling.h"
#include "../Core/SmoothedValue.h"
#include "../Core/StateBlob.h"
#include "../Core/detail/Adaa.h"
#include "../Core/detail/ClipperShape.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class Clipper
 * @brief Real-time audio clipper with analog modeling and anti-aliasing features.
 *
 * @tparam T Sample type (must satisfy dspark::FloatType, typically float or double).
 */
template <FloatType T>
class Clipper
{
public:
    /** @brief Defines the harmonic waveshaping algorithm used for clipping. */
    enum class Mode
    {
        Hard,         ///< Brickwall digital clipping. High odd harmonics.
        Soft,         ///< Tanh soft clipping. Even/odd blend, tape-like.
        Analog,       ///< Sine-based soft clipping. Transformer-like saturation.
        GoldenRatio   ///< Mathematical soft-knee using phi. Extremely transparent until heavy drive.
    };

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the clipper for processing, allocating any necessary internal buffers.
     *
     * If oversampling is configured > 1x, this method allocates the oversampling filters.
     * Call it from the main/setup thread before playback begins (never concurrently
     * with processBlock), and again whenever the sample rate changes.
     *
     * Invalid specs (non-positive/non-finite rate, block size or channel count)
     * are ignored: the previous state is kept and an unprepared instance stays
     * pass-through.
     *
     * @param spec The current audio environment specifications (sample rate, max block size).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;

        prepared_.store(false, std::memory_order_relaxed);
        spec_ = spec;
        mixer_.prepare(spec);

        const int osFactor = osFactor_.load(std::memory_order_relaxed);

        if (osFactor > 1) {
            oversampler_ = std::make_unique<Oversampling<T>>(
                osFactor, Oversampling<T>::Quality::High);
            oversampler_->prepare(spec);
        } else {
            oversampler_.reset();
        }

        // Align the dry copy with the latent (oversampled) wet path so that a
        // dry/wet blend below 100% does not comb-filter. The antialiasing
        // delay is added on top by delaying the dry input (pushDryAligned).
        mixer_.setLatencyCompensation(oversampler_ ? oversampler_->getLatency() : 0);
        processFactor_ = oversampler_ ? oversampler_->getFactor() : 1;
        dryScratch_.resize(spec.numChannels, spec.maxBlockSize);
        resetAntialiasState();

        // Parameter glides run at the processing (possibly oversampled) rate.
        const double procRate = spec.sampleRate * (oversampler_ ? oversampler_->getFactor() : 1);
        for (auto* sm : { &stageGainSm_, &ceilingSm_ })
        {
            sm->prepare(procRate, kParamRampMs);
            sm->setSmoothingType(SmoothedValue<T>::SmoothingType::Linear);
        }
        snapParameters();

        for (int ch = 0; ch < kMaxChannels; ++ch)
            slewPrev_[ch] = T(0);

        prepared_.store(true, std::memory_order_relaxed);
    }

    /**
     * @brief Resets the internal state of the clipper.
     *
     * Clears history buffers, slew limiter states, and oversampling memory.
     * Useful when stopping/starting transport or flushing tails.
     */
    void reset() noexcept
    {
        mixer_.reset();
        if (oversampler_) oversampler_->reset();
        for (int ch = 0; ch < kMaxChannels; ++ch)
            slewPrev_[ch] = T(0);
        snapParameters();
        resetAntialiasState();
        gainReductionDb_.store(T(0), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------

    /**
     * @brief Processes an audio buffer in-place through the clipping algorithm.
     *
     * Handles dry/wet mixing, upsampling, processing, and downsampling
     * transparently. Pass-through until prepare() succeeds.
     *
     * @param buffer View of the audio data to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        if (buffer.getNumSamples() == 0 || buffer.getNumChannels() == 0) return;

        // The oversampler and the dry mixer hold one maximum block: longer
        // blocks used to leave their tail unclipped.
        const int total = buffer.getNumSamples();
        const int maxBlock = std::max(1, spec_.maxBlockSize);
        for (int off = 0; off < total; off += maxBlock)
            processChunk(buffer.getSubView(off, std::min(maxBlock, total - off)));
    }

private:
    void processChunk(AudioBufferView<T> buffer) noexcept
    {
        T mixVal = mix_.load(std::memory_order_relaxed);
        const bool adaa = antialias_.load(std::memory_order_relaxed);
        if (adaa != adaaActive_)
        {
            // Start each stage from its current input rather than from
            // stale history; the latency change itself steps the output.
            adaaActive_ = adaa;
            adaaFresh_ = true;
        }
        pushDryAligned(buffer, adaa ? adaaBaseLatency() : 0);

        if (oversampler_ && oversampler_->getFactor() > 1)
        {
            auto upView = oversampler_->upsample(buffer);
            processInternal(upView, spec_.sampleRate * oversampler_->getFactor());
            oversampler_->downsample(buffer);
        }
        else
        {
            processInternal(buffer, spec_.sampleRate);
        }

        mixer_.mixWet(buffer, mixVal);
    }

public:

    /**
     * @brief Sets the clipping algorithm.
     * @param mode The desired waveshaping Mode (out-of-range values are clamped).
     */
    void setMode(Mode mode) noexcept
    {
        const int m = std::clamp(static_cast<int>(mode), 0,
                                 static_cast<int>(Mode::GoldenRatio));
        mode_.store(static_cast<Mode>(m), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the absolute maximum output level.
     * @param dB Ceiling level in decibels Full Scale (dBFS). Clamped between -60 and 0;
     *           non-finite values are ignored.
     */
    void setCeiling(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        ceilingDb_.store(std::clamp(dB, T(-60), T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the input drive/gain before clipping.
     * @param dB Gain applied to the input signal in dB. Clamped between 0 and 48;
     *           non-finite values are ignored.
     */
    void setInputGain(T dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        inputGainDb_.store(std::clamp(dB, T(0), T(48)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the number of cascaded clipping stages.
     *
     * Multi-stage clipping distributes the input gain logarithmically across multiple
     * clipping algorithms. This alters the harmonic profile, making heavy distortion
     * sound smoother compared to a single aggressive stage.
     *
     * @param count Number of stages. Clamped between 1 and 4.
     */
    void setStages(int count) noexcept
    {
        stages_.store(std::clamp(count, 1, kMaxStages), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the dry/wet ratio of the processor.
     * @param amount Linear ratio where 0.0 is entirely dry and 1.0 is fully processed.
     *               Non-finite values are ignored.
     */
    void setMix(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        mix_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Enables slew limiting to soften clipping edges.
     *
     * Limits how fast the output can change over time. Defined in milliseconds,
     * so the effect stays consistent regardless of the sample rate or
     * oversampling factor. Note that a slew limiter is itself nonlinear: it
     * darkens the transient edges and top octave rather than strictly removing
     * aliasing (use oversampling for that).
     *
     * @param ms Rise time in milliseconds to reach the ceiling from zero.
     *           0 = off; non-finite values are ignored.
     */
    void setSlewLimit(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        slewLimitMs_.store(std::max(ms, T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the oversampling multiplier to mitigate aliasing.
     *
     * Values are rounded UP to the next power of two and capped at 16.
     * @note Changing this requires a subsequent call to `prepare()` to allocate filters.
     *
     * @param factor Oversampling ratio (1 = off, 2, 4, 8, 16).
     */
    void setOversampling(int factor) noexcept
    {
        factor = std::bit_ceil(static_cast<unsigned int>(std::max(1, factor)));
        osFactor_.store(std::min(factor, 16), std::memory_order_relaxed);
    }

    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getCeiling() const noexcept { return ceilingDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getInputGain() const noexcept { return inputGainDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] int getStages() const noexcept { return stages_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSlewLimit() const noexcept { return slewLimitMs_.load(std::memory_order_relaxed); }

    /** @brief Returns the published oversampling factor (applied on the next prepare()). */
    [[nodiscard]] int getOversampling() const noexcept { return osFactor_.load(std::memory_order_relaxed); }

    /**
     * @brief Enables second-order antiderivative antialiasing on every mode.
     *
     * Off by default. Any thread; applied from the next block. Adds
     * ceil(4 / oversampling factor) base samples of latency (reported by
     * getLatency() immediately) and the passband droop described in the file
     * documentation, so pair it with oversampling.
     */
    void setAntialiasing(bool on) noexcept { antialias_.store(on, std::memory_order_relaxed); }

    /** @brief Whether second-order antialiasing is enabled. */
    [[nodiscard]] bool getAntialiasing() const noexcept { return antialias_.load(std::memory_order_relaxed); }

    /** @brief Latency in samples: oversampling filters plus, when enabled,
     *  the antialiasing delay. */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (oversampler_ ? oversampler_->getLatency() : 0)
             + (antialias_.load(std::memory_order_relaxed) ? adaaBaseLatency() : 0);
    }

    /** @brief Retrieves the maximum gain reduction applied during the last block (for UI metering). */
    [[nodiscard]] T getGainReductionDb() const noexcept { return gainReductionDb_.load(std::memory_order_relaxed); }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("CLIP"), 1);
        w.write("mode", static_cast<int32_t>(mode_.load(std::memory_order_relaxed)));
        w.write("ceiling", static_cast<float>(ceilingDb_.load(std::memory_order_relaxed)));
        w.write("inputGain", static_cast<float>(inputGainDb_.load(std::memory_order_relaxed)));
        w.write("stages", stages_.load(std::memory_order_relaxed));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        w.write("slewLimit", static_cast<float>(slewLimitMs_.load(std::memory_order_relaxed)));
        w.write("oversampling", osFactor_.load(std::memory_order_relaxed));
        w.write("antialias", antialias_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob. Oversampling factor applies on
     *  the next prepare() as usual. */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("CLIP")) return false;
        setMode(static_cast<Mode>(r.read("mode", 0)));
        setCeiling(static_cast<T>(r.read("ceiling", 0.0f)));
        setInputGain(static_cast<T>(r.read("inputGain", 0.0f)));
        setStages(r.read("stages", 1));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setSlewLimit(static_cast<T>(r.read("slewLimit", 0.0f)));
        setOversampling(r.read("oversampling", 1));
        setAntialiasing(r.read("antialias", false));
        return true;
    }

protected:
    static constexpr int kMaxStages = 4;
    static constexpr int kMaxChannels = 16;

    /// Mathematical Golden Ratio used for the GoldenRatio soft-knee transition.
    static constexpr T kPhi = static_cast<T>(1.6180339887498948482);

    /**
     * @brief Core DSP routing. Resolves atomics and branches to the per-mode template.
     */
    void processInternal(AudioBufferView<T>& buffer, double currentSampleRate) noexcept
    {
        Mode modeVal   = mode_.load(std::memory_order_relaxed);
        T ceilDb       = ceilingDb_.load(std::memory_order_relaxed);
        T gainDb       = inputGainDb_.load(std::memory_order_relaxed);
        int numStages  = stages_.load(std::memory_order_relaxed);
        T slewMs       = slewLimitMs_.load(std::memory_order_relaxed);

        if (numStages != lastStages_)
        {
            // Keep the running TOTAL gain continuous: the per-stage gain of
            // the old count, applied numStages times, jumped (1 -> 4 stages
            // at +18 dB drove a -20 dBFS tone to the ceiling for 20 ms).
            const T total = std::pow(stageGainSm_.getCurrentValue(), static_cast<T>(lastStages_));
            stageGainSm_.reset(stageGainFor(total, numStages));
            lastStages_ = numStages;
        }
        ceilingSm_.setTargetValue(dbToLinear(ceilDb));
        stageGainSm_.setTargetValue(stageGainFor(dbToLinear(gainDb), numStages));

        // Maximum slew per sample is ceiling * slewScale (time-aware).
        const T slewScale = (slewMs > T(0))
            ? T(1) / (slewMs * T(0.001)) / static_cast<T>(currentSampleRate)
            : T(0);

        // The mode switch is resolved once per block, outside the sample loop.
        if (adaaActive_)
        {
            // Cached antiderivative values belong to one curve.
            if (modeVal != adaaMode_)
            {
                adaaMode_ = modeVal;
                for (auto& chStages : adaaStages_)
                    for (auto& st : chStages) st.state.invalidate();
            }
            switch (modeVal)
            {
                case Mode::Soft:        dispatchAntialiased<detail::AdaaTanh>(buffer, numStages, slewScale); break;
                case Mode::Analog:      dispatchAntialiased<detail::AdaaSinePolynomial>(buffer, numStages, slewScale); break;
                case Mode::GoldenRatio: dispatchAntialiased<detail::AdaaRationalKnee>(buffer, numStages, slewScale); break;
                default:                dispatchAntialiased<detail::AdaaHardClip>(buffer, numStages, slewScale); break;
            }
            return;
        }
        switch (modeVal)
        {
            case Mode::Hard:        dispatchClipping<Mode::Hard>(buffer, numStages, slewScale); break;
            case Mode::Soft:        dispatchClipping<Mode::Soft>(buffer, numStages, slewScale); break;
            case Mode::Analog:      dispatchClipping<Mode::Analog>(buffer, numStages, slewScale); break;
            case Mode::GoldenRatio: dispatchClipping<Mode::GoldenRatio>(buffer, numStages, slewScale); break;
            default:                dispatchClipping<Mode::Hard>(buffer, numStages, slewScale); break;
        }
    }

    /** Antialiasing delay in base samples: kMaxStages processing samples,
     *  rounded up to whole base samples. */
    [[nodiscard]] int adaaBaseLatency() const noexcept
    {
        const int factor = std::max(1, processFactor_);
        return (kMaxStages + factor - 1) / factor;
    }

    void resetAntialiasState() noexcept
    {
        for (auto& chStages : adaaStages_)
            for (auto& st : chStages) st = {};
        for (auto& ring : padRing_) ring.fill(0.0);
        for (auto& ring : dryRing_) ring.fill(T(0));
        padPos_ = 0;
        dryPos_ = 0;
        adaaFresh_ = true;
        adaaStageCount_ = stages_.load(std::memory_order_relaxed);
    }

    /** Captures the dry signal, delayed by @p extra base samples when the
     *  antialiasing delay is active. The short per-channel ring is updated
     *  either way, so enabling it starts from real history. */
    void pushDryAligned(AudioBufferView<T> buffer, int extra) noexcept
    {
        const int nCh = std::min({buffer.getNumChannels(), dryScratch_.getNumChannels(), kMaxChannels});
        const int nS = std::min(buffer.getNumSamples(), dryScratch_.getNumSamples());
        const int len = adaaBaseLatency();
        int pos = dryPos_;
        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* in = buffer.getChannel(ch);
            T* out = dryScratch_.getChannel(ch);
            auto& ring = dryRing_[static_cast<size_t>(ch)];
            pos = dryPos_;
            for (int i = 0; i < nS; ++i)
            {
                out[i] = ring[static_cast<size_t>(pos)];   // input len samples ago
                ring[static_cast<size_t>(pos)] = in[i];
                pos = pos + 1 == len ? 0 : pos + 1;
            }
        }
        dryPos_ = pos;
        // The mixer blends only the prepared channels, so the scratch view
        // (prepared channels) is enough whatever the buffer's width.
        if (extra > 0 && nS == buffer.getNumSamples())
            mixer_.pushDry(dryScratch_.toView().getSubView(0, nS));
        else
            mixer_.pushDry(buffer);
    }

    /**
     * @brief Second-order antialiased per-mode loop. Works in units of the
     * ceiling: each stage maps u -> curve(u) with u = stage gain times the
     * previous stage's output. Every stage delays by one processing sample;
     * a pad brings the total to factor * adaaBaseLatency() samples for any
     * stage count.
     */
    template <class Curve>
    void dispatchAntialiased(AudioBufferView<T>& buffer, int numStages, T slewScale) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();
        const int totalDelay = std::max(1, processFactor_) * adaaBaseLatency();
        const int pad = totalDelay - numStages;

        T peakIn  = T(0);
        T peakOut = T(0);

        if (adaaFresh_)
        {
            // Hold the first input as the whole history of every stage.
            adaaFresh_ = false;
            adaaStageCount_ = numStages;
            if (nS > 0)
            {
                const double c0 = static_cast<double>(ceilingSm_.getCurrentValue());
                const double g0 = static_cast<double>(stageGainSm_.getCurrentValue());
                for (int ch = 0; ch < nCh; ++ch)
                {
                    double u = static_cast<double>(buffer.getChannel(ch)[0]) / c0;
                    for (int s = 0; s < kMaxStages; ++s)
                    {
                        u *= g0;
                        auto& st = adaaStages_[static_cast<size_t>(ch)][static_cast<size_t>(s)];
                        st.state.reset(u);
                        u = Curve::d2(u);
                        st.out1 = st.out2 = u;
                    }
                }
            }
        }
        else if (numStages > adaaStageCount_)
        {
            // Newly active stages continue from the previous stage's recent
            // output, as if they had been running.
            const double g0 = static_cast<double>(stageGainSm_.getCurrentValue());
            for (int ch = 0; ch < nCh; ++ch)
                for (int s = adaaStageCount_; s < numStages; ++s)
                {
                    auto& prev = adaaStages_[static_cast<size_t>(ch)][static_cast<size_t>(s - 1)];
                    auto& st = adaaStages_[static_cast<size_t>(ch)][static_cast<size_t>(s)];
                    st.state.reset(g0 * prev.out2);
                    st.state.track(g0 * prev.out1);
                    st.out1 = st.out2 = prev.out1;
                }
        }
        adaaStageCount_ = numStages;

        int writePos = padPos_;
        for (int start = 0; start < nS; start += kChunk)
        {
            const int n = std::min(kChunk, nS - start);
            const bool gliding = stageGainSm_.isSmoothing() || ceilingSm_.isSmoothing();
            if (gliding)
                for (int j = 0; j < n; ++j)
                {
                    gainRamp_[static_cast<size_t>(j)] = stageGainSm_.getNextValue();
                    ceilRamp_[static_cast<size_t>(j)] = ceilingSm_.getNextValue();
                }
            const T gainConst = stageGainSm_.getCurrentValue();
            const T ceilConst = ceilingSm_.getCurrentValue();

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch) + start;
                auto& stages = adaaStages_[static_cast<size_t>(ch)];
                auto& ring = padRing_[static_cast<size_t>(ch)];
                int pos = writePos;
                for (int j = 0; j < n; ++j)
                {
                    const T stageGain = gliding ? gainRamp_[static_cast<size_t>(j)] : gainConst;
                    const T ceiling   = gliding ? ceilRamp_[static_cast<size_t>(j)] : ceilConst;
                    const double g = static_cast<double>(stageGain);
                    const double c = static_cast<double>(ceiling);
                    double u = static_cast<double>(data[j]) / c;

                    T totalGain = stageGain;
                    for (int s = 1; s < numStages; ++s) totalGain *= stageGain;
                    const T driven = data[j] * totalGain;

                    for (int s = 0; s < numStages; ++s)
                    {
                        auto& st = stages[static_cast<size_t>(s)];
                        u = st.state.template process<Curve>(u * g);
                        st.out2 = st.out1;
                        st.out1 = u;
                    }

                    // Pad to the fixed total delay (ring of kPadRing samples).
                    ring[static_cast<size_t>(pos)] = u;
                    const double delayed = ring[static_cast<size_t>((pos - pad) & (kPadRing - 1))];
                    pos = (pos + 1) & (kPadRing - 1);

                    T sample = static_cast<T>(delayed * c);
                    const T maxSlewDelta = ceiling * slewScale;
                    if (maxSlewDelta > T(0))
                    {
                        const T delta = sample - slewPrev_[ch];
                        if (std::abs(delta) > maxSlewDelta)
                            sample = slewPrev_[ch] + std::copysign(maxSlewDelta, delta);
                    }
                    slewPrev_[ch] = sample;

                    peakIn  = std::max(peakIn, std::abs(driven));
                    peakOut = std::max(peakOut, std::abs(sample));
                    data[j] = sample;
                }
                if (ch == nCh - 1) writePos = pos;
            }
            if (nCh == 0) break;
        }
        padPos_ = writePos;

        if (peakIn > T(1e-6)) {
            auto ratio = std::min(peakOut / peakIn, T(1));
            gainReductionDb_.store(gainToDecibels(ratio, T(-100)), std::memory_order_relaxed);
        } else {
            gainReductionDb_.store(T(0), std::memory_order_relaxed);
        }
    }

    /** Gain per stage: the total split logarithmically across the cascade. */
    [[nodiscard]] static T stageGainFor(T totalGainLin, int numStages) noexcept
    {
        return (numStages > 1) ? std::pow(totalGainLin, T(1) / static_cast<T>(numStages))
                               : totalGainLin;
    }

    /** Jumps both parameter glides to the current targets (prepare/reset). */
    void snapParameters() noexcept
    {
        lastStages_ = stages_.load(std::memory_order_relaxed);
        ceilingSm_.reset(dbToLinear(ceilingDb_.load(std::memory_order_relaxed)));
        stageGainSm_.reset(stageGainFor(dbToLinear(inputGainDb_.load(std::memory_order_relaxed)),
                                        stages_.load(std::memory_order_relaxed)));
    }

    /**
     * @brief Per-mode processing loop (waveshaper inlined at compile time; the
     * slew state and peak metering keep the loop serial per channel).
     */
    template <Mode M>
    void dispatchClipping(AudioBufferView<T>& buffer, int numStages, T slewScale) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS  = buffer.getNumSamples();

        T peakIn  = T(0);
        T peakOut = T(0);

        // Chunks: the parameter glides are computed once per sample and shared
        // by every channel; settled, both are plain constants.
        for (int start = 0; start < nS; start += kChunk)
        {
            const int n = std::min(kChunk, nS - start);
            const bool gliding = stageGainSm_.isSmoothing() || ceilingSm_.isSmoothing();
            if (gliding)
                for (int j = 0; j < n; ++j)
                {
                    gainRamp_[static_cast<size_t>(j)] = stageGainSm_.getNextValue();
                    ceilRamp_[static_cast<size_t>(j)] = ceilingSm_.getNextValue();
                }
            const T gainConst = stageGainSm_.getCurrentValue();
            const T ceilConst = ceilingSm_.getCurrentValue();

            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch) + start;
                for (int j = 0; j < n; ++j)
                {
                    const T stageGain = gliding ? gainRamp_[static_cast<size_t>(j)] : gainConst;
                    const T ceiling   = gliding ? ceilRamp_[static_cast<size_t>(j)] : ceilConst;
                    T sample = data[j];
                    T totalGain = stageGain;
                    for (int s = 1; s < numStages; ++s) totalGain *= stageGain;
                    const T driven = sample * totalGain;

                    // Compiler will unroll this loop for small bounded N
                    for (int s = 0; s < numStages; ++s)
                    {
                        sample *= stageGain;
                        sample = processSample<M>(sample, ceiling);
                    }

                    // Slew Limiter processing (branch highly predictable if static)
                    const T maxSlewDelta = ceiling * slewScale;
                    if (maxSlewDelta > T(0))
                    {
                        T delta = sample - slewPrev_[ch];
                        if (std::abs(delta) > maxSlewDelta)
                            sample = slewPrev_[ch] + std::copysign(maxSlewDelta, delta);
                    }

                    slewPrev_[ch] = sample;

                    peakIn  = std::max(peakIn, std::abs(driven));
                    peakOut = std::max(peakOut, std::abs(sample));
                    data[j] = sample;
                }
            }
        }

        // Safe gain reduction calculation
        if (peakIn > T(1e-6)) {
            auto ratio = std::min(peakOut / peakIn, T(1));
            gainReductionDb_.store(gainToDecibels(ratio, T(-100)), std::memory_order_relaxed);
        } else {
            gainReductionDb_.store(T(0), std::memory_order_relaxed);
        }
    }

    /**
     * @brief Compile-time resolution of the waveshaping math.
     */
    template <Mode M>
    [[nodiscard]] static inline T processSample(T sample, T ceiling) noexcept
    {
        if constexpr (M == Mode::Hard)
            return detail::clipperShape<detail::ClipperCurve::Hard>(sample, ceiling);
        else if constexpr (M == Mode::Soft)
            return detail::clipperShape<detail::ClipperCurve::Tanh>(sample, ceiling);
        else if constexpr (M == Mode::Analog)
            return detail::clipperShape<detail::ClipperCurve::Sine>(sample, ceiling);
        else // Mode::GoldenRatio (the mode switch instantiates no other value)
            return detail::clipperShape<detail::ClipperCurve::GoldenRatio>(sample, ceiling);
    }

    // Mathematical utility helpers
    [[nodiscard]] static T dbToLinear(T dB) noexcept { return std::pow(T(10), dB / T(20)); }
    [[nodiscard]] static T gainToDecibels(T linear, T minusInfinityDb) noexcept
    {
        return linear > T(1e-5) ? T(20) * std::log10(linear) : minusInfinityDb;
    }

    AudioSpec spec_ {};
    std::atomic<bool> prepared_ { false };
    DryWetMixer<T> mixer_;
    std::unique_ptr<Oversampling<T>> oversampler_;

    // Lock-free parameter states
    std::atomic<Mode> mode_ { Mode::Hard };
    std::atomic<T> ceilingDb_ { T(0) };
    std::atomic<T> inputGainDb_ { T(0) };
    std::atomic<int> stages_ { 1 };
    std::atomic<T> mix_ { T(1) };
    std::atomic<T> slewLimitMs_ { T(0) };
    std::atomic<int> osFactor_ { 1 }; // Default to 1 (off)

    // History states per channel
    T slewPrev_[kMaxChannels] {};

    // Parameter glides (processing rate) and their per-chunk ramps.
    static constexpr double kParamRampMs = 20.0;
    static constexpr int kChunk = 256;
    SmoothedValue<T> stageGainSm_;
    SmoothedValue<T> ceilingSm_;
    int lastStages_ = 1;   ///< Stage count the stage-gain glide refers to.
    std::array<T, kChunk> gainRamp_ {};
    std::array<T, kChunk> ceilRamp_ {};

    // Metering state
    std::atomic<T> gainReductionDb_ { T(0) };

    // Second-order antialiasing (processing rate, per channel and stage).
    struct AdaaStage
    {
        detail::Adaa2 state;
        double out1 = 0.0, out2 = 0.0;   ///< Last two stage outputs (ceiling units).
    };
    static constexpr int kPadRing = 32;   ///< > largest pad (16 * 1 - 1).
    std::atomic<bool> antialias_ { false };
    bool adaaActive_ = false;             ///< antialias_ as applied (audio thread).
    bool adaaFresh_ = true;               ///< Seed the stage history on the next block.
    Mode adaaMode_ = Mode::Hard;          ///< Curve the cached values belong to.
    int adaaStageCount_ = 1;
    int processFactor_ = 1;               ///< Oversampling factor in use.
    std::array<std::array<AdaaStage, kMaxStages>, kMaxChannels> adaaStages_ {};
    std::array<std::array<double, kPadRing>, kMaxChannels> padRing_ {};
    int padPos_ = 0;
    std::array<std::array<T, kMaxStages>, kMaxChannels> dryRing_ {};   ///< Dry delay, <= 4 samples.
    int dryPos_ = 0;
    AudioBuffer<T> dryScratch_;
};

} // namespace dspark
