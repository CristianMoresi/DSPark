// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Panner.h
 * @brief High-performance stereo panning toolkit with multiple algorithms.
 *
 * Provides CPU-efficient, zero-allocation panning algorithms suitable for
 * real-time audio threads (the settled equal-power path auto-vectorizes;
 * the smoothing paths are scalar by nature of the per-sample ramp).
 *
 * Algorithms:
 * - **Equal Power**: Constant-power pan (-3 dB center).
 * - **Binaural**: ITD delay + cross-feed with head-shadow ILD.
 * - **Mid Pan**: Preserves side signal, pans center image.
 * - **Side Pan**: Preserves mid signal, pans stereo width.
 * - **Haas**: Inter-channel delay precedence effect.
 * - **Spectral**: Frequency-dependent panning via high-shelf.
 *
 * Dependencies: Core/AudioBuffer.h, Core/AudioSpec.h, Core/Biquad.h,
 * Core/DspMath.h, Core/Smoothers.h, Core/StateBlob.h, Effects/Delay.h.
 *
 * Threading: prepare() belongs to the setup thread; processBlock()/reset() to
 * the audio thread. All parameter setters are safe from any thread (atomics
 * applied at the top of the next processBlock); non-finite values are ignored.
 *
 * @tparam T Floating-point precision type (float or double).
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/Smoothers.h"
#include "../Core/StateBlob.h"
#include "Delay.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dspark {

template <typename T = float>
class Panner
{
public:
    // Removed virtual destructor to maintain zero-cost abstraction (no vtable).
    ~Panner() = default;

    /** @brief Available panning algorithms. */
    enum class Algorithm
    {
        EqualPower, ///< Standard -3 dB constant-power pan.
        Binaural,   ///< Cross-feeding + ITD delay.
        MidPan,     ///< Pans only the centre (mid) image.
        SidePan,    ///< Pans only the stereo (side) image.
        Haas,       ///< Precedence effect via inter-channel delay.
        Spectral    ///< Frequency-dependent panning via high-shelf.
    };

    /**
     * @brief Initializes internal delays, filters, and smoothers.
     * @param spec The audio specification (sample rate, block size).
     *             An invalid spec (non-positive or NaN fields) is ignored.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        sampleRate_ = spec.sampleRate;
        panSmoother_.reset(sampleRate_, smoothingTime_.load(std::memory_order_relaxed));

        float maxMs = std::max(binauralMaxITD_.load(std::memory_order_relaxed),
                               haasMaxDelay_.load(std::memory_order_relaxed));

        AudioSpec monoSpec { sampleRate_, spec.maxBlockSize, 1 };
        // +1 ms of headroom over the configured maximum: the ITD/Haas paths
        // add a ~4-sample common base delay on top of the maximum, and the
        // capacity clamp must never shave it off.
        delayL_.prepareMs(monoSpec, static_cast<double>(maxMs) + 1.0);
        delayR_.prepareMs(monoSpec, static_cast<double>(maxMs) + 1.0);
        delayL_.setSmoother(Delay<T>::SmootherType::CriticallyDamped);
        delayR_.setSmoother(Delay<T>::SmootherType::CriticallyDamped);
        delayL_.setSmoothingTime(smoothingTime_);
        delayR_.setSmoothingTime(smoothingTime_);

        updateSpectralFilters(T(0));
    }

    /**
     * @brief Sets the active panning algorithm safely from any thread.
     * @note Switching is instant (no crossfade). The delay lines and spectral
     *       filters keep their state across switches; call reset() if a stale
     *       tail from the previous algorithm would be audible.
     * @param algo The algorithm enum to use. Out-of-range values are clamped.
     */
    void setAlgorithm(Algorithm algo) noexcept
    {
        algo = static_cast<Algorithm>(std::clamp(static_cast<int>(algo), 0,
                                                 static_cast<int>(Algorithm::Spectral)));
        algorithm_.store(algo, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the target pan position (automatable, smoothed).
     * @param position Target pan from -1.0 (left) to +1.0 (right).
     *                 Non-finite values are ignored.
     */
    void setPan(T position) noexcept
    {
        if (!std::isfinite(position)) return;
        // Publish only: the smoother is audio-thread state, so the target is
        // applied at the top of the next processBlock() (writing it here from
        // a control thread raced against getNextValue()).
        pan_.store(std::clamp(position, T(-1), T(1)), std::memory_order_relaxed);
    }

    /**
     * @brief Processes an audio block in-place. Real-time safe.
     * @param buffer Stereo audio buffer. Modifies the data directly.
     *               Buffers with fewer than 2 channels are left untouched.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumChannels() < 2) return;

        // Apply control-thread publications (audio-thread application point).
        if (smoothingDirty_.exchange(false, std::memory_order_acquire))
            panSmoother_.reset(sampleRate_,
                               smoothingTime_.load(std::memory_order_relaxed),
                               panSmoother_.getCurrentValue());

        float pTarget = static_cast<float>(pan_.load(std::memory_order_relaxed));
        panSmoother_.setTargetValue(pTarget);

        switch (algorithm_.load(std::memory_order_relaxed))
        {
            case Algorithm::EqualPower: applyEqualPower(buffer, pTarget); break;
            case Algorithm::Binaural:   applyCombinedBinaural(buffer, pTarget); break;
            case Algorithm::MidPan:     applyMidPan(buffer, pTarget); break;
            case Algorithm::SidePan:    applySidePan(buffer, pTarget); break;
            case Algorithm::Haas:       applyHaas(buffer, pTarget); break;
            case Algorithm::Spectral:   applySpectral(buffer, pTarget); break;
        }
    }

    /** @brief Clears delay lines and filter states to prevent ghost echoes. */
    void reset() noexcept
    {
        delayL_.reset();
        delayR_.reset();
        spectralL_.reset();
        spectralR_.reset();
        panSmoother_.skip();
    }

    // -- Configuration -------------------------------------------------------
    // All thread-safe; non-finite values are ignored. The ITD / Haas maxima
    // size the delay lines in prepare(): raising them beyond the prepared
    // value takes full effect on the next prepare().

    void setBinauralMaxITD(float ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        binauralMaxITD_.store(std::max(0.0f, ms), std::memory_order_relaxed);
    }
    void setHaasMaxDelay(float ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        haasMaxDelay_.store(std::max(0.0f, ms), std::memory_order_relaxed);
    }
    void setSpectralFrequency(float hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        spectralFreq_.store(std::clamp(hz, 20.0f, 20000.0f), std::memory_order_relaxed);
    }
    void setSpectralMaxGain(float dB) noexcept
    {
        if (!std::isfinite(dB)) return;
        spectralMaxGain_.store(dB, std::memory_order_relaxed);
    }

    /** @brief Sets the pan smoothing time. Thread-safe; applied at the top of
     *  the next processBlock() (previously it only took effect on the next
     *  prepare(), silently). */
    void setSmoothingTime(float ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        smoothingTime_.store(std::max(0.0f, ms), std::memory_order_relaxed);
        delayL_.setSmoothingTime(ms);   // Delay's own publish/apply handoff
        delayR_.setSmoothingTime(ms);
        smoothingDirty_.store(true, std::memory_order_release);
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("PANR"), 1);
        w.write("pan", pan_.load(std::memory_order_relaxed));
        w.write("algorithm", static_cast<int32_t>(algorithm_.load(std::memory_order_relaxed)));
        w.write("smoothing", smoothingTime_.load(std::memory_order_relaxed));
        w.write("binauralITD", binauralMaxITD_.load(std::memory_order_relaxed));
        w.write("haasDelay", haasMaxDelay_.load(std::memory_order_relaxed));
        w.write("spectralFreq", spectralFreq_.load(std::memory_order_relaxed));
        w.write("spectralGain", spectralMaxGain_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("PANR")) return false;
        setPan(static_cast<T>(r.read("pan", 0.0f)));
        setAlgorithm(static_cast<Algorithm>(r.read("algorithm", 0))); // clamped inside
        setSmoothingTime(r.read("smoothing", 50.0f));
        setBinauralMaxITD(r.read("binauralITD", 0.66f));
        setHaasMaxDelay(r.read("haasDelay", 30.0f));
        setSpectralFrequency(r.read("spectralFreq", 4000.0f));
        setSpectralMaxGain(r.read("spectralGain", 6.0f));
        return true;
    }

protected:
    void applyEqualPower(AudioBufferView<T> buffer, float /*panTarget*/) noexcept
    {
        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();
        constexpr T halfPi = pi<T> / T(2);

        if (!panSmoother_.isSmoothing())
        {
            // Static pan: hoist the trig out of the loop entirely.
            const T angle = (static_cast<T>(panSmoother_.getCurrentValue()) * T(0.5) + T(0.5)) * halfPi;
            const T gL = std::cos(angle);
            const T gR = std::sin(angle);
            for (int i = 0; i < n; ++i)
            {
                L[i] *= gL;
                R[i] *= gR;
            }
            return;
        }

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());
            T angle = (p * T(0.5) + T(0.5)) * halfPi;

            // fastSin/fastCos: > 100 dB accurate, several times cheaper than libm.
            L[i] *= fastCos(angle);
            R[i] *= fastSin(angle);
        }
    }

    void applyCombinedBinaural(AudioBufferView<T> buffer, float panTarget) noexcept
    {
        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        T itdMax = T(binauralMaxITD_.load(std::memory_order_relaxed));
        T targetP = static_cast<T>(panTarget);

        // Push the new ITD targets. A small COMMON base delay is added to both ears
        // so neither hits the delay line's 3-sample interpolation floor: that floor
        // used to clamp both ears to 3 for |pan| < ~0.1, killing the ITD cue near
        // centre (a localization dead-zone). With the base offset the ITD difference
        // is linear from the centre, and the shared delay is inaudible.
        const T baseMs = T(4000) / static_cast<T>(sampleRate_); // ~4 samples
        delayL_.setDelayMs(baseMs + itdMax * std::max(T(0), targetP));   // pan>0 => L delayed
        delayR_.setDelayMs(baseMs + itdMax * std::max(T(0), -targetP));  // pan<0 => R delayed

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());

            // Binaural cross-feed model:
            //  - The ipsilateral ear (closer to the source) receives the
            //    near channel essentially intact + a small leakage from the
            //    far channel.
            //  - The contralateral ear (far ear) receives the far channel
            //    attenuated + a leakage from the near channel + the ITD
            //    delay applied later by delayL_/delayR_.
            // This preserves audibility on both ears at hard pan, unlike a
            // straight mute, and produces the head-shadowing illusion typical
            // of real-world hearing (~6 dB ILD between ears at 90 deg).
            T absp     = std::abs(p);                  // 0..1
            T farAtten = T(1) - absp * T(0.5);         // 1.0 -> 0.5 at extreme
            T leakage  = absp * T(0.3);                // 0 -> 0.3 cross-bleed

            T l_temp, r_temp;
            if (p >= T(0))
            {
                // Pan right -> L is the contralateral (delayed) ear.
                l_temp = L[i] * farAtten + R[i] * leakage;
                r_temp = R[i] + L[i] * leakage * T(0.5);  // gentle near-ear bleed
            }
            else
            {
                // Pan left -> R is the contralateral ear.
                l_temp = L[i] + R[i] * leakage * T(0.5);
                r_temp = R[i] * farAtten + L[i] * leakage;
            }

            // Delay::processSample already advances its write index, so we
            // must NOT call advanceWriteIndex() afterwards.
            L[i] = delayL_.processSample(0, l_temp);
            R[i] = delayR_.processSample(0, r_temp);
        }
    }

    void applyMidPan(AudioBufferView<T> buffer, float /*panTarget*/) noexcept
    {
        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();
        constexpr T halfPi = pi<T> / T(2);

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());
            T mid  = (L[i] + R[i]) * T(0.5);
            T side = (L[i] - R[i]) * T(0.5);

            // Equal-power mid pan: gL^2 + gR^2 == 2 for every position, with
            // gL == gR == 1 at centre (transparent to within the fast-trig
            // accuracy, > 100 dB). A hard pan peaks at
            // +3 dB instead of the +6 dB of the old constant-voltage law,
            // keeping headroom predictable while staying mono-compatible.
            T angle = (p * T(0.5) + T(0.5)) * halfPi;
            T gL = fastCos(angle) * sqrt2<T>;
            T gR = fastSin(angle) * sqrt2<T>;

            L[i] = mid * gL + side;
            R[i] = mid * gR - side;
        }
    }

    void applySidePan(AudioBufferView<T> buffer, float /*panTarget*/) noexcept
    {
        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();
        constexpr T halfPi = pi<T> / T(2);

        for (int i = 0; i < n; ++i)
        {
            T p = static_cast<T>(panSmoother_.getNextValue());
            T mid  = (L[i] + R[i]) * T(0.5);
            T side = (L[i] - R[i]) * T(0.5);

            T angle = (p * T(0.5) + T(0.5)) * halfPi;

            // Normalized by sqrt(2) to prevent -3dB attenuation at dead center.
            // Clamped to avoid massive boosts at hard extremes.
            T sideGainL = std::clamp(fastCos(angle) * sqrt2<T>, T(0), T(1));
            T sideGainR = std::clamp(fastSin(angle) * sqrt2<T>, T(0), T(1));

            L[i] = mid + (side * sideGainL);
            R[i] = mid - (side * sideGainR);
        }
    }

    void applyHaas(AudioBufferView<T> buffer, float panTarget) noexcept
    {
        T haasMax = T(haasMaxDelay_.load(std::memory_order_relaxed));
        T pT = static_cast<T>(panTarget);

        // Smoothed inside Delay - no abrupt pointer jumps even on fast moves.
        // Common base delay keeps both ears above the 3-sample interpolation floor
        // so the inter-channel delay is linear from centre (no dead-zone).
        const T baseMs = T(4000) / static_cast<T>(sampleRate_); // ~4 samples
        delayL_.setDelayMs(baseMs + haasMax * std::max(T(0), pT));
        delayR_.setDelayMs(baseMs + haasMax * std::max(T(0), -pT));

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        for (int i = 0; i < n; ++i)
        {
            // Keep the pan smoother stepping in sync with the block, even
            // though Haas does not use the smoothed pan value directly.
            (void)panSmoother_.getNextValue();

            // processSample advances the write index by itself; calling
            // advanceWriteIndex() afterwards corrupts the delay line.
            L[i] = delayL_.processSample(0, L[i]);
            R[i] = delayR_.processSample(0, R[i]);
        }
    }

    void applySpectral(AudioBufferView<T> buffer, float panTarget) noexcept
    {
        // Recompute high-shelf coefficients based on the current pan target
        // (cheap, once per block - the trig math is hoisted out of the inner
        // loop). Biquad::processSample picks up the new coefficients on the
        // first sample via its lock-free fast path, so no extra sync needed.
        updateSpectralFilters(static_cast<T>(panTarget));

        T* L = buffer.getChannel(0);
        T* R = buffer.getChannel(1);
        const int n = buffer.getNumSamples();

        for (int i = 0; i < n; ++i)
        {
            (void)panSmoother_.getNextValue();  // keep smoother in step
            L[i] = spectralL_.processSample(L[i], 0);
            R[i] = spectralR_.processSample(R[i], 0);
        }
    }

    void updateSpectralFilters(T targetPan) noexcept
    {
        float sMaxGain = spectralMaxGain_.load(std::memory_order_relaxed);
        float sFreq    = spectralFreq_.load(std::memory_order_relaxed);

        T gainLdB = -targetPan * static_cast<T>(sMaxGain);
        T gainRdB =  targetPan * static_cast<T>(sMaxGain);

        spectralL_.setCoeffs(BiquadCoeffs<T>::makeHighShelf(
            sampleRate_, static_cast<double>(sFreq), static_cast<double>(gainLdB)));
        spectralR_.setCoeffs(BiquadCoeffs<T>::makeHighShelf(
            sampleRate_, static_cast<double>(sFreq), static_cast<double>(gainRdB)));
    }

    double sampleRate_ = 48000.0;
    std::atomic<Algorithm> algorithm_ { Algorithm::EqualPower };
    std::atomic<T> pan_ { T(0) };

    Delay<T> delayL_, delayR_;
    Smoothers::LinearSmoother panSmoother_;
    Biquad<T, 1> spectralL_, spectralR_;

    std::atomic<float> smoothingTime_   { 50.0f };
    std::atomic<bool>  smoothingDirty_  { false };  ///< Pan smoother re-config pending.
    std::atomic<float> binauralMaxITD_  { 0.66f };
    std::atomic<float> haasMaxDelay_    { 30.0f };
    std::atomic<float> spectralFreq_    { 4000.0f };
    std::atomic<float> spectralMaxGain_ { 6.0f };
};

} // namespace dspark
