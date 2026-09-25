// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Vibrato.h
 * @brief Pitch modulation via LFO-driven variable delay.
 *
 * Modulates pitch by varying a short delay line with a primary LFO. Features
 * FM modulation on the primary LFO for complex, non-static pitch variations,
 * and parameter smoothing to prevent zipper noise during automation.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Phasor.h, RingBuffer.h, DspMath.h, AudioSpec.h, AudioBuffer.h,
 *               StateBlob.h.
 *
 * @code
 *   dspark::Vibrato<float> vibrato;
 *   vibrato.prepare(spec);
 *   vibrato.setRate(5.0f);          // 5 Hz
 *   vibrato.setDepth(0.5f);         // 0.5 semitones
 *   vibrato.processBlock(buffer);
 * @endcode
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Phasor.h"
#include "../Core/RingBuffer.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class Vibrato
 * @brief Professional-grade pitch vibrato with LFO FM and parameter smoothing.
 *
 * The modulation depth is specified in semitones. A secondary oscillator (FM)
 * can modulate the primary LFO rate. Internally applies block-based parameter
 * smoothing to ensure artifact-free automation.
 *
 * All per-channel LFOs share the same phase, so the effect stays
 * mono-compatible. Channels beyond those passed to prepare() are left
 * untouched (pass-through), as is the whole buffer before prepare().
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Vibrato
{
public:
    /**
     * @brief Allocates delay lines and settles the parameter smoothing state.
     *
     * The delay lines are sized for the deepest configuration the setters
     * allow: 4 semitones at the 0.1 Hz rate floor. The peak delay request of
     * that sweep is centre + deviation = 2 * deviation + offset (~35.4k
     * samples at 48 kHz before the ring's power-of-two round-up), because the
     * centre itself sits one deviation above the offset so the trough of the
     * sweep never dips below it.
     *
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment specification. Defines num channels and sample rate.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        sampleRate_ = spec.sampleRate;
        numChannels_ = spec.numChannels;

        // Worst-case LFO deviation in samples: depth * ln2 / 12 converts
        // semitones to a log-frequency ratio, and the peak pitch excursion of
        // a sinusoidal delay d(n) = centre + D * sin(2*pi*f*n/fs) is
        // D * 2*pi*f/fs, so D = depth * ln2 * fs / (2*pi * 12 * f). Computed
        // in double and capped before the int cast (an absurd-but-finite
        // sample rate must not overflow the conversion; the ring clamps
        // further), with 128 samples of padding for safety.
        constexpr double kMinAllowedHz = 0.1;
        constexpr double kMaxAllowedSemitones = 4.0;
        const double maxDeviation =
            (kMaxAllowedSemitones * std::numbers::ln2 * sampleRate_)
          / (2.0 * std::numbers::pi * kMinAllowedHz * 12.0);
        const double required =
            2.0 * maxDeviation + static_cast<double>(kCentreOffset) + 128.0;
        const int maxDelaySamples =
            static_cast<int>(std::min(required, static_cast<double>(1 << 28)));

        delays_.resize(numChannels_);
        phasors_.resize(numChannels_);
        modPhasors_.resize(numChannels_);
        readPos_.assign(static_cast<size_t>(numChannels_), T(-1));

        for (int ch = 0; ch < numChannels_; ++ch)
        {
            delays_[ch].prepare(maxDelaySamples);
            phasors_[ch].prepare(sampleRate_);
            modPhasors_[ch].prepare(sampleRate_);
        }

        // Initialize smoothing state to prevent startup ramps
        currentRate_ = rate_.load(std::memory_order_relaxed);
        currentDepth_ = depthSemitones_.load(std::memory_order_relaxed);
        currentModDepth_ = modDepth_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Processes audio in-place, applying vibrato per channel.
     * @param buffer Audio data (must match channels passed in prepare).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numCh = std::min(buffer.getNumChannels(), numChannels_);
        const int numSamples = buffer.getNumSamples();
        if (numSamples == 0 || numCh == 0) return;

        // Fetch targets
        const T targetRate = rate_.load(std::memory_order_relaxed);
        const T targetDepth = depthSemitones_.load(std::memory_order_relaxed);
        const T modRate = modRate_.load(std::memory_order_relaxed);
        const T targetModDepth = modDepth_.load(std::memory_order_relaxed);

        // Parameter smoothing increments (linear ramp over the block). Rate
        // and depth set the width and centre of the delay sweep; the read
        // position slew limit below turns even a large change into a brief
        // glide. The FM depth is ramped too, so the LFO speed changes
        // smoothly. The FM rate needs no smoothing: it only changes the speed
        // of a continuous phase, which cannot produce a discontinuity.
        const T rateInc = (targetRate - currentRate_) / static_cast<T>(numSamples);
        const T depthInc = (targetDepth - currentDepth_) / static_cast<T>(numSamples);
        const T modDepthInc = (targetModDepth - currentModDepth_) / static_cast<T>(numSamples);

        // Keep the FM oscillator running while any part of the depth ramp is
        // live, so a fade-out drains along the moving LFO instead of freezing
        // mid-ramp. With FM fully off the oscillator holds its phase.
        const bool fmActive = (targetModDepth > T(0)) || (currentModDepth_ > T(0));

        constexpr T kLn2 = static_cast<T>(std::numbers::ln2_v<double>);
        constexpr T kTwoPi = static_cast<T>(2.0 * std::numbers::pi);
        const T deviationScaler = (kLn2 * static_cast<T>(sampleRate_)) / (kTwoPi * T(12));

        for (int ch = 0; ch < numCh; ++ch)
        {
            T* data = buffer.getChannel(ch);
            auto& delay = delays_[ch];
            auto& phasor = phasors_[ch];
            auto& modPhasor = modPhasors_[ch];

            modPhasor.setFrequency(modRate);

            // Local state for smoothing (identical ramps on every channel)
            T smoothRate = currentRate_;
            T smoothDepth = currentDepth_;
            T smoothModDepth = currentModDepth_;

            for (int i = 0; i < numSamples; ++i)
            {
                smoothRate += rateInc;
                smoothDepth += depthInc;
                smoothModDepth += modDepthInc;

                delay.push(data[i]);

                // Sweep geometry from the smoothed base rate only: a sinusoidal
                // delay centre + D * sin(phase) peaks at a pitch excursion of
                // D * 2*pi*f/fs, so D = depth * deviationScaler / rate gives
                // the set depth at the base rate, and the centre sits one
                // deviation above the offset so the trough never dips below
                // it. FM used to feed its instantaneous rate into D and the
                // centre (D ~ rate^-1.5): at deep or fast FM the rate neared
                // the floor and the read point leapt by hundreds of samples
                // per sample (noise, not vibrato).
                const T baseRate = std::max(smoothRate, T(0.1));
                const T deviation = (smoothDepth * deviationScaler) / baseRate;
                const T centre = deviation + kCentreOffset;

                // FM varies only the speed of the LFO, between (1 - modDepth)
                // and (1 + modDepth) times the rate, so the pitch excursion
                // follows the instantaneous speed: up to twice the depth at
                // full FM, and never a jump.
                T speed = T(1);
                if (fmActive)
                    speed += fastSin(modPhasor.advance() * kTwoPi) * smoothModDepth;
                phasor.setFrequency(baseRate * std::max(speed, T(0)));
                const T lfo = fastSin(phasor.advance() * kTwoPi);

                // Safety clamp: the 4-point interpolator reads one sample
                // earlier and two later, hence [1, cap - 4].
                const T delaySamples = std::clamp(centre + lfo * deviation, T(1.0),
                                                  static_cast<T>(delay.getCapacity() - 4));

                // Read-position slew limit. Depth and rate scale the deviation
                // and centre, so a parameter change moves the read point;
                // ramped per block, a small block moved it by hundreds of
                // samples at once (a click, or a backwards read). Limiting the
                // motion to 0.5 samples per sample bounds any parameter jump
                // to a brief pitch glide, and never touches the vibrato itself
                // (its steepest motion, 4 semitones at full FM, is
                // 2 * 4 * ln2 / 12 = 0.46 samples per sample).
                T& pos = readPos_[static_cast<size_t>(ch)];
                pos = (pos < T(0)) ? delaySamples
                                   : pos + std::clamp(delaySamples - pos, -kMaxReadSlope, kMaxReadSlope);

                data[i] = delay.readInterpolated(pos);
            }
        }

        // Update current state for the next block
        currentRate_ = targetRate;
        currentDepth_ = targetDepth;
        currentModDepth_ = targetModDepth;
    }

    /**
     * @brief Clears delay line memory and resets LFO phases.
     *
     * The smoothed parameter state is kept (no ramp is re-triggered).
     */
    void reset() noexcept
    {
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            delays_[ch].reset();
            phasors_[ch].reset();
            modPhasors_[ch].reset();
        }
        std::fill(readPos_.begin(), readPos_.end(), T(-1)); // re-seed on the next sample
    }

    /**
     * @brief Sets the primary LFO rate. Parameter is smoothed internally.
     * @param hz Vibrato frequency in Hz (0.1 to 14 typical). Floored at
     * 0.1 Hz, the minimum the delay-line sizing assumes; there is no upper
     * clamp (higher rates shrink the deviation accordingly). Non-finite
     * values are ignored.
     */
    void setRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return; // NaN/Inf would poison the delay sweep
        rate_.store(std::max(hz, T(0.1)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the vibrato pitch depth. Parameter is smoothed internally.
     * @param semitones Modulation depth, clamped to [0, 4] semitones (the
     * range the delay lines allocated in prepare() cover). Non-finite values
     * are ignored.
     */
    void setDepth(T semitones) noexcept
    {
        if (!std::isfinite(semitones)) return;
        depthSemitones_.store(std::clamp(semitones, T(0), T(4)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the rate of the secondary FM oscillator.
     *
     * Not smoothed by design: a rate change only alters the speed of the FM
     * oscillator's continuous phase, which cannot produce a discontinuity.
     *
     * @param hz Secondary LFO rate in Hz (floored at 0). Set to 0 to disable
     * FM. Non-finite values are ignored.
     */
    void setModRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        modRate_.store(std::max(hz, T(0)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the intensity of the FM modulation on the primary LFO.
     *
     * FM varies the speed of the vibrato LFO between (1 - amount) and
     * (1 + amount) times the rate; the delay sweep keeps the width that depth
     * and rate set, so the pitch excursion follows the instantaneous speed
     * (up to twice the depth at full FM). Smoothed internally. While the
     * depth is zero the FM oscillator holds its phase.
     *
     * @param amount 0.0 (off) to 1.0 (full modulation), clamped. Non-finite
     * values are ignored.
     */
    void setModDepth(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        modDepth_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDepth() const noexcept { return depthSemitones_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getModRate() const noexcept { return modRate_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getModDepth() const noexcept { return modDepth_.load(std::memory_order_relaxed); }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        // The blob stores float (setState reads float back); the explicit
        // casts also keep this overload resolvable when T is double.
        StateWriter w(stateId("VIBR"), 1);
        w.write("rate", static_cast<float>(rate_.load(std::memory_order_relaxed)));
        w.write("depth", static_cast<float>(depthSemitones_.load(std::memory_order_relaxed)));
        w.write("modRate", static_cast<float>(modRate_.load(std::memory_order_relaxed)));
        w.write("modDepth", static_cast<float>(modDepth_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("VIBR")) return false;
        setRate(static_cast<T>(r.read("rate", 5.0f)));
        setDepth(static_cast<T>(r.read("depth", 0.5f)));
        setModRate(static_cast<T>(r.read("modRate", 0.0f)));
        setModDepth(static_cast<T>(r.read("modDepth", 0.0f)));
        return true;
    }

private:
    /// Headroom below the sweep trough so the 4-point interpolator's earliest
    /// tap always reads a written sample.
    static constexpr T kCentreOffset = T(4);
    /// Largest read-position change per sample (see processBlock()).
    static constexpr T kMaxReadSlope = T(0.5);

    double sampleRate_ = 44100.0;
    int numChannels_ = 0;

    // Atomic targets for UI thread publication
    std::atomic<T> rate_ { T(5) };
    std::atomic<T> depthSemitones_ { T(0.5) };
    std::atomic<T> modRate_ { T(0) };
    std::atomic<T> modDepth_ { T(0) };

    // Smoothed state for the audio thread
    T currentRate_ { T(5) };
    T currentDepth_ { T(0.5) };
    T currentModDepth_ { T(0) };

    // Dynamic allocation via STL, safe because it only happens in prepare()
    std::vector<RingBuffer<T>> delays_;
    std::vector<Phasor<T>> phasors_;
    std::vector<Phasor<T>> modPhasors_;
    std::vector<T> readPos_;   ///< Slew-limited read position per channel (-1 = unseeded).
};

} // namespace dspark
