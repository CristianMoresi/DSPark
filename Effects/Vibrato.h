// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Vibrato.h
 * @brief Pitch modulation via LFO-driven variable delay.
 *
 * Modulates pitch by varying a short delay line with a primary LFO. Features
 * FM modulation on the primary LFO for complex, non-static pitch variations,
 * and parameter smoothing to prevent zipper noise during automation. The LFO
 * phase and the read position are computed in double whatever T is (in float
 * a 4-semitone sweep at 0.1 Hz misplaced the read point by about 0.01 sample).
 * Reads use 4-point Hermite interpolation by default, or optionally the
 * framework's 32-tap Kaiser sinc reader (setInterpolation()).
 *
 * setModel(Model::Analog) (off by default) runs each channel through a
 * bucket-brigade line of 1024 stages (Core/BucketBrigade.h) whose clock the
 * same sweep moves (delay = stages / (2 f_clk)), with the device's anti-alias
 * and reconstruction filters at a fifth of the slowest clock the sweep needs
 * (so a deep, slow sweep, a long delay, is darker, as on the hardware) and
 * the line's hiss (setAnalogNoise(), -84 dBFS by default), which grows with
 * the delay. The sweep's trough sits 12 samples behind the input (the
 * fastest clock the line runs). The engines do not crossfade: a switch
 * starts the analog line empty.
 *
 * Threading: prepare() belongs to the setup thread; processBlock() and reset()
 * belong to the audio thread. Setters are lock-free atomic publications, safe
 * from any thread, consumed at the next processBlock(). Non-finite setter
 * arguments are ignored.
 *
 * Dependencies: Phasor.h, RingBuffer.h, DspMath.h, AudioSpec.h, AudioBuffer.h,
 *               BucketBrigade.h, Interpolation.h, StateBlob.h.
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
#include "../Core/BucketBrigade.h"
#include "../Core/DspMath.h"
#include "../Core/Interpolation.h"
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
    /** @brief Fractional read kernel of the modulated delay line. */
    enum class Interpolation
    {
        Hermite,  ///< 4-point cubic (default).
        Sinc      ///< 32-tap Kaiser sinc: transparent to 15 kHz at 44.1/48 kHz,
                  ///< about 5x the read cost; the sweep trough sits 16 samples
                  ///< (0.33 ms at 48 kHz) behind the input instead of 4.
    };

    /** @brief The engine. */
    enum class Model
    {
        Digital,  ///< Interpolated reads of a delay line (default).
        Analog    ///< A bucket-brigade line per channel (see the file comment).
    };

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
        readPos_.assign(static_cast<size_t>(numChannels_), -1.0);
        sinc_ = &sincTable();   // built on this setup thread, once per process

        bbd_.resize(static_cast<size_t>(numChannels_));
        noiseLp_.assign(static_cast<size_t>(numChannels_), 0.0);
        noiseRng_.resize(static_cast<size_t>(numChannels_));
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            delays_[ch].prepare(maxDelaySamples);
            phasors_[ch].prepare(sampleRate_);
            modPhasors_[ch].prepare(sampleRate_);
            bbd_[static_cast<size_t>(ch)].prepare(sampleRate_, kAnalogStages, 0.45 * sampleRate_);
            noiseRng_[static_cast<size_t>(ch)] = 0x5A17C3u + 0x9E3779B9u * static_cast<uint32_t>(ch);
        }
        analogCornerFor_ = -1.0;

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

        constexpr double kTwoPi = 2.0 * std::numbers::pi;
        const double deviationScaler = (std::numbers::ln2 * sampleRate_) / (kTwoPi * 12.0);
        const bool analog = model_.load(std::memory_order_relaxed) == Model::Analog;
        const bool sinc = !analog && sinc_ && interpolation_.load(std::memory_order_relaxed) == Interpolation::Sinc;
        // Trough headroom and read window: Hermite reads 1 sample later and 2
        // earlier; the sinc reader needs 16 later and 16 earlier; the analog
        // line's trough is the fastest clock it runs.
        const double centreOffset = analog ? kAnalogTrough
                                  : sinc ? static_cast<double>(SincInterpolator<T>::kAfter)
                                         : static_cast<double>(kCentreOffset);
        const double maxRead = static_cast<double>(delays_[0].getCapacity()
                             - (sinc ? SincInterpolator<T>::kAfter + 1 : 4));

        double noiseAmp = 0.0;
        if (analog)
        {
            // The filters sit at a fifth of the slowest clock of this block's
            // sweep (its peak delay, 2 * deviation + trough).
            const auto deviationOf = [&](T depth, T rate) {
                return static_cast<double>(depth) * deviationScaler / std::max(static_cast<double>(rate), 0.1);
            };
            const double peak = 2.0 * std::max(deviationOf(currentDepth_, currentRate_), deviationOf(targetDepth, targetRate))
                              + centreOffset;
            const double slowest = static_cast<double>(kAnalogStages) * sampleRate_ / (2.0 * peak);
            if (std::abs(slowest - analogCornerFor_) > 1e-3 * slowest)
            {
                const double corner = std::min(0.2 * slowest, 0.45 * sampleRate_);
                for (auto& b : bbd_) b.setCorner(corner);
                noiseCoeff_ = 1.0 - std::exp(-2.0 * std::numbers::pi * corner / sampleRate_);
                analogCornerFor_ = slowest;
            }
            const T noiseDb = analogNoiseDb_.load(std::memory_order_relaxed);
            // White noise through a one-pole at the corner, scaled by that
            // filter's own gain (the same level at any rate), and by the
            // delay against the sweep's centre (a slower clock, more hiss).
            if (noiseDb > T(-120))
                noiseAmp = std::pow(10.0, static_cast<double>(noiseDb) / 20.0) * 1.7320508075688772
                         / std::sqrt(std::max(1e-9, noiseCoeff_ / (2.0 - noiseCoeff_)));
        }

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
                const double baseRate = std::max(static_cast<double>(smoothRate), 0.1);
                const double deviation = (static_cast<double>(smoothDepth) * deviationScaler) / baseRate;
                const double centre = deviation + centreOffset;

                // FM varies only the speed of the LFO, between (1 - modDepth)
                // and (1 + modDepth) times the rate, so the pitch excursion
                // follows the instantaneous speed: up to twice the depth at
                // full FM, and never a jump.
                double speed = 1.0;
                if (fmActive)
                    speed += fastSin(modPhasor.advance() * kTwoPi) * static_cast<double>(smoothModDepth);
                phasor.setFrequency(baseRate * std::max(speed, 0.0));
                const double lfo = fastSin(phasor.advance() * kTwoPi);

                // Safety clamp to the interpolator's read window.
                const double delaySamples = std::clamp(centre + lfo * deviation,
                                                       (sinc || analog) ? centreOffset : 1.0, maxRead);

                // Read-position slew limit. Depth and rate scale the deviation
                // and centre, so a parameter change moves the read point;
                // ramped per block, a small block moved it by hundreds of
                // samples at once (a click, or a backwards read). Limiting the
                // motion to 0.5 samples per sample bounds any parameter jump
                // to a brief pitch glide, and never touches the vibrato itself
                // (its steepest motion, 4 semitones at full FM, is
                // 2 * 4 * ln2 / 12 = 0.46 samples per sample).
                double& pos = readPos_[static_cast<size_t>(ch)];
                pos = (pos < 0.0) ? delaySamples
                                  : pos + std::clamp(delaySamples - pos, -kMaxReadSlope, kMaxReadSlope);

                if (analog)
                {
                    const auto c = static_cast<size_t>(ch);
                    double y = static_cast<double>(bbd_[c].processSample(data[i], pos / sampleRate_));
                    if (noiseAmp > 0.0)
                    {
                        noiseRng_[c] = noiseRng_[c] * 1664525u + 1013904223u;
                        const double n = static_cast<double>(noiseRng_[c] >> 8) / 8388608.0 - 1.0;
                        noiseLp_[c] += noiseCoeff_ * (n - noiseLp_[c]);
                        y += noiseAmp * (pos / centre) * noiseLp_[c];
                    }
                    data[i] = static_cast<T>(y);
                }
                else
                {
                    data[i] = sinc ? readSinc(delay, pos) : readHermite(delay, pos);
                }
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
        for (auto& b : bbd_) b.reset();
        std::fill(noiseLp_.begin(), noiseLp_.end(), 0.0);
        std::fill(readPos_.begin(), readPos_.end(), -1.0); // re-seed on the next sample
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

    /** @brief Selects the fractional read kernel. Thread-safe; a switch moves
     *  the sweep trough (see Interpolation), which the read-position slew
     *  turns into a glide of a few samples. */
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

    /** @brief Selects the engine (see Model). Thread-safe; the engines do not
     *  crossfade; out-of-range values are clamped. */
    void setModel(Model m) noexcept
    {
        model_.store(static_cast<Model>(std::clamp(static_cast<int>(m), 0, 1)), std::memory_order_relaxed);
    }
    /** @return The engine. */
    [[nodiscard]] Model getModel() const noexcept { return model_.load(std::memory_order_relaxed); }

    /**
     * @brief Hiss of the analog line at the sweep's centre.
     * @param dbfs RMS level in dBFS, clamped to [-140, -40]; -120 or less is
     *             silent. Non-finite values are ignored.
     */
    void setAnalogNoise(T dbfs) noexcept
    {
        if (!std::isfinite(dbfs)) return;
        analogNoiseDb_.store(std::clamp(dbfs, T(-140), T(-40)), std::memory_order_relaxed);
    }
    /** @return The hiss level of the analog line in dBFS. */
    [[nodiscard]] T getAnalogNoise() const noexcept { return analogNoiseDb_.load(std::memory_order_relaxed); }

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
        w.write("interp", static_cast<int32_t>(interpolation_.load(std::memory_order_relaxed)));
        w.write("model", static_cast<int32_t>(model_.load(std::memory_order_relaxed)));
        w.write("analogNoise", static_cast<float>(analogNoiseDb_.load(std::memory_order_relaxed)));
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
        setInterpolation(r.read("interp", 0) == 1 ? Interpolation::Sinc : Interpolation::Hermite);
        setModel(r.read("model", 0) == 1 ? Model::Analog : Model::Digital);
        setAnalogNoise(static_cast<T>(r.read("analogNoise", -84.0f)));
        return true;
    }

private:
    /// Headroom below the sweep trough so the 4-point interpolator's earliest
    /// tap always reads a written sample.
    static constexpr T kCentreOffset = T(4);
    /// Largest read-position change per sample (see processBlock()).
    static constexpr double kMaxReadSlope = 0.5;
    /// Stages of the analog line, and its trough in samples: 1024 stages
    /// clocked at most 64 times per sample give no less than 8 samples.
    static constexpr int kAnalogStages = 1024;
    static constexpr double kAnalogTrough = 12.0;

    /** @brief The shared sinc table (built on first use: prepare() calls it). */
    static const SincInterpolator<T>& sincTable()
    {
        static const SincInterpolator<T> table;
        return table;
    }

    /** @brief Hermite read `delay` samples behind the newest sample, with the
     *  position split in double (RingBuffer::readInterpolated() semantics). */
    [[nodiscard]] static T readHermite(const RingBuffer<T>& ring, double delay) noexcept
    {
        const int d = static_cast<int>(delay);
        const T frac = static_cast<T>(delay - static_cast<double>(d));
        return interpolateHermite(ring.read(d - 1), ring.read(d), ring.read(d + 1), ring.read(d + 2), frac);
    }

    /** @brief Sinc read `delay` samples behind the newest sample (needs
     *  delay >= 16 and 16 samples of history beyond it). */
    [[nodiscard]] T readSinc(const RingBuffer<T>& ring, double delay) const noexcept
    {
        const int d = static_cast<int>(delay);
        const double frac = delay - static_cast<double>(d);
        const int64_t newest = static_cast<int64_t>(ring.getWritePosition()) - 1;
        const int64_t intPos = newest - d - (frac > 0.0 ? 1 : 0);
        return sinc_->readRing(ring.data(), static_cast<int64_t>(ring.getCapacity() - 1), intPos,
                               frac > 0.0 ? 1.0 - frac : 0.0);
    }

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
    std::vector<Phasor<double>> phasors_;      ///< Double phase: see the file comment.
    std::vector<Phasor<double>> modPhasors_;
    std::vector<double> readPos_;   ///< Slew-limited read position per channel (-1 = unseeded).
    std::atomic<Interpolation> interpolation_ { Interpolation::Hermite };
    const SincInterpolator<T>* sinc_ = nullptr;   ///< Set by prepare().

    std::atomic<Model> model_ { Model::Digital };
    std::atomic<T> analogNoiseDb_ { T(-84) };
    std::vector<BucketBrigade<T>> bbd_;       ///< One analog line per channel.
    std::vector<double> noiseLp_;             ///< Hiss filter state per channel.
    std::vector<uint32_t> noiseRng_;          ///< Hiss generator per channel.
    double noiseCoeff_ = 0.0, analogCornerFor_ = -1.0;
};

} // namespace dspark
