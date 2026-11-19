// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file EnvelopeFollower.h
 * @brief Peak/RMS envelope follower with asymmetric attack/release, and a
 *        transient follower that does not depend on the level.
 *
 * The detector every dynamics processor carries internally, exposed as a
 * public building block: a one-pole follower with independent attack and
 * release time constants, in Peak mode (rectified signal) or RMS mode
 * (the ballistics run on the squared signal and readouts return the square
 * root, so the time constants act in the power domain - the standard
 * detector convention). Use it as a sidechain detector, a modulation
 * source (with ModulationRouter), or a meter.
 *
 * Non-finite input samples: the poisoned channel restarts from zero at the
 * next publish and measurement resumes immediately (attack-time recovery);
 * the readouts never serve non-finite values.
 *
 * TransientFollower, in the same header, reads how sudden the signal is
 * rather than how loud: the ratio of a fast-attack follower to a
 * slow-attack one (the same release for both), mapped from decibels to 0..1,
 * with a hold and a release on the result. A drum hit gives the same value
 * at -40 dBFS as at -10 dBFS, and a held note gives none: a modulation
 * source for what happens on the onsets.
 *
 * Threading:
 * - prepare(): setup thread (allocation-free; not concurrent with
 *   processing).
 * - processBlock() / processSample() / reset(): audio thread (stream
 *   owner).
 * - setAttack() / setRelease() / setMode(): control thread (ONE non-audio
 *   writer -- the time setters re-derive the shared coefficient words;
 *   relaxed atomics, non-finite values are ignored).
 * - getEnvelope() / getEnvelopeMax() / getEnvelopeDb() and the parameter
 *   getters: any thread, lock-free.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h, Hilbert.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"
#include "../Core/Hilbert.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>

namespace dspark {

/**
 * @class EnvelopeFollower
 * @brief Attack/release envelope detector (Peak or RMS).
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Maximum tracked channels.
 */
template <FloatType T, int MaxChannels = 16>
class EnvelopeFollower
{
public:
    /** @brief Detection law. */
    enum class Mode { Peak, RMS };

    EnvelopeFollower() noexcept { updateCoeffs(); }

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the follower. Allocation-free.
     *
     * An invalid or non-finite specification is ignored (conservative
     * no-op: the previous state is kept). Channels beyond MaxChannels are
     * clamped.
     */
    void prepare(const AudioSpec& spec) noexcept
    {
        if (!spec.isValid() || !std::isfinite(spec.sampleRate)) return;
        sampleRate_.store(spec.sampleRate, std::memory_order_relaxed);
        numChannels_.store(std::clamp(spec.numChannels, 1, MaxChannels),
                           std::memory_order_relaxed);
        updateCoeffs();
        reset();
    }

    /**
     * @brief Clears all envelopes.
     *
     * Allocation-free, but it writes the same plain per-channel state the
     * processing calls own: call it from the stream owner (or with the
     * stream stopped).
     */
    void reset() noexcept
    {
        for (auto& s : state_) s = 0.0;
        for (auto& e : published_) e.store(T(0), std::memory_order_relaxed);
    }

    // -- Parameters ----------------------------------------------------------------

    /** @brief Attack time in milliseconds (default 10, floor 0.01;
     *         non-finite values are ignored). */
    void setAttack(T ms) noexcept
    {
        const double v = static_cast<double>(ms);
        if (!std::isfinite(v)) return;
        attackMs_.store(std::max(v, 0.01), std::memory_order_relaxed);
        updateCoeffs();
    }

    /** @brief Release time in milliseconds (default 150, floor 0.01;
     *         non-finite values are ignored). */
    void setRelease(T ms) noexcept
    {
        const double v = static_cast<double>(ms);
        if (!std::isfinite(v)) return;
        releaseMs_.store(std::max(v, 0.01), std::memory_order_relaxed);
        updateCoeffs();
    }

    /** @brief Peak (default) or RMS detection. Out-of-range values clamp. */
    void setMode(Mode m) noexcept
    {
        const int v = std::clamp(static_cast<int>(m), 0, 1);
        mode_.store(static_cast<Mode>(v), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------------

    /**
     * @brief Analyzes a block (read-only) and updates per-channel envelopes.
     *
     * Channels beyond the prepared count are ignored.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(),
                                 numChannels_.load(std::memory_order_relaxed));
        const int nS = buffer.getNumSamples();
        const double aAtt = attackA_.load(std::memory_order_relaxed);
        const double aRel = releaseA_.load(std::memory_order_relaxed);
        const bool peak = mode_.load(std::memory_order_relaxed) == Mode::Peak;

        for (int ch = 0; ch < nCh; ++ch)
        {
            const T* d = buffer.getChannel(ch);
            double env = state_[static_cast<size_t>(ch)];
            if (peak)
            {
                for (int i = 0; i < nS; ++i)
                {
                    const double x = std::abs(static_cast<double>(d[i]));
                    const double a = (x > env) ? aAtt : aRel;
                    env = a * env + (1.0 - a) * x;
                }
            }
            else
            {
                for (int i = 0; i < nS; ++i)
                {
                    const double x = static_cast<double>(d[i]) * d[i];
                    const double a = (x > env) ? aAtt : aRel;
                    env = a * env + (1.0 - a) * x;
                }
            }
            // Publish-time guard: a non-finite sample would park NaN in the
            // recursion for good (the branch comparison goes false forever),
            // so the poisoned channel restarts and re-measures on the next
            // block. The same test flushes fully decayed envelopes to true
            // zero (~-1000 dBFS) before they walk into double denormals.
            if (!std::isfinite(env) || env < 1e-100) env = 0.0;
            state_[static_cast<size_t>(ch)] = env;
            published_[static_cast<size_t>(ch)].store(readout(env, peak),
                                                      std::memory_order_relaxed);
        }
    }

    /**
     * @brief Single-sample path on channel 0 (for embedding in processors).
     *
     * Also publishes to the lock-free readout, so getEnvelope() stays
     * coherent with this path.
     *
     * @return The current envelope (linear amplitude).
     */
    [[nodiscard]] T processSample(T input) noexcept
    {
        const double aAtt = attackA_.load(std::memory_order_relaxed);
        const double aRel = releaseA_.load(std::memory_order_relaxed);
        const bool peak = mode_.load(std::memory_order_relaxed) == Mode::Peak;
        double env = state_[0];
        const double x = peak ? std::abs(static_cast<double>(input))
                              : static_cast<double>(input) * input;
        const double a = (x > env) ? aAtt : aRel;
        env = a * env + (1.0 - a) * x;
        if (!std::isfinite(env) || env < 1e-100) env = 0.0;
        state_[0] = env;
        const T out = readout(env, peak);
        published_[0].store(out, std::memory_order_relaxed);
        return out;
    }

    // -- Readout (lock-free, any thread) ------------------------------------------------

    /** @return Envelope of `channel` as linear amplitude. */
    [[nodiscard]] T getEnvelope(int channel = 0) const noexcept
    {
        const int n = numChannels_.load(std::memory_order_relaxed);
        channel = std::clamp(channel, 0, n - 1);
        return published_[static_cast<size_t>(channel)].load(std::memory_order_relaxed);
    }

    /** @return Loudest channel envelope (useful as a mono modulation source). */
    [[nodiscard]] T getEnvelopeMax() const noexcept
    {
        const int n = numChannels_.load(std::memory_order_relaxed);
        T m = T(0);
        for (int ch = 0; ch < n; ++ch)
            m = std::max(m, published_[static_cast<size_t>(ch)].load(std::memory_order_relaxed));
        return m;
    }

    /** @return Envelope of `channel` in dBFS (floor -120 dB). */
    [[nodiscard]] T getEnvelopeDb(int channel = 0) const noexcept
    {
        const double e = static_cast<double>(getEnvelope(channel));
        return static_cast<T>(20.0 * std::log10(std::max(e, 1e-6)));
    }

    /**
     * @return Envelope of `channel` in decibels mapped to 0..1: 0 at
     *         floorDb and under it, 1 at 0 dBFS and over it (a modulation
     *         source that moves evenly in level).
     * @param channel The channel to read.
     * @param floorDb The level that reads 0 (-120 to -1 dBFS; default -60).
     */
    [[nodiscard]] T getEnvelopeNormalized(int channel = 0, T floorDb = T(-60)) const noexcept
    {
        const double floor = std::isfinite(floorDb) ? std::clamp(static_cast<double>(floorDb), -120.0, -1.0) : -60.0;
        const double db = static_cast<double>(getEnvelopeDb(channel));
        return static_cast<T>(std::clamp((db - floor) / -floor, 0.0, 1.0));
    }

    /** @return Attack time in milliseconds. */
    [[nodiscard]] T getAttack() const noexcept
    {
        return static_cast<T>(attackMs_.load(std::memory_order_relaxed));
    }

    /** @return Release time in milliseconds. */
    [[nodiscard]] T getRelease() const noexcept
    {
        return static_cast<T>(releaseMs_.load(std::memory_order_relaxed));
    }

    /** @return Current detection law. */
    [[nodiscard]] Mode getMode() const noexcept
    {
        return mode_.load(std::memory_order_relaxed);
    }

private:
    static_assert(MaxChannels >= 1, "EnvelopeFollower needs at least one channel");

    [[nodiscard]] T readout(double env, bool peak) const noexcept
    {
        return static_cast<T>(peak ? env : std::sqrt(std::max(env, 0.0)));
    }

    void updateCoeffs() noexcept
    {
        const double fs = sampleRate_.load(std::memory_order_relaxed);
        attackA_.store(std::exp(-1.0 / (attackMs_.load(std::memory_order_relaxed) * 0.001 * fs)),
                       std::memory_order_relaxed);
        releaseA_.store(std::exp(-1.0 / (releaseMs_.load(std::memory_order_relaxed) * 0.001 * fs)),
                        std::memory_order_relaxed);
    }

    std::atomic<double> sampleRate_ { 48000.0 };
    std::atomic<int> numChannels_ { 1 };
    std::atomic<double> attackMs_ { 10.0 };
    std::atomic<double> releaseMs_ { 150.0 };
    std::atomic<double> attackA_ { 0.99 };
    std::atomic<double> releaseA_ { 0.999 };
    std::atomic<Mode> mode_ { Mode::Peak };

    std::array<double, MaxChannels> state_ {};
    std::array<std::atomic<T>, MaxChannels> published_ {};
};

/**
 * @class TransientFollower
 * @brief How sudden the signal is, 0..1, whatever its level.
 *
 * The loudest channel's analytic magnitude (a zero-latency Hilbert pair:
 * a steady tone reads its amplitude with no ripple) feeds two followers
 * with the same release (100 ms): one with a fast attack (0.5 ms), one with
 * a slow attack (30 ms). On a steady signal the two meet; at an onset the
 * fast one leaves
 * the slow one behind by as many decibels as the onset rises, for as long as
 * the slow attack takes. That distance, less half a decibel, over the range
 * (setRange(), 12 dB by default) is the raw value; the output takes it at
 * once, holds it (setHold(), 20 ms) and then falls with a release
 * (setRelease(), 80 ms). Both followers scale with the signal, so the value
 * does not depend on the level (under -120 dBFS it is 0).
 *
 * Threading: prepare() and reset() on the setup thread or the stream owner;
 * processBlock() and processSample() on the audio thread; the setters and
 * getValue() from any thread.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class TransientFollower
{
public:
    TransientFollower() noexcept { prepare(48000.0); }

    /** @brief Prepares for a sample rate (an invalid one is ignored) and clears. */
    void prepare(double sampleRate) noexcept
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        fs_ = sampleRate;
        const auto coeff = [this](double ms) { return std::exp(-1.0 / (ms * 0.001 * fs_)); };
        fastAttack_ = coeff(0.5);
        slowAttack_ = coeff(30.0);
        followRelease_ = coeff(100.0);
        reset();
    }

    /** @brief Clears the followers and the output. */
    void reset() noexcept
    {
        fast_ = 0.0;
        slow_ = 0.0;
        out_ = 0.0;
        holdLeft_ = 0;
        for (auto& h : analytic_) h.reset();
        value_.store(T(0), std::memory_order_relaxed);
    }

    /** @brief Range in dB that maps to the full value, 3 to 40 (default 12). */
    void setRange(T dB) noexcept
    {
        if (std::isfinite(dB)) range_.store(std::clamp(dB, T(3), T(40)), std::memory_order_relaxed);
    }
    /** @brief Time the value is held at a peak, 0 to 500 ms (default 20). */
    void setHold(T ms) noexcept
    {
        if (std::isfinite(ms)) holdMs_.store(std::clamp(ms, T(0), T(500)), std::memory_order_relaxed);
    }
    /** @brief Time constant of the fall after the hold, 1 to 2000 ms (default 80). */
    void setRelease(T ms) noexcept
    {
        if (std::isfinite(ms)) releaseMs_.store(std::clamp(ms, T(1), T(2000)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getRange() const noexcept { return range_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getHold() const noexcept { return holdMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }

    /** @brief The value after the last sample (any thread). */
    [[nodiscard]] T getValue() const noexcept { return value_.load(std::memory_order_relaxed); }

    /** @brief One sample (already the loudest channel's, or mono); returns the value. */
    [[nodiscard]] T processSample(T input) noexcept
    {
        const Timing t = timing();
        const double x = static_cast<double>(input);
        const T v = step(analytic_[0].magnitude(std::isfinite(x) ? x : 0.0), t);
        value_.store(v, std::memory_order_relaxed);
        return v;
    }

    /**
     * @brief Follows a block (the loudest channel per frame); the audio is
     *        not touched.
     * @param buffer The audio to follow (up to 16 channels are read).
     * @param out    Optional: receives the value per frame (one per sample of the block).
     */
    void processBlock(AudioBufferView<T> buffer, T* out = nullptr) noexcept
    {
        const Timing t = timing();
        const int nCh = std::min(buffer.getNumChannels(), kChannels), nS = buffer.getNumSamples();
        T v = value_.load(std::memory_order_relaxed);
        for (int i = 0; i < nS; ++i)
        {
            double top = 0.0;
            for (int ch = 0; ch < nCh; ++ch)
            {
                const double x = static_cast<double>(buffer.getChannel(ch)[i]);
                top = std::max(top, analytic_[static_cast<size_t>(ch)].magnitude(std::isfinite(x) ? x : 0.0));
            }
            v = step(top, t);
            if (out) out[i] = v;
        }
        value_.store(v, std::memory_order_relaxed);
    }

private:
    struct Timing
    {
        double range;
        int hold;
        double release;
    };

    [[nodiscard]] Timing timing() const noexcept
    {
        return { static_cast<double>(range_.load(std::memory_order_relaxed)),
                 static_cast<int>(static_cast<double>(holdMs_.load(std::memory_order_relaxed)) * 0.001 * fs_),
                 std::exp(-1.0 / (static_cast<double>(releaseMs_.load(std::memory_order_relaxed)) * 0.001 * fs_)) };
    }

    [[nodiscard]] T step(double x, const Timing& t) noexcept
    {
        const double level = std::isfinite(x) ? std::abs(x) : 0.0;
        fast_ = level + ((level > fast_) ? fastAttack_ : followRelease_) * (fast_ - level);
        slow_ = level + ((level > slow_) ? slowAttack_ : followRelease_) * (slow_ - level);
        double raw = 0.0;
        if (slow_ > 1e-6 && fast_ > 1e-6)
            raw = std::clamp((20.0 * std::log10(fast_ / slow_) - kDeadZoneDb) / t.range, 0.0, 1.0);
        else if (fast_ > 1e-6)
            raw = 1.0;
        if (fast_ < 1e-12) fast_ = 0.0;
        if (slow_ < 1e-12) slow_ = 0.0;
        if (raw >= out_)
        {
            out_ = raw;
            holdLeft_ = t.hold;
        }
        else if (holdLeft_ > 0)
        {
            --holdLeft_;
        }
        else
        {
            out_ = raw + t.release * (out_ - raw);
        }
        return static_cast<T>(out_);
    }

    static constexpr double kDeadZoneDb = 0.5;   ///< Distance a steady signal can leave between the two.
    static constexpr int kChannels = 16;         ///< Channels of a block that are read.

    std::array<HilbertIIR<double>, kChannels> analytic_ {};

    std::atomic<T> range_ { T(12) };
    std::atomic<T> holdMs_ { T(20) };
    std::atomic<T> releaseMs_ { T(80) };
    std::atomic<T> value_ { T(0) };

    double fs_ = 48000.0;
    double fastAttack_ = 0.0, slowAttack_ = 0.0, followRelease_ = 0.0;
    double fast_ = 0.0, slow_ = 0.0, out_ = 0.0;
    int holdLeft_ = 0;
};

} // namespace dspark
