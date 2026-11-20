// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TailShaper.h
 * @brief A dip under the dry signal and a relative gate for the wet signal of a reverb or a
 *        delay, driven by the dry signal.
 *
 * The dip lowers the wet signal while the dry one plays and lets it back
 * while the dry one rests. The dry level is followed (30 ms up, 400 ms down)
 * and read against its own recent peak (which rises at once and falls over
 * 10 s), so a phrase at any level dips it by the same amount: the wet signal
 * is lowered by the depth set times that share (a steady dry signal takes
 * the whole depth).
 *
 * The gate closes the wet signal once the dry one has stayed under a
 * threshold for the hold time. The threshold is relative to the recent
 * peak of a fast follower (1 ms up, 50 ms down); the hold is a window
 * in which any crossing keeps the gate open; the gate's gain is smoothed
 * twice, opening over 2 ms and closing over the release, so it opens without
 * a click and closes as a fade.
 *
 * With neither on, isActive() is false and next() returns exactly 1.
 *
 * Threading: prepare() and reset() on the setup thread or the stream owner;
 * sync() and next() on the audio thread; the setters and getters from any
 * thread (non-finite values are ignored).
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class TailShaper
 * @brief A dip and a relative gate driven by the dry signal.
 *
 * @tparam T Sample type (float or double); the state is double.
 */
template <FloatType T>
class TailShaper
{
public:
    TailShaper() noexcept { prepare(48000.0); }

    /** @brief Prepares for a sample rate (an invalid one is ignored) and clears. */
    void prepare(double sampleRate) noexcept
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        fs_ = sampleRate;
        dipUp_ = coeff(30.0);
        dipDown_ = coeff(400.0);
        gateUp_ = coeff(1.0);
        gateDown_ = coeff(50.0);
        refFall_ = coeff(10000.0);
        open_ = coeff(2.0);
        reset();
        sync();
    }

    /** @brief Clears the followers; the gate starts open and the dip released. */
    void reset() noexcept
    {
        dipEnv_ = gateEnv_ = dipRef_ = gateRef_ = 0.0;
        hold_ = 0;
        g1_ = g2_ = 1.0;
    }

    /** @brief Depth of the dip in dB, 0 to 48 (0, the default, is off). */
    void setDryDip(T dB) noexcept
    {
        if (std::isfinite(dB)) dipDb_.store(std::clamp(dB, T(0), T(48)), std::memory_order_relaxed);
    }

    /**
     * @brief Gate threshold in dB under the dry signal's recent peak, -80 to 0;
     *        below -80 the gate is off (the default).
     */
    void setGate(T thresholdDb) noexcept
    {
        if (std::isfinite(thresholdDb)) gateDb_.store(std::clamp(thresholdDb, T(-200), T(0)), std::memory_order_relaxed);
    }

    /** @brief Time the gate stays open after the last crossing, 0 to 4000 ms (default 100). */
    void setGateHold(T ms) noexcept
    {
        if (std::isfinite(ms)) holdMs_.store(std::clamp(ms, T(0), T(4000)), std::memory_order_relaxed);
    }

    /** @brief Time constant of the gate's closing, 5 to 4000 ms (default 150). */
    void setGateRelease(T ms) noexcept
    {
        if (std::isfinite(ms)) releaseMs_.store(std::clamp(ms, T(5), T(4000)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getDryDip() const noexcept { return dipDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getGate() const noexcept { return gateDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getGateHold() const noexcept { return holdMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getGateRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }

    /** @brief Takes the settings for the next block (audio thread). */
    void sync() noexcept
    {
        dip_ = static_cast<double>(dipDb_.load(std::memory_order_relaxed)) * (2.302585092994046 / 20.0);
        const double gateDb = static_cast<double>(gateDb_.load(std::memory_order_relaxed));
        gateOn_ = gateDb >= -80.0;
        gateLin_ = std::pow(10.0, gateDb / 20.0);
        holdSamples_ = static_cast<int>(static_cast<double>(holdMs_.load(std::memory_order_relaxed)) * 0.001 * fs_);
        release_ = coeff(static_cast<double>(releaseMs_.load(std::memory_order_relaxed)));
    }

    /** @brief Whether next() can return anything but 1 (as of the last sync()). */
    [[nodiscard]] bool isActive() const noexcept { return dip_ > 0.0 || gateOn_ || g2_ != 1.0; }

    /**
     * @brief One sample: the gain for the wet signal.
     * @param dryLevel The dry signal's level at this sample (the largest
     *                 magnitude of its channels).
     */
    [[nodiscard]] T next(T dryLevel) noexcept
    {
        const double x = std::isfinite(dryLevel) ? std::abs(static_cast<double>(dryLevel)) : 0.0;
        dipEnv_ = x + (x > dipEnv_ ? dipUp_ : dipDown_) * (dipEnv_ - x);
        gateEnv_ = x + (x > gateEnv_ ? gateUp_ : gateDown_) * (gateEnv_ - x);
        // Each follower against its own recent peak.
        dipRef_ = dipEnv_ > dipRef_ ? dipEnv_ : dipRef_ * refFall_;
        gateRef_ = gateEnv_ > gateRef_ ? gateEnv_ : gateRef_ * refFall_;
        if (dipRef_ < 1e-12) dipRef_ = 0.0;
        if (gateRef_ < 1e-12) gateRef_ = 0.0;
        double gain = 1.0;
        if (dip_ > 0.0 && dipRef_ > 0.0)
            gain = std::exp(-dip_ * std::min(1.0, dipEnv_ / dipRef_));
        double target = 1.0;
        if (gateOn_)
        {
            if (gateRef_ > 0.0 && gateEnv_ >= gateRef_ * gateLin_) hold_ = holdSamples_ + 1;
            if (hold_ > 0) --hold_;
            else target = 0.0;
        }
        if (g1_ != target || g2_ != target)
        {
            const double a = target > g1_ ? open_ : release_;
            g1_ = target + a * (g1_ - target);
            g2_ = g1_ + a * (g2_ - g1_);
            if (std::abs(g2_ - target) < 1e-9 && std::abs(g1_ - target) < 1e-9) g1_ = g2_ = target;
        }
        return static_cast<T>(gain * g2_);
    }

private:
    [[nodiscard]] double coeff(double ms) const noexcept { return std::exp(-1.0 / (ms * 0.001 * fs_)); }

    std::atomic<T> dipDb_ { T(0) }, gateDb_ { T(-200) }, holdMs_ { T(100) }, releaseMs_ { T(150) };

    double fs_ = 48000.0;
    double dipUp_ = 0.0, dipDown_ = 0.0, gateUp_ = 0.0, gateDown_ = 0.0, refFall_ = 0.0, open_ = 0.0, release_ = 0.0;
    double dip_ = 0.0, gateLin_ = 0.0;
    bool gateOn_ = false;
    int holdSamples_ = 0, hold_ = 0;
    double dipEnv_ = 0.0, gateEnv_ = 0.0, dipRef_ = 0.0, gateRef_ = 0.0, g1_ = 1.0, g2_ = 1.0;
};

} // namespace dspark
