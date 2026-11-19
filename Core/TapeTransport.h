// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file TapeTransport.h
 * @brief A tape transport as a speed state: the delay of a moving tape, with
 *        the wow and flutter of its capstan and reels.
 *
 * The speed of the tape is a state relative to the nominal speed. A change of
 * the nominal speed (setSpeed()) leaves the tape running at the old speed,
 * which glides to the new one with inertia (one time constant to speed up,
 * another to slow down). The speed deviates by a smoothed noise, the
 * capstan's rotation (with some of its second harmonic) and the reel's
 * rotation, at rates that follow the speed of the moment, and by a scrape
 * band; the deviation is deeper at slower speeds (by the square root of the
 * ratio to 15 ips). The reproduce pitch is the speed of the moment: the read
 * point advances by one plus the deviation per sample, so the delay moves by
 * minus the deviation; a slow restoring term holds it at the head distance
 * over the speed, and its change per sample is bounded.
 *
 * An echo's delay time is set the same way (setTargetDelay()): the head
 * distance is fixed, so a delay of D samples is the speed nominal / D, which
 * the tape reaches with the same inertia (the pitch bends while it glides,
 * as on a tape echo whose speed is changed).
 *
 * Threading: everything on the audio thread or the stream owner (the class
 * holds no atomics: its owner publishes its settings).
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace dspark {

/**
 * @class TapeTransport
 * @brief The delay of a tape moved by a transport with inertia, wow and flutter.
 *
 * @tparam T Sample type of the owner (the state is double).
 */
template <FloatType T>
class TapeTransport
{
public:
    /** @brief The transport's constants. */
    struct Settings
    {
        double speedUpSeconds = 0.45;    ///< Inertia when speeding up.
        double slowDownSeconds = 0.7;    ///< ... and when slowing down.
        double restoreSeconds = 1.5;     ///< Restoring time of the delay to the head distance.
        double maxStep = 0.02;           ///< Largest delay change per sample (bounds the pitch).
        double reel = 0.0012;            ///< Speed deviations at full depth and 15 ips:
        double capstan = 0.0006;         ///< reel, capstan, its second harmonic's share,
        double capstanSecond = 0.3;      ///< smoothed noise (RMS) and scrape.
        double noise = 0.0004;
        double scrape = 0.003;
    };

    /**
     * @brief Sets the rate, the head distance (as a delay at the nominal
     *        speed), the nominal speed and the constants; resets.
     * @param sampleRate     Rate in hertz (invalid values are ignored).
     * @param nominalSamples The delay at the nominal speed, in samples.
     * @param ips            Nominal speed, inches per second.
     * @param settings       The constants.
     */
    void prepare(double sampleRate, double nominalSamples, double ips, const Settings& settings = Settings {}) noexcept
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        fs_ = sampleRate;
        s_ = settings;
        nominal_ = std::max(1.0, nominalSamples);
        ips_ = std::max(0.1, ips);
        const double dt = 1.0 / fs_;
        up_ = std::exp(-dt / s_.speedUpSeconds);
        down_ = std::exp(-dt / s_.slowDownSeconds);
        restore_ = dt / s_.restoreSeconds;
        noiseA_ = std::exp(-2.0 * std::numbers::pi * 3.0 * dt);
        scrapeA1_ = std::exp(-2.0 * std::numbers::pi * 90.0 * dt);
        scrapeA2_ = std::exp(-2.0 * std::numbers::pi * 40.0 * dt);
        {
            // RMS of a uniform noise in [-1, 1) through two such one-poles.
            const double a = noiseA_;
            const double gain = (1.0 - a) * (1.0 + a * a) / ((1.0 + a) * (1.0 + a) * (1.0 + a));
            noiseNorm_ = 1.0 / std::sqrt(std::max(1e-30, gain / 3.0));
        }
        lo_ = 3.0;
        hi_ = 1e12;
        reset();
    }

    /** @brief Speed at nominal, delay at the head distance, phases and noise from the start. */
    void reset() noexcept
    {
        speed_ = speedTarget_ = 1.0;
        delay_ = nominal_;
        capPhase_ = reelPhase_ = 0.0;
        noise1_ = noise2_ = scrape1_ = scrape2_ = 0.0;
        rng_ = 0x2C1B3C6Du;
    }

    /** @brief A new nominal speed: the tape glides to it from the speed it has. */
    void setSpeed(double ips) noexcept
    {
        if (!std::isfinite(ips) || !(ips > 0.0) || ips == ips_) return;
        speed_ *= ips_ / ips;
        ips_ = ips;
    }

    /** @brief A new nominal speed, taken at once (no glide). */
    void setSpeedNow(double ips) noexcept
    {
        if (!std::isfinite(ips) || !(ips > 0.0)) return;
        ips_ = ips;
        speed_ = speedTarget_ = 1.0;
    }

    /**
     * @brief A delay to glide to, in samples: the speed nominal / samples,
     *        reached with the transport's inertia (the head distance stays).
     */
    void setTargetDelay(double samples) noexcept
    {
        if (!std::isfinite(samples)) return;
        speedTarget_ = nominal_ / std::clamp(samples, std::max(lo_, 1e-3), hi_);
    }

    /** @brief A delay taken at once: the speed and the delay jump to it. */
    void setTargetDelayNow(double samples) noexcept
    {
        if (!std::isfinite(samples)) return;
        setTargetDelay(samples);
        speed_ = speedTarget_;
        delay_ = std::clamp(samples, lo_, hi_);
    }

    /** @brief The range the delay stays in, in samples (default from 3 up). */
    void setLimits(double lo, double hi) noexcept
    {
        lo_ = std::max(0.0, lo);
        hi_ = std::max(lo_, hi);
        delay_ = std::clamp(delay_, lo_, hi_);
    }

    /** @brief Starts the delay from a value (switching in from another reader). */
    void syncDelay(double samples) noexcept { delay_ = std::clamp(samples, lo_, hi_); }

    [[nodiscard]] double getDelay() const noexcept { return delay_; }
    [[nodiscard]] double getNominal() const noexcept { return nominal_; }
    [[nodiscard]] double getSpeedRatio() const noexcept { return speed_; }

    /**
     * @brief The delay for the next sample.
     * @param depth Depth of the deviation, 0 to 1 (the transport keeps moving at 0).
     */
    [[nodiscard]] double next(double depth) noexcept
    {
        speed_ = speedTarget_ + ((speed_ < speedTarget_) ? up_ : down_) * (speed_ - speedTarget_);
        if (std::abs(speed_ - speedTarget_) < 1e-12) speed_ = speedTarget_;
        const double ipsNow = ips_ * speed_;
        const double dt = 1.0 / fs_;
        capPhase_ += kCapstanHzPerIps * ipsNow * dt;
        if (capPhase_ >= 1.0) capPhase_ -= 1.0;
        reelPhase_ += kReelHzPerIps * ipsNow * dt;
        if (reelPhase_ >= 1.0) reelPhase_ -= 1.0;

        rng_ = rng_ * 1664525u + 1013904223u;
        const double n = static_cast<double>(rng_ >> 8) / 8388608.0 - 1.0;
        noise1_ = noiseA_ * noise1_ + (1.0 - noiseA_) * n;
        noise2_ = noiseA_ * noise2_ + (1.0 - noiseA_) * noise1_;
        scrape1_ = scrapeA1_ * scrape1_ + (1.0 - scrapeA1_) * n;
        scrape2_ = scrapeA2_ * scrape2_ + (1.0 - scrapeA2_) * n;

        double deviation = 0.0;
        if (depth > 0.0)
        {
            const double twoPi = 2.0 * std::numbers::pi;
            const double cap = twoPi * capPhase_;
            const double reel = s_.reel != 0.0 ? s_.reel * std::sin(twoPi * reelPhase_) : 0.0;
            deviation = depth * std::sqrt(15.0 / ips_)
                * (reel
                 + s_.capstan * (std::sin(cap) + s_.capstanSecond * std::sin(2.0 * cap))
                 + s_.noise * noiseNorm_ * noise2_
                 + s_.scrape * (scrape1_ - scrape2_));
        }
        const double target = nominal_ / speed_;
        const double step = std::clamp(-deviation + restore_ * (target - delay_), -s_.maxStep, s_.maxStep);
        delay_ = std::clamp(delay_ + step, lo_, hi_);
        return delay_;
    }

private:
    static constexpr double kCapstanHzPerIps = 1.0 / 3.0;   ///< Capstan rotation: 5 Hz at 15 ips.
    static constexpr double kReelHzPerIps = 0.04;           ///< Reel rotation: 0.6 Hz at 15 ips.

    Settings s_ {};
    double fs_ = 48000.0, nominal_ = 96.0, ips_ = 15.0;
    double up_ = 0.0, down_ = 0.0, restore_ = 0.0, noiseA_ = 0.0, noiseNorm_ = 1.0;
    double scrapeA1_ = 0.0, scrapeA2_ = 0.0;
    double lo_ = 3.0, hi_ = 1e12;
    double speed_ = 1.0, speedTarget_ = 1.0, delay_ = 96.0;
    double capPhase_ = 0.0, reelPhase_ = 0.0, noise1_ = 0.0, noise2_ = 0.0, scrape1_ = 0.0, scrape2_ = 0.0;
    uint32_t rng_ = 0x2C1B3C6Du;
};

} // namespace dspark
