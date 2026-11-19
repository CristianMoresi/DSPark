// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file BucketBrigade.h
 * @brief A bucket-brigade delay line: samples held in a chain of buckets
 *        moved by a clock whose rate sets the delay.
 *
 * A bucket-brigade device of N stages driven by a two-phase clock at f_clk
 * samples its input once per clock period and delivers each sample N / 2
 * periods later: the delay is N / (2 f_clk), so a shorter delay is a faster
 * clock, and a moving delay is a clock whose rate moves. Each sample read
 * out is held until the next one (sample and hold), so the device has its
 * own sampling rate, the clock's: what lies above half of it on the way in
 * folds back, and the held steps carry images of the signal around every
 * multiple of it on the way out. The circuits around such devices low-pass
 * their input and output for that reason (the anti-alias and the
 * reconstruction filters).
 *
 * This class models the device at the host's rate after the published
 * approach to these circuits: the clock is a phase that advances by
 * f_clk / fs per host sample; at each clock tick (several per sample when the
 * clock is faster than the host, or none when it is slower) the input,
 * interpolated linearly to the tick's time, enters the chain and the oldest
 * bucket leaves it; the output is the last bucket out, held, averaged over
 * each host sample (so the steps' images around the clock rate do not fold
 * into the band when the clock is faster than the host), then low-passed.
 * Its two filters are fourth-order Butterworth low-passes (input and
 * output) at a corner the caller sets, usually under half the slowest clock
 * the delay will need.
 *
 * Threading: prepare() on the setup thread (allocates); the rest on the
 * audio thread or the stream owner.
 *
 * Dependencies: Biquad.h, DspMath.h.
 */

#include "Biquad.h"
#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace dspark {

/**
 * @class BucketBrigade
 * @brief One bucket-brigade delay line (one channel).
 *
 * @tparam T Sample type (float or double); the chain and the filters run in
 *           double.
 */
template <FloatType T>
class BucketBrigade
{
public:
    static constexpr int kMinStages = 64;     ///< Smallest chain.
    static constexpr int kMaxStages = 8192;   ///< Largest chain.

    /**
     * @brief Allocates the chain and designs the filters.
     * @param sampleRate Host rate (invalid values are ignored).
     * @param stages     Stages of the device (rounded to an even count, 64 to 8192).
     * @param cornerHz   Corner of the input and output low-passes.
     */
    void prepare(double sampleRate, int stages, double cornerHz)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        fs_ = sampleRate;
        stages_ = std::clamp(stages / 2 * 2, kMinStages, kMaxStages);
        buckets_.assign(static_cast<size_t>(stages_ / 2), 0.0);
        setCorner(cornerHz);
        reset();
    }

    /** @brief Empties the chain and the filters. */
    void reset() noexcept
    {
        std::fill(buckets_.begin(), buckets_.end(), 0.0);
        head_ = 0;
        phase_ = 0.0;
        previousIn_ = 0.0;
        held_ = 0.0;
        for (auto& s : inState_) s = {};
        for (auto& s : outState_) s = {};
    }

    /** @brief Corner of the input and output low-passes, 200 Hz to 0.45 of the rate. */
    void setCorner(double hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        corner_ = std::clamp(hz, 200.0, 0.45 * fs_);
        static constexpr double kQ4[2] = { 0.5411961001461970, 1.3065629648763764 };
        for (int k = 0; k < 2; ++k)
        {
            in_[static_cast<size_t>(k)] = BiquadCoeffs::makeLowPass(fs_, corner_, kQ4[k]);
            out_[static_cast<size_t>(k)] = in_[static_cast<size_t>(k)];
        }
    }

    [[nodiscard]] int getStages() const noexcept { return stages_; }
    [[nodiscard]] double getCorner() const noexcept { return corner_; }

    /** @brief The clock rate a delay needs, in hertz. */
    [[nodiscard]] double clockFor(double delaySeconds) const noexcept
    {
        return static_cast<double>(stages_) / (2.0 * std::max(delaySeconds, 1e-5));
    }

    /**
     * @brief One host sample through the device.
     * @param input        The sample.
     * @param delaySeconds The delay the clock is set for at this sample
     *                     (N / (2 f_clk); at most 64 clock ticks per sample).
     * @return The delayed, held and filtered sample.
     */
    [[nodiscard]] T processSample(T input, double delaySeconds) noexcept
    {
        double x = std::isfinite(input) ? static_cast<double>(input) : 0.0;
        for (size_t k = 0; k < 2; ++k) x = section(in_[k], inState_[k], x);
        const double step = std::min(clockFor(delaySeconds) / fs_, 64.0);
        const double start = phase_;
        phase_ += step;
        const int size = static_cast<int>(buckets_.size());
        // Each tick between the previous sample and this one takes the input
        // at its own time, hands the oldest bucket to the output and moves on.
        // The output is the held value averaged over the host sample (the
        // integral of the steps): sampled at the host's instants, the steps'
        // images around the clock rate would fold into the band.
        double area = 0.0, last = 0.0;
        for (double tick = std::floor(start) + 1.0; tick <= phase_ && size > 0; tick += 1.0)
        {
            const double t = step > 0.0 ? (tick - start) / step : 1.0;
            const double sampled = previousIn_ + t * (x - previousIn_);
            area += held_ * (t - last);
            last = t;
            held_ = buckets_[static_cast<size_t>(head_)];
            buckets_[static_cast<size_t>(head_)] = sampled;
            head_ = head_ + 1 == size ? 0 : head_ + 1;
        }
        area += held_ * (1.0 - last);
        if (phase_ >= 1.0) phase_ -= std::floor(phase_);
        previousIn_ = x;
        double y = area;
        for (size_t k = 0; k < 2; ++k) y = section(out_[k], outState_[k], y);
        return static_cast<T>(y);
    }

private:
    struct State
    {
        double z1 = 0.0, z2 = 0.0;
    };

    static double section(const BiquadCoeffs& c, State& s, double x) noexcept
    {
        const double y = c.b0 * x + s.z1;
        s.z1 = c.b1 * x - c.a1 * y + s.z2;
        s.z2 = c.b2 * x - c.a2 * y;
        return y;
    }

    double fs_ = 48000.0, corner_ = 4000.0;
    int stages_ = 1024;
    std::vector<double> buckets_;
    int head_ = 0;
    double phase_ = 0.0, previousIn_ = 0.0, held_ = 0.0;
    std::array<BiquadCoeffs, 2> in_ {};
    std::array<BiquadCoeffs, 2> out_ {};
    std::array<State, 2> inState_ {}, outState_ {};
};

} // namespace dspark
