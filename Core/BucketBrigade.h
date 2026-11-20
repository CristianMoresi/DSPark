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
 * When the clock runs near or below the host's rate, the held steps carry
 * images at the clock rate minus the signal, which in the device lie above
 * the band and are taken out by its reconstruction filter; at the host's
 * rate they fold into the band first. prepare() can therefore run the line at
 * twice the host's rate (the input interpolated half way between samples,
 * the output read every other step, after its low-pass): the images are then
 * filtered as in the device, at twice the cost.
 *
 * The model has two artifacts of its own, which the device does not have:
 * the input read between steps by a straight line carries images at
 * multiples of the step rate less the signal, which the clock's ticks fold
 * into the band; and the held output, averaged over each step, lets the
 * images of the clock's higher harmonics (k f_clk + f) fold back at the step
 * rate. setPrecision(Precision::High) reads the input by a quintic through
 * six steps (Lagrange; its images fall as the sixth power of the frequency
 * over the step rate, at two steps more delay) and weights the held output by
 * a quadratic B-spline over three steps (those images fall three times as
 * fast in decibels, at one step more delay and a droop of sinc cubed of the
 * frequency over the line's rate in place of sinc: 0.3 dB at 7.8 kHz at
 * 96 kHz).
 *
 * Two lines fed with the same signal and the same corner (the two taps of a
 * chorus) can share the input low-pass through processPair().
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

    /** @brief How the line reads its input and its output (setPrecision()). */
    enum class Precision
    {
        Standard,  ///< Input by a straight line between steps, output by its average over the step (default).
        High       ///< Input by a quintic through six steps, output by a quadratic B-spline over three (see the file notes).
    };

    /**
     * @brief Allocates the chain and designs the filters.
     * @param sampleRate Host rate (invalid values are ignored).
     * @param stages     Stages of the device (rounded to an even count, 64 to 8192).
     * @param cornerHz   Corner of the input and output low-passes.
     * @param factor     1 runs the line at the host's rate; 2 at twice it (see
     *                   the file notes); other values read as 1.
     */
    void prepare(double sampleRate, int stages, double cornerHz, int factor = 1)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        fs_ = sampleRate;
        factor_ = factor == 2 ? 2 : 1;
        rate_ = fs_ * static_cast<double>(factor_);
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
        lastInput_ = 0.0;
        rise1_ = rise2_ = middle1_ = 0.0;
        history_ = {};
        for (auto& s : inState_) s = {};
        for (auto& s : outState_) s = {};
    }

    /** @brief Corner of the input and output low-passes, 200 Hz to 0.45 of the host's rate. */
    void setCorner(double hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        corner_ = std::clamp(hz, 200.0, 0.45 * fs_);
        static constexpr double kQ4[2] = { 0.5411961001461970, 1.3065629648763764 };
        for (int k = 0; k < 2; ++k)
        {
            in_[static_cast<size_t>(k)] = BiquadCoeffs::makeLowPass(rate_, corner_, kQ4[k]);
            out_[static_cast<size_t>(k)] = in_[static_cast<size_t>(k)];
        }
    }

    [[nodiscard]] int getStages() const noexcept { return stages_; }
    [[nodiscard]] double getCorner() const noexcept { return corner_; }
    /** @return 1, or 2 when the line runs at twice the host's rate. */
    [[nodiscard]] int getFactor() const noexcept { return factor_; }

    /** @brief How the line reads its input and its output (out-of-range values read as Standard); call reset() after a change. */
    void setPrecision(Precision precision) noexcept
    {
        precision_ = precision == Precision::High ? Precision::High : Precision::Standard;
    }
    /** @return How the line reads its input and its output. */
    [[nodiscard]] Precision getPrecision() const noexcept { return precision_; }

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
        const double x = std::isfinite(input) ? static_cast<double>(input) : 0.0;
        if (factor_ == 2) (void)chain(lowPass(0.5 * (lastInput_ + x)), delaySeconds);
        lastInput_ = x;
        return static_cast<T>(chain(lowPass(x), delaySeconds));
    }

    /**
     * @brief One host sample through this line and a twin, sharing this
     *        line's input low-pass (the twin's own is not used).
     *
     * The twin must have been prepared like this line (rate, stages, factor)
     * and carry the same corner; each line then gives what processSample()
     * would give it on the same input.
     *
     * @param twin       The other line.
     * @param input      The sample, the same for both.
     * @param delayThis  The delay of this line at this sample.
     * @param delayTwin  The delay of the twin at this sample.
     * @param outThis    Receives this line's output.
     * @param outTwin    Receives the twin's output.
     */
    void processPair(BucketBrigade& twin, T input, double delayThis, double delayTwin,
                     double& outThis, double& outTwin) noexcept
    {
        const double x = std::isfinite(input) ? static_cast<double>(input) : 0.0;
        if (factor_ == 2)
        {
            const double mid = lowPass(0.5 * (lastInput_ + x));
            (void)chain(mid, delayThis);
            (void)twin.chain(mid, delayTwin);
        }
        lastInput_ = x;
        const double filtered = lowPass(x);
        outThis = static_cast<double>(static_cast<T>(chain(filtered, delayThis)));
        outTwin = static_cast<double>(static_cast<T>(twin.chain(filtered, delayTwin)));
    }

private:
    /** The input low-pass on one step of the line's rate. */
    double lowPass(double x) noexcept
    {
        for (size_t k = 0; k < 2; ++k) x = section(in_[k], inState_[k], x);
        return x;
    }

    /**
     * The input at fraction t of the step two before this one (high
     * precision): Lagrange through the six steps around it (nodes -2 to 3;
     * x is the newest).
     */
    [[nodiscard]] double quintic(double t, double x) const noexcept
    {
        const double a = t + 2.0, b = t + 1.0, e = t - 1.0, f = t - 2.0, g = t - 3.0;
        const double ab = a * b, efg = e * f * g;
        return history_[0] * (b * t * efg) * (-1.0 / 120.0)
             + history_[1] * (a * t * efg) * (1.0 / 24.0)
             + history_[2] * (ab * efg) * (-1.0 / 12.0)
             + history_[3] * (ab * t * f * g) * (1.0 / 12.0)
             + history_[4] * (ab * t * e * g) * (-1.0 / 24.0)
             + x * (ab * t * e * f) * (1.0 / 120.0);
    }

    /** One step of the line's rate: the clock's ticks, the held output, its low-pass. */
    double chain(double x, double delaySeconds) noexcept
    {
        const double step = std::min(clockFor(delaySeconds) / rate_, 64.0);
        const double start = phase_;
        phase_ += step;
        const int size = static_cast<int>(buckets_.size());
        // Each tick between the previous step and this one takes the input at
        // its own time, hands the oldest bucket to the output and moves on.
        // The output is the held value averaged over the step (the integral
        // of the steps): sampled at the step's instants, the steps' images
        // around the clock rate would fold into the band.
        // High precision: the ticks of this step read the input of the step
        // two before, by a quintic through the six steps around it, and the
        // output is the held value weighted by a quadratic B-spline over the
        // last three steps: each stretch of this step adds to the spline's
        // falling piece now ((1 - u)^2 / 2), its middle piece one step later
        // (1/2 + u - u^2) and its rising piece two steps later (u^2 / 2).
        const bool high = precision_ == Precision::High;
        double area = 0.0, last = 0.0, rise = 0.0, middle = 0.0, fall = 0.0;
        const auto weigh = [&](double from, double to) noexcept {
            const double d1 = to - from, d2 = 0.5 * (to * to - from * from),
                         d3 = (to * to * to - from * from * from) * (1.0 / 3.0);
            rise += held_ * (0.5 * d3);
            middle += held_ * (0.5 * d1 + d2 - d3);
            fall += held_ * (0.5 * (d1 - 2.0 * d2 + d3));
        };
        for (double tick = std::floor(start) + 1.0; tick <= phase_ && size > 0; tick += 1.0)
        {
            const double t = step > 0.0 ? (tick - start) / step : 1.0;
            const double sampled = high ? quintic(t, x) : previousIn_ + t * (x - previousIn_);
            area += held_ * (t - last);
            if (high) weigh(last, t);
            last = t;
            held_ = buckets_[static_cast<size_t>(head_)];
            buckets_[static_cast<size_t>(head_)] = sampled;
            head_ = head_ + 1 == size ? 0 : head_ + 1;
        }
        area += held_ * (1.0 - last);
        if (phase_ >= 1.0) phase_ -= std::floor(phase_);
        previousIn_ = x;
        double y = area;
        if (high)
        {
            weigh(last, 1.0);
            y = rise2_ + middle1_ + fall;
            rise2_ = rise1_;
            rise1_ = rise;
            middle1_ = middle;
            history_ = { history_[1], history_[2], history_[3], history_[4], x };
        }
        for (size_t k = 0; k < 2; ++k) y = section(out_[k], outState_[k], y);
        return y;
    }

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

    double fs_ = 48000.0, rate_ = 48000.0, corner_ = 4000.0;
    int stages_ = 1024, factor_ = 1;
    std::vector<double> buckets_;
    int head_ = 0;
    double phase_ = 0.0, previousIn_ = 0.0, held_ = 0.0, lastInput_ = 0.0;
    double rise1_ = 0.0, rise2_ = 0.0, middle1_ = 0.0;   ///< High precision: the spline's pieces still to come.
    std::array<double, 5> history_ {};   ///< High precision: the five steps of input before this one.
    Precision precision_ = Precision::Standard;
    std::array<BiquadCoeffs, 2> in_ {};
    std::array<BiquadCoeffs, 2> out_ {};
    std::array<State, 2> inState_ {}, outState_ {};
};

} // namespace dspark
