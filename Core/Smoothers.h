// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>

/**
 * @namespace Smoothers
 * @brief A collection of real-time safe smoothing filters for parameter interpolation in audio.
 *
 * Standalone: no external dependencies beyond the C++ standard library.
 *
 * These classes provide various smoothing techniques to prevent artifacts like zipper noise
 * or clicks during parameter changes. All smoothers are designed for use in the audio thread:
 * lock-free, no dynamic allocations, and noexcept methods. The state structs are kept
 * compact (no over-alignment) so banks of smoothers pack tightly into cache lines.
 *
 * Threading: a smoother is single-threaded by design. It lives inside a processor
 * and every method is meant to be called from that processor's processing thread
 * (publish control-thread changes through your own atomics, as the effects do).
 *
 * Common API:
 * - reset(double sampleRate, float timeConstantMilliseconds, float initialValue): Configure.
 * - setTargetValue(float newTarget): Set the new target value to smooth towards.
 * - getNextValue(): Get the next smoothed value (call per sample).
 * - getCurrentValue(): Get the current smoothed value without advancing.
 * - getTargetValue(): Get the current target value.
 * - isSmoothing(): Check if still smoothing (abs(current - target) > epsilon).
 * - skip(): Instantly set current to target (bypass smoothing).
 */
namespace dspark {
namespace Smoothers
{

namespace Constants
{
    static constexpr float pi      = dspark::pi<float>;
    static constexpr float twoPi   = dspark::twoPi<float>;
    static constexpr float sqrt2   = 1.41421356237309504880f;
} // namespace Constants

//==============================================================================

/**
 * @class LinearSmoother
 * @brief Linear ramp smoother for predictable, uniform interpolation.
 *
 * Use for: Gains, mix/dry-wet, pans, or most faders where exact timing is needed.
 *
 * Internal state is double precision (the public API stays float): a float
 * accumulator rounded every step and drifted, so a 5 s ramp at 192 kHz ended
 * with a 1.3% jump onto the target instead of its 1e-6 step.
 */
struct LinearSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float rampTimeMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(current); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }

    void setCurrentAndTargetValue(float value) noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

    /**
     * @brief Block processing for SIMD optimization.
     * @param buffer Pointer to the audio block float array.
     * @param numSamples Number of samples to process.
     * @param multiply If true, acts as a gain multiplier (buffer[i] *= val). Otherwise, offsets (buffer[i] += val).
     */
    void processBlock(float* buffer, int numSamples, bool multiply = false) noexcept;

private:
    double current    = 0.0;
    double target     = 0.0;
    double step       = 0.0;
    int   stepsToGo   = 0;
    int   totalSteps  = 0;
};

/**
 * @class ExponentialSmoother
 * @brief Exponential (multiplicative) smoother for natural, perceptual responses.
 *
 * Use for: Filter cutoffs, frequencies, volumes in dB.
 *
 * Internal state and coefficient are double precision (the public API stays
 * float). In float the per-step coefficient rounds by up to 1e-7, a large
 * fraction of log(ratio)/steps for long ramps: a delay time smoothed from
 * 96000 to 91200 samples over 200 ms (Delay's default smoother) ended with a
 * 19.8-sample jump instead of its 0.5-sample step.
 *
 * @warning Geometric smoothing is only defined for same-sign transitions. If
 * the new target's sign differs from the current value (ratio <= 0), or the
 * current value is zero, the smoother snaps instantly to the target: an
 * audible step. Keep this class for strictly-positive parameters (Hz, linear
 * gain); use LinearSmoother or OnePoleSmoother for bipolar values (pan,
 * offsets).
 */
struct ExponentialSmoother
{
public:
    static constexpr float epsilon = 1e-10f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 1.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(current); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }

    void setCurrentAndTargetValue(float value) noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    double current   = 1.0;
    double target    = 1.0;
    double coeff     = 0.0;
    int   stepsToGo  = 0;
    int   totalSteps = 0;
};

/**
 * @class OnePoleSmoother
 * @brief Authentic one-pole exponential IIR low-pass smoother.
 *
 * Implements analog-style smoothing. Anti-denormal protected.
 *
 * Internal state is double precision: in float the recursion stalls once
 * coeff * delta rounds back to the same value, parking the output around
 * ulp(target) / (1 - coeff) away from the target (about 1e-5 for a 10 ms
 * tau at 48 kHz and unity-scale targets), outside the settle window, so
 * isSmoothing() would never turn off. The public API stays float.
 */
struct OnePoleSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(current); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    double coeff   = 0.0;
    double current = 0.0;
    double target  = 0.0;
};

/**
 * @class MultiPoleSmoother
 * @brief Templated cascaded multi-pole smoother for steeper roll-off.
 */
template <std::size_t N>
struct MultiPoleSmoother
{
public:
    static_assert(N >= 1, "MultiPoleSmoother needs at least one pole");

    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return poles.back().getCurrentValue(); }
    [[nodiscard]] float getTargetValue() const noexcept { return poles.front().getTargetValue(); }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    std::array<OnePoleSmoother, N> poles;
};

/**
 * @class AsymmetricSmoother
 * @brief Smoother with asymmetric attack/release times.
 *
 * The attack time applies while the value rises towards the target and the
 * release time while it falls, following the envelope-follower convention.
 */
struct AsymmetricSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float attackMilliseconds, float releaseMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(current); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    // Double for the same reason as OnePoleSmoother: float recursion stalls
    // short of the target and the settle check would never fire.
    double attackCoeff  = 0.0;
    double releaseCoeff = 0.0;
    double current      = 0.0;
    double target       = 0.0;
};

/**
 * @class SlewLimiter
 * @brief Rate limiter to cap maximum change per sample.
 *
 * Internal state is double precision (the public API stays float): in float
 * a step below half an ulp of the value was lost, and a limiter at 10000 with
 * a 20 units/s rate at 48 kHz never moved.
 */
struct SlewLimiter
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float maxRatePerSecond, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(current); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    double maxDelta = 0.0;
    double current  = 0.0;
    double target   = 0.0;
};

/**
 * @class StateVariableSmoother
 * @brief Second-order state variable filter (SVF) smoother (TPT implementation).
 *
 * Internal state is double precision: smoothing time constants put the
 * cutoff a few Hz above DC, where g = tan(pi*fc/fs) is ~1e-4 and float
 * recursion stalls several 1e-5 short of the target (the settle check
 * would then never fire). The public API stays float.
 */
struct StateVariableSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float q = 0.707f, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(lowpass); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

    [[nodiscard]] float getBandPassOutput() const noexcept { return static_cast<float>(bandpass); }
    [[nodiscard]] float getHighPassOutput() const noexcept { return static_cast<float>(highpass); }

private:
    double v1 = 0.0;
    double v2 = 0.0;
    double g  = 0.0;
    double k  = 0.0;
    double a1 = 0.0;
    double a2 = 0.0;
    double a3 = 0.0;
    double target   = 0.0;
    double lowpass  = 0.0;
    double bandpass = 0.0;
    double highpass = 0.0;
};

/**
 * @class ButterworthSmoother
 * @brief Butterworth low-pass smoother for maximally flat response.
 *
 * Internal state is double precision. This is not optional here: smoothing
 * time constants put the cutoff a few Hz above DC, where the TDF-II biquad
 * has b0 ~ 1e-8; in float the state recursion loses the tiny increments to
 * rounding and the step response stalls far short of the target. The
 * public API stays float.
 */
struct ButterworthSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(lastOut); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    double b0 = 0.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;
    double s1 = 0.0, s2 = 0.0;
    double target   = 0.0;
    double lastOut  = 0.0;
    double prevOut  = 0.0; ///< One-sample history so isSmoothing() sees motion, not just error.
    double velEps   = 0.0; ///< Per-sample velocity of an epsilon-amplitude residual at fc.
};

/**
 * @class CriticallyDampedSmoother
 * @brief Critically damped smoother (no overshoot, exact Q=0.5).
 */
struct CriticallyDampedSmoother : public StateVariableSmoother
{
public:
    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
};

/**
 * @class HermiteSmoother
 * @brief Cubic glide that lands exactly on the target after a fixed time,
 *        with a continuous slope.
 *
 * Each new target starts a cubic Hermite segment from the current value and
 * velocity to the target, arriving with zero velocity after exactly the ramp
 * time: like LinearSmoother it settles in a known number of samples (and
 * isSmoothing() then turns off exactly), but the value never has a corner,
 * at the start, at the end, or when retargeted mid-glide. From rest the path
 * is the smoothstep 3u^2 - 2u^3: monotonic, with its 10-90% rise at 0.61 of
 * the ramp time. A retarget mid-glide keeps the current velocity, clipped
 * to the monotone limit (three times the remaining distance per ramp) so the
 * glide never passes the new target.
 *
 * Use for: filter cutoffs (in log frequency), any parameter whose glide
 * should end at a known time without the long exponential tail.
 *
 * Internal state is double precision; the public API stays float.
 */
struct HermiteSmoother
{
public:
    void reset(double sampleRate, float rampTimeMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;

    [[nodiscard]] float getCurrentValue() const noexcept { return static_cast<float>(current); }
    [[nodiscard]] float getTargetValue() const noexcept { return static_cast<float>(target); }
    [[nodiscard]] bool isSmoothing() const noexcept { return stepsDone < totalSteps; }
    void skip() noexcept;

private:
    double current = 0.0, target = 0.0;
    double start = 0.0;      ///< Segment start value.
    double slope0 = 0.0;     ///< Segment start velocity, in value per segment.
    double invSteps = 1.0;
    int totalSteps = 0;      ///< Segment length in samples (0 = jump at once).
    int stepsDone = 0;       ///< Samples into the segment; == totalSteps when settled.
    int rampSteps = 0;       ///< Configured ramp length in samples.
};

}  // namespace Smoothers

//==============================================================================
// Inline definitions
//==============================================================================

// --- LinearSmoother ---

inline void Smoothers::LinearSmoother::reset(double sampleRate, float rampTimeMilliseconds, float initialValue) noexcept
{
    totalSteps = static_cast<int>(sampleRate * rampTimeMilliseconds / 1000.0);
    stepsToGo  = 0;
    step       = 0.0f;
    current    = initialValue;
    target     = initialValue;
}

inline void Smoothers::LinearSmoother::setTargetValue(float newTarget) noexcept
{
    if (newTarget == target) return;
    target = newTarget;
    stepsToGo = totalSteps;
    if (stepsToGo > 0)
        step = (target - current) / static_cast<double>(stepsToGo);
    else
        current = target;
}

inline float Smoothers::LinearSmoother::getNextValue() noexcept
{
    if (stepsToGo <= 0) return static_cast<float>(current);
    current += step;
    --stepsToGo;
    if (stepsToGo == 0) current = target;
    return static_cast<float>(current);
}

inline void Smoothers::LinearSmoother::setCurrentAndTargetValue(float value) noexcept
{
    current   = value;
    target    = value;
    step      = 0.0;
    stepsToGo = 0;
}

inline bool Smoothers::LinearSmoother::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::LinearSmoother::skip() noexcept
{
    current   = target;
    step      = 0.0;
    stepsToGo = 0;
}

inline void Smoothers::LinearSmoother::processBlock(float* buffer, int numSamples, bool multiply) noexcept
{
    // Split the ramping section from the settled remainder so each loop is
    // trivially auto-vectorizable (the compiler unswitches the loop-invariant
    // multiply flag out of both bodies).
    int stepsToProcess = std::min(numSamples, stepsToGo);
    int i = 0;

    // Ramping section
    for (; i < stepsToProcess; ++i)
    {
        current += step;
        const float value = static_cast<float>(current);
        buffer[i] = multiply ? buffer[i] * value : buffer[i] + value;
    }

    stepsToGo -= stepsToProcess;
    if (stepsToGo == 0) current = target;

    // Settled section
    for (; i < numSamples; ++i)
    {
        const float value = static_cast<float>(target);
        buffer[i] = multiply ? buffer[i] * value : buffer[i] + value;
    }
}

// --- ExponentialSmoother ---

inline void Smoothers::ExponentialSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    totalSteps = static_cast<int>(sampleRate * timeConstantMilliseconds / 1000.0);
    stepsToGo  = 0;
    coeff      = 1.0;
    current    = initialValue;
    target     = initialValue;
}

inline void Smoothers::ExponentialSmoother::setTargetValue(float newTarget) noexcept
{
    if (std::abs(newTarget) < epsilon)
        newTarget = (newTarget < 0.0f) ? -epsilon : epsilon;

    if (newTarget == target) return;
    target = newTarget;

    if (totalSteps > 0 && std::abs(current) > epsilon)
    {
        const double eps = static_cast<double>(epsilon);
        const double safeCur = (current > 0.0) ? std::max(current, eps) : std::min(current, -eps);
        const double ratio = target / safeCur;
        if (ratio > 0.0)
        {
            stepsToGo = totalSteps;
            coeff = std::exp(std::log(ratio) / static_cast<double>(stepsToGo));
            return;
        }
    }

    // Snap. The ramp must also stop here: leaving stepsToGo armed replays the
    // previous ramp's coefficient, dragging the value away from the target
    // for a whole ramp and then jumping back (a sign flip after a 1 -> 2 ramp
    // went -1 -> -1.997 -> -1).
    current   = target;
    coeff     = 1.0;
    stepsToGo = 0;
}

inline float Smoothers::ExponentialSmoother::getNextValue() noexcept
{
    if (stepsToGo <= 0) return static_cast<float>(current);
    current *= coeff;
    --stepsToGo;
    if (stepsToGo == 0) current = target;
    return static_cast<float>(current);
}

inline void Smoothers::ExponentialSmoother::setCurrentAndTargetValue(float value) noexcept
{
    current   = value;
    target    = value;
    coeff     = 1.0;
    stepsToGo = 0;
}

inline bool Smoothers::ExponentialSmoother::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::ExponentialSmoother::skip() noexcept
{
    current   = target;
    coeff     = 1.0;
    stepsToGo = 0;
}

// --- OnePoleSmoother ---

inline void Smoothers::OnePoleSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    const double timeConstantSeconds = static_cast<double>(timeConstantMilliseconds) / 1000.0;
    const double tau = sampleRate * timeConstantSeconds;
    coeff = tau > 0.0 ? std::exp(-1.0 / tau) : 0.0;
    current = initialValue;
    target = initialValue;
}

inline void Smoothers::OnePoleSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::OnePoleSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return static_cast<float>(target); } // Anti-denormal check
    current = target + coeff * (current - target);
    return static_cast<float>(current);
}

inline bool Smoothers::OnePoleSmoother::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::OnePoleSmoother::skip() noexcept
{
    current = target;
}

// --- MultiPoleSmoother ---

template <std::size_t N>
inline void Smoothers::MultiPoleSmoother<N>::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    for (auto& pole : poles)
        pole.reset(sampleRate, timeConstantMilliseconds, initialValue);
}

template <std::size_t N>
inline void Smoothers::MultiPoleSmoother<N>::setTargetValue(float newTarget) noexcept
{
    poles[0].setTargetValue(newTarget);
}

template <std::size_t N>
inline float Smoothers::MultiPoleSmoother<N>::getNextValue() noexcept
{
    float val = poles[0].getNextValue();
    for (std::size_t i = 1; i < N; ++i)
    {
        poles[i].setTargetValue(val);
        val = poles[i].getNextValue();
    }
    return val;
}

template <std::size_t N>
inline bool Smoothers::MultiPoleSmoother<N>::isSmoothing() const noexcept
{
    // Compare the chain OUTPUT against the GLOBAL target (pole 0's target).
    // Asking the last pole alone is wrong on both ends: right after
    // setTargetValue() only pole 0 knows the new target (the last pole would
    // report "settled" before the ramp even starts), and mid-ramp each pole
    // sits close to its neighbour while the whole chain is far from the goal.
    return std::abs(poles.back().getCurrentValue()
                    - poles.front().getTargetValue()) > epsilon;
}

template <std::size_t N>
inline void Smoothers::MultiPoleSmoother<N>::skip() noexcept
{
    // Propagate the global target down the chain first: inner poles still
    // chase the PREVIOUS output of their predecessor, so skipping them in
    // place would freeze the chain mid-way instead of landing on the target.
    const float t = poles.front().getTargetValue();
    for (auto& pole : poles)
    {
        pole.setTargetValue(t);
        pole.skip();
    }
}

// --- AsymmetricSmoother ---

inline void Smoothers::AsymmetricSmoother::reset(double sampleRate, float attackMilliseconds, float releaseMilliseconds, float initialValue) noexcept
{
    const double attackSeconds = static_cast<double>(attackMilliseconds) / 1000.0;
    const double releaseSeconds = static_cast<double>(releaseMilliseconds) / 1000.0;
    attackCoeff = attackSeconds > 0.0 ? std::exp(-1.0 / (sampleRate * attackSeconds)) : 0.0;
    releaseCoeff = releaseSeconds > 0.0 ? std::exp(-1.0 / (sampleRate * releaseSeconds)) : 0.0;
    current = initialValue;
    target = initialValue;
}

inline void Smoothers::AsymmetricSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::AsymmetricSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return static_cast<float>(target); } // Anti-denormal check
    const double c = (target > current) ? attackCoeff : releaseCoeff;
    current = target + c * (current - target);
    return static_cast<float>(current);
}

inline bool Smoothers::AsymmetricSmoother::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::AsymmetricSmoother::skip() noexcept
{
    current = target;
}

// --- SlewLimiter ---

inline void Smoothers::SlewLimiter::reset(double sampleRate, float maxRatePerSecond, float initialValue) noexcept
{
    // Clamp at zero: a negative rate would flip the clamp bounds below
    // (std::clamp with lo > hi is undefined behaviour). Zero legitimately
    // means "frozen": the value is not allowed to move at all.
    maxDelta = std::max(0.0, static_cast<double>(maxRatePerSecond) / sampleRate);
    current = initialValue;
    target = initialValue;
}

inline void Smoothers::SlewLimiter::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::SlewLimiter::getNextValue() noexcept
{
    double delta = target - current;
    delta = std::clamp(delta, -maxDelta, maxDelta);
    current += delta;
    return static_cast<float>(current);
}

inline bool Smoothers::SlewLimiter::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::SlewLimiter::skip() noexcept
{
    current = target;
}

// --- StateVariableSmoother ---

inline void Smoothers::StateVariableSmoother::reset(double sampleRate, float timeConstantMilliseconds, float q, float initialValue) noexcept
{
    const double timeConstantSeconds = static_cast<double>(timeConstantMilliseconds) / 1000.0;
    const double fs = sampleRate;
    // A zero time constant means "instantaneous", not "frozen": map it to the
    // fastest usable cutoff (fc = 0 would zero the coefficients and the
    // output would never move again). Clamp below Nyquist so tan(pi*fc/fs)
    // stays finite/stable for very small time constants (otherwise the TPT
    // prewarp blows up near fs/2).
    double fc = timeConstantSeconds > 1e-9 ? 1.0 / (static_cast<double>(Constants::twoPi) * timeConstantSeconds) : fs;
    fc = std::min(fc, fs * 0.49);

    g = std::tan(dspark::pi<double> * fc / fs);
    k = 1.0 / static_cast<double>(std::max(q, 0.01f)); // TPT damping uses k = 1/Q.

    a1 = 1.0 / (1.0 + g * (g + k));
    a2 = g * a1;
    a3 = g * a2;

    v1 = 0.0;
    v2 = initialValue;
    target = initialValue;
    lowpass = initialValue;
    bandpass = 0.0;
    highpass = 0.0;
}

inline void Smoothers::StateVariableSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::StateVariableSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return static_cast<float>(target); } // Anti-denormal check

    const double v0 = target;
    const double v3 = v0 - v2;
    const double v1_new = a1 * v1 + a2 * v3;
    const double v2_new = v2 + a2 * v1 + a3 * v3;
    v1 = 2.0 * v1_new - v1;
    v2 = 2.0 * v2_new - v2;

    lowpass = v2_new;
    bandpass = v1_new;
    highpass = v0 - k * v1_new - v2_new;

    return static_cast<float>(lowpass);
}

inline bool Smoothers::StateVariableSmoother::isSmoothing() const noexcept
{
    // The bandpass term is the filter's velocity. Without it, a Q > 0.5
    // response crossing the target (overshoot) can momentarily satisfy
    // |lowpass - target| < epsilon at full speed, and the anti-denormal
    // check in getNextValue() would snap mid-flight, truncating the
    // trajectory with a derivative kink.
    return std::abs(lowpass - target) > epsilon || std::abs(bandpass) > epsilon;
}

inline void Smoothers::StateVariableSmoother::skip() noexcept
{
    v1 = 0.0;
    v2 = target;
    lowpass = target;
    bandpass = 0.0;
    highpass = 0.0;
}

// --- ButterworthSmoother ---

inline void Smoothers::ButterworthSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    constexpr double kSqrt2 = 1.4142135623730950488;
    const double timeConstantSeconds = static_cast<double>(timeConstantMilliseconds) / 1000.0;
    const double fs = sampleRate;
    // Zero time constant maps to the fastest usable cutoff (see the SVF
    // smoother): fc = 0 would freeze the filter short of the target forever.
    double fc = timeConstantSeconds > 1e-9 ? 1.0 / (static_cast<double>(Constants::twoPi) * timeConstantSeconds) : fs;
    fc = std::min(fc, fs * 0.49); // keep below Nyquist for a stable bilinear prewarp
    const double tanw = std::tan(dspark::pi<double> * fc / fs);
    const double tanw2 = tanw * tanw;

    const double denom = 1.0 + kSqrt2 * tanw + tanw2;

    b0 = tanw2 / denom;
    b1 = 2.0 * tanw2 / denom;
    b2 = tanw2 / denom;

    a1 = 2.0 * (tanw2 - 1.0) / denom;
    a2 = (1.0 - kSqrt2 * tanw + tanw2) / denom;

    // The motion threshold for the settle check: the peak per-sample velocity
    // of a residual oscillation of amplitude epsilon at fc. Near the target
    // crossing of the overshoot the output moves fast in filter time but by
    // tiny per-sample amounts (proportional to fc/fs), so comparing the
    // one-sample difference against the plain epsilon would still snap
    // mid-flight for small targets.
    velEps = static_cast<double>(epsilon) * (static_cast<double>(Constants::twoPi) * fc / fs);

    // TDF-II steady state for a constant input/output v requires the s2 term
    // inside s1: s2* = (b2 - a2)*v and s1* = (b1 - a1)*v + s2*. Without the
    // + s2 term the first sample of every new ramp overshot to ~2x the value
    // (an audible click, the opposite of a smoother's job).
    s2 = (b2 - a2) * initialValue;
    s1 = (b1 - a1) * initialValue + s2;
    target = initialValue;
    lastOut = initialValue;
    prevOut = initialValue;
}

inline void Smoothers::ButterworthSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::ButterworthSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return static_cast<float>(target); } // Anti-denormal check

    const double x0 = target;
    const double y0 = b0 * x0 + s1;
    s1 = b1 * x0 - a1 * y0 + s2;
    s2 = b2 * x0 - a2 * y0;

    prevOut = lastOut;
    lastOut = y0;
    return static_cast<float>(y0);
}

inline bool Smoothers::ButterworthSmoother::isSmoothing() const noexcept
{
    // Error AND motion: a Butterworth response (Q = 0.707) overshoots, so
    // the output crosses the target at full speed. Judging by the error
    // alone, the anti-denormal check in getNextValue() would snap exactly
    // at that crossing and truncate the remaining trajectory. Motion is
    // measured against velEps (velocity of an epsilon-amplitude residual),
    // the same role the bandpass term plays in the SVF smoother.
    return std::abs(lastOut - target) > epsilon
        || std::abs(lastOut - prevOut) > velEps;
}

inline void Smoothers::ButterworthSmoother::skip() noexcept
{
    // Same steady-state form as reset(): s1 must include the settled s2.
    s2 = (b2 - a2) * target;
    s1 = (b1 - a1) * target + s2;
    lastOut = target;
    prevOut = target;
}

// --- HermiteSmoother ---

inline void Smoothers::HermiteSmoother::reset(double sampleRate, float rampTimeMilliseconds, float initialValue) noexcept
{
    // A NaN or negative ramp is no ramp; the cap keeps the cast defined.
    const double steps = sampleRate * static_cast<double>(rampTimeMilliseconds) / 1000.0;
    rampSteps = steps > 0.0 ? static_cast<int>(std::min(steps + 0.5, 1.0e9)) : 0;
    current = target = start = initialValue;
    slope0 = 0.0;
    totalSteps = stepsDone = 0;
    invSteps = 1.0;
}

inline void Smoothers::HermiteSmoother::setTargetValue(float newTarget) noexcept
{
    // NaN/Inf would leave the segment unable to land.
    if (!std::isfinite(newTarget) || static_cast<double>(newTarget) == target) return;
    // Velocity of the segment in flight at its current position, in value
    // per sample: d/du of the cubic, divided by the segment length.
    double velocity = 0.0;
    if (stepsDone < totalSteps)
    {
        const double u = stepsDone * invSteps;
        const double d = target - start;
        velocity = ((6.0 * u - 6.0 * u * u) * d + (3.0 * u * u - 4.0 * u + 1.0) * slope0) * invSteps;
    }
    target = newTarget;
    start = current;
    if (rampSteps <= 0)
    {
        current = start = target;
        slope0 = 0.0;
        totalSteps = stepsDone = 0;
        return;
    }
    totalSteps = rampSteps;
    stepsDone = 0;
    invSteps = 1.0 / totalSteps;
    // Start velocity in value per segment; past three times the distance a
    // cubic overshoots, so clip there (Fritsch-Carlson limit). Opposite
    // velocity turns the glide around without passing the target.
    const double d = target - start;
    double m = velocity * totalSteps;
    if (d > 0.0) m = std::min(m, 3.0 * d);
    else if (d < 0.0) m = std::max(m, 3.0 * d);
    else m = 0.0;
    slope0 = m;
}

inline float Smoothers::HermiteSmoother::getNextValue() noexcept
{
    if (stepsDone >= totalSteps) return static_cast<float>(current);
    if (++stepsDone == totalSteps)
    {
        current = target;   // exact arrival
        return static_cast<float>(current);
    }
    const double u = stepsDone * invSteps;
    const double u2 = u * u;
    const double d = target - start;
    // Hermite basis with p(0) = start, p'(0) = slope0, p(1) = target, p'(1) = 0.
    current = start + d * (3.0 * u2 - 2.0 * u2 * u) + slope0 * (u2 * u - 2.0 * u2 + u);
    return static_cast<float>(current);
}

inline void Smoothers::HermiteSmoother::skip() noexcept
{
    current = start = target;
    slope0 = 0.0;
    totalSteps = stepsDone = 0;
}

// --- CriticallyDampedSmoother ---

inline void Smoothers::CriticallyDampedSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    // Critical damping (zeta = 1) corresponds exactly to Q = 0.5, not to the
    // Butterworth Q = 0.707.
    StateVariableSmoother::reset(sampleRate, timeConstantMilliseconds, 0.5f, initialValue);
}

} // namespace dspark