// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>

/**
 * @namespace Smoothers
 * @brief A collection of real-time safe smoothing filters for parameter interpolation in audio.
 *
 * Standalone — no external dependencies beyond the C++ standard library.
 *
 * These classes provide various smoothing techniques to prevent artifacts like zipper noise
 * or clicks during parameter changes. All smoothers are designed for use in the audio thread:
 * lock-free, no dynamic allocations, and noexcept methods. All classes enforce 32-byte alignment
 * to play nicely with SIMD/AVX auto-vectorization.
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
 */
struct alignas(32) LinearSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float rampTimeMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    
    [[nodiscard]] float getCurrentValue() const noexcept { return current; }
    [[nodiscard]] float getTargetValue() const noexcept { return target; }
    
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
    float current     = 0.0f;
    float target      = 0.0f;
    float step        = 0.0f;
    int   stepsToGo   = 0;
    int   totalSteps  = 0;
};

/**
 * @class ExponentialSmoother
 * @brief Exponential (multiplicative) smoother for natural, perceptual responses.
 *
 * Use for: Filter cutoffs, frequencies, volumes in dB.
 *
 * @warning Geometric smoothing is only defined for same-sign transitions. If
 * the new target's sign differs from the current value (ratio <= 0), the
 * smoother snaps instantly to the target — an audible step. Keep this class
 * for strictly-positive parameters (Hz, linear gain); use LinearSmoother or
 * OnePoleSmoother for bipolar values (pan, offsets).
 */
struct alignas(32) ExponentialSmoother
{
public:
    static constexpr float epsilon = 1e-10f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 1.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    
    [[nodiscard]] float getCurrentValue() const noexcept { return current; }
    [[nodiscard]] float getTargetValue() const noexcept { return target; }
    
    void setCurrentAndTargetValue(float value) noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    float current    = 1.0f;
    float target     = 1.0f;
    float coeff      = 0.0f;
    int   stepsToGo  = 0;
    int   totalSteps = 0;
};

/**
 * @class OnePoleSmoother
 * @brief Authentic one-pole exponential IIR low-pass smoother.
 *
 * Implements analog-style smoothing. Anti-denormal protected.
 */
struct alignas(32) OnePoleSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    
    [[nodiscard]] float getCurrentValue() const noexcept { return current; }
    [[nodiscard]] float getTargetValue() const noexcept { return target; }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    float coeff   = 0.0f;
    float current = 0.0f;
    float target  = 0.0f;
};

/**
 * @class MultiPoleSmoother
 * @brief Templated cascaded multi-pole smoother for steeper roll-off.
 */
template <std::size_t N>
struct alignas(32) MultiPoleSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    std::array<OnePoleSmoother, N> poles;
};

/**
 * @class AsymmetricSmoother
 * @brief Smoother with asymmetric attack/release times.
 */
struct alignas(32) AsymmetricSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float attackMilliseconds, float releaseMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    float attackCoeff  = 0.0f;
    float releaseCoeff = 0.0f;
    float current      = 0.0f;
    float target       = 0.0f;
};

/**
 * @class SlewLimiter
 * @brief Rate limiter to cap maximum change per sample.
 */
struct alignas(32) SlewLimiter
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float maxRatePerSecond, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    float maxDelta = 0.0f;
    float current  = 0.0f;
    float target   = 0.0f;
};

/**
 * @class StateVariableSmoother
 * @brief Second-order state variable filter (SVF) smoother (TPT implementation).
 */
struct alignas(32) StateVariableSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float q = 0.707f, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    
    [[nodiscard]] float getCurrentValue() const noexcept { return lowpass; }
    [[nodiscard]] float getTargetValue() const noexcept { return target; }
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;
    
    [[nodiscard]] float getBandPassOutput() const noexcept { return bandpass; }
    [[nodiscard]] float getHighPassOutput() const noexcept { return highpass; }

private:
    float v1 = 0.0f;
    float v2 = 0.0f;
    float g  = 0.0f;
    float k  = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float target   = 0.0f;
    float lowpass  = 0.0f;
    float bandpass = 0.0f;
    float highpass = 0.0f;
};

/**
 * @class ButterworthSmoother
 * @brief Butterworth low-pass smoother for maximally flat response.
 */
struct alignas(32) ButterworthSmoother
{
public:
    static constexpr float epsilon = 1e-6f;

    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
    void setTargetValue(float newTarget) noexcept;
    float getNextValue() noexcept;
    [[nodiscard]] bool isSmoothing() const noexcept;
    void skip() noexcept;

private:
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
    float s1 = 0.0f, s2 = 0.0f; 
    float target  = 0.0f;
    float lastOut = 0.0f; 
};

/**
 * @class CriticallyDampedSmoother
 * @brief Critically damped smoother (no overshoot, exact Q=0.5).
 */
struct alignas(32) CriticallyDampedSmoother : public StateVariableSmoother
{
public:
    void reset(double sampleRate, float timeConstantMilliseconds, float initialValue = 0.0f) noexcept;
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
        step = (target - current) / static_cast<float>(stepsToGo);
    else
        current = target;
}

inline float Smoothers::LinearSmoother::getNextValue() noexcept
{
    if (stepsToGo <= 0) return current;
    current += step;
    --stepsToGo;
    if (stepsToGo == 0) current = target;
    return current;
}

inline void Smoothers::LinearSmoother::setCurrentAndTargetValue(float value) noexcept
{
    current   = value;
    target    = value;
    step      = 0.0f;
    stepsToGo = 0;
}

inline bool Smoothers::LinearSmoother::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::LinearSmoother::skip() noexcept
{
    setCurrentAndTargetValue(target);
}

inline void Smoothers::LinearSmoother::processBlock(float* buffer, int numSamples, bool multiply) noexcept
{
    // Optimizacion SIMD: Dividir bucle transitorio del bucle de estado constante
    int stepsToProcess = std::min(numSamples, stepsToGo);
    int i = 0;

    // Smooth section (Branchless en el core del loop)
    for (; i < stepsToProcess; ++i)
    {
        current += step;
        buffer[i] = multiply ? buffer[i] * current : buffer[i] + current;
    }
    
    stepsToGo -= stepsToProcess;
    if (stepsToGo == 0) current = target;

    // Constant section (Alta vectorización asegurada)
    for (; i < numSamples; ++i)
    {
        buffer[i] = multiply ? buffer[i] * target : buffer[i] + target;
    }
}

// --- ExponentialSmoother ---

inline void Smoothers::ExponentialSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    totalSteps = static_cast<int>(sampleRate * timeConstantMilliseconds / 1000.0);
    stepsToGo  = 0;
    coeff      = 1.0f;
    current    = initialValue;
    target     = initialValue;
}

inline void Smoothers::ExponentialSmoother::setTargetValue(float newTarget) noexcept
{
    if (std::abs(newTarget) < epsilon)
        newTarget = (newTarget < 0.0f) ? -epsilon : epsilon;

    if (newTarget == target) return;
    target = newTarget;
    stepsToGo = totalSteps;

    if (stepsToGo > 0 && std::abs(current) > epsilon)
    {
        float safeCur = (current > 0.0f) ? std::max(current, epsilon)
                                         : std::min(current, -epsilon);
        float ratio = target / safeCur;
        if (ratio > 0.0f) 
            coeff = std::exp(std::log(ratio) / static_cast<float>(stepsToGo));
        else
            current = target; 
    }
    else
        current = target;
}

inline float Smoothers::ExponentialSmoother::getNextValue() noexcept
{
    if (stepsToGo <= 0) return current;
    current *= coeff;
    --stepsToGo;
    if (stepsToGo == 0) current = target;
    return current;
}

inline void Smoothers::ExponentialSmoother::setCurrentAndTargetValue(float value) noexcept
{
    current   = value;
    target    = value;
    coeff     = 1.0f;
    stepsToGo = 0;
}

inline bool Smoothers::ExponentialSmoother::isSmoothing() const noexcept
{
    return std::abs(current - target) > epsilon;
}

inline void Smoothers::ExponentialSmoother::skip() noexcept
{
    setCurrentAndTargetValue(target);
}

// --- OnePoleSmoother ---

inline void Smoothers::OnePoleSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    const float timeConstantSeconds = timeConstantMilliseconds / 1000.0f;
    float tau = static_cast<float>(sampleRate) * timeConstantSeconds;
    coeff = tau > 0.0f ? std::exp(-1.0f / tau) : 0.0f;
    current = initialValue;
    target = initialValue;
}

inline void Smoothers::OnePoleSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::OnePoleSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return target; } // Anti-denormal check
    current = target + coeff * (current - target);
    return current;
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
    return poles.back().isSmoothing();
}

template <std::size_t N>
inline void Smoothers::MultiPoleSmoother<N>::skip() noexcept
{
    for (auto& pole : poles)
        pole.skip();
}

// --- AsymmetricSmoother ---

inline void Smoothers::AsymmetricSmoother::reset(double sampleRate, float attackMilliseconds, float releaseMilliseconds, float initialValue) noexcept
{
    const float fs = static_cast<float>(sampleRate);
    const float attackSeconds = attackMilliseconds / 1000.0f;
    const float releaseSeconds = releaseMilliseconds / 1000.0f;
    attackCoeff = attackSeconds > 0.0f ? std::exp(-1.0f / (fs * attackSeconds)) : 0.0f;
    releaseCoeff = releaseSeconds > 0.0f ? std::exp(-1.0f / (fs * releaseSeconds)) : 0.0f;
    current = initialValue;
    target = initialValue;
}

inline void Smoothers::AsymmetricSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::AsymmetricSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return target; } // Anti-denormal check
    float c = (target > current) ? attackCoeff : releaseCoeff;
    current = target + c * (current - target);
    return current;
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
    maxDelta = maxRatePerSecond / static_cast<float>(sampleRate);
    current = initialValue;
    target = initialValue;
}

inline void Smoothers::SlewLimiter::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::SlewLimiter::getNextValue() noexcept
{
    float delta = target - current;
    delta = std::clamp(delta, -maxDelta, maxDelta);
    current += delta;
    return current;
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
    const float timeConstantSeconds = timeConstantMilliseconds / 1000.0f;
    float fc = timeConstantSeconds > 1e-9f ? 1.0f / (Constants::twoPi * timeConstantSeconds) : 0.0f;
    float fs = static_cast<float>(sampleRate);
    // Clamp below Nyquist so tan(pi*fc/fs) stays finite/stable for very small
    // time constants (otherwise the TPT prewarp blows up near fs/2).
    fc = std::min(fc, fs * 0.49f);

    g = std::tan(Constants::pi * fc / fs);
    k = 1.0f / q; // CORREGIDO: TPT damping usa k = 1/Q. 

    a1 = 1.0f / (1.0f + g * (g + k));
    a2 = g * a1;
    a3 = g * a2;

    v1 = 0.0f;
    v2 = initialValue;
    target = initialValue;
    lowpass = initialValue;
    bandpass = 0.0f;
    highpass = 0.0f;
}

inline void Smoothers::StateVariableSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::StateVariableSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return target; } // Anti-denormal check
    
    float v0 = target;
    float v3 = v0 - v2;
    float v1_new = a1 * v1 + a2 * v3;
    float v2_new = v2 + a2 * v1 + a3 * v3;
    v1 = 2.0f * v1_new - v1;
    v2 = 2.0f * v2_new - v2;

    lowpass = v2_new;
    bandpass = v1_new;
    highpass = v0 - k * v1_new - v2_new;

    return lowpass;
}

inline bool Smoothers::StateVariableSmoother::isSmoothing() const noexcept
{
    return std::abs(lowpass - target) > epsilon;
}

inline void Smoothers::StateVariableSmoother::skip() noexcept
{
    v1 = 0.0f;
    v2 = target;
    lowpass = target;
    bandpass = 0.0f;
    highpass = 0.0f;
}

// --- ButterworthSmoother ---

inline void Smoothers::ButterworthSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    const float timeConstantSeconds = timeConstantMilliseconds / 1000.0f;
    float fc = timeConstantSeconds > 1e-9f ? 1.0f / (Constants::twoPi * timeConstantSeconds) : 0.0f;
    float fs = static_cast<float>(sampleRate);
    fc = std::min(fc, fs * 0.49f); // keep below Nyquist for a stable bilinear prewarp
    float tanw = std::tan(Constants::pi * fc / fs);
    float tanw2 = tanw * tanw;

    float denom = 1.0f + Constants::sqrt2 * tanw + tanw2;

    b0 = tanw2 / denom;
    b1 = 2.0f * tanw2 / denom;
    b2 = tanw2 / denom;

    a1 = 2.0f * (tanw2 - 1.0f) / denom;
    a2 = (1.0f - Constants::sqrt2 * tanw + tanw2) / denom;

    // TDF-II steady state for a constant input/output v requires the s2 term
    // inside s1: s2* = (b2 - a2)*v and s1* = (b1 - a1)*v + s2*. Without the
    // + s2 term the first sample of every new ramp overshot to ~2x the value
    // (an audible click — the opposite of a smoother's job).
    s2 = (b2 - a2) * initialValue;
    s1 = (b1 - a1) * initialValue + s2;
    target = initialValue;
    lastOut = initialValue;
}

inline void Smoothers::ButterworthSmoother::setTargetValue(float newTarget) noexcept
{
    target = newTarget;
}

inline float Smoothers::ButterworthSmoother::getNextValue() noexcept
{
    if (!isSmoothing()) { skip(); return target; } // Anti-denormal check
    
    float x0 = target;
    float y0 = b0 * x0 + s1;
    s1 = b1 * x0 - a1 * y0 + s2;
    s2 = b2 * x0 - a2 * y0;
    
    lastOut = y0;
    return y0;
}

inline bool Smoothers::ButterworthSmoother::isSmoothing() const noexcept
{
    return std::abs(lastOut - target) > epsilon; // CORREGIDO: Utilizar lastOut cacheado
}

inline void Smoothers::ButterworthSmoother::skip() noexcept
{
    // Same steady-state form as reset(): s1 must include the settled s2.
    s2 = (b2 - a2) * target;
    s1 = (b1 - a1) * target + s2;
    lastOut = target;
}

// --- CriticallyDampedSmoother ---

inline void Smoothers::CriticallyDampedSmoother::reset(double sampleRate, float timeConstantMilliseconds, float initialValue) noexcept
{
    // CORREGIDO: La amortiguación crítica (zeta = 1) corresponde exactamente a Q = 0.5f, no a Q = 0.707 (Butterworth).
    StateVariableSmoother::reset(sampleRate, timeConstantMilliseconds, 0.5f, initialValue);
}

} // namespace dspark