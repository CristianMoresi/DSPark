// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SampleAndHold.h
 * @brief Sample-and-hold processor for stepped modulation and bit-crushing.
 *
 * Captures an input sample and holds it for a configurable (fractional)
 * number of samples or until an external trigger fires. This introduces
 * deliberate spectral imaging (aliasing) due to its Zero-Order Hold (ZOH)
 * nature, making it ideal for creative bit-crushing, sample-rate reduction
 * and classic stepped LFO synthesis.
 *
 * Threading: owner-managed. Not internally thread-safe: call setters and
 * process methods from the owning (audio) thread, or synchronise externally.
 *
 * @note This is an effect/modulator, not a band-limited Sample Rate Converter (SRC).
 *
 * Dependencies: DspMath.h (FloatType concept).
 */

#include "DspMath.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dspark {

/**
 * @class SampleAndHold
 * @brief Holds a sample value for N samples or until externally triggered.
 *
 * @tparam T Sample type (must satisfy FloatType concept).
 */
template <FloatType T>
class SampleAndHold
{
public:
    /** @brief Defines the capture behavior of the processor. */
    enum class Mode
    {
        Counter, ///< Holds for a fixed number of samples automatically (Decimation).
        Trigger  ///< Holds until an external trigger captures a new value (level-sensitive).
    };

    /**
     * @brief Sets the operating mode.
     * @param mode Mode::Counter or Mode::Trigger.
     */
    void setMode(Mode mode) noexcept { mode_ = mode; }

    /**
     * @brief Sets the hold duration for Counter mode in whole samples.
     *
     * A value of 1 passes the signal transparently. A value of N reduces the
     * effective sample rate by a factor of N. Capture phase: after reset(),
     * the first capture happens on the Nth call (the initial value is output
     * for the first N-1 samples), then every N samples.
     *
     * @param numSamples Hold period in samples. Clamped to a minimum of 1.
     */
    void setHoldSamples(int numSamples) noexcept
    {
        setHoldPeriod(static_cast<double>(numSamples));
    }

    /**
     * @brief Sets a fractional hold period for Counter mode, in samples.
     *
     * A non-integer period keeps the average capture rate exact, so a swept
     * rate moves continuously instead of jumping between whole-sample periods
     * (at 48 kHz those are 24 k, 16 k, 12 k, 9.6 k...). Each capture takes the
     * input at its exact fractional instant, interpolated linearly between
     * the two neighbouring samples; the held steps themselves still change on
     * sample boundaries. Integer periods behave exactly as setHoldSamples().
     *
     * @param samples Hold period in samples. Values below 1 and NaN clamp to
     *                1; huge values clamp to 2^31 - 1 (holds indefinitely).
     */
    void setHoldPeriod(double samples) noexcept
    {
        constexpr double maxPeriod = static_cast<double>(std::numeric_limits<int>::max());
        holdPeriod_ = (samples >= 1.0) ? std::min(samples, maxPeriod) : 1.0; // NaN -> 1
        // An integer period drops any sub-sample offset left by a fractional
        // one, so it captures on the sample grid (period 1 is transparent).
        if (holdPeriod_ == std::floor(holdPeriod_))
            phase_ = std::floor(phase_);
    }

    /** @brief Returns the Counter-mode hold period in samples. */
    [[nodiscard]] double getHoldPeriod() const noexcept { return holdPeriod_; }

    /**
     * @brief Sets the hold period from a target effective sample rate.
     *
     * The period is the exact ratio actualRate / targetRate (fractional, see
     * setHoldPeriod()), so the effective rate is exactly targetRate on average.
     *
     * @param targetRate The desired effective sample rate in Hz. Invalid
     * values (non-positive or NaN, either argument) reset the period to 1.
     * @param actualRate The current system sample rate in Hz.
     */
    void setHoldRate(double targetRate, double actualRate) noexcept
    {
        if (!(targetRate > 0.0) || !(actualRate > 0.0))
        {
            setHoldPeriod(1.0);
            return;
        }
        setHoldPeriod(actualRate / targetRate);
    }

    /**
     * @brief Processes a single sample.
     *
     * @warning For optimal performance on blocks, use processBlock() instead,
     * as this scalar method contains internal branching based on the Mode.
     *
     * @param input The audio or modulation input sample.
     * @param trigger External trigger state (ignored in Counter mode). The
     * trigger is level-sensitive: while it stays true, the input is tracked
     * sample-by-sample; send single-sample pulses for classic S&H capture.
     * @return The currently held output sample.
     */
    [[nodiscard]] T process(T input, bool trigger = false) noexcept
    {
        if (mode_ == Mode::Counter)
            advanceCounter(input);
        else if (trigger) // Mode::Trigger
            heldValue_ = input;
        previous_ = input;
        return heldValue_;
    }

    /**
     * @brief Processes a block of samples in-place (Counter mode optimized).
     *
     * If the processor is in Trigger mode, calling this method will simply
     * fill the buffer with the last held value. For Trigger mode processing,
     * use the overloaded processBlock with the trigger buffer.
     *
     * @param data Audio buffer to process in-place.
     * @param numSamples Number of samples in the buffer.
     */
    void processBlock(T* data, int numSamples) noexcept
    {
        if (mode_ == Mode::Counter)
        {
            // Branch hoisted out of the hot loop
            for (int i = 0; i < numSamples; ++i)
            {
                const T input = data[i];
                advanceCounter(input);
                previous_ = input;
                data[i] = heldValue_;
            }
        }
        else if (numSamples > 0)
        {
            // In trigger mode with no triggers provided, it holds indefinitely.
            previous_ = data[numSamples - 1];
            for (int i = 0; i < numSamples; ++i)
            {
                data[i] = heldValue_;
            }
        }
    }

    /**
     * @brief Processes a block of samples in-place using an external trigger buffer.
     *
     * Only relevant when Mode is set to Trigger (level-sensitive, same
     * semantics as process()). If Mode is Counter, the triggers buffer is
     * safely ignored.
     *
     * @param data Audio buffer to process in-place.
     * @param triggers Array of boolean triggers, one per sample. A null
     * pointer is treated as "no triggers" (the current value is held).
     * @param numSamples Number of samples in the buffer.
     */
    void processBlock(T* data, const bool* triggers, int numSamples) noexcept
    {
        if (mode_ == Mode::Trigger && triggers != nullptr)
        {
            // Branch hoisted out of the hot loop
            for (int i = 0; i < numSamples; ++i)
            {
                if (triggers[i])
                {
                    heldValue_ = data[i];
                }
                previous_ = data[i];
                data[i] = heldValue_;
            }
        }
        else
        {
            processBlock(data, numSamples);
        }
    }

    /**
     * @brief Retrieves the current held value without advancing the state.
     * @return The held sample value.
     */
    [[nodiscard]] T getHeldValue() const noexcept { return heldValue_; }

    /**
     * @brief Resets the processor state and sets an initial output value.
     *
     * @param initialValue Value to output until the next capture triggers.
     */
    void reset(T initialValue = T(0)) noexcept
    {
        heldValue_ = initialValue;
        previous_ = initialValue;
        phase_ = 0.0; // Ensures initialValue is output before the next capture
    }

private:
    /** Counter-mode step: captures once a full (fractional) period elapsed.
     *  After the wrap, phase_ is how far past the exact capture instant this
     *  sample lies (0 for integer periods, where the capture is exact). */
    inline void advanceCounter(T input) noexcept
    {
        phase_ += 1.0;
        if (phase_ >= holdPeriod_)
        {
            phase_ -= holdPeriod_;
            // The period shrank below the time already elapsed: capture now.
            if (phase_ >= 1.0) phase_ = 0.0;
            heldValue_ = (phase_ > 0.0) ? input - static_cast<T>(phase_) * (input - previous_)
                                        : input;
        }
    }

    Mode mode_ = Mode::Counter;
    double holdPeriod_ = 1.0;
    double phase_ = 0.0;     ///< Samples elapsed since the last capture instant.
    T heldValue_ = T(0);
    T previous_ = T(0);      ///< Last input, for the fractional capture.
};

} // namespace dspark
