// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SampleAndHold.h
 * @brief Sample-and-hold processor for stepped modulation and bit-crushing.
 *
 * Captures an input sample and holds it for a configurable number of samples
 * or until an external trigger fires. This introduces deliberate spectral imaging
 * (aliasing) due to its Zero-Order Hold (ZOH) nature, making it ideal for 
 * creative bit-crushing and classic stepped LFO synthesis.
 *
 * @note This is an effect/modulator, not a band-limited Sample Rate Converter (SRC).
 * * Dependencies: DspMath.h (for FloatType concept).
 */

#include "DspMath.h"

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
        Trigger  ///< Holds indefinitely until an external trigger signal fires.
    };

    /**
     * @brief Sets the operating mode.
     * @param mode Mode::Counter or Mode::Trigger.
     */
    void setMode(Mode mode) noexcept { mode_ = mode; }

    /**
     * @brief Sets the hold duration for Counter mode in samples.
     * * A value of 1 passes the signal transparently. A value of N reduces
     * the effective sample rate by a factor of N.
     *
     * @param numSamples Hold period in samples. Clamped to a minimum of 1.
     */
    void setHoldSamples(int numSamples) noexcept
    {
        holdPeriod_ = numSamples > 0 ? numSamples : 1;
    }

    /**
     * @brief Sets the hold period based on a target effective sample rate.
     *
     * @param targetRate The desired effective sample rate in Hz. Must be > 0.
     * @param actualRate The current system sample rate in Hz.
     */
    void setHoldRate(double targetRate, double actualRate) noexcept
    {
        if (targetRate <= 0.0 || actualRate <= 0.0) 
        {
            setHoldSamples(1);
            return;
        }
        setHoldSamples(static_cast<int>(actualRate / targetRate));
    }

    /**
     * @brief Processes a single sample.
     * * @warning For optimal performance on blocks, use processBlock() instead,
     * as this scalar method contains internal branching based on the Mode.
     *
     * @param input The audio or modulation input sample.
     * @param trigger External trigger state (ignored in Counter mode).
     * @return The currently held output sample.
     */
    [[nodiscard]] T process(T input, bool trigger = false) noexcept
    {
        if (mode_ == Mode::Counter)
        {
            ++counter_;
            if (counter_ >= holdPeriod_)
            {
                heldValue_ = input;
                counter_ = 0;
            }
        }
        else // Mode::Trigger
        {
            if (trigger)
            {
                heldValue_ = input;
            }
        }
        return heldValue_;
    }

    /**
     * @brief Processes a block of samples in-place (Counter mode optimized).
     * * If the processor is in Trigger mode, calling this method will simply 
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
                ++counter_;
                if (counter_ >= holdPeriod_)
                {
                    heldValue_ = data[i];
                    counter_ = 0;
                }
                data[i] = heldValue_;
            }
        }
        else
        {
            // In trigger mode with no triggers provided, it holds indefinitely.
            for (int i = 0; i < numSamples; ++i)
            {
                data[i] = heldValue_;
            }
        }
    }

    /**
     * @brief Processes a block of samples in-place using an external trigger buffer.
     * * Only relevant when Mode is set to Trigger. If Mode is Counter, the 
     * triggers buffer is safely ignored.
     *
     * @param data Audio buffer to process in-place.
     * @param triggers Array of boolean triggers (must be size of numSamples).
     * @param numSamples Number of samples in the buffer.
     */
    void processBlock(T* data, const bool* triggers, int numSamples) noexcept
    {
        if (mode_ == Mode::Trigger)
        {
            // Branch hoisted out of the hot loop
            for (int i = 0; i < numSamples; ++i)
            {
                if (triggers[i])
                {
                    heldValue_ = data[i];
                }
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
     * * @param initialValue Value to output until the next capture triggers.
     */
    void reset(T initialValue = T(0)) noexcept
    {
        heldValue_ = initialValue;
        counter_ = 0; // Fixed: ensures initialValue is outputted before next capture
    }

private:
    Mode mode_ = Mode::Counter;
    int holdPeriod_ = 1;
    int counter_ = 0;
    T heldValue_ = T(0);
};

} // namespace dspark