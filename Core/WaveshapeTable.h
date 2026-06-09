// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file WaveshapeTable.h
 * @brief Table-based waveshaping with Hermite interpolation for high-end audio.
 *
 * Stores a non-linear transfer function as a lookup table and applies it to audio
 * signals using 3rd-order Hermite interpolation. This minimizes table-induced 
 * distortion and quantization noise compared to standard linear interpolation.
 *
 * @note **Thread Safety & Real-Time Constraints:**
 * The `build...()` and `setOversampling()` methods allocate memory. They MUST ONLY 
 * be called from the main thread (UI) or during the `prepare()` phase. 
 * For real-time automation of distortion amount, use the `preGain` parameter in the 
 * `process()` functions instead of rebuilding the table.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, Oversampling.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "Oversampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class WaveshapeTable
 * @brief Zero-latency lookup-table waveshaper with RT-safe gain modulation.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class WaveshapeTable
{
public:
    /** 
     * @brief Constructor. Initializes a safe passthrough table to prevent RT crashes. 
     */
    WaveshapeTable() 
    {
        buildFromFunction([](T x) { return x; }, 4);
    }

    /**
     * @brief Builds the lookup table from an arbitrary transfer function.
     *
     * @warning Allocates memory. Do not call from the audio thread.
     *
     * The table spans the input range [-xMax, +xMax]. A range wider than the
     * nominal [-1, 1] is what makes `preGain` work as a true drive control:
     * driven samples keep following the curve instead of slamming into a hard
     * plateau at the table edge.
     *
     * @param func Transfer function mapping [-xMax, xMax] to output.
     * @param tableSize Number of active table entries (default: 4096).
     * @param xMax Half-range of the table input domain (default: 8).
     */
    void buildFromFunction(std::function<T(T)> func, int tableSize = 4096, T xMax = T(8))
    {
        assert(tableSize >= 4);
        assert(xMax > T(0));
        tableSize_ = tableSize;
        xMax_ = std::max(xMax, T(0.001));
        invRange_ = T(1) / (T(2) * xMax_);

        // Allocate tableSize + 3 to accommodate Hermite interpolation padding.
        // This eliminates conditional branch bounds-checking in the DSP hot-path.
        table_.resize(static_cast<size_t>(tableSize) + 3);

        for (int i = 0; i < tableSize; ++i)
        {
            T x = -xMax_ + T(2) * xMax_ * static_cast<T>(i) / static_cast<T>(tableSize - 1);
            table_[static_cast<size_t>(i + 1)] = func(x); // Offset by 1
        }

        // Pad boundaries for Hermite (mirroring endpoints)
        table_[0] = table_[1];
        table_[static_cast<size_t>(tableSize + 1)] = table_[static_cast<size_t>(tableSize)];
        table_[static_cast<size_t>(tableSize + 2)] = table_[static_cast<size_t>(tableSize)];
    }

    /**
     * @brief Builds a normalized tanh (soft clip) table.
     * @note Use `preGain` in `process()` to drive the saturation in Real-Time.
     * @param tableSize Table entries (default: 4096).
     */
    void buildTanh(int tableSize = 4096)
    {
        buildFromFunction([](T x) -> T { return std::tanh(x); }, tableSize);
    }

    /**
     * @brief Builds a hard-clip table.
     * @param threshold Clipping threshold (0 to 1, default: 0.8).
     * @param tableSize Table entries.
     */
    void buildHardClip(T threshold = T(0.8), int tableSize = 4096)
    {
        buildFromFunction([threshold](T x) -> T {
            return std::clamp(x, -threshold, threshold);
        }, tableSize);
    }

    /**
     * @brief Builds a cubic soft-clip table.
     * @param tableSize Table entries.
     */
    void buildSoftClip(int tableSize = 4096)
    {
        buildFromFunction([](T x) -> T {
            if (x > T(1)) return T(2.0 / 3.0);
            if (x < T(-1)) return T(-2.0 / 3.0);
            return x - (x * x * x) / T(3);
        }, tableSize);
    }

    /**
     * @brief Builds an asymmetric clipping table for even-harmonic generation.
     * @param tableSize Table entries.
     */
    void buildAsymmetric(int tableSize = 4096)
    {
        buildFromFunction([](T x) -> T {
            return x >= T(0) ? std::tanh(x * T(1.2)) : std::tanh(x * T(0.8));
        }, tableSize);
    }

    /**
     * @brief Processes a single sample through the waveshaper with Hermite interpolation.
     *
     * @param input Input sample.
     * @param preGain Real-time drive/gain applied before lookup (default 1.0).
     * @param postGain Real-time makeup gain applied after lookup (default 1.0).
     * @return Shaped output sample.
     */
    [[nodiscard]] inline T process(T input, T preGain = T(1), T postGain = T(1)) const noexcept
    {
        // The table spans [-xMax, xMax] (default 8), so realistic drive values
        // keep tracing the curve instead of flattening at the old +-1 boundary.
        T driven = std::clamp(input * preGain, -xMax_, xMax_);

        // Map to index range [0, tableSize - 1]
        T pos = (driven + xMax_) * invRange_ * static_cast<T>(tableSize_ - 1);

        // Branchless integer extraction
        int idx = static_cast<int>(pos);
        T frac = pos - static_cast<T>(idx);

        // Base index offset by +1 due to left-side padding
        const T* t = table_.data() + idx + 1;

        // 3rd-order Hermite interpolation for analog-like smoothness
        T c0 = t[0];
        T c1 = T(0.5) * (t[1] - t[-1]);
        T c2 = t[-1] - T(2.5) * t[0] + T(2.0) * t[1] - T(0.5) * t[2];
        T c3 = T(1.5) * (t[0] - t[1]) + T(0.5) * (t[2] - t[-1]);

        T output = ((c3 * frac + c2) * frac + c1) * frac + c0;
        
        return output * postGain;
    }

    /**
     * @brief Processes a buffer in-place. SIMD-friendly loop.
     * 
     * @param data Audio samples pointer (should ideally be __restrict).
     * @param numSamples Number of samples to process.
     * @param preGain Gain to drive into the shaper.
     * @param postGain Gain to compensate volume output.
     */
    void process(T* __restrict data, int numSamples, T preGain = T(1), T postGain = T(1)) const noexcept
    {
        for (int i = 0; i < numSamples; ++i)
            data[i] = process(data[i], preGain, postGain);
    }

    // -- Lifecycle & Oversampling ---------------------------------------------

    /**
     * @brief Prepares the waveshaper for oversampled block processing.
     * @param spec Audio specification.
     */
    void prepare(const AudioSpec& spec)
    {
        spec_ = spec;
        if (oversampler_)
            oversampler_->prepare(spec);
    }

    /**
     * @brief Enables oversampling. 
     * @warning Allocates memory. Do not call from the audio thread.
     * @param factor Oversampling factor (1 = off, 2, 4, 8, 16).
     */
    void setOversampling(int factor)
    {
        assert(factor >= 1 && (factor & (factor - 1)) == 0);
        oversamplingFactor_ = factor;
        
        if (factor > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(factor);
            if (spec_.sampleRate > 0)
                oversampler_->prepare(spec_);
        }
        else
        {
            oversampler_.reset();
        }
    }

    [[nodiscard]] int getOversamplingFactor() const noexcept { return oversamplingFactor_; }

    /**
     * @brief Processes a buffer view with optional oversampling.
     * @param buffer Audio buffer view.
     * @param preGain Real-time drive applied per-sample.
     * @param postGain Real-time makeup gain applied per-sample.
     */
    void processBlock(AudioBufferView<T> buffer, T preGain = T(1), T postGain = T(1)) noexcept
    {
        if (oversamplingFactor_ > 1 && oversampler_)
        {
            auto upView = oversampler_->upsample(buffer);
            for (int ch = 0; ch < upView.getNumChannels(); ++ch)
            {
                process(upView.getChannel(ch), upView.getNumSamples(), preGain, postGain);
            }
            oversampler_->downsample(buffer);
        }
        else
        {
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
            {
                process(buffer.getChannel(ch), buffer.getNumSamples(), preGain, postGain);
            }
        }
    }

    void reset() noexcept
    {
        if (oversampler_) oversampler_->reset();
    }

    [[nodiscard]] int getTableSize() const noexcept { return tableSize_; }
    [[nodiscard]] bool isReady() const noexcept { return tableSize_ > 0; }

    /** @brief Returns the half-range of the table's input domain. */
    [[nodiscard]] T getInputRange() const noexcept { return xMax_; }

    /**
     * @brief Reports the processing latency in samples.
     * @return The oversampler group delay (0 when oversampling is off).
     *         Report this to the host for plugin delay compensation.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return (oversampler_ && oversamplingFactor_ > 1) ? oversampler_->getLatency() : 0;
    }

private:
    std::vector<T> table_;
    int tableSize_ = 0;
    T xMax_ = T(8);                 ///< Half-range of the table input domain.
    T invRange_ = T(1) / T(16);     ///< Precomputed 1 / (2 * xMax).

    AudioSpec spec_ {};
    std::unique_ptr<Oversampling<T>> oversampler_;
    int oversamplingFactor_ = 1;
};

} // namespace dspark
