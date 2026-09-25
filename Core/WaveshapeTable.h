// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WaveshapeTable.h
 * @brief Table-based waveshaping with Hermite interpolation for high-end audio.
 *
 * Stores a non-linear transfer function as a lookup table and applies it to audio
 * signals using 3rd-order Hermite interpolation. This minimizes table-induced
 * distortion and quantization noise compared to standard linear interpolation.
 *
 * @note **Aliasing (honest scope).** This is a MEMORYLESS nonlinearity: the
 * curve creates harmonics above Nyquist that fold back as aliasing exactly
 * like any waveshaper. The Hermite interpolation only reduces the error of
 * reading BETWEEN table entries (table-interpolation/quantization noise) - it
 * does NOT reduce harmonic aliasing. Two tools do, and they combine:
 * - the built-in oversampling (setOversampling: factor 1 = off, and 2/4/8/16
 *   supported; getLatency() reports the group delay it adds, 0 when off);
 * - first-order antiderivative anti-aliasing on processBlock()
 *   (setAntialiasing(), after Parker et al. DAFx-16 and Bilbao et al. 2017,
 *   with the antiderivative tabulated exactly from the interpolated curve so
 *   it works for any function, cf. Chowdhury, "Practical Considerations for
 *   Antiderivative Anti-Aliasing", 2020).
 * A finer table lowers interpolation noise but leaves the alias floor
 * unchanged.
 *
 * @note **Thread Safety & Real-Time Constraints:**
 * The `build...()` and `setOversampling()` methods allocate memory. They MUST
 * ONLY be called from the main thread (UI) or during the `prepare()` phase,
 * and never while the audio thread is inside a process call (rebuilding the
 * table concurrently with processing is a data race). For real-time
 * automation of distortion amount, use the `preGain` parameter in the
 * `process()` functions instead of rebuilding the table.
 *
 * Dependencies: DspMath.h, AudioSpec.h, AudioBuffer.h, Oversampling.h,
 * Interpolation.h.
 */

#include "DspMath.h"
#include "AudioSpec.h"
#include "AudioBuffer.h"
#include "Oversampling.h"
#include "Interpolation.h"

#include <algorithm>
#include <array>
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

        // Release-safe clamps: a tableSize below 4 makes process() compute
        // negative table positions (out-of-bounds reads), and a huge one
        // overflows the 32-bit size_t arithmetic on WASM. Invalid xMax
        // (non-positive or NaN) falls back to the default half-range.
        tableSize = std::clamp(tableSize, 4, 1 << 20);
        if (!(xMax > T(0))) xMax = T(8);

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

        // Antiderivative at every knot, integrating the interpolated curve
        // segment by segment (exact for the Hermite cubic), in double: the
        // ADAA difference quotient cancels most of its digits.
        step_ = 2.0 * static_cast<double>(xMax_) / static_cast<double>(tableSize - 1);
        integral_.assign(static_cast<size_t>(tableSize), 0.0);
        for (int i = 0; i + 1 < tableSize; ++i)
            integral_[static_cast<size_t>(i + 1)] = integral_[static_cast<size_t>(i)]
                                                  + step_ * segmentIntegral(i + 1, 1.0);
        resetAntialiasing();
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
        // min/max are ordered so that a NaN (input or preGain) resolves to the
        // upper table edge instead of reaching the int cast below (undefined)
        // and indexing out of bounds: the output stays finite.
        T driven = std::max(-xMax_, std::min(xMax_, input * preGain));

        // Map to index range [0, tableSize - 1]
        T pos = (driven + xMax_) * invRange_ * static_cast<T>(tableSize_ - 1);

        int idx = static_cast<int>(pos);
        T frac = pos - static_cast<T>(idx);

        // Base index offset by +1 due to left-side padding
        const T* t = table_.data() + idx + 1;

        // Catmull-Rom smoothing via the shared Hermite kernel (Interpolation.h)
        return interpolateHermite(t[-1], t[0], t[1], t[2], frac) * postGain;
    }

    /**
     * @brief Processes a buffer in-place.
     *
     * Tight scalar loop: the data-dependent table gather defeats
     * autovectorization, so per-sample cost is the process() call inlined.
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

        if (factor > 1)
        {
            oversampler_ = std::make_unique<Oversampling<T>>(factor);
            // Oversampling normalises invalid factors release-safe (rounds
            // down to a power of two); mirror the value it actually adopted
            // so getOversamplingFactor() never lies about the running rate.
            oversamplingFactor_ = oversampler_->getFactor();
            if (spec_.sampleRate > 0)
                oversampler_->prepare(spec_);
        }
        else
        {
            oversamplingFactor_ = 1;
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
        const bool adaa = antialias_;

        auto shape = [&](T* data, int n, int ch) {
            if (adaa && ch < kAntialiasChannels)
                processAntialiased(data, n, preGain, postGain, ch);
            else
                process(data, n, preGain, postGain);
        };

        if (oversamplingFactor_ > 1 && oversampler_)
        {
            auto upView = oversampler_->upsample(buffer);
            for (int ch = 0; ch < upView.getNumChannels(); ++ch)
                shape(upView.getChannel(ch), upView.getNumSamples(), ch);
            oversampler_->downsample(buffer);
        }
        else
        {
            for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
                shape(buffer.getChannel(ch), buffer.getNumSamples(), ch);
        }
    }

    /**
     * @brief Enables first-order antiderivative anti-aliasing (ADAA) in
     *        processBlock().
     *
     * Each output is the mean of the curve over the segment between two
     * consecutive (driven) inputs, (F(x[n]) - F(x[n-1])) / (x[n] - x[n-1]),
     * which suppresses the harmonics that would fold back. It shifts the
     * signal by half a sample at the processing rate (not reported by
     * getLatency()) and, like any first-order ADAA, averages adjacent samples:
     * at 1x that rolls off the top octave (-11.7 dB at 20 kHz @ 48 kHz), so
     * combine it with setOversampling(2) or more (-2.0 dB at 2x, -0.5 dB at
     * 4x). The first 16 channels are anti-aliased; further channels run the
     * plain memoryless curve. The per-sample process() overloads stay
     * memoryless. Owner-managed like the other setters: call it from the
     * processing thread or while no process call is running. Enabling starts
     * each channel from its next input (no stale history, no click).
     */
    void setAntialiasing(bool enabled) noexcept
    {
        if (enabled && !antialias_) resetAntialiasing();
        antialias_ = enabled;
    }

    /** @brief True when processBlock() applies ADAA. */
    [[nodiscard]] bool isAntialiasingEnabled() const noexcept { return antialias_; }

    void reset() noexcept
    {
        if (oversampler_) oversampler_->reset();
        resetAntialiasing();
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
    static constexpr int kAntialiasChannels = 16;

    /** Integral over [0, u] (u in [0, 1]) of the Hermite segment starting at
     *  padded table index i (knot i - 1), in units of one table step. */
    [[nodiscard]] double segmentIntegral(int i, double u) const noexcept
    {
        const double y0 = table_[static_cast<size_t>(i - 1)];
        const double y1 = table_[static_cast<size_t>(i)];
        const double y2 = table_[static_cast<size_t>(i + 1)];
        const double y3 = table_[static_cast<size_t>(i + 2)];
        // Same coefficients as interpolateHermite(): ((a u - b) u + c) u + y1.
        const double c = (y2 - y0) * 0.5;
        const double v = y1 - y2;
        const double w = c + v;
        const double a = w + v + (y3 - y1) * 0.5;
        const double b = w + a;
        return (((a * 0.25 * u - b / 3.0) * u + c * 0.5) * u + y1) * u;
    }

    /** Antiderivative of the effective curve f(clamp(x, -xMax, xMax)): the
     *  tabulated integral inside the table, linear continuation outside. */
    [[nodiscard]] double antiderivative(double x) const noexcept
    {
        const double xMax = static_cast<double>(xMax_);
        const int last = tableSize_ - 1;
        if (x >= xMax)
            return integral_[static_cast<size_t>(last)]
                 + static_cast<double>(table_[static_cast<size_t>(last + 1)]) * (x - xMax);
        if (x <= -xMax)
            return static_cast<double>(table_[1]) * (x + xMax);
        const double pos = (x + xMax) / step_;
        const int idx = std::min(static_cast<int>(pos), last - 1);
        return integral_[static_cast<size_t>(idx)] + step_ * segmentIntegral(idx + 1, pos - idx);
    }

    void processAntialiased(T* data, int numSamples, T preGain, T postGain, int ch) noexcept
    {
        // Out-of-range drive keeps its finite linear continuation; NaN lands
        // on the upper edge, as in process().
        const double limit = 1.0e3 * static_cast<double>(xMax_);
        const auto c = static_cast<size_t>(ch);
        double x0 = adaaX_[c];
        double f0 = adaaF_[c];
        const double post = static_cast<double>(postGain);
        for (int i = 0; i < numSamples; ++i)
        {
            const double x = std::max(-limit, std::min(limit, static_cast<double>(data[i] * preGain)));
            const double f = antiderivative(x);
            if (!adaaPrimed_[c]) // first input after a reset: nothing to average with yet
            {
                x0 = x;
                f0 = f;
                adaaPrimed_[c] = true;
            }
            const double dx = x - x0;
            double y;
            if (std::abs(dx) > 1.0e-6)
                y = (f - f0) / dx;
            else // ill-conditioned: the mean over a vanishing segment is its midpoint value
                y = static_cast<double>(process(static_cast<T>(0.5 * (x + x0))));
            data[i] = static_cast<T>(y * post);
            x0 = x;
            f0 = f;
        }
        adaaX_[c] = x0;
        adaaF_[c] = f0;
    }

    void resetAntialiasing() noexcept { adaaPrimed_.fill(false); }

    std::vector<T> table_;
    std::vector<double> integral_;  ///< Antiderivative at each knot (ADAA).
    double step_ = 1.0;             ///< Knot spacing in input units.
    bool antialias_ = false;
    std::array<double, kAntialiasChannels> adaaX_ {};   ///< Previous driven input per channel.
    std::array<double, kAntialiasChannels> adaaF_ {};   ///< Its antiderivative.
    std::array<bool, kAntialiasChannels> adaaPrimed_ {};
    int tableSize_ = 0;
    T xMax_ = T(8);                 ///< Half-range of the table input domain.
    T invRange_ = T(1) / T(16);     ///< Precomputed 1 / (2 * xMax).

    AudioSpec spec_ {};
    std::unique_ptr<Oversampling<T>> oversampler_;
    int oversamplingFactor_ = 1;
};

} // namespace dspark
