// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file PhaseCorrelation.h
 * @brief Stereo phase correlation meter with goniometer feed.
 *
 * The standard broadcast correlation meter: the normalized cross-correlation
 * of left and right at lag zero, exponentially windowed,
 *
 *   r = E[L·R] / sqrt(E[L²] · E[R²])   ∈ [-1, +1]
 *
 * +1 means mono-compatible (in phase), 0 uncorrelated (wide/reverberant),
 * -1 out of phase (mono cancellation). A stereo balance readout and a
 * decimated mid/side point ring for goniometer (vectorscope) displays are
 * included; all readouts are lock-free from any thread.
 *
 * Dependencies: AudioSpec.h, AudioBuffer.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>

namespace dspark {

/**
 * @class PhaseCorrelation
 * @brief Correlation/balance meter and goniometer data source.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class PhaseCorrelation
{
public:
    /** @brief One mid/side point for goniometer displays. */
    struct GonioPoint
    {
        float mid;
        float side;
    };

    static constexpr int kGonioSize = 1024;   ///< Goniometer ring length.

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Prepares the meter.
     * @param spec      Audio environment specification (needs >= 2 channels
     *                  for correlation; mono inputs read as r = +1).
     * @param windowMs  Averaging window for the correlation (default 300 ms).
     */
    void prepare(const AudioSpec& spec, double windowMs = 300.0)
    {
        if (spec.sampleRate <= 0.0) return;
        sampleRate_ = spec.sampleRate;
        alpha_ = 1.0 - std::exp(-1.0 / (std::max(windowMs, 1.0) * 0.001 * sampleRate_));
        // Decimate the goniometer feed to ~24k points/s — plenty for displays.
        gonioDecim_ = std::max(1, static_cast<int>(sampleRate_ / 24000.0));
        prepared_ = true;
        reset();
    }

    /** @brief Clears the averages and the goniometer ring. RT-safe. */
    void reset() noexcept
    {
        lr_ = ll_ = rr_ = 0.0;
        decimCount_ = 0;
        gonioWrite_.store(0, std::memory_order_relaxed);
        correlation_.store(T(0), std::memory_order_relaxed);
        balance_.store(T(0), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Analyzes a block (read-only; the audio is not modified). */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_ || buffer.getNumSamples() == 0) return;

        const int nS = buffer.getNumSamples();
        if (buffer.getNumChannels() < 2)
        {
            // Mono is perfectly correlated by definition.
            correlation_.store(T(1), std::memory_order_relaxed);
            balance_.store(T(0), std::memory_order_relaxed);
            return;
        }

        const T* L = buffer.getChannel(0);
        const T* R = buffer.getChannel(1);

        double lr = lr_, ll = ll_, rr = rr_;
        int wp = gonioWrite_.load(std::memory_order_relaxed);

        for (int i = 0; i < nS; ++i)
        {
            const double l = static_cast<double>(L[i]);
            const double r = static_cast<double>(R[i]);
            lr += alpha_ * (l * r - lr);
            ll += alpha_ * (l * l - ll);
            rr += alpha_ * (r * r - rr);

            if (++decimCount_ >= gonioDecim_)
            {
                decimCount_ = 0;
                auto& p = gonio_[static_cast<size_t>(wp)];
                p.mid = static_cast<float>((l + r) * 0.7071067811865476);
                p.side = static_cast<float>((l - r) * 0.7071067811865476);
                wp = (wp + 1) & (kGonioSize - 1);
            }
        }

        lr_ = lr;
        ll_ = ll;
        rr_ = rr;
        gonioWrite_.store(wp, std::memory_order_release);

        const double denom = std::sqrt(ll * rr);
        const double corr = (denom > 1e-12) ? std::clamp(lr / denom, -1.0, 1.0) : 0.0;
        correlation_.store(static_cast<T>(corr), std::memory_order_relaxed);

        const double total = ll + rr;
        const double bal = (total > 1e-12) ? (rr - ll) / total : 0.0;
        balance_.store(static_cast<T>(bal), std::memory_order_relaxed);
    }

    // -- Readout (lock-free, any thread) ------------------------------------------------

    /** @return Correlation in [-1, +1]; +1 in phase, -1 phase-inverted. */
    [[nodiscard]] T getCorrelation() const noexcept
    {
        return correlation_.load(std::memory_order_relaxed);
    }

    /** @return Energy balance in [-1, +1]; -1 = all left, +1 = all right. */
    [[nodiscard]] T getBalance() const noexcept
    {
        return balance_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Copies the newest goniometer points (oldest first).
     * @param dest     Destination array.
     * @param maxCount Capacity of dest (clamped to kGonioSize).
     * @return Number of points copied.
     */
    int getGonioPoints(GonioPoint* dest, int maxCount) const noexcept
    {
        const int count = std::clamp(maxCount, 0, kGonioSize);
        const int wp = gonioWrite_.load(std::memory_order_acquire);
        for (int i = 0; i < count; ++i)
            dest[i] = gonio_[static_cast<size_t>((wp - count + i) & (kGonioSize - 1))];
        return count;
    }

private:
    double sampleRate_ = 48000.0;
    double alpha_ = 0.001;
    bool prepared_ = false;

    double lr_ = 0.0, ll_ = 0.0, rr_ = 0.0;

    int gonioDecim_ = 2;
    int decimCount_ = 0;
    std::array<GonioPoint, kGonioSize> gonio_ {};
    std::atomic<int> gonioWrite_ { 0 };

    std::atomic<T> correlation_ { T(0) };
    std::atomic<T> balance_ { T(0) };
};

} // namespace dspark
