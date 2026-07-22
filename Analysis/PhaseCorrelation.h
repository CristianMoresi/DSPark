// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file PhaseCorrelation.h
 * @brief Stereo phase correlation meter with goniometer feed.
 *
 * The standard broadcast correlation meter: the normalized cross-correlation
 * of left and right at lag zero, exponentially windowed,
 *
 *   r = E[L*R] / sqrt(E[L^2] * E[R^2])   in [-1, +1]
 *
 * +1 means mono-compatible (in phase), 0 uncorrelated (wide/reverberant),
 * -1 out of phase (mono cancellation). A stereo balance readout and a
 * decimated mid/side point ring for goniometer (vectorscope) displays are
 * included.
 *
 * Readout floor: with program below roughly -120 dBFS RMS both correlation
 * and balance read 0 (silence has no meaningful phase relationship).
 *
 * Non-finite input samples: the poisoned averaging window is discarded (the
 * accumulators restart from zero) and the last published readings are held;
 * measurement resumes on the next clean block. Non-finite samples never
 * reach the goniometer ring.
 *
 * Threading:
 * - prepare(): setup thread (allocation-free; not concurrent with processing).
 * - processBlock(): audio thread (single stream owner).
 * - reset(): stream owner or setup (touches the same plain accumulators
 *   processBlock() writes; allocation-free).
 * - getCorrelation() / getBalance() / getGonioPoints(): any thread,
 *   lock-free. Goniometer points are individually atomic, so a concurrent
 *   reader never observes torn values; a reader that overlaps a large
 *   writer block may briefly copy points newer than the published write
 *   index (valid mid/side data, display-only reordering).
 * - getWindowMs(): reflects the last successful prepare(); read it from the
 *   setup thread or once preparation is complete.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DspMath.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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
     * @brief Prepares the meter. Allocation-free.
     * @param spec      Audio environment specification. An invalid or
     *                  non-finite specification is ignored (conservative
     *                  no-op: the previous state is kept).
     * @param windowMs  Averaging time constant for the correlation (one-pole
     *                  tau, default 300 ms). Clamped to [1, 600000]; a
     *                  non-finite value falls back to the default.
     */
    void prepare(const AudioSpec& spec, double windowMs = 300.0) noexcept
    {
        if (!spec.isValid() || !std::isfinite(spec.sampleRate)) return;
        if (!std::isfinite(windowMs)) windowMs = 300.0;
        const double fs = spec.sampleRate;
        windowMs_ = std::clamp(windowMs, 1.0, 600000.0);
        alpha_ = 1.0 - std::exp(-1.0 / (windowMs_ * 0.001 * fs));
        // Decimate the goniometer feed to ~24k points/s - plenty for displays.
        gonioDecim_ = std::max(1, static_cast<int>(std::llround(std::min(fs / 24000.0, 65536.0))));
        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /**
     * @brief Clears the averages, the readouts and the goniometer ring.
     *
     * Allocation-free, but it writes the same plain accumulators
     * processBlock() owns: call it from the stream owner (or with the
     * stream stopped).
     */
    void reset() noexcept
    {
        lr_ = ll_ = rr_ = 0.0;
        decimCount_ = 0;
        for (auto& p : gonio_)
            p.store(0u, std::memory_order_relaxed);
        gonioWrite_.store(0, std::memory_order_relaxed);
        correlation_.store(T(0), std::memory_order_relaxed);
        balance_.store(T(0), std::memory_order_relaxed);
    }

    // -- Processing -------------------------------------------------------------------

    /**
     * @brief Analyzes a block (read-only; the audio is not modified).
     *
     * Single-channel buffers measure as dual mono: r = +1 while signal is
     * present (below the silence floor the readouts rest at 0, same as
     * stereo), balance 0, and the goniometer collapses onto the mid axis.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;

        const int nS = buffer.getNumSamples();
        const int nCh = buffer.getNumChannels();
        if (nS <= 0 || nCh <= 0) return;

        const T* L = buffer.getChannel(0);
        const T* R = (nCh >= 2) ? buffer.getChannel(1) : L;

        const double alpha = alpha_;
        double lr = lr_, ll = ll_, rr = rr_;
        int wp = gonioWrite_.load(std::memory_order_relaxed);

        for (int i = 0; i < nS; ++i)
        {
            const double l = static_cast<double>(L[i]);
            const double r = static_cast<double>(R[i]);
            lr += alpha * (l * r - lr);
            ll += alpha * (l * l - ll);
            rr += alpha * (r * r - rr);

            if (++decimCount_ >= gonioDecim_)
            {
                decimCount_ = 0;
                if (std::isfinite(l) && std::isfinite(r))
                {
                    GonioPoint p;
                    p.mid = static_cast<float>((l + r) * 0.7071067811865476);
                    p.side = static_cast<float>((l - r) * 0.7071067811865476);
                    gonio_[static_cast<size_t>(wp)].store(std::bit_cast<std::uint64_t>(p),
                                                          std::memory_order_relaxed);
                    wp = (wp + 1) & (kGonioSize - 1);
                }
            }
        }

        gonioWrite_.store(wp, std::memory_order_release);

        // A non-finite sample would park NaN in the recursive averages for
        // good (the one-pole never drains it). Discard the poisoned window
        // and hold the last published readings; the accumulators restart
        // from zero, so measurement resumes on the next clean block (the
        // correlation is a ratio - it re-converges immediately once real
        // signal flows again).
        if (!std::isfinite(lr + ll + rr))
        {
            lr_ = ll_ = rr_ = 0.0;
            return;
        }

        // Flush fully decayed averages to true zero: long silences would
        // otherwise walk the one-pole into double denormals and tax hosts
        // that do not enable FTZ. Well below any audible floor.
        if (ll < 1e-100) ll = 0.0;
        if (rr < 1e-100) rr = 0.0;
        if (std::abs(lr) < 1e-100) lr = 0.0;

        lr_ = lr;
        ll_ = ll;
        rr_ = rr;

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
     * @param dest     Destination array (null is tolerated: returns 0).
     * @param maxCount Capacity of dest (clamped to kGonioSize).
     * @return Number of points copied.
     */
    int getGonioPoints(GonioPoint* dest, int maxCount) const noexcept
    {
        if (dest == nullptr) return 0;
        const int count = std::clamp(maxCount, 0, kGonioSize);
        const int wp = gonioWrite_.load(std::memory_order_acquire);
        for (int i = 0; i < count; ++i)
        {
            const auto idx = static_cast<size_t>((wp - count + i) & (kGonioSize - 1));
            dest[i] = std::bit_cast<GonioPoint>(gonio_[idx].load(std::memory_order_relaxed));
        }
        return count;
    }

    /** @return Averaging time constant in ms from the last successful prepare(). */
    [[nodiscard]] double getWindowMs() const noexcept { return windowMs_; }

private:
    static_assert(sizeof(GonioPoint) == sizeof(std::uint64_t),
                  "GonioPoint must pack into one 64-bit atomic");
    static_assert(std::is_trivially_copyable_v<GonioPoint>);

    double windowMs_ = 300.0;
    double alpha_ = 0.001;
    std::atomic<bool> prepared_ { false };

    double lr_ = 0.0, ll_ = 0.0, rr_ = 0.0;

    int gonioDecim_ = 2;
    int decimCount_ = 0;
    std::array<std::atomic<std::uint64_t>, kGonioSize> gonio_ {};
    std::atomic<int> gonioWrite_ { 0 };

    std::atomic<T> correlation_ { T(0) };
    std::atomic<T> balance_ { T(0) };
};

} // namespace dspark
