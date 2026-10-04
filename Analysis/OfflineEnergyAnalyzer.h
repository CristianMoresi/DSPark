// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflineEnergyAnalyzer.h
 * @brief Complete-source linked or mid/side energy maps without retained PCM.
 *
 * OFFLINE ONLY. One complete source scan, O(frames * channels) time and
 * O(duration / bin duration + blockFrames * channels) storage. Linked maps use
 * 100 ms bins; the separate stereo analysis uses 10 ms bins. Bins use actual
 * frame counts, including the final short bin. Source positions are int64; the
 * source supplies bounded blocks. No FFT or beat analysis is performed for an
 * energy map.
 *
 * Threading: analyze() is a synchronous worker call. Returned analysis is
 * immutable and move-only; const access is safe while its owner is alive and
 * not moved. Dependencies: Core/OfflineProcessing.h.
 */

#include "../Core/OfflineProcessing.h"
#include "../Core/detail/ScaledSumSquares.h"

#if DSPARK_HAS_OFFLINE

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>

namespace dspark
{

/** @brief Complete-file fixed-duration energy analysis for offline processors.
 */
template <FloatType T> class OfflineEnergyAnalyzer final
{
  public:
    /** @brief Linked-channel RMS and maximum absolute sample for an actual
     * interval. */
    struct Bin
    {
        double rms = 0;
        double peak = 0;
        std::int64_t frames = 0;
    };

    /** @brief Read-only source-bound energy map; owns bins, never source samples.
     */
    class Analysis final
    {
      public:
        Analysis() = default;
        Analysis(Analysis &&) noexcept = default;
        Analysis &operator=(Analysis &&) noexcept = default;
        /** @brief False for a failed or moved-from result. */
        [[nodiscard]] bool isValid() const noexcept
        {
            return bins_ != nullptr;
        }
        /** @brief Returned spans remain valid only while this analysis is alive and
         * unmoved. */
        [[nodiscard]] std::span<const Bin> bins() const noexcept
        {
            return isValid() ? std::span<const Bin>(bins_.get(), count_) : std::span<const Bin>();
        }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept
        {
            return spec_;
        }
        [[nodiscard]] OfflineFingerprint fingerprint() const noexcept
        {
            return fingerprint_;
        }
        [[nodiscard]] std::int64_t binFrames() const noexcept
        {
            return hop_;
        }
        [[nodiscard]] double samplePeak() const noexcept
        {
            return peak_;
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid() ? count_ * sizeof(Bin) : 0;
        }

      private:
        friend class OfflineEnergyAnalyzer;
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_;
        std::int64_t hop_ = 0;
        std::size_t count_ = 0;
        double peak_ = 0;
        std::unique_ptr<Bin[]> bins_;
    };

    /** @brief Only a successful result owns an analysis; memoryBytes is requested
     * payload. */
    struct Result
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        Analysis analysis;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept
        {
            return offlineSucceeded(status);
        }
    };

    /** @brief Energy of M=(L+R)/2 and S=(L-R)/2 over the same actual interval. */
    struct MidSideBin
    {
        double midRms = 0, sideRms = 0;
        double midPeak = 0, sidePeak = 0;
        std::int64_t frames = 0;
    };

    /** @brief Immutable 10 ms stereo energy map, bound to the supplied PCM/clock.
     *
     * Analyze the rendered stereo signal when decisions concern generated audio:
     * changing generation width requires a new analysis of that resulting signal.
     * The complete-source RMS weights the last short bin by its actual duration.
     * Mid/side conversion avoids overflow for finite extreme double input.
     * Silence has zero RMS and peak; no side-share target is inferred here.
     */
    class MidSideAnalysis final
    {
      public:
        MidSideAnalysis() = default;
        MidSideAnalysis(MidSideAnalysis &&) noexcept = default;
        MidSideAnalysis &operator=(MidSideAnalysis &&) noexcept = default;
        [[nodiscard]] bool isValid() const noexcept
        {
            return bins_ != nullptr;
        }
        /** @brief Borrowed bins; valid while this analysis is alive and unmoved. */
        [[nodiscard]] std::span<const MidSideBin> bins() const noexcept
        {
            return isValid() ? std::span<const MidSideBin>(bins_.get(), count_)
                             : std::span<const MidSideBin>();
        }
        [[nodiscard]] OfflineAudioSpec getSpec() const noexcept
        {
            return spec_;
        }
        [[nodiscard]] OfflineFingerprint fingerprint() const noexcept
        {
            return fingerprint_;
        }
        /** @brief Fingerprint of double-precision side samples, for shared gain
         * rendering. */
        [[nodiscard]] OfflineFingerprint sideFingerprint() const noexcept
        {
            return sideFingerprint_;
        }
        [[nodiscard]] std::int64_t binFrames() const noexcept
        {
            return hop_;
        }
        [[nodiscard]] double samplePeak() const noexcept
        {
            return peak_;
        }
        [[nodiscard]] double midRms() const noexcept
        {
            return midRms_;
        }
        [[nodiscard]] double sideRms() const noexcept
        {
            return sideRms_;
        }
        [[nodiscard]] double sidePeak() const noexcept
        {
            return sidePeak_;
        }
        [[nodiscard]] std::size_t retainedBytes() const noexcept
        {
            return isValid() ? count_ * sizeof(MidSideBin) : 0;
        }

      private:
        friend class OfflineEnergyAnalyzer;
        OfflineAudioSpec spec_;
        OfflineFingerprint fingerprint_, sideFingerprint_;
        std::int64_t hop_ = 0;
        std::size_t count_ = 0;
        double peak_ = 0, midRms_ = 0, sideRms_ = 0, sidePeak_ = 0;
        std::unique_ptr<MidSideBin[]> bins_;
    };

    /** @brief Separate stereo result; failed calls never publish a partial map.
     */
    struct MidSideResult
    {
        OfflineStatus status = OfflineStatus::EmptyInput;
        MidSideAnalysis analysis;
        std::size_t memoryBytes = 0;
        [[nodiscard]] bool succeeded() const noexcept
        {
            return offlineSucceeded(status);
        }
    };

    /** @brief Analyzes a complete mono/stereo source, with no input mutation. */
    [[nodiscard]] Result analyze(OfflineAudioSource<T> &source,
                                 const OfflineJobOptions &options = {}) const
    {
        return measure<false>(source, options);
    }

    /** @brief Analyzes stereo PCM into separate central/lateral 10 ms energy
     * bins.
     *
     * One complete scan; no generation, gain processing, retained PCM or implicit
     * mono duplication. The format must have exactly two channels. Source offsets
     * are relative to zero; the original timelineOrigin remains in the result.
     */
    [[nodiscard]] MidSideResult analyzeMidSide(OfflineAudioSource<T> &source,
                                               const OfflineJobOptions &options = {}) const
    {
        return measure<true>(source, options);
    }

  private:
    template <bool Stereo>
    [[nodiscard]] static auto measure(OfflineAudioSource<T> &source,
                                      const OfflineJobOptions &options)
    {
        using Output = std::conditional_t<Stereo, MidSideResult, Result>;
        using Map = std::conditional_t<Stereo, MidSideAnalysis, Analysis>;
        using Cell = std::conditional_t<Stereo, MidSideBin, Bin>;
        Output result;
        try
        {
            detail::OfflineSession job(options);
            Map candidate;
            candidate.spec_ = source.getSpec();
            detail::offlineValidateSpec(candidate.spec_, 2);
            const auto spec = candidate.spec_;
            if constexpr (Stereo)
                if (spec.channels != 2)
                    detail::offlineFail(OfflineStatus::UnsupportedLayout);
            job.checkpoint(OfflinePhase::Analyze, 0, spec.frames);
            candidate.hop_ = std::max<std::int64_t>(
                1, static_cast<std::int64_t>(spec.sampleRate / (Stereo ? 100 : 10)));
            const auto count = static_cast<std::uint64_t>((spec.frames - 1) / candidate.hop_ + 1);
            candidate.bins_ = job.allocate<Cell>(count);
            candidate.count_ = static_cast<std::size_t>(count);
            detail::OfflineBlock<T> scratch(job, spec, options.blockFrames);
            // Scaled sum-of-squares keeps finite extreme double PCM measurable.
            // A causal RMS follower or unscaled SIMD square sum cannot supply this
            // exact-bin, overflow/underflow-safe statistic.
            detail::ScaledSumSquares energy, sideEnergy, totalMid, totalSide;
            std::int64_t inBin = 0;
            std::size_t bin = 0;
            for (std::int64_t first = 0; first < spec.frames;)
            {
                const int frames =
                    static_cast<int>(std::min<std::int64_t>(scratch.frames(), spec.frames - first));
                auto block = scratch.view(frames);
                detail::offlineRead(source, spec, first, block);
                for (int f = 0; f < frames; ++f)
                {
                    for (int c = 0; c < spec.channels; ++c)
                    {
                        const T value = block.getChannel(c)[f];
                        if (!std::isfinite(value))
                            detail::offlineFail(OfflineStatus::NonFiniteInput);
                        detail::offlineHash(candidate.fingerprint_, value);
                        if constexpr (!Stereo)
                            energy.add(static_cast<double>(value));
                        else
                            candidate.peak_ =
                                std::max(candidate.peak_, std::abs(static_cast<double>(value)));
                    }
                    if constexpr (Stereo)
                    {
                        const double left = static_cast<double>(block.getChannel(0)[f]);
                        const double right = static_cast<double>(block.getChannel(1)[f]);
                        const double mid = std::midpoint(left, right);
                        const double side = std::midpoint(left, -right);
                        energy.add(mid);
                        sideEnergy.add(side);
                        totalMid.add(mid);
                        totalSide.add(side);
                        detail::offlineHash(candidate.sideFingerprint_, side);
                    }
                    ++inBin;
                    if (inBin == candidate.hop_ || first + f == spec.frames - 1)
                    {
                        if constexpr (Stereo)
                        {
                            candidate.bins_[bin++] = {energy.rms(static_cast<double>(inBin)),
                                                      sideEnergy.rms(static_cast<double>(inBin)),
                                                      energy.peak(), sideEnergy.peak(), inBin};
                            sideEnergy.reset();
                        }
                        else
                        {
                            const double rms =
                                energy.rms(static_cast<double>(inBin) * spec.channels);
                            candidate.bins_[bin++] = {rms, energy.peak(), inBin};
                            candidate.peak_ = std::max(candidate.peak_, energy.peak());
                        }
                        inBin = 0;
                        energy.reset();
                    }
                }
                first += frames;
                job.checkpoint(OfflinePhase::Analyze, first, spec.frames);
            }
            if (source.getSpec() != spec)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            if constexpr (Stereo)
            {
                candidate.midRms_ = totalMid.rms(static_cast<double>(spec.frames));
                candidate.sideRms_ = totalSide.rms(static_cast<double>(spec.frames));
                candidate.sidePeak_ = totalSide.peak();
            }
            result.memoryBytes = job.bytes();
            result.analysis = std::move(candidate);
            result.status = OfflineStatus::Success;
        }
        catch (...)
        {
            result.status = detail::offlineExceptionStatus();
        }
        return result;
    }
};

} // namespace dspark

#endif // DSPARK_HAS_OFFLINE
