// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineGain.h
 * @brief Internal verified bandlimited gain rendering.
 * Threading: synchronous worker-only operations; plans and sources stay immutable.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "../TruePeakDetector.h"
#include "OfflineGainSource.h"
#include "OfflineWorker.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace dspark::detail
{

struct OfflineGainRenderSettings
{
    struct BoundaryTarget
    {
        OfflineRegion region;
        double peakGain = 1; // Relative to the complete source peak.
    };
    const OfflineExclusions *exclusions = nullptr;
    double featherFrames = 1;
    bool calibratePeak = false;
    std::array<BoundaryTarget, 2> boundaryTargets{};
};

// Intersection of peak half-planes with the admissible exterior-control box.
// Clipping preserves the complete feasible polygon, including line/point cases.
// The nearest point to zero changes only the necessary exterior continuation.
class OfflineBoundaryFeasibility final
{
  public:
    using Point = std::array<double, 2>;
    OfflineBoundaryFeasibility(OfflineSession &job, std::uint64_t constraints, Point lower)
        : capacity_(static_cast<std::size_t>(constraints + 4)),
          vertices_(job.allocate<Point>(constraints + 4)),
          scratch_(job.allocate<Point>(constraints + 4))
    {
        vertices_[0] = lower;
        vertices_[1] = {0, lower[1]};
        vertices_[2] = {0, 0};
        vertices_[3] = {lower[0], 0};
    }
    void constrain(double a, double b, double bound)
    {
        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(bound))
            offlineFail(OfflineStatus::NumericalFailure);
        if (count_ == 0)
            return;
        std::size_t next = 0;
        const auto append = [&](Point point) {
            if (next && point == scratch_[next - 1])
                return;
            if (next == capacity_)
                offlineFail(OfflineStatus::NumericalFailure);
            scratch_[next++] = point;
        };
        auto previous = vertices_[count_ - 1];
        double pv = a * previous[0] + b * previous[1] - bound;
        for (std::size_t i = 0; i < count_; ++i)
        {
            const auto current = vertices_[i];
            const double cv = a * current[0] + b * current[1] - bound;
            if ((pv > 0) != (cv > 0))
            {
                const double t = pv / (pv - cv);
                append({previous[0] + t * (current[0] - previous[0]),
                        previous[1] + t * (current[1] - previous[1])});
            }
            if (cv <= 0)
                append(current);
            previous = current;
            pv = cv;
        }
        if (next > 1 && scratch_[0] == scratch_[next - 1])
            --next;
        count_ = next;
        vertices_.swap(scratch_);
    }
    [[nodiscard]] bool feasible() const noexcept { return count_ != 0; }
    [[nodiscard]] Point closest() const noexcept
    {
        Point best{};
        double norm = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < count_; ++i)
        {
            const auto a = vertices_[i], b = vertices_[(i + 1) % count_];
            const double dx = b[0] - a[0], dy = b[1] - a[1];
            const double denominator = dx * dx + dy * dy;
            const double t = denominator > 0
                ? std::clamp(-(a[0] * dx + a[1] * dy) / denominator, 0., 1.) : 0;
            const Point point{a[0] + t * dx, a[1] + t * dy};
            const double squared = point[0] * point[0] + point[1] * point[1];
            if (squared < norm)
            {
                norm = squared;
                best = point;
            }
        }
        return best;
    }
  private:
    std::size_t capacity_, count_ = 4;
    std::unique_ptr<Point[]> vertices_, scratch_;
};

// Complete-file product maps are shared by calibration, measurement and output.
// Constant controls keep exact scalar behavior; zero amount preserves PCM bits.
template <class Result, FloatType T, class Report, class CursorFactory>
[[nodiscard]] Result offlineRenderGain(OfflineAudioSource<T> &source, const OfflineAudioSpec &spec,
                                       const OfflineFingerprint &expectedFingerprint,
                                       double inputPeak, bool valid, bool changed,
                                       const Report &report, OfflineAudioSink<T> &sink,
                                       const OfflineJobOptions &options, CursorFactory makeCursor,
                                       OfflineGainRenderSettings settings = {})
{
    Result result;
    OfflineSinkTransaction<T> transaction;
    try
    {
        if (!valid)
            detail::offlineFail(OfflineStatus::InvalidInput);
        if (source.getSpec() != spec)
            detail::offlineFail(OfflineStatus::SourceMismatch);
        detail::OfflineSession job(options);
        job.checkpoint(OfflinePhase::Verify, 0, spec.frames);
        result.report = report;
        auto cursor = makeCursor(job);
        using Cursor = decltype(cursor);
        OfflineGainSource<T, Cursor> cached(source, spec, expectedFingerprint, inputPeak, job,
                                           options, std::move(cursor));
        std::optional<OfflineGainProduct<T, Cursor>> product;
        std::uint64_t boundaryFrames = 0;
        for (const auto target : settings.boundaryTargets)
        {
            if (target.region.begin < 0 || target.region.end < target.region.begin ||
                target.region.end > spec.frames || !std::isfinite(target.peakGain) ||
                !(target.peakGain > 0) || target.peakGain > 1)
                offlineFail(OfflineStatus::InvalidInput);
            boundaryFrames += static_cast<std::uint64_t>(target.region.end - target.region.begin);
        }
        if (changed && inputPeak > 0 && !cached.constant())
            product.emplace(cached, job, boundaryFrames != 0);
        result.report.renderInfo.bandlimited = product.has_value();
        result.report.renderInfo.boundaryGain = cached.baseline();
        result.report.renderInfo.leftBoundaryGain = cached.endpoint(0);
        result.report.renderInfo.rightBoundaryGain = cached.endpoint(1);
        result.report.renderInfo.minimumControlGain = cached.minimumGain();
        result.report.renderInfo.maximumControlGain = cached.maximumGain();
        const int blockSize = cached.block();
        auto output = job.allocate<T>(static_cast<std::uint64_t>(blockSize) * spec.channels);
        result.memoryBytes = job.bytes();
        TruePeakDetector<double, 2> detector;
        double outputPeak = 0, normalizedTruePeak = 0;
        const double normalizer = inputPeak > 0 ? inputPeak : 1;
        const double maximumSample = static_cast<double>(std::numeric_limits<T>::max());
        const auto runPass = [&](OfflinePhase phase, auto &&consume, bool boundaryOnly = false) {
            cached.reset();
            if (product)
                product->reset();
            job.checkpoint(phase, 0, spec.frames);
            for (std::size_t leaf = 0; leaf < cached.leaves(); ++leaf)
            {
                const auto first = static_cast<std::int64_t>(leaf) * blockSize;
                const int frames =
                    static_cast<int>(std::min<std::int64_t>(blockSize, spec.frames - first));
                if (boundaryOnly && std::none_of(settings.boundaryTargets.begin(),
                    settings.boundaryTargets.end(), [&](const auto &target) {
                        return first < target.region.end && first + frames > target.region.begin;
                    }))
                    continue;
                const double *delta = product ? product->evaluate(leaf) : nullptr;
                const T *input = cached.audio(leaf, 0);
                const double *gains = cached.gain(leaf);
                consume(first, frames, input, gains, delta);
                job.checkpoint(phase, first + frames, spec.frames);
            }
            cached.verifySpec();
        };
        const auto maskAt = [&](std::int64_t frame) {
            return settings.exclusions
                       ? settings.exclusions->apply(frame, 2, settings.featherFrames) - 1
                       : 1;
        };
        const auto constrainedDelta = [](double x, double gain, double delta, double mask) {
            if (mask == 0)
                return 0.;
            const double native = x * (gain - 1);
            return native + mask * (delta - native);
        };
        double scale = 1;
        if (product && boundaryFrames)
        {
            (void)offlineBytes(boundaryFrames, static_cast<std::size_t>(2 * spec.channels) *
                                                  sizeof(OfflineBoundaryFeasibility::Point));
            const auto constraints = boundaryFrames * static_cast<std::uint64_t>(2 * spec.channels);
            (void)offlineBytes(constraints + 4, sizeof(OfflineBoundaryFeasibility::Point));
            const std::array<double, 2> lower{
                settings.boundaryTargets[0].region.end > settings.boundaryTargets[0].region.begin
                    ? -cached.endpoint(0) : 0,
                settings.boundaryTargets[1].region.end > settings.boundaryTargets[1].region.begin
                    ? -cached.endpoint(1) : 0};
            OfflineBoundaryFeasibility feasible(job, constraints, lower);
            runPass(OfflinePhase::Verify, [&](std::int64_t first, int frames, const T *input,
                                             const double *gains, const double *delta) {
                const auto *left = product->boundaryResponse(0);
                const auto *right = product->boundaryResponse(1);
                for (const auto target : settings.boundaryTargets)
                    for (auto frame = std::max(first, target.region.begin);
                         frame < std::min(first + frames, target.region.end); ++frame)
                    {
                        const int f = static_cast<int>(frame - first);
                        const double mask = maskAt(frame);
                        const double bound = offlineRepresentablePeakGain<T>(normalizer, target.peakGain);
                        for (int c = 0; c < spec.channels; ++c)
                        {
                            const int k = c * blockSize + f;
                            const double x = static_cast<double>(input[k]) / normalizer;
                            const double y = x + constrainedDelta(x, gains[f], delta[k], mask);
                            // Exclusions and their feathers retain the scalar plan's
                            // admissible level; no target overrides protected PCM.
                            const double limit = std::max(bound, std::abs(x * gains[f]));
                            const double a = mask * left[k], b = mask * right[k];
                            feasible.constrain(a, b, limit - y);
                            feasible.constrain(-a, -b, limit + y);
                        }
                    }
            }, true);
            result.report.renderInfo.targetFeasible = feasible.feasible();
            if (!feasible.feasible())
                offlineFail(OfflineStatus::NumericalFailure);
            const auto delta = feasible.closest();
            auto &info = result.report.renderInfo;
            info.leftBoundaryGain = std::clamp(cached.endpoint(0) + delta[0], 0., cached.endpoint(0));
            info.rightBoundaryGain = std::clamp(cached.endpoint(1) + delta[1], 0., cached.endpoint(1));
            info.boundaryCalibrated = delta[0] != 0 || delta[1] != 0;
            product->setEndpoints(info.leftBoundaryGain, info.rightBoundaryGain);
        }
        // The finite Hilbert row sum is below 2*(1+log(N))/pi. Bound each term
        // of the product identity, including the affine control extension.
        // A factor of two reserves numerical error. Ordinary PCM then avoids a
        // redundant complete-source representability pass; every final sample
        // is still checked before publication. Exact-PCM feathering is convex
        // between the scalar and bandlimited deltas and respects the same bound.
        const double rowSum = 2 * (1 + std::log(static_cast<double>(spec.frames))) / pi<double>;
        const double controlRange = std::max(std::abs(cached.minimumGain() - cached.baseline()),
                                             std::abs(cached.maximumGain() - cached.baseline()));
        // Each endpoint commutator has a diagonal <=1/8 and two Cauchy
        // terms. Centering its digamma weights bounds them by 1+log(N).
        const double endpointBound = .125 + 2 * (1 + std::log(static_cast<double>(spec.frames))) *
                                               rowSum / pi<double>;
        const double endpointRange =
            std::abs(result.report.renderInfo.leftBoundaryGain - cached.baseline()) +
            std::abs(result.report.renderInfo.rightBoundaryGain - cached.baseline());
        const double deltaBound = std::abs(cached.baseline() - 1) +
                                  (.75 + 3 * rowSum * rowSum) * controlRange +
                                  endpointBound * endpointRange;
        const bool couldOverflow = normalizer >= maximumSample / (2 * (1 + deltaBound));
        if (product && (settings.calibratePeak || couldOverflow))
        {
            double lo = 0;
            double hi = settings.calibratePeak && cached.minimumGain() < 1
                            ? 1 / (1 - cached.minimumGain())
                            : 1;
            const double bound =
                settings.calibratePeak
                    ? std::nextafter(cached.scalarPeak(), std::numeric_limits<double>::infinity())
                    : maximumSample / normalizer;
            // Reserve roundoff in the affine sum and final rescaling. Original
            // finite PCM remains admissible, including an exact TYPE_MAX sample.
            const double safeBound =
                settings.calibratePeak
                    ? bound
                    : bound * (1 - 8 * std::numeric_limits<double>::epsilon());
            bool feasible = true;
            runPass(OfflinePhase::Verify, [&](std::int64_t first, int frames, const T *input,
                                             const double *gains, const double *delta) {
                for (int f = 0; f < frames; ++f)
                {
                    const double mask = maskAt(first + f);
                    for (int c = 0; c < spec.channels; ++c)
                    {
                        const int k = c * blockSize + f;
                        const double x = static_cast<double>(input[k]) / normalizer;
                        const double d = constrainedDelta(x, gains[f], delta[k], mask);
                        if (!std::isfinite(d))
                            offlineFail(OfflineStatus::NumericalFailure);
                        if (d == 0)
                            feasible = feasible && std::abs(x) <= bound;
                        else
                        {
                            const double limit = settings.calibratePeak
                                                     ? bound
                                                     : std::max(safeBound, std::abs(x));
                            double a = (-limit - x) / d, b = (limit - x) / d;
                            if (a > b)
                                std::swap(a, b);
                            lo = std::max(lo, a);
                            hi = std::min(hi, b);
                        }
                    }
                }
            });
            result.report.renderInfo.targetFeasible = feasible && lo <= hi;
            if (!result.report.renderInfo.targetFeasible)
                offlineFail(OfflineStatus::NumericalFailure);
            scale = std::clamp(1., lo, hi);
            if (!settings.calibratePeak && scale < 1 && scale > 0)
                scale = std::nextafter(scale, 0.);
            result.report.renderInfo.deltaScale = scale;
            result.report.renderInfo.representabilityLimited = !settings.calibratePeak && scale < 1;
        }
        result.report.renderInfo.minimumControlGain =
            std::max(0., 1 + scale * (cached.minimumGain() - 1));
        result.report.renderInfo.maximumControlGain =
            std::max(0., 1 + scale * (cached.maximumGain() - 1));
        // Source and target calibration are complete. Measure the actual rounded
        // PCM while writing provisional blocks; any later failure aborts them.
        // Publication happens only after source checks and the meter tail finish.
        transaction.sink = &sink;
        offlineCallSink([&] { return sink.begin(spec); });
        runPass(OfflinePhase::Render, [&](std::int64_t first, int frames, const T *input,
                                          const double *gains, const double *delta) {
            for (int f = 0; f < frames; ++f)
            {
                const double gain = gains[f];
                const double mask = maskAt(first + f);
                for (int c = 0; c < spec.channels; ++c)
                {
                    const int k = c * blockSize + f;
                    const T x = input[k];
                    T y = x;
                    if (delta && mask != 0 && scale != 0)
                    {
                        const double xn = static_cast<double>(x) / normalizer;
                        const double d = constrainedDelta(xn, gain, delta[k], mask);
                        const double value = (xn + scale * d) * normalizer;
                        if (!std::isfinite(value) || std::abs(value) > maximumSample)
                            detail::offlineFail(OfflineStatus::NumericalFailure);
                        y = static_cast<T>(value);
                    }
                    else if (!delta && gain != 1)
                    {
                        const double value = static_cast<double>(x) * gain;
                        if (!std::isfinite(value) || std::abs(value) > maximumSample)
                            detail::offlineFail(OfflineStatus::NumericalFailure);
                        y = static_cast<T>(value);
                    }
                    if (!std::isfinite(y))
                        detail::offlineFail(OfflineStatus::NumericalFailure);
                    outputPeak = std::max(outputPeak, std::abs(static_cast<double>(y)));
                    normalizedTruePeak = std::max(
                        normalizedTruePeak,
                        detector.processSample(static_cast<double>(y) / normalizer, c));
                    output[k] = y;
                }
            }
            for (int offset = 0; offset < frames;)
            {
                const int count = std::min(options.blockFrames, frames - offset);
                std::array<const T *, 2> channels{};
                for (int c = 0; c < spec.channels; ++c)
                    channels[c] = output.get() + c * blockSize + offset;
                offlineCallSink([&] {
                    return sink.write(first + offset, AudioBufferView<const T>(
                                                         channels.data(), spec.channels, count));
                });
                offset += count;
            }
        });
        for (int i = 0; i < TruePeakDetector<double, 2>::getTaps() - 1; ++i)
            for (int c = 0; c < spec.channels; ++c)
                normalizedTruePeak = std::max(normalizedTruePeak, detector.processSample(0, c));
        const double silence = -std::numeric_limits<double>::infinity();
        result.report.outputSamplePeakDb = gainToDecibels(outputPeak, silence);
        result.report.outputTruePeakDb =
            normalizedTruePeak > 0
                ? gainToDecibels(normalizedTruePeak, silence) + gainToDecibels(normalizer, silence)
                : silence;
        result.report.peaksMeasured = true;
        job.checkpoint(OfflinePhase::Render, spec.frames, spec.frames);
        offlineCallSink([&] { return sink.commit(); });
        transaction.sink = nullptr;
        result.memoryBytes = job.bytes();
        result.status = changed && scale != 0 ? OfflineStatus::Success : OfflineStatus::NoChange;
    }
    catch (...)
    {
        result.status = detail::offlineExceptionStatus();
    }
    return result;
}

} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
