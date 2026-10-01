// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineWorker.h
 * @brief Shared worker validation, exact exclusions and owning output transactions.
 * Threading: synchronous worker operations; sources and plans remain immutable.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <utility>

namespace dspark::detail
{
// Round a peak target toward the input so PCM quantization cannot demand more
// global reduction than requested. Division before comparison remains meaningful
// when the input/target themselves are subnormal, including denorm_min.
template <FloatType T>
[[nodiscard]] inline double offlineRepresentablePeakGain(double inputPeak, double gain) noexcept
{
    T target = static_cast<T>(inputPeak * gain);
    if (target == T(0))
        target = std::numeric_limits<T>::denorm_min();
    else if (static_cast<double>(target) / inputPeak < gain)
        target = std::nextafter(target, std::numeric_limits<T>::infinity());
    return std::min(1.0, static_cast<double>(target) / inputPeak);
}

struct OfflineSourceSummary
{
    OfflineFingerprint fingerprint;
    double peak = 0;
};

// Full-source validation and summary without a retained feature map. Magnitude
// is channel-linked; consumers choose which additional statistics they need.
template <FloatType T, class Consume>
inline OfflineSourceSummary offlineScanSource(OfflineAudioSource<T> &source,
                                              const OfflineAudioSpec &spec,
                                              OfflineBlock<T> &scratch, const OfflineSession &job,
                                              OfflinePhase phase, Consume consume)
{
    job.checkpoint(phase, 0, spec.frames);
    OfflineSourceSummary summary;
    for (std::int64_t first = 0; first < spec.frames;)
    {
        const int frames =
            static_cast<int>(std::min<std::int64_t>(scratch.frames(), spec.frames - first));
        auto block = scratch.view(frames);
        detail::offlineRead(source, spec, first, block);
        for (int f = 0; f < frames; ++f)
        {
            double magnitude = 0;
            for (int c = 0; c < spec.channels; ++c)
            {
                const T x = block.getChannel(c)[f];
                if (!std::isfinite(x))
                    detail::offlineFail(OfflineStatus::NonFiniteInput);
                detail::offlineHash(summary.fingerprint, x);
                magnitude = std::max(magnitude, std::abs(static_cast<double>(x)));
            }
            summary.peak = std::max(summary.peak, magnitude);
            consume(first + f, magnitude);
        }
        first += frames;
        job.checkpoint(phase, first, spec.frames);
    }
    if (source.getSpec() != spec)
        detail::offlineFail(OfflineStatus::SourceMismatch);
    return summary;
}

// Verify an existing full-source analysis while visiting sample magnitudes.
template <FloatType T, class Consume>
inline void offlineScanMagnitude(OfflineAudioSource<T> &source, const OfflineAudioSpec &spec,
                                 const OfflineFingerprint &expectedFingerprint, double inputPeak,
                                 OfflineBlock<T> &scratch, const OfflineSession &job,
                                 Consume consume)
{
    const auto summary = offlineScanSource(
        source, spec, scratch, job, OfflinePhase::Plan, [&](std::int64_t frame, double magnitude) {
            if (magnitude > inputPeak)
                detail::offlineFail(OfflineStatus::SourceMismatch);
            consume(frame, magnitude);
        });
    if (summary.fingerprint != expectedFingerprint)
        detail::offlineFail(OfflineStatus::SourceMismatch);
}

class OfflineExclusions final
{
  public:
    void assign(OfflineSession &job, std::span<const OfflineRegion> regions, std::int64_t frames)
    {
        capacity_ = regions.size();
        if (capacity_)
        {
            regions_ = job.allocate<OfflineRegion>(capacity_);
            std::copy(regions.begin(), regions.end(), regions_.get());
            for (const auto r : regions)
                if ((r.begin < 0) || (r.end <= r.begin) || (r.end > frames))
                    detail::offlineFail(OfflineStatus::InvalidInput);
            std::sort(regions_.get(), regions_.get() + capacity_,
                      [](OfflineRegion a, OfflineRegion b) { return a.begin < b.begin; });
            for (std::size_t i = 0; i < capacity_; ++i)
            {
                const auto r = regions_[i];
                if (count_ && r.begin <= regions_[count_ - 1].end)
                    regions_[count_ - 1].end = std::max(r.end, regions_[count_ - 1].end);
                else
                    regions_[count_++] = r;
            }
        }
    }
    [[nodiscard]] std::span<const OfflineRegion> view() const noexcept
    {
        return {regions_.get(), count_};
    }
    [[nodiscard]] std::size_t retainedBytes() const noexcept
    {
        return capacity_ * sizeof(OfflineRegion);
    }
    [[nodiscard]] bool covers(std::int64_t frames) const noexcept
    {
        return count_ == 1 && regions_[0].begin == 0 && regions_[0].end == frames;
    }
    [[nodiscard]] double apply(std::int64_t frame, double gain, double featherFrames) const noexcept
    {
        if (count_ == 0 || gain == 1)
            return gain;
        std::size_t lo = 0, hi = count_;
        while (lo < hi)
        {
            const auto mid = lo + (hi - lo) / 2;
            if (regions_[mid].end <= frame)
                lo = mid + 1;
            else
                hi = mid;
        }
        double right = 1, left = 1;
        if (lo < count_)
        {
            if (frame >= regions_[lo].begin)
                return 1;
            right = std::clamp(static_cast<double>(regions_[lo].begin - frame) / featherFrames, 0.0,
                               1.0);
        }
        if (lo > 0)
            left = std::clamp(static_cast<double>(frame - regions_[lo - 1].end) / featherFrames,
                              0.0, 1.0);
        if (left == 1 && right == 1)
            return gain;
        // A nearest-boundary minimum has a cusp where two feathers meet.
        // Multiplying masks removes that corner. Three zero edge derivatives
        // suppress modulation sidebands without touching protected PCM.
        const auto smooth = [](double t) {
            const double u = std::min(t, 1 - t);
            const double value = u * u * u * u * (35 - u * (84 - u * (70 - 20 * u)));
            return t <= .5 ? value : 1 - value;
        };
        return 1 + (gain - 1) * smooth(left) * smooth(right);
    }

  private:
    std::size_t count_ = 0, capacity_ = 0;
    std::unique_ptr<OfflineRegion[]> regions_;
};

template <FloatType T> struct OfflineSinkTransaction
{
    OfflineAudioSink<T> *sink = nullptr;
    ~OfflineSinkTransaction()
    {
        if (sink)
            sink->abort();
    }
};

template <class Function> inline void offlineCallSink(Function &&function)
{
    bool ok = false;
    try
    {
        ok = function();
    }
    catch (const std::bad_alloc &)
    {
        offlineFail(OfflineStatus::AllocationFailure);
    }
    catch (...)
    {
        offlineFail(OfflineStatus::SinkFailed);
    }
    if (!ok)
        offlineFail(OfflineStatus::SinkFailed);
}

template <class Processor, FloatType T, int MaxChannels, class Options>
[[nodiscard]] typename Processor::Result offlineRun(const Processor &processor,
                                                    const AudioBuffer<T, MaxChannels> &input,
                                                    AudioBuffer<T, MaxChannels> &output,
                                                    double sampleRate, const Options &options,
                                                    const OfflineJobOptions &job)
{
    typename Processor::Result result;
    try
    {
        OfflineBufferSource<T, MaxChannels> source(input.toView(), sampleRate);
        auto prepared = processor.analyze(source, options, job);
        if (!prepared.succeeded())
        {
            result.status = prepared.status;
            return result;
        }
        auto outputSpec = source.getSpec();
        if constexpr (requires { prepared.plan.getOutputSpec(); })
            outputSpec = prepared.plan.getOutputSpec();
        detail::offlineValidateSpec(outputSpec, MaxChannels);
        if (outputSpec.frames > std::numeric_limits<int>::max())
            detail::offlineFail(OfflineStatus::MemoryLimit);
        const auto sampleBytes =
            detail::offlineBytes(static_cast<std::uint64_t>(outputSpec.frames), sizeof(T));
        if (sampleBytes > std::numeric_limits<std::size_t>::max() - 31)
            detail::offlineFail(OfflineStatus::MemoryLimit);
        const auto outputBytes = detail::offlineBytes(
            (sampleBytes + 31) / 32, 32 * static_cast<std::size_t>(outputSpec.channels));
        auto renderJob = detail::offlineRemaining(job, prepared.plan.retainedBytes());
        renderJob = detail::offlineRemaining(renderJob, outputBytes);
        class BufferSink final : public OfflineAudioSink<T>
        {
          public:
            explicit BufferSink(AudioBuffer<T, MaxChannels> &destination)
                : destination_(destination)
            {
            }
            bool begin(const OfflineAudioSpec &spec) override
            {
                candidate_.resize(spec.channels, static_cast<int>(spec.frames));
                return true;
            }
            bool write(std::int64_t first, AudioBufferView<const T> block) override
            {
                for (int c = 0; c < block.getNumChannels(); ++c)
                    std::memcpy(candidate_.getChannel(c) + first, block.getChannel(c),
                                static_cast<std::size_t>(block.getNumSamples()) * sizeof(T));
                return true;
            }
            bool commit() override
            {
                destination_ = std::move(candidate_);
                return true;
            }
            void abort() noexcept override
            {
                candidate_ = AudioBuffer<T, MaxChannels>();
            }

          private:
            AudioBuffer<T, MaxChannels> &destination_;
            AudioBuffer<T, MaxChannels> candidate_;
        } sink(output);
        result = processor.render(source, prepared.plan, sink, renderJob);
        result.memoryBytes = std::max(
            prepared.memoryBytes, result.memoryBytes + prepared.plan.retainedBytes() + outputBytes);
    }
    catch (...)
    {
        result.status = detail::offlineExceptionStatus();
    }
    return result;
}
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
