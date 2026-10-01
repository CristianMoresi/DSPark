// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineStereo.h
 * @brief Shared complete-source delta transport for offline stereo generation.
 * Threading: one worker owns each stream, source and provisional consumer.
 */
#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "../../Effects/StereoGenerator.h"
#include "OfflineWorker.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <span>

namespace dspark::detail
{
// Supplies original PCM and the canonical unit-width delta at the same source
// frame. No whole-file PCM is retained. A mono source is duplicated explicitly
// by the caller's validated plan; the original ring itself keeps one channel.
template <FloatType T> class OfflineStereoStream final
{
    OfflineAudioSource<T> &source_;
    OfflineAudioSpec spec_;
    OfflineFingerprint expected_;
    double inputPeak_;
    const OfflineSession &job_;
    OfflineBlock<T> input_, aligned_;
    std::optional<StereoGenerator<double>> generator_;
    std::unique_ptr<T[]> original_;
    int latency_ = 0, ringSize_ = 0;

  public:
    OfflineStereoStream(OfflineAudioSource<T> &source, OfflineAudioSpec spec,
                        OfflineFingerprint fingerprint, double inputPeak,
                        StereoGenerator<double>::Options options, bool generate,
                        OfflineSession &job, int blockFrames)
        : source_(source), spec_(spec), expected_(fingerprint), inputPeak_(inputPeak), job_(job),
          input_(job, spec, std::min(256, blockFrames)),
          aligned_(job, spec, std::min(256, blockFrames))
    {
        if (generate)
        {
            const AudioSpec environment{spec.sampleRate, 256, 2};
            const auto memory =
                StereoGenerator<double>::getPrepareMemoryBound(environment, options);
            if (memory == 0)
                offlineFail(OfflineStatus::InvalidInput);
            job.charge(memory);
            generator_.emplace();
            generator_->setWidth(1);
            if (!generator_->prepare(environment, options))
                offlineFail(OfflineStatus::InvalidInput);
            latency_ = generator_->getLatency();
        }
        ringSize_ = latency_ + input_.frames();
        original_ = job.template allocate<T>(static_cast<std::uint64_t>(ringSize_) * spec.channels);
    }
    [[nodiscard]] int latency() const noexcept
    {
        return latency_;
    }

    template <class Consume> void run(Consume consume, OfflinePhase phase = OfflinePhase::Render)
    {
        job_.checkpoint(phase, 0, spec_.frames);
        if (generator_)
            generator_->resetAtFrame(static_cast<std::uint64_t>(spec_.timelineOrigin));
        OfflineFingerprint fingerprint;
        int write = 0, read = 0, skip = latency_;
        std::int64_t written = 0;
        const auto block = [&](std::int64_t first, int count, bool flush) {
            auto input = input_.view(count);
            if (!flush)
                offlineRead(source_, spec_, first, input);
            std::array<std::array<double, 256>, 2> work{};
            std::array<double, 256> delta{};
            for (int i = 0; i < count; ++i)
            {
                for (int c = 0; c < spec_.channels; ++c)
                {
                    const T sample = flush ? T(0) : input.getChannel(c)[i];
                    if (!std::isfinite(sample))
                        offlineFail(OfflineStatus::NonFiniteInput);
                    if (!flush)
                    {
                        offlineHash(fingerprint, sample);
                        if (std::abs(static_cast<double>(sample)) > inputPeak_)
                            offlineFail(OfflineStatus::SourceMismatch);
                    }
                    original_[static_cast<std::size_t>(c) * ringSize_ + write] = sample;
                    work[c][i] = static_cast<double>(sample);
                }
                if (spec_.channels == 1)
                    work[1][i] = work[0][i];
                write = (write + 1) % ringSize_;
            }
            if (generator_)
            {
                double *channels[]{work[0].data(), work[1].data()};
                if (!generator_->processBlock({channels, 2, count},
                                              {delta.data(), static_cast<std::size_t>(count)}))
                    offlineFail(OfflineStatus::NumericalFailure);
            }
            const int start = std::min(skip, count);
            skip -= start;
            const int produced = count - start;
            auto original = aligned_.view(produced);
            for (int i = 0; i < produced; ++i)
            {
                for (int c = 0; c < spec_.channels; ++c)
                    original.getChannel(c)[i] =
                        original_[static_cast<std::size_t>(c) * ringSize_ + read];
                read = (read + 1) % ringSize_;
            }
            if (produced)
                consume(written, original,
                        std::span<const double>(delta.data() + start,
                                                static_cast<std::size_t>(produced)));
            written += produced;
        };
        for (std::int64_t first = 0; first < spec_.frames;)
        {
            const int count =
                static_cast<int>(std::min<std::int64_t>(input_.frames(), spec_.frames - first));
            block(first, count, false);
            first += count;
            job_.checkpoint(phase, first, spec_.frames);
        }
        // Do not form frames + latency in the signed source clock.
        for (int remaining = latency_; remaining > 0;)
        {
            const int count = std::min(input_.frames(), remaining);
            block(spec_.frames, count, true);
            remaining -= count;
            job_.checkpoint(phase, spec_.frames, spec_.frames);
        }
        if (written != spec_.frames || fingerprint != expected_ || source_.getSpec() != spec_)
            offlineFail(OfflineStatus::SourceMismatch);
    }
};
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
