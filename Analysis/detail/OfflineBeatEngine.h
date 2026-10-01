// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineBeatEngine.h
 * @brief Internal budgeted feature input to the existing BeatTracker engine.
 * Threading: synchronous worker operation with per-job storage and cancellation.
 */
#include "../../Core/OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "../BeatTracker.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>

namespace dspark::detail
{

// Repeated local feature windows reuse bounded scratch blocks. Each block and
// its bookkeeping are one charged allocation; failure cannot leave partially
// published allocator metadata. Reuse requests no payload and needs no refund.
class OfflineBeatMemory final : public std::pmr::memory_resource
{
  public:
    explicit OfflineBeatMemory(OfflineSession &job) : job_(job) {}
    ~OfflineBeatMemory() override
    {
        while (blocks_)
        {
            auto *block = blocks_;
            blocks_ = block->next;
            if (block->alignment > alignof(std::max_align_t))
                ::operator delete(block, std::align_val_t{block->alignment});
            else
                ::operator delete(block);
        }
    }
  private:
    struct alignas(std::max_align_t) Block
    {
        Block *next;
        void *data;
        std::size_t capacity, alignment;
        bool available;
    };
    void *do_allocate(std::size_t bytes, std::size_t alignment) override
    {
        Block *best = nullptr;
        for (auto *b = blocks_; b; b = b->next)
            if (b->available && b->capacity >= bytes && b->alignment >= alignment &&
                (!best || b->capacity < best->capacity))
                best = b;
        if (best)
        {
            best->available = false;
            return best->data;
        }
        std::size_t capacity = 1;
        while (capacity < bytes)
        {
            if (capacity > std::numeric_limits<std::size_t>::max() / 2)
                offlineFail(OfflineStatus::MemoryLimit);
            capacity *= 2;
        }
        alignment = std::max(alignment, alignof(Block));
        if (alignment > std::numeric_limits<std::size_t>::max() - sizeof(Block))
            offlineFail(OfflineStatus::MemoryLimit);
        const auto offset = (sizeof(Block) + alignment - 1) / alignment * alignment;
        if (capacity > std::numeric_limits<std::size_t>::max() - offset)
            offlineFail(OfflineStatus::MemoryLimit);
        const auto total = offset + capacity;
        job_.charge(total);
        void *raw = alignment > alignof(std::max_align_t)
                        ? ::operator new(total, std::align_val_t{alignment}) : ::operator new(total);
        auto *block = new (raw) Block{blocks_, static_cast<std::byte *>(raw) + offset,
                                     capacity, alignment, false};
        blocks_ = block;
        return block->data;
    }
    void do_deallocate(void *p, std::size_t, std::size_t) override
    {
        for (auto *b = blocks_; b; b = b->next)
            if (b->data == p)
            {
                b->available = true;
                return;
            }
    }
    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override
    {
        return this == &other;
    }
    OfflineSession &job_;
    Block *blocks_ = nullptr;
};

template <FloatType T> struct OfflineBeatEngine final
{
    struct Result
    {
        T tempoBpm = 0, secondaryTempoBpm = 0, confidence = 0;
        std::unique_ptr<std::int64_t[]> positions;
        std::size_t count = 0;
        [[nodiscard]] std::span<const std::int64_t> beats() const noexcept
        {
            return positions ? std::span<const std::int64_t>(positions.get(), count)
                             : std::span<const std::int64_t>();
        }
    };
    // FrameAt returns the shared frontend's OdfFrame with an explicit source
    // reference. No FFT, audio copy or second tempo algorithm is constructed.
    template <typename FrameAt>
    [[nodiscard]] static Result
    analyze(double sampleRate, int hop, std::size_t count, FrameAt frameAt,
            double minimumBpm, double maximumBpm, double tightness, OfflineSession &job,
            std::pmr::memory_resource *workspace = nullptr)
    {
        if (!std::isfinite(sampleRate) || sampleRate < 8000 || sampleRate > 384000 || hop < 1 ||
            hop > sampleRate || !std::isfinite(minimumBpm) || !std::isfinite(maximumBpm) ||
            minimumBpm < 20 || maximumBpm > 480 || minimumBpm > maximumBpm ||
            !std::isfinite(tightness) || tightness < 0.001 ||
            tightness > static_cast<double>(std::numeric_limits<T>::max()))
            offlineFail(OfflineStatus::InvalidInput);
        // The shared DP uses int envelope indices; source references stay int64.
        // Reserve arithmetic headroom for its bounded lag/window additions.
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2))
            offlineFail(OfflineStatus::MemoryLimit);
        job.checkpoint(OfflinePhase::Analyze, 0, static_cast<std::int64_t>(count));
        if (count == 0)
            return {};

        BeatTracker<T> tracker;
        tracker.allocationAccount_ = {&job, [](void *context, std::size_t bytes) {
                                          static_cast<OfflineSession *>(context)->charge(bytes);
                                      }};
        if (workspace)
            tracker.allocationAccount_ = {
                workspace, nullptr,
                [](void *context, std::size_t bytes, std::size_t alignment) {
                    return static_cast<std::pmr::memory_resource *>(context)->allocate(bytes, alignment);
                },
                [](void *context, void *p, std::size_t bytes, std::size_t alignment) {
                    static_cast<std::pmr::memory_resource *>(context)->deallocate(p, bytes, alignment);
                }};
        tracker.offlineCheckpointContext_ = &job;
        tracker.offlineCheckpoint_ = [](void *context) {
            static_cast<OfflineSession *>(context)->checkpoint(OfflinePhase::Analyze, 0, 0);
        };
        tracker.sampleRate_ = sampleRate;
        tracker.hop_ = hop;
        tracker.frameRate_ = sampleRate / hop;
        tracker.buildBank();
        tracker.setTempoRange(static_cast<T>(minimumBpm), static_cast<T>(maximumBpm));
        tracker.setTightness(static_cast<T>(tightness));
        tracker.env_.reserve(count);
        tracker.envRef_.reserve(count);
        for (auto &registers : tracker.envReg_)
            registers.reserve(count);
        std::int64_t previous = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            if ((i & 4095) == 0)
                job.checkpoint(OfflinePhase::Analyze, static_cast<std::int64_t>(i),
                               static_cast<std::int64_t>(count));
            const auto frame = frameAt(i);
            if (!std::isfinite(frame.value) || frame.value < T(0))
                offlineFail(OfflineStatus::NonFiniteInput);
            if (i && (previous > std::numeric_limits<std::int64_t>::max() - hop ||
                      frame.referenceSample != previous + hop))
                offlineFail(OfflineStatus::InvalidInput);
            // beatSample() interpolates by at most half a hop on either side.
            if ((frame.referenceSample < std::numeric_limits<std::int64_t>::min() + hop) ||
                (frame.referenceSample > std::numeric_limits<std::int64_t>::max() - hop))
                offlineFail(OfflineStatus::InvalidInput);
            tracker.env_.push_back(static_cast<double>(frame.value));
            tracker.envRef_.push_back(frame.referenceSample);
            for (std::size_t r = 0; r < frame.registers.size(); ++r)
            {
                const auto value = frame.registers[r];
                if (!std::isfinite(value) || value < T(0))
                    offlineFail(OfflineStatus::NonFiniteInput);
                tracker.envReg_[r].push_back(static_cast<double>(value));
            }
            previous = frame.referenceSample;
        }
        Result result;
        struct Output
        {
            Result &result;
            OfflineSession &job;
        } output{result, job};
        const auto summary = tracker.finishEnvelope(
            &output, [](void *context, std::span<const std::int64_t> beats) {
                auto &out = *static_cast<Output *>(context);
                if (!beats.empty())
                {
                    out.result.positions = out.job.template allocate<std::int64_t>(beats.size());
                    std::copy(beats.begin(), beats.end(), out.result.positions.get());
                    out.result.count = beats.size();
                }
            });
        result.tempoBpm = summary.tempoBpm;
        result.secondaryTempoBpm = summary.secondaryTempoBpm;
        result.confidence = summary.confidence;
        if (!std::isfinite(result.tempoBpm) || !std::isfinite(result.secondaryTempoBpm) ||
            !std::isfinite(result.confidence))
            offlineFail(OfflineStatus::NumericalFailure);
        job.checkpoint(OfflinePhase::Analyze, static_cast<std::int64_t>(count),
                       static_cast<std::int64_t>(count));
        return result;
    }
};

} // namespace dspark::detail
#endif
