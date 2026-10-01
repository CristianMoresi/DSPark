// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file OfflineProcessing.h
 * @brief Complete-source worker jobs, bounded audio transport and cancellation.
 *
 * OFFLINE ONLY. Sources and sinks are synchronous worker-thread interfaces, not
 * audio callbacks. They allocate/decode/stage as their host requires. A source
 * must remain immutable throughout a job, including repeated reads. A sink must
 * keep writes provisional until commit(), and discard them on abort().
 *
 * Threading: one worker owns a job and its source/sink. Independent jobs may run
 * concurrently with independent sources/sinks. Only the optional atomic cancel
 * flag may be changed concurrently. Progress callbacks run on that worker; their
 * context and all options remain alive and unmodified until the job returns.
 * Worker jobs require C++ exceptions to translate allocation/host failures into
 * statuses. DSPARK_HAS_OFFLINE is 1 when available, otherwise 0. Define
 * DSPARK_NO_OFFLINE to omit these APIs. Real-time DSP has no such requirement.
 * Dependencies: AudioBuffer.h, DspMath.h and the C++20 standard library.
 */

#if !defined(DSPARK_NO_OFFLINE) && (defined(__cpp_exceptions) || defined(_CPPUNWIND))
#define DSPARK_HAS_OFFLINE 1
#else
#define DSPARK_HAS_OFFLINE 0
#endif

#if DSPARK_HAS_OFFLINE

#include "AudioBuffer.h"
#include "DspMath.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>

namespace dspark
{

/** @brief Explicit completion or no-publication outcome of an offline job. */
enum class OfflineStatus : std::uint8_t
{
    Success,
    NoChange,
    EmptyInput,
    InvalidInput,
    UnsupportedLayout,
    NonFiniteInput,
    SourceMismatch,
    SourceReadFailed,
    SinkFailed,
    CallbackFailed,
    Cancelled,
    MemoryLimit,
    AllocationFailure,
    NumericalFailure,
    TargetUnreachable ///< Exact protected PCM prevents the requested processing target.
};

/** @brief True for successful processing, including an explicit identity result. */
[[nodiscard]] inline bool offlineSucceeded(OfflineStatus status) noexcept
{
    return status == OfflineStatus::Success || status == OfflineStatus::NoChange;
}

/** @brief Immutable source format and host-provided content/timeline identity. */
struct OfflineAudioSpec
{
    double sampleRate = 48000;
    int channels = 0;
    std::int64_t frames = 0;
    std::uint64_t sourceId = 0;
    std::uint64_t revision = 0;
    std::int64_t timelineOrigin = 0; ///< Metadata; read offsets are relative to frame zero.

    /** @brief Checks transport bounds, not the contents or identity of the audio. */
    [[nodiscard]] bool isValid() const noexcept
    {
        return std::isfinite(sampleRate) && sampleRate >= 1 &&
               sampleRate <= static_cast<double>(std::numeric_limits<int>::max()) && channels > 0 &&
               channels <= 16 && frames >= 0 && timelineOrigin >= 0 &&
               frames <= std::numeric_limits<std::int64_t>::max() - timelineOrigin;
    }
    bool operator==(const OfflineAudioSpec &) const noexcept = default;
};

/** @brief Source-frame interval [begin,end), relative to the start of the source. */
struct OfflineRegion
{
    std::int64_t begin = 0;
    std::int64_t end = 0;
    bool operator==(const OfflineRegion &) const noexcept = default;
};

/** @brief Current kind of work; completed/total restart for each pass, and a phase may recur. */
enum class OfflinePhase : std::uint8_t
{
    Analyze,
    Plan,
    Verify,
    Render
};

/** @brief Worker progress in source frames, or control points in the Plan pass. */
struct OfflineProgress
{
    OfflinePhase phase = OfflinePhase::Analyze;
    std::int64_t completed = 0;
    std::int64_t total = 0;
};

/**
 * @brief Resource and cooperative-cancellation controls for one worker operation.
 *
 * memoryBudgetBytes bounds newly requested dynamic payload, including returned
 * analysis/plan storage. It excludes caller-owned input, preexisting output/plans,
 * allocator bookkeeping, stack state and allocations inside source/sink callbacks.
 * Convenience owning renders also account for their provisional output. No hidden
 * thread pool, disk I/O or global cache is created.
 */
struct OfflineJobOptions
{
    int blockFrames = 4096;
    std::size_t memoryBudgetBytes = 256u * 1024u * 1024u;
    const std::atomic<bool> *cancel = nullptr;
    void *progressContext = nullptr;
    bool (*progress)(void *, OfflineProgress) = nullptr; ///< Returning false cancels.
};

/** @brief Measured-render control adjustments; no clipping or master gain trim. */
struct OfflineGainRenderInfo
{
    bool bandlimited = false; ///< Product evaluated before optional exact-PCM constraints.
    bool targetFeasible = true; ///< False if no admissible processing delta meets the peak target.
    bool representabilityLimited = false; ///< Extra gain reduced to avoid floating-point overflow.
    bool boundaryCalibrated = false; ///< Exterior controls adjusted for boundary-event peaks.
    double deltaScale = 1; ///< Applied to the processing delta, leaving protected PCM unchanged.
    double boundaryGain = 1; ///< Affine reference, before deltaScale; mean of endpoint controls.
    double leftBoundaryGain = 1; ///< Independent constant continuation before frame zero.
    double rightBoundaryGain = 1; ///< Independent constant continuation after the final frame.
    double minimumControlGain = 1; ///< Calibrated control range, not the output/input sample ratio.
    double maximumControlGain = 1; ///< Includes any representability cap on the processing delta.
};

/**
 * @brief Rewindable, complete-file source with int64 positions and bounded blocks.
 *
 * Threading: worker-owned, never concurrently read or edited. read() must fill
 * every requested channel/frame or return false. The source owns no destination
 * block and must not retain its pointers. Repeated reads must give the same PCM.
 * Change revision whenever content or its interpretation changes between jobs.
 */
template <FloatType T> class OfflineAudioSource
{
  public:
    virtual ~OfflineAudioSource() = default;
    /** @brief Returns format and provenance by value. */
    [[nodiscard]] virtual OfflineAudioSpec getSpec() const noexcept = 0;
    /** @brief Copies a bounded source-relative range into the supplied block. */
    virtual bool read(std::int64_t first, AudioBufferView<T> destination) = 0;
};

/**
 * @brief Borrowed planar-buffer adapter; does not allocate, copy or own input.
 *
 * Threading: the buffer must stay alive and immutable until the worker job ends.
 * A view's channel capacity is preserved, including unsupported layouts that a
 * particular algorithm must reject rather than silently truncate.
 */
template <FloatType T, int MaxChannels = 16>
class OfflineBufferSource final : public OfflineAudioSource<T>
{
  public:
    /** @brief Binds an existing view and optional host content/revision keys. */
    OfflineBufferSource(AudioBufferView<const T, MaxChannels> audio, double sampleRate,
                        std::uint64_t sourceId = 0, std::uint64_t revision = 0,
                        std::int64_t timelineOrigin = 0) noexcept
        : audio_(audio), spec_{sampleRate, audio.getNumChannels(), audio.getNumSamples(), sourceId,
                               revision,   timelineOrigin}
    {
    }

    [[nodiscard]] OfflineAudioSpec getSpec() const noexcept override
    {
        return spec_;
    }
    bool read(std::int64_t first, AudioBufferView<T> destination) override
    {
        const int count = destination.getNumSamples();
        if (first < 0 || count < 0 || count > spec_.frames || first > spec_.frames - count ||
            destination.getNumChannels() != spec_.channels)
            return false;
        for (int ch = 0; ch < spec_.channels; ++ch)
        {
            if (count == 0)
                continue;
            if (!audio_.getChannel(ch) || !destination.getChannel(ch))
                return false;
            std::memmove(destination.getChannel(ch), audio_.getChannel(ch) + first,
                         static_cast<std::size_t>(count) * sizeof(T));
        }
        return true;
    }

  private:
    AudioBufferView<const T, MaxChannels> audio_;
    OfflineAudioSpec spec_;
};

/**
 * @brief Transactional worker sink for arbitrarily long offline output.
 *
 * begin/write/commit return false on failure. abort() must discard all provisional
 * output, including after a failed begin or commit, and must not throw. commit()
 * returning true is the only publication point. A file sink may stage a temporary
 * file; DSPark does not prescribe storage or perform filesystem operations.
 */
template <FloatType T> class OfflineAudioSink
{
  public:
    virtual ~OfflineAudioSink() = default;
    virtual bool begin(const OfflineAudioSpec &spec) = 0;
    virtual bool write(std::int64_t first, AudioBufferView<const T> block) = 0;
    virtual bool commit() = 0;
    virtual void abort() noexcept = 0;
};

/**
 * @brief Noncryptographic PCM fingerprint, stable across block divisions.
 *
 * Used with format/revision to detect accidental stale analysis. This is not a
 * collision-proof security identity or permission to mutate a source during a job.
 */
struct OfflineFingerprint
{
    std::uint64_t first = 14695981039346656037ull;
    std::uint64_t second = 0x9e3779b97f4a7c15ull;
    bool operator==(const OfflineFingerprint &) const noexcept = default;
};

namespace detail
{

struct OfflineFailure
{
    OfflineStatus status;
};

[[noreturn]] inline void offlineFail(OfflineStatus status)
{
    throw OfflineFailure{status};
}

// Called only inside a catch block. Transport and callback failures are translated
// at their boundary; an otherwise unexpected internal error is numerical failure.
[[nodiscard]] inline OfflineStatus offlineExceptionStatus() noexcept
{
    try
    {
        throw;
    }
    catch (const OfflineFailure &failure)
    {
        return failure.status;
    }
    catch (const std::bad_alloc &)
    {
        return OfflineStatus::AllocationFailure;
    }
    catch (...)
    {
        return OfflineStatus::NumericalFailure;
    }
}

[[nodiscard]] inline std::size_t offlineBytes(std::uint64_t count, std::size_t size)
{
    if (count > std::numeric_limits<std::size_t>::max() / size)
        offlineFail(OfflineStatus::MemoryLimit);
    return static_cast<std::size_t>(count) * size;
}

class OfflineSession final
{
  public:
    explicit OfflineSession(const OfflineJobOptions &options) : options_(options)
    {
        if (options.blockFrames < 1)
            offlineFail(OfflineStatus::InvalidInput);
    }
    void charge(std::size_t bytes)
    {
        if (bytes > options_.memoryBudgetBytes - bytes_)
            offlineFail(OfflineStatus::MemoryLimit);
        bytes_ += bytes;
    }
    template <typename U> [[nodiscard]] std::unique_ptr<U[]> allocate(std::uint64_t count)
    {
        charge(offlineBytes(count, sizeof(U)));
        return std::make_unique<U[]>(static_cast<std::size_t>(count));
    }
    [[nodiscard]] std::size_t bytes() const noexcept
    {
        return bytes_;
    }
    void checkpoint(OfflinePhase phase, std::int64_t completed, std::int64_t total) const
    {
        if (options_.cancel && options_.cancel->load(std::memory_order_relaxed))
            offlineFail(OfflineStatus::Cancelled);
        if (options_.progress)
        {
            bool proceed = false;
            try
            {
                proceed = options_.progress(options_.progressContext, {phase, completed, total});
            }
            catch (...)
            {
                offlineFail(OfflineStatus::CallbackFailed);
            }
            if (!proceed)
                offlineFail(OfflineStatus::Cancelled);
        }
    }

  private:
    const OfflineJobOptions &options_;
    std::size_t bytes_ = 0;
};

[[nodiscard]] inline OfflineJobOptions offlineRemaining(const OfflineJobOptions &job,
                                                        std::size_t used)
{
    if (used > job.memoryBudgetBytes)
        offlineFail(OfflineStatus::MemoryLimit);
    auto out = job;
    out.memoryBudgetBytes -= used;
    return out;
}

inline void offlineValidateSpec(const OfflineAudioSpec &spec, int maximumChannels = 16)
{
    if (!spec.isValid())
        offlineFail(OfflineStatus::InvalidInput);
    if (spec.channels > maximumChannels)
        offlineFail(OfflineStatus::UnsupportedLayout);
    if (spec.frames == 0)
        offlineFail(OfflineStatus::EmptyInput);
}

template <FloatType T> class OfflineBlock final
{
  public:
    OfflineBlock(OfflineSession &session, const OfflineAudioSpec &spec, int blockFrames)
        : frames_(static_cast<int>(std::min<std::int64_t>(spec.frames, blockFrames))),
          channels_(spec.channels),
          data_(session.allocate<T>(static_cast<std::uint64_t>(frames_) * channels_))
    {
    }
    [[nodiscard]] AudioBufferView<T> view(int count) noexcept
    {
        std::array<T *, 16> pointers{};
        for (int c = 0; c < channels_; ++c)
            pointers[static_cast<std::size_t>(c)] =
                data_.get() + static_cast<std::size_t>(c) * frames_;
        return {pointers.data(), channels_, count};
    }
    [[nodiscard]] int frames() const noexcept
    {
        return frames_;
    }

  private:
    int frames_, channels_;
    std::unique_ptr<T[]> data_;
};

template <FloatType T>
inline void offlineRead(OfflineAudioSource<T> &source, const OfflineAudioSpec &expected,
                        std::int64_t first, AudioBufferView<T> block)
{
    if (source.getSpec() != expected)
        offlineFail(OfflineStatus::SourceMismatch);
    bool ok = false;
    try
    {
        ok = source.read(first, block);
    }
    catch (...)
    {
        offlineFail(OfflineStatus::SourceReadFailed);
    }
    if (!ok)
        offlineFail(OfflineStatus::SourceReadFailed);
    if (source.getSpec() != expected)
        offlineFail(OfflineStatus::SourceMismatch);
}

template <FloatType T> inline void offlineHash(OfflineFingerprint &state, T sample) noexcept
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "Offline PCM supports float and double");
    const std::uint64_t word = [&] {
        if constexpr (sizeof(T) == 4)
            return static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(sample));
        else
            return std::bit_cast<std::uint64_t>(sample);
    }();
    state.first = (state.first ^ word) * 1099511628211ull;
    state.second =
        std::rotl(state.second ^ word, 27) * 0x3c79ac492ba7b653ull + 0x1c69b3f74ac4ae35ull;
}

} // namespace detail
} // namespace dspark

#endif // DSPARK_HAS_OFFLINE
