// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpscQueue.h
 * @brief Lock-free single-producer, single-consumer (SPSC) bounded queue.
 *
 * Designed for passing parameter snapshots from a control thread (GUI, automation)
 * to the audio thread without locks or allocations at runtime. Features a local 
 * index caching optimization to aggressively minimize CPU cache-line bouncing.
 *
 * Dependencies: C++20 standard library only (<array>, <atomic>, <cstddef>, <new>).
 *
 * @tparam T        Element type. Must be trivially copyable for lock-free safety.
 * @tparam Capacity Maximum number of slots. Must be a power of two for efficient
 *                  modular arithmetic. Note: Actual usable capacity is (Capacity - 1)
 *                  to safely distinguish between full and empty states.
 *
 * @code
 *   // GUI thread (producer):
 *   dspark::SpscQueue<Params, 32> queue; // Can hold up to 31 items
 *   queue.push(newParams);
 *
 *   // Audio thread (consumer):
 *   Params p;
 *   while (queue.pop(p))
 *       applyParams(p);
 * @endcode
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <new> // For hardware_destructive_interference_size

namespace dspark {

#ifdef __cpp_lib_hardware_interference_size
    constexpr std::size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
    constexpr std::size_t kCacheLineSize = 64; // Safe fallback for older compiler implementations
#endif

template <typename T, std::size_t Capacity = 32>
class SpscQueue
{
    static_assert(Capacity > 0 && (Capacity & (Capacity - 1)) == 0,
                  "SpscQueue: Capacity must be a power of two.");
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscQueue: Element type must be trivially copyable for lock-free safety.");

public:
    /** @brief Constructs an empty SPSC queue. */
    SpscQueue() = default;

    SpscQueue(const SpscQueue&)            = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    /**
     * @brief Pushes an element into the queue (producer side).
     *
     * @param item The element to enqueue.
     * @retval true if the element was successfully enqueued.
     * @retval false if the queue is full (element dropped).
     *
     * @note Lock-free, wait-free, allocation-free. Safe to call from any single producer thread.
     */
    bool push(const T& item) noexcept
    {
        const auto write = writePos_.load(std::memory_order_relaxed);
        const auto next  = (write + 1) & kMask;

        // Check against cached read position first to avoid cross-core atomic load
        if (next == cachedReadPos_)
        {
            cachedReadPos_ = readPos_.load(std::memory_order_acquire);
            if (next == cachedReadPos_)
                return false; // Truly full
        }

        buffer_[write] = item;
        writePos_.store(next, std::memory_order_release);
        return true;
    }

    /**
     * @brief Pops an element from the queue (consumer side).
     *
     * @param[out] item Receives the dequeued element if the queue is not empty.
     * @retval true if an element was dequeued.
     * @retval false if the queue was empty.
     *
     * @note Lock-free, wait-free, allocation-free. Safe to call from any single consumer thread.
     */
    bool pop(T& item) noexcept
    {
        const auto read = readPos_.load(std::memory_order_relaxed);

        // Check against cached write position first to avoid cross-core atomic load
        if (read == cachedWritePos_)
        {
            cachedWritePos_ = writePos_.load(std::memory_order_acquire);
            if (read == cachedWritePos_)
                return false; // Truly empty
        }

        item = buffer_[read];
        readPos_.store((read + 1) & kMask, std::memory_order_release);
        return true;
    }

    /**
     * @brief Returns the approximate number of elements in the queue.
     *
     * @note This is a snapshot and may be stale by the time you act on it.
     *       Use it only for diagnostics, never for control flow.
     * @return Number of elements currently enqueued.
     */
    [[nodiscard]] std::size_t sizeApprox() const noexcept
    {
        const auto w = writePos_.load(std::memory_order_relaxed);
        const auto r = readPos_.load(std::memory_order_relaxed);
        return (w - r) & kMask;
    }

    /** @brief Returns true if the queue appears empty (approximate). */
    [[nodiscard]] bool empty() const noexcept { return sizeApprox() == 0; }

private:
    static constexpr std::size_t kMask = Capacity - 1;

    std::array<T, Capacity> buffer_ {};

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4324) // structure was padded due to alignment specifier
#endif

    // ========================================================================
    // Producer Cache Line
    // ========================================================================
    alignas(kCacheLineSize) std::atomic<std::size_t> writePos_ {0};
    std::size_t cachedReadPos_ {0}; // Only read/written by the producer

    // ========================================================================
    // Consumer Cache Line
    // ========================================================================
    alignas(kCacheLineSize) std::atomic<std::size_t> readPos_ {0};
    std::size_t cachedWritePos_ {0}; // Only read/written by the consumer

#ifdef _MSC_VER
#pragma warning(pop)
#endif
};

} // namespace dspark