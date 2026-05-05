// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file SpinLock.h
 * @brief Lightweight spin lock for real-time audio thread synchronisation.
 *
 * A busy-wait mutex built on std::atomic_flag. Designed for protecting very
 * short critical sections (a few assignments) where sleeping would introduce
 * unacceptable latency — typically parameter hand-off between a GUI thread
 * and the audio thread.
 *
 * Dependencies: C++20 standard library only (<atomic>).
 *
 * @warning Do NOT use for long critical sections or high-contention scenarios.
 *          In those cases, prefer std::mutex or a lock-free data structure.
 *
 * @code
 *   dspark::SpinLock lock;
 *
 *   // From the GUI thread:
 *   {
 *       dspark::SpinLock::ScopedLock guard(lock);
 *       sharedParams = newParams;
 *   }
 *
 *   // From the audio thread (non-blocking attempt):
 *   {
 *       dspark::SpinLock::ScopedTryLock guard(lock);
 *       if (guard.isLocked())
 *           localCopy = sharedParams;
 *   }
 * @endcode
 */

#include <atomic>

#if defined(_MSC_VER) || defined(__SSE2__)
  #include <immintrin.h>
  #define DSPARK_SPIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(__ARM_NEON)
  #define DSPARK_SPIN_PAUSE() __asm__ __volatile__("yield")
#else
  #define DSPARK_SPIN_PAUSE() ((void)0)
#endif

namespace dspark {

/**
 * @class SpinLock
 * @brief A minimal, real-time safe spin lock utilizing C++20 TTAS optimization.
 *
 * Fulfills the C++ BasicLockable requirements.
 * - `lock()` busy-waits using a Test-and-Test-and-Set (TTAS) pattern to avoid cache thrashing.
 * - `tryLock()` attempts a single acquire without waiting — mandatory for the audio thread.
 * - `unlock()` releases the lock safely.
 */
class SpinLock
{
public:
    /** @brief Constructs an unlocked SpinLock. */
    SpinLock() noexcept = default;

    SpinLock(const SpinLock&)            = delete;
    SpinLock& operator=(const SpinLock&) = delete;

    /**
     * @brief Acquires the lock, spinning until successful.
     * 
     * Employs a C++20 TTAS (Test-and-Test-and-Set) loop. It reads the flag 
     * with memory_order_relaxed to keep the cache line in a shared state, 
     * avoiding bus traffic until the lock appears free.
     * 
     * @note Call only when the lock holder is guaranteed to release quickly.
     */
    void lock() noexcept
    {
        for (;;) 
        {
            // Attempt to grab the lock
            if (!flag_.test_and_set(std::memory_order_acquire)) {
                return;
            }
            
            // Spin on a relaxed read to prevent cache line bouncing (C++20 feature)
            while (flag_.test(std::memory_order_relaxed)) {
                DSPARK_SPIN_PAUSE(); // Hint CPU to yield resources during spin
            }
        }
    }

    /**
     * @brief Attempts to acquire the lock without waiting.
     * @return true if the lock was successfully acquired, false if it was already held.
     */
    [[nodiscard]] bool tryLock() noexcept
    {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    /**
     * @brief Releases the lock, restoring it to the clear state.
     */
    void unlock() noexcept
    {
        flag_.clear(std::memory_order_release);
    }

    // ========================================================================

    /**
     * @class ScopedLock
     * @brief RAII wrapper that acquires the lock on construction and releases on destruction.
     */
    class ScopedLock
    {
    public:
        /**
         * @brief Constructs the guard and blocks until the lock is acquired.
         * @param spinLock The SpinLock to manage.
         */
        explicit ScopedLock(SpinLock& spinLock) noexcept : lock_(spinLock) { lock_.lock(); }
        
        /** @brief Destructs the guard and releases the lock. */
        ~ScopedLock() noexcept { lock_.unlock(); }

        ScopedLock(const ScopedLock&)            = delete;
        ScopedLock& operator=(const ScopedLock&) = delete;

    private:
        SpinLock& lock_;
    };

    /**
     * @class ScopedTryLock
     * @brief RAII wrapper that *tries* to acquire the lock without blocking.
     *
     * Check `isLocked()` before accessing the protected resource. Essential for
     * reading shared data safely within the real-time audio thread.
     */
    class ScopedTryLock
    {
    public:
        /**
         * @brief Constructs the guard and attempts a single lock acquisition.
         * @param spinLock The SpinLock to attempt to manage.
         */
        explicit ScopedTryLock(SpinLock& spinLock) noexcept
            : lock_(spinLock), acquired_(spinLock.tryLock()) {}

        /** @brief Destructs the guard and releases the lock ONLY if it was successfully acquired. */
        ~ScopedTryLock() noexcept { if (acquired_) lock_.unlock(); }

        /**
         * @brief Queries whether the lock acquisition was successful.
         * @return true if the lock is held by this guard, false otherwise.
         */
        [[nodiscard]] bool isLocked() const noexcept { return acquired_; }

        ScopedTryLock(const ScopedTryLock&)            = delete;
        ScopedTryLock& operator=(const ScopedTryLock&) = delete;

    private:
        SpinLock& lock_;
        bool      acquired_;
    };

private:
    // C++20 default initialization correctly sets the atomic_flag to the clear state.
    // ATOMIC_FLAG_INIT is deprecated in C++20 and removed in C++26.
    std::atomic_flag flag_{};
};

} // namespace dspark