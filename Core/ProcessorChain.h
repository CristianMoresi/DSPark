// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file ProcessorChain.h
 * @brief Compile-time processor chain for composing DSP effects.
 *
 * Chains multiple processors together so they are prepared, processed,
 * and reset as a single unit. Uses `std::tuple` internally — all dispatch
 * is resolved at compile time with zero runtime overhead.
 *
 * Each processor in the chain should satisfy the framework's AudioProcessor concept.
 *
 * Dependencies: ProcessorTraits.h, AudioSpec.h, AudioBuffer.h.
 */

#include "ProcessorTraits.h"

#include <cstddef>
#include <tuple>
#include <utility>
#include <array>
#include <atomic>

namespace dspark {

/**
 * @class ProcessorChain
 * @brief Compile-time chain of audio processors.
 *
 * Processors are stored in a `std::tuple` and invoked in order.
 * Access individual processors via `get<Index>()` to configure parameters.
 *
 * @tparam T          Sample type (float or double).
 * @tparam Processors Processor types. Must be greater than 0.
 */
template <typename T, typename... Processors>
class ProcessorChain
{
    static_assert(sizeof...(Processors) > 0, "ProcessorChain requires at least one processor.");

public:
    /**
     * @brief Prepares all processors in order.
     * @param spec Audio specification.
     */
    void prepare(const AudioSpec& spec)
    {
        std::apply([&spec](auto&... procs) {
            (procs.prepare(spec), ...);
        }, processors_);
    }

    /**
     * @brief Processes a buffer through all non-bypassed processors in order.
     * @param buffer Audio buffer (modified in-place by each processor).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        processBlockImpl(buffer, std::index_sequence_for<Processors...>{});
    }

    /**
     * @brief Resets the internal state of all processors (e.g., clearing delay lines).
     */
    void reset() noexcept
    {
        std::apply([](auto&... procs) {
            (procs.reset(), ...);
        }, processors_);
    }

    /**
     * @brief Accesses the processor at the given index.
     *
     * @tparam Index Zero-based index into the chain.
     * @return Reference to the processor.
     */
    template <std::size_t Index>
    [[nodiscard]] auto& get() noexcept
    {
        static_assert(Index < sizeof...(Processors), "Processor index out of bounds.");
        return std::get<Index>(processors_);
    }

    /**
     * @brief Const access to the processor at the given index.
     *
     * @tparam Index Zero-based index into the chain.
     * @return Const reference to the processor.
     */
    template <std::size_t Index>
    [[nodiscard]] const auto& get() const noexcept
    {
        static_assert(Index < sizeof...(Processors), "Processor index out of bounds.");
        return std::get<Index>(processors_);
    }

    /**
     * @brief Returns the number of processors in the chain.
     */
    [[nodiscard]] static constexpr std::size_t size() noexcept
    {
        return sizeof...(Processors);
    }

    /**
     * @brief Returns the total latency in samples across all processors.
     *
     * Evaluated at compile-time via requires expressions. Only sums from
     * processors that provide a `getLatency()` method.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return getLatencyImpl(std::index_sequence_for<Processors...>{});
    }

    // -- Bypass control ------------------------------------------------------

    /**
     * @brief Thread-safe toggle to bypass or enable a processor.
     *
     * @warning This is a hard bypass. Toggling this during live playback will cause
     * audio clicks/pops and phase discontinuities if the processor introduces latency.
     * For real-time clickless bypass, prefer using dry/wet controls within the 
     * specific processor itself.
     *
     * @tparam Index Zero-based index into the chain.
     * @param bypassed True to bypass, false to enable.
     */
    template <std::size_t Index>
    void setBypassed(bool bypassed) noexcept
    {
        static_assert(Index < sizeof...(Processors), "Index out of range");
        bypassed_[Index].store(bypassed, std::memory_order_relaxed);
    }

    /**
     * @brief Thread-safe read of the bypass state of a processor.
     */
    template <std::size_t Index>
    [[nodiscard]] bool isBypassed() const noexcept
    {
        static_assert(Index < sizeof...(Processors), "Index out of range");
        return bypassed_[Index].load(std::memory_order_relaxed);
    }

private:
    template <std::size_t... Is>
    [[nodiscard]] int getLatencyImpl(std::index_sequence<Is...>) const noexcept
    {
        return (getProcessorLatency<Is>() + ...);
    }

    template <std::size_t I>
    [[nodiscard]] int getProcessorLatency() const noexcept
    {
        if constexpr (requires { std::get<I>(processors_).getLatency(); })
            return std::get<I>(processors_).getLatency();
        else
            return 0;
    }

    template <std::size_t... Is>
    void processBlockImpl(AudioBufferView<T> buffer, std::index_sequence<Is...>) noexcept
    {
        // Uses memory_order_relaxed as we only care about the state, no memory synchronization needed
        ((!bypassed_[Is].load(std::memory_order_relaxed) ? std::get<Is>(processors_).processBlock(buffer) : (void)0), ...);
    }

    std::tuple<Processors...> processors_;
    std::array<std::atomic<bool>, sizeof...(Processors)> bypassed_ {};
};

} // namespace dspark