// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file ProcessorTraits.h
 * @brief C++20 concepts defining the strictly real-time DSP processor contract.
 *
 * Provides compile-time concepts that formalise what it means to be a
 * DSP processor in this framework. Any class that satisfies these concepts
 * can be used with ProcessorChain and other generic utilities.
 *
 * Architecture Principles Enforced:
 * - No virtual dispatch (resolved at compile time).
 * - Real-time safety: All hot-path methods MUST be natively `noexcept`.
 * - Strict type matching (`std::same_as`) to prevent implicit casting overhead.
 * - In-place processing: the block call must write through the buffer it is
 *   handed, so a type that also accepts a const view is excluded.
 * - Generators must support block processing as well as per-sample pulls.
 *
 * | Concept              | Required methods                                          |
 * |----------------------|-----------------------------------------------------------|
 * | `AudioProcessor`     | `prepare(AudioSpec)`, `processBlock(View) noexcept`, `reset() noexcept` |
 * | `SampleProcessor`    | Above + `processSample(T, int) noexcept -> T`             |
 * | `GeneratorProcessor` | `prepare(AudioSpec)`, `generateBlock(View) noexcept`, `reset() noexcept`, `getSample() noexcept -> T` |
 *
 * Scope note: the contract targets in-place single-buffer inserts and
 * self-contained generators. Multi-buffer or per-call-parameterised
 * utilities (Crossfade, CrossoverFilter, MidSide, AutoGain, the Delay
 * building block, Phasor) intentionally do not model these concepts;
 * wrap them in a small insert class if you need them inside a
 * ProcessorChain.
 *
 * Read-only analysers are outside the contract for the same reason and are
 * meant to be: BeatTracker, ChordDetector, EnvelopeFollower, LoudnessMeter,
 * OnsetDetector, PhaseCorrelation and PitchFollower take an
 * `AudioBufferView<const T>` and return measurements, so they have nothing
 * to contribute to a chain that expects each stage to hand the next one its
 * output. Run them beside a chain, on the same buffer, not inside it.
 *
 * Dependencies: AudioSpec.h, AudioBuffer.h.
 */

#include "AudioSpec.h"
#include "AudioBuffer.h"

#include <concepts>

namespace dspark {

/**
 * @concept AudioProcessor
 * @brief A type that can prepare, process audio blocks, and reset state.
 *
 * This is the foundational concept for effects and filters.
 * Processing and resetting must be guaranteed exception-free (`noexcept`)
 * to be safely executed within the real-time audio thread.
 *
 * The processor must write its output back into the buffer it is given: an
 * insert replaces what it reads. A read-only analyser is therefore NOT an
 * AudioProcessor, however real-time safe it is, and the last clause is what
 * says so. It rejects any type whose `processBlock` also accepts an
 * `AudioBufferView<const T>`, because a type that can take a const view
 * cannot be writing through the view it was handed. That test is about the
 * constness of the SAMPLES, not of the handle: a processor declared
 * `processBlock(const AudioBufferView<T>&)` is in-place and still satisfies
 * this concept.
 *
 * @tparam P Processor type.
 * @tparam T Sample type (float or double).
 */
template <typename P, typename T>
concept AudioProcessor = requires(P p, const AudioSpec& spec, AudioBufferView<T> buf) {
    { p.prepare(spec) };                  // Can throw/allocate (offline phase)
    { p.processBlock(buf) } noexcept;     // Real-time hot path
    { p.reset() } noexcept;               // Must be safe to call from audio thread
} && !requires(P p, AudioBufferView<const T> c) {
    p.processBlock(c);                    // In-place: a const view must NOT bind
};

/**
 * @concept SampleProcessor
 * @brief An AudioProcessor that additionally supports scalar per-sample processing.
 *
 * Useful for feedback loops or non-linear structures where block processing
 * needs scalar fallbacks. Requires strict type return `std::same_as<T>` to 
 * avoid implicit type demotion/promotion cycles (e.g., double <-> float).
 *
 * @tparam P Processor type.
 * @tparam T Sample type.
 */
template <typename P, typename T>
concept SampleProcessor = AudioProcessor<P, T> &&
    requires(P p, T sample, int channel) {
        { p.processSample(sample, channel) } noexcept -> std::same_as<T>;
    };

/**
 * @concept GeneratorProcessor
 * @brief A real-time safe source processor (oscillators, noise, LFOs).
 *
 * Generators produce output to a buffer without requiring input.
 * Must support both scalar `getSample()` and SIMD-friendly `generateBlock()`.
 *
 * A generator writes into the buffer it is given, so the same const-view
 * exclusion the insert contract carries applies here to `generateBlock`.
 *
 * @tparam P Processor type.
 * @tparam T Sample type.
 */
template <typename P, typename T>
concept GeneratorProcessor = requires(P p, const AudioSpec& spec, AudioBufferView<T> buf) {
    { p.prepare(spec) };
    { p.reset() } noexcept;
    { p.generateBlock(buf) } noexcept;                    // Mandatory for SIMD/Cache optimization
    { p.getSample() } noexcept -> std::same_as<T>;        // Strict type matching
} && !requires(P p, AudioBufferView<const T> c) {
    p.generateBlock(c);                                   // Writes: a const view must NOT bind
};

} // namespace dspark