// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file Reverb.h
 * @brief Convolution reverb with one-line IR loading and progressive API.
 *
 * Wraps the Convolver engine into a complete reverb effect with dry/wet mix,
 * pre-delay, and automatic IR management. Supports loading impulse responses
 * from WAV files or from raw sample data. The dry path is delay-compensated
 * against the convolution engine's latency, so dry and wet stay sample-aligned
 * at any mix setting (getLatency() reports the shared latency to the host).
 *
 * IR shaping (setDecayScale / setStretch) reshapes the loaded impulse
 * response without touching the stored original, so the controls are
 * always relative to the file as loaded:
 *
 * - **Decay scale** multiplies the IR's own T60 (estimated from its
 *   Schroeder decay curve). Values below 1 also trim the now-silent tail,
 *   which directly reduces convolution CPU cost.
 * - **Stretch** resamples the IR (tape-speed style): above 1 the space
 *   gets larger and darker, below 1 smaller and brighter.
 *
 * Three levels of API complexity:
 *
 * - **Level 1 (simple):** `reverb.loadIR("hall.wav"); reverb.setMix(0.3f);`
 * - **Level 2 (intermediate):** Pre-delay, IR decay scale / stretch.
 * - **Level 3 (expert):** Direct access to Convolver and DryWetMixer internals.
 *
 * Threading: prepare() belongs to the setup thread (allocates; never call it
 * concurrently with processing). processBlock() and reset() belong to the
 * audio thread. loadIR(), setDecayScale(), setStretch() and setState()
 * rebuild the convolver bank (they allocate) on the calling GUI/setup thread
 * and publish it atomically: safe while audio runs, one writer at a time.
 * setMix()/setPreDelay() are lock-free atomic publications, safe from any
 * thread. Non-finite setter arguments are ignored. Loading an IR changes
 * getLatency(): hosts must be notified.
 *
 * File loading (loadIR from a path) is excluded when DSPARK_NO_FILE_IO is
 * defined; the raw-data overload keeps working on embedded targets.
 *
 * Dependencies: Convolver.h, DryWetMixer.h, RingBuffer.h, AudioSpec.h,
 *               AudioBuffer.h, DspMath.h, Resampler.h, StateBlob.h,
 *               WavFile.h (only without DSPARK_NO_FILE_IO).
 *
 * @code
 *   dspark::Reverb<float> reverb;
 *   reverb.prepare(spec);
 *   reverb.loadIR("hall.wav");   // One line, done
 *   reverb.setMix(0.3f);         // 30% wet
 *   reverb.processBlock(buffer);
 *
 *   // With pre-delay:
 *   reverb.setPreDelay(20.0f);   // 20 ms
 * @endcode
 */

#include "../Core/Convolver.h"
#include "../Core/DryWetMixer.h"
#include "../Core/RingBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/DspMath.h"
#include "../Core/Resampler.h"
#include "../Core/StateBlob.h"
#ifndef DSPARK_NO_FILE_IO
#include "../IO/WavFile.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace dspark {

namespace detail {

/** Internal no-op instrumentation for the fixed Reverb publication protocol. */
struct ReverbPublisherNoopHooks
{
    enum class ControlPoint
    {
        candidateReady,
        beforeFirstScan,
        afterSlotSelection,
        afterPublishedStore,
        beforePendingExchange,
        afterPendingExchange,
        beforeOldRetirement,
        beforeOldReclaim,
        beforeCommit,
        afterCommit,
        noCapacity
    };

    static constexpr void control(ControlPoint) noexcept {}
    static constexpr void audioExchange() noexcept {}
    static constexpr void audioAfterExchange(bool) noexcept {}
    static constexpr void audioStateStore() noexcept {}
    static constexpr void audioValidation(bool) noexcept {}
    static constexpr void controlValidation(bool) noexcept {}
    static constexpr void controlScan() noexcept {}
};

/**
 * Fixed-capacity ownership handoff used by Reverb.
 *
 * The audio owner performs one pending-token exchange and never constructs,
 * destroys, scans or reference-counts a bank. The serialized control owner
 * alone constructs and reclaims the four slot-owned banks.
 */
template <typename Bank,
          typename Hooks = ReverbPublisherNoopHooks,
          std::uint32_t GenerationMax = 134217727u>
class ReverbBankPublisher
{
public:
    static_assert(std::atomic<std::uint32_t>::is_always_lock_free,
                  "Reverb publication requires lock-free 32-bit atomics");
    static_assert(GenerationMax >= 1u && GenerationMax <= 134217727u,
                  "Generation must fit the 27-bit token encoding");

    enum class Phase : std::uint32_t
    {
        free = 0u,
        building = 1u,
        published = 2u,
        active = 3u,
        retired = 4u,
        exhausted = 5u
    };

    enum class PublishResult
    {
        published,
        noCapacity,
        invariantViolation
    };

    ReverbBankPublisher() noexcept = default;
    ~ReverbBankPublisher() noexcept { shutdown(); }

    ReverbBankPublisher(const ReverbBankPublisher&) = delete;
    ReverbBankPublisher& operator=(const ReverbBankPublisher&) = delete;
    ReverbBankPublisher(ReverbBankPublisher&&) = delete;
    ReverbBankPublisher& operator=(ReverbBankPublisher&&) = delete;

    /** Publishes one fully-built bank; called only by the control owner. */
    template <typename Commit>
    PublishResult publish(std::unique_ptr<Bank> candidate,
                          std::uint32_t latency,
                          Commit&& commit) noexcept
    {
        static_assert(std::is_nothrow_invocable_v<Commit&>,
                      "Reverb publication commit must be noexcept");
        assert(candidate != nullptr);
        Hooks::control(Hooks::ControlPoint::candidateReady);
        Hooks::control(Hooks::ControlPoint::beforeFirstScan);

        SlotChoice choice = scanOnce();
        if (!choice.valid)
        {
            const std::uint32_t observed =
                pendingToken_.load(std::memory_order_acquire);
            const bool exactPinnedPending = observed != emptyToken
                && observed == pinnedToken_;
            const bool generationLimitPending = observed != emptyToken
                && tokenGeneration(observed) >= GenerationMax;
            if (!exactPinnedPending && !generationLimitPending)
            {
                Hooks::control(Hooks::ControlPoint::beforePendingExchange);
                const std::uint32_t old =
                    pendingToken_.exchange(emptyToken, std::memory_order_acq_rel);
                Hooks::control(Hooks::ControlPoint::afterPendingExchange);
                if (old != emptyToken && !retireControlOwned(old))
                    return PublishResult::invariantViolation;
                choice = scanOnce(); // The sole bounded rescan.
            }
        }

        if (!choice.valid)
        {
            Hooks::control(Hooks::ControlPoint::noCapacity);
            return PublishResult::noCapacity;
        }

        Hooks::control(Hooks::ControlPoint::afterSlotSelection);
        auto& slot = slots_[choice.index];
        nextGeneration_[choice.index] = choice.generation;
        slot.stateWord.store(encodeState(choice.generation, Phase::building),
                             std::memory_order_relaxed);
        if (choice.reclaimsRetired)
            Hooks::control(Hooks::ControlPoint::beforeOldReclaim);
        slot.bank = std::move(candidate);
        slot.stateWord.store(encodeState(choice.generation, Phase::published),
                             std::memory_order_release);
        Hooks::control(Hooks::ControlPoint::afterPublishedStore);

        const std::uint32_t token = makeToken(choice.generation, choice.index);
        Hooks::control(Hooks::ControlPoint::beforePendingExchange);
        const std::uint32_t old =
            pendingToken_.exchange(token, std::memory_order_acq_rel);
        Hooks::control(Hooks::ControlPoint::afterPendingExchange);
        if (old != emptyToken && !retireControlOwned(old))
        {
            // This is unreachable under the single-writer contract. Keep the
            // newly published exact token intact and report the broken model.
            return PublishResult::invariantViolation;
        }

        Hooks::control(Hooks::ControlPoint::beforeCommit);
        commit();
        latestToken_ = token;
        publicationMetadata_.store(packMetadata(latency),
                                   std::memory_order_release);
        scanStart_ = (choice.index + 1u) & slotMask;
        Hooks::control(Hooks::ControlPoint::afterCommit);
        return PublishResult::published;
    }

    /** One fixed-operation audio-boundary adoption; returns the active bank. */
    [[nodiscard]] Bank* adoptAtBoundary() noexcept
    {
        Hooks::audioExchange();
        const std::uint32_t next =
            pendingToken_.exchange(emptyToken, std::memory_order_acq_rel);
        Hooks::audioAfterExchange(next != emptyToken);
        if (next != emptyToken)
        {
            const std::size_t nextIndex = tokenIndex(next);
            const std::uint32_t nextGeneration = tokenGeneration(next);
            const bool nextValid =
                slots_[nextIndex].stateWord.load(std::memory_order_acquire)
                == encodeState(nextGeneration, Phase::published);
            Hooks::audioValidation(nextValid);
            assert(nextValid);
            if (nextValid)
            {
                slots_[nextIndex].stateWord.store(
                    encodeState(nextGeneration, Phase::active),
                    std::memory_order_release);
                Hooks::audioStateStore();

                const std::uint32_t old = activeToken_;
                activeToken_ = next;
                if (old != emptyToken)
                {
                    const std::size_t oldIndex = tokenIndex(old);
                    const std::uint32_t oldGeneration = tokenGeneration(old);
                    const bool oldValid =
                        slots_[oldIndex].stateWord.load(std::memory_order_acquire)
                        == encodeState(oldGeneration, Phase::active);
                    Hooks::audioValidation(oldValid);
                    assert(oldValid);
                    if (oldValid)
                    {
                        slots_[oldIndex].stateWord.store(
                            encodeState(oldGeneration, Phase::retired),
                            std::memory_order_release);
                        Hooks::audioStateStore();
                    }
                }
            }
        }

        return activeToken_ == emptyToken
            ? nullptr
            : slots_[tokenIndex(activeToken_)].bank.get();
    }

    /** Ends the previous accessor lifetime and pins the exact latest bank. */
    [[nodiscard]] Bank* pinLatest() noexcept
    {
        const std::uint32_t previous = pinnedToken_;
        pinnedToken_ = emptyToken;
        const std::uint32_t latest = latestToken_;
        if (previous != emptyToken && previous != latest)
            reclaimIfRetired(previous);

        if (latest == emptyToken) return nullptr;
        const std::size_t index = tokenIndex(latest);
        const std::uint32_t state =
            slots_[index].stateWord.load(std::memory_order_acquire);
        const Phase phase = statePhase(state);
        const bool valid = stateGeneration(state) == tokenGeneration(latest)
            && phase != Phase::free && phase != Phase::building
            && phase != Phase::exhausted && slots_[index].bank != nullptr;
        Hooks::controlValidation(valid);
        assert(valid);
        if (!valid) return nullptr;
        pinnedToken_ = latest;
        return slots_[index].bank.get();
    }

    [[nodiscard]] bool isLoaded() const noexcept
    {
        return (publicationMetadata_.load(std::memory_order_acquire)
                & loadedMask) != 0u;
    }

    [[nodiscard]] int latency() const noexcept
    {
        return static_cast<int>(publicationMetadata_.load(
            std::memory_order_acquire) & latencyMask);
    }

    /** Joined-owner destruction; safe to call repeatedly. */
    void shutdown() noexcept
    {
        pendingToken_.exchange(emptyToken, std::memory_order_acq_rel);
        activeToken_ = emptyToken;
        latestToken_ = emptyToken;
        pinnedToken_ = emptyToken;
        publicationMetadata_.store(0u, std::memory_order_release);
        for (auto& slot : slots_)
        {
            slot.bank.reset();
            slot.stateWord.store(encodeState(0u, Phase::free),
                                 std::memory_order_relaxed);
        }
        nextGeneration_.fill(0u);
        scanStart_ = 0u;
    }

    // Deterministic detail-level observability used by the dedicated tests.
    [[nodiscard]] std::uint32_t stateWordForTest(std::size_t index) const noexcept
    {
        return slots_[index].stateWord.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint32_t pendingTokenForTest() const noexcept
    {
        return pendingToken_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint32_t activeTokenForTest() const noexcept { return activeToken_; }
    [[nodiscard]] std::uint32_t latestTokenForTest() const noexcept { return latestToken_; }
    [[nodiscard]] std::uint32_t pinnedTokenForTest() const noexcept { return pinnedToken_; }
    [[nodiscard]] std::size_t residentBanksForTest() const noexcept
    {
        std::size_t count = 0;
        for (const auto& slot : slots_) count += slot.bank != nullptr ? 1u : 0u;
        return count;
    }
    [[nodiscard]] const void* atomicWordAddressForTest(
        std::size_t index) const noexcept
    {
        if (index < slots_.size())
            return static_cast<const void*>(&slots_[index].stateWord);
        if (index == slots_.size())
            return static_cast<const void*>(&pendingToken_);
        return static_cast<const void*>(&publicationMetadata_);
    }
    [[nodiscard]] static constexpr std::size_t atomicWordCountForTest() noexcept
    {
        return 6u;
    }
    [[nodiscard]] static constexpr Phase phaseForTest(std::uint32_t state) noexcept
    {
        return statePhase(state);
    }
    [[nodiscard]] static constexpr std::uint32_t generationForTest(
        std::uint32_t state) noexcept
    {
        return stateGeneration(state);
    }

private:
    static constexpr std::uint32_t emptyToken = 0u;
    static constexpr std::uint32_t slotMask = 3u;
    static constexpr std::uint32_t phaseMask = 7u;
    static constexpr std::uint32_t loadedMask = 0x80000000u;
    static constexpr std::uint32_t latencyMask = 0x7fffffffu;

    struct BankSlot
    {
        std::unique_ptr<Bank> bank;
        std::atomic<std::uint32_t> stateWord { 0u };
    };

    struct SlotChoice
    {
        std::size_t index = 0;
        std::uint32_t generation = 0;
        bool valid = false;
        bool reclaimsRetired = false;
    };

    [[nodiscard]] static constexpr std::uint32_t makeToken(
        std::uint32_t generation, std::size_t index) noexcept
    {
        return (generation << 2u) | static_cast<std::uint32_t>(index);
    }

    [[nodiscard]] static constexpr std::size_t tokenIndex(
        std::uint32_t token) noexcept
    {
        return static_cast<std::size_t>(token & slotMask);
    }

    [[nodiscard]] static constexpr std::uint32_t tokenGeneration(
        std::uint32_t token) noexcept
    {
        return token >> 2u;
    }

    [[nodiscard]] static constexpr std::uint32_t encodeState(
        std::uint32_t generation, Phase phase) noexcept
    {
        return (generation << 3u) | static_cast<std::uint32_t>(phase);
    }

    [[nodiscard]] static constexpr std::uint32_t stateGeneration(
        std::uint32_t state) noexcept
    {
        return state >> 3u;
    }

    [[nodiscard]] static constexpr Phase statePhase(std::uint32_t state) noexcept
    {
        return static_cast<Phase>(state & phaseMask);
    }

    [[nodiscard]] static constexpr std::uint32_t packMetadata(
        std::uint32_t latency) noexcept
    {
        return loadedMask | std::min(latency, latencyMask);
    }

    [[nodiscard]] SlotChoice scanOnce() noexcept
    {
        for (std::size_t offset = 0; offset < slots_.size(); ++offset)
        {
            Hooks::controlScan();
            const std::size_t index = (scanStart_ + offset) & slotMask;
            auto& slot = slots_[index];
            const std::uint32_t state =
                slot.stateWord.load(std::memory_order_acquire);
            const Phase phase = statePhase(state);
            const std::uint32_t generation = stateGeneration(state);
            const std::uint32_t exactToken = makeToken(generation, index);

            if (phase == Phase::free)
            {
                const bool pristine = generation == 0u
                    && nextGeneration_[index] == 0u && slot.bank == nullptr;
                Hooks::controlValidation(pristine);
                assert(pristine);
                if (pristine) return { index, 1u, true, false };
                continue;
            }

            if (phase != Phase::retired || exactToken == pinnedToken_)
                continue;

            const bool generationMatches =
                nextGeneration_[index] == generation;
            Hooks::controlValidation(generationMatches);
            assert(generationMatches);
            if (!generationMatches) continue;

            if (generation >= GenerationMax)
            {
                Hooks::control(Hooks::ControlPoint::beforeOldReclaim);
                slot.bank.reset();
                slot.stateWord.store(encodeState(generation, Phase::exhausted),
                                     std::memory_order_release);
                continue;
            }
            return { index, generation + 1u, true, true };
        }
        return {};
    }

    bool retireControlOwned(std::uint32_t token) noexcept
    {
        const std::size_t index = tokenIndex(token);
        const std::uint32_t generation = tokenGeneration(token);
        auto& slot = slots_[index];
        const bool valid =
            slot.stateWord.load(std::memory_order_acquire)
            == encodeState(generation, Phase::published);
        Hooks::controlValidation(valid);
        assert(valid);
        if (!valid) return false;

        Hooks::control(Hooks::ControlPoint::beforeOldRetirement);
        slot.stateWord.store(encodeState(generation, Phase::retired),
                             std::memory_order_release);
        if (token != pinnedToken_)
        {
            Hooks::control(Hooks::ControlPoint::beforeOldReclaim);
            slot.bank.reset();
        }
        return true;
    }

    void reclaimIfRetired(std::uint32_t token) noexcept
    {
        auto& slot = slots_[tokenIndex(token)];
        const std::uint32_t state =
            slot.stateWord.load(std::memory_order_acquire);
        if (state == encodeState(tokenGeneration(token), Phase::retired))
        {
            Hooks::control(Hooks::ControlPoint::beforeOldReclaim);
            slot.bank.reset();
        }
    }

    std::array<BankSlot, 4> slots_ {};
    std::atomic<std::uint32_t> pendingToken_ { emptyToken };
    std::atomic<std::uint32_t> publicationMetadata_ { 0u };
    std::uint32_t activeToken_ = emptyToken;  // Audio-owner only.
    std::uint32_t latestToken_ = emptyToken;  // Control-owner only.
    std::uint32_t pinnedToken_ = emptyToken;  // Control-owner only.
    std::array<std::uint32_t, 4> nextGeneration_ {};
    std::size_t scanStart_ = 0;                // Control-owner only.
};

} // namespace detail

/**
 * @class Reverb
 * @brief Convolution reverb with IR loading, dry/wet, and pre-delay.
 *
 * Internally manages one Convolver per channel. The IR is automatically
 * resampled if its sample rate differs from the processing sample rate.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class Reverb
{
protected:
    struct ConvolverBank
    {
        std::vector<Convolver<T>> convolvers;
    };

    using Publisher = detail::ReverbBankPublisher<ConvolverBank>;

public:
    Reverb() = default;
    ~Reverb() noexcept { bankPublisher_.shutdown(); }

    Reverb(const Reverb&) = delete;
    Reverb& operator=(const Reverb&) = delete;
    Reverb(Reverb&&) = delete;
    Reverb& operator=(Reverb&&) = delete;

    // -- Lifecycle --------------------------------------------------------------

    /**
     * @brief Prepares the reverb for processing.
     *
     * Allocates internal buffers, sets up the dry/wet mixer (including the
     * dry-path compensation for the convolution latency) and the pre-delay.
     * If an IR was loaded before prepare(), it will be re-applied.
     * An invalid spec (non-positive or non-finite fields) is a no-op that
     * keeps the previous state.
     *
     * @param spec Audio environment (sample rate, block size, channels).
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return; // release-safe: keep previous state

        // The convolution engine partitions at the next power of two of the
        // max block size (>= 2, matching Convolver's own normalisation), and
        // that is exactly its processing latency. Clamp before the round-up
        // loop so an absurd block size cannot overflow the shift.
        const int blockSize = std::clamp(spec.maxBlockSize, 1, 1 << 20);
        int fftBlock = 2;
        while (fftBlock < blockSize) fftBlock <<= 1;
        // Build every potentially-throwing setup object locally. The stopped
        // audio/setup ownership contract makes the final moves atomic as one
        // logical transaction even though the members themselves are plain.
        DryWetMixer<T> nextMixer;
        nextMixer.prepare(spec);
        nextMixer.setLatencyCompensation(fftBlock);

        // Pre-delay ring buffers (one per channel, max 500ms)
        const int maxDelaySamples = static_cast<int>(spec.sampleRate * 0.5) + 1;
        std::vector<RingBuffer<T>> nextPreDelayBuffers(
            static_cast<size_t>(spec.numChannels));
        for (auto& rb : nextPreDelayBuffers)
            rb.prepare(maxDelaySamples);
        const int nextPreDelaySamples = calculatePreDelaySamples(spec);

        if (irStorage_.empty())
        {
            spec_ = spec;
            fftBlockSize_ = fftBlock;
            mixer_ = std::move(nextMixer);
            preDelayBuffers_ = std::move(nextPreDelayBuffers);
            preDelaySamples_.store(nextPreDelaySamples,
                                   std::memory_order_relaxed);
            return;
        }

        auto candidate = buildBank(irStorage_, irLength_, irChannels_,
                                   irSampleRate_, spec, fftBlock,
                                   decayScale_.load(std::memory_order_relaxed),
                                   stretch_.load(std::memory_order_relaxed));
        const std::uint32_t latency = bankLatency(*candidate);
        (void)bankPublisher_.publish(
            std::move(candidate), latency,
            [&]() noexcept {
                spec_ = spec;
                fftBlockSize_ = fftBlock;
                mixer_ = std::move(nextMixer);
                preDelayBuffers_ = std::move(nextPreDelayBuffers);
                preDelaySamples_.store(nextPreDelaySamples,
                                       std::memory_order_relaxed);
            });
    }

    /**
     * @brief Processes audio through the reverb.
     *
     * Flow: pushDry -> pre-delay -> convolve -> mixWet (with the dry delayed
     * by the convolution latency so both paths stay aligned). Without a
     * loaded IR the audio passes through untouched (and getLatency() is 0).
     * Channels beyond the prepared count pass through untouched.
     *
     * Thread-safety: a fixed four-slot ownership handoff adopts at most one
     * complete bank at the block boundary. The audio path performs one atomic
     * exchange and uses a raw pointer whose slot cannot be reclaimed until a
     * later boundary release-retires it.
     *
     * @param buffer Audio data to process in-place.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        ConvolverBank* const bank = bankPublisher_.adoptAtBoundary();
        if (!bank || bank->convolvers.empty()) return;

        const int nCh = std::min(buffer.getNumChannels(),
                                 static_cast<int>(bank->convolvers.size()));
        const int nS  = buffer.getNumSamples();

        mixer_.pushDry(buffer);

        int preDelSamp = preDelaySamples_.load(std::memory_order_relaxed);
        T mixVal = mix_.load(std::memory_order_relaxed);

        for (int ch = 0; ch < nCh; ++ch)
        {
            T* data = buffer.getChannel(ch);

            if (preDelSamp > 0)
            {
                auto& ring = preDelayBuffers_[static_cast<size_t>(ch)];
                for (int i = 0; i < nS; ++i)
                {
                    ring.push(data[i]);
                    data[i] = ring.read(preDelSamp);
                }
            }

            bank->convolvers[static_cast<size_t>(ch)].processInPlace(data, nS);
        }

        mixer_.mixWet(buffer, mixVal);
    }

    /**
     * @brief Resets the DSP state (convolver tails, pre-delay, mixer). RT-Safe.
     *
     * The loaded IR stays loaded: resetting used to drop the convolver bank
     * entirely, silently unloading the reverb (a host reset on stop/start
     * left it as a dry passthrough until the next loadIR()).
     */
    void reset() noexcept
    {
        // Reset the snapshot we can see; if a concurrent load publishes a
        // replacement bank it arrives freshly zeroed anyway.
        if (ConvolverBank* const bank = bankPublisher_.adoptAtBoundary())
            for (auto& conv : bank->convolvers)
                conv.reset();
        for (auto& rb : preDelayBuffers_)
            rb.reset();
        mixer_.reset();
    }

    // -- Level 1: Simple API ----------------------------------------------------

#ifndef DSPARK_NO_FILE_IO
    /**
     * @brief Loads an impulse response from a WAV file.
     *
     * The IR is automatically resampled if the WAV sample rate differs from
     * the processing sample rate. Multi-channel IRs are supported (one
     * convolver per channel); mono IRs are duplicated across all channels.
     * Empty or degenerate files are rejected.
     *
     * @param wavFilePath Path to the WAV file.
     * @return True if the IR was loaded successfully.
     */
    bool loadIR(const char* wavFilePath)
    {
        WavFile wav;
        if (!wav.openRead(wavFilePath))
            return false;

        auto info = wav.getInfo();
        if (info.numSamples <= 0 || info.numChannels <= 0
            || info.numSamples > (static_cast<int64_t>(1) << 30)
            || !(info.sampleRate > 0))
        {
            wav.close();
            return false;
        }

        AudioBuffer<T> irBuf;
        irBuf.resize(info.numChannels, static_cast<int>(info.numSamples));
        wav.readSamples(irBuf.toView());
        wav.close();

        const int nextChannels = info.numChannels;
        const int nextLength = static_cast<int>(info.numSamples);
        std::vector<T> nextStorage(
            static_cast<size_t>(nextChannels) * static_cast<size_t>(nextLength));
        for (int ch = 0; ch < nextChannels; ++ch)
        {
            const T* src = irBuf.getChannel(ch);
            T* dst = nextStorage.data()
                   + static_cast<size_t>(ch) * static_cast<size_t>(nextLength);
            std::copy_n(src, nextLength, dst);
        }
        return commitImpulseResponse(std::move(nextStorage), nextLength,
                                     nextChannels, info.sampleRate);
    }
#endif // DSPARK_NO_FILE_IO

    /**
     * @brief Sets the dry/wet mix.
     * @param dryWet 0.0 = fully dry (no reverb), 1.0 = fully wet.
     *               Non-finite values are ignored.
     */
    void setMix(T dryWet) noexcept
    {
        if (!std::isfinite(dryWet)) return;
        mix_.store(std::clamp(dryWet, T(0), T(1)), std::memory_order_relaxed);
    }

    // -- Level 2: Intermediate API ----------------------------------------------

    /**
     * @brief Loads an IR from raw sample data.
     *
     * @param data         Pointer to mono IR samples.
     * @param length       Number of samples (must be > 0).
     * @param irSampleRate Sample rate of the IR data (must be > 0 and finite).
     * @return True if the IR was accepted (invalid arguments are rejected).
     */
    bool loadIR(const T* data, int length, double irSampleRate)
    {
        if (data == nullptr || length <= 0
            || !std::isfinite(irSampleRate) || !(irSampleRate > 0.0))
            return false;

        std::vector<T> nextStorage(data, data + length);
        return commitImpulseResponse(std::move(nextStorage), length, 1,
                                     irSampleRate);
    }

    /**
     * @brief Sets the pre-delay time in milliseconds.
     *
     * Pre-delay adds a gap before the reverb tail starts, creating a sense
     * of room size. Typical values: 0-50 ms. Applied immediately (a live
     * change may click); intended as a setup control, not for automation.
     *
     * @param ms Pre-delay in milliseconds, clamped to [0, 500] (the ring
     *           buffer allocation). Non-finite values are ignored.
     */
    void setPreDelay(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        preDelayMs_.store(std::clamp(ms, T(0), T(500)), std::memory_order_relaxed);
        updatePreDelay();
    }

    /**
     * @brief Scales the decay time (T60) of the loaded IR.
     *
     * The IR's own decay rate is estimated from its Schroeder backward
     * energy curve (T20 fit between -5 dB and -25 dB), then an exponential
     * envelope is applied from the direct-sound peak onward so the shaped
     * IR decays at `scale` times the original T60. The direct sound and
     * early part are preserved.
     *
     * Values below 1 shorten the tail AND trim the IR where its energy
     * falls below -100 dB, so the convolution gets proportionally cheaper
     * (about half the CPU at 0.5 on a typical exponential hall). Values
     * above 1 lengthen the tail; this also lifts whatever noise floor the
     * recording has, so moderate boosts (up to 2x) are the useful range.
     * IRs without a broadly exponential tail (gated/reversed effects) are
     * left unshaped.
     *
     * Setup/UI threads only: rebuilds the convolver bank (allocates) and
     * publishes it atomically, exactly like loadIR(). Not meant for
     * per-block automation. Non-finite values are ignored.
     *
     * @param scale T60 multiplier, clamped to [0.25, 2]. 1 = as loaded.
     */
    void setDecayScale(T scale)
    {
        if (!std::isfinite(scale)) return;
        const T next = std::clamp(scale, T(0.25), T(2));
        if (next == decayScale_.load(std::memory_order_relaxed)) return;
        if (spec_.sampleRate <= 0 || irStorage_.empty())
        {
            decayScale_.store(next, std::memory_order_relaxed);
            return;
        }
        auto candidate = buildBank(irStorage_, irLength_, irChannels_,
                                   irSampleRate_, spec_, fftBlockSize_, next,
                                   stretch_.load(std::memory_order_relaxed));
        const std::uint32_t latency = bankLatency(*candidate);
        (void)bankPublisher_.publish(
            std::move(candidate), latency,
            [&]() noexcept { decayScale_.store(next, std::memory_order_relaxed); });
    }

    /**
     * @brief Stretches the loaded IR in time (tape-speed style).
     *
     * The IR is resampled by the given ratio: 2.0 doubles its length
     * (larger, darker space, roughly an octave down in coloration), 0.5
     * halves it (smaller, brighter). Decay scale and stretch compose:
     * effective T60 is approximately original * decayScale * stretch.
     *
     * Setup/UI threads only: rebuilds the convolver bank (allocates) and
     * publishes it atomically, exactly like loadIR(). Non-finite values are
     * ignored.
     *
     * @param ratio Time-stretch ratio, clamped to [0.5, 2]. 1 = as loaded.
     */
    void setStretch(T ratio)
    {
        if (!std::isfinite(ratio)) return;
        const T next = std::clamp(ratio, T(0.5), T(2));
        if (next == stretch_.load(std::memory_order_relaxed)) return;
        if (spec_.sampleRate <= 0 || irStorage_.empty())
        {
            stretch_.store(next, std::memory_order_relaxed);
            return;
        }
        auto candidate = buildBank(irStorage_, irLength_, irChannels_,
                                   irSampleRate_, spec_, fftBlockSize_,
                                   decayScale_.load(std::memory_order_relaxed),
                                   next);
        const std::uint32_t latency = bankLatency(*candidate);
        (void)bankPublisher_.publish(
            std::move(candidate), latency,
            [&]() noexcept { stretch_.store(next, std::memory_order_relaxed); });
    }

    /** @brief Returns the current IR decay scale. */
    [[nodiscard]] T getDecayScale() const noexcept { return decayScale_.load(std::memory_order_relaxed); }

    /** @brief Returns the current IR stretch ratio. */
    [[nodiscard]] T getStretch() const noexcept { return stretch_.load(std::memory_order_relaxed); }

    // -- Level 3: Expert API ----------------------------------------------------

    /**
     * @brief Direct access to a channel's Convolver (GUI thread only).
     *
     * Lifetime of the returned reference: valid until YOUR next call to
     * getConvolver() on this object, or until the object is destroyed. It is
     * NOT invalidated by a concurrent loadIR() / setDecayScale() / setStretch()
     * / setState() / prepare(), because this accessor pins the exact slot and
     * generation it hands a reference into; the publication those calls make
     * simply is not what you are looking at any more, so what you hold is the
     * bank as of your call, kept alive for you.
     *
     * Getting that wrong is what this accessor used to do: it took a local
     * snapshot of the bank, returned a reference into it, and let the snapshot
     * die at the closing brace -- so the next publication freed the storage
     * under the caller's reference (a heap-use-after-free reachable
     * single-threaded through this documented usage). Ownership at the moment
     * of the call is not the same question as how long the result lives.
     *
     * Still GUI thread only, and still not for the audio thread: one caller
     * owns the pin, and the Convolver it hands back is not synchronized with
     * the audio thread's use of the live bank. Before an IR is loaded (or with
     * an out-of-range channel) it returns an inert fallback engine instead of
     * dereferencing a null bank.
     *
     * @param channel Channel index (clamped into the bank's range).
     */
    Convolver<T>& getConvolver(int channel = 0)
    {
        ConvolverBank* const bank = bankPublisher_.pinLatest();
        if (!bank || bank->convolvers.empty())
            return fallbackConvolver_;
        const int n = static_cast<int>(bank->convolvers.size());
        channel = std::clamp(channel, 0, n - 1);
        return bank->convolvers[static_cast<size_t>(channel)];
    }

    /**
     * @brief Direct access to the DryWetMixer (processing thread).
     *
     * The mixer is live audio-thread state: it carries the dry-signal history
     * processBlock() writes every block. The reference itself never dangles
     * (the mixer is a member, so it lives as long as this object), but reading
     * or mutating it from another thread while audio runs is unsynchronized.
     * Treat it as belonging to the thread that owns processBlock().
     */
    DryWetMixer<T>& getMixer() { return mixer_; }

    /** @brief Returns true if an IR has been loaded and applied. */
    [[nodiscard]] bool isLoaded() const noexcept
    {
        return bankPublisher_.isLoaded();
    }

    /** @brief Returns the current mix value. */
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Returns the current pre-delay in ms. */
    [[nodiscard]] T getPreDelay() const noexcept { return preDelayMs_.load(std::memory_order_relaxed); }

    /**
     * @brief Returns the convolution latency in samples.
     *
     * 0 without an IR (the audio passes through untouched); the convolver's
     * partition latency once an IR is loaded. The dry path is internally
     * delayed by the same amount, so this is the whole effect's latency.
     * Hosts must re-read it after loading an IR.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return bankPublisher_.latency();
    }


    /** @brief Serializes the parameter state. The impulse response itself is
     *  content (load it with loadIR), not a preset parameter. */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("CRVB"), 1);
        w.write("mix", mix_.load(std::memory_order_relaxed));
        w.write("preDelay", preDelayMs_.load(std::memory_order_relaxed));
        w.write("decayScale", decayScale_.load(std::memory_order_relaxed));
        w.write("stretch", stretch_.load(std::memory_order_relaxed));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("CRVB")) return false;
        setMix(static_cast<T>(r.read("mix", 0.3f)));
        setPreDelay(static_cast<T>(r.read("preDelay", 0.0f)));
        // Store both shaping values first, then rebuild once (each setter
        // would otherwise trigger its own IR rebuild). Non-finite blob values
        // keep the current settings.
        T ds = static_cast<T>(r.read("decayScale", 1.0f));
        T st = static_cast<T>(r.read("stretch", 1.0f));
        if (!std::isfinite(ds)) ds = decayScale_.load(std::memory_order_relaxed);
        if (!std::isfinite(st)) st = stretch_.load(std::memory_order_relaxed);
        ds = std::clamp(ds, T(0.25), T(2));
        st = std::clamp(st, T(0.5), T(2));
        const bool shapeChanged =
            ds != decayScale_.load(std::memory_order_relaxed)
            || st != stretch_.load(std::memory_order_relaxed);
        if (!shapeChanged) return true;
        if (spec_.sampleRate <= 0 || irStorage_.empty())
        {
            decayScale_.store(ds, std::memory_order_relaxed);
            stretch_.store(st, std::memory_order_relaxed);
            return true;
        }

        auto candidate = buildBank(irStorage_, irLength_, irChannels_,
                                   irSampleRate_, spec_, fftBlockSize_, ds, st);
        const std::uint32_t latency = bankLatency(*candidate);
        const auto result = bankPublisher_.publish(
            std::move(candidate), latency,
            [&]() noexcept {
                decayScale_.store(ds, std::memory_order_relaxed);
                stretch_.store(st, std::memory_order_relaxed);
            });
        if (result != Publisher::PublishResult::published) return false;
        return true;
    }

protected:
    [[nodiscard]] int calculatePreDelaySamples(const AudioSpec& spec) const noexcept
    {
        if (!(spec.sampleRate > 0)) return 0;
        const int maxSamp = static_cast<int>(spec.sampleRate * 0.5);
        const int samp = static_cast<int>(
            static_cast<T>(spec.sampleRate)
            * preDelayMs_.load(std::memory_order_relaxed) / T(1000));
        return std::clamp(samp, 0, maxSamp);
    }

    void updatePreDelay() noexcept
    {
        if (spec_.sampleRate > 0)
        {
            // The pre-delay ring buffers hold 500 ms; clamp so an over-range pre-delay
            // can't read past the buffer (RingBuffer::read would wrap to a wrong sample).
            preDelaySamples_.store(calculatePreDelaySamples(spec_),
                                   std::memory_order_relaxed);
        }
    }

    bool commitImpulseResponse(std::vector<T> nextStorage,
                               int nextLength,
                               int nextChannels,
                               double nextSampleRate)
    {
        if (spec_.sampleRate <= 0 || fftBlockSize_ <= 0)
        {
            irStorage_.swap(nextStorage);
            irLength_ = nextLength;
            irChannels_ = nextChannels;
            irSampleRate_ = nextSampleRate;
            return true;
        }

        auto candidate = buildBank(nextStorage, nextLength, nextChannels,
                                   nextSampleRate, spec_, fftBlockSize_,
                                   decayScale_.load(std::memory_order_relaxed),
                                   stretch_.load(std::memory_order_relaxed));
        const std::uint32_t latency = bankLatency(*candidate);
        const auto result = bankPublisher_.publish(
            std::move(candidate), latency,
            [&]() noexcept {
                irStorage_.swap(nextStorage);
                irLength_ = nextLength;
                irChannels_ = nextChannels;
                irSampleRate_ = nextSampleRate;
            });
        return result == Publisher::PublishResult::published;
    }

    [[nodiscard]] std::unique_ptr<ConvolverBank> buildBank(
        const std::vector<T>& source,
        int sourceLength,
        int sourceChannels,
        double sourceSampleRate,
        const AudioSpec& processingSpec,
        int fftBlock,
        T decayScale,
        T stretchRatio) const
    {
        auto newBank = std::make_unique<ConvolverBank>();
        newBank->convolvers.resize(
            static_cast<size_t>(processingSpec.numChannels));

        // IR shaping controls, always applied to the stored original.
        // Stretch works by declaring a scaled source rate and letting the
        // resampling stage do the time-scaling (tape-speed semantics).
        const double dScale = static_cast<double>(decayScale);
        const double stretch = static_cast<double>(stretchRatio);
        const bool doShape = std::abs(dScale - 1.0) > 1e-6;
        const double effIrRate = sourceSampleRate / std::max(stretch, 0.01);

        std::vector<T> shaped;   // lazy decay-shaped copy of one IR channel
        int shapedCh = -1;

        for (int ch = 0; ch < processingSpec.numChannels; ++ch)
        {
            // Pick IR channel: use corresponding channel if available, else mono (ch 0)
            const int irCh = (ch < sourceChannels) ? ch : 0;
            const T* irData = source.data()
                            + static_cast<size_t>(irCh)
                            * static_cast<size_t>(sourceLength);
            int irLen = sourceLength;

            if (doShape)
            {
                if (shapedCh != irCh)
                {
                    shaped = shapeDecay(irData, sourceLength, dScale);
                    shapedCh = irCh;
                }
                if (!shaped.empty())
                {
                    irData = shaped.data();
                    irLen = static_cast<int>(shaped.size());
                }
            }

            auto& conv = newBank->convolvers[static_cast<size_t>(ch)];

            // Resample if the (stretch-adjusted) IR rate differs from the engine
            if (std::abs(effIrRate - processingSpec.sampleRate) > 1.0)
            {
                Resampler<T> resampler;
                resampler.prepare(effIrRate, processingSpec.sampleRate);
                // Size from the resampler's own bound (INT_MAX-safe): the old
                // floor(n*ratio)+1 arithmetic could overflow the int cast with
                // an extreme rate ratio.
                std::vector<T> resampled(
                    static_cast<size_t>(resampler.getMaxOutputSamples(irLen)));
                int produced = resampler.processBlock(irData, irLen,
                                                      resampled.data());

                conv.prepare(fftBlock, resampled.data(), produced);
            }
            else
            {
                conv.prepare(fftBlock, irData, irLen);
            }
        }
        return newBank;
    }

    [[nodiscard]] static std::uint32_t bankLatency(
        const ConvolverBank& bank) noexcept
    {
        if (bank.convolvers.empty()) return 0u;
        return static_cast<std::uint32_t>(
            std::max(0, bank.convolvers.front().getLatency()));
    }

    /**
     * @brief Returns a decay-scaled copy of one IR channel (see setDecayScale).
     *
     * Estimates the source decay rate from the Schroeder backward-energy
     * curve (T20 fit: -5 dB to -25 dB crossings of the EDC), then applies
     * exp(-k * (n - peak)) so the result decays at `factor` times the
     * original T60. The direct sound (up to the peak) is untouched.
     * Returns an empty vector when the IR has no measurable exponential
     * decay (too short, or gated); the caller then uses the original data.
     */
    [[nodiscard]] std::vector<T> shapeDecay(const T* ir, int len,
                                            double factor) const
    {
        if (!ir || len < 64) return {};

        // Total energy + direct-sound peak (double accumulation).
        double total = 0.0;
        double peakMag = 0.0;
        int peak = 0;
        for (int n = 0; n < len; ++n)
        {
            const double v = static_cast<double>(ir[n]);
            total += v * v;
            const double m = std::abs(v);
            if (m > peakMag) { peakMag = m; peak = n; }
        }
        if (total <= 1e-30 || peakMag <= 0.0) return {};

        // Schroeder EDC crossings at -5 dB and -25 dB (energy ratios).
        constexpr double r5  = 0.31622776601683794;    // 10^(-5/10)
        constexpr double r25 = 0.0031622776601683794;  // 10^(-25/10)
        int t5 = -1, t25 = -1;
        double tail = total;
        for (int n = 0; n < len; ++n)
        {
            const double ratio = tail / total;
            if (t5 < 0 && ratio <= r5 && n > peak) t5 = n;
            if (ratio <= r25 && n > peak) { t25 = n; break; }
            const double v = static_cast<double>(ir[n]);
            tail -= v * v;
        }
        if (t5 < 0 || t25 < 0 || t25 - t5 < 32) return {};  // no usable slope

        // Amplitude decay rate: the EDC drops 20 dB over (t25 - t5) samples,
        // so exp(-beta * t) with beta = ln(10) / (t25 - t5).
        const double beta = 2.302585092994046 / static_cast<double>(t25 - t5);
        const double k = beta * (1.0 / factor - 1.0);

        std::vector<T> out(static_cast<size_t>(len));
        const double gStep = std::exp(-k);
        double g = 1.0;
        for (int n = 0; n < len; ++n)
        {
            out[static_cast<size_t>(n)] = (n <= peak)
                ? ir[n]
                : static_cast<T>(static_cast<double>(ir[n]) * g);
            if (n >= peak) g *= gStep;
        }

        if (factor < 1.0)
        {
            // Trim where the shaped energy falls below -100 dB of its total:
            // the removed stretch is inaudible, and a shorter IR means fewer
            // convolution partitions (the CPU saving the shaping is for).
            double sTotal = 0.0;
            for (const T v : out) sTotal += static_cast<double>(v) * v;
            if (sTotal > 1e-30)
            {
                double sTail = sTotal;
                int cut = len;
                for (int n = 0; n < len; ++n)
                {
                    if (sTail / sTotal <= 1e-10) { cut = n; break; }
                    const double v = static_cast<double>(out[static_cast<size_t>(n)]);
                    sTail -= v * v;
                }
                cut = std::max(cut, 64);
                if (cut < len) out.resize(static_cast<size_t>(cut));
            }
        }
        else if (factor > 1.0)
        {
            // A raised envelope would end in a cliff at the IR boundary:
            // fade the final stretch (up to 20 ms at 48 kHz) with a raised
            // cosine so the lengthened tail closes cleanly.
            const int fade = std::min(len / 8, 960);
            const int start = len - fade;
            for (int i = 0; i < fade; ++i)
            {
                const double w = 0.5 * (1.0 + std::cos(3.141592653589793
                                        * static_cast<double>(i + 1)
                                        / static_cast<double>(fade)));
                const size_t idx = static_cast<size_t>(start + i);
                out[idx] = static_cast<T>(static_cast<double>(out[idx]) * w);
            }
        }
        return out;
    }

    AudioSpec spec_ {};
    int fftBlockSize_ = 0; ///< Convolver partition size = engine latency (set in prepare()).
    std::atomic<T> mix_ { T(0.3) };
    std::atomic<T> preDelayMs_ { T(0) };
    std::atomic<int> preDelaySamples_ { 0 };
    std::atomic<T> decayScale_ { T(1) };
    std::atomic<T> stretch_ { T(1) };

    // IR storage (GUI-thread only: rebuild source of truth)
    std::vector<T> irStorage_;
    int irLength_ = 0;
    int irChannels_ = 0;
    double irSampleRate_ = 0;

    Publisher bankPublisher_;
    std::vector<RingBuffer<T>> preDelayBuffers_;
    DryWetMixer<T> mixer_;
    Convolver<T> fallbackConvolver_; ///< Inert engine for getConvolver() with no bank.
};

} // namespace dspark
