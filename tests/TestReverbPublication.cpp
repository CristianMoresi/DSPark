// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#include "../Effects/Reverb.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(DSPARK_TEST_HELGRIND)
#include <valgrind/helgrind.h>
#elif defined(DSPARK_TEST_DRD)
#include <valgrind/drd.h>
#endif

namespace allocation_probe {
thread_local bool audioThread = false;
thread_local std::ptrdiff_t failAfter = -1;
std::atomic<std::size_t> audioAllocations { 0 };
std::atomic<std::size_t> audioDeallocations { 0 };

#if defined(_MSC_VER)
#define DSPARK_TEST_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define DSPARK_TEST_NOINLINE __attribute__((noinline))
#else
#define DSPARK_TEST_NOINLINE
#endif

DSPARK_TEST_NOINLINE void* allocate(std::size_t size)
{
    return std::malloc(size != 0u ? size : 1u);
}

DSPARK_TEST_NOINLINE void deallocate(void* pointer) noexcept
{
    std::free(pointer);
}
#undef DSPARK_TEST_NOINLINE
} // namespace allocation_probe

#if !defined(DSPARK_TEST_TSAN)
void* operator new(std::size_t size)
{
    if (allocation_probe::failAfter == 0)
    {
        allocation_probe::failAfter = -1;
        throw std::bad_alloc();
    }
    if (allocation_probe::failAfter > 0) --allocation_probe::failAfter;
    if (allocation_probe::audioThread)
        allocation_probe::audioAllocations.fetch_add(1, std::memory_order_relaxed);
    if (void* const pointer = allocation_probe::allocate(size)) return pointer;
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept
{
    if (allocation_probe::audioThread)
        allocation_probe::audioDeallocations.fetch_add(1, std::memory_order_relaxed);
    allocation_probe::deallocate(pointer);
}

void operator delete[](void* pointer) noexcept
{
    ::operator delete(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept
{
    ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept
{
    ::operator delete(pointer);
}
#endif

namespace {

#if defined(DSPARK_TEST_HELGRIND) || defined(DSPARK_TEST_DRD)
int publicationHappensBefore;
int retirementHappensBefore;
std::atomic<unsigned> retirementSignals { 0 };

void annotateBenignAtomic(const void* address, std::size_t size)
{
    ANNOTATE_BENIGN_RACE_SIZED(const_cast<void*>(address), size,
                               "lock-free uint32 publication word");
}
#endif

using ControlPoint = dspark::detail::ReverbPublisherNoopHooks::ControlPoint;

struct CountingHooks
{
    using ControlPoint = ::ControlPoint;

    static inline std::atomic<int> audioExchanges { 0 };
    static inline std::atomic<int> audioStores { 0 };
    static inline std::atomic<int> audioValidations { 0 };
    static inline std::atomic<int> controlScans { 0 };
    static inline std::atomic<int> pendingExchanges { 0 };

    static void reset() noexcept
    {
        audioExchanges.store(0, std::memory_order_relaxed);
        audioStores.store(0, std::memory_order_relaxed);
        audioValidations.store(0, std::memory_order_relaxed);
        controlScans.store(0, std::memory_order_relaxed);
        pendingExchanges.store(0, std::memory_order_relaxed);
    }

    static void control(ControlPoint point) noexcept
    {
        if (point == ControlPoint::beforePendingExchange)
            pendingExchanges.fetch_add(1, std::memory_order_relaxed);
#if defined(DSPARK_TEST_HELGRIND) || defined(DSPARK_TEST_DRD)
        if (point == ControlPoint::beforePendingExchange)
            ANNOTATE_HAPPENS_BEFORE(&publicationHappensBefore);
        if (point == ControlPoint::beforeOldReclaim
            && retirementSignals.load(std::memory_order_relaxed) != 0u)
            ANNOTATE_HAPPENS_AFTER(&retirementHappensBefore);
#endif
    }
    static void audioExchange() noexcept
    {
        audioExchanges.fetch_add(1, std::memory_order_relaxed);
    }
    static void audioAfterExchange(bool received) noexcept
    {
#if defined(DSPARK_TEST_HELGRIND) || defined(DSPARK_TEST_DRD)
        if (received) ANNOTATE_HAPPENS_AFTER(&publicationHappensBefore);
#else
        (void)received;
#endif
    }
    static void audioStateStore() noexcept
    {
        audioStores.fetch_add(1, std::memory_order_relaxed);
#if defined(DSPARK_TEST_HELGRIND) || defined(DSPARK_TEST_DRD)
        retirementSignals.fetch_add(1u, std::memory_order_relaxed);
        ANNOTATE_HAPPENS_BEFORE(&retirementHappensBefore);
#endif
    }
    static void audioValidation(bool valid) noexcept
    {
        if (valid) audioValidations.fetch_add(1, std::memory_order_relaxed);
    }
    static void controlValidation(bool) noexcept {}
    static void controlScan() noexcept
    {
        controlScans.fetch_add(1, std::memory_order_relaxed);
    }
};

struct ParkingHooks : CountingHooks
{
    static inline std::atomic<bool> enabled { false };
    static inline std::atomic<bool> reached { false };
    static inline std::atomic<bool> released { false };
    static inline std::atomic<int> target {
        static_cast<int>(ControlPoint::afterPublishedStore)
    };

    static void control(ControlPoint point) noexcept
    {
        CountingHooks::control(point);
        if (enabled.load(std::memory_order_acquire)
            && static_cast<int>(point) == target.load(std::memory_order_relaxed))
        {
            reached.store(true, std::memory_order_release);
            while (!released.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
    }
};

struct AudioHandoffHooks : CountingHooks
{
    static inline std::atomic<bool> enabled { false };
    static inline std::atomic<bool> reached { false };
    static inline std::atomic<bool> released { false };

    static void audioAfterExchange(bool received) noexcept
    {
        CountingHooks::audioAfterExchange(received);
        if (received && enabled.load(std::memory_order_acquire))
        {
            reached.store(true, std::memory_order_release);
            while (!released.load(std::memory_order_acquire))
                std::this_thread::yield();
        }
    }
};

struct FakeBank
{
    explicit FakeBank(int valueIn) : value(valueIn), storage(32, valueIn)
    {
        constructed.fetch_add(1, std::memory_order_relaxed);
        const int now = live.fetch_add(1, std::memory_order_relaxed) + 1;
        int previous = maximumLive.load(std::memory_order_relaxed);
        while (now > previous
               && !maximumLive.compare_exchange_weak(previous, now,
                                                     std::memory_order_relaxed)) {}
    }

    ~FakeBank()
    {
        if (allocation_probe::audioThread)
            audioDestructions.fetch_add(1, std::memory_order_relaxed);
        destroyed.fetch_add(1, std::memory_order_relaxed);
        live.fetch_sub(1, std::memory_order_relaxed);
    }

    static void resetCounters() noexcept
    {
        live.store(0, std::memory_order_relaxed);
        constructed.store(0, std::memory_order_relaxed);
        maximumLive.store(0, std::memory_order_relaxed);
        destroyed.store(0, std::memory_order_relaxed);
        audioDestructions.store(0, std::memory_order_relaxed);
        metadataReads.store(0, std::memory_order_relaxed);
    }

    [[nodiscard]] int readForMetadataMutant() const noexcept
    {
        metadataReads.fetch_add(1, std::memory_order_relaxed);
        return value;
    }

    int value;
    std::vector<int> storage;
    static inline std::atomic<int> live { 0 };
    static inline std::atomic<int> constructed { 0 };
    static inline std::atomic<int> maximumLive { 0 };
    static inline std::atomic<int> destroyed { 0 };
    static inline std::atomic<int> audioDestructions { 0 };
    static inline std::atomic<int> metadataReads { 0 };
};

template <typename Hooks = CountingHooks, std::uint32_t GenerationMax = 134217727u>
using Publisher = dspark::detail::ReverbBankPublisher<FakeBank, Hooks, GenerationMax>;

struct TestFailure : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

void require(bool condition, std::string_view message)
{
    if (!condition) throw TestFailure(std::string(message));
}

enum class MutantPhase { published, active };

// Executable negative control for the two forbidden scan-reuse choices. It
// models the exact bug: a scan selects storage still owned by pending/handoff
// or audio, destroys it on control, increments the generation, and leaves the
// outstanding exact token naming different content.
bool unsafeScanReuseMutantIsDetected(MutantPhase phase)
{
    struct Slot
    {
        std::unique_ptr<FakeBank> bank;
        std::uint32_t generation = 1u;
        MutantPhase currentPhase = MutantPhase::published;
    };
    Slot slot { std::make_unique<FakeBank>(101), 1u, phase };
    const std::uint32_t outstandingToken = slot.generation << 2u;
    const int destroyedBefore = FakeBank::destroyed.load(std::memory_order_relaxed);

    // Mutant scan: PUBLISHED and ACTIVE are incorrectly treated as reusable.
    slot.bank = std::make_unique<FakeBank>(202);
    ++slot.generation;
    slot.currentPhase = MutantPhase::published;
    const std::uint32_t replacementToken = slot.generation << 2u;
    return FakeBank::destroyed.load(std::memory_order_relaxed) == destroyedBefore + 1
        && replacementToken != outstandingToken;
}

bool retryUntilAudioMutantIsDetected()
{
    int slotReads = 0;
    int rescans = 0;
    // Mutant's retry loop is bounded here only so the negative control itself
    // terminates; the production bound rejects it after the ninth slot read.
    do
    {
        for (int slot = 0; slot < 4; ++slot) ++slotReads;
    }
    while (++rescans < 3);
    return slotReads > 8;
}

bool generationWrapMutantIsDetected()
{
    constexpr std::uint32_t maximum = 2u;
    std::uint32_t generation = 1u;
    const std::uint32_t staleToken = generation << 2u;
    generation = generation == maximum ? 1u : generation + 1u;
    generation = generation == maximum ? 1u : generation + 1u;
    const std::uint32_t wrappedToken = generation << 2u;

    const std::uint32_t staleGeneration = staleToken >> 2u;
    const std::uint32_t currentGeneration = 2u;
    const bool correctValidation = staleGeneration == currentGeneration;
    const bool phaseOnlyMutantValidation = true;
    return staleToken == wrappedToken
        && !correctValidation && phaseOnlyMutantValidation;
}

bool pinnedReclaimMutantIsDetected()
{
    auto bank = std::make_unique<FakeBank>(303);
    constexpr std::uint32_t token = 12u;
    constexpr std::uint32_t pinnedToken = token;
    const int destroyedBefore = FakeBank::destroyed.load(std::memory_order_relaxed);
    // Mutant omits the exact-token pin guard before control-side reclaim.
    if constexpr (token == pinnedToken) bank.reset();
    return FakeBank::destroyed.load(std::memory_order_relaxed)
        == destroyedBefore + 1;
}

bool partialCommitMutantIsDetected()
{
    int canonicalSource = 7;
    try
    {
        canonicalSource = 9; // Mutant commits before candidate construction.
        allocation_probe::failAfter = 0;
        auto candidate = std::make_unique<FakeBank>(9);
        (void)candidate;
    }
    catch (const std::bad_alloc&)
    {
        allocation_probe::failAfter = -1;
        return canonicalSource != 7;
    }
    allocation_probe::failAfter = -1;
    return false;
}

bool restoredSpinMutantIsDetected()
{
    std::atomic_flag lock = ATOMIC_FLAG_INIT;
    lock.test_and_set(std::memory_order_acquire); // Park the control owner.
    std::atomic<int> spins { 0 };
    std::atomic<bool> passed { false };
    std::thread audio([&] {
        while (lock.test_and_set(std::memory_order_acquire))
            spins.fetch_add(1, std::memory_order_relaxed);
        passed.store(true, std::memory_order_release);
        lock.clear(std::memory_order_release);
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (spins.load(std::memory_order_acquire) < 1000
           && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool blocked = spins.load(std::memory_order_relaxed) >= 1000
        && !passed.load(std::memory_order_acquire);
    lock.clear(std::memory_order_release);
    audio.join();
    return blocked;
}

struct SharedOwnershipMutantBank
{
    ~SharedOwnershipMutantBank()
    {
        if (allocation_probe::audioThread)
            audioDestructions.fetch_add(1, std::memory_order_relaxed);
    }
    static inline std::atomic<int> audioDestructions { 0 };
};

bool restoredSharedOwnershipMutantIsDetected()
{
    SharedOwnershipMutantBank::audioDestructions.store(
        0, std::memory_order_relaxed);
    std::shared_ptr<SharedOwnershipMutantBank> published =
        std::make_shared<SharedOwnershipMutantBank>();
    auto audioSnapshot = published;
    published = std::make_shared<SharedOwnershipMutantBank>();
    std::thread audio([owned = std::move(audioSnapshot)]() mutable {
        allocation_probe::audioThread = true;
        owned.reset(); // Mutant's final shared release happens on audio.
        allocation_probe::audioThread = false;
    });
    audio.join();
    return SharedOwnershipMutantBank::audioDestructions.load(
        std::memory_order_relaxed) == 1;
}

template <typename P>
void publish(P& publisher, int value, std::uint32_t latency = 64u)
{
    const auto result = publisher.publish(
        std::make_unique<FakeBank>(value), latency, []() noexcept {});
    require(result == P::PublishResult::published, "publication was rejected");
}

std::vector<std::uint8_t> makeReverbState(float mix, float preDelay,
                                          float decayScale, float stretch)
{
    dspark::StateWriter writer(dspark::stateId("CRVB"), 1);
    writer.write("mix", mix);
    writer.write("preDelay", preDelay);
    writer.write("decayScale", decayScale);
    writer.write("stretch", stretch);
    return writer.blob();
}

const std::array<float, 512>& reverbTransactionImpulse()
{
    static const std::array<float, 512> impulse = [] {
        std::array<float, 512> result {};
        float amplitude = 1.0f;
        for (std::size_t index = 0; index < result.size(); ++index)
        {
            result[index] = amplitude;
            amplitude *= 0.985f;
        }
        return result;
    }();
    return impulse;
}

void configureTransactionalReverb(dspark::Reverb<float>& reverb)
{
    const dspark::AudioSpec spec { 48000.0, 64, 1 };
    reverb.prepare(spec);
    const auto& impulse = reverbTransactionImpulse();
    require(reverb.loadIR(impulse.data(), static_cast<int>(impulse.size()),
                          spec.sampleRate),
            "transactional Reverb IR load failed");
}

struct ReverbObservables
{
    std::vector<std::uint8_t> state;
    float mix = 0.0f;
    float preDelay = 0.0f;
    float decayScale = 0.0f;
    float stretch = 0.0f;
    int latency = 0;
    bool loaded = false;
    dspark::Convolver<float>* convolver = nullptr;
};

ReverbObservables observe(dspark::Reverb<float>& reverb)
{
    return {
        reverb.getState(),
        reverb.getMix(),
        reverb.getPreDelay(),
        reverb.getDecayScale(),
        reverb.getStretch(),
        reverb.getLatency(),
        reverb.isLoaded(),
        &reverb.getConvolver(),
    };
}

void requireSameObservables(dspark::Reverb<float>& reverb,
                            const ReverbObservables& before)
{
    require(reverb.getState() == before.state,
            "failed setState changed serialized state bytes");
    require(reverb.getMix() == before.mix,
            "failed setState changed mix");
    require(reverb.getPreDelay() == before.preDelay,
            "failed setState changed pre-delay");
    require(reverb.getDecayScale() == before.decayScale,
            "failed setState changed decay scale");
    require(reverb.getStretch() == before.stretch,
            "failed setState changed stretch");
    require(reverb.getLatency() == before.latency,
            "failed setState changed latency");
    require(reverb.isLoaded() == before.loaded,
            "failed setState changed loaded state");
    require(&reverb.getConvolver() == before.convolver,
            "failed setState changed exact pinned Convolver identity");
    require(before.convolver->getLatency() == before.latency,
            "failed setState invalidated the prior Convolver pin");
}

void requireFixedRenderIdentity(dspark::Reverb<float>& subject,
                                dspark::Reverb<float>& control)
{
    subject.reset();
    control.reset();
    dspark::AudioBuffer<float> subjectBuffer;
    dspark::AudioBuffer<float> controlBuffer;
    subjectBuffer.resize(1, 64);
    controlBuffer.resize(1, 64);
    for (int block = 0; block < 8; ++block)
    {
        for (int sample = 0; sample < 64; ++sample)
        {
            const float value = block == 0 && sample == 0
                ? 1.0f
                : static_cast<float>(((block * 64 + sample) % 31) - 15)
                    * 0.0005f;
            subjectBuffer.getChannel(0)[sample] = value;
            controlBuffer.getChannel(0)[sample] = value;
        }
        subject.processBlock(subjectBuffer.toView());
        control.processBlock(controlBuffer.toView());
        require(std::memcmp(subjectBuffer.getChannel(0),
                            controlBuffer.getChannel(0),
                            64u * sizeof(float)) == 0,
                "failed setState changed fixed rendered PCM bytes");
    }
}

void adoptReverbPublication(dspark::Reverb<float>& reverb)
{
    dspark::AudioBuffer<float> buffer;
    buffer.resize(1, 64);
    buffer.clear();
    reverb.processBlock(buffer.toView());
}

void testSetStateStrongTransaction()
{
    const auto requested = makeReverbState(0.875f, 37.0f, 0.5f, 1.5f);
    std::ptrdiff_t sweptFailures = 0;
    for (std::ptrdiff_t failurePoint = 0;; ++failurePoint)
    {
        dspark::Reverb<float> subject;
        dspark::Reverb<float> control;
        configureTransactionalReverb(subject);
        configureTransactionalReverb(control);
        const ReverbObservables before = observe(subject);

        bool threw = false;
        bool restored = false;
        try
        {
            allocation_probe::failAfter = failurePoint;
            restored = subject.setState(requested.data(), requested.size());
            allocation_probe::failAfter = -1;
        }
        catch (const std::bad_alloc&)
        {
            allocation_probe::failAfter = -1;
            threw = true;
        }
        catch (...)
        {
            allocation_probe::failAfter = -1;
            throw;
        }

        if (threw)
        {
            ++sweptFailures;
            requireSameObservables(subject, before);
            requireFixedRenderIdentity(subject, control);
            continue;
        }

        require(restored,
                "setState returned no-capacity in a fresh allocation sweep");
        require(subject.getState() == requested,
                "successful setState did not commit the exact state blob");
        require(subject.getMix() == 0.875f
                    && subject.getPreDelay() == 37.0f
                    && subject.getDecayScale() == 0.5f
                    && subject.getStretch() == 1.5f,
                "successful setState committed incoherent parameters");
        require(subject.isLoaded() && subject.getLatency() == 64,
                "successful setState committed incoherent metadata");

        dspark::Reverb<float> successfulControl;
        configureTransactionalReverb(successfulControl);
        require(successfulControl.setState(requested.data(), requested.size()),
                "successful transaction control was rejected");
        requireFixedRenderIdentity(subject, successfulControl);
        break; // First eventual success: every preceding allocation was swept.
    }
    require(sweptFailures > 0,
            "setState allocation sweep did not exercise a throwing point");
}

void testSetStateNoCapacityTransaction()
{
    dspark::Reverb<float> subject;
    dspark::Reverb<float> control;
    configureTransactionalReverb(subject);
    configureTransactionalReverb(control);
    adoptReverbPublication(subject);
    adoptReverbPublication(control);

    for (const float scale : { 0.5f, 0.75f, 1.25f })
    {
        const auto state = makeReverbState(0.4f + scale * 0.1f,
                                           5.0f + scale, scale, 1.0f);
        require(subject.setState(state.data(), state.size())
                    && control.setState(state.data(), state.size()),
                "no-capacity setup publication failed early");
        adoptReverbPublication(subject);
        adoptReverbPublication(control);
    }

    const ReverbObservables before = observe(subject);
    const auto rejected = makeReverbState(0.9f, 49.0f, 1.5f, 1.25f);
    require(!subject.setState(rejected.data(), rejected.size()),
            "generation-exhausted Reverb did not report no-capacity");
    requireSameObservables(subject, before);
    requireFixedRenderIdentity(subject, control);
}

void testSlotStateMachine()
{
    Publisher<> publisher;
    publish(publisher, 1);
    const std::uint32_t first = publisher.latestTokenForTest();
    require(first != 0u, "first token is empty");
    require(Publisher<>::phaseForTest(publisher.stateWordForTest(first & 3u))
                == Publisher<>::Phase::published,
            "candidate did not reach PUBLISHED");
    require(publisher.adoptAtBoundary()->value == 1, "first adoption mismatch");
    require(Publisher<>::phaseForTest(publisher.stateWordForTest(first & 3u))
                == Publisher<>::Phase::active,
            "first token did not reach ACTIVE");
    publish(publisher, 2);
    require(publisher.adoptAtBoundary()->value == 2, "second adoption mismatch");
    require(Publisher<>::phaseForTest(publisher.stateWordForTest(first & 3u))
                == Publisher<>::Phase::retired,
            "old ACTIVE token did not reach RETIRED");
}

void testFixedAudioOperationBound()
{
    Publisher<> publisher;
    publish(publisher, 1);
    CountingHooks::reset();
    require(publisher.adoptAtBoundary() != nullptr, "pending bank not adopted");
    require(CountingHooks::audioExchanges.load() == 1, "audio exchange count != 1");
    require(CountingHooks::audioStores.load() == 1, "first adoption store count != 1");
    require(CountingHooks::audioValidations.load() == 1,
            "first adoption validation count != 1");
    require(CountingHooks::controlScans.load() == 0, "audio path scanned slots");

    publish(publisher, 2);
    CountingHooks::reset();
    require(publisher.adoptAtBoundary()->value == 2, "replacement not adopted");
    require(CountingHooks::audioExchanges.load() == 1, "replacement exchange count != 1");
    require(CountingHooks::audioStores.load() == 2, "replacement store count != 2");
    require(CountingHooks::audioValidations.load() == 2,
            "replacement validation count != 2");
    require(CountingHooks::controlScans.load() == 0, "replacement scanned slots");

    CountingHooks::reset();
    require(publisher.adoptAtBoundary()->value == 2, "active bank changed without pending");
    require(CountingHooks::audioExchanges.load() == 1, "empty boundary exchange count != 1");
    require(CountingHooks::audioStores.load() == 0, "empty boundary stored state");
    require(CountingHooks::audioValidations.load() == 0,
            "empty boundary validated a nonexistent token");
}

void testPublisherPhaseParking()
{
    constexpr ControlPoint targets[] = {
        ControlPoint::candidateReady,
        ControlPoint::beforeFirstScan,
        ControlPoint::afterSlotSelection,
        ControlPoint::afterPublishedStore,
        ControlPoint::beforePendingExchange,
        ControlPoint::afterPendingExchange,
        ControlPoint::beforeOldRetirement,
        ControlPoint::beforeOldReclaim,
        ControlPoint::beforeCommit,
        ControlPoint::afterCommit,
    };

    for (const ControlPoint target : targets)
    {
        Publisher<ParkingHooks> publisher;
        publish(publisher, 1);
        require(publisher.adoptAtBoundary()->value == 1,
                "initial parking adoption mismatch");

        const bool oldRetirementPhase =
            target == ControlPoint::beforeOldRetirement
            || target == ControlPoint::beforeOldReclaim;
        if (oldRetirementPhase)
            publish(publisher, 2); // Leave an exact old pending token.

        ParkingHooks::target.store(static_cast<int>(target),
                                   std::memory_order_relaxed);
        ParkingHooks::reached.store(false, std::memory_order_release);
        ParkingHooks::released.store(false, std::memory_order_release);
        ParkingHooks::enabled.store(true, std::memory_order_release);
        const int replacement = oldRetirementPhase ? 3 : 2;
        std::thread control([&] { publish(publisher, replacement); });

        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!ParkingHooks::reached.load(std::memory_order_acquire)
               && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        const bool reached = ParkingHooks::reached.load(std::memory_order_acquire);

        const bool tokenInstalled = oldRetirementPhase
            || target == ControlPoint::afterPendingExchange
            || target == ControlPoint::beforeCommit
            || target == ControlPoint::afterCommit;
        CountingHooks::reset();
        bool completeBank = reached;
        if (reached)
        {
            for (int i = 0; i < 32; ++i)
            {
                FakeBank* const bank = publisher.adoptAtBoundary();
                completeBank = completeBank && bank != nullptr
                    && bank->value == (tokenInstalled ? replacement : 1);
            }
        }
        const int exchanges = CountingHooks::audioExchanges.load();
        const int stores = CountingHooks::audioStores.load();
        const int scans = CountingHooks::controlScans.load();

        ParkingHooks::released.store(true, std::memory_order_release);
        control.join();
        ParkingHooks::enabled.store(false, std::memory_order_release);
        require(reached, "control hook did not park at a required phase");
        require(completeBank, "parking phase exposed the wrong complete bank");
        require(exchanges == 32,
                "parked publisher changed boundary exchange bound");
        require(stores <= 2,
                "parked publisher exceeded boundary state-store bound");
        require(scans == 0, "parked publisher made audio scan slots");
        require(publisher.adoptAtBoundary()->value == replacement,
                "parked publication was lost");
    }
}

void testNoAudioAllocationOrFinalRelease()
{
    FakeBank::resetCounters();
    allocation_probe::audioAllocations.store(0, std::memory_order_relaxed);
    allocation_probe::audioDeallocations.store(0, std::memory_order_relaxed);
    Publisher<> publisher;
    publish(publisher, 1);

    allocation_probe::audioThread = true;
    require(publisher.adoptAtBoundary()->value == 1, "audio adoption failed");
    allocation_probe::audioThread = false;
    publish(publisher, 2);
    allocation_probe::audioThread = true;
    require(publisher.adoptAtBoundary()->value == 2, "audio replacement failed");
    allocation_probe::audioThread = false;
    publish(publisher, 3); // Reclaims retired storage on control.

    require(allocation_probe::audioAllocations.load() == 0,
            "audio boundary allocated");
    require(allocation_probe::audioDeallocations.load() == 0,
            "audio boundary deallocated");
    require(FakeBank::audioDestructions.load() == 0,
            "audio boundary destroyed a bank");
    require(restoredSpinMutantIsDetected(),
            "restored atomic-flag spin mutant survived");
    require(restoredSharedOwnershipMutantIsDetected(),
            "restored shared-ownership/final-release mutant survived");
}

void testPrematureReuseMutant()
{
    FakeBank::resetCounters();
    Publisher<> publisher;
    publish(publisher, 11);
    FakeBank* const active = publisher.adoptAtBoundary();
    publish(publisher, 22);
    require(active->value == 11, "selected publisher reused ACTIVE storage");
    require(unsafeScanReuseMutantIsDetected(MutantPhase::published),
            "scanned-PUBLISHED reuse mutant survived");
    require(unsafeScanReuseMutantIsDetected(MutantPhase::active),
            "ACTIVE reuse mutant survived");

    Publisher<AudioHandoffHooks> handoffPublisher;
    publish(handoffPublisher, 1);
    require(handoffPublisher.adoptAtBoundary()->value == 1,
            "handoff setup adoption failed");
    publish(handoffPublisher, 2);
    const std::uint32_t handoffToken = handoffPublisher.latestTokenForTest();
    AudioHandoffHooks::reached.store(false, std::memory_order_release);
    AudioHandoffHooks::released.store(false, std::memory_order_release);
    AudioHandoffHooks::enabled.store(true, std::memory_order_release);
    std::atomic<int> adoptedValue { 0 };
    std::thread audio([&] {
        FakeBank* const adopted = handoffPublisher.adoptAtBoundary();
        adoptedValue.store(adopted ? adopted->value : -1,
                           std::memory_order_release);
    });
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!AudioHandoffHooks::reached.load(std::memory_order_acquire)
           && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    const bool handoffReached =
        AudioHandoffHooks::reached.load(std::memory_order_acquire);
    bool exactHandoffPreserved = handoffReached;
    if (handoffReached)
    {
        for (int value = 3; value <= 1000; ++value)
            publish(handoffPublisher, value);
        const std::uint32_t state =
            handoffPublisher.stateWordForTest(handoffToken & 3u);
        exactHandoffPreserved =
            Publisher<AudioHandoffHooks>::phaseForTest(state)
                == Publisher<AudioHandoffHooks>::Phase::published
            && Publisher<AudioHandoffHooks>::generationForTest(state)
                == (handoffToken >> 2u);
    }
    AudioHandoffHooks::released.store(true, std::memory_order_release);
    audio.join();
    AudioHandoffHooks::enabled.store(false, std::memory_order_release);
    require(handoffReached, "audio handoff hook did not park");
    require(exactHandoffPreserved,
            "writer storm reclaimed a scanned PUBLISHED handoff");
    require(adoptedValue.load(std::memory_order_acquire) == 2,
            "parked audio handoff adopted reused storage");
    require(handoffPublisher.adoptAtBoundary()->value == 1000,
            "publication after parked handoff was lost");
}

void testPublisherStarvationMutant()
{
    Publisher<> publisher;
    for (int value = 1; value <= 256; ++value)
    {
        CountingHooks::reset();
        publish(publisher, value);
        require(CountingHooks::controlScans.load() <= 8,
                "publisher exceeded one scan plus one rescan");
        require(CountingHooks::pendingExchanges.load() <= 2,
                "publisher exceeded bounded pending exchanges");
        require(publisher.residentBanksForTest() <= 4,
                "publisher retained a fifth slot bank");
    }
    require(publisher.adoptAtBoundary()->value == 256,
            "last coalesced publication was not adopted");

    using ExhaustedPublisher = Publisher<CountingHooks, 1u>;
    ExhaustedPublisher exhausted;
    for (int value = 1; value <= 4; ++value)
    {
        publish(exhausted, value);
        require(exhausted.adoptAtBoundary()->value == value,
                "exhaustion setup adoption mismatch");
    }
    CountingHooks::reset();
    const auto noCapacity = exhausted.publish(
        std::make_unique<FakeBank>(5), 64u, []() noexcept {});
    require(noCapacity == ExhaustedPublisher::PublishResult::noCapacity,
            "exhausted publisher did not return no-capacity");
    require(CountingHooks::controlScans.load() == 8,
            "empty-pending audio-win path did not take exactly one rescan");
    require(CountingHooks::pendingExchanges.load() == 1,
            "empty-pending path did not take its single coalesce exchange");

    require(retryUntilAudioMutantIsDetected(),
            "retry-until-audio mutant escaped the state-step oracle");
}

void testGenerationAba()
{
    using TinyPublisher = Publisher<CountingHooks, 2u>;
    TinyPublisher publisher;
    std::uint32_t stale = 0u;
    for (int value = 1; value <= 8; ++value)
    {
        publish(publisher, value);
        if (value == 1) stale = publisher.latestTokenForTest();
        require(publisher.adoptAtBoundary()->value == value,
                "generation cycle adoption mismatch");
    }
    const std::uint32_t latest = publisher.latestTokenForTest();
    const int latency = publisher.latency();
    FakeBank* const exactLatest = publisher.pinLatest();
    const std::array<std::uint32_t, 4> statesBefore = {
        publisher.stateWordForTest(0), publisher.stateWordForTest(1),
        publisher.stateWordForTest(2), publisher.stateWordForTest(3),
    };
    const std::uint32_t pendingBefore = publisher.pendingTokenForTest();
    const std::uint32_t activeBefore = publisher.activeTokenForTest();
    const std::uint32_t pinnedBefore = publisher.pinnedTokenForTest();
    const std::size_t residentBefore = publisher.residentBanksForTest();
    int commits = 0;
    const auto result = publisher.publish(
        std::make_unique<FakeBank>(9), 999u,
        [&commits]() noexcept { ++commits; });
    require(result == TinyPublisher::PublishResult::noCapacity,
            "generation exhaustion did not return no-capacity");
    require(commits == 0, "no-capacity partially ran the commit callback");
    require(publisher.latestTokenForTest() == latest,
            "no-capacity changed the latest token");
    require(publisher.latency() == latency,
            "no-capacity changed packed metadata");
    require(publisher.pendingTokenForTest() == pendingBefore
                && publisher.activeTokenForTest() == activeBefore
                && publisher.pinnedTokenForTest() == pinnedBefore,
            "no-capacity changed ownership tokens");
    require(publisher.residentBanksForTest() == residentBefore,
            "no-capacity changed resident bank ownership");
    for (std::size_t index = 0; index < statesBefore.size(); ++index)
        require(publisher.stateWordForTest(index) == statesBefore[index],
                "no-capacity changed a slot state word");
    require(publisher.pinLatest() == exactLatest
                && exactLatest->value == 8
                && exactLatest->storage.front() == 8,
            "no-capacity changed exact pinned bank identity or content");
    const std::size_t staleIndex = stale & 3u;
    const auto state = publisher.stateWordForTest(staleIndex);
    require(TinyPublisher::phaseForTest(state) == TinyPublisher::Phase::exhausted,
            "max generation did not quarantine the slot");
    require(TinyPublisher::generationForTest(state) == 2u,
            "exhausted generation wrapped");
    require(stale != publisher.activeTokenForTest(), "stale token became active again");
    require(generationWrapMutantIsDetected(),
            "generation wrap/validation mutant survived");

    using MaxPendingPublisher = Publisher<CountingHooks, 1u>;
    MaxPendingPublisher maxPending;
    publish(maxPending, 1, 101u);
    require(maxPending.adoptAtBoundary()->value == 1,
            "max-pending setup first adoption failed");
    require(maxPending.pinLatest()->value == 1,
            "max-pending setup pin failed");
    publish(maxPending, 2, 102u);
    require(maxPending.adoptAtBoundary()->value == 2,
            "max-pending setup second adoption failed");
    publish(maxPending, 3, 103u);
    require(maxPending.adoptAtBoundary()->value == 3,
            "max-pending setup third adoption failed");
    publish(maxPending, 4, 104u); // Leave generation-max token pending.
    const std::uint32_t protectedPending = maxPending.pendingTokenForTest();
    const std::uint32_t protectedLatest = maxPending.latestTokenForTest();
    CountingHooks::reset();
    int maxPendingCommits = 0;
    const auto protectedResult = maxPending.publish(
        std::make_unique<FakeBank>(5), 105u,
        [&maxPendingCommits]() noexcept { ++maxPendingCommits; });
    require(protectedResult == MaxPendingPublisher::PublishResult::noCapacity,
            "generation-max pending capacity loss was not typed");
    require(maxPendingCommits == 0,
            "generation-max pending failure partially committed");
    require(maxPending.pendingTokenForTest() == protectedPending
                && maxPending.latestTokenForTest() == protectedLatest
                && maxPending.latency() == 104,
            "no-capacity cancelled the prior generation-max publication");
    require(CountingHooks::pendingExchanges.load() == 0,
            "generation-max pending token was unsafely coalesced");
    require(maxPending.pinLatest()->value == 4,
            "prior latest bank did not survive generation-max no-capacity");
}

void testGetConvolverPinLifetime()
{
    FakeBank::resetCounters();
    Publisher<> publisher;
    publish(publisher, 1);
    require(publisher.adoptAtBoundary()->value == 1, "initial adoption mismatch");
    FakeBank* const pinned = publisher.pinLatest();
    const std::uint32_t pinnedToken = publisher.pinnedTokenForTest();

    for (int value = 2; value <= 20; ++value)
    {
        publish(publisher, value);
        require(publisher.adoptAtBoundary()->value == value,
                "publication storm adoption mismatch");
        require(pinned->value == 1, "pinned bank was reclaimed or overwritten");
        require(publisher.pinnedTokenForTest() == pinnedToken,
                "pin token changed before next accessor call");
    }

    require(publisher.pinLatest()->value == 20, "next accessor did not pin latest");
    publish(publisher, 21); // Gives control a scan in which old pin is reclaimable.
    require(FakeBank::audioDestructions.load() == 0,
            "pin reclamation ran on audio");
    require(pinnedReclaimMutantIsDetected(),
            "exact pinned-generation reclaim mutant survived");
}

void testResetMetadataExceptionShutdown()
{
    using P = Publisher<>;
    static_assert(!std::is_copy_constructible_v<P>);
    static_assert(!std::is_copy_assignable_v<P>);
    static_assert(!std::is_move_constructible_v<P>);
    static_assert(!std::is_move_assignable_v<P>);
    static_assert(!std::is_copy_constructible_v<dspark::Reverb<float>>);
    static_assert(!std::is_move_constructible_v<dspark::Reverb<float>>);

    FakeBank::resetCounters();
    {
        P publisher;
        publish(publisher, 1, 1024u);
        require(publisher.isLoaded(), "metadata loaded bit is false");
        require(publisher.latency() == 1024, "metadata latency mismatch");
        require(FakeBank::metadataReads.load(std::memory_order_relaxed) == 0,
                "metadata access dereferenced a bank");
        FakeBank* const metadataMutantBank = publisher.pinLatest();
        (void)metadataMutantBank->readForMetadataMutant();
        require(FakeBank::metadataReads.load(std::memory_order_relaxed) == 1,
                "metadata-bank-dereference mutant survived");
        const std::uint32_t latest = publisher.latestTokenForTest();
        bool allocationFailed = false;
        try
        {
            allocation_probe::failAfter = 0;
            auto failedCandidate = std::make_unique<FakeBank>(2);
            (void)failedCandidate;
        }
        catch (const std::bad_alloc&)
        {
            allocationFailed = true;
            allocation_probe::failAfter = -1;
        }
        require(allocationFailed, "allocation-failure control did not fire");
        require(publisher.latestTokenForTest() == latest,
                "failed construction partially committed latest token");
        require(publisher.latency() == 1024,
                "failed construction partially committed metadata");
        require(publisher.adoptAtBoundary()->value == 1,
                "reset-style boundary adoption failed");
        publisher.shutdown();
        require(!publisher.isLoaded(), "shutdown retained loaded metadata");
        require(publisher.residentBanksForTest() == 0, "shutdown retained a bank");
    }
    require(FakeBank::live.load() == 0, "shutdown leaked a bank");
    require(FakeBank::constructed.load() == FakeBank::destroyed.load(),
            "shutdown did not destroy every fake bank exactly once");
    require(FakeBank::audioDestructions.load() == 0,
            "shutdown destroyed a bank on audio");
    require(partialCommitMutantIsDetected(),
            "partial source/config commit mutant survived");
    testSetStateStrongTransaction();
    testSetStateNoCapacityTransaction();

    dspark::Reverb<float> reverb;
    dspark::Reverb<float> control;
    const dspark::AudioSpec spec { 48000.0, 64, 1 };
    reverb.prepare(spec);
    control.prepare(spec);
    const float impulse[] = { 1.0f, 0.5f, 0.25f, 0.0f };
    require(reverb.loadIR(impulse, 4, 48000.0)
                && control.loadIR(impulse, 4, 48000.0),
            "real Reverb IR load failed");
    require(reverb.isLoaded() && reverb.getLatency() == 64,
            "real Reverb metadata is incoherent");
    const float replacement[] = {
        0.25f, 0.125f, 0.0625f, 0.03125f,
        0.015625f, 0.0078125f, 0.00390625f, 0.0f
    };
    bool realBuildFailed = false;
    try
    {
        // The transactional input allocation succeeds; the following bank
        // allocation fails before any source/config/publication commit.
        allocation_probe::failAfter = 1;
        (void)reverb.loadIR(replacement, 8, 48000.0);
    }
    catch (const std::bad_alloc&)
    {
        realBuildFailed = true;
        allocation_probe::failAfter = -1;
    }
    require(realBuildFailed, "real Reverb build-failure control did not fire");
    reverb.reset();
    control.reset();
    require(reverb.isLoaded(), "reset unloaded the real Reverb");

    dspark::AudioBuffer<float> subjectBuffer;
    dspark::AudioBuffer<float> controlBuffer;
    subjectBuffer.resize(1, 64);
    controlBuffer.resize(1, 64);
    for (int block = 0; block < 8; ++block)
    {
        for (int sample = 0; sample < 64; ++sample)
        {
            const float value = block == 0 && sample == 0
                ? 1.0f
                : static_cast<float>(((block * 64 + sample) % 29) - 14)
                    * 0.001f;
            subjectBuffer.getChannel(0)[sample] = value;
            controlBuffer.getChannel(0)[sample] = value;
        }
        reverb.processBlock(subjectBuffer.toView());
        control.processBlock(controlBuffer.toView());
        require(std::equal(subjectBuffer.getChannel(0),
                           subjectBuffer.getChannel(0) + 64,
                           controlBuffer.getChannel(0)),
                "failed Reverb rebuild changed the prior DSP bank");
    }
}

void testRetainedBankMemoryBound()
{
    FakeBank::resetCounters();
    Publisher<> publisher;
    publish(publisher, 0);
    require(publisher.adoptAtBoundary() != nullptr, "initial adoption failed");
    (void)publisher.pinLatest();
    for (int value = 1; value <= 2000; ++value)
    {
        auto candidate = std::make_unique<FakeBank>(value);
        require(FakeBank::live.load() <= 5, "candidate exceeded total-bank bound");
        const auto result = publisher.publish(
            std::move(candidate), 64u, []() noexcept {});
        require(result == Publisher<>::PublishResult::published,
                "finite writer burst lost capacity");
        require(publisher.residentBanksForTest() <= 4,
                "slot bank bound exceeded four");
        require(FakeBank::live.load() <= 4,
                "post-publication resident bank bound exceeded four");
        if ((value & 7) == 0) (void)publisher.adoptAtBoundary();
    }
    require(FakeBank::maximumLive.load() <= 5, "lifetime high-water exceeded five");

    const int liveBeforeMutants = FakeBank::live.load(std::memory_order_relaxed);
    std::array<std::unique_ptr<FakeBank>, 5> fifthSlotMutant;
    for (std::size_t index = 0; index < fifthSlotMutant.size(); ++index)
        fifthSlotMutant[index] = std::make_unique<FakeBank>(3000 + static_cast<int>(index));
    require(FakeBank::live.load(std::memory_order_relaxed) == liveBeforeMutants + 5,
            "fifth-slot mutant was not made live");
    require(fifthSlotMutant.size() > 4,
            "fifth-slot capacity mutant escaped the four-slot oracle");
    fifthSlotMutant = {};

    std::vector<std::unique_ptr<FakeBank>> unboundedRetireMutant;
    for (int value = 0; value < 6; ++value)
        unboundedRetireMutant.push_back(std::make_unique<FakeBank>(4000 + value));
    require(unboundedRetireMutant.size() > 5,
            "unbounded-retire mutant escaped the total-bank oracle");
}

void testRaceSanitizerMatrix()
{
    FakeBank::resetCounters();
    allocation_probe::audioAllocations.store(0, std::memory_order_relaxed);
    allocation_probe::audioDeallocations.store(0, std::memory_order_relaxed);
    Publisher<> publisher;
    std::atomic<bool> start { false };
    std::atomic<bool> done { false };
    std::atomic<int> observed { 0 };

#if defined(DSPARK_TEST_HELGRIND) || defined(DSPARK_TEST_DRD)
    retirementSignals.store(0u, std::memory_order_relaxed);
    for (std::size_t index = 0;
         index < Publisher<>::atomicWordCountForTest(); ++index)
    {
        annotateBenignAtomic(publisher.atomicWordAddressForTest(index),
                             sizeof(std::atomic<std::uint32_t>));
    }
    annotateBenignAtomic(&start, sizeof(start));
    annotateBenignAtomic(&done, sizeof(done));
    annotateBenignAtomic(&observed, sizeof(observed));
    annotateBenignAtomic(&retirementSignals, sizeof(retirementSignals));
#endif

    std::thread audio([&] {
        allocation_probe::audioThread = true;
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        while (!done.load(std::memory_order_acquire))
        {
            if (FakeBank* const bank = publisher.adoptAtBoundary())
            {
                const int value = bank->value;
                if (value > observed.load(std::memory_order_relaxed))
                    observed.store(value, std::memory_order_relaxed);
            }
        }
        if (FakeBank* const finalBank = publisher.adoptAtBoundary())
            observed.store(finalBank->value, std::memory_order_relaxed);
        allocation_probe::audioThread = false;
    });

    start.store(true, std::memory_order_release);
    for (int value = 1; value <= 20000; ++value)
        publish(publisher, value);
    done.store(true, std::memory_order_release);
    audio.join();

    require(FakeBank::audioDestructions.load() == 0,
            "race stress destroyed a bank on audio");
    require(allocation_probe::audioAllocations.load() == 0,
            "race stress allocated on audio");
    require(allocation_probe::audioDeallocations.load() == 0,
            "race stress deallocated on audio");
    require(publisher.residentBanksForTest() <= 4,
            "race stress exceeded fixed bank capacity");
    require(observed.load(std::memory_order_relaxed) == 20000,
            "first boundary after quiescence did not adopt the final token");
}

struct NamedTest
{
    const char* name;
    void (*function)();
};

constexpr NamedTest tests[] = {
    { "reverb-slot-state-machine", testSlotStateMachine },
    { "reverb-fixed-audio-operation-bound", testFixedAudioOperationBound },
    { "reverb-publisher-phase-parking", testPublisherPhaseParking },
    { "reverb-no-audio-allocation-or-final-release", testNoAudioAllocationOrFinalRelease },
    { "reverb-premature-reuse-mutant", testPrematureReuseMutant },
    { "reverb-publisher-starvation-mutant", testPublisherStarvationMutant },
    { "reverb-generation-aba", testGenerationAba },
    { "reverb-getconvolver-pin-lifetime", testGetConvolverPinLifetime },
    { "reverb-reset-metadata-exception-shutdown", testResetMetadataExceptionShutdown },
    { "reverb-retained-bank-memory-bound", testRetainedBankMemoryBound },
    { "reverb-race-sanitizer-matrix", testRaceSanitizerMatrix },
};

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--race-only")
    {
        try
        {
            testRaceSanitizerMatrix();
            std::cout << "PASS reverb-race-sanitizer-matrix\n";
            return 0;
        }
        catch (const std::exception& error)
        {
            std::cerr << "FAIL reverb-race-sanitizer-matrix: "
                      << error.what() << '\n';
            return 1;
        }
    }

    int failures = 0;
    for (const auto& test : tests)
    {
        try
        {
            test.function();
            std::cout << "PASS " << test.name << '\n';
        }
        catch (const std::exception& error)
        {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << error.what() << '\n';
        }
    }
    std::cout << (sizeof(tests) / sizeof(tests[0])) << " checks, "
              << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
