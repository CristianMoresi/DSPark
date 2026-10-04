// DSPark Tests - allocation-failure injection
//
// A SEPARATE EXECUTABLE ON PURPOSE. This translation unit defines replacement
// operator new / operator delete. Replacement allocation functions are
// program-wide: they serve every translation unit linked into the same binary,
// not just this one. Because they route every request to std::malloc and every
// release to std::free, AddressSanitizer sees nothing but matched malloc/free
// pairs and its alloc-dealloc-mismatch / new-delete-type-mismatch checks never
// get to run -- for EVERY case in that binary. Linked into the main suite they
// switched that check class off for the whole suite; here their blast radius is
// this file, and the suite keeps the full set of sanitizer checks.
//
// So: do not move the injector back into a shared translation unit, and do not
// add cases here unless they need it. The target is dspark_alloc_failure and
// CTest runs it as "alloc_failure", so it is covered by a plain ctest run.

#include "dspark_test.h"

#include "../Analysis/SpectrumAnalyzer.h"
#include "../Analysis/LoudnessMeter.h"
#include "../Analysis/AudioIntervalAnalyzer.h"
#include "../Analysis/OnsetDetector.h"
#include "../Analysis/OfflineTransientAnalyzer.h"
#include "../Analysis/OfflineTempoAnalyzer.h"
#include "../Core/FIRFilter.h"
#include "../Core/detail/OfflineProduct.h"
#include "../Core/Resampler.h"
#include "../Effects/OfflineLeveler.h"
#include "../Effects/OfflinePeakCompressor.h"
#include "../Effects/OfflinePunch.h"
#include "../Effects/OfflineBeatCompressor.h"
#include "../Effects/OfflineHardClipper.h"
#include "../Effects/OfflineSoftClipper.h"
#include "../Effects/StereoGenerator.h"
#include "../Effects/OfflineStereoGenerator.h"
#include "../Effects/OfflineStereoBalance.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>
#ifdef _MSC_VER
#include <malloc.h>
#endif

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// Pin for exception-safe SpectrumAnalyzer::prepare(). An earlier version of
// that header updated numBins_ BEFORE the allocations, so a bad_alloc
// mid-prepare left NEW sizes over OLD storage: the copy-out getters then
// OOB-wrote the old snapshot vectors and OOB-read the old slot arrays,
// reset() OOB-wrote the slots, and a throwing FIRST prepare left null slot
// arrays behind numBins_ = 129 (SEGV). All four were reproduced under
// AddressSanitizer. prepare() now documents commit-after-success: every
// allocation completes into locals before ANY member is written. This
// pin provokes a REAL bad_alloc at EVERY allocation index of a re-prepare
// and of a first prepare (fail-once throwing replacement operator new) and
// asserts the documented post-throw state plus full recovery on the next
// successful prepare(). Reverting the commit-after-success discipline turns
// every assertion below red.
//
// The replacement operator new/delete below are program-wide by linkage
// (replaceable allocation functions, [new.delete]) -- which is exactly why
// this file is its own executable; see the file header. They are strictly
// pass-through unless armed, and they are armed ONLY inside the
// single-threaded pin below, between two calls on the same thread with no
// other thread alive, so no other case can ever observe an injected failure.
//
// Sanitizer discipline (the TestCoreFoundation precedent): the AudioBuffer
// allocation-failure case steps aside under ASan/TSan because it needs the
// sanitizer ALLOCATOR itself to refuse a huge request, which sanitizer
// allocators answer with a fatal report instead of bad_alloc. This pin is
// different by construction: the bad_alloc is thrown by OUR replacement
// BEFORE any allocator is asked, every request actually forwarded is tiny
// (< 128 KiB, far below any sanitizer ceiling), and the forwarded std::malloc
// stays intercepted (redzoned/tracked) by the sanitizer. Verified, not
// assumed: under ASan+UBSan this target is green with this case injecting
// real bad_allocs, and the executable's own _Znwm/_ZdlPv definitions preempt
// the sanitizer runtime's (ELF global-scope precedence) in the ASan AND the
// TSan link, confirmed from the linked binaries' symbol tables. If the
// replacement ever failed to take effect the case would fail loudly on
// EXPECT_GT(injected, 10) with injected == 0, never pass silently. The target
// remains runnable under every sanitizer. CI runs this synchronous exhaustive
// sweep in every native job and both ASan/UBSan jobs; the TSan job checks its
// local include graph for thread facilities before omitting this one executable.
namespace dspark_test_failing_alloc {
// constinit, so "no dynamic initialization" is compiler-checked: these words
// are touched by allocations made during the static init of other TUs, which
// may run before this TU's. Atomic words, so an allocation on any live thread
// reads them race-free (the disarmed fast path is one relaxed load).
constinit std::atomic<int> failAt{ -1 }; // -1 = disarmed; N = Nth new throws
constinit std::atomic<int> count{ 0 };   // counts operator new calls if armed
constinit std::atomic<std::size_t> bytes{ 0 }; // requested ordinary and aligned payload
} // namespace dspark_test_failing_alloc

static void dsparkTestBeforeAllocation(std::size_t size)
{
    namespace fa = dspark_test_failing_alloc;
    if (fa::failAt.load(std::memory_order_relaxed) > 0)
    {
        fa::bytes.fetch_add(size, std::memory_order_relaxed);
        const int n = fa::count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n == fa::failAt.load(std::memory_order_relaxed))
        {
            fa::failAt.store(-1, std::memory_order_relaxed); // fail once, disarm
            throw std::bad_alloc();
        }
    }
}

static void* dsparkTestCountedAlloc(std::size_t size)
{
    dsparkTestBeforeAllocation(size);
    if (void* p = std::malloc(size != 0 ? size : 1))
        return p;
    throw std::bad_alloc();
}

void* operator new(std::size_t size) { return dsparkTestCountedAlloc(size); }
void* operator new[](std::size_t size) { return dsparkTestCountedAlloc(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// Pool resources may request over-aligned chunks even for scalar vectors.
// Count and fail those requests too, retaining the matching aligned release.
void* operator new(std::size_t size, std::align_val_t alignment)
{
    dsparkTestBeforeAllocation(size);
    const auto a = static_cast<std::size_t>(alignment);
#ifdef _MSC_VER
    void* p = _aligned_malloc(size ? size : 1, a);
#else
    const auto n = std::max(std::size_t(1), size);
    if (n > std::numeric_limits<std::size_t>::max() - (a - 1))
        throw std::bad_alloc();
    void* p = std::aligned_alloc(a, (n + a - 1) / a * a);
#endif
    if (p)
        return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t a) { return ::operator new(size, a); }
void operator delete(void* p, std::align_val_t) noexcept
{
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }

DSPARK_TEST(SpectrumAnalyzer_recovers_after_failed_prepare_allocation)
{
    namespace fa = dspark_test_failing_alloc;
    constexpr float kFloor = -100.0f; // default floorDb_
    std::vector<float> hop(128, 0.5f);
    std::vector<float> hop2(256, 0.25f);

    // Part 1: bad_alloc at EVERY allocation index of a RE-prepare
    // (256 -> 16384, the archived probe schedule: indices {1,5,7,13} and all
    // the others). Documented contract: analyser left unprepared (pushSamples
    // no-op) with every reader-visible value exactly as before the call.
    int injected = 0;
    bool sweepDone = false;
    for (int failIdx = 1; failIdx <= 200 && !sweepDone; ++failIdx)
    {
        auto sa = std::make_unique<SpectrumAnalyzer<float>>();
        sa->prepare(48000.0, 256); // 129 bins, hop 128
        sa->setPeakHoldEnabled(true);
        sa->pushSamples(hop.data(), 128);             // publish one real frame
        const float* heldMag  = sa->getMagnitudesDb(); // adopt + snapshot
        const float* heldPeak = sa->getPeakHoldDb();
        EXPECT_EQ(sa->getNumBins(), 129);
        std::vector<float> magBefore(heldMag, heldMag + 129);
        std::vector<float> peakBefore(heldPeak, heldPeak + 129);
        (void)sa->isNewDataReady();                    // consume the flag

        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(failIdx, std::memory_order_relaxed); // arm
        bool threw = false;
        try { sa->prepare(48000.0, 16384); }           // 8193-bin target
        catch (const std::bad_alloc&) { threw = true; }
        fa::failAt.store(-1, std::memory_order_relaxed);      // disarm

        if (!threw)
        {
            // failIdx walked past prepare()'s last allocation, so the
            // un-injected re-prepare must have fully succeeded: end of sweep.
            EXPECT_EQ(sa->getNumBins(), 8193);
            EXPECT_EQ(sa->getFFTSize(), 16384);
            sweepDone = true;
            break;
        }
        ++injected;

        // (a) sizes still the PREVIOUS configuration, mutually consistent.
        EXPECT_EQ(sa->getNumBins(), 129);  // pre-fix: 8193 over 129-float storage
        EXPECT_EQ(sa->getFFTSize(), 256);
        // (b) held snapshot pointers untouched by the throwing prepare().
        int changed = 0;
        for (int k = 0; k < 129; ++k)
        {
            if (heldMag[k]  != magBefore[static_cast<size_t>(k)])  ++changed;
            if (heldPeak[k] != peakBefore[static_cast<size_t>(k)]) ++changed;
        }
        EXPECT_EQ(changed, 0);
        // (c) getters re-read memory-safely and reproduce the previous frame
        //     (pre-fix: OOB WRITE into the 516-byte old snapshots).
        const float* m = sa->getMagnitudesDb();
        const float* p = sa->getPeakHoldDb();
        int mismatched = 0;
        for (int k = 0; k < 129; ++k)
        {
            if (m[k] != magBefore[static_cast<size_t>(k)])  ++mismatched;
            if (p[k] != peakBefore[static_cast<size_t>(k)]) ++mismatched;
        }
        EXPECT_EQ(mismatched, 0);
        // (d) pushSamples is the documented no-op: a full hop produces no frame.
        sa->pushSamples(hop.data(), 128);
        EXPECT_FALSE(sa->isNewDataReady());
        // (e) reset() stays inside the real storage (pre-fix: OOB WRITE over
        //     the old slot arrays) and rewrites the snapshots behind the held
        //     pointer (the documented setup-only exception to snapshot
        //     stability).
        sa->reset();
        EXPECT_NEAR(heldMag[5], kFloor, 0.0f);
        EXPECT_NEAR(sa->getMagnitudesDb()[0], kFloor, 0.0f);
        EXPECT_NEAR(sa->getPeakHoldDb()[128], kFloor, 0.0f);
        // (f) the next successful prepare() fully recovers the analyser
        //     (held pointers are now invalidated and not touched again).
        sa->prepare(48000.0, 512); // 257 bins, hop 256
        EXPECT_EQ(sa->getNumBins(), 257);
        sa->pushSamples(hop2.data(), 256);
        EXPECT_TRUE(sa->isNewDataReady());
        EXPECT_TRUE(std::isfinite(sa->getMagnitudesDb()[10]));
    }
    EXPECT_TRUE(sweepDone);   // the sweep must terminate by SUCCESS, loudly
    EXPECT_GT(injected, 10);  // and must have really injected many failures
                              // (14 storage allocations + FFT internals)

    // Part 2: bad_alloc during the FIRST prepare (slot arrays still null).
    // Documented contract: never-prepared state -- size getters report 0 (no
    // readable bins), getters/reset()/pushSamples are memory-safe no-ops.
    injected = 0;
    sweepDone = false;
    for (int failIdx = 1; failIdx <= 200 && !sweepDone; ++failIdx)
    {
        auto sa = std::make_unique<SpectrumAnalyzer<float>>();
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(failIdx, std::memory_order_relaxed); // arm
        bool threw = false;
        try { sa->prepare(48000.0, 256); }
        catch (const std::bad_alloc&) { threw = true; }
        fa::failAt.store(-1, std::memory_order_relaxed);      // disarm

        if (!threw)
        {
            EXPECT_EQ(sa->getNumBins(), 129);
            sweepDone = true;
            break;
        }
        ++injected;

        EXPECT_EQ(sa->getNumBins(), 0);  // pre-fix: 129 over null slot arrays
        EXPECT_EQ(sa->getFFTSize(), 0);
        EXPECT_NEAR(sa->binToFrequency(10), 0.0f, 0.0f);
        // Zero readable bins: calling the getters is legal, dereferencing is
        // not (getNumBins() == 0). Pre-fix this SEGVed on the null slots.
        (void)sa->getMagnitudesDb();
        (void)sa->getPeakHoldDb();
        sa->reset();                        // safe no-op on empty storage
        sa->pushSamples(hop.data(), 128);   // documented no-op
        EXPECT_FALSE(sa->isNewDataReady());
        // Recovery: the analyser is still usable after the failed first try.
        sa->prepare(48000.0, 256);
        EXPECT_EQ(sa->getNumBins(), 129);
        sa->pushSamples(hop.data(), 128);
        EXPECT_TRUE(sa->isNewDataReady());
        EXPECT_TRUE(std::isfinite(sa->getMagnitudesDb()[64]));
    }
    EXPECT_TRUE(sweepDone);
    EXPECT_GT(injected, 10);
}

namespace {

constexpr int kResamplerInputSamples = 64;
constexpr int kResamplerOutputCapacity = 256;

using ResamplerInput = std::array<float, kResamplerInputSamples>;

ResamplerInput makeResamplerInput(unsigned seed)
{
    ResamplerInput input {};
    for (auto& sample : input)
    {
        seed = seed * 1664525u + 1013904223u;
        sample = static_cast<float>(seed >> 8) / 8388608.0f - 1.0f;
    }
    return input;
}

struct MonoRender
{
    std::array<float, kResamplerOutputCapacity> samples {};
    int produced = 0;
};

MonoRender renderMonoBlock(Resampler<float>& resampler,
                           const ResamplerInput& input) noexcept
{
    MonoRender rendered;
    rendered.produced = resampler.processBlock(
        input.data(), static_cast<int>(input.size()), rendered.samples.data());
    return rendered;
}

template <size_t Channels>
using MultiInput = std::array<ResamplerInput, Channels>;

template <size_t Channels>
MultiInput<Channels> makeMultiInput(unsigned seed)
{
    MultiInput<Channels> input {};
    for (auto& channel : input)
    {
        channel = makeResamplerInput(seed);
        seed += 0x9e3779b9u;
    }
    return input;
}

template <size_t Channels>
struct MultiRender
{
    std::array<std::array<float, kResamplerOutputCapacity>, Channels> samples {};
    int produced = 0;
};

template <size_t Channels>
MultiRender<Channels> renderMultiBlock(
    Resampler<float>& resampler, MultiInput<Channels>& input) noexcept
{
    MultiRender<Channels> rendered;
    std::array<float*, Channels> inputPointers {};
    std::array<float*, Channels> outputPointers {};
    for (size_t channel = 0; channel < Channels; ++channel)
    {
        inputPointers[channel] = input[channel].data();
        outputPointers[channel] = rendered.samples[channel].data();
    }
    AudioBufferView<float> inputView(
        inputPointers.data(), static_cast<int>(Channels), kResamplerInputSamples);
    AudioBufferView<float> outputView(
        outputPointers.data(), static_cast<int>(Channels),
        kResamplerOutputCapacity);
    rendered.produced = resampler.processBlock(inputView, outputView);
    return rendered;
}

bool sameRender(const MonoRender& a, const MonoRender& b) noexcept
{
    return a.produced == b.produced && a.samples == b.samples;
}

template <size_t Channels>
bool sameRender(const MultiRender<Channels>& a,
                const MultiRender<Channels>& b) noexcept
{
    return a.produced == b.produced && a.samples == b.samples;
}

template <typename Function>
bool injectResamplerAllocationFailure(int failIndex, Function&& function,
                                      int& observedAllocations)
{
    namespace fa = dspark_test_failing_alloc;
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(failIndex, std::memory_order_relaxed);
    bool threwBadAlloc = false;
    try
    {
        std::forward<Function>(function)();
    }
    catch (const std::bad_alloc&)
    {
        threwBadAlloc = true;
    }
    catch (...)
    {
        fa::failAt.store(-1, std::memory_order_relaxed);
        throw;
    }
    fa::failAt.store(-1, std::memory_order_relaxed);
    observedAllocations = fa::count.load(std::memory_order_relaxed);
    return threwBadAlloc;
}

} // namespace

DSPARK_TEST(Resampler_prepare_is_transactional_and_reset_does_not_allocate)
{
    using R = Resampler<float>;
    using Q = R::Quality;
    namespace fa = dspark_test_failing_alloc;

    static_assert(noexcept(std::declval<R&>().reset()),
                  "Resampler::reset must remain noexcept");

    const auto offlineInput = makeResamplerInput(0x12345678u);
    const auto primeInput = makeResamplerInput(0x87654321u);
    const auto continuationInput = makeResamplerInput(0x0badc0deu);

    // First mono prepare: every allocation failure propagates, preserves the
    // coherent default/unprepared state, and permits immediate recovery.
    int firstMonoFailures = 0;
    bool firstMonoSweepDone = false;
    for (int failIndex = 1; failIndex <= 32 && !firstMonoSweepDone; ++failIndex)
    {
        R candidate;
        int observedAllocations = 0;
        const bool threw = injectResamplerAllocationFailure(
            failIndex,
            [&] { candidate.prepare(44100.0, 48000.0, Q::High); },
            observedAllocations);

        if (!threw)
        {
            EXPECT_EQ(observedAllocations, failIndex - 1);
            EXPECT_NEAR(candidate.getRatio(), 48000.0 / 44100.0, 0.0);
            EXPECT_EQ(candidate.getLatency(), 112);
            firstMonoSweepDone = true;
            break;
        }
        ++firstMonoFailures;
        EXPECT_EQ(observedAllocations, failIndex);
        EXPECT_NEAR(candidate.getRatio(), 1.0, 0.0);
        EXPECT_EQ(candidate.getLatency(), 1);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size())).empty());
        EXPECT_EQ(renderMonoBlock(candidate, offlineInput).produced, 0);

        candidate.prepare(44100.0, 48000.0, Q::High);
        R recoveredReference;
        recoveredReference.prepare(44100.0, 48000.0, Q::High);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size()))
                    == recoveredReference.process(offlineInput.data(),
                                                  static_cast<int>(offlineInput.size())));
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(recoveredReference,
                                               continuationInput)));
    }
    EXPECT_TRUE(firstMonoSweepDone);
    EXPECT_GT(firstMonoFailures, 1);

    // Mono re-prepare grows Draft history to Ultra. Preserve both stateless
    // offline behavior and the exact in-flight streaming position on failure.
    int monoReprepareFailures = 0;
    bool monoReprepareSweepDone = false;
    for (int failIndex = 1; failIndex <= 32 && !monoReprepareSweepDone; ++failIndex)
    {
        R candidate;
        R priorReference;
        candidate.prepare(32000.0, 48000.0, Q::Draft);
        priorReference.prepare(32000.0, 48000.0, Q::Draft);
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, primeInput),
                               renderMonoBlock(priorReference, primeInput)));
        const double priorRatio = candidate.getRatio();
        const int priorLatency = candidate.getLatency();

        int observedAllocations = 0;
        const bool threw = injectResamplerAllocationFailure(
            failIndex,
            [&] { candidate.prepare(96000.0, 44100.0, Q::Ultra); },
            observedAllocations);

        if (!threw)
        {
            EXPECT_EQ(observedAllocations, failIndex - 1);
            EXPECT_NEAR(candidate.getRatio(), 44100.0 / 96000.0, 0.0);
            EXPECT_EQ(candidate.getLatency(), 166);
            monoReprepareSweepDone = true;
            break;
        }
        ++monoReprepareFailures;
        EXPECT_EQ(observedAllocations, failIndex);
        EXPECT_NEAR(candidate.getRatio(), priorRatio, 0.0);
        EXPECT_EQ(candidate.getLatency(), priorLatency);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size()))
                    == priorReference.process(offlineInput.data(),
                                              static_cast<int>(offlineInput.size())));
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(priorReference,
                                               continuationInput)));
        candidate.reset();
        priorReference.reset();
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(priorReference,
                                               continuationInput)));

        candidate.prepare(96000.0, 44100.0, Q::Ultra);
        R recoveredReference;
        recoveredReference.prepare(96000.0, 44100.0, Q::Ultra);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size()))
                    == recoveredReference.process(offlineInput.data(),
                                                  static_cast<int>(offlineInput.size())));
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(recoveredReference,
                                               continuationInput)));
    }
    EXPECT_TRUE(monoReprepareSweepDone);
    EXPECT_GT(monoReprepareFailures, 1);

    // First AudioSpec prepare grows from zero to three independent channels.
    // Its whole allocation frontier must retain first-prepare semantics.
    const AudioSpec firstSpec { 44100.0, kResamplerInputSamples, 3 };
    auto threeChannelInput = makeMultiInput<3>(0x31415926u);
    int firstSpecFailures = 0;
    bool firstSpecSweepDone = false;
    for (int failIndex = 1; failIndex <= 64 && !firstSpecSweepDone; ++failIndex)
    {
        R candidate;
        int observedAllocations = 0;
        const bool threw = injectResamplerAllocationFailure(
            failIndex,
            [&] { candidate.prepare(firstSpec, 48000.0, Q::High); },
            observedAllocations);

        if (!threw)
        {
            EXPECT_EQ(observedAllocations, failIndex - 1);
            EXPECT_NEAR(candidate.getRatio(), 48000.0 / 44100.0, 0.0);
            EXPECT_EQ(candidate.getLatency(), 112);
            const auto rendered = renderMultiBlock(candidate, threeChannelInput);
            EXPECT_GT(rendered.produced, 0);
            firstSpecSweepDone = true;
            break;
        }
        ++firstSpecFailures;
        EXPECT_EQ(observedAllocations, failIndex);
        EXPECT_NEAR(candidate.getRatio(), 1.0, 0.0);
        EXPECT_EQ(candidate.getLatency(), 1);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size())).empty());
        EXPECT_EQ(renderMonoBlock(candidate, offlineInput).produced, 0);

        candidate.prepare(firstSpec, 48000.0, Q::High);
        R recoveredReference;
        recoveredReference.prepare(firstSpec, 48000.0, Q::High);
        EXPECT_TRUE(sameRender(renderMultiBlock(candidate, threeChannelInput),
                               renderMultiBlock(recoveredReference,
                                                threeChannelInput)));
    }
    EXPECT_TRUE(firstSpecSweepDone);
    EXPECT_GT(firstSpecFailures, 4);

    // AudioSpec re-prepare grows both quality/history and the channel count.
    // Exercise prior mono and multichannel streaming state, reset behavior,
    // offline output and recovery after every injected failure.
    const AudioSpec priorSpec { 32000.0, kResamplerInputSamples, 2 };
    const AudioSpec grownSpec { 96000.0, kResamplerInputSamples, 4 };
    auto twoChannelPrime = makeMultiInput<2>(0x27182818u);
    auto twoChannelContinuation = makeMultiInput<2>(0xfeedfaceu);
    auto fourChannelInput = makeMultiInput<4>(0xc001d00du);
    int grownSpecFailures = 0;
    bool grownSpecSweepDone = false;
    for (int failIndex = 1; failIndex <= 64 && !grownSpecSweepDone; ++failIndex)
    {
        R candidate;
        R priorReference;
        candidate.prepare(priorSpec, 48000.0, Q::Draft);
        priorReference.prepare(priorSpec, 48000.0, Q::Draft);
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, primeInput),
                               renderMonoBlock(priorReference, primeInput)));
        EXPECT_TRUE(sameRender(renderMultiBlock(candidate, twoChannelPrime),
                               renderMultiBlock(priorReference,
                                                twoChannelPrime)));
        const double priorRatio = candidate.getRatio();
        const int priorLatency = candidate.getLatency();

        int observedAllocations = 0;
        const bool threw = injectResamplerAllocationFailure(
            failIndex,
            [&] { candidate.prepare(grownSpec, 44100.0, Q::Ultra); },
            observedAllocations);

        if (!threw)
        {
            EXPECT_EQ(observedAllocations, failIndex - 1);
            EXPECT_NEAR(candidate.getRatio(), 44100.0 / 96000.0, 0.0);
            EXPECT_EQ(candidate.getLatency(), 166);
            const auto rendered = renderMultiBlock(candidate, fourChannelInput);
            EXPECT_GT(rendered.produced, 0);
            for (const auto& channel : rendered.samples)
            {
                bool nonzero = false;
                for (int sample = 0; sample < rendered.produced; ++sample)
                    nonzero = nonzero
                        || channel[static_cast<size_t>(sample)] != 0.0f;
                EXPECT_TRUE(nonzero);
            }
            grownSpecSweepDone = true;
            break;
        }
        ++grownSpecFailures;
        EXPECT_EQ(observedAllocations, failIndex);
        EXPECT_NEAR(candidate.getRatio(), priorRatio, 0.0);
        EXPECT_EQ(candidate.getLatency(), priorLatency);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size()))
                    == priorReference.process(offlineInput.data(),
                                              static_cast<int>(offlineInput.size())));
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(priorReference,
                                               continuationInput)));
        EXPECT_TRUE(sameRender(
            renderMultiBlock(candidate, twoChannelContinuation),
            renderMultiBlock(priorReference, twoChannelContinuation)));
        candidate.reset();
        priorReference.reset();
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(priorReference,
                                               continuationInput)));
        EXPECT_TRUE(sameRender(
            renderMultiBlock(candidate, twoChannelContinuation),
            renderMultiBlock(priorReference, twoChannelContinuation)));

        candidate.prepare(grownSpec, 44100.0, Q::Ultra);
        R recoveredReference;
        recoveredReference.prepare(grownSpec, 44100.0, Q::Ultra);
        EXPECT_TRUE(candidate.process(offlineInput.data(),
                                      static_cast<int>(offlineInput.size()))
                    == recoveredReference.process(offlineInput.data(),
                                                  static_cast<int>(offlineInput.size())));
        EXPECT_TRUE(sameRender(renderMonoBlock(candidate, continuationInput),
                               renderMonoBlock(recoveredReference,
                                               continuationInput)));
        EXPECT_TRUE(sameRender(renderMultiBlock(candidate, fourChannelInput),
                               renderMultiBlock(recoveredReference,
                                                fourChannelInput)));
    }
    EXPECT_TRUE(grownSpecSweepDone);
    EXPECT_GT(grownSpecFailures, 5);

    // Counting mode arms the replacement allocator beyond any reachable call
    // index. reset() must touch no allocation function for a fully prepared
    // mono + multichannel object.
    R resetProbe;
    resetProbe.prepare(grownSpec, 44100.0, Q::Ultra);
    (void)renderMonoBlock(resetProbe, primeInput);
    (void)renderMultiBlock(resetProbe, fourChannelInput);
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    resetProbe.reset();
    const int resetAllocations = fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_EQ(resetAllocations, 0);
}

DSPARK_TEST(OfflineLeveler_allocation_failures_preserve_destination_and_prior_plan)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<float> input, output;
    input.resize(1, 36 * 8000);
    output.resize(1, 2);
    output.getChannel(0)[0] = 0.123f;
    for (int f = 0; f < input.getNumSamples(); ++f)
        input.getChannel(0)[f] = static_cast<float>(
            (f < 12 * 8000 ? 0.04 : f < 24 * 8000 ? 0.16 : 0.08)
            * std::sin(twoPi<double> * 173 * f / 8000));
    OfflineBufferSource<float> source(input.toView(), 8000);
    OfflineLeveler<float> leveler;
    OfflineLeveler<float>::Options options;
    options.amount = 1;
    auto prior = leveler.analyze(source, options);
    EXPECT_TRUE(prior.succeeded());
    const double previousGain = prior.plan.gainAt(6 * 8000);
    AudioBuffer<float> probe;
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto measured = leveler.run(input, probe, 8000, options);
    const int allocations = fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    int injected = 0;
    bool completed = false;
    // Include every allocation in the complete render, including product maps.
    for (int index = 1; index <= allocations + 1; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = leveler.run(input, output, 8000, options);
        const int count = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_EQ(prior.plan.gainAt(6 * 8000), previousGain);
        if (result.succeeded())
        {
            EXPECT_EQ(count, index - 1);
            EXPECT_EQ(output.getNumSamples(), input.getNumSamples());
            completed = true;
            break;
        }
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(count, index);
        EXPECT_EQ(output.getNumSamples(), 2);
        EXPECT_EQ(output.getChannel(0)[0], 0.123f);
        ++injected;
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(injected, 8);

    // The existing injector covers ordinary new/new[] (map, plan and block
    // storage). A sink allocation can also fail after verification; exercise
    // that transaction boundary explicitly without changing global aligned new.
    class FailedAllocationSink final : public OfflineAudioSink<float>
    {
    public:
        bool aborted = false;
        bool begin(const OfflineAudioSpec&) override { throw std::bad_alloc(); }
        bool write(std::int64_t, AudioBufferView<const float>) override { return true; }
        bool commit() override { return true; }
        void abort() noexcept override { aborted = true; }
    } sink;
    EXPECT_TRUE(leveler.render(source, prior.plan, sink).status == OfflineStatus::AllocationFailure);
    EXPECT_TRUE(sink.aborted);
}

DSPARK_TEST(Onset_shared_features_recover_from_prepare_failure_and_allocate_no_audio_memory)
{
    namespace fa = dspark_test_failing_alloc;
    const AudioSpec small {48000,256,1}, large {192000,256,1};
    std::array<float,4096> input{};
    for (std::size_t i=0;i<input.size();++i)
        input[i]=static_cast<float>(0.2*std::sin(twoPi<double>*0.013*static_cast<double>(i)));
    int injected=0;
    for (bool reprepare:{false,true})
    {
        bool completed=false;
        for (int index=1;index<=192;++index)
        {
            OnsetDetector<float> detector;
            if (reprepare) detector.prepare(small,512);
            fa::count.store(0,std::memory_order_relaxed);
            fa::failAt.store(index,std::memory_order_relaxed);
            bool failed=false;
            try { detector.prepare(large); }
            catch (const std::bad_alloc&) { failed=true; }
            const int calls=fa::count.load(std::memory_order_relaxed);
            fa::failAt.store(-1,std::memory_order_relaxed);
            if (!failed)
            {
                EXPECT_LT(calls,index);
                completed=true;
                break;
            }
            ++injected;
            // A failed prepare keeps the audio gate closed even if a sub-kernel
            // has finished allocating. reset and the next prepare must be safe.
            detector.reset();
            detector.pushSamples(input);
            EXPECT_EQ(detector.getLastOdfFrame().frameIndex,int64_t(0));
            detector.prepare(small,512);
            detector.pushSamples(input);
            EXPECT_GT(detector.getLastOdfFrame().frameIndex,int64_t(0));
        }
        EXPECT_TRUE(completed);
    }
    EXPECT_GT(injected,20);
    OnsetDetector<float> detector;
    detector.prepare(small,512);
    detail::OnsetSpectrum<float> spectrum;
    detail::OnsetNovelty<float> novelty;
    spectrum.prepare(512);
    novelty.prepare(48000,512);
    const float* channels[]{input.data(),input.data()+512};
    AudioBufferView<const float> frame(channels,2,512);
    fa::count.store(0,std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(),std::memory_order_relaxed);
    detector.pushSamples(input);
    const bool pooled=spectrum.computePooled(frame);
    const auto features=novelty.process(spectrum.magnitudes(),{},
        detail::OnsetNovelty<float>::Method::BothFlux,true);
    detector.reset();
    novelty.reset();
    const int allocations=fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1,std::memory_order_relaxed);
    EXPECT_TRUE(pooled && features.valid);
    EXPECT_EQ(allocations,0);
}

DSPARK_TEST(OfflineTransient_allocation_failure_and_workspace_budget)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<double> input;
    input.resize(2, 8000);
    input.clear();
    for (int f = 0; f < 8000; f += 2000)
        input.getChannel(0)[f] = input.getChannel(1)[f] = 1;
    OfflineBufferSource<double> source(input.toView(), 8000);
    OfflineTransientAnalyzer<double> analyzer;
    auto prior = analyzer.analyze(source);
    EXPECT_TRUE(prior.succeeded());
    EXPECT_EQ(prior.analysis.attacks().size(), std::size_t(4));
    bool completed = false;
    int injected = 0;
    for (int index = 1; index < 128; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        auto result = analyzer.analyze(source);
        const int calls = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_EQ(prior.analysis.attacks().size(), std::size_t(4));
        if (result.succeeded())
        {
            EXPECT_EQ(calls, index - 1);
            completed = true;
            break;
        }
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_FALSE(result.analysis.isValid());
        EXPECT_TRUE(result.analysis.attacks().empty());
        ++injected;
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(injected, 20);
    const auto checkBudget = [&]<typename T>() {
        AudioBuffer<T> buffer;
        buffer.resize(2, 8000);
        buffer.clear();
        buffer.getChannel(0)[0] = buffer.getChannel(1)[0] = T(1);
        for (double rate : {8000.0, 44100.0, 48000.0, 96000.0, 192000.0, 384000.0})
        {
            OfflineBufferSource<T> current(buffer.toView(), rate);
            fa::count.store(0, std::memory_order_relaxed);
            fa::bytes.store(0, std::memory_order_relaxed);
            fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
            const auto result = OfflineTransientAnalyzer<T>().analyze(current);
            const auto requested = fa::bytes.load(std::memory_order_relaxed);
            fa::failAt.store(-1, std::memory_order_relaxed);
            EXPECT_TRUE(result.succeeded());
            EXPECT_TRUE(requested <= result.memoryBytes);
            OfflineJobOptions options;
            options.memoryBudgetBytes = result.memoryBytes;
            const auto exact = OfflineTransientAnalyzer<T>().analyze(current, {}, options);
            EXPECT_TRUE(exact.succeeded());
            options.memoryBudgetBytes = result.memoryBytes - 1;
            const auto insufficient = OfflineTransientAnalyzer<T>().analyze(current, {}, options);
            EXPECT_TRUE(insufficient.status == OfflineStatus::MemoryLimit);
            EXPECT_FALSE(insufficient.analysis.isValid());
        }
    };
    checkBudget.template operator()<float>();
    checkBudget.template operator()<double>();
}

template <class Processor> static void offlineGainAllocationFailures(typename Processor::Options options)
{
    namespace fa=dspark_test_failing_alloc;
    AudioBuffer<float> input,output;
    input.resize(2,8000);input.getChannel(0)[100]=1;input.getChannel(1)[100]=-1;
    output.resize(1,1);output.getChannel(0)[0]=.125f;
    OfflineBufferSource<float> source(input.toView(),8000);
    Processor processor;
    OfflineRegion protectedRegion{2000,2100};
    options.exclusions={&protectedRegion,1};
    auto prior=processor.analyze(source,options);
    EXPECT_TRUE(prior.succeeded());
    const auto previousGain=prior.plan.gainAt(100);
    AudioBuffer<float> measuredOutput;
    fa::count.store(0,std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(),std::memory_order_relaxed);
    const auto measured=processor.run(input,measuredOutput,8000,options);
    const int allocations=fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1,std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    bool completed=false;
    int injected=0;
    for(int index=1;index<=allocations+1;++index)
    {
        fa::count.store(0,std::memory_order_relaxed);
        fa::failAt.store(index,std::memory_order_relaxed);
        auto result=processor.run(input,output,8000,options);
        const auto calls=fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1,std::memory_order_relaxed);
        EXPECT_EQ(prior.plan.gainAt(100),previousGain);
        if(result.succeeded()) { EXPECT_EQ(calls,index-1);completed=true;break; }
        EXPECT_TRUE(result.status==OfflineStatus::AllocationFailure);
        EXPECT_EQ(output.getNumSamples(),1);
        EXPECT_EQ(output.getChannel(0)[0],.125f);
        ++injected;
    }
    EXPECT_TRUE(completed);
    EXPECT_GT(injected,20);
    auto map=OfflineTransientAnalyzer<float>().analyze(source);
    EXPECT_TRUE(map.succeeded());
    fa::bytes.store(0,std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(),std::memory_order_relaxed);
    auto planned=processor.makePlan(source,map.analysis,options);
    const auto bytes=fa::bytes.load(std::memory_order_relaxed);
    fa::failAt.store(-1,std::memory_order_relaxed);
    EXPECT_TRUE(planned.succeeded());
    EXPECT_EQ(bytes,planned.memoryBytes);
}

DSPARK_TEST(OfflinePeak_allocation_failures_preserve_plan_and_output)
{
    offlineGainAllocationFailures<OfflinePeakCompressor<float>>({6});
}

DSPARK_TEST(OfflinePunch_allocation_failures_preserve_plan_and_output)
{
    offlineGainAllocationFailures<OfflinePunch<float>>({6});
}

DSPARK_TEST(FIRDesign_allocation_failures_propagate_without_publishing_coefficients)
{
    namespace fa = dspark_test_failing_alloc;
    const auto check = []<typename T>() {
        for (int shape = 0; shape < 4; ++shape)
        {
            const auto design = [shape] {
                switch (shape)
                {
                case 0:
                    return FIRDesign<T>::lowPass(48000, 8000, 63, T(14));
                case 1:
                    return FIRDesign<T>::highPass(48000, 8000, 63, T(14));
                case 2:
                    return FIRDesign<T>::bandPass(48000, 4000, 12000, 63, T(14));
                default:
                    return FIRDesign<T>::bandStop(48000, 4000, 12000, 63, T(14));
                }
            };
            const auto expected = design();
            const std::vector<T> previous{T(.25), T(-.5)};
            int injected = 0;
            bool completed = false;
            for (int index = 1; index <= 32; ++index)
            {
                auto coefficients = previous;
                fa::count.store(0, std::memory_order_relaxed);
                fa::failAt.store(index, std::memory_order_relaxed);
                bool failed = false;
                try
                {
                    coefficients = design();
                }
                catch (const std::bad_alloc &)
                {
                    failed = true;
                }
                const int calls = fa::count.load(std::memory_order_relaxed);
                fa::failAt.store(-1, std::memory_order_relaxed);
                if (!failed)
                {
                    EXPECT_EQ(calls, index - 1);
                    EXPECT_TRUE(coefficients == expected);
                    completed = true;
                    break;
                }
                ++injected;
                EXPECT_TRUE(coefficients == previous);
                // Recovery must reproduce the exact original coefficients;
                // changing exception propagation must not change the filter.
                EXPECT_TRUE(design() == expected);
            }
            EXPECT_TRUE(completed);
            EXPECT_GT(injected, 2);
        }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

DSPARK_TEST(OfflineProduct_allocation_failures_and_payload_bound)
{
    namespace fa = dspark_test_failing_alloc;
    for (int block : {128, 4096})
    {
        const OfflineJobOptions options;
        std::vector<double> x(513), g(513), expected(513), output(513);
        for (int i = 0; i < 513; ++i)
        {
            x[i] = std::cos(3.13 * i);
            g[i] = .5 + .3 * std::sin(.017 * i);
        }
        const auto left = [&](std::int64_t first, int count, double *out) {
            std::copy_n(x.data() + first, count, out);
        };
        const auto right = [&](std::int64_t first, int count, double *out) {
            std::copy_n(g.data() + first, count, out);
        };
        const auto render = [&](std::vector<double> &destination) {
            detail::OfflineSession job(options);
            detail::offlineBandlimitedProduct(
                job, 513, left, right,
                [&](std::int64_t first, int count, const double *data) {
                    std::copy_n(data, count, destination.data() + first);
                },
                block);
            return job.bytes();
        };
        const auto expectedCharge = render(expected);
        bool completed = false;
        int injected = 0;
        for (int index = 1; index < 256; ++index)
        {
            std::fill(output.begin(), output.end(), -.125);
            fa::count.store(0, std::memory_order_relaxed);
            fa::bytes.store(0, std::memory_order_relaxed);
            fa::failAt.store(index, std::memory_order_relaxed);
            bool failed = false;
            std::size_t charged = 0;
            try
            {
                charged = render(output);
            }
            catch (const std::bad_alloc &)
            {
                failed = true;
            }
            const auto calls = fa::count.load(std::memory_order_relaxed);
            const auto bytes = fa::bytes.load(std::memory_order_relaxed);
            fa::failAt.store(-1, std::memory_order_relaxed);
            if (!failed)
            {
                EXPECT_EQ(calls, index - 1);
                EXPECT_TRUE(bytes <= charged);
                EXPECT_EQ(charged, expectedCharge);
                EXPECT_TRUE(output == expected);
                completed = true;
                break;
            }
            ++injected;
            for (auto value : output)
                EXPECT_EQ(value, -.125);
            EXPECT_EQ(render(output), expectedCharge);
            EXPECT_TRUE(output == expected);
        }
        EXPECT_TRUE(completed);
        EXPECT_GT(injected, 30);
    }
}

DSPARK_TEST(OfflinePeak_adaptive_anticipation_allocation_failures_cannot_publish)
{
    namespace fa = dspark_test_failing_alloc;
    constexpr int rate = 48000, n = rate / 2 + 17;
    AudioBuffer<float> input, output;
    input.resize(1, n);
    for (int i = 0; i < n; ++i)
    {
        const double t = double(i) / rate;
        double amplitude = .003;
        for (double start : {.1, .3})
            if (t >= start && t < start + .06)
                amplitude += .8 * std::exp(-(t - start) / .012) *
                             std::pow(std::clamp((start + .06 - t) / .01, 0., 1.), 2);
        input.getChannel(0)[i] = static_cast<float>(amplitude * std::cos(twoPi<double> * 23952 * t));
    }
    OfflinePeakCompressor<float> processor;
    OfflinePeakCompressor<float>::Options options{18, .1, 80};
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto measured = processor.run(input, output, rate, options);
    const int allocations = fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    EXPECT_GT(measured.report.renderAttempts, 1);
    output.resize(1, 1);
    output.getChannel(0)[0] = .123f;
    for (int index = 1; index <= allocations; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = processor.run(input, output, rate, options);
        const int calls = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(calls, index);
        EXPECT_EQ(output.getNumSamples(), 1);
        EXPECT_EQ(output.getChannel(0)[0], .123f);
    }
    EXPECT_TRUE(processor.run(input, output, rate, options).succeeded());
    EXPECT_EQ(output.getNumSamples(), n);
}

DSPARK_TEST(OfflineTempo_accounts_every_engine_allocation_and_never_publishes_failure)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<double> input;
    input.resize(1, 8000 * 8);
    input.clear();
    for (int beat = 0; beat < 16; ++beat)
        for (int i = 0; i < 400; ++i)
            input.getChannel(0)[beat * 4000 + i] =
                .7 * std::exp(-double(i) / 120) * std::cos(.17 * i);
    OfflineBufferSource<double> source(input.toView(), 8000);
    const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, true});
    EXPECT_TRUE(features.succeeded());
    fa::count.store(0, std::memory_order_relaxed);
    fa::bytes.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto measured = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    const int allocations = fa::count.load(std::memory_order_relaxed);
    const auto payload = fa::bytes.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    EXPECT_GT(measured.analysis.getReport().tempoBpm, 0.0);
    EXPECT_GT(allocations, 30);
    EXPECT_EQ(payload, measured.memoryBytes);
    for (int index = 1; index <= allocations; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = OfflineTempoAnalyzer<double>().analyze(features.analysis);
        const int calls = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_FALSE(result.analysis.isValid());
        EXPECT_EQ(calls, index);
    }
    const auto recovered = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    EXPECT_TRUE(recovered.succeeded());
    EXPECT_EQ(recovered.analysis.getReport().tempoBpm, measured.analysis.getReport().tempoBpm);

    // A returned grid larger than 4096 bytes also avoids std::allocator's
    // implementation-specific large-block payload on the public legacy Result.
    OfflineJobOptions options;
    detail::OfflineSession job(options);
    fa::count.store(0, std::memory_order_relaxed);
    fa::bytes.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto longGrid = detail::OfflineBeatEngine<double>::analyze(
        48000, 240, 60000,
        [](std::size_t i) {
            OnsetDetector<double>::OdfFrame frame;
            const double phase = static_cast<double>(i % 100) - 50;
            frame.value = std::exp(-phase * phase / 4);
            frame.registers.fill(frame.value);
            frame.referenceSample = static_cast<std::int64_t>(i) * 240;
            return frame;
        },
        40, 240, 25, job);
    const auto longPayload = fa::bytes.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_GT(longGrid.beats().size(), std::size_t(512));
    EXPECT_EQ(longPayload, job.bytes());
}

DSPARK_TEST(OfflineTempo_local_windows_account_pool_payload_and_recover_every_failure)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<double> input;
    input.resize(1, 8000 * 27);
    input.clear();
    int position = 0;
    for (int beat = 0; beat < 48; ++beat)
    {
        for (int i = 0; i < 320; ++i)
        {
            const double t = double(i) / 8000;
            input.getChannel(0)[position + i] = .2 * std::exp(-80 * t) *
                (std::cos(twoPi<double> * 113 * t) + .5 * std::cos(twoPi<double> * 2113 * t));
        }
        position += beat < 24 ? 4000 : 4800;
    }
    OfflineBufferSource<double> source(input.toView(), 8000);
    const auto features = OfflineTransientAnalyzer<double>().analyze(source, {false, true, true});
    EXPECT_TRUE(features.succeeded());
    fa::count.store(0, std::memory_order_relaxed);
    fa::bytes.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto measured = OfflineTempoAnalyzer<double>().analyze(features.analysis);
    const int allocations = fa::count.load(std::memory_order_relaxed);
    const auto payload = fa::bytes.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    EXPECT_GT(measured.analysis.getReport().locallyConfirmedIntervals, std::size_t(40));
    EXPECT_EQ(payload, measured.memoryBytes);
    for (int index = 1; index <= allocations; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = OfflineTempoAnalyzer<double>().analyze(features.analysis);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_FALSE(result.analysis.isValid());
    }
    OfflineJobOptions job;
    job.memoryBudgetBytes = measured.memoryBytes;
    EXPECT_TRUE(OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job).succeeded());
    --job.memoryBudgetBytes;
    const auto limited = OfflineTempoAnalyzer<double>().analyze(features.analysis, {}, job);
    EXPECT_TRUE(limited.status == OfflineStatus::MemoryLimit);
    EXPECT_FALSE(limited.analysis.isValid());
}

DSPARK_TEST(OfflineBeat_every_plan_and_render_allocation_failure_preserves_output)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<float> input, output;
    input.resize(2, 8000 * 4);
    input.clear();
    for (int beat = 0; beat < 8; ++beat)
        for (int i = 0; i < 320; ++i)
        {
            const double t = double(i) / 8000;
            const auto x = static_cast<float>((beat % 4 == 0 ? .8 : .2) * std::exp(-80 * t) *
                (std::cos(twoPi<double> * 113 * t) + .5 * std::cos(twoPi<double> * 2113 * t)));
            input.getChannel(0)[beat * 4000 + i] = x;
            input.getChannel(1)[beat * 4000 + i] = -x;
        }
    OfflineBeatCompressor<float> processor;
    OfflineBeatCompressor<float>::Options options;
    options.reductionDb = 6;
    options.manualBpm = 120;
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto measured = processor.run(input, output, 8000, options);
    const int allocations = fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    EXPECT_TRUE(measured.report.renderInfo.bandlimited);
    output.resize(1, 1);
    output.getChannel(0)[0] = .123f;
    for (int index = 1; index <= allocations; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = processor.run(input, output, 8000, options);
        const int calls = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(calls, index);
        EXPECT_EQ(output.getNumSamples(), 1);
        EXPECT_EQ(output.getChannel(0)[0], .123f);
    }
    EXPECT_TRUE(processor.run(input, output, 8000, options).succeeded());
}

namespace
{
template <class Processor> void clipAllocationFailures(int factor, int channels)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<double> input;
    input.resize(channels, 257);
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < 257; ++i)
            input.getChannel(c)[i] = (c ? -.8 : 1) * std::cos(.4 * twoPi<double> * i);
    OfflineBufferSource<double> source(input.toView(), 48000);
    typename Processor::Options options;
    options.oversamplingFactor = factor;
    options.reductionDb = 6;
    const OfflineRegion region{100, 101};
    // Copying a region is tested in plan construction. Render without constraints
    // here so all kernel/setup/output allocations belong to a feasible full job.
    options.exclusions = {&region, 1};
    Processor processor;
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto planned = processor.analyze(source, options);
    const int planAllocations = fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(planned.succeeded());
    for (int index = 1; index <= planAllocations; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto failed = processor.analyze(source, options);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(failed.status == OfflineStatus::AllocationFailure);
        EXPECT_FALSE(failed.plan.isValid());
    }
    options.exclusions = {};
    const auto plan = processor.analyze(source, options);
    class Sink final : public OfflineAudioSink<double>
    {
      public:
        bool began = false, committed = false, aborted = false;
        int allocationsAtBegin = 0;
        bool begin(const OfflineAudioSpec &) override
        {
            began = true;
            allocationsAtBegin = dspark_test_failing_alloc::count.load(std::memory_order_relaxed);
            return true;
        }
        bool write(std::int64_t, AudioBufferView<const double>) override { return true; }
        bool commit() override { committed = true; return true; }
        void abort() noexcept override { aborted = true; }
    } measuredSink;
    fa::count.store(0, std::memory_order_relaxed);
    fa::bytes.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto measured = processor.render(source, plan.plan, measuredSink);
    const int allocations = fa::count.load(std::memory_order_relaxed);
    const auto payload = fa::bytes.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(measured.succeeded());
    EXPECT_TRUE(payload <= measured.memoryBytes);
    EXPECT_EQ(allocations, measuredSink.allocationsAtBegin);
    EXPECT_GT(allocations, 10);
    for (int index = 1; index <= allocations; ++index)
    {
        Sink sink;
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = processor.render(source, plan.plan, sink);
        const int calls = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(calls, index);
        EXPECT_FALSE(sink.committed);
        EXPECT_TRUE(!sink.began || sink.aborted);
    }
    OfflineJobOptions job;
    job.memoryBudgetBytes = measured.memoryBytes;
    Sink sink;
    EXPECT_TRUE(processor.render(source, plan.plan, sink, job).succeeded());
    --job.memoryBudgetBytes;
    Sink refused;
    EXPECT_TRUE(processor.render(source, plan.plan, refused, job).status == OfflineStatus::MemoryLimit);
    EXPECT_FALSE(refused.committed);

    // The owning adapter's provisional AudioBuffer allocation is a sink callback.
    // Fail every allocation in run(), including that final aligned allocation.
    AudioBuffer<double> output;
    fa::count.store(0, std::memory_order_relaxed);
    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
    const auto owning = processor.run(input, output, 48000, options);
    const int owningAllocations = fa::count.load(std::memory_order_relaxed);
    fa::failAt.store(-1, std::memory_order_relaxed);
    EXPECT_TRUE(owning.succeeded());
    output.resize(1, 1);
    output.getChannel(0)[0] = .123;
    for (int index = 1; index <= owningAllocations; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = processor.run(input, output, 48000, options);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(output.getNumSamples(), 1);
        EXPECT_EQ(output.getChannel(0)[0], .123);
    }
}
}
DSPARK_TEST(StereoGenerator_prepare_failures_preserve_stream_and_callbacks_do_not_allocate)
{
    namespace fa = dspark_test_failing_alloc;
    for (int factor : {1, 8})
    for (bool prepared : {false, true})
    {
        bool finished = false;
        int injected = 0;
        for (int index = 1; index < 2048; ++index)
        {
            StereoGenerator<double> effect, reference;
            effect.setWidth(1);
            reference.setWidth(1);
            std::array<double, 1024> left{}, right{}, refLeft{}, refRight{};
            for (int i = 0; i < 1024; ++i)
            {
                left[i] = refLeft[i] = .4 * std::sin(.13 * i);
                right[i] = refRight[i] = .5 * std::cos(.31 * i);
            }
            double *channels[]{left.data(), right.data()}, *ref[]{refLeft.data(), refRight.data()};
            if (prepared)
            {
                EXPECT_TRUE(effect.prepare({48000, 1024, 2}));
                EXPECT_TRUE(reference.prepare({48000, 1024, 2}));
                EXPECT_TRUE(effect.processBlock({channels, 2, 1024}));
                EXPECT_TRUE(reference.processBlock({ref, 2, 1024}));
            }
            int allocations = 0;
            const bool failed = injectResamplerAllocationFailure(index, [&] {
                (void)effect.prepare({96000, 1024, 2}, {factor, 175});
            }, allocations);
            if (!failed)
            {
                finished = true;
                break;
            }
            ++injected;
            EXPECT_EQ(effect.getLatency(), prepared ? reference.getLatency() : 0);
            EXPECT_EQ(effect.getSourceFrame(), prepared ? 1024u : 0u);
            EXPECT_EQ(effect.getOptions().oversampling, 1);
            EXPECT_EQ(effect.getOptions().lowCutHz, 0.f);
            if (prepared)
            {
                // Count without throwing through a noexcept callback. Any allocation
                // fails the test while preserving the allocator sanitizer boundary.
                fa::count.store(0, std::memory_order_relaxed);
                fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
                const bool ok = effect.processBlock({channels, 2, 1024});
                effect.resetAtFrame(1000000000037ULL);
                const int calls = fa::count.load(std::memory_order_relaxed);
                fa::failAt.store(-1, std::memory_order_relaxed);
                EXPECT_TRUE(ok);
                EXPECT_EQ(calls, 0);
                EXPECT_TRUE(reference.processBlock({ref, 2, 1024}));
                EXPECT_TRUE(left == refLeft && right == refRight);
            }
            else
                EXPECT_FALSE(effect.processBlock({channels, 2, 1024}));
            EXPECT_TRUE(effect.prepare({96000, 1024, 2}, {factor, 175}));
        }
        EXPECT_TRUE(finished);
        EXPECT_GT(injected, 50);
    }
}

DSPARK_TEST(OfflineClip_every_allocation_is_accounted_and_failure_is_transactional)
{
    clipAllocationFailures<OfflineHardClipper<double>>(4, 2);
    clipAllocationFailures<OfflineSoftClipper<double>>(1, 1);
    clipAllocationFailures<OfflineSoftClipper<double>>(16, 1);
}

DSPARK_TEST(StereoGenerator_memory_bound_covers_actual_setup_at_supported_endpoints)
{
    namespace fa = dspark_test_failing_alloc;
    const auto check = [&]<typename T>() {
        for (double rate : {8000., 44100., 48000., 96000., 192000., 384000.})
            for (int factor : {1, 2, 4, 8, 16})
                for (float cut : {0.f, 20.f, 175.f, 5000.f})
                {
                    StereoGenerator<T> effect;
                    fa::count.store(0, std::memory_order_relaxed);
                    fa::bytes.store(0, std::memory_order_relaxed);
                    fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
                    const auto bound = StereoGenerator<T>::getPrepareMemoryBound({rate, 1, 2}, {factor, cut});
                    const auto queryAllocations = fa::count.load(std::memory_order_relaxed);
                    const bool prepared = effect.prepare({rate, 1, 2}, {factor, cut});
                    const auto payload = fa::bytes.load(std::memory_order_relaxed);
                    fa::failAt.store(-1, std::memory_order_relaxed);
                    EXPECT_TRUE(prepared);
                    EXPECT_EQ(queryAllocations, 0);
                    if (payload > bound)
                        std::cerr << "Stereo prepare bytes=" << payload << " bound=" << bound
                                  << " rate=" << rate << " factor=" << factor << " cut=" << cut
                                  << " sampleBytes=" << sizeof(T) << '\n';
                    EXPECT_TRUE(payload <= bound);
                }
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

DSPARK_TEST(OfflineStereo_every_allocation_failure_is_transactional_and_budgeted)
{
    namespace fa = dspark_test_failing_alloc;
    using Processor = OfflineStereoGenerator<double>;
    AudioBuffer<double> input, output;
    input.resize(1, 17);
    for (int i = 0; i < 17; ++i) input.getChannel(0)[i] = .5 * std::cos(.3*i);
    OfflineBufferSource<double> source(input.toView(), 8000);
    Processor effect;
    Processor::Options options;
    options.width = .5f; options.lowCutHz = 175; options.duplicateMono = true;
    const OfflineRegion region{7, 9}; options.exclusions = {&region, 1};
    int measuredCalls = 0;
    const auto measure = [&](auto &&function) {
        fa::count.store(0, std::memory_order_relaxed);
        fa::bytes.store(0, std::memory_order_relaxed);
        fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
        auto result = function();
        measuredCalls = fa::count.load(std::memory_order_relaxed);
        const auto payload = fa::bytes.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.succeeded());
        EXPECT_TRUE(payload <= result.memoryBytes);
    };
    measure([&] { return effect.analyze(source, options); });
    const int planCalls = measuredCalls;
    for (int index = 1; index <= planCalls; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = effect.analyze(source, options);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_FALSE(result.plan.isValid());
    }
    const auto plan = effect.analyze(source, options);
    class Sink final : public OfflineAudioSink<double>
    {
      public:
        bool began = false, aborted = false, committed = false;
        int allocationsAtBegin = 0;
        std::array<double, 17> delta{};
        bool begin(const OfflineAudioSpec &) override
        {
            began = true;
            allocationsAtBegin = dspark_test_failing_alloc::count.load(std::memory_order_relaxed);
            return true;
        }
        bool write(std::int64_t first, AudioBufferView<const double> values) override
        {
            std::copy_n(values.getChannel(0), values.getNumSamples(), delta.data()+first);
            return true;
        }
        bool commit() override { committed = true; return true; }
        void abort() noexcept override { aborted = true; }
    } delta;
    auto cache = effect.buildDeltaCache(source, plan.plan, delta);
    const double *channel[]{delta.delta.data()};
    OfflineBufferSource<double> cached({channel, 1, 17}, 8000);
    for (int path : {0, 1, 2})
    {
        Sink measuredSink;
        const auto render = [&](Sink &sink, const OfflineJobOptions &job) {
            if (path == 0) return effect.render(source, plan.plan, sink, job);
            if (path == 1)
            {
                const auto result = effect.buildDeltaCache(source, plan.plan, sink, job);
                Processor::Result converted;
                converted.status = result.status; converted.memoryBytes = result.memoryBytes;
                return converted;
            }
            return effect.renderCached(source, plan.plan, cached, cache.cache, sink, job);
        };
        std::size_t budget = 0;
        measure([&] {
            const auto result = render(measuredSink, {}); budget = result.memoryBytes; return result;
        });
        const auto calls = measuredCalls;
        EXPECT_EQ(calls, measuredSink.allocationsAtBegin);
        for (int index = 1; index <= calls; ++index)
        {
            Sink sink;
            fa::count.store(0, std::memory_order_relaxed);
            fa::failAt.store(index, std::memory_order_relaxed);
            const auto result = render(sink, {});
            const auto actual = fa::count.load(std::memory_order_relaxed);
            fa::failAt.store(-1, std::memory_order_relaxed);
            EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
            EXPECT_EQ(actual, index);
            EXPECT_FALSE(sink.committed);
            EXPECT_TRUE(!sink.began || sink.aborted);
        }
        OfflineJobOptions job; job.memoryBudgetBytes = budget;
        Sink allowed, limited;
        EXPECT_TRUE(render(allowed, job).succeeded());
        --job.memoryBudgetBytes;
        EXPECT_TRUE(render(limited, job).status == OfflineStatus::MemoryLimit);
        EXPECT_FALSE(limited.began);
    }
    measure([&] { return effect.run(input, output, 8000, options); });
    const int ownCalls = measuredCalls;
    output.resize(1, 1); output.getChannel(0)[0] = .123;
    for (int index = 1; index <= ownCalls; ++index)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(index, std::memory_order_relaxed);
        const auto result = effect.run(input, output, 8000, options);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(output.getNumSamples(), 1);
        EXPECT_EQ(output.getChannel(0)[0], .123);
    }
}

DSPARK_TEST(LoudnessMeter_true_peak_finalization_allocates_nothing)
{
    const auto check = []<class T>()
    {
        LoudnessMeter<T> meter;
        meter.prepare(48000.0, 2);
        const T end[] = { T(0.05), T(0.9), T(-0.9) };
        meter.process(end, end, 3);
        namespace fa = dspark_test_failing_alloc;
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
        meter.finalizeTruePeak();
        const T peak = meter.getTruePeakDb();
        meter.finalizeTruePeak();
        const int allocations = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_EQ(allocations, 0);
        EXPECT_EQ(meter.getTruePeakDb(), peak);
        EXPECT_TRUE(std::isfinite(peak));
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

DSPARK_TEST(AudioInterval_setup_process_and_finish_allocate_nothing)
{
    const auto check = []<class T>()
    {
        using A = AudioIntervalAnalyzer<T>;
        A analyzer;
        const T input[] = { T(.1), T(.2), T(-.2), T(.3) };
        const T* channels[] = { input, input };
        namespace fa = dspark_test_failing_alloc;
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
        const bool prepared = analyzer.prepare({ 48000, 4, 2 }, { 1, 3 }, A::Context::IncludeContinuous);
        const bool processed = analyzer.processBlock({ channels, 2, 4 }, 0);
        const auto result = analyzer.finish();
        const int allocations = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_EQ(allocations, 0);
        EXPECT_TRUE(prepared && processed && result.complete());
        EXPECT_EQ(result.samplePeak.value, static_cast<double>(T(.2)));
    };
    check.template operator()<float>();
    check.template operator()<double>();
}

DSPARK_TEST(OfflineStereoBalance_allocation_failures_and_exact_payload_budget)
{
    namespace fa = dspark_test_failing_alloc;
    AudioBuffer<double> input, output;
    input.resize(2, 1601);
    for (int i = 0; i < 1601; ++i)
    {
        const double side = i < 800 ? .6 : .02;
        input.getChannel(0)[i] = .25 + side;
        input.getChannel(1)[i] = .25 - side;
    }
    OfflineBufferSource<double> source(input.toView(), 8000);
    OfflineStereoBalance<double> effect;
    OfflineStereoBalance<double>::Options options;
    options.levelingAmount = 1;
    options.targetSideShare = .2;
    options.guard = true;
    struct Measurement { int calls; std::size_t bytes; bool succeeded; };
    const auto measure = [&](auto &&operation) {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(std::numeric_limits<int>::max(), std::memory_order_relaxed);
        const auto result = operation();
        const int calls = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        return Measurement{calls, result.memoryBytes, result.succeeded()};
    };
    const auto energy = measure([&] { return OfflineEnergyAnalyzer<double>().analyzeMidSide(source); });
    EXPECT_TRUE(energy.succeeded);
    for (int i = 1; i <= energy.calls; ++i)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(i, std::memory_order_relaxed);
        const auto result = OfflineEnergyAnalyzer<double>().analyzeMidSide(source);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_FALSE(result.analysis.isValid());
    }
    const auto own = measure([&] { return effect.run(input, output, 8000, options); });
    EXPECT_TRUE(own.succeeded);
    output.resize(1, 1);
    output.getChannel(0)[0] = .123;
    for (int i = 1; i <= own.calls; ++i)
    {
        fa::count.store(0, std::memory_order_relaxed);
        fa::failAt.store(i, std::memory_order_relaxed);
        const auto result = effect.run(input, output, 8000, options);
        const int actual = fa::count.load(std::memory_order_relaxed);
        fa::failAt.store(-1, std::memory_order_relaxed);
        EXPECT_TRUE(result.status == OfflineStatus::AllocationFailure);
        EXPECT_EQ(actual, i);
        EXPECT_EQ(output.getNumSamples(), 1);
        EXPECT_EQ(output.getChannel(0)[0], .123);
    }
    OfflineJobOptions job;
    job.memoryBudgetBytes = own.bytes;
    EXPECT_TRUE(effect.run(input, output, 8000, options, job).succeeded());
    --job.memoryBudgetBytes;
    EXPECT_TRUE(effect.run(input, output, 8000, options, job).status == OfflineStatus::MemoryLimit);
}
