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

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

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
// TSan link, confirmed from the linked binaries' symbol tables. Running the
// suite under ThreadSanitizer is CI's job, so the RUN under TSan is proven
// there rather than here; if the
// replacement ever failed to take effect the case would fail loudly on
// EXPECT_GT(injected, 10) with injected == 0, never pass silently. It
// therefore runs UNGUARDED under every sanitizer.
namespace dspark_test_failing_alloc {
// constinit, so "no dynamic initialization" is compiler-checked: these words
// are touched by allocations made during the static init of other TUs, which
// may run before this TU's. Atomic words, so an allocation on any live thread
// reads them race-free (the disarmed fast path is one relaxed load).
constinit std::atomic<int> failAt{ -1 }; // -1 = disarmed; N = Nth new throws
constinit std::atomic<int> count{ 0 };   // counts operator new calls if armed
} // namespace dspark_test_failing_alloc

static void* dsparkTestCountedAlloc(std::size_t size)
{
    namespace fa = dspark_test_failing_alloc;
    if (fa::failAt.load(std::memory_order_relaxed) > 0)
    {
        const int n = fa::count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n == fa::failAt.load(std::memory_order_relaxed))
        {
            fa::failAt.store(-1, std::memory_order_relaxed); // fail once, disarm
            throw std::bad_alloc();
        }
    }
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
