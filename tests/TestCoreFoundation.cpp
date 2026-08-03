// DSPark Tests - Core Foundation
// DspMath, AudioSpec, AudioBuffer, SpinLock, SpscQueue, DenormalGuard, SimdOps

#include "dspark_test.h"
#include "TestSignals.h"

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/SpinLock.h"
#include "../Core/SpscQueue.h"
#include "../Core/DenormalGuard.h"
#include "../Core/SimdOps.h"
#include "../Core/TruePeakDetector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

using namespace dspark;
using namespace dspark::test;

// ============================================================================
// DspMath
// ============================================================================

DSPARK_TEST(DspMath_decibelsToGain_0dB)
{
    EXPECT_NEAR(decibelsToGain(0.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(decibelsToGain(0.0), 1.0, 1e-12);
}

DSPARK_TEST(DspMath_decibelsToGain_minus6dB)
{
    float g = decibelsToGain(-6.0206f);
    EXPECT_NEAR(g, 0.5f, 0.001f);
}

DSPARK_TEST(DspMath_decibelsToGain_minusInfinity)
{
    EXPECT_NEAR(decibelsToGain(-100.0f), 0.0f, 1e-10f);
    EXPECT_NEAR(decibelsToGain(-200.0f), 0.0f, 1e-10f);
}

DSPARK_TEST(DspMath_gainToDecibels_unity)
{
    EXPECT_NEAR(gainToDecibels(1.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(gainToDecibels(1.0), 0.0, 1e-12);
}

DSPARK_TEST(DspMath_gainToDecibels_half)
{
    float dB = gainToDecibels(0.5f);
    EXPECT_NEAR(dB, -6.0206f, 0.01f);
}

DSPARK_TEST(DspMath_gainToDecibels_zero)
{
    EXPECT_NEAR(gainToDecibels(0.0f), -100.0f, 1e-6f);
}

DSPARK_TEST(DspMath_dB_roundtrip)
{
    // gain -> dB -> gain should be identity
    for (float g : { 0.001f, 0.1f, 0.5f, 1.0f, 2.0f, 10.0f })
    {
        float roundtrip = decibelsToGain(gainToDecibels(g));
        EXPECT_NEAR(roundtrip, g, g * 1e-5f);
    }
}

DSPARK_TEST(DspMath_fastTanh_zero)
{
    EXPECT_NEAR(fastTanh(0.0f), 0.0f, 1e-10f);
}

DSPARK_TEST(DspMath_fastTanh_accuracy)
{
    for (float x = -3.0f; x <= 3.0f; x += 0.1f)
    {
        float approx = fastTanh(x);
        float exact  = std::tanh(x);
        EXPECT_NEAR(approx, exact, 0.002f); // Pade [5,4]: max error < 0.05%
    }
}

DSPARK_TEST(DspMath_fastTanh_saturation)
{
    // fastTanh clamps its argument to [-3, 3] (the Pade approximant's valid range),
    // so it saturates to ~tanh(3) = 0.9951 rather than exactly 1 - by design, and
    // still a smooth bounded soft-clip. Verify it stays just under +/-1.
    EXPECT_NEAR(fastTanh(10.0f),  1.0f, 0.01f);
    EXPECT_NEAR(fastTanh(-10.0f), -1.0f, 0.01f);
}

DSPARK_TEST(DspMath_mapRange)
{
    EXPECT_NEAR(mapRange(0.5f, 0.0f, 1.0f, 0.0f, 100.0f), 50.0f, 1e-5f);
    EXPECT_NEAR(mapRange(0.0f, 0.0f, 1.0f, 20.0f, 20000.0f), 20.0f, 1e-5f);
    EXPECT_NEAR(mapRange(1.0f, 0.0f, 1.0f, 20.0f, 20000.0f), 20000.0f, 1e-3f);
}

DSPARK_TEST(DspMath_wrapPhase)
{
    float w = wrapPhase(twoPi<float> + 1.0f);
    EXPECT_NEAR(w, 1.0f, 1e-5f);

    float w2 = wrapPhase(-1.0f);
    EXPECT_GT(w2, 0.0f);
    EXPECT_LT(w2, twoPi<float>);
}

DSPARK_TEST(DspMath_constants)
{
    EXPECT_NEAR(pi<float>, 3.14159265f, 1e-5f);
    EXPECT_NEAR(twoPi<float>, 6.28318530f, 1e-5f);
    EXPECT_NEAR(sqrt2<float>, 1.41421356f, 1e-5f);
    EXPECT_NEAR(invSqrt2<float>, 0.70710678f, 1e-5f);
}

DSPARK_TEST(DspMath_fastPow10)
{
    EXPECT_NEAR(fastPow10(0.0f), 1.0f, 1e-5f);
    EXPECT_NEAR(fastPow10(1.0f), 10.0f, 1e-4f);
    EXPECT_NEAR(fastPow10(2.0f), 100.0f, 1e-3f);
}

DSPARK_TEST(DspMath_fastPow10_matches_pow_across_range)
{
    for (float x = -10.0f; x <= 10.0f; x += 0.037f)
    {
        const float exact = std::pow(10.0f, x);
        EXPECT_NEAR(fastPow10(x), exact, exact * 4e-6f);
    }
    for (double x = -12.0; x <= 12.0; x += 0.037)
    {
        const double exact = std::pow(10.0, x);
        EXPECT_NEAR(fastPow10(x), exact, exact * 1e-13);
    }
}

DSPARK_TEST(DspMath_fastLog_matches_log_across_range)
{
    for (double x : { 1e-6, 0.001, 0.02, 0.5, 0.7071, 1.5, 2.0, 10.0, 440.0, 48000.0, 1e9 })
    {
        const double exact = std::log(x);
        EXPECT_NEAR(fastLog(x), exact, std::max(std::abs(exact) * 3e-7, 3e-8));
    }
    EXPECT_NEAR(fastLog(1.0), 0.0, 1e-12);
    for (float x = 0.01f; x < 100.0f; x *= 1.07f)
    {
        const float exact = std::log(x);
        EXPECT_NEAR(fastLog(x), exact, std::max(std::abs(exact) * 2e-6f, 4e-7f));
    }
}

DSPARK_TEST(DspMath_wrapPhase_stays_in_range_for_large_phases)
{
    // Regression: for large phases the k * twoPi product can overshoot the
    // input, and the raw floor-based wrap returned a slightly NEGATIVE phase
    // (a lookup table indexed with it would read out of bounds).
    for (int k = -10000; k <= 10000; k += 7)
    {
        for (float delta : { -1e-4f, -1e-6f, 0.0f, 1e-6f, 1e-4f, 3.14f })
        {
            const float w = wrapPhase(static_cast<float>(k) * twoPi<float> + delta);
            EXPECT_TRUE(w >= 0.0f);
            EXPECT_LT(w, twoPi<float>);
        }
    }
    for (int k = -10000; k <= 10000; k += 7)
    {
        for (double delta : { -1e-9, 0.0, 1e-9, 3.14 })
        {
            const double w = wrapPhase(static_cast<double>(k) * twoPi<double> + delta);
            EXPECT_TRUE(w >= 0.0);
            EXPECT_LT(w, twoPi<double>);
        }
    }
}

// ============================================================================
// AudioSpec
// ============================================================================

DSPARK_TEST(AudioSpec_isValid_rejects_nan_and_nonpositive)
{
    EXPECT_TRUE((AudioSpec { 48000.0, 512, 2 }).isValid());
    EXPECT_FALSE((AudioSpec { std::numeric_limits<double>::quiet_NaN(), 512, 2 }).isValid());
    EXPECT_FALSE((AudioSpec { 0.0, 512, 2 }).isValid());
    EXPECT_FALSE((AudioSpec { -48000.0, 512, 2 }).isValid());
    EXPECT_FALSE((AudioSpec { 48000.0, 0, 2 }).isValid());
    EXPECT_FALSE((AudioSpec { 48000.0, -512, 2 }).isValid());
    EXPECT_FALSE((AudioSpec { 48000.0, 512, 0 }).isValid());
    EXPECT_FALSE((AudioSpec { 48000.0, 512, -2 }).isValid());
}

DSPARK_TEST(AudioSpec_equality_detects_any_field_change)
{
    const AudioSpec a { 48000.0, 512, 2 };
    EXPECT_TRUE(a == (AudioSpec { 48000.0, 512, 2 }));
    EXPECT_FALSE(a == (AudioSpec { 44100.0, 512, 2 }));
    EXPECT_FALSE(a == (AudioSpec { 48000.0, 256, 2 }));
    EXPECT_FALSE(a == (AudioSpec { 48000.0, 512, 1 }));
}

DSPARK_TEST(AudioSpec_defaults)
{
    AudioSpec spec {};
    // A default-constructed spec is intentionally empty (all zero) and therefore
    // invalid: callers must configure it explicitly before prepare().
    EXPECT_NEAR(spec.sampleRate, 0.0, 1e-10);
    EXPECT_EQ(spec.maxBlockSize, 0);
    EXPECT_EQ(spec.numChannels, 0);
    EXPECT_TRUE(!spec.isValid());
}

DSPARK_TEST(AudioSpec_designated_init)
{
    AudioSpec spec { .sampleRate = 96000.0, .maxBlockSize = 1024, .numChannels = 8 };
    EXPECT_NEAR(spec.sampleRate, 96000.0, 1e-10);
    EXPECT_EQ(spec.maxBlockSize, 1024);
    EXPECT_EQ(spec.numChannels, 8);
}

// ============================================================================
// AudioBuffer
// ============================================================================

DSPARK_TEST(AudioBuffer_resize_and_clear)
{
    AudioBuffer<float> buf;
    buf.resize(2, 512);
    EXPECT_EQ(buf.getNumChannels(), 2);
    EXPECT_EQ(buf.getNumSamples(), 512);

    // Should be cleared after resize
    for (int ch = 0; ch < 2; ++ch)
        for (int i = 0; i < 512; ++i)
            EXPECT_NEAR(buf.getChannel(ch)[i], 0.0f, 1e-10f);
}

DSPARK_TEST(AudioBuffer_alignment_32byte)
{
    AudioBuffer<float> buf;
    buf.resize(4, 1024);

    for (int ch = 0; ch < 4; ++ch)
    {
        auto addr = reinterpret_cast<std::uintptr_t>(buf.getChannel(ch));
        EXPECT_EQ(addr % 32, static_cast<std::uintptr_t>(0));
    }
}

DSPARK_TEST(AudioBuffer_write_and_read)
{
    AudioBuffer<float> buf;
    buf.resize(1, 4);
    float* ch0 = buf.getChannel(0);
    ch0[0] = 1.0f; ch0[1] = 2.0f; ch0[2] = 3.0f; ch0[3] = 4.0f;

    EXPECT_NEAR(buf.getChannel(0)[0], 1.0f, 1e-10f);
    EXPECT_NEAR(buf.getChannel(0)[3], 4.0f, 1e-10f);
}

DSPARK_TEST(AudioBuffer_move)
{
    AudioBuffer<float> a;
    a.resize(2, 256);
    a.getChannel(0)[0] = 42.0f;

    AudioBuffer<float> b = std::move(a);
    EXPECT_EQ(b.getNumChannels(), 2);
    EXPECT_EQ(b.getNumSamples(), 256);
    EXPECT_NEAR(b.getChannel(0)[0], 42.0f, 1e-10f);

    // Moved-from should be empty
    EXPECT_EQ(a.getNumChannels(), 0);
    EXPECT_EQ(a.getNumSamples(), 0);
}

DSPARK_TEST(AudioBuffer_double_template)
{
    AudioBuffer<double> buf;
    buf.resize(2, 128);
    buf.getChannel(0)[0] = 3.14159265358979;
    EXPECT_NEAR(buf.getChannel(0)[0], 3.14159265358979, 1e-14);
}

// ============================================================================
// AudioBufferView
// ============================================================================

DSPARK_TEST(AudioBufferView_from_buffer)
{
    auto tb = makeStereoBuffer(256);
    tb.fillSine(440.0f, static_cast<float>(kSampleRate));

    auto view = tb.view();
    EXPECT_EQ(view.getNumChannels(), 2);
    EXPECT_EQ(view.getNumSamples(), 256);
    EXPECT_FALSE(isSilent(view.getChannel(0), 256));
}

DSPARK_TEST(AudioBufferView_clear)
{
    auto tb = makeStereoBuffer(128);
    tb.fillNoise();
    EXPECT_FALSE(isSilent(tb.ch(0), 128));

    tb.view().clear();
    EXPECT_TRUE(isSilent(tb.ch(0), 128));
    EXPECT_TRUE(isSilent(tb.ch(1), 128));
}

DSPARK_TEST(AudioBufferView_applyGain)
{
    auto tb = makeMonoBuffer(128);
    generateDC(tb.ch(0), 128, 1.0f);

    tb.view().applyGain(0.5f);
    EXPECT_NEAR(tb.ch(0)[0], 0.5f, 1e-6f);
    EXPECT_NEAR(tb.ch(0)[127], 0.5f, 1e-6f);
}

DSPARK_TEST(AudioBufferView_getPeakLevel)
{
    auto tb = makeMonoBuffer(256);
    tb.fillSine(1000.0f, static_cast<float>(kSampleRate), 0.75f);
    float peak = tb.view().getPeakLevel();
    EXPECT_GT(peak, 0.7f);
    EXPECT_LT(peak, 0.76f);
}

DSPARK_TEST(AudioBufferView_copyFrom)
{
    auto src = makeMonoBuffer(64);
    auto dst = makeMonoBuffer(64);
    generateDC(src.ch(0), 64, 0.99f);

    dst.view().copyFrom(src.view());
    EXPECT_NEAR(dst.ch(0)[0], 0.99f, 1e-6f);
    EXPECT_NEAR(dst.ch(0)[63], 0.99f, 1e-6f);
}

DSPARK_TEST(AudioBufferView_addFrom)
{
    auto a = makeMonoBuffer(64);
    auto b = makeMonoBuffer(64);
    generateDC(a.ch(0), 64, 0.3f);
    generateDC(b.ch(0), 64, 0.2f);

    a.view().addFrom(b.view());
    EXPECT_NEAR(a.ch(0)[0], 0.5f, 1e-6f);
}

DSPARK_TEST(AudioBufferView_subView)
{
    auto tb = makeMonoBuffer(256);
    for (int i = 0; i < 256; ++i)
        tb.ch(0)[i] = static_cast<float>(i);

    auto sub = tb.view().getSubView(100, 50);
    EXPECT_EQ(sub.getNumSamples(), 50);
    EXPECT_NEAR(sub.getChannel(0)[0], 100.0f, 1e-6f);
    EXPECT_NEAR(sub.getChannel(0)[49], 149.0f, 1e-6f);
}

// ============================================================================
// SpinLock
// ============================================================================

DSPARK_TEST(SpinLock_std_lockable_interop)
{
    SpinLock lock;
    {
        std::unique_lock<SpinLock> guard(lock, std::try_to_lock);
        EXPECT_TRUE(guard.owns_lock());
        EXPECT_FALSE(lock.tryLock()); // held by the guard
    }
    EXPECT_TRUE(lock.tryLock());      // released on guard destruction
    lock.unlock();
}

DSPARK_TEST(SpinLock_mutual_exclusion_under_contention)
{
    SpinLock lock;
    long counter = 0; // deliberately non-atomic: the lock is the only guard
    constexpr int kIters = 100000;
    auto worker = [&]
    {
        for (int i = 0; i < kIters; ++i)
        {
            SpinLock::ScopedLock guard(lock);
            ++counter;
        }
    };
    std::thread t1(worker);
    std::thread t2(worker);
    t1.join();
    t2.join();
    EXPECT_EQ(counter, 2L * kIters);
}

DSPARK_TEST(SpinLock_lock_unlock)
{
    SpinLock lock;
    lock.lock();
    // If we get here, lock succeeded
    lock.unlock();
    EXPECT_TRUE(true);
}

DSPARK_TEST(SpinLock_tryLock)
{
    SpinLock lock;
    EXPECT_TRUE(lock.tryLock());
    // Lock is held, second tryLock should fail
    EXPECT_FALSE(lock.tryLock());
    lock.unlock();
    // After unlock, should succeed again
    EXPECT_TRUE(lock.tryLock());
    lock.unlock();
}

DSPARK_TEST(SpinLock_ScopedLock)
{
    SpinLock lock;
    {
        SpinLock::ScopedLock guard(lock);
        // Lock held - tryLock should fail
        EXPECT_FALSE(lock.tryLock());
    }
    // Lock released - tryLock should succeed
    EXPECT_TRUE(lock.tryLock());
    lock.unlock();
}

DSPARK_TEST(SpinLock_ScopedTryLock)
{
    SpinLock lock;
    {
        SpinLock::ScopedTryLock guard(lock);
        EXPECT_TRUE(guard.isLocked());
    }
    // Verify it was released
    EXPECT_TRUE(lock.tryLock());
    lock.unlock();
}

// ============================================================================
// SpscQueue
// ============================================================================

DSPARK_TEST(SpscQueue_push_pop_FIFO)
{
    SpscQueue<int, 8> q;
    EXPECT_TRUE(q.push(10));
    EXPECT_TRUE(q.push(20));
    EXPECT_TRUE(q.push(30));

    int val = 0;
    EXPECT_TRUE(q.pop(val)); EXPECT_EQ(val, 10);
    EXPECT_TRUE(q.pop(val)); EXPECT_EQ(val, 20);
    EXPECT_TRUE(q.pop(val)); EXPECT_EQ(val, 30);
}

DSPARK_TEST(SpscQueue_empty_pop_fails)
{
    SpscQueue<int, 4> q;
    int val = 0;
    EXPECT_FALSE(q.pop(val));
    EXPECT_TRUE(q.empty());
}

DSPARK_TEST(SpscQueue_full_push_fails)
{
    SpscQueue<int, 4> q;
    // Capacity 4, but usable = 3 (one slot reserved for full detection)
    EXPECT_TRUE(q.push(1));
    EXPECT_TRUE(q.push(2));
    EXPECT_TRUE(q.push(3));
    EXPECT_FALSE(q.push(4)); // Full
}

DSPARK_TEST(SpscQueue_full_and_empty_boundaries)
{
    SpscQueue<int, 8> q;             // usable capacity: 7
    EXPECT_EQ(q.capacity(), static_cast<std::size_t>(7));
    for (int i = 0; i < 7; ++i)
        EXPECT_TRUE(q.push(i));
    EXPECT_FALSE(q.push(99));        // full: the reserved slot keeps states distinct
    int v = -1;
    for (int i = 0; i < 7; ++i)
    {
        EXPECT_TRUE(q.pop(v));
        EXPECT_EQ(v, i);             // strict FIFO
    }
    EXPECT_FALSE(q.pop(v));          // empty again
    EXPECT_TRUE(q.push(42));         // reusable after wrapping
    EXPECT_TRUE(q.pop(v));
    EXPECT_EQ(v, 42);
}

DSPARK_TEST(SpscQueue_threaded_fifo_no_loss)
{
    SpscQueue<int, 64> q;
    constexpr int kCount = 200000;
    std::atomic<bool> failed { false };
    std::thread producer([&]
    {
        for (int i = 0; i < kCount && !failed.load(std::memory_order_relaxed); ++i)
            while (!q.push(i) && !failed.load(std::memory_order_relaxed)) {}
    });
    int expected = 0;
    int v = 0;
    while (expected < kCount)
    {
        if (q.pop(v))
        {
            if (v != expected) { failed.store(true); break; }
            ++expected;
        }
    }
    producer.join();
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(expected, kCount);     // every element arrived, exactly once, in order
}

DSPARK_TEST(SpscQueue_sizeApprox)
{
    SpscQueue<int, 8> q;
    EXPECT_EQ(q.sizeApprox(), static_cast<std::size_t>(0));
    EXPECT_TRUE(q.push(1));
    EXPECT_TRUE(q.push(2));
    EXPECT_EQ(q.sizeApprox(), static_cast<std::size_t>(2));
    int v;
    EXPECT_TRUE(q.pop(v));
    EXPECT_EQ(q.sizeApprox(), static_cast<std::size_t>(1));
}

DSPARK_TEST(SpscQueue_struct_type)
{
    struct Params { float freq; float gain; };
    SpscQueue<Params, 4> q;
    EXPECT_TRUE(q.push({ 440.0f, 0.5f }));

    Params p {};
    EXPECT_TRUE(q.pop(p));
    EXPECT_NEAR(p.freq, 440.0f, 1e-6f);
    EXPECT_NEAR(p.gain, 0.5f, 1e-6f);
}

// ============================================================================
// DenormalGuard
// ============================================================================

namespace {

// FLT_MIN * 0.5 is subnormal under default IEEE arithmetic and exactly 0.0f
// when the CPU flushes denormal results (FTZ). Every access is volatile so
// the multiply cannot be hoisted or sunk across the guard's control-register
// writes: optimisers reorder plain FP ops across asm/intrinsics (the loads
// pin the inputs after construction, the store pins the result before
// destruction).
inline float denormalProbe()
{
    volatile float smallest = std::numeric_limits<float>::min();
    volatile float half     = 0.5f;
    volatile float result   = smallest * half;
    return result;
}

} // namespace

DSPARK_TEST(DenormalGuard_flushes_denormals_and_restores_state)
{
    // The private suite runs on x86/x64, where the guard must be real.
    EXPECT_TRUE(DenormalGuard::isActive());

    // If this fails, an earlier test leaked FTZ/DAZ into the thread: every
    // guard must restore the exact state it captured.
    const float before = denormalProbe();
    EXPECT_TRUE(before != 0.0f);

    float inside = -1.0f, insideNested = -1.0f, afterNested = -1.0f;
    {
        DenormalGuard guard;
        inside = denormalProbe();
        {
            DenormalGuard nested;
            insideNested = denormalProbe();
        }
        afterNested = denormalProbe(); // outer guard must still be in force
    }
    const float after = denormalProbe();

    EXPECT_EQ(inside, 0.0f);       // flushed inside the guard
    EXPECT_EQ(insideNested, 0.0f); // nested guard: still flushed
    EXPECT_EQ(afterNested, 0.0f);  // inner destructor restored FTZ-on, not IEEE
    EXPECT_TRUE(after == before);  // outer destructor restored the original mode
}

// ============================================================================
// SimdOps - every kernel vs a scalar reference, on the active SIMD path.
// Counts cover empty, sub-vector, vector-boundary and unrolled-loop cases so
// both the SIMD body and the scalar tail of each kernel are exercised.
// ============================================================================

namespace {

// Deterministic LCG so failures reproduce identically on every platform.
inline float simdTestRand(uint32_t& s)
{
    s = s * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int>((s >> 16) & 0x7FFFu)) / 16384.0f - 1.0f;
}

constexpr int kSimdCounts[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17,
                                31, 32, 33, 63, 64, 65, 100, 257 };
constexpr int kSimdMax = 257;

} // namespace

DSPARK_TEST(SimdOps_elementwise_kernels_match_scalar)
{
    uint32_t seed = 0xD512D512u;
    for (int count : kSimdCounts)
    {
        float  srcF[kSimdMax + 1], dstF[kSimdMax + 1], refF[kSimdMax + 1], outF[kSimdMax + 1];
        double srcD[kSimdMax + 1], dstD[kSimdMax + 1], refD[kSimdMax + 1], outD[kSimdMax + 1];
        for (int i = 0; i <= kSimdMax; ++i)
        {
            srcF[i] = simdTestRand(seed);
            dstF[i] = simdTestRand(seed);
            srcD[i] = static_cast<double>(simdTestRand(seed));
            dstD[i] = static_cast<double>(simdTestRand(seed));
            refF[i] = dstF[i];
            refD[i] = dstD[i];
            outF[i] = 77.0f;
            outD[i] = 77.0;
        }

        // addWithGain (tolerance admits the fused-FMA rounding difference)
        simd::addWithGain(dstF, srcF, 0.7f, count);
        simd::addWithGain(dstD, srcD, 0.7, count);
        for (int i = 0; i < count; ++i) refF[i] += srcF[i] * 0.7f;
        for (int i = 0; i < count; ++i) refD[i] += srcD[i] * 0.7;
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstF[i], refF[i], 1e-6f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstD[i], refD[i], 1e-14);
        EXPECT_EQ(dstF[count], refF[count]); // past-the-end must be untouched
        EXPECT_EQ(dstD[count], refD[count]);

        // applyGain
        simd::applyGain(dstF, 0.5f, count);
        simd::applyGain(dstD, 0.5, count);
        for (int i = 0; i < count; ++i) refF[i] *= 0.5f;
        for (int i = 0; i < count; ++i) refD[i] *= 0.5;
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstF[i], refF[i], 1e-7f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstD[i], refD[i], 1e-15);

        // add
        simd::add(dstF, srcF, count);
        simd::add(dstD, srcD, count);
        for (int i = 0; i < count; ++i) refF[i] += srcF[i];
        for (int i = 0; i < count; ++i) refD[i] += srcD[i];
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstF[i], refF[i], 1e-7f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstD[i], refD[i], 1e-15);

        // multiply (out-of-place)
        simd::multiply(outF, dstF, srcF, count);
        simd::multiply(outD, dstD, srcD, count);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(outF[i], dstF[i] * srcF[i], 1e-7f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(outD[i], dstD[i] * srcD[i], 1e-15);
        EXPECT_EQ(outF[count], 77.0f);
        EXPECT_EQ(outD[count], 77.0);

        // copyWithGain (out-of-place)
        simd::copyWithGain(outF, srcF, -1.25f, count);
        simd::copyWithGain(outD, srcD, -1.25, count);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(outF[i], srcF[i] * -1.25f, 1e-7f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(outD[i], srcD[i] * -1.25, 1e-15);
    }
}

DSPARK_TEST(SimdOps_reductions_match_scalar)
{
    uint32_t seed = 0xACC1ADE5u;
    for (int count : kSimdCounts)
    {
        float  aF[kSimdMax], bF[kSimdMax];
        double aD[kSimdMax], bD[kSimdMax];
        for (int i = 0; i < kSimdMax; ++i)
        {
            aF[i] = simdTestRand(seed);
            bF[i] = simdTestRand(seed);
            aD[i] = static_cast<double>(simdTestRand(seed));
            bD[i] = static_cast<double>(simdTestRand(seed));
        }

        // peakLevel: abs and max involve no rounding, so the match is exact
        float refPeakF = 0.0f;
        for (int i = 0; i < count; ++i)
        {
            const float v = aF[i] < 0.0f ? -aF[i] : aF[i];
            if (v > refPeakF) refPeakF = v;
        }
        EXPECT_EQ(simd::peakLevel(aF, count), refPeakF);

        double refPeakD = 0.0;
        for (int i = 0; i < count; ++i)
        {
            const double v = aD[i] < 0.0 ? -aD[i] : aD[i];
            if (v > refPeakD) refPeakD = v;
        }
        EXPECT_EQ(simd::peakLevel(aD, count), refPeakD);

        // dotProduct / sumOfSquares vs double-accumulated references
        double refDotF = 0.0, refDotD = 0.0, refSosF = 0.0, refSosD = 0.0;
        for (int i = 0; i < count; ++i)
        {
            refDotF += static_cast<double>(aF[i]) * static_cast<double>(bF[i]);
            refDotD += aD[i] * bD[i];
            refSosF += static_cast<double>(aF[i]) * static_cast<double>(aF[i]);
            refSosD += aD[i] * aD[i];
        }
        EXPECT_NEAR(simd::dotProduct(aF, bF, count), static_cast<float>(refDotF), 2e-4f);
        EXPECT_NEAR(simd::dotProduct(aD, bD, count), refDotD, 1e-12);
        EXPECT_NEAR(simd::sumOfSquares(aF, count), static_cast<float>(refSosF), 2e-4f);
        EXPECT_NEAR(simd::sumOfSquares(aD, count), refSosD, 1e-12);
    }
}

DSPARK_TEST(SimdOps_peak_level_ignores_nan)
{
    const float  nanF = std::numeric_limits<float>::quiet_NaN();
    const double nanD = std::numeric_limits<double>::quiet_NaN();

    float bufF[33];
    for (int i = 0; i < 33; ++i) bufF[i] = 0.01f * static_cast<float>(i % 7) - 0.02f;
    bufF[4]  = -1.5f; // true peak, seen BEFORE the first NaN in its lane
    bufF[6]  = nanF;
    bufF[19] = nanF;
    bufF[32] = nanF;  // scalar-tail NaN
    EXPECT_TRUE(simd::peakLevel(bufF, 33) == 1.5f);

    bufF[0]  = nanF;  // NaN before any finite sample
    bufF[10] = 2.5f;  // and a larger peak AFTER a NaN must still be seen
    EXPECT_TRUE(simd::peakLevel(bufF, 33) == 2.5f);

    double bufD[17];
    for (int i = 0; i < 17; ++i) bufD[i] = 0.01 * static_cast<double>(i % 5) - 0.02;
    bufD[2]  = 1.25;
    bufD[3]  = nanD;
    bufD[16] = nanD;
    EXPECT_TRUE(simd::peakLevel(bufD, 17) == 1.25);
}

DSPARK_TEST(SimdOps_complex_mul_accum_matches_scalar)
{
    uint32_t seed = 0xC011BEEFu;
    for (int bins : { 0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 33, 257 })
    {
        float  aF[2 * kSimdMax], bF[2 * kSimdMax], accF[2 * kSimdMax], acc0F[2 * kSimdMax];
        double aD[2 * kSimdMax], bD[2 * kSimdMax], accD[2 * kSimdMax], acc0D[2 * kSimdMax];
        for (int i = 0; i < 2 * kSimdMax; ++i)
        {
            aF[i]   = simdTestRand(seed);
            bF[i]   = simdTestRand(seed);
            accF[i] = simdTestRand(seed);
            aD[i]   = static_cast<double>(simdTestRand(seed));
            bD[i]   = static_cast<double>(simdTestRand(seed));
            accD[i] = static_cast<double>(simdTestRand(seed));
            acc0F[i] = accF[i];
            acc0D[i] = accD[i];
        }

        simd::complexMulAccum(accF, aF, bF, bins);
        simd::complexMulAccum(accD, aD, bD, bins);

        for (int k = 0; k < bins; ++k)
        {
            const double reF = static_cast<double>(acc0F[2 * k])
                             + static_cast<double>(aF[2 * k]) * bF[2 * k]
                             - static_cast<double>(aF[2 * k + 1]) * bF[2 * k + 1];
            const double imF = static_cast<double>(acc0F[2 * k + 1])
                             + static_cast<double>(aF[2 * k]) * bF[2 * k + 1]
                             + static_cast<double>(aF[2 * k + 1]) * bF[2 * k];
            EXPECT_NEAR(accF[2 * k],     static_cast<float>(reF), 1e-5f);
            EXPECT_NEAR(accF[2 * k + 1], static_cast<float>(imF), 1e-5f);

            const double reD = acc0D[2 * k]     + aD[2 * k] * bD[2 * k] - aD[2 * k + 1] * bD[2 * k + 1];
            const double imD = acc0D[2 * k + 1] + aD[2 * k] * bD[2 * k + 1] + aD[2 * k + 1] * bD[2 * k];
            EXPECT_NEAR(accD[2 * k],     reD, 1e-13);
            EXPECT_NEAR(accD[2 * k + 1], imD, 1e-13);
        }
        if (bins < kSimdMax)
        {
            EXPECT_EQ(accF[2 * bins], acc0F[2 * bins]); // past-the-end untouched
            EXPECT_EQ(accD[2 * bins], acc0D[2 * bins]);
        }
    }
}

DSPARK_TEST(SimdOps_gain_ramps_match_scalar)
{
    uint32_t seed = 0x4A3F4A3Fu;
    const float  gsF = 0.25f, geF = 1.75f;
    const double gsD = 0.25,  geD = 1.75;

    for (int count : kSimdCounts)
    {
        float  dataF[kSimdMax + 1], refF[kSimdMax + 1], srcF[kSimdMax + 1], dstF[kSimdMax + 1], refDstF[kSimdMax + 1];
        double dataD[kSimdMax + 1], refD[kSimdMax + 1], srcD[kSimdMax + 1], dstD[kSimdMax + 1], refDstD[kSimdMax + 1];
        for (int i = 0; i <= kSimdMax; ++i)
        {
            dataF[i] = simdTestRand(seed);
            srcF[i]  = simdTestRand(seed);
            dstF[i]  = simdTestRand(seed);
            dataD[i] = static_cast<double>(simdTestRand(seed));
            srcD[i]  = static_cast<double>(simdTestRand(seed));
            dstD[i]  = static_cast<double>(simdTestRand(seed));
            refF[i] = dataF[i]; refDstF[i] = dstF[i];
            refD[i] = dataD[i]; refDstD[i] = dstD[i];
        }

        simd::applyGainRamp(dataF, gsF, geF, count);
        simd::applyGainRamp(dataD, gsD, geD, count);
        simd::addWithGainRamp(dstF, srcF, gsF, geF, count);
        simd::addWithGainRamp(dstD, srcD, gsD, geD, count);

        if (count > 0)
        {
            const float  stepF = (geF - gsF) / static_cast<float>(count);
            const double stepD = (geD - gsD) / static_cast<double>(count);
            for (int i = 0; i < count; ++i)
            {
                const float  gF = gsF + stepF * static_cast<float>(i);
                const double gD = gsD + stepD * static_cast<double>(i);
                refF[i]    *= gF;
                refD[i]    *= gD;
                refDstF[i] += srcF[i] * gF;
                refDstD[i] += srcD[i] * gD;
            }
        }
        // Tolerance covers the running-gain accumulator drift of the SIMD path.
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dataF[i], refF[i], 5e-5f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dataD[i], refD[i], 1e-13);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstF[i], refDstF[i], 5e-5f);
        for (int i = 0; i < count; ++i) EXPECT_NEAR(dstD[i], refDstD[i], 1e-13);
        EXPECT_EQ(dataF[count], refF[count]); // past-the-end untouched (and count 0 = no-op)
        EXPECT_EQ(dataD[count], refD[count]);
        EXPECT_EQ(dstF[count], refDstF[count]);
        EXPECT_EQ(dstD[count], refDstD[count]);
    }

    // A flat ramp must behave exactly like applyGain
    float flatA[65], flatB[65];
    for (int i = 0; i < 65; ++i) { flatA[i] = simdTestRand(seed); flatB[i] = flatA[i]; }
    simd::applyGainRamp(flatA, 0.6f, 0.6f, 65);
    simd::applyGain(flatB, 0.6f, 65);
    for (int i = 0; i < 65; ++i) EXPECT_NEAR(flatA[i], flatB[i], 1e-7f);
}

// ============================================================================
// AudioBuffer / AudioBufferView - audit regressions
// ============================================================================

DSPARK_TEST(AudioBufferView_const_view_reports_peak)
{
    AudioBuffer<float> buf;
    buf.resize(2, 64);
    buf.getChannel(0)[10] = -0.75f;
    buf.getChannel(1)[20] = 0.5f;

    const AudioBuffer<float>& constRef = buf;
    AudioBufferView<const float> constView = constRef.toView();
    EXPECT_NEAR(constView.getPeakLevel(), 0.75f, 0.0f);
    EXPECT_NEAR(buf.toView().getPeakLevel(), 0.75f, 0.0f);
}

DSPARK_TEST(AudioBuffer_zero_sample_resize_is_safe)
{
    AudioBuffer<float> buf;
    buf.resize(2, 0);       // degenerate: no storage at all
    buf.clear();            // must not touch memory (UBSan-clean)
    auto v = buf.toView();
    EXPECT_EQ(v.getNumChannels(), 2);
    EXPECT_EQ(v.getNumSamples(), 0);
    EXPECT_NEAR(v.getPeakLevel(), 0.0f, 0.0f);

    buf.resize(2, 32);      // grows into a real allocation afterwards
    buf.getChannel(0)[31] = 1.0f;
    EXPECT_NEAR(buf.toView().getPeakLevel(), 1.0f, 0.0f);

    buf.resize(0, 128);     // zero channels is equally inert
    buf.clear();
    EXPECT_EQ(buf.getNumChannels(), 0);
}

// AddressSanitizer replaces the allocator, and its answer to a request it
// cannot serve is its own, not the platform's: by default it reports an
// out-of-memory error and takes the process down, and with
// allocator_may_return_null=1 it returns null straight out of operator new
// rather than throwing, which the C++ allocation contract does not allow.
// Either way the case below has nothing left to observe, so it steps aside.
#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define DSPARK_TEST_UNDER_ASAN 1
#  endif
#  if __has_feature(thread_sanitizer)
#    define DSPARK_TEST_UNDER_TSAN 1
#  endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#  define DSPARK_TEST_UNDER_ASAN 1
#endif
// ThreadSanitizer replaces the allocator the same way and is equally unable
// to honour the C++ allocation contract: its throwing operator new NEVER
// throws std::bad_alloc. A request past its 1 TiB allocator ceiling dies
// with a fatal allocation-size-too-big report (NORETURN, independent of
// halt_on_error), and allocator_may_return_null=1 merely converts that into
// a null return which the throwing-new interceptor then turns into an
// equally fatal out-of-memory report (GCC 13 libsanitizer,
// tsan/tsan_new_delete.cpp OPERATOR_NEW_BODY, tsan/tsan_mman.cpp
// user_alloc_internal, sanitizer_common/sanitizer_allocator_report.cpp).
// CI run 30712892449 aborted the whole TSan suite on exactly this case, so
// it steps aside under TSan too. It still runs fully on every normal build.
#if defined(__SANITIZE_THREAD__)
#  define DSPARK_TEST_UNDER_TSAN 1
#endif

DSPARK_TEST(AudioBuffer_recovers_after_failed_allocation)
{
#if defined(DSPARK_TEST_UNDER_ASAN) || defined(DSPARK_TEST_UNDER_TSAN)
    return;
#else
    // After a throwing resize the buffer must stay coherent: the next, smaller
    // resize has to re-allocate cleanly instead of trusting a stale capacity
    // over a null base pointer.
    //
    // Provoking that failure portably takes care. The request must be refused
    // as a RESERVATION, not discovered when the pages are written, because
    // resize() ends in a full zero-fill. 16 channels of INT_MAX doubles is
    // 256 GiB: Windows refuses it on commit charge and Linux refuses it on its
    // overcommit heuristic, but macOS grants the reservation and the zero-fill
    // that follows then takes the process down - measured as a killed
    // subprocess on the macOS runner, with no output and no exception.
    // 16384 channels of INT_MAX doubles is 281 TiB, past the 128 TiB of
    // address space a 64-bit user process has at all, so operator new has to
    // fail everywhere before a single page is touched.
    constexpr int kUnservableChannels = 16384;
    auto buf = std::make_unique<AudioBuffer<double, kUnservableChannels>>();

    buf->resize(2, 256);
    bool threw = false;
    try
    {
        buf->resize(kUnservableChannels, std::numeric_limits<int>::max());
    }
    catch (const std::bad_alloc&)
    {
        threw = true;
    }
    EXPECT_TRUE(threw);

    buf->resize(2, 64);
    buf->getChannel(1)[63] = 0.25;
    EXPECT_NEAR(buf->toView().getPeakLevel(), 0.25, 0.0);
#endif
}

DSPARK_TEST(AudioBuffer_shrink_reuses_allocation)
{
    AudioBuffer<float> buf;
    buf.resize(2, 1024);
    buf.getChannel(0)[0] = 0.9f;
    float* base = buf.getChannel(0);

    buf.resize(2, 128);     // fits in capacity: same storage, contents cleared
    EXPECT_TRUE(buf.getChannel(0) == base);
    EXPECT_EQ(buf.getNumSamples(), 128);
    EXPECT_NEAR(buf.toView().getPeakLevel(), 0.0f, 0.0f);
}

// ============================================================================
// TruePeakDetector - inter-sample peak recovery (BS.1770-5 interpolator)
// ============================================================================

DSPARK_TEST(TruePeakDetector_finds_intersample_peak_of_offset_sine)
{
    // fs/4 sine sampled at +45 degrees: every sample sits at |1/sqrt(2)|
    // (-3 dBFS) yet the reconstructed waveform peaks at 1.0 BETWEEN samples.
    // The Annex 2 interpolator must report ~0 dBTP, not -3 dBTP.
    TruePeakDetector<float> det;
    det.reset();
    float sampleMax = 0.0f, truePeak = 0.0f;
    for (int i = 0; i < 512; ++i)
    {
        const float v = std::sin(halfPi<float> * static_cast<float>(i) + pi<float> / 4.0f);
        sampleMax = std::max(sampleMax, std::abs(v));
        truePeak  = std::max(truePeak, det.processSample(v, 0));
    }
    EXPECT_NEAR(sampleMax, 0.70711f, 1e-4f); // raw samples never exceed -3 dBFS
    EXPECT_GT(truePeak, 0.98f);              // interpolator recovers ~0 dBTP
    EXPECT_LT(truePeak, 1.02f);

    // Channel isolation: a hot stream on channel 1 must not leak into ch 0.
    TruePeakDetector<float, 2> det2;
    for (int i = 0; i < 64; ++i)
        (void) det2.processSample(1.0f, 1);
    EXPECT_LT(det2.processSample(0.0f, 0), 1e-6f);
}

DSPARK_TEST(AudioBuffer_move_leaves_source_reusable)
{
    AudioBuffer<float> a;
    a.resize(2, 64);
    a.getChannel(0)[5] = 0.5f;

    AudioBuffer<float> b = std::move(a);
    EXPECT_EQ(b.getNumSamples(), 64);
    EXPECT_NEAR(b.toView().getPeakLevel(), 0.5f, 0.0f);
    EXPECT_EQ(a.getNumSamples(), 0);

    a.resize(1, 16);        // the moved-from buffer is fully usable again
    a.getChannel(0)[0] = 1.0f;
    EXPECT_NEAR(a.toView().getPeakLevel(), 1.0f, 0.0f);

    AudioBuffer<float> c;
    c.resize(1, 8);
    c = std::move(b);       // move assignment releases c's own storage
    EXPECT_NEAR(c.toView().getPeakLevel(), 0.5f, 0.0f);
}

// ============================================================================
// Lock-free guarantee for every atomic word type used on an audio-visible
// path in the Core/ and Analysis/ headers that publish across threads.
// A platform where any of these takes a lock would silently violate the
// allocation/lock-free audio-thread contract; fail loudly here instead.
// (float/double were already pinned by FIR_double_instantiation_lockfree_
// and_reprepare; this adds the integer/bool/enum-underlying words:
// seq counters, dirty flags, indices, packed 64-bit readouts.)
DSPARK_TEST(swept_atomic_word_types_are_lock_free)
{
    EXPECT_TRUE(std::atomic<bool>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<int>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<unsigned>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<std::size_t>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<std::uint32_t>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<std::int64_t>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<std::uint64_t>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<float>::is_always_lock_free);
    EXPECT_TRUE(std::atomic<double>::is_always_lock_free);
    // std::atomic_flag (SpinLock) is lock-free by definition ([atomics.flag]).
}
