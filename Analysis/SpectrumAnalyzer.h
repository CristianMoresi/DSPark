// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectrumAnalyzer.h
 * @brief Real-time FFT-based spectrum analyser for audio visualisation.
 *
 * Handles the complete pipeline: windowing -> FFT -> single-sided magnitude
 * (amplitude-calibrated: a full-scale sine reads 0 dB) -> per-bin exponential
 * smoothing -> dB conversion -> optional peak hold with dB/s decay.
 *
 * @note **Performance:** the FFT and transcendental math run synchronously
 * inside pushSamples() whenever a hop completes. For large FFT sizes (>2048)
 * on tight audio callbacks, consider a lock-free queue architecture where the
 * GUI thread pulls samples and computes the FFT independently.
 *
 * Threading:
 * - prepare() / reset(): setup thread (not concurrent with the audio or
 *   reader threads; reset() rewrites the reader-visible slots).
 * - pushSamples(): audio thread (stream owner). No-op before prepare().
 * - getMagnitudesDb() / getPeakHoldDb() / isNewDataReady(): ONE reader
 *   thread (GUI). Wait-free triple-buffer hand-off; never blocks the writer.
 *   Pointer-validity contract: each getter adopts the freshest published
 *   frame and copies it into reader-private snapshot storage; the returned
 *   pointer refers to that snapshot, NEVER into the hand-off slots. Its
 *   contents are overwritten only by the reader's own next call to the SAME
 *   getter, and the pointer itself is invalidated only by prepare(). No
 *   other thread ever writes through a returned pointer, so the reader may
 *   hold it across frames (the values simply go stale). A consecutive
 *   getMagnitudesDb()/getPeakHoldDb() pair may represent adjacent analysis
 *   frames; each snapshot is internally frame-coherent.
 * - setSmoothing() / setPeakDecay() / setPeakHoldEnabled() / setFloorDb():
 *   control thread (ONE non-audio writer, per the framework's SPSC thread
 *   model; single-word relaxed atomics, picked up once per analysis frame).
 *   Non-finite values are ignored.
 *
 * Dependencies: DspMath.h, FFT.h, WindowFunctions.h.
 *
 * @code
 * dspark::SpectrumAnalyzer<float> analyzer;
 * analyzer.prepare(48000.0, 2048);  // 48 kHz, 2048-point FFT
 *
 * // In audio callback:
 * analyzer.pushSamples(buffer.getChannel(0), numSamples);
 *
 * // In GUI paint (Timer at 60Hz):
 * if (analyzer.isNewDataReady()) {
 *     const float* spectrum = analyzer.getMagnitudesDb();
 *     // Draw spectrum...
 * }
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class SpectrumAnalyzer
 * @brief Real-time FFT spectrum analyser with per-bin smoothing and peak hold.
 *
 * The magnitude smoothing is a per-frame exponential (one analysis frame per
 * hop = fftSize/2 samples), so its settling TIME depends on fftSize and the
 * sample rate: frameRate = 2 * sampleRate / fftSize. The peak-hold decay is
 * specified in dB/second and is rate-invariant.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SpectrumAnalyzer
{
public:
    /** @brief Available window types for the FFT analysis. */
    enum class WindowType
    {
        Hann,           ///< Default. Good general-purpose choice.
        Hamming,        ///< Slightly better side lobe rejection.
        Blackman,       ///< High dynamic range.
        BlackmanHarris, ///< Highest side lobe rejection.
        FlatTop,        ///< Amplitude-accurate measurement.
        Rectangular     ///< No windowing (transient analysis).
    };

    /**
     * @brief Prepares the analyser and allocates all necessary buffers.
     *
     * Release-safe: a non-finite or non-positive sample rate is ignored
     * (conservative no-op); fftSize is clamped to [256, 16384] and rounded
     * UP to the next power of two; an out-of-range window enum falls back
     * to Hann. May allocate (setup thread only). If an allocation throws,
     * the analyser is left unprepared (pushSamples becomes a no-op) rather
     * than half-configured.
     *
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size (power of two, 256 to 16384).
     * @param windowType Window function to use (default: Hann).
     */
    void prepare(double sampleRate, int fftSize = 2048, WindowType windowType = WindowType::Hann)
    {
        assert(fftSize >= 256 && fftSize <= 16384 && (fftSize & (fftSize - 1)) == 0);

        if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
            return;

        // Release-safe size sanitation (the assert only guards debug builds;
        // an unvalidated size would make FFTReal throw at stream time).
        fftSize = std::clamp(fftSize, 256, 16384);
        int pow2 = 256;
        while (pow2 < fftSize) pow2 <<= 1;
        fftSize = pow2;

        fft_.reset(); // gate OFF: pushSamples() is a no-op while rebuilding

        sampleRate_ = sampleRate;
        fftSize_ = fftSize;
        numBins_ = fftSize / 2 + 1;
        hopSize_ = fftSize / 2; // 50% overlap
        windowType_ = windowType;

        window_.resize(static_cast<size_t>(fftSize));
        generateWindow(windowType);

        windowGain_ = WindowFunctions<T>::coherentGain(window_.data(), fftSize);
        if (windowGain_ < T(0.001)) windowGain_ = T(1);
        invGain_ = T(2) / (static_cast<T>(fftSize_) * windowGain_);

        inputRing_.assign(static_cast<size_t>(fftSize), T(0));
        fftBuffer_.resize(static_cast<size_t>(fftSize));
        freqBuffer_.resize(static_cast<size_t>(fftSize + 2));

        // Unified DSP State
        const T floorDb = floorDb_.load(std::memory_order_relaxed);
        magnitudesState_.assign(static_cast<size_t>(numBins_), T(0));
        peakState_.assign(static_cast<size_t>(numBins_), floorDb);

        // Triple buffer for tear-free cross-thread reading (relaxed-atomic
        // words; ordering comes from the pendingSlot_ RMWs, see below).
        for (auto& slot : outSlots_)
        {
            slot.magnitudesDb = std::make_unique<std::atomic<T>[]>(static_cast<size_t>(numBins_));
            slot.peakDb = std::make_unique<std::atomic<T>[]>(static_cast<size_t>(numBins_));
            for (int k = 0; k < numBins_; ++k)
            {
                slot.magnitudesDb[static_cast<size_t>(k)].store(floorDb, std::memory_order_relaxed);
                slot.peakDb[static_cast<size_t>(k)].store(floorDb, std::memory_order_relaxed);
            }
        }
        // Reader-private snapshot storage backing the getter return pointers.
        magSnapshot_.assign(static_cast<size_t>(numBins_), floorDb);
        peakSnapshot_.assign(static_cast<size_t>(numBins_), floorDb);
        writeSlot_ = 0;
        readSlot_  = 1;
        pendingSlot_.store(2, std::memory_order_relaxed);

        ringWritePos_ = 0;
        ringMask_ = fftSize_ - 1; // power of two guaranteed by the sanitation
        samplesUntilFFT_ = hopSize_;
        newDataReady_.store(false, std::memory_order_relaxed);

        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize)); // gate ON, last
    }

    /** @brief Resets all internal buffers and state to zero/floor values. */
    void reset() noexcept
    {
        const T floorDb = floorDb_.load(std::memory_order_relaxed);
        std::fill(inputRing_.begin(), inputRing_.end(), T(0));
        std::fill(magnitudesState_.begin(), magnitudesState_.end(), T(0));
        std::fill(peakState_.begin(), peakState_.end(), floorDb);

        for (auto& slot : outSlots_)
        {
            // numBins_ is 0 before prepare(), so the arrays exist here.
            for (int k = 0; k < numBins_; ++k)
            {
                slot.magnitudesDb[static_cast<size_t>(k)].store(floorDb, std::memory_order_relaxed);
                slot.peakDb[static_cast<size_t>(k)].store(floorDb, std::memory_order_relaxed);
            }
        }
        std::fill(magSnapshot_.begin(), magSnapshot_.end(), floorDb);
        std::fill(peakSnapshot_.begin(), peakSnapshot_.end(), floorDb);

        ringWritePos_ = 0;
        samplesUntilFFT_ = hopSize_;
        newDataReady_.store(false, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the per-frame magnitude smoothing factor.
     * @param factor 0 = no smoothing, 0.99 = heaviest. Applied once per
     *               analysis frame (hop), so the settling time scales with
     *               fftSize / sampleRate. Non-finite values are ignored.
     */
    void setSmoothing(T factor) noexcept
    {
        if (!std::isfinite(factor)) return;
        smoothing_.store(std::clamp(factor, T(0), T(0.99)), std::memory_order_relaxed);
    }

    /**
     * @brief Sets the peak-hold decay rate.
     * @param decayDbPerSecond Decay in dB per second (floored at 0;
     *                         non-finite values are ignored).
     */
    void setPeakDecay(T decayDbPerSecond) noexcept
    {
        if (!std::isfinite(decayDbPerSecond)) return;
        peakDecayRate_.store(std::max(T(0), decayDbPerSecond), std::memory_order_relaxed);
    }

    /** @brief Enables or disables peak-hold tracking. */
    void setPeakHoldEnabled(bool enabled) noexcept
    {
        peakHoldEnabled_.store(enabled, std::memory_order_relaxed);
    }

    /**
     * @brief Sets the readout floor in decibels.
     * @param floorDb Values below this floor are clamped to it (non-finite
     *                values are ignored).
     */
    void setFloorDb(T floorDb) noexcept
    {
        if (!std::isfinite(floorDb)) return;
        floorDb_.store(floorDb, std::memory_order_relaxed);
    }

    /** @brief Returns the per-frame magnitude smoothing factor. */
    [[nodiscard]] T getSmoothing() const noexcept { return smoothing_.load(std::memory_order_relaxed); }

    /** @brief Returns the peak-hold decay rate in dB per second. */
    [[nodiscard]] T getPeakDecay() const noexcept { return peakDecayRate_.load(std::memory_order_relaxed); }

    /** @brief Returns true if peak-hold tracking is enabled. */
    [[nodiscard]] bool isPeakHoldEnabled() const noexcept { return peakHoldEnabled_.load(std::memory_order_relaxed); }

    /** @brief Returns the readout floor in decibels. */
    [[nodiscard]] T getFloorDb() const noexcept { return floorDb_.load(std::memory_order_relaxed); }

    /** @brief Returns the window type in use (set at prepare time). */
    [[nodiscard]] WindowType getWindowType() const noexcept { return windowType_; }

    /**
     * @brief Pushes audio samples into the analyser's internal ring buffer.
     *
     * Computes the FFT synchronously when the internal hop size (50% overlap)
     * is met. No-op before prepare() or with a null/empty input.
     *
     * @param samples Pointer to the continuous audio data.
     * @param numSamples Number of samples to process.
     */
    void pushSamples(const T* samples, int numSamples) noexcept
    {
        if (fft_ == nullptr || samples == nullptr || numSamples <= 0)
            return;

        for (int i = 0; i < numSamples; ++i)
        {
            inputRing_[static_cast<size_t>(ringWritePos_)] = samples[i];
            ringWritePos_ = (ringWritePos_ + 1) & ringMask_; // pow2 mask, no division

            if (--samplesUntilFFT_ <= 0)
            {
                computeSpectrum();
                samplesUntilFFT_ = hopSize_; // Restart countdown with 50% overlap
            }
        }
    }

    /**
     * @brief Returns the current magnitude spectrum in decibels.
     *
     * Adopts the freshest published frame (if any) and copies it into
     * reader-private snapshot storage. The returned pointer refers to that
     * snapshot: it is overwritten only by the reader's own next call to
     * THIS getter and invalidated only by prepare(); no other thread ever
     * writes through it (see the Threading pointer-validity contract).
     *
     * @return Pointer to an array of size getNumBins() (0 before prepare()).
     */
    [[nodiscard]] const T* getMagnitudesDb() const noexcept
    {
        acquireLatestSlot();
        const OutSlot& slot = outSlots_[static_cast<size_t>(readSlot_)];
        for (int k = 0; k < numBins_; ++k)
            magSnapshot_[static_cast<size_t>(k)] =
                slot.magnitudesDb[static_cast<size_t>(k)].load(std::memory_order_relaxed);
        return magSnapshot_.data();
    }

    /**
     * @brief Returns the peak-hold spectrum in decibels.
     *
     * Same snapshot semantics as getMagnitudesDb(): the returned pointer
     * refers to reader-private storage overwritten only by the reader's own
     * next call to THIS getter and invalidated only by prepare().
     *
     * @return Pointer to an array of size getNumBins() (0 before prepare()).
     */
    [[nodiscard]] const T* getPeakHoldDb() const noexcept
    {
        acquireLatestSlot();
        const OutSlot& slot = outSlots_[static_cast<size_t>(readSlot_)];
        for (int k = 0; k < numBins_; ++k)
            peakSnapshot_[static_cast<size_t>(k)] =
                slot.peakDb[static_cast<size_t>(k)].load(std::memory_order_relaxed);
        return peakSnapshot_.data();
    }

    /** @brief Consumes and returns the new data flag. True if updated since last call. */
    [[nodiscard]] bool isNewDataReady() noexcept
    {
        return newDataReady_.exchange(false, std::memory_order_relaxed);
    }

    /** @brief Number of spectrum bins (fftSize/2 + 1); 0 before prepare(). */
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }

    /** @brief FFT size in samples; 0 before prepare(). */
    [[nodiscard]] int getFFTSize() const noexcept { return fftSize_; }

    /** @brief Centre frequency of the given bin in Hz (0 before prepare()). */
    [[nodiscard]] T binToFrequency(int bin) const noexcept
    {
        if (fftSize_ <= 0) return T(0);
        return static_cast<T>(bin) * static_cast<T>(sampleRate_) / static_cast<T>(fftSize_);
    }

private:
    void generateWindow(WindowType type)
    {
        switch (type)
        {
            case WindowType::Hamming: WindowFunctions<T>::hamming(window_.data(), fftSize_); break;
            case WindowType::Blackman: WindowFunctions<T>::blackman(window_.data(), fftSize_); break;
            case WindowType::BlackmanHarris: WindowFunctions<T>::blackmanHarris(window_.data(), fftSize_); break;
            case WindowType::FlatTop: WindowFunctions<T>::flatTop(window_.data(), fftSize_); break;
            case WindowType::Rectangular: WindowFunctions<T>::rectangular(window_.data(), fftSize_); break;
            case WindowType::Hann:
            default: // wild enum value: fall back to Hann (never a zeroed window)
                WindowFunctions<T>::hann(window_.data(), fftSize_);
                windowType_ = WindowType::Hann;
                break;
        }
    }

    void computeSpectrum() noexcept
    {
        // Parameters are published atomically; load once per frame.
        const T smoothing   = smoothing_.load(std::memory_order_relaxed);
        const T floorDb     = floorDb_.load(std::memory_order_relaxed);
        const T decayRate   = peakDecayRate_.load(std::memory_order_relaxed);
        const bool peakHold = peakHoldEnabled_.load(std::memory_order_relaxed);

        // 1. Copy & Window (pow2 mask instead of division)
        for (int i = 0; i < fftSize_; ++i)
        {
            int ringIdx = (ringWritePos_ + i) & ringMask_;
            fftBuffer_[static_cast<size_t>(i)] = inputRing_[static_cast<size_t>(ringIdx)] * window_[static_cast<size_t>(i)];
        }

        // 2. Forward FFT
        fft_->forward(fftBuffer_.data(), freqBuffer_.data());

        // 3. Magnitude Calculation & Smoothing
        const T oneMinusSmooth = T(1) - smoothing;

        for (int k = 0; k < numBins_; ++k)
        {
            T re = freqBuffer_[static_cast<size_t>(2 * k)];
            T im = freqBuffer_[static_cast<size_t>(2 * k + 1)];
            T mag = std::sqrt(re * re + im * im) * invGain_;

            // DC and Nyquist carry no mirrored bin, so the single-sided 2x in
            // invGain_ must be undone HERE, on the fresh magnitude. (Scaling
            // the smoothed STATE compounded 0.5x every frame and parked those
            // bins ~9.5 dB low in steady state.)
            if (k == 0 || k == numBins_ - 1) mag *= T(0.5);

            magnitudesState_[static_cast<size_t>(k)] =
                smoothing * magnitudesState_[static_cast<size_t>(k)] + oneMinusSmooth * mag;
        }

        // 4. Time-Domain Peak Decay Calculation (Using real hopSize_)
        const T peakDecayDb = decayRate * static_cast<T>(hopSize_) / static_cast<T>(sampleRate_);

        // 5. Write into the writer-owned slot of the triple buffer
        auto& slot = outSlots_[static_cast<size_t>(writeSlot_)];

        // 6. DB Conversion and Peak Hold (relaxed atomic stores: the slot is
        // writer-owned here, ordering is carried by the publication exchange)
        for (int k = 0; k < numBins_; ++k)
        {
            T dB = gainToDecibels(magnitudesState_[static_cast<size_t>(k)], floorDb);
            slot.magnitudesDb[static_cast<size_t>(k)].store(dB, std::memory_order_relaxed);

            if (peakHold)
            {
                // Canonical hold law: decay, but never below the live value.
                // (The old strict-compare branch decayed a full step whenever
                // dB == peak, so a stationary tone made the peak FLICKER one
                // decay step below the signal every other frame.)
                T& peak = peakState_[static_cast<size_t>(k)];
                peak = std::max(dB, std::max(floorDb, peak - peakDecayDb));
                slot.peakDb[static_cast<size_t>(k)].store(peak, std::memory_order_relaxed);
            }
        }

        // 7. Publish: swap the finished slot into 'pending' (with the fresh
        // bit) and adopt whatever slot was there as the next write target.
        // Wait-free; the reader can never observe a slot mid-write.
        const int old = pendingSlot_.exchange(writeSlot_ | kFreshBit, std::memory_order_acq_rel);
        writeSlot_ = old & kSlotMask;
        newDataReady_.store(true, std::memory_order_release);
    }

    double sampleRate_ = 48000.0;
    int fftSize_ = 0;   // 0 = unprepared (getters report honestly)
    int numBins_ = 0;
    int hopSize_ = 1024;

    std::unique_ptr<FFTReal<T>> fft_; // doubles as the "prepared" gate
    std::vector<T> window_;
    WindowType windowType_ = WindowType::Hann;
    T windowGain_ = T(1);
    T invGain_ = T(1);

    std::vector<T> inputRing_;
    int ringWritePos_ = 0;
    int samplesUntilFFT_ = 0;

    std::vector<T> fftBuffer_;
    std::vector<T> freqBuffer_;

    // Single source of truth for DSP state
    std::vector<T> magnitudesState_;
    std::vector<T> peakState_;

    // Tear-free triple buffer: writer owns writeSlot_, the reader owns
    // readSlot_, and pendingSlot_ (atomic, with a freshness bit) carries the
    // hand-off. Classic wait-free GUI metering scheme.
    //
    // Cross-thread publication derivation (ADR-013 section 6(b)). Two
    // obligations, BOTH required (M-008B lesson: an ownership argument at
    // acquisition time is NOT sufficient -- every returned pointer needs a
    // LIFETIME argument too):
    //
    // 1) OWNERSHIP at access time. A slot's words are accessed only by the
    //    thread that currently owns the slot, and ownership moves
    //    exclusively through atomic read-modify-writes on pendingSlot_:
    //  - writer: exchange(writeSlot_ | fresh, acq_rel) -- the release half
    //    publishes every write to the finished slot; the acquire half
    //    synchronizes with the reader's release when adopting the slot the
    //    reader gave up.
    //  - reader: compare_exchange(expected, readSlot_, acq_rel, acquire) --
    //    on success its acquire half reads from the writer's release
    //    exchange ([atomics.order]/2), so all writes to the adopted slot
    //    happen-before the reader's reads; its release half publishes that
    //    the reader is done with the slot it returns.
    // {writeSlot_, readSlot_, pendingSlot_ & kSlotMask} is a permutation of
    // {0, 1, 2}: it holds at prepare() (0/1/2) and every RMW swaps one
    // party's slot with the pending one atomically, so by induction over the
    // serialized modification order of pendingSlot_ no slot is ever owned by
    // both threads (the same ownership-transfer discipline as
    // Core/SpscQueue.h). ABA is harmless: a successful CAS is an RMW and
    // therefore reads the LATEST value in pendingSlot_'s modification order
    // -- if the same slot index returns, it does so via a writer
    // release-exchange that already published that slot's newest contents.
    //
    // 2) LIFETIME of returned pointers. Ownership alone does not cover the
    //    pointers the getters return: a pointer into a slot would outlive
    //    the reader's ownership as soon as the NEXT acquisition (by either
    //    getter) handed that slot back to the writer -- the M-008B
    //    returned-pointer race (D-M008B-C1). Therefore NO pointer into a
    //    slot ever escapes this class: each getter copies the acquired
    //    slot into reader-private snapshot storage (magSnapshot_ /
    //    peakSnapshot_, touched only by the reader) WHILE it owns the slot,
    //    and returns a pointer to the snapshot. The audio thread cannot
    //    reach the snapshots, so a held pointer can never be written by
    //    another thread, regardless of how long the caller keeps it.
    //
    // Defence in depth + oracle legibility: the slot words are relaxed
    // std::atomic<T>. Under (1)+(2) they are never accessed concurrently,
    // so plain words would be formally correct -- but atomics make any
    // future lifetime regression read defined (stale) values instead of UB,
    // and they make the DRD/Helgrind green criterion mechanical: those
    // oracles model pthread happens-before only and cannot see the RMW
    // edges, so with plain words they reported the slot arrays as racing
    // and the verdict rested on a human classification argument (which
    // M-008B iteration 1 got wrong). With atomic words every residual
    // oracle frame machine-classifies as a std::atomic op (ADR-013 6(a));
    // relaxed load/store compiles to plain moves on x86 and ARM. The
    // C++11-aware oracle for the protocol itself remains ThreadSanitizer
    // (CI) via the pinned concurrent tests (single-getter tear pin and
    // paired-getter lifetime pin).
    static constexpr int kFreshBit = 4;
    static constexpr int kSlotMask = 3;

    static_assert(std::atomic<T>::is_always_lock_free,
                  "SpectrumAnalyzer requires lock-free atomic<T> slot words "
                  "(audio-thread stores must not lock)");

    struct OutSlot
    {
        std::unique_ptr<std::atomic<T>[]> magnitudesDb;
        std::unique_ptr<std::atomic<T>[]> peakDb;
    };
    std::array<OutSlot, 3> outSlots_;
    int writeSlot_ = 0;                       // writer-thread private
    mutable int readSlot_ = 1;                // reader-thread private
    mutable std::atomic<int> pendingSlot_{ 2 };

    // Reader-private snapshot storage backing the getter return pointers
    // (see the lifetime derivation above). Only the reader thread touches
    // these; contents change only on the reader's own getter calls.
    mutable std::vector<T> magSnapshot_;
    mutable std::vector<T> peakSnapshot_;

    /** @brief Reader side: adopt the freshest published slot, if any. */
    void acquireLatestSlot() const noexcept
    {
        int expected = pendingSlot_.load(std::memory_order_acquire);
        while (expected & kFreshBit)
        {
            if (pendingSlot_.compare_exchange_weak(expected, readSlot_,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire))
            {
                readSlot_ = expected & kSlotMask;
                return;
            }
        }
    }

    int ringMask_ = 2047;

    std::atomic<T> smoothing_     { T(0.8) };
    std::atomic<T> peakDecayRate_ { T(10) };
    std::atomic<T> floorDb_       { T(-100) };
    std::atomic<bool> peakHoldEnabled_ { false };

    std::atomic<bool> newDataReady_{ false };
};

} // namespace dspark
