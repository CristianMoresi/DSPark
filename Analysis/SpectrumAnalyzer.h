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
 *   reader threads; reset() rewrites the reader-visible slots AND the
 *   getter snapshots).
 * - pushSamples(): audio thread (stream owner). No-op before prepare().
 * - getMagnitudesDb() / getPeakHoldDb() / isNewDataReady(): ONE reader
 *   thread (GUI). Wait-free triple-buffer hand-off; never blocks the writer.
 *   Frame-coherence contract: the array a getter returns always holds bins
 *   that all come from ONE analysis frame, however slow the reader is
 *   relative to the writer. The copy is validated against the frame's
 *   seqlock and is committed only if it completed against a single frame;
 *   a copy that did not is discarded, the previous (coherent, one frame
 *   older) snapshot is returned instead, and getStaleSnapshotCount()
 *   increments. A torn mixture of two frames is never returned.
 *   Pointer-validity contract: each getter adopts the freshest published
 *   frame and copies it into reader-private snapshot storage; the returned
 *   pointer refers to that snapshot, NEVER into the hand-off slots. Each
 *   getter alternates between TWO such buffers, so a returned pointer
 *   survives the reader's next call to the same getter and its contents are
 *   reused only by the call after that (or by the setup-only
 *   reset()/prepare(); a throwing prepare() leaves them untouched); the
 *   pointer itself is invalidated only by a SUCCESSFUL prepare(). No other
 *   thread ever writes through a returned pointer, so the reader may hold
 *   it across frames (the values simply go stale). A consecutive
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
 * analyzer.prepare(48000.0);        // automatic size: 2048 points at 48 kHz,
 *                                   // 4096 at 96 kHz, 8192 at 192 kHz --
 *                                   // ~23 Hz bins at every rate
 * // analyzer.prepare(48000.0, 2048);  // or pin the count explicitly
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
#include <type_traits>
#include <vector>

namespace dspark {

/**
 * @class SpectrumAnalyzer
 * @brief Real-time FFT spectrum analyser with per-bin smoothing and peak hold.
 *
 * The magnitude smoothing is a per-frame exponential (one analysis frame per
 * hop = fftSize/2 samples), so its settling TIME depends on fftSize and the
 * sample rate: frameRate = 2 * sampleRate / fftSize. The automatic default
 * size holds that frame rate near 47 fps from 44.1 to 192 kHz; a fixed count
 * would let it rise with the rate. The peak-hold decay is specified in
 * dB/second and is rate-invariant either way.
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
     * (conservative no-op); an explicit fftSize is clamped to [256, 16384] and
     * rounded UP to the next power of two; an out-of-range window enum falls
     * back to Hann. May allocate (setup thread only; while re-preparing, the
     * old and the new storage transiently coexist).
     *
     * RESOLUTION IS A RATIO, NOT A COUNT. Everything the transform resolves is
     * a function of fs/fftSize: the bin width is fs/fftSize (23.44 Hz at
     * 48 kHz/2048), the Hann main lobe spans 4*fs/fftSize (93.75 Hz there),
     * and the analysis frame rate is 2*fs/fftSize (46.9 fps there). A constant
     * sample count therefore delivers a coarser display as the rate rises:
     * measured on two partials a fifth apart in the mid register (D4 293.66 Hz
     * and A4 440.00 Hz), a fixed 2048-point transform separates them into two
     * peaks at 44.1, 48, 88.2 and 96 kHz but merges them into one at 176.4 and
     * 192 kHz, where the 375 Hz main lobe is wider than the interval. The
     * default is therefore AUTOMATIC and holds the span, hence the resolution,
     * constant; getFFTSize(), getNumBins() and binToFrequency() report what it
     * chose. The per-frame smoothing follows the frame rate the same way: at a
     * fixed 2048 the default 0.8 factor settles (90% of a step) in 255 ms at
     * 44.1 kHz but 59 ms at 192 kHz, while the automatic size holds it at
     * 235-255 ms across the range.
     *
     * On exceptions-enabled builds: if an allocation throws, the analyser is
     * left unprepared (pushSamples is a no-op until a later prepare()
     * succeeds) and NEVER half-configured: every allocation is built into
     * locals and committed only after all of them succeed, so a throw leaves
     * every reader-visible value exactly as it was on entry. After a throwing
     * FIRST prepare() the analyser is still never-prepared: the size getters
     * report 0 and the spectrum getters return zero readable bins (the
     * returned pointer may be null). After a throwing RE-prepare() the sizes,
     * the storage, the last published frame and any held snapshot pointers are
     * exactly the ones from before the call -- only the pushSamples gate is
     * off. getNumBins() therefore always matches the allocated storage, throw
     * or no throw. (With -fno-exceptions an allocation failure terminates
     * instead of throwing, as elsewhere in the framework, so none of the
     * post-throw states above is observable there.)
     *
     * @param sampleRate Sample rate in Hz.
     * @param fftSize    FFT size. Values <= 0 (the default) select the
     *                   AUTOMATIC size: the smallest power of two in
     *                   [256, 16384] spanning at least 2048/48000 s (~42.7 ms)
     *                   at sampleRate -- 2048 at 44.1/48 kHz, 4096 at
     *                   88.2/96 kHz, 8192 at 176.4/192 kHz, 16384 at 384 kHz,
     *                   512 at 8 kHz. Explicit positive values (power of two,
     *                   256 to 16384) are honoured as given.
     * @param windowType Window function to use (default: Hann).
     */
    void prepare(double sampleRate, int fftSize = 0, WindowType windowType = WindowType::Hann)
    {
        assert(fftSize <= 0 ||
               (fftSize >= 256 && fftSize <= 16384 && (fftSize & (fftSize - 1)) == 0));

        if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
            return;

        if (fftSize <= 0)
        {
            // Automatic: hold the analysis TIME SPAN constant across sample
            // rates (the span 2048 samples cover at 48 kHz). Bin width, main
            // lobe and frame rate are all fs/fftSize up to a constant, so a
            // constant span pins the resolution the display shows.
            const double target = sampleRate * (kAutoSpanRef / kAutoSpanRate);
            int n = kAutoMinFft;
            while (n < kAutoMaxFft && static_cast<double>(n) < target) n <<= 1;
            fftSize = n;
        }

        // Release-safe size sanitation (the assert only guards debug builds;
        // an unvalidated size would make FFTReal throw at stream time).
        fftSize = std::clamp(fftSize, 256, 16384);
        int pow2 = 256;
        while (pow2 < fftSize) pow2 <<= 1;
        fftSize = pow2;

        fft_.reset(); // gate OFF: pushSamples() is a no-op while rebuilding
                      // (and stays OFF if the allocation phase throws below)

        const int numBins = fftSize / 2 + 1;
        const T floorDb = floorDb_.load(std::memory_order_relaxed);

        // --- Allocation phase. EVERYTHING is built into locals and no
        // member is written until every allocation has succeeded, so
        // a bad_alloc from any of them leaves the analyser exactly as it was
        // (minus the fft_ gate above): sizes, slots and snapshots stay
        // mutually consistent -- never NEW sizes over OLD storage.
        std::vector<T> window(static_cast<size_t>(fftSize));
        const WindowType effectiveWindow = generateWindow(window.data(), fftSize, windowType);

        T windowGain = WindowFunctions<T>::coherentGain(window.data(), fftSize);
        if (windowGain < T(0.001)) windowGain = T(1);
        const T invGain = T(2) / (static_cast<T>(fftSize) * windowGain);

        std::vector<T> inputRing(static_cast<size_t>(fftSize), T(0));
        std::vector<T> fftBuffer(static_cast<size_t>(fftSize));
        std::vector<T> freqBuffer(static_cast<size_t>(fftSize + 2));

        // Unified DSP State
        std::vector<T> magnitudesState(static_cast<size_t>(numBins), T(0));
        std::vector<T> peakState(static_cast<size_t>(numBins), floorDb);

        // Triple buffer for tear-free cross-thread reading (relaxed-atomic
        // words; ordering comes from the pendingSlot_ RMWs and from each
        // slot's own seqlock counter, see below).
        std::array<OutSlot, 3> slots;
        for (auto& slot : slots)
        {
            slot.magnitudesDb = std::make_unique<std::atomic<T>[]>(static_cast<size_t>(numBins));
            slot.peakDb = std::make_unique<std::atomic<T>[]>(static_cast<size_t>(numBins));
            slot.seq = std::make_unique<std::atomic<unsigned>>(0u);
            for (int k = 0; k < numBins; ++k)
            {
                slot.magnitudesDb[static_cast<size_t>(k)].store(floorDb, std::memory_order_relaxed);
                slot.peakDb[static_cast<size_t>(k)].store(floorDb, std::memory_order_relaxed);
            }
        }
        // Reader-private snapshot storage backing the getter return pointers:
        // TWO alternating numBins-sized halves per getter, so a copy that
        // fails seqlock validation is discarded without disturbing the
        // coherent snapshot the reader is still allowed to be holding.
        std::vector<T> magSnapshot(static_cast<size_t>(2 * numBins), floorDb);
        std::vector<T> peakSnapshot(static_cast<size_t>(2 * numBins), floorDb);

        auto fft = std::make_unique<FFTReal<T>>(static_cast<size_t>(fftSize));

        // --- Commit phase: moves and scalar stores only, all noexcept, so
        // every size below matches its storage on every exit path (the
        // getter/reset() loop bounds trust that invariant). Setup thread
        // only (documented), so plain stores suffice. The "all noexcept"
        // half of that claim is compiler-checked rather than asserted in
        // prose: a throw BETWEEN two commits is the only way back to a
        // half-committed shape (NEW sizes over OLD storage).
        static_assert(std::is_nothrow_move_assignable_v<std::vector<T>> &&
                      std::is_nothrow_move_assignable_v<std::array<OutSlot, 3>> &&
                      std::is_nothrow_move_assignable_v<std::unique_ptr<FFTReal<T>>>,
                      "prepare()'s commit phase must be non-throwing");

        sampleRate_ = sampleRate;
        fftSize_ = fftSize;
        numBins_ = numBins;
        hopSize_ = fftSize / 2; // 50% overlap
        windowType_ = effectiveWindow;
        window_ = std::move(window);
        windowGain_ = windowGain;
        invGain_ = invGain;

        inputRing_ = std::move(inputRing);
        fftBuffer_ = std::move(fftBuffer);
        freqBuffer_ = std::move(freqBuffer);
        magnitudesState_ = std::move(magnitudesState);
        peakState_ = std::move(peakState);

        outSlots_ = std::move(slots);
        magSnapshot_ = std::move(magSnapshot);
        peakSnapshot_ = std::move(peakSnapshot);
        magSnapshotHalf_ = 0;
        peakSnapshotHalf_ = 0;
        staleSnapshots_ = 0;
        writeSlot_ = 0;
        readSlot_  = 1;
        pendingSlot_.store(2, std::memory_order_relaxed);

        ringWritePos_ = 0;
        ringMask_ = fftSize_ - 1; // power of two guaranteed by the sanitation
        samplesUntilFFT_ = hopSize_;
        newDataReady_.store(false, std::memory_order_relaxed);

        fft_ = std::move(fft); // gate ON, last
    }

    /**
     * @brief Resets all internal buffers and state to zero/floor values.
     *
     * Setup thread only. Rewrites the hand-off slots AND the reader-private
     * snapshots, so values read through a held getter pointer drop to the
     * floor (see the Threading pointer-validity contract).
     */
    void reset() noexcept
    {
        const T floorDb = floorDb_.load(std::memory_order_relaxed);
        std::fill(inputRing_.begin(), inputRing_.end(), T(0));
        std::fill(magnitudesState_.begin(), magnitudesState_.end(), T(0));
        std::fill(peakState_.begin(), peakState_.end(), floorDb);

        for (auto& slot : outSlots_)
        {
            // prepare() commits numBins_ and the slot arrays together (and
            // numBins_ is 0 before the first successful prepare()), so this
            // loop bound always matches the allocated storage.
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
     * reader-private snapshot storage. Every bin in the returned array comes
     * from the SAME analysis frame no matter how slow this thread is
     * relative to the writer: the copy is validated against the frame's
     * seqlock and, if it did not complete against one frame, it is discarded
     * and the previous (coherent, one frame older) snapshot is returned with
     * getStaleSnapshotCount() incremented. The returned pointer refers to
     * reader-private storage no other thread can reach; the getter
     * alternates between two such buffers, so the pointer survives the
     * reader's next call to THIS getter and its contents are reused by the
     * call after that (or by the setup-only reset()/prepare()), and it is
     * invalidated only by a successful prepare() (see the Threading
     * frame-coherence and pointer-validity contracts).
     *
     * @return Pointer to an array of size getNumBins(); the size is 0 before
     *         the first successful prepare() (a throwing first prepare()
     *         included) and the pointer may then be null.
     */
    [[nodiscard]] const T* getMagnitudesDb() const noexcept
    {
        return snapshotFrame(false);
    }

    /**
     * @brief Returns the peak-hold spectrum in decibels.
     *
     * Same frame-coherence, snapshot and pointer-validity semantics as
     * getMagnitudesDb(), over its own pair of reader-private buffers.
     *
     * @return Pointer to an array of size getNumBins(); the size is 0 before
     *         the first successful prepare() (a throwing first prepare()
     *         included) and the pointer may then be null.
     */
    [[nodiscard]] const T* getPeakHoldDb() const noexcept
    {
        return snapshotFrame(true);
    }

    /**
     * @brief Number of getter calls that had to return the previous snapshot
     *        because no attempt copied one frame coherently.
     *
     * Diagnostic for the frame-coherence contract, reader thread only
     * (reset to 0 by a successful prepare(), untouched by reset()). It never
     * counts a torn readout -- a torn copy is discarded, never returned; it
     * counts how often the reader lost every retry against the writer and
     * therefore repeated the previous frame. Expected to stay 0 outside
     * pathological reader stalls.
     */
    [[nodiscard]] long long getStaleSnapshotCount() const noexcept { return staleSnapshots_; }

    /** @brief Consumes and returns the new data flag. True if updated since last call. */
    [[nodiscard]] bool isNewDataReady() noexcept
    {
        return newDataReady_.exchange(false, std::memory_order_relaxed);
    }

    /** @brief Number of spectrum bins (fftSize/2 + 1); 0 before the first
     *         successful prepare(). Always matches the allocated storage,
     *         also after a throwing prepare(). */
    [[nodiscard]] int getNumBins() const noexcept { return numBins_; }

    /** @brief FFT size in samples; 0 before the first successful prepare(). */
    [[nodiscard]] int getFFTSize() const noexcept { return fftSize_; }

    /** @brief Centre frequency of the given bin in Hz (0 before the first
     *         successful prepare()). */
    [[nodiscard]] T binToFrequency(int bin) const noexcept
    {
        if (fftSize_ <= 0) return T(0);
        return static_cast<T>(bin) * static_cast<T>(sampleRate_) / static_cast<T>(fftSize_);
    }

private:
    /** @brief Fills dst with the requested window; returns the type actually
     *         generated (a wild enum value falls back to Hann). Static and
     *         member-free so prepare() can window into locals before any
     *         member is committed. */
    [[nodiscard]] static WindowType generateWindow(T* dst, int size, WindowType type) noexcept
    {
        switch (type)
        {
            case WindowType::Hamming: WindowFunctions<T>::hamming(dst, size); break;
            case WindowType::Blackman: WindowFunctions<T>::blackman(dst, size); break;
            case WindowType::BlackmanHarris: WindowFunctions<T>::blackmanHarris(dst, size); break;
            case WindowType::FlatTop: WindowFunctions<T>::flatTop(dst, size); break;
            case WindowType::Rectangular: WindowFunctions<T>::rectangular(dst, size); break;
            case WindowType::Hann:
            default: // wild enum value: fall back to Hann (never a zeroed window)
                WindowFunctions<T>::hann(dst, size);
                return WindowType::Hann;
        }
        return type;
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
        std::atomic<unsigned>& seq = *slot.seq;

        // Seqlock OPEN, frame-for-frame the same discipline as the reference
        // implementation in Core/Biquad.h setCoeffs(). An odd counter marks
        // this slot's frame as mid-write, so a reader copying the slot can
        // detect that its copy would mix two frames and discard it instead of
        // returning the mixture. The release fence is MANDATORY
        // ([atomics.fences]/2; Boehm, MSPC 2012): without it the relaxed bin
        // stores below may be hoisted above the odd counter, and a copy that
        // read mid-write words would pass validation. Cost is per analysis
        // FRAME (two RMWs and one fence per hop), never per sample.
        seq.fetch_add(1u, std::memory_order_acq_rel);   // -> odd
        std::atomic_thread_fence(std::memory_order_release);

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

        // Seqlock CLOSE: the frame staged in this slot is complete. The
        // release half publishes every bin store above to any reader that
        // reads this counter with acquire.
        seq.fetch_add(1u, std::memory_order_release);  // -> even

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
    // Cross-thread publication derivation. No local race detector can
    // certify relaxed atomics, so the argument is made here. THREE
    // obligations, all required: an ownership argument at acquisition time
    // is not sufficient -- every returned pointer needs a LIFETIME one, and
    // every multi-word readout needs a FRAME-COHERENCE one that holds for
    // the whole duration of the copy, not just at its first instruction.
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
    //    getter) handed that slot back to the writer, which would then
    //    overwrite it under the caller's feet. Therefore NO pointer into a
    //    slot ever escapes this class: each getter copies the acquired
    //    slot into reader-private snapshot storage (magSnapshot_ /
    //    peakSnapshot_, touched only by the reader) WHILE it owns the slot,
    //    and returns a pointer to the snapshot. The audio thread cannot
    //    reach the snapshots, so a held pointer can never be written by
    //    another thread, regardless of how long the caller keeps it.
    //
    // 3) FRAME COHERENCE across a slow copy. (1) and (2) are arguments
    //    about the OWNERSHIP protocol, and both hold only as long as that
    //    protocol is exactly right: a numBins-word copy is not atomic, so a
    //    reader that took a slow path (page fault, preemption, debugger,
    //    an unoptimised build) would return bins from two different frames
    //    the instant a future edit let the writer reach the slot being
    //    copied. Nothing about the copy itself detected that. Each slot
    //    therefore carries its own seqlock counter (OutSlot::seq): the
    //    writer brackets the bin stores with odd/even RMWs and a release
    //    fence, and the reader validates the counter after the copy behind
    //    an acquire fence ([atomics.fences]/2). A copy that could have
    //    mixed frames FAILS validation and is discarded into the spare
    //    snapshot half rather than published, so the contract holds by
    //    construction and not merely by the correctness of (1). It is the
    //    library's canonical seqlock applied per slot; the reference
    //    implementation is Core/Biquad.h.
    //
    // Defence in depth + oracle legibility: the slot words are relaxed
    // std::atomic<T>. Under (1)+(2) they are never accessed concurrently,
    // so plain words would be formally correct -- but atomics make any
    // future lifetime regression read defined (stale) values instead of UB,
    // and they make the DRD/Helgrind green criterion mechanical: those
    // oracles model pthread happens-before only and cannot see the RMW
    // edges, so with plain words they reported the slot arrays as racing
    // and the verdict rested on a human classification argument (which
    // an earlier version of this header got wrong). With atomic words
    // every residual frame machine-classifies as a std::atomic op;
    // relaxed load/store compiles to plain moves on x86 and ARM. The
    // C++11-aware oracle for the protocol itself remains ThreadSanitizer
    // (CI) via the pinned concurrent tests (single-getter tear pin and
    // paired-getter lifetime pin).
    static constexpr int kFreshBit = 4;
    static constexpr int kSlotMask = 3;

    /// Automatic-size policy: the span 2048 samples cover at 48 kHz, resolved
    /// inside [256, 16384] (covers 8 kHz .. 384 kHz without hitting a clamp).
    static constexpr double kAutoSpanRef = 2048.0;
    static constexpr double kAutoSpanRate = 48000.0;
    static constexpr int kAutoMinFft = 256;
    static constexpr int kAutoMaxFft = 16384;

    /** @brief Coherent-copy attempts before the reader gives up and repeats
     *         the previous snapshot. Bounded so the getter stays wait-free:
     *         it never waits for the writer, it only declines to publish a
     *         copy it could not prove coherent. */
    static constexpr int kSnapshotAttempts = 4;

    static_assert(std::atomic<T>::is_always_lock_free,
                  "SpectrumAnalyzer requires lock-free atomic<T> slot words "
                  "(audio-thread stores must not lock)");
    static_assert(std::atomic<unsigned>::is_always_lock_free,
                  "SpectrumAnalyzer requires a lock-free seqlock counter "
                  "(audio-thread RMWs must not lock)");

    struct OutSlot
    {
        std::unique_ptr<std::atomic<T>[]> magnitudesDb;
        std::unique_ptr<std::atomic<T>[]> peakDb;
        // Seqlock counter for the frame staged here: odd = mid-write,
        // even = complete (obligation 3 above). Heap-held so that OutSlot
        // stays movable and prepare()'s commit phase remains noexcept.
        std::unique_ptr<std::atomic<unsigned>> seq;
    };
    std::array<OutSlot, 3> outSlots_;
    int writeSlot_ = 0;                       // writer-thread private
    mutable int readSlot_ = 1;                // reader-thread private
    mutable std::atomic<int> pendingSlot_{ 2 };

    // Reader-private snapshot storage backing the getter return pointers
    // (see the lifetime derivation above). Only the reader thread touches
    // these; contents change only on the reader's own getter calls. Each
    // vector holds TWO numBins_-sized halves: the getter copies into the
    // spare half and swaps the published index only once the copy has been
    // proved frame-coherent, so a discarded copy can never be observed and
    // a pointer handed out last call is never scribbled on mid-copy.
    mutable std::vector<T> magSnapshot_;
    mutable std::vector<T> peakSnapshot_;
    mutable int magSnapshotHalf_ = 0;         // half currently published
    mutable int peakSnapshotHalf_ = 0;
    mutable long long staleSnapshots_ = 0;    // coherent-copy failures

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

    /**
     * @brief Reader side: adopt the freshest frame and copy one of its two
     *        arrays out coherently.
     *
     * Copies into the spare snapshot half and validates the source slot's
     * seqlock around the copy, frame-for-frame the same shape as the
     * reference reader in Core/Biquad.h applyPendingCoeffs(): the
     * counter must be even before the copy and unchanged after it, read
     * behind an acquire fence so neither the compiler nor a weakly ordered
     * CPU can sink the bin loads past the re-read. Only a validated copy is
     * published; otherwise the previously published half is returned
     * unchanged and staleSnapshots_ counts the miss.
     *
     * @param peakArray false = magnitudes, true = peak hold.
     */
    [[nodiscard]] const T* snapshotFrame(bool peakArray) const noexcept
    {
        std::vector<T>& store = peakArray ? peakSnapshot_ : magSnapshot_;
        int& half = peakArray ? peakSnapshotHalf_ : magSnapshotHalf_;

        // Unprepared: no slots, no seq counters, nothing readable. Honest
        // empty result (the pointer may be null), as documented.
        if (numBins_ <= 0) return store.data();

        T* const spare = store.data()
                       + static_cast<size_t>(half ^ 1) * static_cast<size_t>(numBins_);

        for (int attempt = 0; attempt < kSnapshotAttempts; ++attempt)
        {
            acquireLatestSlot();
            const OutSlot& slot = outSlots_[static_cast<size_t>(readSlot_)];
            const std::atomic<T>* const src =
                (peakArray ? slot.peakDb : slot.magnitudesDb).get();
            const std::atomic<unsigned>& seq = *slot.seq;

            const unsigned s0 = seq.load(std::memory_order_acquire);
            if ((s0 & 1u) != 0u) continue;    // a frame is mid-write in here

            for (int k = 0; k < numBins_; ++k)
                spare[static_cast<size_t>(k)] =
                    src[static_cast<size_t>(k)].load(std::memory_order_relaxed);

            std::atomic_thread_fence(std::memory_order_acquire);
            if (seq.load(std::memory_order_relaxed) == s0)
            {
                half ^= 1;                    // commit the validated copy
                return spare;
            }
        }

        ++staleSnapshots_;
        return store.data()
             + static_cast<size_t>(half) * static_cast<size_t>(numBins_);
    }

    int ringMask_ = 2047;

    std::atomic<T> smoothing_     { T(0.8) };
    std::atomic<T> peakDecayRate_ { T(10) };
    std::atomic<T> floorDb_       { T(-100) };
    std::atomic<bool> peakHoldEnabled_ { false };

    std::atomic<bool> newDataReady_{ false };
};

} // namespace dspark
