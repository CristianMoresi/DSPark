// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LoudnessMeter.h
 * @brief Thread-safe EBU R128 / ITU-R BS.1770 loudness meter.
 *
 * Momentary (400 ms), short-term (3 s), integrated (two-pass gated),
 * loudness range (EBU Tech 3342) and true peak (the BS.1770 Annex 2
 * procedure, read by the long-kernel TruePeakEstimator).
 * Integrated loudness and LRA use constant-memory histograms: no allocation
 * and O(bins) readout, real-time safe. Integrated loudness keeps the exact
 * energy of every bin, so it reads the block energies themselves rather than
 * a 0.1 LU grid; LRA keeps every short-term value to 0.005 LU and the exact
 * energy of its relative gate. Tech 3341/3342 tone programmes are checked by
 * the conformance suite on every target, and the official EBU files when
 * DSPARK_EBU_TEST_SET points at them.
 *
 * Threading:
 * - prepare() / reset(): setup thread / stream owner (not concurrent with
 *   the process calls or the readers).
 * - process() / processBlock() / finalizeTruePeak(): audio thread (stream owner).
 *   No-op before prepare(). getMeasurementInfo() also requires the stream owner.
 * - All readouts (getMomentaryLUFS, getShortTermLUFS, getIntegratedLUFS,
 *   getLoudnessRange, getTruePeakDb, isMeasurementValid): any thread,
 *   lock-free; values are approximate while a block is in flight (metering).
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, Biquad.h, DspMath.h, TruePeakDetector.h.
 */

#include "../Core/DspMath.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/Biquad.h"
#include "../Core/TruePeakDetector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <array>

namespace dspark {

/**
 * @class LoudnessMeter
 * @brief Real-time safe EBU R128 loudness meter.
 *
 * Implements Momentary (400ms), Short-Term (3s), and Integrated (gated) measurements.
 * Utilizes a constant-memory histogram for infinite integrated loudness tracking
 * without memory allocation or O(N) CPU scaling, ensuring strict RT compliance.
 *
 * Scalar readouts (`getMomentaryLUFS`, etc.) are lock-free and thread-safe
 * for GUI threads while audio is processing. getMeasurementInfo() instead
 * requires the stream owner or a stopped measurement.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LoudnessMeter
{
public:
    /** @brief Window/gate coverage for a stopped or stream-owner measurement.
     * Durations use the existing floor(sampleRate / 10) sample hop (minimum 1).
     * The last partial hop contributes to neither windows nor histograms yet.
     * Gate counts include complete windows surviving the absolute -70 LUFS gate,
     * before each metric's relative gate. A zero count is not valid loudness.
     */
    struct MeasurementInfo
    {
        std::int64_t committedFrames = 0;
        int pendingFrames = 0;
        int hopFrames = 0;
        std::uint64_t integratedBlocks = 0;
        std::uint64_t rangeBlocks = 0;
    };

    /** @brief Returns window coverage without modifying any state.
     * Call from the stream owner, or after processing has stopped. Unlike the
     * scalar GUI readouts, this reads the writer-owned hop counters. Bounded
     * O(histogram bins) work and no allocation. Invalid measurements must also
     * be rejected using isMeasurementValid().
     */
    [[nodiscard]] MeasurementInfo getMeasurementInfo() const noexcept
    {
        MeasurementInfo info;
        if (!prepared_.load(std::memory_order_relaxed)) return info;
        info.committedFrames = totalCommittedBlocks_ * blockSamples_;
        info.pendingFrames = currentBlockSamples_;
        info.hopFrames = blockSamples_;
        for (int i = 0; i < kNumBins; ++i)
        {
            info.integratedBlocks += histogram_[i].load(std::memory_order_relaxed);
            info.rangeBlocks += lraHistogram_[i].load(std::memory_order_relaxed);
        }
        return info;
    }

    /**
     * @brief Prepares the meter and pre-calculates filter coefficients.
     *
     * A non-finite or non-positive sample rate is ignored (conservative
     * no-op keeping the previous state).
     *
     * @param sampleRate Sample rate in Hz (must be > 0 and finite).
     * @param numChannels Accepted for API symmetry; the channel layout is
     *                    taken from each processed buffer (up to 2 channels
     *                    are measured).
     */
    void prepare(double sampleRate, int numChannels = 2) noexcept
    {
        (void)numChannels;
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0) return;

        sampleRate_ = sampleRate;

        computeKWeighting(sampleRate);

        // 100 ms block length (clamped in double before the cast: an absurd
        // finite rate must not overflow the int conversion)
        blockSamples_ = static_cast<int>(std::clamp(sampleRate * 0.1, 1.0, 1.0e9));

        reset();
        prepared_.store(true, std::memory_order_relaxed);
    }

    /** @brief Unified API preparation. */
    void prepare(const AudioSpec& spec) noexcept
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Processes a non-interleaved buffer view (read-only).
     *
     * The first two channels are measured; additional channels are ignored
     * (multichannel spatial weighting is not implemented).
     *
     * @param buffer AudioBufferView to analyze.
     */
    void processBlock(AudioBufferView<const T> buffer) noexcept
    {
        const int nCh = buffer.getNumChannels();
        const int nS = buffer.getNumSamples();
        if (nCh >= 2)
            process(buffer.getChannel(0), buffer.getChannel(1), nS);
        else if (nCh == 1)
            process(buffer.getChannel(0), nS);
    }

    /**
     * @brief Processes a mono block of samples.
     * @param data Pointer to mono audio data.
     * @param numSamples Number of samples to process.
     */
    void process(const T* data, int numSamples) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed) || data == nullptr || numSamples <= 0)
            return;

        truePeakMax_.store(measureTruePeak(data, 0, numSamples,
                                           truePeakMax_.load(std::memory_order_relaxed)),
                           std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            double filtered = applyKWeighting(static_cast<double>(data[i]), 0);
            currentBlockPower_ += filtered * filtered;
            if (!std::isfinite(currentBlockPower_))
                invalidateMeasurement();

            if (++currentBlockSamples_ >= blockSamples_)
                commitBlock();
        }
    }

    /**
     * @brief Processes a stereo block of samples.
     * @param left Pointer to left channel data.
     * @param right Pointer to right channel data.
     * @param numSamples Number of samples per channel.
     */
    void process(const T* left, const T* right, int numSamples) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)
            || left == nullptr || right == nullptr || numSamples <= 0)
            return;

        const T peakLeft = measureTruePeak(left, 0, numSamples,
                                           truePeakMax_.load(std::memory_order_relaxed));
        truePeakMax_.store(measureTruePeak(right, 1, numSamples, peakLeft), std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
        {
            double filtL = applyKWeighting(static_cast<double>(left[i]), 0);
            double filtR = applyKWeighting(static_cast<double>(right[i]), 1);

            currentBlockPower_ += (filtL * filtL + filtR * filtR);
            if (!std::isfinite(currentBlockPower_))
                invalidateMeasurement();

            if (++currentBlockSamples_ >= blockSamples_)
                commitBlock();
        }
    }

    /**
     * @brief Reads the Momentary loudness (400 ms window).
     * @return Loudness in LUFS (minimum -100.0).
     */
    [[nodiscard]] T getMomentaryLUFS() const noexcept
    {
        return calculateLUFSFromBlocks(4);
    }

    /**
     * @brief Reads the Short-term loudness (3 second window).
     * @return Loudness in LUFS (minimum -100.0).
     */
    [[nodiscard]] T getShortTermLUFS() const noexcept
    {
        return calculateLUFSFromBlocks(30);
    }

    /**
     * @brief Computes the Integrated loudness using standard two-pass gating.
     * @note O(bins) with a constant bin count (no per-block cost growth).
     *       Safe to call from any thread.
     * @return Loudness in LUFS (minimum -100.0).
     */
    [[nodiscard]] T getIntegratedLUFS() const noexcept
    {
        // Pass 1: Absolute Gate (-70 LUFS)
        double sumPowerUngated = 0.0;
        uint64_t countUngated = 0;

        for (int i = 0; i < kNumBins; ++i)
        {
            sumPowerUngated += histogramPower_[i].load(std::memory_order_relaxed);
            countUngated += histogram_[i].load(std::memory_order_relaxed);
        }

        if (countUngated == 0) return T(-100);

        double meanPowerUngated = sumPowerUngated / countUngated;
        double ungatedLUFS = powerToLUFS(meanPowerUngated);

        // Pass 2: Relative Gate (-10 LU below ungated mean)
        double relativeGateLUFS = ungatedLUFS - 10.0;
        const int relativeGateBin = firstBinAtOrAbove(relativeGateLUFS);

        double sumPowerGated = 0.0;
        uint64_t countGated = 0;

        for (int i = relativeGateBin; i < kNumBins; ++i)
        {
            sumPowerGated += histogramPower_[i].load(std::memory_order_relaxed);
            countGated += histogram_[i].load(std::memory_order_relaxed);
        }

        if (countGated == 0) return T(-100);

        return static_cast<T>(powerToLUFS(sumPowerGated / countGated));
    }

    /**
     * @brief Reports whether every measurement since reset was representable.
     *
     * False means a filter/accumulator became non-finite, a block had to be
     * discarded, or a histogram could not represent another value exactly.
     * The streaming meter continues recovering where possible, but an offline
     * caller can use this diagnostic to distinguish recovery from silence.
     * reset() starts a new diagnostic transaction.
     */
    [[nodiscard]] bool isMeasurementValid() const noexcept
    {
        return measurementValid_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Returns the maximum true peak observed since the last reset.
     *
     * The BS.1770 Annex 2 procedure read with Core's TruePeakEstimator (the
     * reconstruction at the quarter points of every sample interval with
     * 64-tap phases, each crest refined): steady sines read within 0.003 dB
     * of their true peak up to 0.49 fs (transients and content above 0.45 fs:
     * see TruePeakEstimator). A sample's own magnitude counts at once; the
     * reconstruction around it is read TruePeakEstimator::getDelay() samples
     * later, and the end of a finite programme by finalizeTruePeak().
     *
     * @return True peak in dBTP.
     */
    [[nodiscard]] T getTruePeakDb() const noexcept
    {
        return gainToDecibels(truePeakMax_.load(std::memory_order_relaxed));
    }

    /** @brief Includes the final interpolation tail of a finite programme.
     * Call after the last input block, from the stream owner; readers may remain
     * concurrent. Repeated calls are idempotent. The Core estimator evaluates
     * zero extension on copied state: programme duration, loudness windows,
     * integrated/LRA gates and live filter history are unchanged. No allocation.
     * Do not call after each block of one continuous programme: the zero-ended
     * prefix peak would then be retained in the maximum. reset() starts a new
     * measurement as usual.
     */
    void finalizeTruePeak() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        T peak = truePeakMax_.load(std::memory_order_relaxed);
        for (int channel = 0; channel < kMaxChannels; ++channel)
        {
            const T tail = truePeak_.getTailPeak(tpChannels_[static_cast<size_t>(channel)]);
            if (std::isfinite(tail))
                peak = std::max(peak, tail);
            else
                invalidateMeasurement();
        }
        truePeakMax_.store(peak, std::memory_order_relaxed);
    }

    /**
     * @brief Computes the EBU R128 Loudness Range (EBU Tech 3342).
     *
     * Short-term (3 s) loudness sampled at 10 Hz (every 100 ms: Tech 3342 has
     * required at least 10 Hz since V3), gated at -70 LUFS absolute and -20 LU
     * relative; LRA is the spread between the 10th and 95th percentiles.
     * O(bins) and real-time safe like the integrated gate.
     *
     * The rate matters on programmes whose level steps: a 1 Hz series reads
     * a 3 s stepped programme 1.0 LU higher, and moves by up to 1.5 LU when the
     * programme starts a fraction of a second later; at 10 Hz the reading
     * agrees with a 100 Hz series within 0.1 LU.
     *
     * Percentiles follow the Tech 3342 reference code: among the n gated
     * short-term values sorted ascending, P is the value at 0-based rank
     * round((n - 1) * P). Every short-term value is held to 0.005 LU (the
     * reading agrees with the reference code run on the exact series within
     * 0.005 LU) and the relative gate is set from the exact mean energy. The
     * first short-term value exists after 3 s of input, so a programme
     * shorter than that reads 0; a steady programme reads 0 at any length.
     *
     * @return Loudness range in LU (0 if not enough material yet).
     */
    [[nodiscard]] T getLoudnessRange() const noexcept
    {
        // Pass 1: relative gate from the exact mean energy of the
        // absolute-gated short-term values.
        double sumPower = 0.0;
        uint64_t count = 0;
        for (int i = 0; i < kNumBins; ++i)
        {
            const uint32_t c = lraHistogram_[i].load(std::memory_order_relaxed);
            if (c > 0)
            {
                sumPower += lraEnergy_[i].load(std::memory_order_relaxed);
                count += c;
            }
        }
        if (count == 0) return T(0);

        // A value takes part when the centre of its 0.005 LU cell clears the
        // gate (as a block does for the integrated gate).
        const double relGateLUFS = powerToLUFS(sumPower / static_cast<double>(count)) - 20.0;
        const int gateCell = firstRangeCellAtOrAbove(relGateLUFS);
        const int gateBin = gateCell / kRangeCells;

        uint64_t gateCells[kRangeCells];
        rangeCellCounts(gateBin, gateCells);
        uint64_t gateBinCount = 0;
        for (int s = gateCell % kRangeCells; s < kRangeCells; ++s)
            gateBinCount += gateCells[s];

        // Pass 2: 10th / 95th percentiles above the relative gate.
        uint64_t gatedCount = gateBinCount;
        for (int i = gateBin + 1; i < kNumBins; ++i)
            gatedCount += lraHistogram_[i].load(std::memory_order_relaxed);
        if (gatedCount == 0) return T(0);

        // Percentiles by the Tech 3342 reference rule: the value of 0-based rank
        // round((n - 1) * p) among the n gated values. Both ranks are always
        // measured values - never the gate threshold - and every n is
        // covered: with one value, or with n equal values, the range is 0.
        const auto rank10 = static_cast<uint64_t>(std::llround(0.10 * static_cast<double>(gatedCount - 1)));
        const auto rank95 = static_cast<uint64_t>(std::llround(0.95 * static_cast<double>(gatedCount - 1)));

        // The 0.1 LU bins locate each rank; the bin's 0.005 LU cells place it.
        auto valueAt = [&](int bin, uint64_t rankInBin) noexcept {
            uint64_t cells[kRangeCells];
            const int first = bin == gateBin ? gateCell % kRangeCells : 0;
            if (bin == gateBin)
                std::copy(gateCells, gateCells + kRangeCells, cells);
            else
                rangeCellCounts(bin, cells);
            uint64_t seen = 0;
            int lastFilled = -1;
            for (int s = first; s < kRangeCells; ++s)
            {
                if (cells[s] == 0) continue;
                lastFilled = s;
                seen += cells[s];
                if (seen > rankInBin) break;
            }
            // A reader that raced the writer may find the bin count ahead of
            // its cells: the last filled cell then stands in.
            const int cell = lastFilled >= 0 ? lastFilled : kRangeCells / 2;
            return rangeCellCentre(bin * kRangeCells + cell);
        };

        double p10 = 0.0, p95 = 0.0;
        uint64_t running = 0;
        bool have10 = false;
        for (int i = gateBin; i < kNumBins; ++i)
        {
            const uint64_t c = i == gateBin ? gateBinCount
                                            : lraHistogram_[i].load(std::memory_order_relaxed);
            if (c == 0) continue;
            // `running` values sit below this bin: ranks running .. running+c-1 here.
            if (!have10 && running + c > rank10)
            {
                p10 = valueAt(i, rank10 - running);
                have10 = true;
            }
            if (running + c > rank95)
            {
                p95 = valueAt(i, rank95 - running);
                break;
            }
            running += c;
        }
        return static_cast<T>(std::max(0.0, p95 - p10));
    }

    /**
     * @brief Clears all measurements and resets filters.
     *
     * Call from the stream owner or a setup thread, not concurrently with
     * the process calls: the accumulators and filter states are plain
     * writer-owned members.
     */
    void reset() noexcept
    {
        for (int ch = 0; ch < kMaxChannels; ++ch)
        {
            preState_[ch] = {};
            rlbState_[ch] = {};
        }

        for (auto& power : blockPowers_)
            power.store(0.0, std::memory_order_relaxed);

        for (auto& bin : histogram_)
            bin.store(0, std::memory_order_relaxed);
        for (auto& energy : histogramPower_)
            energy.store(0.0, std::memory_order_relaxed);
        for (auto& bin : lraHistogram_)
            bin.store(0, std::memory_order_relaxed);
        for (auto& energy : lraEnergy_)
            energy.store(0.0, std::memory_order_relaxed);
        for (auto& cell : lraCells_)
            cell.store(0, std::memory_order_relaxed);

        for (auto& state : tpChannels_)
            state = {};
        truePeakMax_.store(T(0), std::memory_order_relaxed);

        blockWritePos_.store(0, std::memory_order_relaxed);
        totalCommittedBlocks_ = 0;
        currentBlockPower_ = 0.0;
        currentBlockSamples_ = 0;
        measurementValid_.store(true, std::memory_order_relaxed);
    }

private:
    static constexpr int kMaxChannels = 2; // Expandable to 8 with proper spatial weighting

    // Histogram specs: -70 LUFS to +30 LUFS with 0.1 resolution = 1000 bins
    static constexpr double kMinHistogramLUFS = -70.0;
    static constexpr double kMaxHistogramLUFS = 30.0;
    static constexpr double kBinWidth = 0.1;
    static constexpr int kNumBins = 1000;

    // Loudness range cells: each 0.1 LU bin split in 20 cells of 0.005 LU
    // (16-bit counts; the bin's 32-bit count stays authoritative).
    static constexpr int kRangeCells = 20;
    static constexpr double kRangeCellWidth = kBinWidth / kRangeCells;
    static constexpr int kNumRangeCells = kNumBins * kRangeCells;
    static constexpr uint16_t kRangeCellMax = std::numeric_limits<uint16_t>::max();

    // True-peak estimator chunk (the most samples one estimator call takes).
    static constexpr int kTruePeakChunk = TruePeakEstimator<T>::kChunk;

    // Filters state (Forced to double to prevent DF2T precision issues)
    struct BiquadState { double z1 = 0.0, z2 = 0.0; };
    struct BiquadCoeff { double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0; };

    void computeKWeighting(double sr)
    {
        // Pre-warped analog parameterization that reproduces the official
        // ITU-R BS.1770-5 table 1/2 coefficients at 48 kHz to machine
        // precision (verified: max error 8.9e-16). This is an exact 48 kHz
        // parameterization, not a claim of an identical response at every
        // rate: the retained 997 Hz cascade gains are +0.691014 dB at 48 kHz,
        // +0.693776 dB at 44.1 kHz and +0.663943 dB at 192 kHz. Residual
        // full-band deviation stays below 0.05 dB through 384 kHz and inside
        // the standard tolerance. The -0.691 constant in powerToLUFS remains
        // tied to this cascade; an RBJ shelf or a gain-normalized RLB high-pass
        // reads ~0.26 LU low on the EBU conformance vectors. The design lives
        // in BiquadCoeffs so every K-weighted measurement shares it.
        const auto shelf = BiquadCoeffs::makeKWeightingShelf(sr);
        pre_ = { shelf.b0, shelf.b1, shelf.b2, shelf.a1, shelf.a2 };
        const auto rlb = BiquadCoeffs::makeKWeightingHighPass(sr);
        rlb_ = { rlb.b0, rlb.b1, rlb.b2, rlb.a1, rlb.a2 };
    }

    double applyBiquad(double input, const BiquadCoeff& c, BiquadState& s) noexcept
    {
        // Adding 1e-18 protects against denormal numbers (Flush-To-Zero fallback)
        double output = c.b0 * input + s.z1;
        s.z1 = c.b1 * input - c.a1 * output + s.z2;
        s.z2 = c.b2 * input - c.a2 * output + 1e-18;
        return output;
    }

    double applyKWeighting(double input, int channel) noexcept
    {
        double x = applyBiquad(input, pre_, preState_[channel]);
        return applyBiquad(x, rlb_, rlbState_[channel]);
    }

    void commitBlock() noexcept
    {
        const double meanPower = currentBlockPower_ / currentBlockSamples_;
        currentBlockPower_ = 0.0;
        currentBlockSamples_ = 0;

        // A non-finite block (NaN/Inf fed by the caller) would poison the
        // sliding window and freeze the histograms forever: the K-filter
        // recursion never drains a NaN. A meter must report the signal as it
        // is NOW, so drop the block, clear the filter states and resume
        // measuring clean on the next one.
        if (!std::isfinite(meanPower))
        {
            invalidateMeasurement();
            for (int ch = 0; ch < kMaxChannels; ++ch)
            {
                preState_[ch] = {};
                rlbState_[ch] = {};
            }
            return;
        }

        // Update sliding window ring buffer
        int currentPos = blockWritePos_.load(std::memory_order_relaxed);
        blockPowers_[currentPos].store(meanPower, std::memory_order_relaxed);

        int nextPos = (currentPos + 1) % 30; // 30 blocks = 3s max window
        blockWritePos_.store(nextPos, std::memory_order_release);
        ++totalCommittedBlocks_;

        // BS.1770-4 integrated gating uses 400 ms blocks with 75% overlap.
        // With a 100 ms sub-block hop, that is EXACTLY the mean of the last
        // four sub-blocks committed at every 100 ms step. (Gating raw 100 ms
        // blocks has higher variance and fails the EBU conformance vectors.)
        if (totalCommittedBlocks_ >= 4)
        {
            double gating400 = meanPower;
            for (int back = 1; back < 4; ++back)
            {
                int idx = (currentPos - back + 30) % 30;
                gating400 += blockPowers_[idx].load(std::memory_order_relaxed);
            }
            gating400 *= 0.25;

            const double lufs = powerToLUFS(gating400);
            if (!std::isfinite(gating400) || !std::isfinite(lufs))
            {
                invalidateMeasurement();
            }
            else if (lufs >= kMinHistogramLUFS)
            {
                int binIndex = static_cast<int>(std::round((lufs - kMinHistogramLUFS) / kBinWidth));
                if (binIndex >= kNumBins)
                    invalidateMeasurement();
                binIndex = std::clamp(binIndex, 0, kNumBins - 1);
                incrementHistogram(histogram_[binIndex]);
                auto& energy = histogramPower_[static_cast<size_t>(binIndex)];
                energy.store(energy.load(std::memory_order_relaxed) + gating400,
                             std::memory_order_relaxed);
            }
        }

        // EBU Tech 3342 loudness range: short-term (3 s) values sampled at
        // 10 Hz - every 100 ms sub-block, the 2.9 s minimum window overlap
        // Tech 3342 has required since V3 (2016) - absolute-gated at -70 LUFS
        // into a histogram of 0.005 LU cells, with the exact energy per bin.
        if (totalCommittedBlocks_ >= 30)
        {
            const double stPower = meanPowerOfBlocks(30);
            const double stLufs = powerToLUFS(stPower);
            if (!std::isfinite(stPower) || !std::isfinite(stLufs))
            {
                invalidateMeasurement();
            }
            else if (stLufs >= kMinHistogramLUFS)
            {
                const double position = std::floor((stLufs - kMinHistogramLUFS + 0.5 * kBinWidth)
                                                   / kRangeCellWidth);
                if (position >= static_cast<double>(kNumRangeCells))
                    invalidateMeasurement();
                const int cell = static_cast<int>(std::clamp(position, 0.0,
                                                             static_cast<double>(kNumRangeCells - 1)));
                const int bin = cell / kRangeCells;
                if (lraHistogram_[bin].load(std::memory_order_relaxed)
                    == std::numeric_limits<uint32_t>::max())
                {
                    invalidateMeasurement();
                }
                else
                {
                    auto& cellCount = lraCells_[static_cast<size_t>(cell)];
                    const uint16_t c = cellCount.load(std::memory_order_relaxed);
                    if (c < kRangeCellMax) // a full cell hands the rest to its bin count
                        cellCount.store(static_cast<uint16_t>(c + 1), std::memory_order_relaxed);
                    auto& energy = lraEnergy_[static_cast<size_t>(bin)];
                    energy.store(energy.load(std::memory_order_relaxed) + stPower,
                                 std::memory_order_relaxed);
                    lraHistogram_[bin].fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    /// Mean K-weighted power of the last numBlocks committed sub-blocks.
    [[nodiscard]] double meanPowerOfBlocks(int numBlocks) const noexcept
    {
        double sum = 0.0;
        const int currentPos = blockWritePos_.load(std::memory_order_acquire);
        for (int i = 0; i < numBlocks; ++i)
        {
            const int idx = (currentPos - 1 - i + 30) % 30;
            sum += blockPowers_[idx].load(std::memory_order_relaxed);
        }
        return sum / numBlocks;
    }

    /// Effective counts of the 0.005 LU cells of one 0.1 LU bin. A cell that
    /// filled its 16-bit count takes the bin's remainder (one full cell per
    /// bin is exact; that needs 65535 short-term values, 1.8 h, in 0.005 LU).
    void rangeCellCounts(int bin, uint64_t (&out)[kRangeCells]) const noexcept
    {
        uint64_t sum = 0;
        int full = -1;
        for (int s = 0; s < kRangeCells; ++s)
        {
            const uint16_t v = lraCells_[static_cast<size_t>(bin * kRangeCells + s)]
                                   .load(std::memory_order_relaxed);
            out[s] = v;
            sum += v;
            if (v == kRangeCellMax && full < 0) full = s;
        }
        const uint64_t total = lraHistogram_[static_cast<size_t>(bin)].load(std::memory_order_relaxed);
        if (full >= 0 && total > sum)
            out[full] += total - sum;
    }

    /// First range cell whose centre is at or above `lufs`.
    [[nodiscard]] static int firstRangeCellAtOrAbove(double lufs) noexcept
    {
        const double position
            = std::ceil((lufs - kMinHistogramLUFS + 0.5 * kBinWidth) / kRangeCellWidth - 0.5);
        return static_cast<int>(std::clamp(position, 0.0, static_cast<double>(kNumRangeCells - 1)));
    }

    /// Loudness at the centre of a range cell.
    [[nodiscard]] static double rangeCellCentre(int cell) noexcept
    {
        return kMinHistogramLUFS - 0.5 * kBinWidth + (static_cast<double>(cell) + 0.5) * kRangeCellWidth;
    }

    /// Feeds one channel's samples to the true-peak estimator and returns
    /// `peak` raised to its finite readings. A sample's own magnitude counts
    /// at once.
    [[nodiscard]] T measureTruePeak(const T* x, int channel, int numSamples, T peak) noexcept
    {
        auto& state = tpChannels_[static_cast<size_t>(channel)];
        for (int start = 0; start < numSamples; start += kTruePeakChunk)
        {
            const int n = std::min(kTruePeakChunk, numSamples - start);
            // Only FINITE values join the max-hold: one Inf/NaN sample would
            // otherwise pin it at +Inf until reset() (the K-filter / histogram
            // path already self-recovers, so the true peak must too).
            for (int i = 0; i < n; ++i)
            {
                const T a = std::abs(x[start + i]);
                if (!std::isfinite(a))
                    invalidateMeasurement();
                else if (a > peak)
                    peak = a;
            }
            const T chunkPeak = truePeak_.processMax(state, x + start, n, peak);
            if (chunkPeak > peak)
                peak = chunkPeak;
        }
        return peak;
    }

    void invalidateMeasurement() noexcept
    {
        measurementValid_.store(false, std::memory_order_relaxed);
    }

    void incrementHistogram(std::atomic<uint32_t>& bin) noexcept
    {
        if (bin.load(std::memory_order_relaxed)
            == std::numeric_limits<uint32_t>::max())
        {
            invalidateMeasurement();
            return;
        }
        bin.fetch_add(1, std::memory_order_relaxed);
    }

    T calculateLUFSFromBlocks(int numBlocks) const noexcept
    {
        double sum = 0.0;
        int currentPos = blockWritePos_.load(std::memory_order_acquire);

        for (int i = 0; i < numBlocks; ++i)
        {
            int idx = (currentPos - 1 - i + 30) % 30;
            sum += blockPowers_[idx].load(std::memory_order_relaxed);
        }

        return static_cast<T>(powerToLUFS(sum / numBlocks));
    }

    /// First histogram bin whose centre is at or above `lufs`: a block stands
    /// for its bin, so a bin takes part in a gate when its centre clears it.
    [[nodiscard]] static int firstBinAtOrAbove(double lufs) noexcept
    {
        const double position = std::ceil((lufs - kMinHistogramLUFS) / kBinWidth);
        return static_cast<int>(std::clamp(position, 0.0, static_cast<double>(kNumBins - 1)));
    }

    [[nodiscard]] static double powerToLUFS(double meanPower) noexcept
    {
        if (meanPower <= 1e-10) return -100.0; // Prevent log10(0)
        return std::max(-100.0, -0.691 + 10.0 * std::log10(meanPower));
    }

    double sampleRate_ = 48000.0;
    std::atomic<bool> prepared_{ false };

    BiquadCoeff pre_, rlb_;
    BiquadState preState_[kMaxChannels], rlbState_[kMaxChannels];

    int blockSamples_ = 4800;
    double currentBlockPower_ = 0.0;
    int currentBlockSamples_ = 0;

    // Concurrency-safe components
    std::array<std::atomic<double>, 30> blockPowers_;
    std::atomic<int> blockWritePos_{0};
    int64_t totalCommittedBlocks_ = 0;

    // Constant-memory histogram for integrated loudness. The count decides the
    // gates; the energy beside it is the exact sum of the blocks in that bin.
    std::array<std::atomic<uint32_t>, kNumBins> histogram_;
    std::array<std::atomic<double>, kNumBins> histogramPower_;

    // EBU Tech 3342 loudness-range histogram (short-term values, 100 ms hop):
    // per 0.1 LU bin the count, the exact energy and the 0.005 LU cells.
    std::array<std::atomic<uint32_t>, kNumBins> lraHistogram_;
    std::array<std::atomic<double>, kNumBins> lraEnergy_;
    std::array<std::atomic<uint16_t>, kNumRangeCells> lraCells_;

    // Sticky for one reset-to-reset measurement transaction.
    std::atomic<bool> measurementValid_ { true };

    // BS.1770 true-peak (dBTP) tracking
    TruePeakEstimator<T> truePeak_;
    std::array<typename TruePeakEstimator<T>::Channel, kMaxChannels> tpChannels_ {};
    std::atomic<T> truePeakMax_ { T(0) };
};

} // namespace dspark
