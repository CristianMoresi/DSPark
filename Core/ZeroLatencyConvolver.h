// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file ZeroLatencyConvolver.h
 * @brief Zero-latency non-uniform partitioned convolution (Gardner scheme).
 *
 * Convolves arbitrarily long impulse responses with **zero input/output
 * delay** and **flat CPU usage**, after W. G. Gardner, "Efficient Convolution
 * without Input-Output Delay", JAES 43(3), 1995. The IR is split into
 * regions, each handled by the cheapest method that meets its deadline:
 *
 *   [0, 128)        direct FIR head (SIMD dot product)      -- latency 0
 *   [128, 2048)     uniform partitioned FFT, block 128      -- result lands
 *                   exactly when needed (offset == its block latency)
 *   [2048, end)     **growing** FFT partitions, time-distributed: two of
 *                   1024 at [2048, 4096), two of 2048 at [4096, 8192), and so
 *                   on (at most 8192); the last level takes the rest of the
 *                   IR, and it is chosen per IR length to minimise the work
 *                   per sample, so short tails stay on 1024. Each
 *                   level's cycle of work (FFT + per-partition spectral MACs +
 *                   IFFT + overlap-add) is split into units executed under a
 *                   per-sample budget across the following block, so long
 *                   IRs never spike a single audio callback
 *
 * Every tail level starts at twice its block size, which is what creates
 * the full block of scheduling slack the time distribution relies on -- the
 * core idea of Gardner's non-uniform scheme. Doubling the partition size
 * down the IR (after Garcia, "Optimal Filter Partition for Efficient
 * Convolution with Short Input/Output Delay", AES 2002) cuts the per-sample
 * spectral work of long IRs: a 20 s IR at 48 kHz takes 122 partitions in
 * four levels instead of 936 of 1024 (mean cost per 64-sample block 2.9 us
 * against 10.4 us measured; the largest single block 16 us against 34).
 *
 * Compared to the uniform Convolver (latency = block size), the price of
 * zero latency is the direct head (~128 MACs per sample). Use this class for
 * monitoring paths, cabinet simulation and anywhere latency matters; the
 * plain Convolver remains the cheapest choice when block-size latency is
 * acceptable.
 *
 * Mono engine: one instance convolves one channel (like Convolver).
 *
 * Dependencies: Convolver.h, FFT.h, SimdOps.h, AudioBuffer.h.
 *
 * @code
 *   dspark::ZeroLatencyConvolver<float> conv;
 *   conv.prepare(ir.getChannel(0), irLength);
 *
 *   // In the audio callback -- any block size, even 1 sample:
 *   conv.processInPlace(data, numSamples);   // y[0] already includes ir[0]*x[0]
 * @endcode
 */

#include "AudioBuffer.h"
#include "Convolver.h"
#include "FFT.h"
#include "SimdOps.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class ZeroLatencyConvolver
 * @brief Gardner-style non-uniform partitioned convolver (zero latency, flat CPU).
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class ZeroLatencyConvolver
{
public:
    /**
     * @brief Prepares the convolver with an impulse response (allocates).
     *
     * @param irData   Impulse response samples.
     * @param irLength Number of IR samples (any length >= 1).
     * @param headSize Direct-convolution head length, rounded to a power of
     *                 two in [32, 512] (default 128). Larger heads cost more
     *                 per sample; smaller heads grow the uniform mid level.
     */
    void prepare(const T* irData, int irLength, int headSize = 128)
    {
        prepared_ = false;
        if (irData == nullptr || irLength <= 0) return;

        headSize_ = 32;
        while (headSize_ < std::clamp(headSize, 32, 512)) headSize_ <<= 1;

        // --- head: ir[0, headLen) stored reversed for a linear dot product ---
        headLen_ = std::min(irLength, headSize_);
        headRev_.assign(static_cast<size_t>(headLen_), T(0));
        for (int i = 0; i < headLen_; ++i)
            headRev_[static_cast<size_t>(i)] = irData[headLen_ - 1 - i];

        // --- mid: uniform partitioned convolver, block == headSize ----------
        // Its block latency equals this region's offset, so its (delayed)
        // output is exactly the contribution of ir[headSize, tailStart).
        const int midEnd = std::min(irLength, kTailStart);
        hasMid_ = irLength > headSize_;
        if (hasMid_)
        {
            mid_.prepare(headSize_, irData + headSize_, midEnd - headSize_);
            midScratch_.assign(static_cast<size_t>(headSize_), T(0));
        }

        // --- tail: time-distributed levels of growing partitions -------------
        // Level j has block B = kTailBlock * 2^j and starts at 2 B; every
        // level but the last holds two partitions, so the next one starts at
        // 4 B = 2 (2 B). The last level (block kMaxTailBlock, or wherever
        // the IR ends) takes all remaining partitions.
        hasTail_ = irLength > kTailStart;
        numTailParts_ = 0;
        levels_.clear();
        int topBlock = kTailBlock;
        if (hasTail_)
        {
            const int topLevel = chooseTopLevel(irLength);
            int offset = kTailStart;
            int block = kTailBlock;
            while (offset < irLength)
            {
                const int remaining = irLength - offset;
                const bool top = static_cast<int>(levels_.size()) >= topLevel;
                TailLevel lv;
                lv.block = block;
                lv.numParts = std::max(1, (remaining + block - 1) / block);
                if (!top) lv.numParts = std::min(lv.numParts, 2);
                lv.totalUnits = lv.numParts + 3;   // FFT + K MACs + IFFT + scatter
                lv.spec = 2 * block + 2;
                lv.delay = std::min(static_cast<int>(levels_.size()) * (kTailBlock / 8), block / 4);

                lv.fft = std::make_unique<FFTReal<T>>(static_cast<size_t>(2 * block));
                lv.ir.assign(static_cast<size_t>(lv.numParts) * static_cast<size_t>(lv.spec), T(0));
                lv.fdl.assign(static_cast<size_t>(lv.numParts) * static_cast<size_t>(lv.spec), T(0));
                lv.accum.assign(static_cast<size_t>(lv.spec), T(0));
                lv.scratch.assign(static_cast<size_t>(2 * block), T(0));

                for (int p = 0; p < lv.numParts; ++p)
                {
                    const int start = offset + p * block;
                    const int len = std::min(block, irLength - start);
                    std::fill(lv.scratch.begin(), lv.scratch.end(), T(0));
                    std::copy_n(irData + start, len, lv.scratch.begin());
                    lv.fft->forward(lv.scratch.data(), &lv.ir[static_cast<size_t>(p) * static_cast<size_t>(lv.spec)]);
                }

                numTailParts_ += lv.numParts;
                topBlock = block;
                offset += lv.numParts * block;
                levels_.push_back(std::move(lv));
                block *= 2;
            }

            // Output ring: a level scatters [taskStart + B, taskStart + 3 B)
            // while the read position is at least taskStart, so 4 B of the
            // largest level never wrap onto unread samples.
            int outSize = 1;
            while (outSize < 4 * topBlock) outSize <<= 1;
            tailOut_.assign(static_cast<size_t>(outSize), T(0));
            outMask_ = outSize - 1;
        }

        // --- shared input history ring ---------------------------------------
        int ringSize = 2 * headSize_;
        if (hasTail_) ringSize = std::max(ringSize, 4 * topBlock);
        int pow2 = 1;
        while (pow2 < ringSize) pow2 <<= 1;
        inRing_.assign(static_cast<size_t>(pow2), T(0));
        inMask_ = pow2 - 1;

        prepared_ = true;
        reset();
    }

    /** @brief Clears all signal state (keeps the IR). Safe on the audio thread. */
    void reset() noexcept
    {
        if (!prepared_) return;
        std::fill(inRing_.begin(), inRing_.end(), T(0));
        std::fill(tailOut_.begin(), tailOut_.end(), T(0));
        if (hasMid_) mid_.reset();
        absPos_ = static_cast<int64_t>(inRing_.size());   // keep indices positive
        for (auto& lv : levels_)
        {
            std::fill(lv.fdl.begin(), lv.fdl.end(), T(0));
            lv.cyclePos = 0;
            lv.fdlIndex = 0;
            lv.taskPhase = lv.totalUnits;                  // idle
            lv.taskStart = 0;
        }
    }

    /**
     * @brief Convolves out-of-place. Works with any numSamples, even 1.
     * @param input      Input samples.
     * @param output     Output samples (may alias input).
     * @param numSamples Number of samples.
     */
    void process(const T* input, T* output, int numSamples) noexcept
    {
        if (!prepared_)
        {
            if (output != input)
                std::memmove(output, input, static_cast<size_t>(numSamples) * sizeof(T));
            return;
        }

        int i = 0;
        while (i < numSamples)
        {
            int chunk = std::min(numSamples - i, headSize_);
            // Every level's block is a multiple of the first level's, so a
            // chunk that stops at its boundary stops at all of them.
            if (hasTail_)
                chunk = std::min(chunk, levels_[0].block - levels_[0].cyclePos);

            // Mid level reads the pristine input before output overwrites it.
            if (hasMid_)
                mid_.process(input + i, midScratch_.data(), chunk);

            for (int k = 0; k < chunk; ++k)
            {
                inRing_[static_cast<size_t>(absPos_ & inMask_)] = input[i + k];

                T y = headDot();
                if (hasMid_) y += midScratch_[static_cast<size_t>(k)];
                if (hasTail_)
                {
                    auto& slot = tailOut_[static_cast<size_t>(absPos_ & outMask_)];
                    y += slot;
                    slot = T(0);   // clear-on-read keeps the ring self-cleaning
                }
                output[i + k] = y;
                ++absPos_;
            }

            if (hasTail_)
            {
                for (auto& lv : levels_)
                {
                    lv.cyclePos += chunk;
                    // Execute this cycle's proportional share of the pending
                    // task. 64-bit product: totalUnits grows with the
                    // partition count and would wrap int for hours-long IRs.
                    // Each level starts its share a little later than the one
                    // before (lv.delay), so the levels that launch on the same
                    // boundary do not run their transforms in the same callback.
                    const int64_t span = lv.block - lv.delay;
                    const int64_t done = std::max<int64_t>(0, lv.cyclePos - lv.delay);
                    runTaskUnits(lv, static_cast<int>(
                        (static_cast<int64_t>(lv.totalUnits) * done + span - 1) / span));
                    if (lv.cyclePos >= lv.block)
                    {
                        lv.cyclePos = 0;   // task is fully drained (quota == total)
                        launchTask(lv);
                    }
                }
            }
            i += chunk;
        }
    }

    /** @brief Convolves in-place. */
    void processInPlace(T* data, int numSamples) noexcept
    {
        process(data, data, numSamples);
    }

    /**
     * @brief Processes channel 0 of a buffer in-place (unified API).
     * Mono engine -- use one instance per channel for multichannel.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumChannels() > 0 && buffer.getNumSamples() > 0)
            processInPlace(buffer.getChannel(0), buffer.getNumSamples());
    }

    /** @brief Zero -- that is the point. */
    [[nodiscard]] static constexpr int getLatency() noexcept { return 0; }

    /** @brief Resolved direct-head length in samples. */
    [[nodiscard]] int getHeadSize() const noexcept { return headSize_; }

    /** @brief Number of FFT partitions in the time-distributed tail (all levels). */
    [[nodiscard]] int getNumTailPartitions() const noexcept { return numTailParts_; }

    /** @brief Number of tail levels (partition sizes 1024, 2048, ...). */
    [[nodiscard]] int getNumTailLevels() const noexcept { return static_cast<int>(levels_.size()); }

private:
    static constexpr int kTailBlock = 1024;            ///< First tail level's partition size.
    static constexpr int kTailStart = 2 * kTailBlock;  ///< Tail offset (= scheduling slack).
    static constexpr int kMaxTailBlock = 8192;        ///< Largest partition (the last level).

    /// One tail level: partitions of one size, its own transform and task.
    struct TailLevel
    {
        int block = 0;       ///< Partition size B (the level starts at 2 B).
        int numParts = 0;    ///< Partitions in this level.
        int totalUnits = 0;  ///< Task units per cycle: FFT + K MACs + IFFT + scatter.
        int spec = 0;        ///< Scalars per spectrum (2 B + 2).
        int delay = 0;       ///< Samples into the cycle before its work starts (staggering).
        std::unique_ptr<FFTReal<T>> fft;
        std::vector<T> ir;       ///< K partition spectra (contiguous).
        std::vector<T> fdl;      ///< K input spectra (frequency-domain delay line).
        std::vector<T> accum;
        std::vector<T> scratch;
        int cyclePos = 0;
        int fdlIndex = 0;
        int taskPhase = 0;
        int64_t taskStart = 0;
    };

    /** @brief Direct head: dot of reversed head taps with the input history. */
    [[nodiscard]] T headDot() const noexcept
    {
        const int64_t start = absPos_ - headLen_ + 1;
        const int s0 = static_cast<int>(start & inMask_);
        const int ringSize = inMask_ + 1;
        const int first = std::min(headLen_, ringSize - s0);

        T y = simd::dotProduct(headRev_.data(), &inRing_[static_cast<size_t>(s0)], first);
        if (first < headLen_)
            y += simd::dotProduct(headRev_.data() + first, inRing_.data(), headLen_ - first);
        return y;
    }

    /**
     * @brief Index of the last tail level for an IR length: the partition
     *        plan with the least work per sample.
     *
     * Per sample, a level costs about one complex multiply per partition plus
     * its two transforms spread over its block (weighed as 1.5 x log2 of
     * their size, the measured ratio the uniform Convolver uses too). Every
     * level below the last holds two partitions; the last takes the rest.
     * Doubling pays only once the IR is long enough to fill the larger
     * levels, so short tails stay on small partitions (and small transforms).
     */
    [[nodiscard]] static int chooseTopLevel(int irLength) noexcept
    {
        double below = 0.0;   // cost of the full levels under the candidate top
        double best = 1e300;
        int bestLevel = 0;
        int offset = kTailStart;
        int level = 0;
        for (int block = kTailBlock; block <= kMaxTailBlock && offset < irLength; block *= 2, ++level)
        {
            const double fft = 1.5 * std::log2(2.0 * block);
            const double parts = std::ceil(static_cast<double>(irLength - offset) / block);
            const double cost = below + parts + fft;
            if (cost < best) { best = cost; bestLevel = level; }
            if (irLength - offset <= 2 * block) break; // nothing would remain above
            below += 2.0 + fft;
            offset += 2 * block;
        }
        return bestLevel;
    }

    /** @brief Starts a level's cycle task for input block [absPos_ - B, absPos_). */
    void launchTask(TailLevel& lv) noexcept
    {
        lv.taskStart = absPos_;
        lv.fdlIndex = (lv.fdlIndex + 1) % lv.numParts;
        lv.taskPhase = 0;
    }

    /**
     * @brief Executes pending task units until `target` units are done.
     *
     * Unit 0: forward FFT of the task's input block into the FDL (and clears
     * the spectral accumulator). Units 1..K: one complex multiply-accumulate
     * per IR partition. Unit K+1: inverse FFT. Unit K+2: overlap-add scatter
     * into the shared tail output ring at [taskStart + B, taskStart + 3 B).
     */
    void runTaskUnits(TailLevel& lv, int target) noexcept
    {
        const int B = lv.block;
        const size_t spec = static_cast<size_t>(lv.spec);
        target = std::min(target, lv.totalUnits);
        while (lv.taskPhase < target)
        {
            const int unit = lv.taskPhase++;
            if (unit == 0)
            {
                const int64_t blockStart = lv.taskStart - B;
                const int s0 = static_cast<int>(blockStart & inMask_);
                const int ringSize = inMask_ + 1;
                const int first = std::min(B, ringSize - s0);
                std::copy_n(&inRing_[static_cast<size_t>(s0)], first, lv.scratch.begin());
                if (first < B)
                    std::copy_n(inRing_.data(), B - first, lv.scratch.begin() + first);
                std::fill(lv.scratch.begin() + B, lv.scratch.end(), T(0));

                lv.fft->forward(lv.scratch.data(), &lv.fdl[static_cast<size_t>(lv.fdlIndex) * spec]);
                std::fill(lv.accum.begin(), lv.accum.end(), T(0));
            }
            else if (unit <= lv.numParts)
            {
                const int p = unit - 1;
                const int slot = (lv.fdlIndex - p + lv.numParts) % lv.numParts;
                simd::complexMulAccum(lv.accum.data(), &lv.fdl[static_cast<size_t>(slot) * spec],
                                      &lv.ir[static_cast<size_t>(p) * spec], B + 1);
            }
            else if (unit == lv.numParts + 1)
            {
                lv.fft->inverse(lv.accum.data(), lv.scratch.data());
            }
            else
            {
                // Overlap-add in at most two contiguous runs of the ring.
                const int ringSize = outMask_ + 1;
                const int d0 = static_cast<int>((lv.taskStart + B) & outMask_);
                const int first = std::min(2 * B, ringSize - d0);
                T* out = tailOut_.data();
                const T* src = lv.scratch.data();
                for (int m = 0; m < first; ++m) out[d0 + m] += src[m];
                for (int m = first; m < 2 * B; ++m) out[m - first] += src[m];
            }
        }
    }

    // -- Members -----------------------------------------------------------------
    bool prepared_ = false;

    int headSize_ = 128;
    int headLen_ = 0;
    std::vector<T> headRev_;

    bool hasMid_ = false;
    Convolver<T> mid_;
    std::vector<T> midScratch_;

    bool hasTail_ = false;
    int numTailParts_ = 0;
    std::vector<TailLevel> levels_; ///< Tail levels, partition size doubling.
    std::vector<T> tailOut_;      ///< Tail OLA output ring shared by the levels (clear-on-read).
    int outMask_ = 0;

    std::vector<T> inRing_;       ///< Shared input history (head + tail FFT source).
    int inMask_ = 0;
    int64_t absPos_ = 0;
};

} // namespace dspark
