// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file ZeroLatencyConvolver.h
 * @brief Zero-latency non-uniform partitioned convolution (Gardner scheme).
 *
 * Convolves arbitrarily long impulse responses with **zero input/output
 * delay** and **flat CPU usage**, after W. G. Gardner, "Efficient Convolution
 * without Input-Output Delay", JAES 43(3), 1995. The IR is split into three
 * regions, each handled by the cheapest method that meets its deadline:
 *
 *   [0, 128)      direct FIR head (SIMD dot product)        -- latency 0
 *   [128, 2048)   uniform partitioned FFT, block 128        -- result lands
 *                 exactly when needed (offset == its block latency)
 *   [2048, end)   FFT partitions of 1024, **time-distributed**: each cycle's
 *                 work (FFT + per-partition spectral MACs + IFFT + overlap-
 *                 add) is split into units executed under a per-sample budget
 *                 across the following 1024 samples, so long IRs never spike
 *                 a single audio callback
 *
 * The tail partitions start at twice their block size, which is what creates
 * the full block of scheduling slack the time distribution relies on -- the
 * core idea of Gardner's non-uniform scheme.
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

        // --- tail: time-distributed partitions of kTailBlock ----------------
        hasTail_ = irLength > kTailStart;
        numTailParts_ = 0;
        if (hasTail_)
        {
            const int tailLen = irLength - kTailStart;
            numTailParts_ = (tailLen + kTailBlock - 1) / kTailBlock;
            totalUnits_ = numTailParts_ + 3;   // FFT + K MACs + IFFT + scatter

            tailFft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(2 * kTailBlock));

            tailIR_.assign(static_cast<size_t>(numTailParts_) * kTailSpec, T(0));
            tailFdl_.assign(static_cast<size_t>(numTailParts_) * kTailSpec, T(0));
            tailAccum_.assign(static_cast<size_t>(kTailSpec), T(0));
            tailScratch_.assign(static_cast<size_t>(2 * kTailBlock), T(0));
            tailOut_.assign(static_cast<size_t>(kTailOutSize), T(0));

            std::vector<T> padded(static_cast<size_t>(2 * kTailBlock), T(0));
            for (int p = 0; p < numTailParts_; ++p)
            {
                const int offset = kTailStart + p * kTailBlock;
                const int len = std::min(kTailBlock, irLength - offset);
                std::fill(padded.begin(), padded.end(), T(0));
                std::copy_n(irData + offset, len, padded.begin());
                tailFft_->forward(padded.data(), &tailIR_[static_cast<size_t>(p) * kTailSpec]);
            }
        }

        // --- shared input history ring ---------------------------------------
        int ringSize = 2 * headSize_;
        if (hasTail_) ringSize = std::max(ringSize, 4 * kTailBlock);
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
        std::fill(tailFdl_.begin(), tailFdl_.end(), T(0));
        std::fill(tailOut_.begin(), tailOut_.end(), T(0));
        if (hasMid_) mid_.reset();
        absPos_ = static_cast<int64_t>(inRing_.size());   // keep indices positive
        cyclePos_ = 0;
        fdlIndex_ = 0;
        taskPhase_ = totalUnits_;                          // idle
        taskStart_ = 0;
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
            if (hasTail_)
                chunk = std::min(chunk, kTailBlock - cyclePos_);

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
                    auto& slot = tailOut_[static_cast<size_t>(absPos_ & kTailOutMask)];
                    y += slot;
                    slot = T(0);   // clear-on-read keeps the ring self-cleaning
                }
                output[i + k] = y;
                ++absPos_;
            }

            if (hasTail_)
            {
                cyclePos_ += chunk;
                // Execute this cycle's proportional share of the pending task.
                // 64-bit product: totalUnits_ grows with the partition count,
                // and totalUnits_ * cyclePos_ would wrap int for hours-long IRs.
                runTaskUnits(static_cast<int>(
                    (static_cast<int64_t>(totalUnits_) * cyclePos_ + kTailBlock - 1) / kTailBlock));
                if (cyclePos_ >= kTailBlock)
                {
                    cyclePos_ = 0;   // task is fully drained (quota == total)
                    launchTask();
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

    /** @brief Number of FFT partitions in the time-distributed tail. */
    [[nodiscard]] int getNumTailPartitions() const noexcept { return numTailParts_; }

private:
    static constexpr int kTailBlock = 1024;            ///< Tail partition size.
    static constexpr int kTailStart = 2 * kTailBlock;  ///< Tail offset (= scheduling slack).
    static constexpr int kTailSpec  = 2 * kTailBlock + 2; ///< Floats per spectrum.
    static constexpr int kTailOutSize = 8 * kTailBlock;   ///< Tail OLA ring (pow2).
    static constexpr int kTailOutMask = kTailOutSize - 1;

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

    /** @brief Starts the tail cycle task for input block [absPos_-1024, absPos_). */
    void launchTask() noexcept
    {
        taskStart_ = absPos_;
        fdlIndex_ = (fdlIndex_ + 1) % numTailParts_;
        taskPhase_ = 0;
    }

    /**
     * @brief Executes pending task units until `target` units are done.
     *
     * Unit 0: forward FFT of the task's input block into the FDL (and clears
     * the spectral accumulator). Units 1..K: one complex multiply-accumulate
     * per IR partition. Unit K+1: inverse FFT. Unit K+2: overlap-add scatter
     * into the tail output ring at [taskStart + 1024, taskStart + 3072).
     */
    void runTaskUnits(int target) noexcept
    {
        target = std::min(target, totalUnits_);
        while (taskPhase_ < target)
        {
            const int unit = taskPhase_++;
            if (unit == 0)
            {
                const int64_t blockStart = taskStart_ - kTailBlock;
                const int s0 = static_cast<int>(blockStart & inMask_);
                const int ringSize = inMask_ + 1;
                const int first = std::min(kTailBlock, ringSize - s0);
                std::copy_n(&inRing_[static_cast<size_t>(s0)], first, tailScratch_.begin());
                if (first < kTailBlock)
                    std::copy_n(inRing_.data(), kTailBlock - first, tailScratch_.begin() + first);
                std::fill(tailScratch_.begin() + kTailBlock, tailScratch_.end(), T(0));

                tailFft_->forward(tailScratch_.data(),
                                  &tailFdl_[static_cast<size_t>(fdlIndex_) * kTailSpec]);
                std::fill(tailAccum_.begin(), tailAccum_.end(), T(0));
            }
            else if (unit <= numTailParts_)
            {
                const int p = unit - 1;
                const int slot = (fdlIndex_ - p + numTailParts_) % numTailParts_;
                simd::complexMulAccum(tailAccum_.data(),
                                      &tailFdl_[static_cast<size_t>(slot) * kTailSpec],
                                      &tailIR_[static_cast<size_t>(p) * kTailSpec],
                                      kTailBlock + 1);
            }
            else if (unit == numTailParts_ + 1)
            {
                tailFft_->inverse(tailAccum_.data(), tailScratch_.data());
            }
            else
            {
                const int64_t dst = taskStart_ + kTailBlock;
                for (int m = 0; m < 2 * kTailBlock; ++m)
                    tailOut_[static_cast<size_t>((dst + m) & kTailOutMask)]
                        += tailScratch_[static_cast<size_t>(m)];
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
    int totalUnits_ = 0;
    std::unique_ptr<FFTReal<T>> tailFft_;
    std::vector<T> tailIR_;       ///< K partition spectra (contiguous).
    std::vector<T> tailFdl_;      ///< K input spectra (frequency-domain delay line).
    std::vector<T> tailAccum_;
    std::vector<T> tailScratch_;
    std::vector<T> tailOut_;      ///< Tail OLA output ring (clear-on-read).

    std::vector<T> inRing_;       ///< Shared input history (head + tail FFT source).
    int inMask_ = 0;
    int64_t absPos_ = 0;

    int cyclePos_ = 0;
    int fdlIndex_ = 0;
    int64_t taskStart_ = 0;
    int taskPhase_ = 0;
};

} // namespace dspark
