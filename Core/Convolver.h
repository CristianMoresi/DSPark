// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Convolver.h
 * @brief FFT-based partitioned convolution for long impulse responses.
 *
 * Implements the **uniform partitioned overlap-save** algorithm, which is the
 * standard approach for real-time convolution with long impulse responses
 * (reverb IRs, cabinet simulations, etc.). Breaks the IR into blocks, transforms
 * each block with FFT, and convolves in the frequency domain.
 *
 * Complexity: O(N log N) per block, vs O(N * M) for direct convolution.
 * For a 2-second reverb IR at 48 kHz (96000 samples), direct convolution would
 * require ~96000 multiplies per sample. FFT-based convolution needs ~20.
 *
 * Partitioning::NonUniform keeps the latency at the block size but convolves
 * the late part of a long IR in larger partitions (after Gardner, "Efficient
 * Convolution without Input-Output Delay", JAES 1995): a uniform head of
 * block-size partitions covers the first 2 x tail - block samples, and the
 * tail's partitions are processed with their work spread evenly over the
 * samples of each tail block. A long IR at a small block size then costs a
 * fraction of the uniform engine, with no spike in any block.
 *
 * Dependencies: FFT.h, AudioBuffer.h, AudioSpec.h, SimdOps.h.
 *
 * @code
 *   // Load an impulse response from a WAV file:
 *   dspark::WavFile wav;
 *   wav.openRead("cabinet.wav");
 *   auto info = wav.getInfo();
 *   dspark::AudioBuffer<float> ir;
 *   ir.resize(1, static_cast<int>(info.numSamples));
 *   wav.readSamples(ir.toView());
 *   wav.close();
 *
 *   // Set up the convolver:
 *   dspark::Convolver<float> conv;
 *   conv.prepare(512, ir.getChannel(0), static_cast<int>(info.numSamples));
 *
 *   // In your audio callback:
 *   conv.process(inputBlock, outputBlock, blockSize);
 * @endcode
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "FFT.h"
#include "SimdOps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class Convolver
 * @brief Real-time partitioned convolution using overlap-save with FFT.
 *
 * The impulse response is divided into partitions of size `blockSize`. Each
 * partition is pre-transformed to the frequency domain. During processing,
 * each input block is also transformed, multiplied with all IR partitions,
 * and the results are accumulated with appropriate delays.
 *
 * The block size determines both the FFT size (2 * blockSize for overlap-save)
 * and the processing latency (= blockSize samples).
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class Convolver
{
public:
    /** @brief How the impulse response is partitioned (see the file notes). */
    enum class Partitioning
    {
        Uniform,    ///< Every partition the block size (default).
        NonUniform  ///< Block-size head, larger time-distributed tail partitions.
    };

    /**
     * @brief Prepares the convolver with an impulse response.
     *
     * This is the only method that allocates memory. Call once during setup.
     *
     * @param blockSize   Processing block size. Should be a power of two >= 2
     *                    (anything else rounds UP release-safely; check
     *                    getBlockSize()). Determines latency and FFT size.
     *                    Typical: 128, 256, 512.
     * @param irData      Pointer to the impulse response samples.
     * @param irLength    Number of samples in the impulse response.
     * @param partitioning Uniform (default) or NonUniform. The latency is the
     *                    block size either way, and the output the same
     *                    convolution up to rounding. NonUniform only splits an
     *                    IR long enough for it to pay.
     */
    void prepare(int blockSize, const T* irData, int irLength,
                 Partitioning partitioning = Partitioning::Uniform)
    {
        assert(blockSize >= 2 && (blockSize & (blockSize - 1)) == 0);
        assert(irLength > 0);
        assert(irData != nullptr);
        if (irData == nullptr || irLength <= 0) return;

        // Release-safe normalisation: the FFT needs a power of two >= 4, so
        // the block needs a power of two >= 2. Round up (never down: rounding
        // down would shrink the valid overlap-save region below the block).
        int bs = 2;
        while (bs < blockSize) bs <<= 1;

        blockSize_ = bs;
        fftSize_ = bs * 2;
        numBins_ = fftSize_ + 2; // (N/2 + 1) complex bins, interleaved = N+2 scalars

        // NonUniform: the head takes h[0, 2 x tail - bs), the tail the rest.
        tailBlock_ = partitioning == Partitioning::NonUniform ? chooseTailBlock(bs, irLength) : 0;
        const int headLength = tailBlock_ > 0 ? 2 * tailBlock_ - bs : irLength;

        // Create FFT processor
        fft_ = std::make_unique<FFTReal<T>>(fftSize_);

        // Partition the IR into blocks
        numPartitions_ = (headLength + bs - 1) / bs;

        // Pre-transform each IR partition
        irPartitions_.resize(static_cast<size_t>(numPartitions_));
        std::vector<T> paddedBlock(static_cast<size_t>(fftSize_), T(0));

        for (int p = 0; p < numPartitions_; ++p)
        {
            int offset = p * bs;
            int remaining = std::min(bs, headLength - offset);

            // Zero-pad the block to fftSize
            std::fill(paddedBlock.begin(), paddedBlock.end(), T(0));
            std::copy_n(irData + offset, remaining, paddedBlock.begin());

            // Forward FFT and store
            irPartitions_[static_cast<size_t>(p)].resize(static_cast<size_t>(numBins_));
            fft_->forward(paddedBlock.data(), irPartitions_[static_cast<size_t>(p)].data());
        }

        // Allocate working buffers
        inputBuffer_.assign(static_cast<size_t>(fftSize_), T(0));
        outputBuffer_.assign(static_cast<size_t>(fftSize_), T(0));
        overlapBuffer_.assign(static_cast<size_t>(blockSize_), T(0));
        fftAccum_.assign(static_cast<size_t>(numBins_), T(0));

        // Frequency-domain delay line (stores past input FFTs)
        fdlBuffer_.resize(static_cast<size_t>(numPartitions_));
        for (auto& fdl : fdlBuffer_)
            fdl.assign(static_cast<size_t>(numBins_), T(0));

        prepareTail(irData + headLength, irLength - headLength);

        fdlIndex_ = 0;
        inputPos_ = 0;
        resetTail();
    }

    /** @brief Resets the convolution state (clears delay lines and buffers). */
    void reset() noexcept
    {
        std::fill(inputBuffer_.begin(), inputBuffer_.end(), T(0));
        std::fill(outputBuffer_.begin(), outputBuffer_.end(), T(0));
        std::fill(overlapBuffer_.begin(), overlapBuffer_.end(), T(0));
        for (auto& fdl : fdlBuffer_)
            std::fill(fdl.begin(), fdl.end(), T(0));
        fdlIndex_ = 0;
        inputPos_ = 0;
        resetTail();
    }

    /**
     * @brief Processes a block of audio through the convolver.
     *
     * `numSamples` may exceed the convolution block size: the run loop fires a
     * partition transform at every block boundary, so arbitrary lengths are
     * handled. `input` and `output` may be the SAME buffer (full in-place);
     * partially overlapping buffers are not supported.
     *
     * @param input      Input audio samples.
     * @param output     Output audio samples (convolved result).
     * @param numSamples Number of samples to process.
     */
    void process(const T* input, T* output, int numSamples) noexcept
    {
        if (blockSize_ == 0) // not prepared: honest pass-through
        {
            if (output != input)
                std::copy_n(input, numSamples, output);
            return;
        }

        // Copy in contiguous runs up to the next block boundary instead of
        // sample-by-sample (this is the per-channel hot path of Reverb).
        int i = 0;
        while (i < numSamples)
        {
            const int run = std::min(numSamples - i, blockSize_ - inputPos_);

            // Stage the input run first, then overwrite the caller's samples
            // with the previous block's result -- this ordering is what makes
            // full in-place (input == output) legal.
            std::copy_n(input + i, run,
                        inputBuffer_.begin() + blockSize_ + inputPos_);
            if (tailBlock_ > 0)
                for (int k = 0; k < run; ++k)
                    inRing_[static_cast<size_t>((absPos_ + k) & inMask_)] = input[i + k];
            std::copy_n(overlapBuffer_.begin() + inputPos_, run, output + i);
            if (tailBlock_ > 0)
                addTail(output + i, run);

            inputPos_ += run;
            i += run;

            if (inputPos_ >= blockSize_)
            {
                processPartitionBlock();
                inputPos_ = 0;

                std::copy_n(outputBuffer_.begin() + blockSize_, static_cast<size_t>(blockSize_),
                           overlapBuffer_.begin());
            }
        }
    }

    /**
     * @brief Processes audio in-place (output overwrites input).
     *
     * @param data       Audio samples (input and output).
     * @param numSamples Number of samples.
     */
    void processInPlace(T* data, int numSamples) noexcept
    {
        process(data, data, numSamples);
    }

    /**
     * @brief Prepares with AudioSpec and IR data (unified API).
     *
     * Uses spec.maxBlockSize as the convolution block size.
     *
     * @param spec     Audio environment specification.
     * @param irData   Impulse response samples.
     * @param irLength Number of IR samples.
     *
     * @note If `spec.maxBlockSize` is not already a power of two the effective
     *       FFT block rounds up silently (e.g. 300 -> 512), doubling the
     *       effective latency. Use `getBlockSize()` after prepare to inspect
     *       the resolved value, or pre-set `maxBlockSize` to a power of two.
     */
    void prepare(const AudioSpec& spec, const T* irData, int irLength)
    {
        prepare(spec.maxBlockSize, irData, irLength);
    }

    /**
     * @brief Processes an audio buffer in-place (unified API).
     *
     * **Important:** Convolver is a mono processor. This method processes
     * only channel 0. For multi-channel convolution, use separate Convolver
     * instances per channel, or use the Reverb class.
     *
     * @param buffer Audio data (only channel 0 is convolved).
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (buffer.getNumChannels() > 0 && buffer.getNumSamples() > 0)
            processInPlace(buffer.getChannel(0), buffer.getNumSamples());
    }

    /** @brief Returns the processing latency in samples (= block size). */
    [[nodiscard]] int getLatency() const noexcept { return blockSize_; }

    /** @brief Returns the block size. */
    [[nodiscard]] int getBlockSize() const noexcept { return blockSize_; }

    /** @brief Returns the number of IR partitions: the block-size head
     *  partitions, plus the larger tail partitions in NonUniform mode. */
    [[nodiscard]] int getNumPartitions() const noexcept { return numPartitions_ + numTailParts_; }

    /** @brief Tail partition size in NonUniform mode, 0 when the IR is
     *  convolved in block-size partitions only. */
    [[nodiscard]] int getTailBlockSize() const noexcept { return tailBlock_; }

private:
    /**
     * @brief Tail partition size for a block size and IR length, or 0 when
     *        the uniform engine is as cheap.
     *
     * Per sample, the head costs about one complex multiply per head
     * partition and the tail about one per tail partition, plus the tail's
     * two FFTs spread over its block (weighed here as 1.5 x log2 of their
     * size, close to their measured cost). The tail block is the power of two
     * of at least 4 x the block size that minimises the sum.
     */
    static int chooseTailBlock(int bs, int irLength) noexcept
    {
        const double uniform = static_cast<double>(irLength) / bs;
        double best = 0.8 * uniform;   // only split where it clearly pays
        int choice = 0;
        for (int t = 4 * bs; t <= kMaxTailBlock; t <<= 1)
        {
            const int head = 2 * t - bs;
            if (irLength <= head + t) break;   // fewer than two tail partitions
            const double tailParts = std::ceil(static_cast<double>(irLength - head) / t);
            const double cost = static_cast<double>(head) / bs + tailParts
                                + 1.5 * std::log2(2.0 * t);
            if (cost < best) { best = cost; choice = t; }
        }
        return choice;
    }

    /** @brief Builds the tail partitions (and their buffers) for h[head, end). */
    void prepareTail(const T* tailIr, int tailLength)
    {
        numTailParts_ = 0;
        totalUnits_ = 0;
        if (tailBlock_ == 0 || tailLength <= 0)
        {
            tailBlock_ = 0;
            tailFft_.reset();
            tailIR_.clear(); tailFdl_.clear(); tailAccum_.clear();
            tailScratch_.clear(); tailOut_.clear(); inRing_.clear();
            return;
        }
        const int tb = tailBlock_;
        const std::size_t spec = static_cast<std::size_t>(2 * tb + 2);
        numTailParts_ = (tailLength + tb - 1) / tb;
        totalUnits_ = numTailParts_ + 3;   // FFT + K MACs + IFFT + scatter

        tailFft_ = std::make_unique<FFTReal<T>>(2 * tb);
        tailIR_.assign(static_cast<std::size_t>(numTailParts_) * spec, T(0));
        tailFdl_.assign(static_cast<std::size_t>(numTailParts_) * spec, T(0));
        tailAccum_.assign(spec, T(0));
        tailScratch_.assign(static_cast<std::size_t>(2 * tb), T(0));
        tailOut_.assign(static_cast<std::size_t>(8 * tb), T(0));
        inRing_.assign(static_cast<std::size_t>(4 * tb), T(0));
        outMask_ = 8 * tb - 1;
        inMask_ = 4 * tb - 1;

        std::vector<T> padded(static_cast<std::size_t>(2 * tb), T(0));
        for (int p = 0; p < numTailParts_; ++p)
        {
            const int len = std::min(tb, tailLength - p * tb);
            std::fill(padded.begin(), padded.end(), T(0));
            std::copy_n(tailIr + static_cast<std::size_t>(p) * tb, len, padded.begin());
            tailFft_->forward(padded.data(), &tailIR_[static_cast<std::size_t>(p) * spec]);
        }
    }

    void resetTail() noexcept
    {
        std::fill(tailFdl_.begin(), tailFdl_.end(), T(0));
        std::fill(tailOut_.begin(), tailOut_.end(), T(0));
        std::fill(inRing_.begin(), inRing_.end(), T(0));
        absPos_ = static_cast<std::int64_t>(inRing_.size());   // keep indices positive
        cyclePos_ = 0;
        tailFdlIndex_ = 0;
        taskPhase_ = totalUnits_;   // idle until the first tail block is in
        taskStart_ = 0;
    }

    /**
     * @brief Adds the tail's contribution to `count` output samples and runs
     *        this run's share of the pending tail task.
     *
     * The tail is computed on the block-delayed timeline: its partitions hold
     * h[2 x tail - block, end), i.e. the IR shifted by the block latency
     * starts exactly two tail blocks in, which is the scheduling slack of the
     * time-distributed scheme. The runs never cross a tail-block boundary,
     * because the tail block is a multiple of the block size.
     */
    void addTail(T* out, int count) noexcept
    {
        for (int k = 0; k < count; ++k)
        {
            T& slot = tailOut_[static_cast<size_t>((absPos_ + k) & outMask_)];
            out[k] += slot;
            slot = T(0);   // clear-on-read keeps the ring self-cleaning
        }
        absPos_ += count;
        cyclePos_ += count;
        // This cycle's proportional share of the task (64-bit product).
        runTaskUnits(static_cast<int>((static_cast<std::int64_t>(totalUnits_) * cyclePos_
                                       + tailBlock_ - 1) / tailBlock_));
        if (cyclePos_ >= tailBlock_)
        {
            cyclePos_ = 0;   // the task is fully drained (quota == total)
            taskStart_ = absPos_;
            tailFdlIndex_ = (tailFdlIndex_ + 1) % numTailParts_;
            taskPhase_ = 0;
        }
    }

    /**
     * @brief Runs pending tail units until `target` are done. Unit 0: FFT of
     *        the task's input block into the FDL; units 1..K: one partition's
     *        spectral multiply-accumulate each; unit K+1: inverse FFT; unit
     *        K+2: overlap-add into the output ring at [start + tail, start + 3 tail).
     */
    void runTaskUnits(int target) noexcept
    {
        const int tb = tailBlock_;
        const std::size_t spec = static_cast<std::size_t>(2 * tb + 2);
        target = std::min(target, totalUnits_);
        while (taskPhase_ < target)
        {
            const int unit = taskPhase_++;
            if (unit == 0)
            {
                const std::int64_t blockStart = taskStart_ - tb;
                const int s0 = static_cast<int>(blockStart & inMask_);
                const int first = std::min(tb, inMask_ + 1 - s0);
                std::copy_n(&inRing_[static_cast<size_t>(s0)], first, tailScratch_.begin());
                if (first < tb)
                    std::copy_n(inRing_.data(), tb - first, tailScratch_.begin() + first);
                std::fill(tailScratch_.begin() + tb, tailScratch_.end(), T(0));
                tailFft_->forward(tailScratch_.data(),
                                  &tailFdl_[static_cast<size_t>(tailFdlIndex_) * spec]);
                std::fill(tailAccum_.begin(), tailAccum_.end(), T(0));
            }
            else if (unit <= numTailParts_)
            {
                const int p = unit - 1;
                const int slot = (tailFdlIndex_ - p + numTailParts_) % numTailParts_;
                simd::complexMulAccum(tailAccum_.data(), &tailFdl_[static_cast<size_t>(slot) * spec],
                                      &tailIR_[static_cast<size_t>(p) * spec], tb + 1);
            }
            else if (unit == numTailParts_ + 1)
            {
                tailFft_->inverse(tailAccum_.data(), tailScratch_.data());
            }
            else
            {
                const std::int64_t dst = taskStart_ + tb;
                for (int m = 0; m < 2 * tb; ++m)
                    tailOut_[static_cast<size_t>((dst + m) & outMask_)] += tailScratch_[static_cast<size_t>(m)];
            }
        }
    }

    /**
     * @brief Processes one complete block through the frequency-domain convolution.
     *
     * Steps:
     * 1. Forward FFT of the input block (zero-padded to 2*blockSize), written
     *    straight into the frequency-domain delay line slot.
     * 2. Multiply-accumulate all IR partitions with corresponding delayed inputs.
     * 3. Inverse FFT to get the output block.
     */
    void processPartitionBlock() noexcept
    {
        // Forward FFT of the input buffer [previous block | current block],
        // directly into this block's FDL slot (no intermediate copy).
        fft_->forward(inputBuffer_.data(),
                      fdlBuffer_[static_cast<size_t>(fdlIndex_)].data());

        // Complex multiply-accumulate: sum over all partitions
        std::fill(fftAccum_.begin(), fftAccum_.end(), T(0));

        for (int p = 0; p < numPartitions_; ++p)
        {
            int fdlIdx = (fdlIndex_ - p + numPartitions_) % numPartitions_;
            const auto& irPart = irPartitions_[static_cast<size_t>(p)];
            const auto& fdlPart = fdlBuffer_[static_cast<size_t>(fdlIdx)];

            // Complex multiplication and accumulation (SIMD-accelerated)
            int bins = numBins_ / 2;
            simd::complexMulAccum(fftAccum_.data(), fdlPart.data(), irPart.data(), bins);
        }

        // Inverse FFT
        fft_->inverse(fftAccum_.data(), outputBuffer_.data());

        // Shift input buffer: move current block to previous position
        std::copy_n(inputBuffer_.begin() + blockSize_, static_cast<size_t>(blockSize_),
                   inputBuffer_.begin());
        std::fill(inputBuffer_.begin() + blockSize_, inputBuffer_.end(), T(0));

        // Advance delay line index
        fdlIndex_ = (fdlIndex_ + 1) % numPartitions_;
    }

    static constexpr int kMaxTailBlock = 65536;

    int blockSize_ = 0;
    int fftSize_ = 0;
    int numBins_ = 0;
    int numPartitions_ = 0;
    int inputPos_ = 0;
    int fdlIndex_ = 0;

    std::unique_ptr<FFTReal<T>> fft_;

    // Pre-transformed IR partitions (frequency domain)
    std::vector<std::vector<T>> irPartitions_;

    // Frequency-domain delay line (past input FFTs)
    std::vector<std::vector<T>> fdlBuffer_;

    // Working buffers
    std::vector<T> inputBuffer_;
    std::vector<T> outputBuffer_;
    std::vector<T> overlapBuffer_;
    std::vector<T> fftAccum_;

    // NonUniform tail (empty in Uniform mode)
    int tailBlock_ = 0;
    int numTailParts_ = 0;
    int totalUnits_ = 0;
    std::unique_ptr<FFTReal<T>> tailFft_;
    std::vector<T> tailIR_;       ///< K partition spectra (contiguous)
    std::vector<T> tailFdl_;      ///< K input spectra (frequency-domain delay line)
    std::vector<T> tailAccum_;
    std::vector<T> tailScratch_;
    std::vector<T> tailOut_;      ///< tail overlap-add output ring (clear-on-read)
    std::vector<T> inRing_;       ///< input history for the tail transforms
    int outMask_ = 0;
    int inMask_ = 0;
    std::int64_t absPos_ = 0;
    std::int64_t taskStart_ = 0;
    int cyclePos_ = 0;
    int tailFdlIndex_ = 0;
    int taskPhase_ = 0;
};

} // namespace dspark
