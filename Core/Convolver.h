// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

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
 * Dependencies: FFT.h, AudioBuffer.h.
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
#include "FFT.h"
#include "SimdOps.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <type_traits>
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
    /**
     * @brief Prepares the convolver with an impulse response.
     *
     * This is the only method that allocates memory. Call once during setup.
     *
     * @param blockSize   Processing block size (must be power of two). Determines
     *                    latency and FFT size. Typical: 128, 256, 512.
     * @param irData      Pointer to the impulse response samples.
     * @param irLength    Number of samples in the impulse response.
     */
    void prepare(int blockSize, const T* irData, int irLength)
    {
        assert(blockSize >= 1 && (blockSize & (blockSize - 1)) == 0);
        assert(irLength > 0);
        assert(irData != nullptr);
        if (irData == nullptr || irLength <= 0) return;

        blockSize_ = blockSize;
        fftSize_ = blockSize * 2;
        numBins_ = fftSize_ + 2; // N/2+1 complex bins × 2

        // Create FFT processor
        fft_ = std::make_unique<FFTReal<T>>(fftSize_);

        // Partition the IR into blocks
        numPartitions_ = (irLength + blockSize - 1) / blockSize;

        // Pre-transform each IR partition
        irPartitions_.resize(static_cast<size_t>(numPartitions_));
        std::vector<T> paddedBlock(static_cast<size_t>(fftSize_), T(0));

        for (int p = 0; p < numPartitions_; ++p)
        {
            int offset = p * blockSize;
            int remaining = std::min(blockSize, irLength - offset);

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
        overlapBuffer_.assign(static_cast<size_t>(blockSize), T(0));
        fftInput_.assign(static_cast<size_t>(numBins_), T(0));
        fftAccum_.assign(static_cast<size_t>(numBins_), T(0));

        // Frequency-domain delay line (stores past input FFTs)
        fdlBuffer_.resize(static_cast<size_t>(numPartitions_));
        for (auto& fdl : fdlBuffer_)
            fdl.assign(static_cast<size_t>(numBins_), T(0));

        fdlIndex_ = 0;
        inputPos_ = 0;
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
    }

    /**
     * @brief Processes a block of audio through the convolver.
     *
     * Input and output buffers must have at least `numSamples` elements.
     * The output contains the convolved result.
     *
     * @param input      Input audio samples.
     * @param output     Output audio samples (convolved result).
     * @param numSamples Number of samples to process (must be <= blockSize).
     */
    void process(const T* input, T* output, int numSamples) noexcept
    {
        // numSamples may exceed blockSize_: the sample loop triggers a partition
        // transform at each block boundary, so arbitrary lengths are handled.
        for (int i = 0; i < numSamples; ++i)
        {
            inputBuffer_[static_cast<size_t>(blockSize_ + inputPos_)] = input[i];
            output[i] = overlapBuffer_[static_cast<size_t>(inputPos_)];

            ++inputPos_;
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
        // Simple sample-by-sample overlap-save
        for (int i = 0; i < numSamples; ++i)
        {
            inputBuffer_[static_cast<size_t>(blockSize_ + inputPos_)] = data[i];
            data[i] = overlapBuffer_[static_cast<size_t>(inputPos_)];

            ++inputPos_;
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
     * @brief Prepares with AudioSpec and IR data (unified API).
     *
     * Uses spec.maxBlockSize as the convolution block size.
     *
     * @param spec     Audio environment specification.
     * @param irData   Impulse response samples.
     * @param irLength Number of IR samples.
     *
     * @note M7d1: If `spec.maxBlockSize` is not already a power of two the
     *       effective FFT block rounds up silently (e.g. 300 → 512),
     *       doubling the effective latency. Use `getBlockSize()` after
     *       prepare to inspect the resolved value, or pre-set
     *       `maxBlockSize` to a power of two.
     */
    void prepare(const AudioSpec& spec, const T* irData, int irLength)
    {
        int bs = spec.maxBlockSize;
        // Round up to next power of two
        int fftBlock = 1;
        while (fftBlock < bs) fftBlock <<= 1;
        prepare(fftBlock, irData, irLength);
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

    /** @brief Returns the number of IR partitions. */
    [[nodiscard]] int getNumPartitions() const noexcept { return numPartitions_; }

private:
    /**
     * @brief Processes one complete block through the frequency-domain convolution.
     *
     * Steps:
     * 1. Forward FFT of the input block (zero-padded to 2*blockSize).
     * 2. Store in the frequency-domain delay line.
     * 3. Multiply-accumulate all IR partitions with corresponding delayed inputs.
     * 4. Inverse FFT to get the output block.
     */
    void processPartitionBlock() noexcept
    {
        // Forward FFT of the input buffer [previous block | current block]
        fft_->forward(inputBuffer_.data(), fftInput_.data());

        // Store in frequency-domain delay line
        std::copy(fftInput_.begin(), fftInput_.end(),
                 fdlBuffer_[static_cast<size_t>(fdlIndex_)].begin());

        // Complex multiply-accumulate: sum over all partitions
        std::fill(fftAccum_.begin(), fftAccum_.end(), T(0));

        for (int p = 0; p < numPartitions_; ++p)
        {
            int fdlIdx = (fdlIndex_ - p + numPartitions_) % numPartitions_;
            const auto& irPart = irPartitions_[static_cast<size_t>(p)];
            const auto& fdlPart = fdlBuffer_[static_cast<size_t>(fdlIdx)];

            // Complex multiplication and accumulation (SIMD-accelerated for float)
            int bins = numBins_ / 2;
            complexMulAccum(fdlPart.data(), irPart.data(),
                            fftAccum_.data(), bins);
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

    int blockSize_ = 0;
    /// Complex multiply-accumulate: accum += a * b over interleaved bins.
    /// Thin wrapper over the shared SIMD primitive in SimdOps.h.
    static void complexMulAccum(const T* a, const T* b, T* accum, int bins) noexcept
    {
        simd::complexMulAccum(accum, a, b, bins);
    }

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
    std::vector<T> fftInput_;
    std::vector<T> fftAccum_;
};

} // namespace dspark
