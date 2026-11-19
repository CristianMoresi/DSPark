// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file FirBank.h
 * @brief Many long FIR filters over shared inputs, run by one partitioned
 *        convolution in the frequency domain.
 *
 * A processor that needs several long kernels at once (the bands of a
 * linear-phase dynamic equalizer, the branches of a linear-phase crossover)
 * would pay, with one convolver per kernel, one forward transform of the
 * same input per kernel and per hop. Here every input line is transformed
 * once per hop into its own delay line of spectra, and each kernel only
 * multiplies and accumulates over that line and transforms back: the cost of
 * a kernel is its products and one inverse transform, whatever the number of
 * kernels that hear the same line.
 *
 * A kernel hears any set of the input lines (a mask), and gives one output
 * per line it hears. Kernels are replaced while the audio runs: the new
 * kernel is transformed at once, on the calling thread, and comes in at the
 * next hop through a crossfade of one partition against the old one (or
 * against silence, for a kernel that was not there; removeKernel() fades one
 * out the same way).
 *
 * The convolution is uniformly partitioned overlap-save: partitions of P
 * samples, transforms of 2 P, and a latency of P samples from push() to
 * tap(), on top of whatever delay the kernels themselves carry. The output
 * does not depend on how the caller cuts the stream.
 *
 * Use:
 * @code
 *   dspark::FirBank<float> bank;
 *   bank.prepare(4096, 512, 2, 8);            // taps, partition, lines, kernels
 *   bank.setKernel(0, taps, 0b11);            // heard by lines 0 and 1
 *   // per block, in runs that do not cross a hop:
 *   while (left > 0)
 *   {
 *       const int n = std::min(left, bank.room());
 *       const float* wetL = bank.tap(0, 0);   // n samples, P late
 *       const float* wetR = bank.tap(0, 1);
 *       // ... use them ...
 *       bank.push(inputs, n);                 // inputs[line] + n samples each
 *       left -= n;
 *   }
 * @endcode
 *
 * Threading: owner-managed. Every method belongs to the thread that calls
 * push() (or to a single thread, offline), except prepare(), which
 * allocates and belongs to setup. setKernel() does not allocate, and costs
 * one transform per partition of the kernel.
 *
 * Dependencies: DspMath.h, FFT.h, SimdOps.h.
 */

#include "DspMath.h"
#include "FFT.h"
#include "SimdOps.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class FirBank
 * @brief Long FIR kernels over shared input lines, by partitioned convolution.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class FirBank
{
public:
    static constexpr int kMaxInputs = 32;   ///< Input lines a mask can name.

    /**
     * @brief Allocates the bank (setup thread).
     *
     * @param kernelLength Taps of every kernel; rounded up to a whole number of partitions.
     * @param partition    Samples per partition, rounded up to a power of two (at least 16): the latency.
     * @param numInputs    Input lines (1 to kMaxInputs).
     * @param numKernels   Kernels the bank can hold.
     */
    void prepare(int kernelLength, int partition, int numInputs, int numKernels)
    {
        fft_.reset();       // gates everything else: set last
        int p = 16;
        while (p < partition && p < (1 << 20)) p <<= 1;
        partition_ = p;
        parts_ = std::clamp((std::max(kernelLength, 1) + p - 1) / p, 1, 1 << 16);
        inputs_ = std::clamp(numInputs, 1, kMaxInputs);
        kernels_ = std::max(numKernels, 0);
        specLen_ = static_cast<size_t>(2 * p + 2);

        const size_t lineLen = static_cast<size_t>(parts_) * specLen_;
        in_.assign(static_cast<size_t>(inputs_) * static_cast<size_t>(2 * p), T(0));
        fdl_.assign(static_cast<size_t>(inputs_) * lineLen, T(0));
        spectra_.assign(static_cast<size_t>(kernels_) * 2u * lineLen, T(0));
        out_.assign(static_cast<size_t>(kernels_) * static_cast<size_t>(inputs_) * static_cast<size_t>(p), T(0));
        state_.assign(static_cast<size_t>(kernels_), Kernel {});
        accum_.assign(specLen_, T(0));
        block_.assign(static_cast<size_t>(2 * p), T(0));
        timeOld_.assign(static_cast<size_t>(2 * p), T(0));
        timeNew_.assign(static_cast<size_t>(2 * p), T(0));
        pos_ = 0;
        fdlPos_ = 0;
        fft_ = std::make_unique<FFTReal<T>>(static_cast<size_t>(2 * p));
    }

    /** @brief Whether prepare() has run. */
    [[nodiscard]] bool isPrepared() const noexcept { return fft_ != nullptr; }

    /** @brief Samples between push() and tap(): one partition. */
    [[nodiscard]] int getLatency() const noexcept { return partition_; }

    /** @brief Taps of a kernel (a whole number of partitions). */
    [[nodiscard]] int getKernelLength() const noexcept { return parts_ * partition_; }

    /** @brief Samples per partition. */
    [[nodiscard]] int getPartition() const noexcept { return partition_; }

    /** @brief Input lines. */
    [[nodiscard]] int getNumInputs() const noexcept { return inputs_; }

    /** @brief Kernels the bank can hold. */
    [[nodiscard]] int getNumKernels() const noexcept { return kernels_; }

    /**
     * @brief Gives a kernel its taps and the lines that hear it.
     *
     * The kernel comes in at the next hop, crossfaded over one partition
     * against the one it replaces (or against silence). Setting a kernel
     * again before that hop replaces the one that was coming.
     *
     * @param kernel Kernel index; out of range is ignored.
     * @param taps   getKernelLength() taps.
     * @param inputs Mask of the input lines that hear it (bit i: line i).
     */
    void setKernel(int kernel, const T* taps, unsigned inputs) noexcept
    {
        if (!fft_ || kernel < 0 || kernel >= kernels_ || taps == nullptr) return;
        Kernel& k = state_[static_cast<size_t>(kernel)];
        const int set = 1 - k.set;                      // the one not in use
        T* spec = spectraOf(kernel, set);
        const auto p = static_cast<size_t>(partition_);
        for (int part = 0; part < parts_; ++part)
        {
            std::copy_n(taps + static_cast<size_t>(part) * p, p, block_.data());
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(p), block_.end(), T(0));
            fft_->forward(block_.data(), spec + static_cast<size_t>(part) * specLen_);
        }
        k.nextMask = inputs & maskOfAll();
        k.pending = true;
    }

    /** @brief Fades a kernel out over the next partition. */
    void removeKernel(int kernel) noexcept
    {
        if (!fft_ || kernel < 0 || kernel >= kernels_) return;
        Kernel& k = state_[static_cast<size_t>(kernel)];
        if (k.mask == 0u && !k.pending) return;
        k.nextMask = 0u;
        k.pending = true;
    }

    /** @brief Whether the outputs of a kernel are silence now: it is not playing and nothing of it is left to hear. */
    [[nodiscard]] bool isKernelSilent(int kernel) const noexcept
    {
        return kernel < 0 || kernel >= kernels_ || state_[static_cast<size_t>(kernel)].silent;
    }

    /** @brief Whether a kernel is playing, or about to. */
    [[nodiscard]] bool isKernelActive(int kernel) const noexcept
    {
        if (kernel < 0 || kernel >= kernels_) return false;
        const Kernel& k = state_[static_cast<size_t>(kernel)];
        return k.pending ? k.nextMask != 0u : k.mask != 0u;
    }

    /** @brief Clears the memory of the signal. The kernels stay; one that was coming lands at once. */
    void reset() noexcept
    {
        std::fill(in_.begin(), in_.end(), T(0));
        std::fill(fdl_.begin(), fdl_.end(), T(0));
        std::fill(out_.begin(), out_.end(), T(0));
        for (auto& k : state_)
            if (k.pending)
            {
                k.set = 1 - k.set;
                k.mask = k.nextMask;
                k.pending = false;
            }
        pos_ = 0;
        fdlPos_ = 0;
    }

    /** @brief Samples that can be pushed before the next hop (1 to the partition). */
    [[nodiscard]] int room() const noexcept { return partition_ - pos_; }

    /**
     * @brief What a kernel gives for one of its lines: room() samples, one
     *        partition late, valid until the next push().
     *
     * Silence for a line the kernel does not hear.
     */
    [[nodiscard]] const T* tap(int kernel, int input) const noexcept
    {
        return out_.data() + (static_cast<size_t>(kernel) * static_cast<size_t>(inputs_) + static_cast<size_t>(input))
                             * static_cast<size_t>(partition_) + static_cast<size_t>(pos_);
    }

    /**
     * @brief Takes n samples of every input line; runs the hop when the partition fills.
     *
     * @param inputs One pointer per line (a null one is silence).
     * @param n      Samples, at most room().
     */
    void push(const T* const* inputs, int n) noexcept
    {
        if (!fft_ || n <= 0) return;
        n = std::min(n, room());
        const auto p = static_cast<size_t>(partition_);
        for (int i = 0; i < inputs_; ++i)
        {
            T* dest = in_.data() + static_cast<size_t>(i) * 2u * p + p + static_cast<size_t>(pos_);
            if (inputs[i] != nullptr) std::copy_n(inputs[i], static_cast<size_t>(n), dest);
            else std::fill_n(dest, static_cast<size_t>(n), T(0));
        }
        pos_ += n;
        if (pos_ == partition_)
        {
            hop();
            pos_ = 0;
        }
    }

private:
    struct Kernel
    {
        unsigned mask = 0u;         ///< Lines that hear the kernel in use.
        unsigned nextMask = 0u;     ///< Lines of the one that is coming.
        int set = 0;                ///< Which of its two sets of spectra is in use.
        bool pending = false;       ///< A kernel (or its removal) comes in at the next hop.
        bool silent = true;         ///< Its output blocks are already zero.
    };

    [[nodiscard]] unsigned maskOfAll() const noexcept
    {
        return inputs_ >= 32 ? ~0u : ((1u << inputs_) - 1u);
    }

    [[nodiscard]] T* spectraOf(int kernel, int set) noexcept
    {
        return spectra_.data() + (static_cast<size_t>(kernel) * 2u + static_cast<size_t>(set))
                                 * static_cast<size_t>(parts_) * specLen_;
    }

    /** @brief One kernel set over one input line: the last partition of the result into `time` + P. */
    void convolve(const T* spec, int input, T* time) noexcept
    {
        std::fill(accum_.begin(), accum_.end(), T(0));
        const T* line = fdl_.data() + static_cast<size_t>(input) * static_cast<size_t>(parts_) * specLen_;
        for (int part = 0; part < parts_; ++part)
        {
            const int at = (fdlPos_ - part + parts_) % parts_;
            simd::complexMulAccum(accum_.data(), line + static_cast<size_t>(at) * specLen_,
                                  spec + static_cast<size_t>(part) * specLen_, partition_ + 1);
        }
        fft_->inverse(accum_.data(), time);
    }

    /** @brief A full partition has come in: every line into its delay line, every kernel out of it. */
    void hop() noexcept
    {
        const auto p = static_cast<size_t>(partition_);
        fdlPos_ = (fdlPos_ + 1) % parts_;
        for (int i = 0; i < inputs_; ++i)
        {
            T* in = in_.data() + static_cast<size_t>(i) * 2u * p;
            fft_->forward(in, fdl_.data() + (static_cast<size_t>(i) * static_cast<size_t>(parts_)
                                             + static_cast<size_t>(fdlPos_)) * specLen_);
            std::copy_n(in + p, p, in);
        }

        const T step = T(1) / static_cast<T>(partition_);
        for (int kernel = 0; kernel < kernels_; ++kernel)
        {
            Kernel& k = state_[static_cast<size_t>(kernel)];
            const unsigned heard = k.mask | (k.pending ? k.nextMask : 0u);
            T* outs = out_.data() + static_cast<size_t>(kernel) * static_cast<size_t>(inputs_) * p;
            if (heard == 0u)
            {
                if (!k.silent) std::fill_n(outs, static_cast<size_t>(inputs_) * p, T(0));
                k.silent = true;
                if (k.pending) { k.set = 1 - k.set; k.pending = false; }
                continue;
            }
            k.silent = false;
            const T* specOld = spectraOf(kernel, k.set);
            const T* specNew = spectraOf(kernel, 1 - k.set);
            for (int i = 0; i < inputs_; ++i)
            {
                T* out = outs + static_cast<size_t>(i) * p;
                const unsigned bit = 1u << i;
                const bool old = (k.mask & bit) != 0u, coming = k.pending && (k.nextMask & bit) != 0u;
                if (!k.pending)
                {
                    if (!old) { std::fill_n(out, p, T(0)); continue; }
                    convolve(specOld, i, timeOld_.data());
                    std::copy_n(timeOld_.data() + p, p, out);
                    continue;
                }
                // A kernel comes in (or goes): one partition of crossfade.
                if (old) convolve(specOld, i, timeOld_.data());
                if (coming) convolve(specNew, i, timeNew_.data());
                for (size_t j = 0; j < p; ++j)
                {
                    const T was = old ? timeOld_[p + j] : T(0);
                    const T is = coming ? timeNew_[p + j] : T(0);
                    out[j] = was + (is - was) * (static_cast<T>(j + 1) * step);
                }
            }
            if (k.pending)
            {
                k.set = 1 - k.set;
                k.mask = k.nextMask;
                k.pending = false;
            }
        }
    }

    int partition_ = 0, parts_ = 0, inputs_ = 0, kernels_ = 0;
    size_t specLen_ = 0;
    int pos_ = 0;       ///< Samples of the partition in hand.
    int fdlPos_ = 0;    ///< Newest block of the delay lines.

    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> in_;         ///< Per line: the last partition, then the one coming in.
    std::vector<T> fdl_;        ///< Per line: the spectra of the last blocks.
    std::vector<T> spectra_;    ///< Per kernel: two sets of partition spectra.
    std::vector<T> out_;        ///< Per kernel and line: the block being read.
    std::vector<Kernel> state_;
    std::vector<T> accum_, block_, timeOld_, timeNew_;
};

} // namespace dspark
