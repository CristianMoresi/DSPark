// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file ConvolutionRate.h
 * @brief Internal mono FIR rate conversion with an optional matched reference.
 *
 * Setup allocates. Processing and reset are allocation-free and stream-owned.
 * Callers validate settings before construction and supply at most 256 frames.
 */
#include "../AudioBuffer.h"
#include "../Convolver.h"
#include "../FIRFilter.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

namespace dspark::detail
{
class ConvolutionRate final
{
    Convolver<double> up_, referenceUp_, down_;
    AudioBuffer<double> buffer_, reference_;
    int factor_, latency_ = 0;
    bool withReference_;

  public:
    struct Settings
    {
        int filterSpan = 16384;
        int partitionFrames = 512;
        double beta = 18.5;
        double cutoff = 0.5;
    };

    [[nodiscard]] static bool valid(int factor, Settings settings) noexcept
    {
        return factor >= 1 && factor <= 16 && (factor & (factor - 1)) == 0 &&
               settings.filterSpan >= 2 && settings.filterSpan <= 65536 &&
               (settings.filterSpan & 1) == 0 && settings.partitionFrames >= 16 &&
               settings.partitionFrames <= 4096 &&
               (settings.partitionFrames & (settings.partitionFrames - 1)) == 0 &&
               std::isfinite(settings.beta) && settings.beta >= 0 && settings.beta <= 30 &&
               std::isfinite(settings.cutoff) && settings.cutoff > 0 && settings.cutoff <= 0.5;
    }

    // Cumulative setup bound, including temporary FIR designs, FFT plans, vector
    // headers and allocation alignment/proxy overhead. Not a live-payload estimate.
    [[nodiscard]] static std::size_t allocationBound(int factor, Settings settings,
                                                     bool withReference) noexcept
    {
        assert(valid(factor, settings));
        const auto buffers = static_cast<std::size_t>(withReference ? 2 : 1);
        const auto storage = buffers * 256 * static_cast<std::size_t>(factor) * sizeof(double);
        if (factor == 1)
            return storage;
        const auto block = static_cast<std::size_t>(settings.partitionFrames * factor);
        const auto length = static_cast<std::size_t>(settings.filterSpan * factor + 1);
        return storage + 6 * (length * sizeof(double) + 64) +
               (buffers + 1) * convolutionAllocationBound(block, length);
    }

    // Shared preflight for a fresh Core Convolver, including temporary FFT plans.
    [[nodiscard]] static std::size_t convolutionAllocationBound(std::size_t block,
                                                                std::size_t length) noexcept
    {
        const auto partitions = (length + block - 1) / block;
        return ((4 * partitions + 17) * block + 4 * partitions + 2) * sizeof(double) +
               2 * partitions * sizeof(std::vector<double>) + sizeof(FFTReal<double>) + 2048 +
               (2 * partitions + 12) * 64;
    }

    ConvolutionRate(int factor, Settings settings, bool withReference)
        : factor_(factor), withReference_(withReference)
    {
        assert(valid(factor, settings));
        buffer_.resize(1, 256 * factor);
        if (withReference)
            reference_.resize(1, 256 * factor);
        if (factor == 1)
            return;
        const int block = settings.partitionFrames * factor;
        const int length = settings.filterSpan * factor + 1;
        auto up = FIRDesign<double>::lowPass(1., settings.cutoff / factor, length, settings.beta);
        auto down = FIRDesign<double>::lowPass(1., settings.cutoff / factor, length, settings.beta);
        for (auto &value : up)
            value *= factor;
        up_.prepare(block, up.data(), length);
        if (withReference)
            referenceUp_.prepare(block, up.data(), length);
        down_.prepare(block, down.data(), length);
        latency_ = (length - 1 + 2 * block) / factor;
    }

    [[nodiscard]] int latency() const noexcept
    {
        return latency_;
    }

    void reset() noexcept
    {
        up_.reset();
        referenceUp_.reset();
        down_.reset();
    }

    double *upsampleReference(const double *input, int count) noexcept
    {
        assert(withReference_ && count >= 0 && count <= 256);
        auto *out = reference_.getChannel(0);
        std::fill_n(out, count * factor_, 0.);
        for (int i = 0; i < count; ++i)
            out[i * factor_] = input[i];
        if (factor_ > 1)
            referenceUp_.processInPlace(out, count * factor_);
        return out;
    }

    AudioBufferView<double> upsample(AudioBufferView<double> input) noexcept
    {
        const int count = input.getNumSamples();
        assert(input.getNumChannels() == 1 && count <= 256);
        auto *out = buffer_.getChannel(0);
        std::fill_n(out, count * factor_, 0.);
        for (int i = 0; i < count; ++i)
            out[i * factor_] = input.getChannel(0)[i];
        if (factor_ > 1)
            up_.processInPlace(out, count * factor_);
        return buffer_.toView().getSubView(0, count * factor_);
    }

    void downsample(AudioBufferView<double> output) noexcept
    {
        const int count = output.getNumSamples();
        assert(output.getNumChannels() == 1 && count <= 256);
        auto *data = buffer_.getChannel(0);
        if (factor_ > 1)
            down_.processInPlace(data, count * factor_);
        for (int i = 0; i < count; ++i)
            output.getChannel(0)[i] = data[i * factor_];
    }
};
} // namespace dspark::detail
