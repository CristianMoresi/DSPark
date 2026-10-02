// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file StereoColor.h
 * @brief Internal rational color branch for stereo generation.
 *
 * Uses shared continuous clipping, convolution, FIR compensation and DC kernels.
 * The conversion FIR has an explicit 0.45 Fs to 0.50 Fs transition. Setup
 * allocates; processing/reset are allocation-free and owned by the audio stream.
 */
#include "../FIRFilter.h"
#include "ContinuousClip.h"
#include "ContinuousStereoColor.h"
#include "ConvolutionRate.h"
#include "RetimedDcBlock.h"
#include <array>
#include <memory>

namespace dspark::detail
{
template <ClipperCurve C> class OversampledStereoColor final
{
    static_assert(C == ClipperCurve::AsymmetricKnee || C == ClipperCurve::SymmetricKnee);
    ConvolutionRate rate_;
    continuous_clip::Residual<C> residual_;
    FIRFilter<double> eq_;
    RetimedDcBlock dc_;
    std::array<double, 64> dry_{}, pad_{};
    int delay_, padding_, latency_, dryWrite_ = 0, padWrite_ = 0;

  public:
    [[nodiscard]] static int latencyFor(int factor, int sourceFactor) noexcept
    {
        const int delay =
            7 + (factor <= 2 ? 18 : 10) + RetimedDcBlock::filterLength(factor, sourceFactor) / 2;
        return 256 + 2 * 64 + (delay + factor - 1) / factor;
    }
    [[nodiscard]] static std::size_t allocationBound(int factor, int sourceFactor) noexcept
    {
        const std::size_t order = factor <= 2 ? 18 : 10, taps = 2 * order + 1;
        const auto construction = (6 * (order + 1) + (order + 2) * taps) * sizeof(double);
        const auto filter = taps * (sizeof(std::atomic<double>) + 4 * sizeof(double)) + sizeof(int);
        return ConvolutionRate::allocationBound(factor, {256, 64, 18.5, .475}, false) +
               RetimedDcBlock::allocationBound(factor, sourceFactor) + construction + filter +
               (order + 13) * 64;
    }
    OversampledStereoColor(int factor, int sourceFactor)
        : rate_(factor, {256, 64, 18.5, .475}, false), dc_(factor, sourceFactor),
          delay_(7 + (factor <= 2 ? 18 : 10)),
          padding_((factor - (delay_ + dc_.latency()) % factor) % factor),
          latency_(rate_.latency() + (delay_ + dc_.latency() + padding_) / factor)
    {
        assert(factor >= 2);
        const auto coefficients = continuous_clip::inverseBoxPower(factor <= 2 ? 18 : 10);
        eq_.prepare(static_cast<int>(coefficients.size()), 1);
        eq_.setCoefficients(coefficients);
        residual_.reset(.08);
    }

    [[nodiscard]] int latency() const noexcept
    {
        return latency_;
    }

    void reset() noexcept
    {
        rate_.reset();
        residual_.reset(.08);
        eq_.reset();
        dc_.reset();
        dry_.fill(0);
        pad_.fill(0);
        dryWrite_ = padWrite_ = 0;
    }

    void process(AudioBufferView<double> input) noexcept
    {
        auto high = rate_.upsample(input);
        auto *values = high.getChannel(0);
        for (int i = 0; i < high.getNumSamples(); ++i)
        {
            const double sample = values[i];
            double result = eq_.processSample(residual_.process(sample), 0);
            result += dry_[dryWrite_];
            dry_[dryWrite_] = sample;
            dryWrite_ = (dryWrite_ + 1) % delay_;
            result = dc_.process(result);
            if (padding_)
            {
                const double old = pad_[padWrite_];
                pad_[padWrite_] = result;
                padWrite_ = (padWrite_ + 1) % padding_;
                result = old;
            }
            values[i] = result;
        }
        rate_.downsample(input);
    }
};
// Select the complete source-rate algorithm at 1x; retain the established
// explicit 2x/4x/8x/16x paths, including their latency and state arithmetic.
template <ClipperCurve C> class StereoColor final
{
    std::unique_ptr<ContinuousStereoColor<C>> continuous_;
    std::unique_ptr<OversampledStereoColor<C>> oversampled_;

  public:
    [[nodiscard]] static int latencyFor(int factor, int sourceFactor) noexcept
    {
        return factor == 1 ? ContinuousStereoColor<C>::latency()
                           : OversampledStereoColor<C>::latencyFor(factor, sourceFactor);
    }
    [[nodiscard]] static std::size_t allocationBound(int factor, int sourceFactor) noexcept
    {
        return factor == 1 ? sizeof(ContinuousStereoColor<C>) + ContinuousStereoColor<C>::allocationBound()
                           : sizeof(OversampledStereoColor<C>) +
                                 OversampledStereoColor<C>::allocationBound(factor, sourceFactor);
    }
    StereoColor(int factor, int sourceFactor)
    {
        if (factor == 1)
            continuous_ = std::make_unique<ContinuousStereoColor<C>>(sourceFactor);
        else
            oversampled_ = std::make_unique<OversampledStereoColor<C>>(factor, sourceFactor);
    }
    [[nodiscard]] int latency() const noexcept
    {
        return continuous_ ? continuous_->latency() : oversampled_->latency();
    }
    void reset() noexcept
    {
        if (continuous_) continuous_->reset();
        else oversampled_->reset();
    }
    void process(AudioBufferView<double> input) noexcept
    {
        if (continuous_) continuous_->process(input);
        else oversampled_->process(input);
    }
};
} // namespace dspark::detail
