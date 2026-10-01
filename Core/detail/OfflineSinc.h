// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineSinc.h
 * @brief Complete finite-source sinc phases and source-rate projection.
 * Reuses Core convolution and Cauchy moments. Bounded PCM scratch; coarse maps
 * grow with the source length. Readers supply immutable finite samples and may
 * be rewound. Threading: one worker owns each nonmovable instance.
 */
#include "OfflineProduct.h"
#if DSPARK_HAS_OFFLINE
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

// Finite cardinal-sinc phase, using the existing
// Core finite convolution window and the same source's shared Cauchy moments.
namespace dspark::detail::offline_sinc
{
class SincPhase
{
    OfflineProductWorkspace &work_;
    double shift_, sine_ = 0;
    std::optional<OfflineConvolutionWindow> near_;

  public:
    SincPhase(OfflineProductWorkspace &work, double shift) : work_(work), shift_(shift)
    {
        if (!std::isfinite(shift) || std::abs(shift) >= 1)
            offlineFail(OfflineStatus::InvalidInput);
        if (shift == 0)
            return;
        // Preserve a shift one ulp from an integer: multiplying that shift by
        // pi first loses relative accuracy at the nearest sinc numerator zero.
        const double reduced = shift > .5 ? 1 - shift : shift < -.5 ? -1 - shift : shift;
        sine_ = std::sin(std::numbers::pi * reduced);
        near_.emplace(work.job(), work.block(), [this](int lag)
                      { return (lag % 2 ? -sine_ : sine_) / (std::numbers::pi * (lag + shift_)); });
    }
    SincPhase(const SincPhase &) = delete;
    SincPhase &operator=(const SincPhase &) = delete;
    template <class Reader>
    void evaluate(const OfflineHilbertMap &map, std::size_t leaf, Reader &source, double *out)
    {
        const int b = work_.block();
        if (leaf >= work_.leaves())
            offlineFail(OfflineStatus::InvalidInput);
        const auto first = static_cast<std::int64_t>(leaf) * b;
        if (shift_ == 0)
        {
            work_.read(source, first, b, out);
            return;
        }
        map.evaluateCauchyFar(leaf, shift_, out, true);
        work_.read(source, first - b, 3 * b, near_->input());
        const auto *close = near_->process();
        for (int i = 0; i < b; ++i)
            out[i] = close[i] + (i % 2 ? -sine_ : sine_) * out[i];
    }
};

// Project a finite high-rate sequence onto |f|<=1/(2*factor) and sample at
// multiples of factor. h[k]=sin(pi*k/factor)/(pi*k) gives a single real
// Cauchy transform of sin(pi*j/factor)*source[j], plus its diagonal term.
template <class Reader> class SincProjection
{
    Reader source_;
    OfflineProductWorkspace work_;
    int factor_;
    std::array<double, 32> sine_{};
    struct Modulated
    {
        SincProjection *owner;
        void operator()(std::int64_t first, int count, double *out)
        {
            owner->source_(first, count, out);
            for (int i = 0; i < count; ++i)
                out[i] *=
                    owner->sine_[static_cast<std::size_t>((first + i) % (2 * owner->factor_))];
        }
    } modulated_{this};
    std::optional<OfflineHilbertMap> map_;
    std::unique_ptr<double[]> transform_, raw_, output_;

  public:
    SincProjection(OfflineSession &job, std::int64_t frames, int factor, Reader source, int block)
        : source_(std::move(source)),
          work_(job, frames, block, OfflineProductWorkspace::NearFields::External), factor_(factor)
    {
        if (factor < 1 || factor > 16 || (factor & (factor - 1)))
            offlineFail(OfflineStatus::InvalidInput);
        for (int p = 1; p < factor; ++p)
        {
            sine_[p] = std::sin(std::numbers::pi * p / factor);
            sine_[p + factor] = -sine_[p];
        }
        map_.emplace(work_, modulated_);
        transform_ = job.allocate<double>(block);
        raw_ = job.allocate<double>(block);
        output_ = job.allocate<double>(block / factor);
        work_.prepareCauchy();
    }
    SincProjection(const SincProjection &) = delete;
    SincProjection &operator=(const SincProjection &) = delete;
    void rebuild()
    {
        map_->rebuild(work_, modulated_);
    }
    [[nodiscard]] std::size_t leaves() const
    {
        return work_.leaves();
    }
    [[nodiscard]] std::span<const double> get(std::size_t leaf)
    {
        if (leaf >= leaves())
            offlineFail(OfflineStatus::InvalidInput);
        const int block = work_.block();
        const auto first = static_cast<std::int64_t>(leaf) * block;
        const int count = static_cast<int>(std::min<std::int64_t>(block, work_.frames() - first));
        map_->evaluateCauchy(work_, leaf, modulated_, transform_.get());
        work_.read(source_, first, count, raw_.get());
        const int written = (count + factor_ - 1) / factor_;
        for (int k = 0; k < written; ++k)
        {
            const auto frame = first / factor_ + k;
            output_[k] =
                raw_[k * factor_] / factor_ + (frame % 2 ? 1 : -1) * transform_[k * factor_];
        }
        return {output_.get(), static_cast<std::size_t>(written)};
    }
};
template <class Reader, class Consumer>
void projectSinc(OfflineSession &job, std::int64_t frames, int factor, Reader source,
                 Consumer consume, int block = 4096)
{
    SincProjection<Reader> projection(job, frames, factor, std::move(source), block);
    for (std::size_t leaf = 0; leaf < projection.leaves(); ++leaf)
    {
        const auto data = projection.get(leaf);
        consume(static_cast<std::int64_t>(leaf) * (block / factor), static_cast<int>(data.size()),
                data.data());
    }
}
} // namespace dspark::detail::offline_sinc

#endif // DSPARK_HAS_OFFLINE
