// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineProduct.h
 * @brief Internal finite-source bandlimited multiplication for offline workers.
 *
 * Threading: one worker owns the workspace, readers and consumer. No global cache.
 * Readers supply immutable, finite, normalized samples; they are called only in
 * [0,frames), in blocks of at most 4096 frames. They may be rewound. The consumer
 * receives ordered provisional blocks, not a publication/transaction boundary.
 *
 * Both finite sequences are sinc-interpolated with zero extension, multiplied,
 * projected onto the source-rate baseband, then sampled at the original clock.
 * This primitive does not choose a gain-control boundary policy, preserve exact
 * exclusions or calibrate a peak target; the owning renderer supplies those.
 */

#include "../OfflineProcessing.h"
#if DSPARK_HAS_OFFLINE
#include "OfflineConvolutionWindow.h"
#include "../Hilbert.h"
#include "../SimdOps.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <vector>

namespace dspark::detail
{

class OfflineProductWorkspace final
{
  public:
    static constexpr int order = 24;
    using Coefficients = std::array<double, 2 * order>;
    using Matrix = std::array<std::array<double, order>, order>;
    struct Level
    {
        std::size_t count = 0, offset = 0;
    };
    enum class NearFields { Hilbert, External };

    OfflineProductWorkspace(OfflineSession &job, std::int64_t frames, int block = 4096,
                            NearFields nearFields = NearFields::Hilbert)
        : job_(job), frames_(frames), block_(block)
    {
        // A wider internal grid reduces coarse-map memory for upsampled offline
        // sources. The external read contract remains capped at 4096 frames.
        if (frames < 1 || block < 128 || block > 65536 || (block & (block - 1)) != 0)
            offlineFail(OfflineStatus::InvalidInput);
        job.checkpoint(OfflinePhase::Plan, 0, frames);
        const auto leaves = static_cast<std::uint64_t>(frames / block + (frames % block != 0));
        // Check the full tree before narrowing or making a source-sized request.
        (void)offlineBytes(2 * leaves + 64, sizeof(Coefficients));
        leaves_ = static_cast<std::size_t>(leaves);
        std::size_t nodes = 0;
        for (auto count = leaves_;; count = count / 2 + count % 2)
        {
            levels_[levelCount_++] = {count, nodes};
            nodes += count;
            if (count == 1)
                break;
        }
        moments_ = job.allocateScratch<Coefficients>(nodes);
        local_ = job.allocateScratch<Coefficients>(leaves_);
        matrices_ = job.allocateScratch<Matrix>(8);
        weights_ = job.allocateScratch<double>(static_cast<std::uint64_t>(block) * order);
        impulse_ = job.allocateScratch<double>(4 * static_cast<std::uint64_t>(block) - 1);
        input_ = job.allocateScratch<double>(block);
        buildGeometry();
        // Existing product workers reserve both fields before opening a sink.
        // A caller supplying other near-field kernels can explicitly omit them.
        if (nearFields == NearFields::Hilbert)
        {
            prepareNear(near_);
            prepareNear(outerNear_);
        }
    }

    [[nodiscard]] int block() const noexcept { return block_; }
    [[nodiscard]] std::int64_t frames() const noexcept { return frames_; }
    [[nodiscard]] std::size_t leaves() const noexcept { return leaves_; }
    [[nodiscard]] OfflineSession &job() noexcept { return job_; }

    void prepareCauchy()
    {
        if (cauchyNear_)
            return;
        for (int i = 0; i < 4 * block_ - 1; ++i)
        {
            const int lag = i - (2 * block_ - 1);
            impulse_[i] = lag == 0 ? 0 : 1 / (std::numbers::pi * lag);
        }
        prepareNear(cauchyNear_);
    }

    template <class Reader>
    void read(Reader &source, std::int64_t first, int count, double *out) const
    {
        std::fill_n(out, count, 0.0);
        if (first < 0)
        {
            const int skip = static_cast<int>(std::min<std::int64_t>(-first, count));
            out += skip;
            count -= skip;
            first += skip;
        }
        if (first >= frames_ || count == 0)
            return;
        count = static_cast<int>(std::min<std::int64_t>(count, frames_ - first));
        while (count > 0)
        {
            const int length = std::min({count, block_, 4096});
            source(first, length, out);
            for (int i = 0; i < length; ++i)
                if (!std::isfinite(out[i]))
                    offlineFail(OfflineStatus::NonFiniteInput);
            first += length;
            out += length;
            count -= length;
        }
    }

  private:
    friend class OfflineHilbertMap;
    using NearField = std::optional<OfflineConvolutionWindow>;

    void buildGeometry()
    {
        for (int i = 0; i < block_; ++i)
        {
            const double u = (i - (block_ - 1) * .5) / (block_ * .5);
            double power = 1;
            for (int p = 0; p < order; ++p)
            {
                weights_[static_cast<std::size_t>(p) * block_ + i] = power;
                power *= u;
            }
        }
        for (int p = 0; p < order; ++p)
        {
            double choose = 1;
            for (int q = 0; q <= p; ++q)
            {
                const double right = std::ldexp(choose, -p);
                const double left = (p - q) % 2 == 0 ? right : -right;
                matrices_[0][p][q] = matrices_[2][q][p] = left;
                matrices_[1][p][q] = matrices_[3][q][p] = right;
                choose *= static_cast<double>(p - q) / (q + 1);
            }
        }
        constexpr std::array<int, 4> offsets{-3, -2, 2, 3};
        for (int k = 0; k < 4; ++k)
        {
            const double d = offsets[k], r = .5 / d;
            for (int p = 0; p < order; ++p)
            {
                double choose = 1;
                for (int q = 0; q < order; ++q)
                {
                    if (q > 0)
                        choose *= static_cast<double>(p + q) / q;
                    matrices_[4 + k][p][q] = (p % 2 == 0 ? 1. : -1.) * choose *
                                             std::pow(r, p + q) / (std::numbers::pi * d);
                }
            }
        }
        for (int i = 0; i < 4 * block_ - 1; ++i)
            impulse_[i] = .5 * hilbertIdealImpulse(i - (2 * block_ - 1));
    }

    void prepareNear(NearField &field)
    {
        field.emplace(job_, block_, [this](int lag) { return impulse_[lag + 2 * block_ - 1]; });
    }

    template <class Reader>
    const double *nearField(std::size_t leaf, Reader &source, bool outer, bool cauchy = false,
                       bool alternating = false)
    {
        if (cauchy)
            prepareCauchy();
        auto &field = cauchy ? cauchyNear_ : outer ? outerNear_ : near_;
        if (!field)
        {
            // Fractional-phase users supply their own near-field kernel and
            // need no Hilbert convolvers. Construct only the requested field.
            for (int i = 0; i < 4 * block_ - 1; ++i)
                impulse_[i] = .5 * hilbertIdealImpulse(i - (2 * block_ - 1));
            prepareNear(field);
        }
        auto *input = field->input();
        read(source, (static_cast<std::int64_t>(leaf) - 1) * block_, 3 * block_, input);
        if (alternating)
            for (int i = 1; i < 3 * block_; i += 2)
                input[i] = -input[i];
        return field->process();
    }

    OfflineSession &job_;
    std::int64_t frames_;
    int block_;
    std::size_t leaves_ = 0, levelCount_ = 0;
    std::array<Level, 64> levels_{};
    OfflineScratchArray<Coefficients> moments_, local_;
    OfflineScratchArray<Matrix> matrices_;
    OfflineScratchArray<double> weights_, impulse_, input_;
    NearField near_, outerNear_, cauchyNear_;
};

// For h[k]=1/(pi*k) on odd k and zero on even k, evaluate h*x without
// truncating its long tail. Exact neighbouring-block Core convolution and
// dyadic multipole/local expansions cover disjoint source regions. The two
// source parities have distinct moments; only opposite parity contributes.
class OfflineHilbertMap final
{
  public:
    template <class Reader>
    OfflineHilbertMap(OfflineProductWorkspace &work, Reader &source)
        : local_(work.job_.allocateScratch<OfflineProductWorkspace::Coefficients>(work.leaves_)),
          leaves_(work.leaves_), block_(work.block_)
    {
        rebuild(work, source);
    }

    // Reuse all geometry and storage when a worker updates its source function.
    // Existing maps sharing this workspace keep their own independent fields.
    template <class Reader>
    void rebuild(OfflineProductWorkspace &work, Reader &source)
    {
        if (work.leaves_ != leaves_ || work.block_ != block_)
            offlineFail(OfflineStatus::InvalidInput);
        constexpr int order = OfflineProductWorkspace::order;
        using Coefficients = OfflineProductWorkspace::Coefficients;
        const auto last = work.levels_[work.levelCount_ - 1];
        std::fill_n(work.moments_.get(), last.offset + last.count, Coefficients{});
        work.job_.checkpoint(OfflinePhase::Plan, 0, work.frames_);
        for (std::size_t leaf = 0; leaf < leaves_; ++leaf)
        {
            const auto first = static_cast<std::int64_t>(leaf) * block_;
            work.read(source, first, block_, work.input_.get());
            auto &dst = work.moments_[leaf];
            for (int i = 0; i < block_; ++i)
                for (int p = 0; p < order; ++p)
                    dst[static_cast<std::size_t>((i % 2) * order + p)] +=
                        work.input_[i] * work.weights_[static_cast<std::size_t>(p) * block_ + i];
            work.job_.checkpoint(OfflinePhase::Plan,
                                 first + std::min<std::int64_t>(block_, work.frames_ - first),
                                 work.frames_);
        }
        for (std::size_t level = 1; level < work.levelCount_; ++level)
        {
            const auto current = work.levels_[level], previous = work.levels_[level - 1];
            for (std::size_t parent = 0; parent < current.count; ++parent)
            {
                auto &dst = work.moments_[current.offset + parent];
                for (int side = 0; side < 2; ++side)
                {
                    const auto child = 2 * parent + side;
                    if (child >= previous.count)
                        continue;
                    const auto &src = work.moments_[previous.offset + child];
                    add(dst, src, work.matrices_[side], 1);
                }
                if ((parent & 255) == 0)
                    work.job_.checkpoint(OfflinePhase::Plan, work.frames_, work.frames_);
            }
        }
        auto *local = local_.get(), *children = work.local_.get();
        *local = Coefficients{};
        for (std::size_t remaining = work.levelCount_; remaining > 0; --remaining)
        {
            const auto level = remaining - 1;
            const auto current = work.levels_[level];
            const double width = std::ldexp(static_cast<double>(block_), static_cast<int>(level));
            for (std::size_t target = 0; target < current.count; ++target)
            {
                const auto first = static_cast<std::int64_t>(2 * (target / 2)) - 2;
                for (auto other = first; other < first + 6; ++other)
                {
                    if (other < 0 || static_cast<std::uint64_t>(other) >= current.count)
                        continue;
                    const auto offset = static_cast<std::int64_t>(target) - other;
                    if (std::abs(offset) <= 1)
                        continue;
                    const int kernel = offset == -3 ? 4 : offset == -2 ? 5 : offset == 2 ? 6 : 7;
                    add(local[target], work.moments_[current.offset + static_cast<std::size_t>(other)],
                        work.matrices_[kernel], 1 / width);
                }
                if ((target & 255) == 0)
                    work.job_.checkpoint(OfflinePhase::Plan, work.frames_, work.frames_);
            }
            if (level == 0)
                break;
            for (std::size_t child = 0; child < work.levels_[level - 1].count; ++child)
            {
                children[child] = Coefficients{};
                add(children[child], local[child / 2], work.matrices_[2 + child % 2], 1);
            }
            std::swap(local, children);
        }
        if (local != local_.get())
            local_.swap(work.local_);
    }

    template <class Reader>
    void evaluate(OfflineProductWorkspace &work, std::size_t leaf, Reader &source, double *output,
                  bool outer = false) const
    {
        evaluateKernel(work, leaf, source, output, outer, false, false);
    }

    // The same parity moments also represent 1/(pi*k) for every nonzero k.
    // Alternating input needs only a sign on the odd source moments; the source
    // reader remains unchanged. This reuses the audio map for endpoint sums.
    template <class Reader>
    void evaluateCauchy(OfflineProductWorkspace &work, std::size_t leaf, Reader &source,
                        double *output, bool alternating = false) const
    {
        evaluateKernel(work, leaf, source, output, false, true, alternating);
    }

    // Evaluate only the non-neighbouring source contribution at n+shift.
    // The caller supplies the matching three-leaf near field. Fractional
    // sinc phases can therefore reuse this tree without truncating its tail.
    void evaluateCauchyFar(std::size_t leaf, double shift, double *output,
                           bool alternating = false) const
    {
        if (leaf >= leaves_ || !std::isfinite(shift) || std::abs(shift) > 1)
            offlineFail(OfflineStatus::InvalidInput);
        constexpr int order = OfflineProductWorkspace::order;
        const auto *coefficients = local_[leaf].data();
        for (int i = 0; i < block_; ++i)
        {
            const double u = (i + shift - (block_ - 1) * .5) / (block_ * .5);
            double even = coefficients[order - 1], odd = coefficients[2 * order - 1];
            for (int p = order - 2; p >= 0; --p)
            {
                even = even * u + coefficients[p];
                odd = odd * u + coefficients[order + p];
            }
            output[i] = alternating ? even - odd : even + odd;
        }
    }

  private:
    template <class Reader>
    void evaluateKernel(OfflineProductWorkspace &work, std::size_t leaf, Reader &source,
                         double *output, bool outer, bool cauchy, bool alternating) const
    {
        if (leaf >= leaves_ || work.leaves_ != leaves_ || work.block_ != block_)
            offlineFail(OfflineStatus::InvalidInput);
        constexpr int order = OfflineProductWorkspace::order;
        const auto *coefficients = local_[leaf].data();
        const auto *close = work.nearField(leaf, source, outer, cauchy, alternating);
        for (int i = 0; i < block_; ++i)
        {
            const auto *c = coefficients + (1 - i % 2) * order;
            const double u = (i - (block_ - 1) * .5) / (block_ * .5);
            double distant = c[order - 1];
            for (int p = order - 2; p >= 0; --p)
                distant = distant * u + c[p];
            if (cauchy)
            {
                const auto *same = coefficients + (i % 2) * order;
                double other = same[order - 1];
                for (int p = order - 2; p >= 0; --p)
                    other = other * u + same[p];
                distant = alternating ? (i % 2 ? distant - other : other - distant) : distant + other;
            }
            output[i] = close[i] + distant;
        }
    }

    static void add(OfflineProductWorkspace::Coefficients &dst,
                    const OfflineProductWorkspace::Coefficients &src,
                    const OfflineProductWorkspace::Matrix &matrix, double scale) noexcept
    {
        constexpr int order = OfflineProductWorkspace::order;
        for (int parity = 0; parity < 2; ++parity)
            for (int p = 0; p < order; ++p)
                dst[static_cast<std::size_t>(parity * order + p)] +=
                    simd::dotProduct(matrix[p].data(), src.data() + parity * order, order) * scale;
    }
    OfflineScratchArray<OfflineProductWorkspace::Coefficients> local_;
    std::size_t leaves_;
    int block_;
};

template <class LeftReader, class RightReader>
class OfflineProductCombination final
{
  public:
    OfflineProductCombination(OfflineProductWorkspace &work, LeftReader &left, RightReader &right,
                              const OfflineHilbertMap &aLeft, const OfflineHilbertMap &aRight)
        : work_(work), left_(left), right_(right), aLeft_(aLeft), aRight_(aRight),
          samples_(work.job().allocateScratch<double>(15 * static_cast<std::uint64_t>(work.block())))
    {
        leaves_.fill(-1);
    }

    void reset() noexcept { leaves_.fill(-1); }

    // Three neighbouring leaves are the full near-field stencil. Entries also
    // retain operands and their transforms for the final product identity.
    [[nodiscard]] double *get(std::size_t leaf)
    {
        const int block = work_.block();
        const auto slot = leaf % 3;
        auto *data = samples_.get() + slot * 5 * block;
        if (leaves_[slot] != static_cast<std::int64_t>(leaf))
        {
            const auto first = static_cast<std::int64_t>(leaf) * block;
            work_.read(left_, first, block, data);
            work_.read(right_, first, block, data + block);
            aLeft_.evaluate(work_, leaf, left_, data + 2 * block);
            aRight_.evaluate(work_, leaf, right_, data + 3 * block);
            for (int i = 0; i < block; ++i)
                data[4 * block + i] =
                    data[block + i] * data[2 * block + i] + data[i] * data[3 * block + i];
            leaves_[slot] = static_cast<std::int64_t>(leaf);
        }
        return data;
    }
    void operator()(std::int64_t first, int count, double *output)
    {
        while (count > 0)
        {
            const auto leaf = static_cast<std::size_t>(first / work_.block());
            const int offset = static_cast<int>(first % work_.block());
            const int length = std::min(work_.block() - offset, count);
            std::copy_n(get(leaf) + 4 * work_.block() + offset, length, output);
            first += length;
            output += length;
            count -= length;
        }
    }

  private:
    OfflineProductWorkspace &work_;
    LeftReader &left_;
    RightReader &right_;
    const OfflineHilbertMap &aLeft_, &aRight_;
    OfflineScratchArray<double> samples_;
    std::array<std::int64_t, 3> leaves_{};
};

// Integrating three cardinal sinc functions gives 3/4 when all indices match,
// zero for other equal-parity triples, and 1/(pi^2*(k-a)*(k-b)) when k is the
// index of opposite parity. Grouping these terms yields
// P(x,g) = .75*x*g + A(x)*A(g) - A(g*A(x) + x*A(g)).
// No audio oversampling, circular wrapping or whole-source PCM buffer is used.
template <class LeftReader, class RightReader, class Consumer>
inline void offlineBandlimitedProduct(OfflineSession &job, std::int64_t frames, LeftReader left,
                                      RightReader right, Consumer consume, int block = 0)
{
    if (block == 0)
    {
        block = 128;
        while (block < frames && block < 4096)
            block *= 2;
    }
    OfflineProductWorkspace work(job, frames, block);
    OfflineHilbertMap aLeft(work, left), aRight(work, right);
    OfflineProductCombination combination(work, left, right, aLeft, aRight);
    OfflineHilbertMap aCombined(work, combination);
    auto output = job.allocateScratch<double>(block);
    job.checkpoint(OfflinePhase::Render, 0, frames);
    for (std::size_t leaf = 0; leaf < work.leaves(); ++leaf)
    {
        aCombined.evaluate(work, leaf, combination, output.get(), true);
        const auto *data = combination.get(leaf);
        const auto first = static_cast<std::int64_t>(leaf) * block;
        const int count = static_cast<int>(std::min<std::int64_t>(block, frames - first));
        for (int i = 0; i < count; ++i)
        {
            output[i] = .75 * data[i] * data[block + i] +
                        data[2 * block + i] * data[3 * block + i] - output[i];
            if (!std::isfinite(output[i]))
                offlineFail(OfflineStatus::NumericalFailure);
        }
        consume(first, count, output.get());
        job.checkpoint(OfflinePhase::Render, first + count, frames);
    }
}

} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
