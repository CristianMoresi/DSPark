// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineClipProjection.h
 * @brief Bounded complete-source clipping with ideal sinc reconstruction.
 * Internal offline worker engine; reuses Core curves, continuous integration,
 * FIR compensation and finite-source sinc projection. No hidden gain or limiter.
 * Threading: one worker owns each nonmovable instance.
 */
#include "OfflineSinc.h"
#if DSPARK_HAS_OFFLINE
#include "../FIRFilter.h"
#include "ContinuousClip.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace dspark::detail::offline_clip
{
using offline_sinc::SincPhase;
using offline_sinc::SincProjection;

enum class Exterior
{
    Linear,
    Matched
};

// Shared reconstructed clipping stage. Both branches use the same interpolation,
// B-spline moments and compensation. The source/projection boundary policy is
// supplied by BoundedClip; this stage alone does not perform sample-rate conversion.
template <ClipperCurve C> class ReconstructedStage final
{
    static constexpr int degree = 11;
    continuous_clip::Residual<C, degree, true> residual_;
    continuous_clip::Residual<C, degree, false, false> direct_;
    FIRFilter<double> eq_, linear_;
    int order_, filterSize_;

  public:
    ReconstructedStage(OfflineSession &job, int factor)
        : order_(factor <= 2 ? 18 : 10), filterSize_(2 * order_ + degree + 4)
    {
        const int tapsSize = 2 * order_ + 1, linearSize = degree + 4;
        job.charge(static_cast<std::size_t>(6 * (order_ + 1) + (order_ + 2) * tapsSize +
                                            linearSize + filterSize_) *
                       sizeof(double) +
                   static_cast<std::size_t>(order_ + 24) * 64 +
                   static_cast<std::size_t>(tapsSize + filterSize_) *
                       (sizeof(std::atomic<double>) + 4 * sizeof(double)) +
                   2 * sizeof(int));
        const auto taps = continuous_clip::inverseBoxPower(order_),
                   linear = residual_.linearKernel();
        std::vector<double> h(taps.size() + linear.size() - 1);
        for (std::size_t i = 0; i < taps.size(); ++i)
            for (std::size_t j = 0; j < linear.size(); ++j)
                h[i + j] += taps[i] * linear[j];
        eq_.prepare(tapsSize, 1);
        eq_.setCoefficients(taps);
        linear_.prepare(filterSize_, 1);
        linear_.setCoefficients(h);
    }
    void reset(double ceiling) noexcept
    {
        residual_.reset(ceiling);
        direct_.reset(ceiling);
        eq_.reset();
        linear_.reset();
    }
    [[nodiscard]] int filterSize() const noexcept
    {
        return filterSize_;
    }
    [[nodiscard]] int delay() const noexcept
    {
        return (degree + 1) / 2 + 1 + order_;
    }
    void process(double *nonlinear, double *reference, int count, Exterior exterior)
    {
        for (int i = 0; i < count; ++i)
        {
            nonlinear[i] = exterior == Exterior::Matched ? direct_.process(nonlinear[i])
                                                         : residual_.process(nonlinear[i]);
            if (!std::isfinite(nonlinear[i]))
                offlineFail(OfflineStatus::NumericalFailure);
        }
        double *a[]{nonlinear}, *b[]{reference};
        eq_.processBlock({a, 1, count});
        linear_.processBlock({b, 1, count});
    }
};

// The only duration-sized storage is the existing
// coarse Cauchy tree. Upconverted PCM and shaped PCM use three-leaf caches.
template <ClipperCurve C, class Reader> class BoundedClip
{
    static constexpr double slope = clipperSmallSignalSlope<C, double>();
    OfflineSession &job_;
    Reader source_;
    std::int64_t frames_, pad_, extent_;
    int factor_, block_, highBlock_, delay_ = 0, left_ = 0, filterSize_ = 0;
    double ceiling_;
    Exterior exterior_;
    OfflineProductWorkspace work_;
    struct PaddedReader
    {
        BoundedClip *owner;
        bool difference;
        void operator()(std::int64_t first, int count, double *out)
        {
            owner->readPadded(first, count, out, difference);
        }
    } raw_{this, false}, difference_{this, true};
    std::optional<OfflineHilbertMap> rawMap_, differenceMap_;
    std::array<std::optional<SincPhase>, 16> phases_;
    std::unique_ptr<double[]> native_, up_, shaped_, scratchA_, scratchB_, phaseWork_, output_;
    std::array<std::int64_t, 3> nativeLeaf_{-1, -1, -1};
    std::array<std::int64_t, 3> upLeaf_{-1, -1, -1}, shapedLeaf_{-1, -1, -1};
    ReconstructedStage<C> stage_;
    std::uint64_t evaluations_ = 0;
    struct ProjectedReader
    {
        BoundedClip *owner;
        void operator()(std::int64_t first, int count, double *out)
        {
            owner->projectedSource(first, count, out);
        }
    };
    std::optional<SincProjection<ProjectedReader>> projection_;

    static std::int64_t checkedExtent(std::int64_t frames, std::int64_t pad, int factor, int block)
    {
        if (frames < 1 || pad < 64 || factor < 1 || factor > 16 || (factor & (factor - 1)) ||
            block < 128 || block > 16384 || block * factor > 65536 || (block & (block - 1)))
            offlineFail(OfflineStatus::InvalidInput);
        constexpr auto maximum = std::numeric_limits<std::int64_t>::max();
        if (pad > (maximum / factor - frames) / 2 || frames > maximum / factor)
            offlineFail(OfflineStatus::MemoryLimit);
        return frames + 2 * pad;
    }
    void readPadded(std::int64_t first, int count, double *out, bool difference)
    {
        while (count > 0)
        {
            const auto leaf = first / block_;
            const int offset = static_cast<int>(first % block_),
                      length = std::min(count, block_ - offset);
            const auto slot = static_cast<std::size_t>(leaf % 3);
            auto *data = native_.get() + slot * 2 * block_;
            if (nativeLeaf_[slot] != leaf)
            {
                std::fill_n(data, 2 * block_, 0.);
                const auto origin = leaf * block_;
                const auto begin = std::max(origin, pad_),
                           end = std::min(origin + block_, pad_ + frames_);
                if (begin < end)
                {
                    const int at = static_cast<int>(begin - origin),
                              n = static_cast<int>(end - begin);
                    for (int done = 0; done < n;)
                    {
                        const int chunk = std::min(n - done, 4096);
                        source_(begin - pad_ + done, chunk, data + at + done);
                        done += chunk;
                    }
                    for (int i = at; i < at + n; ++i)
                    {
                        if (!std::isfinite(data[i]))
                            offlineFail(OfflineStatus::NonFiniteInput);
                        const double shaped = clipperShape<C>(data[i], ceiling_);
                        data[block_ + i] =
                            exterior_ == Exterior::Matched ? -shaped : slope * data[i] - shaped;
                    }
                }
                nativeLeaf_[slot] = leaf;
            }
            std::copy_n(data + (difference ? block_ : 0) + offset, length, out);
            first += length;
            count -= length;
            out += length;
        }
    }
    double *upsampled(std::int64_t leaf)
    {
        const auto slot = static_cast<std::size_t>(leaf % 3);
        auto *data = up_.get() + slot * 2 * highBlock_;
        if (upLeaf_[slot] == leaf)
            return data;
        for (int p = 0; p < factor_; ++p)
        {
            phases_[p]->evaluate(*rawMap_, static_cast<std::size_t>(leaf), raw_, phaseWork_.get());
            for (int i = 0; i < block_; ++i)
                data[i * factor_ + p] = phaseWork_[i];
            phases_[p]->evaluate(*differenceMap_, static_cast<std::size_t>(leaf), difference_,
                                 phaseWork_.get());
            for (int i = 0; i < block_; ++i)
                data[highBlock_ + i * factor_ + p] = phaseWork_[i];
        }
        upLeaf_[slot] = leaf;
        return data;
    }
    void highInput(std::int64_t first, int count, double *a, double *b)
    {
        const auto highExtent = extent_ * factor_;
        while (count > 0)
        {
            if (first < 0)
            {
                const int n = static_cast<int>(std::min<std::int64_t>(-first, count));
                std::fill_n(a, n, 0.);
                std::fill_n(b, n, 0.);
                first += n;
                count -= n;
                a += n;
                b += n;
                continue;
            }
            if (first >= highExtent)
            {
                std::fill_n(a, count, 0.);
                std::fill_n(b, count, 0.);
                break;
            }
            const auto leaf = first / highBlock_;
            const int offset = static_cast<int>(first % highBlock_);
            const int n = static_cast<int>(
                std::min<std::int64_t>(std::min(highBlock_ - offset, count), highExtent - first));
            const auto *data = upsampled(leaf);
            std::copy_n(data + offset, n, a);
            std::copy_n(data + highBlock_ + offset, n, b);
            first += n;
            count -= n;
            a += n;
            b += n;
        }
    }
    const double *stage(std::int64_t leaf)
    {
        const auto slot = static_cast<std::size_t>(leaf % 3);
        auto *data = shaped_.get() + slot * 2 * highBlock_;
        if (shapedLeaf_[slot] == leaf)
            return data;
        const auto first = leaf * highBlock_;
        const int length = highBlock_ + filterSize_ - 1;
        highInput(first - left_, length, scratchA_.get(), scratchB_.get());
        stage_.reset(ceiling_);
        stage_.process(scratchA_.get(), scratchB_.get(), length, exterior_);
        std::copy_n(scratchA_.get() + filterSize_ - 1, highBlock_, data);
        std::copy_n(scratchB_.get() + filterSize_ - 1, highBlock_, data + highBlock_);
        shapedLeaf_[slot] = leaf;
        ++evaluations_;
        return data;
    }
    void projectedSource(std::int64_t first, int count, double *out)
    {
        while (count > 0)
        {
            const auto leaf = first / highBlock_;
            const int offset = static_cast<int>(first % highBlock_);
            const int n = std::min(highBlock_ - offset, count);
            const auto *data = stage(leaf);
            for (int i = 0; i < n; ++i)
                out[i] = data[offset + i] +
                         (exterior_ == Exterior::Matched ? data[highBlock_ + offset + i] : 0.);
            first += n;
            count -= n;
            out += n;
        }
    }

  public:
    BoundedClip(OfflineSession &job, Reader reader, std::int64_t frames, std::int64_t pad,
                int factor, double ceiling, Exterior exterior, int block = 512)
        : job_(job), source_(std::move(reader)), frames_(frames), pad_(pad),
          extent_(checkedExtent(frames, pad, factor, block)), factor_(factor), block_(block),
          highBlock_(block * factor), ceiling_(ceiling), exterior_(exterior),
          work_(job, extent_, block, OfflineProductWorkspace::NearFields::External),
          stage_(job, factor)
    {
        if (!(ceiling > 0) || !std::isfinite(ceiling))
            offlineFail(OfflineStatus::InvalidInput);
        native_ = job.allocate<double>(6 * static_cast<std::uint64_t>(block_));
        rawMap_.emplace(work_, raw_);
        differenceMap_.emplace(work_, difference_);
        for (int p = 0; p < factor; ++p)
            phases_[p].emplace(work_, double(p) / factor);
        filterSize_ = stage_.filterSize();
        delay_ = stage_.delay();
        left_ = filterSize_ - 1 - delay_;
        up_ = job.allocate<double>(6 * static_cast<std::uint64_t>(highBlock_));
        shaped_ = job.allocate<double>(6 * static_cast<std::uint64_t>(highBlock_));
        scratchA_ = job.allocate<double>(static_cast<std::uint64_t>(highBlock_) + filterSize_ - 1);
        scratchB_ = job.allocate<double>(static_cast<std::uint64_t>(highBlock_) + filterSize_ - 1);
        phaseWork_ = job.allocate<double>(block_);
        output_ = job.allocate<double>(block_);
    }
    BoundedClip(const BoundedClip &) = delete;
    BoundedClip &operator=(const BoundedClip &) = delete;
    [[nodiscard]] bool hasGeometry(std::int64_t pad, int block) const noexcept
    {
        return pad_ == pad && block_ == block;
    }
    void reset(double ceiling, Exterior exterior)
    {
        if (!(ceiling > 0) || !std::isfinite(ceiling))
            offlineFail(OfflineStatus::InvalidInput);
        if (ceiling == ceiling_ && exterior == exterior_)
            return;
        ceiling_ = ceiling;
        exterior_ = exterior;
        nativeLeaf_.fill(-1);
        upLeaf_.fill(-1);
        shapedLeaf_.fill(-1);
        differenceMap_->rebuild(work_, difference_);
        if (projection_)
            projection_->rebuild();
    }
    [[nodiscard]] std::uint64_t evaluations() const
    {
        return evaluations_;
    }
    void prepareProjection()
    {
        if (!projection_)
            projection_.emplace(job_, extent_ * factor_, factor_, ProjectedReader{this},
                                highBlock_);
    }
    [[nodiscard]] std::int64_t firstLeaf() const
    {
        return pad_ / block_;
    }
    [[nodiscard]] std::int64_t endLeaf() const
    {
        return (pad_ + frames_ - 1) / block_ + 1;
    }
    [[nodiscard]] std::int64_t leafOrigin(std::int64_t leaf) const
    {
        return std::max(leaf * block_, pad_) - pad_;
    }
    [[nodiscard]] std::span<const double> get(std::int64_t leaf)
    {
        if (leaf < firstLeaf() || leaf >= endLeaf())
            offlineFail(OfflineStatus::InvalidInput);
        prepareProjection();
        const auto values = projection_->get(static_cast<std::size_t>(leaf));
        const auto first = leaf * block_, begin = std::max(first, pad_),
                   end = std::min(first + static_cast<int>(values.size()), pad_ + frames_);
        const auto *filtered = stage(leaf), *raw = upsampled(leaf);
        const int offset = static_cast<int>(begin - first), length = static_cast<int>(end - begin);
        for (int i = 0; i < length; ++i)
        {
            const int at = offset + i;
            output_[i] = clipperShape<C>(raw[at * factor_], ceiling_) + values[at] +
                         (exterior_ == Exterior::Linear ? filtered[highBlock_ + at * factor_] : 0.);
            if (!std::isfinite(output_[i]))
                offlineFail(OfflineStatus::NumericalFailure);
        }
        return {output_.get(), static_cast<std::size_t>(length)};
    }
    template <class Consumer> void render(Consumer consume)
    {
        prepareProjection();
        for (auto leaf = firstLeaf(); leaf < endLeaf(); ++leaf)
        {
            const auto data = get(leaf);
            consume(leafOrigin(leaf), static_cast<int>(data.size()), data.data());
        }
    }
};
// Outward-rounded positive arithmetic and signed prefix intervals. Abel
// summation bounds a finite alternating Cauchy sum by (|total|+max|prefix|)/d.
// This often avoids the duration-dependent sum-of-magnitudes bound, while
// retaining that bound as the better fallback for a Nyquist-alternating input.
inline double upper(double value)
{
    return std::nextafter(value, std::numeric_limits<double>::infinity());
}
class TailCertificate
{
    double low_ = 0, high_ = 0, mass_ = 0, prefix_ = 0;
    std::int64_t next_ = 0;

  public:
    void append(double x)
    {
        if (!std::isfinite(x))
            offlineFail(OfflineStatus::NonFiniteInput);
        const double a = next_ % 2 ? -x : x;
        low_ = std::nextafter(low_ + a, -std::numeric_limits<double>::infinity());
        high_ = upper(high_ + a);
        mass_ = upper(mass_ + std::abs(x));
        prefix_ = std::max(prefix_, std::max(std::abs(low_), std::abs(high_)));
        ++next_;
    }
    [[nodiscard]] double cauchyEnvelope() const
    {
        const double total = std::max(std::abs(low_), std::abs(high_));
        const double bound = std::min(mass_, upper(total + prefix_));
        return upper(bound / std::nextafter(std::numbers::pi, 0.));
    }
    [[nodiscard]] double mass() const
    {
        return mass_;
    }
};
struct TailPlan
{
    std::int64_t padding = 0;
    Exterior exterior = Exterior::Linear;
    double bound = 0;
};

template <ClipperCurve C> class TailPolicy
{
    double l_ = 0, e_ = 0, radius_ = 0, cubic_ = 0, shapeMaximum_ = 1, shapeGain_ = 1;
    static constexpr int degree = 11;

  public:
    explicit TailPolicy(int factor)
    {
        if (factor < 1 || factor > 16 || (factor & (factor - 1)))
            offlineFail(OfflineStatus::InvalidInput);
        const int order = factor <= 2 ? 18 : 10;
        continuous_clip::Interval<C, degree, true> interval(1.);
        l_ = upper(interval.reconstructionBound() *
                   (1 + 256 * std::numeric_limits<double>::epsilon()));
        for (double v : continuous_clip::inverseBoxPower(order))
            e_ = upper(e_ + std::abs(v));
        radius_ = double(degree + 4 + 2 * order + 2) / factor + 2;
        if constexpr (C == ClipperCurve::Tanh)
            cubic_ = upper(1. / 3);
        if constexpr (C == ClipperCurve::Sine)
        {
            for (int i = 1; i < 5; ++i)
                cubic_ = upper(cubic_ + std::abs(fastSinPolynomial<double>[i]));
            shapeGain_ = shapeMaximum_ = 0;
            const auto lower = [](double x)
            { return std::nextafter(x, -std::numeric_limits<double>::infinity()); };
            for (int part = 0; part < 256; ++part)
            {
                const double low = std::max(0., lower(halfPi<double> * part / 256)),
                             high = upper(halfPi<double> * (part + 1) / 256);
                const double squareLow = std::max(0., lower(low * low)),
                             squareHigh = upper(high * high);
                double a = fastSinPolynomial<double>[4], b = a;
                for (int i = 3; i >= 0; --i)
                {
                    const std::array<double, 4> products{a * squareLow, a * squareHigh,
                                                         b * squareLow, b * squareHigh};
                    a = lower(lower(*std::min_element(products.begin(), products.end())) +
                              fastSinPolynomial<double>[i]);
                    b = upper(upper(*std::max_element(products.begin(), products.end())) +
                              fastSinPolynomial<double>[i]);
                }
                const double gain = std::max(std::abs(a), std::abs(b));
                shapeGain_ = std::max(shapeGain_, gain);
                shapeMaximum_ = std::max(shapeMaximum_, upper(gain * high));
            }
        }
    }
    // Analytic omitted-tail bounds. Roundoff/continuous-kernel errors are
    // separately tested; they are not silently included in this tail budget.
    TailPlan choose(double c, const TailCertificate &source, const TailCertificate &shape,
                    double tolerance = 1e-10, std::int64_t maximumPadding = 1ll << 26) const
    {
        if (!(c > 0) || !std::isfinite(c) || !(tolerance > 0) || !std::isfinite(tolerance))
            offlineFail(OfflineStatus::InvalidInput);
        const double a = upper(l_ * source.cauchyEnvelope()),
                     af = upper(l_ * shape.cauchyEnvelope());
        for (std::int64_t pad = 64; pad <= maximumPadding; pad *= 2)
        {
            const double p = static_cast<double>(pad) - radius_;
            if (p <= 0)
                continue;
            const double amplitude = upper(a / p);
            double bound = std::numeric_limits<double>::infinity();
            if constexpr (C == ClipperCurve::Hard || C == ClipperCurve::GoldenRatio)
            {
                const double threshold = C == ClipperCurve::Hard
                                             ? c
                                             : clipperLinearLimits<ClipperCurve::GoldenRatio>(c)[1];
                if (amplitude < threshold)
                    bound = 0;
            }
            else if constexpr (C == ClipperCurve::Tanh || C == ClipperCurve::Sine)
            {
                if (amplitude <= c)
                {
                    const double ratio = upper(amplitude / c);
                    // Integral of d^-4; factor4 includes both ends and the
                    // altered boundary strip of the finite evaluation domain.
                    bound =
                        upper(upper(upper(4 * e_ * cubic_ / (3 * std::numbers::pi)) * amplitude) *
                              upper(ratio * ratio));
                }
            }
            if (bound <= tolerance)
                return {pad, Exterior::Linear, bound};
            const double shapeA = upper(a * shapeGain_), limit = upper(c * shapeMaximum_);
            double shaped = upper(shapeA / p);
            if (shaped > limit)
            {
                const double logarithm = std::log(shapeA) - std::log(limit) - std::log(p);
                shaped = upper(limit * (1 + logarithm + 1e-12 * (1 + std::abs(logarithm))));
            }
            const double matched =
                upper(upper(4 * e_ / std::numbers::pi) * upper(shaped + upper(af / p)));
            if (matched <= tolerance)
                return {pad, Exterior::Matched, matched};
            if (pad > maximumPadding / 2)
                break;
        }
        offlineFail(OfflineStatus::MemoryLimit);
    }
};
} // namespace dspark::detail::offline_clip
#endif // DSPARK_HAS_OFFLINE
