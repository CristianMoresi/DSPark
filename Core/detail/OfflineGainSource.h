// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/** @file OfflineGainSource.h
 * @brief Internal verified PCM/control cache and reusable offline product maps.
 * Threading: worker-owned; the source and control plan remain immutable.
 */
#include "OfflineProduct.h"
#if DSPARK_HAS_OFFLINE
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace dspark::detail
{

// Closed half-line sums for independent constant gain continuation at each end.
// All special-function arguments are positive half-integers. Recurrences are
// anchored every 128 frames, so long sources need neither a full weight array
// nor a transcendental evaluation for every source sample.
class OfflineEndpointWeights final
{
  public:
    OfflineEndpointWeights(OfflineSession &job, std::int64_t frames, int block)
        : frames_(frames), block_(block),
          values_(job.allocate<double>(12 * static_cast<std::uint64_t>(block)))
    {
        cached_.fill(-1);
        center_ = psi(static_cast<double>(frames) * .5 + .5)[0];
    }

    [[nodiscard]] const double *get(std::size_t leaf, int side, bool diagonal = false)
    {
        const auto slot = leaf % 3;
        if (cached_[slot] != static_cast<std::int64_t>(leaf))
        {
            auto *data = values_.get() + slot * 4 * block_;
            std::fill_n(data, 4 * block_, 0.0);
            const auto first = static_cast<std::int64_t>(leaf) * block_;
            const int count = static_cast<int>(std::min<std::int64_t>(block_, frames_ - first));
            std::array<double, 2> left{}, right{};
            std::int64_t previousLeft = 0, previousRight = 0;
            for (int i = 0; i < count; ++i)
            {
                const auto frame = first + i;
                const auto dl = frame + (frame % 2 ? 2 : 1);
                auto dr = frames_ - frame;
                dr += 1 - dr % 2;
                if (i % 128 == 0)
                {
                    left = psi(static_cast<double>(dl) * .5);
                    right = psi(static_cast<double>(dr) * .5);
                }
                else
                {
                    if (dl != previousLeft)
                    {
                        const double reciprocal = 2 / static_cast<double>(previousLeft);
                        left[0] += reciprocal;
                        left[1] -= reciprocal * reciprocal;
                    }
                    if (dr != previousRight)
                    {
                        const double reciprocal = 2 / static_cast<double>(dr);
                        right[0] -= reciprocal;
                        right[1] += reciprocal * reciprocal;
                    }
                }
                previousLeft = dl;
                previousRight = dr;
                data[i] = .5 * (left[0] - center_);
                data[block_ + i] = -.5 * (right[0] - center_);
                constexpr double divisor = 4 * std::numbers::pi * std::numbers::pi;
                data[2 * block_ + i] = left[1] / divisor;
                data[3 * block_ + i] = right[1] / divisor;
            }
            cached_[slot] = static_cast<std::int64_t>(leaf);
        }
        return values_.get() + (slot * 4 + side + (diagonal ? 2 : 0)) * block_;
    }

  private:
    [[nodiscard]] static std::array<double, 2> psi(double z) noexcept
    {
        double p = 0, q = 0;
        while (z < 32)
        {
            const double u = 1 / z;
            p -= u;
            q += u * u;
            z += 1;
        }
        const double u = 1 / z, v = u * u;
        // Digamma and trigamma recurrence plus Bernoulli asymptotic expansion.
        // The first omitted term at z>=32 is below double rounding accuracy.
        p += std::log(z) - u / 2 - v * (1. / 12 - v * (1. / 120 - v *
             (1. / 252 - v * (1. / 240 - v / 132))));
        q += u + v / 2 + u * v * (1. / 6 - v * (1. / 30 - v *
             (1. / 42 - v * (1. / 30 - v * 5. / 66))));
        return {p, q};
    }
    std::int64_t frames_;
    int block_;
    double center_ = 0;
    std::array<std::int64_t, 3> cached_{};
    std::unique_ptr<double[]> values_;
};

template <FloatType T, class Cursor> class OfflineGainSource final
{
    // The nested product stencil and endpoint transform together need five
    // neighbouring source leaves. Retain all five to avoid repeated PCM reads,
    // hashing and control reconstruction when visiting the same output block.
    static constexpr std::size_t cacheLeaves = 5;
  public:
    OfflineGainSource(OfflineAudioSource<T> &source, const OfflineAudioSpec &spec,
                      const OfflineFingerprint &fingerprint, double peak, OfflineSession &job,
                      const OfflineJobOptions &options, Cursor cursor)
        : source_(source), spec_(spec), expected_(fingerprint), peak_(peak), job_(job),
          options_(options), cursor_(std::move(cursor))
    {
        offlineValidateSpec(spec, 2);
        while (block_ < spec.frames && block_ < 4096)
            block_ *= 2;
        leaves_ = static_cast<std::uint64_t>(spec.frames / block_ + (spec.frames % block_ != 0));
        fingerprints_ = job.allocate<OfflineFingerprint>(leaves_);
        audio_ = job.allocate<T>(cacheLeaves * static_cast<std::uint64_t>(block_) * spec.channels);
        gain_ = job.allocate<double>(cacheLeaves * static_cast<std::uint64_t>(block_));
        reset();
        preflight();
    }

    [[nodiscard]] int block() const noexcept
    {
        return block_;
    }
    [[nodiscard]] std::int64_t frames() const noexcept
    {
        return spec_.frames;
    }
    [[nodiscard]] int channels() const noexcept
    {
        return spec_.channels;
    }
    [[nodiscard]] std::size_t leaves() const noexcept
    {
        return static_cast<std::size_t>(leaves_);
    }
    [[nodiscard]] double normalizer() const noexcept
    {
        return peak_ > 0 ? peak_ : 1;
    }
    [[nodiscard]] double baseline() const noexcept
    {
        return baseline_;
    }
    [[nodiscard]] double endpoint(int side) const noexcept
    {
        return endpoints_[side];
    }
    [[nodiscard]] double minimumGain() const noexcept
    {
        return minimum_;
    }
    [[nodiscard]] double maximumGain() const noexcept
    {
        return maximum_;
    }
    [[nodiscard]] double scalarPeak() const noexcept
    {
        return scalarPeak_;
    }
    [[nodiscard]] bool constant() const noexcept
    {
        return minimum_ == maximum_;
    }

    void reset() noexcept
    {
        cached_.fill(-1);
    }

    [[nodiscard]] const T *audio(std::size_t leaf, int channel)
    {
        load(leaf);
        return audio_.get() + (leaf % cacheLeaves * spec_.channels + channel) * block_;
    }
    [[nodiscard]] const double *gain(std::size_t leaf)
    {
        load(leaf);
        return gain_.get() + leaf % cacheLeaves * block_;
    }

    void readAudio(int channel, std::int64_t first, int count, double *output)
    {
        read(first, count, output, [&](std::size_t leaf, int offset, int length, double *out) {
            const auto *input = audio(leaf, channel) + offset;
            for (int i = 0; i < length; ++i)
                out[i] = static_cast<double>(input[i]) / normalizer();
        });
    }
    void readControl(std::int64_t first, int count, double *output)
    {
        read(first, count, output, [&](std::size_t leaf, int offset, int length, double *out) {
            const auto *input = gain(leaf) + offset;
            for (int i = 0; i < length; ++i)
                out[i] = input[i] - baseline_;
        });
    }
    void verifySpec() const
    {
        if (source_.getSpec() != spec_)
            offlineFail(OfflineStatus::SourceMismatch);
    }

  private:
    template <class Sample> void read(std::int64_t first, int count, double *output, Sample sample)
    {
        if (first < 0 || count < 0 || count > spec_.frames || first > spec_.frames - count)
            offlineFail(OfflineStatus::InvalidInput);
        while (count > 0)
        {
            const int offset = static_cast<int>(first % block_);
            const int length = std::min(block_ - offset, count);
            sample(static_cast<std::size_t>(first / block_), offset, length, output);
            first += length;
            output += length;
            count -= length;
        }
    }
    void load(std::size_t leaf)
    {
        if (leaf >= leaves_)
            offlineFail(OfflineStatus::InvalidInput);
        const auto slot = leaf % cacheLeaves;
        if (cached_[slot] == static_cast<std::int64_t>(leaf))
            return;
        const auto first = static_cast<std::int64_t>(leaf) * block_;
        const int count = static_cast<int>(std::min<std::int64_t>(block_, spec_.frames - first));
        auto *data = audio_.get() + slot * spec_.channels * block_;
        auto *control = gain_.get() + slot * block_;
        std::fill_n(data, spec_.channels * block_, T(0));
        for (int offset = 0; offset < count;)
        {
            const int length = std::min(options_.blockFrames, count - offset);
            std::array<T *, 2> channels{};
            for (int c = 0; c < spec_.channels; ++c)
                channels[c] = data + c * block_ + offset;
            offlineRead(source_, spec_, first + offset,
                        AudioBufferView<T>(channels.data(), spec_.channels, length));
            offset += length;
        }
        OfflineFingerprint fingerprint;
        for (int i = 0; i < count; ++i)
        {
            const double gain = cursor_(first + i);
            if (!(gain > 0) || !std::isfinite(gain))
                offlineFail(OfflineStatus::NumericalFailure);
            control[i] = gain;
            for (int c = 0; c < spec_.channels; ++c)
            {
                const T x = data[c * block_ + i];
                if (!std::isfinite(x))
                    offlineFail(OfflineStatus::NonFiniteInput);
                if (std::abs(static_cast<double>(x)) > peak_)
                    offlineFail(OfflineStatus::SourceMismatch);
                offlineHash(fingerprint, x);
            }
        }
        if (verified_ && fingerprint != fingerprints_[leaf])
            offlineFail(OfflineStatus::SourceMismatch);
        if (!verified_)
            fingerprints_[leaf] = fingerprint;
        cached_[slot] = static_cast<std::int64_t>(leaf);
    }
    void preflight()
    {
        job_.checkpoint(OfflinePhase::Verify, 0, spec_.frames);
        OfflineFingerprint fingerprint;
        double firstGain = 1, lastGain = 1;
        for (std::size_t leaf = 0; leaf < leaves_; ++leaf)
        {
            const auto first = static_cast<std::int64_t>(leaf) * block_;
            const int count =
                static_cast<int>(std::min<std::int64_t>(block_, spec_.frames - first));
            const auto *control = gain(leaf);
            const auto *data = audio(leaf, 0);
            if (leaf == 0)
                firstGain = control[0];
            lastGain = control[count - 1];
            for (int i = 0; i < count; ++i)
            {
                minimum_ = std::min(minimum_, control[i]);
                maximum_ = std::max(maximum_, control[i]);
                for (int c = 0; c < spec_.channels; ++c)
                {
                    const T x = data[c * block_ + i];
                    offlineHash(fingerprint, x);
                    scalarPeak_ =
                        std::max(scalarPeak_,
                                 std::abs(static_cast<double>(x) / normalizer() * control[i]));
                }
            }
            job_.checkpoint(OfflinePhase::Verify, first + count, spec_.frames);
        }
        verifySpec();
        if (fingerprint != expected_)
            offlineFail(OfflineStatus::SourceMismatch);
        // The mean is only an affine reference. The gain product independently
        // continues firstGain to the left and lastGain to the right.
        endpoints_ = {firstGain, lastGain};
        baseline_ = (firstGain + lastGain) * .5;
        verified_ = true;
        reset();
    }

    OfflineAudioSource<T> &source_;
    OfflineAudioSpec spec_;
    OfflineFingerprint expected_;
    double peak_;
    OfflineSession &job_;
    const OfflineJobOptions &options_;
    Cursor cursor_;
    int block_ = 128;
    std::uint64_t leaves_ = 0;
    bool verified_ = false;
    double minimum_ = std::numeric_limits<double>::infinity(), maximum_ = 0;
    double baseline_ = 1, scalarPeak_ = 0;
    std::array<double, 2> endpoints_{1, 1};
    std::array<std::int64_t, cacheLeaves> cached_{};
    std::unique_ptr<OfflineFingerprint[]> fingerprints_;
    std::unique_ptr<T[]> audio_;
    std::unique_ptr<double[]> gain_;
};

template <FloatType T, class Cursor> class OfflineGainProduct final
{
    using Source = OfflineGainSource<T, Cursor>;
    struct AudioReader
    {
        Source *source = nullptr;
        int channel = 0;
        void operator()(std::int64_t first, int count, double *output)
        {
            source->readAudio(channel, first, count, output);
        }
    };
    struct ControlReader
    {
        Source *source = nullptr;
        void operator()(std::int64_t first, int count, double *output)
        {
            source->readControl(first, count, output);
        }
    };
    struct WeightedReader
    {
        AudioReader *audio = nullptr;
        OfflineEndpointWeights *weights = nullptr;
        int block = 0, side = 0;
        std::array<double, 2> coefficients{};
        void operator()(std::int64_t first, int count, double *output)
        {
            (*audio)(first, count, output);
            for (int offset = 0; offset < count;)
            {
                const auto frame = first + offset;
                const int begin = static_cast<int>(frame % block);
                const int length = std::min(block - begin, count - offset);
                const auto leaf = static_cast<std::size_t>(frame / block);
                const auto *weight = weights->get(leaf, side == 2 ? 0 : side);
                const auto *right = side == 2 ? weights->get(leaf, 1) : nullptr;
                for (int i = 0; i < length; ++i)
                    output[offset + i] *= right
                        ? coefficients[0] * weight[begin + i] + coefficients[1] * right[begin + i]
                        : weight[begin + i];
                offset += length;
            }
        }
    };
    using Combination = OfflineProductCombination<AudioReader, ControlReader>;

  public:
    OfflineGainProduct(Source &source, OfflineSession &job, bool calibrateBoundary = false)
        : source_(source), work_(job, source.frames(), source.block()), control_{&source}
    {
        endpoints_ = {source.endpoint(0), source.endpoint(1)};
        gainMap_.emplace(work_, control_);
        for (int c = 0; c < source.channels(); ++c)
        {
            source.reset();
            readers_[c] = {&source, c};
            audioMaps_[c].emplace(work_, readers_[c]);
            combinations_[c].emplace(work_, readers_[c], control_, *audioMaps_[c], *gainMap_);
            source.reset();
            combinedMaps_[c].emplace(work_, *combinations_[c]);
        }
        if (calibrateBoundary || source.endpoint(0) != source.endpoint(1))
        {
            boundaryCount_ = calibrateBoundary ? 2 : 1;
            work_.prepareCauchy();
            weights_.emplace(job, source.frames(), source.block());
            boundaryWork_ = job.allocate<double>(2 * static_cast<std::uint64_t>(source.block()));
            if (calibrateBoundary)
                boundaryResponse_ = job.allocate<double>(
                    2 * static_cast<std::uint64_t>(source.block()) * source.channels());
            for (int c = 0; c < source.channels(); ++c)
                for (int side = 0; side < boundaryCount_; ++side)
                {
                    // Fixed endpoints are one linear combination. Only calibration
                    // needs the two independent basis maps and response buffers.
                    weighted_[c][side] = {&readers_[c], &*weights_, source.block(),
                        calibrateBoundary ? side : 2,
                        {endpoints_[0] - source.baseline(), endpoints_[1] - source.baseline()}};
                    source.reset();
                    boundaryMaps_[c][side].emplace(work_, weighted_[c][side]);
                }
        }
        output_ =
            job.allocate<double>(static_cast<std::uint64_t>(source.block()) * source.channels());
    }
    OfflineGainProduct(const OfflineGainProduct &) = delete;
    OfflineGainProduct &operator=(const OfflineGainProduct &) = delete;
    void setEndpoints(double left, double right) noexcept
    {
        endpoints_ = {left, right};
    }
    [[nodiscard]] const double *boundaryResponse(int side) const noexcept
    {
        return boundaryResponse_.get() + side * source_.channels() * source_.block();
    }
    void reset() noexcept
    {
        source_.reset();
        for (int c = 0; c < source_.channels(); ++c)
            combinations_[c]->reset();
    }
    // Returned planar deltas last until the next evaluate(). The maps are reused
    // for target calibration, verification and provisional output, with no new
    // allocations or whole-source temporary between these passes.
    [[nodiscard]] const double *evaluate(std::size_t leaf)
    {
        const int block = source_.block();
        for (int c = 0; c < source_.channels(); ++c)
        {
            auto *output = output_.get() + c * block;
            combinedMaps_[c]->evaluate(work_, leaf, *combinations_[c], output, true);
            const auto *data = combinations_[c]->get(leaf);
            for (int i = 0; i < block; ++i)
                output[i] = (source_.baseline() - 1) * data[i] + .75 * data[i] * data[block + i] +
                            data[2 * block + i] * data[3 * block + i] - output[i];
            if (weights_)
            {
                auto *a = boundaryWork_.get(), *weighted = a + block;
                audioMaps_[c]->evaluateCauchy(work_, leaf, readers_[c], a, true);
                for (int side = 0; side < boundaryCount_; ++side)
                {
                    boundaryMaps_[c][side]->evaluateCauchy(work_, leaf, weighted_[c][side],
                                                          weighted, true);
                    const auto *f = weights_->get(leaf, side);
                    const auto *d = weights_->get(leaf, side, true);
                    const auto *rightF = boundaryCount_ == 1 ? weights_->get(leaf, 1) : nullptr;
                    const auto *rightD = boundaryCount_ == 1 ? weights_->get(leaf, 1, true) : nullptr;
                    auto *response = boundaryResponse_
                        ? boundaryResponse_.get() + (side * source_.channels() + c) * block : weighted;
                    const double left = endpoints_[0] - source_.baseline();
                    const double right = endpoints_[1] - source_.baseline();
                    const double coefficient = boundaryCount_ == 1 ? 1 : side == 0 ? left : right;
                    for (int i = 0; i < block; ++i)
                    {
                        const double sign = i % 2 ? -1. : 1.;
                        const double fValue = rightF ? left * f[i] + right * rightF[i] : f[i];
                        const double dValue = rightD ? left * d[i] + right * rightD[i] : d[i];
                        response[i] = dValue * data[i] +
                            sign / std::numbers::pi * (fValue * a[i] - weighted[i]);
                        output[i] += coefficient * response[i];
                    }
                }
            }
        }
        return output_.get();
    }

  private:
    Source &source_;
    OfflineProductWorkspace work_;
    std::array<AudioReader, 2> readers_{};
    ControlReader control_;
    std::optional<OfflineHilbertMap> gainMap_;
    std::array<std::optional<OfflineHilbertMap>, 2> audioMaps_, combinedMaps_;
    std::array<std::optional<Combination>, 2> combinations_;
    std::optional<OfflineEndpointWeights> weights_;
    std::array<std::array<WeightedReader, 2>, 2> weighted_{};
    std::array<std::array<std::optional<OfflineHilbertMap>, 2>, 2> boundaryMaps_;
    std::unique_ptr<double[]> boundaryWork_;
    std::unique_ptr<double[]> boundaryResponse_;
    std::array<double, 2> endpoints_{1, 1};
    int boundaryCount_ = 0;
    std::unique_ptr<double[]> output_;
};
} // namespace dspark::detail
#endif // DSPARK_HAS_OFFLINE
