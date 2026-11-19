// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LaneCascade.h
 * @brief A long cascade of biquads on one channel spread over the lanes of a
 *        vector: each lane runs its own stretch of the cascade, one sample
 *        behind the lane before it.
 *
 * A cascade is serial: section k needs section k - 1's output of the same
 * sample, so one channel cannot fill a vector. Cut the cascade into L
 * stretches of K sections and let lane j run stretch j on the signal that
 * left stretch j - 1 one sample earlier: every lane is busy every sample,
 * the K steps of a sample run as K vector operations over the L lanes, and
 * the output is the full cascade, L - 1 samples late. The latency is the
 * price, so BiquadCascade stays the cascade of the processors that must not
 * have any; this one serves long banks (tens of sections) behind an analysis
 * that already has latency.
 *
 * The sections keep their history in double, in direct form I, and their
 * coefficients can move every sample: rampCoeffs() interpolates all five
 * linearly, as BiquadCascade does. The arrays are laid out step by step with
 * the lanes contiguous, so the inner loop over the lanes vectorizes.
 *
 * Threading: owner-managed (every method on the processing thread).
 *
 * Dependencies: AudioBuffer.h, Biquad.h, DspMath.h, SimdOps.h.
 */

#include "AudioBuffer.h"
#include "Biquad.h"
#include "DspMath.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <vector>

namespace dspark {

/**
 * @class LaneCascade
 * @brief A cascade of biquads over the lanes of a vector, with latency.
 *
 * @tparam T     Sample type (float or double); the arithmetic is double.
 * @tparam Lanes Lanes: 2, 4, 8 or 16 (latency Lanes - 1 samples).
 */
template <FloatType T, int Lanes = 8>
class LaneCascade
{
    static_assert(Lanes == 2 || Lanes == 4 || Lanes == 8 || Lanes == 16, "Lanes must be 2, 4, 8 or 16");

public:
    /**
     * @brief Allocates for a number of sections and channels; every section
     *        starts transparent (b0 = 1).
     * @param sections Sections (at least 1; rounded up to a multiple of the lanes).
     * @param channels Channels (at least 1).
     */
    void prepare(int sections, int channels)
    {
        steps_ = (std::max(1, sections) + Lanes - 1) / Lanes;
        sections_ = std::max(1, sections);
        channels_ = std::max(1, channels);
        const auto total = static_cast<size_t>(steps_ * Lanes);
        for (auto* v : { &b0_, &b1_, &b2_, &a1_, &a2_, &db0_, &db1_, &db2_, &da1_, &da2_ })
            v->assign(total, 0.0);
        std::fill(b0_.begin(), b0_.end(), 1.0);
        for (auto* v : { &t0_, &t1_, &t2_, &u1_, &u2_ }) v->assign(total, 0.0);
        std::fill(t0_.begin(), t0_.end(), 1.0);
        rampLeft_ = 0;
        state_.assign(static_cast<size_t>(channels_), std::vector<double>(total * 4, 0.0));
        pipe_.assign(static_cast<size_t>(channels_), std::array<double, Lanes> {});
    }

    [[nodiscard]] int getNumSections() const noexcept { return sections_; }
    /** @return The latency in samples: one per lane after the first. */
    [[nodiscard]] static constexpr int getLatency() noexcept { return Lanes - 1; }

    /** @brief Sets a section's coefficients at once (ends any ramp of it). */
    void setCoeffs(int section, const BiquadCoeffs& c) noexcept
    {
        const size_t i = slot(section);
        if (i == npos) return;
        b0_[i] = t0_[i] = c.b0; b1_[i] = t1_[i] = c.b1; b2_[i] = t2_[i] = c.b2;
        a1_[i] = u1_[i] = c.a1; a2_[i] = u2_[i] = c.a2;
        db0_[i] = db1_[i] = db2_[i] = da1_[i] = da2_[i] = 0.0;
    }

    /**
     * @brief The coefficients a section ramps to at the next startRamp()
     *        (sections not given a target keep the one they had).
     */
    void setTargetCoeffs(int section, const BiquadCoeffs& c) noexcept
    {
        const size_t i = slot(section);
        if (i == npos) return;
        t0_[i] = c.b0; t1_[i] = c.b1; t2_[i] = c.b2; u1_[i] = c.a1; u2_[i] = c.a2;
    }

    /**
     * @brief Ramps every section linearly from where it is to its target over
     *        `samples` (0: at once); the ramp ends exactly on the targets.
     */
    void startRamp(int samples) noexcept
    {
        const size_t total = b0_.size();
        if (samples <= 0)
        {
            b0_ = t0_; b1_ = t1_; b2_ = t2_; a1_ = u1_; a2_ = u2_;
            rampLeft_ = 0;
            return;
        }
        const double inv = 1.0 / samples;
        for (size_t i = 0; i < total; ++i)
        {
            db0_[i] = (t0_[i] - b0_[i]) * inv; db1_[i] = (t1_[i] - b1_[i]) * inv; db2_[i] = (t2_[i] - b2_[i]) * inv;
            da1_[i] = (u1_[i] - a1_[i]) * inv; da2_[i] = (u2_[i] - a2_[i]) * inv;
        }
        rampLeft_ = samples;
    }

    /** @brief Ramps one section to a target over `samples` (startRamp() on every section's target). */
    void rampCoeffs(int section, const BiquadCoeffs& c, int samples) noexcept
    {
        setTargetCoeffs(section, c);
        startRamp(samples);
    }

    /** @return A section's coefficients now. */
    [[nodiscard]] BiquadCoeffs getCoeffs(int section) const noexcept
    {
        BiquadCoeffs c;
        const size_t i = slot(section);
        if (i == npos) return c;
        c.b0 = b0_[i]; c.b1 = b1_[i]; c.b2 = b2_[i]; c.a1 = a1_[i]; c.a2 = a2_[i];
        return c;
    }

    /** @brief Clears the history and the pipeline (the coefficients stay). */
    void reset() noexcept
    {
        for (auto& s : state_) std::fill(s.begin(), s.end(), 0.0);
        for (auto& p : pipe_) p.fill(0.0);
    }

    /** @brief Runs a block in place on the prepared channels (extra ones pass). */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        const int nS = buffer.getNumSamples();
        for (int n = 0; n < nS; ++n)
        {
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                data[n] = static_cast<T>(step(ch, static_cast<double>(data[n])));
            }
            if (rampLeft_ > 0) advanceRamp();
        }
    }

private:
    static constexpr size_t npos = static_cast<size_t>(-1);

    /** @brief Section s is stretch s / steps, step s % steps: slot step * Lanes + lane. */
    [[nodiscard]] size_t slot(int section) const noexcept
    {
        if (section < 0 || section >= steps_ * Lanes) return npos;
        return static_cast<size_t>((section % steps_) * Lanes + section / steps_);
    }

    double step(int ch, double x) noexcept
    {
        auto& pipe = pipe_[static_cast<size_t>(ch)];
        double* st = state_[static_cast<size_t>(ch)].data();
        alignas(64) std::array<double, Lanes> v {};
        v[0] = x;
        for (int j = 1; j < Lanes; ++j) v[static_cast<size_t>(j)] = pipe[static_cast<size_t>(j - 1)];
        constexpr int W = std::min(simd::kVecWidth<double>, Lanes);
        using O = simd::Vec<double, W>;
        using V = typename O::V;
        for (int k = 0; k < steps_; ++k)
        {
            const size_t base = static_cast<size_t>(k * Lanes);
            double* x1 = st + 4 * base;
            double* x2 = x1 + Lanes;
            double* y1 = x2 + Lanes;
            double* y2 = y1 + Lanes;
            for (int j = 0; j < Lanes; j += W)
            {
                const auto i = base + static_cast<size_t>(j);
                const V in = O::load(v.data() + j);
                const V px1 = O::load(x1 + j), px2 = O::load(x2 + j);
                const V py1 = O::load(y1 + j), py2 = O::load(y2 + j);
                const V ff = O::madd(O::load(b0_.data() + i), in,
                                     O::madd(O::load(b1_.data() + i), px1, O::mul(O::load(b2_.data() + i), px2)));
                const V fb = O::madd(O::load(a1_.data() + i), py1, O::mul(O::load(a2_.data() + i), py2));
                const V y = O::sub(ff, fb);
                O::store(x2 + j, px1);
                O::store(x1 + j, in);
                O::store(y2 + j, py1);
                O::store(y1 + j, y);
                O::store(v.data() + j, y);
            }
        }
        for (int j = 0; j < Lanes; ++j) pipe[static_cast<size_t>(j)] = v[static_cast<size_t>(j)];
        return v[static_cast<size_t>(Lanes - 1)];
    }

    void advanceRamp() noexcept
    {
        if (--rampLeft_ == 0)
        {
            b0_ = t0_; b1_ = t1_; b2_ = t2_; a1_ = u1_; a2_ = u2_;   // no allocation: same sizes
            return;
        }
        const size_t total = b0_.size();
        for (size_t i = 0; i < total; ++i)
        {
            b0_[i] += db0_[i]; b1_[i] += db1_[i]; b2_[i] += db2_[i];
            a1_[i] += da1_[i]; a2_[i] += da2_[i];
        }
    }

    int steps_ = 1, sections_ = 1, channels_ = 1;
    int rampLeft_ = 0;
    std::vector<double> b0_, b1_, b2_, a1_, a2_, db0_, db1_, db2_, da1_, da2_;
    std::vector<double> t0_, t1_, t2_, u1_, u2_;   ///< The targets of the ramp.
    std::vector<std::vector<double>> state_;
    std::vector<std::array<double, Lanes>> pipe_;
};

} // namespace dspark
