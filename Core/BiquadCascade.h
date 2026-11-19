// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file BiquadCascade.h
 * @brief A cascade of second-order sections run in one pass, with coefficient
 *        ramps and click-free bypass.
 *
 * `Biquad` is one section with a thread-safe coefficient hand-off. A cascade
 * of them (an equalizer's bands, a steep cut, a crossover branch) is usually
 * run one section after another over the block, and each of those passes is
 * bound by the latency of its own recursion: the processor waits for the
 * previous output before it can start the next sample, with most of its
 * arithmetic units idle. BiquadCascade runs up to four sections per pass with
 * all of them inside the loop: the recursions of different sections do not
 * depend on each other across samples, so they overlap in the processor, and
 * a pass over four sections costs little more than a pass over one.
 *
 * The sections are in direct form I with shared node history: the memory of
 * the cascade is the last two samples at each node (the input, and the output
 * of every section), in double precision, so the output history of a section
 * is the input history of the next and each section adds two values of state.
 * That choice has three consequences:
 *
 * - **The state does not depend on the coefficients.** It is signal, not
 *   weighted sums of it, so coefficients can move every sample (rampCoeffs()
 *   interpolates all five linearly) and the filter carries on from what it
 *   has just played.
 * - **A section whose numerator equals its denominator is transparent with
 *   its poles still running.** BiquadCoeffs::bypassed() is that set, and
 *   ramping a section to it (rampToBypass()) takes it out of the path without
 *   a crossfade or a click: what the section was adding dies away at the rate
 *   of its own poles. A bell at 0 dB is already that set.
 * - **A disabled section costs nothing and comes back clean.** Once
 *   transparent it can be switched off (setSectionEnabled()); switched on
 *   again it starts from the history of its input, as if it had been
 *   transparent all along. (A section whose memory is weighted sums keeps,
 *   while it is skipped, the memory of the moment it left.)
 *
 * Pairs of channels run in the two lanes of one register where the target has
 * them, as in Biquad::processBlock(), and the samples stay in double from the
 * input of the cascade to its output.
 *
 * Threading: owner-managed. Every method belongs to the thread that calls
 * processBlock() (or to a single thread, offline); there is no cross-thread
 * hand-off here. A processor that embeds a cascade publishes its parameters
 * through its own atomics and calls these setters from its audio thread.
 *
 * Dependencies: AudioBuffer.h, Biquad.h, DenormalGuard.h, SimdOps.h.
 */

#include "AudioBuffer.h"
#include "Biquad.h"
#include "DenormalGuard.h"
#include "SimdOps.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace dspark {

/**
 * @class BiquadCascade
 * @brief Up to MaxSections second-order sections in series, run in one pass.
 *
 * @tparam T           Sample type of the buffers (float or double). The
 *                     recursion and its memory are double either way.
 * @tparam MaxSections Largest number of sections.
 * @tparam MaxChannels Largest number of channels.
 */
template <typename T, int MaxSections = 8, int MaxChannels = 8>
class BiquadCascade
{
public:
    /** @brief Sections run together in one pass over the samples. */
    static constexpr int kGroup = 4;

    BiquadCascade() noexcept
    {
        enabled_.fill(true);
        rampLeft_.fill(0);
        rebuildRoute();
    }

    /**
     * @brief Sets how many sections the cascade has.
     *
     * Sections from the new count on are left out of the path as they are
     * (their coefficients are kept); a section that comes into the path
     * starts from the history of its input.
     *
     * @param count Number of sections, clamped to [0, MaxSections].
     */
    void setNumSections(int count) noexcept
    {
        const int clamped = std::clamp(count, 0, MaxSections);
        if (clamped == numSections_) return;
        const int before = numSections_;
        numSections_ = clamped;
        rebuildRoute();
        for (int s = std::max(before, 0); s < clamped; ++s)
            if (enabled_[static_cast<size_t>(s)]) seedFromInput(s);
    }

    /** @brief Number of sections in the cascade. */
    [[nodiscard]] int getNumSections() const noexcept { return numSections_; }

    /**
     * @brief Makes a coefficient set a section's own from the next sample.
     *
     * Ends any ramp of the section. The state is signal, so the section
     * carries on from what it has played; a large change is still a step in
     * its response (use rampCoeffs() to glide).
     *
     * @param section Section index in [0, MaxSections); others are ignored.
     * @param c       The coefficients.
     */
    void setCoeffs(int section, const BiquadCoeffs& c) noexcept
    {
        if (section < 0 || section >= MaxSections) return;
        const auto s = static_cast<size_t>(section);
        current_[s] = c;
        target_[s] = c;
        rampLeft_[s] = 0;
        updateRamping();
    }

    /**
     * @brief Moves a section's five coefficients linearly to a set.
     *
     * The ramp starts from the coefficients in use (wherever a ramp in
     * flight has brought them) and lands exactly on the target after
     * `samples` samples.
     *
     * @param section Section index in [0, MaxSections); others are ignored.
     * @param target  Coefficients to arrive at.
     * @param samples Length of the ramp in samples; 0 or less sets the
     *                target at once.
     */
    void rampCoeffs(int section, const BiquadCoeffs& target, int samples) noexcept
    {
        if (section < 0 || section >= MaxSections) return;
        if (samples <= 0) { setCoeffs(section, target); return; }
        const auto s = static_cast<size_t>(section);
        target_[s] = target;
        rampLeft_[s] = samples;
        const double inverse = 1.0 / static_cast<double>(samples);
        step_[s].b0 = (target.b0 - current_[s].b0) * inverse;
        step_[s].b1 = (target.b1 - current_[s].b1) * inverse;
        step_[s].b2 = (target.b2 - current_[s].b2) * inverse;
        step_[s].a1 = (target.a1 - current_[s].a1) * inverse;
        step_[s].a2 = (target.a2 - current_[s].a2) * inverse;
        ramping_ = true;
    }

    /**
     * @brief Takes a section out of the path without a click.
     *
     * Ramps the numerator of the section to its denominator: the poles and
     * the memory keep running and what the section adds to its input dies
     * away. Once the ramp is over the section passes its input (to rounding)
     * and can be disabled.
     *
     * @param section Section index.
     * @param samples Length of the ramp in samples.
     */
    void rampToBypass(int section, int samples) noexcept
    {
        if (section < 0 || section >= MaxSections) return;
        rampCoeffs(section, target_[static_cast<size_t>(section)].bypassed(), samples);
    }

    /** @brief The coefficients a section is using now (mid ramp, where the ramp is). */
    [[nodiscard]] BiquadCoeffs getCoeffs(int section) const noexcept
    {
        return current_[static_cast<size_t>(std::clamp(section, 0, MaxSections - 1))];
    }

    /** @brief The coefficients a section is heading to (its own when it is not ramping). */
    [[nodiscard]] BiquadCoeffs getTargetCoeffs(int section) const noexcept
    {
        return target_[static_cast<size_t>(std::clamp(section, 0, MaxSections - 1))];
    }

    /** @brief Whether a section's coefficients are moving. */
    [[nodiscard]] bool isRamping(int section) const noexcept
    {
        return section >= 0 && section < MaxSections && rampLeft_[static_cast<size_t>(section)] > 0;
    }

    /** @brief Whether any section's coefficients are moving. */
    [[nodiscard]] bool isRamping() const noexcept { return ramping_; }

    /**
     * @brief Switches a section in or out of the path.
     *
     * A switch, not a fade: disable a section once it is transparent (after
     * rampToBypass(), or while its coefficients are a bypassed() set) and it
     * leaves without a trace. A disabled section is not computed. Enabled
     * again it starts from the history of its input, so enabling a
     * transparent section is silent too.
     *
     * @param section Section index in [0, MaxSections); others are ignored.
     * @param on      True to compute the section.
     */
    void setSectionEnabled(int section, bool on) noexcept
    {
        if (section < 0 || section >= MaxSections) return;
        const auto s = static_cast<size_t>(section);
        if (enabled_[s] == on) return;
        enabled_[s] = on;
        rebuildRoute();
        if (on && section < numSections_) seedFromInput(section);
    }

    /** @brief Whether a section is computed. */
    [[nodiscard]] bool isSectionEnabled(int section) const noexcept
    {
        return section >= 0 && section < MaxSections && enabled_[static_cast<size_t>(section)];
    }

    /** @brief Clears the memory of every node. Coefficients and ramps are kept. */
    void reset() noexcept
    {
        for (auto& channel : node_)
            for (auto& each : channel) each = {};
    }

    /**
     * @brief Runs a block through the cascade, in place.
     *
     * Channels beyond MaxChannels are left untouched. A memory that a
     * non-finite sample has reached is cleared at the end of the block, so
     * the channel recovers on the next one.
     *
     * @param buffer Audio to filter.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        const int numChannels = std::min(buffer.getNumChannels(), MaxChannels);
        const int numSamples = buffer.getNumSamples();
        if (numChannels <= 0 || numSamples <= 0) return;
        if (numActive_ == 0)
        {
            // Nothing in the path: the input node still has to remember what went by.
            for (int ch = 0; ch < numChannels; ++ch)
            {
                const T* data = buffer.getChannel(ch);
                auto& input = node_[static_cast<size_t>(ch)][0];
                if (numSamples >= 2) input.h2 = static_cast<double>(data[numSamples - 2]);
                else input.h2 = input.h1;
                input.h1 = static_cast<double>(data[numSamples - 1]);
            }
            advanceRamps(numSamples);
            return;
        }

        DenormalGuard guard;
        int pos = 0;
        while (pos < numSamples)
        {
            int length = std::min(kChunk, numSamples - pos);
            if (ramping_)
                for (int k = 0; k < numActive_; ++k)
                {
                    const int left = rampLeft_[static_cast<size_t>(active_[static_cast<size_t>(k)])];
                    if (left > 0) length = std::min(length, left);
                }

            int ch = 0;
            if constexpr (kPairChannels)
                for (; ch + 1 < numChannels; ch += 2)
                    runChunk<simd::kVecDoubleNarrow>(buffer.getChannel(ch) + pos, buffer.getChannel(ch + 1) + pos,
                                                     length, ch, ch + 1);
            for (; ch < numChannels; ++ch)
                runChunk<1>(buffer.getChannel(ch) + pos, nullptr, length, ch, ch);

            advanceRamps(length);
            pos += length;
        }

        for (int ch = 0; ch < numChannels; ++ch)
            recover(ch);
    }

    /**
     * @brief One sample of one channel through the cascade, in double.
     *
     * Uses the coefficients as they are: ramps advance in processBlock(), or
     * by the caller's advanceRamps() once per frame.
     *
     * @pre channel in [0, MaxChannels).
     * @param input   Input sample.
     * @param channel Channel index.
     * @return Output of the last enabled section.
     */
    [[nodiscard]] double processSample(double input, int channel) noexcept
    {
        assert(channel >= 0 && channel < MaxChannels && "Channel index out of bounds");
        auto& nodes = node_[static_cast<size_t>(channel)];
        double u = input;
        double u1 = nodes[0].h1, u2 = nodes[0].h2;
        nodes[0].h2 = nodes[0].h1;
        nodes[0].h1 = input;
        for (int k = 0; k < numActive_; ++k)
        {
            const auto s = static_cast<size_t>(active_[static_cast<size_t>(k)]);
            const BiquadCoeffs& c = current_[s];
            auto& out = nodes[s + 1];
            const double p = (c.b1 * u1 + c.b2 * u2) - c.a2 * out.h2;
            const double y = (p + c.b0 * u) - c.a1 * out.h1;
            u1 = out.h1;
            u2 = out.h2;
            out.h2 = out.h1;
            out.h1 = y;
            u = y;
        }
        return u;
    }

    /**
     * @brief Moves every ramp on by a number of samples (for processSample() callers).
     * @param samples Samples gone by; a ramp with fewer left lands on its target.
     */
    void advanceRamps(int samples) noexcept
    {
        if (!ramping_ || samples <= 0) return;
        for (int section = 0; section < MaxSections; ++section)
        {
            const auto s = static_cast<size_t>(section);
            if (rampLeft_[s] <= 0) continue;
            if (samples >= rampLeft_[s])
            {
                current_[s] = target_[s];
                rampLeft_[s] = 0;
                continue;
            }
            rampLeft_[s] -= samples;
            if (moved_[s])
            {
                // The loop that played the section moved it, sample by sample.
                current_[s] = next_[s];
            }
            else
            {
                // Out of the path, or played one sample at a time: it moves here.
                const double count = static_cast<double>(samples);
                current_[s].b0 += step_[s].b0 * count;
                current_[s].b1 += step_[s].b1 * count;
                current_[s].b2 += step_[s].b2 * count;
                current_[s].a1 += step_[s].a1 * count;
                current_[s].a2 += step_[s].a2 * count;
            }
        }
        moved_.fill(false);
        updateRamping();
    }

private:
    static constexpr int kChunk = 64;
    static constexpr int kMaxGroups = (MaxSections + kGroup - 1) / kGroup > 0 ? (MaxSections + kGroup - 1) / kGroup : 1;
    static constexpr bool kPairChannels = simd::kVecDoubleNarrow == 2 && sizeof(void*) == 8;

    struct Node
    {
        double h1 = 0.0;   ///< The sample before the current one.
        double h2 = 0.0;   ///< The one before that.
    };

    /** @brief One or two channels as the lanes of a vector of W doubles. */
    template <int W>
    struct Lanes
    {
        using O = simd::Vec<double, W>;
        using V = typename O::V;
        static V pair(double first, double second) noexcept
        {
            if constexpr (W == 1) { (void)second; return first; }
            else return O::set2(first, second);
        }
        static double first(V v) noexcept
        {
            if constexpr (W == 1) return v;
            else return O::lane0(v);
        }
        static double second(V v) noexcept
        {
            if constexpr (W == 1) return v;
            else return O::lane1(v);
        }
    };

    /** @brief A chunk of one channel (W = 1) or of a pair, through every group. */
    template <int W>
    void runChunk(T* left, T* right, int length, int chLeft, int chRight) noexcept
    {
        using L = Lanes<W>;
        using V = typename L::V;
        V io[kChunk];
        if constexpr (W == 1)
        {
            (void)right;
            for (int i = 0; i < length; ++i) io[i] = static_cast<double>(left[i]);
        }
        else
        {
            for (int i = 0; i < length; ++i)
                io[i] = L::pair(static_cast<double>(left[i]), static_cast<double>(right[i]));
        }

        auto& nodesLeft = node_[static_cast<size_t>(chLeft)];
        auto& nodesRight = node_[static_cast<size_t>(chRight)];

        // Each group reads the history its input node had BEFORE this chunk;
        // the group ahead of it leaves that node at the end of the chunk.
        V startU1[kMaxGroups], startU2[kMaxGroups];
        for (int first = 0, group = 0; first < numActive_; first += kGroup, ++group)
        {
            const auto node = static_cast<size_t>(input_[static_cast<size_t>(active_[static_cast<size_t>(first)])]);
            startU1[group] = L::pair(nodesLeft[node].h1, nodesRight[node].h1);
            startU2[group] = L::pair(nodesLeft[node].h2, nodesRight[node].h2);
        }

        for (int first = 0, group = 0; first < numActive_; first += kGroup, ++group)
        {
            const int count = std::min(kGroup, numActive_ - first);
            bool moving = false;
            for (int k = 0; k < count; ++k)
                moving = moving || rampLeft_[static_cast<size_t>(active_[static_cast<size_t>(first + k)])] > 0;
            switch (count)
            {
                case 1: runGroup<1, W>(io, length, first, moving, startU1[group], startU2[group], nodesLeft, nodesRight); break;
                case 2: runGroup<2, W>(io, length, first, moving, startU1[group], startU2[group], nodesLeft, nodesRight); break;
                case 3: runGroup<3, W>(io, length, first, moving, startU1[group], startU2[group], nodesLeft, nodesRight); break;
                default: runGroup<4, W>(io, length, first, moving, startU1[group], startU2[group], nodesLeft, nodesRight); break;
            }
        }

        for (int i = 0; i < length; ++i)
        {
            left[i] = static_cast<T>(L::first(io[i]));
            if constexpr (W != 1) right[i] = static_cast<T>(L::second(io[i]));
        }
    }

    /**
     * @brief M sections of the path over a chunk, all inside one loop.
     *
     * Each section computes the part of its output that only needs past
     * samples first, then adds its input and takes away a1 times its last
     * output: the step from one sample to the next is one product and one
     * difference, and the sections overlap in the processor.
     */
    template <int M, int W>
    void runGroup(typename Lanes<W>::V* io, int length, int first, bool moving,
                  typename Lanes<W>::V u1, typename Lanes<W>::V u2,
                  std::array<Node, MaxSections + 1>& nodesLeft,
                  std::array<Node, MaxSections + 1>& nodesRight) noexcept
    {
        using L = Lanes<W>;
        using O = typename L::O;
        using V = typename L::V;

        size_t section[M];
        V b0[M], b1[M], b2[M], a1[M], a2[M];
        V y1[M], y2[M];
        for (int k = 0; k < M; ++k)
        {
            section[k] = static_cast<size_t>(active_[static_cast<size_t>(first + k)]);
            const BiquadCoeffs& c = current_[section[k]];
            b0[k] = O::set1(c.b0); b1[k] = O::set1(c.b1); b2[k] = O::set1(c.b2);
            a1[k] = O::set1(c.a1); a2[k] = O::set1(c.a2);
            y1[k] = L::pair(nodesLeft[section[k] + 1].h1, nodesRight[section[k] + 1].h1);
            y2[k] = L::pair(nodesLeft[section[k] + 1].h2, nodesRight[section[k] + 1].h2);
        }

        if (!moving)
        {
            for (int i = 0; i < length; ++i)
            {
                V u = io[i];
                V p1 = u1, p2 = u2;
                u2 = u1;
                u1 = u;
                for (int k = 0; k < M; ++k)
                {
                    const V past = O::sub(O::add(O::mul(b1[k], p1), O::mul(b2[k], p2)), O::mul(a2[k], y2[k]));
                    const V y = O::sub(O::add(past, O::mul(b0[k], u)), O::mul(a1[k], y1[k]));
                    p1 = y1[k];
                    p2 = y2[k];
                    y2[k] = y1[k];
                    y1[k] = y;
                    u = y;
                }
                io[i] = u;
            }
        }
        else
        {
            V s0[M], s1[M], s2[M], t1[M], t2[M];
            for (int k = 0; k < M; ++k)
            {
                const bool ramps = rampLeft_[section[k]] > 0;
                const BiquadCoeffs& d = step_[section[k]];
                s0[k] = O::set1(ramps ? d.b0 : 0.0); s1[k] = O::set1(ramps ? d.b1 : 0.0);
                s2[k] = O::set1(ramps ? d.b2 : 0.0); t1[k] = O::set1(ramps ? d.a1 : 0.0);
                t2[k] = O::set1(ramps ? d.a2 : 0.0);
            }
            for (int i = 0; i < length; ++i)
            {
                V u = io[i];
                V p1 = u1, p2 = u2;
                u2 = u1;
                u1 = u;
                for (int k = 0; k < M; ++k)
                {
                    b0[k] = O::add(b0[k], s0[k]); b1[k] = O::add(b1[k], s1[k]); b2[k] = O::add(b2[k], s2[k]);
                    a1[k] = O::add(a1[k], t1[k]); a2[k] = O::add(a2[k], t2[k]);
                    const V past = O::sub(O::add(O::mul(b1[k], p1), O::mul(b2[k], p2)), O::mul(a2[k], y2[k]));
                    const V y = O::sub(O::add(past, O::mul(b0[k], u)), O::mul(a1[k], y1[k]));
                    p1 = y1[k];
                    p2 = y2[k];
                    y2[k] = y1[k];
                    y1[k] = y;
                    u = y;
                }
                io[i] = u;
            }
            // Where the ramp has brought the coefficients: every channel walks
            // the same steps, so the last one to run leaves the same values.
            for (int k = 0; k < M; ++k)
            {
                if (rampLeft_[section[k]] <= 0) continue;
                next_[section[k]].b0 = L::first(b0[k]); next_[section[k]].b1 = L::first(b1[k]);
                next_[section[k]].b2 = L::first(b2[k]); next_[section[k]].a1 = L::first(a1[k]);
                next_[section[k]].a2 = L::first(a2[k]);
                moved_[section[k]] = true;
            }
        }

        // The input node of the cascade is this loop's to keep; the input of
        // a later group is the output node the group ahead has just written.
        if (first == 0)
        {
            nodesLeft[0].h1 = L::first(u1);
            nodesLeft[0].h2 = L::first(u2);
        }
        for (int k = 0; k < M; ++k)
        {
            nodesLeft[section[k] + 1].h1 = L::first(y1[k]);
            nodesLeft[section[k] + 1].h2 = L::first(y2[k]);
        }
        if constexpr (W != 1)
        {
            if (first == 0)
            {
                nodesRight[0].h1 = L::second(u1);
                nodesRight[0].h2 = L::second(u2);
            }
            for (int k = 0; k < M; ++k)
            {
                nodesRight[section[k] + 1].h1 = L::second(y1[k]);
                nodesRight[section[k] + 1].h2 = L::second(y2[k]);
            }
        }
    }

    /** @brief The sections in the path, in order, and the node each one reads. */
    void rebuildRoute() noexcept
    {
        numActive_ = 0;
        int previous = 0;
        for (int s = 0; s < numSections_; ++s)
        {
            if (!enabled_[static_cast<size_t>(s)]) continue;
            active_[static_cast<size_t>(numActive_++)] = s;
            input_[static_cast<size_t>(s)] = previous;
            previous = s + 1;
        }
    }

    /** @brief Gives a section that enters the path the history of its input. */
    void seedFromInput(int section) noexcept
    {
        const auto from = static_cast<size_t>(input_[static_cast<size_t>(section)]);
        for (auto& channel : node_)
            channel[static_cast<size_t>(section) + 1] = channel[from];
    }

    void updateRamping() noexcept
    {
        ramping_ = false;
        for (int left : rampLeft_) ramping_ = ramping_ || left > 0;
    }

    /** @brief Clears a channel whose memory a non-finite sample has reached. */
    void recover(int channel) noexcept
    {
        auto& nodes = node_[static_cast<size_t>(channel)];
        double sum = nodes[0].h1 + nodes[0].h2;
        for (int k = 0; k < numActive_; ++k)
        {
            const Node& each = nodes[static_cast<size_t>(active_[static_cast<size_t>(k)]) + 1];
            sum += each.h1 + each.h2;
        }
        if (!std::isfinite(sum))
            for (auto& each : nodes) each = {};
    }

    std::array<BiquadCoeffs, MaxSections> current_ {};
    std::array<BiquadCoeffs, MaxSections> target_ {};
    std::array<BiquadCoeffs, MaxSections> step_ {};
    std::array<BiquadCoeffs, MaxSections> next_ {};      ///< Coefficients a ramping loop arrived at.
    std::array<bool, MaxSections> moved_ {};
    std::array<int, MaxSections> rampLeft_ {};
    std::array<bool, MaxSections> enabled_ {};
    std::array<int, MaxSections> active_ {};             ///< Sections in the path, in order.
    std::array<int, MaxSections> input_ {};              ///< Node each section in the path reads.
    int numSections_ = 0;
    int numActive_ = 0;
    bool ramping_ = false;

    std::array<std::array<Node, MaxSections + 1>, MaxChannels> node_ {};
};

} // namespace dspark
