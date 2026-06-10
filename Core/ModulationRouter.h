// DSPark — Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi — MIT License

#pragma once

/**
 * @file ModulationRouter.h
 * @brief Block-rate modulation routing: sources → parameter setters.
 *
 * Generalizes the pitch-tracking-EQ pattern to any parameter without
 * imposing an architecture: a route reads a source (LFO, EnvelopeFollower,
 * PitchFollower, your lambda...), scales it by depth around a base value,
 * smooths it, and applies it through a setter callback — once per block,
 * which is how DSPark parameters are designed to move (their setters are
 * thread-safe and internally smoothed).
 *
 * Routes are configured at setup time (std::function storage may allocate
 * THERE); `update()` in the audio callback is allocation-free — it only
 * invokes the stored callables.
 *
 * @code
 *   router.addRoute(
 *       [&] { return follower.getSmoothedHz(); },              // source
 *       [&](float v) { eq.setFrequency(v); },                  // target
 *       0.9f, 0.0f, 30.0f);                                    // depth, base, smooth ms
 *   // audio callback, after feeding the followers:
 *   router.update(numSamples, sampleRate);
 * @endcode
 *
 * Dependencies: DspMath.h.
 */

#include "DspMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>

namespace dspark {

/**
 * @class ModulationRouter
 * @brief Fixed-capacity source→target router with per-route depth/smoothing.
 *
 * @tparam T         Value type (float or double).
 * @tparam MaxRoutes Route capacity (compile-time, no audio-thread allocation).
 */
template <FloatType T, int MaxRoutes = 16>
class ModulationRouter
{
public:
    /**
     * @brief Adds a route (call from setup/UI threads, not the audio callback).
     *
     * The applied value is `base + source() * depth`, one-pole smoothed.
     *
     * @param source   Returns the current modulation value.
     * @param target   Receives the smoothed, scaled value (a setter).
     * @param depth    Multiplier on the source value.
     * @param base     Constant offset.
     * @param smoothMs One-pole smoothing time (0 = none).
     * @return Route index, or -1 if the router is full.
     */
    int addRoute(std::function<T()> source, std::function<void(T)> target,
                 T depth = T(1), T base = T(0), T smoothMs = T(20))
    {
        if (numRoutes_ >= MaxRoutes) return -1;
        auto& r = routes_[static_cast<size_t>(numRoutes_)];
        r.source = std::move(source);
        r.target = std::move(target);
        r.depth = depth;
        r.base = base;
        r.smoothMs = std::max(smoothMs, T(0));
        r.state = T(0);
        r.primed = false;
        return numRoutes_++;
    }

    /** @brief Changes a route's depth (RT-safe: plain store read by update()). */
    void setDepth(int route, T depth) noexcept
    {
        if (route >= 0 && route < numRoutes_)
            routes_[static_cast<size_t>(route)].depth = depth;
    }

    /** @brief Removes all routes (setup threads only). */
    void clear() noexcept { numRoutes_ = 0; }

    /** @brief Number of configured routes. */
    [[nodiscard]] int getNumRoutes() const noexcept { return numRoutes_; }

    /**
     * @brief Evaluates every route once. Call per audio block.
     * @param numSamples Samples in the block (drives the smoothing rate).
     * @param sampleRate Current sample rate in Hz.
     */
    void update(int numSamples, double sampleRate) noexcept
    {
        const double dt = static_cast<double>(numSamples) / std::max(sampleRate, 1.0);
        for (int i = 0; i < numRoutes_; ++i)
        {
            auto& r = routes_[static_cast<size_t>(i)];
            const T value = r.base + r.source() * r.depth;
            if (!r.primed || r.smoothMs <= T(0))
            {
                r.state = value;     // no sweep-in from zero on the first hit
                r.primed = true;
            }
            else
            {
                const double a = std::exp(-dt * 1000.0 / static_cast<double>(r.smoothMs));
                r.state = static_cast<T>(a * static_cast<double>(r.state)
                                         + (1.0 - a) * static_cast<double>(value));
            }
            r.target(r.state);
        }
    }

private:
    struct Route
    {
        std::function<T()> source;
        std::function<void(T)> target;
        T depth = T(1);
        T base = T(0);
        T smoothMs = T(20);
        T state = T(0);
        bool primed = false;
    };

    std::array<Route, static_cast<size_t>(MaxRoutes)> routes_ {};
    int numRoutes_ = 0;
};

} // namespace dspark
