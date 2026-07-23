// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DCBlocker.h
 * @brief Removes DC offset from audio signals with configurable filter order.
 *
 * Order 1 uses a lightweight 1-pole high-pass filter (2 multiplies, 2 additions
 * per sample). Orders 2-10 use cascaded Butterworth biquad high-pass stages
 * with predefined Q values for maximally-flat passband response.
 *
 * Numerical design (why the core is double regardless of the sample type):
 * a DC blocker parks its poles a hair inside the unit circle - 1 - |z| is
 * about pi*fc/(Q*fs), i.e. 1.2e-4 at 5 Hz / 192 kHz and 2.9e-5 at 768 kHz.
 * The denominator evaluated at DC, 1 + a1 + a2 = (2 - 2*cos w0)/a0, is then a
 * cancellation of terms of size 2 landing on 2.7e-8, well under the float
 * resolution of the terms it is made of (1.2e-7). Measured on the float
 * cascade this replaced, at 5 Hz: that sum rounded to 4.77e-7 at 44.1 kHz
 * (6 % off) and to EXACTLY 0 at 192 kHz and 768 kHz, which puts a realised
 * pole right ON the unit circle - from there every rounding error of the
 * recursion is integrated forever instead of decaying and the "DC blocker"
 * sources DC of its own: -0.72 measured at 768 kHz order 2 on a DC-free
 * -6 dBFS tone (output peaking at 1.42), and a realised DC gain of -53.7 at
 * order 4. In double the same cancellation carries an absolute error of about
 * 2e-16, so the poles land within 1e-8 (relative) of the design instead of on
 * the unit circle. Scalar double throughput matches float on x86-64 and
 * ARM64, so the whole filter core runs in double and only the buffer I/O is T
 * - measured slightly cheaper than the float cascade it replaces, because the
 * per-channel state now lives in registers for the length of a block.
 *
 * The DC zero is structural on top of that: the numerator of both topologies
 * is applied as a difference of the input history ((1 - z^-1) for the 1-pole,
 * (1 - z^-1)^2 for the biquads, which is the exact RBJ high-pass numerator
 * since b1 = -2*b0 and b2 = b0). A constant input therefore drives the
 * recursion with an exact zero in IEEE arithmetic, whatever the coefficients
 * round to, so steady DC always decays to true zero.
 *
 * Dependencies: Core/DspMath.h, Core/Biquad.h, Core/AudioSpec.h,
 * Core/AudioBuffer.h, Core/StateBlob.h.
 *
 * Threading: prepare() belongs to the setup thread; the processing calls and
 * reset() to the audio thread. setOrder()/setCutoff() are safe from any
 * thread (atomics, applied lazily at the top of the next processing call);
 * non-finite cutoffs are ignored. getState()/setState() are setup/UI thread.
 */

#include "../Core/DspMath.h"
#include "../Core/Biquad.h"
#include "../Core/AudioSpec.h"
#include "../Core/AudioBuffer.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class DCBlocker
 * @brief DC blocking filter with configurable Butterworth order (1-10).
 *
 * Designed for strict real-time processing: zero allocations and tight scalar
 * inner loops (the recursive filter state rules out SIMD vectorization).
 *
 * The filter core (coefficients, history and recursion) is always double
 * precision, independent of the sample type: see the @file header for the
 * measurements behind that decision. Rejection is therefore rate-independent
 * - a DC-free signal picks up less than 1e-9 of DC and a steady DC input
 * decays to exactly zero at every rate from 44.1 kHz to 768 kHz.
 *
 * @warning Do not modulate the cutoff frequency at audio rates, as updating
 * biquad coefficients triggers trigonometric functions.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DCBlocker
{
public:
    static constexpr int kMaxBiquadStages = 5;
    static constexpr int kMaxChannels = 16;

    /** @brief Builds a filter coherent with the documented defaults (5 Hz @ 48 kHz). */
    DCBlocker() noexcept { forceUpdateCoefficients(); }

    ~DCBlocker() = default; // Non-virtual to avoid vtable injection

    /**
     * @brief Prepares the DC blocker, resetting internal states and precalculating coefficients.
     *
     * @param sampleRate Sample rate in Hz. Non-positive or NaN rates are ignored.
     * @param numChannels Number of audio channels (max 16, default: 2).
     * @param cutoffHz    Cutoff frequency in Hz (clamped to min 1 Hz). Leave it
     *                    out (or pass a non-positive value) to KEEP the cutoff
     *                    currently configured, which is what a host re-activating
     *                    the plugin wants: re-preparing used to silently drop a
     *                    setCutoff() back to the 5 Hz default. A freshly built
     *                    instance still starts at 5 Hz.
     */
    void prepare(double sampleRate, int numChannels = 2, double cutoffHz = -1.0)
    {
        if (!(sampleRate > 0.0)) return;
        sampleRate_ = sampleRate;
        numChannels_ = std::clamp(numChannels, 1, kMaxChannels);

        reset();

        // Use the thread-safe setter to clamp and initialize safely
        if (cutoffHz > 0.0)
            setCutoff(static_cast<T>(cutoffHz));
        forceUpdateCoefficients();
    }

    /** @brief Prepares from AudioSpec (unified API); keeps the configured cutoff. */
    void prepare(const AudioSpec& spec)
    {
        prepare(spec.sampleRate, spec.numChannels);
    }

    /**
     * @brief Sets the filter order (1-10). Thread-safe.
     *
     * - Order 1: efficient 1-pole filter (6 dB/oct).
     * - Order 2-10: cascaded Butterworth biquad HPFs (12*N dB/oct for even orders).
     *
     * @warning Changing order during playback may result in audio clicks, as filter
     * states are not dynamically crossfaded. Best used during prepare() or silence.
     *
     * @note Order 1 = 1-pole (6 dB/oct). Orders >= 2 use floor(order/2) cascaded
     *       Butterworth biquads, so ODD orders behave like the next lower even
     *       order (e.g. 3 == 2, 5 == 4). Prefer 1 or even values for predictable slopes.
     *
     * @param order Filter order (1-10).
     */
    void setOrder(int order) noexcept
    {
        order_.store(std::clamp(order, 1, 10), std::memory_order_relaxed);
    }

    /** @brief Returns the current filter order. */
    [[nodiscard]] int getOrder() const noexcept
    {
        return order_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Requests a cutoff frequency update.
     *
     * Coefficients are recomputed lazily on the next processing call to
     * ensure thread safety, but beware that calculating high-order biquads involves
     * trigonometric operations.
     *
     * @param hz Cutoff frequency in Hz (clamped to min 1.0). Non-finite values
     *           are ignored (std::max(NaN, 1) returns the NaN, which would
     *           poison every coefficient on the next rebuild).
     */
    void setCutoff(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        cutoffHz_.store(std::max(hz, T(1)), std::memory_order_relaxed);
    }

    /** @brief Returns the requested cutoff frequency in Hz. */
    [[nodiscard]] T getCutoff() const noexcept
    {
        return cutoffHz_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Processes an AudioBufferView in-place.
     *
     * Checks for parameter updates and applies the appropriate filter topology
     * across all channels using tight scalar inner loops.
     *
     * @param buffer Audio buffer to process.
     */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        updateCoefficientsIfNeeded();

        // lastOrder_ is the order the live coefficients were designed for.
        // Reloading the atomic here could pick up a setOrder() that landed
        // after the rebuild and enable stages that hold no coefficients yet.
        const int currentOrder = lastOrder_;
        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        if (nS <= 0) return;

        if (currentOrder <= 1)
        {
            for (int ch = 0; ch < nCh; ++ch)
                runOnePole(buffer.getChannel(ch), nS, ch);
        }
        else
        {
            const int numStages = std::clamp(currentOrder / 2, 1, kMaxBiquadStages);
            for (int ch = 0; ch < nCh; ++ch)
                runCascade(buffer.getChannel(ch), nS, ch, numStages);
        }
    }

    /**
     * @brief Processes a block of samples for one channel in-place.
     *
     * @param channel Channel index. Out-of-range indices leave the data
     *                untouched (the per-channel state is bounded to
     *                kMaxChannels).
     * @param data Audio samples (modified in-place).
     * @param numSamples Number of samples to process.
     */
    void processBlock(int channel, T* data, int numSamples) noexcept
    {
        if (channel < 0 || channel >= kMaxChannels) return;
        if (data == nullptr || numSamples <= 0) return;
        updateCoefficientsIfNeeded();
        const int currentOrder = lastOrder_;

        if (currentOrder <= 1)
            runOnePole(data, numSamples, channel);
        else
            runCascade(data, numSamples, channel,
                       std::clamp(currentOrder / 2, 1, kMaxBiquadStages));
    }

    /**
     * @brief Processes a single sample for a given channel.
     *
     * @note For processing multiple samples, prefer processBlock() to avoid
     * overhead: this path re-checks the published parameters on every call so
     * that a per-sample-only caller still sees setOrder()/setCutoff() (two
     * relaxed loads and two compares; a pending change pays the coefficient
     * rebuild on the sample that observes it, same @warning as the block path).
     * Bit-identical to the block paths sample for sample, except that those
     * flush a fully decayed state (below 1e-30, i.e. after seconds of silence)
     * to zero at the end of the block to keep the recursion out of denormal
     * territory; a per-sample caller owns that decision through its own loop.
     *
     * @param channel Channel index. Out-of-range indices return the input
     *                unprocessed.
     * @param input Input sample.
     * @return Sample with DC offset removed.
     */
    [[nodiscard]] T processSample(int channel, T input) noexcept
    {
        if (channel < 0 || channel >= kMaxChannels) return input;
        updateCoefficientsIfNeeded();
        const auto ch = static_cast<size_t>(channel);
        const double x = static_cast<double>(input);

        if (lastOrder_ <= 1)
            return static_cast<T>(stepOnePole(onePoleR_, onePole_[ch], x));

        const int numStages = std::clamp(lastOrder_ / 2, 1, kMaxBiquadStages);
        double v = x;
        for (int s = 0; s < numStages; ++s)
        {
            auto& sec = sections_[static_cast<size_t>(s)];
            v = stepSection(sec.b0, sec.a1, sec.a2, sec.state[ch], v);
        }
        return static_cast<T>(v);
    }

    /**
     * @brief Clears the internal history states to zero.
     */
    void reset() noexcept
    {
        onePole_.fill(OnePoleState{});
        for (auto& s : sections_)
            s.state.fill(SectionState{});
    }


    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("DCBL"), 1);
        w.write("order", order_.load(std::memory_order_relaxed));
        w.write("cutoff", static_cast<float>(cutoffHz_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("DCBL")) return false;
        setOrder(r.read("order", 1));
        setCutoff(static_cast<T>(r.read("cutoff", 5.0f)));
        return true;
    }

protected:
    void updateCoefficientsIfNeeded() noexcept
    {
        // The ORDER is part of the design, not just a stage count: each order
        // has its own Q table row, and a larger order enables stages that may
        // never have received coefficients. Rebuilding only on cutoff changes
        // left a live setOrder() running a broken hybrid (old-Q first stage +
        // identity stages) until the cutoff happened to move.
        const T cutoff  = cutoffHz_.load(std::memory_order_relaxed);
        const int order = order_.load(std::memory_order_relaxed);
        if (cutoff != lastCutoff_ || order != lastOrder_)
        {
            forceUpdateCoefficients(cutoff);
        }
    }

    void forceUpdateCoefficients(T explicitCutoff = T(0)) noexcept
    {
        const T cutoff = explicitCutoff > T(0) ? explicitCutoff : cutoffHz_.load(std::memory_order_relaxed);
        const double fc = static_cast<double>(cutoff);
        // Loaded ONCE: reading order_ again below could pick up a concurrent
        // setOrder() and record a design (lastOrder_) whose stages were never
        // built, which the processing paths would then run as silent sections.
        const int order = order_.load(std::memory_order_relaxed);

        // 1-Pole update
        onePoleR_ = std::exp(-std::numbers::pi * 2.0 * fc / sampleRate_);

        // Biquad updates
        static constexpr float qTable[6][kMaxBiquadStages] = {
            {},                                                // index 0 (unused)
            { 0.7071f },                                       // order 2
            { 0.5412f, 1.3066f },                              // order 4
            { 0.5177f, 0.7071f, 1.9319f },                     // order 6
            { 0.5098f, 0.6013f, 0.8999f, 2.5628f },            // order 8
            { 0.5062f, 0.5612f, 0.7071f, 1.1013f, 3.1962f }    // order 10
        };

        const int tableIdx = std::clamp(order / 2, 1, 5);
        for (int s = 0; s < tableIdx; ++s)
        {
            // Designed in double and used as b0 * (1 - z^-1)^2 / A(z): the RBJ
            // high-pass numerator IS b0 * (1, -2, 1) - exactly so in binary
            // floating point, since halving and doubling are exact and all
            // three share the same 1/a0 factor - which is what lets the DC
            // zero be applied structurally in the inner loop.
            const auto c = BiquadCoeffs<double>::makeHighPass(
                sampleRate_, fc, static_cast<double>(qTable[tableIdx][s]));
            assert(c.b1 == -2.0 * c.b0 && c.b2 == c.b0
                   && "RBJ high-pass numerator is no longer b0 * (1, -2, 1)");

            auto& sec = sections_[static_cast<size_t>(s)];
            sec.b0 = c.b0;
            sec.a1 = c.a1;
            sec.a2 = c.a2;
        }

        lastCutoff_ = cutoff;
        lastOrder_  = order;
    }

    // -- Double-precision filter core -----------------------------------------

    struct OnePoleState { double x1 = 0.0, y1 = 0.0; };
    struct SectionState { double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0; };

    /** @brief Second-order section: b0 * (1 - z^-1)^2 / (1 + a1 z^-1 + a2 z^-2). */
    struct Section
    {
        double b0 = 0.0, a1 = 0.0, a2 = 0.0;
        std::array<SectionState, kMaxChannels> state {};
    };

    /** @brief y[n] = (x[n] - x[n-1]) + R*y[n-1]; the difference zeroes DC exactly. */
    static double stepOnePole(double r, OnePoleState& z, double x) noexcept
    {
        const double y = (x - z.x1) + r * z.y1;
        z.x1 = x;
        z.y1 = y;
        return y;
    }

    /** @brief Direct Form I with the numerator applied as a second difference. */
    static double stepSection(double b0, double a1, double a2, SectionState& z, double x) noexcept
    {
        // (x - x1) - (x1 - x2) is exactly zero for a constant input in IEEE
        // arithmetic, so steady DC never reaches the recursion at all.
        const double d = (x - z.x1) - (z.x1 - z.x2);
        const double y = b0 * d - a1 * z.y1 - a2 * z.y2;
        z.x2 = z.x1; z.x1 = x;
        z.y2 = z.y1; z.y1 = y;
        return y;
    }

    /**
     * @brief Flushes a fully decayed state to zero at the end of a block.
     *
     * A 5 Hz pole holds energy for seconds: without this, a silent passage
     * walks the state down through the denormal range and parks both the
     * recursion and the emitted samples on the CPU's slow path (the float
     * cascade this replaced was measured at 1e-42 after four seconds of DC).
     * The floor sits ~600 dB below full scale and above the smallest normal
     * float, and only fires when the input history is negligible too, so it
     * can never truncate a live signal.
     */
    static constexpr double kDenormalFloor = 1e-30;

    static void flushDenormals(OnePoleState& z) noexcept
    {
        if (std::abs(z.y1) + std::abs(z.x1) < kDenormalFloor)
            z = OnePoleState{};
    }

    static void flushDenormals(SectionState& z) noexcept
    {
        if (std::abs(z.y1) + std::abs(z.y2) + std::abs(z.x1) + std::abs(z.x2) < kDenormalFloor)
            z = SectionState{};
    }

    void runOnePole(T* data, int numSamples, int channel) noexcept
    {
        const double r = onePoleR_;
        OnePoleState z = onePole_[static_cast<size_t>(channel)];
        for (int i = 0; i < numSamples; ++i)
            data[i] = static_cast<T>(stepOnePole(r, z, static_cast<double>(data[i])));
        flushDenormals(z);
        onePole_[static_cast<size_t>(channel)] = z;
    }

    void runCascade(T* data, int numSamples, int channel, int numStages) noexcept
    {
        // Coefficients and history are lifted into locals for the whole block:
        // the intermediate signal then stays in double across the cascade and
        // the compiler is free to keep the state in registers (writing to
        // data[i] would otherwise be assumed to alias the member state).
        double b0[kMaxBiquadStages] {}, a1[kMaxBiquadStages] {}, a2[kMaxBiquadStages] {};
        SectionState z[kMaxBiquadStages] {};
        const auto ch = static_cast<size_t>(channel);
        for (int s = 0; s < numStages; ++s)
        {
            const auto& sec = sections_[static_cast<size_t>(s)];
            b0[s] = sec.b0; a1[s] = sec.a1; a2[s] = sec.a2;
            z[s]  = sec.state[ch];
        }

        for (int i = 0; i < numSamples; ++i)
        {
            double v = static_cast<double>(data[i]);
            for (int s = 0; s < numStages; ++s)
                v = stepSection(b0[s], a1[s], a2[s], z[s], v);
            data[i] = static_cast<T>(v);
        }

        for (int s = 0; s < numStages; ++s)
        {
            flushDenormals(z[s]);
            sections_[static_cast<size_t>(s)].state[ch] = z[s];
        }
    }

    double sampleRate_ = 48000.0;
    int numChannels_ = 2;
    double onePoleR_ = 0.0;   ///< Derived from the documented defaults in the constructor.

    std::atomic<int> order_ { 1 };
    std::atomic<T> cutoffHz_ { T(5) };
    T lastCutoff_ = T(-1);
    int lastOrder_ = -1;

    // Fixed-size per-channel state (scalar loops: no SIMD alignment needed).
    std::array<OnePoleState, kMaxChannels> onePole_ {};
    std::array<Section, kMaxBiquadStages> sections_ {};
};

} // namespace dspark
