// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file GranularProcessor.h
 * @brief Real-time granular engine: clouds, freeze, per-grain pitch/spread.
 *
 * A continuously captured stereo ring feeds a cloud of up to 64 overlapping
 * grains. Each grain reads from a jittered position behind the write head
 * with its own playback rate (pitch), Hann envelope (shared table) and
 * equal-power pan; `freeze` stops capture so the cloud orbits the held
 * buffer - the classic texture/freeze instrument.
 *
 * Spawning uses a fractional accumulator (exact average density), grain
 * parameters are randomized per spawn with deterministic LCG state, and
 * everything is preallocated: zero allocation, zero latency.
 *
 * Threading model: parameter setters/getters are std::atomic based and safe
 * from any thread (non-finite values are ignored). prepare() is setup-thread
 * only (allocates; invalid specs are ignored and an unprepared instance
 * passes audio through). reset() belongs to the stream owner.
 * getState()/setState() are setup/UI threads. The dry/wet mix is smoothed
 * linearly over one block (the grain cloud is decorrelated from the dry
 * signal, so an unsmoothed step would click). Channels beyond the first two
 * pass through untouched.
 *
 * Dependencies: Core/AudioSpec.h, Core/AudioBuffer.h, Core/DspMath.h,
 * Core/DenormalGuard.h, Core/StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DenormalGuard.h"
#include "../Core/DspMath.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class GranularProcessor
 * @brief Granular clouds and spectral-freeze textures from live input.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class GranularProcessor
{
public:
    static constexpr int kMaxGrains = 64;

    // -- Lifecycle ---------------------------------------------------------------

    /**
     * @brief Allocates the capture ring and grain pool.
     *
     * Invalid specs (non-positive/non-finite rate, block size or channel
     * count) are ignored: the previous state is kept and an unprepared
     * instance stays pass-through. A non-finite bufferSeconds falls back to
     * the 4 s default.
     *
     * @param spec          Audio environment specification.
     * @param bufferSeconds Capture history length, clamped to [0.5, 16] s
     *                      (default 4 s).
     */
    void prepare(const AudioSpec& spec, double bufferSeconds = 4.0)
    {
        if (!spec.isValid()) return;
        if (!std::isfinite(bufferSeconds)) bufferSeconds = 4.0;
        prepared_.store(false, std::memory_order_relaxed);
        sampleRate_ = spec.sampleRate;
        numChannels_ = std::min(spec.numChannels, 2);

        int size = 1;
        while (size < static_cast<int>(sampleRate_ * std::clamp(bufferSeconds, 0.5, 16.0)))
            size <<= 1;
        ringMask_ = size - 1;
        for (int ch = 0; ch < 2; ++ch)
            ring_[ch].assign(static_cast<size_t>(size), T(0));

        window_.resize(kWindowSize);
        for (int i = 0; i < kWindowSize; ++i)
            window_[static_cast<size_t>(i)] = static_cast<T>(
                0.5 - 0.5 * std::cos(2.0 * 3.14159265358979 * i / (kWindowSize - 1)));

        prepared_.store(true, std::memory_order_relaxed);
        reset();
    }

    /** @brief Kills all grains and clears the capture ring. RT-safe. */
    void reset() noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        for (auto& r : ring_) std::fill(r.begin(), r.end(), T(0));
        for (auto& g : grains_) g.active = false;
        writePos_ = 0;
        spawnAcc_ = 0.0;
        rng_ = 0x9E3779B9u;
        currentMix_ = mix_.load(std::memory_order_relaxed);
    }

    // -- Parameters (thread-safe) ---------------------------------------------------

    /** @brief Grain duration in milliseconds [10, 500] (default 80).
     *  Non-finite values are ignored. */
    void setGrainSize(T ms) noexcept
    {
        if (!std::isfinite(ms)) return;
        grainMs_.store(std::clamp(ms, T(10), T(500)), std::memory_order_relaxed);
    }

    /** @brief Grains per second [1, 200] (default 25). Non-finite values are
     *  ignored. */
    void setDensity(T perSecond) noexcept
    {
        if (!std::isfinite(perSecond)) return;
        density_.store(std::clamp(perSecond, T(1), T(200)), std::memory_order_relaxed);
    }

    /** @brief Position jitter [0, 1]: how far back grains may start (default 0.3).
     *  Non-finite values are ignored. */
    void setJitter(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        jitter_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Per-grain pitch shift in semitones [-24, +24] (default 0).
     *  Non-finite values are ignored. */
    void setPitch(T semitones) noexcept
    {
        if (!std::isfinite(semitones)) return;
        pitchSt_.store(std::clamp(semitones, T(-24), T(24)), std::memory_order_relaxed);
    }

    /** @brief Random pitch spread per grain in semitones [0, 12] (default 0).
     *  Non-finite values are ignored. */
    void setPitchJitter(T semitones) noexcept
    {
        if (!std::isfinite(semitones)) return;
        pitchJitterSt_.store(std::clamp(semitones, T(0), T(12)), std::memory_order_relaxed);
    }

    /** @brief Stereo spread of grain panning [0, 1] (default 0.5).
     *  Non-finite values are ignored. */
    void setSpread(T amount) noexcept
    {
        if (!std::isfinite(amount)) return;
        spread_.store(std::clamp(amount, T(0), T(1)), std::memory_order_relaxed);
    }

    /** @brief Freezes capture: grains keep playing the held history. */
    void setFreeze(bool frozen) noexcept
    {
        freeze_.store(frozen, std::memory_order_relaxed);
    }

    /** @brief Dry/wet mix [0, 1] (default 1); smoothed linearly over one
     *  block. Non-finite values are ignored. */
    void setMix(T mix) noexcept
    {
        if (!std::isfinite(mix)) return;
        mix_.store(std::clamp(mix, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getGrainSize() const noexcept { return grainMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDensity() const noexcept { return density_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getJitter() const noexcept { return jitter_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getPitch() const noexcept { return pitchSt_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getPitchJitter() const noexcept { return pitchJitterSt_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSpread() const noexcept { return spread_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getFreeze() const noexcept { return freeze_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }

    /** @brief Zero: the cloud is parallel to the dry path. */
    [[nodiscard]] static constexpr int getLatency() noexcept { return 0; }

    /** @brief Serializes the parameter state (setup/UI threads; allocates). */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("GRAN"), 1);
        // Explicit float casts: the blob stores float, and with T = double the
        // unqualified write(key, double) would be ambiguous (float/int32/bool).
        w.write("grainMs", static_cast<float>(grainMs_.load(std::memory_order_relaxed)));
        w.write("density", static_cast<float>(density_.load(std::memory_order_relaxed)));
        w.write("jitter", static_cast<float>(jitter_.load(std::memory_order_relaxed)));
        w.write("pitch", static_cast<float>(pitchSt_.load(std::memory_order_relaxed)));
        w.write("pitchJitter", static_cast<float>(pitchJitterSt_.load(std::memory_order_relaxed)));
        w.write("spread", static_cast<float>(spread_.load(std::memory_order_relaxed)));
        w.write("freeze", freeze_.load(std::memory_order_relaxed));
        w.write("mix", static_cast<float>(mix_.load(std::memory_order_relaxed)));
        return w.blob();
    }

    /** @brief Restores parameters from a blob (tolerant; rejects foreign ids). */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("GRAN")) return false;
        setGrainSize(static_cast<T>(r.read("grainMs", 80.0f)));
        setDensity(static_cast<T>(r.read("density", 25.0f)));
        setJitter(static_cast<T>(r.read("jitter", 0.3f)));
        setPitch(static_cast<T>(r.read("pitch", 0.0f)));
        setPitchJitter(static_cast<T>(r.read("pitchJitter", 0.0f)));
        setSpread(static_cast<T>(r.read("spread", 0.5f)));
        setFreeze(r.read("freeze", false));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        return true;
    }

    // -- Processing -------------------------------------------------------------------

    /** @brief Processes a block in-place (1 or 2 channels). Pass-through
     *  until prepare() succeeds. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_.load(std::memory_order_relaxed)) return;
        DenormalGuard guard;

        const int nCh = std::min(buffer.getNumChannels(), numChannels_);
        const int nS = buffer.getNumSamples();
        if (nCh == 0 || nS == 0) return;

        const bool frozen = freeze_.load(std::memory_order_relaxed);
        // Linear per-block mix ramp with exact landing (settled: step == 0
        // and the per-sample value reduces to the constant, bit-identically).
        // The cloud is decorrelated from the dry signal: a hard flip clicked
        // at 11x the steady-state sample delta.
        const T mixTarget = mix_.load(std::memory_order_relaxed);
        const T mixStart  = currentMix_;
        const T mixStep   = (mixTarget - mixStart) / static_cast<T>(nS);
        const double spawnPerSample =
            static_cast<double>(density_.load(std::memory_order_relaxed)) / sampleRate_;

        for (int i = 0; i < nS; ++i)
        {
            // Capture (continues the moment freeze releases).
            if (!frozen)
            {
                ring_[0][static_cast<size_t>(writePos_)] = buffer.getChannel(0)[i];
                ring_[1][static_cast<size_t>(writePos_)] =
                    buffer.getChannel(nCh > 1 ? 1 : 0)[i];
                writePos_ = (writePos_ + 1) & ringMask_;
            }

            // Spawn with a fractional accumulator: exact average density.
            spawnAcc_ += spawnPerSample;
            while (spawnAcc_ >= 1.0)
            {
                spawnAcc_ -= 1.0;
                spawnGrain();
            }

            // Sum the cloud.
            T outL = T(0), outR = T(0);
            for (auto& g : grains_)
            {
                if (!g.active) continue;

                const auto idx = static_cast<int64_t>(g.pos);
                const T frac = static_cast<T>(g.pos - static_cast<double>(idx));
                const int i0 = static_cast<int>(idx) & ringMask_;
                const int i1 = (i0 + 1) & ringMask_;

                const double wPos = g.phase * (kWindowSize - 1);
                const auto wIdx = static_cast<int>(wPos);
                const T wFrac = static_cast<T>(wPos - wIdx);
                const T w = window_[static_cast<size_t>(wIdx)]
                          + (window_[static_cast<size_t>(std::min(wIdx + 1, kWindowSize - 1))]
                             - window_[static_cast<size_t>(wIdx)]) * wFrac;

                const T sL = ring_[0][static_cast<size_t>(i0)]
                           + (ring_[0][static_cast<size_t>(i1)] - ring_[0][static_cast<size_t>(i0)]) * frac;
                const T sR = ring_[1][static_cast<size_t>(i0)]
                           + (ring_[1][static_cast<size_t>(i1)] - ring_[1][static_cast<size_t>(i0)]) * frac;

                outL += sL * w * g.gainL;
                outR += sR * w * g.gainR;

                g.pos += g.rate;
                g.phase += g.phaseInc;
                if (g.phase >= 1.0)
                    g.active = false;
            }

            const T mixVal = mixStart + mixStep * static_cast<T>(i);
            const T dryL = buffer.getChannel(0)[i];
            buffer.getChannel(0)[i] = dryL + (outL - dryL) * mixVal;
            if (nCh > 1)
            {
                const T dryR = buffer.getChannel(1)[i];
                buffer.getChannel(1)[i] = dryR + (outR - dryR) * mixVal;
            }
        }
        currentMix_ = mixTarget;   // exact landing
    }

private:
    static constexpr int kWindowSize = 2048;

    struct Grain
    {
        bool active = false;
        double pos = 0.0;       ///< Ring read position (fractional).
        double rate = 1.0;      ///< Playback increment (pitch).
        double phase = 0.0;     ///< Envelope phase [0, 1).
        double phaseInc = 0.0;
        T gainL = T(0), gainR = T(0);
    };

    [[nodiscard]] double frand() noexcept
    {
        rng_ = rng_ * 1664525u + 1013904223u;
        return static_cast<double>(rng_ >> 8) / 16777216.0;   // [0, 1)
    }

    void spawnGrain() noexcept
    {
        for (auto& g : grains_)
        {
            if (g.active) continue;

            const double sizeMs = static_cast<double>(grainMs_.load(std::memory_order_relaxed));
            const double lenSamples = sizeMs * 0.001 * sampleRate_;
            const double jit = static_cast<double>(jitter_.load(std::memory_order_relaxed));
            const double st = static_cast<double>(pitchSt_.load(std::memory_order_relaxed))
                            + (frand() * 2.0 - 1.0)
                              * static_cast<double>(pitchJitterSt_.load(std::memory_order_relaxed));
            const double rate = std::exp2(st / 12.0);

            // Start far enough back that the grain never overtakes the write
            // head even when pitched up: lead = length * max(rate, 1) + margin.
            const double maxSpan = lenSamples * std::max(rate, 1.0) + 64.0;
            const double jitterSpan = jit * 0.5 * static_cast<double>(ringMask_ + 1 - maxSpan - 64.0);
            const double back = maxSpan + frand() * std::max(jitterSpan, 0.0);
            g.pos = static_cast<double>(writePos_) - back;
            while (g.pos < 0.0) g.pos += static_cast<double>(ringMask_ + 1);

            g.rate = rate;
            g.phase = 0.0;
            g.phaseInc = 1.0 / lenSamples;

            const double spreadAmt = static_cast<double>(spread_.load(std::memory_order_relaxed));
            const double pan = 0.5 + (frand() - 0.5) * spreadAmt;          // [0,1]
            const double a = pan * 1.5707963267948966;                      // pi/2
            // 1/sqrt(density*overlap) keeps the cloud near unity loudness.
            const double overlap = std::max(1.0,
                static_cast<double>(density_.load(std::memory_order_relaxed))
                * sizeMs * 0.001);
            const auto norm = static_cast<T>(1.0 / std::sqrt(overlap));
            g.gainL = static_cast<T>(std::cos(a)) * norm;
            g.gainR = static_cast<T>(std::sin(a)) * norm;
            g.active = true;
            return;
        }
    }

    // -- Members --------------------------------------------------------------------
    double sampleRate_ = 48000.0;
    int numChannels_ = 0;
    std::atomic<bool> prepared_ { false };

    std::array<std::vector<T>, 2> ring_;
    int ringMask_ = 0;
    int writePos_ = 0;

    std::vector<T> window_;
    std::array<Grain, kMaxGrains> grains_;
    double spawnAcc_ = 0.0;
    uint32_t rng_ = 0x9E3779B9u;
    T currentMix_ = T(1);   ///< Audio-thread mix ramp state.

    std::atomic<T> grainMs_ { T(80) };
    std::atomic<T> density_ { T(25) };
    std::atomic<T> jitter_ { T(0.3) };
    std::atomic<T> pitchSt_ { T(0) };
    std::atomic<T> pitchJitterSt_ { T(0) };
    std::atomic<T> spread_ { T(0.5) };
    std::atomic<bool> freeze_ { false };
    std::atomic<T> mix_ { T(1) };
};

} // namespace dspark
