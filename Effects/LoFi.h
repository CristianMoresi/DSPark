// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file LoFi.h
 * @brief The lo-fi family in one processor: grains that smear and drop the
 *        pitch, a fold with its level taken back, a lower sample rate and
 *        fewer bits.
 *
 * Each stage is off at its default and costs nothing while off; with every
 * stage off and the mix at 1 the output is the input, to the bit. In order:
 *
 * - **Grains** (setSmear(), setPitch()): the first two channels run through
 *   a GranularProcessor of 30 ms grains at 60 per second, whose read
 *   position wanders by the smear and whose rate follows the pitch. The
 *   grains read the history with a windowed sinc, stretched when they play
 *   faster.
 * - **Fold** (setFold()): sin(g x) with g = (pi / 2) times the fold gain, so
 *   that at 0 dB a full-scale peak just reaches the top of the sine. The
 *   output is scaled so that a sine at -12 dBFS keeps its level whatever the
 *   fold: the mean square of sin(g a sin t) over a period is worked out
 *   whenever the fold moves.
 * - **Rate** (setRate()): each sample is held until the next tick of a
 *   clock at the rate asked for (a fractional phase, so any rate holds on
 *   average). Over it what is above half the rate folds back, as in the
 *   machines this is after; setRateFilter(true) puts a 48 dB per octave
 *   low-pass at 0.45 of the rate in front of the hold instead.
 * - **Bits** (setBits()): the signal is rounded to 2^(bits - 1) steps a
 *   side (bits from 1 to 24, fractional ones too), with triangular dither of
 *   one step when setDither(true).
 *
 * Then the dry/wet mix and an output gain, both ramped over 20 ms. There is
 * no latency at any setting.
 *
 * Threading: setters and getters from any thread (values that are not
 * finite are ignored). prepare() on the setup thread (it allocates);
 * processBlock() and reset() on the audio thread; getState() and setState()
 * on the setup thread.
 *
 * Dependencies: GranularProcessor.h, Core/AudioBuffer.h, Core/AudioSpec.h,
 * Core/BiquadCascade.h, Core/DspMath.h, Core/FilterDesign.h, Core/StateBlob.h.
 */

#include "GranularProcessor.h"
#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/BiquadCascade.h"
#include "../Core/DspMath.h"
#include "../Core/FilterDesign.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class LoFi
 * @brief Grains, fold, sample rate and bit depth, each off by default.
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class LoFi
{
public:
    static constexpr int kMaxChannels = 16;

    LoFi() : grains_(std::make_unique<GranularProcessor<T>>()) {}

    /** @brief Prepares every stage. Invalid specs are ignored; allocates (setup thread). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        spec_ = spec;
        grains_->setGrainSize(T(30));
        grains_->setDensity(T(60));
        grains_->setSpread(T(0));
        grains_->setPitchJitter(T(0));
        grains_->setMix(T(1));
        grains_->prepare(spec, 1.0);
        if (!filter_) filter_ = std::make_unique<Filter>();
        filter_->setNumSections(kFilterSections);
        fit_.prepare(spec.sampleRate);
        rampStep_ = 1.0 / std::max(1.0, 0.02 * spec.sampleRate);
        dry_.resize(std::min(spec.numChannels, kMaxChannels), spec.maxBlockSize);
        prepared_ = true;
        reset();
    }

    /** @brief Clears every stage's memory. */
    void reset() noexcept
    {
        if (!prepared_) return;
        grains_->reset();
        filter_->reset();
        designedRate_ = -1.0;
        held_.fill(T(0));
        phase_ = 1.0;
        mixNow_ = static_cast<double>(mix_.load(std::memory_order_relaxed));
        outputNow_ = std::pow(10.0, static_cast<double>(outputDb_.load(std::memory_order_relaxed)) / 20.0);
        foldNow_ = foldGainFor(static_cast<double>(foldDb_.load(std::memory_order_relaxed)));
        foldScaleNow_ = foldScaleFor(foldNow_);
        rng_ = 0x9E3779B9u;
    }

    /** @brief Processes a block in place. Pass-through until prepare() succeeds. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!prepared_) return;
        const int nS = buffer.getNumSamples();
        const int maxBlock = dry_.getNumSamples();
        for (int off = 0; off < nS; off += maxBlock)
            processChunk(buffer.getSubView(off, std::min(maxBlock, nS - off)));
    }

    // -- Settings (any thread) ---------------------------------------------------

    /** @brief How far the grains' read position wanders, 0 to 1 (0, the default, with no pitch: no grains). */
    void setSmear(T amount) noexcept { store(smear_, amount, T(0), T(1)); }
    /** @brief Pitch of the grains in semitones, -24 to +12 (0, the default, with no smear: no grains). */
    void setPitch(T semitones) noexcept { store(pitch_, semitones, T(-24), T(12)); }
    /** @brief Fold gain in dB, 0 to 36; 0, the default, is off (see the file notes). */
    void setFold(T dB) noexcept { store(foldDb_, dB, T(0), T(36)); }
    /** @brief Sample rate of the hold in hertz, from 100; at or above the stream's rate, the default, it is off. */
    void setRate(T hz) noexcept
    {
        if (!std::isfinite(hz)) return;
        rate_.store(std::max(hz, T(100)), std::memory_order_relaxed);
    }
    /** @brief A 48 dB per octave low-pass at 0.45 of the rate in front of the hold (off by default). */
    void setRateFilter(bool on) noexcept { rateFilter_.store(on, std::memory_order_relaxed); }
    /** @brief Bit depth, 1 to 24 (fractional values too); 24, the default, is off. */
    void setBits(T bits) noexcept { store(bits_, bits, T(1), T(24)); }
    /** @brief Triangular dither of one step before the rounding (off by default). */
    void setDither(bool on) noexcept { dither_.store(on, std::memory_order_relaxed); }
    /** @brief Dry/wet mix, 0 to 1 (default 1); glides over 20 ms. */
    void setMix(T mix) noexcept { store(mix_, mix, T(0), T(1)); }
    /** @brief Output gain in dB, -24 to +12 (default 0); glides over 20 ms. */
    void setOutput(T dB) noexcept { store(outputDb_, dB, T(-24), T(12)); }

    [[nodiscard]] T getSmear() const noexcept { return smear_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getPitch() const noexcept { return pitch_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getFold() const noexcept { return foldDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRate() const noexcept { return rate_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isRateFilter() const noexcept { return rateFilter_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBits() const noexcept { return bits_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool isDither() const noexcept { return dither_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getOutput() const noexcept { return outputDb_.load(std::memory_order_relaxed); }

    /** @brief No latency at any setting. */
    [[nodiscard]] static constexpr int getLatency() noexcept { return 0; }

    /** @brief Serializes the settings. */
    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("LOFI"), 1);
        w.write("smear", static_cast<float>(getSmear()));
        w.write("pitch", static_cast<float>(getPitch()));
        w.write("fold", static_cast<float>(getFold()));
        w.write("rate", static_cast<float>(getRate()));
        w.write("rateFilter", isRateFilter());
        w.write("bits", static_cast<float>(getBits()));
        w.write("dither", isDither());
        w.write("mix", static_cast<float>(getMix()));
        w.write("output", static_cast<float>(getOutput()));
        return w.blob();
    }

    /** @brief Restores what getState() wrote. */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("LOFI")) return false;
        setSmear(static_cast<T>(r.read("smear", 0.0f)));
        setPitch(static_cast<T>(r.read("pitch", 0.0f)));
        setFold(static_cast<T>(r.read("fold", 0.0f)));
        setRate(static_cast<T>(r.read("rate", 1e9f)));
        setRateFilter(r.read("rateFilter", false));
        setBits(static_cast<T>(r.read("bits", 24.0f)));
        setDither(r.read("dither", false));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setOutput(static_cast<T>(r.read("output", 0.0f)));
        return true;
    }

private:
    static constexpr int kFilterSections = 4;
    using Filter = BiquadCascade<T, kFilterSections, kMaxChannels>;

    static void store(std::atomic<T>& field, T value, T lo, T hi) noexcept
    {
        if (std::isfinite(value)) field.store(std::clamp(value, lo, hi), std::memory_order_relaxed);
    }

    /** The argument scale of the sine for a fold in dB (0 when the fold is off). */
    [[nodiscard]] static double foldGainFor(double dB) noexcept
    {
        return dB > 0.0 ? 0.5 * std::numbers::pi * std::pow(10.0, dB / 20.0) : 0.0;
    }

    /**
     * The output scale that keeps a sine at -12 dBFS at its level through
     * sin(g x): its mean square in, over the mean of sin^2(g a sin t) out,
     * the latter by the trapezoid rule over one period (exact to rounding
     * for this smooth periodic integrand at 256 points up to the largest g).
     */
    [[nodiscard]] static double foldScaleFor(double g) noexcept
    {
        if (g <= 0.0) return 1.0;
        constexpr double a = 0.25;
        constexpr int n = 256;
        double sum = 0.0;
        for (int i = 0; i < n; ++i)
        {
            const double s = std::sin(g * a * std::sin(2.0 * std::numbers::pi * i / n));
            sum += s * s;
        }
        const double out = sum / n;
        return out > 1e-12 ? std::sqrt(0.5 * a * a / out) : 1.0;
    }

    /** Triangular dither in [-1, 1) (two uniforms). */
    [[nodiscard]] double triangular() noexcept
    {
        const auto uniform = [this]() noexcept {
            rng_ ^= rng_ << 13;
            rng_ ^= rng_ >> 17;
            rng_ ^= rng_ << 5;
            return static_cast<double>(rng_) * (1.0 / 4294967296.0);
        };
        return uniform() - uniform();
    }

    void processChunk(AudioBufferView<T> buffer) noexcept
    {
        const int nCh = std::min(buffer.getNumChannels(), kMaxChannels);
        const int nS = std::min(buffer.getNumSamples(), dry_.getNumSamples());
        if (nCh <= 0 || nS <= 0) return;
        const int dryCh = std::min(nCh, dry_.getNumChannels());

        const double mixTarget = static_cast<double>(mix_.load(std::memory_order_relaxed));
        const double outputTarget = std::pow(10.0, static_cast<double>(outputDb_.load(std::memory_order_relaxed)) / 20.0);
        const bool mixing = mixTarget < 1.0 || mixNow_ < 1.0;
        if (mixing)
            for (int ch = 0; ch < dryCh; ++ch)
                std::copy_n(buffer.getChannel(ch), nS, dry_.getChannel(ch));

        // Grains.
        const T smear = smear_.load(std::memory_order_relaxed);
        const T pitch = pitch_.load(std::memory_order_relaxed);
        const bool grains = smear > T(0) || pitch != T(0);
        if (grains)
        {
            if (!grainsOn_) grains_->reset();
            grains_->setJitter(smear);
            grains_->setPitch(pitch);
            grains_->processBlock(buffer.getSubView(0, nS));
        }
        grainsOn_ = grains;

        // Fold, with the gain and its scale on a ramp across the block.
        const double foldTarget = foldGainFor(static_cast<double>(foldDb_.load(std::memory_order_relaxed)));
        if (foldTarget > 0.0 || foldNow_ > 0.0)
        {
            const double scaleTarget = (foldTarget != foldNow_) ? foldScaleFor(foldTarget) : foldScaleNow_;
            const double gStep = (foldTarget - foldNow_) / nS, sStep = (scaleTarget - foldScaleNow_) / nS;
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* x = buffer.getChannel(ch);
                for (int i = 0; i < nS; ++i)
                {
                    const double g = foldNow_ + gStep * (i + 1), s = foldScaleNow_ + sStep * (i + 1);
                    x[i] = static_cast<T>(g > 0.0 ? std::sin(g * static_cast<double>(x[i])) * s : static_cast<double>(x[i]));
                }
            }
            foldNow_ = foldTarget;
            foldScaleNow_ = scaleTarget;
        }

        // Rate: a held sample, with the low-pass in front when asked.
        const double rate = static_cast<double>(rate_.load(std::memory_order_relaxed));
        if (rate < spec_.sampleRate)
        {
            if (rateFilter_.load(std::memory_order_relaxed))
            {
                if (rate != designedRate_)
                {
                    BiquadCoeffs sections[kFilterSections];
                    const int count = FilterDesign::digital(FilterDesign::highCut(std::min(0.45 * rate, 0.45 * spec_.sampleRate), 48.0),
                                                            fit_, sections, kFilterSections);
                    for (int s = 0; s < kFilterSections; ++s)
                        filter_->setCoeffs(s, s < count ? sections[s] : BiquadCoeffs {});
                    designedRate_ = rate;
                }
                filter_->processBlock(buffer.getSubView(0, nS));
            }
            const double step = rate / spec_.sampleRate;
            for (int i = 0; i < nS; ++i)
            {
                phase_ += step;
                const bool tick = phase_ >= 1.0;
                if (tick) phase_ -= std::floor(phase_);
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T& x = buffer.getChannel(ch)[i];
                    if (tick) held_[static_cast<size_t>(ch)] = x;
                    x = held_[static_cast<size_t>(ch)];
                }
            }
        }

        // Bits.
        const double bits = static_cast<double>(bits_.load(std::memory_order_relaxed));
        if (bits < 24.0)
        {
            const double steps = std::exp2(bits - 1.0);
            const bool dither = dither_.load(std::memory_order_relaxed);
            for (int i = 0; i < nS; ++i)
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T& x = buffer.getChannel(ch)[i];
                    const double noise = dither ? triangular() : 0.0;
                    x = static_cast<T>(std::round(static_cast<double>(x) * steps + noise) / steps);
                }
        }

        // Mix and output.
        if (mixing || outputTarget != 1.0 || outputNow_ != 1.0)
        {
            double mix = mixNow_, gain = outputNow_;
            for (int i = 0; i < nS; ++i)
            {
                mix += std::clamp(mixTarget - mix, -rampStep_, rampStep_);
                gain += (outputTarget - gain) * rampStep_ * 4.0;
                if (std::abs(outputTarget - gain) < 1e-9) gain = outputTarget;
                for (int ch = 0; ch < nCh; ++ch)
                {
                    T& x = buffer.getChannel(ch)[i];
                    const double wet = static_cast<double>(x);
                    const double dry = (mixing && ch < dryCh) ? static_cast<double>(dry_.getChannel(ch)[i]) : wet;
                    x = static_cast<T>((dry + mix * (wet - dry)) * gain);
                }
            }
            mixNow_ = mix;
            outputNow_ = gain;
        }
    }

    std::unique_ptr<GranularProcessor<T>> grains_;
    std::unique_ptr<Filter> filter_;
    FilterDesign::Band fit_;
    AudioSpec spec_ {};
    AudioBuffer<T> dry_;
    bool prepared_ = false;

    std::atomic<T> smear_ { T(0) }, pitch_ { T(0) }, foldDb_ { T(0) }, rate_ { T(1e9) };
    std::atomic<T> bits_ { T(24) }, mix_ { T(1) }, outputDb_ { T(0) };
    std::atomic<bool> rateFilter_ { false }, dither_ { false };

    // Audio thread.
    bool grainsOn_ = false;
    double designedRate_ = -1.0;
    std::array<T, kMaxChannels> held_ {};
    double phase_ = 1.0;
    double mixNow_ = 1.0, outputNow_ = 1.0, foldNow_ = 0.0, foldScaleNow_ = 1.0, rampStep_ = 1.0 / 960.0;
    uint32_t rng_ = 0x9E3779B9u;
};

} // namespace dspark
