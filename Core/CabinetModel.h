// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file CabinetModel.h
 * @brief A loudspeaker cabinet without convolution and without latency: the
 *        colour of the speaker as filters, the box as a few reflections
 *        and two diffusers.
 *
 * What a guitar or bass amplifier sounds like through its speaker is mostly
 * three things: the low resonance of the cone in its box (a bump, and
 * nothing under it), the cone breaking up and then rolling off steeply in
 * the upper midrange, and the short reflections inside and around the box
 * that a microphone picks up with the direct sound. A measured impulse
 * response through a Convolver captures one cabinet and one microphone; this
 * is the cheap and adjustable alternative:
 *
 * - **Colour**: a 12 dB per octave low cut with a resonance at the cone's
 *   frequency, a shallow dip in the low midrange, the breakup peak, and a
 *   24 dB per octave roll-off, each designed from its analog prototype with
 *   FilterDesign and run in one BiquadCascade.
 * - **Box**: six sparse reflections between 0.3 and 3.4 ms with alternating
 *   signs, through two allpass diffusers, added to the direct sound by the
 *   distance of the microphone; the level is kept as the reflections come
 *   in.
 *
 * There is no lookahead and no block: the first output sample is already
 * the direct sound.
 *
 * Settings move while audio runs: the setters publish from any thread, and
 * the audio thread designs the filters in processBlock() and glides them
 * over 20 ms.
 *
 * Threading: setters, getters, writeState() and readState() from any
 * thread. prepare() on the setup thread (it allocates). processBlock() and
 * reset() on the audio thread.
 *
 * Dependencies: AudioBuffer.h, Biquad.h, BiquadCascade.h, DspMath.h,
 *               FilterDesign.h, StateBlob.h.
 */

#include "AudioBuffer.h"
#include "Biquad.h"
#include "BiquadCascade.h"
#include "DspMath.h"
#include "FilterDesign.h"
#include "StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <new>
#include <vector>

namespace dspark {

/**
 * @class CabinetModel
 * @brief A loudspeaker cabinet: colour filters, sparse reflections and two
 *        diffusers, with no latency.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Largest number of channels.
 */
template <FloatType T, int MaxChannels = 16>
class CabinetModel
{
public:
    static constexpr int kTaps = 6;                 ///< Reflections of the box.
    static constexpr int kSections = 5;             ///< Sections of the colour filters.

    /**
     * @brief Prepares the model for a sample rate and a channel count, and
     *        clears it. Invalid values are ignored. Allocates; setup thread.
     */
    void prepare(double sampleRate, int numChannels)
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0) || numChannels <= 0) return;
        fs_ = sampleRate;
        channels_ = std::min(numChannels, MaxChannels);
        if (!cascade_) cascade_.reset(new (std::nothrow) Cascade());
        if (!cascade_) return;
        cascade_->setNumSections(kSections);
        fit_.prepare(sampleRate);

        for (int k = 0; k < kTaps; ++k)
            tapDelay_[static_cast<size_t>(k)] = std::max(1, static_cast<int>(std::lround(kTapMs[k] * 0.001 * fs_)));
        diffuserDelay_[0] = std::max(1, static_cast<int>(std::lround(kDiffuserMs[0] * 0.001 * fs_)));
        diffuserDelay_[1] = std::max(1, static_cast<int>(std::lround(kDiffuserMs[1] * 0.001 * fs_)));
        size_ = 1;
        while (size_ <= tapDelay_[kTaps - 1] + 1 || size_ <= diffuserDelay_[1] + 1) size_ <<= 1;
        lines_.assign(static_cast<size_t>(channels_) * 3u * static_cast<size_t>(size_), T(0));
        rampSamples_ = std::max(1, static_cast<int>(0.02 * fs_));
        settled_ = false;
        distanceNow_ = static_cast<double>(distance_.load(std::memory_order_relaxed));
        dirty_.store(true, std::memory_order_release);
        reset();
    }

    /** @brief Clears the filters and the reflections (settings are kept). */
    void reset() noexcept
    {
        if (cascade_) cascade_->reset();
        std::fill(lines_.begin(), lines_.end(), T(0));
        pos_ = 0;
        distanceNow_ = static_cast<double>(distance_.load(std::memory_order_relaxed));
    }

    /**
     * @brief Size of the box, 0 to 1: the cone's resonance from 140 Hz
     *        (0) down to 70 Hz (1). Default 0.5 (99 Hz).
     */
    void setSize(T size) noexcept
    {
        if (!std::isfinite(size)) return;
        sizeParam_.store(std::clamp(size, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }
    /**
     * @brief Brightness of the speaker, 0 to 1: the roll-off from 3 kHz (0)
     *        up to 7 kHz (1), with the breakup peak at 0.62 of it. Default 0.5.
     */
    void setBrightness(T brightness) noexcept
    {
        if (!std::isfinite(brightness)) return;
        brightness_.store(std::clamp(brightness, T(0), T(1)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }
    /**
     * @brief Distance of the microphone, 0 to 1: how much of the box's
     *        reflections comes with the direct sound (0: none). Default 0.3.
     */
    void setDistance(T distance) noexcept
    {
        if (!std::isfinite(distance)) return;
        distance_.store(std::clamp(distance, T(0), T(1)), std::memory_order_relaxed);
    }

    [[nodiscard]] T getSize() const noexcept { return sizeParam_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getBrightness() const noexcept { return brightness_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getDistance() const noexcept { return distance_.load(std::memory_order_relaxed); }

    /** @brief The cone's resonance for the size set, in hertz. */
    [[nodiscard]] double getResonanceHz() const noexcept { return 140.0 * std::exp2(-static_cast<double>(getSize())); }
    /** @brief The roll-off corner for the brightness set, in hertz. */
    [[nodiscard]] double getRollOffHz() const noexcept
    {
        return 3000.0 * std::pow(7.0 / 3.0, static_cast<double>(getBrightness()));
    }

    /** @brief Writes the settings under keys that start with a prefix. */
    void writeState(StateWriter& w, const char* prefix) const
    {
        char key[48];
        std::snprintf(key, sizeof(key), "%ssize", prefix);
        w.write(key, static_cast<float>(getSize()));
        std::snprintf(key, sizeof(key), "%sbrightness", prefix);
        w.write(key, static_cast<float>(getBrightness()));
        std::snprintf(key, sizeof(key), "%sdistance", prefix);
        w.write(key, static_cast<float>(getDistance()));
    }

    /** @brief Reads what writeState() wrote (missing keys read as the defaults). */
    void readState(const StateReader& r, const char* prefix) noexcept
    {
        char key[48];
        std::snprintf(key, sizeof(key), "%ssize", prefix);
        setSize(static_cast<T>(r.read(key, 0.5f)));
        std::snprintf(key, sizeof(key), "%sbrightness", prefix);
        setBrightness(static_cast<T>(r.read(key, 0.5f)));
        std::snprintf(key, sizeof(key), "%sdistance", prefix);
        setDistance(static_cast<T>(r.read(key, 0.3f)));
    }

    /** @brief Runs a block in place. Channels beyond the prepared count pass through. */
    void processBlock(AudioBufferView<T> buffer) noexcept
    {
        if (!cascade_ || lines_.empty()) return;
        if (dirty_.load(std::memory_order_relaxed) && dirty_.exchange(false, std::memory_order_acquire))
            design();
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        const int nS = buffer.getNumSamples();
        const double distance = static_cast<double>(distance_.load(std::memory_order_relaxed));
        const double distanceStep = (distance - distanceNow_) / static_cast<double>(std::max(nS, 1));
        const int mask = size_ - 1;
        for (int i = 0; i < nS; ++i)
        {
            const double d = distanceNow_ + distanceStep * static_cast<double>(i + 1);
            // Energy of the reflections, so the level stays as they come in.
            const double norm = 1.0 / std::sqrt(1.0 + d * d * kTapEnergy);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* data = buffer.getChannel(ch);
                T* direct = lines_.data() + static_cast<size_t>(ch) * 3u * static_cast<size_t>(size_);
                T* first = direct + size_;
                T* second = first + size_;
                const double x = static_cast<double>(data[i]);
                const double y = cascade_->processSample(std::isfinite(x) ? x : 0.0, ch);
                direct[pos_] = static_cast<T>(y);

                double reflections = 0.0;
                for (int k = 0; k < kTaps; ++k)
                    reflections += kTapGain[k] * static_cast<double>(direct[(pos_ - tapDelay_[static_cast<size_t>(k)]) & mask]);

                // Two Schroeder allpasses: v = x + g w(n - D); out = w(n - D) - g v.
                const double late1 = static_cast<double>(first[(pos_ - diffuserDelay_[0]) & mask]);
                const double v1 = reflections + kDiffuserGain[0] * late1;
                first[pos_] = static_cast<T>(v1);
                const double a1 = late1 - kDiffuserGain[0] * v1;
                const double late2 = static_cast<double>(second[(pos_ - diffuserDelay_[1]) & mask]);
                const double v2 = a1 + kDiffuserGain[1] * late2;
                second[pos_] = static_cast<T>(v2);
                const double a2 = late2 - kDiffuserGain[1] * v2;

                data[i] = static_cast<T>((y + d * a2) * norm);
            }
            cascade_->advanceRamps(1);
            pos_ = (pos_ + 1) & mask;
        }
        distanceNow_ = distance;
    }

private:
    using Cascade = BiquadCascade<T, kSections, MaxChannels>;

    // Reflections of the box: times (ms) and gains, alternating in sign.
    static constexpr double kTapMs[kTaps] = { 0.29, 0.71, 1.13, 1.87, 2.63, 3.41 };
    static constexpr double kTapGain[kTaps] = { 0.45, -0.32, 0.25, -0.18, 0.12, -0.08 };
    static constexpr double kTapEnergy = 0.45 * 0.45 + 0.32 * 0.32 + 0.25 * 0.25 + 0.18 * 0.18 + 0.12 * 0.12 + 0.08 * 0.08;
    static constexpr double kDiffuserMs[2] = { 1.31, 2.27 };
    static constexpr double kDiffuserGain[2] = { 0.5, 0.4 };

    /** Designs the colour and sends it to the cascade (audio thread). */
    void design() noexcept
    {
        const double resonance = getResonanceHz();
        const double rollOff = std::min(getRollOffHz(), 0.45 * fs_);
        AnalogCascade parts[4] = {
            FilterDesign::lowCut(resonance, 12.0, 1.4),
            FilterDesign::bell(450.0, 1.0, -2.0, 1),
            FilterDesign::bell(0.62 * rollOff, 1.5, 4.0, 1),
            FilterDesign::highCut(rollOff, 24.0),
        };
        BiquadCoeffs sections[kSections];
        int count = 0;
        for (const AnalogCascade& part : parts)
            count += FilterDesign::digital(part, fit_, sections + count, kSections - count);
        const int ramp = settled_ ? rampSamples_ : 0;
        for (int s = 0; s < kSections; ++s)
            cascade_->rampCoeffs(s, s < count ? sections[s] : BiquadCoeffs {}, ramp);
        settled_ = true;
    }

    std::atomic<T> sizeParam_ { T(0.5) };
    std::atomic<T> brightness_ { T(0.5) };
    std::atomic<T> distance_ { T(0.3) };
    std::atomic<bool> dirty_ { true };

    // Audio thread.
    std::unique_ptr<Cascade> cascade_;
    FilterDesign::Band fit_;
    std::array<int, kTaps> tapDelay_ {};
    std::array<int, 2> diffuserDelay_ {};
    std::vector<T> lines_;      ///< Per channel: the direct sound, and each diffuser's line.
    double fs_ = 48000.0;
    double distanceNow_ = 0.3;
    int channels_ = 0;
    int size_ = 1;
    int pos_ = 0;
    int rampSamples_ = 960;
    bool settled_ = false;
};

} // namespace dspark
