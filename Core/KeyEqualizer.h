// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file KeyEqualizer.h
 * @brief A parametric equalizer for the key of a dynamics processor.
 *
 * What a compressor, a gate or an expander reacts to is its key: the
 * programme itself or an external signal. Shaping the key decides what the
 * detector weighs (less low end, so that a kick does not pull the whole mix
 * down; a bell on the band that should trigger the reduction) without
 * touching the audio. KeyEqualizer is that filter: kBands bands, each one a
 * low or a high cut of any slope from 6 to 48 dB per octave, a bell, a shelf
 * or a tilt, designed from its analog prototype with FilterDesign (the curves
 * do not shrink towards Nyquist) and run together in one BiquadCascade.
 *
 * Settings move while audio runs. setBand() publishes them from any thread;
 * the audio thread designs the sections in sync() and takes the coefficients
 * there over kRampSeconds. A band switched off (or left at 0 dB) leaves
 * through its own poles and is then no longer computed, so an equalizer with
 * every band off costs one test per sample.
 *
 * The owner drives it in three calls: sync() once per block (or per frame of
 * a per-sample stream), processSample() for every sample of every channel of
 * the key, and advance() with the number of frames gone by.
 *
 * Threading: setBand(), getBand(), getResponse(), writeState() and
 * readState() from any thread. prepare() on the setup thread (it allocates
 * the cascade and never throws: without memory the equalizer stays out of
 * the path). sync(), isActive(), processSample(), advance() and reset() on
 * the audio thread.
 *
 * KeyMonitor is what goes out when the key is listened to: the key late by
 * the processor's latency and held under full scale.
 *
 * Dependencies: Biquad.h, BiquadCascade.h, DspMath.h, FilterDesign.h,
 *               StateBlob.h.
 */

#include "Biquad.h"
#include "BiquadCascade.h"
#include "DspMath.h"
#include "FilterDesign.h"
#include "StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>

namespace dspark {

/**
 * @class KeyEqualizer
 * @brief Six bands of cuts, bells, shelves and tilts over the key of a
 *        dynamics processor.
 *
 * @tparam T           Sample type (float or double). The filters run in
 *                     double either way.
 * @tparam MaxChannels Largest number of key channels.
 */
template <FloatType T, int MaxChannels = 16>
class KeyEqualizer
{
public:
    static constexpr int kBands = 6;                ///< Bands of the equalizer.
    static constexpr int kBandSections = 5;         ///< Most sections one band takes (a 48 dB per octave cut).
    static constexpr int kSections = kBands * kBandSections;
    static constexpr double kRampSeconds = 0.02;    ///< Time the coefficients take to reach a new setting.

    /** @brief What a band does to the key. */
    enum class Shape
    {
        Off,        ///< Out of the path.
        LowCut,     ///< High-pass: frequency, slope and q.
        HighCut,    ///< Low-pass: frequency, slope and q.
        Bell,       ///< Peak or dip: frequency, gainDb and q.
        LowShelf,   ///< gainDb under the frequency; slope 6 to 24 is its transition.
        HighShelf,  ///< gainDb over the frequency; slope 6 to 24 is its transition.
        Tilt        ///< The same slope across the band, 0 dB at the frequency.
    };

    /** @brief The settings of one band. */
    struct Band
    {
        Shape shape = Shape::Off;
        T frequency = T(1000);  ///< Corner, centre or pivot, in hertz.
        T gainDb = T(0);        ///< Bell and shelves, -24 to +24 dB.
        T q = T(0.7071);        ///< Width of the bell; resonance of a cut (0.7071 is Butterworth).
        T slope = T(12);        ///< Decibels per octave: 6 to 48 for the cuts, 6 to 24 for the
                                ///< shelves (in steps of 6), -6 to +6 for the tilt.
    };

    /**
     * @struct Response
     * @brief The sections of a set of bands, to draw the curve they make.
     */
    struct Response
    {
        std::array<BiquadCoeffs, kSections> sections {};
        int count = 0;
        double sampleRate = 48000.0;

        /** @brief Magnitude of the whole equalizer at a frequency, in decibels. */
        [[nodiscard]] double magnitudeDb(double hz) const noexcept
        {
            return FilterDesign::magnitudeDb(sections.data(), count, hz, sampleRate);
        }
    };

    // -- Setup ------------------------------------------------------------------

    /**
     * @brief Prepares the equalizer for a sample rate and clears its memory.
     *
     * Allocates the cascade the first time. The published bands are kept and
     * taken at once (without a ramp) by the next sync(). An invalid rate is
     * ignored.
     *
     * @param sampleRate Sample rate in hertz.
     */
    void prepare(double sampleRate) noexcept
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        if (!cascade_) cascade_.reset(new (std::nothrow) Cascade());
        if (!cascade_) return;
        cascade_->setNumSections(kSections);
        for (int s = 0; s < kSections; ++s)
        {
            cascade_->setCoeffs(s, BiquadCoeffs {});
            cascade_->setSectionEnabled(s, false);
        }
        cascade_->reset();
        leaving_.fill(false);
        fit_.prepare(sampleRate);
        sampleRate_.store(sampleRate, std::memory_order_relaxed);
        rampSamples_ = std::max(1, static_cast<int>(kRampSeconds * sampleRate));
        settled_ = false;
        active_ = false;
        leavingAny_ = false;
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief Clears the memory of the filters (settings and ramps are kept). */
    void reset() noexcept
    {
        if (cascade_) cascade_->reset();
    }

    // -- Settings (any thread) --------------------------------------------------

    /**
     * @brief Publishes the settings of a band.
     *
     * Values out of range are brought into it; a field that is not finite
     * keeps what it had. The audio thread takes the change in its next
     * sync().
     *
     * @param band     Band index in [0, kBands); others are ignored.
     * @param settings The band.
     */
    void setBand(int band, const Band& settings) noexcept
    {
        if (band < 0 || band >= kBands) return;
        Published& p = published_[static_cast<size_t>(band)];
        p.shape.store(std::clamp(static_cast<int>(settings.shape), 0, static_cast<int>(Shape::Tilt)),
                      std::memory_order_relaxed);
        if (std::isfinite(settings.frequency) && settings.frequency > T(0))
            p.frequency.store(std::clamp(settings.frequency, T(10), T(40000)), std::memory_order_relaxed);
        if (std::isfinite(settings.gainDb))
            p.gainDb.store(std::clamp(settings.gainDb, T(-24), T(24)), std::memory_order_relaxed);
        if (std::isfinite(settings.q))
            p.q.store(std::clamp(settings.q, T(0.1), T(20)), std::memory_order_relaxed);
        if (std::isfinite(settings.slope))
            p.slope.store(std::clamp(settings.slope, T(-48), T(48)), std::memory_order_relaxed);
        dirty_.store(true, std::memory_order_release);
    }

    /** @brief The published settings of a band (a band out of range reads as Off). */
    [[nodiscard]] Band getBand(int band) const noexcept
    {
        Band out;
        if (band < 0 || band >= kBands) return out;
        const Published& p = published_[static_cast<size_t>(band)];
        out.shape = static_cast<Shape>(p.shape.load(std::memory_order_relaxed));
        out.frequency = p.frequency.load(std::memory_order_relaxed);
        out.gainDb = p.gainDb.load(std::memory_order_relaxed);
        out.q = p.q.load(std::memory_order_relaxed);
        out.slope = p.slope.load(std::memory_order_relaxed);
        return out;
    }

    /** @brief Whether any published band is in the path. */
    [[nodiscard]] bool hasBands() const noexcept
    {
        for (const Published& p : published_)
            if (p.shape.load(std::memory_order_relaxed) != static_cast<int>(Shape::Off)) return true;
        return false;
    }

    /**
     * @brief The sections of the published bands, designed as the audio
     *        thread designs them, for a display.
     *
     * A design costs a few microseconds per section: build one Response when
     * the settings change and read every frequency from it.
     */
    [[nodiscard]] Response getResponse() const noexcept
    {
        Response out;
        out.sampleRate = sampleRate_.load(std::memory_order_relaxed);
        const FilterDesign::Band fit(out.sampleRate);
        for (int b = 0; b < kBands; ++b)
            out.count += design(getBand(b), fit, out.sections.data() + out.count);
        return out;
    }

    /**
     * @brief Writes the bands under keys that start with a prefix
     *        ("<prefix><band>.shape", ".freq", ".gain", ".q", ".slope").
     */
    void writeState(StateWriter& w, const char* prefix) const
    {
        char key[64];
        for (int b = 0; b < kBands; ++b)
        {
            const Band band = getBand(b);
            std::snprintf(key, sizeof(key), "%s%d.shape", prefix, b);
            w.write(key, static_cast<int32_t>(band.shape));
            std::snprintf(key, sizeof(key), "%s%d.freq", prefix, b);
            w.write(key, static_cast<float>(band.frequency));
            std::snprintf(key, sizeof(key), "%s%d.gain", prefix, b);
            w.write(key, static_cast<float>(band.gainDb));
            std::snprintf(key, sizeof(key), "%s%d.q", prefix, b);
            w.write(key, static_cast<float>(band.q));
            std::snprintf(key, sizeof(key), "%s%d.slope", prefix, b);
            w.write(key, static_cast<float>(band.slope));
        }
    }

    /** @brief Reads the bands writeState() wrote (a band that is missing reads as Off). */
    void readState(const StateReader& r, const char* prefix) noexcept
    {
        char key[64];
        for (int b = 0; b < kBands; ++b)
        {
            Band band;
            std::snprintf(key, sizeof(key), "%s%d.shape", prefix, b);
            band.shape = static_cast<Shape>(std::clamp(static_cast<int>(r.read(key, int32_t(0))), 0,
                                                       static_cast<int>(Shape::Tilt)));
            std::snprintf(key, sizeof(key), "%s%d.freq", prefix, b);
            band.frequency = static_cast<T>(r.read(key, 1000.0f));
            std::snprintf(key, sizeof(key), "%s%d.gain", prefix, b);
            band.gainDb = static_cast<T>(r.read(key, 0.0f));
            std::snprintf(key, sizeof(key), "%s%d.q", prefix, b);
            band.q = static_cast<T>(r.read(key, 0.7071f));
            std::snprintf(key, sizeof(key), "%s%d.slope", prefix, b);
            band.slope = static_cast<T>(r.read(key, 12.0f));
            setBand(b, band);
        }
    }

    // -- Audio thread -----------------------------------------------------------

    /**
     * @brief Takes the settings published since the last call.
     * @return Whether any section is in the path (see isActive()).
     */
    bool sync() noexcept
    {
        if (!cascade_) return false;
        if (dirty_.load(std::memory_order_relaxed) && dirty_.exchange(false, std::memory_order_acquire))
            apply();
        return active_;
    }

    /** @brief Whether any section is in the path: processSample() does nothing otherwise. */
    [[nodiscard]] bool isActive() const noexcept { return active_; }

    /**
     * @brief One sample of one channel of the key through the equalizer.
     *
     * @pre channel in [0, MaxChannels) and a finite input.
     * @param input   Key sample.
     * @param channel Channel index.
     * @return The shaped key (the input itself while no band is in the path).
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (!active_) return input;
        return static_cast<T>(cascade_->processSample(static_cast<double>(input), channel));
    }

    /**
     * @brief Moves the coefficient ramps on and retires the sections that
     *        have finished leaving.
     * @param samples Frames gone by since the last call.
     */
    void advance(int samples) noexcept
    {
        if (!cascade_ || !cascade_->isRamping()) return;
        cascade_->advanceRamps(samples);
        if (!leavingAny_) return;
        bool retired = false;
        for (int s = 0; s < kSections; ++s)
        {
            const auto index = static_cast<size_t>(s);
            if (!leaving_[index] || cascade_->isRamping(s)) continue;
            cascade_->setSectionEnabled(s, false);
            leaving_[index] = false;
            retired = true;
        }
        if (retired) refresh();
    }

private:
    using Cascade = BiquadCascade<T, kSections, MaxChannels>;

    struct Published
    {
        std::atomic<int> shape { 0 };
        std::atomic<T> frequency { T(1000) };
        std::atomic<T> gainDb { T(0) };
        std::atomic<T> q { T(0.7071) };
        std::atomic<T> slope { T(12) };
    };

    /** @brief The digital sections of one band; returns how many it wrote (at most kBandSections). */
    [[nodiscard]] static int design(const Band& band, const FilterDesign::Band& fit, BiquadCoeffs* out) noexcept
    {
        const double hz = std::clamp(static_cast<double>(band.frequency), 10.0, 0.45 * fit.getSampleRate());
        const double gainDb = static_cast<double>(band.gainDb);
        const double q = static_cast<double>(band.q);
        const double slope = static_cast<double>(band.slope);
        const double steep = std::clamp(std::abs(slope), 6.0, 48.0);
        const int shelfOrder = std::clamp(static_cast<int>(std::lround(std::abs(slope) / 6.0)), 1, 4);
        const bool flat = std::abs(gainDb) < 1e-6;

        AnalogCascade proto;
        switch (band.shape)
        {
            case Shape::LowCut:    proto = FilterDesign::lowCut(hz, steep, q); break;
            case Shape::HighCut:   proto = FilterDesign::highCut(hz, steep, q); break;
            case Shape::Bell:      if (!flat) proto = FilterDesign::bell(hz, q, gainDb, 1); break;
            case Shape::LowShelf:  if (!flat) proto = FilterDesign::lowShelf(hz, gainDb, shelfOrder); break;
            case Shape::HighShelf: if (!flat) proto = FilterDesign::highShelf(hz, gainDb, shelfOrder); break;
            case Shape::Tilt:      proto = FilterDesign::tilt(hz, std::clamp(slope, -6.0, 6.0)); break;
            case Shape::Off:       break;
        }
        return FilterDesign::digital(proto, fit, out, kBandSections);
    }

    /** @brief Designs every band and sends each section to its coefficients. */
    void apply() noexcept
    {
        const int ramp = settled_ ? rampSamples_ : 0;
        for (int b = 0; b < kBands; ++b)
        {
            BiquadCoeffs sections[kBandSections];
            const int count = design(getBand(b), fit_, sections);
            for (int j = 0; j < kBandSections; ++j)
            {
                const int s = b * kBandSections + j;
                const auto index = static_cast<size_t>(s);
                if (j < count)
                {
                    if (!cascade_->isSectionEnabled(s))
                    {
                        // It comes in transparent, on the poles it will have.
                        cascade_->setCoeffs(s, sections[j].bypassed());
                        cascade_->setSectionEnabled(s, true);
                    }
                    cascade_->rampCoeffs(s, sections[j], ramp);
                    leaving_[index] = false;
                }
                else if (cascade_->isSectionEnabled(s) && !leaving_[index])
                {
                    if (ramp > 0)
                    {
                        cascade_->rampToBypass(s, ramp);
                        leaving_[index] = true;
                    }
                    else
                    {
                        cascade_->setSectionEnabled(s, false);
                    }
                }
            }
        }
        settled_ = true;
        refresh();
    }

    void refresh() noexcept
    {
        bool any = false, going = false;
        for (int s = 0; s < kSections; ++s)
        {
            any = any || cascade_->isSectionEnabled(s);
            going = going || leaving_[static_cast<size_t>(s)];
        }
        active_ = any;
        leavingAny_ = going;
    }

    std::array<Published, kBands> published_ {};
    std::atomic<bool> dirty_ { true };
    std::atomic<double> sampleRate_ { 48000.0 };

    // Audio thread.
    std::unique_ptr<Cascade> cascade_;
    FilterDesign::Band fit_;
    std::array<bool, kSections> leaving_ {};   ///< Sections on their way to transparent.
    int rampSamples_ = 960;
    bool settled_ = false;                     ///< The first settings after prepare() are taken at once.
    bool active_ = false;
    bool leavingAny_ = false;
};

/**
 * @class KeyMonitor
 * @brief The key of a dynamics processor as it is listened to: late by the
 *        processor's latency and held under full scale.
 *
 * A processor that plays its key instead of its output, so that the
 * sidechain can be tuned by ear, should leave the track where it is: the
 * key goes out late by the latency the processor reports, and switching the
 * listen on does not move it against the rest of the mix. An equalized key
 * can also be far louder than the programme (a bell of +24 dB on a full
 * mix): the monitor keeps it under a ceiling with one gain for all the
 * channels, which falls at once when a peak comes and returns over
 * kReleaseSeconds. Under the ceiling and with no delay the key goes out as
 * it came, to the bit.
 *
 * Threading: prepare() on the setup thread (it allocates and never throws:
 * without memory the key is not delayed); the rest on the audio thread.
 *
 * @tparam T           Sample type (float or double).
 * @tparam MaxChannels Largest number of key channels.
 */
template <FloatType T, int MaxChannels = 16>
class KeyMonitor
{
public:
    static constexpr double kCeiling = 1.0;          ///< Largest sample that goes out.
    static constexpr double kReleaseSeconds = 0.1;   ///< Time constant of the gain's return.

    /**
     * @brief Sets the rate and the longest delay (allocates).
     * @param sampleRate      Rate in hertz (invalid values leave the monitor as it was).
     * @param maxDelaySamples Longest delay the key will be given.
     */
    void prepare(double sampleRate, int maxDelaySamples) noexcept
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        release_ = std::exp(-1.0 / (kReleaseSeconds * sampleRate));
        int n = 1;
        while (n < std::max(1, maxDelaySamples) + 1 && n < (1 << 24)) n <<= 1;
        lines_.reset(new (std::nothrow) T[static_cast<size_t>(n) * static_cast<size_t>(MaxChannels)]);
        size_ = lines_ ? n : 0;
        reset();
    }

    /** @brief Clears the delay and lets the gain go. */
    void reset() noexcept
    {
        if (lines_) std::fill_n(lines_.get(), static_cast<size_t>(size_) * static_cast<size_t>(MaxChannels), T(0));
        pos_ = 0;
        peak_ = 0.0;
    }

    /**
     * @brief One frame: the key of every channel goes in and comes out late
     *        and under the ceiling.
     * @param frame    The key of each channel, replaced in place.
     * @param channels Channels in the frame (those over MaxChannels are left alone).
     * @param delay    Samples the key is held back (up to the delay given to prepare()).
     */
    void processFrame(T* frame, int channels, int delay) noexcept
    {
        const int nCh = std::min(channels, MaxChannels);
        double top = 0.0;
        if (size_ > 0)
        {
            const int d = std::clamp(delay, 0, size_ - 1);
            for (int ch = 0; ch < nCh; ++ch)
            {
                T* line = lines_.get() + static_cast<size_t>(ch) * static_cast<size_t>(size_);
                line[pos_] = frame[ch];
                frame[ch] = line[(pos_ - d) & (size_ - 1)];
            }
            pos_ = (pos_ + 1) & (size_ - 1);
        }
        for (int ch = 0; ch < nCh; ++ch) top = std::max(top, std::abs(static_cast<double>(frame[ch])));
        peak_ = std::max(top, peak_ * release_);
        if (peak_ > kCeiling)
        {
            const T gain = static_cast<T>(kCeiling / peak_);
            for (int ch = 0; ch < nCh; ++ch) frame[ch] *= gain;
        }
    }

private:
    std::unique_ptr<T[]> lines_;
    int size_ = 0, pos_ = 0;
    double release_ = 0.999, peak_ = 0.0;
};

} // namespace dspark
