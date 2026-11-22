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
 * Settings move while audio runs. setBand() publishes them from any thread
 * and the audio thread takes them in sync(), in one of two ways:
 *
 * - **A band that keeps its form glides.** While the shape, the placement
 *   and the number of sections stay, the band moves to its new setting over
 *   kRampSeconds in the domain of its controls (the frequency and the q
 *   along a logarithmic scale, the gain and the slope along a linear one),
 *   and it is designed again every kStretchSeconds on the way, with the
 *   coefficients moving in a line between one design and the next. Every
 *   filter it passes through is the design of a real setting: a narrow bell
 *   swept over three octaves is a narrow bell all the way.
 * - **A change of form crosses two banks.** When a band changes its shape,
 *   its placement or its number of sections (or goes in or out of the
 *   path), the whole new set goes into a second bank of filters with empty
 *   memory. Its input rises from nothing over kRampSeconds, so it starts
 *   without the transient of a filter switched on under a signal, while the
 *   bank that leaves keeps its input and its output falls over the same
 *   time. Nothing in between is a filter that neither set asked for.
 *   Settings that arrive during a crossing wait for it to end.
 *
 * A band can act on both channels of a pair, on its middle or on its sides
 * (Placement, through processFrame()). With every band off the equalizer
 * costs one test per sample.
 *
 * The owner drives it in three calls: sync() once per block (or per frame of
 * a per-sample stream), processFrame() for every frame of the key (or
 * processSample() for every sample of every channel, where the key comes one
 * channel at a time), and advance() with the number of frames gone by, at
 * least every getStretch() frames for a glide to be followed step by step.
 *
 * Threading: setBand(), getBand(), getResponse(), writeState() and
 * readState() from any thread. prepare() on the setup thread (it allocates
 * the two banks and never throws: without memory the equalizer stays out of
 * the path). sync(), isActive(), isPlaced(), processSample(),
 * processFrame(), advance() and reset() on the audio thread.
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
    static constexpr double kRampSeconds = 0.02;    ///< Time a new setting takes to arrive.
    static constexpr double kStretchSeconds = 1.0 / 1500.0; ///< A band on the move is designed again this often.

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

    /** @brief The part of a two-channel key a band acts on (see processFrame()). */
    enum class Placement
    {
        Stereo,     ///< Both channels (default).
        Mid,        ///< The middle of the pair, (L + R) / 2, alone.
        Side        ///< The sides of the pair, (L - R) / 2, alone.
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
        Placement placement = Placement::Stereo; ///< Part of a two-channel key it acts on.
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
     * Allocates the two banks the first time. The published bands are kept
     * and taken at once (with no glide and no fade) by the next sync(). An
     * invalid rate is ignored.
     *
     * @param sampleRate Sample rate in hertz.
     */
    void prepare(double sampleRate) noexcept
    {
        if (!std::isfinite(sampleRate) || !(sampleRate > 0.0)) return;
        for (Bank& bank : bank_)
            if (!bank.cascade) bank.cascade.reset(new (std::nothrow) Cascade());
        if (!bank_[0].cascade || !bank_[1].cascade) return;
        for (Bank& bank : bank_) clear(bank);
        fit_.prepare(sampleRate);
        sampleRate_.store(sampleRate, std::memory_order_relaxed);
        rampSamples_ = std::max(1, static_cast<int>(kRampSeconds * sampleRate));
        stretch_ = std::max(8, static_cast<int>(std::lround(kStretchSeconds * sampleRate)));
        fadeStep_ = 1.0 / static_cast<double>(rampSamples_);
        for (Glide& g : glide_) g.moving = false;
        count_.fill(0);
        phase_.fill(0);
        live_ = 0;
        stretchLeft_ = stretch_;
        fade_ = 0.0;
        settled_ = false;
        gliding_ = false;
        changing_ = false;
        waiting_ = false;
        active_ = false;
        placed_ = false;
        inParts_ = false;
        dirty_.store(true, std::memory_order_release);
    }

    /**
     * @brief Clears the memory of the filters (settings and glides are kept).
     *
     * A change of design in flight ends at once: the bank that was coming in
     * is the one that stays.
     */
    void reset() noexcept
    {
        if (!bank_[0].cascade || !bank_[1].cascade) return;
        if (changing_) finishChange();
        for (Bank& bank : bank_) bank.cascade->reset();
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
        p.placement.store(std::clamp(static_cast<int>(settings.placement), 0, static_cast<int>(Placement::Side)),
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
        out.placement = static_cast<Placement>(p.placement.load(std::memory_order_relaxed));
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
     *
     * @param part Stereo (default): every band, wherever it is placed. Mid or
     *             Side: what that part of a two-channel key goes through,
     *             the bands on both channels and the ones placed on it.
     */
    [[nodiscard]] Response getResponse(Placement part = Placement::Stereo) const noexcept
    {
        Response out;
        out.sampleRate = sampleRate_.load(std::memory_order_relaxed);
        const FilterDesign::Band fit(out.sampleRate);
        for (int b = 0; b < kBands; ++b)
        {
            const Band band = getBand(b);
            if (part != Placement::Stereo && band.placement != Placement::Stereo && band.placement != part) continue;
            out.count += design(band, fit, out.sections.data() + out.count, false);
        }
        return out;
    }

    /**
     * @brief Writes the bands under keys that start with a prefix
     *        ("<prefix><band>.shape", ".freq", ".gain", ".q", ".slope",
     *        ".place").
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
            std::snprintf(key, sizeof(key), "%s%d.place", prefix, b);
            w.write(key, static_cast<int32_t>(band.placement));
        }
    }

    /** @brief Reads the bands writeState() wrote (a band that is missing reads
     *         as Off, a placement that is missing as Stereo). */
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
            std::snprintf(key, sizeof(key), "%s%d.place", prefix, b);
            band.placement = static_cast<Placement>(std::clamp(static_cast<int>(r.read(key, int32_t(0))), 0,
                                                               static_cast<int>(Placement::Side)));
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
        if (!bank_[0].cascade || !bank_[1].cascade) return false;
        if (dirty_.load(std::memory_order_relaxed) && dirty_.exchange(false, std::memory_order_acquire))
            apply();
        return active_;
    }

    /** @brief Whether any section is in the path: processSample() and
     *         processFrame() do nothing otherwise. */
    [[nodiscard]] bool isActive() const noexcept { return active_; }

    /** @brief Whether a band in the path is placed on the middle or on the
     *         sides: only processFrame() gives those their part. */
    [[nodiscard]] bool isPlaced() const noexcept { return placed_; }

    /** @brief Frames between two designs of a band on the move: advance() at
     *         least this often for a glide to be followed step by step. */
    [[nodiscard]] int getStretch() const noexcept { return stretch_; }

    /**
     * @brief One sample of one channel of the key through the equalizer.
     *
     * Every band acts on the channel, wherever it is placed: a caller that
     * has the key one channel at a time has no middle and no sides to give.
     *
     * @pre channel in [0, MaxChannels) and a finite input.
     * @param input   Key sample.
     * @param channel Channel index.
     * @return The shaped key (the input itself while no band is in the path).
     */
    [[nodiscard]] T processSample(T input, int channel) noexcept
    {
        if (!active_) return input;
        return static_cast<T>(run(static_cast<double>(input), channel, Placement::Stereo));
    }

    /**
     * @brief One frame of the key, every channel of it, in place.
     *
     * With two channels and a band placed on the middle or on the sides,
     * the frame is taken to its middle (L + R) / 2 and its sides
     * (L - R) / 2, each goes through the bands on both channels and the
     * bands placed on it, and the pair comes back as their sum and their
     * difference. With no band placed there is no conversion: each channel
     * goes through every band, exactly as through processSample(). A single
     * channel is its own middle and has no sides: the bands placed on the
     * sides leave it alone. With more than two channels every band acts on
     * every channel.
     *
     * @pre Finite samples.
     * @param frame    The key of each channel, replaced in place.
     * @param channels Channels in the frame (those over MaxChannels are left alone).
     */
    void processFrame(T* frame, int channels) noexcept
    {
        if (!active_) return;
        const int nCh = std::min(channels, MaxChannels);
        const bool parts = nCh == 2 && placed_;
        if (parts != inParts_) turn(parts);
        if (parts)
        {
            const double left = static_cast<double>(frame[0]), right = static_cast<double>(frame[1]);
            const double mid = run(0.5 * (left + right), 0, Placement::Mid);
            const double side = run(0.5 * (left - right), 1, Placement::Side);
            frame[0] = static_cast<T>(mid + side);
            frame[1] = static_cast<T>(mid - side);
            return;
        }
        const Placement part = nCh == 1 ? Placement::Mid : Placement::Stereo;
        for (int ch = 0; ch < nCh; ++ch)
            frame[ch] = static_cast<T>(run(static_cast<double>(frame[ch]), ch, part));
    }

    /**
     * @brief Moves the equalizer on in time: the coefficients between two
     *        designs, the bands on the move to their next design, and a
     *        change of design to its end.
     * @param samples Frames gone by since the last call.
     */
    void advance(int samples) noexcept
    {
        if (!active_ || samples <= 0) return;
        Cascade& live = *bank_[static_cast<size_t>(live_)].cascade;
        if (live.isRamping()) live.advanceRamps(samples);
        if (gliding_)
        {
            stretchLeft_ -= samples;
            if (stretchLeft_ <= 0) nextStretch();
        }
        if (changing_)
        {
            fade_ += fadeStep_ * static_cast<double>(samples);
            phase_.fill(0);
            if (fade_ >= 1.0) finishChange();
        }
    }

private:
    using Cascade = BiquadCascade<T, kSections, MaxChannels>;

    struct Published
    {
        std::atomic<int> shape { 0 };
        std::atomic<int> placement { 0 };
        std::atomic<T> frequency { T(1000) };
        std::atomic<T> gainDb { T(0) };
        std::atomic<T> q { T(0.7071) };
        std::atomic<T> slope { T(12) };
    };

    /// One set of filters: the equalizer runs one, and two while a design changes.
    struct Bank
    {
        std::unique_ptr<Cascade> cascade;
        std::uint64_t midOnly = 0;    ///< Sections of the bands placed on the middle, one bit each.
        std::uint64_t sideOnly = 0;   ///< Sections of the bands placed on the sides.
        bool any = false;             ///< A section is in the path.
    };

    /// A band on its way from one setting to another of the same form.
    struct Glide
    {
        Band from {}, to {};
        int left = 0;                 ///< Frames of the way left at the last design.
        bool moving = false;
    };

    /**
     * @brief The digital sections of one band; returns how many it wrote (at
     *        most kBandSections).
     * @param keepFlat A bell or a shelf at 0 dB is designed all the same (a
     *        section that changes nothing), so that a gain on its way through
     *        0 dB keeps its sections.
     */
    [[nodiscard]] static int design(const Band& band, const FilterDesign::Band& fit, BiquadCoeffs* out,
                                    bool keepFlat) noexcept
    {
        const double hz = std::clamp(static_cast<double>(band.frequency), 10.0, 0.45 * fit.getSampleRate());
        const double gainDb = static_cast<double>(band.gainDb);
        const double q = static_cast<double>(band.q);
        const double slope = static_cast<double>(band.slope);
        const double steep = std::clamp(std::abs(slope), 6.0, 48.0);
        const int shelfOrder = std::clamp(static_cast<int>(std::lround(std::abs(slope) / 6.0)), 1, 4);
        const bool shaped = keepFlat || std::abs(gainDb) >= 1e-6;

        AnalogCascade proto;
        switch (band.shape)
        {
            case Shape::LowCut:    proto = FilterDesign::lowCut(hz, steep, q); break;
            case Shape::HighCut:   proto = FilterDesign::highCut(hz, steep, q); break;
            case Shape::Bell:      if (shaped) proto = FilterDesign::bell(hz, q, gainDb, 1); break;
            case Shape::LowShelf:  if (shaped) proto = FilterDesign::lowShelf(hz, gainDb, shelfOrder); break;
            case Shape::HighShelf: if (shaped) proto = FilterDesign::highShelf(hz, gainDb, shelfOrder); break;
            case Shape::Tilt:      proto = FilterDesign::tilt(hz, std::clamp(slope, -6.0, 6.0)); break;
            case Shape::Off:       break;
        }
        return FilterDesign::digital(proto, fit, out, kBandSections);
    }

    /** @brief Whether two settings are the same in every field. */
    [[nodiscard]] static bool same(const Band& a, const Band& b) noexcept
    {
        return a.shape == b.shape && a.placement == b.placement && a.frequency == b.frequency
            && a.gainDb == b.gainDb && a.q == b.q && a.slope == b.slope;
    }

    /**
     * @brief The setting a share p of the way from one to another: the
     *        frequency and the q along a logarithmic scale, the gain and the
     *        slope along a linear one.
     */
    [[nodiscard]] static Band between(const Band& from, const Band& to, double p) noexcept
    {
        if (p >= 1.0) return to;
        const auto geometric = [p](T a, T b) {
            const double la = std::log(static_cast<double>(a)), lb = std::log(static_cast<double>(b));
            return static_cast<T>(std::exp(la + (lb - la) * p));
        };
        const auto linear = [p](T a, T b) {
            return static_cast<T>(static_cast<double>(a) + (static_cast<double>(b) - static_cast<double>(a)) * p);
        };
        Band at = to;
        at.frequency = geometric(from.frequency, to.frequency);
        at.q = geometric(from.q, to.q);
        at.gainDb = linear(from.gainDb, to.gainDb);
        at.slope = linear(from.slope, to.slope);
        return at;
    }

    /** @brief Empties a bank: no section in the path, no memory. */
    static void clear(Bank& bank) noexcept
    {
        bank.cascade->setNumSections(kSections);
        for (int s = 0; s < kSections; ++s)
        {
            bank.cascade->setCoeffs(s, BiquadCoeffs {});
            bank.cascade->setSectionEnabled(s, false);
        }
        bank.cascade->reset();
        bank.midOnly = 0;
        bank.sideOnly = 0;
        bank.any = false;
    }

    /** @brief Gives a bank a set of bands at once, with empty memory, and
     *         writes how many sections each band took. */
    void build(Bank& bank, const std::array<Band, kBands>& bands, std::array<int, kBands>& counts) noexcept
    {
        bank.midOnly = 0;
        bank.sideOnly = 0;
        bank.any = false;
        for (int b = 0; b < kBands; ++b)
        {
            const Band& band = bands[static_cast<size_t>(b)];
            BiquadCoeffs sections[kBandSections];
            const int count = design(band, fit_, sections, false);
            counts[static_cast<size_t>(b)] = count;
            for (int j = 0; j < kBandSections; ++j)
            {
                const int s = b * kBandSections + j;
                if (j < count)
                {
                    bank.cascade->setCoeffs(s, sections[j]);
                    bank.cascade->setSectionEnabled(s, true);
                    const std::uint64_t bit = std::uint64_t { 1 } << s;
                    if (band.placement == Placement::Mid) bank.midOnly |= bit;
                    if (band.placement == Placement::Side) bank.sideOnly |= bit;
                    bank.any = true;
                }
                else
                {
                    bank.cascade->setSectionEnabled(s, false);
                }
            }
        }
        bank.cascade->reset();
    }

    /**
     * @brief Takes the published bands.
     *
     * The first settings after prepare() go in at once. After that, a band
     * that keeps its form (its shape, its placement and its number of
     * sections) glides to its new setting; if any band changes its form the
     * whole set goes into the other bank and the two cross over. Settings
     * that come while a crossing is in flight wait for it to end.
     */
    void apply() noexcept
    {
        if (changing_) { waiting_ = true; return; }

        std::array<Band, kBands> wanted;
        for (int b = 0; b < kBands; ++b) wanted[static_cast<size_t>(b)] = getBand(b);

        Bank& live = bank_[static_cast<size_t>(live_)];
        if (!settled_)
        {
            build(live, wanted, count_);
            goal_ = wanted;
            settled_ = true;
            refresh();
            return;
        }

        bool sameForm = true;
        for (int b = 0; b < kBands && sameForm; ++b)
        {
            const auto index = static_cast<size_t>(b);
            const Band& now = goal_[index];
            const Band& next = wanted[index];
            if (same(now, next)) continue;
            BiquadCoeffs sections[kBandSections];
            sameForm = next.shape == now.shape && next.placement == now.placement
                    && design(next, fit_, sections, false) == count_[index];
        }
        if (!sameForm) { startChange(wanted); return; }

        // Since the last design: a glide that starts now starts from here.
        const int elapsed = stretch_ - stretchLeft_;
        bool started = false;
        for (int b = 0; b < kBands; ++b)
        {
            const auto index = static_cast<size_t>(b);
            if (same(goal_[index], wanted[index]) || count_[index] == 0)
            {
                goal_[index] = wanted[index];
                continue;
            }
            Glide& g = glide_[index];
            g.from = g.moving ? between(g.from, g.to, progress(g, elapsed)) : goal_[index];
            g.to = wanted[index];
            g.left = rampSamples_ + elapsed;
            g.moving = true;
            goal_[index] = wanted[index];
            started = true;
        }
        if (started)
        {
            gliding_ = true;
            nextStretch();
        }
    }

    /** @brief How far along its way a glide is, `elapsed` frames after its last design. */
    [[nodiscard]] double progress(const Glide& g, int elapsed) const noexcept
    {
        return std::clamp(1.0 - static_cast<double>(g.left - elapsed) / static_cast<double>(rampSamples_), 0.0, 1.0);
    }

    /**
     * @brief Designs every band on the move where it will be a stretch from
     *        now, and sends its coefficients there over the stretch: each
     *        filter on the way is the design of a real setting.
     */
    void nextStretch() noexcept
    {
        const int elapsed = stretch_ - stretchLeft_;
        stretchLeft_ = stretch_;
        Cascade& live = *bank_[static_cast<size_t>(live_)].cascade;
        bool any = false, lost = false;
        for (int b = 0; b < kBands; ++b)
        {
            const auto index = static_cast<size_t>(b);
            Glide& g = glide_[index];
            if (!g.moving) continue;
            g.left -= elapsed;
            const int after = g.left - stretch_;
            const double p = after > 0 ? 1.0 - static_cast<double>(after) / static_cast<double>(rampSamples_) : 1.0;
            BiquadCoeffs sections[kBandSections];
            const int count = design(between(g.from, g.to, p), fit_, sections, true);
            if (count != count_[index])
            {
                // The form changed on the way (a tilt through 0 dB per octave).
                g.moving = false;
                lost = true;
                continue;
            }
            const int length = std::max(1, std::min(stretch_, g.left));
            for (int j = 0; j < count; ++j)
                live.rampCoeffs(b * kBandSections + j, sections[j], length);
            if (after > 0) any = true;
            else g.moving = false;
        }
        gliding_ = any;
        if (lost && !changing_) startChange(goal_);
    }

    /**
     * @brief Starts the crossing to a new set of bands.
     *
     * The other bank takes the set with empty memory and its input rises
     * from nothing over kRampSeconds, so it starts without the transient of
     * a filter switched on under a signal; the bank that leaves keeps its
     * input and its output falls over the same time.
     */
    void startChange(const std::array<Band, kBands>& wanted) noexcept
    {
        Bank& next = bank_[static_cast<size_t>(1 - live_)];
        build(next, wanted, count_);
        goal_ = wanted;
        for (Glide& g : glide_) g.moving = false;
        gliding_ = false;
        phase_.fill(0);
        fade_ = 0.0;
        changing_ = true;
        refresh();
    }

    /** @brief Ends the crossing: the bank that came in stays alone. */
    void finishChange() noexcept
    {
        clear(bank_[static_cast<size_t>(live_)]);
        live_ = 1 - live_;
        changing_ = false;
        fade_ = 0.0;
        stretchLeft_ = stretch_;
        refresh();
        if (waiting_)
        {
            waiting_ = false;
            dirty_.store(true, std::memory_order_release);
        }
    }

    void refresh() noexcept
    {
        const Bank& live = bank_[static_cast<size_t>(live_)];
        const Bank& next = bank_[static_cast<size_t>(1 - live_)];
        active_ = live.any || changing_;
        placed_ = (live.midOnly | live.sideOnly) != 0 || (changing_ && (next.midOnly | next.sideOnly) != 0);
    }

    /**
     * @brief Takes the memory of the pair to its middle and its sides, or
     *        back, when the frames start to be given that way: the filters
     *        carry on from what they have played and nothing steps.
     */
    void turn(bool parts) noexcept
    {
        for (Bank& bank : bank_) bank.cascade->turnPairMemory(0, 1, !parts);
        inParts_ = parts;
    }

    /** @brief The sections a part of the key leaves out in a bank. */
    [[nodiscard]] static std::uint64_t leftOut(const Bank& bank, Placement part) noexcept
    {
        return part == Placement::Mid ? bank.sideOnly : (part == Placement::Side ? bank.midOnly : 0);
    }

    /** @brief One sample of one part of the key through the bank in use, and
     *         through the one coming in while a design changes. */
    [[nodiscard]] double run(double x, int channel, Placement part) noexcept
    {
        Bank& live = bank_[static_cast<size_t>(live_)];
        const double out = live.any ? live.cascade->processSample(x, channel, leftOut(live, part)) : x;
        if (!changing_) return out;

        Bank& next = bank_[static_cast<size_t>(1 - live_)];
        int& at = phase_[static_cast<size_t>(channel)];
        const double w = std::min(1.0, fade_ + fadeStep_ * static_cast<double>(at));
        ++at;
        const double in = w * x;
        const double entering = next.any ? next.cascade->processSample(in, channel, leftOut(next, part)) : in;
        return (1.0 - w) * out + entering;
    }

    std::array<Published, kBands> published_ {};
    std::atomic<bool> dirty_ { true };
    std::atomic<double> sampleRate_ { 48000.0 };

    // Audio thread.
    std::array<Bank, 2> bank_ {};
    FilterDesign::Band fit_;
    std::array<Band, kBands> goal_ {};         ///< Where each band of the bank in use is, or is going.
    std::array<int, kBands> count_ {};         ///< Sections each of them takes.
    std::array<Glide, kBands> glide_ {};
    std::array<int, MaxChannels> phase_ {};    ///< Samples of each channel into the current advance().
    int live_ = 0;                             ///< The bank in use.
    int rampSamples_ = 960;
    int stretch_ = 32;
    int stretchLeft_ = 32;                     ///< Frames until the bands on the move are designed again.
    double fade_ = 0.0, fadeStep_ = 1.0 / 960.0; ///< Share of the crossing done, and its step per frame.
    bool settled_ = false;                     ///< The first settings after prepare() are taken at once.
    bool gliding_ = false;                     ///< A band is on the move.
    bool changing_ = false;                    ///< Two banks are crossing.
    bool waiting_ = false;                     ///< Settings came during the crossing.
    bool active_ = false;
    bool placed_ = false;
    bool inParts_ = false;                     ///< Channels 0 and 1 of the banks hold the middle and the sides.
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
