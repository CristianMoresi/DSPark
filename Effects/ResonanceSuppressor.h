// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file ResonanceSuppressor.h
 * @brief A spectral suppressor of resonances: it finds what stands over the
 *        envelope of the spectrum, frame by frame, and lowers it with a
 *        minimum-phase filter (no pre-echo).
 *
 * The analysis window is about 46 ms and doubles by octaves with the rate
 * (2048 samples at 44.1 and 48 kHz, 4096 at 88.2 and 96, 8192 above), so
 * the resolution is the same at any rate. Each frame of the programme, or
 * of an external key, is read as a power spectrum, smoothed across three
 * bins, held over a floor relative to its maximum and smoothed in time.
 * Its envelope comes from a smoothing whose width grows with frequency
 * (SpectralEnvelope, setWidth() in octaves); a resonance is what stands
 * over it, in decibels (Mode::Smooth) or in amplitude (Mode::Deep). Over
 * setThreshold() the excess becomes a reduction (setStrength(): 1 takes the
 * whole excess out, 0.5 half of it in decibels), limited by setRange(),
 * weighted by a sensitivity curve drawn with six bands (setSensitivityBand():
 * cuts, bells, shelves, tilt) and given, per bin, an attack and a release
 * that are slower in the low end. The gain curve is smoothed across bins
 * and applied by a SpectralFilter: a minimum-phase filter designed every hop
 * and cut to the room the padding leaves, so a reduction never starts before
 * the event.
 *
 * Stereo: Link::PerBin mixes each bin's power with the mean of the channels
 * (setLink() 0 to 1); at 1 both channels share the detector and the filter
 * (half the cost); Link::MidSide processes the mid and the side, the side's
 * reduction scaled by setSideWeight(). setRemovalListen(true) outputs what is
 * taken out; setFocusListen(true) outputs the programme through the
 * sensitivity curve (where the suppressor listens); both are monitoring
 * controls and stay out of the state. setQuality(Offline)
 * pads the transform 8 times and overlaps 16 frames per window (for
 * renders); Realtime pads twice and overlaps four.
 * getDisplayReduction() and getDisplayExcess() give, at 128 points, what
 * the last frame takes out and what it found, for a display on any thread.
 *
 * Latency: the analysis window (getLatency()).
 *
 * Threading: prepare(), setQuality() taking effect and setState() on the
 * setup thread (allocate: setQuality() applies at the next prepare());
 * processBlock() and reset() on the audio thread; the other setters are
 * lock-free atomic publications. Non-finite values are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, AuditoryBands.h, DspMath.h, KeyEqualizer.h,
 * SpectralEnvelope.h, SpectralFilter.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AuditoryBands.h"
#include "../Core/DspMath.h"
#include "../Core/KeyEqualizer.h"
#include "../Core/SpectralEnvelope.h"
#include "../Core/SpectralFilter.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class ResonanceSuppressor
 * @brief Dynamic suppression of resonances by a minimum-phase spectral filter.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class ResonanceSuppressor
{
public:
    enum class Mode { Smooth, Deep };          ///< Resonance in decibels, or in amplitude.
    enum class Link { PerBin, MidSide };     ///< Stereo handling.
    enum class Quality { Realtime, Offline };
    static constexpr int kSensitivityBands = 6;
    using Band = typename KeyEqualizer<T, 2>::Band;

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        spec_ = spec;
        const int octaves = std::max(0, static_cast<int>(std::lround(std::log2(std::max(spec.sampleRate, 44100.0) / 44100.0))));
        const int window = std::min(2048 << octaves, 16384);
        const bool offline = quality_.load(std::memory_order_relaxed) == Quality::Offline;
        filter_.prepare(spec, window, offline ? 8 : 2, offline ? 16 : 4);
        bins_ = filter_.getNumBins();
        envelope_.prepare(bins_, static_cast<double>(width_.load(std::memory_order_relaxed)));
        lastWidth_ = width_.load(std::memory_order_relaxed);
        const auto chans = static_cast<size_t>(std::max(1, std::min(spec.numChannels, 2)));
        smoothPower_.assign(chans, std::vector<T>(static_cast<size_t>(bins_), T(0)));
        gainDb_.assign(chans + 1, std::vector<T>(static_cast<size_t>(bins_), T(0)));
        work_.assign(static_cast<size_t>(bins_), T(0));
        db_.assign(static_cast<size_t>(bins_), T(0));
        env_.assign(static_cast<size_t>(bins_), T(0));
        mixed_.assign(static_cast<size_t>(bins_), T(0));
        weight_.assign(static_cast<size_t>(bins_), T(1));
        attack_.assign(static_cast<size_t>(bins_), T(0));
        release_.assign(static_cast<size_t>(bins_), T(0));
        sensitivity_.prepare(spec.sampleRate);
        reduction_.prepare(filter_.getBinHz(), bins_);
        excess_.prepare(filter_.getBinHz(), bins_);
        refreshTimes();
        weightPos_ = 0;
        weightDirty_.store(true, std::memory_order_release);
        prepared_ = true;
    }

    /** @brief Clears the analysis and the filter. */
    void reset() noexcept
    {
        filter_.reset();
        for (auto& v : smoothPower_) std::fill(v.begin(), v.end(), T(0));
        for (auto& v : gainDb_) std::fill(v.begin(), v.end(), T(0));
        reduction_.clear();
        excess_.clear();
    }

    // -- Controls ------------------------------------------------------------

    /** @brief Over the envelope by this much a bin is a resonance, 0 to 30 dB (default 3). */
    void setThreshold(T db) noexcept { store(threshold_, db, T(0), T(30)); }
    /** @brief How much of the excess is taken out, 0 to 1 (default 0.7). */
    void setStrength(T amount) noexcept { store(strength_, amount, T(0), T(1)); }
    /** @brief Largest reduction, 0 to 48 dB (default 12). */
    void setRange(T db) noexcept { store(range_, db, T(0), T(48)); }
    /** @brief Width of the envelope in octaves, 0.05 to 4 (default 1/3). */
    void setWidth(T octaves) noexcept { store(width_, octaves, T(0.05), T(4)); }
    /** @brief Attack at 1 kHz, 0.5 to 500 ms (default 5; slower below, faster above, by the square root of the frequency). */
    void setAttack(T ms) noexcept { store(attackMs_, ms, T(0.5), T(500)); timesDirty_.store(true, std::memory_order_release); }
    /** @brief Release at 1 kHz, 5 to 2000 ms (default 80). */
    void setRelease(T ms) noexcept { store(releaseMs_, ms, T(5), T(2000)); timesDirty_.store(true, std::memory_order_release); }
    /** @brief Resonance in decibels (Smooth, default) or in amplitude (Deep). */
    void setMode(Mode m) noexcept { mode_.store(m == Mode::Deep ? Mode::Deep : Mode::Smooth, std::memory_order_relaxed); }
    /** @brief Stereo handling (PerBin by default). */
    void setLinkMode(Link l) noexcept { link_.store(l == Link::MidSide ? Link::MidSide : Link::PerBin, std::memory_order_relaxed); }
    /** @brief Per-bin link, 0 (each channel its own) to 1 (one detector and filter), default 1. */
    void setLink(T amount) noexcept { store(linkAmount_, amount, T(0), T(1)); }
    /** @brief In MidSide, the side's reduction as a share of the mid's, 0 to 2 (default 1). */
    void setSideWeight(T share) noexcept { store(sideWeight_, share, T(0), T(2)); }
    /** @brief Outputs what is taken out. */
    void setRemovalListen(bool on) noexcept { removalListen_.store(on, std::memory_order_relaxed); }
    /** @brief Outputs the programme through the sensitivity curve. */
    void setFocusListen(bool on) noexcept { focusListen_.store(on, std::memory_order_relaxed); }
    /** @brief Realtime or Offline (applies at the next prepare()). */
    void setQuality(Quality q) noexcept { quality_.store(q == Quality::Offline ? Quality::Offline : Quality::Realtime, std::memory_order_relaxed); }
    /** @brief One band of the sensitivity curve (KeyEqualizer bands; the curve is a weight, its gain in dB / 12 + 1, clamped 0 to 2). */
    void setSensitivityBand(int index, const Band& band) noexcept
    {
        sensitivity_.setBand(index, band);
        weightDirty_.store(true, std::memory_order_release);
    }

    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getStrength() const noexcept { return strength_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRange() const noexcept { return range_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWidth() const noexcept { return width_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getAttack() const noexcept { return attackMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] Link getLinkMode() const noexcept { return link_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLink() const noexcept { return linkAmount_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSideWeight() const noexcept { return sideWeight_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getRemovalListen() const noexcept { return removalListen_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getFocusListen() const noexcept { return focusListen_.load(std::memory_order_relaxed); }
    [[nodiscard]] Quality getQuality() const noexcept { return quality_.load(std::memory_order_relaxed); }
    [[nodiscard]] Band getSensitivityBand(int index) const noexcept { return sensitivity_.getBand(index); }
    /** @return The latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return filter_.getLatency(); }
    /** @return The reduction of the last frame at a frequency, in dB (audio thread or after processing). */
    [[nodiscard]] T getReduction(double hz) const noexcept
    {
        if (gainDb_.empty() || bins_ < 2) return T(0);
        const int k = std::clamp(static_cast<int>(std::lround(hz / filter_.getBinHz())), 0, bins_ - 1);
        return gainDb_[0][static_cast<size_t>(k)];
    }

    // -- Display (any thread) --------------------------------------------------

    /** @brief Points of the display curves, evenly in log frequency from 20 Hz to 20 kHz. */
    static constexpr int kDisplayPoints = SpectralReadout<T>::kPoints;
    /** @return The frequency of a display point, in hertz. */
    [[nodiscard]] static double getDisplayFrequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }
    /**
     * @return The reduction of the last frame around a display point, in dB
     *         (0 or less), smoothed as it is applied. Channel 0 and 1 (the
     *         mid and the side with Link::MidSide); a linked pair shows the
     *         same curve on both. With setFocusListen(true), the gain of the
     *         focus filter. Any thread.
     */
    [[nodiscard]] T getDisplayReduction(int channel, int point) const noexcept { return reduction_.value(channel, point); }
    /**
     * @return What the last frame found around a display point: the
     *         resonance over the threshold, weighted by the sensitivity
     *         curve, in dB (0 or more), before setStrength() and setRange().
     *         Any thread.
     */
    [[nodiscard]] T getDisplayExcess(int channel, int point) const noexcept { return excess_.value(channel, point); }
    /** @return The deepest reduction of the last frame on either channel, in dB (any thread). */
    [[nodiscard]] T getPeakReduction() const noexcept { return reduction_.extreme(); }

    // -- Processing ----------------------------------------------------------

    /** @brief Processes the programme in place, detecting on itself. */
    void processBlock(AudioBufferView<T> buffer) noexcept { run(buffer, nullptr); }
    /** @brief Processes the programme in place, detecting on a key (same length). */
    void processBlock(AudioBufferView<T> buffer, AudioBufferView<T> key) noexcept
    {
        run(buffer, key.getNumSamples() >= buffer.getNumSamples() && key.getNumChannels() > 0 ? &key : nullptr);
    }

    // -- State ---------------------------------------------------------------

    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("RSUP"), 1);
        w.write("threshold", static_cast<float>(getThreshold()));
        w.write("strength", static_cast<float>(getStrength()));
        w.write("range", static_cast<float>(getRange()));
        w.write("width", static_cast<float>(getWidth()));
        w.write("attack", static_cast<float>(getAttack()));
        w.write("release", static_cast<float>(getRelease()));
        w.write("mode", static_cast<int32_t>(getMode()));
        w.write("linkMode", static_cast<int32_t>(getLinkMode()));
        w.write("link", static_cast<float>(getLink()));
        w.write("sideWeight", static_cast<float>(getSideWeight()));
        w.write("quality", static_cast<int32_t>(getQuality()));
        sensitivity_.writeState(w, "sens");
        return w.blob();
    }

    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("RSUP")) return false;
        setThreshold(static_cast<T>(r.read("threshold", 3.0f)));
        setStrength(static_cast<T>(r.read("strength", 0.7f)));
        setRange(static_cast<T>(r.read("range", 12.0f)));
        setWidth(static_cast<T>(r.read("width", 1.0f / 3.0f)));
        setAttack(static_cast<T>(r.read("attack", 5.0f)));
        setRelease(static_cast<T>(r.read("release", 80.0f)));
        setMode(r.read("mode", 0) == 1 ? Mode::Deep : Mode::Smooth);
        setLinkMode(r.read("linkMode", 0) == 1 ? Link::MidSide : Link::PerBin);
        setLink(static_cast<T>(r.read("link", 1.0f)));
        setSideWeight(static_cast<T>(r.read("sideWeight", 1.0f)));
        setQuality(r.read("quality", 0) == 1 ? Quality::Offline : Quality::Realtime);
        sensitivity_.readState(r, "sens");
        weightDirty_.store(true, std::memory_order_release);
        return true;
    }

private:
    static void store(std::atomic<T>& a, T v, T lo, T hi) noexcept
    {
        if (std::isfinite(v)) a.store(std::clamp(v, lo, hi), std::memory_order_relaxed);
    }

    /** @brief Attack and release per bin: slower below 1 kHz, faster above, by the square root of the frequency. */
    void refreshTimes() noexcept
    {
        const double hopSeconds = static_cast<double>(filter_.getHopSize()) / spec_.sampleRate;
        const double a = 0.001 * static_cast<double>(attackMs_.load(std::memory_order_relaxed));
        const double r = 0.001 * static_cast<double>(releaseMs_.load(std::memory_order_relaxed));
        for (int k = 0; k < bins_; ++k)
        {
            const double f = std::max(20.0, k * filter_.getBinHz());
            const double scale = std::sqrt(1000.0 / f);
            attack_[static_cast<size_t>(k)] = static_cast<T>(std::exp(-hopSeconds / std::max(1e-4, a * scale)));
            release_[static_cast<size_t>(k)] = static_cast<T>(std::exp(-hopSeconds / std::max(1e-3, r * scale)));
        }
    }

    /** @brief A share of the sensitivity curve per hop (the whole curve over eight hops). */
    void refreshWeight() noexcept
    {
        if (weightDirty_.exchange(false, std::memory_order_acquire))
        {
            sensResponse_ = sensitivity_.getResponse();
            weightPos_ = 0;
            weightPending_ = true;
        }
        if (!weightPending_) return;
        const int share = (bins_ + 7) / 8;
        const int end = std::min(bins_, weightPos_ + share);
        for (int k = weightPos_; k < end; ++k)
        {
            const double db = sensResponse_.magnitudeDb(std::max(1.0, k * filter_.getBinHz()));
            weight_[static_cast<size_t>(k)] = static_cast<T>(std::clamp(1.0 + db / 12.0, 0.0, 2.0));
        }
        weightPos_ = end;
        if (weightPos_ >= bins_) weightPending_ = false;
    }

    void run(AudioBufferView<T>& buffer, const AudioBufferView<T>* key) noexcept
    {
        if (!prepared_) return;
        const int nCh = std::min(buffer.getNumChannels(), static_cast<int>(smoothPower_.size()));
        const int nS = buffer.getNumSamples();
        if (nCh <= 0 || nS <= 0) return;
        if (timesDirty_.exchange(false, std::memory_order_acquire)) refreshTimes();
        if (const T w = width_.load(std::memory_order_relaxed); w != lastWidth_) { envelope_.setWidth(static_cast<double>(w)); lastWidth_ = w; }
        refreshWeight();
        const bool ms = nCh == 2 && link_.load(std::memory_order_relaxed) == Link::MidSide;
        const T linkAmount = linkAmount_.load(std::memory_order_relaxed);
        filter_.setLinked(!ms && nCh == 2 && linkAmount >= T(1));
        filter_.setRemovalListen(removalListen_.load(std::memory_order_relaxed));
        if (ms)
        {
            T* l = buffer.getChannel(0);
            T* r = buffer.getChannel(1);
            for (int i = 0; i < nS; ++i) { const T m = T(0.5) * (l[i] + r[i]), s = T(0.5) * (l[i] - r[i]); l[i] = m; r[i] = s; }
        }
        const bool focus = focusListen_.load(std::memory_order_relaxed);
        // A key replaces the programme in the power the design reads (its
        // mid and side in MidSide are read as its first two channels).
        filter_.processBlock(buffer, key, [&](int ch, const T* power, T* gain, int bins) noexcept {
            design(ch, power, gain, bins, ms, focus);
        });
        if (ms)
        {
            T* l = buffer.getChannel(0);
            T* r = buffer.getChannel(1);
            for (int i = 0; i < nS; ++i) { const T m = l[i], s = r[i]; l[i] = m + s; r[i] = m - s; }
        }
    }

    /** @brief The gain curve of one frame (channel -1: linked). */
    void design(int ch, const T* power, T* gain, int bins, bool ms, bool focus) noexcept
    {
        if (focus)
        {
            for (int k = 0; k < bins; ++k)
            {
                gain[k] = weight_[static_cast<size_t>(k)] * T(0.5);
                mixed_[static_cast<size_t>(k)] = T(20) * std::log10(std::max(gain[k], T(1e-6)));
                work_[static_cast<size_t>(k)] = T(0);
            }
            publish(ch, bins);
            return;
        }
        const auto c = static_cast<size_t>(ch < 0 ? 0 : std::min(ch, static_cast<int>(smoothPower_.size()) - 1));
        auto& smooth = smoothPower_[c];
        // Power across three bins, a floor 80 dB under the frame's maximum,
        // a one-pole in time.
        T top = T(0);
        for (int k = 0; k < bins; ++k)
        {
            const T p = (power[std::max(0, k - 1)] + power[k] + power[std::min(bins - 1, k + 1)]) / T(3);
            work_[static_cast<size_t>(k)] = p;
            top = std::max(top, p);
        }
        const T floor = std::max(top * T(1e-8), T(1e-14));
        for (int k = 0; k < bins; ++k)
        {
            const T p = std::max(work_[static_cast<size_t>(k)], floor);
            smooth[static_cast<size_t>(k)] += T(0.5) * (p - smooth[static_cast<size_t>(k)]);
        }
        const bool decibels = mode_.load(std::memory_order_relaxed) == Mode::Smooth;
        for (int k = 0; k < bins; ++k)
        {
            const T p = smooth[static_cast<size_t>(k)];
            db_[static_cast<size_t>(k)] = decibels ? T(10) * std::log10(p) : std::sqrt(p);
        }
        envelope_.resonances(db_.data(), env_.data());
        const T threshold = threshold_.load(std::memory_order_relaxed);
        const T strength = strength_.load(std::memory_order_relaxed);
        const T range = range_.load(std::memory_order_relaxed);
        const T side = ms && ch == 1 ? sideWeight_.load(std::memory_order_relaxed) : T(1);
        auto& g = gainDb_[c];
        for (int k = 0; k < bins; ++k)
        {
            const auto i = static_cast<size_t>(k);
            // The excess in dB over the threshold.
            T excess;
            if (decibels) excess = env_[i];
            else
            {
                const T amp = db_[i], over = env_[i];
                excess = over > T(0) ? T(20) * std::log10(amp / std::max(amp - over, amp * T(1e-3))) : T(0);
            }
            const T found = std::max(T(0), excess - threshold) * weight_[i] * side;
            work_[i] = found;
            const T target = -std::min(range, strength * found);
            g[i] = target + (target < g[i] ? attack_[i] : release_[i]) * (g[i] - target);
        }
        // Across bins, then to gains.
        for (int k = 1; k < bins; ++k) mixed_[static_cast<size_t>(k)] = g[static_cast<size_t>(k)];
        mixed_[0] = g[0];
        AuditoryBands<T>::smoothAcross(mixed_.data(), bins, T(0.5));
        for (int k = 0; k < bins; ++k) gain[k] = std::pow(T(10), mixed_[static_cast<size_t>(k)] / T(20));
        publish(ch, bins);
    }

    /** @brief The display curves of one frame: the reduction in mixed_, what was found in work_. */
    void publish(int ch, int bins) noexcept
    {
        for (int c = ch < 0 ? 0 : ch; c <= (ch < 0 ? 1 : ch); ++c)
        {
            reduction_.publish(c, mixed_.data(), bins);
            excess_.publish(c, work_.data(), bins);
        }
    }

    AudioSpec spec_ {};
    int bins_ = 0, weightPos_ = 0;
    bool prepared_ = false, weightPending_ = false;
    T lastWidth_ = T(1.0 / 3.0);
    std::atomic<T> threshold_ { T(3) }, strength_ { T(0.7) }, range_ { T(12) }, width_ { T(1.0 / 3.0) };
    std::atomic<T> attackMs_ { T(5) }, releaseMs_ { T(80) }, linkAmount_ { T(1) }, sideWeight_ { T(1) };
    std::atomic<Mode> mode_ { Mode::Smooth };
    std::atomic<Link> link_ { Link::PerBin };
    std::atomic<Quality> quality_ { Quality::Realtime };
    std::atomic<bool> removalListen_ { false }, focusListen_ { false }, timesDirty_ { false }, weightDirty_ { true };
    SpectralFilter<T> filter_;
    SpectralReadout<T> reduction_, excess_;
    SpectralEnvelope<T> envelope_;
    KeyEqualizer<T, 2> sensitivity_;
    std::vector<std::vector<T>> smoothPower_, gainDb_;
    typename KeyEqualizer<T, 2>::Response sensResponse_ {};
    std::vector<T> work_, db_, env_, mixed_, weight_, attack_, release_;
};

} // namespace dspark
