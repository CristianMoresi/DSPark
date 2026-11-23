// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file ResonanceSuppressor.h
 * @brief A suppressor of resonances: what stands out of its neighbours and
 *        lasts is lowered by a minimum-phase filter, as fine as setDetail().
 *
 * The spectrum is read on 1024 points of log frequency (DetailAnalysis), by
 * a window that follows the detail: a third of a second in the low end, 21
 * ms at the top. A resonance is a peak that is still sounding (one that drops
 * at once is read, with what is beside it, at the level it sounds at now: one
 * that only grew softer is still taken) and stands
 * over the loudest of the peaks comparable with it, from a detail away to
 * setWidth() further or to a note's next partial where that is further (up
 * to an octave and a quarter; on a side with none of its peers within the
 * width, that stretch is read over a programme's own slope of 3 dB an
 * octave). So a
 * partial like the partials beside it, the edge of a series,
 * a slope and the grain of a noise are no resonances; a peak that is left
 * alone with nothing like it is one. A note's own fundamental (a peak with
 * its second and third harmonics over it, or under 250 Hz either of them,
 * and no series under it) is left. What carries the programme (within 6 dB
 * of the loudest, nothing within 20 dB underneath: a sub bass of a sine) is
 * taken by less the lower it is, not at all under 70 Hz and whole from 320
 * Hz, in either mode (setFoundationGuard()).
 *
 * Mode::Deep is the finer and the stronger: it reads and cuts in half the
 * detail, takes both of two resonances closer than the detail, and takes
 * up to half as deep again as Mode::Smooth. Under about 150 Hz it is no
 * narrower than Smooth and can be a fifth deeper only.
 *
 * What passes setThreshold() is weighted by the sensitivity curve
 * (setSensitivityBand()), counted as it lasts (setPersistence()), and
 * lowered by setStrength() of it (1 brings the peak down to the threshold
 * over its neighbours), by setRange() at most, with an attack and a release
 * that are slower in the low end. The cut is a detail wide and is played by
 * a DetailFilter: minimum phase, with detail in the low end too (see Limits).
 *
 * Stereo: Link::PerBin takes the level each channel is read by towards that
 * of the pair's mean power, point by point (setLink() 0 to 1: at 0 each
 * channel reads itself; in between, that share of the way, in decibels in
 * Mode::Smooth and in amplitude in Mode::Deep); at 1 both channels share the
 * detector and the filter; Link::MidSide processes the mid and the side, the
 * side's reduction scaled by setSideWeight(). setRemovalListen(true) outputs
 * what is taken out; setFocusListen(true) outputs the programme through the
 * sensitivity curve; both are monitoring controls and stay out of the state.
 * getDisplayReduction() is the curve that is applied, getDisplayExcess()
 * what was found to be a resonance, at the same 1024 points, for a display
 * on any thread.
 *
 * Limits: a sixth of an octave is told apart from 50 Hz, a twelfth from
 * 100 Hz, a 24th from 200 Hz; the finer, the later the low end follows (some
 * tenths of a second).
 *
 * Latency: 2048 samples at 44.1 and 48 kHz, doubling by octaves with the rate
 * (getLatency()).
 *
 * Threading: prepare(), setQuality() taking effect and setState() on the
 * setup thread (allocate: setQuality() applies at the next prepare());
 * processBlock() and reset() on the audio thread; the other setters are
 * lock-free atomic publications. Non-finite values are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, AuditoryBands.h, DetailAnalysis.h,
 * DetailFilter.h, DspMath.h, KeyEqualizer.h, SpectralFilter.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/AuditoryBands.h"
#include "../Core/DetailAnalysis.h"
#include "../Core/DetailFilter.h"
#include "../Core/DspMath.h"
#include "../Core/KeyEqualizer.h"
#include "../Core/SpectralFilter.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace dspark {

/**
 * @class ResonanceSuppressor
 * @brief Dynamic suppression of resonances by a minimum-phase filter on a log-frequency grid.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class ResonanceSuppressor
{
public:
    enum class Mode { Smooth, Deep };          ///< As set, or finer and stronger.
    enum class Link { PerBin, MidSide };       ///< Stereo handling.
    enum class Quality { Realtime, Offline };  ///< The long windows in turns, or all of them every hop.
    static constexpr int kSensitivityBands = 6;
    using Band = typename KeyEqualizer<T, 2>::Band;

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        spec_ = spec;
        const bool offline = quality_.load(std::memory_order_relaxed) == Quality::Offline
            || (qualityFollowsRender_.load(std::memory_order_relaxed) && offlineRendering_.load(std::memory_order_relaxed));
        qualityInUse_.store(offline ? Quality::Offline : Quality::Realtime, std::memory_order_relaxed);
        channels_ = std::min(spec.numChannels, DetailFilter<T>::kMaxChannels);
        filter_.prepare(spec);
        analysis_.prepare(spec);
        analysis_.setExhaustive(offline);
        hopSeconds_ = static_cast<double>(filter_.getHopSize()) / spec.sampleRate;

        const auto tracks = static_cast<size_t>(channels_ + 1);   // the channels, and a pair's mean
        for (auto* v : { &smoothPower_, &gainDb_, &lasted_ }) v->assign(tracks, std::vector<T>(kSize, T(0)));
        raw_.assign(2, std::vector<T>(kSize, T(0)));
        amplitude_.assign(kSize, 0.0);
        peaks_.assign(kSize, 0);
        nowLevel_.assign(kSize, T(0));
        for (auto* v : { &level_, &linked_, &found_, &weight_, &weightNext_, &attack_, &release_, &timeSmooth_, &octave_ })
            v->assign(kSize, T(0));
        std::fill(weight_.begin(), weight_.end(), T(1));
        for (auto* v : { &near_, &pick_, &apart_, &beside_, &reach_, &cut_ }) v->assign(kSize, 0);
        foundationRise_.assign(kSize, T(1));
        for (int i = 0; i < kPoints; ++i)
        {
            const double hz = DetailFilter<T>::frequency(i);
            octave_[static_cast<size_t>(i)] = static_cast<T>(std::log2(hz / 1000.0));
            const double up = std::max(0.0, std::log2(hz / kFoundationHz) / kFoundationOctaves);
            foundationRise_[static_cast<size_t>(i)] = static_cast<T>(std::min(1.0, up * up));
        }
        tiltPerPoint_ = static_cast<T>(kTiltDbPerOctave / pointsPerOctave());
        earBest_ = 1e9;
        for (double hz = 1000.0; hz <= 6000.0; hz += 100.0) earBest_ = std::min(earBest_, AuditoryScale::thresholdInQuietDb(hz));
        sensitivity_.prepare(spec.sampleRate);
        excess_.clear();
        shownCurve_ = 0;
        refreshDetector(mode_.load(std::memory_order_relaxed) == Mode::Deep);
        refreshTimes();
        weightPos_ = 0;
        weightPending_ = false;
        weightDirty_.store(true, std::memory_order_release);
        prepared_ = true;
    }

    /** @brief Clears the analysis and the filter. */
    void reset() noexcept
    {
        filter_.reset();
        analysis_.reset();
        for (auto* set : { &smoothPower_, &gainDb_, &lasted_ })
            for (auto& v : *set) std::fill(v.begin(), v.end(), T(0));
        shownCurve_ = 0;
        excess_.clear();
    }

    // -- Controls ------------------------------------------------------------

    /**
     * @brief The detail: how narrow what is found and cut is, from an octave
     *        to 1/24 of one (default 1/12). It moves while playing, at the
     *        same latency.
     */
    void setDetail(T octaves) noexcept { store(detail_, octaves, kFinestDetail, kWidestDetail); }
    /** @brief Over the loudest of its neighbours by this much a peak is a resonance, 0 to 30 dB (default 5). */
    void setThreshold(T db) noexcept { store(threshold_, db, T(0), T(30)); }
    /** @brief How much of what passes the threshold is taken out, 0 to 1 (default 1). */
    void setStrength(T amount) noexcept { store(strength_, amount, T(0), T(1)); }
    /** @brief Largest reduction, 0 to 48 dB (default 9; Mode::Deep reaches half as far again). */
    void setRange(T db) noexcept { store(range_, db, T(0), T(48)); }
    /**
     * @brief How far the neighbours a peak is compared with reach, in
     *        octaves beyond the detail, 0.05 to 4 (default 1/3). A side with
     *        no peak that near is read on to the next partial of a note.
     */
    void setWidth(T octaves) noexcept { store(width_, octaves, T(0.05), T(4)); }
    /** @brief Attack at 1 kHz, 0.5 to 500 ms (default 5; slower below, faster above, by the square root of the frequency). */
    void setAttack(T ms) noexcept { store(attackMs_, ms, T(0.5), T(500)); timesDirty_.store(true, std::memory_order_release); }
    /** @brief Release at 1 kHz, 5 to 2000 ms (default 80). */
    void setRelease(T ms) noexcept { store(releaseMs_, ms, T(5), T(2000)); timesDirty_.store(true, std::memory_order_release); }
    /**
     * @brief Smooth (the default) works at the detail set and takes what
     *        stands over the loudest of its neighbours. Deep is more of both,
     *        detail and strength: it reads and cuts in half the detail (to
     *        1/24 of an octave at the finest), takes both of two resonances
     *        closer than the detail and what stands over the mean of its
     *        peers, and takes up to half as deep again, setRange() included
     *        (under about 150 Hz no narrower, and as little as a fifth deeper).
     */
    void setMode(Mode m) noexcept { mode_.store(m == Mode::Deep ? Mode::Deep : Mode::Smooth, std::memory_order_relaxed); }
    /**
     * @brief How far what carries the programme from under it is spared, 0
     *        to 1 (default 1): what is within 6 dB of the loudest with nothing
     *        within 20 dB underneath, like a sub bass, is then not taken
     *        under 70 Hz and by more as it rises, whole from 320 Hz. At 0 it
     *        is a resonance like any other, as deep in the low end as at the
     *        top. The same in both modes. Non-finite values are ignored.
     */
    void setFoundationGuard(T amount) noexcept { store(guard_, amount, T(0), T(1)); }
    /** @brief Stereo handling (PerBin by default). */
    void setLinkMode(Link l) noexcept { link_.store(l == Link::MidSide ? Link::MidSide : Link::PerBin, std::memory_order_relaxed); }
    /** @brief Per-point link, 0 (each channel its own) to 1 (one detector and filter), default 1. */
    void setLink(T amount) noexcept { store(linkAmount_, amount, T(0), T(1)); }
    /** @brief In MidSide, the side's reduction as a share of the mid's, 0 to 2 (default 1). */
    void setSideWeight(T share) noexcept { store(sideWeight_, share, T(0), T(2)); }
    /** @brief Outputs what is taken out. */
    void setRemovalListen(bool on) noexcept { removalListen_.store(on, std::memory_order_relaxed); }
    /** @brief Outputs the programme through the sensitivity curve. */
    void setFocusListen(bool on) noexcept { focusListen_.store(on, std::memory_order_relaxed); }
    /**
     * @brief How much of the filter is applied, 0 to 1 (default 1). The mix
     *        is taken inside the filter: its gain becomes 1 - mix + mix * gain
     *        before the minimum-phase filter is designed, so the part left
     *        alone and the part reduced never meet in two phases.
     */
    void setMix(T amount) noexcept { store(mix_, amount, T(0), T(1)); }
    /**
     * @brief Soft knee at the threshold, 0 to 12 dB (0, the default, is the
     *        hard knee): the reduction grows smoothly on both sides of the
     *        threshold, so a resonance that hovers at it moves less from
     *        frame to frame.
     */
    void setKnee(T db) noexcept { store(knee_, db, T(0), T(12)); }
    /**
     * @brief Persistence, 0 to 500 ms (default 50; 0 counts at once): how
     *        long a resonance lasts before it counts, not at all until half
     *        of it has gone by and whole half of it later, however loud. A
     *        click, a consonant or the attack of a drum are over before they
     *        weigh.
     */
    void setPersistence(T ms) noexcept { store(persistenceMs_, ms, T(0), T(500)); }
    /**
     * @brief Level under which the suppressor lets go, -120 to 0 dBFS (off
     *        at -200, the default; anything under -190 turns it off). Each
     *        point is read as its amplitude plus this level, so what lies
     *        well over it is treated as before and what lies under it shows
     *        no resonance: quiet passages, tails and the noise of a room are
     *        left alone.
     */
    void setQuietLevel(T dbfs) noexcept
    {
        if (std::isfinite(dbfs)) quietDb_.store(dbfs < T(-190) ? T(-200) : std::clamp(dbfs, T(-120), T(0)), std::memory_order_relaxed);
    }
    /**
     * @brief Weighs the amount by how well the ear hears each frequency (off
     *        by default): a third of the distance in dB from the threshold of
     *        hearing in quiet to its lowest point comes off the weight, so
     *        the same excess weighs at 150 Hz under half of what it does at
     *        3 kHz.
     */
    void setAuditoryWeighting(bool on) noexcept
    {
        auditory_.store(on, std::memory_order_relaxed);
        weightDirty_.store(true, std::memory_order_release);
    }
    /**
     * @brief Narrows setWidth() above a corner: it falls as (corner /
     *        frequency) to the exponent (0, the default, keeps it everywhere;
     *        1 keeps it constant in hertz above the corner). Non-finite
     *        values are ignored.
     * @param cornerHz The corner, 200 Hz to 20 kHz.
     * @param exponent 0 to 2.
     */
    void setWidthTaper(T cornerHz, T exponent) noexcept
    {
        store(taperHz_, cornerHz, T(200), T(20000));
        store(taperExponent_, exponent, T(0), T(2));
    }
    /**
     * @brief The band setRemovalListen() plays, in hertz (0 and 0, the
     *        default, play all of it): what is taken out between the two
     *        frequencies alone, with a sixth of an octave of slope at each
     *        end. A listening aid like the listen itself: not part of the
     *        state. Non-finite values are ignored.
     */
    void setRemovalBand(T lowHz, T highHz) noexcept
    {
        if (!std::isfinite(lowHz) || !std::isfinite(highHz)) return;
        bandLow_.store(std::max(T(0), std::min(lowHz, highHz)), std::memory_order_relaxed);
        bandHigh_.store(std::max(T(0), std::max(lowHz, highHz)), std::memory_order_relaxed);
    }
    /**
     * @brief Realtime reads the long windows of the low end in turns, every
     *        two or four hops; Offline reads all of them every hop, for a
     *        render (applies at the next prepare(); the same latency).
     */
    void setQuality(Quality q) noexcept { quality_.store(q == Quality::Offline ? Quality::Offline : Quality::Realtime, std::memory_order_relaxed); }
    /**
     * @brief With it on, a prepare() made while the host renders offline
     *        takes the Offline quality whatever setQuality() says (off by
     *        default): a bounce is made at the best quality with no setting
     *        to remember. The latency is the same in both.
     */
    void setQualityFollowsRender(bool on) noexcept { qualityFollowsRender_.store(on, std::memory_order_relaxed); }
    /**
     * @brief Tells the suppressor whether the host renders offline (read by
     *        the next prepare(); a plugin forwards its host's notice here
     *        before it prepares).
     */
    void setOfflineRendering(bool on) noexcept { offlineRendering_.store(on, std::memory_order_relaxed); }
    /** @brief One band of the sensitivity curve (KeyEqualizer bands; the curve is a weight, its gain in dB / 12 + 1, clamped 0 to 2). */
    void setSensitivityBand(int index, const Band& band) noexcept
    {
        sensitivity_.setBand(index, band);
        weightDirty_.store(true, std::memory_order_release);
    }

    [[nodiscard]] T getDetail() const noexcept { return detail_.load(std::memory_order_relaxed); }

    /**
     * @brief Plays kernels of half their transform, tapered over their whole
     *        length (off by default; see DetailFilter::setShortKernel()): a
     *        cut rings for a shorter time after the event, at the same depth.
     */
    void setShortKernel(bool on) noexcept { shortKernel_.store(on, std::memory_order_relaxed); }
    /** @return Whether the kernels are short. */
    [[nodiscard]] bool getShortKernel() const noexcept { return shortKernel_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getThreshold() const noexcept { return threshold_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getStrength() const noexcept { return strength_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRange() const noexcept { return range_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWidth() const noexcept { return width_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getAttack() const noexcept { return attackMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRelease() const noexcept { return releaseMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] Mode getMode() const noexcept { return mode_.load(std::memory_order_relaxed); }
    [[nodiscard]] Link getLinkMode() const noexcept { return link_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getFoundationGuard() const noexcept { return guard_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getLink() const noexcept { return linkAmount_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getSideWeight() const noexcept { return sideWeight_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getRemovalListen() const noexcept { return removalListen_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getFocusListen() const noexcept { return focusListen_.load(std::memory_order_relaxed); }
    [[nodiscard]] Quality getQuality() const noexcept { return quality_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getQualityFollowsRender() const noexcept { return qualityFollowsRender_.load(std::memory_order_relaxed); }
    /** @return The quality the last prepare() took (setQuality(), or Offline for an offline render it follows). */
    [[nodiscard]] Quality getQualityInUse() const noexcept { return qualityInUse_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getMix() const noexcept { return mix_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getKnee() const noexcept { return knee_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getPersistence() const noexcept { return persistenceMs_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getQuietLevel() const noexcept { return quietDb_.load(std::memory_order_relaxed); }
    [[nodiscard]] bool getAuditoryWeighting() const noexcept { return auditory_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWidthTaperCorner() const noexcept { return taperHz_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getWidthTaperExponent() const noexcept { return taperExponent_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRemovalBandLow() const noexcept { return bandLow_.load(std::memory_order_relaxed); }
    [[nodiscard]] T getRemovalBandHigh() const noexcept { return bandHigh_.load(std::memory_order_relaxed); }
    [[nodiscard]] Band getSensitivityBand(int index) const noexcept { return sensitivity_.getBand(index); }
    /** @return The latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return filter_.getLatency(); }
    /** @return The reduction of the last frame at a frequency, in dB (audio thread or after processing). */
    [[nodiscard]] T getReduction(double hz) const noexcept
    {
        if (gainDb_.empty() || !(hz > 0.0)) return T(0);
        return gainDb_[0][static_cast<size_t>(std::clamp(pointOf(hz), 0, kPoints - 1))];
    }

    // -- Display (any thread) --------------------------------------------------

    /** @brief Points of the display curves, evenly in log frequency from 20 Hz to 20 kHz. */
    static constexpr int kDisplayPoints = SpectralReadout<T>::kPoints;
    /** @return The frequency of a display point, in hertz. */
    [[nodiscard]] static double getDisplayFrequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }
    /**
     * @return The reduction applied at a display point, in dB (0 or
     *         less): the response of the kernels handed over every 21 ms,
     *         made by the first read after they change. Channel 0 and 1
     *         (the mid and the side with Link::MidSide); a linked pair shows
     *         the same curve on both. With setFocusListen(true), the gain of
     *         the focus filter. Any thread.
     */
    [[nodiscard]] T getDisplayReduction(int channel, int point) const noexcept { return filter_.playedValue(channel, point); }
    /**
     * @return What the last frame found to be a resonance at a display
     *         point: what passes the threshold, weighted by the sensitivity
     *         curve and held by the persistence, in dB (0 or more), before
     *         setStrength() and setRange(). Any thread.
     */
    [[nodiscard]] T getDisplayExcess(int channel, int point) const noexcept { return excess_.value(channel, point); }
    /** @return The deepest reduction of that response, on either channel, in dB (any thread). */
    [[nodiscard]] T getPeakReduction() const noexcept { return filter_.playedExtreme(); }

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
        StateWriter w(stateId("RSUP"), 2);
        w.write("detail", static_cast<float>(getDetail()));
        w.write("shortKernel", getShortKernel());
        w.write("threshold", static_cast<float>(getThreshold()));
        w.write("strength", static_cast<float>(getStrength()));
        w.write("range", static_cast<float>(getRange()));
        w.write("width", static_cast<float>(getWidth()));
        w.write("attack", static_cast<float>(getAttack()));
        w.write("release", static_cast<float>(getRelease()));
        w.write("mode", static_cast<int32_t>(getMode()));
        w.write("foundationGuard", static_cast<float>(getFoundationGuard()));
        w.write("linkMode", static_cast<int32_t>(getLinkMode()));
        w.write("link", static_cast<float>(getLink()));
        w.write("sideWeight", static_cast<float>(getSideWeight()));
        w.write("quality", static_cast<int32_t>(getQuality()));
        w.write("qualityFollowsRender", getQualityFollowsRender());
        w.write("mix", static_cast<float>(getMix()));
        w.write("knee", static_cast<float>(getKnee()));
        w.write("persistence", static_cast<float>(getPersistence()));
        w.write("quietLevel", static_cast<float>(getQuietLevel()));
        w.write("auditory", getAuditoryWeighting());
        w.write("taperHz", static_cast<float>(getWidthTaperCorner()));
        w.write("taperExponent", static_cast<float>(getWidthTaperExponent()));
        sensitivity_.writeState(w, "sens");
        return w.blob();
    }

    /**
     * @brief Restores the settings. A blob of the first version (threshold
     *        and strength over the mean of the spectrum, where 3 dB and 0.7
     *        were the defaults) is moved onto this one's scale.
     */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("RSUP")) return false;
        const bool first = r.processorVersion() < 2;
        setDetail(static_cast<T>(r.read("detail", 1.0f / 12.0f)));
        setShortKernel(r.read("shortKernel", false));
        setThreshold(static_cast<T>(first ? r.read("threshold", 3.0f) + 2.0f : r.read("threshold", 5.0f)));
        setStrength(static_cast<T>(first ? std::min(1.0f, r.read("strength", 0.7f) / 0.7f) : r.read("strength", 1.0f)));
        setRange(static_cast<T>(r.read("range", first ? 12.0f : 9.0f)));
        setWidth(static_cast<T>(r.read("width", 1.0f / 3.0f)));
        setAttack(static_cast<T>(r.read("attack", 5.0f)));
        setRelease(static_cast<T>(r.read("release", 80.0f)));
        setMode(r.read("mode", 0) == 1 ? Mode::Deep : Mode::Smooth);
        setFoundationGuard(static_cast<T>(r.read("foundationGuard", 1.0f)));
        setLinkMode(r.read("linkMode", 0) == 1 ? Link::MidSide : Link::PerBin);
        setLink(static_cast<T>(r.read("link", 1.0f)));
        setSideWeight(static_cast<T>(r.read("sideWeight", 1.0f)));
        setQuality(r.read("quality", 0) == 1 ? Quality::Offline : Quality::Realtime);
        setQualityFollowsRender(r.read("qualityFollowsRender", false));
        setMix(static_cast<T>(r.read("mix", 1.0f)));
        setKnee(static_cast<T>(r.read("knee", 0.0f)));
        setPersistence(static_cast<T>(r.read("persistence", first ? 0.0f : 50.0f)));
        setQuietLevel(static_cast<T>(r.read("quietLevel", -200.0f)));
        setAuditoryWeighting(r.read("auditory", false));
        setWidthTaper(static_cast<T>(r.read("taperHz", 2000.0f)), static_cast<T>(r.read("taperExponent", 0.0f)));
        sensitivity_.readState(r, "sens");
        weightDirty_.store(true, std::memory_order_release);
        timesDirty_.store(true, std::memory_order_release);
        return true;
    }

private:
    static constexpr int kPoints = SpectralReadout<T>::kPoints;
    static constexpr size_t kSize = static_cast<size_t>(kPoints);
    static constexpr T kFinestDetail = static_cast<T>(DetailAnalysis<T>::kFinest);
    static constexpr T kWidestDetail = static_cast<T>(DetailAnalysis<T>::kWidest);
    static constexpr double kTiltDbPerOctave = 3.010299956639812; ///< The slope peaks are compared over.
    static constexpr double kPartialHz = 600.0;   ///< A note's partials are no further apart than this.
    static constexpr double kPartialReach = 1.25; ///< Nor counted further than this, in octaves.
    static constexpr double kShortestSmoothing = 0.04; ///< The detector's, in seconds (half a window where that is longer).
    static constexpr T kPeakOverSurroundings = T(5);   ///< A harmonic is this far over the mean around it.
    static constexpr double kLowNote = 250.0;          ///< Under this one harmonic makes a note: a mix covers the others.
    static constexpr double kFoundationHz = 70.0;      ///< What carries the programme is left under this,
    static constexpr double kFoundationOctaves = 2.2;  ///< and taken whole this far over it.
    static constexpr T kFoundationFade = T(6);         ///< The decibels over which it stops carrying it.
    static constexpr T kDeepDepth = T(1.5);            ///< Deep takes this many times as deep.
    static constexpr double kDeepApart = 1.5;          ///< And compares from this many details away.
    static constexpr T kTwin = T(3);                   ///< Of two closer than the detail, how unlike both may be.
    static constexpr T kDeepFloor = T(20);             ///< Deep averages the neighbours this far under the loudest.
    static constexpr T kPeerUnder = T(20);             ///< A neighbour this far under a peak is none of its peers.
    static constexpr T kStillSounding = T(0.1);        ///< Of the smoothed power, what the last frame alone must have.

    static void store(std::atomic<T>& a, T v, T lo, T hi) noexcept
    {
        if (std::isfinite(v)) a.store(std::clamp(v, lo, hi), std::memory_order_relaxed);
    }

    [[nodiscard]] static int pointOf(double hz) noexcept
    {
        return static_cast<int>(std::lround((kPoints - 1) * std::log(std::max(hz, 1.0) / 20.0) / std::log(1000.0)));
    }

    [[nodiscard]] static double pointsPerOctave() noexcept
    {
        return (kPoints - 1) * std::numbers::ln2 / std::log(1000.0);
    }

    /** @brief What each point leaves out, what it is compared with, and how long it is averaged. */
    void refreshDetector(bool deep) noexcept
    {
        lastDeep_ = deep;
        lastDetail_ = detail_.load(std::memory_order_relaxed);
        const double fine = deep ? std::max(static_cast<double>(kFinestDetail), 0.5 * static_cast<double>(lastDetail_))
                                 : static_cast<double>(lastDetail_);
        lastWidth_ = width_.load(std::memory_order_relaxed);
        lastTaperHz_ = taperHz_.load(std::memory_order_relaxed);
        lastTaperExponent_ = taperExponent_.load(std::memory_order_relaxed);
        analysis_.setWidth(fine);
        const double perOctave = pointsPerOctave();
        for (int i = 0; i < kPoints; ++i)
        {
            const auto u = static_cast<size_t>(i);
            const double hz = DetailFilter<T>::frequency(i), window = std::max(1e-3, analysis_.getWindowSeconds(i));
            // A line's own lobe, where it is wider than the detail.
            const double lobe = std::log2(1.0 + 2.0 / (window * hz));
            const double own = std::max(static_cast<double>(lastDetail_), lobe);
            double width = static_cast<double>(lastWidth_);
            if (lastTaperExponent_ > T(0) && hz > static_cast<double>(lastTaperHz_))
                width *= std::pow(static_cast<double>(lastTaperHz_) / hz, static_cast<double>(lastTaperExponent_));
            const double onward = std::max(width, std::min(kPartialReach, std::log2(1.0 + kPartialHz / hz)));
            near_[u] = std::max(3, static_cast<int>(std::ceil(own * perOctave)));
            pick_[u] = deep ? std::min(near_[u], std::max(2, static_cast<int>(std::ceil(std::max(fine, lobe) * perOctave)))) : near_[u];
            beside_[u] = near_[u] + std::max(2, static_cast<int>(std::ceil(width * perOctave)));
            reach_[u] = std::max(beside_[u], near_[u] + static_cast<int>(std::ceil(onward * perOctave)));
            // Deep leaves a twin's skirt out of what a peak is compared with.
            apart_[u] = deep ? std::min(beside_[u] - 1, static_cast<int>(std::ceil(kDeepApart * near_[u]))) : near_[u];
            // The cut: the detail in use across at half its depth, no narrower than the filter plays whole.
            cut_[u] = std::max({ 3, static_cast<int>(std::ceil(0.9 * fine * perOctave)),
                                 static_cast<int>(std::ceil(1.2 * lobe * perOctave)) });
            timeSmooth_[u] = static_cast<T>(1.0 - std::exp(-hopSeconds_ / std::max(kShortestSmoothing, 0.5 * window)));
        }
    }

    /** @brief Attack and release per point: slower below 1 kHz, faster above, by the square root of the frequency. */
    void refreshTimes() noexcept
    {
        const double a = 0.001 * static_cast<double>(attackMs_.load(std::memory_order_relaxed));
        const double r = 0.001 * static_cast<double>(releaseMs_.load(std::memory_order_relaxed));
        for (int i = 0; i < kPoints; ++i)
        {
            const double scale = std::exp2(-0.5 * static_cast<double>(octave_[static_cast<size_t>(i)]));
            attack_[static_cast<size_t>(i)] = static_cast<T>(std::exp(-hopSeconds_ / std::max(1e-4, a * scale)));
            release_[static_cast<size_t>(i)] = static_cast<T>(std::exp(-hopSeconds_ / std::max(1e-3, r * scale)));
        }
    }

    /**
     * @brief A share of the sensitivity curve per call (the whole curve over
     *        eight), computed aside and taken whole once it is complete: no
     *        frame is weighed by a curve that is part old and part new.
     */
    void refreshWeight() noexcept
    {
        if (weightDirty_.exchange(false, std::memory_order_acquire))
        {
            sensResponse_ = sensitivity_.getResponse();
            weightAuditory_ = auditory_.load(std::memory_order_relaxed);
            weightPos_ = 0;
            weightPending_ = true;
        }
        if (!weightPending_) return;
        const int end = std::min(kPoints, weightPos_ + (kPoints + 7) / 8);
        for (int i = weightPos_; i < end; ++i)
        {
            const double hz = DetailFilter<T>::frequency(i);
            double w = std::clamp(1.0 + sensResponse_.magnitudeDb(hz) / 12.0, 0.0, 2.0);
            if (weightAuditory_)
                w *= std::clamp(std::pow(10.0, -(AuditoryScale::thresholdInQuietDb(hz) - earBest_) / 60.0), 0.05, 1.0);
            weightNext_[static_cast<size_t>(i)] = static_cast<T>(w);
        }
        weightPos_ = end;
        if (weightPos_ >= kPoints)
        {
            weight_.swap(weightNext_);
            weightPending_ = false;
        }
    }

    void run(AudioBufferView<T>& buffer, const AudioBufferView<T>* key) noexcept
    {
        if (!prepared_) return;
        const int nCh = std::min(buffer.getNumChannels(), channels_);
        const int nS = buffer.getNumSamples();
        if (nCh <= 0 || nS <= 0) return;
        if (key != nullptr)
        {
            // A key that is the programme itself is read as no key.
            bool same = true;
            for (int ch = 0; ch < nCh && same; ++ch)
                same = key->getChannel(std::min(ch, key->getNumChannels() - 1)) == buffer.getChannel(ch);
            if (same) key = nullptr;
        }
        const bool deep = mode_.load(std::memory_order_relaxed) == Mode::Deep;
        if (deep != lastDeep_ || detail_.load(std::memory_order_relaxed) != lastDetail_ || width_.load(std::memory_order_relaxed) != lastWidth_
            || taperHz_.load(std::memory_order_relaxed) != lastTaperHz_ || taperExponent_.load(std::memory_order_relaxed) != lastTaperExponent_)
            refreshDetector(deep);
        if (timesDirty_.exchange(false, std::memory_order_acquire)) refreshTimes();
        refreshWeight();

        // The settings of this block's frames.
        Settings now;
        now.midSide = nCh == 2 && link_.load(std::memory_order_relaxed) == Link::MidSide;
        const T linkAmount = linkAmount_.load(std::memory_order_relaxed);
        const bool pairAsOne = !now.midSide && nCh == 2 && linkAmount >= T(1);
        // In between, each channel keeps its own filter and is read that
        // share of the way to the mean of the two.
        now.share = !now.midSide && nCh == 2 && linkAmount > T(0) && linkAmount < T(1) ? linkAmount : T(0);
        now.channels = nCh;
        now.shared = pairAsOne;
        now.deep = deep;
        now.guard = guard_.load(std::memory_order_relaxed);
        now.depth = deep ? kDeepDepth : T(1);
        now.focus = focusListen_.load(std::memory_order_relaxed);
        now.threshold = threshold_.load(std::memory_order_relaxed);
        now.strength = strength_.load(std::memory_order_relaxed);
        now.range = range_.load(std::memory_order_relaxed);
        now.knee = knee_.load(std::memory_order_relaxed);
        now.side = sideWeight_.load(std::memory_order_relaxed);
        now.mix = mix_.load(std::memory_order_relaxed);
        const bool removal = removalListen_.load(std::memory_order_relaxed);
        now.bandHigh = removal ? bandHigh_.load(std::memory_order_relaxed) : T(0);
        if (now.bandHigh > T(0))
        {
            const T low = bandLow_.load(std::memory_order_relaxed);
            now.bandFrom = low > T(0) ? std::log2(low / T(1000)) : T(-1e6);
            now.bandTo = std::log2(now.bandHigh / T(1000));
        }
        now.persist = static_cast<T>(0.001 * static_cast<double>(persistenceMs_.load(std::memory_order_relaxed)) / hopSeconds_);
        const T quiet = quietDb_.load(std::memory_order_relaxed);
        now.quiet = quiet > T(-190) ? std::pow(T(10), quiet / T(20)) : T(0);

        filter_.setLinked(pairAsOne);
        filter_.setRemovalListen(removal);
        analysis_.setLinked(pairAsOne);
        filter_.setShortKernel(shortKernel_.load(std::memory_order_relaxed));
        if (now.midSide)
        {
            T* l = buffer.getChannel(0);
            T* r = buffer.getChannel(1);
            for (int i = 0; i < nS; ++i) { const T m = T(0.5) * (l[i] + r[i]), s = T(0.5) * (l[i] - r[i]); l[i] = m; r[i] = s; }
        }
        // A key replaces the programme in what the detector reads (its mid
        // and side in MidSide are read as its first two channels).
        filter_.processBlock(
            buffer,
            [&](int offset, int count) noexcept {
                for (int ch = 0; ch < nCh; ++ch)
                {
                    const T* from = key != nullptr ? key->getChannel(std::min(ch, key->getNumChannels() - 1)) : buffer.getChannel(ch);
                    analysis_.feed(ch, from + offset, count);
                }
            },
            [&](int ch, T* curveDb) noexcept {
                if (ch == 0) analysis_.update(nCh);
                design(ch, curveDb, now);
            });
        showReduction(now);
        if (now.midSide)
        {
            T* l = buffer.getChannel(0);
            T* r = buffer.getChannel(1);
            for (int i = 0; i < nS; ++i) { const T m = l[i], s = r[i]; l[i] = m + s; r[i] = m - s; }
        }
    }

    struct Settings
    {
        int channels = 0;
        bool midSide = false, shared = false, deep = false, focus = false;
        T share = T(0), threshold = T(0), strength = T(0), range = T(0), knee = T(0), side = T(1), mix = T(1);
        T persist = T(0), quiet = T(0), bandHigh = T(0), bandFrom = T(0), bandTo = T(0);
        T guard = T(1), depth = T(1);
    };

    /** @brief The curve of one hop of a channel (channel 0 for a pair as one). */
    void design(int ch, T* curve, const Settings& now) noexcept
    {
        if (now.focus)
        {
            for (int i = 0; i < kPoints; ++i)
            {
                curve[i] = T(20) * std::log10(std::max(weight_[static_cast<size_t>(i)] * T(0.5), T(1e-6)));
                found_[static_cast<size_t>(i)] = T(0);
            }
            publishExcess(ch, now);
            return;
        }
        const auto c = static_cast<size_t>(ch);
        const bool linking = now.share > T(0);
        if (linking && ch == 0)
        {
            analysis_.read(0, raw_[0].data());
            analysis_.read(1, raw_[1].data());
            for (size_t i = 0; i < kSize; ++i) found_[i] = T(0.5) * (raw_[0][i] + raw_[1][i]);
            levels(found_.data(), smoothPower_[static_cast<size_t>(channels_)], linked_.data(), now.quiet);
        }
        if (!linking) analysis_.read(ch, raw_[0].data());
        const T* power = linking ? raw_[c].data() : raw_[0].data();
        levels(power, smoothPower_[c], level_.data(), now.quiet);
        if (linking)
            for (size_t i = 0; i < kSize; ++i)
            {
                // A share of the way: in decibels, or in amplitude in Deep.
                T& own = level_[i];
                if (!now.deep) own += now.share * (linked_[i] - own);
                else
                {
                    const T a = std::pow(T(10), own / T(20)), b = std::pow(T(10), linked_[i] / T(20));
                    own = T(20) * std::log10(a + now.share * (b - a));
                }
            }

        // Linked, what still sounds on either channel does.
        const T* sounding = power;
        const T* smoothed = smoothPower_[c].data();
        const T* other = linking ? raw_[1 - c].data() : nullptr;
        findResonances(now, sounding, other, smoothed);

        const T side = now.midSide && ch == 1 ? now.side : T(1);
        auto& g = gainDb_[c];
        auto& lasted = lasted_[c];
        for (int i = 0; i < kPoints; ++i)
        {
            const auto u = static_cast<size_t>(i);
            T found = found_[u];
            if (now.persist > T(0))
            {
                // How many hops it has lasted; one missed costs two.
                T& age = lasted[u];
                age = found > T(0) ? std::min(age + T(1), T(2) * now.persist + T(1)) : std::max(age - T(2), T(0));
                const T x = std::clamp(age / now.persist - T(0.5), T(0), T(1));
                found *= x * x * (T(3) - T(2) * x);
            }
            found *= weight_[u] * side;
            found_[u] = found;
            const T target = -now.depth * std::min(now.range, now.strength * found);
            g[u] = target + (target < g[u] ? attack_[u] : release_[u]) * (g[u] - target);
            if (target == T(0) && g[u] > T(-1e-4)) g[u] = T(0);
            curve[i] = g[u];
        }
        // The mix, inside the filter: the curve shown is the one applied.
        if (now.mix < T(1))
            for (int i = 0; i < kPoints; ++i)
                if (curve[i] != T(0))
                    curve[i] = T(20) * std::log10(std::max(T(1) - now.mix + now.mix * std::pow(T(10), curve[i] / T(20)), T(1e-6)));
        // The band the removal is heard in: outside it the filter is left at one.
        if (now.bandHigh > T(0))
            for (int i = 0; i < kPoints; ++i)
            {
                const T at = octave_[static_cast<size_t>(i)];
                const T edge = std::clamp(T(0.5) + T(6) * std::min(at - now.bandFrom, now.bandTo - at), T(0), T(1)); // a sixth of an octave at each end
                const T inside = T(0.5) - T(0.5) * std::cos(std::numbers::pi_v<T> * edge);
                if (curve[i] != T(0))
                    curve[i] = T(20) * std::log10(std::max(T(1) - inside * (T(1) - std::pow(T(10), curve[i] / T(20))), T(1e-6)));
            }
        publishExcess(ch, now);
    }

    /**
     * @brief The level of each point as the detector reads it: a one-pole in
     *        time over a floor 80 dB under the frame's maximum, in decibels
     *        (of the amplitude plus the quiet level, when there is one).
     */
    void levels(const T* power, std::vector<T>& smooth, T* out, T quiet) noexcept
    {
        T top = T(0);
        for (size_t i = 0; i < kSize; ++i)
        {
            const T p = std::isfinite(power[i]) ? power[i] : T(0);
            smooth[i] += timeSmooth_[i] * (p - smooth[i]);
            top = std::max(top, smooth[i]);
        }
        floor_ = std::max(top * T(1e-8), T(1e-14));
        for (size_t i = 0; i < kSize; ++i) out[i] = levelOf(smooth[i], quiet);
    }

    /** @brief A power as the detector reads it (over the floor of the last levels()). */
    [[nodiscard]] T levelOf(T power, T quiet) const noexcept
    {
        const T p = std::max(power, floor_);
        return quiet > T(0) ? T(8.6858896380650366) * std::log(std::sqrt(p) + quiet) : T(4.3429448190325183) * std::log(p);
    }

    /**
     * @brief What a peak at a point is compared with, at the levels `lv`: the
     *        loudest beside it (on a side with no peer that near, on to the
     *        first peer, over the programme's own slope) and, in Deep, never
     *        over the mean in amplitude of its peers. -1000 when there is
     *        none. `amp` holds the amplitudes of `lv` (or nullptr to take them).
     */
    T overOf(int i, const T* lv, T at, T least, bool deep, const double* amp, int peaks) const noexcept
    {
        const auto u = static_cast<size_t>(i);
        const int apart = apart_[u], reach = reach_[u], beside = beside_[u];
        T loudest = T(-1000);
        for (const int side : { -1, 1 })
        {
            if (deep)
            {
                // Deep reads the whole reach: the nearer side's largest and
                // whether it holds a peer, then the farther side's over the
                // slope unless it did (the same largest in any order).
                const int last = std::min(reach, side < 0 ? i : kPoints - 1 - i);
                const int nearEnd = std::min(beside, last);
                bool peer = false;
                for (int d = apart + 1; d <= nearEnd; ++d)
                {
                    const int j = i + side * d;
                    const T v = lv[j];
                    loudest = std::max(loudest, v);
                    peer = peer || (v >= at - kPeerUnder && j > 0 && j < kPoints - 1 && v >= lv[j - 1] && v >= lv[j + 1]);
                }
                const int farFrom = std::max(apart, beside) + 1;
                if (peer)
                    for (int d = farFrom; d <= last; ++d) loudest = std::max(loudest, lv[i + side * d]);
                else
                    for (int d = farFrom; d <= last; ++d)
                    {
                        const int j = i + side * d;
                        loudest = std::max(loudest, lv[j] + tiltPerPoint_ * static_cast<T>(j - i));
                    }
                continue;
            }
            bool peer = false;
            for (int d = apart + 1; d <= reach; ++d)
            {
                const int j = i + side * d;
                if (j < 0 || j >= kPoints) break;
                const T v = lv[j];
                const bool onward = d > beside;
                loudest = std::max(loudest, onward && !peer ? v + tiltPerPoint_ * static_cast<T>(j - i) : v);
                if (at - loudest <= least) break;
                peer = peer || (!onward && v >= at - kPeerUnder && j > 0 && j < kPoints - 1 && v >= lv[j - 1] && v >= lv[j + 1]);
            }
        }
        if (loudest <= T(-1000) || !deep) return loudest;
        double sum = 0.0;
        int count = 0;
        const int first = std::max(1, i - reach), last = std::min(kPoints - 2, i + reach);
        const int* listed = peaks_.data();
        for (int n = static_cast<int>(std::lower_bound(listed, listed + peaks, first) - listed); n < peaks && listed[n] <= last; ++n)
        {
            const int j = listed[n];
            if ((j >= i - apart && j <= i + apart) || lv[j] < loudest - kDeepFloor) continue;
            sum += amp != nullptr ? amp[static_cast<size_t>(j)] : std::exp(static_cast<double>(lv[j]) * 0.11512925464970229);
            ++count;
        }
        return count > 0 ? std::min(loudest, static_cast<T>(20.0 * std::log10(sum / count))) : loudest; // never over Smooth's
    }

    /** @brief What passes the threshold, a detail wide around each resonance, in found_. */
    void findResonances(const Settings& now, const T* sounding, const T* other, const T* smoothed) noexcept
    {
        std::fill(found_.begin(), found_.end(), T(0));
        const T* level = level_.data();
        // Deep averages peers in amplitude: each point's once, not once a peak it stands beside.
        int peaks = 0;
        if (now.deep)
        {
            for (size_t j = 0; j < kSize; ++j) amplitude_[j] = std::exp(static_cast<double>(level[j]) * 0.11512925464970229); // ln(10) / 20
            for (int j = 1; j < kPoints - 1; ++j)
                if (level[j] > level[j - 1] && !(level[j] < level[j + 1])) peaks_[static_cast<size_t>(peaks++)] = j;
        }
        const T least = now.knee > T(0) ? now.threshold - T(4) * now.knee : now.threshold;
        bool nowReady = false;
        for (int i = 1; i < kPoints - 1; ++i)
        {
            const T at = level[i];
            if (!(at > level[i - 1]) || at < level[i + 1]) continue;
            const auto u = static_cast<size_t>(i);
            const int own = near_[u], pick = pick_[u];
            bool out = true;
            for (int d = 2; d <= own && out; ++d)
            {
                // Within a detail Deep allows a twin nearly as loud.
                const T top = d > pick ? at + kTwin : at;
                out = (i - d < 0 || level[i - d] < top) && (i + d >= kPoints || level[i + d] <= top);
            }
            if (!out) continue;
            // It ends with its sound, not with the smoothing's tail: after a
            // sudden drop it is read, with what is beside it, at the levels
            // they sound at now, so what only grew softer is still taken by
            // what it still stands out.
            T latest = T(0);
            for (int j = std::max(0, i - own / 2); j <= std::min(kPoints - 1, i + own / 2); ++j)
                latest = std::max(latest, other != nullptr ? std::max(sounding[j], other[j]) : sounding[j]);
            T heard = at;
            const T* read = level;
            if (latest < kStillSounding * smoothed[i])
            {
                if (!nowReady)
                {
                    for (int j = 0; j < kPoints; ++j)
                    {
                        const T p = other != nullptr ? std::max(sounding[j], other[j]) : sounding[j];
                        nowLevel_[static_cast<size_t>(j)] = std::min(level[j], levelOf(p, now.quiet));
                    }
                    nowReady = true;
                }
                heard = std::min(at, levelOf(latest, now.quiet));
                read = nowLevel_.data();
            }
            const T over = overOf(i, read, heard, least, now.deep, read == level ? amplitude_.data() : nullptr, peaks);
            if (over <= T(-1000)) continue;
            const T excess = heard - over;
            if (excess <= least) continue;
            if (isFundamental(i)) continue;
            const T past = excess - now.threshold;
            T found = now.knee > T(0) ? T(0.5) * (past + std::sqrt(past * past + now.knee * now.knee)) : past;
            const T share = foundationShare(i, now.guard);
            if (share < T(1))
            {
                // A share of what would be taken: the range does not undo it.
                if (now.strength * found > now.range) found = now.range / now.strength;
                found *= share;
                if (found <= T(0)) continue;
            }
            // A detail wide at half its depth and round: an edge steeper than
            // the filter's bins would ring past the range.
            const int half = cut_[u];
            const T flat = T(1.5);
            for (int j = std::max(0, i - half); j <= std::min(kPoints - 1, i + half); ++j)
            {
                const T d = static_cast<T>(j < i ? i - j : j - i);
                const T shape = d <= flat ? T(1) : T(0.5) + T(0.5) * std::cos(std::numbers::pi_v<T> * (d - flat) / (static_cast<T>(half) + T(1) - flat));
                found_[static_cast<size_t>(j)] = std::max(found_[static_cast<size_t>(j)], found * shape);
            }
        }
    }

    /** @return Whether a peak of the level stands at a point, over what is around it. */
    [[nodiscard]] bool hasPeak(double point) const noexcept
    {
        const int centre = static_cast<int>(std::lround(point));
        if (centre < 1 || centre > kPoints - 2) return false;
        const T* level = level_.data();
        const int own = near_[static_cast<size_t>(centre)], slack = std::max(2, own / 2);
        const int first = std::max(1, centre - slack), last = std::min(kPoints - 2, centre + slack);
        int at = -1;
        for (int j = first; j <= last; ++j)
            if (level[j] >= level[j - 1] && level[j] >= level[j + 1] && (at < 0 || level[j] > level[at])) at = j;
        if (at < 0) return false;
        T sum = T(0);
        int count = 0;
        for (int d = own + 1; d <= own + kSurroundings; ++d)
        {
            if (at - d >= 0) { sum += level[at - d]; ++count; }
            if (at + d < kPoints) { sum += level[at + d]; ++count; }
        }
        return count > 0 && level[at] - sum / static_cast<T>(count) >= kPeakOverSurroundings;
    }

    /**
     * @return Whether a peak is the first partial of a note: its second and
     *         third harmonics are there (either of them in the low end), and
     *         it is no multiple of a partial under it (a fundamental at f / n
     *         and the partial next under, at f (n - 1) / n).
     */
    [[nodiscard]] bool isFundamental(int point) const noexcept
    {
        const double perOctave = pointsPerOctave(), at = static_cast<double>(point);
        const bool second = hasPeak(at + perOctave);
        const bool low = DetailFilter<T>::frequency(point) < kLowNote;
        if (!second && !low) return false;
        if (!(second && low) && !hasPeak(at + perOctave * std::log2(3.0))) return false;
        for (int n = 2; n <= kMostSeries; ++n)
        {
            const double root = at - perOctave * std::log2(static_cast<double>(n));
            if (root < 1.0) break;
            if (!hasPeak(root)) continue;
            if (n == 2 || hasPeak(at + perOctave * std::log2((n - 1.0) / n))) return false;
        }
        return true;
    }

    /**
     * @return How much of a peak is taken for where it stands: less the lower
     *         it is as far as it carries the programme (nothing within 20 dB
     *         underneath and nothing 6 dB louder over it: a sub bass of one sine,
     *         which has no harmonic to say it is a note), all of it otherwise.
     */
    [[nodiscard]] T foundationShare(int point, T guard) const noexcept
    {
        const T rise = foundationRise_[static_cast<size_t>(point)];
        if (rise >= T(1) || guard <= T(0)) return T(1);
        const T* level = level_.data();
        const T at = level[point];
        const int underneath = point - near_[static_cast<size_t>(point)];
        // A component underneath, not the skirt of this one (a kick is wide).
        T under = T(-1000), above = T(-1000);
        for (int j = 1; j < underneath; ++j)
            if (level[j] >= level[j - 1] && level[j] >= level[j + 1]) under = std::max(under, level[j]);
        const T alone = std::clamp((at - under - kPeerUnder) / kFoundationFade + T(1), T(0), T(1));
        if (alone <= T(0)) return T(1);
        for (int j = point + 1; j < kPoints; ++j) above = std::max(above, level[j]);
        const T loudest = std::clamp(T(2) - (above - at) / kFoundationFade, T(0), T(1));
        return T(1) - guard * alone * loudest * (T(1) - rise);
    }

    /** @brief What one hop found, for the display. */
    void publishExcess(int ch, const Settings& now) noexcept
    {
        if (ch >= SpectralReadout<T>::kChannels) return;
        excess_.publishPoints(ch, found_.data());
        if (ch == 0 && (now.shared || now.channels == 1)) excess_.publishPoints(1, found_.data());
    }

    /** @brief The reduction as the filter plays it, once a block that has a new curve. */
    void showReduction(const Settings& now) noexcept
    {
        if (!filter_.isDisplayDue(shownCurve_)) return;
        shownCurve_ = filter_.getCurveCount();
        filter_.publishPlayed(now.shared ? 1 : now.channels);
    }

    static constexpr int kSurroundings = 24; ///< Points each side that a harmonic stands over.
    static constexpr int kMostSeries = 16;   ///< The highest multiple looked for under a peak.

    AudioSpec spec_ {};
    int weightPos_ = 0, channels_ = 0, shownCurve_ = 0;
    bool prepared_ = false, weightPending_ = false, weightAuditory_ = false, lastDeep_ = false;
    double hopSeconds_ = 256.0 / 48000.0;
    double earBest_ = 0.0;                     ///< The lowest threshold of hearing, in dB.
    T tiltPerPoint_ = T(0);
    T floor_ = T(1e-14);                       ///< The detector's floor of the last levels().
    T lastDetail_ = T(1.0 / 12.0), lastWidth_ = T(1.0 / 3.0), lastTaperHz_ = T(2000), lastTaperExponent_ = T(0);
    std::atomic<T> detail_ { T(1.0 / 12.0) };
    std::atomic<bool> shortKernel_ { false };
    std::atomic<T> threshold_ { T(5) }, strength_ { T(1) }, range_ { T(9) }, width_ { T(1.0 / 3.0) };
    std::atomic<T> attackMs_ { T(5) }, releaseMs_ { T(80) }, linkAmount_ { T(1) }, sideWeight_ { T(1) };
    std::atomic<Mode> mode_ { Mode::Smooth };
    std::atomic<T> guard_ { T(1) };
    std::atomic<Link> link_ { Link::PerBin };
    std::atomic<Quality> quality_ { Quality::Realtime }, qualityInUse_ { Quality::Realtime };
    std::atomic<bool> qualityFollowsRender_ { false }, offlineRendering_ { false };
    std::atomic<bool> removalListen_ { false }, focusListen_ { false }, timesDirty_ { false }, weightDirty_ { true };
    std::atomic<T> mix_ { T(1) }, knee_ { T(0) }, persistenceMs_ { T(50) }, quietDb_ { T(-200) };
    std::atomic<bool> auditory_ { false };
    std::atomic<T> taperHz_ { T(2000) }, taperExponent_ { T(0) }, bandLow_ { T(0) }, bandHigh_ { T(0) };
    DetailFilter<T> filter_;
    DetailAnalysis<T> analysis_;
    SpectralReadout<T> excess_;
    KeyEqualizer<T, 2> sensitivity_;
    typename KeyEqualizer<T, 2>::Response sensResponse_ {};
    std::vector<std::vector<T>> smoothPower_, gainDb_, raw_;
    std::vector<std::vector<T>> lasted_; ///< Hops what is found at each point has lasted.
    std::vector<T> level_, linked_, found_, weight_, weightNext_, attack_, release_, timeSmooth_, octave_;
    std::vector<T> nowLevel_;                  ///< The levels at what they sound now, for a peak that drops at once.
    std::vector<double> amplitude_;          ///< Deep: each point's level as an amplitude.
    std::vector<int> peaks_;                 ///< Deep: where the level has a peak, in order.
    std::vector<int> near_, beside_, reach_; ///< Points a peak leaves out around itself, how far its neighbours reach, and the next partial.
    std::vector<int> cut_;                   ///< Half the width of a cut, in points.
    std::vector<int> pick_, apart_;          ///< How near two peaks are two, and from where a peak is compared (Deep: not the detail).
    std::vector<T> foundationRise_;          ///< What is taken of what carries the programme, per point.
};

} // namespace dspark
