// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralDynamicEQ.h
 * @brief Dynamic bands that act frequency by frequency inside themselves, as
 *        fine as setDetail(), applied as a minimum-phase filter.
 *
 * A dynamic bell set on a resonance acts on the whole bell when only one
 * frequency in it is loud. Here a band (a bell or a shelf, with its
 * frequency, Q and range) says where, and the detail says how finely: the
 * band covers the frequencies where its static response at full range (its
 * analog prototype's, so it keeps its width up to the top) is over one
 * percent of that range, and inside it every point of a log-frequency grid
 * of 1024 has its own level, read by a window that follows the detail
 * (DetailAnalysis: a third of a second in the low end, 21 ms at the top),
 * held over an eighth of the detail so that what moves keeps its reduction,
 * and weighted by the band's shape. The threshold is relative to the band's
 * own recent peak (it rises at once and falls by 2 dB per second), so it
 * follows the programme's level; over it a point's action is the excess
 * times (1 - 1 / ratio), limited by the range, with an attack and a release
 * that are faster up the band (by the square root of the frequency). The
 * actions times the band's shape stay between flat and its full range, and
 * the bands add in decibels. A DetailFilter plays that curve: minimum phase,
 * with detail in the low end too (see Limits), and the curve shown is that one.
 *
 * When a band is edited at work each frequency keeps what the band was
 * doing there; if its span moved, a second reference of level listens to the
 * new span for half a second before it replaces the first.
 *
 * Limits: a sixth of an octave is told apart from 50 Hz, a twelfth from
 * 100 Hz, a 24th from 200 Hz; the finer, the later the low end follows (some
 * tenths of a second).
 *
 * Latency: 2048 samples at 44.1 and 48 kHz, doubling by octaves with the rate
 * (getLatency()).
 *
 * Threading: prepare() and setState() on the setup thread (allocate);
 * processBlock() and reset() on the audio thread; setBand() and setDetail()
 * publish lock-free (the audio thread takes them at its next block).
 * getDisplayGain() and getPeakGain() read, from any thread, the curve last
 * drawn (one every 21 ms): a short action shows there up to that much late
 * or long. Non-finite fields are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, DetailAnalysis.h, DetailFilter.h,
 * DspMath.h, FilterDesign.h, SpectralFilter.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/DetailAnalysis.h"
#include "../Core/DetailFilter.h"
#include "../Core/DspMath.h"
#include "../Core/FilterDesign.h"
#include "../Core/SpectralFilter.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class SpectralDynamicEQ
 * @brief Dynamic equalizer bands with a level at every frequency inside each band.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SpectralDynamicEQ
{
public:
    static constexpr int kBands = 4; ///< Bands.

    /** @brief The shape of a band. */
    enum class Shape { Off, Bell, LowShelf, HighShelf };

    /** @brief The settings of a band. */
    struct Band
    {
        Shape shape = Shape::Off;
        T frequency = T(1000);   ///< Centre or corner, 20 Hz to 0.45 of the rate.
        T q = T(1);              ///< Width of the bell, 0.1 to 20.
        T threshold = T(-12);    ///< Under the band's recent peak, -60 to 0 dB.
        T ratio = T(3);          ///< 1 to 20.
        T range = T(-12);        ///< Most action, -24 (cut) to +24 dB.
        T attack = T(10);        ///< At 1 kHz, 0.5 to 500 ms.
        T release = T(150);      ///< At 1 kHz, 5 to 2000 ms.
    };

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        spec_ = spec;
        channels_ = std::min(spec.numChannels, DetailFilter<T>::kMaxChannels);
        filter_.prepare(spec);
        filter_.setLinked(true);
        analysis_.setLinked(true);
        analysis_.prepare(spec);
        hopSeconds_ = static_cast<double>(filter_.getHopSize()) / spec.sampleRate;
        for (auto& s : state_)
        {
            for (auto* v : { &s.share, &s.shareDb, &s.action, &s.attack, &s.release }) v->assign(kSize, T(0));
            s.peak = kSilent;
            s.active = false;
            s.first = s.end = 0;
        }
        for (auto* v : { &power_, &level_, &held_, &total_ }) v->assign(kSize, T(0));
        sum_.assign(kSize + 1, 0.0);
        queue_.assign(kSize, 0);
        lastDetail_ = T(-1);   // the detail is taken at the first block
        shownCurve_ = 0;
        for (auto& d : dirty_) d.store(true, std::memory_order_release);
        prepared_ = true;
    }

    /** @brief Clears the analysis, the actions and the filter. */
    void reset() noexcept
    {
        filter_.reset();
        analysis_.reset();
        for (auto& s : state_)
        {
            std::fill(s.action.begin(), s.action.end(), T(0));
            s.peak = kSilent;
            s.watching = 0;
        }
        shownCurve_ = 0;
    }

    /** @brief Sets a band (out-of-range indices are ignored; fields are clamped). */
    void setBand(int index, const Band& band) noexcept
    {
        if (index < 0 || index >= kBands) return;
        auto& p = published_[static_cast<size_t>(index)];
        p.shape.store(std::clamp(static_cast<int>(band.shape), 0, 3), std::memory_order_relaxed);
        const auto put = [](std::atomic<T>& a, T v, T lo, T hi) { if (std::isfinite(v)) a.store(std::clamp(v, lo, hi), std::memory_order_relaxed); };
        put(p.frequency, band.frequency, T(20), T(24000));
        put(p.q, band.q, T(0.1), T(20));
        put(p.threshold, band.threshold, T(-60), T(0));
        put(p.ratio, band.ratio, T(1), T(20));
        put(p.range, band.range, T(-24), T(24));
        put(p.attack, band.attack, T(0.5), T(500));
        put(p.release, band.release, T(5), T(2000));
        dirty_[static_cast<size_t>(index)].store(true, std::memory_order_release);
    }

    /** @return The settings of a band. */
    [[nodiscard]] Band getBand(int index) const noexcept
    {
        Band b;
        if (index < 0 || index >= kBands) return b;
        const auto& p = published_[static_cast<size_t>(index)];
        b.shape = static_cast<Shape>(p.shape.load(std::memory_order_relaxed));
        b.frequency = p.frequency.load(std::memory_order_relaxed);
        b.q = p.q.load(std::memory_order_relaxed);
        b.threshold = p.threshold.load(std::memory_order_relaxed);
        b.ratio = p.ratio.load(std::memory_order_relaxed);
        b.range = p.range.load(std::memory_order_relaxed);
        b.attack = p.attack.load(std::memory_order_relaxed);
        b.release = p.release.load(std::memory_order_relaxed);
        return b;
    }

    /**
     * @brief The detail: how finely the bands move inside themselves, from
     *        an octave to 1/24 of one (default 1/12). It moves while
     *        playing, at the same latency.
     */
    void setDetail(T octaves) noexcept
    {
        if (std::isfinite(octaves))
            detail_.store(std::clamp(octaves, static_cast<T>(DetailAnalysis<T>::kFinest), static_cast<T>(DetailAnalysis<T>::kWidest)), std::memory_order_relaxed);
    }
    /** @return The detail, in octaves. */
    [[nodiscard]] T getDetail() const noexcept { return detail_.load(std::memory_order_relaxed); }

    /**
     * @brief Reads the levels at the top through a window that shortens as
     *        frequency rises, centred where the curve is played (off by
     *        default; see DetailAnalysis::setTimeFocus()), and plays a brief
     *        move as brief as it is drawn (DetailHold::None):
     *        what an event there sets off starts later and lasts less, at
     *        part of its depth and with less detail at the top.
     */
    void setTimeFocus(bool on) noexcept { timeFocus_.store(on, std::memory_order_relaxed); }
    /** @return Whether the levels at the top are read through the focus. */
    [[nodiscard]] bool getTimeFocus() const noexcept { return timeFocus_.load(std::memory_order_relaxed); }

    /** @return The latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return filter_.getLatency(); }
    /** @return The gain of the last frame at a frequency, in dB (audio thread or after processing). */
    [[nodiscard]] double getGain(double hz) const noexcept
    {
        if (total_.empty() || !(hz > 0.0)) return 0.0;
        const double at = (kPoints - 1) * std::log(hz / 20.0) / std::log(1000.0);
        return static_cast<double>(total_[static_cast<size_t>(std::lround(std::clamp(at, 0.0, static_cast<double>(kPoints - 1))))]);
    }

    /** @brief Points of the display curve, evenly in log frequency from 20 Hz to 20 kHz. */
    static constexpr int kDisplayPoints = SpectralReadout<T>::kPoints;
    /** @return The frequency of a display point, in hertz. */
    [[nodiscard]] static double getDisplayFrequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }
    /**
     * @return The gain played at a display point, in dB (all bands): the
     *         response of the kernels handed over every 21 ms, made by the
     *         first read after they change (any thread).
     */
    [[nodiscard]] T getDisplayGain(int point) const noexcept { return filter_.playedValue(0, point); }
    /** @return The gain with the largest magnitude of that response, in dB, its sign kept (any thread). */
    [[nodiscard]] T getPeakGain() const noexcept { return filter_.playedExtreme(); }

    /** @brief Processes the programme in place, its own level driving the bands. */
    void processBlock(AudioBufferView<T> buffer) noexcept { run(buffer, nullptr); }
    /** @brief Processes the programme in place, a key's level driving the bands. */
    void processBlock(AudioBufferView<T> buffer, AudioBufferView<T> key) noexcept
    {
        run(buffer, key.getNumChannels() > 0 && key.getNumSamples() >= buffer.getNumSamples() ? &key : nullptr);
    }

    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("SDEQ"), 2);
        w.write("detail", static_cast<float>(getDetail()));
        w.write("timeFocus", getTimeFocus());
        char key[32];
        for (int i = 0; i < kBands; ++i)
        {
            const Band b = getBand(i);
            const auto put = [&](const char* name, float v) { std::snprintf(key, sizeof(key), "b%d.%s", i, name); w.write(key, v); };
            std::snprintf(key, sizeof(key), "b%d.shape", i);
            w.write(key, static_cast<int32_t>(b.shape));
            put("freq", static_cast<float>(b.frequency));
            put("q", static_cast<float>(b.q));
            put("thr", static_cast<float>(b.threshold));
            put("ratio", static_cast<float>(b.ratio));
            put("range", static_cast<float>(b.range));
            put("att", static_cast<float>(b.attack));
            put("rel", static_cast<float>(b.release));
        }
        return w.blob();
    }

    /**
     * @brief Restores the settings. A blob of the first version, with cells
     *        in each band, gives as detail the width of the finest of them
     *        (two bandwidths over the cells; 32 at a Q of 1 is 1/12).
     */
    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("SDEQ")) return false;
        const bool first = r.processorVersion() < 2;
        double finest = 0.0;
        char key[32];
        for (int i = 0; i < kBands; ++i)
        {
            Band b;
            const auto get = [&](const char* name, float def) { std::snprintf(key, sizeof(key), "b%d.%s", i, name); return static_cast<T>(r.read(key, def)); };
            std::snprintf(key, sizeof(key), "b%d.shape", i);
            b.shape = static_cast<Shape>(std::clamp(r.read(key, 0), 0, 3));
            b.frequency = get("freq", 1000.0f);
            b.q = get("q", 1.0f);
            b.threshold = get("thr", -12.0f);
            b.ratio = get("ratio", 3.0f);
            b.range = get("range", -12.0f);
            b.attack = get("att", 10.0f);
            b.release = get("rel", 150.0f);
            setBand(i, b);
            if (!first || b.shape == Shape::Off) continue;
            std::snprintf(key, sizeof(key), "b%d.cells", i);
            const int cells = std::clamp(r.read(key, 32), 1, 128);
            const double q = std::clamp(static_cast<double>(getBand(i).q), 0.1, 20.0);
            const double band = b.shape == Shape::Bell ? 2.0 * std::asinh(0.5 / q) / std::numbers::ln2 : kShelfOctaves;
            const double width = 2.0 * band / cells;
            if (finest == 0.0 || width < finest) finest = width;
        }
        setDetail(first && finest > 0.0 ? static_cast<T>(finest) : static_cast<T>(r.read("detail", 1.0f / 12.0f)));
        setTimeFocus(r.read("timeFocus", false));
        return true;
    }

private:
    static constexpr int kPoints = SpectralReadout<T>::kPoints;
    static constexpr size_t kSize = static_cast<size_t>(kPoints);
    static constexpr double kSilent = -200.0;
    static constexpr double kLeastLevel = -120.0;  ///< Under this, in dBFS, there is nothing to act on.
    static constexpr double kShelfOctaves = 2.0;   ///< What a shelf counts as, to convert its old cells.

    struct Published
    {
        std::atomic<int> shape { 0 };
        std::atomic<T> frequency { T(1000) }, q { T(1) }, threshold { T(-12) }, ratio { T(3) };
        std::atomic<T> range { T(-12) }, attack { T(10) }, release { T(150) };
    };

    struct State
    {
        Band band {};
        bool active = false;
        int first = 0, end = 0;              ///< The span, in points.
        std::vector<T> share, shareDb;       ///< The band's shape, 0 to 1, and as a weight of the power in dB.
        std::vector<T> action, attack, release;
        double peak = kSilent;
        double candidate = kSilent;          ///< The reference of a span just moved, still listening.
        int watching = 0;                    ///< Hops it still listens before it replaces the peak.
    };

    /** @brief A band's shape, span and times, on the audio thread. */
    void redesign(int index) noexcept
    {
        auto& s = state_[static_cast<size_t>(index)];
        const bool wasActive = s.active;
        const int oldFirst = s.first, oldEnd = s.end;
        s.band = getBand(index);
        const Band& b = s.band;
        s.active = b.shape != Shape::Off && std::abs(static_cast<double>(b.range)) > 1e-3;
        s.watching = 0;
        s.first = s.end = 0;
        std::fill(s.share.begin(), s.share.end(), T(0));
        if (s.active)
        {
            const double f0 = std::clamp(static_cast<double>(b.frequency), 20.0, 0.45 * spec_.sampleRate);
            const double range = static_cast<double>(b.range), q = std::max(static_cast<double>(b.q), 0.025);
            // The band's shape from its analog prototype: a bilinear design
            // narrows toward the top of the band.
            const AnalogSection proto = b.shape == Shape::Bell ? AnalogSection::peak(f0, q, range)
                                      : b.shape == Shape::LowShelf ? AnalogSection::lowShelf(f0, range)
                                      : AnalogSection::highShelf(f0, range);
            const double attack = 0.001 * static_cast<double>(b.attack), release = 0.001 * static_cast<double>(b.release);
            int first = -1, last = -1;
            for (int i = 0; i < kPoints; ++i)
            {
                const auto u = static_cast<size_t>(i);
                const double hz = DetailFilter<T>::frequency(i);
                const double share = std::clamp(proto.magnitudeDb(hz) / range, 0.0, 1.0);
                if (!(share > 0.01)) continue;
                if (first < 0) first = i;
                last = i;
                s.share[u] = static_cast<T>(share);
                s.shareDb[u] = static_cast<T>(10.0 * std::log10(share));
                const double scale = std::sqrt(1000.0 / hz);
                s.attack[u] = static_cast<T>(std::exp(-hopSeconds_ / std::max(1e-4, attack * scale)));
                s.release[u] = static_cast<T>(std::exp(-hopSeconds_ / std::max(1e-3, release * scale)));
            }
            s.active = first >= 0;
            if (s.active)
            {
                s.first = first;
                s.end = last + 1;
            }
        }
        // Each frequency keeps what the band was doing there; outside, nothing.
        for (int i = 0; i < kPoints; ++i)
            if (i < s.first || i >= s.end || s.share[static_cast<size_t>(i)] == T(0)) s.action[static_cast<size_t>(i)] = T(0);
        if (!s.active) return;
        // A span that moved has another level: a second reference listens
        // to it for half a second, then replaces the first.
        if (wasActive && (std::abs(s.first - oldFirst) > 1 || std::abs(s.end - oldEnd) > 1))
        {
            s.candidate = kSilent;
            s.watching = std::max(1, static_cast<int>(0.5 / hopSeconds_));
        }
    }

    void run(AudioBufferView<T>& buffer, const AudioBufferView<T>* key) noexcept
    {
        if (!prepared_) return;
        const int channels = std::min(buffer.getNumChannels(), channels_);
        if (channels <= 0) return;
        for (int i = 0; i < kBands; ++i)
            if (dirty_[static_cast<size_t>(i)].exchange(false, std::memory_order_acquire)) redesign(i);
        if (const T detail = detail_.load(std::memory_order_relaxed); detail != lastDetail_)
        {
            lastDetail_ = detail;
            analysis_.setWidth(static_cast<double>(detail));
            const double points = static_cast<double>(detail) * (kPoints - 1) * std::numbers::ln2 / std::log(1000.0);
            hold_ = std::max(1, static_cast<int>(std::lround(points / 8.0)));
        }
        {
            const bool focus = timeFocus_.load(std::memory_order_relaxed);
            analysis_.setTimeFocus(focus);
            filter_.setHold(focus ? DetailHold::None : DetailHold::Reach);
        }
        filter_.processBlock(
            buffer,
            [&](int offset, int count) noexcept {
                for (int ch = 0; ch < channels; ++ch)
                {
                    const T* from = key != nullptr ? key->getChannel(std::min(ch, key->getNumChannels() - 1)) : buffer.getChannel(ch);
                    analysis_.feed(ch, from + offset, count);
                }
            },
            [&](int, T* curveDb) noexcept {
                analysis_.update(channels);
                design(curveDb);
            });
        if (!filter_.isDisplayDue(shownCurve_)) return;
        shownCurve_ = filter_.getCurveCount();
        filter_.publishPlayed(channels);
    }

    /**
     * @brief The level of each point: the largest within the hold, then
     *        smoothed over half of it. Read from `first` to before `end`
     *        only; the points outside, which reach no band, stay silent.
     */
    void readLevels(int first, int end) noexcept
    {
        analysis_.read(0, power_.data(), first, end);
        std::fill(level_.begin(), level_.begin() + first, T(kSilent));
        for (int i = first; i < end; ++i)
            level_[static_cast<size_t>(i)] = T(4.3429448190325183) * std::log(power_[static_cast<size_t>(i)] + T(1e-20));
        std::fill(level_.begin() + end, level_.end(), T(kSilent));
        holdLargest();
        const int half = hold_ / 2;
        if (half == 0) return;
        double sum = 0.0;
        for (size_t i = 0; i < kSize; ++i)
        {
            sum_[i] = sum;
            sum += static_cast<double>(held_[i]);
        }
        sum_[kSize] = sum;
        for (int i = 0; i < kPoints; ++i)
        {
            const int from = std::max(0, i - half), to = std::min(kPoints, i + half + 1);
            held_[static_cast<size_t>(i)] = static_cast<T>((sum_[static_cast<size_t>(to)] - sum_[static_cast<size_t>(from)]) / (to - from));
        }
    }

    /** @brief The largest level within hold_ points of each, in held_. */
    void holdLargest() noexcept
    {
        if (hold_ <= kFewPoints)
        {
            // A few points: as many passes of three, which compilers vectorise.
            T* from = level_.data();
            T* to = held_.data();
            for (int pass = 0; pass < hold_; ++pass)
            {
                to[0] = std::max(from[0], from[1]);
                for (int i = 1; i < kPoints - 1; ++i) to[i] = std::max(std::max(from[i - 1], from[i]), from[i + 1]);
                to[kPoints - 1] = std::max(from[kPoints - 2], from[kPoints - 1]);
                std::swap(from, to);
            }
            if (from != held_.data()) std::copy_n(from, kPoints, held_.data());
            return;
        }
        int head = 0, tail = 0, next = 0;
        for (int i = 0; i < kPoints; ++i)
        {
            for (; next < kPoints && next <= i + hold_; ++next)
            {
                const T v = level_[static_cast<size_t>(next)];
                while (tail > head && level_[static_cast<size_t>(queue_[static_cast<size_t>(tail - 1)])] <= v) --tail;
                queue_[static_cast<size_t>(tail++)] = next;
            }
            while (queue_[static_cast<size_t>(head)] < i - hold_) ++head;
            held_[static_cast<size_t>(i)] = level_[static_cast<size_t>(queue_[static_cast<size_t>(head)])];
        }
    }

    static constexpr int kFewPoints = 4;

    void design(T* curve) noexcept
    {
        std::fill(total_.begin(), total_.end(), T(0));
        // The points the bands reach, and as far again as the hold and its
        // smoothing read around them.
        int first = kPoints, end = 0;
        for (const auto& s : state_)
            if (s.active)
            {
                first = std::min(first, s.first);
                end = std::max(end, s.end);
            }
        if (end > first)
        {
            const int reach = hold_ + hold_ / 2 + 1;
            readLevels(std::max(0, first - reach), std::min(kPoints, end + reach));
        }
        for (auto& s : state_)
        {
            if (!s.active) continue;
            const Band& b = s.band;
            const T* held = held_.data();
            T top = T(-300);
            for (int i = s.first; i < s.end; ++i) top = std::max(top, held[i] + s.shareDb[static_cast<size_t>(i)]);
            // The band's recent peak: up at once, down by 2 dB per second.
            const double loudest = static_cast<double>(top);
            s.peak = loudest > s.peak ? loudest : std::max(loudest, s.peak - 2.0 * hopSeconds_);
            if (s.watching > 0)
            {
                s.candidate = loudest > s.candidate ? loudest : std::max(loudest, s.candidate - 2.0 * hopSeconds_);
                if (--s.watching == 0) s.peak = s.candidate;
            }
            const T threshold = static_cast<T>(std::max(s.peak + static_cast<double>(b.threshold), kLeastLevel));
            const T range = std::abs(b.range);
            const T perDb = (T(1) - T(1) / b.ratio) / range;
            for (int i = s.first; i < s.end; ++i)
            {
                const auto u = static_cast<size_t>(i);
                const T target = std::min(T(1), std::max(T(0), held[i] + s.shareDb[u] - threshold) * perDb);
                T& a = s.action[u];
                a = target + (target > a ? s.attack[u] : s.release[u]) * (a - target);
                if (target == T(0) && a < T(1e-5)) a = T(0);
                total_[u] += b.range * a * s.share[u];
            }
        }
        std::copy(total_.begin(), total_.end(), curve);
    }

    AudioSpec spec_ {};
    int channels_ = 0, hold_ = 1, shownCurve_ = 0;
    bool prepared_ = false;
    double hopSeconds_ = 256.0 / 48000.0;
    T lastDetail_ = T(-1);
    DetailFilter<T> filter_;
    DetailAnalysis<T> analysis_;
    std::array<Published, kBands> published_ {};
    std::array<std::atomic<bool>, kBands> dirty_ {};
    std::atomic<T> detail_ { T(1.0 / 12.0) };
    std::atomic<bool> timeFocus_ { false };
    std::array<State, kBands> state_ {};
    std::vector<T> power_, level_, held_, total_;
    std::vector<double> sum_;
    std::vector<int> queue_;
};

} // namespace dspark
