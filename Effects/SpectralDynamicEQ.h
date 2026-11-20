// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SpectralDynamicEQ.h
 * @brief Dynamic bands that act frequency by frequency inside themselves: a
 *        band is cut into cells, each cell follows its own level, and the
 *        reduction takes the band's shape, applied as a minimum-phase filter.
 *
 * A dynamic bell set on a resonance acts on the whole bell when only one
 * frequency in it is loud. Here a band (a bell or a shelf, with its
 * frequency, Q and range) covers the frequencies where its static response
 * at full range is over one percent of that range, cut into up to 128
 * cells (setBand(): `cells`) placed by the band's weight: equal steps of
 * its shape integrated over log frequency, so they crowd where the band
 * acts and the skirts take a few wide ones. Every hop the
 * level of a cell is the largest power of the key's spectrum in it,
 * weighted by the band's shape there; the threshold is relative to the
 * band's own recent peak (it rises at once and falls by 2 dB per second),
 * so it follows the programme's level; over the threshold the cell's action
 * is the excess times (1 - 1 / ratio), limited by the range, with an attack
 * and a release that are faster in the upper cells (by the square root of
 * the frequency). The actions are projected onto the bins between the cells'
 * centres, multiplied by the band's shape (so the result stays between the
 * band's base response, flat, and its extreme one, the full range) and the
 * bands add in decibels. The gain curve is applied by a SpectralFilter: a
 * minimum-phase FIR designed every hop, without pre-echo.
 *
 * Latency: the analysis window, about 46 ms (doubling by octaves with the
 * rate; getLatency()).
 *
 * Threading: prepare() and setState() on the setup thread (allocate);
 * processBlock() and reset() on the audio thread; setBand() publishes
 * lock-free (the audio thread redesigns the band at its next block).
 * getDisplayGain() and getPeakGain() read the last frame's gain curve from
 * any thread.
 * Non-finite fields are ignored.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, Biquad.h, DspMath.h,
 * SpectralFilter.h, StateBlob.h.
 */

#include "../Core/AudioBuffer.h"
#include "../Core/AudioSpec.h"
#include "../Core/Biquad.h"
#include "../Core/DspMath.h"
#include "../Core/SpectralFilter.h"
#include "../Core/StateBlob.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dspark {

/**
 * @class SpectralDynamicEQ
 * @brief Dynamic equalizer bands with a level per cell inside each band.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class SpectralDynamicEQ
{
public:
    static constexpr int kBands = 4;      ///< Bands.
    static constexpr int kMaxCells = 128; ///< Most cells in a band.

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
        int cells = 32;          ///< 1 to 128.
    };

    /** @brief Allocates for a rate and a layout (an invalid spec is ignored). */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        spec_ = spec;
        const int octaves = std::max(0, static_cast<int>(std::lround(std::log2(std::max(spec.sampleRate, 44100.0) / 44100.0))));
        filter_.prepare(spec, std::min(2048 << octaves, 16384), 2, 4);
        filter_.setLinked(true);
        bins_ = filter_.getNumBins();
        for (auto& s : state_)
        {
            s.shape.assign(static_cast<size_t>(bins_), 0.0);
            s.cellOf.assign(static_cast<size_t>(bins_), -1);
            s.action.fill(0.0);
            s.peak = -200.0;
            s.active = false;
        }
        gainDb_.assign(static_cast<size_t>(bins_), 0.0);
        readout_.prepare(filter_.getBinHz(), bins_);
        for (auto& d : dirty_) d.store(true, std::memory_order_release);
        prepared_ = true;
    }

    /** @brief Clears the analysis, the actions and the filter. */
    void reset() noexcept
    {
        filter_.reset();
        for (auto& s : state_) { s.action.fill(0.0); s.peak = -200.0; }
        readout_.clear();
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
        p.cells.store(std::clamp(band.cells, 1, kMaxCells), std::memory_order_relaxed);
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
        b.cells = p.cells.load(std::memory_order_relaxed);
        return b;
    }

    /** @return The latency in samples. */
    [[nodiscard]] int getLatency() const noexcept { return filter_.getLatency(); }
    /** @return The gain of the last frame at a frequency, in dB (audio thread or after processing). */
    [[nodiscard]] double getGain(double hz) const noexcept
    {
        if (bins_ < 2) return 0.0;
        return gainDb_[static_cast<size_t>(std::clamp(static_cast<int>(std::lround(hz / filter_.getBinHz())), 0, bins_ - 1))];
    }

    /** @brief Points of the display curve, evenly in log frequency from 20 Hz to 20 kHz. */
    static constexpr int kDisplayPoints = SpectralReadout<T>::kPoints;
    /** @return The frequency of a display point, in hertz. */
    [[nodiscard]] static double getDisplayFrequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }
    /** @return The gain of the last frame around a display point, in dB (all bands; any thread). */
    [[nodiscard]] T getDisplayGain(int point) const noexcept { return readout_.value(0, point); }
    /** @return The gain of the last frame with the largest magnitude, in dB, its sign kept (any thread). */
    [[nodiscard]] T getPeakGain() const noexcept { return readout_.extreme(); }

    /** @brief Processes the programme in place, its own level driving the bands. */
    void processBlock(AudioBufferView<T> buffer) noexcept { run(buffer, nullptr); }
    /** @brief Processes the programme in place, a key's level driving the bands. */
    void processBlock(AudioBufferView<T> buffer, AudioBufferView<T> key) noexcept { run(buffer, &key); }

    [[nodiscard]] std::vector<uint8_t> getState() const
    {
        StateWriter w(stateId("SDEQ"), 1);
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
            std::snprintf(key, sizeof(key), "b%d.cells", i);
            w.write(key, static_cast<int32_t>(b.cells));
        }
        return w.blob();
    }

    bool setState(const uint8_t* data, size_t size)
    {
        StateReader r(data, size);
        if (!r.isValid() || r.processorId() != stateId("SDEQ")) return false;
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
            std::snprintf(key, sizeof(key), "b%d.cells", i);
            b.cells = r.read(key, 32);
            setBand(i, b);
        }
        return true;
    }

private:
    struct Published
    {
        std::atomic<int> shape { 0 }, cells { 32 };
        std::atomic<T> frequency { T(1000) }, q { T(1) }, threshold { T(-12) }, ratio { T(3) };
        std::atomic<T> range { T(-12) }, attack { T(10) }, release { T(150) };
    };

    struct State
    {
        Band band {};
        bool active = false;
        std::vector<double> shape;           ///< Per bin, 0 to 1.
        std::vector<int> cellOf;             ///< Per bin, the cell below it (-1 outside).
        std::array<double, kMaxCells> centre {}, action {}, attack {}, release {};
        std::array<int, kMaxCells + 1> edge {};
        int cells = 0;
        double peak = -200.0;
    };

    /** @brief Designs a band's shape, span and cells on the audio thread. */
    void redesign(int index) noexcept
    {
        auto& s = state_[static_cast<size_t>(index)];
        s.band = getBand(index);
        const Band& b = s.band;
        s.active = b.shape != Shape::Off && std::abs(static_cast<double>(b.range)) > 1e-3;
        std::fill(s.shape.begin(), s.shape.end(), 0.0);
        std::fill(s.cellOf.begin(), s.cellOf.end(), -1);
        s.cells = 0;
        if (!s.active) return;
        const double fs = spec_.sampleRate, binHz = filter_.getBinHz();
        const double f0 = std::clamp(static_cast<double>(b.frequency), 20.0, 0.45 * fs);
        const double range = static_cast<double>(b.range);
        BiquadCoeffs c;
        if (b.shape == Shape::Bell) c = BiquadCoeffs::makePeak(fs, f0, static_cast<double>(b.q), range);
        else if (b.shape == Shape::LowShelf) c = BiquadCoeffs::makeLowShelf(fs, f0, range);
        else c = BiquadCoeffs::makeHighShelf(fs, f0, range);
        int lo = -1, hi = -1;
        for (int k = 1; k < bins_; ++k)
        {
            const double db = 20.0 * std::log10(std::max(c.getMagnitude(k * binHz, fs), 1e-9));
            const double share = std::clamp(db / range, 0.0, 1.0);
            s.shape[static_cast<size_t>(k)] = share;
            if (share > 0.01) { if (lo < 0) lo = k; hi = k; }
        }
        if (lo < 0) { s.active = false; return; }
        // Cells by the band's weight: equal steps of the shape integrated over
        // log frequency (each bin weighs its share over its index), so the
        // cells crowd where the band acts and the skirts, where it barely
        // moves, take a few wide ones. Every cell keeps at least one bin.
        s.cells = std::min(b.cells, hi - lo + 1);
        double total = 0.0;
        for (int k = lo; k <= hi; ++k) total += s.shape[static_cast<size_t>(k)] / static_cast<double>(k);
        s.edge[0] = lo;
        int cell = 1;
        double acc = 0.0;
        for (int k = lo; k <= hi && cell < s.cells; ++k)
        {
            acc += s.shape[static_cast<size_t>(k)] / static_cast<double>(k);
            while (cell < s.cells && acc >= total * static_cast<double>(cell) / static_cast<double>(s.cells))
            {
                const int room = hi + 1 - (s.cells - cell);   // what the cells after this one still need
                s.edge[static_cast<size_t>(cell)] = std::min(room, std::max(s.edge[static_cast<size_t>(cell - 1)] + 1, k + 1));
                ++cell;
            }
        }
        for (; cell < s.cells; ++cell)
            s.edge[static_cast<size_t>(cell)] = std::min(hi + 1 - (s.cells - cell), s.edge[static_cast<size_t>(cell - 1)] + 1);
        s.edge[static_cast<size_t>(s.cells)] = hi + 1;
        const double hopSeconds = static_cast<double>(filter_.getHopSize()) / fs;
        for (int j = 0; j < s.cells; ++j)
        {
            const auto i = static_cast<size_t>(j);
            const int a = s.edge[i], e = std::max(s.edge[i] + 1, s.edge[i + 1]);
            s.centre[i] = 0.5 * (a + e - 1);
            for (int k = a; k < std::min(e, bins_); ++k) s.cellOf[static_cast<size_t>(k)] = j;
            const double scale = std::sqrt(1000.0 / std::max(20.0, s.centre[i] * binHz));
            s.attack[i] = std::exp(-hopSeconds / std::max(1e-4, 0.001 * static_cast<double>(b.attack) * scale));
            s.release[i] = std::exp(-hopSeconds / std::max(1e-3, 0.001 * static_cast<double>(b.release) * scale));
        }
    }

    void run(AudioBufferView<T>& buffer, const AudioBufferView<T>* key) noexcept
    {
        if (!prepared_) return;
        for (int i = 0; i < kBands; ++i)
            if (dirty_[static_cast<size_t>(i)].exchange(false, std::memory_order_acquire)) redesign(i);
        filter_.processBlock(buffer, key, [&](int, const T* power, T* gain, int bins) noexcept { design(power, gain, bins); });
    }

    void design(const T* power, T* gain, int bins) noexcept
    {
        std::fill(gainDb_.begin(), gainDb_.end(), 0.0);
        const double hopSeconds = static_cast<double>(filter_.getHopSize()) / spec_.sampleRate;
        for (auto& s : state_)
        {
            if (!s.active || s.cells <= 0) continue;
            const Band& b = s.band;
            // Each cell's level: the largest power in it, weighted by the shape.
            std::array<double, kMaxCells> level {};
            double top = -300.0;
            for (int j = 0; j < s.cells; ++j)
            {
                const auto i = static_cast<size_t>(j);
                double m = 0.0;
                for (int k = s.edge[i]; k < std::min(s.edge[i + 1], bins); ++k)
                    m = std::max(m, static_cast<double>(power[k]) * s.shape[static_cast<size_t>(k)]);
                level[i] = 10.0 * std::log10(m + 1e-20);
                top = std::max(top, level[i]);
            }
            // The band's recent peak: up at once, down by 2 dB per second.
            s.peak = top > s.peak ? top : std::max(top, s.peak - 2.0 * hopSeconds);
            const double thr = s.peak + static_cast<double>(b.threshold);
            const double slope = 1.0 - 1.0 / static_cast<double>(b.ratio);
            const double range = std::abs(static_cast<double>(b.range));
            for (int j = 0; j < s.cells; ++j)
            {
                const auto i = static_cast<size_t>(j);
                const double target = std::min(1.0, std::max(0.0, level[i] - thr) * slope / range);
                const double c = target > s.action[i] ? s.attack[i] : s.release[i];
                s.action[i] = target + c * (s.action[i] - target);
            }
            // Projected onto the bins between the cells' centres, with the shape.
            const double sign = b.range < T(0) ? -1.0 : 1.0;
            for (int k = 0; k < bins; ++k)
            {
                const double share = s.shape[static_cast<size_t>(k)];
                if (share <= 0.0) continue;
                const int j = s.cellOf[static_cast<size_t>(k)];
                double a;
                if (j < 0) a = k < s.edge[0] ? s.action[0] : s.action[static_cast<size_t>(s.cells - 1)];
                else
                {
                    const double ck = s.centre[static_cast<size_t>(j)];
                    const int other = k < ck ? std::max(0, j - 1) : std::min(s.cells - 1, j + 1);
                    const double co = s.centre[static_cast<size_t>(other)];
                    const double t = other == j || co == ck ? 0.0 : std::clamp((k - ck) / (co - ck), 0.0, 1.0);
                    a = s.action[static_cast<size_t>(j)] + t * (s.action[static_cast<size_t>(other)] - s.action[static_cast<size_t>(j)]);
                }
                gainDb_[static_cast<size_t>(k)] += sign * range * a * share;
            }
        }
        for (int k = 0; k < bins; ++k) gain[k] = static_cast<T>(std::pow(10.0, gainDb_[static_cast<size_t>(k)] / 20.0));
        readout_.publish(0, gainDb_.data(), bins);
    }

    AudioSpec spec_ {};
    int bins_ = 0;
    bool prepared_ = false;
    SpectralFilter<T> filter_;
    SpectralReadout<T> readout_;
    std::array<Published, kBands> published_ {};
    std::array<std::atomic<bool>, kBands> dirty_ {};
    std::array<State, kBands> state_ {};
    std::vector<double> gainDb_;
};

} // namespace dspark
