// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DetailAnalysis.h
 * @brief A short-time spectrum on a log-frequency grid whose window follows
 *        the detail asked: long where that detail is a few hertz wide, short
 *        where it is hundreds.
 *
 * One window cannot read a twelfth of an octave at 100 Hz (it takes a third
 * of a second) and an attack at 10 kHz. The signal is decimated by octaves
 * (up to four) and every level is read through the same window of 1024
 * samples, from 21 ms to 341 ms at 48 kHz. A point of the grid asks for the
 * window that tells apart two lines the width away (four widths' periods),
 * within those bounds, and reads between the two levels around it.
 *
 * What a point reads is the power of what lies at it: the larger of the
 * spectrum there and of the power inside a band a width wide that falls to
 * nothing from a top one step of the grid across (so a line between two
 * points is in both). A sine of full scale reads 1 in both, so a line reads its level at any frequency and
 * at any width; a noise reads the power in the wider of the two. A wide
 * width reads the broad shape and no finer.
 *
 * With setTimeFocus(true) the shortest window is read through one that
 * shortens as frequency rises (SpectralFocus, a sixth of an octave), and
 * its reading is given a hop later, so that the instant it reads is the
 * one the next window down is centred on (at a hop of half a window): what
 * an event at the top sets off starts later and stays next to it, at the
 * cost of detail there.
 *
 * Use: feed() every sample of each channel, update() once a hop, then read().
 * With setLinked(true) the channels are read as the mean of their powers,
 * at little more than the cost of one.
 * The longer the window, the later it follows: at 50 Hz a sixth of an octave
 * is half there after 0.17 s.
 *
 * Threading: prepare() on the setup thread (allocates); everything else on
 * the audio thread (owner-managed, no allocation).
 *
 * Dependencies: AudioSpec.h, DspMath.h, FFT.h, FIRFilter.h, SimdOps.h,
 * SpectralEnvelope.h, SpectralFilter.h, WindowFunctions.h.
 */

#include "AudioSpec.h"
#include "DspMath.h"
#include "FFT.h"
#include "FIRFilter.h"
#include "SimdOps.h"
#include "SpectralEnvelope.h"
#include "SpectralFilter.h"
#include "WindowFunctions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

namespace dspark {

/**
 * @class DetailAnalysis
 * @brief Multiresolution power spectrum on the grid of SpectralReadout.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DetailAnalysis
{
public:
    static constexpr int kPoints = SpectralReadout<T>::kPoints; ///< Points of a reading.
    static constexpr int kMaxChannels = 16;                     ///< Channels read at most.
    static constexpr double kFinest = 1.0 / 24.0;               ///< Narrowest width, in octaves.
    static constexpr double kWidest = 1.0;                      ///< Widest width, in octaves.

    /** @return The frequency of a point, in hertz. */
    [[nodiscard]] static double frequency(int point) noexcept { return SpectralReadout<T>::frequency(point); }

    /**
     * @brief Allocates for a layout (an invalid spec is ignored).
     * @param spec Rate and channels.
     */
    void prepare(const AudioSpec& spec)
    {
        if (!spec.isValid()) return;
        prepared_ = false;
        fs_ = spec.sampleRate;
        channels_ = std::min(spec.numChannels, kMaxChannels);
        const int octaves = std::clamp(static_cast<int>(std::lround(std::log2(std::max(fs_, 44100.0) / 44100.0))), 0, 3);
        n_ = kWindow << octaves;
        bins_ = n_ + 1;   // the transform is twice the window
        // A shorter window takes over beyond these bins of every decimated level, at the finest width.
        usedBins_ = std::min(bins_, static_cast<int>(std::ceil(4.0 * kPeriods / kFinest * std::exp2(kFinest))) + 8);
        mean_.assign(static_cast<std::size_t>(bins_), 0.0);
        levels_ = 1;
        while (levels_ < kMaxLevels && static_cast<double>(n_ << levels_) / fs_ <= kLongest) ++levels_;

        fft_ = std::make_unique<FFTReal<T>>(static_cast<std::size_t>(2 * n_));
        window_.resize(static_cast<std::size_t>(n_));
        WindowFunctions<T>::hann(window_.data(), n_, true);
        {
            std::vector<double> hann(static_cast<std::size_t>(n_));
            WindowFunctions<double>::hann(hann.data(), n_, true);
            // No narrower than a bin of the window itself (the padding only reads between them).
            focus_.prepare(hann.data(), n_, 2 * n_, kFocusWidth, 2.0);
        }
        for (auto* v : { &focusIn_, &focusRe_, &focusIm_ }) v->assign(static_cast<std::size_t>(2 * n_ + 2), 0.0);
        lag_.resize(static_cast<std::size_t>(channels_));
        for (auto& v : lag_)
            for (auto* p : { &v.power, &v.sum, &v.sumOfSum }) p->assign(static_cast<std::size_t>(bins_ + 1), 0.0);
        frame_.assign(static_cast<std::size_t>(2 * n_), T(0));
        spectrum_.assign(static_cast<std::size_t>(2 * n_ + 2), T(0));
        // What a level is read for lies under 0.19 of its rate, so the folds of
        // a short half-band fall clear of it.
        const auto taps = FIRDesign<double>::lowPass(1.0, 0.25, kTaps, FIRDesign<double>::estimateKaiserBeta(100.0));
        for (int i = 0; i < kTaps; ++i) taps_[static_cast<std::size_t>(i)] = static_cast<T>(taps[static_cast<std::size_t>(i)]);

        const auto size = static_cast<std::size_t>(bins_);
        channel_.resize(static_cast<std::size_t>(channels_));
        points_.resize(static_cast<std::size_t>(kPoints));
        for (auto& c : channel_)
            for (int l = 0; l < levels_; ++l)
            {
                auto& v = c[static_cast<std::size_t>(l)];
                v.ring.assign(static_cast<std::size_t>(n_), T(0));
                v.history.assign(static_cast<std::size_t>(kTaps - 1 + n_), T(0));
                v.power.assign(size + 1, 0.0);
                v.sum.assign(size + 1, 0.0);
                v.sumOfSum.assign(size + 1, 0.0);
            }
        for (auto* v : { &evens_, &odds_ }) v->assign(static_cast<std::size_t>((kTaps + n_) / 2 + 1), T(0));
        for (auto* v : { &decimated_, &partial_ }) v->assign(static_cast<std::size_t>(n_ / 2 + 1), T(0));
        for (int k = 0; k < kTaps; ++k)
        {
            T* phase = k % 2 == 0 ? evenTaps_.data() : oddTaps_.data();
            phase[k / 2] = taps_[static_cast<std::size_t>(k)];
        }
        prepared_ = true;
        mapPoints();
        reset();
    }

    /** @brief Clears the signal and the spectra. */
    void reset() noexcept
    {
        if (!prepared_) return;
        for (auto& c : channel_)
            for (int l = 0; l < levels_; ++l)
            {
                auto& v = c[static_cast<std::size_t>(l)];
                std::fill(v.ring.begin(), v.ring.end(), T(0));
                std::fill(v.history.begin(), v.history.end(), T(0));
                std::fill(v.power.begin(), v.power.end(), 0.0);
                std::fill(v.sum.begin(), v.sum.end(), 0.0);
                std::fill(v.sumOfSum.begin(), v.sumOfSum.end(), 0.0);
                v.write = v.pending = 0;
                v.odd = false;
            }
        lagged_.fill(false);
        tick_ = 0;
    }

    /**
     * @brief The width of what is told apart, from kWidest to kFinest octaves
     *        (1/12 by default). No allocation; the next read() follows it.
     * @param octaves The width; non-finite values are ignored.
     */
    void setWidth(double octaves) noexcept
    {
        if (!std::isfinite(octaves)) return;
        const double width = std::clamp(octaves, kFinest, kWidest);
        if (width == width_) return;
        width_ = width;
        if (prepared_) mapPoints();
    }
    /** @return The width in octaves. */
    [[nodiscard]] double getWidth() const noexcept { return width_; }

    /** @brief Reads every level every hop in place of the long ones in turns (for a render). */
    void setExhaustive(bool on) noexcept { exhaustive_ = on; }
    /** @return Whether every level is read every hop. */
    [[nodiscard]] bool isExhaustive() const noexcept { return exhaustive_; }

    /** @brief One reading for every channel, of their mean power; it holds from the next update(). */
    void setLinked(bool on) noexcept { linked_ = on; }
    /** @return Whether the channels are read as one. */
    [[nodiscard]] bool isLinked() const noexcept { return linked_; }

    /**
     * @brief Reads the shortest window through one shorter as frequency
     *        rises and gives that reading a hop later (off by default); it
     *        holds from the next update().
     */
    void setTimeFocus(bool on) noexcept { timeFocus_ = on; }
    /** @return Whether the shortest window is read through the focus. */
    [[nodiscard]] bool isTimeFocus() const noexcept { return timeFocus_; }

    /**
     * @brief Takes the next samples of a channel (any count).
     * @param channel The channel; one out of range is ignored.
     * @param x       The samples.
     * @param count   How many.
     */
    void feed(int channel, const T* x, int count) noexcept
    {
        if (!prepared_ || channel < 0 || channel >= channels_ || x == nullptr) return;
        auto& c = channel_[static_cast<std::size_t>(channel)];
        auto& v = c[0];
        // The signal's own level now; the octaves down are decimated at the
        // next update(), the first to read them (or when a window is waiting).
        for (int at = 0; at < count;)
        {
            const int m = std::min(count - at, n_ - v.pending);
            T* to = v.history.data() + kTaps - 1 + v.pending;
            for (int i = 0; i < m; ++i) to[i] = std::isfinite(x[at + i]) ? x[at + i] : T(0);
            toRing(v, to, m);
            v.pending += m;
            at += m;
            if (levels_ == 1) v.pending = 0;
            else if (v.pending == n_) catchUp(c);
        }
    }

    /**
     * @brief A hop: takes the spectra of the levels due. The two shortest
     *        windows are read every hop, the longer ones every two or four,
     *        in turns.
     * @param channels The channels fed since the last hop.
     */
    void update(int channels) noexcept
    {
        if (!prepared_) return;
        constexpr int every[kMaxLevels] = { 1, 1, 2, 4, 4 };
        constexpr int turn[kMaxLevels] = { 0, 0, 0, 1, 3 };
        const int count = std::min(channels, channels_);
        for (auto& c : channel_) catchUp(c);
        for (int l = 0; l < levels_; ++l)
        {
            if (!exhaustive_ && tick_ % every[l] != turn[l]) continue;
            const auto u = static_cast<std::size_t>(l);
            const int bins = l == 0 ? bins_ : usedBins_;
            if (linked_ && count > 1)
            {
                std::fill_n(mean_.data(), bins, 0.0);
                for (int ch = 0; ch < count; ++ch) addPower(channel_[static_cast<std::size_t>(ch)][u], bins, 1.0 / count, l == 0 && timeFocus_);
                store(channel_[0][u], 0, l, bins);
                continue;
            }
            for (int ch = 0; ch < count; ++ch)
            {
                auto& v = channel_[static_cast<std::size_t>(ch)][u];
                std::fill_n(mean_.data(), bins, 0.0);
                addPower(v, bins, 1.0, l == 0 && timeFocus_);
                store(v, ch, l, bins);
            }
        }
        if (!timeFocus_) lagged_.fill(false);
        tick_ = (tick_ + 1) % 4;
    }

    /**
     * @brief The power at every point (a sine of full scale reads 1).
     * @param channel The channel; one out of range reads zero.
     * @param power   kPoints values.
     */
    void read(int channel, T* power) const noexcept { read(channel, power, 0, kPoints); }

    /**
     * @brief The power at the points from `first` to before `end` only (the
     *        others are left as they are): what read() gives there.
     * @param channel The channel; one out of range reads zero.
     * @param power   kPoints values.
     * @param first   The first point read (clamped to the grid).
     * @param end     One past the last (clamped to the grid).
     */
    void read(int channel, T* power, int first, int end) const noexcept
    {
        first = std::clamp(first, 0, kPoints);
        end = std::clamp(end, first, kPoints);
        if (!prepared_ || channel < 0 || channel >= channels_)
        {
            std::fill(power + first, power + end, T(0));
            return;
        }
        const auto& c = channel_[static_cast<std::size_t>(linked_ ? 0 : channel)];
        for (int i = first; i < end; ++i)
        {
            const Point& p = points_[static_cast<std::size_t>(i)];
            double value = readLevel(c[static_cast<std::size_t>(p.level)], p.read[0]);
            if (p.blend > 0.0) value += p.blend * (readLevel(c[static_cast<std::size_t>(p.level + 1)], p.read[1]) - value);
            power[i] = static_cast<T>(value);
        }
    }

    /** @return The length of the window a point is read through, in seconds. */
    [[nodiscard]] double getWindowSeconds(int point) const noexcept
    {
        if (!prepared_) return 0.0;
        const Point& p = points_[static_cast<std::size_t>(std::clamp(point, 0, kPoints - 1))];
        return static_cast<double>(n_) * std::exp2(p.level + p.blend) / fs_;
    }

private:
    static constexpr int kWindow = 1024;     ///< Window of every level at 44.1 and 48 kHz.
    static constexpr int kMaxLevels = 5;     ///< The signal and four octaves down.
    static constexpr double kLongest = 0.4;  ///< No window longer, in seconds.
    static constexpr double kPeriods = 4.0;  ///< Window wanted, in periods of one width.
    static constexpr double kFocusWidth = 1.0 / 6.0; ///< The focus's smoothing, in octaves.
    static constexpr int kTaps = 23;         ///< The decimator of a level.

    static constexpr int kEvenTaps = (kTaps + 1) / 2, kOddTaps = kTaps / 2;

    struct Level
    {
        std::vector<T> ring;
        std::vector<T> history;                     ///< The decimator's last kTaps - 1 inputs, then those waiting.
        std::vector<double> power, sum, sumOfSum;   ///< Per bin, and its first and second integrals.
        int write = 0, pending = 0;
        bool odd = false;                           ///< An output falls on the next input.
    };

    /// Samples onto a level's ring.
    void toRing(Level& v, const T* x, int m) noexcept
    {
        const int first = std::min(m, n_ - v.write);
        std::copy_n(x, first, v.ring.data() + v.write);
        std::copy_n(x + first, m - first, v.ring.data());
        v.write = (v.write + m) & (n_ - 1);
    }

    /// Every octave down of a channel up to date: what waits at each level is decimated into the next.
    void catchUp(std::array<Level, static_cast<std::size_t>(kMaxLevels)>& c) noexcept
    {
        for (int l = 0; l + 1 < levels_; ++l)
        {
            auto& v = c[static_cast<std::size_t>(l)];
            if (v.pending == 0) continue;
            auto& next = c[static_cast<std::size_t>(l + 1)];
            const int produced = decimate(v);
            toRing(next, decimated_.data(), produced);
            if (l + 2 < levels_)
            {
                std::copy_n(decimated_.data(), produced, next.history.data() + kTaps - 1 + next.pending);
                next.pending += produced;
            }
        }
    }

    /**
     * @brief The half-band decimator over what waits at a level, into
     *        decimated_: an output on every other input, of the kTaps inputs
     *        up to it, by its even and odd phases (two correlations over
     *        contiguous inputs). @return The outputs.
     */
    int decimate(Level& v) noexcept
    {
        const int m = v.pending;
        T* h = v.history.data();
        const int total = kTaps - 1 + m;
        const int start = v.odd ? 0 : 1;
        const int produced = m > start ? (m - start + 1) / 2 : 0;
        if (produced > 0)
        {
            T* even = evens_.data();
            T* odd = odds_.data();
            for (int t = 0; 2 * t < total; ++t) even[t] = h[2 * t];
            for (int t = 0; 2 * t + 1 < total; ++t) odd[t] = h[2 * t + 1];
            // y[j] = sum_k taps[k] h[start + 2j + k]: the even taps read one phase, the odd taps the other.
            const T* withEven = (start & 1) != 0 ? odd + (start - 1) / 2 : even + start / 2;
            const T* withOdd = (start & 1) != 0 ? even + (start + 1) / 2 : odd + start / 2;
            T* y = decimated_.data();
            T* rest = partial_.data();
            simd::firCorrelate(withEven, evenTaps_.data(), kEvenTaps, y, produced, T(1));
            simd::firCorrelate(withOdd, oddTaps_.data(), kOddTaps, rest, produced, T(1));
            for (int j = 0; j < produced; ++j) y[j] += rest[j];
        }
        if ((m & 1) != 0) v.odd = !v.odd;
        std::copy(h + m, h + m + kTaps - 1, h);
        v.pending = 0;
        return produced;
    }

    /// A place between two bins.
    struct Place
    {
        double part = 0.0, beyond = 0.0;
        int bin = 0;
    };

    /// What a point reads at one level: the line, and the four corners of its band.
    struct Reading
    {
        Place line, under, topUnder, topOver, over;
        double perUp = 0.0, perDown = 0.0;   ///< A third over each slope's length.
        bool inside = false;
    };

    /// What a point reads, between two levels.
    struct Point
    {
        std::array<Reading, 2> read {};
        double blend = 0.0;
        int level = 0;
    };

    void mapPoints() noexcept
    {
        const double binHz = fs_ / static_cast<double>(2 * n_), shortest = static_cast<double>(n_) / fs_;
        const double down = std::exp2(-0.5 * width_), up = std::exp2(0.5 * width_);
        const double halfStep = std::pow(1000.0, 0.5 / (kPoints - 1));
        for (int i = 0; i < kPoints; ++i)
        {
            Point& p = points_[static_cast<std::size_t>(i)];
            const double hz = frequency(i);
            const double wanted = kPeriods / (width_ * hz);
            const double level = std::clamp(std::log2(wanted / shortest), 0.0, static_cast<double>(levels_ - 1));
            p.level = std::min(static_cast<int>(level), std::max(0, levels_ - 2));
            p.blend = levels_ > 1 ? level - static_cast<double>(p.level) : 0.0;
            for (int side = 0; side < (levels_ > 1 ? 2 : 1); ++side)
            {
                Reading& r = p.read[static_cast<std::size_t>(side)];
                const int at = p.level + side, bins = at == 0 ? bins_ : usedBins_;
                const double bin = hz / binHz * std::exp2(at);
                // At a weight of nothing a level is not read (nor kept that high).
                r.inside = bin < static_cast<double>(bins - 1);
                if (!r.inside) continue;
                r.line.bin = static_cast<int>(bin);
                r.line.part = bin - static_cast<double>(r.line.bin);
                // The bands hold from half a bin under each bin.
                const auto corner = [bins](double x) {
                    Place c;
                    const double from = std::max(x + 0.5, 0.0), end = static_cast<double>(bins);
                    if (from >= end)
                    {
                        c.bin = bins;
                        c.beyond = from - end;
                        return c;
                    }
                    c.bin = static_cast<int>(from);
                    c.part = from - static_cast<double>(c.bin);
                    return c;
                };
                r.under = corner(bin * down);
                r.topUnder = corner(bin / halfStep);
                r.topOver = corner(bin * halfStep);
                r.over = corner(bin * up);
                // A sine spreads 1.5 bins of the window, three of the transform.
                r.perUp = 1.0 / (3.0 * bin * (1.0 / halfStep - down));
                r.perDown = 1.0 / (3.0 * bin * (up - halfStep));
            }
        }
    }

    /// Adds a share of the power of a level's window into mean_.
    void addPower(const Level& v, int bins, double share, bool focused) noexcept
    {
        const T* r = v.ring.data();
        const T* w = window_.data();
        T* frame = frame_.data();
        const int older = n_ - v.write;
        for (int i = 0; i < older; ++i) frame[i] = r[v.write + i] * w[i];
        for (int i = 0; i < v.write; ++i) frame[older + i] = r[i] * w[older + i];
        std::fill(frame_.begin() + n_, frame_.end(), T(0));
        fft_->forward(frame_.data(), spectrum_.data());
        const double scale = share * 16.0 / (static_cast<double>(n_) * static_cast<double>(n_));
        if (focused)
        {
            // Each bin through a stretch of the window of its own, scaled to read what the whole window would.
            const int all = focus_.getNumBins();
            for (int k = 0; k < 2 * all; ++k) focusIn_[static_cast<std::size_t>(k)] = static_cast<double>(spectrum_[static_cast<std::size_t>(k)]);
            focus_.smooth(focusIn_.data(), focusRe_.data(), focusIm_.data());
            for (int k = 0; k < bins; ++k)
            {
                const auto u = static_cast<std::size_t>(k);
                mean_[u] += scale * focus_.getGain(k) * (focusRe_[u] * focusRe_[u] + focusIm_[u] * focusIm_[u]);
            }
            return;
        }
        for (int k = 0; k < bins; ++k)
        {
            const auto u = static_cast<std::size_t>(k);
            const double re = static_cast<double>(spectrum_[2 * u]), im = static_cast<double>(spectrum_[2 * u + 1]);
            mean_[u] += scale * (re * re + im * im);
        }
    }

    /// mean_ into a level, with its first and second integrals.
    void integrate(Level& v, int bins) noexcept
    {
        double sum = 0.0, sumOfSum = 0.0;
        for (int k = 0; k < bins; ++k)
        {
            const auto u = static_cast<std::size_t>(k);
            const double power = mean_[u];
            v.power[u] = power;
            v.sum[u] = sum;
            v.sumOfSum[u] = sumOfSum;
            sumOfSum += sum + 0.5 * power;
            sum += power;
        }
        v.sum[static_cast<std::size_t>(bins)] = sum;
        v.sumOfSum[static_cast<std::size_t>(bins)] = sumOfSum;
    }

    /// mean_ into a level; the shortest read through the focus shows the reading of the hop before.
    void store(Level& v, int ch, int l, int bins) noexcept
    {
        if (l != 0 || !timeFocus_)
        {
            integrate(v, bins);
            return;
        }
        auto& lag = lag_[static_cast<std::size_t>(ch)];
        auto& ready = lagged_[static_cast<std::size_t>(ch)];
        if (!ready)
        {
            // The first reading through the focus: no hop before it to show.
            integrate(lag, bins);
            std::copy(lag.power.begin(), lag.power.end(), v.power.begin());
            std::copy(lag.sum.begin(), lag.sum.end(), v.sum.begin());
            std::copy(lag.sumOfSum.begin(), lag.sumOfSum.end(), v.sumOfSum.begin());
            ready = true;
            return;
        }
        std::swap(v.power, lag.power);
        std::swap(v.sum, lag.sum);
        std::swap(v.sumOfSum, lag.sumOfSum);
        integrate(lag, bins);
    }

    /// The integral of the integral of the power up to x, a bin k holding from k - 1/2 to k + 1/2.
    [[nodiscard]] static double integral(const Level& v, const Place& x) noexcept
    {
        const auto u = static_cast<std::size_t>(x.bin);
        // The bin past the last has no power: straight from there.
        return v.sumOfSum[u] + (x.part + x.beyond) * v.sum[u] + 0.5 * x.part * x.part * v.power[u];
    }

    [[nodiscard]] static double readLevel(const Level& v, const Reading& r) noexcept
    {
        if (!r.inside) return 0.0;
        const auto u = static_cast<std::size_t>(r.line.bin);
        const double line = v.power[u] + r.line.part * (v.power[u + 1] - v.power[u]);
        // The band by parts.
        const double band = (integral(v, r.over) - integral(v, r.topOver)) * r.perDown
                          - (integral(v, r.topUnder) - integral(v, r.under)) * r.perUp;
        return std::max(line, band);
    }

    double fs_ = 48000.0, width_ = 1.0 / 12.0;
    int channels_ = 0, n_ = kWindow, bins_ = kWindow + 1, usedBins_ = kWindow + 1, levels_ = 1, tick_ = 0;
    bool prepared_ = false, linked_ = false, exhaustive_ = false;
    std::vector<double> mean_;
    std::unique_ptr<FFTReal<T>> fft_;
    std::vector<T> window_, frame_, spectrum_;
    std::vector<T> evens_, odds_, decimated_, partial_;   ///< The decimators' phases and outputs.
    SpectralFocus<double> focus_;                         ///< The shortest window read shorter as frequency rises.
    std::vector<double> focusIn_, focusRe_, focusIm_;
    bool timeFocus_ = false;
    std::vector<Level> lag_;                              ///< Through the focus: the shortest window's newest reading.
    std::array<bool, static_cast<std::size_t>(kMaxChannels)> lagged_ {};
    std::array<T, static_cast<std::size_t>(kTaps)> taps_ {};
    std::array<T, static_cast<std::size_t>(kEvenTaps)> evenTaps_ {};
    std::array<T, static_cast<std::size_t>(kOddTaps)> oddTaps_ {};
    std::vector<Point> points_;   // a third of a megabyte: not inside the owner
    std::vector<std::array<Level, static_cast<std::size_t>(kMaxLevels)>> channel_;
};

} // namespace dspark
