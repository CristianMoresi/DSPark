// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file SampleReader.h
 * @brief Shared band-limited zone reader: the interpolation kernel, the zone
 *        data a set holds, and one voice's read position, without voice,
 *        envelope or event policy.
 *
 * Internal machinery used by Sampler (which layers every matching zone) and
 * Synthesizer (whose sample source feeds the voice's filter and amplifier).
 * Played at or below the recording's rate a reader gathers each output
 * sample as a dot product with a 64-tap Kaiser sinc (100 dB); played up it
 * scatters each recording sample over the output samples with the kernel
 * laid out in output time, so the stopband starts at the output Nyquist
 * frequency. See Sampler.h for the measured quality.
 *
 * The reader produces one output frame per call at a rate the caller may
 * change on every call. Allocation happens in allocate() and when a zone is
 * built; reading is allocation free. One reader belongs to one thread.
 *
 * Dependencies: ../DspMath.h, ../SampleZone.h.
 */

#include "../DspMath.h"
#include "../SampleZone.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace dspark::detail
{

/** @brief One validated zone, its audio copied and its loop crossfade baked. */
template <FloatType T> struct SampleZoneData
{
    std::vector<T> audio;        ///< Channel-major, `length` per channel.
    std::vector<T> baked;        ///< Crossfaded copy of [loopEnd - xfade, loopEnd).
    int channels = 0;
    int64_t length = 0;
    double sampleRate = 48000.0;
    int rootKey = 60;
    double tuneCents = 0.0;
    int keyLow = 0, keyHigh = 127;
    double velLow = 0.0, velHigh = 1.0;
    double gain = 1.0;
    double pan = 0.0;
    int64_t start = 0;
    SampleLoopMode loop = SampleLoopMode::None;
    int64_t loopStart = 0, loopEnd = 0, xfade = 0;
};

/**
 * @brief Validates and copies a zone, clamping its loop and baking the loop
 *        crossfade. False (and `d` unspecified) for a zone with no audio, an
 *        invalid rate or an empty key or velocity range.
 */
template <FloatType T>
bool buildSampleZone(const SampleZoneSpec<T>& z, SampleZoneData<T>& d)
{
    const int ch = z.audio.getNumChannels();
    const int64_t len = z.audio.getNumSamples();
    if (ch < 1 || len < 1 || !(z.sampleRate > 0.0) || !std::isfinite(z.sampleRate))
        return false;
    for (int c = 0; c < ch; ++c)
        if (z.audio.getChannel(c) == nullptr) return false;
    if (z.keyLow > z.keyHigh || z.velocityLow > z.velocityHigh
        || !std::isfinite(z.tuneCents) || !std::isfinite(z.gainDb) || !std::isfinite(z.pan))
        return false;

    d.channels = ch;
    d.length = len;
    d.sampleRate = z.sampleRate;
    d.rootKey = z.rootKey;
    d.tuneCents = z.tuneCents;
    d.keyLow = std::clamp(z.keyLow, 0, 127);
    d.keyHigh = std::clamp(z.keyHigh, 0, 127);
    d.velLow = std::clamp(z.velocityLow, 0.0, 1.0);
    d.velHigh = std::clamp(z.velocityHigh, 0.0, 1.0);
    d.gain = std::pow(10.0, std::clamp(z.gainDb, -120.0, 24.0) / 20.0);
    d.pan = std::clamp(z.pan, -1.0, 1.0);
    d.start = std::clamp<int64_t>(z.startOffset, 0, len - 1);
    d.audio.resize(static_cast<size_t>(ch) * static_cast<size_t>(len));
    for (int c = 0; c < ch; ++c)
    {
        const T* src = z.audio.getChannel(c);
        T* dst = d.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(len);
        for (int64_t i = 0; i < len; ++i)
            dst[i] = std::isfinite(src[i]) ? src[i] : T(0);
    }

    d.loop = z.loopMode;
    d.loopStart = std::clamp<int64_t>(z.loopStart, 0, len);
    d.loopEnd = std::clamp<int64_t>(z.loopEnd, 0, len);
    if (d.loop != SampleLoopMode::None && d.loopEnd - d.loopStart < 2)
        d.loop = SampleLoopMode::None;
    if (d.loop == SampleLoopMode::None) return true;

    d.xfade = std::clamp<int64_t>(z.loopCrossfade, 0,
                                  std::min(d.loopStart, d.loopEnd - d.loopStart));
    if (d.xfade > 0)
    {
        d.baked.resize(static_cast<size_t>(ch) * static_cast<size_t>(d.xfade));
        for (int c = 0; c < ch; ++c)
        {
            const T* src = d.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(len);
            T* dst = d.baked.data() + static_cast<size_t>(c) * static_cast<size_t>(d.xfade);
            for (int64_t i = 0; i < d.xfade; ++i)
            {
                // Fade the loop's tail out into the audio that precedes the
                // loop start, so the jump lands on continuous material.
                const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(d.xfade);
                const double fin = z.equalPowerCrossfade ? std::sin(0.5 * pi<double> * x) : x;
                const double fout = z.equalPowerCrossfade ? std::cos(0.5 * pi<double> * x) : 1.0 - x;
                const double tail = static_cast<double>(src[d.loopEnd - d.xfade + i]);
                const double lead = static_cast<double>(src[d.loopStart - d.xfade + i]);
                dst[i] = static_cast<T>(fout * tail + fin * lead);
            }
        }
    }
    return true;
}

/**
 * @brief Output gains of a zone at `gain`: a mono zone pans with constant
 *        power (unity at centre), a multichannel one balances its first two
 *        channels. `mono` is the gain for a single output channel.
 */
inline void sampleZonePanGains(double pan, int channels, double gain,
                               double& left, double& right, double& mono) noexcept
{
    mono = gain;
    if (channels == 1)
    {
        // cos on both sides (sin(a) = cos(pi/2 - a)): opposite pans get
        // exactly mirrored gains, and a centred zone is exactly mono.
        left = gain * std::cos(0.25 * pi<double> * (1.0 + pan)) * sqrt2<double>;
        right = gain * std::cos(0.25 * pi<double> * (1.0 - pan)) * sqrt2<double>;
    }
    else
    {
        left = gain * std::min(1.0, 1.0 - pan);
        right = gain * std::min(1.0, 1.0 + pan);
    }
}

/**
 * @class SampleReader
 * @brief One voice's position in a zone and its band-limited read.
 */
template <FloatType T>
class SampleReader
{
public:
    /// Zero crossings on each side of the kernel at unit rate: 64 taps.
    static constexpr int kHalfTaps = 32;
    /// Table points per zero crossing (linear interpolation between them).
    static constexpr int kTablePerUnit = 2048;
    /// Highest playback rate; the kernel then spans 8 x 64 taps.
    static constexpr double kMaxRate = 8.0;
    static constexpr int kMaxHalfTaps = static_cast<int>(kHalfTaps * kMaxRate);
    /// Cutoff at 0.45 of the sample rate (normalised to Nyquist): passband to
    /// 0.40, stopband from 0.50, the width a 64-tap Kaiser needs for 100 dB.
    static constexpr double kCutoff = 0.9;
    static constexpr double kBeta = 10.06;
    /// Phases of the unit-rate polyphase table (rows interpolated linearly).
    static constexpr int kPhases = 2048;
    static constexpr int kTaps = 2 * kHalfTaps;
    /// Scratch weights a caller provides to next().
    static constexpr int kWeightsLength = 2 * kMaxHalfTaps + 2;
    /// Scatter accumulator per channel: room for the widest kernel plus a
    /// stretch of output before it slides back.
    static constexpr int kAccLength = 4 * kMaxHalfTaps + 256;

    /** @brief What one call produced. */
    enum class Frame
    {
        Ended,     ///< The zone ended before this sample: nothing produced.
        Produced,  ///< A frame was produced.
        Last       ///< A frame was produced and it was the last one.
    };

    /** Kernel h(t) for t >= 0 in zero crossings, sampled kTablePerUnit per unit. */
    static const std::vector<double>& kernelTable()
    {
        static const std::vector<double> table = [] {
            std::vector<double> t(static_cast<size_t>(kHalfTaps * kTablePerUnit + 2), 0.0);
            auto i0 = [](double x) {
                double sum = 1.0, term = 1.0;
                for (int k = 1; k < 64; ++k)
                {
                    term *= (x / (2.0 * k)) * (x / (2.0 * k));
                    sum += term;
                    if (term < sum * 1e-17) break;
                }
                return sum;
            };
            const double norm = i0(kBeta);
            for (size_t j = 0; j + 1 < t.size(); ++j)
            {
                const double x = static_cast<double>(j) / kTablePerUnit;
                if (x >= kHalfTaps) break;
                const double arg = pi<double> * kCutoff * x;
                const double sinc = (x == 0.0) ? 1.0 : std::sin(arg) / arg;
                const double r = x / kHalfTaps;
                t[j] = kCutoff * sinc * i0(kBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) / norm;
            }
            return t;
        }();
        return table;
    }

    /**
     * The unit-rate kernel as rows: row q holds the 64 tap weights for a
     * fractional position q / kPhases, taps k = -31 .. 32 at distance k - frac.
     * A voice at or below the recording's rate blends two adjacent rows, which
     * is the same kernel as the scalar table read tap by tap, laid out so the
     * blend and the dot products run over contiguous memory.
     */
    static const std::vector<T>& polyphaseTable()
    {
        static const std::vector<T> rows = [] {
            const auto& h = kernelTable();
            std::vector<T> r(static_cast<size_t>((kPhases + 1) * kTaps), T(0));
            for (int q = 0; q <= kPhases; ++q)
                for (int t = 0; t < kTaps; ++t)
                {
                    // |k - frac| in table units, exactly: (t - 31) * P - q.
                    const long u = std::labs(static_cast<long>(t - (kHalfTaps - 1)) * kTablePerUnit - q);
                    r[static_cast<size_t>(q * kTaps + t)] =
                        (static_cast<size_t>(u) < h.size()) ? static_cast<T>(h[static_cast<size_t>(u)]) : T(0);
                }
            return r;
        }();
        return rows;
    }

    /** @brief Allocates the scatter accumulators (setup thread). */
    void allocate()
    {
        for (auto& a : acc_) a.assign(static_cast<size_t>(kAccLength), T(0));
    }

    /**
     * @brief Starts reading a zone from its start offset. A reader that
     *        starts above the recording's rate (`ratio` > 1) scatters for the
     *        rest of the note, otherwise it gathers.
     */
    void start(const SampleZoneData<T>& z, double ratio) noexcept
    {
        zone_ = &z;
        pos_ = static_cast<double>(z.start);
        scatter_ = ratio > 1.0;
        inputDone_ = false;
        drainLeft_ = 0;
        jIn_ = z.start;
        tNext_ = 0.0;
        accPos_ = 0;
        if (scatter_)
            for (auto& a : acc_) std::fill(a.begin(), a.end(), T(0));
        loopEngaged_ = z.loop != SampleLoopMode::None;
        releaseLoop_ = false;
    }

    /** @brief The key is released: a Sustain loop plays on past its end. */
    void release() noexcept
    {
        if (zone_ != nullptr && zone_->loop == SampleLoopMode::Sustain) releaseLoop_ = true;
    }

    /** @brief The zone being read (null before start()). */
    [[nodiscard]] const SampleZoneData<T>* zone() const noexcept { return zone_; }

    /**
     * @brief One output frame at `ratio` recording samples per output sample.
     *
     * @param ratio       Playback rate, within [1 / kMaxRate, kMaxRate].
     * @param outChannels Output channels: 1 mixes the zone's channels to one,
     *                    2 or more read its first two.
     * @param weights     Scratch of kWeightsLength values.
     * @param a0          The frame's first (or only) channel.
     * @param a1          The frame's second channel; not meaningful for one
     *                    output channel or a mono zone, where a0 is used.
     */
    Frame next(double ratio, int outChannels, T* weights, double& a0, double& a1) noexcept
    {
        return scatter_ ? scatterFrame(ratio, outChannels, weights, a0, a1)
                        : gatherFrame(ratio, outChannels, weights, a0, a1);
    }

private:
    /** Dot product over eight independent partial sums: a single running sum
     *  is one long dependency chain the compiler may not reorder. */
    [[nodiscard]] static double dot(const T* w, const T* x, int n) noexcept
    {
        T a[8] = { T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0) };
        int t = 0;
        for (; t + 8 <= n; t += 8)
            for (int l = 0; l < 8; ++l)
                a[l] += w[t + l] * x[t + l];
        T sum = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
        for (; t < n; ++t) sum += w[t] * x[t];
        return static_cast<double>(sum);
    }

    /** Reads sample j of channel c through the loop and the baked crossfade. */
    [[nodiscard]] static double fetch(const SampleZoneData<T>& z, const T* chan, const T* baked,
                                      int64_t j, bool loopEngaged) noexcept
    {
        if (loopEngaged && j >= z.loopEnd)
            j = z.loopStart + (j - z.loopEnd) % (z.loopEnd - z.loopStart);
        if (j < 0 || j >= z.length) return 0.0;
        if (loopEngaged && baked != nullptr && j >= z.loopEnd - z.xfade && j < z.loopEnd)
            return static_cast<double>(baked[j - (z.loopEnd - z.xfade)]);
        return static_cast<double>(chan[j]);
    }

    /** Fills `weights` with the kernel at scale fc for taps k = -half+1 .. half
     *  at distance (k - frac); returns the tap count. */
    static int kernelWeights(T* weights, double frac, double fc, int half) noexcept
    {
        if (fc >= 1.0)
        {
            const T* rows = polyphaseTable().data();
            const double qf = frac * kPhases;
            const int q = std::min(static_cast<int>(qf), kPhases - 1);
            const T g = static_cast<T>(qf - static_cast<double>(q));
            const T* r0 = rows + static_cast<size_t>(q) * kTaps;
            const T* r1 = r0 + kTaps;
            for (int t = 0; t < kTaps; ++t)
                weights[t] = r0[t] + g * (r1[t] - r0[t]);
            return kTaps;
        }
        // Stretched kernel: |k - frac| * fc in table units, one step of fc per tap.
        const auto& table = kernelTable();
        const double step = fc * kTablePerUnit;
        double x = (static_cast<double>(-half + 1) - frac) * step;
        const size_t last = table.size() - 1;
        for (int t = 0; t < 2 * half; ++t, x += step)
        {
            const double u = std::abs(x);
            const size_t iu = static_cast<size_t>(u);
            double w = 0.0;
            if (iu < last)
            {
                const double f = u - static_cast<double>(iu);
                w = table[iu] + f * (table[iu + 1] - table[iu]);
            }
            weights[t] = static_cast<T>(w * fc);
        }
        return 2 * half;
    }

    /**
     * Gather form, for readers that start at or below the recording's rate:
     * each output sample is a dot product of the recording with the kernel
     * centred on the read position. At a rate r above 1 (a bend upwards) the
     * kernel is stretched by r.
     */
    Frame gatherFrame(double ratio, int nCh, T* weights, double& a0, double& a1) noexcept
    {
        const SampleZoneData<T>& z = *zone_;
        const double fc = std::min(1.0, 1.0 / ratio);
        const int half = static_cast<int>(std::ceil(kHalfTaps / fc));
        const double loopLen = static_cast<double>(z.loopEnd - z.loopStart);
        const int zc = z.channels;
        const int used = (nCh == 1) ? zc : std::min(zc, 2);

        // Sustain loop released: leave the loop once the kernel is clear
        // of the crossfade region, so no read jumps sources.
        if (releaseLoop_ && pos_ + half < static_cast<double>(z.loopEnd - z.xfade))
        {
            loopEngaged_ = false;
            releaseLoop_ = false;
        }
        if (!loopEngaged_ && pos_ >= static_cast<double>(z.length)) return Frame::Ended;

        const double fl = std::floor(pos_);
        const int64_t ip = static_cast<int64_t>(fl);
        const int taps = kernelWeights(weights, pos_ - fl, fc, half);

        const int64_t first = ip - half + 1;
        const bool direct = first >= 0 && first + taps <= z.length
                         && (!loopEngaged_ || first + taps <= z.loopEnd - z.xfade);
        double acc[2] = { 0.0, 0.0 };
        for (int c = 0; c < used; ++c)
        {
            const T* chan = z.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(z.length);
            const T* baked = z.baked.empty() ? nullptr
                           : z.baked.data() + static_cast<size_t>(c) * static_cast<size_t>(z.xfade);
            double sum = 0.0;
            if (direct)
                sum = dot(weights, chan + first, taps);
            else
                for (int t = 0; t < taps; ++t)
                    sum += static_cast<double>(weights[t])
                         * fetch(z, chan, baked, first + t, loopEngaged_);
            if (nCh == 1) acc[0] += sum / zc;
            else acc[c] = sum;
        }
        a0 = acc[0];
        a1 = acc[1];

        pos_ += ratio;
        if (loopEngaged_ && pos_ >= static_cast<double>(z.loopEnd))
            pos_ -= loopLen;
        return Frame::Produced;
    }

    /**
     * Scatter (transposed) form, for readers that start above the recording's
     * rate: each recording sample is spread over the output samples around
     * the instant it falls on, with the kernel laid out in OUTPUT time. Its
     * stopband therefore starts at the output Nyquist frequency - exactly
     * the anti-aliasing a note played up needs - and the weights come from
     * the same contiguous polyphase rows as the gather form, so the cost is
     * r times that of an untransposed voice instead of a stretched gather.
     * Loop jumps happen at whole recording samples. At a rate below 1 (a
     * bend downwards) the output-time kernel is stretched by 1 / r, which
     * keeps the stopband at the recording's Nyquist frequency.
     */
    Frame scatterFrame(double ratio, int nCh, T* weights, double& a0, double& a1) noexcept
    {
        const SampleZoneData<T>& z = *zone_;
        const double fc = std::min(1.0, ratio);          // output-time kernel scale
        const int half = static_cast<int>(std::ceil(kHalfTaps / fc));
        const double spacing = 1.0 / ratio;             // output samples per input sample
        const int zc = z.channels;
        const int used = (nCh == 1) ? zc : std::min(zc, 2);
        const int rings = (nCh == 1 || zc == 1) ? 1 : 2;

        // Spread every input whose support reaches this output sample.
        while (!inputDone_ && tNext_ < static_cast<double>(half))
        {
            if (releaseLoop_ && jIn_ < z.loopEnd - z.xfade)
            {
                loopEngaged_ = false;
                releaseLoop_ = false;
            }
            double x[2] = { 0.0, 0.0 };
            for (int c = 0; c < used; ++c)
            {
                const T* chan = z.audio.data() + static_cast<size_t>(c) * static_cast<size_t>(z.length);
                const T* baked = z.baked.empty() ? nullptr
                               : z.baked.data() + static_cast<size_t>(c) * static_cast<size_t>(z.xfade);
                const double sample = fetch(z, chan, baked, jIn_, loopEngaged_);
                if (rings == 1) x[0] += sample / used;
                else x[c] = sample;
            }
            const double fl = std::floor(tNext_);
            const int taps = kernelWeights(weights, tNext_ - fl, fc, half);
            const int base = accPos_ + static_cast<int>(fl) - half + 1;
            for (int r = 0; r < rings; ++r)
            {
                const T xs = static_cast<T>(x[r] * spacing);
                T* acc = acc_[static_cast<size_t>(r)].data();
                const T* w = weights;
                for (int t = 0; t < taps; ++t)
                {
                    const int idx = base + t;
                    if (idx >= accPos_) acc[idx] += w[t] * xs;   // earlier outputs are gone
                }
            }
            ++jIn_;
            if (loopEngaged_ && jIn_ >= z.loopEnd) jIn_ = z.loopStart;
            if (!loopEngaged_ && jIn_ >= z.length) { inputDone_ = true; drainLeft_ = 2 * half; }
            tNext_ += spacing;
        }

        a0 = static_cast<double>(acc_[0][static_cast<size_t>(accPos_)]);
        a1 = (rings == 2) ? static_cast<double>(acc_[1][static_cast<size_t>(accPos_)]) : a0;

        // Advance the output: the accumulator slides once it nears its end.
        for (int r = 0; r < rings; ++r) acc_[static_cast<size_t>(r)][static_cast<size_t>(accPos_)] = T(0);
        ++accPos_;
        tNext_ -= 1.0;
        if (accPos_ + 2 * kMaxHalfTaps + 2 >= kAccLength)
        {
            const int keep = kAccLength - accPos_;
            for (int r = 0; r < rings; ++r)
            {
                T* acc = acc_[static_cast<size_t>(r)].data();
                std::copy(acc + accPos_, acc + kAccLength, acc);
                std::fill(acc + keep, acc + kAccLength, T(0));
            }
            accPos_ = 0;
        }
        if (inputDone_ && --drainLeft_ <= 0) return Frame::Last;
        return Frame::Produced;
    }

    const SampleZoneData<T>* zone_ = nullptr;
    double pos_ = 0.0;            ///< Read position in recording samples (gather form).
    bool loopEngaged_ = false;
    bool releaseLoop_ = false;    ///< Sustain loop: disengage once clear of the fade.
    // Scatter form: the next recording sample to spread, the output time it
    // falls on (relative to the next output sample), and the output
    // accumulators it is spread into.
    bool scatter_ = false;
    bool inputDone_ = false;
    int drainLeft_ = 0;
    int64_t jIn_ = 0;
    double tNext_ = 0.0;
    int accPos_ = 0;
    std::array<std::vector<T>, 2> acc_;
};

} // namespace dspark::detail
