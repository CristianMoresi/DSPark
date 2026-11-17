// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file DownbeatEstimator.h
 * @brief Which beats open a bar: the bar length and its first beat, offline.
 *
 * Given a signal and its beats (from Analysis/BeatTracker.h, a host's grid or
 * a tap), it names the beats that open a bar. It reads one value per beat for
 * each of six cues, asks of every candidate first beat how much of each cue
 * falls on the beats it would make first, and takes the candidate with the
 * most:
 *
 * - **Spectral change** - how far the spectrum of a beat is from that of the
 *   beat before it (a Jensen-Shannon divergence of the two, 30 Hz to 1.4 kHz
 *   at about 5 Hz of resolution). Chords and bass notes tend to change with
 *   the bar; the cue is the one of Davies and Plumbley (2006), read at a
 *   resolution that tells bass notes apart.
 * - **Change between pairs of bars** - how far the two bars after a beat are
 *   from the two before it, in pitch classes, in spectrum and in bass
 *   spectrum: three cues. A candidate that splits the phrases where the music
 *   does scores high on the three.
 * - **Low end at the beat** - the energy below 150 Hz from 30 ms before the
 *   beat to 120 ms after it, which a beat a little early or late still
 *   catches.
 * - **Depth of the bass** - how low the spectrum below 300 Hz sits at the
 *   beat: a kick drum or a root note pulls it down.
 *
 * The last two are the cues of a rhythm section, and they count in proportion
 * to how much of one the music has, read from how flat the spectra of its
 * beats are: in full where there are drums (the geometric mean of a beat's
 * spectrum above 0.45 of its arithmetic mean), not at all in music of pitched
 * lines alone (below 0.30), where a loud low note says nothing of the bar. On
 * drums alone it is the other four that say nothing: the spectrum changes
 * most at the loudest drum, and the low end is what marks the bar.
 *
 * Every cue is brought to zero mean and unit deviation over the beats, so
 * none has a scale to tune; a cue that barely moves (a click track, a held
 * chord) is kept from being blown up to that deviation by a floor far below
 * what music gives it, and the confidence then reads near zero. The bar
 * length, when the caller does not give it, is 3 when the beats resemble each
 * other more three apart than two or four apart (in low end, spectral change
 * and loudness) and a first beat stands out more in bars of three than of
 * four, and 4 otherwise: duple and quadruple time both read as 4. For this
 * the low end counts in full whatever the music: a cue that falls on the
 * wrong beat of the bar still tells its length.
 *
 * Measured on three annotated sets, each figure against a detector of the
 * spectral-difference kind given the same beats and bar length, as the mean
 * F-measure of the first beats within 70 ms. On 698 excerpts of dance music
 * (175 in three, 523 in four): with the annotated beats and bar length 0.872
 * against 0.747 (the right first beat in 87.1 % of the excerpts against
 * 74.6 %; in three 99.4 % against 98.9 %, in four 83.0 % against 66.5 %);
 * with the bar length estimated (right in 98.0 %) 0.868 against 0.740; with
 * BeatTracker's own beats (their F-measure 0.776) 0.628 against 0.567, and
 * 0.630 against 0.567 with the bar length estimated (right in 91.7 %). On 360
 * excerpts of solo guitar, all in four, with no drums and no bass: 0.628
 * against 0.550 (62.8 % against 55.0 %); 0.625 against 0.561 with the bar
 * length estimated (right in 93.1 %); with BeatTracker's own beats (0.520)
 * 0.258 against 0.252, and 0.272 against 0.266. On 243 takes of a drum kit
 * alone, 240 of them in four, with the beats of each take's tempo: 0.770
 * against 0.379 (77.0 % against 37.9 %); 0.759 against 0.385 with the bar
 * length estimated (right in 95.9 %); with BeatTracker's own beats (0.760)
 * 0.557 against 0.270, and 0.557 against 0.272. With the beats given, the
 * estimator is ahead or level in each of the twenty-eight styles of five
 * excerpts or more. With the library's own beats it is ahead in twenty-one,
 * level in one and behind in six, by 0.028 at most: beats that sit on the
 * off-beats or at another metrical level leave no first beat to be right. The
 * confidence sorts the answers: with the annotated beats, 98.9 %, 95.4 % and
 * 100 % are right in the three sets where it reads above 0.75, and 57.6 %,
 * 34.3 % and 57.4 % where it reads below 0.25.
 *
 * Not real-time: estimate() allocates and runs on the calling thread, over
 * the whole signal at once. It shares no state between calls: any thread.
 *
 * Dependencies: Core/DspMath.h, Core/FFT.h, Core/WindowFunctions.h.
 *
 * @code
 *   dspark::BeatTracker<float> tracker;
 *   tracker.prepare(spec);
 *   const auto beats = tracker.analyze(view);
 *   const auto bars = dspark::DownbeatEstimator<float>::estimate(
 *       std::span<const float>(view.getChannel(0), size_t(view.getNumSamples())),
 *       spec.sampleRate, beats.beatSamples);
 *   // bars.beatsPerBar, bars.downbeatSamples
 * @endcode
 */

#include "../Core/DspMath.h"
#include "../Core/FFT.h"
#include "../Core/WindowFunctions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <span>
#include <vector>

namespace dspark {

/**
 * @class DownbeatEstimator
 * @brief Names the beats that open a bar, from a signal and its beats.
 *
 * Role: offline analysis readout. It reads const audio and returns a value.
 *
 * @tparam T Sample type (float or double).
 */
template <FloatType T>
class DownbeatEstimator final
{
public:
    /** @brief What estimate() returns. */
    struct Result
    {
        /// Beats in a bar: the caller's, or 3 or 4 as estimated. 0: nothing was read.
        int beatsPerBar = 0;
        /// Index, into the beats given, of the first one that opens a bar. -1: nothing was read.
        int firstDownbeat = -1;
        /// How far the chosen first beat stands above the others, 0 (level) to 1.
        T confidence = T(0);
        /// The beats that open a bar, ascending: every beatsPerBar-th from firstDownbeat.
        std::vector<int64_t> downbeatSamples;
    };

    /// Fewest beats estimate() reads: two bars of four.
    static constexpr int kMinBeats = 8;
    /// Longest bar a caller may ask for, in beats.
    static constexpr int kMaxBeatsPerBar = 8;

    /**
     * @brief Estimates the bar length and the first beat of the bars.
     *
     * The bars are taken as equal in length from the first beat to the last:
     * a piece that changes its time signature, or drops a beat, gets the
     * bar line that fits most of it.
     *
     * @param mono        Mono samples of the whole signal. Samples that are
     *                    not finite read as zero.
     * @param sampleRate  Sample rate in Hz.
     * @param beatSamples The beats, as sample positions inside the signal,
     *                    strictly ascending.
     * @param beatsPerBar Beats in a bar, 2 to kMaxBeatsPerBar, or 0 to have
     *                    it estimated (as 3 or 4).
     * @return The bar length, the first beat that opens a bar and all of
     *         them. Empty (beatsPerBar 0, firstDownbeat -1) for an invalid
     *         rate or bar length, beats out of order or outside the signal,
     *         fewer than kMinBeats beats or than two bars and a beat of
     *         them, or a signal shorter than the analysis window (about
     *         0.19 s).
     */
    [[nodiscard]] static Result estimate(std::span<const T> mono, double sampleRate,
                                         std::span<const int64_t> beatSamples, int beatsPerBar = 0)
    {
        Result result;
        const size_t beats = beatSamples.size();
        if (!std::isfinite(sampleRate) || sampleRate <= 0.0 || mono.empty()) return result;
        if (beatsPerBar != 0 && (beatsPerBar < 2 || beatsPerBar > kMaxBeatsPerBar)) return result;
        if (beats < static_cast<size_t>(kMinBeats)) return result;
        for (size_t k = 0; k < beats; ++k)
        {
            if (beatSamples[k] < 0 || static_cast<uint64_t>(beatSamples[k]) >= mono.size()) return result;
            if (k > 0 && beatSamples[k] <= beatSamples[k - 1]) return result;
        }

        Readings r;
        if (!read(mono, sampleRate, beatSamples, r)) return result;

        // How much the two low-end cues count towards the first beat: in full
        // where the beats' spectra are as flat as a drum kit's, not at all
        // where they are those of pitched lines alone.
        const double rhythm = std::clamp((r.flatness - kTonalFlatness) / (kNoisyFlatness - kTonalFlatness), 0.0, 1.0);

        // The shares of bars of the caller's length, or of three or four:
        // three when the beats resemble each other more three apart than two
        // or four apart, and a first beat stands out more in bars of three.
        // The bar length is read with the low end in full whatever the music:
        // a cue that falls on the wrong beat of the bar still tells its length.
        std::array<double, kMaxBeatsPerBar> score{};
        int meter = beatsPerBar;
        if (meter != 0)
            score = shares(r, static_cast<size_t>(meter)).with(rhythm);
        else
        {
            const Shares inThree = shares(r, 3), inFour = shares(r, 4);
            const double alike = likeness(r, 3) - std::max(likeness(r, 2), likeness(r, 4));
            const double standsOut = standOut(inThree.with(1.0), 3) - standOut(inFour.with(1.0), 4);
            const bool three = alike + kStandOutWeight * standsOut > 0.0;
            meter = three ? 3 : 4;
            score = (three ? inThree : inFour).with(rhythm);
        }
        const size_t m = static_cast<size_t>(meter);
        if (beats < 2 * m + 1) return result;

        size_t best = 0;
        for (size_t phase = 1; phase < m; ++phase)
            if (score[phase] > score[best]) best = phase;

        // Confidence: the chosen candidate's share of exp(score), from level (0) to all of it (1).
        double total = 0.0;
        for (size_t phase = 0; phase < m; ++phase) total += std::exp(score[phase] - score[best]);
        const double level = 1.0 / static_cast<double>(m);
        result.confidence = static_cast<T>(std::clamp((1.0 / total - level) / (1.0 - level), 0.0, 1.0));
        result.beatsPerBar = meter;
        result.firstDownbeat = static_cast<int>(best);
        result.downbeatSamples.reserve(beats / m + 1);
        for (size_t k = best; k < beats; k += m) result.downbeatSamples.push_back(beatSamples[k]);
        return result;
    }

private:
    /// What is read of the signal at each beat.
    struct Readings
    {
        std::vector<double> lowEnd;      ///< Log energy below 150 Hz around the beat.
        std::vector<double> loudness;    ///< Log energy of the beat, to the next one.
        std::vector<double> fineChange;  ///< Divergence of the beat's spectrum from the one before (0 at the first).
        std::vector<double> bassDepth;   ///< How low the beat's bass sits: its centroid below 300 Hz, in Hz, negated.
        double flatness = 0.0;           ///< Mean log flatness of the beats' spectra (0: noise; far below: pitches).
        std::vector<double> fine;        ///< Per beat, the mean magnitude spectrum to 1.4 kHz: beats x bins.
        std::vector<double> tone;        ///< Per beat, its twelve pitch classes: beats x 12.
        size_t bins = 0;                 ///< Bins of a spectrum in `fine`.
        size_t bassBins = 0;             ///< Of those, the ones below 300 Hz.
    };

    static constexpr double kLowCutHz = 150.0;       ///< Corner of the low-end reading.
    static constexpr double kLowBefore = 0.03;       ///< Its window: from this long before the beat...
    static constexpr double kLowAfter = 0.12;        ///< ...to this long after it, in seconds.
    static constexpr double kWindowSeconds = 0.186;  ///< Analysis window (8192 samples at 44.1 and 48 kHz).
    static constexpr double kTopHz = 1400.0;         ///< Top of the spectra.
    static constexpr double kBassTopHz = 300.0;      ///< Top of the bass spectra.
    static constexpr double kChangeFromHz = 30.0;    ///< The beat-to-beat change reads from here up.
    static constexpr double kToneFromHz = 55.0;      ///< Pitch classes are read from here up.
    /// Least deviations a cue is scaled by: of the two log energies (about 0.4
    /// dB) and of the divergences and distances.
    static constexpr double kLogFloor = 0.05;
    static constexpr double kChangeFloor = 1e-3;
    static constexpr double kDepthFloor = 1.0;       ///< The same for the depth of the bass, in Hz.
    /// The two low-end cues count in full from this flatness of the beats'
    /// spectra up, and not at all from the second one down: natural logs of
    /// the ratio of a spectrum's geometric mean to its arithmetic mean (0.45
    /// and 0.30). Drums and full mixes sit above the first, pitched lines
    /// alone below the second.
    static constexpr double kNoisyFlatness = -0.8;
    static constexpr double kTonalFlatness = -1.2;
    /// A beat whose spectrum sums to less than this share of the loudest
    /// beat's is silent: it says nothing of how flat the music is.
    static constexpr double kSilentShare = 1e-6;
    /// What the bar length takes of how much more a first beat stands out in
    /// bars of three than of four, next to how alike the beats are.
    static constexpr double kStandOutWeight = 0.25;

    /// Reads the per-beat values; false when the signal is shorter than a window.
    static bool read(std::span<const T> mono, double rate, std::span<const int64_t> beatSamples, Readings& r)
    {
        const size_t n = mono.size(), beats = beatSamples.size();
        const auto clean = [](T v) noexcept { return std::isfinite(v) ? static_cast<double>(v) : 0.0; };
        const auto seconds = [&](size_t k) noexcept { return static_cast<double>(beatSamples[k]) / rate; };
        const auto period = [&](size_t k) noexcept {
            const size_t from = k + 1 < beats ? k : k - 1;
            return seconds(from + 1) - seconds(from);
        };
        const auto sampleAt = [&](double t) noexcept {
            const int64_t at = static_cast<int64_t>(t * rate);
            return static_cast<size_t>(std::clamp(at, int64_t{ 0 }, static_cast<int64_t>(n)));
        };

        // Analysis window: the power of two nearest to kWindowSeconds, a frame every quarter of it.
        size_t size = 256;
        while (static_cast<double>(size) * 1.41421356237309505 < kWindowSeconds * rate) size *= 2;
        if (n < size) return false;
        const size_t hop = size / 4;
        const size_t frames = (n - size) / hop + 1;
        const double binHz = rate / static_cast<double>(size);
        const double hopSeconds = static_cast<double>(hop) / rate, half = 0.5 * static_cast<double>(size) / rate;
        r.bins = std::min(size / 2 + 1, static_cast<size_t>(kTopHz / binHz) + 1);
        r.bassBins = std::min(r.bins, static_cast<size_t>(kBassTopHz / binHz) + 1);
        const size_t changeFrom = std::max<size_t>(1, static_cast<size_t>(kChangeFromHz / binHz));
        const auto frameAt = [&](double t) noexcept {
            const double f = std::floor((t - half) / hopSeconds + 0.5);
            return static_cast<size_t>(std::clamp(f, 0.0, static_cast<double>(frames - 1)));
        };

        // One pass over the samples: the low end around every beat and the loudness of every beat.
        r.lowEnd.assign(beats, 0.0);
        r.loudness.assign(beats, 0.0);
        std::vector<size_t> lowFrom(beats), lowTo(beats), beatFrom(beats), beatTo(beats);
        for (size_t k = 0; k < beats; ++k)
        {
            lowFrom[k] = sampleAt(seconds(k) - kLowBefore);
            lowTo[k] = sampleAt(seconds(k) + kLowAfter);
            beatFrom[k] = sampleAt(seconds(k));
            beatTo[k] = sampleAt(seconds(k) + period(k));
        }
        {
            const double pole = std::exp(-2.0 * std::numbers::pi * kLowCutHz / rate);
            double y1 = 0.0, y2 = 0.0;
            size_t lowFirst = 0, beatFirst = 0;
            for (size_t i = 0; i < n; ++i)
            {
                const double x = clean(mono[i]);
                y1 = (1.0 - pole) * x + pole * y1;
                y2 = (1.0 - pole) * y1 + pole * y2;
                while (lowFirst < beats && lowTo[lowFirst] <= i) ++lowFirst;
                for (size_t k = lowFirst; k < beats && lowFrom[k] <= i; ++k)
                    if (i < lowTo[k]) r.lowEnd[k] += y2 * y2;
                while (beatFirst < beats && beatTo[beatFirst] <= i) ++beatFirst;
                for (size_t k = beatFirst; k < beats && beatFrom[k] <= i; ++k)
                    if (i < beatTo[k]) r.loudness[k] += x * x;
            }
            for (size_t k = 0; k < beats; ++k)
            {
                const size_t low = lowTo[k] - lowFrom[k], whole = beatTo[k] - beatFrom[k];
                r.lowEnd[k] = std::log(1e-12 + (low > 0 ? r.lowEnd[k] / static_cast<double>(low) : 0.0));
                r.loudness[k] = std::log(1e-12 + (whole > 0 ? r.loudness[k] / static_cast<double>(whole) : 0.0));
            }
        }

        // One pass over the frames: every beat's mean magnitude spectrum, over the frames centred in it.
        r.fine.assign(beats * r.bins, 0.0);
        std::vector<size_t> frameFrom(beats), frameTo(beats);
        for (size_t k = 0; k < beats; ++k)
        {
            frameFrom[k] = frameAt(seconds(k));
            frameTo[k] = std::max(frameFrom[k] + 1, frameAt(seconds(k) + period(k)));
        }
        {
            FFTReal<T> fft(size);
            std::vector<T> window(size), frame(size), spectrum(size + 2);
            WindowFunctions<T>::hann(window.data(), static_cast<int>(size));
            size_t first = 0;
            for (size_t f = 0; f < frames; ++f)
            {
                while (first < beats && frameTo[first] <= f) ++first;
                if (first == beats) break;
                if (frameFrom[first] > f) continue;
                const T* x = mono.data() + f * hop;
                for (size_t i = 0; i < size; ++i) frame[i] = (std::isfinite(x[i]) ? x[i] : T(0)) * window[i];
                fft.forward(frame.data(), spectrum.data());
                for (size_t k = first; k < beats && frameFrom[k] <= f; ++k)
                {
                    if (f >= frameTo[k]) continue;
                    double* row = r.fine.data() + k * r.bins;
                    for (size_t b = 0; b < r.bins; ++b)
                    {
                        const double re = static_cast<double>(spectrum[2 * b]);
                        const double im = static_cast<double>(spectrum[2 * b + 1]);
                        row[b] += std::sqrt(re * re + im * im);
                    }
                }
            }
        }

        // Per beat: its pitch classes, how far its spectrum is from the one
        // before, how low its bass sits and how flat its spectrum is.
        r.tone.assign(beats * 12, 0.0);
        r.fineChange.assign(beats, 0.0);
        r.bassDepth.assign(beats, 0.0);
        std::vector<double> level(beats, 0.0), flat(beats, 0.0);
        for (size_t k = 0; k < beats; ++k)
        {
            double* row = r.fine.data() + k * r.bins;
            const double count = static_cast<double>(frameTo[k] - frameFrom[k]);
            for (size_t b = 0; b < r.bins; ++b) row[b] /= count;
            double weight = 1e-12, moment = 0.0;
            for (size_t b = 1; b < r.bassBins; ++b)
            {
                weight += row[b];
                moment += static_cast<double>(b) * row[b];
            }
            r.bassDepth[k] = -binHz * moment / weight;
            if (r.bins > changeFrom)
            {
                double logs = 0.0;
                for (size_t b = changeFrom; b < r.bins; ++b)
                {
                    level[k] += row[b];
                    logs += std::log(1e-18 + row[b]);
                }
                const double counted = static_cast<double>(r.bins - changeFrom);
                flat[k] = logs / counted - std::log(1e-18 + level[k] / counted);
            }
            double* tone = r.tone.data() + k * 12;
            for (size_t b = 1; b < r.bins; ++b)
            {
                const double hz = static_cast<double>(b) * binHz;
                if (hz < kToneFromHz || hz > kTopHz) continue;
                const int note = static_cast<int>(std::floor(12.0 * std::log2(hz / 440.0) + 0.5));
                tone[static_cast<size_t>(((note % 12) + 12) % 12)] += row[b];
            }
            if (k > 0) r.fineChange[k] = divergence(row - r.bins, row, changeFrom, r.bins);
        }
        // How flat the spectra are, over the beats that sound.
        {
            const double loudest = *std::max_element(level.begin(), level.end());
            double sum = 0.0;
            size_t sounding = 0;
            for (size_t k = 0; k < beats; ++k)
                if (level[k] > kSilentShare * loudest)
                {
                    sum += flat[k];
                    ++sounding;
                }
            r.flatness = sounding > 0 ? sum / static_cast<double>(sounding) : 0.0;
        }
        return true;
    }

    /// Jensen-Shannon divergence of two spectra over the bins [from, to), each taken as a distribution.
    [[nodiscard]] static double divergence(const double* p, const double* q, size_t from, size_t to) noexcept
    {
        double sumP = 1e-12, sumQ = 1e-12;
        for (size_t b = from; b < to; ++b)
        {
            sumP += p[b];
            sumQ += q[b];
        }
        double js = 0.0;
        for (size_t b = from; b < to; ++b)
        {
            const double a = p[b] / sumP + 1e-9, c = q[b] / sumQ + 1e-9;
            const double mean = 0.5 * (a + c);
            js += 0.5 * a * std::log(a / mean) + 0.5 * c * std::log(c / mean);
        }
        return js;
    }

    /// One minus the cosine of the angle between two vectors.
    [[nodiscard]] static double cosineDistance(const std::vector<double>& a, const std::vector<double>& b) noexcept
    {
        double ab = 0.0, aa = 1e-18, bb = 1e-18;
        for (size_t i = 0; i < a.size(); ++i)
        {
            ab += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        return 1.0 - ab / std::sqrt(aa * bb);
    }

    /// The sum of rows [from, to) of a table of `width` columns.
    static void sumRows(const std::vector<double>& table, size_t width, size_t from, size_t to,
                        std::vector<double>& sum) noexcept
    {
        std::fill(sum.begin(), sum.begin() + static_cast<std::ptrdiff_t>(width), 0.0);
        for (size_t k = from; k < to; ++k)
        {
            const double* row = table.data() + k * width;
            for (size_t i = 0; i < width; ++i) sum[i] += row[i];
        }
    }

    /// The cues' shares on each candidate first beat, in two sums: the cues
    /// that count in any music and the two of the low end.
    struct Shares
    {
        std::array<double, kMaxBeatsPerBar> always{};  ///< The beat-to-beat change and the three between pairs of bars.
        std::array<double, kMaxBeatsPerBar> lowEnd{};  ///< The low end at the beat and the depth of the bass.

        /// The two sums added up, the low end counted `low` times (0 to 1).
        [[nodiscard]] std::array<double, kMaxBeatsPerBar> with(double low) const noexcept
        {
            std::array<double, kMaxBeatsPerBar> sum{};
            for (size_t phase = 0; phase < sum.size(); ++phase) sum[phase] = always[phase] + low * lowEnd[phase];
            return sum;
        }
    };

    /// Every cue's share on each candidate first beat of bars of `m` beats,
    /// each cue counted once: the beat-to-beat change and the three changes
    /// between pairs of bars in one sum, the low end and the depth of the
    /// bass in the other. In music of pitched lines alone a loud low note
    /// says nothing of where the bar starts, so the caller weighs the second
    /// sum by how much of a rhythm section the music has.
    [[nodiscard]] static Shares shares(const Readings& r, size_t m)
    {
        const size_t beats = r.lowEnd.size();
        Shares found;
        std::array<double, kMaxBeatsPerBar>& score = found.always;
        addShare(r.fineChange, 1, beats, m, 1.0, kChangeFloor, score);
        addShare(r.lowEnd, 1, beats, m, 1.0, kLogFloor, found.lowEnd);
        addShare(r.bassDepth, 1, beats, m, 1.0, kDepthFloor, found.lowEnd);
        const size_t span = 2 * m;
        if (beats >= 2 * span + 2)
        {
            std::vector<double> tone(beats, 0.0), spectrum(beats, 0.0), bass(beats, 0.0);
            std::vector<double> before(r.bins), after(r.bins), toneBefore(12), toneAfter(12);
            for (size_t k = span; k + span <= beats; ++k)
            {
                sumRows(r.tone, 12, k - span, k, toneBefore);
                sumRows(r.tone, 12, k, k + span, toneAfter);
                tone[k] = cosineDistance(toneBefore, toneAfter);
                sumRows(r.fine, r.bins, k - span, k, before);
                sumRows(r.fine, r.bins, k, k + span, after);
                spectrum[k] = divergence(before.data(), after.data(), 1, r.bins);
                bass[k] = divergence(before.data(), after.data(), 1, r.bassBins);
            }
            addShare(tone, span, beats - span + 1, m, 1.0, kChangeFloor, score);
            addShare(spectrum, span, beats - span + 1, m, 1.0, kChangeFloor, score);
            addShare(bass, span, beats - span + 1, m, 1.0, kChangeFloor, score);
        }
        return found;
    }

    /// How far the best candidate first beat stands above the mean of the others.
    [[nodiscard]] static double standOut(const std::array<double, kMaxBeatsPerBar>& score, size_t m) noexcept
    {
        double best = score[0], sum = 0.0;
        for (size_t phase = 0; phase < m; ++phase)
        {
            best = std::max(best, score[phase]);
            sum += score[phase];
        }
        return best - (sum - best) / static_cast<double>(m - 1);
    }

    /// Adds to `score`, per candidate first beat, `weight` times the mean of
    /// the cue over the beats that candidate makes first, the cue at zero
    /// mean and unit deviation over the beats [from, to) that carry it (a
    /// deviation under `floor` counts as `floor`).
    static void addShare(const std::vector<double>& cue, size_t from, size_t to, size_t meter, double weight,
                         double floor, std::array<double, kMaxBeatsPerBar>& score) noexcept
    {
        if (to <= from + 1) return;
        const double count = static_cast<double>(to - from);
        double mean = 0.0, variance = 0.0;
        for (size_t k = from; k < to; ++k) mean += cue[k] / count;
        for (size_t k = from; k < to; ++k) variance += (cue[k] - mean) * (cue[k] - mean) / count;
        const double scale = 1.0 / std::sqrt(variance + floor * floor);
        std::array<double, kMaxBeatsPerBar> sum{};
        std::array<int, kMaxBeatsPerBar> counted{};
        for (size_t k = from; k < to; ++k)
        {
            sum[k % meter] += (cue[k] - mean) * scale;
            ++counted[k % meter];
        }
        for (size_t phase = 0; phase < meter; ++phase)
            if (counted[phase] > 0) score[phase] += weight * sum[phase] / static_cast<double>(counted[phase]);
    }

    /// How alike the beats `lag` apart are: the mean, over the low end, the
    /// beat-to-beat change and the loudness, of the cue's autocorrelation at
    /// that lag (the first beat left out).
    [[nodiscard]] static double likeness(const Readings& r, size_t lag) noexcept
    {
        double total = 0.0;
        int used = 0;
        const std::array<const std::vector<double>*, 3> cues = { &r.lowEnd, &r.fineChange, &r.loudness };
        const std::array<double, 3> floors = { kLogFloor, kChangeFloor, kLogFloor };
        for (size_t c = 0; c < cues.size(); ++c)
        {
            const std::vector<double>& v = *cues[c];
            if (v.size() < lag + 4) continue;
            const double count = static_cast<double>(v.size() - 1);
            double mean = 0.0, variance = floors[c] * floors[c], covariance = 0.0;
            for (size_t k = 1; k < v.size(); ++k) mean += v[k] / count;
            for (size_t k = 1; k < v.size(); ++k) variance += (v[k] - mean) * (v[k] - mean) / count;
            for (size_t k = 1; k + lag < v.size(); ++k)
                covariance += (v[k] - mean) * (v[k + lag] - mean) / static_cast<double>(v.size() - 1 - lag);
            total += covariance / variance;
            ++used;
        }
        return used > 0 ? total / used : 0.0;
    }
};

} // namespace dspark
