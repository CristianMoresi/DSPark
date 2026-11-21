// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file FilterDesign.h
 * @brief Analog prototypes and the digital sections that follow their
 *        magnitude up to the Nyquist frequency.
 *
 * A digital filter designed by the bilinear transform has the response of its
 * analog prototype squeezed into the band below the Nyquist frequency: a bell
 * near the top of the band comes out narrower than asked, a shelf reaches its
 * plateau early and a low-pass falls off a cliff. The designs here describe a
 * filter as analog sections (AnalogSection, or an AnalogCascade of them) and
 * turn each into the biquad whose magnitude stays on the analog curve:
 *
 * 1. the poles are the analog poles carried over by impulse invariance, so a
 *    resonance keeps its frequency and its width wherever it is;
 * 2. the numerator is the least-squares fit to the analog power over the band
 *    with those poles. On the unit circle the squared magnitude of a biquad
 *    polynomial is a quadratic in 1 - cos w, so the fit is a linear problem
 *    (M. Vicanek, "Matched Second Order Digital Filters", 2016, solves it
 *    through three points);
 * 3. where that is not within a hundredth of a decibel, up to kFitSteps
 *    steps move numerator and denominator together. Each step is again a
 *    linear least-squares problem, weighted so that what it minimises is the
 *    relative error (C. K. Sanathanan and J. Koerner, 1963). There is nothing
 *    to tune and no search: the same input always takes the same few steps.
 *
 * The fit looks at sixteen frequencies spread over the band and at fourteen
 * more around the poles and the zeros of the section, so a narrow peak is
 * seen however narrow it is. Up to 20 kHz every frequency counts the same;
 * above, less and less.
 *
 * Zeros a prototype has at zero frequency or on the frequency axis (a
 * high-pass, a band-pass, a notch) are kept exactly where they are, and so
 * is the gain at zero frequency of every section that has one.
 *
 * On top of the sections, FilterDesign builds the cascades of an equalizer:
 * cuts of any slope from 0 to 96 dB per octave (the slope is a real number),
 * bells and shelves of higher order with a flat top, band-pass, band-stop
 * and all-pass of any order, a tilt of constant slope across the band, and
 * an elliptic cut that drops like a wall.
 *
 * Threading: every function is a pure function of its arguments. None
 * allocates and the work of each is bounded, so a design can run on the
 * audio thread. A FilterDesign::Band is a plain value: prepare it where the
 * sample rate is set and read it from any thread afterwards.
 *
 * Dependencies: Biquad.h.
 */

#include "Biquad.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <numbers>

namespace dspark {

/**
 * @struct AnalogSection
 * @brief A first or second order analog section, in radians per second:
 *        (b2 s^2 + b1 s + b0) / (a2 s^2 + a1 s + a0).
 */
struct AnalogSection
{
    double b2 = 0.0, b1 = 0.0, b0 = 1.0;
    double a2 = 0.0, a1 = 0.0, a0 = 1.0;

    /** @brief Squared magnitude at an angular frequency (rad/s). */
    [[nodiscard]] double magnitudeSquared(double omega) const noexcept
    {
        const double w2 = omega * omega;
        const double nr = b0 - b2 * w2, ni = b1 * omega;
        const double dr = a0 - a2 * w2, di = a1 * omega;
        return (nr * nr + ni * ni) / (dr * dr + di * di);
    }

    /** @brief Magnitude in decibels at a frequency in hertz. */
    [[nodiscard]] double magnitudeDb(double hz) const noexcept
    {
        return 10.0 * std::log10(std::max(magnitudeSquared(2.0 * std::numbers::pi * hz), 1e-30));
    }

    /** @brief Whether the denominator is of second order. */
    [[nodiscard]] bool isSecondOrder() const noexcept { return a2 != 0.0; }

    /** @brief The frequency of the poles, in hertz (the corner of a first-order section). */
    [[nodiscard]] double poleFrequency() const noexcept
    {
        const double omega = isSecondOrder() ? std::sqrt(std::abs(a0 / a2)) : std::abs(a0 / a1);
        return omega / (2.0 * std::numbers::pi);
    }

    // -- Prototypes (the cookbook's analog forms) -------------------------------

    /** @brief Second-order low-pass, gain Q at the corner. */
    [[nodiscard]] static AnalogSection lowPass(double hz, double q = 0.7071067811865476) noexcept
    {
        const double w = omegaOf(hz);
        return make(0.0, 0.0, w * w, 1.0, w / safeQ(q), w * w);
    }

    /** @brief Second-order high-pass, gain Q at the corner. */
    [[nodiscard]] static AnalogSection highPass(double hz, double q = 0.7071067811865476) noexcept
    {
        const double w = omegaOf(hz);
        return make(1.0, 0.0, 0.0, 1.0, w / safeQ(q), w * w);
    }

    /** @brief Band-pass, 0 dB at the centre. */
    [[nodiscard]] static AnalogSection bandPass(double hz, double q = 0.7071067811865476) noexcept
    {
        const double w = omegaOf(hz);
        return make(0.0, w / safeQ(q), 0.0, 1.0, w / safeQ(q), w * w);
    }

    /** @brief Notch at the centre, 0 dB away from it. */
    [[nodiscard]] static AnalogSection notch(double hz, double q = 0.7071067811865476) noexcept
    {
        const double w = omegaOf(hz);
        return make(1.0, 0.0, w * w, 1.0, w / safeQ(q), w * w);
    }

    /** @brief Second-order all-pass (the magnitude is 1; only the phase turns). */
    [[nodiscard]] static AnalogSection allPass(double hz, double q = 0.7071067811865476) noexcept
    {
        const double w = omegaOf(hz);
        return make(1.0, -w / safeQ(q), w * w, 1.0, w / safeQ(q), w * w);
    }

    /** @brief Bell: gainDb at the centre, bandwidth hz / Q at half that gain in decibels. */
    [[nodiscard]] static AnalogSection peak(double hz, double q, double gainDb) noexcept
    {
        const double w = omegaOf(hz), A = std::pow(10.0, gainDb / 40.0);
        return make(1.0, w * A / safeQ(q), w * w, 1.0, w / (A * safeQ(q)), w * w);
    }

    /** @brief Second-order low shelf: gainDb below the corner, half of it (in decibels) at the corner. */
    [[nodiscard]] static AnalogSection lowShelf(double hz, double gainDb, double slope = 1.0) noexcept
    {
        const double w = omegaOf(hz), A = std::pow(10.0, gainDb / 40.0), d = shelfDamping(A, slope);
        return make(A, A * std::sqrt(A) * d * w, A * A * w * w, A, std::sqrt(A) * d * w, w * w);
    }

    /** @brief Second-order high shelf: gainDb above the corner, half of it (in decibels) at the corner. */
    [[nodiscard]] static AnalogSection highShelf(double hz, double gainDb, double slope = 1.0) noexcept
    {
        const double w = omegaOf(hz), A = std::pow(10.0, gainDb / 40.0), d = shelfDamping(A, slope);
        return make(A * A, A * std::sqrt(A) * d * w, A * w * w, 1.0, std::sqrt(A) * d * w, A * w * w);
    }

    /**
     * @brief Second-order low shelf by its Q: gainDb below the corner, half of
     *        it (in decibels) at the corner whatever the Q. At 0.7071 it is
     *        lowShelf() at a slope of 1; above, the response rises over the
     *        plateau before the corner and dips under 0 dB after it, more the
     *        higher the Q; a cut is the exact inverse of the boost.
     */
    [[nodiscard]] static AnalogSection lowShelfQ(double hz, double gainDb, double q) noexcept
    {
        const double w = omegaOf(hz), A = std::pow(10.0, gainDb / 40.0), d = 1.0 / std::clamp(q, 0.05, 100.0);
        return make(A, A * std::sqrt(A) * d * w, A * A * w * w, A, std::sqrt(A) * d * w, w * w);
    }

    /** @brief Second-order high shelf by its Q: lowShelfQ() mirrored about the corner. */
    [[nodiscard]] static AnalogSection highShelfQ(double hz, double gainDb, double q) noexcept
    {
        const double w = omegaOf(hz), A = std::pow(10.0, gainDb / 40.0), d = 1.0 / std::clamp(q, 0.05, 100.0);
        return make(A * A, A * std::sqrt(A) * d * w, A * w * w, 1.0, std::sqrt(A) * d * w, A * w * w);
    }

    /**
     * @brief Second-order tilt: 0 dB at the pivot, half of gainDb down far
     *        below it and half of it up far above, 12 dB per octave at most
     *        on the way; the Q as in lowShelfQ().
     */
    [[nodiscard]] static AnalogSection tiltSecondOrder(double hz, double gainDb, double q = 0.7071067811865476) noexcept
    {
        // A low shelf of -gainDb with half of gainDb back over the whole band.
        const double w = omegaOf(hz), A = std::pow(10.0, -gainDb / 40.0), d = 1.0 / std::clamp(q, 0.05, 100.0);
        return make(1.0, std::sqrt(A) * d * w, A * w * w, A, std::sqrt(A) * d * w, w * w);
    }

    /** @brief First-order low-pass, -3 dB at the corner. */
    [[nodiscard]] static AnalogSection firstOrderLowPass(double hz) noexcept
    {
        const double w = omegaOf(hz);
        return make(0.0, 0.0, w, 0.0, 1.0, w);
    }

    /** @brief First-order high-pass, -3 dB at the corner. */
    [[nodiscard]] static AnalogSection firstOrderHighPass(double hz) noexcept
    {
        const double w = omegaOf(hz);
        return make(0.0, 1.0, 0.0, 0.0, 1.0, w);
    }

    /** @brief First-order low shelf: gainDb at DC, half of it (in decibels) at the corner. */
    [[nodiscard]] static AnalogSection firstOrderLowShelf(double hz, double gainDb) noexcept
    {
        const double w = omegaOf(hz), root = std::pow(10.0, gainDb / 40.0);
        return make(0.0, 1.0, w * root, 0.0, 1.0, w / root);
    }

    /** @brief First-order high shelf: gainDb at the top, half of it (in decibels) at the corner. */
    [[nodiscard]] static AnalogSection firstOrderHighShelf(double hz, double gainDb) noexcept
    {
        const double w = omegaOf(hz), root = std::pow(10.0, gainDb / 40.0);
        return make(0.0, root, w, 0.0, 1.0 / root, w);
    }

    /** @brief First-order tilt: 0 dB at the pivot, half of gainDb down far below it and half of it up far above. */
    [[nodiscard]] static AnalogSection tilt(double hz, double gainDb) noexcept
    {
        const double w = omegaOf(hz), root = std::pow(10.0, gainDb / 40.0);
        return make(0.0, root, w, 0.0, 1.0, w * root);
    }

    /** @brief Builds a section from its six coefficients. */
    [[nodiscard]] static AnalogSection make(double b2, double b1, double b0,
                                            double a2, double a1, double a0) noexcept
    {
        AnalogSection s;
        s.b2 = b2; s.b1 = b1; s.b0 = b0;
        s.a2 = a2; s.a1 = a1; s.a0 = a0;
        return s;
    }

private:
    [[nodiscard]] static double omegaOf(double hz) noexcept { return 2.0 * std::numbers::pi * std::max(hz, 1e-3); }
    [[nodiscard]] static double safeQ(double q) noexcept { return std::max(q, 1e-3); }
    [[nodiscard]] static double shelfDamping(double A, double slope) noexcept
    {
        const double s = std::clamp(slope, 1e-3, 1.0);
        return std::sqrt(std::max(0.0, (A + 1.0 / A) * (1.0 / s - 1.0) + 2.0));
    }
};

/**
 * @struct AnalogCascade
 * @brief Up to kMaxSections analog sections in series.
 */
struct AnalogCascade
{
    static constexpr int kMaxSections = 16;

    std::array<AnalogSection, kMaxSections> sections {};
    int count = 0;

    /** @brief Appends a section (ignored once the cascade is full). */
    void add(const AnalogSection& section) noexcept
    {
        if (count < kMaxSections) sections[static_cast<size_t>(count++)] = section;
    }

    /** @brief Squared magnitude of the whole cascade at an angular frequency (rad/s). */
    [[nodiscard]] double magnitudeSquared(double omega) const noexcept
    {
        double m = 1.0;
        for (int i = 0; i < count; ++i) m *= sections[static_cast<size_t>(i)].magnitudeSquared(omega);
        return m;
    }

    /** @brief Magnitude of the whole cascade in decibels at a frequency in hertz. */
    [[nodiscard]] double magnitudeDb(double hz) const noexcept
    {
        return 10.0 * std::log10(std::max(magnitudeSquared(2.0 * std::numbers::pi * hz), 1e-30));
    }
};

/**
 * @class FilterDesign
 * @brief Digital sections from analog ones, and the cascades of an equalizer.
 *
 * See the file notes for the method. The figures (worst case and mean
 * distance to the analog magnitude, and the cost of a design) are in the
 * changelog entry of this header and are asserted by its tests.
 */
class FilterDesign
{
public:
    static constexpr int kFitSteps = 2;     ///< Most joint steps a fit takes.

    /**
     * @class Band
     * @brief The frequencies a design is fitted over at one sample rate.
     *
     * From 20 Hz to 20 kHz, or to 0.42 of the rate where that is higher, and
     * never past 0.4536 of the rate (what 20 kHz is of 44.1 kHz: closer to
     * Nyquist no biquad follows an analog curve). An object that designs
     * sections again and again keeps one Band and saves the sines of its
     * frequencies.
     */
    class Band
    {
    public:
        static constexpr int kPoints = 16;   ///< Frequencies spread evenly in octaves.
        static constexpr double kTopShare = 20000.0 / 44100.0 + 1e-4;   ///< The band never goes past this share of the rate.

        Band() noexcept { prepare(48000.0); }
        explicit Band(double sampleRate) noexcept { prepare(sampleRate); }

        /** @brief Sets the sample rate (an invalid one leaves 48 kHz). */
        void prepare(double sampleRate) noexcept
        {
            fs_ = (std::isfinite(sampleRate) && sampleRate > 0.0) ? sampleRate : 48000.0;
            highHz_ = std::min(kTopShare * fs_, std::max(20000.0, 0.42 * fs_));
            lowHz_ = std::min(20.0, 0.01 * highHz_);
            const double ratio = std::pow(highHz_ / lowHz_, 1.0 / (kPoints - 1));
            double hz = lowHz_;
            for (int i = 0; i < kPoints; ++i, hz *= ratio)
            {
                const double omega = 2.0 * std::numbers::pi * std::min(hz, highHz_);
                omega2_[i] = omega * omega;
                y_[i] = yOf(omega / fs_);
                weight_[i] = weightOf(omega);
            }
        }

        [[nodiscard]] double getSampleRate() const noexcept { return fs_; }
        [[nodiscard]] double getLowHz() const noexcept { return lowHz_; }
        [[nodiscard]] double getHighHz() const noexcept { return highHz_; }

    private:
        friend class FilterDesign;
        double fs_ = 48000.0, lowHz_ = 20.0, highHz_ = 20000.0;
        double omega2_[kPoints] {};      // squared angular frequency of each point (rad/s)
        double y_[kPoints] {};           // 1 - cos of its digital frequency
        double weight_[kPoints] {};      // how much its error counts
    };

    // -- One section ------------------------------------------------------------

    /**
     * @brief The bilinear transform of a section, with one frequency mapped exactly.
     *
     * @param section    Analog section.
     * @param sampleRate Sample rate in Hz.
     * @param exactHz    Frequency that keeps its place (from 0.49 of the
     *                   rate up, none does and the plain transform is used).
     */
    [[nodiscard]] static BiquadCoeffs bilinear(const AnalogSection& section, double sampleRate, double exactHz) noexcept
    {
        const double c = (exactHz > 0.0 && exactHz < 0.49 * sampleRate)
                       ? 2.0 * kPi * exactHz / std::tan(kPi * exactHz / sampleRate)
                       : 2.0 * sampleRate;
        const double c2 = c * c;
        const double a0 = section.a2 * c2 + section.a1 * c + section.a0;
        BiquadCoeffs out;
        if (!(std::abs(a0) > 0.0)) return out;
        const double inverse = 1.0 / a0;
        out.b0 = (section.b2 * c2 + section.b1 * c + section.b0) * inverse;
        out.b1 = 2.0 * (section.b0 - section.b2 * c2) * inverse;
        out.b2 = (section.b2 * c2 - section.b1 * c + section.b0) * inverse;
        out.a1 = 2.0 * (section.a0 - section.a2 * c2) * inverse;
        out.a2 = (section.a2 * c2 - section.a1 * c + section.a0) * inverse;
        return out;
    }

    /**
     * @brief The biquad whose magnitude follows an analog section up to Nyquist.
     *
     * The section returned is minimum phase and has positive gain, whatever
     * the sign and the phase of the prototype; an all-pass prototype gives an
     * all-pass section with the poles of the prototype.
     *
     * @param section Analog section with its poles in the left half plane.
     * @param band    The frequencies of the fit (it carries the sample rate).
     * @return A stable section (the bilinear transform of the section when
     *         the prototype is not one this design can follow).
     */
    [[nodiscard]] static BiquadCoeffs digital(const AnalogSection& section, const Band& band) noexcept
    {
        const double fs = band.fs_;
        if (!finite(section)) return BiquadCoeffs {};
        if (!usable(section)) return bilinear(section, fs, section.poleFrequency());

        const Shape shape = shapeOf(section, fs);
        const Denominator start = invariantDenominator(section, fs);
        if (shape.allPass)
        {
            BiquadCoeffs out;
            if (shape.first) { out.b0 = start.a1; out.b1 = 1.0; out.b2 = 0.0; }
            else { out.b0 = start.a2; out.b1 = start.a1; out.b2 = 1.0; }
            out.a1 = start.a1; out.a2 = start.a2;
            return out;
        }

        // Everything is fitted in t = y / y0, with y = 1 - cos w and y0 its
        // value at the pole frequency: the terms of a section are then of the
        // order of one whether its corner is at 20 Hz or next to Nyquist.
        const double poleOmega = shape.first ? section.a0 / section.a1 : std::sqrt(section.a0 / section.a2);
        const double y0 = yOf(std::clamp(poleOmega / fs, 1e-9, kPi));
        Samples x;
        gather(section, shape, band, y0, x);

        const int nN = shape.numeratorTerms(), nD = shape.denominatorTerms();
        double p[5] = {};                                   // numerator terms, then denominator terms
        p[nN] = start.d1 * y0 / start.d0;
        if (nD > 1) p[nN + 1] = start.d2 * y0 * y0 / start.d0;

        // The numerator over the poles carried from the prototype.
        double normal[5][6];
        double cost = 0.0, gap = 0.0;
        if (nN > 0)
        {
            pass(shape, x, p, false, normal, cost, gap);
            if (!advance(shape, false, 1e-12, normal, p)) return bilinear(section, fs, section.poleFrequency());
            clampNumerator(shape, x, p);
        }

        // Most sections are on the curve already.
        double best[5] = {};
        std::copy(p, p + 5, best);
        double bestCost = 0.0;
        pass(shape, x, p, true, nullptr, bestCost, gap);

        // The rest: numerator and denominator together. Each pass measures
        // the section it has and prepares the step to the next.
        for (int i = 0; i < kFitSteps && gap > kClose; ++i)
        {
            pass(shape, x, p, true, normal, cost, gap);
            if (i > 0)
            {
                if (!(cost < bestCost)) break;
                const bool settled = bestCost - cost < kSettled * bestCost;
                std::copy(p, p + 5, best);
                bestCost = cost;
                if (settled || !(gap > kClose)) break;
            }
            if (!advance(shape, true, 1e-6, normal, p)) break;
            if (!positive(p[nN], nD > 1 ? p[nN + 1] : 0.0, x.tMax)) break;
            clampNumerator(shape, x, p);
            if (i + 1 == kFitSteps)
            {
                pass(shape, x, p, true, nullptr, cost, gap);
                if (cost < bestCost) std::copy(p, p + 5, best);
            }
        }

        BiquadCoeffs out;
        if (!assemble(shape, x, best, y0, out)) return bilinear(section, fs, section.poleFrequency());
        return out;
    }

    /** @brief digital() for one design: builds the Band of the rate on the spot. */
    [[nodiscard]] static BiquadCoeffs digital(const AnalogSection& section, double sampleRate) noexcept
    {
        return digital(section, Band(sampleRate));
    }

    /**
     * @brief Every section of a cascade through digital().
     *
     * @param cascade  Analog cascade.
     * @param band     The frequencies of the fit.
     * @param out      Receives the sections.
     * @param capacity Room in out.
     * @return Number of sections written.
     */
    static int digital(const AnalogCascade& cascade, const Band& band, BiquadCoeffs* out, int capacity) noexcept
    {
        const int n = std::min(cascade.count, capacity);
        for (int i = 0; i < n; ++i)
            out[i] = digital(cascade.sections[static_cast<size_t>(i)], band);
        return n;
    }

    /** @brief Every section of a cascade through digital(), for one design. */
    static int digital(const AnalogCascade& cascade, double sampleRate, BiquadCoeffs* out, int capacity) noexcept
    {
        return digital(cascade, Band(sampleRate), out, capacity);
    }

    /** @brief Magnitude in decibels of n digital sections in series at a frequency. */
    [[nodiscard]] static double magnitudeDb(const BiquadCoeffs* sections, int n, double hz, double sampleRate) noexcept
    {
        double db = 0.0;
        for (int i = 0; i < n; ++i)
            db += 20.0 * std::log10(std::max(sections[i].getMagnitude(hz, sampleRate), 1e-15));
        return db;
    }

    // -- Cascades ---------------------------------------------------------------
    //
    // Each returns an analog prototype; digital() turns it into biquads.

    static constexpr double kButterworthQ = 0.7071067811865476;   ///< The Q that leaves a cut maximally flat.
    static constexpr double kMaxCutSlope = 96.0;                  ///< Steepest cut, in decibels per octave.
    static constexpr int kMaxOrder = AnalogCascade::kMaxSections; ///< Highest order of bells, shelves and band filters.

    /**
     * @brief High cut (low-pass) of any slope from 0 to 96 dB per octave.
     *
     * The slope is a real number. Every 6 dB per octave is one unit of order,
     * and whole orders are Butterworth filters. Between them the poles sit on
     * the same semicircle at the angles of a real order, and one section with
     * a finite zero (a pair, or one real pole) keeps the corner at -3 dB and
     * the cascade on the curve 1 / (1 + w^(2 order)) down to -36 dB. Under 6 dB per octave the filter is the fraction
     * of a pole, (1 + s)^(-order), built from six real poles and zeros two
     * octaves apart: it leaves 3 dB times the order at the corner and nothing
     * at a slope of zero.
     *
     * @param hz               Corner frequency.
     * @param slopeDbPerOctave Slope, 0 to 96 (0 gives an empty cascade).
     * @param q                Resonance of the sharpest pole pair, as the Q of
     *                         a 12 dB per octave filter (0.7071 is Butterworth
     *                         at every slope). It has no effect up to 6 dB per
     *                         octave and comes in between 6 and 12.
     */
    [[nodiscard]] static AnalogCascade highCut(double hz, double slopeDbPerOctave, double q = kButterworthQ) noexcept
    {
        AnalogCascade out = cutPrototype(slopeDbPerOctave, q);
        const double omega = omegaOf(hz);
        for (int i = 0; i < out.count; ++i) at(out, i) = scaled(at(out, i), omega);
        return out;
    }

    /** @brief Low cut (high-pass) of any slope from 0 to 96 dB per octave: highCut() mirrored about the corner. */
    [[nodiscard]] static AnalogCascade lowCut(double hz, double slopeDbPerOctave, double q = kButterworthQ) noexcept
    {
        AnalogCascade out = cutPrototype(slopeDbPerOctave, q);
        const double omega = omegaOf(hz);
        for (int i = 0; i < out.count; ++i) at(out, i) = mirrored(at(out, i), omega);
        return out;
    }

    /**
     * @brief Low shelf of any order: gainDb below the corner, half of it (in
     *        decibels) at the corner, 6 dB per octave of transition per order.
     *
     * A Butterworth shelf (S. J. Orfanidis, "High-Order Digital Parametric
     * Equalizer Design", J. Audio Eng. Soc., 2005): it never overshoots, and
     * a cut is the exact inverse of the boost of the same size.
     *
     * From order 2 up, q is the Q of the sharpest pair of poles and of
     * zeros (0.7071: the Butterworth shelf). Above it the response
     * rises over the plateau before the corner and dips after it;
     * the plateau, the half gain at the corner and the inverse cut
     * stay as they are.
     */
    [[nodiscard]] static AnalogCascade lowShelf(double hz, double gainDb, int order, double q = kButterworthQ) noexcept
    {
        AnalogCascade out = shelfPrototype(gainDb, order, q);
        const double omega = omegaOf(hz);
        for (int i = 0; i < out.count; ++i) at(out, i) = scaled(at(out, i), omega);
        return out;
    }

    /** @brief High shelf of any order: lowShelf() mirrored about the corner. */
    [[nodiscard]] static AnalogCascade highShelf(double hz, double gainDb, int order, double q = kButterworthQ) noexcept
    {
        AnalogCascade out = shelfPrototype(gainDb, order, q);
        const double omega = omegaOf(hz);
        for (int i = 0; i < out.count; ++i) at(out, i) = mirrored(at(out, i), omega);
        return out;
    }

    /**
     * @brief Bell of any order: the higher the order, the flatter the top and
     *        the steeper the sides (6 dB per octave each per order).
     *
     * Order 1 is AnalogSection::peak(). The gain is gainDb at the centre and
     * half of it (in decibels) at the two edges hz / q apart, whatever the
     * order; a cut is the exact inverse of the boost. One biquad per order.
     */
    [[nodiscard]] static AnalogCascade bell(double hz, double q, double gainDb, int order) noexcept
    {
        AnalogCascade out;
        const AnalogCascade proto = shelfPrototype(gainDb, order);
        const double w0 = omegaOf(hz), width = w0 / std::clamp(q, 1e-3, 1e3);
        for (int i = 0; i < proto.count; ++i)
        {
            const AnalogSection& s = proto.sections[static_cast<size_t>(i)];
            if (!s.isSecondOrder())
            {
                out.add(AnalogSection::make(1.0, s.b0 * width, w0 * w0, 1.0, s.a0 * width, w0 * w0));
                continue;
            }
            const BandPair num = bandPair(s.b1, s.b0, width, w0), den = bandPair(s.a1, s.a0, width, w0);
            out.add(AnalogSection::make(1.0, num.upper1, num.upper0, 1.0, den.upper1, den.upper0));
            out.add(AnalogSection::make(1.0, num.lower1, num.lower0, 1.0, den.lower1, den.lower0));
        }
        return out;
    }

    /**
     * @brief Butterworth band-pass: 0 dB at the centre, -3 dB at the two
     *        edges hz / q apart, 6 dB per octave on each side per order. One
     *        biquad per order.
     */
    [[nodiscard]] static AnalogCascade bandPass(double hz, double q, int order) noexcept
    {
        return bandFilter(hz, q, order, false);
    }

    /**
     * @brief Butterworth band-stop: a null at the centre, -3 dB at the two
     *        edges hz / q apart, and a floor that widens with the order. One
     *        biquad per order.
     */
    [[nodiscard]] static AnalogCascade bandStop(double hz, double q, int order) noexcept
    {
        return bandFilter(hz, q, order, true);
    }

    /**
     * @brief All-pass of any order: the magnitude stays at 1 and the phase
     *        turns 180 degrees per order around the frequency.
     *
     * The poles are those of the cut of the same order (q acts on the sharpest
     * pair, from order 2 up).
     */
    [[nodiscard]] static AnalogCascade allPass(double hz, double q, int order) noexcept
    {
        AnalogCascade out;
        const int n = std::clamp(order, 1, kMaxOrder);
        const double w = omegaOf(hz), w2 = w * w;
        for (int i = 0; i < n / 2; ++i)
        {
            double damping = 2.0 * std::sin(kPi * (2 * i + 1) / (2.0 * n));
            if (i == 0) damping *= kButterworthQ / std::clamp(q, 0.1, 100.0);
            const double a1 = damping * w;
            out.add(AnalogSection::make(1.0, -a1, w2, 1.0, a1, w2));
        }
        if (n % 2 == 1) out.add(AnalogSection::make(0.0, -1.0, w, 0.0, 1.0, w));
        return out;
    }

    /**
     * @brief Tilt: the same slope across the whole band, 0 dB at the pivot.
     *
     * Eight real zeros and poles two octaves apart, from 2.5 Hz to 160 kHz,
     * in four sections. The slope goes from -6 to +6 dB per octave (positive
     * tilts towards the treble); at the two ends it is one zero and one pole.
     *
     * @param pivotHz          Frequency that keeps its level.
     * @param slopeDbPerOctave Slope, -6 to +6 (0 gives an empty cascade).
     */
    [[nodiscard]] static AnalogCascade tilt(double pivotHz, double slopeDbPerOctave) noexcept
    {
        AnalogCascade out;
        if (!std::isfinite(slopeDbPerOctave)) return out;
        const double order = std::clamp(slopeDbPerOctave / 6.020599913279624, -1.0, 1.0);
        if (std::abs(order) < 1e-9) return out;
        const double toZero = std::pow(kLadderStep, 0.5 * (1.0 - order));
        const double toPole = std::pow(kLadderStep, 0.5 * (1.0 + order));
        double cell = 2.0 * kPi * 632.4555320336759 / (kLadderStep * kLadderStep * kLadderStep * kLadderStep);
        for (int i = 0; i < 4; ++i)
        {
            const double z1 = cell * toZero, p1 = cell * toPole, z2 = z1 * kLadderStep, p2 = p1 * kLadderStep;
            out.add(AnalogSection::make(1.0 / (z1 * z2), 1.0 / z1 + 1.0 / z2, 1.0,
                                        1.0 / (p1 * p2), 1.0 / p1 + 1.0 / p2, 1.0));
            cell *= kLadderStep * kLadderStep;
        }
        const double level = out.magnitudeSquared(omegaOf(pivotHz));
        if (level > 0.0 && std::isfinite(level))
        {
            const double gain = 1.0 / std::sqrt(level);
            AnalogSection& first = at(out, 0);
            first.b2 *= gain; first.b1 *= gain; first.b0 *= gain;
        }
        return out;
    }

    /**
     * @brief The slope a tilt can have at a pivot without lifting the end
     *        that rises (20 kHz for a positive slope, 20 Hz for a negative
     *        one) by more than a ceiling: the slope asked for, or the one that
     *        lands that end on the ceiling. The end that falls is not limited.
     *
     * @param pivotHz          Frequency that keeps its level.
     * @param slopeDbPerOctave Slope asked for, in dB per octave.
     * @param ceilingDb        Most the rising end may be lifted, in dB.
     */
    [[nodiscard]] static double tiltSlopeWithin(double pivotHz, double slopeDbPerOctave, double ceilingDb) noexcept
    {
        if (!std::isfinite(slopeDbPerOctave) || !std::isfinite(pivotHz) || !(ceilingDb > 0.0)) return slopeDbPerOctave;
        const double pivot = std::clamp(pivotHz, 20.0, 20000.0);
        const double octaves = slopeDbPerOctave > 0.0 ? std::log2(20000.0 / pivot) : std::log2(pivot / 20.0);
        const double rise = std::abs(slopeDbPerOctave) * octaves;
        return rise > ceilingDb ? slopeDbPerOctave * (ceilingDb / rise) : slopeDbPerOctave;
    }

    /**
     * @brief Elliptic high cut: flat to the corner within the ripple, then a
     *        wall down to the stop level.
     *
     * With the defaults (order 12, 0.05 dB, 100 dB) the stop band starts a
     * fifth of an octave above the corner.
     *
     * @param hz       End of the pass band.
     * @param order    1 to 16.
     * @param rippleDb Ripple of the pass band, 0.001 to 3 dB.
     * @param stopDb   Attenuation of the stop band, 20 to 140 dB.
     */
    [[nodiscard]] static AnalogCascade ellipticHighCut(double hz, int order = 12, double rippleDb = 0.05,
                                                       double stopDb = 100.0) noexcept
    {
        AnalogCascade out = ellipticPrototype(order, rippleDb, stopDb);
        const double omega = omegaOf(hz);
        for (int i = 0; i < out.count; ++i) at(out, i) = scaled(at(out, i), omega);
        return out;
    }

    /** @brief Elliptic low cut: ellipticHighCut() mirrored about the corner. */
    [[nodiscard]] static AnalogCascade ellipticLowCut(double hz, int order = 12, double rippleDb = 0.05,
                                                      double stopDb = 100.0) noexcept
    {
        AnalogCascade out = ellipticPrototype(order, rippleDb, stopDb);
        const double omega = omegaOf(hz);
        for (int i = 0; i < out.count; ++i) at(out, i) = mirrored(at(out, i), omega);
        return out;
    }

    /**
     * @brief Where the stop band of an elliptic cut starts, as a ratio to its
     *        corner (above 1; the same for the high cut and, inverted, for the
     *        low cut).
     */
    [[nodiscard]] static double ellipticStopRatio(int order = 12, double rippleDb = 0.05, double stopDb = 100.0) noexcept
    {
        double ripple = 0.0, stop = 0.0;
        return 1.0 / ellipticModulus(order, rippleDb, stopDb, ripple, stop);
    }

private:
    static constexpr double kPi = std::numbers::pi;
    static constexpr int kClusterPoints = 7;                 ///< Frequencies around the poles, and around the zeros.
    static constexpr int kMaxPoints = Band::kPoints + 2 * kClusterPoints;
    static constexpr double kFloor = 1e-8;                   ///< Powers this far under the largest count for little.
    static constexpr double kClose = 5.3e-6;                 ///< A hundredth of a decibel, as squared relative power.
    static constexpr double kUltrasonicHz = 3000.0;          ///< Above 20 kHz an error counts half this far up.
    static constexpr double kSettled = 0.05;                 ///< A step that gains less than this share is the last.

    /// What the numerator of a prototype is made of.
    struct Shape
    {
        int kind = 0;             ///< 0 free; 1 one zero at DC; 2 two zeros at DC; 3 zeros on the circle.
        bool first = false;       ///< First-order denominator.
        bool allPass = false;     ///< Numerator mirrors the denominator: nothing to fit.
        double cosNotch = 0.0;    ///< cos of the notch frequency (kind 3).
        double yNotch = 0.0;      ///< 1 - cos of the notch frequency (kind 3).

        /// Terms of the numerator the fit moves. Its value at DC is not one of them: the gain at DC is held.
        [[nodiscard]] int numeratorTerms() const noexcept
        {
            if (first) return 1;
            return kind == 3 ? 0 : (kind == 2 ? 1 : 2);
        }
        [[nodiscard]] int denominatorTerms() const noexcept { return first ? 1 : 2; }
    };

    /// The squared magnitude of a denominator as d0 + d1 y + d2 y^2, and its coefficients.
    struct Denominator
    {
        double d0 = 1.0, d1 = 0.0, d2 = 0.0;
        double a1 = 0.0, a2 = 0.0;
    };

    /// Where the fit looks: t of each frequency and the analog power there.
    struct Samples
    {
        int n = 0;
        double t[kMaxPoints] {};
        double m[kMaxPoints] {};
        double weight[kMaxPoints] {};
        double tMax = 2.0;        ///< t at the Nyquist frequency.
        double tz = 0.0;          ///< t of the notch (kind 3).
        double dc = 0.0;          ///< Analog power at DC, which the numerator keeps (kinds 0 and 3).
        double floor = 0.0;       ///< Power under which a frequency counts for little.
    };

    /**
     * @brief How much the error at an angular frequency counts: all of it up
     *        to 20 kHz, less and less above (what a design does up there must
     *        stay sensible, but not at the price of the band that is heard).
     */
    [[nodiscard]] static double weightOf(double omega) noexcept
    {
        const double over = std::max(0.0, omega * (0.5 / kPi) - 20000.0) * (1.0 / kUltrasonicHz);
        return 1.0 / (1.0 + over * over);
    }

    /** @brief y = 1 - cos w: the squared magnitude of a biquad polynomial is a quadratic in y. */
    [[nodiscard]] static double yOf(double w) noexcept
    {
        const double s = std::sin(0.5 * w);
        return 2.0 * s * s;
    }

    /** @brief sin x for x between 0 and pi / 2, to eleven digits (the cost of a design is mostly sines). */
    [[nodiscard]] static double sine(double x) noexcept
    {
        constexpr double k3 = 1.0 / 6.0, k5 = 1.0 / 20.0, k7 = 1.0 / 42.0, k9 = 1.0 / 72.0;
        constexpr double k11 = 1.0 / 110.0, k13 = 1.0 / 156.0, k15 = 1.0 / 210.0;
        const double x2 = x * x;
        return x * (1.0 - x2 * k3 * (1.0 - x2 * k5 * (1.0 - x2 * k7 * (1.0 - x2 * k9
               * (1.0 - x2 * k11 * (1.0 - x2 * k13 * (1.0 - x2 * k15)))))));
    }

    [[nodiscard]] static bool finite(const AnalogSection& h) noexcept
    {
        return std::isfinite(h.b2) && std::isfinite(h.b1) && std::isfinite(h.b0)
            && std::isfinite(h.a2) && std::isfinite(h.a1) && std::isfinite(h.a0);
    }

    [[nodiscard]] static bool usable(const AnalogSection& h) noexcept
    {
        if (h.b2 == 0.0 && h.b1 == 0.0 && h.b0 == 0.0) return false;
        if (h.isSecondOrder()) return h.a1 / h.a2 > 0.0 && h.a0 / h.a2 > 0.0;
        return h.b2 == 0.0 && h.a1 != 0.0 && h.a0 / h.a1 > 0.0;
    }

    [[nodiscard]] static Shape shapeOf(const AnalogSection& h, double fs) noexcept
    {
        Shape s;
        s.first = !h.isSecondOrder();
        if (s.first)
        {
            if (h.b0 == 0.0) s.kind = 1;
            s.allPass = h.b1 != 0.0 && h.b1 == -h.a1 && h.b0 == h.a0;
            return s;
        }
        if (h.b0 == 0.0) s.kind = (h.b1 == 0.0) ? 2 : 1;
        else if (h.b1 == 0.0 && h.b2 != 0.0 && h.b0 / h.b2 > 0.0)
        {
            const double w = std::sqrt(h.b0 / h.b2) / fs;
            if (w < 0.98 * kPi)
            {
                s.kind = 3;
                s.yNotch = yOf(w);
                s.cosNotch = 1.0 - s.yNotch;
            }
        }
        s.allPass = h.b1 != 0.0 && h.b2 == h.a2 && h.b0 == h.a0 && h.b1 == -h.a1;
        return s;
    }

    /**
     * @brief The denominator with the analog poles carried over by z = e^(s / fs).
     *
     * A pole pair whose frequency nears or passes the Nyquist frequency would
     * alias; its angle is bent smoothly so that it stays under 0.99 pi.
     */
    [[nodiscard]] static Denominator invariantDenominator(const AnalogSection& h, double fs) noexcept
    {
        Denominator d;
        if (!h.isSecondOrder())
        {
            const double x = -(h.a0 / h.a1) / fs;
            const double p = std::exp(x), gap = -std::expm1(x);
            d.d0 = gap * gap; d.d1 = 2.0 * p; d.d2 = 0.0;
            d.a1 = -p; d.a2 = 0.0;
            return d;
        }
        const double sigma = -0.5 * h.a1 / h.a2, det = sigma * sigma - h.a0 / h.a2;
        double atNyquist = 0.0;
        if (det < 0.0)
        {
            const double r = std::exp(sigma / fs), gap = -std::expm1(sigma / fs);
            constexpr double kFree = 0.95 * kPi, kRoom = 0.04 * kPi;
            double theta = std::sqrt(-det) / fs;
            if (theta > kFree) theta = kFree + kRoom * (1.0 - std::exp(-(theta - kFree) / kRoom));
            const double s = std::sin(0.5 * theta), bend = 4.0 * r * s * s;
            const double atDc = gap * gap + bend, top = (1.0 + r) * (1.0 + r) - bend;
            d.d0 = atDc * atDc;
            atNyquist = top * top;
            d.d2 = 4.0 * r * r;
            d.a1 = -2.0 * r * std::cos(theta); d.a2 = r * r;
        }
        else
        {
            const double root = std::sqrt(det);
            const double x1 = (sigma + root) / fs, x2 = (sigma - root) / fs;
            const double p1 = std::exp(x1), p2 = std::exp(x2);
            const double gap = std::expm1(x1) * std::expm1(x2), top = (1.0 + p1) * (1.0 + p2);
            d.d0 = gap * gap;
            atNyquist = top * top;
            d.d2 = 4.0 * p1 * p2;
            d.a1 = -(p1 + p2); d.a2 = p1 * p2;
        }
        d.d1 = 0.5 * (atNyquist - d.d0) - 2.0 * d.d2;
        return d;
    }

    /** @brief Analog power at a squared angular frequency. */
    [[nodiscard]] static double power(const AnalogSection& h, double omega2) noexcept
    {
        const double nr = h.b0 - h.b2 * omega2, dr = h.a0 - h.a2 * omega2;
        return (nr * nr + h.b1 * h.b1 * omega2) / (dr * dr + h.a1 * h.a1 * omega2);
    }

    /**
     * @brief Seven frequencies around a resonance, at its centre and at half,
     *        once and twice its half-power width to each side (kept inside
     *        the band), so that the fit sees a peak however narrow it is.
     */
    static void cluster(const AnalogSection& h, const Band& band, double centre, double q, double y0, Samples& x) noexcept
    {
        const double half = 0.5 / std::max(q, 0.5);
        const double r = std::sqrt(1.0 + half * half) + half, s = std::sqrt(r);
        const double factors[kClusterPoints] = { 1.0 / (r * r), 1.0 / r, 1.0 / s, 1.0, s, r, r * r };
        const double low = 2.0 * kPi * band.lowHz_, high = 2.0 * kPi * band.highHz_;
        const double halfPeriod = 0.5 / band.fs_, scale = 1.0 / y0;
        for (const double ratio : factors)
        {
            const double omega = std::clamp(centre * ratio, low, high);
            const double sn = sine(omega * halfPeriod);
            x.t[x.n] = 2.0 * sn * sn * scale;
            x.m[x.n] = power(h, omega * omega);
            x.weight[x.n] = weightOf(omega);
            ++x.n;
        }
    }

    static void gather(const AnalogSection& h, const Shape& shape, const Band& band, double y0, Samples& x) noexcept
    {
        const double scale = 1.0 / y0;
        x.n = Band::kPoints;
        for (int i = 0; i < Band::kPoints; ++i)
        {
            x.t[i] = band.y_[i] * scale;
            x.m[i] = power(h, band.omega2_[i]);
            x.weight[i] = band.weight_[i];
        }
        if (shape.first)
        {
            cluster(h, band, h.a0 / h.a1, 0.5, y0, x);
            if (h.b1 != 0.0 && h.b0 != 0.0) cluster(h, band, std::abs(h.b0 / h.b1), 0.5, y0, x);
        }
        else
        {
            cluster(h, band, std::sqrt(h.a0 / h.a2), std::sqrt(h.a0 * h.a2) / std::abs(h.a1), y0, x);
            if (shape.kind == 0)
            {
                if (h.b2 != 0.0 && h.b0 / h.b2 > 0.0 && h.b1 != 0.0)
                    cluster(h, band, std::sqrt(h.b0 / h.b2), std::sqrt(h.b0 * h.b2) / std::abs(h.b1), y0, x);
                else if (h.b2 == 0.0 && h.b1 != 0.0)
                    cluster(h, band, std::abs(h.b0 / h.b1), 0.5, y0, x);
            }
        }
        double largest = 0.0;
        for (int i = 0; i < x.n; ++i) largest = std::max(largest, x.m[i]);
        x.floor = kFloor * largest;
        x.tMax = 2.0 * scale;
        x.tz = shape.yNotch * scale;
        x.dc = (shape.kind == 0 || shape.kind == 3) ? (h.b0 / h.a0) * (h.b0 / h.a0) : 0.0;
    }

    /** @brief The terms of the numerator the fit moves, at t. */
    static void numeratorAt(const Shape& s, double t, double* out) noexcept
    {
        if (s.first) out[0] = t;
        else if (s.kind == 2) out[0] = t * t;
        else if (s.kind != 3) { out[0] = t; out[1] = t * t; }
    }

    /** @brief The part of the numerator the fit does not move, at t: what holds the gain at DC. */
    [[nodiscard]] static double heldAt(const Shape& s, const Samples& x, double t) noexcept
    {
        if (s.kind == 0) return x.dc;
        if (s.kind == 3) return x.dc * (t - x.tz) * (t - x.tz) / (x.tz * x.tz);
        return 0.0;
    }

    /** @brief Solves n equations in place (the last column is the right side). */
    [[nodiscard]] static bool solve(int n, double m[5][6]) noexcept
    {
        for (int c = 0; c < n; ++c)
        {
            int pivot = c;
            for (int r = c + 1; r < n; ++r)
                if (std::abs(m[r][c]) > std::abs(m[pivot][c])) pivot = r;
            if (!(std::abs(m[pivot][c]) > 1e-300)) return false;
            if (pivot != c)
                for (int k = 0; k <= n; ++k) std::swap(m[c][k], m[pivot][k]);
            for (int r = 0; r < n; ++r)
            {
                if (r == c) continue;
                const double f = m[r][c] / m[c][c];
                for (int k = c; k <= n; ++k) m[r][k] -= f * m[c][k];
            }
        }
        for (int r = 0; r < n; ++r) m[r][n] /= m[r][r];
        return true;
    }

    /**
     * @brief One pass over the frequencies with the section of terms p: how
     *        far it is from the analog power and, when asked, the equations of
     *        the least-squares step from it (the numerator alone, or numerator
     *        and denominator together).
     *
     * With N and D the squared magnitudes of numerator and denominator (both
     * polynomials in t), the equation N - M D = 0 at each frequency is linear
     * in their terms. Each is weighted by 1 / (M D) with the D of the pass,
     * so that what the squares add up is the relative error of the power
     * (C. K. Sanathanan and J. Koerner, IEEE Trans. Automatic Control, 1963).
     *
     * The distance is the sum over the frequencies of (r - 1)^2 / r, with r
     * the ratio of the two powers: the same going over as going under.
     */
    static void pass(const Shape& s, const Samples& x, const double* p, bool together,
                     double (*normal)[6], double& cost, double& worst) noexcept
    {
        const int nN = s.numeratorTerms(), nD = s.denominatorTerms();
        const int n = together ? nN + nD : nN;
        if (normal != nullptr)
            for (int a = 0; a < 5; ++a)
                for (int b = 0; b < 6; ++b) normal[a][b] = 0.0;
        cost = 0.0;
        worst = 0.0;
        for (int i = 0; i < x.n; ++i)
        {
            const double t = x.t[i], m = x.m[i], t2 = t * t;
            double row[5] = {};
            numeratorAt(s, t, row);
            double value = heldAt(s, x, t);
            for (int j = 0; j < nN; ++j) value += p[j] * row[j];
            const double was = 1.0 + p[nN] * t + (nD > 1 ? p[nN + 1] * t2 : 0.0);
            const double want = (m + x.floor) * was, have = value + x.floor * was;
            const double u = x.weight[i] / want, off = (have - want) * u;
            const double apart = off * off * want / have;
            cost += apart;
            worst = std::max(worst, apart);
            if (normal == nullptr) continue;

            for (int j = 0; j < nN; ++j) row[j] *= u;
            if (together)
            {
                row[nN] = -u * m * t;
                if (nD > 1) row[nN + 1] = -u * m * t2;
            }
            for (int a = 0; a < n; ++a)
            {
                const double ra = row[a];
                for (int b = a; b < n; ++b) normal[a][b] += ra * row[b];
                normal[a][n] -= ra * off;
            }
        }
    }

    /**
     * @brief Solves the equations of a pass and moves p by the step.
     *
     * Every unknown is put on the same scale first, and a touch of damping
     * keeps a direction the frequencies say nothing about where it was.
     */
    [[nodiscard]] static bool advance(const Shape& s, bool together, double damping,
                                      double (*normal)[6], double* p) noexcept
    {
        const int n = together ? s.numeratorTerms() + s.denominatorTerms() : s.numeratorTerms();
        if (n == 0) return true;
        double scale[5];
        for (int a = 0; a < n; ++a)
        {
            if (!(normal[a][a] > 0.0)) return false;
            scale[a] = 1.0 / std::sqrt(normal[a][a]);
        }
        double system[5][6];
        for (int a = 0; a < n; ++a)
        {
            for (int b = a; b < n; ++b) system[a][b] = system[b][a] = normal[a][b] * scale[a] * scale[b];
            system[a][a] = 1.0 + damping;
            system[a][n] = normal[a][n] * scale[a];
        }
        if (!solve(n, system)) return false;
        for (int a = 0; a < n; ++a)
        {
            const double move = system[a][n] * scale[a];
            if (!std::isfinite(move)) return false;
            p[a] += move;
        }
        return true;
    }

    /** @brief Whether 1 + d1 t + d2 t^2 stays clear of zero from DC to Nyquist (the poles stay inside the circle). */
    [[nodiscard]] static bool positive(double d1, double d2, double tMax) noexcept
    {
        constexpr double kMargin = 1e-9;
        const double end = 1.0 + tMax * (d1 + tMax * d2);
        if (!(end > kMargin * (1.0 + tMax * (std::abs(d1) + tMax * std::abs(d2))))) return false;
        if (d2 > 0.0)
        {
            const double tv = -0.5 * d1 / d2;
            if (tv > 0.0 && tv < tMax)
            {
                const double low = 1.0 + 0.5 * d1 * tv;
                if (!(low > kMargin * (1.0 + tv * (std::abs(d1) + tv * std::abs(d2))))) return false;
            }
        }
        return true;
    }

    /**
     * @brief Keeps the numerator a squared magnitude: where the fit would
     *        cross zero between DC and Nyquist, its zeros stop at Nyquist or
     *        on the circle. The value at DC is not touched.
     */
    static void clampNumerator(const Shape& s, const Samples& x, double* p) noexcept
    {
        if (s.kind == 0 && s.first)
        {
            if (x.dc + p[0] * x.tMax < 0.0) p[0] = -x.dc / x.tMax;
        }
        else if (s.kind == 0)
        {
            if (x.dc + x.tMax * (p[0] + x.tMax * p[1]) < 0.0) p[1] = -(x.dc + p[0] * x.tMax) / (x.tMax * x.tMax);
            if (p[1] > 0.0 && p[0] < 0.0 && -0.5 * p[0] / p[1] < x.tMax && p[0] * p[0] > 4.0 * x.dc * p[1])
                p[0] = -2.0 * std::sqrt(x.dc * p[1]);
        }
        else if (s.kind == 1 && !s.first)
        {
            p[0] = std::max(p[0], 0.0);
            if (p[0] + p[1] * x.tMax < 0.0) p[1] = -p[0] / x.tMax;
        }
        else if (s.kind != 3)
        {
            p[0] = std::max(p[0], 0.0);
        }
    }

    /** @brief The minimum-phase c0 + c1 z^-1 + c2 z^-2 whose squared magnitude is k0 + k1 y + k2 y^2. */
    [[nodiscard]] static bool factor(double k0, double k1, double k2, double& c0, double& c1, double& c2) noexcept
    {
        // A polynomial that touches zero at DC or at Nyquist may come out a
        // rounding error under it.
        const double slack = -1e-9 * (std::abs(k0) + 2.0 * std::abs(k1) + 4.0 * std::abs(k2));
        double atDc = k0, atNyquist = k0 + 2.0 * k1 + 4.0 * k2;
        if (!(atDc >= slack) || !(atNyquist >= slack)) return false;
        atDc = std::max(atDc, 0.0);
        atNyquist = std::max(atNyquist, 0.0);
        const double p = std::sqrt(atDc), m = std::sqrt(atNyquist);
        const double sum = 0.5 * (p + m);
        c1 = 0.5 * (p - m);
        const double disc = sum * sum - k2;
        if (disc < -1e-9 * (sum * sum + std::abs(k2))) return false;
        c0 = 0.5 * (sum + std::sqrt(std::max(0.0, disc)));
        if (!(c0 > 0.0)) return false;
        c2 = 0.25 * k2 / c0;
        return true;
    }

    /** @brief From the fitted terms to the coefficients of the section. */
    [[nodiscard]] static bool assemble(const Shape& s, const Samples& x, const double* p, double y0, BiquadCoeffs& out) noexcept
    {
        const int nN = s.numeratorTerms();
        const double scale = 1.0 / y0, scale2 = scale * scale;
        double c0 = 1.0, c1 = 0.0, c2 = 0.0;
        if (!factor(1.0, p[nN] * scale, s.first ? 0.0 : p[nN + 1] * scale2, c0, c1, c2)) return false;

        double e0 = 0.0, e1 = 0.0, e2 = 0.0;
        if (s.kind == 0)
        {
            if (!factor(x.dc, p[0] * scale, s.first ? 0.0 : p[1] * scale2, e0, e1, e2)) return false;
        }
        else if (s.kind == 1 && !s.first)
        {
            // y (n1 + n2 y), and y is half the squared magnitude of 1 - z^-1.
            double f0 = 0.0, f1 = 0.0, unused = 0.0;
            if (!factor(0.5 * p[0] * scale, 0.5 * p[1] * scale2, 0.0, f0, f1, unused)) return false;
            e0 = f0; e1 = f1 - f0; e2 = -f1;
        }
        else if (s.kind == 1)
        {
            const double g = std::sqrt(0.5 * std::max(p[0], 0.0) * scale);
            e0 = g; e1 = -g;
        }
        else if (s.kind == 2)
        {
            // Two zeros at DC: y^2 is a quarter of the squared magnitude of (1 - z^-1)^2.
            const double g = 0.5 * std::sqrt(std::max(p[0], 0.0)) * scale;
            e0 = g; e1 = -2.0 * g; e2 = g;
        }
        else
        {
            // Zeros on the circle: (y - yz)^2 is a quarter of the squared magnitude
            // of 1 - 2 cos(wz) z^-1 + z^-2, and the gain is the one of DC.
            const double g = 0.5 * std::sqrt(x.dc) / s.yNotch;
            e0 = g; e1 = -2.0 * g * s.cosNotch; e2 = g;
        }

        const double inverse = 1.0 / c0;
        out.b0 = e0 * inverse; out.b1 = e1 * inverse; out.b2 = e2 * inverse;
        out.a1 = c1 * inverse; out.a2 = c2 * inverse;
        return std::isfinite(out.b0) && std::isfinite(out.b1) && std::isfinite(out.b2)
            && std::isfinite(out.a1) && std::isfinite(out.a2);
    }

    // -- Cascades: the prototypes ---------------------------------------------------

    static constexpr double kLadderStep = 4.0;     ///< Ratio between the cells of a ladder of real poles and zeros (two octaves).
    static constexpr double kDeep = 3981.0717055349733;   ///< 36 dB as a power ratio: where a cut of real order meets its ideal curve.
    static constexpr double kMiddle = 63.09573444801933;   ///< 18 dB as a power ratio: where a pair meets it as well.

    [[nodiscard]] static AnalogSection& at(AnalogCascade& cascade, int i) noexcept
    {
        return cascade.sections[static_cast<size_t>(i)];
    }

    [[nodiscard]] static double omegaOf(double hz) noexcept
    {
        return 2.0 * kPi * (std::isfinite(hz) ? std::max(hz, 1e-3) : 1000.0);
    }

    /** @brief A section written for a corner of 1 rad/s, moved to omega. */
    [[nodiscard]] static AnalogSection scaled(const AnalogSection& n, double omega) noexcept
    {
        if (!n.isSecondOrder())
            return AnalogSection::make(0.0, n.b1, n.b0 * omega, 0.0, n.a1, n.a0 * omega);
        return AnalogSection::make(n.b2, n.b1 * omega, n.b0 * omega * omega, n.a2, n.a1 * omega, n.a0 * omega * omega);
    }

    /** @brief A section written for a corner of 1 rad/s, mirrored (s to omega / s) and moved to omega. */
    [[nodiscard]] static AnalogSection mirrored(const AnalogSection& n, double omega) noexcept
    {
        if (!n.isSecondOrder())
            return AnalogSection::make(0.0, n.b0, n.b1 * omega, 0.0, n.a0, n.a1 * omega);
        return AnalogSection::make(n.b0, n.b1 * omega, n.b2 * omega * omega, n.a0, n.a1 * omega, n.a2 * omega * omega);
    }

    /** @brief The fraction of a pole, (1 + s)^(-order) for an order under 1: three sections of real poles and zeros. */
    static void fractionalPole(double order, AnalogCascade& out) noexcept
    {
        const double gap = std::pow(kLadderStep, order);                 // from each pole to its zero
        const double leaving = 1.0 - order * order * order * order;      // the last zero goes to infinity as the order nears 1
        double pole = std::pow(kLadderStep, 0.5 * (1.0 - order));
        for (int i = 0; i < 3; ++i)
        {
            const double p1 = pole, p2 = pole * kLadderStep;
            const double iz1 = 1.0 / (p1 * gap), iz2 = (i == 2 ? leaving : 1.0) / (p2 * gap);
            out.add(AnalogSection::make(iz1 * iz2, iz1 + iz2, 1.0, 1.0 / (p1 * p2), 1.0 / p1 + 1.0 / p2, 1.0));
            pole = p2 * kLadderStep;
        }
    }

    /** @brief The low-pass of a real order with its corner (-3 dB) at 1 rad/s. See highCut(). */
    [[nodiscard]] static AnalogCascade cutPrototype(double slopeDbPerOctave, double q) noexcept
    {
        AnalogCascade out;
        if (!std::isfinite(slopeDbPerOctave)) return out;
        const double order = std::clamp(slopeDbPerOctave, 0.0, kMaxCutSlope) / 6.0;
        if (order < 1e-6) return out;
        if (order < 1.0)
        {
            fractionalPole(order, out);
            return out;
        }

        // Pole pairs on the unit semicircle at the angles of a real order.
        // The whole part of the order says what completes them: an odd one,
        // the last pair (two real poles when the order is exactly odd) with a
        // zero; an even one, a real pole with a zero.
        const int whole = static_cast<int>(std::floor(order));
        const int pairs = static_cast<int>(std::floor(order + 1.0)) / 2;
        const bool onPair = (whole % 2) == 1;
        const int plain = onPair ? pairs - 1 : pairs;          // pairs that stay as the angles put them
        double damping[AnalogCascade::kMaxSections] = {};
        for (int k = 0; k < pairs; ++k)
            damping[k] = 2.0 * std::sin(kPi * (2 * k + 1) / (2.0 * order));

        // The section that completes them is what keeps the cascade on the
        // curve 1 / (1 + w^(2 order)) between whole orders: at the corner
        // (-3 dB) and where that curve has fallen 36 dB and, when it is a
        // pair, 18 dB too. Written as powers the conditions are linear in
        // what the section has free (its zero and the terms of its poles),
        // so there is nothing to search. At a whole order the zero cancels a
        // pole or leaves, and what is left is the Butterworth filter.
        const double deep = std::pow(kDeep - 1.0, 1.0 / order);             // w^2 at the 36 dB point
        double atCorner = 1.0, atDeep = 1.0;                                // attenuation of the plain pairs
        for (int k = 0; k < plain; ++k)
        {
            atCorner *= damping[k] * damping[k];
            atDeep *= (1.0 - deep) * (1.0 - deep) + damping[k] * damping[k] * deep;
        }
        const double lift = std::pow(std::clamp(q, 0.1, 100.0) / kButterworthQ, std::clamp(order - 1.0, 0.0, 1.0));
        for (int k = 0; k < plain; ++k)
            out.add(AnalogSection::make(0.0, 0.0, 1.0, 1.0, damping[k] / (k == 0 ? lift : 1.0), 1.0));

        if (onPair)
        {
            // The pair 1 + alpha v + beta v^2 over 1 + z v, through three points.
            const double mid = std::pow(kMiddle - 1.0, 1.0 / order);
            double atMid = 1.0;
            for (int k = 0; k < plain; ++k)
                atMid *= (1.0 - mid) * (1.0 - mid) + damping[k] * damping[k] * mid;
            const double points[3] = { 1.0, mid, deep };
            const double asked[3] = { 2.0 / atCorner, kMiddle / atMid, kDeep / atDeep };
            double system[5][6] = {};
            for (int i = 0; i < 3; ++i)
            {
                system[i][0] = points[i];
                system[i][1] = points[i] * points[i];
                system[i][2] = -asked[i] * points[i];
                system[i][3] = asked[i] - 1.0;
            }
            double alpha = 0.0, beta = 0.0, zero2 = -1.0;
            if (solve(3, system))
            {
                alpha = system[0][3]; beta = system[1][3]; zero2 = system[2][3];
            }
            const double natural2 = beta > 0.0 ? 1.0 / std::sqrt(beta) : 0.0;     // the frequency of the pair, squared
            const double damping2 = alpha * natural2 + 2.0;
            if (zero2 > -1e-9 && zero2 <= 1.0 + 1e-9 && natural2 > 0.0 && std::isfinite(natural2) && damping2 > 1e-6)
            {
                const double a1 = std::sqrt(damping2 * natural2) / (plain == 0 ? lift : 1.0);
                out.add(AnalogSection::make(0.0, std::sqrt(std::max(zero2, 0.0)) * natural2, natural2, 1.0, a1, natural2));
            }
            else
            {
                // Two points with the pair where its angle puts it.
                const double c = 2.0 / atCorner;
                const double slope = deep * (atDeep * c - kDeep);
                double z = 0.0;
                if (std::abs(slope) > 1e-12 * deep * kDeep)
                    z = std::clamp((kDeep - atDeep * ((1.0 - deep) * (1.0 - deep) + c * deep)) / slope, 0.0, 1.0);
                out.add(AnalogSection::make(0.0, std::sqrt(z), 1.0, 1.0,
                                            std::sqrt(c * (1.0 + z)) / (plain == 0 ? lift : 1.0), 1.0));
            }
        }
        else
        {
            // The real pole 1 + u v over 1 + z v, through two points.
            const double c = 2.0 / atCorner;
            const double slope = deep * (atDeep * c - kDeep);
            double z = 0.0;
            if (std::abs(slope) > 1e-12 * deep * kDeep)
                z = std::clamp((kDeep - atDeep * (1.0 + (c - 1.0) * deep)) / slope, 0.0, 1.0);
            const double u = c * (1.0 + z) - 1.0;
            if (u > 1e-12) out.add(AnalogSection::make(0.0, std::sqrt(z), 1.0, 0.0, std::sqrt(u), 1.0));
        }
        return out;
    }

    /**
     * @brief Butterworth low shelf with its corner at 1 rad/s: gainDb at DC,
     *        half of it (in decibels) at the corner, 0 dB far above.
     */
    [[nodiscard]] static AnalogCascade shelfPrototype(double gainDb, int order, double q = kButterworthQ) noexcept
    {
        AnalogCascade out;
        const int n = std::clamp(order, 1, kMaxOrder);
        const double db = std::isfinite(gainDb) ? std::clamp(gainDb, -120.0, 120.0) : 0.0;
        const double g = std::pow(10.0, db / (20.0 * n)), beta = std::pow(10.0, -db / (40.0 * n));
        // The Q acts on the sharpest pair, in its zeros as in its poles: the
        // gain of the pair at the corner does not depend on its damping.
        const double sharpest = kButterworthQ / (std::isfinite(q) ? std::clamp(q, 0.05, 100.0) : kButterworthQ);
        for (int i = 0; i < n / 2; ++i)
        {
            const double s = std::sin(kPi * (2 * i + 1) / (2.0 * n)) * (i == 0 ? sharpest : 1.0);
            out.add(AnalogSection::make(1.0, 2.0 * g * s * beta, g * g * beta * beta, 1.0, 2.0 * s * beta, beta * beta));
        }
        if (n % 2 == 1) out.add(AnalogSection::make(0.0, 1.0, g * beta, 0.0, 1.0, beta));
        return out;
    }

    /// The two quadratics a pair of low-pass roots turns into around a centre: s^2 + c1 s + c0 each.
    struct BandPair
    {
        double upper1 = 0.0, upper0 = 0.0;
        double lower1 = 0.0, lower0 = 0.0;
    };

    /**
     * @brief Low-pass to band-pass for the roots of S^2 + c1 S + c0: with
     *        S = (s^2 + w0^2) / (width s) each root becomes two, one above
     *        the centre and one below.
     */
    [[nodiscard]] static BandPair bandPair(double c1, double c0, double width, double w0) noexcept
    {
        const std::complex<double> root(-0.5 * c1, 0.5 * std::sqrt(std::max(0.0, 4.0 * c0 - c1 * c1)));
        const std::complex<double> sum = root * width;
        const std::complex<double> spread = std::sqrt(sum * sum - 4.0 * w0 * w0);
        std::complex<double> above = 0.5 * (sum + spread), below = 0.5 * (sum - spread);
        if (std::norm(above) < std::norm(below)) std::swap(above, below);
        BandPair pair;
        pair.upper1 = -2.0 * above.real(); pair.upper0 = std::norm(above);
        pair.lower1 = -2.0 * below.real(); pair.lower0 = std::norm(below);
        return pair;
    }

    [[nodiscard]] static AnalogCascade bandFilter(double hz, double q, int order, bool stop) noexcept
    {
        AnalogCascade out;
        const int n = std::clamp(order, 1, kMaxOrder);
        const double w0 = omegaOf(hz), w02 = w0 * w0, width = w0 / std::clamp(q, 1e-3, 1e3);
        const double b2 = stop ? 1.0 : 0.0, b1 = stop ? 0.0 : width, b0 = stop ? w02 : 0.0;
        for (int i = 0; i < n / 2; ++i)
        {
            const BandPair pair = bandPair(2.0 * std::sin(kPi * (2 * i + 1) / (2.0 * n)), 1.0, width, w0);
            out.add(AnalogSection::make(b2, b1, b0, 1.0, pair.upper1, pair.upper0));
            out.add(AnalogSection::make(b2, b1, b0, 1.0, pair.lower1, pair.lower0));
        }
        if (n % 2 == 1) out.add(AnalogSection::make(b2, b1, b0, 1.0, width, w02));
        return out;
    }

    // Elliptic functions through the Landen transformation (S. J. Orfanidis,
    // "Lecture Notes on Elliptic Filter Design", 2006).

    /// The moduli a Landen sequence goes through on its way to zero.
    struct Landen
    {
        double v[14] = {};
        int count = 0;
    };

    [[nodiscard]] static Landen landen(double k, double complement) noexcept
    {
        Landen steps;
        double kc = complement;
        while (steps.count < 14 && k > 1e-18)
        {
            k /= 1.0 + kc;
            k *= k;
            steps.v[steps.count++] = k;
            kc = std::sqrt((1.0 - k) * (1.0 + k));
        }
        return steps;
    }

    /** @brief The complete elliptic integral of the modulus a sequence starts from. */
    [[nodiscard]] static double quarterPeriod(const Landen& steps) noexcept
    {
        double period = 0.5 * kPi;
        for (int i = 0; i < steps.count; ++i) period *= 1.0 + steps.v[i];
        return period;
    }

    /** @brief cd(u K, k) and sn(u K, k), with u in units of the quarter period. */
    [[nodiscard]] static std::complex<double> ascend(std::complex<double> w, const Landen& steps) noexcept
    {
        for (int i = steps.count - 1; i >= 0; --i)
            w = (1.0 + steps.v[i]) * w / (1.0 + steps.v[i] * w * w);
        return w;
    }
    [[nodiscard]] static std::complex<double> ellipticCd(std::complex<double> u, const Landen& steps) noexcept
    {
        return ascend(std::cos(u * (0.5 * kPi)), steps);
    }
    [[nodiscard]] static std::complex<double> ellipticSn(std::complex<double> u, const Landen& steps) noexcept
    {
        return ascend(std::sin(u * (0.5 * kPi)), steps);
    }

    /** @brief The u with sn(u K, k) = w. */
    [[nodiscard]] static std::complex<double> ellipticArcSn(std::complex<double> w, double k, const Landen& steps) noexcept
    {
        double before = k;
        for (int i = 0; i < steps.count; ++i)
        {
            w = w / (1.0 + std::sqrt(1.0 - w * w * (before * before))) * (2.0 / (1.0 + steps.v[i]));
            before = steps.v[i];
        }
        return std::asin(w) * (2.0 / kPi);
    }

    /**
     * @brief The modulus of an elliptic filter (pass band edge over stop band
     *        edge) from its order, ripple and stop level: the degree equation
     *        solved through the nome. Also returns the two ripple factors.
     */
    [[nodiscard]] static double ellipticModulus(int order, double rippleDb, double stopDb,
                                                double& ripple, double& stop) noexcept
    {
        const int n = std::clamp(order, 1, kMaxOrder);
        const double ap = std::isfinite(rippleDb) ? std::clamp(rippleDb, 1e-3, 3.0) : 0.05;
        const double as = std::isfinite(stopDb) ? std::clamp(stopDb, 20.0, 140.0) : 100.0;
        ripple = std::sqrt(std::expm1(ap * 0.23025850929940458));       // sqrt(10^(ap / 10) - 1)
        stop = std::sqrt(std::expm1(as * 0.23025850929940458));
        const double k1 = ripple / stop, k1c = std::sqrt((1.0 - k1) * (1.0 + k1));
        const double ratio = quarterPeriod(landen(k1c, k1)) / quarterPeriod(landen(k1, k1c));
        const double nome = std::exp(-kPi * ratio / n);
        double top = 0.0, bottom = 1.0;
        for (int m = 0; m < 12; ++m)
        {
            top += std::pow(nome, m * (m + 1.0));
            if (m > 0) bottom += 2.0 * std::pow(nome, 1.0 * m * m);
        }
        const double theta = top / bottom;
        return std::min(4.0 * std::sqrt(nome) * theta * theta, 1.0 - 1e-12);
    }

    /** @brief Elliptic low-pass with the end of its pass band at 1 rad/s. */
    [[nodiscard]] static AnalogCascade ellipticPrototype(int order, double rippleDb, double stopDb) noexcept
    {
        AnalogCascade out;
        const int n = std::clamp(order, 1, kMaxOrder);
        double ripple = 0.0, stop = 0.0;
        const double k = ellipticModulus(n, rippleDb, stopDb, ripple, stop);
        const double k1 = ripple / stop;
        const Landen steps = landen(k, std::sqrt((1.0 - k) * (1.0 + k)));
        const Landen steps1 = landen(k1, std::sqrt((1.0 - k1) * (1.0 + k1)));
        const std::complex<double> j(0.0, 1.0);
        const double v0 = (-j * ellipticArcSn(j / ripple, k1, steps1)).real() / n;

        for (int i = 0; i < n / 2; ++i)
        {
            const double u = (2.0 * i + 1.0) / n;
            const double zero = 1.0 / (k * ellipticCd(u, steps).real());
            const std::complex<double> pole = j * ellipticCd(std::complex<double>(u, -v0), steps);
            const double size = std::norm(pole);
            out.add(AnalogSection::make(1.0 / (zero * zero), 0.0, 1.0, 1.0 / size, -2.0 * pole.real() / size, 1.0));
        }
        if (n % 2 == 1)
        {
            const double pole = (j * ellipticSn(j * v0, steps)).real();
            out.add(AnalogSection::make(0.0, 0.0, 1.0, 0.0, -1.0 / pole, 1.0));
        }
        else if (out.count > 0)
        {
            // An even order ripples down from 0 dB: DC sits at the bottom of the ripple.
            const double gain = 1.0 / std::sqrt(1.0 + ripple * ripple);
            AnalogSection& first = at(out, 0);
            first.b2 *= gain; first.b0 *= gain;
        }
        return out;
    }
};

} // namespace dspark
