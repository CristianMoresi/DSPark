// DSPark - Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi - MIT License

#pragma once

/**
 * @file WavetableBank.h
 * @brief Band-limited, mipmapped multi-frame wavetables with a morphing reader.
 *
 * A bank holds one or more single-cycle frames. Each frame is stored as 20
 * mip levels half an octave apart, every level synthesised exactly from the
 * frame's harmonics by inverse FFT, so a level holds no harmonic above its
 * budget. The reader blends two adjacent frames (the morph position) and two
 * adjacent levels (the anti-aliasing crossfade) and interpolates the result.
 *
 * Interpolation: Niemitalo's optimal 6-point, 5th-order polynomial for 4x
 * oversampled data ("Polynomial Interpolators for High-Quality Resampling of
 * Oversampled Audio", 2001). Every level therefore keeps at least four
 * table samples per period of its top harmonic, twice Nyquist: a level of H
 * harmonics is max(2048, 8 H) samples rounded up to a power of two. The
 * interpolator is not flat at the edge of that band (up to 0.4 dB of droop),
 * so each harmonic is pre-emphasised by the inverse of the interpolator's
 * own response at its table frequency, computed when the bank is built: the
 * played amplitudes are the frame's amplitudes. The interpolator is linear in
 * the samples, so frames, and levels of equal size, are blended before one
 * polynomial evaluation.
 *
 * Measured at 48 kHz, worst inharmonic component below 20 kHz relative to the
 * strongest harmonic, 27.5 Hz to 5 kHz: about -165 dB for 1/h and 1/sqrt(h)
 * spectra and -152 dB for a flat spectrum (every harmonic at full scale), the
 * float storage being the limit. The 4-point cubic Hermite reader of 2048
 * sample levels measured -72 dB (1/h) and -74 dB (flat) at 27.5 Hz.
 *
 * Anti-aliasing contract (as WavetableOscillator's): a level is used at a
 * frequency f only while its highest harmonic stays below
 * max(fs/2, fs - 20 kHz), so anything that folds lands above 20 kHz, and the
 * crossfade pairs the brightest such level with the next duller one, so a
 * frequency sweep changes the timbre continuously.
 *
 * Storage is float whatever the reader's sample type, with guard samples
 * around every table so a read never wraps. A bank costs 229 KB per frame.
 * The four built-in banks (builtin()) are built once per process, on first
 * use, and shared.
 *
 * Threading: building (buildFromHarmonics, loadFrames) allocates and belongs
 * to a setup thread, never concurrent with reads of the same bank. A built
 * bank is immutable: any number of threads may read it concurrently.
 * builtin() is safe to call from several threads (function-local statics),
 * but its first call per bank builds it; call it during setup.
 *
 * Dependencies: DspMath.h, FFT.h.
 */

#include "DspMath.h"
#include "FFT.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dspark {

/**
 * @class WavetableBank
 * @brief Mipmapped multi-frame wavetable storage and its morphing reader.
 *
 * @code
 *   const auto& bank = dspark::WavetableBank::builtin(dspark::WavetableBank::Builtin::Basic);
 *   auto mip = bank.selectMip(frequencyHz, sampleRate);    // on pitch changes
 *   double phase = 0.0;
 *   for (int i = 0; i < n; ++i)
 *   {
 *       out[i] = bank.read<float>(phase, position, mip);   // position 0..frames-1
 *       phase += frequencyHz / sampleRate;
 *       if (phase >= 1.0) phase -= 1.0;
 *   }
 * @endcode
 */
class WavetableBank
{
public:
    /// Mip levels, half an octave apart.
    static constexpr int kMipLevels = 20;
    /// Harmonics of the richest level.
    static constexpr int kMaxHarmonics = 1023;
    /// Most frames a bank holds.
    static constexpr int kMaxFrames = 256;
    /// Largest single-cycle frame loadFrames() analyses.
    static constexpr int kMaxFrameSize = 1 << 20;

    /** @brief Built-in banks (see builtin()). */
    enum class Builtin
    {
        Basic,     ///< 6 frames: sine, triangle, saw, square, 25% and 12.5% pulse.
        Harmonic,  ///< 8 frames: 1/h^p spectra from p = 3 (near sine) to p = 0.5 (bright).
        Formant,   ///< 9 frames: vowels A, E, I, O, U and the four transitions between them.
        Fm         ///< 8 frames: sin(x + b sin x) with the index b rising from 0 to 6.
    };
    static constexpr int kNumBuiltins = 4;

    /** @brief How a bank's frames are scaled. */
    enum class Normalization
    {
        PerFrame,  ///< Every frame peaks at 1 (equal peak level across a morph).
        Bank       ///< One factor for the whole bank: the frames keep their relative levels.
    };

    /** @brief One harmonic's cosine and sine amplitudes. */
    struct Harmonic
    {
        double cosine = 0.0;
        double sine = 0.0;
    };

    /** @brief The level pair and crossfade weight for one pitch (selectMip()). */
    struct Mip
    {
        int level = 0;        ///< Brighter level.
        int next = 0;         ///< Duller level it fades into.
        float weight = 1.0f;  ///< Weight of `level`; 1 reads it alone.
    };

    WavetableBank() = default;

    // -- Building (setup thread; allocates) ------------------------------------

    /**
     * @brief Builds the bank from per-frame harmonic amplitudes.
     *
     * @param numFrames    Frame count, 1..kMaxFrames.
     * @param numHarmonics Highest harmonic the frames hold, 1..kMaxHarmonics.
     * @param coefficients Callable `Harmonic(int frame, int harmonic)` for
     *                     harmonic 1..numHarmonics (the DC is not stored).
     * @param normalization Frame scaling.
     * @return False (bank unchanged) on an invalid count or a non-finite
     *         coefficient, or when every frame is silent.
     */
    template <typename Coefficients>
    bool buildFromHarmonics(int numFrames, int numHarmonics, Coefficients&& coefficients,
                            Normalization normalization = Normalization::PerFrame)
    {
        if (numFrames < 1 || numFrames > kMaxFrames || numHarmonics < 1)
            return false;
        numHarmonics = std::min(numHarmonics, kMaxHarmonics);

        std::vector<Harmonic> coeffs(static_cast<size_t>(numFrames) * static_cast<size_t>(numHarmonics));
        for (int f = 0; f < numFrames; ++f)
            for (int h = 1; h <= numHarmonics; ++h)
            {
                const Harmonic c = coefficients(f, h);
                if (!std::isfinite(c.cosine) || !std::isfinite(c.sine)) return false;
                coeffs[static_cast<size_t>(f) * static_cast<size_t>(numHarmonics)
                       + static_cast<size_t>(h - 1)] = c;
            }
        return build(numFrames, numHarmonics, coeffs, normalization);
    }

    /**
     * @brief Builds the bank from consecutive single-cycle frames.
     *
     * Each frame is analysed exactly (an FFT for power-of-two sizes, a direct
     * transform otherwise); its DC is discarded, and so is the Nyquist bin of
     * an even size. A frame of N samples yields (N - 1) / 2 harmonics, at
     * most kMaxHarmonics. Non-finite samples count as 0.
     *
     * @param data          numFrames * frameSize samples, frame after frame.
     * @param frameSize     Samples per cycle, 3..kMaxFrameSize.
     * @param numFrames     Frame count, 1..kMaxFrames.
     * @param normalization Frame scaling (Bank keeps the frames' relative levels).
     * @return False (bank unchanged) on invalid arguments or silent input.
     */
    template <FloatType S>
    bool loadFrames(const S* data, int frameSize, int numFrames,
                    Normalization normalization = Normalization::Bank)
    {
        if (data == nullptr || frameSize < 3 || frameSize > kMaxFrameSize
            || numFrames < 1 || numFrames > kMaxFrames)
            return false;
        const int numHarmonics = std::min((frameSize - 1) / 2, kMaxHarmonics);
        std::vector<Harmonic> coeffs(static_cast<size_t>(numFrames) * static_cast<size_t>(numHarmonics));
        std::vector<double> cycle(static_cast<size_t>(frameSize));
        const bool pow2 = (frameSize & (frameSize - 1)) == 0 && frameSize >= 4;
        std::vector<double> spec(pow2 ? static_cast<size_t>(frameSize + 2) : 0);
        FFTReal<double> fft(pow2 ? static_cast<size_t>(frameSize) : 4);

        for (int f = 0; f < numFrames; ++f)
        {
            const S* src = data + static_cast<size_t>(f) * static_cast<size_t>(frameSize);
            for (int i = 0; i < frameSize; ++i)
                cycle[static_cast<size_t>(i)] = std::isfinite(src[i]) ? static_cast<double>(src[i]) : 0.0;
            Harmonic* dst = coeffs.data() + static_cast<size_t>(f) * static_cast<size_t>(numHarmonics);
            const double scale = 2.0 / static_cast<double>(frameSize);
            if (pow2)
            {
                // X[h] = (N/2) (a_h - j b_h) for x = sum a_h cos + b_h sin.
                fft.forward(cycle.data(), spec.data());
                for (int h = 1; h <= numHarmonics; ++h)
                    dst[h - 1] = { spec[static_cast<size_t>(2 * h)] * scale,
                                   -spec[static_cast<size_t>(2 * h + 1)] * scale };
            }
            else
            {
                for (int h = 1; h <= numHarmonics; ++h)
                {
                    double sumCos = 0.0, sumSin = 0.0;
                    for (int i = 0; i < frameSize; ++i)
                    {
                        // Integer phase reduction keeps the argument exact.
                        const auto k = static_cast<long long>(h) * i % frameSize;
                        const double ph = twoPi<double> * static_cast<double>(k)
                                        / static_cast<double>(frameSize);
                        sumCos += cycle[static_cast<size_t>(i)] * std::cos(ph);
                        sumSin += cycle[static_cast<size_t>(i)] * std::sin(ph);
                    }
                    dst[h - 1] = { sumCos * scale, sumSin * scale };
                }
            }
        }
        return build(numFrames, numHarmonics, coeffs, normalization);
    }

    /**
     * @brief A built-in bank, built on the first call and shared by the process.
     * @note The first call per bank allocates about 1.4 to 2.1 MB and runs the
     *       synthesis: make it from a setup thread.
     */
    [[nodiscard]] static const WavetableBank& builtin(Builtin which)
    {
        switch (which)
        {
            case Builtin::Basic:    { static const WavetableBank b = makeBasic();    return b; }
            case Builtin::Harmonic: { static const WavetableBank b = makeHarmonic(); return b; }
            case Builtin::Formant:  { static const WavetableBank b = makeFormant();  return b; }
            case Builtin::Fm:       { static const WavetableBank b = makeFm();       return b; }
        }
        static const WavetableBank empty;
        return empty;
    }

    // -- Queries -----------------------------------------------------------------

    /** @brief Frames in the bank (0 before a successful build). */
    [[nodiscard]] int getNumFrames() const noexcept { return numFrames_; }

    /** @brief Highest harmonic the frames hold (0 before a build). */
    [[nodiscard]] int getNumHarmonics() const noexcept { return levelHarmonics_[0]; }

    /** @brief Bytes of table storage. */
    [[nodiscard]] size_t getMemoryBytes() const noexcept { return data_.size() * sizeof(float); }

    // -- Playback (real-time safe, any thread) ----------------------------------

    /**
     * @brief Chooses the level pair and crossfade weight for a pitch.
     *
     * Call it when the frequency changes (a synthesizer voice does at its
     * control rate), not per sample. A negative frequency selects as its
     * magnitude; a non-finite one selects the dullest level.
     */
    [[nodiscard]] Mip selectMip(double frequencyHz, double sampleRate) const noexcept
    {
        Mip m;
        if (numFrames_ == 0 || !(sampleRate > 0.0)) return m;
        const double f = std::isfinite(frequencyHz) ? std::abs(frequencyHz) : 1e300;
        // Highest harmonic frequency whose alias still lands above 20 kHz
        // (never below Nyquist itself: at low rates there is no free band).
        const double fold = std::max(sampleRate * 0.5, sampleRate - 20000.0);
        const int last = kMipLevels - 1;
        int level = 0;
        while (level < last && f > fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level)]))
            ++level;
        m.level = level;
        m.next = std::min(level + 1, last);
        if (m.next == level) return m;

        // Bottom of this level's range: the previous level's limit, or, for
        // level 0, one level-spacing below its own limit.
        const double hi = fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level)]);
        const double lo = (level > 0)
            ? fold / static_cast<double>(levelHarmonics_[static_cast<size_t>(level - 1)])
            : hi * static_cast<double>(levelHarmonics_[1]) / static_cast<double>(levelHarmonics_[0]);
        const double span = hi - lo;
        m.weight = (span > 0.0) ? static_cast<float>(std::clamp((hi - f) / span, 0.0, 1.0)) : 1.0f;
        return m;
    }

    /**
     * @brief One sample of the bank.
     * @param phase    Cycle phase in [0, 1) (wrapped if outside).
     * @param position Frame position in [0, frames - 1], blended linearly
     *                 between adjacent frames (clamped).
     * @param mip      Selection from selectMip() for the current pitch.
     * @return 0 for an empty bank.
     */
    template <FloatType T>
    [[nodiscard]] T read(double phase, double position, const Mip& mip) const noexcept
    {
        if (numFrames_ == 0) return T(0);
        if (!(phase >= 0.0 && phase < 1.0))
        {
            phase -= std::floor(phase);
            if (!(phase >= 0.0 && phase < 1.0)) phase = 0.0;
        }
        const double maxPos = static_cast<double>(numFrames_ - 1);
        if (!(position > 0.0)) position = 0.0;          // NaN included
        else if (position > maxPos) position = maxPos;
        const int f0 = static_cast<int>(position);
        const int f1 = (f0 < numFrames_ - 1) ? f0 + 1 : f0;
        const T t = static_cast<T>(position - static_cast<double>(f0));

        const int a = std::clamp(mip.level, 0, kMipLevels - 1);
        const int b = std::clamp(mip.next, 0, kMipLevels - 1);
        const T w = static_cast<T>(std::clamp(mip.weight, 0.0f, 1.0f));

        T ya[6];
        const T xa = gather<T>(a, f0, f1, t, phase, ya);
        if (a == b || w >= T(1))
            return interpolate(ya, xa);

        T yb[6];
        const T xb = gather<T>(b, f0, f1, t, phase, yb);
        if (levelSize_[static_cast<size_t>(a)] == levelSize_[static_cast<size_t>(b)])
        {
            // Same grid: blend the samples, interpolate once.
            for (int k = 0; k < 6; ++k) ya[k] = yb[k] + w * (ya[k] - yb[k]);
            return interpolate(ya, xa);
        }
        const T sa = interpolate(ya, xa);
        const T sb = interpolate(yb, xb);
        return sb + w * (sa - sb);
    }

    /**
     * @brief Amplitude response of the reader's interpolator at a frequency
     *        in cycles per table sample (the value pre-emphasis divides by).
     */
    [[nodiscard]] static double interpolatorResponse(double cyclesPerSample) noexcept
    {
        // The kernel is even, so H(v) = sum_k int_0^1 f_k(x) cos(2 pi v (x - k)) dx,
        // f_k being the weight of sample k (offsets -2..3) at fraction x.
        // 16-point Gauss-Legendre on each unit piece: the integrand is a
        // degree-5 polynomial times a slow cosine.
        static constexpr std::array<double, 8> node = {
            0.0950125098376374, 0.2816035507792589, 0.4580167776572274, 0.6178762444026438,
            0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499 };
        static constexpr std::array<double, 8> weight = {
            0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
            0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541 };
        double sum = 0.0;
        for (int k = -2; k <= 3; ++k)
        {
            double y[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            y[k + 2] = 1.0;
            for (size_t q = 0; q < node.size(); ++q)
                for (const double s : { -1.0, 1.0 })
                {
                    const double x = 0.5 + 0.5 * s * node[q];
                    sum += 0.5 * weight[q] * interpolate(y, x)
                         * std::cos(twoPi<double> * cyclesPerSample * (x - static_cast<double>(k)));
                }
        }
        return sum;
    }

private:
    /// Fewest samples a level table holds.
    static constexpr int kMinTableSize = 2048;
    /// Guard samples on each side of every table (the reader spans -2..+3).
    static constexpr int kGuard = 3;

    /// Harmonic budget per level, half an octave apart from the top down to
    /// a single sine (the same budgets as WavetableOscillator).
    static constexpr std::array<int, kMipLevels> kLevelHarmonics = {
        kMaxHarmonics, 724, 512, 362, 256, 181, 128, 90, 64, 45,
        32, 22, 16, 11, 8, 5, 4, 3, 2, 1 };

    /**
     * @brief Niemitalo's optimal 4x 6-point 5th-order interpolator (z-form).
     * @param y Samples at offsets -2..3 around the interval.
     * @param x Fraction in [0, 1) between y[2] and y[3].
     */
    template <FloatType T>
    [[nodiscard]] static T interpolate(const T* y, T x) noexcept
    {
        const T z = x - T(0.5);
        const T even1 = y[3] + y[2], odd1 = y[3] - y[2];
        const T even2 = y[4] + y[1], odd2 = y[4] - y[1];
        const T even3 = y[5] + y[0], odd3 = y[5] - y[0];
        const T c0 = even1 * T(0.41496902959240894) + even2 * T(0.08343081932889224)
                   + even3 * T(0.00160015038681571);
        const T c1 = odd1 * T(0.31625515004859783) + odd2 * T(0.21197848565176958)
                   + odd3 * T(0.00956166668408054);
        const T c2 = even1 * T(-0.203271896548875371) + even2 * T(0.17989908432249280)
                   + even3 * T(0.02337283412161328);
        const T c3 = odd1 * T(-0.20209241069835732) + odd2 * T(0.01760734419526000)
                   + odd3 * T(0.02985927012435252);
        const T c4 = even1 * T(0.04100948858761910) + even2 * T(-0.06147760875085254)
                   + even3 * T(0.02046802954581191);
        const T c5 = odd1 * T(0.06607747864416924) + odd2 * T(-0.03255079211953620)
                   + odd3 * T(0.00628989632244913);
        return ((((c5 * z + c4) * z + c3) * z + c2) * z + c1) * z + c0;
    }

    /** Six frame-blended samples around the phase on one level; returns the fraction. */
    template <FloatType T>
    T gather(int level, int f0, int f1, T t, double phase, T* y) const noexcept
    {
        const auto L = static_cast<size_t>(level);
        const int size = levelSize_[L];
        const double p = phase * static_cast<double>(size);
        const int i = std::min(static_cast<int>(p), size - 1);
        const size_t stride = static_cast<size_t>(size + 2 * kGuard);
        const float* ta = data_.data() + levelOffset_[L] + static_cast<size_t>(f0) * stride
                        + static_cast<size_t>(kGuard + i - 2);
        if (t == T(0))
        {
            for (int k = 0; k < 6; ++k) y[k] = static_cast<T>(ta[k]);
        }
        else
        {
            const float* tb = ta + static_cast<size_t>(f1 - f0) * stride;
            for (int k = 0; k < 6; ++k)
                y[k] = static_cast<T>(ta[k]) + t * (static_cast<T>(tb[k]) - static_cast<T>(ta[k]));
        }
        return static_cast<T>(p - static_cast<double>(i));
    }

    /** Synthesises, pre-emphasises and normalises every level of every frame. */
    bool build(int numFrames, int numHarmonics, const std::vector<Harmonic>& coeffs,
               Normalization normalization)
    {
        std::array<int, kMipLevels> harmonics {};
        std::array<int, kMipLevels> sizes {};
        std::array<size_t, kMipLevels> offsets {};
        size_t total = 0;
        for (int L = 0; L < kMipLevels; ++L)
        {
            const auto l = static_cast<size_t>(L);
            harmonics[l] = std::max(1, std::min(kLevelHarmonics[l], numHarmonics));
            int size = kMinTableSize;
            while (size < 8 * harmonics[l]) size *= 2;
            sizes[l] = size;
            offsets[l] = total;
            total += static_cast<size_t>(numFrames) * static_cast<size_t>(size + 2 * kGuard);
        }

        // Pre-emphasis per (table size, harmonic): the inverse interpolator
        // response at h / size cycles per sample.
        std::vector<std::vector<double>> emphasis(static_cast<size_t>(kMipLevels));
        for (int L = 0; L < kMipLevels; ++L)
        {
            const auto l = static_cast<size_t>(L);
            int same = -1;
            for (int j = 0; j < L; ++j)
                if (sizes[static_cast<size_t>(j)] == sizes[l]
                    && harmonics[static_cast<size_t>(j)] >= harmonics[l]) { same = j; break; }
            if (same >= 0) { emphasis[l] = emphasis[static_cast<size_t>(same)]; continue; }
            emphasis[l].resize(static_cast<size_t>(harmonics[l] + 1), 1.0);
            for (int h = 1; h <= harmonics[l]; ++h)
                emphasis[l][static_cast<size_t>(h)] =
                    1.0 / interpolatorResponse(static_cast<double>(h) / static_cast<double>(sizes[l]));
        }

        std::vector<double> spec, cycle;
        std::vector<FFTReal<double>> ffts;
        std::vector<int> fftSizes;
        auto fftFor = [&](int size) -> FFTReal<double>& {
            for (size_t j = 0; j < fftSizes.size(); ++j)
                if (fftSizes[j] == size) return ffts[j];
            fftSizes.push_back(size);
            ffts.emplace_back(static_cast<size_t>(size));
            return ffts.back();
        };
        for (int L = 0; L < kMipLevels; ++L) (void)fftFor(sizes[static_cast<size_t>(L)]);

        // One level of one frame into `cycle`, scaled by `gain`.
        auto synthesize = [&](int frame, int L) {
            const auto l = static_cast<size_t>(L);
            const int size = sizes[l];
            spec.assign(static_cast<size_t>(size + 2), 0.0);
            cycle.resize(static_cast<size_t>(size));
            const double half = 0.5 * static_cast<double>(size);
            const Harmonic* c = coeffs.data() + static_cast<size_t>(frame) * static_cast<size_t>(numHarmonics);
            for (int h = 1; h <= harmonics[l] && h <= numHarmonics; ++h)
            {
                // x = sum a cos + b sin has X[h] = (N/2)(a - j b) under the
                // inverse transform's 1/N scaling.
                const double e = emphasis[l][static_cast<size_t>(h)];
                spec[static_cast<size_t>(2 * h)]     =  half * c[h - 1].cosine * e;
                spec[static_cast<size_t>(2 * h + 1)] = -half * c[h - 1].sine * e;
            }
            fftFor(size).inverse(spec.data(), cycle.data());
        };

        // Pass 1: peak per frame over all its levels (in practice the
        // richest). Pass 2: write the scaled tables with their guards.
        std::vector<double> peak(static_cast<size_t>(numFrames), 0.0);
        for (int f = 0; f < numFrames; ++f)
            for (int L = 0; L < kMipLevels; ++L)
            {
                synthesize(f, L);
                for (const double v : cycle)
                    peak[static_cast<size_t>(f)] = std::max(peak[static_cast<size_t>(f)], std::abs(v));
            }
        double bankPeak = 0.0;
        for (const double p : peak) bankPeak = std::max(bankPeak, p);
        if (!(bankPeak > 0.0) || !std::isfinite(bankPeak)) return false;

        std::vector<float> data(total, 0.0f);
        for (int f = 0; f < numFrames; ++f)
        {
            const double framePeak = peak[static_cast<size_t>(f)];
            const double gain = (normalization == Normalization::PerFrame)
                ? (framePeak > 0.0 ? 1.0 / framePeak : 0.0)
                : 1.0 / bankPeak;
            for (int L = 0; L < kMipLevels; ++L)
            {
                synthesize(f, L);
                const auto l = static_cast<size_t>(L);
                const int size = sizes[l];
                float* dst = data.data() + offsets[l]
                           + static_cast<size_t>(f) * static_cast<size_t>(size + 2 * kGuard);
                for (int i = 0; i < size; ++i)
                    dst[kGuard + i] = static_cast<float>(cycle[static_cast<size_t>(i)] * gain);
                for (int g = 0; g < kGuard; ++g)
                {
                    dst[g] = dst[size + g];                     // wraps of the cycle's end
                    dst[kGuard + size + g] = dst[kGuard + g];   // and of its start
                }
            }
        }

        data_ = std::move(data);
        levelHarmonics_ = harmonics;
        levelSize_ = sizes;
        levelOffset_ = offsets;
        numFrames_ = numFrames;
        return true;
    }

    // -- Built-in recipes ----------------------------------------------------------

    /** Fourier coefficients of a +-1 pulse that is high for the first `w` of the cycle. */
    [[nodiscard]] static Harmonic pulse(double w, int h) noexcept
    {
        const double theta = twoPi<double> * static_cast<double>(h) * w;
        const double k = 2.0 / (pi<double> * static_cast<double>(h));
        return { k * std::sin(theta), k * (1.0 - std::cos(theta)) };
    }

    [[nodiscard]] static WavetableBank makeBasic()
    {
        WavetableBank b;
        b.buildFromHarmonics(6, kMaxHarmonics, [](int frame, int h) -> Harmonic {
            const double hd = static_cast<double>(h);
            switch (frame)
            {
                case 0: return { 0.0, h == 1 ? 1.0 : 0.0 };
                case 1:
                {
                    if (h % 2 == 0) return {};
                    const double sign = ((h / 2) % 2 == 0) ? 1.0 : -1.0;
                    return { 0.0, sign / (hd * hd) };
                }
                case 2: return { 0.0, (h % 2 == 0) ? -1.0 / hd : 1.0 / hd };
                case 3: return pulse(0.5, h);
                case 4: return pulse(0.25, h);
                default: return pulse(0.125, h);
            }
        });
        return b;
    }

    [[nodiscard]] static WavetableBank makeHarmonic()
    {
        WavetableBank b;
        b.buildFromHarmonics(8, kMaxHarmonics, [](int frame, int h) -> Harmonic {
            const double p = 3.0 - 2.5 * static_cast<double>(frame) / 7.0;
            return { 0.0, std::pow(static_cast<double>(h), -p) };
        });
        return b;
    }

    [[nodiscard]] static WavetableBank makeFormant()
    {
        // Three formants (Hz) of the vowels A E I O U, placed on the harmonic
        // series of a 110 Hz cycle; odd frames sit halfway between vowels.
        static constexpr double vowels[5][3] = {
            { 730.0, 1090.0, 2440.0 }, { 530.0, 1840.0, 2480.0 }, { 270.0, 2290.0, 3010.0 },
            { 570.0, 840.0, 2410.0 },  { 300.0, 870.0, 2240.0 } };
        static constexpr double gains[3] = { 1.0, 0.5, 0.25 };
        static constexpr double widths[3] = { 90.0, 120.0, 160.0 };
        WavetableBank b;
        b.buildFromHarmonics(9, kMaxHarmonics, [](int frame, int h) -> Harmonic {
            const int v0 = frame / 2;
            const int v1 = std::min(v0 + (frame % 2), 4);
            const double f = 110.0 * static_cast<double>(h);
            double env = 0.01;
            for (int i = 0; i < 3; ++i)
            {
                const double fc = 0.5 * (vowels[v0][i] + vowels[v1][i]);
                const double d = (f - fc) / widths[i];
                env += gains[i] / (1.0 + d * d);
            }
            return { 0.0, env / static_cast<double>(h) };
        });
        return b;
    }

    [[nodiscard]] static WavetableBank makeFm()
    {
        // sin(x + b sin x): a smooth cycle whose harmonics above about b + 12
        // are below -150 dB for b <= 6, so a dense numeric cycle analysed by
        // FFT gives its coefficients exactly.
        constexpr int n = 4096;
        std::vector<double> cycles(static_cast<size_t>(8 * n));
        for (int frame = 0; frame < 8; ++frame)
        {
            const double b = 6.0 * std::pow(static_cast<double>(frame) / 7.0, 1.5);
            for (int i = 0; i < n; ++i)
            {
                const double x = twoPi<double> * static_cast<double>(i) / static_cast<double>(n);
                cycles[static_cast<size_t>(frame * n + i)] = std::sin(x + b * std::sin(x));
            }
        }
        WavetableBank bank;
        bank.loadFrames(cycles.data(), n, 8, Normalization::PerFrame);
        return bank;
    }

    int numFrames_ = 0;
    std::array<int, kMipLevels> levelHarmonics_ {};   ///< Harmonic budget per level.
    std::array<int, kMipLevels> levelSize_ {};        ///< Table samples per level.
    std::array<size_t, kMipLevels> levelOffset_ {};   ///< Start of each level's frame block.
    std::vector<float> data_;                         ///< [level][frame][guard, table, guard].
};

} // namespace dspark
