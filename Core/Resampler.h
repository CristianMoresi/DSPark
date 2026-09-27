// DSPark -- Professional Audio DSP Framework
// Copyright (c) 2026 Cristian Moresi -- MIT License

#pragma once

/**
 * @file Resampler.h
 * @brief High-quality sample rate converter for audio signals.
 *
 * Converts audio between sample rates (44100 <-> 48000 <-> 96000 Hz and any
 * other pair) with a Kaiser-windowed sinc designed for each conversion:
 *
 * - **Designed, not fixed, kernels.** Each quality tier is a specification -
 *   a passband edge as a fraction of the lower Nyquist frequency and a
 *   stopband attenuation - and the kernel length and Kaiser beta are derived
 *   from it (Kaiser's design formulas). The stopband starts exactly at the
 *   lower Nyquist frequency, so nothing above it survives at more than the
 *   stated attenuation: no image when upsampling, no alias when downsampling.
 * - **Exact polyphase phases.** When the rate ratio is rational with a
 *   manageable number of phases (every common audio pair: 44.1 <-> 48 kHz
 *   has 160 or 147) each output phase has its own kernel, computed once, and
 *   positions advance in integer arithmetic, so there is no phase
 *   interpolation error and no drift. Other ratios read a 512-phase table
 *   with cubic interpolation across phases, whose error lies below the
 *   Ultra stopband.
 * - **Identity at equal rates.** A ratio of exactly 1 is a pure delay.
 *
 * Two modes:
 *
 * - **Offline (batch)**: process() converts an entire buffer at once and is
 *   time-aligned (zero latency -- the symmetric kernel reads future samples).
 *   Best for file processing.
 * - **Streaming**: processBlock() converts chunks incrementally and is
 *   causal: the output is delayed by getLatency() samples. Best for
 *   real-time applications where audio arrives in blocks.
 *
 * Threading: prepare() allocates and must run on a setup thread. After
 * prepare(), processBlock() and reset() are allocation-free and belong to
 * the single stream owner; there are no cross-thread parameter setters.
 *
 * Dependencies: AudioBuffer.h, AudioSpec.h, SimdOps.h.
 */

#include "AudioBuffer.h"
#include "AudioSpec.h"
#include "SimdOps.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>
#include <utility>
#include <vector>

namespace dspark {

/**
 * @class Resampler
 * @brief Windowed-sinc sample rate converter optimized for real-time DSP.
 *
 * Quality tiers are filter specifications; the kernel length follows from
 * them and from the ratio (a downsampler's kernel is longer by the ratio,
 * since its band edge is lower relative to the input rate):
 *
 * | Quality | Passband edge | Stopband | Taps, 44.1 <-> 48 kHz |
 * |---------|---------------|----------|-----------------------|
 * | Draft   | 0.80 Nyquist  | 60 dB    | 38 / 40               |
 * | Normal  | 0.90 Nyquist  | 100 dB   | 130 / 140             |
 * | High    | 0.91 Nyquist  | 140 dB   | 206 / 224             |
 * | Ultra   | 0.915 Nyquist | 210 dB   | 332 / 362             |
 *
 * "Nyquist" is the lower of the two rates' Nyquist frequencies; Ultra's
 * passband reaches 20.2 kHz at 44.1 kHz. The float instantiation is limited
 * by float arithmetic to roughly -140 dB; the double one reaches the table.
 * Kernels are capped at kMaxTaps taps, which binds only for downsampling
 * beyond about 25:1 at Ultra: past it the transition band widens.
 *
 * @tparam T Sample type (float or double).
 */
template <typename T>
class Resampler
{
public:
    enum class Quality
    {
        Draft,   ///< Passband 0.80 of Nyquist, 60 dB stopband: previews.
        Normal,  ///< Passband 0.90 of Nyquist, 100 dB stopband.
        High,    ///< Passband 0.91 of Nyquist (20 kHz at 44.1), 140 dB stopband.
        Ultra    ///< Passband 0.915 of Nyquist, 210 dB stopband: mastering.
    };

    /** @brief Longest kernel, in input samples (see the class note). */
    static constexpr int kMaxTaps = 8192;

    /**
     * @brief Prepares the resampler for a given rate conversion.
     *
     * Allocates internal buffers. MUST be called outside the audio thread.
     *
     * @param sourceRate      Source sample rate in Hz.
     * @param targetRate      Target sample rate in Hz.
     * @param quality         Filter specification (default: Normal).
     */
    void prepare(double sourceRate, double targetRate,
                 Quality quality = Quality::Normal)
    {
        prepareTransactional(sourceRate, targetRate, quality,
                             channelStates_.size());
    }

    /**
     * @brief Prepares the resampler using AudioSpec (unified API).
     *
     * Pre-allocates multi-channel states. MUST be called outside the audio thread.
     *
     * @param spec       Audio environment (sampleRate, numChannels used).
     * @param targetRate Target sample rate in Hz.
     * @param quality    Filter specification (default: Normal).
     */
    void prepare(const AudioSpec& spec, double targetRate,
                 Quality quality = Quality::Normal)
    {
        const size_t requestedChannels = spec.numChannels > 0
            ? static_cast<size_t>(spec.numChannels)
            : size_t(0);
        prepareTransactional(spec.sampleRate, targetRate, quality,
                             std::max(channelStates_.size(), requestedChannels));
    }

    /**
     * @brief Resets the internal state (delay lines, all channels) to zero.
     *
     * Safe to call from the audio thread after prepare(): it only fills
     * storage that prepare() already sized and never allocates.
     */
    void reset() noexcept
    {
        resetChannelState(mono_);
        for (auto& cs : channelStates_)
            resetChannelState(cs);
    }

    /**
     * @brief Resamples an entire buffer (offline batch processing).
     *
     * Stateless and time-aligned: output sample k interpolates the input at
     * position k / getRatio() exactly (no latency; the buffer edges are
     * zero-padded). Allocates the output vector -- offline use only.
     *
     * @param input       Source audio samples.
     * @param inputLength Number of input samples.
     * @return Vector of resampled output samples (ceil(inputLength * ratio)).
     */
    [[nodiscard]] std::vector<T> process(const T* input, int inputLength)
    {
        if (table_.empty()) return {}; // not prepared

        // Clamp through double before the int cast: huge ratios could
        // otherwise overflow the cast itself (undefined behaviour).
        const double outLenD =
            std::ceil(static_cast<double>(inputLength) * ratio_);
        const int outputLength = static_cast<int>(
            std::min(outLenD, static_cast<double>(std::numeric_limits<int>::max())));

        std::vector<T> output(static_cast<size_t>(outputLength));
        for (int outIdx = 0; outIdx < outputLength; ++outIdx)
        {
            int64_t intPos;
            int64_t phase = 0;
            double frac = 0.0;
            if (exact_)
            {
                // Integer position arithmetic: output k reads input k*M/L.
                const int64_t num = static_cast<int64_t>(outIdx) * stepM_;
                intPos = num / phasesL_;
                phase = num % phasesL_;
            }
            else
            {
                const double srcPos = static_cast<double>(outIdx) / ratio_;
                intPos = static_cast<int64_t>(srcPos);
                frac = srcPos - static_cast<double>(intPos);
            }
            if (intPos >= inputLength) break;
            output[static_cast<size_t>(outIdx)] =
                interpolateOffline(input, inputLength, static_cast<int>(intPos), phase, frac);
        }
        return output;
    }

    /**
     * @brief Resamples a block of audio in single-channel streaming mode.
     *
     * @param input       Input audio samples.
     * @param inputLength Number of input samples.
     * @param output      Output buffer (must hold at least getMaxOutputSamples()).
     * @return Number of output samples produced.
     */
    int processBlock(const T* input, int inputLength, T* output) noexcept
    {
        return processChannel(input, inputLength, output, mono_);
    }

    /**
     * @brief Resamples multi-channel audio using AudioBufferView (streaming).
     *
     * Processes each channel sequentially with its independent state.
     *
     * @param input  Input audio buffer view.
     * @param output Output audio buffer view (pre-allocated).
     * @return Number of output samples produced per channel.
     */
    int processBlock(AudioBufferView<T> input, AudioBufferView<T> output) noexcept
    {
        int numCh = std::min(input.getNumChannels(), output.getNumChannels());
        int inLen = input.getNumSamples();

        assert(static_cast<int>(channelStates_.size()) >= numCh &&
               "Resampler channels not allocated! Call prepare(spec...) first.");
        // Release-safe: never index channelStates_ past what prepare() allocated
        // (the assert above flags the missing prepare(spec) in debug builds).
        numCh = std::min(numCh, static_cast<int>(channelStates_.size()));

        int outCount = 0;
        for (int ch = 0; ch < numCh; ++ch)
        {
            outCount = processChannel(input.getChannel(ch), inLen,
                                      output.getChannel(ch),
                                      channelStates_[static_cast<size_t>(ch)]);
        }

        return outCount;
    }

    /**
     * @brief Returns the maximum number of output samples for a given input length.
     * @param inputLength Number of input samples.
     * @return Maximum possible output samples.
     */
    [[nodiscard]] int getMaxOutputSamples(int inputLength) const noexcept
    {
        const double maxOut =
            std::ceil(static_cast<double>(inputLength) * ratio_) + 2.0;
        return static_cast<int>(
            std::min(maxOut, static_cast<double>(std::numeric_limits<int>::max())));
    }

    /** @brief Returns the conversion ratio (targetRate / sourceRate). */
    [[nodiscard]] double getRatio() const noexcept { return ratio_; }

    /** @brief Kernel length in input samples (1 at a ratio of exactly 1). */
    [[nodiscard]] int getFilterLength() const noexcept { return taps_; }

    /**
     * @brief Returns the streaming latency in output samples.
     *
     * The streaming path (processBlock) delays the signal by exactly half the
     * kernel length in input samples -- the kernel's group delay -- which
     * this getter reports rounded to the nearest output sample. The offline
     * process() path is already time-aligned and has zero latency.
     */
    [[nodiscard]] int getLatency() const noexcept
    {
        return static_cast<int>(std::round(static_cast<double>(halfTaps_) * ratio_));
    }

private:
    /// Phases of the table read with cubic interpolation (non-rational ratios).
    static constexpr int kTablePhases = 512;
    /// Most exact phases, and most exact-table coefficients, before a ratio
    /// falls back to the interpolated table.
    static constexpr int64_t kMaxExactPhases = 4096;
    static constexpr int64_t kMaxExactCoefficients = int64_t(1) << 21;

    struct ChannelState
    {
        std::vector<T> history;
        int writePos = 0;
        double fractionalPos = 0.0;   ///< Next output's position (table mode).
        int64_t phase = 0;            ///< Next output's phase, 0..L-1 (exact mode).
    };

    struct Spec { double passband; double attenuationDb; };

    static_assert(std::is_nothrow_copy_assignable_v<T>,
                  "Resampler sample storage must be reset without throwing");
    static_assert(std::is_nothrow_swappable_v<ChannelState>,
                  "Resampler state commit must be no-throw");
    static_assert(std::is_nothrow_swappable_v<std::vector<T>>,
                  "Resampler table commit must be no-throw");
    static_assert(std::is_nothrow_swappable_v<std::vector<ChannelState>>,
                  "Resampler channel commit must be no-throw");

    static Spec qualitySpec(Quality q) noexcept
    {
        switch (q)
        {
            case Quality::Draft:  return { 0.80, 60.0 };
            case Quality::Normal: return { 0.90, 100.0 };
            case Quality::High:   return { 0.91, 140.0 };
            case Quality::Ultra:  return { 0.915, 210.0 };
        }
        return { 0.90, 100.0 };
    }

    /** @brief Kaiser's beta for a stopband attenuation (Kaiser 1974). */
    [[nodiscard]] static double kaiserBeta(double a) noexcept
    {
        if (a > 50.0) return 0.1102 * (a - 8.7);
        if (a >= 21.0) return 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
        return 0.0;
    }

    /** @brief Modified Bessel I0 for the continuous Kaiser window. */
    [[nodiscard]] static double besselI0(double x) noexcept
    {
        double sum = 1.0, term = 1.0;
        for (int k = 1; k <= 200; ++k)
        {
            const double half = x / (2.0 * k);
            term *= half * half;
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        return sum;
    }

    /** @brief Kernel parameters of one conversion. */
    struct Design
    {
        int taps = 2;          ///< Even kernel length, in input samples.
        double cutoff = 1.0;   ///< Sinc cutoff, 1 = input Nyquist.
        double beta = 0.0;     ///< Kaiser beta.
        bool identity = false; ///< Ratio exactly 1: a pure delay.
    };

    [[nodiscard]] static Design design(double ratio, Quality quality) noexcept
    {
        Design d;
        if (ratio == 1.0)
        {
            d.identity = true;
            return d;
        }
        const Spec spec = qualitySpec(quality);
        const double scale = std::min(1.0, ratio);   // lower Nyquist / input Nyquist
        // Band edges in cycles per input sample: the passband ends at
        // passband * lower Nyquist, the stopband begins at the lower Nyquist.
        const double transition = 0.5 * scale * (1.0 - spec.passband);
        d.cutoff = scale * 0.5 * (1.0 + spec.passband);
        d.beta = kaiserBeta(spec.attenuationDb);
        // Kaiser's length estimate, N = (A - 7.95) / (14.36 * df).
        const double n = (spec.attenuationDb - 7.95) / (14.36 * transition);
        const double capped = std::min(n, static_cast<double>(kMaxTaps));
        d.taps = 2 * static_cast<int>(std::ceil(0.5 * capped));
        d.taps = std::max(d.taps, 4);
        return d;
    }

    /**
     * @brief The kernel for one fractional position `frac` of the output
     * between two input samples, normalised to unity DC gain.
     *
     * Tap j weighs source sample intPos-halfTaps+1+j, so its distance from
     * the interpolation point intPos + frac is t = (j - halfTaps + 1) - frac:
     * the symmetric placement for an even-length kernel. The MINUS sign on
     * frac is essential: `+ frac` would sample the kernel time-reversed.
     */
    static void buildPhase(T* dst, const Design& d, double frac)
    {
        const int half = d.taps / 2;
        if (d.identity)
        {
            for (int j = 0; j < d.taps; ++j)
                dst[j] = (j == half - 1) ? T(1) : T(0);
            return;
        }
        constexpr double kPi = std::numbers::pi;
        const double i0Beta = besselI0(d.beta);
        double sum = 0.0;
        // Accumulate in double; the table stores T.
        for (int j = 0; j < d.taps; ++j)
        {
            const double t = static_cast<double>(j - half + 1) - frac;
            const double x = t * d.cutoff;
            const double sincVal = (std::abs(x) < 1e-12)
                ? d.cutoff
                : d.cutoff * std::sin(kPi * x) / (kPi * x);
            const double wx = t / static_cast<double>(half);
            const double win = (std::abs(wx) >= 1.0)
                ? 0.0
                : besselI0(d.beta * std::sqrt(1.0 - wx * wx)) / i0Beta;
            dst[j] = static_cast<T>(sincVal * win);
            sum += sincVal * win;
        }
        if (std::abs(sum) > 1e-12)
        {
            const double inv = 1.0 / sum;
            for (int j = 0; j < d.taps; ++j)
                dst[j] = static_cast<T>(static_cast<double>(dst[j]) * inv);
        }
    }

    /**
     * @brief The rate ratio as L / M in lowest terms, when both are small
     * enough for an exact phase table; {0, 0} otherwise.
     */
    [[nodiscard]] static std::pair<int64_t, int64_t> rationalRatio(
        double sourceRate, double targetRate) noexcept
    {
        // Integer rates (every standard one) reduce exactly.
        const double rs = std::round(sourceRate), rt = std::round(targetRate);
        if (std::abs(rs - sourceRate) < 1e-9 && std::abs(rt - targetRate) < 1e-9
            && rs < 1e12 && rt < 1e12)
        {
            int64_t a = static_cast<int64_t>(rt), b = static_cast<int64_t>(rs);
            int64_t x = a, y = b;
            while (y != 0) { const int64_t r = x % y; x = y; y = r; }
            if (x > 0) return { a / x, b / x };
        }
        // Otherwise a continued-fraction approximation that is exact to
        // double precision.
        const double r = targetRate / sourceRate;
        int64_t h0 = 0, h1 = 1, k0 = 1, k1 = 0;
        double v = r;
        for (int it = 0; it < 64; ++it)
        {
            const double fl = std::floor(v);
            if (fl > 1e12) break;
            const auto an = static_cast<int64_t>(fl);
            const int64_t h2 = an * h1 + h0, k2 = an * k1 + k0;
            if (k2 > (int64_t(1) << 32) || h2 > (int64_t(1) << 32)) break;
            h0 = h1; h1 = h2; k0 = k1; k1 = k2;
            if (std::abs(static_cast<double>(h1) / static_cast<double>(k1) - r) <= 1e-15 * r)
                return { h1, k1 };
            const double rem = v - fl;
            if (rem < 1e-15) break;
            v = 1.0 / rem;
        }
        return { 0, 0 };
    }

    static void initialiseChannelState(ChannelState& state, int taps)
    {
        state.history.assign(static_cast<size_t>(taps * 2), T(0));
        state.writePos = 0;
        state.fractionalPos = 0.0;
        state.phase = 0;
    }

    void prepareTransactional(double sourceRate, double targetRate,
                              Quality quality, size_t channelCount)
    {
        // Derive the complete candidate configuration without touching the
        // live object. The std::max behavior intentionally preserves the
        // established treatment of non-positive rates.
        const double stagedSourceRate = std::max(sourceRate, 1.0);
        const double stagedTargetRate = std::max(targetRate, 1.0);
        const double stagedRatio = stagedTargetRate / stagedSourceRate;
        const Design d = design(stagedRatio, quality);

        auto [l, m] = rationalRatio(stagedSourceRate, stagedTargetRate);
        const bool exact = d.identity
            || (l > 0 && l <= kMaxExactPhases
                && l * static_cast<int64_t>(d.taps) <= kMaxExactCoefficients);
        if (d.identity) { l = 1; m = 1; }

        // Every potentially throwing allocation belongs to local state. A
        // failure therefore destroys only the candidate and leaves the live
        // conversion, histories and streaming positions untouched.
        std::vector<T> stagedTable;
        if (exact)
        {
            // Phase p is the kernel for fractional position p / L.
            stagedTable.resize(static_cast<size_t>(l * d.taps));
            for (int64_t p = 0; p < l; ++p)
                buildPhase(stagedTable.data() + p * d.taps, d,
                           static_cast<double>(p) / static_cast<double>(l));
        }
        else
        {
            // kTablePhases + 3 phases: one guard phase before position 0 and
            // two after the last, so cubic interpolation across phases never
            // reads outside the table.
            stagedTable.resize(static_cast<size_t>((kTablePhases + 3) * d.taps));
            for (int p = 0; p < kTablePhases + 3; ++p)
                buildPhase(stagedTable.data() + static_cast<size_t>(p) * static_cast<size_t>(d.taps), d,
                           static_cast<double>(p - 1) / static_cast<double>(kTablePhases));
        }
        ChannelState stagedMono;
        initialiseChannelState(stagedMono, d.taps);
        std::vector<ChannelState> stagedChannels(channelCount);
        for (auto& state : stagedChannels)
            initialiseChannelState(state, d.taps);

        // Scalar assignment and the mechanically checked swaps below cannot
        // throw. Once commit starts, observers can only see the complete new
        // state (prepare itself remains a setup-thread operation).
        sourceRate_ = stagedSourceRate;
        targetRate_ = stagedTargetRate;
        ratio_ = stagedRatio;
        srcStep_ = 1.0 / stagedRatio;
        taps_ = d.taps;
        halfTaps_ = d.taps / 2;
        exact_ = exact;
        phasesL_ = exact ? l : 0;
        stepM_ = exact ? m : 0;
        table_.swap(stagedTable);
        using std::swap;
        swap(mono_, stagedMono);
        channelStates_.swap(stagedChannels);
    }

    /** @brief One output from `taps_` contiguous samples at `src`. */
    T interpolateContiguous(const T* src, int64_t phase, double frac) const noexcept
    {
        if (exact_)
            return simd::dotProduct(table_.data() + phase * taps_, src, taps_);

        // Cubic (Lagrange) interpolation across the four phases around the
        // position: error ~ (pi / (2 * kTablePhases))^4, below every stopband.
        const double exactPhase = frac * static_cast<double>(kTablePhases);
        int p0 = static_cast<int>(exactPhase);
        if (p0 > kTablePhases - 1) p0 = kTablePhases - 1;   // frac is < 1 by contract
        const double u = exactPhase - static_cast<double>(p0);
        const T* k = table_.data() + static_cast<size_t>(p0) * static_cast<size_t>(taps_);
        // Table phase index p0 + 1 holds position p0 / P (one guard phase first).
        const T sm = simd::dotProduct(k, src, taps_);
        const T s0 = simd::dotProduct(k + taps_, src, taps_);
        const T s1 = simd::dotProduct(k + 2 * taps_, src, taps_);
        const T s2 = simd::dotProduct(k + 3 * taps_, src, taps_);
        const double wm = -u * (u - 1.0) * (u - 2.0) / 6.0;
        const double w0 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0;
        const double w1 = -(u + 1.0) * u * (u - 2.0) / 2.0;
        const double w2 = (u + 1.0) * u * (u - 1.0) / 6.0;
        return static_cast<T>(wm * sm + w0 * s0 + w1 * s1 + w2 * s2);
    }

    /** @brief Offline interpolation with boundary checks (zero padding). */
    T interpolateOffline(const T* data, int length, int intPos,
                         int64_t phase, double frac)
    {
        // Tap j weighs data[intPos - halfTaps + 1 + j] (see buildPhase).
        const int firstSrc = intPos - halfTaps_ + 1;
        if (firstSrc >= 0 && firstSrc + taps_ <= length)
            return interpolateContiguous(data + firstSrc, phase, frac);

        // Edge path: gather into a zero-padded window. The kernel is read
        // exactly as in the fast path, so both agree bit for bit.
        std::vector<T>& w = edgeScratch_;
        w.assign(static_cast<size_t>(taps_), T(0));
        for (int j = 0; j < taps_; ++j)
        {
            const int srcIdx = firstSrc + j;
            if (srcIdx >= 0 && srcIdx < length)
                w[static_cast<size_t>(j)] = data[srcIdx];
        }
        return interpolateContiguous(w.data(), phase, frac);
    }

    static void resetChannelState(ChannelState& cs) noexcept
    {
        std::fill(cs.history.begin(), cs.history.end(), T(0));
        cs.writePos = 0;
        cs.fractionalPos = 0.0;
        cs.phase = 0;
    }

    int processChannel(const T* input, int inputLength, T* output,
                       ChannelState& state) noexcept
    {
        if (state.history.empty()) return 0; // not prepared

        // Hot state lives in locals for the duration of the block: this both
        // tells the compiler the fields cannot alias the output writes (no
        // per-sample reloads) and keeps them in registers. The window
        // [writePos, writePos + n) holds the last n inputs, oldest first; the
        // kernel peaks at tap halfTaps-1+frac, so the output stream is
        // delayed by exactly halfTaps input samples (see getLatency()).
        T* const hist = state.history.data();
        const int n = taps_;
        int writePos = state.writePos;
        int outIdx = 0;

        if (exact_)
        {
            const int64_t l = phasesL_, m = stepM_;
            int64_t phase = state.phase;
            for (int i = 0; i < inputLength; ++i)
            {
                // Double-buffered push: the mirror write keeps the read
                // window contiguous (no modulo).
                const T x = input[i];
                hist[writePos] = x;
                hist[writePos + n] = x;
                if (++writePos >= n)
                    writePos = 0;

                while (phase < l)
                {
                    output[outIdx++] = interpolateContiguous(hist + writePos, phase, 0.0);
                    phase += m;
                }
                phase -= l;
            }
            state.phase = phase;
        }
        else
        {
            const double step = srcStep_;
            double frac = state.fractionalPos;
            for (int i = 0; i < inputLength; ++i)
            {
                const T x = input[i];
                hist[writePos] = x;
                hist[writePos + n] = x;
                if (++writePos >= n)
                    writePos = 0;

                while (frac < 1.0)
                {
                    output[outIdx++] = interpolateContiguous(hist + writePos, 0, frac);
                    frac += step;
                }
                frac -= 1.0;
            }
            state.fractionalPos = frac;
        }

        state.writePos = writePos;
        return outIdx;
    }

    double sourceRate_ = 44100.0;
    double targetRate_ = 48000.0;
    double ratio_ = 1.0;
    double srcStep_ = 1.0;

    int taps_ = 2;
    int halfTaps_ = 1;
    bool exact_ = true;
    int64_t phasesL_ = 0;   ///< Exact mode: phases per input sample (L).
    int64_t stepM_ = 0;     ///< Exact mode: phase advance per output (M).

    std::vector<T> table_;
    std::vector<T> edgeScratch_;   ///< Offline edge window (process() allocates anyway).
    ChannelState mono_;
    std::vector<ChannelState> channelStates_;
};

} // namespace dspark
